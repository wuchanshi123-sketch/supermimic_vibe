# SuperMimic, re-implemented

An unofficial reproduction of *SuperMimic: Scaling Humanoid Whole-Body Control with One
Differentiable World Model* (arXiv 2610.28480, CoRL 2026). The authors released no code; this
repository re-implements the method from the paper alone and re-runs its experiments on the
Unitree G1 in mjlab / MuJoCo Warp.

SuperMimic fits one MLP world model to replayed simulator transitions and trains a
motion-tracking policy by backpropagating through 32 imagined steps of that frozen model,
with a small critic bootstrapping the tail of each imagined rollout.

## Results (one RTX A6000; the paper used one H100)

Benchmark (paper App. F): every clip played once from its first frame, friction 1.0, no
randomization or noise, deterministic policy; normalized return = reward sum / maximum possible;
a clip fails on termination. Hours are training wall-clock with evaluation excluded, as in the
paper. An A6000 runs a SuperMimic iteration about 2.7x slower than an H100.

| Claim (paper ref.) | Paper | Ours | Status |
| --- | --- | --- | --- |
| SuperMimic reaches 0.82 on LAFAN1 within 1.2 h on one GPU (Fig. 3a) | 0.82 in 1.2 h; 0.866 at 3 h | 0.833 at 1.34 h; best 0.947, 0/40 failed at 11.4 h | reproduced |
| PPO in the simulator on one GPU stays far below (Fig. 3a) | 0.60 after 18 h | 0.543 after 18 h | reproduced |
| Without the critic, H <= 16 fails (Fig. 3b) | H=16: 0.25 | H=16: best 0.035, 40/40 failed | reproduced |
| With the critic, H=16 matches critic-free H=32 (Fig. 3b) | 0.843 vs 0.84 | 0.905 vs 0.899 | reproduced |
| H=32 is fastest to 0.82; long horizons cost more (Fig. 3c) | 0.88 h / 1.70 h / 4.62 h (H=32/64/128) | 1.34 h / 3.31 h / not reached in 6 h (H=32/64/128) | partly reproduced |
| World-model rollouts beat MuJoCo Warp (Fig. 4a) | 11-26x at 2k-16k envs | 15-22x at 2k-16k envs | reproduced (32k+ envs exceed host RAM here) |
| BPTT needs far fewer imagined steps than PPO on the same world model (Fig. 4b) | 64x at return 0.84 | 27x at PPO's best return in the world model (0.33) | partly reproduced |
| The critic policy recovers from 0.5-4 m/s pushes (Sec. 5.3) | settles in 0.5-2.4 s | 6/64 falls vs 19/64 without critic | partly reproduced |
| RL velocity encoder beats distillation in a step test (Sec. 5.5) | 650 of 672 commands | 652 of 672 | reproduced |
| BONES-SEED filtering (App. E) | 142,220 -> 129,514 clips | 142,220 -> 128,904 clips (261.3 h) | reproduced |
| One GPU beats SONIC on BONES-SEED (Fig. 2a, Table A.4) | 0.94, 25 failed at 3 H100-h vs SONIC 0.90, 179 | SONIC in our simulator 0.893, 642 failed. Paper-faithful: passes SONIC's return at 3.0 h, both counts at 4.3 h (0.906, 637), collapses after ~6 h. With 50% of environments collecting without exploration noise (not in the paper): best 0.923, 45 failed at 14.4 h, no collapse in 16 h | partly reproduced |
| Two GPUs reach 0 failed clips on BONES-SEED (Fig. 2a) | 0 failed at 9.8 h | best 0.909, 791 failed, then declined | not reproduced |
| Compute to target grows slowly with library size (Fig. 2b,c) | 0.54 -> 1.89 H100-h from 1 h to 264 h | 1.41 -> 2.06 A6000-h to 0.88 from 1 h to 261 h | partly reproduced |
| >1000x fewer GPU-hours than SONIC (Sec. 5.1) | 3 GPU-h vs ~21,000 | 3.04 A6000-h to pass SONIC's return vs ~21,000 L40-h | reproduced |
| PPO in the simulator on 4 / 8 GPUs (Fig. 3a) | 0.70 / 0.72 | not run | not attempted |
| Real-robot transfer (Sec. 5.4) | 4 motions | — | out of scope |

On BONES-SEED the policy completes 389 of the 402 clips whose reference turns upside down
(cartwheels, flips, handstands, rolls), but tracks them less closely (mean return 0.645 vs 0.923
for all other clips). Rendered rollouts (robot over a translucent reference) come from
`code/scripts/render_rollouts.py`; `results/videos/*/summary.json` lists the rendered clips and
their returns (the MP4s are not in git).

The running log with every number, date and caveat is [`notes/RESULTS.md`](notes/RESULTS.md);
every place where the implementation follows the paper, fills a gap, or deviates is listed in
[`notes/IMPLEMENTATION.md`](notes/IMPLEMENTATION.md). The audit tables behind these rows are
in `results/audit/*.json` (rebuilt by `code/scripts/export_audit.py`).

## Layout

| Path | What |
| --- | --- |
| `code/supermimic/` | the package: kinematics, motion library, task (reward/obs/termination), simulator env, networks, replay, SuperMimic trainer, PPO-Sim, PPO-WM, velocity interface, evaluation |
| `code/scripts/` | data preparation, push and step tests, rollout benchmark, run status, plots, audit export |
| `code/scripts/dev/` | sanity checks (FK vs MuJoCo) and profilers |
| `slurm/` | batch scripts for every experiment |
| `results/` | small result files, audit export (`results/audit/`), rendered-clip summaries (`results/videos/`) |
| `notes/` | implementation notes, results log, SONIC evaluation notes |

Data (`data/`), runs and checkpoints (`runs/`) and the paper source (`paper/`) are not in git.

## Setup

Python 3.11 and one CUDA GPU.

```bash
pip install -r requirements.txt   # or, inside a Slurm job: source setup_local_env.sh
source env.sh                     # from the repository root: PYTHONPATH and caches
```

`setup_local_env.sh` builds a node-local uv virtualenv from `requirements.txt` (once per node)
and sets `$SMIMIC_PY`; every Slurm script sources it, so the cluster needs no pre-built env.

### Motion data

LAFAN1 retargeted to the G1 (the 40 `g1/*.csv` clips of the Hugging Face dataset
`lvhaidong/LAFAN1_Retargeting_Dataset`) is not redistributed here. Put the CSVs in
`data/raw/lafan1_g1/` and build the 50 Hz motion library:

```bash
$SMIMIC_PY code/scripts/prepare_motions.py --format lafan --fps-in 30 \
  --input data/raw/lafan1_g1 --output data/processed/lafan1_50hz.npz
```

BONES-SEED (`bones-studio/seed`, 23.5 GB for the G1 retarget) needs a Hugging Face account
that accepted its license. Download and convert it, apply the App. E filter, and cut the nested
subsets for the library-size sweep:

```bash
HF_TOKEN=... sbatch slurm/download_bones.slurm   # -> data/processed/bones_seed_raw_50hz.npz
sbatch slurm/filter_bones.slurm data/processed/bones_seed_raw_50hz.npz data/processed/bones_seed_50hz.npz results/bones_filter.json
sbatch slurm/make_subsets.slurm                  # -> data/processed/bones_seed_sub_{1,7,40}h.npz
```

## Running the experiments

Submit from the repository root (logs go to `slurm/logs/`). The scripts request one A6000 on
the `pci` account/partition; change the `#SBATCH` lines for another cluster. Training scripts
resume from `<out>/last.pt` when it exists.

```bash
# Fig. 3a: SuperMimic and PPO-Sim on LAFAN1, one GPU
sbatch slurm/train_supermimic.slurm --out runs/smimic_lafan1_1gpu --wall-hours 12 --eval-every 4000 --ckpt-every 4000
sbatch slurm/train_ppo.slurm --out runs/ppo_lafan1_1gpu --wall-hours 18 --iterations 1000000 --eval-every-hours 0.5 --ckpt-every 500

# Fig. 3b,c: horizon x critic ablation (6 h each)
bash slurm/submit_horizon_ablation.sh
sbatch --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True slurm/train_supermimic.slurm \
  --out runs/abl_H128_critic_true --horizon 128 --use-critic true --wall-hours 6 --eval-every 8000 --ckpt-every 8000

# Data parallel on 2 GPUs
sbatch slurm/train_supermimic_ddp.slurm --out runs/smimic_lafan1_2gpu --wall-hours 6 --eval-every 4000 --ckpt-every 4000

# Fig. 4a: rollout speed
sbatch slurm/bench_rollout.slurm

# Fig. 4b: BPTT vs PPO on the world model, walk/run/sprint clips
sbatch slurm/train_supermimic.slurm --out runs/f4b_smimic_wrs --clip-prefixes walk,run,sprint --wall-hours 5 --eval-every 8000 --ckpt-every 8000
sbatch slurm/train_ppowm.slurm --out runs/f4b_ppowm_16kx32 --clip-prefixes walk,run,sprint --ppo-rollouts 16384 --ppo-steps 32 --wall-hours 5 --eval-every 8000 --ckpt-every 8000

# Sec. 5.5: velocity interface, RL vs distillation, then the step test
sbatch slurm/train_velocity.slurm --out runs/vel_rl --vel-mode rl --clip-prefixes walk,run,sprint --iterations 100000 --eval-every 8000 --ckpt-every 8000
sbatch slurm/train_velocity.slurm --out runs/vel_distill --vel-mode distill --clip-prefixes walk,run,sprint --iterations 100000 --eval-every 8000 --ckpt-every 8000
sbatch slurm/step_test.slurm results/step_test.json runs/vel_rl/last.pt runs/vel_distill/last.pt
```

BONES-SEED (Fig. 2) and the SONIC baseline:

```bash
# Paper-faithful run, scored on all clips by a second job while it trains
sbatch slurm/train_supermimic.slurm --motions data/processed/bones_seed_50hz.npz --out runs/smimic_bones_1gpu --wall-hours 16 --iterations 320000 --eval-every 0 --ckpt-every 4000
sbatch slurm/eval_watch.slurm runs/smimic_bones_1gpu data/processed/bones_seed_50hz.npz
# Best fixed configuration: half the environments collect without exploration noise (not in the paper);
# benchmarks a fixed 8,192-clip subset every 4000 iterations
sbatch slurm/train_supermimic.slurm --motions data/processed/bones_seed_50hz.npz --out runs/bones_fix_detfrac50 --det-frac 0.5 --wall-hours 16 --iterations 320000 --eval-every 4000 --ckpt-every 4000 --eval-clips 8192
# Score every 4th checkpoint on all 128,904 clips (~13 min per checkpoint)
sbatch --time=04:00:00 slurm/eval_runs.slurm runs/bones_fix_detfrac50:data/processed/bones_seed_50hz.npz:eval_full.jsonl:4
# SONIC's released checkpoint (ONNX weights converted by code/scripts/convert_sonic.py)
sbatch slurm/eval_sonic.slurm data/processed/bones_seed_50hz.npz results/sonic_bones_gains.json gains
# Videos of chosen clips
sbatch slurm/render_rollouts.slurm --ckpt runs/bones_fix_detfrac50/ckpt_0184000.pt --motions data/processed/bones_seed_50hz.npz --clips cartwheel_R_001__A135_M,flip_360_002__A304_M --out results/videos/bones
```

Evaluation and tests:

```bash
$SMIMIC_PY -m supermimic.evaluate --run runs/smimic_lafan1_1gpu            # benchmark protocol
$SMIMIC_PY code/scripts/push_test.py results/push_test.json runs/smimic_lafan1_1gpu/ckpt_0048000.pt runs/abl_H32_critic_false/ckpt_0072000.pt
$SMIMIC_PY code/scripts/status.py                                         # one line per run
```

These need a GPU; on a cluster, run them inside a job.

## Hardware note

On the A6000 MuJoCo Warp runs about as fast as the paper's H100 numbers at 4096 envs, while
the MLP-heavy SuperMimic update is about 2.7x slower per iteration (0.28 s vs 0.105 s), so
wall-clock comparisons here are tilted toward PPO-Sim. MLPs run in bf16 autocast (paper: TF32).

## Acknowledgements

Built on [mjlab](https://github.com/mujocolab/mjlab), MuJoCo Warp and the BeyondMimic
motion-tracking setup. All credit for the method goes to the SuperMimic authors; errors in this
re-implementation are ours.
