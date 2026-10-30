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


## Local RTX 5090 D v2 (24 GB) setup

This checkout has a project-local Conda environment at `.venv`, cloned offline from
`amp_mjlab`. The source environment is unchanged. The local compatibility versions
are recorded in `requirements.local.txt` and `runs/setup/environment.json`; they
differ from the original `requirements.txt`. Use the launchers below without
activating another environment. They select this checkout's Python package and caches.

The existing 40-clip G1 CSV dataset is linked at `data/raw/lafan1_g1`. Rebuild the
50 Hz library with `bash scripts/prepare_lafan1_local.sh` if needed.

```bash
bash scripts/train_lafan1_24gb.sh smoke  # 200 iterations; checks the full update loop
bash scripts/train_lafan1_24gb.sh train  # 2,000 iterations; a bounded initial run
bash scripts/evaluate_lafan1_24gb.sh runs/lafan1_24gb_train
```

Both training modes use 1,024 simulator environments, 1,000 replay steps per
environment, 512 world-model windows, 512 imagined rollouts, k=8 and H=32. Automatic
evaluation is disabled; run it separately after training. Initial Torch/Warp
compilation can take several minutes and is cached. These small runs validate the
training pipeline, not the paper's final tracking score.

Resume the initial run with a larger iteration limit:

```bash
bash scripts/train_lafan1_24gb.sh train \
  --resume runs/lafan1_24gb_train/last.pt --iterations 20000
```

Choose a fresh output directory with `--out runs/<new-name>` for an independent run.
Reusing an existing output directory without `--resume` starts new weights and
appends to its log.


### Native live playback

On the Linux desktop, open a MuJoCo window with a live policy/physics loop:

```bash
bash scripts/play_lafan1_24gb.sh --clip walk1_subject1
# Use a particular snapshot and overlay the reference at its true world position:
bash scripts/play_lafan1_24gb.sh --ckpt runs/lafan1_24gb_train/last.pt --clip dance2_subject1 --reference-offset 0
```

The solid robot executes the checkpoint in one MuJoCo-Warp environment. The orange
reference is a display-only copy, offset by 1 m along Y by default; use
`--reference-offset 0` for an overlay. Noise and randomization are disabled.
The window shows the loaded iteration, clip/time, status, root error and accumulated
return divided by the full clip's maximum (partial until playback ends).

Keys: Space pauses/resumes; R restarts; N/P switches clips; L reloads the latest saved
checkpoint; +/- changes target playback speed; F toggles camera following; G toggles
the reference; Q/Esc closes the window. Mouse controls rotate/zoom the camera.
End/failure poses are held for 2 seconds, then the same clip restarts; `--no-loop`
holds the final pose, while `--auto-next` cycles the selectable clips. Use
`--clips walk1_subject1,dance2_subject1` to restrict the selectable set.

For diagnostics, `--duration 5 --report runs/setup/live_gui.json` closes the window
after 5 wall-seconds; `--headless --max-steps 10` checks the loop without a window.


## From-scratch AMP walking

This is an independent flat-ground velocity-command task, inspired by the local
`supermimic_AMP` implementation. Training uses a learned world model and BPTT,
with an adversarial motion prior as a differentiable style reward. A fresh run
randomly initializes the actor, critic, world model and discriminator. There is
no teacher, pretrained policy, tracking checkpoint or old AMP normalizer to load.

The command actor receives four time-ordered frames of 96 values (gyro, gravity,
velocity command, joint position/velocity, previous clean action), totaling 384.
It receives no future reference poses or reference phase. The discriminator uses
195 body features per frame (390 per transition), defined relative to the torso
and each body's local velocities. Expert features are recomputed with this
checkout's G1 FK. The least-squares discriminator and zero-centered gradient
penalty train on real policy/expert transitions; actor gradients pass through its
frozen parameters. Running statistics are updated from raw features, and the
gradient penalty uses normalized expert-pair coordinates.

### Expert data

The prepared library is `data/processed/amp_walk_50hz.npz`: 11 walking, side-step,
and idle-turn clips, 5,370 frames at 50 Hz. It is self-contained at training time.
To rebuild from the local reference project's WalkandRun NPZ files:

```bash
bash scripts/prepare_amp_walk.sh
# For another source directory: AMP_EXPERT_DIR=/path/to/WalkandRun bash scripts/prepare_amp_walk.sh
```

Existing outputs are protected. Use `--overwrite` only when intentionally
rebuilding the library; its JSON manifest lists the source clips and layout.
Half the training resets use expert physical states, half use nominal standing.
This reference-state initialization is not a weight warm-start. Set
`--rsi-probability 0` to use standing resets only. Recovery clips are not included.

### Training and continuation

```bash
bash scripts/train_amp_walk_24gb.sh smoke
bash scripts/train_amp_walk_24gb.sh train
# Continue a run started with the current preset, using a larger total limit:
bash scripts/train_amp_walk_24gb.sh train --resume runs/amp_walk_24gb_train/last.pt --iterations 100000
# Or continue the verified best model into a separate output directory:
bash scripts/train_amp_walk_24gb.sh train --resume runs/amp_walk_best/best.pt --out runs/amp_walk_best_continue --iterations 100000
```

The 24 GB training preset matches the final learning/task settings in
`runs/amp_walk_best/training_config.json`: 4,096 environments, a 1,000-step replay,
512 WM windows per batch, 10 WM batches per update, 256 imagined rollouts,
H=32, k=8, FP32, and 512 discriminator pairs per batch. The iteration limit is
50,000. The first 500 iterations train only the world model and discriminator;
the actor and critic are then enabled.

The preset uses actor LR scale 0.5, linear velocity reward width 0.25,
action-rate weight -0.1, forward commands 0.2–0.8 m/s, and 10% standing commands.
Discriminator positives come from `walk_forward_` clips: 50% original pairs and
50% time-scaled moving pairs at 0.4/0.6/0.8 speed. The command style gate is 0.2;
foot-slip reward remains disabled. Expert selection affects discriminator
positives; reference-state resets still use the prepared motion library.

A fresh run uses these final settings from random weights. It does not replay
the best model's earlier curriculum stages or automatically load its checkpoint.
Older runs using the previous defaults need matching reward/command options
when resumed; use `--allow-reward-change true` only for an intentional curriculum
change. Choose a new `--out` for a fresh run if the default directory already
contains a checkpoint. Command-line options appended to the script override its
preset values.

The smoke preset uses the same task/style settings with H=8, 64 environments,
20 world-model-only iterations and an 80-iteration limit for a short integration
check.

Compared with the teacher-anchored reference, the scratch task uses explicit
upright reward, linear/yaw reward widths of 0.25/0.5, bounded actions, a bounded
critic bootstrap/target, and 25% noise-free collection alongside noisy actions.
These options are saved in `config.json`; no exact parity with the teacher-anchored reference is
claimed. The verified local walking checkpoint is documented below. Task terms are scaled by dt; style is already a per-control-step reward.
The mixed reward is 0.75 task + 0.25 style, with style scale 0.1.

Outputs live in `runs/amp_walk_24gb_train/`: `log.jsonl`, `tb/`, numbered
checkpoints, and `last.pt`. Checkpoints include all four networks, normalization,
optimizers, stage counters, RNG, initial random-weight hashes and expert-library
hash. Resume restores learning state and refills replay from new episodes; it
does not restore the exact physics/replay state. Other checkpoint formats and
changed reward/model semantics are rejected. A fresh run refuses an existing
`last.pt`; use `--resume` or a new `--out`. Ctrl+C saves `last.pt` gracefully.

```bash
.venv/bin/python -m tensorboard.main --logdir runs/amp_walk_24gb_train/tb
bash scripts/test_amp_walk.sh
```

### Command evaluation and native playback

```bash
bash scripts/evaluate_amp_walk.sh runs/amp_walk_24gb_train/last.pt --seconds 20 --num-envs 32
bash scripts/play_amp_walk_24gb.sh --ckpt runs/amp_walk_24gb_train/last.pt --command 0.5,0,0
```

Evaluation starts from nominal standing with no reference-state initialization,
noise or domain randomization. Read survival/falls/coverage together with planar
and yaw errors: low error before an early fall is not successful walking. The
32 copies share the same initial state and are not independent seeds. This is
a locomotion task, so it does not report LAFAN1 tracking return or clip success.

Playback has no reference ghost or motion phase. Keys W/S change forward speed,
A/D change lateral speed, Z/C change yaw rate, X sets zero command, Space pauses,
R resets to standing, L reloads the latest saved AMP actor, F toggles follow,
+/- changes playback speed, and Q/Esc quits. Mouse controls rotate/zoom.

### F710 velocity commands

`bash scripts/play_amp_walk_24gb.sh` automatically detects a Logitech F710 on
Linux. It selects the controller by identity and kernel axis mapping, so an
unrelated `/dev/input/js0` is not mistaken for the gamepad. Playback prefers
`runs/amp_walk_best/best.pt` when available; `--ckpt` selects a specific model.

| Control | Command |
| --- | --- |
| Left stick up/down | Forward/backward velocity, default range −0.5 to +0.8 m/s |
| Left stick left/right | Left/right velocity, default range ±0.5 m/s |
| Right stick left/right | Counterclockwise/clockwise yaw rate, default range ±1 rad/s |
| A | Zero command and reset to standing |
| B | Zero command |
| Start | Pause/resume simulation |
| Keyboard G | Switch between keyboard and gamepad control |

Center the sticks when connecting/reconnecting, and after A/B, before commanding
motion. The default deadzone is 0.15 and the remaining stick travel is rescaled
continuously. Removing the active controller zeros the command. Keyboard
W/S/A/D/Z/C/X takes over immediately; press G to return to the controller.
Continuous stick changes respect the existing hold period after a fall.

```bash
# Inspect device identity and mapping without starting the simulator:
bash scripts/play_amp_walk_24gb.sh --list-gamepads
# Show controller commands for 10 seconds, without a GPU simulation or window:
bash scripts/play_amp_walk_24gb.sh --test-gamepad --duration 10
# Optional device, deadzone and speed limits:
bash scripts/play_amp_walk_24gb.sh --gamepad-device /dev/input/js1 \
  --gamepad-deadzone 0.12 --gamepad-forward-speed 0.6 --gamepad-yaw-speed 0.6
# Preserve a fixed command / keyboard-only input:
bash scripts/play_amp_walk_24gb.sh --gamepad off --command 0.5,0,0
```

`--gamepad on` starts with a zero command while waiting for a controller;
`--gamepad auto` is the default and retains keyboard playback when no F710 is
available. The input reader uses the read-only, nonblocking
[Linux joydev event interface](https://www.kernel.org/doc/html/latest/input/joydev/joystick-api.html)
and adds no Python package dependency.


### Verified local AMP walking checkpoint

The supervised local run selected the H=32 iteration-42000 checkpoint at
`runs/amp_walk_best/best.pt`. Across forward commands 0.3, 0.5 and 0.8 m/s,
192/192 perturbed trials survived 30 seconds; all 64 low-speed trials retained
alternating steps, and 64/64 completed a continuous 40-second command sequence.
Details, videos, configuration and validation reports are in
`runs/amp_walk_best/REPORT.md`; the experiment journal is `notes/AMP_TRAINING.md`.

```bash
bash scripts/play_amp_walk_24gb.sh --ckpt runs/amp_walk_best/best.pt --command 0.3,0,0
```


### Optional high-speed contact refinement

The local validated candidate is saved separately at
`runs/amp_contact_refine_20261030/selected/refined.pt`; the original
`runs/amp_walk_best/best.pt` is preserved. The report is
`runs/amp_contact_refine_20261030/report.html`, including all attempted settings
and a second reset-seed validation. Its 0.8 m/s loaded-contact penetration P95
improves modestly (about 2%), and yaw-rate MAE improves about 51%, while low-speed
alternating steps remain intact. This is not a universal physical improvement:
per-foot-time peak penetration and high-speed velocity RMSE have trade-offs.

Refinement adds optional nonpositive task terms for yaw-rate error, downward
foot-point speed near the ground, and penetration beyond a 0.5 mm allowance.
Foot costs use the actual sphere/capsule collision geometry and rigid-body point
velocities; they are differentiable geometric/kinematic proxies, not contact
force predictions. A command-speed gate ramps from zero at 0.5 m/s to one at
0.8 m/s. Real simulator evaluation still measures actual constraint contacts.
All new task weights default to zero for existing checkpoints and fresh runs.

`--low-speed-anchor-weight` optionally adds a training-only action consistency
loss on real replay start observations. It requires resuming this project's AMP
checkpoint and freezes that actor and its observation normalizer as a reference.
The reference is serialized, hash-checked and retained across future resumes;
inference still uses a single ordinary CommandActor. The existing
`--allow-reward-change true` guard also covers intentional changes to this
training regularizer.

```bash
# View the separately saved candidate:
bash scripts/play_amp_walk_24gb.sh --ckpt runs/amp_contact_refine_20261030/selected/refined.pt --command 0.8,0,0

# Continue with the candidate's reward and regularizer settings in a new output:
bash scripts/train_amp_walk_24gb.sh train \
  --resume runs/amp_contact_refine_20261030/selected/refined.pt \
  --out runs/amp_contact_refine_continue --iterations 50000 \
  --actor-lr-scale 0.25 --yaw-rate-weight -4 \
  --foot-landing-weight -0.4 --foot-penetration-weight -0.1 \
  --low-speed-anchor-weight 1.0
```

The selected checkpoint is iteration 43250, following 1000 iterations
with low-speed retention and 250 further iterations with stronger
contact costs. Its full configuration, provenance and two evaluation reports
are saved beside `refined.pt`. Further training is not assumed to improve it;
compare saved checkpoints using the same contact and gait measurements.
