# SuperMimic reproduction — results log

Hardware: NVIDIA RTX A6000 (48 GB) on Princeton Ionic `pci`; the paper used H100 80 GB SXM5.
All hours below are training wall-clock on one A6000 (evaluation excluded, as in the paper;
the first ~6 min include one-time torch.compile).

## Status (2026-10-22 16:05Z)

| Experiment (paper ref.) | Run | Result | Status |
| --- | --- | --- | --- |
| LAFAN1, SuperMimic 1 GPU (Fig. 3a) | `runs/smimic_lafan1_1gpu` | 0.82 @ 1.34 h; 0.946 (0/40) @ 12 h | reproduced |
| LAFAN1, PPO-Sim 1 GPU (Fig. 3a) | `runs/ppo_lafan1_1gpu` | 0.543 (7/40) @ 18 h (paper 0.60) | reproduced |
| LAFAN1, SuperMimic 2 GPUs data parallel | `runs/smimic_lafan1_2gpu` | 0.938 (0/40) @ 6 h | extra |
| Horizon x critic ablation (Fig. 3b,c) | `runs/abl_H*_critic_*` | 8 runs, H = 8-128 | reproduced (trend), partly (timings) |
| Rollout speed (Fig. 4a) | `results/bench_rollout_a6000.json` | 15-22x at 2k-16k envs | reproduced (32k+ not measurable) |
| PPO-WM vs BPTT (Fig. 4b) | `runs/f4b_*` | 27x at PPO-WM's best return | partly reproduced |
| Push recovery (Sec. 5.3) | `results/push_test.json` | 6/64 vs 19/64 falls | partly reproduced |
| Velocity step test (Sec. 5.5, App. G) | `runs/vel_*`, `results/step_test.json` | RL better on 652/672 (paper 650) | reproduced |
| BONES-SEED vs SONIC (Fig. 2a, Table A.4) | — | — | **blocked: dataset is license-gated; needs the user's HF token** |
| Dataset-length sweep (Fig. 2b,c), filtering (App. E) | — | — | blocked on BONES-SEED |
| PPO-Sim 4/8 GPUs (Fig. 3a) | — | — | not run (needs 4-8 shared GPUs) |

## Early numbers

Benchmark normalized return on LAFAN1 (40 clips, each from frame 0, no DR/noise):

| | ours (A6000) | paper (H100) |
| --- | --- | --- |
| SuperMimic, first eval | 0.512 @ 0.42 h (it 4000), 13/40 failed | 0.524 @ 0.078 h (~2.7k it) |
| PPO-Sim, first eval | 0.240 @ 0.50 h, 25/40 failed | 0.210 @ 0.48 h |
| SuperMimic, it 8000 | 0.678 @ 0.73 h, 10/40 failed | 0.667 @ 0.23 h |
| SuperMimic, it 12000 | 0.756 @ 1.03 h, 7/40 failed | 0.810 @ 1.04 h |
| **SuperMimic, it 16000** | **0.833 @ 1.34 h, 1/40 failed (target 0.82 reached)** | 0.833 @ 1.19 h (0.82 within 1.2 h) |
| PPO-Sim @ 1.5 h | 0.333, 23/40 failed | ~0.30 @ 1.44 h |
| SuperMimic, it 40000 | **0.910 @ 3.19 h, 0/40 failed** | 0.866 @ 2.99 h (final) |
| PPO-Sim @ 3.5 h | 0.385, 20/40 failed | ~0.42 @ 3.4 h |

Rollout speed (control steps/s, one 50 Hz step = 4 MuJoCo Warp substeps):

| envs | MuJoCo Warp | world model TF32 | ratio | paper ratio (H100) |
| --- | --- | --- | --- | --- |
| 512 | 5.07e4 | 5.53e5 | 10.9× | 9.3× |
| 2048 | 1.09e5 | 1.68e6 | 15.5× | 11× |
| 4096 | 1.26e5 | 1.84e6 | 14.6× | 12.7× |
| 8192 | 1.16e5 | 2.04e6 | 17.6× | 17× |
| 16384 | 9.24e4 | 1.99e6 | 21.5× | 26× |

Note on hardware: MuJoCo Warp on the A6000 is about as fast as the paper's H100 numbers at
4096 envs, while the MLP-heavy SuperMimic update is ~2.7× slower per iteration (0.28 s vs
0.105 s). Wall-clock comparisons on A6000 are therefore tilted toward PPO-Sim.


## Horizon × critic ablation (Fig. 3b,c), LAFAN1 — in progress

Benchmark eval (normalized return, failed clips of 40) at matching iterations; training return
is logged too but, unlike the paper's, stays below eval return (our training episodes start
from BeyondMimic's randomized resets and the hard-bin-weighted sampler).

| H | critic | eval @ ~16k it | best eval so far | paper final training return |
| --- | --- | --- | --- | --- |
| 8 | yes | 0.483 (17 failed) @ 1.00 h | 0.483 | 0.392 (0.82 not reached) |
| 16 | yes | 0.769 (4 failed) @ 1.06 h | 0.769 | 0.843 |
| 16 | no | 0.028 (40 failed) @ 0.99 h | 0.030 | 0.25 (fails) |
| 32 | yes (main run) | 0.833 (1 failed) @ 1.34 h | 0.878 @ 1.96 h | 0.866 |
| 32 | no | 0.576 (21 failed) @ 1.11 h | 0.629 @ 1.67 h | 0.84 |
| 64 | yes | 0.633 (11 failed) @ 1.19 h (8k it) | 0.633 | 0.863 |

So far: without a critic, H = 16 fails completely while H = 16 with a critic learns; the critic
also cuts failures sharply at H = 32; H = 64 costs ~2× more wall-clock per iteration.

Update 2026-10-21 23:56Z: H=8 with critic collapsed (best 0.483, then -0.010, 40/40 failed) and
H=16 without critic never learned (best 0.035, 40/40 failed); both stopped at ~2.3 h to free
shared GPUs. H=16 with critic 0.835 (3 failed) @ 2.47 h; H=32 without critic 0.718 (17 failed)
@ 2.78 h; H=64 with critic 0.799 (3 failed) @ 2.25 h. Audit dashboard: notes/AUDIT_DASHBOARD.md.

Update 2026-10-22 01:28Z: SuperMimic main run best 0.926 (0/40 failed) @ 4.72 h; PPO-Sim 0.440
(17 failed) @ 5.0 h. H=16 critic 0.891 (0 failed) @ 3.88 h; H=32 no critic best 0.899 (3 failed)
@ 3.88 h, i.e. it reaches the critic run's nominal return later (as the paper says, the critic
matters most after perturbations: push test pending); H=64 critic 0.844 @ 3.31 h. Fig. 4b
SuperMimic on walk/run/sprint: 0.902 (1/18 failed) @ 1.43 h; PPO-WM queued behind H=64.

## Push recovery (Sec. 5.3) — 2026-10-22 02:33Z, results/push_test.json

Static T-pose reference (synthetic, not in LAFAN1), horizontal root-velocity push at t = 2 s,
8 directions per magnitude; "settled" = root speed < 0.1 m/s for 0.5 s (our criterion).
Matched benchmark return: critic run ckpt 48000 (0.900, 2 failed) vs no-critic ckpt 72000
(0.898, 3 failed).

| push (m/s) | critic: falls / median settle | no critic: falls / median settle |
| --- | --- | --- |
| 0.5 | 0 / 1.2 s | 0 / 1.3 s |
| 1.0 | 0 / 2.8 s | 0 / 1.5 s |
| 1.5 | 0 / 3.3 s | 0 / 1.8 s |
| 2.0 | 0 / 2.6 s (2 never settled) | 0 / 2.6 s |
| 2.5 | 0 / 2.7 s (2 never settled) | 1 / 3.2 s |
| 3.0 | 0 / 5.9 s (1 never settled) | 3 / 3.8 s |
| 3.5 | 2 (5 never settled) | 7 |
| 4.0 | 4 (4 never settled) | 8 (all) |
| total falls | 6 / 64 | 19 / 64 |

Partly reproduced: the critic policy is far more robust to large pushes and the critic-free one
falls at 4 m/s, as in the paper; but our critic-free policy does settle after small pushes (the
paper says it keeps stumbling even at 0.5 m/s) and our critic policy settles slower than the
paper's 0.5-2.4 s.

Update 2026-10-22 03:18Z: H=32 without critic finished (6 h): final 0.823 (10 failed), best 0.899.
Velocity study started (code/supermimic/velocity.py, code/scripts/step_test.py): runs/vel_rl
(job 31941569) and runs/vel_distill (31941570), 100k iterations each on the LAFAN1
walk/run/sprint clips only (paper: LAFAN1 + BONES-SEED walk/run). Dev allocation released.

Update 2026-10-22 04:52Z: SuperMimic main run best 0.941 (0/40) @ 7.77 h (100k iterations);
PPO-Sim 0.470 (14 failed) @ 8.0 h. H=64 critic final 0.914 (0 failed), H=16 critic final 0.901
(1 failed). Velocity runs learning: velocity-env fall share 0.99 -> ~0.06-0.18 by 1.5 h; motion
interface 0.81 on the 18 walk/run/sprint clips. PPO-WM 16k x 32: training return 0.26 after
7.2e8 imagined steps (SuperMimic on the same clips passed 0.82 training return at 2.38 h).
Dashboard now has the Fig. 4b chart with the paper's curves from its figure data.

Update 2026-10-22 06:55Z: main run 0.944 best (0/40) @ 9.30 h; PPO-Sim 0.482 best @ 9 h.
Fig. 4b SuperMimic (walk/run/sprint) finished: 0.948 (0/18) @ 5 h. PPO-WM 16k x 32 @ 3.4 h:
eval 0.492 (5/18 failed); best smoothed training return 0.286 at 1.04e9 imagined steps, which
SuperMimic reached after 8.4e7 -> 12x fewer so far (paper: 64x at the end of PPO-WM's run).
Velocity runs @ 3.5 h: motion interface eval 0.922 (RL run) / 0.896 (distill run).

## Fig. 4b (PPO-WM vs BPTT), LAFAN1 walk/run/sprint — final 2026-10-22 08:31Z

PPO-WM 16k x 32 (the paper's best PPO-WM batch), 5 h: eval 0.523 (4/18 failed); SuperMimic on the
same clips, 5 h: eval 0.948 (0/18). Best smoothed training return of PPO-WM: 0.333 at 2.7e9 imagined
steps; SuperMimic reached 0.333 after 1.0e8 -> 27x fewer imagined steps. At that same return the
paper's own curves (paper/src/figures/data/fig5b) give 12.7x; the paper's 64x is measured at
return 0.84, which its PPO-WM reached after ~6e10 imagined steps (~100 h of our PPO-WM run, not
attempted). Status: partly reproduced (direction and matched-return gap reproduced, the 64x
endpoint not run).

## Main LAFAN1 run finished — 2026-10-22 09:30Z

SuperMimic, 1 x A6000, 12 h of training (155,680 iterations): final benchmark 0.946, 0/40 clips
failed; best 0.947 @ 11.4 h. Target 0.82 first reached @ 1.34 h (paper: within 1.2 h on an H100);
0.866 (the paper's 3 h value) passed @ 1.65 h. PPO-Sim at the same 12 h: 0.492 (12 failed),
still running to 18 h (paper: 0.60 after 18 h).

Update 2026-10-22 11:42Z: rollout benchmark at 32k/64k envs could not be measured: constructing the
mjlab/MuJoCo Warp simulation for >=32,768 worlds exceeded 120 GB of host RAM and thrashed (two
attempts, with and without per-world DR fields); cancelled. Claim C7 rests on 2k-16k (15-22x).
2-GPU data-parallel run (runs/smimic_lafan1_2gpu): 0.323 s/iteration (1 GPU: 0.28 s), eval 0.895
(1/40 failed) @ 1.93 h vs 0.878 @ 1.96 h on one GPU. H=64 without critic: 0.806 (7 failed) @ 1.89 h.
H=128 with critic runs with expandable segments, 1.15 s/iteration, training return 0.17 @ 1.94 h.
Velocity runs @ 8.1 h: motion interface 0.95 on the walk/run/sprint clips; step test at the end.
PPO-Sim @ 14.5 h: 0.522 (9 failed).

## Velocity step test (App. G) — smoke run 2026-10-22 14:18Z, results/step_test_smoke.json

672 held commands (15 x 15 grids on the (vx,vy), (vx,wz), (vy,wz) planes, standing excluded) x 4
trials from standing LAFAN1 frames, 5 s each; relative steady-state error over the last 2 s, median
over non-fallen trials. RL velocity interface at ckpt 88000 (not yet final) vs distilled interface at
100000. Both runs trained on LAFAN1 walk/run/sprint only (paper: LAFAN1 + BONES-SEED walk/run).

| | ours (smoke) | paper |
| --- | --- | --- |
| RL better / distill better (of 672) | 659 / 13 | 650 / 15 |
| median rel. error RL / distill | 0.043 / 0.223 | 0.16 / 0.49 |
| commands where distill fell in all trials | 0 | 7 |

Per plane (median rel. error, RL / distill): vx-vy 0.051 / 0.344, vx-wz 0.039 / 0.154,
vy-wz 0.040 / 0.237. Direction reproduced (RL velocity interface tracks far better); our absolute
errors are lower than the paper's for both, and our distilled interface never falls in all trials.
Final run queued as job 31946447 (after the RL run reaches 100k iterations).

## PPO-Sim final (Fig. 3a, claim C2) — 2026-10-22 15:22Z

PPO-Sim, 1 x A6000, 18 h of training (19,679 iterations, 8192 envs): final benchmark 0.543 with 7/40
clips failed (the same checkpoint evaluated twice at the end gave 0.5429 / 8 failed and 0.5430 / 7
failed); best 0.552 @ 17.0 h. Paper: 0.60 after 18 h on one GPU, never reaching 0.82. Reproduced:
PPO-Sim stays far below SuperMimic, which passed 0.82 at 1.34 h and ended at 0.946 (0/40) after 12 h.
Note the A6000 runs the MLP-heavy SuperMimic update ~2.7x slower than an H100 while MuJoCo Warp is
about as fast, so this wall-clock comparison favours PPO-Sim relative to the paper's setup.

## Velocity step test (App. G, claim C14) — final 2026-10-22 15:40Z, results/step_test.json

RL velocity interface (runs/vel_rl, 100k iterations, motion interface 0.955 on the 18
walk/run/sprint clips) vs distilled interface (runs/vel_distill, 100k iterations, 0.952), same
protocol as the smoke run above (672 commands x 4 trials from standing, 5 s each, error over the
last 2 s). Job 31946447.

| | ours (final) | ours (smoke, RL at 88k) | paper |
| --- | --- | --- | --- |
| RL better / distill better (of 672) | 652 / 20 | 659 / 13 | 650 / 15 |
| median rel. error RL / distill | 0.042 / 0.222 | 0.043 / 0.223 | 0.16 / 0.49 |
| commands where distill fell in all trials | 0 | 0 | 7 |

Per plane (median rel. error, RL / distill): vx-vy 0.045 / 0.343, vx-wz 0.040 / 0.149,
vy-wz 0.042 / 0.235; 90th percentile 0.101 / 0.543. Reproduced: the RL-trained velocity encoder
beats distillation on 652 of 672 commands (paper 650). Both of our interfaces track with lower
absolute error than the paper's and neither falls on every trial of any command; our velocity
study uses only LAFAN1 walk/run/sprint (paper adds BONES-SEED walk/run), a narrower distribution.

## 6 h runs finished — 2026-10-22 16:05Z

- H=64 without critic (job 31945789): final 0.913 (1/40 failed), best 0.940 (0/40) @ 5.45 h; benchmark
  0.82 first @ 3.67 h. Paper: final training return 0.862, 0.82 @ 1.34 h.
- H=128 with critic (job 31945793, expandable segments): final 0.539 (14/40), best 0.556 @ 5.28 h;
  0.82 not reached in 6 h (paper: 0.823 final training return, 0.82 @ 4.62 h). 1.15-1.2 s/iteration
  (~4x H=32). The run sat at the 48 GB memory ceiling the whole time (thousands of allocator
  remap warnings), so its per-iteration cost is an upper bound.
- SuperMimic on 2 GPUs, data parallel (job 31945786): final 0.938 (0/40) @ 6 h (64,905 iterations,
  0.328 s/iteration, lr x sqrt(2)); 1 GPU at the same time: 0.923 (1/40) @ 5.94 h. Benchmark 0.82 first
  @ 1.56 h (1 GPU: 1.34 h; evals are 4000 iterations apart). Two GPUs give a higher plateau on LAFAN1
  but not a faster start; the paper's 2-GPU result is on BONES-SEED (blocked).

Ablation summary (benchmark eval, best / hours to 0.82): H=8 critic 0.483 / never (collapsed);
H=16 no critic 0.035 / never; H=16 critic 0.905 / 2.00 h; H=32 no critic 0.899 / 3.33 h;
H=32 critic 0.947 / 1.34 h; H=64 no critic 0.940 / 3.67 h; H=64 critic 0.914 / 3.31 h;
H=128 critic 0.556 / never. Reproduced: critic needed at short horizons, H=32 fastest, long horizons
much slower per iteration. Not reproduced: in the paper H=64 and H=128 still reach 0.82 within 1.7 h /
4.6 h; ours take 3.3 h and do not get there in 6 h (A6000 is ~2.7x slower per iteration, and the paper's
metric is training return).

## BONES-SEED data (App. E) — 2026-10-22 22:45Z

Downloaded g1.tar.gz (23.5 GB) and metadata v004 with the user's Hugging Face access (archive kept at
data/raw/bones/g1.tar.gz); converted by streaming the archive's CSVs (120 fps -> 50 Hz):
data/processed/bones_seed_raw_50hz.npz, 142,220 clips, 288.35 h, 14.8 GB. Filter
(code/scripts/filter_bones.py, results/bones_filter.json) -> data/processed/bones_seed_50hz.npz:

| stage | ours | paper |
| --- | --- | --- |
| raw clips | 142,220 | 142,220 |
| after name filter | 129,115 | 129,667 |
| after feasibility filter | 128,904 (261.29 h) | 129,514 (264.56 h) |

Feasibility removals: elevated support 126, ground penetration 46, floating feet 39 (211 total;
paper 153). The name filter removes 552 clips more than the paper: the paper gives only SONIC's
list "plus a few terms it misses"; SONIC's list plus the terms the paper names removes 12,261
(paper 12,553), our extra terms (wall_, car_, ledge, ...) remove 844 more, a few of them false
positives (e.g. "ledge" matches "sledgehammering"). Net: 0.5% fewer clips and 1.2% fewer hours than
the paper's training set; not tuned further.

## SONIC's released checkpoint on the BONES-SEED benchmark (Fig. 2a baseline) — 2026-10-22 23:40Z

code/supermimic/sonic.py scores the GEAR-SONIC release (model_encoder.onnx / model_decoder.onnx,
g1 mode) in our simulator with the benchmark protocol and SuperMimic's reward/termination. The
ONNX graphs (batch size 1) are re-implemented as a batched PyTorch module
(code/supermimic/sonic_net.py; code/scripts/convert_sonic.py checks it against onnx.reference:
max pre-quantizer difference 1.5e-7, no token code differs, decoder 2.4e-6). Inputs, joint orders,
default pose, action scale and PD gains follow the deployment reference; SONIC's joint targets are
converted to our action units so the action-rate term is measured the same way for both policies.

| | normalized return | failed (of 128,904) |
| --- | --- | --- |
| SONIC, own PD gains and action scale (paper's setting) | 0.893 (frame-weighted 0.898) | 642 |
| SONIC, also its hip-pitch motor model (7520_22 effort and armature) | 0.893 | 629 |
| paper (SONIC on its 129,514 motions) | 0.90 | 179 |
| SONIC on LAFAN1 (40 clips), for reference | 0.605 | 13 |

The return matches the paper's 0.90; SONIC fails more clips in our simulator (642 vs 179). Only the
hip-pitch gains differ from mjlab's G1 actuators (SONIC assumes the 7520_22 motor there). One
full-benchmark evaluation takes ~17 min on an A6000.

## BONES-SEED runs, check-in 2026-10-23 02:00Z

Benchmark on all 128,904 filtered clips (SONIC in our simulator: 0.893, 642 failed):

| run | 0.43 h | 0.76 h | 1.08 h | 1.41 h | 2.06 h | 2.44 h | 2.71 h |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 A6000 | 0.669 (4,781) | 0.813 (2,482) | 0.856 (1,900) | 0.868 (1,932) | 0.891 (922) | — | 0.893 (893) |
| 2 A6000s (wall-clock) | 0.675 (5,025) | 0.857 (1,634) | 0.881 (1,542) | 0.892 (966) | 0.901 (680) | 0.909 (791) | — |

The 2-GPU run passed SONIC's return in our simulator at 2.11 h wall-clock (0.901); neither run has
fewer failed clips than SONIC yet. Against the paper (Fig. 2a, GPU-hours on H100s): its 1-GPU curve
passes 0.90 at ~0.55 H100-h and is 0.925 at 1 H100-h; with the A6000 ~2.7x slower per iteration, 1 H100-h
is ~2.7 A6000-h, where we are at 0.893. So our BONES-SEED curve is behind the paper's by roughly 2x in
compute beyond the hardware factor; still training (16 h / 26 h budgets).

Dataset-length sweep: training return is not comparable across library sizes under return-based reset
sampling (on 1-40 h libraries every 1 s bin is visited early and the sampler then concentrates 40% of
resets on the hardest bins; on 261 h most bins are never visited and sampling stays near uniform; the
40 h run's training return falls from 0.73 to 0.50 while its episodes shorten). The sweep checkpoints are
therefore being scored on each subset's own benchmark (job 31951478, slurm/eval_runs.slurm).

## BONES-SEED, check-in 2026-10-23 04:05Z

1 A6000 (job 31950477, 5.25 h trained): benchmark 0.902 (884 failed) @ 3.04 h, 0.905 (654) @ 3.36 h,
0.907 (676) @ 4.01 h, 0.914 (695) @ 4.65 h, 0.908 (629) @ 4.98 h. It passed SONIC's return in our simulator
(0.893) at 3.04 h and has fewer failed clips than SONIC (642) at 4.98 h, so it beats SONIC on both counts
in our simulator, but by small margins; the paper's 1-GPU curve is at 0.94 / 25 failed after 3 H100-h
(~8 A6000-h) and keeps rising to 0.96.

2 A6000s (job 31950519, 5.07 h trained): best 0.909 (791 failed) @ 2.44 h, then the benchmark return
declines: 0.902 @ 3.11 h, 0.878 @ 3.78 h, 0.869 @ 4.13 h, 0.835 @ 4.48 h, 0.749 (7,564 failed) @ 4.84 h.
Training return (0.85), training falls (0.5%), world-model error and imagined survival stay healthy;
the critic loss rises (0.084 -> 0.15) and the imagined return J keeps climbing faster than on one GPU.
The drop hits all clip lengths (0-3 s clips 0.904 -> 0.826) and the long clips most (20-40 s 0.933 ->
0.623). Diagnostic job 31952875 (code/scripts/dev/eval_conditions.py) compares the benchmark with
training-like conditions (observation noise, action noise, domain randomization) for checkpoints before
and after the drop.

Dataset-length sweep, each subset scored on its own benchmark (checkpoints every 8000 iterations, so
the hours are coarse):

| library | clips | hours to 0.82 / 0.85 / 0.88 | latest benchmark | 8-step WM error (latest) |
| --- | --- | --- | --- | --- |
| 1 h | 518 | 0.76 / 0.76 / 1.41 | 0.943 (0 failed) @ 2.71 h | 0.076 |
| 7 h | 3,538 | 0.73 / 1.36 / 1.97 | 0.901 (9 failed) @ 2.59 h | 0.156 |
| 40 h | 19,757 | 1.44 / 1.44 / 1.44 | 0.903 (84 failed) @ 3.45 h | 0.248 |
| 261 h (1-GPU run) | 128,904 | 1.08 / 1.08 / 2.06 | 0.908 (629 failed) @ 4.98 h | 0.140 |

Paper (H100 GPU-h to its target return): 0.54, 1.35, 1.91, 1.89 for 1, 7, 40, 264 h. Our hours to 0.88
also grow slowly with the library (1.4 -> 2.1 h for 1 -> 261 h). Our 8-step error is measured on replay
windows, which on small libraries over-sample the hardest bins (see above), so it is not comparable with
the paper's online, nominal-normalized 0.007-0.011.

## The 2-GPU BONES-SEED decline is a deterministic-policy failure — 2026-10-23 05:10Z

code/scripts/dev/eval_conditions.py, 4,096 random BONES-SEED clips, benchmark protocol except as noted
(normalized return / failed clips):

| condition | 2 GPUs ckpt 28k (before) | 2 GPUs ckpt 56k (after) | 1 GPU ckpt 60k |
| --- | --- | --- | --- |
| benchmark (deterministic, no noise, friction 1) | 0.909 / 25 | **0.746 / 274** | 0.908 / 20 |
| + observation noise | 0.906 / 27 | 0.790 / 144 | 0.907 / 20 |
| + action noise 0.1 | 0.887 / 23 | **0.901 / 22** | |
| + domain randomization | 0.905 / 29 | 0.735 / 379 | |
| all three (training conditions) | 0.879 / 31 | 0.893 / 27 | |

With the exploration noise it is trained with, the later 2-GPU policy is as good as the earlier one
(and better under action noise); it is the noise-free, deterministic closed loop that breaks down,
which is why training return and falls stayed healthy. The replay buffer only ever holds states visited
under N(0, 0.1^2) action noise, so the policy is never trained on the states its deterministic closed loop
drifts into; this covariate shift grew during 2-GPU training (lr x sqrt(2)). The 1-GPU policy is still
fine deterministically at 60k. Job 31953843 (code/scripts/dev/rollout_probe.py) checks how the
deterministic rollouts fail (height vs tilt, timing, action chatter).

Latest benchmark: 1 GPU 0.899 (528 failed) @ 5.95 h; 2 GPUs 0.707 (5,912 failed) @ 5.89 h.

## What fails: action chatter in the deterministic closed loop — 2026-10-23 06:00Z

code/scripts/dev/rollout_probe.py, 1,024 BONES-SEED clips of at least 10 s, benchmark protocol:

| checkpoint, condition | failed (height / tilt) | mean |a_t - a_{t-1}| | mean |qd| (rad/s) | mean root height error (m) |
| --- | --- | --- | --- | --- |
| 2 GPUs 28k, deterministic | 5 (2 / 3) | 0.045 | 0.44 | 0.009 |
| 2 GPUs 56k, deterministic | 98 (87 / 11) | 0.092 | 0.66 | 0.018 |
| 2 GPUs 56k, action noise 0.1 | 2 (2 / 0) | 0.145 (mostly the noise) | 0.59 | 0.009 |
| 1 GPU 72k, deterministic | 1 (1 / 0) | 0.049 | 0.47 | 0.009 |

The degraded policy's deterministic actions change twice as much per step as a healthy policy's, joint
speeds are higher than even with injected noise, the root sinks (height error doubles) and the clips end
by the height termination. Under N(0, 0.1^2) action noise the same policy tracks normally. The 1-GPU
policy at 60k behaves normally under all conditions (benchmark 0.908, obs noise 0.907, action noise
0.894, DR 0.905, all 0.885 on 4,096 clips).

Since then the 1-GPU run has begun to slide as well (benchmark 0.914 @ 4.65 h, 0.899, 0.890, 0.880,
0.884 @ 6.93 h; failures 528-637), while the paper's 1-GPU curve rises to 0.96. The 2-GPU run swung
between 0.49 and 0.71 and was stopped at 7.0 h (job 31950519 cancelled with its scorer; checkpoints kept).

Two fix tests on one A6000, 10 h each, in-training benchmark on a fixed 8,192-clip subset every 4,000
iterations:
- runs/bones_fix_detfrac25 (job 31953913): 25% of the environments collect without action noise
  (--det-frac 0.25; not in the paper), so the replay buffer holds deterministic closed-loop states;
- runs/bones_fix_fp32 (job 31953914): networks in fp32 instead of bf16 autocast (--amp false; the paper
  uses TF32), to rule out precision.
The faithful 1-GPU run continues to its 16 h budget.

## Cause of the BONES-SEED collapse: the reset sampler concentrates on near-impossible bins — 2026-10-23 08:30Z

The faithful 1-GPU run collapsed after ~5.8 h: benchmark 0.914 @ 4.65 h -> 0.836 @ 7.91 h -> 0.759
(9,503 failed) @ 8.89 h; training return 0.84 -> 0.47; the 8-step WM error doubled (0.13 -> 0.28) while
the imagined return stayed at ~6.9 and the imagined joint-limit and action-rate penalties grew (the policy
believes it does fine in imagination). The onset coincides with the reset sampler having visited every
1 s bin (bins_visited 0.96 @ 5.4 h, 1.00 @ 6.2 h). Until then unvisited bins (G = 0) carry the softmax
term's mass, so its 40% of resets explore new bins uniformly; afterwards softmax_b(-25 G_b) concentrates on
the lowest-return bins. code/scripts/dev/sampler_stats.py on the run's checkpoints:

| checkpoint | bins visited | bins with G < 0.05 | their share of all resets | half the softmax mass on |
| --- | --- | --- | --- | --- |
| 48k | 0.79 | 213,441 (unvisited) | 0.59 | 107,544 bins / 53,194 clips |
| 72k | 1.00 | 2,249 | 0.18 | 4,549 bins / 3,774 clips (handstands, ...) |
| 108k | 1.00 | 180 | 0.27 | 150 bins / 150 clips (crawl, crouch, ...) |

So by 108k a quarter of all simulator resets start in ~180 near-impossible bins: the replay buffer fills
with falls from motions the robot cannot do, the world model fits them (its error on typical motion
rises), and the policy degrades on everything else. On LAFAN1 (8.8k bins, all learnable) the same
sampler is harmless, which is why the 12 h LAFAN1 run kept improving.

Our G_b update differs from the paper's wording: we update only the bin an episode starts in, with that
episode's return; the paper keeps "a running estimate of the discounted return from [each bin], updated
on the policy's own rollouts", i.e. every bin a rollout passes through, with its return-to-go. Then the
bins leading into a hard segment, not only those starting in it, carry low G, and impossible starting
states are diluted by the many bins traversed by every episode. Implemented as --reset-stat traversed
(sim_env.EnvCfg.reset_stat; every entered bin gets the discounted return-to-go from its entry step over
the largest return attainable over the same remaining horizon; default "start" = all earlier runs).
Test run runs/bones_fix_traversed (job 31957425, 1 A6000, 12 h, 8,192-clip benchmark every 4,000
iterations). The det-frac and fp32 tests keep running as controls; if the sampler is the cause, they will
collapse at a similar point.

## Check-in 2026-10-23 10:20Z: sweep final, fix tests so far

Dataset-length sweep, final (each run 8 h on one A6000, scored on its own library; 8,000-iteration
checkpoints, so hours are coarse):

| library | hours to benchmark 0.82 / 0.85 / 0.88 / 0.90 | best | last |
| --- | --- | --- | --- |
| 1 h (518 clips) | 0.76 / 0.76 / 1.41 / 1.41 | 0.962 (0 failed) @ 7.87 h | same |
| 7 h (3,538) | 0.73 / 1.36 / 1.97 / 2.59 | 0.925 (0) @ 6.44 h | 0.821 (8) @ 7.68 h |
| 40 h (19,757) | 1.44 / 1.44 / 1.44 / 2.78 | 0.915 (68) @ 4.76 h | 0.856 (48) @ 7.37 h |
| 261 h (128,904; faithful 1-GPU run) | 1.08 / 1.08 / 2.06 / 3.04 | 0.914 (695) @ 4.65 h | 0.739 (1,356) @ 11.16 h |

Compute to a fixed return grows slowly with the library (to 0.88: 1.4 -> 2.1 A6000-h from 1 h to 261 h;
paper: 0.54 -> 1.89 H100-h to its target), as the paper reports. All but the 1 h library show the same
late decline as the full run, the 1 h one (whose hardest bins are still learnable) keeps improving —
consistent with the reset-sampler explanation above.

Faithful 1-GPU run at 11.5 h: benchmark oscillates 0.61-0.82 (594-20,682 failed); training return 0.40.

Fix tests (8,192-clip benchmark subset; same scale as the full benchmark within ~0.005):
- detfrac25 @ 3.24 h: 0.904 (51 failed); bins visited 0.75.
- fp32 @ 3.27 h: 0.891 (62 failed); bins visited 0.60 (fp32 is 1.4x slower per iteration).
- traversed @ 1.75 h: 0.883 (69 failed); bins visited 0.83 (traversed updates cover bins much faster).
None has reached full bin coverage yet, where the faithful run started to decline; the faithful run at
similar hours: 0.902 (884 of 128,904, i.e. ~56 per 8,192) @ 3.04 h, 0.868 @ 1.41 h.
