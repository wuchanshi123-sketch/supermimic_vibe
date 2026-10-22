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
