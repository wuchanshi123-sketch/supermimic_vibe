# AMP walking training supervision

All times in this journal use Asia/Shanghai (UTC+8).

The user authorized continuing the scratch AMP training, evaluating the actual gait,
and correcting code or training settings as needed. A successful outcome requires
real simulator evaluation and visual inspection; a small discriminator loss or a
larger imagined return alone is insufficient.

## Initial acceptance targets

- Start from the nominal standing pose without an expert physical reset.
- Walk for 20 seconds under forward commands of 0.3, 0.5 and 0.8 m/s, including
  checking both the nominal reset and varied reset conditions.
- Aim for at least 90% full-horizon survival over varied trials, planar tracking
  RMSE below 0.25 m/s and yaw error below 0.35 rad/s for straight walking.
- Inspect gait videos for alternating steps, foot clearance, foot sliding,
  stable torso posture and natural arm/leg coordination.
- Compare fixed expert-feature diagnostics across checkpoints. The online
  discriminator changes during training, so its reward is not a fixed style metric.
- Evaluate standing, turns and lateral commands separately and report their
  limits rather than allowing them to hide poor forward walking.

These are working acceptance targets, not a claim about achieved performance.

## 2026-10-29 19:34 — baseline continuation

- Original run: `runs/amp_walk_24gb_train`; it had been interrupted at iteration
  2144, with 640 actor updates recorded at iteration 2100.
- At iteration 2100, mean completed-episode duration was 1.95 s, completed-episode
  fall fraction was 1.0, planar velocity error was 0.663 m/s, and WM8 normalized
  error was 0.286. This is early learning, not stable walking.
- Preserved the interrupted checkpoint, logs, configuration and source under
  `runs/amp_supervision/baseline_20261029_193423/`.
- Resumed with the existing configuration to iteration 5000 before making
  algorithm changes. Command and process metadata are in
  `runs/amp_supervision/active.json`; full stdout is kept next to it.
- Separate diagnostics are being added for fixed expert style features,
  gait/actuation measurements, headless video, current-policy dynamics validation,
  and rolling training curves.

No external pretrained policy or teacher is used. Resuming weights learned by
this scratch run preserves its initialization provenance.

## 2026-10-29 19:48 — balancing is not walking

- Completed the unchanged baseline continuation to iteration 5000. Its last
  training window had 18.90 s mean duration, 0.071 completed-episode fall
  fraction, and 0.501 m/s planar command error.
- The independent iteration-4000 evaluation used 16 distinct, seeded mild
  perturbations for each command. All trials survived 20 s, but measured forward
  velocity was approximately zero for commands 0.3, 0.5 and 0.8 m/s (respective
  planar RMSE 0.302, 0.501 and 0.801 m/s). This is a stationary balance solution,
  not successful command-conditioned walking.
- Iteration-2000 and -3000 videos and independent expert-feature reports are
  saved under `runs/amp_supervision/`. The iteration-3000 video likewise showed
  balancing and yaw drift rather than a clear alternating forward gait.
- Current-policy dynamics measurements at iteration 3000 found RSI H=16
  position errors of about 7.5–11.3 mm and planar velocity errors of 0.039–0.056
  m/s. H=32 had materially larger errors. This supports a controlled H=16 trial
  rather than assuming that the short H=8 window is sufficient.
- Next experiment: compare H=8 and H=16 continuations from the identical
  iteration-5000 checkpoint, preserving reward, data and other optimizer settings.
- A performance-only change will route repeated AMP FK through the existing
  compiled simulator FK. Feature and gradient parity must pass before using it.


## 2026-10-30 02:33 — acceptance passed

Selected `runs/amp_walk_forward_expert_h32/ckpt_0042000.pt`, published locally as `runs/amp_walk_best/best.pt`. Final fixed-command trials: 192/192 survived 30 s; RMSE at 0.3/0.5/0.8 m/s was 0.084/0.088/0.103 m/s. All 64 low-speed copies sustained alternating steps both after 2 s and in the final half. All 64 copies survived the 40 s command sequence. Videos were visually inspected. The final CPU suite passed all 120 tests. GPU compiled/eager feature and gradient parity was retained.

The effective final changes are recorded in the selected model configuration: H=32, actor LR scale 0.5, linear width 0.25, action-rate coefficient -0.1, focused forward commands 0.2–0.8 m/s with 10% standing, forward-only positive clip sampling, and a 50% mixture of original and time-scaled moving expert pairs (0.4/0.6/0.8). No external teacher/pretrained weights were used.

Diagnostics ruled out dead clamp gradients, a dominant angular-reward conflict, and incorrect local dynamics action gradients. Matching task command coverage and forward expert support resolved the remaining low-speed shuffle. Full reports, videos, per-copy traces and provenance are in `runs/amp_walk_best/`. All historical checkpoints remain. The supervisor, trainers, evaluators and live monitor were stopped after successful validation; GPU compute-process inventory was empty.
