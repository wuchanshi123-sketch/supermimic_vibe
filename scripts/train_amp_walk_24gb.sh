#!/usr/bin/env bash
# Usage: bash scripts/train_amp_walk_24gb.sh [smoke|train] [AMP training arguments...]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/local_runtime.sh"
mode="${1:-train}"
if (( $# )); then shift; fi
case "$mode" in
  smoke)
    workload=(--horizon 8 --num-envs 64 --buffer-steps 128 --wm-batch 32 --wm-batches 1
              --pi-batch 32 --disc-batch 32 --disc-batches 1 --iterations 80
              --wm-only-iters 20 --log-every 10 --ckpt-every 80
              --compile-env false --compile-wm false)
    ;;
  train)
    workload=(--num-envs 4096 --buffer-steps 1000 --wm-batch 512 --wm-batches 10
              --pi-batch 256 --disc-batch 512 --disc-batches 2 --iterations 50000
              --wm-only-iters 500 --log-every 50 --ckpt-every 1000
              --compile-env true --compile-wm true)
    ;;
  *) echo "Usage: $0 [smoke|train] [AMP training arguments...]" >&2; exit 2 ;;
esac
# Final hyperparameters from runs/amp_walk_best/training_config.json.
# Fresh runs initialize random weights; resume and output paths remain explicit.
exec "$SMIMIC_PY" -m supermimic.amp_train \
  --motions data/processed/amp_walk_50hz.npz \
  --out "runs/amp_walk_24gb_${mode}" \
  --horizon 32 --k 8 --history 4 --utd 0.4 --amp false \
  --actor-lr-scale 0.5 --linear-std 0.25 --action-rate-weight -0.1 \
  --command-low 0.2,0,0 --command-high 0.8,0,0 \
  --standing-command-probability 0.1 \
  --expert-clip-prefixes walk_forward_ \
  --expert-time-scales 0.4,0.6,0.8 --expert-augmented-fraction 0.5 \
  --style-command-gate 0.2 \
  "${workload[@]}" "$@"
