#!/usr/bin/env bash
# Usage: bash scripts/train_lafan1_24gb.sh [smoke|train] [train.py arguments...]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/local_runtime.sh"
mode="${1:-train}"
if (( $# )); then shift; fi
case "$mode" in
  smoke) iterations=200; checkpoint=200; log_interval=20 ;;
  train) iterations=20000; checkpoint=500; log_interval=50 ;;
  *) echo "Usage: $0 [smoke|train] [train.py arguments...]" >&2; exit 2 ;;
esac
exec "$SMIMIC_PY" -m supermimic.train \
  --motions data/processed/lafan1_50hz.npz \
  --out "runs/lafan1_24gb_${mode}" \
  --num-envs 4096 --buffer-steps 1000 \
  --wm-batch 512 --wm-batches 10 --pi-batch 512 \
  --k 8 --horizon 32 --utd 0.4 \
  --amp true --compile-wm true \
  --iterations "$iterations" --ckpt-every "$checkpoint" \
  --log-every "$log_interval" --eval-every 0 \
  "$@"
