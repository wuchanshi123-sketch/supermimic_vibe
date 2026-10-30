#!/usr/bin/env bash
# Usage: bash scripts/evaluate_amp_walk.sh [checkpoint.pt] [evaluation arguments...]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/local_runtime.sh"
checkpoint="${1:-runs/amp_walk_24gb_train/last.pt}"
if (( $# )); then shift; fi
exec "$SMIMIC_PY" -m supermimic.amp_eval \
  --ckpt "$checkpoint" --motions data/processed/amp_walk_50hz.npz "$@"
