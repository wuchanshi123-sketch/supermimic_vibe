#!/usr/bin/env bash
# Usage: bash scripts/evaluate_lafan1_24gb.sh runs/lafan1_24gb_train [evaluate.py arguments...]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/local_runtime.sh"
run_dir="${1:-runs/lafan1_24gb_train}"
if (( $# )); then shift; fi
exec "$SMIMIC_PY" -m supermimic.evaluate \
  --run "$run_dir" --motions data/processed/lafan1_50hz.npz \
  --num-envs 40 "$@"
