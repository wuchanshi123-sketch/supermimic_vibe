#!/usr/bin/env bash
# Rebuild the 40-clip LAFAN1 library using the project environment.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/local_runtime.sh"
input_dir="${LAFAN1_G1_DIR:-$SMIMIC_PROJECT_ROOT/data/raw/lafan1_g1}"
mkdir -p data/processed
exec "$SMIMIC_PY" code/scripts/prepare_motions.py \
  --format lafan --fps-in 30 --fps-out 50 \
  --input "$input_dir" --output data/processed/lafan1_50hz.npz "$@"
