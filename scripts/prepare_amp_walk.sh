#!/usr/bin/env bash
# Convert the existing reference project's walking data; no checkpoint is used.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/local_runtime.sh"
input_dir="${AMP_EXPERT_DIR:-/home/lixiang/workspace/supermimic_AMP/src/assets/motions/g1/amp/WalkandRun}"
exec "$SMIMIC_PY" -m supermimic.amp_data \
  --input "$input_dir" --output data/processed/amp_walk_50hz.npz \
  --preset walking "$@"
