#!/usr/bin/env bash
# Local Python environment and caches for this checkout.
set -euo pipefail
SMIMIC_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SMIMIC_PROJECT_ROOT"
source "$SMIMIC_PROJECT_ROOT/env.sh"
export SMIMIC_PY="$SMIMIC_PROJECT_ROOT/.venv/bin/python"
export PYTHONPATH="$SMIMIC_PROJECT_ROOT/code"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-2}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
if [[ ! -x "$SMIMIC_PY" ]]; then
  echo "Missing project environment: $SMIMIC_PY" >&2
  return 1 2>/dev/null || exit 1
fi
