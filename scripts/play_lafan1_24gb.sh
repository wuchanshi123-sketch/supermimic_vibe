#!/usr/bin/env bash
# Native real-time viewer: bash scripts/play_lafan1_24gb.sh --clip walk1_subject1
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/local_runtime.sh"
export MUJOCO_GL=glfw
export PYOPENGL_PLATFORM=glx
exec "$SMIMIC_PY" code/scripts/play_lafan1.py "$@"
