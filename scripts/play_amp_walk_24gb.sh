#!/usr/bin/env bash
# Native AMP playback; F710 is auto-detected. Use --gamepad off for keyboard only.
# Default checkpoint: validated best.pt when present; --ckpt selects another model.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/local_runtime.sh"
export MUJOCO_GL=glfw
export PYOPENGL_PLATFORM=glx
exec "$SMIMIC_PY" code/scripts/play_amp_walk.py "$@"
