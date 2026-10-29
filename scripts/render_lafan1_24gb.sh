#!/usr/bin/env bash
# Usage: bash scripts/render_lafan1_24gb.sh --ckpt runs/<run>/last.pt --clips walk1_subject1 --out runs/<run>/videos
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/local_runtime.sh"
if ! command -v ffmpeg >/dev/null 2>&1; then
  ffmpeg_path="$("$SMIMIC_PY" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
  mkdir -p "$SMIMIC_PROJECT_ROOT/.cache/bin"
  ln -sfn "$ffmpeg_path" "$SMIMIC_PROJECT_ROOT/.cache/bin/ffmpeg"
  export PATH="$SMIMIC_PROJECT_ROOT/.cache/bin:$PATH"
fi
exec "$SMIMIC_PY" code/scripts/render_rollouts.py \
  --motions data/processed/lafan1_50hz.npz "$@"
