# Source after env.sh inside a job. Builds (once per node) a node-local Python env from
# requirements.txt with uv, downloading wheels instead of importing from a slow shared
# filesystem, then points SMIMIC_PY at it.
export SMIMIC_LOCAL="${SMIMIC_LOCAL:-/tmp/${USER}_smimic}"
LOCAL=$SMIMIC_LOCAL
VENV=$LOCAL/venv
mkdir -p $LOCAL
if [ ! -f "$VENV/.ready" ]; then
  exec 9>$LOCAL/.setup.lock
  flock 9
  if [ ! -f "$VENV/.ready" ]; then
    echo "[setup_local_env] building $VENV on $(hostname)"
    export UV_CACHE_DIR=$LOCAL/uv-cache UV_PYTHON_INSTALL_DIR=$LOCAL/uv-python
    if [ ! -x $LOCAL/bin/uv ]; then
      mkdir -p $LOCAL/bin
      curl -sSL https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz \
        | tar -xz -C $LOCAL/bin --strip-components=1
    fi
    rm -rf $VENV
    $LOCAL/bin/uv venv --python 3.11 $VENV
    # PyPI downloads from the compute nodes fail now and then; retry for ~10 minutes.
    for attempt in 1 2 3 4 5 6; do
      $LOCAL/bin/uv pip install --python $VENV/bin/python -r "$SMIMIC_ROOT/requirements.txt" && break
      echo "[setup_local_env] install attempt $attempt failed; retrying in $((attempt * 30)) s"
      sleep $((attempt * 30))
    done
    $VENV/bin/python -c "import torch, mjlab, mujoco_warp" && touch $VENV/.ready
  fi
  flock -u 9
fi
if [ ! -f "$VENV/.ready" ]; then
  # Optional fallback: an existing interpreter with the same packages (slower to start
  # from a shared filesystem). Without one, stop here instead of running half a job.
  if [ -n "${SMIMIC_FALLBACK_PY:-}" ] && [ -x "$SMIMIC_FALLBACK_PY" ]; then
    echo "[setup_local_env] env build failed; falling back to $SMIMIC_FALLBACK_PY"
    export SMIMIC_PY=$SMIMIC_FALLBACK_PY
    export WARP_CACHE_PATH=$LOCAL/warp-cache
    mkdir -p $WARP_CACHE_PATH
    return 0
  fi
  echo "[setup_local_env] could not build $VENV (package downloads failed)"
  return 1
fi
export SMIMIC_PY=$VENV/bin/python
export WARP_CACHE_PATH=$LOCAL/warp-cache
mkdir -p $WARP_CACHE_PATH
