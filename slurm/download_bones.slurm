#!/usr/bin/env bash
#SBATCH --job-name=bones-dl
#SBATCH --account=pci
#SBATCH --partition=pci
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%j-%x.out
#SBATCH --error=slurm/logs/%j-%x.out
# Downloads BONES-SEED g1.tar.gz (23.5 GB) and its metadata, then converts the archive to one
# 50 Hz motion library by streaming its ~142k CSVs (never unpacked to disk).
# Gated dataset: the Hugging Face account must have been granted access; the token is read
# from $HF_TOKEN or $HF_HOME/token (env.sh puts HF_HOME under .cache/, which git ignores).
# Usage (from the repository root): sbatch slurm/download_bones.slurm
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"  # submit from the repository root
source env.sh
source setup_local_env.sh
if [ -z "${HF_TOKEN:-}" ] && [ ! -s "$HF_HOME/token" ]; then
  echo "no Hugging Face token: set HF_TOKEN or write it to $HF_HOME/token"; exit 1
fi
RAW=$SMIMIC_ROOT/data/raw/bones
mkdir -p "$RAW" data/processed data/raw/bones_meta
# The archive goes to node-local disk when it has room, else to shared storage (one large file).
avail_gb=$(df --output=avail -BG /tmp | tail -1 | tr -dc 0-9)
echo "node-local /tmp free: ${avail_gb} GB"
if [ "$avail_gb" -ge 40 ]; then export DL=$SMIMIC_LOCAL/bones; else export DL=$RAW; fi
mkdir -p "$DL"
SIZE=23499973647  # bytes of g1.tar.gz on the Hub
if [ "$(stat -c %s "$RAW/g1.tar.gz" 2>/dev/null || echo 0)" = "$SIZE" ] && [ -s data/raw/bones_meta/seed_metadata_v004.parquet ]; then
  echo "archive already on shared storage; skipping the download"
  SRC=$RAW/g1.tar.gz
else
  # Install the Hub client into the job's own env only, never into a fallback interpreter.
  [ "$SMIMIC_PY" = "$SMIMIC_LOCAL/venv/bin/python" ] || { echo "download needs the node-local env"; exit 1; }
  $SMIMIC_LOCAL/bin/uv pip install --python $SMIMIC_PY "huggingface_hub[hf_xet]"
  # Downloads from the compute nodes drop now and then: retry, and after three failures
  # switch from the Xet transfer to plain HTTP.
  ok=0
  for attempt in 1 2 3 4 5 6; do
    if [ "$attempt" -ge 4 ]; then export HF_HUB_DISABLE_XET=1; fi
    if $SMIMIC_PY - << 'PY'
import os
from huggingface_hub import hf_hub_download
for f in ["metadata/seed_metadata_v004.parquet", "g1.tar.gz"]:
    p = hf_hub_download("bones-studio/seed", f, repo_type="dataset", local_dir=os.environ["DL"])
    print(p, os.path.getsize(p), flush=True)
PY
    then ok=1; break; fi
    echo "[download] attempt $attempt failed; retrying in $((attempt * 60)) s"
    sleep $((attempt * 60))
  done
  [ "$ok" = 1 ] || { echo "[download] gave up"; exit 1; }
  [ "$(stat -c %s "$DL/g1.tar.gz")" = "$SIZE" ] || { echo "[download] size mismatch"; exit 1; }
  cp "$DL/metadata/seed_metadata_v004.parquet" data/raw/bones_meta/
  # Keep one copy of the archive on shared storage right away, so a failed conversion
  # never costs another download.
  if [ "$DL" != "$RAW" ]; then cp "$DL/g1.tar.gz" "$RAW/"; fi
  SRC=$DL/g1.tar.gz
fi
$SMIMIC_PY code/scripts/prepare_motions.py --format bones --fps-in 120 --input "$SRC" \
  --strip-prefix g1/ --output data/processed/bones_seed_raw_50hz.npz --workers 16
ls -la data/processed/bones_seed_raw_50hz.npz "$RAW"
