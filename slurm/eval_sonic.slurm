#!/usr/bin/env bash
#SBATCH --job-name=eval-sonic
#SBATCH --account=pci
#SBATCH --partition=pci
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%j-%x.out
#SBATCH --error=slurm/logs/%j-%x.out
# Scores SONIC's released g1 policy with the benchmark protocol (supermimic/sonic.py).
# Usage (from the repository root): sbatch slurm/eval_sonic.slurm <motions.npz> <out.json> [gains|full] [clips]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"  # submit from the repository root
source env.sh
source setup_local_env.sh
nvidia-smi --query-gpu=name --format=csv
$SMIMIC_PY -m supermimic.sonic --motions "$1" --out "$2" --motor "${3:-gains}" --clips "${4:-0}"
