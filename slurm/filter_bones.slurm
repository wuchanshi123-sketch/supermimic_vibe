#!/usr/bin/env bash
#SBATCH --job-name=bones-filter
#SBATCH --account=pci
#SBATCH --partition=pci
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%j-%x.out
#SBATCH --error=slurm/logs/%j-%x.out
# Appendix E filter of the raw BONES-SEED G1 library (name filter + FK feasibility filter).
# Usage (from the repository root):
#   sbatch slurm/filter_bones.slurm data/processed/bones_seed_raw_50hz.npz data/processed/bones_seed_50hz.npz results/bones_filter.json
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"  # submit from the repository root
source env.sh
source setup_local_env.sh
nvidia-smi --query-gpu=name --format=csv
$SMIMIC_PY code/scripts/filter_bones.py --input "$1" --output "$2" --report "$3"
