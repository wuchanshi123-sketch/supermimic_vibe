#!/usr/bin/env bash
#SBATCH --job-name=sampler-stats
#SBATCH --account=pci
#SBATCH --partition=pci
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/%j-%x.out
#SBATCH --error=slurm/logs/%j-%x.out
# Diagnostic: reset-sampler mass from checkpoint bin statistics (CPU).
# Usage (from the repository root): sbatch slurm/dev_sampler_stats.slurm motions.npz ckpt [ckpt ...]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"  # submit from the repository root
source env.sh
source setup_local_env.sh
$SMIMIC_PY code/scripts/dev/sampler_stats.py "$@"
