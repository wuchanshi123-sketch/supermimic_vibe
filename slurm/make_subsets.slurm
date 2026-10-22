#!/usr/bin/env bash
#SBATCH --job-name=bones-subsets
#SBATCH --account=pci
#SBATCH --partition=pci
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%j-%x.out
#SBATCH --error=slurm/logs/%j-%x.out
# Nested 1 / 7 / 40 h subsets of the filtered BONES-SEED library (Fig. 2b,c sweep).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"  # submit from the repository root
source env.sh
source setup_local_env.sh
$SMIMIC_PY code/scripts/make_subsets.py data/processed/bones_seed_50hz.npz data/processed/bones_seed_sub 1,7,40
