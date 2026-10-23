#!/usr/bin/env bash
#SBATCH --job-name=eval-conditions
#SBATCH --account=pci
#SBATCH --partition=pci
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%j-%x.out
#SBATCH --error=slurm/logs/%j-%x.out
# Diagnostic: checkpoints under the benchmark protocol vs training-like conditions.
# Usage (from the repository root): sbatch slurm/dev_eval_conditions.slurm motions.npz n_clips ckpt [ckpt ...]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"  # submit from the repository root
source env.sh
source setup_local_env.sh
$SMIMIC_PY code/scripts/dev/eval_conditions.py "$@"
