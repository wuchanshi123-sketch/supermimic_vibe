#!/usr/bin/env bash
#SBATCH --job-name=smimic-evalwatch
#SBATCH --account=pci
#SBATCH --partition=pci
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%j-%x.out
#SBATCH --error=slurm/logs/%j-%x.out
# Scores every new checkpoint of a run with the benchmark protocol (App. F) while it trains,
# as the paper does after training, so evaluation never takes training time. Stops once the
# run has written DONE and every checkpoint is scored. Appends to <run>/eval.jsonl.
# Usage (from the repository root): sbatch slurm/eval_watch.slurm <run dir> <motions.npz> [every]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"  # submit from the repository root
source env.sh
source setup_local_env.sh
nvidia-smi --query-gpu=name --format=csv
$SMIMIC_PY -m supermimic.evaluate --run "$1" --motions "$2" --watch --every "${3:-1}"
