#!/usr/bin/env bash
#SBATCH --job-name=smimic-evalruns
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
# Scores the not-yet-scored checkpoints of several runs once (benchmark protocol, App. F).
# Usage (from the repository root): sbatch slurm/eval_runs.slurm <run dir>:<motions.npz>[:<out file>[:<every>]] [...]
# (out file defaults to eval.jsonl inside the run dir; every = score every n-th checkpoint)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"  # submit from the repository root
source env.sh
source setup_local_env.sh
for spec in "$@"; do
  IFS=: read -r run motions out every <<< "$spec"
  $SMIMIC_PY -m supermimic.evaluate --run "$run" --motions "$motions" --out "${out:-eval.jsonl}" --every "${every:-1}"
done
