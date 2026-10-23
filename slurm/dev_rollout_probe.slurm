#!/usr/bin/env bash
#SBATCH --job-name=rollout-probe
#SBATCH --account=pci
#SBATCH --partition=pci
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%j-%x.out
#SBATCH --error=slurm/logs/%j-%x.out
# Diagnostic: termination type, timing and action chatter in deterministic rollouts.
# Usage (from the repository root): sbatch slurm/dev_rollout_probe.slurm motions.npz n_clips ckpt [ckpt ...]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"  # submit from the repository root
source env.sh
source setup_local_env.sh
$SMIMIC_PY code/scripts/dev/rollout_probe.py "$@"
