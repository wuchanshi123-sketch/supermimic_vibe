#!/usr/bin/env bash
#SBATCH --job-name=smimic-render
#SBATCH --account=pci
#SBATCH --partition=pci
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/%j-%x.out
#SBATCH --error=slurm/logs/%j-%x.out
# Renders benchmark rollouts of a checkpoint to MP4 (code/scripts/render_rollouts.py).
# Usage (from the repository root): sbatch slurm/render_rollouts.slurm --ckpt ... --motions ... --clips a,b --out dir [--title ...]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"  # submit from the repository root
source env.sh
source setup_local_env.sh
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
$SMIMIC_PY code/scripts/render_rollouts.py "$@"
