#!/bin/bash
#SBATCH --job-name=treated15_features
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# SLURM copies the script; submission must supply the repository explicitly.
ROOT="${3:-$ROOT}"
PY="${REBUTTAL_PYTHON:-/new-stg/home/zekai/miniconda/envs/omnipose_env/bin/python}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="${TMPDIR:-/tmp}/forcast_matplotlib"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
cd "$ROOT"
"$PY" Code/Training/Rebuttal/corrected_treated15.py \
  --run_dir "${1:?Pass the corrected run directory}" --out_dir "${2:?Pass a new feature directory}"
