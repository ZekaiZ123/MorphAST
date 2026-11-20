#!/bin/bash
#SBATCH --job-name=rebuttal_cv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=1-10%4

set -euo pipefail
RUN_DIR="${1:?Pass the prepared run directory}"
PY="${REBUTTAL_PYTHON:-/new-stg/home/zekai/miniconda/envs/omnipose_env/bin/python}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1

if [[ -n "${2:-}" ]]; then
  TASK="$2"
  FOLD="$SLURM_ARRAY_TASK_ID"
  if (( FOLD < 1 || FOLD > 5 )); then
    echo "Single-cohort runs require --array=1-5%4" >&2
    exit 2
  fi
elif (( SLURM_ARRAY_TASK_ID >= 1 && SLURM_ARRAY_TASK_ID <= 5 )); then
  TASK=untreated
  FOLD="$SLURM_ARRAY_TASK_ID"
elif (( SLURM_ARRAY_TASK_ID >= 6 && SLURM_ARRAY_TASK_ID <= 10 )); then
  TASK=treated
  FOLD=$(( SLURM_ARRAY_TASK_ID - 5 ))
else
  echo "Default two-cohort runs require array indices 1-10" >&2
  exit 2
fi

"$PY" "$RUN_DIR/source/Code/Training/Rebuttal/run_nested_cv.py" \
  --run_dir "$RUN_DIR" --task "$TASK" --fold "$FOLD"
