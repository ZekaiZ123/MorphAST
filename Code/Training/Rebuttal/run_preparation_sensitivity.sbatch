#!/bin/bash
#SBATCH --job-name=prep_sensitivity
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=1-10%4

set -euo pipefail
RUN_DIR="${1:?Pass the frozen sensitivity run directory}"
PY="${REBUTTAL_PYTHON:-/new-stg/home/zekai/miniconda/envs/omnipose_env/bin/python}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
if (( SLURM_ARRAY_TASK_ID >= 1 && SLURM_ARRAY_TASK_ID <= 5 )); then
  TASK=field_baseline
  FOLD="$SLURM_ARRAY_TASK_ID"
elif (( SLURM_ARRAY_TASK_ID >= 6 && SLURM_ARRAY_TASK_ID <= 10 )); then
  TASK=preparation_sensitivity
  FOLD=$((SLURM_ARRAY_TASK_ID - 5))
else
  echo "Sensitivity runs require array indices 1-10" >&2
  exit 2
fi
"$PY" "$RUN_DIR/source/Code/Training/Rebuttal/run_nested_cv.py" \
  --run_dir "$RUN_DIR" --task "$TASK" --fold "$FOLD"
