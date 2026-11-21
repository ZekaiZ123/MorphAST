#!/bin/bash
#SBATCH --job-name=prep_report
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail
RUN_DIR="${1:?Pass the frozen sensitivity run directory}"
PY="${REBUTTAL_PYTHON:-/new-stg/home/zekai/miniconda/envs/omnipose_env/bin/python}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
"$PY" "$RUN_DIR/source/Code/Training/Rebuttal/summarize_preparation_sensitivity.py" \
  --run_dir "$RUN_DIR" --require_complete
