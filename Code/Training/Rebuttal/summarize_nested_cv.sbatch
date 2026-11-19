#!/bin/bash
#SBATCH --job-name=rebuttal_report
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00

set -euo pipefail
RUN_DIR="${1:?Pass the prepared run directory}"
PY="${REBUTTAL_PYTHON:-/new-stg/home/zekai/miniconda/envs/omnipose_env/bin/python}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
"$PY" "$RUN_DIR/source/Code/Training/Rebuttal/summarize_nested_cv.py" \
  --run_dir "$RUN_DIR" --require_complete
