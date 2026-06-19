#!/bin/bash
#SBATCH --job-name=treated_13c
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --nodelist=gpu-1
#SBATCH --output=/new-stg/home/zekai/Forcast_Model/Model/Benchmark_Treated_Condition_13class_QC/slurm-%j.out

set -euo pipefail

ROOT="/new-stg/home/zekai/Forcast_Model"
PY="$ROOT/../miniconda/envs/omnipose_env/bin/python"
mkdir -p "$ROOT/Model/Benchmark_Treated_Condition_13class_QC/efficientnet_b0"
cd "$ROOT"

$PY Code/Training/Model_Training_Treated_Condition/train_treated_condition_raw.py \
  --manifest_csv Data/Benchmark_Treated_Condition_13class/segmentation_qc_combined/qc_manifest/manifest_with_segmentation_qc.csv \
  --out_dir Model/Benchmark_Treated_Condition_13class_QC/efficientnet_b0 \
  --model efficientnet_b0 \
  --mode crossval \
  --epochs 15 \
  --batch_size 32 \
  --lr 3e-4 \
  --img_size 224 \
  --n_folds 5 \
  --use_amp \
  --filter_auto_dropped
