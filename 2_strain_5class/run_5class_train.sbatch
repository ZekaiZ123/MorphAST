#!/bin/bash
#SBATCH --job-name=0min_bench_qc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --nodelist=gpu-1
#SBATCH --output=/new-stg/home/zekai/Forcast_Model/Model/Benchmark_0min_5class_QC/slurm-%j.out

set -euo pipefail

ROOT="/new-stg/home/zekai/Forcast_Model"
PY="$ROOT/../miniconda/envs/omnipose_env/bin/python"
mkdir -p "$ROOT/Model/Benchmark_0min_5class_QC"
cd "$ROOT"

$PY Code/Training/Model_Training_5_class/run_0min_fiveclass_benchmark.py \
  --manifest_csv Data/Benchmark_0min_5class/segmentation_qc/qc_manifest/manifest_with_segmentation_qc.csv \
  --out_dir Model/Benchmark_0min_5class_QC \
  --models resnet18 efficientnet_b0 swin_tiny \
  --epochs 15 \
  --batch_size 32 \
  --lr 3e-4 \
  --img_size 224 \
  --n_folds 5 \
  --use_amp \
  --filter_auto_dropped
