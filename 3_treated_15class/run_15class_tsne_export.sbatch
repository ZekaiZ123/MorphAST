#!/bin/bash
#SBATCH --job-name=15c_tsne
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --nodelist=gpu-1
#SBATCH --output=/new-stg/home/zekai/Forcast_Model/Code/Paper_Figures/output/prism_15class/slurm-%j.out

# Stage B of the Prism-data plan:
#   Compute embeddings + PCA + t-SNE on the 15-class held-out fields and emit
#   Fig 3A (3 treatment series) + Fig 3G (15 strain x treatment series) Prism CSVs.
#
# Runtime on 1 GPU: ~10-15 min (vs 1.5-3 hr on CPU).

set -euo pipefail

ROOT="/new-stg/home/zekai/Forcast_Model"
PY="$ROOT/../miniconda/envs/omnipose_env/bin/python"
cd "$ROOT"

$PY Code/Paper_Figures/export_tsne_for_prism_15class.py
