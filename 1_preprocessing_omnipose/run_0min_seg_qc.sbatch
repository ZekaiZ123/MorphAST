#!/bin/bash
#SBATCH --job-name=0min_seg_qc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --nodelist=gpu-1
#SBATCH --output=/new-stg/home/zekai/Forcast_Model/Data/Benchmark_0min_5class/segmentation_qc/slurm-%j.out

set -euo pipefail

ROOT="/new-stg/home/zekai/Forcast_Model"
PY="$ROOT/../miniconda/envs/omnipose_env/bin/python"

MANIFEST="$ROOT/Data/Benchmark_0min_5class/manifest_0min_fiveclass.csv"
WORK_DIR="$ROOT/Data/Benchmark_0min_5class/segmentation_qc"
TILES_DIR="$WORK_DIR/tiling"
SEG_OUT="$WORK_DIR/seg_outputs"
QC_OUT="$WORK_DIR/qc_manifest"

mkdir -p "$WORK_DIR" "$SEG_OUT" "$QC_OUT"
cd "$ROOT"

echo "[STEP 1] Tiling 0_min images..."
$PY Code/Training/Model_Training_Treated_Condition/prepare_treated_segmentation_tiles.py \
  --manifest_csv "$MANIFEST" \
  --out_dir "$TILES_DIR" \
  --rows 3 \
  --cols 3 \
  --mode crop \
  --out_ext tif \
  --workers 8

echo "[STEP 2] Running Omnipose segmentation..."
$PY Code/Segementation/run_tiles_with_stats_tif_meta_data_v2.py \
  --tiles_root "$TILES_DIR/tiles" \
  --tile_output_root "$SEG_OUT" \
  --gpu 0 \
  --mask_threshold -2.4 \
  --flow_threshold 0.5 \
  --min_size 20 \
  --pixel_size_um 0.25641 \
  --min_cells 1 \
  --max_cell_area_um2 30 \
  --max_artifact_area_um2 100 \
  --filter_mode tag \
  --scan_policy guess_tiles

echo "[STEP 3] Aggregating QC..."
$PY Code/Training/Model_Training_Treated_Condition/build_treated_segmentation_qc_manifest.py \
  --manifest_csv "$MANIFEST" \
  --tile_image_map_csv "$TILES_DIR/tile_image_map.csv" \
  --seg_metadata_csv "$SEG_OUT/metadata.csv" \
  --out_dir "$QC_OUT" \
  --min_pass_tiles 3 \
  --min_pass_fraction 0.34 \
  --min_total_cells 6 \
  --max_large_cell_tiles 0 \
  --max_artifact_tiles 0

echo "[DONE] 0_min segmentation QC pipeline complete."
