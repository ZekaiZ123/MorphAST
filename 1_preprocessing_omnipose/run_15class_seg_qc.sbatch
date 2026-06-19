#!/bin/bash
#SBATCH --job-name=15class_seg_qc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/new-stg/home/zekai/Forcast_Model/Data/Benchmark_Treated_Condition_15class/slurm-%j.out

# Stage 3 of the 15-class benchmark integration plan:
#   1. Filter the 15-class manifest to just Data_additional rows (~6k new fields).
#   2. Tile the new fields into 3x3 grids.
#   3. Run Omnipose on the new tiles.
#   4. Aggregate per-field seg_qc_* columns for the new rows.
#   5. Merge with the existing 13-class seg_qc manifest -> the full 15-class
#      qc manifest used as training input in Stage 4.
#
# The existing 13-class seg_qc artifacts under
# Data/Benchmark_Treated_Condition_13class/segmentation_qc_combined/
# are NOT touched.

set -euo pipefail

ROOT="/new-stg/home/zekai/Forcast_Model"
PY="$ROOT/../miniconda/envs/omnipose_env/bin/python"

# Inputs
MANIFEST_FULL="$ROOT/Data/Benchmark_Treated_Condition_15class/manifest_treated_condition.csv"
EXISTING_QC_MANIFEST="$ROOT/Data/Benchmark_Treated_Condition_13class/segmentation_qc_combined/qc_manifest/manifest_with_segmentation_qc.csv"

# Work dirs
WORK_DIR="$ROOT/Data/Benchmark_Treated_Condition_15class/segmentation_qc_additional"
FILTER_MANIFEST="$WORK_DIR/manifest_additional_only.csv"
TILES_DIR="$WORK_DIR/tiling"
SEG_OUT="$WORK_DIR/seg_outputs"
QC_OUT_NEW="$WORK_DIR/qc_manifest"

# Combined output
COMBINED_QC_DIR="$ROOT/Data/Benchmark_Treated_Condition_15class/segmentation_qc_combined/qc_manifest"
COMBINED_QC_CSV="$COMBINED_QC_DIR/manifest_with_segmentation_qc.csv"

mkdir -p "$WORK_DIR" "$SEG_OUT" "$QC_OUT_NEW" "$COMBINED_QC_DIR"
cd "$ROOT"

echo "[STAGE 3.1] Filter manifest -> Data_additional rows only"
$PY - <<PYEOF
import pandas as pd
df = pd.read_csv("$MANIFEST_FULL")
new = df[df["source_root"] == "Data_additional"].copy()
print(f"  filtered {len(df)} -> {len(new)} rows in {new['condition_class_name'].nunique()} classes")
new.to_csv("$FILTER_MANIFEST", index=False)
PYEOF

echo "[STAGE 3.2] Tile new fields"
$PY Code/Training/Model_Training_Treated_Condition/prepare_treated_segmentation_tiles.py \
  --manifest_csv "$FILTER_MANIFEST" \
  --out_dir "$TILES_DIR" \
  --rows 3 \
  --cols 3 \
  --mode crop \
  --out_ext tif \
  --workers 8

echo "[STAGE 3.3] Omnipose on new tiles"
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

echo "[STAGE 3.4] Aggregate seg_qc columns for new rows"
$PY Code/Training/Model_Training_Treated_Condition/build_treated_segmentation_qc_manifest.py \
  --manifest_csv "$FILTER_MANIFEST" \
  --tile_image_map_csv "$TILES_DIR/tile_image_map.csv" \
  --seg_metadata_csv "$SEG_OUT/metadata.csv" \
  --out_dir "$QC_OUT_NEW" \
  --min_pass_tiles 3 \
  --min_pass_fraction 0.34 \
  --min_total_cells 6 \
  --max_large_cell_tiles 0 \
  --max_artifact_tiles 0

echo "[STAGE 3.5] Merge with existing 13-class seg_qc manifest -> 15-class combined"
$PY - <<PYEOF
import sys
import pandas as pd
sys.path.insert(0, "/new-stg/home/zekai/Forcast_Model/Code/Training/Model_Training_Treated_Condition")
from build_treated_condition_manifest import CLASS_TO_ID

existing = pd.read_csv("$EXISTING_QC_MANIFEST")
new = pd.read_csv("$QC_OUT_NEW/manifest_with_segmentation_qc.csv")

# Align columns: take the union; missing cols become empty
all_cols = list(dict.fromkeys(list(existing.columns) + [c for c in new.columns if c not in existing.columns]))
existing = existing.reindex(columns=all_cols)
new = new.reindex(columns=all_cols)

combined = pd.concat([existing, new], ignore_index=True)

# CRITICAL: re-derive class_id from condition_class_name using the new 15-class
# CLASS_TO_ID so old (13-class) and new rows share the same numbering.
combined["class_id"] = combined["condition_class_name"].map(CLASS_TO_ID).astype(int)
if "condition_class_id" in combined.columns:
    combined["condition_class_id"] = combined["class_id"]
combined["class_name"] = combined["condition_class_name"]

assert combined.groupby("class_id")["condition_class_name"].nunique().max() == 1, \
    "class_id collision after remap"

print(f"  existing 13-class rows: {len(existing)}")
print(f"  new Data_additional rows: {len(new)}")
print(f"  combined 15-class rows: {len(combined)}")
print(f"  unique classes: {combined['condition_class_name'].nunique()} (after class_id remap)")
print()
print("Per-class retention (pass+ not dropped):")
keep_mask = (combined.get('drop_from_training_seg_qc', 0).fillna(0).astype(int) == 0) & \
            (combined.get('drop_from_training', 0).fillna(0).astype(int) == 0)
retention = combined.groupby('condition_class_name').apply(
    lambda g: (keep_mask.loc[g.index].mean() * 100)
).round(1)
print(retention.to_string())

combined.to_csv("$COMBINED_QC_CSV", index=False)
print(f"\nWrote combined manifest: $COMBINED_QC_CSV")
PYEOF

echo "[DONE] 15-class seg_qc manifest ready at $COMBINED_QC_CSV"
