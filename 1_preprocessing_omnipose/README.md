# 1 — Preprocessing & QC with Omnipose

Tile each raw full-field micrograph into a 3 × 3 grid, run Omnipose
segmentation on every tile, derive per-tile cell counts and exclude flags,
and aggregate to a per-field QC summary. The output is a manifest CSV in
which each field carries `seg_qc_*` columns and a final
`drop_from_training_seg_qc` flag used to filter the training pool in
Sections 2 and 3.

## What this section does **not** do

The classifier never sees the tiles or the segmentation masks. Tiles exist
solely as the spatial granularity at which Omnipose runs; the classifier
in Sections 2 / 3 consumes the **full original field** at 224 × 224.

## Scripts

| Script | Role |
|---|---|
| `prepare_segmentation_tiles.py` | Reads a field-level manifest CSV, center-crops each field to a multiple of 3 in each axis, carves a 3 × 3 grid, writes 9 `.tif` tiles per field, and emits `tile_image_map.csv` (parent-field → 9 tiles). |
| `run_omnipose_segmentation.py` | Walks a tile directory, runs Omnipose on each tile, derives per-cell area / clumping / artifact statistics via a downstream `cell_stats` pass, and writes `metadata.csv` (one row per tile). |
| `build_segmentation_qc_manifest.py` | Joins the manifest with `tile_image_map.csv` and `metadata.csv`, computes per-field `seg_qc_n_tiles`, `seg_qc_n_pass_tiles`, `seg_qc_pass_fraction`, `seg_qc_total_cells`, and exclude-reason counts. Decides field-level `seg_qc_status` and `drop_from_training_seg_qc` from a thresholded combination. |
| `concat_shard_metadata.py` | Concatenates per-shard Omnipose `metadata.csv` files when the segmentation has been sharded across multiple Slurm jobs. |
| `merge_qc_outputs.py` | Merges a freshly-computed QC table with a previously published QC manifest (preserves the published rows; appends new rows). |
| `run_0min_seg_qc.sbatch` | SBATCH template for the 0-min (5-class) preprocessing run. |
| `run_treated_seg_qc.sbatch` | SBATCH template for the treated-condition preprocessing run. |
| `run_15class_seg_qc.sbatch` | SBATCH template for the 15-class build (includes the class-ID remap step in the merge). |

## QC thresholds (paper)

A field is retained if and only if all five conditions hold:

| Threshold | Value |
|---|---|
| `seg_qc_n_pass_tiles ≥ min_pass_tiles` | 3 |
| `seg_qc_pass_fraction ≥ min_pass_fraction` | 0.34 |
| `seg_qc_total_cells ≥ min_total_cells` | 6 |
| `seg_qc_large_cell_tiles ≤ max_large_cell_tiles` | 0 |
| `seg_qc_artifact_tiles ≤ max_artifact_tiles` | 0 |

Per-tile exclude flags emitted by `run_omnipose_segmentation.py`:

| Flag | Triggered when |
|---|---|
| `low_cells` | Fewer than 2 detected cells (`--min_cells`) |
| `large_cells` | Any detected cell larger than 30 µm² (`--max_cell_area_um2`) |
| `artifact` | Any region larger than 100 µm² (`--max_artifact_area_um2`) |

Omnipose defaults used in the paper: `--mask_threshold -2.4 --flow_threshold 0.5 --min_size 20 --pixel_size_um 0.25641`.

## Canonical workflow

```bash
# 1. Tile the manifest's fields into 3×3 grids
python prepare_segmentation_tiles.py \
  --manifest_csv path/to/field_manifest.csv \
  --out_dir       output_root/tiling \
  --rows 3 --cols 3 --mode crop --out_ext tif --workers 8

# 2. Run Omnipose on every tile
python run_omnipose_segmentation.py \
  --tiles_root       output_root/tiling/tiles \
  --tile_output_root output_root/seg_outputs \
  --gpu 0 \
  --mask_threshold -2.4 --flow_threshold 0.5 --min_size 20 \
  --pixel_size_um 0.25641 --min_cells 2 \
  --max_cell_area_um2 30 --max_artifact_area_um2 100

# 3. Aggregate per-tile metadata into a per-field QC manifest
python build_segmentation_qc_manifest.py \
  --manifest_csv         path/to/field_manifest.csv \
  --tile_image_map_csv   output_root/tiling/tile_image_map.csv \
  --seg_metadata_csv     output_root/seg_outputs/metadata.csv \
  --out_dir              output_root/qc_manifest \
  --min_pass_tiles 3 --min_pass_fraction 0.34 \
  --min_total_cells 6 \
  --max_large_cell_tiles 0 --max_artifact_tiles 0
```

The cluster recipe is in the `run_*_seg_qc.sbatch` templates.

## Outputs

```
output_root/
├── tiling/
│   ├── tiles/                  # 9 .tif tiles per field
│   └── tile_image_map.csv      # parent ↔ tiles mapping
├── seg_outputs/
│   └── metadata.csv            # one row per tile (num_cells, exclude_reason, etc.)
└── qc_manifest/
    ├── manifest_with_segmentation_qc.csv   ← training input for Sections 2 and 3
    ├── segmentation_qc_summary.json
    └── segmentation_qc_retention_by_class*.csv
```
