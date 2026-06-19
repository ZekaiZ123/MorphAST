#!/usr/bin/env python3
"""Aggregate fresh Omnipose tile QC back to treated image-level manifest rows."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


_TILE_SUFFIX_RE = re.compile(r"_r\d+_c\d+$")


def infer_tiled_image_id(sample_id: str) -> str:
    return _TILE_SUFFIX_RE.sub("", str(sample_id or "").strip())


def normalize_reason(value) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def count_reason(prefix: str, reasons: pd.Series) -> int:
    return int(reasons.fillna("").astype(str).str.startswith(prefix).sum())


def make_image_qc_summary(tile_sub: pd.DataFrame, args) -> dict:
    reasons = tile_sub["exclude_reason"].fillna("").astype(str)
    n_tiles = int(len(tile_sub))
    n_pass = int((reasons == "").sum())
    n_fail = n_tiles - n_pass
    pass_fraction = float(n_pass / n_tiles) if n_tiles else 0.0
    total_cells = int(tile_sub["num_cells"].fillna(0).astype(float).sum())
    max_cells = int(tile_sub["num_cells"].fillna(0).astype(float).max()) if n_tiles else 0
    n_low_cells = count_reason("low_cells:", reasons)
    n_large_cell = count_reason("large_cell:", reasons)
    n_artifact = count_reason("artifact_area:", reasons)

    qc_reasons = []
    if n_pass < args.min_pass_tiles:
        qc_reasons.append(f"few_pass_tiles:{n_pass}")
    if pass_fraction < args.min_pass_fraction:
        qc_reasons.append(f"low_pass_fraction:{pass_fraction:.3f}")
    if total_cells < args.min_total_cells:
        qc_reasons.append(f"low_total_cells:{total_cells}")
    if n_large_cell > args.max_large_cell_tiles:
        qc_reasons.append(f"large_cell_tiles:{n_large_cell}")
    if n_artifact > args.max_artifact_tiles:
        qc_reasons.append(f"artifact_tiles:{n_artifact}")

    return {
        "seg_qc_n_tiles": n_tiles,
        "seg_qc_n_pass_tiles": n_pass,
        "seg_qc_n_fail_tiles": n_fail,
        "seg_qc_pass_fraction": pass_fraction,
        "seg_qc_total_cells": total_cells,
        "seg_qc_max_cells_per_tile": max_cells,
        "seg_qc_low_cells_tiles": n_low_cells,
        "seg_qc_large_cell_tiles": n_large_cell,
        "seg_qc_artifact_tiles": n_artifact,
        "seg_qc_status": "pass" if not qc_reasons else "fail",
        "seg_qc_reason": ";".join(qc_reasons),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_csv", type=Path, required=True)
    ap.add_argument("--tile_image_map_csv", type=Path, required=True)
    ap.add_argument("--seg_metadata_csv", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--min_pass_tiles", type=int, default=3)
    ap.add_argument("--min_pass_fraction", type=float, default=0.34)
    ap.add_argument("--min_total_cells", type=int, default=6)
    ap.add_argument("--max_large_cell_tiles", type=int, default=0)
    ap.add_argument("--max_artifact_tiles", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest_csv).copy()
    tile_map = pd.read_csv(args.tile_image_map_csv).copy()
    seg_df = pd.read_csv(args.seg_metadata_csv, low_memory=False).copy()

    seg_df["tiled_image_id"] = seg_df["sample_id"].astype(str).apply(infer_tiled_image_id)
    seg_df["exclude_reason"] = seg_df.get("exclude_reason", pd.Series(index=seg_df.index, dtype=object)).apply(normalize_reason)

    grouped_rows = []
    for tiled_image_id, sub in seg_df.groupby("tiled_image_id", sort=False):
        grouped_rows.append({"tiled_image_id": tiled_image_id, **make_image_qc_summary(sub, args)})
    image_qc = pd.DataFrame(grouped_rows)

    merged = tile_map.merge(image_qc, on="tiled_image_id", how="left")
    defaults = {
        "seg_qc_n_tiles": 0,
        "seg_qc_n_pass_tiles": 0,
        "seg_qc_n_fail_tiles": 0,
        "seg_qc_pass_fraction": 0.0,
        "seg_qc_total_cells": 0,
        "seg_qc_max_cells_per_tile": 0,
        "seg_qc_low_cells_tiles": 0,
        "seg_qc_large_cell_tiles": 0,
        "seg_qc_artifact_tiles": 0,
        "seg_qc_status": "missing",
        "seg_qc_reason": "missing_segmentation",
    }
    for col, val in defaults.items():
        merged[col] = merged[col].fillna(val)

    out = manifest.merge(
        merged[
            [
                "image_uid",
                "tiled_image_id",
                "seg_qc_n_tiles",
                "seg_qc_n_pass_tiles",
                "seg_qc_n_fail_tiles",
                "seg_qc_pass_fraction",
                "seg_qc_total_cells",
                "seg_qc_max_cells_per_tile",
                "seg_qc_low_cells_tiles",
                "seg_qc_large_cell_tiles",
                "seg_qc_artifact_tiles",
                "seg_qc_status",
                "seg_qc_reason",
            ]
        ],
        on="image_uid",
        how="left",
    )
    out["drop_from_training_seg_qc"] = (out["seg_qc_status"] != "pass").astype(int)
    base_drop = pd.to_numeric(out.get("drop_from_training", 0), errors="coerce").fillna(0).astype(int)
    out["drop_from_training"] = np.maximum(base_drop.values, out["drop_from_training_seg_qc"].astype(int).values)

    out_csv = args.out_dir / "manifest_with_segmentation_qc.csv"
    out.to_csv(out_csv, index=False)

    retention = (
        out.groupby("class_name")["drop_from_training_seg_qc"]
        .agg(n_images="count", n_drop="sum")
        .reset_index()
    )
    retention["n_keep"] = retention["n_images"] - retention["n_drop"]
    retention["keep_rate"] = retention["n_keep"] / retention["n_images"]
    retention.to_csv(args.out_dir / "segmentation_qc_retention_by_class.csv", index=False)

    tp_retention = (
        out.groupby(["class_name", "timepoint_min"])["drop_from_training_seg_qc"]
        .agg(n_images="count", n_drop="sum")
        .reset_index()
    )
    tp_retention["n_keep"] = tp_retention["n_images"] - tp_retention["n_drop"]
    tp_retention["keep_rate"] = tp_retention["n_keep"] / tp_retention["n_images"]
    tp_retention.to_csv(args.out_dir / "segmentation_qc_retention_by_class_timepoint.csv", index=False)

    summary = {
        "manifest_csv": str(args.manifest_csv),
        "tile_image_map_csv": str(args.tile_image_map_csv),
        "seg_metadata_csv": str(args.seg_metadata_csv),
        "n_images": int(len(out)),
        "qc_status_counts": out["seg_qc_status"].value_counts(dropna=False).to_dict(),
        "n_drop_seg_qc": int(out["drop_from_training_seg_qc"].sum()),
        "n_keep_seg_qc": int((out["drop_from_training_seg_qc"] == 0).sum()),
        "n_drop_combined": int(out["drop_from_training"].sum()),
        "n_keep_combined": int((out["drop_from_training"] == 0).sum()),
        "reason_counts": out["seg_qc_reason"].value_counts(dropna=False).head(50).to_dict(),
        "policy": {
            "min_pass_tiles": int(args.min_pass_tiles),
            "min_pass_fraction": float(args.min_pass_fraction),
            "min_total_cells": int(args.min_total_cells),
            "max_large_cell_tiles": int(args.max_large_cell_tiles),
            "max_artifact_tiles": int(args.max_artifact_tiles),
        },
    }
    (args.out_dir / "segmentation_qc_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[DONE] Segmentation-QC manifest written to {out_csv}")


if __name__ == "__main__":
    main()
