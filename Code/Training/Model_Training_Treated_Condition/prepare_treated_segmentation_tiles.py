#!/usr/bin/env python3
"""Tile treated-manifest images into a fresh segmentation workspace."""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

import sys

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent.parent / "tile_dataset"))
from tile_dataset import _center_crop_to_multiple, _imread, _imsave, _pad_to_multiple, _tile_array


def slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return text.strip("_") or "item"


def make_tiled_image_id(idx: int, row: pd.Series) -> str:
    class_slug = slugify(row["class_name"])
    return f"{class_slug}__{int(row['timepoint_min']):03d}min__{idx:06d}"


def tile_one(row_dict: dict, output_root: Path, rows: int, cols: int, mode: str, out_ext: str) -> dict:
    row = pd.Series(row_dict)
    image_path = Path(str(row["image_path"]))
    arr_np = _imread(image_path)
    arr2 = _center_crop_to_multiple(arr_np, rows, cols) if mode == "crop" else _pad_to_multiple(arr_np, rows, cols)
    tiles = _tile_array(arr2, rows, cols)

    class_slug = slugify(row["class_name"])
    timepoint_slug = f"{int(row['timepoint_min']):03d}min"
    tiled_image_id = str(row["tiled_image_id"])
    tiles_dir = output_root / class_slug / timepoint_slug / f"{tiled_image_id}_tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    for idx, tile in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        out_path = tiles_dir / f"{tiled_image_id}_r{r}_c{c}.{out_ext}"
        _imsave(out_path, tile, out_ext)

    return {
        "image_uid": row["image_uid"],
        "tiled_image_id": tiled_image_id,
        "tiles_dir": str(tiles_dir),
        "n_tiles": len(tiles),
        "status": "ok",
        "error_message": "",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_csv", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--mode", choices=["crop", "pad"], default="crop")
    ap.add_argument("--out_ext", type=str, default="tif")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tiles_root = args.out_dir / "tiles"
    tiles_root.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest_csv).copy()
    manifest = manifest.reset_index(drop=True)
    manifest["tiled_image_id"] = [make_tiled_image_id(idx, row) for idx, row in manifest.iterrows()]

    rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [
            ex.submit(
                tile_one,
                row.to_dict(),
                tiles_root,
                args.rows,
                args.cols,
                args.mode,
                args.out_ext.lstrip("."),
            )
            for _, row in manifest.iterrows()
        ]
        for i, fut in enumerate(as_completed(futs), start=1):
            result = fut.result()
            rows.append(result)
            if i % 500 == 0 or i == len(futs):
                print(f"[TILING] {i}/{len(futs)}", flush=True)

    tile_map = manifest.merge(pd.DataFrame(rows), on=["image_uid", "tiled_image_id"], how="left")
    tile_map_csv = args.out_dir / "tile_image_map.csv"
    tile_map.to_csv(tile_map_csv, index=False)

    summary = {
        "manifest_csv": str(args.manifest_csv),
        "tiles_root": str(tiles_root),
        "tile_image_map_csv": str(tile_map_csv),
        "n_images": int(len(tile_map)),
        "n_ok": int((tile_map["status"] == "ok").sum()),
        "n_errors": int((tile_map["status"] != "ok").sum()),
    }
    (args.out_dir / "tiling_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[DONE] Tiled {len(tile_map)} treated images into {tiles_root}")
    print(f"  tile_map: {tile_map_csv}")


if __name__ == "__main__":
    main()
