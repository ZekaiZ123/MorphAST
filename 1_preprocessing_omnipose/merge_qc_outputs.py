#!/usr/bin/env python3
"""Merge old and new QC outputs (tile_image_map + metadata) into combined files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_tile_map", type=Path, required=True)
    ap.add_argument("--new_tile_map", type=Path, required=True)
    ap.add_argument("--old_metadata", type=Path, required=True)
    ap.add_argument("--new_metadata", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    old_tm = pd.read_csv(args.old_tile_map)
    new_tm = pd.read_csv(args.new_tile_map)
    combined_tm = pd.concat([old_tm, new_tm], ignore_index=True)
    tm_out = args.out_dir / "tile_image_map.csv"
    combined_tm.to_csv(tm_out, index=False)
    print(f"[DONE] tile_image_map: {len(old_tm)} + {len(new_tm)} = {len(combined_tm)} -> {tm_out}")

    old_meta = pd.read_csv(args.old_metadata, low_memory=False)
    new_meta = pd.read_csv(args.new_metadata, low_memory=False)
    combined_meta = pd.concat([old_meta, new_meta], ignore_index=True)
    meta_out = args.out_dir / "metadata.csv"
    combined_meta.to_csv(meta_out, index=False)
    print(f"[DONE] metadata: {len(old_meta)} + {len(new_meta)} = {len(combined_meta)} -> {meta_out}")


if __name__ == "__main__":
    main()
