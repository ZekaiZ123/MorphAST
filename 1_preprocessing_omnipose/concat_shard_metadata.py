#!/usr/bin/env python3
"""Concatenate shard metadata CSVs into a single metadata.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg_out_dir", type=Path, required=True,
                    help="Directory containing metadata_shard*.csv files")
    ap.add_argument("--n_shards", type=int, default=8)
    args = ap.parse_args()

    shard_files = sorted(args.seg_out_dir.glob("metadata_shard*.csv"))
    if not shard_files:
        raise SystemExit(f"No metadata_shard*.csv files found in {args.seg_out_dir}")

    print(f"[INFO] Found {len(shard_files)} shard files: {[f.name for f in shard_files]}")
    if len(shard_files) < args.n_shards:
        print(f"[WARN] Expected {args.n_shards} shards but found {len(shard_files)}")

    dfs = [pd.read_csv(f, low_memory=False) for f in shard_files]
    combined = pd.concat(dfs, ignore_index=True)
    combined.sort_values(
        by=["strain", "antibiotic", "mic", "timepoint_min", "sample_id"],
        key=lambda col: col.astype(str),
        inplace=True,
    )
    combined.reset_index(drop=True, inplace=True)

    out_path = args.seg_out_dir / "metadata.csv"
    combined.to_csv(out_path, index=False)
    print(f"[DONE] Combined {len(combined)} rows from {len(shard_files)} shards -> {out_path}")


if __name__ == "__main__":
    main()
