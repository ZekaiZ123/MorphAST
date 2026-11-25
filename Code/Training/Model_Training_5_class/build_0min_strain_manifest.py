#!/usr/bin/env python3
"""Build a canonical 0 min five-class strain manifest from raw images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CLASS_ORDER = [
    "E.coli5-alpha",
    "E.coli5-alphaKanR",
    "E.coli5-alphaAmpR",
    "AIEC",
    "UPEC",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_ORDER)}
ALIAS_MAP = {
    "Strain_C2987H": "E.coli5-alpha",
    "Strain_E.coli5-alpha": "E.coli5-alpha",
    "Strain_PET67": "E.coli5-alphaKanR",
    "Strain_E.coli5-alphaKanR": "E.coli5-alphaKanR",
    "Strain_PQXC74": "E.coli5-alphaAmpR",
    "Strain_E.coli5-alphaAmpR": "E.coli5-alphaAmpR",
    "Strain_AIEC": "AIEC",
    "Strain_UPEC": "UPEC",
}
IMAGE_SUFFIXES = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}


def parse_folder_metadata(folder_name: str) -> tuple[str, str, str]:
    strain_token, rest = folder_name.split("_Antibiotic_", 1)
    antibiotic_token, susc_token = rest.split("_Susc_", 1)
    canonical = ALIAS_MAP.get(strain_token)
    if canonical is None:
        raise ValueError(f"Unknown strain alias in folder {folder_name!r}")
    return canonical, antibiotic_token, susc_token


def build_rows(root: Path) -> list[dict]:
    rows = []
    dataset_root = root / "New Data + 0_min(untreated control)"
    if not dataset_root.exists():
        raise SystemExit(f"Expected dataset root not found: {dataset_root}")

    for folder in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        timepoint_dir = folder / "0_min"
        if not timepoint_dir.exists():
            continue

        canonical, antibiotic_token, susc_token = parse_folder_metadata(folder.name)
        for image_path in sorted(
            p for p in timepoint_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        ):
            rel_path = image_path.relative_to(dataset_root)
            image_uid = rel_path.parent.as_posix() + "::" + image_path.stem
            rows.append(
                {
                    "image_path": str(image_path.resolve()),
                    "image_uid": image_uid,
                    "group_id": image_uid,
                    "canonical_strain_id": canonical,
                    "class_name": canonical,
                    "class_id": CLASS_TO_ID[canonical],
                    "source_set": "internal_0min_new_root",
                    "source_folder": folder.name,
                    "antibiotic_context": antibiotic_token,
                    "susceptibility_text": susc_token,
                    "image_ext": image_path.suffix.lower(),
                    "timepoint_min": 0,
                    "annotation_status": "unreviewed",
                    "empty_label": "",
                    "empty_prob": "",
                    "drop_from_training": 0,
                }
            )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_root",
        type=Path,
        default=Path("/new-stg/home/zekai/Forcast_Model/Data/New Data_and_0_min(untreated control)"),
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/new-stg/home/zekai/Forcast_Model/Data/Benchmark_0min_5class"),
    )
    args = ap.parse_args()

    rows = build_rows(args.data_root)
    if not rows:
        raise SystemExit("No 0 min images discovered in the trusted new-root dataset.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(rows).sort_values(["class_id", "source_folder", "image_uid"]).reset_index(drop=True)
    manifest_csv = args.out_dir / "manifest_0min_fiveclass.csv"
    manifest_json = args.out_dir / "manifest_0min_fiveclass.json"
    summary_json = args.out_dir / "manifest_0min_fiveclass_summary.json"

    manifest.to_csv(manifest_csv, index=False)
    manifest.to_json(manifest_json, orient="records", indent=2)

    summary = {
        "n_images": int(len(manifest)),
        "class_counts": manifest["class_name"].value_counts().sort_index().to_dict(),
        "source_folder_counts": manifest["source_folder"].value_counts().sort_index().to_dict(),
        "image_ext_counts": manifest["image_ext"].value_counts().sort_index().to_dict(),
    }
    summary_json.write_text(json.dumps(summary, indent=2))

    print(f"[DONE] Wrote {len(manifest)} rows")
    print(f"  manifest_csv : {manifest_csv}")
    print(f"  manifest_json: {manifest_json}")
    print(f"  summary_json : {summary_json}")


if __name__ == "__main__":
    main()
