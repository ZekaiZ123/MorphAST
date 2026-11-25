#!/usr/bin/env python3
"""Build a canonical treated-image manifest for strain+treatment classification."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


IMAGE_SUFFIXES = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
STRAIN_MAP = {
    "Strain_C2987H": "E.coli5-alpha",
    "Strain_E.coli5-alpha": "E.coli5-alpha",
    "Strain_PET67": "E.coli5-alphaKanR",
    "Strain_E.coli5-alphaKanR": "E.coli5-alphaKanR",
    "Strain_PQXC74": "E.coli5-alphaAmpR",
    "Strain_E.coli5-alphaAmpR": "E.coli5-alphaAmpR",
    "Strain_AIEC": "AIEC",
    "Strain_UPEC": "UPEC",
}
TREATMENT_MAP = {
    "Ampicillin": "Ampicillin",
    "Ampicllin": "Ampicillin",
    "CND": "CND",
    "Kan": "Kanamycin",
    "Kana": "Kanamycin",
    "Kanamycin": "Kanamycin",
}
CLASS_ORDER = [
    "AIEC + Ampicillin",
    "AIEC + CND",
    "AIEC + Kanamycin",
    "E.coli5-alpha + Ampicillin",
    "E.coli5-alpha + CND",
    "E.coli5-alpha + Kanamycin",
    "E.coli5-alphaAmpR + Ampicillin",
    "E.coli5-alphaAmpR + CND",
    "E.coli5-alphaAmpR + Kanamycin",
    "E.coli5-alphaKanR + Ampicillin",
    "E.coli5-alphaKanR + CND",
    "E.coli5-alphaKanR + Kanamycin",
    "UPEC + Ampicillin",
    "UPEC + CND",
    "UPEC + Kanamycin",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_ORDER)}


def parse_dataset_a_folder(folder_name: str) -> tuple[str, str, str]:
    strain_token, rest = folder_name.split("_Label_", 1)
    susc_token, antibiotic_token = rest.split("_Antibiotic_", 1)
    return STRAIN_MAP[strain_token], TREATMENT_MAP[antibiotic_token], susc_token


def parse_new_root_folder(folder_name: str) -> tuple[str, str, str]:
    strain_token, rest = folder_name.split("_Antibiotic_", 1)
    antibiotic_token, susc_token = rest.split("_Susc_", 1)
    return STRAIN_MAP[strain_token], TREATMENT_MAP[antibiotic_token], susc_token


def normalize_stem(stem: str, timepoint_min: int) -> str:
    """Normalize likely field identities across timepoints without collapsing to channel-only IDs."""
    text = re.sub(r"\s+\(\d+\)$", "", stem)

    # Unambiguous timepoint-coded prefixes such as `5_Min_00001_CH4` or `0_Min_00001_CH4`.
    text = re.sub(r"(?i)^[A-Za-z]?\d+_min_", "", text, count=1)
    text = re.sub(r"(?i)^[A-Za-z]?\d+min_", "", text, count=1)

    # Bare numeric prefixes are removed only when they match the folder timepoint.
    text = re.sub(rf"(?i)^[A-Za-z]?{int(timepoint_min)}_", "", text, count=1)

    return text


def iter_image_files(folder: Path):
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        if path.name.startswith("._"):
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        yield path


def add_rows_from_condition_folder(
    rows: list[dict],
    folder: Path,
    *,
    source_root: str,
    parse_fn,
) -> None:
    canonical_strain, canonical_treatment, susceptibility_text = parse_fn(folder.name)
    condition_class_name = f"{canonical_strain} + {canonical_treatment}"
    class_id = CLASS_TO_ID[condition_class_name]

    for tp_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        if tp_dir.name == "0_min":
            continue
        timepoint_min = int(tp_dir.name.split("_")[0])
        for image_path in iter_image_files(tp_dir):
            image_uid = f"{source_root}::{folder.name}::{tp_dir.name}::{image_path.name}"
            field_identity = normalize_stem(image_path.stem, timepoint_min=timepoint_min)
            field_group_id = f"{condition_class_name}::{field_identity}"
            rows.append(
                {
                    "image_path": str(image_path.resolve()),
                    "image_uid": image_uid,
                    "group_id": field_group_id,
                    "field_group_id": field_group_id,
                    "canonical_strain": canonical_strain,
                    "canonical_treatment": canonical_treatment,
                    "susceptibility_text": susceptibility_text,
                    "condition_class_name": condition_class_name,
                    "condition_class_id": class_id,
                    "class_name": condition_class_name,
                    "class_id": class_id,
                    "source_root": source_root,
                    "source_folder": folder.name,
                    "timepoint_min": timepoint_min,
                    "image_ext": image_path.suffix.lower(),
                    "bootstrap_empty_prob": "",
                    "bootstrap_drop_candidate": 0,
                    "empty_prob": "",
                    "drop_from_training": 0,
                }
            )


def build_rows(dataset_a_root: Path, new_root: Path, additional_root: Path | None = None) -> list[dict]:
    rows: list[dict] = []
    for folder in sorted(p for p in dataset_a_root.iterdir() if p.is_dir()):
        add_rows_from_condition_folder(rows, folder, source_root="Dataset_A", parse_fn=parse_dataset_a_folder)
    for folder in sorted(p for p in new_root.iterdir() if p.is_dir()):
        add_rows_from_condition_folder(rows, folder, source_root="New_Data_0min_root", parse_fn=parse_new_root_folder)
    if additional_root is not None:
        for folder in sorted(p for p in additional_root.iterdir() if p.is_dir()):
            add_rows_from_condition_folder(rows, folder, source_root="Data_additional", parse_fn=parse_new_root_folder)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset_a_root",
        type=Path,
        default=Path("/new-stg/home/zekai/Forcast_Model/Data/Dataset_A"),
    )
    ap.add_argument(
        "--new_root",
        type=Path,
        default=Path("/new-stg/home/zekai/Forcast_Model/Data/New Data_and_0_min(untreated control)/New Data + 0_min(untreated control)"),
    )
    ap.add_argument(
        "--additional_root",
        type=Path,
        default=None,
        help="Optional 3rd image root for late-arriving strain x treatment fields (same folder convention as --new_root).",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/new-stg/home/zekai/Forcast_Model/Data/Benchmark_Treated_Condition"),
    )
    args = ap.parse_args()

    rows = build_rows(args.dataset_a_root, args.new_root, args.additional_root)
    if not rows:
        raise SystemExit("No treated images discovered.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(rows).sort_values(["class_id", "timepoint_min", "image_uid"]).reset_index(drop=True)
    manifest_csv = args.out_dir / "manifest_treated_condition.csv"
    manifest_json = args.out_dir / "manifest_treated_condition.json"
    summary_json = args.out_dir / "manifest_treated_condition_summary.json"

    manifest.to_csv(manifest_csv, index=False)
    manifest.to_json(manifest_json, orient="records", indent=2)

    class_timepoint_counts = defaultdict(dict)
    grouped = manifest.groupby(["condition_class_name", "timepoint_min"]).size()
    for (condition_name, timepoint_min), n_rows in grouped.items():
        class_timepoint_counts[condition_name][str(int(timepoint_min))] = int(n_rows)

    summary = {
        "n_images": int(len(manifest)),
        "n_classes": int(manifest["condition_class_name"].nunique()),
        "class_counts": manifest["condition_class_name"].value_counts().sort_index().to_dict(),
        "timepoint_counts": manifest["timepoint_min"].value_counts().sort_index().to_dict(),
        "source_root_counts": manifest["source_root"].value_counts().sort_index().to_dict(),
        "class_timepoint_counts": dict(sorted(class_timepoint_counts.items())),
    }
    summary_json.write_text(json.dumps(summary, indent=2))

    print(f"[DONE] Wrote {len(manifest)} treated rows")
    print(f"  manifest_csv : {manifest_csv}")
    print(f"  manifest_json: {manifest_json}")
    print(f"  summary_json : {summary_json}")


if __name__ == "__main__":
    main()
