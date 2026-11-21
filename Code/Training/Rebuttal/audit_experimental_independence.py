#!/usr/bin/env python3
"""Audit provider identifiers and preparation overlap without changing data or folds.

SCID is deliberately excluded. A unique composite metadata match is not a
checksum-verified identity, and recorded identifiers are not proof of biological
independence. Missing identifiers are never counted as another replicate.
"""

from __future__ import annotations

import argparse
import json
import posixpath
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image

from validation import sha256, write_json

ROOT = Path(__file__).resolve().parents[3]
UNKNOWN = "UNKNOWN"
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
ALIASES = {"5-alpha": "E.coli5-alpha", "5-alphaAmpR": "E.coli5-alphaAmpR",
           "5-alphaKanR": "E.coli5-alphaKanR", "AIEC": "AIEC", "UPEC": "UPEC"}
KEYS = ["strain", "treatment", "timepoint", "filename"]
IDENTIFIERS = {
    "culture_id": ("Culture ID", r"CultureID_([^/]+)"),
    "preparation_id": ("PreperationID", r"PreparationID_([^/]+)"),
    "slide_id": ("Slide ID", r"SlideID_([^/]+)"),
    "acquisition_session_id": ("Acquisition Session", r"AcquisitionID_([^/_]+)(?:_Date_[^/]+)?"),
}
MATCHED = "unique_composite_and_dimensions"
PROFILES = {
    "untreated_5class": ("Benchmark_0min_5class", "segmentation_qc", "reviewer1_20261118", "untreated"),
    "treated_13class": ("Benchmark_Treated_Condition_13class", "segmentation_qc_combined", "reviewer1_20261118", "treated"),
    "treated_15class": ("Benchmark_Treated_Condition_15class", "segmentation_qc_combined", "reviewer1_treated15_20261119", "treated15"),
}


def clean(value):
    if pd.isna(value) or str(value).strip().upper() in {"", "NA", "N/A", "NAN", "NONE", "UNKNOWN"}:
        return UNKNOWN
    return str(value).strip()


def read_workbook(path):
    """Preserve sheet names, Excel row numbers and sparse cell coordinates."""
    with ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared = ["".join(t.text or "" for t in n.findall(".//m:t", NS))
                      for n in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        relationships = {r.attrib["Id"]: r.attrib["Target"] for r in
                         ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))}
        tables = {}
        for sheet in ET.fromstring(archive.read("xl/workbook.xml")).findall("m:sheets/m:sheet", NS):
            rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            member = posixpath.normpath(posixpath.join("xl", relationships[rid])).lstrip("/")
            if relationships[rid].startswith("/"):
                member = relationships[rid].lstrip("/")
            rows = []
            with archive.open(member) as stream:
                for _, node in ET.iterparse(stream, events=("end",)):
                    if node.tag != "{" + NS["m"] + "}row":
                        continue
                    values = {"excel_row": int(node.attrib["r"])}
                    for cell in node.findall("m:c", NS):
                        column = re.sub(r"\d", "", cell.attrib["r"])
                        v = cell.find("m:v", NS)
                        value = v.text if v is not None else "".join(t.text or "" for t in cell.findall(".//m:t", NS))
                        values[column] = shared[int(value)] if cell.attrib.get("t") == "s" and value else value or ""
                    rows.append(values)
                    node.clear()
            tables[sheet.attrib["name"]] = pd.DataFrame(rows).fillna("")
    raw = tables["Image Metadata"]
    header = raw.iloc[0].to_dict()
    columns = [c for c in raw if c != "excel_row"]
    if len({header[c] for c in columns}) != len(columns):
        raise ValueError("Duplicate image-metadata headers")
    images = raw.iloc[1:].rename(columns={c: header[c] for c in columns}).reset_index(drop=True)
    images = images.loc[images.Filename.ne("")].copy()
    return images, tables


def normalize_metadata(images):
    rows = []
    for record in images.to_dict("records"):
        relative = str(record["Relative Path"]).replace("\\", "/")
        parts = [p for p in relative.split("/") if p]
        conflicts = []
        strain = ALIASES.get(record["Strain"], UNKNOWN)
        if len(parts) < 3 or ALIASES.get(parts[0], UNKNOWN) != strain or parts[1] != record["Treatment"]:
            conflicts.append("path_condition_disagreement")
        if parts[-1] != record["Filename"]:
            conflicts.append("path_filename_disagreement")
        time = re.fullmatch(r"(\d+)_min", str(record["Timepoint"]))
        minute = str(int(time[1])) if time else UNKNOWN
        path_times = re.findall(r"(?:^|/)(\d+)_min(?=/)", relative)
        if path_times and (len(set(path_times)) != 1 or str(int(path_times[0])) != minute):
            conflicts.append("path_timepoint_disagreement")
        row = {"metadata_row": int(record["excel_row"]), "provider_relative_path": relative,
               "provider_full_path": record["Full Path"], "strain": strain,
               "treatment": record["Treatment"], "timepoint": minute, "filename": record["Filename"],
               "metadata_width": int(record["Image Width (px)"]),
               "metadata_height": int(record["Image Height (px)"]),
               "microscope": clean(record.get("Microscope", "")), "optical_modality": UNKNOWN}
        for name, (column, pattern) in IDENTIFIERS.items():
            values = re.findall(r"(?:^|/)" + pattern + r"(?=/)", relative)
            recorded = clean(record.get(column, ""))
            from_path = values[0] if len(values) == 1 else UNKNOWN
            if len(values) > 1 or (recorded != UNKNOWN and recorded != from_path):
                conflicts.append(name + "_path_disagreement")
            row[name] = recorded if recorded != UNKNOWN else from_path
        serial = clean(record.get("Acquisition Date", ""))
        row["acquisition_date"] = UNKNOWN
        if serial != UNKNOWN:
            try:
                row["acquisition_date"] = (datetime(1899, 12, 30) + timedelta(days=float(serial))).date().isoformat()
            except (ValueError, OverflowError):
                conflicts.append("invalid_acquisition_date")
        date_match = re.search(r"(?:^|/)AcquisitionID_[^/]+_Date_(\d{6})(?=/)", relative)
        if date_match:
            try:
                path_date = datetime.strptime(date_match[1], "%m%d%y").date().isoformat()
                if row["acquisition_date"] not in {UNKNOWN, path_date}:
                    conflicts.append("path_date_disagreement")
                if row["acquisition_date"] == UNKNOWN:
                    row["acquisition_date"] = path_date
            except ValueError:
                conflicts.append("invalid_path_date")
        row["metadata_conflicts"] = "|".join(conflicts)
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.metadata_row.duplicated().any():
        raise ValueError("Duplicate Excel row numbers")
    return result


def normalize_manifest(frame, cohort):
    df = frame.copy()
    if df.image_path.duplicated().any():
        raise ValueError(f"Duplicate paths in {cohort}")
    df["strain"] = df["canonical_strain_id" if cohort == "untreated_5class" else "canonical_strain"]
    df["treatment"] = "Control" if cohort == "untreated_5class" else df.canonical_treatment
    df["timepoint"] = df.timepoint_min.astype(int).astype(str)
    df["filename"] = df.image_path.map(lambda p: Path(p).name)
    df["retained"] = df.drop_from_training.fillna(0).astype(int).eq(0)
    df["cohort"] = cohort
    return df


def image_dimensions(path):
    try:
        with Image.open(path) as image:
            width, height = image.size
        return {"image_path": path, "local_width": width, "local_height": height, "header_error": ""}
    except Exception as exc:
        return {"image_path": path, "local_width": -1, "local_height": -1,
                "header_error": f"{type(exc).__name__}: {exc}"}


def match_metadata(local, metadata, dimensions):
    """Exact keys only: no fuzzy names, treatment changes, or inferred timepoints."""
    counts = metadata.groupby(KEYS, dropna=False).size().rename("n_metadata_candidates").reset_index()
    unique = metadata.loc[~metadata.duplicated(KEYS, keep=False)]
    out = local.merge(counts, on=KEYS, how="left", validate="many_to_one")
    out["n_metadata_candidates"] = out.n_metadata_candidates.fillna(0).astype(int)
    out = out.merge(unique, on=KEYS, how="left", validate="many_to_one")
    out = out.merge(dimensions, on="image_path", how="left", validate="one_to_one")
    out["n_local_key_candidates"] = out.groupby(KEYS, dropna=False).image_path.transform("size")
    out["mapping_status"] = MATCHED
    out.loc[(out.local_width != out.metadata_width) | (out.local_height != out.metadata_height), "mapping_status"] = "dimension_mismatch"
    out.loc[out.header_error.fillna("").ne(""), "mapping_status"] = "image_header_error"
    out.loc[out.metadata_conflicts.fillna("").ne(""), "mapping_status"] = "metadata_conflict"
    out.loc[out.n_local_key_candidates.gt(1), "mapping_status"] = "ambiguous_local_key"
    out.loc[out.n_metadata_candidates.gt(1), "mapping_status"] = "ambiguous_workbook_key"
    out.loc[out.n_metadata_candidates.eq(0), "mapping_status"] = "no_exact_composite_match"
    accepted = out.mapping_status.eq(MATCHED)
    for name in IDENTIFIERS:
        out["candidate_" + name] = out[name].map(clean)
        out[name] = out[name].map(clean).where(accepted, UNKNOWN)
    out["biological_replicate_id"] = UNKNOWN
    out["identity_evidence"] = np.where(accepted, "composite_metadata_association_not_checksum_verified", "unresolved")
    return out


def identifier_counts(df, dimensions):
    rows = []
    for key, group in df.groupby(dimensions, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(dimensions, key))
        row["n_images"] = len(group)
        if "mapping_status" in group:
            row["n_metadata_matches"] = int(group.mapping_status.eq(MATCHED).sum())
        for name in IDENTIFIERS:
            values = sorted(set(group[name].map(clean)) - {UNKNOWN})
            row["n_recorded_" + name] = len(values)
            row[name + "s"] = "|".join(values)
            row["n_unknown_" + name] = int(group[name].map(clean).eq(UNKNOWN).sum())
        row["independent_biological_replicates"] = "NOT_ESTABLISHED"
        rows.append(row)
    return pd.DataFrame(rows)


def hierarchy_conflicts(metadata):
    rows = []
    checks = {"culture_id": ["strain"], "preparation_id": ["culture_id", "strain", "treatment"],
              "slide_id": ["preparation_id", "culture_id", "strain", "treatment"],
              "acquisition_session_id": ["acquisition_date"]}
    for child, parents in checks.items():
        for value, group in metadata.loc[metadata[child].ne(UNKNOWN)].groupby(child):
            for parent in parents:
                values = sorted(set(group[parent]) - {UNKNOWN})
                if len(values) > 1:
                    rows.append({"identifier_type": child, "identifier": value,
                                 "inconsistent_parent": parent, "parent_values": "|".join(values),
                                 "metadata_rows": "|".join(group.metadata_row.astype(str))})
    return pd.DataFrame(rows, columns=["identifier_type", "identifier", "inconsistent_parent", "parent_values", "metadata_rows"])


def overlap_table(df, role_column, fold, protocol, comparisons):
    rows = []
    for name in ["group_id", *IDENTIFIERS]:
        for left_roles, right_role, comparison in comparisons:
            left = df.loc[df[role_column].isin(left_roles)]
            right = df.loc[df[role_column].eq(right_role)]
            left_known = left.loc[left[name].map(clean).ne(UNKNOWN)]
            right_known = right.loc[right[name].map(clean).ne(UNKNOWN)]
            shared = set(left_known[name]) & set(right_known[name])
            n_shared = int(right_known[name].isin(shared).sum())
            rows.append({"protocol": protocol, "fold": fold, "level": name, "comparison": comparison,
                         "n_left_images": len(left), "n_right_images": len(right),
                         "n_right_known": len(right_known), "n_right_unknown": len(right) - len(right_known),
                         "n_left_identifiers": left_known[name].nunique(),
                         "n_right_identifiers": right_known[name].nunique(),
                         "n_shared_identifiers": len(shared), "n_right_images_shared": n_shared,
                         "fraction_known_right_shared": n_shared / len(right_known) if len(right_known) else None,
                         "shared_identifiers": "|".join(sorted(shared))})
    return pd.DataFrame(rows)


def group_fold_table(df):
    rows = []
    for name in IDENTIFIERS:
        for value, group in df.loc[df[name].ne(UNKNOWN)].groupby(name):
            folds = group.outer_fold.value_counts()
            rows.append({"level": name, "identifier": value, "n_images": len(group),
                         "n_outer_folds": len(folds), "spans_outer_folds": len(folds) > 1,
                         "class_names": "|".join(sorted(group.class_name.unique())),
                         **{f"fold_{i}_images": int(folds.get(i, 0)) for i in range(1, 6)}})
    return pd.DataFrame(rows)


def split_feasibility(df):
    rows = []
    for class_name, group in df.groupby("class_name"):
        for level in IDENTIFIERS:
            ids = set(group[level]) - {UNKNOWN}
            rows.append({"class_name": class_name, "level": level, "n_images": len(group),
                         "n_unknown_images": int(group[level].eq(UNKNOWN).sum()),
                         "n_recorded_groups": len(ids), "identifiers": "|".join(sorted(ids)),
                         "two_partition_class_coverage_possible_on_known_subset": len(ids) >= 2,
                         "five_fold_class_coverage_necessary_count_met": len(ids) >= 5,
                         "interpretation": "necessary_count_only_not_a_validated_split_or_biological_replicate_count"})
    return pd.DataFrame(rows)


def audit_folds(root, cohort, spec, qc, mapping, out, evidence):
    benchmark, _, run_name, task = spec
    run = root / "Model/Rebuttal_CV" / run_name
    config_path = run / "run.json"
    config = json.loads(config_path.read_text())
    settings = config["tasks"][task]
    manifest_path = run / task / "manifest.csv"
    if sha256(manifest_path) != settings["manifest_sha256"]:
        raise ValueError("Frozen population hash mismatch")
    manifest = pd.read_csv(manifest_path)
    kept = qc.loc[qc.retained].copy()
    if set(kept.image_path) != set(manifest.image_path):
        raise ValueError(f"QC population differs from frozen run: {cohort}")
    check = kept.set_index("image_path").loc[manifest.image_path]
    if check.class_id.tolist() != manifest.class_id.tolist() or check.group_id.tolist() != manifest.group_id.tolist():
        raise ValueError("Frozen labels/groups differ from QC manifest")
    joined = kept.merge(mapping.drop(columns=KEYS), on="image_path", validate="one_to_one")
    evidence.extend([config_path, manifest_path])
    splits, outer = {}, {}
    for fold in range(1, 6):
        path = run / task / "splits" / f"fold_{fold}.csv"
        if sha256(path) != settings["splits"][str(fold)]:
            raise ValueError("Frozen split hash mismatch")
        split = pd.read_csv(path)
        if split.image_path.tolist() != manifest.image_path.tolist():
            raise ValueError("Frozen split order mismatch")
        evidence.append(path)
        splits[fold] = split
        for image in split.loc[split.role.eq("test"), "image_path"]:
            if image in outer:
                raise ValueError("Repeated outer-test image")
            outer[image] = fold
    if set(outer) != set(manifest.image_path):
        raise ValueError("Incomplete outer coverage")
    membership_check = "frozen_reconstruction_from_original_manifest_order_seed_and_fold_sizes"
    prior_path = root / "Model" / (benchmark + "_QC") / "efficientnet_b0/crossval_predictions.csv"
    if prior_path.exists():
        prior = pd.read_csv(prior_path, usecols=["image_path", "fold"])
        if prior.image_path.duplicated().any() or dict(zip(prior.image_path, prior.fold)) != outer:
            raise ValueError("Original predictions disagree with frozen outer folds")
        evidence.append(prior_path)
        membership_check = "exact_match_to_original_saved_predictions"
    joined["outer_fold"] = joined.image_path.map(outer)
    joined.to_csv(out / "retained_images_with_identifiers_and_folds.csv", index=False)
    overlaps = []
    for fold, split in splits.items():
        df = joined.copy()
        df["role"] = np.where(df.outer_fold.eq(fold), "test", "train")
        overlaps.append(overlap_table(df, "role", fold, "original_cv", [(["train"], "test", "train_vs_heldout")]))
        df["role"] = df.image_path.map(split.set_index("image_path").role)
        overlaps.append(overlap_table(df, "role", fold, "internal_validation_rerun", [
            (["train"], "validation", "train_vs_validation"), (["train"], "test", "train_vs_test"),
            (["train", "validation"], "test", "development_vs_test")]))
    overlaps = pd.concat(overlaps, ignore_index=True)
    if overlaps.loc[overlaps.level.eq("group_id"), "n_shared_identifiers"].ne(0).any():
        raise ValueError("Unexpected original field-group leakage")
    overlaps.to_csv(out / "fold_overlap_counts.csv", index=False)
    groups = group_fold_table(joined)
    groups.to_csv(out / "identifier_fold_membership.csv", index=False)
    split_feasibility(joined).to_csv(out / "grouped_split_class_support.csv", index=False)
    identifier_counts(joined, ["class_name"]).to_csv(out / "retained_identifier_counts_by_condition.csv", index=False)
    identifier_counts(joined, ["class_name", "timepoint"]).to_csv(out / "retained_identifier_counts_by_condition_time.csv", index=False)
    return {"n_retained": len(joined), "n_matched": int(joined.mapping_status.eq(MATCHED).sum()),
            "mapping_status_counts": joined.mapping_status.value_counts().to_dict(),
            "legacy_membership_check": membership_check,
            "levels": {name: {"n_known_images": int(joined[name].ne(UNKNOWN).sum()),
                               "n_recorded_identifiers": int(groups.level.eq(name).sum()),
                               "n_identifiers_spanning_outer_folds": int((groups.level.eq(name) & groups.spans_outer_folds).sum()),
                               "n_known_images_exposed_to_same_identifier_in_other_outer_folds": int(
                                   overlaps.loc[overlaps.protocol.eq("original_cv") & overlaps.level.eq(name), "n_right_images_shared"].sum())}
                       for name in IDENTIFIERS}}


def render_report(summary, workbook_counts):
    lines = ["# Experimental independence audit", "",
             "SCID is excluded at the user's request. This audit does not change labels, images, folds, models or the manuscript.", "",
             "## Recorded identifiers versus biological replication", "",
             f"The provider workbook has {summary['n_workbook_images']:,} image-file rows. Its identifiers are parsed from folder names. "
             "A Culture ID is a recorded provenance label, not proof of an independently inoculated/grown replicate. "
             "Preparation, slide and acquisition-session counts must not be substituted for biological replicate counts.", "",
             "| Strain | Workbook images | Recorded cultures | Culture IDs | Preparations | Slides | Sessions |",
             "| --- | ---: | ---: | --- | ---: | ---: | ---: |"]
    for row in workbook_counts.to_dict("records"):
        lines.append(f"| {row['strain']} | {row['n_images']} | {row['n_recorded_culture_id']} | {row['culture_ids']} | "
                     f"{row['n_recorded_preparation_id']} | {row['n_recorded_slide_id']} | {row['n_recorded_acquisition_session_id']} |")
    lines.extend(["", "These are inventory counts, not independent biological-replicate estimates. "
                  "Only one recorded Culture ID exists per strain in this workbook; multiple independent cultures within each strain "
                  "are not documented by these IDs. The independent biological-replicate count remains NOT ESTABLISHED. "
                  "If Culture ID denotes an independently grown culture, the documented count is one per strain, "
                  "not five biological replicates per strain. If it denotes a parent stock/strain, independent inoculation records are still needed.", "",
                  "## Matching and coverage", "",
                  "Local images are associated only by an exact strain + treatment + explicit timepoint + filename key, unique "
                  "in both local and workbook inventories, with matching first-frame dimensions and no internal metadata conflict. "
                  "This is stronger than filename-only matching but is not checksum verification of the provider's Windows files. "
                  "Missing times, alternative treatments and fuzzy filename variants are NOT silently corrected. "
                  "Unmatched images and unknown identifiers remain explicit. All overlap conclusions are conditional on these composite associations.", "",
                  "| Cohort | Retained images | Composite matches | Known culture/preparation/slide/session images |",
                  "| --- | ---: | ---: | --- |"])
    for cohort, task in summary["cohorts"].items():
        coverage = "/".join(str(task["levels"][name]["n_known_images"]) for name in IDENTIFIERS)
        lines.append(f"| {cohort} | {task['n_retained']} | {task['n_matched']} | {coverage} |")
    lines.extend(["", "`image_metadata_mapping.csv` includes candidate IDs, acceptance status, header checks and exact workbook row references. "
                  "`unmatched_alternative_candidates.csv` lists nonaccepted strain/filename candidates without transferring their labels or IDs. "
                  "Unknown metadata is not evidence of a new independent replicate.", "",
                  "## Original fold overlap", "",
                  "The original field groups have zero overlap, but the recorded higher-level identifiers below are shared. "
                  "Original held-out membership is verified against saved treated predictions. Untreated membership uses the "
                  "previously frozen reconstruction because the original trainer did not save image-level predictions.", "",
                  "| Cohort | Level | Known IDs | IDs shared across outer folds | Known held-out images with identifier seen in training |",
                  "| --- | --- | ---: | ---: | ---: |"])
    for cohort, task in summary["cohorts"].items():
        for name, counts in task["levels"].items():
            lines.append(f"| {cohort} | {name} | {counts['n_recorded_identifiers']} | "
                         f"{counts['n_identifiers_spanning_outer_folds']} | "
                         f"{counts['n_known_images_exposed_to_same_identifier_in_other_outer_folds']}/{counts['n_known_images']} |")
    lines.extend(["", "Per-fold denominators, shared IDs, and separate train/validation/test overlap for the internal-validation reruns "
                  "are in each cohort's `fold_overlap_counts.csv`. These are overlap measurements, not estimates of the amount "
                  "of performance inflation or proof that the network uses a specific batch cue.", "",
                  "## Feasibility and next experiments", "",
                  "On the documented subset, culture-level splitting cannot preserve each strain/condition in both training and testing "
                  "when there is only one recorded culture per strain. Randomly splitting fields, slides, or time points from that "
                  "culture does not create additional biological replication. The checkpoint-selection reruns preserve the same outer "
                  "folds and therefore do not fix this higher-level sharing.", "",
                  "`grouped_split_class_support.csv` gives per-class counts at every retained level. Two disjoint groups per class "
                  "are necessary for a train/test split containing each class on both sides; five are necessary for five test folds "
                  "each containing that class. These are only combinatorial lower bounds, not a sufficient experimental sample size "
                  "or a validated split. Preparations and sessions may be confounded with class or time and are not automatically "
                  "independent biological units.", "",
                  "First obtain definitions of Culture ID and Preparation ID, their scope/uniqueness, independent inoculation/growth "
                  "records, and a checksum-backed old-to-new path mapping for unresolved images. If independent cultures are absent, "
                  "new independently inoculated/grown cultures per strain are required to test between-culture generalization. "
                  "Distribute each new culture across the planned treatment/time panel, record preparation/slide/session lineage, "
                  "and reserve entire independent cultures for testing before model development. Replicate number requires a design "
                  "and precision/power justification; thousands of fields cannot replace independent cultures.", "",
                  "Known conflicting-label byte duplicates and cross-fold duplicate content remain unresolved. Resolve them before "
                  "claiming a leakage-free rerun; all duplicate copies and derived images must stay together. The workbook identifies "
                  "the microscope but does not verify brightfield versus phase-contrast acquisition.", "",
                  "## Workbook consistency", "",
                  f"Image metadata contains {summary['n_metadata_conflict_rows']} internally conflicting rows; "
                  f"{summary['n_hierarchy_conflicts']} identifier/parent inconsistencies were detected. "
                  "The workbook Summary reports 'Missing strain' for all image rows despite the populated Strain column. "
                  "Its Reference notes also say strain is blank unless confirmed. Counts are recomputed from image rows, "
                  "and normalized strain names are dataset-group aliases rather than new biological identity verification.", "",
                  "Evidence hashes, workbook reference text, per-condition/time inventory counts, unresolved rows, and "
                  "pre/post-QC mapping coverage are preserved alongside this report.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    root, workbook, out = args.root.resolve(), args.workbook.resolve(), args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    evidence = [workbook, Path(__file__).resolve()]
    print("[AUDIT] Reading workbook", flush=True)
    images, tables = read_workbook(workbook)
    metadata = normalize_metadata(images)
    metadata.to_csv(out / "workbook_image_records.csv", index=False)
    for sheet in ("Summary", "Reference"):
        tables[sheet].to_csv(out / ("workbook_" + sheet.lower() + "_cells.csv"), index=False)
    conflicts = hierarchy_conflicts(metadata)
    conflicts.to_csv(out / "identifier_hierarchy_conflicts.csv", index=False)
    counts = identifier_counts(metadata, ["strain"])
    counts.to_csv(out / "workbook_identifier_counts_by_strain.csv", index=False)
    identifier_counts(metadata, ["strain", "treatment"]).to_csv(out / "workbook_identifier_counts_by_condition.csv", index=False)
    identifier_counts(metadata, ["strain", "treatment", "timepoint"]).to_csv(out / "workbook_identifier_counts_by_condition_time.csv", index=False)
    frames = {}
    for cohort, (benchmark, qc_dir, _, _) in PROFILES.items():
        path = root / "Data" / benchmark / qc_dir / "qc_manifest/manifest_with_segmentation_qc.csv"
        frames[cohort] = normalize_manifest(pd.read_csv(path), cohort)
        evidence.append(path)
    local = pd.concat([df[["image_path", *KEYS]] for df in frames.values()]).drop_duplicates()
    if local.image_path.duplicated().any():
        raise ValueError("Cohorts disagree on image identity/labels")
    candidates = local.merge(metadata[KEYS].drop_duplicates(), on=KEYS, how="inner").image_path.tolist()
    print(f"[AUDIT] Reading image headers for {len(candidates)} composite candidates", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        dimensions = pd.DataFrame(pool.map(image_dimensions, candidates))
    dimensions.to_csv(out / "local_image_dimensions.csv", index=False)
    mapping = match_metadata(local, metadata, dimensions)
    mapping.to_csv(out / "image_metadata_mapping.csv", index=False)
    unmatched = mapping.loc[~mapping.mapping_status.eq(MATCHED), ["image_path", *KEYS, "mapping_status"]]
    alternatives = unmatched.merge(metadata, on=["strain", "filename"], how="inner", suffixes=("_local", "_provider"))
    alternatives["accepted"] = False
    alternatives.to_csv(out / "unmatched_alternative_candidates.csv", index=False)
    coverage, cohorts = [], {}
    for cohort, frame in frames.items():
        print(f"[AUDIT] Checking {cohort} and all frozen folds", flush=True)
        dest = out / cohort
        dest.mkdir()
        mapped = frame.merge(mapping.drop(columns=KEYS), on="image_path", validate="one_to_one")
        for stage, sub in (("before_qc", mapped), ("after_qc", mapped.loc[mapped.retained])):
            for status, group in sub.groupby("mapping_status"):
                coverage.append({"cohort": cohort, "stage": stage, "mapping_status": status, "n_images": len(group),
                                 **{"n_known_" + name: int(group[name].ne(UNKNOWN).sum()) for name in IDENTIFIERS}})
        identifier_counts(mapped, ["class_name", "timepoint", "retained"]).to_csv(dest / "pre_post_qc_identifier_counts.csv", index=False)
        cohorts[cohort] = audit_folds(root, cohort, PROFILES[cohort], frame, mapping, dest, evidence)
    pd.DataFrame(coverage).to_csv(out / "mapping_coverage_before_after_qc.csv", index=False)
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "scid_used": False,
               "n_workbook_images": len(metadata), "n_local_images_before_qc": len(local),
               "n_metadata_conflict_rows": int(metadata.metadata_conflicts.ne("").sum()),
               "n_hierarchy_conflicts": len(conflicts),
               "workbook_identifiers": {name: {"n_distinct": len(set(metadata[name]) - {UNKNOWN}),
                                              "n_unknown_images": int(metadata[name].eq(UNKNOWN).sum())} for name in IDENTIFIERS},
               "independent_biological_replicates": "NOT_ESTABLISHED", "cohorts": cohorts,
               "source_evidence": [{"path": str(p), "sha256": sha256(p)} for p in dict.fromkeys(evidence)]}
    write_json(out / "summary.json", summary)
    (out / "experimental_independence_audit.md").write_text(render_report(summary, counts))
    print(json.dumps({"report": str(out / "experimental_independence_audit.md"), "cohorts": cohorts}, indent=2), flush=True)


if __name__ == "__main__":
    main()
