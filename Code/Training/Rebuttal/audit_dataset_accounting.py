#!/usr/bin/env python3
"""Reconcile legacy raw image inventories, QC manifests, and training populations.

This audit counts image files/manifest rows, not verified independent physical
fields or biological replicates. It never modifies a source manifest or QC file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from validation import sha256, write_json

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Code/Training/Model_Training_Treated_Condition"))
from build_treated_condition_manifest import parse_dataset_a_folder, parse_new_root_folder

ZERO = "Benchmark_0min_5class"
TWELVE = "Benchmark_Treated_Condition"
THIRTEEN = "Benchmark_Treated_Condition_13class"
FIFTEEN = "Benchmark_Treated_Condition_15class"
PROFILES = {
    "untreated_5class": (ZERO, "manifest_0min_fiveclass.csv", "segmentation_qc"),
    "treated_12class": (TWELVE, "manifest_treated_condition.csv", "segmentation_qc_fresh"),
    "treated_13class": (THIRTEEN, "manifest_treated_condition.csv", "segmentation_qc_combined"),
    "treated_15class": (FIFTEEN, "manifest_treated_condition.csv", "segmentation_qc_combined"),
    "treated_additional_2classes": (FIFTEEN, "segmentation_qc_additional/manifest_additional_only.csv", "segmentation_qc_additional"),
}
SOURCES = {
    "Dataset_A": "Data/Dataset_A",
    "New_Data_0min_root": "Data/New Data_and_0_min(untreated control)/New Data + 0_min(untreated control)",
    "Data_additional": "Data/Data_additional",
}
IMAGE_SUFFIXES = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
REP_COLS = ("replicate_id", "biological_replicate", "culture_id", "experiment_id", "acquisition_session_id", "replicate")
MISSING = {"", "NA", "N/A", "NAN", "NONE", "UNKNOWN", "UNRESOLVED"}
QC_NUMERIC = {
    "seg_qc_n_tiles": "n_tiles", "seg_qc_n_pass_tiles": "n_pass",
    "seg_qc_n_fail_tiles": "n_fail", "seg_qc_pass_fraction": "pass_fraction",
    "seg_qc_total_cells": "total_cells", "seg_qc_max_cells_per_tile": "max_cells",
    "seg_qc_low_cells_tiles": "low_cells", "seg_qc_large_cell_tiles": "large_cells",
    "seg_qc_artifact_tiles": "artifacts",
}
ACQUISITION_IDS = ("culture_id", "preparation_id", "slide_id", "acquisition_session_id")
METADATA_MATCH = "unique_composite_and_dimensions"


def normalize_fields(df):
    out = df.copy()
    if "canonical_strain" not in out:
        out["canonical_strain"] = out.get("canonical_strain_id", out["class_name"])
    if "canonical_treatment" not in out:
        out["canonical_treatment"] = "Untreated"
    if "source_root" not in out:
        out["source_root"] = out.source_set.replace({"internal_0min_new_root": "New_Data_0min_root"})
    out["experimental_replicate"] = "UNKNOWN"
    return out


def classify_decisions(qc):
    status = qc.seg_qc_status.fillna("missing").astype(str)
    keep = pd.to_numeric(qc.drop_from_training, errors="raise").fillna(0).eq(0)
    decisions = pd.Series("other_exclusion", index=qc.index)
    decisions.loc[status.eq("fail") & ~keep] = "qc_failed"
    decisions.loc[status.eq("missing") & ~keep] = "missing_segmentation"
    decisions.loc[keep] = "retained"
    return decisions


def retention_table(frame, dimensions):
    work = frame.copy()
    for value in ("retained", "qc_failed", "missing_segmentation", "other_exclusion"):
        work[value] = work.decision.eq(value).astype(int)
    table = work.groupby(dimensions, dropna=False, sort=True).agg(
        n_before=("image_path", "size"), n_after=("retained", "sum"),
        n_qc_failed=("qc_failed", "sum"), n_missing_segmentation=("missing_segmentation", "sum"),
        n_other_excluded=("other_exclusion", "sum"),
    ).reset_index()
    table["n_excluded"] = table.n_before - table.n_after
    table["retained_fraction"] = table.n_after / table.n_before
    if not (table.n_excluded == table.n_qc_failed + table.n_missing_segmentation + table.n_other_excluded).all():
        raise ValueError("Nonconserving retention table")
    return table


def attach_acquisition_metadata(frame, mapping):
    """Join only accepted provenance associations; never infer biological repeats."""
    if mapping.image_path.isna().any() or mapping.image_path.duplicated().any():
        raise ValueError("Metadata mapping requires unique, nonmissing image paths")
    if not set(frame.image_path).issubset(set(mapping.image_path)):
        raise ValueError("Metadata mapping does not cover the pre-QC population")
    cols = ["image_path", "strain", "treatment", "timepoint", "filename", "mapping_status",
            "metadata_row", "identity_evidence", *ACQUISITION_IDS]
    result = frame.merge(mapping[cols], on="image_path", how="left", validate="one_to_one")
    expected_treatment = result.canonical_treatment.where(result.timepoint_min.ne(0), "Control")
    if (not result.strain.eq(result.canonical_strain).all()
            or not result.treatment.eq(expected_treatment).all()
            or not pd.to_numeric(result.timepoint, errors="raise").eq(result.timepoint_min).all()
            or not result.filename.eq(result.image_path.map(lambda p: Path(p).name)).all()):
        raise ValueError("Metadata mapping changed image condition/time/filename")
    accepted = result.mapping_status.eq(METADATA_MATCH)
    if result.mapping_status.isna().any():
        raise ValueError("Missing metadata mapping status")
    if result.loc[accepted, "metadata_row"].isna().any():
        raise ValueError("Accepted metadata association lacks a workbook row")
    if not result.loc[accepted, "identity_evidence"].eq("composite_metadata_association_not_checksum_verified").all():
        raise ValueError("Unexpected metadata identity evidence")
    for column in ACQUISITION_IDS:
        values = result[column].fillna("UNKNOWN").astype(str).str.strip()
        values = values.mask(values.str.upper().isin(MISSING), "UNKNOWN")
        if (values.ne("UNKNOWN") & ~accepted).any():
            raise ValueError("Unaccepted metadata candidate has an assigned identifier")
        result[column] = values
    result["experimental_replicate"] = "UNKNOWN"
    result["independent_biological_replicates"] = "NOT_ESTABLISHED"
    return result.drop(columns=["strain", "treatment", "timepoint", "filename"])


def export_acquisition_accounting(audit_dir, frames, out, evidence):
    """Reuse the checked workbook audit without upgrading its identity evidence."""
    summary_path, verification_path = audit_dir / "summary.json", audit_dir / "verification.json"
    summary = json.loads(summary_path.read_text())
    verification = json.loads(verification_path.read_text())
    if (summary.get("scid_used") is not False or verification.get("scid_used") is not False
            or verification.get("all_checks_passed") is not True):
        raise ValueError("Require a verified metadata audit that excludes SCID")
    if not summary.get("source_evidence"):
        raise ValueError("Metadata audit lacks source evidence")
    for item in summary["source_evidence"]:
        path = Path(item["path"])
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Metadata-audit evidence changed: {path}")
        evidence.append({**item, "size_bytes": path.stat().st_size})
    mapping_path = audit_dir / "image_metadata_mapping.csv"
    mapping_hash = sha256(mapping_path)
    mapping = pd.read_csv(mapping_path, dtype={key: str for key in ACQUISITION_IDS}, low_memory=False)
    accepted = mapping.loc[mapping.mapping_status.eq(METADATA_MATCH)]
    for column in ("n_metadata_candidates", "n_local_key_candidates"):
        if not accepted[column].eq(1).all():
            raise ValueError("Accepted metadata association is not unique")
    for axis in ("width", "height"):
        if (not accepted["local_" + axis].gt(0).all()
                or not accepted["local_" + axis].eq(accepted["metadata_" + axis]).all()):
            raise ValueError("Accepted metadata association has inconsistent dimensions")
    if any(accepted[column].fillna("").ne("").any() for column in ("header_error", "metadata_conflicts")):
        raise ValueError("Accepted metadata association has unresolved conflicts")
    expected = set().union(*(set(frame.image_path) for frame in frames.values()))
    if set(mapping.image_path) != expected or len(mapping) != summary["n_local_images_before_qc"]:
        raise ValueError("Metadata audit and accounting audit have different image populations")
    enriched, coverage = [], []
    for profile, frame in frames.items():
        joined = attach_acquisition_metadata(frame, mapping)
        enriched.append(joined)
        for stage, subset in (("before_qc", joined), ("after_qc", joined.loc[joined.decision.eq("retained")])):
            row = {"benchmark": profile, "stage": stage, "n_images": len(subset),
                   "n_metadata_associations": int(subset.mapping_status.eq(METADATA_MATCH).sum())}
            for name in ACQUISITION_IDS:
                known = subset[name].ne("UNKNOWN")
                row["n_known_" + name] = int(known.sum())
                row["n_unknown_" + name] = int((~known).sum())
                row["n_recorded_" + name] = int(subset.loc[known, name].nunique())
            row["independent_biological_replicates"] = "NOT_ESTABLISHED"
            coverage.append(row)
            if stage == "after_qc" and profile in summary["cohorts"]:
                original = summary["cohorts"][profile]
                if row["n_images"] != original["n_retained"] or row["n_metadata_associations"] != original["n_matched"]:
                    raise ValueError("Metadata coverage differs from the verified audit")
                for name in ACQUISITION_IDS:
                    if (row["n_known_" + name] != original["levels"][name]["n_known_images"]
                            or row["n_recorded_" + name] != original["levels"][name]["n_recorded_identifiers"]):
                        raise ValueError("Recorded identifier counts differ from the verified audit")
    combined = pd.concat(enriched, ignore_index=True)
    columns = ["benchmark", "image_path", "image_uid", "canonical_strain", "canonical_treatment",
               "timepoint_min", "decision", "seg_qc_reason", "mapping_status", "metadata_row",
               "identity_evidence", *ACQUISITION_IDS, "experimental_replicate", "independent_biological_replicates"]
    combined[columns].to_csv(out / "field_accounting_with_acquisition_ids.csv", index=False)
    dimensions = ["benchmark", "canonical_strain", "canonical_treatment", "timepoint_min"]
    for name in ACQUISITION_IDS:
        retention_table(combined, [*dimensions, name]).to_csv(out / f"retention_by_{name}.csv", index=False)
    retention_table(combined, [*dimensions, *ACQUISITION_IDS]).to_csv(out / "retention_by_acquisition_hierarchy.csv", index=False)
    coverage_table = pd.DataFrame(coverage)
    coverage_table.to_csv(out / "acquisition_metadata_coverage.csv", index=False)
    if sha256(mapping_path) != mapping_hash:
        raise ValueError("Metadata mapping changed during accounting")
    for path in (summary_path, verification_path, mapping_path):
        evidence.append({"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size})
    result = {"audit_dir": str(audit_dir), "scid_used": False,
              "identity_evidence": "composite_metadata_association_not_checksum_verified",
              "independent_biological_replicates": "NOT_ESTABLISHED", "coverage": coverage,
              "workbook_identifiers": summary["workbook_identifiers"]}
    write_json(out / "acquisition_metadata_assessment.json", result)
    return result


def inventory_raw(root):
    rows, ignored = [], []
    for source, relative in SOURCES.items():
        base = root / relative
        parser = parse_dataset_a_folder if source == "Dataset_A" else parse_new_root_folder
        for condition in sorted(base.iterdir()):
            if not condition.is_dir() or condition.name.startswith("."):
                continue
            strain, treatment, susceptibility = parser(condition.name)
            for folder, dirs, files in os.walk(condition):
                dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__MACOSX")
                time = re.fullmatch(r"(\d+)_min", Path(folder).name)
                for name in sorted(files):
                    path = Path(folder) / name
                    if path.suffix.lower() not in IMAGE_SUFFIXES:
                        continue
                    if name.startswith("._") or not time:
                        ignored.append({"image_path": str(path), "reason": "appledouble_sidecar" if name.startswith("._") else "unrecognized_timepoint"})
                        continue
                    minute = int(time[1])
                    rows.append({"image_path": str(path.resolve()), "source_root": source,
                                 "source_folder": condition.name, "canonical_strain": strain,
                                 "canonical_treatment": "Untreated" if minute == 0 else treatment,
                                 "antibiotic_context": treatment, "susceptibility_text": susceptibility,
                                 "timepoint_min": minute, "experimental_replicate": "UNKNOWN"})
    return pd.DataFrame(rows), pd.DataFrame(ignored, columns=["image_path", "reason"])


def read_xlsx_sheet(path, sheet_member):
    """Read identifiers from OOXML, preserving sparse cell positions."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared = ["".join(t.text or "" for t in node.findall(".//m:t", ns))
                      for node in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        rows = []
        for row in ET.fromstring(archive.read(sheet_member)).findall(".//m:row", ns):
            values = {}
            for cell in row:
                column = re.sub(r"\d", "", cell.get("r", ""))
                node = cell.find("m:v", ns)
                value = node.text if node is not None else "".join(t.text or "" for t in cell.findall(".//m:t", ns))
                values[column] = shared[int(value)] if cell.get("t") == "s" and value else value or ""
            rows.append(values)
    header = rows[0]
    return pd.DataFrame([{header.get(col, col): value for col, value in row.items()} for row in rows[1:]])


def inspect_workbook(root, untreated, out):
    path = root / "Data/Dataset_A_0_min_metadata.xlsx"
    data = read_xlsx_sheet(path, "xl/worksheets/sheet2.xml")
    data = data.loc[data.Filename.fillna("").ne("")].copy()
    alias = {"5-alpha": "E.coli5-alpha", "5-alphaAmpR": "E.coli5-alphaAmpR", "5-alphaKanR": "E.coli5-alphaKanR", "AIEC": "AIEC", "UPEC": "UPEC"}
    data["canonical_strain"] = data["Dataset Group"].map(alias)
    keys = ["canonical_strain", "Filename"]
    candidates = data.groupby(keys, dropna=False).agg(
        metadata_candidate_rows=("Filename", "size"),
        candidate_culture_ids=("Culture ID", lambda s: "|".join(sorted(set(s.dropna().astype(str))))),
        candidate_acquisition_sessions=("Acquisition Session", lambda s: "|".join(sorted(set(s.dropna().astype(str))))),
    ).reset_index()
    benchmark = untreated[["image_path", "canonical_strain", "decision"]].copy()
    benchmark["Filename"] = benchmark.image_path.map(lambda p: Path(p).name)
    benchmark["benchmark_name_matches"] = benchmark.groupby(keys)["image_path"].transform("size")
    matches = benchmark.merge(candidates, on=keys, how="left", validate="many_to_one")
    matches["metadata_candidate_rows"] = matches.metadata_candidate_rows.fillna(0).astype(int)
    matches["mapping_status"] = np.where(matches.metadata_candidate_rows.eq(0), "no_filename_candidate", "unverified_filename_candidate")
    matches.to_csv(out / "replicate_metadata_candidates.csv", index=False)
    data.groupby(["Dataset Group", "Culture ID", "Acquisition Session", "PreperationID", "Slide ID"], dropna=False).size().reset_index(name="n_workbook_rows").to_csv(out / "workbook_reported_acquisitions.csv", index=False)
    return {"path": str(path), "sha256": sha256(path), "n_rows": len(data),
            "reported_culture_ids": sorted(data["Culture ID"].dropna().astype(str).unique().tolist()),
            "n_benchmark_rows_with_filename_candidate": int(matches.metadata_candidate_rows.gt(0).sum()),
            "n_retained_rows_with_filename_candidate": int((matches.metadata_candidate_rows.gt(0) & matches.decision.eq("retained")).sum()),
            "n_unambiguous_name_candidates": int((matches.metadata_candidate_rows.eq(1) & matches.benchmark_name_matches.eq(1)).sum()),
            "n_metadata_paths_found": int(data["Full Path"].fillna("").map(lambda p: Path(p).is_file()).sum()),
            "verified_benchmark_replicate_assignments": 0,
            "interpretation": "Workbook describes acquisition/culture IDs, but Windows paths do not identify these benchmark files. Filename-only candidates are not verified replicate assignments."}


def aggregate_tile_metadata(path):
    pieces, sample_ids, replicate_counts = [], Counter(), Counter()
    for chunk in pd.read_csv(path, usecols=["sample_id", "exclude_reason", "num_cells", "replicate"],
                             chunksize=100000, keep_default_na=False):
        sample_ids.update(chunk.sample_id.astype(str))
        replicate_counts.update(chunk.replicate.astype(str))
        reason = chunk.exclude_reason.astype(str).str.strip().replace({"nan": "", "none": ""})
        counts = pd.to_numeric(chunk.num_cells, errors="raise").fillna(0)
        values = pd.DataFrame({"tiled_image_id": chunk.sample_id.astype(str).str.replace(r"_r\d+_c\d+$", "", regex=True),
                               "n_tiles": 1, "n_pass": reason.eq("").astype(int), "total_cells": counts,
                               "max_cells": counts, "low_cells": reason.str.startswith("low_cells:").astype(int),
                               "large_cells": reason.str.startswith("large_cell:").astype(int),
                               "artifacts": reason.str.startswith("artifact_area:").astype(int)})
        pieces.append(values.groupby("tiled_image_id", sort=False).agg({k: "max" if k == "max_cells" else "sum" for k in values if k != "tiled_image_id"}))
    merged = pd.concat(pieces).groupby(level=0, sort=False).agg({k: "max" if k == "max_cells" else "sum" for k in pieces[0]})
    merged["n_fail"] = merged.n_tiles - merged.n_pass
    merged["pass_fraction"] = merged.n_pass / merged.n_tiles
    return merged, {"n_tile_rows": sum(sample_ids.values()), "n_unique_sample_ids": len(sample_ids),
                    "n_duplicate_tile_rows": sum(v - 1 for v in sample_ids.values()),
                    "n_segmented_image_ids": len(merged), "replicate_value_counts": dict(replicate_counts)}


def recompute_field_status(row, policy):
    if pd.isna(row.get("n_tiles")):
        return "missing", "missing_segmentation"
    reasons = []
    if row["n_pass"] < policy["min_pass_tiles"]:
        reasons.append(f"few_pass_tiles:{int(row['n_pass'])}")
    if row["pass_fraction"] < policy["min_pass_fraction"]:
        reasons.append(f"low_pass_fraction:{row['pass_fraction']:.3f}")
    if row["total_cells"] < policy["min_total_cells"]:
        reasons.append(f"low_total_cells:{int(row['total_cells'])}")
    if row["large_cells"] > policy["max_large_cell_tiles"]:
        reasons.append(f"large_cell_tiles:{int(row['large_cells'])}")
    if row["artifacts"] > policy["max_artifact_tiles"]:
        reasons.append(f"artifact_tiles:{int(row['artifacts'])}")
    return ("fail", ";".join(reasons)) if reasons else ("pass", "")


def verify_tile_qc(root, profile, qc, summary_path, out, evidence):
    summary = json.loads(summary_path.read_text())
    metadata_path = root / summary["seg_metadata_csv"]
    map_path = root / summary["tile_image_map_csv"]
    print(f"[AUDIT] Recomputing tile QC: {profile}", flush=True)
    aggregate, stats = aggregate_tile_metadata(metadata_path)
    evidence.extend([{"path": str(p), "sha256": sha256(p), "size_bytes": p.stat().st_size} for p in (metadata_path, map_path)])
    tile_map = pd.read_csv(map_path, usecols=["image_uid", "tiled_image_id", "status", "n_tiles"])
    stats["tile_map_status_counts"] = tile_map.status.value_counts(dropna=False).to_dict()
    stats["n_map_rows"] = len(tile_map)
    if tile_map.image_uid.duplicated().any() or tile_map.tiled_image_id.duplicated().any():
        raise ValueError(f"Duplicate QC join keys: {map_path}")
    mapped = tile_map.merge(aggregate, left_on="tiled_image_id", right_index=True, how="left", suffixes=("_expected", ""))
    calculated = mapped.apply(lambda row: recompute_field_status(row, summary["policy"]), axis=1)
    mapped["recomputed_status"] = [v[0] for v in calculated]
    mapped["recomputed_reason"] = [v[1] for v in calculated]
    compare = qc.merge(mapped.drop(columns=["tiled_image_id"]), on="image_uid", how="left", validate="one_to_one")
    mismatch_columns = {}
    mismatch = np.zeros(len(compare), dtype=bool)
    for column, derived in QC_NUMERIC.items():
        differences = ~np.isclose(pd.to_numeric(compare[column]).fillna(0), compare[derived].fillna(0), equal_nan=True)
        mismatch_columns[column] = int(differences.sum())
        mismatch |= differences
    for column, derived in [("seg_qc_status", "recomputed_status"), ("seg_qc_reason", "recomputed_reason")]:
        differences = compare[column].fillna("").ne(compare[derived].fillna(""))
        mismatch_columns[column] = int(differences.sum())
        mismatch |= differences.to_numpy()
    compare.loc[mismatch].to_csv(out / f"{profile}_tile_qc_disagreements.csv", index=False)
    partial = compare.n_tiles.gt(0) & compare.n_tiles.lt(compare.n_tiles_expected)
    coverage = compare[["image_path", "image_uid", "canonical_strain", "canonical_treatment", "timepoint_min",
                        "decision", "n_tiles_expected", "n_tiles", "n_pass", "pass_fraction", "recomputed_status"]].copy()
    coverage["coverage_status"] = np.where(compare.n_tiles.isna(), "missing", np.where(partial, "partial", "complete"))
    coverage["pass_fraction_over_expected_tiles"] = compare.n_pass.fillna(0) / compare.n_tiles_expected
    coverage.to_csv(out / f"{profile}_tile_coverage.csv", index=False)
    expected_summary = {"n_images": len(qc), "n_keep_combined": int(qc.decision.eq("retained").sum()),
                        "n_drop_combined": int(qc.decision.ne("retained").sum()),
                        "n_keep_seg_qc": int(qc.drop_from_training_seg_qc.eq(0).sum()),
                        "n_drop_seg_qc": int(qc.drop_from_training_seg_qc.ne(0).sum()),
                        "qc_status_counts": qc.seg_qc_status.value_counts().to_dict()}
    stats["summary_json_mismatches"] = {k: {"saved": summary.get(k), "observed": value}
                                        for k, value in expected_summary.items() if summary.get(k) != value}
    stats.update(profile=profile, policy=summary["policy"], metadata_path=str(metadata_path),
                 n_qc_rows=len(qc), n_field_disagreements=int(mismatch.sum()), column_disagreements=mismatch_columns,
                 n_missing_segmentation=int(compare.recomputed_status.eq("missing").sum()),
                 n_partial_tile_fields=int(partial.sum()),
                 n_retained_partial_tile_fields=int((partial & compare.decision.eq("retained")).sum()),
                 n_retained_below_expected_grid_pass_fraction=int((compare.decision.eq("retained") &
                     coverage.pass_fraction_over_expected_tiles.lt(summary["policy"]["min_pass_fraction"])).sum()),
                 n_metadata_image_ids_without_map=len(set(aggregate.index) - set(tile_map.tiled_image_id)))
    return stats


def markdown_table(frame):
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join("---" for _ in frame.columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(v).replace("|", "/") for v in row) + " |")
    return lines


def audit(root, out, metadata_audit=None):
    out.mkdir(parents=True, exist_ok=False)
    sources, findings, frames, flows = [], [], {}, []
    for relative in ["Code/Training/Rebuttal/audit_dataset_accounting.py",
                     "Code/Training/Model_Training_5_class/build_0min_strain_manifest.py",
                     "Code/Training/Model_Training_Treated_Condition/build_treated_condition_manifest.py",
                     "Code/Training/Model_Training_Treated_Condition/build_treated_segmentation_qc_manifest.py",
                     "Code/Training/Model_Training_Treated_Condition/run_15class_seg_qc.sbatch",
                     "Code/Segementation/run_tiles_with_stats_tif_meta_data_v2.py"]:
        path = root / relative
        sources.append({"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size})
    for profile, (directory, manifest_name, qc_directory) in PROFILES.items():
        raw_path = root / "Data" / directory / manifest_name
        qc_path = root / "Data" / directory / qc_directory / "qc_manifest/manifest_with_segmentation_qc.csv"
        before = normalize_fields(pd.read_csv(raw_path, low_memory=False))
        qc = normalize_fields(pd.read_csv(qc_path, low_memory=False))
        for path in (raw_path, qc_path):
            sources.append({"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size})
        for label, frame in [("before", before), ("qc", qc)]:
            for key in ("image_path", "image_uid"):
                if frame[key].isna().any() or frame[key].duplicated().any():
                    raise ValueError(f"{profile} {label}: nonunique or missing {key}")
        if set(before.image_path) != set(qc.image_path) or set(before.image_uid) != set(qc.image_uid):
            raise ValueError(f"{profile}: raw and QC populations differ")
        qc["decision"] = classify_decisions(qc)
        qc["benchmark"] = profile
        qc["reason_combination"] = qc.seg_qc_reason.fillna("").replace("", "none")
        by_path = before.set_index("image_path")
        drift = {}
        for column in ("group_id", "class_id", "class_name", "timepoint_min", "image_uid"):
            drift[column] = int((qc.set_index("image_path")[column].astype(str) != by_path.loc[qc.image_path, column].astype(str)).sum())
        if any(drift.values()):
            findings.append({"kind": "manifest_metadata_drift", "benchmark": profile, "counts": drift})
        flags = qc.drop_from_training.fillna(0).astype(int)
        seg_flags = qc.drop_from_training_seg_qc.fillna(0).astype(int)
        inconsistencies = int((flags.eq(0) != qc.seg_qc_status.eq("pass")).sum())
        if inconsistencies:
            findings.append({"kind": "filter_status_disagreement", "benchmark": profile, "n_rows": inconsistencies})
        counts = qc.decision.value_counts()
        flows.append({"benchmark": profile, "n_before": len(before), "n_classes": qc.class_name.nunique(),
                      "n_qc_failed": int(counts.get("qc_failed", 0)), "n_missing_segmentation": int(counts.get("missing_segmentation", 0)),
                      "n_other_excluded": int(counts.get("other_exclusion", 0)), "n_after": int(counts.get("retained", 0)),
                      "n_groups_before": qc.group_id.nunique(), "n_groups_after": qc.loc[flags.eq(0), "group_id"].nunique(),
                      "n_extra_segflag_exclusions": int((flags.eq(0) & seg_flags.ne(0)).sum()),
                      "recorded_replicate_columns": "|".join(c for c in REP_COLS if c in qc) or "none"})
        frames[profile] = qc
        cols = ["benchmark", "image_path", "image_uid", "group_id", "canonical_strain", "canonical_treatment", "timepoint_min", "source_root", "source_folder", "experimental_replicate", "class_id", "decision", "drop_from_training", "drop_from_training_seg_qc", "seg_qc_status", "seg_qc_reason"]
        qc[cols].to_csv(out / f"{profile}_field_ledger.csv", index=False)
    flow = pd.DataFrame(flows)
    flow.to_csv(out / "benchmark_flow.csv", index=False)
    combined = pd.concat(frames.values(), ignore_index=True)
    dimensions = {
        "strain": ["canonical_strain"], "treatment": ["canonical_treatment"],
        "timepoint": ["timepoint_min"], "strain_treatment": ["canonical_strain", "canonical_treatment"],
        "strain_treatment_timepoint": ["canonical_strain", "canonical_treatment", "timepoint_min"],
        "source_strain_treatment_timepoint_replicate": ["source_root", "source_folder", "canonical_strain", "canonical_treatment", "timepoint_min", "experimental_replicate"],
    }
    for name, cols in dimensions.items():
        retention_table(combined, ["benchmark", *cols]).to_csv(out / f"retention_by_{name}.csv", index=False)
    combinations = combined.groupby(["benchmark", "decision", "reason_combination"], dropna=False).size().reset_index(name="n_fields")
    combinations.to_csv(out / "exclusion_reason_combinations.csv", index=False)
    reason_rows = []
    for profile, frame in frames.items():
        for reason, n in frame.seg_qc_reason.fillna("").str.split(";").explode().str.split(":").str[0].value_counts().items():
            if reason:
                reason_rows.append({"benchmark": profile, "reason": reason, "n_fields_nonexclusive": int(n)})
    pd.DataFrame(reason_rows).to_csv(out / "exclusion_reasons_nonexclusive.csv", index=False)

    raw, ignored = inventory_raw(root)
    raw.to_csv(out / "raw_source_inventory.csv", index=False)
    ignored.to_csv(out / "ignored_raw_entries.csv", index=False)
    raw.groupby(["source_root", "canonical_strain", "canonical_treatment", "antibiotic_context", "timepoint_min"], dropna=False).size().reset_index(name="n_image_files").to_csv(out / "raw_source_counts.csv", index=False)
    inventory_totals = {"untreated_image_files": int(raw.timepoint_min.eq(0).sum()),
                        "treated_image_files": int(raw.timepoint_min.gt(0).sum()),
                        "ignored_appledouble_files": int(ignored.reason.eq("appledouble_sidecar").sum()),
                        "untreated_by_source": raw.loc[raw.timepoint_min.eq(0)].source_root.value_counts().to_dict(),
                        "treated_by_source": raw.loc[raw.timepoint_min.gt(0)].source_root.value_counts().to_dict()}
    main_sets = {"untreated_5class": raw.loc[raw.timepoint_min.eq(0)], "treated_15class": raw.loc[raw.timepoint_min.gt(0)]}
    for profile, inventory in main_sets.items():
        decision = frames[profile].set_index("image_path").decision
        provenance = inventory.copy()
        provenance["decision"] = provenance.image_path.map(decision).fillna("not_selected_for_benchmark")
        missing_paths = frames[profile].loc[~frames[profile].image_path.isin(inventory.image_path), ["image_path", "image_uid"]]
        missing_paths.to_csv(out / f"{profile}_manifest_paths_absent_from_raw_inventory.csv", index=False)
        if len(missing_paths):
            findings.append({"kind": "manifest_images_absent_from_raw_inventory", "benchmark": profile, "n_rows": len(missing_paths)})
        provenance.to_csv(out / f"{profile}_raw_to_benchmark.csv", index=False)
        provenance.groupby(["source_root", "canonical_strain", "canonical_treatment", "timepoint_min", "decision"], dropna=False).size().reset_index(name="n_fields").to_csv(out / f"{profile}_raw_to_benchmark_summary.csv", index=False)
    transitions = []
    for old, new in [("treated_12class", "treated_13class"), ("treated_13class", "treated_15class")]:
        first, second = frames[old], frames[new]
        added = second.loc[~second.image_path.isin(first.image_path)]
        removed = first.loc[~first.image_path.isin(second.image_path)]
        shared = second.loc[second.image_path.isin(first.image_path)]
        original = first.set_index("image_path")
        changed = int(shared.decision.ne(shared.image_path.map(original.decision)).sum())
        record = {"from": old, "to": new, "n_added_before_qc": len(added),
                  "n_added_after_qc": int(added.decision.eq("retained").sum()), "n_removed": len(removed),
                  "shared_decision_changes": changed, "added_classes": sorted(added.class_name.unique().tolist())}
        transitions.append(record)
        added[["image_path", "canonical_strain", "canonical_treatment", "timepoint_min", "source_root", "decision"]].to_csv(out / f"{old}_to_{new}_added_images.csv", index=False)
    write_json(out / "cohort_transitions.json", transitions)

    tile_checks = []
    for profile in ("untreated_5class", "treated_13class", "treated_additional_2classes"):
        directory, _, qc_directory = PROFILES[profile]
        summary_path = root / "Data" / directory / qc_directory / "qc_manifest/segmentation_qc_summary.json"
        sources.append({"path": str(summary_path), "sha256": sha256(summary_path), "size_bytes": summary_path.stat().st_size})
        tile_checks.append(verify_tile_qc(root, profile, frames[profile], summary_path, out, sources))
    joined = pd.concat([frames["treated_13class"], frames["treated_additional_2classes"]]).set_index("image_path")
    full = frames["treated_15class"].set_index("image_path")
    if set(joined.index) != set(full.index):
        raise ValueError("15-class manifest does not equal the 13-class plus additional populations")
    transfer_mismatches = {c: int(joined.loc[full.index, c].fillna("").astype(str).ne(full[c].fillna("").astype(str)).sum())
                           for c in [*QC_NUMERIC, "seg_qc_status", "seg_qc_reason", "decision", "class_name"]}
    write_json(out / "tile_qc_verification.json", {"recomputed_sources": tile_checks, "15class_merge_mismatch_counts": transfer_mismatches})

    models = []
    for profile, directory in [("untreated_5class", "Benchmark_0min_5class_QC"), ("treated_13class", "Benchmark_Treated_Condition_13class_QC"), ("treated_15class", "Benchmark_Treated_Condition_15class_QC")]:
        model_dir = root / "Model" / directory / "efficientnet_b0"
        args_path = model_dir / "training_args.json"
        args = json.loads(args_path.read_text())
        sources.append({"path": str(args_path), "sha256": sha256(args_path), "size_bytes": args_path.stat().st_size})
        retained = frames[profile].loc[frames[profile].decision.eq("retained")]
        class_counts = retained.class_name.value_counts().sort_index().to_dict()
        name_to_id = dict(retained[["class_name", "class_id"]].drop_duplicates().itertuples(index=False, name=None))
        record = {"benchmark": profile, "training_args_n_samples": args["n_samples"], "manifest_n_retained": len(retained),
                  "class_counts_match": class_counts == args["class_counts"],
                  "class_id_mapping_matches": name_to_id == {name: i for i, name in enumerate(args["class_names"])},
                  "n_groups_match": retained.group_id.nunique() == args["n_groups"]}
        predictions_path = model_dir / "crossval_predictions.csv"
        if predictions_path.exists():
            predictions = pd.read_csv(predictions_path, usecols=["image_path", "y_true"])
            sources.append({"path": str(predictions_path), "sha256": sha256(predictions_path), "size_bytes": predictions_path.stat().st_size})
            record.update(prediction_rows=len(predictions),
                          predictions_match_retained_images=set(predictions.image_path) == set(retained.image_path) and not predictions.image_path.duplicated().any(),
                          prediction_labels_match=bool(predictions.y_true.eq(predictions.image_path.map(retained.set_index("image_path").class_id)).all()))
            with predictions_path.open("rb") as stream:
                record["prediction_file_lines_including_header"] = sum(1 for _ in stream)
        models.append(record)
    write_json(out / "model_population_checks.json", models)
    workbook = inspect_workbook(root, frames["untreated_5class"], out)
    write_json(out / "replicate_metadata_assessment.json", workbook)
    acquisition = export_acquisition_accounting(metadata_audit, frames, out, sources) if metadata_audit else None
    claims = pd.DataFrame([
        {"claim": "untreated_collected", "reported": 8803, "observed": inventory_totals["untreated_image_files"], "scope": "image files across original Dataset_A and new-root source trees; not verified unique physical fields"},
        {"claim": "treated_collected", "reported": 67849, "observed": inventory_totals["treated_image_files"], "scope": "15-class pre-QC manifest and matching raw source files"},
        {"claim": "treated_benchmark", "reported": 17124, "observed": len(frames["treated_15class"].query('decision == "retained"')), "scope": "15-class retained rows, consistent with training arguments and prediction rows"},
    ])
    claims["observed_minus_reported"] = claims.observed - claims.reported
    claims.to_csv(out / "reviewer_claim_reconciliation.csv", index=False)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "counting_unit": "image file or manifest row, not verified independent field/replicate",
              "raw_inventory": inventory_totals, "benchmark_flow": flows, "claims": claims.to_dict("records"),
              "cohort_transitions": transitions, "tile_qc_checks": tile_checks,
              "15class_merge_mismatch_counts": transfer_mismatches, "model_population_checks": models,
              "replicate_metadata": workbook, "acquisition_metadata": acquisition,
              "findings": findings, "source_evidence": sources}
    write_json(out / "audit_summary.json", report)
    return report, frames, flow


def write_report(out, report, frames, flow):
    main = flow.loc[flow.benchmark.isin(["untreated_5class", "treated_13class", "treated_15class"]),
                    ["benchmark", "n_before", "n_qc_failed", "n_missing_segmentation", "n_after"]]
    by_benchmark = flow.set_index("benchmark").to_dict("index")
    zero, twelve, thirteen, fifteen = [by_benchmark[p] for p in
                                     ("untreated_5class", "treated_12class", "treated_13class", "treated_15class")]
    inventory = report["raw_inventory"]
    model_check = next(r for r in report["model_population_checks"] if r["benchmark"] == "treated_15class")
    line_count = model_check.get("prediction_file_lines_including_header")
    line_count_text = f"{line_count:,}" if line_count is not None else "an unknown number of"
    treated_difference = 67849 - fifteen["n_before"]
    untreated_difference = inventory["untreated_image_files"] - 8803
    lines = ["# Dataset accounting audit", "", "## Findings", "",
             f"The saved 15-class training population contains **{fifteen['n_after']:,} image rows**, from **{fifteen['n_before']:,} pre-QC rows**. "
             f"The exclusions comprise **{fifteen['n_qc_failed']:,} QC failures** and **{fifteen['n_missing_segmentation']:,} missing-segmentation records**. "
             "Training and prediction-population checks are recorded in `model_population_checks.json`.", "",
             f"The 15-class prediction CSV has {line_count_text} physical lines including its header. "
             "Counting the header would reproduce the reviewer's 17,124 figure in this audit, but the origin of the manuscript number is not documented. "
             f"The claimed 67,849 treated fields exceeds the manifest count by {treated_difference:,}; no corresponding "
             f"{treated_difference:,}-image exclusion stage was found.", "",
             f"The current source trees contain **{inventory['untreated_image_files']:,} untreated image files**: "
             f"{inventory['untreated_by_source'].get('Dataset_A', 0):,} in Dataset_A and "
             f"{inventory['untreated_by_source'].get('New_Data_0min_root', 0):,} in the new-root source. "
             f"The untreated benchmark builder explicitly selects only the latter source. Of those {zero['n_before']:,}, "
             f"{zero['n_qc_failed']:,} fail QC and {zero['n_missing_segmentation']:,} lack segmentation results, leaving **{zero['n_after']:,}**. "
             f"The {inventory['untreated_image_files'] - zero['n_before']:,} unselected files were excluded "
             "by source selection, not by the benchmark's QC. The claimed 8,803 untreated fields is not reproduced; "
             f"the current file inventory differs by {untreated_difference:+,}. Historical changes or deduplication cannot be inferred from this difference.", "",
             "Counts in this audit refer to saved image files/manifest rows. They do not establish that every file represents "
             "a different physical field, and they are not biological-replicate counts. File content deduplication and physical-field "
             "identity verification were not performed.", "", "## Image flow", ""]
    lines.extend(markdown_table(main))
    lines.extend(["", "Before = after + QC failed + missing segmentation for these populations. "
                  "Missing segmentation is an unavailable processing result and must not be presented as evidence of poor image quality.", "",
                  "## Cohort versions", "",
                  f"The original 12-class treated cohort has {twelve['n_before']:,} rows and retains {twelve['n_after']:,}. "
                  f"Adding {thirteen['n_before'] - twelve['n_before']:,} AIEC + Ampicillin images "
                  f"({thirteen['n_after'] - twelve['n_after']:,} retained) produces the 13-class cohort of {thirteen['n_before']:,} / {thirteen['n_after']:,}. "
                  f"Adding {fifteen['n_before'] - thirteen['n_before']:,} further images "
                  f"({fifteen['n_after'] - thirteen['n_after']:,} retained) produces the 15-class cohort of {fifteen['n_before']:,} / {fifteen['n_after']:,}. The final addition comprises "
                  "E.coli5-alphaAmpR + Kanamycin and E.coli5-alphaKanR + Ampicillin.", "",
                  "The approximately 17,124-image Results population corresponds to the **15-class** cohort, not the "
                  "13-class cohort. Cohort versions overlap and must not be summed. The manuscript, rerun, and tables "
                  "need to state the same cohort version. Later duplicate/metadata exclusions in rebuttal sensitivity "
                  "experiments are separate from this historical QC flow.", "",
                  "## Retention by strain and treatment", ""])
    for profile in ("untreated_5class", "treated_15class"):
        lines.extend([f"### {profile}", ""])
        table = retention_table(frames[profile], ["canonical_strain", "canonical_treatment"])
        lines.extend(markdown_table(table.drop(columns=["retained_fraction", "n_other_excluded", "n_excluded"])))
        lines.append("")
    lines.extend(["## Retention by time point", ""])
    lines.extend(markdown_table(retention_table(frames["treated_15class"], ["timepoint_min"]).drop(columns=["retained_fraction", "n_other_excluded", "n_excluded"])))
    lines.extend(["", "Full strain x treatment x time-point accounting is in `retention_by_strain_treatment_timepoint.csv`. "
                  "The source-folder breakdown and explicit UNKNOWN replicate category are in "
                  "`retention_by_source_strain_treatment_timepoint_replicate.csv`.", "",
                  "## QC provenance and checks", "",
                  "QC requires at least 3 passing tiles, pass fraction >= 0.34, at least 6 segmented cells, "
                  "zero large-cell tiles, and zero artifact tiles. For a complete 3 x 3 grid, 3/9 = 0.333 fails the "
                  "0.34 threshold, so at least 4 passing tiles are effectively required. Tile tagging uses "
                  "minimum 1 cell, maximum cell area 30 um^2, and maximum artifact area 100 um^2 in the saved launchers. "
                  "Recorded failure reasons may overlap; the reason-combination table is additive, while individual-reason counts are not.", ""])
    for check in report["tile_qc_checks"]:
        lines.append(f"- {check['profile']}: rebuilt {check['n_qc_rows']:,} image QC records from {check['n_tile_rows']:,} tile rows; "
                     f"{check['n_field_disagreements']:,} disagreements; {check['n_duplicate_tile_rows']:,} duplicate tile rows; "
                     f"{check['n_partial_tile_fields']:,} images have fewer recorded tiles than the tile map expects "
                     f"({check['n_retained_partial_tile_fields']:,} retained).")
    lines.extend(["", "The 15-class QC merge was checked against the 13-class and additional-image QC tables. "
                  f"Per-column mismatches: `{report['15class_merge_mismatch_counts']}`.", "",
                  "The 15-class combined QC folder does not contain its own generated retention summaries; this audit supplies them. "
                  "The historical `broken_classids` manifest/model variant is not used; the current label-to-index mapping is "
                  "checked against the saved 15-class model configuration.", "",
                  "Incomplete tile coverage is itemized in the `*_tile_coverage.csv` tables. The original QC denominator "
                  "is the number of recorded tiles, not the expected grid size. These tables also show the fraction "
                  "over expected tiles for sensitivity checking; the audit does not change the original exclusion decisions.", "",
                  "## Experimental replicates", "",
                  "The benchmark image manifests do not record experimental replicate, culture, slide, or acquisition-session IDs. "
                  "Tile-level `replicate` is a filename parser output, not verified experimental metadata; its full value counts "
                  "are recorded in `tile_qc_verification.json`. Source folder and `group_id` must not be substituted for biological replicates.", ""])
    workbook = report["replicate_metadata"]
    lines.extend([f"`Dataset_A_0_min_metadata.xlsx` contains {workbook['n_rows']:,} image records and reports culture IDs "
                  f"{', '.join(workbook['reported_culture_ids'])}. Its Windows file paths do not resolve to the benchmark images. "
                  f"Strain-plus-filename matching finds candidates for {workbook['n_benchmark_rows_with_filename_candidate']:,} pre-QC benchmark rows "
                  f"({workbook['n_retained_rows_with_filename_candidate']:,} retained), but these are unverified mappings. "
                  "`replicate_metadata_candidates.csv` exposes the candidates and ambiguity; none is treated as a verified replicate assignment.", "",
                  "These older filename-only candidates are not used for identifier-stratified accounting. "
                  "Missing identifiers are reported as UNKNOWN, not as zero experimental replicates.", ""])
    acquisition = report.get("acquisition_metadata")
    if acquisition:
        lines.extend(["### Updated provider metadata", "",
                      "The newer metadata audit is used only after rechecking its source hashes, exact local populations, "
                      "condition/time/filename keys and retained mapping coverage. Associations require exact composite keys "
                      "and matching image dimensions in that audit, but are not checksum verification of provider files. "
                      "SCID is not used; no new biological-replicate assignments are inferred.", "",
                      "`field_accounting_with_acquisition_ids.csv` preserves every pre-QC row, its exclusion decision, "
                      "mapping status and workbook row reference. `retention_by_acquisition_hierarchy.csv` reports "
                      "before/after counts by strain, treatment, time, culture, preparation, slide and session. Separate "
                      "`retention_by_culture_id.csv`, `retention_by_preparation_id.csv`, `retention_by_slide_id.csv` and "
                      "`retention_by_acquisition_session_id.csv` provide additive tables, including UNKNOWN categories. "
                      "Unmatched metadata does not exclude an image from historical QC accounting.", ""])
        coverage = pd.DataFrame(acquisition["coverage"])
        coverage = coverage.loc[coverage.benchmark.isin(["untreated_5class", "treated_13class", "treated_15class"])]
        lines.extend(markdown_table(coverage[["benchmark", "stage", "n_images", "n_metadata_associations",
                                             "n_recorded_culture_id", "n_recorded_preparation_id",
                                             "n_recorded_slide_id", "n_recorded_acquisition_session_id"]]))
        lines.extend(["", "Identifier counts are not biological-replicate counts and must not be added across strata. "
                      "The provider workbook records one Culture ID per strain; independent inoculation/growth is not "
                      "established by these labels. Independent biological-replicate counts remain NOT ESTABLISHED. "
                      "`acquisition_metadata_coverage.csv` reports known and unknown image denominators at each level.", ""])
    else:
        lines.extend(["A checked per-image mapping to experimental-replicate/acquisition identifiers is required to "
                      "complete identifier-stratified accounting. Use --metadata_audit with the verified newer workbook "
                      "audit to export recorded acquisition IDs; those IDs still do not establish biological replication.", ""])
    lines.extend(["## Evidence and limitations", "",
                  "`audit_summary.json` records source paths and SHA-256 hashes, lineage checks, training-population checks, "
                  "and discrepancies. Each benchmark has a row-level decision ledger. Raw-source inventories exclude AppleDouble "
                  "sidecars and derived tiles, archives, and alternate copies outside the three benchmark source roots. "
                  "The inventory is the current on-disk collection, not proof of historical collection totals.", "",
                  f"Additional metadata findings: `{report['findings']}`.", ""])
    (out / "dataset_accounting_audit.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--metadata_audit", type=Path,
                        help="Verified experimental-independence audit for recorded acquisition-ID breakdowns; SCID excluded")
    args = parser.parse_args()
    report, frames, flow = audit(args.root.resolve(), args.out_dir.resolve(),
                                args.metadata_audit.resolve() if args.metadata_audit else None)
    write_report(args.out_dir.resolve(), report, frames, flow)
    print(flow.to_string(index=False), flush=True)
    print(f"[DONE] {args.out_dir / 'dataset_accounting_audit.md'}", flush=True)


if __name__ == "__main__":
    main()
