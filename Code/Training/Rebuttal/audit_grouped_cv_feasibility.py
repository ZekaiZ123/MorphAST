#!/usr/bin/env python3
"""Audit class coverage under crossed culture/slide/session grouping before CV.

This read-only preflight does not train models, resolve labels, infer biological
replicates, or relax existing field groups. SCID is not used.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from validation import sha256, validate_manifest, write_json

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AUDIT = ROOT / "outputs/rebuttal/experimental_independence_20261120"
DEFAULT_HASHES = ROOT / "outputs/rebuttal/duplicate_conflicts_20261118_verified/all_retained_image_hashes.csv"
ID_DTYPES = {key: str for key in ("culture_id", "preparation_id", "slide_id", "acquisition_session_id")}
UNKNOWN = {"", "UNKNOWN", "nan", "None", "<NA>"}
VARIANTS = {
    "field_and_hash": ["legacy_field_group_id", "original_sha256"],
    "preparation_field_and_hash": ["preparation_id", "slide_id", "original_sha256", "legacy_field_group_id"],
    "preparation_slide_and_hash": ["preparation_id", "slide_id", "original_sha256"],
    "culture_and_hash": ["culture_id", "original_sha256"],
    "slide_and_hash": ["slide_id", "original_sha256"],
    "session_and_hash": ["acquisition_session_id", "original_sha256"],
    "culture_slide_session": ["culture_id", "slide_id", "acquisition_session_id", "original_sha256"],
    "culture_slide_session_field": ["culture_id", "slide_id", "acquisition_session_id",
                                    "legacy_field_group_id", "original_sha256"],
}


def known(values):
    return values.notna() & ~values.astype(str).str.strip().isin(UNKNOWN)


def clean_population(metadata, inventory):
    """Identify conflicting content BEFORE excluding unmatched metadata rows."""
    validate_manifest(metadata)
    if inventory.image_path.duplicated().any() or set(metadata.image_path) != set(inventory.image_path):
        raise ValueError("Hash inventory must cover the retained population exactly once")
    if (not inventory.hash_status.eq("ok").all() or inventory.original_sha256.isna().any()
            or not inventory.original_sha256.str.fullmatch(r"[0-9a-f]{64}").all()):
        raise ValueError("Failed or invalid source hashes")
    df = metadata.merge(inventory[["image_path", "original_sha256"]], on="image_path", validate="one_to_one")
    counts = df.groupby("original_sha256").class_id.nunique()
    df["conflicting_label_content"] = df.original_sha256.isin(counts[counts > 1].index)
    df["metadata_eligible"] = df.mapping_status.eq("unique_composite_and_dimensions")
    for column in ID_DTYPES:
        df["metadata_eligible"] &= known(df[column])
    df["exclusion_reason"] = np.select(
        [df.conflicting_label_content, ~df.metadata_eligible],
        ["unresolved_conflicting_label_content", "unverified_or_missing_metadata"], default="included")
    df["included"] = df.exclusion_reason.eq("included")
    retained = df.loc[df.included].copy().reset_index(drop=True)
    if retained.empty:
        raise ValueError("No eligible images")
    retained["legacy_field_group_id"] = retained.group_id.astype(str)
    if "outer_fold" in retained:
        retained = retained.rename(columns={"outer_fold": "legacy_outer_fold"})
    return retained, df


def component_ids(df, columns):
    """Join transitively on ANY identifier, never on the tuple of identifiers."""
    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for column in columns:
        if not known(df[column]).all():
            raise ValueError(f"Unknown grouping identifier: {column}")
        for indices in df.groupby(column, sort=True).indices.values():
            first = find(int(indices[0]))
            for index in indices[1:]:
                parent[find(int(index))] = first
    members = {}
    for index, path in enumerate(df.image_path):
        members.setdefault(find(index), []).append(str(path))
    identifiers = {root: "component_" + hashlib.sha256("\n".join(sorted(paths)).encode()).hexdigest()
                   for root, paths in members.items()}
    return pd.Series([identifiers[find(i)] for i in range(len(df))], index=df.index)


def grouping_tables(df, original_classes, n_folds):
    df = df.copy()
    support, components = [], []
    for variant, columns in VARIANTS.items():
        column = variant + "_group_id"
        df[column] = component_ids(df, columns)
        for label, name in original_classes:
            subset = df.loc[df.class_id.eq(label)]
            count = subset[column].nunique()
            support.append({"variant": variant, "class_id": int(label), "class_name": name,
                            "n_images": len(subset), "n_components": count,
                            "n_preparations": subset.preparation_id.nunique(),
                            "n_cultures": subset.culture_id.nunique(),
                            "n_sessions": subset.acquisition_session_id.nunique(),
                            "train_test_necessary_count_met": count >= 2,
                            "requested_outer_fold_necessary_count_met": count >= n_folds,
                            "interpretation": "necessary_count_only_not_a_validated_split_or_biological_replicate_count"})
        for component, subset in df.groupby(column, sort=True):
            components.append({"variant": variant, "component_id": component, "n_images": len(subset),
                               "n_classes": subset.class_id.nunique(),
                               "class_names": " | ".join(sorted(set(subset.class_name))),
                               "culture_ids": "|".join(sorted(set(subset.culture_id))),
                               "slide_ids": "|".join(sorted(set(subset.slide_id))),
                               "acquisition_session_ids": "|".join(sorted(set(subset.acquisition_session_id)))})
    return df, pd.DataFrame(support), pd.DataFrame(components)


def load_population(root, audit_dir, hash_path):
    summary_path, verification_path = audit_dir / "summary.json", audit_dir / "verification.json"
    summary = json.loads(summary_path.read_text())
    verification = json.loads(verification_path.read_text())
    if summary["scid_used"] or verification["scid_used"] or not verification["all_checks_passed"]:
        raise ValueError("Require a verified metadata audit that excludes SCID")
    for item in summary["source_evidence"]:
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Metadata-audit evidence changed: {item['path']}")
    settings_path = root / "Model/Benchmark_Treated_Condition_15class_QC/efficientnet_b0/training_args.json"
    settings = json.loads(settings_path.read_text())
    manifest_path = root / settings["manifest_csv"]
    original = pd.read_csv(manifest_path)
    original = original.loc[original.drop_from_training.fillna(0).astype(int).eq(0)].reset_index(drop=True)
    if validate_manifest(original) != settings["class_names"] or len(original) != settings["n_samples"]:
        raise ValueError("Original population or label mapping changed")
    metadata_path = audit_dir / "treated_15class/retained_images_with_identifiers_and_folds.csv"
    metadata = pd.read_csv(metadata_path, dtype=ID_DTYPES, low_memory=False)
    if metadata.image_path.duplicated().any() or set(metadata.image_path) != set(original.image_path):
        raise ValueError("Metadata ledger does not match original retained population")
    metadata = metadata.set_index("image_path").loc[original.image_path].reset_index()
    for column in ("group_id", "class_id", "class_name", "timepoint_min", "image_uid"):
        if metadata[column].astype(str).tolist() != original[column].astype(str).tolist():
            raise ValueError(f"Metadata ledger changed original {column}")
    inventory = pd.read_csv(hash_path)
    return metadata, inventory, [settings_path, manifest_path, metadata_path, hash_path, summary_path, verification_path]


def verify_images(inventory, workers):
    def check(row):
        if sha256(Path(row["image_path"])) != row["original_sha256"]:
            raise ValueError(f"Image changed since duplicate audit: {row['image_path']}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(pool.map(check, inventory.to_dict("records")), 1):
            if i % 1000 == 0 or i == len(inventory):
                print(f"[VERIFY] {i}/{len(inventory)} source images", flush=True)


def export_audit(out, metadata, inventory, evidence, n_folds=5, images_rehashed=False):
    df, ledger = clean_population(metadata, inventory)
    classes = list(metadata[["class_id", "class_name"]].drop_duplicates().sort_values("class_id").itertuples(index=False, name=None))
    df, support, components = grouping_tables(df, classes, n_folds)
    out.mkdir(parents=True, exist_ok=False)
    df.to_csv(out / "cleaned_population.csv", index=False)
    ledger.to_csv(out / "inclusion_ledger.csv", index=False)
    support.to_csv(out / "grouping_class_support.csv", index=False)
    components.to_csv(out / "grouping_components.csv", index=False)
    ledger.groupby(["class_name", "timepoint_min", "exclusion_reason"], dropna=False).size().rename("n_images").reset_index().to_csv(out / "population_accounting.csv", index=False)
    bridges = df.groupby(["legacy_field_group_id", "class_name"], sort=True).agg(
        n_images=("image_path", "size"), n_preparations=("preparation_id", "nunique"),
        preparation_ids=("preparation_id", lambda x: "|".join(sorted(set(x)))),
        session_ids=("acquisition_session_id", lambda x: "|".join(sorted(set(x)))),
        timepoints=("timepoint_min", lambda x: "|".join(map(str, sorted(set(x))))),
    ).reset_index()
    bridges.loc[bridges.n_preparations > 1].to_csv(out / "field_groups_linking_preparations.csv", index=False)
    selected = "culture_slide_session_field"
    possible = bool(support.loc[support.variant.eq(selected), "requested_outer_fold_necessary_count_met"].all())
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "status": "necessary_counts_met_split_not_validated" if possible else "infeasible",
              "selected_variant": selected, "n_folds_requested": n_folds,
              "n_original": len(ledger), "n_included": len(df),
              "exclusion_counts": {k: int(v) for k, v in ledger.exclusion_reason.value_counts().items()},
              "n_conflicting_label_images": int(ledger.conflicting_label_content.sum()),
              "n_metadata_ineligible_images": int((~ledger.metadata_eligible).sum()),
              "n_preparations": int(df.preparation_id.nunique()),
              "n_joint_components": int(components.variant.eq(selected).sum()),
              "n_field_groups_linking_preparations": int(bridges.n_preparations.gt(1).sum()),
              "scid_used": False, "independent_biological_replicates": "NOT_ESTABLISHED",
              "source_images_rehashed": len(inventory) if images_rehashed else 0,
              "training_prepared": False, "source_evidence": evidence, "grouping_constraints": VARIANTS}
    write_json(out / "feasibility.json", report)
    lines = ["# Culture/slide/session CV feasibility", "",
             f"Status: **{report['status']}** for joint culture, slide, acquisition session, original field and exact-content grouping with {n_folds} outer folds.", "",
             f"Included {len(df):,}/{len(ledger):,} images across {report['n_preparations']} recorded preparations. "
             f"The requested constraints produce {report['n_joint_components']} connected components. "
             "Conflicting-label content is identified on the full retained cohort before metadata filtering.", "",
             "## Accounting", "",
             *[f"- {reason}: {count:,} images." for reason, count in report["exclusion_counts"].items()], "",
             "Exclusion reasons above are mutually exclusive, with label conflict taking precedence. Separate conflict and metadata flags in the inclusion ledger can overlap.", "",
             "| Class | Field/hash | Preparation+field | Preparation/slide/hash | Culture/hash | Slide/hash | Session/hash | Culture+slide+session | Joint+field |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for _, name in classes:
        counts = support.loc[support.class_name.eq(name)].set_index("variant").n_components
        lines.append(f"| {name} | " + " | ".join(str(counts[v]) for v in VARIANTS) + " |")
    lines.extend(["", "## Interpretation", "",
                  "Any shared culture, slide, session or original field identifier, or shared exact content, connects records transitively. "
                  "Concatenating identifiers instead would allow the same culture to appear on both sides. The requested IDs alone and the more conservative original-field-preserving variant are both reported.", "",
                  "A class in only one component cannot occur in both training and testing. With one recorded culture per strain, "
                  "holding out its culture removes its strain-condition classes from training. This cannot provide the requested closed-set 15-class validation. "
                  "No model, fold assignment, run.json or performance estimate is produced. No weaker grouping is silently substituted.", "",
                  "Group-count thresholds are necessary, not sufficient: crossing class constraints, balance and inner-validation coverage still require an explicit split audit if counts permit splitting.", "",
                  "## Limitations", "",
                  "- Recorded culture IDs do not establish independently grown biological replicates; SCID is excluded.",
                  "- Metadata uses exact composite keys and image dimensions, not checksum matching to the provider's reorganized files.",
                  "- Original field IDs normalize filenames across time; their physical interpretation needs provider confirmation.",
                  "- Preparations can correspond to time points within one acquisition session; preparation counts are not biological replicate counts.",
                  "- All copies of unresolved conflicting-label exact content are excluded, not relabeled. Unmatched metadata remains unknown.",
                  "- Same-label exact copies remain but are linked. Near-duplicates and unrecorded physical-field relationships are not ruled out.",
                  f"- Source images rehashed in this audit: {report['source_images_rehashed']:,}. Zero means the prior full-cohort hash inventory was used without rereading image bytes.", ""])
    (out / "feasibility.md").write_text("\n".join(lines))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--audit_dir", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--hash_inventory", type=Path, default=DEFAULT_HASHES)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--verify_images", action="store_true", help="Rehash every retained source image, including excluded conflicts")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 16 or args.n_folds < 2:
        parser.error("Require 1-16 workers and at least two outer folds")
    out = args.out_dir.resolve()
    if out.exists():
        raise FileExistsError(f"Use a new output directory: {out}")
    metadata, inventory, paths = load_population(ROOT, args.audit_dir, args.hash_inventory)
    clean_population(metadata, inventory)
    if args.verify_images:
        verify_images(inventory, args.workers)
    evidence = [{"path": str(path.resolve()), "sha256": sha256(path)} for path in paths + [Path(__file__)]]
    report = export_audit(out, metadata, inventory, evidence, args.n_folds, args.verify_images)
    for i, item in enumerate(evidence):
        path = Path(item["path"])
        dest = out / "evidence" / f"{i:02d}_{path.name}"
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(path, dest)
        if sha256(dest) != item["sha256"]:
            raise ValueError(f"Evidence changed during audit: {path}")
    print(json.dumps({"status": report["status"], "training_prepared": False, "report": str(out / "feasibility.md")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
