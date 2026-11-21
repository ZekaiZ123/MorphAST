#!/usr/bin/env python3
"""Export original publication-model counts, recorded IDs and raw confusions.

No rerun predictions, relabeling, inference or biological-replicate inference.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_corrected_error_cases import load_metadata, record_source
from audit_dataset_accounting import (
    ACQUISITION_IDS, attach_acquisition_metadata, classify_decisions,
    normalize_fields, retention_table,
)
from audit_error_cases import (
    PROTOCOL, TREATMENTS, add_decisions, confusion_long, markdown_table,
    matrix, per_class_metrics, render_counts, validate_predictions,
)
from audit_experimental_independence import identifier_counts
from validation import sha256, write_json

ROOT = Path(__file__).resolve().parents[3]
COHORTS = {
    "untreated_5class": "Benchmark_0min_5class_QC",
    "treated_13class": "Benchmark_Treated_Condition_13class_QC",
    "treated_15class": "Benchmark_Treated_Condition_15class_QC",
}


def validate_population(manifest, settings):
    names = settings["class_names"]
    if (settings.get("model") != "efficientnet_b0" or settings.get("mode") != "crossval"
            or settings.get("filter_auto_dropped") is not True):
        raise ValueError("Expected original QC-filtered EfficientNet-B0 cross-validation")
    if len(names) != len(set(names)) or manifest.image_path.isna().any() or manifest.image_path.duplicated().any():
        raise ValueError("Ambiguous publication labels or image paths")
    flags = pd.to_numeric(manifest.drop_from_training, errors="raise")
    if not flags.isin([0, 1]).all():
        raise ValueError("Invalid exclusion flags")
    kept = manifest.loc[flags.eq(0)].copy()
    if (len(kept) != settings["n_samples"]
            or kept.class_name.value_counts().to_dict() != settings["class_counts"]
            or kept.group_id.isna().any() or kept.group_id.nunique() != settings["n_groups"]):
        raise ValueError("Publication training population differs from retained manifest")
    if not kept.class_id.map(dict(enumerate(names))).equals(kept.class_name):
        raise ValueError("Publication class ID/order disagreement")
    return kept


def checked_matrix(metrics, names, fold, n_samples):
    values = np.asarray(metrics["confusion_matrix"], dtype=float)
    if (values.shape != (len(names), len(names)) or not np.isfinite(values).all()
            or (values < 0).any() or not np.equal(values, np.floor(values)).all()
            or values.sum() != metrics["n_val"] or metrics["fold"] != fold
            or metrics["n_train"] + metrics["n_val"] != n_samples):
        raise ValueError("Invalid saved publication confusion matrix")
    cm = pd.DataFrame(values.astype(np.int64), index=pd.Index(names, name="true_label"),
                      columns=pd.Index(names, name="predicted_label"))
    measured = per_class_metrics(cm)
    actual = {"accuracy": float(np.trace(cm.to_numpy()) / cm.to_numpy().sum()),
              "balanced_accuracy": float(measured.loc[measured.support.gt(0), "recall"].mean()),
              "macro_f1": float(measured.f1.mean())}
    for key, value in actual.items():
        if not np.isclose(value, metrics[key], atol=1e-9, rtol=1e-7):
            raise ValueError(f"Saved publication metric disagrees with counts: {key}")
    return cm


def class_tables(frame, names):
    """Denominators are images; known ID cardinalities are not replicate N."""
    kept = frame.loc[frame.decision.eq("retained")]
    counts = retention_table(frame, ["class_name", "canonical_strain", "canonical_treatment"])
    ids = identifier_counts(kept, ["class_name"])
    groups = kept.groupby("class_name").group_id.nunique().rename("n_recorded_field_groups")
    result = counts.merge(ids, on="class_name", validate="one_to_one").join(groups, on="class_name")
    result = result.set_index("class_name").loc[names].reset_index()
    if not result.n_after.eq(result.n_images).all() or result.n_after.sum() != len(kept):
        raise ValueError("Class counts do not conserve the retained population")
    result["sample_unit"] = "image_file_not_verified_unique_physical_field"
    result["identifier_count_scope"] = "known_metadata_only_not_biological_replicates"
    result["evaluation_protocol"] = PROTOCOL
    return result


def save_confusions(df, names, out, exact_folds, render):
    total = sum(exact_folds.values())
    total.to_csv(out / "confusion_exact_counts.csv")
    per_class_metrics(total).to_csv(out / "per_class_metrics.csv", index=False)
    fold_rows, fold_support = [], []
    for fold, cm in exact_folds.items():
        long = cm.stack().rename("count").reset_index()
        long["fold"] = fold
        fold_rows.append(long)
        for name in names:
            fold_support.append({"fold": fold, "class_name": name,
                                 "n_heldout": int(cm.loc[name].sum()),
                                 "n_train": int(total.loc[name].sum() - cm.loc[name].sum()),
                                 "heldout_used_for_checkpoint_selection": True})
    pd.concat(fold_rows, ignore_index=True).to_csv(out / "confusion_exact_by_fold.csv", index=False)
    pd.DataFrame(fold_support).to_csv(out / "class_sample_counts_by_fold.csv", index=False)
    if render:
        render_counts(total, out.name + " / publication model", out / "confusion_exact_counts")
    if df is None:
        return total
    modes = [("exact", "class_name", "pred_class_name", names),
             ("treatment_marginal", "true_treatment", "pred_treatment_marginal", TREATMENTS),
             ("treatment_top_class", "true_treatment", "pred_treatment_top_class", TREATMENTS)]
    for mode, true, predicted, labels in modes:
        cm = matrix(df, true, predicted, labels)
        if mode == "exact":
            pd.testing.assert_frame_equal(cm, total)
        else:
            cm.to_csv(out / f"confusion_{mode}_counts.csv")
            if render:
                render_counts(cm, out.name + " / " + mode, out / f"confusion_{mode}_counts")
        for suffix, dimensions in [("fold", ["fold"]), ("time", ["timepoint_min"]),
                                   ("session", ["acquisition_session_id"]),
                                   ("preparation", ["preparation_id"])]:
            confusion_long(df, dimensions, true, predicted, labels).to_csv(
                out / f"confusion_{mode}_by_{suffix}.csv", index=False)
    return total


def export_cohort(root, cohort, mapping, metadata_summary, out, evidence, render=True):
    model = root / "Model" / COHORTS[cohort] / "efficientnet_b0"
    settings_path = model / "training_args.json"
    settings = json.loads(settings_path.read_text())
    manifest_path = root / settings["manifest_csv"]
    names = settings["class_names"]
    if json.loads((model / "class_names.json").read_text()) != names:
        raise ValueError("Saved class names disagree with training settings")
    for path in (settings_path, manifest_path, model / "class_names.json"):
        record_source(path, evidence)
    manifest = pd.read_csv(manifest_path, low_memory=False)
    retained = validate_population(manifest, settings)
    frame = normalize_fields(manifest)
    frame["decision"] = classify_decisions(frame)
    frame = attach_acquisition_metadata(frame, mapping)
    kept = frame.loc[frame.decision.eq("retained")].copy()
    coverage = identifier_counts(kept.assign(cohort=cohort), ["cohort"]).iloc[0].to_dict()
    expected = metadata_summary["cohorts"][cohort]
    if coverage["n_images"] != expected["n_retained"] or coverage["n_metadata_matches"] != expected["n_matched"]:
        raise ValueError("Metadata coverage differs from verified audit")
    for key in ACQUISITION_IDS:
        level = expected["levels"][key]
        if (coverage["n_recorded_" + key] != level["n_recorded_identifiers"]
                or len(kept) - coverage["n_unknown_" + key] != level["n_known_images"]):
            raise ValueError("Recorded identifier totals differ from verified audit")
    pred = None
    if cohort != "untreated_5class":
        path = model / "crossval_predictions.csv"
        record_source(path, evidence)
        pred = validate_predictions(pd.read_csv(path, low_memory=False), manifest, settings)
        columns = ["image_path", *ACQUISITION_IDS, "mapping_status", "metadata_row", "identity_evidence",
                   "experimental_replicate", "independent_biological_replicates"]
        pred = add_decisions(pred.merge(kept[columns], on="image_path", validate="one_to_one"), names)
    folds = {}
    for fold in range(1, settings["n_folds"] + 1):
        path = model / f"fold_{fold}/metrics.json"
        record_source(path, evidence)
        folds[fold] = checked_matrix(json.loads(path.read_text()), names, fold, len(kept))
        if pred is not None:
            observed = matrix(pred.loc[pred.fold.eq(fold)], "class_name", "pred_class_name", names)
            pd.testing.assert_frame_equal(observed, folds[fold])
    pooled = sum(folds.values())
    if pooled.sum(axis=1).to_dict() != retained.class_name.value_counts().to_dict():
        raise ValueError("Raw confusion support differs from publication class counts")
    out.mkdir()
    table = class_tables(frame, names)
    table.to_csv(out / "class_sample_and_recorded_identifier_counts.csv", index=False)
    retention_table(frame, ["class_name", "timepoint_min"]).to_csv(out / "class_sample_counts_by_time.csv", index=False)
    for suffix, dimensions in [("class_time", ["class_name", "timepoint_min"]),
                               ("class", ["class_name"])]:
        identifier_counts(kept, dimensions).to_csv(out / f"recorded_identifiers_by_{suffix}.csv", index=False)
    retention_table(frame, ["class_name", "timepoint_min", *ACQUISITION_IDS]).to_csv(
        out / "sample_counts_by_acquisition_hierarchy.csv", index=False)
    ledger = ["image_path", "image_uid", "group_id", "class_id", "class_name", "canonical_strain",
              "canonical_treatment", "timepoint_min", "decision", *ACQUISITION_IDS, "mapping_status",
              "metadata_row", "identity_evidence", "experimental_replicate", "independent_biological_replicates"]
    frame[ledger].to_csv(out / "image_accounting_with_identifiers.csv", index=False)
    if pred is not None:
        pred.to_csv(out / "publication_predictions_with_identifiers.csv", index=False)
    save_confusions(pred, names, out, folds, render)
    coverage.update(model_dir=str(model), n_before_qc=len(frame), n_recorded_field_groups=int(kept.group_id.nunique()),
                    evaluation_protocol=PROTOCOL,
                    prediction_source="saved_fold_confusion_counts_only" if pred is None else "original_crossval_predictions.csv")
    write_json(out / "summary.json", coverage)
    return table, coverage


def write_report(out, tables, summaries):
    lines = ["# Original publication-model evidence", "",
             "EfficientNet-B0 only, read directly from Model/Benchmark_*_QC/efficientnet_b0. "
             "No corrected-CV or preparation-sensitivity predictions are used. No inference, new exclusions, "
             "relabeling or manuscript changes were performed. The 13-class model is named in the local draft; "
             "the 15-class model is used by later figure exports. These overlapping cohorts are separate, not additive.", "",
             "## Counting and interpretation", "",
             "N means image files. The pre-QC denominator is the model's input manifest, not all collected images. "
             "Retained counts exactly match training_args.json and pooled confusion row sums. "
             "Original field-group IDs do not establish independent experimental replicates.", "",
             "Culture, preparation, slide and session counts include only known, accepted workbook associations. "
             "Unknown images are reported separately, not counted as another replicate. IDs are deduplicated within "
             "each row and must not be summed across classes or timepoints. Cohort totals below deduplicate separately. "
             "Zero known IDs means no accepted identifiers, not zero actual experiments. "
             "Each mapped treated strain has one recorded Culture ID; independent inoculation/growth is not established. "
             "The independent biological/experimental replicate count remains NOT ESTABLISHED. SCID is excluded. "
             "Composite-key/dimension matching is not checksum verification of the provider's image files.", "",
             "Raw confusion matrices have recorded labels in rows and predicted labels in columns, include all zeros, "
             "and pool the five held-out folds once, without normalization. These are original CV results: the same "
             "held-out fold was used for checkpoint selection. They are not unbiased corrected test estimates. "
             "Known conflicting-label duplicate images have not been removed. "
             "Marginal treatment predictions sum class probabilities by treatment before argmax; "
             "mapping the highest-scoring exact class to its treatment is exported as a different decision rule.", "",
             "Untreated original training saved fold confusion counts, not per-image predictions. "
             "Its class/fold supports come from those saved matrices; no untreated image-level error, session-level "
             "confusion or prediction identity is reconstructed. Per-image metadata accounting is available separately.", "",
             "## Cohort totals", ""]
    summary_columns = ["cohort", "n_before_qc", "n_images", "n_metadata_matches", *["n_recorded_" + k for k in ACQUISITION_IDS]]
    lines += [markdown_table(pd.DataFrame(summaries)[summary_columns]), ""]
    columns = ["class_name", "n_before", "n_after", "n_metadata_matches",
               *["n_recorded_" + k for k in ACQUISITION_IDS], "n_unknown_culture_id"]
    for cohort, table in tables.items():
        lines += [f"## {cohort}", "", markdown_table(table[columns]), "",
                  "All recorded-ID counts above refer to retained images; the CSV contains unknown-image counts "
                  "for every ID level and the actual ID lists. Known counts are not total replicate counts.", "",
                  f"[Detailed class counts]({cohort}/class_sample_and_recorded_identifier_counts.csv) | "
                  f"[Raw exact matrix CSV]({cohort}/confusion_exact_counts.csv) | "
                  f"[Raw per-fold matrices]({cohort}/confusion_exact_by_fold.csv)", "",
                  "### Raw exact-label confusion", "",
                  markdown_table(pd.read_csv(out / cohort / "confusion_exact_counts.csv")), ""]
        if cohort != "untreated_5class":
            lines += ["### Raw treatment-marginal confusion", "",
                      markdown_table(pd.read_csv(out / cohort / "confusion_treatment_marginal_counts.csv")), ""]
    (out / "publication_model_evidence.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--metadata_audit", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--cohorts", nargs="+", choices=list(COHORTS), default=list(COHORTS))
    args = parser.parse_args()
    if len(args.cohorts) != len(set(args.cohorts)):
        parser.error("Specify each cohort only once")
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    evidence = {}
    mapping = load_metadata(args.metadata_audit.resolve(), evidence)
    metadata_summary = json.loads((args.metadata_audit / "summary.json").read_text())
    for name in ("export_publication_evidence.py", "audit_corrected_error_cases.py", "audit_dataset_accounting.py",
                 "audit_error_cases.py", "audit_experimental_independence.py", "validation.py"):
        record_source(Path(__file__).parent / name, evidence)
    tables, summaries = {}, []
    for cohort in args.cohorts:
        table, summary = export_cohort(args.root.resolve(), cohort, mapping, metadata_summary,
                                      out / cohort, evidence)
        tables[cohort] = table
        summaries.append(summary)
        print(f"[VERIFIED] {cohort}: {summary['n_images']} retained images", flush=True)
    pd.concat([table.assign(cohort=cohort) for cohort, table in tables.items()], ignore_index=True).to_csv(
        out / "all_class_counts.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "cohort_summary.csv", index=False)
    write_report(out, tables, summaries)
    for path, digest in evidence.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Source changed during export: {path}")
    artifacts = {str(path.relative_to(out)): sha256(path) for path in sorted(out.rglob("*")) if path.is_file()}
    write_json(out / "verification.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "all_checks_passed": True,
        "evaluation_protocol": PROTOCOL, "scid_used": False,
        "biological_replicate_count": "NOT_ESTABLISHED", "cohorts": summaries,
        "source_sha256": evidence, "artifact_sha256": artifacts,
    })
    print(f"[DONE] {out / 'publication_model_evidence.md'}", flush=True)


if __name__ == "__main__":
    main()
