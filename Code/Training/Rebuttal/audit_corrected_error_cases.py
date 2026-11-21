#!/usr/bin/env python3
"""Export verified outer-test error evidence, without inference or relabeling."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

from audit_dataset_accounting import ACQUISITION_IDS, attach_acquisition_metadata
from audit_error_cases import (
    FOCUS, TREATMENTS, UNKNOWN, add_decisions, export_assets, matrix,
    confusion_long, markdown_table, per_class_metrics, performance_table,
)
from audit_grouped_cv_feasibility import verify_images
from preparation_sensitivity import PROTOCOL as SENSITIVITY_PROTOCOL, audit_frozen_partition, verify_frozen_inputs
from summarize_nested_cv import read_verified_result
from validation import PROTOCOL, audit_partition, sha256, write_json

ROOT = Path(__file__).resolve().parents[3]
MATCHED = "unique_composite_and_dimensions"
META_COLUMNS = [*ACQUISITION_IDS, "mapping_status", "metadata_row", "identity_evidence",
                "experimental_replicate", "independent_biological_replicates"]
LIMITATIONS = [
    "Counts use recorded labels and image files, not verified physical fields or independent biological replicates.",
    "Original-cohort checkpoint-selection correction does not remove conflicting-label duplicates or acquisition sharing.",
    "Preparation/slide sensitivity permits original cross-time field-ID overlap and culture/session sharing; time and preparation effects are confounded.",
    "The cleaned and original populations and splits differ. Their performance difference does not isolate label-conflict or batch bias.",
    "Workbook associations are composite-key/dimension matches, not checksum verification of provider files. SCID is excluded.",
    "Batch ID, independently grown biological-replicate ID and optical contrast mode are UNKNOWN; recorded sessions/slides are not relabeled as biological batches.",
    "Quality summaries and image comparisons are descriptive, not evidence for a particular lysis or drug-response mechanism.",
]


def record_source(path, evidence):
    path = Path(path).resolve()
    evidence[str(path)] = sha256(path)


def load_metadata(audit_dir, evidence):
    summary = json.loads((audit_dir / "summary.json").read_text())
    verification = json.loads((audit_dir / "verification.json").read_text())
    if summary.get("scid_used") is not False or verification.get("scid_used") is not False or not verification.get("all_checks_passed"):
        raise ValueError("Require a verified metadata audit excluding SCID")
    for item in summary["source_evidence"]:
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Metadata evidence changed: {item['path']}")
        evidence[item["path"]] = item["sha256"]
    for name in ("summary.json", "verification.json", "image_metadata_mapping.csv"):
        record_source(audit_dir / name, evidence)
    mapping = pd.read_csv(audit_dir / "image_metadata_mapping.csv", dtype={k: str for k in ACQUISITION_IDS}, low_memory=False)
    if len(mapping) != summary["n_local_images_before_qc"] or mapping.image_path.duplicated().any():
        raise ValueError("Metadata mapping population changed")
    accepted = mapping.loc[mapping.mapping_status.eq(MATCHED)]
    if not accepted.n_metadata_candidates.eq(1).all() or not accepted.n_local_key_candidates.eq(1).all():
        raise ValueError("Accepted metadata keys are not unique")
    for axis in ("width", "height"):
        if not accepted["local_" + axis].gt(0).all() or not accepted["local_" + axis].eq(accepted["metadata_" + axis]).all():
            raise ValueError("Accepted metadata dimensions disagree")
    if any(accepted[c].fillna("").ne("").any() for c in ("header_error", "metadata_conflicts")):
        raise ValueError("Accepted metadata has unresolved conflicts")
    return mapping


def load_content_inventory(root, hash_path, evidence, workers):
    manifest_path = root / "Data/Benchmark_Treated_Condition_15class/segmentation_qc_combined/qc_manifest/manifest_with_segmentation_qc.csv"
    manifest = pd.read_csv(manifest_path, low_memory=False)
    manifest = manifest.loc[manifest.drop_from_training.eq(0)].copy()
    hashes = pd.read_csv(hash_path)
    if hashes.image_path.duplicated().any() or set(hashes.image_path) != set(manifest.image_path):
        raise ValueError("Full hash inventory must cover the original retained 15-class population")
    if not hashes.hash_status.eq("ok").all() or not hashes.original_sha256.fillna("").str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("Incomplete content hashes")
    verify_images(hashes, workers)
    record_source(manifest_path, evidence)
    record_source(hash_path, evidence)
    content = manifest[["image_path", "image_uid", "class_name"]].merge(hashes, on="image_path", validate="one_to_one")
    label_counts = content.groupby("original_sha256").class_name.nunique()
    content["conflicting_condition_labels"] = content.original_sha256.isin(label_counts[label_counts.gt(1)].index)
    content["original_cohort_duplicate_size"] = content.groupby("original_sha256").image_path.transform("size")
    return content


def attach_content(df, inventory):
    if inventory.image_path.duplicated().any() or not set(df.image_path).issubset(set(inventory.image_path)):
        raise ValueError("Content evidence does not cover analysis images")
    aligned = inventory.set_index("image_path").loc[df.image_path].reset_index()
    for key in ("image_uid", "class_name"):
        if df[key].astype(str).tolist() != aligned[key].astype(str).tolist():
            raise ValueError(f"Content provenance changed {key}")
    if "original_sha256" in df and df.original_sha256.tolist() != aligned.original_sha256.tolist():
        raise ValueError("Frozen image hashes differ from content evidence")
    result = df.drop(columns="original_sha256", errors="ignore").copy()
    for column in ("original_sha256", "conflicting_condition_labels", "original_cohort_duplicate_size"):
        result["audited_file_sha256" if column == "original_sha256" else column] = aligned[column].to_numpy()
    return result


def load_outer_predictions(run_dir, task, evidence):
    config_path = run_dir / "run.json"
    config = json.loads(config_path.read_text())
    if config["protocol"] not in (PROTOCOL, SENSITIVITY_PROTOCOL):
        raise ValueError("Only frozen internal-validation protocols are accepted")
    settings = config["tasks"][task]
    if settings["img_size"] != 224 or any(" + " not in n for n in settings["class_names"]):
        raise ValueError("This exporter expects 224-pixel treated-condition models")
    record_source(config_path, evidence)
    if config["protocol"] == SENSITIVITY_PROTOCOL:
        verify_frozen_inputs(run_dir, config)
    for relative, digest in config["source_sha256"].items():
        path = run_dir / "source" / relative
        if sha256(path) != digest:
            raise ValueError(f"Frozen source changed: {path}")
        evidence[str(path)] = digest
    # Previews must use exactly the loader used to evaluate these checkpoints.
    loader = "Code/utils/image_utils.py"
    if sha256(ROOT / loader) != config["source_sha256"][loader]:
        raise ValueError("Current image loader differs from the frozen evaluation loader")
    folder = run_dir / task
    path = folder / "manifest.csv"
    if sha256(path) != settings["manifest_sha256"]:
        raise ValueError("Frozen manifest changed")
    record_source(path, evidence)
    manifest = pd.read_csv(path, dtype={k: str for k in ACQUISITION_IDS}, low_memory=False)
    records, splits = [], {}
    for fold in range(1, settings["n_folds"] + 1):
        completion_path = folder / f"fold_{fold}/complete.json"
        completion = json.loads(completion_path.read_text())
        if completion["run_sha256"] != evidence[str(config_path.resolve())]:
            raise ValueError("Incomplete or mismatched outer fold")
        split_path = folder / "splits" / f"fold_{fold}.csv"
        if sha256(split_path) != settings["splits"][str(fold)]:
            raise ValueError("Frozen split changed")
        split = pd.read_csv(split_path)
        if split.row_id.tolist() != list(range(len(manifest))):
            raise ValueError("Split row identities changed")
        for column in ("image_path", "class_id", "group_id"):
            if not split[column].equals(manifest[column]):
                raise ValueError("Split and manifest disagree")
        if config["protocol"] == SENSITIVITY_PROTOCOL:
            audit_frozen_partition(manifest, split, task)
        else:
            audit_partition(manifest, split.role.to_numpy())
        selection_path = folder / f"fold_{fold}/model_selection.json"
        model = json.loads(selection_path.read_text())["selected_model"]
        result, saved = read_verified_result(folder, settings, fold, model)
        if not result["selected_for_primary"]:
            raise ValueError("Not the frozen primary model")
        expected = manifest.loc[split.role.eq("test")].copy().reset_index(drop=True)
        for column in ("class_name", "class_id", "image_uid", "timepoint_min"):
            if not saved[column].astype(str).tolist() == expected[column].astype(str).tolist():
                raise ValueError("Prediction identifiers differ from manifest")
        for column in ["y_true", "y_pred", "pred_class_name", *["prob_" + n for n in settings["class_names"]]]:
            expected[column] = saved[column].to_numpy()
        if expected.y_pred.map(dict(enumerate(settings["class_names"]))).tolist() != expected.pred_class_name.tolist():
            raise ValueError("Predicted class names disagree with class IDs")
        expected["fold"], expected["model"], expected["selected_epoch"] = fold, model, result["epoch"]
        records.append(expected)
        splits[fold] = split
        for path in (completion_path, split_path, selection_path, folder / f"fold_{fold}/{model}/test_metrics.json",
                     folder / f"fold_{fold}/{model}/test_predictions.csv"):
            record_source(path, evidence)
        evidence[str(folder / f"fold_{fold}/{model}/best.pt")] = result["checkpoint_sha256"]
    pooled = pd.concat(records, ignore_index=True)
    if len(pooled) != len(manifest) or pooled.image_path.duplicated().any() or set(pooled.image_path) != set(manifest.image_path):
        raise ValueError("Outer predictions do not cover each image exactly once")
    return pooled, splits, config, settings


def annotate_partition_content(df, splits):
    result = df.copy()
    lookup = result.set_index("image_path").audited_file_sha256
    for role in ("train", "validation"):
        result["test_content_seen_in_" + role] = False
    for fold, split in splits.items():
        test = result.fold.eq(fold)
        for role in ("train", "validation"):
            hashes = set(split.loc[split.role.eq(role), "image_path"].map(lookup))
            result.loc[test, "test_content_seen_in_" + role] = result.loc[test, "audited_file_sha256"].isin(hashes)
    return result


def select_controls(df):
    """Deterministic median errors; known acquisition constraints never relaxed."""
    keys = ["class_name", "pred_class_name", "timepoint_min", "fold", "preparation_id", "acquisition_session_id",
            "conflicting_condition_labels"]
    focus = df.loc[df.canonical_strain.eq(FOCUS) & df.canonical_treatment.isin(["Ampicillin", "Kanamycin"]) & df.any_error]
    records = []
    for _, group in focus.groupby(keys, dropna=False, sort=True):
        error = group.sort_values(["predicted_class_probability", "case_id"]).iloc[(len(group) - 1) // 2]
        row = {"error_case_id": error.case_id, "stratum_error_n": len(group), "error_selection": "lower_median_confidence_within_recorded_stratum"}
        for role, label in (("true_condition", error.class_name), ("predicted_condition", error.pred_class_name)):
            controls = df.loc[~df.any_error & ~df.conflicting_condition_labels & df.class_name.eq(label)
                              & df.timepoint_min.eq(error.timepoint_min) & df.fold.eq(error.fold)
                              & df.source_root.eq(error.source_root)]
            matched = ["timepoint_min", "fold", "source_root"]
            for column in (("culture_id", "preparation_id", "slide_id", "acquisition_session_id")
                           if role == "true_condition" else ("acquisition_session_id",)):
                if error[column] != UNKNOWN:
                    controls = controls.loc[controls[column].eq(error[column])]
                    matched.append(column)
            row[role + "_control_id"] = "" if controls.empty else controls.sort_values("case_id").iloc[0].case_id
            row[role + "_candidate_n"] = len(controls)
            row[role + "_matched_on"] = "|".join(matched)
        records.append(row)
    columns = ["error_case_id", "stratum_error_n", "error_selection", "true_condition_control_id",
               "predicted_condition_control_id", "true_condition_candidate_n", "predicted_condition_candidate_n",
               "true_condition_matched_on", "predicted_condition_matched_on"]
    return pd.DataFrame(records, columns=columns)


def render_matrix(cm, path, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(7, len(cm) * .8), max(6, len(cm) * .7)))
    values = cm.to_numpy()
    ax.imshow(values, cmap="Blues", vmin=0)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, str(values[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if values[i, j] > values.max() / 2 else "black")
    labels = [str(n).replace("E.coli5-alpha", "DH5alpha") for n in cm.index]
    ax.set_xticks(range(len(cm)), labels, rotation=65, ha="right", fontsize=8)
    ax.set_yticks(range(len(cm)), labels, fontsize=8)
    ax.set(xlabel="Predicted label", ylabel="Recorded label", title=label + "\nVerified outer-test image counts")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def export_tables(df, names, out, render=True):
    modes = [("exact", "class_name", "pred_class_name", names),
             ("treatment_marginal", "true_treatment", "pred_treatment_marginal", TREATMENTS),
             ("treatment_top_class", "true_treatment", "pred_treatment_top_class", TREATMENTS)]
    for mode, true, predicted, labels in modes:
        cm = matrix(df, true, predicted, labels)
        cm.to_csv(out / f"confusion_{mode}_counts.csv")
        per_class_metrics(cm).to_csv(out / f"metrics_{mode}_pooled.csv", index=False)
        rows = []
        for fold, group in df.groupby("fold"):
            metrics = per_class_metrics(matrix(group, true, predicted, labels))
            metrics["fold"] = fold
            rows.append(metrics)
        pd.concat(rows).to_csv(out / f"metrics_{mode}_by_fold.csv", index=False)
        if render and mode != "treatment_top_class":
            render_matrix(cm, out / f"confusion_{mode}_counts", out.name + " / " + mode)
        strata = {"fold": ["fold"], "time": ["timepoint_min"],
                  "condition_time_acquisition": ["class_name", "timepoint_min", *ACQUISITION_IDS],
                  "session": ["acquisition_session_id"], "preparation": ["preparation_id"],
                  "content_conflict": ["conflicting_condition_labels"]}
        for suffix, columns in strata.items():
            confusion_long(df, columns, true, predicted, labels).to_csv(out / f"confusion_{mode}_by_{suffix}.csv", index=False)
    strata = {"condition": ["class_name"], "condition_time": ["class_name", "timepoint_min"],
              "condition_fold": ["class_name", "fold"],
              "condition_time_acquisition": ["class_name", "timepoint_min", *ACQUISITION_IDS],
              "condition_time_conflict": ["class_name", "timepoint_min", "conflicting_condition_labels"]}
    for suffix, columns in strata.items():
        performance_table(df, columns).to_csv(out / f"performance_by_{suffix}.csv", index=False)
    qc = [k for k in ("seg_qc_total_cells", "seg_qc_pass_fraction", "predicted_class_probability", "true_class_probability") if k in df]
    df.groupby(["class_name", "timepoint_min", "conflicting_condition_labels", "exact_error"])[qc].agg(
        ["count", "median", "mean", "min", "max"]).to_csv(out / "descriptive_qc_by_condition_time_conflict_error.csv")


def case_card(row):
    esc = lambda v: html.escape(str(v), quote=True)
    warning = "<strong class='warning'>CONFLICTING CONDITION LABELS: byte-identical images have different recorded treatments.</strong>" if row.conflicting_condition_labels else ""
    picture = "<p>Image export failed; see image_assets.csv.</p>"
    if row.asset_status == "ok":
        picture = (f'<a href="{esc(row.original_file)}"><img loading="lazy" src="{esc(row.preview_file)}" alt="Loaded full-field preview"></a>'
                   f'<img loading="lazy" src="{esc(row.model_input_file)}" alt="224 by 224 input before normalization">')
    return (f'<article id="{esc(row.case_id)}"><h3>Recorded: {esc(row.class_name)}</h3>{warning}'
            f'<p>Predicted: {esc(row.pred_class_name)}; probability {row.predicted_class_probability:.4f}; '
            f'marginal treatment: {esc(row.pred_treatment_marginal)}</p>{picture}'
            f'<p>Time: {row.timepoint_min} min; outer fold: {row.fold}; case: {esc(row.case_id)}<br>'
            + "<br>".join(f"{k}: {esc(row[k])}" for k in [*ACQUISITION_IDS, "batch_id", "mapping_status", "metadata_row", "optical_modality", "evaluation_protocol"])
            + f'<br>Test content seen in training: {row.test_content_seen_in_train}; in inner validation: {row.test_content_seen_in_validation}'
              f'<br>Source collection (not batch): {esc(row.source_collection_id)}<br>Source: {esc(row.image_path)}'
              f'<br>SHA256: {esc(row.audited_file_sha256)}</p></article>')


def html_document(title, body):
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(title)}</title><style>body{{font-family:Georgia,serif;background:#f3f1e9;color:#202e2b;max-width:1200px;margin:auto;padding:20px;}}'
            'article{background:white;padding:18px;margin:16px 0;overflow-wrap:anywhere;border:1px solid #bbb;}'
            'img{max-width:48%;height:auto;vertical-align:top;margin:1%;}.warning{color:#a52318;}section{border-top:3px solid #345;}a{color:#175a62;}'
            '@media(max-width:600px){img{max-width:98%;}}</style>'
            f'<h1>{html.escape(title)}</h1><p>Verified outer-test predictions; original images are linked byte-for-byte. '
            'Left: loaded full field resized for display. Right: 224 x 224 evaluation input before ImageNet normalization. '
            'No additional contrast enhancement; the evaluation loader may scale non-uint8 TIFFs. '
            'Recorded acquisition IDs are not biological-replicate counts. UNKNOWN is not a matched batch.</p>'
            '<p><a href="all_error_cases.csv">All error records</a> | <a href="confusion_exact_counts.csv">Full raw confusion counts</a> | '
            '<a href="comparison_gallery.html">Matched controls</a> | <a href="conflict_gallery.html">Conflicting-label pairs</a></p>' + body + '</html>')


def write_galleries(df, pairs, out):
    errors = df.loc[df.any_error].sort_values(["class_name", "timepoint_min", "fold", "case_id"])
    (out / "error_gallery.html").write_text(html_document(out.name + f": all {len(errors)} error images",
                                                         "\n".join(case_card(row) for _, row in errors.iterrows())))
    lookup, sections = df.set_index("case_id", drop=False), []
    for pair in pairs.to_dict("records"):
        body = "<section><h2>Deterministic median-confidence error and available correct controls</h2>"
        for label, key in (("Error", "error_case_id"), ("Same recorded condition", "true_condition_control_id"),
                           ("Predicted condition", "predicted_condition_control_id")):
            body += f"<h3>{label}</h3>"
            if key != "error_case_id":
                body += "<p>Matched on: " + html.escape(pair[key.replace("_control_id", "_matched_on")]) + ". Conflicting-label controls excluded.</p>"
            body += case_card(lookup.loc[pair[key]]) if pair[key] else "<p>No control satisfies the recorded constraints; no silent relaxation.</p>"
        sections.append(body + "</section>")
    (out / "comparison_gallery.html").write_text(html_document(out.name + ": DH5alpha comparisons", "\n".join(sections)))
    sections = []
    conflicts = df.loc[df.conflicting_condition_labels]
    for digest, group in conflicts.groupby("audited_file_sha256", sort=True):
        sections.append("<section><h2>Identical file SHA256: " + digest + "</h2>" +
                        "\n".join(case_card(row) for _, row in group.sort_values("class_name").iterrows()) + "</section>")
    (out / "conflict_gallery.html").write_text(html_document(out.name + ": conflicting-label content",
                                                            "\n".join(sections) or "<p>No known cross-label byte duplicates in this analysis population.</p>"))


def focus_summary(df):
    focus = df.loc[df.class_name.eq(FOCUS + " + Kanamycin")]
    errors = focus.loc[focus.exact_error]
    return {"n": len(df), "n_exact_errors": int(df.exact_error.sum()), "n_error_images": int(df.any_error.sum()),
            "n_known_session": int(df.acquisition_session_id.ne(UNKNOWN).sum()),
            "n_conflicting_label_images": int(df.conflicting_condition_labels.sum()),
            "n_test_content_seen_train": int(df.test_content_seen_in_train.sum()),
            "n_test_content_seen_validation": int(df.test_content_seen_in_validation.sum()),
            "dh5_kan_n": len(focus), "dh5_kan_correct": int((~focus.exact_error).sum()),
            "dh5_kan_errors": len(errors), "dh5_kan_errors_at_30min": int(errors.timepoint_min.eq(30).sum()),
            "dh5_kan_errors_with_conflicting_labels": int(errors.conflicting_condition_labels.sum()),
            "dh5_kan_to_dh5_amp": int(focus.pred_class_name.eq(FOCUS + " + Ampicillin").sum()),
            "dh5_kan_cross_strain_errors": int(focus.strain_error.sum())}


def audit_run(name, run_dir, task, out, mapping, content, evidence):
    print(f"[AUDIT] {name}: verifying frozen predictions", flush=True)
    df, splits, config, settings = load_outer_predictions(run_dir, task, evidence)
    original_ids = df[["image_path", *ACQUISITION_IDS]].copy() if all(c in df for c in ACQUISITION_IDS) else None
    df = attach_acquisition_metadata(df.drop(columns=[*META_COLUMNS, "strain", "treatment", "timepoint", "filename"], errors="ignore"), mapping)
    if original_ids is not None:
        for column in ACQUISITION_IDS:
            if not df[column].equals(original_ids[column]):
                raise ValueError("Current metadata differs from frozen grouping metadata")
    for column in ("batch_id", "biological_replicate_id", "optical_modality"):
        df[column] = UNKNOWN
    df["source_collection_id"] = df.source_root + "::" + df.source_folder
    df = add_decisions(df, settings["class_names"])
    df["evaluation_protocol"] = config["protocol"]
    df = annotate_partition_content(attach_content(df, content), splits)
    if config["protocol"] == SENSITIVITY_PROTOCOL and (df.test_content_seen_in_train.any() or df.test_content_seen_in_validation.any()):
        raise ValueError("Exact content crossed protected sensitivity partitions")
    folder = out / name
    folder.mkdir()
    export_tables(df, settings["class_names"], folder)
    pairs = select_controls(df)
    pairs.to_csv(folder / "comparison_pairs.csv", index=False)
    # Also archive correct copies of conflicting content, not just mistaken copies.
    requests = pd.concat([pairs, pd.DataFrame({"true_condition_control_id": df.loc[df.conflicting_condition_labels, "case_id"],
                                             "predicted_condition_control_id": ""})], ignore_index=True).fillna("")
    df = export_assets(df, requests, folder)
    archived = df.asset_status.eq("ok")
    if not df.loc[archived, "original_sha256"].eq(df.loc[archived, "audited_file_sha256"]).all():
        raise ValueError("Archived images differ from freshly verified content")
    if df.loc[df.any_error | df.conflicting_condition_labels, "asset_status"].ne("ok").any():
        raise ValueError("Required image evidence is incomplete")
    if df.asset_status.eq("failed").any():
        raise ValueError("A selected comparison control failed image export")
    df.to_csv(folder / "all_predictions_with_identifiers.csv", index=False)
    df.loc[df.any_error].to_csv(folder / "all_error_cases.csv", index=False)
    df.loc[df.conflicting_condition_labels].sort_values(["audited_file_sha256", "class_name"]).to_csv(folder / "conflicting_label_images.csv", index=False)
    focus = df.loc[df.class_name.eq(FOCUS + " + Kanamycin")]
    focus.to_csv(folder / "DH5alpha_kanamycin_all_predictions.csv", index=False)
    performance_table(focus, ["timepoint_min"]).to_csv(folder / "DH5alpha_kanamycin_by_time.csv", index=False)
    write_galleries(df, pairs, folder)
    summary = {"analysis": name, "run_dir": str(run_dir), "task": task, "protocol": config["protocol"],
               **focus_summary(df), "n_assets": int(archived.sum()), "n_asset_failures": 0,
               "protocol_limitations": config.get("limitations", LIMITATIONS),
               "n_treatment_decision_rule_disagreements": int(df.pred_treatment_marginal.ne(df.pred_treatment_top_class).sum())}
    write_json(folder / "summary.json", summary)
    print(f"[DONE] {name}: {summary['n_error_images']} error images, {summary['n_assets']} archived assets", flush=True)
    return df, summary


def write_presentation_review(root, out, evidence):
    path = root / "Paper/paper_draft.md"
    issues = []
    patterns = {"unverified_optical_terminology": r"phase.contrast|bright.?field|transmitted.light",
                "pipeline_preprocessing_claim": r"without.*segmentation|without a segmentation step|no (?:image )?preprocessing",
                "unsupported_mechanistic_attribution": r"lysis|convergen|rather than.*mislabel|biological edge cases"}
    if path.exists():
        record_source(path, evidence)
        for n, line in enumerate(path.read_text().splitlines(), 1):
            for issue, pattern in patterns.items():
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append({"path": str(path), "line": n, "review_category": issue, "text": line})
    pd.DataFrame(issues, columns=["path", "line", "review_category", "text"]).to_csv(out / "presentation_review.csv", index=False)
    (out / "presentation_corrections.md").write_text(
        "# Evidence-bounded presentation corrections\n\n"
        "No manuscript edits were applied. presentation_review.csv identifies passages needing review.\n\n"
        "## Optical terminology\n\n"
        "The workbook identifies the microscope but does not establish brightfield versus phase-contrast. "
        "Do not substitute one for the other. Pending acquisition confirmation, use 'microscopy images' "
        "and explicitly mark the contrast method as unverified. Use 'transmitted-light' as an umbrella "
        "term only if the acquisition geometry is confirmed; it does not specify the contrast mode.\n\n"
        "## Suggested preprocessing paragraph\n\n"
        "The classifier operates on whole-field images rather than manually engineered per-cell features. "
        "Before classification, images undergo segmentation-based quality control and exclusion according "
        "to the recorded QC policy; unavailable segmentation results constitute a separate exclusion category. "
        "Images are decoded and converted to three-channel RGB, with grayscale channels replicated. "
        "Non-uint8 TIFF arrays undergo the loader's 1st-99th-percentile intensity scaling. Inputs are bilinearly "
        "resized to 224 x 224 pixels, converted to tensors, and normalized with ImageNet channel statistics "
        "(mean 0.485/0.456/0.406; SD 0.229/0.224/0.225). Random horizontal and vertical flips are used only "
        "during training. Segmentation masks are not classifier input channels but are used for QC and "
        "interpretability. Thus, the workflow does not require manual per-cell feature engineering, "
        "but it is not preprocessing-free or segmentation-free.\n\n"
        "## Discussion constraints\n\n"
        "Use the analysis-specific counts in error_case_audit.md; do not mix 13-class, 15-class or cleaned "
        "populations. Separate exact condition errors, strain errors and treatment errors. Discuss the time "
        "distribution, conflicting-label exact copies, and preparation sensitivity before proposing a "
        "morphological mechanism. Visual resemblance cannot resolve contradictory treatment labels. "
        "Do not call archived error examples proof of lysis, rounded cells, biological convergence, "
        "or absence of acquisition confounding without independent measurements.\n")


def write_report(results, out):
    lines = ["# Corrected outer-test error evidence", "", "No new inference, model selection, relabeling, "
             "QC exclusions or manuscript edits were performed. Each analysis is kept separate.", "",
             *["- " + item for item in LIMITATIONS], "", "## Coverage", ""]
    summaries = pd.DataFrame([s for _, s in results.values()])
    columns = ["analysis", "n", "n_exact_errors", "n_error_images", "n_assets", "n_known_session",
               "n_conflicting_label_images", "n_test_content_seen_train", "n_test_content_seen_validation"]
    lines.extend([markdown_table(summaries[columns]), "",
                  "Confusion matrices use recorded labels as rows and predictions as columns, with integer zeros "
                  "preserved. Each observed stratum has the full label-pair grid. Absent strata are not zero-error "
                  "strata. Treatment-marginal argmax after summing class probabilities and treatment-of-top-class "
                  "argmax are exported separately. Pooled per-class metrics are not mean fold metrics.", "",
                  "Every exact-label OR marginal-treatment error is archived, along with all conflicting-label "
                  "copies and deterministic matched correct controls. The comparison rule selects the lower-median "
                  "confidence error within condition/prediction/time/fold/preparation/session/conflict strata. "
                  "Controls exclude known label conflicts; known matching constraints are never relaxed. "
                  "Recorded sessions, preparations and slides are shown alongside UNKNOWN batch IDs. "
                  "No optical contrast mode or biological replication is inferred from the image appearance.", ""])
    for name, (df, summary) in results.items():
        focus = df.loc[df.class_name.eq(FOCUS + " + Kanamycin")]
        cm = matrix(df, "class_name", "pred_class_name", sorted(df.class_name.unique()))
        metrics = per_class_metrics(cm).set_index("class_name").loc[FOCUS + " + Kanamycin"]
        time = performance_table(focus, ["timepoint_min"])
        time = time[["timepoint_min", "n", "n_exact_errors", "n_treatment_errors", "n_strain_errors", "exact_accuracy"]]
        lines += [f"## {name}", "", f"Frozen protocol: `{summary['protocol']}`; task `{summary['task']}`.", "",
                  f"[All error images]({name}/error_gallery.html) | [Raw confusion matrix]({name}/confusion_exact_counts.csv) | "
                  f"[Confusion figure]({name}/confusion_exact_counts.png) | [Matched controls]({name}/comparison_gallery.html) | "
                  f"[Contradictory-label pairs]({name}/conflict_gallery.html)", "",
                  f"DH5alpha-kanamycin: **{summary['dh5_kan_correct']}/{summary['dh5_kan_n']} exact correct "
                  f"({summary['dh5_kan_correct'] / summary['dh5_kan_n']:.2%})**; pooled precision {metrics.precision:.4f}, "
                  f"recall {metrics.recall:.4f}, F1 {metrics.f1:.4f}. "
                  f"Of {summary['dh5_kan_errors']} exact errors, {summary['dh5_kan_errors_at_30min']} occur at 30 min, "
                  f"{summary['dh5_kan_errors_with_conflicting_labels']} involve known conflicting-label content, "
                  f"and {summary['dh5_kan_cross_strain_errors']} change the strain label. "
                  f"{summary['dh5_kan_to_dh5_amp']} images are assigned specifically to DH5alpha-ampicillin. "
                  "These categories overlap and must not be summed.", "", markdown_table(time.round(4)), "",
                  "Full raw destinations:", "",
                  markdown_table(focus.pred_class_name.value_counts().rename_axis("predicted_condition").reset_index(name="count")), "",
                  "The counts establish which recorded labels, times and acquisitions concentrate the errors; "
                  "they do not establish the biological mechanism. In the original cohorts, contradictory-label "
                  "byte-identical content prevents treating all errors as phenotype ambiguity. The cleaned "
                  "analyses are separately retrained sensitivity experiments, not post hoc rescoring of the "
                  "original cohort or an isolated effect of correcting labels.", ""]
    lines += ["## Presentation", "", "See presentation_corrections.md for proposed preprocessing wording and "
              "terminology limits, and presentation_review.csv for manuscript locations. Known sources of error "
              "include unresolved provenance and acquisition dependence; do not replace these findings with an "
              "unsupported claim that failures primarily reflect late-stage biological convergence.", ""]
    (out / "error_case_audit.md").write_text("\n".join(lines))
    summaries.to_csv(out / "analysis_summary.csv", index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--analysis", nargs=3, action="append", metavar=("NAME", "RUN_DIR", "TASK"), required=True)
    parser.add_argument("--metadata_audit", type=Path, required=True)
    parser.add_argument("--hash_inventory", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8:
        parser.error("Use 1-8 hash workers")
    names = [r[0] for r in args.analysis]
    if len(set(names)) != len(names) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in names):
        parser.error("Analysis names must be unique simple directory names")
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    evidence = {}
    for path in (Path(__file__), ROOT / "Code/Training/Rebuttal/audit_error_cases.py",
                 ROOT / "Code/Training/Rebuttal/audit_dataset_accounting.py", ROOT / "Code/Training/Rebuttal/audit_grouped_cv_feasibility.py",
                 ROOT / "Code/Training/Rebuttal/preparation_sensitivity.py", ROOT / "Code/Training/Rebuttal/summarize_nested_cv.py",
                 ROOT / "Code/Training/Rebuttal/validation.py", ROOT / "Code/utils/image_utils.py"):
        record_source(path, evidence)
    mapping = load_metadata(args.metadata_audit.resolve(), evidence)
    content = load_content_inventory(ROOT, args.hash_inventory.resolve(), evidence, args.workers)
    results = {}
    for name, directory, task in args.analysis:
        results[name] = audit_run(name, Path(directory).resolve(), task, out, mapping, content, evidence)
    write_presentation_review(ROOT, out, evidence)
    write_report(results, out)
    for path, expected in evidence.items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Source evidence changed during export: {path}")
    write_json(out / "verification.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
               "all_checks_passed": True, "analyses": {name: s for name, (_, s) in results.items()},
               "scid_used": False, "n_source_images_rehashed": len(content),
               "command": sys.argv, "python": sys.version,
               "source_evidence": [{"path": p, "sha256": h} for p, h in evidence.items()], "limitations": LIMITATIONS})
    print(f"[DONE] Verified evidence package: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
