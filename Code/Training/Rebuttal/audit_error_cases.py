#!/usr/bin/env python3
"""Export raw legacy confusion counts and traceable, unaltered error evidence.

No model selection or inference is performed. These predictions used the outer
fold for checkpoint selection and must not be called corrected nested-CV results.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Code"))
from utils.image_utils import load_image_as_pil_rgb
from validation import sha256, write_json

UNKNOWN = "UNKNOWN"
FOCUS = "E.coli5-alpha"
TREATMENTS = ["Ampicillin", "CND", "Kanamycin"]
PROTOCOL = "legacy_outer_fold_used_for_checkpoint_selection"
IDENTIFIERS = ["image_path", "image_uid", "group_id", "field_group_id", "class_name",
               "class_id", "canonical_strain", "canonical_treatment", "timepoint_min",
               "source_root", "source_folder"]
ACQUISITION = ["batch_id", "acquisition_session_id", "biological_replicate_id", "optical_modality"]


def validate_predictions(pred, manifest, settings):
    """Reject ambiguous labels, missing fields, and incomplete/duplicated OOF data."""
    required = IDENTIFIERS + ["y_true", "y_pred", "pred_class_name", "fold"]
    if not set(required) <= set(pred) or pred[required].isna().any().any():
        raise ValueError("Predictions lack required identifiers or labels")
    kept = manifest.loc[pd.to_numeric(manifest.drop_from_training).eq(0)]
    if kept.image_path.duplicated().any() or pred.image_path.duplicated().any():
        raise ValueError("Duplicate image paths")
    if pred.image_uid.duplicated().any():
        raise ValueError("Duplicate image UIDs")
    if set(pred.image_path) != set(kept.image_path) or len(pred) != settings["n_samples"]:
        raise ValueError("Prediction population differs from retained manifest/training population")
    ref = kept.set_index("image_path").loc[pred.image_path].reset_index()
    for col in IDENTIFIERS:
        if not pred[col].reset_index(drop=True).astype(str).equals(ref[col].astype(str)):
            raise ValueError(f"Manifest/prediction disagreement: {col}")
    names = settings["class_names"]
    for col in ["y_true", "y_pred", "class_id", "fold"]:
        values = pd.to_numeric(pred[col]).to_numpy()
        if not np.isfinite(values).all() or not np.array_equal(values, values.astype(int)):
            raise ValueError(f"Noninteger labels: {col}")
    if set(pred.fold) != set(range(1, settings["n_folds"] + 1)):
        raise ValueError("Missing or unexpected fold")
    if pred.groupby("group_id").fold.nunique().gt(1).any():
        raise ValueError("Group occurs in more than one held-out fold")
    mapping = dict(enumerate(names))
    if not pred.y_true.equals(pred.class_id) or not pred.y_true.map(mapping).equals(pred.class_name):
        raise ValueError("True class ID/name mismatch")
    if not pred.y_pred.map(mapping).equals(pred.pred_class_name):
        raise ValueError("Predicted class ID/name mismatch")
    probabilities = pred[["prob_" + name for name in names]].to_numpy(float)
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or (probabilities > 1).any() or not np.allclose(probabilities.sum(1), 1, atol=1e-5)):
        raise ValueError("Invalid probabilities")
    if not np.array_equal(probabilities.argmax(1), pred.y_pred.to_numpy()):
        raise ValueError("Predictions disagree with probability argmax")
    if pred.class_name.value_counts().to_dict() != settings["class_counts"]:
        raise ValueError("Training class counts disagree")
    return pred.copy().reset_index(drop=True)


def attach_acquisition(df, metadata=None):
    """Only an explicit, provenance-bearing exact-path mapping can supply batches."""
    out = df.copy()
    for col in ACQUISITION:
        out[col] = UNKNOWN
    out["acquisition_metadata_source"] = "not_provided"
    if metadata is not None:
        required = ["image_path", "metadata_source"] + ACQUISITION
        if not set(required) <= set(metadata) or metadata.image_path.duplicated().any():
            raise ValueError("Acquisition metadata needs unique exact paths and all documented columns")
        if metadata[required].isna().any().any() or metadata[required].astype(str).apply(lambda s: s.str.strip().eq("")).any().any():
            raise ValueError("Use UNKNOWN explicitly for unavailable acquisition identifiers")
        matched = out.image_path.isin(metadata.image_path)
        lookup = metadata.set_index("image_path").loc[out.loc[matched, "image_path"]]
        for col in ACQUISITION:
            out.loc[matched, col] = lookup[col].to_numpy()
        out.loc[matched, "acquisition_metadata_source"] = lookup.metadata_source.to_numpy()
    out["source_collection_id"] = out.source_root + "::" + out.source_folder
    out["source_collection_is_verified_batch"] = False
    return out


def add_decisions(df, names):
    out = df.copy()
    probabilities = out[["prob_" + n for n in names]].to_numpy(float)
    out["predicted_class_probability"] = probabilities[np.arange(len(out)), out.y_pred.to_numpy(int)]
    out["true_class_probability"] = probabilities[np.arange(len(out)), out.y_true.to_numpy(int)]
    for treatment in TREATMENTS:
        cols = ["prob_" + name for name in names if name.rsplit(" + ", 1)[1] == treatment]
        if not cols:
            raise ValueError(f"Missing treatment: {treatment}")
        out["treatment_probability_" + treatment] = out[cols].sum(axis=1)
    out["true_treatment"] = out.canonical_treatment
    out["pred_treatment_marginal"] = out[["treatment_probability_" + t for t in TREATMENTS]].idxmax(axis=1).str.removeprefix("treatment_probability_")
    out["pred_treatment_top_class"] = out.pred_class_name.str.rsplit(" + ", n=1).str[-1]
    out["pred_strain"] = out.pred_class_name.str.rsplit(" + ", n=1).str[0]
    out["exact_error"] = out.y_true.ne(out.y_pred)
    out["treatment_error"] = out.true_treatment.ne(out.pred_treatment_marginal)
    out["strain_error"] = out.canonical_strain.ne(out.pred_strain)
    out["any_error"] = out.exact_error | out.treatment_error
    out["evaluation_protocol"] = PROTOCOL
    out["case_id"] = out.image_uid.map(lambda uid: hashlib.sha256(uid.encode()).hexdigest()[:16])
    if out.case_id.duplicated().any():
        raise ValueError("Case ID hash collision")
    return out


def attach_duplicate_evidence(df, audited):
    required = ["image_path", "image_uid", "class_name", "fold", "original_sha256", "hash_status",
                "n_images", "has_label_conflict", "spans_heldout_folds"]
    if not set(required) <= set(audited) or audited.image_path.duplicated().any():
        raise ValueError("Invalid full-cohort duplicate audit")
    if set(audited.image_path) != set(df.image_path):
        raise ValueError("Duplicate audit population does not match prediction cohort")
    aligned = audited.set_index("image_path").loc[df.image_path].reset_index()
    for col in ["image_uid", "class_name", "fold"]:
        if not df[col].reset_index(drop=True).equals(aligned[col]):
            raise ValueError(f"Duplicate audit metadata mismatch: {col}")
    archived = df.asset_status.eq("ok").to_numpy()
    if not np.array_equal(df.loc[archived, "original_sha256"].to_numpy(), aligned.loc[archived, "original_sha256"].to_numpy()):
        raise ValueError("Archived original differs from full-cohort hash audit")
    out = df.copy()
    for target, source in {"audited_file_sha256": "original_sha256", "byte_duplicate_group_size": "n_images",
                           "conflicting_condition_labels": "has_label_conflict",
                           "duplicate_spans_folds": "spans_heldout_folds", "cohort_hash_status": "hash_status"}.items():
        out[target] = aligned[source].to_numpy()
    return out


def matrix(df, true, predicted, labels):
    cm = pd.crosstab(df[true], df[predicted]).reindex(index=labels, columns=labels, fill_value=0).astype(int)
    cm.index.name = "true_label"
    cm.columns.name = "predicted_label"
    if cm.to_numpy().sum() != len(df):
        raise ValueError("Confusion counts do not conserve population")
    return cm


def confusion_long(df, dimensions, true, predicted, labels):
    pieces = []
    groups = df.groupby(dimensions, dropna=False, sort=True) if dimensions else [((), df)]
    for key, sub in groups:
        key = key if isinstance(key, tuple) else (key,)
        cm = matrix(sub, true, predicted, labels)
        long = cm.stack().rename("count").reset_index()
        long["true_support"] = long.true_label.map(cm.sum(axis=1))
        long["stratum_n"] = len(sub)
        for col, value in zip(dimensions, key):
            long[col] = value
        pieces.append(long)
    return pd.concat(pieces, ignore_index=True)


def performance_table(df, dimensions):
    table = df.groupby(dimensions, dropna=False, sort=True).agg(
        n=("image_path", "size"), n_exact_errors=("exact_error", "sum"),
        n_treatment_errors=("treatment_error", "sum"), n_strain_errors=("strain_error", "sum"),
        median_predicted_probability=("predicted_class_probability", "median"),
    ).reset_index()
    table["exact_accuracy"] = 1 - table.n_exact_errors / table.n
    table["treatment_accuracy"] = 1 - table.n_treatment_errors / table.n
    return table


def per_class_metrics(cm):
    tp = np.diag(cm.to_numpy())
    support = cm.sum(axis=1).to_numpy()
    predicted_n = cm.sum(axis=0).to_numpy()
    precision = np.divide(tp, predicted_n, out=np.zeros(len(tp), dtype=float), where=predicted_n > 0)
    recall = np.divide(tp, support, out=np.zeros(len(tp), dtype=float), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(tp), dtype=float), where=(precision + recall) > 0)
    return pd.DataFrame({"class_name": cm.index, "support": support, "predicted_n": predicted_n,
                         "tp": tp, "fp": predicted_n - tp, "fn": support - tp,
                         "precision": precision, "recall": recall, "f1": f1})


def render_counts(cm, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, len(cm) * .8), max(5, len(cm) * .7)))
    counts = cm.to_numpy()
    ax.imshow(counts, cmap="Blues", vmin=0)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, str(counts[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if counts[i, j] > counts.max() / 2 else "black")
    labels = [n.replace("E.coli5-alpha", "DH5alpha") for n in cm.index]
    ax.set_xticks(range(len(cm)), labels, rotation=65, ha="right", fontsize=8)
    ax.set_yticks(range(len(cm)), labels, fontsize=8)
    ax.set(xlabel="Predicted label", ylabel="Recorded label", title=title + "\nRaw image counts; legacy CV")
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(path.with_suffix("." + extension), dpi=170)
    plt.close(fig)


def select_comparisons(df):
    """Median-confidence error per contrast/time/fold; deterministic same-stratum controls."""
    focus = df.loc[df.canonical_strain.eq(FOCUS) & df.true_treatment.isin(["Ampicillin", "Kanamycin"])]
    errors = focus.loc[focus.any_error]
    keys = ["class_name", "pred_class_name", "timepoint_min", "fold", "source_collection_id"]
    rows = []
    for _, sub in errors.groupby(keys, sort=True):
        ordered = sub.sort_values(["predicted_class_probability", "case_id"])
        error = ordered.iloc[(len(ordered) - 1) // 2]
        result = {"error_case_id": error.case_id, "stratum_error_n": len(sub),
                  "error_selection": "lower_median_confidence_within_contrast_time_fold_source",
                  "true_condition_control_id": "", "predicted_condition_control_id": ""}
        for role, condition in [("true_condition", error.class_name), ("predicted_condition", error.pred_class_name)]:
            control = df.loc[~df.any_error & df.class_name.eq(condition)
                             & df.timepoint_min.eq(error.timepoint_min) & df.fold.eq(error.fold)
                             & df.source_root.eq(error.source_root)]
            if role == "true_condition":
                control = control.loc[control.source_collection_id.eq(error.source_collection_id)]
            if error.batch_id != UNKNOWN:
                control = control.loc[control.batch_id.eq(error.batch_id)]
            if len(control):
                chosen = control.sort_values("case_id").iloc[0]
                result[role + "_control_id"] = chosen.case_id
            result[role + "_candidate_n"] = len(control)
        rows.append(result)
    columns = ["error_case_id", "stratum_error_n", "error_selection", "true_condition_control_id",
               "predicted_condition_control_id", "true_condition_candidate_n", "predicted_condition_candidate_n"]
    return pd.DataFrame(rows, columns=columns)


def export_assets(df, comparisons, out):
    """Copy originals byte-for-byte; previews use the existing training image loader."""
    ids = set(df.loc[df.any_error, "case_id"])
    for col in ["true_condition_control_id", "predicted_condition_control_id"]:
        ids.update(v for v in comparisons[col] if v)
    for folder in ["originals", "previews", "inputs_224"]:
        (out / folder).mkdir()
    records = []
    chosen = df.loc[df.case_id.isin(ids)].sort_values("case_id")

    def export_one(row):
        source = Path(row.image_path)
        record = {"case_id": row.case_id, "image_path": str(source), "asset_status": "ok"}
        try:
            original = Path("originals") / (row.case_id + source.suffix.lower())
            shutil.copyfile(source, out / original)
            digest = sha256(source)
            if digest != sha256(out / original):
                raise ValueError("Copied original checksum mismatch")
            image = load_image_as_pil_rgb(str(source))
            model_input = image.resize((224, 224), Image.Resampling.BILINEAR)
            input_path = Path("inputs_224") / (row.case_id + ".png")
            model_input.save(out / input_path)
            preview_path = Path("previews") / (row.case_id + ".png")
            preview = image.copy()
            preview.thumbnail((512, 512), Image.Resampling.LANCZOS)
            preview.save(out / preview_path)
            record.update(original_file=str(original), preview_file=str(preview_path),
                          model_input_file=str(input_path), original_sha256=digest,
                          width=image.width, height=image.height,
                          model_input_pixels_sha256=hashlib.sha256(np.asarray(model_input).tobytes()).hexdigest())
        except (OSError, ValueError) as error:
            record.update(asset_status="failed", asset_error=str(error))
        return record

    with ThreadPoolExecutor(max_workers=4) as executor:
        for index, record in enumerate(executor.map(export_one, chosen.itertuples()), 1):
            records.append(record)
            if index % 200 == 0:
                print(f"[IMAGES] {out.name}: {index}/{len(chosen)}", flush=True)
    assets = pd.DataFrame(records, columns=["case_id", "image_path", "asset_status", "original_file", "preview_file",
                                          "model_input_file", "original_sha256", "width", "height",
                                          "model_input_pixels_sha256", "asset_error"])
    assets.to_csv(out / "image_assets.csv", index=False)
    return df.merge(assets.drop(columns="image_path"), on="case_id", how="left", validate="one_to_one")


def write_gallery(df, out):
    def esc(value):
        return html.escape(str(value), quote=True)

    rows = df.loc[df.any_error].sort_values(["class_name", "timepoint_min", "fold", "case_id"])
    sections = []
    for row in rows.itertuples():
        warning = ""
        if getattr(row, "conflicting_condition_labels", False) == True:
            warning = '<p style="color:#a32020"><strong>UNRESOLVED LABEL CONFLICT: byte-identical source files carry different condition labels.</strong></p>'
        pictures = "<p>Image unavailable; see image_assets.csv.</p>"
        if row.asset_status == "ok":
            pictures = (f'<a href="{esc(row.original_file)}"><img loading="lazy" src="{esc(row.preview_file)}" alt="Full-field preview"></a>'
                        f'<img loading="lazy" src="{esc(row.model_input_file)}" alt="224-pixel model input before normalization">')
        sections.append(f'<article id="{row.case_id}"><h2>Recorded: {esc(row.class_name)}</h2>{warning}<p>Predicted: {esc(row.pred_class_name)}; '
                        f'p={row.predicted_class_probability:.4f}; treatment marginal: {esc(row.pred_treatment_marginal)}</p>'
                        f'<p>{row.timepoint_min} min; fold {row.fold}; batch: {esc(row.batch_id)}; '
                        f'session: {esc(row.acquisition_session_id)}</p>{pictures}<p>Case: {row.case_id}<br>'
                        f'Collection (not batch): {esc(row.source_collection_id)}<br>Field group: {esc(row.field_group_id)}<br>'
                        f'UID: {esc(row.image_uid)}<br>Source: {esc(row.image_path)}</p></article>')
    content = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Complete error-case evidence</title>'
               '<style>body{font-family:Georgia,serif;max-width:1100px;margin:2rem auto;padding:1rem;background:#f4f2eb;}'
               'article{background:white;padding:1rem;margin:1rem 0;border:1px solid #bbb;overflow-wrap:anywhere;}'
               'img{max-width:48%;height:auto;vertical-align:top;margin-right:1%;}h2{font-size:1.15rem;}</style>'
               f'<h1>{esc(out.name)}: all {len(rows)} error cases</h1><p>Legacy CV, not corrected outer-test results. '
               'Left: full field resized for display; right: 224 x 224 model input before channel normalization. '
               'Click the full-field image for the byte-identical original. No display contrast enhancement. '
               'All exact-label OR marginal-treatment errors are included. Missing batch/session values are UNKNOWN.</p>'
               '<p><a href="all_error_cases.csv">All error metadata and scores</a> | '
               '<a href="comparison_pairs.csv">Deterministic comparison selections</a></p>' + "\n".join(sections) + '</html>')
    (out / "error_gallery.html").write_text(content)


def render_comparisons(df, pairs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = df.set_index("case_id")
    files = []
    for page, offset in enumerate(range(0, len(pairs), 4), 1):
        batch = pairs.iloc[offset:offset + 4]
        height = 5 * len(batch) + 1.2
        fig, axes = plt.subplots(len(batch), 3, figsize=(13, height), squeeze=False)
        for i, pair in enumerate(batch.itertuples()):
            for j, (role, case_id) in enumerate([("Error", pair.error_case_id),
                                               ("Correct: same true condition", pair.true_condition_control_id),
                                               ("Correct: predicted condition", pair.predicted_condition_control_id)]):
                ax = axes[i, j]
                ax.axis("off")
                if not case_id:
                    ax.text(.5, .5, "No matched correct example", ha="center", transform=ax.transAxes)
                    continue
                row = lookup.loc[case_id]
                if row.asset_status == "ok":
                    with Image.open(out / row.preview_file) as picture:
                        ax.imshow(picture)
                else:
                    ax.text(.5, .5, "Image unavailable", ha="center", transform=ax.transAxes)
                label = row.class_name.replace("E.coli5-alpha", "DH5alpha")
                prediction = row.pred_class_name.replace("E.coli5-alpha", "DH5alpha")
                conflict = " | LABEL CONFLICT" if row.get("conflicting_condition_labels", False) == True else ""
                ax.set_title(f"{role}{conflict}\nLabel: {label}\nPred: {prediction}\n"
                             f"Marginal treatment: {row.pred_treatment_marginal}\n"
                             f"{row.timepoint_min} min | fold {row.fold} | batch {row.batch_id}\n"
                             f"{row.source_root} | {case_id}", fontsize=8)
        fig.suptitle(f"{out.name} / legacy CV / no extra display contrast\n"
                     "Controls matched on time, fold and source root; batch UNKNOWN is not a batch match.", fontsize=10)
        fig.subplots_adjust(left=.02, right=.98, bottom=.03, top=1 - 2 / height, hspace=.7, wspace=.12)
        name = f"DH5alpha_comparisons_{page:03d}.png"
        fig.savefig(out / name, dpi=140)
        plt.close(fig)
        files.append(name)
    return files


def export_confusions(df, names, out):
    modes = [("exact", "class_name", "pred_class_name", names),
             ("treatment_marginal", "true_treatment", "pred_treatment_marginal", TREATMENTS),
             ("treatment_top_class", "true_treatment", "pred_treatment_top_class", TREATMENTS)]
    for mode, true, predicted, labels in modes:
        cm = matrix(df, true, predicted, labels)
        cm.to_csv(out / f"confusion_{mode}_counts.csv")
        per_class_metrics(cm).to_csv(out / f"metrics_{mode}_pooled.csv", index=False)
        fold_metrics = []
        for fold, sub in df.groupby("fold"):
            metrics = per_class_metrics(matrix(sub, true, predicted, labels))
            metrics["fold"] = fold
            fold_metrics.append(metrics)
        pd.concat(fold_metrics).to_csv(out / f"metrics_{mode}_by_fold.csv", index=False)
        if mode != "treatment_top_class":
            render_counts(cm, out.name + " / " + mode, out / f"confusion_{mode}_counts")
        for suffix, dimensions in [("fold", ["fold"]), ("time", ["timepoint_min"]),
                                   ("condition_time_source_batch", ["canonical_strain", "canonical_treatment", "timepoint_min",
                                                                    "source_collection_id", "batch_id", "acquisition_session_id"]),
                                   ("fold_time_source_batch", ["fold", "timepoint_min", "source_collection_id", "batch_id", "acquisition_session_id"])]:
            confusion_long(df, dimensions, true, predicted, labels).to_csv(out / f"confusion_{mode}_by_{suffix}.csv", index=False)
    for suffix, dimensions in [("condition", ["class_name"]), ("condition_fold", ["class_name", "fold"]),
                               ("condition_time", ["class_name", "timepoint_min"]),
                               ("condition_time_source_batch_fold", ["class_name", "timepoint_min", "source_collection_id",
                                                                     "batch_id", "acquisition_session_id", "fold"])]:
        performance_table(df, dimensions).to_csv(out / f"performance_by_{suffix}.csv", index=False)
    for flag in ["exact_error", "treatment_error"]:
        df.groupby(["class_name", "timepoint_min", flag], dropna=False)[
            ["predicted_class_probability", "true_class_probability", "seg_qc_total_cells", "seg_qc_pass_fraction"]
        ].agg(["count", "median", "mean", "min", "max"]).to_csv(out / f"descriptive_qc_by_{flag}.csv")


def audit_cohort(root, cohort, out, metadata, duplicate_dir=None):
    model = root / f"Model/Benchmark_Treated_Condition_{cohort}class_QC/efficientnet_b0"
    settings = json.loads((model / "training_args.json").read_text())
    prediction_path = model / "crossval_predictions.csv"
    manifest_path = root / settings["manifest_csv"]
    pred = validate_predictions(pd.read_csv(prediction_path), pd.read_csv(manifest_path), settings)
    names = settings["class_names"]
    df = add_decisions(attach_acquisition(pred, metadata), names)
    out.mkdir()
    export_confusions(df, names, out)
    pairs = select_comparisons(df)
    pairs.to_csv(out / "comparison_pairs.csv", index=False)
    df = export_assets(df, pairs, out)
    if duplicate_dir:
        df = attach_duplicate_evidence(df, pd.read_csv(duplicate_dir / f"treated_{cohort}class/all_predictions_with_hashes.csv"))
    # This screen is limited to archived errors and controls, not the full cohort.
    duplicate_summaries = {}
    for key in ["original_sha256", "model_input_pixels_sha256"]:
        available = df.loc[df.asset_status.eq("ok")]
        repeated = available.loc[available[key].duplicated(keep=False)]
        repeated.to_csv(out / f"archived_duplicate_{key}.csv", index=False)
        duplicate_summaries[key] = {"n_images_screened": len(available), "n_repeated_images": len(repeated),
                                    "n_cross_label_hashes": int(repeated.groupby(key).class_name.nunique().gt(1).sum()),
                                    "n_cross_fold_hashes": int(repeated.groupby(key).fold.nunique().gt(1).sum())}
    df.to_csv(out / "all_predictions_with_identifiers.csv", index=False)
    df.loc[df.any_error].to_csv(out / "all_error_cases.csv", index=False)
    focus = df.loc[df.canonical_strain.eq(FOCUS)]
    focus.to_csv(out / "DH5alpha_all_predictions.csv", index=False)
    matrix(focus, "class_name", "pred_class_name", names).to_csv(out / "DH5alpha_confusion_counts.csv")
    write_gallery(df, out)
    gallery_files = render_comparisons(df, pairs, out)
    summary = {"cohort": cohort, "protocol": PROTOCOL, "n": len(df), "n_exact_errors": int(df.exact_error.sum()),
               "n_treatment_errors": int(df.treatment_error.sum()), "n_error_images": int(df.any_error.sum()),
               "n_asset_failures": int(df.asset_status.eq("failed").sum()),
               "n_unknown_batch": int(df.batch_id.eq(UNKNOWN).sum()),
               "treatment_decision_rule_disagreements": int(df.pred_treatment_marginal.ne(df.pred_treatment_top_class).sum()),
               "comparison_pages": gallery_files,
               "archived_duplicate_screen_only": duplicate_summaries,
               "sources": [{"path": str(p), "sha256": sha256(p)} for p in [prediction_path, manifest_path, model / "training_args.json"]]}
    write_json(out / "summary.json", summary)
    return df, summary


def export_untreated_counts(root, out):
    """Original untreated training stored confusion counts, but no per-image predictions."""
    model = root / "Model/Benchmark_0min_5class_QC/efficientnet_b0"
    settings = json.loads((model / "training_args.json").read_text())
    names = settings["class_names"]
    total = np.zeros((len(names), len(names)), dtype=int)
    rows, sources = [], []
    for fold in range(1, settings["n_folds"] + 1):
        path = model / f"fold_{fold}/metrics.json"
        metrics = json.loads(path.read_text())
        cm = np.array(metrics["confusion_matrix"], dtype=int)
        if cm.shape != total.shape or (cm < 0).any() or cm.sum() != metrics["n_val"]:
            raise ValueError("Invalid untreated confusion matrix")
        total += cm
        for i, true in enumerate(names):
            for j, predicted in enumerate(names):
                rows.append({"fold": fold, "true_label": true, "predicted_label": predicted, "count": int(cm[i, j])})
        sources.append({"path": str(path), "sha256": sha256(path)})
    if total.sum() != settings["n_samples"] or dict(zip(names, total.sum(1))) != settings["class_counts"]:
        raise ValueError("Untreated count population mismatch")
    pd.DataFrame(total, index=pd.Index(names, name="true_label"), columns=names).to_csv(out / "untreated_5class_confusion_counts.csv")
    pd.DataFrame(rows).to_csv(out / "untreated_5class_confusion_by_fold.csv", index=False)
    return sources + [{"path": str(model / "training_args.json"), "sha256": sha256(model / "training_args.json")}]


def markdown_table(df):
    def value(v):
        return str(v).replace("|", "\\|")
    return "\n".join(["| " + " | ".join(map(value, df.columns)) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
                     + ["| " + " | ".join(map(value, row)) + " |" for row in df.itertuples(index=False, name=None)])


def write_report(results, out, duplicate_summary=None):
    lines = ["# Reviewer presentation and error-case evidence", "",
             "## Scope and limitations", "",
             "These are saved legacy cross-validation predictions, not corrected nested-CV estimates. "
             "The held-out fold was also used for checkpoint selection. Cohorts are never pooled. "
             "All counts describe images, not independent cultures. DH5alpha denotes manifest strain E.coli5-alpha; "
             "AmpR and KanR variants remain separate. No manuscript files were changed.", "",
             "Rows in confusion CSVs are true labels; columns are predicted labels. Counts are integers, "
             "including zeros. Stratified long tables preserve all label pairs in each observed stratum. "
             "An absent stratum is not evidence of zero errors. Treatment-marginal predictions sum the "
             "class probabilities for each treatment before argmax; top-class treatment mapping is exported separately.", "",
             "The error archive includes every exact-label OR marginal-treatment error, not selected failures only. "
             "Originals are byte-identical copies with checksums. PNGs show the loaded full field and 224x224 "
             "model input before normalization, without additional contrast enhancement. Comparison pages select "
             "the median-confidence error per true/predicted condition, time, fold and source collection, "
             "with deterministic correct controls at the same time/fold/source root. Same-true-condition controls "
             "also share the source folder. Unknown batches are not treated as matched biological replicates. "
             "Selection and unavailable controls are recorded in comparison_pairs.csv.", "",
             "Batch, acquisition-session and biological-replicate identifiers remain UNKNOWN without an explicit "
             "image-path-to-acquisition mapping. Source folders and CV group IDs are traceability identifiers, "
             "not verified experimental batches. Source effects, batch effects, time and treatment may be confounded.", ""]
    if duplicate_summary:
        lines += ["## Critical provenance finding", "",
                  "The full retained-cohort hash audit confirms byte-identical images with contradictory "
                  "condition labels. This is an observed data conflict, not merely a hypothesis about "
                  "phenotypic resemblance. The confusion matrices use recorded labels, which are not verified "
                  "ground truth for these cases. Conflict flags are included in all error records and galleries.", ""]
        for cohort, counts in duplicate_summary["cohorts"].items():
            lines += [f"{cohort}-class cohort: {counts['n_label_conflict_groups']} cross-label hash groups "
                      f"({counts['n_label_conflict_images']} image rows); {counts['n_cross_fold_groups']} "
                      "duplicate groups span held-out folds. See the full hash audit for the affected "
                      "conditions and timepoints.", ""]
        lines += ["Verify acquisition provenance before assigning either treatment label. "
                  "Do not describe errors on conflicting-label copies as evidence of biological phenotype "
                  "convergence. Inner-validation checkpoint selection alone does not remove duplicated "
                  "content across partitions. No labels, source images or training jobs have been altered.", ""]
    for cohort, (df, summary) in results.items():
        folder = f"treated_{cohort}class"
        focus = df.loc[df.class_name.eq(FOCUS + " + Kanamycin")]
        errors = focus.loc[focus.exact_error]
        at30 = focus.loc[focus.timepoint_min.eq(30)]
        outside = focus.loc[focus.timepoint_min.ne(30)]
        destinations = focus.pred_class_name.value_counts().rename_axis("predicted_condition").reset_index(name="raw_count")
        time = performance_table(focus, ["timepoint_min"])[["timepoint_min", "n", "n_exact_errors", "n_treatment_errors", "exact_accuracy", "treatment_accuracy"]]
        time[["exact_accuracy", "treatment_accuracy"]] = time[["exact_accuracy", "treatment_accuracy"]].round(4)
        folds = performance_table(focus, ["fold"])[["fold", "n", "n_exact_errors", "n_treatment_errors", "exact_accuracy", "treatment_accuracy"]]
        folds[["exact_accuracy", "treatment_accuracy"]] = folds[["exact_accuracy", "treatment_accuracy"]].round(4)
        lines += [f"## Treated {cohort}-class cohort", "",
                  f"Population: {len(df):,}; exact-label errors: {summary['n_exact_errors']:,}; "
                  f"marginal-treatment errors: {summary['n_treatment_errors']:,}. "
                  f"Image export failures: {summary['n_asset_failures']}. "
                  f"Treatment decision rules disagree on {summary['treatment_decision_rule_disagreements']} images.", "",
                  f"[Complete error gallery]({folder}/error_gallery.html) | [Full integer confusion matrix]({folder}/confusion_exact_counts.csv) | "
                  f"[All error records]({folder}/all_error_cases.csv)", "",
                  f"[Raw confusion figure]({folder}/confusion_exact_counts.png) | "
                  f"[First matched comparison page]({folder}/DH5alpha_comparisons_001.png)", "",
                  f"DH5alpha + kanamycin: {len(focus) - len(errors)}/{len(focus)} exact correct "
                  f"({1 - focus.exact_error.mean():.2%}); {len(errors)} errors. "
                  f"The 30-minute stratum contains {len(at30)}/{len(focus)} images and {int(at30.exact_error.sum())}/{len(errors)} errors. "
                  f"Outside 30 minutes: {int((~outside.exact_error).sum())}/{len(outside)} exact correct "
                  f"({1 - outside.exact_error.mean():.2%}).", "", markdown_table(destinations), "", markdown_table(time), "",
                  "DH5alpha-kanamycin fold-specific counts (within-condition accuracy equals recall):", "", markdown_table(folds), "",
                  "All DH5alpha + kanamycin images come from these source collections (not verified batches): "
                  + "; ".join(sorted(focus.source_collection_id.unique())) + ".", "",
                  "### Evidence-bounded discussion", "",
                  f"The poorer DH5alpha-kanamycin performance is not uniform across exposure times: "
                  f"{int(at30.exact_error.sum())} of {len(errors)} exact-label errors occur at 30 minutes, "
                  f"where {int(at30.pred_class_name.eq(FOCUS + ' + Ampicillin').sum())} of {len(at30)} fields "
                  "are assigned specifically to DH5alpha-ampicillin. Separating strain and treatment decisions "
                  "is therefore important: this dominant error preserves the strain label while changing the "
                  "treatment label. Apparent phenotype similarity is one possible explanation, but predictions "
                  "and visual examples alone cannot distinguish it from image-quality effects, loss of cellular "
                  "detail after resizing, mislabeled/duplicated images, or acquisition confounding. "
                  "The paired galleries and stratified QC summaries are descriptive checks, not evidence of "
                  "a specific biological mechanism. Batch-balanced external evaluation and verified acquisition "
                  "identifiers are needed before attributing the deficit to intrinsic drug-induced morphology. "
                  "These legacy results also retain checkpoint-selection bias and must be updated using the "
                  "corrected outer-test predictions before final performance claims.", ""]
        for key, screen in summary["archived_duplicate_screen_only"].items():
            lines += [f"Archived-only duplicate screen ({key}): {screen['n_images_screened']} images checked; "
                      f"{screen['n_repeated_images']} images have a repeated hash, with "
                      f"{screen['n_cross_label_hashes']} cross-label and {screen['n_cross_fold_hashes']} cross-fold hash groups. "
                      "This does not exclude duplicates among unarchived correct cases, near-duplicates or shared physical fields.", ""]
    lines += ["## Terminology and preprocessing corrections", "",
              "Acquisition modality is not verified by these manifests. Do not silently replace brightfield "
              "with phase-contrast, or use them interchangeably. Once the acquisition metadata is supplied, "
              "name the verified contrast method consistently; use transmitted-light only as an umbrella "
              "description when the illumination geometry is confirmed.", "",
              "The source code does not support a claim that no preprocessing is required. The pipeline "
              "includes segmentation-based QC and image exclusion, image decoding/RGB conversion (including "
              "replication of grayscale channels), bilinear resizing to 224x224, conversion to tensors and "
              "ImageNet channel normalization. Non-uint8 TIFF inputs additionally undergo 1st-99th-percentile "
              "intensity scaling in the loader. Random horizontal/vertical flips are applied during training "
              "only. Segmentation is not a classifier input, but is used for dataset QC and interpretability. "
              "A defensible replacement is: 'Classification uses whole-field images without manual per-cell "
              "feature engineering; inputs undergo the specified preprocessing and segmentation-based quality control.'", "",
              "Relevant implementation: Code/utils/image_utils.py; "
              "Code/Training/Model_Training_5_class/train_0min_fiveclass_raw.py (RawStrainDataset); "
              "Code/Training/Model_Training_Treated_Condition/build_treated_segmentation_qc_manifest.py.", "",
              "The untreated EfficientNet-B0 raw 5x5 matrices are also exported, pooled and by fold. "
              "Its original trainer did not save per-image OOF predictions, so no untreated image identities "
              "are inferred from aggregate counts. The complete error-image archive covers the treated cohorts.", ""]
    (out / "error_case_audit.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--cohorts", type=int, choices=[13, 15], nargs="+", default=[13, 15])
    parser.add_argument("--acquisition_metadata", type=Path)
    parser.add_argument("--duplicate_audit", type=Path, help="Completed full-cohort audit_duplicate_conflicts.py output")
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    metadata = pd.read_csv(args.acquisition_metadata, keep_default_na=False) if args.acquisition_metadata else None
    results = {}
    for cohort in dict.fromkeys(args.cohorts):
        print(f"[AUDIT] treated_{cohort}class", flush=True)
        results[cohort] = audit_cohort(args.root, cohort, out / f"treated_{cohort}class", metadata, args.duplicate_audit)
    sources = export_untreated_counts(args.root, out)
    code = [Path(__file__), args.root / "Code/utils/image_utils.py",
            args.root / "Code/Training/Model_Training_5_class/train_0min_fiveclass_raw.py",
            args.root / "Code/Training/Model_Training_Treated_Condition/train_treated_condition_raw.py"]
    sources += [{"path": str(p), "sha256": sha256(p)} for p in code]
    if args.acquisition_metadata:
        sources.append({"path": str(args.acquisition_metadata), "sha256": sha256(args.acquisition_metadata)})
    for _, summary in results.values():
        sources.extend(summary["sources"])
    duplicate_summary = None
    if args.duplicate_audit:
        duplicate_summary = json.loads((args.duplicate_audit / "summary.json").read_text())
        for path in [args.duplicate_audit / "summary.json"] + [args.duplicate_audit / f"treated_{c}class/all_predictions_with_hashes.csv" for c in results]:
            sources.append({"path": str(path), "sha256": sha256(path)})
    write_report(results, out, duplicate_summary)
    write_json(out / "provenance.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
                                       "protocol": PROTOCOL, "sources": sources,
                                       "command": sys.argv, "cohorts": list(results)})
    missing = sum(s["n_asset_failures"] for _, s in results.values())
    print(f"[DONE] {out}; image failures={missing}", flush=True)
    if missing:
        raise SystemExit("Image export incomplete; inspect image_assets.csv")


if __name__ == "__main__":
    main()
