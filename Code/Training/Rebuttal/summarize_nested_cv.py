#!/usr/bin/env python3
"""Verify outer-test predictions and report fold variability and legacy differences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from validation import METRICS, choose_candidate, classification_metrics, fold_statistics, sha256, write_json


def read_verified_result(task_dir, settings, fold, model):
    fold_dir = task_dir / f"fold_{fold}"
    model_dir = fold_dir / model
    selection_path = fold_dir / "model_selection.json"
    selection = json.loads(selection_path.read_text())
    result = json.loads((model_dir / "test_metrics.json").read_text())
    expected_model = choose_candidate(selection["candidates"], settings["candidates"])
    if selection["selected_model"] != expected_model:
        raise ValueError("Backbone selection does not match inner-validation scores")
    if result["selection_sha256"] != sha256(selection_path) or result["test_evaluations"] != 1:
        raise ValueError("Test evaluation did not use the frozen selection")
    if sha256(model_dir / "best.pt") != result["checkpoint_sha256"]:
        raise ValueError("Test checkpoint hash mismatch")
    if sha256(model_dir / "test_predictions.csv") != result["predictions_sha256"]:
        raise ValueError("Test predictions hash mismatch")
    split_path = task_dir / "splits" / f"fold_{fold}.csv"
    if sha256(split_path) != settings["splits"][str(fold)]:
        raise ValueError("Split hash mismatch")
    split = pd.read_csv(split_path)
    test = split.loc[split.role == "test"].reset_index(drop=True)
    predictions = pd.read_csv(model_dir / "test_predictions.csv")
    if predictions.image_path.tolist() != test.image_path.tolist() or predictions.y_true.tolist() != test.class_id.tolist():
        raise ValueError("Predictions are not aligned with the outer-test partition")
    probabilities = predictions[["prob_" + name for name in settings["class_names"]]].to_numpy()
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
            or not np.array_equal(probabilities.argmax(axis=1), predictions.y_pred)):
        raise ValueError("Invalid prediction probabilities")
    measured = classification_metrics(predictions.y_true, predictions.y_pred, len(settings["class_names"]))
    if any(not np.isclose(measured[k], result[k], atol=1e-12, rtol=0) for k in METRICS):
        raise ValueError("Stored metrics differ from predictions")
    if result["selected_for_primary"] != (model == expected_model):
        raise ValueError("Incorrect primary-model indicator")
    return result, predictions


def summarize(run_dir: Path):
    config = json.loads((run_dir / "run.json").read_text())
    report = {
        "protocol": config["protocol"], "run_dir": str(run_dir),
        "run_sha256": sha256(run_dir / "run.json"), "tasks": {}, "status": "complete",
        "limitations": config.get("limitations", []),
    }
    lines = ["# Reviewer 1: corrected cross-validation", "",
             "Checkpoint epochs and, for untreated images, backbone choice are selected on an internal grouped validation subset. "
             "Each outer test fold is evaluated once per frozen candidate. Nominal image fractions are 64% training, "
             "16% validation and 20% test; grouping makes exact sizes vary.", "",
             "All variability below is sample standard deviation across five folds (ddof=1). "
             "Legacy standard deviations are recomputed with the same convention; original runs used ddof=0. "
             "Ranges and individual fold values are also exported. These SDs are descriptive, not confidence intervals.", ""]
    if report["limitations"]:
        lines.extend(["## Known limitations", ""])
        lines.extend(f"- {limitation}" for limitation in report["limitations"])
        lines.append("")
    all_rows = []
    comparisons = []
    for task, settings in config["tasks"].items():
        task_dir = run_dir / task
        complete = []
        pending = []
        failed = []
        for fold in range(1, settings["n_folds"] + 1):
            fold_dir = task_dir / f"fold_{fold}"
            if (fold_dir / "complete.json").exists():
                completion = json.loads((fold_dir / "complete.json").read_text())
                if completion["run_sha256"] != report["run_sha256"]:
                    raise ValueError("Completed fold used another run configuration")
                complete.append(fold)
            elif (fold_dir / "failure.json").exists():
                failed.append(fold)
            else:
                pending.append(fold)
        title = {"untreated": "Untreated (5-class)", "treated": "Treated (13-class)",
                 "treated15": "Treated (15-class)"}.get(task, task.capitalize())
        lines.extend([f"## {title}", ""])
        if pending or failed:
            report["status"] = "incomplete"
            report["tasks"][task] = {"status": "incomplete", "complete_folds": complete,
                                      "pending_folds": pending, "failed_folds": failed}
            lines.extend([f"Results pending: {len(complete)}/{settings['n_folds']} folds complete. "
                          f"Pending/running folds: {pending}. Failed folds: {failed}.",
                          "No corrected five-fold estimate is available yet.", ""])
            continue
        if sha256(task_dir / "manifest.csv") != settings["manifest_sha256"]:
            raise ValueError("Manifest hash mismatch")
        by_model = {m: [] for m in settings["candidates"]}
        primary = []
        primary_predictions = []
        for fold in complete:
            for model in settings["candidates"]:
                row, predictions = read_verified_result(task_dir, settings, fold, model)
                by_model[model].append(row)
                all_rows.append({"task": task, **{k: row[k] for k in ("fold", "model", "epoch", "n_test", "selected_for_primary", *METRICS)}})
                if row["selected_for_primary"]:
                    primary.append(row)
                    primary_predictions.append(predictions)
        if len(primary) != settings["n_folds"]:
            raise ValueError("Expected one primary model per fold")
        pooled = pd.concat(primary_predictions, ignore_index=True)
        manifest = pd.read_csv(task_dir / "manifest.csv")
        if len(pooled) != len(manifest) or pooled.image_path.duplicated().any() or set(pooled.image_path) != set(manifest.image_path):
            raise ValueError("Outer predictions do not cover the population exactly once")
        pooled.to_csv(task_dir / "outer_test_predictions.csv", index=False)
        primary_summary = fold_statistics(primary)
        models = {model: fold_statistics(rows) for model, rows in by_model.items()}
        report["tasks"][task] = {
            "status": "complete", "primary": primary_summary, "per_backbone": models,
            "selected_models_by_fold": {str(row["fold"]): row["model"] for row in primary},
            "pooled_primary_metrics": classification_metrics(pooled.y_true, pooled.y_pred, len(settings["class_names"])),
        }
        lines.extend(["| Model/procedure | Macro-F1 mean +/- SD | Accuracy mean +/- SD | Balanced accuracy mean +/- SD |",
                      "| --- | --- | --- | --- |"])
        display = {"Inner-selected procedure (primary)": primary_summary, **models}
        for label, values in display.items():
            cells = [f"{values[m + '_mean']:.4f} +/- {values[m + '_std']:.4f}" for m in METRICS]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.extend(["", "| Fold | Selected backbone | Epoch | Macro-F1 | Accuracy | Balanced accuracy |",
                      "| --- | --- | --- | --- | --- | --- |"])
        for row in primary:
            lines.append(f"| {row['fold']} | {row['model']} | {row['epoch']} | " + " | ".join(f"{row[m]:.4f}" for m in METRICS) + " |")
        lines.extend(["", "### Comparison with the original procedure", "",
                      "| Backbone | Metric | Original mean +/- SD | Corrected mean +/- SD | Corrected minus original |",
                      "| --- | --- | --- | --- | --- |"])
        task_comparisons = []
        for model, rows in by_model.items():
            original = json.loads((task_dir / "legacy" / f"{model}.json").read_text())
            legacy = fold_statistics(original["fold_metrics"])
            legacy_by_fold = {r["fold"]: r for r in original["fold_metrics"]}
            for metric in METRICS:
                deltas = [r[metric] - legacy_by_fold[r["fold"]][metric] for r in rows]
                entry = {"task": task, "model": model, "metric": metric,
                         "original_mean": legacy[metric + "_mean"], "original_std": legacy[metric + "_std"],
                         "corrected_mean": models[model][metric + "_mean"], "corrected_std": models[model][metric + "_std"],
                         "paired_delta_mean": float(np.mean(deltas)), "paired_delta_std": float(np.std(deltas, ddof=1)),
                         "paired_fold_deltas": deltas}
                comparisons.append(entry)
                task_comparisons.append(entry)
                lines.append(f"| {model} | {metric} | {entry['original_mean']:.4f} +/- {entry['original_std']:.4f} | "
                             f"{entry['corrected_mean']:.4f} +/- {entry['corrected_std']:.4f} | {entry['paired_delta_mean']:+.4f} |")
        report["tasks"][task]["legacy_comparisons"] = task_comparisons
        write_json(task_dir / "crossval_summary.json", report["tasks"][task])
        lines.append("")
    lines.extend(["## Interpretation", "",
                  "The comparison changes both the checkpoint-selection procedure and training-set size (approximately 80% to 64%). "
                  "Its difference quantifies the change in reported performance under the corrected protocol, not a pure estimate of "
                  "checkpoint-selection bias. The original field-group definitions, image processing, ImageNet initialization, "
                  "15-epoch budget and optimizer settings are preserved. This rerun addresses selection/evaluation reuse; "
                  "it does not establish independence beyond the original group identifiers or external-population generalization.", "",
                  "The untreated primary estimate includes backbone selection inside each outer fold. Individual backbone results "
                  "are secondary comparisons and must not be used to select another winner and claim its outer score is an unbiased "
                  "estimate of the selection procedure. The treated candidate is fixed to the published EfficientNet-B0.", "",
                  "Method reference: [scikit-learn nested versus non-nested cross-validation]"
                  "(https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).", ""])
    write_json(run_dir / "corrected_results.json", report)
    (run_dir / "corrected_results.md").write_text("\n".join(lines))
    if all_rows:
        pd.DataFrame(all_rows).to_csv(run_dir / "outer_fold_metrics.csv", index=False)
    if comparisons:
        pd.DataFrame(comparisons).to_csv(run_dir / "legacy_comparison.csv", index=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--require_complete", action="store_true")
    args = parser.parse_args()
    report = summarize(args.run_dir.resolve())
    print(json.dumps({"status": report["status"], "report": str(args.run_dir / "corrected_results.md")}), flush=True)
    if args.require_complete and report["status"] != "complete":
        raise SystemExit("Some folds are incomplete; corrected five-fold results are not yet available")


if __name__ == "__main__":
    main()
