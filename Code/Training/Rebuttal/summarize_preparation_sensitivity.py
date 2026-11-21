#!/usr/bin/env python3
"""Verify and summarize both arms without conflating sensitivity with independence."""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from audit_grouped_cv_feasibility import ID_DTYPES
from preparation_sensitivity import TASKS, audit_frozen_partition, verify_frozen_inputs
from summarize_nested_cv import read_verified_result
from validation import METRICS, classification_metrics, fold_statistics, sha256, write_json


def export_diagnostics(predictions, settings, folder):
    labels, names = list(range(len(settings["class_names"]))), settings["class_names"]
    measured = classification_metrics(predictions.y_true, predictions.y_pred, len(names))
    pd.DataFrame(measured["confusion_matrix"], index=names, columns=names).rename_axis("true_class").to_csv(folder / "confusion_counts.csv")
    precision, recall, f1, support = precision_recall_fscore_support(
        predictions.y_true, predictions.y_pred, labels=labels, zero_division=0)
    pd.DataFrame({"class_id": labels, "class_name": names, "n_test": support,
                  "precision": precision, "recall": recall, "f1": f1}).to_csv(folder / "per_class_metrics.csv", index=False)
    enriched = predictions.copy()
    enriched["correct"] = enriched.y_true.eq(enriched.y_pred)
    enriched.loc[~enriched.correct].to_csv(folder / "error_cases.csv", index=False)
    for column in ("preparation_id", "slide_id", "acquisition_session_id", "timepoint_min"):
        counts = enriched.groupby(["class_name", column], dropna=False).agg(
            n_images=("image_path", "size"), n_correct=("correct", "sum"),
            n_outer_folds=("fold", "nunique")).reset_index()
        counts["n_errors"] = counts.n_images - counts.n_correct
        counts["accuracy_within_class"] = counts.n_correct / counts.n_images
        counts.to_csv(folder / f"errors_by_{column}.csv", index=False)
    for column in ("fold", "timepoint_min", "preparation_id", "acquisition_session_id"):
        enriched.groupby([column, "class_name", "pred_class_name"], dropna=False).size().rename("count").reset_index().to_csv(
            folder / f"confusion_counts_by_{column}.csv", index=False)
    return measured


def summarize(run_dir):
    config = json.loads((run_dir / "run.json").read_text())
    verify_frozen_inputs(run_dir, config)
    run_hash = sha256(run_dir / "run.json")
    report = {"protocol": config["protocol"], "run_sha256": run_hash, "status": "complete",
              "limitations": config["limitations"], "tasks": {}, "comparison": config["comparison"]}
    lines = ["# Preparation/slide sensitivity results", "", "## Interpretation limits", "",
             *["- " + x for x in config["limitations"]], "",
             "All fold summaries use unweighted means and sample SD (ddof=1). SD is not a confidence interval. "
             "Preparation-minus-field differences are not paired fold differences or an isolated estimate of batch bias.", ""]
    populations, all_rows = {}, []
    for task, settings in config["tasks"].items():
        folder = run_dir / task
        if sha256(folder / "manifest.csv") != settings["manifest_sha256"]:
            raise ValueError("Frozen manifest changed")
        manifest = pd.read_csv(folder / "manifest.csv", dtype=ID_DTYPES, low_memory=False)
        populations[task] = manifest[["image_path", "class_id", "original_sha256"]]
        complete, pending, failed = [], [], []
        for fold in range(1, settings["n_folds"] + 1):
            path = folder / "splits" / f"fold_{fold}.csv"
            if sha256(path) != settings["splits"][str(fold)]:
                raise ValueError("Frozen split changed")
            audit_frozen_partition(manifest, pd.read_csv(path), task)
            dest = folder / f"fold_{fold}"
            if (dest / "complete.json").exists():
                if json.loads((dest / "complete.json").read_text())["run_sha256"] != run_hash:
                    raise ValueError("Fold was trained under another configuration")
                complete.append(fold)
            elif (dest / "failure.json").exists():
                failed.append(fold)
            else:
                pending.append(fold)
        lines.extend([f"## {task}", ""])
        if pending or failed:
            report["status"] = "incomplete"
            report["tasks"][task] = {"status": "incomplete", "complete_folds": complete,
                                      "pending_folds": pending, "failed_folds": failed}
            lines.extend([f"Complete {len(complete)}/{settings['n_folds']}; pending/running {pending}; failed {failed}. "
                          "No partial-fold average is presented as a completed result.", ""])
            continue
        rows, predictions = [], []
        for fold in complete:
            model = settings["candidates"][0]
            result, pred = read_verified_result(folder, settings, fold, model)
            split = pd.read_csv(folder / "splits" / f"fold_{fold}.csv")
            expected = manifest.loc[split.role.eq("test")].copy().reset_index(drop=True)
            # Derive identifiers from the frozen manifest, not CSV type inference in predictions.
            for column in ["y_true", "y_pred", "pred_class_name", *["prob_" + n for n in settings["class_names"]]]:
                expected[column] = pred[column].to_numpy()
            expected["fold"] = fold
            expected["model"] = model
            predictions.append(expected)
            row = {"task": task, **{k: result[k] for k in ("fold", "model", "epoch", "n_test", *METRICS)},
                   "n_train": int(split.role.eq("train").sum()), "n_validation": int(split.role.eq("validation").sum())}
            rows.append(row)
            all_rows.append(row)
        pooled = pd.concat(predictions, ignore_index=True)
        if len(pooled) != len(manifest) or pooled.image_path.duplicated().any() or set(pooled.image_path) != set(manifest.image_path):
            raise ValueError("Predictions do not cover the matched population exactly once")
        pooled.to_csv(folder / "outer_test_predictions.csv", index=False)
        pooled_metrics = export_diagnostics(pooled, settings, folder)
        stats = fold_statistics(rows)
        report["tasks"][task] = {"status": "complete", "fold_statistics": stats,
                                  "pooled_metrics": pooled_metrics, "folds": rows}
        write_json(folder / "crossval_summary.json", report["tasks"][task])
        lines.extend(["| Metric | Mean +/- sample SD | Pooled |", "| --- | --- | --- |"])
        for metric in METRICS:
            lines.append(f"| {metric} | {stats[metric + '_mean']:.4f} +/- {stats[metric + '_std']:.4f} | {pooled_metrics[metric]:.4f} |")
        lines.extend(["", "| Fold | Train | Validation | Test | Epoch | Macro-F1 | Accuracy | Balanced accuracy |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for row in rows:
            lines.append(f"| {row['fold']} | {row['n_train']} | {row['n_validation']} | {row['n_test']} | {row['epoch']} | "
                         + " | ".join(f"{row[m]:.4f}" for m in METRICS) + " |")
        lines.append("")
    if not populations["field_baseline"].equals(populations["preparation_sensitivity"]):
        raise ValueError("Comparison arms do not have identical ordered images, labels and source hashes")
    if report["status"] == "complete":
        comparisons = []
        for metric in METRICS:
            field = report["tasks"]["field_baseline"]["fold_statistics"]
            prep = report["tasks"]["preparation_sensitivity"]["fold_statistics"]
            comparisons.append({"metric": metric, "field_mean": field[metric + "_mean"],
                                "preparation_mean": prep[metric + "_mean"],
                                "preparation_minus_field_mean": prep[metric + "_mean"] - field[metric + "_mean"]})
        report["metric_comparison"] = comparisons
        pd.DataFrame(comparisons).to_csv(run_dir / "sensitivity_comparison.csv", index=False)
        lines.extend(["## Matched-population comparison", "",
                      "| Metric | Preparation minus field mean |", "| --- | ---: |"])
        lines.extend(f"| {r['metric']} | {r['preparation_minus_field_mean']:+.4f} |" for r in comparisons)
        lines.append("")
    if all_rows:
        pd.DataFrame(all_rows).to_csv(run_dir / "outer_fold_metrics.csv", index=False)
    write_json(run_dir / "sensitivity_results.json", report)
    (run_dir / "sensitivity_results.md").write_text("\n".join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--require_complete", action="store_true")
    args = parser.parse_args()
    result = summarize(args.run_dir.resolve())
    print(json.dumps({"status": result["status"], "report": str(args.run_dir / "sensitivity_results.md")}), flush=True)
    if args.require_complete and result["status"] != "complete":
        raise SystemExit("Some sensitivity folds are incomplete")


if __name__ == "__main__":
    main()
