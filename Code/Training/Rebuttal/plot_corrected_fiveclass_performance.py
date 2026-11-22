#!/usr/bin/env python3
"""Per-fold bars for the corrected primary five-strain outer-test evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from plot_corrected_fiveclass_confusion import ROOT, load_results, record_source
from plot_corrected_fiveclass_roc import verify_confusion_reference
from validation import PROTOCOL, classification_metrics, fold_statistics, sha256, write_json

METRIC_STYLE = {
    "macro_f1": ("Macro-F1", "#4C78A8"),
    "balanced_accuracy": ("Balanced accuracy", "#72B7B2"),
    "accuracy": ("Overall accuracy", "#E5A33B"),
}


def build_fold_table(predictions, selected_results, names):
    """Keep five independent outer-test evaluations, not pooled or inner scores."""
    if len(names) != 5 or len(set(names)) != 5:
        raise ValueError("Require five distinct strain labels")
    if (predictions.empty or predictions.image_path.isna().any()
            or predictions.image_path.duplicated().any()):
        raise ValueError("Require unique, nonmissing outer-test image paths")
    if predictions.fold.isna().any() or set(predictions.fold) != set(range(1, 6)):
        raise ValueError("Require exactly five outer test folds numbered 1-5")
    for key in ("y_true", "y_pred"):
        values = predictions[key].to_numpy(float)
        if (not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all()
                or (values < 0).any() or (values >= len(names)).any()):
            raise ValueError("Invalid outer-test class IDs")
    selected = pd.DataFrame(selected_results).sort_values("fold").reset_index(drop=True)
    if selected.fold.tolist() != list(range(1, 6)):
        raise ValueError("Require one primary selected result per fold")
    rows = []
    for row in selected.to_dict("records"):
        group = predictions.loc[predictions.fold.eq(row["fold"])]
        if (len(group) != row["n_test"] or not group.model.eq(row["model"]).all()
                or set(group.y_true) != set(range(len(names)))):
            raise ValueError("Outer-test population or selected backbone disagrees")
        measured = classification_metrics(group.y_true, group.y_pred, len(names))
        for metric in METRIC_STYLE:
            if not np.isclose(measured[metric], row[metric], atol=1e-12, rtol=0):
                raise ValueError(f"Per-fold {metric} disagrees with verified predictions")
        rows.append({key: row[key] for key in ("fold", "model", "epoch", "n_test")} |
                    {metric: measured[metric] for metric in METRIC_STYLE})
    return pd.DataFrame(rows)


def summarize_folds(frame):
    stats = fold_statistics(frame.to_dict("records"))
    return pd.DataFrame([
        {"metric": metric, "label": label, "n_folds": 5, "std_ddof": 1,
         **{key: stats[metric + "_" + key] for key in ("mean", "std", "min", "max")},
         "mean_percent": 100 * stats[metric + "_mean"],
         "std_percentage_points": 100 * stats[metric + "_std"]}
        for metric, (label, _) in METRIC_STYLE.items()
    ])


def make_figure(frame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, ax = plt.subplots(figsize=(10.6, 6.8))
    fig.subplots_adjust(left=.085, right=.98, bottom=.23, top=.79)
    positions, width = np.arange(len(frame)), .24
    for offset, (metric, (label, color)) in enumerate(METRIC_STYLE.items()):
        bars = ax.bar(positions + (offset - 1) * width, 100 * frame[metric], width=width,
                      label=label, color=color, edgecolor="white", linewidth=.7, zorder=3)
        ax.bar_label(bars, labels=[f"{value * 100:.1f}" for value in frame[metric]],
                     padding=4, fontsize=9, color="#263744")
    ax.set_ylim(0, 104)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_ylabel("Outer-test score (%)", fontsize=12, labelpad=10)
    ax.set_xticks(positions, [f"Fold {row.fold}\nn = {row.n_test}" for row in frame.itertuples()], fontsize=11)
    ax.tick_params(axis="both", length=0, pad=8)
    ax.grid(axis="y", color="#dce1e5", linewidth=.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#a8afb5")
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.04), ncol=3,
              frameon=False, fontsize=11, handlelength=1.7, columnspacing=2.5)
    fig.suptitle("Five-fold outer-test cross-validation performance", fontsize=17, fontweight="bold", y=.97)
    fig.text(.5, .9, r"$E.\ coli$ strain identification | Primary inner-selected procedure",
             ha="center", fontsize=12, color="#344654")
    summaries = summarize_folds(frame)
    summary_text = "    |    ".join(
        f"{row.label}: {row.mean_percent:.2f} $\\pm$ {row.std_percentage_points:.2f}%"
        for row in summaries.itertuples())
    fig.text(.5, .105, "Across-fold mean $\\pm$ sample SD\n" + summary_text,
             ha="center", fontsize=10, linespacing=1.6, color="#344654")
    fig.text(.5, .035, f"{int(frame.n_test.sum()):,} outer-test images, each evaluated once. "
             "Backbones and checkpoints selected using inner validation only.",
             ha="center", fontsize=9, color="#52616c")
    return fig, ax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, default=ROOT / "Model/Rebuttal_CV/reviewer1_20261118")
    parser.add_argument("--confusion_dir", type=Path, required=True,
                        help="Verified primary directory from plot_corrected_fiveclass_confusion.py")
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    evidence = {}
    for name in ("plot_corrected_fiveclass_performance.py", "plot_corrected_fiveclass_roc.py",
                 "plot_corrected_fiveclass_confusion.py", "summarize_nested_cv.py", "validation.py"):
        record_source(Path(__file__).parent / name, evidence)
    pred, cm, pooled, selected = load_results(args.run_dir.resolve(), "primary", evidence)
    verify_confusion_reference(pred, cm, args.confusion_dir.resolve(), args.run_dir.resolve(), evidence)
    frame = build_fold_table(pred, selected, list(cm.index))
    summary = summarize_folds(frame)
    frame.to_csv(out / "fold_metrics.csv", index=False)
    summary.to_csv(out / "fold_summary.csv", index=False)
    plot_data = frame.melt(id_vars=["fold", "model", "epoch", "n_test"],
                           value_vars=list(METRIC_STYLE), var_name="metric", value_name="score")
    plot_data["score_percent"] = 100 * plot_data.score
    plot_data.to_csv(out / "bar_values.csv", index=False)
    pred.to_csv(out / "outer_test_predictions.csv", index=False)
    cm.to_csv(out / "matched_confusion_counts.csv")
    fig, _ = make_figure(frame)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(out / f"corrected_fiveclass_outer_fold_performance.{extension}", dpi=300, facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)
    models = {str(row.fold): row.model for row in frame.itertuples()}
    caption = (
        "Five-fold outer-test cross-validation performance for E. coli strain identification. "
        "Grouped bars show macro-F1, balanced accuracy (mean recall across the five strains), and "
        "overall accuracy computed separately on each outer test fold from the primary inner-selected "
        "predictions. Backbone and checkpoint selection used only the inner validation subset; outer "
        "test images were excluded from training and selection. This is the same evaluation used for "
        f"the corrected five-class confusion matrix and ROC curves, covering {len(pred):,} retained "
        "untreated images, each appearing in exactly one outer test fold. Test-set image counts "
        "appear below each fold, and bar labels report percentages. Each bar is a single fold-level "
        "estimate, so no within-fold error bars are shown. The footer reports unweighted fold means "
        "and sample standard deviations (ddof=1), not pooled metrics or confidence intervals. "
        "These image-level results preserve the original field grouping and do not establish "
        "independence at the culture or experimental-replicate level.\n"
    )
    (out / "figure_caption.txt").write_text(caption)
    write_json(out / "summary.json", {
        "procedure": "primary_inner_selected", "n_images": len(pred), "n_folds": 5,
        "models_by_fold": models, "fold_metrics": frame.to_dict("records"),
        "fold_statistics": fold_statistics(frame.to_dict("records")),
        "pooled_metrics_not_plotted": pooled, "bar_metric_order": list(METRIC_STYLE),
        "reference_confusion_dir": str(args.confusion_dir.resolve()),
        "variability": "Unweighted fold mean and sample SD (ddof=1); not a confidence interval.",
        "limitations": "Original field grouping; independence beyond field groups is not established.",
    })
    for path, digest in list(evidence.items()):
        record_source(path, evidence, digest)
    write_json(out / "verification.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "all_checks_passed": True,
        "protocol": PROTOCOL, "run_dir": str(args.run_dir.resolve()),
        "same_primary_predictions_and_probabilities_as_confusion": True,
        "n_images": len(pred), "models_by_fold": models, "source_sha256": evidence,
        "artifact_sha256": {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()},
    })
    print(frame.to_string(index=False), flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"[DONE] {out / 'corrected_fiveclass_outer_fold_performance.png'}", flush=True)


if __name__ == "__main__":
    main()
