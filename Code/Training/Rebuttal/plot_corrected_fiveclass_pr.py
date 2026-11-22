#!/usr/bin/env python3
"""One-versus-rest precision-recall curves for verified primary outer tests."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from plot_corrected_fiveclass_confusion import DISPLAY, ROOT, load_results, record_source
from plot_corrected_fiveclass_roc import (
    COLORS, LINESTYLES, compare_reference_predictions, verify_confusion_reference,
)
from validation import PROTOCOL, sha256, write_json

INSET_RECALL = (.8, 1.0)
INSET_PRECISION = (.9, 1.0)


def compute_prs(predictions, names):
    """Use each strain's softmax score; ties stay intact and AP is not trapezoidal."""
    if len(names) != 5 or len(set(names)) != 5:
        raise ValueError("Require five distinct strain labels")
    if predictions.empty or predictions.image_path.isna().any() or predictions.image_path.duplicated().any():
        raise ValueError("Require unique, nonmissing outer-test image paths")
    labels = predictions.y_true.to_numpy(float)
    if (not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all()
            or (labels < 0).any() or (labels >= len(names)).any()):
        raise ValueError("Invalid outer-test class IDs")
    scores = predictions[["prob_" + name for name in names]].to_numpy(float)
    if (not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any()
            or not np.allclose(scores.sum(axis=1), 1, atol=1e-5, rtol=0)):
        raise ValueError("Require finite class probabilities summing to one")
    curves, summary = [], []
    for class_id, name in enumerate(names):
        positive = labels == class_id
        n_positive, n_negative = int(positive.sum()), int((~positive).sum())
        if not n_positive or not n_negative:
            raise ValueError(f"PR needs positive and negative samples: {name}")
        score = scores[:, class_id]
        precision, recall, thresholds = precision_recall_curve(positive, score, drop_intermediate=False)
        ap = float(average_precision_score(positive, score))
        if not np.isclose(ap, -np.sum(np.diff(recall) * precision[:-1]), atol=1e-12, rtol=0):
            raise ValueError("Average precision disagrees with empirical PR steps")
        tp = n_positive - np.searchsorted(np.sort(score[positive]), thresholds, side="left")
        fp = n_negative - np.searchsorted(np.sort(score[~positive]), thresholds, side="left")
        if (not np.allclose(precision[:-1], tp / (tp + fp), atol=1e-12, rtol=0)
                or not np.allclose(recall[:-1], tp / n_positive, atol=1e-12, rtol=0)):
            raise ValueError("PR coordinates disagree with threshold counts")
        # The library's final (recall=0, precision=1) point predicts no positives.
        # Encode it explicitly as a synthetic +infinity threshold, not a data score.
        tp, fp = np.r_[tp, 0], np.r_[fp, 0]
        curves.append(pd.DataFrame({
            "class_id": class_id, "class_name": name, "point_index": np.arange(len(recall)),
            "threshold": np.r_[thresholds, np.inf], "is_synthetic_endpoint": np.r_[np.zeros(len(thresholds), bool), True],
            "recall": recall, "precision": precision, "true_positives": tp, "false_positives": fp,
            "true_negatives": n_negative - fp, "false_negatives": n_positive - tp,
            "n_positive": n_positive, "n_negative": n_negative,
        }))
        summary.append({"class_id": class_id, "class_name": name, "n_positive": n_positive,
                        "n_negative": n_negative, "n_images": len(predictions),
                        "prevalence": n_positive / len(predictions), "average_precision": ap,
                        "ap_scope": "full_recall_range_0_to_1",
                        "ap_method": "noninterpolated_recall_weighted_precision",
                        "comparison": "one_strain_vs_other_four"})
    return pd.concat(curves, ignore_index=True), pd.DataFrame(summary)


def verify_roc_reference(pred, names, roc_dir, run_dir, evidence):
    verification_path = roc_dir / "verification.json"
    record_source(verification_path, evidence)
    verification = json.loads(verification_path.read_text())
    if (verification.get("all_checks_passed") is not True or verification.get("protocol") != PROTOCOL
            or Path(verification["run_dir"]).resolve() != run_dir.resolve()):
        raise ValueError("ROC reference must be a verified corrected export from the same run")
    for name in ("outer_test_predictions.csv", "summary.json"):
        record_source(roc_dir / name, evidence, verification["artifact_sha256"][name])
    summary = json.loads((roc_dir / "summary.json").read_text())
    models = {str(fold): group.model.iloc[0] for fold, group in pred.groupby("fold")}
    if (summary.get("procedure") != "primary_inner_selected" or summary["models_by_fold"] != models
            or summary["n_images"] != len(pred) or summary["n_folds"] != 5):
        raise ValueError("ROC reference must use the same primary inner-selected evaluation")
    compare_reference_predictions(pred, pd.read_csv(roc_dir / "outer_test_predictions.csv"), names)


def make_figure(curves, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch, Rectangle
    from matplotlib.ticker import FormatStrFormatter

    fig, ax = plt.subplots(figsize=(9.2, 8.1))
    fig.subplots_adjust(left=.105, right=.97, bottom=.19, top=.86)
    inset = ax.inset_axes([.12, .45, .47, .38], zorder=5)
    for i, row in enumerate(summary.itertuples(index=False)):
        points = curves.loc[curves.class_name.eq(row.class_name)].sort_values("point_index")
        style = {"color": COLORS[row.class_name], "linestyle": LINESTYLES[i],
                 "linewidth": 2.7 - i * .15, "solid_capstyle": "butt"}
        ax.step(points.recall, points.precision, where="post",
                label=f"{DISPLAY[row.class_name]}   {row.average_precision:.4f}", **style)
        inset.step(points.recall, points.precision, where="post", **style)
    for row in summary.itertuples(index=False):
        ax.axhline(row.prevalence, color=COLORS[row.class_name], linestyle=":", linewidth=.9, alpha=.5, zorder=0)
    ax.set(xlim=(-.01, 1.015), ylim=(-.01, 1.015), xlabel="Recall (sensitivity)", ylabel="Precision (positive predictive value)")
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.tick_params(labelsize=11)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.grid(color="#dce1e5", linewidth=.7, alpha=.75)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    legend = ax.legend(loc="lower left", fontsize=10.5, framealpha=.97, edgecolor="#dce1e5",
                       title="Strain                       Full AP", title_fontsize=10.5,
                       borderpad=.8, labelspacing=.7, handlelength=3.2)
    legend.set_zorder(10)
    inset.set(xlim=(INSET_RECALL[0], 1.003), ylim=(INSET_PRECISION[0], 1.003))
    inset.set_xticks(np.linspace(.8, 1, 5))
    inset.set_yticks(np.linspace(.9, 1, 6))
    inset.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    inset.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    inset.tick_params(labelsize=9)
    inset.set_xlabel("Recall", fontsize=10)
    inset.set_ylabel("Precision", fontsize=10)
    inset.set_title("High-precision / high-recall region", fontsize=10.5, pad=9)
    inset.grid(color="#dce1e5", linewidth=.6)
    inset.set_axisbelow(True)
    for spine in inset.spines.values():
        spine.set_color("#858e96")
    ax.add_patch(Rectangle((.8, .9), .2, .1, fill=False, edgecolor="#707b85", linewidth=1, linestyle="--"))
    connector = ConnectionPatch(xyA=(.8, .9), coordsA=ax.transData, xyB=(1, 1), coordsB=inset.transAxes,
                                color="#a0a8ae", linewidth=.8, linestyle=":", zorder=1)
    ax.add_artist(connector)
    fig.suptitle("Corrected five-class precision-recall curves", fontsize=17, fontweight="bold", y=.965)
    ax.set_title("Primary inner-selected procedure | Pooled outer-test probabilities", fontsize=11, pad=15)
    n = int(summary.n_images.iloc[0])
    fig.text(.5, .045, f"{n:,} images | Five outer test folds | Each strain versus the other four\n"
             "Inset: recall 0.80-1.00, precision 0.90-1.00. Legend: full-range average precision (AP).\n"
             "Dotted horizontal lines: class prevalence (no-skill reference).",
             ha="center", fontsize=9.5, linespacing=1.5, color="#344654")
    return fig, ax, inset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, default=ROOT / "Model/Rebuttal_CV/reviewer1_20261118")
    parser.add_argument("--confusion_dir", type=Path, required=True,
                        help="Verified primary directory from plot_corrected_fiveclass_confusion.py")
    parser.add_argument("--roc_dir", type=Path, required=True,
                        help="Verified primary output from plot_corrected_fiveclass_roc.py")
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    evidence = {}
    for name in ("plot_corrected_fiveclass_pr.py", "plot_corrected_fiveclass_roc.py",
                 "plot_corrected_fiveclass_confusion.py", "summarize_nested_cv.py", "validation.py"):
        record_source(Path(__file__).parent / name, evidence)
    pred, cm, classification, fold_results = load_results(args.run_dir.resolve(), "primary", evidence)
    names = list(cm.index)
    verify_confusion_reference(pred, cm, args.confusion_dir.resolve(), args.run_dir.resolve(), evidence)
    verify_roc_reference(pred, names, args.roc_dir.resolve(), args.run_dir.resolve(), evidence)
    curves, summary = compute_prs(pred, names)
    per_fold = []
    for fold, group in pred.groupby("fold"):
        _, frame = compute_prs(group, names)
        per_fold.append(frame.assign(fold=int(fold), selected_model=group.model.iloc[0]))
    by_fold = pd.concat(per_fold, ignore_index=True)
    variability = by_fold.groupby("class_name").average_precision.agg(["mean", "std", "min", "max"]).rename(
        columns={key: "fold_ap_" + key for key in ("mean", "std", "min", "max")})
    summary = summary.join(variability, on="class_name")
    summary["fold_std_ddof"] = 1
    curves.to_csv(out / "pr_coordinates.csv", index=False)
    summary.to_csv(out / "per_class_ap.csv", index=False)
    by_fold.to_csv(out / "per_class_ap_by_fold.csv", index=False)
    pred.to_csv(out / "outer_test_predictions.csv", index=False)
    cm.to_csv(out / "matched_confusion_counts.csv")
    fig, _, _ = make_figure(curves, summary)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(out / f"corrected_fiveclass_ovr_pr.{extension}", dpi=300, facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)
    models = {str(row["fold"]): row["model"] for row in fold_results}
    caption = (
        "One-versus-rest precision-recall (PR) curves for corrected five-strain E. coli identification. "
        "For each strain, ROC and PR curves were calculated using a one-versus-rest approach, treating "
        "the corresponding class as positive and the remaining four classes as negative. PR curves "
        f"use pooled outer-test softmax probabilities from five-fold cross-validation ({len(pred):,} "
        "images, each evaluated once), with the same primary inner-selected predictions as the ROC "
        "and corrected confusion-matrix figures. Backbone and checkpoint selection used inner "
        "validation only. Curves show all empirical threshold steps without smoothing, tie-breaking "
        "jitter, or a precision envelope. The inset enlarges recall 0.80-1.00 and precision 0.90-1.00; "
        "curves outside this region remain visible in the main axes, and exactly coincident segments "
        "may overlap. The legend reports full-range noninterpolated average precision (AP), computed "
        "as the recall-increment-weighted sum of precision, not trapezoidal PR area or ROC AUC. "
        "Dotted horizontal lines show each strain's prevalence as a no-skill reference. The final "
        "recall=0, precision=1 endpoint is conventional and predicts no positives; it is flagged "
        "with a synthetic +infinity threshold in the coordinate CSV. Per-fold AP and descriptive "
        "sample SDs (ddof=1) are exported separately, not as confidence intervals for pooled AP. "
        "These image-level results do not establish independence beyond the original field groups.\n"
    )
    (out / "figure_caption.txt").write_text(caption)
    write_json(out / "summary.json", {
        "procedure": "primary_inner_selected", "n_images": len(pred), "n_folds": 5,
        "models_by_fold": models, "pooled_classification_metrics": classification,
        "pooled_macro_average_precision": float(summary.average_precision.mean()),
        "per_class": summary.to_dict("records"), "inset_recall": list(INSET_RECALL),
        "inset_precision": list(INSET_PRECISION), "ap_scope": "full_recall_range_0_to_1",
        "ap_method": "noninterpolated_recall_weighted_precision", "curve_drawstyle": "steps-post",
        "reference_confusion_dir": str(args.confusion_dir.resolve()), "reference_roc_dir": str(args.roc_dir.resolve()),
        "limitations": ["Pooled AP and mean fold AP are distinct; fold SD is not a confidence interval.",
                        "Original field grouping does not establish independence at the culture or preparation level."],
    })
    for path, digest in list(evidence.items()):
        record_source(path, evidence, digest)
    write_json(out / "verification.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "all_checks_passed": True,
        "protocol": PROTOCOL, "run_dir": str(args.run_dir.resolve()),
        "same_primary_predictions_and_probabilities_as_confusion": True,
        "same_primary_predictions_and_probabilities_as_roc": True,
        "probability_comparison_atol": 1e-12, "probability_comparison_rtol": 1e-12,
        "n_images": len(pred), "models_by_fold": models, "source_sha256": evidence,
        "artifact_sha256": {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()},
    })
    print(summary[["class_name", "n_positive", "n_negative", "average_precision"]].to_string(index=False), flush=True)
    print(f"[DONE] {out / 'corrected_fiveclass_ovr_pr.png'}", flush=True)


if __name__ == "__main__":
    main()
