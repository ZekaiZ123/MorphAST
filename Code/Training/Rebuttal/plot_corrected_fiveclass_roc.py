#!/usr/bin/env python3
"""Five one-versus-rest ROCs for the corrected primary outer-test evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, roc_curve

from plot_corrected_fiveclass_confusion import DISPLAY, ROOT, load_results, record_source
from validation import PROTOCOL, sha256, write_json

# Preserve the strain colors used by the publication's five-class figures.
COLORS = {"E.coli5-alpha": "#4C78A8", "E.coli5-alphaKanR": "#72B7B2",
          "E.coli5-alphaAmpR": "#54A24B", "AIEC": "#E45756", "UPEC": "#F58518"}
LINESTYLES = ["-", (0, (5, 2)), (0, (3, 1, 1, 1)), (0, (2, 1)), (0, (7, 2, 1, 2))]
INSET_FPR = (0.0, 0.05)
INSET_TPR = (0.8, 1.0)


def compute_rocs(predictions, names):
    """Threshold each class's softmax score, not its multiclass hard prediction."""
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
        binary = labels == class_id
        n_positive, n_negative = int(binary.sum()), int((~binary).sum())
        if not n_positive or not n_negative:
            raise ValueError(f"ROC needs positive and negative samples: {name}")
        fpr, tpr, thresholds = roc_curve(binary, scores[:, class_id], drop_intermediate=False)
        area = float(roc_auc_score(binary, scores[:, class_id]))
        if not np.isclose(area, auc(fpr, tpr), atol=1e-12, rtol=0):
            raise ValueError("ROC coordinates and rank-based AUC disagree")
        # Keep every threshold and tied-score step; no smoothing or jitter.
        tp = np.rint(tpr * n_positive).astype(int)
        fp = np.rint(fpr * n_negative).astype(int)
        curves.append(pd.DataFrame({
            "class_id": class_id, "class_name": name, "point_index": np.arange(len(fpr)),
            "threshold": thresholds, "false_positive_rate": fpr, "true_positive_rate": tpr,
            "specificity": 1 - fpr, "true_positives": tp, "false_positives": fp,
            "true_negatives": n_negative - fp, "false_negatives": n_positive - tp,
            "n_positive": n_positive, "n_negative": n_negative,
        }))
        summary.append({"class_id": class_id, "class_name": name, "n_positive": n_positive,
                        "n_negative": n_negative, "n_images": len(predictions), "auc": area,
                        "auc_scope": "full_fpr_range_0_to_1", "comparison": "one_strain_vs_other_four"})
    return pd.concat(curves, ignore_index=True), pd.DataFrame(summary)


def compare_reference_predictions(pred, reference, names):
    columns = ["image_path", "image_uid", "class_id", "class_name", "y_true", "y_pred", "fold", "model",
               *["prob_" + name for name in names]]
    pd.testing.assert_frame_equal(pred[columns].reset_index(drop=True), reference[columns].reset_index(drop=True),
                                  check_exact=False, rtol=1e-12, atol=1e-12)


def verify_confusion_reference(pred, cm, reference_dir, run_dir, evidence):
    verification_path = reference_dir.parent / "verification.json"
    record_source(verification_path, evidence)
    verification = json.loads(verification_path.read_text())
    if (verification.get("all_checks_passed") is not True or verification.get("protocol") != PROTOCOL
            or Path(verification["run_dir"]).resolve() != run_dir.resolve() or reference_dir.name != "primary"):
        raise ValueError("Reference must be the corrected primary confusion export from the same run")
    for name in ("outer_test_predictions.csv", "confusion_counts.csv", "summary.json"):
        path = reference_dir / name
        expected = verification["artifact_sha256"][str(path.relative_to(reference_dir.parent))]
        record_source(path, evidence, expected)
    compare_reference_predictions(pred, pd.read_csv(reference_dir / "outer_test_predictions.csv"), list(cm.index))
    saved = pd.read_csv(reference_dir / "confusion_counts.csv", index_col=0)
    pd.testing.assert_frame_equal(cm, saved, check_names=False)
    selected = json.loads((reference_dir / "summary.json").read_text())["models_by_fold"]
    actual = {str(fold): group.model.iloc[0] for fold, group in pred.groupby("fold")}
    if selected != actual:
        raise ValueError("Reference used different selected backbones")


def make_figure(curves, summary, *, no_text=False, no_grid=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch, Rectangle
    from matplotlib.ticker import FormatStrFormatter, NullFormatter

    fig, ax = plt.subplots(figsize=(9.2, 8.1))
    fig.subplots_adjust(left=.105, right=.97, bottom=.17, top=.86)
    inset = ax.inset_axes([.12, .16, .46, .48], zorder=5)
    for i, row in enumerate(summary.itertuples(index=False)):
        points = curves.loc[curves.class_name.eq(row.class_name)].sort_values("point_index")
        style = {"color": COLORS[row.class_name], "linestyle": LINESTYLES[i],
                 "linewidth": 2.7 - i * .15, "solid_capstyle": "butt"}
        label = f"{DISPLAY[row.class_name]}   {row.auc:.4f}"
        ax.plot(points.false_positive_rate, points.true_positive_rate, label=label, **style)
        inset.plot(points.false_positive_rate, points.true_positive_rate, **style)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#a8afb5", linewidth=1, zorder=0)
    ax.set(xlim=(-.01, 1.01), ylim=(-.01, 1.015), xlabel="False-positive rate (1 - specificity)",
           ylabel="True-positive rate (sensitivity)")
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.tick_params(labelsize=11)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.grid(color="#dce1e5", linewidth=.7, alpha=.75)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    legend = ax.legend(loc="lower right", fontsize=10.5, framealpha=.97, edgecolor="#dce1e5",
                       title="Strain                       Full AUC", title_fontsize=10.5,
                       borderpad=.8, labelspacing=.8, handlelength=3.2)
    legend.set_zorder(10)
    # A small margin keeps FPR=0/TPR=1 curves off the inset spines.
    inset.set(xlim=(-.001, INSET_FPR[1]), ylim=(INSET_TPR[0], 1.005))
    inset.set_xticks(np.linspace(0, .05, 6))
    inset.set_yticks(np.linspace(.8, 1, 5))
    inset.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    inset.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    inset.tick_params(labelsize=9)
    inset.set_xlabel("False-positive rate", fontsize=10)
    inset.set_ylabel("True-positive rate", fontsize=10)
    inset.set_title("High specificity: 95-100%", fontsize=11, pad=9)
    inset.grid(color="#dce1e5", linewidth=.6)
    inset.set_axisbelow(True)
    for spine in inset.spines.values():
        spine.set_color("#858e96")
    ax.add_patch(Rectangle((0, .8), .05, .2, fill=False, edgecolor="#707b85", linewidth=1, linestyle="--"))
    for source, target in (((0, .8), (0, 1)), ((.05, .8), (1, 1))):
        connector = ConnectionPatch(xyA=source, coordsA=ax.transData, xyB=target, coordsB=inset.transAxes,
                                    color="#a0a8ae", linewidth=.8, linestyle=":", zorder=1)
        ax.add_artist(connector)
    fig.suptitle("Corrected five-class one-versus-rest ROC", fontsize=17, fontweight="bold", y=.965)
    ax.set_title("Primary inner-selected procedure | Pooled outer-test predictions", fontsize=11, pad=15)
    n = int(summary.n_images.iloc[0])
    fig.text(.5, .06, f"{n:,} images | Five outer test folds | Each strain versus the other four\n"
             "Inset: FPR 0-0.05 and TPR 0.80-1.00. Legend AUCs use the full ROC range.",
             ha="center", fontsize=10, color="#344654")
    if no_grid:
        for axis in (ax, inset):
            axis.grid(False, which="both", axis="both")
    if no_text:
        legend.remove()
        for text in list(fig.texts):
            text.remove()
        for axis in (ax, inset):
            axis.set(title="", xlabel="", ylabel="")
            for dimension in (axis.xaxis, axis.yaxis):
                dimension.set_major_formatter(NullFormatter())
                dimension.set_minor_formatter(NullFormatter())
                dimension.get_offset_text().set_visible(False)
        # Reclaim label/title margins without changing any data or inset limits.
        fig.subplots_adjust(left=.045, right=.985, bottom=.045, top=.985)
    return fig, ax, inset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, default=ROOT / "Model/Rebuttal_CV/reviewer1_20261118")
    parser.add_argument("--confusion_dir", type=Path, required=True,
                        help="Verified primary directory from plot_corrected_fiveclass_confusion.py")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--no_text", action="store_true",
                        help="Omit titles, labels, tick labels, legend and footer; retain axes and inset")
    parser.add_argument("--no_grid", action="store_true", help="Remove main and inset background grid lines")
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    evidence = {}
    for name in ("plot_corrected_fiveclass_roc.py", "plot_corrected_fiveclass_confusion.py",
                 "summarize_nested_cv.py", "validation.py"):
        record_source(Path(__file__).parent / name, evidence)
    pred, cm, classification, fold_results = load_results(args.run_dir.resolve(), "primary", evidence)
    names = list(cm.index)
    verify_confusion_reference(pred, cm, args.confusion_dir.resolve(), args.run_dir.resolve(), evidence)
    curves, summary = compute_rocs(pred, names)
    per_fold = []
    for fold, group in pred.groupby("fold"):
        _, frame = compute_rocs(group, names)
        per_fold.append(frame.assign(fold=int(fold), selected_model=group.model.iloc[0]))
    by_fold = pd.concat(per_fold, ignore_index=True)
    variability = by_fold.groupby("class_name").auc.agg(["mean", "std", "min", "max"]).rename(
        columns={key: "fold_auc_" + key for key in ("mean", "std", "min", "max")})
    summary = summary.join(variability, on="class_name")
    summary["fold_std_ddof"] = 1
    curves.to_csv(out / "roc_coordinates.csv", index=False)
    summary.to_csv(out / "per_class_auc.csv", index=False)
    by_fold.to_csv(out / "per_class_auc_by_fold.csv", index=False)
    pred.to_csv(out / "outer_test_predictions.csv", index=False)
    cm.to_csv(out / "matched_confusion_counts.csv")
    fig, _, _ = make_figure(curves, summary, no_text=args.no_text, no_grid=args.no_grid)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(out / f"corrected_fiveclass_ovr_roc.{extension}", dpi=300, facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)
    models = {str(row["fold"]): row["model"] for row in fold_results}
    auc_description = ("All figure text and the legend are omitted; full-range pooled AUCs are in per_class_auc.csv. "
                       if args.no_text else "Full-range pooled AUCs are shown in the legend. ")
    caption = (
        "One-versus-rest ROC curves for corrected untreated five-strain classification. Each curve uses the "
        "predicted softmax probability for one strain to distinguish its images from all images of the other four "
        "strains. Curves pool one outer-test score vector for each of 804 images, using the same primary "
        "inner-selected evaluation and labels as the corrected five-class confusion matrix. Checkpoint epochs "
        "and backbone selection use internal validation only. The inset enlarges the 95-100% specificity "
        f"region (FPR 0-0.05; TPR 0.80-1.00). {auc_description}The dashed diagonal "
        "is the chance reference. Curves connect empirical threshold points without smoothing or jitter; "
        "coincident curves can overlap. Positive/negative counts and exact threshold coordinates are exported. "
        "Per-fold AUCs and their sample SDs are reported separately, not as confidence intervals for pooled AUC. "
        "These image-level results do not establish independence beyond the original field groups.\n"
    )
    if args.no_grid:
        caption += "Background grids are omitted in both the main axes and inset.\n"
    (out / "figure_caption.txt").write_text(caption)
    result = {"procedure": "primary_inner_selected", "n_images": len(pred), "n_folds": 5,
              "models_by_fold": models, "pooled_classification_metrics": classification,
              "pooled_macro_auc": float(summary.auc.mean()), "per_class": summary.to_dict("records"),
              "inset_fpr": list(INSET_FPR), "inset_tpr": list(INSET_TPR),
              "auc_scope": "full_fpr_range_0_to_1", "reference_confusion_dir": str(args.confusion_dir.resolve()),
              "presentation": {"no_text": args.no_text, "no_grid": args.no_grid}}
    write_json(out / "summary.json", result)
    for path, digest in list(evidence.items()):
        record_source(path, evidence, digest)
    write_json(out / "verification.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "all_checks_passed": True,
        "protocol": PROTOCOL, "run_dir": str(args.run_dir.resolve()),
        "same_primary_predictions_and_probabilities_as_confusion": True,
        "probability_comparison_atol": 1e-12, "probability_comparison_rtol": 1e-12,
        "presentation": {"no_text": args.no_text, "no_grid": args.no_grid},
        "n_images": len(pred), "models_by_fold": models, "source_sha256": evidence,
        "artifact_sha256": {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()},
    })
    print(summary[["class_name", "n_positive", "n_negative", "auc"]].to_string(index=False), flush=True)
    print(f"[DONE] {out / 'corrected_fiveclass_ovr_roc.png'}", flush=True)


if __name__ == "__main__":
    main()
