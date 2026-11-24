#!/usr/bin/env python3
"""Six figures from one verified corrected 15-class EfficientNet-B0 evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from corrected_treated15 import (
    DISPLAY, RUN, STRAINS, TREATMENTS, load_evaluation, scores, timepoint_tables,
    treatment_predictions, verify_feature_arrays,
)
from plot_corrected_fiveclass_confusion import record_source
from validation import PROTOCOL, sha256, write_json

TREATMENT_COLORS = {"Ampicillin": "#1D4F91", "CND": "#00857D", "Kanamycin": "#D86800"}
STRAIN_MARKERS = dict(zip(STRAINS, ["o", "s", "^", "D", "P"]))
FIGURES = ["figure1_treatment_confusion", "figure2_timepoint_metrics", "figure3_treatment_roc",
           "figure4_tsne_features", "figure5_strain_treatment_accuracy", "figure6_fifteen_class_confusion"]


def condition_label(name):
    strain, treatment = name.rsplit(" + ", 1)
    return f"{DISPLAY[strain]} + {treatment}"


def count_matrix(true, predicted, names):
    values = confusion_matrix(true, predicted, labels=range(len(names)))
    if values.sum() != len(true):
        raise ValueError("Confusion matrix does not conserve the image population")
    return pd.DataFrame(values, index=pd.Index(names, name="recorded_label"),
                        columns=pd.Index(names, name="predicted_label"))


def row_percent(cm):
    totals = cm.sum(axis=1)
    if totals.eq(0).any():
        raise ValueError("Cannot normalize a class with no images")
    return cm.div(totals, axis=0) * 100


def condition_accuracy(pred, names):
    rows = []
    for name in names:
        group = pred.loc[pred.class_name.eq(name)]
        if group.empty:
            raise ValueError(f"No images for condition {name}")
        strain, treatment = name.rsplit(" + ", 1)
        correct = int(group.y_true.eq(group.y_pred).sum())
        rows.append({"class_name": name, "strain": strain, "treatment": treatment,
                     "n_images": len(group), "n_correct_exact": correct,
                     "exact_class_accuracy": correct / len(group)})
    return pd.DataFrame(rows)


def treatment_rocs(pred):
    coordinates, summaries = [], []
    for class_id, name in enumerate(TREATMENTS):
        positive = pred.true_treatment_id.eq(class_id).to_numpy()
        p = pred["treatment_prob_" + name].to_numpy()
        npos, nneg = int(positive.sum()), int((~positive).sum())
        if not npos or not nneg:
            raise ValueError("Treatment ROC requires positive and negative images")
        fpr, tpr, thresholds = roc_curve(positive, p, drop_intermediate=False)
        area = float(roc_auc_score(positive, p))
        tp, fp = np.rint(tpr * npos).astype(int), np.rint(fpr * nneg).astype(int)
        coordinates.append(pd.DataFrame({"treatment": name, "point_index": np.arange(len(fpr)),
            "threshold": thresholds, "fpr": fpr, "tpr": tpr, "true_positives": tp,
            "false_positives": fp, "true_negatives": nneg - fp, "false_negatives": npos - tp}))
        summaries.append({"treatment": name, "n_positive": npos, "n_negative": nneg,
                          "full_range_auroc": area})
    return pd.concat(coordinates, ignore_index=True), pd.DataFrame(summaries)


def fold_statistics(table):
    rows = []
    for (task, time), group in table.loc[table.fold.gt(0)].groupby(["task", "timepoint_min"]):
        if sorted(group.fold.tolist()) != list(range(1, 6)):
            raise ValueError("Descriptive fold statistics require all five outer folds")
        for metric in ("macro_f1", "macro_auroc", "accuracy"):
            values = group[metric].to_numpy(float)
            valid = bool(np.isfinite(values).all())
            rows.append({"task": task, "timepoint_min": int(time), "metric": metric,
                         "n_folds": len(values), "n_defined_folds": int(np.isfinite(values).sum()),
                         "mean": float(values.mean()) if valid else np.nan,
                         "sample_sd": float(values.std(ddof=1)) if valid else np.nan,
                         "status": "defined" if valid else "undefined_in_at_least_one_fold"})
    return pd.DataFrame(rows)


def save_figure(fig, out, name):
    for extension in ("png", "pdf", "svg"):
        fig.savefig(out / f"{name}.{extension}", dpi=300, facecolor="white")
    plt.close(fig)


def plot_confusion(cm, title):
    n = len(cm)
    fig, ax = plt.subplots(figsize=(7.6, 7.1) if n == 3 else (14.5, 12.8))
    percentages = row_percent(cm).to_numpy()
    im = ax.imshow(percentages, cmap="Blues", vmin=0, vmax=100)
    for i, j in np.ndindex(cm.shape):
        count = int(cm.iloc[i, j])
        text = f"{count:,}\n{percentages[i, j]:.1f}%" if n == 3 else f"{count:,}"
        ax.text(j, i, text, ha="center", va="center", fontsize=14 if n == 3 else 8,
                color="white" if percentages[i, j] >= 55 else "#162b3a")
    labels = list(cm.index) if n == 3 else [condition_label(name) for name in cm.index]
    ax.set_xticks(range(n), labels, rotation=0 if n == 3 else 55, ha="center" if n == 3 else "right", fontsize=11 if n == 3 else 9)
    ax.set_yticks(range(n), [f"{label}  (n={count:,})" for label, count in zip(labels, cm.sum(axis=1))], fontsize=11 if n == 3 else 9)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=.8)
    ax.tick_params(which="both", length=0)
    ax.set_xlabel("Predicted treatment" if n == 3 else "Predicted strain + treatment", fontsize=12, labelpad=12)
    ax.set_ylabel("Recorded treatment" if n == 3 else "Recorded strain + treatment", fontsize=12, labelpad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(im, ax=ax, fraction=.045, pad=.035)
    bar.set_label("Within recorded class (%)", fontsize=11)
    fig.suptitle(title, fontsize=17, fontweight="bold", y=.98)
    ax.set_title("Corrected EfficientNet-B0 | Pooled outer-test predictions", fontsize=11, pad=16)
    fig.text(.5, .02, f"{int(cm.to_numpy().sum()):,} images | Five outer folds | Cells show raw image counts"
             + (" and row percentages" if n == 3 else "; colors show row percentages"), ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .055, 1, .94))
    return fig


def plot_timepoints(table, task):
    frame = table.loc[table.task.eq(task) & table.fold.eq(0)].sort_values("timepoint_min")
    if frame.empty:
        raise ValueError("Missing pooled timepoint results")
    fig, ax = plt.subplots(figsize=(10.8, 6.8))
    x, width = np.arange(len(frame)), .34
    for offset, (metric, label, color) in enumerate([
        ("macro_f1", "Macro-F1", "#1D4F91"), ("macro_auroc", "Macro-AUROC (one-versus-rest)", "#00857D")]):
        values = frame[metric].to_numpy(float)
        finite = np.isfinite(values)
        bars = ax.bar(x[finite] + (offset - .5) * width, values[finite], width=width,
                      color=color, edgecolor="white", linewidth=.7, label=label)
        ax.bar_label(bars, labels=[f"{value:.4f}" for value in values[finite]], padding=5, fontsize=9.5)
        for position in x[~finite]:
            ax.text(position + (offset - .5) * width, .03, "N/A", ha="center", fontsize=11)
    ax.set_xticks(x, [f"{r.timepoint_min} min\nn = {r.n_images:,}" for r in frame.itertuples()], fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_ylabel("Score", fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.13), ncol=2, frameon=False, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    label = "Three-class treatment discrimination" if task == "treatment3" else "Fifteen-class strain-treatment identification"
    fig.suptitle("Timepoint-specific outer-test performance", fontsize=17, fontweight="bold", y=.98)
    fig.text(.5, .91, label + " | Corrected EfficientNet-B0", ha="center", fontsize=12)
    note = "Pooled outer-test predictions at each time point; bars are not fold means. No confidence intervals are implied."
    if task == "exact15":
        note += "\nMacro-F1 uses all 15 labels (absent class contributes zero); AUROC is N/A when a class is absent."
    fig.text(.5, .035, note, ha="center", fontsize=9.5)
    fig.subplots_adjust(left=.08, right=.98, top=.78, bottom=.18 if task == "treatment3" else .21)
    return fig


def plot_rocs(curves, summary):
    fig, ax = plt.subplots(figsize=(9.2, 8.1))
    fig.subplots_adjust(left=.105, right=.97, bottom=.15, top=.86)
    inset = ax.inset_axes([.12, .18, .46, .45], zorder=5)
    for i, row in enumerate(summary.itertuples()):
        group = curves.loc[curves.treatment.eq(row.treatment)].sort_values("point_index")
        style = {"color": TREATMENT_COLORS[row.treatment], "linewidth": 3.6 - i * .2,
                 "linestyle": ["-", (0, (5, 2)), (0, (3, 1, 1, 1))][i]}
        ax.plot(group.fpr, group.tpr, label=f"{row.treatment}   {row.full_range_auroc:.4f}", **style)
        inset.plot(group.fpr, group.tpr, **style)
    ax.plot([0, 1], [0, 1], "--", color="#a8afb5", linewidth=1)
    ax.set(xlim=(-.01, 1.01), ylim=(-.01, 1.015), xlabel="False-positive rate (1 - specificity)",
           ylabel="True-positive rate (sensitivity)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", title="Treatment                  Full AUC", fontsize=10.5, framealpha=.97, handlelength=3)
    inset.set(xlim=(-.002, .1), ylim=(.7, 1.005), xlabel="False-positive rate", ylabel="True-positive rate")
    inset.set_xticks(np.linspace(0, .1, 6))
    inset.set_yticks(np.linspace(.7, 1, 4))
    inset.tick_params(labelsize=9)
    inset.set_title("High specificity: 90-100%", fontsize=11)
    ax.add_patch(Rectangle((0, .7), .1, .3, fill=False, edgecolor="#7d8790", linestyle="--", linewidth=1))
    fig.suptitle("Treatment-level one-versus-rest ROC", fontsize=17, fontweight="bold", y=.965)
    ax.set_title("Corrected EfficientNet-B0 | Summed strain-treatment probabilities", fontsize=11, pad=15)
    n = int(summary.iloc[0].n_positive + summary.iloc[0].n_negative)
    fig.text(.5, .045, f"{n:,} images | Five outer test folds | Each treatment versus the other two\n"
             "Treatment probabilities sum over all five strains. AUCs cover the full ROC range.", ha="center", fontsize=10)
    return fig


def verify_features(feature_dir, pred, names, run_dir, evidence):
    feature_dir = feature_dir.resolve()
    path = feature_dir / "verification.json"
    record_source(path, evidence)
    report = json.loads(path.read_text())
    if (report.get("all_checks_passed") is not True or report.get("protocol") != PROTOCOL
            or Path(report["run_dir"]).resolve() != run_dir.resolve() or report["n_images"] != len(pred)
            or report["feature_dim"] != 1280 or report["coordinates_comparable_across_folds"] is not False):
        raise ValueError("Features must come from this corrected evaluation with separate fold embeddings")
    if [d["fold"] for d in report["folds"]] != list(range(1, 6)):
        raise ValueError("Incomplete feature extraction")
    for name, digest in report["artifact_sha256"].items():
        record_source(feature_dir / name, evidence, digest)
    for name, digest in report["source_sha256"].items():
        record_source(name, evidence, digest)
    for detail in report["folds"]:
        fold = detail["fold"]
        group = pred.loc[pred.fold.eq(fold)].reset_index(drop=True)
        checkpoint = run_dir / f"treated15/fold_{fold}/efficientnet_b0/best.pt"
        record_source(checkpoint, evidence, detail["checkpoint_sha256"])
        if detail["n_images"] != len(group) or detail["argmax_disagreements"] != 0:
            raise ValueError("Feature population or predictions disagree")
        with np.load(feature_dir / f"fold_{fold}_features.npz", allow_pickle=False) as saved:
            if saved["image_uid"].tolist() != group.image_uid.tolist() or saved["image_path"].tolist() != group.image_path.tolist():
                raise ValueError("Feature identities do not match the outer-test rows")
            verify_feature_arrays(saved["features"], saved["probabilities"], group, names)
    coords = pd.read_csv(feature_dir / "tsne_coordinates.csv")
    columns = ["image_path", "image_uid", "fold", "class_id", "class_name", "canonical_strain",
               "canonical_treatment", "timepoint_min", "y_true", "y_pred"]
    pd.testing.assert_frame_equal(coords[columns], pred[columns].reset_index(drop=True))
    if not np.isfinite(coords[["tsne_x", "tsne_y"]].to_numpy()).all():
        raise ValueError("Nonfinite t-SNE coordinates")
    return coords, report


def plot_tsne(coords):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    for fold, ax in zip(range(1, 6), axes.flat):
        group = coords.loc[coords.fold.eq(fold)]
        for strain in STRAINS:
            sub = group.loc[group.canonical_strain.eq(strain)].sample(frac=1, random_state=42)
            ax.scatter(sub.tsne_x, sub.tsne_y, c=sub.canonical_treatment.map(TREATMENT_COLORS),
                       marker=STRAIN_MARKERS[strain], s=11, alpha=.65, linewidths=0, rasterized=True)
        ax.set_title(f"Outer fold {fold}  |  n = {len(group):,}", fontsize=12)
        ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2", xticks=[], yticks=[])
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_aspect("equal", adjustable="datalim")
    legend_ax = axes.flat[-1]
    legend_ax.set_axis_off()
    treatment_legend = legend_ax.legend(handles=[Patch(color=TREATMENT_COLORS[t], label=t) for t in TREATMENTS],
                                        title="Recorded treatment (color)", loc="upper left", frameon=False, fontsize=11)
    legend_ax.add_artist(treatment_legend)
    legend_ax.legend(handles=[Line2D([], [], linestyle="", marker=STRAIN_MARKERS[s], color="#344654", markersize=7,
                                    label=DISPLAY[s]) for s in STRAINS],
                     title="Recorded strain (shape)", loc="center left", frameon=False, fontsize=11)
    legend_ax.text(0, .04, "Separate t-SNE fit for each fold.\nPositions and distances cannot be\ncompared across panels.", fontsize=11, color="#344654")
    fig.suptitle("Held-out EfficientNet-B0 feature representations", fontsize=18, fontweight="bold", y=.98)
    fig.text(.5, .025, "1280-dimensional pooled backbone features | PCA (50 components) + t-SNE (perplexity 30, seed 42)\n"
             "All outer-test images; no training/validation images and no pooled unaligned cross-model features.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .075, 1, .94), h_pad=2, w_pad=2)
    return fig


def plot_accuracy(table):
    accuracy = table.pivot(index="strain", columns="treatment", values="exact_class_accuracy").loc[STRAINS, TREATMENTS]
    counts = table.pivot(index="strain", columns="treatment", values="n_images").loc[STRAINS, TREATMENTS]
    correct = table.pivot(index="strain", columns="treatment", values="n_correct_exact").loc[STRAINS, TREATMENTS]
    fig, ax = plt.subplots(figsize=(8.6, 7.0))
    im = ax.imshow(accuracy.to_numpy() * 100, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    for i, j in np.ndindex(accuracy.shape):
        percent = accuracy.iloc[i, j] * 100
        ax.text(j, i, f"{percent:.1f}%\n{correct.iloc[i, j]:,}/{counts.iloc[i, j]:,}", ha="center", va="center",
                fontsize=13, color="white" if percent >= 65 else "#15283a")
    ax.set_xticks(range(3), TREATMENTS, fontsize=12)
    ax.set_yticks(range(5), [DISPLAY[s] for s in STRAINS], fontsize=12)
    ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    ax.set(xlabel="Recorded treatment", ylabel="Recorded strain")
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    bar.set_label("Exact strain-treatment accuracy (%)")
    fig.suptitle("Strain-by-treatment classification accuracy", fontsize=16, fontweight="bold", y=.98)
    ax.set_title("Corrected EfficientNet-B0 | Pooled outer-test predictions", fontsize=11, pad=15)
    fig.text(.5, .035, "Each cell: exact 15-class accuracy and correct/total image count.\n"
             "Both strain and treatment must be correct; this is class recall, not one-versus-rest accuracy.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .1, 1, .94))
    return fig


def captions(task, n):
    scope = "three-treatment" if task == "treatment3" else "15-class strain-treatment"
    common = (f"All figures use the same {n:,} retained images and five corrected outer-test folds from the "
              "fixed EfficientNet-B0 15-class run. Checkpoints were selected by inner-validation macro-F1, "
              "not outer-test performance. This is not the duplicate-cleaned preparation sensitivity cohort. ")
    return {
        FIGURES[0]: common + "Treatment probabilities are sums of the five strain-specific softmax probabilities for each treatment. "
        "The largest marginal probability defines the treatment prediction, which may differ from the treatment attached to the top 15-class label. "
        "Cells display raw counts and row percentages; colors encode row percentages.",
        FIGURES[1]: common + f"Bars show {scope} macro-F1 and unweighted one-versus-rest macro-AUROC from pooled outer-test predictions "
        "within each time point. Counts appear below each time point. These are not means across folds and have no confidence intervals. "
        "Both treatment-level and exact-class per-fold values and class support are exported. Full-label macro-AUROC is undefined when a "
        "class has no positive or negative examples; missing values are never replaced by zero or averaged away. "
        "Exact-class macro-F1 uses all 15 labels, with zero contribution from absent classes; DH5alpha + CND is absent at 90 minutes.",
        FIGURES[2]: common + "One-versus-rest ROC curves use each treatment's summed softmax probability against the other two treatments. "
        "All threshold points and ties are retained without smoothing. Full-range AUCs are shown; the inset enlarges FPR 0-0.10 and TPR 0.70-1.00.",
        FIGURES[3]: common + "A read-only inference pass extracts 1280-dimensional global-average-pooled backbone features before head LayerNorm, "
        "using each image's own outer-fold checkpoint and frozen evaluation preprocessing. Recomputed class probabilities are checked against "
        "the saved outer-test probabilities (atol=5e-5, rtol=1e-4) and must preserve every argmax prediction. "
        "PCA (50 components) and t-SNE (perplexity 30, 1000 iterations, seed 42, PCA initialization, automatic learning rate) are fit separately "
        "within each fold, without label-dependent tuning. Colors denote recorded treatment and shapes denote recorded strain. "
        "The five learned feature bases are not assumed aligned: coordinates and distances cannot be compared across panels. "
        "t-SNE is descriptive and does not establish classification performance or biological mechanism.",
        FIGURES[4]: common + "Each cell shows the percentage of images with that recorded strain-treatment label for which the full 15-class "
        "prediction is correct, with the numerator and denominator. This equals per-class recall, not a true-negative-dominated one-versus-rest accuracy.",
        FIGURES[5]: common + "The 15-by-15 matrix uses the argmax of the original 15-class softmax vector. Rows are recorded labels and columns "
        "are predictions. Every cell shows a raw count, while color shows percentage within the recorded class. Zero cells are retained.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, default=RUN)
    parser.add_argument("--feature_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--timepoint_task", choices=["treatment3", "exact15"], default="treatment3")
    args = parser.parse_args()
    out, run_dir = args.out_dir.resolve(), args.run_dir.resolve()
    if out.exists():
        raise FileExistsError(f"Use a new output directory: {out}")
    evidence = {}
    for name in ("plot_corrected_treated15.py", "corrected_treated15.py", "audit_corrected_error_cases.py",
                 "summarize_nested_cv.py", "plot_corrected_fiveclass_confusion.py", "validation.py"):
        record_source(Path(__file__).with_name(name), evidence)
    pred, splits, config, settings = load_evaluation(run_dir, evidence)
    names = settings["class_names"]
    pred = treatment_predictions(pred, names)
    coords, feature_report = verify_features(args.feature_dir, pred, names, run_dir, evidence)
    out.mkdir(parents=True, exist_ok=False)
    pred.to_csv(out / "outer_test_predictions.csv", index=False)
    coords.to_csv(out / "tsne_coordinates.csv", index=False)
    write_json(out / "feature_extraction_summary.json", {k: v for k, v in feature_report.items() if k != "artifact_sha256"})
    exact = count_matrix(pred.y_true, pred.y_pred, names)
    ordered = [f"{s} + {t}" for s in STRAINS for t in TREATMENTS]
    exact = exact.loc[ordered, ordered]
    treatment = count_matrix(pred.true_treatment_id, pred.pred_treatment_id, TREATMENTS)
    for label, cm in (("treatment3", treatment), ("exact15", exact)):
        cm.to_csv(out / f"{label}_confusion_counts.csv")
        row_percent(cm).to_csv(out / f"{label}_confusion_row_percent.csv")
    top_class = count_matrix(pred.true_treatment_id, pred.pred_treatment_top_class_id, TREATMENTS)
    top_class.to_csv(out / "treatment_top_class_confusion_counts_not_plotted.csv")
    table, support = timepoint_tables(pred, names)
    table.to_csv(out / "timepoint_metrics_pooled_and_by_fold.csv", index=False)
    support.to_csv(out / "timepoint_class_support.csv", index=False)
    fold_statistics(table).to_csv(out / "timepoint_fold_statistics_descriptive.csv", index=False)
    curve, aucs = treatment_rocs(pred)
    curve.to_csv(out / "treatment_roc_coordinates.csv", index=False)
    aucs.to_csv(out / "treatment_auc.csv", index=False)
    per_fold, partition_counts = [], []
    for fold, group in pred.groupby("fold"):
        for label, columns, true, predicted in (
            ("treatment3", ["treatment_prob_" + t for t in TREATMENTS], "true_treatment_id", "pred_treatment_id"),
            ("exact15", ["prob_" + n for n in names], "y_true", "y_pred"),
        ):
            per_fold.append({"fold": int(fold), "task": label, "selected_epoch": int(group.selected_epoch.iloc[0]),
                             **scores(group[true], group[predicted], group[columns].to_numpy(), len(columns))})
        partition_counts.extend({"fold": int(fold), "role": role, "n_images": int(count)}
                                for role, count in splits[fold].role.value_counts().items())
    pd.DataFrame(per_fold).to_csv(out / "fold_metrics.csv", index=False)
    pd.DataFrame(partition_counts).to_csv(out / "partition_counts.csv", index=False)
    accuracy = condition_accuracy(pred, names)
    accuracy.to_csv(out / "strain_treatment_accuracy.csv", index=False)
    save_figure(plot_confusion(treatment, "Three-class treatment confusion matrix"), out, FIGURES[0])
    save_figure(plot_timepoints(table, args.timepoint_task), out, FIGURES[1])
    save_figure(plot_rocs(curve, aucs), out, FIGURES[2])
    save_figure(plot_tsne(coords), out, FIGURES[3])
    save_figure(plot_accuracy(accuracy), out, FIGURES[4])
    save_figure(plot_confusion(exact, "Fifteen-class strain-treatment confusion matrix"), out, FIGURES[5])
    description = captions(args.timepoint_task, len(pred))
    for name, text in description.items():
        (out / (name + "_caption.txt")).write_text(text + "\n")
    limitations = [
        "Correction separates checkpoint selection from outer testing, but does not establish culture/preparation/session independence.",
        "The retained cohort includes 1946 images in 973 exact-byte pairs with conflicting condition labels; 774 duplicate groups span original outer folds.",
        "No images, labels or original folds were silently cleaned or relabeled for these figures.",
        "Timepoint composition differs; DH5alpha + CND is absent at 90 minutes. Full 15-class AUROC is undefined there.",
        "t-SNE is fit separately by fold because independently fine-tuned feature spaces are not assumed aligned.",
    ]
    summary = {"procedure": "primary_inner_selected", "model": "efficientnet_b0", "n_images": len(pred),
               "n_folds": 5, "class_names": names, "timepoint_plot_task": args.timepoint_task,
               "treatment_decision_rule": "argmax_of_sum_over_five_strains",
               "pooled_exact15": scores(pred.y_true, pred.y_pred, pred[["prob_" + n for n in names]].to_numpy(), 15),
               "pooled_treatment3": scores(pred.true_treatment_id, pred.pred_treatment_id,
                    pred[["treatment_prob_" + t for t in TREATMENTS]].to_numpy(), 3),
               "treatment_auc": aucs.to_dict("records"), "limitations": limitations,
               "marginal_vs_top_class_treatment_disagreements": int(pred.pred_treatment_id.ne(pred.pred_treatment_top_class_id).sum())}
    write_json(out / "summary.json", summary)
    lines = ["# Corrected 15-class EfficientNet-B0 figure set", "", f"Run: `{run_dir}`", "",
             f"{len(pred):,} images; each evaluated in its held-out outer fold. Six figures exported as PNG/PDF/SVG.", ""]
    for name in FIGURES:
        lines.extend([f"## {name}", "", f"[PNG]({name}.png) | [PDF]({name}.pdf) | [SVG]({name}.svg)",
                      "", description[name], ""])
    lines.extend(["## Limitations", "", *["- " + value for value in limitations], ""])
    (out / "README.md").write_text("\n".join(lines))
    for path, digest in list(evidence.items()):
        record_source(path, evidence, digest)
    write_json(out / "verification.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "all_checks_passed": True, "status": "complete",
        "protocol": PROTOCOL, "run_dir": str(run_dir), "n_images": len(pred), "figures": FIGURES,
        "feature_dir": str(args.feature_dir.resolve()), "source_sha256": evidence,
        "artifact_sha256": {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()},
    })
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[DONE] Six figures: {out}", flush=True)


if __name__ == "__main__":
    main()
