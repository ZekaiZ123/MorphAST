#!/usr/bin/env python3
"""Plot raw five-class confusions from verified corrected outer-test predictions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from summarize_nested_cv import read_verified_result
from validation import (
    METRICS, PROTOCOL, audit_partition, classification_metrics, fold_statistics,
    sha256, validate_manifest, write_json,
)

ROOT = Path(__file__).resolve().parents[3]
DISPLAY = {
    "E.coli5-alpha": r"DH5$\alpha$",
    "E.coli5-alphaKanR": r"DH5$\alpha$ KanR",
    "E.coli5-alphaAmpR": r"DH5$\alpha$ AmpR",
    "AIEC": "AIEC", "UPEC": "UPEC",
}


def record_source(path, evidence, expected=None):
    path = Path(path).resolve()
    digest = sha256(path)
    if expected is not None and digest != expected:
        raise ValueError(f"Frozen source changed: {path}")
    if str(path) in evidence and evidence[str(path)] != digest:
        raise ValueError(f"Source changed during plotting: {path}")
    evidence[str(path)] = digest
    return digest


def pooled_counts(predictions, manifest, names):
    if (len(predictions) != len(manifest) or predictions.image_path.duplicated().any()
            or set(predictions.image_path) != set(manifest.image_path)):
        raise ValueError("Outer predictions must cover every image exactly once")
    reference = manifest.set_index("image_path").loc[predictions.image_path]
    if predictions.y_true.tolist() != reference.class_id.tolist():
        raise ValueError("Outer true labels disagree with the frozen manifest")
    for key in ("y_true", "y_pred"):
        values = predictions[key].to_numpy(float)
        if (not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all()
                or (values < 0).any() or (values >= len(names)).any()):
            raise ValueError("Invalid outer prediction class IDs")
    metrics = classification_metrics(predictions.y_true, predictions.y_pred, len(names))
    cm = pd.DataFrame(metrics["confusion_matrix"], index=pd.Index(names, name="recorded_label"),
                      columns=pd.Index(names, name="predicted_label"))
    if cm.sum(axis=1).to_dict() != manifest.class_name.value_counts().to_dict():
        raise ValueError("Confusion row sums disagree with class sample counts")
    return cm, metrics


def load_results(run_dir, procedure, evidence):
    config_path = run_dir / "run.json"
    config_hash = record_source(config_path, evidence)
    config = json.loads(config_path.read_text())
    if config["protocol"] != PROTOCOL:
        raise ValueError("Require corrected internal-validation/outer-test results")
    settings = config["tasks"]["untreated"]
    names = settings["class_names"]
    if set(names) != set(DISPLAY) or settings["n_folds"] != 5:
        raise ValueError("Require the five-class untreated, five-fold evaluation")
    if procedure != "primary" and procedure not in settings["candidates"]:
        raise ValueError("Requested backbone was not a frozen candidate")
    for relative, digest in config["source_sha256"].items():
        record_source(run_dir / "source" / relative, evidence, digest)
    folder = run_dir / "untreated"
    record_source(folder / "manifest.csv", evidence, settings["manifest_sha256"])
    manifest = pd.read_csv(folder / "manifest.csv")
    if (validate_manifest(manifest) != names or len(manifest) != settings["n_samples"]
            or manifest.group_id.nunique() != settings["n_groups"]):
        raise ValueError("Frozen manifest population differs from run settings")
    records, metrics_rows = [], []
    for fold in range(1, 6):
        fold_dir = folder / f"fold_{fold}"
        for path in (fold_dir / "complete.json", fold_dir / "model_selection.json"):
            record_source(path, evidence)
        complete = json.loads((fold_dir / "complete.json").read_text())
        selection = json.loads((fold_dir / "model_selection.json").read_text())
        if complete["run_sha256"] != config_hash or complete["selected_model"] != selection["selected_model"]:
            raise ValueError("Outer-fold completion does not match the frozen run/selection")
        split_path = folder / "splits" / f"fold_{fold}.csv"
        record_source(split_path, evidence, settings["splits"][str(fold)])
        split = pd.read_csv(split_path)
        if split.row_id.tolist() != list(range(len(manifest))):
            raise ValueError("Split row order differs from the frozen manifest")
        for key in ("image_path", "class_id", "group_id"):
            if split[key].tolist() != manifest[key].tolist():
                raise ValueError(f"Split changed manifest identifiers: {key}")
        audit_partition(manifest, split.role.to_numpy())
        model = selection["selected_model"] if procedure == "primary" else procedure
        result, pred = read_verified_result(folder, settings, fold, model)
        selected = selection["candidates"][model]
        if (selected["selection_source"] != "inner_validation_only"
                or result["epoch"] != selected["epoch"]
                or result["checkpoint_sha256"] != selected["checkpoint_sha256"]
                or result["fold"] != fold or result["model"] != model
                or result["n_test"] != len(pred)):
            raise ValueError("Evaluation does not match the inner-selected checkpoint")
        expected = manifest.loc[split.role.eq("test")]
        for key in ("image_uid", "class_id", "class_name", "group_id"):
            if pred[key].tolist() != expected[key].tolist():
                raise ValueError(f"Outer prediction identifiers disagree: {key}")
        if (not pred.fold.eq(fold).all() or not pred.model.eq(model).all()
                or pred.pred_class_name.tolist() != pred.y_pred.map(dict(enumerate(names))).tolist()):
            raise ValueError("Outer prediction model/fold/class names disagree")
        cm, measured = pooled_counts(pred, expected, names)
        if not np.array_equal(cm.to_numpy(), np.asarray(result["confusion_matrix"])):
            raise ValueError("Saved outer confusion differs from verified predictions")
        records.append(pred)
        metrics_rows.append({"fold": fold, "model": model, "epoch": result["epoch"],
                             "n_test": len(pred), **{key: measured[key] for key in METRICS}})
        for name, digest in (("best.pt", result["checkpoint_sha256"]),
                             ("test_predictions.csv", result["predictions_sha256"]), ("test_metrics.json", None)):
            record_source(fold_dir / model / name, evidence, digest)
    pooled = pd.concat(records, ignore_index=True)
    cm, metrics = pooled_counts(pooled, manifest, names)
    if procedure == "primary":
        cached_path = folder / "outer_test_predictions.csv"
        record_source(cached_path, evidence)
        cached = pd.read_csv(cached_path)
        columns = ["image_path", "class_id", "y_true", "y_pred", "fold", "model"]
        pd.testing.assert_frame_equal(pooled[columns], cached[columns])
    return pooled, cm, metrics, metrics_rows


def render_counts(cm, procedure, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    labels = [DISPLAY[name] for name in cm.index]
    counts = cm.to_numpy()
    fig, ax = plt.subplots(figsize=(7.8, 7.2))
    im = ax.imshow(counts, cmap="Blues", vmin=0, vmax=max(1, int(counts.max())))
    for i, j in np.ndindex(counts.shape):
        ax.text(j, i, f"{counts[i, j]:d}", ha="center", va="center", fontsize=15,
                color="white" if counts[i, j] > counts.max() * .55 else "#15283a")
    ax.set_xticks(range(5), labels, rotation=30, ha="right", fontsize=11)
    ax.set_yticks(range(5), [f"{label}  (n={n})" for label, n in zip(labels, counts.sum(1))], fontsize=11)
    ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="both", length=0)
    ax.set_xlabel("Predicted strain", fontsize=12, labelpad=12)
    ax.set_ylabel("Recorded strain", fontsize=12, labelpad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    bar.set_label("Image count", fontsize=11)
    bar.locator = MaxNLocator(nbins=5, integer=True)
    bar.update_ticks()
    subtitle = "Inner-selected procedure (primary)" if procedure == "primary" else "EfficientNet-B0 only (secondary)"
    fig.suptitle("Corrected five-class confusion matrix", fontsize=17, fontweight="bold", y=.98)
    ax.set_title(subtitle, fontsize=11, pad=15)
    fig.text(.5, .025, f"Raw counts | {counts.sum():,} images | Five outer test folds\n"
             "Checkpoint selection used inner validation, not outer test performance.",
             ha="center", fontsize=9, color="#344654")
    fig.tight_layout(rect=(0, .08, 1, .94))
    for extension in ("png", "pdf", "svg"):
        fig.savefig(out / f"confusion_counts.{extension}", dpi=300, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, default=ROOT / "Model/Rebuttal_CV/reviewer1_20261118")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--procedures", nargs="+", choices=["primary", "efficientnet_b0"],
                        default=["primary", "efficientnet_b0"])
    args = parser.parse_args()
    if len(args.procedures) != len(set(args.procedures)):
        parser.error("Specify each procedure only once")
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    evidence, summaries = {}, {}
    for name in ("plot_corrected_fiveclass_confusion.py", "summarize_nested_cv.py", "validation.py"):
        record_source(Path(__file__).parent / name, evidence)
    for procedure in args.procedures:
        pred, cm, metrics, rows = load_results(args.run_dir.resolve(), procedure, evidence)
        folder = out / procedure
        folder.mkdir()
        cm.to_csv(folder / "confusion_counts.csv")
        pred.to_csv(folder / "outer_test_predictions.csv", index=False)
        pd.DataFrame(rows).to_csv(folder / "fold_metrics.csv", index=False)
        fold_counts = []
        for fold, group in pred.groupby("fold"):
            counts = classification_metrics(group.y_true, group.y_pred, len(cm))["confusion_matrix"]
            fold_counts.extend({"fold": int(fold), "recorded_label": true, "predicted_label": predicted,
                                "count": counts[i][j]} for i, true in enumerate(cm.index)
                               for j, predicted in enumerate(cm.columns))
        pd.DataFrame(fold_counts).to_csv(folder / "confusion_counts_by_fold.csv", index=False)
        render_counts(cm, procedure, folder)
        summaries[procedure] = {"n_images": len(pred), "pooled_metrics": metrics, "fold_statistics": fold_statistics(rows),
                                "models_by_fold": {str(row["fold"]): row["model"] for row in rows}}
        write_json(folder / "summary.json", summaries[procedure])
        print(f"[VERIFIED] {procedure}: {len(pred)} images; raw counts {cm.to_numpy().tolist()}", flush=True)
    for path, digest in list(evidence.items()):
        record_source(path, evidence, digest)
    write_json(out / "verification.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "all_checks_passed": True,
        "protocol": PROTOCOL, "run_dir": str(args.run_dir.resolve()), "procedures": summaries,
        "source_sha256": evidence,
        "artifact_sha256": {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()},
        "limitations": ["Checkpoint-selection correction only; independence beyond original field groups is not established.",
                        "Primary backbone selection uses inner validation; fixed EfficientNet-B0 is a separate secondary result.",
                        "Pooled metrics and fold-mean metrics are distinct summaries. No inference or retraining was performed."],
    })
    print(f"[DONE] {out}", flush=True)


if __name__ == "__main__":
    main()
