"""Shared verification, treatment aggregation and held-out feature extraction."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from audit_corrected_error_cases import load_outer_predictions
from plot_corrected_fiveclass_confusion import DISPLAY, ROOT, record_source
from validation import PROTOCOL, sha256, write_json

RUN = ROOT / "Model/Rebuttal_CV/reviewer1_treated15_20261119"
STRAINS = list(DISPLAY)
TREATMENTS = ["Ampicillin", "CND", "Kanamycin"]
FEATURE_ATOL = 5e-5
FEATURE_RTOL = 1e-4


def validate_predictions(pred, names):
    expected = {f"{s} + {t}" for s in STRAINS for t in TREATMENTS}
    if len(names) != 15 or set(names) != expected:
        raise ValueError("Require all fifteen strain-treatment classes")
    if (pred.empty or pred.image_path.isna().any() or pred.image_path.duplicated().any()
            or pred.image_uid.isna().any() or pred.image_uid.duplicated().any()):
        raise ValueError("Require unique nonmissing outer-test image identities")
    for key in ("y_true", "y_pred", "class_id"):
        x = pred[key].to_numpy(float)
        if (not np.isfinite(x).all() or not np.equal(x, np.floor(x)).all()
                or (x < 0).any() or (x >= 15).any()):
            raise ValueError("Invalid class IDs")
    mapping = dict(enumerate(names))
    if (pred.y_true.tolist() != pred.class_id.tolist()
            or pred.y_true.map(mapping).tolist() != pred.class_name.tolist()
            or pred.y_pred.map(mapping).tolist() != pred.pred_class_name.tolist()):
        raise ValueError("Class names and IDs disagree")
    if (pred.class_name.tolist() != (pred.canonical_strain + " + " + pred.canonical_treatment).tolist()
            or pred.timepoint_min.isna().any()):
        raise ValueError("Strain, treatment or time metadata disagree")
    p = pred[["prob_" + n for n in names]].to_numpy(float)
    if (not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
            or not np.allclose(p.sum(1), 1, atol=1e-5, rtol=0)
            or not np.array_equal(p.argmax(1), pred.y_pred)):
        raise ValueError("Invalid fifteen-class probabilities or argmax")
    return p


def load_evaluation(run_dir, evidence):
    run_dir = Path(run_dir).resolve()
    config = json.loads((run_dir / "run.json").read_text())
    settings = config["tasks"].get("treated15", {})
    if (config["protocol"] != PROTOCOL or settings.get("candidates") != ["efficientnet_b0"]
            or settings.get("n_folds") != 5 or len(settings.get("class_names", [])) != 15):
        raise ValueError("Require corrected five-fold 15-class EfficientNet-B0 evaluation")
    pred, splits, config, settings = load_outer_predictions(run_dir, "treated15", evidence)
    names = settings["class_names"]
    validate_predictions(pred, names)
    if (len(pred) != settings["n_samples"] or set(pred.fold) != set(range(1, 6))
            or not pred.model.eq("efficientnet_b0").all() or pred.groupby("group_id").fold.nunique().gt(1).any()):
        raise ValueError("Wrong outer-test population, model or grouping")
    for fold in range(1, 6):
        folder = run_dir / "treated15" / f"fold_{fold}"
        selection = json.loads((folder / "model_selection.json").read_text())
        selected = selection["candidates"]["efficientnet_b0"]
        result = json.loads((folder / "efficientnet_b0/test_metrics.json").read_text())
        complete = json.loads((folder / "complete.json").read_text())
        if (selected["selection_source"] != "inner_validation_only" or result["epoch"] != selected["epoch"]
                or result["checkpoint_sha256"] != selected["checkpoint_sha256"]
                or complete["selected_model"] != "efficientnet_b0"):
            raise ValueError("Evaluation differs from the inner-selected checkpoint")
    cache = run_dir / "treated15/outer_test_predictions.csv"
    record_source(cache, evidence)
    saved = pd.read_csv(cache)
    columns = ["image_path", "image_uid", "group_id", "class_id", "class_name", "timepoint_min",
               "y_true", "y_pred", "fold", "model", *["prob_" + n for n in names]]
    pd.testing.assert_frame_equal(pred[columns].reset_index(drop=True), saved[columns].reset_index(drop=True),
                                  check_exact=False, rtol=1e-12, atol=1e-12)
    return pred, splits, config, settings


def treatment_predictions(pred, names):
    p = validate_predictions(pred, names)
    treatment_ids = np.array([TREATMENTS.index(n.rsplit(" + ", 1)[1]) for n in names])
    marginal = np.column_stack([p[:, treatment_ids == i].sum(1) for i in range(3)])
    out = pred.copy()
    out["true_treatment_id"] = treatment_ids[pred.y_true.to_numpy(int)]
    out["pred_treatment_id"] = marginal.argmax(1)
    out["pred_treatment_top_class_id"] = treatment_ids[pred.y_pred.to_numpy(int)]
    for i, treatment in enumerate(TREATMENTS):
        out["treatment_prob_" + treatment] = marginal[:, i]
    return out


def scores(y_true, y_pred, probabilities, n_classes):
    support = np.bincount(np.asarray(y_true, int), minlength=n_classes)
    areas = [float(roc_auc_score(np.asarray(y_true) == i, probabilities[:, i]))
             if 0 < support[i] < len(y_true) else np.nan for i in range(n_classes)]
    return {"n_images": len(y_true), "n_classes_present": int((support > 0).sum()),
            "macro_f1": float(f1_score(y_true, y_pred, labels=range(n_classes), average="macro", zero_division=0)),
            "macro_auroc": float(np.mean(areas)) if np.isfinite(areas).all() else np.nan,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "auroc_status": "defined" if np.isfinite(areas).all() else "undefined_missing_positive_or_negative_class"}


def timepoint_tables(pred, names):
    rows, support_rows = [], []
    for time, frame in pred.groupby("timepoint_min", sort=True):
        for fold, group in [(0, frame), *list(frame.groupby("fold", sort=True))]:
            for task, n_classes, labels, columns, true, predicted in (
                ("treatment3", 3, TREATMENTS, ["treatment_prob_" + t for t in TREATMENTS], "true_treatment_id", "pred_treatment_id"),
                ("exact15", 15, names, ["prob_" + n for n in names], "y_true", "y_pred"),
            ):
                rows.append({"task": task, "timepoint_min": int(time), "fold": int(fold),
                             "scope": "pooled_outer_test" if fold == 0 else "individual_outer_fold",
                             **scores(group[true], group[predicted], group[columns].to_numpy(float), n_classes)})
                support = np.bincount(group[true].to_numpy(int), minlength=n_classes)
                support_rows.extend({"task": task, "timepoint_min": int(time), "fold": int(fold),
                                     "class_name": label, "n_positive": int(support[i]),
                                     "n_negative": len(group) - int(support[i])} for i, label in enumerate(labels))
    return pd.DataFrame(rows), pd.DataFrame(support_rows)


def verify_feature_arrays(features, probabilities, reference, names):
    expected = reference[["prob_" + n for n in names]].to_numpy(float)
    if features.shape != (len(reference), 1280) or not np.isfinite(features).all():
        raise ValueError("Require finite 1280-dimensional held-out backbone features")
    if probabilities.shape != expected.shape or not np.isfinite(probabilities).all():
        raise ValueError("Feature-extraction probabilities have invalid shape or values")
    disagreements = int(np.count_nonzero(probabilities.argmax(1) != reference.y_pred))
    discrepancy = float(np.max(np.abs(probabilities - expected)))
    if disagreements or not np.allclose(probabilities, expected, atol=FEATURE_ATOL, rtol=FEATURE_RTOL):
        raise ValueError("Feature-extraction probabilities do not reproduce the saved outer-test predictions: "
                         f"argmax_disagreements={disagreements}, max_absolute_error={discrepancy:.8g}")
    return float(np.max(np.abs(probabilities - expected)))


def tsne_fold(features, seed=42, perplexity=30, iterations=1000):
    if (features.ndim != 2 or not np.isfinite(features).all() or len(features) <= perplexity
            or features.shape[1] < 2):
        raise ValueError("Invalid features or insufficient samples for the fixed t-SNE perplexity")
    pca = PCA(n_components=min(50, features.shape[1], len(features) - 1), svd_solver="randomized", random_state=seed)
    reduced = pca.fit_transform(features)
    model = TSNE(n_components=2, perplexity=perplexity, random_state=seed, max_iter=iterations,
                 init="pca", learning_rate="auto", n_jobs=4)
    coords = model.fit_transform(reduced)
    if coords.shape != (len(features), 2) or not np.isfinite(coords).all():
        raise ValueError("Invalid t-SNE coordinates")
    return coords, {"seed": seed, "perplexity": perplexity, "max_iter": iterations,
                    "init": "pca", "learning_rate": "auto", "pca_components": int(pca.n_components_),
                    "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
                    "kl_divergence": float(model.kl_divergence_), "fit_scope": "one_outer_fold_only"}


def extract_features(run_dir, out, device="cuda", workers=4):
    import torch
    import timm

    run_dir, out = Path(run_dir).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(f"Use a new feature directory: {out}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; submit feature extraction to the GPU partition")
    evidence = {}
    pred, _, config, settings = load_evaluation(run_dir, evidence)
    record_source(Path(__file__), evidence)
    record_source(Path(__file__).with_name("audit_corrected_error_cases.py"), evidence)
    record_source(Path(__file__).with_name("summarize_nested_cv.py"), evidence)
    record_source(Path(__file__).with_name("validation.py"), evidence)
    out.mkdir(parents=True, exist_ok=False)
    frozen = run_dir / "source/Code/Training/Model_Training_5_class/train_0min_fiveclass_raw.py"
    spec = importlib.util.spec_from_file_location("corrected15_frozen_training", frozen)
    training = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = training
    spec.loader.exec_module(training)
    torch.set_num_threads(4)
    # Match seed_everything and PyTorch 2.5 defaults used in the frozen evaluation.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    tables, details = [], []
    for fold, group in pred.groupby("fold", sort=True):
        group = group.reset_index(drop=True)
        checkpoint = run_dir / f"treated15/fold_{fold}/efficientnet_b0/best.pt"
        model = training.MultiClassImageClassifier("efficientnet_b0", num_classes=15, pretrained=False).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        loader = training.make_loader(group, img_size=settings["img_size"], batch_size=settings["batch_size"],
                                      num_workers=workers, pin_memory=device == "cuda", train=False, shuffle=False)
        features, probabilities, observed = [], [], []
        with torch.no_grad():
            for batch, (x, y) in enumerate(loader):
                feat = model.backbone(x.to(device))
                prob = model.head(feat).softmax(1)
                features.append(feat.cpu().numpy())
                probabilities.append(prob.cpu().numpy())
                observed.extend(y.numpy().tolist())
                if batch % 25 == 0:
                    print(f"[FEATURES] fold={fold} images={len(observed)}/{len(group)}", flush=True)
        if observed != group.y_true.tolist():
            raise ValueError("Feature-loader row order changed")
        features, probabilities = np.concatenate(features), np.concatenate(probabilities)
        discrepancy = verify_feature_arrays(features, probabilities, group, settings["class_names"])
        np.savez_compressed(out / f"fold_{fold}_features.npz", features=features, probabilities=probabilities,
                            image_uid=group.image_uid.to_numpy(str), image_path=group.image_path.to_numpy(str))
        print(f"[TSNE] fold={fold} n={len(group)} probability_max_abs_error={discrepancy:.3g}", flush=True)
        coords, parameters = tsne_fold(features)
        table = group[["image_path", "image_uid", "fold", "class_id", "class_name", "canonical_strain",
                       "canonical_treatment", "timepoint_min", "y_true", "y_pred"]].copy()
        table["tsne_x"], table["tsne_y"] = coords[:, 0], coords[:, 1]
        tables.append(table)
        details.append({"fold": int(fold), "n_images": len(group), "checkpoint_sha256": sha256(checkpoint),
                        "max_probability_absolute_error": discrepancy, "argmax_disagreements": 0, **parameters})
        del model, features, probabilities, loader
        if device == "cuda":
            torch.cuda.empty_cache()
    pd.concat(tables, ignore_index=True).to_csv(out / "tsne_coordinates.csv", index=False)
    for path, digest in list(evidence.items()):
        record_source(path, evidence, digest)
    write_json(out / "verification.json", {
        "all_checks_passed": True, "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir), "protocol": PROTOCOL, "n_images": len(pred), "feature_dim": 1280,
        "feature_layer": "EfficientNet-B0 global average pool before head LayerNorm",
        "folds": details, "device": device, "torch": torch.__version__, "timm": timm.__version__,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "probability_atol": FEATURE_ATOL, "probability_rtol": FEATURE_RTOL,
        "purpose": "Read-only feature visualization; original predictions and checkpoints are unchanged.",
        "coordinates_comparable_across_folds": False, "source_sha256": evidence,
        "artifact_sha256": {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()},
    })
    print(f"[DONE] {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract verified corrected 15-class outer-test features and fold-specific t-SNE")
    parser.add_argument("--run_dir", type=Path, default=RUN)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    extract_features(args.run_dir, args.out_dir, args.device, args.workers)
