#!/usr/bin/env python3
"""Train and evaluate treated strain+treatment raw-image classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, recall_score

import sys

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "2_strain_5class"))
from train_5class import (
    MODEL_NAMES,
    MultiClassImageClassifier,
    RawStrainDataset,
    build_model,
    class_weights,
    load_manifest,
    make_common_metadata,
    make_loader,
    save_run_metadata,
    to_builtin,
    train_one_epoch,
    write_json,
)

import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from torch import amp as torch_amp

sys.path.insert(0, str(THIS_DIR.parent / "shared"))
from utils.data_utils import seed_everything


def infer_class_names(df: pd.DataFrame) -> list[str]:
    pairs = df[["class_id", "class_name"]].drop_duplicates().sort_values("class_id")
    return pairs["class_name"].astype(str).tolist()


@torch.no_grad()
def evaluate(model, loader, device, class_names: list[str], criterion=None):
    model.eval()
    total = 0.0
    y_true = []
    y_pred = []
    y_prob = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total += loss.item() * y.size(0)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
        y_prob.extend(probs.cpu().numpy().tolist())

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return {
        "loss": total / len(loader.dataset),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob,
    }


def compute_group_metrics(df: pd.DataFrame, group_col: str) -> list[dict]:
    rows = []
    for group_value, sub in df.groupby(group_col, sort=True):
        y_true = sub["y_true"].astype(int).values
        y_pred = sub["y_pred"].astype(int).values
        present_labels = sorted(np.unique(y_true).tolist())
        rows.append(
            {
                group_col: int(group_value) if group_col == "timepoint_min" else str(group_value),
                "n_samples": int(len(sub)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(
                    recall_score(y_true, y_pred, average="macro", labels=present_labels, zero_division=0)
                ),
                "macro_f1": float(
                    f1_score(y_true, y_pred, average="macro", labels=present_labels, zero_division=0)
                ),
            }
        )
    return rows


def attach_predictions(df: pd.DataFrame, metrics: dict, class_names: list[str]) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    out["y_true"] = metrics["y_true"]
    out["y_pred"] = metrics["y_pred"]
    for idx, class_name in enumerate(class_names):
        out[f"prob_{class_name}"] = [row[idx] for row in metrics["y_prob"]]
    out["pred_class_name"] = [class_names[idx] for idx in out["y_pred"].astype(int)]
    return out


def run_crossval(args, df: pd.DataFrame) -> dict:
    class_names = infer_class_names(df)
    n_classes = len(class_names)
    labels = df["class_id"].astype(int).values
    groups = df["group_id"].astype(str).values
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion_weights = class_weights(labels, n_classes).to(device)
    pin_memory = device == "cuda"

    metadata = make_common_metadata(args, df, class_names, device)
    metadata["n_folds"] = int(args.n_folds)
    metadata["mode"] = "crossval"
    save_run_metadata(args.out_dir, metadata)

    skf = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_metrics = []
    all_predictions = []

    start_fold = getattr(args, "start_fold", 1)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, labels, groups), start=1):
        if fold < start_fold:
            # Load existing results for skipped folds
            fold_dir = args.out_dir / f"fold_{fold}"
            metrics_path = fold_dir / "metrics.json"
            preds_path = fold_dir / "val_predictions.csv"
            if metrics_path.exists():
                import json as _json
                fold_metrics.append(_json.loads(metrics_path.read_text()))
            if preds_path.exists():
                all_predictions.append(pd.read_csv(preds_path))
            print(f"[SKIP] Fold {fold} (start_fold={start_fold})", flush=True)
            continue
        seed_everything(args.seed + fold)
        df_tr = df.iloc[train_idx].reset_index(drop=True)
        df_va = df.iloc[val_idx].reset_index(drop=True)
        fold_dir = args.out_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        tr_loader = make_loader(
            df_tr,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            train=True,
            shuffle=True,
        )
        va_loader = make_loader(
            df_va,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            train=False,
            shuffle=False,
        )

        model = build_model(args.model, num_classes=n_classes, device=device)
        criterion = nn.CrossEntropyLoss(weight=criterion_weights)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scaler = torch_amp.GradScaler("cuda") if args.use_amp and device == "cuda" else None
        print(
            json.dumps(
                {
                    "fold": fold,
                    "status": "start",
                    "n_train": int(len(df_tr)),
                    "n_val": int(len(df_va)),
                    "model": args.model,
                    "device": device,
                }
            ),
            flush=True,
        )

        best_macro_f1 = -1.0
        best_state = None
        history = []
        for epoch in range(1, args.epochs + 1):
            tr_loss = train_one_epoch(model, tr_loader, opt, device, scaler=scaler, criterion=criterion)
            metrics = evaluate(model, va_loader, device, class_names=class_names, criterion=criterion)
            row = {
                "epoch": epoch,
                "train_loss": tr_loss,
                "val_loss": metrics["loss"],
                "val_accuracy": metrics["accuracy"],
                "val_balanced_accuracy": metrics["balanced_accuracy"],
                "val_macro_f1": metrics["macro_f1"],
            }
            history.append(row)
            print(json.dumps({"fold": fold, **row}), flush=True)
            if metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = metrics["macro_f1"]
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)
            torch.save(best_state, fold_dir / "best.pt")

        final_metrics = evaluate(model, va_loader, device, class_names=class_names, criterion=criterion)
        fold_predictions = attach_predictions(df_va, final_metrics, class_names=class_names)
        fold_predictions["fold"] = fold
        all_predictions.append(fold_predictions)
        timepoint_metrics = compute_group_metrics(fold_predictions, "timepoint_min") if "timepoint_min" in fold_predictions.columns else []
        fold_result = {
            "fold": fold,
            "n_train": int(len(df_tr)),
            "n_val": int(len(df_va)),
            "accuracy": final_metrics["accuracy"],
            "balanced_accuracy": final_metrics["balanced_accuracy"],
            "macro_f1": final_metrics["macro_f1"],
            "confusion_matrix": final_metrics["confusion_matrix"],
            "classification_report": final_metrics["classification_report"],
            "timepoint_metrics": timepoint_metrics,
        }
        fold_metrics.append(to_builtin(fold_result))
        write_json(fold_dir / "training_history.json", history)
        write_json(fold_dir / "metrics.json", fold_result)
        fold_predictions.to_csv(fold_dir / "val_predictions.csv", index=False)

        del model, opt, scaler, best_state, tr_loader, va_loader
        if device == "cuda":
            torch.cuda.empty_cache()

    predictions_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else df.iloc[0:0].copy()
    if not predictions_df.empty:
        predictions_df.to_csv(args.out_dir / "crossval_predictions.csv", index=False)
    aggregate_timepoint_metrics = (
        compute_group_metrics(predictions_df, "timepoint_min")
        if not predictions_df.empty and "timepoint_min" in predictions_df.columns
        else []
    )

    summary = {
        **metadata,
        "macro_f1_mean": float(np.mean([r["macro_f1"] for r in fold_metrics])),
        "macro_f1_std": float(np.std([r["macro_f1"] for r in fold_metrics])),
        "accuracy_mean": float(np.mean([r["accuracy"] for r in fold_metrics])),
        "accuracy_std": float(np.std([r["accuracy"] for r in fold_metrics])),
        "balanced_accuracy_mean": float(np.mean([r["balanced_accuracy"] for r in fold_metrics])),
        "balanced_accuracy_std": float(np.std([r["balanced_accuracy"] for r in fold_metrics])),
        "fold_metrics": fold_metrics,
        "aggregate_timepoint_metrics": aggregate_timepoint_metrics,
    }
    write_json(args.out_dir / "crossval_summary.json", summary)
    print(f"[DONE] Cross-validation summary written to {args.out_dir / 'crossval_summary.json'}", flush=True)
    return summary


def train_full_dataset(args, df: pd.DataFrame) -> dict:
    class_names = infer_class_names(df)
    n_classes = len(class_names)
    labels = df["class_id"].astype(int).values
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion_weights = class_weights(labels, n_classes).to(device)
    pin_memory = device == "cuda"

    metadata = make_common_metadata(args, df, class_names, device)
    metadata["mode"] = "train_all"
    metadata["n_folds"] = None
    save_run_metadata(args.out_dir, metadata)

    loader = make_loader(
        df,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        train=True,
        shuffle=True,
    )

    model = build_model(args.model, num_classes=n_classes, device=device)
    criterion = nn.CrossEntropyLoss(weight=criterion_weights)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch_amp.GradScaler("cuda") if args.use_amp and device == "cuda" else None

    history = []
    print(
        json.dumps({"status": "start", "mode": "train_all", "n_train": int(len(df)), "model": args.model, "device": device}),
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, loader, opt, device, scaler=scaler, criterion=criterion)
        row = {"epoch": epoch, "train_loss": tr_loss}
        history.append(row)
        print(json.dumps({"mode": "train_all", **row}), flush=True)

    state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(state, args.out_dir / "final_model.pt")

    summary = {
        **metadata,
        "final_checkpoint": str(args.out_dir / "final_model.pt"),
        "training_history_path": str(args.out_dir / "training_history.json"),
        "final_train_loss": float(history[-1]["train_loss"]) if history else None,
    }
    write_json(args.out_dir / "training_history.json", history)
    write_json(args.out_dir / "final_model_summary.json", summary)
    print(f"[DONE] Final-model artifacts written to {args.out_dir}", flush=True)

    del model, opt, scaler, loader
    if device == "cuda":
        torch.cuda.empty_cache()
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_csv", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--model", type=str, choices=sorted(MODEL_NAMES), default="resnet18")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_amp", action="store_true")
    ap.add_argument("--filter_auto_dropped", action="store_true")
    ap.add_argument("--mode", choices=["crossval", "train_all"], default="crossval")
    ap.add_argument("--start_fold", type=int, default=1, help="Skip folds before this number (for resume)")
    return ap


def main():
    args = build_arg_parser().parse_args()
    seed_everything(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_manifest(args.manifest_csv, filter_auto_dropped=args.filter_auto_dropped)
    if args.mode == "crossval":
        run_crossval(args, df)
    else:
        train_full_dataset(args, df)


if __name__ == "__main__":
    main()
