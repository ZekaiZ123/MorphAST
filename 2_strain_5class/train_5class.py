#!/usr/bin/env python3
"""Train and evaluate 0 min five-class raw-image classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import amp as torch_amp
from torch.utils.data import DataLoader, Dataset

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
from utils.data_utils import seed_everything
from utils.image_utils import load_image_as_pil_rgb


MODEL_NAMES = {
    "resnet18": "resnet18",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
}


def to_builtin(value):
    """Recursively convert NumPy / tensor scalars to JSON-safe Python types."""
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [to_builtin(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def write_json(path: Path, payload):
    path.write_text(json.dumps(to_builtin(payload), indent=2))


class RawStrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int = 224, train: bool = True):
        self.df = df.reset_index(drop=True)
        self.labels = self.df["class_id"].astype(np.int64).values
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        base = [T.Resize((img_size, img_size))]
        aug = [T.RandomHorizontalFlip(), T.RandomVerticalFlip()]
        to_t = [T.ToTensor(), T.Normalize(mean, std)]
        self.train_tfms = T.Compose(base + aug + to_t)
        self.val_tfms = T.Compose(base + to_t)
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = load_image_as_pil_rgb(row["image_path"])
        img = (self.train_tfms if self.train else self.val_tfms)(img)
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return img, y


class MultiClassImageClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True, drop: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        dim = self.backbone.num_features
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(drop), nn.Linear(dim, num_classes))

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)


def class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def load_manifest(manifest_csv: Path, filter_auto_dropped: bool = False) -> pd.DataFrame:
    df = pd.read_csv(manifest_csv)
    if filter_auto_dropped and "drop_from_training" in df.columns:
        df = df[df["drop_from_training"].fillna(0).astype(int) == 0].reset_index(drop=True)
    if df.empty:
        raise SystemExit("No rows available for training after filtering.")
    return df


def infer_class_names(df: pd.DataFrame) -> list[str]:
    pairs = df[["class_id", "class_name"]].drop_duplicates().sort_values("class_id")
    return pairs["class_name"].astype(str).tolist()


def make_loader(
    df: pd.DataFrame,
    *,
    img_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    train: bool,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        RawStrainDataset(df, img_size=img_size, train=train),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_model(model_key: str, num_classes: int, device: str) -> nn.Module:
    model_name = MODEL_NAMES[model_key]
    return MultiClassImageClassifier(model_name=model_name, num_classes=num_classes, pretrained=True).to(device)


def train_one_epoch(model, loader, opt, device, scaler=None, criterion=None):
    model.train()
    total = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad(set_to_none=True)
        if scaler:
            with torch_amp.autocast("cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
        total += loss.item() * y.size(0)
    return total / len(loader.dataset)


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


def make_common_metadata(args, df: pd.DataFrame, class_names: list[str], device: str) -> dict:
    return {
        "manifest_csv": str(args.manifest_csv),
        "model": args.model,
        "model_name": MODEL_NAMES[args.model],
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "img_size": int(args.img_size),
        "num_workers": int(args.num_workers),
        "seed": int(args.seed),
        "use_amp": bool(args.use_amp),
        "filter_auto_dropped": bool(args.filter_auto_dropped),
        "device": device,
        "n_samples": int(len(df)),
        "n_groups": int(df["group_id"].astype(str).nunique()),
        "class_names": class_names,
        "class_counts": df["class_name"].value_counts().sort_index().to_dict(),
    }


def save_run_metadata(out_dir: Path, metadata: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "class_names.json", metadata["class_names"])
    write_json(out_dir / "training_args.json", metadata)


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

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, labels, groups), start=1):
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
        fold_result = {
            "fold": fold,
            "n_train": int(len(df_tr)),
            "n_val": int(len(df_va)),
            "accuracy": final_metrics["accuracy"],
            "balanced_accuracy": final_metrics["balanced_accuracy"],
            "macro_f1": final_metrics["macro_f1"],
            "confusion_matrix": final_metrics["confusion_matrix"],
            "classification_report": final_metrics["classification_report"],
        }
        fold_metrics.append(to_builtin(fold_result))
        write_json(fold_dir / "training_history.json", history)
        write_json(fold_dir / "metrics.json", fold_result)

        del model, opt, scaler, best_state, tr_loader, va_loader
        if device == "cuda":
            torch.cuda.empty_cache()

    summary = {
        **metadata,
        "macro_f1_mean": float(np.mean([r["macro_f1"] for r in fold_metrics])),
        "macro_f1_std": float(np.std([r["macro_f1"] for r in fold_metrics])),
        "accuracy_mean": float(np.mean([r["accuracy"] for r in fold_metrics])),
        "accuracy_std": float(np.std([r["accuracy"] for r in fold_metrics])),
        "balanced_accuracy_mean": float(np.mean([r["balanced_accuracy"] for r in fold_metrics])),
        "balanced_accuracy_std": float(np.std([r["balanced_accuracy"] for r in fold_metrics])),
        "fold_metrics": fold_metrics,
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
        json.dumps(
            {
                "status": "start",
                "mode": "train_all",
                "n_train": int(len(df)),
                "model": args.model,
                "device": device,
            }
        ),
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
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_amp", action="store_true")
    ap.add_argument("--filter_auto_dropped", action="store_true")
    ap.add_argument("--mode", choices=["crossval", "train_all"], default="crossval")
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
