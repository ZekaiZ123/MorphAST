#!/usr/bin/env python3
"""Train one frozen outer fold; select all checkpoints before opening its test set."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import timm
import torch

from validation import PROTOCOL, audit_partition, choose_candidate, classification_metrics, sha256, write_json

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE / "Training/Model_Training_5_class"))
from train_0min_fiveclass_raw import (
    MultiClassImageClassifier, MODEL_NAMES, build_model, class_weights, evaluate,
    make_loader, seed_everything, train_one_epoch,
)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def loader(df, settings, device, *, train):
    return make_loader(df, img_size=settings["img_size"], batch_size=settings["batch_size"],
                       num_workers=settings["num_workers"], pin_memory=device == "cuda",
                       train=train, shuffle=train)


def select_checkpoint(model_key, train_df, validation_df, settings, fold, out_dir, device):
    """This function has no test data argument or outer-test loader."""
    selection_path = out_dir / "selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        if selection["checkpoint_sha256"] != sha256(out_dir / "best.pt"):
            raise ValueError(f"Checkpoint changed: {out_dir}")
        return selection
    if out_dir.exists():
        raise FileExistsError(f"Incomplete candidate at {out_dir}; use a new run to restart training")
    out_dir.mkdir(parents=True)
    seed_everything(settings["seed"] + fold)
    model = build_model(model_key, num_classes=len(settings["class_names"]), device=device)
    weights = class_weights(train_df.class_id.to_numpy(dtype=int), len(settings["class_names"])).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    scaler = torch.amp.GradScaler("cuda") if settings["use_amp"] and device == "cuda" else None
    train_loader = loader(train_df, settings, device, train=True)
    validation_loader = loader(validation_df, settings, device, train=False)
    best_score = -float("inf")
    history = []
    best = None
    for epoch in range(1, settings["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler=scaler, criterion=criterion)
        metrics = evaluate(model, validation_loader, device, class_names=settings["class_names"], criterion=criterion)
        record = {"epoch": epoch, "train_loss": train_loss,
                  **{"validation_" + k: float(metrics[k]) for k in ("loss", "macro_f1", "accuracy", "balanced_accuracy")}}
        if not all(np.isfinite(value) for value in record.values()):
            raise ValueError(f"Non-finite training/validation result: {record}")
        history.append(record)
        write_json(out_dir / "training_history.json", history)
        print(json.dumps({"fold": fold, "model": model_key, **record}), flush=True)
        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            # clone also isolates a checkpoint on CPU, where .cpu() alone aliases weights.
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            temp = out_dir / "best.pt.tmp"
            torch.save(state, temp)
            temp.replace(out_dir / "best.pt")
            best = {"model": model_key, "epoch": epoch,
                    "validation_metrics": {k: float(metrics[k]) for k in ("macro_f1", "accuracy", "balanced_accuracy")}}
    if best is None:
        raise RuntimeError("No checkpoint selected")
    best.update(selection_completed_utc=timestamp(), selection_source="inner_validation_only",
                checkpoint_sha256=sha256(out_dir / "best.pt"),
                training_class_weights=weights.detach().cpu().tolist(),
                n_train=len(train_df), n_validation=len(validation_df))
    write_json(selection_path, best)
    del model, optimizer, scaler, train_loader, validation_loader
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return best


def test_checkpoint(model_key, test_df, train_df, settings, checkpoint, device):
    model = MultiClassImageClassifier(MODEL_NAMES[model_key], num_classes=len(settings["class_names"]), pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    weights = class_weights(train_df.class_id.to_numpy(dtype=int), len(settings["class_names"])).to(device)
    metrics = evaluate(model, loader(test_df, settings, device, train=False), device,
                       class_names=settings["class_names"], criterion=torch.nn.CrossEntropyLoss(weight=weights))
    predictions = test_df.copy()
    predictions["y_true"] = metrics["y_true"]
    predictions["y_pred"] = metrics["y_pred"]
    predictions["pred_class_name"] = [settings["class_names"][i] for i in metrics["y_pred"]]
    for i, name in enumerate(settings["class_names"]):
        predictions["prob_" + name] = [p[i] for p in metrics["y_prob"]]
    result = classification_metrics(metrics["y_true"], metrics["y_pred"], len(settings["class_names"]))
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return predictions, result


def run_fold(run_dir: Path, task: str, fold: int, device="cuda"):
    config = json.loads((run_dir / "run.json").read_text())
    if config["protocol"] != PROTOCOL or config["sklearn"] != sklearn.__version__:
        raise ValueError("Protocol or scikit-learn version mismatch")
    for relative, expected in config["source_sha256"].items():
        if sha256(run_dir / "source" / relative) != expected:
            raise ValueError(f"Frozen source changed: {relative}")
    settings = config["tasks"][task]
    task_dir = run_dir / task
    if sha256(task_dir / "manifest.csv") != settings["manifest_sha256"]:
        raise ValueError("Frozen manifest changed")
    split_path = task_dir / "splits" / f"fold_{fold}.csv"
    if sha256(split_path) != settings["splits"][str(fold)]:
        raise ValueError("Frozen split changed")
    df = pd.read_csv(task_dir / "manifest.csv")
    split = pd.read_csv(split_path)
    if split.row_id.tolist() != list(range(len(df))) or not split.image_path.equals(df.image_path):
        raise ValueError("Split row order does not match manifest")
    audit = audit_partition(df, split.role.to_numpy())
    out_dir = task_dir / f"fold_{fold}"
    out_dir.mkdir(exist_ok=True)
    lock = out_dir / "running.lock"
    lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        run_hash = sha256(run_dir / "run.json")
        status_file = out_dir / "environment.json"
        if status_file.exists():
            if json.loads(status_file.read_text())["run_sha256"] != run_hash:
                raise ValueError("Run configuration changed since training started")
        else:
            write_json(status_file, {
                "run_sha256": run_hash, "started_utc": timestamp(), "device": device,
                "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                "torch": torch.__version__, "timm": timm.__version__, "sklearn": sklearn.__version__,
                "python": platform.python_version(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "split_audit": audit,
            })
        if (out_dir / "complete.json").exists():
            return
        train_df = df.loc[split.role == "train"].copy()
        validation_df = df.loc[split.role == "validation"].copy()
        selections = {}
        for model in settings["candidates"]:
            selections[model] = select_checkpoint(model, train_df, validation_df, settings, fold, out_dir / model, device)
        chosen = choose_candidate(selections, settings["candidates"])
        selection_path = out_dir / "model_selection.json"
        if not selection_path.exists():
            write_json(selection_path, {"selected_model": chosen, "candidates": selections,
                                        "frozen_before_test_utc": timestamp(), "criterion": "inner_validation_macro_f1"})
        else:
            frozen = json.loads(selection_path.read_text())
            if frozen["selected_model"] != chosen or frozen["candidates"] != selections:
                raise ValueError("Frozen model selection changed")

        # The outer-test loader is first constructed after every selection is frozen.
        test_df = df.loc[split.role == "test"].copy()
        for model in settings["candidates"]:
            model_dir = out_dir / model
            result_path = model_dir / "test_metrics.json"
            if result_path.exists():
                continue
            evaluation_lock = model_dir / "test_started.json"
            if evaluation_lock.exists():
                raise RuntimeError(f"Previous test evaluation was interrupted: {model_dir}; do not silently evaluate twice")
            write_json(evaluation_lock, {"started_utc": timestamp(), "selection_sha256": sha256(selection_path)})
            predictions, metrics = test_checkpoint(model, test_df, train_df, settings, model_dir / "best.pt", device)
            predictions["fold"] = fold
            predictions["model"] = model
            predictions.to_csv(model_dir / "test_predictions.csv", index=False)
            metrics.update(fold=fold, model=model, selected_for_primary=model == chosen,
                           epoch=selections[model]["epoch"], n_test=len(test_df),
                           predictions_sha256=sha256(model_dir / "test_predictions.csv"),
                           checkpoint_sha256=selections[model]["checkpoint_sha256"],
                           selection_sha256=sha256(selection_path), test_evaluations=1)
            write_json(result_path, metrics)
            print(json.dumps({"outer_test": metrics}), flush=True)
        write_json(out_dir / "complete.json", {"completed_utc": timestamp(), "selected_model": chosen,
                                                "run_sha256": run_hash})
    finally:
        os.close(lock_fd)
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--task", choices=["untreated", "treated"], required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable: submit this job to the GPU partition")
    torch.set_num_threads(1)
    try:
        run_fold(args.run_dir.resolve(), args.task, args.fold, args.device)
    except Exception:
        write_json(args.run_dir / args.task / f"fold_{args.fold}" / "failure.json",
                   {"failed_utc": timestamp(), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
