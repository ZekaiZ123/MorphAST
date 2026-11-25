#!/usr/bin/env python3
"""Run model selection and final training for the 0 min five-class task."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.identity_audit_gate import preflight_legacy_cli

if __name__ == "__main__":
    preflight_legacy_cli()  # Reject v1 before the model-selection module imports torch.

from train_0min_fiveclass_raw import (
    MODEL_NAMES,
    load_manifest,
    run_crossval,
    train_full_dataset,
    write_json,
)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_csv", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--models", nargs="+", choices=sorted(MODEL_NAMES), default=["resnet18", "efficientnet_b0", "swin_tiny"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--final_epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_amp", action="store_true")
    ap.add_argument("--filter_auto_dropped", action="store_true")
    return ap


def main():
    args = build_arg_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_manifest(args.manifest_csv, filter_auto_dropped=args.filter_auto_dropped)

    benchmark_rows = []
    cv_summaries = {}
    for model_key in args.models:
        cv_args = deepcopy(args)
        cv_args.model = model_key
        cv_args.mode = "crossval"
        cv_args.out_dir = args.out_dir / model_key
        summary = run_crossval(cv_args, df)
        cv_summaries[model_key] = summary
        benchmark_rows.append(
            {
                "model": model_key,
                "macro_f1_mean": float(summary["macro_f1_mean"]),
                "macro_f1_std": float(summary["macro_f1_std"]),
                "balanced_accuracy_mean": float(summary["balanced_accuracy_mean"]),
                "balanced_accuracy_std": float(summary["balanced_accuracy_std"]),
                "accuracy_mean": float(summary["accuracy_mean"]),
                "accuracy_std": float(summary["accuracy_std"]),
                "crossval_summary_path": str(cv_args.out_dir / "crossval_summary.json"),
            }
        )

    benchmark_rows.sort(
        key=lambda row: (
            row["macro_f1_mean"],
            row["balanced_accuracy_mean"],
            row["accuracy_mean"],
        ),
        reverse=True,
    )
    best_model = benchmark_rows[0]["model"]

    final_args = deepcopy(args)
    final_args.model = best_model
    final_args.mode = "train_all"
    final_args.epochs = args.final_epochs if args.final_epochs is not None else args.epochs
    final_args.out_dir = args.out_dir / "final_model"
    final_summary = train_full_dataset(final_args, df)

    selection_summary = {
        "selection_metric": "macro_f1_mean",
        "tie_breakers": ["balanced_accuracy_mean", "accuracy_mean"],
        "models": benchmark_rows,
        "best_model": best_model,
        "best_crossval_summary_path": str(args.out_dir / best_model / "crossval_summary.json"),
        "final_model_dir": str(final_args.out_dir),
        "final_model_summary_path": str(final_args.out_dir / "final_model_summary.json"),
        "final_model": final_summary,
    }
    write_json(args.out_dir / "model_selection_summary.json", selection_summary)
    print(f"[DONE] Model selection summary written to {args.out_dir / 'model_selection_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
