#!/usr/bin/env python3
"""Freeze the original benchmark populations, nested partitions, and source code."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sklearn

from validation import PROTOCOL, make_nested_partitions, sha256, validate_manifest, write_json

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Code"))
from utils.identity_audit_gate import reject_v1_in_legacy_trainer

TASKS = {
    "untreated": ("Benchmark_0min_5class_QC", ["resnet18", "efficientnet_b0", "swin_tiny"]),
    "treated": ("Benchmark_Treated_Condition_13class_QC", ["efficientnet_b0"]),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--num_workers", type=int, default=6)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"Use a new run directory: {run_dir}")
    prepared = {}
    for task, (benchmark, candidates) in TASKS.items():
        baseline = ROOT / "Model" / benchmark
        training = json.loads((baseline / "efficientnet_b0/training_args.json").read_text())
        manifest = ROOT / training["manifest_csv"]
        reject_v1_in_legacy_trainer(manifest)
        df = pd.read_csv(manifest)
        df = df.loc[df.drop_from_training.fillna(0).astype(int) == 0].reset_index(drop=True)
        names = validate_manifest(df)
        if len(df) != training["n_samples"] or df.group_id.nunique() != training["n_groups"]:
            raise ValueError(f"{task}: population changed from the original training run")
        if names != training["class_names"]:
            raise ValueError(f"{task}: label mapping changed")
        missing = [p for p in df.image_path if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} missing images, including {missing[:3]}")
        partitions = make_nested_partitions(df, n_folds=training["n_folds"], seed=training["seed"])
        baseline_summaries = {}
        for model in candidates:
            summary = json.loads((baseline / model / "crossval_summary.json").read_text())
            if [r["n_val"] for r in summary["fold_metrics"]] != [a["partitions"]["test"]["n_samples"] for _, a in partitions]:
                raise ValueError(f"{task}/{model}: outer fold sizes differ from original run")
            baseline_summaries[model] = summary
        for fold, (roles, audit) in enumerate(partitions, 1):
            original = baseline / "efficientnet_b0" / f"fold_{fold}" / "val_predictions.csv"
            if original.exists():
                prior = pd.read_csv(original)
                if set(prior.image_path) != set(df.loc[roles == "test", "image_path"]):
                    raise ValueError(f"{task}: outer fold {fold} differs from original predictions")
                audit["legacy_outer_membership_check"] = "exact_image_path_match"
            else:
                audit["legacy_outer_membership_check"] = "same_manifest_order_seed_splitter_and_fold_sizes; original_predictions_unavailable"
        prepared[task] = (df, names, training, manifest, partitions, baseline_summaries, candidates)

    run_dir.mkdir(parents=True)
    config = {
        "protocol": PROTOCOL, "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(ROOT), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "inner_splits": 5, "inner_split_index": 0, "inner_seed_rule": "seed + 1000 + outer_fold",
        "selection_metric": "inner_validation_macro_f1", "checkpoint_tie_break": "earliest_epoch",
        "architecture_tie_break": "declared_candidate_order", "class_weights_source": "inner_training_only",
        "outer_test_evaluations_per_candidate": 1, "refit_on_train_plus_validation": False,
        "std_ddof": 1, "tasks": {},
    }
    for task, (df, names, training, manifest, partitions, summaries, candidates) in prepared.items():
        dest = run_dir / task
        (dest / "splits").mkdir(parents=True)
        (dest / "legacy").mkdir()
        df.to_csv(dest / "manifest.csv", index=False)
        settings = {k: training[k] for k in ("epochs", "batch_size", "lr", "weight_decay", "img_size", "seed", "n_folds", "use_amp")}
        settings.update(num_workers=args.num_workers, candidates=candidates, class_names=names,
                        n_samples=len(df), n_groups=int(df.group_id.nunique()),
                        original_manifest=str(manifest), original_manifest_sha256=sha256(manifest),
                        manifest_sha256=sha256(dest / "manifest.csv"), splits={})
        for fold, (roles, audit) in enumerate(partitions, 1):
            split = df[["image_path", "group_id", "class_id"]].copy()
            split.insert(0, "row_id", range(len(df)))
            split["role"] = roles
            path = dest / "splits" / f"fold_{fold}.csv"
            split.to_csv(path, index=False)
            write_json(path.with_suffix(".json"), audit)
            settings["splits"][str(fold)] = sha256(path)
        for model, summary in summaries.items():
            write_json(dest / "legacy" / f"{model}.json", summary)
        config["tasks"][task] = settings
    source_dirs = ["Code/Training/Rebuttal", "Code/Training/Model_Training_5_class", "Code/utils"]
    config["source_sha256"] = {}
    for directory in source_dirs:
        for source in sorted((ROOT / directory).glob("*.py")):
            relative = source.relative_to(ROOT)
            dest = run_dir / "source" / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            config["source_sha256"][str(relative)] = sha256(dest)
    write_json(run_dir / "run.json", config)
    print(json.dumps({"run_dir": str(run_dir), "tasks": {t: {k: s[k] for k in ("n_samples", "n_groups", "candidates")} for t, s in config["tasks"].items()}}, indent=2))


if __name__ == "__main__":
    main()
