#!/usr/bin/env python3
"""Freeze matched-population field versus preparation/slide sensitivity runs."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import subprocess

import pandas as pd
import sklearn

from audit_grouped_cv_feasibility import (
    DEFAULT_AUDIT, DEFAULT_HASHES, ROOT, clean_population, load_population, verify_images,
)
from preparation_sensitivity import LIMITATIONS, PROTOCOL, TASKS, grouped_manifest, make_partitions, validate_config
from validation import sha256, validate_manifest, write_json


def freeze(run_dir, clean, ledger, training, evidence_paths, num_workers=6, images_rehashed=False):
    if run_dir.exists():
        raise FileExistsError(f"Use a new run directory: {run_dir}")
    names = validate_manifest(clean)
    if names != training["class_names"]:
        raise ValueError("Clean population must preserve every original class")
    prepared = {}
    for task in TASKS:
        manifest = grouped_manifest(clean, task)
        partitions = make_partitions(manifest, task, training["n_folds"], training["seed"])
        prepared[task] = manifest, partitions

    run_dir.mkdir(parents=True)
    config = {
        "protocol": PROTOCOL, "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(ROOT), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "allow_legacy_field_overlap": True, "scid_used": False, "limitations": LIMITATIONS,
        "splitter": "class_pure_largest_component_first_balanced_outer_v1",
        "inner_holdout": "greedy_per_class_20_percent_development_image_target",
        "split_seed_search": False, "selection_metric": "inner_validation_macro_f1",
        "checkpoint_tie_break": "earliest_epoch", "class_weights_source": "inner_training_only",
        "outer_test_evaluations_per_candidate": 1, "refit_on_train_plus_validation": False,
        "comparison": "unpaired_fold_means_identical_cleaned_population", "std_ddof": 1,
        "n_original": len(ledger), "n_included": len(clean), "tasks": {}, "evidence": [],
        "source_images_rehashed": len(ledger) if images_rehashed else 0,
        "source_sha256": {},
    }
    clean.to_csv(run_dir / "cleaned_population.csv", index=False)
    ledger.to_csv(run_dir / "inclusion_ledger.csv", index=False)
    ledger.groupby(["class_name", "timepoint_min", "exclusion_reason"], dropna=False).size().rename("n_images").reset_index().to_csv(run_dir / "population_accounting.csv", index=False)
    for i, source in enumerate(evidence_paths):
        dest = run_dir / "evidence" / f"{i:02d}_{source.name}"
        dest.parent.mkdir(exist_ok=True)
        digest = sha256(source)
        shutil.copy2(source, dest)
        if sha256(dest) != digest:
            raise ValueError(f"Evidence changed during preparation: {source}")
        config["evidence"].append({"path": str(source.resolve()), "frozen_path": str(dest.relative_to(run_dir)), "sha256": digest})
    counts, overlap, time_counts = [], [], []
    for task, (df, partitions) in prepared.items():
        dest = run_dir / task
        (dest / "splits").mkdir(parents=True)
        df.to_csv(dest / "manifest.csv", index=False)
        settings = {k: training[k] for k in ("epochs", "batch_size", "lr", "weight_decay", "img_size", "seed", "n_folds", "use_amp")}
        settings.update(num_workers=num_workers, candidates=["efficientnet_b0"], class_names=names,
                        n_samples=len(df), n_groups=int(df.group_id.nunique()), required_disjoint=TASKS[task],
                        manifest_sha256=sha256(dest / "manifest.csv"), splits={})
        for fold, (roles, audit) in enumerate(partitions, 1):
            split = df[["image_path", "group_id", "class_id"]].copy()
            split.insert(0, "row_id", range(len(df)))
            split["role"] = roles
            path = dest / "splits" / f"fold_{fold}.csv"
            split.to_csv(path, index=False)
            write_json(path.with_suffix(".json"), audit)
            settings["splits"][str(fold)] = sha256(path)
            overlap.extend({"task": task, "fold": fold, **row} for row in audit["identifier_overlap"])
            for role in ("train", "validation", "test"):
                sub = df.loc[roles == role]
                for name, group in sub.groupby("class_name", sort=True):
                    counts.append({"task": task, "fold": fold, "role": role, "class_name": name,
                                   "n_images": len(group), "n_groups": group.group_id.nunique(),
                                   "n_preparations": group.preparation_id.nunique(), "n_sessions": group.acquisition_session_id.nunique()})
                for (name, time), group in sub.groupby(["class_name", "timepoint_min"], sort=True):
                    time_counts.append({"task": task, "fold": fold, "role": role, "class_name": name,
                                        "timepoint_min": int(time), "n_images": len(group)})
        config["tasks"][task] = settings
    pd.DataFrame(counts).to_csv(run_dir / "partition_counts.csv", index=False)
    pd.DataFrame(overlap).to_csv(run_dir / "identifier_overlap.csv", index=False)
    pd.DataFrame(time_counts).to_csv(run_dir / "partition_time_counts.csv", index=False)
    for directory in ("Code/Training/Rebuttal", "Code/Training/Model_Training_5_class", "Code/utils"):
        for source in sorted((ROOT / directory).iterdir()):
            if not source.is_file() or source.suffix not in (".py", ".sbatch"):
                continue
            relative = source.relative_to(ROOT)
            dest = run_dir / "source" / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            config["source_sha256"][str(relative)] = sha256(dest)
    config["artifacts_sha256"] = {name: sha256(run_dir / name) for name in (
        "cleaned_population.csv", "inclusion_ledger.csv", "population_accounting.csv",
        "partition_counts.csv", "identifier_overlap.csv", "partition_time_counts.csv")}
    validate_config(config)
    # Written last: a partially prepared directory is not a runnable experiment.
    write_json(run_dir / "run.json", config)
    lines = ["# Matched preparation/slide sensitivity experiment", "", *["- " + x for x in LIMITATIONS], "",
             f"Population: {len(clean):,}/{len(ledger):,} original retained images; labels unchanged.", "",
             "Two fresh ImageNet-initialized EfficientNet-B0 arms use the original 15-epoch budget and optimizer. "
             "Outer folds are balanced by class using descending component size, with seeded tie breaking. "
             "Inner validation greedily targets 20% of development images per class while reserving training groups. "
             "No seeds, splits or epochs are selected using outer-test performance.", "",
             "See partition_counts.csv and partition_time_counts.csv for actual sizes and time-point coverage. "
             "See identifier_overlap.csv for all train/validation/test overlaps, including allowed field, culture and session sharing.", ""]
    (run_dir / "protocol.md").write_text("\n".join(lines))
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--audit_dir", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--hash_inventory", type=Path, default=DEFAULT_HASHES)
    parser.add_argument("--allow_legacy_field_overlap", action="store_true",
                        help="Required acknowledgement: exploratory preparation folds can share original cross-time field IDs")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=6)
    args = parser.parse_args(argv)
    if not args.allow_legacy_field_overlap:
        parser.error("Use --allow_legacy_field_overlap only for the explicitly approved exploratory sensitivity analysis")
    if not 1 <= args.workers <= 16 or args.num_workers < 0:
        parser.error("Require 1-16 hash workers and nonnegative loader workers")
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"Use a new run directory: {run_dir}")
    metadata, inventory, evidence = load_population(ROOT, args.audit_dir, args.hash_inventory)
    clean, ledger = clean_population(metadata, inventory)
    training = json.loads(evidence[0].read_text())
    if validate_manifest(clean) != training["class_names"]:
        raise ValueError("Cleanup removed an original class")
    verify_images(inventory, args.workers)
    config = freeze(run_dir, clean, ledger, training, evidence, args.num_workers, images_rehashed=True)
    print(json.dumps({"status": "prepared", "run_dir": str(run_dir), "n_images_per_arm": len(clean),
                      "tasks": list(config["tasks"]), "limitations": config["limitations"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
