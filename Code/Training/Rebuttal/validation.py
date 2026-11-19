"""Auditable grouped train/validation/test partitions for reviewer response 1."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

METRICS = ("macro_f1", "accuracy", "balanced_accuracy")
ROLES = ("train", "validation", "test")
PROTOCOL = "outer_cv_with_grouped_inner_holdout_v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def validate_manifest(df: pd.DataFrame) -> list[str]:
    required = ["image_path", "group_id", "class_id", "class_name"]
    if not set(required) <= set(df):
        raise ValueError(f"Manifest requires {required}")
    if df.empty or df[required].isna().any().any():
        raise ValueError("Empty manifest or missing image/group/class identifier")
    if df.image_path.duplicated().any():
        raise ValueError("Duplicate image paths in manifest")
    if any(not str(value).strip() for value in df.group_id):
        raise ValueError("Empty group identifier")
    labels = df.class_id.to_numpy()
    if not np.array_equal(labels, labels.astype(int)):
        raise ValueError("Class IDs must be integers")
    pairs = df[["class_id", "class_name"]].drop_duplicates().sort_values("class_id")
    if pairs.class_id.duplicated().any() or pairs.class_name.duplicated().any():
        raise ValueError("Class IDs and names must have a one-to-one mapping")
    if pairs.class_id.tolist() != list(range(len(pairs))):
        raise ValueError("Class IDs must be contiguous from zero")
    return pairs.class_name.tolist()


def audit_partition(df: pd.DataFrame, roles: np.ndarray) -> dict:
    roles = np.asarray(roles)
    if len(roles) != len(df) or set(roles) != set(ROLES):
        raise ValueError("Every image must belong to exactly one train/validation/test partition")
    groups = {role: set(df.loc[roles == role, "group_id"].astype(str)) for role in ROLES}
    for first, second in combinations(ROLES, 2):
        if groups[first] & groups[second]:
            raise ValueError(f"Group leakage between {first} and {second}")
    expected = set(df.class_id)
    result = {}
    for role in ROLES:
        sub = df.loc[roles == role]
        if set(sub.class_id) != expected:
            raise ValueError(f"Missing classes in {role}; do not silently change the split seed")
        result[role] = {
            "n_samples": len(sub), "n_groups": len(groups[role]),
            "class_counts": {str(k): int(v) for k, v in sub.class_id.value_counts().sort_index().items()},
        }
    return {"group_overlap": 0, "partitions": result}


def make_nested_partitions(df: pd.DataFrame, n_folds=5, inner_splits=5, seed=42):
    """Use the first predefined inner split, without searching seeds or splits."""
    validate_manifest(df)
    labels = df.class_id.to_numpy(dtype=int)
    groups = df.group_id.astype(str).to_numpy()
    outer = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    all_test_indices = []
    partitions = []
    for fold, (development, test) in enumerate(outer.split(df, labels, groups), 1):
        inner_seed = seed + 1000 + fold
        inner = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=inner_seed)
        train_local, val_local = next(inner.split(df.iloc[development], labels[development], groups[development]))
        roles = np.full(len(df), "test", dtype=object)
        roles[development[train_local]] = "train"
        roles[development[val_local]] = "validation"
        audit = audit_partition(df, roles)
        audit.update(fold=fold, inner_seed=inner_seed, inner_split_index=0)
        partitions.append((roles, audit))
        all_test_indices.extend(test.tolist())
    if sorted(all_test_indices) != list(range(len(df))):
        raise ValueError("Outer folds do not cover each sample exactly once")
    return partitions


def classification_metrics(y_true, y_pred, n_classes: int) -> dict:
    labels = list(range(n_classes))
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def choose_candidate(selections: dict, candidates: list[str]) -> str:
    # Ties retain the declared candidate order; outer-test metrics are never inputs.
    return max(candidates, key=lambda model: selections[model]["validation_metrics"]["macro_f1"])


def fold_statistics(rows: list[dict]) -> dict:
    if len(rows) < 2:
        raise ValueError("At least two folds are required for a sample standard deviation")
    result = {"n_folds": len(rows), "std_ddof": 1}
    for metric in METRICS:
        values = np.asarray([r[metric] for r in rows], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite {metric}")
        result.update({
            metric + "_mean": float(values.mean()),
            metric + "_std": float(values.std(ddof=1)),
            metric + "_std_population": float(values.std(ddof=0)),
            metric + "_min": float(values.min()),
            metric + "_max": float(values.max()),
        })
    return result
