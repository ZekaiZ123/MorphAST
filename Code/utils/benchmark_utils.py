#!/usr/bin/env python3
"""Shared helpers for canonical AST manifests, grouping, and evaluation."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from .data_utils import infer_label


_TILE_SUFFIX_RE = re.compile(r"_r\d+_c\d+$")
_STRAIN_ID_RE = re.compile(r"strain_([^_]+)", re.IGNORECASE)


def infer_image_id(sample_id: str) -> str:
    """Collapse tile coordinates so all tiles from the same field share an image ID."""
    sample = str(sample_id or "").strip()
    return _TILE_SUFFIX_RE.sub("", sample)


def infer_strain_id(value: str) -> str:
    """Extract a compact strain identifier from legacy metadata fields."""
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return "unknown"
    match = _STRAIN_ID_RE.search(text)
    if match:
        return match.group(1)
    upper = text.upper()
    if "UPEC" in upper:
        return "UPEC"
    return text


def is_qc_pass(value) -> bool:
    """Return True when a row should be considered QC-pass by default."""
    if pd.isna(value):
        return True
    text = str(value).strip().lower()
    return text in {"", "ok", "none", "nan"}


def add_qc_columns(df: pd.DataFrame, reason_col: str = "exclude_reason") -> pd.DataFrame:
    """Add normalized QC status/reason columns without dropping rows."""
    out = df.copy()
    if reason_col not in out.columns:
        out["qc_status"] = "pass"
        out["qc_reason"] = ""
        return out
    reason = out[reason_col].fillna("").astype(str)
    out["qc_reason"] = reason
    out["qc_status"] = np.where(reason.apply(is_qc_pass), "pass", "fail")
    return out


def ensure_identity_columns(df: pd.DataFrame, sample_id_col: str = "sample_id") -> pd.DataFrame:
    """Guarantee tile/image/isolate/group IDs for downstream training and evaluation."""
    out = df.copy()

    if "tile_id" not in out.columns:
        if sample_id_col in out.columns:
            out["tile_id"] = out[sample_id_col].astype(str)
        else:
            out["tile_id"] = np.arange(len(out)).astype(str)

    if "image_id" not in out.columns:
        if sample_id_col in out.columns:
            out["image_id"] = out[sample_id_col].astype(str).apply(infer_image_id)
        elif "original_tif" in out.columns:
            out["image_id"] = out["original_tif"].astype(str).apply(lambda p: Path(p).stem)
        else:
            out["image_id"] = out["tile_id"]
    out["image_id"] = out["image_id"].fillna("").astype(str)

    if "strain_id" not in out.columns:
        if "strain" in out.columns:
            out["strain_id"] = out["strain"].astype(str).apply(infer_strain_id)
        else:
            out["strain_id"] = "unknown"

    if "isolate_id" not in out.columns:
        out["isolate_id"] = out["image_id"]
    out["isolate_id"] = out["isolate_id"].fillna("").astype(str)
    out.loc[out["isolate_id"] == "", "isolate_id"] = out.loc[out["isolate_id"] == "", "image_id"]

    if "group_id" not in out.columns:
        out["group_id"] = out["image_id"]
    out["group_id"] = out["group_id"].fillna("").astype(str)
    out.loc[out["group_id"] == "", "group_id"] = out.loc[out["group_id"] == "", "image_id"]

    return out


def add_explicit_label_columns(
    df: pd.DataFrame,
    *,
    default_label: int | None = None,
    label_col: str = "label",
    label_name_col: str = "susceptibility_label",
) -> pd.DataFrame:
    """Add explicit integer/string labels while preserving legacy compatibility."""
    out = df.copy()

    if label_col in out.columns:
        labels = pd.to_numeric(out[label_col], errors="coerce")
    else:
        labels = pd.Series(np.nan, index=out.index, dtype=float)

    missing = labels.isna()
    if missing.any():
        if default_label is None:
            inferred = out.loc[missing].apply(infer_label, axis=1).astype(float)
        else:
            inferred = pd.Series(float(default_label), index=out.index[missing])
        labels.loc[missing] = inferred

    labels = labels.fillna(0).astype(int)
    out[label_col] = labels

    if label_name_col not in out.columns:
        out[label_name_col] = labels.map({1: "sensitive", 0: "resistant"})
    else:
        names = out[label_name_col].fillna("").astype(str).str.strip()
        names = np.where(names == "", labels.map({1: "sensitive", 0: "resistant"}), names)
        out[label_name_col] = pd.Series(names, index=out.index)

    return out


def aggregate_prediction_frame(
    df: pd.DataFrame,
    entity_col: str,
    *,
    threshold: float = 0.5,
    metadata_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate tile predictions to image- or isolate-level records."""
    if metadata_cols is None:
        preferred = [
            "source_set",
            "pathogen",
            "strain_id",
            "isolate_id",
            "image_id",
            "antibiotic",
            "mic",
            "timepoint_min",
            "label",
            "susceptibility_label",
            "qc_status",
            "qc_reason",
        ]
        metadata_cols = [col for col in preferred if col in df.columns and col != entity_col]

    agg = {
        "n_tiles": ("y_prob", "count"),
        "y_prob": ("y_prob", "mean"),
        "prob_mean": ("y_prob", "mean"),
        "prob_median": ("y_prob", "median"),
        "prob_max": ("y_prob", "max"),
        "prob_min": ("y_prob", "min"),
        "y_true": ("y_true", "first"),
    }
    for col in metadata_cols:
        agg[col] = (col, "first")

    out = df.groupby(entity_col, as_index=False).agg(**agg)
    out["y_pred"] = (out["y_prob"] >= float(threshold)).astype(int)
    out["y_pred_mean"] = out["y_pred"]
    return out


def build_prediction_views(pred_df: pd.DataFrame, *, threshold: float = 0.5) -> dict[str, pd.DataFrame]:
    """Return tile/image/isolate prediction tables with a consistent schema."""
    tile_df = ensure_identity_columns(pred_df)
    if "y_pred" not in tile_df.columns:
        tile_df["y_pred"] = (tile_df["y_prob"] >= float(threshold)).astype(int)

    image_df = aggregate_prediction_frame(tile_df, "image_id", threshold=threshold)
    isolate_df = aggregate_prediction_frame(tile_df, "isolate_id", threshold=threshold)
    return {"tile": tile_df, "image": image_df, "isolate": isolate_df}


def compute_binary_metrics(y_true, y_prob, y_pred) -> dict[str, float | int | None]:
    """Compute binary metrics, returning None when AUC is undefined."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = np.asarray(y_pred).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec, rec, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = None
    return {
        "n": int(len(y_true)),
        "n_positive": int((y_true == 1).sum()),
        "n_negative": int((y_true == 0).sum()),
        "acc": acc,
        "auc": auc,
        "f1": f1,
        "precision": float(prec),
        "recall": float(rec),
    }


def summarize_prediction_views(prediction_views: dict[str, pd.DataFrame]) -> tuple[dict, dict, dict]:
    """Generate metrics, classification reports, and confusion matrices per level."""
    metrics = {}
    reports = {}
    confusions = {}
    for level, frame in prediction_views.items():
        metrics[level] = compute_binary_metrics(frame["y_true"], frame["y_prob"], frame["y_pred"])
        reports[level] = classification_report(
            frame["y_true"],
            frame["y_pred"],
            digits=4,
            output_dict=True,
            zero_division=0,
        )
        confusions[level] = confusion_matrix(frame["y_true"], frame["y_pred"], labels=[0, 1]).tolist()
    return metrics, reports, confusions


def json_ready(value):
    """Recursively replace NaN-like values with JSON-safe None."""
    if isinstance(value, dict):
        return {key: json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value
