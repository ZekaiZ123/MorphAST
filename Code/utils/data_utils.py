#!/usr/bin/env python3
"""Shared data utilities for training and inference."""

import re
import math
import random
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from .image_utils import load_tif_as_pil_rgb


# ----------------------- seeding -----------------------
def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------- label inference -----------------------
def normalize_binary_label(value):
    """Normalize common sensitive/resistant encodings to 1/0."""
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        if int(value) in (0, 1):
            return int(value)
    if isinstance(value, (float, np.floating)) and not np.isnan(value):
        if float(value) in (0.0, 1.0):
            return int(float(value))

    text = str(value).strip().lower()
    if not text or text == "nan":
        return None
    if text in {"1", "true", "sensitive", "susceptible", "sens", "s"}:
        return 1
    if text in {"0", "false", "resistant", "resistance", "res", "r"}:
        return 0
    if "sensitive" in text or "susceptible" in text or "sentitive" in text:
        return 1
    if "resistant" in text:
        return 0
    return None


def infer_label(row: pd.Series) -> int:
    """Infer binary label from metadata: Sensitive=1, Resistant=0."""
    for col in ("label", "susceptibility_label", "susceptibility", "y_true"):
        val = normalize_binary_label(row.get(col, None))
        if val is not None:
            return val
    for col in ("strain", "sample_id"):
        val = normalize_binary_label(row.get(col, ""))
        if val is not None:
            return val
    return 0


# ----------------------- path resolution -----------------------
def resolve_paths_inplace(df: pd.DataFrame, base_dir: Path, cols):
    """Resolve relative path columns (*_rel) to absolute paths."""
    for c in cols:
        c_rel = c + "_rel"
        if c not in df.columns and c_rel in df.columns:
            df[c] = df[c_rel].apply(lambda p: str((base_dir / Path(str(p))).resolve()))


# ----------------------- cell stat aggregation -----------------------
CELL_FEATURES = [
    ("solidity", "solidity_mean"),
    ("major_axis_length_um", "major_axis_length_um_mean"),
    ("minor_axis_length_um", "minor_axis_length_um_mean"),
    ("bbox_length_um", "bbox_length_um_mean"),
    ("bbox_width_um", "bbox_width_um_mean"),
    ("aspect_ratio", "aspect_ratio_mean"),
    ("roundness", "roundness_mean"),
]


@lru_cache(maxsize=100000)
def _agg_from_cell_csv(csv_path: str) -> dict:
    try:
        dfc = pd.read_csv(csv_path)
    except Exception:
        return {new: np.nan for _, new in CELL_FEATURES}
    res = {}
    for raw, new in CELL_FEATURES:
        if raw in dfc.columns:
            res[new] = float(pd.to_numeric(dfc[raw], errors="coerce").dropna().mean())
        else:
            res[new] = np.nan
    return res


def add_cell_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-tile aggregated cell morphology stats from cell_stats_csv files."""
    if "cell_stats_csv" not in df.columns:
        raise ValueError("metadata.csv missing 'cell_stats_csv' column")
    agg_rows = []
    for p in df["cell_stats_csv"].astype(str).tolist():
        agg_rows.append(_agg_from_cell_csv(p))
    agg_df = pd.DataFrame(agg_rows).fillna(0.0)
    return pd.concat([df.reset_index(drop=True), agg_df.reset_index(drop=True)], axis=1)


# ----------------------- preferred stats columns -----------------------
PREFERRED_STATS_COLS = [
    "timepoint_norm", "num_cells", "mean_area_um2", "median_area_um2", "std_area_um2",
    "area_um2_mean", "perimeter_um_mean",
    "major_axis_length_um_mean", "minor_axis_length_um_mean",
    "bbox_length_um_mean", "bbox_width_um_mean",
    "aspect_ratio_mean", "solidity_mean", "roundness_mean"
]


# ----------------------- dataset -----------------------
class MultiModalTiles(Dataset):
    """Dataset that returns (original, mask_overlay, flow, stats, label) tuples."""

    def __init__(self, df: pd.DataFrame, img_size=224,
                 img_cols=("original_tif", "mask_overlay_tif", "flow_tif"),
                 stats_cols=None, train=True, augment=True):
        self.df = df.reset_index(drop=True)
        self.img_cols = img_cols
        self.stats_cols = list(stats_cols)
        self.train = train

        for c in self.img_cols:
            if c not in self.df.columns:
                raise ValueError(f"Missing column in metadata.csv: {c}")

        self.labels = self.df.apply(infer_label, axis=1).astype(np.int64).values

        for sc in self.stats_cols:
            if sc not in self.df.columns:
                self.df[sc] = 0.0
        self.df[self.stats_cols] = self.df[self.stats_cols].fillna(0.0)

        mean = (0.485, 0.456, 0.406); std = (0.229, 0.224, 0.225)
        base = [T.Resize((img_size, img_size))]
        aug = [T.RandomHorizontalFlip(), T.RandomVerticalFlip()]
        to_t = [T.ToTensor(), T.Normalize(mean, std)]
        self.train_tfms = T.Compose(base + (aug if augment else []) + to_t)
        self.val_tfms   = T.Compose(base + to_t)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        imgs = []
        for c in self.img_cols:
            p = row[c]
            x = load_tif_as_pil_rgb(p)
            x = (self.train_tfms if self.train else self.val_tfms)(x)
            imgs.append(x)
        stats = torch.tensor(row[self.stats_cols].values.astype(np.float32), dtype=torch.float32)
        y = torch.tensor(float(self.labels[idx]), dtype=torch.float32)
        return imgs[0], imgs[1], imgs[2], stats, y


# ----------------------- group/split utilities -----------------------
def infer_group_from_sample_id(sid: str) -> str:
    """Collapse tile suffixes like '_r0_c2' so tiles from the same field stay together."""
    return re.sub(r"_r\d+_c\d+$", "", str(sid))


def is_control_row(row, mic_col: str = "mic", control_regex: str = r"^(?i:control|ctrl)$") -> bool:
    if "is_control" in row and pd.notna(row["is_control"]):
        try:
            return bool(int(row["is_control"])) if isinstance(row["is_control"], (int, float, str)) else bool(row["is_control"])
        except Exception:
            return bool(row["is_control"])
    mic_val = str(row.get(mic_col, "")).strip()
    return bool(re.match(control_regex, mic_val)) if mic_val is not None else False


def make_group_stratified_test_split(
    df: pd.DataFrame,
    test_split: float,
    seed: int,
    *,
    antibiotic_col: str = "antibiotic",
    mic_col: str = "mic",
    time_col: str = "timepoint_min",
    sample_id_col: str = "sample_id",
    test_min_per_stratum: int = 1,
    exclude_control_mic_test: bool = True,
    control_mic_regex: str = r"^(?i:control|ctrl)$",
):
    """
    Returns boolean masks (is_test, is_rest) ensuring TEST coverage
    across all (antibiotic, mic, timepoint_min) strata.
    """
    rng = np.random.RandomState(seed)

    for col in (antibiotic_col, mic_col, time_col, sample_id_col):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found. Available: {list(df.columns)}")

    groups = df[sample_id_col].astype(str).map(infer_group_from_sample_id)
    strata_key = df[antibiotic_col].astype(str) + "||" + df[mic_col].astype(str) + "||" + df[time_col].astype(str)

    stratum_to_groups = {}
    for s, g in zip(strata_key, groups):
        stratum_to_groups.setdefault(s, set()).add(g)

    chosen_test_groups = set()
    for s, gset in stratum_to_groups.items():
        glist = sorted(gset)
        n = len(glist)
        if n == 0:
            continue
        target = max(test_min_per_stratum, int(math.ceil(test_split * n)))
        target = min(target, n)
        picked = list(glist)
        rng.shuffle(picked)
        picked = picked[:target]
        chosen_test_groups.update(picked)

    if exclude_control_mic_test:
        pre_is_test = groups.isin(chosen_test_groups).values
        mask_control = df.apply(lambda r: is_control_row(r, mic_col=mic_col, control_regex=control_mic_regex), axis=1).values
        test_groups_kept = set(groups[pre_is_test & ~mask_control].unique())
        chosen_test_groups = test_groups_kept

    is_test = groups.isin(chosen_test_groups).values
    is_rest = ~is_test
    return is_test, is_rest
