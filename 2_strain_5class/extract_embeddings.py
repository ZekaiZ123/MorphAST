#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract fusion embeddings from the SwinFusion model for t-SNE/UMAP visualization.

Hooks the 1920-dim fusion vector before the classification head and saves
embeddings with metadata for downstream visualization.

Usage:
  python extract_embeddings.py --metadata_csv /path/to/metadata_merged_all.csv \
      --ckpt /path/to/best.pt --out_dir /path/to/output
"""

import os, json, argparse, sys, re
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
from models.swin_fusion import SwinFusionModel
from utils.data_utils import (
    seed_everything, infer_label, resolve_paths_inplace,
    add_cell_aggregates, MultiModalTiles, PREFERRED_STATS_COLS,
    infer_group_from_sample_id,
)


class EmbeddingHook:
    """Hook to capture the fusion vector h before the classification head."""
    def __init__(self, model):
        self.embeddings = []
        # Hook the LayerNorm in the head (first module)
        # The input to head is the concatenated fusion vector h
        self.hook = model.head[0].register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        # input[0] is the fusion vector h before LayerNorm
        self.embeddings.append(input[0].detach().cpu().numpy())

    def get_embeddings(self):
        return np.concatenate(self.embeddings, axis=0)

    def remove(self):
        self.hook.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata_csv", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_samples", type=int, default=None, help="Limit samples for faster processing")

    ap.add_argument("--only_ok", action="store_true")
    ap.add_argument("--exclude_reason_regex", type=str, default=None)
    ap.add_argument("--drop_no_bacteria", action="store_true")

    args = ap.parse_args()
    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    state_dict = torch.load(args.ckpt, map_location="cpu")
    model = SwinFusionModel(pretrained=False)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    # load metadata
    df = pd.read_csv(args.metadata_csv, low_memory=False)
    base_dir = Path(args.metadata_csv).parent
    resolve_paths_inplace(df, base_dir, cols=["original_tif","mask_overlay_tif","flow_tif","cell_stats_csv"])

    # cleaning
    if "exclude_reason" in df.columns and args.only_ok:
        ok = df["exclude_reason"].isna() | (df["exclude_reason"].astype(str).str.strip()=="") | \
             (df["exclude_reason"].astype(str).str.lower().isin(["ok","none","nan"]))
        df = df[ok]
    if args.exclude_reason_regex and "exclude_reason" in df.columns:
        pat = re.compile(args.exclude_reason_regex, re.IGNORECASE)
        df = df[~df["exclude_reason"].astype(str).apply(lambda s: bool(pat.search(s)))]
    if args.drop_no_bacteria and "num_cells" in df.columns:
        df = df[df["num_cells"] > 0]
    for c in ("original_tif","mask_overlay_tif","flow_tif","cell_stats_csv"):
        df = df[df[c].astype(str).apply(lambda p: Path(p).exists())]

    df = add_cell_aggregates(df)
    df["timepoint_min"] = pd.to_numeric(df["timepoint_min"], errors="coerce").fillna(0.0).astype(np.float32)
    tmax = float(df["timepoint_min"].max()) if float(df["timepoint_min"].max()) > 0 else 1.0
    df["timepoint_norm"] = (df["timepoint_min"] / tmax).astype(np.float32)

    stats_cols = [c for c in PREFERRED_STATS_COLS if c in df.columns]

    # filter to non-control, non-time0
    def _mask_control(d):
        mic = d["mic"].astype(str).str.strip()
        return ~(d["mic"].isna() | (mic == "") | (mic.str.upper() == "NA") | (mic.str.lower() == "control"))
    def _mask_time0(d):
        tp = pd.to_numeric(d["timepoint_min"], errors="coerce")
        return ~(tp.fillna(-1) == 0)

    df = df[_mask_control(df) & _mask_time0(df)].reset_index(drop=True)

    if args.max_samples and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=args.seed).reset_index(drop=True)

    print(f"[INFO] Extracting embeddings for {len(df)} samples")

    # dataset
    ds = MultiModalTiles(df, img_size=args.img_size, train=False, stats_cols=stats_cols)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # hook + forward pass
    hook = EmbeddingHook(model)

    all_probs = []
    with torch.no_grad():
        for x_o, x_m, x_f, x_s, y in loader:
            x_o, x_m, x_f, x_s = x_o.to(device), x_m.to(device), x_f.to(device), x_s.to(device)
            logit = model(x_o, x_m, x_f, x_s)
            probs = torch.sigmoid(logit).cpu().numpy()
            all_probs.extend(probs.tolist())

    embeddings = hook.get_embeddings()
    hook.remove()

    print(f"[INFO] Embeddings shape: {embeddings.shape}")

    # save embeddings
    np.save(os.path.join(args.out_dir, "embeddings.npy"), embeddings)

    # save metadata for each embedding
    meta = df[["sample_id"]].copy()
    meta["label"] = df.apply(infer_label, axis=1).values
    meta["label_str"] = meta["label"].map({1: "Sensitive", 0: "Resistant"})
    if "antibiotic" in df.columns:
        meta["antibiotic"] = df["antibiotic"].values
    if "strain" in df.columns:
        meta["strain"] = df["strain"].values
    else:
        meta["strain"] = df["sample_id"].astype(str).apply(
            lambda s: re.search(r'Strain_([^_]+)', s).group(1) if re.search(r'Strain_([^_]+)', s) else "unknown")
    meta["timepoint_min"] = df["timepoint_min"].values
    meta["y_prob"] = all_probs
    meta["y_pred"] = (np.array(all_probs) >= 0.5).astype(int)
    meta.to_csv(os.path.join(args.out_dir, "embedding_metadata.csv"), index=False)

    print(f"[DONE] Embeddings saved to {args.out_dir}")
    print(f"  embeddings.npy: {embeddings.shape}")
    print(f"  embedding_metadata.csv: {len(meta)} rows")


if __name__ == "__main__":
    main()
