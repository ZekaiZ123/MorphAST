#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grad-CAM visualization for the SwinFusion model.

Generates Grad-CAM heatmaps for the Swin Transformer branch (raw images)
and ResNet18 branches (mask overlay, flow) to show what image regions
drive sensitive vs. resistant predictions.

Usage:
  python extract_gradcam.py --metadata_csv /path/to/metadata_merged_all.csv \
      --ckpt /path/to/best.pt --out_dir /path/to/output --n_samples 20
"""

import os, json, argparse, sys, re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import tifffile as tiff

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
from models.swin_fusion import SwinFusionModel
from utils.image_utils import load_tif_as_pil_rgb, normalize_to_uint8
from utils.data_utils import (
    seed_everything, infer_label, resolve_paths_inplace,
    add_cell_aggregates, MultiModalTiles, PREFERRED_STATS_COLS,
    infer_group_from_sample_id,
)

import torchvision.transforms as T


class GradCAMHook:
    """Register forward/backward hooks to capture activations and gradients."""
    def __init__(self, model, target_layer):
        self.activations = None
        self.gradients = None
        self.fwd_hook = target_layer.register_forward_hook(self._fwd)
        self.bwd_hook = target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, module, input, output):
        self.activations = output.detach()

    def _bwd(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()

    def compute_cam(self):
        """Compute Grad-CAM from stored activations/gradients."""
        if self.activations is None or self.gradients is None:
            return None
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # GAP over spatial dims
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze(1)  # [B, H, W]
        # normalize per sample
        for i in range(cam.size(0)):
            c = cam[i]
            cmin, cmax = c.min(), c.max()
            if cmax > cmin:
                cam[i] = (c - cmin) / (cmax - cmin)
            else:
                cam[i] = torch.zeros_like(c)
        return cam.cpu().numpy()


def get_target_layer_cnn(model, branch="cnn_overlay"):
    """Get the last conv layer of a ResNet18 branch."""
    cnn = getattr(model, branch)
    return cnn.layer4[-1].conv2


def get_target_layer_swin(model):
    """Get the last stage normalization of the Swin Transformer."""
    return model.swin.norm


def overlay_cam_on_image(img_np, cam, alpha=0.5, cmap="viridis"):
    """Overlay Grad-CAM heatmap on the original image."""
    h, w = img_np.shape[:2]
    cam_resized = np.array(Image.fromarray(cam).resize((w, h), Image.BILINEAR))
    colormap = plt.cm.get_cmap(cmap)
    heatmap = colormap(cam_resized)[..., :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    overlay = (alpha * heatmap + (1 - alpha) * img_np).astype(np.uint8)
    return overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata_csv", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_samples", type=int, default=20, help="Number of samples per class to visualize")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)

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

    if "exclude_reason" in df.columns and args.only_ok:
        ok = df["exclude_reason"].isna() | (df["exclude_reason"].astype(str).str.strip()=="") | \
             (df["exclude_reason"].astype(str).str.lower().isin(["ok","none","nan"]))
        df = df[ok]
    for c in ("original_tif","mask_overlay_tif","flow_tif","cell_stats_csv"):
        df = df[df[c].astype(str).apply(lambda p: Path(p).exists())]

    df = add_cell_aggregates(df)
    df["timepoint_min"] = pd.to_numeric(df["timepoint_min"], errors="coerce").fillna(0.0).astype(np.float32)
    tmax = float(df["timepoint_min"].max()) if float(df["timepoint_min"].max()) > 0 else 1.0
    df["timepoint_norm"] = (df["timepoint_min"] / tmax).astype(np.float32)

    stats_cols = [c for c in PREFERRED_STATS_COLS if c in df.columns]
    df["label"] = df.apply(infer_label, axis=1)

    # select samples: n_samples sensitive + n_samples resistant, non-control, non-time0
    def _mask_control(d):
        mic = d["mic"].astype(str).str.strip()
        return ~(d["mic"].isna() | (mic == "") | (mic.str.upper() == "NA") | (mic.str.lower() == "control"))
    def _mask_time0(d):
        tp = pd.to_numeric(d["timepoint_min"], errors="coerce")
        return ~(tp.fillna(-1) == 0)

    df_filtered = df[_mask_control(df) & _mask_time0(df)].reset_index(drop=True)

    sensitive = df_filtered[df_filtered["label"] == 1].sample(n=min(args.n_samples, len(df_filtered[df_filtered["label"]==1])),
                                                               random_state=args.seed)
    resistant = df_filtered[df_filtered["label"] == 0].sample(n=min(args.n_samples, len(df_filtered[df_filtered["label"]==0])),
                                                               random_state=args.seed)
    samples = pd.concat([sensitive, resistant]).reset_index(drop=True)

    # transforms
    mean = (0.485, 0.456, 0.406); std = (0.229, 0.224, 0.225)
    tf = T.Compose([T.Resize((args.img_size, args.img_size)), T.ToTensor(), T.Normalize(mean, std)])

    # Hook into ResNet18 overlay branch (Grad-CAM works best with conv layers)
    hook_overlay = GradCAMHook(model, get_target_layer_cnn(model, "cnn_overlay"))
    hook_flow = GradCAMHook(model, get_target_layer_cnn(model, "cnn_flow"))

    results = []
    for idx in range(len(samples)):
        row = samples.iloc[idx]
        label = int(row["label"])
        label_str = "sensitive" if label == 1 else "resistant"

        # load images
        img_orig = load_tif_as_pil_rgb(row["original_tif"])
        img_mask = load_tif_as_pil_rgb(row["mask_overlay_tif"])
        img_flow = load_tif_as_pil_rgb(row["flow_tif"])

        x_o = tf(img_orig).unsqueeze(0).to(device)
        x_m = tf(img_mask).unsqueeze(0).to(device)
        x_f = tf(img_flow).unsqueeze(0).to(device)

        stats_vals = row[stats_cols].values.astype(np.float32)
        x_s = torch.tensor(stats_vals, dtype=torch.float32).unsqueeze(0).to(device)

        # forward + backward for Grad-CAM
        x_o.requires_grad_(True)
        model.zero_grad()
        logit = model(x_o, x_m, x_f, x_s)
        prob = torch.sigmoid(logit).item()
        pred = 1 if prob >= 0.5 else 0

        # backward pass for the predicted class
        logit.backward()

        # get CAMs
        cam_overlay = hook_overlay.compute_cam()
        cam_flow = hook_flow.compute_cam()

        # original image as numpy for overlay
        img_orig_np = np.array(img_orig.resize((args.img_size, args.img_size)))
        if img_orig_np.ndim == 2:
            img_orig_np = np.stack([img_orig_np]*3, axis=-1)

        # save figure
        sample_id = str(row.get("sample_id", f"sample_{idx}"))
        safe_id = re.sub(r'[^\w\-.]', '_', sample_id)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(img_orig_np)
        axes[0].set_title(f"Original ({label_str})")
        axes[0].axis("off")

        if cam_overlay is not None:
            overlay_img = overlay_cam_on_image(img_orig_np, cam_overlay[0])
            axes[1].imshow(overlay_img)
        axes[1].set_title("Grad-CAM (Mask branch)")
        axes[1].axis("off")

        if cam_flow is not None:
            flow_overlay = overlay_cam_on_image(img_orig_np, cam_flow[0])
            axes[2].imshow(flow_overlay)
        axes[2].set_title("Grad-CAM (Flow branch)")
        axes[2].axis("off")

        axes[3].text(0.5, 0.5, f"True: {label_str}\nPred: {'sensitive' if pred==1 else 'resistant'}\nProb: {prob:.3f}",
                     ha="center", va="center", fontsize=14, transform=axes[3].transAxes)
        axes[3].axis("off")

        plt.suptitle(f"Sample: {sample_id}", fontsize=10)
        plt.tight_layout()
        fig_path = os.path.join(args.out_dir, f"gradcam_{safe_id}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()

        results.append({
            "sample_id": sample_id, "label": label_str, "pred": "sensitive" if pred==1 else "resistant",
            "prob": prob, "figure": fig_path
        })

    hook_overlay.remove()
    hook_flow.remove()

    with open(os.path.join(args.out_dir, "gradcam_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"[DONE] Grad-CAM visualizations saved to {args.out_dir} ({len(results)} samples)")


if __name__ == "__main__":
    main()
