#!/usr/bin/env python3
"""
Grad-CAM + embedding extraction for MultiClassImageClassifier models.

Supports both the 0-min 5-class and treated 12-class tasks.
Extracts:
  - Grad-CAM heatmaps overlaid on sample images
  - Backbone embeddings for t-SNE/UMAP visualization

Usage:
  python extract_gradcam_multiclass.py --task 0min --out_dir output/interpretability/0min
  python extract_gradcam_multiclass.py --task treated --out_dir output/interpretability/treated
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "2_strain_5class"))

from train_5class import MultiClassImageClassifier, RawStrainDataset
from utils.image_utils import load_image_as_pil_rgb


# ---- Task-specific defaults -------------------------------------------------------

TASK_CONFIGS = {
    "0min": {
        "ckpt": "/new-stg/home/zekai/Forcast_Model/Data/Benchmark_0min_5class/model_benchmark_gpu/final_model/final_model.pt",
        "manifest_csv": "/new-stg/home/zekai/Forcast_Model/Data/Benchmark_0min_5class/empty_rejector/manifest_with_empty_rejector.csv",
        "model_name": "efficientnet_b0",
        "class_names": ["E.coli5-alpha", "E.coli5-alphaKanR", "E.coli5-alphaAmpR", "AIEC", "UPEC"],
    },
    "treated": {
        "ckpt": "/new-stg/home/zekai/Forcast_Model/Model/Benchmark_Treated_Condition_QC/efficientnet_b0/fold_1/best.pt",
        "manifest_csv": "/new-stg/home/zekai/Forcast_Model/Data/Benchmark_Treated_Condition/segmentation_qc_fresh/qc_manifest/manifest_with_segmentation_qc.csv",
        "model_name": "efficientnet_b0",
        "class_names": [
            "AIEC + CND", "AIEC + Kanamycin",
            "E.coli5-alpha + Ampicillin", "E.coli5-alpha + CND", "E.coli5-alpha + Kanamycin",
            "E.coli5-alphaAmpR + Ampicillin", "E.coli5-alphaAmpR + CND",
            "E.coli5-alphaKanR + CND", "E.coli5-alphaKanR + Kanamycin",
            "UPEC + Ampicillin", "UPEC + CND", "UPEC + Kanamycin",
        ],
    },
}


# ---- Grad-CAM Hook ----------------------------------------------------------------

class GradCAMHook:
    def __init__(self, target_layer):
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
        if self.activations is None or self.gradients is None:
            return None
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze(1)
        for i in range(cam.size(0)):
            c = cam[i]
            cmin, cmax = c.min(), c.max()
            cam[i] = (c - cmin) / (cmax - cmin) if cmax > cmin else torch.zeros_like(c)
        return cam.cpu().numpy()


class EmbeddingHook:
    def __init__(self, model):
        self.embeddings = []
        self.hook = model.head[0].register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.embeddings.append(input[0].detach().cpu().numpy())

    def get_embeddings(self):
        return np.concatenate(self.embeddings, axis=0)

    def remove(self):
        self.hook.remove()


# ---- Target layer for EfficientNet-B0 via timm ------------------------------------

def get_target_layer(model):
    """Get the last convolutional layer before global pooling for Grad-CAM."""
    backbone = model.backbone
    # timm EfficientNet: backbone.conv_head or last block
    if hasattr(backbone, "conv_head"):
        return backbone.conv_head
    # fallback: last layer in the feature extractor
    modules = list(backbone.modules())
    for m in reversed(modules):
        if isinstance(m, torch.nn.Conv2d):
            return m
    raise RuntimeError("Cannot find target conv layer for Grad-CAM")


# ---- Overlay utilities -------------------------------------------------------------

def overlay_cam_on_image(img_np, cam, alpha=0.5):
    h, w = img_np.shape[:2]
    cam_resized = np.array(Image.fromarray(cam).resize((w, h), Image.BILINEAR))
    cmap = plt.cm.get_cmap("inferno")
    heatmap = (cmap(cam_resized)[..., :3] * 255).astype(np.uint8)
    return (alpha * heatmap + (1 - alpha) * img_np).astype(np.uint8)


# ---- Main extraction ---------------------------------------------------------------

def extract_all(args):
    cfg = TASK_CONFIGS[args.task]
    ckpt_path = args.ckpt or cfg["ckpt"]
    manifest_csv = args.manifest_csv or cfg["manifest_csv"]
    class_names = cfg["class_names"]
    n_classes = len(class_names)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Task={args.task}, device={device}, classes={n_classes}")

    # Load model
    model = MultiClassImageClassifier(cfg["model_name"], num_classes=n_classes, pretrained=False)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device).eval()

    # Load manifest
    df = pd.read_csv(manifest_csv)
    for col in ["drop_from_training", "drop_from_training_seg_qc"]:
        if col in df.columns:
            df = df[df[col].fillna(0).astype(int) == 0]
    df = df.reset_index(drop=True)

    # Transforms
    img_size = args.img_size
    tfms = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # ---- 1. Grad-CAM on selected samples ------------------------------------------
    target_layer = get_target_layer(model)
    gradcam_hook = GradCAMHook(target_layer)

    n_gradcam = args.n_gradcam_per_class
    gradcam_dir = out_dir / "gradcam"
    gradcam_dir.mkdir(exist_ok=True)

    for class_id, class_name in enumerate(class_names):
        class_df = df[df["class_id"] == class_id]
        if class_df.empty:
            continue
        samples = class_df.sample(n=min(n_gradcam, len(class_df)), random_state=42)
        for idx, (_, row) in enumerate(samples.iterrows()):
            img_pil = load_image_as_pil_rgb(row["image_path"])
            img_tensor = tfms(img_pil).unsqueeze(0).to(device)
            img_tensor.requires_grad_(True)

            model.zero_grad()
            logits = model(img_tensor)
            pred_class = logits.argmax(dim=1).item()
            logits[0, pred_class].backward()

            cam = gradcam_hook.compute_cam()
            if cam is not None:
                img_np = np.asarray(img_pil.resize((img_size, img_size)))
                overlay = overlay_cam_on_image(img_np, cam[0])

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(img_np)
                axes[0].set_title("Original", fontsize=11)
                axes[1].imshow(cam[0], cmap="inferno")
                axes[1].set_title("Grad-CAM", fontsize=11)
                axes[2].imshow(overlay)
                pred_name = class_names[pred_class]
                correct = "correct" if pred_class == class_id else f"WRONG ({pred_name})"
                axes[2].set_title(f"Overlay ({correct})", fontsize=11)
                for a in axes:
                    a.axis("off")
                fig.suptitle(f"True: {class_name}", fontsize=13, fontweight="bold")
                fig.tight_layout()
                safe_name = class_name.replace(" ", "_").replace("+", "").replace("/", "_")
                fig.savefig(gradcam_dir / f"gradcam_{safe_name}_{idx}.png", dpi=200, bbox_inches="tight")
                plt.close(fig)

    gradcam_hook.remove()
    print(f"[INFO] Grad-CAM saved to {gradcam_dir}")

    # ---- 2. Embedding extraction ---------------------------------------------------
    emb_hook = EmbeddingHook(model)

    if args.max_embed_samples and len(df) > args.max_embed_samples:
        embed_df = df.groupby("class_id", group_keys=False).apply(
            lambda g: g.sample(n=min(args.max_embed_samples // n_classes, len(g)), random_state=42)
        ).reset_index(drop=True)
    else:
        embed_df = df

    dataset = RawStrainDataset(embed_df, img_size=img_size, train=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    all_preds = []
    all_probs = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            probs = torch.softmax(logits, dim=1)
            all_preds.extend(probs.argmax(dim=1).cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    embeddings = emb_hook.get_embeddings()
    emb_hook.remove()

    np.save(out_dir / "embeddings.npy", embeddings)

    meta = embed_df[["image_path", "class_id", "class_name"]].copy()
    if "timepoint_min" in embed_df.columns:
        meta["timepoint_min"] = embed_df["timepoint_min"]
    if "canonical_strain" in embed_df.columns:
        meta["strain"] = embed_df["canonical_strain"]
    if "canonical_treatment" in embed_df.columns:
        meta["treatment"] = embed_df["canonical_treatment"]
    meta["y_pred"] = all_preds
    meta["y_prob_max"] = [max(p) for p in all_probs]
    meta.to_csv(out_dir / "embedding_metadata.csv", index=False)

    summary = {
        "task": args.task,
        "ckpt": ckpt_path,
        "n_classes": n_classes,
        "class_names": class_names,
        "embedding_dim": int(embeddings.shape[1]),
        "n_embeddings": int(embeddings.shape[0]),
        "n_gradcam_per_class": n_gradcam,
    }
    (out_dir / "extraction_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[DONE] Embeddings ({embeddings.shape}) saved to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["0min", "treated"])
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--manifest_csv", type=str, default=None)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--n_gradcam_per_class", type=int, default=3)
    ap.add_argument("--max_embed_samples", type=int, default=5000)
    args = ap.parse_args()
    extract_all(args)


if __name__ == "__main__":
    main()
