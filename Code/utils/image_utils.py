#!/usr/bin/env python3
"""Shared image loading and normalization utilities."""

from pathlib import Path

import numpy as np
from PIL import Image
import tifffile as tiff


def normalize_to_uint8(x: np.ndarray) -> np.ndarray:
    """Normalize array to uint8 using 1st-99th percentile clipping."""
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, (1, 99))
    if hi <= lo:
        lo = float(x.min())
        hi = float(x.max() if x.max() > lo else lo + 1.0)
    x = np.clip((x - lo) / (hi - lo + 1e-8), 0, 1) * 255.0
    return x.astype(np.uint8)


def _array_to_pil_rgb(arr: np.ndarray) -> Image.Image:
    """Convert image-like arrays to RGB PIL images."""
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.ndim == 3 and arr.shape[-1] == 3:
        if arr.dtype != np.uint8:
            arr = normalize_to_uint8(arr)
        return Image.fromarray(arr)
    if arr.ndim > 2:
        arr = arr[..., 0]
    if arr.dtype != np.uint8:
        arr = normalize_to_uint8(arr)
    arr = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(arr)


def load_image_as_pil_rgb(path: str) -> Image.Image:
    """Load TIFF/JPEG/PNG-like inputs and convert them to RGB PIL Images."""
    suffix = Path(path).suffix.lower()
    if suffix in {".tif", ".tiff"}:
        arr = tiff.imread(path)
        return _array_to_pil_rgb(arr)

    with Image.open(path) as img:
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        elif img.mode == "RGBA":
            img = img.convert("RGB")
        return img.copy()


def load_tif_as_pil_rgb(path: str) -> Image.Image:
    """Backward-compatible wrapper for the mixed-format image loader."""
    return load_image_as_pil_rgb(path)
