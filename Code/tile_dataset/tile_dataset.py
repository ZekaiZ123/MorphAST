#!/usr/bin/env python3
"""
Tile a dataset of microscopy images into an RxC grid (default 3x3).

- Recursively walks input_root and preserves relative folder structure.
- Robust TIFF handling with tifffile when available (16-bit, big tiffs).
- Choose crop (center) or pad (edge) to handle non-divisible sizes.
- Concurrency for speed (--workers).
- Save as TIFF (default) or PNG/JPG.
- Optional per-image subfolder layout or flat layout.
- Deterministic naming: <stem>_r{row}_c{col}.<ext>
"""
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional imports
_tifffile = None
try:
    import tifffile as _tifffile  # type: ignore
except Exception:
    _tifffile = None

try:
    from PIL import Image
except ImportError as e:
    raise SystemExit("Pillow is required. Install with: pip install pillow") from e


def find_images(root: Path, exts: Tuple[str, ...]) -> List[Path]:
    return sorted([p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in exts])


def _center_crop_to_multiple(arr: np.ndarray, rows: int, cols: int) -> np.ndarray:
    h, w = arr.shape[:2]
    tile_h = h // rows
    tile_w = w // cols
    if tile_h == 0 or tile_w == 0:
        raise ValueError(f"Image too small for {rows}x{cols} tiling: {h}x{w}")
    new_h = tile_h * rows
    new_w = tile_w * cols
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    return arr[top:top+new_h, left:left+new_w, ...]


def _pad_to_multiple(arr: np.ndarray, rows: int, cols: int) -> np.ndarray:
    h, w = arr.shape[:2]
    tile_h = (h + rows - 1) // rows  # ceil
    tile_w = (w + cols - 1) // cols  # ceil
    new_h = tile_h * rows
    new_w = tile_w * cols
    pad_h = new_h - h
    pad_w = new_w - w
    pad_top = pad_h // 2
    pad_left = pad_w // 2
    pad_bottom = pad_h - pad_top
    pad_right = pad_w - pad_left
    if arr.ndim == 2:
        pad_cfg = ((pad_top, pad_bottom), (pad_left, pad_right))
    else:
        pad_cfg = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    return np.pad(arr, pad_cfg, mode='edge')


def _tile_array(arr: np.ndarray, rows: int, cols: int) -> List[np.ndarray]:
    h, w = arr.shape[:2]
    tile_h = h // rows
    tile_w = w // cols
    tiles = []
    for r in range(rows):
        for c in range(cols):
            y0 = r * tile_h
            x0 = c * tile_w
            tile = arr[y0:y0+tile_h, x0:x0+tile_w, ...] if arr.ndim == 3 else arr[y0:y0+tile_h, x0:x0+tile_w]
            tiles.append(tile)
    return tiles


def _imread(path: Path) -> np.ndarray:
    if _tifffile is not None and path.suffix.lower() in ('.tif', '.tiff'):
        arr = _tifffile.imread(str(path))
        # If multi-page, take the first page
        if arr.ndim == 3 and arr.shape[0] not in (3, 4):  # (pages,H,W)
            arr = arr[0]
        return arr
    with Image.open(path) as im:
        if im.mode == 'P':
            im = im.convert('RGB')
        return np.array(im)


def _ensure_pil_image(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        if arr.dtype == np.uint16:
            return Image.fromarray(arr, mode='I;16')
        if not np.issubdtype(arr.dtype, np.integer):
            arr = np.clip(arr, 0, 1)
            arr = (arr * 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return Image.fromarray(arr)
    else:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0, 1)
            arr = (arr * 255).astype(np.uint8)
        elif arr.dtype not in (np.uint8, np.uint16):
            arr = arr.astype(np.uint8)
        if arr.dtype == np.uint16:
            arr = (arr / 257).astype(np.uint8)
        return Image.fromarray(arr)


def _imsave(path: Path, arr: np.ndarray, fmt: str):
    fmt = fmt.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt in ('tif', 'tiff'):
        if _tifffile is None:
            img = _ensure_pil_image(arr)
            img.save(path, format='TIFF')
        else:
            _tifffile.imwrite(str(path), arr, photometric='minisblack' if arr.ndim == 2 else 'rgb')
    elif fmt in ('png', 'jpg', 'jpeg'):
        img = _ensure_pil_image(arr)
        if fmt in ('jpg', 'jpeg'):
            img.save(path, format='JPEG', quality=95, subsampling=0)
        else:
            img.save(path, format='PNG', compress_level=3)
    else:
        raise ValueError(f"Unsupported output format: {fmt}")


def process_one(in_path: Path,
                input_root: Path,
                output_root: Path,
                rows: int,
                cols: int,
                mode: str,
                layout: str,
                out_ext: str) -> Tuple[Path, int, Optional[str]]:
    """Return (file, n_tiles, error_message)."""
    try:
        rel_dir = in_path.parent.relative_to(input_root)
        stem = in_path.stem
        arr = _imread(in_path)

        if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
            # channels-first -> HWC
            arr = np.moveaxis(arr, 0, -1)

        arr2 = _center_crop_to_multiple(arr, rows, cols) if mode == 'crop' else _pad_to_multiple(arr, rows, cols)
        tiles = _tile_array(arr2, rows, cols)

        save_dir = (output_root / rel_dir / f"{stem}_tiles") if layout == 'subfolder' else (output_root / rel_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for idx, t in enumerate(tiles):
            r = idx // cols
            c = idx % cols
            out_name = f"{stem}_r{r}_c{c}.{out_ext}"
            out_path = save_dir / out_name
            _imsave(out_path, t, out_ext)
            saved += 1

        return (in_path, saved, None)
    except Exception as e:
        return (in_path, 0, str(e))


def main():
    parser = argparse.ArgumentParser(description="Tile images into an RxC grid while preserving folder structure.")
    parser.add_argument("--input_root", required=True, type=Path)
    parser.add_argument("--output_root", required=True, type=Path)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--mode", choices=["crop", "pad"], default="crop",
                        help="Handle non-divisible sizes by center-cropping or edge-padding.")
    parser.add_argument("--layout", choices=["subfolder", "flat"], default="subfolder",
                        help="Save into per-image subfolder or keep flat in rel dir.")
    parser.add_argument("--exts", type=str, default=".tif,.tiff,.png,.jpg,.jpeg",
                        help="Comma-separated extensions to read.")
    parser.add_argument("--out_ext", type=str, default="tif", help="Output format: tif,tiff,png,jpg,jpeg")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers.")
    parser.add_argument("--dry_run", action="store_true", help="Scan and print what would be processed, then exit.")
    args = parser.parse_args()

    input_root: Path = args.input_root.expanduser().resolve()
    output_root: Path = args.output_root.expanduser().resolve()
    out_ext = args.out_ext.lower().lstrip('.')
    exts = tuple(e.strip().lower() if e.strip().startswith('.') else f".{e.strip().lower()}"
                 for e in args.exts.split(','))

    if not input_root.exists():
        raise SystemExit(f"Input root does not exist: {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    files = find_images(input_root, exts)
    if not files:
        print("No images found with extensions:", exts)
        return

    print(f"Found {len(files)} images under: {input_root}")
    if args.dry_run:
        for p in files[:20]:
            print(" -", p.relative_to(input_root))
        if len(files) > 20:
            print(f" ... and {len(files)-20} more")
        return

    n_ok = 0
    n_err = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [
            ex.submit(process_one, p, input_root, output_root,
                      args.rows, args.cols, args.mode, args.layout, out_ext)
            for p in files
        ]
        for i, fut in enumerate(as_completed(futs), 1):
            in_path, saved, err = fut.result()
            rel = in_path.relative_to(input_root)
            if err is None:
                n_ok += 1
                print(f"[{i}/{len(files)}] OK  | {rel} -> {saved} tiles")
            else:
                n_err += 1
                print(f"[{i}/{len(files)}] ERR | {rel} -> {err}")

    print(f"Done. Success: {n_ok}, Errors: {n_err}")


if __name__ == "__main__":
    main()
