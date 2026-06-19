#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tiling + Omnipose + Metadata (v2, GPU-safe)

- Validates the requested GPU index against the *visible* devices.
- Passes a proper torch.device to Cellpose/Omnipose (no raw ints).
- Produces a clear error if CUDA is unavailable or the GPU index is invalid.
"""
import os, re, csv, json, argparse, shutil, sys
from pathlib import Path
import numpy as np
import tifffile as tiff
from skimage.measure import regionprops, label
from skimage.morphology import remove_small_objects
from skimage.color import label2rgb, hsv2rgb

# ---------------- discovery ----------------
SUF = {".tif",".tiff",".png",".jpg",".jpeg"}

def list_images(root, scan_policy="guess_tiles"):
    rootp = Path(root)
    out = []
    for p in rootp.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUF:
            if scan_policy == "tiles_only":
                if p.parent.name.endswith("_tiles"): out.append(str(p))
            elif scan_policy == "guess_tiles":
                if p.parent.name.endswith("_tiles") or re.search(r"_r\d+_c\d+\.[^.]+$", p.name, re.I):
                    out.append(str(p))
            else:
                out.append(str(p))
    return sorted(out)

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def load_gray(path):
    arr = tiff.imread(path)
    if arr.ndim > 2 and arr.shape[-1] in (3,4):
        arr = arr[...,0]
    return arr

def normalize01(im):
    im = im.astype(np.float32)
    p1,p99 = np.percentile(im,(1,99))
    if p99<=p1: p1, p99 = float(im.min()), float(im.max() or 1.0)
    return np.clip((im - p1)/(p99-p1+1e-8), 0, 1)

def flow_to_rgb(flow_hw2):
    fx, fy = flow_hw2[...,0], flow_hw2[...,1]
    ang = np.arctan2(fy, fx); mag = np.sqrt(fx**2 + fy**2)
    H = (ang + np.pi)/(2*np.pi); S = np.ones_like(H); V = np.clip(mag / (np.percentile(mag,99)+1e-8), 0, 1)
    return (hsv2rgb(np.stack([H,S,V],-1))*255).astype(np.uint8)

def extract_flow_array(flows_obj):
    cand = None
    if isinstance(flows_obj,(list,tuple)):
        for it in flows_obj:
            if isinstance(it,np.ndarray) and it.ndim==3 and 2 in it.shape:
                cand = it; break
    elif isinstance(flows_obj,np.ndarray) and flows_obj.ndim==3 and 2 in flows_obj.shape:
        cand = flows_obj
    if cand is None: raise RuntimeError("No (H,W,2) flow in flows.")
    if cand.shape[0]==2: cand = np.moveaxis(cand,0,-1)
    return cand.astype(np.float32)

# ---------------- per-cell statistics ----------------
CELL_COLS = [
    "area_um2","perimeter_um","major_axis_length_um","minor_axis_length_um",
    "bbox_length_um","bbox_width_um","aspect_ratio","solidity","roundness"
]
def cell_stats(mask_lbl, px_um):
    px  = float(px_um); px2 = px*px
    rows = []
    for r in regionprops(mask_lbl):
        a_px = float(r.area)
        per  = float(getattr(r, "perimeter_crofton", r.perimeter))
        maj  = float(getattr(r, "major_axis_length", 0.0))
        mn   = float(getattr(r, "minor_axis_length", 1e-8))
        r0,c0,r1,c1 = r.bbox
        rows.append(dict(
            cell_id=int(r.label),
            area_um2=a_px*px2,
            perimeter_um=per*px,
            major_axis_length_um=maj*px,
            minor_axis_length_um=mn*px,
            bbox_length_um=(r1-r0)*px,
            bbox_width_um=(c1-c0)*px,
            aspect_ratio=(maj*px)/max(mn*px,1e-8),
            solidity=float(getattr(r,"solidity",0.0)),
            roundness=float(4*np.pi*(a_px*px2)/((per*px)**2+1e-8)),
        ))
    if rows:
        agg = {f"{k}_mean": float(np.mean([r[k] for r in rows])) for k in CELL_COLS}
    else:
        agg = {f"{k}_mean": 0.0 for k in CELL_COLS}
    agg.update(
        num_cells=len(rows),
        mean_area_um2=float(np.mean([r["area_um2"] for r in rows])) if rows else 0.0,
        median_area_um2=float(np.median([r["area_um2"] for r in rows])) if rows else 0.0,
        std_area_um2=float(np.std([r["area_um2"] for r in rows])) if rows else 0.0,
        max_cell_area_um2=max([r["area_um2"] for r in rows], default=0.0)
    )
    return rows, agg

# ---------------- parsing helpers ----------------
def _time_to_min(tok):
    m = re.fullmatch(r"(\d{1,4})(min|m|h|hr|hrs|hour|hours)?", tok.lower())
    if not m: return "NA"
    n,u = int(m.group(1)), (m.group(2) or "min")
    return str(n*60 if u in ("h","hr","hrs","hour","hours") else n)

def _find_antibiotic(parts):
    joined = "_".join(parts).lower()
    m = re.search(r"antibiotic_([a-z0-9]+)", joined)
    if m: return m.group(1).capitalize()
    for p in parts:
        pl = p.lower()
        if any(k in pl for k in ("ampicillin","kanamycin","cnd","chloramphenicol","tetracycline")):
            return re.sub(r".*_(ampicillin|kanamycin|cnd|chloramphenicol|tetracycline).*", r"\1", pl, flags=re.I).capitalize()
    return "Unknown"

def _find_strain(parts):
    for p in reversed(parts):
        pl = p.lower()
        if pl.endswith("_tiles"): continue
        if "strain" in pl or "coli" in pl: return p
    for p in reversed(parts):
        if not p.lower().endswith("_tiles"): return p
    return "NA"

def _extract_mic(text):
    s = re.sub(r"[^a-z0-9\.x]+","", text.lower())
    m = re.search(r"(?:(\d+(?:\.\d+)?)x|x(\d+(?:\.\d+)?))(mbc|mic)", s)
    if m:
        val = m.group(1) or m.group(2)
        return f"{val}x{m.group(3).upper()}"
    m = re.search(r"(\d+(?:\.\d+)?)(mbc|mic)", s)
    if m and float(m.group(1))<=8.0:
        return f"{m.group(1)}{m.group(2).upper()}"
    m = re.search(r"(mbc|mic)", s)
    return m.group(1).upper() if m else "NA"

def _extract_time(text, mic_token):
    s = re.sub(r"[^a-z0-9hminrs]+","", text.lower())
    TIME=r"(\d{1,4}(?:min|m|h|hr|hrs|hour|hours)?)"
    if mic_token!="NA":
        k = "mbc" if "MBC" in mic_token else "mic"
        for pat in (k+TIME, TIME+k):
            m = re.search(pat, s)
            if m: return _time_to_min(m.group(1))
    m = re.search(TIME, s)
    return _time_to_min(m.group(1)) if m else "NA"

def parse_from_path(rel_dir: Path, stem: str):
    parent = rel_dir.parent
    parts = list(parent.parts)
    strain = _find_strain(parts)
    antibiotic = _find_antibiotic(parts)
    mic = _extract_mic("_".join(parts)) or _extract_mic(stem) or "NA"
    if mic == "NA": mic = _extract_mic(stem)
    timepoint = _extract_time(stem, mic) if mic!="NA" else _extract_time("_".join(parts), "NA")
    replicate="NA"; r_id=None; c_id=None
    toks = stem.split("_")
    if len(toks)>=2 and re.fullmatch(r"\d+", toks[1]): replicate = toks[1]
    for t in toks:
        mr = re.fullmatch(r"r(\d+)",t); mc = re.fullmatch(r"c(\d+)",t)
        if mr: r_id = mr.group(1)
        if mc: c_id = mc.group(1)
    return strain, antibiotic, mic, timepoint, replicate, r_id, c_id

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles_root", required=True)
    ap.add_argument("--tile_output_root", required=True)
    ap.add_argument("--gpu", type=int, default=0, help="visible GPU index (0..N-1)")
    ap.add_argument("--pixel_size_um", type=float, default=0.25641)

    ap.add_argument("--scan_policy", choices=["tiles_only","guess_tiles","all"], default="guess_tiles")
    ap.add_argument("--discovery_only", action="store_true")

    # omnipose
    ap.add_argument("--mask_threshold", type=float, default=-2.4)
    ap.add_argument("--flow_threshold", type=float, default=0.5)
    ap.add_argument("--min_size", type=int, default=20)

    # filtering (ONLY these three)
    ap.add_argument("--min_cells", type=int, default=2)
    ap.add_argument("--max_artifact_area_um2", type=float, default=100.0, help="large_artifact_area")
    ap.add_argument("--max_cell_area_um2", type=float, default=30.0, help="large_cells_area")
    ap.add_argument("--filter_mode", choices=["tag","skip"], default="tag")
    
    ap.add_argument(
    "--file_list",
    default=None,
    help="Optional text file with one image path per line to process (absolute or tiles_root-relative)"
    )
    
    # sharding
    ap.add_argument("--shard", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    args = ap.parse_args()

    # ---- GPU validation & device construction ----
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        ndev = torch.cuda.device_count()
    except Exception:
        cuda_ok, ndev = False, 0
        torch = None  # type: ignore

    if cuda_ok and ndev > 0:
        if not (0 <= args.gpu < ndev):
            print(f"[FATAL] Requested --gpu {args.gpu} but only {ndev} visible device(s): 0..{ndev-1}", file=sys.stderr)
            sys.exit(2)
        # IMPORTANT: do not change CUDA_VISIBLE_DEVICES here; Slurm already masks it.
        torch_device = torch.device("cuda", args.gpu)
        use_gpu = True
    else:
        print("[WARN] CUDA not available or no visible devices; running on CPU.")
        torch_device = torch.device("cpu") if torch is not None else "cpu"
        use_gpu = False

    print(f"[DEVICE] use_gpu={use_gpu} | device={torch_device}")

    # files = list_images(args.tiles_root, args.scan_policy)
    # print(f"[INFO] discovered {len(files)} files")
    files = list_images(args.tiles_root, args.scan_policy)
    if args.file_list:
        with open(args.file_list, "r") as f:
            file_list = [line.strip() for line in f if line.strip()]

        resolved = []
        for p in file_list:
            if os.path.isabs(p):
                resolved.append(p)
            else:
                resolved.append(os.path.normpath(os.path.join(args.tiles_root, p)))

        files = resolved
        print(f"[INFO] Using file_list: {len(files)} files")
    else:
        print(f"[INFO] Discovered {len(files)} files")

    if args.discovery_only:
        ensure_dir(args.tile_output_root)
        with open(os.path.join(args.tile_output_root,"discovery_manifest.txt"),"w") as f:
            for p in files: f.write(p+"\n")
        return
    if not files: raise SystemExit("No images found.")

    if args.shard<1: args.shard=1
    files = [p for i,p in enumerate(files) if i % args.shard == args.shard_id]
    print(f"[INFO] shard {args.shard_id}/{args.shard}: {len(files)} files")

    # ---- Build model with explicit torch.device ----
    from cellpose_omni import models
    model = models.CellposeModel(
        gpu=use_gpu,
        device=torch_device,                 # <<<<<< key fix (no raw int)
        model_type='bact_phase_omni'
    )
    eval_kwargs = dict(
        channels=[0,0],
        omni=True, cluster=True,
        mask_threshold=args.mask_threshold, flow_threshold=args.flow_threshold
    )

    # ---- Main loop ----
    rows_meta = []
    for in_path in files:
        try:
            gray_raw = load_gray(in_path)
            base = normalize01(gray_raw)

            #masks, flows, _ = model.eval([gray_raw], **eval_kwargs)[:3]
            # # Try normal eval (cluster=True), but if neighbors error happens, rerun with cluster=False
            try:
                masks, flows, _ = model.eval([gray_raw], **eval_kwargs)[:3]
            except Exception as e:
                msg = str(e)
                if "Expected n_neighbors <=" in msg or "n_neighbors" in msg:
                    eval_kwargs2 = dict(eval_kwargs)
                    eval_kwargs2["cluster"] = False
                    masks, flows, _ = model.eval([gray_raw], **eval_kwargs2)[:3]
                    print(f"[WARN] cluster fallback for {in_path}: {msg}")
                else:
                    raise
            m = masks[0] if isinstance(masks,(list,tuple)) else masks
            fobj = flows[0] if isinstance(flows,(list,tuple)) else flows
            flow = extract_flow_array(fobj)

            if args.min_size>0:
                m = remove_small_objects(m, args.min_size)
                m = label(m>0, connectivity=1)

            _, agg = cell_stats(m, args.pixel_size_um)
            num_cells = agg["num_cells"]
            max_cell_area_um2 = agg["max_cell_area_um2"]
            large_artifact_area = max_cell_area_um2  # proxy

            excl = None
            if num_cells < args.min_cells:
                excl = f"low_cells:{num_cells}"
            elif args.max_cell_area_um2 is not None and max_cell_area_um2 > args.max_cell_area_um2:
                excl = f"large_cell:{max_cell_area_um2:.1f}"
            elif args.max_artifact_area_um2 is not None and large_artifact_area > args.max_artifact_area_um2:
                excl = f"artifact_area:{large_artifact_area:.1f}"

            rel = os.path.relpath(in_path, args.tiles_root)
            out_dir = os.path.join(args.tile_output_root, Path(rel).parent)
            ensure_dir(out_dir)
            stem = Path(in_path).stem
            write = not (args.filter_mode=="skip" and excl is not None)

            orig_rel   = os.path.join(Path(rel).parent, f"{stem}_original.tif")
            overlay_rel= os.path.join(Path(rel).parent, f"{stem}_mask_overlay.tif")
            mask_rel   = os.path.join(Path(rel).parent, f"{stem}_mask_raw.tif")
            flow_rel   = os.path.join(Path(rel).parent, f"{stem}_flow.tif")
            flownpy_rel= os.path.join(Path(rel).parent, f"{stem}_flow.npy")
            stats_rel  = os.path.join(Path(rel).parent, f"{stem}_cell_stats.csv")

            if write:
                # ORIGINAL
                src_path = in_path
                dst_path = os.path.join(args.tile_output_root, orig_rel)
                ensure_dir(os.path.dirname(dst_path))
                shutil.copy2(src_path, dst_path)

                # OVERLAY
                overlay = (label2rgb(m, image=base, bg_label=0, alpha=0.4)*255).astype(np.uint8)
                tiff.imwrite(os.path.join(args.tile_output_root, overlay_rel), overlay, photometric="rgb")

                # RAW MASK
                tiff.imwrite(os.path.join(args.tile_output_root, mask_rel), m.astype(np.int32))

                # FLOW VIZ + RAW
                tiff.imwrite(os.path.join(args.tile_output_root, flow_rel), flow_to_rgb(flow), photometric="rgb")
                np.save(os.path.join(args.tile_output_root, flownpy_rel), flow)

                # PER-CELL CSV
                from skimage.measure import regionprops  # local import for speed
                with open(os.path.join(args.tile_output_root, stats_rel), "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["cell_id"]+CELL_COLS)
                    w.writeheader()
                    px  = float(args.pixel_size_um); px2 = px*px
                    for r in regionprops(m):
                        a_px=float(r.area); per=float(getattr(r,"perimeter_crofton", r.perimeter))
                        maj=float(getattr(r,"major_axis_length",0.0)); mn=float(getattr(r,"minor_axis_length",1e-8))
                        r0,c0,r1,c1=r.bbox
                        w.writerow(dict(
                            cell_id=int(r.label),
                            area_um2=a_px*px2, perimeter_um=per*px,
                            major_axis_length_um=maj*px, minor_axis_length_um=mn*px,
                            bbox_length_um=(r1-r0)*px, bbox_width_um=(c1-c0)*px,
                            aspect_ratio=(maj*px)/max(mn*px,1e-8),
                            solidity=float(getattr(r,"solidity",0.0)),
                            roundness=float(4*np.pi*(a_px*px2)/((per*px)**2+1e-8)),
                        ))

            rel_dir = Path(rel).parent
            strain, antibiotic, mic, timepoint, rep, r_id, c_id = parse_from_path(rel_dir, stem)
            def abs_or_blank(p):
                return os.path.normpath(os.path.join(args.tile_output_root,p)) if (write and p) else ""

            meta = {
                "sample_id": stem,
                "strain": strain,
                "antibiotic": antibiotic,
                "mic": mic,
                "timepoint_min": timepoint,
                "replicate": rep,
                "tile_row": r_id,
                "tile_col": c_id,
                "original_tif_rel":   orig_rel if write else "",
                "mask_overlay_tif_rel": overlay_rel if write else "",
                "mask_raw_tif_rel":   mask_rel if write else "",
                "flow_tif_rel":       flow_rel if write else "",
                "flow_npy_rel":       flownpy_rel if write else "",
                "cell_stats_csv_rel": stats_rel if write else "",
                "original_tif":       abs_or_blank(orig_rel),
                "mask_overlay_tif":   abs_or_blank(overlay_rel),
                "mask_raw_tif":       abs_or_blank(mask_rel),
                "flow_tif":           abs_or_blank(flow_rel),
                "flow_npy":           abs_or_blank(flownpy_rel),
                "cell_stats_csv":     abs_or_blank(stats_rel),
                "num_cells":          num_cells,
                "mean_area_um2":      agg["mean_area_um2"],
                "median_area_um2":    agg["median_area_um2"],
                "std_area_um2":       agg["std_area_um2"],
                "max_cell_area_um2":  max_cell_area_um2,
                "exclude_reason":     excl,
                "filter_mode":        args.filter_mode,
                "area_um2_mean":                  agg["area_um2_mean"],
                "perimeter_um_mean":              agg["perimeter_um_mean"],
                "major_axis_length_um_mean":      agg["major_axis_length_um_mean"],
                "minor_axis_length_um_mean":      agg["minor_axis_length_um_mean"],
                "bbox_length_um_mean":            agg["bbox_length_um_mean"],
                "bbox_width_um_mean":             agg["bbox_width_um_mean"],
                "aspect_ratio_mean":              agg["aspect_ratio_mean"],
                "solidity_mean":                  agg["solidity_mean"],
                "roundness_mean":                 agg["roundness_mean"],
                "pixel_size_um":      args.pixel_size_um,
            }
            rows_meta.append(meta)
            tag = f"EXCL({excl})" if excl else "OK"
            print(f"[{tag}] {in_path}")

        except Exception as e:
            print(f"[ERROR] {in_path}: {e}")

    ensure_dir(args.tile_output_root)
    if rows_meta:
        rows_meta.sort(key=lambda r:(str(r["strain"]), str(r["antibiotic"]), str(r["mic"]), str(r["timepoint_min"]), str(r["sample_id"])))
        suffix = f"_shard{args.shard_id}" if args.shard > 1 else ""
        meta_csv  = os.path.join(args.tile_output_root, f"metadata{suffix}.csv")
        meta_json = os.path.join(args.tile_output_root, f"metadata{suffix}.json")
        with open(meta_csv,"w",newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_meta[0].keys()))
            w.writeheader(); w.writerows(rows_meta)
        with open(meta_json,"w") as f:
            json.dump(rows_meta, f, indent=2)
        print(f"[DONE] metadata saved:\n  CSV : {meta_csv}\n  JSON: {meta_json}")
    else:
        print("[WARN] No metadata rows collected; nothing written.")

if __name__ == "__main__":
    main()
