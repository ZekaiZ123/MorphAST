#!/usr/bin/env python3
"""Audit byte-identical image conflicts across the full retained treated cohorts.

No labels, images, manifests, models, or training jobs are modified. Exact file
identity establishes duplicate input, not which condition annotation is correct.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from audit_error_cases import ROOT, PROTOCOL, UNKNOWN, load_image_as_pil_rgb, markdown_table, validate_predictions
from validation import sha256, write_json


def hash_file(path):
    try:
        return {"image_path": path, "original_sha256": sha256(Path(path)), "hash_status": "ok", "hash_error": ""}
    except OSError as error:
        return {"image_path": path, "original_sha256": "", "hash_status": "failed", "hash_error": str(error)}


def summarize_hash_groups(predictions, inventory):
    if inventory.image_path.duplicated().any() or predictions.image_path.duplicated().any():
        raise ValueError("Duplicate image paths in hash join")
    if not set(predictions.image_path) <= set(inventory.image_path):
        raise ValueError("Incomplete hash inventory")
    joined = predictions.merge(inventory, on="image_path", how="left", validate="one_to_one")
    valid = joined.loc[joined.hash_status.eq("ok")]
    groups = valid.groupby("original_sha256", sort=True).agg(
        n_images=("image_path", "size"), n_condition_labels=("class_name", "nunique"),
        n_folds=("fold", "nunique"), n_field_groups=("group_id", "nunique"),
        conditions=("class_name", lambda s: " | ".join(sorted(set(s)))),
        timepoints=("timepoint_min", lambda s: " | ".join(map(str, sorted(set(s))))),
        source_collections=("source_folder", lambda s: " | ".join(sorted(set(s)))),
    ).reset_index()
    groups["is_duplicate"] = groups.n_images.gt(1)
    groups["has_label_conflict"] = groups.n_condition_labels.gt(1)
    groups["spans_heldout_folds"] = groups.n_folds.gt(1)
    joined = joined.merge(groups[["original_sha256", "n_images", "n_condition_labels", "n_folds", "n_field_groups",
                                  "is_duplicate", "has_label_conflict", "spans_heldout_folds"]],
                          on="original_sha256", how="left", validate="many_to_one")
    # Failed hashes remain unknown, never silently classified as unique.
    joined["exact_error"] = joined.y_true.ne(joined.y_pred)
    joined["batch_id"] = UNKNOWN
    joined["acquisition_session_id"] = UNKNOWN
    joined["evaluation_protocol"] = PROTOCOL
    return joined, groups


def render_conflict_examples(joined, out):
    """Display the first four conflicting hashes, not a visually selected subset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conflicting = joined.loc[joined.has_label_conflict.eq(True)]
    hashes = sorted(conflicting.original_sha256.unique())[:4]
    if not hashes:
        return
    height = 5 * len(hashes) + 1
    fig, axes = plt.subplots(len(hashes), 2, figsize=(11, height), squeeze=False)
    selected = []
    for i, digest in enumerate(hashes):
        members = conflicting.loc[conflicting.original_sha256.eq(digest)].sort_values(["class_name", "image_path"])
        members = members.drop_duplicates("class_name").iloc[:2]
        for j, row in enumerate(members.itertuples()):
            ax = axes[i, j]
            ax.axis("off")
            # Check these displayed original files again, independent of the saved table.
            if sha256(Path(row.image_path)) != digest:
                raise ValueError(f"Source image changed after hash audit: {row.image_path}")
            image = load_image_as_pil_rgb(row.image_path)
            image.thumbnail((512, 512))
            ax.imshow(image)
            ax.set_title(f"Recorded label: {row.class_name.replace('E.coli5-alpha', 'DH5alpha')}\n"
                         f"{row.timepoint_min} min | fold {row.fold} | batch UNKNOWN\n"
                         f"File: {Path(row.image_path).name}\nSHA256: {digest[:20]}...", fontsize=8)
            selected.append({"example_row": i + 1, "image_path": row.image_path, "image_uid": row.image_uid,
                             "recorded_condition": row.class_name, "timepoint_min": row.timepoint_min,
                             "fold": row.fold, "batch_id": UNKNOWN, "original_sha256": digest})
    fig.suptitle("Byte-identical files with conflicting treatment labels\n"
                 "Same hash within each row. Labels are unresolved; no biological interpretation is assigned.", fontsize=11)
    fig.subplots_adjust(left=.02, right=.98, bottom=.025, top=1 - 2 / height, hspace=.65, wspace=.12)
    fig.savefig(out / "conflicting_label_examples.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(selected).to_csv(out / "conflicting_label_examples.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    predictions, sources = {}, []
    for cohort in [13, 15]:
        model = args.root / f"Model/Benchmark_Treated_Condition_{cohort}class_QC/efficientnet_b0"
        settings_path = model / "training_args.json"
        settings = json.loads(settings_path.read_text())
        manifest_path = args.root / settings["manifest_csv"]
        pred_path = model / "crossval_predictions.csv"
        predictions[cohort] = validate_predictions(pd.read_csv(pred_path), pd.read_csv(manifest_path), settings)
        sources += [{"path": str(p), "sha256": sha256(p)} for p in [settings_path, manifest_path, pred_path]]
    paths = sorted(set().union(*(set(df.image_path) for df in predictions.values())))
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for i, result in enumerate(executor.map(hash_file, paths), 1):
            records.append(result)
            if i % 1000 == 0:
                print(f"[HASH] {i}/{len(paths)}", flush=True)
    inventory = pd.DataFrame(records)
    inventory.to_csv(out / "all_retained_image_hashes.csv", index=False)
    summaries = {}
    lines = ["# Full retained-cohort exact-duplicate audit", "",
             "This audit hashes all retained images, not only errors. Files shared between the 13- and "
             "15-class cohorts are read once but analyzed separately. SHA256 is computed over the entire "
             "source file. Exact byte identity is stronger evidence than visual resemblance. Nonidentical "
             "hashes do not exclude recompression, near-duplicates, or repeated physical fields.", "",
             "Batch and session identifiers remain UNKNOWN. These are legacy CV predictions, not corrected "
             "outer-test results. No data, labels, training jobs or models were changed.", ""]
    for cohort, pred in predictions.items():
        joined, groups = summarize_hash_groups(pred, inventory)
        folder = out / f"treated_{cohort}class"
        folder.mkdir()
        joined.to_csv(folder / "all_predictions_with_hashes.csv", index=False)
        duplicate = groups.loc[groups.is_duplicate]
        conflict = groups.loc[groups.has_label_conflict]
        duplicate.to_csv(folder / "duplicate_groups.csv", index=False)
        conflict.to_csv(folder / "conflicting_label_groups.csv", index=False)
        joined.loc[joined.has_label_conflict.eq(True)].to_csv(folder / "conflicting_label_images.csv", index=False)
        joined.loc[joined.spans_heldout_folds.eq(True)].to_csv(folder / "cross_fold_duplicate_images.csv", index=False)
        by_condition = joined.groupby(["class_name", "timepoint_min", "source_root", "source_folder", "batch_id", "acquisition_session_id"], dropna=False).agg(
            n=("image_path", "size"), n_duplicate_images=("is_duplicate", "sum"),
            n_label_conflict_images=("has_label_conflict", "sum"),
            n_cross_fold_images=("spans_heldout_folds", "sum"), n_errors=("exact_error", "sum"),
        ).reset_index()
        by_condition.to_csv(folder / "counts_by_condition_time_source.csv", index=False)
        errors = joined.loc[joined.exact_error]
        summaries[str(cohort)] = {"n": len(joined), "n_hash_failed": int(joined.hash_status.ne("ok").sum()),
                                  "n_duplicate_groups": len(duplicate), "n_duplicate_images": int(duplicate.n_images.sum()),
                                  "n_label_conflict_groups": len(conflict), "n_label_conflict_images": int(conflict.n_images.sum()),
                                  "n_cross_fold_groups": int(groups.spans_heldout_folds.sum()),
                                  "n_errors": len(errors), "n_errors_with_conflicting_labels": int(errors.has_label_conflict.eq(True).sum())}
        focus = joined.loc[joined.class_name.eq("E.coli5-alpha + Kanamycin")].copy()
        focus["label_conflict_status"] = focus.has_label_conflict.map({True: "conflict", False: "no_exact_conflict_detected"}).fillna("hash_failed")
        rates = focus.groupby(["timepoint_min", "label_conflict_status"], dropna=False).agg(
            n=("image_path", "size"), errors=("exact_error", "sum"),
        ).reset_index()
        rates["exact_correct_fraction"] = (1 - rates.errors / rates.n).round(4)
        rates.to_csv(folder / "DH5alpha_kanamycin_error_counts_by_conflict.csv", index=False)
        render_conflict_examples(joined, folder)
        combinations = conflict.groupby(["conditions", "timepoints"], dropna=False).agg(
            hash_groups=("original_sha256", "size"), image_rows=("n_images", "sum"),
        ).reset_index()
        lines += [f"## {cohort}-class cohort", "", "```json", json.dumps(summaries[str(cohort)], indent=2), "```", "",
                  markdown_table(combinations), "", "DH5alpha-kanamycin errors by exact-duplicate conflict status:", "",
                  markdown_table(rates), "",
                  f"[All conflicting-label image paths and predictions](treated_{cohort}class/conflicting_label_images.csv) | "
                  f"[Cross-fold duplicate image paths](treated_{cohort}class/cross_fold_duplicate_images.csv) | "
                  f"[Byte-identical paired examples](treated_{cohort}class/conflicting_label_examples.png)", ""]
    lines += ["## Interpretation and required follow-up", "",
              "Identical input files assigned to different treatments constitute unresolved label/provenance "
              "conflicts, not evidence that two treatments caused merely similar morphology. The original "
              "acquisition records are needed to establish which assignment, if either, is correct. "
              "Do not relabel a file according to the model prediction or decide which copy is genuine from "
              "the filename alone.", "",
              "A duplicate hash spanning held-out folds means image content is not isolated by the existing "
              "field-group split. An inner-validation checkpoint split does not by itself fix this problem. "
              "The queued reviewer-1 run retains these manifests and grouping identifiers; correcting checkpoint "
              "selection alone will not establish duplicate-free or batch-independent evaluation. "
              "Resolve contradictory labels using acquisition records, then regenerate partitions with "
              "verified physical-field/duplicate groups and, when available, acquisition-level grouping. "
              "Any exclusion policy and revised denominators must be reported before retraining. "
              "The conflict-status tables are descriptive post hoc diagnostics, not performance estimates "
              "for a cleaned, independently evaluated dataset.", ""]
    (out / "duplicate_conflict_audit.md").write_text("\n".join(lines))
    sources.append({"path": str(Path(__file__)), "sha256": sha256(Path(__file__))})
    sources.append({"path": str(ROOT / "Code/Training/Rebuttal/audit_error_cases.py"),
                    "sha256": sha256(ROOT / "Code/Training/Rebuttal/audit_error_cases.py")})
    write_json(out / "summary.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
                                     "n_unique_source_paths": len(paths), "cohorts": summaries, "sources": sources})
    print(f"[DONE] {out}", flush=True)
    if inventory.hash_status.ne("ok").any():
        raise SystemExit("Some source images could not be hashed; see all_retained_image_hashes.csv")


if __name__ == "__main__":
    main()
