#!/usr/bin/env python3
"""Reproduce the bacterial identity audit without extracting or selecting images.

Run from the repository root:
    python Code/Training/Strain_Recognition_v1/audit_dataset.py
Requires Pillow and NumPy. No PyTorch, GPU, model loading, or predictions.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Code"))
from utils.identity_audit import (  # noqa: E402
    SCHEMA, IDENTITIES, LEGACY_CLASS_IDS, LEDGER_FIELDS, canonical_json, class_mapping,
    eligible_member, inspect_image, overlap_ledger, read_csv, sha256_file,
    summarize_groups, write_csv, write_json,
)

OUTPUT_NAMES = ("dataset_manifest.csv", "class_mapping.json", "acquisition_groups.csv",
                "duplicate_overlap_ledger.csv", "audit_report.md", "audit_status.json")


def relative(path, root):
    return Path(path).resolve().relative_to(root).as_posix()


def report(rows, legacy, mapping, edge_counts, ignored):
    internal = [r for r in rows if r["dataset"] == "Dataset_B"]
    external = [r for r in rows if r["dataset"] == "External_Test_Culture_Free"]
    lines = [
        "# Nine-class bacterial identity recognition — dataset audit", "",
        "Audit complete; split NOT_READY; biological provenance and acquisition confounding UNRESOLVED; "
        "external evaluation SEALED; model training NOT APPROVED.", "",
        "## Inventory", "",
        "Counts are primary acquisition/export image files, not independent biological samples. "
        "Preparation and biological labels are reported, not specimen-verified.", "",
        "| Class | Source label | Class name | Images | Recorded XY positions | Filename series | "
        "Candidate spatial components | Capture dates | Independent cultures | Preparation | Dimensions / format / mode |",
        "|---:|---|---|---:|---:|---:|---:|---|---|---|---|",
    ]
    for subset in ([r for r in internal if r["class_id"] == c] for c in range(9)):
        if subset:
            lines.append(inventory_line(subset))
    if external:
        lines.append(inventory_line(external))
    preview_count = sum(len(json.loads(r["secondary_pages_json"])) for r in rows)
    lines += ["", f"Dataset_B: **{len(internal):,}** images. Reserved external: **{len(external)}**. "
              f"Master manifest: **{len(rows):,}** rows.", "",
              f"Excluded non-primary archive entries: `{canonical_json(ignored)}`. "
              f"Embedded secondary TIFF pages: **{preview_count}**, retained as source metadata, not sample rows.", "",
              "No separately named crop derivatives were identified. Grid acquisitions are themselves "
              "overlapping field images; this does not establish that every export is an untouched camera raw. "
              "Strain_001 includes 110 stored 4080×3060 images whose acquisition XML reports original "
              "1920×1440 dimensions. Preserve both representations rather than interpreting export size as optical resolution.", "",
              "Capture dates are decoded from the embedded .NET DateTime bits; the original value is retained "
              "and its timezone is unverified. ZIP timestamps and filename date tokens are not acquisition IDs. "
              "The external filenames start with 20260721, but their decoded captures are dated 2026-09-02.", "",
              "## Biological identity and grouping", "",
              "These are heterogeneous identity classes: strain variants, pathotypes, species, and a reported "
              "resistance phenotype. Unknown biological fields are null in JSON and empty in CSV. MRSA's species "
              "is unset pending specimen confirmation; the expected organism to verify is Staphylococcus aureus. "
              "Original folder labels, including contradictory spellings, are retained in class_mapping.json.", "",
              "No culture, slide, or independently verified acquisition/session mapping was supplied. "
              "Verified independent units are therefore zero established, not evidence that zero cultures existed. "
              "Distinct stage coordinates and disconnected components are not biological replicates.", "",
              "Stage-overlap components use field centers and recorded field dimensions, conservatively assuming "
              "a common slide coordinate frame within each source label and decoded date. They include legacy "
              "training images as possible bridges. They are candidate groups, not confirmed independent units. "
              "Different labels are not joined solely by stage coordinates because slide identity is unknown.", "",
              "Examined grids use steps approximately 70% of field width/height, hence approximately 30% adjacent "
              "overlap. Several different filename series revisit almost the same footprint. Suffixes such as "
              "'(2)' do not reliably identify scan boundaries. Grouping by filename alone is insufficient.", "",
              "## Acquisition confounding — formal split blocker", "",
              "All inspected acquisition records name the same 100× oil objective and report BrightField; this "
              "does not establish matched acquisition conditions or verify the specimen preparation protocol. "
              "Binning, exposure, illumination, channel, pixel mode, contrast processing, export format, dimensions, "
              "and date vary with class. Instrument identity itself is not established by a shared objective string.", "",
              "A minimum of three verified independent biological/acquisition units per class is necessary but "
              "insufficient. Even many independent cultures acquired under class-specific conditions do not "
              "establish identity generalization. Matched or crossed acquisition-design evidence and an explicit "
              "confounding assessment are required. Resizing or converting files cannot resolve this gate.", "",
              "The preflight requires documented biological evidence and an explicit pipeline initialization history. "
              "Each prespecified acquisition-condition stratum must support all nine classes in each partition "
              "where it is represented. Strata may combine settings only where the scientific design supports "
              "comparability; relabeling class-specific regimes as one stratum is not resolution.", "",
              "The condition signatures below include acquisition and export settings; session candidates are "
              "source-label/date buckets, not verified sessions. Full metadata and signature definitions are in the manifest.", "",
              "### Class × acquisition condition", "",
              "| Condition signature | Dataset | Class counts |", "|---|---|---|"]
    by_condition = defaultdict(Counter)
    conditions = {}
    for r in rows:
        by_condition[(r["condition_signature"], r["dataset"])][str(r["class_id"])] += 1
        conditions[r["condition_signature"]] = json.loads(r["acquisition_conditions_json"])
    for (signature, dataset), counts in sorted(by_condition.items()):
        lines.append(f"| {signature} | {dataset} | {canonical_json(dict(counts))} |")
    lines += ["", "### Acquisition condition definitions", ""]
    for signature, definition in sorted(conditions.items()):
        lines += [f"- `{signature}`: `{canonical_json(definition)}`"]
    lines += ["", "### Class × candidate session", "",
              "| Candidate session | Dataset | Class | Decoded date | Images |", "|---|---|---:|---|---:|"]
    sessions = Counter((r["candidate_session_id"], r["dataset"], r["class_id"], r["acquisition_date"]) for r in rows)
    for (session, dataset, class_id, date), count in sorted(sessions.items()):
        lines.append(f"| {session} | {dataset} | {class_id} | {date} | {count} |")
    matches = Counter(r["class_id"] for r in internal if r["legacy_exact_match_count"])
    possible = sum(bool(r["legacy_overlap_candidate_count"]) for r in internal)
    legacy_hash_counts = Counter(r["sha256"] for r in legacy)
    within_legacy = sum(n * (n - 1) // 2 for n in legacy_hash_counts.values())
    incoming_legacy = sum(r["legacy_exact_match_count"] for r in rows)
    within_incoming = edge_counts.get("exact_sha256", 0) - within_legacy - incoming_legacy
    lines += ["", "## Duplicate / overlap evidence and legacy exposure", "",
              f"**{sum(matches.values())} Dataset_B images** match the documented **{len(legacy)}-image** legacy "
              f"training corpus by SHA-256. Per-class matches: `{canonical_json(dict(matches))}`. "
              "All are retained and tagged; none were deleted or excluded based on model behavior.", "",
              f"Ledger relationship counts: `{canonical_json(edge_counts)}`. Dataset_B images with direct "
              f"candidate legacy spatial overlap: **{possible}** (includes byte-identical matches).", "",
              f"Exact-match edge breakdown: **{incoming_legacy}** incoming-to-legacy, **{within_legacy}** "
              f"within legacy, **{within_incoming}** within incoming datasets. "
              "Edge counts are not counts of additional incoming images.", "",
              "Ledger endpoints retain IDs and paths even when an endpoint is in the legacy corpus and outside "
              "the 8,655-row master manifest. Exact hashes are compared globally. Non-identical files with repeated "
              "capture metadata or overlapping stage footprints are included as candidates. Candidate overlap "
              "fractions are geometric, not proof of pixel identity or specimen continuity. Image-content "
              "near-duplicate/crop/compression similarity detection is NOT performed; no detected match is NOT proof of unseen data.", "",
              "Any future old-checkpoint inference is **Legacy five-class model diagnostic**, never an unseen "
              "Dataset_B baseline. A 9×5 prediction table is diagnostic only and cannot establish nine-class "
              "generalization. Its results must not influence image selection or test assignment. If v1 reuses "
              "legacy weights, all units linked to legacy training exposure must be excluded from clean validation/test claims.", "",
              "## Readiness and external reservation", "",
              "Every Dataset_B row remains split=pending. Every external row remains split=reserved_external, "
              "evaluation_status=sealed. This audit reads metadata and validates pixels; it generates no predictions.", "",
              "Training requires verified biological/slide/session relationships, ≥3 independent units per class "
              "after relationship merging, all nine classes in train/validation/test, no cross-split related units, "
              "and resolved acquisition confounding. AIEC/MRSA spatial components may still belong to one culture; "
              "component counts cannot clear these requirements.", "",
              "Future v1 consumers must call utils.identity_audit_gate.assert_training_ready before model creation "
              "or image loading, and assert_external_evaluation_ready before generating or consuming external predictions. "
              "Existing legacy raw-classifier entrypoints explicitly reject v1 manifests, even if re-frozen, because "
              "their automatic cross-validation does not consume the frozen v1 split.", "",
              "External release requires a model-freeze record binding the checkpoint, inference/preprocessing "
              "configuration, class mapping, and complete internal split snapshot. Only then may rows become "
              "external_test/ready, followed by evaluated after execution. The freeze record must reference the "
              "sealed audit snapshot; changing internal membership or configuration invalidates release.", "",
              "Future internal evaluation: overall accuracy, balanced accuracy, macro-F1, per-class precision/recall, "
              "and 9×9 confusion matrix. External evaluation: class-006 recognition and confidence changes, not "
              "external nine-class accuracy or a preparation-only causal effect.", "",
              "## Reproduction", "", "```bash",
              "python Code/Training/Strain_Recognition_v1/audit_dataset.py",
              "python -m unittest discover -s tests -p 'test_identity_audit*.py'",
              "python Code/utils/identity_audit_gate.py training experiments/strain_recognition_v1",
              "```", "",
              "The last command is expected to fail while NOT_READY. To reproduce an existing unapproved, sealed "
              "audit, add --refresh to the audit command. Refresh refuses a READY or released audit. "
              "Audit output is deterministic for unchanged inputs and code. audit_status.json records input, "
              "code, and artifact SHA-256 hashes. These are integrity checks, not a security boundary against "
              "deliberately bypassing the workflow.", ""]
    return "\n".join(lines)


def inventory_line(rows):
    first = rows[0]
    specs = Counter(f"{r['width']}×{r['height']} {r['format']} {r['stored_pixel_mode']}" for r in rows)
    spec_text = "; ".join(f"{k}: {v}" for k, v in sorted(specs.items()))
    label = first["source_label_id"] + (" (external)" if first["dataset"] != "Dataset_B" else "")
    return (f"| {first['class_id']} | {label} | {first['class_name']} | {len(rows)} | "
            f"{len({(r['stage_x'], r['stage_y']) for r in rows})} | "
            f"{len({r['filename_series'] for r in rows})} | "
            f"{len({r['candidate_spatial_group_id'] for r in rows})} | "
            f"{', '.join(sorted({r['acquisition_date'] for r in rows}))} | Unknown | "
            f"{first['preparation']} | {spec_text} |")


def build(args):
    root = args.repo_root.resolve()
    out = (root / args.out_dir).resolve()
    if out.exists() and any((out / name).exists() for name in OUTPUT_NAMES):
        if not args.refresh:
            raise ValueError("Audit artifacts already exist; use --refresh for a sealed, unapproved audit")
        previous = json.loads((out / "audit_status.json").read_text())
        if previous.get("schema_version") != SCHEMA:
            raise ValueError("Refusing to replace reviewed provenance; use provenance_workflow.py or a new audit directory")
        if (previous.get("SPLIT_STATUS") != "NOT_READY" or previous.get("EXTERNAL_TEST") != "SEALED"
                or previous.get("MODEL_TRAINING_APPROVED") is not False):
            raise ValueError("Refusing to refresh a ready, approved, or released audit")
    inputs = [
        ("dataset_b_archive", root / args.dataset_b), ("external_archive", root / args.external),
        ("legacy_manifest", root / args.legacy_manifest),
        ("legacy_checkpoint", root / args.legacy_checkpoint),
        ("legacy_training_args", root / args.legacy_training_args),
        ("audit_code", Path(__file__)), ("audit_library", root / "Code/utils/identity_audit.py"),
        ("gate_code", root / "Code/utils/identity_audit_gate.py"),
        ("utility_package", root / "Code/utils/__init__.py"),
        ("legacy_training_preflight", root / "Code/Training/Model_Training_5_class/train_0min_fiveclass_raw.py"),
        ("legacy_selection_preflight", root / "Code/Training/Model_Training_5_class/run_0min_fiveclass_benchmark.py"),
    ]
    input_files = [{"role": role, "path": relative(path, root), "sha256": sha256_file(path)}
                   for role, path in inputs]
    snapshots = {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for _, path in inputs}
    rows, ignored = [], Counter()
    mapping = class_mapping()
    for dataset, archive in (("Dataset_B", root / args.dataset_b),
                             ("External_Test_Culture_Free", root / args.external)):
        archive_rel = relative(archive, root)
        with zipfile.ZipFile(archive) as z:
            names = [i.filename for i in z.infolist()]
            if len(names) != len(set(names)):
                raise ValueError(f"Ambiguous duplicate ZIP member paths: {archive}")
            for info in sorted(z.infolist(), key=lambda i: i.filename):
                if not eligible_member(info):
                    ignored["directories" if info.is_dir() else "non_image_or_AppleDouble"] += 1
                    continue
                parts = PurePosixPath(info.filename).parts
                folder = next((p for p in parts if p.startswith("Strain_")), None)
                if folder is None:
                    raise ValueError(f"Missing source class: {info.filename}")
                class_id = int(folder.split("_")[1]) - 1
                if not 0 <= class_id < 9 or (dataset != "Dataset_B" and class_id != 5):
                    raise ValueError(f"Unexpected source class: {info.filename}")
                labels = mapping["classes"][class_id]["original_folder_labels"]
                if folder not in labels:
                    labels.append(folder)
                image_path = f"zip://{archive_rel}!{info.filename}"
                try:
                    data = z.read(info)
                except (zipfile.BadZipFile, OSError, RuntimeError) as exc:
                    row = inspect_image(b"", image_path, dataset, class_id, archive_rel, info.filename)
                    row.update(sha256=None, size_bytes=info.file_size, primary_integrity_status="UNRESOLVED",
                               inventory_error="Archive member read failed: " + str(exc))
                else:
                    row = inspect_image(data, image_path, dataset, class_id, archive_rel, info.filename)
                rows.append(row)
                if len(rows) % 500 == 0:
                    print(f"Validated {len(rows)} primary images", flush=True)
    rows.sort(key=lambda r: (r["dataset"], r["class_id"], r["image_path"]))
    expected = dict(enumerate([367, 174, 324, 539, 686, 1050, 1460, 1400, 2621]))
    actual = dict(Counter(r["class_id"] for r in rows if r["dataset"] == "Dataset_B"))
    if actual != expected or sum(r["dataset"] != "Dataset_B" for r in rows) != 34:
        raise ValueError(f"Dataset inventory changed; review expected population before updating: {actual}")
    if any(not p["is_preview"] for r in rows for p in json.loads(r["secondary_pages_json"])):
        raise ValueError("Unrecognized secondary TIFF image pages require review")
    legacy_args = json.loads((root / args.legacy_training_args).read_text())
    legacy_rows = read_csv(root / args.legacy_manifest)
    if legacy_args.get("filter_auto_dropped"):
        legacy_rows = [r for r in legacy_rows if not int(r.get("drop_from_training") or 0)]
    if len(legacy_rows) != legacy_args["n_samples"]:
        raise ValueError("Legacy training corpus does not match checkpoint documentation")
    legacy = []
    for source in legacy_rows:
        path = Path(source["image_path"])
        if not path.is_absolute():
            path = root / path
        class_id = LEGACY_CLASS_IDS[source["class_name"]]
        legacy.append(inspect_image(path.read_bytes(), relative(path, root), "Legacy_training", class_id))
    if dict(Counter(r["class_name"] for r in legacy_rows)) != legacy_args["class_counts"]:
        raise ValueError("Legacy per-class population differs from training_args.json")
    # Record every legacy byte source so future guards also detect stale overlap evidence.
    input_files += [{"role": "legacy_image", "path": r["image_path"], "sha256": r["sha256"]}
                    for r in legacy]
    print(f"Validated {len(rows)} incoming images and {len(legacy)} legacy training images; building ledger", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="identity-audit-", dir=out.parent) as temp:
        stage = Path(temp)
        records = rows + legacy
        with (stage / "duplicate_overlap_ledger.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS)
            writer.writeheader()
            edges = overlap_ledger(records, writer.writerow)
        write_csv(stage / "dataset_manifest.csv", rows)
        write_json(stage / "class_mapping.json", mapping)
        write_csv(stage / "acquisition_groups.csv", summarize_groups(records))
        (stage / "audit_report.md").write_text(report(rows, legacy, mapping, edges, dict(ignored)), encoding="utf-8")
        blockers = [
            {"rule": "BIOLOGICAL_PROVENANCE", "classes": list(range(9)), "reason": "Culture/slide/session independence unverified"},
            {"rule": "ACQUISITION_CONFOUNDING", "classes": list(range(9)), "reason": "Class-associated acquisition/export settings; no matched/crossed design evidence"},
            {"rule": "MINIMUM_INDEPENDENT_UNITS", "classes": list(range(9)), "reason": "At least 3 verified units per class required after relationship merging; currently none established"},
            {"rule": "OVERLAP_PROVENANCE", "reason": "Candidate coordinate relationships require slide/coordinate-frame confirmation; content similarity not performed"},
            {"rule": "IDENTITY_VERIFICATION", "classes": [6], "reason": "MRSA organism identity requires specimen confirmation"},
            {"rule": "SPLIT_ASSIGNMENT", "reason": "Dataset_B remains pending; no frozen internal partition"},
        ]
        status = {
            "schema_version": SCHEMA, "AUDIT_STATUS": "COMPLETE", "SPLIT_STATUS": "NOT_READY",
            "BIOLOGICAL_PROVENANCE": "UNRESOLVED", "ACQUISITION_CONFOUNDING": "UNRESOLVED",
            "EXTERNAL_TEST": "SEALED", "MODEL_TRAINING_APPROVED": False,
            "blockers": blockers, "input_files": input_files,
            "artifact_sha256": {name: sha256_file(stage / name) for name in OUTPUT_NAMES if name != "audit_status.json"},
            "image_counts": {"Dataset_B": actual, "External_Test_Culture_Free": {5: 34}},
            "manifest_rows": len(rows), "legacy_training_rows": len(legacy),
            "legacy_exact_match_images": sum(bool(r["legacy_exact_match_count"]) for r in rows),
            "ledger_relationship_counts": edges, "content_similarity_status": "NOT_PERFORMED",
            "acquisition_design_assessment": {"status": "UNRESOLVED", "design": None,
                                               "evidence_files": [], "condition_strata": {}},
            "biological_provenance_evidence": [],
            "initialization_history": {"status": "UNRESOLVED", "uses_legacy_checkpoint": None,
                                       "evidence_files": []},
            "reproduction_command": "python Code/Training/Strain_Recognition_v1/audit_dataset.py",
        }
        write_json(stage / "audit_status.json", status)
        for _, path in inputs:
            if (path.stat().st_size, path.stat().st_mtime_ns) != snapshots[str(path)]:
                raise ValueError(f"Input changed while audit was running: {path}")
        out.mkdir(parents=True, exist_ok=True)
        # Publish status last. Partial publication cannot pass the hash/readiness gate.
        for name in OUTPUT_NAMES:
            os.replace(stage / name, out / name)
    print(f"Wrote six artifacts to {out}; SPLIT_STATUS=NOT_READY; EXTERNAL_TEST=SEALED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/strain_recognition_v1"))
    parser.add_argument("--dataset-b", type=Path, default=Path("Data/Dataset_B.zip"))
    parser.add_argument("--external", type=Path, default=Path("Data/Dataset_Culture_Free_test.zip"))
    parser.add_argument("--legacy-manifest", type=Path, default=Path("Data/Benchmark_0min_5class/segmentation_qc/qc_manifest/manifest_with_segmentation_qc.csv"))
    parser.add_argument("--legacy-checkpoint", type=Path, default=Path("Model/Benchmark_0min_5class_QC/final_model/final_model.pt"))
    parser.add_argument("--legacy-training-args", type=Path, default=Path("Model/Benchmark_0min_5class_QC/final_model/training_args.json"))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    try:
        build(args)
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as exc:
        raise SystemExit(f"AUDIT FAILED (no new COMPLETE status published): {exc}") from exc


if __name__ == "__main__":
    main()
