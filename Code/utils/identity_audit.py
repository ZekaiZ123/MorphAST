"""Archive inventory and conservative overlap evidence for bacterial identity v1.

Coordinates and filenames provide candidate relationships, never culture independence.
Only primary image pages are samples; TIFF previews stay attached to their source.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image


SCHEMA = "bacterial_identity_v1"
ECOLI = "Escherichia coli"
IDENTITIES = [
    ("E. coli 5-Alpha", ECOLI, "5-Alpha", None, None),
    ("E. coli 5-Alpha AmpR", ECOLI, "5-Alpha", None, "AmpR"),
    ("E. coli 5-Alpha KanR", ECOLI, "5-Alpha", None, "KanR"),
    ("UPEC", ECOLI, None, "UPEC", None),
    ("AIEC", ECOLI, None, "AIEC", None),
    ("E. coli 25922", ECOLI, "25922", None, None),
    ("MRSA", None, None, None, "methicillin-resistant (reported)"),
    ("S. pneumoniae", "Streptococcus pneumoniae", None, None, None),
    ("S. aureus", "Staphylococcus aureus", None, None, None),
]
IDENTITY_FIELDS = ("class_name", "species", "strain", "pathotype", "resistance_phenotype")
LEGACY_CLASS_IDS = {"E.coli5-alpha": 0, "E.coli5-alphaAmpR": 1,
                    "E.coli5-alphaKanR": 2, "UPEC": 3, "AIEC": 4}
CONDITION_KEYS = [
    "Lens/LensName", "Lens/PhaseContrast", "Shooting/Observation", "Shooting/Channel",
    "Image/DigitalZoom", "Image/PowerZoom", "Image/Calibration",
    "Shooting/Parameter/Binnin", "Shooting/Parameter/PixelMode",
    "Shooting/Parameter/ExposureTime/Numerator", "Shooting/Parameter/ExposureTime/Denominator",
    "Shooting/Parameter/CameraGain", "Shooting/Parameter/CameraGainUnit",
    "Shooting/Parameter/CameraHardwareGain", "Shooting/Parameter/Aperture",
    "Shooting/Parameter/TransIlluminationRate", "Shooting/Parameter/NeutralDensityFilter",
    "ImageCorrection/LookupTable/OperationType",
    "ImageCorrection/LookupTable/DistributionParameter0/Shadow",
    "ImageCorrection/LookupTable/DistributionParameter0/Highlight",
    "ImageCorrection/LookupTable/DistributionParameter0/Gamma",
    "ImageCorrection/NoiseFilterEnabled", "ImageCorrection/RealtimeFilter",
    "ImageCorrection/HazeReduction", "ImageCorrection/WhiteBalance/OperationType",
]


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def stable_id(prefix, value):
    return prefix + "_" + hashlib.sha256(canonical_json(value).encode()).hexdigest()[:24]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def class_mapping():
    classes = []
    for class_id, values in enumerate(IDENTITIES):
        classes.append({"class_id": class_id, "source_label_id": f"Strain_{class_id+1:03d}",
                        **dict(zip(IDENTITY_FIELDS, values)),
                        "identity_status": "reported_unverified",
                        "identity_evidence": "User-provided class labels; specimen records not supplied",
                        "organism_verification_required": class_id == 6,
                        "expected_species_to_verify": "Staphylococcus aureus" if class_id == 6 else None,
                        "original_folder_labels": []})
    return {"schema_version": SCHEMA, "project": "Nine-class bacterial identity recognition",
            "classes": classes}


def eligible_member(info):
    p = PurePosixPath(info.filename)
    if info.is_dir() or "__MACOSX" in p.parts or p.name.startswith("._"):
        return False
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"Unsafe archive member: {p}")
    return p.suffix.lower() in {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"}


def parse_metadata(data):
    start = data.find(b"<SingleFileProperty")
    end = data.find(b"</SingleFileProperty>", start)
    if start < 0 or end < 0:
        raise ValueError("Embedded Keyence acquisition XML missing")
    root = ET.fromstring(data[start:end + len(b"</SingleFileProperty>")])
    out = {}

    def visit(node, path):
        if not len(node):
            out[path] = (node.text or "").strip()
        for child in node:
            visit(child, f"{path}/{child.tag}" if path else child.tag)

    visit(root, "")
    return out


def filename_series(name):
    stem = PurePosixPath(name).stem
    if "_CH" in stem:
        return re.sub(r"_\d+(?=_CH\d)", "_FIELD", stem)
    return re.sub(r"_\d+(?= \(\d+\)$|$)", "", stem)


def _inspect_image(data, image_path, dataset, class_id, archive_path="", member=""):
    meta = parse_metadata(data)
    with Image.open(io.BytesIO(data)) as im:
        width, height = im.size
        mode, fmt = im.mode, im.format
        im.load()  # Validate the primary pixels, not only the header.
        secondary = []
        for index in range(1, getattr(im, "n_frames", 1)):
            im.seek(index)
            secondary.append({"page": index, "width": im.width, "height": im.height,
                              "mode": im.mode, "is_preview": im.size == (160, 120)})
            im.load()
    stamp = meta["Image/ShootingDateTime"]
    # Preserve raw .NET DateTime bits. The decoded clock is not a verified timezone.
    ticks = int(stamp) & ((1 << 62) - 1)
    captured = dt.datetime(1, 1, 1) + dt.timedelta(microseconds=ticks // 10)
    xy = [int(meta["Shooting/StageLocation" + a]) for a in ("X", "Y")]
    footprint = [int(meta["Shooting/XyStageRegion/" + a]) for a in ("Width", "Height")]
    if min(footprint) <= 0:
        raise ValueError("Invalid field footprint")
    conditions = {k: meta.get(k) for k in CONDITION_KEYS}
    conditions.update(format=fmt, width=width, height=height, stored_pixel_mode=mode)
    image_id = stable_id("image", [dataset, image_path])
    external = dataset == "External_Test_Culture_Free"
    legacy = dataset == "Legacy_training"
    return {
        "schema_version": SCHEMA, "image_id": image_id, "image_path": image_path,
        "archive_path": archive_path, "archive_member": member,
        "source_label_id": f"Strain_{class_id+1:03d}", "class_id": class_id,
        **dict(zip(IDENTITY_FIELDS, IDENTITIES[class_id])),
        "identity_status": "reported_unverified", "dataset": dataset,
        "preparation": "culture_free" if external else "standard",
        "preparation_status": "reported_unverified", "replicate_id": None,
        "culture_id": None, "slide_id": None, "acquisition_session_id": None,
        "independent_unit_id": None, "biological_provenance_status": "unresolved",
        "field_id": None, "field_status": "unverified_slide_identity",
        "candidate_field_id": stable_id("field", [class_id, captured.date().isoformat(), xy]),
        "source_image_id": image_id, "source_image_status": "primary_acquisition_export",
        "split": "legacy_training" if legacy else "reserved_external" if external else "pending",
        "evaluation_status": "sealed" if external else "not_evaluated",
        "format": fmt, "width": width, "height": height, "stored_pixel_mode": mode,
        "primary_page_index": 0, "secondary_pages_json": canonical_json(secondary),
        "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data),
        "capture_time_raw": stamp, "capture_time_decoded": captured.isoformat(),
        "capture_timezone_status": "unverified", "acquisition_date": captured.date().isoformat(),
        "stage_x": xy[0], "stage_y": xy[1], "stage_z": meta["Shooting/StageLocationZ"],
        "field_width_stage_units": footprint[0], "field_height_stage_units": footprint[1],
        "channel": meta.get("Shooting/Channel"),
        "candidate_session_id": stable_id("session", [class_id, captured.date().isoformat()]),
        "filename_series": filename_series(member or image_path),
        "condition_signature": stable_id("condition", conditions),
        "acquisition_conditions_json": canonical_json(conditions),
        "acquisition_metadata_json": canonical_json(meta),
        "candidate_spatial_group_id": None, "grouping_status": "candidate_unverified",
        "legacy_exact_match_count": 0, "legacy_overlap_candidate_count": 0,
        "legacy_exposure_status": "not_assessed",
    }


def inspect_image(data, image_path, dataset, class_id, archive_path="", member=""):
    """Keep identifiable inventory rows even when decoding/metadata extraction fails.

    Integrity outcomes are observations, not analysis exclusions: only a frozen QC
    policy may turn a technical failure into QC_EXCLUDED.
    """
    try:
        row = _inspect_image(data, image_path, dataset, class_id, archive_path, member)
        row.update(primary_integrity_status="PASS", metadata_status="OBSERVED", inventory_error="")
        return row
    except (OSError, ValueError, KeyError, ET.ParseError) as exc:
        status, width, height, fmt, mode = "FAIL", "", "", "", ""
        try:
            with Image.open(io.BytesIO(data)) as im:
                width, height, fmt, mode = im.width, im.height, im.format, im.mode
                im.load()
                status = "PASS"
        except (OSError, ValueError):
            pass
        image_id = stable_id("image", [dataset, image_path])
        return {"schema_version": SCHEMA, "image_id": image_id, "image_path": image_path,
            "archive_path": archive_path, "archive_member": member, "source_label_id": f"Strain_{class_id+1:03d}",
            "class_id": class_id, **dict(zip(IDENTITY_FIELDS, IDENTITIES[class_id])),
            "identity_status": "reported_unverified", "dataset": dataset,
            "preparation": "culture_free" if dataset == "External_Test_Culture_Free" else "standard",
            "preparation_status": "reported_unverified", "replicate_id": None, "culture_id": None,
            "slide_id": None, "acquisition_session_id": None, "independent_unit_id": None,
            "biological_provenance_status": "unresolved", "field_id": None, "field_status": "unverified_slide_identity",
            "candidate_field_id": "", "source_image_id": image_id, "source_image_status": "unresolved",
            "split": "reserved_external" if dataset == "External_Test_Culture_Free" else
                "legacy_training" if dataset == "Legacy_training" else "pending",
            "evaluation_status": "sealed" if dataset == "External_Test_Culture_Free" else "not_evaluated",
            "format": fmt, "width": width, "height": height, "stored_pixel_mode": mode,
            "primary_page_index": 0, "secondary_pages_json": "[]", "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data), "capture_time_raw": "", "capture_time_decoded": "",
            "capture_timezone_status": "unverified", "acquisition_date": "", "stage_x": "", "stage_y": "",
            "stage_z": "", "field_width_stage_units": "", "field_height_stage_units": "", "channel": "",
            "candidate_session_id": "", "filename_series": filename_series(member or image_path),
            "condition_signature": "", "acquisition_conditions_json": "{}", "acquisition_metadata_json": "{}",
            "candidate_spatial_group_id": None, "grouping_status": "candidate_unverified",
            "legacy_exact_match_count": 0, "legacy_overlap_candidate_count": 0, "legacy_exposure_status": "not_assessed",
            "primary_integrity_status": status, "metadata_status": "UNRESOLVED", "inventory_error": type(exc).__name__+": "+str(exc)}


class UnionFind:
    def __init__(self, n):
        self.parents = list(range(n))

    def find(self, a):
        while self.parents[a] != a:
            self.parents[a] = self.parents[self.parents[a]]
            a = self.parents[a]
        return a

    def union(self, a, b):
        self.parents[self.find(b)] = self.find(a)


LEDGER_FIELDS = ["image_id_a", "image_id_b", "image_path_a", "image_path_b", "dataset_a",
                 "dataset_b", "relationship", "evidence_status", "overlap_fraction_a",
                 "overlap_fraction_b", "same_capture_metadata", "assumption", "resolution",
                 "resolution_evidence"]
SPATIAL_ASSUMPTION = (
    "Same source label and decoded acquisition date provisionally share a slide coordinate frame; "
    "slide identity, timezone, registration, and specimen continuity remain unverified."
)


def overlap_ledger(records, emit):
    """Emit evidence edges; union conservatively, including legacy bridges.

    Exact hashes are compared globally. Spatial comparisons use label/date buckets,
    not filename series; cross-label geometry alone cannot identify a common slide.
    """
    uf = UnionFind(len(records))
    counts = Counter()

    def edge(a, b, kind, fa="", fb=""):
        left, right = records[a], records[b]
        exact = kind == "exact_sha256"
        if exact and left["dataset"] != "External_Test_Culture_Free" and right["dataset"] != "External_Test_Culture_Free":
            uf.union(a, b)
        same_capture = all(left[k] == right[k] for k in
                           ("capture_time_raw", "stage_x", "stage_y", "channel"))
        row = dict(zip(LEDGER_FIELDS, [
            left["image_id"], right["image_id"], left["image_path"], right["image_path"],
            left["dataset"], right["dataset"], kind, "confirmed" if exact else "candidate",
            fa, fb, str(same_capture).lower(), "Byte-identical primary image file" if exact else SPATIAL_ASSUMPTION,
            "must_coassign" if exact else "unresolved", "",
        ]))
        emit(row)
        counts[kind] += 1
        for current, other in ((left, right), (right, left)):
            if current["dataset"] != "Legacy_training" and other["dataset"] == "Legacy_training":
                current["legacy_exact_match_count" if exact else "legacy_overlap_candidate_count"] += 1

    hashes, spatial = defaultdict(list), defaultdict(list)
    for i, row in enumerate(records):
        if row.get("sha256"):
            hashes[row["sha256"]].append(i)
        if all(row.get(k) not in (None, "") for k in
               ("stage_x", "stage_y", "field_width_stage_units", "field_height_stage_units", "acquisition_date")):
            spatial[(row["class_id"], row["acquisition_date"])].append(i)
    for indices in hashes.values():
        for pos, a in enumerate(indices):
            for b in indices[pos + 1:]:
                edge(a, b, "exact_sha256")
    for indices in spatial.values():
        centers = np.array([[records[i]["stage_x"], records[i]["stage_y"]] for i in indices], float)
        sizes = np.array([[records[i]["field_width_stage_units"], records[i]["field_height_stage_units"]]
                          for i in indices], float)
        for pos, a in enumerate(indices):
            intersection = (np.minimum(centers[pos] + sizes[pos] / 2, centers[pos+1:] + sizes[pos+1:] / 2)
                            - np.maximum(centers[pos] - sizes[pos] / 2, centers[pos+1:] - sizes[pos+1:] / 2))
            for offset in np.flatnonzero((intersection[:, 0] > 0) & (intersection[:, 1] > 0)):
                bpos = pos + 1 + int(offset)
                area = float(np.prod(intersection[offset]))
                edge(a, indices[bpos], "stage_footprint_overlap",
                     round(area / np.prod(sizes[pos]), 8), round(area / np.prod(sizes[bpos]), 8))
    components = defaultdict(list)
    for i in range(len(records)):
        components[uf.find(i)].append(i)
    for indices in components.values():
        group_id = stable_id("spatial", sorted(records[i]["image_id"] for i in indices))
        for i in indices:
            row = records[i]
            row["candidate_spatial_group_id"] = group_id
            row["legacy_exposure_status"] = (
                "documented_legacy_training" if row["dataset"] == "Legacy_training" else
                "exact_legacy_training_match" if row["legacy_exact_match_count"] else
                "possible_legacy_training_overlap" if row["legacy_overlap_candidate_count"] else
                "no_detected_match_not_proof_of_unseen"
            )
    return dict(counts)


def summarize_groups(records):
    buckets = defaultdict(list)
    for row in records:
        buckets[row["candidate_spatial_group_id"]].append(row)
    result = []
    for group_id, rows in sorted(buckets.items()):
        result.append({
            "candidate_spatial_group_id": group_id, "grouping_status": "candidate_unverified",
            "culture_id": None, "slide_id": None, "acquisition_session_id": None,
            "independent_unit_id": None, "independence_status": "unresolved",
            "verified_independent_units": 0, "assumption": SPATIAL_ASSUMPTION,
            "class_ids_json": canonical_json(sorted({r["class_id"] for r in rows})),
            "dataset_counts_json": canonical_json(dict(Counter(r["dataset"] for r in rows))),
            "candidate_session_ids_json": canonical_json(sorted({r["candidate_session_id"] for r in rows})),
            "condition_counts_json": canonical_json(dict(Counter(r["condition_signature"] for r in rows))),
            "acquisition_dates_json": canonical_json(sorted({r["acquisition_date"] for r in rows})),
            "filename_series_json": canonical_json(sorted({r["filename_series"] for r in rows})),
            "image_ids_json": canonical_json(sorted(r["image_id"] for r in rows)),
            "capture_start": min(r["capture_time_decoded"] for r in rows),
            "capture_end": max(r["capture_time_decoded"] for r in rows),
        })
    return result
