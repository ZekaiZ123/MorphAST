"""Versioned provenance packet/import/assessment orchestration.

Generated artifacts are reproducible evidence products, not hand-edited approvals.
Original inventory and returned response bytes are retained in immutable snapshots.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import os
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from .identity_provenance import (SCHEMA, POLICY, REVIEW_KEYS, ProvenanceError, check,
    canonical, content_hash, stable_id, initial_row, decoded, build_graph,
    prepare_problem, analytical_failures, validate_assignment, class_report)
from .identity_split_milp import DEFAULT_PROTOCOL, solve

ROOT = Path(__file__).resolve().parents[2]
BASE_FILES = ("dataset_manifest.csv", "class_mapping.json", "acquisition_groups.csv",
              "duplicate_overlap_ledger.csv", "audit_report.md", "audit_status.json")
OUTPUT_FILES = BASE_FILES[:-1] + ("provenance_context.json", "split_feasibility.json",
    "acquisition_condition_schema.json", "qc_policy.json", "eligible_population.json")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)+"\n")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    fields = fields or sorted({k for row in rows for k in row})
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: canonical(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def evidence_record(path, root):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(Path(root).resolve())), "sha256": digest(path)}


def verify_record(record, root):
    path = (Path(root) / record["path"]).resolve()
    path.relative_to(Path(root).resolve())
    check(path.is_file() and digest(path) == record["sha256"], "Missing/stale evidence: " + str(path))
    return path


def code_records(root):
    paths = [Path(__file__), Path(__file__).with_name("identity_provenance.py"),
             Path(__file__).with_name("identity_split_milp.py"),
             Path(__file__).with_name("identity_audit_gate.py"),
             Path(__file__).with_name("identity_provenance_gate.py"),
             ROOT/"Code/Training/Strain_Recognition_v1/provenance_workflow.py"]
    # Bind actual executing code, also when auditing a fixture/data root elsewhere.
    return [{"path": str(p.resolve().relative_to(Path(root).resolve())) if
             p.resolve().is_relative_to(Path(root).resolve()) else str(p.resolve()), "sha256": digest(p)} for p in paths]


def load(directory, root=ROOT, *, check_sources=False):
    directory = Path(directory)
    status = read_json(directory/"audit_status.json")
    check(status.get("schema_version") == SCHEMA, "Provenance v2 migration required")
    check(set(status["artifact_sha256"]) == set(OUTPUT_FILES), "Incomplete artifact bindings")
    for name, expected in status["artifact_sha256"].items():
        check(digest(directory/name) == expected, "Stale artifact: " + name)
    context = read_json(directory/"provenance_context.json")
    for record in context["evidence_registry"].values():
        verify_record(record, root)
    if check_sources:
        for record in context["internal_source_files"]:
            verify_record(record, root)
    rows = read_csv(directory/"dataset_manifest.csv")
    check(len(rows) == status["manifest_rows"], "Manifest population changed")
    check(len({r["image_id"] for r in rows}) == len(rows), "Duplicate image IDs")
    check(all(r["schema_version"] == SCHEMA for r in rows), "Mixed manifest schema")
    mapping = read_json(directory/"class_mapping.json")
    classes = {int(c["class_id"]): c for c in mapping["classes"]}
    check(set(classes) == set(range(9)) and len(mapping["classes"]) == 9, "Invalid identity mapping")
    for r in rows:
        cls = classes[int(r["class_id"])]
        check(all(r[k] == str(cls[k] or "") for k in
                  ("class_name", "source_label_id", "species", "strain", "pathotype", "resistance_phenotype")),
              "Identity mapping mismatch")
    return status, rows, mapping, context, read_csv(directory/"duplicate_overlap_ledger.csv")


def draft_policies(rows):
    factors = sorted({"conditions."+k for r in rows for k in decoded(r.get("acquisition_conditions_json"), {})} |
        {"verified_acquisition_date", "acquisition_session_id", "operator", "instrument_id", "export_pipeline"})
    schema = {"version": "acquisition-v1-draft", "status": "DRAFT", "evidence_ids": [],
        "required_sources": factors,
        "acquisition_design_policy": POLICY, "joint_batch_factors": [],
        "factors": [{"source": f, "role": "UNKNOWN_BLOCKING", "normalization": "exact", "justification": ""}
                    for f in factors]}
    qc = {"version": "qc-v1-draft", "status": "DRAFT", "evidence_ids": [],
          "rules": [{"id": "primary_readability", "kind": "integrity_decode", "on_failure": "QC_EXCLUDED"}]}
    return schema, qc


def prepare(directory, root=ROOT):
    directory, root = Path(directory).resolve(), Path(root).resolve()
    status = read_json(directory/"audit_status.json")
    if status.get("schema_version") == SCHEMA:
        load(directory, root)
        return {"already_prepared": True, "packet": str(directory/"provenance/review_packet_v1")}
    check(status.get("SPLIT_STATUS") == "NOT_READY" and status.get("EXTERNAL_TEST") == "SEALED",
          "Only an unfrozen sealed inventory can be migrated")
    for name, expected in status["artifact_sha256"].items():
        check(digest(directory/name) == expected, "Original audit artifact changed: " + name)
    provenance = directory/"provenance"
    provenance.mkdir(exist_ok=True)
    snapshot = provenance/"inventory_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in BASE_FILES:
        target = snapshot/name
        if target.exists():
            check(digest(target) == digest(directory/name), "Original snapshot differs; refusing overwrite")
        else:
            shutil.copy2(directory/name, target)
    rows = [initial_row(r) for r in read_csv(snapshot/"dataset_manifest.csv")]
    mapping = read_json(snapshot/"class_mapping.json")
    mapping["schema_version"] = SCHEMA
    schema, qc = draft_policies(rows)
    registry = {stable_id("inventory", name): evidence_record(snapshot/name, root) for name in BASE_FILES}
    context = {"schema_version": SCHEMA, "review_packet_version": "review-packet-v1",
        "inventory_snapshot": str(snapshot.relative_to(root)), "evidence_registry": registry,
        "responses": [], "unit_reviews": [], "schema": schema, "qc": qc,
        "solver_protocol": dict(DEFAULT_PROTOCOL), "code_files": code_records(root),
        "internal_source_files": [r for r in status["input_files"] if r["role"] == "dataset_b_archive"],
        "external_source_files": [r for r in status["input_files"] if r["role"] == "external_archive"],
        "legacy_source_files": [r for r in status["input_files"] if r["role"].startswith("legacy_") and
                                 "preflight" not in r["role"]],
        "qc_runs": [], "external_reviews": []}
    packet = provenance/"review_packet_v1"
    packet.mkdir(exist_ok=True)
    buckets = defaultdict(list)
    for r in rows:
        buckets[(r["dataset"], r["class_id"], r["filename_series"])].append(r)
    series, assignments = [], []
    for (dataset, cls, name), images in sorted(buckets.items()):
        sid = stable_id("review_series", [dataset, cls, name])
        series.append({"review_series_id": sid, "dataset": dataset, "class_id": cls, "filename_series": name,
            "image_count": len(images), "image_ids": sorted(r["image_id"] for r in images),
            "capture_start": min(r["capture_time_decoded"] for r in images),
            "capture_end": max(r["capture_time_decoded"] for r in images),
            "condition_signatures": sorted({r["condition_signature"] for r in images}),
            "warning": "Review bucket only; series/date/coordinates do not establish independence",
            **{k: "" for k in REVIEW_KEYS}, "verification_status": "UNRESOLVED"})
        for r in images:
            assignments.append({"image_id": r["image_id"], "image_path": r["image_path"], "review_series_id": sid,
                **{k: "" for k in REVIEW_KEYS}, "verification_status": "UNRESOLVED"})
    write_csv(packet/"series_review.csv", series)
    write_csv(packet/"image_assignments.csv", assignments)
    write_json(packet/"acquisition_condition_schema.json", schema)
    write_json(packet/"qc_policy.json", qc)
    write_json(packet/"review_response.template.json", {"review_packet_version": "review-packet-v1",
        "reviewer": "", "review_timestamp": "", "evidence_reference": "", "evidence_files": [],
        "supersedes": [], "series_assignments": [], "image_assignments": [], "unit_reviews": [],
        "edge_adjudications": [], "identity_verifications": [], "external_reviews": [],
        "acquisition_schema": None, "qc_policy": None})
    (packet/"REQUEST.md").write_text(
        "# Acquisition-team provenance request\n\n"
        "Return the original completed response and supporting records. Do not invent IDs from filenames, dates, or stage coordinates.\n\n"
        "The CSV files are review worksheets. Submit reviewed values using review_response.template.json; "
        "the workflow README documents the exact import contract. Series claims expand only to the packet's exact membership; "
        "use image claims instead when a series contains several preparations/sessions.\n\n"
        "Provide culture, sample/preparation, slide, session, date/timezone, instrument, operator, optics, settings, export pipeline, "
        "source series, and coordinate frame evidence. Verify MRSA organism identity. Review acquisition and QC policies before feasibility.\n\n"
        "42 MRSA filename buckets have interleaved capture times; they are not 42 verified sessions. "
        "External metadata date differs from its filename date. All external material remains sealed: "
        "no predictions, embeddings, morphology, feature distributions, or image-derived development analysis.\n")
    packet_files = sorted(p for p in packet.iterdir() if p.is_file())
    packet_record = {"version": "review-packet-v1", "files": [evidence_record(p, root) for p in packet_files]}
    write_json(packet/"packet.json", packet_record)
    context["packet_record"] = evidence_record(packet/"packet.json", root)
    context["series_membership"] = {s["review_series_id"]: s["image_ids"] for s in series}
    ledger = read_csv(snapshot/"duplicate_overlap_ledger.csv")
    result = assess_rows(rows, mapping, context, ledger, run_solver=False)
    publish(directory, root, rows, mapping, context, result)
    return {"packet": str(packet), "review_buckets": len(series), "state": result["state"]}


def external_only_response(response, rows, context, ledger):
    """Only externally scoped evidence can change after the internal split freezes."""
    external = {r["image_id"] for r in rows if r["dataset"] == "External_Test_Culture_Free"}
    for key in ("unit_reviews", "identity_verifications", "acquisition_schema", "qc_policy"):
        check(not response.get(key), "Frozen partition: internal/policy review changes require a new audit version")
    for claim in response.get("image_assignments", []):
        check(claim["image_id"] in external, "Frozen partition: internal claims cannot change")
    for claim in response.get("series_assignments", []):
        check(set(context["series_membership"][claim["review_series_id"]]) <= external, "Frozen internal series cannot change")
    for claim in response.get("external_reviews", []):
        check(set(claim["image_ids"]) <= external, "External review contains internal images")
    external_edges = {e["edge_id"] for e in ledger if e["image_id_a"] in external or e["image_id_b"] in external}
    for claim in response.get("edge_adjudications", []):
        check(set(claim.get("edge_ids", [claim.get("edge_id")])) <= external_edges, "Frozen internal relationship cannot change")
    for claim in response.get("additional_relationships", []):
        check(claim["image_id_a"] in external or claim["image_id_b"] in external, "New internal relationship requires a new audit version")


def import_response(directory, response_path, root=ROOT):
    directory, root = Path(directory).resolve(), Path(root).resolve()
    status, original_rows, _, context, original_ledger = load(directory, root)
    frozen = status["SPLIT_STATUS"] == "READY"
    response_path = Path(response_path).resolve()
    raw = response_path.read_bytes()
    response = json.loads(raw)
    if frozen:
        check(status["EXTERNAL_TEST"] == "SEALED", "Released external evidence cannot change")
        external_only_response(response, original_rows, context, original_ledger)
        for prior in context["responses"]:
            if prior["id"] in response.get("supersedes", []):
                external_only_response(read_json(verify_record(prior["record"], root)), original_rows, context, original_ledger)
        from .identity_provenance_gate import assert_split_ready
        assert_split_ready(directory, root)
    check(response.get("review_packet_version") == context["review_packet_version"], "Response packet version mismatch")
    for key in ("reviewer", "review_timestamp", "evidence_reference"):
        check(bool(response.get(key)), "Missing response " + key)
    stamp = dt.datetime.fromisoformat(response["review_timestamp"].replace("Z", "+00:00"))
    check(stamp.tzinfo is not None, "Reviewer timestamp must include timezone")
    response_hash = hashlib.sha256(raw).hexdigest()
    rid = "response_" + response_hash
    if any(r["id"] == rid for r in context["responses"]):
        return {"already_imported": rid}
    previous = {r["id"] for r in context["responses"]}
    check(set(response.get("supersedes", [])) <= previous, "Unknown superseded response")
    dest = directory/"provenance/responses"/response_hash
    dest.mkdir(parents=True, exist_ok=True)
    original = dest/"original_review_response.json"
    if original.exists():
        check(original.read_bytes() == raw, "Immutable response snapshot changed")
    else:
        original.write_bytes(raw)
    retained = []
    for record in response.get("evidence_files", []):
        source = verify_record(record, root)
        target = dest/(record["sha256"]+"_"+source.name)
        if target.exists():
            check(digest(target) == record["sha256"], "Immutable evidence snapshot changed")
        else:
            shutil.copy2(source, target)
        retained.append(evidence_record(target, root))
    receipt = {"review_packet_version": context["review_packet_version"], "reviewer": response["reviewer"],
        "review_timestamp": response["review_timestamp"], "evidence_reference": response["evidence_reference"],
        "original_review_response": evidence_record(original, root), "response_hash": response_hash,
        "import_code_version": content_hash(code_records(root)), "import_timestamp": now(), "retained_evidence": retained}
    write_json(dest/"receipt.json", receipt)
    context["evidence_registry"][rid] = evidence_record(original, root)
    context["evidence_registry"][rid+"_receipt"] = evidence_record(dest/"receipt.json", root)
    for i, record in enumerate(retained):
        context["evidence_registry"][rid+"_evidence_"+str(i)] = record
    context["responses"].append({"id": rid, "record": evidence_record(original, root)})
    rows, mapping, ledger = compile_reviews(context, root)
    context["code_files"] = code_records(root)
    if frozen:
        saved = read_json(directory/"split_feasibility.json")
        graph = build_graph(rows, ledger, context["unit_reviews"])
        problem = validate_assignment(rows, graph, context["schema"], context["qc"], mapping,
                                      saved["assignment"], saved["population_hash"])
        for row in rows:
            if row["dataset"] == "Dataset_B":
                row["split"] = saved["assignment"].get(row["dependency_component_id"], "not_assigned_qc")
        normalized = [{k: "" if v is None else str(v) for k, v in r.items()} for r in rows]
        check(internal_digest(normalized, mapping, context, graph) == status["frozen_split_hash"],
              "External-only import altered internal split evidence")
        result = {**saved, "graph": graph, "problem": problem}
    else:
        result = assess_rows(rows, mapping, context, ledger, run_solver=False)
    publish(directory, root, rows, mapping, context, result, frozen=frozen)
    return {"imported": rid, "state": result["state"], "blocker_count": len(result["blockers"])}


def compile_reviews(context, root):
    snapshot = Path(root)/context["inventory_snapshot"]
    rows = [initial_row(r) for r in read_csv(snapshot/"dataset_manifest.csv")]
    mapping = read_json(snapshot/"class_mapping.json"); mapping["schema_version"] = SCHEMA
    ledger = read_csv(snapshot/"duplicate_overlap_ledger.csv")
    by_id = {r["image_id"]: r for r in rows}
    responses = [(r["id"], read_json(verify_record(r["record"], root))) for r in context["responses"]]
    policy_versions = {"acquisition_schema": {}, "qc_policy": {}}
    for _, response in responses:
        for key in policy_versions:
            policy = response.get(key)
            if not policy:
                continue
            version = policy.get("version")
            signature = content_hash({k: v for k, v in policy.items() if k not in {"status", "evidence_ids"}})
            history = policy_versions[key]
            if version in history:
                check(history[version] == signature, "Policy version reused with changed rules")
            elif history:
                check(response.get("policy_change_justification") and response.get("not_feasibility_driven") is True,
                      "Policy revision requires documented non-feasibility-driven justification")
            history[version] = signature
    superseded = {i for _, r in responses for i in r.get("supersedes", [])}
    active = [(i, r) for i, r in responses if i not in superseded]
    claims = defaultdict(list)
    schema, qc = draft_policies(rows)
    context["unit_reviews"] = []
    adjudications, identities, schemas, policies = defaultdict(list), defaultdict(list), [], []
    external_claims = defaultdict(list)
    for rid, response in active:
        for claim in response.get("series_assignments", []):
            check(claim["review_series_id"] in context["series_membership"], "Unknown review series")
            for image_id in context["series_membership"][claim["review_series_id"]]:
                claims[image_id].append((rid, claim))
        for claim in response.get("image_assignments", []):
            check(claim["image_id"] in by_id, "Unknown image assignment")
            claims[claim["image_id"]].append((rid, claim))
        for review in response.get("unit_reviews", []):
            check(set(review["image_ids"]) <= set(by_id), "Unknown unit-review images")
            check(len(review["image_ids"]) == len(set(review["image_ids"])), "Duplicate unit-review images")
            context["unit_reviews"].append({**review, "independence_evidence_ids": [rid]})
        for edge in response.get("edge_adjudications", []):
            ids = edge.get("edge_ids", [edge.get("edge_id")])
            check(ids and all(ids), "Missing adjudication edge IDs")
            for eid in ids:
                adjudications[eid].append({**edge, "edge_id": eid, "adjudication_evidence_ids": [rid]})
        for identity in response.get("identity_verifications", []):
            identities[int(identity["class_id"])].append((rid, identity))
        for review in response.get("external_reviews", []):
            for i in review["image_ids"]:
                check(i in by_id and by_id[i]["dataset"] == "External_Test_Culture_Free", "Not an external image")
                check(review["independence_status"] in {"VERIFIED", "UNRESOLVED", "FAILED"}, "Invalid external review")
                external_claims[i].append((rid, review["independence_status"]))
        for key, bucket in (("acquisition_schema", schemas), ("qc_policy", policies)):
            if response.get(key) is not None:
                policy = response[key]
                check(policy.get("status") == "APPROVED" and policy.get("version"), "Policy must be explicitly approved and versioned")
                bucket.append({**policy, "status": "FROZEN", "evidence_ids": [rid]})
    for image_id, values in claims.items():
        row, candidates = by_id[image_id], defaultdict(list)
        for rid, claim in values:
            check(set(claim["values"]) <= set(REVIEW_KEYS), "Unrecognized provenance field")
            check(claim.get("verification_status") in {"VERIFIED", "UNRESOLVED"}, "Invalid claim verification status")
            for key, value in claim["values"].items():
                if value not in (None, ""):
                    check(isinstance(value, str), "Provenance identifiers must be strings")
                    if key == "verified_imaging_settings_json":
                        check(isinstance(json.loads(value), dict), "Verified imaging settings must be a JSON object")
                    candidates[key].append((value, rid, claim["verification_status"]))
        verified, evidence, conflicts = [], set(), []
        for key, entries in candidates.items():
            distinct = {v for v, _, _ in entries}
            if len(distinct) != 1:
                conflicts.append({"field": key, "claims": entries})
                continue
            if all(status == "VERIFIED" for _, _, status in entries):
                row[key] = entries[0][0]
                verified.append(key); evidence.update(rid for _, rid, _ in entries)
        row.update(verified_fields_json=canonical(sorted(verified)), provenance_evidence_ids=canonical(sorted(evidence)),
                   provenance_conflicts_json=canonical(conflicts))
    for i, values in external_claims.items():
        statuses = {v for _, v in values}
        by_id[i]["external_independence_status"] = next(iter(statuses)) if len(statuses) == 1 else "UNRESOLVED"
        by_id[i]["external_independence_evidence_ids"] = canonical(sorted({rid for rid, _ in values}))
    from .identity_provenance import normalize_edge
    edge_map = {normalize_edge(e)["edge_id"]: e for e in ledger}
    known_nodes = {i: r["dataset"] for i, r in by_id.items()}
    for edge in ledger:
        for side in ("a", "b"):
            known_nodes[edge["image_id_"+side]] = edge["dataset_"+side]
    for rid, response in active:
        for relationship in response.get("additional_relationships", []):
            a, b = relationship["image_id_a"], relationship["image_id_b"]
            check(a in known_nodes and b in known_nodes, "Unknown additional relationship endpoint")
            check(relationship.get("relationship") and relationship["relationship"] != "exact_sha256",
                  "Exact equality must come from hashes, not a manual relationship claim")
            new_edge = normalize_edge({**relationship, "dataset_a": known_nodes[a], "dataset_b": known_nodes[b],
                                      "adjudication_evidence_ids": [rid]})
            check(new_edge["edge_id"] not in edge_map, "Existing relationship needs explicit adjudication")
            edge_map[new_edge["edge_id"]] = new_edge
    check(set(adjudications) <= set(edge_map), "Unknown edge adjudication")
    for eid, decisions in adjudications.items():
        states = {d["link_state"] for d in decisions}
        if len(states) == 1:
            edge_map[eid].update(link_state=decisions[0]["link_state"],
                adjudication_evidence_ids=sorted({i for d in decisions for i in d["adjudication_evidence_ids"]}))
        else:
            edge_map[eid].update(link_state="CANDIDATE_LINK", other_hard_gate_domains=["contradictory_adjudication"])
    for c, reviews in identities.items():
        check(c in range(9), "Unknown identity class")
        species = {v["species"] for _, v in reviews}
        if len(species) == 1 and all(v["identity_status"] == "verified" for _, v in reviews):
            mapping["classes"][c].update(species=next(iter(species)), identity_status="verified",
                                        identity_evidence_ids=[rid for rid, _ in reviews])
            for row in rows:
                if int(row["class_id"]) == c:
                    row.update(species=next(iter(species)), identity_status="verified")
    for bucket, key, default in ((schemas, "schema", schema), (policies, "qc", qc)):
        if bucket:
            unique = {content_hash({k: v for k, v in p.items() if k != "evidence_ids"}) for p in bucket}
            check(len(unique) == 1, "Conflicting active policies; explicitly supersede the prior response")
            context[key] = {**bucket[0], "evidence_ids": sorted({i for p in bucket for i in p["evidence_ids"]})}
        else:
            context[key] = default
    # QC outcomes are reusable only for exactly the policy and source population tested.
    for run in context.get("qc_runs", []):
        if run["policy_hash"] == content_hash(context["qc"]):
            outcomes = read_json(verify_record(run["record"], root))
            for i, outcome in outcomes["outcomes"].items():
                by_id[i].update(outcome, qc_evidence_ids=canonical([run["id"]]))
    return rows, mapping, list(edge_map.values())


def run_qc(directory, root=ROOT):
    from PIL import Image, UnidentifiedImageError
    directory, root = Path(directory).resolve(), Path(root).resolve()
    status, rows, mapping, context, ledger = load(directory, root, check_sources=True)
    check(status["SPLIT_STATUS"] != "READY", "Cannot change QC after split freezing")
    qc = context["qc"]
    check(qc.get("status") == "FROZEN" and qc.get("evidence_ids"), "QC policy requires prior approval")
    check(qc["rules"] == [{"id": "primary_readability", "kind": "integrity_decode", "on_failure": "QC_EXCLUDED"}],
          "Unsupported QC rule implementation")
    for record in context["external_source_files"]:
        verify_record(record, root)
    archives, outcomes = {}, {}
    try:
        for row in rows:
            result, reason = "UNRESOLVED", ""
            try:
                archive = root/row["archive_path"]
                if archive not in archives:
                    archives[archive] = zipfile.ZipFile(archive)
                data = archives[archive].read(row["archive_member"])
                check(not row.get("sha256") or hashlib.sha256(data).hexdigest() == row["sha256"], "Image bytes changed")
                with Image.open(io.BytesIO(data)) as image:
                    image.seek(0); image.load()  # Integrity only, including for sealed external.
                result = "ELIGIBLE"
            except (UnidentifiedImageError, zipfile.BadZipFile, OSError) as exc:
                # Filesystem/permission failures are not evidence of corrupt pixels.
                if isinstance(exc, (FileNotFoundError, PermissionError)) or (isinstance(exc, OSError) and exc.errno is not None):
                    result = "UNRESOLVED"
                else:
                    result = "QC_EXCLUDED"
                reason = type(exc).__name__ + ": " + str(exc)
            outcomes[row["image_id"]] = {"analysis_eligibility": result,
                "exclusion_reason": reason, "exclusion_policy_version": qc["version"]}
    finally:
        for archive in archives.values():
            archive.close()
    record_data = {"policy_hash": content_hash(qc), "outcomes": outcomes,
        "source_files": context["internal_source_files"]+context["external_source_files"], "code_files": code_records(root)}
    run_id = stable_id("qc", record_data)
    target = directory/"provenance/qc_runs"/(run_id+".json")
    target.parent.mkdir(exist_ok=True)
    write_json(target, record_data)
    record = evidence_record(target, root)
    context["evidence_registry"][run_id] = record
    context["qc_runs"].append({"id": run_id, "record": record, "policy_hash": content_hash(qc)})
    for row in rows:
        row.update(outcomes[row["image_id"]], qc_evidence_ids=canonical([run_id]))
    result = assess_rows(rows, mapping, context, ledger, run_solver=False)
    publish(directory, root, rows, mapping, context, result)
    return dict(Counter(r["analysis_eligibility"] for r in rows))


def assess_rows(rows, mapping, context, ledger, *, run_solver=True, expected_population_hash=None):
    graph = build_graph(rows, ledger, context["unit_reviews"])
    problem, blockers = prepare_problem(rows, graph, context["schema"], context["qc"], mapping)
    if not blockers and expected_population_hash is not None and content_hash(problem) != expected_population_hash:
        blockers.append("Predeclared eligible population/strata changed before optimization; review and redeclare through evidence import/QC")
    if blockers:
        result = {"state": "BLOCKED", "optimization_status": "NOT_RUN", "assignment": None}
    elif analytical_failures(problem):
        result = {"state": "INFEASIBLE", "optimization_status": "NOT_RUN", "assignment": None,
                  "proof": analytical_failures(problem)}
    elif not run_solver:
        result = {"state": "UNKNOWN", "optimization_status": "NOT_RUN", "assignment": None,
                  "reason": "Explicit assessment required"}
    else:
        result = solve(problem, lambda a: validate_assignment(rows, graph, context["schema"], context["qc"], mapping, a),
                       context["solver_protocol"])
    result.update(blockers=blockers, population_hash=content_hash(problem),
                  policy=context["schema"]["acquisition_design_policy"], problem=problem,
                  class_summary=class_report(rows, graph, result.get("assignment")), graph=graph)
    return result


def assessment(directory, root=ROOT):
    status, rows, mapping, context, ledger = load(directory, root, check_sources=True)
    check(status["SPLIT_STATUS"] != "READY", "Frozen assessment cannot be replaced")
    rows, mapping, ledger = verify_projection(rows, mapping, context, root)
    context["code_files"] = code_records(Path(root).resolve())
    result = assess_rows(rows, mapping, context, ledger,
                        expected_population_hash=content_hash(read_json(Path(directory)/"eligible_population.json")))
    publish(Path(directory), Path(root), rows, mapping, context, result, preserve_declaration=True)
    return {"state": result["state"], "blocker_count": len(result["blockers"]),
            "optimization_status": result["optimization_status"]}


def verify_projection(rows, mapping, context, root):
    """Do not freeze hand-edited conclusions, even if artifact hashes were rebound."""
    expected, expected_mapping, ledger = compile_reviews(context, root)
    graph = build_graph(expected, ledger, context["unit_reviews"])
    check(mapping == expected_mapping, "Mapping differs from reviewed evidence")
    by_id = {r["image_id"]: r for r in rows}
    check(set(by_id) == {r["image_id"] for r in expected}, "Inventory rows were added/removed")
    for row in expected:
        actual = by_id[row["image_id"]]
        for key, value in row.items():
            if key in {"split", "evaluation_status", "acquisition_stratum"}:
                continue
            # New unresolved schema fields can be added during a draft migration,
            # but a verified conclusion must always originate in the snapshots.
            if key not in actual and value in ("UNRESOLVED", "[]", ""):
                continue
            scalar = "" if value is None else str(value)
            check(actual.get(key, "") == scalar, "Conclusion differs from reviewed evidence: " + key)
    return expected, expected_mapping, ledger


def internal_digest(rows, mapping, context, graph):
    # External eligibility and experiment exposure are deliberately out of scope.
    excluded = {"external_eligibility", "legacy_relationship_status", "legacy_exposure_status",
                "legacy_exact_match_count", "legacy_overlap_candidate_count", "evaluation_status"}
    internal = [{k: v for k, v in r.items() if k not in excluded and not k.startswith("external_")}
                for r in rows if r["dataset"] == "Dataset_B"]
    gates = [{"edge_id": e["edge_id"], "domains": e["blocking_domains"]} for e in graph["edges"]
             if any(d not in {"external_independence", "experiment_exposure"} for d in e["blocking_domains"])]
    return content_hash({"rows": sorted(internal, key=lambda r: r["image_id"]), "mapping": mapping,
        "schema": context["schema"], "qc": context["qc"], "groups": graph["components"], "internal_gate_impacts": gates})


def report_text(rows, result, status):
    lines = ["# Nine-class bacterial identity recognition — Provenance Resolution & Split Feasibility", "",
        f"AUDIT_STATUS={status['AUDIT_STATUS']}; SPLIT_STATUS={status['SPLIT_STATUS']}; "
        f"assessment={result['state']}; EXTERNAL_TEST=SEALED.", "",
        "Confirmed connectivity is not proof of independence. Unresolved candidates never merge components.", "",
        "| Class | Images | Eligible images | QC excluded | Verified cultures | Slides | Sessions | Verified eligible units |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in result["class_summary"]:
        lines.append("| " + " | ".join(str(r[k]) for k in ("class_id", "images", "analysis_eligible_images",
            "qc_excluded_images", "verified_cultures", "verified_slides", "verified_sessions", "verified_independent_eligible_units")) + " |")
    lines += ["", "Zero verified units means none established, not evidence that zero cultures existed.", "",
        "## Frozen-policy assessment", "", f"Policy: `{result['policy']}`. "
        "Predeclared analysis-eligible strata come from approved acquisition/QC rules, never optimizer selection.", "",
        f"Replication: {status['EVALUATION_REPLICATION']}. Held-out counts accompany the machine-readable class summary.", "",
        "## Next evidence or collection requirements", ""]
    reasons = result["blockers"] if result["state"] == "BLOCKED" else result.get("proof", [])
    lines += ["- " + reason for reason in reasons[:35]]
    if len(reasons) > 35:
        lines.append(f"- {len(reasons)-35} additional image/unit-specific blockers are in split_feasibility.json.")
    lines += ["", "New data collection is indicated only by demonstrated inadequacy under the frozen rules; "
        "missing evidence calls for provenance/QC resolution first.", "",
        "## Legacy and external boundaries", "",
        "All original images and the 692 legacy exact-match tags are retained. Legacy checkpoint outputs are "
        "Legacy five-class model diagnostic, never an unseen Dataset_B baseline. Initialization compatibility is experiment-specific.", "",
        "External rows remain reserved_external/sealed and outside internal components. Only provenance, file-integrity "
        "and acquisition-metadata audits are allowed. No external predictions, embeddings, morphology measurements, "
        "feature distributions, or image-derived development outputs may be generated or consumed.", ""]
    return "\n".join(lines)


def publish(directory, root, rows, mapping, context, result, *, frozen=False, preserve_declaration=False):
    directory, root = Path(directory).resolve(), Path(root).resolve()
    assignment = result.get("assignment") if frozen else None
    for row in rows:
        if row["dataset"] == "External_Test_Culture_Free":
            row.update(split="reserved_external", evaluation_status="sealed")
        else:
            row["split"] = assignment.get(row["dependency_component_id"], "not_assigned_qc") if assignment else "pending"
    # CSV is the canonical public representation; digest exactly its scalar forms.
    rows = [{k: "" if v is None else str(v) for k, v in r.items()} for r in rows]
    heldout = [r["held_out"][p] for r in result["class_summary"] for p in ("validation", "test") if r["held_out"]]
    eligible = [r for r in rows if r["dataset"] == "Dataset_B" and r["analysis_eligibility"] == "ELIGIBLE"]
    from .identity_provenance import PROVENANCE_KEYS
    relevant = [r for r in rows if r["dataset"] == "Dataset_B" and r["analysis_eligibility"] != "QC_EXCLUDED"]
    verified = sum(set(PROVENANCE_KEYS) <= set(decoded(r.get("verified_fields_json"), [])) and
                   r["biological_independence_status"] == "VERIFIED" for r in relevant)
    external = [r["external_eligibility"] for r in rows if r["dataset"] == "External_Test_Culture_Free"]
    status = {"schema_version": SCHEMA, "AUDIT_STATUS": "COMPLETE", "SPLIT_STATUS": "READY" if frozen else "NOT_READY",
        "BIOLOGICAL_PROVENANCE": "RESOLVED" if relevant and verified == len(relevant) else "PARTIALLY_RESOLVED" if verified else "UNRESOLVED",
        "ACQUISITION_CONFOUNDING": "RESOLVED" if result["state"] == "FEASIBLE" else "UNRESOLVED",
        "SPLIT_FEASIBILITY": "PASS" if result["state"] == "FEASIBLE" else result["state"],
        "EVALUATION_REPLICATION": "LIMITED" if any(min(r["cultures"], r["units"]) <= 1 for r in heldout) else "UNRATED",
        "EXTERNAL_TEST": "SEALED", "EXTERNAL_ELIGIBILITY": "CONTAMINATED" if "CONTAMINATED" in external else
            "CLEAN" if external and set(external) == {"CLEAN"} else "UNRESOLVED",
        "manifest_rows": len(rows), "blockers": result["blockers"],
        "frozen_split_hash": internal_digest(rows, mapping, context, result["graph"]) if frozen else None}
    clean_result = {k: v for k, v in result.items() if k not in {"graph", "problem"}}
    with tempfile.TemporaryDirectory(prefix=".provenance-stage-", dir=directory) as temp:
        stage = Path(temp)
        write_csv(stage/"dataset_manifest.csv", rows)
        write_json(stage/"class_mapping.json", mapping)
        write_csv(stage/"acquisition_groups.csv", list(result["graph"]["components"].values()),
                  ["dependency_component_id", "image_ids", "class_ids", "independence_status", "biological_independence_status",
                   "acquisition_independence_status", "independence_evidence_ids", "assignment_unit_eligibility"])
        write_csv(stage/"duplicate_overlap_ledger.csv", result["graph"]["edges"],
                  sorted({k for e in result["graph"]["edges"] for k in e}) or ["edge_id"])
        write_json(stage/"provenance_context.json", context)
        write_json(stage/"split_feasibility.json", clean_result)
        write_json(stage/"acquisition_condition_schema.json", context["schema"])
        write_json(stage/"qc_policy.json", context["qc"])
        write_json(stage/"eligible_population.json", read_json(directory/"eligible_population.json")
                   if preserve_declaration else result["problem"])
        (stage/"audit_report.md").write_text(report_text(rows, result, status))
        status["artifact_sha256"] = {name: digest(stage/name) for name in OUTPUT_FILES}
        write_json(stage/"audit_status.json", status)
        revision = directory/"provenance/revisions"/content_hash(status)
        revision.parent.mkdir(parents=True, exist_ok=True)
        if not revision.exists():
            shutil.copytree(stage, revision)
        for name in (*OUTPUT_FILES, "audit_status.json"):
            os.replace(stage/name, directory/name)  # Status last: partial publication fails closed.
    return status


def freeze(directory, root=ROOT):
    directory, root = Path(directory).resolve(), Path(root).resolve()
    status, rows, mapping, context, ledger = load(directory, root, check_sources=True)
    check(status["SPLIT_STATUS"] != "READY", "Partition is already frozen")
    saved = read_json(directory/"split_feasibility.json")
    check(saved["state"] == "FEASIBLE" and saved["optimization_status"] == "COMPLETE", "No completed validated selection protocol")
    check(saved.get("solver", {}).get("protocol") == context["solver_protocol"], "Solver protocol changed")
    check(context["code_files"] == code_records(root), "Implementation changed since assessment")
    rows, mapping, ledger = verify_projection(rows, mapping, context, root)
    graph = build_graph(rows, ledger, context["unit_reviews"])
    problem = validate_assignment(rows, graph, context["schema"], context["qc"], mapping,
                                  saved["assignment"], saved["population_hash"])
    result = {**saved, "graph": graph, "problem": problem}
    return publish(directory, root, rows, mapping, context, result, frozen=True)
