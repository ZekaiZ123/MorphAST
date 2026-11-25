"""Separate dataset, experiment and external-access boundaries; stdlib only."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
from .identity_provenance import (SCHEMA, PARTITIONS, ProvenanceError, check, canonical,
                                 content_hash, build_graph, validate_assignment)
from .identity_provenance_workflow import (ROOT, OUTPUT_FILES, digest, read_csv as rows_from,
    read_json, load, verify_record, code_records, compile_reviews, internal_digest, write_json)

AuditGateError = ProvenanceError
ARTIFACTS = set(OUTPUT_FILES)


def population_digest(rows):
    return content_hash(sorted(rows, key=lambda r: r["image_id"]))


checked_file = verify_record
require = check


def scalars(row):
    return {k: canonical(v) if isinstance(v, (dict, list)) else "" if v is None else str(v) for k, v in row.items()}


def split_bundle(audit_dir, repo_root):
    root = Path(repo_root).resolve()
    quick = read_json(Path(audit_dir)/"audit_status.json")
    check(quick.get("SPLIT_STATUS") == "READY", "SPLIT_STATUS must be READY")
    check(quick.get("schema_version") == SCHEMA, "Old audit approval cannot authorize a v2 experiment")
    status, rows, mapping, context, ledger = load(audit_dir, root, check_sources=True)
    check(status["AUDIT_STATUS"] == "COMPLETE" and not status["blockers"], "Dataset audit has blockers")
    check(context["code_files"] == code_records(root), "Implementation changed since assessment")
    expected_rows, expected_mapping, expected_ledger = compile_reviews(context, root)
    check(mapping == expected_mapping, "Mapping differs from original review evidence")
    graph = build_graph(expected_rows, expected_ledger, context["unit_reviews"])
    by_id = {r["image_id"]: r for r in rows}
    check(set(by_id) == {r["image_id"] for r in expected_rows}, "Original inventory population changed")
    for expected in expected_rows:
        if expected["dataset"] != "Dataset_B":
            continue
        for k, value in scalars(expected).items():
            if k not in {"split", "acquisition_stratum", "evaluation_status"}:
                check(by_id[expected["image_id"]].get(k, "") == value, "Manifest conclusion differs from evidence: " + k)
    groups = rows_from(Path(audit_dir)/"acquisition_groups.csv")
    check(sorted([scalars(g) for g in graph["components"].values()], key=lambda g: g["dependency_component_id"]) ==
          sorted(groups, key=lambda g: g["dependency_component_id"]), "Acquisition groups differ from evidence")
    assignment = {}
    for r in rows:
        if r["dataset"] == "Dataset_B" and r["analysis_eligibility"] == "ELIGIBLE":
            cid = r["dependency_component_id"]
            check(cid not in assignment or assignment[cid] == r["split"], "Dependency crosses partitions")
            assignment[cid] = r["split"]
    for r in rows:
        if r["dataset"] == "Dataset_B" and r["analysis_eligibility"] == "QC_EXCLUDED":
            check(r["split"] == assignment.get(r["dependency_component_id"], "not_assigned_qc"),
                  "Excluded row must retain its component assignment")
    for r in expected_rows:
        r["split"] = by_id[r["image_id"]]["split"]
    saved = read_json(Path(audit_dir)/"split_feasibility.json")
    check(saved["state"] == "FEASIBLE" and saved["optimization_status"] == "COMPLETE", "Selection protocol not complete")
    check(saved["assignment"] == assignment and saved["solver"]["protocol"] == context["solver_protocol"],
          "Frozen assignment/protocol changed")
    validate_assignment(expected_rows, graph, context["schema"], context["qc"], mapping, assignment, saved["population_hash"])
    check(internal_digest(rows, mapping, context, graph) == status.get("frozen_split_hash"), "Frozen split hash changed")
    return status, rows, mapping, context, graph


def assert_split_ready(audit_dir, repo_root=ROOT):
    try:
        _, rows, _, _, _ = split_bundle(audit_dir, repo_root)
        return {p: [r for r in rows if r["dataset"] == "Dataset_B" and
                    r["analysis_eligibility"] == "ELIGIBLE" and r["split"] == p] for p in PARTITIONS}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise AuditGateError("Split preflight failed: " + str(exc)) from exc


def initialization_compatibility(audit_dir, experiment_record, initialization_files, repo_root=ROOT):
    """Actual files include ALL learned pipeline weights; random initialization uses []."""
    answer = {"INITIALIZATION_COMPATIBILITY": "FAIL", "CLEAN_EVALUATION_APPROVED": False,
              "MODEL_TRAINING_APPROVED": False, "reasons": []}
    try:
        root = Path(repo_root).resolve()
        status, rows, _, context, graph = split_bundle(audit_dir, root)
        check(status["EXTERNAL_TEST"] == "SEALED", "Training blocked after external release")
        check(experiment_record is not None and initialization_files is not None,
              "Explicit experiment record and actual initialization file list required")
        experiment = read_json(experiment_record)
        check(experiment["frozen_split_hash"] == status["frozen_split_hash"], "Experiment references another split")
        declared = experiment["initialization_checkpoints"]
        expected_paths = {verify_record(r, root) for r in declared}
        actual_paths = {(root/Path(p)).resolve() for p in initialization_files}
        check(expected_paths == actual_paths and len(declared) == len(expected_paths), "Actual initialization weights differ")
        kind = experiment["initialization_kind"]
        check(kind in {"random", "pretrained"} and ((kind == "random") == (not declared)), "Ambiguous initialization kind")
        check(experiment.get("initialization_checkpoint_hash") == (declared[0]["sha256"] if declared else None),
              "Primary initialization checkpoint hash mismatch")
        history = experiment["legacy_exposure_history"]
        check(history["status"] == "VERIFIED" and history["evidence_files"], "Initialization exposure history unresolved")
        for record in history["evidence_files"]:
            verify_record(record, root)
        histories = {r["sha256"]: r for r in history["checkpoints"]}
        check(len(histories) == len(history["checkpoints"]) and set(histories) == {r["sha256"] for r in declared},
              "Incomplete/duplicate per-checkpoint exposure history")
        legacy_hashes = {r["sha256"] for r in context["legacy_source_files"] if r["role"] == "legacy_checkpoint"}
        by_id = {r["image_id"]: r for r in rows if r["dataset"] == "Dataset_B"}
        held_out = {r["dependency_component_id"] for r in by_id.values()
                    if r["analysis_eligibility"] == "ELIGIBLE" and r["split"] in {"validation", "test"}}
        for checkpoint_hash, history in histories.items():
            scope = history["scope"]
            check(scope in {"NO_DATASET_B_EXPOSURE", "DOCUMENTED_EXPOSURE"}, "Unknown checkpoint exposure scope")
            known_legacy = checkpoint_hash in legacy_hashes
            check(not known_legacy or (scope == "DOCUMENTED_EXPOSURE" and history.get("uses_legacy_corpus") is True),
                  "Known legacy checkpoint cannot claim no Dataset_B exposure")
            if history.get("uses_legacy_corpus"):
                for record in context["legacy_source_files"]:
                    verify_record(record, root)
                check(not any(r["dependency_component_id"] in held_out and r["legacy_relationship_status"] in
                              {"CONFIRMED", "POSSIBLE"} for r in by_id.values()),
                      "Legacy exposure reaches or may reach held-out assignment units")
            exposed = history.get("exposed_image_ids", [])
            check(scope != "NO_DATASET_B_EXPOSURE" or (not exposed and not history.get("uses_legacy_corpus")),
                  "Contradictory exposure declaration")
            check(set(exposed) <= set(by_id), "Unknown exposed image IDs")
            check(not ({by_id[i]["dependency_component_id"] for i in exposed} & held_out), "Checkpoint has held-out exposure")
        answer.update(INITIALIZATION_COMPATIBILITY="PASS", CLEAN_EVALUATION_APPROVED=True,
                      MODEL_TRAINING_APPROVED=True, frozen_split_hash=status["frozen_split_hash"],
                      experiment_record_sha256=digest(experiment_record))
    except (ProvenanceError, OSError, ValueError, KeyError, TypeError) as exc:
        answer["reasons"].append(str(exc))
    return answer


def assert_training_ready(audit_dir, repo_root=ROOT, *, experiment_record=None, initialization_files=None):
    result = initialization_compatibility(audit_dir, experiment_record, initialization_files, repo_root)
    check(result["MODEL_TRAINING_APPROVED"], "; ".join(result["reasons"]))
    return assert_split_ready(audit_dir, repo_root)


def assert_external_access(audit_dir, *, operation, purpose="generate", repo_root=ROOT, **freeze_args):
    try:
        status, rows, _, context, _ = load(audit_dir, repo_root)
        check(purpose in {"generate", "consume"}, "Invalid external purpose")
        if operation in {"provenance", "file_integrity", "acquisition_metadata"}:
            for record in context["external_source_files"]:
                verify_record(record, repo_root)
            return [r for r in rows if r["dataset"] == "External_Test_Culture_Free"]
        check(operation in {"predictions", "embeddings", "morphology", "feature_distributions", "image_derived"},
              "Unknown external operation (denied)")
        check(status["EXTERNAL_TEST"] == "RELEASED", "External image-derived information is SEALED")
        return assert_external_evaluation_ready(audit_dir, purpose=purpose, repo_root=repo_root, **freeze_args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise AuditGateError("External access failed: " + str(exc)) from exc


def assert_external_evaluation_ready(audit_dir, freeze_record=None, *, checkpoint=None, pipeline_config=None,
                                     purpose="generate", repo_root=ROOT):
    try:
        root = Path(repo_root).resolve()
        quick = read_json(Path(audit_dir)/"audit_status.json")
        check(quick["EXTERNAL_TEST"] == "RELEASED", "External image-derived information is SEALED")
        status, rows, mapping, context, _ = split_bundle(audit_dir, root)
        check(status["EXTERNAL_ELIGIBILITY"] == "CLEAN", "External benchmark independence not established")
        check(purpose in {"generate", "consume"}, "Invalid external purpose")
        external = [r for r in rows if r["dataset"] == "External_Test_Culture_Free"]
        allowed = {"ready"} if purpose == "generate" else {"ready", "evaluated"}
        check(external and all(r["split"] == "external_test" and r["evaluation_status"] in allowed for r in external),
              "External rows not released for this operation")
        expected, _, ledger = compile_reviews(context, root)
        build_graph(expected, ledger, context["unit_reviews"])
        check(all(r["external_eligibility"] == "CLEAN" for r in expected if r["dataset"] == "External_Test_Culture_Free"),
              "External cleanliness differs from original evidence")
        for record in context["external_source_files"]:
            verify_record(record, root)
        freeze = read_json(freeze_record)
        check(freeze["status"] == "FROZEN" and freeze["frozen_split_hash"] == status["frozen_split_hash"], "Final model/split not frozen")
        check((root/Path(checkpoint)).resolve() == verify_record(freeze["checkpoint"], root), "Actual checkpoint differs")
        config = verify_record(freeze["pipeline_config"], root)
        check((root/Path(pipeline_config)).resolve() == config, "Actual pipeline differs")
        pipeline = read_json(config)
        check(isinstance(pipeline["preprocessing"], dict) and pipeline["inference_code_files"], "Incomplete frozen pipeline")
        for record in pipeline["inference_code_files"] + pipeline["additional_model_files"]:
            verify_record(record, root)
        sealed_status = read_json(verify_record(freeze["sealed_audit_status"], root))
        check(sealed_status["SPLIT_STATUS"] == "READY" and sealed_status["EXTERNAL_TEST"] == "SEALED", "Invalid sealed snapshot")
        check(sealed_status["frozen_split_hash"] == status["frozen_split_hash"], "Sealed internal split changed")
        sealed_path = verify_record(freeze["sealed_manifest"], root)
        check(digest(sealed_path) == sealed_status["artifact_sha256"]["dataset_manifest.csv"], "Unbound sealed cohort")
        original = [r for r in rows_from(sealed_path) if r["dataset"] == "External_Test_Culture_Free"]
        normalized = [{**r, "split": "reserved_external", "evaluation_status": "sealed"} for r in external]
        check(population_digest(original) == population_digest(normalized), "External cohort changed after model freezing")
        return [r for r in external if r["analysis_eligibility"] == "ELIGIBLE"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise AuditGateError("External preflight failed: " + str(exc)) from exc


def reject_v1_in_legacy_trainer(manifest):
    with Path(manifest).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = set(reader.fieldnames or [])
        first = next(reader, {})
    if first.get("schema_version") in {"bacterial_identity_v1", SCHEMA} or {"source_label_id", "evaluation_status"} <= header:
        raise AuditGateError("Bacterial identity v1 manifest rejected: legacy trainers regenerate folds; use the provenance guards and frozen partitions")


def preflight_legacy_cli(argv=None):
    args = sys.argv[1:] if argv is None else argv
    for i, arg in enumerate(args):
        if arg.startswith("--manifest_csv="):
            reject_v1_in_legacy_trainer(arg.split("=", 1)[1])
        elif arg == "--manifest_csv" and i+1 < len(args):
            reject_v1_in_legacy_trainer(args[i+1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("split", "training", "experiment", "external"))
    parser.add_argument("audit_dir", type=Path)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--experiment-record", type=Path)
    parser.add_argument("--initialization-file", action="append", type=Path, default=[])
    parser.add_argument("--freeze-record", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--pipeline-config", type=Path)
    parser.add_argument("--purpose", choices=("generate", "consume"), default="generate")
    parser.add_argument("--external-operation", default="predictions")
    args = parser.parse_args()
    try:
        if args.operation == "split":
            result = {p: len(r) for p, r in assert_split_ready(args.audit_dir, args.repo_root).items()}
        elif args.operation == "training":
            result = {p: len(r) for p, r in assert_training_ready(args.audit_dir, args.repo_root,
                experiment_record=args.experiment_record, initialization_files=args.initialization_file).items()}
        elif args.operation == "experiment":
            result = initialization_compatibility(args.audit_dir, args.experiment_record, args.initialization_file, args.repo_root)
            target = args.audit_dir/"provenance/experiments"/(content_hash(result)+".json")
            target.parent.mkdir(parents=True, exist_ok=True)
            write_json(target, result)
        else:
            result = {"images": len(assert_external_access(args.audit_dir, operation=args.external_operation,
                purpose=args.purpose, repo_root=args.repo_root, freeze_record=args.freeze_record,
                checkpoint=args.checkpoint, pipeline_config=args.pipeline_config))}
        print(json.dumps(result, indent=2))
    except (ProvenanceError, OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit("BLOCKED: " + str(exc)) from exc
