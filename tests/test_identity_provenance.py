"""Scientific boundary and end-to-end tests for the provenance milestone."""
import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/"Code"))
from test_identity_audit import image_bytes
from utils.identity_audit import inspect_image, class_mapping, LEDGER_FIELDS
from utils.identity_provenance import (SCHEMA, POLICY, PROVENANCE_KEYS, PARTITIONS, ProvenanceError,
    canonical, content_hash, initial_row, normalize_edge, build_graph, prepare_problem,
    analytical_failures, validate_assignment, class_report, acquisition_levels, policy_errors)
from utils.identity_split_milp import solve, DEFAULT_PROTOCOL
from utils.identity_provenance_workflow import (BASE_FILES, OUTPUT_FILES, prepare, import_response, run_qc,
    assessment, freeze, load, read_json, write_json, read_csv, write_csv, evidence_record,
    digest, draft_policies, compile_reviews, code_records)
from utils.identity_audit_gate import (AuditGateError, assert_split_ready, assert_training_ready,
    assert_external_access, assert_external_evaluation_ready, initialization_compatibility, reject_v1_in_legacy_trainer)


def fixture_rows(n=3):
    rows = []
    for unit in range(n):
        for c in range(9):
            i = unit*9+c
            row = initial_row(inspect_image(image_bytes(i), f"{unit}_{c}.png", "Dataset_B", c))
            row.update(culture_id=f"culture-{unit}-{c}", sample_id=f"sample-{unit}-{c}",
                preparation_id=f"prep-{unit}-{c}", slide_id=f"slide-{unit}-{c}",
                acquisition_session_id=f"session-{unit}", field_id=f"field-{i}",
                verified_fields_json=canonical(list(PROVENANCE_KEYS)), provenance_evidence_ids='["review"]',
                analysis_eligibility="ELIGIBLE", exclusion_reason="", exclusion_policy_version="qc-v1",
                qc_evidence_ids='["qc-run"]')
            if c == 6:
                row.update(species="Staphylococcus aureus", identity_status="verified")
            rows.append(row)
    return rows


def fixture_policies(rows):
    schema, qc = draft_policies(rows)
    schema.update(version="acquisition-v1", status="FROZEN", evidence_ids=["review"])
    for factor in schema["factors"]:
        factor.update(role="INFORMATIONAL", justification="Synthetic fixture: prespecified comparable observations")
        if factor["source"] == "conditions.format":
            factor["role"] = "HARD_MATCH"
        if factor["source"] == "acquisition_session_id":
            factor.update(role="CROSSED_BATCH", requirement={"scope": "both", "no_class_unique_level": True,
                          "minimum_identity_support_per_level": 9})
    qc.update(version="qc-v1", status="FROZEN", evidence_ids=["review"])
    return schema, qc


def unit_reviews(rows):
    buckets = {}
    for row in rows:
        buckets.setdefault(row["acquisition_session_id"], []).append(row["image_id"])
    return [{"image_ids": sorted(ids), "biological_independence_status": "VERIFIED",
             "acquisition_independence_status": "VERIFIED", "independence_evidence_ids": ["review"]}
            for ids in buckets.values()]


def edge(a, b, state="CANDIDATE_LINK", kind="stage_footprint_overlap", dataset_b="Dataset_B"):
    return normalize_edge({"image_id_a": a["image_id"], "image_id_b": b["image_id"],
        "dataset_a": a.get("dataset", "Dataset_B"), "dataset_b": b.get("dataset", dataset_b),
        "relationship": kind, "link_state": state,
        "adjudication_evidence_ids": [] if state == "CANDIDATE_LINK" else ["review"]})


class GraphPolicyTests(unittest.TestCase):
    def setUp(self):
        self.rows = fixture_rows()
        self.schema, self.qc = fixture_policies(self.rows)
        self.mapping = class_mapping()
        self.mapping["classes"][6].update(species="Staphylococcus aureus", identity_status="verified")
        self.reviews = unit_reviews(self.rows)

    def graph(self, edges=()):
        return build_graph(self.rows, edges, self.reviews)

    def problem(self, graph):
        return prepare_problem(self.rows, graph, self.schema, self.qc, self.mapping)

    def test_multiclass_components_and_independence(self):
        graph = self.graph()
        problem, blockers = self.problem(graph)
        self.assertFalse(blockers)
        self.assertEqual(len(graph["components"]), 3)
        self.assertTrue(all(u["A"] == [1]*9 for u in problem["units"].values()))
        self.assertEqual([r["verified_independent_eligible_units"] for r in class_report(self.rows, graph)], [3]*9)

    def test_disconnected_images_are_not_verified_replicates(self):
        self.reviews = []
        for row in self.rows:
            row["verified_fields_json"] = "[]"
        graph = self.graph()
        self.assertEqual(len(graph["components"]), 27)
        self.assertTrue(self.problem(graph)[1])
        self.assertEqual(sum(r["verified_independent_eligible_units"] for r in class_report(self.rows, graph)), 0)

    def test_candidate_chain_blocks_without_merging(self):
        graph = self.graph([edge(self.rows[0], self.rows[9]), edge(self.rows[9], self.rows[18])])
        self.assertEqual(len(graph["components"]), 3)
        self.assertTrue(all("internal_independence" in e["blocking_domains"] for e in graph["edges"]))
        self.assertTrue(self.problem(graph)[1])

    def test_redundant_candidate_is_nonblocking(self):
        graph = self.graph([edge(self.rows[0], self.rows[1])])
        self.assertFalse(graph["edges"][0]["blocking_domains"])
        self.assertFalse(self.problem(graph)[1])

    def test_redundant_candidate_other_gate_still_blocks(self):
        candidate = edge(self.rows[0], self.rows[1])
        candidate["other_hard_gate_domains"] = ["identity_verification"]
        self.assertTrue(self.problem(self.graph([candidate]))[1])

    def test_confirmed_relationship_merges_and_requires_new_review(self):
        graph = self.graph([edge(self.rows[0], self.rows[9], "CONFIRMED_MUST_LINK")])
        self.assertEqual(len(graph["components"]), 2)
        self.assertTrue(self.problem(graph)[1])

    def test_rejected_candidate_has_no_constraint(self):
        graph = self.graph([edge(self.rows[0], self.rows[9], "REJECTED")])
        self.assertEqual(len(graph["components"]), 3)
        self.assertFalse(self.problem(graph)[1])

    def test_exact_duplicate_cannot_be_rejected(self):
        with self.assertRaisesRegex(ProvenanceError, "Exact duplicates"):
            edge(self.rows[0], self.rows[9], "REJECTED", "exact_sha256")

    def test_qc_exclusion_does_not_break_graph_bridge(self):
        self.rows[0].update(analysis_eligibility="QC_EXCLUDED", exclusion_reason="Prespecified decode failure")
        graph = self.graph([edge(self.rows[0], self.rows[9], "CONFIRMED_MUST_LINK")])
        self.assertEqual(len(graph["components"]), 2)

    def test_candidate_between_wholly_excluded_units_does_not_block_assignment(self):
        for row in self.rows:
            row.update(analysis_eligibility="QC_EXCLUDED", exclusion_reason="Frozen technical QC")
        graph = self.graph([edge(self.rows[0], self.rows[9])])
        self.assertFalse(graph["edges"][0]["blocking_domains"])

    def test_external_contamination_does_not_merge_internal_components(self):
        ext = initial_row(inspect_image(image_bytes(100), "external.png", "External_Test_Culture_Free", 5))
        self.rows.append(ext)
        graph = self.graph([edge(self.rows[0], ext, "CONFIRMED_MUST_LINK"), edge(self.rows[9], ext, "CONFIRMED_MUST_LINK")])
        self.assertEqual(len(graph["components"]), 3)
        self.assertEqual(ext["external_eligibility"], "CONTAMINATED")
        self.assertEqual(ext["dependency_component_id"], "")
        self.assertFalse(self.problem(graph)[1])

    def test_collective_candidates_through_legacy_block_internal(self):
        legacy = {"image_id": "legacy", "dataset": "Legacy_training"}
        graph = self.graph([edge(self.rows[0], legacy), edge(self.rows[9], legacy)])
        self.assertEqual(len(graph["components"]), 3)
        self.assertTrue(all("internal_independence" in e["blocking_domains"] for e in graph["edges"]))

    def test_legacy_only_candidate_is_experiment_not_split_blocker(self):
        graph = self.graph([edge(self.rows[0], {"image_id": "legacy", "dataset": "Legacy_training"})])
        self.assertEqual(graph["edges"][0]["blocking_domains"], ["experiment_exposure"])
        self.assertFalse(self.problem(graph)[1])

    def test_separate_biological_acquisition_assessments(self):
        self.reviews[0]["acquisition_independence_status"] = "FAILED"
        graph = self.graph()
        group = graph["components"][self.rows[0]["dependency_component_id"]]
        self.assertEqual(group["biological_independence_status"], "VERIFIED")
        self.assertEqual(group["acquisition_independence_status"], "FAILED")
        self.assertEqual(group["assignment_unit_eligibility"], "INELIGIBLE")
        self.assertTrue(self.problem(graph)[1])

    def test_unknown_factor_blocks(self):
        self.schema["factors"][0]["role"] = "UNKNOWN_BLOCKING"
        self.assertTrue(self.problem(self.graph())[1])

    def test_omitted_observed_factor_blocks(self):
        self.schema["factors"].pop()
        self.assertTrue(self.problem(self.graph())[1])

    def test_bounded_variation_has_explicit_nonambiguous_intervals(self):
        factor = {"source": "width", "role": "BOUNDED_VARIATION", "normalization": "numeric",
                  "justification": "Approved fixture tolerance", "intervals": [{"label": "small", "min": 30, "max": 40}]}
        self.schema["factors"].append(factor)
        a, _ = acquisition_levels(self.rows[0], self.schema)
        self.rows[0]["width"] = 40
        with self.assertRaisesRegex(ProvenanceError, "outside"):
            acquisition_levels(self.rows[0], self.schema)

    def test_batch_unique_level_is_proven_inadequate(self):
        self.schema["factors"].append({"source": "class_id", "role": "CROSSED_BATCH", "normalization": "string",
            "justification": "Deliberate confounding fixture", "requirement": {"scope": "dataset",
             "no_class_unique_level": True, "minimum_identity_support_per_level": 2}})
        problem, blockers = self.problem(self.graph())
        self.assertFalse(blockers)
        self.assertTrue(any("crossing" in r for r in analytical_failures(problem)))

    def test_exact_class_specific_strata_are_infeasible_not_relaxed(self):
        self.schema["factors"].append({"source": "class_id", "role": "HARD_MATCH", "justification": "Deliberate fixture"})
        problem, blockers = self.problem(self.graph())
        self.assertFalse(blockers)
        self.assertTrue(any(POLICY in r for r in analytical_failures(problem)))

    def test_two_documented_units_are_infeasible(self):
        self.rows = self.rows[:18]
        self.reviews = unit_reviews(self.rows)
        problem, blockers = self.problem(self.graph())
        self.assertFalse(blockers)
        self.assertEqual(len(analytical_failures(problem)), 9)

    def test_qc_policy_change_invalidates_outcomes(self):
        self.qc["version"] = "qc-v2"
        self.assertTrue(self.problem(self.graph())[1])

    def test_validator_rejects_missing_unit_or_class(self):
        graph = self.graph()
        assignment = {u: p for u, p in zip(sorted(graph["components"]), PARTITIONS)}
        validate_assignment(self.rows, graph, self.schema, self.qc, self.mapping, assignment)
        assignment.pop(next(iter(assignment)))
        with self.assertRaises(ProvenanceError):
            validate_assignment(self.rows, graph, self.schema, self.qc, self.mapping, assignment)

    def test_large_confirmed_group_is_iterative(self):
        for i in range(1100):
            self.rows.append({**self.rows[0], "image_id": f"extra-{i}"})
        self.reviews = unit_reviews(self.rows)
        self.assertEqual(len(self.graph()["components"]), 3)

    def test_milp_multiclass_deterministic_and_replication_limited(self):
        graph = self.graph()
        problem, blockers = self.problem(graph)
        validator = lambda a: validate_assignment(self.rows, graph, self.schema, self.qc, self.mapping, a)
        first, second = solve(problem, validator), solve(problem, validator)
        self.assertEqual(first["state"], "FEASIBLE", first)
        self.assertEqual(first["optimization_status"], "COMPLETE", first)
        self.assertEqual(first["assignment"], second["assignment"])
        report = class_report(self.rows, graph, first["assignment"])
        self.assertTrue(all(r["held_out"]["test"]["units"] == 1 for r in report))

    def test_validator_rejection_never_freezes_solver_candidate(self):
        graph = self.graph()
        problem, _ = self.problem(graph)
        def reject(_):
            raise ProvenanceError("Deliberately rejected independent validation")
        result = solve(problem, reject)
        self.assertEqual(result["state"], "UNKNOWN")
        self.assertIsNone(result["assignment"])

    def test_missing_solver_is_unknown_not_infeasible(self):
        import builtins
        original = builtins.__import__
        def importing(name, *args, **kwargs):
            if name == "scipy":
                raise ImportError("No SciPy")
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=importing):
            self.assertEqual(solve(self.problem(self.graph())[0], lambda a: None)["state"], "UNKNOWN")

    def test_interrupted_solver_valid_witness_is_feasible_not_freezable(self):
        import numpy as np
        graph = self.graph()
        problem, _ = self.problem(graph)
        def interrupted(cost, **kwargs):
            values = np.zeros(len(cost))
            for i in range(3):
                values[3*i+i] = 1
            return SimpleNamespace(status=1, message="time limit", x=values, fun=None)
        with patch("scipy.optimize.milp", side_effect=interrupted):
            result = solve(problem, lambda a: validate_assignment(self.rows, graph, self.schema, self.qc, self.mapping, a))
        self.assertEqual(result["state"], "FEASIBLE")
        self.assertEqual(result["optimization_status"], "INCOMPLETE")

    def test_interruption_without_witness_is_unknown(self):
        graph = self.graph()
        with patch("scipy.optimize.milp", return_value=SimpleNamespace(status=1, message="time limit", x=None)):
            self.assertEqual(solve(self.problem(graph)[0], lambda a: None)["state"], "UNKNOWN")

    def test_genuine_milp_infeasibility_with_three_units_per_class(self):
        # Four triples of a four-unit set cannot all be rainbow-colored with
        # three partitions, despite each class having three documented units.
        self.rows = [r for r in fixture_rows(4) if int(r["class_id"]) % 4 != int(r["acquisition_session_id"].split("-")[-1])]
        self.reviews = unit_reviews(self.rows)
        self.schema, self.qc = fixture_policies(self.rows)
        for f in self.schema["factors"]:
            if f["source"] == "acquisition_session_id":
                f["role"] = "INFORMATIONAL"
        graph = self.graph()
        problem, blockers = self.problem(graph)
        self.assertFalse(blockers)
        self.assertFalse(analytical_failures(problem))
        result = solve(problem, lambda a: validate_assignment(self.rows, graph, self.schema, self.qc, self.mapping, a))
        self.assertEqual(result["state"], "INFEASIBLE", result)

    def test_component_balance_precedes_large_image_count(self):
        self.rows = fixture_rows(10)
        for c in range(9):
            for i in range(40):
                self.rows.append({**self.rows[c], "image_id": f"many-images-{c}-{i}"})
        self.reviews = unit_reviews(self.rows)
        self.schema, self.qc = fixture_policies(self.rows)
        graph = self.graph()
        problem, blockers = self.problem(graph)
        self.assertFalse(blockers)
        result = solve(problem, lambda a: validate_assignment(self.rows, graph, self.schema, self.qc, self.mapping, a))
        self.assertEqual(result["optimization_status"], "COMPLETE", result)
        self.assertEqual(sum(p == "train" for p in result["assignment"].values()), 7)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.audit = self.root/"audit"; self.audit.mkdir()
        self.baseline = []
        self.archive = self.root/"internal.zip"
        with zipfile.ZipFile(self.archive, "w") as z:
            for i in range(27):
                data = b"corrupt pixel payload" if i == getattr(self, "corrupt_index", None) else image_bytes(i)
                member = f"Strain_{i%9+1:03d}/{i}.png"
                z.writestr(member, data)
                self.baseline.append(inspect_image(data, f"zip://internal.zip!{member}", "Dataset_B", i%9, "internal.zip", member))
        self.external_archive = self.root/"external.zip"
        with zipfile.ZipFile(self.external_archive, "w") as z:
            data = image_bytes(100)
            z.writestr("external.png", data)
        self.baseline.append(inspect_image(data, "zip://external.zip!external.png", "External_Test_Culture_Free", 5,
                                           "external.zip", "external.png"))
        self.legacy_checkpoint = self.root/"legacy.pt"; self.legacy_checkpoint.write_bytes(b"legacy-weight-fixture")
        write_csv(self.audit/"dataset_manifest.csv", self.baseline)
        write_csv(self.audit/"acquisition_groups.csv", [], ["candidate_spatial_group_id"])
        write_csv(self.audit/"duplicate_overlap_ledger.csv", [], LEDGER_FIELDS)
        write_json(self.audit/"class_mapping.json", class_mapping())
        (self.audit/"audit_report.md").write_text("Synthetic original inventory, not laboratory evidence.\n")
        write_json(self.audit/"audit_status.json", {"schema_version": "bacterial_identity_v1", "SPLIT_STATUS": "NOT_READY",
            "EXTERNAL_TEST": "SEALED", "artifact_sha256": {name: digest(self.audit/name) for name in BASE_FILES[:-1]},
            "input_files": [{**evidence_record(self.archive, self.root), "role": "dataset_b_archive"},
                {**evidence_record(self.external_archive, self.root), "role": "external_archive"},
                {**evidence_record(self.legacy_checkpoint, self.root), "role": "legacy_checkpoint"}]})
        prepare(self.audit, self.root)

    def tearDown(self):
        self.temp.cleanup()

    def response(self):
        _, rows, _, context, _ = load(self.audit, self.root)
        schema, qc = fixture_policies(rows)
        schema["status"] = qc["status"] = "APPROVED"
        assignments = []
        reviews = []
        for unit in range(3):
            ids = []
            for c in range(9):
                row = self.baseline[unit*9+c]
                ids.append(row["image_id"])
                values = {"culture_id": f"culture-{unit}-{c}", "sample_id": f"sample-{unit}-{c}",
                    "preparation_id": f"prep-{unit}-{c}", "slide_id": f"slide-{unit}-{c}",
                    "acquisition_session_id": f"session-{unit}", "field_id": f"field-{unit}-{c}",
                    "source_image_id": row["image_id"]}
                assignments.append({"image_id": row["image_id"], "values": values, "verification_status": "VERIFIED"})
            reviews.append({"image_ids": ids, "biological_independence_status": "VERIFIED", "acquisition_independence_status": "VERIFIED"})
        return {"review_packet_version": "review-packet-v1", "reviewer": "Synthetic reviewer",
            "review_timestamp": "2026-09-09T12:00:00+00:00", "evidence_reference": "Synthetic unit-test evidence only",
            "evidence_files": [], "image_assignments": assignments, "unit_reviews": reviews,
            "identity_verifications": [{"class_id": 6, "species": "Staphylococcus aureus", "identity_status": "verified"}],
            "acquisition_schema": schema, "qc_policy": qc,
            "external_reviews": [{"image_ids": [self.baseline[-1]["image_id"]], "independence_status": "VERIFIED"}]}

    def import_review(self, response=None, filename="response.json"):
        path = self.root/filename
        write_json(path, self.response() if response is None else response)
        return import_response(self.audit, path, self.root)

    def freeze_fixture(self):
        self.import_review()
        run_qc(self.audit, self.root)
        result = assessment(self.audit, self.root)
        self.assertEqual(result["state"], "FEASIBLE", read_json(self.audit/"split_feasibility.json"))
        return freeze(self.audit, self.root)

    def experiment(self):
        status = read_json(self.audit/"audit_status.json")
        evidence = self.root/"initialization_history.txt"; evidence.write_text("Synthetic random initialization; no learned pipeline weights.")
        record = {"frozen_split_hash": status["frozen_split_hash"], "initialization_kind": "random",
            "initialization_checkpoint_hash": None, "initialization_checkpoints": [],
            "legacy_exposure_history": {"status": "VERIFIED", "evidence_files": [evidence_record(evidence, self.root)], "checkpoints": []}}
        target = self.root/"experiment.json"; write_json(target, record)
        return target

    def test_prepare_preserves_inventory_and_is_blocked(self):
        status, rows, _, _, _ = load(self.audit, self.root)
        self.assertEqual(status["SPLIT_STATUS"], "NOT_READY")
        self.assertNotIn("MODEL_TRAINING_APPROVED", status)
        self.assertEqual(len(rows), 28)
        self.assertTrue(all(r["analysis_eligibility"] == "UNRESOLVED" for r in rows))
        self.assertTrue(prepare(self.audit, self.root)["already_prepared"])

    def test_original_response_snapshot_and_idempotent_import(self):
        result = self.import_review()
        _, _, _, context, _ = load(self.audit, self.root)
        rid = result["imported"]
        retained = self.root/context["evidence_registry"][rid]["path"]
        self.assertEqual(retained.read_bytes(), (self.root/"response.json").read_bytes())
        self.assertEqual(self.import_review()["already_imported"], rid)
        receipt = read_json(self.root/context["evidence_registry"][rid+"_receipt"]["path"])
        for k in ("review_packet_version", "reviewer", "review_timestamp", "evidence_reference",
                  "original_review_response", "response_hash", "import_code_version", "import_timestamp"):
            self.assertIn(k, receipt)

    def test_conflicting_claim_is_not_silently_overwritten(self):
        self.import_review()
        response = self.response()
        response.pop("acquisition_schema"); response.pop("qc_policy")
        response["image_assignments"][0]["values"]["culture_id"] = "contradictory-culture"
        self.import_review(response, "conflict.json")
        rows = read_csv(self.audit/"dataset_manifest.csv")
        target = next(r for r in rows if r["image_id"] == self.baseline[0]["image_id"])
        self.assertTrue(json.loads(target["provenance_conflicts_json"]))
        self.assertEqual(read_json(self.audit/"split_feasibility.json")["state"], "BLOCKED")

    def test_qc_requires_frozen_policy(self):
        with self.assertRaisesRegex(ProvenanceError, "prior approval"):
            run_qc(self.audit, self.root)

    def test_predeclared_population_checked_before_solver(self):
        self.import_review()
        run_qc(self.audit, self.root)
        population = read_json(self.audit/"eligible_population.json")
        population["eligible_image_ids"].pop()
        write_json(self.audit/"eligible_population.json", population)
        status = read_json(self.audit/"audit_status.json")
        status["artifact_sha256"]["eligible_population.json"] = digest(self.audit/"eligible_population.json")
        write_json(self.audit/"audit_status.json", status)
        with patch("utils.identity_provenance_workflow.solve") as mocked:
            result = assessment(self.audit, self.root)
            repeated = assessment(self.audit, self.root)
        self.assertEqual(result["state"], "BLOCKED")
        self.assertEqual(repeated["state"], "BLOCKED")
        mocked.assert_not_called()

    def test_corrupt_image_remains_an_explicit_excluded_row(self):
        self.tearDown()
        self.corrupt_index = 0
        self.setUp()
        self.assertEqual(self.baseline[0]["primary_integrity_status"], "FAIL")
        self.import_review()
        result = run_qc(self.audit, self.root)
        self.assertEqual(result["QC_EXCLUDED"], 1)
        rows = read_csv(self.audit/"dataset_manifest.csv")
        self.assertEqual(len(rows), 28)
        failed = next(r for r in rows if r["image_id"] == self.baseline[0]["image_id"])
        self.assertEqual(failed["analysis_eligibility"], "QC_EXCLUDED")
        self.assertTrue(failed["exclusion_reason"])
        self.assertTrue(failed["dependency_component_id"])

    def test_policy_version_cannot_be_reused_with_changed_rules(self):
        original = self.import_review()
        response = self.response()
        response["supersedes"] = [original["imported"]]
        response["acquisition_schema"]["factors"][0]["role"] = "HARD_MATCH"
        with self.assertRaisesRegex(ProvenanceError, "version reused"):
            self.import_review(response, "changed_policy.json")

    def test_new_policy_version_requires_scientific_change_rationale(self):
        original = self.import_review()
        response = self.response()
        response["supersedes"] = [original["imported"]]
        response["qc_policy"]["version"] = "qc-v2"
        with self.assertRaisesRegex(ProvenanceError, "justification"):
            self.import_review(response, "changed_policy.json")

    def test_unknown_microscopy_qc_rule_is_not_silently_accepted(self):
        response = self.response()
        response["qc_policy"]["rules"].append({"id": "posthoc", "kind": "morphology"})
        self.import_review(response)
        with self.assertRaisesRegex(ProvenanceError, "Unsupported QC"):
            run_qc(self.audit, self.root)

    def test_end_to_end_split_is_independent_of_initialization(self):
        status = self.freeze_fixture()
        self.assertEqual(status["SPLIT_STATUS"], "READY")
        self.assertEqual(status["EVALUATION_REPLICATION"], "LIMITED")
        self.assertEqual({p: len(r) for p, r in assert_split_ready(self.audit, self.root).items()},
                         {"train": 9, "validation": 9, "test": 9})
        with self.assertRaisesRegex(ProvenanceError, "experiment record"):
            assert_training_ready(self.audit, self.root)
        record = self.experiment()
        result = initialization_compatibility(self.audit, record, [], self.root)
        self.assertTrue(result["CLEAN_EVALUATION_APPROVED"], result)
        self.assertEqual(len(assert_training_ready(self.audit, self.root, experiment_record=record, initialization_files=[])["test"]), 9)

    def test_heldout_exposure_fails_experiment_not_split(self):
        self.freeze_fixture()
        target = self.experiment(); record = read_json(target)
        test_image = assert_split_ready(self.audit, self.root)["test"][0]["image_id"]
        weights = self.root/"weights.pt"; weights.write_bytes(b"exposed fixture")
        record.update(initialization_kind="pretrained", initialization_checkpoint_hash=digest(weights),
                      initialization_checkpoints=[evidence_record(weights, self.root)])
        record["legacy_exposure_history"]["checkpoints"] = [{"sha256": digest(weights), "scope": "DOCUMENTED_EXPOSURE",
            "exposed_image_ids": [test_image], "uses_legacy_corpus": False}]
        write_json(target, record)
        self.assertFalse(initialization_compatibility(self.audit, target, [weights], self.root)["CLEAN_EVALUATION_APPROVED"])
        self.assertEqual(len(assert_split_ready(self.audit, self.root)["test"]), 9)

    def test_known_legacy_hash_cannot_claim_unseen_history(self):
        self.freeze_fixture()
        target = self.experiment(); record = read_json(target)
        record.update(initialization_kind="pretrained", initialization_checkpoint_hash=digest(self.legacy_checkpoint),
                      initialization_checkpoints=[evidence_record(self.legacy_checkpoint, self.root)])
        record["legacy_exposure_history"]["checkpoints"] = [{"sha256": digest(self.legacy_checkpoint), "scope": "NO_DATASET_B_EXPOSURE"}]
        write_json(target, record)
        result = initialization_compatibility(self.audit, target, [self.legacy_checkpoint], self.root)
        self.assertFalse(result["CLEAN_EVALUATION_APPROVED"])
        self.assertIn("Known legacy", result["reasons"][0])

    def test_no_images_or_qc_changes_after_freeze(self):
        self.freeze_fixture()
        with self.assertRaises(ProvenanceError):
            run_qc(self.audit, self.root)
        with self.assertRaises(ProvenanceError):
            self.import_review()

    def test_external_only_provenance_update_preserves_frozen_internal_split(self):
        before = self.freeze_fixture()
        response = {"review_packet_version": "review-packet-v1", "reviewer": "Synthetic external reviewer",
            "review_timestamp": "2026-09-09T13:00:00+00:00", "evidence_reference": "Synthetic derivative evidence",
            "additional_relationships": [{"image_id_a": self.baseline[0]["image_id"],
                "image_id_b": self.baseline[-1]["image_id"], "relationship": "verified_derivative",
                "link_state": "CONFIRMED_MUST_LINK"}]}
        self.import_review(response, "external_review.json")
        after = read_json(self.audit/"audit_status.json")
        self.assertEqual(before["frozen_split_hash"], after["frozen_split_hash"])
        self.assertEqual(after["EXTERNAL_ELIGIBILITY"], "CONTAMINATED")
        self.assertEqual(len(assert_split_ready(self.audit, self.root)["test"]), 9)

    def test_external_seal_covers_generation_and_consumption(self):
        for operation in ("predictions", "embeddings", "morphology", "feature_distributions", "image_derived"):
            for purpose in ("generate", "consume"):
                with self.subTest(operation=operation, purpose=purpose), self.assertRaisesRegex(ProvenanceError, "SEALED"):
                    assert_external_access(self.audit, operation=operation, purpose=purpose, repo_root=self.root)
        for operation in ("provenance", "file_integrity", "acquisition_metadata"):
            self.assertEqual(len(assert_external_access(self.audit, operation=operation, repo_root=self.root)), 1)

    def release_fixture(self):
        self.freeze_fixture()
        self.checkpoint = self.root/"final.pt"; self.checkpoint.write_bytes(b"synthetic frozen final model")
        code = self.root/"inference.py"; code.write_text("# Synthetic inference identity\n")
        self.pipeline = self.root/"pipeline.json"
        write_json(self.pipeline, {"preprocessing": {}, "inference_code_files": [evidence_record(code, self.root)],
                                   "additional_model_files": []})
        sealed_manifest, sealed_status = self.root/"sealed_manifest.csv", self.root/"sealed_status.json"
        sealed_manifest.write_bytes((self.audit/"dataset_manifest.csv").read_bytes())
        sealed_status.write_bytes((self.audit/"audit_status.json").read_bytes())
        status = read_json(sealed_status)
        self.freeze_record = self.root/"model_freeze.json"
        write_json(self.freeze_record, {"status": "FROZEN", "frozen_split_hash": status["frozen_split_hash"],
            "checkpoint": evidence_record(self.checkpoint, self.root), "pipeline_config": evidence_record(self.pipeline, self.root),
            "sealed_audit_status": evidence_record(sealed_status, self.root), "sealed_manifest": evidence_record(sealed_manifest, self.root)})
        rows = read_csv(self.audit/"dataset_manifest.csv")
        for r in rows:
            if r["dataset"] == "External_Test_Culture_Free":
                r.update(split="external_test", evaluation_status="ready")
        write_csv(self.audit/"dataset_manifest.csv", rows)
        status["EXTERNAL_TEST"] = "RELEASED"
        status["artifact_sha256"]["dataset_manifest.csv"] = digest(self.audit/"dataset_manifest.csv")
        write_json(self.audit/"audit_status.json", status)

    def external_check(self, **kwargs):
        return assert_external_evaluation_ready(self.audit, self.freeze_record, checkpoint=self.checkpoint,
                    pipeline_config=self.pipeline, repo_root=self.root, **kwargs)

    def test_valid_external_release_and_changed_pipeline_rejection(self):
        self.release_fixture()
        self.assertEqual(len(self.external_check()), 1)
        self.assertEqual(len(self.external_check(purpose="consume")), 1)
        self.assertEqual(len(assert_split_ready(self.audit, self.root)["test"]), 9)
        with self.assertRaisesRegex(ProvenanceError, "after external release"):
            assert_training_ready(self.audit, self.root, experiment_record=self.experiment(), initialization_files=[])
        self.pipeline.write_text("modified")
        with self.assertRaisesRegex(ProvenanceError, "stale"):
            self.external_check()

    def test_changed_external_cohort_invalidates_release(self):
        self.release_fixture()
        rows = read_csv(self.audit/"dataset_manifest.csv")
        for r in rows:
            if r["dataset"] == "External_Test_Culture_Free":
                r["sha256"] = "changed"
        write_csv(self.audit/"dataset_manifest.csv", rows)
        status = read_json(self.audit/"audit_status.json")
        status["artifact_sha256"]["dataset_manifest.csv"] = digest(self.audit/"dataset_manifest.csv")
        write_json(self.audit/"audit_status.json", status)
        with self.assertRaisesRegex(ProvenanceError, "cohort changed"):
            self.external_check()

    def test_rebound_manifest_hash_cannot_fabricate_reviewed_provenance(self):
        self.freeze_fixture()
        rows = read_csv(self.audit/"dataset_manifest.csv")
        rows[0]["culture_id"] = "fabricated"
        write_csv(self.audit/"dataset_manifest.csv", rows)
        status = read_json(self.audit/"audit_status.json")
        status["artifact_sha256"]["dataset_manifest.csv"] = digest(self.audit/"dataset_manifest.csv")
        write_json(self.audit/"audit_status.json", status)
        with self.assertRaisesRegex(ProvenanceError, "differs from evidence"):
            assert_split_ready(self.audit, self.root)

    def test_tampered_artifact_is_rejected(self):
        self.freeze_fixture()
        (self.audit/"audit_report.md").write_text("tampered")
        with self.assertRaisesRegex(ProvenanceError, "Stale"):
            assert_split_ready(self.audit, self.root)

    def test_manual_ready_flags_cannot_manufacture_evidence(self):
        status = read_json(self.audit/"audit_status.json")
        status.update(SPLIT_STATUS="READY", blockers=[])
        write_json(self.audit/"audit_status.json", status)
        with self.assertRaises(ProvenanceError):
            assert_split_ready(self.audit, self.root)

    def test_legacy_trainers_reject_both_identity_schemas_before_importing_torch(self):
        for name in ("train_0min_fiveclass_raw.py", "run_0min_fiveclass_benchmark.py"):
            result = subprocess.run([sys.executable, str(REPO/"Code/Training/Model_Training_5_class"/name),
                "--manifest_csv", str(self.audit/"dataset_manifest.csv"), "--out_dir", str(self.root/"never")],
                text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("v1 manifest rejected", result.stderr)
            self.assertNotIn("ModuleNotFoundError", result.stderr)
            self.assertFalse((self.root/"never").exists())

    def test_guard_import_is_lightweight(self):
        code = (f"import sys; sys.path.insert(0, {str(REPO/'Code')!r}); "
                "import utils.identity_audit_gate; assert not any(m in sys.modules for m in ('torch','numpy','PIL','scipy'))")
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_unrelated_legacy_manifest_still_accepted(self):
        target = self.root/"unrelated.csv"
        write_csv(target, [{"image_path": "other.png", "class_id": "0"}])
        reject_v1_in_legacy_trainer(target)


if __name__ == "__main__":
    unittest.main()
