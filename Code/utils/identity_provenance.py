"""Dataset-only evidence, graph, policy and partition validation (no model imports).

Unknown relationships are assessed for impact, never used as confirmed unions.
All QC rows remain graph nodes. External nodes never bridge internal components.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict

SCHEMA = "bacterial_identity_provenance_v2"
PARTITIONS = ("train", "validation", "test")
CLASSES = set(range(9))
POLICY = "FULLY_CROSSED_WITHIN_PARTITION"
ROLES = {"HARD_MATCH", "BOUNDED_VARIATION", "CROSSED_BATCH", "INFORMATIONAL",
         "UNKNOWN_BLOCKING", "NOT_APPLICABLE"}
PROVENANCE_KEYS = ("culture_id", "sample_id", "preparation_id", "slide_id",
                   "acquisition_session_id", "field_id", "source_image_id")
REVIEW_KEYS = PROVENANCE_KEYS + ("source_series_id", "coordinate_frame_id", "instrument_id",
    "operator", "export_pipeline", "verified_acquisition_date", "verified_timezone", "optics",
    "verified_imaging_settings_json", "verified_preparation_protocol")


class ProvenanceError(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def content_hash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def stable_id(prefix, value):
    return prefix + "_" + content_hash(value)[:24]


def check(condition, message):
    if not condition:
        raise ProvenanceError(message)


def decoded(value, default=None):
    if isinstance(value, str):
        return json.loads(value) if value else default
    return value if value is not None else default


class UnionFind:
    def __init__(self, ids=()):
        self.parents = {i: i for i in ids}

    def find(self, key):
        self.parents.setdefault(key, key)
        while self.parents[key] != key:
            self.parents[key] = self.parents[self.parents[key]]
            key = self.parents[key]
        return key

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parents[max(a, b)] = min(a, b)


def initial_row(row):
    row = dict(row)
    row.update(schema_version=SCHEMA, dependency_component_id="", independence_status="UNRESOLVED",
               biological_independence_status="UNRESOLVED", acquisition_independence_status="UNRESOLVED",
               independence_evidence_ids="[]", assignment_unit_eligibility="UNRESOLVED",
               analysis_eligibility="UNRESOLVED", exclusion_reason="QC policy not frozen",
               exclusion_policy_version="", qc_evidence_ids="[]", provenance_evidence_ids="[]",
               verified_fields_json="[]", provenance_conflicts_json="[]",
               acquisition_stratum="", external_eligibility="UNRESOLVED",
               external_independence_status="UNRESOLVED", external_independence_evidence_ids="[]")
    for key in REVIEW_KEYS:
        row.setdefault(key, "")
    return row


def normalize_edge(edge):
    edge = dict(edge)
    edge.setdefault("edge_id", stable_id("edge", [*sorted([edge["image_id_a"], edge["image_id_b"]]),
                                                  edge["relationship"]]))
    state = edge.get("link_state")
    if not state:
        state = "CONFIRMED_MUST_LINK" if edge["relationship"] == "exact_sha256" else "CANDIDATE_LINK"
    check(state in {"CONFIRMED_MUST_LINK", "CANDIDATE_LINK", "REJECTED"}, "Invalid link state")
    check(edge["relationship"] != "exact_sha256" or state == "CONFIRMED_MUST_LINK",
          "Exact duplicates cannot be rejected or downgraded")
    if state != "CANDIDATE_LINK" and edge["relationship"] != "exact_sha256":
        check(bool(decoded(edge.get("adjudication_evidence_ids"), [])), "Adjudication evidence missing")
    edge.update(link_state=state)
    return edge


def geometry_edges(rows):
    """Only reviewed, same-slide coordinate frames justify geometric confirmation."""
    buckets = defaultdict(list)
    for r in rows:
        verified = set(decoded(r.get("verified_fields_json"), []))
        if {"slide_id", "coordinate_frame_id"} <= verified and r.get("coordinate_frame_id"):
            try:
                x, y, w, h = (float(r[k]) for k in
                    ("stage_x", "stage_y", "field_width_stage_units", "field_height_stage_units"))
                if all(math.isfinite(v) for v in (x, y, w, h)) and min(w, h) > 0:
                    buckets[(r["dataset"], r["slide_id"], r["coordinate_frame_id"])].append(
                        (x-w/2, x+w/2, y-h/2, y+h/2, r))
            except (ValueError, KeyError, TypeError):
                continue
    result = []
    for group in buckets.values():
        group.sort(key=lambda t: (t[0], t[4]["image_id"]))
        for i, a in enumerate(group):
            for b in group[i+1:]:
                if b[0] >= a[1]:
                    break
                if min(a[3], b[3]) <= max(a[2], b[2]):
                    continue
                left, right = a[4], b[4]
                result.append(normalize_edge({"image_id_a": left["image_id"], "image_id_b": right["image_id"],
                    "dataset_a": left["dataset"], "dataset_b": right["dataset"],
                    "relationship": "verified_stage_overlap", "link_state": "CONFIRMED_MUST_LINK",
                    "adjudication_evidence_ids": sorted(set(decoded(left["provenance_evidence_ids"], []) +
                                                               decoded(right["provenance_evidence_ids"], [])))}))
    return result


def build_graph(rows, ledger, unit_reviews=()):
    by_id = {r["image_id"]: r for r in rows}
    check(len(by_id) == len(rows), "Duplicate image IDs")
    internal = {i for i, r in by_id.items() if r["dataset"] == "Dataset_B"}
    external = set(by_id) - internal
    edges = {e["edge_id"]: e for e in map(normalize_edge, ledger)}
    for edge in geometry_edges(rows):
        edges.setdefault(edge["edge_id"], edge)
    uf = UnionFind(internal)
    nodes = {i: r["dataset"] for i, r in by_id.items()}
    for e in edges.values():
        for suffix in ("a", "b"):
            i, dataset = e["image_id_"+suffix], e["dataset_"+suffix]
            check(i not in nodes or nodes[i] == dataset, "Conflicting ledger endpoint")
            check(i in nodes or dataset == "Legacy_training", "Unknown ledger endpoint")
            nodes[i] = dataset
    relation_buckets = defaultdict(list)
    for r in rows:
        verified = set(decoded(r.get("verified_fields_json"), []))
        for key in (*PROVENANCE_KEYS, "sha256"):
            if r.get(key) and (key == "sha256" or key in verified):
                relation_buckets[(key, r[key])].append(r)
    contradictions = []
    external_confirmed, external_candidates = set(), set()
    for (key, value), group in relation_buckets.items():
        inside = [r for r in group if r["image_id"] in internal]
        outside = [r for r in group if r["image_id"] in external]
        if key == "sha256" and len({int(r["class_id"]) for r in inside}) > 1 and any(
                r.get("analysis_eligibility") != "QC_EXCLUDED" for r in inside):
            contradictions.append("Byte-identical internal images have contradictory identities")
        for r in inside[1:]:
            uf.union(inside[0]["image_id"], r["image_id"])
        if inside:
            external_confirmed.update(r["image_id"] for r in outside)
    for e in edges.values():
        a, b = e["image_id_a"], e["image_id_b"]
        if a in external or b in external:
            ext = a if a in external else b
            other = b if a in external else a
            if other not in external:
                if e["link_state"] == "CONFIRMED_MUST_LINK":
                    external_confirmed.add(ext)
                elif e["link_state"] == "CANDIDATE_LINK":
                    external_candidates.add(ext)
            continue  # External nodes never bridge internal assignments.
        if e["link_state"] == "CONFIRMED_MUST_LINK":
            uf.union(a, b)  # Legacy aliases can prove internal-to-internal identity.
    members = defaultdict(list)
    for i in sorted(internal):
        members[uf.find(i)].append(i)
    components, component_of = {}, {}
    review_map = defaultdict(list)
    for review in unit_reviews:
        review_map[tuple(sorted(review["image_ids"]))].append(review)
    for ids in members.values():
        cid = stable_id("dependency", ids)
        for i in ids:
            component_of[i] = cid
        reviews = review_map[tuple(ids)]
        bio = {r["biological_independence_status"] for r in reviews}
        acq = {r["acquisition_independence_status"] for r in reviews}
        evidence = sorted({e for r in reviews for e in r.get("independence_evidence_ids", [])})
        bs = next(iter(bio)) if len(bio) == 1 else "UNRESOLVED"
        ass = next(iter(acq)) if len(acq) == 1 else "UNRESOLVED"
        check(bs in {"VERIFIED", "UNRESOLVED", "FAILED"} and ass in {"VERIFIED", "UNRESOLVED", "FAILED"},
              "Invalid independence assessment")
        eligible = [by_id[i] for i in ids if by_id[i].get("analysis_eligibility") == "ELIGIBLE"]
        documented = all(set(PROVENANCE_KEYS) <= set(decoded(r.get("verified_fields_json"), []))
                         and all(r.get(k) for k in PROVENANCE_KEYS)
                         and not decoded(r.get("provenance_conflicts_json"), []) for r in eligible)
        verified = bool(eligible) and bs == ass == "VERIFIED" and bool(evidence) and documented
        eligibility = "ELIGIBLE" if verified else "INELIGIBLE" if "FAILED" in (bs, ass) or not eligible else "UNRESOLVED"
        components[cid] = {"dependency_component_id": cid, "image_ids": ids,
            "class_ids": sorted({int(r["class_id"]) for r in eligible}),
            "biological_independence_status": bs, "acquisition_independence_status": ass,
            "independence_evidence_ids": evidence, "assignment_unit_eligibility": eligibility,
            "independence_status": "VERIFIED_INDEPENDENT" if verified else "LINKED" if len(ids)>1 else "UNRESOLVED"}
    # A separate *possible* graph is used solely for impact/exposure, never counts.
    possible = UnionFind(nodes)
    for i in nodes:
        if i not in external:
            possible.union(i, uf.find(i))
    for e in edges.values():
        if e["link_state"] == "CANDIDATE_LINK" and not ({e["image_id_a"], e["image_id_b"]} & external):
            possible.union(e["image_id_a"], e["image_id_b"])
    possible_units = defaultdict(set)
    possible_analysis = set()
    for i in internal:
        possible_units[possible.find(i)].add(component_of[i])
        if by_id[i].get("analysis_eligibility") != "QC_EXCLUDED":
            possible_analysis.add(possible.find(i))
    impacts = []
    for e in edges.values():
        domains = []
        a, b = e["image_id_a"], e["image_id_b"]
        if e["link_state"] == "CANDIDATE_LINK":
            if a in external or b in external:
                domains.append("external_independence")
            elif (uf.find(a) != uf.find(b) and len(possible_units[possible.find(a)]) > 1
                  and possible.find(a) in possible_analysis):
                domains.append("internal_independence")
            if (nodes[a] == "Legacy_training" or nodes[b] == "Legacy_training") and uf.find(a) != uf.find(b):
                domains.append("experiment_exposure")
            if a not in external and b not in external:
                domains.extend(decoded(e.get("other_hard_gate_domains"), []))
        impacts.append({**e, "blocking_domains": sorted(set(domains)),
            "impact_reason": "; ".join(sorted(set(domains))) if domains else "No unresolved hard-gate impact",
            "assessment_version": SCHEMA})
    legacy = [i for i, dataset in nodes.items() if dataset == "Legacy_training"]
    confirmed_exposure = {uf.find(i) for i in legacy}
    possible_exposure = {possible.find(i) for i in legacy}
    for i in internal:
        row = by_id[i]
        group = components[component_of[i]]
        for key in ("dependency_component_id", "independence_status", "biological_independence_status",
                    "acquisition_independence_status", "assignment_unit_eligibility"):
            row[key] = group[key]
        row["independence_evidence_ids"] = canonical(group["independence_evidence_ids"])
        row["legacy_relationship_status"] = ("CONFIRMED" if uf.find(i) in confirmed_exposure else
            "POSSIBLE" if possible.find(i) in possible_exposure else "NO_DETECTED_LINK")
    for i in external:
        by_id[i]["dependency_component_id"] = ""
        by_id[i]["external_eligibility"] = "CONTAMINATED" if i in external_confirmed else "UNRESOLVED"
        if i not in external_confirmed | external_candidates and by_id[i].get("external_independence_status") == "VERIFIED" \
                and decoded(by_id[i].get("external_independence_evidence_ids"), []):
            by_id[i]["external_eligibility"] = "CLEAN"
    return {"components": components, "edges": impacts, "contradictions": contradictions}


def factor_value(row, source):
    if source.startswith("conditions."):
        return decoded(row.get("acquisition_conditions_json"), {}).get(source[11:])
    if source.startswith("verified_settings."):
        return decoded(row.get("verified_imaging_settings_json"), {}).get(source[18:])
    return row.get(source)


def policy_errors(schema, qc):
    errors = []
    for name, policy in (("Acquisition schema", schema), ("QC policy", qc)):
        if policy.get("status") != "FROZEN" or not policy.get("version") or not policy.get("evidence_ids"):
            errors.append(name + " requires frozen, reviewed evidence")
    if schema.get("acquisition_design_policy") != POLICY:
        errors.append("Unsupported acquisition policy")
    if not schema.get("factors"):
        errors.append("No acquisition factors defined")
    sources = [f.get("source") for f in schema.get("factors", [])]
    if len(sources) != len(set(sources)) or None in sources:
        errors.append("Duplicate/missing acquisition factor source")
    if not set(schema.get("required_sources", [])) <= set(sources):
        errors.append("Required acquisition factors were omitted")
    for f in schema.get("factors", []):
        role = f.get("role")
        if role not in ROLES or role == "UNKNOWN_BLOCKING":
            errors.append("Unresolved acquisition factor: " + str(f.get("source")))
        if not f.get("justification"):
            errors.append("Acquisition factor justification missing: " + str(f.get("source")))
        if f.get("normalization", "exact") not in {"exact", "string", "numeric"}:
            errors.append("Unsupported factor normalization")
        if role == "CROSSED_BATCH":
            req = f.get("requirement", {})
            if (req.get("scope") not in {"dataset", "partition", "both"}
                    or type(req.get("no_class_unique_level")) is not bool
                    or type(req.get("minimum_identity_support_per_level")) is not int
                    or not 1 <= req["minimum_identity_support_per_level"] <= 9):
                errors.append("Incomplete crossing requirement: " + f["source"])
        if role == "BOUNDED_VARIATION":
            intervals = f.get("intervals", [])
            if not intervals or any(not isinstance(v.get("label"), str) or
                                    not isinstance(v.get("min"), (float, int)) or
                                    not isinstance(v.get("max"), (float, int)) or v["min"] >= v["max"] for v in intervals):
                errors.append("Bounded factors require named half-open intervals")
    batch_names = {f["source"] for f in schema.get("factors", []) if f["role"] == "CROSSED_BATCH"}
    for joint in schema.get("joint_batch_factors", []):
        req = joint.get("requirement", {})
        if (not joint.get("name") or joint["name"] in batch_names or not joint.get("sources")
                or not set(joint["sources"]) <= batch_names or req.get("scope") not in {"dataset", "partition", "both"}
                or type(req.get("no_class_unique_level")) is not bool
                or type(req.get("minimum_identity_support_per_level")) is not int
                or not 1 <= req["minimum_identity_support_per_level"] <= 9):
            errors.append("Invalid joint batch crossing requirement")
    if qc.get("rules") != [{"id": "primary_readability", "kind": "integrity_decode", "on_failure": "QC_EXCLUDED"}]:
        errors.append("Unsupported QC rules; add and test a prespecified rule implementation before use")
    return errors


def acquisition_levels(row, schema):
    strata, batches = {}, {}
    for factor in schema["factors"]:
        role, source = factor["role"], factor["source"]
        if role in {"INFORMATIONAL", "NOT_APPLICABLE"}:
            continue
        value = factor_value(row, source)
        check(value is not None and value != "", "Missing required acquisition factor: " + source)
        normalization = factor.get("normalization", "exact")
        if normalization == "string":
            value = str(value)
        elif normalization == "numeric":
            value = float(value)
            check(math.isfinite(value), "Nonfinite acquisition value")
        if role == "BOUNDED_VARIATION":
            matches = [v["label"] for v in factor["intervals"] if v["min"] <= float(value) < v["max"]]
            check(len(matches) == 1, "Value outside/ambiguous bounded variation: " + source)
            strata[source] = matches[0]
        elif role == "HARD_MATCH":
            strata[source] = value
        elif role == "CROSSED_BATCH":
            batches[source] = canonical(value)
        else:
            raise ProvenanceError("Unresolved acquisition role")
    for joint in schema.get("joint_batch_factors", []):
        check(all(source in batches for source in joint["sources"]), "Joint check requires crossed batch sources")
        batches[joint["name"]] = canonical([batches[source] for source in joint["sources"]])
    return stable_id("stratum", strata), batches


def crossing_rules(schema):
    return {f["source"]: f["requirement"] for f in schema["factors"] if f["role"] == "CROSSED_BATCH"} | {
        f["name"]: f["requirement"] for f in schema.get("joint_batch_factors", [])}


def prepare_problem(rows, graph, schema, qc, mapping):
    blockers = policy_errors(schema, qc) + graph["contradictions"]
    internal = [r for r in rows if r["dataset"] == "Dataset_B"]
    observed_sources = {"conditions."+k for r in internal for k in decoded(r.get("acquisition_conditions_json"), {})}
    observed_sources.update({"verified_acquisition_date", "acquisition_session_id", "operator", "instrument_id", "export_pipeline"})
    if not observed_sources <= {f.get("source") for f in schema.get("factors", [])}:
        blockers.append("Observed acquisition factors lack a prespecified classification")
    for r in internal:
        state = r.get("analysis_eligibility")
        if state not in {"ELIGIBLE", "QC_EXCLUDED"}:
            blockers.append("Unresolved QC: " + r["image_id"])
        elif r.get("exclusion_policy_version") != qc.get("version") or not decoded(r.get("qc_evidence_ids"), []):
            blockers.append("Missing/stale QC evidence: " + r["image_id"])
        if state == "QC_EXCLUDED" and not r.get("exclusion_reason"):
            blockers.append("QC exclusion lacks reason: " + r["image_id"])
        if state == "ELIGIBLE" and r.get("assignment_unit_eligibility") != "ELIGIBLE":
            blockers.append("Assignment independence unresolved/failed: " + r["dependency_component_id"])
    for e in graph["edges"]:
        if any(d not in {"experiment_exposure", "external_independence"} for d in e["blocking_domains"]):
            blockers.append("Impactful candidate: " + e["edge_id"])
    classes = {int(c["class_id"]): c for c in mapping["classes"]}
    if set(classes) != CLASSES or len(mapping["classes"]) != 9:
        blockers.append("Invalid nine-class identity mapping")
    if classes.get(6, {}).get("identity_status") != "verified" or classes.get(6, {}).get("species") != "Staphylococcus aureus":
        blockers.append("MRSA organism verification unresolved")
    eligible = [r for r in internal if r.get("analysis_eligibility") == "ELIGIBLE"]
    units = {}
    strata, batches = defaultdict(lambda: defaultdict(set)), defaultdict(lambda: defaultdict(set))
    for row in eligible:
        cid, cls = row["dependency_component_id"], int(row["class_id"])
        check(cls in CLASSES, "Unknown class")
        unit = units.setdefault(cid, {"A": [0]*9, "N": [0]*9})
        unit["A"][cls] = 1
        unit["N"][cls] += 1
        if not policy_errors(schema, qc):
            try:
                stratum, levels = acquisition_levels(row, schema)
                row["acquisition_stratum"] = stratum
                strata[stratum][cls].add(cid)
                for factor, level in levels.items():
                    batches[(factor, level)][cls].add(cid)
            except (ProvenanceError, ValueError, TypeError) as exc:
                blockers.append(str(exc) + ": " + row["image_id"])
    problem = {"units": dict(sorted(units.items())),
        "strata": {s: {c: sorted(v) for c, v in classes.items()} for s, classes in sorted(strata.items())},
        "batches": [{"factor": f, "level": l, "classes": {c: sorted(v) for c, v in classes.items()}}
                    for (f, l), classes in sorted(batches.items())],
        "crossing_rules": crossing_rules(schema) if not policy_errors(schema, qc) else {},
        "eligible_image_ids": sorted(r["image_id"] for r in eligible),
        "schema_hash": content_hash(schema), "qc_policy_hash": content_hash(qc),
        "policy": schema.get("acquisition_design_policy")}
    return problem, sorted(set(blockers))


def analytical_failures(problem):
    failures = []
    for c in range(9):
        if sum(u["A"][c] for u in problem["units"].values()) < 3:
            failures.append(f"Class {c}: fewer than 3 verified-independent eligible assignment units")
    for stratum, classes in problem["strata"].items():
        if set(map(int, classes)) != CLASSES:
            failures.append(f"{POLICY}: predeclared eligible stratum {stratum} lacks all nine identities")
    for batch in problem["batches"]:
        req = problem["crossing_rules"][batch["factor"]]
        minimum = max(req["minimum_identity_support_per_level"], 2 if req["no_class_unique_level"] else 1)
        if len(batch["classes"]) < minimum:
            failures.append("Insufficient identity crossing: " + batch["factor"] + "=" + batch["level"])
    return failures


def validate_assignment(rows, graph, schema, qc, mapping, assignment, frozen_population=None):
    """Recompute incidence and hard constraints independently of MILP output."""
    problem, blockers = prepare_problem(rows, graph, schema, qc, mapping)
    check(not blockers, "BLOCKED: " + "; ".join(blockers[:8]))
    check(not analytical_failures(problem), "; ".join(analytical_failures(problem)))
    check(set(assignment) == set(problem["units"]), "Assignment population mismatch")
    check(all(p in PARTITIONS for p in assignment.values()), "Invalid partition")
    if frozen_population is not None:
        check(content_hash(problem) == frozen_population, "Eligible population/rules changed after assessment")
    for p in PARTITIONS:
        check({c for u, value in problem["units"].items() if assignment[u] == p
               for c, present in enumerate(value["A"]) if present} == CLASSES, "Every partition needs all nine identities")
    for stratum, classes in problem["strata"].items():
        support = defaultdict(set)
        for c, units in classes.items():
            for u in units:
                support[assignment[u]].add(int(c))
        check(all(v == CLASSES for v in support.values()), POLICY + ": stratum " + stratum)
    for batch in problem["batches"]:
        req = problem["crossing_rules"][batch["factor"]]
        minimum = max(req["minimum_identity_support_per_level"], 2 if req["no_class_unique_level"] else 1)
        if req["scope"] in {"partition", "both"}:
            support = defaultdict(set)
            for c, units in batch["classes"].items():
                for u in units:
                    support[assignment[u]].add(int(c))
            check(all(len(v) >= minimum for v in support.values()), "Partition batch crossing failed")
    return problem


def class_report(rows, graph, assignment=None):
    result = []
    for c in range(9):
        group = [r for r in rows if r["dataset"] == "Dataset_B" and int(r["class_id"]) == c]
        eligible = [r for r in group if r.get("analysis_eligibility") == "ELIGIBLE"]
        def verified_ids(key, subset):
            return {r[key] for r in subset if r.get(key) and key in decoded(r.get("verified_fields_json"), [])}
        entry = {"class_id": c, "images": len(group), "analysis_eligible_images": len(eligible),
            "qc_excluded_images": sum(r.get("analysis_eligibility") == "QC_EXCLUDED" for r in group),
            "verified_cultures": len(verified_ids("culture_id", group)),
            "verified_slides": len(verified_ids("slide_id", group)),
            "verified_sessions": len(verified_ids("acquisition_session_id", group)),
            "eligible_verified_cultures": len(verified_ids("culture_id", eligible)),
            "confirmed_components": len({r["dependency_component_id"] for r in group}),
            "verified_independent_eligible_units": len({r["dependency_component_id"] for r in eligible
                if r["assignment_unit_eligibility"] == "ELIGIBLE"}), "held_out": {}}
        if assignment:
            for p in PARTITIONS:
                subset = [r for r in eligible if assignment[r["dependency_component_id"]] == p]
                entry["held_out"][p] = {"cultures": len(verified_ids("culture_id", subset)),
                    "units": len({r["dependency_component_id"] for r in subset}), "images": len(subset)}
        result.append(entry)
    return result
