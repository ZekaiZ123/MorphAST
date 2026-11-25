"""Discrete, lexicographic partition selection using scipy.optimize.milp/HiGHS."""
from __future__ import annotations

import time

from .identity_provenance import PARTITIONS, ProvenanceError

DEFAULT_PROTOCOL = {"version": "milp-lex-v1", "target_percent": [70, 15, 15],
    "time_limit_seconds": 300, "mip_rel_gap": 0.0, "presolve": True,
    "objective_tolerance": 1e-8, "integrality_tolerance": 1e-6,
    "tie_break": "sorted_component_id_then_train_validation_test_sequential_feasibility"}


def solve(problem, validator, protocol=None):
    """A solution is FEASIBLE only if the caller's independent validator accepts it.

    Optimization completion is separate from scientific feasibility. No generic
    continuous optimizer or rounding of fractional assignments is permitted.
    """
    protocol = dict(DEFAULT_PROTOCOL if protocol is None else protocol)
    if set(protocol) != set(DEFAULT_PROTOCOL) or any(protocol[k] != DEFAULT_PROTOCOL[k] for k in
            ("version", "target_percent", "mip_rel_gap", "tie_break")):
        raise ProvenanceError("Unsupported solver protocol; explicitly version changes")
    if (not 0 < protocol["time_limit_seconds"] <= 86400
            or not 0 < protocol["objective_tolerance"] <= 1e-7
            or not 0 < protocol["integrality_tolerance"] <= 1e-5
            or type(protocol["presolve"]) is not bool):
        raise ProvenanceError("Invalid solver parameters")
    try:
        import numpy as np
        import scipy
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix
        try:
            from scipy.optimize._highspy._core import HIGHS_VERSION_MAJOR, HIGHS_VERSION_MINOR, HIGHS_VERSION_PATCH
            highs_version = f"{HIGHS_VERSION_MAJOR}.{HIGHS_VERSION_MINOR}.{HIGHS_VERSION_PATCH}"
        except ImportError:
            # scipy <=1.14 embeds HiGHS without public version constants. Bind
            # its exact compiled module bytes instead of guessing the version.
            import hashlib
            from pathlib import Path
            import scipy.optimize._highs._highs_wrapper as wrapper
            highs_version = "embedded-binary-sha256:" + hashlib.sha256(Path(wrapper.__file__).read_bytes()).hexdigest()
    except ImportError as exc:
        return {"state": "UNKNOWN", "optimization_status": "NOT_RUN", "reason": str(exc),
                "solver_protocol": protocol, "assignment": None}
    solver = {"scipy_version": scipy.__version__, "highs_version": highs_version,
              "numpy_version": np.__version__, "protocol": protocol}
    ids = sorted(problem["units"])
    index = {u: i for i, u in enumerate(ids)}
    lower, upper, integer = [], [], []
    constraints = []

    def variable(lo=0., hi=1., binary=True):
        i = len(lower)
        lower.append(lo); upper.append(hi); integer.append(int(binary))
        return i

    def constraint(coefficients, lo=-np.inf, hi=np.inf):
        constraints.append((dict(coefficients), lo, hi))

    x = {(u, p): variable() for u in ids for p in range(3)}
    for u in ids:
        constraint({x[u, p]: 1 for p in range(3)}, 1, 1)
    for c in range(9):
        for p in range(3):
            constraint({x[u, p]: 1 for u in ids if problem["units"][u]["A"][c]}, 1)

    def support_constraints(classes, minimum):
        # h[c,p] is exactly the presence of class c at this level/stratum.
        for p in range(3):
            level = variable()
            support = []
            for c, units in sorted(classes.items(), key=lambda kv: int(kv[0])):
                present = variable()
                support.append(present)
                for u in units:
                    constraint({x[u, p]: 1, present: -1}, hi=0)
                constraint({present: 1, **{x[u, p]: -1 for u in units}}, hi=0)
                constraint({present: 1, level: -1}, hi=0)
            constraint({level: 1, **{v: -1 for v in support}}, hi=0)
            constraint({**{v: 1 for v in support}, level: -minimum}, lo=0)

    for classes in problem["strata"].values():
        support_constraints(classes, 9)
    for batch in problem["batches"]:
        req = problem["crossing_rules"][batch["factor"]]
        if req["scope"] in {"partition", "both"}:
            support_constraints(batch["classes"], max(req["minimum_identity_support_per_level"],
                                                       2 if req["no_class_unique_level"] else 1))
    objectives = []
    for metric in ("A", "N"):
        objective = {}
        for c in range(9):
            total = sum(problem["units"][u][metric][c] for u in ids)
            if not total:
                raise ProvenanceError("Analytical class-coverage check must precede solver")
            for p, percent in enumerate(protocol["target_percent"]):
                deviation = variable(0, np.inf, False)
                objective[deviation] = 1 / 9
                coeff = {x[u, p]: problem["units"][u][metric][c] / total for u in ids}
                constraint({**coeff, deviation: -1}, hi=percent/100)
                constraint({**{k: -v for k, v in coeff.items()}, deviation: -1}, hi=-percent/100)
        objectives.append(objective)
    start = time.monotonic()
    assignment, trace = None, []

    def run(objective):
        nonlocal assignment
        remaining = protocol["time_limit_seconds"] - (time.monotonic() - start)
        if remaining <= 0:
            return None
        rr, cc, vv = [], [], []
        for r, (co, lo, hi) in enumerate(constraints):
            for col, val in co.items():
                if val:
                    rr.append(r); cc.append(col); vv.append(val)
        matrix = coo_matrix((vv, (rr, cc)), shape=(len(constraints), len(lower))).tocsc()
        cost = np.zeros(len(lower))
        for i, value in objective.items():
            cost[i] = value
        result = milp(cost, integrality=np.asarray(integer), bounds=Bounds(lower, upper),
            constraints=LinearConstraint(matrix, [t[1] for t in constraints], [t[2] for t in constraints]),
            options={"time_limit": remaining, "mip_rel_gap": protocol["mip_rel_gap"], "presolve": protocol["presolve"]})
        trace.append({"status": int(result.status), "message": str(result.message)})
        result.validated_assignment = False
        if result.x is not None:
            values = np.array([[result.x[x[u, p]] for p in range(3)] for u in ids])
            if (np.all(np.isfinite(values)) and
                np.all(np.abs(values-np.rint(values)) <= protocol["integrality_tolerance"]) and
                np.all(np.sum(np.rint(values), axis=1) == 1)):
                candidate = {u: PARTITIONS[int(np.argmax(values[i]))] for i, u in enumerate(ids)}
                try:
                    validator(candidate)
                    assignment = candidate
                    result.validated_assignment = True
                except (ProvenanceError, ValueError, KeyError, TypeError):
                    trace[-1]["validator"] = "REJECTED"
        return result

    def answer(complete=False, reason=None, infeasible=False):
        return {"state": "FEASIBLE" if assignment is not None else "INFEASIBLE" if infeasible else "UNKNOWN",
            "optimization_status": "COMPLETE" if complete else "INCOMPLETE", "assignment": assignment,
            "solver": solver, "trace": trace, "reason": reason}

    try:
        for stage, objective in enumerate(objectives):
            result = run(objective)
            if result is None or result.status != 0 or not result.validated_assignment:
                return answer(reason="Balance optimization incomplete", infeasible=bool(
                    stage == 0 and result is not None and result.status == 2 and assignment is None))
            optimum = float(result.fun)
            constraint(objective, hi=optimum+protocol["objective_tolerance"])
        # Sequential feasibility produces a canonical assignment even when many
        # partitions share the same lexicographic objective values.
        for u in ids:
            chosen = False
            for p in range(3):
                constraint({x[u, p]: 1}, 1, 1)
                result = run({})
                if result is not None and result.status == 0 and result.validated_assignment and assignment[u] == PARTITIONS[p]:
                    chosen = True
                    break
                constraints.pop()
                if result is None or result.status != 2:
                    return answer(reason="Deterministic tie resolution incomplete")
            if not chosen:
                return answer(reason="No tie-breaking candidate validated")
        validator(assignment)
        return answer(complete=True)
    except (ValueError, RuntimeError, TypeError) as exc:
        return answer(reason="Solver error: " + str(exc))
