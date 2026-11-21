"""Outcome-blind grouped splits for a matched-population sensitivity experiment."""

from itertools import combinations

import numpy as np

from audit_grouped_cv_feasibility import component_ids, known
from validation import ROLES, audit_partition, sha256, validate_manifest

PROTOCOL = "matched_field_preparation_slide_sensitivity_v1"
TASKS = {
    "field_baseline": ["legacy_field_group_id", "original_sha256"],
    "preparation_sensitivity": ["preparation_id", "slide_id", "original_sha256"],
}
IDENTIFIERS = ("legacy_field_group_id", "original_sha256", "preparation_id",
               "slide_id", "acquisition_session_id", "culture_id")
LIMITATIONS = [
    "EXPLORATORY SENSITIVITY ONLY: preparation/slide folds allow original cross-time field IDs to cross partitions; their physical meaning is unconfirmed.",
    "The comparison does not establish culture, session, or biological-replicate independence; SCID is excluded.",
    "Preparations often correspond to different time points within a session, confounding preparation and time effects.",
    "Composite metadata associations are not checksum matching to the provider's reorganized files.",
    "All conflicting-label exact copies and images without verified metadata are excluded; labels are not corrected.",
    "Same-label exact copies remain linked; recompressed/near-duplicate content and unrecorded repeated physical fields are not ruled out.",
    "Both arms use the same images but different partitions and potentially different training sizes; differences do not isolate batch bias.",
    "Fold SDs are descriptive, not confidence intervals or independent biological replicates. Fold numbers are not paired experimental units across arms.",
]


def grouped_manifest(df, task):
    result = df.copy()
    result["group_id"] = component_ids(result, TASKS[task])
    validate_manifest(result)
    if result.groupby("group_id").class_id.nunique().max() != 1:
        raise ValueError("Splitter requires class-pure grouping components")
    return result


def group_sizes(df):
    return df.groupby("group_id", sort=True).agg(class_id=("class_id", "first"), n=("image_path", "size"))


def outer_assignment(df, n_folds, seed):
    """Largest components first; minimize per-class image counts, then total size."""
    rng = np.random.default_rng(seed)
    assigned, totals = {}, np.zeros(n_folds, dtype=int)
    for _, subset in group_sizes(df).groupby("class_id", sort=True):
        if len(subset) < n_folds:
            raise ValueError("Insufficient components per class for requested outer folds")
        subset = subset.assign(tie=rng.random(len(subset))).sort_values(["n", "tie"], ascending=[False, True])
        counts, numbers = np.zeros(n_folds, dtype=int), np.zeros(n_folds, dtype=int)
        order = rng.permutation(n_folds).tolist()
        for group, row in subset.iterrows():
            choices = np.flatnonzero(numbers == 0).tolist() or list(range(n_folds))
            fold = min(choices, key=lambda f: (counts[f], totals[f], order.index(f)))
            assigned[group] = fold + 1
            counts[fold] += int(row.n)
            totals[fold] += int(row.n)
            numbers[fold] += 1
    return df.group_id.map(assigned).to_numpy(dtype=int)


def inner_validation_groups(development, fraction, seed):
    """Approach a per-class image target, keeping at least one training component."""
    rng, chosen = np.random.default_rng(seed), set()
    for _, subset in group_sizes(development).groupby("class_id", sort=True):
        if len(subset) < 2:
            raise ValueError("Inner selection needs >=2 development components per class")
        sizes = subset.n.to_dict()
        ties = dict(zip(subset.index, rng.random(len(subset))))
        target, total = float(subset.n.sum()) * fraction, 0
        remaining, selected = set(subset.index), set()
        while len(remaining) > 1:
            best = min(remaining, key=lambda g: (abs(total + sizes[g] - target), ties[g], g))
            updated = total + sizes[best]
            if selected and abs(updated - target) >= abs(total - target):
                break
            selected.add(best)
            remaining.remove(best)
            total = updated
        chosen.update(selected)
    return chosen


def audit_identifiers(df, roles, required_disjoint):
    roles = np.asarray(roles)
    result = audit_partition(df, roles)
    overlaps = []
    for column in IDENTIFIERS:
        if not known(df[column]).all():
            raise ValueError(f"Missing identifier in frozen analysis: {column}")
        for left, right in combinations(ROLES, 2):
            a, b = df.loc[roles == left, column], df.loc[roles == right, column]
            shared = set(a) & set(b)
            if shared and column in required_disjoint:
                raise ValueError(f"Forbidden {column} overlap between {left} and {right}")
            overlaps.append({"identifier": column, "partition_a": left, "partition_b": right,
                             "n_shared_identifiers": len(shared),
                             "n_images_a_with_shared_identifier": int(a.isin(shared).sum()),
                             "n_images_b_with_shared_identifier": int(b.isin(shared).sum())})
    result["identifier_overlap"] = overlaps
    return result


def make_partitions(df, task, n_folds=5, seed=42, validation_fraction=.2):
    validate_manifest(df)
    if n_folds < 3 or not 0 < validation_fraction < 1:
        raise ValueError("Require >=3 outer folds and validation fraction between zero and one")
    if df.groupby("group_id").class_id.nunique().max() != 1:
        raise ValueError("Splitter requires class-pure grouping components")
    outer = outer_assignment(df, n_folds, seed)
    result = []
    for fold in range(1, n_folds + 1):
        groups = inner_validation_groups(df.loc[outer != fold], validation_fraction, seed + 1000 + fold)
        roles = np.full(len(df), "train", dtype=object)
        roles[df.group_id.isin(groups)] = "validation"
        roles[outer == fold] = "test"
        audit = audit_identifiers(df, roles, TASKS[task])
        audit.update(fold=fold, inner_seed=seed + 1000 + fold,
                     requested_validation_fraction_of_development=validation_fraction)
        result.append((roles, audit))
    if not np.all(np.sum([roles == "test" for roles, _ in result], axis=0) == 1):
        raise ValueError("Outer folds do not cover every image exactly once")
    return result


def validate_config(config):
    if config["protocol"] != PROTOCOL or config.get("allow_legacy_field_overlap") is not True:
        raise ValueError("Sensitivity protocol requires explicit legacy-field overlap acknowledgement")
    if set(config["tasks"]) != set(TASKS) or config.get("scid_used") is not False:
        raise ValueError("Unexpected sensitivity arms or SCID use")
    for task, constraints in TASKS.items():
        if config["tasks"][task]["required_disjoint"] != constraints:
            raise ValueError(f"Changed required grouping constraints: {task}")


def verify_frozen_inputs(run_dir, config):
    validate_config(config)
    for name, digest in config["artifacts_sha256"].items():
        if sha256(run_dir / name) != digest:
            raise ValueError(f"Frozen sensitivity artifact changed: {name}")
    for item in config["evidence"]:
        if sha256(run_dir / item["frozen_path"]) != item["sha256"]:
            raise ValueError(f"Frozen evidence changed: {item['frozen_path']}")
    for relative, digest in config["source_sha256"].items():
        if sha256(run_dir / "source" / relative) != digest:
            raise ValueError(f"Frozen source changed: {relative}")


def audit_frozen_partition(df, split, task):
    if (split.row_id.tolist() != list(range(len(df)))
            or not split.image_path.equals(df.image_path)
            or not split.class_id.equals(df.class_id)
            or not split.group_id.equals(df.group_id)):
        raise ValueError("Split identities, labels or grouping do not match frozen manifest")
    if not component_ids(df, TASKS[task]).equals(df.group_id):
        raise ValueError("Frozen grouping does not match required connected components")
    return audit_identifiers(df, split.role.to_numpy(), TASKS[task])
