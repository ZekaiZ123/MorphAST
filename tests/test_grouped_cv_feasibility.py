"""Fail-closed grouping preflight for the experimental-independence rebuttal."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code/Training/Rebuttal"))
from audit_grouped_cv_feasibility import (
    clean_population, component_ids, export_audit, grouping_tables, known, verify_images,
)


def population():
    rows, hashes = [], []
    for label in range(2):
        for preparation in range(6):
            path = f"/class_{label}/image_{preparation}.png"
            rows.append({"image_path": path, "group_id": f"field_{label}_{preparation}",
                         "class_id": label, "class_name": f"class_{label}",
                         "culture_id": f"culture_{label}", "preparation_id": f"prep_{label}_{preparation}",
                         "slide_id": f"slide_{label}_{preparation}", "acquisition_session_id": f"session_{preparation}",
                         "timepoint_min": preparation, "mapping_status": "unique_composite_and_dimensions"})
            hashes.append({"image_path": path, "original_sha256": hashlib.sha256(path.encode()).hexdigest(),
                           "hash_status": "ok"})
    return pd.DataFrame(rows), pd.DataFrame(hashes)


class CleaningTests(unittest.TestCase):
    def test_conflicts_excluded_before_unknown_metadata(self):
        metadata, hashes = population()
        hashes.loc[6, "original_sha256"] = hashes.loc[0, "original_sha256"]
        metadata.loc[6, "mapping_status"] = "no_exact_composite_match"
        clean, ledger = clean_population(metadata, hashes)
        self.assertEqual(len(clean), 10)
        self.assertNotIn(metadata.loc[0, "image_path"], set(clean.image_path))
        self.assertEqual(ledger.conflicting_label_content.sum(), 2)
        self.assertEqual(ledger.exclusion_reason.eq("unresolved_conflicting_label_content").sum(), 2)
        self.assertEqual((~ledger.metadata_eligible).sum(), 1)

    def test_missing_identifiers_are_not_independent_groups(self):
        metadata, hashes = population()
        metadata.loc[0, "culture_id"] = "UNKNOWN"
        metadata.loc[1, "slide_id"] = None
        clean, ledger = clean_population(metadata, hashes)
        self.assertEqual(len(clean), 10)
        self.assertEqual(ledger.exclusion_reason.eq("unverified_or_missing_metadata").sum(), 2)
        self.assertFalse(known(pd.Series(["", "UNKNOWN", "nan", "None", "<NA>", None])).any())

    def test_duplicate_paths_failed_hashes_and_missing_inventory_rejected(self):
        metadata, hashes = population()
        for invalid in (hashes.iloc[:-1], pd.concat([hashes, hashes.iloc[:1]])):
            with self.assertRaisesRegex(ValueError, "inventory"):
                clean_population(metadata, invalid)
        for value in (None, "invalid"):
            invalid = hashes.copy()
            invalid.loc[0, "original_sha256"] = value
            with self.assertRaisesRegex(ValueError, "hashes"):
                clean_population(metadata, invalid)
        hashes.loc[0, "hash_status"] = "failed"
        with self.assertRaisesRegex(ValueError, "hashes"):
            clean_population(metadata, hashes)

    def test_original_labels_order_and_inputs_unchanged(self):
        metadata, hashes = population()
        before, hashes_before = metadata.copy(deep=True), hashes.copy(deep=True)
        clean, _ = clean_population(metadata, hashes)
        pd.testing.assert_frame_equal(metadata, before)
        pd.testing.assert_frame_equal(hashes, hashes_before)
        self.assertEqual(clean.class_id.tolist(), metadata.class_id.tolist())
        self.assertEqual(clean.image_path.tolist(), metadata.image_path.tolist())
        self.assertEqual(clean.legacy_field_group_id.tolist(), metadata.group_id.tolist())


class ComponentTests(unittest.TestCase):
    def test_crossed_identifier_union_not_tuple_grouping(self):
        df = pd.DataFrame({"image_path": ["a", "b", "c", "d"],
                           "culture_id": ["C1", "C1", "C2", "C3"],
                           "slide_id": ["S1", "S2", "S3", "S4"],
                           "acquisition_session_id": ["A1", "A2", "A2", "A3"]})
        columns = ["culture_id", "slide_id", "acquisition_session_id"]
        components = component_ids(df, columns)
        self.assertEqual(components.iloc[:3].nunique(), 1)
        self.assertNotEqual(components.iloc[0], components.iloc[3])
        self.assertEqual(len(df[columns].drop_duplicates()), 4)
        shuffled = df.sample(frac=1, random_state=21)
        self.assertEqual(dict(zip(df.image_path, components)),
                         dict(zip(shuffled.image_path, component_ids(shuffled, columns))))

    def test_unknown_grouping_identifier_fails(self):
        df = pd.DataFrame({"image_path": ["a", "b"], "culture_id": ["C1", "UNKNOWN"]})
        with self.assertRaisesRegex(ValueError, "Unknown grouping identifier"):
            component_ids(df, ["culture_id"])

    def test_same_label_hash_bridges_preparations_and_slides(self):
        metadata, hashes = population()
        hashes.loc[1, "original_sha256"] = hashes.loc[0, "original_sha256"]
        clean, _ = clean_population(metadata, hashes)
        self.assertEqual(len(clean), len(metadata))
        grouped, support, _ = grouping_tables(clean, [(0, "class_0"), (1, "class_1")], 5)
        col = "preparation_slide_and_hash_group_id"
        self.assertEqual(grouped.loc[0, col], grouped.loc[1, col])
        row = support.loc[support.variant.eq("preparation_slide_and_hash") & support.class_id.eq(0)].iloc[0]
        self.assertEqual(row.n_components, 5)

    def test_one_culture_per_class_prevents_closed_set_split(self):
        metadata, hashes = population()
        clean, _ = clean_population(metadata, hashes)
        _, support, components = grouping_tables(clean, [(0, "class_0"), (1, "class_1")], 5)
        joint = support.loc[support.variant.eq("culture_slide_session_field")]
        self.assertEqual(joint.n_components.tolist(), [1, 1])
        self.assertFalse(joint.train_test_necessary_count_met.any())
        self.assertEqual(components.variant.eq("culture_slide_session_field").sum(), 1)
        self.assertTrue(support.loc[support.variant.eq("preparation_slide_and_hash"),
                                    "requested_outer_fold_necessary_count_met"].all())

    def test_classes_removed_by_cleanup_still_report_zero_support(self):
        metadata, hashes = population()
        metadata.loc[metadata.class_id.eq(1), "culture_id"] = "UNKNOWN"
        clean, _ = clean_population(metadata, hashes)
        _, support, _ = grouping_tables(clean, [(0, "class_0"), (1, "class_1")], 5)
        absent = support.loc[support.class_id.eq(1)]
        self.assertTrue(absent.n_images.eq(0).all())
        self.assertTrue(absent.n_components.eq(0).all())


class ArtifactTests(unittest.TestCase):
    def test_sufficient_counts_are_not_reported_as_validated_cv(self):
        metadata, hashes = population()
        metadata["culture_id"] = metadata.preparation_id
        with tempfile.TemporaryDirectory() as directory:
            report = export_audit(Path(directory) / "audit", metadata, hashes, [], n_folds=5)
            self.assertEqual(report["status"], "necessary_counts_met_split_not_validated")
            self.assertFalse(report["training_prepared"])

    def test_infeasible_report_never_produces_trainable_folds(self):
        metadata, hashes = population()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "audit"
            report = export_audit(out, metadata, hashes, [], n_folds=5)
            self.assertEqual(report["status"], "infeasible")
            self.assertFalse(report["training_prepared"])
            self.assertFalse(report["scid_used"])
            self.assertFalse((out / "run.json").exists())
            self.assertFalse((out / "splits").exists())
            self.assertEqual(report["source_images_rehashed"], 0)
            self.assertIn("cannot occur in both training and testing", (out / "feasibility.md").read_text())
            self.assertEqual(json.loads((out / "feasibility.json").read_text()), report)
            with self.assertRaises(FileExistsError):
                export_audit(out, metadata, hashes, [])

    def test_rehash_detects_changed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            path.write_bytes(b"original")
            inventory = pd.DataFrame([{"image_path": str(path), "original_sha256": hashlib.sha256(b"original").hexdigest()}])
            verify_images(inventory, 1)
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Image changed"):
                verify_images(inventory, 1)


if __name__ == "__main__":
    unittest.main()
