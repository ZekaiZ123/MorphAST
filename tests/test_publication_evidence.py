"""Original publication counts, unknown replicate coverage and raw-matrix checks."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code/Training/Rebuttal"))
from export_publication_evidence import (
    ACQUISITION_IDS, checked_matrix, class_tables, save_confusions, validate_population,
)


def fixture():
    frame = pd.DataFrame({
        "image_path": [f"/images/{i}.tif" for i in range(5)],
        "group_id": ["a", "b", "c", "d", "e"],
        "class_id": [0, 0, 1, 1, 1], "class_name": ["A", "A", "B", "B", "B"],
        "canonical_strain": ["A", "A", "B", "B", "B"], "canonical_treatment": "Untreated",
        "drop_from_training": [0, 0, 0, 0, 1],
        "decision": ["retained"] * 4 + ["qc_failed"],
        "mapping_status": ["unique_composite_and_dimensions"] * 2 + ["no_exact_composite_match"] * 3,
        **{key: ["101", "101", "UNKNOWN", "UNKNOWN", "UNKNOWN"] for key in ACQUISITION_IDS},
    })
    settings = {"model": "efficientnet_b0", "mode": "crossval", "filter_auto_dropped": True,
                "n_samples": 4, "n_groups": 4, "class_names": ["A", "B"], "class_counts": {"A": 2, "B": 2}}
    return frame, settings


class PublicationEvidenceTests(unittest.TestCase):
    def test_class_tables_keep_unknown_images_without_inventing_replicates(self):
        frame, settings = fixture()
        table = class_tables(frame, settings["class_names"]).set_index("class_name")
        self.assertEqual(table.n_before.sum(), 5)
        self.assertEqual(table.n_after.sum(), 4)
        self.assertEqual(table.loc["A", "n_recorded_preparation_id"], 1)
        self.assertEqual(table.loc["B", "n_recorded_preparation_id"], 0)
        self.assertEqual(table.loc["B", "n_unknown_preparation_id"], 2)
        self.assertTrue(table.independent_biological_replicates.eq("NOT_ESTABLISHED").all())
        self.assertEqual(table.loc["A", "preparation_ids"], "101")

    def test_excluded_images_do_not_inflate_retained_identifier_counts(self):
        frame, settings = fixture()
        frame.loc[4, "preparation_id"] = "new_id"
        table = class_tables(frame, settings["class_names"])
        self.assertEqual(table.n_recorded_preparation_id.sum(), 1)
        self.assertEqual(table.n_qc_failed.sum(), 1)

    def test_population_and_label_order_must_match_publication_settings(self):
        frame, settings = fixture()
        self.assertEqual(len(validate_population(frame, settings)), 4)
        for changed in ({**settings, "n_samples": 3}, {**settings, "n_groups": 3},
                        {**settings, "class_names": ["B", "A"]},
                        {**settings, "model": "resnet18"}, {**settings, "filter_auto_dropped": False}):
            with self.assertRaises(ValueError):
                validate_population(frame, changed)
        for column, value in (("drop_from_training", np.nan), ("image_path", "/images/1.tif"),
                              ("class_id", .5)):
            changed = frame.copy()
            changed[column] = changed[column].astype(object)
            changed.loc[0, column] = value
            with self.assertRaises(ValueError):
                validate_population(changed, settings)

    def metrics(self):
        return {"fold": 1, "n_train": 2, "n_val": 2, "confusion_matrix": [[1, 0], [1, 0]],
                "accuracy": .5, "balanced_accuracy": .5, "macro_f1": 1 / 3}

    def test_rejects_fractional_negative_nonfinite_and_wrong_saved_counts(self):
        checked_matrix(self.metrics(), ["A", "B"], 1, 4)
        for values in ([[.5, .5], [1, 0]], [[-1, 2], [1, 0]], [[1, 0], [float("nan"), 0]],
                       [[1, 0]], [[1, 0], [0, 0]]):
            with self.assertRaises(ValueError):
                checked_matrix({**self.metrics(), "confusion_matrix": values}, ["A", "B"], 1, 4)
        for key, value in (("fold", 2), ("n_train", 1), ("accuracy", .9), ("macro_f1", .9)):
            with self.assertRaises(ValueError):
                checked_matrix({**self.metrics(), key: value}, ["A", "B"], 1, 4)

    def test_saved_matrices_conserve_samples_and_fold_training_support(self):
        cm = checked_matrix(self.metrics(), ["A", "B"], 1, 4)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            total = save_confusions(None, ["A", "B"], out, {1: cm, 2: cm}, render=False)
            self.assertEqual(total.to_numpy().sum(), 4)
            folds = pd.read_csv(out / "confusion_exact_by_fold.csv")
            self.assertEqual(len(folds), 8)
            self.assertEqual(folds["count"].sum(), 4)
            self.assertTrue(folds["count"].eq(0).any())
            support = pd.read_csv(out / "class_sample_counts_by_fold.csv")
            self.assertTrue(support.n_train.eq(1).all())
            self.assertTrue(support.n_heldout.eq(1).all())
            self.assertFalse((out / "publication_predictions_with_identifiers.csv").exists())


if __name__ == "__main__":
    unittest.main()
