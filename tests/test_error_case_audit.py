"""Raw-count conservation, strict provenance joins, and honest error exports."""

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code/Training/Rebuttal"))
from audit_error_cases import (
    ACQUISITION, UNKNOWN, add_decisions, attach_acquisition, attach_duplicate_evidence, confusion_long,
    export_assets, matrix, per_class_metrics, select_comparisons, validate_predictions, write_gallery,
)


NAMES = ["E.coli5-alpha + Ampicillin", "E.coli5-alpha + CND", "E.coli5-alpha + Kanamycin",
         "AIEC + Kanamycin"]


def fixture():
    labels = [0, 1, 2, 3]
    probabilities = [[.40, .05, .30, .25], [.02, .96, .01, .01],
                     [.60, .05, .30, .05], [.01, .01, .01, .97]]
    rows = []
    for i, (label, p) in enumerate(zip(labels, probabilities)):
        strain, treatment = NAMES[label].split(" + ")
        rows.append({"image_path": f"/images/{i}.png", "image_uid": f"uid-{i}",
                     "group_id": f"group-{i}", "field_group_id": f"group-{i}",
                     "class_id": label, "class_name": NAMES[label], "canonical_strain": strain,
                     "canonical_treatment": treatment, "source_root": "collection",
                     "source_folder": NAMES[label], "timepoint_min": 30, "drop_from_training": 0,
                     "y_true": label, "y_pred": int(np.argmax(p)), "pred_class_name": NAMES[int(np.argmax(p))],
                     "fold": 1 + i % 2, **dict(zip(["prob_" + n for n in NAMES], p))})
    pred = pd.DataFrame(rows)
    settings = {"n_samples": 4, "n_folds": 2, "class_names": NAMES, "class_counts": dict.fromkeys(NAMES, 1)}
    return pred, pred.copy(), settings


class ErrorCaseTests(unittest.TestCase):
    def test_full_raw_counts_conserve_population_and_zeros(self):
        pred, _, _ = fixture()
        df = add_decisions(attach_acquisition(pred), NAMES)
        counts = matrix(df, "class_name", "pred_class_name", NAMES)
        self.assertEqual(counts.to_numpy().sum(), 4)
        self.assertEqual(counts.shape, (4, 4))
        long = confusion_long(df, ["fold"], "class_name", "pred_class_name", NAMES)
        self.assertEqual(len(long), 32)
        self.assertEqual(long["count"].sum(), 4)
        self.assertTrue(long["count"].eq(0).any())

    def test_marginal_treatment_is_not_top_class_mapping(self):
        pred, _, _ = fixture()
        df = add_decisions(attach_acquisition(pred), NAMES)
        self.assertFalse(df.iloc[0].exact_error)
        self.assertTrue(df.iloc[0].treatment_error)
        self.assertTrue(df.iloc[0].any_error)
        self.assertEqual(df.iloc[0].pred_treatment_top_class, "Ampicillin")
        self.assertEqual(df.iloc[0].pred_treatment_marginal, "Kanamycin")

    def test_per_class_metrics_include_false_positives_and_absent_labels(self):
        pred, _, _ = fixture()
        cm = matrix(pred.iloc[:3], "class_name", "pred_class_name", NAMES)
        metrics = per_class_metrics(cm).set_index("class_name")
        self.assertEqual(metrics.loc[NAMES[0], "tp"], 1)
        self.assertEqual(metrics.loc[NAMES[0], "fp"], 1)
        self.assertEqual(metrics.loc[NAMES[0], "precision"], .5)
        self.assertEqual(metrics.loc[NAMES[2], "fn"], 1)
        self.assertEqual(metrics.loc[NAMES[3], "support"], 0)
        self.assertEqual(metrics.loc[NAMES[3], "f1"], 0)

    def test_batches_are_not_inferred_from_folders(self):
        pred, _, _ = fixture()
        df = attach_acquisition(pred)
        self.assertTrue(df.batch_id.eq(UNKNOWN).all())
        self.assertFalse(df.source_collection_is_verified_batch.any())
        metadata = pd.DataFrame([{"image_path": pred.image_path.iloc[0], "metadata_source": "acquisition log",
                                  **dict.fromkeys(ACQUISITION, UNKNOWN), "batch_id": "B01"}])
        attached = attach_acquisition(pred, metadata)
        self.assertEqual(attached.batch_id.tolist(), ["B01", UNKNOWN, UNKNOWN, UNKNOWN])
        with self.assertRaisesRegex(ValueError, "unique exact paths"):
            attach_acquisition(pred, pd.concat([metadata, metadata]))

    def test_strict_population_labels_and_probabilities(self):
        pred, manifest, settings = fixture()
        self.assertEqual(len(validate_predictions(pred, manifest, settings)), 4)
        for column, value in [("pred_class_name", "wrong"), ("class_name", "wrong"), ("timepoint_min", 5),
                              ("prob_" + NAMES[0], -.5), ("y_pred", .5)]:
            broken = pred.copy()
            if column == "y_pred":
                broken[column] = broken[column].astype(float)
            broken.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                validate_predictions(broken, manifest, settings)
        with self.assertRaisesRegex(ValueError, "population"):
            validate_predictions(pred.iloc[:-1], manifest, settings)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_predictions(pd.concat([pred, pred.iloc[[0]]]), manifest, settings)

    def test_missing_controls_are_not_silently_relaxed(self):
        pred, _, _ = fixture()
        df = add_decisions(attach_acquisition(pred), NAMES)
        pairs = select_comparisons(df)
        self.assertTrue(pairs.true_condition_control_id.eq("").all())
        self.assertTrue(pairs.predicted_condition_control_id.eq("").all())
        shuffled = select_comparisons(df.sample(frac=1, random_state=5))
        pd.testing.assert_frame_equal(pairs, shuffled)

    def test_duplicate_evidence_checks_population_and_archived_bytes(self):
        pred, _, _ = fixture()
        pred["asset_status"] = "ok"
        pred["original_sha256"] = ["a", "b", "c", "d"]
        audit = pred.copy()
        audit["hash_status"] = "ok"
        audit["n_images"] = 2
        audit["has_label_conflict"] = True
        audit["spans_heldout_folds"] = True
        attached = attach_duplicate_evidence(pred, audit)
        self.assertTrue(attached.conflicting_condition_labels.all())
        with self.assertRaisesRegex(ValueError, "population"):
            attach_duplicate_evidence(pred, audit.iloc[:-1])
        audit.loc[0, "original_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "differs"):
            attach_duplicate_evidence(pred, audit)

    def test_original_bytes_preserved_and_every_error_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pred, _, _ = fixture()
            for i in pred.index:
                path = root / f"source-{i}.png"
                Image.fromarray(np.full((24, 30), i * 50, dtype=np.uint8)).save(path)
                pred.loc[i, "image_path"] = str(path)
            df = add_decisions(attach_acquisition(pred), NAMES)
            pairs = select_comparisons(df)
            out = root / "export"
            out.mkdir()
            exported = export_assets(df, pairs, out)
            self.assertEqual(exported.asset_status.eq("ok").sum(), 2)
            for row in exported.loc[exported.any_error].itertuples():
                self.assertEqual(Path(row.image_path).read_bytes(), (out / row.original_file).read_bytes())
                self.assertEqual(hashlib.sha256(Path(row.image_path).read_bytes()).hexdigest(), row.original_sha256)
                with Image.open(out / row.model_input_file) as image:
                    self.assertEqual(image.size, (224, 224))
            write_gallery(exported, out)
            gallery = (out / "error_gallery.html").read_text()
            for case_id in exported.loc[exported.any_error, "case_id"]:
                self.assertIn(case_id, gallery)

    def test_missing_image_is_reported_not_dropped(self):
        pred, _, _ = fixture()
        df = add_decisions(attach_acquisition(pred), NAMES)
        with tempfile.TemporaryDirectory() as directory:
            exported = export_assets(df, select_comparisons(df), Path(directory))
            self.assertEqual(exported.asset_status.eq("failed").sum(), 2)
            self.assertEqual(len(exported), len(df))


if __name__ == "__main__":
    unittest.main()
