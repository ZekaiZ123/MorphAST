"""Corrected outer-fold count conservation and safe plotting checks."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code/Training/Rebuttal"))
from plot_corrected_fiveclass_confusion import DISPLAY, load_results, pooled_counts, record_source, render_counts


class CorrectedFiveclassConfusionTests(unittest.TestCase):
    def population(self):
        names = list(DISPLAY)
        frame = pd.DataFrame({"image_path": [f"/image/{i}" for i in range(6)],
                              "class_id": [0, 1, 2, 3, 3, 4]})
        frame["class_name"] = frame.class_id.map(dict(enumerate(names)))
        predictions = frame.assign(y_true=frame.class_id, y_pred=[0, 1, 2, 3, 4, 3])
        return frame, predictions, names

    def test_raw_counts_include_zeros_and_preserve_class_order(self):
        manifest, predictions, names = self.population()
        cm, metrics = pooled_counts(predictions, manifest, names)
        self.assertEqual(cm.index.tolist(), names)
        self.assertEqual(cm.to_numpy().sum(), 6)
        self.assertEqual(cm.loc["AIEC", "UPEC"], 1)
        self.assertEqual(cm.loc["UPEC", "AIEC"], 1)
        self.assertEqual(cm.loc[names[0], names[1]], 0)
        self.assertAlmostEqual(metrics["accuracy"], 4 / 6)

    def test_duplicate_or_missing_outer_images_are_rejected(self):
        manifest, predictions, names = self.population()
        for changed in (predictions.iloc[:-1], pd.concat([predictions, predictions.iloc[:1]])):
            with self.assertRaisesRegex(ValueError, "exactly once"):
                pooled_counts(changed, manifest, names)

    def test_invalid_or_changed_labels_are_rejected(self):
        manifest, predictions, names = self.population()
        for key, value in (("y_true", 1), ("y_pred", 5), ("y_pred", -.1), ("y_pred", float("nan"))):
            changed = predictions.copy()
            changed[key] = changed[key].astype(float)
            changed.loc[0, key] = value
            with self.assertRaises(ValueError):
                pooled_counts(changed, manifest, names)

    def test_changed_evidence_and_legacy_protocol_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "run.json"
            path.write_text(json.dumps({"protocol": "legacy"}))
            evidence = {}
            digest = record_source(path, evidence)
            with self.assertRaisesRegex(ValueError, "corrected"):
                load_results(root, "primary", {})
            path.write_text("changed")
            with self.assertRaises(ValueError):
                record_source(path, evidence, digest)
            with self.assertRaises(ValueError):
                record_source(path, evidence)

    def test_plot_exports_all_formats_without_mutating_counts(self):
        manifest, predictions, names = self.population()
        cm, _ = pooled_counts(predictions, manifest, names)
        before = cm.copy()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            render_counts(cm, "primary", root)
            for suffix in ("png", "pdf", "svg"):
                self.assertGreater((root / f"confusion_counts.{suffix}").stat().st_size, 1000)
            pd.testing.assert_frame_equal(cm, before)


if __name__ == "__main__":
    unittest.main()
