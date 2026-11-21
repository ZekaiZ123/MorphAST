"""Exact duplicate input conflicts must not be mistaken for phenotype similarity."""

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code/Training/Rebuttal"))
from audit_duplicate_conflicts import hash_file, summarize_hash_groups


class DuplicateConflictTests(unittest.TestCase):
    def test_same_bytes_different_names_labels_and_folds(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ["amp.tif", "kan.tif", "unique.tif"]]
            for path, content in zip(paths, [b"identical input", b"identical input", b"different input"]):
                path.write_bytes(content)
            inventory = pd.DataFrame([hash_file(str(p)) for p in paths])
            pred = pd.DataFrame({"image_path": list(map(str, paths)), "class_name": ["Amp", "Kan", "Kan"],
                                 "fold": [1, 2, 1], "group_id": ["g1", "g2", "g3"], "timepoint_min": [30] * 3,
                                 "source_folder": ["amp", "kan", "kan"], "y_true": [0, 1, 1], "y_pred": [1, 1, 1]})
            joined, groups = summarize_hash_groups(pred, inventory)
            self.assertEqual(groups.has_label_conflict.sum(), 1)
            self.assertEqual(groups.spans_heldout_folds.sum(), 1)
            self.assertEqual(joined.has_label_conflict.tolist(), [True, True, False])
            self.assertEqual(joined.exact_error.tolist(), [True, False, False])

    def test_failed_hashes_are_unknown_not_unique(self):
        pred = pd.DataFrame({"image_path": ["/missing"], "class_name": ["Kan"], "fold": [1], "group_id": ["g"],
                             "timepoint_min": [30], "source_folder": ["kan"], "y_true": [1], "y_pred": [0]})
        joined, groups = summarize_hash_groups(pred, pd.DataFrame([hash_file("/missing")]))
        self.assertTrue(groups.empty)
        self.assertTrue(pd.isna(joined.has_label_conflict.iloc[0]))
        self.assertEqual(joined.hash_status.iloc[0], "failed")


if __name__ == "__main__":
    unittest.main()
