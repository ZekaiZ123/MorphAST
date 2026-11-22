"""One-versus-rest scores, threshold counts, provenance and inset checks."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code/Training/Rebuttal"))
from plot_corrected_fiveclass_roc import (
    DISPLAY, PROTOCOL, compare_reference_predictions, compute_rocs, make_figure, verify_confusion_reference,
)


def population():
    names = list(DISPLAY)
    labels = np.tile(np.arange(5), 3)
    probabilities = np.full((len(labels), 5), .05)
    probabilities[np.arange(len(labels)), labels] = .8
    df = pd.DataFrame({"image_path": [f"/image/{i}" for i in range(len(labels))],
                       "image_uid": [f"uid-{i}" for i in range(len(labels))],
                       "class_id": labels, "y_true": labels, "y_pred": labels,
                       "fold": np.repeat([1, 2, 3], 5), "model": "selected"})
    df["class_name"] = df.class_id.map(dict(enumerate(names)))
    for i, name in enumerate(names):
        df["prob_" + name] = probabilities[:, i]
    return df, names


class CorrectedFiveclassROCTests(unittest.TestCase):
    def test_perfect_scores_and_tied_scores(self):
        df, names = population()
        curves, summary = compute_rocs(df, names)
        self.assertEqual(len(summary), 5)
        self.assertTrue(summary.auc.eq(1).all())
        self.assertTrue(summary.n_positive.eq(3).all())
        self.assertTrue(summary.n_negative.eq(12).all())
        df[["prob_" + n for n in names]] = .2
        _, tied = compute_rocs(df, names)
        self.assertTrue(tied.auc.eq(.5).all())
        self.assertEqual(set(curves.class_name), set(names))

    def test_all_other_strains_are_negative_and_threshold_counts_are_exact(self):
        df, names = population()
        df.loc[0, ["prob_" + n for n in names]] = [.15, .55, .1, .1, .1]
        curves, summary = compute_rocs(df, names)
        for row in summary.itertuples():
            y = df.y_true.eq(row.class_id).to_numpy()
            score = df["prob_" + row.class_name].to_numpy()
            self.assertAlmostEqual(row.auc, roc_auc_score(y, score))
            self.assertEqual(row.n_positive + row.n_negative, len(df))
            for point in curves.loc[curves.class_name.eq(row.class_name)].itertuples():
                positive = score >= point.threshold
                self.assertEqual(point.true_positives, int((positive & y).sum()))
                self.assertEqual(point.false_positives, int((positive & ~y).sum()))
                self.assertAlmostEqual(point.specificity, 1 - point.false_positive_rate)

    def test_roc_does_not_use_hard_predictions(self):
        df, names = population()
        before, _ = compute_rocs(df, names)
        df["y_pred"] = 0
        after, _ = compute_rocs(df, names)
        pd.testing.assert_frame_equal(before, after)

    def test_rejects_duplicates_missing_classes_and_invalid_probabilities(self):
        df, names = population()
        for changed in (pd.concat([df, df.iloc[:1]]), df.loc[df.y_true.ne(4)]):
            with self.assertRaises(ValueError):
                compute_rocs(changed, names)
        for value in (-.1, 1.1, float("nan"), float("inf"), .7):
            changed = df.copy()
            changed.loc[0, "prob_" + names[0]] = value
            with self.assertRaises(ValueError):
                compute_rocs(changed, names)

    def test_reference_check_includes_scores_and_selected_models(self):
        df, names = population()
        compare_reference_predictions(df, df.copy(), names)
        for key, value in (("model", "different"), ("fold", 5), ("prob_" + names[0], .79)):
            changed = df.copy()
            changed.loc[0, key] = value
            with self.assertRaises(AssertionError):
                compare_reference_predictions(df, changed, names)

    def test_reference_must_be_primary_and_same_corrected_run(self):
        df, names = population()
        cm = pd.DataFrame(np.eye(5, dtype=int) * 3, index=names, columns=names)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "verification.json").write_text(json.dumps({
                "all_checks_passed": True, "protocol": PROTOCOL, "run_dir": str(root)}))
            with self.assertRaisesRegex(ValueError, "primary"):
                verify_confusion_reference(df, cm, root / "efficientnet_b0", root, {})
            with self.assertRaisesRegex(ValueError, "same run"):
                verify_confusion_reference(df, cm, root / "primary", root / "different", {})

    def test_inset_reuses_all_five_empirical_curves_without_jitter(self):
        df, names = population()
        curves, summary = compute_rocs(df, names)
        fig, ax, inset = make_figure(curves, summary)
        try:
            self.assertEqual(len(ax.lines), 6)  # Five ROC curves and chance diagonal.
            self.assertEqual(len(inset.lines), 5)
            self.assertEqual(inset.get_xlim()[1], .05)
            self.assertEqual(inset.get_ylim()[0], .8)
            for main, zoom in zip(ax.lines[:5], inset.lines):
                np.testing.assert_array_equal(main.get_xdata(), zoom.get_xdata())
                np.testing.assert_array_equal(main.get_ydata(), zoom.get_ydata())
            with tempfile.TemporaryDirectory() as directory:
                for extension in ("png", "pdf", "svg"):
                    path = Path(directory) / ("roc." + extension)
                    fig.savefig(path)
                    self.assertGreater(path.stat().st_size, 1000)
        finally:
            import matplotlib.pyplot as plt
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
