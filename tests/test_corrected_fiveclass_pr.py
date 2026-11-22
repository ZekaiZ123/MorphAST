"""PR score semantics, tied thresholds, ROC provenance, and inset checks."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import auc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code/Training/Rebuttal"))
from plot_corrected_fiveclass_pr import (
    DISPLAY, INSET_PRECISION, INSET_RECALL, PROTOCOL, compute_prs, main, make_figure, verify_roc_reference,
)
from validation import sha256


def population():
    names = list(DISPLAY)
    labels = np.tile(np.arange(5), 5)
    scores = np.full((len(labels), 5), .05)
    scores[np.arange(len(labels)), labels] = .8
    frame = pd.DataFrame({"image_path": [f"/image/{i}" for i in range(len(labels))],
                          "image_uid": [f"uid-{i}" for i in range(len(labels))],
                          "class_id": labels, "y_true": labels, "y_pred": labels,
                          "fold": np.repeat(np.arange(1, 6), 5), "model": "selected"})
    frame["class_name"] = frame.class_id.map(dict(enumerate(names)))
    for i, name in enumerate(names):
        frame["prob_" + name] = scores[:, i]
    return frame, names


def write_roc_fixture(root, pred, procedure="primary_inner_selected"):
    pred.to_csv(root / "outer_test_predictions.csv", index=False)
    (root / "summary.json").write_text(json.dumps({
        "procedure": procedure, "models_by_fold": {str(i): "selected" for i in range(1, 6)},
        "n_images": len(pred), "n_folds": 5}))
    (root / "verification.json").write_text(json.dumps({
        "all_checks_passed": True, "protocol": PROTOCOL, "run_dir": str(root),
        "artifact_sha256": {name: sha256(root / name) for name in ("outer_test_predictions.csv", "summary.json")}}))


class CorrectedFiveclassPRTests(unittest.TestCase):
    def test_perfect_scores_and_tied_scores(self):
        pred, names = population()
        _, summary = compute_prs(pred, names)
        self.assertTrue(summary.average_precision.eq(1).all())
        self.assertTrue(summary.n_positive.eq(5).all())
        self.assertTrue(summary.n_negative.eq(20).all())
        pred[["prob_" + name for name in names]] = .2
        curves, tied = compute_prs(pred, names)
        np.testing.assert_array_equal(tied.average_precision, tied.prevalence)
        for _, group in curves.groupby("class_id"):
            self.assertEqual(len(group), 2)
            self.assertAlmostEqual(group.precision.iloc[0], .2)
            self.assertAlmostEqual(group.recall.iloc[0], 1)
            self.assertNotAlmostEqual(auc(group.recall, group.precision), .2)

    def test_threshold_counts_use_all_four_other_strains_and_preserve_ties(self):
        pred, names = population()
        pred.loc[0, ["prob_" + name for name in names]] = [.15, .55, .1, .1, .1]
        curves, summary = compute_prs(pred, names)
        for row in summary.itertuples():
            y = pred.y_true.eq(row.class_id).to_numpy()
            scores = pred["prob_" + row.class_name].to_numpy()
            group = curves.loc[curves.class_name.eq(row.class_name)]
            self.assertEqual(len(group), len(np.unique(scores)) + 1)
            self.assertAlmostEqual(row.average_precision, -np.sum(np.diff(group.recall) * group.precision.to_numpy()[:-1]))
            for point in group.itertuples():
                positive = scores >= point.threshold
                tp, fp = int((positive & y).sum()), int((positive & ~y).sum())
                self.assertEqual(point.true_positives, tp)
                self.assertEqual(point.false_positives, fp)
                self.assertEqual(point.true_negatives, (~y).sum() - fp)
                self.assertEqual(point.false_negatives, y.sum() - tp)
                self.assertAlmostEqual(point.recall, tp / y.sum())
                self.assertAlmostEqual(point.precision, tp / (tp + fp) if tp + fp else 1)

    def test_endpoint_is_explicit_not_an_observed_threshold(self):
        pred, names = population()
        curves, _ = compute_prs(pred, names)
        for _, group in curves.groupby("class_id"):
            endpoint = group.iloc[-1]
            self.assertTrue(endpoint.is_synthetic_endpoint)
            self.assertEqual(group.is_synthetic_endpoint.sum(), 1)
            self.assertTrue(np.isposinf(endpoint.threshold))
            self.assertEqual((endpoint.recall, endpoint.precision), (0, 1))
            self.assertEqual((endpoint.true_positives, endpoint.false_positives), (0, 0))

    def test_pr_uses_softmax_scores_not_hard_predictions(self):
        pred, names = population()
        before, summary = compute_prs(pred, names)
        pred["y_pred"] = 0
        after, changed = compute_prs(pred, names)
        pd.testing.assert_frame_equal(before, after)
        pd.testing.assert_frame_equal(summary, changed)

    def test_invalid_population_labels_and_probabilities_are_rejected(self):
        pred, names = population()
        for changed in (pred.iloc[:0], pd.concat([pred, pred.iloc[:1]]), pred.loc[pred.y_true.ne(4)]):
            with self.assertRaises(ValueError):
                compute_prs(changed, names)
        for value in (-.1, 1.1, float("nan"), float("inf"), .7):
            changed = pred.copy()
            changed.loc[0, "prob_" + names[0]] = value
            with self.assertRaises(ValueError):
                compute_prs(changed, names)
        for value in (-1, 5, .5, float("nan")):
            changed = pred.copy()
            changed["y_true"] = changed.y_true.astype(float)
            changed.loc[0, "y_true"] = value
            with self.assertRaises(ValueError):
                compute_prs(changed, names)
        for bad_names in (names[:4], names[:4] + names[:1]):
            with self.assertRaises(ValueError):
                compute_prs(pred, bad_names)

    def test_roc_reference_requires_identical_primary_scores_and_source_hashes(self):
        pred, names = population()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_roc_fixture(root, pred)
            evidence = {}
            verify_roc_reference(pred, names, root, root, evidence)
            self.assertEqual(len(evidence), 3)
            with self.assertRaisesRegex(ValueError, "same run"):
                verify_roc_reference(pred, names, root, root / "other", {})
            changed = pred.copy()
            changed.loc[0, "prob_" + names[0]] = .79
            with self.assertRaises(AssertionError):
                verify_roc_reference(changed, names, root, root, {})
            changed.to_csv(root / "outer_test_predictions.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Frozen source changed"):
                verify_roc_reference(pred, names, root, root, {})
            write_roc_fixture(root, pred, procedure="secondary")
            with self.assertRaisesRegex(ValueError, "primary"):
                verify_roc_reference(pred, names, root, root, {})

    def test_inset_reuses_empirical_steps_and_prevalence_is_the_baseline(self):
        pred, names = population()
        curves, summary = compute_prs(pred, names)
        fig, ax, inset = make_figure(curves, summary)
        try:
            self.assertEqual(len(ax.lines), 10)
            self.assertEqual(len(inset.lines), 5)
            self.assertEqual(inset.get_xlim()[0], INSET_RECALL[0])
            self.assertEqual(inset.get_ylim()[0], INSET_PRECISION[0])
            for main_line, zoom in zip(ax.lines[:5], inset.lines):
                np.testing.assert_array_equal(main_line.get_xdata(), zoom.get_xdata())
                np.testing.assert_array_equal(main_line.get_ydata(), zoom.get_ydata())
                self.assertEqual(main_line.get_drawstyle(), "steps-post")
                self.assertEqual(zoom.get_drawstyle(), "steps-post")
            for baseline, prevalence in zip(ax.lines[5:], summary.prevalence):
                np.testing.assert_array_equal(baseline.get_ydata(), [prevalence, prevalence])
            with tempfile.TemporaryDirectory() as directory:
                for extension in ("png", "pdf", "svg"):
                    path = Path(directory) / ("pr." + extension)
                    fig.savefig(path)
                    self.assertGreater(path.stat().st_size, 1000)
        finally:
            import matplotlib.pyplot as plt
            plt.close(fig)

    def test_existing_outputs_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "existing.txt"
            marker.write_text("preserve earlier outputs")
            with patch.object(sys, "argv", ["plot", "--confusion_dir", "unused", "--roc_dir", "unused", "--out_dir", directory]):
                with patch("plot_corrected_fiveclass_pr.load_results") as load:
                    with self.assertRaises(FileExistsError):
                        main()
                    load.assert_not_called()
            self.assertEqual(marker.read_text(), "preserve earlier outputs")


if __name__ == "__main__":
    unittest.main()
