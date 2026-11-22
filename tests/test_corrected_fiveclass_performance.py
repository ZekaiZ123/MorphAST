"""Per-fold metric, primary-model consistency, and grouped-bar checks."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code/Training/Rebuttal"))
from plot_corrected_fiveclass_performance import (
    METRIC_STYLE, build_fold_table, main, make_figure, summarize_folds,
)
from validation import classification_metrics


def population():
    frames, results = [], []
    names = [f"strain_{i}" for i in range(5)]
    for fold in range(1, 6):
        true = np.repeat(np.arange(5), np.arange(1, 6) + fold)
        predicted = true.copy()
        predicted[:fold] = 1
        model = "resnet18" if fold == 3 else "efficientnet_b0"
        frames.append(pd.DataFrame({"image_path": [f"/fold_{fold}/{i}" for i in range(len(true))],
                                    "fold": fold, "model": model, "y_true": true, "y_pred": predicted}))
        metrics = classification_metrics(true, predicted, 5)
        results.append({"fold": fold, "model": model, "epoch": fold + 1, "n_test": len(true),
                        **{key: metrics[key] for key in METRIC_STYLE}})
    return pd.concat(frames, ignore_index=True), results, names


class CorrectedFiveclassPerformanceTests(unittest.TestCase):
    def test_metrics_are_computed_separately_for_each_outer_fold(self):
        pred, results, names = population()
        frame = build_fold_table(pred.sample(frac=1, random_state=7), list(reversed(results)), names)
        self.assertEqual(frame.fold.tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(frame.model.tolist(), [row["model"] for row in results])
        self.assertEqual(frame.n_test.sum(), len(pred))
        for row in frame.itertuples():
            group = pred.loc[pred.fold.eq(row.fold)]
            counts = np.zeros((5, 5), dtype=int)
            np.add.at(counts, (group.y_true, group.y_pred), 1)
            tp = counts.diagonal()
            self.assertAlmostEqual(row.accuracy, tp.sum() / counts.sum())
            self.assertAlmostEqual(row.balanced_accuracy, (tp / counts.sum(axis=1)).mean())
            self.assertAlmostEqual(row.macro_f1, (2 * tp / (counts.sum(axis=0) + counts.sum(axis=1))).mean())
        self.assertFalse(np.allclose(frame.accuracy, frame.balanced_accuracy))

    def test_rejects_missing_duplicate_and_invalid_outer_population(self):
        pred, results, names = population()
        missing_class = pred.loc[~(pred.fold.eq(1) & pred.y_true.eq(4))]
        cases = [pd.concat([pred, pred.iloc[:1]]), pred.loc[pred.fold.ne(5)], missing_class]
        for key, value in (("image_path", None), ("fold", float("nan")), ("fold", 1.5),
                           ("y_true", -1), ("y_pred", 5), ("y_pred", .5), ("y_pred", float("nan"))):
            changed = pred.copy()
            if key != "image_path":
                changed[key] = changed[key].astype(float)
            changed.loc[0, key] = value
            cases.append(changed)
        for changed in cases:
            with self.assertRaises(ValueError):
                build_fold_table(changed, results, names)
        with self.assertRaises(ValueError):
            build_fold_table(pred, results, names[:4])

    def test_rejects_different_backbone_metrics_and_counts(self):
        pred, results, names = population()
        for key, value in (("model", "other"), ("n_test", 1), ("macro_f1", .1),
                           ("accuracy", .1), ("balanced_accuracy", float("nan"))):
            changed = [dict(row) for row in results]
            changed[0][key] = value
            with self.assertRaises(ValueError):
                build_fold_table(pred, changed, names)
        changed = pred.copy()
        changed.loc[0, "model"] = "other"
        with self.assertRaises(ValueError):
            build_fold_table(changed, results, names)

    def test_rejects_missing_or_duplicate_primary_results(self):
        pred, results, names = population()
        for rows in (results[:4], results + [results[0]], results[:4] + [results[0]]):
            with self.assertRaises(ValueError):
                build_fold_table(pred, rows, names)

    def test_variability_uses_unweighted_mean_and_sample_sd(self):
        pred, results, names = population()
        frame = build_fold_table(pred, results, names)
        summary = summarize_folds(frame)
        self.assertEqual(summary.metric.tolist(), list(METRIC_STYLE))
        for row in summary.itertuples():
            values = frame[row.metric].to_numpy()
            self.assertEqual(row.n_folds, 5)
            self.assertEqual(row.std_ddof, 1)
            self.assertAlmostEqual(row.mean, values.mean())
            self.assertAlmostEqual(row.std, values.std(ddof=1))
            self.assertAlmostEqual(row.mean_percent, 100 * values.mean())
            self.assertAlmostEqual(row.std_percentage_points, 100 * values.std(ddof=1))
            self.assertNotAlmostEqual(row.mean, np.average(values, weights=frame.n_test))

    def test_existing_outputs_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "existing.txt"
            marker.write_text("preserve earlier outputs")
            with patch.object(sys, "argv", ["plot", "--confusion_dir", "unused", "--out_dir", directory]):
                with patch("plot_corrected_fiveclass_performance.load_results") as load:
                    with self.assertRaises(FileExistsError):
                        main()
                    load.assert_not_called()
            self.assertEqual(marker.read_text(), "preserve earlier outputs")

    def test_fifteen_bars_use_exact_percent_values_and_zero_baseline(self):
        pred, results, names = population()
        frame = build_fold_table(pred, results, names)
        fig, ax = make_figure(frame)
        try:
            self.assertEqual(len(ax.patches), 15)
            self.assertEqual(len(ax.containers), 3)
            self.assertEqual(ax.get_ylim()[0], 0)
            self.assertGreaterEqual(ax.get_ylim()[1], 100)
            self.assertEqual([text.get_text() for text in ax.get_legend().get_texts()],
                             [style[0] for style in METRIC_STYLE.values()])
            for metric, bars in zip(METRIC_STYLE, ax.containers):
                np.testing.assert_array_equal([bar.get_height() for bar in bars], frame[metric] * 100)
                self.assertTrue(all(bar.get_y() == 0 for bar in bars))
            self.assertEqual(len(ax.texts), 15)
            for row, label in zip(frame.itertuples(), ax.get_xticklabels()):
                self.assertIn(f"Fold {row.fold}", label.get_text())
                self.assertIn(f"n = {row.n_test}", label.get_text())
            with tempfile.TemporaryDirectory() as directory:
                for extension in ("png", "pdf", "svg"):
                    path = Path(directory) / ("folds." + extension)
                    fig.savefig(path)
                    self.assertGreater(path.stat().st_size, 1000)
        finally:
            import matplotlib.pyplot as plt
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
