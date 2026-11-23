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
    DISPLAY, HIGH_CONTRAST_COLORS, PROTOCOL, compare_reference_predictions, compute_rocs, make_figure,
    verify_confusion_reference,
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

    def test_text_free_grid_free_export_preserves_curves_and_inset(self):
        from matplotlib.text import Text
        import matplotlib.pyplot as plt

        df, names = population()
        curves, summary = compute_rocs(df, names)
        original, original_ax, original_inset = make_figure(curves, summary)
        fig, ax, inset = make_figure(curves, summary, no_text=True, no_grid=True)
        try:
            fig.canvas.draw()
            self.assertFalse(any(text.get_visible() and text.get_text() for text in fig.findobj(Text)))
            self.assertIsNone(ax.get_legend())
            self.assertEqual(len(ax.lines), 6)
            self.assertEqual(len(inset.lines), 5)
            for before, after in ((original_ax, ax), (original_inset, inset)):
                self.assertEqual(before.get_xlim(), after.get_xlim())
                self.assertEqual(before.get_ylim(), after.get_ylim())
                np.testing.assert_array_equal(before.get_xticks(), after.get_xticks())
                np.testing.assert_array_equal(before.get_yticks(), after.get_yticks())
                self.assertFalse(any(line.get_visible() for line in after.get_xgridlines() + after.get_ygridlines()))
                for first, second in zip(before.lines, after.lines):
                    np.testing.assert_array_equal(first.get_xdata(), second.get_xdata())
                    np.testing.assert_array_equal(first.get_ydata(), second.get_ydata())
                    self.assertEqual(first.get_color(), second.get_color())
                    self.assertEqual(first.get_linestyle(), second.get_linestyle())
            with tempfile.TemporaryDirectory() as directory:
                for extension in ("png", "pdf", "svg"):
                    path = Path(directory) / ("roc_no_text_no_grid." + extension)
                    fig.savefig(path)
                    self.assertGreater(path.stat().st_size, 1000)
                svg = (Path(directory) / "roc_no_text_no_grid.svg").read_text()
                self.assertNotIn("<text", svg)
                self.assertNotIn("DejaVuSans", svg)
        finally:
            plt.close(original)
            plt.close(fig)

    def test_grid_can_be_removed_without_removing_text(self):
        from matplotlib.text import Text
        import matplotlib.pyplot as plt

        df, names = population()
        curves, summary = compute_rocs(df, names)
        fig, ax, inset = make_figure(curves, summary, no_grid=True)
        try:
            fig.canvas.draw()
            self.assertIsNotNone(ax.get_legend())
            self.assertTrue(ax.get_xlabel())
            self.assertTrue(inset.get_title())
            self.assertTrue(any(text.get_visible() and text.get_text() for text in fig.findobj(Text)))
            for axis in (ax, inset):
                self.assertFalse(any(line.get_visible() for line in axis.get_xgridlines() + axis.get_ygridlines()))
        finally:
            plt.close(fig)

    def test_high_contrast_thickens_and_darkens_curves_without_changing_data(self):
        from matplotlib.colors import to_rgb
        from matplotlib.text import Text
        import matplotlib.pyplot as plt

        def luminance(color):
            rgb = np.asarray(to_rgb(color))
            linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
            return linear @ np.array([.2126, .7152, .0722])

        df, names = population()
        curves, summary = compute_rocs(df, names)
        original, original_ax, original_inset = make_figure(curves, summary, no_text=True, no_grid=True)
        fig, ax, inset = make_figure(curves, summary, no_text=True, no_grid=True, high_contrast=True)
        try:
            fig.canvas.draw()
            self.assertFalse(any(text.get_visible() and text.get_text() for text in fig.findobj(Text)))
            for before, after in ((original_ax, ax), (original_inset, inset)):
                self.assertEqual(before.get_xlim(), after.get_xlim())
                self.assertEqual(before.get_ylim(), after.get_ylim())
                self.assertEqual(len(before.lines), len(after.lines))
                self.assertFalse(any(line.get_visible() for line in after.get_xgridlines() + after.get_ygridlines()))
                for i, (first, second) in enumerate(zip(before.lines[:5], after.lines[:5])):
                    np.testing.assert_array_equal(first.get_xdata(), second.get_xdata())
                    np.testing.assert_array_equal(first.get_ydata(), second.get_ydata())
                    self.assertEqual(first.get_linestyle(), second.get_linestyle())
                    self.assertEqual(second.get_linewidth(), 5.0 - i * .2)
                    self.assertGreater(second.get_linewidth(), 1.8 * first.get_linewidth())
                    self.assertEqual(second.get_color(), HIGH_CONTRAST_COLORS[names[i]])
                    self.assertLess(luminance(second.get_color()), luminance(first.get_color()))
            self.assertEqual(original_ax.lines[-1].get_linewidth(), ax.lines[-1].get_linewidth())
            self.assertEqual(original_ax.lines[-1].get_color(), ax.lines[-1].get_color())
            for main_line, zoom in zip(ax.lines[:5], inset.lines):
                self.assertEqual(main_line.get_color(), zoom.get_color())
                self.assertEqual(main_line.get_linewidth(), zoom.get_linewidth())
            with tempfile.TemporaryDirectory() as directory:
                for extension in ("png", "pdf", "svg"):
                    path = Path(directory) / ("roc_bold." + extension)
                    fig.savefig(path)
                    self.assertGreater(path.stat().st_size, 1000)
        finally:
            plt.close(original)
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
