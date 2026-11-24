"""Treatment marginalization, held-out feature checks and six-figure exports."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code/Training/Rebuttal"))
from corrected_treated15 import (
    STRAINS, TREATMENTS, scores, timepoint_tables, treatment_predictions,
    tsne_fold, validate_predictions, verify_feature_arrays,
)
from plot_corrected_treated15 import (
    FIGURES, condition_accuracy, count_matrix, fold_statistics, main, plot_accuracy,
    plot_confusion, plot_rocs, plot_timepoints, plot_tsne, row_percent, save_figure,
    treatment_rocs, verify_features,
)
from validation import PROTOCOL, sha256


def population():
    names = sorted(f"{s} + {t}" for s in STRAINS for t in TREATMENTS)
    labels = np.tile(np.arange(15), 10)
    p = np.full((len(labels), 15), .01)
    p[np.arange(len(labels)), labels] = .86
    frame = pd.DataFrame({"image_path": [f"/image/{i}" for i in range(len(labels))],
        "image_uid": [f"uid-{i}" for i in range(len(labels))], "class_id": labels,
        "y_true": labels, "y_pred": labels, "fold": np.repeat(np.arange(1, 6), 30),
        "timepoint_min": np.tile(np.repeat([5, 90], 15), 5)})
    frame["class_name"] = frame.y_true.map(dict(enumerate(names)))
    frame["pred_class_name"] = frame.class_name
    frame["canonical_strain"] = frame.class_name.str.rsplit(" + ", n=1).str[0]
    frame["canonical_treatment"] = frame.class_name.str.rsplit(" + ", n=1).str[1]
    for i, name in enumerate(names):
        frame["prob_" + name] = p[:, i]
    return frame, names


class CorrectedTreated15Tests(unittest.TestCase):
    def test_marginalization_sums_all_strains_not_top_class_treatment(self):
        frame, names = population()
        p = np.zeros(15)
        p[names.index("AIEC + Ampicillin")] = .4
        for strain in STRAINS:
            p[names.index(strain + " + Kanamycin")] = .12
        frame.loc[0, ["prob_" + name for name in names]] = p
        frame.loc[0, "y_pred"] = names.index("AIEC + Ampicillin")
        frame.loc[0, "pred_class_name"] = "AIEC + Ampicillin"
        result = treatment_predictions(frame, names)
        self.assertEqual(result.loc[0, "pred_treatment_top_class_id"], 0)
        self.assertEqual(result.loc[0, "pred_treatment_id"], 2)
        np.testing.assert_allclose(result[["treatment_prob_" + t for t in TREATMENTS]].sum(axis=1), 1)
        np.testing.assert_allclose(result.loc[0, ["treatment_prob_" + t for t in TREATMENTS]].to_numpy(float), [.4, 0, .6])
        self.assertEqual(result.y_pred.tolist(), frame.y_pred.tolist())

    def test_invalid_identity_labels_metadata_and_probabilities_rejected(self):
        frame, names = population()
        for bad in (frame.iloc[:0], pd.concat([frame, frame.iloc[:1]])):
            with self.assertRaises(ValueError):
                validate_predictions(bad, names)
        for column, value in (("y_true", 15), ("y_pred", 1), ("class_id", -.5),
                              ("class_name", "wrong"), ("canonical_strain", "wrong"),
                              ("timepoint_min", np.nan), ("prob_" + names[0], np.nan),
                              ("prob_" + names[0], -.1), ("prob_" + names[0], .7)):
            bad = frame.copy()
            if column == "class_id":
                bad[column] = bad[column].astype(float)
            bad.loc[0, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                validate_predictions(bad, names)
        with self.assertRaises(ValueError):
            validate_predictions(frame, names[:14])

    def test_confusion_counts_and_exact_accuracy_share_denominators(self):
        frame, names = population()
        frame.loc[0, "y_pred"] = 1
        cm = count_matrix(frame.y_true, frame.y_pred, names)
        self.assertEqual(cm.to_numpy().sum(), len(frame))
        self.assertEqual(cm.iloc[0, 1], 1)
        np.testing.assert_allclose(row_percent(cm).sum(axis=1), 100)
        accuracy = condition_accuracy(frame, names)
        np.testing.assert_array_equal(accuracy.n_correct_exact, np.diag(cm))
        np.testing.assert_array_equal(accuracy.n_images, cm.sum(axis=1))
        self.assertEqual(accuracy.exact_class_accuracy.iloc[0], .9)
        with self.assertRaises(ValueError):
            count_matrix([0, 3], [0, 1], TREATMENTS)
        with self.assertRaises(ValueError):
            row_percent(pd.DataFrame([[1, 0], [0, 0]]))

    def test_missing_exact_class_does_not_hide_undefined_auroc(self):
        frame, names = population()
        frame = frame.loc[~(frame.timepoint_min.eq(90) & frame.class_name.eq("E.coli5-alpha + CND"))]
        table, support = timepoint_tables(treatment_predictions(frame, names), names)
        exact = table.loc[table.task.eq("exact15") & table.timepoint_min.eq(90)]
        self.assertTrue(exact.macro_auroc.isna().all())
        np.testing.assert_allclose(exact.macro_f1, 14 / 15)
        treatment = table.loc[table.task.eq("treatment3")]
        self.assertTrue(treatment.macro_auroc.eq(1).all())
        absent = support.loc[support.task.eq("exact15") & support.timepoint_min.eq(90)
                             & support.class_name.eq("E.coli5-alpha + CND")]
        self.assertEqual(len(absent), 6)
        self.assertTrue(absent.n_positive.eq(0).all())
        pooled = table.loc[table.task.eq("treatment3") & table.fold.eq(0)]
        self.assertEqual(pooled.n_images.sum(), len(frame))

    def test_fold_sd_does_not_silently_average_over_missing_folds(self):
        frame, names = population()
        table, _ = timepoint_tables(treatment_predictions(frame, names), names)
        table.loc[table.task.eq("exact15") & table.fold.eq(1), "macro_auroc"] = np.nan
        stats = fold_statistics(table)
        missing = stats.loc[stats.task.eq("exact15") & stats.metric.eq("macro_auroc")]
        self.assertTrue(missing["mean"].isna().all())
        self.assertTrue(missing.sample_sd.isna().all())
        self.assertTrue(missing.n_defined_folds.eq(4).all())
        with self.assertRaises(ValueError):
            fold_statistics(table.loc[table.fold.ne(5)])

    def test_treatment_roc_threshold_counts_and_ties(self):
        frame, names = population()
        pred = treatment_predictions(frame, names)
        curves, summary = treatment_rocs(pred)
        self.assertTrue(summary.full_range_auroc.eq(1).all())
        self.assertTrue(summary.n_positive.eq(50).all())
        self.assertTrue(summary.n_negative.eq(100).all())
        for row in curves.itertuples():
            positive = pred.true_treatment_id.eq(TREATMENTS.index(row.treatment)).to_numpy()
            selected = pred["treatment_prob_" + row.treatment].to_numpy() >= row.threshold
            self.assertEqual(row.true_positives, int((selected & positive).sum()))
            self.assertEqual(row.false_positives, int((selected & ~positive).sum()))
            self.assertEqual(row.false_negatives + row.true_positives, 50)
            self.assertEqual(row.true_negatives + row.false_positives, 100)

    def test_roc_uses_probabilities_not_hard_labels(self):
        frame, names = population()
        pred = treatment_predictions(frame, names)
        before, aucs = treatment_rocs(pred)
        pred["pred_treatment_id"] = 0
        after, changed = treatment_rocs(pred)
        pd.testing.assert_frame_equal(before, after)
        pd.testing.assert_frame_equal(aucs, changed)

    def test_feature_verification_rejects_wrong_vectors_or_probabilities(self):
        frame, names = population()
        p = validate_predictions(frame, names)
        features = np.zeros((len(frame), 1280))
        self.assertEqual(verify_feature_arrays(features, p, frame, names), 0)
        for bad in (features[:, :1279], features * np.nan):
            with self.assertRaises(ValueError):
                verify_feature_arrays(bad, p, frame, names)
        for bad in (p[::-1], p[:, :14], p * np.nan, p + .01):
            with self.assertRaises(ValueError):
                verify_feature_arrays(features, bad, frame, names)
        # Even within numeric tolerance, a changed argmax must fail.
        frame[["prob_" + n for n in names]] = 0.
        frame["prob_" + names[0]], frame["prob_" + names[1]] = .5, .5
        frame["y_pred"] = 0
        tied = frame[["prob_" + n for n in names]].to_numpy()
        tied[:, 0], tied[:, 1] = .499999, .500001
        with self.assertRaisesRegex(ValueError, "argmax_disagreements"):
            verify_feature_arrays(features, tied, frame, names)

    def test_tsne_is_deterministic_and_rejects_insufficient_samples(self):
        features = np.random.default_rng(42).normal(size=(45, 8)).astype(np.float32)
        first, info = tsne_fold(features, iterations=300)
        second, _ = tsne_fold(features, iterations=300)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (45, 2))
        self.assertEqual(info["fit_scope"], "one_outer_fold_only")
        for bad in (features[:30], features[:, :1], features * np.nan):
            with self.assertRaises(ValueError):
                tsne_fold(bad)

    def test_feature_provenance_checks_fold_identity_and_artifact_hashes(self):
        pred, names = population()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, features = root / "run", root / "features"
            features.mkdir()
            details = []
            for fold, group in pred.groupby("fold"):
                checkpoint = run / f"treated15/fold_{fold}/efficientnet_b0/best.pt"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_text(f"fixture checkpoint {fold}")
                np.savez_compressed(features / f"fold_{fold}_features.npz",
                    features=np.zeros((len(group), 1280)),
                    probabilities=group[["prob_" + n for n in names]].to_numpy(),
                    image_uid=group.image_uid.to_numpy(str), image_path=group.image_path.to_numpy(str))
                details.append({"fold": int(fold), "checkpoint_sha256": sha256(checkpoint),
                                "n_images": len(group), "argmax_disagreements": 0})
            coords = pred.copy()
            coords["tsne_x"], coords["tsne_y"] = np.arange(len(pred)), 0.
            coords.to_csv(features / "tsne_coordinates.csv", index=False)
            report = {"all_checks_passed": True, "protocol": PROTOCOL, "run_dir": str(run),
                "n_images": len(pred), "feature_dim": 1280, "coordinates_comparable_across_folds": False,
                "folds": details, "source_sha256": {},
                "artifact_sha256": {p.name: sha256(p) for p in features.iterdir()}}
            metadata = features / "verification.json"
            metadata.write_text(json.dumps(report))
            checked, _ = verify_features(features, pred, names, run, {})
            self.assertEqual(checked.image_uid.tolist(), pred.image_uid.tolist())
            with self.assertRaisesRegex(ValueError, "this corrected evaluation"):
                verify_features(features, pred, names, root / "other_run", {})
            with self.assertRaisesRegex(ValueError, "identities do not match"):
                verify_features(features, pred.iloc[::-1], names, run, {})
            coords.loc[0, "tsne_x"] = 500
            coords.to_csv(features / "tsne_coordinates.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Frozen source changed"):
                verify_features(features, pred, names, run, {})

    def test_exact_timepoint_plot_marks_absent_class_auc_as_unavailable(self):
        frame, names = population()
        frame = frame.loc[~(frame.timepoint_min.eq(90) & frame.y_true.eq(0))]
        table, _ = timepoint_tables(treatment_predictions(frame, names), names)
        fig = plot_timepoints(table, "exact15")
        self.assertEqual(len(fig.axes[0].patches), 3)
        self.assertTrue(any(text.get_text() == "N/A" for text in fig.axes[0].texts))
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_roc_inset_preserves_exact_coordinates(self):
        frame, names = population()
        curves, aucs = treatment_rocs(treatment_predictions(frame, names))
        fig = plot_rocs(curves, aucs)
        ax = fig.axes[0]
        inset = ax.child_axes[0]
        for line, zoom in zip(ax.lines[:3], inset.lines):
            np.testing.assert_array_equal(line.get_xdata(), zoom.get_xdata())
            np.testing.assert_array_equal(line.get_ydata(), zoom.get_ydata())
            self.assertGreaterEqual(line.get_linewidth(), 3)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_six_figures_export_all_formats(self):
        frame, names = population()
        pred = treatment_predictions(frame, names)
        table, _ = timepoint_tables(pred, names)
        curves, aucs = treatment_rocs(pred)
        coords = pred.copy()
        coords["tsne_x"], coords["tsne_y"] = np.arange(len(pred)), np.sin(np.arange(len(pred)))
        figures = [plot_confusion(count_matrix(pred.true_treatment_id, pred.pred_treatment_id, TREATMENTS), "Treatment"),
                   plot_timepoints(table, "treatment3"), plot_rocs(curves, aucs), plot_tsne(coords),
                   plot_accuracy(condition_accuracy(pred, names)), plot_confusion(count_matrix(pred.y_true, pred.y_pred, names), "Exact")]
        self.assertEqual(len(figures[1].axes[0].patches), 4)
        self.assertEqual(len(figures[3].axes), 6)
        for fold, ax in enumerate(figures[3].axes[:5], 1):
            self.assertEqual(sum(len(collection.get_offsets()) for collection in ax.collections), int(pred.fold.eq(fold).sum()))
        with tempfile.TemporaryDirectory() as directory:
            for name, fig in zip(FIGURES, figures):
                save_figure(fig, Path(directory), name)
                for extension in ("png", "pdf", "svg"):
                    self.assertGreater((Path(directory) / f"{name}.{extension}").stat().st_size, 1000)

    def test_existing_outputs_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "preserve.txt"
            marker.write_text("unchanged")
            with patch.object(sys, "argv", ["plot", "--feature_dir", "unused", "--out_dir", directory]):
                with patch("plot_corrected_treated15.load_evaluation") as load:
                    with self.assertRaises(FileExistsError):
                        main()
                    load.assert_not_called()
            self.assertEqual(marker.read_text(), "unchanged")


if __name__ == "__main__":
    unittest.main()
