"""Regression tests for selection/test separation and corrected reporting."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import sklearn
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code/Training/Rebuttal"))
import run_nested_cv as runner
from summarize_nested_cv import summarize
from validation import (
    PROTOCOL, audit_partition, choose_candidate, classification_metrics, fold_statistics,
    make_nested_partitions, sha256, write_json,
)


def sample_manifest():
    return pd.DataFrame([
        {"image_path": f"/synthetic/class_{cls}/group_{group}/image_{image}.png",
         "group_id": f"{cls}_{group}", "class_id": cls, "class_name": f"class_{cls}"}
        for cls in range(3) for group in range(30) for image in range(2)
    ])


class PartitionTests(unittest.TestCase):
    def test_outer_test_coverage_group_separation_and_repeatability(self):
        df = sample_manifest()
        partitions = make_nested_partitions(df)
        repeated = make_nested_partitions(df)
        for (roles, audit), (roles_again, _) in zip(partitions, repeated):
            self.assertTrue(np.array_equal(roles, roles_again))
            self.assertEqual(audit["group_overlap"], 0)
            self.assertEqual(sum(p["n_samples"] for p in audit["partitions"].values()), len(df))
        self.assertTrue(np.all(np.sum([roles == "test" for roles, _ in partitions], axis=0) == 1))

    def test_group_leakage_rejected(self):
        df = sample_manifest()
        roles, _ = make_nested_partitions(df)[0]
        first = int(np.where(roles == "train")[0][0])
        roles[first] = "validation"
        with self.assertRaisesRegex(ValueError, "Group leakage"):
            audit_partition(df, roles)

    def test_backbone_selection_ignores_outer_metrics_and_ties_are_stable(self):
        scores = {"a": {"validation_metrics": {"macro_f1": .8}, "test_macro_f1": .1},
                  "b": {"validation_metrics": {"macro_f1": .7}, "test_macro_f1": 1.0}}
        self.assertEqual(choose_candidate(scores, ["a", "b"]), "a")
        scores["b"]["validation_metrics"]["macro_f1"] = .8
        self.assertEqual(choose_candidate(scores, ["a", "b"]), "a")

    def test_sample_and_population_standard_deviations(self):
        rows = [{m: x for m in ("macro_f1", "accuracy", "balanced_accuracy")} for x in (.5, .6, .7, .8, .9)]
        stats = fold_statistics(rows)
        self.assertAlmostEqual(stats["accuracy_mean"], .7)
        self.assertAlmostEqual(stats["accuracy_std"], np.std([.5, .6, .7, .8, .9], ddof=1))
        self.assertAlmostEqual(stats["accuracy_std_population"], np.std([.5, .6, .7, .8, .9], ddof=0))


class CheckpointTests(unittest.TestCase):
    def test_epoch_selection_uses_inner_validation_and_cpu_snapshot_is_independent(self):
        df = sample_manifest()
        roles, _ = make_nested_partitions(df)[0]
        train = df.loc[roles == "train"]
        validation = df.loc[roles == "validation"]
        settings = {"seed": 42, "class_names": ["a", "b", "c"], "lr": .001,
                    "weight_decay": .0001, "use_amp": False, "epochs": 3}
        model = torch.nn.Linear(1, 3, bias=False)
        epochs = []
        validation_calls = []

        def train_epoch(model, data, *args, **kwargs):
            self.assertEqual(set(data.image_path), set(train.image_path))
            epochs.append(len(epochs) + 1)
            with torch.no_grad():
                model.weight.fill_(epochs[-1])
            return .5

        def evaluate(model, data, *args, **kwargs):
            self.assertEqual(set(data.image_path), set(validation.image_path))
            validation_calls.append(1)
            score = [.9, .6, .8][len(validation_calls) - 1]
            return {"macro_f1": score, "accuracy": score, "balanced_accuracy": score, "loss": .5}

        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "candidate"
            with patch.object(runner, "build_model", return_value=model), \
                 patch.object(runner, "loader", side_effect=lambda df, *a, **kw: df), \
                 patch.object(runner, "train_one_epoch", side_effect=train_epoch), \
                 patch.object(runner, "evaluate", side_effect=evaluate):
                selected = runner.select_checkpoint("resnet18", train, validation, settings, 1, dest, "cpu")
            self.assertEqual(selected["epoch"], 1)
            self.assertEqual(len(validation_calls), 3)
            saved = torch.load(dest / "best.pt", weights_only=True)
            self.assertTrue(torch.all(saved["weight"] == 1))
            self.assertTrue(torch.all(model.weight == 3))
            counts = np.bincount(train.class_id)
            np.testing.assert_allclose(selected["training_class_weights"], counts.sum() / (3 * counts))


class PipelineTests(unittest.TestCase):
    def test_complete_report_recomputes_metrics_and_rejects_tampered_predictions(self):
        df = sample_manifest()
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            task_dir = run_dir / "untreated"
            (task_dir / "splits").mkdir(parents=True)
            (task_dir / "legacy").mkdir()
            df.to_csv(task_dir / "manifest.csv", index=False)
            settings = {"n_folds": 5, "candidates": ["resnet18"],
                        "class_names": ["class_0", "class_1", "class_2"],
                        "manifest_sha256": sha256(task_dir / "manifest.csv"), "splits": {}}
            legacy_rows = []
            for fold, (roles, _) in enumerate(make_nested_partitions(df), 1):
                split = df[["image_path", "class_id", "group_id"]].copy()
                split["role"] = roles
                split.to_csv(task_dir / "splits" / f"fold_{fold}.csv", index=False)
                settings["splits"][str(fold)] = sha256(task_dir / "splits" / f"fold_{fold}.csv")
                model_dir = task_dir / f"fold_{fold}" / "resnet18"
                model_dir.mkdir(parents=True)
                torch.save({}, model_dir / "best.pt")
                selection_path = model_dir.parent / "model_selection.json"
                write_json(selection_path, {"selected_model": "resnet18", "candidates": {
                    "resnet18": {"validation_metrics": {"macro_f1": .8}}}})
                predictions = df.loc[roles == "test"].copy()
                predictions["y_true"] = predictions.class_id
                predictions["y_pred"] = predictions.class_id
                predictions["fold"] = fold
                for cls in range(3):
                    predictions[f"prob_class_{cls}"] = (predictions.class_id == cls).astype(float)
                predictions.to_csv(model_dir / "test_predictions.csv", index=False)
                result = classification_metrics(predictions.y_true, predictions.y_pred, 3)
                result.update(fold=fold, model="resnet18", epoch=1, n_test=len(predictions),
                              selected_for_primary=True, test_evaluations=1,
                              selection_sha256=sha256(selection_path),
                              checkpoint_sha256=sha256(model_dir / "best.pt"),
                              predictions_sha256=sha256(model_dir / "test_predictions.csv"))
                write_json(model_dir / "test_metrics.json", result)
                legacy_rows.append({"fold": fold, "macro_f1": .9, "accuracy": .9, "balanced_accuracy": .9})
            write_json(task_dir / "legacy/resnet18.json", {"fold_metrics": legacy_rows})
            write_json(run_dir / "run.json", {"protocol": PROTOCOL, "tasks": {"untreated": settings}})
            for fold in range(1, 6):
                write_json(task_dir / f"fold_{fold}/complete.json", {"run_sha256": sha256(run_dir / "run.json")})
            report = summarize(run_dir)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["tasks"]["untreated"]["primary"]["macro_f1_mean"], 1.)
            self.assertAlmostEqual(report["tasks"]["untreated"]["legacy_comparisons"][0]["paired_delta_mean"], .1)
            predictions_path = task_dir / "fold_1/resnet18/test_predictions.csv"
            predictions_path.write_text(predictions_path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "predictions hash mismatch"):
                summarize(run_dir)

    def test_test_loader_is_sealed_until_all_selections_and_incomplete_report_has_no_average(self):
        df = sample_manifest()
        roles, _ = make_nested_partitions(df)[0]
        settings = {"seed": 42, "n_folds": 5, "class_names": ["class_0", "class_1", "class_2"],
                    "candidates": ["resnet18", "efficientnet_b0"]}
        selected = []
        tested = []
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            task_dir = run_dir / "untreated"
            (task_dir / "splits").mkdir(parents=True)
            df.to_csv(task_dir / "manifest.csv", index=False)
            split = df[["image_path", "group_id", "class_id"]].copy()
            split.insert(0, "row_id", range(len(df)))
            split["role"] = roles
            split.to_csv(task_dir / "splits/fold_1.csv", index=False)
            settings.update(manifest_sha256=sha256(task_dir / "manifest.csv"),
                            splits={"1": sha256(task_dir / "splits/fold_1.csv")})
            config = {"protocol": PROTOCOL, "sklearn": sklearn.__version__,
                      "source_sha256": {}, "tasks": {"untreated": settings}}
            write_json(run_dir / "run.json", config)

            def select(model, train, validation, settings, fold, dest, device):
                self.assertEqual(set(train.image_path), set(df.loc[roles == "train", "image_path"]))
                self.assertEqual(set(validation.image_path), set(df.loc[roles == "validation", "image_path"]))
                self.assertFalse(tested)
                dest.mkdir()
                torch.save({}, dest / "best.pt")
                selected.append(model)
                return {"epoch": 1, "checkpoint_sha256": sha256(dest / "best.pt"),
                        "validation_metrics": {"macro_f1": .8 if model == "resnet18" else .7}}

            def test(model, test_df, train, settings, checkpoint, device):
                self.assertEqual(selected, settings["candidates"])
                frozen = json.loads((task_dir / "fold_1/model_selection.json").read_text())
                self.assertEqual(frozen["selected_model"], "resnet18")
                self.assertEqual(set(test_df.image_path), set(df.loc[roles == "test", "image_path"]))
                tested.append(model)
                predictions = test_df.copy()
                predictions["y_true"] = test_df.class_id
                predictions["y_pred"] = test_df.class_id
                return predictions, {"accuracy": 1., "macro_f1": 1., "balanced_accuracy": 1.}

            with patch.object(runner, "select_checkpoint", side_effect=select), \
                 patch.object(runner, "test_checkpoint", side_effect=test):
                runner.run_fold(run_dir, "untreated", 1, device="cpu")
                runner.run_fold(run_dir, "untreated", 1, device="cpu")
            self.assertEqual(tested, settings["candidates"])
            report = summarize(run_dir)
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["tasks"]["untreated"]["complete_folds"], [1])
            self.assertNotIn("primary", report["tasks"]["untreated"])


if __name__ == "__main__":
    unittest.main()
