"""Regression coverage for matched preparation/slide sensitivity training."""

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code/Training/Rebuttal"))
import prepare_preparation_sensitivity as prepare
import run_nested_cv as runner
from preparation_sensitivity import (
    TASKS, audit_frozen_partition, audit_identifiers, grouped_manifest, make_partitions, validate_config,
)
from summarize_preparation_sensitivity import summarize
from validation import classification_metrics, sha256


def population():
    return pd.DataFrame([
        {"image_path": f"/synthetic/{label}/{prep}/{image}.png", "class_id": label,
         "class_name": f"class_{label}", "group_id": f"field_{label}_{image}",
         "legacy_field_group_id": f"field_{label}_{image}", "original_sha256": f"{label}_{prep}_{image}",
         "preparation_id": f"prep_{label}_{prep}", "slide_id": f"slide_{label}_{prep}",
         "culture_id": str(label + 101), "acquisition_session_id": str(prep // 4 + 110),
         "timepoint_min": prep * 5, "exclusion_reason": "included"}
        for label in range(3) for prep in range(8) for image in range(8)
    ])


class SplitTests(unittest.TestCase):
    def test_complete_class_coverage_required_disjointness_and_reproducibility(self):
        for task in TASKS:
            df = grouped_manifest(population(), task)
            partitions = make_partitions(df, task)
            repeated = make_partitions(df, task)
            np.testing.assert_array_equal(np.sum([r == "test" for r, _ in partitions], axis=0), 1)
            for (roles, audit), (again, _) in zip(partitions, repeated):
                np.testing.assert_array_equal(roles, again)
                for part in audit["partitions"].values():
                    self.assertEqual(set(part["class_counts"]), {"0", "1", "2"})
                for row in audit["identifier_overlap"]:
                    if row["identifier"] in TASKS[task]:
                        self.assertEqual(row["n_shared_identifiers"], 0)

    def test_allowed_field_overlap_is_measured_not_hidden(self):
        df = grouped_manifest(population(), "preparation_sensitivity")
        _, audit = make_partitions(df, "preparation_sensitivity")[0]
        rows = [r for r in audit["identifier_overlap"] if r["identifier"] == "legacy_field_group_id"]
        self.assertTrue(all(r["n_shared_identifiers"] > 0 for r in rows))

    def test_duplicates_bridge_preparations_and_remain_together(self):
        source = population()
        source.loc[8, "original_sha256"] = source.loc[0, "original_sha256"]
        for task in TASKS:
            df = grouped_manifest(source, task)
            self.assertEqual(df.group_id.iloc[0], df.group_id.iloc[8])
            for roles, _ in make_partitions(df, task):
                self.assertEqual(roles[0], roles[8])

    def test_insufficient_groups_or_mixed_label_groups_fail(self):
        source = population().loc[lambda d: d.timepoint_min.lt(20)].reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            make_partitions(grouped_manifest(source, "preparation_sensitivity"), "preparation_sensitivity")
        source = population()
        source.loc[64, "original_sha256"] = source.loc[0, "original_sha256"]
        with self.assertRaisesRegex(ValueError, "class-pure"):
            grouped_manifest(source, "field_baseline")

    def test_false_group_labels_cannot_bypass_identifier_overlap(self):
        df = population()
        df["group_id"] = df.image_path
        roles = np.tile(["train", "validation", "test"], len(df) // 3)
        with self.assertRaisesRegex(ValueError, "Forbidden"):
            audit_identifiers(df, roles, TASKS["preparation_sensitivity"])

    def test_frozen_split_label_or_group_changes_fail(self):
        df = grouped_manifest(population(), "preparation_sensitivity")
        roles, _ = make_partitions(df, "preparation_sensitivity")[0]
        split = df[["image_path", "class_id", "group_id"]].copy()
        split["row_id"] = range(len(df))
        split["role"] = roles
        audit_frozen_partition(df, split, "preparation_sensitivity")
        for column, value in (("class_id", 2), ("group_id", "wrong"), ("image_path", "wrong")):
            changed = split.copy()
            changed.loc[0, column] = value
            with self.assertRaisesRegex(ValueError, "do not match"):
                audit_frozen_partition(df, changed, "preparation_sensitivity")

    def test_cli_requires_explicit_overlap_acknowledgement(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            prepare.main(["--run_dir", "/unused"])


class PipelineTests(unittest.TestCase):
    def frozen(self, root):
        source = population()
        for folder in ("Code/Training/Rebuttal", "Code/Training/Model_Training_5_class", "Code/utils"):
            dest = root / folder
            dest.mkdir(parents=True)
            (dest / "stub.py").write_text("# fixture\n")
        training = {"class_names": [f"class_{i}" for i in range(3)], "epochs": 15, "batch_size": 32,
                    "lr": .0003, "weight_decay": .0001, "img_size": 224, "seed": 42,
                    "n_folds": 5, "use_amp": True}
        with patch.object(prepare, "ROOT", root), patch.object(prepare.subprocess, "check_output", side_effect=["commit\n", "branch\n"]):
            config = prepare.freeze(root / "run", source, source, training, [], num_workers=0)
        return root / "run", config

    def test_freeze_matching_arms_warning_and_incomplete_report(self):
        with tempfile.TemporaryDirectory() as directory:
            run, config = self.frozen(Path(directory))
            validate_config(config)
            self.assertEqual(set(config["tasks"]), set(TASKS))
            self.assertTrue(config["allow_legacy_field_overlap"])
            self.assertIn("EXPLORATORY", config["limitations"][0])
            report = summarize(run)
            self.assertEqual(report["status"], "incomplete")
            self.assertNotIn("metric_comparison", report)
            self.assertNotIn("fold_statistics", report["tasks"]["field_baseline"])
            self.assertTrue((run / "identifier_overlap.csv").exists())
            self.assertTrue((run / "partition_time_counts.csv").exists())
            config["allow_legacy_field_overlap"] = False
            with self.assertRaisesRegex(ValueError, "acknowledgement"):
                validate_config(config)

    def test_report_rejects_tampered_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            run, _ = self.frozen(Path(directory))
            (run / "partition_counts.csv").write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                summarize(run)

    def test_training_dispatch_and_complete_report_recompute_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            run, config = self.frozen(Path(directory))
            selections, evaluations = [], []

            def select(model, train, validation, settings, fold, dest, device):
                self.assertFalse(set(train.group_id) & set(validation.group_id))
                dest.mkdir()
                torch.save({}, dest / "best.pt")
                selections.append((dest.parent.parent.name, fold))
                return {"epoch": 2, "checkpoint_sha256": sha256(dest / "best.pt"),
                        "validation_metrics": {"macro_f1": .8}}

            def test(model, data, train, settings, checkpoint, device):
                task = checkpoint.parent.parent.parent.name
                fold = int(checkpoint.parent.parent.name.split("_")[1])
                self.assertIn((task, fold), selections)
                self.assertTrue((checkpoint.parent.parent / "model_selection.json").exists())
                self.assertFalse(set(data.group_id) & set(train.group_id))
                evaluations.append((task, fold))
                pred = data.copy()
                pred["y_true"] = pred.class_id
                pred["y_pred"] = pred.class_id
                if task == "preparation_sensitivity":
                    pred.loc[pred.index[0], "y_pred"] = (int(pred.y_true.iloc[0]) + 1) % 3
                pred["pred_class_name"] = [settings["class_names"][i] for i in pred.y_pred]
                for i, name in enumerate(settings["class_names"]):
                    pred["prob_" + name] = pred.y_pred.eq(i).astype(float)
                return pred, classification_metrics(pred.y_true, pred.y_pred, 3)

            with patch.object(runner, "select_checkpoint", side_effect=select), \
                 patch.object(runner, "test_checkpoint", side_effect=test), contextlib.redirect_stdout(io.StringIO()):
                for task in TASKS:
                    for fold in range(1, 6):
                        runner.run_fold(run, task, fold, device="cpu")
                runner.run_fold(run, "field_baseline", 1, device="cpu")
            self.assertEqual(len(evaluations), 10)
            report = summarize(run)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["tasks"]["field_baseline"]["fold_statistics"]["macro_f1_mean"], 1.)
            self.assertTrue(all(r["preparation_minus_field_mean"] < 0 for r in report["metric_comparison"]))
            for task in TASKS:
                folder = run / task
                self.assertEqual(len(pd.read_csv(folder / "outer_test_predictions.csv")), len(population()))
                self.assertTrue((folder / "confusion_counts.csv").exists())
                self.assertTrue((folder / "errors_by_preparation_id.csv").exists())
            self.assertEqual(len(pd.read_csv(run / "preparation_sensitivity/error_cases.csv")), 5)
            path = run / "field_baseline/fold_1/efficientnet_b0/test_predictions.csv"
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "predictions hash mismatch"):
                summarize(run)


class SlurmTests(unittest.TestCase):
    def test_array_routing(self):
        script = ROOT / "Code/Training/Rebuttal/run_preparation_sensitivity.sbatch"
        for index in range(0, 12):
            env = {**os.environ, "SLURM_ARRAY_TASK_ID": str(index), "REBUTTAL_PYTHON": "/bin/echo"}
            result = subprocess.run(["bash", str(script), "/tmp/frozen run"], env=env, capture_output=True, text=True)
            if 1 <= index <= 10:
                self.assertEqual(result.returncode, 0, result.stderr)
                task, fold = ("field_baseline", index) if index <= 5 else ("preparation_sensitivity", index - 5)
                self.assertIn(f"--task {task} --fold {fold}", result.stdout)
            else:
                self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
