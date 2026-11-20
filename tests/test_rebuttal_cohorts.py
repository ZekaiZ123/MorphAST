"""Cohort selection and Slurm routing for the independent 15-class rerun."""

import contextlib
import io
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code/Training/Rebuttal"))
import prepare_nested_cv as prepare
from summarize_nested_cv import summarize
from validation import make_nested_partitions, sha256, write_json


class CohortPreparationTests(unittest.TestCase):
    def make_baseline(self, root):
        images = root / "images"
        images.mkdir()
        rows = []
        for cls in range(15):
            for group in range(30):
                path = images / f"{cls}_{group}.png"
                path.touch()
                rows.append({"image_path": str(path), "group_id": f"{cls}_{group}",
                             "class_id": cls, "class_name": f"class_{cls}",
                             "drop_from_training": 0})
        df = pd.DataFrame(rows)
        manifest = root / "manifest.csv"
        df.to_csv(manifest, index=False)
        baseline = root / "Model/Benchmark_Treated_Condition_15class_QC/efficientnet_b0"
        baseline.mkdir(parents=True)
        training = {"manifest_csv": "manifest.csv", "n_samples": len(df), "n_groups": len(df),
                    "class_names": [f"class_{i}" for i in range(15)], "n_folds": 5,
                    "seed": 42, "epochs": 15, "batch_size": 32, "lr": .0003,
                    "weight_decay": .0001, "img_size": 224, "use_amp": True}
        write_json(baseline / "training_args.json", training)
        folds = []
        for fold, (roles, _) in enumerate(make_nested_partitions(df), 1):
            test = df.loc[roles == "test"]
            dest = baseline / f"fold_{fold}"
            dest.mkdir()
            test.to_csv(dest / "val_predictions.csv", index=False)
            folds.append({"fold": fold, "n_val": len(test)})
        write_json(baseline / "crossval_summary.json", {"fold_metrics": folds})
        for directory in ("Code/Training/Rebuttal", "Code/Training/Model_Training_5_class", "Code/utils"):
            (root / directory).mkdir(parents=True)
        (root / "Code/Training/Rebuttal/run_nested_cv.sbatch").write_text("#!/bin/bash\n")
        return df, baseline

    def test_prepares_only_15class_and_preserves_warnings_and_outer_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            df, _ = self.make_baseline(root)
            run_dir = root / "run"
            limitation = "Conflicting labels and cross-fold duplicate content remain unresolved."
            with patch.object(prepare, "ROOT", root), \
                 patch.object(prepare.subprocess, "check_output", side_effect=["commit\n", "branch\n"]), \
                 contextlib.redirect_stdout(io.StringIO()):
                prepare.main(["--run_dir", str(run_dir), "--tasks", "treated15", "--limitation", limitation])
            config = json.loads((run_dir / "run.json").read_text())
            self.assertEqual(list(config["tasks"]), ["treated15"])
            settings = config["tasks"]["treated15"]
            self.assertEqual(len(settings["class_names"]), 15)
            self.assertEqual(settings["candidates"], ["efficientnet_b0"])
            self.assertEqual(settings["benchmark"], "Benchmark_Treated_Condition_15class_QC")
            self.assertEqual(settings["n_samples"], len(df))
            self.assertFalse((run_dir / "treated").exists())
            assignments = []
            for fold in range(1, 6):
                path = run_dir / f"treated15/splits/fold_{fold}.csv"
                self.assertEqual(sha256(path), settings["splits"][str(fold)])
                split = pd.read_csv(path)
                assignments.append(split.role.to_numpy())
                audit = json.loads(path.with_suffix(".json").read_text())
                self.assertEqual(audit["legacy_outer_membership_check"], "exact_image_path_match")
                self.assertEqual(audit["group_overlap"], 0)
                for partition in audit["partitions"].values():
                    self.assertEqual(len(partition["class_counts"]), 15)
            self.assertTrue(np.all(np.sum([a == "test" for a in assignments], axis=0) == 1))
            source = "Code/Training/Rebuttal/run_nested_cv.sbatch"
            self.assertEqual(config["source_sha256"][source], sha256(run_dir / "source" / source))
            report = summarize(run_dir)
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["limitations"], [limitation])
            self.assertNotIn("primary", report["tasks"]["treated15"])
            markdown = (run_dir / "corrected_results.md").read_text()
            self.assertIn("Treated (15-class)", markdown)
            self.assertIn(limitation, markdown)

    def test_mismatched_legacy_outer_membership_rejected_before_run_created(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, baseline = self.make_baseline(root)
            path = baseline / "fold_1/val_predictions.csv"
            prior = pd.read_csv(path)
            prior.loc[0, "image_path"] = "wrong_image.png"
            prior.to_csv(path, index=False)
            with patch.object(prepare, "ROOT", root), \
                 self.assertRaisesRegex(ValueError, "outer fold 1 differs"):
                prepare.main(["--run_dir", str(root / "run"), "--tasks", "treated15"])
            self.assertFalse((root / "run").exists())

    def test_duplicate_task_request_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            prepare.main(["--run_dir", "/unused", "--tasks", "treated15", "treated15"])


class SlurmRoutingTests(unittest.TestCase):
    def dispatch(self, index, task=None):
        command = ["bash", str(ROOT / "Code/Training/Rebuttal/run_nested_cv.sbatch"), "/tmp/frozen run"]
        if task:
            command.append(task)
        env = {**os.environ, "SLURM_ARRAY_TASK_ID": str(index), "REBUTTAL_PYTHON": "/bin/echo"}
        return subprocess.run(command, env=env, capture_output=True, text=True, check=False)

    def test_single_cohort_indices_and_legacy_default_mapping(self):
        for index in range(1, 6):
            result = self.dispatch(index, "treated15")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"--task treated15 --fold {index}", result.stdout)
        for index, task, fold in ((1, "untreated", 1), (5, "untreated", 5),
                                  (6, "treated", 1), (10, "treated", 5)):
            result = self.dispatch(index)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(shlex.split(result.stdout)[-4:], ["--task", task, "--fold", str(fold)])

    def test_invalid_array_index_fails_before_python(self):
        for index, task in ((0, "treated15"), (6, "treated15"), (11, None), (0, None)):
            result = self.dispatch(index, task)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
