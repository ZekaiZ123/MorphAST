"""The source-only MorphAST port must run independently of the original checkout."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "Code/Training/Rebuttal"


class MorphASTPortTests(unittest.TestCase):
    def test_entrypoints_load_without_data_or_original_pythonpath(self):
        names = ("prepare_nested_cv", "run_nested_cv", "summarize_nested_cv",
                 "corrected_treated15", "plot_corrected_treated15",
                 "audit_dataset_accounting", "prepare_preparation_sensitivity",
                 "export_publication_evidence")
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["HF_HUB_OFFLINE"] = "1"
        with tempfile.TemporaryDirectory() as directory:
            env["MPLCONFIGDIR"] = directory
            for name in names:
                with self.subTest(entrypoint=name):
                    result = subprocess.run([sys.executable, "-E", "-s", str(WORKFLOW / (name + ".py")), "--help"],
                        cwd=directory, env=env, capture_output=True, text=True, timeout=90)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout)

    def test_internal_modules_resolve_to_this_checkout(self):
        script = (
            "import sys; from pathlib import Path; "
            f"root=Path({str(ROOT)!r}); sys.path.insert(0, str(root/'Code/Training/Rebuttal')); "
            "import run_nested_cv, audit_dataset_accounting, corrected_treated15, train_0min_fiveclass_raw; "
            "from utils import image_utils, identity_audit_gate; "
            "modules=(run_nested_cv,audit_dataset_accounting,corrected_treated15,"
            "train_0min_fiveclass_raw,image_utils,identity_audit_gate); "
            "assert all(Path(m.__file__).resolve().is_relative_to(root) for m in modules)"
        )
        result = subprocess.run([sys.executable, "-I", "-c", script],
                                capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_hashed_compatibility_dependencies_are_present(self):
        for relative in (
            "Code/Training/Model_Training_5_class/train_0min_fiveclass_raw.py",
            "Code/Training/Model_Training_5_class/build_0min_strain_manifest.py",
            "Code/Training/Model_Training_Treated_Condition/build_treated_condition_manifest.py",
            "Code/Training/Model_Training_Treated_Condition/build_treated_segmentation_qc_manifest.py",
            "Code/Training/Model_Training_Treated_Condition/run_15class_seg_qc.sbatch",
            "Code/Training/Model_Training_Treated_Condition/train_treated_condition_raw.py",
            "Code/Segementation/run_tiles_with_stats_tif_meta_data_v2.py",
            "Code/Training/Strain_Recognition_v1/provenance_workflow.py",
            "Code/utils/image_utils.py", "Code/utils/data_utils.py",
        ):
            with self.subTest(path=relative):
                self.assertTrue((ROOT / relative).is_file())

    @unittest.skipUnless(shutil.which("git"), "Git required to check the source-only publication boundary")
    def test_generated_artifacts_are_ignored_and_not_tracked(self):
        if not (ROOT / ".git").exists():
            self.skipTest("Source archive has no Git index")
        examples = ["Data/example.csv", "Model/example/best.pt", "outputs/rebuttal/example.csv",
                    "experiments/strain_recognition_v1/audit_status.json", ".worktrees/example/file.py",
                    "fold_1_features.npz"]
        result = subprocess.run(["git", "check-ignore", "--stdin"], cwd=ROOT,
            input="\n".join(examples) + "\n", capture_output=True, text=True, check=True)
        self.assertEqual(set(result.stdout.splitlines()), set(examples))
        tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
        forbidden = ("Data/", "Model/", "outputs/", "output/", "experiments/", ".worktrees/", "Paper/")
        self.assertFalse([name for name in tracked if name.startswith(forbidden)])


if __name__ == "__main__":
    unittest.main()
