"""Verified prediction, content-provenance and complete error-evidence checks."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code/Training/Rebuttal"))
import audit_corrected_error_cases as audit
from audit_error_cases import add_decisions, export_assets
from validation import PROTOCOL, sha256

NAMES = ["E.coli5-alpha + Ampicillin", "E.coli5-alpha + CND", "E.coli5-alpha + Kanamycin"]


def population():
    labels, predictions = [2, 2, 0, 0, 1], [0, 2, 0, 2, 1]
    rows = []
    for i, (true, pred) in enumerate(zip(labels, predictions)):
        rows.append({"image_path": f"/images/{i}.png", "image_uid": f"uid-{i}", "class_id": true,
                     "class_name": NAMES[true], "canonical_strain": "E.coli5-alpha",
                     "canonical_treatment": NAMES[true].split(" + ")[1], "timepoint_min": 30,
                     "source_root": "collection", "source_folder": NAMES[true], "source_collection_id": NAMES[true],
                     "y_true": true, "y_pred": pred, "pred_class_name": NAMES[pred], "fold": 1,
                     "culture_id": "101", "preparation_id": "p" + str(true), "slide_id": "s" + str(true),
                     "acquisition_session_id": "a1", "batch_id": "UNKNOWN", "optical_modality": "UNKNOWN",
                     "mapping_status": audit.MATCHED, "metadata_row": i + 2,
                     "test_content_seen_in_train": False, "test_content_seen_in_validation": False,
                     **{"prob_" + n: .8 if j == pred else .1 for j, n in enumerate(NAMES)}})
    df = add_decisions(pd.DataFrame(rows), NAMES)
    df["evaluation_protocol"] = PROTOCOL
    df["audited_file_sha256"] = ["hash" + str(i) for i in range(len(df))]
    df["conflicting_condition_labels"] = False
    return df


class CorrectedEvidenceTests(unittest.TestCase):
    def test_zero_inclusive_confusions_conserve_population(self):
        df = population()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            audit.export_tables(df, NAMES, out, render=False)
            for path in out.glob("confusion_*.csv"):
                table = pd.read_csv(path)
                total = table["count"].sum() if "count" in table else table.iloc[:, 1:].to_numpy().sum()
                self.assertEqual(total, len(df), path.name)
            exact = pd.read_csv(out / "confusion_exact_by_time.csv")
            self.assertEqual(len(exact), 9)
            self.assertTrue(exact["count"].eq(0).any())

    def test_full_inventory_conflict_survives_subsetting(self):
        df = population()
        inventory = df[["image_path", "image_uid", "class_name"]].copy()
        inventory["original_sha256"] = ["shared", "b", "shared", "d", "e"]
        inventory["conflicting_condition_labels"] = [True, False, True, False, False]
        inventory["original_cohort_duplicate_size"] = [2, 1, 2, 1, 1]
        result = audit.attach_content(df.iloc[[0]], inventory)
        self.assertTrue(result.conflicting_condition_labels.iloc[0])
        self.assertNotIn("original_sha256", result)
        df["original_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "Frozen image hashes"):
            audit.attach_content(df, inventory)
        inventory.loc[0, "class_name"] = "wrong"
        with self.assertRaisesRegex(ValueError, "class_name"):
            audit.attach_content(df.drop(columns="original_sha256"), inventory)

    def test_overlap_uses_actual_development_partitions(self):
        df = population()
        df["fold"] = [1, 2, 3, 2, 3]
        df["audited_file_sha256"] = ["shared", "b", "shared", "d", "e"]
        splits = {}
        for fold in (1, 2, 3):
            splits[fold] = pd.DataFrame({"image_path": df.image_path,
                                        "role": np.where(df.fold.eq(fold), "test", "train")})
        splits[1].loc[2, "role"] = "validation"
        result = audit.annotate_partition_content(df, splits)
        self.assertTrue(result.loc[0, "test_content_seen_in_validation"])
        self.assertFalse(result.loc[0, "test_content_seen_in_train"])
        self.assertTrue(result.loc[2, "test_content_seen_in_train"])

    def test_controls_are_deterministic_and_do_not_relax_known_sessions(self):
        df = population()
        pairs = audit.select_controls(df)
        shuffled = audit.select_controls(df.sample(frac=1, random_state=3))
        pd.testing.assert_frame_equal(pairs, shuffled)
        row = pairs.loc[pairs.error_case_id.eq(df.case_id.iloc[0])].iloc[0]
        self.assertEqual(row.true_condition_control_id, df.case_id.iloc[1])
        self.assertEqual(row.predicted_condition_control_id, df.case_id.iloc[2])
        df.loc[1, "acquisition_session_id"] = "different"
        df.loc[2, "conflicting_condition_labels"] = True
        row = audit.select_controls(df).loc[lambda d: d.error_case_id.eq(df.case_id.iloc[0])].iloc[0]
        self.assertEqual(row.true_condition_control_id, "")
        self.assertEqual(row.predicted_condition_control_id, "")

    def test_complete_galleries_show_ids_conflicts_and_unchanged_originals(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            df = population()
            for i in df.index:
                path = out / f"image-{i}.png"
                Image.fromarray(np.arange(900, dtype=np.uint8).reshape(30, 30)).save(path)
                df.loc[i, "image_path"] = str(path)
                df.loc[i, "audited_file_sha256"] = sha256(path)
            df.loc[[0, 2], "conflicting_condition_labels"] = True
            pairs = audit.select_controls(df)
            requests = pd.concat([pairs, pd.DataFrame({"true_condition_control_id": [df.case_id.iloc[2]],
                                                      "predicted_condition_control_id": [""]})]).fillna("")
            result = export_assets(df, requests, out)
            audit.write_galleries(result, pairs, out)
            page = (out / "error_gallery.html").read_text()
            self.assertNotIn("Legacy CV", page)
            self.assertIn("preparation_id", page)
            self.assertIn("batch_id: UNKNOWN", page)
            self.assertIn("CONFLICTING CONDITION LABELS", page)
            for row in result.loc[result.any_error].itertuples():
                self.assertIn(row.case_id, page)
                self.assertEqual(sha256(out / row.original_file), sha256(Path(row.image_path)))
            conflict = (out / "conflict_gallery.html").read_text()
            self.assertIn(df.case_id.iloc[2], conflict)
            self.assertIn("No control satisfies", (out / "comparison_gallery.html").read_text())

    def test_focus_summary_separates_strain_and_condition_errors(self):
        df = population()
        df.loc[0, "conflicting_condition_labels"] = True
        result = audit.focus_summary(df)
        self.assertEqual(result["dh5_kan_n"], 2)
        self.assertEqual(result["dh5_kan_errors"], 1)
        self.assertEqual(result["dh5_kan_errors_with_conflicting_labels"], 1)
        self.assertEqual(result["dh5_kan_cross_strain_errors"], 0)

    def test_loader_rejects_legacy_or_tampered_frozen_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            config = {"protocol": "legacy"}
            (run / "run.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "internal-validation"):
                audit.load_outer_predictions(run, "task", {})
            config.update(protocol=PROTOCOL, tasks={"task": {"img_size": 224, "class_names": NAMES}},
                          source_sha256={"changed.py": "wrong"})
            (run / "source").mkdir()
            (run / "source/changed.py").write_text("changed")
            (run / "run.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "Frozen source changed"):
                audit.load_outer_predictions(run, "task", {})

    def test_preview_matches_evaluation_resize_and_normalization(self):
        import torchvision.transforms as T
        from utils.image_utils import load_image_as_pil_rgb
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            Image.fromarray(np.arange(400, dtype=np.uint8).reshape(20, 20)).save(path)
            image = load_image_as_pil_rgb(str(path))
            direct = T.Resize((224, 224))(image)
            preview = image.resize((224, 224), Image.Resampling.BILINEAR)
            np.testing.assert_array_equal(np.asarray(direct), np.asarray(preview))


if __name__ == "__main__":
    unittest.main()
