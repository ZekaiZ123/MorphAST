"""Accounting conservation, missing-data distinctions, and metadata parser checks."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code/Training/Rebuttal"))
from audit_dataset_accounting import (
    ACQUISITION_IDS, METADATA_MATCH, aggregate_tile_metadata, attach_acquisition_metadata,
    classify_decisions, export_acquisition_accounting, read_xlsx_sheet,
    recompute_field_status, retention_table,
)
from validation import sha256


class AccountingTests(unittest.TestCase):
    def test_missing_segmentation_is_not_a_qc_failure(self):
        df = pd.DataFrame({"image_path": ["a", "b", "c", "d"], "strain": ["s"] * 4,
                           "seg_qc_status": ["pass", "fail", "missing", "pass"],
                           "drop_from_training": [0, 1, 1, 1]})
        df["decision"] = classify_decisions(df)
        self.assertEqual(df.decision.tolist(), ["retained", "qc_failed", "missing_segmentation", "other_exclusion"])
        result = retention_table(df, ["strain"]).iloc[0]
        self.assertEqual(result.n_before, 4)
        self.assertEqual(result.n_after, 1)
        self.assertEqual(result.n_excluded, 3)
        self.assertEqual(result.n_missing_segmentation, 1)

    def test_three_of_nine_tiles_fail_point34_threshold(self):
        policy = {"min_pass_tiles": 3, "min_pass_fraction": .34, "min_total_cells": 6,
                  "max_large_cell_tiles": 0, "max_artifact_tiles": 0}
        row = {"n_tiles": 9, "n_pass": 3, "pass_fraction": 3/9,
               "total_cells": 20, "large_cells": 0, "artifacts": 0}
        self.assertEqual(recompute_field_status(row, policy), ("fail", "low_pass_fraction:0.333"))
        self.assertEqual(recompute_field_status({"n_tiles": float("nan")}, policy), ("missing", "missing_segmentation"))

    def test_tile_counts_preserve_missing_replicate_and_report_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.csv"
            pd.DataFrame({"sample_id": ["field_r0_c0", "field_r0_c1", "field_r0_c1"],
                          "exclude_reason": ["", "low_cells:0", "low_cells:0"],
                          "num_cells": [7, 0, 0], "replicate": ["NA"] * 3}).to_csv(path, index=False)
            values, stats = aggregate_tile_metadata(path)
            self.assertEqual(values.loc["field", "total_cells"], 7)
            self.assertEqual(values.loc["field", "n_pass"], 1)
            self.assertEqual(stats["n_duplicate_tile_rows"], 1)
            self.assertEqual(stats["replicate_value_counts"], {"NA": 3})

    def test_xlsx_sparse_cells_do_not_shift_culture_ids(self):
        xml = '''<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>
        <row r="1"><c r="A1" t="inlineStr"><is><t>Filename</t></is></c><c r="B1" t="inlineStr"><is><t>Culture</t></is></c><c r="C1" t="inlineStr"><is><t>Session</t></is></c></row>
        <row r="2"><c r="A2" t="inlineStr"><is><t>image.tif</t></is></c><c r="C2"><v>109</v></c></row>
        </sheetData></worksheet>'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book.xlsx"
            with ZipFile(path, "w") as archive:
                archive.writestr("xl/worksheets/sheet1.xml", xml)
            result = read_xlsx_sheet(path, "xl/worksheets/sheet1.xml")
            self.assertEqual(result.iloc[0]["Session"], "109")
            self.assertNotIn("Culture", result.columns)


class AcquisitionAccountingTests(unittest.TestCase):
    def population(self):
        frame = pd.DataFrame({
            "benchmark": ["untreated_5class"] * 4,
            "image_path": [f"/images/{name}.tif" for name in "abcd"],
            "image_uid": list("abcd"), "canonical_strain": ["strain"] * 4,
            "canonical_treatment": ["Untreated"] * 4, "timepoint_min": [0] * 4,
            "decision": ["retained", "qc_failed", "missing_segmentation", "retained"],
            "seg_qc_reason": ["", "few_pass_tiles:0", "missing_segmentation", ""],
        })
        mapping = pd.DataFrame({
            "image_path": frame.image_path, "strain": ["strain"] * 4,
            "treatment": ["Control"] * 4, "timepoint": ["0"] * 4,
            "filename": [f"{name}.tif" for name in "abcd"],
            "mapping_status": [METADATA_MATCH] * 2 + ["no_exact_composite_match"] * 2,
            "metadata_row": [2, 3, None, None],
            "identity_evidence": ["composite_metadata_association_not_checksum_verified"] * 2 + ["unresolved"] * 2,
            "n_metadata_candidates": [1, 1, 0, 0], "n_local_key_candidates": [1] * 4,
            "local_width": [100] * 4, "metadata_width": [100] * 4,
            "local_height": [100] * 4, "metadata_height": [100] * 4,
            "header_error": [""] * 4, "metadata_conflicts": [""] * 4,
            **{key: ["101", "101", "UNKNOWN", "UNKNOWN"] for key in ACQUISITION_IDS},
        })
        return frame, mapping

    def fixture(self, root):
        frame, mapping = self.population()
        source = root / "evidence.txt"
        source.write_text("unchanged source")
        audit, out = root / "metadata", root / "accounting"
        audit.mkdir()
        out.mkdir()
        mapping.to_csv(audit / "image_metadata_mapping.csv", index=False)
        summary = {"scid_used": False, "n_local_images_before_qc": 4,
                   "source_evidence": [{"path": str(source), "sha256": sha256(source)}],
                   "cohorts": {"untreated_5class": {"n_retained": 2, "n_matched": 1, "levels": {
                       key: {"n_known_images": 1, "n_recorded_identifiers": 1} for key in ACQUISITION_IDS}}},
                   "workbook_identifiers": {key: {"n_distinct": 1} for key in ACQUISITION_IDS}}
        (audit / "summary.json").write_text(json.dumps(summary))
        (audit / "verification.json").write_text(json.dumps({"scid_used": False, "all_checks_passed": True}))
        return audit, out, {"untreated_5class": frame}

    def test_identifiers_do_not_change_population_or_infer_replicates(self):
        frame, mapping = self.population()
        result = attach_acquisition_metadata(frame, mapping)
        self.assertEqual(result.image_path.tolist(), frame.image_path.tolist())
        self.assertEqual(result.decision.tolist(), frame.decision.tolist())
        self.assertEqual(result.experimental_replicate.unique().tolist(), ["UNKNOWN"])
        self.assertTrue(result.independent_biological_replicates.eq("NOT_ESTABLISHED").all())
        table = retention_table(result, ["culture_id"]).set_index("culture_id")
        self.assertEqual(table.loc["101", "n_before"], 2)
        self.assertEqual(table.loc["UNKNOWN", "n_after"], 1)
        self.assertEqual(table.loc["UNKNOWN", "n_missing_segmentation"], 1)

    def test_join_rejects_missing_or_duplicate_paths(self):
        frame, mapping = self.population()
        for changed in (mapping.iloc[:-1], pd.concat([mapping, mapping.iloc[:1]])):
            with self.assertRaises(ValueError):
                attach_acquisition_metadata(frame, changed)

    def test_join_rejects_changed_condition_time_filename_or_evidence(self):
        frame, mapping = self.population()
        for column, value in (("strain", "other"), ("treatment", "Ampicillin"), ("timepoint", "5"),
                              ("filename", "other.tif"), ("identity_evidence", "verified_biological_repeat"),
                              ("metadata_row", None)):
            changed = mapping.copy()
            changed.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                attach_acquisition_metadata(frame, changed)

    def test_candidate_identifiers_cannot_be_assigned_to_unmatched_images(self):
        frame, mapping = self.population()
        mapping.loc[2, "culture_id"] = "999"
        with self.assertRaisesRegex(ValueError, "Unaccepted"):
            attach_acquisition_metadata(frame, mapping)

    def test_export_conserves_counts_at_all_identifier_levels(self):
        with tempfile.TemporaryDirectory() as directory:
            audit, out, frames = self.fixture(Path(directory))
            evidence = []
            result = export_acquisition_accounting(audit, frames, out, evidence)
            self.assertFalse(result["scid_used"])
            self.assertEqual(len(evidence), 4)
            for name in [*ACQUISITION_IDS, "acquisition_hierarchy"]:
                table = pd.read_csv(out / f"retention_by_{name}.csv")
                self.assertEqual(table.n_before.sum(), 4)
                self.assertEqual(table.n_after.sum(), 2)
                self.assertEqual(table.n_qc_failed.sum(), 1)
                self.assertEqual(table.n_missing_segmentation.sum(), 1)
            coverage = pd.read_csv(out / "acquisition_metadata_coverage.csv").set_index("stage")
            self.assertEqual(coverage.loc["after_qc", "n_unknown_culture_id"], 1)
            ledger = pd.read_csv(out / "field_accounting_with_acquisition_ids.csv")
            self.assertEqual(len(ledger), 4)
            self.assertFalse(any("scid" in name.lower() for name in ledger.columns))

    def test_export_rejects_changed_source_or_failed_verification(self):
        for mode in ("source", "verification", "scid"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                audit, out, frames = self.fixture(root)
                if mode == "source":
                    (root / "evidence.txt").write_text("changed")
                else:
                    (audit / "verification.json").write_text(json.dumps({
                        "scid_used": mode == "scid", "all_checks_passed": mode != "verification"}))
                with self.assertRaises(ValueError):
                    export_acquisition_accounting(audit, frames, out, [])

    def test_export_rejects_nonunique_or_inconsistent_accepted_matches(self):
        for column, value in (("n_metadata_candidates", 2), ("n_local_key_candidates", 2),
                              ("metadata_width", 99), ("local_height", -1),
                              ("header_error", "unreadable"), ("metadata_conflicts", "wrong_ids")):
            with self.subTest(column=column), tempfile.TemporaryDirectory() as directory:
                audit, out, frames = self.fixture(Path(directory))
                path = audit / "image_metadata_mapping.csv"
                mapping = pd.read_csv(path)
                if isinstance(value, str):
                    mapping[column] = mapping[column].astype(object)
                mapping.loc[0, column] = value
                mapping.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    export_acquisition_accounting(audit, frames, out, [])


if __name__ == "__main__":
    unittest.main()
