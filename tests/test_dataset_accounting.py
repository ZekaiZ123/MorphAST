"""Accounting conservation, missing-data distinctions, and metadata parser checks."""

import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code/Training/Rebuttal"))
from audit_dataset_accounting import (
    aggregate_tile_metadata, classify_decisions, read_xlsx_sheet,
    recompute_field_status, retention_table,
)


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


if __name__ == "__main__":
    unittest.main()
