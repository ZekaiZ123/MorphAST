"""Conservative metadata matching, unknown handling, and frozen-fold overlap."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code/Training/Rebuttal"))
from audit_experimental_independence import (
    IDENTIFIERS, KEYS, MATCHED, UNKNOWN, audit_folds, hierarchy_conflicts,
    identifier_counts, match_metadata, normalize_metadata, overlap_table,
    read_workbook, split_feasibility,
)
from validation import sha256, write_json


def metadata_record(filename="a.tif", **updates):
    record = {"excel_row": 2, "Relative Path": f"5-alpha/Ampicillin/5_min/AcquisitionID_10_Date_010126/CultureID_01/SCID_999/PreparationID_20/SlideID_30/{filename}",
              "Full Path": f"D:/provider/{filename}", "Strain": "5-alpha", "Treatment": "Ampicillin",
              "Timepoint": "5_min", "Filename": filename, "Image Width (px)": "8", "Image Height (px)": "6",
              "Culture ID": "01", "PreperationID": "20", "Slide ID": "30", "Acquisition Session": "10",
              "Acquisition Date": "", "Microscope": "Instrument"}
    record.update(updates)
    return record


def local_records():
    return pd.DataFrame([{"image_path": "/local/a.tif", "strain": "E.coli5-alpha",
                          "treatment": "Ampicillin", "timepoint": "5", "filename": "a.tif"}])


def dimensions():
    return pd.DataFrame([{"image_path": "/local/a.tif", "local_width": 8,
                          "local_height": 6, "header_error": ""}])


class MetadataTests(unittest.TestCase):
    def test_sparse_workbook_preserves_excel_row_and_sheet_relationship(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book.xlsx"
            with ZipFile(path, "w") as archive:
                archive.writestr("xl/workbook.xml", '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Image Metadata" r:id="r1"/></sheets></workbook>')
                archive.writestr("xl/_rels/workbook.xml.rels", '<Relationships><Relationship Id="r1" Target="worksheets/sheet9.xml"/></Relationships>')
                archive.writestr("xl/worksheets/sheet9.xml", '''<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>
                <row r="1"><c r="A1" t="inlineStr"><is><t>Filename</t></is></c><c r="B1" t="inlineStr"><is><t>Culture ID</t></is></c><c r="C1" t="inlineStr"><is><t>Slide ID</t></is></c></row>
                <row r="7"><c r="A7" t="inlineStr"><is><t>a.tif</t></is></c><c r="C7"><v>30</v></c></row>
                </sheetData></worksheet>''')
            records, _ = read_workbook(path)
            self.assertEqual(records.iloc[0].excel_row, 7)
            self.assertEqual(records.iloc[0]["Culture ID"], "")
            self.assertEqual(records.iloc[0]["Slide ID"], "30")

    def test_scid_excluded_and_missing_identifiers_not_imputed(self):
        absent = metadata_record("b.tif", excel_row=3, **{
            "Relative Path": "5-alpha/Ampicillin/5_min/b.tif", "Culture ID": "", "Slide ID": "",
            "PreperationID": "", "Acquisition Session": ""})
        records = normalize_metadata(pd.DataFrame([metadata_record(), absent]))
        self.assertFalse(any("scid" in c.lower() for c in records.columns))
        self.assertEqual(records.iloc[0].culture_id, "01")
        self.assertEqual(records.iloc[0].acquisition_date, "2026-01-01")
        self.assertEqual(records.iloc[1].culture_id, UNKNOWN)
        counted = identifier_counts(records, ["strain"]).iloc[0]
        self.assertEqual(counted.n_recorded_culture_id, 1)
        self.assertEqual(counted.n_unknown_culture_id, 1)
        self.assertEqual(counted.independent_biological_replicates, "NOT_ESTABLISHED")

    def test_identifier_disagreement_blocks_mapping(self):
        records = normalize_metadata(pd.DataFrame([metadata_record(**{"Culture ID": "02"})]))
        result = match_metadata(local_records(), records, dimensions()).iloc[0]
        self.assertEqual(result.mapping_status, "metadata_conflict")
        self.assertEqual(result.culture_id, UNKNOWN)
        self.assertEqual(result.candidate_culture_id, "02")

    def test_dimensions_and_unique_keys_required(self):
        metadata = normalize_metadata(pd.DataFrame([metadata_record()]))
        result = match_metadata(local_records(), metadata, dimensions()).iloc[0]
        self.assertEqual(result.mapping_status, MATCHED)
        self.assertEqual(result.biological_replicate_id, UNKNOWN)
        wrong = dimensions()
        wrong["local_width"] = 9
        self.assertEqual(match_metadata(local_records(), metadata, wrong).iloc[0].mapping_status, "dimension_mismatch")
        duplicated = pd.concat([metadata, metadata.assign(metadata_row=3)], ignore_index=True)
        self.assertEqual(match_metadata(local_records(), duplicated, dimensions()).iloc[0].mapping_status, "ambiguous_workbook_key")
        duplicate_local = pd.concat([local_records(), local_records().assign(image_path="/other/a.tif")], ignore_index=True)
        self.assertTrue(match_metadata(duplicate_local, metadata, dimensions()).culture_id.eq(UNKNOWN).all())

    def test_treatment_or_missing_time_not_silently_reassigned(self):
        metadata = normalize_metadata(pd.DataFrame([metadata_record()]))
        for column, value in (("treatment", "Kanamycin"), ("timepoint", "0"), ("filename", "a (2).tif")):
            local = local_records()
            local[column] = value
            row = match_metadata(local, metadata, dimensions()).iloc[0]
            self.assertEqual(row.mapping_status, "no_exact_composite_match")
            self.assertEqual(row.preparation_id, UNKNOWN)

    def test_parent_conflicts_detected_without_splitting_identifier(self):
        metadata = normalize_metadata(pd.DataFrame([metadata_record()]))
        extra = metadata.copy()
        extra["culture_id"] = "02"
        result = hierarchy_conflicts(pd.concat([metadata, extra]))
        self.assertTrue(((result.identifier_type == "preparation_id") & (result.inconsistent_parent == "culture_id")).any())


class OverlapTests(unittest.TestCase):
    def test_unknowns_excluded_and_technical_overlap_not_biological_replicates(self):
        df = pd.DataFrame({"role": ["train", "train", "test", "test"], "group_id": ["a", "b", "c", "d"],
                           **{name: ["1", UNKNOWN, "1", UNKNOWN] for name in IDENTIFIERS}})
        table = overlap_table(df, "role", 1, "original_cv", [(["train"], "test", "train_vs_heldout")])
        field = table.loc[table.level.eq("group_id")].iloc[0]
        prep = table.loc[table.level.eq("preparation_id")].iloc[0]
        self.assertEqual(field.n_shared_identifiers, 0)
        self.assertEqual(prep.n_shared_identifiers, 1)
        self.assertEqual(prep.n_right_known, 1)
        self.assertEqual(prep.n_right_unknown, 1)
        self.assertEqual(prep.fraction_known_right_shared, 1.)
        df["class_name"] = "condition"
        feasibility = split_feasibility(df)
        self.assertTrue(feasibility.n_recorded_groups.eq(1).all())
        self.assertFalse(feasibility.two_partition_class_coverage_possible_on_known_subset.any())

    def test_saved_outer_folds_verified_and_nested_overlap_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "Model/Rebuttal_CV/test_run"
            task = run / "treated15"
            task.mkdir(parents=True)
            qc = pd.DataFrame([{"image_path": f"/{i}.tif", "group_id": f"field_{i}", "class_id": 0,
                                "class_name": "condition", "retained": True, "strain": "strain",
                                "treatment": "treatment", "timepoint": "5", "filename": f"{i}.tif"}
                               for i in range(5)])
            qc.to_csv(task / "manifest.csv", index=False)
            settings = {"manifest_sha256": sha256(task / "manifest.csv"), "splits": {}}
            (task / "splits").mkdir()
            for fold in range(1, 6):
                split = qc[["image_path", "group_id", "class_id"]].copy()
                split["role"] = "train"
                split.loc[fold - 1, "role"] = "test"
                split.loc[fold % 5, "role"] = "validation"
                path = task / f"splits/fold_{fold}.csv"
                split.to_csv(path, index=False)
                settings["splits"][str(fold)] = sha256(path)
            write_json(run / "run.json", {"tasks": {"treated15": settings}})
            baseline = root / "Model/benchmark_QC/efficientnet_b0"
            baseline.mkdir(parents=True)
            prior = qc[["image_path"]].copy()
            prior["fold"] = range(1, 6)
            prior.to_csv(baseline / "crossval_predictions.csv", index=False)
            mapping = qc[["image_path", *KEYS]].copy()
            for name in IDENTIFIERS:
                mapping[name] = "same_experiment"
            mapping["mapping_status"] = MATCHED
            dest = root / "out"
            dest.mkdir()
            spec = ("benchmark", "unused", "test_run", "treated15")
            report = audit_folds(root, "treated_15class", spec, qc, mapping, dest, [])
            self.assertEqual(report["legacy_membership_check"], "exact_match_to_original_saved_predictions")
            self.assertEqual(report["levels"]["preparation_id"]["n_identifiers_spanning_outer_folds"], 1)
            self.assertEqual(report["levels"]["culture_id"]["n_known_images_exposed_to_same_identifier_in_other_outer_folds"], 5)
            prior.loc[0, "fold"] = 2
            prior.to_csv(baseline / "crossval_predictions.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Original predictions disagree"):
                audit_folds(root, "treated_15class", spec, qc, mapping, dest, [])


if __name__ == "__main__":
    unittest.main()
