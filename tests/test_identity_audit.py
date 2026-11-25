"""Scientific boundary tests for inventory, leakage constraints, and sealed evaluation."""
import datetime as dt
import io
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "Code"))
from utils.identity_audit import (
    LEDGER_FIELDS, SCHEMA, class_mapping, eligible_member, inspect_image,
    overlap_ledger, write_csv, write_json,
)
from utils.identity_audit_gate import (
    ARTIFACTS, AuditGateError, assert_external_evaluation_ready,
    assert_training_ready, digest, population_digest, rows_from, reject_v1_in_legacy_trainer,
)


def image_bytes(index=0, *, x=None, y=0, fmt="PNG", preview=False):
    """Small camera export with real, parseable embedded acquisition metadata."""
    captured = dt.datetime(2026, 1, 1) + dt.timedelta(seconds=index)
    ticks = int((captured - dt.datetime(1, 1, 1)).total_seconds()) * 10_000_000
    x = index * 1000 if x is None else x
    xml = (f'<SingleFileProperty><Image><ShootingDateTime>{ticks}</ShootingDateTime></Image>'
           f'<Shooting><StageLocationX>{x}</StageLocationX><StageLocationY>{y}</StageLocationY>'
           '<StageLocationZ>0</StageLocationZ><Channel>Channel4</Channel>'
           '<XyStageRegion><Width>100</Width><Height>100</Height></XyStageRegion>'
           '</Shooting></SingleFileProperty>')
    stream = io.BytesIO()
    image = Image.new("RGB", (32, 24), (index % 255, 30, 50))
    if preview:
        image.save(stream, format="TIFF", save_all=True, append_images=[Image.new("RGB", (160, 120))])
    else:
        image.save(stream, format=fmt)
    return stream.getvalue() + xml.encode()


class InventoryTests(unittest.TestCase):
    def test_appledouble_and_preview_exclusion(self):
        for name in ("__MACOSX/a.jpg", "Dataset_B/Strain_001/._image.tif", "directory/"):
            self.assertFalse(eligible_member(zipfile.ZipInfo(name)))
        self.assertTrue(eligible_member(zipfile.ZipInfo("Dataset_B/Strain_001/image.tif")))
        with self.assertRaises(ValueError):
            eligible_member(zipfile.ZipInfo("../image.jpg"))
        row = inspect_image(image_bytes(preview=True), "test.tif", "Dataset_B", 0)
        self.assertEqual((row["width"], row["height"]), (32, 24))
        self.assertEqual(json.loads(row["secondary_pages_json"]),
                         [{"page": 1, "width": 160, "height": 120, "mode": "RGB", "is_preview": True}])

    def test_original_biological_labels_are_not_all_strains(self):
        classes = class_mapping()["classes"]
        self.assertEqual(classes[1]["strain"], "5-Alpha")
        self.assertEqual(classes[1]["resistance_phenotype"], "AmpR")
        self.assertEqual(classes[3]["pathotype"], "UPEC")
        self.assertIsNone(classes[3]["strain"])
        self.assertIsNone(classes[6]["species"])
        self.assertTrue(classes[6]["organism_verification_required"])

    def test_overlap_crosses_filename_series_and_byte_encoding(self):
        first = inspect_image(image_bytes(1, x=0), "A_00001_CH4.png", "Dataset_B", 0)
        second = inspect_image(image_bytes(2, x=70, fmt="JPEG"), "B_00001_CH4.jpg", "Dataset_B", 0)
        self.assertNotEqual(first["sha256"], second["sha256"])
        ledger = []
        overlap_ledger([first, second], ledger.append)
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0]["overlap_fraction_a"], 0.3)
        self.assertEqual(ledger[0]["evidence_status"], "candidate")
        self.assertEqual(ledger[0]["resolution"], "unresolved")
        self.assertNotEqual(first["candidate_spatial_group_id"], second["candidate_spatial_group_id"])
        self.assertIsNone(first["replicate_id"])

    def test_recompression_same_field_without_hash_match(self):
        records = [inspect_image(image_bytes(1, x=0, fmt=fmt), f"a.{fmt}", "Dataset_B", 0)
                   for fmt in ("PNG", "JPEG")]
        ledger = []
        overlap_ledger(records, ledger.append)
        self.assertEqual(ledger[0]["overlap_fraction_a"], 1)
        self.assertEqual(ledger[0]["same_capture_metadata"], "true")
        self.assertNotEqual(records[0]["sha256"], records[1]["sha256"])

    def test_exact_legacy_match_retained_and_tagged(self):
        data = image_bytes()
        records = [inspect_image(data, "incoming.tif", "Dataset_B", 0),
                   inspect_image(data, "legacy.tif", "Legacy_training", 0)]
        edges = []
        overlap_ledger(records, edges.append)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["legacy_exact_match_count"], 1)
        self.assertEqual(records[0]["legacy_exposure_status"], "exact_legacy_training_match")
        self.assertIn("exact_sha256", {e["relationship"] for e in edges})

    def test_capture_dates_not_filename_dates_and_unknown_provenance(self):
        row = inspect_image(image_bytes(), "20260721_wrong.tif", "External_Test_Culture_Free", 5)
        self.assertEqual(row["acquisition_date"], "2026-01-01")
        self.assertEqual(row["capture_timezone_status"], "unverified")
        self.assertIsNone(row["acquisition_session_id"])
        self.assertEqual((row["split"], row["evaluation_status"]), ("reserved_external", "sealed"))

    def test_deterministic_capture_ids(self):
        data = image_bytes()
        self.assertEqual(inspect_image(data, "same", "Dataset_B", 0),
                         inspect_image(data, "same", "Dataset_B", 0))



if __name__ == "__main__":
    unittest.main()
