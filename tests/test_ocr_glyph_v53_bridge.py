# -*- coding: utf-8 -*-
"""V5.3 -> V5.5 bridge tests using synthetic data, NOT the actual 1259-task book."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.ocr_glyph_v53_to_v55 import convert, _read_package
from scripts.ocr_glyph_batch_v55 import build, read_jsonl


class V53BridgeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.zip_path = self.root / "v5_original.zip"
        rows = [
            {"id": "T00001", "c": ["魔上", "魔王"]},
            {"id": "T00002", "c": ["日", "目"]},
            {"id": "T00003", "c": ["三", "三"]},
        ]
        with zipfile.ZipFile(self.zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(
                "nested/04_ai_tasks_compact.jsonl",
                "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows),
            )
        self.audit_path = self.root / "1036_slot_audit.csv"
        self.write_audit([
            {"task_id": "T00001", "offset_a": "1", "pair": "上¦王",
             "status": "PROVISIONAL_NATIVE_PIXEL_MATCH", "source_page": "6.0",
             "col_id": "p00004:c006", "group_id": "V53G003",
             "pixel_sha256": "f" * 64},
            {"task_id": "T00002", "offset_a": "0", "pair": "日¦目",
             "status": "TEXT_INDEX_AMBIGUOUS", "source_page": "",
             "col_id": "", "group_id": "", "pixel_sha256": ""},
        ])

    def write_audit(self, rows):
        fields = [
            "task_id", "offset_a", "pair", "status", "source_page", "col_id",
            "group_id", "pixel_sha256",
        ]
        with self.audit_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_original_compact_zip_is_read_without_extracting(self):
        name, rows = _read_package(self.zip_path)
        self.assertEqual(name, "04_ai_tasks_compact.jsonl")
        self.assertEqual(len(rows), 3)

    def test_convert_matches_exact_v53_offsets(self):
        summary = convert(self.zip_path, self.audit_path, self.root / "out")
        self.assertEqual(summary["source_v5_task_count"], 3)
        self.assertEqual(summary["editable_conflict_task_count"], 2)
        self.assertEqual(summary["matched_v53_slot_count"], 2)
        self.assertEqual(summary["verified_bbox_count"], 0)
        data = read_jsonl(self.root / "out" / "GLYPH_JOBS_NEED_COORDINATES.jsonl")
        self.assertEqual(data[0]["a_index"], 1)
        self.assertEqual(data[0]["b_index"], 1)
        self.assertFalse(data[0]["position_verified"])
        self.assertEqual(data[0]["box"], [])
        self.assertEqual(data[0]["image_path"], "")
        self.assertEqual(data[0]["v53_group_id"], "V53G003")

    def test_mismatched_v53_candidate_pair_is_not_used(self):
        self.write_audit([{
            "task_id": "T00001", "offset_a": "1", "pair": "王¦上",
            "status": "PROVISIONAL_NATIVE_PIXEL_MATCH",
        }])
        report = convert(self.zip_path, self.audit_path, self.root / "out")
        self.assertEqual(report["matched_v53_slot_count"], 0)
        self.assertEqual(report["unmatched_or_changed_v53_slots"], 1)

    def test_stale_v53_offset_is_not_used(self):
        self.write_audit([{"task_id": "T00001", "offset_a": "5",
                           "pair": "上¦王", "status": "TEST"}])
        report = convert(self.zip_path, self.audit_path, self.root / "out")
        self.assertEqual(report["matched_v53_slot_count"], 0)
        self.assertEqual(report["unmatched_or_changed_v53_slots"], 1)

    def test_duplicate_audit_rows_rejected(self):
        duplicated = {"task_id": "T00001", "offset_a": "1", "pair": "上¦王", "status": "TEST"}
        self.write_audit([duplicated, duplicated])
        with self.assertRaises(ValueError):
            convert(self.zip_path, self.audit_path, self.root / "out")

    def test_original_frozen_native_v5_is_not_exported(self):
        native = self.root / "03_conflict_items.jsonl"
        native.write_text(
            json.dumps({"item_id": "T00001", "model_action_required": False,
                        "candidates": [{"raw_text": "魔上"}, {"raw_text": "魔王"}]},
                       ensure_ascii=False) + "\n", encoding="utf-8"
        )
        report = convert(native, self.audit_path, self.root / "out")
        self.assertEqual(report["editable_conflict_task_count"], 0)
        self.assertEqual(report["matched_v53_slot_count"], 0)

    def test_stale_candidate_digest_rejected_before_image_probing(self):
        converted = self.root / "stale-in"
        convert(self.zip_path, self.audit_path, converted)
        jobs_path = converted / "GLYPH_JOBS_NEED_COORDINATES.jsonl"
        jobs = read_jsonl(jobs_path)
        jobs[0]["candidate_sha256"] = "obsolete-candidates"
        jobs_path.write_text(
            "".join(json.dumps(job, ensure_ascii=False) + "\n" for job in jobs),
            encoding="utf-8",
        )
        result = build(
            conflicts_path=converted / "V5_EDITABLE_CONFLICTS.jsonl",
            jobs_path=jobs_path,
            out_dir=self.root / "stale-review",
        )
        rejected = read_jsonl(self.root / "stale-review" / "GLYPH_REJECTED_JOBS.jsonl")
        self.assertEqual(result["accepted_glyph_job_count"], 0)
        self.assertEqual(rejected[0]["reason"], "missing_or_stale_job_candidate_sha256")

    def test_missing_candidate_digest_rejected(self):
        converted = self.root / "missing-in"
        convert(self.zip_path, self.audit_path, converted)
        jobs_path = converted / "GLYPH_JOBS_NEED_COORDINATES.jsonl"
        jobs = read_jsonl(jobs_path)
        del jobs[0]["candidate_sha256"]
        jobs_path.write_text(
            "".join(json.dumps(job, ensure_ascii=False) + "\n" for job in jobs),
            encoding="utf-8",
        )
        build(
            conflicts_path=converted / "V5_EDITABLE_CONFLICTS.jsonl",
            jobs_path=jobs_path,
            out_dir=self.root / "missing-review",
        )
        rejected = read_jsonl(self.root / "missing-review" / "GLYPH_REJECTED_JOBS.jsonl")
        self.assertEqual(rejected[0]["reason"], "missing_or_stale_job_candidate_sha256")

    def test_unverified_jobs_cannot_generate_answers(self):
        converted = self.root / "out"
        convert(self.zip_path, self.audit_path, converted)
        result = build(
            conflicts_path=converted / "V5_EDITABLE_CONFLICTS.jsonl",
            jobs_path=converted / "GLYPH_JOBS_NEED_COORDINATES.jsonl",
            out_dir=self.root / "review",
        )
        self.assertEqual(result["full_answer_draft_count"], 0)
        self.assertEqual(result["accepted_glyph_job_count"], 0)
        self.assertEqual(result["rejected_glyph_job_count"], 2)


if __name__ == "__main__":
    unittest.main()
