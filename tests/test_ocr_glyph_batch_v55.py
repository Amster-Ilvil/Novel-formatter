# -*- coding: utf-8 -*-
"""Safety regressions for V5.5. Run: python -m unittest discover -s tests -p test_ocr_glyph_batch_v55.py -v"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from engine.ocr_glyph_batch_v55 import (
    candidate_digest,
    compose_complete_sentence,
    exact_glyph_groups,
    make_glyph_proof,
    make_substitution_slots,
    map_column_characters,
    propose_slot,
    summarize_batch,
)
from adapters.vertical_glyph_grid_v55 import suggest_vertical_cells


class V55GlyphBatchSafety(unittest.TestCase):
    def setUp(self):
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            self.skipTest("Pillow is optional for pixel-based tests")
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        for index in range(2):
            image = Image.new("L", (60, 60), 255)
            draw = ImageDraw.Draw(image)
            draw.rectangle((20, 18, 30, 32), fill=0)
            if index == 1:
                # Change source image bytes OUTSIDE the exact glyph crop.
                draw.point((0, 0), fill=123)
            image.save(self.root / f"column-{index}.png")

    def proof(self, task="T01", index=0, verified=True):
        return make_glyph_proof(
            self.root / f"column-{index}.png",
            task_id=task, page_id=f"page-{index}",
            column_id=f"col-{index}", box=(10, 10, 40, 40),
            position_verified=verified,
        )

    def decision(self, proof, slot, char="王"):
        return {
            "approved": True,
            "glyph_key": proof.glyph_key,
            "character": char,
            "review_note": "已核对原稿笔画和句子语境，确认该印刷字形",
            "allowed_task_ids": [proof.task_id],
            "candidate_sha256_by_task": {
                proof.task_id: slot["candidate_sha256"],
            },
            "proof_by_task": {
                proof.task_id: {
                    "source_sha256": proof.source_sha256,
                    "page_id": proof.page_id,
                    "column_id": proof.column_id,
                    "box": list(proof.box),
                },
            },
        }

    def test_exact_raw_pixel_group_independent_sources(self):
        a, b = self.proof("T01", 0), self.proof("T02", 1)
        self.assertEqual(a.pixels_sha256, b.pixels_sha256)
        self.assertNotEqual(a.source_sha256, b.source_sha256)
        groups, rejected = exact_glyph_groups([a, b])
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["member_count"], 2)
        self.assertEqual(rejected, [])
        self.assertEqual(summarize_batch([a, b], groups)["potential_repeated_visual_checks_avoided"], 1)

    def test_identical_source_with_forged_page_id_not_counted_twice(self):
        a = self.proof("T01", 0)
        b = make_glyph_proof(
            self.root / "column-0.png", task_id="T02",
            page_id="different-page-name", column_id="other-col-name",
            box=a.box, position_verified=True,
        )
        groups, dupes = exact_glyph_groups([a, b])
        self.assertEqual(groups, [])
        self.assertEqual(len(dupes), 1)

    def test_blank_crop_rejected(self):
        with self.assertRaises(ValueError):
            make_glyph_proof(
                self.root / "column-0.png", task_id="X",
                page_id="0", column_id="0", box=(40, 40, 58, 58),
            )

    def test_out_of_bounds_rejected(self):
        with self.assertRaises(ValueError):
            make_glyph_proof(
                self.root / "column-0.png", task_id="X",
                page_id="0", column_id="0", box=(-1, 0, 20, 20),
            )

    def test_two_substitutions_in_one_sentence_are_distinct(self):
        texts = ["日上だ。", "目王だ。"]
        slots = make_substitution_slots("T01", texts)
        self.assertEqual(len(slots), 2)
        self.assertEqual([s["a"] for s in slots], ["日", "上"])
        proposals = [
            {"task_id": "T01", "a_index": s["a_index"], "b_index": s["b_index"],
             "candidate_sha256": s["candidate_sha256"], "character": s["b"],
             "status": "suggested"}
            for s in slots
        ]
        done = compose_complete_sentence("T01", texts, proposals)
        self.assertTrue(done["complete"])
        self.assertEqual(done["selected_text"], "目王だ。")

    def test_partial_composite_never_yields_full_answer(self):
        texts = ["日上だ。", "目王だ。"]
        slot = make_substitution_slots("T01", texts)[0]
        one = [{"task_id": "T01", "a_index": slot["a_index"],
                "b_index": slot["b_index"], "candidate_sha256": slot["candidate_sha256"],
                "character": slot["b"], "status": "suggested"}]
        done = compose_complete_sentence("T01", texts, one)
        self.assertFalse(done["complete"])
        self.assertEqual(len(done["missing"]), 1)

    def test_insert_delete_kept_unresolved(self):
        result = compose_complete_sentence("T01", ["日", "日々"], [])
        self.assertFalse(result["complete"])

    def test_independent_third_model_conflict_cannot_be_ignored(self):
        result = compose_complete_sentence("T01", ["日", "目", "口"], [])
        self.assertFalse(result["complete"])
        self.assertEqual(result["reason"], "independent_third_candidate_needs_its_own_alignment")

    def test_no_conflict_is_not_an_editable_answer(self):
        self.assertFalse(compose_complete_sentence("T01", ["魔王", "魔王"], [])["complete"])

    def test_group_reuse_requires_verified_coordinates(self):
        proof = self.proof(verified=False)
        slot = make_substitution_slots("T01", ["魔上", "魔王"])[0]
        r = propose_slot(slot, proof, self.decision(proof, slot))
        self.assertEqual(r["reason"], "physical_position_not_verified")

    def test_group_reuse_requires_provenance_binding(self):
        proof = self.proof()
        slot = make_substitution_slots("T01", ["魔上", "魔王"])[0]
        d = self.decision(proof, slot)
        del d["proof_by_task"]
        self.assertEqual(propose_slot(slot, proof, d)["reason"], "no_physical_provenance_binding")

    def test_group_reuse_rejects_source_change(self):
        proof = self.proof()
        slot = make_substitution_slots("T01", ["魔上", "魔王"])[0]
        d = self.decision(proof, slot)
        d["proof_by_task"]["T01"]["source_sha256"] = "not-the-original-scan"
        self.assertEqual(propose_slot(slot, proof, d)["reason"], "physical_provenance_changed")

    def test_group_reuse_rejects_changed_ocr_candidates(self):
        proof = self.proof()
        slot = make_substitution_slots("T01", ["魔上", "魔王"])[0]
        d = self.decision(proof, slot)
        d["candidate_sha256_by_task"]["T01"] = "old-ocr"
        self.assertEqual(propose_slot(slot, proof, d)["reason"], "ocr_candidates_changed")

    def test_group_reuse_refuses_unapproved_character(self):
        proof = self.proof()
        slot = make_substitution_slots("T01", ["魔上", "魔王"])[0]
        d = self.decision(proof, slot, char="目")
        self.assertEqual(propose_slot(slot, proof, d)["reason"], "not_an_existing_single_character_candidate")

    def test_unicode_homograph_requires_explicit_editorial_policy(self):
        proof = self.proof()
        slot = make_substitution_slots("T01", ["ニノ", "二ノ"])[0]
        d = self.decision(proof, slot, char="ニ")
        self.assertEqual(propose_slot(slot, proof, d)["reason"], "same_shape_different_unicode_requires_editorial_policy")
        d["codepoint_policy"] = "explicit_book_editorial"
        self.assertEqual(propose_slot(slot, proof, d)["status"], "suggested")

    def test_context_exception_blocks_reuse(self):
        proof = self.proof()
        slot = make_substitution_slots("T01", ["魔上", "魔王"])[0]
        d = self.decision(proof, slot)
        d["exceptions"] = ["T01"]
        self.assertEqual(propose_slot(slot, proof, d)["reason"], "exception")

    def test_approved_nonambiguous_reuse_is_only_a_draft(self):
        proof = self.proof()
        slot = make_substitution_slots("T01", ["魔上", "魔王"])[0]
        result = propose_slot(slot, proof, self.decision(proof, slot))
        self.assertEqual(result["status"], "suggested")
        complete = compose_complete_sentence("T01", ["魔上", "魔王"], [result])
        self.assertTrue(complete["complete"])
        self.assertEqual(complete["selected_text"], "魔王")
        self.assertEqual(complete["status"], "draft_only_not_written_back")

    def test_equal_length_mapping_still_requires_visual_check(self):
        result = map_column_characters("魔王様", cell_count=3)
        self.assertTrue(result["ok"])
        self.assertTrue(result["requires_visual_check"])
        self.assertEqual(result["mapping"], {0: 0, 1: 1, 2: 2})

    def test_unequal_length_without_anchor_is_rejected(self):
        result = map_column_characters("魔王様", cell_count=4)
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "unequal_length_no_anchors")

    def test_anchored_gaps_are_not_guessed(self):
        result = map_column_characters(
            "魔王様", cell_count=4, verified_anchors={0: 0, 2: 3},
        )
        self.assertTrue(result["ok"])
        self.assertTrue(result["requires_visual_check"])
        self.assertNotIn(1, result["mapping"])

    def test_nonmonotonic_anchor_rejected(self):
        result = map_column_characters(
            "魔王様", cell_count=3, verified_anchors={0: 2, 2: 0},
        )
        self.assertFalse(result["ok"])

    def test_projection_not_a_column(self):
        self.assertFalse(suggest_vertical_cells(self.root / "column-0.png")["ok"])

    def test_projection_blank_column(self):
        from PIL import Image
        path = self.root / "blank.png"
        Image.new("L", (40, 400), 255).save(path)
        self.assertFalse(suggest_vertical_cells(path)["ok"])

    def test_unapproved_decisions_never_propagate(self):
        proof = self.proof()
        slot = make_substitution_slots("T01", ["魔上", "魔王"])[0]
        d = self.decision(proof, slot)
        d["approved"] = False
        self.assertEqual(propose_slot(slot, proof, d)["reason"], "not_approved")


if __name__ == "__main__":
    unittest.main()
