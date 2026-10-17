# -*- coding: utf-8 -*-
"""Pure-Python dynamic-programming alignment safety tests (synthetic data)."""
from __future__ import annotations
import unittest

from engine.ocr_glyph_grid_alignment_v55 import (
    cross_check_independent_ocr_positions,
    locate_target_by_verified_anchors,
)


class AnchoredGlyphAlignmentTest(unittest.TestCase):
    def test_exact_length_with_bracketing_anchors(self):
        got = locate_target_by_verified_anchors(
            "魔目だ。", cell_count=4, target_index=1,
            verified_cell_labels={0: "魔", 2: "だ", 3: "。"},
        )
        self.assertTrue(got["ok"])
        self.assertEqual(got["cell_index"], 1)
        self.assertTrue(got["bracketed_by_visual_anchors"])
        self.assertFalse(got["position_verified"])

    def test_insertion_away_from_target_is_located(self):
        got = cross_check_independent_ocr_positions(
            "魔X王目だ。", "魔王日だ。",
            first_target_index=3, second_target_index=2,
            cell_count=5, verified_cell_labels={0: "魔", 1: "王", 3: "だ", 4: "。"},
            first_independently_executed=True,
            second_independently_executed=True,
        )
        self.assertTrue(got["ok"])
        self.assertEqual(got["candidate_cell_index"], 2)
        self.assertTrue(got["bracketed_by_verified_glyphs"])

    def test_ambiguous_insertion_neighbor_does_not_guess(self):
        got = locate_target_by_verified_anchors(
            "魔X目だ。", cell_count=4, target_index=2,
            verified_cell_labels={0: "魔", 3: "。"},
        )
        self.assertFalse(got["ok"])

    def test_reused_model_is_not_a_second_vote(self):
        got = cross_check_independent_ocr_positions(
            "魔目だ。", "魔日だ。",
            first_target_index=1, second_target_index=1,
            cell_count=4, verified_cell_labels={0: "魔", 2: "だ", 3: "。"},
            first_independently_executed=True,
            second_independently_executed=False,
        )
        self.assertFalse(got["ok"])
        self.assertEqual(got["reason"], "seeded_or_reused_ocr_is_not_an_independent_vote")

    def test_visual_anchor_must_not_be_the_target(self):
        got = locate_target_by_verified_anchors(
            "魔目だ。", cell_count=4, target_index=1,
            verified_cell_labels={0: "魔", 1: "目", 3: "。"},
        )
        self.assertFalse(got["ok"])
        self.assertEqual(got["reason"], "target_cell_was_used_as_anchor")

    def test_not_enough_real_visual_anchors(self):
        got = locate_target_by_verified_anchors(
            "魔目だ。", cell_count=4, target_index=1,
            verified_cell_labels={0: "魔"},
        )
        self.assertFalse(got["ok"])
        self.assertEqual(got["reason"], "insufficient_independent_visual_anchors")

    def test_invalid_target_index_refused(self):
        got = locate_target_by_verified_anchors(
            "魔目", cell_count=2, target_index=6,
            verified_cell_labels={0: "魔", 1: "目"},
        )
        self.assertFalse(got["ok"])

    def test_position_only_never_grants_correct_unicode(self):
        got = cross_check_independent_ocr_positions(
            "魔目だ。", "魔日だ。",
            first_target_index=1, second_target_index=1,
            cell_count=4, verified_cell_labels={0: "魔", 2: "だ", 3: "。"},
            first_independently_executed=True,
            second_independently_executed=True,
        )
        self.assertTrue(got["ok"])
        self.assertNotIn("character", got)
        self.assertFalse(got["position_verified"])


if __name__ == "__main__":
    unittest.main()
