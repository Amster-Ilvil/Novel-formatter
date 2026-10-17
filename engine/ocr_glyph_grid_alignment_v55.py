# -*- coding: utf-8 -*-
"""V5.5: globally optimal OCR-to-physical-glyph alignment, with safe abstention.

The input grid comes from the original vertical column image. The anchors MUST
be visually confirmed labels on OTHER physical cells. Hayai and NDL are
aligned separately; location is accepted only when all minimal-cost alignments
place the target on the same cell in each independently executed OCR result.

No OCR model is loaded. This module NEVER determines the target's codepoint.
"""
from __future__ import annotations

from typing import Mapping

_GAP = 2
_INF = 10 ** 9


def _match_cost(char: str, cell: int, verified_labels: Mapping[int, str]) -> int:
    known = verified_labels.get(cell)
    return 0 if known is None or known == char else _INF


def _forward(text: str, cells: int, anchors: Mapping[int, str]) -> list[list[int]]:
    n = len(text)
    f = [[_INF] * (cells + 1) for _ in range(n + 1)]
    f[0][0] = 0
    for i in range(n + 1):
        for j in range(cells + 1):
            cost = f[i][j]
            if cost >= _INF:
                continue
            if i < n and f[i + 1][j] > cost + _GAP:
                f[i + 1][j] = cost + _GAP
            if j < cells and f[i][j + 1] > cost + _GAP:
                f[i][j + 1] = cost + _GAP
            if i < n and j < cells:
                match = _match_cost(text[i], j, anchors)
                if match < _INF and f[i + 1][j + 1] > cost + match:
                    f[i + 1][j + 1] = cost + match
    return f


def _backward(text: str, cells: int, anchors: Mapping[int, str]) -> list[list[int]]:
    n = len(text)
    b = [[_INF] * (cells + 1) for _ in range(n + 1)]
    b[n][cells] = 0
    for i in range(n, -1, -1):
        for j in range(cells, -1, -1):
            cost = b[i][j]
            if cost >= _INF:
                continue
            if i > 0 and b[i - 1][j] > cost + _GAP:
                b[i - 1][j] = cost + _GAP
            if j > 0 and b[i][j - 1] > cost + _GAP:
                b[i][j - 1] = cost + _GAP
            if i > 0 and j > 0:
                match = _match_cost(text[i - 1], j - 1, anchors)
                if match < _INF and b[i - 1][j - 1] > cost + match:
                    b[i - 1][j - 1] = cost + match
    return b


def locate_target_by_verified_anchors(
    text: str,
    *,
    cell_count: int,
    target_index: int,
    verified_cell_labels: Mapping[int, str],
    min_anchors: int = 2,
) -> dict:
    """Find unique target CELL across ALL global minimum-cost paths.

    Correct OCR neighbor characters are not inferred by a majority vote.
    At least two independently confirmed image-glyph anchors are mandatory.
    A target that can also be an OCR insertion on an optimal path is refused.
    """
    n, m, k = len(text), int(cell_count), int(target_index)
    if not (n > 0 and m > 0 and 0 <= k < n):
        return {"ok": False, "reason": "invalid_length_or_target"}
    try:
        anchors = {int(i): str(c) for i, c in verified_cell_labels.items()}
    except (ValueError, TypeError):
        return {"ok": False, "reason": "invalid_anchors"}
    if len(anchors) < min_anchors:
        return {"ok": False, "reason": "insufficient_independent_visual_anchors"}
    if any(i < 0 or i >= m or len(ch) != 1 for i, ch in anchors.items()):
        return {"ok": False, "reason": "bad_anchor_cell_or_codepoint"}
    f = _forward(text, m, anchors)
    b = _backward(text, m, anchors)
    best = f[n][m]
    if best >= _INF:
        return {"ok": False, "reason": "alignment_unavailable"}

    # Edges, not a single arbitrarily chosen traceback, establish uniqueness.
    candidates = []
    for cell in range(m):
        step = _match_cost(text[k], cell, anchors)
        if step < _INF and f[k][cell] + step + b[k + 1][cell + 1] == best:
            candidates.append(cell)
    can_be_skipped = any(
        f[k][j] + _GAP + b[k + 1][j] == best
        for j in range(m + 1)
    )
    if not candidates or can_be_skipped:
        return {
            "ok": False, "reason": "target_may_be_inserted_or_skipped",
            "candidate_cells": candidates,
        }
    if len(candidates) != 1:
        return {
            "ok": False, "reason": "several_optimal_target_cells",
            "candidate_cells": candidates,
        }
    target = candidates[0]
    if target in anchors:
        # This would let recognition of the TARGET itself influence its
        # coordinate, defeating independent neighbor-anchor verification.
        return {"ok": False, "reason": "target_cell_was_used_as_anchor"}
    left = sum(i < target for i in anchors)
    right = sum(i > target for i in anchors)
    return {
        "ok": True,
        "cell_index": target,
        "alignment_cost": best,
        "anchor_count": len(anchors),
        "bracketed_by_visual_anchors": bool(left and right),
        "edge_unbracketed": not (left and right),
        "position_verified": False,
        "reason": "unique_location_candidate_requires_pixel_confirmation",
    }


def cross_check_independent_ocr_positions(
    first_text: str, second_text: str,
    *,
    first_target_index: int, second_target_index: int,
    cell_count: int, verified_cell_labels: Mapping[int, str],
    first_independently_executed: bool, second_independently_executed: bool,
) -> dict:
    """Require Hayai and NDL (or two other ACTUAL independent outputs) to agree."""
    if not first_independently_executed or not second_independently_executed:
        return {"ok": False, "reason": "seeded_or_reused_ocr_is_not_an_independent_vote"}
    a = locate_target_by_verified_anchors(
        first_text, cell_count=cell_count, target_index=first_target_index,
        verified_cell_labels=verified_cell_labels,
    )
    b = locate_target_by_verified_anchors(
        second_text, cell_count=cell_count, target_index=second_target_index,
        verified_cell_labels=verified_cell_labels,
    )
    if not a.get("ok") or not b.get("ok"):
        return {"ok": False, "reason": "one_or_both_models_ambiguous", "first": a, "second": b}
    if a["cell_index"] != b["cell_index"]:
        return {"ok": False, "reason": "model_position_disagreement", "first": a, "second": b}
    return {
        "ok": True,
        "candidate_cell_index": a["cell_index"],
        "bracketed_by_verified_glyphs": bool(
            a["bracketed_by_visual_anchors"] and b["bracketed_by_visual_anchors"]
        ),
        "position_verified": False,
        "requires_visual_spot_check": True,
        "reason": "both_independent_ocr_positions_uniquely_agree",
    }
