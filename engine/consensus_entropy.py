#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training-free OCR agreement diagnostics inspired by Consensus Entropy.

The signal is deliberately *diagnostic only*.  Novel Formatter never uses it to
rewrite OCR text, pick a model, or clear a multi-model conflict.  It measures
how tightly independent OCR hypotheses cluster so review/export code can route
hard rows first without changing the authoritative whole-sentence semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from engine.ocr_unicode_standardizer import japanese_ocr_comparison_key


def normalized_edit_distance(left: str, right: str) -> float:
    """Return Levenshtein distance / max length in ``[0, 1]``."""
    a = str(left or "")
    b = str(right or "")
    if a == b:
        return 0.0
    if not a or not b:
        return 1.0
    if len(a) > len(b):
        a, b = b, a
    previous = list(range(len(a) + 1))
    for row_index, char_b in enumerate(b, start=1):
        current = [row_index]
        for col_index, char_a in enumerate(a, start=1):
            current.append(min(
                current[-1] + 1,
                previous[col_index] + 1,
                previous[col_index - 1] + (char_a != char_b),
            ))
        previous = current
    return float(previous[-1]) / float(max(len(a), len(b), 1))


@dataclass(frozen=True, slots=True)
class ConsensusEntropyProfile:
    scores: tuple[float, ...]
    best_index: int
    minimum: float
    mean: float
    maximum: float
    active_count: int
    difficulty: str
    review_required: bool


def _difficulty(mean: float, maximum: float, active_count: int) -> tuple[str, bool]:
    if active_count < 2:
        return "insufficient", True
    # Thresholds are intentionally conservative.  They only control review
    # priority/diagnostics and never auto-accept a row.
    if maximum <= 0.04 and mean <= 0.03:
        return "low", False
    if maximum <= 0.12 and mean <= 0.08:
        return "medium", False
    return "high", True


def calculate_consensus_entropy(
    texts: Sequence[str] | Iterable[str],
    *,
    comparison_keys: Sequence[str] | None = None,
    excluded_indices: Iterable[int] = (),
) -> ConsensusEntropyProfile:
    """Compute per-candidate average normalized disagreement.

    ``excluded_indices`` is used for fast-consensus seeded models because copied
    text is not independent evidence.  Scores for excluded/empty candidates are
    ``1.0`` so callers cannot accidentally treat them as a strong vote.
    """
    values = [str(value or "") for value in texts]
    excluded = {int(value) for value in excluded_indices}
    if comparison_keys is None:
        keys = [japanese_ocr_comparison_key(value)[0] for value in values]
    else:
        keys = [str(value or "") for value in comparison_keys]
        if len(keys) < len(values):
            keys.extend("" for _ in range(len(values) - len(keys)))
        keys = keys[: len(values)]

    active = [
        index for index, key in enumerate(keys)
        if index not in excluded and bool(str(values[index] or "").strip()) and bool(key)
    ]
    scores = [1.0] * len(values)
    if len(active) == 1:
        scores[active[0]] = 1.0
    elif len(active) >= 2:
        for index in active:
            distances = [
                normalized_edit_distance(keys[index], keys[other])
                for other in active if other != index
            ]
            scores[index] = sum(distances) / max(1, len(distances))

    if active:
        active_scores = [scores[index] for index in active]
        best_index = min(active, key=lambda index: (scores[index], index))
        minimum = min(active_scores)
        mean = sum(active_scores) / len(active_scores)
        maximum = max(active_scores)
    else:
        best_index = -1
        minimum = mean = maximum = 1.0
    difficulty, review_required = _difficulty(mean, maximum, len(active))
    return ConsensusEntropyProfile(
        scores=tuple(round(float(score), 6) for score in scores),
        best_index=int(best_index),
        minimum=round(float(minimum), 6),
        mean=round(float(mean), 6),
        maximum=round(float(maximum), 6),
        active_count=len(active),
        difficulty=difficulty,
        review_required=bool(review_required),
    )


__all__ = [
    "ConsensusEntropyProfile",
    "calculate_consensus_entropy",
    "normalized_edit_distance",
]
