from __future__ import annotations

"""Risk-only ordering for the human OCR review queue.

The score never chooses text.  It only decides which unresolved row is shown
first so expensive human attention is spent on the most uncertain evidence.
"""

from dataclasses import dataclass
import re
from typing import Sequence

_SENSITIVE = re.compile(r"(?:[0-9０-９]+|[A-Z]{2,}|Lv\.?\s*[0-9０-９]+|HP|MP|ランク|スキル|技能|装備)", re.I)


@dataclass(frozen=True, slots=True)
class ReviewPriority:
    score: int
    level: str
    reasons: tuple[str, ...]


def _candidate_texts(state) -> list[str]:
    values: list[str] = []
    for candidate in list(getattr(state, "candidates", ()) or ()):
        if str(getattr(candidate, "audit_level", "") or "") == "historical_ocr_evidence":
            continue
        value = str(getattr(candidate, "text", "") or "")
        if value not in values:
            values.append(value)
    return values


def review_priority(row, state) -> ReviewPriority:
    texts = _candidate_texts(state)
    nonempty = [value for value in texts if value.strip()]
    reasons: list[str] = []
    score = 0

    if bool(getattr(row, "is_conflict", False)):
        score += 40
        reasons.append("model_disagreement")
    distinct = len(set(nonempty))
    if distinct >= 3:
        score += 25
        reasons.append("three_way_disagreement")
    elif distinct == 2:
        score += 12
        reasons.append("two_way_disagreement")

    if any(not value.strip() or "□" in value or "�" in value for value in texts):
        score += 28
        reasons.append("empty_or_placeholder")

    lengths = [len(value.strip()) for value in nonempty]
    if len(lengths) >= 2 and max(lengths) - min(lengths) >= max(3, int(max(lengths) * 0.12)):
        score += 18
        reasons.append("length_disagreement")

    if any(_SENSITIVE.search(value) for value in nonempty):
        score += 18
        reasons.append("numeric_or_structured_content")

    difficulty = str(getattr(row, "consensus_entropy_difficulty", "") or "")
    if difficulty == "high":
        score += 18
        reasons.append("high_consensus_entropy")
    elif difficulty == "medium":
        score += 8
        reasons.append("medium_consensus_entropy")

    warnings: Sequence[str] = tuple(getattr(row, "warnings", ()) or ())
    if warnings:
        score += min(15, len(warnings) * 5)
        reasons.append("ocr_warnings")

    if score >= 80:
        level = "critical"
    elif score >= 55:
        level = "high"
    elif score >= 30:
        level = "medium"
    else:
        level = "normal"
    return ReviewPriority(score=score, level=level, reasons=tuple(dict.fromkeys(reasons)))


def prioritized_unresolved_rows(rows, states) -> tuple[int, ...]:
    values = []
    for index, state in enumerate(states):
        if not bool(getattr(state, "unresolved", False)):
            continue
        row = rows[index] if index < len(rows) else None
        priority = review_priority(row, state)
        values.append((-priority.score, index))
    values.sort()
    return tuple(index for _neg_score, index in values)
