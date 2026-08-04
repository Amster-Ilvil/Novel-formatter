#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Review helpers for text-comparison AI results.

The AI preview is intentionally kept separate from the document currently used
by Formatter/EPUB.  This module creates deterministic before/after change
windows, assigns conservative risk levels and can rebuild a preview from an
explicit per-change acceptance selection.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
import re
from typing import Sequence

from engine.text_compare import CompareLine, align_lines, parse_image_marker


@dataclass(slots=True)
class AIReviewChange:
    index: int
    kind: str
    display_start: int
    display_end: int
    before_start: int
    before_end: int
    after_start: int
    after_end: int
    before_lines: list[str]
    after_lines: list[str]
    similarity: float
    char_delta: int
    severity: str
    warnings: list[str]
    # Optional evidence from a mostly-correct draft/reference text.  The
    # reference is advisory because a published edition may intentionally
    # differ from its draft.
    reference_relation: str = "unmatched"
    reference_confidence: float = 0.0
    reference_before_score: float = 0.0
    reference_after_score: float = 0.0
    reference_excerpt: str = ""
    reference_line_start: int = -1
    reference_line_end: int = -1

    def to_dict(self) -> dict:
        return asdict(self)


_JA_CHAR = r"ぁ-んァ-ヶー一-龯々〆ヵヶ"
_END = r"。！？!?」』）】》〉…‥ー―—"
_CONJUNCTION_END_RE = re.compile(
    r"(?:けど|けれど|けれども|ので|のに|から|ても|つつ|ながら)[。]?[」』]?$"
)


def _compact(text: str) -> str:
    return re.sub(r"[\s　]+", "", text or "")


def _group_similarity(before_lines: Sequence[str], after_lines: Sequence[str]) -> float:
    left = _compact("".join(before_lines))
    right = _compact("".join(after_lines))
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def _quote_signature(text: str) -> tuple[int, int, int, int]:
    return text.count("「"), text.count("」"), text.count("『"), text.count("』")


def _suspicious_repetition(text: str) -> bool:
    compact = _compact(text)
    # Repeated closing fragments are a common post-merge residue, e.g.
    # 感情』情』.  Require punctuation/quote participation to avoid flagging
    # legitimate Japanese reduplication.
    if re.search(r"([%s]{1,5}[」』）】])\1" % _JA_CHAR, compact):
        return True
    # OCR/AI occasionally duplicates one kana immediately before punctuation or
    # an auxiliary ending: みたいい, らしいい, なのの.  Only flag, never auto-fix.
    if re.search(r"([ぁ-んァ-ヶ])\1(?:じゃ|だ|です|ない|の|。|、|」|』)", compact):
        return True
    return False


def _looks_reordered(before: str, after: str, similarity: float) -> bool:
    """Detect same-length semantic rewriting/reordering without NLP guesses."""
    a = _compact(before)
    b = _compact(after)
    if min(len(a), len(b)) < 30:
        return False
    ratio = len(b) / max(1, len(a))
    if not 0.86 <= ratio <= 1.14:
        return False
    # A conservative OCR correction normally remains very close.  A middling
    # similarity with near-identical length usually means sentence/clause order
    # or author wording was rewritten rather than merely corrected.
    return similarity < 0.78


def _warnings(before_lines: Sequence[str], after_lines: Sequence[str], similarity: float) -> tuple[str, list[str]]:
    before = "\n".join(before_lines)
    after = "\n".join(after_lines)
    warnings: list[str] = []
    before_compact = _compact(before)
    after_compact = _compact(after)
    if before_compact:
        ratio = len(after_compact) / max(1, len(before_compact))
        if ratio < 0.82:
            warnings.append(f"文字量减少 {round((1-ratio)*100)}%")
        elif ratio > 1.22:
            warnings.append(f"文字量增加 {round((ratio-1)*100)}%")
    if _looks_reordered(before, after, similarity):
        warnings.append("疑似重排或改写原句")
    elif len(before_compact) >= 40 and similarity < 0.66:
        warnings.append("改写幅度很大")

    before_quotes = _quote_signature(before)
    after_quotes = _quote_signature(after)
    if after.count("「") != after.count("」"):
        warnings.append("会话引号不平衡")
    if after.count("『") != after.count("』"):
        warnings.append("引用引号不平衡")
    if before_quotes != after_quotes and any(before_quotes):
        warnings.append("引号结构发生变化")
    if any(parse_image_marker(line) for line in list(before_lines) + list(after_lines)):
        warnings.append("涉及插图锚点")

    stripped = after.strip()
    if re.search(rf"^[{_JA_CHAR}][。』」](?=.{{8,}})", stripped):
        warnings.append("疑似残片开头")
    if re.search(r"[^。！？!?」』\n]「", after) and not stripped.startswith("「"):
        warnings.append("对白疑似粘在正文中")
    if "「" not in after and "」" in after:
        warnings.append("缺少会话开引号")
    if "『" not in after and "』" in after:
        warnings.append("缺少引用开引号")
    if _suspicious_repetition(after) and not _suspicious_repetition(before):
        warnings.append("AI 新增了疑似重复字或合并残留")

    before_dialogue = bool(re.search(r"「.+?」", before, re.S))
    after_dialogue = bool(re.search(r"「.+?」", after, re.S))
    if before_dialogue and not after_dialogue and "「" not in after and "」" not in after:
        warnings.append("对白标记被删除")
    if after_dialogue and _CONJUNCTION_END_RE.search(stripped):
        warnings.append("对白疑似在强接续词处被截断")

    high_markers = {
        "会话引号不平衡", "引用引号不平衡", "涉及插图锚点",
        "疑似重排或改写原句", "对白标记被删除", "缺少会话开引号",
        "缺少引用开引号", "对白疑似粘在正文中", "疑似残片开头",
        "AI 新增了疑似重复字或合并残留", "对白疑似在强接续词处被截断",
    }
    extreme_length_change = False
    if before_compact:
        length_ratio = len(after_compact) / max(1, len(before_compact))
        extreme_length_change = length_ratio < 0.65 or length_ratio > 1.55
    if any(item in high_markers for item in warnings) or extreme_length_change or (len(before_compact) >= 20 and similarity < 0.55):
        severity = "high"
    elif warnings:
        severity = "medium"
    else:
        severity = "low"
    return severity, warnings


def build_ai_review_changes(before: Sequence[CompareLine], after: Sequence[CompareLine]) -> list[AIReviewChange]:
    """Return small, reviewable non-equal alignment windows.

    Consecutive replacement rows are kept separate so one risky line does not
    force an unrelated neighbouring correction to be rejected.  Gap rows that
    belong to a line merge/split stay attached to the nearest replacement.
    """
    rows = align_lines(before, after)
    result: list[AIReviewChange] = []
    before_cursor = 0
    after_cursor = 0
    row_index = 0
    change_index = 1
    while row_index < len(rows):
        row = rows[row_index]
        if row.tag == "equal":
            if row.left is not None:
                before_cursor += 1
            if row.right is not None:
                after_cursor += 1
            row_index += 1
            continue

        display_start = row_index
        before_start = before_cursor
        after_start = after_cursor
        before_lines: list[str] = []
        after_lines: list[str] = []
        starts_with_replace = row.tag == "replace"
        saw_replace = False

        while row_index < len(rows) and rows[row_index].tag != "equal":
            current = rows[row_index]
            # A gap before a replacement is a standalone insertion/deletion.
            # A replacement may absorb following gap rows because those usually
            # represent a line merge/split belonging to that replacement.
            if not starts_with_replace and current.tag == "replace":
                break
            if current.tag == "replace" and saw_replace:
                break
            if current.tag == "replace":
                saw_replace = True
            if current.left is not None:
                before_lines.append(current.left.text)
                before_cursor += 1
            if current.right is not None:
                after_lines.append(current.right.text)
                after_cursor += 1
            row_index += 1
            if starts_with_replace and saw_replace and row_index < len(rows) and rows[row_index].tag == "replace":
                break

        display_end = row_index - 1
        before_end = before_cursor - 1
        after_end = after_cursor - 1
        if before_lines and after_lines:
            kind = "replace"
        elif before_lines:
            kind = "delete"
        else:
            kind = "insert"
        similarity = _group_similarity(before_lines, after_lines)
        severity, warnings = _warnings(before_lines, after_lines, similarity)
        result.append(AIReviewChange(
            index=change_index,
            kind=kind,
            display_start=display_start,
            display_end=display_end,
            before_start=before_start,
            before_end=before_end,
            after_start=after_start,
            after_end=after_end,
            before_lines=before_lines,
            after_lines=after_lines,
            similarity=similarity,
            char_delta=len(_compact("".join(after_lines))) - len(_compact("".join(before_lines))),
            severity=severity,
            warnings=warnings,
        ))
        change_index += 1
    return result


def default_review_decisions(changes: Sequence[AIReviewChange]) -> dict[int, bool]:
    """Build conservative defaults using risk plus optional reference evidence.

    A pre-publication draft is not allowed to overrule a structurally high-risk
    edit automatically.  It *can* reject an AI rewrite when the draft clearly
    supports the pre-AI wording, and it can strengthen the default acceptance of
    an otherwise low/medium-risk OCR correction.
    """
    result: dict[int, bool] = {}
    for change in changes:
        relation = getattr(change, "reference_relation", "unmatched")
        confidence = float(getattr(change, "reference_confidence", 0.0) or 0.0)
        if relation == "supports_before" and confidence >= 0.45:
            result[change.index] = False
        elif change.severity == "high":
            result[change.index] = False
        else:
            result[change.index] = True
    return result


def resolve_review_records(
    before: Sequence[CompareLine],
    after: Sequence[CompareLine],
    changes: Sequence[AIReviewChange],
    decisions: dict[int, bool] | set[int] | Sequence[int],
) -> list[CompareLine]:
    """Rebuild records from ``before`` using only explicitly accepted changes.

    Changes are applied in descending source-coordinate order so insertion,
    deletion and replacement windows cannot invalidate earlier coordinates.
    Image markers and record metadata are copied from the chosen side.
    """
    if isinstance(decisions, dict):
        accepted = {index for index, value in decisions.items() if value}
    else:
        accepted = {int(index) for index in decisions}
    output = [copy.deepcopy(item) for item in before]
    for change in sorted(changes, key=lambda item: (item.before_start, item.index), reverse=True):
        if change.index not in accepted:
            continue
        replacement = [copy.deepcopy(item) for item in after[change.after_start:change.after_end + 1]]
        start = max(0, change.before_start)
        stop = max(start, change.before_end + 1)
        output[start:stop] = replacement
    return output


def review_summary(changes: Sequence[AIReviewChange], decisions: dict[int, bool] | None = None) -> dict:
    summary = {
        "groups": len(changes),
        "replace": 0,
        "insert": 0,
        "delete": 0,
        "high": 0,
        "medium": 0,
        "low": 0,
        "char_delta": 0,
        "before_lines": 0,
        "after_lines": 0,
        "accepted": 0,
        "rejected": 0,
        "accepted_high": 0,
        "reference_supports_after": 0,
        "reference_supports_before": 0,
        "reference_neutral": 0,
    }
    for change in changes:
        summary[change.kind] += 1
        summary[change.severity] += 1
        relation = getattr(change, "reference_relation", "unmatched")
        if relation == "supports_after":
            summary["reference_supports_after"] += 1
        elif relation == "supports_before":
            summary["reference_supports_before"] += 1
        elif relation == "neutral":
            summary["reference_neutral"] += 1
        summary["char_delta"] += change.char_delta
        summary["before_lines"] += len(change.before_lines)
        summary["after_lines"] += len(change.after_lines)
        if decisions is not None:
            if decisions.get(change.index, False):
                summary["accepted"] += 1
                if change.severity == "high":
                    summary["accepted_high"] += 1
            else:
                summary["rejected"] += 1
    return summary
