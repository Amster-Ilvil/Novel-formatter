#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Physical-column constrained character-level OCR fusion.

The comparison layer already aligns each OCR result to immutable physical column
IDs.  This module operates only inside one aligned row, so it cannot pull text
from a neighbouring printed column.  Three OCR strings are globally aligned at
character level and voted position by position with role-aware reliability:

* NDLOCR/YomiToku are favoured for continuous skeleton/completeness;
* 48px AR and YomiToku are favoured for substitutions;
* Apple Vision is useful for omissions but single-model insertions are penalised
  because duplicated text is its most common failure mode.

The implementation is deliberately conservative.  It emits a synthetic candidate
only when three non-empty OCR results provide enough shared evidence, and marks
uncertain rows for local re-recognition instead of silently inventing text.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import math
import re
import unicodedata
from typing import Sequence

from engine.external_ocr import line_quality
from engine.text_compare import normalise_for_alignment
from engine.adaptive_ocr_ensemble import is_sensitive_text, standard_japanese_key

_GAP = None
_JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff々〆ヵヶー。、！？「」『』【】（）…—・]", re.UNICODE)


@dataclass(slots=True)
class CharacterFusionResult:
    text: str = ""
    confidence: float = 0.0
    auto_select: bool = False
    review_required: bool = False
    local_reocr_recommended: bool = False
    reason: str = ""
    warnings: tuple[str, ...] = ()
    evidence: dict = field(default_factory=dict)


def model_role(label: str) -> str:
    value = str(label or "").strip().lower()
    if "ndl" in value:
        return "ndlocr"
    if "48px" in value or "manga_48" in value or "48 px" in value:
        return "manga48"
    if "apple" in value or "vision" in value or "macocr" in value or "macos" in value:
        return "apple"
    if "yomi" in value or "parseq" in value:
        return "yomitoku"
    if "manga" in value:
        return "manga"
    return "generic"


_ROLE_BACKBONE_WEIGHT = {
    "ndlocr": 1.10,
    "yomitoku": 1.07,
    "generic": 1.00,
    "manga": 0.99,
    "apple": 0.98,
    "manga48": 0.96,
}
_ROLE_SUBSTITUTION_WEIGHT = {
    "yomitoku": 1.10,
    "manga48": 1.08,
    "apple": 1.04,
    "ndlocr": 1.00,
    "manga": 1.00,
    "generic": 1.00,
}
_ROLE_SINGLE_INSERTION_WEIGHT = {
    "yomitoku": 0.96,
    "ndlocr": 0.91,
    "generic": 0.86,
    "manga": 0.80,
    "manga48": 0.76,
    "apple": 0.52,
}


def _visible_text(value: str) -> str:
    return str(value or "").replace("\r", "").replace("\n", "").strip()


def _is_plausible_char(char: str) -> bool:
    if not char:
        return False
    if _JAPANESE_RE.fullmatch(char):
        return True
    if char.isascii():
        return char.isalnum() or char in " !?.,:;()[]{}'\"-/+%&"
    return not char.isspace() and not unicodedata.category(char).startswith("C")


def _adjacent_repeat_score(text: str) -> float:
    """Return 0..1 likelihood that ``text`` contains an OCR duplicate span."""
    value = _visible_text(text)
    n = len(value)
    if n < 12:
        return 0.0
    best = 0
    max_span = min(80, n // 2)
    for span in range(max_span, 5, -1):
        found = False
        for start in range(0, n - span * 2 + 1):
            if value[start:start + span] == value[start + span:start + span * 2]:
                best = span
                found = True
                break
        if found:
            break
    return min(1.0, best / max(12.0, n * 0.42))


def _pairwise_global_alignment(left: str, right: str) -> tuple[list[str | None], list[str | None]]:
    """Needleman-Wunsch alignment with deterministic Japanese-friendly ties."""
    a = list(left)
    b = list(right)
    n, m = len(a), len(b)
    # Rows are normally one sentence/physical-column group (<250 chars).  A
    # linear fallback avoids pathological memory use on malformed pasted text.
    if n * m > 160_000:
        matcher = SequenceMatcher(None, left, right, autojunk=False)
        out_a: list[str | None] = []
        out_b: list[str | None] = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                out_a.extend(a[i1:i2]); out_b.extend(b[j1:j2])
                continue
            span = max(i2 - i1, j2 - j1)
            for offset in range(span):
                out_a.append(a[i1 + offset] if i1 + offset < i2 else _GAP)
                out_b.append(b[j1 + offset] if j1 + offset < j2 else _GAP)
        return out_a, out_b

    # Costs: exact=0, substitution=1.05, gap=1.0.  Equal-cost ties prefer a
    # substitution to keep neighbouring physical glyph positions together.
    previous = [float(j) for j in range(m + 1)]
    trace = [bytearray(m + 1) for _ in range(n + 1)]
    for j in range(1, m + 1):
        trace[0][j] = 2  # insertion/right
    for i in range(1, n + 1):
        current = [float(i)] + [0.0] * m
        trace[i][0] = 1  # deletion/down
        for j in range(1, m + 1):
            subst = previous[j - 1] + (0.0 if a[i - 1] == b[j - 1] else 1.05)
            delete = previous[j] + 1.0
            insert = current[j - 1] + 1.0
            best = min(subst, delete, insert)
            current[j] = best
            if subst <= delete + 1e-9 and subst <= insert + 1e-9:
                trace[i][j] = 0
            elif delete <= insert:
                trace[i][j] = 1
            else:
                trace[i][j] = 2
        previous = current

    out_a: list[str | None] = []
    out_b: list[str | None] = []
    i, j = n, m
    while i or j:
        direction = trace[i][j]
        if i and j and direction == 0:
            out_a.append(a[i - 1]); out_b.append(b[j - 1]); i -= 1; j -= 1
        elif i and (j == 0 or direction == 1):
            out_a.append(a[i - 1]); out_b.append(_GAP); i -= 1
        else:
            out_a.append(_GAP); out_b.append(b[j - 1]); j -= 1
    out_a.reverse(); out_b.reverse()
    return out_a, out_b


def _alignment_votes(pivot: str, text: str) -> tuple[list[str | None], dict[int, str]]:
    """Map one OCR string onto pivot positions and insertion boundaries."""
    aligned_pivot, aligned_text = _pairwise_global_alignment(pivot, text)
    position_votes: list[str | None] = [_GAP] * len(pivot)
    insertion_parts: dict[int, list[str]] = {}
    pivot_pos = 0
    for a, b in zip(aligned_pivot, aligned_text):
        if a is _GAP:
            if b is not _GAP:
                insertion_parts.setdefault(pivot_pos, []).append(b)
            continue
        if pivot_pos < len(position_votes):
            position_votes[pivot_pos] = b
        pivot_pos += 1
    insertions = {boundary: "".join(parts) for boundary, parts in insertion_parts.items() if parts}
    return position_votes, insertions


def _choose_pivot(
    texts: Sequence[str],
    roles: Sequence[str],
    reliabilities: Sequence[float],
) -> int:
    lengths = [len(value) for value in texts]
    ordered = sorted(lengths)
    median = ordered[len(ordered) // 2] or 1
    scored: list[tuple[float, int]] = []
    for index, (text, role) in enumerate(zip(texts, roles)):
        quality = line_quality(text).score
        length_ratio = min(len(text), median) / max(len(text), median, 1)
        duplicate = _adjacent_repeat_score(text)
        reliability = reliabilities[index] if index < len(reliabilities) else 1.0
        score = quality * .43 + length_ratio * .32 + _ROLE_BACKBONE_WEIGHT.get(role, 1.0) * .17 + reliability * .08
        score -= duplicate * .24
        # A severe short result must never become the backbone merely because it
        # is clean; this is the characteristic 48px whole-span omission failure.
        if len(text) < median * .62 and median >= 18:
            score -= .28
        scored.append((score, index))
    scored.sort(reverse=True)
    return scored[0][1]


def _char_vote(
    candidates: Sequence[tuple[int, str | None]],
    roles: Sequence[str],
    reliabilities: Sequence[float],
) -> tuple[str | None, bool, bool]:
    """Return (character/gap, majority_supported, all_different)."""
    counts: dict[str | None, list[int]] = {}
    for model_index, char in candidates:
        counts.setdefault(char, []).append(model_index)
    majority = [item for item in counts.items() if len(item[1]) >= 2]
    if majority:
        majority.sort(key=lambda item: (len(item[1]), item[0] is not _GAP), reverse=True)
        return majority[0][0], True, False

    weighted: list[tuple[float, str | None]] = []
    for char, model_indices in counts.items():
        score = 0.0
        for model_index in model_indices:
            role = roles[model_index]
            reliability = reliabilities[model_index] if model_index < len(reliabilities) else 1.0
            if char is _GAP:
                score += _ROLE_BACKBONE_WEIGHT.get(role, 1.0) * .86 * reliability
            else:
                score += _ROLE_SUBSTITUTION_WEIGHT.get(role, 1.0) * reliability
                if _is_plausible_char(char):
                    score += .06
        weighted.append((score, char))
    weighted.sort(key=lambda item: (item[0], item[1] is not _GAP), reverse=True)
    return weighted[0][1] if weighted else _GAP, False, len(counts) >= 3


def _fuse_insertions(
    boundary: int,
    insertions_by_model: Sequence[dict[int, str]],
    roles: Sequence[str],
    reliabilities: Sequence[float],
) -> tuple[str, bool, bool]:
    values = [(index, mapping.get(boundary, "")) for index, mapping in enumerate(insertions_by_model)]
    nonempty = [(index, value) for index, value in values if value]
    if not nonempty:
        return "", True, False
    grouped: dict[str, list[int]] = {}
    for index, value in nonempty:
        grouped.setdefault(value, []).append(index)
    agreed = [item for item in grouped.items() if len(item[1]) >= 2]
    if agreed:
        agreed.sort(key=lambda item: (len(item[1]), len(item[0])), reverse=True)
        return agreed[0][0], True, False

    # All insertion strings differ or only one model supplies text.  Do not
    # silently accept a long Apple-only fragment: that is usually duplicated or
    # cross-column text.  A one-character plausible insertion from a strong
    # completeness model is retained only as a low-confidence suggestion.
    scored: list[tuple[float, str, int]] = []
    for model_index, value in nonempty:
        role = roles[model_index]
        reliability = reliabilities[model_index] if model_index < len(reliabilities) else 1.0
        score = _ROLE_SINGLE_INSERTION_WEIGHT.get(role, .84) * reliability
        score += min(.08, len(value) * .015)
        score -= _adjacent_repeat_score(value) * .25
        if all(_is_plausible_char(char) for char in value):
            score += .05
        if len(value) > 4:
            score -= .16
        scored.append((score, value, model_index))
    scored.sort(reverse=True)
    best_score, best_value, _ = scored[0]
    retain = best_score >= .98 and len(best_value) <= 2
    return (best_value if retain else ""), False, True


def build_character_fusion(
    texts: Sequence[str],
    labels: Sequence[str] | None = None,
    *,
    physical_column_ids: Sequence[str] | None = None,
    model_confidences: Sequence[float] | None = None,
) -> CharacterFusionResult:
    values = [_visible_text(value) for value in texts]
    nonempty_indices = [index for index, value in enumerate(values) if value]
    if len(values) != 3 or len(nonempty_indices) < 3:
        return CharacterFusionResult()

    visible_labels = list(labels or [])
    while len(visible_labels) < len(values):
        visible_labels.append(f"模型{len(visible_labels) + 1}")
    roles = [model_role(label) for label in visible_labels[:len(values)]]
    raw_confidences = list(model_confidences or ())
    normalized_confidences: list[float] = []
    reliabilities: list[float] = []
    for index in range(len(values)):
        try:
            confidence = float(raw_confidences[index]) if index < len(raw_confidences) else 0.0
        except (TypeError, ValueError, OverflowError):
            confidence = 0.0
        if not math.isfinite(confidence) or confidence <= 0:
            normalized_confidences.append(0.0)
            reliabilities.append(1.0)
        else:
            confidence = max(0.0, min(1.0, confidence))
            normalized_confidences.append(confidence)
            # Confidence is supporting evidence, not a veto.  OCR engines use
            # different calibration scales, so keep the influence bounded.
            reliabilities.append(0.88 + confidence * 0.24)
    normalized = [standard_japanese_key(value) for value in values]
    if len(set(normalized)) <= 1:
        return CharacterFusionResult()

    exact_groups: dict[str, list[int]] = {}
    for index, value in enumerate(normalized):
        exact_groups.setdefault(value, []).append(index)
    exact_majority = max(exact_groups.values(), key=len)
    if len(exact_majority) >= 2:
        lengths = [len(value) for value in values]
        consensus_length = max(1, round(sum(lengths[index] for index in exact_majority) / len(exact_majority)))
        warnings: list[str] = []
        outliers = [index for index in range(len(values)) if index not in exact_majority]
        for index in outliers:
            if len(values[index]) < consensus_length * .70 and consensus_length - len(values[index]) >= 8:
                warnings.append(f"{visible_labels[index]}明显偏短，但另外两模型完全一致，按一致组保留")
            if _adjacent_repeat_score(values[index]) >= .34:
                warnings.append(f"{visible_labels[index]}疑似重复，但另外两模型完全一致，按一致组保留")
        return CharacterFusionResult(
            reason="两个模型逐字符完全一致，字符融合不生成重复候选",
            warnings=tuple(warnings),
            evidence={
                "exact_majority_model_indices": exact_majority,
                "physical_column_ids": [str(value) for value in (physical_column_ids or ())],
                "model_confidences": [round(value, 6) for value in normalized_confidences],
                "reliabilities": [round(value, 6) for value in reliabilities],
            },
        )

    pivot_index = _choose_pivot(values, roles, reliabilities)
    pivot = values[pivot_index]
    position_votes: list[list[tuple[int, str | None]]] = [
        [(pivot_index, char)] for char in pivot
    ]
    insertion_by_model: list[dict[int, str]] = [dict() for _ in values]
    for model_index, value in enumerate(values):
        if model_index == pivot_index:
            continue
        mapped, insertions = _alignment_votes(pivot, value)
        insertion_by_model[model_index] = insertions
        for position, char in enumerate(mapped):
            position_votes[position].append((model_index, char))
    # Pivot has no explicit insertion strings; add gap votes from every model at
    # every boundary so singleton additions remain identifiable.

    output: list[str] = []
    supported = 0
    total_decisions = 0
    all_different = 0
    unique_insertions = 0
    omitted_unique_chars = 0
    majority_deletions = 0
    retained_insertion_chars = 0

    for boundary in range(len(pivot) + 1):
        insertion, insertion_supported, insertion_unique = _fuse_insertions(
            boundary, insertion_by_model, roles, reliabilities,
        )
        if insertion:
            output.append(insertion)
            retained_insertion_chars += len(insertion)
        if insertion_unique:
            unique_insertions += 1
            if not insertion:
                omitted_unique_chars += max(
                    (len(mapping.get(boundary, "")) for mapping in insertion_by_model),
                    default=0,
                )
        if insertion:
            total_decisions += max(1, len(insertion))
            if insertion_supported:
                supported += max(1, len(insertion))
        if boundary == len(pivot):
            break

        votes = list(position_votes[boundary])
        present_models = {model for model, _char in votes}
        for model_index in range(len(values)):
            if model_index not in present_models:
                votes.append((model_index, _GAP))
        char, majority, different = _char_vote(votes, roles, reliabilities)
        total_decisions += 1
        if majority:
            supported += 1
        if different:
            all_different += 1
        if char is _GAP:
            if majority:
                majority_deletions += 1
            continue
        output.append(char)

    fused = "".join(output)
    if not fused:
        return CharacterFusionResult(
            review_required=True,
            local_reocr_recommended=True,
            reason="字符级对齐没有得到可靠文字",
            warnings=("三模型字符对齐失败，建议局部重识别",),
        )

    # If voting merely reproduces one source, keep the original source card; no
    # fourth card is needed, but still return risk diagnostics for review.
    pairwise = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            pairwise.append(SequenceMatcher(None, values[i], values[j], autojunk=False).ratio())
    mean_similarity = sum(pairwise) / max(1, len(pairwise))
    min_similarity = min(pairwise, default=0.0)
    lengths = [len(value) for value in values]
    shortest, longest = min(lengths), max(lengths)
    severe_length_gap = longest >= 16 and shortest / max(1, longest) < .66 and longest - shortest >= 9
    duplicate_models = [
        index for index, value in enumerate(values)
        if _adjacent_repeat_score(value) >= .34
    ]

    support_ratio = supported / max(1, total_decisions)
    output_quality = line_quality(fused).score
    confidence = support_ratio * .58 + mean_similarity * .24 + output_quality * .18
    confidence -= min(.18, all_different / max(1, total_decisions) * .65)
    confidence -= min(.10, unique_insertions * .025)
    if severe_length_gap:
        confidence -= .10
    if duplicate_models:
        # Duplicate detection increases the value of fusion but also means the
        # row deserves inspection unless the other two models strongly agree.
        confidence -= .025
    confidence = max(0.0, min(.99, confidence))

    differs_from_all = all(fused != value for value in values)
    sensitive_content = any(is_sensitive_text(value) for value in values)

    warnings: list[str] = []
    if severe_length_gap:
        short_index = lengths.index(shortest)
        warnings.append(
            f"{visible_labels[short_index]}明显偏短，疑似整段漏识"
        )
    for model_index in duplicate_models:
        warnings.append(f"{visible_labels[model_index]}疑似包含重复片段")
    if all_different:
        warnings.append(f"有 {all_different} 个字符位置三模型均不同")
    if unique_insertions:
        warnings.append(
            f"有 {unique_insertions} 处仅单模型插入，已保守过滤 {omitted_unique_chars} 字"
        )
    if majority_deletions:
        warnings.append(f"依据两模型空位删除 {majority_deletions} 个疑似多余字符")
    if sensitive_content and differs_from_all:
        warnings.append("数字、等级或结构化字段禁止自动字符融合，必须独立模型或图像复核")
    if physical_column_ids:
        warnings.append(f"字符融合锁定在 {len(tuple(physical_column_ids))} 个物理列 ID 内")

    strong = (
        differs_from_all
        and not sensitive_content
        and confidence >= .90
        and support_ratio >= .91
        and all_different <= max(1, round(total_decisions * .008))
        and unique_insertions <= 1
        and not severe_length_gap
        and min_similarity >= .60
    )
    local_reocr = (
        sensitive_content and differs_from_all
        or severe_length_gap
        or min_similarity < .48
        or all_different >= max(2, round(total_decisions * .025))
        or unique_insertions >= 3
        or len(duplicate_models) >= 1
    )
    review = not strong

    reason = (
        f"物理列内字符级对齐：多数支持 {supported}/{max(1, total_decisions)}，"
        f"平均相似度 {mean_similarity:.3f}"
    )
    if strong:
        reason += "；高置信字符融合自动采用"
    elif differs_from_all:
        reason += "；生成字符融合建议，保留三份原始 OCR 供人工选择"
    else:
        reason += "；融合结果等同现有模型，不新增候选"

    evidence = {
        "pivot_model_index": pivot_index,
        "pivot_model_label": visible_labels[pivot_index],
        "roles": roles,
        "model_confidences": [round(value, 6) for value in normalized_confidences],
        "reliabilities": [round(value, 6) for value in reliabilities],
        "support_ratio": round(support_ratio, 6),
        "mean_pairwise_similarity": round(mean_similarity, 6),
        "minimum_pairwise_similarity": round(min_similarity, 6),
        "all_different_positions": all_different,
        "unique_insertion_boundaries": unique_insertions,
        "omitted_unique_characters": omitted_unique_chars,
        "majority_deletions": majority_deletions,
        "retained_insertion_characters": retained_insertion_chars,
        "severe_length_gap": severe_length_gap,
        "sensitive_content": sensitive_content,
        "duplicate_model_indices": duplicate_models,
        "physical_column_ids": [str(value) for value in (physical_column_ids or ())],
    }

    # Do not add a redundant synthetic card when it exactly matches a model.
    candidate_text = fused if differs_from_all else ""
    return CharacterFusionResult(
        text=candidate_text,
        confidence=confidence,
        auto_select=bool(strong),
        review_required=bool(review or local_reocr),
        local_reocr_recommended=bool(local_reocr),
        reason=reason,
        warnings=tuple(dict.fromkeys(warnings)),
        evidence=evidence,
    )
