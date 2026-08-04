#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adaptive, diversity-aware OCR ensemble policy for Japanese books.

This module is intentionally independent of Qt and OCR runtimes.  It operates on
already aligned candidates and provides three things that the former simple
majority policy could not provide safely:

* one conservative Japanese comparison key shared by early consensus, the OCR
  compare workspace, and the V4 disagreement package;
* book-local model reliability estimated only from rows where independent
  models agree (a weakly supervised calibration anchor);
* diversity-aware weighted voting, so two highly correlated recognizers do not
  automatically outvote a different OCR family.

The policy never rewrites kanji, kana, digits, ranks, names, dashes or lexical
forms.  It can settle exact/Unicode-layout equivalence and strong independent
majorities.  Everything else remains reviewable or is routed to the optional
third OCR engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
import math
import re
from typing import Any, Mapping, Sequence

from engine.external_ocr import line_quality
from engine.ocr_unicode_standardizer import (
    japanese_ocr_comparison_key,
    normalize_japanese_ocr_text,
)

_PLACEHOLDER_RE = re.compile(r"^[\s□■�\x00]+$")
_JAPANESE_RE = re.compile(r"[ぁ-ゟ゠-ヿ一-龯々〆ヵヶ]")
_JP_SPACE_RE = re.compile(
    r"(?<=[ぁ-ゟ゠-ヿ一-龯々〆ヵヶ、。！？：；（）「」『』【】〈〉《》])"
    r"[ \t\u3000]+"
    r"(?=[ぁ-ゟ゠-ヿ一-龯々〆ヵヶ、。！？：；（）「」『』【】〈〉《》])"
)
_ASCII_JP_PUNCT = str.maketrans({
    "?": "？", "!": "！", ":": "：", ";": "；",
    "(": "（", ")": "）", "[": "［", "]": "］",
    "{": "｛", "}": "｝",
})
_SENSITIVE_RE = re.compile(
    r"(?:[0-9０-９]+|(?:Lv\.?|LEVEL|レベル)\s*[0-9０-９]+|"
    r"HP|MP|技能|スキル|装備|称号|職業|ランク|[A-Z]{2,})",
    re.I,
)


@dataclass(slots=True)
class ModelReliability:
    label: str
    family: str
    usable_ratio: float = 1.0
    anchor_count: int = 0
    anchor_correct: int = 0
    anchor_accuracy: float = 1.0
    wilson_lower: float = 0.5
    reliability: float = 1.0
    voting_enabled: bool = True
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.label,
            "family": self.family,
            "usable_ratio": round(self.usable_ratio, 6),
            "anchor_count": self.anchor_count,
            "anchor_correct": self.anchor_correct,
            "anchor_accuracy": round(self.anchor_accuracy, 6),
            "anchor_wilson_lower": round(self.wilson_lower, 6),
            "reliability": round(self.reliability, 6),
            "voting_enabled": self.voting_enabled,
            "reason": self.reason,
        }


@dataclass(slots=True)
class EnsembleDecision:
    status: str = "unresolved"
    chosen_index: int = 0
    chosen_text: str = ""
    chosen_key: str = ""
    confidence: float = 0.0
    support_count: int = 0
    family_support_count: int = 0
    score_margin: float = 0.0
    requires_more_models: bool = True
    requires_review: bool = True
    sensitive: bool = False
    reason: str = ""
    warnings: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)


# Families are intentionally coarse.  The goal is to avoid double-counting
# recognizers with the same decoding failure pattern, not to claim identical
# architectures.
def model_family(label: str) -> str:
    value = str(label or "").strip().casefold()
    if "ndl" in value:
        return "ndl_layout_recognizer"
    if "yomi" in value:
        return "yomitoku_document_ai"
    if "48px" in value or "manga_48" in value or "48 px" in value:
        return "manga48_ar"
    if "manga" in value:
        return "manga_transformer"
    if "apple" in value or "vision" in value or "macocr" in value or "macos" in value:
        return "apple_vision"
    if "paddle" in value:
        return "paddle_recognizer"
    if "google" in value:
        return "google_vision"
    if "tesseract" in value:
        return "tesseract"
    return "generic:" + (value or "unknown")


def canonical_japanese(text: str) -> str:
    value, _report = normalize_japanese_ocr_text(str(text or ""))
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    if _JAPANESE_RE.search(value):
        value = value.translate(_ASCII_JP_PUNCT)
        value = _JP_SPACE_RE.sub("", value)
    return value


def standard_japanese_key(text: str) -> str:
    canonical = canonical_japanese(text)
    key, _report = japanese_ocr_comparison_key(canonical)
    return re.sub(r"\s+", "", key)


def is_usable_candidate(text: str) -> bool:
    value = str(text or "").strip()
    return bool(value) and not bool(_PLACEHOLDER_RE.fullmatch(value))


def is_sensitive_text(text: str) -> bool:
    return bool(_SENSITIVE_RE.search(str(text or "")))


def _wilson_lower(correct: int, total: int, z: float = 1.0) -> float:
    if total <= 0:
        return 0.5
    p = correct / total
    denominator = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    spread = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return max(0.0, min(1.0, (centre - spread) / denominator))


def estimate_model_reliability(
    rows: Sequence[Sequence[str]],
    labels: Sequence[str],
) -> dict[str, ModelReliability]:
    """Estimate per-book reliability from independent-consensus anchors.

    An anchor exists only when at least two *other* usable model outputs share a
    comparison key.  A candidate is evaluated against that anchor.  This avoids
    treating a model as correct merely because it agrees with itself, and keeps
    calibration deterministic without ground truth.
    """
    model_labels = [str(label or f"model_{i + 1}") for i, label in enumerate(labels)]
    totals = [0] * len(model_labels)
    usable = [0] * len(model_labels)
    anchor_total = [0] * len(model_labels)
    anchor_correct = [0] * len(model_labels)

    for raw_row in rows:
        row = list(raw_row)
        while len(row) < len(model_labels):
            row.append("")
        keys = [standard_japanese_key(text) if is_usable_candidate(text) else "" for text in row[:len(model_labels)]]
        for i, text in enumerate(row[:len(model_labels)]):
            totals[i] += 1
            if is_usable_candidate(text):
                usable[i] += 1

        for i in range(len(model_labels)):
            if not keys[i]:
                continue
            other_groups: dict[str, set[str]] = defaultdict(set)
            other_counts: dict[str, int] = defaultdict(int)
            for j, key in enumerate(keys):
                if j == i or not key:
                    continue
                other_counts[key] += 1
                other_groups[key].add(model_family(model_labels[j]))
            anchors = [
                key for key, count in other_counts.items()
                if count >= 2 and len(other_groups[key]) >= 2
            ]
            if not anchors:
                continue
            anchor_key = max(anchors, key=lambda key: (other_counts[key], len(key)))
            anchor_total[i] += 1
            if keys[i] == anchor_key:
                anchor_correct[i] += 1

    output: dict[str, ModelReliability] = {}
    for i, label in enumerate(model_labels):
        use_ratio = usable[i] / max(1, totals[i])
        anchors = anchor_total[i]
        correct = anchor_correct[i]
        accuracy = correct / anchors if anchors else 1.0
        lower = _wilson_lower(correct, anchors)
        # Confidence scales differ greatly between OCR engines, so reliability
        # is learned from agreement anchors and kept in a narrow safe range.
        if anchors >= 20:
            reliability = 0.72 + lower * 0.48
        elif anchors:
            reliability = 0.88 + (accuracy - 0.5) * 0.20
        else:
            reliability = 1.0
        reliability *= 0.84 + min(1.0, use_ratio) * 0.16
        reliability = max(0.62, min(1.20, reliability))
        enabled = not (totals[i] >= 50 and use_ratio < 0.35)
        reason = ""
        if enabled and anchors >= 30 and accuracy < 0.55:
            enabled = False
            reason = "book_local_anchor_accuracy_below_0.55"
        elif not enabled:
            reason = "usable_candidate_ratio_below_0.35"
        elif anchors:
            reason = "book_local_consensus_calibration"
        else:
            reason = "insufficient_anchor_rows_neutral_weight"
        output[label] = ModelReliability(
            label=label,
            family=model_family(label),
            usable_ratio=use_ratio,
            anchor_count=anchors,
            anchor_correct=correct,
            anchor_accuracy=accuracy,
            wilson_lower=lower,
            reliability=reliability,
            voting_enabled=enabled,
            reason=reason,
        )
    return output


def _confidence_factor(value: float) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError, OverflowError):
        confidence = 0.0
    if not math.isfinite(confidence) or confidence <= 0.0:
        return 1.0
    confidence = max(0.0, min(1.0, confidence))
    # Bounded because model confidences are not calibrated on the same scale.
    return 0.94 + confidence * 0.12


def decide_ensemble(
    texts: Sequence[str],
    labels: Sequence[str] | None = None,
    confidences: Sequence[float] | None = None,
    reliabilities: Mapping[str, ModelReliability | Mapping[str, Any]] | None = None,
    *,
    require_independent_families: bool = True,
    verify_sensitive_two_model_agreement: bool = True,
) -> EnsembleDecision:
    values = [str(text or "") for text in texts]
    model_labels = list(labels or [])
    while len(model_labels) < len(values):
        model_labels.append(f"model_{len(model_labels) + 1}")
    model_labels = model_labels[:len(values)]
    confidence_values = list(confidences or ())
    while len(confidence_values) < len(values):
        confidence_values.append(0.0)

    candidates: list[dict[str, Any]] = []
    for index, (text, label) in enumerate(zip(values, model_labels)):
        if not is_usable_candidate(text):
            continue
        key = standard_japanese_key(text)
        if not key:
            continue
        raw_rel = (reliabilities or {}).get(label) if reliabilities else None
        if isinstance(raw_rel, ModelReliability):
            rel = raw_rel
        elif isinstance(raw_rel, Mapping):
            rel = ModelReliability(
                label=label,
                family=str(raw_rel.get("family") or model_family(label)),
                reliability=float(raw_rel.get("reliability", 1.0) or 1.0),
                voting_enabled=bool(raw_rel.get("voting_enabled", True)),
            )
        else:
            rel = ModelReliability(label=label, family=model_family(label))
        if not rel.voting_enabled:
            continue
        quality = line_quality(text)
        weight = rel.reliability * _confidence_factor(confidence_values[index])
        weight *= 0.90 + quality.score * 0.10
        candidates.append({
            "index": index,
            "label": label,
            "family": rel.family,
            "text": text,
            "canonical": canonical_japanese(text),
            "key": key,
            "weight": weight,
            "quality": quality.score,
            "warnings": quality.warnings,
        })

    if not candidates:
        return EnsembleDecision(
            status="no_usable_candidate", confidence=0.0,
            reason="所有候选均为空、占位符或已被模型健康门隔离",
            warnings=("没有可投票候选",),
        )
    if len(candidates) == 1:
        only = candidates[0]
        return EnsembleDecision(
            status="single_candidate", chosen_index=only["index"],
            chosen_text=only["canonical"], chosen_key=only["key"],
            confidence=min(0.56, 0.42 + only["quality"] * 0.12),
            support_count=1, family_support_count=1,
            requires_more_models=True, requires_review=True,
            sensitive=is_sensitive_text(only["text"]),
            reason="只有一个健康模型产生文字，必须补充模型或人工复核",
            warnings=tuple(only["warnings"]),
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate["key"]].append(candidate)

    scored_groups: list[tuple[float, str, list[dict[str, Any]], int]] = []
    for key, group in groups.items():
        by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for candidate in group:
            by_family[candidate["family"]].append(candidate)
        score = 0.0
        for family_candidates in by_family.values():
            family_candidates.sort(key=lambda item: item["weight"], reverse=True)
            score += family_candidates[0]["weight"]
            # Correlated recognizers still add evidence, but cannot count as an
            # independent full voter.
            for duplicate in family_candidates[1:]:
                score += duplicate["weight"] * 0.35
        scored_groups.append((score, key, group, len(by_family)))
    scored_groups.sort(key=lambda item: (item[0], len(item[2]), len(item[1])), reverse=True)
    winner_score, winner_key, winner_group, winner_families = scored_groups[0]
    runner_score = scored_groups[1][0] if len(scored_groups) > 1 else 0.0
    margin = winner_score - runner_score
    best = max(winner_group, key=lambda item: (item["quality"], item["weight"], -item["index"]))
    support = len(winner_group)
    all_same_key = len(groups) == 1
    all_same_raw = len({candidate["text"] for candidate in candidates}) == 1
    sensitive = any(is_sensitive_text(candidate["text"]) for candidate in candidates)
    independent = winner_families >= 2 or not require_independent_families
    warnings: list[str] = []
    for candidate in winner_group:
        warnings.extend(candidate["warnings"])

    if all_same_key:
        status = "exact_consensus" if all_same_raw else "normalized_consensus"
        needs_verification = bool(
            verify_sensitive_two_model_agreement
            and sensitive
            and len(candidates) == 2
        )
        if require_independent_families and winner_families < 2:
            needs_verification = True
            warnings.append("一致候选来自同一OCR家族，需要独立模型验证")
        confidence = 0.985 if len(candidates) >= 3 else 0.965
        if needs_verification:
            confidence = min(confidence, 0.86)
        return EnsembleDecision(
            status=status,
            chosen_index=best["index"], chosen_text=best["canonical"],
            chosen_key=winner_key, confidence=confidence,
            support_count=support, family_support_count=winner_families,
            score_margin=margin, requires_more_models=needs_verification,
            requires_review=False if not needs_verification else sensitive,
            sensitive=sensitive,
            reason=(
                "所有健康模型原文完全一致" if all_same_raw
                else "所有健康模型仅存在安全Unicode、全半角或版面空格差异"
            ),
            warnings=tuple(dict.fromkeys(warnings)),
            evidence={"winner_score": winner_score, "runner_score": runner_score},
        )

    if support >= 2 and independent:
        # With a dissenting candidate, keep the majority as provisional.  High-
        # risk numbers/status fields remain reviewable even after a 2:1 vote.
        critical_dissent = sensitive and len(candidates) >= 3
        confidence = 0.93 if support >= 3 else 0.89
        confidence += min(0.04, max(0.0, margin) * 0.03)
        if critical_dissent:
            confidence = min(confidence, 0.84)
            warnings.append("数字、等级或结构化字段存在少数模型异议")
        return EnsembleDecision(
            status="majority_consensus",
            chosen_index=best["index"], chosen_text=best["canonical"],
            chosen_key=winner_key, confidence=max(0.0, min(0.99, confidence)),
            support_count=support, family_support_count=winner_families,
            score_margin=margin, requires_more_models=False,
            requires_review=critical_dissent,
            sensitive=sensitive,
            reason=f"{support} 个候选、{winner_families} 个独立OCR家族形成多数",
            warnings=tuple(dict.fromkeys(warnings)),
            evidence={"winner_score": winner_score, "runner_score": runner_score},
        )

    # A weighted single-model winner is never auto accepted.  It is useful only
    # for choosing the best provisional display text while routing more compute.
    confidence = max(0.42, min(0.68, 0.50 + max(0.0, margin) * 0.08))
    return EnsembleDecision(
        status="unresolved",
        chosen_index=best["index"], chosen_text=best["canonical"],
        chosen_key=winner_key, confidence=confidence,
        support_count=support, family_support_count=winner_families,
        score_margin=margin, requires_more_models=True, requires_review=True,
        sensitive=sensitive,
        reason="健康模型没有形成独立多数，保留最佳候选但不自动定稿",
        warnings=tuple(dict.fromkeys(warnings)),
        evidence={
            "winner_score": winner_score,
            "runner_score": runner_score,
            "candidate_group_count": len(groups),
        },
    )
