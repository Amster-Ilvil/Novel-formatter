#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Column-level early consensus for multi-engine OCR.

The normal comparison workspace remains sentence based.  This module operates
one stage earlier, while every OCR result still has an immutable ``column_id``.
Two engines can therefore settle exact matches immediately, while only
conflicting physical columns are sent to an optional third engine or to later
sentence-context recovery.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from engine.external_ocr import line_quality
from engine.adaptive_ocr_ensemble import (
    ModelReliability, decide_ensemble, estimate_model_reliability,
    standard_japanese_key,
)
from models.document import Block, UnifiedDocument


@dataclass(frozen=True, slots=True)
class ColumnValue:
    text: str
    confidence: float
    block: Block | None = None


@dataclass(slots=True)
class ColumnConsensusPlan:
    ordered_ids: list[str] = field(default_factory=list)
    agreed_ids: set[str] = field(default_factory=set)
    conflict_ids: set[str] = field(default_factory=set)
    unresolved_ids: set[str] = field(default_factory=set)
    chosen: dict[str, ColumnValue] = field(default_factory=dict)
    support_count: dict[str, int] = field(default_factory=dict)
    model_count: int = 0
    exact_ids: set[str] = field(default_factory=set)
    normalized_ids: set[str] = field(default_factory=set)
    majority_ids: set[str] = field(default_factory=set)
    verification_ids: set[str] = field(default_factory=set)
    sensitive_dissent_ids: set[str] = field(default_factory=set)
    decision_status: dict[str, str] = field(default_factory=dict)
    decision_confidence: dict[str, float] = field(default_factory=dict)
    model_reliability: dict[str, dict[str, object]] = field(default_factory=dict)

    @property
    def exact_count(self) -> int:
        return len(self.agreed_ids)

    @property
    def conflict_count(self) -> int:
        return len(self.conflict_ids)

    @property
    def settled(self) -> bool:
        return bool(self.ordered_ids) and not self.conflict_ids

    def seed_payload(self) -> dict[str, dict[str, object]]:
        return {
            column_id: {
                "text": value.text,
                "confidence": float(value.confidence or 0.0),
            }
            for column_id, value in self.chosen.items()
        }


def _column_id(block: Block) -> str:
    metadata = block.metadata if isinstance(block.metadata, dict) else {}
    return str(metadata.get("column_id", "") or "")


def extract_column_values(doc: UnifiedDocument) -> tuple[list[str], dict[str, ColumnValue]]:
    """Extract one value per physical column from a pre-reflow OCR document."""
    ordered: list[str] = []
    values: dict[str, ColumnValue] = {}
    for block in doc.blocks:
        column_id = _column_id(block)
        if not column_id or column_id in values:
            continue
        ordered.append(column_id)
        metadata = block.metadata if isinstance(block.metadata, dict) else {}
        if bool(metadata.get("column_consensus_seeded", False)):
            continue
        values[column_id] = ColumnValue(
            text=str(block.text or ""),
            confidence=float(getattr(block, "confidence", 0.0) or 0.0),
            block=block,
        )
    return ordered, values


def _best_value(candidates: Iterable[ColumnValue]) -> ColumnValue:
    values = list(candidates)
    if not values:
        return ColumnValue("", 0.0, None)

    def score(value: ColumnValue) -> tuple[float, float, int]:
        quality = line_quality(value.text)
        return (
            float(quality.score),
            float(value.confidence or 0.0),
            len(standard_japanese_key(value.text)),
        )

    return max(values, key=score)


def _document_label(doc: UnifiedDocument, index: int) -> str:
    metadata = getattr(doc, "metadata", None)
    raw = getattr(metadata, "__dict__", {}) if metadata is not None else {}
    if isinstance(metadata, dict):
        raw = metadata
    return str(
        raw.get("multi_ocr_model_label")
        or raw.get("source_engine")
        or getattr(metadata, "source_engine", "")
        or f"OCR 模型 {index + 1}"
    )


def build_column_consensus(docs: Sequence[UnifiedDocument]) -> ColumnConsensusPlan:
    """Return adaptive consensus using immutable physical column IDs.

    The first two engines settle ordinary exact/normalised agreements.  Columns
    containing digits, levels or structured status text are selectively sent to
    an independent third engine even when two candidates agree.  With three
    engines, a diverse 2:1 majority is provisional; sensitive dissent remains
    reviewable and is never hidden by a raw confidence score.
    """
    plan = ColumnConsensusPlan(model_count=len(docs))
    if not docs:
        return plan

    extracted = [extract_column_values(doc) for doc in docs]
    plan.ordered_ids = list(extracted[0][0])
    maps = [item[1] for item in extracted]
    labels = [_document_label(doc, index) for index, doc in enumerate(docs)]

    reliability_rows: list[list[str]] = []
    all_ids: list[str] = list(plan.ordered_ids)
    for _ordered, mapping in extracted[1:]:
        for column_id in mapping:
            if column_id not in all_ids:
                all_ids.append(column_id)
    for column_id in all_ids:
        reliability_rows.append([mapping.get(column_id, ColumnValue("", 0.0)).text for mapping in maps])
    reliabilities = estimate_model_reliability(reliability_rows, labels)
    # A selectively invoked third model contains consensus-seeded blocks for
    # untouched columns.  Its low whole-book availability is intentional and
    # must not trip the full-pass health gate; only its real target outputs vote.
    for doc_index, doc in enumerate(docs):
        selective = any(
            bool((block.metadata if isinstance(block.metadata, dict) else {}).get("column_consensus_seeded", False))
            for block in doc.blocks
        )
        if selective and labels[doc_index] in reliabilities:
            rel = reliabilities[labels[doc_index]]
            rel.usable_ratio = 1.0
            rel.voting_enabled = True
            rel.reason = "selective_third_model_pass_not_full_book"
    plan.model_reliability = {label: value.as_dict() for label, value in reliabilities.items()}

    for column_id in plan.ordered_ids:
        values = [mapping.get(column_id, ColumnValue("", 0.0, None)) for mapping in maps]
        decision = decide_ensemble(
            [value.text for value in values],
            labels,
            [value.confidence for value in values],
            reliabilities,
            verify_sensitive_two_model_agreement=True,
        )
        plan.decision_status[column_id] = decision.status
        plan.decision_confidence[column_id] = float(decision.confidence)
        chosen_value = values[decision.chosen_index] if 0 <= decision.chosen_index < len(values) else _best_value(values)
        if not str(chosen_value.text or "").strip():
            chosen_value = _best_value(values)
        plan.chosen[column_id] = chosen_value
        plan.support_count[column_id] = int(decision.support_count)

        if decision.status == "exact_consensus":
            plan.exact_ids.add(column_id)
        elif decision.status == "normalized_consensus":
            plan.normalized_ids.add(column_id)
        elif decision.status == "majority_consensus":
            plan.majority_ids.add(column_id)

        if decision.requires_more_models:
            plan.conflict_ids.add(column_id)
            plan.unresolved_ids.add(column_id)
            plan.verification_ids.add(column_id)
            continue

        if decision.status in {"exact_consensus", "normalized_consensus", "majority_consensus"}:
            plan.agreed_ids.add(column_id)
            if decision.requires_review:
                plan.unresolved_ids.add(column_id)
                plan.sensitive_dissent_ids.add(column_id)
            continue

        plan.conflict_ids.add(column_id)
        plan.unresolved_ids.add(column_id)

    # Secondary/third documents may contain a column absent from the primary.
    # Preserve the primary structural contract and flag those extras for review.
    primary_ids = set(plan.ordered_ids)
    for _ordered, mapping in extracted[1:]:
        for column_id in mapping:
            if column_id not in primary_ids:
                plan.conflict_ids.add(column_id)
                plan.unresolved_ids.add(column_id)
                plan.decision_status[column_id] = "extra_secondary_column"
    return plan


def seed_from_document(doc: UnifiedDocument) -> dict[str, dict[str, object]]:
    _ordered, values = extract_column_values(doc)
    return {
        column_id: {
            "text": value.text,
            "confidence": float(value.confidence or 0.0),
        }
        for column_id, value in values.items()
    }
