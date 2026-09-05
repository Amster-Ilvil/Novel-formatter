#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conservative comparison and fusion for two or three OCR documents.

Model 1 remains the structural authority.  Fixed-region column OCR gets a
stronger path than ordinary text comparison: every model sees the same physical
column IDs, so alignment is first anchored by those IDs and only then grouped by
column-tail sentence boundaries.  One printed column is indivisible, and one
chapter-title column is always one atomic row.

When physical column IDs are unavailable (for imported/manual text), a monotonic
many-to-many dynamic-programming aligner handles 1↔N / N↔1 OCR line differences.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
from difflib import SequenceMatcher
import math
import re
import uuid
from typing import Sequence

from models.document import Block, BlockType, BoundingBox, UnifiedDocument
from engine.external_ocr import line_quality
from engine.adaptive_ocr_ensemble import canonical_japanese, decide_ensemble, standard_japanese_key
from engine.text_compare import (
    CompareLine,
    align_lines,
    line_similarity,
    looks_like_chapter_title,
    normalise_for_alignment,
)
from engine.column_sentence_reflow import has_sentence_terminal, join_column_parts, normalize_column_text

_TRAILING_CLOSERS = set("」』】）》〉〕］）】")
_TITLE_TYPES = {BlockType.CHAPTER, BlockType.SECTION, BlockType.TOC_ENTRY}
_TEXT_TYPES = {
    BlockType.PARAGRAPH, BlockType.DIALOGUE, BlockType.CHAPTER,
    BlockType.SECTION, BlockType.RUBY, BlockType.TOC_ENTRY,
}
_MAX_GROUP_SPAN = 3


def _block_metadata(block: Block) -> dict:
    """Return metadata safely for legacy/corrupted project documents."""
    value = getattr(block, "metadata", None)
    return value if isinstance(value, dict) else {}


@dataclass(slots=True)
class OcrSentenceUnit:
    uid: str
    text: str
    block_index: int
    block_id: str
    block_type: str
    page: int
    segment_index: int
    atomic: bool = False
    confidence: float = 0.0


@dataclass(slots=True)
class MultiOcrRow:
    index: int
    texts: list[str]
    model_confidences: tuple[float, ...] = ()
    chosen_index: int = 0
    confidence: float = 0.0
    reason: str = ""
    warnings: tuple[str, ...] = ()
    primary_unit_index: int | None = None
    primary_block_index: int | None = None
    primary_block_indices: tuple[int, ...] = ()
    primary_block_id: str = ""
    primary_segment_index: int = 0
    block_type: str = BlockType.PARAGRAPH.value
    page: int = 0
    insert_before_block_index: int | None = None
    column_ids: tuple[str, ...] = ()
    atomic: bool = False
    alignment_repaired: bool = False
    alignment_notes: tuple[str, ...] = ()
    alignment_status: str = "unreviewed"
    sentence_group_id: str = ""
    repair_reason: str = ""
    consensus_seeded_models: tuple[int, ...] = ()
    character_fused_text: str = ""
    character_fusion_confidence: float = 0.0
    character_fusion_reason: str = ""
    character_fusion_warnings: tuple[str, ...] = ()
    character_fusion_auto_selected: bool = False
    local_reocr_recommended: bool = False
    character_fusion_evidence: dict = field(default_factory=dict)
    # Optional equivalence keys created by the OCR compare "标准化修复" action.
    # They are never exported and are used only while their source snapshot still
    # matches ``texts``; any later manual edit automatically invalidates them.
    comparison_keys: tuple[str, ...] = ()
    comparison_key_sources: tuple[str, ...] = ()
    # Imported per-model adjudication may make all current OCR texts identical.
    # Keep the pre-correction candidates as immutable review history so the
    # resolved sentence remains discoverable without reopening it as pending.
    source_correction_resolved: bool = False
    historical_ocr_texts: tuple[str, ...] = ()
    historical_ocr_labels: tuple[str, ...] = ()
    historical_ocr_disagreement: bool = False
    historical_resolution_reason: str = ""
    historical_resolution_confidence: float = 0.0

    def effective_comparison_values(self) -> list[str]:
        sources = tuple(str(value or "") for value in self.texts)
        if (
            self.comparison_keys
            and len(self.comparison_keys) == len(sources)
            and tuple(self.comparison_key_sources) == sources
        ):
            return [str(value or "") for value in self.comparison_keys]
        return [normalise_for_alignment(value) for value in sources]

    @property
    def output_text(self) -> str:
        if self.character_fusion_auto_selected and self.character_fused_text:
            return self.character_fused_text
        if 0 <= self.chosen_index < len(self.texts):
            return self.texts[self.chosen_index]
        return next((text for text in self.texts if text), "")

    def _comparison_values(self) -> list[tuple[int, str]]:
        values = list(self.effective_comparison_values())
        while len(values) < len(self.texts):
            values.append("")
        return [(index, str(values[index] or "")) for index in range(len(self.texts))]

    def _nonempty_comparison_values(self) -> list[tuple[int, str]]:
        return [
            (index, value)
            for index, value in self._comparison_values()
            if str(self.texts[index] or "").strip()
        ]

    def _independent_comparison_values(self) -> list[tuple[int, str]]:
        seeded = {
            int(index) for index in (self.consensus_seeded_models or ())
            if 0 <= int(index) < len(self.texts)
        }
        return [
            (index, value)
            for index, value in self._comparison_values()
            if index not in seeded
        ]

    @property
    def is_conflict(self) -> bool:
        """Whether independently executed OCR models actually disagree.

        Empty output from one executed model versus text from another is a real
        disagreement and must enter review. A fast-consensus placeholder is
        excluded because it is copied evidence, not an independent OCR result.
        """
        independent = [value for _index, value in self._independent_comparison_values()]
        return len(independent) >= 2 and len(set(independent)) > 1

    @property
    def provisional_consensus(self) -> bool:
        """Two independent models agree while at least one model was skipped.

        This is deliberately neither a three-model exact consensus nor a text
        disagreement. It remains reviewable because the seeded model did not
        independently recognise the physical columns. Empty agreement is kept
        under the ordinary true-empty audit instead of being called a candidate.
        """
        seeded = {
            int(index) for index in (self.consensus_seeded_models or ())
            if 0 <= int(index) < len(self.texts)
        }
        if not seeded or self.is_conflict:
            return False
        independent = [value for _index, value in self._independent_comparison_values()]
        return (
            len(independent) >= 2
            and len(set(independent)) == 1
            and bool(str(independent[0] or "").strip())
        )

    @property
    def exact_consensus(self) -> bool:
        if self.consensus_seeded_models:
            return False
        independent = [value for _index, value in self._independent_comparison_values()]
        return len(independent) >= 2 and len(set(independent)) == 1

    @property
    def review_classification(self) -> str:
        if self.is_conflict:
            return "conflict"
        if self.provisional_consensus:
            return "provisional_consensus"
        return "exact_consensus"


@dataclass(slots=True)
class MultiOcrComparison:
    labels: list[str]
    rows: list[MultiOcrRow] = field(default_factory=list)
    exact_rows: int = 0
    conflict_rows: int = 0
    provisional_consensus_rows: int = 0
    low_confidence_rows: int = 0
    insertion_rows: int = 0
    alignment_mode: str = "text_many_to_many"
    column_anchored_rows: int = 0
    chapter_atomic_rows: int = 0
    alignment_shift_repairs: int = 0
    unresolved_empty_cells: int = 0
    alignment_revision: int = 2
    physical_column_source: str = "source_column_primary_texts"
    true_empty_rows: int = 0
    single_model_only_rows: int = 0
    character_fused_rows: int = 0
    character_auto_selected_rows: int = 0
    local_reocr_rows: int = 0

    @property
    def summary(self) -> str:
        mode = (
            f"物理列锚定 {self.column_anchored_rows} 句"
            if self.alignment_mode == "column_id_consensus"
            else "字符相似度多对多对齐"
        )
        title = f"；章节原子句 {self.chapter_atomic_rows}" if self.chapter_atomic_rows else ""
        repairs = (
            f"；相邻错位修复 {self.alignment_shift_repairs} 处"
            if self.alignment_shift_repairs else ""
        )
        empty = (
            f"；仍有单模型空白 {self.unresolved_empty_cells} 处"
            if self.unresolved_empty_cells else ""
        )
        true_empty = f"；全模型空句 {self.true_empty_rows} 处" if self.true_empty_rows else ""
        single_only = (
            f"；仅单模型有字 {self.single_model_only_rows} 句"
            if self.single_model_only_rows else ""
        )
        char_fused = (
            f"；字符融合建议 {self.character_fused_rows} 句（自动采用 {self.character_auto_selected_rows}）"
            if self.character_fused_rows else ""
        )
        local_reocr = (
            f"；建议局部重识别 {self.local_reocr_rows} 句"
            if self.local_reocr_rows else ""
        )
        return (
            f"{mode}{title}{repairs}{empty}{true_empty}{single_only}{char_fused}{local_reocr}；共 {len(self.rows)} 行；"
            f"真正一致 {self.exact_rows}；两模型共同候选 {self.provisional_consensus_rows}；"
            f"真正分歧 {self.conflict_rows}；低置信 {self.low_confidence_rows}；"
            f"其他模型独有句 {self.insertion_rows}。"
        )


@dataclass(slots=True)
class _ColumnRecord:
    column_id: str
    text: str
    block_index: int
    block_id: str
    block_type: str
    page: int
    position_in_block: int
    block_column_count: int
    terminal: bool
    atomic_title: bool
    consensus_seeded: bool = False
    confidence: float = 0.0

    @property
    def block_tail(self) -> bool:
        return self.position_in_block == self.block_column_count - 1


@dataclass(slots=True)
class _ColumnExtraction:
    records: list[_ColumnRecord]
    sentence_candidates: dict[tuple[str, ...], str]
    terminal_candidate_end_ids: set[str]
    used_primary_texts: bool = False


@dataclass(slots=True)
class _AlignmentGroup:
    primary_indices: tuple[int, ...]
    secondary_indices: tuple[int, ...]


def _split_sentences(text: str) -> list[str]:
    """Legacy punctuation splitter for non-column documents only.

    Fixed-region/column-reflow blocks and chapter titles never call this
    function.  This preserves the hard rule that punctuation inside one physical
    column cannot create another comparison row.
    """
    value = str(text or "").replace("\r", "").strip()
    if not value:
        return []
    result: list[str] = []
    start = 0
    for pos, char in enumerate(value):
        if char not in "。！？!?":
            continue
        end = pos + 1
        while end < len(value) and value[end] in _TRAILING_CLOSERS:
            end += 1
        segment = value[start:end].strip()
        if segment:
            result.append(segment)
        start = end
    tail = value[start:].strip()
    if tail:
        result.append(tail)
    return result or [value]


def _block_is_atomic(block: Block, doc: UnifiedDocument) -> bool:
    metadata = _block_metadata(block)
    return bool(
        block.type in _TITLE_TYPES
        or metadata.get("chapter_title_atomic")
        or metadata.get("atomic_ocr_sentence")
        or metadata.get("column_sentence_reflow")
        or getattr(doc.metadata, "manual_ocr_alignment_lines", False)
    )


def sentence_units(doc: UnifiedDocument) -> list[OcrSentenceUnit]:
    units: list[OcrSentenceUnit] = []
    for block_index, block in enumerate(doc.blocks):
        if block.type not in _TEXT_TYPES or block.type == BlockType.IMAGE_REF:
            continue
        atomic = _block_is_atomic(block, doc)
        parts = [str(block.text or "").strip()] if atomic else _split_sentences(block.text)
        for segment_index, part in enumerate(part for part in parts if part):
            units.append(OcrSentenceUnit(
                uid=f"u{block_index}:{segment_index}",
                text=part,
                block_index=block_index,
                block_id=block.id,
                block_type=block.type.value,
                page=int(getattr(block, "page", 0) or 0),
                segment_index=segment_index,
                atomic=atomic,
                confidence=float(getattr(block, "confidence", 0.0) or 0.0),
            ))
    return units


def _as_compare_lines(units: Sequence[OcrSentenceUnit]) -> list[CompareLine]:
    return [
        CompareLine(
            text=unit.text,
            block_ids=[unit.uid],
            block_indices=[index],
            block_type=unit.block_type,
            page=unit.page,
        )
        for index, unit in enumerate(units)
    ]


def _auto_choose(
    texts: Sequence[str], comparison_values: Sequence[str] | None = None,
    labels: Sequence[str] | None = None,
    model_confidences: Sequence[float] | None = None,
) -> tuple[int, float, str, tuple[str, ...]]:
    available = [(idx, text) for idx, text in enumerate(texts) if str(text or "").strip()]
    compare_values = list(comparison_values or ())

    def compare_value(index: int, text: str) -> str:
        if 0 <= index < len(compare_values):
            return str(compare_values[index] or "")
        return normalise_for_alignment(text)

    if not available:
        return 0, 0.0, "所有模型均为空", ("空文本",)
    if len(available) == 1:
        idx, text = available[0]
        q = line_quality(text)
        length = len(compare_value(idx, text))
        warnings = list(q.warnings)
        warnings.append("其他模型对应物理列为空")
        if length >= 40:
            warnings.append("长句仅有单模型结果，可能存在整段漏识")
        confidence = min(.56, .45 + q.score * .11)
        if length >= 80:
            confidence = min(confidence, .50)
        return (
            idx,
            confidence,
            "仅该模型的对应物理列识别到文字；保留候选并强制进入人工复核",
            tuple(dict.fromkeys(warnings)),
        )

    adaptive = decide_ensemble(
        texts, labels, model_confidences,
        verify_sensitive_two_model_agreement=False,
    )
    if adaptive.status in {"exact_consensus", "normalized_consensus", "majority_consensus"}:
        adaptive_warnings = list(adaptive.warnings or ())
        if adaptive.requires_review:
            adaptive_warnings.append("高风险字段存在模型异议，自动结果仅作暂定")
        return (
            int(adaptive.chosen_index),
            float(adaptive.confidence),
            str(adaptive.reason),
            tuple(dict.fromkeys(adaptive_warnings)),
        )

    groups: dict[str, list[tuple[int, str]]] = {}
    for idx, text in available:
        groups.setdefault(compare_value(idx, text), []).append((idx, text))
    consensus = max(groups.values(), key=len)
    if len(consensus) >= 2:
        best_idx, best_text = max(consensus, key=lambda item: line_quality(item[1]).score)
        warnings = list(line_quality(best_text).warnings)
        missing_count = max(0, len(texts) - len(available))
        if missing_count:
            warnings.append(f"{missing_count} 个模型对应物理列为空")
        confidence = .98 if len(consensus) == len(available) else .91
        if missing_count:
            confidence = min(confidence, .86)
        return best_idx, confidence, (
            f"{len(consensus)} 个模型文字一致，采用一致组中噪声最少的结果"
        ), tuple(dict.fromkeys(warnings))

    lengths = sorted(len(compare_value(idx, text)) for idx, text in available)
    median_length = lengths[len(lengths) // 2] or 1
    scored: list[tuple[float, int, tuple[str, ...]]] = []
    for idx, text in available:
        quality = line_quality(text)
        similarities = [
            line_similarity(compare_value(idx, text), compare_value(j, other))
            for j, other in available
            if j != idx
        ]
        consensus_score = sum(similarities) / max(1, len(similarities))
        length = len(compare_value(idx, text))
        length_ratio = min(length, median_length) / max(length, median_length, 1)
        score = quality.score * .58 + consensus_score * .27 + length_ratio * .15
        if idx == 0:
            score += .025
        scored.append((score, idx, quality.warnings))
    scored.sort(reverse=True)
    top_score, top_idx, warnings = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    gap = top_score - second_score
    confidence = max(.42, min(.89, .53 + gap * 1.8 + top_score * .18))
    reason = "综合日文字符质量、异常符号、句长完整度及模型间相似度自动选优"
    normalized_lengths = [len(compare_value(idx, text)) for idx, text in available]
    shortest = min(normalized_lengths, default=0)
    longest = max(normalized_lengths, default=0)
    severe_length_gap = (
        longest >= 12
        and longest - shortest >= 10
        and shortest / max(1, longest) < .55
    )
    warning_values = list(warnings)
    if severe_length_gap:
        reason += "；候选长度差异过大，可能存在整段漏识或相邻句错位"
        warning_values.append("候选长度差异过大：检查整段漏识或相邻句错位")
        confidence = min(confidence, .54)
    if gap < .045:
        reason += "；候选接近，建议人工确认"
        confidence = min(confidence, .58)
    return top_idx, confidence, reason, tuple(dict.fromkeys(warning_values))


def _finalize_comparison(comparison: MultiOcrComparison) -> MultiOcrComparison:
    comparison.exact_rows = 0
    comparison.conflict_rows = 0
    comparison.provisional_consensus_rows = 0
    comparison.low_confidence_rows = 0
    comparison.insertion_rows = 0
    comparison.chapter_atomic_rows = 0
    comparison.unresolved_empty_cells = 0
    comparison.true_empty_rows = 0
    comparison.single_model_only_rows = 0
    comparison.character_fused_rows = 0
    comparison.character_auto_selected_rows = 0
    comparison.local_reocr_rows = 0
    from engine.character_level_fusion import build_character_fusion
    for index, row in enumerate(comparison.rows):
        row.index = index
        choice, confidence, reason, warnings = _auto_choose(
            row.texts, row.effective_comparison_values(),
            comparison.labels, row.model_confidences,
        )
        if row.alignment_repaired:
            reason += "；已依据相邻物理列和其他模型候选修复一行偏移"
            warnings = tuple(dict.fromkeys((*warnings, *row.alignment_notes)))
            confidence = min(confidence, .88)
        if row.consensus_seeded_models:
            seeded_labels = "、".join(
                f"模型{index + 1}" for index in row.consensus_seeded_models
            )
            reason += f"；{seeded_labels}对已一致物理列未重复推理，沿用前序共识底稿"
            warnings = tuple(dict.fromkeys((
                *warnings,
                f"{seeded_labels}含共识复用列（不是重复 OCR 输出）",
            )))
        fusion = build_character_fusion(
            row.texts,
            comparison.labels,
            physical_column_ids=row.column_ids,
            model_confidences=row.model_confidences,
        )
        row.character_fused_text = str(fusion.text or "")
        row.character_fusion_confidence = float(fusion.confidence or 0.0)
        row.character_fusion_reason = str(fusion.reason or "")
        row.character_fusion_warnings = tuple(fusion.warnings or ())
        row.character_fusion_auto_selected = bool(fusion.auto_select and fusion.text)
        row.local_reocr_recommended = bool(fusion.local_reocr_recommended)
        row.character_fusion_evidence = dict(fusion.evidence or {})
        if row.character_fusion_reason and row.character_fusion_warnings:
            reason += f"；{row.character_fusion_reason}"
            warnings = tuple(dict.fromkeys((*warnings, *row.character_fusion_warnings)))
        if row.character_fused_text:
            comparison.character_fused_rows += 1
            if row.character_fusion_reason and not row.character_fusion_warnings:
                reason += f"；{row.character_fusion_reason}"
            if row.character_fusion_auto_selected:
                comparison.character_auto_selected_rows += 1
                confidence = max(confidence, row.character_fusion_confidence)
            else:
                confidence = min(confidence, max(.52, row.character_fusion_confidence))
        if row.local_reocr_recommended:
            comparison.local_reocr_rows += 1
            warnings = tuple(dict.fromkeys((*warnings, "三模型证据仍冲突，建议只对本物理列局部重识别")))
            confidence = min(confidence, .59)
        row.chosen_index = choice
        row.confidence = confidence
        row.reason = reason
        row.warnings = warnings
        if row.is_conflict:
            comparison.conflict_rows += 1
        elif row.provisional_consensus:
            comparison.provisional_consensus_rows += 1
            reason += "；两个独立模型文字一致，第3模型由快速共识复用，仍需 AI 或人工确认"
            warnings = tuple(dict.fromkeys((
                *warnings,
                "两模型共同候选：跳过的模型未提供独立 OCR 证据",
            )))
            confidence = min(confidence, .90)
            row.reason = reason
            row.warnings = warnings
            row.confidence = confidence
        elif row.exact_consensus:
            comparison.exact_rows += 1
        else:
            # Defensive fallback for malformed/legacy rows with insufficient
            # independent evidence. Keep them reviewable rather than inflating
            # the exact-consensus statistic.
            comparison.conflict_rows += 1
            reason += "；独立 OCR 证据不足，不能计入真正一致"
            warnings = tuple(dict.fromkeys((*warnings, "独立 OCR 证据不足：需要人工确认")))
            confidence = min(confidence, .49)
            row.reason = reason
            row.warnings = warnings
            row.confidence = confidence
        if confidence < .60:
            comparison.low_confidence_rows += 1
        if row.primary_unit_index is None and row.primary_block_index is None:
            comparison.insertion_rows += 1
        if row.atomic and row.block_type in {item.value for item in _TITLE_TYPES}:
            comparison.chapter_atomic_rows += 1
        nonempty = sum(1 for text in row.texts if str(text or "").strip())
        if not row.sentence_group_id:
            group_seed = "|".join((
                str(row.page or 0),
                str(row.primary_block_id or ""),
                str(row.primary_segment_index or 0),
                ",".join(str(value) for value in (row.column_ids or ())),
            ))
            row.sentence_group_id = f"sg:{uuid.uuid5(uuid.NAMESPACE_URL, group_seed).hex[:16]}"
        if row.alignment_repaired:
            row.alignment_status = "shifted"
            row.repair_reason = row.repair_reason or "cross_column_shift"
        elif nonempty == 0:
            row.alignment_status = "true_empty"
            comparison.true_empty_rows += 1
        elif nonempty < len(row.texts):
            row.alignment_status = "single_model_only"
            comparison.single_model_only_rows += 1
            comparison.unresolved_empty_cells += max(0, len(row.texts) - nonempty)
        elif row.character_fusion_auto_selected:
            row.alignment_status = "character_fused"
        elif row.local_reocr_recommended:
            row.alignment_status = "local_reocr_recommended"
        elif row.is_conflict:
            row.alignment_status = "conflict"
        elif row.provisional_consensus:
            row.alignment_status = "provisional_consensus"
        else:
            row.alignment_status = "exact"
    return comparison


def _metadata_list(metadata: dict, key: str) -> list[str]:
    if not isinstance(metadata, dict):
        return []
    value = metadata.get(key) or []
    if isinstance(value, str):
        value = [value]
    elif not isinstance(value, (list, tuple, set)):
        return []
    return [str(item) for item in value]


def _alignment_seed_flags(metadata: dict, column_ids: Sequence[str]) -> list[bool]:
    values = metadata.get("source_column_consensus_seed_flags") or []
    if isinstance(values, (list, tuple)) and len(values) == len(column_ids):
        return [bool(value) for value in values]
    flag = bool(metadata.get("column_consensus_seeded", False))
    return [flag for _column_id in column_ids]


def _alignment_column_texts(
    metadata: dict,
    column_ids: Sequence[str],
    block_text: str,
) -> tuple[list[str], bool]:
    """Return immutable per-physical-column text for model alignment.

    Sentence-context re-OCR deliberately stores the corrected full sentence in
    ``source_column_texts`` with preceding empty cells.  That representation is
    useful for rendering one final sentence, but it is not suitable for
    cross-model alignment because different OCR engines may choose different
    terminal columns.  ``source_column_primary_texts`` keeps the original text
    of every physical column and is therefore the authoritative alignment
    source whenever its cardinality matches the immutable column IDs.
    """
    primary = _metadata_list(metadata, "source_column_primary_texts")
    if len(primary) == len(column_ids) and any(str(value or "").strip() for value in primary):
        return [str(value or "") for value in primary], True

    effective = _metadata_list(metadata, "source_column_texts")
    if len(effective) == len(column_ids):
        return [str(value or "") for value in effective], False

    if len(column_ids) == 1:
        return [str(block_text or "")], False
    return [], False


def _candidate_compatible_with_columns(candidate: str, column_texts: Sequence[str]) -> bool:
    value = str(candidate or "").strip()
    if not value:
        return False
    baseline = join_column_parts(column_texts)
    if not baseline:
        return True
    candidate_norm = normalise_for_alignment(value)
    baseline_norm = normalise_for_alignment(baseline)
    if not candidate_norm or not baseline_norm:
        return False
    ratio = min(len(candidate_norm), len(baseline_norm)) / max(
        1, len(candidate_norm), len(baseline_norm),
    )
    similarity = line_similarity(value, baseline)
    # Context re-OCR may repair many characters, but an accepted candidate must
    # still describe the same physical-column span rather than a neighbouring
    # sentence or stale page result.
    return similarity >= .34 and ratio >= .42


def _sentence_candidate_for_block(
    block: Block,
    column_ids: Sequence[str],
    column_texts: Sequence[str],
) -> str:
    metadata = _block_metadata(block)
    candidates: list[str] = []
    context_applied = bool(
        metadata.get("sentence_context_reocr_applied")
        or metadata.get("sentence_context_reocr_accepted")
    )
    if context_applied:
        candidates.extend([
            str(metadata.get("sentence_context_reocr_text", "") or ""),
            str(metadata.get("sentence_context_reocr_candidate", "") or ""),
        ])

    effective = _metadata_list(metadata, "source_column_texts")
    if len(effective) == len(column_ids):
        effective_joined = join_column_parts(effective)
        primary_joined = join_column_parts(column_texts)
        if effective_joined and normalise_for_alignment(effective_joined) != normalise_for_alignment(primary_joined):
            candidates.append(effective_joined)

    # An accepted context-reOCR block may expose the corrected text only through
    # ``block.text`` in some legacy versions.  Ordinary/title blocks stay on the
    # joined physical-column representation so whitespace normalization and
    # previous comparison behaviour remain unchanged.
    if context_applied:
        candidates.append(str(block.text or ""))
    for candidate in candidates:
        if _candidate_compatible_with_columns(candidate, column_texts):
            # Apply the same vertical-layout whitespace normalization as the
            # physical-column path.  This keeps chapter/title rows and accepted
            # sentence-context candidates byte-compatible with the old output.
            return join_column_parts([str(candidate or "")])
    return ""


def _extract_column_records(doc: UnifiedDocument) -> _ColumnExtraction | None:
    """Return exact per-column records and exact-span sentence candidates.

    ``None`` means the document lacks enough one-to-one column metadata and must
    use text alignment instead.  Empty record lists are valid for empty docs.
    """
    records: list[_ColumnRecord] = []
    sentence_candidates: dict[tuple[str, ...], str] = {}
    terminal_candidate_end_ids: set[str] = set()
    used_primary_texts = False
    seen: set[str] = set()
    for block_index, block in enumerate(doc.blocks):
        if block.type not in _TEXT_TYPES:
            continue
        metadata = _block_metadata(block)
        column_ids = _metadata_list(metadata, "source_column_ids")
        if not column_ids:
            column_id = str(metadata.get("column_id", "") or "")
            if column_id:
                column_ids = [column_id]
        if not column_ids:
            return None

        column_texts, used_primary = _alignment_column_texts(metadata, column_ids, block.text)
        if len(column_texts) != len(column_ids):
            return None
        seed_flags = _alignment_seed_flags(metadata, column_ids)
        comparison_texts = [str(text or "") for text in column_texts]
        used_primary_texts = used_primary_texts or used_primary

        candidate = _sentence_candidate_for_block(block, column_ids, comparison_texts)
        column_key = tuple(column_ids)
        if candidate:
            sentence_candidates[column_key] = candidate
            if has_sentence_terminal(candidate):
                terminal_candidate_end_ids.add(column_ids[-1])

        raw_flags = metadata.get("source_column_terminal_flags") or []
        terminal_flags = [bool(value) for value in raw_flags] if isinstance(raw_flags, list) else []
        atomic_title = bool(block.type in _TITLE_TYPES or metadata.get("chapter_title_atomic"))
        for position, (column_id, text) in enumerate(zip(column_ids, comparison_texts)):
            if not column_id or column_id in seen:
                return None
            seen.add(column_id)
            terminal = (
                True if atomic_title
                else terminal_flags[position] if position < len(terminal_flags)
                else has_sentence_terminal(text)
            )
            records.append(_ColumnRecord(
                column_id=column_id,
                text=str(text or ""),
                block_index=block_index,
                block_id=block.id,
                block_type=block.type.value,
                page=int(getattr(block, "page", 0) or 0),
                position_in_block=position,
                block_column_count=len(column_ids),
                terminal=terminal,
                atomic_title=atomic_title,
                consensus_seeded=bool(
                    seed_flags[position] if position < len(seed_flags) else False
                ),
                confidence=float(getattr(block, "confidence", 0.0) or 0.0),
            ))
    return _ColumnExtraction(
        records=records,
        sentence_candidates=sentence_candidates,
        terminal_candidate_end_ids=terminal_candidate_end_ids,
        used_primary_texts=used_primary_texts,
    )


def _compare_by_shared_columns(
    docs: Sequence[UnifiedDocument],
    labels: Sequence[str],
) -> MultiOcrComparison | None:
    extractions = [_extract_column_records(doc) for doc in docs]
    if any(extraction is None for extraction in extractions):
        return None
    bundles = [extraction for extraction in extractions if extraction is not None]
    primary_records = bundles[0].records if bundles else []
    if not primary_records:
        return MultiOcrComparison(labels=list(labels), alignment_mode="column_id_consensus")

    model_maps: list[dict[str, _ColumnRecord]] = []
    primary_ids = [record.column_id for record in primary_records]
    primary_id_set = set(primary_ids)
    for extraction in bundles:
        mapping = {record.column_id: record for record in extraction.records}
        # Shared fixed-column preparation should make coverage effectively 100%.
        # Keep a tolerant floor for imported legacy runs, otherwise fall back to
        # the text many-to-many aligner rather than producing a false exact map.
        coverage = len(primary_id_set.intersection(mapping)) / max(1, len(primary_id_set))
        if coverage < .80:
            return None
        model_maps.append(mapping)

    comparison = MultiOcrComparison(
        labels=list(labels),
        alignment_mode="column_id_consensus",
        physical_column_source=(
            "source_column_primary_texts"
            if any(extraction.used_primary_texts for extraction in bundles)
            else "source_column_texts"
        ),
    )
    pending: list[_ColumnRecord] = []
    segment_counts: dict[int, int] = {}

    def emit(records: Sequence[_ColumnRecord], *, atomic: bool = False) -> None:
        if not records:
            return
        column_ids = tuple(record.column_id for record in records)
        texts: list[str] = []
        model_confidences: list[float] = []
        for extraction, mapping in zip(bundles, model_maps):
            physical_text = join_column_parts(
                mapping[column_id].text
                for column_id in column_ids
                if column_id in mapping
            )
            # A sentence-context candidate may improve character accuracy, but
            # it is eligible only when it covers this exact canonical physical
            # column tuple.  A model whose reflow ended one column later cannot
            # move that full sentence into the neighbouring comparison row.
            candidate_text = extraction.sentence_candidates.get(column_ids, physical_text)
            texts.append(candidate_text)
            confidence_values = [
                float(mapping[column_id].confidence or 0.0)
                for column_id in column_ids
                if column_id in mapping and float(mapping[column_id].confidence or 0.0) > 0
            ]
            # Manual-review placeholders are structural omission markers, not a
            # model prediction.  A sentence row can contain one valid physical
            # column and one unresolved ``□`` column; averaging only the positive
            # confidence values previously gave strings such as ``□□`` a 0.77
            # confidence in the fusion package.  Any unresolved marker makes the
            # complete model candidate unsupported and therefore zero-confidence.
            if "□" in candidate_text or "�" in candidate_text:
                model_confidences.append(0.0)
            else:
                model_confidences.append(
                    sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
                )
        seeded_models = tuple(
            model_index
            for model_index, mapping in enumerate(model_maps)
            if any(
                column_id in mapping and mapping[column_id].consensus_seeded
                for column_id in column_ids
            )
        )
        first = records[0]
        segment_index = segment_counts.get(first.block_index, 0)
        segment_counts[first.block_index] = segment_index + 1
        comparison.rows.append(MultiOcrRow(
            index=len(comparison.rows),
            texts=texts,
            model_confidences=tuple(model_confidences),
            primary_unit_index=len(comparison.rows),
            primary_block_index=first.block_index,
            primary_block_indices=tuple(dict.fromkeys(record.block_index for record in records)),
            primary_block_id=first.block_id,
            primary_segment_index=segment_index,
            block_type=first.block_type,
            page=first.page,
            column_ids=column_ids,
            atomic=atomic,
            consensus_seeded_models=seeded_models,
        ))

    index = 0
    while index < len(primary_records):
        record = primary_records[index]
        if record.atomic_title:
            if pending:
                emit(pending)
                pending = []
            # Lock the complete title block as one row, even when punctuation is
            # embedded in the title.  Normally this is one column; grouping the
            # same block defensively prevents a legacy multi-fragment title split.
            title_group = [record]
            cursor = index + 1
            while (
                cursor < len(primary_records)
                and primary_records[cursor].atomic_title
                and primary_records[cursor].block_index == record.block_index
            ):
                title_group.append(primary_records[cursor])
                cursor += 1
            emit(title_group, atomic=True)
            index = cursor
            continue

        pending.append(record)
        available_flags: list[bool] = []
        for extraction, mapping in zip(bundles, model_maps):
            candidate = mapping.get(record.column_id)
            if candidate is None or not str(candidate.text or "").strip(" □"):
                if record.column_id in extraction.terminal_candidate_end_ids:
                    available_flags.append(True)
                continue
            # Recompute from the current candidate text instead of trusting a
            # model's earlier paragraph split.  This repairs two common cases:
            # NDL joins the next sentence after a visible terminal, while Mac
            # OCR inserts a newline even though the physical column tail has no
            # terminal.  Shared physical-column IDs remain the only boundaries.
            available_flags.append(has_sentence_terminal(candidate.text))
            if record.column_id in extraction.terminal_candidate_end_ids:
                available_flags.append(True)

        # A single visible terminal is enough.  Missing punctuation is common,
        # but inventing the same terminal independently is much rarer.  Never
        # use a model's block tail as a sentence boundary: model-provided line
        # breaks are advisory only and may be early or late.
        canonical_terminal = any(available_flags)
        if canonical_terminal:
            emit(pending)
            pending = []
        index += 1

    if pending:
        emit(pending)
    comparison.column_anchored_rows = len(comparison.rows)
    comparison.alignment_shift_repairs = repair_adjacent_alignment_shifts(comparison)
    return _finalize_comparison(comparison)


def physical_column_text_snapshot(doc: UnifiedDocument) -> tuple[dict[str, str], str]:
    """Return immutable column-ID text used by multi-model alignment.

    External round-trip packages store this compact snapshot so a later import
    can audit exact physical-column lineage without embedding two or three full
    300-page documents.
    """
    extraction = _extract_column_records(doc)
    if extraction is None:
        return {}, "unavailable"
    return (
        {record.column_id: str(record.text or "") for record in extraction.records},
        "source_column_primary_texts"
        if extraction.used_primary_texts else "source_column_texts",
    )


def _reference_text_for_shift(row: MultiOcrRow, model_index: int) -> str:
    values = [
        str(text or "")
        for index, text in enumerate(row.texts)
        if index != model_index and str(text or "").strip()
    ]
    if not values:
        return ""
    groups: dict[str, list[str]] = {}
    for value in values:
        groups.setdefault(normalise_for_alignment(value), []).append(value)
    consensus = max(groups.values(), key=len)
    return max(consensus, key=lambda value: (line_quality(value).score, len(value)))


def _best_adjacent_split(
    combined: str,
    left_reference: str,
    right_reference: str,
) -> tuple[str, str] | None:
    value = str(combined or "").strip()
    left_ref = str(left_reference or "").strip()
    right_ref = str(right_reference or "").strip()
    if not value or not left_ref or not right_ref or len(value) < 4:
        return None
    full_reference = left_ref + right_ref
    if line_similarity(value, full_reference) < .78:
        return None

    expected = round(len(value) * len(left_ref) / max(1, len(full_reference)))
    radius = max(10, round(len(value) * .24))
    start = max(1, expected - radius)
    end = min(len(value) - 1, expected + radius)
    best: tuple[float, float, float, str, str] | None = None
    for boundary in range(start, end + 1):
        left = value[:boundary].strip()
        right = value[boundary:].strip()
        if not left or not right:
            continue
        left_similarity = line_similarity(left, left_ref)
        right_similarity = line_similarity(right, right_ref)
        left_length = len(normalise_for_alignment(left))
        right_length = len(normalise_for_alignment(right))
        left_ref_length = len(normalise_for_alignment(left_ref))
        right_ref_length = len(normalise_for_alignment(right_ref))
        length_penalty = (
            abs(left_length - left_ref_length) / max(1, left_length, left_ref_length)
            + abs(right_length - right_ref_length) / max(1, right_length, right_ref_length)
        )
        score = left_similarity + right_similarity - .10 * length_penalty
        candidate = (score, left_similarity, right_similarity, left, right)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    if best is None:
        return None
    score, left_similarity, right_similarity, left, right = best
    if left_similarity < .64 or right_similarity < .64 or score < 1.34:
        return None
    return left, right


def repair_adjacent_alignment_shifts(comparison: MultiOcrComparison) -> int:
    """Repair conservative one-row shifts without changing row/column IDs.

    Legacy packages may contain ``row N = empty`` and ``row N+1 = sentence N +
    sentence N+1`` for one model because that model attached a context-reOCR
    sentence to a later terminal column.  When the other model(s) provide both
    adjacent references and a high-confidence monotonic split exists, move the
    prefix back to row N.  No rows are inserted/deleted, so external package IDs
    and manual edits remain stable.
    """
    rows = comparison.rows
    existing_repairs = int(getattr(comparison, "alignment_shift_repairs", 0) or 0)
    if len(rows) < 2:
        return 0
    model_count = max((len(row.texts) for row in rows), default=0)
    repairs = 0
    for _pass in range(2):
        changed = False
        for model_index in range(model_count):
            for row_index in range(len(rows) - 1):
                left_row = rows[row_index]
                right_row = rows[row_index + 1]
                if left_row.atomic or right_row.atomic:
                    continue
                if model_index >= len(left_row.texts) or model_index >= len(right_row.texts):
                    continue
                left_value = str(left_row.texts[model_index] or "").strip()
                right_value = str(right_row.texts[model_index] or "").strip()
                left_reference = _reference_text_for_shift(left_row, model_index)
                right_reference = _reference_text_for_shift(right_row, model_index)
                if not left_reference or not right_reference:
                    continue

                split: tuple[str, str] | None = None
                direction = ""
                if not left_value and right_value:
                    split = _best_adjacent_split(right_value, left_reference, right_reference)
                    direction = "next_to_previous"
                elif left_value and not right_value:
                    split = _best_adjacent_split(left_value, left_reference, right_reference)
                    direction = "previous_to_next"
                if split is None:
                    continue

                left_text, right_text = split
                left_row.texts[model_index] = left_text
                right_row.texts[model_index] = right_text
                note = f"模型{model_index + 1}相邻句错位自动复位:{direction}"
                left_row.alignment_repaired = True
                right_row.alignment_repaired = True
                left_row.alignment_notes = tuple(dict.fromkeys((*left_row.alignment_notes, note)))
                right_row.alignment_notes = tuple(dict.fromkeys((*right_row.alignment_notes, note)))
                repairs += 1
                changed = True
        if not changed:
            break
    comparison.alignment_shift_repairs = existing_repairs + repairs
    return repairs


def _concat_units(units: Sequence[OcrSentenceUnit], indices: Sequence[int]) -> str:
    return "".join(units[index].text for index in indices)


def _group_pair_allowed(
    left: Sequence[OcrSentenceUnit],
    right: Sequence[OcrSentenceUnit],
    left_indices: Sequence[int],
    right_indices: Sequence[int],
) -> bool:
    left_titles = [left[index].block_type in {item.value for item in _TITLE_TYPES} for index in left_indices]
    right_titles = [right[index].block_type in {item.value for item in _TITLE_TYPES} for index in right_indices]
    if any(left_titles) or any(right_titles):
        # Titles are indivisible and may only align title↔title one-to-one.
        return (
            len(left_indices) == len(right_indices) == 1
            and left_titles[0]
            and right_titles[0]
        )
    return True


def _align_changed_window_many(
    left: Sequence[OcrSentenceUnit],
    right: Sequence[OcrSentenceUnit],
    left_offset: int,
    right_offset: int,
) -> list[_AlignmentGroup]:
    n, m = len(left), len(right)
    if not n:
        return [_AlignmentGroup((), (right_offset + index,)) for index in range(m)]
    if not m:
        return [_AlignmentGroup((left_offset + index,), ()) for index in range(n)]
    if n * m > 40_000:
        # Existing anchor-aware NW implementation is a safe linear-memory-ish
        # fallback for exceptionally large fully changed windows.
        aligned = align_lines(_as_compare_lines(left), _as_compare_lines(right))
        left_uid = {unit.uid: left_offset + index for index, unit in enumerate(left)}
        right_uid = {unit.uid: right_offset + index for index, unit in enumerate(right)}
        groups: list[_AlignmentGroup] = []
        for row in aligned:
            li = ()
            ri = ()
            if row.left and row.left.block_ids:
                li = (left_uid[row.left.block_ids[0]],)
            if row.right and row.right.block_ids:
                ri = (right_uid[row.right.block_ids[0]],)
            groups.append(_AlignmentGroup(li, ri))
        return groups

    inf = float("inf")
    dp = [[inf] * (m + 1) for _ in range(n + 1)]
    move: list[list[tuple[int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    gap_cost = .66
    merge_penalty = .105
    for i in range(n + 1):
        for j in range(m + 1):
            base = dp[i][j]
            if base == inf:
                continue
            if i < n and base + gap_cost < dp[i + 1][j]:
                dp[i + 1][j] = base + gap_cost
                move[i + 1][j] = (1, 0)
            if j < m and base + gap_cost < dp[i][j + 1]:
                dp[i][j + 1] = base + gap_cost
                move[i][j + 1] = (0, 1)
            for left_span in range(1, min(_MAX_GROUP_SPAN, n - i) + 1):
                left_indices = tuple(range(i, i + left_span))
                left_text = _concat_units(left, left_indices)
                for right_span in range(1, min(_MAX_GROUP_SPAN, m - j) + 1):
                    right_indices = tuple(range(j, j + right_span))
                    if not _group_pair_allowed(left, right, left_indices, right_indices):
                        continue
                    right_text = _concat_units(right, right_indices)
                    similarity = line_similarity(left_text, right_text)
                    pair_cost = (1.0 - similarity) + merge_penalty * (left_span + right_span - 2)
                    if similarity < .16:
                        pair_cost += .42
                    ni, nj = i + left_span, j + right_span
                    candidate = base + pair_cost
                    if candidate < dp[ni][nj]:
                        dp[ni][nj] = candidate
                        move[ni][nj] = (left_span, right_span)

    groups: list[_AlignmentGroup] = []
    i, j = n, m
    while i or j:
        step = move[i][j]
        if step is None:
            # Defensive fallback; should only occur with malformed all-title windows.
            if i:
                step = (1, 0)
            else:
                step = (0, 1)
        left_span, right_span = step
        groups.append(_AlignmentGroup(
            tuple(left_offset + index for index in range(i - left_span, i)),
            tuple(right_offset + index for index in range(j - right_span, j)),
        ))
        i -= left_span
        j -= right_span
    groups.reverse()
    return groups


def _align_units_many_to_many(
    left: Sequence[OcrSentenceUnit],
    right: Sequence[OcrSentenceUnit],
) -> list[_AlignmentGroup]:
    left_keys = [normalise_for_alignment(unit.text) for unit in left]
    right_keys = [normalise_for_alignment(unit.text) for unit in right]
    matcher = SequenceMatcher(None, left_keys, right_keys, autojunk=False)
    groups: list[_AlignmentGroup] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            groups.extend(
                _AlignmentGroup((left_index,), (right_index,))
                for left_index, right_index in zip(range(i1, i2), range(j1, j2))
            )
        elif tag == "delete":
            groups.extend(_AlignmentGroup((index,), ()) for index in range(i1, i2))
        elif tag == "insert":
            groups.extend(_AlignmentGroup((), (index,)) for index in range(j1, j2))
        else:
            groups.extend(_align_changed_window_many(left[i1:i2], right[j1:j2], i1, j1))
    return groups


def _map_source_boundary_to_dest(source: str, dest: str, boundary: int) -> int:
    """Map one source character boundary through a Levenshtein alignment."""
    try:
        from rapidfuzz.distance import Levenshtein
        opcodes = Levenshtein.opcodes(source, dest)
        for opcode in opcodes:
            tag, s1, s2, d1, d2 = tuple(opcode)
            if boundary < s1:
                return d1
            if s1 <= boundary <= s2:
                if tag == "equal":
                    return d1 + min(boundary - s1, d2 - d1)
                if tag == "replace":
                    ratio = (boundary - s1) / max(1, s2 - s1)
                    return int(round(d1 + ratio * (d2 - d1)))
                if tag == "delete":
                    return d1
                if tag == "insert":
                    return d2
        return len(dest)
    except Exception:
        return round(len(dest) * boundary / max(1, len(source)))


def _fit_secondary_texts_to_primary(
    primary_texts: Sequence[str],
    secondary_texts: Sequence[str],
) -> list[str]:
    if not primary_texts:
        return []
    if not secondary_texts:
        return [""] * len(primary_texts)
    if len(primary_texts) == 1:
        return ["".join(secondary_texts)]
    if len(primary_texts) == len(secondary_texts):
        return list(secondary_texts)

    source = "".join(primary_texts)
    dest = "".join(secondary_texts)
    boundaries: list[int] = []
    consumed = 0
    for text in primary_texts[:-1]:
        consumed += len(text)
        boundaries.append(_map_source_boundary_to_dest(source, dest, consumed))
    boundaries = [max(0, min(len(dest), value)) for value in boundaries]
    for index in range(1, len(boundaries)):
        boundaries[index] = max(boundaries[index], boundaries[index - 1])
    result: list[str] = []
    start = 0
    for boundary in boundaries:
        result.append(dest[start:boundary])
        start = boundary
    result.append(dest[start:])
    return result


def _compare_by_text_alignment(
    docs: Sequence[UnifiedDocument],
    labels: Sequence[str],
) -> MultiOcrComparison:
    all_units = [sentence_units(doc) for doc in docs]
    primary_units = all_units[0]
    rows_by_primary: list[MultiOcrRow] = [
        MultiOcrRow(
            index=index,
            texts=[unit.text] + [""] * (len(docs) - 1),
            model_confidences=(float(unit.confidence or 0.0),) + (0.0,) * (len(docs) - 1),
            primary_unit_index=index,
            primary_block_index=unit.block_index,
            primary_block_indices=(unit.block_index,),
            primary_block_id=unit.block_id,
            primary_segment_index=unit.segment_index,
            block_type=unit.block_type,
            page=unit.page,
            atomic=unit.atomic,
        )
        for index, unit in enumerate(primary_units)
    ]
    insertions: list[tuple[int, int, OcrSentenceUnit]] = []

    for model_index in range(1, len(docs)):
        secondary_units = all_units[model_index]
        groups = _align_units_many_to_many(primary_units, secondary_units)
        last_primary_index = -1
        for group in groups:
            primary_indices = list(group.primary_indices)
            secondary_indices = list(group.secondary_indices)
            if primary_indices:
                primary_texts = [primary_units[index].text for index in primary_indices]
                secondary_texts = [secondary_units[index].text for index in secondary_indices]
                fitted = _fit_secondary_texts_to_primary(primary_texts, secondary_texts)
                secondary_confidence_values = [
                    float(secondary_units[index].confidence or 0.0)
                    for index in secondary_indices
                    if float(secondary_units[index].confidence or 0.0) > 0
                ]
                secondary_confidence = (
                    sum(secondary_confidence_values) / len(secondary_confidence_values)
                    if secondary_confidence_values else 0.0
                )
                for primary_index, fitted_text in zip(primary_indices, fitted):
                    rows_by_primary[primary_index].texts[model_index] = fitted_text
                    confidences = list(rows_by_primary[primary_index].model_confidences or (0.0,) * len(docs))
                    while len(confidences) < len(docs):
                        confidences.append(0.0)
                    confidences[model_index] = secondary_confidence
                    rows_by_primary[primary_index].model_confidences = tuple(confidences)
                last_primary_index = primary_indices[-1]
            elif secondary_indices:
                unit_text = "".join(secondary_units[index].text for index in secondary_indices)
                first = secondary_units[secondary_indices[0]]
                insertions.append((last_primary_index, model_index, OcrSentenceUnit(
                    uid="+".join(secondary_units[index].uid for index in secondary_indices),
                    text=unit_text,
                    block_index=first.block_index,
                    block_id=first.block_id,
                    block_type=first.block_type,
                    page=first.page,
                    segment_index=first.segment_index,
                    atomic=all(secondary_units[index].atomic for index in secondary_indices),
                    confidence=(
                        sum(float(secondary_units[index].confidence or 0.0) for index in secondary_indices)
                        / max(1, len(secondary_indices))
                    ),
                )))

    insertion_rows: list[tuple[int, MultiOcrRow]] = []
    grouped: dict[int, list[tuple[int, OcrSentenceUnit]]] = {}
    for position, model_index, unit in insertions:
        grouped.setdefault(position, []).append((model_index, unit))
    for position in sorted(grouped):
        buckets: list[MultiOcrRow] = []
        for model_index, unit in grouped[position]:
            matched = None
            for candidate in buckets:
                existing = next((text for text in candidate.texts if text), "")
                if (
                    normalise_for_alignment(existing) == normalise_for_alignment(unit.text)
                    or line_similarity(existing, unit.text) >= .82
                ):
                    matched = candidate
                    break
            if matched is None:
                next_primary = position + 1
                insert_before = (
                    primary_units[next_primary].block_index
                    if 0 <= next_primary < len(primary_units)
                    else len(docs[0].blocks)
                )
                matched = MultiOcrRow(
                    index=-1,
                    texts=[""] * len(docs),
                    model_confidences=(0.0,) * len(docs),
                    page=unit.page,
                    block_type=unit.block_type,
                    insert_before_block_index=insert_before,
                    atomic=unit.atomic,
                )
                buckets.append(matched)
            matched.texts[model_index] = unit.text
            confidences = list(matched.model_confidences or (0.0,) * len(docs))
            while len(confidences) < len(docs):
                confidences.append(0.0)
            confidences[model_index] = float(unit.confidence or 0.0)
            matched.model_confidences = tuple(confidences)
        insertion_rows.extend((position, row) for row in buckets)

    before_map: dict[int, list[MultiOcrRow]] = {}
    for position, row in insertion_rows:
        before_map.setdefault(position + 1, []).append(row)
    ordered: list[MultiOcrRow] = []
    for primary_index, row in enumerate(rows_by_primary):
        ordered.extend(before_map.get(primary_index, []))
        ordered.append(row)
    ordered.extend(before_map.get(len(rows_by_primary), []))

    comparison = MultiOcrComparison(labels=list(labels), rows=ordered, alignment_mode="text_many_to_many")
    return _finalize_comparison(comparison)


def compare_ocr_documents(
    documents: Sequence[UnifiedDocument],
    labels: Sequence[str] | None = None,
) -> MultiOcrComparison:
    docs = list(documents)
    if not 2 <= len(docs) <= 3:
        raise ValueError("OCR 对比只支持 2～3 份结果")
    model_labels = list(labels or [])
    while len(model_labels) < len(docs):
        model_labels.append(f"OCR 模型 {len(model_labels) + 1}")
    model_labels = model_labels[:len(docs)]

    column_comparison = _compare_by_shared_columns(docs, model_labels)
    if column_comparison is not None:
        return column_comparison
    return _compare_by_text_alignment(docs, model_labels)


def _document_from_editor_text(text: str) -> UnifiedDocument:
    """Build a comparison-only document from freely edited explicit lines."""
    doc = UnifiedDocument()
    doc.metadata.__dict__["manual_ocr_alignment_lines"] = True
    lines = str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for index, line in enumerate(lines):
        value = line.strip()
        if not value:
            continue
        if looks_like_chapter_title(value):
            block_type = BlockType.CHAPTER
        elif value.startswith("「"):
            block_type = BlockType.DIALOGUE
        else:
            block_type = BlockType.PARAGRAPH
        block = Block(type=block_type, text=value, page=index + 1)
        if block_type in _TITLE_TYPES:
            block.metadata.update({"chapter_title_atomic": True, "atomic_ocr_sentence": True})
        doc.blocks.append(block)
    if not doc.blocks:
        doc.blocks.append(Block(type=BlockType.PARAGRAPH, text="", page=1))
    return doc


def _collapse_remapped_title_rows(comparison: MultiOcrComparison) -> None:
    """Guarantee that one original chapter block remains exactly one row."""
    title_values = {item.value for item in _TITLE_TYPES}
    collapsed: list[MultiOcrRow] = []
    index = 0
    while index < len(comparison.rows):
        row = comparison.rows[index]
        if row.primary_block_index is None or row.block_type not in title_values:
            collapsed.append(row)
            index += 1
            continue
        group = [row]
        cursor = index + 1
        while (
            cursor < len(comparison.rows)
            and comparison.rows[cursor].primary_block_index == row.primary_block_index
            and comparison.rows[cursor].block_type in title_values
        ):
            group.append(comparison.rows[cursor])
            cursor += 1
        if len(group) > 1:
            merged = copy.deepcopy(group[0])
            merged.texts = ["".join(item.texts[model] for item in group) for model in range(len(row.texts))]
            merged.atomic = True
            collapsed.append(merged)
        else:
            row.atomic = True
            collapsed.append(row)
        index = cursor
    comparison.rows = collapsed


def _remap_primary_structure(
    comparison: MultiOcrComparison,
    template: MultiOcrComparison,
) -> None:
    """Attach freely re-aligned model-1 rows back to original structural blocks."""
    old_rows = [row for row in template.rows if row.primary_block_index is not None]
    new_rows = [row for row in comparison.rows if row.primary_block_index is not None]
    if not old_rows or not new_rows:
        return
    old_values = [normalise_for_alignment(row.texts[0]) for row in old_rows]
    new_values = [normalise_for_alignment(row.texts[0]) for row in new_rows]
    matcher = SequenceMatcher(None, old_values, new_values, autojunk=False)
    mapping: dict[int, MultiOcrRow] = {}
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        old_count = i2 - i1
        new_count = j2 - j1
        if tag == "equal":
            for offset in range(min(old_count, new_count)):
                mapping[j1 + offset] = old_rows[i1 + offset]
            continue
        if new_count <= 0:
            continue
        if old_count > 0:
            for offset in range(new_count):
                relative = min(old_count - 1, int(offset * old_count / max(1, new_count)))
                mapping[j1 + offset] = old_rows[i1 + relative]
        else:
            anchor = old_rows[i1 - 1] if i1 > 0 else old_rows[min(i1, len(old_rows) - 1)]
            for offset in range(new_count):
                mapping[j1 + offset] = anchor

    segment_counts: dict[int, int] = {}
    for index, row in enumerate(new_rows):
        source = mapping.get(index)
        if source is None:
            source = old_rows[min(
                len(old_rows) - 1,
                round(index * (len(old_rows) - 1) / max(1, len(new_rows) - 1)),
            )]
        row.primary_unit_index = source.primary_unit_index
        row.primary_block_index = source.primary_block_index
        row.primary_block_indices = source.primary_block_indices or ((source.primary_block_index,) if source.primary_block_index is not None else ())
        row.primary_block_id = source.primary_block_id
        row.block_type = source.block_type
        row.page = source.page
        row.insert_before_block_index = None
        block_index = int(source.primary_block_index or 0)
        row.primary_segment_index = segment_counts.get(block_index, 0)
        segment_counts[block_index] = row.primary_segment_index + 1
        row.atomic = source.atomic

    max_block = max((row.primary_block_index or 0) for row in old_rows) + 1
    for index, row in enumerate(comparison.rows):
        if row.primary_block_index is not None:
            continue
        next_block = next((
            later.primary_block_index for later in comparison.rows[index + 1:]
            if later.primary_block_index is not None
        ), None)
        row.insert_before_block_index = int(next_block if next_block is not None else max_block)
    _collapse_remapped_title_rows(comparison)
    comparison.alignment_mode = "manual_realign"
    _finalize_comparison(comparison)


def realign_ocr_texts(
    source_texts: Sequence[str],
    labels: Sequence[str] | None,
    template: MultiOcrComparison,
) -> MultiOcrComparison:
    """Re-align freely edited lines with monotonic many-to-many matching.

    Explicit newlines are the user's candidate boundaries.  Punctuation inside a
    line is never split again, and any row mapped to an original chapter-title
    block is collapsed back to exactly one atomic title row.
    """
    texts = list(source_texts)
    if not 2 <= len(texts) <= 3:
        raise ValueError("OCR 对比只支持 2～3 份结果")
    documents = [_document_from_editor_text(text) for text in texts]
    comparison = _compare_by_text_alignment(documents, list(labels or []))
    _remap_primary_structure(comparison, template)
    return comparison


def refresh_row_character_fusion(
    row: MultiOcrRow,
    labels: Sequence[str],
) -> MultiOcrRow:
    """Recalculate the synthetic character candidate after manual OCR edits."""
    from engine.character_level_fusion import build_character_fusion
    fusion = build_character_fusion(
        row.texts, labels, physical_column_ids=row.column_ids,
        model_confidences=row.model_confidences,
    )
    row.character_fused_text = str(fusion.text or "")
    row.character_fusion_confidence = float(fusion.confidence or 0.0)
    row.character_fusion_reason = str(fusion.reason or "")
    row.character_fusion_warnings = tuple(fusion.warnings or ())
    row.character_fusion_auto_selected = bool(fusion.auto_select and fusion.text)
    row.local_reocr_recommended = bool(fusion.local_reocr_recommended)
    row.character_fusion_evidence = dict(fusion.evidence or {})
    return row


def selected_lines(comparison: MultiOcrComparison) -> list[str]:
    return [row.output_text for row in comparison.rows]


def _derived_body_type(value: str, original: BlockType) -> BlockType:
    if original in _TITLE_TYPES:
        return original
    if value.startswith(("「", "『")) or value.endswith(("」", "』")):
        return BlockType.DIALOGUE
    return BlockType.PARAGRAPH


def _column_baseline_and_boundaries(parts: Sequence[str]) -> tuple[str, list[int], list[str]]:
    """Join source physical columns and retain the end boundary of each one."""
    normalized = [normalize_column_text(str(value or "")) for value in parts]
    output = ""
    boundaries: list[int] = []
    for part in normalized:
        if part:
            if (
                output
                and output[-1].isascii()
                and output[-1].isalnum()
                and part[0].isascii()
                and part[0].isalnum()
            ):
                output += " "
            output += part
        boundaries.append(len(output))
    return output, boundaries, normalized


def _map_source_boundary_to_target(source: str, target: str, position: int) -> int:
    """Map a physical-column boundary from one OCR text onto fused text."""
    source_position = max(0, min(len(source), int(position)))
    if not source:
        return 0
    if source == target:
        return min(len(target), source_position)
    matcher = SequenceMatcher(None, source, target, autojunk=False)
    for tag, source_start, source_end, target_start, target_end in matcher.get_opcodes():
        if source_position < source_start:
            return target_start
        if source_start <= source_position <= source_end:
            if tag == "equal":
                return target_start + min(target_end - target_start, source_position - source_start)
            source_width = max(1, source_end - source_start)
            ratio = (source_position - source_start) / source_width
            return int(round(target_start + (target_end - target_start) * ratio))
    return len(target)


def project_fused_text_to_physical_columns(
    text: str,
    source_column_texts: Sequence[str],
    *,
    column_count: int | None = None,
) -> list[str]:
    """Project one final OCR sentence onto its immutable physical columns.

    The image/text proofreader must use the same physical column IDs and regions
    as OCR, while the displayed vertical text must equal the selected/fused OCR
    result rather than stale model-1 text.  Source column texts provide only the
    structural boundaries; all returned fragments concatenate exactly to
    ``text`` and remain in Japanese right-to-left reading order.
    """
    target = str(text or "").replace("\r", "").replace("\n", "")
    requested = int(column_count or 0)
    count = max(1, requested, len(source_column_texts or ()))
    parts = [str(value or "") for value in (source_column_texts or ())]
    if len(parts) < count:
        parts.extend([""] * (count - len(parts)))
    elif len(parts) > count:
        # Cardinality must follow immutable regions/IDs.  Fold any unexpected
        # tail metadata into the final physical column instead of adding a fake
        # visual column.
        parts = parts[: count - 1] + [join_column_parts(parts[count - 1 :])]

    if count == 1:
        return [target]
    if not target:
        return [""] * count

    baseline, source_boundaries, normalized_parts = _column_baseline_and_boundaries(parts)
    weights = [max(1, len(value)) for value in normalized_parts]
    total_weight = max(1, sum(weights))

    use_alignment = bool(baseline)
    if use_alignment:
        similarity = SequenceMatcher(None, baseline, target, autojunk=False).ratio()
        # Even different OCR engines normally retain enough shared Japanese text
        # for boundary alignment.  Below this guard, weighted proportional
        # placement is less misleading than an arbitrary matcher collapse.
        use_alignment = similarity >= 0.24

    boundaries = [0]
    if use_alignment:
        boundaries.extend(
            _map_source_boundary_to_target(baseline, target, boundary)
            for boundary in source_boundaries[:-1]
        )
    else:
        running = 0
        for weight in weights[:-1]:
            running += weight
            boundaries.append(round(len(target) * running / total_weight))
    boundaries.append(len(target))

    # Keep boundaries monotonic and, when possible, give every non-empty source
    # column at least one target character.  This prevents the tail of one
    # column from appearing under a neighbouring image strip.
    nonempty_total = sum(1 for value in normalized_parts if value)
    can_keep_nonempty = len(target) >= nonempty_total
    for index in range(1, count):
        previous = boundaries[index - 1]
        minimum = previous + (1 if can_keep_nonempty and normalized_parts[index - 1] else 0)
        remaining_required = (
            sum(1 for value in normalized_parts[index:] if value)
            if can_keep_nonempty else 0
        )
        maximum = max(minimum, len(target) - remaining_required)
        boundaries[index] = max(minimum, min(maximum, int(boundaries[index])))
    boundaries[-1] = len(target)

    projected = [
        target[start:end]
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    # Defensive invariant: no character may disappear or be duplicated while
    # adapting the final result to the source image columns.
    if "".join(projected) != target:
        projected = [""] * (count - 1) + [target]
    return projected


def build_fused_document(
    primary: UnifiedDocument,
    comparison: MultiOcrComparison,
    result_lines: Sequence[str] | None = None,
    *,
    delete_flags: Sequence[bool] | None = None,
    ruby_overlay_source: UnifiedDocument | dict | None = None,
) -> UnifiedDocument:
    lines = list(result_lines if result_lines is not None else selected_lines(comparison))
    if len(lines) != len(comparison.rows):
        raise ValueError(
            f"融合结果必须保持当前自动对齐的 {len(comparison.rows)} 行；当前为 {len(lines)} 行。"
            "请先点击“重新自动对齐”，再应用融合稿。"
        )
    flags = list(delete_flags) if delete_flags is not None else [False] * len(lines)
    if len(flags) != len(comparison.rows):
        raise ValueError(
            f"有意删除标记必须保持当前自动对齐的 {len(comparison.rows)} 行；当前为 {len(flags)} 行。"
        )
    flags = [bool(value) for value in flags]

    # An unresolved fusion row deliberately has no selected candidate, so the
    # GUI state returns an empty string.  Empty must never mean “drop this OCR
    # sentence”: only an explicit delete flag has that meaning.  Earlier builds
    # left a wholly unresolved primary block untouched (and therefore without
    # review-row metadata), while a mixed selected/unresolved block silently
    # omitted the unresolved sentence.  Both behaviours made OCR 对比 and
    # 图文对照 use different row lists.  Preserve the current OCR baseline for
    # unresolved rows and still attach the complete comparison lineage.
    supplied_lines = [str(value or "").strip("\r\n") for value in lines]
    resolved_lines: list[str] = []
    unresolved_fallback_rows: set[int] = set()
    for row, supplied, delete_intentionally in zip(comparison.rows, supplied_lines, flags):
        value = supplied
        if not value and not delete_intentionally:
            value = str(row.output_text or "").strip("\r\n")
            if not value:
                value = next(
                    (str(candidate or "").strip("\r\n") for candidate in row.texts
                     if str(candidate or "").strip()),
                    "",
                )
            unresolved_fallback_rows.add(int(row.index))
        resolved_lines.append(value)
    lines = resolved_lines

    result = copy.deepcopy(primary)
    for result_block in result.blocks:
        if not isinstance(getattr(result_block, "metadata", None), dict):
            result_block.metadata = {}

    def _review_regions(block: Block) -> list[dict]:
        metadata = _block_metadata(block)
        raw = metadata.get("ocr_review_regions") or []
        if isinstance(raw, dict):
            raw = [raw]

        regions: list[dict] = []
        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox") or ()
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            try:
                page = int(item.get("page", 0) or 0)
                x, y, w, h = (float(value) for value in bbox[:4])
            except (TypeError, ValueError, OverflowError):
                continue
            if page <= 0 or not all(math.isfinite(value) for value in (x, y, w, h)):
                continue
            left = max(0.0, min(1.0, x))
            top = max(0.0, min(1.0, y))
            right = max(left, min(1.0, x + w))
            bottom = max(top, min(1.0, y + h))
            if right <= left or bottom <= top:
                continue
            region = copy.deepcopy(item)
            region["page"] = page
            region["bbox"] = [left, top, right - left, bottom - top]
            regions.append(region)

        if not regions and block.bbox is not None:
            try:
                page = int(getattr(block, "page", 0) or 0)
                x = float(block.bbox.x)
                y = float(block.bbox.y)
                w = float(block.bbox.w)
                h = float(block.bbox.h)
            except (TypeError, ValueError, OverflowError):
                page = 0
            if page > 0 and all(math.isfinite(value) for value in (x, y, w, h)):
                left = max(0.0, min(1.0, x))
                top = max(0.0, min(1.0, y))
                right = max(left, min(1.0, x + w))
                bottom = max(top, min(1.0, y + h))
                if right > left and bottom > top:
                    regions = [{
                        "page": page,
                        "bbox": [left, top, right - left, bottom - top],
                        "column_id": str(metadata.get("column_id", "") or ""),
                    }]
        return regions

    def _review_column_ids(block: Block) -> list[str]:
        metadata = _block_metadata(block)
        values = metadata.get("source_column_ids") or metadata.get("multi_ocr_column_ids") or []
        if isinstance(values, str):
            values = [values]
        elif not isinstance(values, (list, tuple, set)):
            values = []
        ids = [str(value) for value in values if str(value)]
        column_id = str(metadata.get("column_id", "") or "")
        if column_id and column_id not in ids:
            ids.append(column_id)
        return ids

    review_info_by_block: dict[int, dict] = {}
    review_region_by_column: dict[str, dict] = {}
    review_text_by_column: dict[str, str] = {}
    for source_index, source_block in enumerate(result.blocks):
        source_metadata = source_block.metadata or {}
        source_ids = _review_column_ids(source_block)
        source_regions = _review_regions(source_block)
        raw_column_texts = (
            source_metadata.get("source_column_primary_texts")
            or source_metadata.get("source_column_texts")
            or []
        )
        if isinstance(raw_column_texts, str):
            raw_column_texts = [raw_column_texts]
        elif not isinstance(raw_column_texts, (list, tuple)):
            raw_column_texts = []
        source_column_texts = [str(value or "") for value in raw_column_texts]
        review_info_by_block[source_index] = {
            "column_ids": source_ids,
            "regions": source_regions,
            "column_texts": source_column_texts,
            "preferred": str(
                source_metadata.get("ocr_review_preferred_image_path")
                or source_metadata.get("ocr_review_sentence_image_path")
                or ""
            ),
            "layout": str(source_metadata.get("ocr_review_layout") or ""),
        }
        for region in source_regions:
            column_id = str(region.get("column_id", "") or "")
            if column_id:
                review_region_by_column.setdefault(column_id, copy.deepcopy(region))
        if len(source_ids) == len(source_regions):
            for column_id, region in zip(source_ids, source_regions):
                mapped = copy.deepcopy(region)
                if not str(mapped.get("column_id", "") or ""):
                    mapped["column_id"] = column_id
                review_region_by_column.setdefault(column_id, mapped)
        if len(source_ids) == len(source_column_texts):
            for column_id, column_text in zip(source_ids, source_column_texts):
                review_text_by_column.setdefault(column_id, str(column_text or ""))

    def _allocate_review_rows_to_columns(
        rows_for_target: Sequence[MultiOcrRow],
        values_for_rows: Sequence[str],
    ) -> dict[int, dict]:
        """Map several aligned sentence rows back to disjoint source columns.

        Text many-to-many alignment can split one primary OCR block into two or
        more comparison rows while the rows themselves carry no physical column
        IDs.  Copying the complete source block image to every row makes the
        image/text proofreader pair a half-sentence with the same full-page crop.

        Source columns are immutable and ordered, so solve a tiny monotonic
        partition problem: each row receives one contiguous, non-empty group of
        source columns and the concatenated column text is compared with that
        row's fused text.  Only a sufficiently trustworthy partition is used;
        otherwise the caller keeps the image as explicitly approximate context
        rather than presenting a guessed crop as exact lineage.
        """
        row_list = list(rows_for_target)
        if len(row_list) < 2:
            return {}

        positions_by_source: dict[int, list[int]] = {}
        for position, row in enumerate(row_list):
            if any(str(column_id) for column_id in row.column_ids):
                continue
            indices = list(dict.fromkeys(
                int(index) for index in (
                    row.primary_block_indices
                    or ((row.primary_block_index,) if row.primary_block_index is not None else ())
                )
            ))
            if len(indices) == 1:
                positions_by_source.setdefault(indices[0], []).append(position)

        overrides: dict[int, dict] = {}
        for source_index, positions in positions_by_source.items():
            if len(positions) < 2:
                continue
            info = review_info_by_block.get(source_index, {})
            ids = [str(value) for value in (info.get("column_ids") or []) if str(value)]
            regions_for_source = [copy.deepcopy(value) for value in (info.get("regions") or [])]
            texts_for_source = [str(value or "") for value in (info.get("column_texts") or [])]
            column_count = len(ids)
            if (
                column_count < len(positions)
                or len(regions_for_source) != column_count
                or len(texts_for_source) != column_count
                or not any(normalise_for_alignment(value) for value in texts_for_source)
            ):
                continue

            row_texts = [
                str(values_for_rows[position] or "")
                if position < len(values_for_rows)
                else str(row_list[position].output_text or "")
                for position in positions
            ]
            row_norms = [normalise_for_alignment(value) for value in row_texts]
            column_norms = [normalise_for_alignment(value) for value in texts_for_source]
            if any(not value for value in row_norms):
                continue

            # A global similarity guard prevents an unrelated/stale column-text
            # record from being force-partitioned merely because a mathematical
            # maximum always exists.
            full_rows = "".join(row_norms)
            full_columns = "".join(column_norms)
            if not full_columns:
                continue
            global_similarity = SequenceMatcher(
                None, full_rows, full_columns, autojunk=False,
            ).ratio()
            if global_similarity < 0.45:
                continue

            row_count = len(positions)
            negative = float("-inf")
            dp = [[negative] * (column_count + 1) for _ in range(row_count + 1)]
            previous = [[-1] * (column_count + 1) for _ in range(row_count + 1)]
            ratios: dict[tuple[int, int, int], float] = {}
            dp[0][0] = 0.0

            for row_number in range(1, row_count + 1):
                min_end = row_number
                max_end = column_count - (row_count - row_number)
                for end in range(min_end, max_end + 1):
                    min_start = row_number - 1
                    max_start = end - 1
                    for start in range(min_start, max_start + 1):
                        if dp[row_number - 1][start] == negative:
                            continue
                        source_value = "".join(column_norms[start:end])
                        if not source_value:
                            continue
                        target_value = row_norms[row_number - 1]
                        ratio = SequenceMatcher(
                            None, target_value, source_value, autojunk=False,
                        ).ratio()
                        length_delta = abs(len(target_value) - len(source_value)) / max(
                            1, len(target_value), len(source_value),
                        )
                        # Similarity is primary; a mild length penalty resolves
                        # ambiguous punctuation-heavy partitions without making
                        # ordinary OCR substitutions fail the guard.
                        score = ratio - 0.12 * length_delta
                        candidate = dp[row_number - 1][start] + score
                        if candidate > dp[row_number][end]:
                            dp[row_number][end] = candidate
                            previous[row_number][end] = start
                            ratios[(row_number, start, end)] = ratio

            if dp[row_count][column_count] == negative:
                continue
            spans: list[tuple[int, int, float]] = []
            end = column_count
            valid = True
            for row_number in range(row_count, 0, -1):
                start = previous[row_number][end]
                if start < 0:
                    valid = False
                    break
                ratio = ratios.get((row_number, start, end), 0.0)
                spans.append((start, end, ratio))
                end = start
            if not valid or end != 0:
                continue
            spans.reverse()
            span_ratios = [item[2] for item in spans]
            if min(span_ratios, default=0.0) < 0.28 or sum(span_ratios) / len(span_ratios) < 0.48:
                continue

            for position, (start, end, _ratio) in zip(positions, spans):
                selected_ids = ids[start:end]
                selected_regions = [copy.deepcopy(value) for value in regions_for_source[start:end]]
                selected_texts = texts_for_source[start:end]
                if len(selected_ids) != len(selected_regions):
                    continue
                overrides[position] = {
                    "column_ids": selected_ids,
                    "regions": selected_regions,
                    "column_texts": selected_texts,
                    "source_index": source_index,
                    "exact": True,
                }
        return overrides

    def _apply_fused_review_lineage(
        target: Block,
        rows_for_target: Sequence[MultiOcrRow],
        source_block_indices: Sequence[int],
        row_values: Sequence[str] | None = None,
    ) -> None:
        source_indices = list(dict.fromkeys(int(value) for value in source_block_indices))
        column_ids = list(dict.fromkeys(
            str(column_id)
            for row in rows_for_target
            for column_id in row.column_ids
            if str(column_id)
        ))
        regions = [
            copy.deepcopy(review_region_by_column[column_id])
            for column_id in column_ids
            if column_id in review_region_by_column
        ]
        # Never show a partial sentence image.  If even one requested physical
        # column lacks exact lineage, fall back to the complete source block set.
        if column_ids and len(regions) != len(column_ids):
            regions = []
        if not regions:
            seen_regions: set[tuple] = set()
            for source_index in source_indices:
                for region in review_info_by_block.get(source_index, {}).get("regions", []):
                    key = (
                        int(region.get("page", 0) or 0),
                        tuple(region.get("bbox") or ()),
                        str(region.get("column_id", "") or ""),
                    )
                    if key in seen_regions:
                        continue
                    seen_regions.add(key)
                    regions.append(copy.deepcopy(region))

        preferred = ""
        original_layout = ""
        if len(source_indices) == 1:
            info = review_info_by_block.get(source_indices[0], {})
            info_ids = list(info.get("column_ids") or [])
            full_block = not column_ids or column_ids == info_ids
            if full_block:
                preferred = str(info.get("preferred") or "")
                original_layout = str(info.get("layout") or "")

        pages = list(dict.fromkeys(
            int(region.get("page", 0) or 0)
            for region in regions
            if int(region.get("page", 0) or 0) > 0
        ))
        if not pages:
            pages = list(dict.fromkeys(
                int(getattr(result.blocks[index], "page", 0) or 0)
                for index in source_indices
                if 0 <= index < len(result.blocks)
                and int(getattr(result.blocks[index], "page", 0) or 0) > 0
            ))

        # Keep the structural block geometry aligned with the pixels presented
        # in image/text review.  Split rows get their own physical column box;
        # same-page merged rows get the union.  Cross-page sentences cannot be
        # represented by one bbox, so retain the primary structural geometry.
        if regions and len(pages) == 1:
            normalized_boxes: list[tuple[float, float, float, float]] = []
            for region in regions:
                raw_bbox = region.get("bbox") or ()
                if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) < 4:
                    normalized_boxes = []
                    break
                try:
                    x, y, w, h = (float(value) for value in raw_bbox[:4])
                except (TypeError, ValueError, OverflowError):
                    normalized_boxes = []
                    break
                if w <= 0 or h <= 0:
                    normalized_boxes = []
                    break
                normalized_boxes.append((x, y, w, h))
            if normalized_boxes:
                left = min(box[0] for box in normalized_boxes)
                top = min(box[1] for box in normalized_boxes)
                right = max(box[0] + box[2] for box in normalized_boxes)
                bottom = max(box[1] + box[3] for box in normalized_boxes)
                target.page = pages[0]
                target.bbox = BoundingBox(left, top, right - left, bottom - top)

        effective_count = max(1, len(column_ids) or len(regions) or 1)
        # Free manual line re-alignment has no immutable physical-column IDs.
        # The source pixels are still useful for proofreading, but they are a
        # source-block context image rather than a guaranteed exact split.  Mark
        # this explicitly so the GUI never presents approximate lineage as an
        # exact one-sentence crop.
        lineage_missing = not regions and not preferred
        lineage_approximate = bool(
            not lineage_missing
            and comparison.alignment_mode == "manual_realign"
            and not column_ids
        )

        # Preserve the aligned OCR-row boundaries separately from the structural
        # block.  Text alignment can legitimately map two sentence rows back to
        # one primary paragraph; without this list the image/text review page
        # incorrectly presents both source sentence images as one giant item.
        values_for_rows = list(row_values or [])
        row_column_overrides = _allocate_review_rows_to_columns(
            rows_for_target,
            values_for_rows,
        )
        review_sentence_groups: list[dict] = []
        for row_position, row in enumerate(rows_for_target):
            value = (
                str(values_for_rows[row_position] or "")
                if row_position < len(values_for_rows)
                else str(row.output_text or "")
            )
            if not value.strip():
                continue
            row_override = row_column_overrides.get(row_position, {})
            row_column_ids = list(row_override.get("column_ids") or dict.fromkeys(
                str(column_id) for column_id in row.column_ids if str(column_id)
            ))
            row_regions = [copy.deepcopy(region) for region in (row_override.get("regions") or [])]
            if not row_regions:
                row_regions = [
                    copy.deepcopy(review_region_by_column[column_id])
                    for column_id in row_column_ids
                    if column_id in review_region_by_column
                ]
            if row_column_ids and len(row_regions) != len(row_column_ids):
                row_regions = []
            row_source_indices = list(dict.fromkeys(
                int(index) for index in (
                    row.primary_block_indices
                    or ((row.primary_block_index,) if row.primary_block_index is not None else ())
                )
            ))
            if not row_regions:
                seen_row_regions: set[tuple] = set()
                for source_index in row_source_indices:
                    for region in review_info_by_block.get(source_index, {}).get("regions", []):
                        key = (
                            int(region.get("page", 0) or 0),
                            tuple(region.get("bbox") or ()),
                            str(region.get("column_id", "") or ""),
                        )
                        if key in seen_row_regions:
                            continue
                        seen_row_regions.add(key)
                        row_regions.append(copy.deepcopy(region))
            row_preferred = ""
            row_layout = ""
            if len(row_source_indices) == 1:
                row_info = review_info_by_block.get(row_source_indices[0], {})
                info_ids = list(row_info.get("column_ids") or [])
                if not row_column_ids or row_column_ids == info_ids:
                    row_preferred = str(row_info.get("preferred") or "")
                    row_layout = str(row_info.get("layout") or "")
            row_pages = list(dict.fromkeys(
                int(region.get("page", 0) or 0)
                for region in row_regions
                if int(region.get("page", 0) or 0) > 0
            ))
            source_row_column_texts = [
                str(item or "") for item in (row_override.get("column_texts") or [])
            ]
            if not source_row_column_texts:
                source_row_column_texts = [
                    str(review_text_by_column.get(column_id, "") or "")
                    for column_id in row_column_ids
                ]
            row_physical_count = max(
                1,
                len(row_column_ids),
                len(row_regions),
                len(source_row_column_texts),
            )
            # The left OCR-vertical pane and the image strips now share one
            # immutable column count/order.  Text fragments are projected from
            # the final selected result, so their concatenation is identical to
            # the OCR result page instead of showing stale model-1 columns.
            row_column_texts = project_fused_text_to_physical_columns(
                value,
                source_row_column_texts,
                column_count=row_physical_count,
            )
            fusion_candidate_texts = [str(item or "") for item in row.texts]
            fusion_candidate_labels = [
                str(comparison.labels[index] if index < len(comparison.labels) else f"模型{index + 1}")
                for index in range(len(fusion_candidate_texts))
            ]
            fusion_candidate_confidences = [
                float(row.model_confidences[index])
                if index < len(row.model_confidences) else 0.0
                for index in range(len(fusion_candidate_texts))
            ]
            # Keep the pre-correction OCR alternatives visible as audit history.
            # They are appended after the current model candidates, so the
            # selected current candidate index remains stable and history can
            # never become authoritative merely by being displayed.
            historical_texts = tuple(str(value or "") for value in (row.historical_ocr_texts or ()))
            historical_labels = tuple(str(value or "") for value in (row.historical_ocr_labels or ()))
            for history_index, history_text in enumerate(historical_texts):
                fusion_candidate_texts.append(history_text)
                base_label = (
                    historical_labels[history_index]
                    if history_index < len(historical_labels) and historical_labels[history_index]
                    else f"模型{history_index + 1}"
                )
                fusion_candidate_labels.append(f"纠错前·{base_label}")
                fusion_candidate_confidences.append(0.0)
            selected_candidate_index = int(row.chosen_index)
            character_candidate = str(row.character_fused_text or "")
            if character_candidate and character_candidate not in fusion_candidate_texts:
                fusion_candidate_texts.append(character_candidate)
                fusion_candidate_labels.append("字符融合")
                fusion_candidate_confidences.append(float(row.character_fusion_confidence or 0.0))
                if value == character_candidate:
                    selected_candidate_index = len(fusion_candidate_texts) - 1
            current_nonempty_candidate_count = sum(
                1 for item in row.texts if str(item or "").strip()
            )
            correction_resolved = bool(row.source_correction_resolved)
            requires_judgement = bool(
                not correction_resolved
                and (
                    row.is_conflict
                    or row.local_reocr_recommended
                    or float(row.confidence or 0.0) < 0.90
                    or current_nonempty_candidate_count < min(2, len(row.texts))
                )
            )
            review_sentence_groups.append({
                "row_index": int(row.index),
                "text": value,
                "column_ids": row_column_ids,
                "column_texts": row_column_texts,
                "source_column_reference_texts": source_row_column_texts,
                "regions": row_regions,
                "preferred_image_path": row_preferred,
                "layout": (
                    row_layout
                    or ("single_column" if max(len(row_column_ids), len(row_regions), 1) == 1 else "column_sentence")
                ),
                "pages": row_pages,
                "fusion_candidate_texts": fusion_candidate_texts,
                "fusion_candidate_labels": fusion_candidate_labels,
                "fusion_candidate_confidences": fusion_candidate_confidences,
                "fusion_selected_candidate_index": selected_candidate_index,
                # Store the same raw OCR-conflict predicate used by OCR 对比.
                # 图文对照 must not recompute a different disagreement list from
                # whatever candidates happen to be visible after fusion.
                "fusion_has_ocr_disagreement": bool(
                    row.is_conflict or row.historical_ocr_disagreement
                ),
                "fusion_has_historical_correction": correction_resolved,
                "fusion_reviewed": correction_resolved,
                "fusion_has_provisional_consensus": bool(row.provisional_consensus),
                "fusion_review_classification": (
                    "source_correction_resolved"
                    if correction_resolved else str(row.review_classification)
                ),
                "fusion_unresolved_baseline_preserved": int(row.index) in unresolved_fallback_rows,
                "fusion_requires_judgement": requires_judgement,
                "fusion_judgement_reason": str(
                    row.historical_resolution_reason if correction_resolved
                    else row.reason or ""
                ),
                "fusion_judgement_warnings": list(dict.fromkeys(
                    [str(item) for item in (*row.warnings, *row.character_fusion_warnings) if str(item)]
                )),
            })
        source_block_ids = [
            str(result.blocks[source_index].id)
            for source_index in source_indices
            if 0 <= source_index < len(result.blocks) and str(result.blocks[source_index].id)
        ]
        target.metadata = {
            **(target.metadata or {}),
            "multi_ocr_source_block_ids": source_block_ids,
            "ocr_review_regions": regions,
            "ocr_review_preferred_image_path": preferred,
            "ocr_review_sentence_image_path": preferred,
            "ocr_review_layout": (
                "no_primary_source"
                if lineage_missing
                else "source_block_context"
                if lineage_approximate
                else original_layout
                if preferred and original_layout
                else "single_column" if effective_count == 1
                else "column_sentence"
            ),
            "ocr_review_lineage_approximate": lineage_approximate,
            "ocr_review_lineage_missing": lineage_missing,
            "ocr_review_column_count": effective_count,
            "column_count": effective_count,
            "source_pages": pages,
            "atomic_ocr_sentence": True,
            "ocr_review_sentence_groups": review_sentence_groups,
        }
        if column_ids:
            target.metadata["source_column_ids"] = column_ids
            target.metadata["multi_ocr_column_ids"] = column_ids
            # A fused candidate is sentence-level text.  Retain one immutable
            # record per physical column by placing the complete sentence on the
            # final column ID, matching the context-reOCR convention.
            target.metadata["source_column_texts"] = (
                [""] * max(0, len(column_ids) - 1) + [str(target.text or "")]
            )
            target.metadata["source_column_terminal_flags"] = (
                [False] * max(0, len(column_ids) - 1)
                + [has_sentence_terminal(str(target.text or ""))]
            )
            target.metadata["last_column_text"] = str(target.text or "")
            target.metadata["sentence_terminal"] = has_sentence_terminal(str(target.text or ""))

    block_rows: dict[int, list[tuple[MultiOcrRow, str, bool]]] = {}
    covered_primary_blocks: set[int] = set()
    insertions: dict[int, list[tuple[MultiOcrRow, str, bool]]] = {}
    decisions: list[dict] = []
    for row, text, delete_intentionally in zip(comparison.rows, lines, flags):
        value = str(text or "").strip("\r\n")
        block_indices = tuple(
            dict.fromkeys(
                int(index) for index in (
                    row.primary_block_indices
                    or ((row.primary_block_index,) if row.primary_block_index is not None else ())
                )
            )
        )
        if block_indices:
            anchor_index = block_indices[0]
            row.primary_block_index = anchor_index
            row.primary_block_indices = block_indices
            block_rows.setdefault(anchor_index, []).append((row, value, delete_intentionally))
            covered_primary_blocks.update(block_indices)
        elif value.strip():
            target = int(row.insert_before_block_index if row.insert_before_block_index is not None else len(result.blocks))
            insertions.setdefault(target, []).append((row, value, delete_intentionally))
        decisions.append({
            "row": row.index,
            "choice": row.chosen_index,
            "confidence": round(row.confidence, 4),
            "reason": row.reason,
            "texts": list(row.texts),
            "model_confidences": list(row.model_confidences),
            "output": value,
            "delete_intentionally": delete_intentionally,
            "unresolved_baseline_preserved": int(row.index) in unresolved_fallback_rows,
            "column_ids": list(row.column_ids),
            "primary_block_indices": list(row.primary_block_indices),
            "atomic": row.atomic,
            "character_fused_text": row.character_fused_text,
            "character_fusion_confidence": round(row.character_fusion_confidence, 4),
            "character_fusion_auto_selected": row.character_fusion_auto_selected,
            "character_fusion_reason": row.character_fusion_reason,
            "character_fusion_warnings": list(row.character_fusion_warnings),
            "local_reocr_recommended": row.local_reocr_recommended,
            "character_fusion_evidence": copy.deepcopy(row.character_fusion_evidence),
        })

    rebuilt: list[Block] = []
    changed = 0
    split_structural_rows = comparison.alignment_mode in {"column_id_consensus", "manual_realign"}

    def append_insertion(row: MultiOcrRow, value: str, page: int) -> None:
        nonlocal changed
        inserted = Block(
            type=_derived_body_type(value, BlockType.PARAGRAPH),
            text=value,
            ocr_raw=value,
            page=page,
            source_format="multi_ocr_fusion",
        )
        inserted.modified_by = "multi_ocr_manual_fusion"
        inserted.metadata.update({
            "multi_ocr_insertion": True,
            "multi_ocr_column_ids": list(row.column_ids),
        })
        _apply_fused_review_lineage(
            inserted,
            [row],
            row.primary_block_indices,
            [value],
        )
        rebuilt.append(inserted)
        changed += 1

    for block_index, block in enumerate(result.blocks):
        for row, value, _delete_intentionally in insertions.get(block_index, []):
            append_insertion(row, value, int(row.page or getattr(block, "page", 0) or 0))

        rows = block_rows.get(block_index, [])
        if not rows or block.type not in _TEXT_TYPES:
            # A canonical physical-column sentence may span more than one of the
            # primary model's mistaken OCR paragraphs.  Those later paragraphs
            # are consumed by the row anchored at the first block and must not
            # survive as duplicated text.  Non-text assets are never consumed.
            if block.type in _TEXT_TYPES and block_index in covered_primary_blocks:
                continue
            rebuilt.append(copy.deepcopy(block))
            continue

        nonempty_rows = [
            (row, value) for row, value, _delete_intentionally in rows if value.strip()
        ]
        deleted_rows = [
            row for row, value, delete_intentionally in rows
            if delete_intentionally and not value.strip()
        ]
        if not nonempty_rows:
            if not deleted_rows:
                rebuilt.append(copy.deepcopy(block))
                continue
            # Intentional deletion means blanking the text while preserving the
            # stable block ID, page, bbox, anchors and EPUB structure.  An empty
            # string is therefore distinct from an unresolved/unselected row.
            current = copy.deepcopy(block)
            current.ocr_raw = current.ocr_raw or block.text
            current.text = ""
            current.modified_by = (current.modified_by + ",multi_ocr_fusion").strip(",")
            current.metadata = {
                **(current.metadata or {}),
                "multi_ocr_delete_intentionally": True,
                "multi_ocr_deleted_row_ids": [f"row:{row.index}" for row in deleted_rows],
            }
            current.metadata.setdefault("multi_ocr_audit", []).append({
                "before": block.text,
                "after": "",
                "delete_intentionally": True,
            })
            # Do not recompute review geometry for an empty replacement.  The
            # copied block already carries the exact immutable bbox/column
            # lineage; normalising it again can introduce floating-point drift.
            rebuilt.append(current)
            if str(block.text or ""):
                changed += 1
            continue

        if split_structural_rows and len(nonempty_rows) > 1 and block.type not in _TITLE_TYPES:
            for part_index, (row, value) in enumerate(nonempty_rows):
                current = copy.deepcopy(block)
                if part_index:
                    current.id = uuid.uuid4().hex
                current.ocr_raw = current.ocr_raw or block.text
                current.text = value
                current.type = _derived_body_type(value, block.type)
                current.modified_by = (current.modified_by + ",multi_ocr_fusion").strip(",")
                current.metadata = {
                    **(current.metadata or {}),
                    "multi_ocr_split_from_block": block.id,
                    "multi_ocr_split_part": part_index + 1,
                    "multi_ocr_split_count": len(nonempty_rows),
                    "multi_ocr_column_ids": list(row.column_ids),
                }
                current.metadata.setdefault("multi_ocr_audit", []).append({
                    "before": block.text,
                    "after": value,
                })
                if deleted_rows:
                    current.metadata["multi_ocr_deleted_row_ids"] = [
                        f"row:{deleted.index}" for deleted in deleted_rows
                    ]
                _apply_fused_review_lineage(
                    current,
                    [row],
                    row.primary_block_indices or (block_index,),
                    [value],
                )
                rebuilt.append(current)
            changed += len(nonempty_rows)
            continue

        current = copy.deepcopy(block)
        fused_text = "".join(value for _row, value in nonempty_rows).strip()
        if fused_text and (fused_text != block.text or deleted_rows):
            current.ocr_raw = current.ocr_raw or block.text
            current.text = fused_text
            current.modified_by = (current.modified_by + ",multi_ocr_fusion").strip(",")
            current.metadata.setdefault("multi_ocr_audit", []).append({
                "before": block.text,
                "after": fused_text,
                "delete_intentionally": bool(deleted_rows),
            })
            current.metadata["multi_ocr_column_ids"] = [
                column_id for row, _value in nonempty_rows for column_id in row.column_ids
            ]
            if deleted_rows:
                current.metadata["multi_ocr_deleted_row_ids"] = [
                    f"row:{deleted.index}" for deleted in deleted_rows
                ]
            changed += 1
        lineage_indices = tuple(dict.fromkeys(
            index
            for row, _value in nonempty_rows
            for index in (row.primary_block_indices or ((row.primary_block_index,) if row.primary_block_index is not None else ()))
        )) or (block_index,)
        _apply_fused_review_lineage(
            current,
            [row for row, _value in nonempty_rows],
            lineage_indices,
            [value for _row, value in nonempty_rows],
        )
        rebuilt.append(current)

    for row, value, _delete_intentionally in insertions.get(len(result.blocks), []):
        append_insertion(row, value, int(row.page or 0))

    result.blocks = rebuilt
    for order, block in enumerate(result.blocks):
        block.reading_order = order

    # Splits, merges and insertions change block indices.  TOC entries point to
    # indices rather than IDs, so remap them through the stable chapter block ID
    # retained by the fusion code.  Without this, a body split before chapter 2
    # silently sends EPUB/navigation links to the wrong paragraph.
    rebuilt_index_by_id: dict[str, int] = {}
    for index, rebuilt_block in enumerate(result.blocks):
        rebuilt_index_by_id.setdefault(str(rebuilt_block.id), index)
    original_block_ids = [str(block.id) for block in primary.blocks]
    for toc_entry in result.toc:
        try:
            old_index = int(getattr(toc_entry, "block_index", -1))
        except (TypeError, ValueError, OverflowError):
            continue
        if 0 <= old_index < len(original_block_ids):
            new_index = rebuilt_index_by_id.get(original_block_ids[old_index])
            if new_index is not None:
                toc_entry.block_index = new_index
                target_block = result.blocks[new_index]
                if target_block.type in _TITLE_TYPES and str(target_block.text or "").strip():
                    toc_entry.title = str(target_block.text).strip()

    local_reocr_queue = [
        {
            "row": row.index,
            "page": int(row.page or 0),
            "column_ids": list(row.column_ids),
            "reason": row.character_fusion_reason or row.reason,
            "warnings": list(dict.fromkeys((*row.warnings, *row.character_fusion_warnings))),
            "model_texts": list(row.texts),
            "character_fused_text": row.character_fused_text,
        }
        for row in comparison.rows
        if row.local_reocr_recommended
    ]
    result.metadata.source_engine = "multi_ocr_fusion"
    result.metadata.__dict__.update({
        "multi_ocr_fusion": True,
        "multi_ocr_labels": list(comparison.labels),
        "multi_ocr_summary": comparison.summary,
        "multi_ocr_alignment_mode": comparison.alignment_mode,
        "multi_ocr_low_confidence": comparison.low_confidence_rows,
        "multi_ocr_character_fused_rows": comparison.character_fused_rows,
        "multi_ocr_character_auto_selected_rows": comparison.character_auto_selected_rows,
        "multi_ocr_local_reocr_queue": local_reocr_queue,
        "multi_ocr_decisions": decisions,
    })
    result.add_log("multi_ocr_fusion", comparison.summary, changed)
    # Ruby is a locked structural side-channel, never an OCR voting candidate.
    # Re-attach it only after the authoritative fused/AI-edited prose is final.
    try:
        from adapters.findtext_centernet_ruby import apply_ruby_overlay, strip_ruby_overlay
        if ruby_overlay_source is not None:
            apply_ruby_overlay(result, ruby_overlay_source)
        else:
            # Rebuilding a fusion without an explicit overlay source is a hard
            # Ruby-OFF boundary.  Never revive stale metadata inherited from the
            # primary OCR document or an older snapshot.
            strip_ruby_overlay(result, strip_candidate_geometry=False, strip_logs=False)
    except Exception:
        # Ruby preservation is optional and must never break the main OCR path.
        pass
    return result


def choose_best_text(texts: Sequence[str]) -> tuple[int, float, str, tuple[str, ...]]:
    """Public wrapper used by the manual comparison UI after source edits."""
    return _auto_choose(texts)

# ── OCR compare visual helpers ────────────────────────────────────────────────

def exact_candidate_groups(texts: Sequence[str]) -> list[tuple[str, tuple[int, ...]]]:
    """Group OCR candidates by the shared conservative Japanese compare key.

    Safe Unicode/full-width punctuation/layout-space differences are collapsed
    into one candidate card and the visible candidate uses ``canonical_japanese``.
    Kanji, kana, digits, dashes and lexical forms remain distinct.
    """
    grouped: dict[str, list[int]] = {}
    representative: dict[str, str] = {}
    order: list[str] = []
    for index, text in enumerate(texts):
        raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not raw:
            continue
        key = standard_japanese_key(raw)
        if not key:
            continue
        if key not in grouped:
            grouped[key] = []
            representative[key] = canonical_japanese(raw)
            order.append(key)
        grouped[key].append(index)
    return [(representative[key], tuple(grouped[key])) for key in order]


def exact_consensus_candidate(texts: Sequence[str]) -> tuple[str, tuple[int, ...]] | None:
    """Return a candidate only when every configured OCR model agrees.

    A 2-of-3 match is useful evidence but is no longer treated as authoritative
    consensus.  This prevents two identical OCR outputs from silently selecting
    the final text ahead of an explicit AI adjudication or human decision.  In a
    genuine two-model comparison both models must agree; empty/missing outputs
    keep the row unresolved.
    """
    values = list(texts)
    if len(values) < 2:
        return None
    groups = exact_candidate_groups(values)
    return next((item for item in groups if len(item[1]) == len(values)), None)


def intraline_match_masks(texts: Sequence[str]) -> list[list[bool]]:
    """Return per-character masks for text shared at the same aligned position.

    Characters are green only when they participate in an equal opcode against
    every other non-empty OCR result. Insertions, deletions and replacements are
    red. This is a deliberately conservative multi-model extension of a standard
    Myers/SequenceMatcher intraline diff.
    """
    values = [str(text or "") for text in texts]
    masks = [[False] * len(value) for value in values]
    nonempty_keys = [standard_japanese_key(value) for value in values if value]
    if nonempty_keys and len(set(nonempty_keys)) == 1:
        return [[True] * len(value) for value in values]
    nonempty = [index for index, value in enumerate(values) if value]
    if not nonempty:
        return masks
    if len(nonempty) == 1:
        only = nonempty[0]
        masks[only] = [True] * len(values[only])
        return masks

    base_index = nonempty[0]
    base = values[base_index]
    base_common = [True] * len(base)
    mappings: dict[int, dict[int, int]] = {}

    for other_index in nonempty[1:]:
        other = values[other_index]
        matcher = SequenceMatcher(None, base, other, autojunk=False)
        matched_base = [False] * len(base)
        mapping: dict[int, int] = {}
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                continue
            span = min(i2 - i1, j2 - j1)
            for offset in range(span):
                base_pos = i1 + offset
                other_pos = j1 + offset
                matched_base[base_pos] = True
                mapping[base_pos] = other_pos
        base_common = [left and right for left, right in zip(base_common, matched_base)]
        mappings[other_index] = mapping

    masks[base_index] = base_common
    for other_index, mapping in mappings.items():
        other_mask = masks[other_index]
        for base_pos, other_pos in mapping.items():
            if base_common[base_pos] and 0 <= other_pos < len(other_mask):
                other_mask[other_pos] = True
    return masks


def refresh_comparison_after_text_standardization(
    comparison: MultiOcrComparison,
) -> MultiOcrComparison:
    """Recalculate comparison/fusion statistics after representation-only edits.

    The caller owns ``comparison`` and should pass a copy when the original OCR
    result must remain immutable.  Alignment metadata and physical column IDs are
    retained; only text-derived confidence, conflicts and character fusion are
    refreshed.
    """
    return _finalize_comparison(comparison)
