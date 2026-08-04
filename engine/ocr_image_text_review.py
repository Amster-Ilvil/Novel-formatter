#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sentence-level OCR image/text proofreading helpers.

This module is intentionally independent from Qt.  It converts the immutable
column/sentence lineage already stored on OCR blocks into one review entry per
sentence and can render the matching source pixels lazily.  OCR inputs and
recognition results are never regenerated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import math
import copy
import hashlib
import difflib
import threading

from PIL import Image

from models.document import BlockType, UnifiedDocument

_TEXT_TYPES = {
    BlockType.PARAGRAPH,
    BlockType.DIALOGUE,
    BlockType.CHAPTER,
    BlockType.SECTION,
    BlockType.RUBY,
    BlockType.FOOTNOTE,
    BlockType.TOC_ENTRY,
}


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _existing_file(value) -> str:
    try:
        path = Path(str(value or ""))
        return str(path) if path.is_file() else ""
    except (OSError, ValueError, TypeError):
        return ""


def _safe_bbox(value) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        result = [float(item) for item in value[:4]]
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(item) for item in result):
        return None
    return result


@dataclass
class OCRReviewEntry:
    block_id: str
    block_index: int
    text: str
    page: int
    pages: tuple[int, ...] = ()
    column_ids: tuple[str, ...] = ()
    column_count: int = 1
    layout: str = "single_column"
    preferred_image_path: str = ""
    regions: list[dict] = field(default_factory=list)
    column_texts: tuple[str, ...] = ()
    segment_index: int = 0
    segment_count: int = 1
    segment_key: str = ""
    reviewed: bool = False
    changed: bool = False
    fusion_candidate_texts: tuple[str, ...] = ()
    fusion_candidate_labels: tuple[str, ...] = ()
    fusion_candidate_confidences: tuple[float, ...] = ()
    selected_candidate_index: int = -1
    # Stable row index from the multi-model comparison.  This is deliberately
    # separate from block/segment indices so 图文对照 can publish a reviewed
    # final candidate back to the exact OCR comparison row without touching any
    # source-model text.
    source_row_index: int = -1
    # Explicit raw-model disagreement copied from MultiOcrRow.is_conflict.
    # ``None`` is reserved for legacy/single-OCR entries that need the old
    # candidate-text fallback.
    source_ocr_disagreement: bool | None = None
    requires_judgement: bool = False
    judgement_reason: str = ""
    judgement_warnings: tuple[str, ...] = ()

    @property
    def cache_key(self) -> str:
        payload = "|".join([
            self.block_id,
            self.segment_key,
            self.preferred_image_path,
            repr(self.regions),
            repr(self.column_texts),
        ])
        return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()


_REVIEW_STRONG_TERMINALS = set("。．！？!?｡")
_REVIEW_QUOTE_OPENERS = "「『"
_REVIEW_QUOTE_CLOSERS = "」』"
_REVIEW_EXPECTED_CLOSE = {"「": "」", "『": "』"}


def _review_sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return conservative sentence spans without changing OCR text.

    The OCR reflow layer deliberately does not split punctuation occurring in
    the middle of one physical column.  That is correct for lossless OCR, but a
    sentence-review screen must not show two complete sentences as one item.
    This scanner therefore operates only on the already recognised block text.

    Japanese quotation continuations such as ``「行く。」と彼は言った。`` stay
    together.  A terminal inside a quote becomes a boundary only after the
    outer quote closes and the following text is not a quotative continuation.
    """
    value = str(text or "")
    if not value.strip():
        return []
    try:
        from engine.column_sentence_reflow import starts_post_quote_continuation
    except Exception:  # pragma: no cover - defensive import fallback
        starts_post_quote_continuation = lambda _value: False

    spans: list[tuple[int, int]] = []
    stack: list[str] = []
    start = 0
    quoted_terminal_pending = False

    def append_until(end: int) -> None:
        nonlocal start
        end = max(start, min(len(value), int(end)))
        # Keep inter-sentence whitespace with the preceding sentence so joining
        # edited segments reconstructs the original block byte-for-byte unless
        # the user changes text.
        while end < len(value) and value[end] in " \t\r\n　":
            end += 1
        if value[start:end].strip():
            spans.append((start, end))
        start = end

    for index, char in enumerate(value):
        if char in _REVIEW_QUOTE_OPENERS:
            stack.append(char)
            continue
        if char in _REVIEW_QUOTE_CLOSERS:
            if stack:
                expected = _REVIEW_EXPECTED_CLOSE.get(stack[-1], "")
                if char == expected or len(stack) > 1:
                    stack.pop()
            if not stack and quoted_terminal_pending:
                following = value[index + 1 :]
                if not starts_post_quote_continuation(following):
                    append_until(index + 1)
                quoted_terminal_pending = False
            continue
        if char not in _REVIEW_STRONG_TERMINALS:
            continue
        if stack:
            quoted_terminal_pending = True
            continue
        append_until(index + 1)

    if start < len(value) and value[start:].strip():
        spans.append((start, len(value)))
    return spans or [(0, len(value))]


def _normalized_column_parts(values) -> list[str]:
    try:
        from engine.column_sentence_reflow import normalize_column_text
    except Exception:  # pragma: no cover
        normalize_column_text = lambda value: str(value or "").strip()
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple)):
        return []
    return [normalize_column_text(str(value or "")) for value in values]


def _join_with_ranges(parts: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Join physical-column texts and retain each column's character range."""
    output = ""
    ranges: list[tuple[int, int]] = []
    for part in parts:
        prefix = ""
        if (
            output
            and part
            and output[-1].isascii()
            and output[-1].isalnum()
            and part[0].isascii()
            and part[0].isalnum()
        ):
            prefix = " "
        start = len(output)
        output += prefix + part
        ranges.append((start, len(output)))
    return output, ranges


def _map_target_position(source: str, target: str, target_position: int) -> int:
    """Map one character boundary in ``target`` onto the source baseline."""
    position = max(0, min(len(target), int(target_position)))
    if not target:
        return 0
    if source == target:
        return min(len(source), position)
    matcher = difflib.SequenceMatcher(None, source, target, autojunk=False)
    for tag, source_start, source_end, target_start, target_end in matcher.get_opcodes():
        if position < target_start:
            return source_start
        if target_start <= position <= target_end:
            if tag == "equal":
                return source_start + min(source_end - source_start, position - target_start)
            target_width = max(1, target_end - target_start)
            ratio = (position - target_start) / target_width
            return int(round(source_start + (source_end - source_start) * ratio))
    return len(source)


def _clip_vertical_region(region: dict, start_ratio: float, end_ratio: float) -> dict:
    clipped = dict(region)
    bbox = _safe_bbox(clipped.get("bbox"))
    if bbox is None:
        return clipped
    start_ratio = max(0.0, min(1.0, float(start_ratio)))
    end_ratio = max(start_ratio, min(1.0, float(end_ratio)))
    x, y, width, height = bbox
    clipped["bbox"] = [x, y + height * start_ratio, width, max(1e-7, height * (end_ratio - start_ratio))]
    clipped["sentence_slice"] = [round(start_ratio, 7), round(end_ratio, 7)]
    return clipped


def _split_block_for_review(
    *,
    block_id: str,
    block_index: int,
    text: str,
    page: int,
    pages: tuple[int, ...],
    column_ids: tuple[str, ...],
    layout: str,
    preferred: str,
    regions: list[dict],
    column_texts: list[str],
    reviewed: bool,
    changed: bool,
    preserve_physical_columns: bool = False,
) -> list[OCRReviewEntry]:
    """Build review items without ever cutting a locked physical column.

    ``column_sentence_reflow`` has already applied the authoritative boundary
    rule: consume each printed vertical column in full, inspect punctuation only
    at that column's bottom, and continue into the next column when the tail is
    not terminal.  Re-scanning the merged text for every internal ``。`` here
    used to cut a column halfway through (for example ``...ある。伝えられ``),
    which made the proofreader display a partial image and appear to lose text.

    When immutable physical-column lineage is present, the completed reflow
    block is therefore one indivisible review item.  Non-column legacy blocks
    retain the older sentence-span mapping behaviour.
    """
    sentence_spans = (
        [(0, len(str(text or "")))]
        if preserve_physical_columns and str(text or "").strip()
        else _review_sentence_spans(text)
    )
    can_map = (
        len(sentence_spans) > 1
        and regions
        and column_texts
        and len(regions) == len(column_texts)
    )
    if not can_map:
        return [OCRReviewEntry(
            block_id=block_id,
            block_index=block_index,
            text=text,
            page=page,
            pages=pages,
            column_ids=column_ids,
            column_count=max(1, len(column_ids) or len(regions) or 1),
            layout=layout,
            preferred_image_path=preferred,
            regions=regions,
            column_texts=tuple(column_texts),
            segment_index=0,
            segment_count=1,
            segment_key=f"{block_id}:s001",
            reviewed=reviewed,
            changed=changed,
        )]

    baseline, column_ranges = _join_with_ranges(column_texts)
    if not baseline:
        return _split_block_for_review(
            block_id=block_id,
            block_index=block_index,
            text=text,
            page=page,
            pages=pages,
            column_ids=column_ids,
            layout=layout,
            preferred=preferred,
            regions=regions,
            column_texts=[],
            reviewed=reviewed,
            changed=changed,
        )

    boundaries = [0]
    for _start, end in sentence_spans[:-1]:
        boundaries.append(_map_target_position(baseline, text, end))
    boundaries.append(len(baseline))
    # Make mapped boundaries monotonic.  If OCR substitutions collapse a short
    # range, assign at least the nearest following character/column instead of
    # creating an empty sentence image.
    for idx in range(1, len(boundaries)):
        boundaries[idx] = max(boundaries[idx - 1], min(len(baseline), boundaries[idx]))

    entries: list[OCRReviewEntry] = []
    total_segments = len(sentence_spans)
    for segment_index, ((text_start, text_end), baseline_start, baseline_end) in enumerate(
        zip(sentence_spans, boundaries[:-1], boundaries[1:])
    ):
        segment_regions: list[dict] = []
        segment_column_texts: list[str] = []
        segment_column_ids: list[str] = []
        for column_index, ((column_start, column_end), region, column_text) in enumerate(
            zip(column_ranges, regions, column_texts)
        ):
            overlap_start = max(baseline_start, column_start)
            overlap_end = min(baseline_end, column_end)
            if overlap_end <= overlap_start:
                continue
            width = max(1, column_end - column_start)
            relative_start = (overlap_start - column_start) / width
            relative_end = (overlap_end - column_start) / width
            segment_regions.append(_clip_vertical_region(region, relative_start, relative_end))
            raw_start = max(0, min(len(column_text), int(round(relative_start * len(column_text)))))
            raw_end = max(raw_start, min(len(column_text), int(round(relative_end * len(column_text)))))
            fragment = column_text[raw_start:raw_end] or column_text
            segment_column_texts.append(fragment)
            if column_index < len(column_ids):
                segment_column_ids.append(column_ids[column_index])
            else:
                segment_column_ids.append(str(region.get("column_id", "") or ""))

        # Mapping can be ambiguous when the context OCR differs heavily from the
        # first-pass columns.  In that rare case keep the sentence editable but
        # do not pair it with a misleading partial image.
        mapped_pages = tuple(dict.fromkeys(
            _safe_int(region.get("page", 0))
            for region in segment_regions
            if _safe_int(region.get("page", 0)) > 0
        ))
        segment_layout = "single_column" if len(segment_regions) <= 1 else "column_sentence"
        entries.append(OCRReviewEntry(
            block_id=block_id,
            block_index=block_index,
            text=text[text_start:text_end],
            page=(mapped_pages[0] if mapped_pages else page),
            pages=mapped_pages or pages,
            column_ids=tuple(value for value in segment_column_ids if value),
            column_count=max(1, len(segment_regions) or len(segment_column_texts) or 1),
            layout=segment_layout,
            # The exact sentence-context image may contain several recovered
            # sentences.  Split entries must therefore render only their own
            # immutable source regions instead of reusing the whole group image.
            preferred_image_path="",
            regions=segment_regions,
            column_texts=tuple(segment_column_texts),
            segment_index=segment_index,
            segment_count=total_segments,
            segment_key=f"{block_id}:s{segment_index + 1:03d}",
            reviewed=reviewed,
            changed=changed,
        ))
    return entries


def _resolved_review_regions(raw_regions, page_path_by_no: dict[int, str]) -> list[dict]:
    regions: list[dict] = []
    if isinstance(raw_regions, dict):
        raw_regions = [raw_regions]
    for raw in raw_regions if isinstance(raw_regions, list) else []:
        if not isinstance(raw, dict):
            continue
        region = dict(raw)
        page = _safe_int(region.get("page", 0) or 0)
        embedded_path = _existing_file(region.get("page_path"))
        current_page_path = _existing_file(page_path_by_no.get(page, ""))
        region["page_path"] = embedded_path or current_page_path or str(region.get("page_path") or "")
        bbox = _safe_bbox(region.get("bbox"))
        if page > 0 and bbox is not None:
            region["page"] = page
            region["bbox"] = bbox
            regions.append(region)
    return regions


def _current_group_texts(block_text: str, raw_groups: list[dict]) -> list[str]:
    """Keep stored row boundaries while reflecting a later whole-block edit."""
    original_parts = [str(group.get("text", "") or "") for group in raw_groups]
    original = "".join(original_parts)
    current = str(block_text or "")
    if not original_parts:
        return []
    if original == current:
        return original_parts
    boundaries = [0]
    cursor = 0
    for part in original_parts[:-1]:
        cursor += len(part)
        # Map an original structural boundary into the current edited text.
        boundaries.append(_map_target_position(current, original, cursor))
    boundaries.append(len(current))
    for index in range(1, len(boundaries)):
        boundaries[index] = max(boundaries[index - 1], min(len(current), boundaries[index]))
    return [current[start:end] for start, end in zip(boundaries[:-1], boundaries[1:])]


def _bbox_region(block, page_path_by_no: dict[int, str]) -> dict | None:
    bbox = getattr(block, "bbox", None)
    page = _safe_int(getattr(block, "page", 0) or 0)
    metadata = block.metadata if isinstance(getattr(block, "metadata", None), dict) else {}
    if bbox is None or page <= 0:
        return None
    normalized = _safe_bbox([
        getattr(bbox, "x", None), getattr(bbox, "y", None),
        getattr(bbox, "w", None), getattr(bbox, "h", None),
    ])
    if normalized is None or normalized[2] <= 0 or normalized[3] <= 0:
        return None
    return {
        "page": page,
        "page_path": page_path_by_no.get(page, ""),
        "bbox": normalized,
        "column_id": str(metadata.get("column_id", "") or ""),
    }


def build_review_entries(doc: UnifiedDocument | None) -> list[OCRReviewEntry]:
    if doc is None:
        return []
    page_path_by_no: dict[int, str] = {}
    for page in getattr(doc, "pages", []):
        page_no = _safe_int(getattr(page, "page_no", 0) or 0)
        if page_no > 0:
            page_path_by_no[page_no] = str(getattr(page, "image_path", "") or "")
    entries: list[OCRReviewEntry] = []
    for index, block in enumerate(getattr(doc, "blocks", [])):
        if block.type not in _TEXT_TYPES:
            continue
        raw_meta = getattr(block, "metadata", None)
        meta = dict(raw_meta) if isinstance(raw_meta, dict) else {}
        reviewed_segment_map = (
            dict(meta.get("ocr_image_text_review_checked_segments") or {})
            if isinstance(meta.get("ocr_image_text_review_checked_segments") or {}, dict)
            else {}
        )
        changed_segment_map = (
            dict(meta.get("ocr_image_text_review_changed_segments") or {})
            if isinstance(meta.get("ocr_image_text_review_changed_segments") or {}, dict)
            else {}
        )
        if meta.get("layout_placeholder") or meta.get("consumed_by_column_sentence_reflow"):
            continue
        text = str(block.text or "")
        if not text.strip():
            continue

        raw_sentence_groups = meta.get("ocr_review_sentence_groups") or []
        if isinstance(raw_sentence_groups, dict):
            raw_sentence_groups = [raw_sentence_groups]
        valid_sentence_groups = [
            dict(group) for group in raw_sentence_groups
            if isinstance(group, dict) and str(group.get("text", "") or "").strip()
        ] if isinstance(raw_sentence_groups, list) else []
        if valid_sentence_groups:
            group_texts = _current_group_texts(text, valid_sentence_groups)
            total_groups = len(valid_sentence_groups)
            for group_index, (group, group_text) in enumerate(zip(valid_sentence_groups, group_texts)):
                group_regions = _resolved_review_regions(group.get("regions") or [], page_path_by_no)
                group_ids = group.get("column_ids") or []
                if isinstance(group_ids, str):
                    group_ids = [group_ids]
                elif not isinstance(group_ids, (list, tuple, set)):
                    group_ids = []
                group_ids = tuple(str(value) for value in group_ids if str(value))

                # Column IDs are the immutable reading-order key.  Reorder
                # source regions by those IDs whenever every region can be
                # resolved, so the first OCR text fragment and first image crop
                # always describe the same rightmost Japanese column.
                if group_ids and group_regions:
                    region_by_id = {
                        str(region.get("column_id", "") or ""): region
                        for region in group_regions
                        if str(region.get("column_id", "") or "")
                    }
                    if all(column_id in region_by_id for column_id in group_ids):
                        group_regions = [region_by_id[column_id] for column_id in group_ids]

                group_column_texts = _normalized_column_parts(group.get("column_texts") or [])
                group_reference_texts = _normalized_column_parts(
                    group.get("source_column_reference_texts") or group_column_texts
                )
                physical_count = (
                    len(group_ids)
                    or len(group_regions)
                    or len(group_column_texts)
                    or 1
                )
                # Older fused projects may contain model-1 per-column text next
                # to a later chosen/fused sentence.  Re-project the current row
                # text onto the exact physical columns so the vertical OCR pane
                # equals the OCR result and keeps one fragment per image strip.
                if (
                    len(group_column_texts) != physical_count
                    or "".join(group_column_texts) != str(group_text or "").replace("\r", "").replace("\n", "")
                ):
                    from engine.multi_ocr_compare import project_fused_text_to_physical_columns
                    group_column_texts = project_fused_text_to_physical_columns(
                        group_text,
                        group_reference_texts,
                        column_count=physical_count,
                    )
                group_pages = group.get("pages") or [region.get("page") for region in group_regions]
                if not isinstance(group_pages, (list, tuple, set)):
                    group_pages = [group_pages]
                group_page_tuple = tuple(dict.fromkeys(
                    page_no for page_no in (_safe_int(value) for value in group_pages) if page_no > 0
                ))
                count = max(1, physical_count)
                group_layout = str(group.get("layout") or ("single_column" if count == 1 else "column_sentence"))
                segment_key = f"{block.id}:r{_safe_int(group.get('row_index', group_index), group_index):06d}"
                raw_candidate_texts = group.get("fusion_candidate_texts") or []
                raw_candidate_labels = group.get("fusion_candidate_labels") or []
                raw_candidate_confidences = group.get("fusion_candidate_confidences") or []
                if not isinstance(raw_candidate_texts, (list, tuple)):
                    raw_candidate_texts = []
                if not isinstance(raw_candidate_labels, (list, tuple)):
                    raw_candidate_labels = []
                if not isinstance(raw_candidate_confidences, (list, tuple)):
                    raw_candidate_confidences = []
                candidate_texts = tuple(str(value or "") for value in raw_candidate_texts)
                candidate_labels = tuple(str(value or "") for value in raw_candidate_labels)
                candidate_confidences = tuple(
                    float(value or 0.0) if isinstance(value, (int, float)) else 0.0
                    for value in raw_candidate_confidences
                )
                selected_candidate_index = _safe_int(
                    group.get("review_selected_candidate_index", group.get("fusion_selected_candidate_index", -1)),
                    -1,
                )
                judgement_warnings = group.get("fusion_judgement_warnings") or []
                if not isinstance(judgement_warnings, (list, tuple, set)):
                    judgement_warnings = [judgement_warnings]
                entries.append(OCRReviewEntry(
                    block_id=str(block.id),
                    block_index=index,
                    text=group_text,
                    page=(group_page_tuple[0] if group_page_tuple else _safe_int(getattr(block, "page", 0) or 0)),
                    pages=group_page_tuple,
                    column_ids=group_ids,
                    column_count=count,
                    layout=group_layout,
                    preferred_image_path=str(group.get("preferred_image_path") or ""),
                    regions=group_regions,
                    column_texts=tuple(group_column_texts),
                    segment_index=group_index,
                    segment_count=total_groups,
                    segment_key=segment_key,
                    reviewed=(
                        bool(group.get("fusion_reviewed", False))
                        or (
                            bool(reviewed_segment_map.get(segment_key))
                            if reviewed_segment_map else bool(meta.get("ocr_image_text_review_checked"))
                        )
                    ),
                    changed=(
                        bool(changed_segment_map.get(segment_key))
                        if changed_segment_map else bool(meta.get("ocr_image_text_review_changed"))
                    ),
                    fusion_candidate_texts=candidate_texts,
                    fusion_candidate_labels=candidate_labels,
                    fusion_candidate_confidences=candidate_confidences,
                    selected_candidate_index=selected_candidate_index,
                    source_row_index=_safe_int(group.get("row_index", -1), -1),
                    source_ocr_disagreement=(
                        bool(group.get("fusion_has_ocr_disagreement"))
                        if "fusion_has_ocr_disagreement" in group else None
                    ),
                    requires_judgement=bool(group.get("fusion_requires_judgement", False)),
                    judgement_reason=str(group.get("fusion_judgement_reason", "") or ""),
                    judgement_warnings=tuple(str(value) for value in judgement_warnings if str(value)),
                ))
            continue

        raw_regions = meta.get("ocr_review_regions") or []
        regions = _resolved_review_regions(raw_regions, page_path_by_no)
        if not regions:
            fallback = _bbox_region(block, page_path_by_no)
            if fallback is not None:
                regions.append(fallback)

        column_ids = meta.get("source_column_ids") or meta.get("multi_ocr_column_ids") or []
        if isinstance(column_ids, str):
            column_ids = [column_ids]
        elif not isinstance(column_ids, (list, tuple, set)):
            column_ids = []
        column_ids = tuple(str(value) for value in column_ids if str(value))
        if not column_ids:
            column_id = str(meta.get("column_id", "") or "")
            column_ids = (column_id,) if column_id else ()

        pages = meta.get("source_pages") or [region.get("page") for region in regions]
        if not isinstance(pages, (list, tuple, set)):
            pages = [pages]
        page_tuple = tuple(dict.fromkeys(
            page_no for page_no in (_safe_int(value) for value in pages) if page_no > 0
        ))
        preferred = str(
            meta.get("ocr_review_preferred_image_path")
            or meta.get("ocr_review_sentence_image_path")
            or ""
        )
        raw_column_texts = (
            meta.get("source_column_primary_texts")
            or meta.get("source_column_texts")
            or []
        )
        column_texts = _normalized_column_parts(raw_column_texts)
        count_hint = meta.get("column_count", 0) or len(column_ids) or len(regions) or len(column_texts) or 1
        count = max(1, _safe_int(count_hint, len(column_ids) or len(regions) or len(column_texts) or 1))
        layout = str(meta.get("ocr_review_layout") or "")
        if not layout:
            layout = "single_column" if count == 1 else "column_sentence"
        block_entries = _split_block_for_review(
            block_id=str(block.id),
            block_index=index,
            text=text,
            page=_safe_int(getattr(block, "page", 0) or (page_tuple[0] if page_tuple else 0)),
            pages=page_tuple,
            column_ids=column_ids,
            layout=layout,
            preferred=preferred,
            regions=regions,
            column_texts=column_texts,
            reviewed=bool(meta.get("ocr_image_text_review_checked")),
            changed=bool(meta.get("ocr_image_text_review_changed")),
            preserve_physical_columns=bool(
                (column_ids or regions or column_texts)
                and (
                    meta.get("terminal_checked_on_last_column_only")
                    or meta.get("column_sentence_reflow")
                    or meta.get("atomic_ocr_sentence")
                )
            ),
        )
        if reviewed_segment_map or changed_segment_map:
            for entry in block_entries:
                entry.reviewed = bool(reviewed_segment_map.get(entry.segment_key, False))
                entry.changed = bool(changed_segment_map.get(entry.segment_key, False))
        entries.extend(block_entries)

    # A multi-model fused document carries one immutable comparison row index
    # per review entry.  Structural block order is normally identical, but a
    # split/merge/insertion can anchor rows to a different primary block.  When
    # every entry has a stable row identity, sort by that identity so “第 N 句”
    # means the same sentence in OCR 对比 and 图文对照.
    if entries and all(int(entry.source_row_index) >= 0 for entry in entries):
        entries.sort(key=lambda entry: int(entry.source_row_index))
    return entries


def apply_review_text(
    doc: UnifiedDocument,
    block_id: str,
    text: str,
    *,
    block_index: int | None = None,
) -> tuple[bool, bool]:
    """Save one reviewed sentence. Returns ``(found, changed_from_review_base)``.

    ``block_index`` is preferred when available.  It avoids an O(n) scan on every
    Next/Previous action and also disambiguates malformed legacy documents that
    accidentally contain duplicate block IDs.
    """
    value = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip("\n")

    resolved_index: int | None = None
    if block_index is not None:
        try:
            candidate_index = int(block_index)
        except (TypeError, ValueError, OverflowError):
            candidate_index = -1
        if 0 <= candidate_index < len(doc.blocks):
            candidate = doc.blocks[candidate_index]
            if str(candidate.id) == str(block_id):
                resolved_index = candidate_index
    if resolved_index is None:
        resolved_index = next(
            (index for index, block in enumerate(doc.blocks) if str(block.id) == str(block_id)),
            None,
        )
    if resolved_index is None:
        return False, False

    block = doc.blocks[resolved_index]
    before = str(block.text or "")
    raw_metadata = getattr(block, "metadata", None)
    metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
    stored_review_base = metadata.get("ocr_image_text_review_original_text")
    review_base = before if stored_review_base is None else str(stored_review_base)
    metadata["ocr_image_text_review_original_text"] = review_base
    changed_from_base = value != review_base

    if value != before:
        block.ocr_raw = block.ocr_raw or before
        block.text = value

    tags = [item for item in str(block.modified_by or "").split(",") if item]
    if changed_from_base:
        if "ocr_image_text_review" not in tags:
            tags.append("ocr_image_text_review")
    else:
        tags = [item for item in tags if item != "ocr_image_text_review"]
    block.modified_by = ",".join(tags)
    block.metadata = {
        **metadata,
        "ocr_image_text_review_checked": True,
        "ocr_image_text_review_changed": changed_from_base,
    }

    if block.type in {BlockType.CHAPTER, BlockType.SECTION}:
        for toc in doc.toc:
            if _safe_int(toc.block_index, -1) == resolved_index:
                toc.title = value
    return True, changed_from_base


def clone_for_review(doc: UnifiedDocument) -> UnifiedDocument:
    return copy.deepcopy(doc)


def _paper_colour(image: Image.Image) -> tuple[int, int, int]:
    rgb = image.convert("RGB")
    try:
        points = [
            rgb.getpixel((0, 0)),
            rgb.getpixel((max(0, rgb.width - 1), 0)),
            rgb.getpixel((0, max(0, rgb.height - 1))),
            rgb.getpixel((max(0, rgb.width - 1), max(0, rgb.height - 1))),
        ]
        return tuple(int(sum(point[i] for point in points) / len(points)) for i in range(3))
    finally:
        rgb.close()


def _is_readable_image(path: Path) -> bool:
    try:
        if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
            return False
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError, TypeError):
        return False


def _save_png_atomic(image: Image.Image, output: Path) -> None:
    """Write a cache image atomically so interrupted saves are never reused."""
    temporary = output.with_name(
        f".{output.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        image.save(temporary, format="PNG", compress_level=1)
        temporary.replace(output)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def render_review_image(entry: OCRReviewEntry, output_path: str | Path) -> str:
    """Return an existing exact sentence image or lazily compose source regions.

    Region order follows Japanese reading order: the first logical column is
    placed at the far right, matching the OCR sentence-group constructor.
    Corrupt preferred/cache files are ignored and rebuilt from immutable source
    regions instead of leaving the GUI paired with the previous sentence image.
    """
    preferred = Path(entry.preferred_image_path) if entry.preferred_image_path else None
    if preferred is not None and _is_readable_image(preferred):
        return str(preferred)
    if not entry.regions:
        return ""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if _is_readable_image(output):
        return str(output)
    try:
        output.unlink(missing_ok=True)
    except OSError:
        pass

    strips: list[Image.Image] = []
    source_cache: dict[str, Image.Image] = {}
    background = (255, 255, 255)
    incomplete = False
    try:
        for region in entry.regions:
            if not isinstance(region, dict):
                incomplete = True
                continue
            page_path = str(region.get("page_path") or "")
            bbox = region.get("bbox")
            if (
                not page_path
                or not Path(page_path).exists()
                or not isinstance(bbox, (list, tuple))
                or len(bbox) < 4
            ):
                incomplete = True
                continue
            source = source_cache.get(page_path)
            if source is None:
                try:
                    with Image.open(page_path) as opened:
                        source = opened.convert("RGB")
                except Exception:
                    incomplete = True
                    continue
                source_cache[page_path] = source
            try:
                x, y, w, h = [float(value) for value in bbox[:4]]
            except (TypeError, ValueError, OverflowError):
                incomplete = True
                continue
            if not all(math.isfinite(value) for value in (x, y, w, h)):
                incomplete = True
                continue
            if not strips:
                background = _paper_colour(source)
            left = max(0, min(source.width - 1, round(x * source.width)))
            top = max(0, min(source.height - 1, round(y * source.height)))
            right = max(left + 1, min(source.width, round((x + w) * source.width)))
            bottom = max(top + 1, min(source.height, round((y + h) * source.height)))
            width = max(1, right - left)
            context_x = max(8, round(width * 0.18))
            context_y = max(12, round(width * 0.55))
            crop = source.crop((
                max(0, left - context_x),
                max(0, top - context_y),
                min(source.width, right + context_x),
                min(source.height, bottom + context_y),
            )).convert("RGB")
            if any((
                bool(region.get("column_auto_filter_ruby", False)),
                bool(region.get("column_filter_fragments", False)),
                bool(region.get("column_smart_crop", False)),
            )):
                try:
                    from adapters.column_image_cleanup import cleanup_column_image
                    cleaned = cleanup_column_image(
                        crop,
                        auto_filter_ruby=bool(region.get("column_auto_filter_ruby", False)),
                        filter_fragments=bool(region.get("column_filter_fragments", False)),
                        smart_crop=bool(region.get("column_smart_crop", False)),
                        ruby_strength=str(region.get("column_ruby_strength", "standard") or "standard"),
                        background=background,
                    )
                    crop.close()
                    crop = cleaned.image
                except Exception:
                    # Review must remain usable even if a third-party Pillow
                    # build cannot run the optional cleanup pass.
                    pass
            strips.append(crop)
        # A partial crop is more dangerous than no crop because it visually pairs
        # a complete OCR sentence with only some of its source columns/pages.
        if incomplete or len(strips) != len(entry.regions):
            return ""
        if not strips:
            return ""
        if len(strips) == 1:
            _save_png_atomic(strips[0], output)
            return str(output)

        widths = [strip.width for strip in strips]
        typical = sorted(widths)[len(widths) // 2]
        gap = max(6, round(typical * 0.42))
        margin_x = max(10, round(typical * 0.50))
        margin_y = max(8, round(typical * 0.30))
        canvas_width = sum(widths) + gap * (len(strips) - 1) + margin_x * 2
        canvas_height = max(strip.height for strip in strips) + margin_y * 2
        canvas = Image.new("RGB", (canvas_width, canvas_height), background)
        try:
            cursor = canvas_width - margin_x
            for strip in strips:
                cursor -= strip.width
                canvas.paste(strip, (cursor, margin_y))
                cursor -= gap
            _save_png_atomic(canvas, output)
        finally:
            canvas.close()
        return str(output)
    finally:
        for strip in strips:
            strip.close()
        for source in source_cache.values():
            source.close()

