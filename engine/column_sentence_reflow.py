#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Column-aware sentence reconstruction for vertical OCR.

OCR adapters often return one physical vertical column (or a small region) per
block.  A sentence may therefore span 2, 3, or more blocks and may cross a page
boundary.  This module implements an explicit state machine:

    start column -> append columns until a true terminal -> emit one paragraph

A pending sentence is deliberately *not* flushed at a page boundary.  Titles
are structural barriers and are always emitted as standalone blocks.
"""
from __future__ import annotations

import copy
import re
import statistics
from typing import Iterable

from models.document import UnifiedDocument, Block, BlockType

BODY_TYPES = {BlockType.PARAGRAPH, BlockType.DIALOGUE, BlockType.RUBY}
TITLE_TYPES = {BlockType.CHAPTER, BlockType.SECTION, BlockType.TOC_ENTRY}
CLOSING_QUOTES = "」』）】》〉〕］〗〙〛”’\"'"
STRONG_TERMINALS = "。．！？!?｡!?"
ELLIPSIS_TERMINALS = ("……", "…", "‥‥", "...")
DASH_TERMINALS = ("——", "――", "━━", "──")
REFLOW_VERSION = 9

TITLE_RE = re.compile(
    r"^(?:序章|終章|プロローグ|エピローグ|後記|あとがき|幕間|"
    r"第[一二三四五六七八九十百千〇零\d０-９]+[章話節巻回幕篇編]|"
    r"(?:Chapter|Episode|EP)\s*[.．_-]*[\d０-９]+)", re.I,
)

# OCR 经常把同一个省略号拆成多个区域，或混用半角点、全角点、中点、
# 省略号。连续两个以上才归一化，单个“・”仍保留为正常日文中点。
_DOT_RUN_RE = re.compile(r"[.．・･…‥]{2,}")
_QUOTE_OPENERS = "「『"
_QUOTE_CLOSERS = "」』"
_EXPECTED_QUOTE_CLOSE = {"「": "」", "『": "』"}


def _normalize_ellipsis_runs(value: str) -> str:
    return _DOT_RUN_RE.sub("……", value)


def _quote_stack_after(value: str) -> list[str]:
    """Return still-open Japanese quotes, tolerating one nested OCR mismatch.

    A common vertical-OCR error is ``「...「...』...」``: the nested opener is
    read as 「 while its closer remains 』.  When a mismatched closer occurs
    inside an already-open outer quote, consume only the nested level.  This
    preserves the outer dialogue boundary without changing the source text.
    """
    stack: list[str] = []
    for ch in value:
        if ch in _QUOTE_OPENERS:
            stack.append(ch)
            continue
        if ch not in _QUOTE_CLOSERS or not stack:
            continue
        expected = _EXPECTED_QUOTE_CLOSE[stack[-1]]
        if ch == expected:
            stack.pop()
        elif len(stack) > 1:
            # Nested quote mismatch: close the inner level only.
            stack.pop()
    return stack


def _has_unclosed_japanese_quote(value: str) -> bool:
    return bool(_quote_stack_after(value))


def normalize_column_text(text: str) -> str:
    """Remove layout whitespace while preserving intentional inner ASCII spaces."""
    value = (text or "").strip(" \t\r\n　")
    # Vertical OCR commonly inserts line breaks or full-width spaces between
    # characters/short runs.  Japanese text does not require those spaces.
    value = re.sub(r"[\r\n]+", "", value)
    value = re.sub(r"(?<=[\u3040-\u30ff\u3400-\u9fff」』）】》〉])\s+(?=[\u3040-\u30ff\u3400-\u9fff「『（【《〈])", "", value)
    return _normalize_ellipsis_runs(value)



def join_column_parts(parts: Iterable[str]) -> str:
    output = ""
    for raw in parts:
        part = normalize_column_text(raw)
        if not part:
            continue
        if output and output[-1].isascii() and output[-1].isalnum() and part[0].isascii() and part[0].isalnum():
            output += " "
        output += part
    # Individual OCR regions may each contain only one dot.  Normalize once more
    # after joining so ``." + "." + ".`` becomes one ellipsis rather than
    # three artificial sentences.
    return _normalize_ellipsis_runs(output)


_QUOTE_OPEN_FOR_CLOSE = {"」": "「", "』": "『"}


def _closes_outer_quoted_sentence(value: str, start: int, close_index: int) -> bool:
    """True when ``close_index`` closes the sentence-level outer quotation.

    ``『...』`` inside ordinary prose remains inline.  For a physical column that
    starts with a Japanese opening quote, nested quote pairs are tracked with a
    small OCR-tolerant stack so the final closing quote can be recognized as the
    *column-tail* terminal without inspecting or splitting any inner punctuation.
    """
    segment = value[start:close_index + 1].lstrip(" \t　")
    if not segment or segment[0] not in _QUOTE_OPENERS:
        return False

    stack: list[str] = []
    for offset, ch in enumerate(segment):
        if ch in _QUOTE_OPENERS:
            stack.append(ch)
            continue
        if ch not in _QUOTE_CLOSERS or not stack:
            continue

        expected = _EXPECTED_QUOTE_CLOSE[stack[-1]]
        if ch == expected:
            stack.pop()
        elif len(stack) > 1:
            # OCR-confused nested pair, e.g. inner 「 closed as 』.
            stack.pop()
        else:
            # Do not let a mismatched closer consume the outermost quote.
            continue

        if not stack:
            return offset == len(segment) - 1
    return False


def has_sentence_terminal(text: str) -> bool:
    """Unicode-aware terminal check shared by Japanese/Chinese/English text.

    ``、`` / ``,`` / ``，`` are intentionally *not* terminals.  A bare closing
    quote ends a sentence only for a sentence-level quotation such as
    ``「行く」``; inline terms such as ``感情は『怒り』『恐怖』`` stay together.
    A dash counts only when it is at the actual end of the reconstructed text.
    """
    value = normalize_column_text(text).rstrip()
    if not value:
        return False

    # A punctuation mark inside an as-yet-unclosed dialogue is not the end of
    # the OCR sentence/column group.  This keeps ``「だって.....`` attached to
    # the following text until the final 」 arrives.
    if value[-1] not in _QUOTE_CLOSERS and _has_unclosed_japanese_quote(value):
        return False

    if value[-1] in "」』":
        inner = value[:-1].rstrip()
        if inner and has_sentence_terminal(inner):
            return True
        opening = _QUOTE_OPEN_FOR_CLOSE[value[-1]]
        # OCR sometimes drops the opening quote but keeps the final close quote.
        # Preserve the long-standing dialogue fallback only for an orphan close;
        # a present inline opening quote must pass the sentence-level test below.
        if opening not in value:
            return True
        return _closes_outer_quoted_sentence(value, 0, len(value) - 1)

    core = value.rstrip(CLOSING_QUOTES).rstrip()
    if not core:
        return False
    if core[-1] in STRONG_TERMINALS:
        return True
    if core[-1] == ".":
        # Keep English periods, but a lone OCR dot after Japanese text is often
        # just one fragment of ``...`` and must wait for the next region.
        if len(core) < 2 or not core[-2].isascii() or not core[-2].isalnum():
            return False
        return True
    if any(core.endswith(mark) for mark in ELLIPSIS_TERMINALS):
        return True
    if any(core.endswith(mark) for mark in DASH_TERMINALS):
        return True
    return False


_POST_QUOTE_CONTINUATION_PREFIXES = (
    "と", "って", "という", "との", "とは", "と、", "と。",
    "が", "は", "を", "に", "で", "へ", "も", "の", "から", "ので",
    "けど", "けれど", "ながら", "ため",
)


def is_provisional_quote_terminal(text: str) -> bool:
    value = normalize_column_text(text).rstrip()
    if not value or value[-1] not in "」』" or not has_sentence_terminal(value):
        return False
    inner = value[:-1].rstrip()
    if not inner:
        return False
    # ``。」「？」「！」` are final.  A bare sentence-level close quote is only
    # provisional because the next physical column may begin with quotative ``と``.
    return not has_sentence_terminal(inner)


def starts_post_quote_continuation(text: str) -> bool:
    value = normalize_column_text(text).lstrip()
    return value.startswith(_POST_QUOTE_CONTINUATION_PREFIXES)


def column_group_line(text: str) -> str:
    """Normalize one completed physical-column group without internal splitting.

    The OCR reflow rule is intentionally *column-tail only*: punctuation inside
    a physical column is content, not a boundary.  A completed group therefore
    always becomes exactly one output block/line.
    """
    return normalize_column_text(text)


def _page(block: Block) -> int:
    for value in (block.page_index, block.page_number, block.page):
        if value is not None:
            try:
                return int(value)
            except Exception:
                continue
    return 0


def _order(block: Block) -> tuple[int, int, int]:
    page = _page(block)
    within = block.order_in_page
    if within is None:
        within = block.reading_order
    return page, int(within or 0), int(block.reading_order or 0)


def _looks_title(block: Block) -> bool:
    if block.type in TITLE_TYPES:
        return True
    text = normalize_column_text(block.text)
    return bool(text and TITLE_RE.match(text) and len(text) <= 100)


def _merge_type(parts: list[Block], text: str) -> BlockType:
    if text.startswith(("「", "『")) or text.endswith(("」", "』")):
        return BlockType.DIALOGUE
    if any(part.type == BlockType.RUBY for part in parts):
        return BlockType.RUBY
    return BlockType.PARAGRAPH


def _append_modified_by(value: str, name: str) -> str:
    items = [item for item in (value or "").split(",") if item]
    if name not in items:
        items.append(name)
    return ",".join(items)


def _source_column_ids(block: Block) -> list[str]:
    metadata = block.metadata or {}
    values = metadata.get("source_column_ids") or []
    if isinstance(values, str):
        values = [values]
    out = [str(value) for value in values if str(value)]
    column_id = str(metadata.get("column_id", ""))
    if column_id and column_id not in out:
        out.append(column_id)
    return out


def _source_column_seed_flags(block: Block) -> list[bool]:
    metadata = block.metadata or {}
    column_ids = _source_column_ids(block)
    values = metadata.get("source_column_consensus_seed_flags") or []
    if isinstance(values, (list, tuple)) and len(values) == len(column_ids):
        return [bool(value) for value in values]
    flag = bool(metadata.get("column_consensus_seeded", False))
    return [flag for _column_id in column_ids]


def _collect_source_columns(blocks: list[Block]) -> tuple[list[str], list[bool]]:
    ids: list[str] = []
    flags: list[bool] = []
    seen: set[str] = set()
    for block in blocks:
        block_ids = _source_column_ids(block)
        block_flags = _source_column_seed_flags(block)
        for index, column_id in enumerate(block_ids):
            if not column_id or column_id in seen:
                continue
            seen.add(column_id)
            ids.append(column_id)
            flags.append(bool(block_flags[index]) if index < len(block_flags) else False)
    return ids, flags


def _source_column_metadata_values(
    block: Block, *, scalar_key: str, array_key: str, default: str = ""
) -> list[str]:
    metadata = block.metadata or {}
    column_ids = _source_column_ids(block)
    values = metadata.get(array_key) or []
    if isinstance(values, (list, tuple)) and len(values) == len(column_ids):
        return [str(value or default) for value in values]
    scalar = str(metadata.get(scalar_key, default) or default)
    return [scalar for _column_id in column_ids]


def _collect_source_column_metadata(
    blocks: list[Block], *, scalar_key: str, array_key: str, default: str = ""
) -> list[str]:
    values_by_id: dict[str, str] = {}
    order: list[str] = []
    for block in blocks:
        block_ids = _source_column_ids(block)
        block_values = _source_column_metadata_values(
            block, scalar_key=scalar_key, array_key=array_key, default=default
        )
        for index, column_id in enumerate(block_ids):
            if not column_id or column_id in values_by_id:
                continue
            order.append(column_id)
            values_by_id[column_id] = (
                block_values[index] if index < len(block_values) else default
            )
    return [values_by_id[column_id] for column_id in order]


def _review_region(block: Block) -> dict | None:
    """Capture immutable source geometry for sentence-level image proofreading."""
    bounds = _bbox_tuple(block)
    page = _page(block)
    if bounds is None or page <= 0:
        return None
    x, y, w, h = bounds
    metadata = block.metadata or {}
    return {
        "page": page,
        "bbox": [x, y, w, h],
        "column_id": str(metadata.get("column_id", "") or ""),
        "column_auto_filter_ruby": bool(metadata.get("column_auto_filter_ruby", False)),
        "column_filter_fragments": bool(metadata.get("column_filter_fragments", False)),
        "column_smart_crop": bool(metadata.get("column_smart_crop", False)),
        "column_ruby_strength": str(metadata.get("column_ruby_strength", "standard") or "standard"),
    }


def _is_ocr_column_document(doc: UnifiedDocument) -> bool:
    if bool(getattr(doc.metadata, "pdf_text_layer_mode", False)):
        return False
    source = str(getattr(doc.metadata, "source_engine", "") or "").lower()
    known = (
        "ocr", "vision", "ndlocr", "paddle", "manga",
        "pdf_craft", "google_vision", "hybrid",
    )
    if any(token in source for token in known):
        return True
    evidence = 0
    for block in doc.blocks:
        if block.type not in BODY_TYPES:
            continue
        if block.source_format == "ocr" or block.text_direction or block.bbox is not None:
            evidence += 1
            if evidence >= 2:
                return True
    return False


_PUNCT_ONLY_FRAGMENT_RE = re.compile(r"^[.．・･…‥。！？!?、，,」』）】》〉〕］〗〙〛ー―—─━]+$")


def _bbox_tuple(block: Block) -> tuple[float, float, float, float] | None:
    box = block.bbox
    if box is None:
        return None
    try:
        x, y, w, h = float(box.x), float(box.y), float(box.w), float(box.h)
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def _is_vertical_fragment(block: Block, *, allow_punctuation: bool = False) -> bool:
    bounds = _bbox_tuple(block)
    if bounds is None:
        return False
    direction = str(block.text_direction or "").strip().lower()
    if direction:
        if "horizontal" in direction:
            return False
        if "vertical" in direction:
            return True
    _, _, w, h = bounds
    if h >= w * 1.15:
        return True
    if allow_punctuation and _PUNCT_ONLY_FRAGMENT_RE.fullmatch(normalize_column_text(block.text)):
        return True
    return False


def _same_physical_vertical_column(parts: list[Block], candidate: Block) -> bool:
    """Return True when ``candidate`` belongs to the same printed vertical column.

    OCR engines can split one printed column at punctuation, blank space, ruby,
    or paragraph boundaries.  Reading-order blocks are therefore not equivalent
    to physical columns.  We compare normalized x bands only; y is deliberately
    ignored because fragments in one column are expected to be vertically
    disjoint.
    """
    candidate_box = _bbox_tuple(candidate)
    if candidate_box is None or not _is_vertical_fragment(candidate, allow_punctuation=True):
        return False
    usable = [part for part in parts if _bbox_tuple(part) is not None]
    if not usable or not any(_is_vertical_fragment(part) for part in usable):
        return False

    centers = []
    widths = []
    for part in usable:
        x, _, w, _ = _bbox_tuple(part)  # type: ignore[misc]
        centers.append(x + w / 2.0)
        widths.append(w)
    center = statistics.median(centers)
    width = statistics.median(widths)

    cx, _, cw, _ = candidate_box
    candidate_center = cx + cw / 2.0
    left = center - width / 2.0
    right = center + width / 2.0
    candidate_right = cx + cw
    overlap = max(0.0, min(right, candidate_right) - max(left, cx))
    overlap_ratio = overlap / max(min(width, cw), 1e-6)

    # Same-column fragments usually have nearly identical centers.  The fixed
    # floor handles tiny punctuation boxes; the capped width term avoids
    # swallowing the adjacent printed column on narrow Japanese book pages.
    center_tolerance = max(0.008, min(0.024, max(width, cw) * 0.55))
    center_distance = abs(candidate_center - center)
    relaxed_tolerance = max(center_tolerance, min(0.030, max(width, cw) * 0.75))
    return (
        center_distance <= center_tolerance
        or (overlap_ratio >= 0.70 and center_distance <= relaxed_tolerance)
    )


def _union_bbox(parts: list[Block]):
    boxes = [_bbox_tuple(part) for part in parts]
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return None
    from models.document import BoundingBox
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[0] + box[2] for box in boxes)
    bottom = max(box[1] + box[3] for box in boxes)
    return BoundingBox(x=left, y=top, w=right - left, h=bottom - top)


def _coalesce_physical_column_fragments(
    result: UnifiedDocument,
) -> tuple[set[int], int, int]:
    """Merge OCR fragments that occupy the same physical vertical column.

    Returns ``(removed_positions, merged_fragment_count, physical_column_count)``.
    This pass runs before sentence reflow.  It never uses punctuation to decide
    a boundary; only page, geometry, direction, and x-band alignment are used.
    """
    ordered = list(enumerate(result.blocks))
    ordered.sort(key=lambda pair: (_order(pair[1]), pair[0]))
    removed_positions: set[int] = set()
    merged_fragment_count = 0
    physical_column_count = 0
    preserve_layout = bool(getattr(result.metadata, "preserve_ocr_layout", False))

    index = 0
    while index < len(ordered):
        pos, block = ordered[index]
        text = normalize_column_text(block.text)
        if (
            block.type not in BODY_TYPES
            or not text
            or _looks_title(block)
            or not _is_vertical_fragment(block)
        ):
            index += 1
            continue

        group: list[tuple[int, Block]] = [(pos, block)]
        cursor = index + 1
        while cursor < len(ordered):
            next_pos, next_block = ordered[cursor]
            next_text = normalize_column_text(next_block.text)
            if _page(next_block) != _page(block):
                break
            if next_block.type not in BODY_TYPES or not next_text or _looks_title(next_block):
                break
            if not _same_physical_vertical_column([item[1] for item in group], next_block):
                break
            group.append((next_pos, next_block))
            cursor += 1

        physical_column_count += 1
        if len(group) > 1:
            # Printed vertical text reads top-to-bottom inside one x band.
            text_order = sorted(
                group,
                key=lambda item: (
                    (_bbox_tuple(item[1]) or (0.0, 0.0, 0.0, 0.0))[1],
                    int(item[1].order_in_page or item[1].reading_order or 0),
                    item[0],
                ),
            )
            anchor_pos, anchor = group[0]
            source_blocks = [item[1] for item in text_order]
            source_texts = [normalize_column_text(item.text) for item in source_blocks]
            merged_text = join_column_parts(source_texts)
            source_ids = [item.id for item in source_blocks]
            original_texts = [item.text for item in source_blocks]
            source_column_ids, source_column_seed_flags = _collect_source_columns(source_blocks)
            source_input_hashes = _collect_source_column_metadata(
                source_blocks,
                scalar_key="column_ocr_input_sha256",
                array_key="source_column_ocr_input_sha256",
            )
            source_input_profiles = _collect_source_column_metadata(
                source_blocks,
                scalar_key="column_ocr_input_profile",
                array_key="source_column_ocr_input_profile",
                default="custom",
            )
            source_profile_hashes = _collect_source_column_metadata(
                source_blocks,
                scalar_key="column_ocr_input_profile_sha256",
                array_key="source_column_ocr_input_profile_sha256",
            )

            anchor.text = merged_text
            anchor.ocr_raw = merged_text
            anchor.type = _merge_type(source_blocks, merged_text)
            anchor.text_direction = "vertical"
            anchor.bbox = _union_bbox(source_blocks)
            anchor.modified_by = _append_modified_by(anchor.modified_by, "physical_column_coalesce")
            anchor.metadata = {
                **(anchor.metadata or {}),
                "physical_column_coalesced": True,
                "physical_column_fragment_count": len(source_blocks),
                "physical_column_source_ids": source_ids,
                "physical_column_source_texts": original_texts,
                "source_column_ids": source_column_ids,
                "source_column_consensus_seed_flags": source_column_seed_flags,
                "source_column_ocr_input_sha256": source_input_hashes,
                "source_column_ocr_input_profile": source_input_profiles,
                "source_column_ocr_input_profile_sha256": source_profile_hashes,
                "physical_column_tail_text": source_texts[-1],
                "physical_column_x_center": (
                    anchor.bbox.x + anchor.bbox.w / 2.0 if anchor.bbox is not None else None
                ),
            }

            for duplicate_pos, duplicate in group[1:]:
                duplicate.ocr_raw = duplicate.ocr_raw or duplicate.text
                duplicate.text = ""
                duplicate.modified_by = _append_modified_by(
                    duplicate.modified_by, "physical_column_coalesce"
                )
                duplicate.metadata = {
                    **(duplicate.metadata or {}),
                    "consumed_by_physical_column": anchor.id,
                    "layout_placeholder": preserve_layout,
                }
                if not preserve_layout:
                    removed_positions.add(duplicate_pos)
            merged_fragment_count += len(group) - 1

        index = cursor if cursor > index + 1 else index + 1

    return removed_positions, merged_fragment_count, physical_column_count


def reflow_columns_into_sentences(
    doc: UnifiedDocument,
    *,
    max_columns: int = 64,
    cancel_check=None,
) -> UnifiedDocument:
    """Merge sequential OCR columns until a complete sentence is reached.

    The pending buffer survives page transitions.  Structural titles terminate
    the buffer and stay on their own line.  Completion is decided only from the
    current physical column's final effective character; punctuation inside that
    column is never used to split it.  ``max_columns`` is a corruption guard.
    """
    if callable(cancel_check) and cancel_check():
        raise InterruptedError("OCR 已停止")
    result = copy.deepcopy(doc)
    if callable(cancel_check) and cancel_check():
        raise InterruptedError("OCR 已停止")
    if bool(getattr(result.metadata, "column_sentence_reflow_applied", False)):
        result.add_log("column_sentence_reflow", "逐列成句已在 OCR 界面执行，Formatter 跳过重复处理", 0)
        return result
    if not _is_ocr_column_document(result):
        result.add_log("column_sentence_reflow", "非 OCR/无物理列信息文档，跳过逐列成句", 0)
        return result
    if max_columns < 2:
        max_columns = 2

    # First reconstruct true printed columns.  Layout engines may return several
    # boxes for one x band (for example ``いや。`` and the following text), and
    # punctuation must never turn those OCR fragments into separate columns.
    fragment_removed_positions, fragment_merge_count, physical_column_count = (
        _coalesce_physical_column_fragments(result)
    )

    indexed = list(enumerate(result.blocks))
    # OCR adapters normally already provide global reading_order, but sorting by
    # page/order makes cross-page behavior deterministic for imported JSON too.
    indexed.sort(key=lambda pair: (_order(pair[1]), pair[0]))

    pending: list[Block] = []
    pending_positions: list[int] = []
    removed_positions: set[int] = set(fragment_removed_positions)
    merged_count = 0
    cross_page_count = 0
    capped_count = 0

    def flush(*, reason: str) -> None:
        nonlocal pending, pending_positions, merged_count, cross_page_count, capped_count
        if not pending:
            return
        first = pending[0]
        texts = [normalize_column_text(block.text) for block in pending]
        merged_text = join_column_parts(texts)
        if not merged_text:
            pending = []
            pending_positions = []
            return

        pages = list(dict.fromkeys(_page(block) for block in pending if _page(block)))
        source_ids = [block.id for block in pending]
        original_texts = [block.text for block in pending]
        source_column_ids, source_column_seed_flags = _collect_source_columns(pending)
        source_input_hashes = _collect_source_column_metadata(
            pending,
            scalar_key="column_ocr_input_sha256",
            array_key="source_column_ocr_input_sha256",
        )
        source_input_profiles = _collect_source_column_metadata(
            pending,
            scalar_key="column_ocr_input_profile",
            array_key="source_column_ocr_input_profile",
            default="custom",
        )
        source_profile_hashes = _collect_source_column_metadata(
            pending,
            scalar_key="column_ocr_input_profile_sha256",
            array_key="source_column_ocr_input_profile_sha256",
        )
        review_regions = [
            region for region in (_review_region(block) for block in pending)
            if region is not None
        ]
        preferred_review_image = str(
            (first.metadata or {}).get("ocr_review_sentence_image_path", "") or ""
        )
        effective_column_texts = list(texts)
        context_applied = False
        context_candidate = ""
        context_meta = first.metadata or {}
        context_ids = context_meta.get("sentence_context_reocr_column_ids") or []
        if isinstance(context_ids, str):
            context_ids = [context_ids]
        context_ids = [str(value) for value in context_ids if str(value)]
        context_baseline = normalize_column_text(
            str(context_meta.get("sentence_context_reocr_baseline", "") or "")
        )
        if (
            bool(context_meta.get("sentence_context_reocr_owner"))
            and bool(context_meta.get("sentence_context_reocr_accepted"))
            and context_ids == source_column_ids
            and (not context_baseline or context_baseline == join_column_parts(texts))
        ):
            context_candidate = normalize_column_text(
                str(context_meta.get("sentence_context_reocr_candidate", "") or "")
            )
            if context_candidate and has_sentence_terminal(context_candidate):
                merged_text = context_candidate
                # Preserve one record per immutable physical column for strict
                # auditing and multi-model alignment.  Put the complete sentence
                # on the final column ID so the canonical terminal still occurs
                # at the real end of this sentence group.
                effective_column_texts = [""] * max(0, len(texts) - 1) + [context_candidate]
                context_applied = True
        first.text = column_group_line(merged_text)
        first.type = _merge_type(pending, first.text)
        first.ocr_raw = first.ocr_raw or original_texts[0]
        first.modified_by = _append_modified_by(first.modified_by, "column_sentence_reflow")
        first.metadata = {
            **(first.metadata or {}),
            "column_sentence_reflow": True,
            "column_count": len(pending),
            "source_block_ids": source_ids,
            "source_column_ids": source_column_ids,
            "source_column_texts": effective_column_texts,
            "source_column_primary_texts": texts,
            "source_column_consensus_seed_flags": source_column_seed_flags,
            "source_column_ocr_input_sha256": source_input_hashes,
            "source_column_ocr_input_profile": source_input_profiles,
            "source_column_ocr_input_profile_sha256": source_profile_hashes,
            "source_column_terminal_flags": [
                has_sentence_terminal(item) for item in effective_column_texts
            ],
            "atomic_ocr_sentence": True,
            "source_pages": pages,
            "flush_reason": reason,
            "sentence_terminal": has_sentence_terminal(merged_text),
            "terminal_checked_on_last_column_only": True,
            "last_column_text": effective_column_texts[-1],
            "sentence_context_reocr_applied": context_applied,
            "ocr_review_regions": review_regions,
            "ocr_review_preferred_image_path": preferred_review_image,
            "ocr_review_layout": (
                str(context_meta.get("sentence_context_reocr_layout") or "sentence_group")
                if preferred_review_image
                else ("single_column" if len(review_regions) <= 1 else "column_sentence")
            ),
            "ocr_review_column_count": max(1, len(source_column_ids) or len(review_regions)),
        }
        if context_applied:
            first.metadata.update({
                "sentence_context_reocr_primary_joined": join_column_parts(texts),
                "sentence_context_reocr_text": context_candidate,
            })
        if reason in {"title_boundary", "max_columns", "document_end"} and not has_sentence_terminal(texts[-1]):
            first.metadata["sentence_incomplete_needs_review"] = True
        if reason == "max_columns":
            first.metadata["column_reflow_safety_cap"] = max_columns
            capped_count += 1
        if len(pending) > 1:
            merged_count += len(pending) - 1
        if len(pages) > 1:
            cross_page_count += 1

        preserve_layout = bool(getattr(result.metadata, "preserve_ocr_layout", False))
        for block, pos, original in zip(pending[1:], pending_positions[1:], original_texts[1:]):
            if preserve_layout:
                block.ocr_raw = block.ocr_raw or original
                block.text = ""
                block.modified_by = _append_modified_by(block.modified_by, "column_sentence_reflow")
                block.metadata = {
                    **(block.metadata or {}),
                    "consumed_by_column_sentence_reflow": first.id,
                    "consumed_text": original,
                    "layout_placeholder": True,
                }
            else:
                removed_positions.add(pos)
        pending = []
        pending_positions = []

    for loop_index, (pos, block) in enumerate(indexed):
        if loop_index % 32 == 0 and callable(cancel_check) and cancel_check():
            raise InterruptedError("OCR 已停止")
        text = normalize_column_text(block.text)
        if _looks_title(block):
            flush(reason="title_boundary")
            block.text = (block.text or "").strip(" \t\r\n")
            # Hard invariant: one physical chapter/section title column is one
            # indivisible comparison sentence.  No later OCR comparison or
            # formatter step may split it at punctuation inside the title.
            if block.type not in TITLE_TYPES and TITLE_RE.match(normalize_column_text(block.text)):
                block.type = BlockType.CHAPTER
            title_ids = _source_column_ids(block)
            title_seed_flags = _source_column_seed_flags(block)
            title_input_hashes = _source_column_metadata_values(
                block,
                scalar_key="column_ocr_input_sha256",
                array_key="source_column_ocr_input_sha256",
            )
            title_input_profiles = _source_column_metadata_values(
                block,
                scalar_key="column_ocr_input_profile",
                array_key="source_column_ocr_input_profile",
                default="custom",
            )
            title_profile_hashes = _source_column_metadata_values(
                block,
                scalar_key="column_ocr_input_profile_sha256",
                array_key="source_column_ocr_input_profile_sha256",
            )
            title_region = _review_region(block)
            block.metadata = {
                **(block.metadata or {}),
                "chapter_title_atomic": True,
                "atomic_ocr_sentence": True,
                "source_column_ids": title_ids,
                "source_column_texts": [normalize_column_text(block.text)],
                "source_column_consensus_seed_flags": title_seed_flags,
                "source_column_ocr_input_sha256": title_input_hashes,
                "source_column_ocr_input_profile": title_input_profiles,
                "source_column_ocr_input_profile_sha256": title_profile_hashes,
                "source_column_terminal_flags": [True],
                "column_count": 1,
                "flush_reason": "chapter_title_atomic",
                "ocr_review_regions": [title_region] if title_region is not None else [],
                "ocr_review_preferred_image_path": str(
                    (block.metadata or {}).get("ocr_review_sentence_image_path", "") or ""
                ),
                "ocr_review_layout": "single_column",
                "ocr_review_column_count": 1,
            }
            continue
        if block.type not in BODY_TYPES or not text:
            # Page images, blank placeholders and headers do not terminate a
            # sentence; this lets a page-final fragment continue on next page.
            continue

        if pending:
            pending_text = join_column_parts(b.text for b in pending)
            if is_provisional_quote_terminal(pending_text) and not starts_post_quote_continuation(text):
                flush(reason="provisional_quote_terminal")

        pending.append(block)
        pending_positions.append(pos)
        last_column_text = text
        if is_provisional_quote_terminal(last_column_text):
            # Hold one-block lookahead.  Most dialogues will flush before the
            # next block; quoted speech followed by ``と彼は言った`` continues.
            pass
        elif has_sentence_terminal(last_column_text):
            # Only the final effective character of the current physical column
            # decides whether this column group is complete.  Never scan or
            # split punctuation that appears inside the same column.
            flush(reason="terminal")
        elif len(pending) >= max_columns:
            flush(reason="max_columns")

    if callable(cancel_check) and cancel_check():
        raise InterruptedError("OCR 已停止")
    flush(reason="document_end")
    if callable(cancel_check) and cancel_check():
        raise InterruptedError("OCR 已停止")
    result.blocks = [block for pos, block in enumerate(result.blocks) if pos not in removed_positions]
    # Keep the original interleaving of images/titles; only consumed text blocks
    # are removed.  Do not inspect punctuation inside a physical column: each
    # completed column group is already exactly one output line.
    result.metadata.column_sentence_reflow_applied = True
    result.metadata.column_sentence_reflow_version = REFLOW_VERSION
    result.metadata.column_sentence_reflow_max_columns = max_columns

    result.add_log(
        "column_sentence_reflow",
        f"同列坐标归并 {fragment_merge_count} 个 OCR 碎片为 {physical_column_count} 条物理列；"
        f"逐列成句合并 {merged_count} 个续列；跨页续接 {cross_page_count} 组；"
        f"同列内部不拆分；安全上限待复核 {capped_count} 组",
        fragment_merge_count + merged_count,
    )
    return result
