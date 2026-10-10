#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hierarchical reading-order recovery for Japanese vertical OCR.

The old implementation clustered the whole page only by X.  That is correct for
ordinary one-stage vertical prose, but it interleaves upper/lower two-stage
(``二段組``) layouts because the right-most column in both stages has almost the
same X coordinate.  This module first discovers horizontal page regions, then
sorts columns inside each region from right to left and glyph/line fragments
from top to bottom.

The implementation is deliberately geometry-only and conservative:
- no OCR text is changed;
- a page is split only when a large horizontal separation is supported by
  duplicated X lanes on both sides of the separation;
- shallow, wide cross-column title/header blocks are treated as independent
  separators instead of being pulled into a neighbouring vertical column;
- pages without reliable geometry fall back to the historical X-clustering
  behaviour.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.document import UnifiedDocument, Block, BlockType


def _x_center(block: Block) -> float:
    if block.bbox is None:
        return 0.5
    return block.bbox.x + block.bbox.w / 2


def _y_center(block: Block) -> float:
    if block.bbox is None:
        return 0.5
    return block.bbox.y + block.bbox.h / 2


def _avg_width(blocks: list[Block]) -> float:
    widths = [b.bbox.w for b in blocks if b.bbox and b.bbox.w > 0]
    return sum(widths) / len(widths) if widths else 0.05


def _median(values: list[float], default: float = 0.0) -> float:
    if not values:
        return default
    ordered = sorted(float(v) for v in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _looks_cross_column(block: Block, *, median_width: float) -> bool:
    """Return True for a shallow horizontal block spanning several text columns.

    A real vertical prose column is normally tall and narrow.  Chapter captions
    and cross-stage headings are often shallow and several times wider than the
    median prose column.  Pulling such a block into an X cluster can bridge two
    otherwise independent layout regions.
    """
    box = block.bbox
    if box is None or box.w <= 0 or box.h <= 0:
        return False
    width_gate = max(0.24, median_width * 3.2)
    return box.w >= width_gate and box.h <= 0.18 and box.w >= box.h * 1.8


def _x_lane_overlap(left: list[Block], right: list[Block]) -> float:
    """How strongly two horizontal groups reuse the same vertical X lanes."""
    if not left or not right:
        return 0.0
    tolerance = max(0.025, _avg_width(left + right) * 0.75)
    a = [_x_center(block) for block in left]
    b = [_x_center(block) for block in right]
    matches = 0
    used: set[int] = set()
    for x in a:
        best = None
        best_distance = tolerance + 1.0
        for index, y in enumerate(b):
            if index in used:
                continue
            distance = abs(x - y)
            if distance <= tolerance and distance < best_distance:
                best = index
                best_distance = distance
        if best is not None:
            used.add(best)
            matches += 1
    return matches / max(1, min(len(a), len(b)))


def _best_horizontal_split(blocks: list[Block]) -> tuple[list[Block], list[Block]] | None:
    """Find one conservative upper/lower stage split.

    The strongest signal for a true Japanese upper/lower two-stage page is a
    large jump in Y *and* reuse of the same X column lanes above and below it.
    Ordinary prose fragments stacked within one vertical column may have Y gaps,
    but do not form two groups with several matching X lanes.
    """
    if len(blocks) < 4:
        return None
    ordered = sorted(blocks, key=_y_center)
    centers = [_y_center(block) for block in ordered]
    candidates: list[tuple[float, int, float]] = []
    for index in range(2, len(ordered) - 1):
        gap = centers[index] - centers[index - 1]
        if gap < 0.12:
            continue
        upper = ordered[:index]
        lower = ordered[index:]
        if len(upper) < 2 or len(lower) < 2:
            continue
        overlap = _x_lane_overlap(upper, lower)
        if overlap < 0.45:
            continue
        # Prefer a physically empty gutter when boxes themselves expose it.
        upper_bottom = max(
            (b.bbox.y + b.bbox.h for b in upper if b.bbox),
            default=centers[index - 1],
        )
        lower_top = min((b.bbox.y for b in lower if b.bbox), default=centers[index])
        gutter = max(0.0, lower_top - upper_bottom)
        score = gap + min(0.18, gutter) + overlap * 0.08
        candidates.append((score, index, overlap))
    if not candidates:
        return None
    _score, index, _overlap = max(candidates, key=lambda item: item[0])
    return ordered[:index], ordered[index:]


def _partition_horizontal_bands(blocks: list[Block], *, max_bands: int = 3) -> list[list[Block]]:
    """Recursively partition a page into top-to-bottom reading regions."""
    if not blocks:
        return []
    bands = [list(blocks)]
    while len(bands) < max_bands:
        best_index = -1
        best_split = None
        best_gap = 0.0
        for index, band in enumerate(bands):
            split = _best_horizontal_split(band)
            if split is None:
                continue
            upper, lower = split
            gap = _y_center(lower[0]) - _y_center(upper[-1])
            if gap > best_gap:
                best_gap = gap
                best_index = index
                best_split = split
        if best_split is None:
            break
        bands[best_index:best_index + 1] = [best_split[0], best_split[1]]
    return sorted(bands, key=lambda band: min((_y_center(b) for b in band), default=0.0))


def _sort_single_band(blocks: list[Block]) -> list[Block]:
    """Historical right-to-left column clustering, scoped to one page band."""
    if len(blocks) <= 1:
        return list(blocks)
    gap_threshold = _avg_width(blocks) * 0.35
    if gap_threshold < 0.01:
        gap_threshold = 0.03

    sorted_by_x = sorted(blocks, key=lambda b: _x_center(b), reverse=True)
    columns: list[list[Block]] = []
    current_col: list[Block] = [sorted_by_x[0]]
    for block in sorted_by_x[1:]:
        # Compare against the running lane centre rather than only the previous
        # member, so a slightly noisy box cannot drift the complete cluster.
        lane_x = sum(_x_center(item) for item in current_col) / len(current_col)
        if abs(_x_center(block) - lane_x) <= gap_threshold:
            current_col.append(block)
        else:
            columns.append(current_col)
            current_col = [block]
    columns.append(current_col)

    for column in columns:
        column.sort(key=lambda b: (_y_center(b), b.reading_order))
    columns.sort(
        key=lambda column: sum(_x_center(item) for item in column) / len(column),
        reverse=True,
    )
    return [block for column in columns for block in column]


@dataclass(slots=True)
class _LayoutUnit:
    y: float
    kind: str
    blocks: list[Block]


def sort_blocks_by_reading_order(blocks: list[Block]) -> list[Block]:
    """Sort vertical Japanese OCR blocks with two-stage/cross-column awareness.

    1. Separate shallow cross-column title/header blocks.
    2. Detect upper/lower page stages (``二段組``) using Y separation + X-lane reuse.
    3. Sort each stage right-to-left by column and top-to-bottom within a column.
    4. Reinsert cross-column blocks according to their vertical position.
    """
    has_bbox = [b for b in blocks if b.bbox is not None]
    no_bbox = [b for b in blocks if b.bbox is None]
    if not has_bbox:
        return list(blocks)

    median_width = _median(
        [b.bbox.w for b in has_bbox if b.bbox and b.bbox.w > 0],
        default=0.05,
    )
    separators = [
        b for b in has_bbox if _looks_cross_column(b, median_width=median_width)
    ]
    separator_ids = {id(block) for block in separators}
    normal = [b for b in has_bbox if id(b) not in separator_ids]
    bands = _partition_horizontal_bands(normal) if normal else []

    units: list[_LayoutUnit] = []
    for band in bands:
        units.append(_LayoutUnit(
            y=min((_y_center(block) for block in band), default=0.0),
            kind="band",
            blocks=_sort_single_band(band),
        ))
    for separator in separators:
        units.append(_LayoutUnit(
            y=_y_center(separator),
            kind="separator",
            blocks=[separator],
        ))
    units.sort(key=lambda unit: (unit.y, 0 if unit.kind == "separator" else 1))

    result = [block for unit in units for block in unit.blocks]
    result.extend(no_bbox)
    return result


def restore_reading_order(doc: UnifiedDocument) -> UnifiedDocument:
    """Restore reading order per page without moving structural blocks.

    ``IMAGE_REF``/``CHAPTER`` and geometry-less blocks retain their original slot.
    Selectable-PDF text-layer documents remain exempt because that adapter has a
    more precise character-coordinate ordering pass of its own.
    """
    if doc.metadata.source_engine == "pdf_text_layer":
        return doc.snapshot_clone()

    doc = doc.snapshot_clone()
    sortable_types = {BlockType.PARAGRAPH, BlockType.DIALOGUE}
    pages: dict[int, list[Block]] = defaultdict(list)
    for block in doc.blocks:
        pages[block.page].append(block)

    new_blocks: list[Block] = []
    global_order = 0
    multi_band_pages = 0
    cross_column_blocks = 0

    for page_no in sorted(pages.keys()):
        page_blocks = pages[page_no]
        sortable = [
            b for b in page_blocks if b.type in sortable_types and b.bbox is not None
        ]
        if sortable:
            median_width = _median(
                [b.bbox.w for b in sortable if b.bbox and b.bbox.w > 0],
                default=0.05,
            )
            cross_column_blocks += sum(
                1 for b in sortable if _looks_cross_column(b, median_width=median_width)
            )
            normal = [
                b for b in sortable
                if not _looks_cross_column(b, median_width=median_width)
            ]
            if len(_partition_horizontal_bands(normal)) > 1:
                multi_band_pages += 1
            sorted_blocks = sort_blocks_by_reading_order(sortable)
        else:
            sorted_blocks = []

        sort_index = 0
        for original in page_blocks:
            if original.type in sortable_types and original.bbox is not None:
                block = sorted_blocks[sort_index]
                sort_index += 1
            else:
                block = original
            block.reading_order = global_order
            global_order += 1
            new_blocks.append(block)

    doc.blocks = new_blocks
    detail = f"；二段/多段页 {multi_band_pages}；跨栏块 {cross_column_blocks}" if (multi_band_pages or cross_column_blocks) else ""
    doc.add_log("reading_order", f"按层次化阅读顺序重排 {len(doc.blocks)} 个块{detail}")
    return doc
