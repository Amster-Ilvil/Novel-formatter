#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Coordinate-safe horizontal reading order for OCR fragments.

OCR engines may return one logical Chinese line as several neighbouring boxes.
This module groups only boxes that clearly share a baseline/vertical overlap,
then joins them from left to right.  It never runs for the Japanese vertical
profile, so existing right-to-left column behaviour is untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from adapters.ocr_profiles import is_chinese_horizontal, normalize_chinese_text


@dataclass(frozen=True)
class _Geom:
    index: int
    item: dict
    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return max(1.0, self.right - self.left)

    @property
    def height(self) -> float:
        return max(1.0, self.bottom - self.top)

    @property
    def cy(self) -> float:
        return (self.top + self.bottom) / 2.0


def _geometry(index: int, item: dict) -> _Geom | None:
    box = item.get("box")
    if not isinstance(box, (list, tuple)) or len(box) < 2:
        return None
    try:
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
    except (TypeError, ValueError, IndexError):
        return None
    if not xs or not ys:
        return None
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    if right <= left or bottom <= top:
        return None
    return _Geom(index, item, left, top, right, bottom)


def _same_row(row: list[_Geom], candidate: _Geom) -> bool:
    row_top = min(part.top for part in row)
    row_bottom = max(part.bottom for part in row)
    overlap = max(0.0, min(row_bottom, candidate.bottom) - max(row_top, candidate.top))
    min_height = max(1.0, min(row_bottom - row_top, candidate.height))
    if overlap / min_height >= 0.38:
        return True
    row_cy = median(part.cy for part in row)
    row_height = median(part.height for part in row)
    return abs(candidate.cy - row_cy) <= max(row_height, candidate.height) * 0.52


def _smart_join(left: str, right: str, gap: float, typical_height: float) -> str:
    a = normalize_chinese_text(left)
    b = normalize_chinese_text(right)
    if not a:
        return b
    if not b:
        return a
    no_space_after = "（《【「『“‘—…"
    no_space_before = "，。！？；：、）》】」』”’…％%"
    if a[-1] in no_space_after or b[0] in no_space_before:
        return a + b
    a_cjk = "\u3400" <= a[-1] <= "\u9fff"
    b_cjk = "\u3400" <= b[0] <= "\u9fff"
    if a_cjk or b_cjk:
        return a + b
    # Preserve a visible separator for Latin/number fragments only when the
    # detector reports a real word-sized gap.  Tight kerning stays unmodified.
    return a + (" " if gap >= max(2.0, typical_height * 0.18) else "") + b


def _merge_row(row: list[_Geom], row_index: int) -> dict:
    ordered = sorted(row, key=lambda part: (part.left, part.top, part.index))
    typical_height = median(part.height for part in ordered)
    text = ""
    previous_right = ordered[0].left
    for part in ordered:
        fragment = str(part.item.get("text") or "")
        text = _smart_join(text, fragment, part.left - previous_right, typical_height)
        previous_right = max(previous_right, part.right)
    confidence_values = []
    for part in ordered:
        try:
            confidence_values.append(float(part.item.get("confidence", 0.9)))
        except (TypeError, ValueError):
            pass
    merged = dict(ordered[0].item)
    merged.update({
        "text": normalize_chinese_text(text),
        "confidence": (
            sum(confidence_values) / len(confidence_values) if confidence_values else 0.9
        ),
        "box": [
            [min(part.left for part in ordered), min(part.top for part in ordered)],
            [max(part.right for part in ordered), min(part.top for part in ordered)],
            [max(part.right for part in ordered), max(part.bottom for part in ordered)],
            [min(part.left for part in ordered), max(part.bottom for part in ordered)],
        ],
        "direction": "horizontal",
        "layout_group": "horizontal_line",
        "layout_order": row_index,
        "horizontal_fragment_count": len(ordered),
        "horizontal_source_indices": [part.index for part in ordered],
    })
    return merged


def prepare_horizontal_items(items: list[dict], *, merge_fragments: bool = True) -> list[dict]:
    boxed: list[_Geom] = []
    unboxed: list[tuple[int, dict]] = []
    for index, raw in enumerate(items or []):
        item = dict(raw or {})
        item["text"] = normalize_chinese_text(str(item.get("text") or ""))
        geom = _geometry(index, item)
        if geom is None:
            unboxed.append((index, item))
        else:
            boxed.append(geom)

    if not boxed:
        return [item for _index, item in unboxed if item.get("text")]

    rows: list[list[_Geom]] = []
    for geom in sorted(boxed, key=lambda part: (part.cy, part.top, part.left, part.index)):
        best_index = None
        best_distance = None
        for row_index, row in enumerate(rows):
            if not _same_row(row, geom):
                continue
            distance = abs(geom.cy - median(part.cy for part in row))
            if best_distance is None or distance < best_distance:
                best_index, best_distance = row_index, distance
        if best_index is None:
            rows.append([geom])
        else:
            rows[best_index].append(geom)

    rows.sort(key=lambda row: (min(part.top for part in row), min(part.left for part in row)))
    output: list[dict] = []
    for row_index, row in enumerate(rows):
        if merge_fragments:
            merged = _merge_row(row, row_index)
            if merged.get("text"):
                output.append(merged)
        else:
            for order, part in enumerate(sorted(row, key=lambda p: (p.left, p.top, p.index))):
                item = dict(part.item)
                item.update(direction="horizontal", layout_group="horizontal_line", layout_order=order)
                if item.get("text"):
                    output.append(item)

    # Engines without coordinates are already expected to return transcript
    # order.  Keep them after the coordinate-sorted content rather than guessing.
    output.extend(item for _index, item in sorted(unboxed) if item.get("text"))
    return output


def prepare_items_for_mode(
    items: list[dict], mode: str | None, *, merge_horizontal_fragments: bool = True
) -> list[dict]:
    if not is_chinese_horizontal(mode):
        return list(items or [])
    return prepare_horizontal_items(items or [], merge_fragments=merge_horizontal_fragments)


__all__ = ["prepare_horizontal_items", "prepare_items_for_mode"]
