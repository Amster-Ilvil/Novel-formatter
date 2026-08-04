#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Japanese handwriting trace review for fixed-region vertical-column OCR.

This module does not replace the primary OCR engine.  It opens an optional,
column-by-column verification workspace after masked-column OCR has completed:

* the detected printed column remains visible as the authoritative reference;
* a clicked glyph is placed underneath a handwriting canvas;
* the user can trace it manually or ask the local auto-trace helper to turn a
  thinned bitmap skeleton into approximate pen strokes;
* JLect JHR proposes one-character Japanese candidates;
* the chosen candidate replaces the selected OCR character;
* edited physical columns return to the existing column-tail sentence reflow.

Automatic trace is deliberately a *candidate helper*.  Printed raster images do
not contain true stroke order or pen direction, so silent automatic replacement
would be unsafe.  Text changes are applied only after the user accepts the review.
"""
from __future__ import annotations

import html
import json
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps

from models.document import Block, BlockType, UnifiedDocument
from adapters.handwriting_image_tools import (
    mask_main_text_band, segment_black_ink_glyphs, segment_black_ink_glyphs_slider,
)
from adapters.unicode_safety import clean_json_value, clean_text, dumps as safe_json_dumps


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_JLECT_JS = _PROJECT_ROOT / "third_party" / "jlect_jhr" / "jlect-jhr.compressed.js"
_JLECT_LICENSE = _PROJECT_ROOT / "third_party" / "jlect_jhr" / "LICENSE.txt"


def _page_number(block: Block) -> int:
    for value in (block.page_index, block.page_number, block.page):
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    return 0


def _is_reviewable_column(block: Block) -> bool:
    metadata = block.metadata or {}
    return (
        metadata.get("layout_group") == "fixed_region_column"
        and bool(metadata.get("column_id"))
        and block.bbox is not None
        and _page_number(block) > 0
    )


def _crop_bbox(image: Image.Image, block: Block, *, margin_ratio: float = 0.10) -> Image.Image:
    """Crop a normalized OCR block from the already-fixed body-region image."""
    bbox = block.bbox
    if bbox is None:
        raise ValueError("block has no bbox")
    width, height = image.size
    x0 = int(round(float(bbox.x) * width))
    y0 = int(round(float(bbox.y) * height))
    x1 = int(round(float(bbox.x + bbox.w) * width))
    y1 = int(round(float(bbox.y + bbox.h) * height))
    margin_x = max(2, int(round(max(1, x1 - x0) * margin_ratio)))
    margin_y = max(2, int(round(max(1, x1 - x0) * 0.16)))
    x0 = max(0, x0 - margin_x)
    x1 = min(width, x1 + margin_x)
    y0 = max(0, y0 - margin_y)
    y1 = min(height, y1 + margin_y)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("invalid normalized column bbox")
    return image.crop((x0, y0, x1, y1)).convert("RGB")



def _review_otsu_threshold(image: Image.Image) -> int:
    """Return a conservative grayscale threshold for review-only box recovery."""
    gray = ImageOps.grayscale(image)
    try:
        hist = gray.histogram()
    finally:
        gray.close()
    total = sum(hist)
    if total <= 0:
        return 220
    weighted = sum(index * count for index, count in enumerate(hist))
    weight_bg = 0
    sum_bg = 0
    best = 180
    best_variance = -1.0
    for threshold, count in enumerate(hist):
        weight_bg += count
        if not weight_bg:
            continue
        weight_fg = total - weight_bg
        if not weight_fg:
            break
        sum_bg += threshold * count
        mean_bg = sum_bg / weight_bg
        mean_fg = (weighted - sum_bg) / weight_fg
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance > best_variance:
            best_variance = variance
            best = threshold
    return max(70, min(242, int(best) + 8))


def _normalise_review_boxes(
    boxes: Iterable[object], *, width: int, height: int,
) -> list[dict]:
    output: list[dict] = []
    for raw in boxes or []:
        if not isinstance(raw, dict):
            continue
        try:
            x0 = max(0, min(width, int(round(float(raw.get("x0", 0))))))
            x1 = max(0, min(width, int(round(float(raw.get("x1", width))))))
            y0 = max(0, min(height, int(round(float(raw.get("y0", 0))))))
            y1 = max(0, min(height, int(round(float(raw.get("y1", height))))))
        except (TypeError, ValueError):
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        item = dict(raw)
        item.update({"x0": x0, "x1": x1, "y0": y0, "y1": y1})
        output.append(item)
    output.sort(key=lambda item: (
        int(item.get("y0", 0)), int(item.get("y1", 0)), int(item.get("x0", 0)),
    ))
    return output


def _align_anchor_sequence(anchor_text: str, target_text: str) -> list[int | None]:
    """Map each target character to one monotonically ordered anchor index."""
    source = list(str(anchor_text or ""))
    target = list(str(target_text or ""))
    m, n = len(source), len(target)
    if not n:
        return []
    if not m:
        return [None] * n
    inf = float("inf")
    dp = [[inf] * (n + 1) for _ in range(m + 1)]
    parent: list[list[tuple[int, int, str] | None]] = [[None] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0.0
    for i in range(m + 1):
        for j in range(n + 1):
            current = dp[i][j]
            if current == inf:
                continue
            if i < m:
                value = current + 1.0
                if value < dp[i + 1][j]:
                    dp[i + 1][j] = value
                    parent[i + 1][j] = (i, j, "drop_source")
            if j < n:
                value = current + 0.92
                if value < dp[i][j + 1]:
                    dp[i][j + 1] = value
                    parent[i][j + 1] = (i, j, "missing_target")
            if i < m and j < n:
                same = source[i] == target[j]
                punctuation_pair = (
                    source[i] in "、。，．・…―—ー！？!?『』「」（）()［］[]"
                    and target[j] in "、。，．・…―—ー！？!?『』「」（）()［］[]"
                )
                cost = 0.0 if same else (0.38 if punctuation_pair else 0.72)
                value = current + cost
                if value < dp[i + 1][j + 1]:
                    dp[i + 1][j + 1] = value
                    parent[i + 1][j + 1] = (i, j, "assign")
    mapping: list[int | None] = [None] * n
    i, j = m, n
    while i > 0 or j > 0:
        step = parent[i][j]
        if step is None:
            break
        pi, pj, action = step
        if action == "assign":
            mapping[j - 1] = i - 1
        i, j = pi, pj
    return mapping


def _tighten_review_slot(
    image: Image.Image, *, y0: int, y1: int, fallback_x0: int, fallback_x1: int,
) -> tuple[int, int, int, int, int]:
    width, height = image.size
    y0 = max(0, min(height - 1, int(y0)))
    y1 = max(y0 + 1, min(height, int(y1)))
    gray = ImageOps.grayscale(image)
    try:
        threshold = _review_otsu_threshold(image)
        crop = gray.crop((0, y0, width, y1))
        try:
            mask = crop.point(lambda value: 255 if value < threshold else 0, mode="L")
            bbox = mask.getbbox()
            ink = int(mask.histogram()[255])
            mask.close()
        finally:
            crop.close()
    finally:
        gray.close()
    if bbox:
        bx0, by0, bx1, by1 = bbox
        pad_x = max(2, int(round(max(1, bx1 - bx0) * 0.12)))
        pad_y = max(1, int(round(max(1, by1 - by0) * 0.08)))
        return (
            max(0, bx0 - pad_x), max(0, y0 + by0 - pad_y),
            min(width, bx1 + pad_x), min(height, y0 + by1 + pad_y), ink,
        )
    return (
        max(0, min(width - 1, fallback_x0)), y0,
        max(1, min(width, fallback_x1)), y1, 0,
    )


def _review_ink_geometry(image: Image.Image) -> tuple[bytearray, list[int], int, int, int, int, int]:
    """Return binary ink geometry used by the OCR-locked review grid.

    Review boxes must represent *characters*, not disconnected strokes.  Apple
    Vision and connected-component previews can legitimately return several
    rectangles for one printed kanji (for example each horizontal in ``言``).
    This helper therefore reduces the raster to x/y ink projections and leaves
    the final one-character partition to :func:`align_review_glyph_boxes`.
    """
    gray = ImageOps.grayscale(image)
    try:
        threshold = _review_otsu_threshold(image)
        width, height = gray.size
        pixels = gray.tobytes()
    finally:
        gray.close()

    binary = bytearray(value < threshold for value in pixels)
    x_projection = [0] * width
    y_projection = [0] * height
    for index, value in enumerate(binary):
        if not value:
            continue
        x = index % width
        y = index // width
        # Ignore a one-pixel scanner/frame edge.  It otherwise becomes a very
        # strong false body band and makes every character crop full width.
        if 0 < x < width - 1 and 0 < y < height - 1:
            x_projection[x] += 1
            y_projection[y] += 1

    total_ink = sum(y_projection)
    if total_ink <= 0:
        return binary, y_projection, 0, width, 0, height, threshold

    def weighted_bounds(values: list[int], low_ratio: float, high_ratio: float) -> tuple[int, int]:
        total = max(1, sum(values))
        low_target = total * low_ratio
        high_target = total * high_ratio
        cumulative = 0
        low = 0
        high = len(values)
        for index, value in enumerate(values):
            cumulative += value
            if cumulative >= low_target:
                low = index
                break
        cumulative = 0
        for index, value in enumerate(values):
            cumulative += value
            if cumulative >= high_target:
                high = index + 1
                break
        return max(0, low), max(low + 1, min(len(values), high))

    # Percentile trimming removes isolated dust while keeping punctuation and
    # detached dakuten.  The column has already passed the main-band mask.
    body_x0, body_x1 = weighted_bounds(x_projection, 0.01, 0.99)
    ink_y0, ink_y1 = weighted_bounds(y_projection, 0.002, 0.998)
    x_pad = max(2, int(round(max(1, body_x1 - body_x0) * 0.08)))
    body_x0 = max(0, body_x0 - x_pad)
    body_x1 = min(width, body_x1 + x_pad)
    return binary, y_projection, body_x0, body_x1, ink_y0, ink_y1, threshold


def _ocr_locked_vertical_boundaries(
    row_ink: list[int], *, top: int, bottom: int, count: int,
) -> tuple[list[int], str]:
    """Partition one vertical column into exactly ``count`` stable cells.

    Boundaries are selected near an OCR-count grid but snap to low-ink valleys.
    A small dynamic programme prevents greedy drift: every resulting interval
    remains close to one character pitch and all intervals stay monotonic.
    """
    height = len(row_ink)
    top = max(0, min(height - 1, int(top))) if height else 0
    bottom = max(top + 1, min(height, int(bottom))) if height else 1
    count = max(1, int(count))
    if count == 1:
        return [top, bottom], "single_cell"

    span = max(count, bottom - top)
    pitch = span / float(count)
    maximum = max(1.0, float(max(row_ink[top:bottom], default=1)))

    # Three-row smoothing makes a one-pixel antialiasing line less influential
    # without erasing genuine whitespace between vertically set characters.
    smooth = [0.0] * height
    for y in range(top, bottom):
        left = max(top, y - 1)
        right = min(bottom, y + 2)
        smooth[y] = sum(row_ink[left:right]) / max(1, right - left)

    candidates_by_boundary: list[list[tuple[int, float]]] = []
    for boundary_index in range(1, count):
        ideal = top + boundary_index * pitch
        radius = max(3, int(round(pitch * 0.42)))
        lo = max(top + 1, int(round(ideal)) - radius)
        hi = min(bottom - 1, int(round(ideal)) + radius)
        scored: list[tuple[float, int]] = []
        for y in range(lo, hi + 1):
            ink_cost = smooth[y] / maximum
            distance_cost = abs(y - ideal) / max(1.0, radius)
            # Cutting black ink is much worse than a modest displacement from
            # the ideal grid.  This is the key to keeping all strokes of one
            # kanji in one review frame.
            score = ink_cost * 4.0 + distance_cost * 0.42
            scored.append((score, y))
        scored.sort(key=lambda item: (item[0], abs(item[1] - ideal), item[1]))
        selected = scored[: min(18, len(scored))]
        ideal_y = max(lo, min(hi, int(round(ideal))))
        if all(y != ideal_y for _score, y in selected):
            selected.append((abs(ideal_y - ideal) * 0.42 / max(1.0, radius), ideal_y))
        candidates_by_boundary.append(sorted({y: score for score, y in selected}.items()))

    # State: candidate y -> (cost, previous y).  Absolute ideal windows already
    # constrain drift; transition penalties keep neighbouring cells coherent.
    states: dict[int, tuple[float, int | None]] = {top: (0.0, None)}
    parents: list[dict[int, int]] = []
    minimum_height = max(2.0, pitch * 0.46)
    maximum_height = max(minimum_height + 1.0, pitch * 1.58)
    for boundary_number, candidates in enumerate(candidates_by_boundary, start=1):
        next_states: dict[int, tuple[float, int | None]] = {}
        parent_map: dict[int, int] = {}
        ideal = top + boundary_number * pitch
        for y, boundary_score in candidates:
            best: tuple[float, int] | None = None
            for previous_y, (previous_cost, _unused) in states.items():
                cell_height = y - previous_y
                if not (minimum_height <= cell_height <= maximum_height):
                    continue
                ratio = cell_height / max(1.0, pitch)
                transition = previous_cost + boundary_score + (ratio - 1.0) ** 2 * 1.35
                # Retain a weak absolute-grid term so a sequence of equally low
                # internal kanji valleys cannot pull all later cells upward.
                transition += (abs(y - ideal) / max(1.0, pitch)) ** 2 * 0.18
                if best is None or transition < best[0]:
                    best = (transition, previous_y)
            if best is not None:
                next_states[y] = (best[0], best[1])
                parent_map[y] = best[1]
        if not next_states:
            uniform = [int(round(top + i * pitch)) for i in range(count + 1)]
            uniform[0], uniform[-1] = top, bottom
            return uniform, "uniform_grid_fallback"
        states = next_states
        parents.append(parent_map)

    best_last: tuple[float, int] | None = None
    for previous_y, (previous_cost, _unused) in states.items():
        final_height = bottom - previous_y
        if not (minimum_height <= final_height <= maximum_height):
            continue
        ratio = final_height / max(1.0, pitch)
        cost = previous_cost + (ratio - 1.0) ** 2 * 1.35
        if best_last is None or cost < best_last[0]:
            best_last = (cost, previous_y)
    if best_last is None:
        uniform = [int(round(top + i * pitch)) for i in range(count + 1)]
        uniform[0], uniform[-1] = top, bottom
        return uniform, "uniform_grid_fallback"

    boundaries = [bottom, best_last[1]]
    current = best_last[1]
    for parent_map in reversed(parents[1:]):
        current = parent_map[current]
        boundaries.append(current)
    boundaries.append(top)
    boundaries = list(reversed(boundaries))
    if len(boundaries) != count + 1 or any(b <= a for a, b in zip(boundaries, boundaries[1:])):
        uniform = [int(round(top + i * pitch)) for i in range(count + 1)]
        uniform[0], uniform[-1] = top, bottom
        return uniform, "uniform_grid_fallback"
    return boundaries, "ocr_locked_valley_dp"


def _review_cell_box(
    image: Image.Image,
    binary: bytearray,
    *,
    body_x0: int,
    body_x1: int,
    cell_y0: int,
    cell_y1: int,
    pitch: float,
) -> tuple[int, int, int, int, int]:
    """Build one whole-character frame for a fixed vertical cell."""
    width, height = image.size
    cell_y0 = max(0, min(height - 1, int(cell_y0)))
    cell_y1 = max(cell_y0 + 1, min(height, int(cell_y1)))
    xs: list[int] = []
    ink = 0
    for y in range(cell_y0, cell_y1):
        row = y * width
        for x in range(max(0, body_x0), min(width, body_x1)):
            if binary[row + x]:
                xs.append(x)
                ink += 1
    band_width = max(1, body_x1 - body_x0)
    if xs:
        x0, x1 = min(xs), max(xs) + 1
        pad_x = max(2, int(round(max(1, x1 - x0) * 0.12)))
        x0 = max(0, x0 - pad_x)
        x1 = min(width, x1 + pad_x)
    else:
        x0, x1 = body_x0, body_x1

    # A stroke fragment must never produce a tiny middle preview.  Expand every
    # x-frame to a substantial portion of the main column band while preserving
    # the exact non-overlapping y cell shared by left/middle/right.
    minimum_width = max(8, int(round(band_width * 0.72)))
    if x1 - x0 < minimum_width:
        centre = (x0 + x1) / 2.0 if x1 > x0 else (body_x0 + body_x1) / 2.0
        x0 = int(round(centre - minimum_width / 2.0))
        x1 = x0 + minimum_width
        if x0 < 0:
            x1 -= x0
            x0 = 0
        if x1 > width:
            x0 -= x1 - width
            x1 = width
        x0 = max(0, x0)

    # Do not tighten y to individual connected components and do not overlap
    # neighbouring cells.  The middle preview adds its own visual padding.
    return x0, cell_y0, x1, cell_y1, ink


def align_review_glyph_boxes(
    image: Image.Image,
    boxes: Iterable[object],
    target_text: str,
    *,
    anchor_text: str = "",
) -> tuple[list[dict], dict]:
    """Return one whole-character frame for every OCR character.

    Detector boxes are treated as weak hints only.  A printed Japanese glyph can
    contain many disconnected strokes, and Apple Vision may return one box for a
    radical or a horizontal bar instead of one box for the complete character.
    The review workspace therefore locks geometry to the OCR character count and
    rebuilds a monotonic valley-snapped cell grid from the original column raster.
    Left highlight, middle crop and right editor selection all consume this same
    list, so they cannot drift to different indexes.
    """
    target = list(str(target_text or ""))
    width, height = image.size
    parsed = _normalise_review_boxes(boxes, width=width, height=height)
    diagnostics = {
        "alignment_target_count": len(target),
        "alignment_input_box_count": len(parsed),
        "alignment_synthetic_count": max(0, len(target) - len(parsed)),
        "alignment_source": "none",
        "alignment_exact": False,
        "alignment_rebuilt": False,
        "alignment_fragment_box_count": 0,
        "alignment_boundary_method": "none",
        "alignment_pitch": 0.0,
    }
    if not target:
        return parsed, diagnostics

    binary, row_ink, body_x0, body_x1, ink_y0, ink_y1, threshold = _review_ink_geometry(image)
    count = len(target)

    # Establish the vertical span from real ink first.  Parsed boxes are only
    # allowed to widen it when they are plausible; isolated OCR fragments cannot
    # shrink or shift the full sequence.
    if ink_y1 <= ink_y0:
        ink_y0, ink_y1 = 0, height
    parsed_tops = sorted(int(item.get("y0", 0)) for item in parsed)
    parsed_bottoms = sorted(int(item.get("y1", height)) for item in parsed)
    if parsed_tops and parsed_bottoms:
        low_index = min(len(parsed_tops) - 1, max(0, int(round((len(parsed_tops) - 1) * 0.08))))
        high_index = min(len(parsed_bottoms) - 1, max(0, int(round((len(parsed_bottoms) - 1) * 0.92))))
        plausible_top = parsed_tops[low_index]
        plausible_bottom = parsed_bottoms[high_index]
        if plausible_bottom > plausible_top:
            ink_y0 = min(ink_y0, plausible_top)
            ink_y1 = max(ink_y1, plausible_bottom)

    initial_pitch = max(4.0, (ink_y1 - ink_y0) / max(1, count))
    vertical_pad = max(1, int(round(initial_pitch * 0.10)))
    top = max(0, ink_y0 - vertical_pad)
    bottom = min(height, ink_y1 + vertical_pad)
    if bottom - top < count * 3:
        top, bottom = 0, height
    pitch = max(1.0, (bottom - top) / max(1, count))

    # Quantify how unreliable the incoming geometry is.  This is diagnostic only:
    # even an exact count is rebuilt, because 39 fragment boxes can still represent
    # fewer than 39 actual characters.
    fragment_count = 0
    overlap_count = 0
    previous_y1 = -1
    for item in parsed:
        box_height = max(1.0, float(item.get("y1", 0)) - float(item.get("y0", 0)))
        box_width = max(1.0, float(item.get("x1", 0)) - float(item.get("x0", 0)))
        if box_height < pitch * 0.46 or box_width < max(3.0, (body_x1 - body_x0) * 0.18):
            fragment_count += 1
        if previous_y1 > int(item.get("y0", 0)) + pitch * 0.16:
            overlap_count += 1
        previous_y1 = max(previous_y1, int(item.get("y1", 0)))
    diagnostics["alignment_fragment_box_count"] = int(fragment_count)
    diagnostics["alignment_overlap_count"] = int(overlap_count)

    boundaries, boundary_method = _ocr_locked_vertical_boundaries(
        row_ink, top=top, bottom=bottom, count=count,
    )
    pitch = max(1.0, (boundaries[-1] - boundaries[0]) / max(1, count))
    diagnostics["alignment_boundary_method"] = boundary_method
    diagnostics["alignment_pitch"] = round(float(pitch), 3)
    diagnostics["alignment_threshold"] = int(threshold)

    # Align anchor characters only for metadata.  Their raw geometry is not used
    # directly because a detector box may describe one stroke/radical.
    anchor_mapping: list[int | None] = []
    if anchor_text:
        anchor_mapping = _align_anchor_sequence(anchor_text, "".join(target))
    elif parsed:
        # Geometry is never copied from these boxes, but a monotonic rank map
        # records which target positions had any detector support.  This avoids
        # falsely labelling every OCR-locked cell as a synthetic补框 when Apple
        # returned boxes without usable anchor text.
        anchor_mapping = [None] * count
        if count == 1:
            anchor_mapping[0] = 0
        else:
            used: set[int] = set()
            for target_index in range(count):
                ideal = target_index * (len(parsed) - 1) / max(1, count - 1)
                for candidate in sorted(
                    range(len(parsed)), key=lambda index: (abs(index - ideal), index),
                ):
                    if candidate not in used:
                        anchor_mapping[target_index] = candidate
                        used.add(candidate)
                        break

    output: list[dict] = []
    for target_index, char in enumerate(target):
        cell_y0 = int(boundaries[target_index])
        cell_y1 = int(boundaries[target_index + 1])
        x0, y0, x1, y1, ink = _review_cell_box(
            image,
            binary,
            body_x0=body_x0,
            body_x1=body_x1,
            cell_y0=cell_y0,
            cell_y1=cell_y1,
            pitch=pitch,
        )
        source_index = None
        if target_index < len(anchor_mapping):
            source_index = anchor_mapping[target_index]
        item = {
            "index": target_index,
            "text_index": target_index,
            "target_char": char,
            "anchor_text": char,
            "x0": int(x0),
            "x1": int(x1),
            "y0": int(y0),
            "y1": int(y1),
            "cell_y0": int(cell_y0),
            "cell_y1": int(cell_y1),
            "ink_pixels": int(ink),
            "source": "ocr_locked_valley_cell",
            "geometry_source": boundary_method,
            "synthetic": not (source_index is not None and 0 <= source_index < len(parsed)),
        }
        if source_index is not None and 0 <= source_index < len(parsed):
            source_box = parsed[source_index]
            item["anchor_confidence"] = float(source_box.get("anchor_confidence", 0.0) or 0.0)
            item["detector_source"] = str(source_box.get("source", "") or "")
        output.append(item)

    diagnostics["alignment_synthetic_count"] = sum(bool(item.get("synthetic")) for item in output)
    diagnostics["alignment_source"] = "ocr_locked_valley_cells"
    diagnostics["alignment_exact"] = len(output) == len(target)
    diagnostics["alignment_rebuilt"] = True
    diagnostics["alignment_rebuilt_for_fragmentation"] = bool(
        fragment_count or overlap_count or len(parsed) != len(target)
    )
    return output, diagnostics



def _reconcile_review_text_to_physical_slots(
    text: str,
    boxes: list[dict],
) -> tuple[str, dict]:
    """Map OCR text to independently detected physical slots.

    Extra physical slots become visible ``□`` placeholders instead of silently
    disappearing.  Anchor labels are used when available; otherwise a monotonic
    proportional assignment preserves order without guessing a character.
    """
    value = str(text or "")
    slot_count = len(boxes)
    char_count = len(value)
    info = {
        "physical_slot_count": slot_count,
        "ocr_text_count": char_count,
        "placeholder_indices": [],
        "ocr_index_to_slot": list(range(char_count)),
        "text_reconciled_to_physical_slots": False,
    }
    if slot_count <= char_count or slot_count <= 0:
        return value, info

    anchors = [
        str(box.get("anchor_text", "") or "")[:1]
        if isinstance(box, dict) else ""
        for box in boxes
    ]
    inf = float("inf")
    dp = [[inf] * (char_count + 1) for _ in range(slot_count + 1)]
    parent: list[list[tuple[int, int, str] | None]] = [
        [None] * (char_count + 1) for _ in range(slot_count + 1)
    ]
    dp[0][0] = 0.0
    for slot in range(slot_count):
        remaining_slots = slot_count - slot
        for char_index in range(char_count + 1):
            current = dp[slot][char_index]
            if current == inf:
                continue
            remaining_chars = char_count - char_index
            # Leave a visible placeholder only when enough later slots remain.
            if remaining_slots - 1 >= remaining_chars:
                anchor_penalty = 0.62 if anchors[slot] else 0.30
                candidate = current + anchor_penalty
                if candidate < dp[slot + 1][char_index]:
                    dp[slot + 1][char_index] = candidate
                    parent[slot + 1][char_index] = (slot, char_index, "placeholder")
            if char_index < char_count:
                anchor = anchors[slot]
                char = value[char_index]
                if not anchor:
                    anchor_cost = 0.18
                elif anchor == char:
                    anchor_cost = 0.0
                else:
                    anchor_cost = 0.85
                slot_pos = slot / max(1, slot_count - 1)
                char_pos = char_index / max(1, char_count - 1)
                position_cost = abs(slot_pos - char_pos) * 0.34
                candidate = current + anchor_cost + position_cost
                if candidate < dp[slot + 1][char_index + 1]:
                    dp[slot + 1][char_index + 1] = candidate
                    parent[slot + 1][char_index + 1] = (slot, char_index, "assign")

    if dp[slot_count][char_count] == inf:
        # Deterministic proportional fallback.
        selected: dict[int, str] = {}
        used: set[int] = set()
        index_to_slot = [-1] * char_count
        for char_index, char in enumerate(value):
            ideal = char_index * (slot_count - 1) / max(1, char_count - 1)
            candidates = sorted(range(slot_count), key=lambda index: (abs(index - ideal), index))
            slot = next(index for index in candidates if index not in used)
            used.add(slot)
            selected[slot] = char
            index_to_slot[char_index] = slot
        output = "".join(selected.get(index, "□") for index in range(slot_count))
    else:
        output_slots = ["□"] * slot_count
        index_to_slot = [-1] * char_count
        slot, char_index = slot_count, char_count
        while slot > 0:
            step = parent[slot][char_index]
            if step is None:
                break
            previous_slot, previous_char, action = step
            if action == "assign":
                output_slots[slot - 1] = value[char_index - 1]
                index_to_slot[char_index - 1] = slot - 1
            slot, char_index = previous_slot, previous_char
        output = "".join(output_slots)

    placeholders = [index for index, char in enumerate(output) if char == "□"]
    info.update({
        "placeholder_indices": placeholders,
        "ocr_index_to_slot": index_to_slot,
        "text_reconciled_to_physical_slots": True,
    })
    return output, info


def prepare_review_records(
    doc: UnifiedDocument,
    *,
    crop_rect: tuple[float, float, float, float] | None,
    output_dir: str | Path,
    mask_main_band: bool = False,
    enable_character_sweep: bool = False,
) -> list[dict]:
    """Create stable column images and JSON records.

    The physical top-to-bottom character sweep is opt-in and is enabled only by
    the per-character review windows.  Background OCR/candidate preparation uses
    the existing OCR-count projection path and therefore cannot change ordinary
    OCR segmentation or text merely because review support is configured.
    """
    from adapters.apple_vision_adapter import crop_for_ocr

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    page_paths = {
        int(page.page_no): str(page.image_path)
        for page in doc.pages
        if int(page.page_no or 0) > 0 and str(page.image_path or "")
    }
    blocks = sorted(
        (block for block in doc.blocks if _is_reviewable_column(block)),
        key=lambda block: (
            _page_number(block),
            int(block.order_in_page if block.order_in_page is not None else block.reading_order or 0),
        ),
    )
    if not blocks:
        return []

    cropped_pages: dict[int, str] = {}
    records: list[dict] = []
    current_page_no = -1
    current_page_image: Image.Image | None = None
    for serial, block in enumerate(blocks, start=1):
        page_no = _page_number(block)
        original_path = page_paths.get(page_no)
        if not original_path or not Path(original_path).exists():
            continue
        if page_no not in cropped_pages:
            cropped_pages[page_no] = crop_for_ocr(
                original_path,
                crop_rect=crop_rect,
                out_dir=str(target / "fixed_pages"),
            )
        mask_info = None
        metadata = block.metadata or {}
        glyph_boxes: list[dict] = []
        # Physical slider geometry is kept separately from editor/OCR alignment.
        # The review canvas must always show the real one-glyph frames even when
        # OCR produced a different number of characters and the editable text
        # therefore needs supplemental alignment cells.
        physical_glyph_boxes: list[dict] = []
        segmentation_info: dict = {}
        try:
            # Blocks are already page-sorted. Decode each fixed page once, then
            # crop all of its physical columns from the same in-memory raster.
            if current_page_image is None or current_page_no != page_no:
                if current_page_image is not None:
                    current_page_image.close()
                with Image.open(cropped_pages[page_no]) as source:
                    current_page_image = source.convert("RGB")
                current_page_no = page_no
            column = _crop_bbox(current_page_image, block)
            if mask_main_band:
                masked, mask_info = mask_main_text_band(column)
                column.close()
                column = masked
            # Review geometry is now independent of OCR character count.  The
            # top-to-bottom ink slider determines physical glyph slots first;
            # ordinary OCR text is then mapped onto those slots.  This makes a
            # missed OCR character visible as a □ slot instead of suppressing its
            # frame, while still preserving every original OCR character.
            original_review_text = str(block.text or "")
            review_text = original_review_text
            raw_boxes = (
                metadata.get("handwriting_input_glyph_boxes")
                or metadata.get("black_ink_preview_glyph_boxes")
                or []
            )
            detector_boxes = _normalise_review_boxes(
                raw_boxes if isinstance(raw_boxes, list) else [],
                width=column.width, height=column.height,
            )
            character_anchors = list(
                metadata.get("black_ink_vision_character_anchors") or []
            )
            # Cached detector labels can still help map OCR text to physical
            # slots, but their fragmented geometry never defines a frame.
            if not character_anchors:
                character_anchors = detector_boxes

            if not enable_character_sweep:
                # Candidate screening during OCR only needs stable column crops.
                # Rebuild a monotonic OCR-count grid from weak detector hints,
                # but do not run the physical slider or create/delete OCR slots.
                if original_review_text:
                    glyph_boxes, alignment_info = align_review_glyph_boxes(
                        column, detector_boxes, original_review_text,
                    )
                else:
                    glyph_boxes, alignment_info = [], {
                        "alignment_target_count": 0,
                        "alignment_input_box_count": len(detector_boxes),
                        "alignment_synthetic_count": 0,
                        "alignment_source": "ocr_candidate_no_text",
                        "alignment_exact": True,
                        "alignment_rebuilt": False,
                    }
                review_text = original_review_text
                physical_glyph_boxes = [dict(box) for box in glyph_boxes]
                physical_count = len(physical_glyph_boxes)
                original_count = len(original_review_text)
                placeholder_indices: list[int] = []
                segmentation_info.update(alignment_info)
                segmentation_info.update({
                    "segmentation_mode": "ocr_candidate_projection",
                    "character_sweep_enabled": False,
                    "review_geometry_text_mismatch": False,
                    "review_physical_slot_count": int(physical_count),
                    "review_original_text_count": int(original_count),
                    "review_alignment_needs_macocr": False,
                    "review_candidate_anchor_suggested": False,
                    "placeholder_indices": [],
                    "ocr_index_to_slot": list(range(original_count)),
                })
            else:
                segmented_image, segments, slider_info = segment_black_ink_glyphs_slider(
                    column,
                    apply_main_band_mask=not bool(mask_main_band),
                    expected_count=(len(original_review_text) or None),
                    character_anchors=character_anchors,
                )
                slider_boxes: list[dict] = []
                try:
                    for index, segment in enumerate(segments):
                        slider_boxes.append({
                            "index": index,
                            "text_index": index,
                            "x0": int(segment.x0), "x1": int(segment.x1),
                            "y0": int(segment.y0), "y1": int(segment.y1),
                            "anchor_text": str(segment.anchor_text or ""),
                            "anchor_confidence": float(segment.anchor_confidence or 0.0),
                            "source": "vertical_ink_slider",
                            "geometry_source": "vertical_ink_slider",
                            "ink_pixels": int(segment.ink_pixels or 0),
                            "synthetic": False,
                        })
                finally:
                    for segment in segments:
                        try:
                            segment.image.close()
                        except Exception:
                            pass
                    segmented_image.close()
                slider_boxes = _normalise_review_boxes(
                    slider_boxes, width=column.width, height=column.height,
                )
                segmentation_info.update(slider_info)
                segmentation_info["character_sweep_enabled"] = True
                physical_glyph_boxes = [dict(box) for box in slider_boxes]
                physical_count = len(physical_glyph_boxes)
                original_count = len(original_review_text)

                if not slider_boxes:
                    # Last-resort compatibility fallback.  The fallback remains
                    # OCR-count locked only when the source raster has no usable
                    # black-ink runs at all.
                    fallback_image, fallback_segments, fallback_info = segment_black_ink_glyphs(
                        column,
                        apply_main_band_mask=not bool(mask_main_band),
                        expected_count=(original_count or None),
                        character_anchors=character_anchors,
                    )
                    try:
                        slider_boxes = [
                            {
                                "index": index,
                                "text_index": index,
                                "x0": int(segment.x0), "x1": int(segment.x1),
                                "y0": int(segment.y0), "y1": int(segment.y1),
                                "anchor_text": str(segment.anchor_text or ""),
                                "anchor_confidence": float(segment.anchor_confidence or 0.0),
                                "source": str(segment.segmentation_source or "projection_valley_hybrid"),
                                "geometry_source": str(segment.segmentation_source or "projection_valley_hybrid"),
                                "ink_pixels": int(segment.ink_pixels or 0),
                                "synthetic": False,
                            }
                            for index, segment in enumerate(fallback_segments)
                        ]
                    finally:
                        for segment in fallback_segments:
                            try:
                                segment.image.close()
                            except Exception:
                                pass
                        fallback_image.close()
                    slider_boxes = _normalise_review_boxes(
                        slider_boxes, width=column.width, height=column.height,
                    )
                    segmentation_info.update({
                        **fallback_info,
                        "segmentation_mode": "vertical_ink_slider_fallback",
                        "slider_fallback_used": True,
                    })
                    physical_glyph_boxes = [dict(box) for box in slider_boxes]
                    physical_count = len(physical_glyph_boxes)

                review_text, reconcile_info = _reconcile_review_text_to_physical_slots(
                    original_review_text, slider_boxes,
                )
                segmentation_info.update(reconcile_info)
                placeholder_indices = list(reconcile_info.get("placeholder_indices") or [])
                if not review_text and slider_boxes:
                    review_text = "□" * len(slider_boxes)
                    placeholder_indices = list(range(len(slider_boxes)))
                    segmentation_info["placeholder_indices"] = placeholder_indices
                    segmentation_info["text_reconciled_to_physical_slots"] = True

                if len(slider_boxes) == len(review_text):
                    glyph_boxes = []
                    for index, box in enumerate(slider_boxes):
                        item = dict(box)
                        item["index"] = index
                        item["text_index"] = index
                        item["target_char"] = review_text[index] if index < len(review_text) else "□"
                        glyph_boxes.append(item)
                    alignment_info = {
                        "alignment_target_count": len(review_text),
                        "alignment_input_box_count": len(slider_boxes),
                        "alignment_synthetic_count": 0,
                        "alignment_source": "vertical_ink_slider_slots",
                        "alignment_exact": True,
                        "alignment_rebuilt": False,
                    }
                else:
                    # OCR produced more characters than physical runs.  Preserve all
                    # OCR text and create monotonic supplemental cells only for this
                    # direction of mismatch; no character is silently dropped.
                    glyph_boxes, alignment_info = align_review_glyph_boxes(
                        column, slider_boxes, review_text,
                    )
                segmentation_info.update(alignment_info)
                geometry_mismatch = physical_count != original_count
                segmentation_info["review_geometry_text_mismatch"] = bool(geometry_mismatch)
                segmentation_info["review_physical_slot_count"] = int(physical_count)
                segmentation_info["review_original_text_count"] = int(original_count)
                # The physical black-ink slider is the authoritative geometry.
                # macOCR remains available as a manual text/anchor aid, but it must
                # not automatically replace frames merely to force them to match the
                # OCR string length (that was the source of left/middle/right drift).
                segmentation_info["review_alignment_needs_macocr"] = bool(not glyph_boxes)
                segmentation_info["review_candidate_anchor_suggested"] = bool(
                    geometry_mismatch
                    or placeholder_indices
                    or alignment_info.get("alignment_synthetic_count", 0)
                )

            filename = f"column_{serial:06d}.png"
            column.save(target / filename, format="PNG", compress_level=3)
            column.close()
        except Exception:
            try:
                column.close()
            except Exception:
                pass
            continue

        # Store geometry in the document metadata as a cache only; OCR text and
        # reading order remain untouched.
        if enable_character_sweep and (glyph_boxes or physical_glyph_boxes):
            metadata = dict(metadata)
            metadata["review_character_sweep_glyph_boxes"] = glyph_boxes
            metadata["review_character_sweep_physical_boxes"] = physical_glyph_boxes
            metadata["review_character_sweep_segmentation"] = segmentation_info
            block.metadata = metadata
        index_map = [
            int(value) for value in (segmentation_info.get("ocr_index_to_slot") or [])
            if isinstance(value, (int, float)) or str(value).lstrip("-").isdigit()
        ]

        def remap_ocr_index(value) -> int:
            try:
                index = int(value)
            except (TypeError, ValueError):
                return 0
            if 0 <= index < len(index_map) and index_map[index] >= 0:
                return int(index_map[index])
            return max(0, index)

        preview = metadata.get("handwriting_input_auto_preview") or []
        candidate_preview = []
        if isinstance(preview, list):
            for item in preview[:128]:
                if not isinstance(item, dict):
                    continue
                candidate_preview.append({
                    "index": remap_ocr_index(item.get("i", 0)),
                    "ocr": str(item.get("ocr", "") or ""),
                    "candidate": str(item.get("out", "") or ""),
                    "score": float(item.get("s", 0.0) or 0.0),
                    "candidates": list(item.get("c") or []),
                    "ambiguous": bool(item.get("amb", False)),
                    "reason": str(item.get("r", "") or ""),
                })
        risk_score = int(metadata.get("ocr_review_risk_score", 0) or 0)
        risk_reasons = [str(value) for value in (metadata.get("ocr_review_reasons") or []) if str(value)]
        risk_indices = {
            remap_ocr_index(value)
            for value in (metadata.get("ocr_review_indices") or [])
            if isinstance(value, (int, float)) or str(value).lstrip("-").isdigit()
        }
        geometry_mismatch = bool(segmentation_info.get("review_geometry_text_mismatch", False))
        physical_slots = int(segmentation_info.get("review_physical_slot_count", 0) or 0)
        original_chars = int(segmentation_info.get("review_original_text_count", len(original_review_text)) or 0)
        placeholder_indices = [int(value) for value in (segmentation_info.get("placeholder_indices") or [])]
        if geometry_mismatch:
            delta = physical_slots - original_chars
            reason = (
                f"逐字推子检测到 {physical_slots} 个物理字框，OCR 只有 {original_chars} 字，疑似漏识 {delta} 字"
                if delta > 0 else
                f"逐字推子检测到 {physical_slots} 个物理字框，OCR 有 {original_chars} 字，疑似多识 {abs(delta)} 字"
            )
            if reason not in risk_reasons:
                risk_reasons.append(reason)
            risk_score = max(risk_score, 72 if delta > 0 else 48)
        if placeholder_indices:
            reason = f"存在 {len(placeholder_indices)} 个 OCR 未识别的物理字框（以 □ 显示）"
            if reason not in risk_reasons:
                risk_reasons.append(reason)
            risk_indices.update(placeholder_indices)
            risk_score = max(risk_score, 90)
        risk_score = min(100, risk_score)
        review_required = bool(
            metadata.get("ocr_review_required", False)
            or risk_score >= 25
            or geometry_mismatch
            or placeholder_indices
        )

        records.append({
            "block_id": block.id,
            "page": page_no,
            "column": int(metadata.get("column_index", 0) or 0),
            "column_id": str(metadata.get("column_id", "")),
            "text": review_text,
            "confidence": float(block.confidence or 0.0),
            "image": filename,
            "glyph_boxes": glyph_boxes,
            "physical_glyph_boxes": physical_glyph_boxes,
            "glyph_segmentation": segmentation_info,
            "character_sweep_enabled": bool(enable_character_sweep),
            "glyph_alignment_needs_macocr": bool(segmentation_info.get("review_alignment_needs_macocr", False)),
            "glyph_alignment_source": str(segmentation_info.get("alignment_source", "") or ""),
            "risk_score": risk_score,
            "risk_reasons": risk_reasons,
            "risk_indices": sorted(risk_indices),
            "review_required": review_required,
            "physical_slot_count": physical_slots,
            "original_text_count": original_chars,
            "placeholder_indices": placeholder_indices,
            "column_ocr_empty": bool(metadata.get("column_ocr_empty", False)),
            "column_requires_handwriting": bool(metadata.get("column_requires_handwriting", False)),
            "column_manual_placeholder": bool(metadata.get("column_manual_placeholder", False)),
            "candidate_preview": candidate_preview,
            "main_band_masked": bool(mask_info and mask_info.applied),
            "main_band_x0": int(mask_info.x0) if mask_info else 0,
            "main_band_x1": int(mask_info.x1) if mask_info else 0,
            "main_band_removed_ink": int(mask_info.removed_ink_pixels) if mask_info else 0,
        })
    if current_page_image is not None:
        current_page_image.close()
    return records


def apply_character_sweep_to_review_record(
    record: dict,
    image_path: str | Path,
    *,
    mask_main_band: bool = False,
) -> dict:
    """Build physical one-character frames for one visible review column only.

    This is deliberately lazy: opening the review window no longer segments all
    columns in the book, and macOCR is never invoked.  The black-ink slider is
    run only for the column currently shown to the user.  OCR text remains a
    baseline and is mapped onto the resulting physical slots; an extra physical
    slot is exposed as ``□`` rather than silently discarded.
    """
    path = Path(image_path)
    if not path.is_file():
        record["character_sweep_error"] = "列图不存在"
        return record
    current_mode = str((record.get("glyph_segmentation") or {}).get("segmentation_mode", ""))
    if record.get("character_sweep_enabled") and current_mode.startswith("vertical_ink_slider"):
        return record

    with Image.open(path) as source:
        column = source.convert("RGB")
    try:
        original_text = str(record.get("text", ""))
        detector_boxes = _normalise_review_boxes(
            list(record.get("glyph_boxes") or []), width=column.width, height=column.height,
        )
        segmented_image, segments, slider_info = segment_black_ink_glyphs_slider(
            column,
            apply_main_band_mask=not bool(mask_main_band),
            expected_count=(len(original_text) or None),
            character_anchors=detector_boxes,
        )
        slider_boxes: list[dict] = []
        try:
            for index, segment in enumerate(segments):
                slider_boxes.append({
                    "index": index,
                    "text_index": index,
                    "x0": int(segment.x0), "x1": int(segment.x1),
                    "y0": int(segment.y0), "y1": int(segment.y1),
                    "anchor_text": str(segment.anchor_text or ""),
                    "anchor_confidence": float(segment.anchor_confidence or 0.0),
                    "source": "vertical_ink_slider",
                    "geometry_source": "vertical_ink_slider",
                    "ink_pixels": int(segment.ink_pixels or 0),
                    "synthetic": False,
                })
        finally:
            for segment in segments:
                try:
                    segment.image.close()
                except Exception:
                    pass
            segmented_image.close()
        slider_boxes = _normalise_review_boxes(
            slider_boxes, width=column.width, height=column.height,
        )
        segmentation_info = dict(slider_info or {})
        segmentation_info["character_sweep_enabled"] = True

        if not slider_boxes:
            fallback_image, fallback_segments, fallback_info = segment_black_ink_glyphs(
                column,
                apply_main_band_mask=not bool(mask_main_band),
                expected_count=(len(original_text) or None),
                character_anchors=detector_boxes,
            )
            try:
                slider_boxes = [
                    {
                        "index": index,
                        "text_index": index,
                        "x0": int(segment.x0), "x1": int(segment.x1),
                        "y0": int(segment.y0), "y1": int(segment.y1),
                        "anchor_text": str(segment.anchor_text or ""),
                        "anchor_confidence": float(segment.anchor_confidence or 0.0),
                        "source": str(segment.segmentation_source or "projection_valley_hybrid"),
                        "geometry_source": str(segment.segmentation_source or "projection_valley_hybrid"),
                        "ink_pixels": int(segment.ink_pixels or 0),
                        "synthetic": False,
                    }
                    for index, segment in enumerate(fallback_segments)
                ]
            finally:
                for segment in fallback_segments:
                    try:
                        segment.image.close()
                    except Exception:
                        pass
                fallback_image.close()
            slider_boxes = _normalise_review_boxes(
                slider_boxes, width=column.width, height=column.height,
            )
            segmentation_info.update(fallback_info or {})
            segmentation_info.update({
                "segmentation_mode": "vertical_ink_slider_fallback",
                "slider_fallback_used": True,
                "character_sweep_enabled": True,
            })

        review_text, reconcile_info = _reconcile_review_text_to_physical_slots(
            original_text, slider_boxes,
        )
        segmentation_info.update(reconcile_info)
        placeholder_indices = list(reconcile_info.get("placeholder_indices") or [])
        if not review_text and slider_boxes:
            review_text = "□" * len(slider_boxes)
            placeholder_indices = list(range(len(slider_boxes)))
            segmentation_info["placeholder_indices"] = placeholder_indices
            segmentation_info["text_reconciled_to_physical_slots"] = True

        if len(slider_boxes) == len(review_text):
            glyph_boxes = []
            for index, box in enumerate(slider_boxes):
                item = dict(box)
                item["index"] = index
                item["text_index"] = index
                item["target_char"] = review_text[index] if index < len(review_text) else "□"
                glyph_boxes.append(item)
            alignment_info = {
                "alignment_target_count": len(review_text),
                "alignment_input_box_count": len(slider_boxes),
                "alignment_synthetic_count": 0,
                "alignment_source": "vertical_ink_slider_slots",
                "alignment_exact": True,
                "alignment_rebuilt": False,
            }
        else:
            glyph_boxes, alignment_info = align_review_glyph_boxes(
                column, slider_boxes, review_text,
            )
        segmentation_info.update(alignment_info)

        physical_count = len(slider_boxes)
        original_count = len(original_text)
        geometry_mismatch = physical_count != original_count
        segmentation_info.update({
            "review_geometry_text_mismatch": bool(geometry_mismatch),
            "review_physical_slot_count": int(physical_count),
            "review_original_text_count": int(original_count),
            "review_alignment_needs_macocr": False,
            "review_candidate_anchor_suggested": bool(
                geometry_mismatch or placeholder_indices
                or alignment_info.get("alignment_synthetic_count", 0)
            ),
            "lazy_current_column_sweep": True,
        })

        reasons = [
            str(value) for value in (record.get("risk_reasons") or [])
            if str(value) and not str(value).startswith("逐字推子检测到")
            and "OCR 未识别的物理字框" not in str(value)
        ]
        risk_indices = {
            int(value) for value in (record.get("risk_indices") or [])
            if isinstance(value, (int, float)) or str(value).lstrip("-").isdigit()
        }
        risk_score = int(record.get("risk_score", 0) or 0)
        if geometry_mismatch:
            delta = physical_count - original_count
            reason = (
                f"逐字推子检测到 {physical_count} 个物理字框，OCR 只有 {original_count} 字，疑似漏识 {delta} 字"
                if delta > 0 else
                f"逐字推子检测到 {physical_count} 个物理字框，OCR 有 {original_count} 字，疑似多识 {abs(delta)} 字"
            )
            reasons.append(reason)
            risk_score = max(risk_score, 72 if delta > 0 else 48)
        if placeholder_indices:
            reasons.append(f"存在 {len(placeholder_indices)} 个 OCR 未识别的物理字框（以 □ 显示）")
            risk_indices.update(int(value) for value in placeholder_indices)
            risk_score = max(risk_score, 90)

        record.update({
            "text": review_text,
            "glyph_boxes": glyph_boxes,
            "physical_glyph_boxes": [dict(box) for box in slider_boxes],
            "glyph_segmentation": segmentation_info,
            "character_sweep_enabled": True,
            "character_sweep_pending": False,
            "glyph_alignment_needs_macocr": False,
            "glyph_alignment_source": str(alignment_info.get("alignment_source", "") or ""),
            "risk_score": min(100, risk_score),
            "risk_reasons": reasons,
            "risk_indices": sorted(risk_indices),
            "review_required": bool(
                record.get("review_required") or geometry_mismatch or placeholder_indices
                or risk_score >= 25
            ),
            "physical_slot_count": int(physical_count),
            "original_text_count": int(original_count),
            "placeholder_indices": placeholder_indices,
        })
        record.pop("_display_boxes_cache_key", None)
        record.pop("_display_boxes_cache", None)
        record.pop("character_sweep_error", None)
        return record
    except Exception as exc:
        record["character_sweep_error"] = str(exc)
        record["character_sweep_pending"] = False
        return record
    finally:
        column.close()


def _append_modified_by(value: str, name: str) -> str:
    parts = [item for item in str(value or "").split(",") if item]
    if name not in parts:
        parts.append(name)
    return ",".join(parts)


def _block_type_for_text(text: str, fallback: BlockType) -> BlockType:
    stripped = str(text or "").strip()
    if stripped.startswith(("「", "『")) or stripped.endswith(("」", "』")):
        return BlockType.DIALOGUE
    if fallback == BlockType.DIALOGUE:
        return BlockType.PARAGRAPH
    return fallback


def apply_review_payload(doc: UnifiedDocument, payload: Iterable[dict]) -> tuple[int, int]:
    """Apply explicitly accepted column edits.  Returns (reviewed, changed)."""
    by_id = {block.id: block for block in doc.blocks}
    reviewed = changed = 0
    for item in payload:
        block_id = str(item.get("block_id", ""))
        block = by_id.get(block_id)
        if block is None or not _is_reviewable_column(block):
            continue
        new_text = str(item.get("text", "")).strip()
        old_text = str(block.text or "")
        explicitly_reviewed = bool(item.get("reviewed")) or new_text != old_text
        if not explicitly_reviewed:
            continue
        reviewed += 1
        metadata = dict(block.metadata or {})
        unresolved_placeholder = bool(
            metadata.get("column_ocr_empty")
            or metadata.get("column_requires_handwriting")
            or metadata.get("column_manual_placeholder")
        )
        placeholder_chars = set("□■◻◼�")
        contains_placeholder = any(ch in placeholder_chars for ch in new_text)
        replacement_is_resolved = bool(new_text.strip("□■◻◼� \t\r\n　")) and not contains_placeholder
        unresolved_placeholder = bool(unresolved_placeholder or contains_placeholder)
        metadata["handwriting_reviewed"] = True
        metadata["handwriting_review_source"] = "ocr_baseline_manual_input"
        metadata["handwriting_review_disagreements"] = []
        # The per-character window now includes the same immutable source-image
        # context as 图文对照.  Record the shared review state so the standalone
        # sentence tab can display that this text was already checked here.
        metadata.setdefault("ocr_image_text_review_original_text", old_text)
        metadata["ocr_image_text_review_checked"] = True
        metadata["ocr_image_text_review_changed"] = bool(new_text != old_text)
        metadata["ocr_image_text_review_source"] = "combined_handwriting_review"
        if unresolved_placeholder and not replacement_is_resolved:
            # A user may inspect an empty OCR column and still leave it unresolved.
            # Keep the 100-point risk and the visible placeholder; never convert a
            # mere click on “已核对” into a false successful recognition.
            metadata["ocr_review_risk_resolved"] = False
            metadata["ocr_review_risk_score"] = 100
            metadata["ocr_review_reasons"] = [
                "仍含未识别物理字框（□），需逐字输入"
                if contains_placeholder else "OCR 返回空列，仍需人工输入"
            ]
            metadata["ocr_review_indices"] = [
                index for index, char in enumerate(new_text) if char in placeholder_chars
            ] or [0]
            metadata["ocr_review_required"] = True
        else:
            metadata["ocr_review_risk_resolved"] = True
            metadata["ocr_review_risk_score"] = 0
            metadata["ocr_review_reasons"] = []
            metadata["ocr_review_indices"] = []
            metadata["ocr_review_required"] = False
            metadata["column_ocr_empty"] = False
            metadata["column_requires_handwriting"] = False
            metadata["preserve_empty_ocr_column"] = False
            metadata["column_manual_placeholder"] = False
            metadata["column_manual_recovered"] = bool(unresolved_placeholder)
        block.metadata = metadata
        if new_text == old_text:
            continue
        block.text = new_text
        if not block.ocr_raw:
            block.ocr_raw = old_text
        block.type = _block_type_for_text(new_text, block.type)
        block.modified_by = _append_modified_by(block.modified_by, "handwriting_trace_review")
        block.confidence = 1.0
        changed += 1
    # Refresh the extended column audit after manual recovery so the GUI can
    # distinguish physical-column preservation from actual text completion.
    audit = getattr(doc.metadata, "column_ocr_audit", {}) or {}
    if isinstance(audit, dict) and isinstance(audit.get("pages"), dict):
        pending_by_page: dict[int, list[str]] = {}
        for block in doc.blocks:
            if not _is_reviewable_column(block):
                continue
            metadata = block.metadata or {}
            unresolved = bool(
                metadata.get("column_ocr_empty")
                or metadata.get("column_requires_handwriting")
                or metadata.get("column_manual_placeholder")
            ) and not str(block.text or "").strip("□■◻◼� \t\r\n　")
            if unresolved:
                pending_by_page.setdefault(_page_number(block), []).append(
                    str(metadata.get("column_id", ""))
                )
        total_pending = total_text = 0
        for page_key, page_info in audit["pages"].items():
            try:
                page_no = int(page_key)
            except (TypeError, ValueError):
                page_no = 0
            pending_ids = [item for item in pending_by_page.get(page_no, []) if item]
            expected = int(page_info.get("expected", 0) or 0)
            page_info["pending_manual_ids"] = pending_ids
            page_info["pending_manual"] = len(pending_ids)
            page_info["text_recognized"] = max(0, expected - len(pending_ids))
            page_info["text_recognition_complete"] = not pending_ids
            total_pending += len(pending_ids)
            total_text += page_info["text_recognized"]
        totals = audit.setdefault("totals", {})
        totals["pending_manual"] = total_pending
        totals["text_recognized"] = total_text
        audit["manual_review_required"] = total_pending > 0
        audit["text_recognition_complete"] = total_pending == 0
        doc.metadata.column_ocr_audit = audit

    if reviewed:
        doc.add_log(
            "handwriting_trace_review",
            f"OCR + 手写人工纠错：检查 {reviewed} 列，人工确认修改 {changed} 列",
            changed,
        )
    return reviewed, changed


_HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OCR + 日语手写人工纠错</title>
<style>
:root { color-scheme: light; --ink:#1d1d1f; --muted:#6e6e73; --line:#dedee3; --blue:#0071e3; --bg:#f5f5f7; }
* { box-sizing:border-box; }
html, body { margin:0; height:100%; overflow:hidden; font-family:-apple-system,BlinkMacSystemFont,"Hiragino Sans","Yu Gothic UI","Noto Sans CJK JP",sans-serif; color:var(--ink); background:var(--bg); }
#app { height:100%; display:grid; grid-template-columns:minmax(240px, 0.85fr) minmax(350px, 1.15fr) minmax(360px, 1.2fr); gap:10px; padding:10px; }
.panel { background:white; border:1px solid var(--line); border-radius:12px; overflow:hidden; min-width:0; display:flex; flex-direction:column; }
.head { padding:10px 12px; border-bottom:1px solid var(--line); display:flex; align-items:center; gap:8px; min-height:45px; }
.head strong { font-size:14px; }
.head .meta { margin-left:auto; color:var(--muted); font-size:12px; }
.toolbar { display:flex; flex-wrap:wrap; gap:6px; padding:8px 10px; border-top:1px solid var(--line); }
button { border:1px solid #c9cbd3; background:#eceef3; border-radius:8px; padding:7px 10px; cursor:pointer; font-size:12px; }
button:hover { background:#f0f6ff; border-color:#aacbfa; }
button.primary { background:var(--blue); color:#fff; border-color:var(--blue); }
#sourceScroll { flex:1; overflow:auto; padding:10px; text-align:center; background:#fafafa; }
#sourceWrap { position:relative; display:inline-block; min-width:180px; }
#sourceImage { display:block; width:220px; height:auto; image-rendering:auto; background:white; border:1px solid #ddd; }
#sourceSelection { position:absolute; left:0; right:0; border:2px solid #ff3b30; background:rgba(255,59,48,.10); pointer-events:none; display:none; }
#canvasArea { flex:1; overflow:auto; padding:12px; text-align:center; }
#can { width:301px; height:301px; border:1px solid #bbb; touch-action:none; background-color:white; background-repeat:no-repeat; background-position:center; background-size:contain; }
.canvasHint { color:var(--muted); font-size:12px; line-height:1.45; margin:8px auto; max-width:330px; }
.candidates { padding:8px 10px; border-top:1px solid var(--line); max-height:230px; overflow:auto; }
.group { margin-bottom:7px; }
.group b { display:block; color:var(--muted); font-size:11px; margin-bottom:3px; }
.kmatch, .candidate { display:inline-flex; align-items:center; justify-content:center; min-width:34px; height:34px; padding:0 6px; margin:2px; border:1px solid #c9cbd3; border-radius:7px; background:#eceef3; color:#111; text-decoration:none; font-size:21px; cursor:pointer; }
.kmatch:hover, .candidate:hover { background:#e8f1fe; border-color:#8bbaf4; }
#editor { flex:1; display:flex; flex-direction:column; padding:10px; gap:8px; min-height:0; }
#columnText { width:100%; flex:1; min-height:180px; resize:none; border:1px solid #cfcfd5; border-radius:9px; padding:10px; font-size:18px; line-height:1.65; font-family:"Hiragino Mincho ProN","Yu Mincho",serif; }
.symbols { display:flex; flex-wrap:wrap; gap:4px; }
.symbols button { font-size:18px; min-width:36px; padding:5px 7px; }
.notice { padding:8px 10px; background:#fff8e8; border:1px solid #f1d79a; border-radius:8px; font-size:11px; line-height:1.5; color:#5f4a18; }
.status { font-size:11px; color:var(--muted); }
.risk { padding:8px 10px; border-radius:8px; font-size:12px; line-height:1.5; background:#f5f5f7; border:1px solid var(--line); }
.risk.high { background:#fff0ef; border-color:#ffb7b1; color:#8a1c13; }
.risk.medium { background:#fff8e8; border-color:#f1d79a; color:#5f4a18; }
.suggestion { display:inline-flex; gap:4px; align-items:center; margin:2px 4px 2px 0; padding:4px 7px; border:1px solid #c9cbd3; border-radius:7px; background:#eceef3; cursor:pointer; }
.suggestion small { color:var(--muted); }
.hidden { display:none !important; }
</style>
<script>
if (!CanvasRenderingContext2D.prototype.reset) {
  CanvasRenderingContext2D.prototype.reset = function(){ this.setTransform(1,0,0,1,0,0); this.clearRect(0,0,this.canvas.width,this.canvas.height); };
}
</script>
<script src="jlect-jhr.compressed.js"></script>
</head>
<body>
<div id="app">
  <section class="panel">
    <div class="head"><strong>① OCR 原图定位</strong><span id="columnMeta" class="meta"></span></div>
    <div id="sourceScroll"><div id="sourceWrap"><img id="sourceImage" alt="column"><div id="sourceSelection"></div></div></div>
    <div class="toolbar">
      <button id="prevColumn">← 上一列</button><button id="nextColumn">下一列 →</button>
      <button id="prevIssue">⚠ 上一疑点</button><button class="primary" id="nextIssue">下一疑点 ⚠</button>
      <button id="prevGlyph">↑ 上一字</button><button id="nextGlyph">下一字 ↓</button>
    </div>
  </section>

  <section class="panel">
    <div class="head"><strong>② 分笔描摹 / 候选</strong><span id="glyphMeta" class="meta"></span></div>
    <div id="canvasArea">
      <canvas id="can" width="301" height="301"></canvas>
      <div class="canvasHint">在左侧点击目标字。画布会把原印刷字放到底层；可沿字形分笔描摹，也可点“自动描摹候选”。候选只供参考，绝不会静默覆盖 OCR。</div>
      <div class="toolbar" style="justify-content:center;border:0;padding:3px">
        <button class="primary" id="autoTrace">自动描摹候选</button>
        <button id="jhr-clear">清空画笔</button><button id="jhr-undo">撤销一笔</button>
      </div>
    </div>
    <div class="candidates">
      <div class="group"><b>近似候选</b><div id="jhr-guess"></div></div>
      <div class="group"><b>同笔画数候选</b><div id="jhr-slength"></div></div>
      <div class="group"><b>模糊候选</b><div id="jhr-fuzzy"></div></div>
      <div class="group"><b>包含相同笔画</b><div id="jhr-similarity"></div></div>
      <div class="group"><b>错误笔顺候选</b><div id="jhr-wrongorder"></div></div>
      <span id="jhr-angles" class="hidden"></span><span id="jhr-direction" class="hidden"></span>
      <span id="jhr-overlap" class="hidden"></span><span id="jhr-saver" class="hidden"></span>
    </div>
  </section>

  <section class="panel">
    <div class="head"><strong>③ 修改 OCR 底稿</strong><span id="editMeta" class="meta"></span></div>
    <div id="editor">
      <div class="notice">当前文本来自你选择的普通 OCR（推荐 NDL OCR）并保持原样。程序只标记疑点，不自动改字。点击左侧原图定位后，可用描摹候选、键盘、macOS 日语输入源或系统手写输入进行替换、插入和删除。</div>
      <div id="riskBox" class="risk"></div>
      <div><b style="font-size:11px;color:#6e6e73">预计算候选冲突</b><div id="preCandidates"></div></div>
      <textarea id="columnText" lang="ja" spellcheck="false"></textarea>
      <div><b style="font-size:11px;color:#6e6e73">常用符号</b><div id="symbols" class="symbols"></div></div>
      <div class="toolbar" style="padding:0;border:0">
        <button class="primary" id="focusIme">⌨ 使用 macOS 日语输入法</button>
        <button id="insertMode">当前：替换单字</button>
        <button id="deleteGlyph">删除当前字</button>
        <button id="restoreColumn">恢复本列 OCR</button>
        <button id="markReviewed">标记本列已核对</button>
      </div>
      <div id="status" class="status"></div>
    </div>
  </section>
</div>
<script>
const REVIEW_DATA = __REVIEW_DATA__;
const state = { index:0, charIndex:0, insert:false, reviewed:new Set(), originals:REVIEW_DATA.map(x=>x.text), columnImages:[], issueIndices:REVIEW_DATA.map((x,i)=>x.review_required?i:-1).filter(i=>i>=0) };
const sourceImage = document.getElementById('sourceImage');
const sourceSelection = document.getElementById('sourceSelection');
const columnText = document.getElementById('columnText');
const canvas = document.getElementById('can');
const statusEl = document.getElementById('status');
const riskBox = document.getElementById('riskBox');
const preCandidates = document.getElementById('preCandidates');
const symbols = ['。','、','！','？','「','」','『','』','（','）','…','‥','―','—','・','：','；','〜','ー'];

function saveCurrent(){ if(!REVIEW_DATA.length) return; REVIEW_DATA[state.index].text = columnText.value; }
function current(){ return REVIEW_DATA[state.index] || null; }
function charCount(){ return Array.from(columnText.value).length; }
function chars(){ return Array.from(columnText.value); }
function setChars(arr){ columnText.value = arr.join(''); }
function selectTextChar(){
  const arr = chars(); if(!arr.length){ columnText.setSelectionRange(0,0); return; }
  state.charIndex = Math.max(0, Math.min(state.charIndex, arr.length-1));
  let start=0; for(let i=0;i<state.charIndex;i++) start += arr[i].length;
  columnText.focus(); columnText.setSelectionRange(start,start+arr[state.charIndex].length);
  document.getElementById('glyphMeta').textContent = `第 ${state.charIndex+1}/${arr.length || 1} 字`;
}
function replaceSelected(value){
  const arr = chars();
  if(state.insert || !arr.length){ arr.splice(Math.max(0,Math.min(state.charIndex,arr.length)),0,value); }
  else { state.charIndex=Math.max(0,Math.min(state.charIndex,arr.length-1)); arr[state.charIndex]=value; }
  setChars(arr); saveCurrent(); selectTextChar();
  statusEl.textContent = `已写入「${value}」；本列尚需点击“标记已核对”或继续检查。`;
}
function currentGlyphBox(){
  const item=current(); const boxes=Array.isArray(item?.glyph_boxes)?item.glyph_boxes:[];
  if(!boxes.length) return null;
  const charsN=Math.max(1,charCount());
  const mapped=boxes.length===charsN
    ? Math.min(state.charIndex,boxes.length-1)
    : Math.round(Math.min(state.charIndex,charsN-1)*(boxes.length-1)/Math.max(1,charsN-1));
  const box=boxes[Math.max(0,Math.min(boxes.length-1,mapped))];
  if(!box) return null;
  const x0=Number(box.x0),x1=Number(box.x1),y0=Number(box.y0),y1=Number(box.y1);
  return [x0,y0,x1,y1].every(Number.isFinite)&&x1>x0&&y1>y0?{x0,x1,y0,y1}:null;
}
function currentCenterY(){
  const box=currentGlyphBox(); if(box) return (box.y0+box.y1)/2;
  const img=sourceImage; const n=Math.max(1,charCount());
  return img.naturalHeight * ((Math.min(state.charIndex,n-1)+0.5)/n);
}
function updateSelection(centerY){
  if(!sourceImage.naturalWidth || !sourceImage.naturalHeight) return;
  const box=currentGlyphBox();
  const fallbackH=Math.min(sourceImage.naturalHeight, Math.max(sourceImage.naturalWidth*1.35, 28));
  const cropH=Math.min(sourceImage.naturalHeight, Math.max(8, box?(box.y1-box.y0):fallbackH));
  const y=Math.max(0,Math.min(sourceImage.naturalHeight-cropH,box?box.y0:centerY-cropH/2));
  const scale=sourceImage.clientHeight/sourceImage.naturalHeight;
  const scaleX=sourceImage.clientWidth/sourceImage.naturalWidth;
  sourceSelection.style.left=`${(box?box.x0:0)*scaleX}px`;
  sourceSelection.style.right='auto';
  sourceSelection.style.width=`${(box?Math.max(4,box.x1-box.x0):sourceImage.naturalWidth)*scaleX}px`;
  sourceSelection.style.top=`${y*scale}px`; sourceSelection.style.height=`${cropH*scale}px`; sourceSelection.style.display='block';
  setGlyphBackground(y,cropH,box);
  selectTextChar();
  const displayY=(y+cropH/2)*scale-sourceImage.parentElement.parentElement.clientHeight/2;
  sourceImage.parentElement.parentElement.scrollTop=Math.max(0,displayY);
}
function setGlyphBackground(y,h,box=null){
  const off=document.createElement('canvas'); off.width=301; off.height=301;
  const c=off.getContext('2d'); c.fillStyle='#fff'; c.fillRect(0,0,301,301);
  const pad=18, usable=301-pad*2;
  const sx=box?Math.max(0,box.x0):0, sw=box?Math.max(1,box.x1-box.x0):sourceImage.naturalWidth;
  c.drawImage(sourceImage,sx,y,sw,h,pad,pad,usable,usable);
  canvas.style.backgroundImage=`url(${off.toDataURL('image/png')})`;
  canvas.dataset.cropY=String(y); canvas.dataset.cropH=String(h);
}
function renderRisk(item){
  const score=Number(item.risk_score||0), reasons=Array.isArray(item.risk_reasons)?item.risk_reasons:[];
  riskBox.className='risk '+(score>=60?'high':(score>=25?'medium':''));
  riskBox.innerHTML=score>=25?`<b>疑点 ${score}/100</b><br>${reasons.map(x=>'• '+x).join('<br>')}`:'<b>未发现明显结构性疑点</b><br>仍可按原图人工抽查。';
  preCandidates.innerHTML='';
  const rows=(item.candidate_preview||[]).filter(x=>x.candidate && x.candidate!==x.ocr && Number(x.score||0)>=.72);
  for(const row of rows.slice(0,12)){
    const b=document.createElement('button');b.className='suggestion';
    b.innerHTML=`<span>${row.ocr||'∅'} → <b>${row.candidate}</b></span><small>${Math.round(Number(row.score||0)*100)}%</small>`;
    b.onclick=()=>{state.charIndex=Math.max(0,Number(row.index||0));updateSelection(currentCenterY());replaceSelected(row.candidate);};
    preCandidates.appendChild(b);
  }
  if(!rows.length) preCandidates.textContent='无高置信冲突候选；可手动描摹或直接输入。';
}
function loadColumn(index){
  saveCurrent(); if(!REVIEW_DATA.length) return;
  state.index=(index+REVIEW_DATA.length)%REVIEW_DATA.length; const item=current();
  columnText.value=item.text;
  const risky=Array.isArray(item.risk_indices)?item.risk_indices:[];
  state.charIndex=risky.length?Math.max(0,Number(risky[0]||0)):0;
  sourceSelection.style.display='none'; canvas.style.backgroundImage='none';
  if(typeof erase==='function') try{erase();}catch(e){}
  sourceImage.onload=()=>updateSelection(currentCenterY());
  sourceImage.src=item.image;
  const issuePos=state.issueIndices.indexOf(state.index);
  document.getElementById('columnMeta').textContent=`第 ${item.page} 页 · 右起第 ${item.column} 列`;
  document.getElementById('editMeta').textContent=`${state.index+1}/${REVIEW_DATA.length}${issuePos>=0?' · 疑点 '+(issuePos+1)+'/'+state.issueIndices.length:''}`;
  renderRisk(item);
  statusEl.textContent=state.reviewed.has(item.block_id)?'本列已标记核对。':(item.review_required?'已定位到疑点列，请对照原图确认。':'当前为 OCR 底稿，尚未标记核对。');
}
function jumpIssue(delta){
  if(!state.issueIndices.length){statusEl.textContent='当前没有自动标记的疑点列。';return;}
  let pos=state.issueIndices.indexOf(state.index);
  if(pos<0) pos=delta>0?-1:0;
  pos=(pos+delta+state.issueIndices.length)%state.issueIndices.length;
  loadColumn(state.issueIndices[pos]);
}
sourceImage.addEventListener('click',ev=>{
  const r=sourceImage.getBoundingClientRect(); const naturalY=(ev.clientY-r.top)/r.height*sourceImage.naturalHeight;
  const n=Math.max(1,charCount()); state.charIndex=Math.max(0,Math.min(n-1,Math.floor(naturalY/sourceImage.naturalHeight*n)));
  updateSelection(naturalY);
});
columnText.addEventListener('input',saveCurrent);
columnText.addEventListener('click',()=>{
  const pos=columnText.selectionStart; const value=columnText.value.slice(0,pos); state.charIndex=Array.from(value).length;
  updateSelection(currentCenterY());
});
document.addEventListener('click',ev=>{
  const target=ev.target;
  if(target && target.classList && target.classList.contains('kmatch')){ ev.preventDefault(); replaceSelected(target.textContent.trim()); }
});
for(const s of symbols){ const b=document.createElement('button'); b.textContent=s; b.addEventListener('click',()=>replaceSelected(s)); document.getElementById('symbols').appendChild(b); }
document.getElementById('prevColumn').onclick=()=>loadColumn(state.index-1);
document.getElementById('nextColumn').onclick=()=>loadColumn(state.index+1);
document.getElementById('prevIssue').onclick=()=>jumpIssue(-1);
document.getElementById('nextIssue').onclick=()=>jumpIssue(1);
document.getElementById('prevGlyph').onclick=()=>{state.charIndex=Math.max(0,state.charIndex-1);updateSelection(currentCenterY());};
document.getElementById('nextGlyph').onclick=()=>{state.charIndex=Math.min(Math.max(0,charCount()-1),state.charIndex+1);updateSelection(currentCenterY());};
document.getElementById('focusIme').onclick=()=>{
  selectTextChar();
  columnText.focus();
  statusEl.textContent='输入框已聚焦：现在可直接使用 macOS 日语输入法或系统手写输入，提交内容会作为本列最终结果。';
};
document.getElementById('insertMode').onclick=ev=>{state.insert=!state.insert;ev.target.textContent=state.insert?'当前：插入新字':'当前：替换单字';};
document.getElementById('deleteGlyph').onclick=()=>{const arr=chars();if(arr.length){arr.splice(Math.max(0,Math.min(state.charIndex,arr.length-1)),1);setChars(arr);saveCurrent();state.charIndex=Math.min(state.charIndex,Math.max(0,arr.length-1));updateSelection(currentCenterY());statusEl.textContent='已删除当前字；请继续核对。';}};
document.getElementById('restoreColumn').onclick=()=>{columnText.value=state.originals[state.index];saveCurrent();state.charIndex=0;updateSelection(currentCenterY());};
document.getElementById('markReviewed').onclick=()=>{saveCurrent();state.reviewed.add(current().block_id);statusEl.textContent='本列已标记核对。';};

function otsu(gray){
  const hist=new Array(256).fill(0); for(const v of gray) hist[v]++;
  let total=gray.length,sum=0; for(let i=0;i<256;i++) sum+=i*hist[i];
  let wB=0,sumB=0,best=0,max=-1;
  for(let t=0;t<256;t++){wB+=hist[t];if(!wB)continue;const wF=total-wB;if(!wF)break;sumB+=t*hist[t];const mB=sumB/wB,mF=(sum-sumB)/wF,v=wB*wF*(mB-mF)*(mB-mF);if(v>max){max=v;best=t;}}
  return Math.min(235,Math.max(70,best+8));
}
function thinZhangSuen(img,w,h){
  const at=(x,y)=>img[y*w+x]; let changed=true,round=0;
  while(changed && round++<80){ changed=false; let remove=[];
    for(let pass=0;pass<2;pass++){ remove=[];
      for(let y=1;y<h-1;y++)for(let x=1;x<w-1;x++)if(at(x,y)){
        const p=[at(x,y-1),at(x+1,y-1),at(x+1,y),at(x+1,y+1),at(x,y+1),at(x-1,y+1),at(x-1,y),at(x-1,y-1)];
        const n=p.reduce((a,b)=>a+b,0); if(n<2||n>6)continue;
        let trans=0;for(let i=0;i<8;i++)if(!p[i]&&p[(i+1)%8])trans++;if(trans!==1)continue;
        if(pass===0){if(p[0]*p[2]*p[4]||p[2]*p[4]*p[6])continue;}else{if(p[0]*p[2]*p[6]||p[0]*p[4]*p[6])continue;}
        remove.push(y*w+x);
      }
      if(remove.length){changed=true;for(const i of remove)img[i]=0;}
    }
  }
  return img;
}
function skeletonPaths(binary,w,h){
  const key=(x,y)=>y*w+x, xy=i=>[i%w,Math.floor(i/w)];
  const nbr=i=>{const [x,y]=xy(i),o=[];for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){if(!dx&&!dy)continue;const nx=x+dx,ny=y+dy;if(nx>=0&&ny>=0&&nx<w&&ny<h&&binary[key(nx,ny)])o.push(key(nx,ny));}return o;};
  const pixels=[];for(let i=0;i<binary.length;i++)if(binary[i])pixels.push(i);
  const nodes=new Set(pixels.filter(i=>nbr(i).length!==2)); const used=new Set(),paths=[];
  const ekey=(a,b)=>a<b?`${a}:${b}`:`${b}:${a}`;
  function trace(start,next){const p=[start], seen=new Set([start]);let prev=start,cur=next;used.add(ekey(prev,cur));
    while(true){p.push(cur);if(nodes.has(cur)&&cur!==start)break;const choices=nbr(cur).filter(n=>n!==prev&&!used.has(ekey(cur,n)));if(!choices.length)break;let n=choices[0];prev=cur;cur=n;used.add(ekey(prev,cur));if(seen.has(cur))break;seen.add(cur);}return p;}
  for(const n of nodes)for(const m of nbr(n))if(!used.has(ekey(n,m)))paths.push(trace(n,m));
  for(const p of pixels)for(const m of nbr(p))if(!used.has(ekey(p,m)))paths.push(trace(p,m));
  return paths.map(path=>path.map(xy)).filter(p=>p.length>=4);
}
function simplify(points,eps){
  if(points.length<3)return points; const a=points[0],b=points[points.length-1];let max=0,idx=0;
  const dx=b[0]-a[0],dy=b[1]-a[1],den=Math.hypot(dx,dy)||1;
  for(let i=1;i<points.length-1;i++){const d=Math.abs(dy*points[i][0]-dx*points[i][1]+b[0]*a[1]-b[1]*a[0])/den;if(d>max){max=d;idx=i;}}
  if(max>eps){const l=simplify(points.slice(0,idx+1),eps),r=simplify(points.slice(idx),eps);return l.slice(0,-1).concat(r);}return [a,b];
}
function dispatchStroke(points,w,h){
  const rect=canvas.getBoundingClientRect(),pad=22,sx=(canvas.width-pad*2)/w,sy=(canvas.height-pad*2)/h;
  const send=(type,p)=>canvas.dispatchEvent(new MouseEvent(type,{bubbles:true,clientX:rect.left+(pad+p[0]*sx)*(rect.width/canvas.width),clientY:rect.top+(pad+p[1]*sy)*(rect.height/canvas.height),buttons:type==='mouseup'?0:1}));
  if(!points.length)return;send('mousedown',points[0]);for(let i=1;i<points.length;i++)send('mousemove',points[i]);send('mouseup',points[points.length-1]);
}
function autoTrace(){
  if(!sourceImage.naturalWidth)return; if(typeof erase==='function')try{erase();}catch(e){}
  const size=96,off=document.createElement('canvas');off.width=size;off.height=size;const c=off.getContext('2d',{willReadFrequently:true});c.fillStyle='#fff';c.fillRect(0,0,size,size);
  const y=parseFloat(canvas.dataset.cropY||'0'),h=parseFloat(canvas.dataset.cropH||String(sourceImage.naturalHeight));c.drawImage(sourceImage,0,y,sourceImage.naturalWidth,h,0,0,size,size);
  const data=c.getImageData(0,0,size,size).data,gray=[];for(let i=0;i<data.length;i+=4)gray.push(Math.round(.299*data[i]+.587*data[i+1]+.114*data[i+2]));const th=otsu(gray),bin=new Uint8Array(size*size);for(let i=0;i<gray.length;i++)bin[i]=gray[i]<th?1:0;
  thinZhangSuen(bin,size,size);let paths=skeletonPaths(bin,size,size).map(p=>simplify(p,1.6));paths.sort((a,b)=>Math.min(...a.map(p=>p[1]))-Math.min(...b.map(p=>p[1]))||Math.min(...a.map(p=>p[0]))-Math.min(...b.map(p=>p[0])));paths=paths.filter(p=>p.length>=2).slice(0,32);
  for(const p of paths)dispatchStroke(p,size,size);statusEl.textContent=`自动描摹生成 ${paths.length} 笔，请从候选中人工选择；未直接改动文本。`;
}
document.getElementById('autoTrace').onclick=autoTrace;
window.exportReview=()=>{saveCurrent();return REVIEW_DATA.map(x=>({block_id:x.block_id,text:x.text,reviewed:state.reviewed.has(x.block_id)}));};
window.addEventListener('load',()=>loadColumn(state.issueIndices.length?state.issueIndices[0]:0));
</script>
</body></html>'''


def build_review_html(records: list[dict]) -> str:
    safe_records = []
    for record in records:
        item = clean_json_value(dict(record))
        item["image"] = Path(clean_text(item.get("image", ""))).name
        safe_records.append(item)
    data = safe_json_dumps(safe_records, ensure_ascii=False).replace("</", "<\\/")
    return _HTML_TEMPLATE.replace("__REVIEW_DATA__", data)


class HandwritingTraceReviewDialog:
    """Small wrapper that imports QtWebEngine only when the mode is used."""

    def __init__(
        self,
        parent,
        doc: UnifiedDocument,
        *,
        crop_rect: tuple[float, float, float, float] | None,
        mask_main_band: bool = True,
    ):
        from PySide6.QtCore import QUrl
        from PySide6.QtWidgets import (
            QDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout,
        )
        try:
            from PySide6.QtWebEngineCore import QWebEngineSettings
            from PySide6.QtWebEngineWidgets import QWebEngineView
        except Exception as exc:  # pragma: no cover - depends on local Qt wheel
            raise RuntimeError(
                "日语手写描摹复核需要 PySide6 的 QtWebEngine 组件。请使用 requirements.txt 安装完整 PySide6。"
            ) from exc

        self._dialog = QDialog(parent)
        self._dialog.setWindowTitle("OCR + 日语手写人工纠错 · 疑点优先")
        self._dialog.resize(1320, 860)
        self._payload: list[dict] | None = None
        self._temp = tempfile.TemporaryDirectory(prefix="novel_formatter_handwriting_review_")
        root = Path(self._temp.name)
        records = prepare_review_records(
            doc,
            crop_rect=crop_rect,
            output_dir=root,
            mask_main_band=bool(mask_main_band),
            enable_character_sweep=True,
        )
        if not records:
            self._temp.cleanup()
            raise RuntimeError("没有找到可供人工纠错的固定区域物理列。请先启用分列 OCR。")
        if not _JLECT_JS.exists():
            self._temp.cleanup()
            raise FileNotFoundError(f"缺少手写候选引擎：{_JLECT_JS}")
        shutil.copy2(_JLECT_JS, root / _JLECT_JS.name)
        if _JLECT_LICENSE.exists():
            shutil.copy2(_JLECT_LICENSE, root / "JLECT_LICENSE.txt")
        (root / "review.html").write_text(build_review_html(records), encoding="utf-8")

        layout = QVBoxLayout(self._dialog)
        layout.setContentsMargins(10, 10, 10, 10)
        info = QLabel(
            "以普通 OCR 文本为底稿，按风险优先定位疑点列。左侧核对原图，中央可分笔描摹生成候选，"
            "右侧可直接用键盘、macOS 日语输入源或系统手写输入修改。候选不会自动覆盖正文；"
            "只有点击“应用复核结果”后，明确修改或标记核对的列才会写回。"
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        self._view = QWebEngineView(self._dialog)
        settings = self._view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        layout.addWidget(self._view, 1)

        buttons = QHBoxLayout()
        attribution = QLabel("候选引擎：JLect JHR（CC BY-SA 3.0）")
        attribution.setStyleSheet("color:#777;font-size:11px")
        buttons.addWidget(attribution)
        buttons.addStretch(1)
        skip = QPushButton("跳过复核，保留 OCR 底稿")
        apply_btn = QPushButton("应用人工纠错结果")
        apply_btn.setDefault(True)
        buttons.addWidget(skip)
        buttons.addWidget(apply_btn)
        layout.addLayout(buttons)
        self._apply_btn = apply_btn
        self._message_box = QMessageBox
        skip.clicked.connect(self._dialog.reject)
        apply_btn.clicked.connect(self._collect)
        self._view.loadFinished.connect(lambda ok: self._apply_btn.setEnabled(bool(ok)))
        self._apply_btn.setEnabled(False)
        self._view.setUrl(QUrl.fromLocalFile(str(root / "review.html")))

    @property
    def payload(self) -> list[dict] | None:
        return self._payload

    def _collect(self):
        self._apply_btn.setEnabled(False)
        self._view.page().runJavaScript(
            "JSON.stringify(window.exportReview ? window.exportReview() : [])",
            self._on_collected,
        )

    def _on_collected(self, value):
        try:
            if not isinstance(value, str):
                raise ValueError("网页未返回复核数据")
            payload = json.loads(value)
            if not isinstance(payload, list):
                raise ValueError("复核数据格式错误")
            self._payload = payload
            self._dialog.accept()
        except Exception as exc:
            self._apply_btn.setEnabled(True)
            self._message_box.warning(self._dialog, "无法应用复核结果", str(exc))

    def exec(self) -> bool:
        try:
            return bool(self._dialog.exec())
        finally:
            self._view.setUrl(__import__("PySide6.QtCore", fromlist=["QUrl"]).QUrl("about:blank"))
            self._temp.cleanup()


def recognize_review_glyph_candidates(
    image: Image.Image,
    *,
    recognition_backend: str = "auto",
    expected_char: str = "",
    strategy: str = "balanced",
    scope_key: str = "",
    card_factory=None,
) -> list[dict]:
    """Return explicit candidates for one printed glyph without changing OCR text.

    The stable review window calls this only for the currently selected glyph.
    ``auto`` now preserves the real backend cascade used by
    :class:`JapaneseHandwritingCard`: Apple PKStrokeRecognizer first when the
    compiled bridge reports Japanese support, then the local fallbacks.  This
    keeps interactive review consistent with the already-connected Apple bridge.
    """
    from adapters.handwriting_input_card import JapaneseHandwritingCard

    effective_backend = str(recognition_backend or "auto").lower()
    if effective_backend not in {"auto", "jlect", "apple", "openvino"}:
        effective_backend = "auto"
    factory = card_factory or JapaneseHandwritingCard
    card = factory(
        recognition_backend=effective_backend,
        vision_candidate_fusion=False,
        glyph_memory_enabled=True,
        glyph_memory_scope_key=str(scope_key or ""),
    )
    merged: dict[str, dict] = {}
    backend_notes: list[str] = []

    def add(text, score, source, reason=""):
        value = str(text or "").strip()
        if not value:
            return
        # A per-glyph candidate must never insert a whole word/line. Preserve a
        # combining/variation selector only when it belongs to the first base.
        first = getattr(card, "_first_recognized_glyph", lambda x: x[:1])(value)
        if not first or first in {"□", "■", "◻", "◼", "�"}:
            return
        item = {
            "text": first, "score": float(score or 0.0),
            "source": str(source or "candidate"), "reason": str(reason or ""),
        }
        previous = merged.get(first)
        if previous is None or item["score"] > previous["score"]:
            merged[first] = item

    try:
        threshold = {"conservative": 0.72, "balanced": 0.45, "aggressive": 0.0}.get(
            str(strategy or "balanced"), 0.45
        )
        for candidate in list(card.recognize_image_candidates(image, expected_char=expected_char) or []):
            if float(getattr(candidate, "score", 0.0) or 0.0) >= threshold:
                add(
                    getattr(candidate, "text", ""), getattr(candidate, "score", 0.0),
                    getattr(candidate, "source", "jlect"), getattr(candidate, "reason", ""),
                )
        memory_getter = getattr(card, "_memory_candidates_for_glyph", None)
        if callable(memory_getter):
            for candidate in list(memory_getter(image) or []):
                add(
                    getattr(candidate, "text", ""), getattr(candidate, "score", 0.0),
                    getattr(candidate, "source", "glyph_memory"), getattr(candidate, "reason", ""),
                )

        if effective_backend in {"auto", "apple"}:
            recognizer = getattr(card, "_recognize_with_apple", None)
            apple_instance = getattr(card, "_apple", None)
            # Test doubles and explicit Apple mode may not expose ``_apple``;
            # auto mode only calls the bridge when the card actually prepared it.
            should_call_apple = effective_backend == "apple" or apple_instance is not None or not hasattr(card, "_apple")
            if callable(recognizer) and should_call_apple:
                try:
                    apple_text = recognizer([image])
                    add(
                        apple_text, 0.995, "apple_pkstroke",
                        "Apple PKStrokeRecognizer 当前字首结果",
                    )
                except Exception as exc:
                    backend_notes.append(f"PKStrokeRecognizer：{exc}")
            elif effective_backend == "auto":
                apple_error = str(getattr(card, "_apple_error", "") or "").strip()
                if apple_error:
                    backend_notes.append(f"PKStrokeRecognizer：{apple_error}")
        if effective_backend == "openvino":
            ensure = getattr(card, "_ensure_openvino", None)
            if callable(ensure) and ensure(strict=True):
                recognizer = getattr(getattr(card, "_openvino", None), "recognize", None)
                if callable(recognizer):
                    add(recognizer(image), 0.96, "openvino_handwriting", "用户选择的当前字 OpenVINO 候选")

        result = sorted(merged.values(), key=lambda item: (-item["score"], item["source"], item["text"]))[:12]
        note = "；".join(dict.fromkeys(item for item in backend_notes if item))
        if note:
            for item in result:
                item["backend_note"] = note
        return result
    finally:
        close = getattr(card, "close", None)
        if callable(close):
            close()


class OCRManualReviewDialog:
    """Pure-PySide6 OCR correction dialog used by the stable workflow.

    The older :class:`HandwritingTraceReviewDialog` remains available as an
    optional WebEngine/JLect experiment, but this dialog is the default because
    it does not start a Chromium helper process and does not precompute
    handwriting candidates for an entire book.  macOS Japanese IME and system
    handwriting input work directly in the Qt text fields.
    """

    def __init__(
        self,
        parent,
        doc: UnifiedDocument,
        *,
        crop_rect: tuple[float, float, float, float] | None,
        mask_main_band: bool = True,
        recognition_backend: str = "auto",
        strategy: str = "balanced",
    ):
        from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
        from PySide6.QtGui import (
            QColor, QDesktopServices, QFont, QKeySequence, QPainter, QPen,
            QPixmap, QShortcut, QTextCursor,
        )
        from PySide6.QtWidgets import (
            QApplication, QComboBox, QDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
            QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QSplitter,
            QVBoxLayout, QWidget,
        )

        class ClickableImageLabel(QLabel):
            clickedRatio = Signal(float)
            doubleClicked = Signal()

            def mousePressEvent(self, event):  # noqa: N802 - Qt API
                if self.height() > 0:
                    ratio = max(0.0, min(1.0, float(event.position().y()) / float(self.height())))
                    self.clickedRatio.emit(ratio)
                super().mousePressEvent(event)

            def mouseDoubleClickEvent(self, event):  # noqa: N802 - Qt API
                self.doubleClicked.emit()
                event.accept()

        class CandidateSignals(QObject):
            finished = Signal(object)
            error = Signal(str)

        class ColumnOCRSignals(QObject):
            finished = Signal(object)
            error = Signal(object)

        self._Qt = Qt
        self._QPixmap = QPixmap
        self._QPainter = QPainter
        self._QPen = QPen
        self._QColor = QColor
        self._QTextCursor = QTextCursor
        self._QTimer = QTimer
        self._QKeySequence = QKeySequence
        self._QShortcut = QShortcut
        self._QApplication = QApplication
        self._QDesktopServices = QDesktopServices
        self._QUrl = QUrl
        self._message_box = QMessageBox
        self._CandidateSignals = CandidateSignals
        self._ColumnOCRSignals = ColumnOCRSignals
        self._recognition_backend = str(recognition_backend or "auto")
        self._candidate_strategy = str(strategy or "balanced")
        from adapters.glyph_memory_db import document_scope_key
        self._glyph_scope_key = document_scope_key(doc)
        self._candidate_generation = 0
        self._candidate_running = False
        self._column_ocr_generation = 0
        self._column_ocr_running = False
        self._column_ocr_signals = None
        self._syncing_selection = False
        self._payload: list[dict] | None = None
        self._temp = tempfile.TemporaryDirectory(prefix="novel_formatter_ocr_review_")
        self._root = Path(self._temp.name)
        self._mask_main_band = bool(mask_main_band)
        # Build only stable column crops here.  Physical one-character frames are
        # generated lazily for the visible column, avoiding an all-book image pass
        # and keeping macOCR completely outside the preview path.
        self._records = prepare_review_records(
            doc,
            crop_rect=crop_rect,
            output_dir=self._root,
            mask_main_band=self._mask_main_band,
            enable_character_sweep=False,
        )
        for record in self._records:
            record["character_sweep_pending"] = True
        if not self._records:
            self._temp.cleanup()
            raise RuntimeError("没有找到可供人工纠错的固定区域物理列。请先启用分列 OCR。")

        self._originals = [str(item.get("text", "")) for item in self._records]
        self._reviewed: set[str] = set()
        self._review_context_entries = []
        self._review_context_by_block: dict[str, list] = {}
        self._review_context_by_column: dict[str, list] = {}
        try:
            from engine.ocr_image_text_review import build_review_entries
            self._review_context_entries = build_review_entries(doc)
            for entry in self._review_context_entries:
                self._review_context_by_block.setdefault(str(entry.block_id), []).append(entry)
                for column_id in tuple(entry.column_ids or ()):
                    self._review_context_by_column.setdefault(str(column_id), []).append(entry)
        except Exception:
            # Image/text context is an optional review aid; correction itself must
            # remain available even for old projects without sentence lineage.
            self._review_context_entries = []
        self._index = 0
        self._char_index = 0
        self._source_pixmap = QPixmap()
        self._display_width = 300
        issue_indices = [i for i, item in enumerate(self._records) if item.get("review_required")]
        self._issue_indices = issue_indices
        if issue_indices:
            self._index = issue_indices[0]

        self._dialog = QDialog(parent)
        self._dialog.setWindowTitle("OCR + 日语逐字审校 · 字符扫描仅在本窗口启用")
        self._dialog.resize(1260, 820)
        self._dialog.setStyleSheet(
            "QDialog { background: #F5F5F7; }"
            "QLabel { color: #1F1F24; }"
            "QPushButton { background: #0A84FF; color: white; border: 1px solid #0067D8; border-radius: 8px; padding: 7px 12px; min-height: 22px; font-weight: 500; }"
            "QPushButton:hover { background: #0077ED; border-color: #0062CC; }"
            "QPushButton:pressed { background: #005BB8; border-color: #004B97; }"
            "QPushButton:disabled { background: #ECECF1; color: #73737B; border: 1px solid #C9C9D1; }"
            "QPushButton[role='secondary'] { background: #ECEEF3; color: #1F1F24; border: 1px solid #D6D6DE; }"
            "QPushButton[role='secondary']:hover { background: #EFEFF4; border-color: #C8C8D2; }"
            "QPushButton[role='secondary']:pressed { background: #E5E5EC; border-color: #B9B9C4; }"
            "QPushButton[role='secondary']:disabled { background: #F1F1F4; color: #8A8A92; border-color: #D7D7DE; }"
            "QPushButton[role='danger'] { background: #FFF5F4; color: #C9342F; border: 1px solid #F1B8B4; }"
            "QPushButton[role='danger']:hover { background: #FDE9E7; border-color: #EA9A95; }"
            "QPushButton[role='danger']:pressed { background: #F9DCD8; border-color: #E2847E; }"
            "QPushButton[role='danger']:disabled { background: #F5F1F1; color: #AA8D8A; border-color: #E6D8D7; }"
            "QPushButton[group='nav'] { padding: 6px 10px; min-height: 20px; }"
            "QPushButton[group='compact'] { padding: 6px 10px; min-height: 20px; }"
            "QLineEdit, QComboBox, QPlainTextEdit { background: white; color: #1F1F24; border: 1px solid #D6D6DE; border-radius: 8px; }"
            "QLineEdit { padding: 6px 10px; min-height: 22px; }"
            "QComboBox { padding: 6px 10px; min-height: 22px; }"
            "QPlainTextEdit { padding: 8px; }"
            "QLineEdit:disabled, QComboBox:disabled { background: #F4F4F6; color: #707078; border-color: #D3D3DB; }"
            "QScrollArea { background: transparent; border: none; }"
        )
        root_layout = QVBoxLayout(self._dialog)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(8)

        intro = QLabel(
            "普通 OCR 文本是底稿。字符扫描逐字框只在打开本审校窗口后启用，"
            "不会参与普通 OCR、简体中文横排 OCR 或后台候选筛查。"
            "程序只标记疑点，不会自动替换字符；OCR 三次仍为空的物理列会显示为 □。"
            "左侧始终保留对应原图，文本框支持 macOS 日语输入法和系统手写输入。"
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(
            "background:#eef6ff;border:1px solid #b9d9ff;border-radius:8px;"
            "padding:8px;color:#174a7e;"
        )
        root_layout.addWidget(intro)

        splitter = QSplitter(Qt.Orientation.Horizontal, self._dialog)
        root_layout.addWidget(splitter, 1)

        # Left: original column image and navigation.
        left = QWidget(splitter)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)
        self._column_meta = QLabel()
        self._column_meta.setStyleSheet("font-weight:600;")
        left_layout.addWidget(self._column_meta)
        self._image_label = ClickableImageLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self._image_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._image_label.setStyleSheet("background:white;border:1px solid #d8d8dc;")
        self._image_label.clickedRatio.connect(self._on_image_ratio)
        self._image_label.doubleClicked.connect(self._open_current_column_image)
        image_scroll = QScrollArea()
        image_scroll.setWidgetResizable(False)
        image_scroll.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        image_scroll.setWidget(self._image_label)
        self._image_scroll = image_scroll
        left_layout.addWidget(image_scroll, 1)
        nav = QHBoxLayout()
        prev_col = QPushButton("← 上一列")
        next_col = QPushButton("下一列 →")
        prev_issue = QPushButton("⚠ 上一疑点")
        next_issue = QPushButton("下一疑点 ⚠")
        for _btn in (prev_col, next_col, prev_issue, next_issue):
            _btn.setProperty("role", "secondary")
            _btn.setProperty("group", "nav")
        prev_col.clicked.connect(lambda: self._load_column(self._index - 1))
        next_col.clicked.connect(lambda: self._load_column(self._index + 1))
        prev_issue.clicked.connect(lambda: self._jump_issue(-1))
        next_issue.clicked.connect(lambda: self._jump_issue(1))
        nav.addWidget(prev_col)
        nav.addWidget(next_col)
        nav.addWidget(prev_issue)
        nav.addWidget(next_issue)
        left_layout.addLayout(nav)

        # Middle: selected glyph crop and risk explanation.
        middle = QWidget(splitter)
        middle_layout = QVBoxLayout(middle)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.setSpacing(8)
        title = QLabel("当前字形与疑点")
        title.setStyleSheet("font-weight:600;")
        middle_layout.addWidget(title)
        self._glyph_label = QLabel("点击左侧原图选择字符")
        self._glyph_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._glyph_label.setMinimumSize(280, 280)
        self._glyph_label.setFrameShape(QFrame.Shape.StyledPanel)
        self._glyph_label.setStyleSheet("background:white;")
        middle_layout.addWidget(self._glyph_label)
        self._glyph_index_label = QLabel("左图框 · 中间字形 · 右侧光标使用同一字符索引")
        self._glyph_index_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._glyph_index_label.setStyleSheet("color:#5f6368;font-size:11px;")
        middle_layout.addWidget(self._glyph_index_label)
        glyph_nav = QHBoxLayout()
        prev_glyph = QPushButton("↑ 上一字")
        next_glyph = QPushButton("下一字 ↓")
        for _btn in (prev_glyph, next_glyph):
            _btn.setProperty("role", "secondary")
            _btn.setProperty("group", "nav")
        prev_glyph.clicked.connect(lambda: self._move_char(-1))
        next_glyph.clicked.connect(lambda: self._move_char(1))
        glyph_nav.addWidget(prev_glyph)
        glyph_nav.addWidget(next_glyph)
        middle_layout.addLayout(glyph_nav)
        self._risk_label = QLabel()
        self._risk_label.setWordWrap(True)
        self._risk_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._risk_label.setStyleSheet(
            "background:#fff8e8;border:1px solid #f1d79a;border-radius:8px;"
            "padding:9px;color:#5f4a18;"
        )
        middle_layout.addWidget(self._risk_label)
        backend_display = (
            "auto（Apple PKStrokeRecognizer 优先）"
            if self._recognition_backend == "auto" else self._recognition_backend
        )
        self._candidate_title = QLabel(
            f"当前字候选 · 后端：{backend_display} · 策略：{self._candidate_strategy}"
        )
        candidate_title = self._candidate_title
        candidate_title.setStyleSheet("font-weight:600;")
        middle_layout.addWidget(candidate_title)
        candidate_row = QHBoxLayout()
        self._recognize_candidate_btn = QPushButton(
            "PKStroke 识别当前字" if self._recognition_backend in {"auto", "apple"}
            else "识别当前字"
        )
        self._recognize_candidate_btn.setProperty("role", "secondary")
        self._candidate_combo = QComboBox()
        self._candidate_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._apply_candidate_btn = QPushButton("采用候选")
        self._apply_candidate_btn.setProperty("group", "compact")
        self._apply_candidate_btn.setEnabled(False)
        self._recognize_candidate_btn.clicked.connect(self._recognize_current_glyph)
        self._apply_candidate_btn.clicked.connect(self._apply_selected_candidate)
        candidate_row.addWidget(self._recognize_candidate_btn)
        candidate_row.addWidget(self._candidate_combo, 1)
        candidate_row.addWidget(self._apply_candidate_btn)
        middle_layout.addLayout(candidate_row)
        self._candidate_status = QLabel("候选只针对当前框运行，不会自动覆盖正文。")
        self._candidate_status.setWordWrap(True)
        self._candidate_status.setStyleSheet("color:#6e6e73;font-size:11px;")
        middle_layout.addWidget(self._candidate_status)
        ime_hint = QLabel(
            "点击“PKStroke 识别当前字”会把当前印刷字骨架转换为 PKDrawing 笔画并调用已连接的 Apple PKStrokeRecognizer；"
            "右侧仍可直接使用 macOS 日语输入源/系统手写。候选不会后台批量覆盖正文。"
        )
        ime_hint.setWordWrap(True)
        ime_hint.setStyleSheet("color:#6e6e73;font-size:11px;")
        middle_layout.addWidget(ime_hint)
        self._pkstroke_panel_btn = QPushButton("打开 Apple PKStroke 手写板")
        self._pkstroke_panel_btn.setProperty("role", "secondary")
        self._pkstroke_panel_btn.setToolTip(
            "打开已桥接的原生 PencilKit 手写面板；可手写日语并复制识别结果回右侧输入框。"
        )
        self._pkstroke_panel_btn.clicked.connect(self._open_pkstroke_manual_panel)
        middle_layout.addWidget(self._pkstroke_panel_btn)
        middle_layout.addStretch(1)

        # Right: editable OCR text and explicit character operations.
        right = QWidget(splitter)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(7)
        right_title = QLabel("OCR 底稿（可直接编辑）")
        right_title.setStyleSheet("font-weight:600;")
        right_layout.addWidget(right_title)
        self._editor = QPlainTextEdit()
        self._editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        font = QFont("Hiragino Mincho ProN")
        font.setPointSize(16)
        self._editor.setFont(font)
        self._editor.textChanged.connect(self._on_editor_changed)
        self._editor.cursorPositionChanged.connect(self._on_cursor_changed)
        right_layout.addWidget(self._editor, 2)

        context_title = QLabel("图文对照（与当前人工纠错同步）")
        context_title.setStyleSheet("font-weight:600;")
        right_layout.addWidget(context_title)
        self._context_image_label = QLabel("正在载入当前列/句原图…")
        self._context_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._context_image_label.setStyleSheet(
            "background:white;border:1px solid #d8d8dc;border-radius:8px;padding:4px;"
        )
        self._context_image_scroll = QScrollArea()
        self._context_image_scroll.setWidgetResizable(True)
        self._context_image_scroll.setMinimumHeight(150)
        self._context_image_scroll.setWidget(self._context_image_label)
        right_layout.addWidget(self._context_image_scroll, 1)
        self._context_text = QPlainTextEdit()
        self._context_text.setReadOnly(True)
        self._context_text.setMaximumHeight(94)
        self._context_text.setPlaceholderText("显示当前列所属句子的 OCR 文本和来源信息")
        right_layout.addWidget(self._context_text)

        input_row = QHBoxLayout()
        self._char_input = QLineEdit()
        self._char_input.setPlaceholderText("输入替换字；支持日语输入法/手写")
        self._char_input.setMaxLength(16)
        self._char_input.returnPressed.connect(self._replace_char)
        replace_btn = QPushButton("替换当前字")
        replace_btn.setProperty("role", "secondary")
        insert_btn = QPushButton("在前面插入")
        insert_btn.setProperty("role", "secondary")
        delete_btn = QPushButton("删除当前字")
        delete_btn.setProperty("role", "danger")
        replace_btn.clicked.connect(self._replace_char)
        insert_btn.clicked.connect(self._insert_char)
        delete_btn.clicked.connect(self._delete_char)
        input_row.addWidget(self._char_input, 1)
        input_row.addWidget(replace_btn)
        input_row.addWidget(insert_btn)
        input_row.addWidget(delete_btn)
        right_layout.addLayout(input_row)

        macocr_row = QHBoxLayout()
        self._macocr_result = QLineEdit()
        self._macocr_result.setReadOnly(True)
        self._macocr_result.setPlaceholderText("仅在手动点击后运行 macOCR；不会后台自动识别")
        self._macocr_column_btn = QPushButton("macOCR 识别本列并复制")
        self._macocr_column_btn.setProperty("role", "secondary")
        self._macocr_apply_btn = QPushButton("用 macOCR 替换本列")
        self._macocr_apply_btn.setProperty("role", "secondary")
        self._macocr_apply_btn.setEnabled(False)
        self._open_column_btn = QPushButton("用 Mac 预览打开本列（⌘O）")
        self._open_column_btn.setProperty("role", "secondary")
        self._open_column_btn.setToolTip("双击左侧列图或按 ⌘O，在 macOS 预览中打开当前列原图。")
        self._macocr_column_btn.clicked.connect(lambda: self._start_column_macocr(copy_result=True))
        self._macocr_apply_btn.clicked.connect(self._apply_macocr_column)
        self._open_column_btn.clicked.connect(self._open_current_column_image)
        macocr_row.addWidget(self._macocr_result, 1)
        macocr_row.addWidget(self._macocr_column_btn)
        macocr_row.addWidget(self._macocr_apply_btn)
        macocr_row.addWidget(self._open_column_btn)
        right_layout.addLayout(macocr_row)

        action_row = QHBoxLayout()
        focus_ime = QPushButton("⌨ 聚焦日语输入")
        focus_ime.setProperty("role", "secondary")
        restore = QPushButton("恢复本列 OCR")
        restore.setProperty("role", "secondary")
        mark = QPushButton("标记本列已核对")
        mark.setProperty("group", "compact")
        focus_ime.clicked.connect(self._focus_ime)
        restore.clicked.connect(self._restore_column)
        mark.clicked.connect(self._mark_reviewed)
        action_row.addWidget(focus_ime)
        action_row.addWidget(restore)
        action_row.addWidget(mark)
        right_layout.addLayout(action_row)
        self._status = QLabel()
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color:#6e6e73;font-size:11px;")
        right_layout.addWidget(self._status)

        splitter.addWidget(left)
        splitter.addWidget(middle)
        splitter.addWidget(right)
        splitter.setSizes([330, 330, 560])

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        skip = QPushButton("跳过，保留 OCR 底稿")
        skip.setProperty("role", "secondary")
        apply_btn = QPushButton("应用人工纠错结果")
        apply_btn.setProperty("group", "compact")
        apply_btn.setDefault(True)
        skip.clicked.connect(self._dialog.reject)
        apply_btn.clicked.connect(self._accept)
        buttons.addWidget(skip)
        buttons.addWidget(apply_btn)
        root_layout.addLayout(buttons)

        self._open_preview_shortcut = QShortcut(QKeySequence.StandardKey.Open, self._dialog)
        self._open_preview_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self._open_preview_shortcut.activated.connect(self._open_current_column_image)

        self._loading = False
        self._load_column(self._index, save_current=False)

    @property
    def payload(self) -> list[dict] | None:
        return self._payload

    def _current(self) -> dict:
        return self._records[self._index]

    def _current_text(self) -> str:
        return self._editor.toPlainText()

    def _save_current(self) -> None:
        if not self._records or self._loading:
            return
        self._records[self._index]["text"] = self._current_text()

    def _refresh_issue_indices(self) -> None:
        self._issue_indices = [
            index for index, item in enumerate(self._records)
            if bool(item.get("review_required"))
        ]

    def _ensure_current_character_sweep(self, image_path: Path) -> None:
        item = self._current()
        if not item.get("character_sweep_pending") and item.get("character_sweep_enabled"):
            return
        before = str(item.get("text", ""))
        apply_character_sweep_to_review_record(
            item,
            image_path,
            mask_main_band=self._mask_main_band,
        )
        # Slider-generated □ slots are review UI placeholders, not user edits.
        # Treat the lazily reconciled text as this dialog's baseline until the
        # user explicitly changes or marks the column.
        if 0 <= self._index < len(self._originals):
            self._originals[self._index] = str(item.get("text", before))
        self._refresh_issue_indices()

    def _context_entry_for_current(self):
        item = self._current()
        column_id = str(item.get("column_id", "") or "")
        block_id = str(item.get("block_id", "") or "")
        candidates = list(self._review_context_by_column.get(column_id, [])) if column_id else []
        if not candidates:
            candidates = list(self._review_context_by_block.get(block_id, []))
        if not candidates:
            return None
        for entry in candidates:
            if column_id and column_id in tuple(entry.column_ids or ()):
                return entry
        return candidates[0]

    def _load_image_text_context(self) -> None:
        if not hasattr(self, "_context_image_label"):
            return
        item = self._current()
        entry = self._context_entry_for_current()
        context_path = ""
        context_text = ""
        if entry is not None:
            try:
                from engine.ocr_image_text_review import render_review_image
                context_path = render_review_image(
                    entry,
                    self._root / "image_text_context" / f"{entry.cache_key}.png",
                )
            except Exception:
                context_path = ""
            context_text = str(entry.text or "")
        if not context_path:
            context_path = str(self._current_column_path())
        pixmap = self._QPixmap(context_path)
        if pixmap.isNull():
            self._context_image_label.setPixmap(self._QPixmap())
            self._context_image_label.setText("当前列/句原图无法载入")
        else:
            target_width = max(360, self._context_image_scroll.viewport().width() - 14)
            displayed = pixmap.scaledToWidth(
                target_width,
                self._Qt.TransformationMode.SmoothTransformation,
            )
            self._context_image_label.setText("")
            self._context_image_label.setPixmap(displayed)
            self._context_image_label.setMinimumSize(displayed.size())
        label = (
            f"所属句 OCR：{context_text}\n\n当前列人工纠错：{self._current_text()}"
            if context_text else f"当前列人工纠错：{self._current_text()}"
        )
        self._context_text.setPlainText(label)

    def _load_column(self, index: int, *, save_current: bool = True) -> None:
        if save_current:
            self._save_current()
        if not self._records:
            return
        self._index = int(index) % len(self._records)
        if hasattr(self, "_candidate_combo"):
            self._clear_candidates()
        item = self._current()
        image_path = self._root / str(item.get("image", ""))
        self._ensure_current_character_sweep(image_path)
        item = self._current()
        self._loading = True
        self._editor.setPlainText(str(item.get("text", "")))
        item["_display_text_count"] = len(str(item.get("text", "")))
        risk_indices = item.get("risk_indices") or []
        try:
            self._char_index = max(0, int(risk_indices[0])) if risk_indices else 0
        except (TypeError, ValueError):
            self._char_index = 0
        self._source_pixmap = self._QPixmap(str(image_path))
        self._loading = False
        issue_pos = self._issue_indices.index(self._index) + 1 if self._index in self._issue_indices else 0
        issue_note = f" · 疑点 {issue_pos}/{len(self._issue_indices)}" if issue_pos else ""
        self._column_meta.setText(
            f"第 {item.get('page', 0)} 页 · 右起第 {item.get('column', 0)} 列 · "
            f"{self._index + 1}/{len(self._records)}{issue_note}"
        )
        self._macocr_result.setText(str(item.get("macocr_column_text", "") or ""))
        self._macocr_apply_btn.setEnabled(bool(self._macocr_result.text().strip()))
        self._render_risk()
        self._select_char()
        # macOCR is intentionally manual-only.  Automatic per-column calls caused
        # repeated Vision sessions and could make the UI appear frozen.
        self._load_image_text_context()
        if item.get("column_ocr_empty") or item.get("column_manual_placeholder"):
            self._status.setText("该列 OCR 三次均为空。请删除 □ 并在右侧输入整列原文。")
        else:
            self._status.setText(
                "已定位疑点，请对照原图确认。" if item.get("review_required")
                else "当前列未发现明显结构性疑点，可按需抽查。"
            )

    def _open_pkstroke_manual_panel(self) -> None:
        try:
            from adapters.apple_pkstroke_engine import launch_manual_test_panel
            launch_manual_test_panel(auto_build=False)
            self._candidate_status.setText(
                "已打开 Apple PKStrokeRecognizer 原生手写板；识别后可复制结果到右侧单字输入框。"
            )
        except Exception as exc:
            self._candidate_status.setText(f"Apple PKStroke 手写板不可用：{exc}")

    def _current_column_path(self) -> Path:
        return self._root / str(self._current().get("image", ""))

    def _open_current_column_image(self) -> None:
        path = self._current_column_path()
        if not path.exists():
            self._status.setText("本列原图不存在，无法打开。")
            return
        if sys.platform == "darwin":
            try:
                completed = subprocess.run(
                    ["open", "-b", "com.apple.Preview", str(path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=5.0,
                )
                if completed.returncode == 0:
                    self._status.setText(
                        "已在 macOS 预览中打开本列原图（双击左图或按 ⌘O 可再次打开）。"
                    )
                    return
            except Exception:
                pass
        if not self._QDesktopServices.openUrl(self._QUrl.fromLocalFile(str(path))):
            self._status.setText("系统未能打开本列原图。")
        else:
            self._status.setText("已使用系统图片查看器打开本列原图。")

    @staticmethod
    def _compact_column_text(value: str) -> str:
        return "".join(part.strip() for part in str(value or "").splitlines() if part.strip())

    def _start_column_macocr(self, *, copy_result: bool) -> None:
        if self._column_ocr_running:
            self._status.setText("macOCR 正在识别当前列，请稍候。")
            return
        image_path = self._current_column_path()
        if not image_path.exists():
            self._status.setText("本列原图不存在，无法运行 macOCR。")
            return
        generation = self._column_ocr_generation + 1
        self._column_ocr_generation = generation
        record_index = self._index
        target_text = self._current_text()
        self._column_ocr_running = True
        self._macocr_column_btn.setEnabled(False)
        self._status.setText("正在手动运行 macOCR 识别当前单列；逐字框不会被替换…")
        signals = self._ColumnOCRSignals()
        self._column_ocr_signals = signals

        def worker():
            backend = None
            try:
                from adapters.vision_backends.base import OCRConfig
                from adapters.vision_backends.native_helper_backend import NativeVisionHelperBackend
                backend = NativeVisionHelperBackend()
                result = backend.recognize(
                    str(image_path),
                    OCRConfig(
                        recognition_level="accurate",
                        languages=["ja-JP"],
                        vertical=True,
                        timeout=35.0,
                        use_language_correction=True,
                        automatically_detect_language=False,
                        minimum_text_height_fraction=0.001,
                        candidate_count=5,
                        orientation="up",
                        vertical_compatibility_mode=True,
                        character_boxes=True,
                    ),
                )
                with Image.open(image_path) as opened:
                    image = opened.convert("RGB")
                try:
                    width, height = image.size
                    anchor_boxes: list[dict] = []
                    anchor_chars: list[str] = []
                    for block in list(result.blocks or []):
                        text = self._compact_column_text(getattr(block, "text", ""))
                        char = next(iter(text), "")
                        bbox = getattr(block, "bbox", None)
                        if not char or bbox is None:
                            continue
                        x, y, box_w, box_h = (float(value) for value in bbox)
                        # Vision rectangles use a lower-left origin. Convert to
                        # the top-left pixel coordinates used by PIL/review UI.
                        x0 = int(round(x * width))
                        x1 = int(round((x + box_w) * width))
                        y0 = int(round((1.0 - y - box_h) * height))
                        y1 = int(round((1.0 - y) * height))
                        if x1 <= x0 or y1 <= y0:
                            continue
                        anchor_chars.append(char)
                        anchor_boxes.append({
                            "x0": x0, "x1": x1, "y0": y0, "y1": y1,
                            "anchor_text": char,
                            "anchor_confidence": float(getattr(block, "confidence", 0.0) or 0.0),
                            "source": "macocr_character_box",
                        })
                    anchor_boxes.sort(key=lambda item: (int(item["y0"]), int(item["x0"])))
                    # Keep text order synchronized with sorted geometry.
                    anchor_text = "".join(str(item.get("anchor_text", "")) for item in anchor_boxes)
                    recognized_text = self._compact_column_text(result.full_text) or anchor_text
                    aligned, alignment_info = align_review_glyph_boxes(
                        image, anchor_boxes, target_text,
                        anchor_text=anchor_text,
                    )
                finally:
                    image.close()
                signals.finished.emit({
                    "generation": generation,
                    "record_index": record_index,
                    "text": recognized_text,
                    "boxes": aligned,
                    "alignment_info": alignment_info,
                    "copy_result": bool(copy_result),
                })
            except Exception as exc:
                signals.error.emit({
                    "generation": generation,
                    "record_index": record_index,
                    "message": str(exc),
                })
            finally:
                if backend is not None:
                    try:
                        backend.close()
                    except Exception:
                        pass

        signals.finished.connect(self._column_macocr_finished)
        signals.error.connect(self._column_macocr_failed)
        threading.Thread(target=worker, daemon=True).start()

    def _column_macocr_finished(self, payload) -> None:
        if int(payload.get("generation", -1)) != self._column_ocr_generation:
            return
        self._column_ocr_running = False
        self._macocr_column_btn.setEnabled(True)
        record_index = int(payload.get("record_index", -1))
        if not (0 <= record_index < len(self._records)):
            return
        item = self._records[record_index]
        boxes = list(payload.get("boxes") or [])
        text = self._compact_column_text(payload.get("text", ""))
        alignment_info = dict(payload.get("alignment_info") or {})
        segmentation = dict(item.get("glyph_segmentation") or {})
        physical_slider = str(segmentation.get("segmentation_mode", "") or "").startswith(
            "vertical_ink_slider"
        ) and bool(item.get("glyph_boxes"))
        if boxes:
            if physical_slider:
                # Keep physical ink frames stable.  macOCR contributes text and
                # optional anchors only; forcing geometry back to OCR length
                # would reintroduce the alignment bug this workflow fixes.
                item["macocr_anchor_boxes"] = boxes
                item["glyph_alignment_needs_macocr"] = False
                item["glyph_segmentation"] = {
                    **segmentation,
                    "macocr_anchor_boxes_available": True,
                    "macocr_geometry_preserved_physical_slider": True,
                }
            else:
                item["glyph_boxes"] = boxes
                item["glyph_alignment_needs_macocr"] = False
                item["glyph_alignment_source"] = str(
                    alignment_info.get("alignment_source") or "ocr_locked_valley_cells"
                )
                item["glyph_segmentation"] = {
                    **segmentation,
                    **alignment_info,
                    "macocr_anchor_boxes_used": True,
                }
                item.pop("_display_boxes_cache_key", None)
                item.pop("_display_boxes_cache", None)
        if text:
            item["macocr_column_text"] = text
        if record_index != self._index:
            return
        self._macocr_result.setText(text)
        self._macocr_apply_btn.setEnabled(bool(text))
        if bool(payload.get("copy_result")) and text:
            self._QApplication.clipboard().setText(text)
        self._render_column_image()
        self._render_glyph_crop()
        if text:
            self._status.setText("macOCR 本列结果已显示并复制到剪贴板；可选择替换本列。")
        else:
            self._status.setText("macOCR 完成了字符框对齐，但未返回可复制文本。")

    def _column_macocr_failed(self, payload) -> None:
        if int(payload.get("generation", -1)) != self._column_ocr_generation:
            return
        self._column_ocr_running = False
        self._macocr_column_btn.setEnabled(True)
        message = str(payload.get("message") or "macOCR 未返回结果")
        record_index = int(payload.get("record_index", -1))
        if 0 <= record_index < len(self._records):
            self._records[record_index]["macocr_alignment_error"] = message
        if record_index == self._index:
            self._status.setText(f"macOCR 当前列识别失败：{message}")

    def _apply_macocr_column(self) -> None:
        text = self._macocr_result.text().strip()
        if not text:
            return
        self._editor.setPlainText(text)
        self._reviewed.add(str(self._current().get("block_id", "")))
        self._status.setText("已用 macOCR 结果替换本列；应用人工纠错结果后写回正文。")

    def _clear_candidates(self, message: str = "候选只针对当前框运行，不会自动覆盖正文。") -> None:
        self._candidate_generation += 1
        self._candidate_running = False
        self._candidate_combo.clear()
        self._apply_candidate_btn.setEnabled(False)
        self._recognize_candidate_btn.setEnabled(True)
        self._candidate_status.setText(message)

    def _current_glyph_pil(self) -> Image.Image:
        item = self._current()
        image_path = self._root / str(item.get("image", ""))
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        x0, y0, x1, y1 = self._glyph_box_for_index()
        pad_x = max(2, int(round(max(1, x1 - x0) * 0.12)))
        pad_y = max(2, int(round(max(1, y1 - y0) * 0.12)))
        crop = image.crop((
            max(0, x0 - pad_x), max(0, y0 - pad_y),
            min(image.width, x1 + pad_x), min(image.height, y1 + pad_y),
        )).convert("RGB")
        image.close()
        return crop

    def _recognize_current_glyph(self) -> None:
        if self._candidate_running:
            return
        try:
            crop = self._current_glyph_pil()
        except Exception as exc:
            self._candidate_status.setText(f"当前字形裁切失败：{exc}")
            return
        text = self._current_text()
        expected = text[self._char_index] if text and self._char_index < len(text) else ""
        generation = self._candidate_generation + 1
        self._candidate_generation = generation
        record_index = self._index
        char_index = self._char_index
        self._candidate_running = True
        self._recognize_candidate_btn.setEnabled(False)
        self._apply_candidate_btn.setEnabled(False)
        self._candidate_combo.clear()
        self._candidate_status.setText("正在识别当前字；正文保持不变…")
        signals = self._CandidateSignals()
        self._candidate_signals = signals

        def worker():
            try:
                candidates = recognize_review_glyph_candidates(
                    crop, recognition_backend=self._recognition_backend,
                    expected_char=expected, strategy=self._candidate_strategy,
                    scope_key=self._glyph_scope_key,
                )
                signals.finished.emit({
                    "generation": generation, "record_index": record_index,
                    "char_index": char_index, "candidates": candidates,
                })
            except Exception as exc:
                signals.error.emit(str(exc))
            finally:
                crop.close()

        signals.finished.connect(self._candidate_finished)
        signals.error.connect(lambda message, g=generation: self._candidate_failed(message, g))
        threading.Thread(target=worker, daemon=True).start()

    def _candidate_finished(self, payload) -> None:
        if int(payload.get("generation", -1)) != self._candidate_generation:
            return
        if int(payload.get("record_index", -1)) != self._index or int(payload.get("char_index", -1)) != self._char_index:
            return
        self._candidate_running = False
        self._recognize_candidate_btn.setEnabled(True)
        candidates = list(payload.get("candidates") or [])
        self._candidate_combo.clear()
        for item in candidates:
            text = str(item.get("text", ""))
            source = str(item.get("source", "candidate"))
            score = float(item.get("score", 0.0) or 0.0)
            self._candidate_combo.addItem(f"{text} · {score:.1%} · {source}", text)
            self._candidate_combo.setItemData(
                self._candidate_combo.count() - 1, str(item.get("reason", "")),
                self._Qt.ItemDataRole.ToolTipRole,
            )
        self._apply_candidate_btn.setEnabled(bool(candidates))
        sources = ", ".join(dict.fromkeys(str(item.get("source", "candidate")) for item in candidates))
        backend_note = next((str(item.get("backend_note", "") or "") for item in candidates if item.get("backend_note")), "")
        if candidates:
            message = f"得到 {len(candidates)} 个候选 · 来源：{sources}；选择后点击“采用候选”。"
            if backend_note:
                message += f" 备用说明：{backend_note}"
        else:
            message = "当前后端没有给出可靠单字候选，请直接人工输入。"
        self._candidate_status.setText(message)

    def _candidate_failed(self, message: str, generation: int) -> None:
        if int(generation) != self._candidate_generation:
            return
        self._candidate_running = False
        self._recognize_candidate_btn.setEnabled(True)
        self._apply_candidate_btn.setEnabled(False)
        self._candidate_status.setText(f"当前字候选失败：{message}")

    def _apply_selected_candidate(self) -> None:
        value = str(self._candidate_combo.currentData() or "")
        if not value:
            return
        self._char_input.setText(value)
        self._replace_char()
        self._clear_candidates(f"已采用候选“{value}”；请继续对照原图确认。")

    def _render_risk(self) -> None:
        item = self._current()
        score = int(item.get("risk_score", 0) or 0)
        reasons = [str(x) for x in (item.get("risk_reasons") or []) if str(x)]
        aligned_count = len(self._display_boxes()) if hasattr(self, "_editor") else len(item.get("glyph_boxes") or [])
        physical_count = len(self._physical_boxes()) if hasattr(self, "_editor") else len(item.get("physical_glyph_boxes") or [])
        text_count = len(self._current_text()) if hasattr(self, "_editor") else len(str(item.get("text", "")))
        alignment_source = str(item.get("glyph_alignment_source", "") or "")
        alignment_line = (
            f"物理逐字框：{physical_count}；OCR 编辑位：{aligned_count}/{text_count}，"
            "当前红框按物理字槽定位"
        )
        if alignment_source:
            alignment_line += f"（{alignment_source}）"
        if reasons:
            self._risk_label.setText(
                f"疑点分数：{score}/100\n" + "\n".join(f"• {r}" for r in reasons)
                + f"\n• {alignment_line}"
            )
        else:
            self._risk_label.setText(
                "未发现明显结构性疑点。手动抽查仍以原图为准。\n"
                f"• {alignment_line}"
            )

    def _char_count(self) -> int:
        return len(self._current_text())

    def _physical_boxes(self) -> list[dict]:
        """Return authoritative review-only slider frames.

        These boxes are intentionally independent from the editable OCR string.
        When OCR has one extra/missing character, the left preview must still
        draw every real physical frame instead of replacing them with an
        OCR-count grid that can merge several printed glyphs into one rectangle.
        """
        item = self._current()
        raw = item.get("physical_glyph_boxes")
        if not isinstance(raw, list) or not raw:
            raw = item.get("glyph_boxes") or []
        valid: list[dict] = []
        for box in raw:
            if not isinstance(box, dict):
                continue
            try:
                x0 = int(round(float(box.get("x0", 0))))
                x1 = int(round(float(box.get("x1", 0))))
                y0 = int(round(float(box.get("y0", 0))))
                y1 = int(round(float(box.get("y1", 0))))
            except (TypeError, ValueError):
                continue
            if x1 <= x0 or y1 <= y0:
                continue
            copied = dict(box)
            copied.update({"x0": x0, "x1": x1, "y0": y0, "y1": y1})
            valid.append(copied)
        valid.sort(key=lambda box: (
            int(box.get("y0", 0)), int(box.get("y1", 0)), int(box.get("x0", 0))
        ))
        return valid

    def _physical_box_for_char_index(self) -> dict | None:
        boxes = self._physical_boxes()
        if not boxes:
            return None
        item = self._current()
        segmentation = item.get("glyph_segmentation") or {}
        mapping = segmentation.get("ocr_index_to_slot") or []
        slot: int | None = None
        if isinstance(mapping, list) and 0 <= int(self._char_index) < len(mapping):
            try:
                mapped = int(mapping[int(self._char_index)])
                if mapped >= 0:
                    slot = mapped
            except (TypeError, ValueError):
                slot = None
        if slot is None:
            text_count = max(1, self._char_count())
            if len(boxes) == text_count:
                slot = int(self._char_index)
            else:
                # Stable monotonic fallback for OCR-more-than-physical mismatch.
                ratio = (float(self._char_index) + 0.5) / float(text_count)
                slot = int(ratio * len(boxes))
        return boxes[max(0, min(len(boxes) - 1, int(slot)))]

    def _display_boxes(self) -> list[dict]:
        """Return one current geometry slot per editor character."""
        item = self._current()
        text = self._current_text()
        raw_boxes = list(item.get("glyph_boxes") or [])
        box_signature = tuple(
            (
                int(float(box.get("x0", 0) or 0)),
                int(float(box.get("x1", 0) or 0)),
                int(float(box.get("y0", 0) or 0)),
                int(float(box.get("y1", 0) or 0)),
                str(box.get("source", "") or ""),
            )
            for box in raw_boxes
            if isinstance(box, dict)
        )
        # Geometry depends on character count, not on the actual replacement
        # characters.  Replacing one glyph with another must remain instant and
        # must not rescan the full column raster on every keystroke.
        cache_key = (len(text), box_signature)
        if item.get("_display_boxes_cache_key") == cache_key:
            cached = item.get("_display_boxes_cache") or []
            if isinstance(cached, list):
                return cached
        boxes = raw_boxes
        locked_sources = {
            "ocr_locked_valley_cell",
            "vertical_ink_slider",
            "vertical_ink_slider_slots",
        }
        already_locked = len(boxes) == len(text) and all(
            isinstance(box, dict)
            and int(box.get("text_index", index) or 0) == index
            and (
                str(box.get("source", "") or "") in locked_sources
                or str(box.get("geometry_source", "") or "") in locked_sources
            )
            for index, box in enumerate(boxes)
        )
        if already_locked:
            aligned = boxes
        else:
            image_path = self._root / str(item.get("image", ""))
            try:
                with Image.open(image_path) as source:
                    converted = source.convert("RGB")
                try:
                    aligned, info = align_review_glyph_boxes(
                        converted, boxes, text,
                    )
                    item["glyph_alignment_source"] = str(
                        info.get("alignment_source") or item.get("glyph_alignment_source", "")
                    )
                finally:
                    converted.close()
            except Exception:
                aligned = boxes
        item["_display_boxes_cache_key"] = cache_key
        item["_display_boxes_cache"] = aligned
        return aligned

    def _select_char(self) -> None:
        text = self._current_text()
        if text:
            self._char_index = max(0, min(self._char_index, len(text) - 1))
            cursor = self._editor.textCursor()
            self._syncing_selection = True
            try:
                cursor.setPosition(self._char_index)
                cursor.movePosition(
                    self._QTextCursor.MoveOperation.Right,
                    self._QTextCursor.MoveMode.KeepAnchor,
                    1,
                )
                self._editor.setTextCursor(cursor)
                self._editor.ensureCursorVisible()
            finally:
                self._syncing_selection = False
        else:
            self._char_index = 0
        self._render_column_image()
        self._render_glyph_crop()
        self._QTimer.singleShot(0, self._scroll_selected_into_view)

    def _glyph_box_for_index(self) -> tuple[int, int, int, int]:
        src_w = max(1, self._source_pixmap.width())
        src_h = max(1, self._source_pixmap.height())
        physical = self._physical_box_for_char_index()
        boxes = [physical] if physical is not None else self._display_boxes()
        if boxes:
            try:
                box = boxes[0] if physical is not None else boxes[max(0, min(len(boxes) - 1, int(self._char_index)))]
                x0 = max(0, min(src_w, int(round(float(box.get("x0", 0))))))
                x1 = max(0, min(src_w, int(round(float(box.get("x1", src_w))))))
                y0 = max(0, min(src_h, int(round(float(box.get("y0", 0))))))
                y1 = max(0, min(src_h, int(round(float(box.get("y1", src_h))))))
                if x1 > x0 and y1 > y0:
                    return x0, y0, x1, y1
            except (AttributeError, TypeError, ValueError):
                pass
        count = max(1, self._char_count())
        y0 = int(round(src_h * self._char_index / count))
        y1 = int(round(src_h * (self._char_index + 1) / count))
        return 0, y0, src_w, max(y0 + 1, y1)

    def _render_column_image(self) -> None:
        if self._source_pixmap.isNull():
            self._image_label.setText("列图加载失败")
            self._image_label.adjustSize()
            return
        base = self._source_pixmap.scaledToWidth(
            self._display_width,
            self._Qt.TransformationMode.SmoothTransformation,
        )
        painted = self._QPixmap(base)
        scale_x = painted.width() / max(1, self._source_pixmap.width())
        scale_y = painted.height() / max(1, self._source_pixmap.height())
        painter = self._QPainter(painted)
        # Always draw the physical slider frames.  OCR-aligned/synthetic cells
        # remain available for editor-index bookkeeping but never replace the
        # real one-glyph geometry on the preview.
        physical_boxes = self._physical_boxes()
        guide_boxes = physical_boxes or self._display_boxes()
        guide_pen = self._QPen(self._QColor(22, 119, 255, 235))
        guide_pen.setWidth(2)
        painter.setPen(guide_pen)
        for box in guide_boxes:
            try:
                gx0 = int(round(float(box.get("x0", 0)) * scale_x))
                gx1 = int(round(float(box.get("x1", 0)) * scale_x))
                gy0 = int(round(float(box.get("y0", 0)) * scale_y))
                gy1 = int(round(float(box.get("y1", 0)) * scale_y))
            except Exception:
                continue
            painter.drawRect(max(1, gx0), max(1, gy0), max(3, gx1 - gx0), max(3, gy1 - gy0))
        x0, y0, x1, y1 = self._glyph_box_for_index()
        x0 = int(round(x0 * scale_x)); x1 = int(round(x1 * scale_x))
        y0 = int(round(y0 * scale_y)); y1 = int(round(y1 * scale_y))
        pen = self._QPen(self._Qt.GlobalColor.red)
        pen.setWidth(3)
        painter.setPen(pen)
        painter.drawRect(max(1, x0), max(1, y0), max(4, x1 - x0), max(4, y1 - y0))
        painter.end()
        self._image_label.setPixmap(painted)
        self._image_label.resize(painted.size())

    def _scroll_selected_into_view(self) -> None:
        """Center the selected left-hand frame without changing its index."""
        if self._source_pixmap.isNull() or not hasattr(self, "_image_scroll"):
            return
        try:
            _x0, y0, _x1, y1 = self._glyph_box_for_index()
            scale = self._display_width / max(1.0, float(self._source_pixmap.width()))
            selected_center = ((float(y0) + float(y1)) / 2.0) * scale
            viewport_height = max(1, self._image_scroll.viewport().height())
            bar = self._image_scroll.verticalScrollBar()
            bar.setValue(int(round(selected_center - viewport_height / 2.0)))
        except Exception:
            return

    def _render_glyph_crop(self) -> None:
        if self._source_pixmap.isNull():
            self._glyph_label.setText("字形图不可用")
            return
        src_h = self._source_pixmap.height()
        src_w = self._source_pixmap.width()
        x0, y0, x1, y1 = self._glyph_box_for_index()
        box_w = max(1, x1 - x0); box_h = max(1, y1 - y0)
        pad_x = max(2, int(round(box_w * 0.12)))
        pad_y = max(2, int(round(box_h * 0.12)))
        x = max(0, x0 - pad_x); y = max(0, y0 - pad_y)
        x2 = min(src_w, x1 + pad_x); y2 = min(src_h, y1 + pad_y)
        crop = self._source_pixmap.copy(x, y, max(1, x2 - x), max(1, y2 - y))
        crop = crop.scaled(
            260, 260,
            self._Qt.AspectRatioMode.KeepAspectRatio,
            self._Qt.TransformationMode.SmoothTransformation,
        )
        self._glyph_label.setPixmap(crop)
        text = self._current_text()
        ch = text[self._char_index] if text and self._char_index < len(text) else "∅"
        box = self._physical_box_for_char_index() or (
            self._display_boxes()[self._char_index]
            if self._display_boxes() and self._char_index < len(self._display_boxes())
            else {}
        )
        source = str(box.get("source", "") or "") if isinstance(box, dict) else ""
        synthetic_note = " · 补框" if isinstance(box, dict) and box.get("synthetic") else ""
        self._glyph_label.setToolTip(f"当前字符：{ch} · 第 {self._char_index + 1}/{max(1, len(text))} 字{synthetic_note}")
        self._glyph_index_label.setText(
            f"第 {self._char_index + 1}/{max(1, len(text))} 字 · “{ch}” · 左/中/右索引已锁定"
            + (f" · {source}" if source else "")
        )

    def _on_image_ratio(self, ratio: float) -> None:
        self._clear_candidates()
        count = max(1, self._char_count())
        boxes = self._display_boxes()
        if boxes and not self._source_pixmap.isNull():
            target_y = max(0.0, min(1.0, float(ratio))) * self._source_pixmap.height()
            self._char_index = min(
                range(len(boxes)),
                key=lambda index: abs(
                    (float(boxes[index].get("y0", 0)) + float(boxes[index].get("y1", 0))) / 2.0
                    - target_y
                ),
            )
        else:
            self._char_index = max(0, min(count - 1, int(float(ratio) * count)))
        self._char_index = max(0, min(count - 1, int(self._char_index)))
        self._select_char()

    def _on_editor_changed(self) -> None:
        if self._loading:
            return
        item = self._current()
        previous_count = int(item.get("_display_text_count", len(str(item.get("text", "")))) or 0)
        self._save_current()
        new_count = self._char_count()
        if new_count != previous_count:
            item.pop("_display_boxes_cache_key", None)
            item.pop("_display_boxes_cache", None)
        item["_display_text_count"] = new_count
        if self._char_count():
            self._char_index = min(self._char_index, self._char_count() - 1)
        else:
            self._char_index = 0
        self._render_column_image()
        self._render_glyph_crop()
        if hasattr(self, "_context_text"):
            entry = self._context_entry_for_current()
            sentence = str(entry.text or "") if entry is not None else ""
            value = (
                f"所属句 OCR：{sentence}\n\n当前列人工纠错：{self._current_text()}"
                if sentence else f"当前列人工纠错：{self._current_text()}"
            )
            self._context_text.setPlainText(value)

    def _on_cursor_changed(self) -> None:
        if self._loading or self._syncing_selection:
            return
        cursor = self._editor.textCursor()
        pos = int(cursor.selectionStart() if cursor.hasSelection() else cursor.position())
        if self._char_count():
            new_index = max(0, min(self._char_count() - 1, pos))
            if new_index != self._char_index:
                self._clear_candidates()
            self._char_index = new_index
            self._render_column_image()
            self._render_glyph_crop()
            self._QTimer.singleShot(0, self._scroll_selected_into_view)

    def _move_char(self, delta: int) -> None:
        self._clear_candidates()
        count = self._char_count()
        if not count:
            return
        self._char_index = max(0, min(count - 1, self._char_index + int(delta)))
        self._select_char()

    def _jump_issue(self, delta: int) -> None:
        if not self._issue_indices:
            self._status.setText("当前没有自动标记的疑点列。")
            return
        try:
            pos = self._issue_indices.index(self._index)
        except ValueError:
            pos = -1 if delta > 0 else 0
        pos = (pos + int(delta)) % len(self._issue_indices)
        self._load_column(self._issue_indices[pos])

    def _replacement_value(self) -> str:
        return self._char_input.text()

    def _replace_char(self) -> None:
        value = self._replacement_value()
        if not value:
            self._char_input.setFocus()
            return
        text = self._current_text()
        if text:
            index = max(0, min(self._char_index, len(text) - 1))
            text = text[:index] + value + text[index + 1:]
        else:
            text = value
            self._char_index = 0
        self._loading = True
        self._editor.setPlainText(text)
        self._loading = False
        self._char_input.clear()
        self._save_current()
        self._reviewed.add(str(self._current().get("block_id", "")))
        self._select_char()
        self._status.setText(f"已替换为“{value}”。")

    def _insert_char(self) -> None:
        value = self._replacement_value()
        if not value:
            self._char_input.setFocus()
            return
        text = self._current_text()
        index = max(0, min(self._char_index, len(text)))
        text = text[:index] + value + text[index:]
        self._loading = True
        self._editor.setPlainText(text)
        self._loading = False
        self._char_input.clear()
        self._save_current()
        self._reviewed.add(str(self._current().get("block_id", "")))
        self._select_char()
        self._status.setText(f"已插入“{value}”。")

    def _delete_char(self) -> None:
        text = self._current_text()
        if not text:
            return
        index = max(0, min(self._char_index, len(text) - 1))
        text = text[:index] + text[index + 1:]
        self._loading = True
        self._editor.setPlainText(text)
        self._loading = False
        self._save_current()
        self._reviewed.add(str(self._current().get("block_id", "")))
        self._char_index = min(self._char_index, max(0, len(text) - 1))
        self._select_char()
        self._status.setText("已删除当前字。")

    def _restore_column(self) -> None:
        text = self._originals[self._index]
        self._loading = True
        self._editor.setPlainText(text)
        self._loading = False
        self._records[self._index]["text"] = text
        self._char_index = 0
        self._reviewed.discard(str(self._current().get("block_id", "")))
        self._select_char()
        self._status.setText("已恢复本列原始 OCR。")

    def _mark_reviewed(self) -> None:
        self._save_current()
        self._reviewed.add(str(self._current().get("block_id", "")))
        self._status.setText("本列已标记为人工核对。")

    def _focus_ime(self) -> None:
        self._char_input.setFocus()
        self._char_input.selectAll()
        self._status.setText("输入框已聚焦，可切换 macOS 日语输入法或系统手写输入。")

    def _learn_manual_glyphs(self) -> int:
        """Store only explicit one-to-one manual corrections in glyph memory."""
        from adapters.glyph_memory_db import GlyphMemoryDB
        database = GlyphMemoryDB()
        learned = 0
        for original, item in zip(self._originals, self._records):
            text = str(item.get("text", ""))
            boxes = item.get("glyph_boxes") or []
            if not isinstance(boxes, list) or len(original) != len(text) or len(text) != len(boxes):
                continue
            image_path = self._root / str(item.get("image", ""))
            if not image_path.exists():
                continue
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            try:
                for index, (before, after) in enumerate(zip(original, text)):
                    if before == after or len(after) != 1 or after in {"□", "■", "◻", "◼", "�"}:
                        continue
                    box = boxes[index]
                    try:
                        x0 = int(round(float(box.get("x0", 0)))); x1 = int(round(float(box.get("x1", 0))))
                        y0 = int(round(float(box.get("y0", 0)))); y1 = int(round(float(box.get("y1", 0))))
                    except (AttributeError, TypeError, ValueError):
                        continue
                    if x1 <= x0 or y1 <= y0:
                        continue
                    pad_x = max(2, int(round((x1 - x0) * 0.10)))
                    pad_y = max(2, int(round((y1 - y0) * 0.10)))
                    crop = image.crop((
                        max(0, x0 - pad_x), max(0, y0 - pad_y),
                        min(image.width, x1 + pad_x), min(image.height, y1 + pad_y),
                    )).convert("RGB")
                    try:
                        database.add_sample(
                            crop, after, scope_key=self._glyph_scope_key,
                            source="ocr_manual_review", also_global=False,
                        )
                        learned += 1
                    finally:
                        crop.close()
            finally:
                image.close()
        return learned

    def _accept(self) -> None:
        self._save_current()
        unresolved = [
            item for item in self._records
            if bool(item.get("column_ocr_empty") or item.get("column_manual_placeholder"))
            and not str(item.get("text", "")).strip("□■◻◼� \t\r\n　")
        ]
        if unresolved:
            answer = self._message_box.question(
                self._dialog,
                "仍有空列未输入",
                f"还有 {len(unresolved)} 列 OCR 三次均为空，尚未人工输入。\n\n"
                "继续后会保留醒目的 □ 标记，不会静默删除这些列；建议返回逐列补全。\n\n"
                "仍要应用当前结果吗？",
                self._message_box.StandardButton.Yes | self._message_box.StandardButton.No,
                self._message_box.StandardButton.No,
            )
            if answer != self._message_box.StandardButton.Yes:
                first = unresolved[0]
                try:
                    self._load_column(self._records.index(first))
                except ValueError:
                    pass
                return
        try:
            learned = self._learn_manual_glyphs()
            if learned:
                self._status.setText(f"已将 {learned} 个明确人工改字写入当前书字形记忆。")
        except Exception as exc:
            # Memory is an optional accelerator; never block applying reviewed text.
            self._status.setText(f"字形记忆写入失败，人工纠错仍会正常应用：{exc}")
        payload = []
        for original, item in zip(self._originals, self._records):
            text = str(item.get("text", ""))
            block_id = str(item.get("block_id", ""))
            payload.append({
                "block_id": block_id,
                "text": text,
                "reviewed": block_id in self._reviewed or text != original,
            })
        self._payload = payload
        self._dialog.accept()

    def exec(self) -> bool:
        try:
            return bool(self._dialog.exec())
        finally:
            self._candidate_generation += 1
            self._temp.cleanup()
