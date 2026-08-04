#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lossless OCR-only cleanup for isolated Japanese vertical columns.

The source page and review preview are never modified.  This module receives the
already isolated recognition image, masks likely ruby/side debris with the
estimated paper colour, and optionally removes guaranteed blank canvas.  It does
not resize or sharpen glyph pixels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from PIL import Image, ImageChops, ImageDraw, ImageStat


@dataclass(frozen=True)
class ColumnCleanupResult:
    image: Image.Image
    crop_box: tuple[int, int, int, int]
    main_band: tuple[int, int] | None
    removed_bands: tuple[tuple[int, int, str], ...]
    threshold: int


_PROFILES = {
    "weak": {
        "ruby_width": 0.43,
        "ruby_score": 0.34,
        "ruby_gap": 0.52,
        "fragment_width": 0.20,
        "fragment_score": 0.07,
        "x_margin": 0.22,
        "y_margin": 0.70,
    },
    "standard": {
        "ruby_width": 0.60,
        "ruby_score": 0.52,
        "ruby_gap": 0.82,
        "fragment_width": 0.32,
        "fragment_score": 0.13,
        "x_margin": 0.17,
        "y_margin": 0.58,
    },
    "strong": {
        "ruby_width": 0.78,
        "ruby_score": 0.72,
        "ruby_gap": 1.08,
        "fragment_width": 0.48,
        "fragment_score": 0.23,
        "x_margin": 0.12,
        "y_margin": 0.46,
    },
}


def normalise_ruby_strength(value: str | None) -> str:
    raw = str(value or "standard").strip().lower()
    aliases = {
        "low": "weak", "conservative": "weak", "弱": "weak",
        "normal": "standard", "balanced": "standard", "标准": "standard", "標準": "standard",
        "high": "strong", "aggressive": "strong", "强": "strong", "強": "strong",
    }
    raw = aliases.get(raw, raw)
    return raw if raw in _PROFILES else "standard"


def _paper_colour(image: Image.Image) -> tuple[int, int, int]:
    rgb = image.convert("RGB")
    try:
        w, h = rgb.size
        sample = max(2, min(12, w // 5 or 2, h // 5 or 2))
        patches = [
            rgb.crop((0, 0, sample, sample)),
            rgb.crop((max(0, w - sample), 0, w, sample)),
            rgb.crop((0, max(0, h - sample), sample, h)),
            rgb.crop((max(0, w - sample), max(0, h - sample), w, h)),
        ]
        values = []
        for patch in patches:
            stat = ImageStat.Stat(patch)
            values.append(tuple(int(round(v)) for v in stat.mean[:3]))
            patch.close()
        values.sort(key=lambda item: sum(item))
        return values[len(values) // 2]
    finally:
        rgb.close()


def _otsu_threshold(gray: Image.Image) -> int:
    hist = gray.histogram()[:256]
    total = sum(hist)
    if total <= 0:
        return 210
    weighted_total = sum(index * count for index, count in enumerate(hist))
    bg_weight = 0
    bg_sum = 0.0
    best = -1.0
    threshold = 210
    for index, count in enumerate(hist):
        bg_weight += count
        if bg_weight <= 0:
            continue
        fg_weight = total - bg_weight
        if fg_weight <= 0:
            break
        bg_sum += index * count
        bg_mean = bg_sum / bg_weight
        fg_mean = (weighted_total - bg_sum) / fg_weight
        variance = bg_weight * fg_weight * (bg_mean - fg_mean) ** 2
        if variance > best:
            best = variance
            threshold = index
    # Scanned pages often have a slightly grey paper background.  The cap keeps
    # paper texture out while retaining antialiased printed strokes.
    return max(145, min(232, int(threshold + 18)))


def _projection(mask: Image.Image) -> list[int]:
    if mask.width <= 0:
        return []
    collapsed = mask.resize((mask.width, 1), Image.Resampling.BOX)
    try:
        getter = getattr(collapsed, "get_flattened_data", None)
        return [int(v) for v in (getter() if callable(getter) else collapsed.getdata())]
    finally:
        collapsed.close()


def _fill_short_gaps(active: list[bool], max_gap: int) -> list[bool]:
    result = list(active)
    index = 0
    while index < len(result):
        if result[index]:
            index += 1
            continue
        start = index
        while index < len(result) and not result[index]:
            index += 1
        if start > 0 and index < len(result) and index - start <= max_gap:
            result[start:index] = [True] * (index - start)
    return result


def _runs(values: list[int], width: int) -> list[dict]:
    if not values:
        return []
    peak = max(values)
    if peak <= 0:
        return []
    gate = max(2, int(round(peak * 0.075)))
    active = _fill_short_gaps([value >= gate for value in values], max(1, round(width * 0.012)))
    items: list[dict] = []
    index = 0
    while index < len(active):
        if not active[index]:
            index += 1
            continue
        start = index
        while index < len(active) and active[index]:
            index += 1
        end = index
        score = float(sum(values[start:end]))
        if end - start >= 1 and score > 0:
            items.append({"start": start, "end": end, "width": end - start, "score": score})
    return items


def _main_band(items: list[dict], width: int) -> dict | None:
    if not items:
        return None
    # The body column is normally both wider and darker than ruby.  Width is
    # deliberately only a square-root bonus so a broad faint shadow cannot win.
    return max(
        items,
        key=lambda item: item["score"] * max(1.0, item["width"] ** 0.5)
        * (1.0 + 0.10 * (1.0 - abs(((item["start"] + item["end"]) / 2.0) - width / 2.0) / max(1.0, width / 2.0))),
    )


def _band_gap(item: dict, main: dict) -> int:
    if item["end"] <= main["start"]:
        return main["start"] - item["end"]
    if item["start"] >= main["end"]:
        return item["start"] - main["end"]
    return 0


def _expand_main(items: list[dict], main: dict) -> dict:
    start, end = int(main["start"]), int(main["end"])
    score = float(main["score"])
    changed = True
    while changed:
        changed = False
        current_width = max(1, end - start)
        for item in items:
            if item is main or (item["start"] >= start and item["end"] <= end):
                continue
            gap = 0
            if item["end"] <= start:
                gap = start - item["end"]
            elif item["start"] >= end:
                gap = item["start"] - end
            else:
                gap = 0
            substantial = item["width"] >= current_width * 0.58 or item["score"] >= score * 0.56
            if substantial and gap <= max(2, round(current_width * 0.18)):
                start = min(start, int(item["start"]))
                end = max(end, int(item["end"]))
                score += float(item["score"])
                changed = True
    return {"start": start, "end": end, "width": end - start, "score": score}


def _dominant_body_core(
    values: list[int],
    main: dict,
    width: int,
    profile_name: str,
) -> tuple[int, int] | None:
    """Return a conservative body-only x band inside a merged body/Ruby run.

    Ruby can touch the main glyph band and therefore disappear into the same
    projection run.  The body column still has much larger accumulated vertical
    ink.  A high-density core plus a glyph-sized safety pad removes side readings
    without resampling the printed body pixels.
    """
    if not values or width <= 0:
        return None
    left = max(0, int(main.get("start", 0)))
    right = min(width, int(main.get("end", width)))
    if right - left < 4:
        return None
    local = values[left:right]
    peak = max(local or [0])
    if peak <= 0:
        return None
    ratio = {"weak": 0.10, "standard": 0.15, "strong": 0.20}[profile_name]
    active = [value >= max(2, int(round(peak * ratio))) for value in local]
    active = _fill_short_gaps(active, max(1, round((right - left) * 0.035)))
    runs = _runs([255 if flag else 0 for flag in active], right - left)
    if not runs:
        return None
    peak_x = max(range(left, right), key=lambda x: values[x])
    peak_local = peak_x - left
    containing = [item for item in runs if item["start"] <= peak_local < item["end"]]
    core = max(containing or runs, key=lambda item: (item["score"], item["width"]))
    core_left = left + int(core["start"])
    core_right = left + int(core["end"])
    main_width = max(1, right - left)
    min_width = max(5, round(main_width * 0.46))
    if core_right - core_left < min_width:
        centre = (core_left + core_right) / 2.0
        core_left = max(left, int(round(centre - min_width / 2.0)))
        core_right = min(right, core_left + min_width)
        core_left = max(left, core_right - min_width)
    pad_ratio = {"weak": 0.38, "standard": 0.29, "strong": 0.22}[profile_name]
    pad = max(2, round((core_right - core_left) * pad_ratio))
    core_left = max(0, core_left - pad)
    core_right = min(width, core_right + pad)
    if core_right - core_left < 4:
        return None
    return core_left, core_right


def _shift(image: Image.Image, dx: int, dy: int) -> Image.Image:
    shifted = Image.new("L", image.size, 0)
    shifted.paste(image, (dx, dy))
    return shifted


def _remove_isolated_single_pixels(mask: Image.Image) -> Image.Image:
    """Remove only one-pixel islands; punctuation and real strokes are untouched."""
    neighbours = Image.new("L", mask.size, 0)
    try:
        for dx, dy in ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)):
            shifted = _shift(mask, dx, dy)
            combined = ImageChops.lighter(neighbours, shifted)
            neighbours.close()
            shifted.close()
            neighbours = combined
        supported = ImageChops.multiply(mask, neighbours)
        return supported
    finally:
        neighbours.close()


def _remove_small_components_outside_main(
    source: Image.Image,
    mask: Image.Image,
    main: dict,
    paper: tuple[int, int, int],
    profile_name: str,
) -> int:
    """Erase only tiny detached components outside the正文主字带."""
    width, height = mask.size
    if width * height > 1_800_000:
        return 0
    main_width = max(1, int(main["width"]))
    area_ratio = {"weak": 0.006, "standard": 0.016, "strong": 0.032}[profile_name]
    max_area = max(2, min(96, round(main_width * main_width * area_ratio)))
    max_side = max(2, round(main_width * {"weak": 0.16, "standard": 0.25, "strong": 0.36}[profile_name]))
    erased: list[tuple[int, int]] = []

    for band_left, band_right in ((0, int(main["start"])), (int(main["end"]), width)):
        if band_right <= band_left:
            continue
        band = mask.crop((band_left, 0, band_right, height))
        try:
            bw = band.width
            data = bytearray(band.tobytes())
        finally:
            band.close()
        for start in range(len(data)):
            if data[start] == 0:
                continue
            stack = [start]
            data[start] = 0
            component: list[int] = []
            min_x = max_x = start % bw
            min_y = max_y = start // bw
            while stack:
                index = stack.pop()
                component.append(index)
                x = index % bw
                y = index // bw
                min_x = min(min_x, x); max_x = max(max_x, x)
                min_y = min(min_y, y); max_y = max(max_y, y)
                for ny in range(max(0, y - 1), min(height, y + 2)):
                    row = ny * bw
                    for nx in range(max(0, x - 1), min(bw, x + 2)):
                        neighbour = row + nx
                        if data[neighbour]:
                            data[neighbour] = 0
                            stack.append(neighbour)
            comp_w = max_x - min_x + 1
            comp_h = max_y - min_y + 1
            if len(component) <= max_area and comp_w <= max_side and comp_h <= max_side:
                erased.extend((band_left + (index % bw), index // bw) for index in component)

    if not erased:
        return 0
    source_pixels = source.load()
    mask_pixels = mask.load()
    for x, y in erased:
        source_pixels[x, y] = paper
        mask_pixels[x, y] = 0
    return len(erased)


def cleanup_column_image(
    image: Image.Image,
    *,
    auto_filter_ruby: bool = True,
    filter_fragments: bool = True,
    smart_crop: bool = True,
    ruby_strength: str = "standard",
    background: tuple[int, int, int] | None = None,
    preserve_body_pixels: bool = False,
) -> ColumnCleanupResult:
    """Return an OCR-only column image without resampling glyph pixels.

    ``preserve_body_pixels`` is the production OCR safety contract.  When it is
    enabled, cleanup may remove only guaranteed blank outer canvas.  No ink
    component, side band, detached dakuten/handakuten, punctuation, or thin
    radical is painted over.  The legacy destructive Ruby filter remains
    available for review/debug callers by explicitly passing ``False``.
    """
    source = image.convert("RGB")
    width, height = source.size
    if width < 4 or height < 4:
        return ColumnCleanupResult(source, (0, 0, width, height), None, (), 210)

    paper = background or _paper_colour(source)
    gray = source.convert("L")
    threshold = _otsu_threshold(gray)
    mask = gray.point(lambda value: 255 if value < threshold else 0, mode="L")
    gray.close()
    if filter_fragments and not preserve_body_pixels:
        cleaned_mask = _remove_isolated_single_pixels(mask)
        mask.close()
        mask = cleaned_mask
    bbox = mask.getbbox()
    if bbox is None:
        mask.close()
        return ColumnCleanupResult(source, (0, 0, width, height), None, (), threshold)

    profile_name = normalise_ruby_strength(ruby_strength)
    profile = _PROFILES[profile_name]
    values = _projection(mask)
    items = _runs(values, width)
    main = _main_band(items, width)
    if main is None:
        mask.close()
        return ColumnCleanupResult(source, (0, 0, width, height), None, (), threshold)
    main = _expand_main(items, main)
    main_width = max(1, int(main["width"]))
    main_score = max(1.0, float(main["score"]))

    if preserve_body_pixels:
        # Production OCR is intentionally one-way lossless: the binary mask is
        # used only to locate guaranteed blank canvas.  The RGB source remains
        # byte-for-byte unchanged inside the returned crop.  This prevents
        # detached dakuten/handakuten, small kana, punctuation, quote tips and
        # fragmented kanji radicals from being mistaken for Ruby or debris.
        final_bbox = mask.getbbox()
        mask.close()
        if final_bbox is None or not smart_crop:
            return ColumnCleanupResult(
                source, (0, 0, width, height),
                (int(main["start"]), int(main["end"])), (), threshold
            )
        left, top, right, bottom = final_bbox
        # Use the most conservative margins independent of the selected legacy
        # Ruby profile.  Strength must never shrink the body-pixel safety area.
        x_margin = max(6, round(main_width * max(item["x_margin"] for item in _PROFILES.values())))
        y_margin = max(10, round(main_width * max(item["y_margin"] for item in _PROFILES.values())))
        left = max(0, left - x_margin)
        right = min(width, right + x_margin)
        top = max(0, top - y_margin)
        bottom = min(height, bottom + y_margin)
        if right - left < 4 or bottom - top < 4:
            return ColumnCleanupResult(
                source, (0, 0, width, height),
                (int(main["start"]), int(main["end"])), (), threshold
            )
        cropped = source.crop((left, top, right, bottom)).convert("RGB")
        source.close()
        return ColumnCleanupResult(
            cropped, (left, top, right, bottom),
            (int(main["start"]), int(main["end"])), (), threshold
        )

    removed: list[tuple[int, int, str]] = []
    for item in items:
        if item["end"] > main["start"] and item["start"] < main["end"]:
            continue
        gap = _band_gap(item, main)
        width_ratio = float(item["width"]) / main_width
        score_ratio = float(item["score"]) / main_score
        is_right = item["start"] >= main["end"]
        ruby_like = bool(
            auto_filter_ruby
            and gap <= max(3, round(main_width * profile["ruby_gap"]))
            and width_ratio <= profile["ruby_width"]
            and score_ratio <= profile["ruby_score"]
            and (
                is_right
                or profile_name == "strong"
                or (width_ratio <= profile["ruby_width"] * 0.72 and score_ratio <= profile["ruby_score"] * 0.70)
            )
        )
        fragment_like = bool(
            filter_fragments
            and width_ratio <= profile["fragment_width"]
            and score_ratio <= profile["fragment_score"]
        )
        if ruby_like or fragment_like:
            pad = 2 if profile_name == "strong" else 1
            left = max(0, int(item["start"]) - pad)
            right = min(width, int(item["end"]) + pad)
            removed.append((left, right, "ruby" if ruby_like else "fragment"))

    # Strong mode is the publication-OCR default: keep only the dominant body
    # band.  This catches Ruby that touches the body run and therefore cannot be
    # removed by the separate-band test above.
    body_core = _dominant_body_core(values, main, width, profile_name) if auto_filter_ruby else None
    if body_core is not None and profile_name == "strong":
        core_left, core_right = body_core
        if core_left > 0:
            removed.append((0, core_left, "ruby_side_all"))
        if core_right < width:
            removed.append((core_right, width, "ruby_side_all"))

    if removed:
        # Merge overlapping bands so repeated detectors cannot leave one-pixel
        # seams that Apple Vision turns into brackets or vertical bars.
        merged: list[tuple[int, int, str]] = []
        for band_left, band_right, reason in sorted(removed, key=lambda item: (item[0], item[1])):
            if not merged or band_left > merged[-1][1]:
                merged.append((band_left, band_right, reason))
            else:
                old_left, old_right, old_reason = merged[-1]
                merged[-1] = (old_left, max(old_right, band_right), old_reason if old_reason == reason else "ruby_or_fragment")
        removed = merged
        draw = ImageDraw.Draw(source)
        mask_draw = ImageDraw.Draw(mask)
        for left, right, _reason in removed:
            draw.rectangle((left, 0, max(left, right - 1), height - 1), fill=paper)
            mask_draw.rectangle((left, 0, max(left, right - 1), height - 1), fill=0)

    if filter_fragments:
        removed_pixels = _remove_small_components_outside_main(
            source, mask, main, paper, profile_name
        )
        if removed_pixels:
            removed.append((-1, removed_pixels, "small_components"))

    final_bbox = mask.getbbox()
    mask.close()
    if final_bbox is None:
        return ColumnCleanupResult(source, (0, 0, width, height), (int(main["start"]), int(main["end"])), tuple(removed), threshold)

    if not smart_crop:
        return ColumnCleanupResult(source, (0, 0, width, height), (int(main["start"]), int(main["end"])), tuple(removed), threshold)

    left, top, right, bottom = final_bbox
    x_margin = max(4, round(main_width * profile["x_margin"]))
    y_margin = max(8, round(main_width * profile["y_margin"]))
    left = max(0, left - x_margin)
    right = min(width, right + x_margin)
    top = max(0, top - y_margin)
    bottom = min(height, bottom + y_margin)
    if right - left < 4 or bottom - top < 4:
        return ColumnCleanupResult(source, (0, 0, width, height), (int(main["start"]), int(main["end"])), tuple(removed), threshold)
    cropped = source.crop((left, top, right, bottom)).convert("RGB")
    source.close()
    return ColumnCleanupResult(cropped, (left, top, right, bottom), (int(main["start"]), int(main["end"])), tuple(removed), threshold)
