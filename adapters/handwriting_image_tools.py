#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Image helpers for the Japanese handwriting-input OCR card.

The functions in this module deliberately preserve the original raster size.
They mask unwanted neighbouring text / ruby using estimated paper colour rather
than cropping and rescaling the whole column. This keeps stroke shapes stable for
both automatic tracing and manual review.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from statistics import median
from typing import Sequence

from PIL import Image, ImageDraw, ImageOps

from adapters.segmentation import classify_vertical_ruby, detect_center_intervals


@dataclass(slots=True)
class MainBandMaskInfo:
    applied: bool
    x0: int
    x1: int
    original_width: int
    original_height: int
    ink_columns: int
    removed_ink_pixels: int
    background: tuple[int, int, int]


def _gray(image: Image.Image) -> Image.Image:
    return image if image.mode == "L" else ImageOps.grayscale(image)


def _pixel_values(image: Image.Image) -> list[int]:
    getter = getattr(image, "get_flattened_data", None)
    if callable(getter):
        return list(getter())
    return list(image.getdata())


def _otsu(values: Sequence[int]) -> int:
    hist = [0] * 256
    for value in values:
        hist[int(value)] += 1
    total = len(values)
    if not total:
        return 180
    total_sum = sum(i * hist[i] for i in range(256))
    weight_b = 0
    sum_b = 0
    best_t = 127
    best_var = -1.0
    for t in range(256):
        weight_b += hist[t]
        if not weight_b:
            continue
        weight_f = total - weight_b
        if not weight_f:
            break
        sum_b += t * hist[t]
        mean_b = sum_b / weight_b
        mean_f = (total_sum - sum_b) / weight_f
        variance = weight_b * weight_f * (mean_b - mean_f) ** 2
        if variance > best_var:
            best_var = variance
            best_t = t
    return min(238, max(65, best_t + 8))


def estimate_paper_background(image: Image.Image) -> tuple[int, int, int]:
    rgb = image.convert("RGB")
    w, h = rgb.size
    if w <= 0 or h <= 0:
        return (255, 255, 255)
    border = max(1, min(w, h) // 18)
    samples: list[tuple[int, int, int]] = []
    px = rgb.load()
    for y in range(h):
        for x in range(w):
            if x < border or x >= w - border or y < border or y >= h - border:
                r, g, b = px[x, y]
                # Exclude obvious dark ink from the paper estimate.
                if (r + g + b) / 3 >= 150:
                    samples.append((r, g, b))
    if not samples:
        return (255, 255, 255)
    return (
        int(median(v[0] for v in samples)),
        int(median(v[1] for v in samples)),
        int(median(v[2] for v in samples)),
    )


def _runs(indices: Sequence[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    out: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for raw in indices[1:]:
        value = int(raw)
        if value <= prev + 1:
            prev = value
            continue
        out.append((start, prev + 1))
        start = prev = value
    out.append((start, prev + 1))
    return out


_FAST_COMPONENTS_BACKEND = None
_FAST_COMPONENTS_IMPORT_ATTEMPTED = False


def _fast_connected_components(binary: Sequence[int], w: int, h: int) -> list[dict] | None:
    """Optional OpenCV fast path with byte-for-byte geometry parity.

    Paddle/OpenCV installations already provide ``cv2`` and NumPy on most OCR
    machines.  Import them lazily so minimal/manual installations keep the pure
    Python implementation.  Only foreground pixels are grouped, avoiding a full
    label-map sort when a page is mostly paper.
    """
    global _FAST_COMPONENTS_BACKEND, _FAST_COMPONENTS_IMPORT_ATTEMPTED
    if os.environ.get("NOVEL_FORMATTER_DISABLE_FAST_COMPONENTS", "").strip().lower() in {"1", "true", "yes"}:
        return None
    if not _FAST_COMPONENTS_IMPORT_ATTEMPTED:
        _FAST_COMPONENTS_IMPORT_ATTEMPTED = True
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
            _FAST_COMPONENTS_BACKEND = (cv2, np)
        except Exception:
            _FAST_COMPONENTS_BACKEND = None
    if _FAST_COMPONENTS_BACKEND is None or w <= 0 or h <= 0:
        return None
    cv2, np = _FAST_COMPONENTS_BACKEND
    try:
        array = np.asarray(binary, dtype=np.uint8).reshape((h, w))
        label_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            array, connectivity=8, ltype=cv2.CV_32S,
        )
        if label_count <= 1:
            return []
        flat_labels = labels.reshape(-1)
        foreground_indices = np.flatnonzero(flat_labels)
        if foreground_indices.size == 0:
            return []
        foreground_labels = flat_labels[foreground_indices]
        order = np.argsort(foreground_labels, kind="stable")
        grouped_indices = foreground_indices[order]
        counts = np.bincount(foreground_labels, minlength=label_count)
        offsets = np.cumsum(counts)
        components: list[dict] = []
        left_col = int(getattr(cv2, "CC_STAT_LEFT", 0))
        top_col = int(getattr(cv2, "CC_STAT_TOP", 1))
        width_col = int(getattr(cv2, "CC_STAT_WIDTH", 2))
        height_col = int(getattr(cv2, "CC_STAT_HEIGHT", 3))
        area_col = int(getattr(cv2, "CC_STAT_AREA", 4))
        for label in range(1, label_count):
            start = int(offsets[label - 1])
            end = int(offsets[label])
            if end <= start:
                continue
            pixels = grouped_indices[start:end].astype(np.int64, copy=False).tolist()
            x0 = int(stats[label, left_col])
            y0 = int(stats[label, top_col])
            x1 = x0 + int(stats[label, width_col])
            y1 = y0 + int(stats[label, height_col])
            components.append({
                "x0": x0, "x1": x1, "y0": y0, "y1": y1,
                "area": int(stats[label, area_col]), "pixels": pixels,
                "cx": (x0 + x1) / 2.0,
                "cy": (y0 + y1) / 2.0,
                "_first_pixel": int(pixels[0]),
            })
        # OpenCV labels normally follow scan order, but sorting makes the result
        # deterministic and identical to the historical row-major flood fill.
        components.sort(key=lambda item: int(item.pop("_first_pixel", 0)))
        return components
    except Exception:
        return None


def _connected_components(binary: Sequence[int], w: int, h: int) -> list[dict]:
    fast = _fast_connected_components(binary, w, h)
    if fast is not None:
        return fast
    visited = bytearray(w * h)
    components: list[dict] = []
    for idx, value in enumerate(binary):
        if not value or visited[idx]:
            continue
        stack = [idx]
        visited[idx] = 1
        x0 = x1 = idx % w
        y0 = y1 = idx // w
        pixels: list[int] = []
        while stack:
            cur = stack.pop()
            pixels.append(cur)
            x = cur % w
            y = cur // w
            x0 = min(x0, x)
            x1 = max(x1, x)
            y0 = min(y0, y)
            y1 = max(y1, y)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if not dx and not dy:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        ni = ny * w + nx
                        if binary[ni] and not visited[ni]:
                            visited[ni] = 1
                            stack.append(ni)
        components.append({
            "x0": x0, "x1": x1 + 1, "y0": y0, "y1": y1 + 1,
            "area": len(pixels), "pixels": pixels,
            "cx": (x0 + x1 + 1) / 2.0,
            "cy": (y0 + y1 + 1) / 2.0,
        })
    return components


def mask_main_text_band(
    image: Image.Image,
    *,
    padding_ratio: float = 0.16,
    preserve_large_symbols: bool = True,
) -> tuple[Image.Image, MainBandMaskInfo]:
    """Mask ruby / neighbouring residue while preserving original dimensions.

    The dominant main-text band is selected from x-axis ink projections and
    connected-component mass. Pixels outside the selected band are replaced by
    the estimated paper colour. Large punctuation components close to the main
    band can be retained.
    """
    rgb = image.convert("RGB")
    w, h = rgb.size
    background = estimate_paper_background(rgb)
    if w < 6 or h < 6:
        return rgb, MainBandMaskInfo(False, 0, w, w, h, 0, 0, background)

    gray = _gray(rgb)
    values = _pixel_values(gray)
    threshold = _otsu(values)
    binary = [1 if value < threshold else 0 for value in values]
    total_ink = sum(binary)
    if total_ink < max(4, w * h // 1000):
        return rgb, MainBandMaskInfo(False, 0, w, w, h, 0, 0, background)

    x_projection = [0] * w
    for idx, value in enumerate(binary):
        if value:
            x_projection[idx % w] += 1
    active_threshold = max(1, int(round(h * 0.008)))
    active = [x for x, count in enumerate(x_projection) if count >= active_threshold]
    runs = _runs(active)
    if not runs:
        return rgb, MainBandMaskInfo(False, 0, w, w, h, 0, 0, background)

    components = _connected_components(binary, w, h)
    centre = w / 2.0
    best: tuple[float, int, int] | None = None
    for x0, x1 in runs:
        width = max(1, x1 - x0)
        mass = sum(x_projection[x0:x1])
        component_mass = sum(
            comp["area"] for comp in components
            if comp["cx"] >= x0 - width * 0.35 and comp["cx"] <= x1 + width * 0.35
        )
        run_centre = (x0 + x1) / 2.0
        centre_bonus = 1.0 - min(1.0, abs(run_centre - centre) / max(1.0, w / 2.0))
        width_bonus = min(1.0, width / max(1.0, w * 0.35))
        score = mass + component_mass * 0.7 + total_ink * 0.15 * centre_bonus + total_ink * 0.12 * width_bonus
        if best is None or score > best[0]:
            best = (score, x0, x1)
    assert best is not None
    _, band_x0, band_x1 = best

    # Expand to include the bulk of large components touching the selected band.
    initial_width = max(1, band_x1 - band_x0)
    for comp in components:
        comp_w = comp["x1"] - comp["x0"]
        comp_h = comp["y1"] - comp["y0"]
        is_substantial = comp["area"] >= max(5, total_ink * 0.008) or comp_h >= h * 0.04
        touches = comp["x1"] >= band_x0 - initial_width * 0.45 and comp["x0"] <= band_x1 + initial_width * 0.45
        if is_substantial and touches:
            band_x0 = min(band_x0, comp["x0"])
            band_x1 = max(band_x1, comp["x1"])

    padding = max(2, int(round(max(1, band_x1 - band_x0) * padding_ratio)))
    band_x0 = max(0, band_x0 - padding)
    band_x1 = min(w, band_x1 + padding)

    keep = bytearray(w * h)
    for idx, value in enumerate(binary):
        if not value:
            continue
        x = idx % w
        if band_x0 <= x < band_x1:
            keep[idx] = 1

    if preserve_large_symbols:
        for comp in components:
            comp_w = comp["x1"] - comp["x0"]
            comp_h = comp["y1"] - comp["y0"]
            near = comp["x1"] >= band_x0 - max(3, initial_width * 0.22) and comp["x0"] <= band_x1 + max(3, initial_width * 0.22)
            symbol_like = (
                comp["area"] >= max(5, total_ink * 0.012)
                or comp_h >= h * 0.055
                or comp_w >= w * 0.18
            )
            if near and symbol_like:
                for idx in comp["pixels"]:
                    keep[idx] = 1

    out = Image.new("RGB", (w, h), background)
    src = rgb.load()
    dst = out.load()
    removed = 0
    for y in range(h):
        for x in range(w):
            idx = y * w + x
            if keep[idx] or not binary[idx]:
                # Keep all paper pixels inside/outside band to preserve texture;
                # only dark ink outside the mask is painted away.
                dst[x, y] = src[x, y]
            else:
                dst[x, y] = background
                removed += 1

    applied = removed > 0 and (band_x0 > 0 or band_x1 < w)
    return out, MainBandMaskInfo(applied, band_x0, band_x1, w, h, len(active), removed, background)


def mask_single_glyph(image: Image.Image, *, margin_ratio: float = 0.10) -> Image.Image:
    """Conservatively remove side residue while preserving every glyph stroke.

    A Japanese glyph can consist of many disconnected components: the separate
    horizontals in ``言``, dakuten/handakuten, dots, hooks and detached kana
    curves.  The former implementation anchored the mask to the largest
    component and applied a vertical-distance limit.  That deleted legitimate
    strokes before skeletonisation.  This version uses only an x-band test to
    reject ruby/neighbour residue; the full y range of the segmented glyph is
    preserved.
    """
    rgb = image.convert("RGB")
    w, h = rgb.size
    background = estimate_paper_background(rgb)
    gray = _gray(rgb)
    values = _pixel_values(gray)
    threshold = _otsu(values)
    binary = [1 if value < threshold else 0 for value in values]
    components = _connected_components(binary, w, h)
    if not components:
        return rgb

    best = max(components, key=lambda comp: int(comp.get("area", 0)))
    best_width = max(1.0, float(best["x1"] - best["x0"]))
    total_area = max(1, sum(int(comp.get("area", 0)) for comp in components))

    # The largest main-text component is a more reliable x anchor than an
    # area-weighted mean when side ruby/noise repeats down the full glyph crop.
    # Y is deliberately unrestricted so detached strokes remain intact.
    anchor_x = float(best["cx"])
    band_half = max(w * 0.22, best_width * 0.82)
    band_x0 = max(0.0, anchor_x - band_half)
    band_x1 = min(float(w), anchor_x + band_half)
    tiny_limit = max(1, int(round(total_area * 0.0012)))

    keep_components = []
    for comp in components:
        area = int(comp.get("area", 0))
        overlaps_band = float(comp["x1"]) >= band_x0 and float(comp["x0"]) <= band_x1
        centre_near = abs(float(comp["cx"]) - anchor_x) <= max(w * 0.30, best_width * 1.15)
        substantial = area >= max(tiny_limit, int(round(int(best["area"]) * 0.004)))
        # Keep all meaningful components in the main x band, regardless of y.
        # This retains separated bars/dots while still removing side ruby.
        if overlaps_band and (centre_near or substantial):
            keep_components.append(comp)

    if not keep_components:
        keep_components = components

    keep = bytearray(w * h)
    for comp in keep_components:
        for idx in comp["pixels"]:
            keep[idx] = 1
    out = Image.new("RGB", (w, h), background)
    src = rgb.load()
    dst = out.load()
    for y in range(h):
        for x in range(w):
            idx = y * w + x
            if keep[idx] or not binary[idx]:
                dst[x, y] = src[x, y]
            else:
                dst[x, y] = background
    return out


@dataclass(slots=True)
class BlackInkGlyphSegment:
    """One glyph cell in a masked printed vertical column."""

    index: int
    y0: int
    y1: int
    x0: int
    x1: int
    ink_pixels: int
    image: Image.Image
    anchor_text: str = ""
    anchor_confidence: float = 0.0
    segmentation_source: str = "projection_valley_hybrid"


def _split_tall_band_by_valleys(
    binary: Sequence[int],
    w: int,
    y0: int,
    y1: int,
    target_pitch: float,
) -> list[tuple[int, int]]:
    """Split vertically fused glyphs only at a real low-ink valley.

    A tall solid glyph must never be divided merely because its bounding box is
    larger than the estimated pitch. The former unconditional split turned each
    filled test rectangle into two glyphs and could split bold kana.
    """
    height = y1 - y0
    if height <= target_pitch * 1.30:
        return [(y0, y1)]
    rows = [sum(binary[y * w:(y + 1) * w]) for y in range(y0, y1)]
    peak = max(rows, default=0)
    if peak <= 0:
        return [(y0, y1)]
    expected_parts = max(2, int(round(height / max(1.0, target_pitch))))
    cuts: list[int] = []
    previous = y0
    valley_limit = max(2, int(round(peak * 0.38)))
    for part in range(1, expected_parts):
        ideal = y0 + int(round(height * part / expected_parts))
        radius = max(3, int(round(target_pitch * 0.30)))
        minimum_piece = max(3, int(round(target_pitch * 0.34)))
        lo = max(previous + minimum_piece, ideal - radius)
        hi = min(y1 - minimum_piece, ideal + radius)
        if hi <= lo:
            continue
        candidates = list(range(lo, hi + 1))
        cut = min(
            candidates,
            key=lambda yy: (
                rows[yy - y0] + abs(yy - ideal) * 0.48,
                abs(yy - ideal),
            ),
        )
        if rows[cut - y0] > valley_limit:
            continue
        cuts.append(cut)
        previous = cut
    if not cuts:
        return [(y0, y1)]
    bounds = [y0, *cuts, y1]
    return [(a, b) for a, b in zip(bounds, bounds[1:]) if b - a >= 3]


def _split_slider_band_by_valleys(
    row_ink: Sequence[int],
    y0: int,
    y1: int,
    target_pitch: float,
) -> list[tuple[int, int]]:
    """Split a fused slider interval using only the isolated body-band ink.

    Side ruby, neighbour-column residue and page rules are intentionally absent
    from ``row_ink``.  The previous splitter summed the full raster again, so a
    tiny side annotation could fill an otherwise white inter-character valley
    and leave several printed characters inside one review frame.
    """
    y0 = max(0, int(y0))
    y1 = min(len(row_ink), int(y1))
    height = y1 - y0
    pitch = max(6.0, float(target_pitch))
    if height <= pitch * 1.20:
        return [(y0, y1)]

    values = [max(0, int(value)) for value in row_ink[y0:y1]]
    peak = max(values, default=0)
    if peak <= 0:
        return [(y0, y1)]

    # A 3-row minimum suppresses antialiasing specks but preserves true white
    # valleys.  Zero runs are preferred; low-ink valleys are accepted only when
    # they are close to the expected cell boundary.
    smooth: list[float] = []
    for index in range(len(values)):
        lo = max(0, index - 1)
        hi = min(len(values), index + 2)
        smooth.append(sum(values[lo:hi]) / max(1, hi - lo))

    expected_parts = max(2, int(round(height / pitch)))
    minimum_piece = max(3, int(round(pitch * 0.32)))
    radius = max(4, int(round(pitch * 0.36)))
    valley_limit = max(1.0, peak * 0.30)
    cuts: list[int] = []
    previous = y0

    for part in range(1, expected_parts):
        ideal = y0 + int(round(height * part / expected_parts))
        lo = max(previous + minimum_piece, ideal - radius)
        hi = min(y1 - minimum_piece, ideal + radius)
        if hi <= lo:
            continue
        candidates = list(range(lo, hi + 1))
        cut = min(
            candidates,
            key=lambda yy: (
                smooth[yy - y0] + abs(yy - ideal) * 0.10,
                abs(yy - ideal),
            ),
        )
        local_value = smooth[cut - y0]
        # For very tall bands, a clear relative minimum is enough even if one
        # antialiased pixel remains.  A solid bold glyph without a valley stays
        # intact because its minimum remains close to the local peak.
        window_values = smooth[max(0, lo - y0): min(len(smooth), hi - y0 + 1)]
        local_peak = max(window_values, default=peak)
        relative_valley = local_value <= max(1.0, local_peak * 0.34)
        if local_value > valley_limit and not relative_valley:
            continue
        cuts.append(cut)
        previous = cut

    if not cuts:
        return [(y0, y1)]
    bounds = [y0, *cuts, y1]
    return [(a, b) for a, b in zip(bounds, bounds[1:]) if b - a >= 3]


def _row_ink_runs(row_ink: Sequence[int], *, threshold: int) -> list[dict]:
    runs: list[dict] = []
    start: int | None = None
    mass = 0
    for y, value in enumerate(row_ink):
        if int(value) >= threshold:
            if start is None:
                start = y
                mass = 0
            mass += int(value)
        elif start is not None:
            runs.append({"y0": start, "y1": y, "mass": mass})
            start = None
            mass = 0
    if start is not None:
        runs.append({"y0": start, "y1": len(row_ink), "mass": mass})
    return runs


def _merge_runs_for_pitch(runs: list[dict], base_pitch: float) -> list[dict]:
    """Merge detached strokes that still belong to one printed character cell."""
    merged: list[dict] = []
    max_gap = max(2.0, base_pitch * 0.16)
    max_height = max(6.0, base_pitch * 1.16)
    for run in runs:
        if not merged:
            merged.append(dict(run))
            continue
        previous = merged[-1]
        gap = float(run["y0"] - previous["y1"])
        combined = float(run["y1"] - previous["y0"])
        tiny = int(run["mass"]) <= max(8, int(previous["mass"]) * 0.18)
        if gap <= max_gap and (combined <= max_height or tiny):
            previous["y1"] = run["y1"]
            previous["mass"] = int(previous["mass"]) + int(run["mass"])
        else:
            merged.append(dict(run))
    return merged


def _refine_main_body_band(
    binary: Sequence[int],
    w: int,
    h: int,
    band_info: MainBandMaskInfo,
) -> tuple[int, int]:
    """Return a narrower x band for the printed body text inside one column.

    ``mask_main_text_band`` removes obvious neighbour columns but can still leave
    side ruby/furigana inside the masked crop.  The main printed body is usually
    the widest/heaviest x run.  This helper keeps that run and only expands a
    little for punctuation-sized companions so the equal glyph grid is aligned to
    the real body text rather than the annotations.
    """
    if w < 4 or h < 4:
        return 0, w
    x_projection = [0] * w
    for idx, value in enumerate(binary):
        if value:
            x_projection[idx % w] += 1
    if max(x_projection, default=0) <= 0:
        return max(0, int(band_info.x0)), min(w, int(band_info.x1))

    active_threshold = max(1, int(round(h * 0.022)))
    active = [x for x, count in enumerate(x_projection) if count >= active_threshold]
    runs = _runs(active)
    if not runs:
        # Fall back to any faint but non-zero body trace.
        active = [x for x, count in enumerate(x_projection) if count > 0]
        runs = _runs(active)
    if not runs:
        return max(0, int(band_info.x0)), min(w, int(band_info.x1))

    components = _connected_components(binary, w, h)
    focus_centre = (
        (float(band_info.x0 + band_info.x1) / 2.0)
        if (band_info.x1 - band_info.x0) > 0 else (w / 2.0)
    )
    best: tuple[float, int, int] | None = None
    for x0, x1 in runs:
        width = max(1, x1 - x0)
        mass = sum(x_projection[x0:x1])
        run_centre = (x0 + x1) / 2.0
        component_mass = sum(
            int(comp.get('area', 0))
            for comp in components
            if float(comp['cx']) >= x0 - max(2.0, width * 0.18)
            and float(comp['cx']) <= x1 + max(2.0, width * 0.18)
        )
        centre_penalty = abs(run_centre - focus_centre) / max(1.0, w)
        width_bonus = min(1.0, width / max(1.0, w * 0.32))
        score = mass + component_mass * 0.72 + width * h * 0.06 + width_bonus * h * 0.20 - centre_penalty * h * 0.18
        if best is None or score > best[0]:
            best = (score, x0, x1)
    assert best is not None
    _, body_x0, body_x1 = best

    # Keep this as a *core* body band.  Do not absorb merely-near side
    # components here: vertical ruby/furigana can sit only one or two pixels
    # away from the body glyphs and, when it is included in the width estimate,
    # the inferred character pitch becomes almost two cells high.  That is the
    # direct cause of boxes such as ``波動`` / ``それ`` being emitted as one box.
    #
    # Components that genuinely cross the selected body run are still allowed
    # to extend it a little so antialiased outer strokes and detached dakuten are
    # not clipped.  Side-only components are left for ``classify_vertical_ruby``
    # to classify using this narrow seed band.
    core_x0, core_x1 = int(body_x0), int(body_x1)
    base_width = max(1, core_x1 - core_x0)
    outward_limit = max(2, int(round(base_width * 0.18)))
    for comp in components:
        comp_x0 = int(comp['x0'])
        comp_x1 = int(comp['x1'])
        overlap = max(0, min(core_x1, comp_x1) - max(core_x0, comp_x0))
        if overlap <= 0:
            continue
        comp_width = max(1, comp_x1 - comp_x0)
        centre = float(comp.get('cx', (comp_x0 + comp_x1) / 2.0))
        overlap_ratio = overlap / comp_width
        # A real body stroke either has its centre in the core run or overlaps
        # a meaningful fraction of it.  A side ruby glyph that only grazes the
        # edge must not widen the pitch-estimation band.
        if not (core_x0 <= centre <= core_x1 or overlap_ratio >= 0.28):
            continue
        body_x0 = min(body_x0, max(core_x0 - outward_limit, comp_x0))
        body_x1 = max(body_x1, min(core_x1 + outward_limit, comp_x1))

    padding = max(1, int(round(max(1, body_x1 - body_x0) * 0.08)))
    body_x0 = max(0, body_x0 - padding)
    body_x1 = min(w, body_x1 + padding)
    if body_x1 <= body_x0:
        return max(0, int(band_info.x0)), min(w, int(band_info.x1))
    return body_x0, body_x1


def _uniform_grid_pitch_and_phase(
    binary: Sequence[int],
    w: int,
    h: int,
    band_info: MainBandMaskInfo,
    *,
    expected_count: int | None = None,
) -> tuple[float, float, list[int], list[dict]]:
    """Infer one fixed print-cell pitch and one phase for a whole vertical column.

    Printed novel text is laid out on a regular character grid.  We therefore do
    not let every disconnected dot or punctuation fragment create a new glyph.
    Instead, a single pitch/phase is selected for the complete column and every
    dark pixel is assigned to exactly one equal-height cell.
    """
    row_ink = [sum(binary[y * w:(y + 1) * w]) for y in range(h)]
    active_rows = [y for y, value in enumerate(row_ink) if value > 0]
    if not active_rows:
        return 0.0, 0.0, row_ink, []

    band_width = max(6.0, float(max(1, band_info.x1 - band_info.x0)))
    base_pitch = max(8.0, min(float(h), band_width * 1.08))
    runs = _row_ink_runs(row_ink, threshold=max(1, int(round(w * 0.010))))
    groups = _merge_runs_for_pitch(runs, base_pitch)
    major_groups = [
        group for group in groups
        if (group["y1"] - group["y0"]) >= base_pitch * 0.22
        or int(group["mass"]) >= base_pitch * band_width * 0.025
    ]
    if len(major_groups) < 2:
        major_groups = groups

    total_ink_mass = max(1, sum(binary))
    # Row runs have already been merged across the short gaps normally found
    # inside one printed glyph. Keep punctuation-sized groups as grid evidence:
    # omitting them would make a legitimate Japanese period disappear into the
    # neighbouring full-size character cell. Only microscopic scan noise is
    # excluded.
    centre_groups = [
        group for group in groups
        if int(group.get("mass", 0)) >= max(3, int(round(total_ink_mass * 0.0025)))
        or (int(group["y1"]) - int(group["y0"])) >= 3
    ]
    centres = [
        (float(group["y0"] + group["y1"]) / 2.0, int(group["mass"]))
        for group in centre_groups
    ]
    candidates: set[float] = set()
    for factor in (0.78, 0.88, 0.96, 1.00, 1.06, 1.12, 1.22, 1.35, 1.50):
        candidates.add(round(base_pitch * factor, 3))

    # Distances between substantial printed glyphs often equal one or several
    # cell pitches.  Dividing by nearby integer multiples recovers the shared
    # pitch even when punctuation or paragraph blanks occur between them.
    for left_index, (left, _mass) in enumerate(centres):
        for right, _ in centres[left_index + 1:left_index + 6]:
            distance = right - left
            if distance <= 0:
                continue
            max_multiple = min(8, max(1, int(round(distance / max(4.0, base_pitch * 0.62)))))
            for multiple in range(1, max_multiple + 1):
                pitch = distance / multiple
                if base_pitch * 0.58 <= pitch <= base_pitch * 2.75:
                    candidates.add(round(pitch, 3))

    first_ink, last_ink = active_rows[0], active_rows[-1] + 1
    if expected_count and int(expected_count) > 0:
        hinted = (last_ink - first_ink + base_pitch * 0.45) / max(1, int(expected_count))
        if base_pitch * 0.58 <= hinted <= base_pitch * 2.75:
            for factor in (0.94, 1.0, 1.06):
                candidates.add(round(hinted * factor, 3))

    components = _connected_components(binary, w, h)
    total_ink = max(1, sum(binary))
    best: tuple[float, float, float] | None = None
    for raw_pitch in sorted(candidates):
        pitch = max(6.0, min(float(h), float(raw_pitch)))
        phase_steps = max(8, min(240, int(round(pitch))))
        for step in range(phase_steps):
            phase = pitch * step / phase_steps
            # Boundaries should pass through low-ink rows and must not bisect a
            # substantial connected component.  This chooses one global phase;
            # the boxes themselves remain perfectly uniform.
            boundary_cost = 0.0
            split_cost = 0.0
            first_k = int((first_ink - phase) // pitch) - 1
            last_k = int((last_ink - phase) // pitch) + 2
            boundaries: list[float] = []
            for k in range(first_k, last_k + 1):
                boundary = phase + k * pitch
                if 1 <= boundary < h - 1:
                    boundaries.append(boundary)
                    yy = int(round(boundary))
                    boundary_cost += (
                        row_ink[max(0, yy - 1)] * 0.45
                        + row_ink[yy] * 1.0
                        + row_ink[min(h - 1, yy + 1)] * 0.45
                    )
            edge_tolerance = max(2.0, pitch * 0.075)
            for comp in components:
                area = int(comp.get("area", 0))
                if area < 3:
                    continue
                for boundary in boundaries:
                    y0 = float(comp["y0"]); y1 = float(comp["y1"])
                    if y0 < boundary < y1:
                        # Printed strokes can touch a cell edge by a few pixels.
                        # Penalise deep cuts strongly, but allow a boundary that
                        # only grazes the antialiased outer edge of the glyph.
                        depth = min(boundary - y0, y1 - boundary)
                        ratio = min(1.0, max(0.0, depth / edge_tolerance))
                        split_cost += area * ratio * ratio
                        break
            # Row-run groups intentionally combine detached strokes that belong
            # to one printed glyph (dakuten, separate bars, hooks).  A boundary
            # through such a group is much more suspicious than an ordinary
            # low-ink valley and must not create an extra character slot.
            for group in groups:
                for boundary in boundaries:
                    y0 = float(group["y0"]); y1 = float(group["y1"])
                    if y0 < boundary < y1:
                        depth = min(boundary - y0, y1 - boundary)
                        ratio = min(1.0, max(0.0, depth / edge_tolerance))
                        split_cost += int(group.get("mass", 0)) * 1.35 * ratio * ratio
                        break

            occupied: dict[int, int] = {}
            for centre, mass in centres:
                cell = int((centre - phase) // pitch)
                occupied[cell] = occupied.get(cell, 0) + max(1, mass)
            collision_count = max(0, len(centres) - len(occupied))
            prior = abs(pitch - base_pitch) / max(1.0, base_pitch)
            score = (
                boundary_cost / total_ink
                + split_cost / total_ink * 1.65
                + collision_count * 1.25
                + prior * 0.035
            )
            if best is None or score < best[0]:
                best = (score, pitch, phase)

    if best is None:
        return base_pitch, 0.0, row_ink, groups
    return best[1], best[2], row_ink, groups


def _anchor_number(item: object, name: str, fallback: float = 0.0) -> float:
    try:
        if isinstance(item, dict):
            return float(item.get(name, fallback) or fallback)
        return float(getattr(item, name, fallback) or fallback)
    except Exception:
        return float(fallback)


def _anchor_text(item: object) -> str:
    try:
        value = item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "")
    except Exception:
        value = ""
    text = str(value or "").strip()
    return next(iter(text), "")


def _vision_character_cells(
    anchors: Sequence[object],
    *,
    w: int,
    h: int,
    body_x0: int,
    body_x1: int,
    binary: Sequence[int],
) -> tuple[list[dict], dict]:
    """Convert Apple Vision character boxes into weak top-to-bottom hints.

    Vision coordinates are normalized with a lower-left origin. The Swift helper
    has already undone the 90-degree rotation, so the boxes are in the original
    vertical-column coordinate system. These cells are never used as the final
    segmentation geometry: they only supply a pitch hint and optional candidate
    labels for the projection-valley segments.
    """
    body_width = max(1.0, float(body_x1 - body_x0))
    parsed: list[dict] = []
    for raw in anchors or []:
        text = _anchor_text(raw)
        if not text or text.isspace():
            continue
        x = _anchor_number(raw, "x")
        y = _anchor_number(raw, "y")
        width = _anchor_number(raw, "width")
        height = _anchor_number(raw, "height")
        confidence = max(0.0, min(1.0, _anchor_number(raw, "confidence")))
        if width <= 0.0 or height <= 0.0:
            continue
        x0 = max(0.0, min(float(w), x * w))
        x1 = max(0.0, min(float(w), (x + width) * w))
        y0 = max(0.0, min(float(h), (1.0 - y - height) * h))
        y1 = max(0.0, min(float(h), (1.0 - y) * h))
        if x1 <= x0 or y1 <= y0:
            continue
        overlap = max(0.0, min(x1, float(body_x1)) - max(x0, float(body_x0)))
        overlap_ratio = overlap / max(1.0, x1 - x0)
        centre_x = (x0 + x1) / 2.0
        # Side furigana is both narrow and outside the dominant body band.  A
        # full-size punctuation box may be narrow in ink, but its Vision box is
        # still centred on the body cell and therefore survives this filter.
        outside_body = centre_x < body_x0 - body_width * 0.12 or centre_x > body_x1 + body_width * 0.12
        ruby_sized = (x1 - x0) < body_width * 0.58
        if overlap_ratio < 0.24 and (outside_body or ruby_sized):
            continue
        parsed.append({
            "text": text,
            "confidence": confidence,
            "x0": x0,
            "x1": x1,
            "y0": y0,
            "y1": y1,
            "centre": (y0 + y1) / 2.0,
        })

    if not parsed:
        return [], {"anchor_count": 0, "inserted_slots": 0, "anchor_pitch": 0.0}
    parsed.sort(key=lambda item: (item["centre"], item["x0"]))

    prior_pitch = max(8.0, body_width * 1.06)
    box_heights = [max(1.0, item["y1"] - item["y0"]) for item in parsed]
    rough_candidates = [prior_pitch]
    rough_candidates.extend(value for value in box_heights if prior_pitch * 0.45 <= value <= prior_pitch * 1.85)
    rough_pitch = float(median(rough_candidates))

    body_row_ink = [
        sum(binary[y * w + body_x0:y * w + body_x1])
        for y in range(h)
    ]
    ink_runs = _row_ink_runs(body_row_ink, threshold=max(1, int(round(body_width * 0.015))))
    ink_groups = _merge_runs_for_pitch(ink_runs, rough_pitch)
    ink_centres = [
        (float(group["y0"] + group["y1"]) / 2.0)
        for group in ink_groups
        if (int(group["y1"]) - int(group["y0"])) >= max(3.0, rough_pitch * 0.22)
        or int(group.get("mass", 0)) >= max(3, int(round(body_width * rough_pitch * 0.02)))
    ]
    ink_diffs = [right - left for left, right in zip(ink_centres, ink_centres[1:]) if right > left]
    reliable_ink_pitch = 0.0
    if len(ink_diffs) >= 2:
        candidate_ink_pitch = float(median(ink_diffs))
        spread = max(abs(value - candidate_ink_pitch) for value in ink_diffs) / max(1.0, candidate_ink_pitch)
        if spread <= 0.28 and prior_pitch * 0.55 <= candidate_ink_pitch <= prior_pitch * 4.2:
            reliable_ink_pitch = candidate_ink_pitch
            rough_pitch = candidate_ink_pitch

    # Remove duplicate/near-identical boxes occasionally returned for the same
    # grapheme. Keep the more confident, more body-sized observation.
    deduped: list[dict] = []
    for item in parsed:
        if deduped and item["centre"] - deduped[-1]["centre"] < rough_pitch * 0.34:
            old = deduped[-1]
            old_score = old["confidence"] + min(1.0, (old["x1"] - old["x0"]) / body_width) * 0.12
            new_score = item["confidence"] + min(1.0, (item["x1"] - item["x0"]) / body_width) * 0.12
            if new_score > old_score:
                deduped[-1] = item
            continue
        deduped.append(item)
    parsed = deduped

    centres = [float(item["centre"]) for item in parsed]
    pitch_candidates = [prior_pitch]
    pitch_candidates.extend(max(1.0, item["y1"] - item["y0"]) for item in parsed)
    for left, right in zip(centres, centres[1:]):
        distance = right - left
        if distance <= rough_pitch * 0.42:
            continue
        multiple = max(1, int(round(distance / max(1.0, rough_pitch))))
        candidate = distance / multiple
        if prior_pitch * 0.52 <= candidate <= prior_pitch * 1.72:
            pitch_candidates.append(candidate)
    pitch = reliable_ink_pitch if reliable_ink_pitch > 0 else float(median(pitch_candidates))
    pitch = max(6.0, min(float(h), pitch))

    slots: list[dict] = []
    inserted = 0
    for index, item in enumerate(parsed):
        if not slots:
            slots.append(dict(item))
            continue
        previous = slots[-1]
        gap = float(item["centre"] - previous["centre"])
        multiple = max(1, int(round(gap / max(1.0, pitch))))
        if multiple > 1 and gap / multiple >= pitch * 0.58:
            for step in range(1, multiple):
                slots.append({
                    "text": "",
                    "confidence": 0.0,
                    "x0": float(body_x0),
                    "x1": float(body_x1),
                    "y0": 0.0,
                    "y1": 0.0,
                    "centre": previous["centre"] + gap * step / multiple,
                })
                inserted += 1
        slots.append(dict(item))

    # Use only body-band ink to decide whether characters exist before/after the
    # first/last Vision anchor. This recovers a missed leading punctuation or a
    # low-confidence final character without allowing side ruby to create slots.
    active_rows = [y for y, value in enumerate(body_row_ink) if value > 0]
    if active_rows:
        first_ink = float(active_rows[0])
        last_ink = float(active_rows[-1] + 1)
        while slots and slots[0]["centre"] - pitch * 0.62 > first_ink:
            first = slots[0]
            slots.insert(0, {
                "text": "", "confidence": 0.0,
                "x0": float(body_x0), "x1": float(body_x1),
                "y0": 0.0, "y1": 0.0,
                "centre": first["centre"] - pitch,
            })
            inserted += 1
        while slots and slots[-1]["centre"] + pitch * 0.62 < last_ink:
            last = slots[-1]
            slots.append({
                "text": "", "confidence": 0.0,
                "x0": float(body_x0), "x1": float(body_x1),
                "y0": 0.0, "y1": 0.0,
                "centre": last["centre"] + pitch,
            })
            inserted += 1

    slots.sort(key=lambda item: item["centre"])

    # Build non-overlapping hint cells around Vision character centres. Their
    # boundaries are used only for matching optional labels to the final
    # projection-valley segments, never as the final crop geometry.
    smooth_rows = _smooth_projection(body_row_ink, max(1, int(round(pitch * 0.025))))
    boundaries: list[int] = []
    first_boundary = int(round(float(slots[0]["centre"]) - pitch / 2.0))
    boundaries.append(max(0, min(h, first_boundary)))
    for left, right in zip(slots, slots[1:]):
        left_centre = float(left["centre"])
        right_centre = float(right["centre"])
        ideal = (left_centre + right_centre) / 2.0
        radius = max(2, int(round(pitch * 0.18)))
        lo = max(boundaries[-1] + 2, int(round(left_centre + pitch * 0.18)), int(round(ideal - radius)))
        hi = min(h - 2, int(round(right_centre - pitch * 0.18)), int(round(ideal + radius)))
        if hi < lo:
            boundary = int(round(ideal))
        else:
            boundary = min(
                range(lo, hi + 1),
                key=lambda yy: (smooth_rows[yy], abs(yy - ideal)),
            )
        boundaries.append(max(boundaries[-1] + 1, min(h, boundary)))
    last_boundary = int(round(float(slots[-1]["centre"]) + pitch / 2.0))
    boundaries.append(max(boundaries[-1] + 1, min(h, last_boundary)))

    cells: list[dict] = []
    for index, item in enumerate(slots):
        y0 = max(0, min(h, int(boundaries[index])))
        y1 = max(y0 + 1, min(h, int(boundaries[index + 1])))
        if y1 <= y0:
            continue
        # Guard against a pathological long cell even after gap filling.
        max_cell = max(8, int(round(pitch * 1.48)))
        if y1 - y0 > max_cell:
            midpoint = int(round(float(item["centre"])))
            y0 = max(0, midpoint - max_cell // 2)
            y1 = min(h, y0 + max_cell)
        cells.append({
            "index": len(cells),
            "y0": y0,
            "y1": y1,
            "anchor_text": str(item.get("text") or ""),
            "anchor_confidence": float(item.get("confidence") or 0.0),
        })
    return cells, {
        "anchor_count": len(parsed),
        "inserted_slots": inserted,
        "anchor_pitch": round(pitch, 3),
    }



def _smooth_projection(values: Sequence[int], radius: int) -> list[float]:
    if radius <= 0 or len(values) < 3:
        return [float(value) for value in values]
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + float(value))
    out: list[float] = []
    for index in range(len(values)):
        left = max(0, index - radius)
        right = min(len(values), index + radius + 1)
        out.append((prefix[right] - prefix[left]) / max(1, right - left))
    return out


def _median_number(values: Sequence[float], default: float) -> float:
    ordered = sorted(float(value) for value in values if float(value) > 0)
    if not ordered:
        return float(default)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _projection_pitch_candidates(
    row_ink: Sequence[int],
    groups: Sequence[dict],
    *,
    body_width: float,
    expected_count: int | None,
) -> list[float]:
    """Build plausible local character heights without committing to one grid."""
    active = [index for index, value in enumerate(row_ink) if int(value) > 0]
    if not active:
        return []
    first, last = active[0], active[-1] + 1
    base = max(8.0, body_width * 1.08)
    candidates: set[float] = set()
    for factor in (0.78, 0.86, 0.94, 1.00, 1.06, 1.14, 1.24, 1.38):
        candidates.add(round(base * factor, 3))

    centres = [
        (float(group['y0'] + group['y1']) / 2.0, int(group.get('mass', 0)))
        for group in groups
        if int(group.get('mass', 0)) >= 2
    ]
    # Distances between visible ink islands may be one or several cells.  Keep
    # all plausible integer divisions and let the valley optimiser decide.
    for left_index, (left, _mass) in enumerate(centres):
        for right, _ in centres[left_index + 1:left_index + 7]:
            distance = right - left
            if distance <= 0:
                continue
            for multiple in range(1, 7):
                pitch = distance / multiple
                if max(6.0, base * 0.50) <= pitch <= min(float(len(row_ink)), base * 1.90):
                    candidates.add(round(pitch, 3))

    if expected_count and int(expected_count) > 0:
        hinted = (last - first + base * 0.30) / max(1, int(expected_count))
        for factor in (0.90, 0.96, 1.0, 1.04, 1.10):
            if 6.0 <= hinted * factor <= len(row_ink):
                candidates.add(round(hinted * factor, 3))

    # A robust row autocorrelation adds a useful candidate when glyphs are
    # nearly touching and run centres are unreliable.
    smooth = _smooth_projection(row_ink, max(1, int(round(base * 0.035))))
    mean = sum(smooth[first:last]) / max(1, last - first)
    lo = max(6, int(round(base * 0.55)))
    hi = min(max(lo, int(round(base * 1.75))), max(lo, last - first - 1))
    best_corr: tuple[float, int] | None = None
    for lag in range(lo, hi + 1):
        score = 0.0
        norm_a = norm_b = 0.0
        for y in range(first, max(first, last - lag)):
            a = smooth[y] - mean
            b = smooth[y + lag] - mean
            score += a * b
            norm_a += a * a
            norm_b += b * b
        if norm_a <= 0 or norm_b <= 0:
            continue
        corr = score / ((norm_a * norm_b) ** 0.5)
        corr -= abs(lag - base) / max(1.0, base) * 0.06
        if best_corr is None or corr > best_corr[0]:
            best_corr = (corr, lag)
    if best_corr is not None and best_corr[0] > -0.15:
        candidates.add(float(best_corr[1]))

    raw_candidates = sorted(
        max(6.0, min(float(len(row_ink)), float(value)))
        for value in candidates
    )
    # The former implementation evaluated every distance-derived pitch. A long
    # novel column could produce dozens of near-duplicate candidates, and each
    # candidate runs a full dynamic-programming partition. Keep a compact,
    # geometrically diverse set around the body-width prior instead. This does
    # not change the selected segmentation family; it only removes redundant
    # work.
    minimum_separation = max(1.5, base * 0.035)
    selected: list[float] = []
    for value in sorted(raw_candidates, key=lambda item: (abs(item - base), item)):
        if any(abs(value - existing) < minimum_separation for existing in selected):
            continue
        selected.append(value)
        if len(selected) >= 14:
            break
    return sorted(selected)


def _projection_partition_python(
    row_ink: Sequence[int],
    *,
    first_ink: int,
    last_ink: int,
    target_height: float,
) -> tuple[list[int], float]:
    """Optimise one-dimensional cut lines so they pass through ink valleys.

    The recurrence and tie-breaking are identical to the original implementation,
    but all costs that depend only on a cut position or interval length are
    precomputed once. A long vertical column therefore avoids millions of
    repeated Python function calls while producing the same cut sequence.
    """
    h = len(row_ink)
    inf = float('inf')
    if h <= 1 or last_ink <= first_ink:
        return [], inf
    target = max(6.0, float(target_height))
    radius = max(1, int(round(target * 0.035)))
    smooth = _smooth_projection(row_ink, radius)
    positives = sorted(value for value in smooth[first_ink:last_ink] if value > 0)
    scale = positives[min(len(positives) - 1, int(round((len(positives) - 1) * 0.82)))] if positives else 1.0
    scale = max(1.0, float(scale))

    start_min = max(0, int(round(first_ink - target * 0.46)))
    start_max = max(start_min, min(first_ink, int(round(first_ink - target * 0.015))))
    end_min = min(h, max(last_ink, int(round(last_ink + target * 0.015))))
    end_max = min(h, int(round(last_ink + target * 0.46)))
    min_step = max(4, int(round(target * 0.52)))
    max_step = max(min_step + 1, int(round(target * 1.52)))

    prefix = [0.0]
    append_prefix = prefix.append
    running_total = 0.0
    for value in row_ink:
        running_total += float(value)
        append_prefix(running_total)

    # Preserve the old arithmetic order exactly so equal-cost candidates retain
    # the same first-minimum winner and therefore the same glyph boundaries.
    cut_costs = [0.0] * (end_max + 1)
    last_smooth_index = h - 1
    for y in range(1, min(h, end_max + 1)):
        left = smooth[y - 1]
        centre = smooth[y]
        right = smooth[y + 1] if y < last_smooth_index else smooth[last_smooth_index]
        cut_costs[y] = ((left * 0.55 + centre + right * 0.55) / (scale * 2.10)) * 3.2

    spacing_costs = [0.0] * (max_step + 1)
    low_spacing_limit = target * 0.70
    high_spacing_limit = target * 1.30
    target_denominator = max(1.0, target)
    for distance in range(min_step, max_step + 1):
        deviation = (distance - target) / target_denominator
        spacing = deviation * deviation * 2.75
        if distance < low_spacing_limit:
            spacing += ((low_spacing_limit - distance) / target) * 1.25
        elif distance > high_spacing_limit:
            spacing += ((distance - high_spacing_limit) / target) * 1.25
        spacing_costs[distance] = spacing

    dp = [inf] * (end_max + 1)
    parent = [-1] * (end_max + 1)
    segments_used = [0] * (end_max + 1)
    ideal_top_margin = target * 0.16
    edge_denominator = max(1.0, target * 0.28)
    for start in range(start_min, start_max + 1):
        margin = max(0.0, float(first_ink - start))
        edge = ((margin - ideal_top_margin) / edge_denominator) ** 2 * 0.20
        dp[start] = cut_costs[start] + edge

    empty_threshold = max(1.0, scale * 0.25)
    # Bind hot-loop structures locally. This is measurable for book pages with
    # hundreds of candidate cut positions and changes no recognition policy.
    dp_local = dp
    parent_local = parent
    segments_local = segments_used
    prefix_local = prefix
    cut_costs_local = cut_costs
    spacing_costs_local = spacing_costs
    for y in range(start_min + min_step, end_max + 1):
        p0 = max(start_min, y - max_step)
        p1 = y - min_step
        if p1 < p0:
            continue
        best_value = inf
        best_parent = -1
        cut_y = cut_costs_local[y]
        prefix_y = prefix_local[y]
        for previous in range(p0, p1 + 1):
            previous_cost = dp_local[previous]
            if previous_cost == inf:
                continue
            distance = y - previous
            empty_penalty = 0.32 if prefix_y - prefix_local[previous] <= empty_threshold else 0.0
            value = previous_cost + cut_y + spacing_costs_local[distance] + empty_penalty
            if value < best_value:
                best_value = value
                best_parent = previous
        if best_parent >= 0:
            dp_local[y] = best_value
            parent_local[y] = best_parent
            segments_local[y] = segments_local[best_parent] + 1

    ideal_bottom_margin = target * 0.16
    best_end = -1
    best_score = inf
    for end in range(end_min, end_max + 1):
        if end >= len(dp_local) or dp_local[end] == inf:
            continue
        margin = max(0.0, float(end - last_ink))
        edge = ((margin - ideal_bottom_margin) / edge_denominator) ** 2 * 0.20
        count = max(1, segments_local[end])
        score = dp_local[end] + edge + count * 0.015
        if score < best_score:
            best_score = score
            best_end = end
    if best_end < 0:
        return [], inf

    cuts = [best_end]
    cursor = best_end
    while cursor >= 0 and parent_local[cursor] >= 0:
        cursor = parent_local[cursor]
        cuts.append(cursor)
    cuts.reverse()
    if len(cuts) < 2:
        return [], inf
    return cuts, best_score / max(1, len(cuts) - 1)



def _projection_partition_numpy(
    row_ink: Sequence[int],
    *,
    first_ink: int,
    last_ink: int,
    target_height: float,
) -> tuple[list[int], float] | None:
    """Vectorised equivalent of ``_projection_partition_python``.

    The DP recurrence, candidate window, penalties and first-minimum tie break
    are unchanged. Only the inner scan over possible previous cuts is performed
    with NumPy float64 arrays. Return ``None`` when NumPy is unavailable so the
    historical implementation remains a complete fallback.
    """
    if os.environ.get("NOVEL_FORMATTER_DISABLE_FAST_PROJECTION", "").strip().lower() in {"1", "true", "yes"}:
        return None
    global _FAST_COMPONENTS_BACKEND, _FAST_COMPONENTS_IMPORT_ATTEMPTED
    if not _FAST_COMPONENTS_IMPORT_ATTEMPTED:
        _fast_connected_components([], 0, 0)
    if _FAST_COMPONENTS_BACKEND is None:
        return None
    _cv2, np = _FAST_COMPONENTS_BACKEND
    h = len(row_ink)
    if h <= 1 or last_ink <= first_ink:
        return [], float('inf')
    try:
        target = max(6.0, float(target_height))
        radius = max(1, int(round(target * 0.035)))
        smooth = _smooth_projection(row_ink, radius)
        positives = sorted(value for value in smooth[first_ink:last_ink] if value > 0)
        scale = positives[min(len(positives) - 1, int(round((len(positives) - 1) * 0.82)))] if positives else 1.0
        scale = max(1.0, float(scale))

        start_min = max(0, int(round(first_ink - target * 0.46)))
        start_max = max(start_min, min(first_ink, int(round(first_ink - target * 0.015))))
        end_min = min(h, max(last_ink, int(round(last_ink + target * 0.015))))
        end_max = min(h, int(round(last_ink + target * 0.46)))
        min_step = max(4, int(round(target * 0.52)))
        max_step = max(min_step + 1, int(round(target * 1.52)))

        row_array = np.asarray(row_ink, dtype=np.float64)
        prefix = np.empty(h + 1, dtype=np.float64)
        prefix[0] = 0.0
        np.cumsum(row_array, out=prefix[1:])
        smooth_array = np.asarray(smooth, dtype=np.float64)
        cut_costs = np.zeros(end_max + 1, dtype=np.float64)
        if h > 1:
            y_values = np.arange(1, min(h, end_max + 1), dtype=np.int32)
            left = smooth_array[np.maximum(0, y_values - 1)]
            center = smooth_array[y_values]
            right = smooth_array[np.minimum(h - 1, y_values + 1)]
            cut_costs[y_values] = ((left * 0.55 + center + right * 0.55) / (scale * 2.10)) * 3.2

        inf = float('inf')
        dp = np.full(end_max + 1, inf, dtype=np.float64)
        parent = np.full(end_max + 1, -1, dtype=np.int32)
        segments_used = np.zeros(end_max + 1, dtype=np.int32)
        ideal_top_margin = target * 0.16
        starts = np.arange(start_min, start_max + 1, dtype=np.int32)
        if starts.size:
            margins = np.maximum(0.0, first_ink - starts.astype(np.float64))
            edges = ((margins - ideal_top_margin) / max(1.0, target * 0.28)) ** 2 * 0.20
            dp[starts] = cut_costs[starts] + edges

        all_indices = np.arange(end_max + 1, dtype=np.int32)
        for y in range(start_min + min_step, end_max + 1):
            p0 = max(start_min, y - max_step)
            p1 = y - min_step
            if p1 < p0:
                continue
            previous = all_indices[p0:p1 + 1]
            valid_mask = np.isfinite(dp[previous])
            if not bool(np.any(valid_mask)):
                continue
            previous = previous[valid_mask]
            distances = y - previous
            deviation = (distances.astype(np.float64) - target) / max(1.0, target)
            spacing = deviation * deviation * 2.75
            low_mask = distances < target * 0.70
            high_mask = distances > target * 1.30
            spacing = spacing + np.where(
                low_mask, ((target * 0.70 - distances) / target) * 1.25, 0.0,
            )
            spacing = spacing + np.where(
                high_mask, ((distances - target * 1.30) / target) * 1.25, 0.0,
            )
            ink_mass = prefix[y] - prefix[previous]
            empty_penalty = np.where(ink_mass <= max(1.0, scale * 0.25), 0.32, 0.0)
            values = dp[previous] + cut_costs[y] + spacing + empty_penalty
            local_index = int(np.argmin(values))
            best_parent = int(previous[local_index])
            dp[y] = float(values[local_index])
            parent[y] = best_parent
            segments_used[y] = int(segments_used[best_parent]) + 1

        ideal_bottom_margin = target * 0.16
        best_end = -1
        best_score = inf
        for end in range(end_min, end_max + 1):
            if end >= len(dp) or not bool(np.isfinite(dp[end])):
                continue
            margin = max(0.0, float(end - last_ink))
            edge = ((margin - ideal_bottom_margin) / max(1.0, target * 0.28)) ** 2 * 0.20
            count = max(1, int(segments_used[end]))
            score = float(dp[end]) + edge + count * 0.015
            if score < best_score:
                best_score = score
                best_end = end
        if best_end < 0:
            return [], inf

        cuts = [best_end]
        cursor = best_end
        while cursor >= 0 and int(parent[cursor]) >= 0:
            cursor = int(parent[cursor])
            cuts.append(cursor)
        cuts.reverse()
        if len(cuts) < 2:
            return [], inf
        return cuts, best_score / max(1, len(cuts) - 1)
    except Exception:
        return None


def _projection_partition(
    row_ink: Sequence[int],
    *,
    first_ink: int,
    last_ink: int,
    target_height: float,
) -> tuple[list[int], float]:
    # The precomputed pure-Python recurrence is faster for the short rolling
    # windows used by printed Japanese columns. Keep NumPy as an explicit
    # diagnostic opt-in instead of paying array-allocation overhead by default.
    enable_numpy = os.environ.get("NOVEL_FORMATTER_ENABLE_NUMPY_PROJECTION", "").strip().lower() in {"1", "true", "yes"}
    if enable_numpy:
        fast = _projection_partition_numpy(
            row_ink, first_ink=first_ink, last_ink=last_ink,
            target_height=target_height,
        )
        if fast is not None:
            return fast
    return _projection_partition_python(
        row_ink, first_ink=first_ink, last_ink=last_ink,
        target_height=target_height,
    )


def _split_ambiguous_projection_intervals(
    intervals: list[tuple[int, int]],
    row_ink: Sequence[int],
    *,
    target_height: float,
) -> tuple[list[tuple[int, int]], int, int]:
    """Repair locally under-segmented intervals.

    A tall interval is first split at a genuine projection valley close to the
    expected character boundary.  When no deep valley exists (bold printing or
    touching glyphs), only that local interval falls back to an approximately
    equal split.  The old whole-column pitch/phase grid is never reintroduced.
    """
    out: list[tuple[int, int]] = []
    valley_splits = 0
    local_grid_splits = 0
    target = max(6.0, float(target_height))
    smooth = _smooth_projection(row_ink, max(1, int(round(target * 0.028))))

    for interval_index, (y0, y1) in enumerate(intervals):
        height = y1 - y0
        if height <= target * 1.40:
            out.append((y0, y1))
            continue

        desired_parts = max(2, int(round(height / target)))
        # Do not invent an excessive number of characters from one damaged
        # interval.  Very long spans will be revisited by later passes.
        desired_parts = min(desired_parts, max(2, int(height // max(4.0, target * 0.48))))
        minimum_piece = max(3, int(round(target * 0.43)))
        bounds = [y0]
        previous = y0

        for part in range(1, desired_parts):
            remaining = desired_parts - part
            ideal = y0 + height * part / desired_parts
            search_radius = max(3, int(round(target * 0.34)))
            lo = max(previous + minimum_piece, int(round(ideal - search_radius)))
            hi = min(y1 - remaining * minimum_piece, int(round(ideal + search_radius)))
            if hi < lo:
                # Geometry is too tight for the requested number of pieces.
                # Keep a deterministic local boundary rather than abandoning the
                # complete tall interval as a double-character box.
                cut = max(previous + minimum_piece, min(y1 - remaining * minimum_piece, int(round(ideal))))
                cut = max(previous + 1, min(y1 - 1, cut))
                bounds.append(cut)
                previous = cut
                local_grid_splits += 1
                continue

            local_lo = max(y0, int(round(ideal - target * 0.46)))
            local_hi = min(y1, int(round(ideal + target * 0.46)) + 1)
            local_values = [float(smooth[yy]) for yy in range(local_lo, local_hi)]
            local_median = _median_number([value for value in local_values if value > 0], 0.0)
            local_peak = max(local_values, default=0.0)

            def valley_key(yy: int) -> tuple[float, float]:
                ink = (
                    smooth[max(0, yy - 1)] * 0.35
                    + smooth[yy]
                    + smooth[min(len(smooth) - 1, yy + 1)] * 0.35
                ) / 1.70
                distance = abs(yy - ideal) / max(1.0, target)
                return (ink + distance * max(0.35, local_median * 0.20), distance)

            valley_cut = min(range(lo, hi + 1), key=valley_key)
            valley_value = float(smooth[valley_cut])
            deep_valley = (
                local_median <= 0.0
                or valley_value <= local_median * 0.70
                or (local_peak > 0.0 and valley_value <= local_peak * 0.30)
            )
            if deep_valley:
                cut = valley_cut
                valley_splits += 1
            else:
                # Local equal-grid fallback: choose the least-ink row only in a
                # narrow window around the ideal boundary. This prevents a weak
                # distant valley from producing one huge and one tiny box.
                narrow_radius = max(1, int(round(target * 0.12)))
                fallback_lo = max(lo, int(round(ideal)) - narrow_radius)
                fallback_hi = min(hi, int(round(ideal)) + narrow_radius)
                cut = min(
                    range(fallback_lo, fallback_hi + 1),
                    key=lambda yy: (smooth[yy], abs(yy - ideal)),
                )
                local_grid_splits += 1

            cut = max(previous + 1, min(y1 - 1, int(cut)))
            bounds.append(cut)
            previous = cut

        bounds.append(y1)
        out.extend((a, b) for a, b in zip(bounds, bounds[1:]) if b - a >= 3)

    return out, valley_splits, local_grid_splits



def _looks_like_trailing_punctuation_component(
    component: dict,
    *,
    target_height: float,
    body_width: float,
    median_area: float,
) -> bool:
    """Conservatively recognise a compact Japanese punctuation ink island.

    This is deliberately geometry-only. It is used only to keep a sentence-final
    ``。``/``、``-sized island from being reattached to the preceding full glyph;
    it does not emit or guess a symbol character.
    """
    comp_w = max(1.0, float(component.get('x1', 0)) - float(component.get('x0', 0)))
    comp_h = max(1.0, float(component.get('y1', 0)) - float(component.get('y0', 0)))
    area = max(1.0, float(component.get('area', 0)))
    box_area = max(1.0, comp_w * comp_h)
    fill = area / box_area
    return (
        comp_h <= max(5.0, target_height * 0.42)
        and comp_w <= max(5.0, body_width * 0.58)
        and area <= max(10.0, median_area * 0.38)
        and 0.08 <= fill <= 1.0
    )


def _split_trailing_punctuation_intervals(
    intervals: list[tuple[int, int]],
    components_by_interval: dict[int, list[dict]],
    *,
    target_height: float,
    body_width: float,
    median_area: float,
) -> tuple[list[tuple[int, int]], dict[int, list[dict]], int]:
    """Detach a close sentence-final punctuation island from its previous glyph.

    Projection DP sometimes keeps a very close ``字。`` pair in one interval
    because the period occupies only the top corner of its own em cell. We split
    only when a compact low-area component trails a clearly full-size body glyph
    by roughly half a character advance. Dakuten/handakuten are normally above or
    vertically overlapping their base glyph, so this trailing-only rule leaves
    them attached.
    """
    if not intervals:
        return intervals, components_by_interval, 0
    target = max(6.0, float(target_height))
    new_intervals: list[tuple[int, int]] = []
    new_mapping: dict[int, list[dict]] = {}
    split_count = 0

    for old_index, (cell_y0, cell_y1) in enumerate(intervals):
        items = list(components_by_interval.get(old_index, []))
        best_choice: tuple[float, dict, list[dict], int] | None = None
        if len(items) >= 2:
            for candidate in items:
                if not _looks_like_trailing_punctuation_component(
                    candidate,
                    target_height=target,
                    body_width=body_width,
                    median_area=median_area,
                ):
                    continue
                remaining = [item for item in items if item is not candidate]
                remaining_area = sum(float(item.get('area', 0)) for item in remaining)
                candidate_area = max(1.0, float(candidate.get('area', 0)))
                if remaining_area < max(candidate_area * 2.2, median_area * 0.45):
                    continue
                main_y0 = min(float(item['y0']) for item in remaining)
                main_y1 = max(float(item['y1']) for item in remaining)
                main_height = main_y1 - main_y0
                if main_height < target * 0.42:
                    continue
                main_cy = sum(float(item.get('cy', 0.0)) * max(1.0, float(item.get('area', 0))) for item in remaining) / max(1.0, remaining_area)
                candidate_cy = float(candidate.get('cy', 0.0))
                # Reading direction is top -> bottom. The punctuation centre must
                # be substantially after the preceding glyph centre.
                advance = candidate_cy - main_cy
                if advance < target * 0.40:
                    continue
                if float(candidate.get('y0', 0)) < main_cy + target * 0.15:
                    continue
                if float(candidate.get('y1', 0)) > cell_y1 + 1:
                    continue
                score = advance + candidate_area / max(1.0, median_area) * target * 0.10
                cut = int(round((main_y1 + float(candidate.get('y0', 0))) / 2.0))
                if float(candidate.get('y0', 0)) <= main_y1:
                    cut = int(round((main_cy + candidate_cy) / 2.0))
                cut = max(cell_y0 + 2, min(cell_y1 - 2, cut))
                if not (main_y0 < cut <= float(candidate.get('y1', 0))):
                    continue
                choice = (score, candidate, remaining, cut)
                if best_choice is None or choice[0] > best_choice[0]:
                    best_choice = choice

        if best_choice is None:
            new_index = len(new_intervals)
            new_intervals.append((cell_y0, cell_y1))
            if items:
                new_mapping[new_index] = items
            continue

        _score, punctuation, main_items, cut = best_choice
        first_index = len(new_intervals)
        new_intervals.append((cell_y0, cut))
        new_mapping[first_index] = main_items
        second_index = len(new_intervals)
        new_intervals.append((cut, cell_y1))
        new_mapping[second_index] = [punctuation]
        split_count += 1

    return new_intervals, new_mapping, split_count



_OCR_GUIDED_COMPACT_PUNCTUATION = set("。、，．・！？!?…‥：:；;｡､")
_OCR_GUIDED_SMALL_KANA = set("ぁぃぅぇぉゃゅょっゎァィゥェォャュョッヮヵヶ")


def _ocr_guided_box_number(raw: object, name: str, default: float = 0.0) -> float:
    try:
        if isinstance(raw, dict):
            value = raw.get(name, default)
        else:
            value = getattr(raw, name, default)
        return float(default if value is None else value)
    except Exception:
        return float(default)


def _ocr_guided_tight_box(
    binary: Sequence[int],
    *,
    width: int,
    height: int,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
) -> tuple[int, int, int, int, int] | None:
    """Return the exact black-ink bounds inside one candidate cell."""
    x0 = max(0, min(width, int(x0)))
    x1 = max(0, min(width, int(x1)))
    y0 = max(0, min(height, int(y0)))
    y1 = max(0, min(height, int(y1)))
    if x1 <= x0 or y1 <= y0:
        return None
    ink = 0
    min_x = x1
    max_x = x0 - 1
    min_y = y1
    max_y = y0 - 1
    for yy in range(y0, y1):
        offset = yy * width
        for xx in range(x0, x1):
            if not binary[offset + xx]:
                continue
            ink += 1
            if xx < min_x:
                min_x = xx
            if xx > max_x:
                max_x = xx
            if yy < min_y:
                min_y = yy
            if yy > max_y:
                max_y = yy
    if ink <= 0 or max_x < min_x or max_y < min_y:
        return None
    return min_x, max_x + 1, min_y, max_y + 1, ink


def _ocr_guided_split_box_ranges(
    binary: Sequence[int],
    *,
    width: int,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    pitch: float,
    minimum_height_ratio: float = 1.45,
) -> list[tuple[int, int]]:
    """Split one clearly over-height cached box only at strong ink valleys.

    This is deliberately stricter than the ordinary projection splitter.  It is
    a repair pass for a cached ``座の``-style two-character box, not a general
    license to divide tall kanji.  A solid glyph with no genuine valley remains
    untouched.
    """
    box_height = max(0, int(y1) - int(y0))
    target = max(6.0, float(pitch))
    if box_height < target * max(1.20, float(minimum_height_ratio)):
        return [(int(y0), int(y1))]
    desired_parts = max(2, int(round(box_height / target)))
    desired_parts = min(4, desired_parts)
    rows: list[int] = []
    for yy in range(int(y0), int(y1)):
        offset = yy * width
        rows.append(sum(1 for xx in range(int(x0), int(x1)) if binary[offset + xx]))
    peak = max(rows, default=0)
    positive = sorted(value for value in rows if value > 0)
    if peak <= 0 or not positive:
        return [(int(y0), int(y1))]
    median_positive = float(positive[len(positive) // 2])
    minimum_piece = max(3, int(round(target * 0.30)))
    bounds = [int(y0)]
    previous = int(y0)
    for part in range(1, desired_parts):
        remaining = desired_parts - part
        ideal = int(round(int(y0) + box_height * part / desired_parts))
        radius = max(3, int(round(target * 0.28)))
        lo = max(previous + minimum_piece, ideal - radius)
        hi = min(int(y1) - remaining * minimum_piece, ideal + radius)
        if hi < lo:
            return [(int(y0), int(y1))]

        def cut_key(yy: int) -> tuple[float, float]:
            local = yy - int(y0)
            left = rows[max(0, local - 1)]
            centre = rows[local]
            right = rows[min(len(rows) - 1, local + 1)]
            smoothed = (left * 0.35 + centre + right * 0.35) / 1.70
            return smoothed + abs(yy - ideal) * 0.20, abs(yy - ideal)

        cut = min(range(lo, hi + 1), key=cut_key)
        local = cut - int(y0)
        valley = (
            rows[max(0, local - 1)] * 0.35
            + rows[local]
            + rows[min(len(rows) - 1, local + 1)] * 0.35
        ) / 1.70
        # Require a real gap/valley.  This protects a single tall kanji or bold
        # kana from being split just because its tight bounding box is large.
        if valley > max(1.5, peak * 0.34) and valley > median_positive * 0.56:
            return [(int(y0), int(y1))]
        bounds.append(int(cut))
        previous = int(cut)
    bounds.append(int(y1))
    pieces = [(a, b) for a, b in zip(bounds, bounds[1:]) if b - a >= 3]
    return pieces if len(pieces) == desired_parts else [(int(y0), int(y1))]


def _ocr_guided_assignment_cost(
    box: dict,
    character: str,
    *,
    pitch: float,
    median_ink: float,
) -> float:
    height_ratio = max(0.01, (float(box['y1']) - float(box['y0'])) / max(1.0, pitch))
    ink_ratio = max(0.0, float(box.get('ink_pixels', 0)) / max(1.0, median_ink))
    if character in _OCR_GUIDED_COMPACT_PUNCTUATION:
        cost = abs(min(height_ratio, 0.72) - 0.38) * 0.45
        if height_ratio > 1.20:
            cost += 5.0 + (height_ratio - 1.20) * 5.0
        if ink_ratio > 1.55:
            cost += min(2.0, (ink_ratio - 1.55) * 0.75)
        return cost
    if character in _OCR_GUIDED_SMALL_KANA:
        cost = abs(height_ratio - 0.72) * 0.55
        if height_ratio < 0.24:
            cost += 4.0
        elif height_ratio < 0.38:
            cost += 1.2
        if height_ratio > 1.42:
            cost += 5.0 + (height_ratio - 1.42) * 4.0
        if ink_ratio < 0.06:
            cost += 4.0
        return cost

    cost = abs(height_ratio - 0.88) * 0.65
    if height_ratio < 0.30:
        cost += 8.0
    elif height_ratio < 0.47:
        cost += 3.0
    elif height_ratio < 0.58:
        cost += 0.8
    if height_ratio > 1.48:
        cost += 8.0 + (height_ratio - 1.48) * 5.0
    elif height_ratio > 1.28:
        cost += 1.5
    if ink_ratio < 0.06:
        cost += 8.0
    elif ink_ratio < 0.18:
        cost += 3.0
    elif ink_ratio < 0.30:
        cost += 0.8
    return cost


def _ocr_guided_drop_cost(box: dict, *, pitch: float, median_ink: float) -> float:
    height_ratio = max(0.01, (float(box['y1']) - float(box['y0'])) / max(1.0, pitch))
    ink_ratio = max(0.0, float(box.get('ink_pixels', 0)) / max(1.0, median_ink))
    if height_ratio <= 0.20 and ink_ratio <= 0.08:
        return 0.05
    if height_ratio <= 0.34 and ink_ratio <= 0.18:
        return 0.20
    if height_ratio <= 0.48 and ink_ratio <= 0.32:
        return 0.85
    return 6.0 + min(4.0, height_ratio + ink_ratio * 0.45)


def repair_precomputed_glyph_boxes_with_ocr_text(
    source: Image.Image,
    boxes: Sequence[object],
    expected_text: str,
) -> tuple[list[dict], dict]:
    """Repair cached glyph geometry using OCR only as a sequence-length/type hint.

    The OCR characters are *not* accepted here.  They only describe how many
    cells should exist and whether each position is a normal glyph or compact
    punctuation.  The pass can split a clear two-character cached box and drop
    an equally clear dust/ruby fragment, which fixes local offset even when the
    total cached box count already happens to equal the OCR character count.
    """
    expected = list(str(expected_text or '').replace('\n', '').replace('\r', ''))
    original_boxes = [dict(item) for item in boxes if isinstance(item, dict)]
    diagnostics = {
        'ocr_guided_box_repair_attempted': False,
        'ocr_guided_box_repair_selected': False,
        'ocr_guided_box_repair_reason': '',
        'ocr_guided_expected_count': len(expected),
        'ocr_guided_original_box_count': 0,
        'ocr_guided_split_boxes': 0,
        'ocr_guided_dropped_boxes': 0,
    }
    if len(expected) < 2 or not boxes:
        diagnostics['ocr_guided_box_repair_reason'] = 'missing_sequence_or_boxes'
        return original_boxes, diagnostics

    width, height = source.size
    parsed: list[dict] = []
    pitch_hints: list[float] = []
    for raw in boxes:
        if not isinstance(raw, dict):
            continue
        x0 = max(0, min(width, int(round(_ocr_guided_box_number(raw, 'x0', 0.0)))))
        x1 = max(0, min(width, int(round(_ocr_guided_box_number(raw, 'x1', width)))))
        y0 = max(0, min(height, int(round(_ocr_guided_box_number(raw, 'y0', 0.0)))))
        y1 = max(0, min(height, int(round(_ocr_guided_box_number(raw, 'y1', height)))))
        if x1 <= x0 or y1 <= y0:
            continue
        item = dict(raw)
        item.update({'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1})
        item['cell_y0'] = max(0, min(height, int(round(_ocr_guided_box_number(raw, 'cell_y0', y0)))))
        item['cell_y1'] = max(0, min(height, int(round(_ocr_guided_box_number(raw, 'cell_y1', y1)))))
        raw_pitch = _ocr_guided_box_number(raw, 'target_pitch', 0.0)
        if raw_pitch > 0:
            pitch_hints.append(raw_pitch)
        parsed.append(item)
    parsed.sort(key=lambda item: (int(item['y0']), int(item['y1'])))
    diagnostics['ocr_guided_original_box_count'] = len(parsed)
    if not parsed:
        diagnostics['ocr_guided_box_repair_reason'] = 'no_valid_boxes'
        return [], diagnostics

    cell_heights = [
        max(1.0, float(item.get('cell_y1', item['y1'])) - float(item.get('cell_y0', item['y0'])))
        for item in parsed
    ]
    tight_heights = [max(1.0, float(item['y1']) - float(item['y0'])) for item in parsed]
    pitch = float(median(pitch_hints)) if pitch_hints else float(median(cell_heights or tight_heights))
    pitch = max(6.0, pitch)

    # Normal columns take the zero-copy cached path.  Only inspect pixels when
    # the preview count differs from OCR or raw geometry already contains a
    # clearly over-height/tiny cell.  This keeps the repair from becoming a new
    # per-column performance tax.
    raw_tall = any(
        max(1.0, float(item['y1']) - float(item['y0'])) >= pitch * 1.45
        or max(1.0, float(item.get('cell_y1', item['y1'])) - float(item.get('cell_y0', item['y0']))) >= pitch * 1.72
        for item in parsed
    )
    raw_tiny = any(
        max(1.0, float(item['y1']) - float(item['y0'])) <= pitch * 0.24
        and float(item.get('ink_pixels', 0) or 0) <= max(16.0, pitch * pitch * 0.025)
        for item in parsed
    )
    if len(parsed) == len(expected) and not raw_tall and not raw_tiny:
        diagnostics['ocr_guided_box_repair_reason'] = 'geometry_already_consistent_fast_path'
        return original_boxes, diagnostics

    gray = source.convert('L')
    try:
        values = _pixel_values(gray)
    finally:
        gray.close()
    threshold = _otsu(values)
    binary = [1 if value < threshold else 0 for value in values]

    candidates: list[dict] = []
    split_count = 0
    count_deficit = len(parsed) < len(expected)
    split_height_ratio = 1.28 if count_deficit else 1.45
    for item in parsed:
        x0, x1, y0, y1 = int(item['x0']), int(item['x1']), int(item['y0']), int(item['y1'])
        tight = _ocr_guided_tight_box(
            binary, width=width, height=height, x0=x0, x1=x1, y0=y0, y1=y1,
        )
        if tight is None:
            continue
        tx0, tx1, ty0, ty1, ink = tight
        item.update({'x0': tx0, 'x1': tx1, 'y0': ty0, 'y1': ty1, 'ink_pixels': ink})
        box_height = max(1.0, float(ty1 - ty0))
        cell_height = max(1.0, float(item.get('cell_y1', ty1)) - float(item.get('cell_y0', ty0)))
        clearly_tall = box_height >= pitch * split_height_ratio or cell_height >= pitch * (1.54 if count_deficit else 1.72)
        ranges = _ocr_guided_split_box_ranges(
            binary, width=width, x0=tx0, x1=tx1, y0=ty0, y1=ty1, pitch=pitch,
            minimum_height_ratio=split_height_ratio,
        ) if clearly_tall else [(ty0, ty1)]
        pieces: list[dict] = []
        for part_y0, part_y1 in ranges:
            part = _ocr_guided_tight_box(
                binary, width=width, height=height,
                x0=tx0, x1=tx1, y0=part_y0, y1=part_y1,
            )
            if part is None:
                continue
            px0, px1, py0, py1, pink = part
            piece = dict(item)
            piece.update({
                'x0': px0, 'x1': px1, 'y0': py0, 'y1': py1,
                'cell_y0': int(part_y0), 'cell_y1': int(part_y1),
                'ink_pixels': int(pink), 'target_pitch': float(pitch),
                'source': 'ocr_guided_cached_local_split' if len(ranges) > 1 else str(item.get('source') or 'projection_valley_cached_preview'),
            })
            pieces.append(piece)
        if len(pieces) >= 2:
            split_count += len(pieces) - 1
            candidates.extend(pieces)
        else:
            candidates.append(item)

    diagnostics['ocr_guided_box_repair_attempted'] = bool(split_count or len(candidates) != len(expected))
    diagnostics['ocr_guided_split_boxes'] = int(split_count)
    if len(candidates) < len(expected):
        diagnostics['ocr_guided_box_repair_reason'] = 'not_enough_boxes_after_safe_splits'
        return original_boxes, diagnostics
    extra = len(candidates) - len(expected)
    if extra > max(3, int(round(len(expected) * 0.12))):
        diagnostics['ocr_guided_box_repair_reason'] = 'too_many_extra_boxes'
        return original_boxes, diagnostics

    ink_values = [
        float(item.get('ink_pixels', 0))
        for item in candidates
        if 0.42 <= (float(item['y1']) - float(item['y0'])) / pitch <= 1.35
        and float(item.get('ink_pixels', 0)) > 0
    ]
    if not ink_values:
        ink_values = [float(item.get('ink_pixels', 0)) for item in candidates if float(item.get('ink_pixels', 0)) > 0]
    median_ink = float(median(ink_values)) if ink_values else 1.0

    m, n = len(candidates), len(expected)
    inf = float('inf')
    dp = [[inf] * (n + 1) for _ in range(m + 1)]
    parent: list[list[tuple[int, int, str] | None]] = [[None] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0.0
    for i in range(m):
        for j in range(n + 1):
            current = dp[i][j]
            if current == inf:
                continue
            remaining_boxes = m - (i + 1)
            remaining_chars = n - j
            if remaining_boxes >= remaining_chars:
                dropped = current + _ocr_guided_drop_cost(candidates[i], pitch=pitch, median_ink=median_ink)
                if dropped < dp[i + 1][j]:
                    dp[i + 1][j] = dropped
                    parent[i + 1][j] = (i, j, 'drop')
            if j < n:
                assigned = current + _ocr_guided_assignment_cost(
                    candidates[i], expected[j], pitch=pitch, median_ink=median_ink,
                )
                if assigned < dp[i + 1][j + 1]:
                    dp[i + 1][j + 1] = assigned
                    parent[i + 1][j + 1] = (i, j, 'assign')

    if dp[m][n] == inf:
        diagnostics['ocr_guided_box_repair_reason'] = 'sequence_alignment_failed'
        return original_boxes, diagnostics
    selected_indices: list[int] = []
    dropped_indices: list[int] = []
    i, j = m, n
    while i > 0:
        step = parent[i][j]
        if step is None:
            diagnostics['ocr_guided_box_repair_reason'] = 'sequence_backtrack_failed'
            return original_boxes, diagnostics
        previous_i, previous_j, action = step
        if action == 'assign':
            selected_indices.append(i - 1)
        else:
            dropped_indices.append(i - 1)
        i, j = previous_i, previous_j
    selected_indices.reverse()
    dropped_indices.reverse()
    if len(selected_indices) != n:
        diagnostics['ocr_guided_box_repair_reason'] = 'selected_count_mismatch'
        return original_boxes, diagnostics

    # Never discard a full-size printed glyph merely to satisfy OCR length.  A
    # removable candidate must itself look like dust, detached ruby or a tiny
    # scan artefact.  Compact punctuation is protected by the assignment DP.
    if any(_ocr_guided_drop_cost(candidates[index], pitch=pitch, median_ink=median_ink) > 1.25 for index in dropped_indices):
        diagnostics['ocr_guided_box_repair_reason'] = 'drop_candidate_not_clear_noise'
        return original_boxes, diagnostics

    selected = [dict(candidates[index]) for index in selected_indices]
    assigned_costs = [
        _ocr_guided_assignment_cost(box, ch, pitch=pitch, median_ink=median_ink)
        for box, ch in zip(selected, expected)
    ]
    if any(cost >= 6.0 for cost in assigned_costs) or (sum(assigned_costs) / max(1, n)) > 1.80:
        diagnostics['ocr_guided_box_repair_reason'] = 'repaired_geometry_still_implausible'
        return original_boxes, diagnostics
    changed = bool(split_count or dropped_indices or len(parsed) != n)
    if not changed:
        diagnostics['ocr_guided_box_repair_reason'] = 'geometry_already_consistent'
        return original_boxes, diagnostics

    for index, item in enumerate(selected):
        item['index'] = index + 1
        item['target_pitch'] = float(pitch)
        item['source'] = str(item.get('source') or 'ocr_guided_cached_repair')
    diagnostics.update({
        'ocr_guided_box_repair_selected': True,
        'ocr_guided_box_repair_reason': 'safe_local_split_and_noise_alignment',
        'ocr_guided_dropped_boxes': len(dropped_indices),
        'ocr_guided_repaired_box_count': len(selected),
        'ocr_guided_pitch': round(float(pitch), 3),
        'ocr_guided_dropped_box_indices': [int(index) for index in dropped_indices],
    })
    return selected, diagnostics


def _segments_from_precomputed_boxes(
    source: Image.Image,
    boxes: Sequence[object],
) -> tuple[list[BlackInkGlyphSegment], dict]:
    """Rebuild glyph rasters from preview-time boxes without re-running layout.

    Live preview already performs the expensive ruby filtering, connected
    components and projection DP. Recognition can reuse those exact boxes and
    only rasterise their black pixels, cutting the second segmentation pass from
    every column.
    """
    w, h = source.size
    if w <= 0 or h <= 0:
        return [], {}
    background = estimate_paper_background(source)
    gray = source.convert('L')
    values = _pixel_values(gray)
    threshold = _otsu(values)
    threshold_lut = [255 if value < threshold else 0 for value in range(256)]
    segments: list[BlackInkGlyphSegment] = []
    valid_boxes: list[dict] = []
    pitch_hints: list[float] = []
    requested_box_count = sum(1 for raw in boxes if isinstance(raw, dict))
    for raw in boxes:
        if not isinstance(raw, dict):
            continue
        try:
            x0 = max(0, min(w, int(round(float(raw.get('x0', 0))))))
            x1 = max(0, min(w, int(round(float(raw.get('x1', w))))))
            y0 = max(0, min(h, int(round(float(raw.get('y0', 0))))))
            y1 = max(0, min(h, int(round(float(raw.get('y1', h))))))
        except Exception:
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        crop_height = y1 - y0
        # Preview boxes already passed ruby filtering and projection splitting.
        # Rasterise only ink inside that exact box using PIL's C-level crop/paste
        # instead of scanning every pixel in Python and running the same connected
        # component mask a second time. This preserves detached dakuten and small
        # punctuation that belong to the cached cell.
        rgb_crop = source.crop((x0, y0, x1, y1)).convert('RGB')
        gray_crop = gray.crop((x0, y0, x1, y1))
        ink_mask = gray_crop.point(threshold_lut, mode='L')
        copied = int(ink_mask.histogram()[255])
        gray_crop.close()
        if copied <= 0:
            ink_mask.close()
            rgb_crop.close()
            continue
        canvas = Image.new('RGB', (w, crop_height), background)
        canvas.paste(rgb_crop, (x0, 0), ink_mask)
        ink_mask.close()
        rgb_crop.close()
        source_name = str(raw.get('source') or 'projection_valley_cached')
        segment = BlackInkGlyphSegment(
            index=len(segments), y0=y0, y1=y1, x0=x0, x1=x1,
            ink_pixels=copied, image=canvas,
            anchor_text=str(raw.get('anchor_text') or ''),
            anchor_confidence=float(raw.get('anchor_confidence') or 0.0),
            segmentation_source=source_name,
        )
        segments.append(segment)
        raw_pitch = raw.get('target_pitch', raw.get('pitch', 0.0))
        try:
            pitch_value = float(raw_pitch or 0.0)
        except Exception:
            pitch_value = 0.0
        if pitch_value > 0:
            pitch_hints.append(pitch_value)
        valid_boxes.append({
            'index': segment.index, 'x0': x0, 'x1': x1,
            'y0': y0, 'y1': y1, 'ink_pixels': copied,
            'source': source_name,
            'cell_y0': int(raw.get('cell_y0', y0) or y0),
            'cell_y1': int(raw.get('cell_y1', y1) or y1),
            'target_pitch': pitch_value,
        })
    # Cached preview geometry is an optimization only. If even one preview box
    # became blank/invalid, or the boxes are no longer strictly top-to-bottom,
    # reject the entire cache and let the fresh projection splitter recover. A
    # partial cached result would silently drop or merge characters.
    rejection_reason = ""
    if requested_box_count <= 0 or len(valid_boxes) != requested_box_count:
        rejection_reason = "cached_box_count_or_ink_mismatch"
    else:
        for previous, current in zip(valid_boxes, valid_boxes[1:]):
            previous_height = max(1, int(previous['y1']) - int(previous['y0']))
            current_height = max(1, int(current['y1']) - int(current['y0']))
            overlap = int(previous['y1']) - int(current['y0'])
            if int(current['y0']) < int(previous['y0']):
                rejection_reason = "cached_boxes_not_reading_order"
                break
            if overlap > max(1, int(round(min(previous_height, current_height) * 0.20))):
                rejection_reason = "cached_boxes_excessive_overlap"
                break
    gray.close()
    if rejection_reason:
        for segment in segments:
            segment.image.close()
        return [], {
            'precomputed_boxes_used': False,
            'precomputed_boxes_rejected_reason': rejection_reason,
        }
    if not segments:
        return [], {}
    inferred_pitch = float(median(pitch_hints)) if pitch_hints else float(median([item.y1 - item.y0 for item in segments]))
    return segments, {
        'segmentation_mode': 'projection_valley_cached_preview',
        'threshold': threshold,
        'target_pitch': inferred_pitch,
        'projection_target_height': inferred_pitch,
        'projection_cuts': [],
        'projection_intervals': len(segments),
        'projection_local_valley_splits': 0,
        'projection_local_grid_splits': 0,
        'projection_local_fallback_splits': 0,
        'projection_component_slices': 0,
        'uniform_fallback_used': False,
        'center_fallback_used': False,
        'center_candidate_count': 0,
        'center_confidence': 0.0,
        'ruby_components_removed': 0,
        'ruby_filter_confidence': 0.0,
        'ruby_boxes': [],
        'grid_phase': 0.0,
        'grid_cells': len(segments),
        'blank_cells_skipped': 0,
        'components': 0,
        'groups': 0,
        'segments': len(segments),
        'segment_boxes': valid_boxes,
        'mask_applied': False,
        'mask_x0': min(item['x0'] for item in valid_boxes),
        'mask_x1': max(item['x1'] for item in valid_boxes),
        'main_band_x0': min(item['x0'] for item in valid_boxes),
        'main_band_x1': max(item['x1'] for item in valid_boxes),
        'removed_ink_pixels': 0,
        'expected_count_hint': 0,
        'vision_character_anchor_count': 0,
        'vision_anchor_pitch': 0.0,
        'precomputed_boxes_used': True,
    }




def _estimate_review_slider_pitch(
    intervals: Sequence[tuple[int, int]],
    *,
    first_ink: int,
    last_ink: int,
    body_width: int,
    expected_count: int | None,
) -> float:
    """Estimate one print-cell height without locking geometry to OCR text.

    The body-band width is the primary prior because Japanese vertical type is
    laid out on an approximately square em grid.  OCR length is only a weak
    sanity hint: it may reveal a clearly fused multi-character interval, but it
    can never force the final number of physical frames.  This distinction is
    important for genuine OCR omissions, which must remain visible as ``□``.
    """
    base = max(8.0, float(body_width) * 1.02)
    candidates: list[float] = [base, base]
    span = max(1.0, float(last_ink - first_ink))
    count = int(expected_count or 0)
    if count > 0:
        hinted = span / count
        # A weak hint is useful when a loose crop makes the x-band wider than one
        # actual glyph.  Reject implausible values so a wrong OCR count cannot
        # dictate the physical segmentation.
        if base * 0.46 <= hinted <= base * 1.72:
            candidates.extend([hinted, hinted])

    heights = sorted(
        float(y1 - y0) for y0, y1 in intervals
        if y1 > y0 and (y1 - y0) >= max(3.0, base * 0.20)
    )
    if heights:
        # Ink bounding boxes are normally smaller than the em cell.  The upper
        # half is therefore a better pitch clue than tiny kana/punctuation.
        upper = heights[len(heights) // 2 :]
        observed = median(upper) if upper else median(heights)
        observed_cell = observed / 0.78
        if base * 0.56 <= observed_cell <= base * 1.48:
            candidates.append(observed_cell)

    pitch = float(median(candidates))
    return max(base * 0.58, min(base * 1.38, pitch))


def _best_local_slider_cut(
    row_ink: Sequence[int],
    *,
    lo: int,
    hi: int,
    ideal: float,
) -> int:
    """Choose a deterministic low-ink cut near ``ideal``."""
    lo = max(1, int(lo))
    hi = min(len(row_ink) - 1, int(hi))
    if hi < lo:
        return max(1, min(len(row_ink) - 1, int(round(ideal))))
    radius = max(1.0, float(hi - lo))
    return min(
        range(lo, hi + 1),
        key=lambda y: (
            int(row_ink[y - 1]) * 0.45
            + int(row_ink[y])
            + int(row_ink[min(len(row_ink) - 1, y + 1)]) * 0.45
            + abs(float(y) - ideal) / radius * 0.18,
            abs(float(y) - ideal),
            y,
        ),
    )


def _split_oversized_review_slider_intervals(
    intervals: Sequence[tuple[int, int]],
    row_ink: Sequence[int],
    *,
    pitch: float,
    expected_count: int | None,
) -> tuple[list[tuple[int, int]], int]:
    """Guarantee that one clearly oversized frame cannot contain many glyphs.

    First use real projection valleys.  If bold/touching print leaves no clean
    valley, only the oversized local interval receives an equal-cell fallback;
    the rest of the column remains purely ink-driven.  OCR count may request an
    additional split only when the candidate interval is already physically
    larger than a normal cell.
    """
    pitch = max(6.0, float(pitch))
    output: list[tuple[int, int]] = []
    forced = 0
    for y0, y1 in intervals:
        height = y1 - y0
        parts = _split_slider_band_by_valleys(row_ink, y0, y1, pitch)
        if len(parts) == 1 and height > pitch * 1.52:
            desired = max(2, int(round(height / pitch)))
            minimum_piece = max(3, int(round(pitch * 0.42)))
            bounds = [y0]
            previous = y0
            for part in range(1, desired):
                remaining = desired - part
                ideal = y0 + height * part / desired
                radius = max(3, int(round(pitch * 0.32)))
                lo = max(previous + minimum_piece, int(round(ideal - radius)))
                hi = min(y1 - remaining * minimum_piece, int(round(ideal + radius)))
                if hi < lo:
                    cut = max(previous + 1, min(y1 - 1, int(round(ideal))))
                else:
                    cut = _best_local_slider_cut(row_ink, lo=lo, hi=hi, ideal=ideal)
                if cut <= previous or cut >= y1:
                    continue
                bounds.append(cut)
                previous = cut
            bounds.append(y1)
            candidate = [(a, b) for a, b in zip(bounds, bounds[1:]) if b - a >= 3]
            if len(candidate) >= 2:
                parts = candidate
                forced += len(candidate) - 1
        output.extend(parts)

    target = int(expected_count or 0)
    # OCR count is never a target for the complete physical geometry.  It is used
    # only to expose a missed boundary inside an interval that is already too tall
    # to be a normal print cell.
    while target > 0 and len(output) < target:
        candidates = [
            (b - a, index, a, b)
            for index, (a, b) in enumerate(output)
            if (b - a) > pitch * 1.34
        ]
        if not candidates:
            break
        _height, index, y0, y1 = max(candidates)
        ideal = (y0 + y1) / 2.0
        radius = max(3, int(round(pitch * 0.28)))
        cut = _best_local_slider_cut(
            row_ink,
            lo=max(y0 + 3, int(round(ideal - radius))),
            hi=min(y1 - 3, int(round(ideal + radius))),
            ideal=ideal,
        )
        if cut <= y0 + 2 or cut >= y1 - 2:
            break
        output[index:index + 1] = [(y0, cut), (cut, y1)]
        forced += 1

    output = sorted((int(a), int(b)) for a, b in output if b - a >= 2)
    return output, forced

def segment_black_ink_glyphs_slider(
    image: Image.Image,
    *,
    apply_main_band_mask: bool = True,
    expected_count: int | None = None,
    character_anchors: Sequence[object] | None = None,
) -> tuple[Image.Image, list[BlackInkGlyphSegment], dict]:
    """Segment a vertical Japanese column with a top-to-bottom ink slider.

    A horizontal probe moves down the isolated main-text band.  A glyph begins
    when the probe first touches black ink and ends only after a *sustained*
    white gap.  Short white gaps inside one kanji are bridged, while detached
    punctuation remains an independent cell when it is separated by a real
    inter-character gap.  OCR text length is diagnostic only and never dictates
    the number of boxes, so a missed OCR character still appears as a physical
    review slot.
    """
    source = image.convert("RGB")
    if apply_main_band_mask:
        working, band_info = mask_main_text_band(source)
        source.close()
    else:
        working = source.copy()
        source.close()
        background = estimate_paper_background(working)
        band_info = MainBandMaskInfo(
            False, 0, working.width, working.width, working.height, 0, 0, background,
        )

    gray = _gray(working)
    values = _pixel_values(gray)
    if gray is not working:
        gray.close()
    threshold = _otsu(values)
    width, height = working.size
    binary = [1 if value < threshold else 0 for value in values]
    if not any(binary):
        return working, [], {
            "segmentation_mode": "vertical_ink_slider",
            "threshold": threshold,
            "segments": 0,
            "expected_count_hint": int(expected_count or 0),
        }

    body_x0, body_x1 = _refine_main_body_band(binary, width, height, band_info)
    body_width = max(1, body_x1 - body_x0)
    components = _connected_components(binary, width, height)
    ruby_result = classify_vertical_ruby(
        components,
        body_x0=body_x0,
        body_x1=body_x1,
        image_width=width,
        target_height=max(8.0, body_width * 1.08),
    )
    body_x0, body_x1 = int(ruby_result.body_x0), int(ruby_result.body_x1)
    body_width = max(1, body_x1 - body_x0)
    kept_components = [
        component for component in ruby_result.kept_components
        if float(component.get("x1", 0)) > body_x0
        and float(component.get("x0", 0)) < body_x1
    ]

    row_ink = [0] * height
    for component in kept_components:
        for pixel_index in component.get("pixels", []):
            x = int(pixel_index) % width
            y = int(pixel_index) // width
            if body_x0 <= x < body_x1 and 0 <= y < height:
                row_ink[y] += 1
    active_rows = [index for index, mass in enumerate(row_ink) if mass > 0]
    if not active_rows:
        return working, [], {
            "segmentation_mode": "vertical_ink_slider",
            "threshold": threshold,
            "segments": 0,
            "expected_count_hint": int(expected_count or 0),
            "main_band_x0": body_x0,
            "main_band_x1": body_x1,
        }

    # One vertical print cell is approximately square.  Keep the pitch tied
    # to the isolated body band rather than the full crop width, which may still
    # include white margins or side ruby.
    nominal_pitch = max(8.0, body_width * 1.02)
    # A row must contain more than one antialiasing speck to start the slider.
    row_threshold = max(1, int(round(body_width * 0.012)))
    active = [mass >= row_threshold for mass in row_ink]
    first, last = active_rows[0], active_rows[-1] + 1

    # The white-run threshold is deliberately proportional to em size.  It is
    # long enough to bridge the detached bars/dots of a kanji but short enough
    # to stop before the next printed cell.  This is the user's requested
    # "touch black -> start; sustained white -> stop" behaviour.
    # Stop soon after a real white valley.  Internal stroke gaps are merged in
    # the next stage using the one-cell span limit; using a long white timeout
    # here can swallow two or three tightly printed characters into one frame.
    stop_white_rows = max(2, min(7, int(round(nominal_pitch * 0.042))))
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    last_black = -1
    white_run = 0
    for y in range(first, last):
        if active[y]:
            if start is None:
                start = y
            last_black = y
            white_run = 0
            continue
        if start is None:
            continue
        white_run += 1
        if white_run >= stop_white_rows:
            intervals.append((start, last_black + 1))
            start = None
            last_black = -1
            white_run = 0
    if start is not None and last_black >= start:
        intervals.append((start, last_black + 1))

    effective_pitch = _estimate_review_slider_pitch(
        intervals,
        first_ink=first,
        last_ink=last,
        body_width=body_width,
        expected_count=expected_count,
    )

    # Merge detached strokes into one printed cell when the combined span still
    # fits one em.  This is intentionally not limited to a "tiny" component:
    # kanji such as 三/言/門 can contain two or three substantial horizontal
    # islands separated by completely white rows.  Treating every island as a
    # character was the main source of one-glyph-many-boxes.  Two neighbouring
    # full-size characters are not merged because their combined span exceeds
    # the one-cell limit; punctuation remains separate when followed by a real
    # inter-cell gap.
    def interval_mass(interval: tuple[int, int]) -> int:
        return sum(row_ink[interval[0]:interval[1]])

    merged: list[tuple[int, int]] = []
    median_mass = median([interval_mass(item) for item in intervals]) if intervals else 0
    for interval in intervals:
        if not merged:
            merged.append(interval)
            continue
        previous = merged[-1]
        gap = interval[0] - previous[1]
        combined_span = interval[1] - previous[0]
        current_mass = interval_mass(interval)
        previous_mass = interval_mass(previous)
        current_height = interval[1] - interval[0]
        previous_height = previous[1] - previous[0]
        current_small = current_mass <= max(4, median_mass * 0.20)
        previous_small = previous_mass <= max(4, median_mass * 0.20)
        close_fragment = gap <= max(stop_white_rows + 1, int(round(effective_pitch * 0.16)))
        one_cell_span = combined_span <= effective_pitch * 1.14
        incomplete_cell = (
            previous_height < effective_pitch * 0.78
            or current_height < effective_pitch * 0.78
        )
        # A very small punctuation run followed by another full-size glyph
        # should not be swallowed merely because the typesetting is tight.
        punctuation_then_glyph = (
            previous_small
            and previous_height < effective_pitch * 0.28
            and current_height >= effective_pitch * 0.55
            and gap >= effective_pitch * 0.14
        )
        if (
            close_fragment
            and one_cell_span
            and (incomplete_cell or current_small or previous_small)
            and not punctuation_then_glyph
        ):
            merged[-1] = (previous[0], interval[1])
        else:
            merged.append(interval)

    # A heavy run can contain two touching characters.  Split against the
    # isolated main-body projection, not the full raster: side ruby or neighbour
    # residue must never erase the white valley between two body characters.
    split_intervals: list[tuple[int, int]] = []
    for y0, y1 in merged:
        split_intervals.extend(
            _split_slider_band_by_valleys(row_ink, y0, y1, effective_pitch)
        )
    intervals = [(y0, y1) for y0, y1 in split_intervals if y1 - y0 >= 2]
    intervals, forced_oversized_splits = _split_oversized_review_slider_intervals(
        intervals,
        row_ink,
        pitch=effective_pitch,
        expected_count=expected_count,
    )

    anchors = list(character_anchors or [])
    anchor_cells, anchor_info = _vision_character_cells(
        anchors, w=width, h=height,
        body_x0=body_x0, body_x1=body_x1, binary=binary,
    )

    segments: list[BlackInkGlyphSegment] = []
    segment_boxes: list[dict] = []
    for interval_index, (y0, y1) in enumerate(intervals):
        xs: list[int] = []
        ink_pixels = 0
        for y in range(max(0, y0), min(height, y1)):
            base = y * width
            for x in range(max(0, body_x0), min(width, body_x1)):
                if binary[base + x]:
                    xs.append(x)
                    ink_pixels += 1
        if ink_pixels <= 0:
            continue
        if xs:
            x0, x1 = min(xs), max(xs) + 1
        else:
            x0, x1 = body_x0, body_x1
        pad_x = max(2, int(round(max(1, x1 - x0) * 0.12)))
        pad_y = max(1, int(round(effective_pitch * 0.045)))
        x0 = max(0, x0 - pad_x)
        x1 = min(width, x1 + pad_x)
        # Expand towards surrounding paper but stop at the midpoint between
        # neighbouring ink runs.  Adjacent frames therefore never overlap,
        # even for tightly printed kana, while every glyph retains a small
        # amount of white context for visual checking and recognition.
        previous_boundary = 0
        next_boundary = height
        if interval_index > 0:
            previous_boundary = (intervals[interval_index - 1][1] + y0) // 2
        if interval_index + 1 < len(intervals):
            next_boundary = (y1 + intervals[interval_index + 1][0]) // 2
        y0p = max(previous_boundary, y0 - pad_y)
        y1p = min(next_boundary, y1 + pad_y)
        if y1p <= y0p:
            y0p, y1p = max(0, y0), min(height, max(y0 + 1, y1))
        # Keep a useful minimum width for tiny punctuation in the centre preview.
        minimum_width = max(6, int(round(body_width * 0.46)))
        if x1 - x0 < minimum_width:
            centre = (x0 + x1) / 2.0
            x0 = max(0, int(round(centre - minimum_width / 2.0)))
            x1 = min(width, x0 + minimum_width)
            x0 = max(0, x1 - minimum_width)

        anchor_text = ""
        anchor_confidence = 0.0
        best_overlap = 0.0
        for anchor in anchor_cells:
            overlap = max(
                0.0,
                min(float(anchor.get("y1", 0)), y1p)
                - max(float(anchor.get("y0", 0)), y0p),
            )
            if overlap > best_overlap:
                best_overlap = overlap
                anchor_text = str(anchor.get("anchor_text") or "")
                anchor_confidence = float(anchor.get("anchor_confidence") or 0.0)
        crop = working.crop((x0, y0p, x1, y1p)).convert("RGB")
        segment = BlackInkGlyphSegment(
            index=len(segments),
            y0=y0p, y1=y1p, x0=x0, x1=x1,
            ink_pixels=ink_pixels,
            image=crop,
            anchor_text=anchor_text,
            anchor_confidence=anchor_confidence,
            segmentation_source="vertical_ink_slider",
        )
        segments.append(segment)
        segment_boxes.append({
            "index": segment.index,
            "x0": x0, "x1": x1, "y0": y0p, "y1": y1p,
            "ink_pixels": ink_pixels,
            "anchor_text": anchor_text,
            "anchor_confidence": anchor_confidence,
            "source": "vertical_ink_slider",
        })

    return working, segments, {
        "segmentation_mode": "vertical_ink_slider",
        "threshold": threshold,
        "row_threshold": row_threshold,
        "slider_stop_white_rows": stop_white_rows,
        "target_pitch": round(float(effective_pitch), 3),
        "nominal_pitch": round(float(nominal_pitch), 3),
        "forced_oversized_splits": int(forced_oversized_splits),
        "one_char_frame_validation": True,
        "segments": len(segments),
        "segment_boxes": segment_boxes,
        "expected_count_hint": int(expected_count or 0),
        "geometry_text_count_delta": len(segments) - int(expected_count or 0),
        "main_band_x0": body_x0,
        "main_band_x1": body_x1,
        "ruby_components_removed": int(ruby_result.ruby_component_count),
        "ruby_filter_confidence": float(ruby_result.confidence),
        "vision_character_anchor_count": int(anchor_info.get("anchor_count") or 0),
        "mask_applied": bool(band_info.applied),
    }


def segment_black_ink_glyphs(
    image: Image.Image,
    *,
    apply_main_band_mask: bool = True,
    glyph_padding_ratio: float = 0.10,
    expected_count: int | None = None,
    character_anchors: Sequence[object] | None = None,
    precomputed_boxes: Sequence[object] | None = None,
) -> tuple[Image.Image, list[BlackInkGlyphSegment], dict]:
    """Split a printed vertical column with adaptive projection-valley cuts.

    The adaptive projection-valley splitter is the primary geometry.  Each cut
    is allowed to drift locally around a soft character-height prior, and every
    resulting segment is tightened to its assigned body-band ink.  Apple Vision
    per-Character boxes are retained only as weak pitch/text hints; they never
    dictate the full column partition.  The old uniform grid is a last resort.
    """
    del glyph_padding_ratio  # retained for backward API compatibility
    source = image.convert('RGB')
    if precomputed_boxes:
        cached_segments, cached_info = _segments_from_precomputed_boxes(source, precomputed_boxes)
        if cached_segments:
            return source, cached_segments, cached_info
    if apply_main_band_mask:
        working, band_info = mask_main_text_band(source)
        source.close()
    else:
        working = source.copy()
        source.close()
        bg = estimate_paper_background(working)
        band_info = MainBandMaskInfo(False, 0, working.width, working.width, working.height, 0, 0, bg)

    gray = _gray(working)
    values = _pixel_values(gray)
    if gray is not working:
        gray.close()
    threshold = _otsu(values)
    w, h = working.size
    binary = [1 if value < threshold else 0 for value in values]
    if sum(binary) <= 0:
        return working, [], {
            'segmentation_mode': 'projection_valley_hybrid',
            'threshold': threshold,
            'target_pitch': 0.0,
            'projection_target_height': 0.0,
            'projection_cuts': [],
            'segments': 0,
            'mask_applied': band_info.applied,
        }

    # First find the dominant body band, then run a dedicated side-ruby
    # classifier.  The projection profile is built from kept connected
    # components only, so repeated furigana cannot create extra valleys/centres.
    body_x0, body_x1 = _refine_main_body_band(binary, w, h, band_info)
    initial_body_width = max(1.0, float(body_x1 - body_x0))
    all_components = _connected_components(binary, w, h)
    ruby_result = classify_vertical_ruby(
        all_components,
        body_x0=body_x0,
        body_x1=body_x1,
        image_width=w,
        target_height=max(8.0, initial_body_width * 1.08),
    )
    body_x0, body_x1 = ruby_result.body_x0, ruby_result.body_x1
    body_width = max(1.0, float(body_x1 - body_x0))
    base_height = max(8.0, body_width * 1.08)
    keep_margin = max(2.0, body_width * 0.14)
    min_noise_area = max(3, int(round(base_height * body_width * 0.0009)))
    components: list[dict] = []
    for component in ruby_result.kept_components:
        area = int(component.get('area', 0))
        comp_h = int(component['y1']) - int(component['y0'])
        overlaps_body = float(component['x1']) > body_x0 and float(component['x0']) < body_x1
        near_body = float(component['x1']) >= body_x0 - keep_margin and float(component['x0']) <= body_x1 + keep_margin
        substantial = area >= min_noise_area or comp_h >= max(4.0, base_height * 0.20)
        if overlaps_body or (near_body and substantial):
            components.append(component)

    body_row_ink = [0] * h
    for component in components:
        for pixel_index in component.get('pixels', []):
            xx = int(pixel_index) % w
            yy = int(pixel_index) // w
            if body_x0 <= xx < body_x1 and 0 <= yy < h:
                body_row_ink[yy] += 1
    active_rows = [y for y, value in enumerate(body_row_ink) if value > 0]
    if not active_rows:
        return working, [], {
            'segmentation_mode': 'projection_valley_hybrid',
            'threshold': threshold,
            'target_pitch': 0.0,
            'projection_target_height': 0.0,
            'projection_cuts': [],
            'segments': 0,
            'mask_applied': band_info.applied,
            'main_band_x0': body_x0,
            'main_band_x1': body_x1,
            'ruby_components_removed': int(ruby_result.ruby_component_count),
            'ruby_boxes': [list(box) for box in ruby_result.ruby_boxes],
        }

    raw_runs = _row_ink_runs(body_row_ink, threshold=max(1, int(round(body_width * 0.015))))
    groups = _merge_runs_for_pitch(raw_runs, base_height)
    first_ink, last_ink = active_rows[0], active_rows[-1] + 1
    candidates = _projection_pitch_candidates(
        body_row_ink, groups, body_width=body_width, expected_count=expected_count,
    )
    anchor_cells, anchor_info = _vision_character_cells(
        list(character_anchors or []), w=w, h=h,
        body_x0=body_x0, body_x1=body_x1, binary=binary,
    )
    anchor_pitch = float(anchor_info.get("anchor_pitch") or 0.0)
    if anchor_pitch > 0:
        candidates = sorted(set([*candidates, anchor_pitch]))

    # Apple Vision per-Character boxes are deliberately only weak hints here.
    # They may contribute a plausible character height and later provide an
    # overlapping candidate label, but the actual geometry always comes from
    # the black-ink projection valleys below.  This prevents one missed or
    # merged Vision character from shifting the complete column partition.

    used_uniform_fallback = False
    used_center_fallback = False
    center_result = None
    local_valley_splits = 0
    local_grid_splits = 0
    # Adaptive projection valleys are always the primary geometry. Candidate
    # height selection combines body width, projection rhythm and only a weak
    # Apple anchor pitch hint.
    best: tuple[float, float, list[int]] | None = None
    for target_height in candidates:
        cuts, score = _projection_partition(
            body_row_ink,
            first_ink=first_ink,
            last_ink=last_ink,
            target_height=target_height,
        )
        if len(cuts) < 2:
            continue
        split_area = 0.0
        for cut in cuts[1:-1]:
            for component in components:
                if float(component['y0']) < cut < float(component['y1']):
                    depth = min(cut - float(component['y0']), float(component['y1']) - cut)
                    tolerance = max(2.0, target_height * 0.08)
                    ratio = min(1.0, max(0.0, depth / tolerance))
                    split_area += int(component.get('area', 0)) * ratio * ratio
        total_area = max(1, sum(int(component.get('area', 0)) for component in components))
        adjusted = score + split_area / total_area * 2.8
        adjusted += abs(target_height - base_height) / max(1.0, base_height) * 0.16
        if expected_count and int(expected_count) > 0:
            occupied = sum(
                1 for y0, y1 in zip(cuts, cuts[1:])
                if sum(body_row_ink[y0:y1]) >= min_noise_area
            )
            adjusted += abs(occupied - int(expected_count)) / max(1, int(expected_count)) * 0.08
        if best is None or adjusted < best[0]:
            best = (adjusted, target_height, cuts)

    projection_target = float(best[1]) if best is not None else base_height
    center_result = detect_center_intervals(
        components,
        body_row_ink,
        first_ink=first_ink,
        last_ink=last_ink,
        target_height=projection_target,
        body_width=body_width,
    )

    if best is None:
        if center_result.intervals and center_result.confidence >= 0.48:
            target_height = max(6.0, float(center_result.estimated_height))
            cuts = [center_result.intervals[0][0], *[item[1] for item in center_result.intervals]]
            used_center_fallback = True
        else:
            refined_band = MainBandMaskInfo(
                band_info.applied, body_x0, body_x1,
                band_info.original_width, band_info.original_height,
                band_info.ink_columns, band_info.removed_ink_pixels,
                band_info.background,
            )
            fallback_pitch, fallback_phase, _rows, _groups = _uniform_grid_pitch_and_phase(
                binary, w, h, refined_band, expected_count=expected_count,
            )
            target_height = max(6.0, fallback_pitch or base_height)
            first_cell = int((first_ink - fallback_phase) // target_height)
            last_cell = int((max(first_ink, last_ink - 1) - fallback_phase) // target_height)
            cuts = [int(round(fallback_phase + first_cell * target_height))]
            cuts.extend(int(round(fallback_phase + (cell + 1) * target_height)) for cell in range(first_cell, last_cell + 1))
            cuts = sorted(set(max(0, min(h, value)) for value in cuts))
            used_uniform_fallback = True
    else:
        best_score, target_height, cuts = best
        projection_count = max(0, len(cuts) - 1)
        center_count = len(center_result.intervals)
        # A strong centre result may repair a projection partition that clearly
        # merged several local characters.  It never replaces a similarly sized
        # healthy projection result.
        if (
            center_count >= 2
            and center_result.confidence >= 0.72
            and (projection_count <= 1 or center_count >= projection_count + 2)
        ):
            cuts = [center_result.intervals[0][0], *[item[1] for item in center_result.intervals]]
            target_height = max(6.0, float(center_result.estimated_height))
            used_center_fallback = True

    intervals = [(a, b) for a, b in zip(cuts, cuts[1:]) if b - a >= 3]
    intervals, local_valley_splits, local_grid_splits = _split_ambiguous_projection_intervals(
        intervals, body_row_ink, target_height=target_height,
    )

    def best_interval_for_component(component: dict) -> int | None:
        best_index: int | None = None
        best_key: tuple[float, float] | None = None
        centre = float(component.get('cy', 0.0))
        for index, (y0, y1) in enumerate(intervals):
            overlap = max(0.0, min(float(component['y1']), y1) - max(float(component['y0']), y0))
            if overlap <= 0 and not (y0 <= centre < y1):
                continue
            distance = abs(((y0 + y1) / 2.0) - centre)
            key = (-overlap, distance)
            if best_key is None or key < best_key:
                best_key = key
                best_index = index
        return best_index

    def sliced_component(component: dict, interval_index: int) -> dict | None:
        y0, y1 = intervals[interval_index]
        pixels = [
            int(pixel_index)
            for pixel_index in component.get('pixels', [])
            if y0 <= (int(pixel_index) // w) < y1
        ]
        if not pixels:
            return None
        xs = [pixel_index % w for pixel_index in pixels]
        ys = [pixel_index // w for pixel_index in pixels]
        x0 = min(xs); x1 = max(xs) + 1
        py0 = min(ys); py1 = max(ys) + 1
        return {
            'x0': x0, 'x1': x1, 'y0': py0, 'y1': py1,
            'area': len(pixels), 'pixels': pixels,
            'cx': (x0 + x1) / 2.0, 'cy': (py0 + py1) / 2.0,
            'projection_slice': True,
        }

    components_by_interval: dict[int, list[dict]] = {}
    projection_component_slices = 0
    for component in components:
        overlapping = [
            index for index, (y0, y1) in enumerate(intervals)
            if max(float(component['y0']), y0) < min(float(component['y1']), y1)
        ]
        component_height = float(component['y1']) - float(component['y0'])
        if len(overlapping) >= 2 and component_height >= float(target_height) * 1.32:
            pieces = []
            for index in overlapping:
                piece = sliced_component(component, index)
                if piece is not None:
                    pieces.append((index, piece))
            meaningful = [
                (index, piece) for index, piece in pieces
                if int(piece.get('area', 0)) >= max(2, int(round(int(component.get('area', 0)) * 0.10)))
            ]
            if len(meaningful) >= 2:
                for index, piece in meaningful:
                    components_by_interval.setdefault(index, []).append(piece)
                projection_component_slices += len(meaningful) - 1
                continue
        index = best_interval_for_component(component)
        if index is not None:
            components_by_interval.setdefault(index, []).append(component)

    # Reattach detached dakuten/handakuten or tiny stroke fragments when a cut
    # placed them alone immediately beside a substantial glyph. A real Japanese
    # punctuation cell normally sits roughly one character advance away, so the
    # very small gap threshold below does not absorb it.
    substantial_areas = sorted(
        int(component.get('area', 0))
        for component in components
        if int(component.get('area', 0)) >= min_noise_area
    )
    median_area = float(substantial_areas[len(substantial_areas) // 2]) if substantial_areas else float(min_noise_area)
    tiny_area_limit = max(min_noise_area, int(round(median_area * 0.18)))
    attachment_gap = max(2.0, float(target_height) * 0.18)
    for interval_index in range(len(intervals)):
        current_items = list(components_by_interval.get(interval_index, []))
        if not current_items:
            continue
        has_substantial = any(int(item.get('area', 0)) > tiny_area_limit for item in current_items)
        if has_substantial:
            continue
        for component in list(current_items):
            if int(component.get('area', 0)) > tiny_area_limit:
                continue
            # A close sentence-final period/comma may occupy the top corner of
            # its own em cell and sit only a few pixels below the previous glyph.
            # Keep it independent when its centre has already advanced by roughly
            # half a character. Detached dakuten usually overlap the base glyph
            # vertically or sit above it, so they still follow the attachment path.
            previous_major = [
                item for item in components_by_interval.get(interval_index - 1, [])
                if int(item.get('area', 0)) > tiny_area_limit
            ]
            if (
                len(current_items) == 1
                and previous_major
                and _looks_like_trailing_punctuation_component(
                    component, target_height=float(target_height),
                    body_width=body_width, median_area=median_area,
                )
            ):
                previous_area = sum(max(1.0, float(item.get('area', 0))) for item in previous_major)
                previous_cy = sum(
                    float(item.get('cy', 0.0)) * max(1.0, float(item.get('area', 0)))
                    for item in previous_major
                ) / max(1.0, previous_area)
                if float(component.get('cy', 0.0)) - previous_cy >= float(target_height) * 0.40:
                    continue
            best_target: tuple[float, int] | None = None
            for neighbour in (interval_index - 1, interval_index + 1):
                neighbour_items = components_by_interval.get(neighbour, [])
                major_items = [item for item in neighbour_items if int(item.get('area', 0)) > tiny_area_limit]
                if not major_items:
                    continue
                union_y0 = min(float(item['y0']) for item in major_items)
                union_y1 = max(float(item['y1']) for item in major_items)
                if float(component['y1']) <= union_y0:
                    gap = union_y0 - float(component['y1'])
                elif float(component['y0']) >= union_y1:
                    gap = float(component['y0']) - union_y1
                else:
                    gap = 0.0
                major_cx = sum(float(item.get('cx', 0.0)) for item in major_items) / len(major_items)
                x_distance = abs(float(component.get('cx', 0.0)) - major_cx)
                if gap <= attachment_gap and x_distance <= max(3.0, body_width * 0.62):
                    key = gap + x_distance * 0.08
                    if best_target is None or key < best_target[0]:
                        best_target = (key, neighbour)
            if best_target is not None:
                components_by_interval[interval_index].remove(component)
                components_by_interval.setdefault(best_target[1], []).append(component)

    intervals, components_by_interval, trailing_punctuation_splits = _split_trailing_punctuation_intervals(
        intervals, components_by_interval,
        target_height=float(target_height),
        body_width=body_width,
        median_area=median_area,
    )

    background = estimate_paper_background(working)
    source_pixels = working.load()
    segments: list[BlackInkGlyphSegment] = []
    blank_intervals = 0
    segment_boxes: list[dict] = []
    for interval_index, (cell_y0, cell_y1) in enumerate(intervals):
        assigned = components_by_interval.get(interval_index, [])
        copied_area = sum(int(component.get('area', 0)) for component in assigned)
        anchor_cell = None
        best_anchor_overlap = 0.0
        for anchor in anchor_cells:
            overlap = max(0.0, min(float(anchor.get("y1", 0)), cell_y1) - max(float(anchor.get("y0", 0)), cell_y0))
            if overlap > best_anchor_overlap:
                best_anchor_overlap = overlap
                anchor_cell = anchor
        forced_anchor_text = ""
        if anchor_cell is not None and best_anchor_overlap >= max(2.0, min(cell_y1-cell_y0, float(anchor_cell.get("y1", 0))-float(anchor_cell.get("y0", 0))) * 0.28):
            forced_anchor_text = str(anchor_cell.get("anchor_text") or "")
        if not assigned or copied_area < min_noise_area:
            blank_intervals += 1
            continue

        margin_y = max(1, int(round(target_height * 0.055)))
        margin_x = max(1, int(round(body_width * 0.055)))
        union_x0 = min(int(component['x0']) for component in assigned)
        union_x1 = max(int(component['x1']) for component in assigned)
        union_y0 = min(int(component['y0']) for component in assigned)
        union_y1 = max(int(component['y1']) for component in assigned)
        tight_y0 = max(cell_y0, union_y0 - margin_y)
        tight_y1 = min(cell_y1, union_y1 + margin_y)
        tight_x0 = max(body_x0, union_x0 - margin_x)
        tight_x1 = min(body_x1, union_x1 + margin_x)
        if tight_y1 <= tight_y0 or tight_x1 <= tight_x0:
            blank_intervals += 1
            continue

        crop_height = max(1, tight_y1 - tight_y0)
        crop = Image.new('RGB', (w, crop_height), background)
        crop_pixels = crop.load()
        copied_pixels = 0
        for component in assigned:
            for pixel_index in component.get('pixels', []):
                xx = int(pixel_index) % w
                yy = int(pixel_index) // w
                target_y = yy - tight_y0
                # The projection stage has already selected the glyph's body band.
                # Clip directly to its exact tight box instead of rebuilding a
                # full-width raster and running mask_single_glyph/CC once more.
                if tight_x0 <= xx < tight_x1 and 0 <= target_y < crop_height:
                    crop_pixels[xx, target_y] = source_pixels[xx, yy]
                    copied_pixels += 1
        if copied_pixels < min_noise_area and not forced_anchor_text:
            crop.close()
            blank_intervals += 1
            continue
        masked_crop = crop
        anchor_text = forced_anchor_text
        anchor_confidence = float((anchor_cell or {}).get("anchor_confidence") or 0.0)
        segment = BlackInkGlyphSegment(
            index=len(segments), y0=tight_y0, y1=tight_y1,
            x0=tight_x0, x1=tight_x1,
            ink_pixels=copied_pixels, image=masked_crop,
            anchor_text=anchor_text,
            anchor_confidence=anchor_confidence,
            segmentation_source=(
                'projection_valley_uniform_fallback' if used_uniform_fallback
                else 'projection_valley_center_fallback' if used_center_fallback
                else 'projection_valley_hybrid'
            ),
        )
        segments.append(segment)
        segment_boxes.append({
            'index': segment.index,
            'x0': tight_x0, 'x1': tight_x1,
            'y0': tight_y0, 'y1': tight_y1,
            'cell_y0': cell_y0, 'cell_y1': cell_y1,
            'ink_pixels': copied_pixels,
        })

    mode = (
        'projection_valley_uniform_fallback' if used_uniform_fallback
        else 'projection_valley_center_fallback' if used_center_fallback
        else 'projection_valley_hybrid'
    )
    return working, segments, {
        'segmentation_mode': mode,
        'threshold': threshold,
        'target_pitch': round(float(target_height), 3),
        'projection_target_height': round(float(target_height), 3),
        'projection_cuts': [int(value) for value in cuts],
        'projection_intervals': len(intervals),
        'projection_local_valley_splits': int(local_valley_splits),
        'projection_local_grid_splits': int(local_grid_splits),
        'projection_local_fallback_splits': int(local_grid_splits),
        'projection_component_slices': int(projection_component_slices),
        'trailing_punctuation_splits': int(trailing_punctuation_splits),
        'uniform_fallback_used': bool(used_uniform_fallback),
        'center_fallback_used': bool(used_center_fallback),
        'center_candidate_count': len(center_result.centers) if center_result is not None else 0,
        'center_confidence': float(center_result.confidence) if center_result is not None else 0.0,
        'ruby_components_removed': int(ruby_result.ruby_component_count),
        'ruby_filter_confidence': float(ruby_result.confidence),
        'ruby_boxes': [list(box) for box in ruby_result.ruby_boxes],
        'grid_phase': 0.0,
        'grid_cells': len(intervals),
        'blank_cells_skipped': blank_intervals,
        'components': len(components),
        'groups': len(groups),
        'segments': len(segments),
        'segment_boxes': segment_boxes,
        'mask_applied': band_info.applied,
        'mask_x0': band_info.x0,
        'mask_x1': band_info.x1,
        'main_band_x0': body_x0,
        'main_band_x1': body_x1,
        'removed_ink_pixels': band_info.removed_ink_pixels,
        'expected_count_hint': int(expected_count or 0),
        'vision_character_anchor_count': int(anchor_info.get('anchor_count') or 0),
        'vision_inserted_slots': int(anchor_info.get('inserted_slots') or 0),
        'vision_anchor_pitch_hint': round(anchor_pitch, 3) if anchor_pitch else 0.0,
        'vision_anchor_fallback': bool(character_anchors and not anchor_cells),
    }

def render_uniform_glyph_grid_preview(
    image: Image.Image,
    columns: Sequence[object],
    *,
    character_anchors_by_column: dict[int, Sequence[object]] | None = None,
    crop_boxes_by_column: dict[int, Sequence[int]] | None = None,
    skip_glyph_columns: Sequence[int] | None = None,
) -> tuple[Image.Image, list[dict]]:
    """Draw physical columns and the exact adaptive glyph boxes used by recognition.

    The historical function name is kept for UI compatibility.  Blue boxes are
    always produced by projection-valley segmentation and per-glyph ink
    tightening.  Apple character anchors are displayed only as weak labels and
    pitch hints; they do not define the box geometry.
    """
    annotated = image.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    line_width = max(2, round(min(annotated.size) / 520))
    diagnostics: list[dict] = []
    anchors_map = character_anchors_by_column or {}
    crop_map = crop_boxes_by_column or {}
    skip_set = {int(value) for value in (skip_glyph_columns or []) if int(value) > 0}
    for column_index, column in enumerate(columns, start=1):
        left = max(0, int(getattr(column, "left", 0)))
        right = min(annotated.width, int(getattr(column, "right", annotated.width)))
        hard_left = max(0, int(getattr(column, "hard_left", left)))
        hard_right = min(annotated.width, int(getattr(column, "hard_right", right)))
        top = max(0, int(getattr(column, "top", 0)))
        bottom = min(annotated.height, int(getattr(column, "bottom", annotated.height)))
        expected = int(getattr(column, "estimated_chars", 0) or 0)
        draw.rectangle(
            (left, top, max(left + 1, right - 1), max(top + 1, bottom - 1)),
            outline=(220, 30, 45), width=line_width,
        )
        draw.text((left + line_width * 2, top + line_width * 2), f"C{column_index}", fill=(220, 30, 45))

        raw_crop_box = crop_map.get(column_index)
        if raw_crop_box and len(raw_crop_box) >= 4:
            crop_x0 = max(0, int(raw_crop_box[0])); crop_y0 = max(0, int(raw_crop_box[1]))
            crop_x1 = min(image.width, int(raw_crop_box[2])); crop_y1 = min(image.height, int(raw_crop_box[3]))
        else:
            crop_x0, crop_y0, crop_x1, crop_y1 = hard_left, 0, hard_right, image.height
        if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
            continue
        if column_index in skip_set:
            diagnostics.append({
                "column": column_index, "boxes": [], "box_count": 0,
                "pitch": 0.0, "phase": 0.0,
                "segmentation_mode": "ordinary_ocr_title_bypass",
                "projection_cuts": [], "vision_character_anchor_count": 0,
                "ruby_components_removed": 0, "center_fallback_used": False,
                "center_confidence": 0.0, "main_band_x0": left,
                "main_band_x1": right,
                "crop_box": [crop_x0, crop_y0, crop_x1, crop_y1],
                "glyph_segmentation_skipped": True,
            })
            continue

        column_crop = image.crop((crop_x0, crop_y0, crop_x1, crop_y1)).convert("RGB")
        segments: list[BlackInkGlyphSegment] = []
        try:
            masked, segments, info = segment_black_ink_glyphs(
                column_crop,
                apply_main_band_mask=True,
                expected_count=expected or None,
                character_anchors=list(anchors_map.get(column_index) or []),
            )
            masked.close()
            band_left = max(left, crop_x0 + int(info.get("main_band_x0", 0) or 0))
            band_right = min(right, crop_x0 + int(info.get("main_band_x1", crop_x1 - crop_x0) or (crop_x1 - crop_x0)))
            if band_right <= band_left:
                band_left, band_right = left, right
            boxes = []
            local_segment_boxes = {
                int(item.get("index", index)): item
                for index, item in enumerate(info.get("segment_boxes", []) or [])
                if isinstance(item, dict)
            }
            pitch_value = float(info.get("target_pitch", 0.0) or 0.0)
            for glyph_index, segment in enumerate(segments, start=1):
                local_box = local_segment_boxes.get(glyph_index - 1, {})
                y0 = max(0, crop_y0 + int(segment.y0))
                y1 = min(annotated.height, crop_y0 + int(segment.y1))
                glyph_left = max(left, min(right - 1, crop_x0 + int(segment.x0)))
                glyph_right = min(right, max(glyph_left + 1, crop_x0 + int(segment.x1)))
                draw.rectangle(
                    (glyph_left, y0, max(glyph_left + 1, glyph_right - 1), max(y0 + 1, y1 - 1)),
                    outline=(0, 113, 227), width=max(1, line_width - 1),
                )
                if glyph_index <= 99:
                    draw.text((max(0, glyph_right - 28), y0 + 1), str(glyph_index), fill=(0, 113, 227))
                boxes.append({
                    "index": glyph_index,
                    "x0": glyph_left, "x1": glyph_right,
                    "y0": y0, "y1": y1,
                    "cell_y0": crop_y0 + int(local_box.get("cell_y0", segment.y0)),
                    "cell_y1": crop_y0 + int(local_box.get("cell_y1", segment.y1)),
                    "target_pitch": pitch_value,
                    "ink_pixels": int(local_box.get("ink_pixels", segment.ink_pixels) or segment.ink_pixels),
                    "anchor_text": segment.anchor_text,
                    "anchor_confidence": float(segment.anchor_confidence or 0.0),
                    "source": segment.segmentation_source,
                })
            diagnostics.append({
                "column": column_index,
                "boxes": boxes,
                "box_count": len(boxes),
                "pitch": info.get("target_pitch", 0.0),
                "phase": info.get("grid_phase", 0.0),
                "segmentation_mode": info.get("segmentation_mode", "projection_valley_hybrid"),
                "projection_cuts": info.get("projection_cuts", []),
                "vision_character_anchor_count": info.get("vision_character_anchor_count", 0),
                "ruby_components_removed": info.get("ruby_components_removed", 0),
                "center_fallback_used": info.get("center_fallback_used", False),
                "center_confidence": info.get("center_confidence", 0.0),
                "main_band_x0": band_left,
                "main_band_x1": band_right,
                "crop_box": [crop_x0, crop_y0, crop_x1, crop_y1],
            })
        finally:
            for segment in segments:
                try:
                    segment.image.close()
                except Exception:
                    pass
            column_crop.close()
    return annotated, diagnostics
