#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Non-destructive scan cleanup for photographed books and documents.

This module deliberately lives *before* the OCR adapters.  It writes processed
copies into the application's session temporary directory and never mutates the
source images.  The implementation depends only on Pillow so enabling it cannot
pull OpenCV/numpy into the main GUI environment or disturb existing OCR model
runtimes.

The geometry detector is conservative: when a page boundary is not sufficiently
clear it returns the original geometry instead of guessing.  This is important
for full-bleed illustrations and already-cropped PDF rasters.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import statistics
import threading
from typing import Callable, Iterable, Sequence

from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps

try:  # Optional; requirements include pillow-heif, but keep import resilient.
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass


ProgressCallback = Callable[[int, int, str], None]
CancelCheck = Callable[[], bool]


@dataclass(frozen=True)
class ScanPreprocessOptions:
    """Settings for one scan-cleanup batch.

    Defaults are intentionally mild.  ``split_spread`` remains disabled because
    it changes page count; all other geometry operations are confidence-gated.
    """

    auto_crop: bool = True
    perspective: bool = True
    deskew: bool = True
    split_spread: bool = False
    spread_order: str = "right_to_left"
    enhancement: str = "soft"  # none / soft / strong / ocr
    preserve_color: bool = True
    crop_margin_percent: float = 0.8
    max_deskew_degrees: float = 3.0

    def normalized(self) -> "ScanPreprocessOptions":
        enhancement = str(self.enhancement or "soft").strip().lower()
        if enhancement not in {"none", "soft", "strong", "ocr"}:
            enhancement = "soft"
        order = str(self.spread_order or "right_to_left").strip().lower()
        if order not in {"right_to_left", "left_to_right"}:
            order = "right_to_left"
        return ScanPreprocessOptions(
            auto_crop=bool(self.auto_crop),
            perspective=bool(self.perspective),
            deskew=bool(self.deskew),
            split_spread=bool(self.split_spread),
            spread_order=order,
            enhancement=enhancement,
            preserve_color=bool(self.preserve_color),
            crop_margin_percent=max(0.0, min(5.0, float(self.crop_margin_percent))),
            max_deskew_degrees=max(0.0, min(6.0, float(self.max_deskew_degrees))),
        )

    def to_dict(self) -> dict:
        return asdict(self.normalized())

    @classmethod
    def from_dict(cls, payload: dict | None) -> "ScanPreprocessOptions":
        values = dict(payload or {})
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in values.items() if key in allowed}).normalized()


@dataclass(frozen=True)
class ProcessedScanPage:
    source_path: str
    output_path: str
    source_index: int
    part: str = "single"  # single / left / right
    geometry_applied: bool = False
    enhancement_applied: str = "none"


@dataclass(frozen=True)
class ScanBatchResult:
    pages: tuple[ProcessedScanPage, ...]
    source_count: int
    output_count: int
    split_count: int
    geometry_count: int
    enhanced_count: int

    @property
    def output_paths(self) -> list[str]:
        return [page.output_path for page in self.pages]


@dataclass(frozen=True)
class _DetectedQuad:
    # Source-image coordinates ordered clockwise from upper-left.
    tl: tuple[float, float]
    tr: tuple[float, float]
    br: tuple[float, float]
    bl: tuple[float, float]
    confidence: float


_SAVE_LOCKS: dict[str, threading.Lock] = {}
_SAVE_LOCKS_GUARD = threading.Lock()


def _save_lock(path: Path) -> threading.Lock:
    key = str(path.resolve(strict=False))
    with _SAVE_LOCKS_GUARD:
        return _SAVE_LOCKS.setdefault(key, threading.Lock())


def _resampling_lanczos():
    return getattr(Image, "Resampling", Image).LANCZOS


def _transform_quad():
    return getattr(Image, "Transform", Image).QUAD


def _safe_median(values: Sequence[float], default: float = 255.0) -> float:
    return float(statistics.median(values)) if values else float(default)


def _line_fit(points: Sequence[tuple[float, float]]) -> tuple[float, float] | None:
    """Least-squares fit y = slope*x + intercept with simple robust trimming."""
    if len(points) < 4:
        return None

    def fit(data):
        n = float(len(data))
        sx = sum(x for x, _ in data)
        sy = sum(y for _, y in data)
        sxx = sum(x * x for x, _ in data)
        sxy = sum(x * y for x, y in data)
        denominator = n * sxx - sx * sx
        if abs(denominator) < 1e-8:
            return 0.0, sy / n
        slope = (n * sxy - sx * sy) / denominator
        intercept = (sy - slope * sx) / n
        return slope, intercept

    slope, intercept = fit(points)
    residuals = [abs(y - (slope * x + intercept)) for x, y in points]
    median = _safe_median(residuals, 0.0)
    threshold = max(1.5, median * 2.8)
    trimmed = [point for point, residual in zip(points, residuals) if residual <= threshold]
    if len(trimmed) >= max(4, len(points) // 2):
        slope, intercept = fit(trimmed)
    return float(slope), float(intercept)


def _intersection(
    line_a: tuple[str, float, float],
    line_b: tuple[str, float, float],
) -> tuple[float, float] | None:
    """Intersect lines represented as x=a*y+b or y=a*x+b."""
    kind_a, a1, b1 = line_a
    kind_b, a2, b2 = line_b
    if kind_a == kind_b:
        return None
    if kind_a == "x":
        # x = a1*y+b1 and y = a2*x+b2
        denominator = 1.0 - a1 * a2
        if abs(denominator) < 1e-8:
            return None
        x = (a1 * b2 + b1) / denominator
        y = a2 * x + b2
        return x, y
    # y = a1*x+b1 and x = a2*y+b2
    denominator = 1.0 - a1 * a2
    if abs(denominator) < 1e-8:
        return None
    y = (a1 * b2 + b1) / denominator
    x = a2 * y + b2
    return x, y


def _polygon_area(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return abs(total) * 0.5


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _downsample_gray(image: Image.Image, max_side: int = 700) -> tuple[Image.Image, float, float]:
    width, height = image.size
    scale = min(1.0, float(max_side) / max(width, height, 1))
    target = (max(24, round(width * scale)), max(24, round(height * scale)))
    gray = ImageOps.grayscale(image)
    if target != image.size:
        resized = gray.resize(target, _resampling_lanczos())
        gray.close()
        gray = resized
    return gray, width / gray.width, height / gray.height


def _border_samples(gray: Image.Image) -> list[int]:
    width, height = gray.size
    px = gray.load()
    strip_x = max(1, round(width * 0.035))
    strip_y = max(1, round(height * 0.035))
    step = max(1, min(width, height) // 220)
    samples: list[int] = []
    for y in range(0, height, step):
        for x in range(0, strip_x, step):
            samples.append(int(px[x, y]))
        for x in range(max(0, width - strip_x), width, step):
            samples.append(int(px[x, y]))
    for x in range(0, width, step):
        for y in range(0, strip_y, step):
            samples.append(int(px[x, y]))
        for y in range(max(0, height - strip_y), height, step):
            samples.append(int(px[x, y]))
    return samples


def _center_samples(gray: Image.Image) -> list[int]:
    width, height = gray.size
    px = gray.load()
    left, right = round(width * 0.28), round(width * 0.72)
    top, bottom = round(height * 0.28), round(height * 0.72)
    step = max(1, min(width, height) // 120)
    return [int(px[x, y]) for y in range(top, bottom, step) for x in range(left, right, step)]


def _document_mask(gray: Image.Image) -> tuple[Image.Image | None, float]:
    """Return a conservative page/background mask and contrast estimate."""
    border = _border_samples(gray)
    center = _center_samples(gray)
    border_median = _safe_median(border)
    center_median = _safe_median(center)
    contrast = abs(center_median - border_median)

    # If the page already fills the frame, border and centre are often nearly
    # identical.  Refusing geometry here protects full-bleed illustrations and
    # PDF-rendered pages from accidental trimming.
    if contrast < 14.0:
        return None, contrast

    midpoint = (center_median + border_median) * 0.5
    if center_median > border_median:
        raw_mask = gray.point(lambda value: 255 if value >= midpoint else 0, mode="L")
    else:
        raw_mask = gray.point(lambda value: 255 if value <= midpoint else 0, mode="L")

    # Connect page regions interrupted by text/illustration while removing tiny
    # background speckles.  Kernels stay small on the downsampled image.
    kernel = 7 if min(gray.size) >= 240 else 5
    expanded = raw_mask.filter(ImageFilter.MaxFilter(kernel))
    raw_mask.close()
    mask = expanded.filter(ImageFilter.MinFilter(kernel))
    expanded.close()
    return mask, contrast


def detect_document_quad(image: Image.Image) -> _DetectedQuad | None:
    """Detect a photographed page boundary without external CV dependencies.

    The result is returned only when the border/background contrast and fitted
    edge coverage are strong enough.  Already-cropped scans normally return
    ``None`` and retain their exact geometry.
    """
    gray, scale_x, scale_y = _downsample_gray(image)
    try:
        mask, contrast = _document_mask(gray)
        if mask is None:
            return None
        try:
            width, height = mask.size
            px = mask.load()
            row_edges: list[tuple[int, int, int]] = []
            for y in range(height):
                xs = [x for x in range(width) if px[x, y] >= 128]
                if len(xs) >= max(4, round(width * 0.08)):
                    row_edges.append((y, min(xs), max(xs)))
            col_edges: list[tuple[int, int, int]] = []
            for x in range(width):
                ys = [y for y in range(height) if px[x, y] >= 128]
                if len(ys) >= max(4, round(height * 0.08)):
                    col_edges.append((x, min(ys), max(ys)))

            if len(row_edges) < height * 0.45 or len(col_edges) < width * 0.45:
                return None

            # Ignore the first/last few fitted samples where shadows and page
            # curls are most unstable.
            def middle(values):
                if len(values) < 20:
                    return values
                trim = max(2, len(values) // 20)
                return values[trim:-trim]

            row_fit = middle(row_edges)
            col_fit = middle(col_edges)
            left_fit = _line_fit([(float(y), float(left)) for y, left, _ in row_fit])
            right_fit = _line_fit([(float(y), float(right)) for y, _, right in row_fit])
            top_fit = _line_fit([(float(x), float(top)) for x, top, _ in col_fit])
            bottom_fit = _line_fit([(float(x), float(bottom)) for x, _, bottom in col_fit])
            if not all((left_fit, right_fit, top_fit, bottom_fit)):
                return None

            left_line = ("x", left_fit[0], left_fit[1])
            right_line = ("x", right_fit[0], right_fit[1])
            top_line = ("y", top_fit[0], top_fit[1])
            bottom_line = ("y", bottom_fit[0], bottom_fit[1])
            tl = _intersection(left_line, top_line)
            tr = _intersection(right_line, top_line)
            br = _intersection(right_line, bottom_line)
            bl = _intersection(left_line, bottom_line)
            if not all((tl, tr, br, bl)):
                return None

            quad_small = [tl, tr, br, bl]
            tolerance_x = width * 0.10
            tolerance_y = height * 0.10
            if any(
                x < -tolerance_x or x > width + tolerance_x
                or y < -tolerance_y or y > height + tolerance_y
                for x, y in quad_small
            ):
                return None
            area_fraction = _polygon_area(quad_small) / max(1.0, width * height)
            if not 0.25 <= area_fraction <= 0.985:
                return None

            # Require a real border removal.  A mask covering virtually the
            # entire frame is not useful and may be a full-bleed page.
            xs = [point[0] for point in quad_small]
            ys = [point[1] for point in quad_small]
            margin_fraction = (
                max(0.0, min(xs)) + max(0.0, width - max(xs))
                + max(0.0, min(ys)) + max(0.0, height - max(ys))
            ) / max(1.0, 2.0 * (width + height))
            if margin_fraction < 0.008:
                return None

            confidence = min(1.0, 0.35 + contrast / 90.0 + margin_fraction * 3.0)
            scaled = [
                (max(0.0, min(image.width - 1.0, x * scale_x)),
                 max(0.0, min(image.height - 1.0, y * scale_y)))
                for x, y in quad_small
            ]
            return _DetectedQuad(
                tl=scaled[0], tr=scaled[1], br=scaled[2], bl=scaled[3],
                confidence=confidence,
            )
        finally:
            mask.close()
    finally:
        gray.close()


def _quad_bbox(quad: _DetectedQuad, image_size: tuple[int, int], margin_percent: float) -> tuple[int, int, int, int]:
    width, height = image_size
    xs = [quad.tl[0], quad.tr[0], quad.br[0], quad.bl[0]]
    ys = [quad.tl[1], quad.tr[1], quad.br[1], quad.bl[1]]
    margin_x = width * max(0.0, margin_percent) / 100.0
    margin_y = height * max(0.0, margin_percent) / 100.0
    left = max(0, int(math.floor(min(xs) - margin_x)))
    top = max(0, int(math.floor(min(ys) - margin_y)))
    right = min(width, int(math.ceil(max(xs) + margin_x)))
    bottom = min(height, int(math.ceil(max(ys) + margin_y)))
    return left, top, max(left + 1, right), max(top + 1, bottom)


def _warp_quad(image: Image.Image, quad: _DetectedQuad, margin_percent: float) -> Image.Image:
    top_width = _distance(quad.tl, quad.tr)
    bottom_width = _distance(quad.bl, quad.br)
    left_height = _distance(quad.tl, quad.bl)
    right_height = _distance(quad.tr, quad.br)
    target_width = max(32, round(max(top_width, bottom_width)))
    target_height = max(32, round(max(left_height, right_height)))

    # Reject implausible warps.  They are almost always caused by an illustration
    # or a hand being mistaken for the page boundary.
    source_ratio = image.width / max(1.0, image.height)
    target_ratio = target_width / max(1.0, target_height)
    if target_ratio < source_ratio / 3.0 or target_ratio > source_ratio * 3.0:
        return image.copy()

    warped = image.transform(
        (target_width, target_height),
        _transform_quad(),
        (
            quad.tl[0], quad.tl[1],
            quad.bl[0], quad.bl[1],
            quad.br[0], quad.br[1],
            quad.tr[0], quad.tr[1],
        ),
        resample=Image.Resampling.BICUBIC,
    )
    margin = max(0, round(min(warped.size) * max(0.0, margin_percent) / 100.0))
    if margin <= 0:
        return warped
    canvas = Image.new("RGB", (warped.width + margin * 2, warped.height + margin * 2), "white")
    canvas.paste(warped, (margin, margin))
    warped.close()
    return canvas


def _projection_score(gray: Image.Image) -> float:
    """Score alignment using the sharper of horizontal/vertical ink profiles.

    Pillow's BOX resize computes the row/column averages in native code.  This
    avoids a Python pixel loop for every deskew candidate and keeps long-book
    preprocessing practical without adding numpy.
    """
    ink = ImageOps.invert(ImageOps.autocontrast(gray, cutoff=0.3))
    try:
        width, height = ink.size
        box = getattr(Image, "Resampling", Image).BOX
        row_projection = ink.resize((1, height), box)
        column_projection = ink.resize((width, 1), box)
        try:
            row_values = list(row_projection.getdata())
            column_values = list(column_projection.getdata())
        finally:
            row_projection.close()
            column_projection.close()

        def variance(values):
            if not values:
                return 0.0
            mean = sum(values) / len(values)
            return sum((value - mean) ** 2 for value in values) / len(values)

        return max(variance(row_values), variance(column_values))
    finally:
        ink.close()


def estimate_deskew_angle(image: Image.Image, max_degrees: float = 3.0) -> float:
    """Return the rotation angle that best aligns text, or zero when uncertain."""
    limit = max(0.0, min(6.0, float(max_degrees)))
    if limit < 0.5:
        return 0.0
    gray, _, _ = _downsample_gray(image, max_side=520)
    try:
        baseline = _projection_score(gray)
        candidates: list[tuple[float, float]] = [(baseline, 0.0)]
        step = 0.5
        count = int(round(limit / step))
        for index in range(-count, count + 1):
            angle = index * step
            if abs(angle) < 1e-9:
                continue
            rotated = gray.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=255)
            try:
                candidates.append((_projection_score(rotated), angle))
            finally:
                rotated.close()
        best_score, best_angle = max(candidates, key=lambda pair: pair[0])
        # Tiny or weak improvements are more likely noise than real skew.
        if abs(best_angle) < 0.45 or best_score < baseline * 1.035:
            return 0.0
        return float(best_angle)
    finally:
        gray.close()


def _apply_deskew(image: Image.Image, max_degrees: float) -> tuple[Image.Image, bool]:
    angle = estimate_deskew_angle(image, max_degrees=max_degrees)
    if abs(angle) < 0.01:
        return image.copy(), False
    result = image.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=(255, 255, 255),
    )
    return result, True


def _flatten_luminance(image: Image.Image, mode: str, preserve_color: bool) -> Image.Image:
    mode = str(mode or "none").lower()
    if mode == "none":
        return image.copy()

    gray = ImageOps.grayscale(image)
    shortest = max(1, min(image.size))
    radius = max(10, min(70, round(shortest * 0.028)))
    background = gray.filter(ImageFilter.GaussianBlur(radius=radius))
    try:
        normalized = ImageChops.subtract(gray, background, scale=1.0, offset=255)
    finally:
        background.close()

    cutoff = 0.25 if mode == "soft" else 0.45
    normalized = ImageOps.autocontrast(normalized, cutoff=cutoff)
    strength = {"soft": 0.58, "strong": 0.80, "ocr": 0.92}.get(mode, 0.58)
    mixed = Image.blend(gray, normalized, strength)
    normalized.close()
    contrast = {"soft": 1.05, "strong": 1.12, "ocr": 1.16}.get(mode, 1.05)
    mixed2 = ImageEnhance.Contrast(mixed).enhance(contrast)
    mixed.close()
    sharpened = mixed2.filter(
        ImageFilter.UnsharpMask(
            radius=0.65 if mode == "soft" else 0.85,
            percent=45 if mode == "soft" else 70,
            threshold=4,
        )
    )
    mixed2.close()
    gray.close()

    if mode == "ocr":
        return sharpened.convert("RGB")

    # Replace only luminance.  Chroma is preserved, and saturated illustration
    # pixels can be blended back toward the original to avoid washed-out art.
    ycbcr = image.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    y.close()
    merged = Image.merge("YCbCr", (sharpened, cb, cr)).convert("RGB")
    cb.close(); cr.close(); ycbcr.close(); sharpened.close()

    if not preserve_color:
        return merged

    hsv = image.convert("HSV")
    _, saturation, _ = hsv.split()
    # Saturation <= 28 stays fully enhanced; colourful pixels increasingly use
    # the source.  A blurred mask prevents hard halos around illustrations.
    protect = saturation.point(
        lambda value: 0 if value <= 28 else min(220, round((value - 28) * 1.35)),
        mode="L",
    ).filter(ImageFilter.GaussianBlur(radius=2.0))
    saturation.close(); hsv.close()
    protected = Image.composite(image, merged, protect)
    protect.close(); merged.close()
    return protected


def _spread_gutter_x(image: Image.Image) -> int:
    """Estimate the gutter position inside the centre 40% of a spread."""
    gray, scale_x, _ = _downsample_gray(image, max_side=720)
    try:
        width, height = gray.size
        if width < 80:
            return image.width // 2
        crop_top = round(height * 0.08)
        crop_bottom = max(crop_top + 1, round(height * 0.92))
        px = gray.load()
        start = round(width * 0.32)
        end = round(width * 0.68)
        profiles: list[tuple[float, int]] = []
        step_y = max(1, (crop_bottom - crop_top) // 260)
        for x in range(start, end):
            column = [int(px[x, y]) for y in range(crop_top, crop_bottom, step_y)]
            mean = sum(column) / max(1, len(column))
            dark = sum(1 for value in column if value < 175) / max(1, len(column))
            profiles.append((mean - dark * 75.0, x))

        # A gutter can be a bright blank valley (high score) or a dark book-spine
        # trench (low score).  Choose whichever deviates more from neighbouring
        # text columns, while gently preferring the physical centre.
        scores = [value for value, _ in profiles]
        median = _safe_median(scores)
        best_x = width // 2
        best_strength = -1.0
        for score, x in profiles:
            deviation = abs(score - median)
            centre_penalty = abs(x - width / 2.0) / max(1.0, width * 0.18)
            strength = deviation - centre_penalty * 7.0
            if strength > best_strength:
                best_strength = strength
                best_x = x
        if best_strength < 4.0:
            best_x = width // 2
        return max(1, min(image.width - 1, round(best_x * scale_x)))
    finally:
        gray.close()


def split_book_spread(image: Image.Image, order: str = "right_to_left") -> list[tuple[str, Image.Image]]:
    """Split a likely two-page photograph, otherwise return one copied image."""
    width, height = image.size
    ratio = width / max(1.0, height)
    if ratio < 1.12 or width < 640:
        return [("single", image.copy())]
    gutter = _spread_gutter_x(image)
    if gutter < width * 0.36 or gutter > width * 0.64:
        gutter = width // 2
    overlap = max(0, round(width * 0.003))
    left = image.crop((0, 0, min(width, gutter + overlap), height))
    right = image.crop((max(0, gutter - overlap), 0, width, height))
    if min(left.width, right.width) < height * 0.38:
        left.close(); right.close()
        return [("single", image.copy())]
    if str(order).lower() == "left_to_right":
        return [("left", left), ("right", right)]
    return [("right", right), ("left", left)]


def process_scan_image(
    image: Image.Image,
    options: ScanPreprocessOptions,
) -> tuple[list[tuple[str, Image.Image]], dict]:
    """Process an already-open image and return owned RGB images plus metrics."""
    options = options.normalized()
    oriented = ImageOps.exif_transpose(image)
    try:
        source = oriented.convert("RGB")
    finally:
        if oriented is not image:
            oriented.close()
    try:
        parts = split_book_spread(source, options.spread_order) if options.split_spread else [("single", source.copy())]
    finally:
        source.close()

    outputs: list[tuple[str, Image.Image]] = []
    geometry_applied = 0
    enhanced = 0
    for part_name, part_image in parts:
        current = part_image
        geometry_changed = False
        try:
            quad = detect_document_quad(current) if (options.auto_crop or options.perspective) else None
            if quad is not None:
                if options.perspective:
                    next_image = _warp_quad(current, quad, options.crop_margin_percent)
                    # _warp_quad may conservatively return an identical copy.
                    geometry_changed = next_image.size != current.size or quad.confidence >= 0.55
                elif options.auto_crop:
                    box = _quad_bbox(quad, current.size, options.crop_margin_percent)
                    next_image = current.crop(box)
                    geometry_changed = box != (0, 0, current.width, current.height)
                else:
                    next_image = current.copy()
                current.close()
                current = next_image

            if options.deskew:
                deskewed, did_deskew = _apply_deskew(current, options.max_deskew_degrees)
                current.close()
                current = deskewed
                geometry_changed = geometry_changed or did_deskew

            final_image = _flatten_luminance(current, options.enhancement, options.preserve_color)
            current.close()
            current = None
            outputs.append((part_name, final_image))
            if geometry_changed:
                geometry_applied += 1
            if options.enhancement != "none":
                enhanced += 1
        except Exception:
            if current is not None:
                current.close()
            for _, output in outputs:
                output.close()
            raise

    return outputs, {
        "split": len(outputs) > 1,
        "geometry_count": geometry_applied,
        "enhanced_count": enhanced,
    }


def _output_name(source: Path, source_index: int, part: str) -> str:
    safe_stem = "".join(
        ch if ch.isalnum() or ch in "-_." else "_" for ch in source.stem
    ).strip("._")[:72] or "page"
    suffix = "" if part == "single" else f"_{part}"
    return f"{source_index:05d}_{safe_stem}{suffix}.png"


def _atomic_save_png(image: Image.Image, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with _save_lock(target):
        temporary = target.with_name(target.name + f".tmp-{os.getpid()}-{threading.get_ident()}")
        try:
            image.save(temporary, "PNG", compress_level=2, optimize=False)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)


def process_scan_page(
    image_path: str,
    output_dir: str | Path,
    options: ScanPreprocessOptions,
    *,
    source_index: int = 1,
) -> list[ProcessedScanPage]:
    """Create non-destructive processed copies for one source page."""
    source = Path(image_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(str(source))
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as opened:
        generated, metrics = process_scan_image(opened, options)
    pages: list[ProcessedScanPage] = []
    try:
        for part, image in generated:
            target = out_dir / _output_name(source, int(source_index), part)
            _atomic_save_png(image, target)
            pages.append(ProcessedScanPage(
                source_path=str(source),
                output_path=str(target),
                source_index=int(source_index),
                part=part,
                geometry_applied=bool(metrics.get("geometry_count")),
                enhancement_applied=options.normalized().enhancement,
            ))
    finally:
        for _, image in generated:
            image.close()
    return pages


def process_scan_batch(
    image_paths: Iterable[str | Path],
    output_dir: str | Path,
    options: ScanPreprocessOptions,
    *,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> ScanBatchResult:
    """Process a batch in stable source order.

    The caller may mix processed and untouched pages outside this function.  A
    source page that is split emits two adjacent outputs in the configured book
    order, preserving deterministic page lineage for page-type remapping.
    """
    paths = [str(Path(path).expanduser().resolve()) for path in image_paths]
    pages: list[ProcessedScanPage] = []
    split_count = 0
    geometry_count = 0
    enhanced_count = 0
    total = len(paths)
    for source_index, path in enumerate(paths, start=1):
        if cancel_check is not None and cancel_check():
            raise RuntimeError("扫描件预处理已取消")
        produced = process_scan_page(
            path, output_dir, options, source_index=source_index,
        )
        pages.extend(produced)
        if len(produced) > 1:
            split_count += 1
        geometry_count += sum(1 for page in produced if page.geometry_applied)
        enhanced_count += sum(1 for page in produced if page.enhancement_applied != "none")
        if progress_callback is not None:
            progress_callback(source_index, total, Path(path).name)
    return ScanBatchResult(
        pages=tuple(pages),
        source_count=total,
        output_count=len(pages),
        split_count=split_count,
        geometry_count=geometry_count,
        enhanced_count=enhanced_count,
    )


def remap_page_metadata(
    source_pages: Sequence[ProcessedScanPage],
    page_overrides: dict[int, str] | None,
    auto_suggested: set[int] | None,
) -> tuple[dict[int, str], set[int]]:
    """Inherit page classifications after optional two-page splitting."""
    overrides = {int(key): str(value) for key, value in (page_overrides or {}).items()}
    suggested = {int(value) for value in (auto_suggested or set())}
    output_overrides: dict[int, str] = {}
    output_suggested: set[int] = set()
    for output_index, page in enumerate(source_pages, start=1):
        source_index = int(page.source_index)
        output_overrides[output_index] = overrides.get(source_index, "paragraph")
        if source_index in suggested:
            output_suggested.add(output_index)
    return output_overrides, output_suggested
