#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Neutral helpers for OCR models that consume isolated vertical text crops.

This module deliberately contains no recognizer/model runtime.  Hayai OCR and
48px AR share the same conservative physical-column segmentation helpers
without depending on any model-specific integration.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps


@dataclass(frozen=True, slots=True)
class OcrCropSegment:
    path: str
    expected_chars: int
    column_index: int
    segment_index: int


def compact_text(text: str) -> str:
    return "".join(str(text or "").split())


def japanese_ratio(text: str) -> float:
    value = compact_text(text)
    if not value:
        return 0.0
    count = 0
    for ch in value:
        code = ord(ch)
        if (
            0x3040 <= code <= 0x30FF
            or 0x31F0 <= code <= 0x31FF
            or 0x3400 <= code <= 0x4DBF
            or 0x4E00 <= code <= 0x9FFF
            or 0xF900 <= code <= 0xFAFF
            or ch in "、。！？…―ー・「」『』（）［］【】〈〉《》〜～＝0123456789０１２３４５６７８９"
        ):
            count += 1
    return count / max(1, len(value))


def _ink_bbox(image: Image.Image, threshold: int = 242) -> tuple[int, int, int, int] | None:
    gray = ImageOps.grayscale(image)
    try:
        mask = gray.point(lambda value: 255 if value < threshold else 0, mode="1")
        return mask.getbbox()
    finally:
        gray.close()


def _horizontal_projection(image: Image.Image, threshold: int = 242) -> list[int]:
    gray = ImageOps.grayscale(image)
    try:
        width, height = gray.size
        pixels = gray.load()
        return [sum(1 for x in range(width) if pixels[x, y] < threshold) for y in range(height)]
    finally:
        gray.close()


def _choose_split(projection: list[int], start: int, ideal: int, end: int, guard: int) -> int:
    low = max(start + guard, ideal - max(guard, (ideal - start) // 3))
    high = min(end - guard, ideal + max(guard, (ideal - start) // 3))
    if high <= low:
        return min(end, max(start + guard, ideal))
    return min(range(low, high + 1), key=lambda y: (projection[y], abs(y - ideal)))


def _split_column_image(
    image: Image.Image,
    *,
    output_dir: Path,
    stem: str,
    column_index: int,
    estimated_chars: int = 0,
    max_aspect: float = 7.2,
    max_chars: int = 12,
) -> list[OcrCropSegment]:
    bbox = _ink_bbox(image)
    if bbox is None:
        return []
    left, top, right, bottom = bbox
    ink_width = max(1, right - left)
    margin_x = max(8, round(ink_width * 0.18))
    margin_y = max(7, round(ink_width * 0.24))
    crop_left = max(0, left - margin_x)
    crop_right = min(image.width, right + margin_x)
    base = image.crop((crop_left, top, crop_right, bottom)).convert("RGB")
    try:
        projection = _horizontal_projection(base)
        total_height = base.height
        import math
        count_by_aspect = max(1, math.ceil(total_height / max(96.0, base.width * max_aspect)))
        count_by_chars = (
            max(1, math.ceil(max(1, int(estimated_chars or 0)) / max_chars))
            if estimated_chars else 1
        )
        segment_count = max(count_by_aspect, count_by_chars)
        target_height = max(72, round(total_height / segment_count))
        min_height = max(54, round(max(1, base.width) * 1.65))
        ranges: list[tuple[int, int]] = []
        cursor = 0
        while len(ranges) + 1 < segment_count and total_height - cursor > min_height * 1.2:
            ideal = min(total_height, cursor + target_height)
            split = _choose_split(
                projection, cursor, ideal, total_height,
                guard=max(10, round(base.width * 0.38)),
            )
            if split - cursor < min_height:
                split = min(total_height, cursor + target_height)
            ranges.append((cursor, split))
            cursor = split
        if cursor < total_height:
            if ranges and total_height - cursor < min_height * 0.55:
                ranges[-1] = (ranges[-1][0], total_height)
            else:
                ranges.append((cursor, total_height))
        if not ranges:
            ranges = [(0, total_height)]

        output: list[OcrCropSegment] = []
        for segment_index, (seg_top, seg_bottom) in enumerate(ranges):
            if seg_bottom <= seg_top:
                continue
            region = base.crop((0, seg_top, base.width, seg_bottom)).convert("RGB")
            try:
                segment_bbox = _ink_bbox(region)
                if segment_bbox is None:
                    continue
                sl, st, sr, sb = segment_bbox
                clean = region.crop((
                    max(0, sl - margin_x), max(0, st - margin_y),
                    min(region.width, sr + margin_x), min(region.height, sb + margin_y),
                )).convert("RGB")
                try:
                    digest = hashlib.sha1(
                        f"{stem}:{column_index}:{segment_index}:{seg_top}:{seg_bottom}".encode("utf-8")
                    ).hexdigest()[:10]
                    path = output_dir / f"{stem}_c{column_index:03d}_s{segment_index:03d}_{digest}.png"
                    clean.save(path, format="PNG", compress_level=1)
                finally:
                    clean.close()
            finally:
                region.close()
            if estimated_chars > 0:
                expected = max(
                    1,
                    round(estimated_chars * (seg_bottom - seg_top) / max(1, total_height)),
                )
            else:
                expected = max(1, round((seg_bottom - seg_top) / max(12.0, ink_width * 0.92)))
            output.append(OcrCropSegment(str(path), expected, column_index, segment_index))
        return output
    finally:
        base.close()


def prepare_vertical_ocr_segments(
    image_path: str,
    output_dir: Path,
    *,
    max_aspect: float = 7.2,
    max_chars: int = 12,
    already_isolated: bool = False,
    estimate_isolated_chars: bool = False,
    physical_column_boxes: tuple[tuple[int, int, int, int], ...] = (),
) -> tuple[list[OcrCropSegment], int]:
    """Create short, unrotated vertical chunks in Japanese reading order."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    try:
        bbox = _ink_bbox(image)
        if bbox is None:
            return [], 0

        stem = Path(image_path).stem.replace(" ", "_")[:48] or "ocr"
        if physical_column_boxes:
            segments = []
            for column_index, box in enumerate(physical_column_boxes):
                if len(box) != 4:
                    raise ValueError("Invalid physical column box")
                left, top, right, bottom = map(int, box)
                if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
                    raise ValueError("Physical column box is outside the retry image")
                column_image = image.crop((left, top, right, bottom))
                try:
                    column_bbox = _ink_bbox(column_image)
                    estimated_chars = 0
                    if column_bbox and estimate_isolated_chars:
                        l, t, r, b = column_bbox
                        estimated_chars = max(1, round((b - t) / max(12.0, (r - l) * 0.92)))
                    segments.extend(_split_column_image(
                        column_image, output_dir=output_dir, stem=stem,
                        column_index=column_index, estimated_chars=estimated_chars,
                        max_aspect=max_aspect, max_chars=max_chars,
                    ))
                finally:
                    column_image.close()
            return segments, len(physical_column_boxes)
        if already_isolated:
            estimated_chars = 0
            if estimate_isolated_chars:
                left, top, right, bottom = bbox
                ink_width = max(1, right - left)
                ink_height = max(1, bottom - top)
                estimated_chars = max(1, round(ink_height / max(12.0, ink_width * 0.92)))
            segments = _split_column_image(
                image,
                output_dir=output_dir,
                stem=stem,
                column_index=0,
                estimated_chars=estimated_chars,
                max_aspect=max_aspect,
                max_chars=max_chars,
            )
            return segments, 1 if segments else 0

        from adapters.column_ocr_adapter import detect_vertical_columns, _isolated_column_image
        detected = detect_vertical_columns(image, sensitivity=48, padding_percent=6, max_columns=48)

        if image.width <= 420:
            main_band = max(
                detected,
                key=lambda item: (
                    int(getattr(item, "width", 0) or 0),
                    float(getattr(item, "ink_score", 0.0) or 0.0),
                ),
                default=None,
            )
            if main_band is not None:
                columns = [main_band]
            else:
                class _CompactColumn:
                    hard_left, top, hard_right, bottom = bbox
                    left, right = bbox[0], bbox[2]
                    width = max(1, bbox[2] - bbox[0])
                    full_height_slot = False
                    estimated_chars = max(1, round((bbox[3] - bbox[1]) / max(12.0, width * 0.72)))
                columns = [_CompactColumn()]
        elif detected:
            columns = detected
        else:
            class _FallbackColumn:
                hard_left, top, hard_right, bottom = bbox
                left, right = bbox[0], bbox[2]
                width = max(1, bbox[2] - bbox[0])
                full_height_slot = False
                estimated_chars = max(1, round((bbox[3] - bbox[1]) / max(12.0, width * 0.72)))
            columns = [_FallbackColumn()]

        segments: list[OcrCropSegment] = []
        for column_index, column in enumerate(columns):
            compact = _isolated_column_image(image, column, retry=False, background=(255, 255, 255))
            try:
                segments.extend(_split_column_image(
                    compact,
                    output_dir=output_dir,
                    stem=stem,
                    column_index=column_index,
                    estimated_chars=int(getattr(column, "estimated_chars", 0) or 0),
                    max_aspect=max_aspect,
                    max_chars=max_chars,
                ))
            finally:
                compact.close()
        return segments, len(columns)
    finally:
        image.close()


def looks_like_full_page(path: str) -> bool:
    try:
        with Image.open(path) as image:
            width, height = image.size
    except Exception:
        return False
    if width < 700 or height < 900:
        return False
    area = width * height
    aspect = width / max(1, height)
    return area >= 900_000 and 0.42 <= aspect <= 1.35
