#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conservative image variants for Japanese printed-text OCR.

The helpers in this module intentionally avoid geometry-changing cleanup on the
primary OCR image.  They are used only as targeted fallback variants when the
original masked column is empty or structurally suspicious.

No OpenCV/numpy dependency is required.  Pillow operations are deliberately
mild so dakuten, small kana, punctuation and thin kanji strokes are not erased.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps


@dataclass(frozen=True)
class OCRImageVariant:
    """One generated OCR fallback image.

    ``geometry`` documents whether the original page geometry is retained.
    This is useful for diagnostics and prevents a narrow isolated crop from
    being confused with the normal full-page masked input.
    """

    name: str
    image: Image.Image
    geometry: str


def _safe_radius(image: Image.Image, fraction: float, minimum: int, maximum: int) -> int:
    shortest = max(1, min(image.size))
    return max(minimum, min(maximum, round(shortest * fraction)))


def balance_paper_background(image: Image.Image) -> Image.Image:
    """Flatten uneven paper illumination without hard-thresholding strokes."""
    gray = ImageOps.grayscale(image)
    radius = _safe_radius(gray, 0.035, 12, 45)
    background = gray.filter(ImageFilter.GaussianBlur(radius=radius))
    try:
        # Local high-pass illumination normalization: equal paper regions become
        # white while darker printed strokes remain dark.  Unlike hard thresholding
        # this keeps antialiasing and tiny punctuation.
        normalized = ImageChops.subtract(gray, background, scale=1.0, offset=255)
    finally:
        background.close()
        gray.close()

    normalized = ImageOps.autocontrast(normalized, cutoff=0.35)
    contrasted = ImageEnhance.Contrast(normalized).enhance(1.10)
    normalized.close()
    sharpened = contrasted.filter(ImageFilter.UnsharpMask(radius=0.7, percent=55, threshold=4))
    contrasted.close()
    return sharpened.convert("RGB")


def adaptive_binary(image: Image.Image) -> Image.Image:
    """Create a gentle local-background binary candidate.

    This is not applied to the primary image.  It is a final fallback for paper
    shadows or faint print.  Local background subtraction is less destructive
    than a single global threshold on photographed pages.
    """
    gray = ImageOps.grayscale(image)
    radius = _safe_radius(gray, 0.018, 7, 25)
    local = gray.filter(ImageFilter.BoxBlur(radius=radius))
    try:
        # Dark ink produces a high positive difference (local - gray).
        ink_strength = ImageChops.subtract(local, gray, scale=1.0, offset=0)
    finally:
        local.close()
        gray.close()
    # Keep faint punctuation: threshold is intentionally conservative.
    binary = ink_strength.point(lambda value: 0 if value >= 12 else 255, mode="L")
    ink_strength.close()
    # Do not run morphology/median filtering here: dakuten, punctuation and small
    # kana can be only a few pixels wide and must survive this last-resort branch.
    result = binary.convert("RGB")
    binary.close()
    return result


def upscale_for_ocr(
    image: Image.Image,
    *,
    scale: float = 2.0,
    max_pixels: int = 12_000_000,
) -> Image.Image:
    """Upscale with Lanczos while bounding memory use."""
    width, height = image.size
    requested = max(1.0, float(scale))
    target_width = max(1, round(width * requested))
    target_height = max(1, round(height * requested))
    pixels = target_width * target_height
    if pixels > max_pixels:
        shrink = (max_pixels / max(1, pixels)) ** 0.5
        target_width = max(width, round(target_width * shrink))
        target_height = max(height, round(target_height * shrink))
    if (target_width, target_height) == image.size:
        return image.copy()
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)



def build_fallback_variant(
    name: str,
    *,
    masked_page: Image.Image,
    isolated_column: Image.Image,
) -> OCRImageVariant:
    """Build exactly one requested fallback image.

    The adaptive column pipeline assigns at most one rescue family to a physical
    column.  Building every historical variant would waste CPU and memory even
    when only one image can be submitted, so this helper materialises just the
    selected route.
    """
    key = str(name or "").strip().lower()
    if key == "balanced_full":
        return OCRImageVariant(key, balance_paper_background(masked_page), "full_page")
    if key == "balanced_crop_2x":
        balanced_crop = balance_paper_background(isolated_column)
        try:
            image = upscale_for_ocr(balanced_crop, scale=2.0)
        finally:
            balanced_crop.close()
        return OCRImageVariant(key, image, "isolated_column")
    if key == "adaptive_binary_crop_2x":
        binary = adaptive_binary(isolated_column)
        try:
            image = upscale_for_ocr(binary, scale=2.0)
        finally:
            binary.close()
        return OCRImageVariant(key, image, "isolated_column")
    raise ValueError(f"unknown OCR fallback variant: {name}")

def build_fallback_variants(
    *,
    masked_page: Image.Image,
    isolated_column: Image.Image,
    include_binary: bool = True,
) -> list[OCRImageVariant]:
    """Build ordered fallback variants for one suspicious column.

    The first variant retains full-page geometry.  The next variants isolate
    and enlarge the physical column, which helps engines that returned empty on
    a mostly-white masked page.  Callers own and must close all returned images.
    """
    variants: list[OCRImageVariant] = []

    variants.append(build_fallback_variant(
        "balanced_full", masked_page=masked_page, isolated_column=isolated_column
    ))
    variants.append(build_fallback_variant(
        "balanced_crop_2x", masked_page=masked_page, isolated_column=isolated_column
    ))

    if include_binary:
        variants.append(build_fallback_variant(
            "adaptive_binary_crop_2x",
            masked_page=masked_page,
            isolated_column=isolated_column,
        ))

    return variants


def close_variants(variants: Iterable[OCRImageVariant]) -> None:
    for variant in variants:
        try:
            variant.image.close()
        except Exception:
            pass
