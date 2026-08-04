#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apple Vision character anchors for one printed Japanese vertical column.

The column is masked to its dominant body-text band, tightly cropped and rotated
left by the bundled Swift helper, then submitted exactly once to
``RecognizeTextRequest`` in ``.fast`` mode.  The helper asks Vision for
``boundingBox(for:)`` on every Swift ``Character`` and maps the rectangles back
to the original vertical-column coordinates.

This module only supplies geometric anchors.  The existing independent pipeline
still crops every resulting box and performs its normal Apple single-glyph
recognition on that crop.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Iterable

from PIL import Image

from adapters.handwriting_image_tools import mask_main_text_band
from adapters.unicode_safety import clean_text
from adapters.vision_backends.base import OCRConfig
from adapters.vision_backends.native_helper_backend import NativeVisionHelperBackend


@dataclass(slots=True)
class AppleCharacterAnchor:
    text: str
    confidence: float
    bbox: tuple[float, float, float, float]

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": float(self.confidence),
            "x": float(self.bbox[0]),
            "y": float(self.bbox[1]),
            "width": float(self.bbox[2]),
            "height": float(self.bbox[3]),
            "source": "apple_vision_fast_character_box",
        }


class AppleVisionCharacterBoxRecognizer:
    """Persistent wrapper around the bundled Apple Vision Swift helper."""

    def __init__(self):
        self._backend = NativeVisionHelperBackend()
        ok, reason = self._backend.is_available()
        if not ok:
            raise RuntimeError(reason or "Apple Vision字符框不可用")
        self.last_error = ""

    @staticmethod
    def _one_character(value: object) -> str:
        text = clean_text(value).strip()
        if not text:
            return ""
        return next(iter(text), "")

    def recognize(self, image: Image.Image, *, timeout: float = 120.0) -> list[AppleCharacterAnchor]:
        masked, _mask_info = mask_main_text_band(image, preserve_large_symbols=True)
        try:
            with tempfile.TemporaryDirectory(prefix="novel_formatter_apple_char_boxes_") as temp_dir:
                image_path = Path(temp_dir) / "column.png"
                masked.save(image_path, format="PNG", compress_level=2)
                result = self._backend.recognize(
                    str(image_path),
                    OCRConfig(
                        recognition_level="fast",
                        languages=["ja-JP"],
                        vertical=True,
                        timeout=max(15.0, float(timeout)),
                        use_language_correction=False,
                        automatically_detect_language=False,
                        minimum_text_height_fraction=0.002,
                        candidate_count=1,
                        orientation="up",
                        vertical_compatibility_mode=True,
                        character_boxes=True,
                    ),
                )
        finally:
            masked.close()

        anchors: list[AppleCharacterAnchor] = []
        for block in result.blocks:
            char = self._one_character(block.text)
            if not char or block.bbox is None:
                continue
            x, y, width, height = (float(v) for v in block.bbox)
            if width <= 0.0 or height <= 0.0:
                continue
            anchors.append(AppleCharacterAnchor(
                text=char,
                confidence=max(0.0, min(1.0, float(block.confidence or 0.0))),
                bbox=(x, y, width, height),
            ))
        anchors.sort(key=lambda item: (-item.bbox[1], item.bbox[0]))
        return anchors

    def recognize_dicts(self, image: Image.Image, *, timeout: float = 120.0) -> list[dict]:
        try:
            anchors = self.recognize(image, timeout=timeout)
            self.last_error = ""
            return [item.as_dict() for item in anchors]
        except Exception as exc:
            self.last_error = clean_text(exc)
            return []

    def close(self) -> None:
        self._backend.close()

    def __del__(self):  # pragma: no cover - interpreter shutdown timing varies
        try:
            self.close()
        except Exception:
            pass
