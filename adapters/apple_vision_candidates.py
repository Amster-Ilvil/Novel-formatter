#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apple Vision single-glyph candidates, independent of PKStrokeRecognizer.

Printed-image OCR and PencilKit stroke recognition are separate Apple routes.
This module deliberately never requires the PKStroke bridge.  Interactive
review never uses the macOS Shortcuts route.  Automatic printed OCR does not
call this module; manual review opens the source image in macOS Preview so the
user can use the native Live Text selection/copy interaction directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any

from PIL import Image, ImageOps

from adapters.unicode_safety import clean_text
from adapters.vision_backends import OCRConfig
from adapters.vision_backends.native_helper_backend import (
    BINARY,
    HelperInfrastructureError,
    NativeVisionHelperBackend,
    VisionRecognitionError,
)


@dataclass(slots=True)
class VisionGlyphCandidate:
    text: str
    confidence: float
    rank: int
    language_correction: bool


def prepare_vision_glyph(image: Image.Image, *, size: int = 512) -> Image.Image:
    """Upscale a tight single-glyph crop for Apple image OCR."""
    gray = ImageOps.autocontrast(ImageOps.grayscale(image))
    mask = gray.point(lambda value: 255 if value < 242 else 0, mode="L")
    bbox = mask.getbbox()
    crop = gray.crop(bbox) if bbox else gray
    target = max(64, int(round(size * 0.68)))
    scale = min(target / max(1, crop.width), target / max(1, crop.height))
    resized = crop.resize(
        (max(1, int(round(crop.width * scale))), max(1, int(round(crop.height * scale)))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("L", (size, size), 255)
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return canvas.convert("RGB")


class AppleVisionCandidateRecognizer:
    """Return Apple image-OCR candidates for one segmented glyph.

    ``auto_build=False`` is the interactive-safe mode: clicking recognition
    never launches a potentially long Swift compiler job.  An existing helper
    is used immediately; without one, the already configured Apple Shortcut is
    tried.  ``auto_build=True`` remains available to non-interactive callers.
    """

    def __init__(self, *, auto_build: bool = True):
        self.auto_build = bool(auto_build)
        self._native = NativeVisionHelperBackend()
        self.last_result: dict[str, Any] = {}
        self.last_error = ""
        self.backend_used = ""

    @staticmethod
    def _base_glyph(text: str) -> str:
        value = "".join(ch for ch in clean_text(text or "") if not ch.isspace())
        return value[:1]

    @classmethod
    def _decode_groups(cls, payload: dict) -> dict[str, list[VisionGlyphCandidate]]:
        """Compatibility decoder for saved/native candidate payloads."""
        output: dict[str, list[VisionGlyphCandidate]] = {"corrected": [], "raw": []}
        for payload_key, group_name, corrected in (
            ("correctedCandidates", "corrected", True),
            ("rawCandidates", "raw", False),
        ):
            seen: set[str] = set()
            for index, item in enumerate(payload.get(payload_key) or []):
                if not isinstance(item, dict):
                    continue
                glyph = cls._base_glyph(item.get("text", ""))
                if not glyph or glyph in seen:
                    continue
                seen.add(glyph)
                output[group_name].append(VisionGlyphCandidate(
                    text=glyph,
                    confidence=max(0.0, min(1.0, float(item.get("confidence") or 0.0))),
                    rank=max(1, int(item.get("rank") or index + 1)),
                    language_correction=bool(item.get("languageCorrection", corrected)),
                ))
        return output

    @classmethod
    def _from_result(cls, result, *, corrected: bool, top_n: int) -> list[VisionGlyphCandidate]:
        ranked: list[tuple[str, float]] = []
        seen: set[str] = set()
        for block in list(getattr(result, "blocks", []) or []):
            for text, confidence in list(getattr(block, "candidates", []) or []):
                glyph = cls._base_glyph(text)
                if glyph and glyph not in seen:
                    seen.add(glyph)
                    ranked.append((glyph, float(confidence or 0.0)))
            glyph = cls._base_glyph(getattr(block, "text", ""))
            if glyph and glyph not in seen:
                seen.add(glyph)
                ranked.append((glyph, float(getattr(block, "confidence", 0.0) or 0.0)))
        for ch in clean_text(getattr(result, "full_text", "") or ""):
            glyph = cls._base_glyph(ch)
            if glyph and glyph not in seen:
                seen.add(glyph)
                ranked.append((glyph, 0.60))
        return [
            VisionGlyphCandidate(
                text=text,
                confidence=max(0.0, min(1.0, confidence)),
                rank=index + 1,
                language_correction=corrected,
            )
            for index, (text, confidence) in enumerate(ranked[: max(1, int(top_n))])
        ]

    @staticmethod
    def _config(*, corrected: bool, top_n: int, timeout: float) -> OCRConfig:
        return OCRConfig(
            recognition_level="accurate",
            languages=["ja-JP"],
            vertical=False,
            timeout=float(timeout),
            use_language_correction=bool(corrected),
            automatically_detect_language=False,
            minimum_text_height_fraction=0.0,
            candidate_count=max(1, min(10, int(top_n))),
            orientation="up",
            vertical_compatibility_mode=False,
            character_boxes=False,
        )

    def _recognize_native(self, image_path: str, *, corrected: bool, top_n: int, timeout: float):
        if not self.auto_build and not BINARY.exists():
            raise HelperInfrastructureError(
                "Apple Vision Helper 尚未编译；交互识别不会在点击后临时编译"
            )
        result = self._native.recognize(
            image_path,
            self._config(corrected=corrected, top_n=top_n, timeout=timeout),
        )
        self.backend_used = "apple_vision_recognize_text"
        return result

    def _recognize_fallback(self, image_path: str, *, top_n: int, timeout: float):
        del image_path, top_n, timeout
        raise HelperInfrastructureError(
            "交互式 Apple 图片取字不使用快捷指令；请在人工复核中用 macOS 预览实况文本打开原图"
        )

    def recognize_fast(
        self,
        image: Image.Image,
        *,
        top_n: int = 10,
        timeout: float = 14.0,
    ) -> dict[str, list[VisionGlyphCandidate]]:
        """Interactive single-glyph OCR with bounded per-route waits."""
        with tempfile.TemporaryDirectory(prefix="novel_formatter_vision_glyph_") as temp_dir:
            image_path = Path(temp_dir) / "glyph.png"
            prepare_vision_glyph(image).save(image_path, format="PNG", compress_level=2)
            errors: list[str] = []
            corrected_result = None

            # RecognizeTextRequest exposes real Top-N candidates. Prefer it only
            # when its binary already exists (or a non-interactive caller opted
            # into building it).
            try:
                corrected_result = self._recognize_native(
                    str(image_path), corrected=True, top_n=top_n, timeout=timeout,
                )
            except (HelperInfrastructureError, VisionRecognitionError, RuntimeError, TimeoutError) as exc:
                errors.append(str(exc))

            candidates = (
                self._from_result(corrected_result, corrected=True, top_n=top_n)
                if corrected_result else []
            )
            if not candidates:
                errors.append(
                    "未运行快捷指令回退；人工复核请使用 macOS 预览实况文本直接从原图复制"
                )

        self.last_error = "；".join(dict.fromkeys(error for error in errors if error))
        self.last_result = {
            "backend": self.backend_used,
            "corrected": [item.text for item in candidates],
            "raw": [],
            "error": self.last_error,
        }
        if not candidates:
            raise RuntimeError(self.last_error or "Apple Vision 未返回单字候选")
        return {"corrected": candidates, "raw": []}

    def recognize(self, image: Image.Image, *, top_n: int = 10) -> dict[str, list[VisionGlyphCandidate]]:
        """Batch/fusion route; keep corrected and raw Vision alternatives."""
        with tempfile.TemporaryDirectory(prefix="novel_formatter_vision_glyph_") as temp_dir:
            image_path = Path(temp_dir) / "glyph.png"
            prepare_vision_glyph(image).save(image_path, format="PNG", compress_level=2)
            corrected_result = raw_result = None
            errors: list[str] = []
            try:
                corrected_result = self._recognize_native(
                    str(image_path), corrected=True, top_n=top_n, timeout=24.0,
                )
            except (HelperInfrastructureError, VisionRecognitionError, RuntimeError, TimeoutError) as exc:
                errors.append(str(exc))
            if corrected_result is not None:
                try:
                    raw_result = self._recognize_native(
                        str(image_path), corrected=False, top_n=top_n, timeout=18.0,
                    )
                except (HelperInfrastructureError, VisionRecognitionError, RuntimeError, TimeoutError) as exc:
                    errors.append(str(exc))
            if corrected_result is None and raw_result is None:
                errors.append("Apple Vision Helper 不可用；已禁用快捷指令回退")

        corrected = self._from_result(corrected_result, corrected=True, top_n=top_n) if corrected_result else []
        raw = self._from_result(raw_result, corrected=False, top_n=top_n) if raw_result else []
        self.last_error = "；".join(dict.fromkeys(error for error in errors if error))
        self.last_result = {
            "backend": self.backend_used,
            "corrected": [item.text for item in corrected],
            "raw": [item.text for item in raw],
            "error": self.last_error,
        }
        if not corrected and not raw:
            raise RuntimeError(self.last_error or "Apple Vision 未返回单字候选")
        return {"corrected": corrected, "raw": raw}

    def recognize_many(
        self,
        images: list[Image.Image] | tuple[Image.Image, ...],
        *,
        top_n: int = 10,
    ) -> list[dict[str, list[VisionGlyphCandidate]]]:
        results: list[dict[str, list[VisionGlyphCandidate]]] = []
        errors: list[str] = []
        for image in images:
            try:
                results.append(self.recognize(image, top_n=top_n))
            except Exception as exc:
                errors.append(str(exc))
                results.append({"corrected": [], "raw": []})
        if errors:
            self.last_error = "；".join(dict.fromkeys(errors[-3:]))
        return results

    def close(self) -> None:
        try:
            self._native.close()
        except Exception:
            pass
