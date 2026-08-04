#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Targeted multi-view OCR rescue for uncertain single-glyph boxes.

The whole column OCR is still executed only once.  This module is invoked only
for slots that did not pass the conservative column gate.  It follows ideas
used by modern text-recognition projects:

* preserve token/character evidence instead of trusting one line-average score;
* use several deterministic views of the *same* glyph crop;
* reject unstable disagreement rather than selecting the highest-looking score;
* batch all views in one worker invocation to avoid per-glyph process startup.

Apple image OCR is intentionally forbidden here.  PKStroke remains a later
fallback for genuinely unresolved boxes and manual handwriting.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import re
import tempfile
import unicodedata
from typing import Callable, Mapping, Sequence

from PIL import Image, ImageEnhance, ImageOps, ImageFilter

from adapters.ocr_recognition_bridge import recognizer_iterator

_SPACE_RE = re.compile(r"\s+")
_INVALID = set("□■◼◻�?？")


@dataclass(slots=True)
class GlyphViewEvidence:
    view: str
    character: str = ""
    confidence: float = 0.0
    raw_text: str = ""
    error: str = ""


@dataclass(slots=True)
class GlyphRescueDecision:
    glyph_index: int
    original_character: str = ""
    candidate_character: str = ""
    score: float = 0.0
    support: int = 0
    total_views: int = 0
    disagreement_count: int = 0
    confirmed_original: bool = False
    confident_fill: bool = False
    reason: str = ""
    views: list[GlyphViewEvidence] = field(default_factory=list)

    @property
    def stable(self) -> bool:
        return bool(self.candidate_character and self.support >= 2 and self.disagreement_count == 0)


def _clean_text(value: object) -> str:
    text = unicodedata.normalize("NFC", str(value or ""))
    text = _SPACE_RE.sub("", text)
    return "".join(ch for ch in text if ch not in {"\ufeff", "\u200b", "\u2060"})


def _single_character_from_blocks(blocks: Sequence[Mapping[str, object]] | None) -> tuple[str, float, str]:
    """Return one strict character. Multi-character output is rejected."""
    candidates: list[tuple[str, float, str]] = []
    for block in blocks or []:
        raw = _clean_text(block.get("text", ""))
        if len(raw) != 1 or raw in _INVALID:
            continue
        try:
            score = max(0.0, min(1.0, float(block.get("confidence", 0.0) or 0.0)))
        except Exception:
            score = 0.0
        candidates.append((raw, score, raw))
    if not candidates:
        raw_joined = _clean_text("".join(str(block.get("text", "")) for block in (blocks or [])))
        return "", 0.0, raw_joined
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[0]


def _tight_crop(image: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray, cutoff=0)
    # A forgiving foreground mask. It keeps detached dakuten/handakuten while
    # removing large white margins introduced by projection boxes.
    mask = gray.point(lambda p: 255 if p < 238 else 0)
    bbox = mask.getbbox()
    if bbox:
        x0, y0, x1, y1 = bbox
        pad = max(2, round(max(x1 - x0, y1 - y0) * 0.10))
        bbox = (
            max(0, x0 - pad), max(0, y0 - pad),
            min(gray.width, x1 + pad), min(gray.height, y1 + pad),
        )
        gray = gray.crop(bbox)
    return gray


def _square_canvas(image: Image.Image, *, size: int = 192, margin: float = 0.16) -> Image.Image:
    source = image.convert("L")
    usable = max(16, int(size * (1.0 - 2.0 * margin)))
    ratio = min(usable / max(1, source.width), usable / max(1, source.height))
    new_size = (max(1, round(source.width * ratio)), max(1, round(source.height * ratio)))
    resized = source.resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new("L", (size, size), 255)
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return canvas


def _vertical_canvas(image: Image.Image, *, width: int = 128, height: int = 256) -> Image.Image:
    source = image.convert("L")
    usable_w = int(width * 0.72)
    usable_h = int(height * 0.58)
    ratio = min(usable_w / max(1, source.width), usable_h / max(1, source.height))
    resized = source.resize(
        (max(1, round(source.width * ratio)), max(1, round(source.height * ratio))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("L", (width, height), 255)
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return canvas


def build_glyph_views(image: Image.Image) -> dict[str, Image.Image]:
    """Three deterministic views, including a detector-friendly vertical strip."""
    tight = _tight_crop(image)
    base = _square_canvas(tight)
    contrast = ImageEnhance.Contrast(ImageOps.autocontrast(base)).enhance(1.35)
    vertical = _vertical_canvas(ImageEnhance.Contrast(tight).enhance(1.25))
    # Mild median filtering suppresses isolated scan noise without thickening
    # strokes as aggressively as morphology/dilation.
    clean = contrast.filter(ImageFilter.MedianFilter(size=3))
    hist = clean.histogram()
    total = sum(hist)
    cumulative = 0
    threshold = 180
    for i, count in enumerate(hist):
        cumulative += count
        if cumulative >= total * 0.18:
            threshold = min(225, max(105, i + 35))
            break
    binary = clean.point(lambda p, t=threshold: 0 if p < t else 255)
    return {
        "gray": base.convert("RGB"),
        "vertical": vertical.convert("RGB"),
        "binary": binary.convert("RGB"),
    }


class LowConfidenceGlyphRescuer:
    """Batch uncertain glyph views through one non-Apple OCR worker call."""

    def __init__(
        self,
        *,
        engine: str,
        engine_options: Mapping[str, object] | None = None,
        max_glyphs: int = 12,
        runner: Callable = recognizer_iterator,
    ):
        self.engine = str(engine or "")
        self.engine_options = dict(engine_options or {})
        # A single glyph does not need document layout/VLM parsing. Reuse the
        # lightweight Japanese recognition pipeline even when the page OCR used
        # PP-Structure or PaddleOCR-VL.
        if self.engine == "paddle_ocr":
            self.engine_options["pipeline"] = "ocr"
            self.engine_options.setdefault("lang", "japan")
        self.max_glyphs = max(0, int(max_glyphs or 0))
        self.runner = runner
        self.last_error = ""
        self.last_requested_indices: list[int] = []
        self.last_image_count = 0
        self._persistent_session = None

    @property
    def available(self) -> bool:
        return bool(self.engine and self.engine != "apple_vision" and self.max_glyphs > 0)

    def rescue(
        self,
        glyph_images: Sequence[Image.Image],
        *,
        target_indices: Sequence[int],
        original_characters: Mapping[int, str] | None = None,
    ) -> dict[int, GlyphRescueDecision]:
        self.last_error = ""
        self.last_requested_indices = []
        self.last_image_count = 0
        if not self.available:
            return {}
        valid_indices: list[int] = []
        for raw in target_indices:
            try:
                index = int(raw)
            except Exception:
                continue
            if 0 <= index < len(glyph_images) and index not in valid_indices:
                valid_indices.append(index)
            if len(valid_indices) >= self.max_glyphs:
                break
        if not valid_indices:
            return {}
        self.last_requested_indices = list(valid_indices)
        originals = {int(k): str(v or "")[:1] for k, v in (original_characters or {}).items()}

        decisions: dict[int, GlyphRescueDecision] = {
            index: GlyphRescueDecision(
                glyph_index=index,
                original_character=originals.get(index, ""),
            )
            for index in valid_indices
        }
        path_map: dict[str, tuple[int, str]] = {}
        try:
            with tempfile.TemporaryDirectory(prefix="novel_formatter_glyph_rescue_") as tmp_name:
                tmp = Path(tmp_name)
                image_paths: list[str] = []
                for index in valid_indices:
                    views = build_glyph_views(glyph_images[index])
                    for view_name, view_image in views.items():
                        path = tmp / f"glyph_{index:04d}_{view_name}.png"
                        view_image.save(path)
                        view_image.close()
                        image_paths.append(str(path))
                        path_map[str(path)] = (index, view_name)
                self.last_image_count = len(image_paths)
                manifest = tmp / "manifest.json"
                manifest.write_text("{}", encoding="utf-8")
                stream = None
                if self.runner is recognizer_iterator and self.engine in {"ndlocr_lite", "paddle_ocr"}:
                    try:
                        if self._persistent_session is None:
                            from adapters.persistent_recognition_session import PersistentRecognitionSession
                            self._persistent_session = PersistentRecognitionSession(
                                engine=self.engine, engine_options=self.engine_options,
                            )
                        stream = self._persistent_session.recognize(image_paths)
                    except Exception as exc:
                        self.last_error = str(exc)
                        stream = None
                if stream is None:
                    stream = self.runner(
                        self.engine,
                        image_paths,
                        str(manifest),
                        verbose=False,
                        engine_options=self.engine_options,
                    )
                for path, blocks, error in stream:
                    key = path_map.get(str(path))
                    if key is None:
                        # Some workers normalize/resolve the path.
                        key = path_map.get(str(Path(path)))
                    if key is None:
                        continue
                    index, view_name = key
                    char, confidence, raw_text = _single_character_from_blocks(blocks)
                    decisions[index].views.append(GlyphViewEvidence(
                        view=view_name,
                        character=char,
                        confidence=confidence,
                        raw_text=raw_text,
                        error=str(error or ""),
                    ))
        except Exception as exc:
            self.last_error = str(exc)
            return decisions

        for index, decision in decisions.items():
            usable = [view for view in decision.views if view.character]
            decision.total_views = len(decision.views)
            if not usable:
                decision.reason = "glyph_rescue_no_single_character"
                continue
            counts = Counter(view.character for view in usable)
            winner, support = counts.most_common(1)[0]
            runner_up = counts.most_common(2)[1][1] if len(counts) > 1 else 0
            winner_scores = [view.confidence for view in usable if view.character == winner]
            average = sum(winner_scores) / max(1, len(winner_scores))
            decision.candidate_character = winner
            decision.support = support
            decision.score = average
            decision.disagreement_count = sum(
                1 for view in usable if view.character != winner
            )
            original = decision.original_character
            # Safe acceptance rules:
            # * original column character: 3/3 agreement at moderate score, or
            #   2 agreeing views with no contradictory non-empty view at high score;
            # * empty slot: only unanimous 3-view, very-high-score fill;
            # * different candidate: expose for manual review, never silently replace.
            unanimous = support >= 3 and runner_up == 0
            two_clean = support >= 2 and runner_up == 0
            if original and winner == original and (
                (unanimous and average >= 0.82)
                or (two_clean and average >= 0.93)
            ):
                decision.confirmed_original = True
                decision.reason = "glyph_rescue_multiview_confirms_column_character"
            elif not original and unanimous and average >= 0.96:
                decision.confident_fill = True
                decision.reason = "glyph_rescue_unanimous_empty_slot_fill"
            elif winner != original:
                decision.reason = "glyph_rescue_candidate_disagrees_with_column"
            else:
                decision.reason = "glyph_rescue_insufficient_consensus"
        return decisions
    def close(self) -> None:
        session = self._persistent_session
        self._persistent_session = None
        if session is not None:
            try:
                session.close()
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

