#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent Japanese printed-glyph recognition card backend.

The GUI path derives physical columns and equal character cells from black
pixels, extracts a complete trace for each occupied cell, and uses only Apple
PKStrokeRecognizer (optionally Apple Vision candidates and explicit glyph
memory). OpenVINO/JLect are not automatic recognition fallbacks. Apple bridge
failures are recoverable: the existing cell is kept as ``□`` instead of
terminating the application or inserting another symbol.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import ast
import hashlib
import math
import os
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFilter, ImageOps

from models.document import Block, BlockType, TocEntry, UnifiedDocument
from adapters.handwriting_trace_review import prepare_review_records
from adapters.unicode_safety import clean_text
from adapters.handwriting_image_tools import (
    _connected_components, mask_main_text_band, mask_single_glyph,
    repair_precomputed_glyph_boxes_with_ocr_text, segment_black_ink_glyphs,
    segment_black_ink_glyphs_slider,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_JLECT_JS = _PROJECT_ROOT / "third_party" / "jlect_jhr" / "jlect-jhr.compressed.js"

_PUNCTUATION = set("。、，！？!?,.；;：:（）()［］【】〔〕〈〉《》「」『』“”‘’ー―…・〜～　 ")
_JP_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff々〆ヶ]")
_UNKNOWN_CHARS = {"?", "？", "□", "■", "◻", "◼", "_"}


@dataclass(slots=True)
class HandwritingCandidate:
    text: str
    score: float
    source: str = "jlect_python"
    pattern: str = ""
    reason: str = ""


@dataclass(slots=True)
class AutoCharDecision:
    index: int
    ocr_char: str
    chosen_char: str
    changed: bool
    score: float = 0.0
    reason: str = ""
    candidates: list[HandwritingCandidate] = field(default_factory=list)


@dataclass(slots=True)
class AutoColumnResult:
    block_id: str
    original_text: str
    text: str
    changed: bool
    reviewed: bool
    auto_segments: int
    auto_changed_chars: int
    decisions: list[AutoCharDecision] = field(default_factory=list)


@dataclass(slots=True)
class GlyphStats:
    width: int
    height: int
    black_pixels: int
    components: int
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    bbox_area_ratio: float
    fill_ratio: float
    center_x: float
    center_y: float
    comp_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)


_JLECT_CACHE: list[tuple[str, str, int]] | None = None


def _append_modified_by(value: str, name: str) -> str:
    parts = [item for item in str(value or "").split(",") if item]
    if name not in parts:
        parts.append(name)
    return ",".join(parts)


def _review_evidence_from_decision(decision: AutoCharDecision) -> dict | None:
    """Return non-destructive review evidence for one physical character slot.

    A disagreement is not limited to an automatic replacement.  When the OCR
    character remains selected but a visually plausible alternative has nearly
    the same score, that ambiguity is precisely what a human should inspect.
    Expected-character promotion in ``recognize_image_candidates`` therefore
    cannot hide a close alternative.
    """
    ocr_char = str(decision.ocr_char or "")
    chosen_char = str(decision.chosen_char or "")
    unique: dict[str, HandwritingCandidate] = {}
    for candidate in decision.candidates or []:
        text = str(candidate.text or "")
        if not text:
            continue
        previous = unique.get(text)
        if previous is None or float(candidate.score or 0.0) > float(previous.score or 0.0):
            unique[text] = candidate

    if chosen_char and chosen_char != ocr_char:
        candidate = unique.get(chosen_char)
        score = float(candidate.score if candidate is not None else decision.score or 0.0)
        return {
            "index": int(decision.index),
            "ocr": ocr_char,
            "candidate": chosen_char,
            "score": round(score, 4),
            "ocr_score": round(float(unique.get(ocr_char).score), 4) if ocr_char in unique else 0.0,
            "margin": None,
            "ambiguous": False,
            "reason": str(decision.reason or "candidate_differs_from_ocr"),
            "candidates": [cand.text for cand in decision.candidates[:5]],
        }

    if not ocr_char:
        return None
    alternatives = sorted(
        (candidate for text, candidate in unique.items() if text != ocr_char),
        key=lambda item: (-float(item.score or 0.0), item.text),
    )
    if not alternatives:
        return None
    alternative = alternatives[0]
    alternative_score = float(alternative.score or 0.0)
    ocr_candidate = unique.get(ocr_char)
    ocr_score = float(ocr_candidate.score or 0.0) if ocr_candidate is not None else 0.0
    margin = ocr_score - alternative_score if ocr_candidate is not None else None
    symbol = _is_symbol_char(ocr_char) or _is_symbol_char(str(alternative.text or ""))
    if ocr_candidate is None:
        credible = alternative_score >= (0.96 if symbol else 0.92)
    else:
        credible = (
            alternative_score >= (0.94 if symbol else 0.90)
            and margin <= (0.08 if symbol else 0.16)
        ) or (
            alternative_score >= (0.88 if symbol else 0.82)
            and margin <= (0.035 if symbol else 0.06)
        )
    if not credible:
        return None
    return {
        "index": int(decision.index),
        "ocr": ocr_char,
        "candidate": str(alternative.text or ""),
        "score": round(alternative_score, 4),
        "ocr_score": round(ocr_score, 4),
        "margin": round(float(margin), 4) if margin is not None else None,
        "ambiguous": True,
        "reason": "close_visual_alternative_to_ocr",
        "candidates": [cand.text for cand in decision.candidates[:5]],
    }


def _is_japanese_char(ch: str) -> bool:
    return bool(ch and _JP_RE.search(ch))


def _is_symbol_char(ch: str) -> bool:
    return bool(ch and ch in _PUNCTUATION)


def _should_attempt_recognition(ch: str) -> bool:
    if ch is None:
        return False
    s = str(ch)
    return bool(s and s not in {" ", "\t", "\n", "\r", "　"})



def _primary_ocr_geometry_sequence(
    candidate_text: str,
    *,
    raw_confidence: float,
    engine: str,
    threshold: float,
) -> str:
    """Return a high-confidence ordinary-OCR sequence for geometry repair only.

    Unlike automatic text acceptance, this helper may use one ordinary OCR
    engine by itself.  The sequence never writes characters and never bypasses
    manual review; it can only split an obviously merged cached box or discard a
    clearly tiny noise fragment.  Apple image OCR and engines with synthetic
    confidence values remain excluded.
    """
    try:
        from adapters.selective_column_ocr import clean_column_text, engine_profile
    except Exception:
        return ""
    text = clean_column_text(candidate_text)
    if len(text) < 2 or any(ch in _UNKNOWN_CHARS or ch.isspace() for ch in text):
        return ""
    try:
        confidence = max(0.0, min(1.0, float(raw_confidence or 0.0)))
        configured = max(0.0, min(1.0, float(threshold or 0.95)))
    except Exception:
        return ""
    profile = engine_profile(engine)
    if not profile.automatic_allowed or not profile.confidence_is_real:
        return ""
    effective = max(0.0, confidence - float(profile.confidence_penalty or 0.0))
    required = max(0.88, configured - 0.045)
    return text if effective >= required else ""


def _stable_sequence_count_hint(
    candidate_text: str,
    *,
    raw_confidence: float,
    engine: str,
    threshold: float,
    variants: Sequence[dict] | None,
    require_stability: bool,
) -> int | None:
    """Return a conservative OCR length hint for one repair-only resegmentation.

    The hint never writes OCR characters and never forces a box count. In the
    default stable mode, two distinct OCR engines must return the exact same
    normalized sequence. This is used only when cached geometry differs by one
    or two boxes, to retry the black-ink splitter with a better pitch prior.
    """
    try:
        from adapters.selective_column_ocr import (
            clean_column_text, engine_profile, normalize_engine_name,
        )
    except Exception:
        return None
    text = clean_column_text(candidate_text)
    if not text:
        return None
    try:
        confidence = max(0.0, min(1.0, float(raw_confidence or 0.0)))
        required = max(0.0, min(1.0, float(threshold or 0.95)))
    except Exception:
        return None
    profile = engine_profile(engine)
    if not profile.automatic_allowed or confidence < required:
        return None
    primary_engine = normalize_engine_name(engine)
    supporting_engines = {primary_engine}
    for raw in variants or []:
        if not isinstance(raw, dict):
            continue
        variant_text = clean_column_text(str(raw.get("text") or ""))
        if variant_text != text:
            continue
        try:
            variant_confidence = max(0.0, min(1.0, float(raw.get("confidence") or 0.0)))
        except Exception:
            variant_confidence = 0.0
        variant_engine = normalize_engine_name(str(raw.get("engine") or raw.get("label") or ""))
        variant_profile = engine_profile(variant_engine)
        if not variant_profile.automatic_allowed or variant_confidence < max(0.0, required - 0.025):
            continue
        supporting_engines.add(variant_engine)
    if require_stability and len(supporting_engines) < 2:
        return None
    return len(text)


def _load_jlect_table() -> list[tuple[str, str, int]]:
    global _JLECT_CACHE
    if _JLECT_CACHE is not None:
        return _JLECT_CACHE
    if not _JLECT_JS.exists():
        _JLECT_CACHE = []
        return _JLECT_CACHE
    raw = _JLECT_JS.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"kanji=\[(.*?)]\s*,testk=", raw, flags=re.S)
    if not m:
        _JLECT_CACHE = []
        return _JLECT_CACHE
    payload = "[" + m.group(1) + "]"
    table = ast.literal_eval(payload)
    parsed: list[tuple[str, str, int]] = []
    for item in table:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        ch = str(item[0] or "")
        pattern = str(item[1] or "")
        if not ch or not pattern:
            continue
        try:
            direction = int(item[2]) if len(item) >= 3 and item[2] is not None else 0
        except Exception:
            direction = 0
        parsed.append((ch, pattern, direction))
    _JLECT_CACHE = parsed
    return parsed


class JapaneseHandwritingCard:
    def __init__(
        self,
        *,
        recognition_backend: str = "auto",
        vision_candidate_fusion: bool = False,
        glyph_memory_enabled: bool = False,
        glyph_memory_scope_key: str = "",
        glyph_memory_include_global: bool = True,
        low_confidence_glyph_rescuer=None,
    ):
        self.engine = None
        self.recognition_backend = str(recognition_backend or "auto")
        self.vision_candidate_fusion = bool(vision_candidate_fusion)
        self.glyph_memory_enabled = bool(glyph_memory_enabled)
        self.glyph_memory_scope_key = str(glyph_memory_scope_key or "__global__")
        self.glyph_memory_include_global = bool(glyph_memory_include_global)
        # Retain Novel-formatter-1's dependency-free fallback and OpenVINO path.
        # The newer glyph fusion layer treats them as fallbacks; it does not make
        # them authoritative over ordinary OCR or human review.
        self._table = _load_jlect_table()
        self._apple = None
        self._apple_error = ""
        self._openvino = None
        self._openvino_error = ""
        self._last_backend = "jlect_black_ink"
        self._last_fallback_note = ""
        self._last_apple_attempted = False
        self._last_apple_succeeded = False
        self._vision_candidates = None
        self._vision_candidate_error = ""
        self._column_anchor_recognizer = None
        self._column_anchor_error = ""
        self._glyph_memory = None
        self._last_glyph_images: list[Image.Image] = []
        self._last_glyph_segments = []
        self._last_segmentation_info: dict = {}
        self._last_apple_details: list[dict] = []
        self._last_vision_groups: list[object] = []
        self._learning_session = None
        self._learning_context = {"page": 0, "column": 0, "block_id": ""}
        self._last_selective_report = None
        self._last_apple_requested_indices: list[int] = []
        self._low_confidence_glyph_rescuer = low_confidence_glyph_rescuer
        self._last_glyph_rescue_results: dict[int, object] = {}
        self._last_glyph_rescue_requested_indices: list[int] = []
        self._last_glyph_rescue_error = ""
        self._session_exact_glyph_cache: dict[str, str] = {}

        if self.glyph_memory_enabled:
            try:
                from adapters.glyph_memory_db import GlyphMemoryDB
                self._glyph_memory = GlyphMemoryDB()
            except Exception as exc:
                self._vision_candidate_error = f"字形记忆库不可用：{exc}"

        if self.vision_candidate_fusion:
            try:
                from adapters.apple_vision_candidates import AppleVisionCandidateRecognizer
                self._vision_candidates = AppleVisionCandidateRecognizer(auto_build=False)
            except Exception as exc:
                self._vision_candidate_error = str(exc)

        if self.recognition_backend in {"auto", "apple"}:
            try:
                from adapters.apple_pkstroke_engine import ApplePKStrokeRecognizer
                # Preserve the original explicit Apple mode behaviour. Auto mode
                # may build once, then the protocol-11 bridge remains persistent.
                self._apple = ApplePKStrokeRecognizer(auto_build=True)
                self._last_backend = "apple_pkstroke"
            except Exception as exc:
                # Native Apple support is optional. Keep the selected backend
                # visible in diagnostics, but fall through to the preserved
                # OpenVINO/JLect paths instead of aborting the entire OCR run.
                self._apple_error = str(exc)

        if self._apple is None and self.recognition_backend in {"auto", "openvino"}:
            self._ensure_openvino(strict=self.recognition_backend == "openvino")

    def set_learning_session(self, session) -> None:
        self._learning_session = session

    def close(self) -> None:
        apple = self._apple
        if apple is not None:
            try:
                close = getattr(apple, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass

    def set_learning_context(self, *, page: int, column: int, block_id: str = "") -> None:
        self._learning_context = {
            "page": int(page or 0), "column": int(column or 0),
            "block_id": clean_text(block_id),
        }

    def _ensure_openvino(self, *, strict: bool = False) -> bool:
        """Lazily prepare the original raster handwriting fallback."""
        if self._openvino is not None:
            return True
        try:
            from adapters.openvino_handwriting_engine import OpenVINOJapaneseHandwritingRecognizer
            self._openvino = OpenVINOJapaneseHandwritingRecognizer()
            return True
        except Exception as exc:
            self._openvino_error = str(exc)
            if strict:
                raise
            return False

    @property
    def active_backend(self) -> str:
        return str(self._last_backend or "apple_pkstroke_unavailable")

    # ---- image preprocessing -------------------------------------------------
    @staticmethod
    def _to_grayscale(image: Image.Image) -> Image.Image:
        if image.mode != "L":
            image = ImageOps.grayscale(image)
        return image

    @staticmethod
    def _gray_data(image: Image.Image) -> list[int]:
        getter = getattr(image, "get_flattened_data", None)
        if callable(getter):
            return list(getter())
        return list(image.getdata())

    @staticmethod
    def _otsu(gray: Sequence[int]) -> int:
        hist = [0] * 256
        for v in gray:
            hist[int(v)] += 1
        total = len(gray)
        if total <= 0:
            return 180
        sum_all = sum(i * hist[i] for i in range(256))
        w_b = 0
        sum_b = 0
        best = 127
        best_var = -1.0
        for t in range(256):
            w_b += hist[t]
            if not w_b:
                continue
            w_f = total - w_b
            if not w_f:
                break
            sum_b += t * hist[t]
            m_b = sum_b / w_b
            m_f = (sum_all - sum_b) / w_f
            var = w_b * w_f * (m_b - m_f) * (m_b - m_f)
            if var > best_var:
                best_var = var
                best = t
        return min(235, max(70, best + 8))

    @staticmethod
    def _binary_from_image(image: Image.Image) -> tuple[list[int], int, int]:
        gray_img = JapaneseHandwritingCard._to_grayscale(image)
        gray = JapaneseHandwritingCard._gray_data(gray_img)
        w, h = gray_img.size
        th = JapaneseHandwritingCard._otsu(gray)
        binary = [1 if v < th else 0 for v in gray]
        return binary, w, h

    @staticmethod
    def _thin_zhang_suen(binary: list[int], w: int, h: int) -> list[int]:
        img = binary[:]

        def at(x: int, y: int) -> int:
            return img[y * w + x]

        changed = True
        rounds = 0
        while changed and rounds < 80:
            changed = False
            rounds += 1
            for phase in (0, 1):
                remove: list[int] = []
                for y in range(1, h - 1):
                    for x in range(1, w - 1):
                        if not at(x, y):
                            continue
                        p = [
                            at(x, y - 1), at(x + 1, y - 1), at(x + 1, y), at(x + 1, y + 1),
                            at(x, y + 1), at(x - 1, y + 1), at(x - 1, y), at(x - 1, y - 1),
                        ]
                        n = sum(p)
                        if n < 2 or n > 6:
                            continue
                        trans = 0
                        for i in range(8):
                            if not p[i] and p[(i + 1) % 8]:
                                trans += 1
                        if trans != 1:
                            continue
                        if phase == 0:
                            if p[0] * p[2] * p[4] or p[2] * p[4] * p[6]:
                                continue
                        else:
                            if p[0] * p[2] * p[6] or p[0] * p[4] * p[6]:
                                continue
                        remove.append(y * w + x)
                if remove:
                    changed = True
                    for idx in remove:
                        img[idx] = 0
        return img

    @staticmethod
    def _skeleton_paths(binary: Sequence[int], w: int, h: int) -> list[list[tuple[int, int]]]:
        def key(x: int, y: int) -> int:
            return y * w + x

        def xy(i: int) -> tuple[int, int]:
            return (i % w, i // w)

        def neighbors(i: int) -> list[int]:
            x, y = xy(i)
            out: list[int] = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if not dx and not dy:
                        continue
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < w and 0 <= ny < h and binary[key(nx, ny)]):
                        continue
                    if dx and dy:
                        # Suppress diagonal shortcut edges around an orthogonal
                        # corner/junction. True diagonal strokes have neither
                        # orthogonal bridge pixel and remain connected.
                        orth_a = binary[key(x + dx, y)] if 0 <= x + dx < w else 0
                        orth_b = binary[key(x, y + dy)] if 0 <= y + dy < h else 0
                        if orth_a or orth_b:
                            continue
                    out.append(key(nx, ny))
            return out

        pixels = [i for i, v in enumerate(binary) if v]
        if not pixels:
            return []
        nodes = {i for i in pixels if len(neighbors(i)) != 2}
        used: set[tuple[int, int]] = set()
        paths: list[list[int]] = []

        def ekey(a: int, b: int) -> tuple[int, int]:
            return (a, b) if a < b else (b, a)

        def trace(start: int, nxt: int) -> list[int]:
            p = [start]
            seen = {start}
            prev, cur = start, nxt
            used.add(ekey(prev, cur))
            while True:
                p.append(cur)
                if cur in nodes and cur != start:
                    break
                choices = [n for n in neighbors(cur) if n != prev and ekey(cur, n) not in used]
                if not choices:
                    break
                n = choices[0]
                prev, cur = cur, n
                used.add(ekey(prev, cur))
                if cur in seen:
                    break
                seen.add(cur)
            return p

        for n in nodes:
            for m in neighbors(n):
                if ekey(n, m) not in used:
                    paths.append(trace(n, m))
        for p in pixels:
            for m in neighbors(p):
                if ekey(p, m) not in used:
                    paths.append(trace(p, m))
        out = [[xy(i) for i in path] for path in paths]
        covered = {point for path in out for point in path}
        # A small dot can thin to a single isolated pixel. PencilKit needs at
        # least two points, so represent it as a tiny tap-like segment rather
        # than silently deleting dakuten, handakuten or punctuation.
        for pixel in pixels:
            point = xy(pixel)
            if point in covered:
                continue
            x, y = point
            x2 = min(w - 1, x + 1) if x + 1 < w else max(0, x - 1)
            out.append([(x, y), (x2, y)])
        return [p for p in out if len(p) >= 2]

    @staticmethod
    def _simplify(points: list[tuple[int, int]], eps: float) -> list[tuple[int, int]]:
        if len(points) < 3:
            return points
        a = points[0]
        b = points[-1]
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        den = math.hypot(dx, dy) or 1.0
        max_d = 0.0
        idx = 0
        for i in range(1, len(points) - 1):
            p = points[i]
            d = abs(dy * p[0] - dx * p[1] + b[0] * a[1] - b[1] * a[0]) / den
            if d > max_d:
                max_d = d
                idx = i
        if max_d > eps:
            left = JapaneseHandwritingCard._simplify(points[: idx + 1], eps)
            right = JapaneseHandwritingCard._simplify(points[idx:], eps)
            return left[:-1] + right
        return [a, b]

    @staticmethod
    def _path_code(points: Sequence[tuple[int, int]]) -> str:
        if len(points) < 2:
            return ""
        start = points[0]
        end = points[-1]
        angle = (round((math.atan2(start[1] - end[1], start[0] - end[0]) * (180.0 / math.pi) + 180.0) / 45.0) * 45) % 360
        if angle in (0, 180, 360):
            return "H"
        if angle in (90, 270):
            return "V"
        if angle in (45, 225):
            return "2"
        if angle in (135, 315):
            return "3"
        return ""

    @staticmethod
    def _direction_shift_count(paths: Sequence[Sequence[tuple[int, int]]]) -> int:
        shifts = 0
        prev_sign: int | None = None
        for path in paths:
            if len(path) < 2:
                continue
            dx = path[-1][0] - path[0][0]
            if abs(dx) <= 2:
                continue
            sign = 1 if dx < 0 else 0
            if prev_sign is not None and sign != prev_sign:
                shifts += 1
            prev_sign = sign
        return shifts

    @staticmethod
    def _binary_stats(binary: Sequence[int], w: int, h: int) -> GlyphStats | None:
        """Measure glyph geometry with the shared accelerated CC backend."""
        if w <= 0 or h <= 0:
            return None
        components = _connected_components(binary, w, h)
        if not components:
            return None
        black = sum(int(item.get("area", 0)) for item in components)
        if black <= 0:
            return None
        x0 = min(int(item["x0"]) for item in components)
        x1_exclusive = max(int(item["x1"]) for item in components)
        y0 = min(int(item["y0"]) for item in components)
        y1_exclusive = max(int(item["y1"]) for item in components)
        bw = max(1, x1_exclusive - x0)
        bh = max(1, y1_exclusive - y0)
        bbox_area = bw * bh
        sum_x = 0
        sum_y = 0
        comp_boxes: list[tuple[int, int, int, int]] = []
        for item in components:
            comp_boxes.append((
                int(item["x0"]), int(item["y0"]),
                max(int(item["x0"]), int(item["x1"]) - 1),
                max(int(item["y0"]), int(item["y1"]) - 1),
            ))
            for pixel_index in item.get("pixels", []):
                pixel_index = int(pixel_index)
                sum_x += pixel_index % w
                sum_y += pixel_index // w

        return GlyphStats(
            width=w,
            height=h,
            black_pixels=black,
            components=len(components),
            bbox_x=x0 / max(1, w),
            bbox_y=y0 / max(1, h),
            bbox_w=bw / max(1, w),
            bbox_h=bh / max(1, h),
            bbox_area_ratio=bbox_area / max(1, w * h),
            fill_ratio=black / max(1, bbox_area),
            center_x=(sum_x / black) / max(1, w),
            center_y=(sum_y / black) / max(1, h),
            comp_boxes=comp_boxes,
        )

    def recognize_pattern(self, pattern: str, *, direction_shift: int = 0, top_n: int = 12) -> list[HandwritingCandidate]:
        if not pattern or not self._table:
            return []
        scores: dict[str, HandwritingCandidate] = {}
        pv = pattern.count("V")
        ph = pattern.count("H")
        p2 = pattern.count("2")
        p3 = pattern.count("3")
        pdiag = p2 + p3
        for ch, candidate_pattern, candidate_dir in self._table:
            reason = ""
            score = 0.0
            if candidate_pattern == pattern:
                score = 0.98 if candidate_dir == direction_shift else 0.94
                reason = "exact"
            elif len(pattern) >= 4 and candidate_pattern.startswith(pattern):
                score = 0.82 - max(0, len(candidate_pattern) - len(pattern)) * 0.015
                reason = "prefix"
            elif len(pattern) >= 2 and pattern in candidate_pattern:
                score = 0.66 - max(0, len(candidate_pattern) - len(pattern)) * 0.01
                reason = "contains"
            else:
                cv = candidate_pattern.count("V")
                chn = candidate_pattern.count("H")
                c2 = candidate_pattern.count("2")
                c3 = candidate_pattern.count("3")
                cdiag = c2 + c3
                if abs(cv - pv) <= 1 and abs(chn - ph) <= 2:
                    if len(pattern) < 5:
                        if cv == pv and chn == ph and abs(cdiag - pdiag) <= 1 and _is_japanese_char(ch):
                            score = 0.53
                            reason = "fuzzy"
                    elif abs(direction_shift - candidate_dir) <= 1 and _is_japanese_char(ch):
                        score = 0.56
                        reason = "fuzzy"
            if score <= 0.0:
                continue
            prev = scores.get(ch)
            cand = HandwritingCandidate(text=ch, score=round(score, 4), pattern=candidate_pattern, reason=reason)
            if prev is None or cand.score > prev.score:
                scores[ch] = cand
        ranked = sorted(scores.values(), key=lambda c: (-c.score, len(c.pattern), c.text))
        return ranked[:top_n]

    def recognize_strokes(self, strokes):
        pattern = "".join(self._path_code(path) for path in strokes if len(path) >= 2)
        return self.recognize_pattern(pattern, direction_shift=self._direction_shift_count(strokes))

    @staticmethod
    def _prepare_char_image(image: Image.Image, *, target_size: int = 192) -> Image.Image | None:
        """Create a high-resolution binary glyph without deleting components.

        Threshold first, crop the complete ink bbox, then resize the binary mask.
        The previous grayscale 96×96 resize blurred narrow strokes and could
        break them at the second threshold. A 192×192 binary work grid keeps
        hooks, short bars and kana curves connected before thinning.
        """
        gray = JapaneseHandwritingCard._to_grayscale(image)
        values = JapaneseHandwritingCard._gray_data(gray)
        th = JapaneseHandwritingCard._otsu(values)
        w, h = gray.size
        mask = Image.new("L", (w, h), 255)
        mask.putdata([0 if value < th else 255 for value in values])
        ink_bbox = ImageOps.invert(mask).getbbox()
        if ink_bbox is None:
            return None
        x0, y0, x1, y1 = ink_bbox
        x0, y0 = max(0, x0 - 1), max(0, y0 - 1)
        x1, y1 = min(w, x1 + 1), min(h, y1 + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        crop = mask.crop((x0, y0, x1, y1))
        side = max(crop.size)
        padding = max(8, int(round(side * 0.10)))
        canvas_side = side + padding * 2
        canvas = Image.new("L", (canvas_side, canvas_side), 255)
        ox = (canvas_side - crop.size[0]) // 2
        oy = (canvas_side - crop.size[1]) // 2
        canvas.paste(crop, (ox, oy))
        prepared = canvas.resize((target_size, target_size), Image.Resampling.NEAREST)

        # Heal only one-pixel raster gaps introduced by scanning/resizing. This
        # is a black-ink closing operation and is intentionally very mild.
        black_pixels = sum(1 for value in JapaneseHandwritingCard._gray_data(prepared) if value < 128)
        if black_pixels >= 24:
            prepared = prepared.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
        return prepared

    def recognize_symbol_candidates(self, image: Image.Image, *, use_glyph_mask: bool = True) -> list[HandwritingCandidate]:
        # Symbol recognition inspects the segment at its original size. Optional
        # glyph masking removes adjacent ruby / neighbour residue first.
        cleaned = mask_single_glyph(image) if use_glyph_mask else image.convert("RGB")
        gray = self._to_grayscale(cleaned)
        binary, w, h = self._binary_from_image(gray)
        stats = self._binary_stats(binary, w, h)
        if stats is None:
            return []
        candidates: list[HandwritingCandidate] = []
        bx, by, bw, bh = stats.bbox_x, stats.bbox_y, stats.bbox_w, stats.bbox_h
        fill = stats.fill_ratio
        cx, cy = stats.center_x, stats.center_y
        pixel_w = max(1.0, bw * stats.width)
        pixel_h = max(1.0, bh * stats.height)
        horizontal_aspect = pixel_w / pixel_h
        vertical_aspect = pixel_h / pixel_w

        # Single small dot / comma-like symbols. Segmentation is often tight in
        # the vertical direction, so use column-relative width / absolute ink
        # mass in addition to normalized glyph height.
        compact_dot = (
            stats.components <= 2
            and bw <= 0.34
            and (
                bh <= 0.40
                or stats.black_pixels <= max(18, int(stats.width * stats.width * 0.085))
            )
        )
        if compact_dot:
            # Distinguish the hollow Japanese period from the solid, diagonal
            # comma. The former x-position-only heuristic classified many
            # vertical commas on the right side of the cell as ``。``.
            x0_px = max(0, int(round(bx * w)))
            y0_px = max(0, int(round(by * h)))
            x1_px = min(w, int(round((bx + bw) * w)))
            y1_px = min(h, int(round((by + bh) * h)))

            def has_enclosed_hole() -> bool:
                if x1_px - x0_px < 4 or y1_px - y0_px < 4:
                    return False
                white = {
                    (xx, yy) for yy in range(y0_px, y1_px) for xx in range(x0_px, x1_px)
                    if not binary[yy * w + xx]
                }
                stack = [
                    point for point in white
                    if point[0] in {x0_px, x1_px - 1} or point[1] in {y0_px, y1_px - 1}
                ]
                outside = set(stack)
                while stack:
                    xx, yy = stack.pop()
                    for nx, ny in ((xx-1,yy),(xx+1,yy),(xx,yy-1),(xx,yy+1)):
                        point = (nx, ny)
                        if point in white and point not in outside:
                            outside.add(point)
                            stack.append(point)
                return len(white - outside) >= 2

            points = [(idx % w, idx // w) for idx, value in enumerate(binary) if value]
            anisotropy = 1.0
            if len(points) >= 3:
                mean_x = sum(point[0] for point in points) / len(points)
                mean_y = sum(point[1] for point in points) / len(points)
                cov_xx = sum((point[0] - mean_x) ** 2 for point in points) / len(points)
                cov_yy = sum((point[1] - mean_y) ** 2 for point in points) / len(points)
                cov_xy = sum((point[0] - mean_x) * (point[1] - mean_y) for point in points) / len(points)
                trace = cov_xx + cov_yy
                determinant = cov_xx * cov_yy - cov_xy * cov_xy
                spread = max(0.0, trace * trace / 4.0 - determinant) ** 0.5
                major = trace / 2.0 + spread
                minor = max(1e-6, trace / 2.0 - spread)
                anisotropy = major / minor

            if has_enclosed_hole():
                candidates.append(HandwritingCandidate("。", 0.985, source="symbol_heuristic", reason="period_hollow_ring"))
                candidates.append(HandwritingCandidate("・", 0.70, source="symbol_heuristic", reason="middle_dot_like"))
            elif anisotropy >= 1.55:
                candidates.append(HandwritingCandidate("、", 0.975, source="symbol_heuristic", reason="comma_solid_diagonal"))
                candidates.append(HandwritingCandidate("・", 0.70, source="symbol_heuristic", reason="middle_dot_like"))
            elif 0.38 <= cx <= 0.62:
                # A very small filled round mark is ambiguous between ``。``
                # and ``・``. In vertical prose a sentence period is much more
                # frequent; keep middle-dot as the backup and let the editable
                # pixel database learn book-specific exceptions.
                candidates.append(HandwritingCandidate("。", 0.945, source="symbol_heuristic", reason="period_compact_filled"))
                candidates.append(HandwritingCandidate("・", 0.90, source="symbol_heuristic", reason="middle_dot_compact"))
                candidates.append(HandwritingCandidate("、", 0.78, source="symbol_heuristic", reason="comma_like"))
            else:
                candidates.append(HandwritingCandidate("、", 0.91, source="symbol_heuristic", reason="comma_compact"))
                candidates.append(HandwritingCandidate("・", 0.80, source="symbol_heuristic", reason="middle_dot_like"))

        # Long bars. In vertical Japanese layout the long-vowel mark / dash may
        # appear as a vertical bar, while horizontal material still uses a
        # horizontal bar.
        if bw >= 0.52 and bh <= 0.30 and horizontal_aspect >= 2.2:
            candidates.append(HandwritingCandidate("ー", 0.96, source="symbol_heuristic", reason="horizontal_long_dash"))
            candidates.append(HandwritingCandidate("―", 0.92, source="symbol_heuristic", reason="horizontal_long_bar"))
        if bh >= 0.56 and bw <= 0.24 and stats.components <= 2 and vertical_aspect >= 2.2:
            candidates.append(HandwritingCandidate("ー", 0.96, source="symbol_heuristic", reason="vertical_long_dash"))
            candidates.append(HandwritingCandidate("―", 0.92, source="symbol_heuristic", reason="vertical_long_bar"))

        # Ellipsis / repeated dots in either horizontal or vertical layout.
        if 2 <= stats.components <= 5 and (bh <= 0.30 or bw <= 0.30):
            candidates.append(HandwritingCandidate("…", 0.90, source="symbol_heuristic", reason="ellipsis_like"))

        # Bracket-like rough heuristics.
        if bh >= 0.56 and bw <= 0.34 and fill <= 0.46:
            if bx <= 0.36:
                candidates.append(HandwritingCandidate("「", 0.72, source="symbol_heuristic", reason="open_quote_like"))
                candidates.append(HandwritingCandidate("『", 0.66, source="symbol_heuristic", reason="open_quote_like"))
            if bx + bw >= 0.64:
                candidates.append(HandwritingCandidate("」", 0.72, source="symbol_heuristic", reason="close_quote_like"))
                candidates.append(HandwritingCandidate("』", 0.66, source="symbol_heuristic", reason="close_quote_like"))

        # Parenthesis-like rough heuristics.
        if bh >= 0.52 and bw <= 0.30 and 0.10 <= fill <= 0.40:
            if cx < 0.50:
                candidates.append(HandwritingCandidate("（", 0.62, source="symbol_heuristic", reason="paren_like"))
            else:
                candidates.append(HandwritingCandidate("）", 0.62, source="symbol_heuristic", reason="paren_like"))

        ranked: dict[str, HandwritingCandidate] = {}
        for cand in candidates:
            prev = ranked.get(cand.text)
            if prev is None or cand.score > prev.score:
                ranked[cand.text] = cand
        return sorted(ranked.values(), key=lambda c: (-c.score, c.text))[:8]

    def _trace_paths_from_crop(
        self,
        image: Image.Image,
        *,
        use_glyph_mask: bool = True,
    ) -> tuple[list[list[tuple[int, int]]], Image.Image | None]:
        # ``segment_black_ink_glyphs`` already yields a tight, masked single
        # glyph. The optional conservative mask is retained for standalone
        # callers, but never anchors to only the largest connected component.
        source = mask_single_glyph(image) if use_glyph_mask else image.convert("RGB")
        prepared = self._prepare_char_image(source, target_size=192)
        if prepared is None:
            return [], None
        binary, w, h = self._binary_from_image(prepared)
        thin = self._thin_zhang_suen(binary, w, h)
        paths = self._skeleton_paths(thin, w, h)
        simplified = [self._simplify(path, 0.55) for path in paths]
        simplified = [path for path in simplified if len(path) >= 2]
        simplified.sort(
            key=lambda path: (
                min(point[1] for point in path),
                min(point[0] for point in path),
            )
        )
        return simplified[:128], prepared

    def _render_trace_glyph(self, image: Image.Image, *, size: int = 96) -> Image.Image:
        paths, _prepared = self._trace_paths_from_crop(image, use_glyph_mask=True)
        canvas = Image.new("L", (size, size), 255)
        if not paths:
            return canvas
        draw = ImageDraw.Draw(canvas)
        all_points = [point for path in paths for point in path]
        min_x = min(point[0] for point in all_points)
        max_x = max(point[0] for point in all_points)
        min_y = min(point[1] for point in all_points)
        max_y = max(point[1] for point in all_points)
        span_x = max(1, max_x - min_x)
        span_y = max(1, max_y - min_y)
        padding = max(5, size // 12)
        scale = min((size - 2 * padding) / span_x, (size - 2 * padding) / span_y)
        offset_x = (size - span_x * scale) / 2.0
        offset_y = (size - span_y * scale) / 2.0
        width = max(1, int(round(size * 0.025)))
        for path in paths:
            points = [
                (
                    int(round(offset_x + (x - min_x) * scale)),
                    int(round(offset_y + (y - min_y) * scale)),
                )
                for x, y in path
            ]
            if len(points) >= 2:
                draw.line(points, fill=0, width=width, joint="curve")
        return canvas

    def _render_trace_line(self, glyph_images: Sequence[Image.Image]) -> Image.Image:
        glyph_size = 96
        gap = 8
        count = max(1, len(glyph_images))
        width = min(2000, count * glyph_size + max(0, count - 1) * gap)
        line = Image.new("L", (width, glyph_size), 255)
        x = 0
        for glyph in glyph_images:
            if x + glyph_size > width:
                break
            line.paste(self._render_trace_glyph(glyph, size=glyph_size), (x, 0))
            x += glyph_size + gap
        return line

    @staticmethod
    def _naturalize_path_direction(
        path: Sequence[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Orient a skeleton branch in a handwriting-like direction.

        A printed raster contains no real pen order. This rule does not claim to
        restore historical stroke order; it merely avoids feeding obviously
        reversed long strokes to an online recognizer: horizontal branches run
        left-to-right and predominantly vertical/diagonal branches top-to-bottom.
        """
        points = list(path)
        if len(points) < 2:
            return points
        start, end = points[0], points[-1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        reverse = (abs(dx) >= abs(dy) and dx < 0) or (abs(dy) > abs(dx) and dy < 0)
        return list(reversed(points)) if reverse else points

    @staticmethod
    def _polyline_length(points: Sequence[tuple[float, float]]) -> float:
        return sum(
            math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
            for a, b in zip(points, points[1:])
        )

    @staticmethod
    def _merge_touching_paths(
        paths: Sequence[Sequence[tuple[float, float]]],
        *,
        max_gap: float = 3.2,
    ) -> list[list[tuple[float, float]]]:
        """Join skeleton branches that meet at the same junction.

        The raw skeleton extractor splits a glyph at every endpoint/junction,
        which can turn one printed character into dozens of tiny PKStrokes.
        PencilKit expects pen-like trajectories, so greedily join touching
        branches while preserving pen lifts across real gaps.
        """
        pool = [list(path) for path in paths if len(path) >= 2]
        merged: list[list[tuple[float, float]]] = []
        while pool:
            current = pool.pop(0)
            changed = True
            while changed and pool:
                changed = False
                best: tuple[float, int, str] | None = None
                endpoints = (current[0], current[-1])
                for index, candidate in enumerate(pool):
                    options = (
                        (math.dist(endpoints[1], candidate[0]), "append"),
                        (math.dist(endpoints[1], candidate[-1]), "append_reverse"),
                        (math.dist(endpoints[0], candidate[-1]), "prepend"),
                        (math.dist(endpoints[0], candidate[0]), "prepend_reverse"),
                    )
                    distance, mode = min(options, key=lambda item: item[0])
                    if distance <= max_gap and (best is None or distance < best[0]):
                        best = (distance, index, mode)
                if best is None:
                    continue
                _distance, index, mode = best
                candidate = pool.pop(index)
                if mode == "append":
                    current.extend(candidate[1:])
                elif mode == "append_reverse":
                    current.extend(list(reversed(candidate[:-1])))
                elif mode == "prepend":
                    current = candidate[:-1] + current
                else:
                    current = list(reversed(candidate[1:])) + current
                changed = True
            merged.append(current)
        return merged

    @staticmethod
    def _endpoint_direction(
        path: Sequence[tuple[float, float]],
        *,
        at_start: bool,
    ) -> tuple[float, float]:
        if len(path) < 2:
            return (0.0, 0.0)
        sample = min(3, len(path) - 1)
        if at_start:
            a, b = path[0], path[sample]
        else:
            a, b = path[-sample - 1], path[-1]
        dx, dy = float(b[0] - a[0]), float(b[1] - a[1])
        length = math.hypot(dx, dy) or 1.0
        return (dx / length, dy / length)

    @classmethod
    def _join_collinear_paths(
        cls,
        paths: Sequence[Sequence[tuple[float, float]]],
        *,
        max_gap: float = 2.8,
        min_cosine: float = 0.72,
    ) -> list[list[tuple[float, float]]]:
        """Join only straight continuations through skeleton junctions.

        The former endpoint-only merge collapsed an entire connected kanji
        skeleton into one or two Euler-like scribbles.  A handwriting recognizer
        needs pen lifts at branches.  This routine joins two fragments only when
        their endpoint tangents continue in approximately the same direction;
        crossing and side branches stay separate strokes.
        """
        pool = [list(path) for path in paths if len(path) >= 2]
        result: list[list[tuple[float, float]]] = []
        while pool:
            current = pool.pop(0)
            while pool:
                best: tuple[float, float, int, str] | None = None
                cur_end_dir = cls._endpoint_direction(current, at_start=False)
                cur_start_dir = cls._endpoint_direction(current, at_start=True)
                for index, candidate in enumerate(pool):
                    cand_start_dir = cls._endpoint_direction(candidate, at_start=True)
                    cand_end_dir = cls._endpoint_direction(candidate, at_start=False)
                    options = [
                        (math.dist(current[-1], candidate[0]), cur_end_dir[0] * cand_start_dir[0] + cur_end_dir[1] * cand_start_dir[1], "append"),
                        (math.dist(current[-1], candidate[-1]), -(cur_end_dir[0] * cand_end_dir[0] + cur_end_dir[1] * cand_end_dir[1]), "append_reverse"),
                        (math.dist(current[0], candidate[-1]), cur_start_dir[0] * cand_end_dir[0] + cur_start_dir[1] * cand_end_dir[1], "prepend"),
                        (math.dist(current[0], candidate[0]), -(cur_start_dir[0] * cand_start_dir[0] + cur_start_dir[1] * cand_start_dir[1]), "prepend_reverse"),
                    ]
                    for distance, cosine, mode in options:
                        if distance <= max_gap and cosine >= min_cosine:
                            score = cosine - distance * 0.04
                            if best is None or score > best[0]:
                                best = (score, distance, index, mode)
                if best is None:
                    break
                _score, _distance, index, mode = best
                candidate = pool.pop(index)
                if mode == "append":
                    current.extend(candidate[1:])
                elif mode == "append_reverse":
                    current.extend(list(reversed(candidate[:-1])))
                elif mode == "prepend":
                    current = candidate[:-1] + current
                else:
                    current = list(reversed(candidate[1:])) + current
            result.append(current)
        return result

    @staticmethod
    def _chaikin_smooth(
        points: Sequence[tuple[float, float]],
        *,
        iterations: int = 1,
    ) -> list[tuple[float, float]]:
        out = [(float(x), float(y)) for x, y in points]
        for _ in range(max(0, iterations)):
            if len(out) < 3:
                break
            smoothed = [out[0]]
            for a, b in zip(out, out[1:]):
                smoothed.append((0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]))
                smoothed.append((0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]))
            smoothed.append(out[-1])
            out = smoothed
        return out

    @staticmethod
    def _resample_polyline(
        points: Sequence[tuple[float, float]],
        *,
        spacing: float = 2.4,
    ) -> list[tuple[float, float]]:
        """Return evenly spaced control points for a PencilKit stroke.

        The skeleton graph often yields just a few RDP points.  PencilKit works
        better when curves carry a stable stream of samples rather than only
        two distant endpoints.
        """
        source = [(float(x), float(y)) for x, y in points]
        if len(source) < 2:
            return source
        out = [source[0]]
        carry = 0.0
        for a, b in zip(source, source[1:]):
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            segment = math.hypot(dx, dy)
            if segment <= 1e-6:
                continue
            distance = spacing - carry
            while distance < segment:
                ratio = distance / segment
                out.append((a[0] + dx * ratio, a[1] + dy * ratio))
                distance += spacing
            carry = max(0.0, segment - (distance - spacing))
            if carry >= spacing:
                carry = 0.0
        if math.hypot(out[-1][0] - source[-1][0], out[-1][1] - source[-1][1]) > 0.4:
            out.append(source[-1])
        return out

    def _pkstroke_paths_from_glyphs(
        self,
        glyph_images: Sequence[Image.Image],
        *,
        page_width: float = 842.0,
        page_height: float = 595.0,
        left_margin: float = 64.0,
        top_margin: float = 150.0,
        cell_width: float = 116.0,
        cell_height: float = 148.0,
    ) -> tuple[list[list[tuple[float, float]]], list[int], float, float]:
        """Convert skeletons into handwriting-sized strokes on an A4-like page.

        PKStrokeRecognizer expects handwriting scaled like writing on standard
        paper.  The previous implementation created an extremely wide 96-point
        strip, which could produce an empty result even though the bridge ran.
        """
        strokes: list[list[tuple[float, float]]] = []
        glyph_indices: list[int] = []
        usable_width = max(cell_width, page_width - 2.0 * left_margin)
        max_cells = max(1, int(usable_width // cell_width))
        row_gap = 28.0

        for glyph_index, glyph in enumerate(glyph_images):
            paths, _prepared = self._trace_paths_from_crop(glyph, use_glyph_mask=False)
            if not paths:
                continue
            all_points = [point for path in paths for point in path]
            min_x = min(point[0] for point in all_points)
            max_x = max(point[0] for point in all_points)
            min_y = min(point[1] for point in all_points)
            max_y = max(point[1] for point in all_points)
            span_x = max(1.0, float(max_x - min_x))
            span_y = max(1.0, float(max_y - min_y))
            pad_x = 6.0
            pad_y = 7.0
            scale = min(
                (cell_width - 2.0 * pad_x) / span_x,
                (cell_height - 2.0 * pad_y) / span_y,
            )
            local_width = span_x * scale
            local_height = span_y * scale
            row = glyph_index // max_cells
            column = glyph_index % max_cells
            cell_origin_x = left_margin + column * cell_width
            cell_origin_y = top_margin + row * (cell_height + row_gap)
            origin_x = cell_origin_x + (cell_width - local_width) / 2.0
            origin_y = cell_origin_y + (cell_height - local_height) / 2.0

            # Convert all branches first, prune only tiny serif spurs, then
            # merge branches that share a skeleton junction into pen-like paths.
            converted_paths: list[list[tuple[float, float]]] = []
            glyph_diagonal = math.hypot(local_width, local_height)
            minimum_length = max(0.8, glyph_diagonal * 0.0045)
            for source_path in paths:
                converted_raw = [
                    (
                        origin_x + (float(x) - min_x) * scale,
                        origin_y + (float(y) - min_y) * scale,
                    )
                    for x, y in source_path
                ]
                if self._polyline_length(converted_raw) >= minimum_length:
                    converted_paths.append(converted_raw)
            if not converted_paths and paths:
                source_path = max(paths, key=lambda path: len(path))
                converted_paths.append([
                    (
                        origin_x + (float(x) - min_x) * scale,
                        origin_y + (float(y) - min_y) * scale,
                    )
                    for x, y in source_path
                ])

            # Preserve pen lifts at graph branches. Only join nearly collinear
            # continuations; the previous endpoint-only merge produced long
            # backtracking scribbles that looked unlike human Japanese writing.
            writing_paths = self._join_collinear_paths(
                converted_paths, max_gap=max(1.2, scale * 0.70), min_cosine=0.82
            )
            writing_paths = [self._naturalize_path_direction(path) for path in writing_paths]
            writing_paths.sort(key=lambda path: (min(p[1] for p in path), min(p[0] for p in path)))

            # Extra pen lifts are safer than joining unrelated branches. Prune
            # only after length ranking and keep a generous upper bound.
            if len(writing_paths) > 96:
                writing_paths = sorted(
                    writing_paths, key=self._polyline_length, reverse=True
                )[:96]
                writing_paths.sort(key=lambda path: (min(p[1] for p in path), min(p[0] for p in path)))

            for converted in writing_paths:
                if len(converted) >= 4:
                    converted = self._chaikin_smooth(converted, iterations=1)
                sampled = self._resample_polyline(converted, spacing=1.8)
                if len(sampled) >= 2:
                    strokes.append(sampled)
                    glyph_indices.append(glyph_index)

        return strokes, glyph_indices, page_width, page_height

    def _write_apple_trace_comparison(
        self,
        glyph: Image.Image,
        strokes: Sequence[Sequence[tuple[float, float]]],
        *,
        canvas_width: float,
        canvas_height: float,
    ) -> dict[str, str]:
        """Save the exact four-stage auto-trace diagnostic used by the panel.

        Order in the comparison image: original segmented glyph, conservative
        x-band mask, high-resolution binary input, final PKStroke polylines.
        """
        try:
            root = _PROJECT_ROOT / "debug" / "apple_pkstroke"
            root.mkdir(parents=True, exist_ok=True)
            source = glyph.convert("RGB")
            masked = mask_single_glyph(source)
            prepared = self._prepare_char_image(source, target_size=192)
            if prepared is None:
                return {}

            final_canvas = Image.new(
                "RGB",
                (max(1, int(round(canvas_width))), max(1, int(round(canvas_height)))),
                "white",
            )
            final_draw = ImageDraw.Draw(final_canvas)
            for stroke in strokes:
                if len(stroke) >= 2:
                    final_draw.line(list(stroke), fill="black", width=4, joint="curve")
            final_gray = ImageOps.grayscale(final_canvas)
            bbox = ImageOps.invert(final_gray).getbbox()
            if bbox is not None:
                bx0, by0, bx1, by1 = bbox
                pad = 16
                final_canvas = final_canvas.crop((
                    max(0, bx0 - pad), max(0, by0 - pad),
                    min(final_canvas.width, bx1 + pad), min(final_canvas.height, by1 + pad),
                ))

            tile_size = 220
            def tile(image: Image.Image) -> Image.Image:
                rgb = image.convert("RGB")
                fitted = ImageOps.contain(rgb, (tile_size - 20, tile_size - 20), Image.Resampling.LANCZOS)
                out = Image.new("RGB", (tile_size, tile_size), "white")
                out.paste(fitted, ((tile_size - fitted.width) // 2, (tile_size - fitted.height) // 2))
                return out

            source_path = root / "latest-auto-glyph-source.png"
            masked_path = root / "latest-auto-glyph-masked.png"
            binary_path = root / "latest-auto-glyph-binary.png"
            final_path = root / "latest-auto-glyph-pkdrawing.png"
            comparison_path = root / "latest-auto-glyph-comparison.png"
            source.save(source_path)
            masked.save(masked_path)
            prepared.save(binary_path)
            final_canvas.save(final_path)
            comparison = Image.new("RGB", (tile_size * 4, tile_size), "white")
            for index, image in enumerate((source, masked, prepared, final_canvas)):
                comparison.paste(tile(image), (index * tile_size, 0))
            comparison.save(comparison_path)
            return {
                "source": str(source_path),
                "masked": str(masked_path),
                "binary": str(binary_path),
                "pkdrawing": str(final_path),
                "comparison": str(comparison_path),
            }
        except Exception:
            return {}

    @staticmethod
    def _first_recognized_glyph(text: str) -> str:
        """Return one logical Unicode glyph from an Apple single-char result."""
        normalized = unicodedata.normalize("NFC", clean_text(text))
        normalized = "".join(ch for ch in normalized if not ch.isspace())
        if not normalized:
            return ""
        out = normalized[0]
        # Preserve variation selectors / combining marks belonging to the first
        # base character, but never accept a second base character.
        for ch in normalized[1:]:
            if unicodedata.combining(ch) or "\ufe00" <= ch <= "\ufe0f":
                out += ch
            else:
                break
        return out

    @staticmethod
    def _exact_session_glyph_key(image: Image.Image) -> str:
        """Hash the same normalized binary raster used by glyph memory.

        This is intentionally exact rather than perceptual: two different scans
        are never merged merely because they look similar. Therefore the cache
        changes runtime only, not the recognizer's acceptance threshold.
        """
        try:
            from adapters.glyph_memory_db import normalize_glyph
            normalized = normalize_glyph(image)
            try:
                payload = normalized.tobytes()
                return hashlib.sha256(
                    normalized.mode.encode("ascii", errors="ignore")
                    + b":" + str(normalized.size).encode("ascii") + b":" + payload
                ).hexdigest()
            finally:
                normalized.close()
        except Exception:
            return ""

    def _recognize_with_apple(self, glyph_images: Sequence[Image.Image]) -> str:
        """Recognize strictly one glyph per Apple PKDrawing/request.

        ``_last_apple_details`` stores the exact point sequences and result for
        every submitted glyph so the learning-package exporter can show what
        Apple actually received. Normal OCR additionally reuses only exact
        normalized raster matches within this process.
        """
        self._last_apple_details = []
        if self._apple is None:
            return ""
        self._last_apple_attempted = True
        self._last_apple_succeeded = False
        outputs: list[str] = []
        failures: list[str] = []
        latest_payload_saved = False
        trace_debug_enabled = bool(
            self._learning_session is not None
            or os.environ.get("NOVEL_FORMATTER_APPLE_TRACE_DEBUG", "").strip().lower() in {"1", "true", "yes"}
        )
        cache_allowed = not trace_debug_enabled

        for glyph_index, glyph in enumerate(glyph_images):
            cache_key = self._exact_session_glyph_key(glyph) if cache_allowed else ""
            cached = self._session_exact_glyph_cache.get(cache_key, "") if cache_key else ""
            if cached and cached not in _UNKNOWN_CHARS:
                outputs.append(cached)
                self._last_apple_succeeded = True
                self._last_apple_details.append({
                    "glyph_index": glyph_index, "strokes": [],
                    "canvas_width": 0.0, "canvas_height": 0.0,
                    "result": cached, "error": "",
                    "result_source": "session_exact_glyph_cache",
                    "session_cache_hit": True, "debug_images": {},
                })
                continue

            strokes, _stroke_glyph_indices, width, height = self._pkstroke_paths_from_glyphs(
                [glyph],
                page_width=560.0, page_height=640.0,
                left_margin=80.0, top_margin=120.0,
                cell_width=400.0, cell_height=400.0,
            )
            detail = {
                "glyph_index": glyph_index, "strokes": strokes,
                "canvas_width": width, "canvas_height": height,
                "result": "", "error": "", "session_cache_hit": False,
            }
            if not strokes:
                outputs.append("□")
                detail["error"] = f"第 {glyph_index + 1} 字没有可提交的中心线笔画"
                failures.append(detail["error"])
                self._last_apple_details.append(detail)
                continue

            # Comparison PNGs and latest payload files are diagnostics, not OCR
            # inputs. Keep full per-glyph output only for explicit learning/debug
            # sessions; normal OCR writes at most one inspectable payload/column.
            debug_images = (
                self._write_apple_trace_comparison(
                    glyph, strokes, canvas_width=width, canvas_height=height
                )
                if trace_debug_enabled else {}
            )
            detail["debug_images"] = debug_images
            save_latest_payload = bool(trace_debug_enabled or not latest_payload_saved)
            kwargs_variants = [
                {
                    "canvas_width": width, "canvas_height": height,
                    "preferred_languages": ("ja-JP",),
                    "glyph_indices": [0] * len(strokes), "glyph_count": 1,
                    "single_glyph_only": True, "debug_images": debug_images,
                    "save_latest_payload": save_latest_payload,
                },
                {
                    "canvas_width": width, "canvas_height": height,
                    "preferred_languages": ("ja-JP",),
                    "glyph_indices": [0] * len(strokes), "glyph_count": 1,
                    "single_glyph_only": True, "debug_images": debug_images,
                },
                {
                    "canvas_width": width, "canvas_height": height,
                    "preferred_languages": ("ja-JP",),
                    "glyph_indices": [0] * len(strokes), "glyph_count": 1,
                    "single_glyph_only": True,
                },
                {
                    "canvas_width": width, "canvas_height": height,
                    "preferred_languages": ("ja-JP",),
                },
            ]
            try:
                text = ""
                last_signature_error: TypeError | None = None
                for call_index, kwargs in enumerate(kwargs_variants):
                    try:
                        text = self._apple.recognize_strokes(strokes, **kwargs)
                        if kwargs.get("save_latest_payload"):
                            latest_payload_saved = True
                        last_signature_error = None
                        break
                    except TypeError as signature_error:
                        message = str(signature_error)
                        optional_names = (
                            "save_latest_payload", "debug_images", "glyph_indices",
                            "glyph_count", "single_glyph_only",
                        )
                        if not any(name in message for name in optional_names):
                            raise
                        last_signature_error = signature_error
                        if call_index == len(kwargs_variants) - 1:
                            raise
                if last_signature_error is not None:
                    raise last_signature_error
            except Exception as exc:
                outputs.append("□")
                detail["error"] = clean_text(exc)
                failures.append(f"第 {glyph_index + 1} 字：{detail['error']}")
                self._last_apple_details.append(detail)
                continue

            recognized = self._first_recognized_glyph(text)
            detail["result"] = recognized
            detail["result_source"] = "apple_pkstroke"
            if not recognized:
                outputs.append("□")
                detail["error"] = "PKStrokeRecognizer returned no text"
                failures.append(f"第 {glyph_index + 1} 字：{detail['error']}")
                self._last_apple_details.append(detail)
                continue
            outputs.append(recognized)
            self._last_apple_succeeded = True
            if cache_key and recognized not in _UNKNOWN_CHARS:
                self._session_exact_glyph_cache[cache_key] = recognized
            self._last_apple_details.append(detail)

        self._apple_error = "；".join(failures[-3:]) if failures else ""
        return "".join(outputs)

    def auto_trace_from_crop(self, image: Image.Image, *, use_glyph_mask: bool = True):
        paths, _prepared = self._trace_paths_from_crop(image, use_glyph_mask=use_glyph_mask)
        if not paths:
            return []
        pattern = "".join(self._path_code(path) for path in paths[:32])
        if not pattern:
            return []
        return self.recognize_pattern(
            pattern,
            direction_shift=self._direction_shift_count(paths),
            top_n=12,
        )

    def recognize_image_candidates(
        self,
        image: Image.Image,
        *,
        expected_char: str = "",
        use_glyph_mask: bool = True,
    ) -> list[HandwritingCandidate]:
        cleaned = mask_single_glyph(image) if use_glyph_mask else image.convert("RGB")
        merged: dict[str, HandwritingCandidate] = {}
        for source_candidates in (
            self.recognize_symbol_candidates(cleaned, use_glyph_mask=False),
            self.auto_trace_from_crop(cleaned, use_glyph_mask=False),
        ):
            for cand in source_candidates:
                prev = merged.get(cand.text)
                if prev is None or cand.score > prev.score:
                    merged[cand.text] = cand
        ranked = sorted(merged.values(), key=lambda c: (-c.score, c.source, c.text))
        if expected_char and ranked:
            # If OCR already matches one of the strong candidates, move it slightly up.
            for idx, cand in enumerate(ranked):
                if cand.text == expected_char and cand.score >= 0.82:
                    ranked.insert(0, ranked.pop(idx))
                    break
        return ranked[:12]

    @staticmethod
    def split_vertical_column_into_characters(
        image: Image.Image,
        *,
        expected_count: int | None = None,
        mask_main_band: bool = True,
        use_character_sweep: bool = False,
    ) -> list[tuple[int, int, Image.Image]]:
        # The physical top-to-bottom sweep is deliberately review-only.  Normal
        # OCR and background candidate screening keep the proven projection
        # path, so opening/closing the per-character review window cannot alter
        # OCR segmentation, OCR text, or another workspace's state.
        if use_character_sweep:
            try:
                working, physical_segments, _slider_info = segment_black_ink_glyphs_slider(
                    image,
                    apply_main_band_mask=bool(mask_main_band),
                    expected_count=expected_count,
                )
                try:
                    if physical_segments:
                        output: list[tuple[int, int, Image.Image]] = []
                        for segment in physical_segments:
                            y0 = int(segment.y0)
                            y1 = int(segment.y1)
                            # Keep the complete column width for recognition.  A
                            # punctuation recognizer needs to know whether a tiny
                            # dot sits at the lower-left/right of the print cell;
                            # a tight x crop destroys that positional evidence.
                            crop = working.crop((0, y0, working.width, y1)).convert("RGB")
                            output.append((y0, y1, crop))
                            segment.image.close()
                        return output
                finally:
                    working.close()
            except Exception:
                # Keep the historical pure-projection path as a compatibility
                # fallback for unusual blank/transparent rasters.
                pass

        if mask_main_band:
            masked_column, _ = mask_main_text_band(image)
            image = masked_column
        else:
            image = image.convert("RGB")
        gray = JapaneseHandwritingCard._to_grayscale(image)
        binary, w, h = JapaneseHandwritingCard._binary_from_image(gray)
        row_sum = [0] * h
        for y in range(h):
            base = y * w
            row_sum[y] = sum(binary[base : base + w])
        threshold = max(1, int(round(w * 0.012)))
        groups: list[list[int]] = []
        start: int | None = None
        for y, value in enumerate(row_sum):
            if value >= threshold:
                if start is None:
                    start = y
            elif start is not None:
                groups.append([start, y])
                start = None
        if start is not None:
            groups.append([start, h])
        if not groups:
            groups = [[0, h]]

        # Merge tiny fragments into neighbours.
        merged: list[list[int]] = []
        for grp in groups:
            if merged and (grp[1] - grp[0] <= 4 or grp[0] - merged[-1][1] <= 2):
                merged[-1][1] = grp[1]
            else:
                merged.append(grp[:])
        groups = merged

        if expected_count and expected_count > 0:
            while len(groups) > expected_count:
                best_idx = 0
                best_cost = None
                for i in range(len(groups) - 1):
                    gap = max(0, groups[i + 1][0] - groups[i][1])
                    height = (groups[i][1] - groups[i][0]) + (groups[i + 1][1] - groups[i + 1][0])
                    cost = gap * 4 + height
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best_idx = i
                groups[best_idx][1] = groups[best_idx + 1][1]
                del groups[best_idx + 1]
            while len(groups) < expected_count:
                heights = [grp[1] - grp[0] for grp in groups]
                idx = max(range(len(groups)), key=lambda i: heights[i])
                y0, y1 = groups[idx]
                if y1 - y0 < 10:
                    break
                mid = (y0 + y1) // 2
                groups[idx: idx + 1] = [[y0, mid], [mid, y1]]

        result: list[tuple[int, int, Image.Image]] = []
        for y0, y1 in groups:
            pad = max(2, int(round((y1 - y0) * 0.08)))
            yy0 = max(0, y0 - pad)
            yy1 = min(h, y1 + pad)
            crop = image.crop((0, yy0, image.size[0], yy1))
            result.append((yy0, yy1, crop))
        return result

    def _should_replace(self, ocr_char: str, top: HandwritingCandidate, second: HandwritingCandidate | None, *, strategy: str) -> tuple[bool, str]:
        gap = top.score - (second.score if second else 0.0)
        if ocr_char == top.text:
            return False, "same_as_ocr"

        if ocr_char in _UNKNOWN_CHARS:
            if top.score >= 0.50:
                return True, "first_candidate_fill_unknown"
            return False, "keep_unknown"

        # For symbol recognition, automatic mode now allows direct first-candidate
        # output under looser thresholds because the user requested IME-like
        # "take the first result" behaviour.
        if _is_symbol_char(ocr_char) or top.source == "symbol_heuristic":
            if strategy == "conservative":
                return (top.score >= 0.86, "first_symbol_candidate")
            if strategy == "aggressive":
                return (top.score >= 0.56, "first_symbol_candidate")
            return (top.score >= 0.72, "first_symbol_candidate")

        if not _should_attempt_recognition(ocr_char):
            return False, "preserve_blank"

        if strategy == "conservative":
            if top.score >= 0.982 and gap >= 0.20:
                return True, "first_candidate_conservative"
        elif strategy == "aggressive":
            if top.score >= 0.88 and gap >= 0.05:
                return True, "first_candidate_aggressive"
        else:  # balanced
            if top.score >= 0.94 and gap >= 0.10:
                return True, "first_candidate_balanced"
        return False, "keep_ocr"

    def _standalone_symbol_candidate(
        self,
        image: Image.Image,
        *,
        strategy: str,
        use_glyph_mask: bool = True,
    ) -> HandwritingCandidate | None:
        candidates = self.recognize_symbol_candidates(image, use_glyph_mask=use_glyph_mask)
        if not candidates:
            return None
        top = candidates[0]
        threshold = 0.94 if strategy == "conservative" else (0.72 if strategy == "aggressive" else 0.86)
        return top if top.score >= threshold else None

    def _candidate_can_fill_empty_ocr(self, candidate: HandwritingCandidate, *, strategy: str) -> bool:
        if candidate.source == "symbol_heuristic":
            threshold = 0.90 if strategy == "conservative" else (0.62 if strategy == "aggressive" else 0.78)
        else:
            threshold = 0.985 if strategy == "conservative" else (0.90 if strategy == "aggressive" else 0.95)
        return candidate.score >= threshold

    @staticmethod
    def _candidate_base_char(text: str) -> str:
        return JapaneseHandwritingCard._first_recognized_glyph(clean_text(text))

    def _memory_candidates_for_glyph(self, image: Image.Image) -> list[HandwritingCandidate]:
        if self._glyph_memory is None:
            return []
        try:
            matches = self._glyph_memory.lookup(
                image,
                scope_key=self.glyph_memory_scope_key,
                include_global=self.glyph_memory_include_global,
                limit=8,
            )
        except Exception:
            return []
        candidates: list[HandwritingCandidate] = []
        for match in matches:
            trusted_similar = bool(
                not match.exact
                and not bool(getattr(match, "topology_mismatch", False))
                and float(match.score or 0.0) >= 0.87
                and float(match.shape_similarity or 0.0) >= 0.82
                and float(match.pixel_difference or 1.0) <= 0.08
                and int(match.hash_distance or 64) <= 10
            )
            source = (
                "glyph_memory_exact" if match.exact
                else ("glyph_memory_trusted_similar" if trusted_similar else "glyph_memory_similar")
            )
            candidates.append(HandwritingCandidate(
                text=match.character,
                score=float(match.score),
                source=source,
                reason=(
                    f"scope={match.scope_key};pixel_diff={match.pixel_difference:.4f};"
                    f"hash_distance={match.hash_distance};shape_similarity={match.shape_similarity:.4f};"
                    f"topology_mismatch={int(bool(getattr(match, 'topology_mismatch', False)))};"
                    f"confirmations={match.confirmations}"
                ),
            ))
        return candidates

    def _vision_candidates_from_groups(self, groups) -> list[HandwritingCandidate]:
        output: list[HandwritingCandidate] = []
        for group_name, source_name in (("corrected", "apple_vision_corrected"), ("raw", "apple_vision_raw")):
            for item in groups.get(group_name, []):
                char = self._candidate_base_char(item.text)
                if not char:
                    continue
                rank_decay = max(0.35, 1.0 - 0.075 * max(0, item.rank - 1))
                # Vision confidence is not calibrated as a full probability and
                # may be modest even for a useful second candidate.  Preserve a
                # rank prior so Top-N alternatives can still be promoted by
                # PKStrokeRecognizer or glyph-memory agreement.
                score = 0.65 * max(0.0, float(item.confidence)) + 0.35 * rank_decay
                output.append(HandwritingCandidate(
                    text=char,
                    score=score,
                    source=source_name,
                    reason=f"rank={item.rank};confidence={item.confidence:.4f};full={item.text}",
                ))
        return output

    def _vision_candidates_for_glyph(self, image: Image.Image) -> list[HandwritingCandidate]:
        if self._vision_candidates is None:
            return []
        try:
            groups = self._vision_candidates.recognize(image, top_n=10)
        except Exception as exc:
            self._vision_candidate_error = str(exc)
            return []
        return self._vision_candidates_from_groups(groups)

    def _fuse_glyph_candidates(
        self,
        image: Image.Image,
        *,
        apple_char: str,
        strategy: str,
        use_character_mask: bool,
        vision_groups=None,
        external_candidates: Sequence[HandwritingCandidate] | None = None,
        layout_anchor_char: str = "",
        layout_anchor_confidence: float = 0.0,
        segment_height_ratio: float = 1.0,
        allow_symbol_override: bool = True,
    ) -> AutoCharDecision:
        evidence: dict[str, dict] = {}

        def add(candidate: HandwritingCandidate, weight: float, *, exact_memory: bool = False):
            char = self._candidate_base_char(candidate.text)
            if not char:
                return
            entry = evidence.setdefault(char, {"total": 0.0, "sources": set(), "reasons": [], "exact": False})
            entry["total"] += max(0.0, float(candidate.score)) * weight
            entry["sources"].add(candidate.source)
            if candidate.reason:
                entry["reasons"].append(candidate.reason)
            entry["exact"] = bool(entry["exact"] or exact_memory)

        memory_candidates = self._memory_candidates_for_glyph(image)
        for candidate in memory_candidates:
            if candidate.source == "glyph_memory_exact":
                add(candidate, 2.35, exact_memory=True)
            else:
                add(candidate, 1.30)

        vision_candidates = (
            self._vision_candidates_from_groups(vision_groups)
            if vision_groups is not None
            else self._vision_candidates_for_glyph(image)
        )
        for candidate in vision_candidates:
            add(candidate, 1.12 if candidate.source.endswith("corrected") else 1.00)

        for candidate in list(external_candidates or []):
            # Rejected/low-confidence column OCR is useful only as agreement
            # evidence. It receives a deliberately weak weight and cannot become
            # output by itself. Confirmed multi-view rescue keeps the old weight.
            weight = 0.78 if candidate.source == "low_confidence_column_ocr" else 1.18
            add(candidate, weight)

        apple_char = self._candidate_base_char(apple_char)
        # Geometry can identify punctuation when Apple returns nothing or a
        # symbol. Standalone candidate fusion may also override an implausible
        # normal glyph in a very compact cell. The full Apple one-box/one-result
        # pass disables that override so its slot alignment remains authoritative.
        compact_cell = 0.0 < float(segment_height_ratio or 0.0) <= 0.58
        should_check_symbol = bool(
            not apple_char
            or _is_symbol_char(apple_char)
            or (compact_cell and allow_symbol_override)
        )
        symbol_candidates = (
            self.recognize_symbol_candidates(image, use_glyph_mask=False)
            if should_check_symbol else []
        )
        for candidate in symbol_candidates[:4]:
            if float(candidate.score or 0.0) < 0.72:
                continue
            add(candidate, 1.42 if compact_cell else 0.88)

        if apple_char and apple_char != "□":
            add(HandwritingCandidate(
                text=apple_char, score=1.0, source="apple_pkstroke_recognizer",
                reason="single_glyph_recognizedText_first_result",
            ), 1.28)

        layout_anchor_char = self._candidate_base_char(layout_anchor_char)
        if layout_anchor_char and layout_anchor_char != "□":
            add(HandwritingCandidate(
                text=layout_anchor_char,
                score=max(0.35, min(1.0, float(layout_anchor_confidence or 0.0))),
                source="apple_vision_column_anchor",
                reason="RecognizeTextRequest.fast boundingBox(for: Character)",
            ), 1.05)

        ranked: list[tuple[str, float, dict]] = []
        for char, entry in evidence.items():
            diversity = max(0, len(entry["sources"]) - 1)
            total = float(entry["total"]) + min(0.54, diversity * 0.18)
            if "apple_vision_corrected" in entry["sources"] and "apple_vision_raw" in entry["sources"]:
                total += 0.10
            if "apple_pkstroke_recognizer" in entry["sources"] and any(
                source.startswith("apple_vision") for source in entry["sources"]
            ):
                total += 0.22
            ranked.append((char, total, entry))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        # A pixel-exact, user-confirmed glyph memory is an explicit rule for
        # this stored shape and must outrank model guesses, even when Vision and
        # PKStrokeRecognizer agree on another visually similar character.
        exact_ranked = [item for item in ranked if bool(item[2].get("exact"))]
        if exact_ranked:
            exact_ranked.sort(key=lambda item: (-item[1], item[0]))
            chosen_exact = exact_ranked[0]
            ranked = [chosen_exact] + [item for item in ranked if item is not chosen_exact]

        if not ranked:
            return AutoCharDecision(
                index=0, ocr_char="", chosen_char="□", changed=True,
                score=0.0, reason="apple_only_unresolved", candidates=[],
            )

        candidates = [
            HandwritingCandidate(
                text=char,
                score=min(1.0, total / 2.40),
                source="+".join(sorted(entry["sources"])),
                reason=" | ".join(entry["reasons"][:4]),
            )
            for char, total, entry in ranked[:10]
        ]
        top_char, top_total, top_entry = ranked[0]
        source_set = set(top_entry["sources"])
        exact_memory = bool(top_entry["exact"])
        multi_source = len(source_set) >= 2
        has_vision = any(source.startswith("apple_vision") for source in source_set)
        has_apple_stroke = "apple_pkstroke_recognizer" in source_set
        has_memory = any(source.startswith("glyph_memory") for source in source_set)
        has_glyph_rescue = any(source.startswith("glyph_rescue") for source in source_set)
        has_symbol_geometry = "symbol_heuristic" in source_set

        threshold = 0.98 if strategy == "conservative" else (0.53 if strategy == "aggressive" else 0.70)
        accept = exact_memory or (
            top_total >= threshold and (
                has_vision or has_apple_stroke or has_memory or has_glyph_rescue
                or has_symbol_geometry
            )
        )
        # A weak near-memory candidate is useful for the review list, but must
        # never become printed OCR output by itself.  The old 1.30 weight meant
        # even a ~0.55 look-alike crossed the balanced threshold and filled pages
        # with unrelated learned characters whenever PKStroke returned empty.
        # Trusted near-memory matches already short-circuit above at >=0.90 with
        # a uniqueness margin; everything weaker now remains □ unless Apple also
        # supports it.
        memory_only = has_memory and not has_vision and not has_apple_stroke and not has_glyph_rescue
        if memory_only and not exact_memory:
            accept = False
        if strategy == "conservative" and not exact_memory:
            accept = accept and (
                multi_source
                or (has_vision and top_total >= 1.15)
                or (compact_cell and has_symbol_geometry and top_total >= 1.20)
            )
        chosen = top_char if accept else "□"
        reason = "fusion_" + "+".join(sorted(source_set)) if accept else "fusion_low_confidence"
        return AutoCharDecision(
            index=0, ocr_char="", chosen_char=chosen, changed=True,
            score=min(1.0, top_total / 2.40), reason=reason,
            candidates=candidates,
        )

    def _apply_low_confidence_glyph_rescue(self, report, glyph_images: Sequence[Image.Image]) -> None:
        """Run targeted multi-view OCR only for uncertain, aligned glyph slots.

        This does not rescan the whole column.  The rescuer receives a bounded
        batch of single-glyph crops and may confirm the existing column OCR
        character or provide an additional candidate for PKStroke/manual review.
        """
        self._last_glyph_rescue_results = {}
        self._last_glyph_rescue_requested_indices = []
        self._last_glyph_rescue_error = ""
        rescuer = self._low_confidence_glyph_rescuer
        if rescuer is None or report is None or not glyph_images:
            return
        if bool(getattr(report, "preserve_original_text", False)):
            # No safe slot mapping exists, so per-box results must not be
            # attached to arbitrary sequence positions.
            return

        candidates = []
        for item in list(getattr(report, "decisions", []) or []):
            if bool(getattr(item, "accepted", False)):
                continue
            if (
                not bool(getattr(item, "automatic_fallback", True))
                and not bool(getattr(item, "provisional", False))
            ):
                # Truly manual-only slots have no safe automatic alignment. A
                # provisional ordinary-OCR character is different: its slot is
                # aligned and the three-view rescue can independently confirm
                # it before PKStroke without changing the original sequence.
                continue
            reason = str(getattr(item, "reason", "") or "")
            if reason in {"suspicious_glyph_box", "column_length_mismatch_preserve_primary_for_manual"}:
                continue
            index = int(getattr(item, "index", -1) or 0)
            if not (0 <= index < len(glyph_images)):
                continue
            score = float(getattr(item, "effective_character_confidence", 0.0) or 0.0)
            candidates.append((score, index, item))
        candidates.sort(key=lambda row: (row[0], row[1]))
        target_indices = [row[1] for row in candidates]
        originals = {
            row[1]: str(getattr(row[2], "ocr_char", "") or "")[:1]
            for row in candidates
        }
        if not target_indices:
            return
        try:
            results = rescuer.rescue(
                glyph_images,
                target_indices=target_indices,
                original_characters=originals,
            )
        except Exception as exc:
            self._last_glyph_rescue_error = str(exc)
            return
        self._last_glyph_rescue_results = dict(results or {})
        self._last_glyph_rescue_requested_indices = list(
            getattr(rescuer, "last_requested_indices", []) or []
        )
        self._last_glyph_rescue_error = str(getattr(rescuer, "last_error", "") or "")

        by_index = {int(getattr(item, "index", -1)): item for item in report.decisions}
        for index, rescue in self._last_glyph_rescue_results.items():
            item = by_index.get(int(index))
            if item is None:
                continue
            rescue_char = str(getattr(rescue, "candidate_character", "") or "")[:1]
            item.glyph_rescue_character = rescue_char
            item.glyph_rescue_score = float(getattr(rescue, "score", 0.0) or 0.0)
            item.glyph_rescue_support = int(getattr(rescue, "support", 0) or 0)
            item.glyph_rescue_total = int(getattr(rescue, "total_views", 0) or 0)
            item.glyph_rescue_reason = str(getattr(rescue, "reason", "") or "")
            item.glyph_rescue_view_characters = [
                str(getattr(view, "character", "") or "") or "∅"
                for view in list(getattr(rescue, "views", []) or [])
            ]
            confirmed = bool(
                getattr(rescue, "confirmed_original", False)
                or getattr(rescue, "confident_fill", False)
            )
            item.glyph_rescue_confirmed = confirmed
            if not confirmed or not rescue_char:
                continue
            item.accepted = True
            item.provisional = False
            item.automatic_fallback = False
            item.output_char = rescue_char
            item.score = max(float(getattr(item, "score", 0.0) or 0.0), item.glyph_rescue_score)
            item.reason = str(getattr(rescue, "reason", "") or "glyph_rescue_confirmed")

        report.accepted_indices = sorted(
            int(item.index) for item in report.decisions if bool(getattr(item, "accepted", False))
        )
        accepted = set(report.accepted_indices)
        report.rejected_indices = [index for index in range(len(glyph_images)) if index not in accepted]
        report.provisional_indices = [
            int(item.index) for item in report.decisions if bool(getattr(item, "provisional", False))
        ]
        report.automatic_fallback_indices = [
            int(item.index) for item in report.decisions
            if not bool(getattr(item, "accepted", False))
            and bool(getattr(item, "automatic_fallback", True))
        ]
        report.manual_only_indices = [
            int(item.index) for item in report.decisions
            if not bool(getattr(item, "accepted", False))
            and not bool(getattr(item, "automatic_fallback", True))
        ]

    def _recognize_fused_black_ink_glyphs(
        self,
        glyph_images: Sequence[Image.Image],
        *,
        strategy: str,
        use_character_mask: bool,
        preaccepted_ocr: dict[int, object] | None = None,
    ) -> AutoColumnResult:
        """Fuse only Apple-derived candidates plus explicit glyph memory.

        Every occupied print cell, including punctuation, remains one slot and
        is analysed by Apple.  Geometry/JLect no longer inserts extra symbols.
        """
        count = len(glyph_images)
        preaccepted_ocr = dict(preaccepted_ocr or {})
        self._last_fallback_note = ""
        self._last_apple_attempted = False
        self._last_apple_succeeded = False
        self._apple_error = ""
        self._vision_candidate_error = ""
        decisions: list[AutoCharDecision | None] = [None] * count
        full_apple_details: list[dict] = [{} for _ in range(count)]
        self._last_apple_details = full_apple_details
        self._last_vision_groups = [None for _ in range(count)]
        self._last_apple_requested_indices = []
        unresolved_indices: list[int] = []
        segment_heights = [
            max(1, int(getattr(segment, "y1", 0) or 0) - int(getattr(segment, "y0", 0) or 0))
            for segment in (self._last_glyph_segments or [])
        ]
        median_segment_height = float(sorted(segment_heights)[len(segment_heights) // 2]) if segment_heights else 0.0

        for index, glyph in enumerate(glyph_images):
            memory_candidates = self._memory_candidates_for_glyph(glyph)
            exact = next((item for item in memory_candidates if item.source == "glyph_memory_exact"), None)
            if exact is not None:
                decisions[index] = AutoCharDecision(
                    index=index, ocr_char="", chosen_char=self._candidate_base_char(exact.text),
                    changed=True, score=float(exact.score),
                    reason="glyph_memory_exact_short_circuit_no_apple",
                    candidates=memory_candidates[:10],
                )
                continue

            # The same printed character can acquire a few changed edge pixels
            # after a second scan/crop. A unique, very-high-score memory match
            # is treated as a trusted learned variant and also skips Apple.
            strong_similar = next(
                (item for item in memory_candidates
                 if item.source == "glyph_memory_trusted_similar"),
                None,
            )
            competing = [
                item for item in memory_candidates
                if strong_similar is not None and item.text != strong_similar.text
            ]
            if strong_similar is not None and (
                not competing or float(strong_similar.score) - float(competing[0].score or 0.0) >= 0.06
            ):
                decisions[index] = AutoCharDecision(
                    index=index, ocr_char="", chosen_char=self._candidate_base_char(strong_similar.text),
                    changed=True, score=float(strong_similar.score),
                    reason="glyph_memory_strong_similar_short_circuit_no_apple",
                    candidates=memory_candidates[:10],
                )
                continue

            ocr_evidence = preaccepted_ocr.get(index)
            if ocr_evidence is not None:
                ocr_char = str(getattr(ocr_evidence, "ocr_char", "") or "")[:1]
                output_char = str(getattr(ocr_evidence, "output_char", "") or "")[:1]
                chosen = output_char or ocr_char or "□"
                ocr_score = float(getattr(ocr_evidence, "score", 0.0) or 0.0)
                engine = str(getattr(self._last_selective_report, "engine", "unknown") or "unknown")
                is_provisional = bool(getattr(ocr_evidence, "provisional", False))
                is_accepted = bool(getattr(ocr_evidence, "accepted", False))
                rescued = bool(getattr(ocr_evidence, "glyph_rescue_confirmed", False))
                if is_accepted and rescued:
                    decision_reason = "low_confidence_glyph_multiview_confirmed_short_circuit"
                    source = f"glyph_rescue_confirmed:{engine}"
                elif is_accepted:
                    decision_reason = "selective_column_ocr_high_confidence_short_circuit"
                    source = f"selective_column_ocr:{engine}"
                elif is_provisional:
                    decision_reason = "selective_column_ocr_provisional_preserved_no_pkstroke"
                    source = f"provisional_column_ocr:{engine}"
                else:
                    decision_reason = "selective_column_ocr_manual_only_no_pkstroke"
                    source = f"manual_review_required:{engine}"
                candidates = list(memory_candidates[:9])
                rescue_char = str(getattr(ocr_evidence, "glyph_rescue_character", "") or "")[:1]
                rescue_score = float(getattr(ocr_evidence, "glyph_rescue_score", 0.0) or 0.0)
                if rescue_char and rescue_char != "□" and rescue_char != ocr_char:
                    candidates.insert(0, HandwritingCandidate(
                        text=rescue_char, score=rescue_score, source="glyph_rescue_multiview",
                        reason=str(getattr(ocr_evidence, "glyph_rescue_reason", "") or "targeted glyph OCR candidate"),
                    ))
                if ocr_char and ocr_char != "□":
                    candidates.insert(0, HandwritingCandidate(
                        text=ocr_char, score=ocr_score, source=source,
                        reason=str(getattr(ocr_evidence, "reason", "") or decision_reason),
                    ))
                decisions[index] = AutoCharDecision(
                    index=index, ocr_char=ocr_char, chosen_char=chosen,
                    changed=True, score=ocr_score, reason=decision_reason,
                    candidates=candidates,
                )
                continue

            unresolved_indices.append(index)

        selective_decisions_by_index = {
            int(getattr(item, "index", -1)): item
            for item in list(getattr(self._last_selective_report, "decisions", []) or [])
            if int(getattr(item, "index", -1)) >= 0
        }
        if unresolved_indices:
            self._last_apple_requested_indices = list(unresolved_indices)
            unresolved_images = [glyph_images[index] for index in unresolved_indices]
            apple_text = ""
            subset_details: list[dict] = []
            if self._apple is not None:
                apple_text = str(self._recognize_with_apple(unresolved_images) or "")
                subset_details = list(self._last_apple_details)
            apple_slots = list(apple_text)
            for local_index, global_index in enumerate(unresolved_indices):
                if local_index < len(subset_details):
                    detail = dict(subset_details[local_index] or {})
                    detail["glyph_index"] = global_index
                    full_apple_details[global_index] = detail
            self._last_apple_details = full_apple_details

            vision_batches = [None] * len(unresolved_images)
            if self._vision_candidates is not None:
                try:
                    recognize_many = getattr(self._vision_candidates, "recognize_many", None)
                    if callable(recognize_many):
                        vision_batches = list(recognize_many(list(unresolved_images), top_n=10))
                except Exception as exc:
                    self._vision_candidate_error = clean_text(exc)
                    vision_batches = [None] * len(unresolved_images)

            for local_index, global_index in enumerate(unresolved_indices):
                glyph = glyph_images[global_index]
                apple_char = apple_slots[local_index] if local_index < len(apple_slots) else ""
                vision_group = vision_batches[local_index] if local_index < len(vision_batches) else None
                self._last_vision_groups[global_index] = vision_group
                segment = self._last_glyph_segments[global_index] if global_index < len(self._last_glyph_segments) else None
                rescue = self._last_glyph_rescue_results.get(global_index)
                external_evidence: list[HandwritingCandidate] = []
                rescue_char = str(getattr(rescue, "candidate_character", "") or "")[:1] if rescue is not None else ""
                if rescue_char and rescue_char != "□":
                    external_evidence.append(HandwritingCandidate(
                        text=rescue_char,
                        score=float(getattr(rescue, "score", 0.0) or 0.0),
                        source="glyph_rescue_multiview",
                        reason=str(getattr(rescue, "reason", "") or "targeted multi-view glyph OCR"),
                    ))
                # Keep a rejected ordinary-OCR character only as weak consensus
                # evidence. Never attach it when box/sequence alignment itself is
                # suspicious, and never let it produce output without Apple,
                # Vision, memory, rescue or symbol geometry support.
                rejected_ocr = selective_decisions_by_index.get(global_index)
                if rejected_ocr is not None:
                    rejected_reason = str(getattr(rejected_ocr, "reason", "") or "")
                    rejected_char = str(getattr(rejected_ocr, "ocr_char", "") or "")[:1]
                    if (
                        rejected_char and rejected_char not in _UNKNOWN_CHARS
                        and rejected_reason not in {
                            "suspicious_glyph_box",
                            "column_length_mismatch_preserve_primary_for_manual",
                        }
                    ):
                        rejected_score = float(
                            getattr(rejected_ocr, "effective_character_confidence", 0.0)
                            or getattr(rejected_ocr, "score", 0.0) or 0.0
                        )
                        external_evidence.append(HandwritingCandidate(
                            text=rejected_char, score=max(0.0, min(0.94, rejected_score)),
                            source="low_confidence_column_ocr",
                            reason=rejected_reason or "ordinary OCR below automatic threshold",
                        ))
                segment_height = max(
                    1.0, float(getattr(segment, "y1", 0) or 0) - float(getattr(segment, "y0", 0) or 0)
                ) if segment is not None else 1.0
                height_ratio = segment_height / median_segment_height if median_segment_height > 0 else 1.0
                decision = self._fuse_glyph_candidates(
                    glyph, apple_char=apple_char, strategy=strategy,
                    use_character_mask=use_character_mask, vision_groups=vision_group,
                    external_candidates=external_evidence,
                    layout_anchor_char=str(getattr(segment, "anchor_text", "") or ""),
                    layout_anchor_confidence=float(getattr(segment, "anchor_confidence", 0.0) or 0.0),
                    segment_height_ratio=height_ratio,
                    allow_symbol_override=False,
                )
                decision.index = global_index
                decisions[global_index] = decision

        final_decisions: list[AutoCharDecision] = []
        for index, decision in enumerate(decisions):
            if decision is None:
                decision = AutoCharDecision(
                    index=index, ocr_char="", chosen_char="□", changed=True,
                    score=0.0, reason="fusion_unresolved", candidates=[],
                )
            final_decisions.append(decision)
        output = [item.chosen_char for item in final_decisions]
        text = "".join(output)
        used = set()
        for decision in final_decisions:
            if decision.candidates:
                used.update(decision.candidates[0].source.split("+"))
        if used:
            self._last_backend = "candidate_fusion"
        if unresolved_indices and self._last_apple_attempted and not self._last_apple_succeeded:
            notes = ["PKStrokeRecognizer未给出可用单字；仅保留Apple Vision/字形记忆候选或□"]
            if self._vision_candidate_error:
                notes.append(self._vision_candidate_error)
            self._last_fallback_note = "；".join(notes)
        elif not unresolved_indices:
            self._last_fallback_note = ""
        return AutoColumnResult(
            block_id="", original_text="", text=text, changed=bool(text), reviewed=True,
            auto_segments=count, auto_changed_chars=sum(ch != "□" for ch in output),
            decisions=final_decisions,
        )

    def _record_learning_items(self, glyph_segments, result: AutoColumnResult) -> None:
        if self._learning_session is None:
            return
        page = int(self._learning_context.get("page", 0) or 0)
        column = int(self._learning_context.get("column", 0) or 0)
        for index, segment in enumerate(glyph_segments):
            if index >= len(result.decisions):
                break
            decision = result.decisions[index]
            detail = self._last_apple_details[index] if index < len(self._last_apple_details) else {}
            is_symbol = bool(
                decision.reason.startswith("symbol_geometry_direct")
                or _is_symbol_char(decision.chosen_char)
                or any(candidate.source == "symbol_heuristic" for candidate in decision.candidates)
            )
            try:
                vision_groups = self._last_vision_groups[index] if index < len(self._last_vision_groups) else None
                self._learning_session.record(
                    image=segment.image, page=page, column=column, glyph_index=index,
                    segment=segment, decision=decision, apple_detail=detail,
                    vision_groups=vision_groups, is_symbol=is_symbol,
                )
            except Exception:
                continue

    def recognize_black_ink_column(
        self,
        column_image: Image.Image,
        *,
        strategy: str = "balanced",
        use_character_mask: bool = True,
        manual_placeholders: bool = False,
        expected_count: int | None = None,
        character_anchors: Sequence[object] | None = None,
        precomputed_boxes: Sequence[object] | None = None,
        selective_ocr_text: str = "",
        selective_ocr_confidence: float = 0.0,
        selective_ocr_engine: str = "",
        selective_ocr_threshold: float = 0.95,
        selective_ocr_reject_conflicts: bool = True,
        selective_ocr_character_confidences: Sequence[float] | None = None,
        selective_ocr_variants: Sequence[dict] | None = None,
        selective_ocr_require_stability: bool = False,
        preserve_sequence_ocr: bool = False,
    ) -> AutoColumnResult:
        """Recognize a vertical column without consulting any OCR string."""
        # Automatic printed OCR must never start Apple image OCR.  Character
        # anchors may still be supplied by imported historical metadata, but an
        # absent value now means “use local projection only”, not “call Vision”.
        if character_anchors is None:
            character_anchors = []
            self._column_anchor_error = "Apple图片OCR仅限人工复核；自动切框使用本地投影"
        geometry_sequence = _primary_ocr_geometry_sequence(
            selective_ocr_text,
            raw_confidence=selective_ocr_confidence,
            engine=selective_ocr_engine,
            threshold=selective_ocr_threshold,
        )
        prepared_boxes = precomputed_boxes
        geometry_repair_info: dict = {}
        if precomputed_boxes and geometry_sequence and not preserve_sequence_ocr:
            prepared_boxes, geometry_repair_info = repair_precomputed_glyph_boxes_with_ocr_text(
                column_image, precomputed_boxes, geometry_sequence,
            )
        masked_column, glyph_segments, segmentation_info = segment_black_ink_glyphs(
            column_image,
            apply_main_band_mask=use_character_mask,
            expected_count=expected_count,
            character_anchors=character_anchors,
            precomputed_boxes=prepared_boxes,
        )
        if geometry_repair_info:
            segmentation_info = dict(segmentation_info or {})
            segmentation_info.update(geometry_repair_info)
        # Preview geometry is normally reused verbatim for speed. If two
        # independent ordinary OCR engines agree on the exact sequence but its
        # length differs from the cached boxes by only one or two slots, perform
        # one fresh black-ink split with that length as a *soft pitch hint*. The
        # fresh geometry is selected only when it lands on the agreed count;
        # otherwise the original cached boxes remain untouched.
        stable_count_hint = _stable_sequence_count_hint(
            selective_ocr_text,
            raw_confidence=selective_ocr_confidence,
            engine=selective_ocr_engine,
            threshold=selective_ocr_threshold,
            variants=selective_ocr_variants,
            require_stability=bool(selective_ocr_require_stability),
        )
        cached_count = len(glyph_segments)
        repair_limit = max(2, int(round((stable_count_hint or 0) * 0.08)))
        should_retry_geometry = bool(
            precomputed_boxes
            and stable_count_hint
            and stable_count_hint >= 2
            and cached_count != stable_count_hint
            and abs(cached_count - stable_count_hint) <= repair_limit
        )
        if should_retry_geometry:
            fresh_masked, fresh_segments, fresh_info = segment_black_ink_glyphs(
                column_image,
                apply_main_band_mask=use_character_mask,
                expected_count=stable_count_hint,
                character_anchors=character_anchors,
                precomputed_boxes=None,
            )
            if len(fresh_segments) == stable_count_hint:
                masked_column.close()
                for old_segment in glyph_segments:
                    try:
                        old_segment.image.close()
                    except Exception:
                        pass
                masked_column = fresh_masked
                glyph_segments = fresh_segments
                segmentation_info = dict(fresh_info or {})
                segmentation_info.update({
                    "count_guided_resegmentation_attempted": True,
                    "count_guided_resegmentation_selected": True,
                    "stable_ocr_count_hint": int(stable_count_hint),
                    "cached_box_count": int(cached_count),
                })
            else:
                fresh_masked.close()
                for fresh_segment in fresh_segments:
                    try:
                        fresh_segment.image.close()
                    except Exception:
                        pass
                segmentation_info = dict(segmentation_info or {})
                segmentation_info.update({
                    "count_guided_resegmentation_attempted": True,
                    "count_guided_resegmentation_selected": False,
                    "stable_ocr_count_hint": int(stable_count_hint),
                    "cached_box_count": int(cached_count),
                    "fresh_box_count": int(len(fresh_segments)),
                })
        masked_column.close()
        for old_image in self._last_glyph_images:
            try:
                old_image.close()
            except Exception:
                pass
        for old_segment in self._last_glyph_segments:
            try:
                old_segment.image.close()
            except Exception:
                pass
        glyph_images = [segment.image for segment in glyph_segments]
        self._last_glyph_segments = list(glyph_segments)
        self._last_glyph_images = [image.copy() for image in glyph_images]
        self._last_segmentation_info = dict(segmentation_info or {})
        self._last_selective_report = None
        self._last_apple_requested_indices = []
        self._last_glyph_rescue_results = {}
        self._last_glyph_rescue_requested_indices = []
        self._last_glyph_rescue_error = ""
        if preserve_sequence_ocr and selective_ocr_text and not manual_placeholders:
            preserved_text = clean_text(selective_ocr_text).replace("\n", "").replace("\r", "")
            self._last_backend = "provisional_title_sequence_ocr"
            self._last_fallback_note = (
                "章节标题保留第一遍普通OCR完整序列；不按蓝色字框截断，也不调用PKStrokeRecognizer"
            )
            decisions = [
                AutoCharDecision(
                    index=index,
                    ocr_char=character,
                    chosen_char=character,
                    changed=False,
                    score=float(selective_ocr_confidence or 0.0),
                    reason="chapter_title_sequence_preserved_no_pkstroke",
                    candidates=[HandwritingCandidate(
                        text=character,
                        score=float(selective_ocr_confidence or 0.0),
                        source=f"chapter_title_ocr:{selective_ocr_engine or 'unknown'}",
                        reason="isolated chapter-title column; preserve whole sequence",
                    )],
                )
                for index, character in enumerate(preserved_text)
            ]
            return AutoColumnResult(
                block_id="",
                original_text=preserved_text,
                text=preserved_text,
                changed=False,
                reviewed=False,
                auto_segments=len(glyph_images),
                auto_changed_chars=0,
                decisions=decisions,
            )
        preaccepted_ocr: dict[int, object] = {}
        if selective_ocr_text and glyph_images and not manual_placeholders:
            try:
                from adapters.selective_column_ocr import (
                    SelectiveOcrCalibrationDB, evaluate_selective_column,
                )
                suspicious = _suspicious_learning_indices(
                    glyph_segments, self._last_segmentation_info,
                )
                report = evaluate_selective_column(
                    enabled=True,
                    candidate_text=selective_ocr_text,
                    raw_confidence=float(selective_ocr_confidence or 0.0),
                    engine=selective_ocr_engine,
                    fallback_characters=["□"] * len(glyph_images),
                    fallback_reasons=["pre_fallback_unresolved"] * len(glyph_images),
                    fallback_sources=[[] for _ in glyph_images],
                    suspicious_indices=suspicious,
                    threshold=float(selective_ocr_threshold or 0.95),
                    forbid_apple_automatic=True,
                    reject_conflicts=bool(selective_ocr_reject_conflicts),
                    calibration_db=SelectiveOcrCalibrationDB(),
                    candidate_character_confidences=selective_ocr_character_confidences,
                    variants=selective_ocr_variants,
                    require_stability=bool(selective_ocr_require_stability),
                )
                self._last_selective_report = report
                self._apply_low_confidence_glyph_rescue(report, glyph_images)
                preaccepted_ocr = {
                    item.index: item for item in report.decisions
                    if item.accepted or bool(getattr(item, "provisional", False))
                    or not bool(getattr(item, "automatic_fallback", True))
                }
            except Exception:
                self._last_selective_report = None
                preaccepted_ocr = {}
        if self._last_selective_report is not None and bool(
            getattr(self._last_selective_report, "preserve_original_text", False)
        ):
            preserved_text = clean_text(selective_ocr_text).replace("\n", "").replace("\r", "")
            preserved_decisions: list[AutoCharDecision] = []
            for index in range(len(glyph_images)):
                char = preserved_text[index] if index < len(preserved_text) else "□"
                preserved_decisions.append(AutoCharDecision(
                    index=index, ocr_char=char if char != "□" else "", chosen_char=char,
                    changed=True, score=float(selective_ocr_confidence or 0.0),
                    reason="column_length_mismatch_preserved_for_manual_no_pkstroke",
                    candidates=([HandwritingCandidate(
                        text=char, score=float(selective_ocr_confidence or 0.0),
                        source=f"provisional_column_ocr:{selective_ocr_engine or 'unknown'}",
                        reason="OCR字数与蓝框数不一致；保留第一遍OCR原文并整列人工检查",
                    )] if char != "□" else []),
                ))
            self._last_fallback_note = (
                "第一遍OCR字数与蓝色字框数不一致，无法在单OCR条件下安全定位缺字；"
                "已保留OCR原文并标记整列人工检查，未调用PKStrokeRecognizer"
            )
            self._last_backend = "provisional_column_ocr_manual"
            return AutoColumnResult(
                block_id="", original_text="", text=preserved_text, changed=True,
                reviewed=False, auto_segments=len(glyph_images),
                auto_changed_chars=len(preserved_text), decisions=preserved_decisions,
            )

        if not glyph_images:
            return AutoColumnResult(
                block_id="", original_text="", text="", changed=False,
                reviewed=False, auto_segments=0, auto_changed_chars=0,
                decisions=[],
            )

        if manual_placeholders:
            placeholders = "□" * len(glyph_images)
            manual_result = AutoColumnResult(
                block_id="", original_text="", text=placeholders, changed=True,
                reviewed=False, auto_segments=len(glyph_images), auto_changed_chars=0,
                decisions=[
                    AutoCharDecision(
                        index=index, ocr_char="", chosen_char="□", changed=True,
                        score=0.0, reason="black_ink_manual_placeholder", candidates=[],
                    ) for index in range(len(glyph_images))
                ],
            )
            self._record_learning_items(glyph_segments, manual_result)
            return manual_result

        # Candidate-fusion mode uses the same segmented glyph for four
        # independent signals: persistent memory, Apple Vision Top-N (with and
        # without language correction), PKStrokeRecognizer and local geometry.
        self._last_fallback_note = ""
        if self.vision_candidate_fusion or self.glyph_memory_enabled or preaccepted_ocr:
            fused_result = self._recognize_fused_black_ink_glyphs(
                glyph_images, strategy=strategy, use_character_mask=use_character_mask,
                preaccepted_ocr=preaccepted_ocr,
            )
            if self._last_selective_report is not None:
                initially_accepted = set(self._last_selective_report.accepted_indices)
                actual = [
                    decision.index for decision in fused_result.decisions
                    if decision.reason in {
                        "selective_column_ocr_high_confidence_short_circuit",
                        "low_confidence_glyph_multiview_confirmed_short_circuit",
                    }
                ]
                actual_set = set(actual)
                conflicts = set(self._last_selective_report.conflict_indices)
                for index in sorted(initially_accepted - actual_set):
                    if index >= len(fused_result.decisions):
                        continue
                    decision = fused_result.decisions[index]
                    expected = ""
                    if index < len(self._last_selective_report.decisions):
                        expected = str(self._last_selective_report.decisions[index].ocr_char or "")[:1]
                    if expected and decision.chosen_char not in {"□", expected}:
                        conflicts.add(index)
                self._last_selective_report.accepted_indices = actual
                self._last_selective_report.rejected_indices = [
                    index for index in range(len(glyph_images)) if index not in actual_set
                ]
                self._last_selective_report.conflict_indices = sorted(conflicts)
            self._record_learning_items(glyph_segments, fused_result)
            return fused_result

        # Apple-only single-glyph path. Apple Vision character anchors define
        # the cells; PKStrokeRecognizer still analyses every cell independently.
        # If a single-glyph result is unresolved, the corresponding Apple Vision
        # whole-column Character result may fill that same slot. No non-Apple
        # recognizer is consulted.
        self._last_apple_attempted = False
        self._last_apple_succeeded = False
        pkstroke_text = ""
        if self._apple is not None:
            pkstroke_text = str(self._recognize_with_apple(glyph_images) or "").strip()
        pkstroke_slots = list(pkstroke_text) if len(pkstroke_text) == len(glyph_images) else []
        output: list[str] = []
        decisions: list[AutoCharDecision] = []
        used_pkstroke = False
        used_anchor = False
        for index, segment in enumerate(glyph_segments):
            pk_char = pkstroke_slots[index] if index < len(pkstroke_slots) else ""
            anchor_char = self._candidate_base_char(getattr(segment, "anchor_text", ""))
            anchor_confidence = float(getattr(segment, "anchor_confidence", 0.0) or 0.0)
            if pk_char and pk_char != "□":
                chosen = pk_char
                reason = "apple_pkstroke_first_result"
                candidates = [HandwritingCandidate(
                    text=chosen, score=1.0, source="apple_pkstroke_recognizer",
                    reason="single_glyph_recognizedText_first_result",
                )]
                used_pkstroke = True
            elif anchor_char and anchor_char != "□":
                chosen = anchor_char
                reason = "apple_vision_character_anchor_fallback"
                candidates = [HandwritingCandidate(
                    text=chosen,
                    score=max(0.35, min(1.0, anchor_confidence)),
                    source="apple_vision_column_anchor",
                    reason="RecognizeTextRequest.fast boundingBox(for: Character)",
                )]
                used_anchor = True
            else:
                chosen = "□"
                reason = "apple_only_unresolved"
                candidates = []
            output.append(chosen)
            decisions.append(AutoCharDecision(
                index=index, ocr_char="", chosen_char=chosen, changed=True,
                score=(candidates[0].score if candidates else 0.0),
                reason=reason, candidates=candidates,
            ))

        text = "".join(output)
        if any(ch != "□" for ch in output):
            if used_pkstroke and used_anchor:
                self._last_backend = "apple_pkstroke+vision_character_anchor"
            elif used_pkstroke:
                self._last_backend = "apple_pkstroke_partial" if "□" in text else "apple_pkstroke"
            else:
                self._last_backend = "apple_vision_character_anchor"
            unresolved = sum(ch == "□" for ch in output)
            if unresolved:
                self._last_fallback_note = (
                    f"Apple逐字子集识别成功/字符锚点已解决 {len(output)-unresolved}/{len(output)} 字；"
                    "其余单字框保留为□，未调用其他识别器"
                )
            result = AutoColumnResult(
                block_id="", original_text="", text=text, changed=True,
                reviewed=True, auto_segments=len(glyph_images),
                auto_changed_chars=sum(ch != "□" for ch in output), decisions=decisions,
            )
            self._record_learning_items(glyph_segments, result)
            return result

        # Novel-formatter-1 compatibility: Apple is preferred but never the
        # only available recognizer.  If Apple/anchors cannot resolve the
        # column, retain the existing OpenVINO and bundled JLect fallbacks.
        if self.recognition_backend in {"auto", "openvino", "jlect"}:
            if self._openvino is None and self.recognition_backend in {"auto", "openvino"}:
                self._ensure_openvino(strict=False)
            if self._openvino is not None:
                try:
                    traced_line = self._render_trace_line(glyph_images)
                    fallback_text = str(self._openvino.recognize(traced_line) or "").strip()
                    plausible = bool(fallback_text) and len(fallback_text) <= max(1, len(glyph_images) * 2)
                    if plausible:
                        self._last_backend = "openvino"
                        self._last_fallback_note = "Apple 未返回可用文字；已使用 OpenVINO 输出"
                        result = AutoColumnResult(
                            block_id="", original_text="", text=fallback_text, changed=True,
                            reviewed=True, auto_segments=len(glyph_images),
                            auto_changed_chars=len(fallback_text),
                            decisions=[
                                AutoCharDecision(
                                    index=index, ocr_char="", chosen_char=ch, changed=True,
                                    score=1.0, reason="openvino_trace_line",
                                    candidates=[HandwritingCandidate(
                                        text=ch, score=1.0, source="openvino_handwriting",
                                        reason="ctc_first_result",
                                    )],
                                )
                                for index, ch in enumerate(fallback_text)
                            ],
                        )
                        self._record_learning_items(glyph_segments, result)
                        return result
                except Exception as exc:
                    self._openvino_error = str(exc)

            self._last_backend = "jlect_black_ink"
            output: list[str] = []
            fallback_decisions: list[AutoCharDecision] = []
            for index, glyph in enumerate(glyph_images):
                candidates = self.recognize_image_candidates(
                    glyph, expected_char="", use_glyph_mask=use_character_mask,
                )
                if candidates and self._candidate_can_fill_empty_ocr(candidates[0], strategy=strategy):
                    top = candidates[0]
                    chosen = top.text
                    score = top.score
                    reason = "black_ink_first_candidate"
                else:
                    chosen = "□"
                    score = candidates[0].score if candidates else 0.0
                    reason = "black_ink_unresolved"
                output.append(chosen)
                fallback_decisions.append(AutoCharDecision(
                    index=index, ocr_char="", chosen_char=chosen, changed=True,
                    score=score, reason=reason, candidates=candidates[:5],
                ))
            fallback_text = "".join(output)
            self._last_fallback_note = (
                "Apple 未返回可用文字；已使用 Novel-formatter-1 本地 JLect/符号候选，"
                "未决字保留为□"
            )
            result = AutoColumnResult(
                block_id="", original_text="", text=fallback_text, changed=bool(fallback_text),
                reviewed=True, auto_segments=len(glyph_images),
                auto_changed_chars=sum(ch != "□" for ch in output),
                decisions=fallback_decisions,
            )
            self._record_learning_items(glyph_segments, result)
            return result

        self._last_backend = "apple_pkstroke_unavailable"
        reason_detail = self._apple_error or (
            "Apple桥接不可用" if self._apple is None else "Apple返回空结果或字数与逐字框不一致"
        )
        self._last_fallback_note = (
            f"Apple未返回完整逐框结果：{reason_detail}；"
            "已按实际逐字框保留□，未调用OpenVINO/JLect"
        )
        text = "□" * len(glyph_images)
        decisions = [
            AutoCharDecision(
                index=index,
                ocr_char="",
                chosen_char="□",
                changed=True,
                score=0.0,
                reason="apple_only_unresolved",
                candidates=[],
            )
            for index in range(len(glyph_images))
        ]
        result = AutoColumnResult(
            block_id="",
            original_text="",
            text=text,
            changed=bool(text),
            reviewed=True,
            auto_segments=len(glyph_images),
            auto_changed_chars=0,
            decisions=decisions,
        )
        self._record_learning_items(glyph_segments, result)
        return result

    def auto_recognize_column(
        self,
        column_image: Image.Image,
        ocr_text: str,
        *,
        strategy: str = "balanced",
        use_character_mask: bool = True,
        insert_missing_symbols: bool = True,
        independent: bool = False,
        manual_placeholders: bool = False,
        expected_count: int | None = None,
        character_anchors: Sequence[object] | None = None,
        precomputed_boxes: Sequence[object] | None = None,
        selective_ocr_text: str = "",
        selective_ocr_confidence: float = 0.0,
        selective_ocr_engine: str = "",
        selective_ocr_threshold: float = 0.95,
        selective_ocr_reject_conflicts: bool = True,
        selective_ocr_character_confidences: Sequence[float] | None = None,
        selective_ocr_variants: Sequence[dict] | None = None,
        selective_ocr_require_stability: bool = False,
        preserve_sequence_ocr: bool = False,
    ) -> AutoColumnResult:
        if independent:
            return self.recognize_black_ink_column(
                column_image,
                strategy=strategy,
                use_character_mask=use_character_mask,
                manual_placeholders=manual_placeholders,
                expected_count=expected_count,
                character_anchors=character_anchors,
                precomputed_boxes=precomputed_boxes,
                selective_ocr_text=selective_ocr_text,
                selective_ocr_confidence=selective_ocr_confidence,
                selective_ocr_engine=selective_ocr_engine,
                selective_ocr_threshold=selective_ocr_threshold,
                selective_ocr_reject_conflicts=selective_ocr_reject_conflicts,
                selective_ocr_character_confidences=selective_ocr_character_confidences,
                selective_ocr_variants=selective_ocr_variants,
                selective_ocr_require_stability=selective_ocr_require_stability,
                preserve_sequence_ocr=preserve_sequence_ocr,
            )
        text = str(ocr_text or "")
        chars = list(text)
        raw_segments = self.split_vertical_column_into_characters(
            column_image,
            expected_count=None,
            mask_main_band=use_character_mask,
        )

        # When the primary OCR returned nothing, still allow the handwriting
        # engine to recover high-confidence first candidates from the masked
        # column. This is especially useful for isolated punctuation columns.
        if not chars:
            out_chars: list[str] = []
            decisions: list[AutoCharDecision] = []
            for seg_index, (_, _, seg_img) in enumerate(raw_segments):
                candidates = self.recognize_image_candidates(
                    seg_img, use_glyph_mask=use_character_mask,
                )
                if not candidates or not self._candidate_can_fill_empty_ocr(candidates[0], strategy=strategy):
                    continue
                top = candidates[0]
                out_chars.append(top.text)
                decisions.append(AutoCharDecision(
                    index=len(out_chars) - 1,
                    ocr_char="",
                    chosen_char=top.text,
                    changed=True,
                    score=top.score,
                    reason="first_candidate_empty_ocr",
                    candidates=candidates[:5],
                ))
            new_text = "".join(out_chars)
            return AutoColumnResult(
                block_id="",
                original_text=text,
                text=new_text,
                changed=bool(new_text),
                reviewed=bool(new_text),
                auto_segments=len(raw_segments),
                auto_changed_chars=len(new_text),
                decisions=decisions,
            )

        # Preserve extra compact segments that look like high-confidence symbols.
        # This allows automatic recovery of punctuation omitted by the primary OCR.
        inserted_symbols: dict[int, HandwritingCandidate] = {}
        extra_needed = max(0, len(raw_segments) - len(chars))
        if insert_missing_symbols and 0 < extra_needed <= 6:
            symbol_options: list[tuple[float, int, HandwritingCandidate]] = []
            for seg_index, (_, _, seg_img) in enumerate(raw_segments):
                candidate = self._standalone_symbol_candidate(
                    seg_img,
                    strategy=strategy,
                    use_glyph_mask=use_character_mask,
                )
                if candidate is not None:
                    symbol_options.append((candidate.score, seg_index, candidate))
            for _, seg_index, candidate in sorted(symbol_options, reverse=True)[:extra_needed]:
                inserted_symbols[seg_index] = candidate
            if len(raw_segments) - len(inserted_symbols) != len(chars):
                inserted_symbols.clear()

        if inserted_symbols:
            segments = raw_segments
        else:
            # Fall back to OCR-count-guided segmentation if the raw projection
            # cannot be explained by a small number of missing symbols.
            segments = self.split_vertical_column_into_characters(
                column_image,
                expected_count=len(chars),
                mask_main_band=use_character_mask,
            )

        out_chars: list[str] = []
        decisions: list[AutoCharDecision] = []
        changed_count = 0
        char_index = 0

        for seg_index, (_, _, seg_img) in enumerate(segments):
            if seg_index in inserted_symbols:
                top = inserted_symbols[seg_index]
                out_chars.append(top.text)
                changed_count += 1
                decisions.append(AutoCharDecision(
                    index=len(out_chars) - 1,
                    ocr_char="",
                    chosen_char=top.text,
                    changed=True,
                    score=top.score,
                    reason="insert_missing_symbol_first_candidate",
                    candidates=[top],
                ))
                continue

            if char_index >= len(chars):
                break
            orig = chars[char_index]
            char_index += 1
            candidates = self.recognize_image_candidates(
                seg_img,
                expected_char=orig,
                use_glyph_mask=use_character_mask,
            ) if _should_attempt_recognition(orig) else []
            if candidates:
                top = candidates[0]
                second = candidates[1] if len(candidates) > 1 else None
                do_replace, reason = self._should_replace(orig, top, second, strategy=strategy)
                chosen = top.text if do_replace else orig
                changed = chosen != orig
                if changed:
                    changed_count += 1
                decisions.append(AutoCharDecision(
                    index=len(out_chars),
                    ocr_char=orig,
                    chosen_char=chosen,
                    changed=changed,
                    score=top.score,
                    reason=reason,
                    candidates=candidates[:5],
                ))
                out_chars.append(chosen)
            else:
                out_chars.append(orig)
                decisions.append(AutoCharDecision(
                    index=len(out_chars) - 1,
                    ocr_char=orig,
                    chosen_char=orig,
                    changed=False,
                    score=0.0,
                    reason="no_candidates",
                    candidates=[],
                ))

        # Preserve OCR tail characters if segmentation still ended early.
        while char_index < len(chars):
            ch = chars[char_index]
            char_index += 1
            out_chars.append(ch)
            decisions.append(AutoCharDecision(
                index=len(out_chars) - 1,
                ocr_char=ch,
                chosen_char=ch,
                changed=False,
                score=0.0,
                reason="segmentation_short",
                candidates=[],
            ))

        new_text = "".join(out_chars)
        return AutoColumnResult(
            block_id="",
            original_text=text,
            text=new_text,
            changed=(new_text != text),
            reviewed=True,
            auto_segments=len(raw_segments),
            auto_changed_chars=changed_count,
            decisions=decisions,
        )

    def merge_column_result(self, chars, punctuation=True):
        text = "".join(chars)
        if punctuation:
            return text
        return text


def _block_type_for_text(text: str, fallback: BlockType) -> BlockType:
    stripped = str(text or "").strip()
    if stripped.startswith(("「", "『")) or stripped.endswith(("」", "』")):
        return BlockType.DIALOGUE
    if fallback == BlockType.DIALOGUE:
        return BlockType.PARAGRAPH
    return fallback


def _is_direct_ordinary_ocr_title(block: Block) -> bool:
    """Return True when the existing ordinary OCR block is already a title.

    Title recognition is a sequence task, not a per-glyph correction task.  A
    block may already be promoted to CHAPTER by the common OCR adapter even when
    the geometry-only ``column_role`` label was not set.  The old handwritten
    pass checked only that label and therefore sent some correctly recognized
    titles through glyph segmentation again, where they could be truncated or
    disappear.  Treat all explicit title signals as authoritative for this
    bypass while leaving ordinary body columns unchanged.
    """
    text = clean_text(getattr(block, "text", "")).strip()
    if not text:
        return False
    metadata = dict(getattr(block, "metadata", None) or {})
    if getattr(block, "type", None) in {BlockType.CHAPTER, BlockType.SECTION}:
        return True
    if str(metadata.get("column_role") or "") == "chapter_title":
        return True
    if bool(metadata.get("column_title_candidate")):
        return True
    if str(metadata.get("label") or "").lower() in {"title", "chapter", "section"}:
        return True
    try:
        from engine.formatter import CHAPTER_RE
        compact = re.sub(r"[\s　]+", "", text)
        return bool(CHAPTER_RE.match(text) or CHAPTER_RE.match(compact))
    except Exception:
        return bool(re.match(
            r"^(?:プロローグ|フロローグ|ブロローグ|エピローグ|序章|終章|幕間|第.+?[章話節回])",
            re.sub(r"[\s　]+", "", text),
        ))


def _ensure_title_toc_entry(doc: UnifiedDocument, block: Block) -> None:
    """Keep a direct-OCR title visible to Formatter/EPUB navigation."""
    try:
        block_index = doc.blocks.index(block)
    except ValueError:
        return
    existing_entry = next(
        (item for item in doc.toc if int(item.block_index) == block_index),
        None,
    )
    if block.type == BlockType.SECTION:
        return
    if block.type != BlockType.CHAPTER:
        block.type = BlockType.CHAPTER
    if existing_entry is not None:
        existing_entry.title = str(block.text or "").strip()
        if int(getattr(existing_entry, "chapter_index", 0) or 0) > 0:
            block.chapter_index = int(existing_entry.chapter_index)
        return
    previous_indices = [
        int(getattr(item, "chapter_index", 0) or 0)
        for item in doc.blocks[:block_index]
        if getattr(item, "type", None) == BlockType.CHAPTER
    ]
    chapter_index = int(getattr(block, "chapter_index", 0) or 0)
    if chapter_index <= 0:
        chapter_index = max(previous_indices or [0]) + 1
        block.chapter_index = chapter_index
    doc.toc.append(TocEntry(
        title=str(block.text or "").strip(),
        chapter_index=chapter_index,
        block_index=block_index,
    ))
    doc.toc.sort(key=lambda item: int(item.block_index))


def _suspicious_learning_indices(segments, segmentation_info: dict | None = None) -> set[int]:
    """Return oversized glyph boxes that must not enter the memory database.

    A vertically merged two-character crop is usually much taller than the
    page's normal glyph pitch.  Recognition may still show it for correction,
    but learning it as one character would create a destructive exact match.
    """
    heights = [
        max(1.0, float(getattr(segment, "y1", 0) or 0) - float(getattr(segment, "y0", 0) or 0))
        for segment in (segments or [])
    ]
    if len(heights) < 2:
        return set()
    ordered = sorted(heights)
    middle = len(ordered) // 2
    median_height = (
        ordered[middle] if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )
    info = dict(segmentation_info or {})
    target = float(
        info.get("projection_target_height")
        or info.get("target_pitch")
        or median_height
        or 1.0
    )
    # Do not let detached punctuation lower the baseline.  Use whichever of
    # the projection target and observed median is larger.
    baseline = max(1.0, target, median_height)
    threshold = max(baseline * 1.55, baseline + 8.0)
    return {index for index, height in enumerate(heights) if height > threshold}


def run_handwriting_input_card(
    doc: UnifiedDocument,
    *,
    crop_rect: tuple[float, float, float, float] | None,
    mode: str = "hybrid",
    strategy: str = "balanced",
    character_mask: bool = True,
    insert_missing_symbols: bool = True,
    recognition_backend: str = "auto",
    independent: bool = True,
    candidate_only: bool = False,
    progress_callback=None,
) -> dict:
    """Run handwriting analysis on fixed-region physical columns.

    ``independent=True`` preserves the legacy experimental mode: OCR text is not
    consulted and black-pixel glyphs become the output.  ``independent=False``
    aligns glyph crops to the existing OCR text.  With ``candidate_only=True``
    (the recommended OCR + manual-correction workflow), disagreements are saved
    as review metadata but the OCR text is never modified automatically.
    """
    mode = str(mode or "hybrid")
    strategy = str(strategy or "balanced")
    recognition_backend = str(recognition_backend or "auto")
    candidate_only = bool(candidate_only)
    summary = {
        "mode": mode,
        "strategy": strategy,
        "blocks": 0,
        "changed_blocks": 0,
        "changed_chars": 0,
        "written_chars": 0,
        "fallback_columns": 0,
        "records": [],
        "manual_review_recommended": False,
        "uses_masked_columns": bool(character_mask),
        "insert_missing_symbols": bool(insert_missing_symbols),
        "unresolved_empty_columns": 0,
        "recovered_empty_columns": 0,
        "recognition_backend": recognition_backend,
        "independent_black_ink": bool(independent),
        "candidate_only": candidate_only,
        "unresolved_glyphs": 0,
        "used_backends": [],
        "disagreement_columns": 0,
    }

    # Baseline comparison currently uses the deterministic bundled JLect table.
    # Avoid initializing Apple/OpenVINO just to rank review candidates; the user
    # can still use Apple's native panel or the macOS Japanese input source in
    # the manual review window.
    effective_backend = recognition_backend if independent else "jlect"
    card = JapaneseHandwritingCard(recognition_backend=effective_backend)
    if not card._table and card._openvino is None and card._apple is None:
        doc.add_log(
            "handwriting_input_card",
            "手写候选分析：没有可用的本地候选器",
            0,
        )
        summary["manual_review_recommended"] = True
        return summary

    with tempfile.TemporaryDirectory(prefix="novel_formatter_hw_black_ink_") as tmpdir:
        records = prepare_review_records(
            doc,
            crop_rect=crop_rect,
            output_dir=tmpdir,
            mask_main_band=False,
            enable_character_sweep=False,
        )
        if not records:
            doc.add_log(
                "handwriting_input_card",
                "手写候选分析：没有找到固定区域物理列",
                0,
            )
            summary["manual_review_recommended"] = mode != "auto"
            return summary

        by_id = {block.id: block for block in doc.blocks}
        total_records = len(records)
        if progress_callback is not None:
            baseline_note = "以普通 OCR 为底稿，仅生成疑点候选" if not independent else "普通 OCR 文本不参与"
            progress_callback(
                "handwriting_prepare", 0, total_records,
                f"准备 {total_records} 个原始列图；{baseline_note}",
            )

        for record_index, record in enumerate(records, start=1):
            block_id = str(record.get("block_id", ""))
            block = by_id.get(block_id)
            if block is None:
                continue
            image_name = Path(str(record.get("image", ""))).name
            image_path = Path(tmpdir) / image_name
            if not image_path.exists():
                continue
            page_no = int(record.get("page", 0) or 0)
            column_no = int(record.get("column", 0) or 0)
            detail_prefix = f"第 {page_no} 页 · 右起第 {column_no} 列"
            old_text = str(block.text or "")

            try:
                if progress_callback is not None:
                    progress_callback(
                        "handwriting_segment", record_index, total_records,
                        f"{detail_prefix} · 根据黑色连通区域和空白谷值切字",
                    )
                with Image.open(image_path) as im:
                    column_image = im.convert("RGB")
                    if progress_callback is not None:
                        progress_callback(
                            "handwriting_trace", record_index, total_records,
                            f"{detail_prefix} · 提取中心线并与 OCR 字符逐位比较",
                        )
                        if independent and card._apple is not None and mode != "manual":
                            progress_callback(
                                "handwriting_pkdrawing", record_index, total_records,
                                f"{detail_prefix} · 每次只准备当前单字的分笔点序列",
                            )
                            progress_callback(
                                "handwriting_apple", record_index, total_records,
                                f"{detail_prefix} · 单字提交→等待首结果→清空→再处理下一字（ja-JP）",
                            )
                    result = card.auto_recognize_column(
                        column_image,
                        "" if independent else old_text,
                        strategy=strategy,
                        use_character_mask=bool(character_mask),
                        insert_missing_symbols=bool(insert_missing_symbols),
                        independent=bool(independent),
                        manual_placeholders=bool(independent and mode == "manual"),
                    )
                if progress_callback is not None:
                    progress_callback(
                        "handwriting_recognize", record_index, total_records,
                        f"{detail_prefix} · {'生成复核候选' if not independent else '独立识别器返回首结果'}",
                    )
            except Exception as exc:
                if progress_callback is not None:
                    progress_callback(
                        "handwriting_error", record_index, total_records,
                        f"{detail_prefix} · {exc}",
                    )
                continue

            result.block_id = block_id
            was_empty = not bool(old_text.strip())
            summary["blocks"] += 1
            summary["changed_chars"] += result.auto_changed_chars
            summary["written_chars"] += len(str(result.text or ""))
            if card._last_fallback_note:
                summary["fallback_columns"] += 1
                if progress_callback is not None:
                    progress_callback(
                        "handwriting_fallback", record_index, total_records,
                        f"{detail_prefix} · {card._last_fallback_note}",
                    )

            disagreements = []
            for decision in result.decisions[:256]:
                evidence = _review_evidence_from_decision(decision)
                if evidence is not None:
                    disagreements.append(evidence)
            if disagreements:
                summary["disagreement_columns"] += 1

            # Candidate-only mode is non-destructive by definition.
            should_apply = bool(not candidate_only and (result.changed or result.text != old_text))
            if should_apply:
                summary["changed_blocks"] += 1
                block.text = result.text
                if old_text and not block.ocr_raw:
                    block.ocr_raw = old_text
                block.type = _block_type_for_text(block.text, block.type)
                block.modified_by = _append_modified_by(
                    block.modified_by, "black_ink_handwriting",
                )
                block.confidence = 1.0 if card.active_backend in {"apple_pkstroke", "apple_pkstroke_partial", "openvino"} else 0.65

            current_text = str(block.text or "")
            is_still_empty = not bool(current_text.strip())
            unresolved_glyphs = (str(result.text or "").count("□") if independent else current_text.count("□"))
            requires_manual = bool(mode == "manual" or is_still_empty or unresolved_glyphs or disagreements)
            summary["unresolved_glyphs"] += unresolved_glyphs
            if was_empty and not is_still_empty and not unresolved_glyphs:
                summary["recovered_empty_columns"] += 1
            if is_still_empty:
                summary["unresolved_empty_columns"] += 1

            metadata = dict(block.metadata or {})
            metadata.update({
                "handwriting_input_mode": mode,
                "handwriting_input_strategy": strategy,
                "handwriting_input_auto_processed": True,
                "handwriting_input_uses_masked_columns": bool(character_mask),
                "handwriting_input_insert_missing_symbols": bool(insert_missing_symbols),
                "handwriting_input_auto_segments": int(result.auto_segments),
                "handwriting_input_auto_changed_chars": int(result.auto_changed_chars),
                "handwriting_input_auto_changed": bool(result.changed),
                "handwriting_input_output_policy": (
                    "ocr_baseline_candidate_only" if candidate_only else
                    ("independent_first_result" if independent else "ocr_assisted_replace")
                ),
                "handwriting_input_recognition_backend": card.active_backend,
                "handwriting_input_apple_pkstroke_used": card.active_backend in {"apple_pkstroke", "apple_pkstroke_partial"},
                "handwriting_input_apple_pkstroke_error": str(card._apple_error or ""),
                "handwriting_input_fallback_note": str(card._last_fallback_note or ""),
                "handwriting_input_apple_attempted": bool(card._last_apple_attempted),
                "handwriting_input_apple_succeeded": bool(card._last_apple_succeeded),
                "handwriting_input_independent_black_ink": bool(independent),
                "handwriting_input_ordinary_ocr_used": bool(not independent),
                "handwriting_input_candidate_only": candidate_only,
                "handwriting_input_manual_ime_supported": True,
                "handwriting_input_still_empty": bool(is_still_empty),
                "handwriting_input_unresolved_glyphs": int(unresolved_glyphs),
                "handwriting_review_disagreements": disagreements,
                "column_ocr_empty": bool(is_still_empty),
                "column_requires_handwriting": bool(requires_manual),
            })
            preview = []
            disagreement_by_index = {
                int(item.get("index", -1)): item for item in disagreements
                if isinstance(item, dict)
            }
            for decision in result.decisions[:64]:
                evidence = disagreement_by_index.get(int(decision.index))
                preview.append({
                    "i": int(decision.index),
                    "ocr": str(decision.ocr_char or ""),
                    "out": str(
                        evidence.get("candidate", "") if evidence is not None
                        else (decision.chosen_char or "")
                    ),
                    "s": round(float(
                        evidence.get("score", 0.0) if evidence is not None
                        else (decision.score or 0.0)
                    ), 4),
                    "r": str(
                        evidence.get("reason", "") if evidence is not None
                        else decision.reason
                    ),
                    "amb": bool(evidence and evidence.get("ambiguous")),
                    "src": decision.candidates[0].source if decision.candidates else "",
                    "c": [cand.text for cand in decision.candidates[:5]],
                })
            if preview:
                metadata["handwriting_input_auto_preview"] = preview
            block.metadata = metadata
            summary["records"].append(result)
            if card.active_backend not in summary["used_backends"]:
                summary["used_backends"].append(card.active_backend)

            if progress_callback is not None:
                if candidate_only:
                    outcome = f"OCR 保持不变；发现 {len(disagreements)} 个候选冲突"
                elif mode == "manual" and independent:
                    outcome = f"检测到 {result.auto_segments} 个黑像素字位，以 □ 等待 macOS 输入法填写"
                elif unresolved_glyphs:
                    outcome = f"输出 {result.auto_changed_chars} 字，另有 {unresolved_glyphs} 个字形保留为 □"
                elif result.text:
                    outcome = f"首结果输出：{result.text}"
                else:
                    outcome = "未获得结果，等待人工输入"
                progress_callback(
                    "handwriting_output", record_index, total_records,
                    f"{detail_prefix} · {outcome}",
                )

    if candidate_only:
        message = (
            f"OCR + 手写候选筛查：处理 {summary['blocks']} 列，"
            f"{summary['disagreement_columns']} 列存在候选冲突；正文未自动修改"
        )
    else:
        message = (
            f"黑像素独立临摹识别：处理 {summary['blocks']} 列；"
            f"写入 {summary['written_chars']} 字位，其中自动识别 {summary['changed_chars']} 字；"
            f"未决字形 {summary['unresolved_glyphs']}；回退 {summary['fallback_columns']} 列；"
            f"实际后端 {','.join(summary['used_backends']) or recognition_backend}；"
            f"普通 OCR 文本{'参与' if not independent else '未参与'}"
        )
    doc.add_log("handwriting_input_card", message, int(summary["changed_chars"]))
    summary["manual_review_recommended"] = mode in {"hybrid", "manual"}
    return summary
