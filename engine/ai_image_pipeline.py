# -*- coding: utf-8 -*-
"""AI image-book extraction/translation pipeline.

This is a deliberately separate path from traditional OCR.  Page Manager owns
page order/type/assets; a multimodal model owns textual interpretation and may
return already-structured prose.  No traditional OCR comparison or Formatter
pass is required unless the user explicitly chooses those legacy workspaces.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import math
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping

from PIL import Image, ImageOps

from ai.config import AISettings, CONFIG_PATH
from ai.multimodal_client import MultimodalClient
from ai.provider_factory import create_provider
from models.document import Block, BlockType, BoundingBox, Metadata, PageInfo, TocEntry, UnifiedDocument
from engine.page_ocr_policy import page_type_value

PIPELINE_VERSION = "ai-image-book-v2.2"
_TEXT_BLOCK_TYPES = {
    "paragraph": BlockType.PARAGRAPH,
    "dialogue": BlockType.DIALOGUE,
    "chapter": BlockType.CHAPTER,
    "section": BlockType.SECTION,
    "footnote": BlockType.FOOTNOTE,
}

# AI image-book mode is intentionally not identical to the legacy OCR admission
# policy.  Semantic text pages such as AFTERWORD should still be read/translated;
# visual-only/front-matter pages remain local assets and are restored by Page
# Manager during EPUB handoff.
_AI_PRESERVED_ASSET_PAGE_TYPES = frozenset({
    BlockType.COVER.value,
    BlockType.COLOR_ILLUS.value,
    BlockType.BLANK.value,
    BlockType.TOC_PAGE.value,
    BlockType.ILLUSTRATION.value,
    BlockType.COLOPHON.value,
    BlockType.HALF_ILLUS.value,
    BlockType.TITLE_PAGE.value,
    BlockType.FRONTISPIECE.value,
    BlockType.INSERT.value,
    BlockType.ADVERTISEMENT.value,
    BlockType.MAP_PAGE.value,
    BlockType.CHARACTER_SHEET.value,
})


def _should_skip_ai_page(value: object) -> bool:
    return page_type_value(value) in _AI_PRESERVED_ASSET_PAGE_TYPES


@dataclass(slots=True)
class AIImageOptions:
    mode: str = "extract"  # extract | extract_translate
    target_language: str = "zh-Hans"
    quality: str = "balanced"  # fast | balanced | publication
    batch_pages: int = 2
    trim_white_margins: bool = True
    auto_review: bool = True
    use_cache: bool = True
    max_visual_reviews: int = 96
    adaptive_batch_split: bool = True
    translation_repair: bool = False
    max_translation_repairs: int = 160
    verify_source_fingerprint: bool = True
    audit_suspicious_short_pages: bool = True
    continuity_text_check: bool = True
    max_short_page_audits: int = 24
    max_continuity_checks: int = 96
    visual_concurrency: int = 0  # 0=auto; custom/Ollama stay conservative unless explicitly set
    max_auto_visual_concurrency: int = 16
    initial_auto_visual_concurrency: int = 4
    prepared_image_format: str = "webp"  # webp | png
    prepared_webp_quality: int = 88
    audit_flat_layout_pages: bool = True
    max_structure_audits: int = 32
    direct_full_page_api: bool = True
    direct_original_when_safe: bool = True
    edge_integrity_guard: bool = True
    max_edge_integrity_audits: int = 24
    layout_geometry_guard: bool = True
    max_layout_geometry_audits: int = 48
    layout_geometry_min_columns: int = 5
    audit_reasoning_high: bool = True

    @property
    def translate(self) -> bool:
        return self.mode == "extract_translate"

    @property
    def max_edge(self) -> int:
        return {"fast": 1600, "publication": 2560}.get(self.quality, 1920)

    @property
    def review_max_edge(self) -> int:
        return {"fast": 1800, "publication": 3600}.get(self.quality, 2800)


@dataclass(slots=True)
class PreparedPage:
    page_no: int
    source_path: str
    prepared_path: str
    source_size: tuple[int, int]
    source_crop: tuple[int, int, int, int]
    page_type: str
    source_sha256: str


@dataclass(slots=True)
class AITextRecord:
    block_id: str
    page_no: int
    block_type: str
    source: str
    translation: str = ""
    confidence: str = "high"
    source_sha256: str = ""
    merged_from: tuple[str, ...] = ()


@dataclass(slots=True)
class AIReviewRegion:
    page_no: int
    block_id: str
    bbox: tuple[float, float, float, float]
    reason: str = "uncertain"


@dataclass(slots=True)
class AIImageResult:
    source_document: UnifiedDocument
    translated_document: UnifiedDocument | None
    stats: dict = field(default_factory=dict)
    page_results: dict[int, list[AITextRecord]] = field(default_factory=dict)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stale_ai_image_pages(doc: UnifiedDocument | None, page_images: Iterable[str | Path]) -> list[int]:
    """Return AI text pages whose current Page Manager source bytes changed.

    Processing-time checks protect an active run.  This second guard protects the
    gap between a completed run and a later EPUB click, when an external editor
    may have replaced a scan at the same path without reopening the project.
    """
    if doc is None:
        return []
    images = [str(Path(item)) for item in page_images]
    expected: dict[int, str] = {}
    for block in getattr(doc, "blocks", []) or []:
        if getattr(block, "type", None) == BlockType.IMAGE_REF:
            continue
        try:
            page_no = int(getattr(block, "page", 0) or 0)
        except Exception:
            continue
        metadata = dict(getattr(block, "metadata", {}) or {})
        fingerprint = str(metadata.get("source_page_sha256") or "").strip()
        if page_no > 0 and fingerprint:
            expected.setdefault(page_no, fingerprint)
    stale: list[int] = []
    for page_no, fingerprint in sorted(expected.items()):
        if page_no > len(images):
            stale.append(page_no)
            continue
        try:
            current = _sha256_file(images[page_no - 1])
        except Exception:
            stale.append(page_no)
            continue
        if current != fingerprint:
            stale.append(page_no)
    return stale


def _json_object(raw: str) -> dict:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("AI 返回内容为空")
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.I | re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except Exception:
        pass
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        value = json.loads(text[first:last + 1])
        if isinstance(value, dict):
            return value
    raise ValueError("AI 未返回有效 JSON 对象")


def _safe_type(value: object) -> str:
    item = str(value or "paragraph").strip().lower()
    return item if item in _TEXT_BLOCK_TYPES else "paragraph"


def _record_type(value: object, source: str) -> str:
    """Normalise the model block label without rewriting text.

    A complete Japanese quote occupying a whole returned block is a dialogue
    paragraph in light-novel layout even when a provider labels everything as
    ``paragraph``.  This is intentionally conservative: embedded quotations stay
    narrative and chapter/section labels supplied by the model are untouched.
    """
    block_type = _safe_type(value)
    text = str(source or "").strip()
    quote_pairs = (("「", "」"), ("『", "』"))
    if block_type == "paragraph" and any(text.startswith(a) and text.endswith(b) for a, b in quote_pairs):
        return "dialogue"
    return block_type


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float)):
        return str(value).strip()
    return ""


def _bbox(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x, y, w, h = [float(x) for x in value]
    except Exception:
        return None
    if not all(math.isfinite(v) for v in (x, y, w, h)):
        return None
    x = min(1.0, max(0.0, x)); y = min(1.0, max(0.0, y))
    w = min(1.0 - x, max(0.0, w)); h = min(1.0 - y, max(0.0, h))
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)




_JP_KANA_RE = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff]")
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_SPACE_RE = re.compile(r"\s+")


def _text_identity(value: str) -> str:
    return _SPACE_RE.sub("", str(value or "")).strip()


def _records_text(records: Iterable[AITextRecord]) -> str:
    return "".join(str(record.source or "") for record in records)


def _records_char_count(records: Iterable[AITextRecord]) -> int:
    return len(_text_identity(_records_text(records)))


def _source_overlap_ratio(old_text: str, new_text: str) -> float:
    """How much of the original OCR survives in a proposed fuller-page rescue.

    A targeted rescue is allowed to fix OCR mistakes and block boundaries, but it
    must not replace a short page with unrelated hallucinated prose.  SequenceMatcher
    gives a conservative ordered-character coverage score without requiring exact
    punctuation identity.
    """
    old = _text_identity(old_text)
    new = _text_identity(new_text)
    if not old:
        return 1.0
    if not new:
        return 0.0
    matched = sum(block.size for block in difflib.SequenceMatcher(None, old, new, autojunk=False).get_matching_blocks())
    return matched / max(1, len(old))


_STRONG_PARAGRAPH_END_RE = re.compile(r"[。！？!?…‥」』】）》）\]\)]$")
_DIALOGUE_SPAN_RE = re.compile(r"(?:「[^」]{2,}」|『[^』]{2,}』)")
_TRANSLATED_DIALOGUE_SPAN_RE = re.compile(
    r'(?:「[^」]{2,}」|『[^』]{2,}』|“[^”]{2,}”|\"[^\"]{2,}\")'
)


def _has_embedded_dialogue(text: str) -> bool:
    value = str(text or "").strip()
    for match in _DIALOGUE_SPAN_RE.finditer(value):
        before = _text_identity(value[:match.start()])
        after = _text_identity(value[match.end():])
        if len(before) >= 3 or len(after) >= 3:
            return True
    return False


def _dialogue_segments(text: str, pattern: re.Pattern[str]) -> list[tuple[str, str]]:
    """Split only explicit complete quote spans while preserving every character."""
    value = str(text or "")
    segments: list[tuple[str, str]] = []
    cursor = 0
    for match in pattern.finditer(value):
        if match.start() > cursor:
            segments.append(("paragraph", value[cursor:match.start()]))
        segments.append(("dialogue", match.group(0)))
        cursor = match.end()
    if cursor < len(value):
        segments.append(("paragraph", value[cursor:]))
    return [(kind, part) for kind, part in segments if part]


def _translation_needs_repair(record: AITextRecord, target_language: str) -> bool:
    """Conservative deterministic filter for missing / obviously untranslated text.

    This never attempts to judge translation quality.  It only catches output
    that is empty, byte-for-byte the source after whitespace folding, or still
    dominated by Japanese script for targets where that cannot be the intended
    final prose.  Suspect blocks are repaired with a text-only request so no page
    image needs to be uploaded again.
    """
    source = _text_identity(record.source)
    translated = _text_identity(record.translation)
    if not source or not translated:
        return True
    if source == translated:
        return True
    compact = translated
    if target_language in {"zh-Hans", "zh-Hant"}:
        kana = len(_JP_KANA_RE.findall(compact))
        return kana >= 3 and kana / max(1, len(compact)) >= 0.12
    if target_language.lower().startswith("en"):
        jp = len(_JP_KANA_RE.findall(compact)) + len(_CJK_RE.findall(compact))
        return jp >= 4 and jp / max(1, len(compact)) >= 0.20
    return False


def _merge_usage(*snapshots: Mapping | None) -> dict:
    merged: dict[str, int] = {}
    for snapshot in snapshots:
        for key, value in dict(snapshot or {}).items():
            try:
                merged[key] = merged.get(key, 0) + max(0, int(value or 0))
            except Exception:
                continue
    return merged

def _validated_page_payloads(payload: dict, expected_pages: set[int]) -> dict[int, dict]:
    """Require one explicit result object for every image sent in a batch.

    An eligible page may legitimately have no body text, but the model must
    still return ``b=[]`` for it.  Silently accepting a missing page would turn
    an API formatting mistake into an undetectable book omission.
    """
    rows = payload.get("p", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        raise ValueError("AI 返回缺少 p 页面数组")
    result: dict[int, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            page_no = int(row.get("n"))
        except Exception:
            continue
        if page_no not in expected_pages:
            continue
        if page_no in result:
            raise ValueError(f"AI 重复返回第 {page_no} 页")
        blocks = row.get("b", [])
        if not isinstance(blocks, list):
            raise ValueError(f"AI 第 {page_no} 页的 b 不是数组")
        result[page_no] = row
    missing = sorted(expected_pages - set(result))
    if missing:
        raise ValueError("AI 漏回页面：" + ", ".join(str(x) for x in missing))
    return result


class AIImageBookProcessor:
    def __init__(
        self,
        settings: AISettings,
        options: AIImageOptions | None = None,
        *,
        client_factory: Callable[[AISettings], MultimodalClient] = MultimodalClient,
        text_provider_factory: Callable[[AISettings], object] = create_provider,
    ):
        self.settings = settings
        self.options = options or AIImageOptions()
        self.client_factory = client_factory
        self.text_provider_factory = text_provider_factory
        self.cache_root = Path(os.environ.get(
            "NOVEL_FORMATTER_AI_IMAGE_CACHE_DIR",
            str(CONFIG_PATH.parent / "ai_image_cache"),
        )).expanduser()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        (self.cache_root / "images").mkdir(exist_ok=True)
        (self.cache_root / "responses").mkdir(exist_ok=True)
        (self.cache_root / "reviews").mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Image preparation
    # ------------------------------------------------------------------
    @staticmethod
    def _conservative_content_crop(image: Image.Image) -> tuple[int, int, int, int]:
        """Remove only obvious white margins, capped at 12% per side."""
        width, height = image.size
        if width < 64 or height < 64:
            return (0, 0, width, height)
        thumb = image.convert("L")
        scale = min(1.0, 900.0 / max(width, height))
        if scale < 1.0:
            thumb = thumb.resize((max(1, round(width * scale)), max(1, round(height * scale))))
        # Pixels darker than 246 are conservatively treated as content.
        mask = thumb.point(lambda p: 255 if p < 246 else 0)
        box = mask.getbbox()
        if not box:
            return (0, 0, width, height)
        sx = width / thumb.width; sy = height / thumb.height
        left = int(box[0] * sx); top = int(box[1] * sy)
        right = int(math.ceil(box[2] * sx)); bottom = int(math.ceil(box[3] * sy))
        pad_x = max(12, int(width * 0.015)); pad_y = max(12, int(height * 0.015))
        left = max(0, left - pad_x); top = max(0, top - pad_y)
        right = min(width, right + pad_x); bottom = min(height, bottom + pad_y)
        max_x = int(width * 0.12); max_y = int(height * 0.12)
        left = min(left, max_x); top = min(top, max_y)
        right = max(right, width - max_x); bottom = max(bottom, height - max_y)
        if (right - left) * (bottom - top) < width * height * 0.62:
            return (0, 0, width, height)
        return (left, top, right, bottom)

    @staticmethod
    def _source_content_bounds(image: Image.Image) -> tuple[int, int, int, int] | None:
        """Return visible-ink bounds without ever cropping direct API input."""
        width, height = image.size
        if width < 32 or height < 32:
            return None
        thumb = image.convert("L")
        scale = min(1.0, 900.0 / max(width, height))
        if scale < 1.0:
            thumb = thumb.resize((max(1, round(width * scale)), max(1, round(height * scale))))
        mask = thumb.point(lambda px: 255 if px < 242 else 0)
        box = mask.getbbox()
        if not box:
            return None
        sx = width / thumb.width; sy = height / thumb.height
        return (
            max(0, int(box[0] * sx)), max(0, int(box[1] * sy)),
            min(width, int(math.ceil(box[2] * sx))), min(height, int(math.ceil(box[3] * sy))),
        )

    @classmethod
    def _source_edge_risk(cls, image: Image.Image) -> tuple[bool, str]:
        bounds = cls._source_content_bounds(image)
        if bounds is None:
            return False, ""
        width, height = image.size
        _left, top, right, _bottom = bounds
        risks = []
        if width and right >= width * 0.985:
            risks.append("right_edge")
        if height and top <= height * 0.015:
            risks.append("top_edge")
        return bool(risks), "+".join(risks)

    @staticmethod
    def _vertical_layout_hint_from_image(image: Image.Image) -> dict:
        """Return conservative geometry-only hints for clean vertical novel pages.

        This is *not* OCR. It only estimates main vertical text columns and
        first-line indentation from dark-pixel geometry. Ruby-sized side runs are
        filtered by width. The result is advisory unless a strong two-level top
        pattern exists, so illustrated or irregular pages simply return
        ``usable=False`` and continue through the normal multimodal path.
        """
        try:
            import numpy as np
        except Exception:
            return {"usable": False, "reason": "numpy_unavailable"}
        try:
            gray = image.convert("L")
            width, height = gray.size
            if width < 320 or height < 480:
                return {"usable": False, "reason": "page_too_small"}
            # Keep analysis inexpensive while preserving vertical column geometry.
            scale = min(1.0, 1800.0 / max(width, height))
            if scale < 1.0:
                gray = gray.resize((max(1, round(width * scale)), max(1, round(height * scale))))
            arr = np.asarray(gray, dtype=np.uint8)
            # Require a predominantly light page; irregular art pages should not
            # manufacture paragraph anchors from image texture.
            if float(np.median(arr)) < 238.0:
                return {"usable": False, "reason": "non_plain_background"}
            ink = arr < 205
            if int(ink.sum()) < max(120, int(arr.size * 0.001)):
                return {"usable": False, "reason": "too_little_ink"}
            xproj = ink.sum(axis=0).astype(float)
            smooth = np.convolve(xproj, np.ones(5, dtype=float), mode="same")
            threshold = max(12.0, arr.shape[0] * 0.020)
            bands = []
            start = None
            for x, value in enumerate(smooth):
                if value > threshold and start is None:
                    start = x
                elif value <= threshold and start is not None:
                    end = x - 1
                    if end - start + 1 >= 4:
                        bands.append((start, end))
                    start = None
            if start is not None:
                bands.append((start, len(smooth) - 1))
            if len(bands) < 3:
                return {"usable": False, "reason": "no_column_bands"}
            raw_widths = [b - a + 1 for a, b in bands]
            median_width = float(np.median(raw_widths))
            min_main_width = max(9.0, median_width * 0.55)
            main = []
            for left, right in bands:
                band_width = right - left + 1
                if band_width < min_main_width:
                    continue
                sub = ink[:, left:right + 1]
                ys = np.where(sub)[0]
                if ys.size == 0:
                    continue
                top = int(ys.min()); bottom = int(ys.max())
                if bottom - top < max(55, band_width * 2.4):
                    continue
                main.append((left, right, top, bottom))
            if len(main) < 3:
                return {"usable": False, "reason": "too_few_main_columns"}
            # Physical Japanese vertical reading order is right -> left.
            main.sort(key=lambda item: -((item[0] + item[1]) / 2.0))
            widths = [right - left + 1 for left, right, _top, _bottom in main]
            median_main_width = float(np.median(widths))
            tops = [top for _left, _right, top, _bottom in main]
            baseline = float(min(tops))
            indent_threshold = max(10.0, median_main_width * 0.55)
            deltas = [max(0.0, float(top) - baseline) for top in tops]
            # Only expose paragraph starts when there is a real second top level.
            # Otherwise all columns may simply share the same top margin.
            strong_indent = sum(delta >= indent_threshold for delta in deltas)
            continuation = sum(delta <= median_main_width * 0.25 for delta in deltas)
            indent_evidence = strong_indent >= 2 and continuation >= 1
            starts = [
                index + 1 for index, delta in enumerate(deltas)
                if indent_evidence and delta >= indent_threshold
            ]
            return {
                "usable": True,
                "main_columns": len(main),
                "paragraph_start_columns": starts,
                "indent_evidence": bool(indent_evidence),
                "top_norm": [round(top / max(1, arr.shape[0]), 4) for top in tops],
                "bottom_norm": [round(bottom / max(1, arr.shape[0]), 4) for _l, _r, _t, bottom in main],
                "median_column_width_norm": round(median_main_width / max(1, arr.shape[1]), 5),
            }
        except Exception:
            return {"usable": False, "reason": "geometry_analysis_failed"}

    def _vertical_layout_hint(self, prepared: PreparedPage) -> dict:
        if not self.options.layout_geometry_guard:
            return {"usable": False, "reason": "disabled"}
        try:
            with Image.open(prepared.source_path) as src:
                src.load()
                image = ImageOps.exif_transpose(src).convert("RGB")
            return self._vertical_layout_hint_from_image(image)
        except Exception:
            return {"usable": False, "reason": "image_open_failed"}

    def _prepare_page(self, page_no: int, path: str, page_type: str) -> PreparedPage:
        source_hash = _sha256_file(path)
        image_format = str(self.options.prepared_image_format or "webp").strip().lower()
        if image_format not in {"webp", "png"}:
            image_format = "webp"
        key = _sha256_bytes(
            f"{PIPELINE_VERSION}|{source_hash}|{self.options.max_edge}|{int(self.options.trim_white_margins)}|"
            f"{int(self.options.direct_full_page_api)}|{int(self.options.direct_original_when_safe)}|"
            f"{image_format}|{int(self.options.prepared_webp_quality)}".encode()
        )
        target = self.cache_root / "images" / f"{key}.{'webp' if image_format == 'webp' else 'png'}"
        meta_path = target.with_suffix(".json")
        if self.options.use_cache and target.exists() and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if str(meta.get("source_sha256") or "") != source_hash:
                    raise ValueError("cached source fingerprint mismatch")
                return PreparedPage(
                    page_no, path, str(target), tuple(meta["source_size"]),
                    tuple(meta["source_crop"]), page_type, source_hash,
                )
            except Exception:
                pass
        with Image.open(path) as src:
            src.load()
            image = ImageOps.exif_transpose(src).convert("RGB")
        source_size = image.size
        full_crop = (0, 0, *source_size)
        crop = full_crop if self.options.direct_full_page_api else (
            self._conservative_content_crop(image) if self.options.trim_white_margins else full_crop
        )
        if crop != full_crop:
            image = image.crop(crop)
        max_edge = max(image.size)
        source_suffix = Path(path).suffix.lower()
        can_send_original = (
            self.options.direct_full_page_api
            and self.options.direct_original_when_safe
            and crop == full_crop
            and max_edge <= self.options.max_edge
            and source_suffix in {".png", ".jpg", ".jpeg", ".webp"}
        )
        if can_send_original:
            return PreparedPage(page_no, path, path, source_size, full_crop, page_type, source_hash)
        if max_edge > self.options.max_edge:
            ratio = self.options.max_edge / max_edge
            image = image.resize((max(1, round(image.width * ratio)), max(1, round(image.height * ratio))), Image.Resampling.LANCZOS)
        target.parent.mkdir(parents=True, exist_ok=True)
        if image_format == "webp":
            try:
                image.save(
                    target, format="WEBP", quality=max(60, min(100, int(self.options.prepared_webp_quality))),
                    method=4, exact=True,
                )
            except Exception:
                target = target.with_suffix(".png")
                image.save(target, format="PNG", optimize=False)
        else:
            image.save(target, format="PNG", optimize=False)
        meta_path.write_text(json.dumps({
            "source_size": source_size, "source_crop": crop, "source_sha256": source_hash,
            "direct_full_page_api": bool(self.options.direct_full_page_api),
        }), encoding="utf-8")
        return PreparedPage(page_no, path, str(target), source_size, crop, page_type, source_hash)

    def _assert_source_unchanged(self, prepared: PreparedPage) -> None:
        if not self.options.verify_source_fingerprint:
            return
        current = _sha256_file(prepared.source_path)
        if current != prepared.source_sha256:
            raise RuntimeError(
                f"第 {prepared.page_no} 页源图在 AI 任务运行期间发生变化；已停止回写，避免把旧结果绑定到新页面。"
            )

    def _prepare_review_crop(self, prepared: PreparedPage, bbox: tuple[float, float, float, float], index: int) -> str:
        self._assert_source_unchanged(prepared)
        x, y, w, h = bbox
        source_w, source_h = prepared.source_size
        base_l, base_t, base_r, base_b = prepared.source_crop
        base_w = base_r - base_l; base_h = base_b - base_t
        l = base_l + x * base_w; t = base_t + y * base_h
        r = l + w * base_w; b = t + h * base_h
        # Small context around the model-requested uncertainty; never send the
        # full page a second time merely because one glyph is unclear.
        pad_x = max(12.0, (r - l) * 0.30)
        pad_y = max(12.0, (b - t) * 0.18)
        box = (
            max(0, int(l - pad_x)), max(0, int(t - pad_y)),
            min(source_w, int(math.ceil(r + pad_x))), min(source_h, int(math.ceil(b + pad_y))),
        )
        key = _sha256_bytes(
            f"{prepared.source_sha256}|{box}|{self.options.review_max_edge}|{index}".encode()
        )
        target = self.cache_root / "reviews" / f"{key}.webp"
        if self.options.use_cache and target.exists():
            return str(target)
        with Image.open(prepared.source_path) as src:
            src.load()
            image = ImageOps.exif_transpose(src).convert("RGB").crop(box)
        if max(image.size) > self.options.review_max_edge:
            ratio = self.options.review_max_edge / max(image.size)
            image = image.resize((max(1, round(image.width * ratio)), max(1, round(image.height * ratio))), Image.Resampling.LANCZOS)
        try:
            image.save(target, format="WEBP", quality=92, method=4, exact=True)
        except Exception:
            target = target.with_suffix(".png")
            image.save(target, format="PNG", optimize=False)
        return str(target)

    def _prepare_page_rescue(self, prepared: PreparedPage) -> str:
        """Render one eligible page at review resolution for a single rescue pass.

        This is intentionally used only when the first pass returned *no* body
        text for a page that Page Manager admitted as text.  It avoids a blanket
        second high-resolution pass over the whole book while protecting against
        silent omissions caused by tiny type, bad down-scaling, or a transient
        model miss.
        """
        self._assert_source_unchanged(prepared)
        key = _sha256_bytes(
            f"rescue|{prepared.source_sha256}|{prepared.source_crop}|{self.options.review_max_edge}".encode()
        )
        target = self.cache_root / "reviews" / f"page-{key}.webp"
        if self.options.use_cache and target.exists():
            return str(target)
        with Image.open(prepared.source_path) as src:
            src.load()
            image = ImageOps.exif_transpose(src).convert("RGB").crop(prepared.source_crop)
        if max(image.size) > self.options.review_max_edge:
            ratio = self.options.review_max_edge / max(image.size)
            image = image.resize(
                (max(1, round(image.width * ratio)), max(1, round(image.height * ratio))),
                Image.Resampling.LANCZOS,
            )
        try:
            image.save(target, format="WEBP", quality=92, method=4, exact=True)
        except Exception:
            target = target.with_suffix(".png")
            image.save(target, format="PNG", optimize=False)
        return str(target)

    def _rescue_prompt(self, prepared: PreparedPage) -> str:
        translate_rule = (
            f"同时给出{self.options.target_language}译文；译文忠实自然，不省略、不添加。"
            if self.options.translate else "不要翻译，译文字段留空。"
        )
        return (
            "这是一次遗漏保护复核。页面管理已确认该页进入正文处理，但普通清晰度首轮没有返回正文。"
            "请用当前更高清整页重新核对：只转录实际可见的日文正文，按真实阅读顺序恢复段落、对白、章/节标题；"
            "振假名只用于辨字，不重复成正文；页码和跑题页眉不输出；不得猜写、润色或总结。"
            + translate_rule +
            "如果确实没有任何应进入正文的文字，也必须明确返回空 b。严格只返回JSON："
            '{"p":[{"n":页号,"b":[["paragraph|dialogue|chapter|section|footnote","原文","译文","high|medium|low"]],"v":[]}],"u":[]}。'
            f"P=[[{prepared.page_no},{json.dumps(prepared.page_type, ensure_ascii=False)}]]"
        )

    # ------------------------------------------------------------------
    # Prompt / cache
    # ------------------------------------------------------------------
    def _first_prompt(
        self,
        batch: list[PreparedPage],
        context_tail: list[AITextRecord],
        layout_hints: Mapping[int, Mapping] | None = None,
    ) -> str:
        pages = [[p.page_no, p.page_type] for p in batch]
        context = [[r.block_id, r.source, r.translation] for r in context_tail]
        hints = {
            str(p.page_no): dict((layout_hints or {}).get(p.page_no) or {})
            for p in batch
            if bool(dict((layout_hints or {}).get(p.page_no) or {}).get("usable"))
        }
        translate_rule = (
            f"同时给出{self.options.target_language}译文；译文忠实自然，不省略、不添加。"
            if self.options.translate else "不要翻译，译文字段留空。"
        )
        return (
            "你在直接读取日文书籍的完整物理页面。页面管理给出的页号/页型是事实，不要重新分类。"
            "每张输入图都必须先检查物理页面最右侧；竖排严格按右→左列、列内上→下。先确认最右一列顶端没有漏字，再继续下一列。"
            "按日文真实阅读顺序精确转录可读正文。若图像本身在页边截断字符，绝对不要凭语法、人名或上下文补写图外字符；相关块设为low并给v。"
            "振假名只作辨字依据，不要作为独立正文重复输出；页码/跑题页眉不输出。"
            "最重要：b不是物理竖列，也不是一句一块；每个b必须对应原书版面中的一个真实排版段落、独立对白段、章标题或小节。"
            "看到首字缩进、段间留白、独立起段的「……」/『……』对白或标题时必须分块；同一自然段跨越多根竖列时必须保持为同一个b。"
            "禁止把多个可见自然段压成一个大块，也禁止按每个句号机械拆段。"
            "原文字段只抄图上实际字形：不得按语法补助词/语尾、不得规范化措辞、不得润色、猜写、总结；翻译不得反过来影响原文转录。"
            "如果H提供了几何提示：main_columns是本地仅按墨迹几何检测到的主竖列数，paragraph_start_columns是存在明确首行缩进的列号（1=最右列）。"
            "这些提示不包含OCR文字，但有indent_evidence=true时必须优先按这些可见缩进恢复换段；不能把这些段落边界按语义重新解释掉。"
            + translate_rule +
            "看不清时保留最可信文本，并只为确实需要放大的局部给v；v坐标是当前输入图的0~1归一化[x,y,w,h]。"
            "视觉首轮按批次独立并行处理，C通常为空；跨页接续由程序后续用纯文本检查统一处理。u必须返回空数组。"
            "严格只返回JSON。紧凑schema："
            '{"p":[{"n":页号,"b":[["paragraph|dialogue|chapter|section|footnote","原文","译文","high|medium|low"]],'
            '"v":[[块序号,x,y,w,h,"原因"]]}],"u":[["旧block_id","修正原文","修正译文"]]}。'
            f"P={json.dumps(pages, ensure_ascii=False, separators=(',', ':'))};"
            f"C={json.dumps(context, ensure_ascii=False, separators=(',', ':'))};"
            f"H={json.dumps(hints, ensure_ascii=False, separators=(',', ':'))}"
        )

    def _review_prompt(self, regions: list[AIReviewRegion], record_map: dict[str, AITextRecord]) -> str:
        items = []
        for idx, region in enumerate(regions, start=1):
            record = record_map[region.block_id]
            items.append([idx, region.block_id, record.source, record.translation, region.reason])
        translate_rule = (
            f"若原文改变，同时给出对应{self.options.target_language}译文。"
            if self.options.translate else "译文字段留空。"
        )
        return (
            "每张图片依次对应I中的同序号疑难局部。只核对图中实际字形，不润色、不补写。"
            + translate_rule +
            "严格JSON：{\"r\":[[\"block_id\",\"正确原文\",\"正确译文\"]]}。"
            f"I={json.dumps(items, ensure_ascii=False, separators=(',', ':'))}"
        )

    def _translation_repair_prompt(self, records: list[AITextRecord]) -> str:
        items = [[record.block_id, record.source] for record in records]
        return (
            f"把以下日文轻小说正文忠实翻译为{self.options.target_language}。只翻译给定文本，不补写、不省略、不解释。"
            "严格只返回JSON：{\"r\":[[\"block_id\",\"译文\"]]}；每个输入ID必须且只能出现一次。"
            f"I={json.dumps(items, ensure_ascii=False, separators=(',', ':'))}"
        )

    def _records_from_page_payload(self, page_payload: Mapping, page_no: int) -> list[AITextRecord]:
        records: list[AITextRecord] = []
        blocks = page_payload.get("b", []) if isinstance(page_payload, Mapping) else []
        if not isinstance(blocks, list):
            return records
        for local_index, row in enumerate(blocks):
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            source = _safe_text(row[1])
            if not source:
                continue
            translation = _safe_text(row[2]) if len(row) >= 3 and self.options.translate else ""
            confidence = _safe_text(row[3]) if len(row) >= 4 else "high"
            records.append(AITextRecord(
                block_id=f"ai-p{page_no:05d}-b{local_index + 1:04d}",
                page_no=page_no,
                block_type=_record_type(row[0], source),
                source=source,
                translation=translation,
                confidence=confidence if confidence in {"high", "medium", "low"} else "medium",
                source_sha256=_sha256_bytes(source.encode("utf-8")),
            ))
        return records

    def _structure_anomaly_candidates(
        self,
        page_results: Mapping[int, list[AITextRecord]],
        layout_hints: Mapping[int, Mapping] | None = None,
    ) -> list[tuple[int, str, int, int]]:
        """Return pages likely flattened into too few semantic blocks.

        The first pass is deliberately fast and provider-neutral. Some vision
        models read the Japanese accurately but collapse an entire light-novel
        page into one paragraph.  We only re-upload pages with strong signals:
        embedded full dialogue inside narration, a very long single block, or a
        long paragraph containing many sentence endings while the page has almost
        no block boundaries.  Ordinary pages never pay for this second visual pass.
        """
        if not self.options.audit_flat_layout_pages or self.options.quality == "fast":
            return []
        candidates: list[tuple[int, str, int, int]] = []
        for page_no in sorted(page_results):
            records = list(page_results.get(page_no) or [])
            if not records:
                continue
            if any(record.block_type in {"chapter", "section"} for record in records):
                # Chapter/opening pages legitimately contain very uneven blocks.
                continue
            total_chars = _records_char_count(records)
            reasons: list[str] = []
            for record in records:
                text = str(record.source or "")
                char_count = len(_text_identity(text))
                if record.block_type == "paragraph" and char_count >= 90 and _has_embedded_dialogue(text):
                    reasons.append("narration_and_dialogue_collapsed")
            if len(records) == 1 and total_chars >= 320:
                reasons.append("single_full_page_block")
            hint = dict((layout_hints or {}).get(page_no) or {})
            if bool(hint.get("usable")) and bool(hint.get("indent_evidence")):
                starts = [int(x) for x in (hint.get("paragraph_start_columns") or []) if str(x).isdigit()]
                expected = len(starts)
                # A page may begin with a continuation from the preceding page, so
                # expected semantic blocks are usually starts or starts+1. Leave a
                # small tolerance for quote punctuation / ornamental initials.
                if expected >= 3 and (len(records) < max(1, expected - 1) or len(records) > expected + 3):
                    reasons.insert(0, f"layout_geometry_mismatch:starts={expected},blocks={len(records)},columns={int(hint.get('main_columns', 0) or 0)}")
            if reasons:
                candidates.append((page_no, reasons[0], total_chars, len(records)))
        return candidates

    def _structure_audit_prompt(
        self,
        prepared: PreparedPage,
        records: list[AITextRecord],
        reason: str,
        layout_hint: Mapping | None = None,
    ) -> str:
        first_blocks = [[r.block_type, r.source, r.translation] for r in records]
        hint = dict(layout_hint or {})
        translate_rule = (
            f"每个新块同时给出{self.options.target_language}译文；译文必须与该块一一对应，不省略、不添加。"
            if self.options.translate else "不要翻译，译文字段留空。"
        )
        return (
            "这是一次日文轻小说高清版面结构 + OCR忠实度审计。首轮文字可能基本正确，但可能把多个排版段落压成一个大块，"
            "也可能存在少量相似假名/助词误读。请以当前高清原页为唯一依据重新读取整页。"
            "每个b只能代表原书一个真实排版段落：paragraph=叙述自然段，dialogue=版面上独立起段的对白，chapter/section=标题。"
            "判断换段必须看原页的首字缩进、段间空白、对白是否独立起段和标题位置；不要把每个句号拆成一段，也不要把每根竖列当一段。"
            "同一自然段跨多根竖列时合并成同一个b；独立对白与前后叙述不可粘在同一个b。"
            "原文必须逐字忠实于图像，不允许为了语法顺畅而补字、加助词、改语尾、统一表达或润色。"
            "若某字确实看不清，将该块置信度设为low/medium并在v中给出需要放大的区域；不要猜。"
            "若H.usable=true：还必须返回c，c是从最右到最左每一根主竖列的逐列原文，长度必须等于H.main_columns；振假名不进入c。"
            "把c按顺序无空格拼接后，必须与b中原文按顺序无空格拼接后逐字一致；这用于证明没有漏掉整列。"
            "H.indent_evidence=true时，H.paragraph_start_columns表示可见首行缩进位置，b的段落边界必须与这些列起点一致；不得按语义擅自合并。"
            + translate_rule +
            "必须返回该页完整正文而不是差异。严格只返回JSON："
            '{"p":[{"n":页号,"b":[["paragraph|dialogue|chapter|section|footnote","原文","译文","high|medium|low"]],'
            '"c":["最右主列原文","下一主列原文"],"v":[[块序号,x,y,w,h,"原因"]]}],"u":[]}。'
            f"P=[[{prepared.page_no},{json.dumps(prepared.page_type, ensure_ascii=False)}]];"
            f"R={json.dumps(reason, ensure_ascii=False)};"
            f"F={json.dumps(first_blocks, ensure_ascii=False, separators=(',', ':'))};"
            f"H={json.dumps(hint, ensure_ascii=False, separators=(',', ':'))}"
        )

    @staticmethod
    def _layout_audit_consistency(
        page_payload: Mapping,
        records: list[AITextRecord],
        layout_hint: Mapping | None,
    ) -> tuple[bool, dict]:
        hint = dict(layout_hint or {})
        if not bool(hint.get("usable")):
            return True, {"verified": False, "reason": "no_geometry_hint"}
        expected_columns = int(hint.get("main_columns", 0) or 0)
        columns = page_payload.get("c", []) if isinstance(page_payload, Mapping) else []
        if expected_columns < 3:
            return True, {"verified": False, "reason": "too_few_columns"}
        if not isinstance(columns, list) or len(columns) != expected_columns:
            return False, {"verified": False, "reason": "column_count_mismatch", "expected_columns": expected_columns, "returned_columns": len(columns) if isinstance(columns, list) else -1}
        column_texts = [_safe_text(value) for value in columns]
        if any(not text for text in column_texts):
            return False, {"verified": False, "reason": "empty_physical_column"}
        column_join = _text_identity("".join(column_texts))
        block_join = _text_identity(_records_text(records))
        if not column_join or not block_join:
            return False, {"verified": False, "reason": "empty_joined_text"}
        similarity = difflib.SequenceMatcher(None, column_join, block_join, autojunk=False).ratio()
        col_cov = _source_overlap_ratio(column_join, block_join)
        block_cov = _source_overlap_ratio(block_join, column_join)
        if similarity < 0.965 or col_cov < 0.965 or block_cov < 0.965:
            return False, {"verified": False, "reason": "column_block_text_mismatch", "similarity": similarity, "column_coverage": col_cov, "block_coverage": block_cov}
        matched = total = 0
        if bool(hint.get("indent_evidence")):
            starts = sorted({int(x) for x in (hint.get("paragraph_start_columns") or []) if str(x).isdigit() and 1 <= int(x) <= expected_columns})
            # Column 1 corresponds to text position 0, which is always a block
            # start and therefore does not test a boundary.
            expected_positions = []
            running = 0
            for idx, text in enumerate(column_texts, start=1):
                if idx in starts and idx > 1:
                    expected_positions.append(running)
                running += len(_text_identity(text))
            observed = []
            running = 0
            for record in records[:-1]:
                running += len(_text_identity(record.source))
                observed.append(running)
            tolerance = max(2, round((running / max(1, len(column_texts))) * 0.12))
            total = len(expected_positions)
            for position in expected_positions:
                if any(abs(position - candidate) <= tolerance for candidate in observed):
                    matched += 1
            if total >= 2 and matched / total < 0.72:
                return False, {"verified": False, "reason": "paragraph_boundaries_mismatch", "boundary_matches": matched, "boundary_total": total, "tolerance": tolerance}
        return True, {"verified": True, "reason": "ok", "similarity": similarity, "column_coverage": col_cov, "block_coverage": block_cov, "boundary_matches": matched, "boundary_total": total}

    @staticmethod
    def _structure_replacement_is_safe(
        old_records: list[AITextRecord],
        new_records: list[AITextRecord],
        *,
        layout_verified: bool = False,
    ) -> tuple[bool, float, float, float]:
        if not old_records or not new_records:
            return False, 0.0, 0.0, 0.0
        old_text = _records_text(old_records)
        new_text = _records_text(new_records)
        old_count = _records_char_count(old_records)
        new_count = _records_char_count(new_records)
        if old_count <= 0 or new_count <= 0:
            return False, 0.0, 0.0, 0.0
        length_ratio = new_count / old_count
        old_coverage = _source_overlap_ratio(old_text, new_text)
        new_coverage = _source_overlap_ratio(new_text, old_text)
        if layout_verified:
            # Geometry-verified column coverage may legitimately recover a whole
            # omitted column from the first pass. Preserve most old evidence but
            # allow a fuller second transcription.
            if not (0.80 <= length_ratio <= 1.35):
                return False, old_coverage, new_coverage, length_ratio
            if old_coverage < 0.78 or new_coverage < 0.60:
                return False, old_coverage, new_coverage, length_ratio
        else:
            if not (0.84 <= length_ratio <= 1.16):
                return False, old_coverage, new_coverage, length_ratio
            if old_coverage < 0.80 or new_coverage < 0.78:
                return False, old_coverage, new_coverage, length_ratio
        old_types = [r.block_type for r in old_records]
        new_types = [r.block_type for r in new_records]
        structure_improved = len(new_records) > len(old_records) or new_types != old_types
        similarity = difflib.SequenceMatcher(
            None, _text_identity(old_text), _text_identity(new_text), autojunk=False
        ).ratio()
        # A same-shape replacement is accepted only for a very small OCR repair.
        # This lets the audit fix e.g. a missing kana without turning the stage
        # into an unconstrained second transcription pass.
        tiny_ocr_repair = new_text != old_text and similarity >= 0.93
        return bool(structure_improved or tiny_ocr_repair), old_coverage, new_coverage, length_ratio

    def _split_flattened_dialogue_records(
        self,
        page_results: dict[int, list[AITextRecord]],
        record_map: dict[str, AITextRecord],
        *,
        skip_pages: set[int] | None = None,
    ) -> int:
        """Recover explicit dialogue boundaries without another model request.

        Fast mode intentionally skips the visual structure audit. Some providers
        still return a whole page as one paragraph even when a complete Japanese
        quote is visibly embedded in narration. Splitting that exact quote is
        deterministic and character-preserving. In translation mode we only do
        it when source and translation have the same paragraph/dialogue shape,
        so a translation can never be attached to the wrong source block.
        """
        split_count = 0
        skip = set(skip_pages or set())
        for page_no in sorted(page_results):
            if page_no in skip:
                continue
            updated: list[AITextRecord] = []
            for record in page_results.get(page_no) or []:
                if record.block_type != "paragraph":
                    updated.append(record)
                    continue
                source_segments = _dialogue_segments(record.source, _DIALOGUE_SPAN_RE)
                source_shape = [kind for kind, _part in source_segments]
                if "dialogue" not in source_shape or "paragraph" not in source_shape:
                    updated.append(record)
                    continue

                translation_segments: list[tuple[str, str]] = []
                if self.options.translate and str(record.translation or ""):
                    translation_segments = _dialogue_segments(
                        record.translation, _TRANSLATED_DIALOGUE_SPAN_RE
                    )
                    if [kind for kind, _part in translation_segments] != source_shape:
                        updated.append(record)
                        continue

                record_map.pop(record.block_id, None)
                for index, (block_type, source) in enumerate(source_segments, start=1):
                    translation = (
                        translation_segments[index - 1][1]
                        if translation_segments else ""
                    )
                    child = AITextRecord(
                        block_id=f"{record.block_id}-s{index:02d}",
                        page_no=record.page_no,
                        block_type=block_type,
                        source=source,
                        translation=translation,
                        confidence=record.confidence,
                        source_sha256=_sha256_bytes(source.encode("utf-8")),
                        merged_from=tuple(record.merged_from),
                    )
                    updated.append(child)
                    record_map[child.block_id] = child
                split_count += 1
            page_results[page_no] = updated
        return split_count

    def _edge_integrity_candidates(
        self,
        prepared: Mapping[int, PreparedPage],
        page_results: Mapping[int, list[AITextRecord]],
    ) -> list[tuple[int, str]]:
        if not self.options.edge_integrity_guard or self.options.quality == "fast":
            return []
        candidates: list[tuple[int, str]] = []
        for page_no in sorted(page_results):
            item = prepared.get(page_no)
            records = list(page_results.get(page_no) or [])
            if item is None or not records:
                continue
            try:
                with Image.open(item.source_path) as src:
                    src.load()
                    image = ImageOps.exif_transpose(src).convert("RGB")
                risky, reason = self._source_edge_risk(image)
            except Exception:
                continue
            if risky:
                candidates.append((page_no, reason))
        return candidates[:max(0, int(self.options.max_edge_integrity_audits))]

    def _edge_integrity_prompt(
        self, prepared: PreparedPage, records: list[AITextRecord], reason: str,
    ) -> str:
        first_blocks = [[r.block_type, r.source, r.translation] for r in records]
        translate_rule = (
            f"每个块同时给出{self.options.target_language}译文；译文与原文块一一对应。"
            if self.options.translate else "不要翻译，译文字段留空。"
        )
        return (
            "这是日文竖排轻小说的原页边缘完整性复核。当前输入是完整原页/更高清原页，不是正文裁剪图。"
            "必须从物理页面最右侧开始检查，再按右→左列、列内上→下读取；特别检查最右一列顶端，不能从第二列开始。"
            "首轮F可能漏掉最右列、第一句或页顶几个字，也可能完全正确。请以图像为最高证据重新返回整页全部正文。"
            "如果图像本身真的在页边把字符裁掉，只能转录仍然可见的字符并将相关块设为low；绝对禁止根据人名、语法或上下文补写图外字符。"
            "保留原书真实段落、独立对白和标题；不要按物理竖列拆块，也不要把整页压成一个块。"
            "振假名只用于辨字，页码/跑题页眉不输出。原文不得润色、规范化或总结。"
            + translate_rule +
            "必须返回完整页面而不是差异。严格只返回JSON："
            '{"p":[{"n":页号,"b":[["paragraph|dialogue|chapter|section|footnote","原文","译文","high|medium|low"]],"v":[]}],"u":[]}。'
            f"P=[[{prepared.page_no},{json.dumps(prepared.page_type, ensure_ascii=False)}]];"
            f"R={json.dumps(reason, ensure_ascii=False)};"
            f"F={json.dumps(first_blocks, ensure_ascii=False, separators=(',', ':'))}"
        )

    @staticmethod
    def _edge_replacement_is_safe(
        old_records: list[AITextRecord], new_records: list[AITextRecord],
    ) -> tuple[bool, float, float, float]:
        if not old_records or not new_records:
            return False, 0.0, 0.0, 0.0
        old_text = _records_text(old_records); new_text = _records_text(new_records)
        old_count = _records_char_count(old_records); new_count = _records_char_count(new_records)
        if old_count <= 0 or new_count <= 0:
            return False, 0.0, 0.0, 0.0
        length_ratio = new_count / old_count
        old_coverage = _source_overlap_ratio(old_text, new_text)
        new_coverage = _source_overlap_ratio(new_text, old_text)
        safe = 0.90 <= length_ratio <= 1.35 and old_coverage >= 0.88 and new_coverage >= 0.68
        return bool(safe), old_coverage, new_coverage, length_ratio

    def _short_page_candidates(
        self,
        page_results: Mapping[int, list[AITextRecord]],
    ) -> list[tuple[int, int, int]]:
        """Find non-empty pages whose textual yield is an extreme local outlier.

        This is intentionally neighbour-based instead of using a fixed characters
        per page threshold.  Chapter/opening pages are skipped, and both adjacent
        pages must contain substantial prose.  The result is only a *review queue*;
        a high-resolution rescue still has to preserve most of the first-pass text
        before it can replace anything.
        """
        if not self.options.audit_suspicious_short_pages or self.options.quality == "fast":
            return []
        candidates: list[tuple[int, int, int]] = []
        pages = sorted(page_results)
        page_set = set(pages)
        for page_no in pages:
            if page_no - 1 not in page_set or page_no + 1 not in page_set:
                continue
            records = page_results.get(page_no) or []
            left = page_results.get(page_no - 1) or []
            right = page_results.get(page_no + 1) or []
            if not records or not left or not right:
                continue
            if any(record.block_type in {"chapter", "section"} for record in records):
                continue
            current = _records_char_count(records)
            left_count = _records_char_count(left)
            right_count = _records_char_count(right)
            baseline = min(left_count, right_count)
            # Require two ordinary, text-heavy neighbours.  A page below roughly
            # one third of both neighbours is suspicious but not automatically bad.
            if baseline < 120 or current < 8:
                continue
            if current <= min(120, max(28, int(baseline * 0.34))):
                candidates.append((page_no, current, baseline))
        return candidates[:max(0, int(self.options.max_short_page_audits))]

    def _short_page_prompt(self, prepared: PreparedPage, records: list[AITextRecord]) -> str:
        first_text = _records_text(records)
        translate_rule = (
            f"同时给出{self.options.target_language}译文；译文忠实自然，不省略、不添加。"
            if self.options.translate else "不要翻译，译文字段留空。"
        )
        return (
            "这是一次非空页面的漏文审计。首轮已经识别出一些正文，但相邻正文页的文本量明显更多。"
            "请重新阅读当前高清整页，返回该页全部应进入电子书的日文正文，而不是只补差异。"
            "按真实阅读顺序恢复段落/对白/章或节标题；振假名只用于辨字；页码和跑题页眉不输出；"
            "不得根据首轮文本续写、猜写或总结。首轮文本F只用于确认不要漏掉已经看见的内容。"
            + translate_rule +
            "如果首轮其实已经完整，原样返回即可。严格只返回JSON："
            '{"p":[{"n":页号,"b":[["paragraph|dialogue|chapter|section|footnote","原文","译文","high|medium|low"]],"v":[]}],"u":[]}。'
            f"P=[[{prepared.page_no},{json.dumps(prepared.page_type, ensure_ascii=False)}]];"
            f"F={json.dumps(first_text, ensure_ascii=False)}"
        )

    @staticmethod
    def _continuity_candidates(
        page_results: Mapping[int, list[AITextRecord]],
    ) -> list[tuple[AITextRecord, AITextRecord]]:
        pairs: list[tuple[AITextRecord, AITextRecord]] = []
        pages = sorted(page_results)
        for left_page, right_page in zip(pages, pages[1:]):
            if right_page != left_page + 1:
                continue
            left_rows = page_results.get(left_page) or []
            right_rows = page_results.get(right_page) or []
            if not left_rows or not right_rows:
                continue
            left, right = left_rows[-1], right_rows[0]
            if left.block_type != "paragraph" or right.block_type != "paragraph":
                continue
            a = str(left.source or "").rstrip()
            b = str(right.source or "").lstrip()
            if len(_text_identity(a)) < 4 or len(_text_identity(b)) < 4:
                continue
            if _text_identity(a) == _text_identity(b):
                continue
            if _STRONG_PARAGRAPH_END_RE.search(a):
                continue
            pairs.append((left, right))
        return pairs

    def _continuity_prompt(self, pairs: list[tuple[AITextRecord, AITextRecord]]) -> str:
        items = [[left.block_id, left.source, right.block_id, right.source] for left, right in pairs]
        return (
            "检查以下日文轻小说相邻物理页的页尾/页首正文是否只是同一段在换页处被切开。"
            "只判断结构，不改写任何字。只有语法和语义明确连续时才merge；新的段落、场景转折或不确定时keep。"
            "严格只返回JSON：{\"r\":[[\"left_id\",\"right_id\",\"merge|keep\"]]}；"
            "每个输入边界必须且只能出现一次。"
            f"I={json.dumps(items, ensure_ascii=False, separators=(',', ':'))}"
        )

    def _repair_cross_page_continuity(
        self,
        page_results: dict[int, list[AITextRecord]],
        record_map: dict[str, AITextRecord],
        *,
        cancel_event: threading.Event | None,
        log: Callable[[str], None],
    ) -> tuple[int, int, set[int], dict]:
        if not self.options.continuity_text_check or self.options.quality == "fast":
            return 0, 0, set(), {}
        candidates = self._continuity_candidates(page_results)[:max(0, int(self.options.max_continuity_checks))]
        if not candidates:
            return 0, 0, set(), {}
        checked = 0
        merged = 0
        consumed_pages: set[int] = set()
        try:
            provider_context = self.text_provider_factory(self.settings)
        except Exception as exc:
            log(f"跨页接续文本检查不可用，保留原分段且不重复上传图片：{exc}")
            return 0, 0, set(), {}
        try:
            with provider_context as provider:
                queue = [candidates[i:i + 20] for i in range(0, len(candidates), 20)]
                while queue:
                    if cancel_event and cancel_event.is_set():
                        raise RuntimeError("AI 图文处理已停止")
                    batch = queue.pop(0)
                    expected = {(left.block_id, right.block_id): (left, right) for left, right in batch}
                    log(f"仅用文本检查 {len(batch)} 个跨页接续候选（不上传图片）…")
                    try:
                        payload = _json_object(provider.call_json(self._continuity_prompt(batch), 0.0))
                        rows = payload.get("r", [])
                        if not isinstance(rows, list):
                            raise ValueError("跨页接续返回缺少 r 数组")
                        decisions: dict[tuple[str, str], str] = {}
                        for row in rows:
                            if not isinstance(row, (list, tuple)) or len(row) < 3:
                                continue
                            key = (_safe_text(row[0]), _safe_text(row[1]))
                            action = _safe_text(row[2]).lower()
                            if key not in expected or key in decisions or action not in {"merge", "keep"}:
                                continue
                            decisions[key] = action
                        if set(decisions) != set(expected):
                            raise ValueError("跨页接续漏回、重复或返回未知 block_id")
                    except Exception as exc:
                        if len(batch) > 1:
                            mid = max(1, len(batch) // 2)
                            queue[:0] = [batch[:mid], batch[mid:]]
                            log(f"跨页接续批次结构异常，自动拆分重试：{exc}")
                            continue
                        log(f"跨页接续未能确认 {batch[0][0].block_id}/{batch[0][1].block_id}：{exc}")
                        continue

                    checked += len(batch)
                    for key, action in decisions.items():
                        if action != "merge":
                            continue
                        left, right = expected[key]
                        # Re-read current rows in case an earlier merge in this same
                        # run removed the target.  Never resurrect stale objects.
                        if record_map.get(left.block_id) is not left or record_map.get(right.block_id) is not right:
                            continue
                        left.source = str(left.source or "").rstrip() + str(right.source or "").lstrip()
                        left.source_sha256 = _sha256_bytes(left.source.encode("utf-8"))
                        left.merged_from = tuple(left.merged_from) + (right.block_id,) + tuple(right.merged_from)
                        if self.options.translate:
                            # A paragraph-level merge changes translation context.
                            # Clear it and let the existing text-only repair stage
                            # translate the final merged source exactly once.
                            left.translation = ""
                        confidence_rank = {"low": 0, "medium": 1, "high": 2}
                        left.confidence = min(
                            (left.confidence, right.confidence),
                            key=lambda x: confidence_rank.get(x, 1),
                        )
                        page_results[right.page_no] = [
                            row for row in (page_results.get(right.page_no) or [])
                            if row.block_id != right.block_id
                        ]
                        record_map.pop(right.block_id, None)
                        if not page_results.get(right.page_no):
                            consumed_pages.add(right.page_no)
                        merged += 1
                usage = provider.usage_snapshot() if hasattr(provider, "usage_snapshot") else {}
        except RuntimeError:
            if cancel_event and cancel_event.is_set():
                raise
            log("跨页接续文本检查运行失败，保留原分段且不重复上传图片。")
            return checked, merged, consumed_pages, {}
        except Exception as exc:
            log(f"跨页接续文本检查不可用，保留尚未处理的原分段：{exc}")
            return checked, merged, consumed_pages, {}
        return checked, merged, consumed_pages, dict(usage or {})

    def _repair_translations(
        self,
        page_results: Mapping[int, list[AITextRecord]],
        *,
        cancel_event: threading.Event | None,
        log: Callable[[str], None],
    ) -> tuple[int, int, dict]:
        if not self.options.translate or not self.options.translation_repair:
            return 0, 0, {}
        candidates = [
            record for page_no in sorted(page_results) for record in page_results[page_no]
            if _translation_needs_repair(record, self.options.target_language)
        ]
        candidates = candidates[:max(0, int(self.options.max_translation_repairs))]
        if not candidates:
            return 0, 0, {}
        repaired = 0
        try:
            provider_context = self.text_provider_factory(self.settings)
        except Exception as exc:
            log(f"文本补译不可用，保留日文原文并阻止未完成译文直接导出：{exc}")
            return len(candidates), 0, {}
        try:
            with provider_context as provider:
                queue = [candidates[i:i + 24] for i in range(0, len(candidates), 24)]
                while queue:
                    if cancel_event and cancel_event.is_set():
                        raise RuntimeError("AI 图文处理已停止")
                    batch = queue.pop(0)
                    expected = {record.block_id: record for record in batch}
                    prompt = self._translation_repair_prompt(batch)
                    log(f"仅用文本补译 {len(batch)} 个缺失/疑似未翻译块（不重复上传图片）…")
                    try:
                        raw = provider.call_json(prompt, 0.0)
                        payload = _json_object(raw)
                        rows = payload.get("r", [])
                        if not isinstance(rows, list):
                            raise ValueError("文本补译返回缺少 r 数组")
                        seen: set[str] = set()
                        updates: list[tuple[AITextRecord, str]] = []
                        for row in rows:
                            if not isinstance(row, (list, tuple)) or len(row) < 2:
                                continue
                            block_id = _safe_text(row[0]); translated = _safe_text(row[1])
                            if block_id not in expected or block_id in seen or not translated:
                                continue
                            seen.add(block_id); updates.append((expected[block_id], translated))
                        if seen != set(expected):
                            raise ValueError("文本补译漏回或重复 block_id")
                        for record, translated in updates:
                            record.translation = translated
                            repaired += 1
                    except Exception as exc:
                        if len(batch) > 1:
                            mid = max(1, len(batch) // 2)
                            queue[:0] = [batch[:mid], batch[mid:]]
                            log(f"文本补译批次结构异常，自动拆分重试：{exc}")
                            continue
                        log(f"文本补译仍未完成 {batch[0].block_id}：{exc}")
                usage = provider.usage_snapshot() if hasattr(provider, "usage_snapshot") else {}
        except RuntimeError:
            if cancel_event and cancel_event.is_set():
                raise
            log("文本补译运行失败，保留日文原文并阻止未完成译文直接导出。")
            return len(candidates), repaired, {}
        except Exception as exc:
            log(f"文本补译不可用，保留日文原文并阻止未完成译文直接导出：{exc}")
            return len(candidates), repaired, {}
        return len(candidates), repaired, dict(usage or {})

    @staticmethod
    def _boundary_duplicate_candidates(page_results: Mapping[int, list[AITextRecord]]) -> list[tuple[AITextRecord, AITextRecord]]:
        pairs: list[tuple[AITextRecord, AITextRecord]] = []
        pages = sorted(page_results)
        for left_page, right_page in zip(pages, pages[1:]):
            if right_page != left_page + 1:
                continue
            left = page_results.get(left_page) or []
            right = page_results.get(right_page) or []
            if not left or not right:
                continue
            a, b = left[-1], right[0]
            key = _text_identity(a.source)
            if len(key) >= 8 and key == _text_identity(b.source):
                pairs.append((a, b))
        return pairs

    def _boundary_prompt(self, left: AITextRecord, right: AITextRecord) -> str:
        return (
            "这两张图片是相邻书页。AI在前页末块与后页首块提取了完全相同的正文。"
            "请只依据两页可见字形判断这段正文实际出现在哪一页。不要改写文本。"
            "严格JSON：{\"a\":\"keep_both|drop_left|drop_right\"}。"
            f"L=[{left.page_no},{json.dumps(left.source, ensure_ascii=False)}];"
            f"R=[{right.page_no},{json.dumps(right.source, ensure_ascii=False)}]"
        )

    def _resolve_visual_concurrency(self) -> tuple[int, bool]:
        explicit = int(self.options.visual_concurrency or 0) or int(getattr(self.settings, "concurrency", 0) or 0)
        if explicit > 0:
            return max(1, min(32, explicit)), False
        provider = str(getattr(self.settings, "provider", "") or "").strip().lower()
        # Local/custom endpoints are conservative by default because capability
        # and server limits are unknown. Users can explicitly raise concurrency.
        if provider in {"ollama", "custom"}:
            return 1, False
        initial = max(1, int(self.options.initial_auto_visual_concurrency or 4))
        maximum = max(initial, min(32, int(self.options.max_auto_visual_concurrency or 16)))
        return min(initial, maximum), True

    @staticmethod
    def _usage_delta(before: Mapping | None, after: Mapping | None, key: str) -> int:
        try:
            return max(0, int(dict(after or {}).get(key, 0) or 0) - int(dict(before or {}).get(key, 0) or 0))
        except Exception:
            return 0

    def _cached_call(
        self, client: MultimodalClient, prompt: str, images: list[str], *,
        namespace: str, reasoning_effort: str | None = None,
    ) -> tuple[str, bool]:
        image_hashes = [_sha256_file(path) for path in images]
        identity = {
            "v": PIPELINE_VERSION,
            "provider": self.settings.provider,
            "model": self.settings.model,
            "base": self.settings.base_url,
            "prompt": prompt,
            "images": image_hashes,
            "reasoning_effort": reasoning_effort or "",
        }
        key = _sha256_bytes(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        path = self.cache_root / "responses" / f"{namespace}-{key}.json"
        if self.options.use_cache and path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and isinstance(payload.get("raw"), str):
                    return payload["raw"], True
            except Exception:
                pass
        if reasoning_effort:
            try:
                raw = client.call_json(prompt, images, temperature=0.0, reasoning_effort=reasoning_effort)
            except TypeError as exc:
                # Test doubles and third-party custom clients may still implement
                # the pre-v2.2 signature. Only fall back for that exact contract.
                if "reasoning_effort" not in str(exc):
                    raise
                raw = client.call_json(prompt, images, temperature=0.0)
        else:
            raw = client.call_json(prompt, images, temperature=0.0)
        if self.options.use_cache:
            path.write_text(json.dumps({"raw": raw}, ensure_ascii=False), encoding="utf-8")
        return raw, False

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------
    def run(
        self,
        page_images: Iterable[str | Path],
        confirmed_page_types: Mapping[int | str, object] | None = None,
        *,
        cancel_event: threading.Event | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
    ) -> AIImageResult:
        images = [str(Path(x)) for x in page_images]
        if not images:
            raise ValueError("页面管理中没有可处理页面")
        overrides = {int(k): page_type_value(v) for k, v in (confirmed_page_types or {}).items()}
        eligible = []
        skipped = []
        for page_no, path in enumerate(images, start=1):
            page_type = overrides.get(page_no, BlockType.PARAGRAPH.value)
            if _should_skip_ai_page(page_type):
                skipped.append((page_no, page_type))
            else:
                eligible.append((page_no, path, page_type or BlockType.PARAGRAPH.value))
        if not eligible:
            raise ValueError("页面管理中没有正文/未分类页面可供 AI 处理")

        def log(message: str):
            if log_callback:
                log_callback(message)

        prepared: dict[int, PreparedPage] = {}
        prepared_lock = threading.Lock()
        layout_hints: dict[int, dict] = {}
        layout_lock = threading.Lock()

        def ensure_prepared(row: tuple[int, str, str]) -> PreparedPage:
            page_no, path, page_type = row
            with prepared_lock:
                existing = prepared.get(page_no)
            if existing is not None:
                with layout_lock:
                    have_hint = page_no in layout_hints
                if not have_hint:
                    hint = self._vertical_layout_hint(existing)
                    with layout_lock:
                        layout_hints.setdefault(page_no, hint)
                return existing
            if cancel_event and cancel_event.is_set():
                raise RuntimeError("AI 图文处理已停止")
            item = self._prepare_page(page_no, path, page_type)
            hint = self._vertical_layout_hint(item)
            with layout_lock:
                layout_hints.setdefault(page_no, hint)
            with prepared_lock:
                prepared.setdefault(page_no, item)
                return prepared[page_no]

        page_results: dict[int, list[AITextRecord]] = {}
        record_map: dict[str, AITextRecord] = {}
        review_regions: list[AIReviewRegion] = []
        cache_hits = 0
        # Do not put pages separated by a preserved cover/illustration into the
        # same image request.  First-pass visual requests are independent and may
        # run concurrently; cross-page continuity is repaired later using text only.
        batch_limit = 1 if self.options.direct_full_page_api else max(1, int(self.options.batch_pages))
        groups: list[list[tuple[int, str, str]]] = []
        current: list[tuple[int, str, str]] = []
        for row in eligible:
            if current and (row[0] != current[-1][0] + 1 or len(current) >= batch_limit):
                groups.append(current); current = []
            current.append(row)
        if current:
            groups.append(current)
        completed_pages = 0
        page_rescue_requested = 0
        page_rescue_accepted = 0
        batch_splits = 0
        rejected_context_updates = 0
        boundary_checks = 0
        boundary_deduped = 0
        short_page_audit_requested = 0
        short_page_audit_accepted = 0
        structure_audit_requested = 0
        structure_audit_accepted = 0
        structure_audit_rejected = 0
        structure_audit_deferred = 0
        structure_unresolved_pages: set[int] = set()
        edge_integrity_requested = 0
        edge_integrity_accepted = 0
        edge_integrity_rejected = 0
        edge_unresolved_pages: set[int] = set()
        visual_usage: dict = {}
        visual_concurrency, auto_visual_concurrency = self._resolve_visual_concurrency()
        visual_concurrency_initial = visual_concurrency
        visual_concurrency_peak = visual_concurrency
        visual_parallel_waves = 0
        max_auto_visual_concurrency = max(
            visual_concurrency, min(32, int(self.options.max_auto_visual_concurrency or 16))
        )

        with self.client_factory(self.settings) as client:
            def process_group(rows: list[tuple[int, str, str]]):
                if cancel_event and cancel_event.is_set():
                    raise RuntimeError("AI 图文处理已停止")
                batch = [ensure_prepared(row) for row in rows]
                prompt = self._first_prompt(batch, [], layout_hints)
                page_label = str(rows[0][0]) if len(rows) == 1 else f"{rows[0][0]}–{rows[-1][0]}"
                log(f"AI 正在处理第 {page_label} 页…")
                image_paths = [p.prepared_path for p in batch]
                local_hits = 0
                local_splits = 0
                raw, hit = self._cached_call(client, prompt, image_paths, namespace="page")
                local_hits += int(hit)
                known_pages = {p.page_no for p in batch}
                try:
                    payload = _json_object(raw)
                    page_payload_map = _validated_page_payloads(payload, known_pages)
                    return [(rows, batch, payload, page_payload_map)], local_hits, local_splits
                except Exception as first_error:
                    retry_prompt = (
                        prompt
                        + "\n上一响应结构不完整。必须为P中的每个页面各返回且只返回一个p条目；"
                          "即使该页无正文也返回b=[]。不得漏页。"
                    )
                    log(f"AI 返回结构不完整，正在重试本批次：{first_error}")
                    raw, retry_hit = self._cached_call(client, retry_prompt, image_paths, namespace="page-retry")
                    local_hits += int(retry_hit)
                    try:
                        payload = _json_object(raw)
                        page_payload_map = _validated_page_payloads(payload, known_pages)
                        return [(rows, batch, payload, page_payload_map)], local_hits, local_splits
                    except Exception as retry_error:
                        if self.options.adaptive_batch_split and len(rows) > 1:
                            mid = max(1, len(rows) // 2)
                            local_splits += 1
                            log(f"本批次连续返回不完整，自动拆为更小批次继续，不丢页：{retry_error}")
                            left, left_hits, left_splits = process_group(rows[:mid])
                            right, right_hits, right_splits = process_group(rows[mid:])
                            return (
                                [*left, *right],
                                local_hits + left_hits + right_hits,
                                local_splits + left_splits + right_splits,
                            )
                        raise

            pending_groups = list(groups)
            while pending_groups:
                if cancel_event and cancel_event.is_set():
                    raise RuntimeError("AI 图文处理已停止")
                wave_size = max(1, min(visual_concurrency, len(pending_groups)))
                wave = pending_groups[:wave_size]
                del pending_groups[:wave_size]
                visual_parallel_waves += 1
                if wave_size > 1:
                    log(f"视觉首轮并行处理：{wave_size} 个批次同时进行（当前并发 {visual_concurrency}）…")
                before_usage = client.usage_snapshot() if hasattr(client, "usage_snapshot") else {}
                wave_outputs = []
                with ThreadPoolExecutor(max_workers=wave_size, thread_name_prefix="nf-ai-vision") as executor:
                    futures = {executor.submit(process_group, rows): rows for rows in wave}
                    for future in as_completed(futures):
                        outputs, local_hits, local_splits = future.result()
                        wave_outputs.extend(outputs)
                        cache_hits += local_hits
                        batch_splits += local_splits
                        finished = sum(len(item[0]) for item in outputs)
                        completed_pages += finished
                        if progress_callback:
                            progress_callback(completed_pages, len(eligible))
                after_usage = client.usage_snapshot() if hasattr(client, "usage_snapshot") else {}
                retries_in_wave = self._usage_delta(before_usage, after_usage, "transport_retries")
                if auto_visual_concurrency and pending_groups:
                    previous = visual_concurrency
                    if retries_in_wave > 0:
                        visual_concurrency = max(1, visual_concurrency // 2)
                    else:
                        visual_concurrency = min(max_auto_visual_concurrency, visual_concurrency + 1)
                    visual_concurrency_peak = max(visual_concurrency_peak, visual_concurrency)
                    if visual_concurrency != previous:
                        reason = "检测到网络/限流重试，自动降并发" if retries_in_wave > 0 else "本轮稳定，自动升并发"
                        log(f"{reason}：{previous} → {visual_concurrency}")

                # Apply results in physical page order even though requests finished
                # out of order. This keeps block IDs, logs and EPUB output stable.
                for rows, batch, payload, page_payload_map in sorted(wave_outputs, key=lambda item: item[0][0][0]):
                    known_pages = {p.page_no for p in batch}
                    for page_payload in page_payload_map.values():
                        if not isinstance(page_payload, dict):
                            continue
                        try:
                            page_no = int(page_payload.get("n"))
                        except Exception:
                            continue
                        if page_no not in known_pages:
                            continue
                        records: list[AITextRecord] = []
                        blocks = page_payload.get("b", [])
                        if isinstance(blocks, list):
                            for local_index, row in enumerate(blocks):
                                if not isinstance(row, (list, tuple)) or len(row) < 2:
                                    continue
                                source = _safe_text(row[1])
                                if not source:
                                    continue
                                translation = _safe_text(row[2]) if len(row) >= 3 and self.options.translate else ""
                                confidence = _safe_text(row[3]) if len(row) >= 4 else "high"
                                block_id = f"ai-p{page_no:05d}-b{local_index + 1:04d}"
                                record = AITextRecord(
                                    block_id=block_id,
                                    page_no=page_no,
                                    block_type=_record_type(row[0], source),
                                    source=source,
                                    translation=translation,
                                    confidence=confidence if confidence in {"high", "medium", "low"} else "medium",
                                    source_sha256=_sha256_bytes(source.encode("utf-8")),
                                )
                                records.append(record); record_map[block_id] = record
                        page_results[page_no] = records

                        regions = page_payload.get("v", [])
                        if isinstance(regions, list):
                            for item in regions:
                                if not isinstance(item, (list, tuple)) or len(item) < 5:
                                    continue
                                try:
                                    block_index = int(item[0])
                                except Exception:
                                    continue
                                if 1 <= block_index <= len(records):
                                    record = records[block_index - 1]
                                elif 0 <= block_index < len(records):
                                    record = records[block_index]
                                else:
                                    continue
                                box = _bbox(item[1:5])
                                if not box:
                                    continue
                                review_regions.append(AIReviewRegion(
                                    page_no=page_no,
                                    block_id=record.block_id,
                                    bbox=box,
                                    reason=_safe_text(item[5]) if len(item) > 5 else "uncertain",
                                ))

                    # First-pass batches are intentionally context-independent.
                    # Reject any unsolicited u update rather than letting a parallel
                    # request mutate a previously completed page.
                    updates = payload.get("u", [])
                    if isinstance(updates, list):
                        rejected_context_updates += sum(
                            1 for row in updates if isinstance(row, (list, tuple)) and len(row) >= 2
                        )

            visual_usage = client.usage_snapshot()

            # Omission guard: only pages that produced zero blocks get one
            # higher-resolution full-page rescue.  This is far cheaper than a
            # second pass over every page and closes the most dangerous failure
            # mode for an AI-direct workflow: a silently dropped page.
            if self.options.quality != "fast":
                empty_pages = [
                    p for p in prepared.values()
                    if not page_results.get(p.page_no)
                    and page_type_value(p.page_type) in {
                        BlockType.PARAGRAPH.value, BlockType.AFTERWORD.value, BlockType.UNKNOWN.value,
                    }
                ]
                for prepared_page in empty_pages:
                    if cancel_event and cancel_event.is_set():
                        raise RuntimeError("AI 图文处理已停止")
                    page_rescue_requested += 1
                    rescue_image = self._prepare_page_rescue(prepared_page)
                    prompt = self._rescue_prompt(prepared_page)
                    log(f"第 {prepared_page.page_no} 页首轮无正文，正在进行一次高清遗漏保护复核…")
                    raw, hit = self._cached_call(client, prompt, [rescue_image], namespace="page-rescue")
                    cache_hits += int(hit)
                    payload = _json_object(raw)
                    page_payload = _validated_page_payloads(payload, {prepared_page.page_no})[prepared_page.page_no]
                    replacement: list[AITextRecord] = []
                    for local_index, row in enumerate(page_payload.get("b", [])):
                        if not isinstance(row, (list, tuple)) or len(row) < 2:
                            continue
                        source = _safe_text(row[1])
                        if not source:
                            continue
                        translation = _safe_text(row[2]) if len(row) >= 3 and self.options.translate else ""
                        confidence = _safe_text(row[3]) if len(row) >= 4 else "high"
                        block_id = f"ai-p{prepared_page.page_no:05d}-b{local_index + 1:04d}"
                        record = AITextRecord(
                            block_id=block_id, page_no=prepared_page.page_no,
                            block_type=_record_type(row[0], source), source=source, translation=translation,
                            confidence=confidence if confidence in {"high", "medium", "low"} else "medium",
                            source_sha256=_sha256_bytes(source.encode("utf-8")),
                        )
                        replacement.append(record)
                    if replacement:
                        page_results[prepared_page.page_no] = replacement
                        for record in replacement:
                            record_map[record.block_id] = record
                        page_rescue_accepted += 1

            # Some multimodal models read all glyphs correctly but flatten a
            # whole vertical light-novel page into one giant paragraph.  Audit
            # only pages with strong flattening signals.  The replacement must
            # remain character-overlap compatible with the first pass, so this
            # cannot silently become a free-form rewrite.
            structure_candidates = self._structure_anomaly_candidates(page_results, layout_hints)
            structure_budget = max(
                0,
                max(
                    int(self.options.max_structure_audits or 0),
                    int(self.options.max_layout_geometry_audits or 0),
                ),
            )
            for candidate_index, (page_no, reason, first_chars, first_blocks) in enumerate(structure_candidates):
                if candidate_index >= structure_budget:
                    structure_audit_deferred += 1
                    structure_unresolved_pages.add(page_no)
                    log(
                        f"第 {page_no} 页仍存在版面结构异常，但已达到本轮高清结构复核预算；"
                        "保留首轮并标记为未解决，不会静默当成出版成稿。"
                    )
                    continue
                if cancel_event and cancel_event.is_set():
                    raise RuntimeError("AI 图文处理已停止")
                prepared_page = prepared.get(page_no)
                old_records = list(page_results.get(page_no) or [])
                if prepared_page is None or not old_records:
                    continue
                structure_audit_requested += 1
                rescue_image = self._prepare_page_rescue(prepared_page)
                hint = dict(layout_hints.get(page_no) or {})
                prompt = self._structure_audit_prompt(prepared_page, old_records, reason, hint)
                log(
                    f"第 {page_no} 页疑似段落被压平（{first_blocks} 块 / {first_chars} 字），"
                    "正在按高清原页复核真实换段与字形…"
                )
                try:
                    audit_effort = "high" if (self.options.audit_reasoning_high and str(self.settings.provider or "").lower() in {"zhipu", "zai"}) else None
                    raw, hit = self._cached_call(
                        client, prompt, [rescue_image], namespace="page-structure-audit",
                        reasoning_effort=audit_effort,
                    )
                    cache_hits += int(hit)
                    payload = _json_object(raw)
                    page_payload = _validated_page_payloads(payload, {page_no})[page_no]
                    replacement_records = self._records_from_page_payload(page_payload, page_no)
                except Exception as exc:
                    structure_audit_rejected += 1
                    structure_unresolved_pages.add(page_no)
                    log(f"第 {page_no} 页结构复核返回异常，保留首轮结果：{exc}")
                    continue
                layout_ok, layout_report = self._layout_audit_consistency(page_payload, replacement_records, hint)
                if not layout_ok:
                    structure_audit_rejected += 1
                    structure_unresolved_pages.add(page_no)
                    log(f"第 {page_no} 页高清结构复核未通过物理列/缩进一致性校验，保留首轮：{layout_report.get('reason')}。")
                    continue
                safe, old_cov, new_cov, length_ratio = self._structure_replacement_is_safe(
                    old_records, replacement_records, layout_verified=bool(layout_report.get("verified")),
                )
                if not safe:
                    structure_audit_rejected += 1
                    structure_unresolved_pages.add(page_no)
                    log(
                        f"第 {page_no} 页结构复核未通过防改写门槛，保留首轮："
                        f"首轮覆盖 {old_cov:.0%} / 复核覆盖 {new_cov:.0%} / 长度比 {length_ratio:.2f}。"
                    )
                    continue
                for old in old_records:
                    record_map.pop(old.block_id, None)
                page_results[page_no] = replacement_records
                for record in replacement_records:
                    record_map[record.block_id] = record
                review_regions[:] = [region for region in review_regions if region.page_no != page_no]
                # Preserve uncertainty regions returned by the structure audit;
                # their coordinates refer to the full rescue page just supplied.
                regions = page_payload.get("v", [])
                if isinstance(regions, list):
                    for item in regions:
                        if not isinstance(item, (list, tuple)) or len(item) < 5:
                            continue
                        try:
                            block_index = int(item[0])
                        except Exception:
                            continue
                        if 1 <= block_index <= len(replacement_records):
                            record = replacement_records[block_index - 1]
                        elif 0 <= block_index < len(replacement_records):
                            record = replacement_records[block_index]
                        else:
                            continue
                        box = _bbox(item[1:5])
                        if box:
                            review_regions.append(AIReviewRegion(
                                page_no=page_no,
                                block_id=record.block_id,
                                bbox=box,
                                reason=_safe_text(item[5]) if len(item) > 5 else "structure_audit_uncertain",
                            ))
                structure_audit_accepted += 1
                if self._structure_anomaly_candidates({page_no: replacement_records}, {page_no: hint}):
                    structure_unresolved_pages.add(page_no)
                    log(
                        f"第 {page_no} 页已采用高清字形复核，但段落压平信号仍存在；"
                        "均衡模式会警告，出版模式将阻止直接导出。"
                    )
                else:
                    structure_unresolved_pages.discard(page_no)
                boundary_note = ""
                if layout_report.get("verified"):
                    boundary_note = f"；物理列校验通过，缩进边界 {layout_report.get('boundary_matches', 0)}/{layout_report.get('boundary_total', 0)}"
                log(
                    f"第 {page_no} 页已采用高清结构复核：{len(old_records)}→{len(replacement_records)} 块；"
                    f"文字覆盖 {old_cov:.0%}/{new_cov:.0%}{boundary_note}。"
                )

            # Direct full-page mode already avoids local margin clipping.  If the
            # source ink itself reaches the physical top/right edge, verify the
            # first vertical column once at review resolution.
            for page_no, reason in self._edge_integrity_candidates(prepared, page_results):
                if cancel_event and cancel_event.is_set():
                    raise RuntimeError("AI 图文处理已停止")
                prepared_page = prepared.get(page_no)
                old_records = list(page_results.get(page_no) or [])
                if prepared_page is None or not old_records:
                    continue
                edge_integrity_requested += 1
                rescue_image = self._prepare_page_rescue(prepared_page)
                prompt = self._edge_integrity_prompt(prepared_page, old_records, reason)
                log(f"第 {page_no} 页正文接近物理页顶/右边缘，正在核对最右列与第一句是否完整…")
                try:
                    raw, hit = self._cached_call(client, prompt, [rescue_image], namespace="edge-integrity")
                    cache_hits += int(hit)
                    payload = _json_object(raw)
                    page_payload = _validated_page_payloads(payload, {page_no})[page_no]
                    replacement_records = self._records_from_page_payload(page_payload, page_no)
                except Exception as exc:
                    edge_integrity_rejected += 1
                    edge_unresolved_pages.add(page_no)
                    log(f"第 {page_no} 页边缘完整性复核失败，保留首轮并标记：{exc}")
                    continue
                safe, old_cov, new_cov, length_ratio = self._edge_replacement_is_safe(old_records, replacement_records)
                if not safe:
                    edge_integrity_rejected += 1
                    edge_unresolved_pages.add(page_no)
                    log(
                        f"第 {page_no} 页边缘复核未通过防猜写门槛，保留首轮："
                        f"旧文保留 {old_cov:.0%} / 新文覆盖 {new_cov:.0%} / 长度比 {length_ratio:.2f}。"
                    )
                    continue
                for old_record in old_records:
                    record_map.pop(old_record.block_id, None)
                page_results[page_no] = replacement_records
                for record in replacement_records:
                    record_map[record.block_id] = record
                review_regions[:] = [region for region in review_regions if region.page_no != page_no]
                edge_integrity_accepted += 1
                edge_unresolved_pages.discard(page_no)
                log(
                    f"第 {page_no} 页已采用完整原页边缘复核：{len(old_records)}→{len(replacement_records)} 块；"
                    f"文字覆盖 {old_cov:.0%}/{new_cov:.0%}。"
                )

            # A non-empty page can still be catastrophically incomplete.  Audit
            # only strong local text-yield outliers, at most a small capped set,
            # and accept a fuller-page result only when it preserves most of the
            # first-pass text while adding substantial new content.  This borrows
            # the "retry only suspicious work units" idea from mature image/LLM
            # translators without paying for a blanket second pass.
            for page_no, first_chars, neighbour_floor in self._short_page_candidates(page_results):
                if cancel_event and cancel_event.is_set():
                    raise RuntimeError("AI 图文处理已停止")
                prepared_page = prepared.get(page_no)
                old_records = list(page_results.get(page_no) or [])
                if prepared_page is None or not old_records:
                    continue
                short_page_audit_requested += 1
                rescue_image = self._prepare_page_rescue(prepared_page)
                prompt = self._short_page_prompt(prepared_page, old_records)
                log(
                    f"第 {page_no} 页首轮已有正文但文本量异常偏少 "
                    f"({first_chars} 字；相邻至少 {neighbour_floor} 字)，正在做一次定向高清漏文审计…"
                )
                try:
                    raw, hit = self._cached_call(client, prompt, [rescue_image], namespace="page-short-rescue")
                    cache_hits += int(hit)
                    payload = _json_object(raw)
                    page_payload = _validated_page_payloads(payload, {page_no})[page_no]
                    replacement_records = self._records_from_page_payload(page_payload, page_no)
                except Exception as exc:
                    log(f"第 {page_no} 页漏文审计返回异常，保留首轮结果：{exc}")
                    continue
                old_text = _records_text(old_records)
                new_text = _records_text(replacement_records)
                old_count = _records_char_count(old_records)
                new_count = _records_char_count(replacement_records)
                coverage = _source_overlap_ratio(old_text, new_text)
                minimum_growth = max(12, int(math.ceil(old_count * 0.25)))
                if new_count < old_count + minimum_growth or coverage < 0.68:
                    log(
                        f"第 {page_no} 页高清结果未满足安全替换门槛，保留首轮："
                        f"{old_count}→{new_count} 字，首轮覆盖 {coverage:.0%}"
                    )
                    continue
                for old in old_records:
                    record_map.pop(old.block_id, None)
                page_results[page_no] = replacement_records
                for record in replacement_records:
                    record_map[record.block_id] = record
                review_regions[:] = [region for region in review_regions if region.page_no != page_no]
                short_page_audit_accepted += 1
                log(f"第 {page_no} 页确认首轮漏文，安全采用高清完整结果：{old_count}→{new_count} 字。")

            reviewed = 0
            if self.options.auto_review and review_regions:
                limited = review_regions[:max(0, int(self.options.max_visual_reviews))]
                for group_start in range(0, len(limited), 4):
                    if cancel_event and cancel_event.is_set():
                        raise RuntimeError("AI 图文处理已停止")
                    group = limited[group_start:group_start + 4]
                    crop_paths = [
                        self._prepare_review_crop(prepared[region.page_no], region.bbox, group_start + i)
                        for i, region in enumerate(group)
                    ]
                    prompt = self._review_prompt(group, record_map)
                    log(f"视觉复核疑难区域 {group_start + 1}–{min(group_start + len(group), len(limited))} / {len(limited)}")
                    raw, hit = self._cached_call(client, prompt, crop_paths, namespace="review")
                    cache_hits += int(hit)
                    payload = _json_object(raw)
                    expected_ids = {region.block_id for region in group}
                    confirmed_ids: set[str] = set()
                    for row in payload.get("r", []) if isinstance(payload.get("r"), list) else []:
                        if not isinstance(row, (list, tuple)) or len(row) < 2:
                            continue
                        block_id = _safe_text(row[0])
                        if block_id not in expected_ids or block_id in confirmed_ids:
                            continue
                        record = record_map.get(block_id)
                        if record is None:
                            continue
                        source = _safe_text(row[1])
                        source_changed = bool(source and source != record.source)
                        if source:
                            record.source = source
                            record.source_sha256 = _sha256_bytes(source.encode("utf-8"))
                            # This result came from the dedicated high-resolution
                            # crop, so visual uncertainty for this block is resolved.
                            record.confidence = "high"
                        if self.options.translate:
                            translated = _safe_text(row[2]) if len(row) >= 3 else ""
                            if translated:
                                record.translation = translated
                            elif source_changed:
                                record.translation = ""
                        confirmed_ids.add(block_id)
                    reviewed += len(confirmed_ids)

            # Exact cross-page duplicates are rare and dangerous in an AI-direct
            # workflow.  Audit only those anomalous boundaries with the two already
            # prepared page images; normal pages spend no extra visual tokens.
            if self.options.quality != "fast":
                for left, right in self._boundary_duplicate_candidates(page_results):
                    if cancel_event and cancel_event.is_set():
                        raise RuntimeError("AI 图文处理已停止")
                    boundary_checks += 1
                    prompt = self._boundary_prompt(left, right)
                    paths = [prepared[left.page_no].prepared_path, prepared[right.page_no].prepared_path]
                    log(f"检测到第 {left.page_no}/{right.page_no} 页边界重复，正在做一次定向视觉核验…")
                    raw, hit = self._cached_call(client, prompt, paths, namespace="boundary")
                    cache_hits += int(hit)
                    try:
                        action = _safe_text(_json_object(raw).get("a"))
                    except Exception:
                        action = "keep_both"
                    target = None
                    if action == "drop_left":
                        target = left
                    elif action == "drop_right":
                        target = right
                    if target is not None:
                        rows_for_page = page_results.get(target.page_no) or []
                        page_results[target.page_no] = [row for row in rows_for_page if row.block_id != target.block_id]
                        record_map.pop(target.block_id, None)
                        boundary_deduped += 1

            visual_usage = client.usage_snapshot()

        local_dialogue_splits = self._split_flattened_dialogue_records(
            page_results, record_map, skip_pages=structure_unresolved_pages | edge_unresolved_pages,
        )
        if local_dialogue_splits:
            for page_no in list(structure_unresolved_pages):
                if not self._structure_anomaly_candidates({page_no: page_results.get(page_no, [])}):
                    structure_unresolved_pages.discard(page_no)
            log(
                f"本地零 Token 段落保护：拆开 {local_dialogue_splits} 个粘连的叙述/独立对白块；"
                "原文与译文字符均未改写。"
            )

        continuity_checks, continuity_merged, continuity_consumed_pages, continuity_usage = self._repair_cross_page_continuity(
            page_results, record_map, cancel_event=cancel_event, log=log,
        )
        translation_repair_requested, translation_repair_accepted, translation_usage = self._repair_translations(
            page_results, cancel_event=cancel_event, log=log,
        )
        usage = _merge_usage(visual_usage, continuity_usage, translation_usage)

        # Final zero-token integrity guard: the source file behind every prepared
        # page must still be the one that produced the AI result.  This catches a
        # late Page Manager replacement even if no visual review happened after it.
        for prepared_page in prepared.values():
            self._assert_source_unchanged(prepared_page)

        page_hashes = {page_no: item.source_sha256 for page_no, item in prepared.items()}
        source_doc, translated_doc = self._build_documents(images, overrides, page_results, page_hashes=page_hashes)
        all_records = [record for records in page_results.values() for record in records]
        low_confidence_blocks = sum(1 for record in all_records if record.confidence == "low")
        medium_confidence_blocks = sum(1 for record in all_records if record.confidence == "medium")
        translation_missing_blocks = (
            sum(1 for record in all_records if not str(record.translation or "").strip())
            if self.options.translate else 0
        )
        unresolved_text_pages = [
            int(row[0]) for row in eligible
            if not page_results.get(int(row[0])) and int(row[0]) not in continuity_consumed_pages
        ]
        deferred_review = max(0, len(review_regions) - reviewed)
        source_export_ready = not unresolved_text_pages
        if self.options.quality == "publication" and (
            low_confidence_blocks or deferred_review or structure_unresolved_pages
        ):
            source_export_ready = False
        translation_export_ready = bool(self.options.translate and source_export_ready and translation_missing_blocks == 0)
        if not source_export_ready:
            preflight_status = "blocked"
        elif low_confidence_blocks or deferred_review or translation_missing_blocks or structure_unresolved_pages:
            preflight_status = "warning"
        else:
            preflight_status = "pass"
        stats = {
            "total_pages": len(images),
            "processed_pages": len(eligible),
            "processed_page_numbers": [int(row[0]) for row in eligible],
            "skipped_asset_pages": len(skipped),
            "text_blocks": sum(len(v) for v in page_results.values()),
            "review_requested": len(review_regions),
            "reviewed_regions": reviewed,
            "page_rescue_requested": page_rescue_requested,
            "page_rescue_accepted": page_rescue_accepted,
            "batch_splits": batch_splits,
            "visual_concurrency_initial": visual_concurrency_initial,
            "visual_concurrency_peak": visual_concurrency_peak,
            "visual_concurrency_final": visual_concurrency,
            "visual_concurrency_auto": auto_visual_concurrency,
            "visual_parallel_waves": visual_parallel_waves,
            "direct_original_pages": sum(1 for p in prepared.values() if p.prepared_path == p.source_path),
            "prepared_image_format": str(self.options.prepared_image_format or "webp").lower(),
            "rejected_context_updates": rejected_context_updates,
            "boundary_checks": boundary_checks,
            "boundary_deduped": boundary_deduped,
            "short_page_audit_requested": short_page_audit_requested,
            "short_page_audit_accepted": short_page_audit_accepted,
            "structure_audit_requested": structure_audit_requested,
            "structure_audit_accepted": structure_audit_accepted,
            "structure_audit_rejected": structure_audit_rejected,
            "structure_audit_deferred": structure_audit_deferred,
            "structure_unresolved_pages": sorted(structure_unresolved_pages),
            "layout_geometry_pages": sum(1 for hint in layout_hints.values() if bool(hint.get("usable"))),
            "layout_indent_evidence_pages": sum(1 for hint in layout_hints.values() if bool(hint.get("indent_evidence"))),
            "local_dialogue_splits": local_dialogue_splits,
            "edge_integrity_requested": edge_integrity_requested,
            "edge_integrity_accepted": edge_integrity_accepted,
            "edge_integrity_rejected": edge_integrity_rejected,
            "edge_unresolved_pages": sorted(edge_unresolved_pages),
            "direct_full_page_api": bool(self.options.direct_full_page_api),
            "one_page_per_visual_request": bool(self.options.direct_full_page_api),
            "continuity_checks": continuity_checks,
            "continuity_merged": continuity_merged,
            "continuity_consumed_pages": sorted(continuity_consumed_pages),
            "translation_repair_requested": translation_repair_requested,
            "translation_repair_accepted": translation_repair_accepted,
            "review_deferred": deferred_review,
            "low_confidence_blocks": low_confidence_blocks,
            "medium_confidence_blocks": medium_confidence_blocks,
            "cache_hits": cache_hits,
            "unresolved_text_pages": unresolved_text_pages,
            "source_export_ready": source_export_ready,
            "translation_export_ready": translation_export_ready,
            "preflight_status": preflight_status,
            "usage": usage,
            "usage_visual": dict(visual_usage or {}),
            "usage_continuity": dict(continuity_usage or {}),
            "usage_translation_repair": dict(translation_usage or {}),
            "provider": self.settings.provider,
            "model": self.settings.model,
            "mode": self.options.mode,
            "quality": self.options.quality,
        }
        source_doc.metadata.ai_image_report = dict(stats)
        if translated_doc is not None:
            missing_count = int((translated_doc.metadata.ai_image_report or {}).get("translation_missing_blocks", 0) or 0)
            translated_doc.metadata.ai_image_report = dict(stats)
            translated_doc.metadata.ai_image_report["translation_missing_blocks"] = missing_count
            stats["translation_missing_blocks"] = missing_count
        return AIImageResult(source_doc, translated_doc, stats, page_results)

    # ------------------------------------------------------------------
    # Document construction
    # ------------------------------------------------------------------
    def _build_documents(
        self,
        images: list[str],
        overrides: Mapping[int, str],
        page_results: Mapping[int, list[AITextRecord]],
        *,
        page_hashes: Mapping[int, str] | None = None,
    ) -> tuple[UnifiedDocument, UnifiedDocument | None]:
        source = UnifiedDocument()
        source.metadata = Metadata(
            language="ja",
            ocr_mode="ja_vertical",
            writing_direction="vertical-rl",
            source_engine="ai_multimodal_image",
            preserve_ocr_layout=True,
            ai_processing_mode="vision_extract",
            ai_layout_locked=True,
            authoritative_text=True,
        )
        source.metadata.ai_image_provider = self.settings.provider
        source.metadata.ai_image_model = self.settings.model
        source.metadata.ai_image_pipeline_version = PIPELINE_VERSION

        translated = None
        if self.options.translate:
            translated = UnifiedDocument()
            translated.metadata = Metadata(
                language=self.options.target_language,
                ocr_mode="zh_hans_horizontal" if self.options.target_language == "zh-Hans" else "translated_horizontal",
                writing_direction="horizontal-tb",
                source_engine="ai_multimodal_translation",
                preserve_ocr_layout=True,
                ai_processing_mode="vision_translate",
                ai_layout_locked=True,
                authoritative_text=True,
            )
            translated.metadata.ai_image_provider = self.settings.provider
            translated.metadata.ai_image_model = self.settings.model
            translated.metadata.ai_image_pipeline_version = PIPELINE_VERSION
            translated.metadata.ai_translation_target = self.options.target_language

        for page_no, path in enumerate(images, start=1):
            page_type_raw = overrides.get(page_no, BlockType.PARAGRAPH.value)
            try:
                page_type = BlockType(page_type_raw)
            except Exception:
                page_type = BlockType.PARAGRAPH
            width = height = 0
            try:
                with Image.open(path) as image:
                    width, height = image.size
            except Exception:
                pass
            info = PageInfo(page_no, page_type, path, width, height, 1.0)
            source.pages.append(info)
            if translated is not None:
                translated.pages.append(PageInfo(page_no, page_type, path, width, height, 1.0))

        chapter_index_source = 0
        chapter_index_translation = 0
        missing_translation_count = 0
        for page_no in sorted(page_results):
            for order_in_page, record in enumerate(page_results[page_no]):
                btype = _TEXT_BLOCK_TYPES.get(record.block_type, BlockType.PARAGRAPH)
                metadata = {
                    "ai_image_processing": True,
                    "ai_confidence": record.confidence,
                    "ai_model": self.settings.model,
                    "source_sha256": record.source_sha256,
                    "source_block_id": record.block_id,
                    "source_page_sha256": str((page_hashes or {}).get(page_no, "") or ""),
                    "merged_source_block_ids": list(record.merged_from),
                }
                source_block = Block(
                    id=record.block_id,
                    type=btype,
                    text=record.source,
                    page=page_no,
                    page_number=page_no,
                    order_in_page=order_in_page,
                    reading_order=len(source.blocks),
                    text_direction="vertical-rl",
                    source_format="ai_multimodal",
                    confidence={"high": 0.98, "medium": 0.85, "low": 0.65}.get(record.confidence, 0.85),
                    metadata=metadata,
                )
                if btype == BlockType.CHAPTER:
                    chapter_index_source += 1
                    source_block.chapter_index = chapter_index_source
                    source.toc.append(TocEntry(record.source, chapter_index_source, len(source.blocks)))
                source.blocks.append(source_block)

                if translated is not None:
                    translation_missing = not bool(str(record.translation or "").strip())
                    text = record.translation or record.source
                    if translation_missing:
                        missing_translation_count += 1
                    translated_id = f"tr-{record.block_id}"
                    tr_metadata = {
                        "ai_image_processing": True,
                        "ai_translation": True,
                        "translation_missing": translation_missing,
                        "source_block_id": record.block_id,
                        "source_sha256": record.source_sha256,
                        "source_page_sha256": str((page_hashes or {}).get(page_no, "") or ""),
                        "merged_source_block_ids": list(record.merged_from),
                        "ai_model": self.settings.model,
                    }
                    tr_block = Block(
                        id=translated_id,
                        type=btype,
                        text=text,
                        page=page_no,
                        page_number=page_no,
                        order_in_page=order_in_page,
                        reading_order=len(translated.blocks),
                        text_direction="horizontal-tb",
                        source_format="ai_multimodal_translation",
                        confidence=source_block.confidence,
                        metadata=tr_metadata,
                    )
                    if btype == BlockType.CHAPTER:
                        chapter_index_translation += 1
                        tr_block.chapter_index = chapter_index_translation
                        translated.toc.append(TocEntry(text, chapter_index_translation, len(translated.blocks)))
                    translated.blocks.append(tr_block)

        source.add_log("ai_image_processing", f"AI image extraction: {len(source.blocks)} blocks", len(source.blocks))
        if translated is not None:
            translated.metadata.ai_image_report = {
                "translation_missing_blocks": missing_translation_count,
            }
            translated.add_log(
                "ai_image_processing",
                f"AI image extraction + translation: {len(translated.blocks)} blocks; missing translations {missing_translation_count}",
                len(translated.blocks),
            )
        return source, translated


def test_multimodal_model(settings: AISettings, image_path: str | Path) -> tuple[str, dict]:
    """Perform one deliberately tiny real image request; callers must opt in."""
    cache_root = CONFIG_PATH.parent / "ai_image_test"
    cache_root.mkdir(parents=True, exist_ok=True)
    target = cache_root / "vision-capability-test.png"
    with Image.open(image_path) as src:
        src.load()
        image = ImageOps.exif_transpose(src).convert("RGB")
    if max(image.size) > 768:
        ratio = 768 / max(image.size)
        image = image.resize((max(1, round(image.width * ratio)), max(1, round(image.height * ratio))), Image.Resampling.LANCZOS)
    image.save(target, "PNG", optimize=True)
    prompt = (
        "这是图片能力连通测试。读取图片中最明显的一小段可见文字。"
        "只返回JSON：{\"ok\":true,\"text\":\"...\"}。不要描述图片。"
    )
    with MultimodalClient(settings) as client:
        raw = client.call_json(prompt, [str(target)], temperature=0.0)
        payload = _json_object(raw)
        if not bool(payload.get("ok")):
            raise ValueError("模型返回了响应，但没有确认图片读取成功")
        text = _safe_text(payload.get("text"))
        return text, client.usage_snapshot()
