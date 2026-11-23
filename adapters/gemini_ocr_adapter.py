"""Gemini OCR on *locally anchored* Japanese vertical column crops.

This recognizer does not discover page layout: it consumes the same physical
column masks and coordinates as the existing local OCR engines. Failed or
ambiguous remote responses never become authoritative text. The surrounding
multi-OCR pipeline retains local candidates and manual review.
"""
from __future__ import annotations

import os
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from PIL import Image, ImageChops

from ai.config import api_key_validation_error, load_ai_settings, load_gemini_ocr_key
from ai.gemini_free_tier import DEFAULT_RPD, LocalQuotaExceeded
from ai.gemini_ocr_transport import (
    DEFAULT_MODEL, OCR_SCHEMA, GeminiJsonTransport, GeminiRequestError,
    GeminiResponseRejected, GeminiTransportError, validate_items,
)
from core.ocr_segment_cache import OcrSegmentCache, file_sha256
from adapters.vertical_crop_segments import looks_like_full_page

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BATCH_ITEMS = 30
_SHEET_EDGE = 2048
_SHEET_PIXELS = 3_000_000
_TILE_MARGIN = 8
_LABEL_HEIGHT = 24
_ColumnResult = tuple[str, float, str | None]
PROMPT = ("Literal Japanese OCR. Each numbered box is an independent physical body column "
          "masked by the local layout detector. Read the Japanese main printed text from "
          "top to bottom, preserving every printed character, punctuation, small kana and "
          "whitespace; never translate, rewrite, complete missing text or guess. "
          "Return only the main body glyphs. Omit all small side readings (furigana/ruby); "
          "never append them in parentheses or output ruby markup. "
          "Return only the characters inside each box, even when a sentence continues in another box. "
          "Return one item for every indicated ID with exact matching id. "
          "If invisible, cropped, or ambiguous, text must be empty. "
          "Respond as JSON {\"items\":[{\"id\":\"c001\",\"text\":\"...\"}]}; no other text.")


@dataclass(frozen=True, slots=True)
class _ColumnSheet:
    paths: tuple[str, ...]
    sizes: tuple[tuple[int, int], ...]
    offsets: tuple[tuple[int, int], ...]
    size: tuple[int, int]


def _sheet_layout(
    paths: list[str], sizes: dict[str, tuple[int, int]],
) -> _ColumnSheet | None:
    native_sizes = tuple(sizes[path] for path in paths)
    tiles = [
        (max(56, width + 2 * _TILE_MARGIN), height + _LABEL_HEIGHT + _TILE_MARGIN)
        for width, height in native_sizes
    ]
    widest = max(width for width, _ in tiles)
    best = None
    best_score = None
    for row_width in sorted({widest, *range(512, _SHEET_EDGE + 1, 256)}):
        if row_width < widest:
            continue
        x = y = row_height = used_width = 0
        offsets = []
        for width, height in tiles:
            if x and x + width > row_width:
                y += row_height
                x = row_height = 0
            offsets.append((x, y))
            x += width
            used_width = max(used_width, x)
            row_height = max(row_height, height)
        used_height = y + row_height
        pixels = used_width * used_height
        if len(paths) > 1 and (max(used_width, used_height) > _SHEET_EDGE or pixels > _SHEET_PIXELS):
            continue
        # Prefer compact shapes so a long strip is less likely to be scaled down upstream.
        score = (pixels + abs(used_width - used_height) ** 2 // 4, pixels)
        if best_score is None or score < best_score:
            best_score = score
            best = _ColumnSheet(tuple(paths), native_sizes, tuple(offsets), (used_width, used_height))
    return best


def plan_column_sheets(paths: list[str], batch_items: int) -> Iterator[_ColumnSheet]:
    """Keep reading order and split on size before sacrificing any image resolution."""
    limit = max(1, min(30, int(batch_items)))
    pending: list[str] = []
    sizes: dict[str, tuple[int, int]] = {}
    layout = None
    for path in paths:
        with Image.open(path) as image:
            size = image.size
        sizes[path] = size
        proposed = _sheet_layout([*pending, path], sizes)
        if layout is not None and (proposed is None or len(pending) == limit):
            yield layout
            pending = []
            sizes = {path: size}
            proposed = _sheet_layout([path], sizes)
        pending.append(path)
        layout = proposed
    if layout is not None:
        yield layout


def _contact_sheet(sheet: _ColumnSheet, output: Path) -> tuple[Path, list[str]]:
    from PIL import ImageDraw, ImageFont
    ids: list[str] = []
    with Image.new("RGB", sheet.size, "white") as canvas:
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.load_default(size=16)
        except TypeError:
            font = ImageFont.load_default()
        for offset, (path, (width, height), (x, y)) in enumerate(
            zip(sheet.paths, sheet.sizes, sheet.offsets, strict=True), 1
        ):
            row_id = f"c{offset:03d}"
            ids.append(row_id)
            tile_width = max(56, width + 2 * _TILE_MARGIN)
            draw.rectangle(
                (x + 1, y + 1, x + tile_width - 3, y + height + _LABEL_HEIGHT + _TILE_MARGIN - 3),
                outline=(190, 190, 190),
            )
            draw.text((x + _TILE_MARGIN, y + 5), row_id, fill="black", font=font)
            with Image.open(path) as opened, opened.convert("RGB") as image:
                canvas.paste(image, (x + (tile_width - width) // 2, y + _LABEL_HEIGHT))
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output, format="WEBP", lossless=True, method=4)
    return output, ids


def _prepare_column_slices(path: str, target_dir: Path) -> list[str]:
    from adapters.vertical_crop_segments import prepare_vertical_ocr_segments

    with Image.open(path) as opened:
        width, height = opened.size
        if height <= 640 or height / max(1, width) <= 6:
            return [path]
        # Full-height masks can contain a short title or status field. Remove
        # only solid white rows before deciding whether a split is necessary.
        with opened.convert("RGB") as image, Image.new("RGB", opened.size, "white") as white:
            with ImageChops.difference(image, white) as difference:
                bounds = difference.getbbox()
            if bounds is None:
                return []
            left, top, right, bottom = bounds
            ink_width, ink_height = right - left, bottom - top
            if ink_height <= 600 and ink_height <= 18 * max(12, ink_width * 0.92):
                margin = max(8, round(ink_width * 0.24))
                target_dir.mkdir(parents=True, exist_ok=True)
                trimmed = target_dir / "column.png"
                with image.crop((0, max(0, top - margin), width, min(height, bottom + margin))) as crop:
                    crop.save(trimmed, format="PNG", compress_level=1)
                return [str(trimmed)]
    segments, count = prepare_vertical_ocr_segments(
        path, target_dir, max_aspect=5.6, max_chars=18,
        already_isolated=True, estimate_isolated_chars=True,
    )
    if count != 1:
        return []
    return [segment.path for segment in sorted(segments, key=lambda value: value.segment_index)]


class GeminiOcrSession:
    def __init__(self, *, engine_options: dict | None = None, cancel_check=None, transport=None):
        self.options = dict(engine_options or {})
        key = str(self.options.get("api_key") or load_gemini_ocr_key()
                  or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "")
        if not key:
            settings = load_ai_settings()
            if settings.provider == "gemini":
                key = settings.api_key
        self.model = str(self.options.get("model") or
                         os.environ.get("NOVEL_FORMATTER_GEMINI_OCR_MODEL") or DEFAULT_MODEL)
        self._key_error = api_key_validation_error(key) if transport is None else ""
        if not key.strip() and transport is None:
            self._key_error = "Gemini OCR 未配置 API 密钥，请在识别引擎下方填写"
        self.transport = transport or GeminiJsonTransport(
            api_key=key, model=self.model, max_retries=int(self.options.get("retries", 2)),
            request_timeout=float(self.options.get("request_timeout", 120)),
            rpm=int(self.options.get("rpm_limit", 0)),
            tpm=int(self.options.get("tpm_limit", 0)),
            rpd=int(self.options.get("rpd_limit", DEFAULT_RPD)),
            free_mode=bool(self.options.get("free_mode", True)),
            fallback_model=str(self.options.get("fallback_model") or ""),
            enable_fallback=bool(self.options.get("enable_fallback", False)),
        )
        self.cancel_check = cancel_check
        self.used_fallback_paths: set[str] = set()
        self.batch_items = max(1, min(30, int(self.options.get("batch_items", DEFAULT_BATCH_ITEMS))))
        self.concurrency = max(1, min(8, int(self.options.get("concurrency", 2))))
        self._cache = None
        if self.options.get("cache", True):
            # The prompt is part of the cache identity, so updates cannot reuse stale OCR.
            from core.ocr_segment_cache import stable_fingerprint
            cache_root = Path(self.options.get("cache_dir") or ROOT / ".model-cache" / "gemini-ocr")
            fp = stable_fingerprint({"model": self.model, "prompt": PROMPT, "mode": "native-pixel-sheet-v2"})
            self._cache = OcrSegmentCache(cache_root, engine="gemini_ocr", runtime_id=fp)

    def close(self):
        if self._cache is not None:
            self._cache.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def _read_sheet(self, sheet: _ColumnSheet) -> tuple[dict[str, _ColumnResult], int, bool]:
        if self.cancel_check and self.cancel_check():
            raise RuntimeError("OCR 已取消")
        retry_limit = min(8, len(sheet.paths) // 2)
        fallback_used = False
        try:
            with tempfile.TemporaryDirectory(prefix="nf_gemini_columns_") as td:
                image, ids = _contact_sheet(sheet, Path(td) / "columns.webp")
                raw = self.transport.generate(
                    f"{PROMPT} This image contains exactly {len(ids)} boxes, c001 through {ids[-1]}.",
                    [image], schema=OCR_SCHEMA,
                    max_output_tokens=max(1500, min(16000, 1000 * len(ids))),
                    cancel_check=self.cancel_check,
                )
                provenance = getattr(self.transport, "request_provenance", None)
                fallback_used = callable(provenance) and bool(provenance().get("fallback_used"))
                values = validate_items(raw, ids, allow_partial=True)
        except (GeminiRequestError, GeminiResponseRejected, LocalQuotaExceeded) as exc:
            return dict.fromkeys(sheet.paths, ("", 0.0, str(exc))), 0, False
        except GeminiTransportError:
            result = dict.fromkeys(sheet.paths, ("", 0.0, "Gemini OCR 返回格式不完整"))
        else:
            result = {}
            for path, key in zip(sheet.paths, ids, strict=True):
                text = values.get(key, "")
                result[path] = (text, 0.0, None) if text.strip() else (
                    "", 0.0, "Gemini 未返回该列的有效文字"
                )
        return result, 0 if fallback_used else retry_limit, fallback_used

    def _recognize_slices(self, paths: list[str]) -> dict[str, _ColumnResult]:
        # A cache identity proves identical bytes. Each unique image needs at most
        # one entry in a request, even when it occurs on several pages.
        slices: dict[str, _ColumnResult] = {}
        representative_by_hash: dict[str, str] = {}
        duplicates: dict[str, str] = {}
        for path in dict.fromkeys(paths):
            found = self._cache.load(path, "segment_text") if self._cache else None
            if found:
                slices[path] = found
                continue
            representative = representative_by_hash.setdefault(file_sha256(path), path)
            if representative != path:
                duplicates[path] = representative
        uncached = list(representative_by_hash.values())
        retry_limits: dict[str, int] = {}

        def read_round(pool, sheets, *, retry: bool = False) -> None:
            futures = [pool.submit(self._read_sheet, sheet) for sheet in sheets]
            for future in as_completed(futures):
                if self.cancel_check and self.cancel_check():
                    raise RuntimeError("OCR 已取消")
                recognized, retry_limit, fallback_used = future.result()
                for path, result in recognized.items():
                    slices[path] = result
                    if result[0] and not result[2]:
                        if fallback_used:
                            self.used_fallback_paths.add(path)
                        elif self._cache:
                            self._cache.save(path, "segment_text", result)
                    elif retry_limit and not retry:
                        retry_limits[path] = retry_limit

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            read_round(pool, plan_column_sheets(uncached, self.batch_items))
            retry_groups = defaultdict(list)
            for path in uncached:
                if path in retry_limits:
                    retry_groups[retry_limits[path]].append(path)
            read_round(pool, (
                sheet for limit, group in retry_groups.items()
                for sheet in plan_column_sheets(group, limit)
            ), retry=True)
        for path, representative in duplicates.items():
            slices[path] = slices[representative]
            if representative in self.used_fallback_paths:
                self.used_fallback_paths.add(path)
        return slices

    def recognize(self, paths: list[str]) -> dict[str, tuple[str, float, str | None]]:
        ordered = list(dict.fromkeys(str(p) for p in paths))
        self.used_fallback_paths.clear()
        if self._key_error:
            return {p: ("", 0., self._key_error) for p in ordered}
        results: dict[str, tuple[str, float, str | None]] = {}
        with tempfile.TemporaryDirectory(prefix="nf_gemini_segmented_") as td:
            segments_by_source: dict[str, list[str]] = {}
            todo: list[str] = []
            for offset, path in enumerate(ordered):
                if self.cancel_check and self.cancel_check():
                    raise RuntimeError("OCR 已取消")
                if not Path(path).is_file():
                    results[path] = ("", 0.0, "Gemini 输入文件不存在")
                    continue
                if looks_like_full_page(path):
                    results[path] = ("", 0.0, "Gemini 禁止裸扫整页：请先启用本地物理分列")
                    continue
                found = self._cache.load(path, "base_text") if self._cache else None
                if found:
                    results[path] = found
                    continue
                slices = _prepare_column_slices(path, Path(td) / f"source-{offset:06d}")
                if not slices:
                    results[path] = ("", 0.0, "本地切列后无法建立完整的可读文字段")
                    continue
                segments_by_source[path] = slices
                todo.extend(slices)
            slice_results = self._recognize_slices(todo)
            for original, slices in segments_by_source.items():
                parts = [slice_results.get(path, ("", 0., "Gemini 缺少切片返回")) for path in slices]
                issues = [str(error) for text, confidence, error in parts if error or not str(text).strip()]
                if issues:
                    # An unresolved slice must not silently disappear from a novel column.
                    results[original] = ("", 0., "；".join(dict.fromkeys(issues))[:500])
                    continue
                text = "".join(str(part[0]) for part in parts)
                results[original] = (text, 0., None)
                if any(path in self.used_fallback_paths for path in slices):
                    self.used_fallback_paths.add(original)
                elif self._cache:
                    self._cache.save(original, "base_text", results[original])
        return results


def recognize_crops(crop_paths: list[str], manifest_path: str, *, cancel_check=None,
                    verbose=True, engine_options=None, input_metadata=None) -> Iterator[tuple[str, list[dict] | None, str | None]]:
    del manifest_path, verbose, input_metadata
    with GeminiOcrSession(engine_options=engine_options, cancel_check=cancel_check) as session:
        results = session.recognize(crop_paths)
        fallback_paths = set(session.used_fallback_paths)
    for path in crop_paths:
        text, confidence, error = results.get(str(path), ("", 0.0, "Gemini 未返回结果"))
        yield path, ([{"text": text, "confidence": confidence, "confidence_kind": "unverified_remote_fallback" if path in fallback_paths else "unverified_remote",
                       "box": None}] if text else None), error


def run(*, verbose=True, **kwargs):
    from adapters.column_ocr_adapter import run as run_column_ocr
    engine = str(kwargs.pop("recognition_engine", "gemini_ocr"))
    if engine != "gemini_ocr":
        raise ValueError("Gemini OCR 不支持覆盖识别引擎")
    # The fixed physical-column layout engine owns every pixel coordinate.
    return run_column_ocr(recognition_engine="gemini_ocr", verbose=verbose, **kwargs)
