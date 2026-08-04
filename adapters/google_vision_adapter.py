#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Google Cloud Vision DOCUMENT_TEXT_DETECTION adapter.

The app already expands PDFs to page images, so this adapter uses the synchronous
images:annotate endpoint one page at a time.  That avoids requiring a GCS bucket
for the asynchronous PDF/TIFF API and preserves the common page/crop workflow.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from adapters.ocr_engine_common import run_ocr_engine

DEFAULT_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
_BREAK_SPACE = {"SPACE", "SURE_SPACE", "EOL_SURE_SPACE"}
_BREAK_LINE = {"LINE_BREAK"}


def _vertices(box: dict | None) -> list[list[float]] | None:
    vertices = (box or {}).get("vertices") or (box or {}).get("normalizedVertices") or []
    if not vertices:
        return None
    # normalizedVertices cannot be converted without page size. Google normally
    # returns pixel vertices for fullTextAnnotation paragraph boxes.
    if "normalizedVertices" in (box or {}) and not (box or {}).get("vertices"):
        return None
    points = [[float(v.get("x", 0)), float(v.get("y", 0))] for v in vertices]
    if len(points) < 4:
        return None
    return points[:4]


def _symbol_text(symbol: dict) -> tuple[str, str]:
    text = str(symbol.get("text") or "")
    break_type = str(
        (((symbol.get("property") or {}).get("detectedBreak") or {}).get("type") or "")
    )
    return text, break_type


def _paragraph_text(paragraph: dict) -> str:
    out: list[str] = []
    for word in paragraph.get("words") or []:
        for symbol in word.get("symbols") or []:
            text, break_type = _symbol_text(symbol)
            out.append(text)
            if break_type in _BREAK_SPACE:
                out.append(" ")
            elif break_type in _BREAK_LINE:
                out.append("\n")
    return "".join(out).strip()


def _paragraph_confidence(paragraph: dict) -> float:
    values: list[float] = []
    for word in paragraph.get("words") or []:
        try:
            values.append(float(word.get("confidence")))
        except (TypeError, ValueError):
            pass
    if values:
        return sum(values) / len(values)
    try:
        return float(paragraph.get("confidence", 0.9))
    except (TypeError, ValueError):
        return 0.9


def parse_full_text_annotation(response: dict) -> list[dict]:
    annotation = response.get("fullTextAnnotation") or {}
    blocks: list[dict] = []
    for page in annotation.get("pages") or []:
        for block_index, block in enumerate(page.get("blocks") or []):
            for paragraph_index, paragraph in enumerate(block.get("paragraphs") or []):
                text = _paragraph_text(paragraph)
                if not text:
                    continue
                box = _vertices(paragraph.get("boundingBox")) or _vertices(block.get("boundingBox"))
                blocks.append({
                    "text": text,
                    "confidence": _paragraph_confidence(paragraph),
                    "box": box,
                    "layout_group": block_index,
                    "layout_order": paragraph_index,
                    "direction": "vertical" if box and _is_vertical(box) else "horizontal",
                    "label": str(block.get("blockType") or "TEXT"),
                })
    if blocks:
        return _sort_reading_order(blocks)

    # Defensive fallback for unusual responses that only contain one flat text.
    flat = str(annotation.get("text") or "").strip()
    return [{"text": flat, "confidence": 0.9, "box": None}] if flat else []


def _is_vertical(box: list[list[float]]) -> bool:
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    return (max(ys) - min(ys)) > (max(xs) - min(xs)) * 1.25


def _sort_reading_order(blocks: list[dict]) -> list[dict]:
    boxed = [item for item in blocks if item.get("box")]
    if len(boxed) < 2:
        return blocks
    vertical = sum(1 for item in boxed if item.get("direction") == "vertical") >= len(boxed) / 2

    def key(item: dict):
        box = item.get("box")
        if not box:
            return (10**9, 10**9)
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        return (-cx, cy) if vertical else (cy, cx)

    return sorted(blocks, key=key)


def _annotate_image(path: str, *, api_key: str, language_hints: list[str], endpoint: str,
                    timeout: int = 120) -> list[dict]:
    content = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    image_context = {"languageHints": language_hints} if language_hints else {}
    request_item = {
        "image": {"content": content},
        "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
    }
    if image_context:
        request_item["imageContext"] = image_context
    payload = json.dumps({"requests": [request_item]}).encode("utf-8")
    url = endpoint.rstrip("?") + "?key=" + urllib.parse.quote(api_key)
    req = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json", "User-Agent": "NovelFormatterStudio/2"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Google Vision HTTP {exc.code}: {body[-2000:]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"无法连接 Google Vision: {exc}") from exc

    item = (data.get("responses") or [{}])[0]
    if item.get("error"):
        error = item["error"]
        raise RuntimeError(f"Google Vision API 错误 {error.get('code', '')}: {error.get('message', error)}")
    return parse_full_text_annotation(item)


def run(
    *,
    api_key: str = "",
    language_hints: str | list[str] = "",
    endpoint: str = DEFAULT_ENDPOINT,
    verbose: bool = True,
    **kwargs,
):
    key = (api_key or os.environ.get("GOOGLE_CLOUD_VISION_API_KEY", "")).strip()
    if not key:
        raise ValueError("请填写 Google Cloud Vision API Key，或设置 GOOGLE_CLOUD_VISION_API_KEY。")
    if isinstance(language_hints, str):
        hints = [part.strip() for part in language_hints.replace(";", ",").split(",") if part.strip()]
    else:
        hints = [str(part).strip() for part in language_hints if str(part).strip()]
    from adapters.ocr_profiles import get_ocr_profile, normalize_ocr_mode
    mode = normalize_ocr_mode(kwargs.get("ocr_mode", "ja_vertical"))
    profile = get_ocr_profile(mode)
    if not hints:
        hints = list(profile.google_language_hints)

    def worker_fn(ocr_paths, cancel_check):
        for path in ocr_paths:
            if cancel_check is not None and cancel_check():
                break
            try:
                yield path, _annotate_image(
                    path, api_key=key, language_hints=hints, endpoint=endpoint
                ), None
            except Exception as exc:
                yield path, None, str(exc)

    return run_ocr_engine(
        worker_fn,
        source_engine="google_vision_document_text_detection",
        verbose=verbose,
        **kwargs,
    )
