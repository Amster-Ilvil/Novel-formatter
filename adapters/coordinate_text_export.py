#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Coordinate-preserving text export for scanned Japanese pages.

The schema is intentionally small and stable.  It follows the useful mokuro
product idea (page image + selectable text coordinates) without claiming binary
compatibility with mokuro's private/internal formats.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from models.document import UnifiedDocument


SCHEMA_NAME = "novel-formatter-coordinate-text"
SCHEMA_VERSION = 1


def _page_dimensions(doc: UnifiedDocument) -> dict[int, tuple[int, int, str]]:
    result: dict[int, tuple[int, int, str]] = {}
    for page in getattr(doc, "pages", []) or []:
        number = int(getattr(page, "page_no", 0) or 0)
        if number <= 0:
            continue
        result[number] = (
            int(getattr(page, "width", 0) or 0),
            int(getattr(page, "height", 0) or 0),
            str(getattr(page, "image_path", "") or ""),
        )
    return result


def _safe_box(values: Any) -> list[int]:
    if not isinstance(values, (list, tuple)) or len(values) < 4:
        return []
    try:
        x0, y0, x1, y1 = [int(round(float(value))) for value in values[:4]]
    except Exception:
        return []
    if x1 <= x0 or y1 <= y0:
        return []
    return [x0, y0, x1, y1]


def _column_entries_from_metadata(doc: UnifiedDocument) -> list[dict]:
    stored = getattr(getattr(doc, "metadata", None), "handwriting_coordinate_columns", None)
    if isinstance(stored, list):
        return [dict(item) for item in stored if isinstance(item, dict)]

    entries: list[dict] = []
    for block in getattr(doc, "blocks", []) or []:
        metadata = dict(getattr(block, "metadata", {}) or {})
        glyph_boxes = metadata.get("handwriting_input_glyph_boxes")
        if not isinstance(glyph_boxes, list) or not glyph_boxes:
            continue
        entries.append({
            "page": int(getattr(block, "page", 0) or getattr(block, "page_number", 0) or 0),
            "column": int(metadata.get("column_index", 0) or 0),
            "column_id": str(metadata.get("column_id", "") or ""),
            "text": str(getattr(block, "text", "") or ""),
            "confidence": float(getattr(block, "confidence", 0.0) or 0.0),
            "crop_box": _safe_box(metadata.get("black_ink_exact_crop_box")),
            "glyph_boxes": glyph_boxes,
            "candidate_preview": metadata.get("handwriting_input_auto_preview") or [],
            "segmentation_mode": str(metadata.get("handwriting_input_segmentation_mode", "") or ""),
        })
    return entries


def document_to_coordinate_payload(doc: UnifiedDocument) -> dict:
    dimensions = _page_dimensions(doc)
    page_map: dict[int, dict] = {}
    columns = _column_entries_from_metadata(doc)
    for serial, column in enumerate(columns):
        page_no = int(column.get("page", 0) or 0)
        if page_no <= 0:
            continue
        page_w, page_h, image_path = dimensions.get(page_no, (0, 0, ""))
        page_payload = page_map.setdefault(page_no, {
            "page": page_no,
            "width": page_w,
            "height": page_h,
            "image": image_path,
            "columns": [],
        })
        crop_box = _safe_box(column.get("crop_box"))
        crop_x0 = crop_box[0] if crop_box else 0
        crop_y0 = crop_box[1] if crop_box else 0
        text = str(column.get("text", "") or "")
        glyph_boxes = column.get("glyph_boxes") or []
        preview = column.get("candidate_preview") or []
        preview_by_index = {
            int(item.get("i", -1)): item
            for item in preview if isinstance(item, dict) and int(item.get("i", -1)) >= 0
        }
        chars: list[dict] = []
        for glyph_index, raw_box in enumerate(glyph_boxes):
            if not isinstance(raw_box, dict):
                continue
            try:
                local_x0 = int(raw_box.get("x0", 0) or 0)
                local_y0 = int(raw_box.get("y0", 0) or 0)
                local_x1 = int(raw_box.get("x1", 0) or 0)
                local_y1 = int(raw_box.get("y1", 0) or 0)
            except Exception:
                continue
            if local_x1 <= local_x0 or local_y1 <= local_y0:
                continue
            char = text[glyph_index] if glyph_index < len(text) else "□"
            preview_item = preview_by_index.get(glyph_index, {})
            confidence = float(preview_item.get("s", raw_box.get("anchor_confidence", 0.0)) or 0.0)
            chars.append({
                "index": glyph_index,
                "text": char,
                "box": [
                    crop_x0 + local_x0,
                    crop_y0 + local_y0,
                    crop_x0 + local_x1,
                    crop_y0 + local_y1,
                ],
                "local_box": [local_x0, local_y0, local_x1, local_y1],
                "confidence": round(confidence, 4),
                "source": str(raw_box.get("source", "") or ""),
                "ruby": False,
                "unresolved": char == "□",
            })
        page_payload["columns"].append({
            "order": int(column.get("column", 0) or serial + 1),
            "column_id": str(column.get("column_id", "") or ""),
            "direction": "vertical-rl",
            "crop_box": crop_box,
            "text": text,
            "confidence": round(float(column.get("confidence", 0.0) or 0.0), 4),
            "segmentation_mode": str(column.get("segmentation_mode", "") or ""),
            "chars": chars,
        })

    pages = [page_map[key] for key in sorted(page_map)]
    for page in pages:
        page["columns"].sort(key=lambda item: int(item.get("order", 0) or 0))
    return {
        "schema": SCHEMA_NAME,
        "version": SCHEMA_VERSION,
        "language": str(getattr(getattr(doc, "metadata", None), "language", "ja") or "ja"),
        "title": str(getattr(getattr(doc, "metadata", None), "title", "") or ""),
        "reading_order": "page-ascending; columns-right-to-left; glyphs-top-to-bottom",
        "pages": pages,
    }


def export_coordinate_text_json(doc: UnifiedDocument, output_path: str | Path) -> str:
    target = Path(output_path)
    if target.suffix.lower() != ".json":
        target = target.with_suffix(".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = document_to_coordinate_payload(doc)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
    return str(target)
