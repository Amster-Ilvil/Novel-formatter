#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lossless external-AI round-trip packages for OCR documents.

The package deliberately separates immutable structure from editable text:

* pages, cover/illustration assets, block order, coordinates, anchors, TOC,
  column IDs and sentence lineage live in ``structure_document``;
* an external model edits only ``edited_text`` fields;
* import validates unique stable IDs and complete coverage, then writes text by
  ID.  It never performs global string replacement, so repeated sentences do
  not create duplicate replacements.

JSON is the canonical representation.  Markdown embeds the same canonical JSON
manifest in a compressed HTML comment and exposes only human-readable editable
sections.  Both formats therefore use exactly the same validator and importer.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import re
import time
import uuid
import zlib
from pathlib import Path
from typing import Iterable, Sequence

from models.document import Block, BlockType, UnifiedDocument

SCHEMA = "novel_formatter.ocr_roundtrip.v1"
MODE_SINGLE = "single_ocr"
MODE_MULTI = "multi_model_fusion"
_TEXT_TYPES = {
    BlockType.PARAGRAPH,
    BlockType.DIALOGUE,
    BlockType.CHAPTER,
    BlockType.SECTION,
    BlockType.RUBY,
    BlockType.FOOTNOTE,
    BlockType.TOC_ENTRY,
}
_MD_MANIFEST_RE = re.compile(
    r"<!--\s*NFRT_MANIFEST_ZLIB_BASE64:([A-Za-z0-9+/=\r\n]+?)\s*-->",
    re.DOTALL,
)
_MD_EDIT_RE = re.compile(
    r"<!--\s*NFRT_EDIT_BEGIN\s+([^\s>]+)\s*-->\r?\n"
    r"(.*?)\r?\n<!--\s*NFRT_EDIT_END\s+\1\s*-->",
    re.DOTALL,
)
_MD_DELETE_MARKER = "<!-- NFRT_DELETE_INTENTIONALLY -->"


class RoundtripPackageError(ValueError):
    """A package is incomplete, duplicated, malformed, or structurally unsafe."""


def _explicit_bool(value, *, field: str = "布尔字段") -> bool:
    """Parse only explicit JSON-like booleans.

    Python treats every non-empty string as true, so an externally edited
    ``"false"`` used to become an intentional deletion.  Accept the common
    structured-output spellings but reject ambiguous values instead of silently
    changing the book.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, float) and value in (0.0, 1.0):
        return bool(int(value))
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes"}:
            return True
        if token in {"false", "0", "no", ""}:
            return False
    raise RoundtripPackageError(f"{field} 必须是明确的 true/false 或 0/1。")


def _json_bytes(value) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value) -> str:
    data = value if isinstance(value, (bytes, bytearray)) else _json_bytes(value)
    return hashlib.sha256(data).hexdigest()


def _safe_metadata(value) -> dict:
    return copy.deepcopy(value) if isinstance(value, dict) else {}


def _finite_float(value, default=0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _bbox_dict(block: Block | None):
    bbox = getattr(block, "bbox", None) if block is not None else None
    if bbox is None:
        return None
    return {
        "x": _finite_float(getattr(bbox, "x", 0.0)),
        "y": _finite_float(getattr(bbox, "y", 0.0)),
        "w": _finite_float(getattr(bbox, "w", 0.0)),
        "h": _finite_float(getattr(bbox, "h", 0.0)),
    }


def _column_geometry_snapshot(doc: UnifiedDocument) -> dict[str, dict]:
    """Return deterministic per-column geometry for independent order checks."""
    raw: list[tuple[int, str, dict, str]] = []
    for block in doc.blocks:
        if block.type not in _TEXT_TYPES:
            continue
        metadata = block.metadata if isinstance(block.metadata, dict) else {}
        values = metadata.get("source_column_ids") or metadata.get("multi_ocr_column_ids") or metadata.get("column_id") or []
        if isinstance(values, str):
            values = [values]
        elif not isinstance(values, (list, tuple, set)):
            values = []
        ids = [str(value) for value in values if str(value)]
        if not ids:
            continue
        bbox = _bbox_dict(block)
        if bbox is None:
            continue
        for column_id in ids:
            raw.append((int(getattr(block, "page", 0) or 0), column_id, bbox, str(getattr(block, "id", "") or "")))
    by_page: dict[int, list[tuple[int, str, dict, str]]] = {}
    for entry in raw:
        by_page.setdefault(entry[0], []).append(entry)
    result: dict[str, dict] = {}
    vertical = str(getattr(getattr(doc, "metadata", None), "writing_direction", "") or "vertical-rl").startswith("vertical")
    for page, entries in by_page.items():
        entries = sorted(
            entries,
            key=(lambda entry: (-float(entry[2]["x"] + entry[2]["w"] / 2.0), float(entry[2]["y"])))
            if vertical else (lambda entry: (float(entry[2]["y"]), float(entry[2]["x"]))),
        )
        for rank, (_page, column_id, bbox, block_id) in enumerate(entries):
            result.setdefault(column_id, {
                "column_id": column_id,
                "page": page,
                "bbox": [bbox["x"], bbox["y"], bbox["w"], bbox["h"]],
                "center_x": bbox["x"] + bbox["w"] / 2.0,
                "center_y": bbox["y"] + bbox["h"] / 2.0,
                "reading_rank": rank,
                "source_block_id": block_id,
                "geometry_source": "ocr_block_bbox",
            })
    return result


def _structure_document_dict(doc: UnifiedDocument) -> dict:
    """Return a complete skeleton while keeping original text as a fallback.

    The structure hash below ignores mutable text, but the embedded document
    keeps it so the package can be reopened even after the app has restarted.
    """
    return copy.deepcopy(doc.to_dict())


def _structure_projection(document_dict: dict) -> dict:
    value = copy.deepcopy(document_dict if isinstance(document_dict, dict) else {})
    value.pop("processing_log", None)
    metadata = value.get("metadata")
    if isinstance(metadata, dict):
        # Processing reports and mutable text-authority flags are not layout.
        for key in list(metadata):
            if key.endswith("_report") or key in {
                "replacement_source_hash",
                "replacement_output_hash",
                "replacement_source_chars",
                "replacement_output_chars",
                "replacement_missing_chars",
                "replacement_extra_chars",
            }:
                metadata.pop(key, None)
    for block in value.get("blocks", []) if isinstance(value.get("blocks"), list) else []:
        if not isinstance(block, dict):
            continue
        block["text"] = ""
        block.pop("ocr_raw", None)
        block.pop("modified_by", None)
        # Candidate texts are editable OCR evidence, not geometry.  IDs, regions,
        # page lineage and image paths remain in the projection.
        meta = block.get("metadata")
        if isinstance(meta, dict):
            for key in list(meta):
                if key in {
                    "source_column_texts",
                    "last_column_text",
                    "candidate_texts",
                    "column_ocr_candidates",
                    "multi_ocr_candidates",
                }:
                    meta.pop(key, None)
    return value


def structure_hash(doc_or_dict: UnifiedDocument | dict) -> str:
    data = doc_or_dict.to_dict() if isinstance(doc_or_dict, UnifiedDocument) else doc_or_dict
    return _sha256(_structure_projection(data))


def _layout_projection(doc_or_dict: UnifiedDocument | dict) -> dict:
    data = doc_or_dict.to_dict() if isinstance(doc_or_dict, UnifiedDocument) else doc_or_dict
    value = _structure_projection(data)
    # Book metadata (title, processing mode, audit reports) may legitimately
    # change while the physical OCR layout stays identical.
    value["metadata"] = {}
    # Asset files may be moved while the same book is reopened.  Geometry,
    # IDs, page types, anchors, dimensions and ordering remain authoritative;
    # absolute paths are excluded only from the current-book compatibility hash.
    for page in value.get("pages", []) if isinstance(value.get("pages"), list) else []:
        if isinstance(page, dict):
            page["image_path"] = ""
    for block in value.get("blocks", []) if isinstance(value.get("blocks"), list) else []:
        if not isinstance(block, dict):
            continue
        block["image_path"] = ""
        meta = block.get("metadata")
        if isinstance(meta, dict):
            structural_keys = {
                "column_id", "source_column_ids", "multi_ocr_column_ids",
                "ocr_review_regions", "source_pages", "page_type",
                "placement_required", "column_count", "ocr_review_column_count",
                "atomic_ocr_sentence", "column_sentence_reflow",
                "chapter_title_atomic", "ocr_review_layout",
                "ocr_review_lineage_missing", "ocr_review_lineage_approximate",
            }
            block["metadata"] = {
                key: ("" if "path" in key.lower() else copy.deepcopy(value))
                for key, value in meta.items()
                if key in structural_keys
            }
    return value


def layout_hash(doc_or_dict: UnifiedDocument | dict) -> str:
    return _sha256(_layout_projection(doc_or_dict))


def _asset_manifest(doc: UnifiedDocument) -> list[dict]:
    assets: list[dict] = []
    for page in doc.pages:
        if not str(getattr(page, "image_path", "") or ""):
            continue
        page_type = getattr(getattr(page, "page_type", None), "value", str(getattr(page, "page_type", "")))
        assets.append({
            "kind": "page",
            "page_no": int(getattr(page, "page_no", 0) or 0),
            "page_type": page_type,
            "is_cover": page_type == BlockType.COVER.value,
            "image_path": str(getattr(page, "image_path", "") or ""),
            "width": int(getattr(page, "width", 0) or 0),
            "height": int(getattr(page, "height", 0) or 0),
            "bbox_normalized": [0.0, 0.0, 1.0, 1.0],
        })
    for order, block in enumerate(doc.blocks):
        if block.type != BlockType.IMAGE_REF and not str(getattr(block, "image_path", "") or ""):
            continue
        if not str(getattr(block, "image_path", "") or ""):
            continue
        assets.append({
            "kind": "block",
            "block_id": str(block.id or ""),
            "block_order": order,
            "page": int(getattr(block, "page", 0) or 0),
            "block_type": block.type.value,
            "image_path": str(block.image_path or ""),
            "image_anchor": str(block.image_anchor or ""),
            "bbox": _bbox_dict(block),
            "metadata": _safe_metadata(block.metadata),
        })
    return assets


def _format_manifest(doc: UnifiedDocument) -> dict:
    return {
        "page_order": [
            {
                "page_no": int(getattr(page, "page_no", 0) or 0),
                "page_type": getattr(getattr(page, "page_type", None), "value", str(getattr(page, "page_type", ""))),
                "image_path": str(getattr(page, "image_path", "") or ""),
                "width": int(getattr(page, "width", 0) or 0),
                "height": int(getattr(page, "height", 0) or 0),
                "bbox_normalized": [0.0, 0.0, 1.0, 1.0],
            }
            for page in doc.pages
        ],
        "block_order": [
            {
                "order": index,
                "block_id": str(block.id or ""),
                "block_type": block.type.value,
                "page": int(getattr(block, "page", 0) or 0),
                "bbox": _bbox_dict(block),
                "reading_order": int(getattr(block, "reading_order", 0) or 0),
                "chapter_index": int(getattr(block, "chapter_index", 0) or 0),
                "image_path": str(getattr(block, "image_path", "") or ""),
                "image_anchor": str(getattr(block, "image_anchor", "") or ""),
                "column_ids": _metadata_list(block.metadata, "source_column_ids"),
            }
            for index, block in enumerate(doc.blocks)
        ],
        "toc": [entry.to_dict() for entry in doc.toc],
    }


def _book_summary(doc: UnifiedDocument) -> dict:
    return {
        "title": str(getattr(doc.metadata, "title", "") or ""),
        "author": str(getattr(doc.metadata, "author", "") or ""),
        "language": str(getattr(doc.metadata, "language", "") or ""),
        "source_engine": str(getattr(doc.metadata, "source_engine", "") or ""),
        "page_count": len(doc.pages),
        "block_count": len(doc.blocks),
        "text_block_count": sum(1 for block in doc.blocks if block.type in _TEXT_TYPES),
        "asset_count": len(_asset_manifest(doc)),
        "structure_sha256": structure_hash(doc),
        "layout_sha256": layout_hash(doc),
    }


def _base_package(doc: UnifiedDocument, mode: str) -> dict:
    return {
        "schema": SCHEMA,
        "mode": mode,
        "package_id": uuid.uuid4().hex,
        "created_at_unix": int(time.time()),
        "book": _book_summary(doc),
        "instructions": {
            "editable_field": "edited_text",
            "do_not_change": [
                "schema",
                "mode",
                "package_id",
                "structure_document",
                "structure_sha256",
                "immutable_manifest_sha256",
                "item_id / row_id",
                "block_id",
                "page / bbox / coordinates",
                "assets / cover / illustration paths and anchors",
                "column_ids",
                "physical_column_candidates",
            ],
            "strict_import": (
                "每个 editable_items 条目必须保留且 ID 唯一；只修改 edited_text。"
                "原文非空而需要删除时，必须同时设置 delete_intentionally=true。"
            ),
        },
        "structure_sha256": structure_hash(doc),
        "layout_sha256": layout_hash(doc),
        "structure_document": _structure_document_dict(doc),
        "format_manifest": _format_manifest(doc),
        "assets": _asset_manifest(doc),
    }


def _editable_structure_hash(items: Sequence[dict]) -> str:
    immutable = []
    for raw in items:
        item = copy.deepcopy(raw if isinstance(raw, dict) else {})
        item.pop("edited_text", None)
        item.pop("delete_intentionally", None)
        immutable.append(item)
    return _sha256(immutable)


def _immutable_manifest_hash(package: dict) -> str:
    """Protect every non-editable package field without blocking text edits."""
    value = copy.deepcopy(package if isinstance(package, dict) else {})
    value.pop("immutable_manifest_sha256", None)
    items = value.get("editable_items")
    if isinstance(items, list):
        for raw in items:
            if isinstance(raw, dict):
                raw.pop("edited_text", None)
                raw.pop("delete_intentionally", None)
    return _sha256(value)


def _seal_package(package: dict) -> dict:
    package["immutable_manifest_sha256"] = _immutable_manifest_hash(package)
    return package


def export_single_package(doc: UnifiedDocument) -> dict:
    package = _base_package(doc, MODE_SINGLE)
    items: list[dict] = []
    for block_order, block in enumerate(doc.blocks):
        if block.type not in _TEXT_TYPES or (block.metadata or {}).get("consumed"):
            continue
        block_id = str(block.id or "")
        if not block_id:
            raise RoundtripPackageError(f"第 {block_order + 1} 个正文块没有稳定 block ID，无法安全导出。")
        item_id = f"block:{block_id}"
        text = str(block.text or "")
        items.append({
            "item_id": item_id,
            "block_id": block_id,
            "block_order": block_order,
            "block_type": block.type.value,
            "page": int(getattr(block, "page", 0) or 0),
            "page_index": getattr(block, "page_index", None),
            "page_number": getattr(block, "page_number", None),
            "reading_order": int(getattr(block, "reading_order", 0) or 0),
            "order_in_page": getattr(block, "order_in_page", None),
            "chapter_index": int(getattr(block, "chapter_index", 0) or 0),
            "bbox": _bbox_dict(block),
            "source_column_ids": _metadata_list(block.metadata, "source_column_ids"),
            "original_text": text,
            "original_text_sha256": _sha256(text.encode("utf-8")),
            "edited_text": text,
            "delete_intentionally": False,
        })
    ids = [item["item_id"] for item in items]
    if len(ids) != len(set(ids)):
        raise RoundtripPackageError("正文 block ID 存在重复，无法生成无歧义校对包。")
    package["editable_items"] = items
    package["editable_count"] = len(items)
    package["editable_structure_sha256"] = _editable_structure_hash(items)
    return _seal_package(package)


def _metadata_list(metadata, key: str) -> list[str]:
    if not isinstance(metadata, dict):
        return []
    value = metadata.get(key) or []
    if isinstance(value, str):
        value = [value]
    elif not isinstance(value, (list, tuple, set)):
        return []
    return [str(item) for item in value if str(item)]


def _row_payload(row, labels: Sequence[str], row_index: int) -> dict:
    texts = [str(value or "") for value in list(getattr(row, "texts", []) or [])]
    # Row IDs are structural. Repair/audit fields stay outside this lineage,
    # so rebuilding or repairing alignment never changes stable row IDs.
    lineage = {
        "primary_unit_index": getattr(row, "primary_unit_index", None),
        "primary_block_index": getattr(row, "primary_block_index", None),
        "primary_block_indices": list(getattr(row, "primary_block_indices", ()) or ()),
        "primary_block_id": str(getattr(row, "primary_block_id", "") or ""),
        "primary_segment_index": int(getattr(row, "primary_segment_index", 0) or 0),
        "insert_before_block_index": getattr(row, "insert_before_block_index", None),
        "column_ids": [str(value) for value in (getattr(row, "column_ids", ()) or ())],
        "atomic": bool(getattr(row, "atomic", False)),
    }
    stable_source = {
        "row_index": row_index,
        "lineage": lineage,
        "block_type": str(getattr(row, "block_type", BlockType.PARAGRAPH.value) or BlockType.PARAGRAPH.value),
        "page": int(getattr(row, "page", 0) or 0),
    }
    row_id = f"row:{row_index:06d}:{_sha256(stable_source)[:16]}"
    chosen = int(getattr(row, "chosen_index", 0) or 0)
    output_property = getattr(row, "output_text", "")
    original_fused = str(output_property() if callable(output_property) else output_property or "")
    if not original_fused:
        original_fused = texts[chosen] if 0 <= chosen < len(texts) else next((text for text in texts if text), "")
    candidates = []
    model_confidences = list(getattr(row, "model_confidences", ()) or ())
    for model_index, text in enumerate(texts):
        confidence = 0.0
        if model_index < len(model_confidences):
            try:
                confidence = float(model_confidences[model_index] or 0.0)
            except (TypeError, ValueError, OverflowError):
                confidence = 0.0
        candidates.append({
            "model_index": model_index,
            "model_label": labels[model_index] if model_index < len(labels) else f"模型{model_index + 1}",
            "text": text,
            "confidence": confidence,
            "text_sha256": _sha256(text.encode("utf-8")),
        })
    return {
        "row_id": row_id,
        "row_index": row_index,
        "block_type": stable_source["block_type"],
        "page": stable_source["page"],
        **lineage,
        "recommended_model_index": chosen,
        "model_confidences": [float(value or 0.0) for value in model_confidences],
        "confidence": float(getattr(row, "confidence", 0.0) or 0.0),
        "reason": str(getattr(row, "reason", "") or ""),
        "warnings": [str(value) for value in (getattr(row, "warnings", ()) or ())],
        "alignment_repaired": bool(getattr(row, "alignment_repaired", False)),
        "alignment_notes": [str(value) for value in (getattr(row, "alignment_notes", ()) or ())],
        "alignment_status": str(getattr(row, "alignment_status", "unreviewed") or "unreviewed"),
        "sentence_group_id": str(getattr(row, "sentence_group_id", "") or ""),
        "repair_reason": str(getattr(row, "repair_reason", "") or ""),
        "consensus_seeded_models": [
            int(value) for value in (getattr(row, "consensus_seeded_models", ()) or ())
        ],
        "review_classification": str(getattr(row, "review_classification", "") or ""),
        "character_fused_text": str(getattr(row, "character_fused_text", "") or ""),
        "character_fusion_confidence": float(getattr(row, "character_fusion_confidence", 0.0) or 0.0),
        "character_fusion_reason": str(getattr(row, "character_fusion_reason", "") or ""),
        "character_fusion_warnings": [str(value) for value in (getattr(row, "character_fusion_warnings", ()) or ())],
        "character_fusion_auto_selected": bool(getattr(row, "character_fusion_auto_selected", False)),
        "local_reocr_recommended": bool(getattr(row, "local_reocr_recommended", False)),
        "character_fusion_evidence": copy.deepcopy(getattr(row, "character_fusion_evidence", {}) or {}),
        "candidates": candidates,
        "original_fused_text": original_fused,
        "edited_text": original_fused,
        "delete_intentionally": False,
    }


def _comparison_payload(comparison) -> dict:
    return {
        "labels": list(getattr(comparison, "labels", []) or []),
        "alignment_mode": str(getattr(comparison, "alignment_mode", "text_many_to_many") or "text_many_to_many"),
        "exact_rows": int(getattr(comparison, "exact_rows", 0) or 0),
        "provisional_consensus_rows": int(getattr(comparison, "provisional_consensus_rows", 0) or 0),
        "conflict_rows": int(getattr(comparison, "conflict_rows", 0) or 0),
        "low_confidence_rows": int(getattr(comparison, "low_confidence_rows", 0) or 0),
        "insertion_rows": int(getattr(comparison, "insertion_rows", 0) or 0),
        "column_anchored_rows": int(getattr(comparison, "column_anchored_rows", 0) or 0),
        "chapter_atomic_rows": int(getattr(comparison, "chapter_atomic_rows", 0) or 0),
        "alignment_shift_repairs": int(getattr(comparison, "alignment_shift_repairs", 0) or 0),
        "unresolved_empty_cells": int(getattr(comparison, "unresolved_empty_cells", 0) or 0),
        "alignment_revision": int(getattr(comparison, "alignment_revision", 2) or 2),
        "physical_column_source": str(
            getattr(comparison, "physical_column_source", "source_column_primary_texts")
            or "source_column_primary_texts"
        ),
        "true_empty_rows": int(getattr(comparison, "true_empty_rows", 0) or 0),
        "single_model_only_rows": int(getattr(comparison, "single_model_only_rows", 0) or 0),
        "character_fused_rows": int(getattr(comparison, "character_fused_rows", 0) or 0),
        "character_auto_selected_rows": int(getattr(comparison, "character_auto_selected_rows", 0) or 0),
        "local_reocr_rows": int(getattr(comparison, "local_reocr_rows", 0) or 0),
    }


def export_multi_package(
    documents: Sequence[UnifiedDocument],
    labels: Sequence[str],
    comparison,
    *,
    result_lines: Sequence[str] | None = None,
    delete_flags: Sequence[bool] | None = None,
) -> dict:
    docs = list(documents or [])
    if len(docs) < 2:
        raise RoundtripPackageError("多模型校对包至少需要两份 OCR 文档。")
    rows = list(getattr(comparison, "rows", []) or [])
    package = _base_package(docs[0], MODE_MULTI)
    safe_labels = [str(value) for value in list(labels or [])[:len(docs)]]
    while len(safe_labels) < len(docs):
        safe_labels.append(f"模型{len(safe_labels) + 1}")
    items = [_row_payload(row, safe_labels, index) for index, row in enumerate(rows)]
    from engine.multi_ocr_compare import physical_column_text_snapshot
    physical_snapshots: list[tuple[dict[str, str], str]] = [
        physical_column_text_snapshot(doc) for doc in docs
    ]
    geometry_snapshots: list[dict[str, dict]] = [_column_geometry_snapshot(doc) for doc in docs]
    for item in items:
        column_ids = [str(value) for value in (item.get("column_ids") or [])]
        item["physical_column_candidates"] = []
        for model_index, ((snapshot, source), label) in enumerate(
            zip(physical_snapshots, safe_labels)
        ):
            fragments = [str(snapshot.get(column_id, "") or "") for column_id in column_ids]
            geometry = [copy.deepcopy(geometry_snapshots[model_index].get(column_id)) for column_id in column_ids]
            geometry = [value for value in geometry if isinstance(value, dict)]
            item["physical_column_candidates"].append({
                "model_index": model_index,
                "model_label": label,
                "source": source,
                "column_texts": fragments,
                "column_text_sha256": _sha256(fragments),
                "column_geometry": geometry,
            })
        item["column_geometry"] = copy.deepcopy(
            [geometry_snapshots[0].get(column_id) for column_id in column_ids if geometry_snapshots and geometry_snapshots[0].get(column_id)]
        )
    if result_lines is None and delete_flags is not None:
        raise RoundtripPackageError("delete_flags 只能与 result_lines 一起导出。")
    if result_lines is not None:
        values = list(result_lines)
        if len(values) != len(items):
            raise RoundtripPackageError(
                f"融合文本必须与对齐结果同为 {len(items)} 句，当前为 {len(values)} 句。"
            )
        raw_flags = list(delete_flags) if delete_flags is not None else [False] * len(items)
        if len(raw_flags) != len(items):
            raise RoundtripPackageError(
                f"有意删除标记必须与对齐结果同为 {len(items)} 项，当前为 {len(raw_flags)} 项。"
            )
        flags = [
            _explicit_bool(value, field=f"第 {index + 1} 行的 delete_intentionally")
            for index, value in enumerate(raw_flags)
        ]
        for item, text, delete_intentionally in zip(items, values, flags):
            item["original_fused_text"] = str(text or "")
            item["edited_text"] = str(text or "")
            item["delete_intentionally"] = delete_intentionally
    row_ids = [item["row_id"] for item in items]
    if len(row_ids) != len(set(row_ids)):
        raise RoundtripPackageError("多模型对齐行 ID 重复，无法安全导出。")
    model_sources = []
    for index, (doc, label) in enumerate(zip(docs, safe_labels)):
        snapshot, source = physical_snapshots[index]
        ordered_text = "\n".join(
            str(item["candidates"][index]["text"] or "")
            for item in items
            if index < len(item.get("candidates") or [])
        )
        model_sources.append({
            "model_index": index,
            "model_label": label,
            "source_engine": str(getattr(doc.metadata, "source_engine", "") or ""),
            "metadata": copy.deepcopy(doc.metadata.to_dict()),
            "page_count": len(doc.pages),
            "block_count": len(doc.blocks),
            "aligned_sentence_count": len(items),
            "aligned_text_sha256": _sha256(ordered_text.encode("utf-8")),
            "layout_sha256": layout_hash(doc),
            "physical_column_source": source,
            "physical_column_count": len(snapshot),
            "physical_column_text_count": sum(
                1 for value in snapshot.values() if str(value or "").strip()
            ),
        })
    package.update({
        "model_labels": safe_labels,
        # Every model's complete aligned text is already present in each row's
        # candidates.  Keep model metadata/sums here instead of duplicating
        # three full 300-page documents and making the AI package enormous.
        "model_sources": model_sources,
        "comparison": _comparison_payload(comparison),
        "editable_items": items,
        "editable_count": len(items),
    })
    package["editable_structure_sha256"] = _editable_structure_hash(items)
    return _seal_package(package)


def _validate_common(
    package: dict,
    expected_mode: str | None = None,
    *,
    validate_editable_structure: bool = True,
    validate_immutable_manifest: bool = True,
) -> None:
    if not isinstance(package, dict):
        raise RoundtripPackageError("校对包根节点必须是 JSON 对象。")
    if package.get("schema") != SCHEMA:
        raise RoundtripPackageError(f"不支持的校对包 schema：{package.get('schema')!r}")
    mode = str(package.get("mode", "") or "")
    if expected_mode and mode != expected_mode:
        raise RoundtripPackageError(f"需要 {expected_mode} 校对包，实际为 {mode or '未知'}。")
    structure = package.get("structure_document")
    if not isinstance(structure, dict):
        raise RoundtripPackageError("校对包缺少 structure_document，无法保留原始版式。")
    expected_hash = str(package.get("structure_sha256", "") or "")
    actual_hash = structure_hash(structure)
    if not expected_hash or actual_hash != expected_hash:
        raise RoundtripPackageError("原始结构、坐标、封面或插图信息已被改动；为避免错版，已拒绝导入。")
    expected_layout_hash = str(package.get("layout_sha256", "") or "")
    if not expected_layout_hash or layout_hash(structure) != expected_layout_hash:
        raise RoundtripPackageError("原始页面/块几何指纹不匹配；为避免错位，已拒绝导入。")
    items = package.get("editable_items")
    if not isinstance(items, list):
        raise RoundtripPackageError("校对包缺少 editable_items。")
    expected_count = int(package.get("editable_count", len(items)) or 0)
    if expected_count != len(items):
        raise RoundtripPackageError(
            f"校对条目不完整：应有 {expected_count} 条，当前只有 {len(items)} 条。"
        )
    if validate_editable_structure:
        _validate_editable_structure(package)
    if validate_immutable_manifest:
        _validate_immutable_manifest(package)


def _validate_editable_structure(package: dict) -> None:
    items = package.get("editable_items")
    if not isinstance(items, list):
        raise RoundtripPackageError("校对包缺少 editable_items。")
    expected_editable_hash = str(package.get("editable_structure_sha256", "") or "")
    if not expected_editable_hash or _editable_structure_hash(items) != expected_editable_hash:
        raise RoundtripPackageError(
            "校对条目的 ID、顺序、原文、模型候选或坐标映射已被改动；只允许修改 edited_text。"
        )


def _validate_immutable_manifest(package: dict) -> None:
    expected_hash = str(package.get("immutable_manifest_sha256", "") or "")
    if not expected_hash or _immutable_manifest_hash(package) != expected_hash:
        raise RoundtripPackageError(
            "校对包的模型标签、候选清单、格式清单或其他不可编辑信息已被改动；只允许修改 edited_text。"
        )


def _strict_item_map(items: Iterable[dict], id_key: str) -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    for position, raw in enumerate(items):
        if not isinstance(raw, dict):
            raise RoundtripPackageError(f"第 {position + 1} 个校对条目不是对象。")
        item_id = str(raw.get(id_key, "") or "")
        if not item_id:
            raise RoundtripPackageError(f"第 {position + 1} 个校对条目缺少 {id_key}。")
        if item_id in mapping:
            raise RoundtripPackageError(f"检测到重复 {id_key}：{item_id}。已拒绝重复替换。")
        if "edited_text" not in raw or not isinstance(raw.get("edited_text"), str):
            raise RoundtripPackageError(f"条目 {item_id} 缺少字符串 edited_text。")
        original = str(raw.get("original_text", raw.get("original_fused_text", "")) or "")
        edited = str(raw.get("edited_text", "") or "")
        delete_intentionally = _explicit_bool(
            raw.get("delete_intentionally", False),
            field=f"条目 {item_id} 的 delete_intentionally",
        )
        if original.strip() and not edited.strip() and not delete_intentionally:
            raise RoundtripPackageError(
                f"条目 {item_id} 的 edited_text 为空。若确实要删除，请设置 delete_intentionally=true。"
            )
        mapping[item_id] = raw
    return mapping


def _document_from_dict_lossless(data: dict) -> UnifiedDocument:
    """Restore dynamic Metadata fields that the generic legacy loader ignores."""
    document = UnifiedDocument.from_dict(copy.deepcopy(data))
    raw_metadata = data.get("metadata") if isinstance(data, dict) else None
    if isinstance(raw_metadata, dict):
        for key, value in raw_metadata.items():
            setattr(document.metadata, str(key), copy.deepcopy(value))
    return document


def _rebind_assets(base: UnifiedDocument, current: UnifiedDocument | None) -> UnifiedDocument:
    """Prefer currently valid image paths without changing package geometry."""
    if current is None:
        return base
    page_by_no = {int(getattr(page, "page_no", 0) or 0): page for page in current.pages}
    for page in base.pages:
        candidate = page_by_no.get(int(getattr(page, "page_no", 0) or 0))
        if candidate is None:
            continue
        current_path = str(getattr(candidate, "image_path", "") or "")
        base_path = str(getattr(page, "image_path", "") or "")
        if current_path and (Path(current_path).exists() or not base_path or not Path(base_path).exists()):
            page.image_path = current_path
            page.width = int(getattr(candidate, "width", 0) or page.width)
            page.height = int(getattr(candidate, "height", 0) or page.height)

    current_by_id = {str(block.id): block for block in current.blocks if str(getattr(block, "id", "") or "")}
    current_images_by_page: dict[int, list[Block]] = {}
    for block in current.blocks:
        if block.type == BlockType.IMAGE_REF and str(block.image_path or ""):
            current_images_by_page.setdefault(int(getattr(block, "page", 0) or 0), []).append(block)
    for block in base.blocks:
        if block.type != BlockType.IMAGE_REF:
            continue
        candidate = current_by_id.get(str(block.id or ""))
        if candidate is None:
            candidates = current_images_by_page.get(int(getattr(block, "page", 0) or 0), [])
            candidate = next(
                (
                    value for value in candidates
                    if Path(str(value.image_path or "")).name == Path(str(block.image_path or "")).name
                ),
                candidates[0] if len(candidates) == 1 else None,
            )
        if candidate is None:
            continue
        current_path = str(candidate.image_path or "")
        base_path = str(block.image_path or "")
        if current_path and (Path(current_path).exists() or not base_path or not Path(base_path).exists()):
            block.image_path = current_path
    return base


def _validate_required_epub_assets(doc: UnifiedDocument) -> None:
    """Refuse silent cover/illustration loss when rebuilding the EPUB."""
    required: list[tuple[str, str]] = []
    for page in doc.pages:
        page_type = getattr(getattr(page, "page_type", None), "value", str(getattr(page, "page_type", "")))
        if page_type == BlockType.COVER.value:
            required.append(("封面", str(getattr(page, "image_path", "") or "")))
    for block in doc.blocks:
        if block.type in {BlockType.COVER, BlockType.IMAGE_REF, BlockType.ILLUSTRATION}:
            label = "封面" if block.type == BlockType.COVER else "插图"
            required.append((f"{label}块 {block.id}", str(getattr(block, "image_path", "") or "")))
    missing = []
    for label, path in required:
        if not path or not Path(path).is_file():
            missing.append(f"{label}: {path or '路径为空'}")
    if missing:
        preview = "；".join(missing[:3])
        suffix = f"（另有 {len(missing) - 3} 项）" if len(missing) > 3 else ""
        raise RoundtripPackageError(
            f"封面/插图资源文件不可用，已拒绝导入以避免 EPUB 漏图：{preview}{suffix}"
        )


def _sync_toc(doc: UnifiedDocument) -> None:
    """Update titles/indices without rebuilding or dropping special TOC rows."""
    if not doc.toc:
        return
    chapters_by_index: dict[int, list[tuple[int, Block]]] = {}
    for block_index, block in enumerate(doc.blocks):
        if block.type not in {BlockType.CHAPTER, BlockType.SECTION}:
            continue
        chapter_index = int(getattr(block, "chapter_index", 0) or 0)
        chapters_by_index.setdefault(chapter_index, []).append((block_index, block))
    for entry in doc.toc:
        resolved_index = int(getattr(entry, "block_index", -1) or -1)
        block = doc.blocks[resolved_index] if 0 <= resolved_index < len(doc.blocks) else None
        if block is None or block.type not in {BlockType.CHAPTER, BlockType.SECTION}:
            candidates = chapters_by_index.get(int(getattr(entry, "chapter_index", 0) or 0), [])
            if candidates:
                resolved_index, block = candidates[0]
        if block is not None:
            entry.block_index = resolved_index
            entry.title = str(block.text or entry.title)
            if int(getattr(block, "chapter_index", 0) or 0):
                entry.chapter_index = int(block.chapter_index)


def import_single_package(package: dict, *, current_document: UnifiedDocument | None = None) -> UnifiedDocument:
    _validate_common(
        package, MODE_SINGLE,
        validate_editable_structure=False,
        validate_immutable_manifest=False,
    )
    if current_document is not None and layout_hash(current_document) != str(package.get("layout_sha256", "") or ""):
        raise RoundtripPackageError("当前书与校对包的页面、块 ID 或坐标不一致，不能交叉导入。")
    structure = _document_from_dict_lossless(package["structure_document"])
    result = _rebind_assets(structure, current_document)
    item_map = _strict_item_map(package["editable_items"], "item_id")
    expected_ids: list[str] = []
    for block in result.blocks:
        if block.type not in _TEXT_TYPES or (block.metadata or {}).get("consumed"):
            continue
        item_id = f"block:{block.id}"
        expected_ids.append(item_id)
        item = item_map.get(item_id)
        if item is None:
            raise RoundtripPackageError(f"校对包缺少正文条目 {item_id}，为避免漏段已拒绝导入。")
        if str(item.get("block_id", "") or "") != str(block.id or ""):
            raise RoundtripPackageError(f"条目 {item_id} 的 block_id 不匹配。")
        block.text = str(item.get("edited_text", ""))
        if not block.ocr_raw:
            block.ocr_raw = str(item.get("original_text", "") or "")
        if block.text != str(item.get("original_text", "") or ""):
            block.modified_by = "external_ai_roundtrip"
            block.metadata = {
                **_safe_metadata(block.metadata),
                "external_ai_roundtrip": True,
                "external_ai_package_id": str(package.get("package_id", "") or ""),
            }
    extras = sorted(set(item_map) - set(expected_ids))
    if extras:
        raise RoundtripPackageError(f"校对包包含 {len(extras)} 个未知正文 ID；首个为 {extras[0]}。")
    if len(expected_ids) != len(item_map):
        raise RoundtripPackageError("正文条目数量与原始结构不一致。")
    _validate_editable_structure(package)
    _validate_immutable_manifest(package)
    _validate_required_epub_assets(result)
    _sync_toc(result)
    result.metadata.source_engine = f"{result.metadata.source_engine or 'ocr'}+external_ai_roundtrip"
    result.add_log("external_ai_roundtrip", f"严格导入单 OCR 校对包，共 {len(expected_ids)} 个正文块", len(expected_ids))
    return result


def _comparison_from_package(package: dict):
    from engine.multi_ocr_compare import (
        MultiOcrComparison,
        MultiOcrRow,
        _finalize_comparison,
        repair_adjacent_alignment_shifts,
    )

    comp_data = package.get("comparison") or {}
    labels = [str(value) for value in (package.get("model_labels") or comp_data.get("labels") or [])]
    items = list(package.get("editable_items") or [])
    rows: list[MultiOcrRow] = []
    for item in items:
        candidates = item.get("candidates") or []
        if not isinstance(candidates, list):
            raise RoundtripPackageError(f"行 {item.get('row_id')} 的 candidates 不是数组。")
        texts = [str(candidate.get("text", "") or "") for candidate in candidates if isinstance(candidate, dict)]
        raw_model_confidences = item.get("model_confidences") or [
            candidate.get("confidence", 0.0)
            for candidate in candidates if isinstance(candidate, dict)
        ]
        model_confidences = []
        for value in raw_model_confidences if isinstance(raw_model_confidences, (list, tuple)) else []:
            try:
                model_confidences.append(float(value or 0.0))
            except (TypeError, ValueError, OverflowError):
                model_confidences.append(0.0)
        rows.append(MultiOcrRow(
            index=int(item.get("row_index", len(rows)) or 0),
            texts=texts,
            model_confidences=tuple(model_confidences),
            chosen_index=int(item.get("recommended_model_index", 0) or 0),
            confidence=float(item.get("confidence", 0.0) or 0.0),
            reason=str(item.get("reason", "") or ""),
            warnings=tuple(str(value) for value in (item.get("warnings") or [])),
            primary_unit_index=item.get("primary_unit_index"),
            primary_block_index=item.get("primary_block_index"),
            primary_block_indices=tuple(int(value) for value in (item.get("primary_block_indices") or [])),
            primary_block_id=str(item.get("primary_block_id", "") or ""),
            primary_segment_index=int(item.get("primary_segment_index", 0) or 0),
            block_type=str(item.get("block_type", BlockType.PARAGRAPH.value) or BlockType.PARAGRAPH.value),
            page=int(item.get("page", 0) or 0),
            insert_before_block_index=item.get("insert_before_block_index"),
            column_ids=tuple(str(value) for value in (item.get("column_ids") or [])),
            atomic=bool(item.get("atomic", False)),
            alignment_repaired=bool(item.get("alignment_repaired", False)),
            alignment_notes=tuple(str(value) for value in (item.get("alignment_notes") or [])),
            alignment_status=str(item.get("alignment_status", "unreviewed") or "unreviewed"),
            sentence_group_id=str(item.get("sentence_group_id", "") or ""),
            repair_reason=str(item.get("repair_reason", "") or ""),
            consensus_seeded_models=tuple(
                int(value) for value in (item.get("consensus_seeded_models") or [])
            ),
            character_fused_text=str(item.get("character_fused_text", "") or ""),
            character_fusion_confidence=float(item.get("character_fusion_confidence", 0.0) or 0.0),
            character_fusion_reason=str(item.get("character_fusion_reason", "") or ""),
            character_fusion_warnings=tuple(str(value) for value in (item.get("character_fusion_warnings") or [])),
            character_fusion_auto_selected=bool(item.get("character_fusion_auto_selected", False)),
            local_reocr_recommended=bool(item.get("local_reocr_recommended", False)),
            character_fusion_evidence=copy.deepcopy(item.get("character_fusion_evidence") or {}),
        ))
    comparison = MultiOcrComparison(
        labels=labels,
        rows=rows,
        exact_rows=int(comp_data.get("exact_rows", 0) or 0),
        provisional_consensus_rows=int(comp_data.get("provisional_consensus_rows", 0) or 0),
        conflict_rows=int(comp_data.get("conflict_rows", 0) or 0),
        low_confidence_rows=int(comp_data.get("low_confidence_rows", 0) or 0),
        insertion_rows=int(comp_data.get("insertion_rows", 0) or 0),
        alignment_mode=str(comp_data.get("alignment_mode", "text_many_to_many") or "text_many_to_many"),
        column_anchored_rows=int(comp_data.get("column_anchored_rows", 0) or 0),
        chapter_atomic_rows=int(comp_data.get("chapter_atomic_rows", 0) or 0),
        alignment_shift_repairs=int(comp_data.get("alignment_shift_repairs", 0) or 0),
        unresolved_empty_cells=int(comp_data.get("unresolved_empty_cells", 0) or 0),
        alignment_revision=int(comp_data.get("alignment_revision", 1) or 1),
        physical_column_source=str(
            comp_data.get("physical_column_source", "source_column_texts")
            or "source_column_texts"
        ),
        true_empty_rows=int(comp_data.get("true_empty_rows", 0) or 0),
        single_model_only_rows=int(comp_data.get("single_model_only_rows", 0) or 0),
        character_fused_rows=int(comp_data.get("character_fused_rows", 0) or 0),
        character_auto_selected_rows=int(comp_data.get("character_auto_selected_rows", 0) or 0),
        local_reocr_rows=int(comp_data.get("local_reocr_rows", 0) or 0),
    )
    repairs = repair_adjacent_alignment_shifts(comparison)
    if repairs:
        comparison.alignment_revision = 2
        comparison.physical_column_source = "legacy_package_adjacent_shift_repair"
        _finalize_comparison(comparison)
    return comparison


def import_multi_package(
    package: dict,
    *,
    current_primary: UnifiedDocument | None = None,
) -> tuple[UnifiedDocument, object, list[str]]:
    _validate_common(
        package, MODE_MULTI,
        validate_editable_structure=False,
        validate_immutable_manifest=False,
    )
    if current_primary is not None and layout_hash(current_primary) != str(package.get("layout_sha256", "") or ""):
        raise RoundtripPackageError("当前多模型结构底稿与校对包不一致，不能交叉导入。")
    item_map = _strict_item_map(package["editable_items"], "row_id")
    items = list(package["editable_items"])
    expected_ids = [str(item.get("row_id", "") or "") for item in items]
    if len(expected_ids) != len(item_map):
        raise RoundtripPackageError("多模型行 ID 不唯一。")
    for expected_index, item in enumerate(items):
        if int(item.get("row_index", -1)) != expected_index:
            raise RoundtripPackageError(
                f"第 {expected_index + 1} 行的 row_index 被改变；为避免顺序错乱已拒绝导入。"
            )
    _validate_editable_structure(package)
    _validate_immutable_manifest(package)
    primary = _document_from_dict_lossless(package["structure_document"])
    primary = _rebind_assets(primary, current_primary)
    _validate_required_epub_assets(primary)
    comparison = _comparison_from_package(package)
    if len(comparison.rows) != len(items):
        raise RoundtripPackageError("比较行数与 editable_items 不一致。")
    result_lines: list[str] = []
    delete_flags: list[bool] = []
    for row, row_id in zip(comparison.rows, expected_ids):
        item = item_map[row_id]
        edited = str(item.get("edited_text", ""))
        original = str(item.get("original_fused_text", ""))
        delete_intentionally = _explicit_bool(
            item.get("delete_intentionally", False),
            field=f"条目 {row_id} 的 delete_intentionally",
        )
        # Legacy packages produced before physical-column alignment revision 2
        # may contain an untouched duplicated sentence caused by a one-row
        # model shift.  Use the repaired candidate only when the user has not
        # changed the exported text; explicit external edits always win.
        if row.alignment_repaired and edited == original and not delete_intentionally:
            result_lines.append(str(row.output_text or ""))
        else:
            result_lines.append(edited)
        delete_flags.append(delete_intentionally)
    from engine.multi_ocr_compare import build_fused_document
    result = build_fused_document(
        primary,
        comparison,
        result_lines,
        delete_flags=delete_flags,
    )
    result.metadata.source_engine = f"{primary.metadata.source_engine or 'ocr'}+external_ai_multi_fusion"
    result.add_log("external_ai_multi_fusion", f"严格导入多模型融合包，共 {len(items)} 句", len(items))
    return result, comparison, result_lines


def repair_multi_package_alignment(package: dict) -> tuple[dict, dict]:
    """Upgrade a legacy multi-model package without changing row/column IDs.

    Only conservative adjacent one-row shifts are repaired.  Candidate texts,
    hashes, confidence and default fused text are resealed.  A user-modified
    ``edited_text`` is never replaced; untouched defaults follow the repaired
    candidate so importing the upgraded package cannot duplicate the shifted
    sentence.
    """
    _validate_common(
        package, MODE_MULTI,
        validate_editable_structure=True,
        validate_immutable_manifest=True,
    )
    repaired = copy.deepcopy(package)
    existing_comparison = repaired.get("comparison") or {}
    existing_operations = int(existing_comparison.get("alignment_shift_repairs", 0) or 0)
    comparison = _comparison_from_package(repaired)
    items = list(repaired.get("editable_items") or [])
    existing_repaired_flags = [bool(item.get("alignment_repaired", False)) for item in items]
    if len(items) != len(comparison.rows):
        raise RoundtripPackageError("比较行数与 editable_items 不一致，无法修复。")

    repaired_rows = 0
    preserved_manual_edits = 0
    changed_defaults = 0
    for item_index, (item, row) in enumerate(zip(items, comparison.rows)):
        # Audit metadata is refreshed for every row. It does not affect row IDs,
        # column IDs, structure hashes, or user-edited text.
        item["alignment_repaired"] = bool(row.alignment_repaired)
        item["alignment_notes"] = [str(value) for value in (row.alignment_notes or ())]
        item["alignment_status"] = str(row.alignment_status or "unreviewed")
        item["sentence_group_id"] = str(row.sentence_group_id or "")
        item["repair_reason"] = str(row.repair_reason or "")
        item["character_fused_text"] = str(row.character_fused_text or "")
        item["character_fusion_confidence"] = float(row.character_fusion_confidence or 0.0)
        item["character_fusion_reason"] = str(row.character_fusion_reason or "")
        item["character_fusion_warnings"] = [str(value) for value in (row.character_fusion_warnings or ())]
        item["character_fusion_auto_selected"] = bool(row.character_fusion_auto_selected)
        item["local_reocr_recommended"] = bool(row.local_reocr_recommended)
        item["character_fusion_evidence"] = copy.deepcopy(row.character_fusion_evidence or {})
        item["recommended_model_index"] = int(row.chosen_index)
        item["confidence"] = float(row.confidence)
        item["reason"] = str(row.reason or "")
        item["warnings"] = [str(value) for value in (row.warnings or ())]
        if not row.alignment_repaired:
            continue
        if not existing_repaired_flags[item_index]:
            repaired_rows += 1
        old_original = str(item.get("original_fused_text", "") or "")
        old_edited = str(item.get("edited_text", "") or "")
        candidates = item.get("candidates") or []
        if not isinstance(candidates, list):
            candidates = []
            item["candidates"] = candidates
        while len(candidates) < len(row.texts):
            model_index = len(candidates)
            labels = list(comparison.labels or [])
            candidates.append({
                "model_index": model_index,
                "model_label": labels[model_index] if model_index < len(labels) else f"模型{model_index + 1}",
            })
        for model_index, text in enumerate(row.texts):
            candidate = candidates[model_index]
            if not isinstance(candidate, dict):
                candidate = {"model_index": model_index}
                candidates[model_index] = candidate
            value = str(text or "")
            candidate["text"] = value
            candidate["text_sha256"] = _sha256(value.encode("utf-8"))
        new_original = str(row.output_text or "")
        item["original_fused_text"] = new_original
        if old_edited == old_original:
            item["edited_text"] = new_original
            if new_original != old_edited:
                changed_defaults += 1
        else:
            preserved_manual_edits += 1

    total_repaired_rows = sum(1 for row in comparison.rows if row.alignment_repaired)
    total_operations = int(comparison.alignment_shift_repairs)
    report = {
        "alignment_revision": 2,
        "repair_kind": "adjacent_physical_column_shift",
        "repaired_rows": repaired_rows,
        "repair_operations": max(0, total_operations - existing_operations),
        "total_repaired_rows": total_repaired_rows,
        "total_repair_operations": total_operations,
        "changed_untouched_defaults": changed_defaults,
        "preserved_manual_edits": preserved_manual_edits,
        "remaining_empty_cells": int(comparison.unresolved_empty_cells),
        "true_empty_rows": int(comparison.true_empty_rows),
        "single_model_only_rows": int(comparison.single_model_only_rows),
    }
    repaired["comparison"] = _comparison_payload(comparison)
    repaired["alignment_repair_report"] = report
    repaired["editable_structure_sha256"] = _editable_structure_hash(items)
    repaired.pop("immutable_manifest_sha256", None)
    _seal_package(repaired)
    return repaired, report


def package_to_json(package: dict, *, indent: int = 2) -> str:
    return json.dumps(package, ensure_ascii=False, indent=indent) + "\n"


def _markdown_fence(text: str) -> str:
    longest = max((len(match.group(0)) for match in re.finditer(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def package_to_markdown(package: dict) -> str:
    _validate_common(package)
    manifest = copy.deepcopy(package)
    # The human-readable sections below are authoritative for edited_text.
    # Keeping edited text in the manifest too makes an untouched MD round-trip
    # exact and allows recovery if a tool strips only the prose around it.
    packed = base64.b64encode(zlib.compress(_json_bytes(manifest), level=9)).decode("ascii")
    title = str((package.get("book") or {}).get("title", "") or "OCR 校对包")
    mode_text = "多模型 OCR 对比融合" if package.get("mode") == MODE_MULTI else "单 OCR 原格式校对"
    lines = [
        f"# {title} · {mode_text}",
        "",
        "> 只修改每个 `NFRT_EDIT_BEGIN / NFRT_EDIT_END` 之间的文字。",
        "> 确实要删除整条非空正文时，将编辑区内容改为 `<!-- NFRT_DELETE_INTENTIONALLY -->`。",
        "> 不要删除、复制、重排标记，也不要修改 ID、坐标、封面、插图或结构清单。",
        "",
        f"<!-- NFRT_MANIFEST_ZLIB_BASE64:{packed} -->",
        "",
    ]
    for item in package.get("editable_items", []):
        item_id = str(item.get("row_id") or item.get("item_id") or "")
        if package.get("mode") == MODE_MULTI:
            lines.extend([
                f"## 句 {int(item.get('row_index', 0)) + 1} · `{item_id}`",
                f"- 页码：{int(item.get('page', 0) or 0)}",
                f"- 类型：{item.get('block_type', '')}",
                f"- 列 ID：{', '.join(item.get('column_ids') or []) or '无'}",
            ])
            for candidate in item.get("candidates", []):
                text = str(candidate.get("text", "") or "")
                fence = _markdown_fence(text)
                lines.extend([
                    f"### {candidate.get('model_label', '模型候选')}",
                    fence + "text",
                    text,
                    fence,
                ])
        else:
            lines.extend([
                f"## 块 {int(item.get('block_order', 0)) + 1} · `{item_id}`",
                f"- 页码：{int(item.get('page', 0) or 0)}",
                f"- 类型：{item.get('block_type', '')}",
                f"- 坐标：`{json.dumps(item.get('bbox'), ensure_ascii=False)}`",
                f"- 列 ID：{', '.join(item.get('source_column_ids') or []) or '无'}",
            ])
            original = str(item.get("original_text", "") or "")
            fence = _markdown_fence(original)
            lines.extend(["### 原始 OCR", fence + "text", original, fence])
        edited_value = str(item.get("edited_text", "") or "")
        if _explicit_bool(
            item.get("delete_intentionally", False),
            field=f"条目 {item_id} 的 delete_intentionally",
        ) and not edited_value:
            edited_value = _MD_DELETE_MARKER
        lines.extend([
            "### 大模型校对结果（只改这里）",
            f"<!-- NFRT_EDIT_BEGIN {item_id} -->",
            edited_value,
            f"<!-- NFRT_EDIT_END {item_id} -->",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def _loose_json(text: str) -> dict:
    raw = str(text or "").lstrip("\ufeff").strip()
    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL | re.IGNORECASE)
    if fenced:
        value = json.loads(fenced.group(1))
        if isinstance(value, dict):
            return value
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", raw):
        try:
            value, _end = decoder.raw_decode(raw[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("schema") == SCHEMA:
            return value
    raise RoundtripPackageError("没有找到有效的 Novel Formatter OCR 校对包 JSON。")


def package_from_markdown(text: str) -> dict:
    match = _MD_MANIFEST_RE.search(str(text or ""))
    if not match:
        raise RoundtripPackageError("Markdown 缺少 NFRT 结构清单，无法安全恢复封面和坐标。")
    try:
        encoded = re.sub(r"\s+", "", match.group(1))
        manifest = json.loads(zlib.decompress(base64.b64decode(encoded)).decode("utf-8"))
    except Exception as exc:
        raise RoundtripPackageError(f"Markdown 结构清单损坏：{exc}") from exc
    _validate_common(manifest)
    edits: dict[str, str] = {}
    for item_id, value in _MD_EDIT_RE.findall(str(text or "")):
        key = str(item_id or "")
        if key in edits:
            raise RoundtripPackageError(f"Markdown 中重复出现校对 ID：{key}")
        edits[key] = value
    id_key = "row_id" if manifest.get("mode") == MODE_MULTI else "item_id"
    expected = [str(item.get(id_key, "") or "") for item in manifest.get("editable_items", [])]
    missing = [item_id for item_id in expected if item_id not in edits]
    extras = [item_id for item_id in edits if item_id not in set(expected)]
    if missing:
        raise RoundtripPackageError(f"Markdown 缺少 {len(missing)} 个校对段；首个为 {missing[0]}。")
    if extras:
        raise RoundtripPackageError(f"Markdown 包含未知校对段：{extras[0]}。")
    for item in manifest.get("editable_items", []):
        value = edits[str(item.get(id_key, "") or "")]
        if value.strip() == _MD_DELETE_MARKER:
            item["edited_text"] = ""
            item["delete_intentionally"] = True
        else:
            item["edited_text"] = value
            item["delete_intentionally"] = False
    return manifest


def load_package(path: str | Path) -> dict:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"文件不存在：{source}")
    text = source.read_text(encoding="utf-8-sig", errors="strict")
    if source.suffix.lower() in {".md", ".markdown"}:
        return package_from_markdown(text)
    return _loose_json(text)


def save_package(package: dict, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() in {".md", ".markdown"}:
        text = package_to_markdown(package)
    else:
        if target.suffix.lower() != ".json":
            target = target.with_suffix(".json")
        text = package_to_json(package)
    temp = target.with_name(f".{target.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        temp.write_text(text, encoding="utf-8")
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)
    return target
