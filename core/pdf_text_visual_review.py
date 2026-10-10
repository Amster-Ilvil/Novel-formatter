#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Targeted visual review helpers for damaged native PDF text layers.

The native text layer remains the source of truth for geometry and reading
order.  This module only targets explicit U+FFFD replacement glyphs whose
Unicode mapping is already broken in the source PDF.

Phase30 adds a second, safer acceleration layer: when MuPDF exposes the same
embedded-font name + glyph id for repeated U+FFFD positions, those positions
can be reviewed as one glyph cluster and expanded back to stable item ids.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Mapping


def collect_pdf_text_replacement_items(doc) -> list[dict]:
    """Return stable review items from replacement-glyph metadata."""
    items: list[dict] = []
    for block_index, block in enumerate(getattr(doc, "blocks", []) or []):
        meta = dict(getattr(block, "metadata", {}) or {})
        physical_page = int(meta.get("pdf_source_physical_page", 0) or 0)
        part = str(meta.get("pdf_logical_part", "full") or "full")
        text = str(getattr(block, "text", "") or "")
        for raw in list(meta.get("pdf_text_replacement_glyphs") or []):
            try:
                text_index = int(raw.get("text_index"))
                bbox = [float(v) for v in raw.get("source_bbox", [])]
            except Exception:
                continue
            if len(bbox) != 4 or text_index < 0 or text_index >= len(text):
                continue
            if text[text_index] != "\ufffd":
                continue
            item_id = f"P{physical_page:04d}-{part}-B{block_index:05d}-C{text_index:04d}"
            left = max(0, text_index - 12)
            right = min(len(text), text_index + 13)
            item = {
                "item_id": item_id,
                "block_index": block_index,
                "logical_page": int(getattr(block, "page", 0) or 0),
                "physical_page": physical_page,
                "part": part,
                "text_index": text_index,
                "source_bbox": bbox,
                "context": text[left:right],
                "context_target_offset": text_index - left,
            }
            pdf_font = str(raw.get("pdf_font", "") or "")
            glyph_id = raw.get("pdf_glyph_id")
            glyph_key = str(raw.get("pdf_glyph_key", "") or "")
            if pdf_font and glyph_id is not None:
                item["pdf_font"] = pdf_font
                item["pdf_glyph_id"] = int(glyph_id)
                item["pdf_glyph_key"] = glyph_key or f"{pdf_font}:{int(glyph_id)}"
            items.append(item)
    return items


def collect_pdf_text_glyph_clusters(doc) -> list[dict]:
    """Group repeated broken glyphs by embedded font + glyph id.

    Cluster ids are deterministic for one document snapshot.  Items without a
    glyph identity are intentionally excluded and must still be reviewed at the
    per-item level.
    """
    grouped: dict[str, list[dict]] = {}
    for item in collect_pdf_text_replacement_items(doc):
        key = str(item.get("pdf_glyph_key", "") or "")
        if not key:
            continue
        grouped.setdefault(key, []).append(item)

    clusters: list[dict] = []
    for ordinal, key in enumerate(sorted(grouped), start=1):
        members = grouped[key]
        first = members[0]
        clusters.append({
            "cluster_id": f"G{ordinal:04d}",
            "pdf_glyph_key": key,
            "pdf_font": first.get("pdf_font", ""),
            "pdf_glyph_id": int(first.get("pdf_glyph_id", 0) or 0),
            "count": len(members),
            "representative_item_id": first["item_id"],
            "member_item_ids": [item["item_id"] for item in members],
            "context_samples": [item["context"] for item in members[:5]],
        })
    return clusters


def expand_pdf_text_glyph_corrections(doc, glyph_corrections: Mapping[str, str]) -> dict[str, str]:
    """Expand cluster-level corrections to stable per-item corrections."""
    clusters = collect_pdf_text_glyph_clusters(doc)
    by_cluster_id = {row["cluster_id"]: row for row in clusters}
    by_glyph_key = {row["pdf_glyph_key"]: row for row in clusters}
    expanded: dict[str, str] = {}
    for raw_key, raw_replacement in dict(glyph_corrections or {}).items():
        key = str(raw_key)
        replacement = str(raw_replacement or "")
        if len(replacement) != 1:
            continue
        cluster = by_cluster_id.get(key) or by_glyph_key.get(key)
        if cluster is None:
            continue
        for item_id in cluster["member_item_ids"]:
            expanded[str(item_id)] = replacement
    return expanded


def _adjust_pdf_guard_baseline_for_repairs(target, applied_rows: list[tuple[int, str, str]]) -> None:
    """Keep the character guard truthful after explicit visual adjudication.

    ``ocr_raw`` deliberately stays untouched, but a repaired U+FFFD is no longer
    the expected publication character.  If a guard baseline already exists,
    replace the corresponding U+FFFD counts with the reviewed characters.
    When the guard is created later, ``prepare_pdf_text_layer`` also recognizes
    ``pdf_visual_repair_applied`` and uses the repaired block text as baseline.
    """
    metadata = getattr(target, "metadata", None)
    if metadata is None:
        return
    counts = Counter(getattr(metadata, "pdf_text_source_char_counts", {}) or {})
    if not counts:
        return
    for _index, replacement, _item_id in applied_rows:
        if counts.get("\ufffd", 0) > 0:
            counts["\ufffd"] -= 1
            if counts["\ufffd"] <= 0:
                counts.pop("\ufffd", None)
            counts[replacement] += 1
    metadata.pdf_text_source_char_counts = dict(counts)
    metadata.pdf_text_source_chars = sum(counts.values())


def apply_pdf_text_visual_repairs(doc, corrections: Mapping[str, str], *, copy_document: bool = True):
    """Apply GPT/OCR-reviewed replacements only at verified U+FFFD positions.

    ``corrections`` maps stable ``item_id`` values returned by
    :func:`collect_pdf_text_replacement_items` to replacement strings. Empty
    replacements are ignored.  The function refuses to modify positions that no
    longer contain U+FFFD, preventing stale review packages from corrupting a
    newer document snapshot.
    """
    target = doc.snapshot_clone() if copy_document else doc
    items = {item["item_id"]: item for item in collect_pdf_text_replacement_items(target)}
    by_block: dict[int, list[tuple[int, str, str]]] = {}
    skipped: list[str] = []
    for item_id, replacement in dict(corrections or {}).items():
        item = items.get(str(item_id))
        repl = str(replacement or "")
        if item is None or len(repl) != 1:
            skipped.append(str(item_id))
            continue
        by_block.setdefault(int(item["block_index"]), []).append(
            (int(item["text_index"]), repl, str(item_id))
        )

    applied: list[str] = []
    applied_rows: list[tuple[int, str, str]] = []
    for block_index, edits in by_block.items():
        block = target.blocks[block_index]
        text = str(block.text or "")
        local_applied: list[str] = []
        # Reverse order keeps source indices stable even if this rule is relaxed
        # in the future.  Phase30 still requires exactly one replacement char.
        for text_index, replacement, item_id in sorted(edits, key=lambda row: row[0], reverse=True):
            if text_index < 0 or text_index >= len(text) or text[text_index] != "\ufffd":
                skipped.append(item_id)
                continue
            text = text[:text_index] + replacement + text[text_index + 1:]
            applied.append(item_id)
            local_applied.append(item_id)
            applied_rows.append((text_index, replacement, item_id))
        block.text = text
        # ocr_raw intentionally remains the untouched source text-layer string.
        meta = dict(getattr(block, "metadata", {}) or {})
        if local_applied:
            meta["pdf_visual_repair_applied"] = sorted(set(
                list(meta.get("pdf_visual_repair_applied") or []) + local_applied
            ))
        block.metadata = meta

    _adjust_pdf_guard_baseline_for_repairs(target, applied_rows)

    metadata = getattr(target, "metadata", None)
    if metadata is not None:
        remaining = sum(str(getattr(block, "text", "") or "").count("\ufffd") for block in target.blocks)
        metadata.pdf_text_remaining_replacement_glyph_count = remaining
        metadata.pdf_text_visual_repair_report = {
            "requested": len(dict(corrections or {})),
            "applied": len(applied),
            "skipped": len(skipped),
            "remaining": remaining,
            "applied_item_ids": sorted(applied),
            "skipped_item_ids": sorted(set(skipped)),
        }
    return target


def apply_pdf_text_visual_review_payload(doc, payload: Mapping, *, copy_document: bool = True):
    """Apply either v1 item corrections or Phase30 cluster corrections."""
    data = dict(payload or {})
    corrections = data.get("corrections")
    if corrections is None and "glyph_corrections" in data:
        corrections = expand_pdf_text_glyph_corrections(doc, data.get("glyph_corrections") or {})
    if corrections is None:
        # Backward-compatible bare {item_id: char} payload.
        corrections = data
    if not isinstance(corrections, dict):
        raise ValueError("JSON 必须包含 corrections 或 glyph_corrections 字典")
    return apply_pdf_text_visual_repairs(doc, corrections, copy_document=copy_document)
