#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI helpers for the editable text-comparison workspace.

The compare UI keeps the OCR/Formatter structure separate from the editable
right-hand text.  This module converts a selected row range into a tiny
``UnifiedDocument`` suitable for the existing compact AI patch protocol and
converts the result back to line records.  Illustration marker rows are never
sent to the model.
"""
from __future__ import annotations

import copy
from typing import Sequence

from engine.text_compare import CompareLine, document_lines, parse_image_marker
from models.document import Block, BlockType, UnifiedDocument


COMPARE_AI_TYPESET_PROMPT = """Proofread and typeset Japanese light-novel OCR while preserving facts, names, voice and order.
Input is one ordered part: {"b":[[id,type,text],...],"g":0}. Types: p=paragraph,d=dialogue,c=chapter,s=section,r=ruby,f=footnote,t=toc,u=other.
Fix clear OCR character errors, duplicated/missing characters, punctuation and obvious grammar. Repair broken sentences, merge or split blocks only when the boundary is unambiguous, place each complete dialogue turn on its own block, and separate dialogue from narration.
Use conservative reconstruction: never invent events, explanations, names or long missing passages. Never reorder clauses or sentences, paraphrase the author's wording, flatten expressive punctuation, change the speaker, or move text between unrelated paragraphs. Keep source-id ownership stable; merge/split only when a visible broken boundary makes the operation unambiguous. For a source range of 40 or more Japanese characters, omit the operation if the corrected text would change more than roughly 15% of non-space characters. If a long passage would change substantially, leave it unchanged for manual review. Do not merge across a chapter or illustration boundary. If the original cannot be uniquely recovered, leave it unchanged. Preserve chapter order and every source block.
Return ONLY changed contiguous ranges as compact JSON: {"o":[[[source_ids...],[[type,text],...]],...]}. Each operation replaces exactly those contiguous source ids; operations must not overlap. Omit unchanged ranges. If nothing changes return {"o":[]}.
No explanations, Markdown, stylesheet or unchanged text. Every replacement text must be non-empty.
INPUT:\n{{INPUT}}"""


def _coerce_type(value: str, text: str) -> BlockType:
    try:
        kind = BlockType(str(value or "paragraph"))
    except Exception:
        kind = BlockType.PARAGRAPH
    stripped = (text or "").strip()
    if stripped.startswith("「") and stripped.endswith("」"):
        return BlockType.DIALOGUE
    if kind in {BlockType.CHAPTER, BlockType.SECTION, BlockType.TOC_ENTRY,
                BlockType.RUBY, BlockType.FOOTNOTE, BlockType.DIALOGUE}:
        return kind
    return BlockType.PARAGRAPH


def records_to_ai_document(records: Sequence[CompareLine], *, row_offset: int = 0) -> UnifiedDocument:
    """Build a compact document from editable rows.

    Image marker rows are protected structural anchors.  A local AI operation
    spanning one is rejected so a model can never relocate or remove an image.
    Empty alignment rows are omitted.
    """
    doc = UnifiedDocument()
    chapter_index = 0
    for local_index, record in enumerate(records):
        text = str(record.text or "")
        if parse_image_marker(text):
            raise ValueError("所选范围包含插图标记。请缩小选择范围，不要让 AI 跨越插图位置。")
        if not text.strip():
            continue
        kind = _coerce_type(record.block_type, text)
        if kind == BlockType.CHAPTER:
            chapter_index += 1
        block = Block(
            type=kind,
            text=text,
            page=int(record.page or 0),
            chapter_index=chapter_index,
            id=f"compare_ai_row_{row_offset + local_index:08d}",
        )
        block.metadata = dict(block.metadata or {})
        block.metadata["compare_source_row"] = row_offset + local_index
        block.metadata["compare_original_ids"] = list(record.block_ids or [])
        doc.blocks.append(block)
    for index, block in enumerate(doc.blocks):
        block.reading_order = index
    return doc


def ai_document_to_records(doc: UnifiedDocument) -> list[CompareLine]:
    """Return editable text rows from an AI result, excluding assets."""
    return [copy.deepcopy(line) for line in document_lines(doc)
            if not parse_image_marker(line.text)]


def splice_records(
    original: Sequence[CompareLine],
    start_row: int,
    end_row: int,
    replacement: Sequence[CompareLine],
) -> list[CompareLine]:
    """Replace one inclusive row range while preserving all other row records."""
    if start_row < 0 or end_row < start_row or end_row >= len(original):
        raise IndexError("AI 局部处理范围超出文本行数")
    return (
        [copy.deepcopy(item) for item in original[:start_row]]
        + [copy.deepcopy(item) for item in replacement]
        + [copy.deepcopy(item) for item in original[end_row + 1:]]
    )
