#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vertical OCR canonicalization for text replacement.

This module builds a read-only logical paragraph layer from physical OCR
blocks. It intentionally does not mutate the source UnifiedDocument.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import unicodedata

from models.document import UnifiedDocument, Block, BlockType
from engine.japanese_normalizer import compare_key


TEXT_TYPES = {BlockType.PARAGRAPH, BlockType.DIALOGUE, BlockType.CHAPTER, BlockType.SECTION, BlockType.RUBY}
TERMINAL = "。！？!?」』）】〉》"
OPEN_TO_CLOSE = {"「": "」", "『": "』", "（": "）", "(": ")", "【": "】", "〈": "〉", "《": "》"}
CONTINUATION_ENDINGS = (
    "そして", "その", "この", "ため", "こと", "もの", "ので", "から", "ながら",
    "という", "として", "して", "れて", "せて", "には", "では", "なら", "まで",
)


@dataclass(frozen=True)
class BlockRef:
    block_index: int
    page: int = 0
    page_index: int | None = None
    order_in_page: int | None = None


@dataclass
class LogicalParagraph:
    display_text: str
    match_text: str
    block_refs: list[BlockRef] = field(default_factory=list)
    page_start: int = 0
    page_end: int = 0
    confidence: float = 1.0
    is_title: bool = False

    @property
    def text(self) -> str:
        return self.display_text

    @property
    def index(self) -> int:
        return self.block_refs[0].block_index if self.block_refs else -1


@dataclass
class MatchDocument:
    logical_paragraphs: list[LogicalParagraph]
    physical_blocks: list[Block]
    is_vertical: bool = False


def normalize_for_alignment(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("﹁", "「").replace("﹂", "」").replace("『", "「").replace("』", "」")
    text = re.sub(r"[―─—–−ｰー]{2,}", "ー", text)
    text = text.replace("―", "ー").replace("─", "ー").replace("—", "ー")
    text = text.replace("ハツキリ", "ハッキリ")
    return compare_key(re.sub(r"[\s　]+", "", text))


def _metadata_label(block: Block) -> str:
    meta = block.metadata or {}
    for key in ("block_label", "label", "type_label"):
        value = meta.get(key)
        if value:
            return str(value)
    return ""


def _metadata_bbox(block: Block):
    if block.bbox is not None:
        return block.bbox.x, block.bbox.y, block.bbox.w, block.bbox.h
    raw = (block.metadata or {}).get("bbox") or (block.metadata or {}).get("block_bbox")
    if isinstance(raw, dict):
        x = float(raw.get("x", raw.get("left", 0)) or 0)
        y = float(raw.get("y", raw.get("top", 0)) or 0)
        w = float(raw.get("w", raw.get("width", 0)) or 0)
        h = float(raw.get("h", raw.get("height", 0)) or 0)
        return x, y, w, h
    if isinstance(raw, (list, tuple)) and len(raw) >= 4:
        x1, y1, x2, y2 = [float(v or 0) for v in raw[:4]]
        return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)
    return 0.0, 0.0, 0.0, 0.0


def _is_vertical_block(block: Block) -> bool:
    label = _metadata_label(block)
    if label == "vertical_text":
        return True
    if label and label != "text":
        return False
    if block.text_direction and "vertical" in block.text_direction:
        return True
    _, _, w, h = _metadata_bbox(block)
    return bool(h and w and h > w * 1.8)


def _sort_key(block_with_index: tuple[int, Block], vertical: bool):
    _, block = block_with_index
    x, y, w, h = _metadata_bbox(block)
    x_center = x + w / 2
    y_top = y
    if not (w or h):
        return (block.order_in_page if block.order_in_page is not None else block.reading_order, 0)
    return (-x_center, y_top) if vertical else (y_top, x)


def _has_unclosed_bracket(text: str) -> bool:
    stack: list[str] = []
    closes = {v: k for k, v in OPEN_TO_CLOSE.items()}
    for ch in text:
        if ch in OPEN_TO_CLOSE:
            stack.append(ch)
        elif ch in closes and stack and stack[-1] == closes[ch]:
            stack.pop()
    return bool(stack)


def _is_complete(text: str, next_block: Block | None = None) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if _has_unclosed_bracket(stripped):
        return False
    if any(stripped.endswith(suffix) for suffix in CONTINUATION_ENDINGS):
        return False
    if stripped[-1] in TERMINAL:
        return True
    if next_block is not None and next_block.type in (BlockType.CHAPTER, BlockType.SECTION):
        return True
    return False


def _join_display(parts: list[str]) -> str:
    return re.sub(r"[\s　]+", "", "".join(p.strip() for p in parts if p and p.strip()))


def _normalized_piece(text: str) -> str:
    return re.sub(r"[\s　]+", "", (text or "").strip())


def _append_without_overlap(parts: list[str], text: str) -> None:
    """向逻辑段落追加物理块，同时消费 OCR 重扫造成的重复/重叠。"""
    piece = _normalized_piece(text)
    if not piece:
        return
    if not parts:
        parts.append(piece)
        return

    current = _join_display(parts)
    if piece == current or current.endswith(piece):
        return
    if piece.startswith(current):
        parts[:] = [piece]
        return

    # 最长公共后缀/前缀：ABCDEF + DEFG -> ABCDEFG
    max_overlap = min(len(current), len(piece))
    overlap = 0
    for size in range(max_overlap, 7, -1):
        if current[-size:] == piece[:size]:
            overlap = size
            break
    if overlap:
        parts.append(piece[overlap:])
    else:
        parts.append(piece)


class VerticalOCRCanonicalizer:
    """Build logical horizontal paragraphs from physical OCR blocks."""

    def is_applicable(self, document: UnifiedDocument) -> bool:
        text_blocks = [b for b in document.blocks if b.type in TEXT_TYPES and b.text.strip()]
        if not text_blocks:
            return False
        vertical = sum(1 for b in text_blocks if _is_vertical_block(b))
        return vertical >= max(1, len(text_blocks) // 4)

    def build_logical_document(self, document: UnifiedDocument) -> MatchDocument:
        indexed = [(i, b) for i, b in enumerate(document.blocks) if b.type in TEXT_TYPES and b.text.strip()]
        by_page: dict[int, list[tuple[int, Block]]] = {}
        for item in indexed:
            _, block = item
            page_key = block.page_index if block.page_index is not None else block.page
            by_page.setdefault(page_key or 0, []).append(item)

        ordered: list[tuple[int, Block]] = []
        page_modes: dict[int, bool] = {}
        for page in sorted(by_page):
            blocks = by_page[page]
            vertical_count = sum(1 for _, b in blocks if _is_vertical_block(b))
            is_vertical = vertical_count >= max(1, len(blocks) // 2)
            page_modes[page] = is_vertical
            ordered.extend(sorted(blocks, key=lambda item: _sort_key(item, is_vertical)))

        logical: list[LogicalParagraph] = []
        buf_texts: list[str] = []
        buf_refs: list[BlockRef] = []
        buf_conf: list[float] = []

        def flush():
            nonlocal buf_texts, buf_refs, buf_conf
            display = _join_display(buf_texts)
            if display and buf_refs:
                match_text = normalize_for_alignment(display)
                # 两个完整 OCR 块都以终止符结束时，第一块会先 flush。固定排版
                # 模式下必须在逻辑段落提交阶段再次检查等长重复，并把物理引用
                # 合并到上一逻辑段，供 replacement 在写回时消费重复块。
                if logical and not logical[-1].is_title and logical[-1].match_text == match_text:
                    logical[-1].block_refs.extend(buf_refs)
                    logical[-1].page_end = buf_refs[-1].page
                    logical[-1].confidence = min(logical[-1].confidence, sum(buf_conf) / len(buf_conf) if buf_conf else 1.0)
                else:
                    logical.append(LogicalParagraph(
                        display_text=display,
                        match_text=match_text,
                        block_refs=list(buf_refs),
                        page_start=buf_refs[0].page,
                        page_end=buf_refs[-1].page,
                        confidence=sum(buf_conf) / len(buf_conf) if buf_conf else 1.0,
                        is_title=False,
                    ))
            buf_texts, buf_refs, buf_conf = [], [], []

        for pos, (block_index, block) in enumerate(ordered):
            if block.type in (BlockType.CHAPTER, BlockType.SECTION):
                flush()
                text = block.text.strip()
                logical.append(LogicalParagraph(
                    display_text=text,
                    match_text=normalize_for_alignment(text),
                    block_refs=[BlockRef(block_index, block.page, block.page_index, block.order_in_page)],
                    page_start=block.page,
                    page_end=block.page,
                    confidence=block.confidence,
                    is_title=True,
                ))
                continue

            _append_without_overlap(buf_texts, block.text)
            buf_refs.append(BlockRef(block_index, block.page, block.page_index, block.order_in_page))
            buf_conf.append(block.confidence)
            next_block = ordered[pos + 1][1] if pos + 1 < len(ordered) else None
            if _is_complete(_join_display(buf_texts), next_block):
                flush()

        flush()
        return MatchDocument(
            logical_paragraphs=logical,
            physical_blocks=document.blocks,
            is_vertical=any(page_modes.values()),
        )
