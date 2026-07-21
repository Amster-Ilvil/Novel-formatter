#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR 输出配准引擎（专用版）
功能：用 PaddleOCR 的识别文本替换 Mac OCR 的文本，保留 Mac 的版式（块结构/坐标/分页）
架构：按页分块 → 字符级索引映射 → 局部序列对齐 → 按块回填
不依赖：段落相似度、阈值、Needleman-Wunsch、章节检测
"""

from __future__ import annotations

import difflib
import copy
from collections import defaultdict
from typing import List, Tuple, Dict

from models.document import UnifiedDocument, Block, BlockType


class OcrAligner:
    """两个 OCR 输出之间的字符级配准器"""

    def __init__(self, mac_doc: UnifiedDocument, paddle_doc: UnifiedDocument):
        self.mac_doc = mac_doc
        self.paddle_doc = paddle_doc
        self.TEXT_TYPES = {BlockType.PARAGRAPH, BlockType.DIALOGUE, BlockType.CHAPTER, BlockType.SECTION}

    def align(self) -> Tuple[UnifiedDocument, dict]:
        """
        执行配准并返回新文档 + 统计报告
        """
        # 1. 按页分组
        mac_pages = self._group_blocks_by_page(self.mac_doc)
        paddle_pages = self._group_blocks_by_page(self.paddle_doc)

        new_doc = copy.deepcopy(self.mac_doc)
        replaced_count = 0
        total_text_blocks = 0

        for page, mac_blocks in mac_pages.items():
            if page not in paddle_pages:
                continue  # 该页没有 Paddle 结果，保留 Mac 原文

            paddle_blocks = paddle_pages[page]

            # 按阅读顺序排序（优先用 reading_order，否则按 y 坐标）
            mac_blocks = self._sort_blocks(mac_blocks)
            paddle_blocks = self._sort_blocks(paddle_blocks)

            # 过滤空文本
            mac_blocks = [b for b in mac_blocks if b.text.strip()]
            paddle_blocks = [b for b in paddle_blocks if b.text.strip()]

            if not mac_blocks or not paddle_blocks:
                continue

            # 2. 构建文本流 + 字符→块索引的映射表
            mac_text, mac_char_owner = self._build_text_stream(mac_blocks)
            paddle_text, _ = self._build_text_stream(paddle_blocks)

            if not mac_text or not paddle_text:
                continue

            # 3. 仅在「单页」内做局部对齐（每页最多几千字，极快）
            matcher = difflib.SequenceMatcher(None, mac_text, paddle_text)
            opcodes = matcher.get_opcodes()

            # 4. 构建「Mac 字符位置 → Paddle 字符位置」的映射字典
            pos_map: Dict[int, int] = {}
            for tag, i1, i2, j1, j2 in opcodes:
                if tag in ('equal', 'replace'):
                    length = min(i2 - i1, j2 - j1)
                    for offset in range(length):
                        pos_map[i1 + offset] = j1 + offset
                # 'insert' 不映射（Mac 没有对应位置，不主动插入新块）
                # 'delete' 不映射（Mac 独有内容，保留原文）

            # 5. 遍历 Mac 的每个 Block，用映射表回填 Paddle 文本
            cursor = 0
            for b in mac_blocks:
                total_text_blocks += 1

                # 找到这个块在 mac_text 中的绝对起止位置（用累积偏移，避免 find() 重复句子问题）
                start = mac_text.find(b.text, cursor)
                if start == -1:
                    cursor += len(b.text) + 1
                    continue
                end = start + len(b.text)
                cursor = end

                mapped_positions = []
                for idx in range(start, end):
                    if idx in pos_map:
                        mapped_positions.append(pos_map[idx])

                if not mapped_positions:
                    continue

                pad_start = min(mapped_positions)
                pad_end = max(mapped_positions) + 1
                new_text = paddle_text[pad_start:pad_end]

                if new_text.strip() and new_text != b.text:
                    if not b.ocr_raw:
                        b.ocr_raw = b.text
                    b.text = new_text
                    b.modified_by = "ocr_alignment"
                    replaced_count += 1

        report = {
            "total_text_blocks": total_text_blocks,
            "replaced": replaced_count,
            "skipped": total_text_blocks - replaced_count,
            "pages_processed": len([p for p in mac_pages if p in paddle_pages]),
        }

        new_doc.add_log(
            "ocr_alignment",
            f"用 PaddleOCR 替换 {replaced_count} 个块（共 {total_text_blocks} 个文字块）",
            replaced_count
        )

        return new_doc, report

    # ----------------------------------------------------------------------
    # 辅助方法
    # ----------------------------------------------------------------------

    def _group_blocks_by_page(self, doc: UnifiedDocument) -> dict[int, list[Block]]:
        pages = defaultdict(list)
        for b in doc.blocks:
            if b.type in self.TEXT_TYPES and b.text.strip():
                pages[b.page].append(b)
        return pages

    def _sort_blocks(self, blocks: List[Block]) -> List[Block]:
        def key_func(b: Block) -> tuple:
            if hasattr(b, 'reading_order') and b.reading_order is not None:
                return (b.reading_order, 0)
            if b.bbox:
                return (b.bbox.y, b.bbox.x)
            return (0, 0)
        return sorted(blocks, key=key_func)

    def _build_text_stream(self, blocks: List[Block]) -> Tuple[str, List[int]]:
        texts = [b.text for b in blocks]
        text = ''.join(texts)
        owner = []
        for idx, b in enumerate(blocks):
            owner.extend([idx] * len(b.text))
        return text, owner