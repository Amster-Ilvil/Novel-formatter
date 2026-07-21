#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GapTree 阅读顺序恢复
竖排日文 OCR 输出的 blocks 往往没有正确的阅读顺序（右→左列，上→下行）。
本模块基于 bbox 坐标进行列聚类，然后按 右→左、上→下 排序。
"""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.document import UnifiedDocument, Block, BlockType


def _x_center(block: Block) -> float:
    if block.bbox is None:
        return 0.5
    return block.bbox.x + block.bbox.w / 2


def _y_center(block: Block) -> float:
    if block.bbox is None:
        return 0.5
    return block.bbox.y + block.bbox.h / 2


def _avg_width(blocks: list[Block]) -> float:
    widths = [b.bbox.w for b in blocks if b.bbox and b.bbox.w > 0]
    return sum(widths) / len(widths) if widths else 0.05


def sort_blocks_by_reading_order(blocks: list[Block]) -> list[Block]:
    """
    对带有 bbox 的 blocks 按竖排日文阅读顺序排序：
    1. 按 X 中心聚类为"列"（gap 阈值 = avg_width * 0.35）
    2. 列按 X 降序排列（右→左）
    3. 列内 blocks 按 Y 升序排列（上→下）
    """
    has_bbox = [b for b in blocks if b.bbox is not None]
    no_bbox = [b for b in blocks if b.bbox is None]

    if not has_bbox:
        return blocks

    gap_threshold = _avg_width(has_bbox) * 0.35
    if gap_threshold < 0.01:
        gap_threshold = 0.03

    sorted_by_x = sorted(has_bbox, key=lambda b: _x_center(b), reverse=True)

    columns: list[list[Block]] = []
    current_col: list[Block] = [sorted_by_x[0]]

    for b in sorted_by_x[1:]:
        if abs(_x_center(b) - _x_center(current_col[-1])) <= gap_threshold:
            current_col.append(b)
        else:
            columns.append(current_col)
            current_col = [b]
    columns.append(current_col)

    for col in columns:
        col.sort(key=lambda b: _y_center(b))

    columns.sort(key=lambda col: _x_center(col[0]), reverse=True)

    result: list[Block] = []
    for col in columns:
        result.extend(col)
    result.extend(no_bbox)

    return result


def restore_reading_order(doc: UnifiedDocument) -> UnifiedDocument:
    """
    对 UnifiedDocument 中的 blocks 按页分组，每页内对 PARAGRAPH/DIALOGUE
    类型的 blocks 进行阅读顺序排序。
    IMAGE_REF、CHAPTER 等结构性 blocks 保持原位（原本在哪两个可排序块之间，
    排序后仍然在同一相对位置之间 —— 不会被搬到页面的开头或结尾）。

    注：若一批块完全没有 bbox（例如快捷指令版 Apple Vision 适配器的输出），
    本步骤退化为空操作，blocks 保持原始顺序不变。

    注：来自 PDF 文字层直读（source_engine == "pdf_text_layer"）的文档本步骤
    整体跳过——那些 block 的顺序是直接从 PDF 里每个字符的精确坐标按列算出来的
    （见 adapters/pdf_text_layer.py），比这里用 bbox 做的 GapTree 列聚类准得多；
    再用 GapTree 重排一遍反而会把本来正确的顺序打乱、段落错位。
    """
    if doc.metadata.source_engine == "pdf_text_layer":
        return copy.deepcopy(doc)

    doc = copy.deepcopy(doc)

    SORTABLE = {BlockType.PARAGRAPH, BlockType.DIALOGUE}

    pages: dict[int, list[Block]] = defaultdict(list)
    for b in doc.blocks:
        pages[b.page].append(b)

    new_blocks: list[Block] = []
    global_order = 0

    for page_no in sorted(pages.keys()):
        page_blocks = pages[page_no]
        sortable_blocks = [b for b in page_blocks if b.type in SORTABLE and b.bbox]

        if sortable_blocks:
            sorted_blocks = sort_blocks_by_reading_order(sortable_blocks)
        else:
            sorted_blocks = []

        # 按原始顺序遍历该页所有块：可排序的槽位依次替换为排好序的块，
        # 结构性块（IMAGE_REF/CHAPTER 等，或没有 bbox 的块）保持原位不动。
        s_idx = 0
        for orig_block in page_blocks:
            if orig_block.type in SORTABLE and orig_block.bbox:
                b = sorted_blocks[s_idx]
                s_idx += 1
            else:
                b = orig_block
            b.reading_order = global_order
            global_order += 1
            new_blocks.append(b)

    doc.blocks = new_blocks
    doc.add_log("reading_order", f"按阅读顺序重排 {len(doc.blocks)} 个块")
    return doc
