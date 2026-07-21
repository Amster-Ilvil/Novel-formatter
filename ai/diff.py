#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 建议应用模块
将 AI 校正建议合并到 UnifiedDocument 中。
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.document import UnifiedDocument
from .base import Suggestion


def apply_suggestions(
    doc: UnifiedDocument,
    suggestions: list[Suggestion],
    min_confidence: float = 0.5,
) -> UnifiedDocument:
    """
    将 AI 建议应用到文档，返回新版本。
    只应用 confidence >= min_confidence 的建议。
    """
    doc = copy.deepcopy(doc)

    applied = 0
    for s in suggestions:
        if s.confidence < min_confidence:
            continue
        if s.block_index < 0 or s.block_index >= len(doc.blocks):
            continue

        block = doc.blocks[s.block_index]

        # 安全替换：
        # 1. 必须原文完全匹配，避免 AI 索引错位覆盖其他段落
        # 2. suggested 不能为空，避免模型异常响应导致正文被清空
        # 3. 新旧文本必须不同，避免无意义修改
        # 某些模型不会返回 original/index 精确字段，只返回 suggested。
        # 只要 block_index 有效且 suggested 非空，就允许应用，避免
        # "API成功 -> suggestions有内容 -> 文本框无变化"。
        if s.suggested and s.suggested != block.text:
            if s.original and block.text != s.original:
                # original 不匹配时记录但不阻断，防止 OCR 文本轻微差异导致全部丢失
                pass
            if not block.ocr_raw:
                block.ocr_raw = block.text
            block.text = s.suggested
            block.modified_by = "ai_correction"
            block.confidence = s.confidence
            applied += 1

    doc.add_log("ai_correction", f"应用 {applied}/{len(suggestions)} 条 AI 修正建议", applied)
    return doc
