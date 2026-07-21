#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paragraph —— Text Replacement Engine 用的轻量文本单元。

比 UnifiedDocument.Block 更简单：只关心"这段文字属于哪一章、在原文里第几个、
是不是标题"，不携带 bbox/图片/页面这些结构信息——那些结构永远来自 OCR 那一份
UnifiedDocument，Paragraph 只代表"用来替换正文的高质量文本"，两者角色不同，
不该共用同一个数据结构。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Paragraph:
    text: str
    index: int = 0          # 在来源文档里的顺序位置（0-based）
    chapter: str = ""       # 所属章节标题（从最近一个标题继承）
    source: str = ""        # 来源文件名，方便对齐报告里标注
    is_title: bool = False  # 这段本身是不是章节/小节标题
