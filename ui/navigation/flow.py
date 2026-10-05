# -*- coding: utf-8 -*-
"""工作流顺序（纯 Python，无 Qt 依赖，可直接单元测试）。

侧边栏的顺序是 页面管理/OCR/格式处理/文字校对/EPUB，但实际工作顺序是
页面 → OCR → 文字校对 → 格式处理 → EPUB（与工作区的“项目流程”一致）。
"""
from __future__ import annotations

# (侧边栏索引, 简称)
FLOW = ((1, "页面"), (2, "OCR"), (4, "校对"), (3, "格式"), (5, "EPUB"))
_ORDER = [idx for idx, _ in FLOW]


def flow_position(section: int):
    """返回 section 在流程中的位置（0 起）；不在流程中返回 None。"""
    try:
        return _ORDER.index(int(section))
    except ValueError:
        return None


def next_section(section: int):
    pos = flow_position(section)
    if pos is None or pos + 1 >= len(_ORDER):
        return None
    return _ORDER[pos + 1]


def step_state(step_section: int, current_section: int) -> str:
    """'done' = 流程中位于当前之前；'current'；'todo'。"""
    cur, step = flow_position(current_section), flow_position(step_section)
    if cur is None or step is None:
        return "todo"
    return "current" if step == cur else ("done" if step < cur else "todo")
