# -*- coding: utf-8 -*-
"""校对流程的纯逻辑（无 Qt）：进度条分桶、待判断步进、数字键选候选。

参考：VS Code 合并编辑器的“上一个/下一个冲突”、eScriptorium/Transkribus 的
“Enter/↓ 保存并到下一行”，以及按状态着色的总览条。
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

# 数值越大越“需要用户注意”：分桶时取最需要处理的状态，剩余工作永远不会被盖住
_PRIORITY = {"reviewed": 0, "changed": 1, "pending": 2, "judge": 3}


def bucket_states(states: Sequence[str], slots: int) -> list[str]:
    """把任意长度的状态序列压缩成 slots 个格子（每格取最需要处理的状态）。"""
    states = list(states)
    slots = max(1, int(slots))
    if not states:
        return []
    if len(states) <= slots:
        return states
    out = []
    for i in range(slots):
        lo, hi = i * len(states) // slots, max(i * len(states) // slots + 1, (i + 1) * len(states) // slots)
        chunk = states[lo:hi]
        out.append(max(chunk, key=lambda s: _PRIORITY.get(s, 0)))
    return out


def bucket_of(index: int, total: int, slots: int) -> int:
    """原始序号 → 所在格子序号（用于高亮当前位置、点击跳转的反向映射）。"""
    if total <= 0:
        return 0
    slots = max(1, min(int(slots), total))
    return min(slots - 1, max(0, int(index)) * slots // total)


def index_of_bucket(bucket: int, total: int, slots: int) -> int:
    """格子序号 → 该格子里第一个“最需要处理”的原始序号由调用方决定；这里给出起点。"""
    slots = max(1, min(int(slots), max(1, total)))
    return min(max(0, total - 1), max(0, int(bucket)) * total // slots)


def step_pending(rows: Iterable[int], current: int, forward: bool = True) -> Optional[int]:
    """在“待判断行号集合”里找 current 之后/之前的下一个；没有则返回 None（不循环）。"""
    ordered = sorted(set(int(r) for r in rows))
    if forward:
        return next((r for r in ordered if r > current), None)
    return next((r for r in reversed(ordered) if r < current), None)


def candidate_for_key(number: int, candidate_count: int) -> Optional[int]:
    """Alt+1..9 → 候选下标（0 起）；越界返回 None。"""
    n = int(number)
    return n - 1 if 1 <= n <= min(9, int(candidate_count)) else None


def progress_text(done: int, total: int, pending: int) -> str:
    if total <= 0:
        return "暂无待核对内容"
    pct = 100.0 * done / total
    tail = f" · 剩余待判断 {pending}" if pending else " · 全部完成"
    return f"已核对 {done} / {total}（{pct:.0f}%）{tail}"
