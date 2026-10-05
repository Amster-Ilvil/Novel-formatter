# -*- coding: utf-8 -*-
"""交互细节的纯逻辑（无 Qt 依赖，可直接单元测试）：提示条时长/位置、拖放决策、对话框尺寸与默认按钮。"""
from __future__ import annotations

import re
from typing import Sequence

# ── 提示条 ────────────────────────────────────────────────────────────────
def toast_duration_ms(text: str, kind: str = "success") -> int:
    """按文字长度给阅读时间：短句 2.2 秒，长句最多 7 秒；警告多留 1.2 秒。"""
    ms = 1800 + 55 * len(str(text or ""))
    if kind == "warning":
        ms += 1200
    return min(7000, max(2200, ms))


def toast_origin(parent_w: int, parent_h: int, w: int, h: int, bottom: int = 44) -> tuple[int, int]:
    """水平居中、离窗口底部 bottom 像素；窗口太小也不会出现负坐标。"""
    return max(0, (int(parent_w) - int(w)) // 2), max(0, int(parent_h) - int(h) - int(bottom))


# ── 拖放 ──────────────────────────────────────────────────────────────────
def drop_decision(sources: Sequence[str], busy: bool) -> str:
    """'empty' 没有可导入的文件；'busy' OCR 运行中不允许换页面源；'import' 可以导入。"""
    if not sources:
        return "empty"
    return "busy" if busy else "import"


def drop_overlay_text(count: int, title: str = "松开鼠标，导入文件") -> str:
    """标题可先翻译再传入；数字单独追加，避免把计数混进翻译表。"""
    return f"{title} · {int(count)}"


# ── 对话框 ────────────────────────────────────────────────────────────────
def clamp_dialog_min(min_w: int, min_h: int, avail_w: int, avail_h: int, ratio: float = 0.92) -> tuple[int, int]:
    """把对话框的最小尺寸限制在可用屏幕的 ratio 以内，小屏幕上不再出现比屏幕还大的窗口。"""
    limit_w, limit_h = int(avail_w * ratio), int(avail_h * ratio)
    return (min(int(min_w), limit_w) if limit_w > 0 else int(min_w),
            min(int(min_h), limit_h) if limit_h > 0 else int(min_h))


_PRIMARY_PREFIXES = (
    "保存", "开始", "应用", "确定", "确认", "好的",
    "OK", "Save", "Apply", "Start", "Confirm",
    "保存", "開始", "適用", "確定", "確認", "実行",
)
_LEADING_JUNK = re.compile(r"^[^\w\u3040-\u30ff\u3400-\u9fff]+")


def pick_default_button(texts: Sequence[str], has_text_input: bool) -> int | None:
    """窗口里有输入框时不设默认按钮（避免在输入框里按回车就误提交）；否则取第一个主操作按钮。"""
    if has_text_input:
        return None
    for index, raw in enumerate(texts):
        text = _LEADING_JUNK.sub("", str(raw or "")).strip()
        if text.startswith(_PRIMARY_PREFIXES):
            return index
    return None
