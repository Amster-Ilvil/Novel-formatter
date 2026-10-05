# -*- coding: utf-8 -*-
"""页面管理的导入来源整理（纯 Python，无 Qt 依赖，可直接单元测试）。

「导入文件」「导入文件夹」和拖放共用同一套规则：
文件夹展开为其中的图片与 PDF（不递归），按自然序排列，忽略隐藏文件与 macOS 的 ._ 资源叉。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".heic", ".tif", ".tiff", ".bmp", ".gif")
SOURCE_EXTS = IMAGE_EXTS + (".pdf",)


def natural_key(text: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(text))]


def _is_hidden(path: Path) -> bool:
    return path.name.startswith(".")           # 含 .DS_Store 与 ._xxx.png


def folder_sources(folder: str | Path) -> list[str]:
    """文件夹内可导入的图片/PDF，自然序；没有任何可导入文件时返回空列表。"""
    try:
        items = [p for p in Path(folder).iterdir()
                 if p.is_file() and not _is_hidden(p) and p.suffix.lower() in SOURCE_EXTS]
    except OSError:
        return []
    return [str(p) for p in sorted(items, key=lambda p: natural_key(p.name))]


def collect_sources(paths: Iterable[str | Path]) -> list[str]:
    """把拖入/选择的路径统一整理成文件列表：文件原样保留（仅限图片/PDF），文件夹展开。"""
    out: list[str] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            out.extend(folder_sources(p))
        elif p.is_file() and not _is_hidden(p) and p.suffix.lower() in SOURCE_EXTS:
            out.append(str(p))
    return out


def import_label(paths: list[str], fallback: str = "") -> str:
    """导入后的书名：单个来源取文件名，多个来源写「首个 等 N 个文件」。"""
    if not paths:
        return fallback
    first = Path(paths[0]).stem
    return first if len(paths) == 1 else f"{first} 等 {len(paths)} 个文件"
