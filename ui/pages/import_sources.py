# -*- coding: utf-8 -*-
"""页面管理的导入来源整理（纯 Python，无 Qt 依赖，可直接单元测试）。

「导入文件」「导入文件夹」和拖放共用同一套规则：
文件夹展开为其中的图片、PDF 与图片压缩包（不递归），按自然序排列，忽略隐藏文件与 macOS 的 ._ 资源叉。

Files with a known source suffix are accepted by name.  Files without a suffix
or with an unknown one are classified by content via
:func:`adapters.file_type.detect_file_type`, so a scan saved as ``page`` or
``page.dat`` is still imported.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from adapters.file_type import (
    EPUB_EXTENSIONS,
    TEXT_EXTENSIONS,
    detect_file_type,
    is_supported_source,
)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".heic", ".tif", ".tiff", ".bmp", ".gif", ".webp")
SOURCE_EXTS = IMAGE_EXTS + (".pdf", ".cbz", ".cbr", ".zip", ".rar")
# Known non-source suffixes are rejected without reading the file.  Office/ODF
# documents and app bundles are ZIP containers and would otherwise be sniffed
# as image archives.
_NON_SOURCE_EXTS = frozenset(TEXT_EXTENSIONS | EPUB_EXTENSIONS | {
    ".docx", ".xlsx", ".pptx", ".odt", ".ods", ".odp", ".jar", ".apk", ".ipa",
})


def natural_key(text: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(text))]


def _is_hidden(path: Path) -> bool:
    return path.name.startswith(".")           # 含 .DS_Store 与 ._xxx.png


def _is_source_file(path: Path) -> bool:
    """Known suffix decides by name; a missing or unknown suffix is sniffed by content."""
    if not path.is_file() or _is_hidden(path):
        return False
    suffix = path.suffix.lower()
    if suffix in SOURCE_EXTS:
        return True
    if suffix in _NON_SOURCE_EXTS:
        return False
    return is_supported_source(detect_file_type(path))


def folder_sources(folder: str | Path) -> list[str]:
    """文件夹内可导入的图片/PDF/压缩包，自然序；没有任何可导入文件时返回空列表。"""
    try:
        items = [p for p in Path(folder).iterdir() if _is_source_file(p)]
    except OSError:
        return []
    return [str(p) for p in sorted(items, key=lambda p: natural_key(p.name))]


def collect_sources(paths: Iterable[str | Path]) -> list[str]:
    """把拖入/选择的路径统一整理成文件列表：文件原样保留（仅限可导入来源），文件夹展开。"""
    out: list[str] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            out.extend(folder_sources(p))
        elif _is_source_file(p):
            out.append(str(p))
    return out


def import_label(paths: list[str], fallback: str = "") -> str:
    """导入后的书名：单个来源取文件名，多个来源写「首个 等 N 个文件」。"""
    if not paths:
        return fallback
    first = Path(paths[0]).stem
    return first if len(paths) == 1 else f"{first} 等 {len(paths)} 个文件"
