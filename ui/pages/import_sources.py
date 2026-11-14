# -*- coding: utf-8 -*-
"""统一页面来源收集：只选择已被底层 PDF/归档导入器支持的文件。

从 fork PR #12 选择性适配：文件夹、拖放入口支持 WebP/HEIF、CBZ/CBR
及由文件头识别的无扩展名图像；保留原有自然排序与非递归语义。
只读取未知扩展名的文件头，不在此处下载、执行或解压任何文件。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from adapters.file_type import detect_file_type, is_supported_source

IMAGE_EXTS = (
    ".png", ".jpg", ".jpeg", ".heic", ".heif", ".tif", ".tiff",
    ".bmp", ".gif", ".webp", ".ppm",
)
SOURCE_EXTS = IMAGE_EXTS + (".pdf", ".cbz", ".cbr", ".zip", ".rar")
# These are ZIP containers or document types, not comic-image archive sources.
# A recognised document suffix wins over its incidental embedded images.
_NON_SOURCE_EXTS = frozenset({
    ".txt", ".md", ".markdown", ".epub", ".docx", ".xlsx",
    ".pptx", ".odt", ".ods", ".odp", ".jar", ".apk", ".ipa",
})


def natural_key(text: str):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(text))]


def _is_hidden(path: Path) -> bool:
    return path.name.startswith(".")  # macOS .DS_Store / AppleDouble 也跳过


def _is_source_file(path: Path) -> bool:
    if not path.is_file() or _is_hidden(path):
        return False
    suffix = path.suffix.lower()
    if suffix in SOURCE_EXTS:
        return True  # 原有文件后端仍会做格式/归档安全校验
    if suffix in _NON_SOURCE_EXTS:
        return False
    # 如图像被命名为 page / page.dat，仅在未知扩展名时探测内容。
    try:
        detected = detect_file_type(path)
    except (OSError, ValueError, RuntimeError):
        # 文件在扫描时被删除、无读取权限或无效：过滤而非中断整本导入。
        return False
    return is_supported_source(detected) and detected.detail != "by-extension"


def folder_sources(folder: str | Path) -> list[str]:
    """非递归展开受支持来源；按自然序排列并跳过隐藏文件。"""
    try:
        items = [p for p in Path(folder).iterdir() if _is_source_file(p)]
    except OSError:
        return []
    return [str(p) for p in sorted(items, key=lambda p: natural_key(p.name))]


def collect_sources(paths: Iterable[str | Path]) -> list[str]:
    """统一拖放/导入文件/文件夹规则，保留用户传入来源的相对顺序。"""
    out: list[str] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            out.extend(folder_sources(p))
        elif _is_source_file(p):
            out.append(str(p))
    return out


def import_label(paths: list[str], fallback: str = "") -> str:
    if not paths:
        return fallback
    first = Path(paths[0]).stem
    return first if len(paths) == 1 else f"{first} 等 {len(paths)} 个文件"
