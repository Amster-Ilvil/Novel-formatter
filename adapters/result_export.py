#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Formatter result export helpers.

The GUI previously exposed a JSON-only "save result" button plus a separate
DOCX button.  Keeping format selection and extension handling here gives the
Formatter one predictable save path and makes the non-GUI parts testable.
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

from models.document import Block, BlockType, UnifiedDocument


FORMAT_EXTENSIONS: dict[str, str] = {
    "json": ".json",
    "docx": ".docx",
    "markdown": ".md",
    "text": ".txt",
}

SAVE_FILTERS: tuple[tuple[str, str], ...] = (
    ("docx", "Word 文档 (*.docx)"),
    ("json", "JSON 数据 (*.json)"),
    ("markdown", "Markdown 文本 (*.md)"),
    ("text", "纯文本 (*.txt)"),
)

_TEXT_BLOCK_TYPES = {
    BlockType.PARAGRAPH,
    BlockType.DIALOGUE,
    BlockType.CHAPTER,
    BlockType.SECTION,
    BlockType.RUBY,
    BlockType.FOOTNOTE,
}


def save_filter_string() -> str:
    return ";;".join(label for _fmt, label in SAVE_FILTERS)


def format_from_filter(selected_filter: str, path: str = "") -> str:
    """Resolve an export format from QFileDialog's selected filter or suffix."""
    selected = str(selected_filter or "")
    for fmt, label in SAVE_FILTERS:
        if selected == label or label.split(" (")[0] in selected:
            return fmt

    suffix = Path(path).suffix.lower()
    for fmt, ext in FORMAT_EXTENSIONS.items():
        if suffix == ext:
            return fmt
    return "docx"


def ensure_export_extension(path: str, fmt: str) -> str:
    """Append the selected format extension when the user omitted one.

    A user-entered known extension wins, even if the selected filter differs;
    this mirrors normal desktop save-dialog behaviour and avoids ``name.txt.md``.
    """
    p = Path(path)
    known = {ext.lower() for ext in FORMAT_EXTENSIONS.values()}
    if p.suffix.lower() in known:
        return str(p)
    ext = FORMAT_EXTENSIONS.get(fmt, ".docx")
    return str(p.with_suffix(ext))


def safe_result_filename(title: str, default: str = "formatter_result") -> str:
    """Create a cross-platform filename stem from document metadata."""
    stem = str(title or "").strip() or default
    stem = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", stem)
    stem = re.sub(r"\s+", " ", stem).strip(" .")
    return stem[:120] or default


def iter_text_blocks(doc: UnifiedDocument) -> Iterable[Block]:
    for block in doc.blocks:
        if (block.metadata or {}).get("consumed"):
            continue
        if block.type not in _TEXT_BLOCK_TYPES:
            continue
        if not str(block.text or "").strip():
            continue
        yield block


def document_to_plain_text(doc: UnifiedDocument) -> str:
    """Return readable text with one blank line between logical blocks."""
    return "\n\n".join(str(block.text).strip() for block in iter_text_blocks(doc)) + "\n"


def document_to_markdown(doc: UnifiedDocument) -> str:
    """Export a lightweight, editable Markdown representation."""
    lines: list[str] = []
    for block in iter_text_blocks(doc):
        text = str(block.text).strip()
        if block.type == BlockType.CHAPTER:
            lines.append(f"# {text}")
        elif block.type == BlockType.SECTION:
            lines.append(f"## {text}")
        elif block.type == BlockType.FOOTNOTE:
            lines.append(f"> {text}")
        else:
            lines.append(text)
    return "\n\n".join(lines) + "\n"


def export_text_result(doc: UnifiedDocument, output_path: str, fmt: str) -> str:
    """Write JSON/Markdown/TXT and return the normalized output path.

    DOCX is intentionally handled by ``builder.word_builder`` because it may be
    comparatively slow and the GUI runs that operation in a worker thread.
    """
    fmt = str(fmt or "").lower()
    if fmt not in {"json", "markdown", "text"}:
        raise ValueError(f"不支持的文本导出格式: {fmt}")

    normalized = ensure_export_extension(output_path, fmt)
    target = Path(normalized)
    target.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        payload = doc.to_json()
    elif fmt == "markdown":
        payload = document_to_markdown(doc)
    else:
        payload = document_to_plain_text(doc)

    # Atomic replacement prevents a cancelled/crashed save from leaving a
    # half-written user file.
    temp_path = target.with_name(f".{target.name}.tmp")
    try:
        temp_path.write_text(payload, encoding="utf-8")
        temp_path.replace(target)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
    return str(target)
