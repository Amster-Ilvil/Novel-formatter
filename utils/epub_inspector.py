# -*- coding: utf-8 -*-
"""Safe EPUB archive inspection used by the GUI preview tree.

The preview is rebuilt many times in one application session.  Using
``ZipFile.extractall`` together with stale Qt tree items made consecutive-book
exports fragile on some PySide6/macOS combinations.  This helper is deliberately
Qt-free, validates every archive path, and extracts files one by one into a fresh
directory so it can be regression-tested without a GUI runtime.
"""
from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

from utils.safe_archive import safe_member_parts, validate_zip


_TEXT_SUFFIXES = {".xhtml", ".html", ".htm", ".opf", ".css", ".xml", ".ncx"}


@dataclass(frozen=True)
class EPUBInspection:
    names: tuple[str, ...]
    contents: dict[str, str]
    extract_dir: str


def _safe_member_parts(name: str) -> tuple[str, ...]:
    """Backward-compatible wrapper around the shared archive validator."""
    return safe_member_parts(name)


def inspect_epub_archive(epub_path: str | Path, extract_dir: str | Path) -> EPUBInspection:
    """Read and safely extract an EPUB into ``extract_dir``.

    ``extract_dir`` is always treated as disposable preview state.  Existing
    contents are removed first, which prevents a second book from inheriting
    files from the first book even when filenames overlap or disappear.
    """
    archive = Path(epub_path)
    target_root = Path(extract_dir)
    if target_root.exists():
        shutil.rmtree(target_root, ignore_errors=True)
    target_root.mkdir(parents=True, exist_ok=True)

    names: list[str] = []
    contents: dict[str, str] = {}

    with zipfile.ZipFile(archive) as zf:
        validate_zip(zf)
        for info in sorted(zf.infolist(), key=lambda item: item.filename):
            name = info.filename
            parts = _safe_member_parts(name)
            names.append(name)
            if not parts:
                continue

            target = target_root.joinpath(*parts)
            if info.is_dir() or name.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)

            if target.suffix.lower() in _TEXT_SUFFIXES:
                contents[name] = target.read_text(encoding="utf-8", errors="replace")

    return EPUBInspection(tuple(names), contents, str(target_root))
