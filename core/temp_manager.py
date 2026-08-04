# -*- coding: utf-8 -*-
"""Safe lifecycle management for OCR temporary crop directories."""
from __future__ import annotations

import logging
import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

_INVALID_NAME = re.compile(r'[\\/:\x00-\x1f]+')


class TempCropManager:
    """Create and clean temporary directories without path traversal.

    ``book_name`` is always converted to one safe leaf directory. Cleanup is
    restricted to this manager's own root, preventing an accidental absolute
    path or ``..`` value from deleting unrelated user files.
    """

    def __init__(self, base_dir=None, name="ocr_crop"):
        base = Path(base_dir).expanduser() if base_dir else Path(tempfile.gettempdir())
        safe_name = self._safe_leaf(name, fallback="ocr_crop")
        self.base_dir = base
        self.root = base / "novel_formatter" / safe_name

    @staticmethod
    def _safe_leaf(value, *, fallback: str) -> str:
        raw = str(value or "").strip()
        # Treat both POSIX and Windows separators as separators even on macOS.
        raw = raw.replace("\\", "/").rsplit("/", 1)[-1]
        safe = _INVALID_NAME.sub("_", raw).strip(" .")
        if not safe or safe in {".", ".."}:
            return fallback
        # Keep paths readable but avoid filesystem/pathological name lengths.
        return safe[:120]

    def _resolved_root(self) -> Path:
        return self.root.resolve(strict=False)

    def _validate_managed_path(self, path: Path) -> Path:
        resolved = path.expanduser().resolve(strict=False)
        root = self._resolved_root()
        if resolved != root and root not in resolved.parents:
            raise ValueError(f"拒绝清理管理目录之外的路径: {resolved}")
        return resolved

    def create(self, book_name="default") -> Path:
        leaf = self._safe_leaf(book_name, fallback="default")
        path = self.root / leaf
        self._validate_managed_path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def cleanup(self, path=None, *, strict: bool = False) -> bool:
        target = self._validate_managed_path(Path(path) if path is not None else self.root)
        if not target.exists():
            return False
        try:
            shutil.rmtree(target)
            return True
        except OSError:
            if strict:
                raise
            logger.warning("Failed to remove temporary directory %s", target, exc_info=True)
            return False

    @contextmanager
    def managed(self, book_name="default") -> Iterator[Path]:
        path = self.create(book_name)
        try:
            yield path
        finally:
            self.cleanup(path)
