# -*- coding: utf-8 -*-
"""Lifetime manager for OCR preview/crop temporary directories.

OCR preview pages must remain available after recognition finishes so the user
can browse every processed image.  Directories are therefore retained for the
whole OCR workspace and are deleted only when that workspace is explicitly
cleared or the application closes.
"""
from __future__ import annotations

import shutil
import threading
from pathlib import Path


class OCRPreviewTempStore:
    """Thread-safe owner for run-local OCR temporary directories."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._roots: set[str] = set()
        self._epoch = 0
        self._closing = False

    def begin_run(self) -> int:
        """Return the current workspace epoch for a newly starting OCR run."""
        with self._lock:
            return self._epoch

    def register(self, path: str, epoch: int) -> bool:
        """Retain *path* when it still belongs to the active workspace.

        A clear/close can race with a worker that has just created its directory.
        Stale directories are removed immediately instead of being leaked.
        """
        normalized = str(Path(path).expanduser().resolve())
        with self._lock:
            keep = not self._closing and int(epoch) == self._epoch
            if keep:
                self._roots.add(normalized)
        if not keep:
            shutil.rmtree(normalized, ignore_errors=True)
        return keep

    def cleanup(self, *, closing: bool = False) -> tuple[str, ...]:
        """Delete all retained roots and invalidate workers from older epochs."""
        with self._lock:
            self._epoch += 1
            if closing:
                self._closing = True
            roots = tuple(sorted(self._roots))
            self._roots.clear()
        for root in roots:
            shutil.rmtree(root, ignore_errors=True)
        return roots

    def finish_run(self, path: str, epoch: int) -> bool:
        """Finalize a worker without deleting a valid retained run.

        If an explicit clear raced with the worker, downstream code may have
        recreated part of the directory after the first cleanup pass.  A stale
        worker performs this second deletion when it finally exits.
        """
        normalized = str(Path(path).expanduser().resolve())
        with self._lock:
            keep = (
                not self._closing
                and int(epoch) == self._epoch
                and normalized in self._roots
            )
        if not keep:
            shutil.rmtree(normalized, ignore_errors=True)
        return keep

    @property
    def roots(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._roots))

    @property
    def closing(self) -> bool:
        with self._lock:
            return self._closing
