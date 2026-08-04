# -*- coding: utf-8 -*-
"""Crash-safe atomic file replacement helpers.

All temporary files are created beside the destination so ``os.replace`` stays
on the same filesystem.  Data and the containing directory are flushed before
returning on POSIX/macOS, reducing the chance of a power loss leaving a zero-byte
configuration, project snapshot, or EPUB.
"""
from __future__ import annotations

import json
import os
import tempfile
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO, BinaryIO, Any


def _sync_directory(directory: Path) -> None:
    if os.name != "posix":
        return
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


@contextmanager
def atomic_open(
    path: str | os.PathLike[str],
    mode: str = "w",
    *,
    encoding: str | None = "utf-8",
    newline: str | None = None,
) -> Iterator[TextIO | BinaryIO]:
    """Open a sibling temporary file and atomically replace ``path`` on success."""
    if "a" in mode or "+" in mode or "x" in mode or "r" in mode:
        raise ValueError("atomic_open only supports fresh write modes")
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    previous_mode: int | None = None
    try:
        previous_mode = stat.S_IMODE(target.stat().st_mode)
    except OSError:
        previous_mode = None
    suffix = target.suffix + ".tmp"
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=suffix, dir=target.parent)
    temp = Path(temp_name)
    file_obj = None
    committed = False
    try:
        if "b" in mode:
            file_obj = os.fdopen(fd, mode)
        else:
            file_obj = os.fdopen(fd, mode, encoding=encoding or "utf-8", newline=newline)
        with file_obj as handle:
            yield handle
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        if previous_mode is not None:
            try:
                os.chmod(temp, previous_mode)
            except OSError:
                pass
        os.replace(temp, target)
        _sync_directory(target.parent)
        committed = True
    finally:
        if file_obj is None:
            try:
                os.close(fd)
            except OSError:
                pass
        if not committed:
            temp.unlink(missing_ok=True)


def atomic_write_text(
    path: str | os.PathLike[str], text: str, *, encoding: str = "utf-8", newline: str | None = None
) -> Path:
    target = Path(path).expanduser()
    with atomic_open(target, "w", encoding=encoding, newline=newline) as handle:
        handle.write(str(text))
    return target


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> Path:
    target = Path(path).expanduser()
    with atomic_open(target, "wb", encoding=None) as handle:
        handle.write(data)
    return target


def atomic_write_json(
    path: str | os.PathLike[str], payload: Any, *, ensure_ascii: bool = False, indent: int | None = 2
) -> Path:
    return atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=ensure_ascii, indent=indent) + ("\n" if indent is not None else ""),
    )
