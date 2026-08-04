# -*- coding: utf-8 -*-
"""Bounded, traversal-safe ZIP extraction.

The implementation follows the same core safety properties used by mature
package installers: validate containment before writing, stream each member
instead of loading it all into memory, reject links/special files, and impose
member/size/ratio limits to avoid archive bombs.
"""
from __future__ import annotations

import os
import shutil
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


class UnsafeArchiveError(ValueError):
    pass


@dataclass(frozen=True)
class ZipExtractionLimits:
    max_members: int = 50_000
    max_total_uncompressed: int = 8 * 1024 * 1024 * 1024
    max_single_file: int = 4 * 1024 * 1024 * 1024
    max_compression_ratio: float = 2_000.0


def safe_member_parts(name: str) -> tuple[str, ...]:
    raw = str(name or "").replace("\\", "/")
    if "\x00" in raw:
        raise UnsafeArchiveError("ZIP 成员名包含 NUL 字符")
    path = PurePosixPath(raw)
    parts = path.parts
    if path.is_absolute() or any(part in {"", ".", ".."} for part in parts):
        raise UnsafeArchiveError(f"ZIP 包含不安全路径：{name!r}")
    if parts and len(parts[0]) >= 2 and parts[0][1] == ":":
        raise UnsafeArchiveError(f"ZIP 包含 Windows 绝对路径：{name!r}")
    return tuple(parts)


def _member_mode(info: zipfile.ZipInfo) -> int:
    return (int(info.external_attr) >> 16) & 0xFFFF


def _validate_member_type(info: zipfile.ZipInfo) -> None:
    mode = _member_mode(info)
    if not mode:
        return
    if stat.S_ISLNK(mode):
        raise UnsafeArchiveError(f"ZIP 不允许符号链接：{info.filename!r}")
    # Some writers store only permission bits (for example 0o600) without the
    # POSIX file-type bits.  Treat those as ordinary files; reject only an
    # explicitly encoded special type.
    file_type = stat.S_IFMT(mode)
    if file_type and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
        raise UnsafeArchiveError(f"ZIP 不允许特殊文件：{info.filename!r}")


def validate_zip(archive: zipfile.ZipFile, *, limits: ZipExtractionLimits | None = None) -> None:
    limits = limits or ZipExtractionLimits()
    infos = archive.infolist()
    if len(infos) > limits.max_members:
        raise UnsafeArchiveError(f"ZIP 成员过多：{len(infos)} > {limits.max_members}")
    total = 0
    seen: set[tuple[str, ...]] = set()
    seen_casefold: set[tuple[str, ...]] = set()
    for info in infos:
        parts = safe_member_parts(info.filename)
        folded = tuple(part.casefold() for part in parts)
        if parts in seen or folded in seen_casefold:
            raise UnsafeArchiveError(f"ZIP 包含重复或大小写冲突路径：{info.filename!r}")
        seen.add(parts)
        seen_casefold.add(folded)
        _validate_member_type(info)
        size = max(0, int(info.file_size))
        compressed = max(0, int(info.compress_size))
        if size > limits.max_single_file:
            raise UnsafeArchiveError(f"ZIP 单文件过大：{info.filename!r}")
        total += size
        if total > limits.max_total_uncompressed:
            raise UnsafeArchiveError("ZIP 解压总大小超过安全上限")
        if size and compressed == 0 and not info.is_dir():
            raise UnsafeArchiveError(f"ZIP 压缩信息异常：{info.filename!r}")
        if compressed and size / compressed > limits.max_compression_ratio:
            raise UnsafeArchiveError(f"ZIP 压缩比异常：{info.filename!r}")


def safe_extract_zip(
    source: str | os.PathLike[str] | zipfile.ZipFile,
    destination: str | os.PathLike[str],
    *,
    limits: ZipExtractionLimits | None = None,
) -> tuple[Path, ...]:
    """Extract validated regular files/directories and return written paths."""
    root = Path(destination).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    root_resolved = root.resolve(strict=False)
    own_archive = not isinstance(source, zipfile.ZipFile)
    archive = zipfile.ZipFile(source, "r") if own_archive else source
    written: list[Path] = []
    try:
        validate_zip(archive, limits=limits)
        for info in archive.infolist():
            parts = safe_member_parts(info.filename)
            target = root.joinpath(*parts)
            resolved = target.resolve(strict=False)
            if resolved != root_resolved and root_resolved not in resolved.parents:
                raise UnsafeArchiveError(f"ZIP 目标越界：{info.filename!r}")
            if info.is_dir() or info.filename.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with archive.open(info, "r") as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
            except Exception:
                target.unlink(missing_ok=True)
                raise
            written.append(target)
        return tuple(written)
    finally:
        if own_archive:
            archive.close()
