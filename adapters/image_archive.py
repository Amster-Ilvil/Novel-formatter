# -*- coding: utf-8 -*-
"""PR #12 adaptation: bounded CBZ/ZIP and optional CBR/RAR image import.

Input images are written only to the existing session temporary root. No
original document or user folder is modified. Validates archives before
extracting page bytes; stream limits defend against inconsistent headers.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import threading
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

from adapters.file_type import detect_file_type, sniff_bytes, FileKind
from utils.safe_archive import safe_member_parts, validate_zip, ZipExtractionLimits, UnsafeArchiveError
from utils.session_temp import session_temp_registry

_MAX_FILES = 20_000
_MAX_BYTES = 4 * 1024 ** 3
_MAX_MEMBER = 512 * 1024 ** 2
_ARCHIVE_LOCK = threading.RLock()


class ImageArchiveError(ValueError):
    pass


def _natural_key(name: str):
    import re
    return [int(p) if p.isdigit() else p.casefold() for p in re.split(r'(\d+)', name)]


def _is_noise(name: str) -> bool:
    return any(p.startswith('.') or p == '__MACOSX' for p in name.replace('\\', '/').split('/'))


@contextmanager
def _open_archive(source: Path) -> Iterator[tuple[object, list]]:
    typ = detect_file_type(source)
    if typ.kind is FileKind.ZIP:
        try:
            with zipfile.ZipFile(source) as z:
                validate_zip(z, limits=ZipExtractionLimits(
                    max_members=_MAX_FILES, max_total_uncompressed=_MAX_BYTES,
                    max_single_file=_MAX_MEMBER, max_compression_ratio=1000,
                ))
                yield z, z.infolist()
        except (UnsafeArchiveError, zipfile.BadZipFile, OSError) as exc:
            raise ImageArchiveError(f'归档文件不安全或已损坏：{exc}') from exc
    elif typ.kind is FileKind.RAR:
        try:
            import rarfile
        except ImportError as exc:
            raise ImageArchiveError('读取 CBR/RAR 需要可选的 rarfile 和 unrar/bsdtar；其他图片/PDF 格式不受影响') from exc
        try:
            with rarfile.RarFile(source) as r:
                infos = r.infolist()
                if len(infos) > _MAX_FILES or sum(max(0, x.file_size) for x in infos) > _MAX_BYTES:
                    raise ImageArchiveError('RAR 超过安全解压数量/体积限制')
                yield r, infos
        except ImageArchiveError:
            raise
        except Exception as exc:
            raise ImageArchiveError(f'无法读取 RAR：{exc}') from exc
    else:
        raise ImageArchiveError(f'不支持的归档文件类型：{source.name} ({typ.kind})')


def extract_image_archive(
    source: str | Path, *, cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[str]:
    # A cached extraction must never bypass the current job's Stop request.
    if cancel_check and cancel_check():
        raise ImageArchiveError('归档解压已取消')
    source = Path(source).expanduser().resolve()
    stat_info = source.stat()
    fingerprint = hashlib.sha256(
        f'{source}|{stat_info.st_size}|{stat_info.st_mtime_ns}|archive3'.encode('utf-8')
    ).hexdigest()[:24]
    with _ARCHIVE_LOCK:
        directory = session_temp_registry().path('archive-pages', fingerprint)
        manifest_path = directory / 'manifest.json'
        if manifest_path.exists():
            try:
                metadata = json.loads(manifest_path.read_text(encoding='utf-8'))
                cached = [directory / n for n in metadata['pages']]
                if metadata['fingerprint'] == fingerprint and cached and all(p.is_file() and p.stat().st_size for p in cached):
                    return list(map(str, cached))
            except (ValueError, KeyError, TypeError, OSError):
                pass
        # Never allow an old incomplete extraction to survive a retry.
        if directory.exists():
            for p in directory.iterdir():
                if p.is_file():
                    p.unlink()
        else:
            directory.mkdir(parents=True)
        pages: list[Path] = []
        total_written = 0
        try:
            with _open_archive(source) as (archive, entries):
                entries = sorted(entries, key=lambda e: _natural_key(e.filename))
                total = len(entries)
                for index, entry in enumerate(entries, 1):
                    if cancel_check and cancel_check():
                        raise ImageArchiveError('归档解压已取消')
                    name = entry.filename
                    try:
                        parts = safe_member_parts(name)
                    except UnsafeArchiveError:
                        continue
                    if not parts or _is_noise(name) or entry.is_dir():
                        continue
                    mode = (int(getattr(entry, 'external_attr', 0)) >> 16) & 0xffff
                    if stat.S_ISLNK(mode):
                        continue
                    if int(entry.file_size) > _MAX_MEMBER:
                        raise ImageArchiveError('归档成员超出单文件限制')
                    needs_password = getattr(entry, 'needs_password', None)
                    encrypted = (needs_password() if callable(needs_password) else bool(needs_password))
                    if getattr(entry, 'flag_bits', 0) & 1 or encrypted:
                        continue
                    with archive.open(entry) as stream:
                        first = stream.read(8192)
                        detected = sniff_bytes(first, extension=Path(name).suffix.lower())
                        if detected.kind is not FileKind.IMAGE or detected.detail == 'by-extension':
                            continue
                        safe_stem = ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in Path(name).stem)[:48]
                        output = directory / f'{len(pages)+1:05d}_{safe_stem}{detected.extension}'
                        partial = output.with_suffix(output.suffix + '.part')
                        written = 0
                        try:
                            with partial.open('wb') as dst:
                                dst.write(first)
                                written = len(first)
                                while True:
                                    if cancel_check and cancel_check():
                                        raise ImageArchiveError('归档解压已取消')
                                    chunk = stream.read(1024 * 1024)
                                    if not chunk:
                                        break
                                    written += len(chunk)
                                    if written > _MAX_MEMBER or written + total_written > _MAX_BYTES:
                                        raise ImageArchiveError('归档成员实际大小超过安全限制')
                                    dst.write(chunk)
                            os.replace(partial, output)
                        finally:
                            partial.unlink(missing_ok=True)
                        pages.append(output)
                        total_written += written
                    if progress_callback:
                        progress_callback(index, total)
            if not pages:
                raise ImageArchiveError('归档内没有可识别的图片页面')
            tmp = directory / '.manifest.tmp'
            tmp.write_text(json.dumps({'fingerprint': fingerprint, 'pages': [p.name for p in pages]}), encoding='utf-8')
            os.replace(tmp, manifest_path)
            return [str(p) for p in pages]
        except BaseException:
            for p in pages:
                p.unlink(missing_ok=True)
            raise
