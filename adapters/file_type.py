# -*- coding: utf-8 -*-
"""PR #12 adaptation: detect importable books by signatures, not suffixes.

No files are changed or extracted while sniffing. An inconclusive signature
may use a legacy extension for compatibility, but archive/PDF ingestion does
not trust a suffix alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import struct
import zipfile


class FileKind(str, Enum):
    IMAGE = 'image'
    PDF = 'pdf'
    ZIP = 'zip'
    RAR = 'rar'
    EPUB = 'epub'
    UNKNOWN = 'unknown'


@dataclass(frozen=True)
class DetectedType:
    kind: FileKind
    mime: str = 'application/octet-stream'
    extension: str = ''
    detail: str = ''


_IMAGE_SIGNATURES = (
    (b'\x89PNG\r\n\x1a\n', 'image/png', '.png'),
    (b'\xff\xd8\xff', 'image/jpeg', '.jpg'),
    (b'GIF87a', 'image/gif', '.gif'),
    (b'GIF89a', 'image/gif', '.gif'),
    (b'II*\x00', 'image/tiff', '.tif'),
    (b'MM\x00*', 'image/tiff', '.tif'),
)
_BMP_DIB_SIZES = frozenset({12, 40, 52, 56, 64, 108, 124})
_HEIF_BRANDS = frozenset({b'heic', b'heix', b'hevc', b'hevx', b'heim', b'heis', b'mif1', b'msf1'})
_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff', '.webp', '.heic', '.heif', '.ppm'}


def _is_valid_bmp_header(data: bytes) -> bool:
    """Require a recognised DIB header, not just the ambiguous ASCII 'BM'."""
    return (len(data) >= 18 and data.startswith(b'BM')
            and struct.unpack_from('<I', data, 14)[0] in _BMP_DIB_SIZES)


def _is_heif(data: bytes) -> bool:
    """Detect HEIF major and compatible brands inside the ftyp box."""
    if len(data) < 12 or data[4:8] != b'ftyp':
        return False
    if data[8:12] in _HEIF_BRANDS:
        return True
    box_size = struct.unpack_from('>I', data, 0)[0]
    limit = min(len(data), box_size) if box_size >= 16 else len(data)
    return any(data[i:i + 4] in _HEIF_BRANDS for i in range(16, limit - 3, 4))


def sniff_bytes(data: bytes, *, extension: str = '') -> DetectedType:
    if b'%PDF-' in data[:1024]:
        return DetectedType(FileKind.PDF, 'application/pdf', '.pdf')
    for magic, mime, ext in _IMAGE_SIGNATURES:
        if data.startswith(magic):
            return DetectedType(FileKind.IMAGE, mime, ext)
    if _is_valid_bmp_header(data):
        return DetectedType(FileKind.IMAGE, 'image/bmp', '.bmp')
    if data.startswith(b'RIFF') and data[8:12] == b'WEBP':
        return DetectedType(FileKind.IMAGE, 'image/webp', '.webp')
    if _is_heif(data):
        return DetectedType(FileKind.IMAGE, 'image/heic', '.heic')
    if data.startswith((b'P6\n', b'P5\n', b'P3\n', b'P2\n')):
        return DetectedType(FileKind.IMAGE, 'image/x-portable-pixmap', '.ppm')
    if data.startswith((b'PK\x03\x04', b'PK\x05\x06', b'PK\x07\x08')):
        return DetectedType(FileKind.ZIP, 'application/zip', '.zip')
    if data.startswith((b'Rar!\x1a\x07\x00', b'Rar!\x1a\x07\x01\x00')):
        return DetectedType(FileKind.RAR, 'application/vnd.rar', '.rar')
    ext = str(extension).lower()
    if ext in _IMAGE_EXTENSIONS:
        return DetectedType(FileKind.IMAGE, 'application/octet-stream', ext, 'by-extension')
    return DetectedType(FileKind.UNKNOWN, extension=ext, detail='unsupported')


def detect_file_type(path: str | Path) -> DetectedType:
    source = Path(path)
    with source.open('rb') as f:
        data = f.read(8192)
    detected = sniff_bytes(data, extension=source.suffix.lower())
    if detected.kind is FileKind.ZIP:
        try:
            with zipfile.ZipFile(source) as archive:
                # EPUB identification: mimetype need not be first for malformed
                # but otherwise usable EPUBs. Never treat EPUB as page CBZ.
                if ('mimetype' in archive.namelist() and archive.getinfo('mimetype').file_size <= 256
                        and archive.read('mimetype')[:128].strip() == b'application/epub+zip'):
                    return DetectedType(FileKind.EPUB, 'application/epub+zip', '.epub')
        except (OSError, zipfile.BadZipFile, RuntimeError, ValueError):
            return DetectedType(FileKind.UNKNOWN, detail='broken-zip')
    return detected


def is_supported_source(value: DetectedType) -> bool:
    return value.kind in (FileKind.IMAGE, FileKind.PDF, FileKind.ZIP, FileKind.RAR)
