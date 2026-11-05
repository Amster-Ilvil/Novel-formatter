#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Content-based input file type detection (pure stdlib).

Input sources arrive with unreliable suffixes: OCR tools save PNG bytes under
``.jpg``, downloads lose their extension, archives are renamed to ``.cbz``.
This module sniffs the leading bytes of a file and only falls back to the
extension when no signature matches.  It deliberately has no dependency on
``ui``/PySide6 or third-party packages so it can be used from workers, the
CLI, and tests alike.

Extension policy
----------------
``DetectedType.extension`` is the canonical suffix for the detected content
(``.jpg`` not ``.jpeg``, ``.tiff`` not ``.tif``).  The one exception is comic
archives: a ZIP/RAR named ``.cbz``/``.cbr`` keeps that suffix and gets
``detail="cbz"``/``"cbr"`` so the UI can still present it as a comic book
archive; MIME and ``kind`` stay those of the underlying container.
"""
from __future__ import annotations

import codecs
import struct
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

SNIFF_SIZE = 8192
OCTET_STREAM = "application/octet-stream"

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif", ".ppm"}
)
ARCHIVE_EXTENSIONS: frozenset[str] = frozenset({".zip", ".cbz", ".rar", ".cbr"})
PDF_EXTENSIONS: frozenset[str] = frozenset({".pdf"})
TEXT_EXTENSIONS: frozenset[str] = frozenset({".txt", ".md", ".markdown"})
EPUB_EXTENSIONS: frozenset[str] = frozenset({".epub"})
SOURCE_EXTENSIONS: frozenset[str] = IMAGE_EXTENSIONS | ARCHIVE_EXTENSIONS | PDF_EXTENSIONS

SUPPORTED_SOURCES_HINT = (
    "Supported: images (PNG/JPEG/GIF/BMP/TIFF/WebP/HEIC), PDF, "
    "ZIP/CBZ and RAR/CBR image archives."
)


class FileKind(str, Enum):
    IMAGE = "image"
    PDF = "pdf"
    ARCHIVE_ZIP = "archive_zip"
    ARCHIVE_RAR = "archive_rar"
    TEXT = "text"
    EPUB = "epub"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DetectedType:
    kind: FileKind
    mime: str
    extension: str
    detail: str = ""

    def describe(self) -> str:
        """Human-readable summary, e.g. ``image/png (.png)`` or ``image/png (.png) [by-extension]``."""
        text = f"{self.mime} ({self.extension})" if self.extension else self.mime
        if self.detail:
            text = f"{text} [{self.detail}]"
        return text


_UNKNOWN = DetectedType(FileKind.UNKNOWN, OCTET_STREAM, "")

# --- extension fallback table -------------------------------------------------

_BY_EXTENSION: dict[str, tuple[FileKind, str, str]] = {
    ".png": (FileKind.IMAGE, "image/png", ".png"),
    ".jpg": (FileKind.IMAGE, "image/jpeg", ".jpg"),
    ".jpeg": (FileKind.IMAGE, "image/jpeg", ".jpg"),
    ".gif": (FileKind.IMAGE, "image/gif", ".gif"),
    ".bmp": (FileKind.IMAGE, "image/bmp", ".bmp"),
    ".tif": (FileKind.IMAGE, "image/tiff", ".tiff"),
    ".tiff": (FileKind.IMAGE, "image/tiff", ".tiff"),
    ".webp": (FileKind.IMAGE, "image/webp", ".webp"),
    ".heic": (FileKind.IMAGE, "image/heic", ".heic"),
    ".heif": (FileKind.IMAGE, "image/heif", ".heif"),
    ".ppm": (FileKind.IMAGE, "image/x-portable-pixmap", ".ppm"),
    ".pdf": (FileKind.PDF, "application/pdf", ".pdf"),
    ".zip": (FileKind.ARCHIVE_ZIP, "application/zip", ".zip"),
    ".cbz": (FileKind.ARCHIVE_ZIP, "application/zip", ".cbz"),
    ".rar": (FileKind.ARCHIVE_RAR, "application/vnd.rar", ".rar"),
    ".cbr": (FileKind.ARCHIVE_RAR, "application/vnd.rar", ".cbr"),
    ".epub": (FileKind.EPUB, "application/epub+zip", ".epub"),
    ".txt": (FileKind.TEXT, "text/plain", ".txt"),
    ".md": (FileKind.TEXT, "text/plain", ".txt"),
    ".markdown": (FileKind.TEXT, "text/plain", ".txt"),
}

# --- signatures ---------------------------------------------------------------

_HEIF_BRANDS = frozenset({b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1"})
# DIB header sizes: BITMAPCOREHEADER, INFO, V2, V3, OS/2 v2, V4, V5.
_BMP_DIB_SIZES = frozenset({12, 40, 52, 56, 64, 108, 124})
_PDF_SCAN_WINDOW = 1024
_TEXT_ALLOWED_CONTROLS = frozenset({0x09, 0x0A, 0x0D, 0x0C, 0x1B})
_TEXT_CONTROL_RATIO = 0.05


def _sniff_image(header: bytes) -> tuple[str, str] | None:
    """Return ``(mime, extension)`` for a recognised raster image header."""
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg", ".jpg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png", ".png"
    if header.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif", ".gif"
    if header.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff", ".tiff"
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp", ".webp"
    if header.startswith(b"BM") and len(header) >= 18:
        # "BM" alone is too weak (plain text can start with it); require a sane DIB header size.
        (dib_size,) = struct.unpack_from("<I", header, 14)
        if dib_size in _BMP_DIB_SIZES:
            return "image/bmp", ".bmp"
    if _is_heif(header):
        return "image/heic", ".heic"
    return None


def _is_heif(header: bytes) -> bool:
    """ISO BMFF ``ftyp`` box whose major or compatible brand is a HEIF/HEIC brand."""
    if len(header) < 12 or header[4:8] != b"ftyp":
        return False
    if header[8:12] in _HEIF_BRANDS:
        return True
    (box_size,) = struct.unpack_from(">I", header, 0)
    end = min(len(header), box_size) if box_size >= 16 else len(header)
    for offset in range(16, end - 3, 4):
        if header[offset:offset + 4] in _HEIF_BRANDS:
            return True
    return False


def _is_pdf(header: bytes) -> bool:
    return header[:_PDF_SCAN_WINDOW].find(b"%PDF-") != -1


def _is_zip(header: bytes) -> bool:
    return header.startswith((b"PK\x03\x04", b"PK\x05\x06"))


def _zip_is_epub(header: bytes) -> bool:
    """EPUB OCF: the first local entry must be ``mimetype`` holding ``application/epub+zip``."""
    if not header.startswith(b"PK\x03\x04") or len(header) < 30:
        return False
    name_len, extra_len = struct.unpack_from("<HH", header, 26)
    if header[30:30 + name_len] != b"mimetype":
        return False
    data_start = 30 + name_len + extra_len
    return b"application/epub+zip" in header[data_start:data_start + 64]


def _is_rar(header: bytes) -> bool:
    return header.startswith((b"Rar!\x1a\x07\x00", b"Rar!\x1a\x07\x01\x00"))


def _looks_like_text(header: bytes) -> bool:
    if not header:
        return False
    if header.startswith((b"\xef\xbb\xbf", b"\xff\xfe", b"\xfe\xff")):
        return True
    if b"\x00" in header:
        return False
    decoder = codecs.getincrementaldecoder("utf-8")()
    try:
        # final=False tolerates a multi-byte sequence cut off at the buffer end.
        text = decoder.decode(header, final=False)
    except UnicodeDecodeError:
        return False
    if not text:
        return False
    controls = sum(1 for ch in text if ord(ch) < 0x20 and ord(ch) not in _TEXT_ALLOWED_CONTROLS)
    return controls / len(text) < _TEXT_CONTROL_RATIO


# --- public API ---------------------------------------------------------------


def sniff_bytes(header: bytes, *, extension: str = "") -> DetectedType:
    """Classify a file by its leading bytes; ``extension`` is only a fallback hint.

    A recognised signature always wins over the extension (PNG bytes in a
    ``.jpg`` file yield ``image/png`` / ``.png``).  The extension is consulted
    for comic-archive labelling, Markdown labelling, and as a last resort when
    no signature matches (``detail="by-extension"``).
    """
    header = bytes(header or b"")
    extension = (extension or "").lower()

    image = _sniff_image(header)
    if image is not None:
        mime, ext = image
        return DetectedType(FileKind.IMAGE, mime, ext)
    if _is_pdf(header):
        return DetectedType(FileKind.PDF, "application/pdf", ".pdf")
    if _is_zip(header):
        if _zip_is_epub(header):
            return DetectedType(FileKind.EPUB, "application/epub+zip", ".epub")
        if extension == ".cbz":
            return DetectedType(FileKind.ARCHIVE_ZIP, "application/zip", ".cbz", "cbz")
        return DetectedType(FileKind.ARCHIVE_ZIP, "application/zip", ".zip")
    if _is_rar(header):
        if extension == ".cbr":
            return DetectedType(FileKind.ARCHIVE_RAR, "application/vnd.rar", ".cbr", "cbr")
        return DetectedType(FileKind.ARCHIVE_RAR, "application/vnd.rar", ".rar")
    if _looks_like_text(header):
        detail = "markdown" if extension in {".md", ".markdown"} else ""
        return DetectedType(FileKind.TEXT, "text/plain", ".txt", detail)

    fallback = _BY_EXTENSION.get(extension)
    if fallback is not None:
        kind, mime, ext = fallback
        return DetectedType(kind, mime, ext, "by-extension")
    return _UNKNOWN


def detect_file_type(path: str | Path) -> DetectedType:
    """Sniff the first :data:`SNIFF_SIZE` bytes of ``path``.

    Never raises for missing paths or directories; those come back as
    ``UNKNOWN`` with ``detail`` set to ``"missing"`` / ``"directory"``.
    """
    target = Path(path)
    if target.is_dir():
        return DetectedType(FileKind.UNKNOWN, OCTET_STREAM, "", "directory")
    try:
        with target.open("rb") as handle:
            header = handle.read(SNIFF_SIZE)
    except FileNotFoundError:
        return DetectedType(FileKind.UNKNOWN, OCTET_STREAM, "", "missing")
    except OSError as exc:
        return DetectedType(FileKind.UNKNOWN, OCTET_STREAM, "", f"unreadable: {type(exc).__name__}")
    return sniff_bytes(header, extension=target.suffix.lower())


def is_image(detected: DetectedType) -> bool:
    return detected.kind is FileKind.IMAGE


def is_archive(detected: DetectedType) -> bool:
    return detected.kind in (FileKind.ARCHIVE_ZIP, FileKind.ARCHIVE_RAR)


def is_supported_source(detected: DetectedType) -> bool:
    """True for inputs the OCR pipeline can expand into pages."""
    return detected.kind in (
        FileKind.IMAGE,
        FileKind.PDF,
        FileKind.ARCHIVE_ZIP,
        FileKind.ARCHIVE_RAR,
    )


class UnsupportedFileTypeError(ValueError):
    """Raised when none of the selected inputs is a supported source.

    ``rejected`` keeps every ``(path, DetectedType)`` pair so callers can
    present the full list; the message already includes all paths verbatim.
    """

    def __init__(self, rejected: Iterable[tuple[str | Path, DetectedType]]) -> None:
        self.rejected: list[tuple[str, DetectedType]] = [
            (str(path), detected) for path, detected in rejected
        ]
        lines = ["No supported input sources found. Rejected files:"]
        lines.extend(f"  - {path}: {detected.describe()}" for path, detected in self.rejected)
        lines.append(SUPPORTED_SOURCES_HINT)
        super().__init__("\n".join(lines))


__all__ = [
    "ARCHIVE_EXTENSIONS",
    "DetectedType",
    "EPUB_EXTENSIONS",
    "FileKind",
    "IMAGE_EXTENSIONS",
    "OCTET_STREAM",
    "PDF_EXTENSIONS",
    "SNIFF_SIZE",
    "SOURCE_EXTENSIONS",
    "SUPPORTED_SOURCES_HINT",
    "TEXT_EXTENSIONS",
    "UnsupportedFileTypeError",
    "detect_file_type",
    "is_archive",
    "is_image",
    "is_supported_source",
    "sniff_bytes",
]
