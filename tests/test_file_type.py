"""Tests for adapters.file_type (content sniffing with extension fallback)."""
from __future__ import annotations

import os
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image, features

from adapters.file_type import (
    ARCHIVE_EXTENSIONS,
    IMAGE_EXTENSIONS,
    SOURCE_EXTENSIONS,
    DetectedType,
    FileKind,
    UnsupportedFileTypeError,
    detect_file_type,
    is_archive,
    is_image,
    is_supported_source,
    sniff_bytes,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _write_image(path: Path, fmt: str) -> Path:
    image = Image.new("RGB", (8, 8), (200, 30, 30))
    image.save(path, format=fmt)
    return path


def _random_binary(size: int = 512) -> bytes:
    # Guarantee a NUL byte so the text heuristic can never accept it.
    return os.urandom(size) + b"\x00" + os.urandom(16)


# --- images -------------------------------------------------------------------

_IMAGE_CASES = [
    ("PNG", "image/png", ".png"),
    ("JPEG", "image/jpeg", ".jpg"),
    ("GIF", "image/gif", ".gif"),
    ("BMP", "image/bmp", ".bmp"),
    ("TIFF", "image/tiff", ".tiff"),
    ("WEBP", "image/webp", ".webp"),
]


@pytest.mark.parametrize("fmt,mime,ext", _IMAGE_CASES)
@pytest.mark.parametrize("suffix", [".txt", ""])
def test_real_images_detected_regardless_of_suffix(tmp_path: Path, fmt: str, mime: str, ext: str, suffix: str) -> None:
    if fmt == "WEBP" and not features.check("webp"):
        pytest.skip("Pillow built without WebP support")
    path = _write_image(tmp_path / f"sample_{fmt.lower()}{suffix}", fmt)
    detected = detect_file_type(path)
    assert detected.kind is FileKind.IMAGE
    assert detected.mime == mime
    assert detected.extension == ext
    assert detected.detail == ""
    assert is_image(detected)
    assert is_supported_source(detected)


def test_png_bytes_under_jpg_suffix_report_png(tmp_path: Path) -> None:
    path = _write_image(tmp_path / "mislabeled.jpg", "PNG")
    detected = detect_file_type(path)
    assert detected.kind is FileKind.IMAGE
    assert detected.mime == "image/png"
    assert detected.extension == ".png"


def test_heic_synthetic_header() -> None:
    header = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic"
    detected = sniff_bytes(header)
    assert detected.kind is FileKind.IMAGE
    assert detected.mime == "image/heic"
    assert detected.extension == ".heic"


def test_heic_compatible_brand_only() -> None:
    # Major brand is generic ISO BMFF; HEIF shows up only in compatible brands.
    header = b"\x00\x00\x00\x1cftypisom\x00\x00\x00\x00isommif1heic"
    assert sniff_bytes(header).mime == "image/heic"


def test_plain_text_starting_with_bm_is_not_bmp() -> None:
    detected = sniff_bytes(b"BM is a text file that starts with those letters\n")
    assert detected.kind is FileKind.TEXT


# --- PDF ----------------------------------------------------------------------


def _minimal_pdf(path: Path) -> Path:
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    document.new_page()
    document.save(str(path))
    document.close()
    return path


def test_pdf_with_odd_suffix(tmp_path: Path) -> None:
    path = _minimal_pdf(tmp_path / "document.bin")
    detected = detect_file_type(path)
    assert detected.kind is FileKind.PDF
    assert detected.mime == "application/pdf"
    assert detected.extension == ".pdf"
    assert is_supported_source(detected)


def test_pdf_with_leading_garbage(tmp_path: Path) -> None:
    clean = _minimal_pdf(tmp_path / "clean.pdf")
    dirty = tmp_path / "dirty.bin"
    garbage = bytes(range(1, 101))
    assert len(garbage) == 100 and b"%PDF-" not in garbage
    dirty.write_bytes(garbage + clean.read_bytes())
    assert detect_file_type(dirty).kind is FileKind.PDF


def test_pdf_signature_beyond_window_is_ignored() -> None:
    header = b"A" * 1100 + b"%PDF-1.4"
    assert sniff_bytes(header).kind is not FileKind.PDF


# --- archives -----------------------------------------------------------------


def test_epub_container(tmp_path: Path) -> None:
    path = tmp_path / "book.bin"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        archive.writestr("META-INF/container.xml", "<container/>", compress_type=zipfile.ZIP_DEFLATED)
    detected = detect_file_type(path)
    assert detected.kind is FileKind.EPUB
    assert detected.mime == "application/epub+zip"
    assert detected.extension == ".epub"
    assert not is_supported_source(detected)


def test_zip_with_images(tmp_path: Path) -> None:
    png = _write_image(tmp_path / "page.png", "PNG")
    path = tmp_path / "pages.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.write(png, "page.png")
    detected = detect_file_type(path)
    assert detected.kind is FileKind.ARCHIVE_ZIP
    assert detected.mime == "application/zip"
    assert detected.extension == ".zip"
    assert detected.detail == ""
    assert is_archive(detected)
    assert is_supported_source(detected)


def test_cbz_keeps_comic_suffix(tmp_path: Path) -> None:
    png = _write_image(tmp_path / "page.png", "PNG")
    path = tmp_path / "comic.cbz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.write(png, "page.png")
    detected = detect_file_type(path)
    assert detected.kind is FileKind.ARCHIVE_ZIP
    assert detected.mime == "application/zip"
    assert detected.extension == ".cbz"
    assert detected.detail == "cbz"


def test_empty_zip_end_of_central_directory() -> None:
    assert sniff_bytes(b"PK\x05\x06" + b"\x00" * 18).kind is FileKind.ARCHIVE_ZIP


@pytest.mark.parametrize(
    "signature",
    [b"Rar!\x1a\x07\x01\x00", b"Rar!\x1a\x07\x00"],
    ids=["rar5", "rar4"],
)
def test_rar_signatures(tmp_path: Path, signature: bytes) -> None:
    path = tmp_path / "archive.bin"
    path.write_bytes(signature + os.urandom(256))
    detected = detect_file_type(path)
    assert detected.kind is FileKind.ARCHIVE_RAR
    assert detected.mime == "application/vnd.rar"
    assert detected.extension == ".rar"
    assert is_archive(detected)


def test_cbr_keeps_comic_suffix(tmp_path: Path) -> None:
    path = tmp_path / "comic.cbr"
    path.write_bytes(b"Rar!\x1a\x07\x01\x00" + os.urandom(64))
    detected = detect_file_type(path)
    assert detected.kind is FileKind.ARCHIVE_RAR
    assert detected.extension == ".cbr"
    assert detected.detail == "cbr"


# --- text ---------------------------------------------------------------------


def test_utf8_text_without_bom(tmp_path: Path) -> None:
    path = tmp_path / "notes.bin"
    path.write_text("第一章\n吾輩は猫である。\nПривет, мир!\n", encoding="utf-8")
    detected = detect_file_type(path)
    assert detected.kind is FileKind.TEXT
    assert detected.mime == "text/plain"
    assert detected.extension == ".txt"
    assert detected.detail == ""


def test_utf8_text_with_bom(tmp_path: Path) -> None:
    path = tmp_path / "notes.bin"
    path.write_bytes(b"\xef\xbb\xbf" + "hello\n".encode("utf-8"))
    assert detect_file_type(path).kind is FileKind.TEXT


def test_utf16le_text_with_bom(tmp_path: Path) -> None:
    path = tmp_path / "notes.bin"
    path.write_bytes(b"\xff\xfe" + "hello\n".encode("utf-16-le"))
    assert detect_file_type(path).kind is FileKind.TEXT


def test_markdown_detail(tmp_path: Path) -> None:
    path = tmp_path / "README.md"
    path.write_text("# Title\n\nBody text.\n", encoding="utf-8")
    detected = detect_file_type(path)
    assert detected.kind is FileKind.TEXT
    assert detected.detail == "markdown"


def test_truncated_multibyte_tail_is_still_text() -> None:
    data = ("あ" * 100).encode("utf-8")
    # Chop in the middle of the last 3-byte character.
    assert sniff_bytes(data[:-1]).kind is FileKind.TEXT


def test_empty_buffer_is_unknown() -> None:
    detected = sniff_bytes(b"")
    assert detected.kind is FileKind.UNKNOWN
    assert detected.extension == ""


# --- fallback / unknown ---------------------------------------------------------


def test_random_binary_without_suffix_is_unknown(tmp_path: Path) -> None:
    path = tmp_path / "blob"
    path.write_bytes(_random_binary())
    detected = detect_file_type(path)
    assert detected.kind is FileKind.UNKNOWN
    assert detected.mime == "application/octet-stream"
    assert detected.extension == ""
    assert not is_supported_source(detected)


def test_random_binary_with_png_suffix_falls_back_to_extension(tmp_path: Path) -> None:
    path = tmp_path / "blob.png"
    path.write_bytes(_random_binary())
    detected = detect_file_type(path)
    assert detected.kind is FileKind.IMAGE
    assert detected.mime == "image/png"
    assert detected.extension == ".png"
    assert detected.detail == "by-extension"


def test_extension_fallback_normalises_aliases() -> None:
    assert sniff_bytes(_random_binary(), extension=".JPEG").extension == ".jpg"
    assert sniff_bytes(_random_binary(), extension=".tif").mime == "image/tiff"
    assert sniff_bytes(_random_binary(), extension=".cbr").kind is FileKind.ARCHIVE_RAR


def test_missing_and_directory_paths(tmp_path: Path) -> None:
    missing = detect_file_type(tmp_path / "does-not-exist.png")
    assert missing.kind is FileKind.UNKNOWN
    assert missing.detail == "missing"
    directory = detect_file_type(tmp_path)
    assert directory.kind is FileKind.UNKNOWN
    assert directory.detail == "directory"


def test_exported_extension_sets() -> None:
    assert {".png", ".jpg", ".jpeg", ".heic", ".webp", ".ppm"} <= IMAGE_EXTENSIONS
    assert ARCHIVE_EXTENSIONS == {".zip", ".cbz", ".rar", ".cbr"}
    assert ".pdf" in SOURCE_EXTENSIONS
    assert IMAGE_EXTENSIONS <= SOURCE_EXTENSIONS
    assert ARCHIVE_EXTENSIONS <= SOURCE_EXTENSIONS
    assert ".epub" not in SOURCE_EXTENSIONS
    assert ".txt" not in SOURCE_EXTENSIONS


# --- describe / error -------------------------------------------------------------


def test_describe_formats() -> None:
    assert DetectedType(FileKind.IMAGE, "image/png", ".png").describe() == "image/png (.png)"
    assert (
        DetectedType(FileKind.IMAGE, "image/png", ".png", "by-extension").describe()
        == "image/png (.png) [by-extension]"
    )
    assert DetectedType(FileKind.UNKNOWN, "application/octet-stream", "").describe() == "application/octet-stream"


def test_unsupported_error_lists_every_path(tmp_path: Path) -> None:
    first = tmp_path / "chapter one.epub"
    second = Path("/some/where/notes.txt")
    first_type = DetectedType(FileKind.EPUB, "application/epub+zip", ".epub")
    second_type = DetectedType(FileKind.TEXT, "text/plain", ".txt")
    rejected = [(first, first_type), (second, second_type)]
    error = UnsupportedFileTypeError(rejected)
    message = str(error)
    assert isinstance(error, ValueError)
    assert message.startswith("No supported input sources found. Rejected files:")
    assert str(first) in message
    assert str(second) in message
    assert "application/epub+zip (.epub)" in message
    assert "Supported: images (PNG/JPEG/GIF/BMP/TIFF/WebP/HEIC), PDF, ZIP/CBZ and RAR/CBR image archives." in message
    assert error.rejected == [(str(first), first_type), (str(second), second_type)]


# --- import hygiene -----------------------------------------------------------------


def test_import_does_not_pull_ui_or_qt() -> None:
    code = (
        "import sys\n"
        "import adapters.file_type\n"
        "bad = sorted(m for m in sys.modules if m == 'ui' or m.startswith(('ui.', 'PySide6')))\n"
        "print(repr(bad))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "[]", result.stdout
