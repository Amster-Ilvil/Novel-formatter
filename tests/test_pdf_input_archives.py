"""Image-archive extraction and content-based input expansion in ``adapters.pdf_input``."""
from __future__ import annotations

import io
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from adapters.file_type import DetectedType, FileKind, UnsupportedFileTypeError, detect_file_type
from adapters.pdf_input import ArchiveExtractionError, expand_inputs, extract_image_archive

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_GARBAGE = b"\x00\x01\x02garbage\x00" * 16


def _image_bytes(fmt: str, color: tuple[int, int, int] = (200, 30, 30)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), color).save(buffer, format=fmt)
    return buffer.getvalue()


def _write_png(path: Path, color: tuple[int, int, int] = (200, 30, 30)) -> Path:
    path.write_bytes(_image_bytes("PNG", color))
    return path


def _zip(path: Path, entries: list[tuple[str, bytes]]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in entries:
            if name.endswith("/"):
                archive.writestr(zipfile.ZipInfo(name), b"")
            else:
                archive.writestr(name, data)
    return path


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, DetectedType]] = []

    def __call__(self, name: str, detected: DetectedType) -> None:
        self.calls.append((name, detected))

    @property
    def names(self) -> list[str]:
        return [name for name, _detected in self.calls]


def _files_under(root: Path) -> list[Path]:
    return [path for path in root.rglob("*") if path.is_file()]


# --- extract_image_archive ------------------------------------------------------


def test_zip_natural_order_and_macos_noise_ignored(tmp_path: Path) -> None:
    archive = _zip(tmp_path / "pages.zip", [
        ("p10.png", _image_bytes("PNG", (10, 10, 10))),
        ("p2.png", _image_bytes("PNG", (20, 20, 20))),
        ("__MACOSX/._p2.png", b"\x00\x05\x16\x07AppleDouble resource fork"),
        ("scans/", b""),
    ])
    assert detect_file_type(archive).kind is FileKind.ARCHIVE_ZIP

    out = tmp_path / "out"
    skipped = _Recorder()
    pages = extract_image_archive(archive, work_dir=out, skipped_callback=skipped)

    assert [Path(page).name for page in pages] == ["00001_p2.png", "00002_p10.png"]
    first, second = pages
    with Image.open(first) as image:
        assert image.getpixel((0, 0)) == (20, 20, 20)
    with Image.open(second) as image:
        assert image.getpixel((0, 0)) == (10, 10, 10)
    # __MACOSX and directory entries are dropped silently, not reported.
    assert skipped.calls == []
    assert sorted(path.name for path in out.iterdir()) == ["00001_p2.png", "00002_p10.png"]


def test_zip_slip_entries_are_skipped_and_reported(tmp_path: Path) -> None:
    """Chosen behaviour: traversal entries are skipped with ``detail="unsafe-path"``.

    The archive as a whole is not rejected, so safe pages still extract; nothing
    is ever written outside ``work_dir``.
    """
    png = _image_bytes("PNG")
    archive = _zip(tmp_path / "evil.zip", [
        ("../evil.png", png),
        ("..\\evil2.png", png),
        ("ok.png", png),
    ])
    out = tmp_path / "out"
    skipped = _Recorder()
    pages = extract_image_archive(archive, work_dir=out, skipped_callback=skipped)

    assert [Path(page).name for page in pages] == ["00001_ok.png"]
    assert not (tmp_path / "evil.png").exists()
    assert not (tmp_path / "evil2.png").exists()
    resolved_out = out.resolve()
    assert all(Path(page).resolve().is_relative_to(resolved_out) for page in pages)
    for path in _files_under(tmp_path):
        assert path == archive or path.resolve().is_relative_to(resolved_out), path
    assert sorted(skipped.names) == ["../evil.png", "../evil2.png"]
    assert {detected.detail for _name, detected in skipped.calls} == {"unsafe-path"}


def test_extension_follows_signature_and_non_images_are_reported(tmp_path: Path) -> None:
    archive = _zip(tmp_path / "mixed.cbz", [
        ("cat.png", _image_bytes("JPEG")),
        ("notes.txt", b"just some notes\n"),
        ("blob.png", _GARBAGE),
    ])
    skipped = _Recorder()
    pages = extract_image_archive(archive, work_dir=tmp_path / "out", skipped_callback=skipped)

    (page,) = pages
    assert Path(page).name == "00001_cat.jpg"
    with Image.open(page) as image:
        assert image.format == "JPEG"
    reported = {name: detected for name, detected in skipped.calls}
    assert set(reported) == {"notes.txt", "blob.png"}
    assert reported.get("notes.txt").kind is FileKind.TEXT
    # A .png suffix without a PNG signature is not trusted inside archives.
    assert reported.get("blob.png").detail == "by-extension"


def test_archive_without_images_raises(tmp_path: Path) -> None:
    archive = _zip(tmp_path / "text-only.zip", [("readme.txt", b"hello\n")])
    with pytest.raises(ArchiveExtractionError, match="text-only.zip"):
        extract_image_archive(archive, work_dir=tmp_path / "out")


def test_cbr_without_rarfile_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "comic.cbr"
    path.write_bytes(b"Rar!\x1a\x07\x01\x00" + b"\x00\x13\x37" * 32)
    detected = detect_file_type(path)
    assert detected.kind is FileKind.ARCHIVE_RAR
    assert detected.detail == "cbr"

    monkeypatch.setitem(sys.modules, "rarfile", None)
    with pytest.raises(ArchiveExtractionError, match="rarfile"):
        extract_image_archive(path, work_dir=tmp_path / "out")


def test_session_cache_reuses_pages_and_replays_skips(tmp_path: Path) -> None:
    archive = _zip(tmp_path / "book.cbz", [
        ("a1.png", _image_bytes("PNG")),
        ("readme.txt", b"not a page\n"),
    ])
    first_skips = _Recorder()
    pages = extract_image_archive(archive, skipped_callback=first_skips)
    second_skips = _Recorder()
    again = extract_image_archive(archive, skipped_callback=second_skips)

    assert again == pages
    (page,) = pages
    assert Path(page).is_file()
    assert "archive-pages" in Path(page).parts
    assert first_skips.names == ["readme.txt"]
    assert second_skips.names == ["readme.txt"]


# --- expand_inputs ----------------------------------------------------------------


def test_expand_inputs_mixed_directory(tmp_path: Path) -> None:
    fitz = pytest.importorskip("fitz")
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    png = _write_png(mixed / "a_page.png")
    document = fitz.open()
    document.new_page()
    document.save(str(mixed / "b_doc.pdf"))
    document.close()
    _zip(mixed / "c_comic.cbz", [("inner.png", _image_bytes("PNG"))])
    (mixed / "notes.txt").write_text("not a page\n", encoding="utf-8")
    (mixed / "blob").write_bytes(_GARBAGE)
    _write_png(mixed / ".hidden.png")

    skipped = _Recorder()
    result = expand_inputs(
        [str(mixed)], work_dir=str(tmp_path), dpi=36, skipped_callback=skipped,
    )

    png_page, pdf_page, cbz_page = result
    assert png_page == str(png)
    assert Path(pdf_page).is_file()
    assert "pdf-pages" in Path(pdf_page).parts
    assert Path(cbz_page).name == "00001_inner.png"
    assert "archive-pages" in Path(cbz_page).parts
    assert not any(".hidden" in page for page in result)
    assert sorted(skipped.names) == sorted([str(mixed / "blob"), str(mixed / "notes.txt")])
    kinds = {Path(name).name: detected.kind for name, detected in skipped.calls}
    assert kinds == {"notes.txt": FileKind.TEXT, "blob": FileKind.UNKNOWN}


def test_expand_inputs_broken_archive_is_reported_not_fatal(tmp_path: Path) -> None:
    folder = tmp_path / "folder"
    folder.mkdir()
    png = _write_png(folder / "page.png")
    (folder / "broken.zip").write_bytes(b"PK\x03\x04" + _GARBAGE)

    skipped = _Recorder()
    assert expand_inputs([str(folder)], skipped_callback=skipped) == [str(png)]
    ((name, detected),) = skipped.calls
    assert name == str(folder / "broken.zip")
    assert detected.kind is FileKind.ARCHIVE_ZIP
    assert "cannot open ZIP" in detected.detail


def test_expand_inputs_only_garbage_raises(tmp_path: Path) -> None:
    folder = tmp_path / "garbage"
    folder.mkdir()
    (folder / "garbage.bin").write_bytes(b"\x00\xff" * 64)

    with pytest.raises(UnsupportedFileTypeError) as info:
        expand_inputs([str(folder)])
    assert "garbage.bin" in str(info.value)
    assert [path for path, _detected in info.value.rejected] == [str(folder / "garbage.bin")]


def test_expand_inputs_missing_path_raises() -> None:
    with pytest.raises(UnsupportedFileTypeError) as info:
        expand_inputs(["/nonexistent/file.png"])
    ((path, detected),) = info.value.rejected
    assert path == "/nonexistent/file.png"
    assert detected.detail == "missing"


def test_expand_inputs_empty_directory_returns_empty(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert expand_inputs([str(empty)]) == []


def test_expand_inputs_backward_compatible_image_list(tmp_path: Path) -> None:
    png = _write_png(tmp_path / "page.png")
    assert expand_inputs([str(png)]) == [str(png)]
    assert expand_inputs([str(png)], str(tmp_path)) == [str(png)]


def test_expand_inputs_sniffs_files_with_image_suffix(tmp_path: Path) -> None:
    """An image suffix is not trusted: UTF-8 text named ``.png`` is rejected."""
    text_named = tmp_path / "text_named.png"
    text_named.write_text("not an image, just notes\nページではない\n", encoding="utf-8")

    with pytest.raises(UnsupportedFileTypeError) as info:
        expand_inputs([str(text_named)])
    ((path, detected),) = info.value.rejected
    assert path == str(text_named)
    assert detected.kind is FileKind.TEXT

    good = _write_png(tmp_path / "good.png")
    skipped = _Recorder()
    result = expand_inputs([str(good), str(text_named)], skipped_callback=skipped)
    assert result == [str(good)]
    assert skipped.names == [str(text_named)]
    assert all(detected.kind is FileKind.TEXT for _name, detected in skipped.calls)


def test_expand_inputs_returns_mislabelled_image_unchanged(tmp_path: Path) -> None:
    payload = _image_bytes("PNG")
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(payload)
    assert detect_file_type(photo).mime == "image/png"

    assert expand_inputs([str(photo)]) == [str(photo)]
    assert photo.read_bytes() == payload
    assert sorted(path.name for path in tmp_path.iterdir()) == ["photo.jpg"]


def test_import_does_not_pull_ui_or_qt() -> None:
    code = (
        "import sys, adapters.pdf_input\n"
        "bad = sorted(m for m in sys.modules if m in ('ui', 'PySide6') "
        "or m.startswith(('ui.', 'PySide6.')))\n"
        "print(bad)\n"
        "raise SystemExit(1 if bad else 0)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_PROJECT_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
