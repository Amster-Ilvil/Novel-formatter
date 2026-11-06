"""Image-archive extraction and content-based input expansion in ``adapters.pdf_input``."""
from __future__ import annotations

import io
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from adapters import pdf_input
from adapters.file_type import DetectedType, FileKind, UnsupportedFileTypeError, detect_file_type
from adapters.pdf_input import ArchiveExtractionError, expand_inputs, extract_image_archive
from utils.session_temp import SessionTempRegistry

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


@pytest.mark.parametrize("entry_count", (1, 2))
def test_archive_compression_ratio_rejected_and_reported(
    tmp_path: Path, isolated_page_registry, entry_count: int,
) -> None:
    archive = tmp_path / "compressed.cbz"
    size = 11 * 1024 * 1024 if entry_count == 1 else 6 * 1024 * 1024
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for index in range(entry_count):
            handle.writestr(f"p{index}.png", _image_bytes("PNG") + b"\0" * size)
    out = tmp_path / "out"
    out.mkdir()
    keep = out / "keep"
    keep.write_bytes(b"pre-existing")
    with pytest.raises(ArchiveExtractionError, match="compression ratio"):
        extract_image_archive(archive, work_dir=out)
    assert list(out.iterdir()) == [keep]
    good = _write_png(tmp_path / "good.png")
    skipped = _Recorder()
    assert expand_inputs([str(archive), str(good)], skipped_callback=skipped) == [str(good)]
    assert skipped.names == [str(archive)]
    ((reported_name, detected),) = skipped.calls
    assert reported_name == str(archive)
    assert "compression ratio" in detected.detail
    assert keep.read_bytes() == b"pre-existing"
    assert not list(isolated_page_registry.root.rglob("*.png"))


def test_archive_entry_limit_counts_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pdf_input, "_ARCHIVE_MAX_MEMBERS", 2)
    archive = _zip(tmp_path / "entries.zip", [
        ("a/", b""), ("b/", b""), ("page.png", _image_bytes("PNG")),
    ])
    out = tmp_path / "out"
    with pytest.raises(ArchiveExtractionError, match="too many archive entries"):
        extract_image_archive(archive, work_dir=out)
    assert _files_under(out) == []


@pytest.mark.parametrize("entry_count", (1, 2))
def test_archive_cancel_rolls_back_only_created_pages(
    tmp_path: Path, entry_count: int,
) -> None:
    archive = _zip(tmp_path / "pages.zip", [
        (f"p{index}.png", _image_bytes("PNG")) for index in range(1, entry_count + 1)
    ])
    out = tmp_path / "out"
    out.mkdir()
    keep = out / "keep.png"
    keep.write_bytes(b"pre-existing")
    cancelled = False

    def progress(done: int, total: int) -> None:
        nonlocal cancelled
        cancelled = True

    with pytest.raises(RuntimeError, match="cancelled"):
        extract_image_archive(
            archive, work_dir=out,
            progress_callback=progress, cancel_check=lambda: cancelled,
        )
    assert list(out.iterdir()) == [keep]
    assert keep.read_bytes() == b"pre-existing"


def test_archive_cancel_during_stream_cleans_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _zip(tmp_path / "pages.zip", [("p1.png", _image_bytes("PNG"))])
    out = tmp_path / "out"
    out.mkdir()
    keep = out / ".00001_p1.png.part"
    keep.write_bytes(b"pre-existing partial")
    monkeypatch.setattr(pdf_input, "_ARCHIVE_COPY_CHUNK", 8)
    checks = 0

    def cancelled() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 4

    with pytest.raises(RuntimeError, match="cancelled"):
        extract_image_archive(archive, work_dir=out, cancel_check=cancelled)
    assert checks >= 4
    assert list(out.iterdir()) == [keep]
    assert keep.read_bytes() == b"pre-existing partial"


def test_archive_collision_preserves_existing_file_and_rolls_back(
    tmp_path: Path,
) -> None:
    archive = _zip(tmp_path / "pages.zip", [
        ("p1.png", _image_bytes("PNG")), ("p2.png", _image_bytes("PNG")),
    ])
    out = tmp_path / "out"
    out.mkdir()
    keep = out / "00002_p2.png"
    keep.write_bytes(b"original page")
    with pytest.raises(ArchiveExtractionError, match="refusing to overwrite"):
        extract_image_archive(archive, work_dir=out)
    assert list(out.iterdir()) == [keep]
    assert keep.read_bytes() == b"original page"


def test_archive_stream_limit_rolls_back_with_understated_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    png = _image_bytes("PNG")
    archive = _zip(tmp_path / "pages.zip", [("p1.png", png), ("p2.png", png)])
    original_open = pdf_input._open_archive

    def understated_open(source, detected):
        handle, members, fatal_errors = original_open(source, detected)
        return handle, [replace(member, size=1) for member in members], fatal_errors

    monkeypatch.setattr(pdf_input, "_open_archive", understated_open)
    monkeypatch.setattr(pdf_input, "_ARCHIVE_MAX_TOTAL_BYTES", len(png) + 8)
    out = tmp_path / "out"
    out.mkdir()
    keep = out / "keep"
    keep.write_bytes(b"pre-existing")
    with pytest.raises(ArchiveExtractionError, match="uncompressed size exceeds"):
        extract_image_archive(archive, work_dir=out)
    assert list(out.iterdir()) == [keep]
    assert keep.read_bytes() == b"pre-existing"


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


@pytest.mark.parametrize("kind", ["image", "pdf", "archive"])
@pytest.mark.parametrize("file_first", [False, True])
@pytest.mark.parametrize("alias_kind", ["parent", "symlink"])
def test_expand_inputs_deduplicates_sources_before_expansion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    kind: str, file_first: bool, alias_kind: str,
) -> None:
    folder = tmp_path / "inputs"
    folder.mkdir()
    first = _write_png(folder / "a.png")
    calls: list[Path] = []
    if kind == "image":
        source = _write_png(folder / "c.png")
        expected_pages = [str(source)]
    elif kind == "pdf":
        source = folder / "c.pdf"
        source.write_bytes(b"%PDF-1.4\n")
        expected_pages = ["rendered-p1.png", "rendered-p2.png"]
    else:
        source = _zip(folder / "c.cbz", [("page.png", _image_bytes("PNG"))])
        expected_pages = ["extracted-p1.png", "extracted-p2.png"]

    def expand_source(path, **kwargs):
        calls.append(Path(path).resolve())
        return expected_pages

    if kind == "pdf":
        monkeypatch.setattr(pdf_input, "pdf_to_images", expand_source)
    elif kind == "archive":
        monkeypatch.setattr(pdf_input, "extract_image_archive", expand_source)

    if alias_kind == "parent":
        alias = folder / ".." / folder.name / source.name
    else:
        alias = tmp_path / source.name
        try:
            alias.symlink_to(source)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"Symlinks unavailable: {exc}")

    inputs = [source, folder] if file_first else [folder, source]
    result = expand_inputs([str(path) for path in [*inputs, alias, folder]])
    expected = [*expected_pages, str(first)] if file_first else [str(first), *expected_pages]
    assert result == expected
    if kind != "image":
        assert calls == [source.resolve()]


def test_expand_inputs_preserves_first_image_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_png(tmp_path / "page.png")
    monkeypatch.chdir(tmp_path)
    assert expand_inputs(["page.png", str(source)]) == ["page.png"]


@pytest.fixture
def isolated_page_registry(monkeypatch: pytest.MonkeyPatch):
    registry = SessionTempRegistry()
    monkeypatch.setattr(pdf_input, "session_temp_registry", lambda: registry)
    try:
        yield registry
    finally:
        registry.cleanup()


@pytest.mark.parametrize(
    ("namespace", "suffix", "metadata"),
    [
        ("pdf-pages", ".png", {"backend": "pymupdf"}),
        ("pdf-pages", ".bin", {"backend": "pdf2image"}),
        ("archive-pages", ".dat", {"kind": "archive"}),
    ],
)
def test_release_caches_uses_manifest_source(
    tmp_path: Path, isolated_page_registry,
    namespace: str, suffix: str, metadata: dict,
) -> None:
    registry = isolated_page_registry
    source = tmp_path / f"book{suffix}"
    source.write_bytes(b"original source")
    cache = registry.path(namespace, "selected")
    cache.mkdir()
    (cache / "page.png").write_bytes(b"cached page")
    pdf_input._write_manifest(cache, {"source": str(source.resolve()), **metadata})
    unrelated = registry.path(namespace, "unrelated")
    unrelated.mkdir()
    (unrelated / "keep").write_bytes(b"unrelated cache")
    pdf_input._write_manifest(unrelated, {"source": str(tmp_path / "other.pdf"), **metadata})
    external = tmp_path / "external-cache"
    external.mkdir()
    (external / "keep").write_bytes(b"external data")
    pdf_input._write_manifest(external, {"source": str(source.resolve()), **metadata})

    alias = tmp_path / ".." / tmp_path.name / source.name
    assert pdf_input.release_pdf_caches([str(alias)]) == 1
    assert not cache.exists()
    assert source.read_bytes() == b"original source"
    assert (unrelated / "keep").read_bytes() == b"unrelated cache"
    assert (external / "keep").read_bytes() == b"external data"
    assert pdf_input.release_pdf_caches([str(source)]) == 0


def test_release_archive_cache_after_extraction(
    tmp_path: Path, isolated_page_registry,
) -> None:
    archive = _zip(tmp_path / "book.cbz", [("page.png", _image_bytes("PNG"))])
    pages = extract_image_archive(archive)
    assert all(Path(page).is_file() for page in pages)
    assert pdf_input.release_pdf_caches([str(archive)]) == 1
    assert all(not Path(page).exists() for page in pages)
    assert archive.is_file()


def test_release_caches_does_not_follow_directory_symlinks(
    tmp_path: Path, isolated_page_registry,
) -> None:
    registry = isolated_page_registry
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-1.4\n")
    external = tmp_path / "external"
    external.mkdir()
    (external / "keep").write_bytes(b"external data")
    pdf_input._write_manifest(external, {"source": str(source.resolve())})
    link = registry.path("pdf-pages", "linked")
    try:
        link.symlink_to(external, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Symlinks unavailable: {exc}")
    assert pdf_input.release_pdf_caches([str(source)]) == 0
    assert link.is_symlink()
    assert (external / "keep").read_bytes() == b"external data"


@pytest.mark.parametrize("suffix", ["", ".png"])
@pytest.mark.parametrize("file_first", [False, True])
def test_real_pdf_duplicate_sources_keep_identical_pages_and_release_cache(
    tmp_path: Path, isolated_page_registry, suffix: str, file_first: bool,
) -> None:
    fitz = pytest.importorskip("fitz")
    folder = tmp_path / "inputs"
    folder.mkdir()
    source = folder / f"book{suffix}"
    with fitz.open() as document:
        document.new_page(width=72, height=72)
        document.new_page(width=72, height=72)
        document.save(str(source))
    original = source.read_bytes()
    paths = [source, folder] if file_first else [folder, source]
    pages = expand_inputs([str(path) for path in paths], dpi=36)

    assert len(pages) == 2
    first_page, second_page = pages
    assert first_page != second_page
    assert Path(first_page).read_bytes() == Path(second_page).read_bytes()
    assert pdf_input.release_pdf_caches([str(source)]) == 1
    assert all(not Path(page).exists() for page in pages)
    assert source.read_bytes() == original


def test_distinct_sources_with_identical_content_are_not_deduplicated(tmp_path: Path) -> None:
    first = _write_png(tmp_path / "first.png")
    second = _write_png(tmp_path / "second.png")
    assert first.read_bytes() == second.read_bytes()
    assert expand_inputs([str(first), str(second), str(first)]) == [str(first), str(second)]


def test_zip_absolute_paths_and_symlinks_are_skipped(tmp_path: Path) -> None:
    import stat

    png = _image_bytes("PNG")
    outside = tmp_path / "escaped.png"
    names = [str(outside), "C:/escaped.png", "//server/share/escaped.png", "a/../../escaped.png"]
    archive = _zip(tmp_path / "unsafe.zip", [(name, png) for name in names])
    with zipfile.ZipFile(archive, "a") as handle:
        link = zipfile.ZipInfo("link.png")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) * 65536
        handle.writestr(link, png)
        handle.writestr("safe.png", png)
    skipped = _Recorder()
    out = tmp_path / "out"

    pages = extract_image_archive(archive, work_dir=out, skipped_callback=skipped)

    assert [Path(page).name for page in pages] == ["00001_safe.png"]
    assert not outside.exists()
    assert {name: detected.detail for name, detected in skipped.calls} == {
        **{name: "unsafe-path" for name in names},
        "link.png": "symlink",
    }
    assert list(out.iterdir()) == [Path(page) for page in pages]


@pytest.mark.parametrize("violation", ["members", "zero-compressed-size"])
def test_archive_limits_reject_before_opening_any_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, violation: str,
) -> None:
    from dataclasses import replace

    archive = _zip(tmp_path / "pages.zip", [("page.png", _image_bytes("PNG"))])
    original_open = pdf_input._open_archive

    def modified_open(source, detected):
        handle, members, fatal_errors = original_open(source, detected)
        if violation == "zero-compressed-size":
            members = [
                replace(member, size=pdf_input._ARCHIVE_RATIO_MIN_BYTES + 1, compressed_size=0)
                for member in members
            ]
        return handle, members, fatal_errors

    def forbidden_open(*args, **kwargs):
        pytest.fail("Archive member opened before metadata limits were checked")

    monkeypatch.setattr(pdf_input, "_open_archive", modified_open)
    monkeypatch.setattr(zipfile.ZipFile, "open", forbidden_open)
    if violation == "members":
        monkeypatch.setattr(pdf_input, "_ARCHIVE_MAX_MEMBERS", 0)
    message = "too many archive entries" if violation == "members" else "compression ratio"
    out = tmp_path / "out"
    with pytest.raises(ArchiveExtractionError, match=message):
        extract_image_archive(archive, work_dir=out)
    assert list(out.iterdir()) == []


def test_archive_write_error_rolls_back_completed_and_partial_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_page_registry,
) -> None:
    import errno

    archive = _zip(tmp_path / "pages.zip", [
        ("p1.png", _image_bytes("PNG")), ("p2.png", _image_bytes("PNG")),
    ])
    out = tmp_path / "out"
    out.mkdir()
    keep = out / "keep"
    keep.write_bytes(b"pre-existing")
    original_link = pdf_input.os.link
    published: list[Path] = []

    def failing_link(source, target):
        if published:
            raise OSError(errno.ENOSPC, "synthetic disk full")
        original_link(source, target)
        published.append(Path(target))

    monkeypatch.setattr(pdf_input.os, "link", failing_link)
    with pytest.raises(ArchiveExtractionError, match="synthetic disk full"):
        extract_image_archive(archive, work_dir=out)
    assert len(published) == 1
    assert list(out.iterdir()) == [keep]
    assert keep.read_bytes() == b"pre-existing"

    published.clear()
    good = _write_png(tmp_path / "good.png")
    skipped = _Recorder()
    assert expand_inputs([str(archive), str(good)], skipped_callback=skipped) == [str(good)]
    ((name, detected),) = skipped.calls
    assert name == str(archive)
    assert "synthetic disk full" in detected.detail
    assert not list(isolated_page_registry.root.rglob("*.png"))


def test_archive_read_error_mid_stream_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    png = _image_bytes("PNG")
    archive = _zip(tmp_path / "pages.zip", [("p1.png", png), ("p2.png", png)])
    original_open = zipfile.ZipFile.open
    streams = []

    class BrokenStream(io.BytesIO):
        def read(self, size=-1):
            if self.tell() >= pdf_input._ARCHIVE_SNIFF_BYTES:
                raise OSError("synthetic read failure")
            return super().read(size)

    def broken_open(handle, member, *args, **kwargs):
        if member.filename == "p2.png":
            stream = BrokenStream(png)
            streams.append(stream)
            return stream
        return original_open(handle, member, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", broken_open)
    out = tmp_path / "out"
    out.mkdir()
    keep = out / "keep"
    keep.write_bytes(b"pre-existing")
    completed = []
    with pytest.raises(ArchiveExtractionError, match="synthetic read failure"):
        extract_image_archive(
            archive, work_dir=out,
            progress_callback=lambda done, total: completed.append(done),
        )
    assert completed == list(range(1, 2))
    assert streams and all(stream.closed for stream in streams)
    assert list(out.iterdir()) == [keep]
    assert keep.read_bytes() == b"pre-existing"


def test_rar_paths_and_links_with_synthetic_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    source = tmp_path / "book.cbr"
    source.write_bytes(b"Rar!\x1a\x07\x01\x00")
    png = _image_bytes("PNG")

    def info(name, *, symlink=False, redirection=None):
        return SimpleNamespace(
            filename=name, file_size=len(png), compress_size=len(png),
            is_dir=lambda: False, is_symlink=lambda: symlink,
            needs_password=lambda: False, file_redir=redirection,
        )

    unsafe = ["../escaped.png", "/escaped.png", "C:/escaped.png"]
    entries = [
        *(info(name) for name in unsafe),
        info("symlink.png", symlink=True),
        info("redirect.png", redirection=("synthetic-redirection",)),
        info("safe.png"),
    ]
    opened = []
    closed = []

    def open_member(member):
        opened.append(member.filename)
        return io.BytesIO(png)

    backend = SimpleNamespace(
        infolist=lambda: entries, is_solid=lambda: False,
        open=open_member, close=lambda: closed.append(True),
    )
    monkeypatch.setattr(
        pdf_input, "_import_rarfile",
        lambda path: SimpleNamespace(RarFile=lambda filename: backend),
    )
    skipped = _Recorder()
    pages = extract_image_archive(source, work_dir=tmp_path / "out", skipped_callback=skipped)
    assert [Path(page).name for page in pages] == ["00001_safe.png"]
    assert opened == ["safe.png"]
    assert closed == [True]
    assert {name: detected.detail for name, detected in skipped.calls} == {
        **{name: "unsafe-path" for name in unsafe},
        "symlink.png": "symlink",
        "redirect.png": "symlink",
    }


@pytest.mark.parametrize("solid", [False, True, None])
def test_rar_ratio_uses_only_reliable_sizes_with_synthetic_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, solid: bool | None,
) -> None:
    from types import SimpleNamespace

    source = tmp_path / "book.cbr"
    source.write_bytes(b"Rar!\x1a\x07\x01\x00")
    png = _image_bytes("PNG")
    member = SimpleNamespace(
        filename="page.png", file_size=pdf_input._ARCHIVE_RATIO_MIN_BYTES + 1,
        compress_size=0, is_dir=lambda: False, is_symlink=lambda: False,
        needs_password=lambda: False,
    )
    opened = []
    closed = []

    def open_member(info):
        opened.append(info.filename)
        return io.BytesIO(png)

    backend = SimpleNamespace(
        infolist=lambda: [member], open=open_member, close=lambda: closed.append(True),
    )
    if solid is not None:
        backend.is_solid = lambda: solid
    monkeypatch.setattr(
        pdf_input, "_import_rarfile",
        lambda path: SimpleNamespace(RarFile=lambda filename: backend),
    )
    out = tmp_path / "out"
    if solid is False:
        with pytest.raises(ArchiveExtractionError, match="compression ratio"):
            extract_image_archive(source, work_dir=out)
        assert opened == []
        assert list(out.iterdir()) == []
    else:
        pages = extract_image_archive(source, work_dir=out)
        assert [Path(page).read_bytes() for page in pages] == [png]
        assert opened == ["page.png"]
    assert closed == [True]


def test_session_cleanup_removes_pdf_and_archive_caches_not_sources(
    tmp_path: Path, isolated_page_registry,
) -> None:
    registry = isolated_page_registry
    source = _zip(tmp_path / "book", [("page.png", _image_bytes("PNG"))])
    original = source.read_bytes()
    pages = extract_image_archive(source)
    pdf_cache = registry.path("pdf-pages", "pending-render")
    pdf_cache.mkdir()
    (pdf_cache / "partial.png").write_bytes(b"unfinished raster")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    keep = work_dir / "keep"
    keep.write_bytes(b"unrelated")

    registry.cleanup()

    assert not registry.root.exists()
    assert all(not Path(page).exists() for page in pages)
    assert source.read_bytes() == original
    assert keep.read_bytes() == b"unrelated"


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
