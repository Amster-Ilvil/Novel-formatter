"""Source collection in ``ui/pages/import_sources.py``.

The module is loaded straight from its file path so that ``ui/__init__.py``
(and therefore PySide6) is never imported.  The "no Qt" check runs in a clean
subprocess so it does not depend on which other tests already imported Qt.
"""
from __future__ import annotations

import importlib.util
import io
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest
from PIL import Image

_MODULE_PATH = Path(__file__).resolve().parent.parent / "ui" / "pages" / "import_sources.py"


def _load_import_sources() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_nf_import_sources_under_test", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def import_sources() -> ModuleType:
    return _load_import_sources()


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def _names(paths: list[str]) -> list[str]:
    return list(Path(path).name for path in paths)


@pytest.fixture
def source_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "book"
    folder.mkdir()
    png = _png_bytes()
    (folder / "a.png").write_bytes(png)
    (folder / "b.webp").write_bytes(b"RIFF\x24\x00\x00\x00WEBPVP8 " + b"\x00" * 24)
    with zipfile.ZipFile(folder / "c.cbz", "w") as archive:
        archive.writestr("p1.png", png)
    (folder / "noext").write_bytes(png)
    (folder / "notes.txt").write_text("just notes\n", encoding="utf-8")
    (folder / ".hidden.png").write_bytes(png)
    (folder / "._a.png").write_bytes(b"\x00\x05\x16\x07AppleDouble resource fork")
    (folder / "sub").mkdir()
    return folder


def test_loading_module_does_not_pull_qt(import_sources: ModuleType) -> None:
    assert callable(import_sources.folder_sources)
    probe = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('_nf_probe', {str(_MODULE_PATH)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "assert callable(module.folder_sources)\n"
        "print(any(n == 'PySide6' or n.startswith('PySide6.') for n in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False,
        cwd=str(_MODULE_PATH.parent.parent.parent),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_new_suffixes_are_known_sources(import_sources: ModuleType) -> None:
    assert ".webp" in import_sources.IMAGE_EXTS
    for suffix in (".pdf", ".cbz", ".cbr", ".zip", ".rar", ".webp"):
        assert suffix in import_sources.SOURCE_EXTS


def test_folder_sources_accepts_new_suffixes_and_sniffed_files(
    import_sources: ModuleType, source_folder: Path,
) -> None:
    names = _names(import_sources.folder_sources(source_folder))
    assert names == ["a.png", "b.webp", "c.cbz", "noext"]
    assert "notes.txt" not in names
    assert not any(name.startswith(".") for name in names)


def test_collect_sources_files_and_folders(import_sources: ModuleType, source_folder: Path) -> None:
    result = import_sources.collect_sources([
        source_folder,
        source_folder / "notes.txt",
        source_folder / ".hidden.png",
        source_folder / "noext",
    ])
    assert _names(result) == ["a.png", "b.webp", "c.cbz", "noext", "noext"]


def test_unknown_suffix_is_sniffed_known_non_source_is_not(
    import_sources: ModuleType, tmp_path: Path,
) -> None:
    png = _png_bytes()
    (tmp_path / "scan.dat").write_bytes(png)
    (tmp_path / "blob.bin").write_bytes(b"\x00\xff" * 64)
    with zipfile.ZipFile(tmp_path / "report.docx", "w") as archive:
        archive.writestr("word/media/image1.png", png)
    (tmp_path / "png_bytes.txt").write_bytes(png)

    assert _names(import_sources.folder_sources(tmp_path)) == ["scan.dat"]


@pytest.mark.parametrize("file_first", [False, True])
def test_collected_folder_and_archive_expand_once(
    import_sources: ModuleType, source_folder: Path,
    monkeypatch: pytest.MonkeyPatch, file_first: bool,
) -> None:
    from adapters import pdf_input
    from utils.session_temp import SessionTempRegistry

    archive = source_folder / "c.cbz"
    raw = [archive, source_folder] if file_first else [source_folder, archive]
    registry = SessionTempRegistry()
    monkeypatch.setattr(pdf_input, "session_temp_registry", lambda: registry)
    try:
        collected = import_sources.collect_sources(raw)
        pages = pdf_input.expand_inputs(collected)
        archive_pages = [page for page in pages if "archive-pages" in Path(page).parts]
        assert len(archive_pages) == 1
        first = str(source_folder / "a.png")
        second = str(source_folder / "b.webp")
        noext = str(source_folder / "noext")
        expected = (
            [*archive_pages, first, second, noext]
            if file_first else [first, second, *archive_pages, noext]
        )
        assert pages == expected
        assert Path(first).read_bytes() == Path(noext).read_bytes()
        assert pdf_input.release_pdf_caches(collected) == 1
        assert all(not Path(page).exists() for page in archive_pages)
        assert archive.is_file()
        assert Path(first).is_file()
        assert Path(noext).is_file()
    finally:
        registry.cleanup()


def test_import_label_unchanged(import_sources: ModuleType) -> None:
    assert import_sources.import_label(["/books/vol1.png"]) == "vol1"
    assert import_sources.import_label(["/books/a.png", "/books/b.png"]) == "a 等 2 个文件"
    assert import_sources.import_label([], fallback="untitled") == "untitled"
