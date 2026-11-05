"""Source collection in ``ui/pages/import_sources.py``.

The module is loaded straight from its file path so that ``ui/__init__.py``
(and therefore PySide6, which is not installed in the test venv) is never
imported.
"""
from __future__ import annotations

import importlib.util
import io
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
    assert not any(name == "PySide6" or name.startswith("PySide6.") for name in sys.modules)


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


def test_import_label_unchanged(import_sources: ModuleType) -> None:
    assert import_sources.import_label(["/books/vol1.png"]) == "vol1"
    assert import_sources.import_label(["/books/a.png", "/books/b.png"]) == "a 等 2 个文件"
    assert import_sources.import_label([], fallback="untitled") == "untitled"
