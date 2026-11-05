#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast PDF/image input expansion with session-owned raster caching.

PDF pages are rendered once per source fingerprint into the application's
session temporary root.  Page Manager, OCR preview, and the OCR worker reuse the
same files.  No ``_pdf_*`` directories are written beside the user's book, and
all generated pages are removed on application shutdown (or reclaimed after a
crash on the next launch).
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import re
import stat
import threading
import zipfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from adapters.file_type import (
    ARCHIVE_EXTENSIONS,
    IMAGE_EXTENSIONS,
    OCTET_STREAM,
    DetectedType,
    FileKind,
    UnsupportedFileTypeError,
    detect_file_type,
    sniff_bytes,
)
from utils.safe_archive import UnsafeArchiveError, safe_member_parts
from utils.session_temp import session_temp_registry

_NUM_RE = re.compile(r"(\d+)")
# Legacy suffixes stay explicit; IMAGE_EXTENSIONS adds .webp/.heif.
_IMAGE_EXTS = {
    ".png", ".jpg", ".jpeg", ".heic", ".tif", ".tiff", ".bmp", ".gif", ".ppm",
} | set(IMAGE_EXTENSIONS)
_ARCHIVE_EXTS = {".zip", ".cbz", ".rar", ".cbr"} | set(ARCHIVE_EXTENSIONS)
_CACHE_VERSION = 2
_ARCHIVE_MAX_MEMBERS = 20_000
_ARCHIVE_MAX_TOTAL_BYTES = 4 * 1024 * 1024 * 1024
_ARCHIVE_SNIFF_BYTES = 64
_ARCHIVE_COPY_CHUNK = 1024 * 1024
_RAR_UNAVAILABLE = (
    "RAR support requires the optional `rarfile` package and an unrar/bsdtar binary"
)
_RENDER_LOCKS: dict[str, threading.RLock] = {}
_RENDER_LOCKS_GUARD = threading.Lock()

ProgressCallback = Callable[[int, int], None]
PageReadyCallback = Callable[[str, int, int], None]
SkippedCallback = Callable[[str, DetectedType], None]

_PDF_RENDER_CANCEL_EVENT = None


def _init_pdf_render_worker(cancel_event) -> None:
    global _PDF_RENDER_CANCEL_EVENT
    _PDF_RENDER_CANCEL_EVENT = cancel_event


def _render_pdf_chunk(pdf_path: str, indices: list[int], targets: list[str], dpi: int) -> list[int]:
    """Render one chunk in an isolated process; PyMuPDF is not thread-safe."""
    import fitz

    completed: list[int] = []
    matrix = fitz.Matrix(max(0.25, float(dpi) / 72.0), max(0.25, float(dpi) / 72.0))
    with fitz.open(pdf_path) as document:
        for index, target in zip(indices, targets):
            if _PDF_RENDER_CANCEL_EVENT is not None and _PDF_RENDER_CANCEL_EVENT.is_set():
                break
            page = document.load_page(index)
            pix = page.get_pixmap(
                matrix=matrix,
                colorspace=fitz.csRGB,
                alpha=False,
                annots=False,
            )
            _save_pixmap_fast(pix, Path(target))
            completed.append(index)
    return completed


def natural_sort_key(path) -> list:
    name = Path(path).name
    return [int(tok) if tok.isdigit() else tok.lower() for tok in _NUM_RE.split(name)]


def pdf_available() -> tuple[bool, str]:
    try:
        import fitz  # noqa: F401
        return True, "pymupdf"
    except ImportError:
        pass
    try:
        import pdf2image  # noqa: F401
        return True, "pdf2image"
    except ImportError:
        pass
    return False, ""


def _source_fingerprint(pdf_path: str, dpi: int) -> str:
    source = Path(pdf_path).expanduser().resolve()
    stat = source.stat()
    payload = "|".join((
        str(_CACHE_VERSION), str(source), str(stat.st_size),
        str(stat.st_mtime_ns), str(int(dpi)),
    ))
    return hashlib.sha256(payload.encode("utf-8", errors="surrogatepass")).hexdigest()[:24]


def _render_lock(key: str) -> threading.RLock:
    with _RENDER_LOCKS_GUARD:
        return _RENDER_LOCKS.setdefault(key, threading.RLock())


def _cache_dir(pdf_path: str, dpi: int) -> Path:
    registry = session_temp_registry()
    root = registry.path("pdf-pages", _source_fingerprint(pdf_path, dpi))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _read_manifest(directory: Path) -> dict:
    try:
        return json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_manifest(directory: Path, payload: dict) -> None:
    target = directory / "manifest.json"
    temporary = directory / ".manifest.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, target)


def _page_paths(directory: Path, stem: str, page_count: int) -> list[Path]:
    safe_stem = re.sub(r"[^\w.-]+", "_", stem, flags=re.UNICODE).strip("._")[:64] or "document"
    return [directory / f"{safe_stem}_p{index:05d}.png" for index in range(1, page_count + 1)]


def _save_pixmap_fast(pix, target: Path) -> None:
    """Write a lossless PNG with low compression latency and atomic replace."""
    temporary = target.with_name(f".{target.name}.{threading.get_ident()}.tmp.png")
    try:
        pil_save = getattr(pix, "pil_save", None)
        if callable(pil_save):
            pil_save(str(temporary), format="PNG", compress_level=1, optimize=False)
        else:
            pix.save(str(temporary))
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def pdf_page_count(pdf_path: str) -> int:
    ok, backend = pdf_available()
    if not ok:
        raise RuntimeError(
            "未安装 PDF 转换库。请运行：pip3 install pymupdf，"
            "或安装 pdf2image + poppler。"
        )
    if backend == "pymupdf":
        import fitz
        with fitz.open(pdf_path) as document:
            return int(document.page_count)
    from pdf2image.pdf2image import pdfinfo_from_path
    info = pdfinfo_from_path(pdf_path)
    return int(info.get("Pages", 0) or 0)


def pdf_to_images(
    pdf_path: str,
    out_dir: str | None = None,
    dpi: int = 200,
    *,
    progress_callback: ProgressCallback | None = None,
    page_ready_callback: PageReadyCallback | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[str]:
    """Render a PDF once and return stable session-local page image paths.

    ``out_dir`` remains accepted for API compatibility but is intentionally not
    used for PDFs.  Earlier versions wrote hundreds of pages beside the source
    file; the session cache prevents both repeated rendering and leaked local
    folders.
    """
    del out_dir
    source = Path(pdf_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(str(source))
    ok, backend = pdf_available()
    if not ok:
        raise RuntimeError(
            "未安装 PDF 转换库。请运行以下任一命令后重试：\n"
            "  pip3 install pymupdf\n"
            "  或\n"
            "  pip3 install pdf2image  &&  brew install poppler"
        )

    key = _source_fingerprint(str(source), int(dpi))
    directory = _cache_dir(str(source), int(dpi))
    with _render_lock(key):
        if backend == "pymupdf":
            import fitz
            document = fitz.open(str(source))
            try:
                total = int(document.page_count)
                pages = _page_paths(directory, source.stem, total)
                manifest = _read_manifest(directory)
                manifest_ok = (
                    int(manifest.get("version", 0) or 0) == _CACHE_VERSION
                    and int(manifest.get("dpi", 0) or 0) == int(dpi)
                    and int(manifest.get("page_count", -1) or -1) == total
                    and str(manifest.get("fingerprint", "")) == key
                )
                if not manifest_ok:
                    for stale in directory.glob("*.png"):
                        stale.unlink(missing_ok=True)

                missing_indices = [
                    index for index, target in enumerate(pages)
                    if not target.exists() or target.stat().st_size < 64
                ]
                missing_set = set(missing_indices)
                completed_count = total - len(missing_indices)
                # Cached pages are immediately usable by the page manager.  The
                # callback is deliberately emitted in page order, while newly
                # rendered pages may arrive out of order from the worker pool.
                if page_ready_callback is not None:
                    for index, target in enumerate(pages):
                        if index not in missing_set:
                            page_ready_callback(str(target), index + 1, total)
                if progress_callback is not None and completed_count:
                    progress_callback(completed_count, total)

                if missing_indices:
                    # PyMuPDF explicitly does not support concurrent rendering
                    # from multiple threads. Spawn isolated processes instead;
                    # the GUI's import worker remains responsive without risking
                    # interpreter crashes from threaded MuPDF calls.
                    worker_count = min(3, max(1, (os.cpu_count() or 2) // 2), len(missing_indices))
                    chunk_size = min(32, max(1, (len(missing_indices) + worker_count - 1) // worker_count))
                    chunks = [missing_indices[offset:offset + chunk_size]
                              for offset in range(0, len(missing_indices), chunk_size)]
                    context = multiprocessing.get_context("spawn")
                    cancel_event = context.Event()
                    pending = {}
                    with ProcessPoolExecutor(
                        max_workers=worker_count,
                        mp_context=context,
                        initializer=_init_pdf_render_worker,
                        initargs=(cancel_event,),
                    ) as executor:
                        for chunk in chunks:
                            targets = [str(pages[index]) for index in chunk]
                            future = executor.submit(
                                _render_pdf_chunk, str(source), chunk, targets, int(dpi),
                            )
                            pending[future] = chunk
                        while pending:
                            if cancel_check is not None and cancel_check():
                                cancel_event.set()
                                raise RuntimeError("PDF 加载已取消")
                            done, _ = wait(
                                pending, timeout=0.2, return_when=FIRST_COMPLETED,
                            )
                            for future in done:
                                chunk = pending.pop(future)
                                rendered_indices = future.result()
                                for page_index in rendered_indices:
                                    if page_ready_callback is not None:
                                        page_ready_callback(
                                            str(pages[page_index]), page_index + 1, total,
                                        )
                                    completed_count += 1
                                    if progress_callback is not None:
                                        progress_callback(completed_count, total)
                                if len(rendered_indices) != len(chunk):
                                    if cancel_check is not None and cancel_check():
                                        cancel_event.set()
                                        raise RuntimeError("PDF 加载已取消")
                _write_manifest(directory, {
                    "version": _CACHE_VERSION,
                    "fingerprint": key,
                    "source": str(source),
                    "dpi": int(dpi),
                    "page_count": total,
                    "backend": "pymupdf",
                })
                return [str(path) for path in pages]
            finally:
                document.close()

        # Compatibility fallback.  Render in bounded chunks so pdf2image never
        # holds an entire long book in RAM at once.
        from pdf2image import convert_from_path
        total = pdf_page_count(str(source))
        pages = _page_paths(directory, source.stem, total)
        if page_ready_callback is not None:
            for index, target in enumerate(pages, start=1):
                if target.exists() and target.stat().st_size >= 64:
                    page_ready_callback(str(target), index, total)
        chunk_size = 8
        for first in range(1, total + 1, chunk_size):
            if cancel_check is not None and cancel_check():
                raise RuntimeError("PDF 加载已取消")
            last = min(total, first + chunk_size - 1)
            missing = [index for index in range(first, last + 1) if not pages[index - 1].exists()]
            if missing:
                converted = convert_from_path(
                    str(source), dpi=int(dpi), first_page=first, last_page=last,
                    fmt="png", thread_count=min(2, max(1, os.cpu_count() or 1)),
                )
                try:
                    for offset, image in enumerate(converted):
                        index = first + offset
                        target = pages[index - 1]
                        temporary = target.with_name(f".{target.name}.tmp.png")
                        image.save(temporary, "PNG", compress_level=1, optimize=False)
                        os.replace(temporary, target)
                        if page_ready_callback is not None:
                            page_ready_callback(str(target), index, total)
                finally:
                    for image in converted:
                        try:
                            image.close()
                        except Exception:
                            pass
            for index in range(first, last + 1):
                if progress_callback is not None:
                    progress_callback(index, total)
        _write_manifest(directory, {
            "version": _CACHE_VERSION,
            "fingerprint": key,
            "source": str(source),
            "dpi": int(dpi),
            "page_count": total,
            "backend": "pdf2image",
        })
        return [str(path) for path in pages]


class ArchiveExtractionError(RuntimeError):
    """An image archive cannot be opened, violates safety limits, or holds no images."""


@dataclass(frozen=True)
class _ArchiveMember:
    name: str
    size: int
    is_dir: bool
    is_link: bool
    encrypted: bool
    info: Any


def _natural_member_key(name: str) -> list:
    """Natural sort key over the full POSIX member path (``p2`` < ``p10``)."""
    return [int(tok) if tok.isdigit() else tok.lower() for tok in _NUM_RE.split(name)]


def _skip_type(reason: str) -> DetectedType:
    return DetectedType(FileKind.UNKNOWN, OCTET_STREAM, "", reason)


def _archive_fingerprint(source: Path) -> str:
    info = source.stat()
    payload = "|".join((
        str(_CACHE_VERSION), "archive", str(source), str(info.st_size),
        str(info.st_mtime_ns),
    ))
    return hashlib.sha256(payload.encode("utf-8", errors="surrogatepass")).hexdigest()[:24]


def _safe_member_stem(name: str) -> str:
    stem = PurePosixPath(name).stem
    return re.sub(r"[^\w.-]+", "_", stem, flags=re.UNICODE).strip("._")[:64] or "page"


def _import_rarfile(source: Path) -> Any:
    """Import ``rarfile`` lazily so a missing optional dependency is reported per call."""
    try:
        import rarfile
    except ImportError:
        rarfile = None
    if rarfile is None:
        raise ArchiveExtractionError(f"{source}: {_RAR_UNAVAILABLE}")
    return rarfile


def _open_archive(
    source: Path, detected: DetectedType,
) -> tuple[Any, list[_ArchiveMember], tuple[type[BaseException], ...]]:
    """Open a ZIP/RAR container and normalise its entries.

    Returns the open archive, its members, and the exception types that must
    abort the whole extraction instead of skipping one entry.
    """
    if detected.kind is FileKind.ARCHIVE_ZIP:
        try:
            archive = zipfile.ZipFile(source, "r")
        except (zipfile.BadZipFile, OSError) as exc:
            raise ArchiveExtractionError(f"{source}: cannot open ZIP archive ({exc})") from exc
        members: list[_ArchiveMember] = []
        for info in archive.infolist():
            name = info.filename.replace("\\", "/")
            mode = (int(info.external_attr) >> 16) & 0xFFFF
            members.append(_ArchiveMember(
                name=name,
                size=max(0, int(info.file_size)),
                is_dir=info.is_dir() or name.endswith("/"),
                is_link=stat.S_ISLNK(mode),
                encrypted=bool(info.flag_bits & 0x1),
                info=info,
            ))
        return archive, members, ()
    if detected.kind is FileKind.ARCHIVE_RAR:
        rarfile = _import_rarfile(source)
        try:
            archive = rarfile.RarFile(str(source))
            infos = list(archive.infolist())
        except Exception as exc:
            raise ArchiveExtractionError(f"{source}: cannot open RAR archive ({exc})") from exc
        members = []
        for info in infos:
            name = str(info.filename).replace("\\", "/")
            is_symlink = getattr(info, "is_symlink", None)
            needs_password = getattr(info, "needs_password", None)
            members.append(_ArchiveMember(
                name=name,
                size=max(0, int(getattr(info, "file_size", 0) or 0)),
                is_dir=bool(info.is_dir()) or name.endswith("/"),
                is_link=bool(is_symlink()) if callable(is_symlink) else False,
                encrypted=bool(needs_password()) if callable(needs_password) else False,
                info=info,
            ))
        fatal = tuple(
            exc_type for exc_type in (getattr(rarfile, "RarCannotExec", None),)
            if isinstance(exc_type, type) and issubclass(exc_type, BaseException)
        )
        return archive, members, fatal
    raise ArchiveExtractionError(
        f"{source}: not a ZIP/CBZ or RAR/CBR archive ({detected.describe()})"
    )


def _read_up_to(stream: Any, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _stream_member(stream: Any, header: bytes, target: Path, *, budget: int, source: Path) -> int:
    """Write ``header`` plus the rest of ``stream`` atomically, enforcing ``budget`` bytes."""
    temporary = target.with_name(f".{target.name}.part")
    written = 0
    try:
        with temporary.open("wb") as handle:
            chunk = header
            while chunk:
                written += len(chunk)
                if written > budget:
                    raise ArchiveExtractionError(
                        f"{source}: uncompressed size exceeds {_ARCHIVE_MAX_TOTAL_BYTES} bytes"
                    )
                handle.write(chunk)
                chunk = stream.read(_ARCHIVE_COPY_CHUNK)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return written


def _extract_archive_images(
    source: Path,
    root: Path,
    *,
    cancel_check: Callable[[], bool] | None,
    progress_callback: ProgressCallback | None,
    skipped_callback: SkippedCallback | None,
) -> tuple[list[str], list[tuple[str, DetectedType]]]:
    skipped: list[tuple[str, DetectedType]] = []

    def skip(name: str, detected: DetectedType) -> None:
        skipped.append((name, detected))
        if skipped_callback is not None:
            skipped_callback(name, detected)

    root.mkdir(parents=True, exist_ok=True)
    root_resolved = root.resolve()
    archive, members, fatal_errors = _open_archive(source, detect_file_type(source))
    try:
        if len(members) > _ARCHIVE_MAX_MEMBERS:
            raise ArchiveExtractionError(
                f"{source}: too many archive entries ({len(members)} > {_ARCHIVE_MAX_MEMBERS})"
            )
        declared = sum(member.size for member in members if not member.is_dir)
        if declared > _ARCHIVE_MAX_TOTAL_BYTES:
            raise ArchiveExtractionError(
                f"{source}: declared uncompressed size {declared} exceeds "
                f"{_ARCHIVE_MAX_TOTAL_BYTES} bytes"
            )

        candidates: list[_ArchiveMember] = []
        for member in members:
            if member.is_dir:
                continue
            try:
                parts = safe_member_parts(member.name)
            except UnsafeArchiveError:
                skip(member.name, _skip_type("unsafe-path"))
                continue
            # macOS resource forks and dotfiles are archive noise, not pages.
            if not parts or any(part == "__MACOSX" or part.startswith(".") for part in parts):
                continue
            if member.is_link:
                skip(member.name, _skip_type("symlink"))
                continue
            if member.encrypted:
                skip(member.name, _skip_type("encrypted"))
                continue
            candidates.append(member)
        candidates.sort(key=lambda member: _natural_member_key(member.name))

        total = len(candidates)
        written_total = 0
        pages: list[str] = []
        for done, member in enumerate(candidates, start=1):
            if cancel_check is not None and cancel_check():
                raise RuntimeError(f"Archive extraction cancelled: {source}")
            suffix = PurePosixPath(member.name).suffix.lower()
            accepted = False
            try:
                with archive.open(member.info) as stream:
                    header = _read_up_to(stream, _ARCHIVE_SNIFF_BYTES)
                    detected = sniff_bytes(header, extension=suffix)
                    if detected.kind is FileKind.IMAGE and detected.detail != "by-extension":
                        name = f"{len(pages) + 1:05d}_{_safe_member_stem(member.name)}{detected.extension}"
                        resolved = (root / name).resolve(strict=False)
                        if not resolved.is_relative_to(root_resolved):
                            raise ArchiveExtractionError(
                                f"{source}: entry escapes extraction root: {member.name!r}"
                            )
                        written_total += _stream_member(
                            stream, header, resolved,
                            budget=_ARCHIVE_MAX_TOTAL_BYTES - written_total, source=source,
                        )
                        pages.append(str(resolved))
                        accepted = True
            except ArchiveExtractionError:
                raise
            except fatal_errors as exc:
                raise ArchiveExtractionError(f"{source}: {_RAR_UNAVAILABLE} ({exc})") from exc
            except Exception as exc:
                detected = _skip_type(f"unreadable: {type(exc).__name__}")
            if not accepted:
                skip(member.name, detected)
            if progress_callback is not None:
                progress_callback(done, total)

        if not pages:
            raise ArchiveExtractionError(f"No images found in archive: {source}")
        return pages, skipped
    finally:
        archive.close()


def _cached_archive_pages(
    directory: Path, key: str,
) -> tuple[list[str], list[tuple[str, DetectedType]]] | None:
    manifest = _read_manifest(directory)
    if (
        int(manifest.get("version", 0) or 0) != _CACHE_VERSION
        or manifest.get("kind") != "archive"
        or str(manifest.get("fingerprint", "")) != key
    ):
        return None
    names = manifest.get("pages")
    if not isinstance(names, list) or not names:
        return None
    pages = [directory / str(name) for name in names]
    if not all(page.is_file() for page in pages):
        return None
    skipped: list[tuple[str, DetectedType]] = []
    for item in manifest.get("skipped") or []:
        try:
            skipped.append((str(item.get("name", "")), DetectedType(
                FileKind(item.get("kind", FileKind.UNKNOWN.value)),
                str(item.get("mime", OCTET_STREAM)),
                str(item.get("extension", "")),
                str(item.get("detail", "")),
            )))
        except (AttributeError, ValueError):
            continue
    return [str(page) for page in pages], skipped


def extract_image_archive(
    path: str | Path,
    work_dir: str | Path | None = None,
    *,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: ProgressCallback | None = None,
    skipped_callback: SkippedCallback | None = None,
) -> list[str]:
    """Extract the images of a ZIP/CBZ or RAR/CBR archive into flat, ordered files.

    Entries are ordered naturally by their full member path and written as
    ``00001_<stem><ext>`` where ``<ext>`` comes from the sniffed signature, so
    a JPEG stored as ``cat.png`` becomes ``00001_cat.jpg``.  Only entries with a
    confirmed image signature are kept; everything else (text, unknown data,
    unsafe ``..``/absolute paths, symlinks, encrypted entries) is reported via
    ``skipped_callback`` and left out.  Directories, ``__MACOSX/`` and hidden
    ``.``/``._`` entries are ignored silently.

    With ``work_dir`` the images are written there.  Without it they go to the
    session temporary root under a source fingerprint (path/size/mtime) and a
    manifest makes repeated calls free, mirroring :func:`pdf_to_images`.

    Raises :class:`ArchiveExtractionError` for unreadable or unsupported
    containers, a missing ``rarfile`` backend, exceeded limits (20 000 entries,
    4 GiB uncompressed) and archives without any image.  Cancellation raises
    ``RuntimeError`` so callers do not mistake it for a rejected file.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ArchiveExtractionError(f"Archive not found: {source}")
    if work_dir is not None:
        pages, _skipped = _extract_archive_images(
            source, Path(work_dir).expanduser(),
            cancel_check=cancel_check,
            progress_callback=progress_callback,
            skipped_callback=skipped_callback,
        )
        return pages

    key = _archive_fingerprint(source)
    with _render_lock(key):
        directory = session_temp_registry().path("archive-pages", key)
        directory.mkdir(parents=True, exist_ok=True)
        cached = _cached_archive_pages(directory, key)
        if cached is not None:
            pages, skipped = cached
            if skipped_callback is not None:
                for name, detected in skipped:
                    skipped_callback(name, detected)
            if progress_callback is not None:
                progress_callback(len(pages), len(pages))
            return pages
        for stale in directory.iterdir():
            if stale.is_file():
                stale.unlink(missing_ok=True)
        pages, skipped = _extract_archive_images(
            source, directory,
            cancel_check=cancel_check,
            progress_callback=progress_callback,
            skipped_callback=skipped_callback,
        )
        _write_manifest(directory, {
            "version": _CACHE_VERSION,
            "kind": "archive",
            "fingerprint": key,
            "source": str(source),
            "pages": [Path(page).name for page in pages],
            "skipped": [
                {
                    "name": name,
                    "kind": detected.kind.value,
                    "mime": detected.mime,
                    "extension": detected.extension,
                    "detail": detected.detail,
                }
                for name, detected in skipped
            ],
        })
        return pages


def expand_inputs(
    paths: list[str],
    work_dir: str | None = None,
    *,
    dpi: int = 200,
    progress_callback: ProgressCallback | None = None,
    page_ready_callback: PageReadyCallback | None = None,
    cancel_check: Callable[[], bool] | None = None,
    skipped_callback: SkippedCallback | None = None,
) -> list[str]:
    """Expand folders, PDFs, image archives and images into ordered image paths.

    Files are classified by content (:func:`adapters.file_type.detect_file_type`),
    so a wrong suffix neither hides a page nor smuggles in a non-image.
    Folders are scanned one level deep in natural order; hidden entries and
    AppleDouble ``._*`` files are skipped silently.  Unsupported inputs (text,
    EPUB, unknown binaries, broken or image-less archives, missing paths) are
    collected: if nothing usable remains, :class:`UnsupportedFileTypeError`
    listing all of them is raised; otherwise each one is reported through
    ``skipped_callback``.  An empty folder still yields ``[]``.

    Every single file is sniffed, including ones with an image suffix: the
    signature wins over the extension, so text named ``.png`` is rejected and a
    PDF named ``.png`` is rendered as a PDF.  Images whose format has no
    signature (``detail="by-extension"``, e.g. ``.ppm``) are still accepted.
    A confirmed image costs one header read and is returned as-is (no copy,
    no render, no cache lookup), which keeps the multi-model OCR path, where
    the same image list is re-expanded per role, cheap.

    ``work_dir`` is accepted for API compatibility only: generated pages always
    live in the session temporary root, never beside the user's files.
    """
    del work_dir
    expanded_paths = [Path(raw).expanduser() for raw in paths]
    images: list[str] = []
    rejected: list[tuple[str, DetectedType]] = []

    def expand_file(path: Path) -> None:
        detected = detect_file_type(path)
        if detected.kind is FileKind.IMAGE:
            images.append(str(path))
        elif detected.kind is FileKind.PDF:
            images.extend(pdf_to_images(
                str(path), dpi=int(dpi), progress_callback=progress_callback,
                page_ready_callback=page_ready_callback,
                cancel_check=cancel_check,
            ))
        elif detected.kind in (FileKind.ARCHIVE_ZIP, FileKind.ARCHIVE_RAR):
            try:
                images.extend(extract_image_archive(
                    path,
                    cancel_check=cancel_check,
                    progress_callback=progress_callback,
                    skipped_callback=skipped_callback,
                ))
            except ArchiveExtractionError as exc:
                rejected.append((str(path), DetectedType(
                    detected.kind, detected.mime, detected.extension, str(exc),
                )))
        else:
            rejected.append((str(path), detected))

    for path in expanded_paths:
        if cancel_check is not None and cancel_check():
            raise RuntimeError("输入加载已取消")
        if path.is_dir():
            for item in sorted(path.iterdir(), key=natural_sort_key):
                if item.name.startswith(".") or item.is_dir():
                    continue
                if cancel_check is not None and cancel_check():
                    raise RuntimeError("输入加载已取消")
                expand_file(item)
        else:
            expand_file(path)

    if not images:
        if rejected:
            raise UnsupportedFileTypeError(rejected)
        return images
    if skipped_callback is not None:
        for path_text, detected in rejected:
            skipped_callback(path_text, detected)
    return images


def release_pdf_caches(paths: list[str] | tuple[str, ...] | set[str]) -> int:
    """Delete session-local PDF rasters belonging to ``paths``.

    Only directories below this process' application-owned session root are
    considered.  Original PDFs, images, persistent model caches, and files from
    another running process can never be removed by this helper.
    """
    sources: set[str] = set()
    for raw in paths or []:
        try:
            path = Path(raw).expanduser().resolve()
        except Exception:
            continue
        if path.suffix.lower() == ".pdf":
            sources.add(str(path))
    if not sources:
        return 0

    registry = session_temp_registry()
    cache_root = registry.path("pdf-pages")
    if not cache_root.exists():
        return 0
    removed = 0
    for directory in list(cache_root.iterdir()):
        if not directory.is_dir():
            continue
        manifest = _read_manifest(directory)
        try:
            manifest_source = str(Path(str(manifest.get("source", ""))).expanduser().resolve())
        except Exception:
            manifest_source = ""
        if manifest_source not in sources:
            continue
        if registry.release(directory, delete=True):
            removed += 1
    return removed
