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
import threading
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Callable

from utils.session_temp import session_temp_registry

_NUM_RE = re.compile(r"(\d+)")
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".heic", ".tif", ".tiff", ".bmp", ".gif", ".ppm"}
_CACHE_VERSION = 2
_RENDER_LOCKS: dict[str, threading.RLock] = {}
_RENDER_LOCKS_GUARD = threading.Lock()

ProgressCallback = Callable[[int, int], None]
PageReadyCallback = Callable[[str, int, int], None]

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


def expand_inputs(
    paths: list[str],
    work_dir: str | None = None,
    *,
    dpi: int = 200,
    progress_callback: ProgressCallback | None = None,
    page_ready_callback: PageReadyCallback | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[str]:
    """Expand folders, PDFs, and images into naturally ordered image paths.

    Multi-model OCR resolves PDF/folder inputs once at run start and then hands
    every role the same immutable image list.  Detect that hot path lexically
    and return it directly: no repeated directory probes, PDF cache checks or
    filesystem stats are needed for each later OCR engine.
    """
    del work_dir
    expanded_paths = [Path(raw).expanduser() for raw in paths]
    if expanded_paths and all(path.suffix.lower() in _IMAGE_EXTS for path in expanded_paths):
        if cancel_check is not None and cancel_check():
            raise RuntimeError("输入加载已取消")
        return [str(path) for path in expanded_paths]
    images: list[str] = []
    for path in expanded_paths:
        if cancel_check is not None and cancel_check():
            raise RuntimeError("输入加载已取消")
        if path.is_dir():
            images.extend(sorted(
                (str(item) for item in path.iterdir() if item.suffix.lower() in _IMAGE_EXTS),
                key=natural_sort_key,
            ))
        elif path.suffix.lower() == ".pdf":
            images.extend(pdf_to_images(
                str(path), dpi=int(dpi), progress_callback=progress_callback,
                page_ready_callback=page_ready_callback,
                cancel_check=cancel_check,
            ))
        elif path.suffix.lower() in _IMAGE_EXTS:
            images.append(str(path))
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
