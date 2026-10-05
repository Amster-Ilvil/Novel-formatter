from __future__ import annotations

"""Shared structure preparation for role-based multi-OCR.

The expensive recognizers must not each rediscover the same page structure.
This module prepares authoritative physical-column geometry once for the whole
run and can project a true full-page OCR document back onto those shared
columns.  Projection is CPU-only: it never calls an OCR model.
"""

import copy
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from PIL import Image

from adapters.column_ocr_adapter import (
    DetectedColumn,
    _prepare_page_columns_cached,
    _shared_geometry_profile_dir,
)
from engine.column_sentence_reflow import reflow_columns_into_sentences
from models.document import Block, BlockType, BoundingBox, UnifiedDocument


@dataclass(frozen=True, slots=True)
class PreparedPageGeometry:
    page_number: int
    page_path: str
    width: int
    height: int
    columns: tuple[DetectedColumn, ...]


@dataclass(frozen=True, slots=True)
class StructurePrewarmStats:
    pages_requested: int
    pages_prepared: int
    pages_failed: int
    columns: int
    seconds: float
    geometry_fingerprint: str


@dataclass(frozen=True, slots=True)
class PageGeometryProjectionStats:
    pages: int
    columns: int
    mapped_blocks: int
    unboxed_blocks: int
    empty_columns: int
    sentences: int
    ruby_filtered_blocks: int = 0
    outside_geometry_blocks: int = 0


@dataclass(slots=True)
class MultiOcrStructureCache:
    pages: dict[int, PreparedPageGeometry]
    page_by_path: dict[str, PreparedPageGeometry]
    stats: StructurePrewarmStats

    def column_ids(self) -> tuple[str, ...]:
        values: list[str] = []
        for page_number in sorted(self.pages):
            page = self.pages[page_number]
            values.extend(
                f"p{page_number:05d}:c{index + 1:03d}"
                for index in range(len(page.columns))
            )
        return tuple(values)


def _is_image(path: str) -> bool:
    return Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".jp2"}


def prewarm_multi_ocr_structure(
    page_paths: Iterable[str],
    *,
    shared_prepare_base: str | Path,
    sensitivity: int,
    padding_percent: int,
    max_columns: int,
    fixed_region_rect: Sequence[float] | None = None,
    detector_mode: str = "components",
    capture_ruby_candidates: bool = False,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    max_workers: int | None = None,
    excluded_page_numbers: Iterable[int] | None = None,
) -> MultiOcrStructureCache:
    """Detect physical columns once for every image page in a multi-OCR run.

    No OCR recognizer is loaded or called here.  The sidecars are written into
    the exact shared geometry namespace consumed later by column/sentence roles,
    so their first model run becomes a cache hit instead of another detection.
    """
    paths = [str(Path(path).expanduser().resolve()) for path in page_paths if _is_image(str(path))]
    excluded = {int(value) for value in (excluded_page_numbers or ()) if int(value) > 0}
    base = Path(shared_prepare_base)
    geometry_dir, fingerprint = _shared_geometry_profile_dir(
        base,
        detector_mode=detector_mode,
        sensitivity=int(sensitivity),
        padding_percent=int(padding_percent),
        max_columns=int(max_columns),
        fixed_region_rect=fixed_region_rect,
        capture_ruby_candidates=bool(capture_ruby_candidates),
    )
    started = time.perf_counter()
    prepared: dict[int, PreparedPageGeometry] = {}
    failures = 0

    def prepare(item: tuple[int, str]):
        page_number, path = item
        if callable(cancel_check) and cancel_check():
            raise InterruptedError("OCR 已停止")
        columns, error = _prepare_page_columns_cached(
            page_number,
            path,
            geometry_dir,
            sensitivity=int(sensitivity),
            padding_percent=int(padding_percent),
            max_columns=int(max_columns),
            fixed_region_rect=fixed_region_rect,
            detector_mode=detector_mode,
            capture_ruby_candidates=bool(capture_ruby_candidates),
        )
        width = height = 0
        if columns:
            try:
                with Image.open(path) as image:
                    width, height = image.size
            except Exception:
                width = height = 0
        return page_number, path, columns, error, int(width), int(height)

    items = [
        (page_number, path)
        for page_number, path in enumerate(paths, start=1)
        if page_number not in excluded
    ]
    total = len(items)
    workers = max_workers
    if workers is None:
        workers = min(6, max(1, total))
    workers = max(1, min(int(workers), max(1, total)))
    if workers == 1:
        results = [prepare(item) for item in items]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(prepare, item): item for item in items}
            for future in as_completed(futures):
                results.append(future.result())

    completed = 0
    for page_number, path, columns, error, width, height in sorted(results):
        if callable(cancel_check) and cancel_check():
            raise InterruptedError("OCR 已停止")
        completed += 1
        if columns and not error:
            page = PreparedPageGeometry(
                page_number=page_number,
                page_path=path,
                width=width,
                height=height,
                columns=tuple(columns),
            )
            prepared[page_number] = page
        else:
            failures += 1
        if callable(progress_callback):
            progress_callback(completed, total, Path(path).name)

    stats = StructurePrewarmStats(
        pages_requested=total,
        pages_prepared=len(prepared),
        pages_failed=failures,
        columns=sum(len(page.columns) for page in prepared.values()),
        seconds=max(0.0, time.perf_counter() - started),
        geometry_fingerprint=fingerprint,
    )
    return MultiOcrStructureCache(
        pages=prepared,
        page_by_path={page.page_path: page for page in prepared.values()},
        stats=stats,
    )


def _block_page(block: Block) -> int:
    for value in (block.page_index, block.page_number, block.page):
        if value is None:
            continue
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number > 0:
            return number
    return 0


def _block_rect_pixels(block: Block, width: int, height: int) -> tuple[float, float, float, float] | None:
    box = block.bbox
    if box is None or width <= 0 or height <= 0:
        return None
    try:
        x1 = float(box.x) * width
        y1 = float(box.y) * height
        x2 = float(box.x + box.w) * width
        y2 = float(box.y + box.h) * height
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _best_column_index(rect: tuple[float, float, float, float], columns: Sequence[DetectedColumn]) -> int | None:
    x1, _y1, x2, _y2 = rect
    center = (x1 + x2) / 2.0
    best_index = None
    best_score = -1.0
    for index, column in enumerate(columns):
        left = float(column.hard_left)
        right = float(column.hard_right)
        overlap = max(0.0, min(x2, right) - max(x1, left))
        width = max(1.0, x2 - x1)
        center_bonus = 1.0 if left <= center <= right else 0.0
        score = overlap / width + center_bonus
        if score > best_score:
            best_score = score
            best_index = index
    return best_index if best_score > 0 else None








def page_document_to_shared_sentences(
    page_document: UnifiedDocument,
    structure: MultiOcrStructureCache,
    *,
    max_columns: int = 64,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[UnifiedDocument, PageGeometryProjectionStats]:
    """Map one full-page OCR result onto prewarmed columns, then merge to sentences.

    This function performs no recognition.  It is the missing bridge that lets
    a page-role model (NDLOCR-Lite by default) run exactly once per page while
    still becoming authoritative sentence/column structure for later roles.
    """
    result = copy.deepcopy(page_document)
    result.blocks = []
    result.metadata = copy.deepcopy(page_document.metadata)
    result.metadata.source_engine = f"{page_document.metadata.source_engine or 'page_ocr'}+shared_geometry"
    result.metadata.__dict__["multi_ocr_shared_geometry_projection"] = True
    result.metadata.__dict__["multi_ocr_geometry_fingerprint"] = structure.stats.geometry_fingerprint

    blocks_by_page: dict[int, list[Block]] = {}
    for block in page_document.blocks:
        page = _block_page(block)
        if page <= 0:
            page = 1
        blocks_by_page.setdefault(page, []).append(block)

    mapped_blocks = 0
    unboxed_blocks = 0
    empty_columns = 0
    ruby_filtered_blocks = 0
    outside_geometry_blocks = 0
    output_order = 0
    for page_number in sorted(structure.pages):
        if callable(cancel_check) and cancel_check():
            raise InterruptedError("OCR 已停止")
        prepared = structure.pages[page_number]
        assignments: dict[int, list[tuple[float, float, Block]]] = {
            index: [] for index in range(len(prepared.columns))
        }
        unboxed: list[Block] = []
        for block in blocks_by_page.get(page_number, []):
            text = str(block.text or "").strip()
            if not text:
                continue
            rect = _block_rect_pixels(block, prepared.width, prepared.height)
            if rect is None:
                unboxed.append(block)
                unboxed_blocks += 1
                continue
            column_index = _best_column_index(rect, prepared.columns)
            if column_index is None:
                unboxed.append(block)
                unboxed_blocks += 1
                continue
            assignments[column_index].append((rect[1], rect[0], block))
            mapped_blocks += 1

        # Coordinate-free page OCR is uncommon, but keep a deterministic fallback
        # that does not invoke another model: distribute its ordered blocks across
        # currently empty physical columns in page reading order.
        if unboxed:
            empty_indices = [index for index, values in assignments.items() if not values]
            for block, column_index in zip(unboxed, empty_indices):
                assignments[column_index].append((0.0, 0.0, block))
                mapped_blocks += 1

        for column_index, column in enumerate(prepared.columns):
            values = assignments[column_index]
            if not values:
                empty_columns += 1
                continue
            values.sort(key=lambda item: (item[0], item[1], item[2].reading_order))
            text = "".join(str(item[2].text or "").strip() for item in values).strip()
            if not text:
                empty_columns += 1
                continue
            confidence_weights = [max(1, len(str(item[2].text or ""))) for item in values]
            confidence = sum(
                float(item[2].confidence or 0.0) * weight
                for item, weight in zip(values, confidence_weights)
            ) / max(1, sum(confidence_weights))
            column_id = f"p{page_number:05d}:c{column_index + 1:03d}"
            bbox = None
            if prepared.width > 0 and prepared.height > 0:
                bbox = BoundingBox.from_pixels(
                    column.hard_left,
                    column.top,
                    column.hard_right,
                    column.bottom,
                    prepared.width,
                    prepared.height,
                )
            source_type = values[0][2].type if values else BlockType.PARAGRAPH
            result.blocks.append(Block(
                type=source_type if source_type in {BlockType.PARAGRAPH, BlockType.DIALOGUE, BlockType.CHAPTER, BlockType.SECTION, BlockType.RUBY, BlockType.TOC_ENTRY} else BlockType.PARAGRAPH,
                text=text,
                page=page_number,
                page_index=page_number,
                page_number=page_number,
                reading_order=output_order,
                order_in_page=column_index,
                bbox=bbox,
                confidence=max(0.0, min(1.0, confidence)),
                ocr_raw=text,
                source_format="ocr",
                text_direction="vertical-rl",
                metadata={
                    "column_id": column_id,
                    "column_index": column_index,
                    "column_count": len(prepared.columns),
                    "page_geometry_projected": True,
                    "page_geometry_source_engine": page_document.metadata.source_engine or "page_ocr",
                    "multi_ocr_geometry_fingerprint": structure.stats.geometry_fingerprint,
                },
                modified_by="page_geometry_projection",
            ))
            output_order += 1

    result.metadata.__dict__["column_ocr"] = True
    result.metadata.__dict__["column_count"] = sum(len(page.columns) for page in structure.pages.values())
    result = reflow_columns_into_sentences(
        result,
        max_columns=max_columns,
        cancel_check=cancel_check,
    )
    stats = PageGeometryProjectionStats(
        pages=len(structure.pages),
        columns=sum(len(page.columns) for page in structure.pages.values()),
        mapped_blocks=mapped_blocks,
        unboxed_blocks=unboxed_blocks,
        empty_columns=empty_columns,
        sentences=len(result.blocks),
        ruby_filtered_blocks=ruby_filtered_blocks,
        outside_geometry_blocks=outside_geometry_blocks,
    )
    result.metadata.__dict__["page_geometry_projection_stats"] = {
        "pages": stats.pages,
        "columns": stats.columns,
        "mapped_blocks": stats.mapped_blocks,
        "unboxed_blocks": stats.unboxed_blocks,
        "empty_columns": stats.empty_columns,
        "sentences": stats.sentences,
        "ruby_filtered_blocks": stats.ruby_filtered_blocks,
        "outside_geometry_blocks": stats.outside_geometry_blocks,
    }
    return result, stats
