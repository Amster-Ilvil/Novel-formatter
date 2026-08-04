# -*- coding: utf-8 -*-
"""Authoritative page-level OCR admission policy.

Page Manager classifications are an input contract, not a post-OCR hint.  Pages
explicitly classified as cover/front matter/illustration/back matter must never
be sent to an OCR recognizer or its expensive crop/column-preparation stages.
They remain in the document as page assets and are synchronized into EPUB later.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

from models.document import BlockType


# Page-level classes that are assets/layout pages rather than OCR body pages.
# ``UNKNOWN`` is intentionally not included: an unresolved page may still need
# OCR so the user can inspect it.  Marking a page as ``paragraph`` is the manual
# force-OCR action in the current Page Manager UI.
NON_OCR_PAGE_TYPES = frozenset({
    BlockType.COVER.value,
    BlockType.COLOR_ILLUS.value,
    BlockType.BLANK.value,
    BlockType.TOC_PAGE.value,
    BlockType.ILLUSTRATION.value,
    BlockType.AFTERWORD.value,
    BlockType.COLOPHON.value,
    BlockType.HALF_ILLUS.value,
    BlockType.TITLE_PAGE.value,
    BlockType.FRONTISPIECE.value,
    BlockType.INSERT.value,
    BlockType.ADVERTISEMENT.value,
    BlockType.INDEX_PAGE.value,
    BlockType.APPENDIX.value,
    BlockType.MAP_PAGE.value,
    BlockType.CHARACTER_SHEET.value,
})


def page_type_value(value) -> str:
    if isinstance(value, BlockType):
        return value.value
    return str(value or "").strip()


def should_skip_page_ocr(value) -> bool:
    """Return True only for an explicit non-OCR page classification."""
    return page_type_value(value) in NON_OCR_PAGE_TYPES


def should_ocr_page(page_no: int, overrides: Mapping[int | str, object] | None) -> bool:
    normalized = {int(key): value for key, value in (overrides or {}).items()}
    return not should_skip_page_ocr(normalized.get(int(page_no)))


def split_ocr_pages(
    image_paths: Iterable[str | Path],
    overrides: Mapping[int | str, object] | None,
) -> tuple[list[tuple[int, str]], list[tuple[int, str, str]]]:
    """Split an ordered page list into OCR pages and preserved asset pages."""
    normalized = {int(key): page_type_value(value) for key, value in (overrides or {}).items()}
    ocr_pages: list[tuple[int, str]] = []
    skipped_pages: list[tuple[int, str, str]] = []
    for page_no, path in enumerate(image_paths, start=1):
        path_text = str(path)
        page_type = normalized.get(page_no, "")
        if should_skip_page_ocr(page_type):
            skipped_pages.append((page_no, path_text, page_type))
        else:
            ocr_pages.append((page_no, path_text))
    return ocr_pages, skipped_pages


def _normalized_paths(values: Iterable[str | Path] | None) -> tuple[str, ...]:
    result: list[str] = []
    for value in values or []:
        try:
            result.append(str(Path(value).expanduser().resolve()))
        except Exception:
            result.append(str(value))
    return tuple(result)


def confirmed_overrides_for_active_inputs(
    *,
    active_inputs: Iterable[str | Path] | None,
    page_images: Iterable[str | Path] | None,
    raw_inputs: Iterable[str | Path] | None,
    page_overrides: Mapping[int | str, object] | None,
    auto_suggested: Iterable[int] | None,
) -> dict[int, str]:
    """Return confirmed classifications when the active OCR batch matches.

    OCR may be started from the expanded page list while Page Manager remembers
    the original PDF/folder in ``raw_inputs``.  The old implementation compared
    only those unlike representations, silently dropped all classifications and
    consequently OCRed covers and illustrations.  Matching either authoritative
    representation fixes the race without allowing an unrelated book's labels to
    leak into a direct OCR run.
    """
    active = _normalized_paths(active_inputs)
    pages = _normalized_paths(page_images)
    raw = _normalized_paths(raw_inputs)
    if not active:
        return {}
    same_expanded_batch = bool(pages) and active == pages
    same_raw_batch = bool(raw) and active == raw
    if not (same_expanded_batch or same_raw_batch):
        return {}

    suggested = {int(page_no) for page_no in (auto_suggested or set())}
    result: dict[int, str] = {}
    for page_no, page_type in (page_overrides or {}).items():
        number = int(page_no)
        if number in suggested:
            continue
        result[number] = page_type_value(page_type)
    return result


__all__ = [
    "NON_OCR_PAGE_TYPES",
    "page_type_value",
    "should_skip_page_ocr",
    "should_ocr_page",
    "split_ocr_pages",
    "confirmed_overrides_for_active_inputs",
]
