# -*- coding: utf-8 -*-
"""Synchronize Page Manager image/page classifications into a UnifiedDocument.

Text workspaces and Page Manager remain independent.  At comparison/export time
this module overlays the latest page assets while preserving any explicit image
anchors already present in the document.  This is crucial for flattened OCR/MD
sources whose text blocks have page=0: page numbers alone cannot reconstruct an
illustration's exact position, so a manually positioned IMAGE_REF must never be
removed and re-appended to the end of the book.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from models.document import Block, BlockType, PageInfo, TocEntry, UnifiedDocument


@dataclass(frozen=True)
class PageAssetSyncReport:
    managed_pages: int = 0
    image_pages: int = 0
    removed_stale_blocks: int = 0
    inserted_image_blocks: int = 0
    unplaced_image_blocks: int = 0
    changed: bool = False


def _is_image_page(page_type: BlockType) -> bool:
    return page_type not in (BlockType.PARAGRAPH, BlockType.BLANK)


def _coerce_page_type(value, fallback: BlockType = BlockType.PARAGRAPH) -> BlockType:
    if isinstance(value, BlockType):
        return value
    try:
        return BlockType(str(value))
    except (TypeError, ValueError):
        return fallback


def _same_path(left: str, right: str) -> bool:
    if not left or not right:
        return False
    try:
        return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()
    except Exception:
        return str(left) == str(right)


_FRONT_MATTER_IMAGE_TYPES = {
    BlockType.COVER,
    BlockType.TOC_PAGE,
    BlockType.TITLE_PAGE,
    BlockType.FRONTISPIECE,
}


def _page_for_existing_ref(block: Block, images: list[str]) -> int:
    page_no = int(getattr(block, "page", 0) or 0)
    if page_no > 0:
        return page_no
    for index, path in enumerate(images, start=1):
        if _same_path(block.image_path, path):
            return index
    return 0


def _overlay_is_satisfied(doc: UnifiedDocument, images: list[str], managed_types: dict[int, BlockType]) -> bool:
    found: dict[int, int] = {}
    for block in doc.blocks:
        if block.type != BlockType.IMAGE_REF:
            continue
        page_no = _page_for_existing_ref(block, images)
        if page_no in managed_types:
            found[page_no] = found.get(page_no, 0) + 1
    for page_no, page_type in managed_types.items():
        expected = 1 if _is_image_page(page_type) else 0
        if found.get(page_no, 0) != expected:
            return False
    return True


def sync_page_manager_assets(
    doc: UnifiedDocument,
    page_images: Iterable[str | Path],
    confirmed_overrides: Mapping[int | str, str | BlockType] | None = None,
    *,
    preserve_unmanaged_pages: bool = True,
    copy_document: bool = True,
) -> tuple[UnifiedDocument, PageAssetSyncReport]:
    """Return a document synchronized with the current Page Manager overlay.

    Existing IMAGE_REF blocks are matched by page/path, updated in place and kept
    at their current block position.  Only missing image pages are inserted.
    When the text has no usable page numbers, a missing body illustration is
    marked ``placement_required`` and appended as a visible compare marker; the
    program does not pretend that its exact position can be inferred.
    """
    images = [str(Path(p)) for p in page_images]
    if not images:
        return (copy.deepcopy(doc) if copy_document else doc), PageAssetSyncReport()

    out = copy.deepcopy(doc) if copy_document else doc
    overrides = {int(k): v for k, v in (confirmed_overrides or {}).items()}
    signature = repr((
        tuple(images),
        tuple(sorted((int(k), str(getattr(v, "value", v))) for k, v in overrides.items())),
        bool(preserve_unmanaged_pages),
    ))
    existing_pages = {int(p.page_no): p for p in out.pages}

    managed_types: dict[int, BlockType] = {}
    new_pages: list[PageInfo] = []
    changed = False
    for page_no, image_path in enumerate(images, start=1):
        old = existing_pages.get(page_no)
        old_type = old.page_type if old else BlockType.PARAGRAPH
        page_type = _coerce_page_type(overrides.get(page_no), old_type)
        managed_types[page_no] = page_type
        new_pages.append(PageInfo(
            page_no=page_no,
            page_type=page_type,
            image_path=image_path,
            width=old.width if old else 0,
            height=old.height if old else 0,
            confidence=1.0 if page_no in overrides else (old.confidence if old else 1.0),
        ))
        if old is None or old.page_type != page_type or not _same_path(old.image_path, image_path):
            changed = True

    if preserve_unmanaged_pages:
        new_pages.extend(copy.deepcopy(p) for p in out.pages if int(p.page_no) > len(images))
    out.pages = new_pages

    if (
        getattr(out.metadata, "page_asset_sync_signature", None) == signature
        and _overlay_is_satisfied(out, images, managed_types)
    ):
        image_pages = sum(1 for p in managed_types.values() if _is_image_page(p))
        unplaced = sum(
            1 for b in out.blocks
            if b.type == BlockType.IMAGE_REF and (b.metadata or {}).get("placement_required")
        )
        return out, PageAssetSyncReport(
            managed_pages=len(images), image_pages=image_pages,
            unplaced_image_blocks=unplaced, changed=False,
        )

    strict_authoritative = str(getattr(out.metadata, "replacement_mode", "") or "") in {
        "strict_full", "strict_literal", "authoritative_right_layout"
    }
    kept: list[Block] = []
    removed = 0
    preserved_pages: set[int] = set()

    for block in out.blocks:
        page_no = int(getattr(block, "page", 0) or 0)
        if block.type == BlockType.IMAGE_REF:
            ref_page = _page_for_existing_ref(block, images)
            if ref_page not in managed_types:
                if preserve_unmanaged_pages:
                    kept.append(block)
                else:
                    removed += 1
                    changed = True
                continue
            page_type = managed_types[ref_page]
            if not _is_image_page(page_type) or ref_page in preserved_pages:
                removed += 1
                changed = True
                continue
            block.page = ref_page
            block.image_path = images[ref_page - 1]
            block.metadata = dict(block.metadata or {})
            block.metadata.update({"page_type": page_type.value, "source": "page_manager"})
            # A user-moved marker is authoritative; it is no longer unplaced.
            if block.image_anchor and block.image_anchor not in {"start", "unplaced"}:
                block.metadata.pop("placement_required", None)
            preserved_pages.add(ref_page)
            kept.append(block)
            continue

        if page_no not in managed_types:
            if not preserve_unmanaged_pages and page_no > len(images) and page_no > 0:
                removed += 1
                changed = True
                continue
            kept.append(block)
            continue
        page_type = managed_types[page_no]
        if page_type != BlockType.PARAGRAPH and not strict_authoritative:
            removed += 1
            changed = True
            continue
        kept.append(block)

    out.blocks = kept
    meaningful_text_pages = any(
        block.type != BlockType.IMAGE_REF and int(getattr(block, "page", 0) or 0) > 0
        for block in out.blocks
    )
    inserted = 0
    unplaced = 0

    for page_no in range(1, len(images) + 1):
        page_type = managed_types[page_no]
        if not _is_image_page(page_type) or page_no in preserved_pages:
            continue
        image_path = images[page_no - 1]
        placement_required = False

        if meaningful_text_pages:
            insert_at = len(out.blocks)
            for idx, block in enumerate(out.blocks):
                block_page = int(getattr(block, "page", 0) or 0)
                if block_page > page_no:
                    insert_at = idx
                    break
        elif page_type in _FRONT_MATTER_IMAGE_TYPES:
            # Keep front matter before正文 and in page order.
            insert_at = 0
            while insert_at < len(out.blocks):
                b = out.blocks[insert_at]
                if b.type != BlockType.IMAGE_REF:
                    break
                existing_page = int(getattr(b, "page", 0) or 0)
                if existing_page > page_no:
                    break
                insert_at += 1
        else:
            # Flattened text has no page coordinates.  Exact placement is
            # unknowable; append a marker that the Compare workspace can move.
            insert_at = len(out.blocks)
            placement_required = True
            unplaced += 1

        previous = out.blocks[insert_at - 1] if insert_at > 0 else None
        anchor = previous.id if previous is not None else "start"
        metadata = {"page_type": page_type.value, "source": "page_manager"}
        if placement_required:
            metadata["placement_required"] = True
            anchor = "unplaced"
        image_block = Block(
            type=BlockType.IMAGE_REF,
            page=page_no,
            reading_order=insert_at,
            image_path=image_path,
            image_anchor=anchor,
            confidence=1.0,
            metadata=metadata,
        )
        out.blocks.insert(insert_at, image_block)
        inserted += 1
        changed = True

    for index, block in enumerate(out.blocks):
        block.reading_order = index
        if block.type == BlockType.IMAGE_REF and index > 0 and not (block.metadata or {}).get("placement_required"):
            block.image_anchor = out.blocks[index - 1].id

    rebuilt_toc: list[TocEntry] = []
    chapter_counter = 0
    for block_index, block in enumerate(out.blocks):
        if block.type != BlockType.CHAPTER or not str(block.text or "").strip():
            continue
        chapter_counter += 1
        block.chapter_index = chapter_counter
        rebuilt_toc.append(TocEntry(block.text, chapter_counter, block_index))
    if rebuilt_toc or out.toc:
        out.toc = rebuilt_toc

    image_pages = sum(1 for p in managed_types.values() if _is_image_page(p))
    if changed or removed or inserted:
        out.add_log(
            "page_manager_assets",
            f"synchronized {len(images)} pages; preserved {len(preserved_pages)} image refs; "
            f"inserted {inserted}; unplaced {unplaced}; removed {removed}",
            inserted,
        )
    out.metadata.page_asset_sync_signature = signature
    return out, PageAssetSyncReport(
        managed_pages=len(images),
        image_pages=image_pages,
        removed_stale_blocks=removed,
        inserted_image_blocks=inserted,
        unplaced_image_blocks=unplaced,
        changed=bool(changed or removed or inserted),
    )
