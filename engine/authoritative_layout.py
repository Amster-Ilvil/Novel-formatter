# -*- coding: utf-8 -*-
"""Project a trusted right-hand manuscript onto OCR/page assets.

Unlike compare/fusion patching, this module treats every non-empty right-hand
text row as the complete final body.  No OCR prose is allowed to survive.  The
OCR document contributes only pages, chapter/image anchors and other assets.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, asdict
from typing import Sequence

from models.document import Block, BlockType, TocEntry, UnifiedDocument
from models.paragraph import Paragraph
from engine.replacement_engine import strict_replace_text
from engine.text_compare import (
    CompareLine, is_alignment_placeholder, looks_like_chapter_title, parse_image_marker,
)

_TEXT_TYPES = {
    BlockType.PARAGRAPH, BlockType.DIALOGUE, BlockType.CHAPTER,
    BlockType.SECTION, BlockType.RUBY, BlockType.FOOTNOTE, BlockType.TOC_ENTRY,
}


@dataclass(slots=True)
class AuthoritativeLayoutReport:
    source_lines: int = 0
    source_chars: int = 0
    output_chars: int = 0
    exact_text_match: bool = False
    chapter_count: int = 0
    image_markers: int = 0
    images_placed: int = 0
    images_pending: int = 0
    explicit_marker_positions: int = 0
    removed_ocr_text_blocks: int = 0
    warnings: tuple[str, ...] = ()

    @property
    def summary(self) -> str:
        status = "正文100%来自右侧" if self.exact_text_match else "正文完整性异常"
        return (
            f"{status}；正文 {self.source_lines} 段/{self.source_chars} 字；"
            f"目录 {self.chapter_count} 项；图片 {self.images_placed} 张"
            f"（待确认 {self.images_pending}）"
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _canonical_text(values: Sequence[str]) -> str:
    # Block boundaries may be rebuilt from OCR layout, so integrity is measured
    # on the exact character stream rather than newline placement between blocks.
    return "".join(str(value or "").replace("\r\n", "\n").replace("\r", "\n") for value in values)


def _is_title(record: CompareLine, text: str) -> bool:
    if str(record.block_type or "") in {
        BlockType.CHAPTER.value, BlockType.SECTION.value, BlockType.TOC_ENTRY.value,
    }:
        return True
    return looks_like_chapter_title(text)


def records_to_authoritative_paragraphs(
    records: Sequence[CompareLine], *, source_name: str = "trusted_right"
) -> tuple[list[Paragraph], list[tuple[str, int]]]:
    """Return trusted paragraphs plus explicit image marker positions.

    Marker positions are expressed as ``(image_block_id, source_paragraph_count)``.
    A marker at count N belongs immediately before source paragraph N, or at the
    end when N equals the paragraph count.
    """
    paragraphs: list[Paragraph] = []
    markers: list[tuple[str, int]] = []
    chapter = ""
    for record in records:
        raw = str(record.text or "")
        if is_alignment_placeholder(raw):
            continue
        marker = parse_image_marker(raw)
        if marker:
            markers.append((marker, len(paragraphs)))
            continue
        text = raw.strip("\r\n")
        if not text.strip():
            continue
        is_title = _is_title(record, text)
        if is_title:
            chapter = text.strip()
        paragraphs.append(Paragraph(
            text=text,
            index=len(paragraphs),
            chapter=chapter,
            source=source_name,
            is_title=is_title,
        ))
    return paragraphs, markers


def _source_text_from_blocks(doc: UnifiedDocument) -> str:
    values = [
        block.text for block in doc.blocks
        if block.type in _TEXT_TYPES and str(block.text or "") != ""
    ]
    return _canonical_text(values)


def _image_identity(block: Block) -> tuple[str, int, str]:
    return (str(block.image_path or ""), int(block.page or 0), str((block.metadata or {}).get("page_type", "")))


def _insert_explicit_markers(
    out: UnifiedDocument,
    structure_doc: UnifiedDocument,
    markers: Sequence[tuple[str, int]],
) -> tuple[int, int]:
    """Honor image rows visible in the compare editor.

    Strict replacement already maps all page assets by OCR page order.  When the
    user has explicit marker rows, matching original IMAGE_REF blocks are moved
    to the requested source-paragraph boundary.  Other page-manager images keep
    their automatically mapped location.
    """
    if not markers:
        return 0, 0

    originals = {
        block.id: copy.deepcopy(block)
        for block in structure_doc.blocks
        if block.type == BlockType.IMAGE_REF and block.image_path
    }
    if not originals:
        return 0, len(markers)

    requested_ids = [marker_id for marker_id, _position in markers if marker_id in originals]
    requested_set = set(requested_ids)
    missing = sum(1 for marker_id, _position in markers if marker_id not in originals)

    # Keep non-requested images from strict page mapping; remove requested images
    # by identity because strict_replace_text creates fresh image block ids.
    requested_identities = {_image_identity(originals[marker_id]) for marker_id in requested_set}
    base_blocks = [
        block for block in out.blocks
        if not (block.type == BlockType.IMAGE_REF and _image_identity(block) in requested_identities)
    ]

    # Source paragraph index is attached by strict replacement/reflow.  For a
    # marker before paragraph N, insert before the first output text segment whose
    # source index is >= N.  This survives one-to-many paragraph reflow.
    def insertion_index(source_position: int) -> int:
        candidate = len(base_blocks)
        for idx, block in enumerate(base_blocks):
            if block.type == BlockType.IMAGE_REF:
                continue
            src = int((block.metadata or {}).get("source_paragraph_index", 10**9))
            if src >= source_position:
                candidate = idx
                break
        return candidate

    placed = 0
    for marker_id, source_position in markers:
        original = originals.get(marker_id)
        if original is None:
            continue
        image = copy.deepcopy(original)
        image.metadata = dict(image.metadata or {})
        image.metadata["authoritative_right_marker"] = True
        insert_at = insertion_index(int(source_position))
        insert_at = max(0, min(insert_at, len(base_blocks)))
        base_blocks.insert(insert_at, image)
        placed += 1

    out.blocks = base_blocks
    previous_text: Block | None = None
    chapter_index = 0
    for index, block in enumerate(out.blocks):
        block.reading_order = index
        if block.type == BlockType.CHAPTER:
            chapter_index += 1
            block.chapter_index = chapter_index
        elif block.type != BlockType.IMAGE_REF:
            block.chapter_index = chapter_index
        if block.type == BlockType.IMAGE_REF:
            block.image_anchor = previous_text.id if previous_text is not None else "start"
            block.chapter_index = chapter_index
        else:
            previous_text = block

    out.toc = []
    chapter_no = 0
    for index, block in enumerate(out.blocks):
        if block.type == BlockType.CHAPTER and str(block.text or "").strip():
            chapter_no += 1
            block.chapter_index = chapter_no
            out.toc.append(TocEntry(block.text.strip(), chapter_no, index))
    return placed, missing


def apply_authoritative_right_to_layout(
    structure_doc: UnifiedDocument,
    records: Sequence[CompareLine],
    *,
    source_name: str = "trusted_right",
    reflow: bool = True,
) -> tuple[UnifiedDocument, AuthoritativeLayoutReport]:
    """Rebuild the book body exclusively from the right-hand editor.

    This is intentionally different from ``apply_compare_records``.  It never
    reuses unmatched OCR text blocks and therefore cannot resurrect old OCR
    fragments after a full-text paste or manual rewrite.
    """
    paragraphs, markers = records_to_authoritative_paragraphs(records, source_name=source_name)
    if not paragraphs:
        raise ValueError("右侧没有可用正文，无法套入 OCR 版面。")

    out, replacement = strict_replace_text(structure_doc, paragraphs, reflow=reflow)
    explicit_placed, marker_missing = _insert_explicit_markers(out, structure_doc, markers)

    source_text = _canonical_text([paragraph.text for paragraph in paragraphs])
    output_text = _source_text_from_blocks(out)
    exact = source_text == output_text

    source_text_blocks = sum(
        1 for block in structure_doc.blocks
        if block.type in _TEXT_TYPES and str(block.text or "").strip()
    )
    image_blocks = [block for block in out.blocks if block.type == BlockType.IMAGE_REF]
    pending = sum(
        1 for block in image_blocks
        if (block.metadata or {}).get("placement_required") or block.image_anchor == "unplaced"
    ) + marker_missing

    warnings: list[str] = []
    if not exact:
        warnings.append("right_text_integrity_mismatch")
    if marker_missing:
        warnings.append("foreign_or_missing_image_marker")
    if getattr(replacement, "image_anchors_pending", 0):
        warnings.append("approximate_image_mapping")

    out.metadata.replacement_mode = "authoritative_right_layout"
    out.metadata.replacement_source_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    out.metadata.replacement_output_hash = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
    out.metadata.replacement_exact_match = exact
    out.metadata.replacement_source_chars = len(source_text)
    out.metadata.replacement_output_chars = len(output_text)
    out.metadata.replacement_missing_chars = 0 if exact else abs(len(source_text) - len(output_text))
    out.metadata.replacement_extra_chars = 0 if exact else abs(len(output_text) - len(source_text))
    out.metadata.replacement_pending_images = pending
    out.metadata.__dict__["authoritative_right_explicit_markers"] = explicit_placed
    out.metadata.__dict__["authoritative_right_ocr_text_reused"] = 0
    out.add_log(
        "authoritative_right_layout",
        f"右侧可信正文套入 OCR 版面：{len(paragraphs)} 段；OCR 正文残留 0；图片 {len(image_blocks)}",
        len(paragraphs),
    )

    report = AuthoritativeLayoutReport(
        source_lines=len(paragraphs),
        source_chars=len(source_text),
        output_chars=len(output_text),
        exact_text_match=exact,
        chapter_count=len(out.toc or []),
        image_markers=len(markers),
        images_placed=len(image_blocks),
        images_pending=pending,
        explicit_marker_positions=explicit_placed,
        removed_ocr_text_blocks=source_text_blocks,
        warnings=tuple(warnings),
    )
    return out, report
