from __future__ import annotations

"""Project page-granularity OCR evidence onto canonical sentence rows.

Page OCR remains a true full-page recognition pass.  This module runs *after*
recognition and converts that evidence into the same sentence identity used by
column/sentence roles, so the OCR comparison UI never has to compare an entire
page against one sentence.
"""

import copy
from dataclasses import dataclass

from models.document import Block, BlockType, UnifiedDocument
from engine.multi_ocr_compare import compare_ocr_documents

_TEXT_TYPES = {
    BlockType.PARAGRAPH, BlockType.DIALOGUE, BlockType.CHAPTER,
    BlockType.SECTION, BlockType.RUBY, BlockType.TOC_ENTRY,
}


@dataclass(frozen=True, slots=True)
class ProjectionStats:
    rows: int
    empty_rows: int
    column_anchored_rows: int
    alignment_mode: str


def _row_block_type(value: str) -> BlockType:
    try:
        return BlockType(str(value))
    except Exception:
        return BlockType.PARAGRAPH


def project_page_document_to_sentences(
    page_document: UnifiedDocument,
    canonical_document: UnifiedDocument,
    *,
    page_label: str = "整页主模型",
) -> tuple[UnifiedDocument, ProjectionStats]:
    """Return one page-model candidate per canonical sentence.

    The page model is never re-run and no synthetic OCR text is invented.  The
    existing many-to-many monotonic aligner is used once to associate full-page
    blocks with canonical sentence spans.  The returned document then carries
    the canonical ``source_column_ids`` on every row, allowing the normal
    ``column_id_consensus`` comparator to take over for the final multi-model UI.
    """
    comparison = compare_ocr_documents(
        [canonical_document, page_document],
        ["canonical", str(page_label or "page")],
    )
    projected = canonical_document.snapshot_clone()
    projected.blocks = []
    source_pages = list(getattr(page_document, "pages", []) or [])
    projected.pages = copy.deepcopy(source_pages or getattr(canonical_document, "pages", []) or [])
    projected.metadata = copy.deepcopy(page_document.metadata)
    projected.metadata.source_engine = f"{page_document.metadata.source_engine or 'page_ocr'}+sentence_projection"
    projected.metadata.__dict__["page_sentence_projection"] = True
    projected.metadata.__dict__["page_sentence_projection_alignment_mode"] = comparison.alignment_mode

    empty_rows = 0
    anchored_rows = 0
    absorbed_unanchored_rows = 0
    pending_prefix: dict[int, str] = {}
    last_block_by_page: dict[int, Block] = {}
    for row in comparison.rows:
        text = str(row.texts[1] if len(row.texts) > 1 else "" or "").strip()
        page_number = int(row.page or 0)
        column_ids = [str(value) for value in (row.column_ids or ()) if str(value)]
        if not column_ids and row.primary_block_index is not None:
            try:
                canonical_block = canonical_document.blocks[int(row.primary_block_index)]
                metadata_source = canonical_block.metadata if isinstance(canonical_block.metadata, dict) else {}
                raw_ids = metadata_source.get("source_column_ids") or []
                if isinstance(raw_ids, str):
                    raw_ids = [raw_ids]
                column_ids = [str(value) for value in raw_ids if str(value)]
            except (IndexError, TypeError, ValueError):
                column_ids = []

        if not column_ids:
            # A secondary-only page fragment must never become a comparison row
            # without physical lineage. Preserve its text by attaching it to the
            # neighbouring canonical sentence on the same page. This keeps V5
            # adjudication/48px review addressable without inventing OCR calls.
            if text:
                previous = last_block_by_page.get(page_number)
                if previous is not None:
                    previous.text = (str(previous.text or "") + text).strip()
                    previous.ocr_raw = previous.text
                    previous.metadata["page_sentence_projection_absorbed_unanchored"] = True
                else:
                    pending_prefix[page_number] = (pending_prefix.get(page_number, "") + text).strip()
            absorbed_unanchored_rows += 1
            continue

        if pending_prefix.get(page_number):
            text = (pending_prefix.pop(page_number) + text).strip()
        if not text:
            empty_rows += 1
        anchored_rows += 1
        metadata = {
            "page_sentence_projected": True,
            "page_sentence_projection_source": page_document.metadata.source_engine or "page_ocr",
            "source_column_ids": column_ids,
            "source_column_texts": ([""] * max(0, len(column_ids) - 1) + [text]) if column_ids else [],
            "source_column_primary_texts": ([""] * max(0, len(column_ids) - 1) + [text]) if column_ids else [],
            "source_column_consensus_seed_flags": [False] * len(column_ids),
            "column_sentence_reflow": bool(column_ids),
            "column_count": len(column_ids),
            "atomic_ocr_sentence": True,
            "canonical_sentence_group_id": str(row.sentence_group_id or ""),
        }
        block = Block(
            type=_row_block_type(row.block_type),
            text=text,
            page=page_number,
            reading_order=len(projected.blocks),
            order_in_page=len(projected.blocks),
            confidence=(
                float(row.model_confidences[1])
                if len(row.model_confidences) > 1 else float(row.confidence or 0.0)
            ),
            ocr_raw=text,
            metadata=metadata,
            modified_by="page_sentence_projection",
        )
        projected.blocks.append(block)
        last_block_by_page[page_number] = block

    # Extremely unusual: page OCR emitted text before any canonical row and no
    # later anchored row existed on that page. Keep it on the nearest projected
    # sentence rather than creating an unaddressable row.
    for page_number, prefix in list(pending_prefix.items()):
        if not prefix:
            continue
        target = last_block_by_page.get(page_number)
        if target is not None:
            target.text = (prefix + str(target.text or "")).strip()
            target.ocr_raw = target.text
            target.metadata["page_sentence_projection_absorbed_unanchored"] = True

    projected.metadata.column_sentence_reflow_applied = bool(anchored_rows)
    projected.add_log(
        "page_sentence_projection",
        f"整页 OCR 已投影为 {len(projected.blocks)} 个 canonical sentence；空候选 {empty_rows} 个；吸收无列ID片段 {absorbed_unanchored_rows} 个",
        empty_rows,
    )
    return projected, ProjectionStats(
        rows=len(projected.blocks),
        empty_rows=empty_rows,
        column_anchored_rows=anchored_rows,
        alignment_mode=str(comparison.alignment_mode or ""),
    )


def canonicalize_page_document_sentences(
    page_document: UnifiedDocument,
) -> tuple[UnifiedDocument, ProjectionStats]:
    """Convert a page-only OCR document into sentence-level comparison rows.

    This path is used only when the page role is the sole main role, so there is
    no independent column/sentence document that can provide canonical sentence
    geometry.  Recognition is *not* repeated: punctuation boundaries are split
    conservatively from the already-recognized page text, while page order,
    block type, confidence and source metadata are retained.  If another main
    role exists, :func:`project_page_document_to_sentences` is preferred because
    it can anchor every sentence to immutable shared column IDs.
    """
    import uuid
    from engine.multi_ocr_compare import sentence_units

    projected = page_document.snapshot_clone()
    projected.blocks = []
    units = sentence_units(page_document)
    for order, unit in enumerate(units):
        try:
            source = page_document.blocks[int(unit.block_index)]
        except (IndexError, TypeError, ValueError):
            continue
        block = copy.deepcopy(source)
        block.id = uuid.uuid4().hex
        block.text = str(unit.text or "").strip()
        block.ocr_raw = block.text
        block.reading_order = order
        block.order_in_page = order
        metadata = dict(block.metadata or {})
        metadata.update({
            "page_sentence_canonicalized": True,
            "atomic_ocr_sentence": True,
            "canonical_sentence_group_id": f"page:{int(unit.page or 0)}:{order}",
        })
        block.metadata = metadata
        block.modified_by = "page_sentence_canonicalization"
        projected.blocks.append(block)

    # snapshot_clone() already detached metadata exactly.
    projected.metadata.__dict__["page_sentence_canonicalized"] = True
    projected.metadata.__dict__["page_sentence_projection_alignment_mode"] = "page_self_sentence_split"
    projected.add_log(
        "page_sentence_canonicalization",
        f"仅整页主模型：已将一次整页 OCR 结果整理为 {len(projected.blocks)} 个单句比较单元",
        0,
    )
    return projected, ProjectionStats(
        rows=len(projected.blocks),
        empty_rows=0,
        column_anchored_rows=0,
        alignment_mode="page_self_sentence_split",
    )
