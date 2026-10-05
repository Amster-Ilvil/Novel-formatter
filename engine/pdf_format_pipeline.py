"""Dedicated publication-format pipeline for selectable-text PDFs.

Phase42 keeps this path intentionally small.  The PDF adapter extracts faithful
text + geometry (and performs only filters that require raw font/coordinate
evidence).  This module then applies only the PDF-specific layout operations
that materially change the document.  Chapter/TOC and semantic judgement are
left to the later AI stage.

Unlike the generic Formatter, the dedicated PDF path mutates one private copy in
place.  This avoids repeatedly deep-copying a 50k-100k block document for steps
that are no-ops in PDF mode and makes multi-thousand-page PDFs practical.
"""
from __future__ import annotations

import copy
import gc
from typing import Callable, Optional

from models.document import UnifiedDocument

# User-visible PDF stages.  These are the only stages that materially affect a
# selectable-PDF document.  Do not re-add generic chapter detection, embedded
# title regex passes, OCR sentence guessing, or PDF no-op wrapper steps here.
PDF_FORMAT_STEPS: tuple[str, ...] = (
    "pdf_text_prepare",
    "strip_chapter_notes",
    "remove_duplicates",
    "dialogue_restore",
    "restore_indents",
    "strip_boilerplate",
    "normalize_punct",
    "pdf_text_finalize",
)


def _clone_pdf_format_document(doc: UnifiedDocument) -> UnifiedDocument:
    """Create an isolated working copy without recursively cloning immutable evidence.

    PDF formatting mutates block text/type and replaces each block metadata dict
    before writing PDF-specific fields; source strings, bbox objects and most
    nested extraction evidence are read-only.  A structural clone therefore
    preserves caller isolation while avoiding the memory spike of a full
    ``deepcopy`` on 100k+ raw blocks.
    """
    out = copy.copy(doc)
    out.metadata = copy.deepcopy(doc.metadata)
    out.pages = list(doc.pages)
    out.toc = [copy.copy(item) for item in doc.toc]
    out.processing_log = [dict(item) for item in doc.processing_log]
    out.blocks = []
    mutable_metadata_keys = {
        "pdf_text_review_flags", "pdf_text_review_evidence",
        "source_block_ids", "pdf_source_texts",
        "pdf_guard_intentional_removed_chars",
    }
    for block in doc.blocks:
        cloned = copy.copy(block)
        metadata = dict(block.metadata or {})
        for key in mutable_metadata_keys:
            if key in metadata:
                metadata[key] = copy.deepcopy(metadata[key])
        cloned.metadata = metadata
        cloned.bbox = copy.copy(block.bbox) if block.bbox is not None else None
        out.blocks.append(cloned)
    return out


def is_selectable_pdf_document(doc: UnifiedDocument | None) -> bool:
    if doc is None:
        return False
    meta = getattr(doc, "metadata", None)
    if meta is None:
        return False
    return bool(
        getattr(meta, "pdf_text_layer_mode", False)
        or str(getattr(meta, "source_engine", "") or "") == "pdf_text_layer"
    )


def run_pdf_format_pipeline(
    doc: UnifiedDocument,
    *,
    keep_author_notes: bool = False,
    remove_generated_matter: bool = True,
    restore_indents: bool = True,
    verbose: bool = True,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    repo_path: str | None = None,
) -> UnifiedDocument:
    """Run the minimal, source-safe selectable-PDF formatting profile.

    Extraction owns faithful glyph/geometry recovery, Ruby filtering and verified
    footer folios.  This stage owns physical-column reconstruction, explicit
    author-note/site-matter cleanup, exact-coordinate dedupe, dialogue layout,
    paragraph indentation and the final character guard.  Chapter/TOC detection
    is deliberately deferred to AI.

    One private structural clone is created for undo safety, then all PDF-specific stages
    run in-place.  This prevents the old O(steps x document-size) deepcopy cost
    from exploding on 4k+ page PDFs.
    """
    if not is_selectable_pdf_document(doc):
        raise ValueError("PDF格式处理仅接受 PDF 文字层（pdf_text_layer）文档")

    from engine.pdf_text_layer_formatter import (
        finalize_pdf_text_layer,
        normalize_pdf_text_punctuation,
        prepare_pdf_text_layer,
        preserve_pdf_afterwords,
        preserve_pdf_boilerplate,
        remove_pdf_coordinate_duplicates,
        restore_pdf_dialogue_columns,
        restore_pdf_indents,
    )

    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
        base = _clone_pdf_format_document(doc)
        base.metadata.pdf_text_layer_mode = True
        base.metadata.preserve_ocr_layout = False
        base.metadata.pdf_keep_afterwords = bool(keep_author_notes)
        base.metadata.pdf_remove_generated_matter = bool(remove_generated_matter)
        base.metadata.pdf_restore_indents = bool(restore_indents)

        stages: list[tuple[str, Callable[[UnifiedDocument], UnifiedDocument]]] = [
            ("pdf_text_prepare", lambda current: prepare_pdf_text_layer(current, inplace=True)),
            ("strip_chapter_notes", lambda current: preserve_pdf_afterwords(current, inplace=True)),
            ("remove_duplicates", lambda current: remove_pdf_coordinate_duplicates(current, inplace=True)),
            ("dialogue_restore", lambda current: restore_pdf_dialogue_columns(current, inplace=True)),
        ]
        if restore_indents:
            stages.append(("restore_indents", lambda current: restore_pdf_indents(current, inplace=True)))
        stages.extend([
            ("strip_boilerplate", lambda current: preserve_pdf_boilerplate(current, inplace=True)),
            ("normalize_punct", lambda current: normalize_pdf_text_punctuation(current, inplace=True)),
            ("pdf_text_finalize", lambda current: finalize_pdf_text_layer(current, inplace=True)),
        ])

        before_blocks = len(base.blocks)
        total = len(stages)
        current = base
        for index, (step_id, fn) in enumerate(stages):
            if progress_callback is not None:
                progress_callback(step_id, index, total)
            current = fn(current)
            if verbose and current.processing_log:
                print(f"  ▶  {step_id} ... {current.processing_log[-1].get('message', 'done')}")
        if progress_callback is not None:
            progress_callback("done", total, total)
    finally:
        if gc_was_enabled:
            gc.enable()


    out = current
    guard = dict(getattr(out.metadata, "pdf_text_guard_report", {}) or {})
    report = {
        "profile": "selectable_pdf_publication_v1",
        "execution_mode": "streamlined_inplace",
        "blocks_before": before_blocks,
        "blocks_after": len(out.blocks),
        "toc_entries": 0,
        "chapter_detection": "deferred_to_ai",
        "chapter_candidates_preserved": int(getattr(out.metadata, "pdf_text_chapter_candidates_deferred", 0) or 0),
        "keep_author_notes": bool(keep_author_notes),
        "remove_generated_matter": bool(remove_generated_matter),
        "restore_indents": bool(restore_indents),
        "ruby_chars_filtered_at_extraction": int(
            getattr(out.metadata, "pdf_text_furigana_chars_skipped", 0) or 0
        ),
        "page_number_chars_filtered_at_extraction": int(
            getattr(out.metadata, "pdf_text_page_number_chars_skipped", 0) or 0
        ),
        "character_guard": guard,
        "internal_snapshots": 1,
    }
    out.metadata.pdf_format_processed = True
    out.metadata.pdf_format_report = report
    out.add_log(
        "pdf_format_pipeline",
        (
            f"PDF格式处理完成：{before_blocks}→{len(out.blocks)} blocks；"
            f"章节/目录交由 AI；Ruby {report['ruby_chars_filtered_at_extraction']} 字符、"
            f"页码 {report['page_number_chars_filtered_at_extraction']} 字符在提取层过滤；"
            f"字符守卫 {'PASS' if guard.get('passed', True) else 'FAIL'}；"
            "专用管线仅保存最终 1 个版本"
        ),
        0 if guard.get("passed", True) else int(guard.get("missing_chars", 0) or 0) + int(guard.get("extra_chars", 0) or 0),
    )

    # One user-visible PDF-format operation == one history commit.
    target_repo = None
    if doc.repo is not None:
        out.repo = doc.repo
        out.commit_id = doc.commit_id
        target_repo = str(doc.repo.path)
    elif repo_path:
        target_repo = str(repo_path)
    else:
        from models.document import new_temp_repo_path
        target_repo = new_temp_repo_path()
    # Standard-library JSON serialization of a 100k-block snapshot can itself
    # take longer than the whole PDF extraction/format pass.  When the optional
    # fast repository encoder is unavailable, skip only the history snapshot for
    # very large inputs; the workspace still persists the formatted document and
    # the source PDF can be re-extracted deterministically.
    fast_repo = False
    try:
        import orjson  # noqa: F401
        fast_repo = True
    except ImportError:
        fast_repo = False
    if before_blocks > 60000 and not fast_repo:
        out.metadata.pdf_format_history_skipped = True
        out.metadata.pdf_format_history_skip_reason = "large_pdf_without_fast_json"
        out.add_log("pdf_format_history", "超长 PDF：未安装可选 fast-json，跳过一次性历史快照以避免界面长时间卡顿", 0)
        out.repo = doc.repo
        out.commit_id = doc.commit_id
    else:
        # In an end-to-end 4k+ page run the caller still owns the 100k+ block
        # raw document while this formatted snapshot is serialized.  Re-enabling
        # cyclic GC just before commit made CPython scan both large object graphs
        # repeatedly and could turn an otherwise ~35s workflow into >120s.
        # Keep cyclic GC paused only for the one final history snapshot; normal
        # reference counting still releases temporaries and text semantics are
        # unchanged.
        commit_gc_was_enabled = gc.isenabled()
        if commit_gc_was_enabled:
            gc.disable()
        try:
            out.commit(target_repo, "pdf_format_pipeline", "PDF专用格式处理完成")
        finally:
            if commit_gc_was_enabled:
                gc.enable()
    return out
