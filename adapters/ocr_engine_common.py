#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""外部 OCR 引擎的共享运行框架。

PaddleOCR / Manga-OCR / NDLOCR-Lite 等引擎的流程完全一致：
    展开输入（图片/PDF）→ 可选裁剪 → 跳过非正文页 → worker 子进程逐页识别
    → 可选页眉过滤 → 页面自动分类 → 组装 UnifiedDocument
唯一不同的是 worker 怎么起。这里把公共流程收拢成 run_ocr_engine()，
各适配器只提供 worker_fn(ocr_paths, cancel_check) -> (path, blocks, error) 迭代器。

blocks 协议：[{"text": str, "confidence": float, "box": [[x,y]×4] | None}, ...]
box 为 None 时不生成 bbox。Manga OCR 页面入口会先进入物理分列适配器，不再直接识别整页。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.document import (
    UnifiedDocument, Block, BlockType, PageInfo, BoundingBox, Metadata, TocEntry
)
from adapters.apple_vision_adapter import (
    detect_running_headers, auto_classify_pages as auto_classify_pages_japanese, CHAPTER_RE,
)
from adapters.ocr_profiles import (
    apply_profile_metadata, auto_classify_pages as auto_classify_pages_profile,
    get_ocr_profile, is_chapter_title, is_chinese_horizontal, normalize_ocr_mode,
)
from adapters.horizontal_text_layout import prepare_items_for_mode
from engine.page_ocr_policy import split_ocr_pages

_KNOWN_SPURIOUS_OCR_TOKENS = {
    "RTMICO",
}
_JAPANESE_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_SHORT_LATIN_NOISE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\s,._:;|/\\-]{2,15}")
_OCR_PLACEHOLDER_CHARS = frozenset("□■◻◼�")


def is_spurious_ocr_item(text: str, ocr_mode: str = "ja_vertical") -> bool:
    """Return True for isolated OCR debris that should not become prose.

    Some OCR engines can occasionally read page-edge artifacts as short Latin tokens
    such as "RTMICO".  Real Japanese light-novel body text should contain
    Japanese characters; this deliberately only catches short standalone
    Latin/digit/symbol blocks, not mixed text.
    """
    stripped = str(text or "").strip()
    compact = re.sub(r"\s+", "", stripped)
    if not compact:
        return True
    # A recognizer may use visible squares/replacement characters to say that
    # it did not decode the supplied physical column.  Treat a placeholder-only
    # packet as an OCR failure rather than Japanese prose.  Mixed partial text
    # remains available for manual review, but its confidence is forced to zero
    # by the authoritative column pipeline.
    if compact and all(ch in _OCR_PLACEHOLDER_CHARS for ch in compact):
        return True
    if compact in _KNOWN_SPURIOUS_OCR_TOKENS:
        return True
    # Horizontal Chinese pages commonly contain standalone Latin headings,
    # ISBNs, URLs and numbered labels.  The old Japanese-only heuristic would
    # silently delete those short blocks, so the Chinese profile keeps them.
    if is_chinese_horizontal(ocr_mode):
        return False
    if _JAPANESE_CHAR_RE.search(stripped):
        return False
    return bool(_SHORT_LATIN_NOISE_RE.fullmatch(stripped))


def page_size(image_path: str) -> tuple[int, int]:
    from PIL import Image
    with Image.open(image_path) as img:
        return img.size  # (w, h)


def _append_tail(lines: list[str], line: str, *, limit: int = 400) -> None:
    value = str(line or "").rstrip()
    if not value:
        return
    lines.append(value)
    if len(lines) > limit:
        del lines[: max(1, limit // 2)]


def iter_worker_jsonl(cmd: list[str], cancel_check=None, engine_label: str = "OCR"):
    """Run a JSONL worker with drained stderr, cancellation and an idle timeout."""
    from adapters.subprocess_watchdog import (
        LinePump, ProcessCancelled, env_seconds, isolated_process_kwargs,
        terminate_process,
    )

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        **isolated_process_kwargs(),
    )
    stdout_pump = LinePump(proc.stdout, name=f"{engine_label}-stdout")
    stderr_pump = LinePump(proc.stderr, name=f"{engine_label}-stderr")
    stderr_lines: list[str] = []
    timeout = env_seconds("NOVEL_FORMATTER_OCR_REQUEST_TIMEOUT", 300.0, minimum=30.0)
    terminated = False

    def drain_stderr() -> None:
        for item in stderr_pump.get_nowait_lines():
            _append_tail(stderr_lines, item)

    try:
        while True:
            try:
                line = stdout_pump.readline(
                    proc=proc,
                    timeout=timeout,
                    cancel_check=cancel_check,
                    label=engine_label,
                    on_wait=drain_stderr,
                )
            except ProcessCancelled:
                terminated = True
                return
            if line is None:
                break
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                _append_tail(stderr_lines, line)
                continue
            if data.get("ok"):
                yield data.get("path", ""), data.get("blocks") or [], None
            else:
                yield data.get("path", ""), None, data.get("error", "未知错误")
    finally:
        drain_stderr()
        if proc.poll() is None:
            ret = terminate_process(proc) if terminated else None
            if ret is None:
                try:
                    ret = proc.wait(timeout=20)
                except Exception:
                    ret = terminate_process(proc)
        else:
            ret = proc.poll()
        stdout_pump.close()
        stderr_pump.close()
        if ret not in (0, -15, None) and not terminated:
            tail = "\n".join(stderr_lines[-40:])
            raise RuntimeError(f"{engine_label} worker 异常退出 (code={ret}):\n{tail}")


def iter_server_worker_jsonl(
    cmd: list[str],
    image_paths: list[str],
    *,
    cancel_check=None,
    engine_label: str = "OCR",
    batch_size: int = 64,
):
    """Run a persistent JSONL OCR worker with bounded requests and a watchdog."""
    from adapters.subprocess_watchdog import (
        LinePump, ProcessCancelled, env_seconds, isolated_process_kwargs,
        terminate_process,
    )

    paths = [str(path) for path in image_paths]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        **isolated_process_kwargs(),
    )
    stdout_pump = LinePump(proc.stdout, name=f"{engine_label}-stdout")
    stderr_pump = LinePump(proc.stderr, name=f"{engine_label}-stderr")
    stderr_lines: list[str] = []
    request_timeout = env_seconds("NOVEL_FORMATTER_OCR_REQUEST_TIMEOUT", 300.0, minimum=30.0)
    terminated = False

    def drain_stderr() -> None:
        for item in stderr_pump.get_nowait_lines():
            _append_tail(stderr_lines, item)

    try:
        assert proc.stdin is not None
        size = max(1, int(batch_size or 64))
        request_id = 0
        for start in range(0, len(paths), size):
            if callable(cancel_check) and cancel_check():
                terminated = True
                terminate_process(proc)
                break
            request_id += 1
            chunk = paths[start:start + size]
            request = {"request_id": request_id, "images": chunk}
            proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            proc.stdin.flush()

            while True:
                try:
                    line = stdout_pump.readline(
                        proc=proc,
                        timeout=request_timeout,
                        cancel_check=cancel_check,
                        label=f"{engine_label} request {request_id}",
                        on_wait=drain_stderr,
                    )
                except ProcessCancelled:
                    terminated = True
                    break
                if line is None:
                    tail = "\n".join(stderr_lines[-40:])
                    raise RuntimeError(
                        f"{engine_label} server 提前退出 (code={proc.poll()})\n{tail}"
                    )
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    _append_tail(stderr_lines, line)
                    continue
                if (
                    (data.get("batch_done") or data.get("type") == "request_done")
                    and data.get("request_id") == request_id
                ):
                    break
                path = str(data.get("path", ""))
                if data.get("ok"):
                    yield path, data.get("blocks") or [], None
                else:
                    yield path, None, data.get("error", "未知错误")
            if terminated:
                break

        if not terminated and proc.poll() is None:
            try:
                proc.stdin.write(json.dumps({"command": "close"}) + "\n")
                proc.stdin.flush()
            except Exception:
                pass
    finally:
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        drain_stderr()
        if proc.poll() is None:
            try:
                ret = proc.wait(timeout=20)
            except Exception:
                ret = terminate_process(proc)
        else:
            ret = proc.poll()
        stdout_pump.close()
        stderr_pump.close()
        if ret not in (0, -15, None) and not terminated:
            tail = "\n".join(stderr_lines[-40:])
            raise RuntimeError(f"{engine_label} server 异常退出 (code={ret}):\n{tail}")

def run_ocr_engine(
    worker_fn,
    source_engine: str,
    image_folder: str | None = None,
    page_overrides: dict[int, str] | None = None,
    verbose: bool = True,
    input_paths: list[str] | None = None,
    progress_callback=None,
    cancel_check=None,
    crop_top: float = 0.0,
    crop_bottom: float = 0.0,
    crop_rect: tuple[float, float, float, float] | None = None,
    temp_crop_dir: str | None = None,
    reuse_existing_crops: bool = False,
    filter_running_headers: bool = True,
    force_text_pages: bool = False,
    strict_column_audit: bool = False,
    ocr_mode: str = "ja_vertical",
    merge_horizontal_fragments: bool = True,
) -> UnifiedDocument:
    """通用引擎流程；参数语义与 apple_vision_adapter.run() 一致。"""
    ocr_mode = normalize_ocr_mode(ocr_mode)
    profile = get_ocr_profile(ocr_mode)
    overrides = {int(k): v for k, v in (page_overrides or {}).items()}

    from adapters.pdf_input import expand_inputs, natural_sort_key

    raw_inputs = input_paths if input_paths else ([image_folder] if image_folder else [])
    if not raw_inputs:
        raise ValueError("必须提供 image_folder 或 input_paths")

    work_dir = image_folder or str(Path(raw_inputs[0]).parent)
    image_paths = expand_inputs(
        raw_inputs, work_dir=work_dir, cancel_check=cancel_check
    )
    image_paths = sorted(set(image_paths), key=natural_sort_key)

    if not image_paths:
        raise FileNotFoundError(f"未找到可处理的图片/PDF: {raw_inputs}")

    # Admission is decided before any crop, mask, column split or recognizer
    # setup.  Classified asset pages must not pay preprocessing cost or leak
    # cover/TOC/colophon text into cross-page header statistics.
    source_pages_to_ocr, skipped_asset_pages = split_ocr_pages(image_paths, overrides)
    ocr_inputs_by_page: dict[int, str] = {}
    cancelled_before_recognition = False
    if crop_rect is not None or crop_top > 0 or crop_bottom > 0:
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
        from adapters.apple_vision_adapter import crop_for_ocr
        source_paths = [path for _, path in source_pages_to_ocr]
        # Crop only admitted body/unknown pages. Cancellation must also stop the
        # preprocessing queue: the former pool.map/list construct waited for all
        # queued pages even after the GUI Stop button had been pressed.
        executor = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4))
        futures = [
            executor.submit(
                crop_for_ocr,
                source_path,
                crop_top=crop_top,
                crop_bottom=crop_bottom,
                crop_rect=crop_rect,
                out_dir=temp_crop_dir,
                reuse_existing=bool(reuse_existing_crops),
            )
            for source_path in source_paths
        ]
        crop_cancelled = False
        try:
            for source_entry, future in zip(source_pages_to_ocr, futures):
                if callable(cancel_check) and cancel_check():
                    crop_cancelled = True
                    break
                while True:
                    if callable(cancel_check) and cancel_check():
                        crop_cancelled = True
                        break
                    try:
                        cropped = future.result(timeout=0.10)
                        break
                    except FutureTimeout:
                        continue
                if crop_cancelled:
                    break
                if callable(cancel_check) and cancel_check():
                    crop_cancelled = True
                    break
                ocr_inputs_by_page[source_entry[0]] = cropped
        finally:
            if crop_cancelled:
                cancelled_before_recognition = True
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=True)
    else:
        ocr_inputs_by_page = {
            page_no: path for page_no, path in source_pages_to_ocr
        }

    pages_to_ocr = [
        (page_no, ocr_inputs_by_page[page_no])
        for page_no, _source in source_pages_to_ocr
        if page_no in ocr_inputs_by_page
    ]

    if verbose:
        print(f"📂  共 {len(image_paths)} 张图片（含 PDF 转换页面）")
        if skipped_asset_pages:
            detail = ", ".join(
                f"第{page_no}页={page_type}"
                for page_no, _path, page_type in skipped_asset_pages
            )
            print(
                f"🚫  OCR 前置隔离 {len(skipped_asset_pages)} 页：{detail}；"
                "不裁剪、不分列、不调用识别模型"
            )

    # ── OCR：流式处理——每页结果一到就写入并立刻回报进度/预览，
    #    不等整批完成（否则 GUI 的实时预览会一直卡在第一页）。
    raw_items_per_page: list[list[dict]] = [[] for _ in image_paths]
    page_idx_by_path = {path: i for i, path in pages_to_ocr}
    processed = 0
    if pages_to_ocr:
        ocr_paths = [p for _, p in pages_to_ocr]
        for path, blocks, error in worker_fn(ocr_paths, cancel_check):
            i = page_idx_by_path.get(path)
            if i is None:
                continue
            if error:
                if verbose:
                    print(f"  ⚠️  第{i}页识别失败: {error}")
                blocks = []
            raw_items_per_page[i - 1] = blocks
            processed += 1
            if verbose:
                print(f"  [{i:3d}/{len(image_paths)}] OCR → {len(blocks)} 个文本块")
            if progress_callback is not None:
                # 预览用原图（未裁剪），这样 GUI 上画的框选矩形才对得上显示的图片
                progress_callback(processed, len(pages_to_ocr), os.path.basename(path), image_paths[i - 1])
    cancelled = bool(
        cancelled_before_recognition
        or (
            processed < len(pages_to_ocr)
            and cancel_check is not None
            and cancel_check()
        )
    )

    # The Simplified-Chinese profile owns a separate whole-page horizontal
    # layout stage.  Japanese mode returns the original list byte-for-byte.
    raw_items_per_page = [
        prepare_items_for_mode(
            items, ocr_mode, merge_horizontal_fragments=merge_horizontal_fragments
        )
        for items in raw_items_per_page
    ]

    # ── 可选页眉/页脚检测与过滤 ────────────────────────────────────────────────
    # “逐列成句”以文字完整性为最高优先级。旧逻辑会在进入逐列重建之前，
    # 根据跨页重复文本自动删除疑似页眉；竖排小说最左/最右正文列一旦被误判，
    # 后续状态机已经拿不到原文，便会出现整列、整句吞失。调用方在启用逐列成句
    # 时会传 filter_running_headers=False：保留所有真实 OCR 文本，只清除明确的
    # 空块/孤立噪声。页眉若确实需要删除，应由 Formatter 的“清理模块”显式完成。
    all_texts_per_page = [[it["text"] for it in items] for items in raw_items_per_page]
    running_headers = (
        detect_running_headers(all_texts_per_page) if filter_running_headers else set()
    )
    if verbose and running_headers:
        print(f"\n🔍  检测到页眉/页脚（将过滤）：")
        for h in sorted(running_headers):
            print(f"    「{h}」")
    elif verbose and not filter_running_headers:
        print("\n🛡️  逐列成句保护：跳过自动页眉/页脚删除，保留 OCR 原始文字列")

    filtered_items_per_page = [
        [
            it for it in items
            if (not filter_running_headers or it["text"].strip() not in running_headers)
            and (
                bool(it.get("preserve_ocr_item"))
                or not is_spurious_ocr_item(it.get("text", ""), ocr_mode)
            )
        ]
        for items in raw_items_per_page
    ]

    # Fixed-region column OCR carries immutable column IDs.  Build a page audit
    # before classification/Formatter can change structure, then verify the same
    # IDs reach the UnifiedDocument.  Counts alone are insufficient because a
    # duplicated middle column could otherwise hide a missing final column.
    column_audit_seed: dict[int, dict] = {}
    for page_index, items in enumerate(raw_items_per_page, start=1):
        columns = [it for it in items if it.get("layout_group") == "fixed_region_column"]
        if not columns:
            continue
        expected = max(
            [int(it.get("column_expected_count", 0) or 0) for it in columns] or [len(columns)]
        )
        ids = [str(it.get("column_id", "")) for it in columns if str(it.get("column_id", ""))]
        recognized_ids = list(dict.fromkeys(ids))
        pending_manual_ids = list(dict.fromkeys(
            str(it.get("column_id", ""))
            for it in columns
            if str(it.get("column_id", ""))
            and bool(it.get("column_ocr_empty") or it.get("column_requires_handwriting"))
        ))
        rescue_methods: dict[str, int] = {}
        for item in columns:
            if not bool(item.get("column_ocr_rescue_used", False)):
                continue
            method = str(item.get("column_ocr_rescue_method", "") or "unspecified")
            rescue_methods[method] = rescue_methods.get(method, 0) + 1
        column_audit_seed[page_index] = {
            "expected": expected,
            # ``recognized`` remains the number of physical column records
            # returned by the adapter, preserving the v2 audit contract.
            "recognized": len(recognized_ids),
            "text_recognized": max(0, len(recognized_ids) - len(pending_manual_ids)),
            "pending_manual": len(pending_manual_ids),
            "pending_manual_ids": pending_manual_ids,
            "rescue_count": sum(rescue_methods.values()),
            "rescue_methods": rescue_methods,
            "column_ids": recognized_ids,
        }

    filtered_texts_per_page = [[it["text"] for it in items] for items in filtered_items_per_page]
    auto_types = (
        auto_classify_pages_profile(image_paths, filtered_texts_per_page, ocr_mode)
        if is_chinese_horizontal(ocr_mode)
        else auto_classify_pages_japanese(image_paths, filtered_texts_per_page)
    )

    # ── 构建 UnifiedDocument ─────────────────────────────────────────────────
    doc = UnifiedDocument()
    doc.metadata = Metadata(source_engine=source_engine, language=profile.language)
    apply_profile_metadata(doc, ocr_mode)

    order_counter = 0
    skipped_image_count = 0
    text_page_count = 0
    chapter_index = 0

    for i, path in enumerate(image_paths):
        page_no = i + 1
        fname = os.path.basename(path)

        if page_no in overrides:
            try:
                ptype = BlockType(overrides[page_no])
                conf = 1.0
            except ValueError:
                ptype = auto_types[i]
                conf = 0.90
        else:
            ptype = auto_types[i]
            conf = 0.90

        # Once a user has fixed the body region and the column detector found
        # physical text columns, auto-classification must not turn a short final
        # page into an illustration/colophon and discard all recognized text.
        if force_text_pages and page_no in column_audit_seed:
            ptype = BlockType.PARAGRAPH
            conf = 1.0

        doc.pages.append(PageInfo(page_no=page_no, page_type=ptype, image_path=path, confidence=conf))

        if ptype == BlockType.BLANK:
            if verbose:
                print(f"  ·空白  第{page_no}页: {fname}")
            continue

        if ptype != BlockType.PARAGRAPH:
            last_idx = len(doc.blocks) - 1
            anchor = f"block_{last_idx}" if last_idx >= 0 else "start"
            doc.blocks.append(Block(
                type=BlockType.IMAGE_REF, image_path=path, image_anchor=anchor,
                page=page_no, reading_order=order_counter, confidence=conf,
            ))
            order_counter += 1
            skipped_image_count += 1
            if verbose:
                print(f"  🖼️  图片  第{page_no}页: {fname} ({ptype.value})  anchor={anchor}")
            continue

        # 正文页：需要页面像素尺寸才能把 box 归一化成 bbox。固定正文框现在
        # 使用整页同尺寸掩膜，因此 OCR 输入和原页共享同一坐标系；这里仍读取
        # 实际 OCR 输入尺寸，兼容没有框选和其他调用路径。
        try:
            page_w, page_h = page_size(ocr_inputs_by_page.get(page_no, image_paths[i]))
        except Exception:
            page_w, page_h = (0, 0)

        for order_in_page, item in enumerate(filtered_items_per_page[i]):
            text = str(item.get("text", "") or "").strip()
            preserve_empty = bool(
                item.get("preserve_empty_ocr_column")
                or item.get("column_requires_handwriting")
                or item.get("black_ink_layout_only")
            )
            if not text and not preserve_empty:
                continue

            if text and (item.get("label") == "title" or is_chapter_title(text, ocr_mode)):
                btype = BlockType.CHAPTER
            elif text.startswith(('「', '『')) or text.endswith(('」', '』')):
                btype = BlockType.DIALOGUE
            else:
                btype = BlockType.PARAGRAPH

            bbox = None
            box = item.get("box")
            if box and page_w and page_h:
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                bbox = BoundingBox.from_pixels(min(xs), min(ys), max(xs), max(ys), page_w, page_h)

            item_metadata = {
                key: item[key]
                for key in (
                    "layout_group", "layout_order", "layout_text", "recognizer", "label",
                    "horizontal_fragment_count", "horizontal_source_indices",
                    "column_id", "column_index", "column_expected_count",
                    "column_ocr_empty", "column_requires_handwriting",
                    "preserve_empty_ocr_column", "preserve_ocr_item",
                    "column_manual_placeholder", "column_ocr_attempts",
                    "column_detection_failed", "column_count_unverified",
                    "column_detector_mode", "column_detector_version",
                    "column_detection_error",
                    "column_ocr_failure_reason", "column_ocr_selected_variant",
                    "column_ocr_preprocess_used", "column_ocr_candidate_conflict",
                    "column_consensus_seeded",
                    "column_ocr_candidates", "column_ocr_compare_mode",
                    "column_ocr_rescue_policy", "column_ocr_rescue_budget",
                    "column_ocr_rescue_used", "column_ocr_rescue_method",
                    "column_ocr_rescue_reason",
                    "column_ocr_transport", "column_ndlocr_page_mode",
                    "sentence_context_reocr_group",
                    "sentence_context_reocr_column_ids",
                    "sentence_context_reocr_column_count",
                    "sentence_context_reocr_layout",
                    "sentence_context_reocr_page_runs",
                    "sentence_context_reocr_candidate",
                    "sentence_context_reocr_baseline",
                    "sentence_context_reocr_confidence",
                    "sentence_context_reocr_accepted",
                    "sentence_context_reocr_skipped",
                    "sentence_context_reocr_strategy",
                    "sentence_context_reocr_reason",
                    "sentence_context_reocr_owner_column_id",
                    "sentence_context_reocr_position",
                    "sentence_context_reocr_owner",
                    "ocr_review_sentence_image_path",
                    "black_ink_layout_only", "black_ink_estimated_chars",
                    "black_ink_content_spans",
                )
                if item.get(key) is not None
            }
            if item.get("layout_group") == "fixed_region_column":
                item_metadata["column_model_written"] = True
            block = Block(
                type=btype, text=text, ocr_raw=text, page=page_no,
                page_index=page_no, page_number=page_no,
                order_in_page=order_in_page,
                text_direction=(
                    "horizontal" if is_chinese_horizontal(ocr_mode)
                    else item.get("direction")
                ),
                source_format="ocr",
                metadata=item_metadata,
                bbox=bbox, reading_order=order_counter,
                confidence=item.get("confidence", 0.9),
            )

            if btype == BlockType.CHAPTER:
                chapter_index += 1
                block.chapter_index = chapter_index
                doc.toc.append(TocEntry(title=text, chapter_index=chapter_index, block_index=len(doc.blocks)))

            doc.blocks.append(block)
            order_counter += 1

        text_page_count += 1
        if verbose:
            print(f"  📄 正文  第{page_no}页: {fname}  → {len(filtered_items_per_page[i])} 块")

    # Final per-page integrity report: expected detector columns, successful OCR
    # columns, and columns actually written into the document model must be the
    # same ID set.  DOCX export performs the fourth/final verification later.
    if column_audit_seed:
        written_by_page: dict[int, list[str]] = {}
        for block in doc.blocks:
            column_id = str((block.metadata or {}).get("column_id", ""))
            if not column_id:
                continue
            written_by_page.setdefault(int(block.page or 0), []).append(column_id)

        pages_report: dict[str, dict] = {}
        mismatch_lines: list[str] = []
        total_expected = total_recognized = total_written = 0
        total_text_recognized = total_pending_manual = total_rescue_count = 0
        total_rescue_methods: dict[str, int] = {}
        for page_no in sorted(column_audit_seed):
            seed = column_audit_seed[page_no]
            expected_ids = list(seed["column_ids"])
            written_ids = list(dict.fromkeys(written_by_page.get(page_no, [])))
            expected = int(seed["expected"])
            recognized = int(seed["recognized"])
            text_recognized = int(seed.get("text_recognized", recognized) or 0)
            pending_manual = int(seed.get("pending_manual", 0) or 0)
            pending_manual_ids = list(seed.get("pending_manual_ids") or [])
            rescue_count = int(seed.get("rescue_count", 0) or 0)
            rescue_methods = dict(seed.get("rescue_methods") or {})
            model_written = len(written_ids)
            missing_ids = [column_id for column_id in expected_ids if column_id not in written_ids]
            extra_ids = [column_id for column_id in written_ids if column_id not in expected_ids]
            passed = (
                expected == recognized == model_written
                and not missing_ids
                and not extra_ids
            )
            pages_report[str(page_no)] = {
                "expected": expected,
                "recognized": recognized,
                "text_recognized": text_recognized,
                "pending_manual": pending_manual,
                "pending_manual_ids": pending_manual_ids,
                "rescue_count": rescue_count,
                "rescue_methods": rescue_methods,
                "model_written": model_written,
                "docx_written": 0,
                "column_ids": expected_ids,
                "missing_model_ids": missing_ids,
                "extra_model_ids": extra_ids,
                "passed": passed,
                "text_recognition_complete": pending_manual == 0,
            }
            total_expected += expected
            total_recognized += recognized
            total_text_recognized += text_recognized
            total_pending_manual += pending_manual
            total_rescue_count += rescue_count
            for method, count in rescue_methods.items():
                total_rescue_methods[method] = total_rescue_methods.get(method, 0) + int(count or 0)
            total_written += model_written
            if not passed:
                mismatch_lines.append(
                    f"第 {page_no} 页：预计 {expected} / 识别 {recognized} / 文档写入 {model_written}"
                )

        last_page = max(column_audit_seed)
        last_page_passed = bool(pages_report[str(last_page)]["passed"])
        doc.metadata.column_ocr_audit = {
            "version": 2,
            "pages": pages_report,
            "totals": {
                "expected": total_expected,
                "recognized": total_recognized,
                "text_recognized": total_text_recognized,
                "pending_manual": total_pending_manual,
                "rescue_count": total_rescue_count,
                "rescue_methods": total_rescue_methods,
                "model_written": total_written,
                "docx_written": 0,
            },
            "last_ocr_page": last_page,
            "last_page_all_columns_exported": last_page_passed,
            "model_integrity_passed": not mismatch_lines and last_page_passed,
            "text_recognition_complete": total_pending_manual == 0,
            "manual_review_required": total_pending_manual > 0,
            "docx_integrity_passed": False,
        }
        doc.metadata.column_ocr_integrity_passed = bool(
            doc.metadata.column_ocr_audit["model_integrity_passed"]
        )
        doc.add_log(
            "column_ocr_audit",
            f"列对账：预计 {total_expected} / 已保全 {total_recognized} / OCR有字 {total_text_recognized} / "
            f"列级救援 {total_rescue_count} / 待人工 {total_pending_manual} / 文档写入 {total_written}；"
            f"最后正文页 {last_page} {'完整' if last_page_passed else '不完整'}",
            total_written,
        )
        if mismatch_lines and strict_column_audit and not cancelled:
            raise RuntimeError(
                "精准分列列数对账失败。为避免最后若干列静默丢失，已停止输出：\n"
                + "\n".join(f"• {line}" for line in mismatch_lines[:20])
            )

    doc.add_log(
        "ocr_profile",
        f"OCR 模式：{profile.label} · {profile.language} · {profile.writing_direction}",
        len(image_paths),
    )
    if cancelled:
        doc.add_log(f"{source_engine}_adapter", f"OCR 已暂停，仅识别 {processed}/{len(pages_to_ocr)} 页", processed)
    else:
        doc.add_log(f"{source_engine}_adapter", "OCR完成", len(image_paths))
    if filter_running_headers:
        doc.add_log("header_filter", f"过滤页眉: {sorted(running_headers)}", len(running_headers))
    else:
        doc.add_log("header_filter", "逐列成句保护：未执行自动页眉/页脚删除", 0)
    doc.add_log("page_classify", f"正文页{text_page_count}，图片页{skipped_image_count}，空白页{len(image_paths)-text_page_count-skipped_image_count}")
    if skipped_asset_pages:
        doc.add_log(
            "ocr_page_admission",
            f"OCR 前置隔离 {len(skipped_asset_pages)} 个已分类非正文页；未执行裁剪、分列或识别",
            len(skipped_asset_pages),
        )
    doc.add_log("chapter_detect", f"识别章节 {chapter_index} 个", chapter_index)

    if verbose:
        print(f"\n✅  完成: {text_page_count} 正文页，{skipped_image_count} 图片页，{chapter_index} 章节，{len(doc.blocks)} 个块")

    return doc


def setup_python_venv(
    venv_dir: Path,
    packages: list[str],
    marker_import: str,
    verbose: bool = True,
    label: str = "",
) -> Path:
    """创建独立 venv 并安装依赖（幂等）。返回 venv 的 python 路径。

    以当前解释器为基础创建（App 自带的独立 Python 一定存在）；
    与 .venv-paddle 不同，torch 系引擎不挑 Python 小版本。
    """
    venv_python = venv_dir / "bin" / "python"
    if not venv_python.exists():
        if verbose:
            print(f"🔧  首次使用 {label}：创建独立虚拟环境 {venv_dir} ...")
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True, timeout=300)
        subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "--quiet"], check=True, timeout=1800)
    probe = subprocess.run([str(venv_python), "-c", f"import {marker_import}"], capture_output=True, timeout=60)
    if probe.returncode != 0:
        if verbose:
            print(f"📦  安装 {' '.join(packages)}（首次下载较大，需要几分钟）...")
        subprocess.run([str(venv_python), "-m", "pip", "install", *packages], check=True, timeout=3600)
    return venv_python
