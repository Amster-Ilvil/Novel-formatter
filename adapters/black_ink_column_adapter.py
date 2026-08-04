#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Black-ink-only layout adapter for the independent handwriting OCR card.

This adapter never invokes a text OCR engine. The common runner prepares the
fixed body crop, then this module detects physical vertical columns directly
from black-pixel projection and emits stable empty column blocks. The independent
handwriting stage fills those blocks from the column raster itself.
"""
from __future__ import annotations

import os
from PIL import Image

from adapters.column_ocr_adapter import detect_vertical_columns
from adapters.ocr_engine_common import run_ocr_engine


def _iter_black_ink_pages(
    page_paths: list[str],
    *,
    sensitivity: int,
    padding_percent: int,
    max_columns: int,
    strict: bool,
    cancel_check=None,
    phase_callback=None,
    fixed_region_rect=None,
):
    for page_index, page_path in enumerate(page_paths, start=1):
        if cancel_check is not None and cancel_check():
            break
        try:
            with Image.open(page_path) as source:
                image = source.convert("RGB")
            try:
                columns = detect_vertical_columns(
                    image,
                    sensitivity=sensitivity,
                    padding_percent=padding_percent,
                    max_columns=max_columns,
                    fixed_region_rect=fixed_region_rect,
                    fixed_region_already_masked=bool(fixed_region_rect),
                )
            finally:
                image.close()
        except Exception as exc:
            yield page_path, None, f"黑色像素分列失败：{exc}"
            continue

        if phase_callback is not None:
            phase_callback(
                "black_ink_columns",
                page_index,
                len(page_paths),
                f"{os.path.basename(page_path)} · 黑色像素检测到 {len(columns)} 列",
            )

        if not columns:
            if strict:
                yield page_path, None, "固定区域中没有检测到稳定的黑色竖排文字列"
            else:
                yield page_path, [], None
            continue

        expected = len(columns)
        blocks: list[dict] = []
        for column_index, column in enumerate(columns):
            column_id = f"p{page_index:05d}:c{column_index + 1:03d}"
            blocks.append({
                "text": "",
                "confidence": 0.0,
                "box": column.polygon(),
                "direction": "vertical",
                "layout_group": "fixed_region_column",
                "layout_order": column_index,
                "recognizer": "black_ink_trace",
                "column_id": column_id,
                "column_index": column_index + 1,
                "column_expected_count": expected,
                "column_ocr_empty": True,
                "column_requires_handwriting": True,
                "preserve_empty_ocr_column": True,
                "preserve_ocr_item": True,
                "black_ink_layout_only": True,
                "black_ink_estimated_chars": int(getattr(column, "estimated_chars", 0) or 0),
                "black_ink_content_spans": [
                    list(span) for span in (getattr(column, "content_spans", ()) or ())
                ],
            })
        yield page_path, blocks, None


def run(
    *,
    column_sensitivity: int = 55,
    column_padding_percent: int = 10,
    strict_column_validation: bool = True,
    max_columns: int = 80,
    verbose: bool = True,
    **kwargs,
):
    """Build a fixed-region column document using only black pixels."""
    phase_callback = kwargs.pop("phase_callback", None)
    fixed_region_rect = kwargs.get("crop_rect")

    def worker_fn(ocr_paths, cancel_check):
        yield from _iter_black_ink_pages(
            list(ocr_paths),
            sensitivity=int(column_sensitivity),
            padding_percent=int(column_padding_percent),
            max_columns=int(max_columns),
            strict=bool(strict_column_validation),
            cancel_check=cancel_check,
            phase_callback=phase_callback,
            fixed_region_rect=fixed_region_rect,
        )

    return run_ocr_engine(
        worker_fn,
        source_engine="black_ink_handwriting",
        verbose=verbose,
        force_text_pages=True,
        strict_column_audit=bool(strict_column_validation),
        **kwargs,
    )
