#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PDF Craft adapter with isolated environment and opt-in model installation."""
from __future__ import annotations

import os
from pathlib import Path

from adapters.ocr_engine_common import iter_worker_jsonl, run_ocr_engine
from adapters.runtime_env import ensure_venv

ROOT = Path(__file__).parent.parent
VENV_DIR = ROOT / ".venv-pdf-craft"
WORKER_SCRIPT = Path(__file__).parent / "pdf_craft_worker.py"
MODEL_CACHE = ROOT / ".model-cache" / "pdf-craft"
PACKAGE = os.environ.get("NOVEL_FORMATTER_PDF_CRAFT_PACKAGE", "pdf-craft==1.0.13")


def setup_venv(verbose: bool = True) -> Path:
    return ensure_venv(
        VENV_DIR,
        label="PDF Craft",
        marker_code="from pdf_craft import transform_markdown, predownload_models; import torch, PIL",
        packages=["torch", "torchvision", "pillow", PACKAGE],
        verbose=verbose,
        min_minor=10,
        max_minor=13,
    )


def run(*, ocr_size: str = "base", verbose: bool = True, **kwargs):
    if ocr_size not in {"tiny", "small", "base", "large", "gundam"}:
        raise ValueError(f"未知 PDF Craft OCR 模型尺寸: {ocr_size}")
    from adapters.ocr_runtime_catalog import runtime_ready
    prepare_models = not runtime_ready("pdf_craft")
    python = setup_venv(verbose=verbose)
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)

    def worker_fn(ocr_paths, cancel_check):
        cmd = [
            str(python), str(WORKER_SCRIPT),
            "--model-cache", str(MODEL_CACHE),
            "--ocr-size", ocr_size,
            *(["--prepare-models"] if prepare_models else []),
            *list(ocr_paths),
        ]
        return iter_worker_jsonl(cmd, cancel_check=cancel_check, engine_label="PDF Craft")

    doc = run_ocr_engine(
        worker_fn,
        source_engine=f"pdf_craft:{ocr_size}",
        verbose=verbose,
        **kwargs,
    )
    from adapters.ocr_runtime_catalog import mark_runtime_ready
    # Only mark after the worker completed without a process-level exception.
    # Per-page model/CUDA errors remain visible and do not falsely imply support.
    if any((block.text or "").strip() for block in doc.blocks):
        mark_runtime_ready("pdf_craft", ocr_size=ocr_size)
    return doc
