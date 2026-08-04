#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Isolated PDF Craft worker; emits one JSON object per source image."""
from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
import tempfile
from pathlib import Path


def markdown_blocks(text: str) -> list[dict]:
    blocks: list[dict] = []
    pending: list[str] = []

    def flush():
        if not pending:
            return
        value = "".join(part.strip() for part in pending if part.strip()).strip()
        pending.clear()
        if value:
            blocks.append({"text": value, "confidence": 0.92, "box": None, "label": "paragraph"})

    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        heading = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if heading:
            flush()
            blocks.append({"text": heading.group(1).strip(), "confidence": 0.95, "box": None, "label": "title"})
            continue
        if re.match(r"^!\[[^]]*]\([^)]*\)\s*$", line):
            flush()
            continue
        if line.startswith(("- ", "* ", "+ ")):
            flush()
            blocks.append({"text": line[2:].strip(), "confidence": 0.9, "box": None, "label": "paragraph"})
            continue
        pending.append(line)
    flush()
    return blocks


def process(path: str, *, cache: Path, ocr_size: str) -> list[dict]:
    from PIL import Image
    from pdf_craft import transform_markdown

    with tempfile.TemporaryDirectory(prefix="novel_formatter_pdf_craft_page_") as td:
        temp = Path(td)
        pdf_path = temp / "page.pdf"
        md_path = temp / "page.md"
        assets = temp / "assets"
        analysis = temp / "analysis"
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.save(pdf_path, "PDF", resolution=300.0)
            image.close()
        # Keep the JSONL stdout protocol clean; upstream progress goes to stderr.
        with contextlib.redirect_stdout(sys.stderr):
            transform_markdown(
                pdf_path=str(pdf_path),
                markdown_path=str(md_path),
                markdown_assets_path=str(assets),
                analysing_path=str(analysis),
                ocr_size=ocr_size,
                models_cache_path=str(cache),
                local_only=True,
                dpi=300,
                includes_cover=True,
                includes_footnotes=True,
                ignore_pdf_errors=False,
                ignore_ocr_errors=False,
                generate_plot=False,
                toc_assumed=False,
            )
        return markdown_blocks(md_path.read_text(encoding="utf-8", errors="replace"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-cache", required=True)
    parser.add_argument("--ocr-size", default="base", choices=["tiny", "small", "base", "large", "gundam"])
    parser.add_argument("--prepare-models", action="store_true")
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()

    cache = Path(args.model_cache)
    cache.mkdir(parents=True, exist_ok=True)
    if args.prepare_models:
        try:
            from pdf_craft import predownload_models
            with contextlib.redirect_stdout(sys.stderr):
                predownload_models(models_cache_path=str(cache), revision=None)
        except Exception as exc:
            message = (
                "PDF Craft 模型准备失败。上游当前实际 OCR 转换要求 Poppler 与 CUDA；"
                "在 macOS/纯 CPU 环境可能无法运行。原始错误：" + str(exc)
            )
            for path in args.paths:
                print(json.dumps({"ok": False, "path": path, "error": message}, ensure_ascii=False), flush=True)
            return 0

    for path in args.paths:
        try:
            blocks = process(path, cache=cache, ocr_size=args.ocr_size)
            print(json.dumps({"ok": True, "path": path, "blocks": blocks}, ensure_ascii=False), flush=True)
        except Exception as exc:
            message = (
                "PDF Craft 识别失败。请确认已安装 Poppler，并在受支持的 CUDA 环境运行；"
                "Mac/CPU 当前可能不被上游 DeepSeek-OCR 推理支持。原始错误：" + str(exc)
            )
            print(json.dumps({"ok": False, "path": path, "error": message}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
