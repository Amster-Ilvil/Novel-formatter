# -*- coding: utf-8 -*-
"""Batch OCR/formatter/EPUB orchestration."""
from __future__ import annotations

import shutil
import tempfile
import traceback
from pathlib import Path

from core.temp_manager import TempCropManager


class BatchProcessor:
    """Process each direct child directory as one book.

    Failures remain isolated to a single book, while the batch-level temporary
    root is now always removed—even when iteration or setup fails midway.
    """

    def __init__(self, input_dir, output_dir, preview_enabled=True):
        self.input_dir = Path(input_dir).expanduser()
        self.output_dir = Path(output_dir).expanduser()
        self.preview_enabled = bool(preview_enabled)
        self.temp_root = Path(tempfile.mkdtemp(prefix="novel_ocr_batch_"))

    def run(self) -> dict[str, object]:
        processed: list[str] = []
        failed: list[str] = []
        try:
            if not self.input_dir.exists():
                raise FileNotFoundError(f"批量输入目录不存在: {self.input_dir}")
            if not self.input_dir.is_dir():
                raise NotADirectoryError(f"批量输入不是目录: {self.input_dir}")

            self.output_dir.mkdir(parents=True, exist_ok=True)
            books = sorted(
                (path for path in self.input_dir.iterdir() if path.is_dir()),
                key=lambda path: path.name.casefold(),
            )
            if not books:
                print(f"[BATCH] 未找到书籍子目录: {self.input_dir}")

            for book in books:
                try:
                    self.process_book(book)
                    processed.append(book.name)
                except Exception:
                    failed.append(book.name)
                    print(f"[FAILED] {book.name}")
                    traceback.print_exc()

            print(f"[BATCH] 完成 {len(processed)} 本，失败 {len(failed)} 本")
            return {"processed": processed, "failed": failed}
        finally:
            self.cleanup()

    def process_book(self, book_dir):
        from adapters.apple_vision_adapter import run as ocr_run
        from builder.epub_builder import build_epub
        from engine.formatter import run_pipeline

        book_dir = Path(book_dir)
        crop_manager = TempCropManager(base_dir=self.temp_root, name="novel_crops")
        with crop_manager.managed(book_dir.name) as crop_path:
            doc = ocr_run(
                image_folder=str(book_dir),
                verbose=True,
                preview_enabled=self.preview_enabled,
                temp_crop_dir=str(crop_path),
            )
            formatted = run_pipeline(doc, verbose=True)
            build_epub(
                formatted,
                output_path=str(self.output_dir / f"{book_dir.name}.epub"),
                verbose=True,
            )

    def cleanup(self) -> None:
        shutil.rmtree(self.temp_root, ignore_errors=True)
