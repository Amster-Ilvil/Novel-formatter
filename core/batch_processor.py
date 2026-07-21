from pathlib import Path
from core.temp_manager import TempCropManager
import traceback
import shutil
import tempfile

class BatchProcessor:
    """批量 OCR/EPUB 调度器。每个子目录视为一本书。"""

    def __init__(self, input_dir, output_dir, preview_enabled=True):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.preview_enabled = preview_enabled
        self.temp_root = Path(tempfile.mkdtemp(prefix="novel_ocr_batch_"))

    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for book in sorted(p for p in self.input_dir.iterdir() if p.is_dir()):
            try:
                self.process_book(book)
            except Exception:
                print(f"[FAILED] {book.name}")
                traceback.print_exc()

    def process_book(self, book_dir):
        from adapters.apple_vision_adapter import run as ocr_run
        from engine.formatter import run_pipeline
        from builder.epub_builder import build_epub

        crop_manager = TempCropManager(base_dir=self.temp_root, name="novel_crops")
        crop_path = crop_manager.create(book_dir.name)
        try:
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
        finally:
            crop_manager.cleanup(crop_path)


    def cleanup(self):
        shutil.rmtree(self.temp_root, ignore_errors=True)
