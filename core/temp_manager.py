from pathlib import Path
import shutil
import tempfile


class TempCropManager:
    """统一管理 OCR 临时裁剪目录。"""

    def __init__(self, base_dir=None, name="ocr_crop"):
        self.base_dir = Path(base_dir) if base_dir else Path(tempfile.gettempdir())
        self.root = self.base_dir / "novel_formatter" / name

    def create(self, book_name="default"):
        path = self.root / book_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def cleanup(self, path=None):
        target = Path(path) if path else self.root
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
