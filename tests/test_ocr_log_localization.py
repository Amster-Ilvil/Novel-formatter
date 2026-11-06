"""Smoke: the English OCR log carries no CJK UI text from the backend.

Runs the production column OCR path on a few pages with the deterministic
stub recognizer from tools/bench_pipeline.py.  Every adapter callback is
formatted the way ui/ocr/run_controller.py writes it to the OCR log and passed
through translate_dialog_text(..., LANG_EN), the function LocalizedLogView
applies on append.  Remaining CJK is allowed only when it quotes book text.
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT, ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from ui.dialog_localization import translate_dialog_text  # noqa: E402
from ui.localization import LANG_EN  # noqa: E402

_CJK_RUN = re.compile("[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff66-\uff9f]+")
TEXT_PAGES = 4
PHASE_LABEL = "模型1·Apple Vision·逐列识别"


def _render_pages(pdf: Path, target_dir: Path) -> list[str]:
    import pymupdf

    paths: list[str] = []
    with pymupdf.open(str(pdf)) as document:
        for index in range(1, 1 + TEXT_PAGES):
            target = target_dir / f"page_{index:03d}.png"
            document.load_page(index).get_pixmap(dpi=150).save(str(target))
            paths.append(str(target))
    return paths


class OcrLogLocalizationSmokeTest(unittest.TestCase):
    def test_english_ocr_log_has_no_backend_cjk(self) -> None:
        import bench_pipeline
        import adapters.column_ocr_adapter as column_ocr

        raw_lines: list[str] = []

        def phase(name, *args, **kwargs):
            values = list(args) + list(kwargs.values())
            numbers = [v for v in values if isinstance(v, int) and not isinstance(v, bool)]
            texts = [v for v in values if isinstance(v, str)]
            number_iter = iter(numbers)
            current = next(number_iter, 0)
            total = next(number_iter, 1)
            detail = next(iter(texts), "")
            if name == "recognition_work":
                raw_lines.append(f"  [{PHASE_LABEL}] {detail or '逐列识别'}")
            else:
                raw_lines.append(f"  [{PHASE_LABEL} {current:3d}/{max(total, 1)}] {detail}")
            raw_lines.extend(t for t in texts[1:] if t)

        def progress(current, total, filename, *_args, **_kwargs):
            raw_lines.append(f"  [{PHASE_LABEL} {int(current):3d}/{max(int(total), 1)}] {filename}")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pdf = bench_pipeline.DEFAULT_PDF
            if not pdf.exists():
                pdf = bench_pipeline.generate_pdf(tmp_path / "bench.pdf")
            pages = _render_pages(pdf, tmp_path)
            (tmp_path / "home").mkdir()
            restore = bench_pipeline.install_stub_recognizer()
            try:
                with mock.patch.dict(os.environ, {"HOME": str(tmp_path / "home")}):
                    doc = column_ocr.run(
                        recognition_engine="apple_vision", input_paths=pages, page_overrides={},
                        verbose=False, phase_callback=phase, progress_callback=progress,
                    )
            finally:
                restore()

        book_text = json.dumps(doc.to_dict(), ensure_ascii=False)
        book_text += "".join(bench_pipeline._STUB_SENTENCES) + bench_pipeline._STUB_KANA
        self.assertTrue(raw_lines, "the OCR run produced no log lines")

        leaks: list[str] = []
        for line in raw_lines:
            shown = translate_dialog_text(line, LANG_EN)
            leftover = [run for run in _CJK_RUN.findall(shown) if run not in book_text]
            if leftover:
                leaks.append(f"{line!r} -> {shown!r}: {leftover}")
        self.assertEqual(leaks, [], "CJK left in the English OCR log:\n" + "\n".join(leaks[:25]))


if __name__ == "__main__":
    unittest.main()
