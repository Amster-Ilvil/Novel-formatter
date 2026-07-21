"""Manual responsive smoke test for Novel Formatter Studio.

Run with: python tests/manual_ui_responsive.py
The window cycles through common sizes so a developer can verify wrapping,
compact sidebar, white OCR/PDF logs, light OCR preview, swatches, and percent
progress displays.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from gui_pyside6 import MainWindow

SIZES = [(760, 560), (900, 620), (1100, 700), (1320, 840), (1600, 1000)]


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    index = {"value": 0}

    def apply_next_size():
        i = index["value"]
        if i >= len(SIZES):
            print("Manual responsive pass complete. Please close the window after inspection.")
            return
        width, height = SIZES[i]
        print(f"Checking {width}x{height}")
        window.resize(width, height)
        index["value"] = i + 1
        QTimer.singleShot(2000, apply_next_size)

    QTimer.singleShot(500, apply_next_size)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
