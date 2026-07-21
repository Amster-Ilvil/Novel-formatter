from __future__ import annotations
from PySide6.QtWidgets import QFrame, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar

PAGE_TYPE_COLORS = {
    "cover": "#F26B4A", "color_illustration": "#EC6F9E", "color_illus": "#EC6F9E",
    "blank": "#D2D2D7", "toc": "#4F8FEF", "toc_page": "#4F8FEF",
    "illustration": "#32A47C", "text": "#6C63D8", "paragraph": "#6C63D8",
    "afterword": "#C68A2D", "copyright": "#7D7D83", "colophon": "#7D7D83", "unknown": "#B8B8BD",
}

class ColorSwatch(QFrame):
    def __init__(self, color: str, parent=None):
        super().__init__(parent)
        self.setFixedSize(14, 14)
        self.setStyleSheet(f"QFrame {{ background: {color}; border: 1px solid rgba(0, 0, 0, 28); border-radius: 4px; }}")

class ProgressStatusWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.label = QLabel("等待开始")
        self.percent = QLabel("0%")
        self.bar = QProgressBar()
        header = QHBoxLayout(); header.addWidget(self.label); header.addStretch(); header.addWidget(self.percent)
        layout = QVBoxLayout(self); layout.setContentsMargins(0,0,0,0); layout.setSpacing(6); layout.addLayout(header); layout.addWidget(self.bar)
    def set_progress(self, current: int, total: int, message: str = ""):
        total = max(total, 1); percent = round(current * 100 / total)
        self.bar.setRange(0, total); self.bar.setValue(current); self.percent.setText(f"{percent}%"); self.label.setText(message or f"{current} / {total}")
    def set_error(self, message="处理失败"):
        self.label.setText(message); self.label.setProperty("status", "error")
