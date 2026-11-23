"""Read-only source-evidence viewer for OCR proofreading.

Pure Qt / PySide6: no browser runtime or extra wheels. The displayed image
may be zoomed, but the original file and stable OCR row are never changed.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout,
)

from ui.ocr.review_button_style import style_review_button


class _ZoomScrollArea(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._zoom_owner = None

    def wheelEvent(self, event):
        if (event.modifiers() & Qt.ControlModifier) and self._zoom_owner:
            steps = event.angleDelta().y() / 120.0
            if steps:
                self._zoom_owner.zoom_by(1.18 ** steps)
                event.accept()
                return
        super().wheelEvent(event)


class ReferenceZoomDialog(QDialog):
    """Fit / 100% / wheel zoom with scrollable original image evidence."""

    def __init__(self, image_path: str, parent=None, *, normalized_bbox=None):
        super().__init__(parent)
        self.setWindowTitle('OCR 原图证据 · 缩放查看')
        self.resize(940, 820)
        self.setMinimumSize(520, 460)
        self._pixmap = QPixmap(str(image_path))
        if normalized_bbox is not None and not self._pixmap.isNull():
            # Only crop this temporary, in-memory display pixmap.  The source
            # page, original OCR input, and evidence remain byte-identical.
            try:
                x, y, w, h = (float(v) for v in normalized_bbox)
                from math import isfinite
                if all(isfinite(v) for v in (x, y, w, h)) and w > 0 and h > 0:
                    margin_x, margin_y = max(0.008, w * 0.12), max(0.008, h * 0.035)
                    left = max(0, round((x - margin_x) * self._pixmap.width()))
                    top = max(0, round((y - margin_y) * self._pixmap.height()))
                    right = min(self._pixmap.width(), round((x + w + margin_x) * self._pixmap.width()))
                    bottom = min(self._pixmap.height(), round((y + h + margin_y) * self._pixmap.height()))
                    if right > left and bottom > top:
                        self._pixmap = self._pixmap.copy(left, top, right - left, bottom - top)
                        self.setWindowTitle('OCR 原图证据 · 仅看所选物理列')
            except (ValueError, TypeError, OverflowError):
                pass
        self._zoom = 1.0
        self._fit_mode = True

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(9)
        header = QHBoxLayout()
        header.setSpacing(7)
        self._description = QLabel(self)
        self._description.setText(
            f'原图 · {self._pixmap.width()} × {self._pixmap.height()} px'
            if not self._pixmap.isNull() else '无法打开原图'
        )
        self._description.setStyleSheet('font-weight:600;font-size:12px;color:#34445B;')
        header.addWidget(self._description, 1)
        self._fit_btn = QPushButton('适应窗口', self)
        self._actual_btn = QPushButton('100%', self)
        self._out_btn = QPushButton('−', self)
        self._in_btn = QPushButton('+', self)
        self._zoom_status = QLabel('100%', self)
        self._zoom_status.setMinimumWidth(48)
        self._zoom_status.setAlignment(Qt.AlignCenter)
        for btn in (self._fit_btn, self._actual_btn, self._out_btn, self._in_btn):
            btn.setMinimumHeight(31)
            btn.setStyleSheet("QPushButton{border-radius:7px;padding:4px 10px;}")
            style_review_button(btn)
            header.addWidget(btn)
        header.addWidget(self._zoom_status)
        root.addLayout(header)

        self._scroll = _ZoomScrollArea(self)
        self._scroll._zoom_owner = self
        self._scroll.setWidgetResizable(False)
        self._scroll.setAlignment(Qt.AlignCenter)
        self._scroll.setStyleSheet('QScrollArea{border:1px solid #DCE6F2;background:#F3F6FB;border-radius:7px;}')
        self._image = QLabel(self._scroll)
        self._image.setAlignment(Qt.AlignCenter)
        self._scroll.setWidget(self._image)
        root.addWidget(self._scroll, 1)

        footer = QLabel('Ctrl + 滚轮缩放 · 鼠标滚轮移动视图 · 图片只读，原始 OCR 不会修改', self)
        footer.setStyleSheet('color:#687A92;font-size:11px;')
        root.addWidget(footer)
        self._fit_btn.clicked.connect(self.fit_to_window)
        self._actual_btn.clicked.connect(lambda: self.set_zoom(1.0))
        self._in_btn.clicked.connect(lambda: self.zoom_by(1.25))
        self._out_btn.clicked.connect(lambda: self.zoom_by(0.8))
        for btn in (self._fit_btn, self._actual_btn, self._in_btn, self._out_btn):
            btn.setEnabled(not self._pixmap.isNull())
        if self._pixmap.isNull():
            self._image.setText(f'无法打开：{Path(image_path).name}')
        else:
            QTimer.singleShot(0, self.fit_to_window)

    def fit_to_window(self):
        if self._pixmap.isNull():
            return
        area = self._scroll.viewport().size()
        self._fit_mode = True
        factor = min(
            max(1.0, area.width() - 12) / self._pixmap.width(),
            max(1.0, area.height() - 12) / self._pixmap.height(),
            1.0,
        )
        self.set_zoom(factor, fit=True)

    def zoom_by(self, factor: float):
        self.set_zoom(self._zoom * float(factor))

    def set_zoom(self, zoom: float, *, fit: bool = False):
        if self._pixmap.isNull():
            return
        self._fit_mode = bool(fit)
        # Preserve reasonable memory usage when reading exceptionally large
        # evidence pages: large originals are limited to ~24M rendered pixels.
        max_scale = min(4.0, (24_000_000 / max(1, self._pixmap.width() * self._pixmap.height())) ** 0.5)
        self._zoom = min(max(0.10, float(zoom)), max(0.10, max_scale))
        width = max(1, int(self._pixmap.width() * self._zoom))
        height = max(1, int(self._pixmap.height() * self._zoom))
        transformed = self._pixmap.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._image.setPixmap(transformed)
        self._image.resize(transformed.size())
        self._zoom_status.setText(f'{self._zoom * 100:.0f}%')

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit_mode:
            QTimer.singleShot(0, self.fit_to_window)
