from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QDialog, QLabel, QPushButton, QScrollArea, QVBoxLayout, QHBoxLayout
from PySide6.QtCore import Qt, QPoint, QTimer
from PySide6.QtGui import QCursor, QPixmap, QImageReader

from ui.common.styling import MUTED, EDITOR_SCROLLBAR_STYLE

class _PagePreviewImageLabel(QLabel):
    """Full-resolution page viewer with wheel zoom and hand panning."""

    def __init__(self, dialog, parent=None):
        super().__init__(parent)
        self._dialog = dialog
        self._drag_origin: QPoint | None = None
        self._scroll_origin: tuple[int, int] | None = None
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#FFFFFF;border:none;")
        self.setCursor(QCursor(Qt.OpenHandCursor))
        self.setMouseTracking(True)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta:
            self._dialog.zoom_by(1.18 if delta > 0 else 1 / 1.18)
            event.accept()
            return
        super().wheelEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dialog.toggle_fit_100()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            scroll = self._dialog._scroll
            self._drag_origin = event.globalPosition().toPoint()
            self._scroll_origin = (
                scroll.horizontalScrollBar().value(),
                scroll.verticalScrollBar().value(),
            )
            self.setCursor(QCursor(Qt.ClosedHandCursor))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_origin is not None and self._scroll_origin is not None:
            point = event.globalPosition().toPoint()
            delta = point - self._drag_origin
            scroll = self._dialog._scroll
            scroll.horizontalScrollBar().setValue(self._scroll_origin[0] - delta.x())
            scroll.verticalScrollBar().setValue(self._scroll_origin[1] - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._drag_origin is not None:
            self._drag_origin = None
            self._scroll_origin = None
            self.setCursor(QCursor(Qt.OpenHandCursor))
            event.accept()
            return
        super().mouseReleaseEvent(event)


class PageImagePreviewDialog(QDialog):
    """Lazy one-page-at-a-time full-resolution preview for page management."""

    def __init__(self, paths, page_index: int = 0, parent=None):
        super().__init__(parent)
        self._paths = [Path(value) for value in (paths or [])]
        self._index = max(0, min(int(page_index), max(0, len(self._paths) - 1)))
        self._original = QPixmap()
        self._scale = 1.0
        self._fit_mode = True
        self.setWindowTitle("页面高清预览")
        self.resize(1100, 820)
        self.setMinimumSize(680, 520)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(7)

        toolbar = QHBoxLayout()
        self._previous_btn = QPushButton("← 上一页")
        self._previous_btn.clicked.connect(self.previous_page)
        toolbar.addWidget(self._previous_btn)
        self._next_btn = QPushButton("下一页 →")
        self._next_btn.clicked.connect(self.next_page)
        toolbar.addWidget(self._next_btn)
        self._fit_btn = QPushButton("适应窗口")
        self._fit_btn.clicked.connect(self.fit_to_window)
        toolbar.addWidget(self._fit_btn)
        self._actual_btn = QPushButton("100%")
        self._actual_btn.clicked.connect(self.actual_size)
        toolbar.addWidget(self._actual_btn)
        for button in (self._previous_btn, self._next_btn, self._fit_btn, self._actual_btn):
            button.setAutoDefault(False)
            button.setDefault(False)
        toolbar.addStretch(1)
        self._info = QLabel("")
        self._info.setStyleSheet(f"color:{MUTED};font-size:11px;")
        toolbar.addWidget(self._info)
        root.addLayout(toolbar)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(False)
        self._scroll.setAlignment(Qt.AlignCenter)
        self._scroll.setStyleSheet("QScrollArea{background:#FFFFFF;border:1px solid #E2E5E9;border-radius:8px;}" + EDITOR_SCROLLBAR_STYLE)
        self._image = _PagePreviewImageLabel(self)
        self._scroll.setWidget(self._image)
        root.addWidget(self._scroll, 1)

        hint = QLabel("滚轮缩放 · 拖动平移 · 双击切换适应窗口/100% · ←/→ 翻页 · Esc 关闭")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet(f"color:{MUTED};font-size:10px;")
        root.addWidget(hint)
        self._load_current()

    def _read_pixmap(self, path: Path) -> QPixmap:
        reader = QImageReader(str(path))
        reader.setAutoTransform(True)
        image = reader.read()
        if image.isNull():
            return QPixmap()
        return QPixmap.fromImage(image)

    def _load_current(self):
        if not self._paths:
            self._original = QPixmap()
            self._image.setText("没有可预览页面")
            return
        path = self._paths[self._index]
        self._original = self._read_pixmap(path)
        if self._original.isNull():
            self._image.setText(f"无法读取图片\n{path}")
            self._image.adjustSize()
        else:
            self.fit_to_window()
        self._previous_btn.setEnabled(self._index > 0)
        self._next_btn.setEnabled(self._index + 1 < len(self._paths))
        size_text = (
            f"{self._original.width()} × {self._original.height()} px"
            if not self._original.isNull() else "读取失败"
        )
        self._info.setText(
            f"第 {self._index + 1}/{len(self._paths)} 页 · {size_text} · {path.name}"
        )
        self.setWindowTitle(f"页面高清预览 · 第 {self._index + 1} 页 · {path.name}")

    def _apply_scale(self):
        if self._original.isNull():
            return
        self._scale = max(0.05, min(8.0, float(self._scale)))
        width = max(1, round(self._original.width() * self._scale))
        height = max(1, round(self._original.height() * self._scale))
        mode = Qt.SmoothTransformation if self._scale < 1.0 else Qt.FastTransformation
        pixmap = self._original.scaled(width, height, Qt.KeepAspectRatio, mode)
        self._image.setPixmap(pixmap)
        self._image.resize(pixmap.size())
        self._fit_btn.setText(f"适应窗口 · {round(self._scale * 100)}%")

    def fit_to_window(self):
        if self._original.isNull():
            return
        viewport = self._scroll.viewport().size()
        available_w = max(40, viewport.width() - 18)
        available_h = max(40, viewport.height() - 18)
        self._scale = min(
            available_w / max(1, self._original.width()),
            available_h / max(1, self._original.height()),
        )
        self._fit_mode = True
        self._apply_scale()

    def actual_size(self):
        self._scale = 1.0
        self._fit_mode = False
        self._apply_scale()

    def toggle_fit_100(self):
        if self._fit_mode:
            self.actual_size()
        else:
            self.fit_to_window()

    def zoom_by(self, factor: float):
        if self._original.isNull():
            return
        self._fit_mode = False
        self._scale *= float(factor)
        self._apply_scale()

    def previous_page(self):
        if self._index > 0:
            self._index -= 1
            self._load_current()

    def next_page(self):
        if self._index + 1 < len(self._paths):
            self._index += 1
            self._load_current()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Left, Qt.Key_Up, Qt.Key_PageUp):
            self.previous_page(); event.accept(); return
        if event.key() in (Qt.Key_Right, Qt.Key_Down, Qt.Key_PageDown, Qt.Key_Space):
            self.next_page(); event.accept(); return
        if event.key() == Qt.Key_Escape:
            self.reject(); return
        if event.key() in (Qt.Key_0, Qt.Key_F):
            self.fit_to_window(); event.accept(); return
        if event.key() == Qt.Key_1:
            self.actual_size(); event.accept(); return
        super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit_mode and not self._original.isNull():
            QTimer.singleShot(0, self.fit_to_window)

