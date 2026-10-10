from __future__ import annotations

from PySide6.QtWidgets import QLabel, QWidget, QSizePolicy
from collections import OrderedDict
from pathlib import Path

from PySide6.QtCore import Qt, QRect, QSize, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap, QFont, QFontMetrics

from ui.common.styling import INK, MUTED

LIGHT_IMAGE_PREVIEW_STYLE = f"""
background: #FFFFFF;
color: {MUTED};
border: 1px solid #E2E5E9;
border-radius: 10px;
padding: 12px;
"""

class OCRProofreadImageLabel(QLabel):
    """Aspect-ratio image viewer with drag-to-column scrub linkage."""

    column_scrubbed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source_pixmap: QPixmap | None = None
        # Sentence-review crops are revisited frequently while adjudicating.
        # Keep only a tiny, byte-bounded LRU so back/forward navigation does not
        # synchronously decode the same PNG again.  This is display-only cache;
        # OCR inputs and evidence files remain untouched on disk.
        self._pixmap_cache: OrderedDict[tuple[str, int, int], tuple[QPixmap, int]] = OrderedDict()
        self._pixmap_cache_bytes = 0
        self._pixmap_cache_limit_bytes = 64 * 1024 * 1024
        self._pixmap_cache_limit_items = 8
        self._last_scaled_size = QSize()
        self._scrub_position: float | None = None
        self._column_intervals: tuple[tuple[float, float], ...] = ()
        self._active_column_index = -1
        self._scrub_dragging = False
        self._smooth_scale_timer = QTimer(self)
        self._smooth_scale_timer.setSingleShot(True)
        self._smooth_scale_timer.setInterval(70)
        self._smooth_scale_timer.timeout.connect(lambda: self._rescale(smooth=True))
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(360, 360)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMouseTracking(True)
        self.setWordWrap(True)
        self.setStyleSheet(LIGHT_IMAGE_PREVIEW_STYLE)
        self.setText("完成 OCR 后，右侧显示当前句对应的单列或多列原图。")
        self.setToolTip("按住鼠标左键在图片上横向拖动，可联动定位旁边的 OCR 物理列。")

    @staticmethod
    def _pixmap_cost(pixmap: QPixmap) -> int:
        return max(0, int(pixmap.width())) * max(0, int(pixmap.height())) * 4

    def _cached_pixmap(self, path: str) -> QPixmap:
        source = str(path or "")
        if not source:
            return QPixmap()
        try:
            stat = Path(source).stat()
            key = (source, int(stat.st_size), int(stat.st_mtime_ns))
        except OSError:
            key = (source, -1, -1)
        cached = self._pixmap_cache.get(key)
        if cached is not None:
            self._pixmap_cache.move_to_end(key)
            return cached[0]
        pixmap = QPixmap(source)
        if pixmap.isNull():
            return pixmap
        cost = self._pixmap_cost(pixmap)
        if cost <= self._pixmap_cache_limit_bytes:
            self._pixmap_cache[key] = (pixmap, cost)
            self._pixmap_cache.move_to_end(key)
            self._pixmap_cache_bytes += cost
            while (
                len(self._pixmap_cache) > self._pixmap_cache_limit_items
                or self._pixmap_cache_bytes > self._pixmap_cache_limit_bytes
            ):
                _old_key, (_old_pixmap, old_cost) = self._pixmap_cache.popitem(last=False)
                self._pixmap_cache_bytes = max(0, self._pixmap_cache_bytes - int(old_cost))
        return pixmap

    def set_image_path(self, path: str) -> bool:
        pixmap = self._cached_pixmap(str(path or ""))
        if pixmap.isNull():
            self._source_pixmap = None
            self.clear()
            self.setText("当前句没有可用的原图裁片。")
            return False
        self._source_pixmap = pixmap
        self._last_scaled_size = QSize()
        self._scrub_position = None
        self._active_column_index = -1
        self.setText("")
        # Fast first paint keeps next/previous navigation immediate. One smooth
        # pass follows after resize/navigation events have settled.
        self._rescale(smooth=False)
        self._smooth_scale_timer.start()
        return True

    def clear_image(self, message: str = "尚未载入图文校对内容") -> None:
        self._smooth_scale_timer.stop()
        self._source_pixmap = None
        self._last_scaled_size = QSize()
        self._scrub_position = None
        self._column_intervals = ()
        self._active_column_index = -1
        self._scrub_dragging = False
        self.clear()
        self.setText(message)

    def set_scrub_position(self, value: float | None) -> None:
        if value is None:
            self._scrub_position = None
        else:
            self._scrub_position = max(0.0, min(1.0, float(value)))
        self.update()

    def set_column_intervals(self, intervals) -> None:
        values: list[tuple[float, float]] = []
        for raw in intervals or ():
            if not isinstance(raw, (list, tuple)) or len(raw) < 2:
                continue
            try:
                left = max(0.0, min(1.0, float(raw[0])))
                right = max(left, min(1.0, float(raw[1])))
            except (TypeError, ValueError, OverflowError):
                continue
            if right > left:
                values.append((left, right))
        self._column_intervals = tuple(values)
        if self._active_column_index >= len(self._column_intervals):
            self._active_column_index = -1
        self.update()

    def set_active_column(self, index: int) -> None:
        value = int(index)
        if value < 0 or value >= len(self._column_intervals):
            value = -1
        self._active_column_index = value
        self.update()

    def _displayed_pixmap_rect(self) -> QRect:
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            return QRect()
        content = self.contentsRect()
        width = min(content.width(), pixmap.width())
        height = min(content.height(), pixmap.height())
        # Alignment is centered, so account for letterboxing before converting
        # the pointer position to a normalized source-image x coordinate.
        left = content.left() + max(0, (content.width() - width) // 2)
        top = content.top() + max(0, (content.height() - height) // 2)
        return QRect(left, top, width, height)

    def _emit_scrub_from_event(self, event) -> None:
        rect = self._displayed_pixmap_rect()
        if rect.isNull() or rect.width() <= 1:
            return
        try:
            x = float(event.position().x())
        except AttributeError:  # pragma: no cover - Qt5 compatibility style
            x = float(event.pos().x())
        normalized = (x - rect.left()) / max(1.0, float(rect.width()))
        normalized = max(0.0, min(1.0, normalized))
        self._scrub_position = normalized
        self.column_scrubbed.emit(normalized)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._source_pixmap is not None:
            self._scrub_dragging = True
            self.setCursor(Qt.SizeHorCursor)
            self._emit_scrub_from_event(event)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._scrub_dragging:
            self._emit_scrub_from_event(event)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._scrub_dragging and event.button() == Qt.LeftButton:
            self._emit_scrub_from_event(event)
            self._scrub_dragging = False
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        rect = self._displayed_pixmap_rect()
        if rect.isNull():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        if 0 <= self._active_column_index < len(self._column_intervals):
            left_ratio, right_ratio = self._column_intervals[self._active_column_index]
            band_left = rect.left() + round(left_ratio * rect.width())
            band_right = rect.left() + round(right_ratio * rect.width())
            band = QRect(
                band_left, rect.top() + 1, max(2, band_right - band_left), max(2, rect.height() - 2)
            )
            painter.fillRect(band, QColor(47, 107, 255, 18))
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor("#2F6BFF"), 2))
            painter.drawRoundedRect(band.adjusted(1, 1, -1, -1), 8, 8)
        elif self._scrub_position is not None:
            x = rect.left() + round(self._scrub_position * rect.width())
            painter.fillRect(
                QRect(max(rect.left(), x - 9), rect.top(), 18, rect.height()),
                QColor(37, 99, 235, 28),
            )
            painter.setPen(QPen(QColor("#2F6BFF"), 2))
            painter.drawLine(x, rect.top(), x, rect.bottom())
        painter.end()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return
        self._rescale(smooth=False)
        self._smooth_scale_timer.start()

    def _rescale(self, *, smooth: bool = False) -> None:
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return
        target = self.contentsRect().size()
        if target.width() <= 1 or target.height() <= 1:
            return
        if smooth and target == self._last_scaled_size:
            # The fast preview used the same dimensions; replace it once with a
            # smooth version. Repeated resize events are coalesced by the timer.
            pass
        mode = Qt.SmoothTransformation if smooth else Qt.FastTransformation
        self.setPixmap(self._source_pixmap.scaled(target, Qt.KeepAspectRatio, mode))
        self._last_scaled_size = QSize(target)


class OCRVerticalColumnTextWidget(QWidget):
    """Complete OCR columns in Japanese right-to-left reading order."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._columns = []
        self._active_column_index = -1
        self.setMinimumSize(200, 420)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setToolTip("完整裁决文字：物理列按右→左排列，可滚动查看全部文字。")

    def _layout_metrics(self):
        font = QFont(self.font())
        font.setPointSize(max(17, font.pointSize() + 4))
        metrics = QFontMetrics(font)
        char_step = max(29, metrics.height() + 6)
        column_step = max(58, metrics.horizontalAdvance("国") + 26)
        return font, metrics, char_step, column_step

    def set_columns(self, columns):
        self._columns = [str(v or "").replace("\r", "").replace("\n", "") for v in (columns or [])] or [""]
        self._active_column_index = -1
        self.setToolTip("完整裁决文字：物理列按右→左排列，可滚动查看全部文字。")
        _, _, char_step, column_step = self._layout_metrics()
        margin_y = 22
        max_length = max(map(len, self._columns))
        height = max(420, margin_y * 2 + max_length * char_step + 16)
        self.setMinimumSize(max(200, 24 + len(self._columns) * column_step), height)
        self.updateGeometry()
        self.update()

    def set_active_column(self, index):
        self._active_column_index = int(index) if 0 <= int(index) < len(self._columns) else -1
        self.update()

    def column_center_x(self, index):
        _, _, _, step = self._layout_metrics()
        total = len(self._columns) * step
        return round((self.width() + total) / 2 - (int(index) + 0.5) * step)

    def column_center_y(self, index):
        return 40

    def clear_columns(self, message="完成 OCR 后显示逐列文字"):
        self._columns = []
        self._active_column_index = -1
        self.setToolTip(message)
        self.setMinimumSize(200, 420)
        self.updateGeometry()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#FFFFFF"))
        if not self._columns:
            painter.setPen(QColor(MUTED))
            painter.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap, "完成 OCR 后显示逐列文字")
            return
        font, _, char_step, column_step = self._layout_metrics()
        title_font = QFont(self.font())
        title_font.setPointSize(10)
        for index, text in enumerate(self._columns):
            x = self.column_center_x(index)
            if index == self._active_column_index:
                painter.fillRect(QRect(x - column_step // 2 + 2, 4, column_step - 4, self.height() - 8), QColor("#F2F4F7"))
            painter.setPen(QColor(MUTED))
            painter.setFont(title_font)
            painter.drawText(QRect(x-column_step//2, 8, column_step, 22), Qt.AlignCenter, f"列 {index+1}")
            painter.setFont(font)
            painter.setPen(QColor(INK))
            for row, char in enumerate(text):
                rect = QRect(x - column_step // 2, 38 + row * char_step, column_step, char_step)
                if rect.intersects(event.rect()):
                    painter.drawText(rect, Qt.AlignCenter, char)
        painter.end()
