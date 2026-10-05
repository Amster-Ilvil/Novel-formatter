# -*- coding: utf-8 -*-
"""整本进度条：每格一句（或一段），按状态着色，点击跳转。"""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QToolTip, QWidget

from ui.ocr.review_flow import bucket_of, bucket_states, index_of_bucket
from ui.theme.tokens import ACC, BORDER_STRONG, DARK_PAIRS, SUCCESS, TONAL_HOVER, WARN

_COLORS = {"reviewed": SUCCESS, "changed": ACC, "judge": WARN, "pending": TONAL_HOVER}
_LABELS = {"reviewed": "已核对", "changed": "已修改", "judge": "需要判断", "pending": "未核对"}


class SentenceStrip(QWidget):
    jump = Signal(int)          # 请求跳转到原始序号

    def __init__(self, parent=None):
        super().__init__(parent)
        self._states: list[str] = []
        self._current = -1
        self._hover = -1
        self.setFixedHeight(18)
        self.setMinimumWidth(120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)
        self.setCursor(QCursor(Qt.PointingHandCursor))

    def set_states(self, states, current: int) -> None:
        states = list(states)
        if states != self._states or current != self._current:
            self._states, self._current = states, int(current)
            self.update()

    def _slots(self) -> int:
        return max(1, min(len(self._states), self.width() // 5))

    def _dark(self) -> bool:
        return self.palette().window().color().lightness() < 128

    def _color(self, hex_value: str) -> QColor:
        return QColor(DARK_PAIRS.get(hex_value, hex_value) if self._dark() else hex_value)

    def paintEvent(self, _event) -> None:
        if not self._states:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        buckets = bucket_states(self._states, self._slots())
        gap, width = 1.5, self.width() / len(buckets)
        cur = bucket_of(self._current, len(self._states), len(buckets)) if self._current >= 0 else -1
        for i, state in enumerate(buckets):
            rect = QRectF(i * width + gap / 2, 4, max(1.0, width - gap), 10)
            painter.setPen(Qt.NoPen)
            painter.setBrush(self._color(_COLORS.get(state, TONAL_HOVER)))
            painter.drawRoundedRect(rect, 2, 2)
            if i == cur:
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(self._color(BORDER_STRONG if not self._dark() else "#FFFFFF"), 1.6))
                painter.drawRoundedRect(rect.adjusted(-1, -2, 1, 2), 3, 3)

    def _bucket_at(self, x: float) -> int:
        n = len(bucket_states(self._states, self._slots()))
        return max(0, min(n - 1, int(x / max(1.0, self.width() / max(1, n))))) if n else -1

    def mousePressEvent(self, event) -> None:
        if self._states and event.button() == Qt.LeftButton:
            self.jump.emit(index_of_bucket(self._bucket_at(event.position().x()), len(self._states), self._slots()))

    def mouseMoveEvent(self, event) -> None:
        if not self._states:
            return
        b = self._bucket_at(event.position().x())
        if b != self._hover:
            self._hover = b
            idx = index_of_bucket(b, len(self._states), self._slots())
            QToolTip.showText(event.globalPosition().toPoint(), f"第 {idx + 1} 句 · {_LABELS.get(self._states[idx], '')}", self)
