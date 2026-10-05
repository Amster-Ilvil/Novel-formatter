# -*- coding: utf-8 -*-
"""不打断操作的提示条：保存、导入、复制这类“告知一下就行”的消息不再弹模态窗口。

用法：``notify(self, "已保存：xxx", "success")``。找不到可见主窗口时自动退回到普通消息框，
文字走与消息框相同的离线本地化。
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtWidgets import QFrame, QGraphicsDropShadowEffect, QHBoxLayout, QLabel, QMessageBox, QWidget
from PySide6.QtGui import QColor

from ui.common.interaction_logic import toast_duration_ms, toast_origin
from ui.theme.tokens import ACC_TEXT, BORDER, CARD, INK, SUCCESS, WARN

_ICON = {"success": ("✓", SUCCESS), "warning": ("!", WARN), "info": ("i", ACC_TEXT)}


def _translate(text: str) -> str:
    try:
        from ui.dialog_localization import translate_dialog_text
        from ui.localized_dialogs import current_ui_language
        return translate_dialog_text(text, current_ui_language())
    except Exception:
        return text


class _Toast(QFrame):
    def __init__(self, window: QWidget):
        super().__init__(window)
        self.setObjectName("nfToast")
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setCursor(Qt.PointingHandCursor)
        row = QHBoxLayout(self)
        row.setContentsMargins(14, 10, 18, 10)
        row.setSpacing(10)
        self._icon = QLabel()
        self._icon.setFixedSize(22, 22)
        self._icon.setAlignment(Qt.AlignCenter)
        self._text = QLabel()
        self._text.setWordWrap(True)
        self._text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._text.setStyleSheet(f"color:{INK};font-size:13px;font-weight:600;background:none;")
        row.addWidget(self._icon)
        row.addWidget(self._text, 1)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(30, 60, 110, 70))
        self.setGraphicsEffect(shadow)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)
        window.installEventFilter(self)          # 窗口缩放时保持在底部居中
        self.hide()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Resize and self.isVisible():
            self._place()
        return False

    def _place(self) -> None:
        parent = self.parentWidget()
        if parent is not None:
            # Keep long paths/reports readable without creating a toast wider
            # than the work area. Short messages remain compact.
            max_text_width = max(280, min(720, parent.width() - 120))
            self._text.setMaximumWidth(max_text_width)
            self.adjustSize()
            self.move(*toast_origin(parent.width(), parent.height(), self.width(), self.height()))

    def show_message(self, text: str, kind: str, ms: int) -> None:
        glyph, color = _ICON.get(kind, _ICON["info"])
        self._icon.setText(glyph)
        self._icon.setStyleSheet(
            f"background:{color};color:#FFFFFF;border-radius:11px;font-size:12px;font-weight:800;")
        self._text.setText(text)
        self.setStyleSheet(f"QFrame#nfToast{{background:{CARD};border:1px solid {BORDER};border-radius:14px;}}")
        self._place()
        self.show()
        self.raise_()
        self._timer.start(ms)

    def mousePressEvent(self, event) -> None:       # 点一下就收起
        self.hide()


def notify(widget, text: str, kind: str = "success", ms: int | None = None) -> None:
    """显示一条自动消失的提示。kind: success / info / warning。"""
    text = _translate(str(text))
    window = widget.window() if widget is not None else None
    if window is None or not window.isVisible():
        QMessageBox.information(widget, "提示", text)
        return
    toast = getattr(window, "_nf_toast", None)
    if toast is None:
        toast = _Toast(window)
        window._nf_toast = toast
    toast.show_message(text, kind, int(ms or toast_duration_ms(text, kind)))
