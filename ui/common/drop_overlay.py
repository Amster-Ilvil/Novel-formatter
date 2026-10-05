# -*- coding: utf-8 -*-
"""把文件拖进主窗口时盖在整个窗口上的提示层。"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from ui.common.interaction_logic import drop_overlay_text
from ui.common.toast import _translate
from ui.theme.tokens import ACC_FILL, ACC_TEXT


class DropOverlay(QFrame):
    def __init__(self, window: QWidget):
        super().__init__(window)
        self.setObjectName("nfDropOverlay")
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setStyleSheet(
            f"QFrame#nfDropOverlay{{background:rgba(228,238,255,225);border:3px dashed {ACC_FILL};border-radius:22px;}}")
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        self._label = QLabel()
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet(f"color:{ACC_TEXT};font-size:22px;font-weight:750;background:none;border:none;")
        hint = QLabel(_translate("图片 · PDF · 文件夹都可以"))
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet(f"color:{ACC_TEXT};font-size:13px;background:none;border:none;")
        lay.addWidget(self._label)
        lay.addWidget(hint)
        self.hide()

    def show_for(self, count: int) -> None:
        parent = self.parentWidget()
        if parent is not None:
            self.setGeometry(14, 14, max(0, parent.width() - 28), max(0, parent.height() - 28))
        self._label.setText(drop_overlay_text(count, _translate("松开鼠标，导入文件")))
        self.show()
        self.raise_()
