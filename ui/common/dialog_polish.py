# -*- coding: utf-8 -*-
"""所有对话框统一的细节处理，不需要逐个窗口改代码：
1. 最小尺寸不超过屏幕可用区域的 92%（小屏幕不会出现比屏幕还大的窗口）；
2. 没有输入框的窗口，主操作按钮（保存/开始/应用/确定）自动成为默认按钮，回车即可确认。
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QAbstractSpinBox, QDialog, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QTextEdit

from ui.common.interaction_logic import clamp_dialog_min, pick_default_button

_SKIP = ("CommandPalette",)


def polish_dialog(dialog: QDialog) -> None:
    if isinstance(dialog, QMessageBox) or type(dialog).__name__ in _SKIP:
        return
    dialog.setProperty("nfPolished", True)
    try:
        screen = dialog.screen() or QGuiApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            new_w, new_h = clamp_dialog_min(dialog.minimumWidth(), dialog.minimumHeight(), avail.width(), avail.height())
            if (new_w, new_h) != (dialog.minimumWidth(), dialog.minimumHeight()):
                dialog.setMinimumSize(new_w, new_h)
                dialog.resize(min(dialog.width(), int(avail.width() * 0.96)), min(dialog.height(), int(avail.height() * 0.96)))
        # Clear Qt's implicit default/auto-default state on *every* button first,
        # including temporarily disabled buttons. A disabled editor action can
        # otherwise retain an implicit default and become dangerous when it is
        # enabled later (for example FormatProfileDialog's Delete button).
        every_button = list(dialog.findChildren(QPushButton))
        for button in every_button:
            button.setDefault(False)
            button.setAutoDefault(False)

        enabled_buttons = [b for b in every_button if b.isEnabled()]
        visible_buttons = [b for b in enabled_buttons if b.isVisible()]
        buttons = visible_buttons or enabled_buttons
        def is_real_line_input(widget):
            if not widget.isEnabled() or widget.isReadOnly():
                return False
            parent = widget.parentWidget()
            while parent is not None and parent is not dialog:
                if isinstance(parent, QAbstractSpinBox):
                    return False
                parent = parent.parentWidget()
            return True
        line_inputs = [w for w in dialog.findChildren(QLineEdit) if is_real_line_input(w)]
        plain_inputs = [w for w in dialog.findChildren(QPlainTextEdit) if w.isEnabled() and not w.isReadOnly()]
        rich_inputs = [w for w in dialog.findChildren(QTextEdit) if w.isEnabled() and not w.isReadOnly()]
        has_input = bool(line_inputs or plain_inputs or rich_inputs)
        index = pick_default_button([b.text() for b in buttons], has_input)

        # Only an explicit primary action is allowed to become the Enter key
        # target. Editing dialogs intentionally get no default action.
        if index is not None:
            buttons[index].setAutoDefault(True)
            buttons[index].setDefault(True)
    except Exception:
        pass            # 细节美化永远不能影响对话框本身


class _DialogPolishFilter(QObject):
    def eventFilter(self, obj, event):
        event_type = event.type()
        if event_type == QEvent.Show and isinstance(obj, QDialog):
            polish_dialog(obj)
            # Qt may assign automatic defaults and some dialogs create platform
            # action buttons lazily after showEvent. Re-apply across two queued
            # cycles so late buttons follow the same primary-action policy.
            QTimer.singleShot(0, lambda target=obj: self._polish_twice(target))
        elif event_type in (QEvent.Show, QEvent.EnabledChange) and isinstance(obj, QPushButton):
            try:
                dialog = obj.window()
            except RuntimeError:
                dialog = None
            if isinstance(dialog, QDialog) and not isinstance(dialog, QMessageBox):
                QTimer.singleShot(0, lambda target=dialog: self._safe_polish(target))
        return False

    def _polish_twice(self, dialog):
        self._safe_polish(dialog)
        QTimer.singleShot(0, lambda target=dialog: self._safe_polish(target))

    @staticmethod
    def _safe_polish(dialog):
        try:
            polish_dialog(dialog)
        except RuntimeError:
            pass


def install_dialog_polish(app) -> None:
    flt = _DialogPolishFilter(app)
    app.installEventFilter(flt)
    app._nf_dialog_polish = flt          # 防止被回收
