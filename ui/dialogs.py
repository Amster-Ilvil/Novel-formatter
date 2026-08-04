# -*- coding: utf-8 -*-
"""Centralised message-box styling, localisation, and exception reporting.

Qt's native ``QMessageBox`` creates its detail editor and the details toggle
lazily.  Styling only the application-level ``QPushButton`` selector therefore
leaves some platforms with low-contrast text or English ``OK/Details`` labels.
This module polishes every message box when it is shown and keeps all popup
behaviour in one place.
"""
from __future__ import annotations

import sys
import threading
import traceback
from collections.abc import Callable
from typing import Optional

from PySide6.QtCore import QEvent, QObject, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTextEdit,
    QWidget,
)


_DIALOG_STYLE = """
QMessageBox {
    background-color: #FFFFFF;
}
QMessageBox QLabel {
    color: #1D1D1F;
    background: transparent;
    min-width: 280px;
}
QMessageBox QPushButton {
    min-width: 76px;
    min-height: 30px;
    padding: 4px 14px;
    border-radius: 7px;
    border: 1px solid #0071E3;
    background-color: #0071E3;
    color: #FFFFFF;
    font-weight: 600;
}
QMessageBox QPushButton:hover {
    background-color: #0077ED;
    border-color: #0077ED;
}
QMessageBox QPushButton:pressed {
    background-color: #0058B8;
    border-color: #0058B8;
}
QMessageBox QPushButton[dialogRole="secondary"] {
    background-color: #ECEEF3;
    color: #1D1D1F;
    border-color: #C7C7CC;
}
QMessageBox QPushButton[dialogRole="secondary"]:hover {
    background-color: #F2F2F7;
    border-color: #AFAFB5;
}
QMessageBox QPushButton[dialogRole="details"] {
    background-color: #F2F6FC;
    color: #005DBA;
    border-color: #B9D5F2;
    min-width: 92px;
}
QMessageBox QPushButton[dialogRole="details"]:hover {
    background-color: #E7F0FA;
    border-color: #8DBBE8;
}
QMessageBox QTextEdit,
QMessageBox QPlainTextEdit {
    background-color: #F7F8FA;
    color: #242428;
    border: 1px solid #D6D8DE;
    border-radius: 7px;
    padding: 8px;
    selection-background-color: #DCE7FF;
    selection-color: #1D1D1F;
}
"""


_STANDARD_TEXT = {
    QMessageBox.Ok: "确定",
    QMessageBox.Save: "保存",
    QMessageBox.SaveAll: "全部保存",
    QMessageBox.Open: "打开",
    QMessageBox.Yes: "是",
    QMessageBox.YesToAll: "全部是",
    QMessageBox.No: "否",
    QMessageBox.NoToAll: "全部否",
    QMessageBox.Abort: "中止",
    QMessageBox.Retry: "重试",
    QMessageBox.Ignore: "忽略",
    QMessageBox.Close: "关闭",
    QMessageBox.Cancel: "取消",
    QMessageBox.Discard: "放弃",
    QMessageBox.Help: "帮助",
    QMessageBox.Apply: "应用",
    QMessageBox.Reset: "重置",
    QMessageBox.RestoreDefaults: "恢复默认",
}

_SECONDARY_BUTTONS = {
    QMessageBox.No,
    QMessageBox.NoToAll,
    QMessageBox.Cancel,
    QMessageBox.Close,
    QMessageBox.Discard,
    QMessageBox.Ignore,
    QMessageBox.Help,
    QMessageBox.Reset,
    QMessageBox.RestoreDefaults,
}


def _is_details_button(button: QAbstractButton) -> bool:
    text = button.text().replace("&", "").strip().lower()
    return "detail" in text or "详情" in text


def _detail_editors(box: QMessageBox) -> list[QWidget]:
    # QObject.findChildren accepts one Qt meta-type at a time in PySide6.
    return [*box.findChildren(QTextEdit), *box.findChildren(QPlainTextEdit)]


def _details_are_visible(box: QMessageBox) -> bool:
    return any(editor.isVisible() for editor in _detail_editors(box))


class DialogPolishFilter(QObject):
    """Apply consistent, high-contrast styling to all ``QMessageBox`` objects."""

    _EVENTS = {
        QEvent.Type.Show,
        QEvent.Type.Polish,
    }

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 (Qt API)
        if isinstance(watched, QMessageBox) and event.type() in self._EVENTS:
            # Some child widgets are created during QMessageBox::showEvent.
            # Run once now and once after the current event has completed.
            self.polish(watched)
            QTimer.singleShot(0, lambda box=watched: self._safe_polish(box))
        return False

    def _safe_polish(self, box: QMessageBox) -> None:
        try:
            self.polish(box)
        except RuntimeError:
            # The C++ dialog may already have been deleted before the queued call.
            return

    def polish(self, box: QMessageBox) -> None:
        if not isinstance(box, QMessageBox):
            return

        # Do not rewrite the same stylesheet during every polish event.  On
        # macOS, setStyleSheet()/unpolish()/polish() emits LayoutRequest events;
        # listening to those events and styling again creates an event storm.
        # The dialog still looks responsive, but its standard button may appear
        # to ignore clicks because the close event is starved by repaint/layout
        # work.  Keep the operation idempotent and only react to Show/Polish.
        if not box.property("nfDialogStyled"):
            box.setProperty("nfDialogStyled", True)
            box.setStyleSheet(_DIALOG_STYLE)
        if box.minimumWidth() < 420:
            box.setMinimumWidth(420)

        for label in box.findChildren(QLabel):
            if not label.wordWrap():
                label.setWordWrap(True)
            flags = label.textInteractionFlags() | Qt.TextSelectableByMouse
            if label.textInteractionFlags() != flags:
                label.setTextInteractionFlags(flags)

        fixed_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        fixed_font.setPointSize(max(10, fixed_font.pointSize()))
        for editor in _detail_editors(box):
            if not editor.isReadOnly():
                editor.setReadOnly(True)
            if editor.font() != fixed_font:
                editor.setFont(fixed_font)
            if editor.minimumWidth() < 520 or editor.minimumHeight() < 180:
                editor.setMinimumSize(520, 180)
            if not editor.property("nfDetailStyled"):
                editor.setProperty("nfDetailStyled", True)
                editor.setStyleSheet(
                    "background:#F7F8FA;color:#242428;border:1px solid #D6D8DE;"
                    "border-radius:7px;padding:8px;selection-background-color:#DCE7FF;"
                    "selection-color:#1D1D1F;"
                )

        for button in box.findChildren(QPushButton):
            standard = box.standardButton(button)
            if standard in _STANDARD_TEXT:
                target_text = _STANDARD_TEXT[standard]
                if button.text() != target_text:
                    button.setText(target_text)
                role = "secondary" if standard in _SECONDARY_BUTTONS else "primary"
                if button.property("dialogRole") != role:
                    button.setProperty("dialogRole", role)
                    button.style().unpolish(button)
                    button.style().polish(button)

                # QMessageBox normally closes itself through an internal Qt
                # connection.  Add a one-shot safety connection so a platform
                # style/plugin cannot leave a standard closing button inert.
                # ``done(StandardButton)`` preserves the result returned by
                # exec()/question()/warning().
                if standard in _CLOSING_STANDARD_BUTTONS and not button.property("nfCloseHooked"):
                    button.setProperty("nfCloseHooked", True)
                    button.clicked.connect(
                        lambda _checked=False, target=box, code=standard: _finish_message_box(
                            target, code
                        )
                    )
                continue

            if _is_details_button(button) or button.property("dialogRole") == "details":
                if button.property("dialogRole") != "details":
                    button.setProperty("dialogRole", "details")
                    button.style().unpolish(button)
                    button.style().polish(button)
                target_text = "隐藏详情" if _details_are_visible(box) else "显示详情"
                if button.text() != target_text:
                    button.setText(target_text)
                if not button.property("nfDetailsHooked"):
                    button.setProperty("nfDetailsHooked", True)
                    button.clicked.connect(
                        lambda _checked=False, target=box: QTimer.singleShot(
                            0, lambda: self._safe_polish(target)
                        )
                    )


_CLOSING_STANDARD_BUTTONS = {
    QMessageBox.Ok,
    QMessageBox.Save,
    QMessageBox.SaveAll,
    QMessageBox.Open,
    QMessageBox.Yes,
    QMessageBox.YesToAll,
    QMessageBox.No,
    QMessageBox.NoToAll,
    QMessageBox.Abort,
    QMessageBox.Retry,
    QMessageBox.Ignore,
    QMessageBox.Close,
    QMessageBox.Cancel,
    QMessageBox.Discard,
}


def _finish_message_box(box: QMessageBox, code: QMessageBox.StandardButton) -> None:
    """Close a live message box without reopening or recursively polishing it."""
    try:
        if box.isVisible():
            box.done(int(code))
    except RuntimeError:
        # The native handler may already have destroyed the C++ object.
        return


def install_dialog_polish(app: QApplication) -> DialogPolishFilter:
    """Install one idempotent popup polisher on the application."""
    existing = getattr(app, "_novel_formatter_dialog_filter", None)
    if isinstance(existing, DialogPolishFilter):
        return existing
    dialog_filter = DialogPolishFilter(app)
    app.installEventFilter(dialog_filter)
    app._novel_formatter_dialog_filter = dialog_filter
    return dialog_filter


def show_error_dialog(
    parent: Optional[QWidget],
    title: str,
    traceback_text: str,
    *,
    summary: str = "",
) -> None:
    """Show a readable error summary with a copyable full traceback."""
    details = str(traceback_text or "").strip()
    last_line = details.splitlines()[-1] if details else "未知错误"
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Critical)
    box.setWindowTitle(title or "程序错误")
    box.setText(summary.strip() or last_line)
    if summary.strip() and last_line != summary.strip():
        box.setInformativeText(last_line)
    if details:
        box.setDetailedText(details)
    box.setStandardButtons(QMessageBox.Ok)
    box.exec()


class _ExceptionBridge(QObject):
    raised = Signal(str, str)

    def __init__(self, parent_provider: Optional[Callable[[], Optional[QWidget]]], parent=None):
        super().__init__(parent)
        self._parent_provider = parent_provider
        self.raised.connect(self._show, Qt.QueuedConnection)

    @Slot(str, str)
    def _show(self, title: str, details: str) -> None:
        parent = self._parent_provider() if self._parent_provider else None
        try:
            show_error_dialog(parent, title, details)
        except Exception:
            # Exception reporting must never create a recursive crash loop.
            print(details, file=sys.stderr)


def install_exception_hooks(
    app: QApplication,
    parent_provider: Optional[Callable[[], Optional[QWidget]]] = None,
) -> _ExceptionBridge:
    """Report otherwise-unhandled GUI and worker-thread exceptions visibly."""
    existing = getattr(app, "_novel_formatter_exception_bridge", None)
    if isinstance(existing, _ExceptionBridge):
        return existing

    bridge = _ExceptionBridge(parent_provider, app)
    app._novel_formatter_exception_bridge = bridge
    previous_sys_hook = sys.excepthook
    previous_thread_hook = getattr(threading, "excepthook", None)

    def format_exception(exc_type, exc_value, exc_tb) -> str:
        return "".join(traceback.format_exception(exc_type, exc_value, exc_tb))

    def sys_hook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
            previous_sys_hook(exc_type, exc_value, exc_tb)
            return
        details = format_exception(exc_type, exc_value, exc_tb)
        print(details, file=sys.stderr)
        bridge.raised.emit("未处理的程序错误", details)

    def thread_hook(args):
        if issubclass(args.exc_type, (KeyboardInterrupt, SystemExit)):
            if previous_thread_hook:
                previous_thread_hook(args)
            return
        details = format_exception(args.exc_type, args.exc_value, args.exc_traceback)
        thread_name = getattr(args.thread, "name", "后台任务") or "后台任务"
        print(details, file=sys.stderr)
        bridge.raised.emit(f"后台任务异常：{thread_name}", details)

    sys.excepthook = sys_hook
    if hasattr(threading, "excepthook"):
        threading.excepthook = thread_hook
    return bridge
