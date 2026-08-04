# -*- coding: utf-8 -*-
"""Runtime contrast guard for buttons and other clickable controls.

The interface intentionally uses white secondary buttons and coloured primary
buttons.  Qt/macOS can occasionally keep a primary button's white text while
falling back to a white native background, most visibly when the control is
disabled.  This module applies a small, idempotent contrast safeguard to every
button created at startup or later by a dialog/workspace.
"""
from __future__ import annotations

import re
from typing import Final

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QLabel,
    QPushButton,
    QTabBar,
    QWidget,
)

_FORBIDDEN_BACKGROUND: Final[re.Pattern[str]] = re.compile(
    r"(?i)(background(?:-color)?\s*:\s*)"
    r"(?:transparent\b|rgba?\(\s*255\s*,\s*255\s*,\s*255\s*,\s*0(?:\.0+)?\s*\))"
)
_LIGHT_TEXT: Final[re.Pattern[str]] = re.compile(
    r"(?i)(color\s*:\s*)(?:white\b|#fff(?:fff)?\b|rgb\(\s*255\s*,\s*255\s*,\s*255\s*\))"
)
_EXPLICIT_TEXT_COLOR: Final[re.Pattern[str]] = re.compile(r"(?i)\bcolor\s*:")

_CLICKABLE_BG: Final[str] = "#FFFFFF"
_CLICKABLE_LABEL_BG: Final[str] = "#F7F9FC"
_DARK_TEXT: Final[str] = "#202733"
_DISABLED_TEXT: Final[str] = "#475467"
_NORMAL_MARKER: Final[str] = "nf-secondary-button-contrast"
_PRIMARY_MARKER: Final[str] = "nf-primary-disabled-contrast"


def _is_clickable_text_label(widget: QLabel) -> bool:
    try:
        flags = widget.textInteractionFlags()
        return bool(
            widget.openExternalLinks()
            or flags & Qt.TextInteractionFlag.LinksAccessibleByMouse
            or flags & Qt.TextInteractionFlag.LinksAccessibleByKeyboard
            or widget.property("nfClickable")
        )
    except RuntimeError:
        return False


def _replacement(widget: QWidget) -> str:
    if isinstance(widget, QLabel):
        return _CLICKABLE_LABEL_BG
    return _CLICKABLE_BG


def _sanitize_local_stylesheet(widget: QWidget) -> None:
    """Replace transparent local backgrounds without touching layout or text."""
    if widget.property("nfNoWhiteGuardBusy"):
        return
    try:
        current = widget.styleSheet() or ""
    except RuntimeError:
        return
    if not current or not _FORBIDDEN_BACKGROUND.search(current):
        return
    updated = _FORBIDDEN_BACKGROUND.sub(lambda m: m.group(1) + _replacement(widget), current)
    if updated == current:
        return
    try:
        widget.setProperty("nfNoWhiteGuardBusy", True)
        widget.setStyleSheet(updated)
    finally:
        try:
            widget.setProperty("nfNoWhiteGuardBusy", False)
        except RuntimeError:
            pass


def _set_button_palette(button: QPushButton, enabled_text: str, disabled_text: str) -> None:
    """Set palette fallbacks for platform styles that ignore part of QSS."""
    try:
        palette = button.palette()
        for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive):
            palette.setColor(group, QPalette.ColorRole.ButtonText, QColor(enabled_text))
            palette.setColor(group, QPalette.ColorRole.WindowText, QColor(enabled_text))
        palette.setColor(
            QPalette.ColorGroup.Disabled,
            QPalette.ColorRole.ButtonText,
            QColor(disabled_text),
        )
        palette.setColor(
            QPalette.ColorGroup.Disabled,
            QPalette.ColorRole.WindowText,
            QColor(disabled_text),
        )
        button.setPalette(palette)
    except RuntimeError:
        return


def _ensure_push_button_contrast(button: QPushButton) -> None:
    """Make a QPushButton readable without changing its signal or geometry."""
    if button.property("nfNoWhiteGuardBusy"):
        return
    try:
        role = str(button.property("role") or "")
        dialog_role = str(button.property("dialogRole") or "")
        current = button.styleSheet() or ""
    except RuntimeError:
        return

    # QMessageBox buttons have their own complete style and role handling.
    if dialog_role:
        return

    if role == "primary":
        # Enabled primary text remains white.  Disabled primary controls use
        # dark text so they stay readable even if a platform drops the fill.
        _set_button_palette(button, "#FFFFFF", _DISABLED_TEXT)
        if _PRIMARY_MARKER in current or "nf-primary-button-contrast" in current:
            return
        appendix = f"""
/* {_PRIMARY_MARKER} */
QPushButton:disabled {{ color: {_DISABLED_TEXT}; background-color: #EAF3FF; border-color: #C5DCF7; }}
"""
    elif role == "danger":
        _set_button_palette(button, "#B42318", "#7A271A")
        return
    else:
        _set_button_palette(button, _DARK_TEXT, _DISABLED_TEXT)
        if _NORMAL_MARKER in current:
            return
        # A non-primary white button must never retain a local white text rule.
        cleaned = _LIGHT_TEXT.sub(lambda m: m.group(1) + _DARK_TEXT, current)
        # Preserve deliberately coloured text (danger/link buttons).  When no
        # local text colour exists, add an explicit dark fallback so the native
        # macOS palette cannot turn it white.
        if _EXPLICIT_TEXT_COLOR.search(cleaned):
            if cleaned != current:
                appendix = f"\n/* {_NORMAL_MARKER} */\nQPushButton:disabled {{ color: {_DISABLED_TEXT}; }}\n"
                current = cleaned
            else:
                return
        else:
            current = cleaned
            appendix = f"""
/* {_NORMAL_MARKER} */
QPushButton {{ color: {_DARK_TEXT}; }}
QPushButton:hover, QPushButton:pressed {{ color: {_DARK_TEXT}; }}
QPushButton:disabled {{ color: {_DISABLED_TEXT}; }}
"""

    try:
        button.setProperty("nfNoWhiteGuardBusy", True)
        button.setStyleSheet(current + appendix)
    finally:
        try:
            button.setProperty("nfNoWhiteGuardBusy", False)
        except RuntimeError:
            pass


def enforce_button_contrast_tree(root: QWidget) -> None:
    """Apply the contrast rule to every existing QPushButton below ``root``."""
    if isinstance(root, QPushButton):
        _ensure_push_button_contrast(root)
    for button in root.findChildren(QPushButton):
        _ensure_push_button_contrast(button)


class NoWhiteClickableGuard(QObject):
    """Polish and protect clickable controls created at runtime."""

    _EVENTS = {
        QEvent.Type.Show,
        QEvent.Type.Polish,
        QEvent.Type.StyleChange,
        QEvent.Type.DynamicPropertyChange,
        QEvent.Type.EnabledChange,
    }

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() not in self._EVENTS or not isinstance(watched, QWidget):
            return False

        if isinstance(watched, QAbstractButton):
            if watched.property("nfNoWhiteClickable") is not True:
                watched.setProperty("nfNoWhiteClickable", True)
            _sanitize_local_stylesheet(watched)
            if isinstance(watched, QPushButton):
                _ensure_push_button_contrast(watched)
        elif isinstance(watched, QTabBar):
            if watched.property("nfNoWhiteClickable") is not True:
                watched.setProperty("nfNoWhiteClickable", True)
            _sanitize_local_stylesheet(watched)
        elif isinstance(watched, QLabel) and _is_clickable_text_label(watched):
            if watched.property("nfClickable") is not True:
                watched.setProperty("nfClickable", True)
            _sanitize_local_stylesheet(watched)

        # Child controls of complex dialogs may be created at the end of the
        # current event.  Queue a second pass, but only for the watched widget;
        # this avoids expensive whole-window rescans and event storms.
        if event.type() in {QEvent.Type.Show, QEvent.Type.Polish}:
            QTimer.singleShot(0, lambda w=watched: self._safe_sanitize(w))
        return False

    @staticmethod
    def _safe_sanitize(widget: QWidget) -> None:
        try:
            if isinstance(widget, (QAbstractButton, QTabBar)):
                _sanitize_local_stylesheet(widget)
                if isinstance(widget, QPushButton):
                    _ensure_push_button_contrast(widget)
            elif isinstance(widget, QLabel) and _is_clickable_text_label(widget):
                widget.setProperty("nfClickable", True)
                _sanitize_local_stylesheet(widget)
        except RuntimeError:
            return


def install_no_white_clickable_guard(app: QApplication) -> NoWhiteClickableGuard:
    """Install exactly one application-wide clickable-background guard."""
    existing = getattr(app, "_nf_no_white_clickable_guard", None)
    if isinstance(existing, NoWhiteClickableGuard):
        return existing
    guard = NoWhiteClickableGuard(app)
    app.installEventFilter(guard)
    # Keep both a QObject parent and a Python reference.  Some PySide builds can
    # otherwise collect a filter that is only referenced by C++.
    app._nf_no_white_clickable_guard = guard
    return guard
