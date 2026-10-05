# -*- coding: utf-8 -*-
"""Persist main-window geometry and splitter ratios without changing business state."""
from __future__ import annotations

from PySide6.QtCore import QSettings, QTimer
from PySide6.QtGui import QGuiApplication

# Keep these aligned with SystemSettingsTab so Reset/portable settings stay coherent.
_ORG = "NovelFormatter"
_APP = "NovelFormatter1"


def _settings() -> QSettings:
    return QSettings(_ORG, _APP)


def _ensure_visible(window) -> None:
    """Move a restored window back onto a live screen after monitor changes."""
    try:
        frame = window.frameGeometry()
        screens = list(QGuiApplication.screens())
        if any(screen.availableGeometry().intersects(frame) for screen in screens):
            return
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        avail = screen.availableGeometry()
        x = avail.x() + max(0, (avail.width() - window.width()) // 2)
        y = avail.y() + max(0, (avail.height() - window.height()) // 2)
        window.move(x, y)
    except Exception:
        pass


def restore_window_state(window) -> bool:
    try:
        data = _settings().value("window/geometry")
        restored = bool(data) and bool(window.restoreGeometry(data))
        if restored:
            QTimer.singleShot(0, lambda: _ensure_visible(window))
        return restored
    except Exception:
        return False


def save_window_state(window) -> None:
    try:
        if not window.isMaximized() and not window.isFullScreen():
            settings = _settings()
            settings.setValue("window/geometry", window.saveGeometry())
            settings.sync()
    except Exception:
        pass


def install_window_state_saver(app, window) -> None:
    app.aboutToQuit.connect(lambda: save_window_state(window))


def bind_splitter(splitter, key: str) -> None:
    """Restore and persist user-adjusted splitter sizes."""
    try:
        settings = _settings()
        saved = settings.value(f"splitter/{key}")
        if isinstance(saved, str):
            saved = [part for part in saved.split(",") if part.strip()]
        sizes = [int(v) for v in saved] if saved else []
        if len(sizes) == splitter.count() and all(v > 0 for v in sizes):
            QTimer.singleShot(0, lambda values=list(sizes): splitter.setSizes(values))

        def persist(*_args):
            try:
                current = [int(v) for v in splitter.sizes()]
                if len(current) == splitter.count() and all(v >= 0 for v in current):
                    settings.setValue(f"splitter/{key}", current)
            except Exception:
                pass

        splitter.splitterMoved.connect(persist)
    except Exception:
        pass
