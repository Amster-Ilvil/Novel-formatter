"""Keep cyclic Qt wrapper destruction on the QApplication thread.

CPython can start automatic cyclic collection in any allocating worker thread.
A cycle containing a QWidget must be collected by the GUI thread, especially
when Shiboken and Python 3.14 run alongside native painting callbacks.
"""
from __future__ import annotations

import gc
from time import monotonic


def mark_gui_navigation(app) -> None:
    if app is not None:
        app._novel_gui_last_navigation = monotonic()


def install_gui_gc():
    from PySide6.QtCore import QThread, QTimer
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None or QThread.currentThread() != app.thread():
        return None
    existing = getattr(app, '_novel_gui_gc_timer', None)
    if existing is not None:
        return existing

    gc.disable()
    timer = QTimer(app)
    timer.setInterval(5000)
    last_full_collection = monotonic()

    def collect_pending():
        nonlocal last_full_collection
        now = monotonic()
        last_navigation = getattr(app, "_novel_gui_last_navigation", 0.0)
        if now - last_navigation < 2:
            return
        if now - last_full_collection >= 60:
            gc.collect()
            last_full_collection = now
            return
        if gc.get_count()[0] >= gc.get_threshold()[0]:
            gc.collect(0)

    timer.timeout.connect(collect_pending)
    app._novel_gui_gc_timer = timer
    timer.start()
    return timer
