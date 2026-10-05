from __future__ import annotations

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage

class WorkerSignals(QObject):
    finished = Signal(object)
    error = Signal(str)
    log = Signal(str)
    progress = Signal(int, int)
    page_ready = Signal(str, int, int)
    phase_progress = Signal(str, int, int)
    overall_progress = Signal(object)
    current_image_data = Signal(QImage)
    current_column_image_data = Signal(QImage, object)
    preview_page_ready = Signal(str, str, str, object, str)


class ImageReviewRenderSignals(QObject):
    """Queued bridge for sentence-crop rendering performed off the GUI thread."""

    finished = Signal(int, str, str)
    error = Signal(int, str, str)



def safe_qt_emit(bound_signal, *args) -> bool:
    """Emit a Qt signal unless its QObject source was already destroyed.

    Background probes may legitimately finish after their owning window has
    closed. PySide then raises ``RuntimeError: Signal source has been deleted``.
    Treat only that lifecycle case as a no-op; other RuntimeError instances
    remain visible so real signal misuse is not hidden.
    """
    try:
        bound_signal.emit(*args)
        return True
    except RuntimeError as exc:
        if "signal source has been deleted" in str(exc).lower():
            return False
        raise
