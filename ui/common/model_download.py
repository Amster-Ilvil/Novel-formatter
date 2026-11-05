# -*- coding: utf-8 -*-
"""Background model download with a cancellable progress dialog.

The heavy lifting is :meth:`utils.model_manager.ModelManager.ensure`; this
module only runs it off the GUI thread, shows ``MB / total MB`` progress with a
Cancel button, and reports :class:`ModelError` messages in a dialog and the log
without letting exceptions escape into Qt.
"""
from __future__ import annotations

import contextlib
import logging
import threading
import time
from pathlib import Path
from typing import Callable, Iterable

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QProgressDialog, QWidget

from ui.localization import translate_text
from ui.localized_dialogs import LocalizedMessageBox, current_ui_language
from utils.model_manager import (
    ModelCancelledError,
    ModelError,
    ModelStatus,
    default_manager,
    get_spec,
    user_requested_download,
)

LOGGER = logging.getLogger(__name__)
Requirement = tuple[str, "Path | None"]

_ACTIVE: set["_ModelDownloadSignals"] = set()


def _tr(text: str) -> str:
    return translate_text(text, current_ui_language())


class _ModelDownloadSignals(QObject):
    progress = Signal(str, object, object)
    finished = Signal()
    failed = Signal(str)
    cancelled = Signal()


def pending_models(requirements: Iterable[Requirement]) -> list[Requirement]:
    """Return requirements whose status is MISSING or CORRUPT (call off the GUI thread)."""
    manager = default_manager()
    pending: list[Requirement] = []
    for model_id, target_dir in requirements:
        if manager.status(model_id, target_dir=target_dir) is not ModelStatus.READY:
            pending.append((model_id, target_dir))
    return pending


def _format_progress(label: str, done: int, total: int | None) -> str:
    done_mb = max(0, int(done)) / (1024 * 1024)
    if total:
        return f"{label}\n{done_mb:.1f} / {int(total) / (1024 * 1024):.1f} MB"
    return f"{label}\n{done_mb:.1f} MB"


def ensure_models_in_background(
    parent: QWidget,
    requirements: Iterable[Requirement],
    *,
    on_ready: Callable[[], None],
    explicit: bool = False,
    log: Callable[[str], None] | None = None,
) -> None:
    """Download/verify ``requirements`` in a worker thread, then call ``on_ready``.

    ``explicit=True`` marks the downloads as user-requested so they run even
    when automatic downloads are disabled in Settings.
    """
    items = list(requirements)
    if not items:
        on_ready()
        return

    cancel_event = threading.Event()
    signals = _ModelDownloadSignals()
    _ACTIVE.add(signals)

    dialog = QProgressDialog(parent)
    dialog.setWindowTitle(_tr("下载 OCR 模型"))
    dialog.setLabelText(_tr("正在准备模型下载…"))
    dialog.setCancelButtonText(_tr("取消"))
    dialog.setRange(0, 0)
    dialog.setMinimumDuration(0)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    dialog.setWindowModality(Qt.WindowModal)
    dialog.canceled.connect(cancel_event.set)
    dialog.show()

    def emit_log(message: str) -> None:
        LOGGER.info("%s", message)
        if log is not None:
            try:
                log(message)
            except Exception:
                LOGGER.debug("model download log callback failed", exc_info=True)

    def release() -> None:
        _ACTIVE.discard(signals)
        try:
            dialog.close()
            dialog.deleteLater()
        except RuntimeError:
            pass

    def on_progress(label: str, done, total) -> None:
        try:
            if total:
                dialog.setRange(0, 1000)
                dialog.setValue(min(1000, int(int(done) * 1000 / max(1, int(total)))))
            else:
                dialog.setRange(0, 0)
            dialog.setLabelText(_format_progress(label, int(done or 0), int(total) if total else None))
        except RuntimeError:
            pass

    def on_finished() -> None:
        release()
        emit_log(_tr("模型已下载并通过校验。"))
        on_ready()

    def on_failed(message: str) -> None:
        release()
        emit_log(f"{_tr('模型下载失败')}: {message}")
        LocalizedMessageBox.critical(parent, _tr("模型下载失败"), message)

    def on_cancelled() -> None:
        release()
        emit_log(_tr("模型下载已取消。"))

    signals.progress.connect(on_progress)
    signals.finished.connect(on_finished)
    signals.failed.connect(on_failed)
    signals.cancelled.connect(on_cancelled)

    downloading = _tr("正在下载模型")

    def worker() -> None:
        manager = default_manager()
        scope = user_requested_download() if explicit else contextlib.nullcontext()
        try:
            with scope:
                for model_id, target_dir in items:
                    label = f"{downloading}: {get_spec(model_id).display_name}"
                    last_emit = 0.0

                    def progress(done: int, total: int | None, _label: str = label) -> None:
                        nonlocal last_emit
                        now = time.monotonic()
                        if now - last_emit >= 0.1 or (total and done >= total):
                            last_emit = now
                            signals.progress.emit(_label, done, total)

                    signals.progress.emit(label, 0, None)
                    manager.ensure(
                        model_id,
                        target_dir=target_dir,
                        progress_callback=progress,
                        cancel_check=cancel_event.is_set,
                    )
        except ModelCancelledError:
            signals.cancelled.emit()
            return
        except ModelError as exc:
            signals.failed.emit(str(exc))
            return
        except Exception as exc:  # never let a worker exception reach Qt
            LOGGER.exception("unexpected model download failure")
            signals.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        signals.finished.emit()

    threading.Thread(target=worker, daemon=True, name="model-download").start()


__all__ = ["ensure_models_in_background", "pending_models"]
