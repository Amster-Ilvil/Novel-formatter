# -*- coding: utf-8 -*-
"""Qt coordinator for independent workspaces.

The coordinator owns only lightweight lifecycle state. Business documents stay
inside their workspaces/snapshot registry, while activation and stable-row
navigation are routed through one fault-isolated boundary. A broken or already
deleted workspace therefore cannot prevent the remaining areas from switching.
"""
from __future__ import annotations

import logging
from typing import Iterable

from PySide6.QtCore import QObject, QTimer, Signal

logger = logging.getLogger(__name__)


class WorkspaceCoordinator(QObject):
    stable_row_changed = Signal(int, str)
    activation_changed = Signal(str, bool)
    workspace_error = Signal(str, str)

    def __init__(self, parent=None, *, row_interval_ms: int = 16):
        super().__init__(parent)
        self._widgets: dict[str, object] = {}
        self._active: set[str] = set()
        self._pending_row: tuple[int, str] | None = None
        self._last_emitted_row: int | None = None
        self._shutdown = False
        self._row_timer = QTimer(self)
        self._row_timer.setSingleShot(True)
        self._row_timer.setInterval(min(1000, max(0, int(row_interval_ms))))
        self._row_timer.timeout.connect(self._flush_row)

    @staticmethod
    def _normalise_key(key: object) -> str:
        value = str(key or "").strip()
        if not value:
            raise ValueError("workspace key cannot be empty")
        return value

    def register(self, key: str, widget: object) -> None:
        if self._shutdown:
            return
        workspace_key = self._normalise_key(key)
        if widget is None:
            raise ValueError(f"workspace {workspace_key!r} cannot be None")
        previous = self._widgets.get(workspace_key)
        if previous is widget:
            return
        if previous is not None:
            self.unregister(workspace_key)
        self._widgets[workspace_key] = widget

        destroyed = getattr(widget, "destroyed", None)
        connector = getattr(destroyed, "connect", None)
        if callable(connector):
            try:
                connector(lambda *_args, _key=workspace_key: self.unregister(_key))
            except Exception:
                logger.debug("Could not attach workspace destruction hook: %s", workspace_key, exc_info=True)

    def unregister(self, key: str) -> object | None:
        try:
            workspace_key = self._normalise_key(key)
        except ValueError:
            return None
        widget = self._widgets.pop(workspace_key, None)
        self._active.discard(workspace_key)
        return widget

    def registered_keys(self) -> tuple[str, ...]:
        return tuple(self._widgets)

    def active_keys(self) -> tuple[str, ...]:
        return tuple(key for key in self._widgets if key in self._active)

    def set_active(self, keys: Iterable[str]) -> None:
        if self._shutdown:
            return
        wanted: set[str] = set()
        for key in keys:
            try:
                wanted.add(self._normalise_key(key))
            except ValueError:
                continue

        for key, widget in tuple(self._widgets.items()):
            active = key in wanted
            was_active = key in self._active
            if active == was_active:
                continue
            setter = getattr(widget, "set_workspace_active", None)
            try:
                if callable(setter):
                    setter(active)
            except Exception as exc:
                # Keep the internal state unchanged when the target rejected the
                # transition. Other workspaces still receive their transitions.
                message = f"{type(exc).__name__}: {exc}"
                logger.warning("Workspace activation failed for %s: %s", key, message, exc_info=True)
                self.workspace_error.emit(key, message)
                continue

            if active:
                self._active.add(key)
            else:
                self._active.discard(key)
            self.activation_changed.emit(key, active)

    def publish_stable_row(self, row_index: int, source: str) -> None:
        if self._shutdown:
            return
        try:
            row = max(0, int(row_index))
        except (TypeError, ValueError, OverflowError):
            return
        self._pending_row = (row, str(source or "unknown"))
        if not self._row_timer.isActive():
            self._row_timer.start()

    def reset_navigation(self) -> None:
        """Allow the same row number to be emitted for a newly loaded session."""
        if self._row_timer.isActive():
            self._row_timer.stop()
        self._pending_row = None
        self._last_emitted_row = None

    def flush(self) -> None:
        if self._shutdown:
            return
        if self._row_timer.isActive():
            self._row_timer.stop()
        self._flush_row()

    def _flush_row(self) -> None:
        if self._shutdown:
            return
        pending = self._pending_row
        self._pending_row = None
        if pending is not None:
            row, source = pending
            if self._last_emitted_row == row:
                return
            self._last_emitted_row = row
            self.stable_row_changed.emit(row, source)

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self._row_timer.stop()
        self._pending_row = None
        self._last_emitted_row = None
        self._widgets.clear()
        self._active.clear()
