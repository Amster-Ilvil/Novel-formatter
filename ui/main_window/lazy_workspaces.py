# -*- coding: utf-8 -*-
"""Small, deterministic lazy-workspace registry for the Qt composition root.

The registry deliberately owns *construction only*.  Business state remains in
MainWindow/controllers and the real workspace widgets, so deferred UI creation
cannot become a second document model or task scheduler.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from PySide6.QtWidgets import QApplication, QWidget

T = TypeVar("T", bound=QWidget)


@dataclass(slots=True)
class _Entry(Generic[T]):
    factory: Callable[[], T]
    installer: Callable[[T], None]
    instance: T | None = None
    constructing: bool = False


class LazyWorkspaceRegistry:
    """Create heavy workspaces once, on first real use.

    ``installer`` is called exactly once after successful construction and is
    responsible for replacing the lightweight placeholder, wiring signals and
    registering the workspace with the coordinator.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}

    def register(self, key: str, factory: Callable[[], T], installer: Callable[[T], None]) -> None:
        name = str(key or "").strip()
        if not name:
            raise ValueError("lazy workspace key cannot be empty")
        if name in self._entries:
            raise ValueError(f"lazy workspace already registered: {name}")
        self._entries[name] = _Entry(factory=factory, installer=installer)

    def get(self, key: str) -> QWidget:
        name = str(key or "").strip()
        try:
            entry = self._entries[name]
        except KeyError as exc:
            raise KeyError(f"unknown lazy workspace: {name}") from exc
        if entry.instance is not None:
            return entry.instance
        if entry.constructing:
            raise RuntimeError(f"recursive lazy workspace construction: {name}")
        entry.constructing = True
        try:
            # The application-wide clickable guard intentionally avoids widget
            # construction on macOS/PySide: reacting to Polish while private
            # scrollbars are still being built can cache a temporary base-class
            # wrapper. Eager startup used to avoid this by installing the guard
            # afterwards; deferred pages preserve the same safety boundary.
            app = QApplication.instance()
            previous_guard_state = app.property("nfSuspendClickableGuard") if app is not None else None
            if app is not None:
                app.setProperty("nfSuspendClickableGuard", True)
            try:
                widget = entry.factory()
            finally:
                if app is not None:
                    app.setProperty("nfSuspendClickableGuard", previous_guard_state)
            if not isinstance(widget, QWidget):
                raise TypeError(f"lazy workspace {name!r} did not create QWidget")
            entry.installer(widget)
            entry.instance = widget
            return widget
        finally:
            entry.constructing = False

    def loaded(self, key: str) -> QWidget | None:
        entry = self._entries.get(str(key or "").strip())
        return None if entry is None else entry.instance

    def is_loaded(self, key: str) -> bool:
        return self.loaded(key) is not None

    def loaded_items(self) -> tuple[tuple[str, QWidget], ...]:
        return tuple(
            (key, entry.instance)
            for key, entry in self._entries.items()
            if entry.instance is not None
        )

    def loaded_widgets(self) -> tuple[QWidget, ...]:
        return tuple(widget for _key, widget in self.loaded_items())
