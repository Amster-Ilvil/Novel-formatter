# -*- coding: utf-8 -*-
"""Lazy global command palette for existing Novel Formatter actions."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QLineEdit, QListWidget, QListWidgetItem, QLabel,
)

from core.command_catalog import CommandSpec, rank_command_specs
from ui.common.styling import BORDER, CARD, INK, MUTED, SURFACE
from ui.interface_preferences import manager_for
from ui.localization import LANG_ZH, translate_text


@dataclass(frozen=True, slots=True)
class CommandAction:
    spec: CommandSpec
    callback: Callable[[], None]
    enabled: bool = True
    disabled_reason: str = ""


class CommandPalette(QDialog):
    """Searchable index over already-owned application actions.

    The dialog is constructed lazily on the first explicit request.  Opening it
    never initializes OCR models, loads large project payloads or starts work.
    """

    def __init__(self, provider: Callable[[], Iterable[CommandAction]], parent=None):
        super().__init__(parent)
        self._provider = provider
        self._actions: dict[str, CommandAction] = {}
        self.setWindowTitle("命令面板")
        self.setModal(False)
        self.resize(620, 430)
        self.setMinimumSize(500, 320)
        self.setStyleSheet(
            f"QDialog{{background:{SURFACE};}}"
            f"QLineEdit,QListWidget{{background:{CARD};color:{INK};border:1px solid {BORDER};border-radius:8px;}}"
            "QLineEdit{padding:9px 11px;font-size:13px;}"
            "QListWidget{padding:5px;}"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 12)
        root.setSpacing(9)
        self._query = QLineEdit()
        self._query.setPlaceholderText("输入命令，例如 OCR、裁决、EPUB、项目…")
        self._query.setClearButtonEnabled(True)
        root.addWidget(self._query)
        self._list = QListWidget()
        self._list.setUniformItemSizes(True)
        root.addWidget(self._list, 1)
        self._hint = QLabel("↑↓ 选择 · Enter 执行 · Esc 关闭")
        self._hint.setStyleSheet(f"color:{MUTED};font-size:10px;")
        root.addWidget(self._hint)

        self._query.textChanged.connect(self._refresh_results)
        self._query.returnPressed.connect(self._activate_current)
        self._list.itemActivated.connect(self._activate_item)

    def open_palette(self) -> None:
        self._query.clear()
        self._refresh_results()
        self.show()
        self.raise_()
        self.activateWindow()
        self._query.setFocus(Qt.ShortcutFocusReason)

    def _load_actions(self) -> tuple[CommandAction, ...]:
        try:
            actions = tuple(self._provider() or ())
        except Exception as exc:
            self._actions = {}
            self._list.clear()
            self._hint.setText(f"命令列表加载失败：{exc}")
            return ()
        self._actions = {action.spec.command_id: action for action in actions}
        return actions

    def _refresh_results(self) -> None:
        actions = self._load_actions()
        by_id = {action.spec.command_id: action for action in actions}
        app = QApplication.instance()
        language = getattr(manager_for(app), "language", LANG_ZH) if app is not None else LANG_ZH
        display_specs = tuple(
            replace(
                action.spec,
                title=translate_text(action.spec.title, language),
                category=translate_text(action.spec.category, language),
                keywords=action.spec.keywords + (
                    action.spec.title, action.spec.category,
                    translate_text(action.spec.title, language),
                    translate_text(action.spec.category, language),
                ),
            )
            for action in actions
        )
        ranked = rank_command_specs(display_specs, self._query.text())
        self._list.clear()
        first_enabled_row = -1
        for spec in ranked:
            action = by_id.get(spec.command_id)
            if action is None:
                continue
            shortcut = f"    {spec.shortcut}" if spec.shortcut else ""
            subtitle = spec.category
            if not action.enabled and action.disabled_reason:
                subtitle = f"{subtitle} · {action.disabled_reason}" if subtitle else action.disabled_reason
            text = f"{spec.title}{shortcut}"
            if subtitle:
                text += f"\n{subtitle}"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, spec.command_id)
            item.setToolTip(action.disabled_reason or "")
            if not action.enabled:
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            self._list.addItem(item)
            if action.enabled and first_enabled_row < 0:
                first_enabled_row = self._list.count() - 1
        if self._list.count():
            self._list.setCurrentRow(first_enabled_row if first_enabled_row >= 0 else 0)
            self._hint.setText("↑↓ 选择 · Enter 执行 · Esc 关闭")
        else:
            self._hint.setText("没有匹配命令")

    def _move_selection(self, delta: int) -> None:
        """Move only across executable rows, skipping disabled commands.

        This mirrors mature IDE command palettes: keyboard navigation should
        never land on an action that Enter cannot execute.
        """
        count = self._list.count()
        if count <= 0:
            return
        row = self._list.currentRow()
        if row < 0:
            row = 0 if delta >= 0 else count - 1
        for step in range(1, count + 1):
            candidate = (row + (step * (1 if delta >= 0 else -1))) % count
            item = self._list.item(candidate)
            if item is not None and bool(item.flags() & Qt.ItemIsEnabled):
                self._list.setCurrentRow(candidate)
                return

    def _activate_current(self) -> None:
        item = self._list.currentItem()
        if item is not None:
            self._activate_item(item)

    def _activate_item(self, item: QListWidgetItem) -> None:
        command_id = str(item.data(Qt.UserRole) or "")
        action = self._actions.get(command_id)
        if action is None or not action.enabled:
            return
        owner = self.parentWidget()
        self.hide()
        action.callback()
        # A non-modal palette can remain the active/focus window after hide on
        # some Qt platforms.  Explicitly return focus so the owner's WindowShortcut
        # actions work immediately without requiring an extra mouse click.
        if owner is not None:
            owner.activateWindow()
            owner.setFocus(Qt.ShortcutFocusReason)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self.hide()
            event.accept()
            return
        if key in (Qt.Key_Down, Qt.Key_Up):
            self._move_selection(1 if key == Qt.Key_Down else -1)
            event.accept()
            return
        super().keyPressEvent(event)


__all__ = ["CommandAction", "CommandPalette"]
