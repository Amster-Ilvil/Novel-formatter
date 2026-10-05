#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""System settings UI and manual OCR model update dialog."""

from __future__ import annotations

import os
import platform
import re
import sys
import tempfile
import threading
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QSettings, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication, QWidget, QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QCheckBox, QComboBox, QLineEdit, QPlainTextEdit, QProgressBar,
    QFrame, QScrollArea, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView,
)

from core.app_meta import VERSION
from ui.common.editor_controls import NoWheelComboBox
from ui.common.signals import WorkerSignals
from ui.common.toast import notify
from ui.common.styling import LIGHT_LOG_STYLE, CARD, BORDER, INK, MUTED, accent_button, wrap_in_card
from ui.interface_preferences import THEME_LIGHT, THEME_DARK, manager_for
from ui.localization import LANG_ZH, LANG_JA, LANG_EN
from ui.localized_dialogs import LocalizedMessageBox, ui_message
from ui.navigation.sidebar import (
    REFERENCE_SECTION_ITEMS, SECTION_WORKSPACE, SECTION_PAGE, SECTION_SYSTEM,
)
from ui.ocr.catalog import OCR_ADAPTERS, adapter_available_on_current_platform
from ui.settings.ai_dialog import AISettingsDialog

QMessageBox = LocalizedMessageBox
PROJECT_ROOT = Path(__file__).resolve().parents[2]

class OCRModelUpdateDialog(QDialog):
    """Manual-only OCR model version checker and updater.

    Opening this dialog only reads local files. Network access starts exclusively
    after pressing “检查更新”, and model replacement starts exclusively after a
    second explicit “更新/修复” action and confirmation.
    """

    _STATE_TEXT = {
        "not_checked": "未检查",
        "not_installed": "未安装",
        "current": "已是最新",
        "current_compatible": "兼容版可用",
        "update_available": "有可用更新",
        "repair_required": "需要安装/修复",
        "compatibility_locked": "兼容性锁定",
        "upstream_differs_locked": "上游不同·禁止盲升",
        "system_managed": "系统托管",
        "cloud_managed": "云端托管",
        "check_error": "检查失败",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("OCR 模型手动更新")
        self.resize(980, 610)
        self.setMinimumSize(820, 520)
        self._statuses = {}
        self._busy = False
        self._checked_once = False
        self._active_signals = None
        self._build_ui()
        self._load_local_statuses()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 14)
        root.setSpacing(10)

        title = QLabel("OCR 模型更新（仅手动）")
        title.setStyleSheet("font-size:16px;font-weight:700;")
        root.addWidget(title)

        policy = QLabel(
            "本更新功能不会在启动时检查、不会后台轮询、不会自动下载更新包或替换已安装模型。"
            "打开本窗口只读取本地版本；必须点击“检查更新”，再选择模型并确认，才会联网更新。"
            "首次启用尚未安装的 OCR 引擎仍沿用原有的用户触发安装流程。"
            + ("Windows 下载优先使用系统 BITS，失败后回退 curl.exe 和 Python HTTPS。" if os.name == "nt" else "macOS 下载继续使用现有 Python HTTPS / 系统 curl 路径。")
        )
        policy.setWordWrap(True)
        policy.setStyleSheet(
            "background:#F7F8FA;border:1px solid #E2E5E9;border-radius:8px;"
            "padding:9px;color:#14202E;"
        )
        root.addWidget(policy)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(("模型", "管理方式", "本地版本", "官方版本", "状态"))
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self._table.itemSelectionChanged.connect(self._selection_changed)
        root.addWidget(self._table, 1)

        self._detail = QLabel("选择一个模型查看说明。")
        self._detail.setWordWrap(True)
        self._detail.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._detail.setStyleSheet(
            "background:#F7F8FA;border:1px solid #E2E5E9;border-radius:8px;"
            "padding:9px;color:#5B6B80;"
        )
        root.addWidget(self._detail)

        self._progress_label = QLabel("")
        self._progress_label.setWordWrap(True)
        self._progress_label.setObjectName("settingsCardSubtitle")
        self._progress_label.setVisible(False)
        root.addWidget(self._progress_label)
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setTextVisible(True)
        root.addWidget(self._progress)

        buttons = QHBoxLayout()
        self._refresh_local_btn = QPushButton("刷新本地状态")
        self._refresh_local_btn.setToolTip("只读取本机文件，不联网")
        self._refresh_local_btn.clicked.connect(self._load_local_statuses)
        buttons.addWidget(self._refresh_local_btn)

        self._check_btn = accent_button("检查更新")
        self._check_btn.setToolTip("仅在点击后连接各模型的官方仓库")
        self._check_btn.clicked.connect(self._check_updates)
        buttons.addWidget(self._check_btn)

        self._source_btn = QPushButton("打开官方来源")
        self._source_btn.clicked.connect(self._open_selected_source)
        buttons.addWidget(self._source_btn)

        buttons.addStretch(1)
        self._update_btn = QPushButton("选择可更新模型")
        self._update_btn.setEnabled(False)
        self._update_btn.clicked.connect(self._update_selected)
        buttons.addWidget(self._update_btn)

        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.reject)
        buttons.addWidget(close_btn)
        root.addLayout(buttons)
        for button in (self._refresh_local_btn, self._check_btn, self._source_btn, self._update_btn, close_btn):
            button.setAutoDefault(False)
            button.setDefault(False)

    def _load_local_statuses(self, _checked=False) -> None:
        if self._busy:
            return
        try:
            from utils.manual_ocr_model_updates import local_statuses

            statuses = local_statuses()
        except Exception as exc:
            QMessageBox.critical(self, "读取失败", f"无法读取 OCR 模型本地状态：\n{exc}")
            return
        self._checked_once = False
        self._set_statuses(statuses)
        self._progress_label.setVisible(False)
        self._progress.setVisible(False)

    def _set_statuses(self, statuses) -> None:
        selected_id = self._selected_component_id()
        self._statuses = {item.component_id: item for item in statuses}
        self._table.setRowCount(0)
        for item in statuses:
            row = self._table.rowCount()
            self._table.insertRow(row)
            values = (
                item.label,
                item.management,
                self._short_revision(item.local_revision),
                self._short_revision(item.remote_revision),
                self._STATE_TEXT.get(item.state, item.state),
            )
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value)
                if column == 0:
                    cell.setData(Qt.UserRole, item.component_id)
                cell.setToolTip(
                    f"本地：{item.local_revision}\n官方：{item.remote_revision}\n{item.detail}"
                )
                self._table.setItem(row, column, cell)
            if item.component_id == selected_id:
                self._table.selectRow(row)
        if self._table.rowCount() and self._table.currentRow() < 0:
            self._table.selectRow(0)
        self._selection_changed()

    @staticmethod
    def _short_revision(value: str) -> str:
        text = str(value or "—")
        if re.fullmatch(r"[0-9a-fA-F]{20,}", text):
            return text[:12]
        return text if len(text) <= 34 else text[:31] + "…"

    def _selected_component_id(self) -> str:
        row = self._table.currentRow()
        if row < 0:
            return ""
        item = self._table.item(row, 0)
        return str(item.data(Qt.UserRole) or "") if item is not None else ""

    def _selection_changed(self) -> None:
        component_id = self._selected_component_id()
        status = self._statuses.get(component_id)
        if status is None:
            self._detail.setText("选择一个模型查看说明。")
            self._update_btn.setText("选择可更新模型")
            self._update_btn.setEnabled(False)
            self._source_btn.setEnabled(False)
            return
        self._detail.setText(
            f"{status.label} · {status.management}\n"
            f"本地版本：{status.local_revision}    官方版本：{status.remote_revision}\n"
            f"{status.detail}"
        )
        self._source_btn.setEnabled(bool(status.source_url) and not self._busy)
        self._update_btn.setText(status.action_label or "更新所选模型")
        self._update_btn.setEnabled(bool(status.can_update) and not self._busy)
        self._update_btn.setAutoDefault(False)
        self._update_btn.setDefault(False)

    def _set_busy(self, busy: bool, text: str = "") -> None:
        self._busy = bool(busy)
        self._refresh_local_btn.setEnabled(not busy)
        self._check_btn.setEnabled(not busy)
        self._table.setEnabled(not busy)
        self._source_btn.setEnabled(not busy and bool(self._selected_component_id()))
        self._progress.setVisible(busy)
        self._progress_label.setVisible(busy)
        if text:
            self._progress_label.setText(text)
        self._selection_changed()

    def _check_updates(self, _checked=False) -> None:
        if self._busy:
            return
        from utils.manual_ocr_model_updates import check_updates

        signals = WorkerSignals()
        self._active_signals = signals
        signals.finished.connect(self._check_finished)
        signals.error.connect(self._operation_failed)
        signals.overall_progress.connect(self._progress_changed)
        self._progress.setRange(0, max(1, len(self._statuses)))
        self._progress.setValue(0)
        self._set_busy(True, "正在连接官方来源检查版本…")

        def callback(stage, current, total, detail):
            signals.overall_progress.emit(
                {"stage": stage, "current": current, "total": total, "detail": detail}
            )

        def worker():
            try:
                signals.finished.emit(check_updates(progress_callback=callback))
            except Exception as exc:
                signals.error.emit(str(exc))

        threading.Thread(target=worker, daemon=True, name="manual-ocr-model-check").start()

    def _check_finished(self, statuses) -> None:
        self._checked_once = True
        self._set_statuses(statuses)
        self._set_busy(False)
        self._progress.setRange(0, 1)
        self._progress.setValue(1)
        self._progress_label.setText("检查完成。没有任何自动下载或自动替换。")
        self._progress_label.setVisible(True)
        self._progress.setVisible(True)
        self._active_signals = None

    def _ocr_is_busy(self) -> bool:
        window = self.window()
        ocr_tab = getattr(window, "_tab_ocr", None)
        return bool(getattr(ocr_tab, "_ocr_run_active", False))

    def _update_selected(self, _checked=False) -> None:
        if self._busy:
            return
        component_id = self._selected_component_id()
        status = self._statuses.get(component_id)
        if status is None or not status.can_update:
            return
        if self._ocr_is_busy():
            QMessageBox.warning(
                self,
                "OCR 正在运行",
                "请先停止当前 OCR 任务，再更新模型。运行中的 worker 不会被强制替换。",
            )
            return
        target = status.remote_revision
        if target in {"", "未检查", "检查失败"} and component_id != "manga_48px":
            notify(self, "请先点击“检查更新”，确认官方目标版本后再更新。", "warning")
            return
        action = status.action_label or "更新所选模型"
        answer = QMessageBox.question(
            self,
            f"确认{action}",
            ui_message(
                "model_update.confirm.body",
                label=status.label, local=status.local_revision, target=target,
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        from utils.manual_ocr_model_updates import update_component

        signals = WorkerSignals()
        self._active_signals = signals
        signals.finished.connect(self._update_finished)
        signals.error.connect(self._operation_failed)
        signals.overall_progress.connect(self._progress_changed)
        self._progress.setRange(0, 0)
        self._set_busy(True, f"正在手动处理 {status.label}…")

        def callback(stage, current, total, detail):
            signals.overall_progress.emit(
                {"stage": stage, "current": current, "total": total, "detail": detail}
            )

        def worker():
            try:
                target_revision = None if component_id == "manga_48px" else target
                result = update_component(
                    component_id,
                    target_revision=target_revision,
                    progress_callback=callback,
                )
                signals.finished.emit(result)
            except Exception as exc:
                signals.error.emit(str(exc))

        threading.Thread(target=worker, daemon=True, name=f"manual-model-update-{component_id}").start()

    def _update_finished(self, status) -> None:
        self._statuses[status.component_id] = status
        ordered = [
            self._statuses[key]
            for key in (
                "apple_vision", "ndlocr_lite", "hayai_ocr", "manga_48px",
                "paddle_ocr", "paddle_aistudio",
            )
            if key in self._statuses
        ]
        self._set_statuses(ordered)
        self._set_busy(False)
        self._progress.setRange(0, 1)
        self._progress.setValue(1)
        self._progress_label.setText(status.detail)
        self._progress_label.setVisible(True)
        self._progress.setVisible(True)
        self._active_signals = None
        notify(self, status.detail, "success")

    def _progress_changed(self, payload) -> None:
        if not isinstance(payload, dict):
            return
        current = int(payload.get("current") or 0)
        total = int(payload.get("total") or 0)
        detail = str(payload.get("detail") or "正在处理…")
        self._progress_label.setText(detail)
        if total > 0:
            self._progress.setRange(0, total)
            self._progress.setValue(min(total, current))
        else:
            self._progress.setRange(0, 0)

    def _operation_failed(self, message: str) -> None:
        self._set_busy(False)
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress_label.setText("操作失败；现有模型未被静默覆盖。")
        self._progress_label.setVisible(True)
        self._progress.setVisible(True)
        self._active_signals = None
        QMessageBox.critical(self, "OCR 模型操作失败", str(message))

    def _open_selected_source(self, _checked=False) -> None:
        status = self._statuses.get(self._selected_component_id())
        if status is None or not status.source_url:
            return
        QDesktopServices.openUrl(QUrl(status.source_url))

    def reject(self) -> None:
        if self._busy:
            notify(self, "模型检查或更新正在进行，完成后才能关闭此窗口。", "warning")
            return
        super().reject()

    def closeEvent(self, event) -> None:
        if self._busy:
            event.ignore()
            notify(self, "模型检查或更新正在进行，完成后才能关闭此窗口。", "warning")
            return
        super().closeEvent(event)


class SystemSettingsTab(QWidget):
    """完整系统设置页：保存界面、启动、OCR 默认行为与运行状态。"""

    workspace_requested = Signal(str)
    settings_changed = Signal(object)
    ocr_runtime_check_requested = Signal()

    SETTINGS_ORG = "NovelFormatter"
    SETTINGS_APP = "NovelFormatter1"

    RESTORE_KEY = "reference_ui/restore_workspace"
    REMEMBER_SUBTABS_KEY = "reference_ui/remember_subtabs"
    DEFAULT_SECTION_KEY = "reference_ui/default_section"
    START_MAXIMIZED_KEY = "reference_ui/start_maximized"
    LANGUAGE_KEY = "reference_ui/language"
    APPEARANCE_KEY = "reference_ui/appearance"
    NAVIGATION_SCHEMA_KEY = "reference_ui/navigation_schema_version"
    NAVIGATION_SCHEMA_VERSION = 3
    PAGE_MANAGER_STARTUP_MIGRATION_KEY = "reference_ui/page_manager_startup_migration_v3"

    OCR_ENGINE_KEY = "ocr/default_engine"
    OCR_SETTINGS_TAB_KEY = "ocr/default_settings_tab"
    OCR_REVIEW_DEFAULT_KEY = "ocr/review_enabled_by_default"
    OCR_REVIEW_EXPANDED_KEY = "ocr/review_panel_expanded"
    OCR_PREVIEW_DEFAULT_KEY = "ocr/live_preview_default"
    OCR_PROGRESS_DEFAULT_KEY = "ocr/progress_default"
    OCR_REVIEW_DEFAULT_CLOSED_MIGRATION_KEY = "ocr/review_default_closed_v2"
    OCR_REVIEW_PANEL_OPEN_MIGRATION_KEY = "ocr/review_panel_open_default_v3"
    OCR_ENGINE_TAB_STARTUP_MIGRATION_KEY = "ocr/engine_tab_startup_migration_v1"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._settings = QSettings(self.SETTINGS_ORG, self.SETTINGS_APP)
        self._device_detection_busy = False
        self._device_detection_signals = None
        self._device_report = None
        # Source updater is opt-in only. No network call is made during startup.
        self._source_update_info = None
        self._source_update_check_busy = False
        self._source_update_install_busy = False
        self._source_update_signals = None
        self._source_update_thread = None
        self._source_update_busy_provider = None
        self._migrate_navigation_schema()
        if not self._settings.value(self.PAGE_MANAGER_STARTUP_MIGRATION_KEY, False, type=bool):
            self._settings.setValue(self.DEFAULT_SECTION_KEY, SECTION_PAGE)
            self._settings.setValue(self.RESTORE_KEY, False)
            self._settings.setValue(self.OCR_SETTINGS_TAB_KEY, 0)
            self._settings.setValue(self.PAGE_MANAGER_STARTUP_MIGRATION_KEY, True)
            self._settings.sync()
        if not self._settings.value(
            self.OCR_REVIEW_DEFAULT_CLOSED_MIGRATION_KEY, False, type=bool
        ):
            self._settings.setValue(self.OCR_REVIEW_DEFAULT_KEY, False)
            self._settings.setValue(self.OCR_REVIEW_DEFAULT_CLOSED_MIGRATION_KEY, True)
        if not self._settings.value(
            self.OCR_REVIEW_PANEL_OPEN_MIGRATION_KEY, False, type=bool
        ):
            # Review remains disabled by default, while its settings are visible
            # immediately so users do not mistake a collapsed panel for a
            # missing feature.
            self._settings.setValue(self.OCR_REVIEW_DEFAULT_KEY, False)
            self._settings.setValue(self.OCR_REVIEW_EXPANDED_KEY, True)
            self._settings.setValue(self.OCR_SETTINGS_TAB_KEY, 0)
            self._settings.setValue(self.OCR_REVIEW_PANEL_OPEN_MIGRATION_KEY, True)
        if not self._settings.value(self.OCR_ENGINE_TAB_STARTUP_MIGRATION_KEY, False, type=bool):
            self._settings.setValue(self.OCR_SETTINGS_TAB_KEY, 0)
            self._settings.setValue(self.OCR_ENGINE_TAB_STARTUP_MIGRATION_KEY, True)
        self._settings.sync()
        self._build()
        self._load()

    def _migrate_navigation_schema(self) -> None:
        """Remap saved numeric sections across navigation schema changes.

        v1 -> v2 swapped Format/Proof.  v3 inserts the independent Workspace
        page before the historical six sections, so all previous indices shift
        by +1.
        """
        try:
            version = int(self._settings.value(self.NAVIGATION_SCHEMA_KEY, 1))
        except (TypeError, ValueError):
            version = 1
        if version >= self.NAVIGATION_SCHEMA_VERSION:
            return

        def remap(value, fallback=SECTION_PAGE):
            try:
                index = int(value)
            except (TypeError, ValueError):
                # fallback uses the current schema; convert to the old page index
                # before applying migrations below.
                index = 0
            if version < 2:
                index = {2: 3, 3: 2}.get(index, index)
            if version < 3:
                index = max(0, min(5, index)) + 1
            return index

        self._settings.setValue(
            self.DEFAULT_SECTION_KEY,
            remap(self._settings.value(self.DEFAULT_SECTION_KEY, 0)),
        )
        self._settings.setValue(
            "reference_ui/last_section",
            remap(self._settings.value("reference_ui/last_section", 0)),
        )
        self._settings.setValue(self.NAVIGATION_SCHEMA_KEY, self.NAVIGATION_SCHEMA_VERSION)

    @staticmethod
    def _make_card(title: str, subtitle: str = "") -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setObjectName("settingsCard")
        box = QVBoxLayout(frame)
        box.setContentsMargins(14, 12, 14, 12)
        box.setSpacing(8)
        heading = QLabel(title)
        heading.setObjectName("settingsCardTitle")
        box.addWidget(heading)
        if subtitle:
            note = QLabel(subtitle)
            note.setWordWrap(True)
            note.setObjectName("settingsCardSubtitle")
            box.addWidget(note)
        return frame, box

    @staticmethod
    def _scroll_page() -> tuple[QScrollArea, QWidget, QVBoxLayout]:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        page = QWidget()
        page.setObjectName("settingsPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        scroll.setWidget(page)
        return scroll, page, layout

    @staticmethod
    def _fixed_enabled_row(text: str, detail: str = "") -> QWidget:
        """Render a fixed safety contract as status, not a fake disabled option."""
        row = QWidget()
        line = QHBoxLayout(row)
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(8)
        marker = QLabel("✓")
        marker.setToolTip("核心运行合同：始终启用，不是可切换选项")
        marker.setStyleSheet(
            "font-weight:700;color:#147A4A;background:#EAF8F0;"
            "border:1px solid #B9E2CA;border-radius:8px;padding:1px 5px;"
        )
        line.addWidget(marker, 0, Qt.AlignTop)
        title = QLabel(text)
        title.setStyleSheet("font-weight:600;")
        line.addWidget(title, 0, Qt.AlignTop)
        if detail:
            note = QLabel(detail)
            note.setWordWrap(True)
            note.setObjectName("settingsCardSubtitle")
            line.addWidget(note, 1)
        else:
            line.addStretch(1)
        return row

    def _build(self) -> None:
        root = wrap_in_card(self)
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(28, 7, 28, 18)
        outer.setSpacing(10)

        topbar = QWidget()
        title_row = QHBoxLayout(topbar)
        title_row.setContentsMargins(0, 0, 0, 0)
        title = QLabel("设置")
        title.setStyleSheet("font-size:15px;font-weight:700;")
        title.setVisible(False)  # PageHeader owns the redesigned page title.
        title_row.addWidget(title)
        title_row.addStretch(1)
        self._save_top_btn = accent_button("保存设置")
        self._save_top_btn.clicked.connect(self._save)
        self._save_top_btn.setVisible(False)
        title_row.addWidget(self._save_top_btn)
        self._settings_detail_btn = QPushButton("详细设置")
        self._settings_detail_btn.clicked.connect(lambda: self._show_settings_details())
        title_row.addWidget(self._settings_detail_btn)
        self._settings_topbar = topbar
        self._settings_topbar.setVisible(False)
        outer.addWidget(self._settings_topbar)

        self._tabs = QTabWidget()
        self._tabs.setObjectName("systemSettingsTabs")
        self._tabs.setDocumentMode(True)
        self._tabs.addTab(self._build_general_page(), "常规设置")
        self._tabs.addTab(self._build_ocr_page(), "OCR 设置")
        self._tabs.addTab(self._build_performance_page(), "性能设置")
        self._tabs.addTab(self._build_shortcuts_page(), "快捷键")
        self._tabs.addTab(self._build_update_page(), "更新与诊断")
        self._tabs.addTab(self._build_about_page(), "关于")
        self._tabs.setVisible(False)

        self._settings_overview = self._build_settings_overview()
        outer.addWidget(self._settings_overview, 1)
        outer.addWidget(self._tabs, 1)
        root.addWidget(container, 1)

    def _build_settings_overview(self) -> QWidget:
        page = QWidget()
        grid = QGridLayout(page)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(16)

        def overview_card(title: str, badge: str, subtitle: str, rows: list[tuple[str, str]], tab_index: int) -> QFrame:
            card = QFrame()
            card.setObjectName("settingsOverviewCard")
            card.setCursor(Qt.PointingHandCursor)
            card.setFixedHeight(161)
            card.setStyleSheet(
                f"QFrame#settingsOverviewCard{{background:{CARD};border:1px solid {BORDER};border-radius:16px;}}"
                "QFrame#settingsOverviewCard:hover{border-color:#AFCBF3;}"
            )
            box = QVBoxLayout(card)
            box.setContentsMargins(20, 16, 20, 14)
            box.setSpacing(7)
            hdr = QHBoxLayout()
            name = QLabel(title)
            name.setStyleSheet("font-size:13px;font-weight:750;")
            hdr.addWidget(name)
            hdr.addStretch(1)
            marker = QLabel(badge)
            marker.setStyleSheet(
                "color:#147A4A;background:#EAF8F0;border:1px solid #B9E2CA;"
                "border-radius:9px;padding:2px 7px;font-size:10px;font-weight:700;"
            )
            hdr.addWidget(marker)
            box.addLayout(hdr)
            sub = QLabel(subtitle)
            sub.setStyleSheet(f"color:{MUTED};font-size:10px;")
            box.addWidget(sub)
            sep = QFrame()
            sep.setFixedHeight(1)
            sep.setStyleSheet(f"background:{BORDER};border:none;")
            box.addWidget(sep)
            for row_index, (left, right) in enumerate(rows):
                line = QHBoxLayout()
                l = QLabel(left)
                l.setStyleSheet(f"color:{INK};font-size:11px;")
                r = QLabel(right)
                r.setStyleSheet(f"color:{MUTED};font-size:11px;font-weight:600;")
                r.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                line.addWidget(l)
                line.addStretch(1)
                line.addWidget(r)
                box.addLayout(line)
                if row_index != len(rows) - 1:
                    row_sep = QFrame()
                    row_sep.setFixedHeight(1)
                    row_sep.setStyleSheet(f"background:{BORDER};border:none;")
                    box.addWidget(row_sep)
            card.mousePressEvent = lambda event, idx=tab_index: self._show_settings_details(idx)
            return card

        enabled_by_id = {
            aid: name for aid, name, _badge, _color, _desc, enabled in OCR_ADAPTERS
            if enabled and adapter_available_on_current_platform(aid)
        }
        ocr_rows = [
            (enabled_by_id.get("hayai_ocr", "Hayai OCR"), "可用"),
            (enabled_by_id.get("ndlocr_lite", "NDLOCR-Lite"), "可用"),
        ]

        machine = platform.machine() or "当前设备"
        device_rows = [
            ("设备", f"{platform.system() or '系统'} · {machine}"),
            ("显存策略", "自动"),
        ]
        self._overview_interface_rows = [
            ("主题", "跟随系统"),
            ("语言", "简体中文"),
        ]
        ai_rows = [
            ("服务", "按需调用"),
            ("自动联网", "关闭"),
        ]
        self._overview_ocr_card = overview_card("OCR 模型", "已就绪", "本地引擎与运行环境", ocr_rows, 1)
        self._overview_device_card = overview_card("计算设备", "自动", "自动检测加速设备", device_rows, 2)
        self._overview_interface_card = overview_card("界面", "当前", "主题与语言", self._overview_interface_rows, 0)
        self._overview_ai_card = overview_card("AI 裁决", "可用", "GPT 共识裁决服务", ai_rows, 1)
        grid.addWidget(self._overview_ocr_card, 0, 0)
        grid.addWidget(self._overview_device_card, 0, 1)
        grid.addWidget(self._overview_interface_card, 1, 0)
        grid.addWidget(self._overview_ai_card, 1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(2, 1)
        return page

    def _show_settings_details(self, tab_index: int | None = None) -> None:
        if tab_index is not None:
            self._tabs.setCurrentIndex(max(0, min(self._tabs.count() - 1, int(tab_index))))
        self._settings_overview.setVisible(False)
        self._tabs.setVisible(True)
        self._settings_topbar.setVisible(True)
        self._settings_detail_btn.setVisible(True)
        self._save_top_btn.setVisible(True)
        self._settings_detail_btn.setText("概览")
        try:
            self._settings_detail_btn.clicked.disconnect()
        except RuntimeError:
            pass
        self._settings_detail_btn.clicked.connect(self._show_settings_overview)

    def _show_settings_overview(self) -> None:
        self._tabs.setVisible(False)
        self._settings_overview.setVisible(True)
        self._save_top_btn.setVisible(False)
        self._settings_detail_btn.setText("详细设置")
        self._settings_topbar.setVisible(False)
        try:
            self._settings_detail_btn.clicked.disconnect()
        except RuntimeError:
            pass
        self._settings_detail_btn.clicked.connect(lambda: self._show_settings_details())

    def _build_general_page(self) -> QWidget:
        scroll, _page, layout = self._scroll_page()

        interface_card, form_box = self._make_card(
            "界面与显示",
            "可切换浅色/深色主题与中、日、英界面语言；只改变界面显示，不修改 OCR、文档或模型数据。",
        )
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(8)
        self._language_combo = NoWheelComboBox()
        # Language names stay self-identifying in every locale; itemData is the
        # stable setting value used by the runtime manager.
        self._language_combo.setProperty("nfNoTranslateItems", True)
        self._language_combo.addItem("简体中文", LANG_ZH)
        self._language_combo.addItem("日本語", LANG_JA)
        self._language_combo.addItem("English", LANG_EN)
        self._appearance_combo = NoWheelComboBox()
        self._appearance_combo.addItem("浅色", THEME_LIGHT)
        self._appearance_combo.addItem("深色", THEME_DARK)
        self._language_combo.currentIndexChanged.connect(self._preview_interface_preferences)
        self._appearance_combo.currentIndexChanged.connect(self._preview_interface_preferences)
        form.addRow("界面语言", self._language_combo)
        form.addRow("外观", self._appearance_combo)
        form_box.addLayout(form)
        layout.addWidget(interface_card)

        startup_card, startup_box = self._make_card(
            "启动设置",
            "恢复上次功能区优先于默认启动功能区；子页签可独立记忆。",
        )
        self._restore_workspace_cb = QCheckBox("启动时恢复上次功能区")
        self._remember_subtabs_cb = QCheckBox("记住 OCR 与文字校对子页签")
        self._start_maximized_cb = QCheckBox("启动时最大化窗口")
        startup_box.addWidget(self._restore_workspace_cb)
        startup_box.addWidget(self._remember_subtabs_cb)
        startup_box.addWidget(self._start_maximized_cb)
        default_row = QHBoxLayout()
        default_row.addWidget(QLabel("默认启动功能区"))
        self._default_section_combo = NoWheelComboBox()
        for index, (_key, label) in enumerate(REFERENCE_SECTION_ITEMS):
            self._default_section_combo.addItem(label, index)
        default_row.addWidget(self._default_section_combo, 1)
        startup_box.addLayout(default_row)
        layout.addWidget(startup_card)

        path_card, path_box = self._make_card(
            "文件与诊断目录",
            "缓存、预览裁片和崩溃日志继续由原有安全清理流程管理。",
        )
        path_row = QHBoxLayout()
        self._temp_path = QLineEdit(str(Path(tempfile.gettempdir()).resolve()))
        self._temp_path.setReadOnly(True)
        path_row.addWidget(self._temp_path, 1)
        open_temp_btn = QPushButton("打开临时文件夹")
        open_temp_btn.clicked.connect(self._open_temp_folder)
        path_row.addWidget(open_temp_btn)
        path_box.addLayout(path_row)

        debug_row = QHBoxLayout()
        debug_path = PROJECT_ROOT / "debug"
        self._debug_path = QLineEdit(str(debug_path))
        self._debug_path.setReadOnly(True)
        debug_row.addWidget(self._debug_path, 1)
        open_debug_btn = QPushButton("打开诊断目录")
        open_debug_btn.clicked.connect(self._open_debug_folder)
        debug_row.addWidget(open_debug_btn)
        path_box.addLayout(debug_row)
        layout.addWidget(path_card)

        layout.addStretch(1)
        action_row = QHBoxLayout()
        action_row.addStretch(1)
        defaults_btn = QPushButton("恢复默认设置")
        defaults_btn.clicked.connect(self._reset_defaults)
        action_row.addWidget(defaults_btn)
        save_btn = accent_button("保存设置")
        save_btn.clicked.connect(self._save)
        action_row.addWidget(save_btn)
        layout.addLayout(action_row)
        return scroll

    def _build_ocr_page(self) -> QWidget:
        scroll, _page, layout = self._scroll_page()

        defaults_card, defaults_box = self._make_card(
            "OCR 默认行为",
            "这些选项在启动时应用，也可保存后立即同步到当前 OCR 工作区。",
        )
        engine_row = QHBoxLayout()
        engine_row.addWidget(QLabel("默认识别引擎"))
        self._ocr_engine_combo = NoWheelComboBox()
        for aid, name, badge_text, _color, _desc, enabled in OCR_ADAPTERS:
            if enabled and adapter_available_on_current_platform(aid):
                self._ocr_engine_combo.addItem(f"{name} · {badge_text}", aid)
        engine_row.addWidget(self._ocr_engine_combo, 1)
        defaults_box.addLayout(engine_row)

        tab_row = QHBoxLayout()
        tab_row.addWidget(QLabel("默认打开设置页"))
        self._ocr_settings_tab_combo = NoWheelComboBox()
        self._ocr_settings_tab_combo.addItem("引擎", 0)
        self._ocr_settings_tab_combo.addItem("分列与组句", 1)
        self._ocr_settings_tab_combo.addItem("逐字审校", 2)
        tab_row.addWidget(self._ocr_settings_tab_combo, 1)
        defaults_box.addLayout(tab_row)

        self._ocr_review_default_cb = QCheckBox("逐字审校默认启用")
        self._ocr_review_expanded_cb = QCheckBox("逐字审校设置默认展开")
        self._ocr_review_expanded_cb.setChecked(True)
        self._ocr_review_expanded_cb.setVisible(False)
        self._ocr_preview_default_cb = QCheckBox("默认开启实时图像预览")
        self._ocr_progress_default_cb = QCheckBox("默认显示实时进度与预计时间")
        defaults_box.addWidget(self._ocr_review_default_cb)
        defaults_box.addWidget(self._ocr_review_expanded_cb)
        defaults_box.addWidget(self._ocr_preview_default_cb)
        defaults_box.addWidget(self._ocr_progress_default_cb)
        layout.addWidget(defaults_card)

        workspace_card, workspace_box = self._make_card(
            "OCR 工作区入口",
            "识别引擎、固定蓝框、分列顺序、多模型、逐字审校和运行日志仍集中在 OCR 识别区。",
        )
        row = QHBoxLayout()
        open_ocr = accent_button("前往 OCR 识别")
        open_ocr.clicked.connect(lambda: self.workspace_requested.emit("ocr"))
        row.addWidget(open_ocr)
        open_pdf = QPushButton("打开 PDF 文字层")
        open_pdf.clicked.connect(lambda: self.workspace_requested.emit("pdf"))
        row.addWidget(open_pdf)
        runtime_btn = QPushButton("检测 OCR 运行环境")
        runtime_btn.clicked.connect(lambda _checked=False: self.ocr_runtime_check_requested.emit())
        row.addWidget(runtime_btn)
        row.addStretch(1)
        workspace_box.addLayout(row)
        layout.addWidget(workspace_card)

        ai_card, ai_box = self._make_card(
            "AI 服务设置",
            "Formatter、文字校对和 OCR 裁决共用原有 AI 服务商、密钥、模型和并发配置。",
        )
        ai_btn = QPushButton("打开 AI 设置")
        ai_btn.clicked.connect(lambda: AISettingsDialog(self).exec())
        ai_box.addWidget(ai_btn, 0, Qt.AlignLeft)
        layout.addWidget(ai_card)

        model_card, model_box = self._make_card(
            "OCR 模型更新（仅手动）",
            "不会启动时检查、不会后台轮询、不会自动下载更新包或替换已安装模型。只有点击检查并再次确认后才会更新。",
        )
        model_box.addWidget(
            self._fixed_enabled_row(
                "自动检查与自动更新永久关闭",
                "打开管理窗口只读取本地版本；联网检查和模型替换必须由用户分别手动触发。",
            )
        )
        model_row = QHBoxLayout()
        manage_models = accent_button("管理 OCR 模型更新")
        manage_models.clicked.connect(self._open_ocr_model_updates)
        model_row.addWidget(manage_models)
        download_note = "Windows：BITS → curl.exe → Python HTTPS。" if os.name == "nt" else "macOS：保持 Python HTTPS / 系统 curl 下载路径。"
        model_note = QLabel("支持官方版本检查；仅对可安全独立替换的模型提供更新/修复。" + download_note)
        model_note.setWordWrap(True)
        model_note.setObjectName("settingsCardSubtitle")
        model_row.addWidget(model_note, 1)
        model_box.addLayout(model_row)
        layout.addWidget(model_card)
        layout.addStretch(1)
        return scroll

    def _build_performance_page(self) -> QWidget:
        scroll, _page, layout = self._scroll_page()

        protection_card, protection_box = self._make_card(
            "稳定性保护",
            "下列保护属于核心安全机制，保持开启且不会被界面设置误关闭。",
        )
        for text, detail in (
            ("OCR 子进程 watchdog", "卡死或超时会结束异常子进程，并保留诊断信息。"),
            ("跨平台加速安全回退", "CUDA、DirectML 或 MPS 不可用时按原有策略回退 CPU，检测结果不会自动改设置。"),
            ("后台任务代次隔离", "切书、清空或关闭窗口后，旧任务不能覆盖新界面。"),
            ("退出与清空时释放临时预览", "OCR 临时图只在工作区生命周期内保留。"),
        ):
            protection_box.addWidget(self._fixed_enabled_row(text, detail))
        layout.addWidget(protection_card)

        device_card, device_box = self._make_card(
            "设备与 GPU 检测",
            "只在点击后读取本机硬件和已安装 OCR 独立环境；不会联网、不会安装驱动，也不会自动切换 OCR 设备。",
        )
        device_actions = QHBoxLayout()
        self._detect_device_btn = accent_button("检测设备与 GPU")
        self._detect_device_btn.clicked.connect(self._detect_devices)
        device_actions.addWidget(self._detect_device_btn)
        self._device_detection_status = QLabel("尚未检测。Windows 会检查 CIM、NVIDIA 驱动、CUDA/DirectML；Mac 会检查 Metal/MPS。")
        self._device_detection_status.setWordWrap(True)
        self._device_detection_status.setObjectName("settingsCardSubtitle")
        device_actions.addWidget(self._device_detection_status, 1)
        device_box.addLayout(device_actions)
        self._device_detection_progress = QProgressBar()
        self._device_detection_progress.setRange(0, 3)
        self._device_detection_progress.setValue(0)
        self._device_detection_progress.setVisible(False)
        device_box.addWidget(self._device_detection_progress)
        self._device_detection_output = QPlainTextEdit()
        self._device_detection_output.setReadOnly(True)
        self._device_detection_output.setMinimumHeight(180)
        self._device_detection_output.setPlaceholderText("点击“检测设备与 GPU”后显示 CPU、内存、显卡、驱动以及各 OCR 运行时可用的 CUDA / DirectML / MPS 后端。")
        self._device_detection_output.setStyleSheet(LIGHT_LOG_STYLE)
        device_box.addWidget(self._device_detection_output)
        layout.addWidget(device_card)

        long_book_card, long_book_box = self._make_card(
            "长篇书籍处理",
            "减少 300 页以上项目的重复对齐、索引和界面刷新。",
        )
        for text, detail in (
            ("图文对照按需建索引", "真正打开图文对照页时才读取整本图像索引。"),
            ("OCR 实时进度可关闭", "关闭后停止高频进度刷新，但识别任务继续。"),
            ("OCR 实时预览可关闭", "关闭后停止生成新预览，减少磁盘和界面负担。"),
        ):
            long_book_box.addWidget(self._fixed_enabled_row(text, detail))
        layout.addWidget(long_book_card)

        runtime_card, runtime_box = self._make_card("当前运行环境")
        runtime_grid = QGridLayout()
        runtime_grid.setHorizontalSpacing(18)
        runtime_grid.setVerticalSpacing(7)
        runtime_rows = (
            ("操作系统", f"{platform.system() or sys.platform} {platform.release()} · {platform.machine() or 'unknown'}"),
            ("Python", sys.version.split()[0]),
            ("CPU 逻辑核心", str(os.cpu_count() or "未知")),
            ("程序目录", str(PROJECT_ROOT)),
            ("临时目录", str(Path(tempfile.gettempdir()).resolve())),
        )
        for row, (name, value) in enumerate(runtime_rows):
            runtime_grid.addWidget(QLabel(name), row, 0)
            value_label = QLabel(value)
            value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            value_label.setWordWrap(True)
            value_label.setObjectName("settingsCardSubtitle")
            runtime_grid.addWidget(value_label, row, 1)
        runtime_grid.setColumnStretch(1, 1)
        runtime_box.addLayout(runtime_grid)
        layout.addWidget(runtime_card)
        layout.addStretch(1)
        return scroll

    def _detect_devices(self, _checked=False) -> None:
        if self._device_detection_busy:
            return
        from utils.device_detection import detect_devices

        self._device_detection_busy = True
        self._detect_device_btn.setEnabled(False)
        self._device_detection_progress.setVisible(True)
        self._device_detection_progress.setRange(0, 3)
        self._device_detection_progress.setValue(0)
        self._device_detection_status.setText("正在检测本机设备和已安装 OCR 运行时…")
        signals = WorkerSignals()
        self._device_detection_signals = signals
        signals.finished.connect(self._device_detection_finished)
        signals.error.connect(self._device_detection_failed)
        signals.overall_progress.connect(self._device_detection_progress_changed)

        def callback(stage, current, total, detail):
            signals.overall_progress.emit(
                {"stage": stage, "current": current, "total": total, "detail": detail}
            )

        def worker():
            try:
                signals.finished.emit(detect_devices(progress_callback=callback))
            except Exception as exc:
                signals.error.emit(str(exc))

        threading.Thread(target=worker, daemon=True, name="manual-device-gpu-detection").start()

    def _device_detection_progress_changed(self, payload) -> None:
        if not isinstance(payload, dict):
            return
        current = int(payload.get("current") or 0)
        total = int(payload.get("total") or 0)
        detail = str(payload.get("detail") or "正在检测…")
        self._device_detection_status.setText(detail)
        if total > 0:
            self._device_detection_progress.setRange(0, total)
            self._device_detection_progress.setValue(min(current, total))
        else:
            self._device_detection_progress.setRange(0, 0)

    @staticmethod
    def _format_device_report(report) -> str:
        lines = [
            f"系统：{report.platform_name} {report.platform_release} · {report.architecture}",
            f"CPU：{report.cpu}" + (f" · {report.cpu_topology}" if report.cpu_topology else ""),
            f"逻辑核心：{report.logical_cores}",
            f"内存：{report.memory_gb:.1f} GB" if report.memory_gb else "内存：未能读取",
            "",
            "GPU：",
        ]
        if report.neural_engine:
            lines.append("Apple AI：" + report.neural_engine)
        if report.gpus:
            for index, gpu in enumerate(report.gpus, start=1):
                details = [gpu.vendor, f"显存 {gpu.memory_mb} MB" if gpu.memory_mb else "", f"驱动 {gpu.driver}" if gpu.driver else "", gpu.source]
                lines.append(f"  {index}. {gpu.name}" + (" · " + " · ".join(item for item in details if item) if any(details) else ""))
        else:
            lines.append("  未检测到可枚举的独立/集成 GPU。")
        lines += ["", "OCR 运行时后端："]
        if report.runtimes:
            for runtime in report.runtimes:
                backends = []
                if runtime.cuda_available:
                    backends.append("PyTorch CUDA")
                if runtime.mps_available:
                    backends.append("Apple MPS")
                elif runtime.mps_built:
                    backends.append("PyTorch MPS 已编译但初始化失败")
                if runtime.mlx_available:
                    backends.append("MLX-VLM（已安装，未启动）")
                if "CUDAExecutionProvider" in runtime.onnx_providers:
                    backends.append("ONNX CUDA")
                if "DmlExecutionProvider" in runtime.onnx_providers:
                    backends.append("ONNX DirectML")
                if "CoreMLExecutionProvider" in runtime.onnx_providers:
                    backends.append("ONNX CoreML")
                if not backends:
                    backends.append("CPU/未安装加速后端")
                versions = []
                if runtime.torch_version:
                    versions.append("torch " + runtime.torch_version)
                if runtime.onnxruntime_version:
                    versions.append("onnxruntime " + runtime.onnxruntime_version)
                if runtime.mlx_version:
                    versions.append("mlx " + runtime.mlx_version)
                if runtime.mlx_vlm_version:
                    versions.append("mlx-vlm " + runtime.mlx_vlm_version)
                arch = str(getattr(runtime, "python_architecture", "") or "").strip()
                bits = int(getattr(runtime, "python_bits", 0) or 0)
                if arch:
                    versions.append(f"Python {arch}" + (f"/{bits}" if bits else ""))
                suffix = " · ".join([*backends, *versions])
                lines.append(f"  - {runtime.runtime}：{suffix}")
                if runtime.cuda_devices:
                    lines.append("    CUDA 设备：" + "、".join(runtime.cuda_devices))
                if runtime.onnx_providers:
                    lines.append("    ONNX Providers：" + "、".join(runtime.onnx_providers))
                if runtime.detail:
                    lines.append("    说明：" + runtime.detail)
        else:
            lines.append("  未发现可探测的 Python/OCR 运行环境。")
        lines += ["", "结论：" + report.acceleration_summary]
        if report.notes:
            lines += ["", *["注意：" + note for note in report.notes]]
        return "\n".join(lines)

    def _device_detection_finished(self, report) -> None:
        self._device_report = report
        self._device_detection_busy = False
        self._detect_device_btn.setEnabled(True)
        self._device_detection_progress.setRange(0, 3)
        self._device_detection_progress.setValue(3)
        self._device_detection_progress.setVisible(True)
        self._device_detection_status.setText(report.acceleration_summary)
        self._device_detection_output.setPlainText(self._format_device_report(report))
        self._device_detection_signals = None

    def _device_detection_failed(self, message: str) -> None:
        self._device_detection_busy = False
        self._detect_device_btn.setEnabled(True)
        self._device_detection_progress.setRange(0, 3)
        self._device_detection_progress.setValue(0)
        self._device_detection_status.setText("设备检测失败；未修改任何 OCR 设置。")
        self._device_detection_output.setPlainText(str(message))
        self._device_detection_signals = None
        QMessageBox.critical(self, "设备检测失败", str(message))

    def _build_shortcuts_page(self) -> QWidget:
        scroll, _page, layout = self._scroll_page()
        card, box = self._make_card(
            "功能区与校对快捷键",
            "快捷键只在对应工作区生效，不会抢占文本输入框的普通编辑按键。",
        )
        grid = QGridLayout()
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(8)
        mod = "⌘" if sys.platform == "darwin" else "Ctrl+"
        rows = [
            ("工作区", f"{mod}1"),
            ("页面管理", f"{mod}2"),
            ("OCR 识别", f"{mod}3"),
            ("格式处理", f"{mod}4"),
            ("文字校对", f"{mod}5"),
            ("EPUB生成", f"{mod}6"),
            ("设置", f"{mod}7"),
            ("命令面板", f"{mod}K"),
            ("保存当前校对", "Ctrl + Enter"),
            ("上一句 / 下一句", "Ctrl + ↑ / ↓"),
            ("上一列 / 下一列", "Ctrl + ← / →"),
            ("OCR 预览上一页 / 下一页", "Option + ← / →"),
        ]
        for row, (label, shortcut) in enumerate(rows):
            name = QLabel(label)
            key = QLabel(shortcut)
            key.setObjectName("shortcutKey")
            grid.addWidget(name, row, 0)
            grid.addWidget(key, row, 1)
        grid.setColumnStretch(0, 1)
        box.addLayout(grid)
        layout.addWidget(card)
        layout.addStretch(1)
        return scroll

    def _build_update_page(self) -> QWidget:
        """Manual source-update and diagnostic surface, isolated from business tabs."""
        from utils.source_update import (
            DEFAULT_BRANCH, DEFAULT_REPOSITORY, diagnostic_summary,
            discover_project_root, read_project_version,
        )

        scroll, _page, layout = self._scroll_page()
        update_card, update_box = self._make_card(
            "Git 仓库更新",
            "参考 Folirina 的安全更新思路：只在手动点击后联网；仓库与 main 分支锁定。"
            "Git 工作区只允许 clean + fast-forward；便携源码包先下载到临时区、完整校验，再事务式替换程序文件，失败自动回滚。",
        )
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(8)
        self._source_repo = QLineEdit(DEFAULT_REPOSITORY)
        self._source_repo.setReadOnly(True)
        self._source_repo.setToolTip("更新源已锁定，配置和界面都不能改到其它仓库。")
        self._source_branch = QLineEdit(DEFAULT_BRANCH)
        self._source_branch.setReadOnly(True)
        self._source_branch.setToolTip("更新分支固定为 main。")
        try:
            root = discover_project_root(PROJECT_ROOT)
            local_version = read_project_version(root)
        except Exception:
            root = PROJECT_ROOT
            local_version = VERSION
        self._source_root = QLineEdit(str(root))
        self._source_root.setReadOnly(True)
        grid.addWidget(QLabel("GitHub 仓库"), 0, 0)
        grid.addWidget(self._source_repo, 0, 1)
        grid.addWidget(QLabel("更新分支"), 1, 0)
        grid.addWidget(self._source_branch, 1, 1)
        grid.addWidget(QLabel("本地程序目录"), 2, 0)
        grid.addWidget(self._source_root, 2, 1)
        grid.addWidget(QLabel("当前版本"), 3, 0)
        self._source_local_version = QLabel(f"v{local_version}")
        self._source_local_version.setObjectName("settingsCardSubtitle")
        grid.addWidget(self._source_local_version, 3, 1)
        grid.setColumnStretch(1, 1)
        update_box.addLayout(grid)

        actions = QHBoxLayout()
        self._source_check_btn = accent_button("检查 Git 更新")
        self._source_check_btn.clicked.connect(self._check_source_update)
        self._source_install_btn = accent_button("从 Git 更新程序")
        self._source_install_btn.setEnabled(False)
        self._source_install_btn.clicked.connect(self._install_source_update)
        open_repo = QPushButton("打开 GitHub 仓库")
        open_repo.clicked.connect(self._open_source_repository)
        open_root = QPushButton("打开程序目录")
        open_root.clicked.connect(self._open_source_root)
        actions.addWidget(self._source_check_btn)
        actions.addWidget(self._source_install_btn)
        actions.addWidget(open_repo)
        actions.addWidget(open_root)
        actions.addStretch(1)
        update_box.addLayout(actions)

        self._source_update_status = QLabel("尚未检查更新。启动程序不会自动联网检查。")
        self._source_update_status.setWordWrap(True)
        self._source_update_status.setObjectName("settingsCardSubtitle")
        update_box.addWidget(self._source_update_status)
        self._source_remote_detail = QLabel("")
        self._source_remote_detail.setWordWrap(True)
        self._source_remote_detail.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._source_remote_detail.setObjectName("settingsCardSubtitle")
        update_box.addWidget(self._source_remote_detail)
        self._source_update_log = QPlainTextEdit()
        self._source_update_log.setReadOnly(True)
        self._source_update_log.setMaximumBlockCount(500)
        self._source_update_log.setMinimumHeight(120)
        self._source_update_log.setPlaceholderText("检查与更新过程会显示在这里。")
        self._source_update_log.setStyleSheet(LIGHT_LOG_STYLE)
        update_box.addWidget(self._source_update_log)
        protection = QLabel(
            "更新保护：不会自动更新、不会自动降级；Git 工作树有未提交修改时拒绝更新；"
            "便携包同版本且没有可信 commit 基线时也拒绝覆盖。更新程序代码时不会删除 .venv、模型缓存、OCR 运行时、debug/logs、输出书籍或用户工作区。"
        )
        protection.setWordWrap(True)
        protection.setObjectName("settingsCardSubtitle")
        update_box.addWidget(protection)
        layout.addWidget(update_card)

        diag_card, diag_box = self._make_card(
            "诊断信息",
            "复制的内容只包含程序版本、平台、Python、程序目录和 Git 更新基线，不包含书籍正文、API 密钥或模型文件。",
        )
        diag_actions = QHBoxLayout()
        copy_diag = QPushButton("复制诊断信息")
        copy_diag.clicked.connect(self._copy_source_diagnostics)
        open_debug = QPushButton("打开诊断目录")
        open_debug.clicked.connect(self._open_debug_folder)
        diag_actions.addWidget(copy_diag)
        diag_actions.addWidget(open_debug)
        diag_actions.addStretch(1)
        diag_box.addLayout(diag_actions)
        self._source_diag_preview = QPlainTextEdit()
        self._source_diag_preview.setReadOnly(True)
        self._source_diag_preview.setMaximumHeight(150)
        try:
            base = diagnostic_summary(root)
        except Exception as exc:
            base = f"诊断摘要暂不可用：{exc}"
        self._source_diag_preview.setPlainText(
            base + f"\n平台: {platform.system()} {platform.release()} · {platform.machine()}"
            f"\nPython: {sys.version.split()[0]}"
        )
        self._source_diag_preview.setStyleSheet(LIGHT_LOG_STYLE)
        diag_box.addWidget(self._source_diag_preview)
        layout.addWidget(diag_card)
        layout.addStretch(1)
        return scroll

    def set_source_update_busy_provider(self, provider) -> None:
        self._source_update_busy_provider = provider if callable(provider) else None

    def source_update_in_progress(self) -> bool:
        return bool(self._source_update_check_busy or self._source_update_install_busy)

    def _external_source_update_busy(self) -> bool:
        try:
            return bool(self._source_update_busy_provider and self._source_update_busy_provider())
        except Exception:
            return True

    def _append_source_update_log(self, message: str) -> None:
        text = str(message or "").strip()
        if text and hasattr(self, "_source_update_log"):
            self._source_update_log.appendPlainText(text)

    def _check_source_update(self, _checked=False) -> None:
        if self.source_update_in_progress():
            return
        self._source_update_info = None
        self._source_update_check_busy = True
        self._source_check_btn.setEnabled(False)
        self._source_install_btn.setEnabled(False)
        self._source_update_status.setText("正在连接 GitHub 检查 main 分支版本和 commit…")
        self._source_remote_detail.setText("")
        self._append_source_update_log("开始检查 Git 更新…")
        signals = WorkerSignals()
        self._source_update_signals = signals
        signals.finished.connect(self._source_update_check_finished)
        signals.error.connect(self._source_update_failed)

        def worker():
            try:
                from utils.source_update import check_source_update
                result = check_source_update(project_root=PROJECT_ROOT)
                signals.finished.emit(result)
            except Exception as exc:
                signals.error.emit(f"{type(exc).__name__}: {exc}")

        thread = threading.Thread(target=worker, daemon=True, name="source-update-check")
        self._source_update_thread = thread
        thread.start()

    def _source_update_check_finished(self, info) -> None:
        self._source_update_check_busy = False
        self._source_update_thread = None
        self._source_update_signals = None
        self._source_update_info = info
        self._source_check_btn.setEnabled(True)
        self._source_update_status.setText(str(getattr(info, "reason", "检查完成")))
        local = str(getattr(info, "local_short", "未记录"))
        remote = str(getattr(info, "remote_short", ""))
        detail = (
            f"本地：v{getattr(info, 'local_version', VERSION)} · commit {local}\n"
            f"远端：v{getattr(info, 'remote_version', '未知')} · commit {remote} · {getattr(info, 'branch', 'main')}\n"
            f"安装形态：{getattr(info, 'install_layout', '未知')}"
        )
        message = str(getattr(info, "remote_message", "") or "")
        date = str(getattr(info, "remote_date", "") or "")
        if message:
            detail += f"\n最新提交：{message}" + (f" · {date}" if date else "")
        self._source_remote_detail.setText(detail)
        self._append_source_update_log(str(getattr(info, "reason", "检查完成")))
        self._source_install_btn.setEnabled(bool(getattr(info, "available", False)) and not self._external_source_update_busy())

    def _source_update_failed(self, message: str) -> None:
        was_install = self._source_update_install_busy
        self._source_update_check_busy = False
        self._source_update_install_busy = False
        self._source_update_thread = None
        self._source_update_signals = None
        self._source_check_btn.setEnabled(True)
        self._source_install_btn.setEnabled(False)
        self._source_update_status.setText("程序更新失败；现有版本保持不变。" if was_install else "检查更新失败；现有版本不受影响。")
        self._source_remote_detail.setText(str(message))
        self._append_source_update_log("失败：" + str(message))
        QMessageBox.critical(self, "程序更新失败" if was_install else "检查更新失败", str(message))

    def _install_source_update(self, _checked=False) -> None:
        if self.source_update_in_progress():
            return
        if self._external_source_update_busy() or self._device_detection_busy:
            notify(self, "请先等待当前 OCR、排版、导出、模型或设备检测任务结束，再更新程序。", "warning")
            return
        info = self._source_update_info
        if info is None:
            notify(self, "请先点击“检查 Git 更新”。", "warning")
            return
        if not bool(getattr(info, "available", False)):
            notify(self, str(getattr(info, "reason", "当前没有可安装更新")), "info")
            return
        answer = QMessageBox.question(
            self,
            "确认更新程序",
            ui_message(
                "app_update.confirm.body",
                repository=getattr(info, "repository", ""),
                branch=getattr(info, "branch", "main"),
                local=getattr(info, "local_version", VERSION),
                remote=getattr(info, "remote_version", ""),
                remote_short=getattr(info, "remote_short", ""),
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self._source_update_install_busy = True
        self._source_check_btn.setEnabled(False)
        self._source_install_btn.setEnabled(False)
        self._source_update_status.setText("正在安全更新程序代码…")
        signals = WorkerSignals()
        self._source_update_signals = signals
        signals.log.connect(self._append_source_update_log)
        signals.finished.connect(self._source_update_install_finished)
        signals.error.connect(self._source_update_failed)

        def worker():
            try:
                from utils.source_update import install_source_update
                result = install_source_update(info, progress=signals.log.emit)
                signals.finished.emit(result)
            except Exception as exc:
                signals.error.emit(f"{type(exc).__name__}: {exc}")

        thread = threading.Thread(target=worker, daemon=True, name="source-update-install")
        self._source_update_thread = thread
        thread.start()

    def _source_update_install_finished(self, result) -> None:
        self._source_update_install_busy = False
        self._source_update_thread = None
        self._source_update_signals = None
        self._source_update_info = None
        self._source_check_btn.setEnabled(True)
        self._source_install_btn.setEnabled(False)
        new_version = str(getattr(result, "new_version", ""))
        commit = str(getattr(result, "commit", ""))[:10]
        requirements_changed = bool(getattr(result, "requirements_changed", False))
        suffix = "；依赖清单有变化" if requirements_changed else ""
        self._source_update_status.setText(f"更新完成：v{new_version} · {commit}{suffix}。请重新启动程序。")
        self._append_source_update_log("更新完成；当前运行中的窗口仍使用旧内存代码，重启后加载新版本。")
        completion_key = (
            "app_update.complete_dependencies" if requirements_changed else "app_update.complete"
        )
        QMessageBox.information(
            self, "更新完成", ui_message(completion_key, version=new_version)
        )

    def _open_source_repository(self, _checked=False) -> None:
        QDesktopServices.openUrl(QUrl("https://github.com/Amster-Ilvil/Novel-formatter"))

    def _open_source_root(self, _checked=False) -> None:
        text = self._source_root.text() if hasattr(self, "_source_root") else str(PROJECT_ROOT)
        path = Path(text).expanduser().resolve()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _copy_source_diagnostics(self, _checked=False) -> None:
        try:
            from utils.source_update import diagnostic_summary
            text = diagnostic_summary(PROJECT_ROOT)
        except Exception as exc:
            text = f"Novel Formatter {VERSION}\n诊断摘要生成失败：{exc}"
        text += f"\n平台: {platform.system()} {platform.release()} · {platform.machine()}\nPython: {sys.version.split()[0]}"
        QApplication.clipboard().setText(text)
        if hasattr(self, "_source_diag_preview"):
            self._source_diag_preview.setPlainText(text)
        notify(self, "诊断信息已复制到剪贴板。", "success")

    def _build_about_page(self) -> QWidget:
        scroll, _page, layout = self._scroll_page()
        card, box = self._make_card("Novel Formatter 2.0")
        summary = QLabel(
            "日文书籍 OCR、逐列与逐字审校、文本修订、Formatter 和出版级 EPUB 导出的统一桌面工作台。"
        )
        summary.setWordWrap(True)
        box.addWidget(summary)
        version = QLabel(f"版本：{VERSION}")
        version.setObjectName("settingsCardSubtitle")
        box.addWidget(version)
        build_info = QLabel(f"Python：{sys.version.split()[0]} · 平台：{sys.platform}")
        build_info.setObjectName("settingsCardSubtitle")
        box.addWidget(build_info)
        layout.addWidget(card)

        principles, pbox = self._make_card("界面与功能原则")
        for text in (
            "七个主功能区；工作区独立存在，PDF、替换、OCR 对比和图文对照通过顶部页签融合。",
            "帮助文档入口不显示；程序与 OCR 模型更新仅在设置中由用户手动触发。",
            "OCR 开始按钮只位于 OCR 日志标题栏，不进入文字校对区。",
            "界面边框统一为淡蓝色，白色按钮文字统一为黑色。",
            "视觉与设置改造不改变 OCR、Formatter、替换、校对和导出数据链。",
        ):
            label = QLabel("• " + text)
            label.setWordWrap(True)
            pbox.addWidget(label)
        layout.addWidget(principles)
        layout.addStretch(1)
        return scroll

    def _load(self) -> None:
        self._restore_workspace_cb.setChecked(
            self._settings.value(self.RESTORE_KEY, False, type=bool)
        )
        self._remember_subtabs_cb.setChecked(
            self._settings.value(self.REMEMBER_SUBTABS_KEY, True, type=bool)
        )
        self._start_maximized_cb.setChecked(
            self._settings.value(self.START_MAXIMIZED_KEY, False, type=bool)
        )
        self._set_combo_data(
            self._language_combo,
            str(self._settings.value(self.LANGUAGE_KEY, LANG_ZH)),
        )
        self._set_combo_data(
            self._appearance_combo,
            str(self._settings.value(self.APPEARANCE_KEY, THEME_LIGHT)),
        )
        self._set_combo_data(
            self._default_section_combo,
            self._settings.value(self.DEFAULT_SECTION_KEY, SECTION_PAGE, type=int),
        )
        self._set_combo_data(
            self._ocr_engine_combo,
            str(self._settings.value(self.OCR_ENGINE_KEY, "apple_vision")),
        )
        self._set_combo_data(
            self._ocr_settings_tab_combo,
            self._settings.value(self.OCR_SETTINGS_TAB_KEY, 0, type=int),
        )
        self._ocr_review_default_cb.setChecked(
            self._settings.value(self.OCR_REVIEW_DEFAULT_KEY, False, type=bool)
        )
        self._ocr_review_expanded_cb.setChecked(
            self._settings.value(self.OCR_REVIEW_EXPANDED_KEY, True, type=bool)
        )
        self._ocr_preview_default_cb.setChecked(
            self._settings.value(self.OCR_PREVIEW_DEFAULT_KEY, True, type=bool)
        )
        self._ocr_progress_default_cb.setChecked(
            self._settings.value(self.OCR_PROGRESS_DEFAULT_KEY, True, type=bool)
        )

    @staticmethod
    def _set_combo_data(combo: QComboBox, value) -> None:
        index = combo.findData(value)
        if index < 0:
            try:
                index = combo.findData(int(value))
            except (TypeError, ValueError):
                index = -1
        combo.setCurrentIndex(max(0, index))

    def current_preferences(self) -> dict:
        return {
            "restore_workspace": self._restore_workspace_cb.isChecked(),
            "remember_subtabs": self._remember_subtabs_cb.isChecked(),
            "default_section": int(self._default_section_combo.currentData() or 0),
            "start_maximized": self._start_maximized_cb.isChecked(),
            "language": str(self._language_combo.currentData() or LANG_ZH),
            "appearance": str(self._appearance_combo.currentData() or THEME_LIGHT),
            "ocr_engine": str(self._ocr_engine_combo.currentData() or "apple_vision"),
            "ocr_settings_tab": int(self._ocr_settings_tab_combo.currentData() or 0),
            "ocr_review_default": self._ocr_review_default_cb.isChecked(),
            "ocr_review_expanded": self._ocr_review_expanded_cb.isChecked(),
            "ocr_preview_default": self._ocr_preview_default_cb.isChecked(),
            "ocr_progress_default": self._ocr_progress_default_cb.isChecked(),
        }

    def _save(self, _checked=False, *, show_message: bool = True) -> None:
        prefs = self.current_preferences()
        self._settings.setValue(self.RESTORE_KEY, prefs["restore_workspace"])
        self._settings.setValue(self.REMEMBER_SUBTABS_KEY, prefs["remember_subtabs"])
        self._settings.setValue(self.DEFAULT_SECTION_KEY, prefs["default_section"])
        self._settings.setValue(self.START_MAXIMIZED_KEY, prefs["start_maximized"])
        self._settings.setValue(self.LANGUAGE_KEY, prefs["language"])
        self._settings.setValue(self.APPEARANCE_KEY, prefs["appearance"])
        self._settings.setValue(self.OCR_ENGINE_KEY, prefs["ocr_engine"])
        self._settings.setValue(self.OCR_SETTINGS_TAB_KEY, prefs["ocr_settings_tab"])
        self._settings.setValue(self.OCR_REVIEW_DEFAULT_KEY, prefs["ocr_review_default"])
        self._settings.setValue(self.OCR_REVIEW_EXPANDED_KEY, prefs["ocr_review_expanded"])
        self._settings.setValue(self.OCR_PREVIEW_DEFAULT_KEY, prefs["ocr_preview_default"])
        self._settings.setValue(self.OCR_PROGRESS_DEFAULT_KEY, prefs["ocr_progress_default"])
        self._settings.sync()
        self.settings_changed.emit(dict(prefs))
        if show_message:
            notify(self, "设置已保存；OCR 默认行为已同步到当前工作区。", "success")

    def _reset_defaults(self) -> None:
        self._restore_workspace_cb.setChecked(False)
        self._remember_subtabs_cb.setChecked(True)
        self._start_maximized_cb.setChecked(False)
        self._set_combo_data(self._language_combo, LANG_ZH)
        self._set_combo_data(self._appearance_combo, THEME_LIGHT)
        self._set_combo_data(self._default_section_combo, SECTION_PAGE)
        self._set_combo_data(self._ocr_engine_combo, "apple_vision")
        self._set_combo_data(self._ocr_settings_tab_combo, 0)
        self._ocr_review_default_cb.setChecked(False)
        self._ocr_review_expanded_cb.setChecked(True)
        self._ocr_preview_default_cb.setChecked(True)
        self._ocr_progress_default_cb.setChecked(True)
        self._save(show_message=True)

    def _preview_interface_preferences(self, _index=0) -> None:
        """Preview language/theme immediately without persisting OCR settings."""
        manager = manager_for()
        if manager is None or not hasattr(self, "_language_combo"):
            return
        manager.apply_preferences(
            language=str(self._language_combo.currentData() or LANG_ZH),
            theme=str(self._appearance_combo.currentData() or THEME_LIGHT),
        )

    def apply_to_ocr_tab(self, tab) -> None:
        """Apply persisted OCR defaults without changing OCR business methods."""
        prefs = self.current_preferences()
        engine_combo = getattr(tab, "_adapter_combo", None)
        if engine_combo is not None:
            index = engine_combo.findData(prefs["ocr_engine"])
            if index >= 0:
                engine_combo.setCurrentIndex(index)
        settings_tabs = getattr(tab, "_ocr_settings_tabs", None)
        if settings_tabs is not None:
            settings_tabs.setCurrentIndex(max(0, min(settings_tabs.count() - 1, prefs["ocr_settings_tab"])))
        preview = getattr(tab, "_preview_enabled_cb", None)
        if preview is not None:
            preview.setChecked(bool(prefs["ocr_preview_default"]))
        progress = getattr(tab, "_progress_display_cb", None)
        if progress is not None:
            progress.setChecked(bool(prefs["ocr_progress_default"]))
        review = getattr(tab, "_handwriting_trace_check", None)
        if review is not None:
            review.setChecked(bool(prefs["ocr_review_default"]))
        expanded = getattr(tab, "_handwriting_card_toggle", None)
        if expanded is not None:
            expanded.setChecked(True)

    def restore_workspace_enabled(self) -> bool:
        return self._settings.value(self.RESTORE_KEY, False, type=bool)

    def remember_subtabs_enabled(self) -> bool:
        return self._settings.value(self.REMEMBER_SUBTABS_KEY, True, type=bool)

    def default_section_index(self) -> int:
        return max(SECTION_WORKSPACE, min(SECTION_SYSTEM, self._settings.value(self.DEFAULT_SECTION_KEY, SECTION_PAGE, type=int)))

    def start_maximized_enabled(self) -> bool:
        return self._settings.value(self.START_MAXIMIZED_KEY, False, type=bool)

    def _open_ocr_model_updates(self, _checked=False) -> None:
        OCRModelUpdateDialog(self).exec()

    def _open_temp_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(tempfile.gettempdir()).resolve())))

    def _open_debug_folder(self) -> None:
        path = PROJECT_ROOT / "debug"
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


__all__ = ["OCRModelUpdateDialog", "SystemSettingsTab"]
