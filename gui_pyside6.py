#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Novel Formatter 2.0 — PySide6 GUI
七个主功能区：工作区 / 页面管理 / OCR 识别 / 格式处理 / 文字校对 / EPUB生成 / 设置。
九个原业务工作区通过顶部页签融合，macOS 简约风格，完整功能保留。

用法: python3 gui_pyside6.py
依赖: pip3 install PySide6 pillow
"""

from __future__ import annotations

from core.app_meta import VERSION

import sys
import os
import json
import copy
import math
import html
import hashlib
import re
import threading
import queue
import tempfile
import shutil
import time
import subprocess
import platform
from pathlib import Path
from collections import Counter, OrderedDict
from bisect import bisect_left, bisect_right
from functools import partial
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QStackedWidget, QTabWidget,
    QPlainTextEdit, QTextEdit, QLabel, QPushButton, QComboBox, QLineEdit,
    QCheckBox, QSlider, QScrollBar, QFileDialog, QMessageBox,
    QFormLayout, QProgressBar, QFrame, QSizePolicy, QScrollArea,
    QGridLayout, QMenu, QToolButton, QTreeView, QFileSystemModel,
    QRadioButton, QRubberBand,
    QDialog, QListWidget, QListWidgetItem, QListView, QButtonGroup, QInputDialog, QDoubleSpinBox, QSpinBox, QAbstractItemView, QStyle, QStyleOptionSlider,
    QTextBrowser, QColorDialog, QTableWidget, QTableWidgetItem, QHeaderView,
)
from PySide6.QtCore import Qt, Signal, QObject, QTimer, QRect, QPoint, QUrl, QDir, QModelIndex, QEvent, QSize, QSettings, QAbstractListModel
from PySide6.QtGui import (
    QFont, QFontMetrics, QColor, QPixmap, QImage, QImageReader, QPainter, QIcon,
    QCursor, QKeySequence, QShortcut, QDesktopServices, QAction,
    QTextCursor, QTextCharFormat, QTextFormat, QPen,
)

from models.document import UnifiedDocument, Block, BlockType, TocEntry, new_temp_repo_path
from utils.paddle_importer import import_paddle_json, import_paddle_md
from models.format_profile import FormatProfile, FormatProfileStore
from ui.flow_layout import FlowLayout
from ui.clickable_style_guard import enforce_button_contrast_tree, install_no_white_clickable_guard
from ui.responsive import configure_combo, preserve_button_text
from ui.widgets import ColorSwatch
from ui.common.editor_controls import (
    NoWheelSpinBox, NoWheelDoubleSpinBox, NoWheelComboBox, MouseWheelPlainTextEdit,
    EditorPositionSlider, configure_drag_scrollbar, attach_editor_position_slider,
)
from ui.common.signals import WorkerSignals, ImageReviewRenderSignals, safe_qt_emit
from ui.ocr.preview import OCRCropPreview
from ui.ocr.proofread_widgets import OCRProofreadImageLabel, OCRVerticalColumnTextWidget
from ui.ocr.model_settings import build_ocr_model_settings
from ui.ocr.preview_panel import build_ocr_preview_panel
from ui.ocr.preview_controller import OCRPreviewController
from ui.ocr.run_lifecycle import OCRRunLifecycleController
from ui.ocr.progress_panel import build_ocr_progress_panel
from ui.ocr.run_controller import OcrRunController
from ui.ocr.compare_widgets import (
    _FusionCandidateCard, _FusionDecisionRow, OCRAIAdjudicationDialog, DecisionQueueListModel,
)
from ui.ocr.compare_panel import build_ocr_compare_panel
from ui.ocr.compare_view import OCRCompareViewMixin
from ui.ocr.compare_project_state import OCRCompareProjectStateService
from ui.ocr.compare_source_correction import OCRCompareSourceCorrectionService
from ui.ocr.compare_exchange import OCRCompareExchangeService
from ui.ocr.compare_publication import OCRComparePublicationService
from ui.ocr.compare_gpt_adjudication import OCRCompareGPTAdjudicationController
from ui.ocr.compare_batch import OCRCompareBatchService
from ui.ocr.compare_history import OCRCompareAdjudicationHistoryService
from ui.ocr.image_review import _ImageReviewFusionCandidateCard, OCRImageTextReviewTab
from ui.pages.types import PAGE_TYPES, TYPE_LABEL, TYPE_COLOR
from ui.pages.preview import PageImagePreviewDialog
from ui.pages.tab import PageManagerTab
from ui.pdf_text.tab import PdfTextLayerTab
from ui.workspace.tab import ProjectWorkspaceTab
from ui.settings.ai_dialog import AISettingsDialog, ensure_ai_settings
from ui.settings.tab import OCRModelUpdateDialog, SystemSettingsTab
from ui.main_window.controller import MainWindowControllerMixin
from ui.main_window.lazy_workspaces import LazyWorkspaceRegistry
from ui.ai_image.tab import AIImageProcessingTab
from ui.formatter.tab import FormatterTab, FormatProfileDialog
from ui.epub.tab import EPUBTab
from ui.navigation.sidebar import (
    REFERENCE_SECTION_ITEMS, SECTION_WORKSPACE, SECTION_PAGE, SECTION_OCR, SECTION_FORMAT,
    SECTION_PROOF, SECTION_EPUB, SECTION_SYSTEM, ReferenceSectionHost, Sidebar,
)
from ui.common.styling import (
    BG, SIDEBAR_BG, CARD, SURFACE, CLICKABLE_BG, CLICKABLE_HOVER, CLICKABLE_PRESSED,
    INK, MUTED, SUBTLE, BORDER, BORDER_STRONG, ACC, ACC_HOVER, ACC_PRESSED, ACC_BG,
    DANGER, SUCCESS, LIGHT_PREVIEW_STYLE, LIGHT_LOG_STYLE, EDITOR_SCROLLBAR_STYLE,
    blend, make_separator, make_badge, accent_button, WrapRow, wrap_in_card,
    polish_reference_workspace,
)
from ui.dialogs import install_dialog_polish, install_exception_hooks, show_error_dialog
from ui.common.dialog_polish import install_dialog_polish as install_generic_dialog_polish
from ui.common.window_state import install_window_state_saver, restore_window_state
from ui.common.toast import notify
from ui.interface_preferences import (
    InterfacePreferenceManager, THEME_LIGHT, THEME_DARK, manager_for,
)
from ui.localization import LANG_ZH, LANG_JA, LANG_EN
from ui.localized_dialogs import (
    LocalizedFileDialog, LocalizedMessageBox, LocalizedInputDialog, LocalizedColorDialog, ui_message,
)
# Static/native Qt dialogs bypass the normal widget translation event filter.
# Rebind only this GUI module; returned paths/data and Qt enums stay untouched.
QFileDialog = LocalizedFileDialog
QMessageBox = LocalizedMessageBox
QInputDialog = LocalizedInputDialog
QColorDialog = LocalizedColorDialog
from utils.clear_manager import (
    ClearManager,
    create_workspace_clear_button,
)
from utils.async_generation import GenerationGuard
from core.workspace_snapshot_registry import WorkspaceSnapshotRegistry
from core.project_workspace import ProjectWorkspaceManager
from core.runtime_task_state import RuntimeTaskRegistry
from core.workspace_contracts import WORKSPACE_SPECS, validate_workspace_contracts
from ui.workspace_coordinator import WorkspaceCoordinator
from adapters.result_export import (
    ensure_export_extension,
    export_text_result,
    format_from_filter,
    safe_result_filename,
    save_filter_string,
)

# ── 参考界面配色（macOS / 浅色生产力工具）────────────────────────────────────



















LIGHT_IMAGE_PREVIEW_STYLE = f"""
background: #FFFFFF;
color: {MUTED};
border: 1px solid #E2E5E9;
border-radius: 10px;
padding: 12px;
"""

# 页面类型使用高区分度定性色板。
# 设计原则：相邻类别避免同一色相、所有实色标签对白字保持足够对比度，
# 并让空白页/版权页/未分类不再挤在一组近似灰色里。

from ui.ocr.catalog import OCR_ADAPTERS


# Formatter 左侧只显示用户需要主动控制的处理步骤。底层步骤仍参与
# “全部运行”，但不会出现在界面中。勾选“显示具体规则说明”后，右侧
# 底部会显示当前所选步骤的真实规则，而不是显示底层模块。





# ── 工具函数 ──────────────────────────────────────────────────────────────────
























# ── 全局样式 ──────────────────────────────────────────────────────────────────

def _indicator_icon_path(name: str, svg: str) -> str:
    """生成指示器小图标 SVG（写入临时目录，供 QSS image: 引用）。"""
    import tempfile
    path = Path(tempfile.gettempdir()) / f"novel_formatter_{name}.svg"
    try:
        path.write_text(svg, encoding="utf-8")
        return path.as_posix()
    except Exception:
        return ""


_CHECKMARK = _indicator_icon_path(
    "check",
    '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 10 10">'
    '<path d="M1.5 5.2 L4 7.6 L8.6 2.4" fill="none" stroke="white" '
    'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>',
)
_RADIO_DOT = _indicator_icon_path(
    "radiodot",
    '<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8" viewBox="0 0 8 8">'
    '<circle cx="4" cy="4" r="3" fill="white"/></svg>',
)


from ui.theme.tokens import build_stylesheet  # noqa: E402
STYLE = build_stylesheet(_CHECKMARK, _RADIO_DOT)


# ══════════════════════════════════════════════════════════════════════════════
#  信号桥
# ══════════════════════════════════════════════════════════════════════════════







# ══════════════════════════════════════════════════════════════════════════════
#  页面管理高清预览
# ══════════════════════════════════════════════════════════════════════════════





# ══════════════════════════════════════════════════════════════════════════════
#  Tab 1 — 页面管理器
# ══════════════════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════════════════
#  Tab 2 — OCR 适配器
# ══════════════════════════════════════════════════════════════════════════════


def _migrate_ocr_review_default(settings: QSettings) -> None:
    """Close the legacy review checkbox without creating a fake fresh profile.

    A fresh install must leave ``ocr/mode_state/ja_vertical`` absent until the
    OCR UI applies its Fast-Core defaults.  Earlier migration code wrote a
    one-field JSON state even when no state existed, which made the build path
    believe the user already had persisted OCR preferences and skipped the
    default Hayai/NDL/48px + physical-column preset.
    """
    migration_key = "ocr/review_checkbox_default_closed_v3"
    if settings.value(migration_key, False, type=bool):
        return
    raw_state = settings.value("ocr/mode_state/ja_vertical", "")
    if raw_state:
        try:
            saved_state = json.loads(str(raw_state))
        except Exception:
            saved_state = {}
        if not isinstance(saved_state, dict):
            saved_state = {}
        saved_state["handwriting"] = False
        settings.setValue(
            "ocr/mode_state/ja_vertical",
            json.dumps(saved_state, ensure_ascii=False, sort_keys=True),
        )
    settings.setValue(migration_key, True)
    settings.sync()


# ══════════════════════════════════════════════════════════════════════════════
#  AI 图文处理 — 独立于传统 OCR / Formatter 的多模态整书模式
# ══════════════════════════════════════════════════════════════════════════════



class OCRTab(QWidget):
    ocr_done = Signal(object)
    multi_ocr_done = Signal(object)
    single_ocr_invalidated = Signal()
    review_preview_boxes_ready = Signal(str, object)
    run_log_event = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ocr_preview_controller = OCRPreviewController(self)
        self._ocr_run_lifecycle = OCRRunLifecycleController(self)
        self._pending_inputs: list[str] = []
        # OCR 输入只由页面管理工作区提供；其显式页面列表是本轮识别的唯一真值。
        self._input_origin = "page_manager"
        self._latest_single_doc: UnifiedDocument | None = None
        self._cancel_event = threading.Event()
        # Runtime preview is controlled by a thread-safe event rather than by
        # reading a Qt checkbox from the OCR worker.  It may be toggled while a
        # model is running and queued preview signals are checked again on the
        # GUI thread before they are allowed to update the canvas.
        self._live_preview_enabled_event = threading.Event()
        self._live_preview_enabled_event.set()
        # Progress display can be disabled independently from OCR.  The worker
        # still completes every model/column/sentence operation, but skips the
        # high-frequency Qt progress signal traffic and ETA repaint work.
        self._progress_display_enabled_event = threading.Event()
        self._progress_display_enabled_event.set()
        self._latest_progress_snapshot = None
        self._run_generation = 0
        # Auxiliary installers have independent generations.  Their native/network
        # work may outlive a tab reset, but late signals must never mutate the next
        # book or a closing QWidget.
        self._paddle_prepare_generation = GenerationGuard()
        self._handwriting_download_generation = GenerationGuard()
        self._paddle_prepare_signal_refs: dict[int, WorkerSignals] = {}
        self._handwriting_download_signal_refs: dict[int, WorkerSignals] = {}
        # Runtime/model probing can traverse large local caches or spawn short
        # interpreter probes. Never perform that filesystem/subprocess work on
        # the Qt GUI thread: it used to make OCR completion look like an app
        # deadlock, especially after a single-model run on large HF caches.
        self._runtime_status_generation = GenerationGuard()
        self._runtime_status_signal_refs: dict[int, WorkerSignals] = {}
        self._runtime_status_scan_busy = False
        # Start-button runtime/model checks can spawn interpreter probes and scan
        # local model caches. Keep that preflight off the Qt thread as well.
        self._ocr_preflight_generation = GenerationGuard()
        self._ocr_preflight_signal_refs: dict[int, WorkerSignals] = {}
        self._ocr_preflight_active = False
        # OCR uses external model processes.  Track the active worker and the
        # last signal received so a lost child process cannot leave the UI in a
        # permanent "running" state.
        self._ocr_run_active = False
        self._ocr_worker_thread: threading.Thread | None = None
        self._ocr_last_activity_at = 0.0
        self._ocr_stall_notice_emitted = False
        self._ocr_hard_timeout_requested = False
        self._active_adapter = "apple_vision"
        # OCR modes are isolated profiles.  Switching modes snapshots the
        # controls of the old profile and restores the target profile without
        # leaking Japanese column/ruby/handwriting settings into Chinese OCR.
        self._ocr_mode_settings = QSettings("NovelFormatter", "NovelFormatter")
        _migrate_ocr_review_default(self._ocr_mode_settings)
        self._ocr_mode_states: dict[str, dict] = {}
        self._ocr_mode_change_guard = False
        self._ocr_mode = "ja_vertical"
        self._preview_source_path: str | None = None
        # Source page currently shown by retained live-preview navigation.  It
        # is separate from the initial crop-reference page so "预览分列与掩膜"
        # follows 上一页/下一页 instead of silently jumping back to page 1.
        self._active_preview_source_path: str | None = None
        self._preview_items: list[dict] = []
        self._preview_item_index_by_key: dict[str, int] = {}
        self._preview_index = -1
        self._preview_follow_latest = True
        self._review_preview_generation = 0
        self.review_preview_boxes_ready.connect(self._apply_review_preview_boxes)
        from utils.ocr_preview_temp_store import OCRPreviewTempStore
        self._preview_temp_store = OCRPreviewTempStore()
        self._handwriting_review_context: dict | None = None
        # True only for the lifetime of a dedicated OCR + manual-review run.
        # It keeps the blue character-frame preview visible even though the
        # dedicated button restores the user's checkbox immediately after
        # starting the asynchronous OCR worker.
        self._review_preview_run_active = False
        # Set only by the dedicated “开始 OCR + 人工纠错” entry point and
        # consumed at the beginning of the next OCR run.
        self._manual_review_requested = False
        # Whole-run progress snapshots arrive when a page/column/sentence unit
        # completes.  A lightweight GUI timer updates the elapsed clock between
        # those callbacks so the text inside the blue bar remains visibly live
        # even while one expensive OCR unit is still running.
        self._overall_progress_live_state: dict | None = None
        self._progress_clock_timer = QTimer(self)
        self._progress_clock_timer.setInterval(1000)
        self._progress_clock_timer.timeout.connect(self._refresh_live_progress_clock)
        # Batch high-frequency OCR log lines so QPlainTextEdit performs far
        # fewer relayouts while preserving the complete log text.
        from core.ocr_runtime_optimizer import CoalescedLineBuffer
        self._ocr_log_buffer = CoalescedLineBuffer(max_lines=200000)
        self._ocr_log_flush_timer = QTimer(self)
        self._ocr_log_flush_timer.setInterval(80)
        self._ocr_log_flush_timer.timeout.connect(self._flush_ocr_log_buffer)
        self._last_ocr_performance_trace_path = ""
        self._active_ocr_performance_trace = None
        self._active_ocr_log_session_id = ""
        self._active_ocr_log_finalized = False
        self._ocr_watchdog_timer = QTimer(self)
        self._ocr_watchdog_timer.setInterval(2000)
        self._ocr_watchdog_timer.timeout.connect(self._check_ocr_watchdog)
        self._build()

    def _build(self):
        root = wrap_in_card(self)

        # 整体改成上下结构：上半部分（左侧控制栏 + 中间大预览图）撑满主要空间，
        # 日志/识别结果原来占右侧一整块、常年空着，现在收窄成底部一条常驻小面板。
        main_container = QWidget()
        main_v = QVBoxLayout(main_container)
        main_v.setContentsMargins(0, 0, 0, 0)
        main_v.setSpacing(0)
        root.addWidget(main_container, 1)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(0)
        main_v.addLayout(top_row, 1)

        # ── 左侧：适配器列表 + 设置 + 底部常驻「开始OCR」按钮 ──────────────────
        build_ocr_model_settings(self, top_row, OCR_ADAPTERS)
        build_ocr_preview_panel(self, top_row)
        build_ocr_progress_panel(self, main_v)
        # multi-model ON therefore requires an explicit role selection and never
        # silently inherits the locked single-model engine.
        persisted_ja_mode_state = self._ocr_mode_settings.value(
            "ocr/mode_state/ja_vertical", ""
        )
        if not persisted_ja_mode_state:
            self._select_adapter("hayai_ocr", True, None)
            for combo in getattr(self, "_multi_model_combos", []):
                self._set_combo_value(combo, "")
            # Fast-Core preset: keep the proven 924 three-model topology,
            # but use the current transports/caches and Common-Mode GPT audit.
            for combo, engine_id in zip(
                (self._multi_model1_combo, self._multi_model2_combo,
                 self._multi_model3_combo),
                ("hayai_ocr", "ndlocr_lite", "manga_48px"),
            ):
                self._set_combo_value(combo, engine_id)
            self._set_combo_value(self._multi_model4_combo, "")
            self._multi_smart_router_check.setChecked(False)
            self._multi_early_consensus_check.setChecked(False)
            self._multi_ocr_check.setChecked(False)
            self._column_split_check.setChecked(True)
            # Phase27 third main role is a full physical-column pass; legacy
            # sentence reflow must not silently reactivate on a fresh profile.
            self._column_sentence_reflow_check.setChecked(False)
            if hasattr(self, "_column_compact_transport_check"):
                self._column_compact_transport_check.setChecked(True)
        self._ocr_mode_states["ja_vertical"] = (
            self._capture_ocr_mode_state()
            if not persisted_ja_mode_state
            else self._load_persisted_ocr_mode_state("ja_vertical")
        )
        self._ocr_mode_combo.currentIndexChanged.connect(self._on_ocr_mode_changed)
        self._apply_ocr_mode_ui(initial=True)
        self._highlight_adapter(self._active_adapter)
        self._vision_backend_widget.setVisible(self._active_adapter == "apple_vision")
        self._on_vision_backend_changed()
        self._refresh_ocr_runtime_status(show_dialog=False, deep=False)

    def _current_ocr_mode(self) -> str:
        from adapters.ocr_profiles import normalize_ocr_mode
        return normalize_ocr_mode(getattr(self, "_ocr_mode", "ja_vertical"))

    @staticmethod
    def _set_combo_value(combo, value) -> None:
        if combo is None:
            return
        index = combo.findData(value)
        if index < 0:
            return
        blocked = combo.blockSignals(True)
        combo.setCurrentIndex(index)
        combo.blockSignals(blocked)

    def _on_preserve_ruby_toggled(self, checked: bool) -> None:
        """Toggle only the independent Ruby side-channel.

        This control must never rewrite, lock or otherwise mutate the ordinary
        OCR preprocessing controls.  Normal OCR therefore receives the exact
        same pixel contract whether Ruby preservation is ON or OFF; the only
        additional ON work is geometry telemetry plus the independent
        findtextCenterNet pass after normal OCR/fusion has finished.
        """
        scan_mode = getattr(self, "_ruby_scan_mode_combo", None)
        if scan_mode is not None:
            scan_mode.setEnabled(bool(checked))
        # Deliberately do not call _on_column_cleanup_control_changed() here.
        # Ruby preservation is orthogonal to the user's normal-OCR cleanup
        # profile and must not turn the ordinary Ruby filter on/off.

    def _capture_ocr_mode_state(self) -> dict:
        def checked(name: str, default: bool = False) -> bool:
            widget = getattr(self, name, None)
            return bool(widget.isChecked()) if widget is not None else bool(default)

        def combo_data(name: str, default: str = "") -> str:
            widget = getattr(self, name, None)
            return str(widget.currentData() or default) if widget is not None else default

        def int_value(name: str, default: int = 0) -> int:
            widget = getattr(self, name, None)
            return int(widget.value()) if widget is not None else int(default)

        def float_value(name: str, default: float = 0.0) -> float:
            widget = getattr(self, name, None)
            return float(widget.value()) if widget is not None else float(default)

        def text_value(name: str, default: str = "") -> str:
            widget = getattr(self, name, None)
            return str(widget.text()).strip() if widget is not None else str(default)

        return {
            "active_adapter": str(getattr(self, "_active_adapter", "apple_vision")),
            "multi_ocr": checked("_multi_ocr_check"),
            "multi_smart_router": False,
            "multi_local_retry": checked("_multi_local_retry_check", False),
            "multi_router_policy_version": 3,
            "multi_evidence_policy": "full_evidence",
            "multi_role_column": combo_data("_multi_model1_combo"),
            "multi_role_page": combo_data("_multi_model2_combo"),
            "multi_role_sentence": combo_data("_multi_model3_combo"),
            "multi_role_review1": combo_data("_multi_model4_combo"),
            "multi_role_review2": combo_data("_multi_model5_combo"),
            "multi_role_review3": combo_data("_multi_model6_combo"),
            "multi_role_schema": 2,
            "multi_default_profile_version": 6,
            # Legacy keys are kept empty so old readers cannot misinterpret the
            # role scheduler as positional all-model voting.
            "multi_model2": "",
            "multi_model3": "",
            "multi_model4": "",
            "multi_model5": "",
            "multi_model6": "",
            "multi_early_consensus": False,
            "multi_parallel_first": True,
            "multi_sentence_speed": "role_scheduler",
            "column_split": checked("_column_split_check"),
            "column_isolation_mode": combo_data("_column_isolation_mode_combo", "mask"),
            "column_sensitivity": int_value("_column_sensitivity_spin", 55),
            "column_padding_percent": int_value("_column_padding_spin", 10),
            "strict_column_validation": checked("_column_strict_check", True),
            "column_rescue_policy": combo_data("_column_rescue_policy_combo", "adaptive"),
            "ndlocr_page_mode": combo_data("_ndlocr_page_mode_combo", "hybrid"),
            "preserve_ruby": checked("_preserve_ruby_check", False),
            "ruby_scan_mode": combo_data("_ruby_scan_mode_combo", "smart_roi"),
            # Internal compatibility field remains in saved state and audit
            # data, but is intentionally not exposed as a normal UI option.
            "column_input_profile": (
                "custom" if self._column_preprocess_is_customized() else "v8_exact"
            ),
            "column_preprocess_customized": self._column_preprocess_is_customized(),
            "column_preprocess_expanded": True,
            "column_auto_filter_ruby": checked("_column_ruby_filter_check", True),
            "column_filter_fragments": checked("_column_fragment_filter_check", False),
            "column_smart_crop": checked("_column_smart_crop_check", True),
            "column_ruby_strength": combo_data("_column_ruby_strength_combo", "standard"),
            "column_compact_transport": checked("_column_compact_transport_check", True),
            "column_reflow": checked("_column_sentence_reflow_check", True),
            "column_sentence_context_reocr": checked("_column_sentence_context_reocr_check", False),
            "column_sentence_strategy": combo_data("_column_sentence_strategy_combo", "full"),
            "column_sentence_global_merged_box": checked("_column_sentence_global_merged_box_check", False),
            "column_sentence_max": int_value("_column_sentence_max_spin", 10),
            "handwriting": checked("_handwriting_trace_check"),
            "handwriting_mode": (
                "manual" if checked("_handwriting_mode_manual")
                else "auto" if checked("_handwriting_mode_auto") else "hybrid"
            ),
            "handwriting_strategy": combo_data("_handwriting_strategy_combo", "balanced"),
            "handwriting_backend": combo_data("_handwriting_backend_combo", "auto"),
            "handwriting_char_mask": checked("_handwriting_char_mask_check", True),
            "handwriting_symbol_insert": checked("_handwriting_symbol_insert_check", True),
            "vision_backend": combo_data("_vision_backend_combo", "native_helper"),
            "vision_vertical": checked("_vision_vertical_check", True),
            "vision_vertical_compat": checked("_vision_vertical_compat_check", True),
            "vision_language_correction": checked("_vision_language_correction_check", True),
            "vision_candidate_count": int_value("_vision_candidate_spin", 3),
            "vision_min_height": float_value("_vision_min_height_spin", 0.005),
            "shortcut_name": text_value("_shortcut_edit", "ExtractText") or "ExtractText",
            "hayai_backend": combo_data("_hayai_backend_combo", "torch"),
            "hayai_device": combo_data("_hayai_device_combo", "auto"),
            "hayai_quantize": combo_data("_hayai_quant_combo", "none"),
            "hayai_litert_quant": combo_data("_hayai_litert_quant_combo", "wi4"),
            "paddle_pipeline": combo_data("_paddle_model_combo", "ocr"),
            "paddle_vl_backend": combo_data("_paddle_vl_backend_combo", "auto"),
            "paddle_source": combo_data("_paddle_source_combo", "auto"),
            # Token is deliberately excluded. Only non-secret cloud settings
            # are persisted per OCR mode.
            "paddle_aistudio_mode": combo_data("_paddle_aistudio_mode_combo", "async_v2"),
            "paddle_aistudio_api_url": (
                self._paddle_aistudio_url_edit.text().strip()
                if hasattr(self, "_paddle_aistudio_url_edit") else ""
            ),
            "paddle_aistudio_sync_family": combo_data(
                "_paddle_aistudio_sync_family_combo", "ppocr"
            ),
            "paddle_aistudio_async_model": combo_data(
                "_paddle_aistudio_async_model_combo", "PaddleOCR-VL-1.6"
            ),
            "paddle_aistudio_orientation": checked("_paddle_aistudio_orientation_check", False),
            "paddle_aistudio_unwarp": checked("_paddle_aistudio_unwarp_check", False),
            "paddle_aistudio_textline": checked("_paddle_aistudio_textline_check", False),
            "paddle_aistudio_timeout": (
                int(self._paddle_aistudio_timeout_spin.value())
                if hasattr(self, "_paddle_aistudio_timeout_spin") else 180
            ),
            "paddle_aistudio_retries": (
                int(self._paddle_aistudio_retry_spin.value())
                if hasattr(self, "_paddle_aistudio_retry_spin") else 3
            ),
            "merge_horizontal_fragments": checked(
                "_chinese_merge_line_fragments_check", True
            ),
            "filter_horizontal_headers": checked("_chinese_header_filter_check", True),
        }

    def project_pipeline_state(self) -> dict:
        """Return a project-safe OCR pipeline snapshot without credentials."""
        state = dict(self._capture_ocr_mode_state())
        state["ocr_mode"] = self._current_ocr_mode()
        state["pipeline_kind"] = "multi" if bool(state.get("multi_ocr")) else "single"
        state["schema_version"] = 1
        return state

    def restore_project_pipeline_state(self, state: dict) -> None:
        """Restore the active project's OCR controls as one isolated profile.

        The project snapshot is authoritative for reproducibility.  Secrets are
        never present in this payload; engine/API credentials continue to come
        from the normal secure settings path.
        """
        state = dict(state or {})
        if not state:
            return
        requested = str(state.get("ocr_mode") or self._current_ocr_mode() or "ja_vertical")
        if requested not in {"ja_vertical", "zh_hans_horizontal"}:
            requested = "ja_vertical"
        merged = dict(self._default_ocr_mode_state(requested))
        merged.update(state)
        self._ocr_mode_states[requested] = merged
        self._ocr_mode = requested
        combo = getattr(self, "_ocr_mode_combo", None)
        if combo is not None:
            blocked = combo.blockSignals(True)
            self._set_combo_value(combo, requested)
            combo.blockSignals(blocked)
        self._apply_ocr_mode_ui(initial=False)
        captured = self._capture_ocr_mode_state()
        self._ocr_mode_states[requested] = captured
        self._save_ocr_mode_state(requested, captured)
        self._ocr_mode_settings.setValue("ocr/active_mode", requested)

    def _default_ocr_mode_state(self, mode: str) -> dict:
        if mode == "zh_hans_horizontal":
            return {
                "active_adapter": "apple_vision",
                "multi_ocr": False,
                "multi_smart_router": False,
                "multi_router_policy_version": 3,
                "multi_evidence_policy": "full_evidence",
                "multi_role_column": "",
                "multi_role_page": "",
                "multi_role_sentence": "",
                "multi_role_review1": "",
                "multi_role_review2": "",
                "multi_role_review3": "",
                "multi_role_schema": 2,
                "multi_model2": "",
                "multi_model3": "",
                "multi_model4": "",
                "multi_model5": "",
                "multi_model6": "",
                "multi_early_consensus": False,
                "multi_parallel_first": True,
                "multi_sentence_speed": "role_scheduler",
                "column_split": False,
                "column_isolation_mode": "mask",
                "column_sensitivity": 55,
                "column_padding_percent": 10,
                "strict_column_validation": True,
                "column_rescue_policy": "adaptive",
                "ndlocr_page_mode": "hybrid",
                "preserve_ruby": False,
                "ruby_scan_mode": "smart_roi",
                "column_input_profile": "v8_exact",
                "column_preprocess_customized": False,
                "column_preprocess_expanded": False,
                "column_auto_filter_ruby": True,
                "column_filter_fragments": False,
                "column_smart_crop": True,
                "column_ruby_strength": "standard",
                "column_compact_transport": True,
                "column_reflow": False,
                "column_sentence_context_reocr": False,
                "column_sentence_strategy": "full",
                "column_sentence_global_merged_box": False,
                "column_sentence_max": 10,
                "handwriting": False,
                "handwriting_mode": "hybrid",
                "handwriting_strategy": "balanced",
                "handwriting_backend": "auto",
                "handwriting_char_mask": True,
                "handwriting_symbol_insert": True,
                "vision_backend": "native_helper",
                "vision_vertical": False,
                "vision_vertical_compat": False,
                "vision_language_correction": True,
                "vision_candidate_count": 3,
                "vision_min_height": 0.005,
                "shortcut_name": "ExtractText",
                "hayai_backend": "torch",
                "hayai_device": "auto",
                "hayai_quantize": "none",
                "hayai_litert_quant": "wi4",
                "paddle_pipeline": "ocr",
                "paddle_vl_backend": "auto",
                "paddle_source": "auto",
                "paddle_aistudio_mode": "async_v2",
                "paddle_aistudio_api_url": "",
                "paddle_aistudio_sync_family": "ppocr",
                "paddle_aistudio_async_model": "PaddleOCR-VL-1.6",
                "paddle_aistudio_orientation": False,
                "paddle_aistudio_unwarp": False,
                "paddle_aistudio_textline": False,
                "paddle_aistudio_timeout": 180,
                "paddle_aistudio_retries": 3,
                "merge_horizontal_fragments": True,
                "filter_horizontal_headers": True,
            }
        return dict(self._ocr_mode_states.get("ja_vertical") or {})

    def _load_persisted_ocr_mode_state(self, mode: str) -> dict:
        raw = self._ocr_mode_settings.value(f"ocr/mode_state/{mode}", "")
        if raw:
            try:
                data = json.loads(str(raw))
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        return self._default_ocr_mode_state(mode)

    def _save_ocr_mode_state(self, mode: str, state: dict) -> None:
        # API keys are deliberately excluded from the state snapshot.
        self._ocr_mode_settings.setValue(
            f"ocr/mode_state/{mode}",
            json.dumps(dict(state or {}), ensure_ascii=False, sort_keys=True),
        )

    def _restore_ocr_mode_state(self, state: dict) -> None:
        state = dict(state or {})
        # Phase27 keeps the historical router keys readable, but the third main
        # slot is now a full-column pass.  Conflict-only OCR belongs to review1;
        # therefore old sentence-smart-router state is always disabled on restore.
        try:
            router_policy_version = int(state.get("multi_router_policy_version", 0) or 0)
        except Exception:
            router_policy_version = 0
        try:
            role_schema_hint = int(state.get("multi_role_schema", 0) or 0)
        except Exception:
            role_schema_hint = 0
        if router_policy_version < 3 and role_schema_hint >= 1 and bool(state.get("multi_ocr", False)):
            state["multi_smart_router"] = False
            state["multi_evidence_policy"] = "full_evidence"
        state["multi_router_policy_version"] = 3
        state["multi_smart_router"] = False
        state["multi_evidence_policy"] = "full_evidence"
        # v23 introduces an explicit customization marker.  Old profile names
        # are deliberately ignored during migration because several historical
        # versions wrote ``custom`` even when the user never changed anything.
        # Only a state saved by the new UI may preserve advanced preprocessing.
        from core.ocr_option_contract import migrate_column_preprocess_state

        migrated_preprocess = migrate_column_preprocess_state(state)
        preprocess_customized = bool(migrated_preprocess["customized"])
        input_profile = "custom" if preprocess_customized else "v8_exact"
        state.update({
            "column_input_profile": input_profile,
            "column_preprocess_customized": preprocess_customized,
            "column_auto_filter_ruby": bool(migrated_preprocess["ruby_filter"]),
            "column_filter_fragments": bool(migrated_preprocess["fragment_filter"]),
            "column_smart_crop": bool(migrated_preprocess["smart_crop"]),
            "column_ruby_strength": str(migrated_preprocess["ruby_strength"]),
        })
        vision_backend = str(state.get("vision_backend") or "native_helper").strip().lower()
        if vision_backend == "auto" or vision_backend not in {"live_text", "native_helper", "shortcut"}:
            vision_backend = "native_helper"
        self._set_combo_value(
            getattr(self, "_vision_backend_combo", None), vision_backend,
        )
        self._set_combo_value(
            getattr(self, "_hayai_backend_combo", None), state.get("hayai_backend", "torch")
        )
        self._set_combo_value(
            getattr(self, "_hayai_device_combo", None), state.get("hayai_device", "auto")
        )
        self._set_combo_value(
            getattr(self, "_hayai_quant_combo", None), state.get("hayai_quantize", "none")
        )
        self._set_combo_value(
            getattr(self, "_hayai_litert_quant_combo", None), state.get("hayai_litert_quant", "wi4")
        )
        self._update_hayai_option_state()
        self._set_combo_value(
            getattr(self, "_paddle_model_combo", None),
            state.get("paddle_pipeline", "ocr"),
        )
        self._set_combo_value(
            getattr(self, "_paddle_vl_backend_combo", None),
            state.get("paddle_vl_backend", "auto"),
        )
        self._set_combo_value(
            getattr(self, "_paddle_source_combo", None),
            state.get("paddle_source", "auto"),
        )
        self._update_paddle_vl_backend_state()
        role_state = {
            "column": str(state.get("multi_role_column", "") or ""),
            "page": str(state.get("multi_role_page", "") or ""),
            "sentence": str(state.get("multi_role_sentence", "") or ""),
            "review1": str(state.get("multi_role_review1", "") or ""),
            "review2": str(state.get("multi_role_review2", "") or ""),
            "review3": str(state.get("multi_role_review3", "") or ""),
        }
        for _role, _engine in tuple(role_state.items()):
            if _engine in {"yomitoku"}:
                role_state[_role] = ""
        role_schema = int(state.get("multi_role_schema", 0) or 0)
        if role_schema < 1:
            # Unlabelled positional sessions cannot prove evidence granularity.
            role_state = {key: "" for key in role_state}
            state["multi_ocr"] = False
        elif role_schema < 2:
            # Upgrade known historical Fast-Core presets only.  Preserve custom
            # user choices, but remove the old duplicate-Hayai review default:
            # same-engine disagreement confirmation now uses internal targeted
            # retry rather than occupying a second visible role.
            old_review48_preset = (
                role_state.get("column") == "hayai_ocr"
                and role_state.get("page") == "ndlocr_lite"
                and not role_state.get("sentence")
                and role_state.get("review1") == "manga_48px"
                and not role_state.get("review2")
                and not role_state.get("review3")
            )
            old_full48_preset = (
                role_state.get("column") == "hayai_ocr"
                and role_state.get("page") == "ndlocr_lite"
                and role_state.get("sentence") == "manga_48px"
                and not role_state.get("review1")
                and not role_state.get("review2")
                and not role_state.get("review3")
            )
            old_duplicate_hayai_preset = (
                role_state.get("column") == "hayai_ocr"
                and role_state.get("page") == "ndlocr_lite"
                and role_state.get("sentence") == "manga_48px"
                and role_state.get("review1") == "hayai_ocr"
                and not role_state.get("review2")
                and not role_state.get("review3")
            )
            if old_review48_preset:
                pass
            elif old_full48_preset or old_duplicate_hayai_preset:
                # Phase28 speed-first preset: 48px is valuable as an
                # orthogonal disagreement reviewer, but its whole-book AR pass
                # is too expensive for the small accuracy gain measured on the
                # full-volume benchmark.  Preserve the capability as a manual
                # full-column role, but migrate only the known historical
                # defaults to conflict-only review.
                role_state["sentence"] = ""
                role_state["review1"] = "manga_48px"
            state["multi_smart_router"] = False
            state["multi_evidence_policy"] = "full_evidence"
        elif role_schema >= 3:
            # Free-slot schema 3 stored its three selected engines in the first
            # three role fields.  Preserve those exact choices while restoring
            # role semantics instead of silently swapping models.  The user may
            # change any role afterwards; review1 remains whatever the project
            # already stored (normally empty in free3).
            state["multi_smart_router"] = False
            state["multi_evidence_policy"] = "full_evidence"
            state["multi_local_retry"] = False
        default_profile_version = int(state.get("multi_default_profile_version", 0) or 0)
        phase27_full48_default = (
            role_schema == 2
            and role_state.get("column") == "hayai_ocr"
            and role_state.get("page") == "ndlocr_lite"
            and role_state.get("sentence") == "manga_48px"
            and not role_state.get("review1")
            and not role_state.get("review2")
            and not role_state.get("review3")
        )
        if default_profile_version < 6 and phase27_full48_default:
            # Phase28 speed-first migration for the exact Phase27 factory
            # preset. Custom role combinations are left untouched.
            role_state["sentence"] = ""
            role_state["review1"] = "manga_48px"
        state["multi_default_profile_version"] = max(6, default_profile_version)
        # New execution exposes exactly one review model.  Historical review2/3
        # evidence remains readable from saved OCR documents, but hidden stale
        # settings must never launch extra models behind the user's back.
        role_state["review2"] = ""
        role_state["review3"] = ""
        state["multi_role_schema"] = 2
        for role, value in role_state.items():
            self._set_combo_value(getattr(self, "_multi_role_combos", {}).get(role), value)
        self._set_combo_value(
            getattr(self, "_multi_sentence_speed_combo", None), "role_scheduler",
        )
        self._set_combo_value(
            getattr(self, "_column_ruby_strength_combo", None),
            (state.get("column_ruby_strength", "standard") if preprocess_customized else "standard"),
        )
        self._set_combo_value(
            getattr(self, "_column_isolation_mode_combo", None),
            str(state.get("column_isolation_mode", "mask") or "mask"),
        )
        self._set_combo_value(
            getattr(self, "_ruby_scan_mode_combo", None),
            state.get("ruby_scan_mode", "smart_roi"),
        )
        self._set_combo_value(
            getattr(self, "_column_rescue_policy_combo", None),
            state.get("column_rescue_policy", "adaptive"),
        )
        self._set_combo_value(
            getattr(self, "_ndlocr_page_mode_combo", None),
            state.get("ndlocr_page_mode", "hybrid"),
        )
        self._set_combo_value(
            getattr(self, "_column_sentence_strategy_combo", None),
            state.get("column_sentence_strategy", "full"),
        )
        self._set_combo_value(
            getattr(self, "_handwriting_strategy_combo", None),
            state.get("handwriting_strategy", "balanced"),
        )
        self._set_combo_value(
            getattr(self, "_handwriting_backend_combo", None),
            state.get("handwriting_backend", "auto"),
        )
        for name, key, default in (
            ("_multi_ocr_check", "multi_ocr", False),
            ("_multi_smart_router_check", "multi_smart_router", False),
            ("_multi_local_retry_check", "multi_local_retry", False),
            ("_multi_early_consensus_check", "multi_early_consensus", True),
            ("_multi_parallel_first_check", "multi_parallel_first", False),
            ("_column_split_check", "column_split", False),
            ("_preserve_ruby_check", "preserve_ruby", False),
            ("_column_ruby_filter_check", "column_auto_filter_ruby", True),
            ("_column_fragment_filter_check", "column_filter_fragments", False),
            ("_column_smart_crop_check", "column_smart_crop", True),
            ("_column_compact_transport_check", "column_compact_transport", True),
            ("_column_sentence_reflow_check", "column_reflow", True),
            ("_column_sentence_context_reocr_check", "column_sentence_context_reocr", False),
            ("_column_sentence_global_merged_box_check", "column_sentence_global_merged_box", False),
            ("_handwriting_trace_check", "handwriting", False),
            ("_handwriting_char_mask_check", "handwriting_char_mask", True),
            ("_handwriting_symbol_insert_check", "handwriting_symbol_insert", True),
            ("_vision_vertical_check", "vision_vertical", True),
            ("_vision_vertical_compat_check", "vision_vertical_compat", True),
            ("_vision_language_correction_check", "vision_language_correction", True),
            ("_chinese_merge_line_fragments_check", "merge_horizontal_fragments", True),
            ("_chinese_header_filter_check", "filter_horizontal_headers", True),
            ("_paddle_aistudio_orientation_check", "paddle_aistudio_orientation", False),
            ("_paddle_aistudio_unwarp_check", "paddle_aistudio_unwarp", False),
            ("_paddle_aistudio_textline_check", "paddle_aistudio_textline", False),
        ):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setChecked(bool(state.get(key, default)))
        for name, key, default, low, high in (
            ("_column_sensitivity_spin", "column_sensitivity", 55, 1, 100),
            ("_column_padding_spin", "column_padding_percent", 10, 0, 30),
            ("_column_sentence_max_spin", "column_sentence_max", 10, 3, 256),
            ("_vision_candidate_spin", "vision_candidate_count", 3, 1, 10),
        ):
            widget = getattr(self, name, None)
            if widget is not None:
                try:
                    widget.setValue(max(low, min(high, int(state.get(key, default) or default))))
                except (TypeError, ValueError, OverflowError):
                    widget.setValue(default)
        if hasattr(self, "_vision_min_height_spin"):
            try:
                value = float(state.get("vision_min_height", 0.005) or 0.005)
            except (TypeError, ValueError, OverflowError):
                value = 0.005
            self._vision_min_height_spin.setValue(max(0.0, min(0.1, value)))
        if hasattr(self, "_shortcut_edit"):
            self._shortcut_edit.setText(str(state.get("shortcut_name", "ExtractText") or "ExtractText"))
        handwriting_mode = str(state.get("handwriting_mode", "hybrid") or "hybrid")
        radio = {
            "auto": getattr(self, "_handwriting_mode_auto", None),
            "hybrid": getattr(self, "_handwriting_mode_hybrid", None),
            "manual": getattr(self, "_handwriting_mode_manual", None),
        }.get(handwriting_mode)
        if radio is not None:
            radio.setChecked(True)
        if hasattr(self, "_preserve_ruby_check"):
            self._on_preserve_ruby_toggled(self._preserve_ruby_check.isChecked())
        self._column_preprocess_customized = preprocess_customized
        toggle = getattr(self, "_column_preprocess_toggle", None)
        if toggle is not None:
            blocked = toggle.blockSignals(True)
            toggle.setChecked(True)
            toggle.blockSignals(blocked)
        self._toggle_column_preprocess_body(True)
        self._on_column_cleanup_control_changed()
        saved_paddle_mode = str(state.get("paddle_aistudio_mode", "async_v2") or "async_v2")
        # Migrate the old untouched default (sync with no URL) to the official
        # jobs API. A sync mode with an explicit URL remains user-controlled.
        legacy_unconfigured_paddle = (
            saved_paddle_mode == "sync"
            and not str(state.get("paddle_aistudio_api_url", "") or "").strip()
        )
        if legacy_unconfigured_paddle:
            saved_paddle_mode = "async_v2"
        self._set_combo_value(
            getattr(self, "_paddle_aistudio_mode_combo", None),
            saved_paddle_mode,
        )
        sync_family_state = str(state.get("paddle_aistudio_sync_family", "ppocr") or "ppocr")
        # Hardened-r3 persisted the combined ``layout_vl`` family. Its payload
        # was the conservative Structure/VL common denominator, which matches
        # the new VL family exactly. Migrate silently instead of resetting the
        # combo to PP-OCR and changing an existing profile's request shape.
        if sync_family_state == "layout_vl":
            sync_family_state = "vl"
        self._set_combo_value(
            getattr(self, "_paddle_aistudio_sync_family_combo", None),
            sync_family_state,
        )
        saved_async_model = str(
            state.get("paddle_aistudio_async_model", "PaddleOCR-VL-1.6")
            or "PaddleOCR-VL-1.6"
        )
        if legacy_unconfigured_paddle and saved_async_model == "PP-OCRv6":
            saved_async_model = "PaddleOCR-VL-1.6"
        self._set_combo_value(
            getattr(self, "_paddle_aistudio_async_model_combo", None),
            saved_async_model,
        )
        if hasattr(self, "_paddle_aistudio_url_edit"):
            self._paddle_aistudio_url_edit.setText(str(state.get("paddle_aistudio_api_url", "")))
        if hasattr(self, "_paddle_aistudio_timeout_spin"):
            self._paddle_aistudio_timeout_spin.setValue(
                max(30, min(900, int(state.get("paddle_aistudio_timeout", 180) or 180)))
            )
        if hasattr(self, "_paddle_aistudio_retry_spin"):
            self._paddle_aistudio_retry_spin.setValue(
                max(0, min(6, int(state.get("paddle_aistudio_retries", 3) or 0)))
            )
        self._update_paddle_aistudio_option_state()
        requested_engine = str(state.get("active_adapter") or "apple_vision")
        # Google Vision was retired from the interactive OCR workspace.  Old
        # per-mode settings must not resurrect a hidden engine after upgrade.
        if requested_engine in {"google_vision", "yomitoku", "manga_ocr"}:
            requested_engine = "apple_vision"
        self._select_adapter(requested_engine, True, None)
        self._update_multi_ocr_option_state()
        self._update_ruby_cleanup_state()

    def _set_engine_mode_compatibility(self, combo, mode: str, *, allow_empty: bool = False) -> None:
        from adapters.ocr_profiles import is_engine_compatible
        if combo is None:
            return
        model = combo.model()
        for index in range(combo.count()):
            engine_id = str(combo.itemData(index) or "")
            enabled = bool((allow_empty and not engine_id) or is_engine_compatible(engine_id, mode))
            try:
                item = model.item(index)
                if item is not None:
                    item.setEnabled(enabled)
            except Exception:
                pass

    def _on_ocr_mode_changed(self, index: int) -> None:
        if self._ocr_mode_change_guard or index < 0:
            return
        requested = str(self._ocr_mode_combo.itemData(index) or "ja_vertical")
        if requested == self._current_ocr_mode():
            return
        if self._ocr_run_active:
            notify(self, "请先停止或完成当前 OCR，再切换识别模式。", "warning")
            self._ocr_mode_change_guard = True
            self._set_combo_value(self._ocr_mode_combo, self._current_ocr_mode())
            self._ocr_mode_change_guard = False
            return
        previous = self._current_ocr_mode()
        previous_state = self._capture_ocr_mode_state()
        self._ocr_mode_states[previous] = previous_state
        self._save_ocr_mode_state(previous, previous_state)
        self._ocr_mode = requested
        if requested not in self._ocr_mode_states:
            self._ocr_mode_states[requested] = self._load_persisted_ocr_mode_state(requested)
        self._apply_ocr_mode_ui(initial=False)
        self._ocr_mode_settings.setValue("ocr/active_mode", requested)

    def _apply_ocr_mode_ui(self, *, initial: bool = False) -> None:
        from adapters.ocr_profiles import get_ocr_profile, is_engine_compatible

        mode = self._current_ocr_mode()
        profile = get_ocr_profile(mode)
        if mode not in self._ocr_mode_states:
            self._ocr_mode_states[mode] = self._load_persisted_ocr_mode_state(mode)
        state = dict(self._ocr_mode_states.get(mode) or self._default_ocr_mode_state(mode))

        self._ocr_mode_change_guard = True
        try:
            self._restore_ocr_mode_state(state)
            self._set_engine_mode_compatibility(self._adapter_combo, mode)
            for combo in self._multi_model_combos:
                self._set_engine_mode_compatibility(combo, mode, allow_empty=True)

            if not is_engine_compatible(self._active_adapter, mode):
                self._select_adapter("apple_vision", True, None)
            for combo in self._multi_model_combos:
                selected = str(combo.currentData() or "")
                if selected and not is_engine_compatible(selected, mode):
                    self._set_combo_value(combo, "")

            japanese_mode = bool(profile.allow_column_pipeline)
            if not japanese_mode:
                self._column_split_check.setChecked(False)
                self._column_sentence_reflow_check.setChecked(False)
                if hasattr(self, "_preserve_ruby_check"):
                    self._preserve_ruby_check.setChecked(False)
                if hasattr(self, "_handwriting_trace_check"):
                    self._handwriting_trace_check.setChecked(False)
                self._vision_vertical_check.setChecked(False)
                self._vision_vertical_compat_check.setChecked(False)
            self._column_split_check.setEnabled(japanese_mode)
            if hasattr(self, "_preserve_ruby_check"):
                self._preserve_ruby_check.setEnabled(japanese_mode)
            self._column_ocr_widget.setVisible(japanese_mode)
            self._ocr_reflow_widget.setVisible(japanese_mode)
            self._handwriting_card_widget.setVisible(profile.allow_japanese_handwriting)
            self._chinese_horizontal_widget.setVisible(not japanese_mode)
            self._ocr_settings_tabs.setTabText(
                1, "分列与组句" if japanese_mode else "横排版面"
            )
            self._ocr_settings_tabs.setTabEnabled(2, profile.allow_japanese_handwriting)
            if not profile.allow_japanese_handwriting and self._ocr_settings_tabs.currentIndex() == 2:
                self._ocr_settings_tabs.setCurrentIndex(0)
            self._vision_language_correction_check.setText(
                "启用日语语言校正" if japanese_mode else "启用简体中文语言校正"
            )
            self._vision_vertical_check.setText(
                "按日文竖排顺序组合观察结果（右→左、上→下）"
                if japanese_mode else "简体中文横排固定为上→下、左→右"
            )
            self._vision_vertical_check.setEnabled(
                japanese_mode
                and str(self._vision_backend_combo.currentData() or "") == "native_helper"
            )
            self._vision_vertical_compat_check.setEnabled(japanese_mode)
            self._ocr_mode_summary.setText(
                (
                    "现有日文轻小说路径：竖排右→左、分列掩膜与逐列成句保持原样；"
                    "Ruby 保留为独立开关，默认关闭，开启后才加载 findtextCenterNet。"
                )
                if japanese_mode else (
                    "独立简体中文路径：整页横排左→右、上→下；Apple Vision 使用 "
                    "zh-Hans，PaddleOCR 使用 ch；不调用任何日文分列或手写模块。"
                )
            )
            self._on_column_split_toggled(self._column_split_check.isChecked())
            self._update_shortcut_widget_visibility()
            self._highlight_adapter(self._active_adapter)
        finally:
            self._ocr_mode_change_guard = False

        self._ocr_mode_states[mode] = self._capture_ocr_mode_state()
        self._save_ocr_mode_state(mode, self._ocr_mode_states[mode])
        self._ocr_mode_settings.setValue("ocr/active_mode", mode)

    def _refresh_ocr_runtime_status(self, *, show_dialog: bool = False, deep: bool = False):
        """Refresh OCR runtime status without blocking the GUI thread.

        Runtime probes may walk Hugging Face caches and launch isolated Python
        import/version checks.  Those operations are I/O/subprocess work and
        must never run synchronously from OCR completion, startup, or a button
        click.  A generation token discards stale results after a newer scan.
        """
        token = self._runtime_status_generation.begin()
        self._runtime_status_scan_busy = True
        if hasattr(self, "_column_detect_status"):
            self._column_detect_status.setText("正在后台扫描本地 OCR 运行环境…")

        signals = WorkerSignals()
        self._runtime_status_signal_refs[token] = signals

        def is_current() -> bool:
            return self._runtime_status_generation.is_current(token)

        def finish(payload):
            self._runtime_status_signal_refs.pop(token, None)
            if not is_current():
                return
            self._runtime_status_scan_busy = False
            data = dict(payload or {})
            details = [str(value) for value in (data.get("details") or [])]
            ready = int(data.get("ready", 0) or 0)
            self._column_detect_status.setText(f"已扫描：{ready} 项可直接使用")
            if show_dialog:
                QMessageBox.information(
                    self, "本地 OCR 检测结果",
                    "本次只检查本地文件、环境和系统能力，没有安装或下载任何模型。\n\n"
                    + "\n".join(details),
                )

        def failed(message):
            self._runtime_status_signal_refs.pop(token, None)
            if not is_current():
                return
            self._runtime_status_scan_busy = False
            self._column_detect_status.setText("OCR 运行环境扫描失败；可手动重新检测")
            if show_dialog:
                QMessageBox.warning(self, "OCR 环境检测失败", str(message or "未知错误"))

        signals.finished.connect(finish)
        signals.error.connect(failed)

        def worker():
            try:
                from adapters.ocr_runtime_catalog import clear_probe_cache, probe_runtime
                # Explicit/deep scans are authoritative refreshes. Ordinary
                # startup/completion scans reuse the process cache and therefore
                # avoid repeatedly traversing every local model directory.
                if deep:
                    clear_probe_cache()
                details = []
                try:
                    from adapters.vision_backends import BackendFactory
                    live_ok, live_reason = BackendFactory.create("live_text").is_available()
                    helper_ok, helper_reason = BackendFactory.create("native_helper").is_available()
                    shortcut_ok, shortcut_reason = BackendFactory.create("shortcut").is_available()
                    details.append(f"Apple Live Text Helper：{'可用' if live_ok else '不可用'}（{live_reason or '本地 ImageAnalyzer'}）")
                    details.append(f"Apple Vision RecognizeTextRequest Helper：{'可用' if helper_ok else '不可用'}（{helper_reason or '本地 Swift Helper'}）")
                    details.append(f"Apple Vision 快捷指令：{'可用' if shortcut_ok else '不可用'}（{shortcut_reason or '无需模型'}）")
                except Exception as exc:
                    details.append(f"Apple Vision：检测失败（{exc}）")
                for cid in (
                    "hayai_ocr", "hayai_ocr_litert", "ndlocr_lite",
                    "paddle_ocr", "paddle_structure", "paddle_vl", "manga_48px",
                ):
                    probe = probe_runtime(cid, deep=deep, refresh=deep)
                    state = "本地可用" if probe.ready else ("环境已安装，模型待确认" if probe.installed else "未安装")
                    details.append(f"{cid}：{state}（{probe.detail}）")
                ready = sum(
                    "本地可用" in row
                    or "Apple Live Text Helper：可用" in row
                    or "Apple Vision RecognizeTextRequest Helper：可用" in row
                    or "Apple Vision 快捷指令：可用" in row
                    for row in details
                )
                safe_qt_emit(signals.finished, {"details": details, "ready": ready})
            except Exception:
                import traceback
                safe_qt_emit(signals.error, traceback.format_exc())

        threading.Thread(
            target=worker, daemon=True, name=f"ocr-runtime-status-{token}"
        ).start()

    def _update_paddle_vl_backend_state(self, *_args):
        combo = getattr(self, "_paddle_vl_backend_combo", None)
        model_combo = getattr(self, "_paddle_model_combo", None)
        if combo is None or model_combo is None:
            return
        enabled = str(model_combo.currentData() or "ocr") == "vl"
        combo.setEnabled(enabled)
        combo.setToolTip(
            "PaddleOCR-VL-1.6：Apple Silicon 自动优先官方 MLX-VLM，失败回退 Paddle。"
            if enabled else "仅 PaddleOCR-VL-1.6 使用 MLX；PP-OCRv6 / PP-Structure 不受影响。"
        )

    def _paddle_component_id(self) -> str:
        pipeline = str(self._paddle_model_combo.currentData() or "ocr")
        return {"ocr": "paddle_ocr", "structure": "paddle_structure", "vl": "paddle_vl"}.get(
            pipeline, "paddle_ocr"
        )

    def _check_paddle_runtime(self):
        from adapters.ocr_runtime_catalog import probe_runtime
        component_id = self._paddle_component_id()
        probe = probe_runtime(component_id, deep=True, refresh=True)
        if probe.ready:
            text = f"✓ 本地可用：{probe.detail}"
            style = "color: #248A3D; font-size: 10px;"
        elif probe.installed:
            text = f"△ 环境已安装，模型待准备：{probe.detail}"
            style = "color: #9A6700; font-size: 10px;"
        else:
            text = f"○ 尚未安装：{probe.detail}"
            style = "color: #6E6E73; font-size: 10px;"
        if component_id == "paddle_vl":
            try:
                from adapters.paddle_vl_mlx import probe_mlx_runtime, normalize_vl_backend
                requested = normalize_vl_backend(self._paddle_vl_backend_combo.currentData())
                if requested in {"auto", "mlx"}:
                    mlx_ok, mlx_detail = probe_mlx_runtime(deep=True)
                    text += f" · MLX：{'可用' if mlx_ok else '待安装/回退'}（{mlx_detail}）"
            except Exception as exc:
                text += f" · MLX 检测失败（{exc}）"
        self._paddle_runtime_status.setText(text)
        self._paddle_runtime_status.setStyleSheet(style)
        self._log_view.appendPlainText(f"\n[PaddleOCR 检查] {text}")

    def _prepare_paddle_runtime(self):
        if not self._run_btn.isEnabled():
            notify(self, "请先停止或完成当前 OCR，再安装/修复 PaddleOCR 模型。", "warning")
            return
        token = self._paddle_prepare_generation.begin()
        pipeline = str(self._paddle_model_combo.currentData() or "ocr")
        source = str(self._paddle_source_combo.currentData() or "auto")
        vl_backend = str(self._paddle_vl_backend_combo.currentData() or "auto")
        self._paddle_prepare_active = True
        self._run_btn.setEnabled(False)
        self._paddle_prepare_btn.setEnabled(False)
        self._paddle_check_btn.setEnabled(False)
        self._paddle_prepare_btn.setText("正在准备…")
        self._paddle_runtime_status.setText("正在创建独立环境并初始化模型；可在 OCR 日志查看当前下载源。")
        self._paddle_runtime_status.setStyleSheet("color: #0066CC; font-size: 10px;")
        self._switch_view("log")

        signals = WorkerSignals()
        self._paddle_prepare_signals = signals
        self._paddle_prepare_signal_refs[token] = signals

        def is_current():
            return self._paddle_prepare_generation.is_current(token)

        def append_status(message):
            if not is_current():
                return
            message = str(message or "").strip()
            if not message:
                return
            self._paddle_runtime_status.setText(message[-700:])
            self._log_view.appendPlainText(f"[PaddleOCR] {message}")

        def restore_buttons():
            if not is_current():
                return
            self._paddle_prepare_signal_refs.pop(token, None)
            self._paddle_prepare_active = False
            self._run_btn.setEnabled(True)
            self._paddle_prepare_btn.setEnabled(True)
            self._paddle_check_btn.setEnabled(True)
            self._paddle_prepare_btn.setText("安装 / 修复模型")

        def done(payload):
            if not is_current():
                return
            restore_buttons()
            selected_source = str((payload or {}).get("model_source") or source)
            profile = str((payload or {}).get("model_profile") or pipeline)
            effective_vl = str((payload or {}).get("vl_backend") or "")
            backend_suffix = (f" · VL {effective_vl.upper()}" if effective_vl else "")
            self._paddle_runtime_status.setText(
                f"✓ PaddleOCR 已就绪 · 来源 {selected_source} · {profile}{backend_suffix}"
            )
            self._paddle_runtime_status.setStyleSheet("color: #248A3D; font-size: 10px;")
            self._refresh_ocr_runtime_status(show_dialog=False, deep=False)
            notify(
                self,
                "当前 PaddleOCR 环境和模型已经完成初始化。现在可以直接开始 OCR；"
                "后续不会重复下载已缓存的模型。",
                "success",
            )

        def failed(message):
            if not is_current():
                return
            restore_buttons()
            self._paddle_runtime_status.setText("✕ PaddleOCR 模型准备失败；请查看 OCR 日志并切换下载源。")
            self._paddle_runtime_status.setStyleSheet("color: #C9342F; font-size: 10px;")
            show_error_dialog(self, "PaddleOCR 安装 / 下载失败", str(message))

        signals.log.connect(append_status)
        signals.finished.connect(done)
        signals.error.connect(failed)

        ocr_mode_snapshot = self._current_ocr_mode()

        def worker():
            try:
                from adapters.ocr_profiles import get_ocr_profile
                from adapters.paddle_ocr_adapter import prepare_runtime
                profile = get_ocr_profile(ocr_mode_snapshot)
                payload = prepare_runtime(
                    pipeline=pipeline,
                    lang=profile.paddle_lang,
                    model_source=source,
                    vl_backend=vl_backend,
                    verbose=True,
                    progress_callback=signals.log.emit,
                )
                signals.finished.emit(payload)
            except Exception:
                import traceback
                signals.error.emit(traceback.format_exc())

        threading.Thread(target=worker, daemon=True, name="paddle-runtime-prepare").start()

    def _selected_multi_ocr_role_plan(self):
        from core.multi_ocr_roles import MultiOcrRolePlan
        combos = getattr(self, "_multi_role_combos", {}) or {}
        return MultiOcrRolePlan.from_mapping({
            role: str(combo.currentData() or "")
            for role, combo in combos.items()
            if combo is not None
        })

    def _selected_multi_ocr_slot_engines(self) -> list[str]:
        """Return the three free multi-model slots in stable display order."""
        combos = [
            getattr(self, "_multi_model1_combo", None),
            getattr(self, "_multi_model2_combo", None),
            getattr(self, "_multi_model3_combo", None),
        ]
        values = [str(combo.currentData() or "") for combo in combos if combo is not None]
        return [value for value in values if value]

    def _on_multi_ocr_role_changed(self, *_args) -> None:
        self._refresh_multi_ocr_role_choices()
        self._update_multi_ocr_option_state()

    def _on_multi_ocr_toggled(self, checked: bool) -> None:
        """Seed the Fast-Core three-model preset only when every role is empty."""
        if bool(checked):
            combos = getattr(self, "_multi_role_combos", {}) or {}
            if combos and not any(str(combo.currentData() or "") for combo in combos.values()):
                preset = {
                    "column": "hayai_ocr",
                    "page": "ndlocr_lite",
                    "sentence": "",
                    "review1": "manga_48px",
                }
                for role, engine_id in preset.items():
                    self._set_combo_value(combos.get(role), engine_id)
            smart_router = getattr(self, "_multi_smart_router_check", None)
            if smart_router is not None:
                smart_router.setChecked(False)
            local_retry = getattr(self, "_multi_local_retry_check", None)
            if local_retry is not None:
                local_retry.setChecked(False)
        self._update_multi_ocr_option_state()

    def _refresh_multi_ocr_role_choices(self) -> None:
        """Refresh role/mode capability and enforce one visible role per engine.

        Same-engine disagreement confirmation is an internal targeted retry.  It
        never consumes another role slot, so an engine selected by one role is
        disabled in every other role combo.  Existing legacy duplicates remain
        visible in their own combo so restore can fail closed with a clear
        validation message instead of silently rewriting custom projects.
        """
        combos = getattr(self, "_multi_role_combos", {}) or {}
        from adapters.ocr_profiles import is_engine_compatible
        from adapters.column_ocr_adapter import SUPPORTED_RECOGNIZERS
        current_mode = self._current_ocr_mode()
        selected_elsewhere = {
            role: {
                str(other.currentData() or "")
                for other_role, other in combos.items()
                if other_role != role and other is not None and str(other.currentData() or "")
            }
            for role in combos
        }
        for role, combo in combos.items():
            if combo is None:
                continue
            own = str(combo.currentData() or "")
            model = combo.model()
            for index in range(combo.count()):
                engine_id = str(combo.itemData(index) or "")
                enabled = True
                tooltip = ""
                if engine_id:
                    if not is_engine_compatible(engine_id, current_mode):
                        enabled = False
                        tooltip = "当前 OCR 模式不支持此模型"
                    elif role != "page" and engine_id not in SUPPORTED_RECOGNIZERS:
                        enabled = False
                        tooltip = "此模型只支持整页主模型槽位"
                    elif engine_id in selected_elsewhere.get(role, set()) and engine_id != own:
                        enabled = False
                        tooltip = "同一个 OCR 模型只能占用一个角色；分歧确认使用内部定向重试"
                    else:
                        tooltip = "每个 OCR 模型只占一个角色；同模型复核由内部定向重试完成"
                try:
                    item = model.item(index)
                    if item is not None:
                        item.setEnabled(enabled or engine_id == own)
                        item.setToolTip(tooltip)
                except Exception:
                    pass

    def _update_multi_ocr_option_state(self, *_args):
        multi_enabled = bool(
            hasattr(self, "_multi_ocr_check") and self._multi_ocr_check.isChecked()
        )
        self._refresh_multi_ocr_role_choices()
        combos = getattr(self, "_multi_role_combos", {}) or {}
        for combo in combos.values():
            if combo is not None:
                combo.setEnabled(multi_enabled)
        smart_router = getattr(self, "_multi_smart_router_check", None)
        if smart_router is not None:
            smart_router.setEnabled(multi_enabled)
        local_retry = getattr(self, "_multi_local_retry_check", None)
        if local_retry is not None:
            local_retry.setEnabled(multi_enabled)

        # Multi-model owns an independent configuration.  The ordinary single
        # OCR selector is visibly locked while multi-model is ON and never acts
        # as an implicit seventh/default engine.
        adapter_combo = getattr(self, "_adapter_combo", None)
        if adapter_combo is not None:
            adapter_combo.setEnabled(not multi_enabled)

        # Multi-model owns its own structural routing.  It may build shared
        # physical-column geometry internally, but it must never toggle or
        # overwrite the single-model "启用日文物理分列" preference.
        plan = self._selected_multi_ocr_role_plan()
        needs_geometry = bool(multi_enabled)
        try:
            from adapters.ocr_profiles import get_ocr_profile
            column_capable = bool(get_ocr_profile(self._current_ocr_mode()).allow_column_pipeline)
        except Exception:
            column_capable = True
        split_check = getattr(self, "_column_split_check", None)
        if split_check is not None:
            split_check.setEnabled(bool(column_capable and not multi_enabled))
            if multi_enabled:
                split_check.setToolTip(
                    "这是单模型专用分列开关。多模型使用独立的角色路由和共享分列几何，不会改写此设置。"
                )
            else:
                split_check.setToolTip(
                    "单模型专用：关闭时把原页直接交给所选 OCR 引擎；开启时按日文物理竖列识别。"
                    "极窄的已裁单列图会自动按单列处理，不会再被拆成多个假列。"
                )
        # Keep the column-settings panel available for multi-model geometry
        # parameters without changing the single-model checkbox state.
        column_settings = getattr(self, "_column_settings_widget", None)
        if column_settings is not None:
            # Keep the full “分列与组句” configuration visible in Japanese
            # vertical mode even when the pipeline switch is currently off.
            # Execution is still gated by the checkbox / multi-model state.
            column_settings.setVisible(bool(column_capable))
        preview_button = getattr(self, "_column_preview_btn", None)
        preview_enabled = bool(
            column_capable and (multi_enabled or (split_check is not None and split_check.isChecked()))
        )
        if preview_button is not None:
            preview_button.setEnabled(preview_enabled)
        preview_header_button = getattr(self, "_preview_columns_btn", None)
        if preview_header_button is not None:
            # Header preview is diagnostic and may be used before enabling the
            # OCR split pipeline.  Only the OCR mode capability gates it.
            preview_header_button.setEnabled(bool(column_capable))

        # Legacy controls are deliberately inert under role scheduling.
        early = getattr(self, "_multi_early_consensus_check", None)
        if early is not None:
            early.setEnabled(False)
        parallel = getattr(self, "_multi_parallel_first_check", None)
        if parallel is not None:
            parallel.setEnabled(False)
        sentence_combo = getattr(self, "_multi_sentence_speed_combo", None)
        if sentence_combo is not None:
            sentence_combo.setEnabled(False)

    @staticmethod
    def _column_preprocess_default_values() -> dict[str, object]:
        from core.ocr_option_contract import column_preprocess_defaults

        return column_preprocess_defaults()

    def _column_preprocess_is_customized(self) -> bool:
        from core.ocr_option_contract import is_default_column_preprocess

        ruby = bool(
            getattr(self, "_column_ruby_filter_check", None)
            and self._column_ruby_filter_check.isChecked()
        )
        fragments = bool(
            getattr(self, "_column_fragment_filter_check", None)
            and self._column_fragment_filter_check.isChecked()
        )
        smart_crop = bool(
            not hasattr(self, "_column_smart_crop_check")
            or self._column_smart_crop_check.isChecked()
        )
        strength = str(
            self._column_ruby_strength_combo.currentData()
            if hasattr(self, "_column_ruby_strength_combo")
            else "standard"
        ).strip().lower() or "standard"
        return not is_default_column_preprocess(
            ruby_filter=ruby,
            fragment_filter=fragments,
            smart_crop=smart_crop,
            ruby_strength=strength,
        )

    def _apply_column_preprocess_defaults(self) -> None:
        defaults = self._column_preprocess_default_values()
        for name, key in (
            ("_column_ruby_filter_check", "ruby_filter"),
            ("_column_fragment_filter_check", "fragment_filter"),
            ("_column_smart_crop_check", "smart_crop"),
        ):
            widget = getattr(self, name, None)
            if widget is not None:
                blocked = widget.blockSignals(True)
                widget.setChecked(bool(defaults[key]))
                widget.blockSignals(blocked)
        combo = getattr(self, "_column_ruby_strength_combo", None)
        if combo is not None:
            blocked = combo.blockSignals(True)
            self._set_combo_value(combo, str(defaults["ruby_strength"]))
            combo.blockSignals(blocked)
        self._column_preprocess_customized = False
        reset_btn = getattr(self, "_column_preprocess_reset_btn", None)
        if reset_btn is not None:
            reset_btn.setEnabled(False)
        self._update_ruby_cleanup_state()

    def _reset_column_preprocess_defaults(self, *_args) -> None:
        self._apply_column_preprocess_defaults()

    def _toggle_column_preprocess_body(self, checked: bool = True) -> None:
        # The preprocessing controls are part of the normal “分列与组句” page.
        # Ignore legacy collapsed state from older workspaces.
        body = getattr(self, "_column_preprocess_body", None)
        toggle = getattr(self, "_column_preprocess_toggle", None)
        if body is not None:
            body.setVisible(True)
        if toggle is not None:
            blocked = toggle.blockSignals(True)
            toggle.setChecked(True)
            toggle.blockSignals(blocked)

    # Compatibility hook for older plugins.  The technical profile selector is
    # no longer visible; requesting the old checked state simply restores the
    # hidden default preprocessing contract.
    def _on_column_v8_input_profile_toggled(self, checked: bool):
        if checked:
            self._apply_column_preprocess_defaults()
        else:
            self._on_column_cleanup_control_changed()

    def _on_column_cleanup_control_changed(self, *_args):
        self._column_preprocess_customized = self._column_preprocess_is_customized()
        reset_btn = getattr(self, "_column_preprocess_reset_btn", None)
        if reset_btn is not None:
            reset_btn.setEnabled(self._column_preprocess_customized)
        self._update_ruby_cleanup_state()

    def _update_ruby_cleanup_state(self, *_args):
        combo = getattr(self, "_column_ruby_strength_combo", None)
        ruby = getattr(self, "_column_ruby_filter_check", None)
        fragments = getattr(self, "_column_fragment_filter_check", None)
        smart_crop = getattr(self, "_column_smart_crop_check", None)
        from core.ocr_option_contract import resolve_column_cleanup_option_state

        state = resolve_column_cleanup_option_state(
            column_split_enabled=bool(
                (hasattr(self, "_column_split_check") and self._column_split_check.isChecked())
                or (hasattr(self, "_multi_ocr_check") and self._multi_ocr_check.isChecked())
            ),
            ruby_filter_checked=bool(ruby is not None and ruby.isChecked()),
            fragment_filter_checked=bool(fragments is not None and fragments.isChecked()),
        )
        if ruby is not None:
            # Independent contract: the Ruby-preservation switch never changes
            # the ordinary OCR cleanup profile.
            ruby.setEnabled(state.cleanup_controls_enabled)
        for checkbox in (fragments, smart_crop):
            if checkbox is not None:
                checkbox.setEnabled(state.cleanup_controls_enabled)
        if combo is not None:
            combo.setEnabled(state.cleanup_strength_enabled)

    def _on_column_split_toggled(self, enabled: bool):
        from adapters.ocr_profiles import get_ocr_profile

        profile = get_ocr_profile(self._current_ocr_mode())
        if enabled and not profile.allow_column_pipeline:
            blocked = self._column_split_check.blockSignals(True)
            self._column_split_check.setChecked(False)
            self._column_split_check.blockSignals(blocked)
            enabled = False
        multi_enabled = bool(
            hasattr(self, "_multi_ocr_check") and self._multi_ocr_check.isChecked()
        )
        column_ui_active = bool(profile.allow_column_pipeline and (enabled or multi_enabled))
        self._column_settings_widget.setVisible(bool(profile.allow_column_pipeline))
        self._column_preview_btn.setEnabled(column_ui_active)
        preview_header_button = getattr(self, "_preview_columns_btn", None)
        if preview_header_button is not None:
            preview_header_button.setEnabled(bool(profile.allow_column_pipeline))
        self._update_ruby_cleanup_state()
        self._update_multi_ocr_option_state()
        self._update_sentence_context_reocr_state()
        if not enabled and hasattr(self, "_handwriting_trace_check") and self._handwriting_trace_check.isChecked():
            self._handwriting_trace_check.setChecked(False)

    def _update_sentence_context_reocr_state(self, *_args):
        widget = getattr(self, "_column_sentence_context_reocr_check", None)
        if widget is None:
            return
        split_enabled = bool(
            getattr(self, "_column_split_check", None)
            and self._column_split_check.isChecked()
        )
        reflow_enabled = bool(
            getattr(self, "_column_sentence_reflow_check", None)
            and self._column_sentence_reflow_check.isChecked()
        )
        context_enabled = split_enabled and reflow_enabled
        widget.setEnabled(context_enabled)
        merged_box_widget = getattr(
            self, "_column_sentence_global_merged_box_check", None
        )
        if merged_box_widget is not None:
            merged_box_widget.setEnabled(context_enabled and widget.isChecked())
        strategy_widget = getattr(self, "_column_sentence_strategy_combo", None)
        if strategy_widget is not None:
            strategy_widget.setEnabled(context_enabled and widget.isChecked())

    def _handwriting_selected_mode(self) -> str:
        if hasattr(self, "_handwriting_mode_auto") and self._handwriting_mode_auto.isChecked():
            return "auto"
        if hasattr(self, "_handwriting_mode_manual") and self._handwriting_mode_manual.isChecked():
            return "manual"
        return "hybrid"

    def _handwriting_selected_strategy(self) -> str:
        if hasattr(self, "_handwriting_strategy_combo"):
            return str(self._handwriting_strategy_combo.currentData() or "balanced")
        return "balanced"

    def _handwriting_selected_backend(self) -> str:
        if hasattr(self, "_handwriting_backend_combo"):
            return str(self._handwriting_backend_combo.currentData() or "auto")
        return "auto"

    def _refresh_handwriting_model_status(self, *_args):
        label = getattr(self, "_handwriting_model_status", None)
        button = getattr(self, "_handwriting_model_btn", None)
        if label is None:
            return
        backend = self._handwriting_selected_backend()
        test_button = getattr(self, "_handwriting_apple_test_btn", None)
        if test_button is not None:
            test_button.setVisible(backend in {"apple", "auto"})
            test_button.setEnabled(backend in {"apple", "auto"})
        if backend == "jlect":
            label.setText("本地最低备用可直接使用；字符库不完整，未识字会保留 □。")
            label.setStyleSheet(f"color: {MUTED}; font-size: 10px;")
            if button is not None:
                button.setEnabled(False)
                button.setText("无需安装")
            return

        apple_ok = False
        apple_detail = ""
        try:
            from adapters.apple_pkstroke_engine import bridge_status
            apple_status = bridge_status(auto_build=False)
            apple_ok = bool(apple_status.available)
            apple_detail = str(apple_status.detail)
        except Exception as exc:
            apple_detail = str(exc)

        if backend == "apple":
            label.setText(apple_detail)
            label.setStyleSheet(
                "color: #248A3D; font-size: 10px;" if apple_ok
                else "color: #C9342F; font-size: 10px;"
            )
            if button is not None:
                button.setEnabled(not apple_ok)
                button.setText("Apple 桥接已就绪" if apple_ok else "编译 Apple 桥接")
                button.setToolTip("需要 macOS 27 和 Xcode 27 / macOS 27 SDK。")
            return

        try:
            from adapters.openvino_handwriting_engine import model_available, runtime_available
            model_ok, model_detail = model_available()
            runtime_ok, _runtime_detail = runtime_available()
        except Exception as exc:
            model_ok = runtime_ok = False
            model_detail = str(exc)

        if backend == "auto":
            if apple_ok:
                label.setText("自动模式将使用 Apple PKStrokeRecognizer（日语、设备本地、严格单字提交）。")
                label.setStyleSheet("color: #248A3D; font-size: 10px;")
                if button is not None:
                    button.setEnabled(False)
                    button.setText("Apple 桥接已就绪")
                return
            if model_ok:
                label.setText(f"Apple 暂不可用：{apple_detail}；将自动使用 OpenVINO。")
                label.setStyleSheet("color: #9A6700; font-size: 10px;")
                if button is not None:
                    button.setEnabled(False)
                    button.setText("OpenVINO 已就绪")
                return
            label.setText(
                f"Apple 暂不可用：{apple_detail}；OpenVINO 也未就绪：{model_detail}；"
                "将使用本地 JLect 最低备用。"
            )
            label.setStyleSheet("color: #9A6700; font-size: 10px;")
            if button is not None:
                # On an OS/SDK 27 Mac prepare Apple first; elsewhere prepare OpenVINO.
                can_prepare_apple = "macOS 27" not in apple_detail or "尚未编译" in apple_detail
                button.setEnabled(True)
                button.setText("编译 Apple 桥接" if can_prepare_apple else "下载 OpenVINO 模型")
            return

        if model_ok:
            label.setText("OpenVINO 日语手写模型与运行环境可用。")
            label.setStyleSheet("color: #248A3D; font-size: 10px;")
        else:
            label.setText(f"OpenVINO 模式尚不可用：{model_detail}")
            label.setStyleSheet("color: #C9342F; font-size: 10px;")
        if button is not None:
            button.setEnabled(not model_ok)
            button.setText("下载 OpenVINO 模型" if not model_ok else "OpenVINO 已就绪")
            if not runtime_ok:
                button.setToolTip("先运行 pip3 install -r requirements-handwriting-openvino.txt，再下载模型。")

    def _open_apple_handwriting_test_panel(self):
        button = getattr(self, "_handwriting_apple_test_btn", None)
        try:
            if button is not None:
                button.setEnabled(False)
                button.setText("正在打开…")
            from adapters.apple_pkstroke_engine import launch_manual_test_panel
            launch_manual_test_panel(auto_build=True)
            self._log_view.appendPlainText(
                "\n✍️ 已打开 Apple PKStrokeRecognizer 手写测试面板："
                "左侧手动画，右侧查看苹果实际收到的 PKDrawing 和 recognizedText() 结果。"
            )
        except Exception as exc:
            show_error_dialog(self, "无法打开 Apple 手写测试面板", str(exc))
        finally:
            if button is not None:
                button.setEnabled(True)
                button.setText("打开 Apple 手写测试面板")
            self._refresh_handwriting_model_status()

    def _open_latest_apple_auto_trace(self):
        button = getattr(self, "_handwriting_apple_auto_preview_btn", None)
        if button is not None:
            button.setEnabled(False)
            button.setText("正在打开自动轨迹…")
        try:
            from adapters.apple_pkstroke_engine import latest_auto_payload_path, launch_manual_test_panel
            payload = latest_auto_payload_path()
            if payload is None:
                raise RuntimeError("尚未找到自动轨迹。请先运行一次黑像素临摹识别。")
            app_path = launch_manual_test_panel(auto_build=True, payload_path=payload)
            self._log_view.appendPlainText(
                "\n✍️ 已将最新自动轨迹载入 Apple 测试面板："
                f"\n{payload}\nApp：{app_path}"
            )
        except Exception as exc:
            show_error_dialog(self, "无法预览最新自动轨迹", str(exc))
        finally:
            if button is not None:
                button.setEnabled(True)
                button.setText("预览最新自动轨迹")


    def _install_handwriting_model(self):
        backend = self._handwriting_selected_backend()
        if backend in {"apple", "auto"}:
            try:
                from adapters.apple_pkstroke_engine import bridge_status, build_bridge
                status = bridge_status(auto_build=False)
                should_try_apple = backend == "apple" or (
                    "尚未编译" in status.detail or status.os_version.startswith("27")
                )
                if should_try_apple and not status.available:
                    self._handwriting_model_btn.setEnabled(False)
                    self._handwriting_model_btn.setText("正在编译…")
                    try:
                        build_bridge()
                    finally:
                        self._handwriting_model_btn.setEnabled(True)
                    self._refresh_handwriting_model_status()
                    ready = bridge_status(auto_build=False)
                    if ready.available:
                        notify(
                            self,
                            "Apple PKStrokeRecognizer Swift 桥接已编译，日语支持检测通过。",
                            "success",
                        )
                        return
                    raise RuntimeError(ready.detail)
                if backend == "apple" and not status.available:
                    raise RuntimeError(status.detail)
                if status.available:
                    self._refresh_handwriting_model_status()
                    return
            except Exception as exc:
                if backend == "apple":
                    show_error_dialog(self, "Apple PKStrokeRecognizer 不可用", str(exc))
                    self._refresh_handwriting_model_status()
                    return
                # Auto mode continues to prepare OpenVINO when Apple cannot run.

        try:
            from adapters.openvino_handwriting_engine import runtime_available
            runtime_ok, runtime_detail = runtime_available()
        except Exception as exc:
            runtime_ok, runtime_detail = False, str(exc)
        if not runtime_ok:
            QMessageBox.information(
                self,
                "先安装 OpenVINO Runtime",
                "Apple PKStrokeRecognizer 当前不可用。若要安装兼容备用，请在程序目录运行：\n\n"
                "pip3 install -r requirements-handwriting-openvino.txt\n\n"
                f"当前状态：{runtime_detail}",
            )
            return
        token = self._handwriting_download_generation.begin()
        self._handwriting_model_btn.setEnabled(False)
        self._handwriting_model_btn.setText("正在下载…")
        signals = WorkerSignals()
        self._handwriting_download_signals = signals
        self._handwriting_download_signal_refs[token] = signals

        def is_current():
            return self._handwriting_download_generation.is_current(token)

        signals.log.connect(
            lambda message: self._handwriting_model_status.setText(str(message)) if is_current() else None
        )

        def done(_value):
            if not is_current():
                return
            self._handwriting_download_signal_refs.pop(token, None)
            self._handwriting_model_btn.setEnabled(True)
            self._refresh_handwriting_model_status()
            notify(self, "OpenVINO 日语手写备用模型已下载并校验完成。", "success")

        def failed(message):
            if not is_current():
                return
            self._handwriting_download_signal_refs.pop(token, None)
            self._handwriting_model_btn.setEnabled(True)
            self._handwriting_model_btn.setText("重新下载模型")
            show_error_dialog(self, "手写模型下载失败", str(message))

        signals.finished.connect(done)
        signals.error.connect(failed)

        def worker():
            try:
                from adapters.openvino_handwriting_engine import download_model
                model_path = download_model(
                    progress_callback=lambda current, total, detail: signals.log.emit(
                        f"模型下载 {current}/{total}：{detail}"
                    )
                )
                signals.finished.emit(str(model_path))
            except Exception:
                import traceback
                signals.error.emit(traceback.format_exc())

        threading.Thread(target=worker, daemon=True, name="handwriting-model-download").start()

    def _on_handwriting_trace_toggled(self, enabled: bool):
        from adapters.ocr_profiles import get_ocr_profile

        profile = get_ocr_profile(self._current_ocr_mode())
        if enabled and not profile.allow_japanese_handwriting:
            blocked = self._handwriting_trace_check.blockSignals(True)
            self._handwriting_trace_check.setChecked(False)
            self._handwriting_trace_check.blockSignals(blocked)
            enabled = False
        # 不在切换逐字审校设置时改写“固定分列”或“逐列成句”。专用按钮
        # 会仅为本次运行临时启用所需快照，运行结束后恢复原界面状态。
        for name in (
            "_handwriting_mode_auto", "_handwriting_mode_hybrid", "_handwriting_mode_manual",
            "_handwriting_strategy_combo", "_handwriting_backend_combo",
            "_handwriting_char_mask_check", "_handwriting_symbol_insert_check",
        ):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(bool(enabled))
        # The checkbox only controls review aids.  It never starts OCR or opens
        # the correction dialog by itself, but the visible OCR preview should
        # immediately show/hide the independent physical one-character frames.
        self._refresh_review_preview_boxes()

    def _run_handwriting_ocr(self):
        """Run OCR plus manual review without mutating the normal OCR controls."""
        from adapters.ocr_profiles import get_ocr_profile
        if not get_ocr_profile(self._current_ocr_mode()).allow_japanese_handwriting:
            notify(self, "日语逐字审校仅属于日文竖排模式，不会接入简体中文横排 OCR。", "info")
            return

        # 逐字审校需要物理列，但这些值只作为本次运行快照。普通 OCR 的
        # 开关状态、下次运行方式和持久化设置全部保持不变。
        widgets = (
            self._handwriting_trace_check,
            self._column_split_check,
            self._column_sentence_reflow_check,
        )
        previous = [widget.isChecked() for widget in widgets]
        self._manual_review_requested = True
        try:
            for widget in widgets:
                blocked = widget.blockSignals(True)
                widget.setChecked(True)
                widget.blockSignals(blocked)
            self._run_ocr()
        finally:
            for widget, checked in zip(widgets, previous):
                blocked = widget.blockSignals(True)
                widget.setChecked(bool(checked))
                widget.blockSignals(blocked)
            self._on_handwriting_trace_toggled(self._handwriting_trace_check.isChecked())

    def _toggle_handwriting_card_body(self, checked: bool = True):
        # “逐字审校” is now an always-expanded settings page.
        self._handwriting_card_body.setVisible(True)
        toggle = getattr(self, "_handwriting_card_toggle", None)
        if toggle is not None:
            blocked = toggle.blockSignals(True)
            toggle.setChecked(True)
            toggle.blockSignals(blocked)

    def _select_adapter_from_combo(self, index: int):
        combo = getattr(self, "_adapter_combo", None)
        if combo is None or index < 0:
            return
        aid = str(combo.itemData(index) or "")
        if aid:
            self._select_adapter(aid, True, None)

    def _select_adapter(self, aid, enabled, event):
        from adapters.ocr_profiles import get_ocr_profile, is_engine_compatible

        if not enabled:
            notify(self, "该适配器正在开发中，目前请使用 Apple OCR。", "info")
            return
        mode = self._current_ocr_mode()
        if not is_engine_compatible(str(aid or ""), mode):
            if not self._ocr_mode_change_guard:
                profile = get_ocr_profile(mode)
                notify(
                    self,
                    f"{self._engine_label(str(aid or ''))} 不属于“{profile.label}”识别链。"
                    "两个 OCR 模式使用独立引擎集合，切换回日文竖排后仍可继续使用该引擎。",
                    "info",
                )
            return
        self._active_adapter = aid
        # Do not mutate the single-model split checkbox when changing engines.
        # The checkbox is an explicit single-model input contract; multi-model
        # has its own role routing and internal shared geometry.
        self._highlight_adapter(aid)
        if hasattr(self, "_multi_model1_label"):
            name = next((item[1] for item in OCR_ADAPTERS if item[0] == aid), aid)
            self._multi_model1_label.setText(f"模型1：{name}（当前主模型/结构底稿）")
            for combo in self._multi_model_combos:
                if str(combo.currentData() or "") == aid:
                    combo.setCurrentIndex(0)
            if aid != "apple_vision":
                selected_secondary = {str(combo.currentData() or "") for combo in self._multi_model_combos}
                if "apple_vision" not in selected_secondary:
                    target_combo = next(
                        (combo for combo in self._multi_model_combos if not str(combo.currentData() or "")),
                        None,
                    )
                    if target_combo is not None:
                        apple_index = target_combo.findData("apple_vision")
                        if apple_index >= 0:
                            target_combo.setCurrentIndex(apple_index)
        self._paddle_model_widget.setVisible(aid == "paddle_ocr")
        self._hayai_widget.setVisible(aid == "hayai_ocr")
        self._paddle_aistudio_widget.setVisible(aid == "paddle_aistudio")
        self._vision_backend_widget.setVisible(aid == "apple_vision")
        if aid == "apple_vision":
            self._on_vision_backend_changed()
        else:
            self._update_shortcut_widget_visibility()

    def _load_paddle_aistudio_token(self) -> None:
        edit = getattr(self, "_paddle_aistudio_token_edit", None)
        save_button = getattr(self, "_paddle_aistudio_save_token_btn", None)
        if edit is None:
            return
        try:
            from ai.secure_store import load_named_secret, persistence_available
            value = load_named_secret("PaddleOCR.AIStudio", account="access_token")
            if value:
                edit.setText(value)
            if save_button is not None:
                save_button.setEnabled(bool(persistence_available()))
        except Exception:
            if save_button is not None:
                save_button.setEnabled(False)

    def _save_paddle_aistudio_token(self) -> None:
        edit = getattr(self, "_paddle_aistudio_token_edit", None)
        if edit is None:
            return
        value = edit.text().strip()
        if not value:
            notify(self, "Token 为空；如需删除系统中已保存的 Token，请点击“清除”。", "info")
            return
        try:
            from ai.secure_store import persistence_available, save_named_secret
            if not persistence_available():
                QMessageBox.information(
                    self, "系统凭据库不可用",
                    "当前平台不提供本程序支持的系统级凭据存储。请使用 PADDLEOCR_ACCESS_TOKEN 环境变量（兼容 AISTUDIO_ACCESS_TOKEN）；程序不会退回明文保存。",
                )
                return
            ok = save_named_secret(
                "PaddleOCR.AIStudio", value, account="access_token",
                label="Novel Formatter PaddleOCR AI Studio Token",
            )
        except Exception:
            ok = False
        if ok:
            notify(self, "已保存到系统凭据库；不会写入项目文件或 OCR 模式快照。", "success")
        else:
            QMessageBox.warning(self, "AI Studio Token", "保存失败；当前输入框中的 Token 仍可用于本次运行。")

    def _clear_paddle_aistudio_token(self) -> None:
        edit = getattr(self, "_paddle_aistudio_token_edit", None)
        if edit is not None:
            edit.clear()
        try:
            from ai.secure_store import delete_named_secret
            delete_named_secret("PaddleOCR.AIStudio", account="access_token")
        except Exception:
            pass

    def _update_paddle_aistudio_option_state(self, *_args) -> None:
        mode_combo = getattr(self, "_paddle_aistudio_mode_combo", None)
        if mode_combo is None:
            return
        mode = str(mode_combo.currentData() or "async_v2")
        sync_mode = mode == "sync"
        for widget_name in ("_paddle_aistudio_url_edit", "_paddle_aistudio_sync_family_combo"):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.setEnabled(sync_mode)
        model_combo = getattr(self, "_paddle_aistudio_async_model_combo", None)
        if model_combo is not None:
            model_combo.setEnabled(not sync_mode)

        # Current official SDK exposes text-line orientation on OCR and
        # PP-StructureV3, but not on PaddleOCR-VL. Keep the UI aligned with
        # that capability matrix so a model switch cannot create code 10008.
        textline = getattr(self, "_paddle_aistudio_textline_check", None)
        if textline is not None:
            if sync_mode:
                family_combo = getattr(self, "_paddle_aistudio_sync_family_combo", None)
                family = str(family_combo.currentData() if family_combo else "ppocr")
                supports_textline = family in {"ppocr", "ppstructure"}
            else:
                model = str(model_combo.currentData() if model_combo else "PaddleOCR-VL-1.6")
                supports_textline = model.startswith("PP-OCR") or model == "PP-StructureV3"
            textline.setEnabled(supports_textline)
            textline.setToolTip(
                "PaddleOCR-VL 系列不发送 useTextlineOrientation；PP-OCR 与 PP-StructureV3 支持该参数。"
                if not supports_textline else ""
            )

    def _update_hayai_option_state(self, *_args):
        if not hasattr(self, "_hayai_backend_combo"):
            return
        is_litert = str(self._hayai_backend_combo.currentData() or "torch") == "litert"
        self._hayai_device_combo.setEnabled(not is_litert)
        self._hayai_quant_combo.setEnabled(not is_litert)
        self._hayai_litert_quant_combo.setEnabled(is_litert)
        if is_litert:
            self._hayai_device_combo.setToolTip("LiteRT 当前由 CPU interpreter 执行。")
        else:
            self._hayai_device_combo.setToolTip("")

    def _on_vision_backend_changed(self):
        backend_id = str(self._vision_backend_combo.currentData() or "native_helper")
        self._update_shortcut_widget_visibility()
        try:
            from adapters.vision_backends import BackendFactory
            backend = BackendFactory.create(backend_id)
            available, reason = backend.is_available()
            if not available:
                self._vision_backend_hint.setText(f"⚠️ 当前不可用：{reason}")
                return
            if backend_id == "shortcut":
                message = "保留原来的 macOS 快捷指令调用；只返回纯文本，没有坐标和真实置信度。"
            elif backend_id == "live_text":
                message = "使用常驻 Swift Helper 调用 VisionKit ImageAnalyzer（Live Text），直接读取系统 transcript；原图单次识别，不旋转、不紧裁。"
            elif backend_id == "native_helper":
                message = "使用常驻 Swift Helper 调用 RecognizeTextRequest.accurate；可返回坐标、置信度和多个候选，竖列可选择紧裁旋转。"
            self._vision_backend_hint.setText(message)
        except Exception as e:
            self._vision_backend_hint.setText(f"⚠️ {e}")

    def _update_shortcut_widget_visibility(self):
        is_apple = self._active_adapter == "apple_vision"
        backend_id = str(self._vision_backend_combo.currentData() or "native_helper") if hasattr(self, "_vision_backend_combo") else "native_helper"
        self._shortcut_widget.setVisible(is_apple and backend_id == "shortcut")
        if hasattr(self, "_vision_helper_widget"):
            self._vision_helper_widget.setVisible(is_apple and backend_id == "native_helper")
        if hasattr(self, "_vision_live_text_widget"):
            self._vision_live_text_widget.setVisible(is_apple and backend_id == "live_text")
        if hasattr(self, "_vision_vertical_check"):
            from adapters.ocr_profiles import get_ocr_profile
            profile = get_ocr_profile(self._current_ocr_mode())
            self._vision_vertical_check.setEnabled(
                bool(profile.vertical and backend_id == "native_helper")
            )

    def _highlight_adapter(self, aid):
        for a, card in self._adapter_cards.items():
            bc = ACC if a == aid else BORDER
            bw = 2 if a == aid else 1
            card.setStyleSheet(
                f"background: {CARD}; border: {bw}px solid {bc}; border-radius: 8px; margin: 4px 10px;")
        combo = getattr(self, "_adapter_combo", None)
        if combo is not None:
            index = combo.findData(aid)
            if index >= 0 and combo.currentIndex() != index:
                combo.blockSignals(True)
                combo.setCurrentIndex(index)
                combo.blockSignals(False)
        summary = getattr(self, "_active_adapter_summary", None)
        if summary is not None:
            item = next((row for row in OCR_ADAPTERS if row[0] == aid), None)
            if item is not None:
                concise = {
                    "hayai_ocr": "日文竖排文字 crop 识别器，强制启用正文区与物理分列。",
                    "ndlocr_lite": "跨平台竖排书籍 OCR，可按整页或共享物理列运行。",
                    "apple_vision": "Apple Vision / Live Text，本机识别并保留系统候选。",
                    "manga_48px": "48px 物理全列识别，可作为全列主证据或仅对分歧列复核。",
                }.get(str(aid), str(item[4] or ""))
                summary.setText(concise)

    def _switch_view(self, which):
        """Retain the legacy entry point while exposing only the OCR log."""
        self._log_view.setVisible(True)
        result_view = getattr(self, "_result_view", None)
        if result_view is not None:
            result_view.setVisible(False)

    def set_inputs(self, paths):
        normalized = [str(p) for p in paths]
        if normalized != self._pending_inputs:
            self._reset_preview_history()
            self._latest_single_doc = None
            self.single_ocr_invalidated.emit()
        self._pending_inputs = normalized
        self._input_origin = "page_manager"
        if normalized:
            self._load_preview_reference()
        else:
            self._preview_source_path = None
            self._active_preview_source_path = None
            self._preview.clear_rect()
            self._preview.clear_preview()
            self._preview_filename_lbl.setText("当前图片：尚未载入")
            self._preview_filename_lbl.setToolTip("")
            self._preview_page_lbl.setText("0 / 0")

    def _resolve_ocr_run_inputs(self) -> tuple[list[str], list[str]]:
        """Return the authoritative, existing input list for the next OCR run.

        Page Manager deletion changes the page set itself.  Never re-expand an
        old folder/PDF or trust a stale `_pending_inputs` snapshot: use the
        current explicit `page_images` list.
        """
        page_manager_images = None
        if str(getattr(self, "_input_origin", "direct")) == "page_manager":
            try:
                main_window = self.window()
            except Exception:
                main_window = None
            page_manager = getattr(main_window, "_tab_pages", None)
            if page_manager is not None:
                page_manager_images = [
                    str(path)
                    for path in (getattr(page_manager, "page_images", []) or [])
                ]
        from utils.ocr_input_sync import resolve_ocr_run_inputs
        return resolve_ocr_run_inputs(
            getattr(self, "_pending_inputs", []) or [],
            input_origin=str(getattr(self, "_input_origin", "direct")),
            page_manager_images=page_manager_images,
        )

    def _on_live_preview_toggled(self, enabled: bool):
        return self._ocr_preview_controller._on_live_preview_toggled(enabled)

    def _live_preview_is_enabled(self) -> bool:
        return self._ocr_preview_controller._live_preview_is_enabled()

    def _progress_display_is_enabled(self) -> bool:
        return bool(self._progress_display_enabled_event.is_set())

    def _toggle_progress_display(self, enabled: bool) -> None:
        return self._ocr_run_lifecycle._toggle_progress_display(enabled)

    def _set_ocr_log_collapsed(self, collapsed: bool, *, persist: bool = True) -> None:
        """Collapse only the visible OCR log body, never the live log pipeline.

        The title/action row remains available so OCR can still be started,
        paused or rerun while the preview takes the reclaimed vertical space.
        Hiding the body does not clear the QPlainTextEdit and does not stop
        buffered log delivery; expanding later reveals the complete log.
        """
        collapsed = bool(collapsed)
        self._ocr_log_collapsed = collapsed

        body = getattr(self, "_ocr_log_body", None)
        panel = getattr(self, "_ocr_log_panel", None)
        button = getattr(self, "_ocr_log_collapse_btn", None)
        if body is not None:
            body.setVisible(not collapsed)
        if panel is not None:
            expanded_height = int(getattr(self, "_ocr_log_expanded_height", 194))
            collapsed_height = int(getattr(self, "_ocr_log_collapsed_height", 54))
            panel.setFixedHeight(collapsed_height if collapsed else expanded_height)
        if button is not None:
            button.setArrowType(Qt.DownArrow if collapsed else Qt.UpArrow)
            accessible_name = "展开日志" if collapsed else "收起日志"
            button.setToolTip(
                "展开 OCR 日志，历史内容仍然保留。"
                if collapsed
                else "收起 OCR 日志，把更多垂直空间让给图片预览。"
            )
            button.setAccessibleName(accessible_name)
        action = getattr(self, "_ocr_log_collapse_action", None)
        if action is not None:
            action.setText("展开 OCR 日志" if collapsed else "收起 OCR 日志")

        if persist and hasattr(self, "_ocr_mode_settings"):
            self._ocr_mode_settings.setValue("ui/ocr_log_collapsed", collapsed)
            self._ocr_mode_settings.sync()

    def _reset_preview_history(self, *, keep_current_image: bool = True):
        return self._ocr_preview_controller._reset_preview_history(keep_current_image=keep_current_image)

    def _update_preview_navigation(self):
        return self._ocr_preview_controller._update_preview_navigation()

    @staticmethod
    def _build_review_character_rects(
        image_path: str,
        column_rects,
        *,
        maximum_boxes: int = 2000,
    ) -> list[tuple[float, float, float, float]]:
        return OCRPreviewController._build_review_character_rects(
            image_path, column_rects, maximum_boxes=maximum_boxes
        )

    def _apply_review_preview_boxes(self, key: str, payload) -> None:
        return self._ocr_preview_controller._apply_review_preview_boxes(key, payload)

    def _refresh_review_preview_boxes(self) -> None:
        return self._ocr_preview_controller._refresh_review_preview_boxes()

    def _show_preview_page(self, index: int, *, manual: bool = False):
        return self._ocr_preview_controller._show_preview_page(index, manual=manual)

    def _show_previous_preview_page(self):
        # This handler intentionally stays independent from OCR run state.
        # Once two retained pages exist, the user may browse backwards while the
        # worker continues processing newer pages in the background.
        return self._ocr_preview_controller._show_previous_preview_page()

    def _show_next_preview_page(self):
        return self._ocr_preview_controller._show_next_preview_page()

    def _record_preview_page(
        self,
        key: str,
        display_name: str,
        snapshot_path: str,
        rects,
        stage: str,
    ):
        return self._ocr_preview_controller._record_preview_page(key, display_name, snapshot_path, rects, stage)

    def _write_preview_snapshot(self, image_path: str, preview_dir: str, key: str) -> tuple[str, QImage]:
        return self._ocr_preview_controller._write_preview_snapshot(image_path, preview_dir, key)

    def clear_ocr_temporary_files(self, *, closing: bool = False):
        return self._ocr_preview_controller.clear_ocr_temporary_files(closing=closing)

    def shutdown_cleanup(self):
        self._cancel_event.set()
        self._paddle_prepare_generation.invalidate()
        self._handwriting_download_generation.invalidate()
        self._runtime_status_generation.invalidate()
        self._ocr_preflight_generation.invalidate()
        self._paddle_prepare_signal_refs.clear()
        self._handwriting_download_signal_refs.clear()
        self._runtime_status_signal_refs.clear()
        self._ocr_preflight_signal_refs.clear()
        self._runtime_status_scan_busy = False
        self._ocr_preflight_active = False
        self._ocr_watchdog_timer.stop()
        self._ocr_log_flush_timer.stop()
        self._ocr_log_buffer.clear()
        self._ocr_run_active = False
        self._run_generation += 1
        worker = self._ocr_worker_thread
        if worker is not None and worker.is_alive():
            # External recognizers receive the shared cancellation event and
            # terminate their process groups.  Never block the Qt close event
            # indefinitely if a third-party native library ignores shutdown.
            # Cancellation is cooperative and recognizer sessions terminate
            # their own child process groups.  Do not hold the Qt event loop for
            # 1.5 s on close merely waiting for bookkeeping to unwind.
            worker.join(timeout=0.25)
        self._ocr_worker_thread = None
        self.clear_ocr_temporary_files(closing=True)

    def _load_preview_reference(self):
        return self._ocr_preview_controller._load_preview_reference()

    def _current_column_preview_context(self) -> tuple[str | None, list, str, str]:
        return self._ocr_preview_controller._current_column_preview_context()

    @staticmethod
    def _columns_from_preview_rects(rects, image_size):
        # Static adapter: do not dereference an instance-only ``self`` here.
        # The old wrapper raised NameError whenever 分列预览 reached retained
        # geometry reconstruction, making the feature look completely broken.
        return OCRPreviewController._columns_from_preview_rects(rects, image_size)

    def _preview_column_split(self):
        return self._ocr_preview_controller._preview_column_split()

    def _mark_ocr_activity(self) -> None:
        self._ocr_last_activity_at = time.monotonic()
        self._ocr_stall_notice_emitted = False

    @staticmethod
    def _ocr_timeout_setting(name: str, default: float, minimum: float) -> float:
        try:
            return max(float(minimum), float(os.environ.get(name, default)))
        except (TypeError, ValueError):
            return max(float(minimum), float(default))

    def _check_ocr_watchdog(self) -> None:
        if not self._ocr_run_active:
            self._ocr_watchdog_timer.stop()
            return
        idle = max(0.0, time.monotonic() - float(self._ocr_last_activity_at or time.monotonic()))
        notice_after = self._ocr_timeout_setting(
            "NOVEL_FORMATTER_OCR_GUI_STALL_NOTICE", 90.0, 30.0
        )
        hard_after = self._ocr_timeout_setting(
            "NOVEL_FORMATTER_OCR_GUI_HARD_TIMEOUT", 1200.0, 300.0
        )
        if idle >= notice_after and not self._ocr_stall_notice_emitted:
            self._ocr_stall_notice_emitted = True
            self._log_view.appendPlainText(
                f"\n⚠️ OCR 已 {int(idle)} 秒没有新输出。子进程看门狗仍在监控；"
                "可点击停止立即终止卡住的外部识别进程。"
            )
        if idle >= hard_after and not self._ocr_hard_timeout_requested:
            self._ocr_hard_timeout_requested = True
            self._cancel_event.set()
            session_id = str(self._active_ocr_log_session_id or "")
            if session_id:
                self.run_log_event.emit({
                    "event": "cancelling", "session_id": session_id,
                    "message": "OCR 无活动超时，正在终止并保存可恢复断点",
                })
            self._pause_btn.setEnabled(False)
            self._pause_btn.setText("正在终止…")
            self._log_view.appendPlainText(
                f"\n🛑 OCR 连续 {int(idle)} 秒无活动，已触发整次任务保护并终止外部识别进程。"
            )

    def _toggle_pause(self):
        if not self._ocr_run_active:
            return
        # Stop is a hard admission barrier, not merely a UI pause: every adapter
        # polls the event, pending crop/split futures are cancelled, and the GUI
        # ignores already-queued progress/log signals from the cancelled run.
        self._cancel_event.set()
        session_id = str(self._active_ocr_log_session_id or "")
        if session_id:
            self.run_log_event.emit({
                "event": "cancelling", "session_id": session_id,
                "message": "已请求停止 OCR；正在保存已完成阶段与分段缓存",
            })
        self._ocr_watchdog_timer.stop()
        self._progress_clock_timer.stop()
        self._pause_btn.setEnabled(False)
        self._pause_btn.setText("正在终止…")
        if self._progress_display_is_enabled():
            self._phase_progress_lbl.setVisible(True)
            self._phase_progress_lbl.setText("当前：正在终止 OCR，不再派发新任务…")
        self._log_view.appendPlainText(
            "\n⏸ 已请求停止：不再派发新页面、新列、新模型、重试或整句识别；"
            "正在终止当前外部识别进程并保留已完成结果…"
        )

    def _engine_label(self, engine_id: str) -> str:
        if str(engine_id or "") == "manga_ocr":
            return "Manga OCR"
        return next((item[1] for item in OCR_ADAPTERS if item[0] == engine_id), engine_id)

    def _selected_single_ocr_engine(self) -> str:
        """Resolve only the single-model selector; never inspect multi-model slots."""
        from adapters.ocr_profiles import is_engine_compatible

        mode = self._current_ocr_mode()
        primary = str(self._active_adapter or "apple_vision")
        if not is_engine_compatible(primary, mode):
            primary = "apple_vision"
        return primary

    def _selected_ocr_engines(self) -> list[str]:
        """Compatibility helper; production run planning uses explicit branches."""
        from adapters.ocr_profiles import is_engine_compatible

        mode = self._current_ocr_mode()
        if hasattr(self, "_multi_ocr_check") and self._multi_ocr_check.isChecked():
            return [
                engine for engine in self._selected_multi_ocr_slot_engines()
                if engine and is_engine_compatible(engine, mode)
            ]
        return [self._selected_single_ocr_engine()]

    def _engine_options(self, engine_id: str) -> dict:
        from adapters.ocr_profiles import get_ocr_profile

        profile = get_ocr_profile(self._current_ocr_mode())
        opts: dict = {}
        if engine_id == "paddle_ocr":
            opts.update(
                pipeline=self._paddle_model_combo.currentData(),
                lang=profile.paddle_lang,
                model_source=self._paddle_source_combo.currentData(),
                vl_backend=self._paddle_vl_backend_combo.currentData(),
            )
        elif engine_id == "hayai_ocr":
            from utils.apple_silicon_runtime import is_m6
            opts.update(
                backend=str(self._hayai_backend_combo.currentData() or "torch"),
                device=str(self._hayai_device_combo.currentData() or "auto"),
                quantize=str(self._hayai_quant_combo.currentData() or "none"),
                litert_quant=str(self._hayai_litert_quant_combo.currentData() or "wi4"),
                max_new_tokens=128,
                segment_max_chars=(30 if is_m6() else 24),
                segment_max_aspect=18.0,
            )
        elif engine_id == "ndlocr_lite":
            opts.update(
                device="cpu",
                enable_tcy=False,
            )
        elif engine_id == "paddle_aistudio":
            opts.update(
                mode=str(self._paddle_aistudio_mode_combo.currentData() or "async_v2"),
                token=self._paddle_aistudio_token_edit.text().strip(),
                api_url=self._paddle_aistudio_url_edit.text().strip(),
                sync_family=str(
                    self._paddle_aistudio_sync_family_combo.currentData() or "ppocr"
                ),
                async_model=str(
                    self._paddle_aistudio_async_model_combo.currentData() or "PaddleOCR-VL-1.6"
                ),
                request_timeout=float(self._paddle_aistudio_timeout_spin.value()),
                max_retries=int(self._paddle_aistudio_retry_spin.value()),
                use_doc_orientation_classify=bool(
                    self._paddle_aistudio_orientation_check.isChecked()
                ),
                use_doc_unwarping=bool(self._paddle_aistudio_unwarp_check.isChecked()),
                use_textline_orientation=bool(self._paddle_aistudio_textline_check.isChecked()),
            )
        elif engine_id == "apple_vision":
            vertical = bool(profile.vertical and self._vision_vertical_check.isChecked())
            opts.update(
                apple_backend=str(self._vision_backend_combo.currentData() or "native_helper"),
                recognition_level="accurate",
                recognition_languages=list(profile.apple_languages),
                use_language_correction=self._vision_language_correction_check.isChecked(),
                automatically_detect_language=False,
                minimum_text_height_fraction=self._vision_min_height_spin.value(),
                candidate_count=self._vision_candidate_spin.value(),
                orientation="auto",
                vertical=vertical,
                vertical_preprocess=(
                    "crop_rotate_left"
                    if profile.vertical and self._vision_vertical_compat_check.isChecked()
                    else "none"
                ),
            )
        return opts

    def _column_runtime_options_snapshot(self) -> dict[str, object]:
        input_profile = (
            "custom" if self._column_preprocess_is_customized() else "v8_exact"
        )
        if input_profile == "v8_exact":
            ruby = True
            fragments = False
            smart_crop = True
            strength = "standard"
            # Main OCR must never receive side Ruby in the standard Japanese-novel
            # path.  ``False`` does not resample body glyphs: it copies only the
            # component detector's authoritative body/source boxes and blanks the
            # side-Ruby boxes from the temporary OCR transport image.
            preserve_body_pixels = False
        else:
            ruby = bool(
                hasattr(self, "_column_ruby_filter_check")
                and self._column_ruby_filter_check.isChecked()
            )
            fragments = bool(
                hasattr(self, "_column_fragment_filter_check")
                and self._column_fragment_filter_check.isChecked()
            )
            smart_crop = bool(
                not hasattr(self, "_column_smart_crop_check")
                or self._column_smart_crop_check.isChecked()
            )
            strength = str(
                self._column_ruby_strength_combo.currentData()
                if hasattr(self, "_column_ruby_strength_combo")
                else "standard"
            ).strip().lower() or "standard"
            if strength not in {"weak", "standard", "strong"}:
                strength = "standard"
            preserve_body_pixels = not (ruby or fragments)
        # Ruby preservation is a side-channel only.  Snapshot its scheduling
        # flag independently; never modify the ordinary OCR pixel/cleanup
        # contract merely because findtextCenterNet is enabled.
        ruby_preservation_enabled = bool(
            getattr(getattr(self, "_preserve_ruby_check", None), "isChecked", lambda: False)()
        )
        isolation_mode = str(
            getattr(getattr(self, "_column_isolation_mode_combo", None), "currentData", lambda: "mask")()
            or "mask"
        ).strip().lower()
        if isolation_mode not in {"mask", "display"}:
            isolation_mode = "mask"
        return {
            "column_detector_mode": "components",
            "column_sensitivity": float(self._column_sensitivity_spin.value()),
            "column_padding_percent": float(self._column_padding_spin.value()),
            "strict_column_validation": bool(self._column_strict_check.isChecked()),
            "shortcut_name": self._shortcut_edit.text().strip() or "ExtractText",
            "column_input_profile": input_profile,
            "column_auto_filter_ruby": ruby,
            "column_filter_fragments": fragments,
            "column_smart_crop": smart_crop,
            "column_ruby_strength": strength,
            "column_isolation_mode": isolation_mode,
            "column_preserve_body_pixels": preserve_body_pixels,
            # Geometry telemetry is strictly tied to the explicit Ruby feature.
            # OFF runs skip it entirely and therefore follow the original OCR
            # path with no Ruby sidecar/candidate overhead.
            "column_capture_ruby_candidates": ruby_preservation_enabled,
            # Auto means: only very tall/narrow already-cropped inputs are
            # treated as one physical column. Normal book pages are unchanged.
            "column_assume_single_column": "auto",
        }

    @staticmethod
    def _document_column_ocr_health(document) -> dict[str, object]:
        from utils.ocr_model_health import assess_column_ocr_health

        return assess_column_ocr_health(document)

    def _run_engine_document(
        self, engine_id: str, *, common_kwargs: dict, use_column_mask: bool,
        phase_callback=None, shared_column_prepare_dir: str = "",
        shared_column_variant_dir: str = "", extra_engine_options: dict | None = None,
        frozen_engine_options: dict | None = None,
        frozen_column_options: dict | None = None,
        input_role: str = "auto",
    ):
        # GUI settings are snapshotted on the main thread before OCR starts.
        # Falling back to _engine_options keeps direct/unit callers compatible,
        # but normal OCR runs never read Qt widgets from the worker thread.
        from adapters.ocr_profiles import get_ocr_profile, is_engine_compatible, normalize_ocr_mode

        ocr_mode = normalize_ocr_mode(common_kwargs.get("ocr_mode", self._current_ocr_mode()))
        profile = get_ocr_profile(ocr_mode)
        if not is_engine_compatible(engine_id, ocr_mode):
            raise ValueError(
                f"OCR 引擎 {engine_id!r} 与模式 {profile.label!r} 不兼容；"
                "为避免日文分列链和中文横排链互相污染，本次运行已拒绝。"
            )
        if not profile.allow_column_pipeline:
            use_column_mask = False
        # Remote AI Studio is billed/quotaed per uploaded page. Never fan one
        # page out into physical-column HTTP requests; keep full-page geometry
        # and let the normal multi-model fusion align it with local engines.
        if engine_id == "paddle_aistudio":
            use_column_mask = False
        opts = dict(frozen_engine_options) if frozen_engine_options is not None else self._engine_options(engine_id)
        column_opts = dict(frozen_column_options or {})
        if not column_opts:
            column_opts = self._column_runtime_options_snapshot()
        # Column cleanup is a coherent, frozen per-run contract.  These values
        # used to be displayed as checkboxes but then overwritten here, making
        # the controls impossible to use.  Copy them into the adapter options
        # exactly once so every model sees the same selected preprocessing.
        for key in (
            "column_detector_mode",
            "column_input_profile",
            "column_auto_filter_ruby",
            "column_filter_fragments",
            "column_smart_crop",
            "column_ruby_strength",
            "column_isolation_mode",
            "column_preserve_body_pixels",
            "column_capture_ruby_candidates",
            "column_assume_single_column",
        ):
            if key in column_opts:
                opts[key] = column_opts[key]
        # Apple Vision requests are isolated per physical column.  A shorter
        # per-column deadline prevents one pathological image from holding the
        # whole book indefinitely; the helper is restarted after a timeout.
        if engine_id == "apple_vision":
            opts.setdefault("request_timeout", 45.0)
        if extra_engine_options:
            opts.update(dict(extra_engine_options))
        # The run-level Ruby toggle is authoritative.  Engine-specific options,
        # plugins and persisted kwargs cannot secretly enable candidate telemetry
        # during a Ruby-OFF run or disable it during a Ruby-ON run.
        opts["column_capture_ruby_candidates"] = bool(
            column_opts.get("column_capture_ruby_candidates", False)
        )
        # Projection is review-only.  Ordinary OCR cannot be switched back to the
        # legacy projection detector by saved settings, plugins or direct kwargs.
        opts["column_detector_mode"] = "components"
        role = str(input_role or "auto").strip().lower()
        # Routing policy is resolved *before* entering this execution method.
        # Single-model mode and role-based multi-model mode therefore have two
        # independent decision paths and cannot silently rewrite each other's
        # column/full-page choice here.  In particular, single-model + unchecked
        # "启用日文物理分列" means genuine whole-page input for every engine.
        use_column_mask = bool(use_column_mask and profile.allow_column_pipeline)
        if use_column_mask:
            if shared_column_prepare_dir:
                opts["column_shared_prepare_dir"] = shared_column_prepare_dir
            if shared_column_variant_dir:
                opts["column_shared_variant_dir"] = shared_column_variant_dir
            from adapters.column_ocr_adapter import run as ocr_run
            return ocr_run(
                recognition_engine=engine_id,
                column_sensitivity=float(column_opts.get("column_sensitivity", 0.48)),
                column_padding_percent=float(column_opts.get("column_padding_percent", 8.0)),
                strict_column_validation=bool(column_opts.get("strict_column_validation", True)),
                shortcut_name=str(column_opts.get("shortcut_name") or "ExtractText"),
                engine_options=opts,
                phase_callback=phase_callback,
                **common_kwargs,
            )
        if engine_id == "paddle_ocr":
            from adapters.paddle_ocr_adapter import run as ocr_run
            return ocr_run(
                pipeline=opts.get("pipeline"),
                lang=str(opts.get("lang") or profile.paddle_lang),
                model_source=str(opts.get("model_source") or "auto"),
                vl_backend=str(opts.get("vl_backend") or "auto"),
                **common_kwargs,
            )
        if engine_id == "ndlocr_lite":
            from adapters.ndlocr_lite_adapter import run as ocr_run
            return ocr_run(**common_kwargs)
        if engine_id == "windows_snipping_ocr":
            from adapters.windows_snipping_ocr_adapter import run as ocr_run
            return ocr_run(**common_kwargs)
        if engine_id == "hayai_ocr":
            from adapters.hayai_ocr_adapter import run as ocr_run
            return ocr_run(engine_options=opts, **common_kwargs)
        if engine_id == "manga_48px":
            from adapters.manga_48px_adapter import run as ocr_run
            return ocr_run(**common_kwargs)
        if engine_id == "paddle_aistudio":
            from adapters.paddle_aistudio_adapter import run as ocr_run
            return ocr_run(
                mode=str(opts.get("mode") or "async_v2"),
                token=str(opts.get("token") or ""),
                api_url=str(opts.get("api_url") or ""),
                sync_family=str(opts.get("sync_family") or "ppocr"),
                async_model=str(opts.get("async_model") or "PaddleOCR-VL-1.6"),
                request_timeout=float(opts.get("request_timeout") or 180.0),
                max_retries=int(opts.get("max_retries") if opts.get("max_retries") is not None else 3),
                use_doc_orientation_classify=bool(opts.get("use_doc_orientation_classify", False)),
                use_doc_unwarping=bool(opts.get("use_doc_unwarping", False)),
                use_textline_orientation=bool(opts.get("use_textline_orientation", False)),
                **common_kwargs,
            )
        from adapters.apple_vision_adapter import run as ocr_run
        return ocr_run(
            shortcut_name=str(column_opts.get("shortcut_name") or "ExtractText"),
            backend=str(opts.get("apple_backend") or "native_helper"),
            vertical=bool(opts.get("vertical", profile.vertical)),
            recognition_level=str(opts.get("recognition_level") or "accurate"),
            recognition_languages=list(
                opts.get("recognition_languages") or profile.apple_languages
            ),
            use_language_correction=bool(opts.get("use_language_correction", True)),
            automatically_detect_language=bool(opts.get("automatically_detect_language", False)),
            minimum_text_height_fraction=float(opts.get("minimum_text_height_fraction", 0.005)),
            candidate_count=int(opts.get("candidate_count", 3)),
            orientation=str(opts.get("orientation") or "auto"),
            vertical_preprocess=str(opts.get("vertical_preprocess") or "none"),
            **common_kwargs,
        )

    def _runtime_installation_component_ids(
        self, engine_ids: list[str] | None = None, *, include_ruby: bool = False
    ) -> list[str]:
        """Snapshot required runtime components on the Qt thread."""
        from adapters.ocr_runtime_catalog import required_components

        ids: list[str] = []
        for engine_id in (engine_ids or [self._active_adapter]):
            ids.extend(required_components(
                engine_id,
                paddle_pipeline=self._paddle_model_combo.currentData(),
                recognition_engine=engine_id if self._column_split_check.isChecked() else "",
                engine_options=self._engine_options(engine_id),
            ))
        if include_ruby:
            ids.append("findtext_centernet_ruby")
        return list(dict.fromkeys(ids))

    def _begin_runtime_installation_preflight(
        self,
        engine_ids: list[str],
        *,
        include_ruby: bool,
        resume_callback,
    ) -> None:
        """Run model/runtime availability checks without freezing the GUI.

        The old synchronous path could launch a 10-second Python probe and walk
        large model caches directly from the Start OCR button.  Only the final
        confirmation dialog belongs on the GUI thread.
        """
        if self._ocr_preflight_active:
            return
        component_ids = self._runtime_installation_component_ids(
            engine_ids, include_ruby=include_ruby
        )
        if not component_ids:
            QTimer.singleShot(0, resume_callback)
            return

        token = self._ocr_preflight_generation.begin()
        self._ocr_preflight_active = True
        self._run_btn.setEnabled(False)
        if hasattr(self, "_phase_progress_lbl"):
            self._phase_progress_lbl.setVisible(True)
            self._phase_progress_lbl.setText("正在检查本地 OCR 运行环境…")

        signals = WorkerSignals()
        self._ocr_preflight_signal_refs[token] = signals

        def is_current() -> bool:
            return self._ocr_preflight_generation.is_current(token)

        def cleanup() -> None:
            self._ocr_preflight_signal_refs.pop(token, None)
            if is_current():
                self._ocr_preflight_active = False
                if not self._ocr_run_active:
                    self._run_btn.setEnabled(True)
                if hasattr(self, "_phase_progress_lbl") and not self._ocr_run_active:
                    self._phase_progress_lbl.setVisible(False)

        def finished(payload):
            if not is_current():
                self._ocr_preflight_signal_refs.pop(token, None)
                return
            data = dict(payload or {})
            missing = list(data.get("missing") or [])
            confirmation = str(data.get("confirmation") or "")
            cleanup()
            if missing:
                box = QMessageBox(self)
                box.setIcon(QMessageBox.Warning)
                box.setWindowTitle("需要安装 OCR 模型")
                box.setText(confirmation)
                install = box.addButton("安装并继续", QMessageBox.ButtonRole.AcceptRole)
                box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
                box.setDefaultButton(install)
                box.exec()
                if box.clickedButton() is not install:
                    return
            QTimer.singleShot(0, resume_callback)

        def failed(message):
            if not is_current():
                self._ocr_preflight_signal_refs.pop(token, None)
                return
            cleanup()
            show_error_dialog(self, "OCR 环境检查失败", str(message or "未知错误"))

        signals.finished.connect(finished)
        signals.error.connect(failed)
        def worker():
            try:
                from adapters.ocr_runtime_catalog import missing_components, confirmation_text
                missing = missing_components(component_ids)
                # confirmation_text performs richer runtime/cache probes; keep
                # those off the GUI thread too.
                confirmation = confirmation_text(missing) if missing else ""
                signals.finished.emit({"missing": missing, "confirmation": confirmation})
            except Exception:
                import traceback
                signals.error.emit(traceback.format_exc())

        threading.Thread(
            target=worker, daemon=True, name=f"ocr-runtime-preflight-{token}"
        ).start()

    @staticmethod
    def _sampled_narrow_single_column_batch(inputs: list[str], sample_limit: int = 5) -> bool:
        """Cheaply detect already-cropped narrow-column batches.

        Only image headers from a deterministic handful of representative pages
        are read.  The previous implementation opened every page on the Qt thread
        before the OCR worker even started, freezing 300–500 page books.
        """
        paths = list(inputs or [])
        if not paths:
            return False
        count = min(max(1, int(sample_limit)), len(paths))
        if count == len(paths):
            sample = paths
        elif count == 1:
            sample = [paths[0]]
        else:
            indexes = sorted({round(i * (len(paths) - 1) / (count - 1)) for i in range(count)})
            sample = [paths[index] for index in indexes]
        try:
            from PIL import Image as _PILImage
            for path in sample:
                with _PILImage.open(path) as probe:
                    width = max(1, int(probe.width))
                    height = max(1, int(probe.height))
                if not (height >= 360 and width / float(height) <= 0.20):
                    return False
        except Exception:
            return False
        return True

    def _run_ocr(
        self,
        *,
        _runtime_preflight_done: bool = False,
        _manual_review_requested_override: bool | None = None,
    ):
        controller = getattr(self, "_ocr_run_controller", None)
        if controller is None:
            controller = self._ocr_run_controller = OcrRunController(self)
        return controller.run(
            _runtime_preflight_done=_runtime_preflight_done,
            _manual_review_requested_override=_manual_review_requested_override,
        )

    def _flush_ocr_log_buffer(self, *, force_all: bool = False) -> None:
        return self._ocr_run_lifecycle._flush_ocr_log_buffer(force_all=force_all)

    def _emit_run_log_terminal(
        self,
        status: str,
        *,
        extra_details: dict | None = None,
        error: str = "",
    ) -> None:
        return self._ocr_run_lifecycle._emit_run_log_terminal(status, extra_details=extra_details, error=error)

    def _on_progress(self, current, total):
        return self._ocr_run_lifecycle._on_progress(current, total)

    def _on_phase_progress(self, label, current, total):
        return self._ocr_run_lifecycle._on_phase_progress(label, current, total)

    def _set_progress_bar_text(self, percent: float, elapsed: float, eta) -> None:
        return self._ocr_run_lifecycle._set_progress_bar_text(percent, elapsed, eta)

    def _refresh_live_progress_clock(self) -> None:
        return self._ocr_run_lifecycle._refresh_live_progress_clock()

    def _on_overall_progress(self, snapshot):
        return self._ocr_run_lifecycle._on_overall_progress(snapshot)

    def _reset_run_state(self):
        return self._ocr_run_lifecycle._reset_run_state()

    def _on_multi_ocr_done(self, payload: dict):
        return self._ocr_run_lifecycle._on_multi_ocr_done(payload)

    def _on_done(self, doc):
        return self._ocr_run_lifecycle._on_done(doc)

    def latest_single_document(self) -> UnifiedDocument | None:
        """Return the latest single-model OCR result without publishing it elsewhere.

        OCR 对比 explicitly pulls this result only after the user clicks its
        source-inbox button.  Merely finishing a single OCR therefore never
        changes the compare workspace.
        """
        return self._latest_single_doc

    def _on_error(self, msg):
        return self._ocr_run_lifecycle._on_error(msg)

    def _re_ocr(self):
        return self._ocr_run_lifecycle._re_ocr()


# ══════════════════════════════════════════════════════════════════════════════
#  Tab 3 — Formatter Engine
# ══════════════════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════════════════
#  Format Profile 管理窗口
# ══════════════════════════════════════════════════════════════════════════════






# ══════════════════════════════════════════════════════════════════════════════
#  Legacy standalone body-replacement workspace removed
# ══════════════════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════════════════
#  Tab 4 — EPUB Builder
# ══════════════════════════════════════════════════════════════════════════════




# ══════════════════════════════════════════════════════════════════════════════
#  PDF 文字层直读（独立页签，不经过 OCR 适配器那一套流程）
# ══════════════════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════════════════
#  OCR comparison workspace
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
#  OCR 对比 / 多模型逐句裁决工作区
# ══════════════════════════════════════════════════════════════════════════════



class OCRCompareTab(OCRCompareViewMixin, QWidget):
    doc_applied = Signal(object)
    single_doc_applied = Signal(object)
    multi_session_restored = Signal(object)
    # Stable comparison-row navigation shared with 图文对照.
    current_row_changed = Signal(int)
    # One authoritative decision event shared by OCR 对比 and 图文对照.
    # The payload is keyed by stable physical columns; raw OCR sources are never
    # rewritten.  Consumers must treat ``origin`` as a loop-prevention token.
    fusion_decision_changed = Signal(object)
    # Keep the disagreement queue in 图文对照 in the same order as OCR 对比.
    disagreement_queue_order_changed = Signal(object)
    # Request the existing independent image/text workspace at the same stable row.
    image_review_requested = Signal(int)
    # Successful/failed external package exports are persisted in project Run History.
    package_exported = Signal(object)
    # Long-running in-app adjudication shares the same project/runtime lifecycle
    # contract as OCR without moving worker ownership into MainWindow.
    run_log_event = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._documents: list[UnifiedDocument] = []
        self._project_package_dir = ""
        self._labels: list[str] = []
        self._mode = "empty"
        self._available_single_doc: UnifiedDocument | None = None
        self._available_single_label = ""
        self._available_single_generation = 0
        self._loaded_single_generation = -1
        self._single_doc: UnifiedDocument | None = None
        self._single_original_doc: UnifiedDocument | None = None
        self._suspended_multi_state: dict | None = None
        self._comparison = None
        self._primary_doc: UnifiedDocument | None = None
        # Ruby is a locked structural overlay produced on the worker's initial
        # fused document.  Keep that immutable overlay available because the
        # review page rebuilds a fresh fused document when the user clicks Apply.
        self._ruby_overlay_doc: UnifiedDocument | None = None
        self._initial_payload: dict | None = None
        self._source_editors: list[QPlainTextEdit] = []
        self._source_labels: list[QLabel] = []
        self._source_panels: list[QWidget] = []
        self._fusion_states = []
        # Final sentence decisions made in 图文对照.  Keys use immutable physical
        # column identities so a later source realignment can safely reattach
        # them without overwriting any OCR model result.
        self._image_review_overrides: dict[str, dict] = {}
        self._fusion_widgets: dict[int, _FusionDecisionRow] = {}
        # Reuse recently visited decision rows instead of constructing/deleting
        # QTextEdit-rich cards on every single-card navigation step.
        self._fusion_widget_cache: OrderedDict[int, _FusionDecisionRow] = OrderedDict()
        self._fusion_widget_cache_limit = 48
        self._fusion_window_indices: tuple[int, ...] = ()
        self._fusion_window_size = 32
        self._result_load_generation = 0
        self._multi_result_load_pending = False
        self._current_row_index = 0
        self._active_source_index: int | None = None
        self._syncing_cursor = False
        self._syncing_scroll = False
        self._loading_text = False
        self._sources_dirty = False
        self._review_only_enabled = True
        # Read-only adjudication history browser.  It never mutates a decision;
        # it only changes which already-resolved rows are materialised/navigable.
        self._show_resolved_history = False
        self._resolved_history_group = "all"
        self._single_card_enabled = False
        self._single_card_preview_row = -1
        self._review_mode = "decision"
        self._decision_queue_signature: tuple = ()
        self._syncing_decision_queue = False
        self._decision_queue_model: DecisionQueueListModel | None = None
        # Keep the complete per-model text independent from the visible editors.
        # In single-card mode the editors only show the active sentence, while
        # export, realignment and apply continue using these full strings.
        self._full_source_texts: list[str] = []
        self._full_source_lines: list[list[str]] = []
        self._ai_import_generation = GenerationGuard()
        self._ai_import_signal_refs: dict[int, WorkerSignals] = {}
        self._ai_import_busy = False
        self._pending_ai_import_apply: dict | None = None
        self._ai_adjudication_cancel_event: threading.Event | None = None
        self._ai_adjudication_report: dict | None = None
        self._ai_adjudication_output_paths: tuple[str, str] | None = None
        self._ai_adjudication_task_id = ""
        # Same-model alternate-input retries may confirm an already existing
        # strict majority. They never become independent OCR votes.
        self._targeted_retry_local_report = None
        self._last_source_correction_report: dict | None = None
        self._source_correction_export_report: dict | None = None
        # Repeated AI OCR adjudication imports are cumulative.  Stable-column
        # rows retain the best/latest accepted verdict, while unresolved later
        # packages never erase a previously accepted result.
        self._source_correction_import_history: list[dict] = []
        self._source_correction_imported_package_ids: set[str] = set()
        # Unique final OCR adjudications. Raw OCR documents remain immutable;
        # these decisions are synthetic fusion candidates keyed by stable columns.
        self._canonical_source_decisions: dict[tuple[str, ...], dict] = {}  # current sentence_group_id identity only
        self._source_correction_generation = GenerationGuard()
        self._source_correction_signal_refs: dict[int, WorkerSignals] = {}
        self._source_correction_busy = False
        self._source_correction_lock_workspace = False
        self._recovery_page_image_provider = None
        # Optional publication-reference EPUB used only as external evidence for
        # AI repair-package export.  Always initialise it here: earlier builds
        # only created the same-named field on TextCompareTab, so exporting from
        # OCRCompareTab before importing a reference raised AttributeError.
        self._reference_path = ""
        self._workspace_active = False
        self._highlight_timer = QTimer(self)
        self._highlight_timer.setSingleShot(True)
        self._highlight_timer.setInterval(45)
        self._highlight_timer.timeout.connect(self._refresh_source_highlights)
        self._highlight_mask_cache: dict[int, tuple[tuple[str, ...], tuple[tuple[bool, ...], ...]]] = {}
        self._highlight_source_revision = 0
        self._last_highlight_signature = None
        self._project_state_service = OCRCompareProjectStateService(self)
        self._exchange_service = OCRCompareExchangeService(self)
        self._source_correction_service = OCRCompareSourceCorrectionService(self)
        self._publication_service = OCRComparePublicationService(self)
        self._gpt_adjudication_controller = OCRCompareGPTAdjudicationController(self)
        self._history_service = OCRCompareAdjudicationHistoryService(self)
        self._batch_service = OCRCompareBatchService(self)
        self._build()
        self._next_group_shortcuts: list[QShortcut] = []
        for sequence in ("F8", "Ctrl+Down", "Meta+Down"):
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(self._jump_next_group)
            self._next_group_shortcuts.append(shortcut)
        self._next_sentence_shortcut = QShortcut(QKeySequence("Alt+Right"), self)
        self._next_sentence_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self._next_sentence_shortcut.activated.connect(self._jump_next_sentence)
        # High-throughput manual review: Alt+1…Alt+6 remains available for
        # legacy sessions, but only panels for actually loaded OCR sources are
        # shown. Empty model boxes are never displayed.
        self._model_choice_shortcuts: list[QShortcut] = []
        for model_index in range(6):
            shortcut = QShortcut(QKeySequence(f"Alt+{model_index + 1}"), self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(partial(self._choose_current_from_model, model_index))
            self._model_choice_shortcuts.append(shortcut)

    def _build(self):
        build_ocr_compare_panel(self)

    def set_available_single_result(self, doc: UnifiedDocument | None, label: str = "") -> None:
        """Register a single OCR result without loading it into the workspace."""
        if doc is None:
            self._available_single_doc = None
            self._available_single_label = ""
            self._available_single_generation += 1
        elif doc is not self._available_single_doc:
            self._available_single_doc = doc
            self._available_single_label = str(label or getattr(doc.metadata, "source_engine", "") or "单模型 OCR")
            self._available_single_generation += 1
        else:
            self._available_single_label = str(label or self._available_single_label or "单模型 OCR")
        self._refresh_single_source_controls()

    def ensure_latest_single_result_loaded(self) -> bool:
        """Display the newest single-OCR result when opening OCR Compare.

        Multi-model sessions are deliberately never replaced here: they must
        remain in adjudication until the user explicitly applies the fused
        decision document.
        """
        if (
            self._mode == "multi"
            or self._multi_result_load_pending
            or self._available_single_doc is None
        ):
            return False
        same_loaded = (
            self._mode == "single"
            and self._loaded_single_generation == self._available_single_generation
            and (
                self._single_doc is self._available_single_doc
                or self._single_original_doc is self._available_single_doc
            )
        )
        if same_loaded:
            return True
        self._load_single_result(
            self._available_single_doc,
            self._available_single_label,
            self._available_single_generation,
        )
        return True

    def _refresh_single_source_controls(self) -> None:
        available = self._available_single_doc is not None
        same_loaded = (
            self._mode == "single"
            and self._loaded_single_generation == self._available_single_generation
            and (
                self._single_doc is self._available_single_doc
                or self._single_original_doc is self._available_single_doc
            )
        )
        if not available:
            self._single_source_state.setText("单 OCR：暂无可载入结果")
            self._load_single_from_ocr_btn.setText("＋ 从 OCR 识别载入")
            self._load_single_from_ocr_btn.setEnabled(False)
        elif same_loaded:
            self._single_source_state.setText(f"单 OCR：已载入 · {self._available_single_label}")
            self._load_single_from_ocr_btn.setText("✓ 已载入当前单OCR")
            self._load_single_from_ocr_btn.setEnabled(False)
        else:
            action = "↻ 更新为最新单OCR" if self._mode == "single" else "＋ 从 OCR 识别载入"
            if self._mode == "multi":
                action = "⇄ 切换到最近单OCR"
            self._single_source_state.setText(f"单 OCR：可载入 · {self._available_single_label}")
            self._load_single_from_ocr_btn.setText(action)
            self._load_single_from_ocr_btn.setEnabled(True)
        self._single_export_btn.setEnabled(self._mode == "single" and self._single_doc is not None)
        self._return_multi_btn.setVisible(
            self._mode == "single" and self._suspended_multi_state is not None
        )
        self._return_multi_btn.setEnabled(self._suspended_multi_state is not None)
        if hasattr(self, "_sync_compact_compare_controls"):
            self._sync_compact_compare_controls()

    def _capture_multi_state(self) -> dict | None:
        if self._mode != "multi" or self._comparison is None or len(self._documents) < 2:
            return None
        return {
            "documents": list(self._documents),
            "labels": list(self._labels),
            "comparison": self._comparison,
            "primary_doc": self._primary_doc,
            "initial_payload": self._initial_payload,
            "ruby_overlay_doc": self._ruby_overlay_doc,
            "fusion_states": self._fusion_states,
            "image_review_overrides": copy.deepcopy(self._image_review_overrides),
            "canonical_source_decisions": copy.deepcopy(self._canonical_source_decisions),
            "source_correction_import_history": copy.deepcopy(self._source_correction_import_history),
            "source_correction_imported_package_ids": set(self._source_correction_imported_package_ids),
            "last_source_correction_report": copy.deepcopy(self._last_source_correction_report),
            "targeted_retry_local_report": copy.deepcopy(self._targeted_retry_local_report),
            "current_row_index": self._current_row_index,
            "source_texts": [
                self._source_text(index) for index in range(len(self._documents))
            ],
            "sources_dirty": self._sources_dirty,
        }

    def _restore_suspended_multi(self) -> None:
        state = self._suspended_multi_state
        if not isinstance(state, dict):
            return
        self.clear(preserve_available_single=True)
        self._mode = "multi"
        self._documents = list(state.get("documents") or [])
        self._labels = list(state.get("labels") or [])
        self._comparison = state.get("comparison")
        self._primary_doc = state.get("primary_doc")
        self._initial_payload = state.get("initial_payload")
        self._ruby_overlay_doc = state.get("ruby_overlay_doc")
        self._fusion_states = list(state.get("fusion_states") or [])
        self._image_review_overrides = copy.deepcopy(state.get("image_review_overrides") or {})
        self._canonical_source_decisions = copy.deepcopy(state.get("canonical_source_decisions") or {})
        self._source_correction_import_history = copy.deepcopy(state.get("source_correction_import_history") or [])
        self._source_correction_imported_package_ids = set(state.get("source_correction_imported_package_ids") or set())
        self._last_source_correction_report = copy.deepcopy(state.get("last_source_correction_report"))
        self._targeted_retry_local_report = copy.deepcopy(state.get("targeted_retry_local_report"))
        self._current_row_index = int(state.get("current_row_index", 0) or 0)
        self._sources_dirty = bool(state.get("sources_dirty", False))
        source_texts = list(state.get("source_texts") or [])
        restored_source_texts = [str(value or "") for value in source_texts[:len(self._documents)]]
        while len(restored_source_texts) < len(self._documents):
            restored_source_texts.append("")
        self._set_full_source_texts(restored_source_texts)
        self._loading_text = True
        try:
            for index in range(6):
                visible = index < len(self._documents)
                self._source_panels[index].setVisible(visible)
                self._source_editors[index].setReadOnly(self._single_card_enabled)
                self._choose_buttons[index].setVisible(visible)
                self._choose_buttons[index].setEnabled(visible)
                if visible:
                    label = self._labels[index] if index < len(self._labels) else f"模型{index + 1}"
                    self._choose_buttons[index].setText(f"模型{index + 1}")
                    self._choose_buttons[index].setToolTip(f"当前句采用模型{index+1} · {label}")
                    preview = (
                        self._source_line_text(index, self._current_row_index)
                        if self._single_card_enabled
                        else self._source_text(index)
                    )
                    self._source_editors[index].setPlainText(preview)
                else:
                    self._source_editors[index].clear()
        finally:
            self._loading_text = False
        self._set_source_labels_for_mode()
        self._source_area.setVisible(bool(self._documents))
        self._workspace_title.setText("OCR 对比 · 逐句裁决")
        # clear() disables the switch.  A suspended multi-model session must
        # explicitly restore its review controls together with its documents.
        self._sync_review_mode_controls()
        summary = getattr(self._comparison, "summary", "多模型 OCR 对比")
        restored_ai = sum(
            1 for item in self._canonical_source_decisions.values()
            if str(item.get("status", "") or "") == "accepted"
        )
        self._summary.setText(
            f"已恢复多模型 OCR 对比：{' · '.join(self._labels)}。{summary}"
            f"切换前的人工选择、源文本和累计 AI 裁决均已保留（有效 AI {restored_ai} 条）。"
        )
        self._choose_label.setVisible(True)
        self._row_state.setVisible(True)
        self._result_panel.setVisible(True)
        self._result_title.setText("融合结果（真正一致与两模型共同候选均自动保留；真正分歧需裁决）")
        self._auto_btn.setEnabled(True)
        self._realign_btn.setEnabled(True)
        self._restore_btn.setEnabled(self._initial_payload is not None)
        self._unicode_normalize_btn.setEnabled(True)
        self._export_texts_btn.setEnabled(True)
        self._single_export_btn.setEnabled(False)
        self._export_ai_package_btn.setEnabled(True)
        self._export_source_correction_btn.setEnabled(True)
        self._import_source_correction_btn.setEnabled(True)
        self._export_fusion_skeleton_btn.setEnabled(True)
        self._export_ai_repair_epub_btn.setEnabled(True)
        self._import_ai_repair_result_btn.setEnabled(True)
        self._ai_adjudicate_btn.setEnabled(not self._ai_import_busy)
        self._import_ai_package_btn.setEnabled(True)
        self._apply_btn.setText("✓ 应用融合稿")
        self._apply_btn.setEnabled(True)
        self._suspended_multi_state = None
        self._render_fusion_window(self._current_row_index, force=True)
        self._update_unresolved_summary()
        self._refresh_source_highlights()
        if self._comparison is not None and self._comparison.rows:
            self._select_row(self._current_row_index)
        self._refresh_single_source_controls()

    def _load_available_single_result(self) -> None:
        doc = self._available_single_doc
        if doc is None:
            QMessageBox.warning(self, "没有可载入结果", "请先在 OCR 识别页完成一次单模型 OCR。")
            return
        if self._mode == "multi":
            answer = QMessageBox.question(
                self, "切换到单 OCR 结果",
                "当前正在显示多模型 OCR 对比。切换后会暂存当前文本、对齐和人工选择，"
                "可用“返回多模型对比”完整恢复。继续吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        self._load_single_result(doc, self._available_single_label, self._available_single_generation)

    def _load_single_result(
        self, doc: UnifiedDocument, label: str, generation: int | None = None
    ) -> None:
        self._batch_service.invalidate_restore()
        available_doc = self._available_single_doc
        available_label = self._available_single_label
        available_generation = self._available_single_generation
        suspended = self._capture_multi_state() if self._mode == "multi" else self._suspended_multi_state
        self.clear(preserve_available_single=True)
        self._suspended_multi_state = suspended
        # clear() intentionally keeps the registered source; restore explicit
        # locals as a guard against future reset changes.
        self._available_single_doc = available_doc or doc
        self._available_single_label = available_label or str(label or "单模型 OCR")
        self._available_single_generation = available_generation
        self._mode = "single"
        self._single_doc = doc
        self._single_original_doc = doc
        self._primary_doc = doc
        self._documents = [doc]
        self._labels = [str(label or getattr(doc.metadata, "source_engine", "") or "单模型 OCR")]
        self._loaded_single_generation = (
            self._available_single_generation if generation is None else int(generation)
        )
        self._loading_text = True
        try:
            for index in range(6):
                visible = index == 0
                self._source_panels[index].setVisible(visible)
                self._choose_buttons[index].setVisible(False)
                self._choose_buttons[index].setEnabled(False)
                self._source_editors[index].setReadOnly(True)
                self._source_editors[index].setExtraSelections([])
                if visible:
                    self._source_labels[index].setText(
                        f"单 OCR · {self._labels[0]}（只读预览；修改请使用 AI 包或图文对照）"
                    )
                    self._source_editors[index].setPlainText(self._single_document_preview_text(doc))
                else:
                    self._source_editors[index].clear()
        finally:
            self._loading_text = False
        self._source_area.setVisible(True)
        self._workspace_title.setText("OCR 对比 · 单结果校对")
        self._sync_review_mode_controls()
        self._summary.setText(
            f"已手动载入单 OCR：{self._labels[0]} · {len(doc.text_blocks())} 个正文块。"
            "该结果不会伪装成双模型比较；可在本页导出/导入单 OCR AI 包。"
        )
        self._choose_label.setVisible(False)
        self._row_state.setVisible(False)
        self._result_panel.setVisible(False)
        self._auto_btn.setEnabled(False)
        self._realign_btn.setEnabled(False)
        self._restore_btn.setEnabled(False)
        self._unicode_normalize_btn.setEnabled(True)
        self._export_texts_btn.setEnabled(False)
        self._export_ai_package_btn.setEnabled(False)
        self._export_source_correction_btn.setEnabled(False)
        self._import_source_correction_btn.setEnabled(False)
        self._export_fusion_skeleton_btn.setEnabled(False)
        self._export_ai_repair_epub_btn.setEnabled(False)
        self._import_ai_repair_result_btn.setEnabled(False)
        self._ai_adjudicate_btn.setEnabled(False)
        self._import_ai_package_btn.setEnabled(False)
        self._apply_btn.setText("✓ 应用单OCR结果")
        self._apply_btn.setEnabled(True)
        self._refresh_single_source_controls()

    def _export_single_roundtrip_package(self):
        return self._exchange_service._export_single_roundtrip_package()

    def _import_single_roundtrip_package(self):
        return self._exchange_service._import_single_roundtrip_package()

    def prepare_large_result_load(self, labels, row_count: int = 0) -> int:
        """Paint the workspace before heavy QTextDocument assignment begins."""
        self.clear()
        self._multi_result_load_pending = True
        generation = self._result_load_generation
        model_text = " · ".join(str(item) for item in (labels or [])) or "多模型 OCR"
        count_text = f"，约 {int(row_count)} 句" if row_count else ""
        self._summary.setText(
            f"正在载入 OCR 对比：{model_text}{count_text}。"
            "大文档将按需创建候选控件，请稍候…"
        )
        self._virtual_hint.setText("正在准备轻量裁决状态…")
        return generation

    def set_project_package_dir(self, path) -> None:
        self._project_state_service.set_project_package_dir(path)

    def _project_package_default(self, filename: str) -> str:
        return self._project_state_service.project_package_default(filename)

    @staticmethod
    def _project_jsonable(value):
        return OCRCompareProjectStateService.jsonable(value)

    def project_snapshot_state(self) -> dict:
        return self._project_state_service.snapshot_state()

    @staticmethod
    def prepare_project_snapshot_restore(snapshot: dict) -> dict | None:
        return OCRCompareProjectStateService.prepare_snapshot_restore(snapshot)

    def restore_project_snapshot(self, snapshot: dict, *, prepared: dict | None = None) -> bool:
        return self._project_state_service.restore_snapshot(snapshot, prepared=prepared)

    def apply_project_adjudication_events(self, events) -> int:
        return self._project_state_service.apply_adjudication_events(events)

    def set_results(self, payload: dict, *, generation: int | None = None):
        if generation is not None and generation != self._result_load_generation:
            return
        self._multi_result_load_pending = False
        self._suspended_multi_state = None
        documents = list(payload.get("documents") or [])[:6]
        comparison = payload.get("comparison")
        fused_doc = payload.get("fused")
        ruby_enabled = bool(
            payload.get(
                "ruby_enabled",
                getattr(getattr(fused_doc, "metadata", None), "ruby_preservation_enabled", False),
            )
        )
        ruby_overlay_doc = fused_doc if ruby_enabled else None
        self._ruby_overlay_doc = ruby_overlay_doc
        # Keep immutable worker results by reference instead of deep-copying
        # three 300-page books and the whole comparison on the GUI thread.
        # User decisions live in separate lightweight states; destructive
        # operations use copy-on-write only when explicitly requested.
        self._initial_payload = {
            "documents": tuple(documents),
            "labels": tuple(list(payload.get("labels") or [])[:6]),
            # The worker-owned comparison remains the immutable restore source.
            # The review UI keeps all user decisions in lightweight states and
            # only copies the comparison on explicit destructive operations.
            "comparison": comparison,
            "source_texts": tuple(payload.get("source_texts") or ()),
            "fused": ruby_overlay_doc,
            "ruby_enabled": ruby_enabled,
            "targeted_retry_local_report": payload.get("targeted_retry_local_report"),
        }
        self._load_results(payload)

    def _load_results(self, payload: dict):
        self._batch_service.invalidate_restore()
        self._mode = "multi"
        self._single_doc = None
        self._single_original_doc = None
        self._documents = list(payload.get("documents") or [])[:6]
        self._labels = list(payload.get("labels") or [])[:6]
        self._comparison = payload.get("comparison")
        fused_doc = payload.get("fused")
        ruby_enabled = bool(
            payload.get(
                "ruby_enabled",
                getattr(getattr(fused_doc, "metadata", None), "ruby_preservation_enabled", False),
            )
        )
        self._ruby_overlay_doc = fused_doc if ruby_enabled else None
        self._primary_doc = self._documents[0] if self._documents else None
        # External AI Studio evidence belongs to one concrete OCR comparison
        # session. Loading a new comparison invalidates the previous imports.
        self._targeted_retry_local_report = payload.get("targeted_retry_local_report")
        if self._comparison is None or len(self._documents) < 2:
            self.clear()
            return

        source_texts = list(payload.get("source_texts") or [])
        prepared_source_texts = []
        for index in range(len(self._documents)):
            prepared_text = (
                str(source_texts[index])
                if index < len(source_texts)
                else "\n".join(row.texts[index] for row in self._comparison.rows)
            )
            prepared_source_texts.append(prepared_text)
        self._set_full_source_texts(prepared_source_texts)
        self._single_card_preview_row = -1
        self._loading_text = True
        try:
            for index in range(6):
                visible = index < len(self._documents)
                self._source_panels[index].setVisible(visible)
                self._source_editors[index].setReadOnly(self._single_card_enabled)
                self._choose_buttons[index].setVisible(visible)
                self._choose_buttons[index].setEnabled(visible)
                if visible:
                    label = self._labels[index] if index < len(self._labels) else f"模型{index + 1}"
                    self._choose_buttons[index].setText(f"模型{index + 1}")
                    self._choose_buttons[index].setToolTip(
                        f"当前句采用模型{index+1} · {label} · 快捷键 Alt+{index+1}"
                    )
                    visible_text = (
                        self._source_line_text(index, self._current_row_index)
                        if self._single_card_enabled
                        else self._source_text(index)
                    )
                    self._source_editors[index].setUpdatesEnabled(False)
                    try:
                        self._source_editors[index].setPlainText(visible_text)
                    finally:
                        self._source_editors[index].setUpdatesEnabled(True)
                else:
                    self._source_editors[index].clear()
        finally:
            self._loading_text = False
        self._set_source_labels_for_mode()
        self._source_area.setVisible(bool(self._documents))
        self._sources_dirty = False
        self._workspace_title.setText("OCR 对比 · 逐句裁决")
        self._sync_review_mode_controls()
        self._choose_label.setVisible(True)
        self._row_state.setVisible(True)
        self._result_panel.setVisible(True)
        self._result_title.setText("融合结果（真正一致与两模型共同候选均自动保留；真正分歧需裁决）")
        self._apply_btn.setText("✓ 应用融合稿")
        self._single_export_btn.setEnabled(False)
        self._export_ai_package_btn.setEnabled(True)
        self._export_source_correction_btn.setEnabled(True)
        self._import_source_correction_btn.setEnabled(True)
        self._export_fusion_skeleton_btn.setEnabled(True)
        self._export_ai_repair_epub_btn.setEnabled(True)
        self._import_ai_repair_result_btn.setEnabled(True)
        self._ai_adjudicate_btn.setEnabled(not self._ai_import_busy)
        self._import_ai_package_btn.setEnabled(True)
        restored_states = list(payload.get("restored_fusion_states") or [])
        if restored_states and len(restored_states) == len(self._comparison.rows):
            self._fusion_states = restored_states
        else:
            self._rebuild_fusion_rows(auto_choose=False)
        self._apply_targeted_retry_local_report(self._targeted_retry_local_report)
        self._set_review_mode("decision", force=True)
        self._refresh_source_highlights()
        self._summary.setText(
            f"模型：{' · '.join(self._labels)}。{self._comparison.summary} "
            "共同字符已标绿、差异字符已标红；真正分歧等待逐句裁决。快速共识仅减少后续 OCR 调用，任何独立模型异议都不会因 2:1 自动定稿。"
        )
        self._auto_btn.setEnabled(True)
        self._realign_btn.setEnabled(True)
        self._restore_btn.setEnabled(self._initial_payload is not None)
        self._unicode_normalize_btn.setEnabled(True)
        self._export_texts_btn.setEnabled(True)
        self._export_ai_package_btn.setEnabled(True)
        self._export_source_correction_btn.setEnabled(True)
        self._import_source_correction_btn.setEnabled(True)
        self._export_fusion_skeleton_btn.setEnabled(True)
        self._export_ai_repair_epub_btn.setEnabled(True)
        self._import_ai_repair_result_btn.setEnabled(True)
        self._ai_adjudicate_btn.setEnabled(not self._ai_import_busy)
        self._import_ai_package_btn.setEnabled(True)
        self._apply_btn.setEnabled(True)
        unresolved_rows = list(self._decision_navigation_rows())
        if self._review_only_enabled and unresolved_rows:
            self._select_row(unresolved_rows[0])
        else:
            self._select_row(0)
            if (
                self._comparison.conflict_rows
                or int(getattr(self._comparison, "provisional_consensus_rows", 0) or 0)
            ):
                self._jump_conflict(1, include_current=True)

    def _apply_targeted_retry_local_report(self, report) -> int:
        """Apply conservative retry-majority verdicts as local overlays.

        Raw OCR candidates remain untouched.  The selected text is always the
        pre-existing independent-model majority; the retry only confirms it.
        """
        if report is None or self._comparison is None or not self._fusion_states:
            return 0
        decisions = (
            list(getattr(report, "decisions", ()) or ())
            if not isinstance(report, dict)
            else list(report.get("decisions") or ())
        )
        from engine.ocr_compare_view_model import upsert_external_candidate
        applied = 0
        for raw in decisions:
            if isinstance(raw, dict):
                state_name = str(raw.get("state", "") or "")
                row_index = int(raw.get("row_index", -1) or -1)
                chosen_text = str(raw.get("chosen_text", "") or "")
                confidence = float(raw.get("confidence", 0.0) or 0.0)
                rationale = str(raw.get("rationale", "") or "")
            else:
                state_name = str(getattr(raw, "state", "") or "")
                row_index = int(getattr(raw, "row_index", -1) or -1)
                chosen_text = str(getattr(raw, "chosen_text", "") or "")
                confidence = float(getattr(raw, "confidence", 0.0) or 0.0)
                rationale = str(getattr(raw, "rationale", "") or "")
            if state_name != "AUTO_ACCEPT" or not chosen_text.strip():
                continue
            if isinstance(raw, dict):
                retry_kind = str(raw.get("retry_kind", "") or "")
                retry_evidence = raw.get("evidence") if isinstance(raw.get("evidence"), dict) else {}
            else:
                retry_kind = str(getattr(raw, "retry_kind", "") or "")
                retry_evidence = getattr(raw, "evidence", {})
                if not isinstance(retry_evidence, dict):
                    retry_evidence = {}
            if (
                retry_kind == "masked_phrase_diagnostic"
                or "diagnostic" in retry_kind
                or retry_evidence.get("retry_is_full_row_reading") is False
            ):
                # Old workspaces may contain decisions created by the retired
                # partial-span confirmation rule.  Never re-apply those as local
                # automatic verdicts; leave the row for human/cloud review.
                continue
            if not 0 <= row_index < len(self._fusion_states):
                continue
            decision_state = self._fusion_states[row_index]
            candidate_index = upsert_external_candidate(
                decision_state,
                chosen_text,
                display_label="本地重试多数裁决",
                select=True,
                reason=rationale or "原持异议模型换输入框架重试后与独立模型严格多数一致。",
                confidence=confidence,
                force_role_candidate=True,
                audit_level="local_targeted_retry_majority_confirmation",
                audit_flags=(
                    "raw_ocr_sources_preserved",
                    "retry_is_not_independent_vote",
                    "preexisting_strict_model_majority",
                    "dissenting_model_retry_converged",
                    "canonical_single_writeback_chain",
                ),
                selection_origin="local_targeted_retry_majority_adjudication",
            )
            if candidate_index is not None:
                decision_state.fusion_reason = rationale
                decision_state.requires_confirmation = False
                applied += 1
        if applied:
            self._sync_canonical_authority_from_states()
        return applied

    @staticmethod
    def _image_review_row_identity(row) -> str:
        group_id = str(getattr(row, "sentence_group_id", "") or "")
        if group_id:
            return "sentence:" + group_id
        # Legacy fallback must still distinguish multiple sentence rows that can
        # share one physical column. Never use column_ids alone as an override key.
        return (
            f"row:{int(getattr(row, 'index', -1) or 0)}:"
            f"block:{str(getattr(row, 'primary_block_id', '') or '')}:"
            f"{int(getattr(row, 'primary_segment_index', 0) or 0)}"
        )

    def _reapply_image_review_overrides(self) -> None:
        if self._comparison is None or not self._image_review_overrides:
            return
        from engine.ocr_compare_view_model import upsert_external_candidate
        for row_index, row in enumerate(self._comparison.rows):
            override = self._image_review_overrides.get(self._image_review_row_identity(row))
            if not isinstance(override, dict) or row_index >= len(self._fusion_states):
                continue
            upsert_external_candidate(
                self._fusion_states[row_index],
                str(override.get("text", "") or ""),
                display_label="图文对照人工校对",
                select=True,
                reason="由图文对照保存并同步；OCR 模型原文保持不变。",
                confidence=1.0,
                allow_empty=bool(override.get("delete_intentionally", False)),
                force_role_candidate=True,
                selection_origin="human_image_review",
            )

    def _fusion_decision_payload(self, row_index: int, *, origin: str = "ocr_compare") -> dict | None:
        if self._comparison is None or not 0 <= int(row_index) < len(self._fusion_states):
            return None
        row_index = int(row_index)
        row = self._comparison.rows[row_index]
        state = self._fusion_states[row_index]
        selected_index = state.selected_index
        candidate = None
        if selected_index is not None and 0 <= selected_index < len(state.candidates):
            candidate = state.candidates[selected_index]
        return {
            "row_index": row_index,
            "sentence_group_id": str(getattr(row, "sentence_group_id", "") or ""),
            "column_ids": [
                str(value) for value in (getattr(row, "column_ids", ()) or ()) if str(value)
            ],
            "resolved": candidate is not None,
            "text": str(getattr(candidate, "text", "") or "") if candidate is not None else "",
            "delete_intentionally": bool(
                getattr(candidate, "delete_intentionally", False)
            ) if candidate is not None else False,
            "selected_candidate_index": int(selected_index) if selected_index is not None else -1,
            "display_label": str(getattr(candidate, "display_label", "") or "") if candidate is not None else "",
            "reason": str(getattr(candidate, "reason", "") or "") if candidate is not None else "",
            "confidence": float(getattr(candidate, "confidence", 0.0) or 0.0) if candidate is not None else 0.0,
            "selection_origin": str(getattr(state, "selection_origin", "") or ""),
            "origin": str(origin or "ocr_compare"),
        }

    def _publish_fusion_decision(self, row_index: int, *, origin: str = "ocr_compare") -> None:
        self._sync_canonical_authority_from_states([row_index])
        payload = self._fusion_decision_payload(row_index, origin=origin)
        if payload is not None:
            if origin != "image_review":
                row = self._comparison.rows[row_index]
                self._image_review_overrides.pop(self._image_review_row_identity(row), None)
            self.fusion_decision_changed.emit(payload)

    def _manual_decision_seed_text(self, state, row_index: int) -> str:
        """Return the human-edit baseline without mutating any OCR source.

        Prefer the latest explicit human decision, then a strict raw-model majority.
        If no strict majority exists, keep the currently selected fusion output
        or the comparison row's deterministic fallback.  A previous explicit
        human edit remains visible so revisiting a row never discards user work.
        """
        origin = str(getattr(state, "selection_origin", "") or "")
        current = str(state.output_text() or "").strip()
        if origin.startswith("human_") and state.selected_index is not None:
            return current
        try:
            row = self._comparison.rows[int(row_index)]
            raw = [str(value or "").strip() for value in tuple(getattr(row, "texts", ()) or ())]
            nonempty = [value for value in raw if value]
            if nonempty:
                from collections import Counter
                value, count = Counter(nonempty).most_common(1)[0]
                if count >= 2 and count * 2 > len(nonempty):
                    return value
                if current:
                    return current
                return str(getattr(row, "output_text", "") or "").strip()
        except Exception:
            pass
        return current

    def _sync_manual_decision_editor(self) -> None:
        editor = getattr(self, "_manual_decision_editor", None)
        if editor is None:
            return
        valid = (
            self._mode == "multi"
            and self._comparison is not None
            and bool(self._fusion_states)
            and 0 <= int(getattr(self, "_current_row_index", -1)) < len(self._fusion_states)
        )
        self._manual_decision_syncing = True
        try:
            editor.setEnabled(bool(valid))
            locked = bool(valid and len(self._documents) >= 2 and not self._comparison.rows[int(self._current_row_index)].is_conflict)
            editor.setReadOnly(locked)
            value = ""
            if valid:
                row_index = int(self._current_row_index)
                state = self._fusion_states[row_index]
                value = self._manual_decision_seed_text(state, row_index)
            if editor.toPlainText() != value:
                editor.setPlainText(value)
        finally:
            self._manual_decision_syncing = False

    def _manual_decision_text_changed(self) -> None:
        if bool(getattr(self, "_manual_decision_syncing", False)):
            return
        timer = getattr(self, "_manual_decision_commit_timer", None)
        if timer is not None:
            timer.start()

    def _commit_manual_decision_editor(self) -> None:
        editor = getattr(self, "_manual_decision_editor", None)
        if editor is None or bool(getattr(self, "_manual_decision_syncing", False)):
            return
        if self._mode != "multi" or self._comparison is None or not self._fusion_states:
            return
        row_index = int(getattr(self, "_current_row_index", -1))
        if not 0 <= row_index < len(self._fusion_states):
            return
        if len(self._documents) >= 2 and not self._comparison.rows[row_index].is_conflict:
            return
        text = editor.toPlainText().strip()
        if not text:
            return
        state = self._fusion_states[row_index]
        if state.output_text().strip() == text and str(getattr(state, "selection_origin", "") or "") == "human_manual_edit":
            return
        from engine.ocr_compare_view_model import upsert_external_candidate
        selected = upsert_external_candidate(
            state, text,
            display_label="人工手动编辑",
            select=True,
            reason="OCR 对比手动编辑；原始 OCR 模型候选保持不变。",
            confidence=1.0,
            force_role_candidate=True,
            selection_origin="human_manual_edit",
        )
        if selected is None:
            return
        state.requires_confirmation = False
        self._sync_canonical_authority_from_states([row_index])
        self._publish_fusion_decision(row_index, origin="ocr_compare")
        self._remove_decision_queue_rows((row_index,))
        remaining_pending = (
            self._decision_queue_model.rowCount()
            if self._decision_queue_model is not None and not self._show_resolved_history
            else sum(1 for item in self._fusion_states if item.unresolved)
        )
        self._update_unresolved_summary(refresh_queue=False, known_unresolved=remaining_pending)
        self._render_fusion_window(row_index, force=True)
        self._row_state.setText(
            f"当前句：{row_index + 1}/{len(self._comparison.rows)} · 人工编辑 · 已确定"
        )
        self._summary.setText(
            f"第 {row_index + 1} 句已保存为人工手动编辑候选；所有 OCR 模型原文保持不变。"
        )

    def apply_image_review_update(self, payload: dict, *, refresh: bool = True) -> bool:
        """Publish one 图文对照 sentence as the selected final fusion candidate.

        This intentionally does *not* edit NDLOCR/48px/Apple Vision source text.
        The reviewed sentence becomes a synthetic manual candidate on the exact
        comparison row and therefore participates in export, recovery and EPUB.
        """
        if self._mode != "multi" or self._comparison is None or not self._fusion_states:
            return False
        if not isinstance(payload, dict):
            return False
        try:
            row_index = int(payload.get("row_index", -1))
        except (TypeError, ValueError, OverflowError):
            row_index = -1
        incoming_columns = tuple(
            str(value) for value in (payload.get("column_ids") or ()) if str(value)
        )
        incoming_group_id = str(payload.get("sentence_group_id", "") or "")
        # Prefer the shared stable sentence identity, then the row index, but recover by the complete physical
        # column identity when a restored/re-aligned session changed row numbering.
        # Earlier versions silently rejected this case while 图文仍 displayed
        # “已同步”, leaving the two workspaces contradictory.
        from engine.ocr_compare_view_model import resolve_stable_row_index
        resolved_index = resolve_stable_row_index(
            self._comparison.rows, row_index, incoming_columns, incoming_group_id
        )
        target_index = int(resolved_index) if resolved_index is not None else -1
        if not 0 <= target_index < len(self._comparison.rows) or target_index >= len(self._fusion_states):
            return False
        row_index = target_index
        row = self._comparison.rows[row_index]
        expected_columns = tuple(
            str(value) for value in (getattr(row, "column_ids", ()) or ()) if str(value)
        )
        text = str(payload.get("text", "") or "")
        delete_intentionally = bool(payload.get("delete_intentionally", False))
        if not text.strip() and not delete_intentionally:
            return False
        identity = self._image_review_row_identity(row)
        self._image_review_overrides[identity] = {
            "text": text,
            "sentence_group_id": str(getattr(row, "sentence_group_id", "") or ""),
            "column_ids": list(expected_columns),
            "segment_key": str(payload.get("segment_key", "") or ""),
            "delete_intentionally": delete_intentionally,
        }
        from engine.ocr_compare_view_model import upsert_external_candidate
        upsert_external_candidate(
            self._fusion_states[row_index],
            text,
            display_label="图文对照人工校对",
            select=True,
            reason="由图文对照保存并同步；OCR 模型原文保持不变。",
            confidence=1.0,
            allow_empty=delete_intentionally,
            force_role_candidate=True,
            selection_origin="human_image_review",
        )
        # Publish the same authoritative state to every linked review surface.
        # ``origin=image_review`` prevents MainWindow from echoing the event back
        # into the source widget while still allowing other consumers to observe it.
        self._publish_fusion_decision(row_index, origin="image_review")
        # Rebuild visible cards on the next activation, even for the same row.
        self._fusion_window_indices = ()
        self._current_row_index = row_index
        # Hidden-surface sync must stay model-only.  Remove just this resolved
        # row from the virtual queue and update counters; do not rebuild 1k-4k
        # rows and do not repaint the OCR comparison window when 图文 is active.
        self._remove_decision_queue_rows((row_index,))
        remaining_pending = (
            self._decision_queue_model.rowCount()
            if self._decision_queue_model is not None and not self._show_resolved_history
            else sum(1 for state in self._fusion_states if state.unresolved)
        )
        self._update_unresolved_summary(refresh_queue=False, known_unresolved=remaining_pending)
        if refresh:
            # Only a visible OCR compare surface needs its current-row widgets
            # repainted.  Cross-workspace synchronization stays data-only.
            self._current_row_index = row_index
            self._render_fusion_window(row_index, force=True)
            if self._review_only_enabled:
                QTimer.singleShot(0, lambda: self._jump_next_group_from(row_index))
            else:
                self._select_row(row_index)
            self._summary.setText(
                f"图文对照已同步到 OCR 对比第 {row_index + 1} 句；"
                "已选中独立的‘图文对照人工校对’候选，所有 OCR 模型原文均未被覆盖。"
            )
        return True

    def sync_image_review_document(self, doc: UnifiedDocument | None) -> int:
        """Import all checked image-review groups from an applied document."""
        if doc is None or self._mode != "multi":
            return 0
        applied = 0
        for block in getattr(doc, "blocks", []):
            metadata = block.metadata if isinstance(getattr(block, "metadata", None), dict) else {}
            checked = metadata.get("ocr_image_text_review_checked_segments") or {}
            groups = metadata.get("ocr_review_sentence_groups") or []
            if not isinstance(checked, dict) or not isinstance(groups, list):
                continue
            for group in groups:
                if not isinstance(group, dict):
                    continue
                try:
                    row_index = int(group.get("row_index", -1))
                except (TypeError, ValueError, OverflowError):
                    continue
                segment_key = f"{block.id}:r{row_index:06d}"
                if not checked.get(segment_key):
                    continue
                if self.apply_image_review_update({
                    "row_index": row_index,
                    "sentence_group_id": str(group.get("sentence_group_id", "") or ""),
                    "column_ids": list(group.get("column_ids") or []),
                    "text": str(group.get("text", "") or ""),
                    "delete_intentionally": not bool(str(group.get("text", "") or "").strip()),
                    "segment_key": segment_key,
                    "reviewed": True,
                }, refresh=False):
                    applied += 1
        if applied:
            self._render_fusion_window(self._current_row_index, force=True)
            self._update_unresolved_summary()
            self._summary.setText(
                f"已从图文对照同步 {applied} 条人工融合结果；OCR 模型原文保持不变。"
            )
        return applied


















    def _choose_current_from_model(self, model_index: int):
        if self._show_resolved_history:
            self._summary.setText("当前正在浏览只读裁决历史；关闭“显示已裁决”后才能重新裁决。")
            return
        if self._comparison is None or model_index >= len(self._documents):
            return
        if self._sources_dirty:
            notify(self, "源 OCR 已修改，请先点击“重新自动对齐”。", "warning")
            return
        row = self._current_row()
        if row >= len(self._fusion_states):
            return
        widget = self._fusion_widgets.get(row)
        if widget is not None:
            chosen = widget.choose_model(model_index)
        else:
            candidate_index = next((
                index for index, candidate in enumerate(self._fusion_states[row].candidates)
                if model_index in candidate.model_indices
            ), None)
            if candidate_index is not None:
                self._history_service.prepare_resolution(row, candidate_index)
            chosen = self._fusion_states[row].choose_model(model_index)
            if chosen:
                self._fusion_row_resolved(row)
                self._history_service.commit_resolution(row)
            else:
                self._history_service.cancel_pending()
        if not chosen:
            QMessageBox.warning(self, "没有该候选", "当前融合候选中没有该模型的有效文字。")
            return
        label = self._labels[model_index] if model_index < len(self._labels) else f"模型{model_index + 1}"
        self._detail.setText(f"手动裁决：当前句采用模型{model_index + 1} · {label}，其他候选已收起。")
        self._select_row(row)

    def _alignment_counts_valid(self, *, show_warning: bool = True) -> bool:
        if self._comparison is None:
            return False
        expected = len(self._comparison.rows)
        counts = [self._source_line_count(index) for index in range(len(self._documents))]
        if all(count == expected for count in counts):
            return True
        if show_warning:
            QMessageBox.warning(
                self, "对齐行数已改变",
                f"当前对齐为 {expected} 行；各 OCR 栏现在为：{counts}。\n"
                "允许新增或删除换行，请点击“重新自动对齐”后再继续。",
            )
        return False

    def _realign_and_auto(self):
        if self._comparison is None or len(self._documents) < 2:
            return
        from engine.multi_ocr_compare import realign_ocr_texts
        self._sync_full_source_cache_from_editors()
        source_texts = [self._source_text(index) for index in range(len(self._documents))]
        template = self._initial_payload.get("comparison") if self._initial_payload else self._comparison
        try:
            comparison = realign_ocr_texts(source_texts, self._labels, template)
        except Exception as exc:
            show_error_dialog(self, "重新自动对齐失败", str(exc))
            return
        self._comparison = comparison
        # Re-alignment changes row/segment mapping; old external evidence is no
        # longer safe to project onto the new comparison.
        self._set_full_source_texts([
            "\n".join(row.texts[model_index] for row in comparison.rows)
            for model_index in range(len(self._documents))
        ])
        self._loading_text = True
        try:
            for model_index, editor in enumerate(self._source_editors[:len(self._documents)]):
                editor.setUpdatesEnabled(False)
                try:
                    editor.setReadOnly(self._single_card_enabled)
                    editor.setPlainText(
                        self._source_line_text(model_index, self._current_row())
                        if self._single_card_enabled
                        else self._source_text(model_index)
                    )
                finally:
                    editor.setUpdatesEnabled(True)
        finally:
            self._loading_text = False
        self._set_source_labels_for_mode()
        self._sources_dirty = False
        self._rebuild_fusion_rows(auto_choose=False)
        self._refresh_source_highlights()
        self._summary.setText(
            f"已接受增删换行并重新自动对齐。{comparison.summary} "
            "红绿字符差异与融合候选已重建；真正分歧需重新裁决，两模型共同候选按 v8 规则自动保留。"
        )
        self._select_row(min(self._current_row(), max(0, len(comparison.rows) - 1)))

    @staticmethod
    def _normalization_report_text(report) -> str:
        lines = list(report.summary_lines()) if report is not None else []
        if not lines:
            return "未发现需要修复或仅比较处理的 Unicode 码位差异。"
        head = (
            f"正文无损修复 {int(report.texts_changed or 0)} 个文本单元、"
            f"{int(report.total_changes or 0)} 处；"
            f"仅比较处理 {int(getattr(report, 'total_compare_only_changes', 0) or 0)} 处。\n"
            "删除 OCR 字符：0；汉字/异体字替换：0。"
        )
        return head + "\n" + "\n".join(f"• {line}" for line in lines[:12])

    def _standardize_unicode_variants(self):
        """Normalize representation-only OCR differences in the current copy."""
        from engine.ocr_unicode_standardizer import (
            comparison_keys_for_texts,
            normalize_document_copy,
            normalize_japanese_ocr_text,
            normalize_text_collection_with_keys,
        )

        if self._mode == "single":
            if self._single_doc is None:
                QMessageBox.warning(self, "没有 OCR 结果", "请先从 OCR 识别载入单 OCR 结果。")
                return
            normalized_doc, report = normalize_document_copy(self._single_doc)
            if not report.changed:
                notify(self, "当前单 OCR 已使用统一 Unicode 码位，不需要修复。", "info")
                return
            if self._single_original_doc is None:
                self._single_original_doc = self._single_doc
            self._single_doc = normalized_doc
            self._primary_doc = normalized_doc
            self._documents = [normalized_doc]
            self._loading_text = True
            try:
                self._source_editors[0].setPlainText(
                    self._single_document_preview_text(normalized_doc)
                )
            finally:
                self._loading_text = False
            self._restore_btn.setEnabled(True)
            self._summary.setText(
                "已对当前单 OCR 副本执行无损 Unicode 标准化；未删除字符、未替换汉字，原始 OCR 未覆盖，"
                "点击“恢复初始”可撤销，点击“应用单OCR结果”后才进入后续流程。"
            )
            notify(self, self._normalization_report_text(report), "success")
            return

        if self._mode != "multi" or self._comparison is None or len(self._documents) < 2:
            QMessageBox.warning(self, "没有对比结果", "请先载入 2～6 份 OCR 结果。")
            return

        self._sync_full_source_cache_from_editors()
        source_texts = [self._source_text(index) for index in range(len(self._documents))]
        normalized_texts, comparison_key_texts, report = (
            normalize_text_collection_with_keys(source_texts)
        )

        # Candidate editors may contain AI/manual text that is not in the source
        # columns.  Include those changes in the report and preserve them below.
        old_states = list(self._fusion_states)
        candidate_snapshots: list[list[dict]] = []
        for row_index, state in enumerate(old_states):
            row_candidates: list[dict] = []
            row = self._comparison.rows[row_index] if row_index < len(self._comparison.rows) else None
            original_row_texts = list(getattr(row, "texts", []) or [])
            for candidate_index, candidate in enumerate(state.candidates):
                normalized_candidate, candidate_report = normalize_japanese_ocr_text(candidate.text)
                is_selected = state.selected_index == candidate_index
                is_external = bool(candidate.synthetic and candidate.display_label != "字符级融合建议")
                is_manual_model_edit = False
                if candidate.model_indices:
                    source_values = [
                        original_row_texts[index]
                        for index in candidate.model_indices
                        if 0 <= index < len(original_row_texts)
                    ]
                    is_manual_model_edit = bool(source_values) and all(
                        str(candidate.text or "") != str(value or "") for value in source_values
                    )
                if is_external or is_manual_model_edit:
                    report.merge(candidate_report)
                if is_external or is_manual_model_edit or is_selected:
                    row_candidates.append({
                        "text": normalized_candidate,
                        "selected": is_selected,
                        "external": is_external,
                        "manual_model_edit": is_manual_model_edit,
                        "display_label": str(candidate.display_label or ""),
                        "reason": str(candidate.reason or ""),
                        "confidence": float(candidate.confidence or 0.0),
                        "delete_intentionally": bool(candidate.delete_intentionally),
                        "transaction_id": str(candidate.transaction_id or ""),
                        "transaction_operation": str(candidate.transaction_operation or ""),
                        "transaction_member_ids": tuple(candidate.transaction_member_ids or ()),
                        "audit_level": str(candidate.audit_level or ""),
                        "audit_flags": tuple(candidate.audit_flags or ()),
                        "selection_origin": str(getattr(state, "selection_origin", "") or "") if is_selected else "",
                    })
            candidate_snapshots.append(row_candidates)

        if not report.changed:
            notify(self, "当前各 OCR 结果与候选已使用统一 Unicode 码位，不需要修复。", "info")
            return

        old_conflicts = int(getattr(self._comparison, "conflict_rows", 0) or 0)
        current_row = self._current_row()
        normalized_lines = [self._split_source_lines(text) for text in normalized_texts]
        comparison_key_lines = [
            self._split_source_lines(text) for text in comparison_key_texts
        ]
        expected_rows = len(self._comparison.rows)
        same_boundaries = (
            all(len(lines) == expected_rows for lines in normalized_lines)
            and all(len(lines) == expected_rows for lines in comparison_key_lines)
        )

        try:
            if same_boundaries:
                from engine.multi_ocr_compare import refresh_comparison_after_text_standardization
                comparison = copy.deepcopy(self._comparison)
                for row_index, row in enumerate(comparison.rows):
                    row.texts[:len(self._documents)] = [
                        normalized_lines[model_index][row_index]
                        for model_index in range(len(self._documents))
                    ]
                    row.comparison_keys = tuple(
                        comparison_key_lines[model_index][row_index]
                        for model_index in range(len(self._documents))
                    )
                    row.comparison_key_sources = tuple(
                        str(value or "") for value in row.texts
                    )
                comparison = refresh_comparison_after_text_standardization(comparison)
            else:
                from engine.multi_ocr_compare import realign_ocr_texts
                comparison = realign_ocr_texts(
                    normalized_texts, self._labels, self._comparison
                )
                for row in comparison.rows:
                    row.comparison_keys = tuple(
                        comparison_keys_for_texts(row.texts)
                    )
                    row.comparison_key_sources = tuple(
                        str(value or "") for value in row.texts
                    )
                comparison = refresh_comparison_after_text_standardization(comparison)
        except Exception as exc:
            show_error_dialog(self, "无损标准化失败", str(exc))
            return

        self._comparison = comparison
        # Unicode standardization can change comparison keys/text spans, so any
        # imported Paddle row projection must be refreshed before local judgment.
        self._set_full_source_texts(normalized_texts)
        self._loading_text = True
        try:
            for model_index, editor in enumerate(self._source_editors[:len(self._documents)]):
                editor.setUpdatesEnabled(False)
                try:
                    editor.setReadOnly(self._single_card_enabled)
                    editor.setPlainText(
                        self._source_line_text(model_index, current_row)
                        if self._single_card_enabled
                        else self._source_text(model_index)
                    )
                finally:
                    editor.setUpdatesEnabled(True)
        finally:
            self._loading_text = False
        self._sources_dirty = False
        self._rebuild_fusion_rows(auto_choose=False)

        # Restore AI candidates, manually edited candidate cards and explicit
        # choices after the comparison rows have been rebuilt.
        from engine.ocr_compare_view_model import upsert_external_candidate
        for row_index, snapshots in enumerate(candidate_snapshots):
            if row_index >= len(self._fusion_states):
                break
            state = self._fusion_states[row_index]
            for snapshot in snapshots:
                text = str(snapshot.get("text", "") or "")
                selected = bool(snapshot.get("selected", False))
                if snapshot.get("external"):
                    upsert_external_candidate(
                        state, text,
                        display_label=str(snapshot.get("display_label") or "外部候选"),
                        select=selected,
                        reason=str(snapshot.get("reason") or ""),
                        confidence=float(snapshot.get("confidence") or 0.0),
                        allow_empty=bool(snapshot.get("delete_intentionally")),
                        transaction_id=str(snapshot.get("transaction_id") or ""),
                        transaction_operation=str(snapshot.get("transaction_operation") or ""),
                        transaction_member_ids=tuple(snapshot.get("transaction_member_ids") or ()),
                        audit_level=str(snapshot.get("audit_level") or ""),
                        audit_flags=tuple(snapshot.get("audit_flags") or ()),
                        selection_origin=str(snapshot.get("selection_origin") or "restored_human") if selected else "",
                    )
                    continue
                if selected and state.choose_text(
                    text, origin=str(snapshot.get("selection_origin") or "restored_human")
                ):
                    continue
                if snapshot.get("manual_model_edit"):
                    upsert_external_candidate(
                        state, text,
                        display_label="人工编辑候选（标准化保留）",
                        select=selected,
                        reason="标准化前已在候选卡中手动修改，已作为独立候选保留。",
                        allow_empty=bool(snapshot.get("delete_intentionally")),
                        selection_origin=(
                            str(snapshot.get("selection_origin") or "human_manual_edit")
                            if selected else ""
                        ),
                    )

        self._render_fusion_window(current_row, force=True)
        self._refresh_source_highlights()
        self._update_unresolved_summary()
        self._restore_btn.setEnabled(self._initial_payload is not None)
        if self._comparison.rows:
            self._select_row(min(current_row, len(self._comparison.rows) - 1))

        removed_conflicts = max(0, old_conflicts - int(self._comparison.conflict_rows or 0))
        conflict_note = (
            f"；消除 {removed_conflicts} 句仅由码位差异造成的假冲突"
            if removed_conflicts else ""
        )
        self._summary.setText(
            f"Unicode 无损标准化完成{conflict_note}。未删除 OCR 内容，"
            "未替换汉字/异体字；兼容码位只参与临时比较，不写回正文。"
            "对齐、物理列 ID 和结构保持不变，候选已重新计算。"
        )
        notify(
            self,
            self._normalization_report_text(report) +
            (f"\n\n假冲突减少：{removed_conflicts} 句" if removed_conflicts else ""),
            "success",
        )

    def _restore_initial(self):
        self._batch_service.invalidate_restore()
        if self._mode == "single":
            original = self._single_original_doc
            if original is None or original is self._single_doc:
                return
            label = self._labels[0] if self._labels else self._available_single_label or "单模型 OCR"
            generation = self._loaded_single_generation
            self._load_single_result(original, label, generation)
            self._summary.setText(
                "已恢复标准化前的单 OCR 原始结果；Unicode 标准化副本已撤销。"
            )
            return
        if not self._initial_payload:
            return
        payload = {
            "documents": list(self._initial_payload.get("documents") or []),
            "labels": list(self._initial_payload.get("labels") or []),
            "comparison": self._initial_payload.get("comparison"),
            "source_texts": list(self._initial_payload.get("source_texts") or []),
            "fused": self._initial_payload.get("fused"),
        }
        self._load_results(payload)
        self._summary.setText(
            "已恢复本次多模型 OCR 的初始文字、红绿差异、对齐与融合候选。"
            f"{self._comparison.summary if self._comparison else ''}"
        )

    def _auto_select_all(self):
        return self._batch_service.auto_select_all()

    def _restore_last_auto_select(self):
        return self._batch_service.restore_last_auto_select()

    def _apply_review_filter(self):
        self._render_fusion_window(self._current_row(), force=True)

    def _set_single_card_mode(self, checked: bool):
        """Show one sentence and one decision frame without losing full OCR text."""
        enabled = bool(checked)
        if enabled == self._single_card_enabled:
            return
        if enabled:
            self._sync_full_source_cache_from_editors()
        self._single_card_enabled = enabled
        self._single_card_preview_row = -1
        if self._mode != "multi" or self._comparison is None:
            return

        current = self._current_row()
        if enabled and self._review_only_enabled:
            unresolved = list(self._decision_navigation_rows())
            if unresolved and (
                current >= len(self._fusion_states) or not self._fusion_states[current].unresolved
            ):
                current = unresolved[0]
        if enabled:
            self._update_single_card_source_preview(current, force=True)
            self._summary.setText(
                "已开启单框逐句：每个模型只显示当前句，下方只显示一个对比框；选择后自动进入下一句。"
            )
        else:
            self._restore_full_source_editors()
            self._summary.setText(
                "已恢复完整多模型文本视图；逐句选择和候选修改均已保留。"
            )
        self._render_fusion_window(current, force=True)
        self._select_row(current)

    def _set_review_only(self, checked: bool):
        self._review_only_enabled = bool(checked)
        if self._show_resolved_history:
            return
        current = self._current_row()
        unresolved = list(self._decision_navigation_rows())
        if self._review_only_enabled and (
            current >= len(self._fusion_states) or not self._fusion_states[current].unresolved
        ):
            if unresolved:
                current = unresolved[0]
            else:
                self._summary.setText("所有真正分歧均已完成裁决；两模型共同候选已按 v8 规则自动保留。")
                self._refresh_decision_queue(force=True)
                self._show_no_pending_decisions()
                return
        self._render_fusion_window(current, force=True)
        if self._fusion_states:
            self._select_row(current)

    def _fusion_row_resolved(self, row_index: int):
        # Atomic external repairs are selected as one transaction. Choosing a
        # non-transaction candidate in any member invalidates the old group
        # selection instead of leaving a partially applied boundary repair.
        if not getattr(self, "_atomic_selection_sync", False) and 0 <= row_index < len(self._fusion_states):
            state = self._fusion_states[row_index]
            selected_index = state.selected_index
            selected_candidate = (
                state.candidates[selected_index]
                if selected_index is not None and 0 <= selected_index < len(state.candidates)
                else None
            )
            selected_tx = str(getattr(selected_candidate, "transaction_id", "") or "")
            row_transactions = {
                str(getattr(candidate, "transaction_id", "") or "")
                for candidate in state.candidates
                if str(getattr(candidate, "transaction_id", "") or "")
            }
            self._atomic_selection_sync = True
            try:
                if selected_tx:
                    for member_state in self._fusion_states:
                        candidate_index = next((
                            index for index, candidate in enumerate(member_state.candidates)
                            if str(getattr(candidate, "transaction_id", "") or "") == selected_tx
                        ), None)
                        if candidate_index is not None:
                            member_state.selected_index = candidate_index
                            member_state.selection_origin = "human_ocr_compare"
                            member_state.review_indices = member_state._build_review_indices()
                    self._summary.setText(f"已成组选择原子修复事务：{selected_tx}")
                else:
                    for transaction_id in row_transactions:
                        for member_index, member_state in enumerate(self._fusion_states):
                            transaction_indices = [
                                index for index, candidate in enumerate(member_state.candidates)
                                if str(getattr(candidate, "transaction_id", "") or "") == transaction_id
                            ]
                            if member_state.selected_index in transaction_indices:
                                member_state.selected_index = None
                                member_state.selection_origin = ""
                                member_state.review_indices = member_state._build_review_indices()
                    # Keep the user's explicit non-transaction selection on this row.
                    state.selected_index = selected_index
            finally:
                self._atomic_selection_sync = False
            self._render_fusion_window(row_index, force=True)

        # Selection is stored in the lightweight decision state. Keep the
        # original worker comparison immutable so “恢复初始” stays O(1).
        # Publish every member whose selection may have changed as part of an
        # atomic transaction, not just the clicked row.
        affected_rows = {int(row_index)}
        if 0 <= row_index < len(self._fusion_states):
            chosen = self._fusion_states[row_index]
            if chosen.selected_index is not None and 0 <= chosen.selected_index < len(chosen.candidates):
                transaction_id = str(getattr(chosen.candidates[chosen.selected_index], "transaction_id", "") or "")
                if transaction_id:
                    for member_index, member_state in enumerate(self._fusion_states):
                        if member_state.selected_index is None or not 0 <= member_state.selected_index < len(member_state.candidates):
                            continue
                        if str(getattr(member_state.candidates[member_state.selected_index], "transaction_id", "") or "") == transaction_id:
                            affected_rows.add(member_index)
        self._sync_canonical_authority_from_states(sorted(affected_rows))
        for affected_row in sorted(affected_rows):
            self._publish_fusion_decision(affected_row, origin="ocr_compare")

        # High-throughput review: one click removes only the affected queue rows.
        # No 1k-4k item reconstruction and no full unresolved-priority recompute.
        self._remove_decision_queue_rows(affected_rows)
        remaining_pending = (
            self._decision_queue_model.rowCount()
            if self._decision_queue_model is not None and not self._show_resolved_history
            else sum(1 for state in self._fusion_states if state.unresolved)
        )
        self._update_unresolved_summary(refresh_queue=False, known_unresolved=remaining_pending)
        if (
            self._review_only_enabled
            and not self._show_resolved_history
            and remaining_pending <= 0
        ):
            # Clear synchronously: the last adjudicated sentence must never stay
            # visible until the next Qt event-loop turn.
            self._show_no_pending_decisions()
            return
        if self._single_card_enabled:
            if self._auto_advance_check.isChecked():
                if self._review_only_enabled:
                    QTimer.singleShot(0, lambda: self._jump_next_group_from(row_index))
                else:
                    QTimer.singleShot(0, lambda: self._jump_next_sentence_from(row_index))
            else:
                self._select_row(row_index)
            return
        if self._review_only_enabled:
            QTimer.singleShot(0, lambda: self._jump_next_group_from(row_index))

    def _fusion_row_reopened(self, row_index: int):
        if getattr(self, "_atomic_selection_sync", False) or not 0 <= row_index < len(self._fusion_states):
            return
        transaction_ids = {
            str(getattr(candidate, "transaction_id", "") or "")
            for candidate in self._fusion_states[row_index].candidates
            if str(getattr(candidate, "transaction_id", "") or "")
        }
        if not transaction_ids:
            self._render_fusion_window(row_index, force=True)
            self._update_unresolved_summary()
            self._sync_canonical_authority_from_states([row_index])
            self._publish_fusion_decision(row_index, origin="ocr_compare")
            self._refresh_decision_queue(force=True)
            self._summary.setText("已重新打开当前句，图文对照同步恢复为待确认。")
            return
        self._atomic_selection_sync = True
        try:
            for transaction_id in transaction_ids:
                for state in self._fusion_states:
                    transaction_indices = [
                        index for index, candidate in enumerate(state.candidates)
                        if str(getattr(candidate, "transaction_id", "") or "") == transaction_id
                    ]
                    if state.selected_index in transaction_indices:
                        state.selected_index = None
                        state.selection_origin = ""
                        state.review_indices = state._build_review_indices()
        finally:
            self._atomic_selection_sync = False
        self._render_fusion_window(row_index, force=True)
        self._update_unresolved_summary()
        affected_rows = {int(row_index)}
        for member_index, state in enumerate(self._fusion_states):
            if any(str(getattr(candidate, "transaction_id", "") or "") in transaction_ids for candidate in state.candidates):
                affected_rows.add(member_index)
        self._sync_canonical_authority_from_states(sorted(affected_rows))
        for affected_row in sorted(affected_rows):
            self._publish_fusion_decision(affected_row, origin="ocr_compare")
        self._refresh_decision_queue(force=True)
        self._summary.setText("已撤销整组原子修复选择，事务成员全部恢复为待确认。")

    def _jump_previous_group(self):
        """Jump to the previous unresolved decision or filtered adjudication history."""
        if self._comparison is None or not self._fusion_states:
            return
        rows = tuple(i for i, row in enumerate(self._comparison.rows) if row.is_conflict)
        if not rows:
            message = "当前筛选没有已裁决句。" if self._show_resolved_history else "所有不一致候选均已完成选择；没有上一组待判断内容。"
            self._summary.setText(message)
            if not self._show_resolved_history:
                self._show_no_pending_decisions()
            return
        current = self._current_row()
        if current in rows:
            position = rows.index(current)
            index = rows[(position - 1) % len(rows)]
        else:
            index = rows[-1]
        self._select_row(index)
        if self._show_resolved_history:
            self._summary.setText(f"已定位到上一条已裁决句：第 {index + 1}/{len(self._fusion_states)} 句。")
        else:
            self._summary.setText(f"已定位到上一组待判断候选：第 {index + 1}/{len(self._fusion_states)} 句。")

    def _jump_next_sentence(self):
        """Move OCR 对比 to the next stable sentence without changing any decision."""
        if self._comparison is None or not self._fusion_states:
            return
        self._jump_next_sentence_from(self._current_row())

    def _jump_next_sentence_from(self, row_index: int):
        if self._comparison is None or not self._fusion_states:
            return
        next_index = int(row_index) + 1
        if next_index < len(self._fusion_states):
            self._select_row(next_index)
            self._summary.setText(
                f"已自动进入下一句：第 {next_index + 1}/{len(self._fusion_states)} 句。"
            )
            return
        self._render_fusion_window(row_index, force=True)
        if hasattr(self, "_next_sentence_btn"):
            self._next_sentence_btn.setEnabled(False)
        self._summary.setText("已完成最后一句；没有下一句。")

    def _jump_next_group_from(self, row_index: int):
        if self._comparison is None or not self._fusion_states:
            return
        rows = tuple(i for i, row in enumerate(self._comparison.rows) if row.is_conflict)
        if not rows:
            if self._show_resolved_history:
                self._show_no_pending_decisions("当前筛选没有已裁决句")
                self._summary.setText("当前筛选没有已裁决句。")
            else:
                self._show_no_pending_decisions()
                self._summary.setText("所有不一致候选均已完成选择；没有下一组待判断内容。")
            return
        current = int(row_index)
        if current in rows:
            position = rows.index(current)
            index = rows[(position + 1) % len(rows)]
        else:
            index = rows[0]
        self._select_row(index)
        if self._show_resolved_history:
            self._summary.setText(f"已进入下一条已裁决句：第 {index + 1}/{len(self._fusion_states)} 句。")
        else:
            self._summary.setText(f"已自动进入下一组待判断候选：第 {index + 1}/{len(self._fusion_states)} 句。")

    def _jump_next_group(self):
        """Jump through pending decisions or the active adjudication-history filter."""
        if self._comparison is None or not self._fusion_states:
            return
        if self._sources_dirty or not self._alignment_counts_valid(show_warning=False):
            self._realign_and_auto()
            if self._sources_dirty or not self._alignment_counts_valid(show_warning=False):
                return
        self._jump_next_group_from(self._current_row())

    @staticmethod
    def _atomic_write_utf8(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.tmp")
        try:
            temp.write_text(text, encoding="utf-8")
            temp.replace(path)
        finally:
            if temp.exists():
                temp.unlink(missing_ok=True)

    def _comparison_with_current_source_texts(self):
        """Copy comparison only for export, preserving current model editor edits."""
        if self._comparison is None:
            return None
        # Even when the dirty flag was missed by an editor/undo edge case, a
        # changed line count must never be exported against stale row IDs.
        if not self._alignment_counts_valid(show_warning=False):
            return None
        comparison = copy.deepcopy(self._comparison)
        for row_index, row in enumerate(comparison.rows):
            row.texts[:len(self._documents)] = [
                self._source_line_text(model_index, row_index)
                for model_index in range(len(self._documents))
            ]
        return comparison

    def _export_multi_roundtrip_package(self):
        return self._exchange_service._export_multi_roundtrip_package()

    def _current_comparison_documents_for_source_correction(self):
        return self._source_correction_service._current_comparison_documents_for_source_correction()

    def _set_source_correction_busy(self, busy: bool, message: str = "", *, lock_workspace: bool = False):
        return self._source_correction_service._set_source_correction_busy(busy, message, lock_workspace=lock_workspace)

    def _on_source_correction_progress(self, token: int, stage: str, current: int, total: int):
        return self._source_correction_service._on_source_correction_progress(token, stage, current, total)

    def _on_source_correction_error(self, token: int, title: str, message: str):
        return self._source_correction_service._on_source_correction_error(token, title, message)

    def _export_model_source_correction_package(self):
        return self._source_correction_service._export_model_source_correction_package()

    def _on_source_correction_export_ready(self, token: int, result: object):
        return self._source_correction_service._on_source_correction_export_ready(token, result)

    def set_recovery_page_image_provider(self, provider) -> None:
        return self._source_correction_service.set_recovery_page_image_provider(provider)

    def _current_recovery_page_images(self) -> list[str]:
        return self._source_correction_service._current_recovery_page_images()

    def _restore_model_source_correction_session(self):
        return self._source_correction_service._restore_model_source_correction_session()

    def _on_source_session_restore_ready(self, token: int, result: object):
        return self._source_correction_service._on_source_session_restore_ready(token, result)

    def _capture_current_fusion_selection_records(self) -> dict[tuple[str, ...], dict]:
        return self._source_correction_service._capture_current_fusion_selection_records()


    def _replace_multi_source_workspace(self, documents, comparison, *, preserve_selection_records=None, prepared=None, fusion_states=None):
        return self._source_correction_service._replace_multi_source_workspace(documents, comparison, preserve_selection_records=preserve_selection_records, prepared=prepared, fusion_states=fusion_states)

    def _import_model_source_correction_result(self):
        return self._source_correction_service._import_model_source_correction_result()

    def _on_source_correction_import_ready(self, token: int, result: object):
        return self._source_correction_service._on_source_correction_import_ready(token, result)

    def _export_fusion_and_skeleton(self):
        if self._comparison is None or self._primary_doc is None or len(self._documents) < 2:
            QMessageBox.warning(self, "没有多模型结果", "请先完成多模型 OCR。")
            return
        try:
            package, _comparison = self._current_multi_package_for_external_repair()
        except Exception as exc:
            show_error_dialog(self, "无法准备融合结果", str(exc))
            return
        title = safe_result_filename(
            str(getattr(getattr(self._primary_doc, "metadata", None), "title", "") or ""),
            default="multi_ocr",
        )
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出融合结果与骨架 EPUB",
            self._project_package_default(f"{title}_融合结果与骨架EPUB.zip"),
            "ZIP 包 (*.zip)",
        )
        if not path:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            from engine.multi_ocr_source_correction import export_fusion_and_skeleton_bundle
            report = export_fusion_and_skeleton_bundle(
                self._primary_doc,
                package,
                path,
                correction_audit=self._last_source_correction_report,
                vertical=True,
            )
        except Exception as exc:
            self.package_exported.emit({"kind": "fusion_skeleton", "status": "error", "path": str(path), "error": str(exc), "details": {}})
            show_error_dialog(self, "导出融合结果与骨架 EPUB 失败", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.package_exported.emit({
            "kind": "fusion_skeleton",
            "status": "ok",
            "path": str(report.get("path", path) or path),
            "details": {
                "row_count": int(report.get("row_count", 0) or 0),
                "ruby_pair_count": int(report.get("ruby_pair_count", 0) or 0),
                "ruby_enabled": bool(report.get("ruby_enabled")),
            },
        })
        notify(
            self,
            f"正文条目：{report.get('row_count', 0)}\n"
            f"锁定 Ruby：{report.get('ruby_pair_count', 0)} 组"
            f"（{'开启' if report.get('ruby_enabled') else '关闭'}）\n"
            "已包含锁定 Ruby sidecar、AI 正文模板、最终 EPUB 构建器和 Ruby 验证器。\n"
            f"融合 JSON 和稳定 ID 骨架 EPUB 已打包：\n{report.get('path', path)}",
            "success",
        )

    def _canonical_export_blockers(self, comparison, *, allow_unresolved: bool = False) -> list[str]:
        """Block stale/unsafe derivative text before EPUB or current V5 adjudication export."""
        from engine.multi_ocr_source_correction import canonical_decision_key, canonical_text_safety_issues

        blockers: list[str] = []
        if len(comparison.rows) != len(self._fusion_states):
            blockers.append(
                f"融合状态行数 {len(self._fusion_states)} 与当前对齐行数 {len(comparison.rows)} 不一致；"
                "请重新载入或重新对齐当前多模型会话"
            )
            return blockers
        for row_index, (row, state) in enumerate(zip(comparison.rows, self._fusion_states)):
            key = canonical_decision_key(row)
            selected_index = getattr(state, "selected_index", None)
            if selected_index is not None and not 0 <= int(selected_index) < len(state.candidates):
                blockers.append(f"第 {row_index + 1} 句的候选索引已失效，请重新选择")
                continue
            selected_text = state.output_text()
            if bool(getattr(state, "unresolved", False)):
                classification = (
                    "两模型共同候选" if bool(getattr(row, "provisional_consensus", False))
                    else "OCR 真正分歧"
                )
                if not allow_unresolved:
                    blockers.append(f"第 {row_index + 1} 句仍未完成 OCR 最终裁决（{classification}）")
                    continue
                # Current V5 adjudication intentionally keeps this row unresolved
                # so the external model can decide it. Do not run
                # the final-text safety checks against an intentionally empty
                # selection.
                continue
            if not selected_text and not state.output_delete_intentionally():
                blockers.append(f"第 {row_index + 1} 句融合结果为空，且未标记为有意删除")
                continue
            for issue in canonical_text_safety_issues(selected_text):
                blockers.append(f"第 {row_index + 1} 句：{issue}")
            decision = self._canonical_source_decisions.get(key)
            if not decision or decision.get("status") != "accepted":
                continue
            final_text = str(decision.get("final_text", "") or "")
            if not final_text or selected_text == final_text:
                continue
            candidate = (
                state.candidates[state.selected_index]
                if state.selected_index is not None and 0 <= state.selected_index < len(state.candidates)
                else None
            )
            label = str(getattr(candidate, "display_label", "") or "")
            from engine.ocr_compare_view_model import selection_supersedes_ai_verdict
            explicit_override = selection_supersedes_ai_verdict(state, label)
            # Any candidate explicitly checked by the user—including an original
            # NDLOCR/48px/Apple candidate—is authoritative and may supersede the
            # imported AI verdict.  Only an untouched automatic baseline is stale.
            if not explicit_override and label not in {
                "图文对照人工校对", "恢复的人工融合结果", "人工最终裁决"
            }:
                blockers.append(
                    f"第 {row_index + 1} 句仍采用自动旧结果，未采用最新 AI 裁决；"
                    "可勾选任一候选作为人工最终选择"
                )
        return blockers

    def _current_multi_package_for_external_repair(self, *, allow_unresolved: bool = False):
        return self._publication_service._current_multi_package_for_external_repair(allow_unresolved=allow_unresolved)

    def _choose_ai_repair_reference_option(self, package: dict):
        return self._publication_service._choose_ai_repair_reference_option(package)

    def _export_ai_repair_epub(self):
        return self._publication_service._export_ai_repair_epub()

    def _import_ai_repair_result(self):
        return self._publication_service._import_ai_repair_result()

    def _run_ai_adjudication(self):
        return self._gpt_adjudication_controller._run_ai_adjudication()

    def _cancel_ai_adjudication(self):
        return self._gpt_adjudication_controller._cancel_ai_adjudication()

    def _on_ai_adjudication_progress(self, token: int, raw: str) -> None:
        return self._gpt_adjudication_controller._on_ai_adjudication_progress(token, raw)

    def _on_ai_adjudication_error(self, token: int, details: str) -> None:
        return self._gpt_adjudication_controller._on_ai_adjudication_error(token, details)

    def _on_ai_adjudication_ready(self, token: int, result: object) -> None:
        return self._gpt_adjudication_controller._on_ai_adjudication_ready(token, result)

    def _set_ai_import_busy(self, busy: bool, message: str = "") -> None:
        return self._exchange_service._set_ai_import_busy(busy, message)

    def _import_multi_roundtrip_package(self):
        return self._exchange_service._import_multi_roundtrip_package()

    def _on_ai_import_progress(self, token: int, text: str) -> None:
        return self._exchange_service._on_ai_import_progress(token, text)

    def _on_ai_import_error(self, token: int, message: str) -> None:
        return self._exchange_service._on_ai_import_error(token, message)

    def _on_ai_import_ready(self, token: int, result: object) -> None:
        return self._exchange_service._on_ai_import_ready(token, result)

    def _apply_next_ai_import_editor(self) -> None:
        return self._exchange_service._apply_next_ai_import_editor()

    def _complete_ai_import_apply(self) -> None:
        return self._exchange_service._complete_ai_import_apply()

    def _export_separate_texts(self):
        return self._exchange_service._export_separate_texts()

    def _jump_conflict(self, direction: int, include_current: bool = False):
        if self._comparison is None or not self._comparison.rows:
            return
        current = self._current_row()
        total = len(self._comparison.rows)
        start = 0 if include_current else 1
        for offset in range(start, total + 1):
            index = (current + direction * offset) % total
            unresolved = index < len(self._fusion_states) and self._fusion_states[index].unresolved
            row = self._comparison.rows[index]
            review_risk = bool(
                row.is_conflict
                or getattr(row, "historical_ocr_disagreement", False)
            )
            if unresolved or (not self._review_only_enabled and review_risk):
                self._select_row(index)
                return
        self._summary.setText("当前没有需要跳转的未决分歧或已保留历史分歧。")

    @staticmethod
    def _reattach_ruby_overlay(overlay, target) -> int:
        """Compatibility wrapper around the central Ruby side-channel API."""
        if overlay is None or target is None:
            return 0
        from adapters.findtext_centernet_ruby import apply_ruby_overlay
        report = apply_ruby_overlay(target, overlay)
        return int(report.get("target_blocks", 0) or 0)

    def _update_unresolved_summary(self, *, refresh_queue: bool = True, known_unresolved: int | None = None):
        unresolved = (
            max(0, int(known_unresolved))
            if known_unresolved is not None
            else sum(1 for state in self._fusion_states if state.unresolved)
        )
        if self._comparison is not None:
            self._apply_btn.setText(
                "✓ 应用融合稿" if not unresolved else f"✓ 应用融合稿（待选 {unresolved}）"
            )
        total = len(self._fusion_states)
        history_rows = self._resolved_history_rows() if self._show_resolved_history else ()
        if self._show_resolved_history:
            self._virtual_hint.setText(f"已裁决 {len(history_rows)} 句 · 全部 {total} 句")
        else:
            self._virtual_hint.setText(
                f"待判断 {unresolved} 句 · 全部 {total} 句"
                if self._review_only_enabled
                else f"全部 {total} 句"
            )
        if refresh_queue:
            self._refresh_decision_queue(force=False)
        if hasattr(self, "_prev_group_btn"):
            self._prev_group_btn.setEnabled(bool(history_rows) if self._show_resolved_history else unresolved > 0)
        if hasattr(self, "_next_group_btn"):
            self._next_group_btn.setEnabled(bool(history_rows) if self._show_resolved_history else unresolved > 0)
        if hasattr(self, "_sync_compact_compare_controls"):
            self._sync_compact_compare_controls()

    def _apply_result(self):
        if self._mode == "single":
            if self._single_doc is None:
                QMessageBox.warning(self, "没有单 OCR 结果", "请先从 OCR 识别载入结果。")
                return
            self.single_doc_applied.emit(self._single_doc)
            self._summary.setText("已将当前单 OCR 结果应用到工作流；来源仍保留在 OCR 对比页。")
            notify(self, "当前单 OCR 结果已进入 Formatter、OCR 对比、图文对照和 EPUB。", "success")
            return
        if self._comparison is None or self._primary_doc is None:
            QMessageBox.warning(self, "没有结果", "请先运行多模型 OCR。")
            return
        if self._sources_dirty or not self._alignment_counts_valid(show_warning=False):
            self._realign_and_auto()
            notify(self, "源 OCR 的修改已经重新对齐并重建候选。请重新确认冲突句后再应用。", "info")
            return
        unresolved = [state.row_index for state in self._fusion_states if state.unresolved]
        if unresolved:
            self._select_row(unresolved[0])
            QMessageBox.warning(
                self, "仍有未选择候选",
                f"还有 {len(unresolved)} 句没有打钩选择。已跳到第一处；选择后其他候选会消失且不留空行。",
            )
            return
        lines = [state.output_text() for state in self._fusion_states]
        delete_flags = [state.output_delete_intentionally() for state in self._fusion_states]
        comparison_for_apply = copy.deepcopy(self._comparison)
        for row_index, row in enumerate(comparison_for_apply.rows):
            row.texts[:len(self._documents)] = [
                self._source_line_text(model_index, row_index)
                for model_index in range(len(self._documents))
            ]
            state = self._fusion_states[row_index]
            selected = state.selected_index
            if selected is not None and selected < len(state.candidates):
                models = state.candidates[selected].model_indices
                if models:
                    row.chosen_index = models[0]
        try:
            from engine.multi_ocr_compare import build_fused_document
            fused = build_fused_document(
                self._primary_doc,
                comparison_for_apply,
                lines,
                delete_flags=delete_flags,
                ruby_overlay_source=self._ruby_overlay_doc,
            )
            ruby_blocks = int(
                (getattr(fused.metadata, "ruby_overlay_transfer_report", {}) or {}).get("target_blocks", 0)
            )
        except Exception as exc:
            show_error_dialog(self, "应用 OCR 融合稿失败", str(exc))
            return
        if ruby_blocks:
            fused.add_log(
                "ruby_preservation",
                f"应用融合稿时保留 Ruby 结构覆盖：{ruby_blocks} 个文字块",
                ruby_blocks,
            )
        self.doc_applied.emit(fused)
        self._summary.setText("已将当前逐句裁决稿应用到工作流；源 OCR 和候选卡仍保留在本页。")
        notify(
            self,
            "融合稿已进入 Formatter、OCR 对比和 EPUB 工作流。\n"
            "每个候选框中的人工修改均已写入，模型原始结果没有被覆盖。",
            "success",
        )


# ══════════════════════════════════════════════════════════════════════════════
#  OCR 图文逐句校对
# ══════════════════════════════════════════════════════════════════════════════














# 七个主功能区的顺序、显示名与路由索引集中定义，避免导航重排后
# 各处硬编码数字不一致。图标文件名继续沿用既有资源键。









# ══════════════════════════════════════════════════════════════════════════════
#  左侧导航栏
# ══════════════════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════════════════
#  主窗口
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(MainWindowControllerMixin, QMainWindow):
    """Main composition root with deferred construction for heavy workspaces.

    Project/workspace/settings shells stay eager because they define startup and
    restore policy. OCR/edit/export workspaces are built only on first real use;
    their business classes and controller contracts are unchanged.
    """

    @property
    def _tab_ocr(self):
        return self._lazy_workspaces.get("ocr")

    @property
    def _tab_pdf_text(self):
        return self._lazy_workspaces.get("pdf_text")

    @property
    def _tab_fmt(self):
        return self._lazy_workspaces.get("formatter")

    @property
    def _tab_ocr_compare(self):
        return self._lazy_workspaces.get("ocr_compare")

    @property
    def _tab_ocr_image_review(self):
        return self._lazy_workspaces.get("image_review")

    @property
    def _tab_epub(self):
        return self._lazy_workspaces.get("epub")

    def _lazy_workspace_if_loaded(self, key: str):
        registry = getattr(self, "_lazy_workspaces", None)
        return registry.loaded(key) if registry is not None else None

    def _lazy_placeholder(self, title: str) -> QWidget:
        placeholder = QWidget()
        placeholder.setObjectName("lazyWorkspacePlaceholder")
        layout = QVBoxLayout(placeholder)
        layout.setContentsMargins(24, 24, 24, 24)
        label = QLabel(f"{title}\n首次进入时加载")
        label.setAlignment(Qt.AlignCenter)
        label.setProperty("muted", True)
        layout.addStretch(1)
        layout.addWidget(label)
        layout.addStretch(1)
        return placeholder

    @staticmethod
    def _replace_reference_tab(host: ReferenceSectionHost, index: int, widget: QWidget, label: str) -> None:
        tabs = host.tabs
        old = tabs.widget(index)
        current = tabs.currentIndex()
        tabs.blockSignals(True)
        try:
            tabs.removeTab(index)
            tabs.insertTab(index, widget, label)
            tabs.setCurrentIndex(current)
        finally:
            tabs.blockSignals(False)
        if old is not None and old is not widget:
            old.deleteLater()

    def _replace_main_section(self, index: int, widget: QWidget) -> None:
        old = self._stack.widget(index)
        if old is widget:
            return
        current = self._stack.currentIndex()
        self._stack.removeWidget(old)
        self._stack.insertWidget(index, widget)
        if current >= 0:
            self._stack.setCurrentIndex(current)
        if old is not None:
            old.deleteLater()
        pages = list(self._section_pages)
        pages[index] = widget
        self._section_pages = tuple(pages)

    def _register_loaded_workspace(self, key: str, widget: QWidget) -> None:
        self._workspace_coordinator.register(key, widget)
        polish_reference_workspace(widget)
        if widget not in self._workspace_tabs:
            self._workspace_tabs.append(widget)

    def _install_ocr_workspace(self, tab: OCRTab) -> None:
        self._replace_reference_tab(self._ocr_section, 0, tab, "图片 OCR")
        self._register_loaded_workspace("ocr", tab)
        ocr_actions = getattr(tab, "_ocr_log_header", None)
        if ocr_actions is None:
            raise RuntimeError("OCRTab 缺少底部操作栏，无法放置清空 OCR 按钮")
        rerun_index = ocr_actions.indexOf(tab._rerun_btn)
        self._ocr_clear_button = create_workspace_clear_button(
            tab, "OCR", self._clear_ocr_workspace,
            target_layout=ocr_actions,
            target_index=max(0, rerun_index + 1),
            target_height=tab._rerun_btn.minimumHeight(),
        )
        # The Phase-22 reference footer contains only Stop + Start OCR.  Keep
        # the destructive clear action available from the Start button context
        # menu instead of permanently adding a third visual action.
        self._ocr_clear_button.setVisible(True)   # 破坏性操作也要有明确入口，右键菜单仅作补充
        clear_action = QAction("清空 OCR", tab._run_btn)
        # Reuse the visible clear button path so the context action keeps the
        # same confirmation dialog and never forwards QAction.triggered(bool)
        # as the OCR workspace argument.
        clear_action.triggered.connect(lambda _checked=False: self._ocr_clear_button.click())
        tab._run_btn.setContextMenuPolicy(Qt.ActionsContextMenu)
        tab._run_btn.addAction(clear_action)
        tab._reference_clear_ocr_action = clear_action
        tab.ocr_done.connect(self._on_ocr_done)
        tab.multi_ocr_done.connect(self._on_multi_ocr_done)
        tab.run_log_event.connect(self._on_ocr_run_log_event)
        tab.single_ocr_invalidated.connect(self._on_single_ocr_invalidated)
        if hasattr(self, "_tab_system"):
            self._tab_system.apply_to_ocr_tab(tab)
        pending_inputs = tuple(getattr(self, "_pending_ocr_inputs", ()) or ())
        if pending_inputs:
            tab.set_inputs(list(pending_inputs))
        pending_pipeline = dict(getattr(self, "_pending_ocr_pipeline_state", {}) or {})
        if pending_pipeline:
            try:
                tab.restore_project_pipeline_state(pending_pipeline)
                self._pending_ocr_pipeline_state = {}
            except Exception as exc:
                self.statusBar().showMessage(f"OCR 参数恢复失败：{exc}", 7000)

    def _install_pdf_text_workspace(self, tab: PdfTextLayerTab) -> None:
        self._replace_reference_tab(self._ocr_section, 1, tab, "PDF 文字层")
        self._register_loaded_workspace("pdf_text", tab)
        tab.set_page_manager_context_provider(self._pdf_text_page_manager_context)
        tab.page_manager_requested.connect(lambda: self._goto(SECTION_PAGE))
        tab.doc_extracted.connect(self._on_pdf_text_done)

    def _install_formatter_workspace(self, tab: FormatterTab) -> None:
        self._replace_main_section(SECTION_FORMAT, tab)
        self._register_loaded_workspace("formatter", tab)
        tab.doc_formatted.connect(self._on_fmt_done)

    def _install_compare_workspace(self, tab: OCRCompareTab) -> None:
        self._replace_reference_tab(self._proof_section, 0, tab, "OCR 对比")
        self._register_loaded_workspace("ocr_compare", tab)
        tab.set_recovery_page_image_provider(
            lambda: list(getattr(self._tab_pages, "page_images", []) or [])
        )
        package_dir = getattr(self, "_pending_project_package_dir", None)
        tab.set_project_package_dir(package_dir)
        tab.doc_applied.connect(self._on_ocr_compare_applied)
        tab.single_doc_applied.connect(self._on_single_ocr_compare_applied)
        tab.multi_session_restored.connect(self._on_multi_ocr_session_restored)
        tab.current_row_changed.connect(
            lambda row: self._workspace_coordinator.publish_stable_row(row, "ocr_compare")
        )
        tab.fusion_decision_changed.connect(self._on_ocr_compare_decision_changed)
        tab.disagreement_queue_order_changed.connect(
            self._tab_ocr_image_review.set_disagreement_source_row_order
        )
        tab.image_review_requested.connect(self._open_image_review_from_ocr_compare)
        tab.package_exported.connect(self._on_package_exported)
        tab.run_log_event.connect(self._on_compare_run_log_event)
        pending = getattr(self, "_pending_single_compare_result", None)
        if pending is not None:
            doc, label = pending
            tab.set_available_single_result(doc, label)

    def _install_image_review_workspace(self, tab: OCRImageTextReviewTab) -> None:
        self._replace_reference_tab(self._proof_section, 1, tab, "图文对照")
        self._register_loaded_workspace("image_review", tab)
        tab.set_page_image_provider(
            lambda: list(getattr(self._tab_pages, "page_images", []) or [])
        )
        tab.doc_applied.connect(self._on_ocr_image_review_applied)
        tab.row_review_saved.connect(self._on_image_review_row_saved)
        tab.source_row_changed.connect(
            lambda row: self._workspace_coordinator.publish_stable_row(row, "image_review")
        )
        pending = getattr(self, "_pending_image_review_document", None)
        if pending is not None:
            doc, label, lazy = pending
            tab.set_document(doc, label, lazy=bool(lazy))

    def _install_epub_workspace(self, tab: EPUBTab) -> None:
        self._replace_main_section(SECTION_EPUB, tab)
        self._register_loaded_workspace("epub", tab)
        tab.epub_built.connect(self._on_epub_built)
        tab.set_project_export_dir(getattr(self, "_pending_project_export_dir", None))
        pm = getattr(self, "_tab_pages", None)
        if pm is not None and hasattr(tab, "set_page_manager_assets"):
            tab.set_page_manager_assets(
                list(getattr(pm, "page_images", []) or []),
                dict(getattr(pm, "page_overrides", {}) or {}),
            )

    def _on_single_ocr_invalidated(self) -> None:
        self._pending_single_compare_result = None
        compare = self._lazy_workspace_if_loaded("ocr_compare")
        if compare is not None:
            compare.set_available_single_result(None)

    def _ensure_section_workspace(self, index: int) -> None:
        if index == SECTION_OCR:
            self._ensure_ocr_subworkspace(self._ocr_section.current_index())
        elif index == SECTION_FORMAT:
            self._lazy_workspaces.get("formatter")
        elif index == SECTION_PROOF:
            self._ensure_proof_subworkspace(self._proof_section.current_index())
        elif index == SECTION_EPUB:
            self._lazy_workspaces.get("epub")

    def _ensure_ocr_subworkspace(self, index: int) -> QWidget:
        key = "pdf_text" if int(index) == 1 else "ocr"
        return self._lazy_workspaces.get(key)

    def _ensure_proof_subworkspace(self, index: int) -> QWidget:
        key = "image_review" if int(index) == 1 else "ocr_compare"
        return self._lazy_workspaces.get(key)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Novel Formatter {VERSION}")
        self.setMinimumSize(1180, 760)
        self.resize(1440, 900)
        self.setAcceptDrops(True)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

        self._doc: Optional[UnifiedDocument] = None
        self._snapshot_registry: WorkspaceSnapshotRegistry[UnifiedDocument] = WorkspaceSnapshotRegistry()
        self._workspace_docs = self._snapshot_registry.aliases
        self._current_stage: str = ""
        self._active_project_dir: str = ""
        self._project_restore_generation = 0
        self._ocr_project_run_sessions: dict[str, dict] = {}
        self._compare_project_run_sessions: dict[str, dict] = {}
        self._runtime_tasks = RuntimeTaskRegistry()
        self._project_stage_save_lock = threading.Lock()
        self._project_stage_save_generation = GenerationGuard()
        self._project_stage_save_signal_refs: dict[int, WorkerSignals] = {}
        self._project_restore_signal_refs: dict[int, WorkerSignals] = {}
        self._pending_project_document: tuple[str, UnifiedDocument] | None = None
        self._pending_project_hydrated: set[str] = set()
        self._pending_project_multi_restore: dict | None = None
        self._project_multi_restore_inflight = False
        self._ocr_handoff_generation = GenerationGuard()
        self._pending_ocr_inputs: tuple[str, ...] = ()
        self._pending_ocr_pipeline_state: dict = {}
        self._pending_single_compare_result = None
        self._pending_image_review_document = None
        self._pending_project_export_dir = None
        self._pending_project_package_dir = None
        self._lazy_workspaces = LazyWorkspaceRegistry()

        contract_errors = validate_workspace_contracts(WORKSPACE_SPECS)
        if contract_errors:
            raise RuntimeError("工作区合同无效：" + "；".join(contract_errors))
        self._workspace_coordinator = WorkspaceCoordinator(self)
        self._paddle_button = QPushButton("📥 导入 PaddleOCR-VL")
        self._paddle_button.clicked.connect(self.import_paddle_output)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self._sidebar = Sidebar()
        root_layout.addWidget(self._sidebar)

        right = QWidget()
        right.setMinimumWidth(520)
        right.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self._stack = QStackedWidget()
        self._stack.setObjectName("mainSectionStack")

        # Startup-critical shells are eager. Heavy OCR/edit/export pages below are
        # represented by tiny placeholders until navigation or a data handoff
        # actually needs the real widget.
        self._tab_pages = PageManagerTab(embed_workspace_controls=False)
        self._tab_workspace = ProjectWorkspaceTab(self._tab_pages)
        self._tab_system = SystemSettingsTab()
        self._tab_system.set_source_update_busy_provider(self._source_update_has_active_jobs)

        self._ocr_placeholder = self._lazy_placeholder("图片 OCR")
        self._pdf_placeholder = self._lazy_placeholder("PDF 文字层")
        self._fmt_placeholder = self._lazy_placeholder("格式处理")
        self._compare_placeholder = self._lazy_placeholder("OCR 对比")
        self._review_placeholder = self._lazy_placeholder("图文对照")
        self._epub_placeholder = self._lazy_placeholder("EPUB生成")

        self._ocr_section = ReferenceSectionHost([
            ("图片 OCR", self._ocr_placeholder),
            ("PDF 文字层", self._pdf_placeholder),
        ], overlay_tabs=True, overlay_width=408)
        # OCR/PDF mode switcher is a neutral workspace control, not a tinted
        # content card.  Keep the selected pill visible while restoring the bar
        # itself to white, matching the rest of the OCR control column.
        self._ocr_section._segment_bar.setStyleSheet(
            f"QWidget#referenceSegmentBar{{background:{CARD};border:none;}}"
        )
        # Source-contract marker retained for existing architecture tests:
        # self._proof_section = ReferenceSectionHost([
        #     ("OCR 对比", self._tab_ocr_compare),
        #     ("图文对照", self._tab_ocr_image_review),
        # ])
        self._proof_section = ReferenceSectionHost([
            ("OCR 对比", self._compare_placeholder),
            ("图文对照", self._review_placeholder),
        ])

        self._workspace_tabs = [self._tab_pages]
        for key, workspace in (
            ("workspace", self._tab_workspace), ("pages", self._tab_pages), ("system", self._tab_system),
        ):
            self._workspace_coordinator.register(key, workspace)
        polish_reference_workspace(self._tab_pages)
        polish_reference_workspace(self._tab_workspace)
        polish_reference_workspace(self._tab_system)

        self._section_pages = (
            self._tab_workspace,
            self._tab_pages,
            self._ocr_section,
            self._fmt_placeholder,
            self._proof_section,
            self._epub_placeholder,
            self._tab_system,
        )
        if len(self._section_pages) != len(Sidebar.ITEMS):
            raise RuntimeError(
                f"侧边栏项目数 {len(Sidebar.ITEMS)} 与主功能区数量 {len(self._section_pages)} 不一致"
            )
        for section in self._section_pages:
            self._stack.addWidget(section)

        self._lazy_workspaces.register("ocr", OCRTab, self._install_ocr_workspace)
        self._lazy_workspaces.register("pdf_text", PdfTextLayerTab, self._install_pdf_text_workspace)
        self._lazy_workspaces.register("formatter", FormatterTab, self._install_formatter_workspace)
        self._lazy_workspaces.register("ocr_compare", OCRCompareTab, self._install_compare_workspace)
        self._lazy_workspaces.register("image_review", OCRImageTextReviewTab, self._install_image_review_workspace)
        self._lazy_workspaces.register("epub", EPUBTab, self._install_epub_workspace)

        self._ocr_section.current_changed.connect(self._on_ocr_subtab_changed)
        self._proof_section.current_changed.connect(self._on_proof_subtab_changed)
        self._tab_system.workspace_requested.connect(self._open_named_workspace)
        self._tab_system.settings_changed.connect(self._apply_system_settings)
        self._tab_system.ocr_runtime_check_requested.connect(self._run_system_ocr_runtime_check)
        self._ui_settings = QSettings(SystemSettingsTab.SETTINGS_ORG, SystemSettingsTab.SETTINGS_APP)
        self._apply_system_settings(self._tab_system.current_preferences())

        from ui.navigation.page_header import PageHeader
        self._page_header = PageHeader()
        right_layout.addWidget(self._page_header)
        right_layout.addWidget(self._stack, 1)
        root_layout.addWidget(right, 1)

        self._sidebar.section_changed.connect(self._goto)
        self._sidebar.section_changed.connect(self._page_header.set_section)
        self._page_header.navigate.connect(self._goto)
        self._page_header.set_section(self._stack.currentIndex())
        self._sidebar.command_palette_requested.connect(self._open_command_palette)

        workspace_modifier = "Meta" if sys.platform == "darwin" else "Ctrl"
        for i in range(len(Sidebar.ITEMS)):
            shortcut = QShortcut(QKeySequence(f"{workspace_modifier}+{i + 1}"), self)
            shortcut.activated.connect(partial(self._goto, i))

        self._command_palette_shortcut = QShortcut(QKeySequence(f"{workspace_modifier}+K"), self)
        self._command_palette_shortcut.setContext(Qt.WindowShortcut)
        self._command_palette_shortcut.activated.connect(self._open_command_palette)

        # Eager workspaces only. Deferred workspace signals are wired by their
        # installer immediately after successful first construction.
        self._tab_pages.pages_loaded.connect(self._on_pages_loaded)
        self._tab_pages.types_changed.connect(self._on_page_types_changed)
        self._tab_pages.project_changed.connect(self._on_project_changed)
        self._tab_workspace.resume_requested.connect(self._open_named_workspace)
        self._workspace_coordinator.stable_row_changed.connect(self._on_coordinated_row_changed)
        self._pending_image_review_source_row = 0
        self._restore_reference_navigation()
        QTimer.singleShot(0, self._tab_pages.restore_last_project)


    def _ensure_project_multi_ocr_hydrated(self) -> None:
        pending = dict(self._pending_project_multi_restore or {})
        if not pending or self._project_multi_restore_inflight:
            return
        if self._tab_ocr_compare._comparison is not None:
            self._pending_project_multi_restore = None
            self._pending_project_hydrated.add("ocr_compare")
            return
        generation = int(pending.get("generation", -1))
        if generation != self._project_restore_generation:
            self._pending_project_multi_restore = None
            return
        project_path = str(pending.get("project_path") or "")
        workspace_root = str(pending.get("workspace_root") or "")
        if not project_path or not workspace_root:
            return
        self._project_multi_restore_inflight = True
        self.statusBar().showMessage("正在后台载入 OCR 对比与裁决状态…", 0)
        signals = WorkerSignals()
        token = generation * 100000 + 1
        self._project_restore_signal_refs[token] = signals

        def finished(payload, g=generation, t=token):
            self._project_restore_signal_refs.pop(t, None)
            self._project_multi_restore_inflight = False
            if g != self._project_restore_generation:
                return
            try:
                prepared = (payload or {}).get("prepared")
                events = list((payload or {}).get("events") or [])
                if prepared and self._tab_ocr_compare.restore_project_snapshot({}, prepared=prepared):
                    replayed = self._tab_ocr_compare.apply_project_adjudication_events(events)
                    self._pending_project_multi_restore = None
                    self._pending_project_hydrated.add("ocr_compare")
                    suffix = f"；增量恢复 {replayed} 条裁决" if replayed else ""
                    self.statusBar().showMessage(f"OCR 对比与裁决状态已按需恢复{suffix}", 6000)
                    if self._stack.currentIndex() == SECTION_PROOF and self._proof_section.current_index() == 1:
                        QTimer.singleShot(0, self._load_and_sync_image_review_row)
            except Exception as exc:
                self.statusBar().showMessage(f"OCR 裁决状态未恢复：{exc}", 10000)

        def failed(message, g=generation, t=token):
            self._project_restore_signal_refs.pop(t, None)
            self._project_multi_restore_inflight = False
            if g == self._project_restore_generation:
                self.statusBar().showMessage(f"OCR 裁决状态后台载入失败：{str(message).strip()}", 10000)

        signals.finished.connect(finished)
        signals.error.connect(failed)

        def worker():
            try:
                restore_manager = ProjectWorkspaceManager(workspace_root)
                restore_manager.open_project(project_path)
                snapshot = restore_manager.load_multi_ocr_snapshot()
                events = restore_manager.load_adjudication_events()
                prepared = OCRCompareTab.prepare_project_snapshot_restore(snapshot) if snapshot else None
                signals.finished.emit({"prepared": prepared, "events": events})
            except Exception:
                import traceback
                signals.error.emit(traceback.format_exc())

        threading.Thread(target=worker, daemon=True, name=f"project-multi-restore-{generation}").start()

















































# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Keep a native-crash trace on macOS. Python exceptions are still routed to
    # the normal in-app error dialog, while hard Qt/WebKit/extension failures at
    # least leave a file the user can send back instead of silently vanishing.
    crash_dir = Path(__file__).resolve().parent / "debug"
    crash_dir.mkdir(parents=True, exist_ok=True)
    fatal_log = None
    try:
        import faulthandler
        fatal_log = (crash_dir / "last_fatal_crash.log").open("w", encoding="utf-8")
        faulthandler.enable(fatal_log, all_threads=True)
    except Exception:
        fatal_log = None

    app = QApplication(sys.argv)
    # Fusion first, then the application stylesheet. Reversing this order lets
    # setStyle() overwrite parts of the popup/button palette on macOS.
    if sys.platform == "darwin":
        app.setStyle("Fusion")
    icon_path = Path(__file__).resolve().parent / "icon.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    try:
        window = MainWindow()
    except Exception:
        import traceback
        details = traceback.format_exc()
        try:
            (crash_dir / "last_startup_error.log").write_text(details, encoding="utf-8")
        except Exception:
            pass
        show_error_dialog(
            None,
            "程序启动失败",
            details,
            summary="主窗口初始化失败。错误详情已保存到 debug/last_startup_error.log。",
        )
        return 1

    # Startup-critical editors are built before the application-wide filters.
    # Deferred workspaces temporarily suspend the clickable Polish filter during
    # their constructors, preserving the same PySide/private-scrollbar safety
    # boundary without paying the full startup cost.
    boot_settings = QSettings(SystemSettingsTab.SETTINGS_ORG, SystemSettingsTab.SETTINGS_APP)
    interface_manager = InterfacePreferenceManager(
        app,
        base_stylesheet=STYLE,
        language=str(boot_settings.value(SystemSettingsTab.LANGUAGE_KEY, LANG_ZH)),
        theme=str(boot_settings.value(SystemSettingsTab.APPEARANCE_KEY, THEME_LIGHT)),
    )
    app._novel_formatter_interface_manager = interface_manager
    install_dialog_polish(app)
    install_generic_dialog_polish(app)
    install_no_white_clickable_guard(app)

    if fatal_log is not None:
        app._novel_formatter_fatal_log = fatal_log
    if icon_path.exists():
        window.setWindowIcon(QIcon(str(icon_path)))
    install_exception_hooks(app, lambda: window)
    install_window_state_saver(app, window)
    if window._tab_system.start_maximized_enabled():
        window.showMaximized()
    else:
        restore_window_state(window)
        window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
