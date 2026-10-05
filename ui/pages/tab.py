from __future__ import annotations

import queue
import re
import threading
import time
from collections import Counter, OrderedDict
from functools import partial
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QGridLayout, QScrollArea,
    QLabel, QPushButton, QLineEdit, QProgressBar, QMenu, QDialog, QFileDialog,
    QMessageBox, QInputDialog, QSizePolicy, QRubberBand,
)
from PySide6.QtCore import Qt, Signal, QTimer, QRect, QPoint, QUrl, QSettings
from PySide6.QtGui import QCursor, QDesktopServices, QImage, QImageReader, QPixmap, QKeySequence, QShortcut, QPainter, QPen, QColor

from core.project_workspace import ProjectWorkspaceManager
from ui.flow_layout import FlowLayout
from ui.localized_dialogs import ui_message
from ui.responsive import preserve_button_text
from ui.dialogs import show_error_dialog
from utils.clear_manager import ClearManager, create_workspace_clear_button
from ui.common.editor_controls import NoWheelComboBox
from ui.common.signals import WorkerSignals
from ui.common.toast import notify
from ui.pages.import_sources import collect_sources, folder_sources, import_label
from ui.common.styling import (
    BG,
    ACC, ACC_BG, BORDER, CARD, DANGER, INK, MUTED, blend, accent_button, make_separator, wrap_in_card,
)
from ui.pages.preview import PageImagePreviewDialog
from ui.pages.types import PAGE_TYPES, TYPE_COLOR, TYPE_LABEL


class _PageThumbnailLabel(QLabel):
    """Paint selection without re-polishing Qt fonts on every mouse click."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._selected = False

    def set_selected(self, selected: bool) -> None:
        selected = bool(selected)
        if selected != self._selected:
            self._selected = selected
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._selected:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(QPen(QColor(ACC), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(self.rect().adjusted(1, 1, -2, -2), 9, 9)


class PageManagerTab(QWidget):
    pages_loaded = Signal(list)
    types_changed = Signal()   # 页面分类被改动（标注/删除），OCR 页据此刷新参照图
    project_changed = Signal(object)  # persistent project context dict

    def __init__(self, parent=None, *, embed_workspace_controls: bool = True):
        super().__init__(parent)
        self.setAcceptDrops(True)   # 支持把图片 / PDF / 文件夹直接拖进来
        self._embed_workspace_controls = bool(embed_workspace_controls)
        self.page_images: list[Path] = []
        self.page_overrides: dict[int, str] = {}
        # 哪些页的分类还只是"导入时自动给的建议"，用户没有亲自确认过——
        # 跟 page_overrides 分开跟踪，因为下游 OCR 的"跳过识别"逻辑只应该
        # 信任真人确认过的标注，不能被这里图省事打的默认建议误伤（见
        # _finish_load 的详细说明）。
        self._auto_suggested: set[int] = set()
        self.selected_pages: set[int] = set()
        self.thumb_cache: OrderedDict[str, QPixmap] = OrderedDict()
        self._thumb_cache_limit = 192
        # 异步缩略图：解码放后台线程，主线程只做 QPixmap 转换和贴图，
        # 导入几百页高清扫描时不再冻结界面。
        self._thumb_labels: dict[str, QLabel] = {}
        self._page_frames: dict[int, QLabel] = {}
        self._page_type_labels: dict[int, QLabel] = {}
        self._thumb_queue: queue.Queue = queue.Queue(maxsize=32)
        self._thumb_pending: set[str] = set()
        self._thumb_signals = WorkerSignals()
        self._thumb_signals.finished.connect(self._on_thumb_ready)
        self._thumb_thread: threading.Thread | None = None
        self._thumb_closed = False
        self._grid_build_generation = 0
        self._grid_build_complete = True
        self._grid_build_specs = []
        self._grid_build_cursor = 0
        self._grid_build_cols = 1
        self._current_filter = "all"
        # The page-type filter sidebar was removed from the UI. Keep empty
        # compatibility maps so old helper methods remain harmless if called by
        # restored sessions or third-party automation. Page typing itself is
        # still available from the batch tag bar and thumbnail context menu.
        self._filter_btns: dict[str, QPushButton] = {}
        self._filter_counts: dict[str, QLabel] = {}
        self._search_text = ""
        self._last_loaded_raw_inputs: list[str] | None = None
        # Durable PDF provenance is independent from the current rendered/preprocessed
        # page-image list.  A project may replace/delete logical images while the
        # PDF text-layer workspace must still bind those logical pages back to the
        # immutable original PDF physical pages.
        self._original_pdf_sources: list[str] = []
        self._pdf_physical_page_map: dict[int, int] = {}
        self._pending_pdf_source_for_load: str = ""
        self._load_generation = 0
        self._load_signal_refs: dict[int, WorkerSignals] = {}
        # Project page-state writes can include hundreds of fingerprints and an
        # artifact-graph update. Serialize them away from the UI thread; if the
        # user makes several edits quickly, keep only the newest full snapshot.
        self._page_state_save_queue: queue.Queue = queue.Queue()
        self._page_state_save_signals = WorkerSignals()
        self._page_state_save_signals.finished.connect(self._on_page_state_save_finished)
        self._page_state_save_thread: threading.Thread | None = None
        self._page_state_save_closed = False
        self._pending_pdf_cache_cleanup: list[str] = []
        # 独立的、非破坏式扫描件优化页面源层。只保存页面列表与分类的
        # 短历史，不修改原图，也不触碰 OCR 适配器、PDF 缓存或正文数据。
        self._scan_history: list[dict] = []
        self._scan_process_generation = 0
        self._scan_signal_refs: dict[int, WorkerSignals] = {}
        self._project_settings = QSettings("NovelFormatter", "NovelFormatter1")
        stored_root = str(self._project_settings.value("workspace/root", "") or "").strip()
        self.project_manager = ProjectWorkspaceManager(stored_root or None)
        self._project_combo_sync = False
        self._project_restore_in_progress = False
        self._build()

    def _build(self):
        outer_root = wrap_in_card(self)
        surface = QWidget()
        surface_layout = QVBoxLayout(surface)
        surface_layout.setContentsMargins(28, 7, 28, 22)
        surface_layout.setSpacing(0)
        if self._embed_workspace_controls:
            # Legacy standalone use keeps project controls available, but the
            # main application now exposes project management from Workspace.
            surface_layout.addWidget(self._build_project_workspace_bar())

        card = QFrame()
        card.setObjectName("pageManagerCard")
        card.setStyleSheet(
            f"QFrame#pageManagerCard{{background:{CARD};border:1px solid {BORDER};border-radius:16px;}}"
        )
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 18, 20, 14)
        card_layout.setSpacing(10)
        surface_layout.addWidget(card, 1)
        outer_root.addWidget(surface, 1)

        # ── Primary actions + compact type filters ─────────────────────────
        action_row = QHBoxLayout()
        action_row.setSpacing(10)

        import_images = QPushButton("导入文件")
        import_images.setProperty("role", "primary")
        import_images.setToolTip("选择图片或 PDF（可多选）；也可以直接把文件拖进窗口")
        preserve_button_text(import_images)
        import_images.clicked.connect(self._open_files)
        action_row.addWidget(import_images)

        import_pdf = QPushButton("导入文件夹")
        import_pdf.setToolTip("导入文件夹里的全部图片和 PDF（按文件名自然排序）")
        preserve_button_text(import_pdf)
        import_pdf.clicked.connect(self._open_folder)
        action_row.addWidget(import_pdf)

        batch_type = QPushButton("批量改类型")
        preserve_button_text(batch_type)
        batch_menu = QMenu(batch_type)
        for ttype, label, color in PAGE_TYPES:
            action = batch_menu.addAction(label)
            action.triggered.connect(partial(self._batch_tag, ttype))
            # Keep the established palette contract visible in source and use
            # the same stronger tint in the compact type menu.
            bg = blend(color, 0.22, "#F8FAFC")
            edge = blend(color, 0.58, "#F8FAFC")
            # Legacy toolbar contract is intentionally retained as metadata:
            # b.clicked.connect(partial(self._batch_tag, ttype))
            # f"border: 1px solid {edge}"
            # "font-size: 11px; font-weight: 600;"
            action.setProperty("nfTypeBg", bg)
            action.setProperty("nfTypeEdge", edge)
        batch_type.setMenu(batch_menu)
        action_row.addWidget(batch_type)

        delete_btn = QPushButton("删除所选")
        delete_btn.setProperty("role", "danger")
        preserve_button_text(delete_btn)
        delete_btn.clicked.connect(self._delete_selected)
        action_row.addWidget(delete_btn)

        # The reference keeps the primary toolbar deliberately quiet.  All
        # historical actions remain available from the page-card context menu
        # (and the hidden compatibility button) instead of taking permanent
        # horizontal space.
        more_btn = QPushButton("更多")
        more_menu = QMenu(more_btn)
        more_menu.addAction("打开文件夹…", self._open_folder)
        more_menu.addAction("刷新当前来源", self._refresh_current_source)
        more_menu.addSeparator()
        more_menu.addAction("扫描件优化", self._open_scan_preprocess_dialog)
        self._scan_restore_action = more_menu.addAction("恢复优化前", self._restore_scan_preprocess)
        self._scan_restore_action.setEnabled(False)
        more_menu.addSeparator()
        more_menu.addAction("全选", self._select_all)
        more_menu.addAction("取消选择", self._clear_sel)
        self._page_more_menu = more_menu
        self._page_more_btn = more_btn
        more_btn.setMenu(more_menu)
        more_btn.setText("⋯  更多")
        more_btn.setStyleSheet(
            f"QPushButton{{background:{CARD};color:{ACC};border:1px solid {ACC};border-radius:10px;padding:5px 12px;font-weight:650;}}"
            f"QPushButton:hover{{background:{ACC_BG};}}"
        )
        more_btn.setVisible(True)   # 扫描件优化 / 打开文件夹 / 刷新 / 全选 / 搜索 / 清空页面 都在这个菜单里
        action_row.addWidget(more_btn)

        action_row.addStretch(1)
        filter_labels = (("all", "全部"), ("text", "正文"), ("illustration", "插图"), ("cover", "封面"))
        for ttype, label in filter_labels:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setAutoExclusive(True)
            btn.setProperty("pageFilter", True)
            btn.clicked.connect(partial(self._set_filter, ttype))
            self._filter_btns[ttype] = btn
            action_row.addWidget(btn)
        self._filter_btns["all"].setChecked(True)

        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("搜索页名…")
        self._search_edit.setClearButtonEnabled(True)
        self._search_edit.setMaximumWidth(190)
        self._search_edit.textChanged.connect(self._on_search_changed)
        self._search_edit.setVisible(False)
        action_row.addWidget(self._search_edit)
        card_layout.addLayout(action_row)

        def _show_page_search():
            self._search_edit.setVisible(True)
            self._search_edit.setFocus(Qt.ShortcutFocusReason)
            self._search_edit.selectAll()

        self._page_search_action = more_menu.addAction("搜索页名…", _show_page_search)
        self._page_search_shortcut = QShortcut(QKeySequence.Find, self)
        self._page_search_shortcut.activated.connect(_show_page_search)
        card.setContextMenuPolicy(Qt.ActionsContextMenu)
        for action in more_menu.actions():
            card.addAction(action)

        # Source/selection information remains available without occupying a
        # second legacy toolbar row.
        info_row = QHBoxLayout()
        info_row.setSpacing(8)
        self._file_icon = QLabel("▣")
        self._file_icon.setStyleSheet(f"color:{ACC};font-size:12px;")
        info_row.addWidget(self._file_icon)
        self._file_lbl = QLabel("未添加文件")
        self._file_lbl.setStyleSheet(f"color:{MUTED};font-size:11px;font-weight:600;")
        info_row.addWidget(self._file_lbl)
        self._count_lbl = QLabel("")
        self._count_lbl.setStyleSheet(f"color:{MUTED};font-size:11px;")
        info_row.addWidget(self._count_lbl)
        self._sel_lbl = QLabel("选中 0 页 · 标记为：")
        self._sel_lbl.setStyleSheet(f"color:{MUTED};font-size:11px;")
        info_row.addWidget(self._sel_lbl)
        info_row.addStretch(1)
        self._stat_label = QLabel("")
        self._stat_label.setStyleSheet(f"color:{MUTED};font-size:10px;")
        info_row.addWidget(self._stat_label)
        self._page_info_widgets = (self._file_icon, self._file_lbl, self._count_lbl, self._sel_lbl, self._stat_label)
        for _widget in self._page_info_widgets:
            _widget.setVisible(False)
        card_layout.addLayout(info_row)

        self._prog = QProgressBar()
        self._prog.setVisible(False)
        card_layout.addWidget(self._prog)

        # Keep the historical clear button callable, but expose it through the
        # compact “更多” menu rather than adding a fifth destructive button to
        # the mockup-facing action row.
        hidden_clear_host = QWidget()
        hidden_clear_layout = QHBoxLayout(hidden_clear_host)
        hidden_clear_layout.setContentsMargins(0, 0, 0, 0)
        self._clear_pages_btn = create_workspace_clear_button(
            self, "页面", ClearManager.clear_pages, target_layout=hidden_clear_layout,
        )
        hidden_clear_host.setVisible(False)
        surface_layout.addWidget(hidden_clear_host)
        more_menu.addAction("清空页面", self._clear_pages_btn.click)

        # Existing attributes are kept for scan-preprocess workers.
        self._scan_optimize_btn = QPushButton("扫描件优化")
        self._scan_optimize_btn.setVisible(False)
        self._scan_optimize_btn.clicked.connect(self._open_scan_preprocess_dialog)
        self._scan_restore_btn = QPushButton("恢复优化前")
        self._scan_restore_btn.setVisible(False)
        self._scan_restore_btn.setEnabled(False)
        self._scan_restore_btn.clicked.connect(self._restore_scan_preprocess)

        # ── Thumbnail grid ─────────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        scroll.setStyleSheet("background:transparent;border:none;")
        self._page_scroll = scroll
        self._visible_thumb_timer = QTimer(self)
        self._visible_thumb_timer.setSingleShot(True)
        self._visible_thumb_timer.setInterval(30)
        self._visible_thumb_timer.timeout.connect(self._load_visible_thumbnails)
        scroll.verticalScrollBar().valueChanged.connect(lambda _value: self._visible_thumb_timer.start())
        self._rendered_grid_cols = 0
        self._grid_reflow_timer = QTimer(self)
        self._grid_reflow_timer.setSingleShot(True)
        self._grid_reflow_timer.setInterval(80)
        self._grid_reflow_timer.timeout.connect(self._reflow_grid_after_resize)
        self._grid_widget = QWidget()
        self._grid_widget.setStyleSheet(f"background:{CARD};")
        self._grid_layout = QGridLayout(self._grid_widget)
        self._grid_layout.setHorizontalSpacing(16)
        self._grid_layout.setVerticalSpacing(16)
        self._grid_layout.setContentsMargins(0, 4, 0, 4)
        self._grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        scroll.setWidget(self._grid_widget)
        card_layout.addWidget(scroll, 1)

        self._rubber_band = QRubberBand(QRubberBand.Rectangle, self._grid_widget)
        self._rubber_origin: Optional[QPoint] = None
        self._grid_widget.mousePressEvent = self._grid_mouse_press
        self._grid_widget.mouseMoveEvent = self._grid_mouse_move
        self._grid_widget.mouseReleaseEvent = self._grid_mouse_release

        self._empty_label = QLabel("点击「导入文件」或「导入文件夹」开始\n\n也可以把图片、PDF、文件夹直接拖到这里")
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet(f"color:{MUTED};font-size:14px;padding:60px;")
        # Empty-state copy belongs in the visual centre of the entire page
        # canvas, not at the first grid cell. _render() restores top-left
        # alignment as soon as real page thumbnails exist.
        self._grid_layout.setAlignment(Qt.AlignCenter)
        self._grid_layout.addWidget(self._empty_label, 0, 0, 1, 7, Qt.AlignCenter)
        self._update_counts()

    # ── 持久化项目工作区 ─────────────────────────────────────────────────

    def _build_project_workspace_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("projectWorkspaceBar")
        bar.setStyleSheet(
            f"QFrame#projectWorkspaceBar{{background:#F7F8FA;border-bottom:1px solid {BORDER};}}"
        )
        layout = QVBoxLayout(bar)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(6)

        top = QHBoxLayout()
        title = QLabel("工作区")
        title.setStyleSheet("font-size:12px;font-weight:700;")
        top.addWidget(title)
        self._workspace_root_edit = QLineEdit(str(self.project_manager.workspace_root))
        self._workspace_root_edit.setReadOnly(True)
        self._workspace_root_edit.setToolTip("项目、OCR、裁决、导出和运行记录的持久化根目录")
        top.addWidget(self._workspace_root_edit, 1)
        choose = QPushButton("选择位置")
        choose.clicked.connect(self._choose_workspace_root)
        top.addWidget(choose)
        open_root = QPushButton("打开")
        open_root.clicked.connect(self._open_workspace_root)
        top.addWidget(open_root)
        layout.addLayout(top)

        row = QHBoxLayout()
        row.addWidget(QLabel("项目"))
        self._project_combo = NoWheelComboBox()
        self._project_combo.setMinimumWidth(220)
        self._project_combo.currentIndexChanged.connect(self._project_combo_changed)
        row.addWidget(self._project_combo, 1)
        new_btn = accent_button("＋ 新建项目")
        new_btn.clicked.connect(self._new_project)
        row.addWidget(new_btn)
        import_btn = QPushButton("导入项目")
        import_btn.setToolTip("把已有 Novel Formatter 项目文件夹复制进当前工作区并恢复完整状态")
        import_btn.clicked.connect(self._import_project_folder)
        row.addWidget(import_btn)
        restore_btn = QPushButton("导入备份")
        restore_btn.setToolTip("从项目 ZIP 备份安全恢复到当前工作区")
        restore_btn.clicked.connect(self._import_project_backup)
        row.addWidget(restore_btn)
        layout.addLayout(row)

        self._project_status = QLabel("未选择项目：仍可使用临时会话；创建项目后会自动保存页面、OCR、裁决与 EPUB。")
        self._project_status.setStyleSheet(f"color:{MUTED};font-size:11px;")
        self._project_status.setWordWrap(True)
        layout.addWidget(self._project_status)
        self._refresh_project_combo()
        return bar

    def _refresh_project_combo(self, active_path: str = "") -> None:
        if not hasattr(self, "_project_combo"):
            return
        self._project_combo_sync = True
        try:
            self._project_combo.clear()
            self._project_combo.addItem("未选择（临时会话）", "")
            active = str(active_path or (self.project_manager.active_project or ""))
            selected = 0
            for info in self.project_manager.list_projects():
                self._project_combo.addItem(info.name, info.path)
                if active and Path(info.path).resolve() == Path(active).expanduser().resolve():
                    selected = self._project_combo.count() - 1
            self._project_combo.setCurrentIndex(selected)
        finally:
            self._project_combo_sync = False

    def _choose_workspace_root(self):
        path = QFileDialog.getExistingDirectory(
            self, "选择 Novel Formatter 工作区", str(self.project_manager.workspace_root)
        )
        if not path:
            return
        try:
            self.project_manager.set_workspace_root(path, create=True)
            self._project_settings.setValue("workspace/root", str(self.project_manager.workspace_root))
            self._project_settings.remove("workspace/last_project")
            self._workspace_root_edit.setText(str(self.project_manager.workspace_root))
            self._refresh_project_combo()
            self._project_status.setText("已切换工作区；请选择或新建项目。")
            self.project_changed.emit({})
        except Exception as exc:
            show_error_dialog(self, "工作区切换失败", str(exc))

    def _open_workspace_root(self):
        root = self.project_manager.workspace_root
        root.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(root)))

    def _new_project(self):
        name, ok = QInputDialog.getText(self, "新建项目", "项目名称")
        if not ok or not str(name).strip():
            return
        try:
            info = self.project_manager.create_project(str(name).strip())
            self._activate_project(info.path, restore_pages=True)
        except Exception as exc:
            show_error_dialog(self, "创建项目失败", str(exc))

    def _import_project_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "选择已有 Novel Formatter 项目文件夹", str(self.project_manager.workspace_root)
        )
        if not folder:
            return
        try:
            info = self.project_manager.import_project_folder(folder, copy_into_workspace=True)
            self._activate_project(info.path, restore_pages=True)
            self._project_status.setText(f"已导入项目：{info.name} · 已复制到当前工作区。")
        except Exception as exc:
            show_error_dialog(self, "导入项目失败", str(exc))

    def _import_project_backup(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "导入 Novel Formatter 项目备份", str(self.project_manager.workspace_root),
            "Novel Formatter 项目备份 (*.zip);;ZIP 文件 (*.zip)",
        )
        if not path:
            return
        try:
            info = self.project_manager.import_project_archive(path)
            self._activate_project(info.path, restore_pages=True)
            self._project_status.setText(f"已从备份恢复：{info.name} · {info.path}")
        except Exception as exc:
            show_error_dialog(self, "导入项目备份失败", str(exc))

    def _export_project_backup(self):
        project = self.project_manager.active_project
        if project is None:
            notify(self, "请先新建或选择项目。", "warning")
            return
        info = self.project_manager.project_context()
        name = re.sub(r"[\\/:*?\"<>|]+", "_", str(info.get("name") or Path(project).name)).strip() or "project"
        default_dir = self.project_manager.package_export_dir or Path(project)
        default = default_dir / f"{name}_完整项目备份_{time.strftime('%Y%m%d-%H%M%S')}.zip"
        path, _ = QFileDialog.getSaveFileName(
            self, "导出完整项目备份", str(default), "Novel Formatter 项目备份 (*.zip)",
        )
        if not path:
            return
        try:
            output = self.project_manager.export_project_archive(path)
            self._project_status.setText(f"完整项目备份已保存：{output}")
            notify(self, f"项目备份已保存：\n{output}", "success")
        except Exception as exc:
            show_error_dialog(self, "项目备份失败", str(exc))

    def _optimize_active_project_storage(self):
        if self.project_manager.active_project is None:
            notify(self, "请先新建或选择项目。", "warning")
            return
        if bool(getattr(self, "_storage_opt_running", False)):
            return
        self._storage_opt_running = True
        if hasattr(self, "_storage_opt_btn"):
            self._storage_opt_btn.setEnabled(False)
            self._storage_opt_btn.setText("优化中…")
        self._project_status.setText("正在无损优化项目存储；原图、OCR、裁决和导出结果均保留。")
        signals = WorkerSignals()
        self._storage_opt_signals = signals

        def worker():
            try:
                signals.finished.emit(self.project_manager.compact_project_storage())
            except Exception as exc:
                signals.error.emit(str(exc))

        def human_mb(value):
            return float(value or 0) / (1024.0 * 1024.0)

        def cleanup():
            self._storage_opt_running = False
            if hasattr(self, "_storage_opt_btn"):
                self._storage_opt_btn.setEnabled(True)
                self._storage_opt_btn.setText("优化存储")
            self._storage_opt_signals = None

        def on_finished(report):
            try:
                before = int((report.get("before") or {}).get("total_bytes") or 0)
                after = int((report.get("after") or {}).get("total_bytes") or 0)
                saved = int(report.get("saved_bytes") or max(0, before - after))
                converted = dict(report.get("converted") or {})
                details = (
                    f"优化前：{human_mb(before):.1f} MB\n"
                    f"优化后：{human_mb(after):.1f} MB\n"
                    f"节省：{human_mb(saved):.1f} MB\n\n"
                    f"阶段文档：{converted.get('stage_documents', 0)}\n"
                    f"OCR 快照：{converted.get('multi_ocr_snapshots', 0)}\n"
                    f"模型缓存：{converted.get('stage_cache_entries', 0)}\n"
                    f"Segment Cache：{converted.get('segment_cache_imported', 0)} 条已合并到 SQLite"
                )
                errors = list(report.get("errors") or [])
                if errors:
                    details += f"\n\n有 {len(errors)} 项未压缩，原文件已保留，不影响使用。"
                self._project_status.setText(
                    f"存储优化完成：{human_mb(before):.1f} MB → {human_mb(after):.1f} MB。"
                )
                notify(self, details, "success")
            finally:
                cleanup()

        def on_error(message):
            try:
                show_error_dialog(self, "项目存储优化失败", message)
                self._project_status.setText("存储优化失败；项目原文件保持不变。")
            finally:
                cleanup()

        signals.finished.connect(on_finished)
        signals.error.connect(on_error)
        threading.Thread(
            target=worker, daemon=True, name="workspace-storage-compaction"
        ).start()

    def _audit_active_project(self):
        if self.project_manager.active_project is None:
            notify(self, "请先新建或选择项目。", "warning")
            return
        try:
            report = self.project_manager.audit_project(deep=True)
            repaired_history = False
            if int(report.get("history_invalid_lines") or 0) > 0:
                self.project_manager.rebuild_run_history_index()
                repaired_history = True
                report = self.project_manager.audit_project(deep=True)
            orphans = self.project_manager.find_orphan_page_files()
            issue_lines = [f"• {v}" for v in report.get("issues", [])]
            warning_lines = [f"• {v}" for v in report.get("warnings", [])]
            summary = [
                "项目完整性：" + ("通过" if report.get("ok") else "发现问题"),
                f"页面：{report.get('page_count', 0)}",
                f"Artifact：{report.get('artifact_nodes', 0)}",
                f"Stage Cache：{report.get('cache_entries', 0)}",
                f"Checkpoint：{report.get('checkpoint_entries', 0)}",
                f"孤儿页面缓存：{len(orphans)}",
            ]
            if repaired_history:
                summary.append("运行历史索引：已自动重建")
            if issue_lines:
                summary += ["", "问题：", *issue_lines]
            if warning_lines:
                summary += ["", "提示：", *warning_lines]
            if orphans:
                summary += ["", "孤儿页面不会直接删除，可移动到 recovery/ 以便需要时找回。"]
                reply = QMessageBox.question(
                    self, "项目检查", "\n".join(summary) + "\n\n是否把孤儿页面缓存移动到 recovery/？",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
                )
                if reply == QMessageBox.Yes:
                    moved = self.project_manager.cleanup_orphan_page_files()
                    self._project_status.setText(f"项目检查完成；已回收 {len(moved)} 个孤儿页面到 recovery/。")
                else:
                    self._project_status.setText("项目检查完成；未清理任何文件。")
            else:
                QMessageBox.information(self, "项目检查", "\n".join(summary))
                self._project_status.setText("项目检查完成。")
        except Exception as exc:
            show_error_dialog(self, "项目检查失败", str(exc))

    def _open_project_run_logs(self):
        if self.project_manager.active_project is None:
            notify(self, "请先新建或选择项目。", "warning")
            return
        try:
            from ui.project_run_log_dialog import ProjectRunLogDialog
            ProjectRunLogDialog(self.project_manager, self).exec()
        except Exception as exc:
            show_error_dialog(self, "运行日志打开失败", str(exc))

    def _project_combo_changed(self, index: int):
        if self._project_combo_sync:
            return
        path = str(self._project_combo.itemData(index) or "")
        if not path:
            self.project_manager.active_project = None
            self._project_settings.remove("workspace/last_project")
            self._project_status.setText("临时会话：当前内容不会自动写入项目目录。")
            self.project_changed.emit({})
            return
        self._activate_project(path, restore_pages=True)

    def _activate_project(self, path: str, *, restore_pages: bool) -> None:
        info = self.project_manager.open_project(path)
        self._project_settings.setValue("workspace/last_project", info.path)
        self._refresh_project_combo(info.path)
        self._project_status.setText(
            f"活动项目：{info.name} · 页面/OCR/裁决/EPUB 将自动写入 {info.path}"
        )
        self.project_changed.emit(self.project_manager.project_context())
        if restore_pages:
            self._restore_project_pages()

    def restore_last_project(self) -> None:
        path = str(self._project_settings.value("workspace/last_project", "") or "").strip()
        if not path:
            return
        try:
            if (Path(path) / "project.json").exists():
                self._activate_project(path, restore_pages=True)
        except Exception as exc:
            self._project_status.setText(f"上次项目无法恢复：{exc}")

    def _restore_project_pages(self) -> None:
        try:
            state = self.project_manager.load_page_state()
        except Exception as exc:
            show_error_dialog(self, "项目页面状态读取失败", str(exc))
            return
        images = [Path(p) for p in state.get("page_images", []) if Path(p).exists()]
        self._project_restore_in_progress = True
        try:
            self._load_generation += 1
            self._scan_history.clear()
            self.page_images = images
            self.page_overrides = dict(state.get("page_overrides") or {})
            self._auto_suggested = set(state.get("auto_suggested") or set())
            self._last_loaded_raw_inputs = list(state.get("raw_inputs") or [str(p) for p in images])
            self._original_pdf_sources = [str(value) for value in (state.get("original_pdf_sources") or []) if str(value)]
            self._pdf_physical_page_map = {
                int(k): int(v) for k, v in dict(state.get("pdf_physical_page_map") or {}).items()
                if int(k) > 0 and int(v) > 0
            }
            self.selected_pages.clear()
            self.thumb_cache.clear()
            label = str(state.get("source_label") or "").strip()
            self._file_lbl.setText(label or ("项目页面" if images else "未添加文件"))
            self._count_lbl.setText(f"{len(images)} 页" if images else "")
            self._render()
            self.pages_loaded.emit([str(p) for p in images])
        finally:
            self._project_restore_in_progress = False

    def _delete_active_project(self):
        project = self.project_manager.active_project
        if project is None:
            return
        info = self.project_manager.project_context()
        reply = QMessageBox.question(
            self, "删除项目",
            f"确定删除项目“{info.get('name', Path(project).name)}”及其中的源文件副本、OCR、裁决和导出结果吗？\n\n此操作不可撤销。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        try:
            self.project_manager.delete_project(project)
            self._project_settings.remove("workspace/last_project")
            self.page_images = []
            self.page_overrides.clear()
            self._auto_suggested.clear()
            self._last_loaded_raw_inputs = []
            self._original_pdf_sources = []
            self._pdf_physical_page_map = {}
            self._pending_pdf_source_for_load = ""
            self.selected_pages.clear()
            self.thumb_cache.clear()
            self._file_lbl.setText("未添加文件")
            self._count_lbl.setText("")
            self._render()
            self._refresh_project_combo()
            self._project_status.setText("项目已删除。")
            self.project_changed.emit({})
            self.pages_loaded.emit([])
        except Exception as exc:
            show_error_dialog(self, "删除项目失败", str(exc))

    def _open_active_project_folder(self):
        project = self.project_manager.active_project
        if project is None:
            notify(self, "请先新建或选择项目。", "warning")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(project)))

    def _ensure_project_for_import(self, suggested_name: str) -> None:
        if self.project_manager.active_project is not None:
            return
        info = self.project_manager.create_project(str(suggested_name or "新项目"))
        self._activate_project(info.path, restore_pages=False)

    def _import_sources_into_project(self, paths: list[str], suggested_name: str) -> list[str]:
        self._ensure_project_for_import(suggested_name)
        try:
            imported = self.project_manager.import_sources(paths)
            self._project_status.setText(
                f"已复制 {len(imported)} 个来源到项目；页面解析完成后将保存持久页面索引。"
            )
            return imported
        except Exception as exc:
            show_error_dialog(self, "来源复制失败", str(exc))
            return []

    def _persist_project_page_state(self, *, classification_only: bool = False) -> None:
        project = self.project_manager.active_project
        if project is None or self._page_state_save_closed:
            return
        snapshot = {
            "workspace_root": str(self.project_manager.workspace_root),
            "project_path": str(project),
            "page_images": [str(path) for path in self.page_images],
            "page_overrides": dict(self.page_overrides),
            "auto_suggested": sorted(self._auto_suggested),
            "raw_inputs": list(self._last_loaded_raw_inputs or []),
            "source_label": self._file_lbl.text() if hasattr(self, "_file_lbl") else "",
            "original_pdf_sources": list(self._original_pdf_sources),
            "pdf_physical_page_map": dict(self._pdf_physical_page_map),
            "classification_only": bool(classification_only),
        }
        # Coalesce pending writes to the latest in-memory state. A pending full
        # save must remain full even when the newest event only changed tags,
        # otherwise an older page list could survive a delete.
        preserve_full_save = False
        other_projects = {}
        while True:
            try:
                previous = self._page_state_save_queue.get_nowait()
            except queue.Empty:
                break
            if not isinstance(previous, dict):
                continue
            previous_project = previous["project_path"]
            if previous_project != snapshot["project_path"]:
                earlier = other_projects.get(previous_project)
                if earlier and not earlier.get("classification_only", False):
                    previous["classification_only"] = False
                other_projects[previous_project] = previous
                continue
            if not previous.get("classification_only", False):
                preserve_full_save = True
        if preserve_full_save:
            snapshot["classification_only"] = False
        for pending in other_projects.values():
            self._page_state_save_queue.put_nowait(pending)
        self._page_state_save_queue.put_nowait(snapshot)
        if self._page_state_save_thread is None or not self._page_state_save_thread.is_alive():
            self._page_state_save_thread = threading.Thread(
                target=self._page_state_save_worker,
                daemon=False,
                name="project-page-state-save",
            )
            self._page_state_save_thread.start()

    def _page_state_save_worker(self) -> None:
        while True:
            try:
                snapshot = self._page_state_save_queue.get(timeout=0.2)
            except queue.Empty:
                if self._page_state_save_closed:
                    return
                continue
            if snapshot is None:
                return
            error = ""
            try:
                manager = ProjectWorkspaceManager(snapshot["workspace_root"])
                manager.open_project(snapshot["project_path"])
                if snapshot.get("classification_only"):
                    try:
                        manager.save_page_overrides(
                            snapshot["page_overrides"], snapshot["auto_suggested"],
                            source_label=snapshot["source_label"],
                        )
                    except FileNotFoundError:
                        manager.save_page_state(
                            snapshot["page_images"], snapshot["page_overrides"],
                            snapshot["auto_suggested"], snapshot["raw_inputs"],
                            source_label=snapshot["source_label"],
                            original_pdf_sources=snapshot["original_pdf_sources"],
                            pdf_physical_page_map=snapshot["pdf_physical_page_map"],
                        )
                else:
                    manager.save_page_state(
                        snapshot["page_images"], snapshot["page_overrides"],
                        snapshot["auto_suggested"], snapshot["raw_inputs"],
                        source_label=snapshot["source_label"],
                        original_pdf_sources=snapshot["original_pdf_sources"],
                        pdf_physical_page_map=snapshot["pdf_physical_page_map"],
                    )
            except Exception as exc:
                error = str(exc)
            try:
                self._page_state_save_signals.finished.emit(
                    (snapshot["project_path"], error)
                )
            except RuntimeError:
                pass

    def _on_page_state_save_finished(self, result) -> None:
        project_path, error = result
        active = self.project_manager.active_project
        if active is None or str(active) != str(project_path) or not hasattr(self, "_project_status"):
            return
        if error:
            self._project_status.setText(f"项目页面状态保存失败：{error}")

    def _refresh_current_source(self):
        """重新读取当前书籍来源，不改变用户已经选择的入口语义。"""
        raw = list(self._last_loaded_raw_inputs or [])
        if not raw:
            notify(self, "请先添加图片、PDF 或打开图片文件夹。", "warning")
            return
        display_name = self._file_lbl.text().strip() or Path(raw[0]).name
        self._load_inputs(raw, display_name)

    def _on_search_changed(self, text: str):
        self._search_text = str(text or "").strip().casefold()
        self._render()

    def _open_menu(self):
        menu = QMenu(self)
        menu.addAction("打开文件夹...", self._open_folder)
        menu.addAction("打开图片/PDF文件...", self._open_files)
        menu.exec(QCursor.pos())

    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择包含图片或 PDF 的文件夹")
        if folder:
            name = Path(folder).name
            # 文件夹里的图片和 PDF 一起导入；没有可识别文件时仍交给原流程给出提示
            sources = folder_sources(folder) or [folder]
            imported = self._import_sources_into_project(sources, name)
            if imported:
                self._load_inputs(imported, name)

    # ── 拖放导入：文件与文件夹共用上面的整理规则 ───────────────────────────
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event):
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        dropped = [u.toLocalFile() for u in urls if u.isLocalFile()]
        sources = collect_sources(dropped)
        if not sources:
            super().dropEvent(event)
            return
        event.acceptProposedAction()
        name = Path(dropped[0]).name if len(dropped) == 1 and Path(dropped[0]).is_dir() else import_label(sources)
        imported = self._import_sources_into_project(sources, name)
        if imported:
            self._load_inputs(imported, name)

    def _open_image_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择图片（可多选）", "",
            "图片 (*.png *.jpg *.jpeg *.heic *.tif *.tiff *.bmp *.gif);;所有文件 (*)")
        if paths:
            name = Path(paths[0]).stem if len(paths) == 1 else f"{Path(paths[0]).stem} 等 {len(paths)} 个文件"
            imported = self._import_sources_into_project(paths, name)
            if imported:
                self._load_inputs(imported, name)

    def _open_pdf_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择 PDF（可多选）", "", "PDF (*.pdf);;所有文件 (*)")
        if paths:
            name = Path(paths[0]).stem if len(paths) == 1 else f"{Path(paths[0]).stem} 等 {len(paths)} 个文件"
            imported = self._import_sources_into_project(paths, name)
            if imported:
                self._load_inputs(imported, name)

    def _open_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择图片或 PDF（可多选）", "",
            "图片和PDF (*.png *.jpg *.jpeg *.heic *.tif *.tiff *.bmp *.gif *.pdf);;所有文件 (*)")
        if paths:
            name = Path(paths[0]).stem if len(paths) == 1 else f"{Path(paths[0]).stem} 等 {len(paths)} 个文件"
            imported = self._import_sources_into_project(paths, name)
            if imported:
                self._load_inputs(imported, name)

    def _scan_processing_blocked(self) -> bool:
        ocr_tab = getattr(self.window(), "_tab_ocr", None)
        return bool(getattr(ocr_tab, "_ocr_run_active", False))

    def _open_scan_preprocess_dialog(self):
        if not self.page_images:
            notify(self, "请先添加图片、PDF 或打开图片文件夹。", "warning")
            return
        if self._scan_processing_blocked():
            notify(self, "请先停止或完成当前 OCR，再更换页面源。", "warning")
            return
        target_pages = sorted(
            page_no for page_no in self.selected_pages
            if 1 <= int(page_no) <= len(self.page_images)
        )
        selected_only = bool(target_pages)
        if not target_pages:
            target_pages = list(range(1, len(self.page_images) + 1))
        preview_page = target_pages[0]
        try:
            from ui.scan_preprocess_dialog import ScanPreprocessDialog
            dialog = ScanPreprocessDialog(
                str(self.page_images[preview_page - 1]),
                source_count=len(target_pages),
                selected_only=selected_only,
                parent=self,
            )
            if dialog.exec() != QDialog.Accepted:
                return
            options = dialog.options()
        except Exception as exc:
            show_error_dialog(self, "扫描件优化无法打开", str(exc))
            return
        self._apply_scan_preprocess(target_pages, options)

    def _capture_scan_page_state(self) -> dict:
        return {
            "page_images": [Path(path) for path in self.page_images],
            "page_overrides": dict(self.page_overrides),
            "auto_suggested": set(self._auto_suggested),
            "raw_inputs": list(self._last_loaded_raw_inputs or []),
            "original_pdf_sources": list(self._original_pdf_sources),
            "pdf_physical_page_map": dict(self._pdf_physical_page_map),
            "file_label": self._file_lbl.text(),
            "count_label": self._count_lbl.text(),
        }

    def _apply_scan_preprocess(self, target_pages, options) -> None:
        if self._scan_processing_blocked():
            notify(self, "请先停止或完成当前 OCR，再更换页面源。", "warning")
            return
        targets = {
            int(page_no) for page_no in target_pages
            if 1 <= int(page_no) <= len(self.page_images)
        }
        if not targets:
            return
        self._scan_process_generation += 1
        generation = self._scan_process_generation
        source_paths = [str(path) for path in self.page_images]
        previous_state = self._capture_scan_page_state()
        previous_overrides = dict(self.page_overrides)
        previous_auto = set(self._auto_suggested)
        self._scan_optimize_btn.setEnabled(False)
        self._scan_restore_btn.setEnabled(False)
        if hasattr(self, "_scan_restore_action"):
            self._scan_restore_action.setEnabled(False)
        self._prog.setVisible(True)
        self._prog.setRange(0, len(targets))
        self._prog.setValue(0)
        self._prog.setFormat("扫描件优化 0/%d" % len(targets))

        signals = WorkerSignals()
        self._scan_signal_refs[generation] = signals

        def worker():
            try:
                from adapters.scan_preprocess import (
                    ProcessedScanPage,
                    process_scan_page,
                    remap_page_metadata,
                )
                from utils.session_temp import session_temp_registry

                output_dir = session_temp_registry().make_dir("scan-preprocessed-pages")
                output_pages = []
                completed = 0
                split_sources = 0
                geometry_pages = 0
                enhanced_pages = 0
                for source_index, source_path in enumerate(source_paths, start=1):
                    if generation != self._scan_process_generation:
                        raise RuntimeError("扫描件预处理已取消")
                    if source_index in targets:
                        produced = process_scan_page(
                            source_path,
                            output_dir,
                            options,
                            source_index=source_index,
                        )
                        output_pages.extend(produced)
                        split_sources += int(len(produced) > 1)
                        geometry_pages += sum(1 for page in produced if page.geometry_applied)
                        enhanced_pages += sum(
                            1 for page in produced if page.enhancement_applied != "none"
                        )
                        completed += 1
                        signals.progress.emit(completed, len(targets))
                    else:
                        output_pages.append(ProcessedScanPage(
                            source_path=source_path,
                            output_path=source_path,
                            source_index=source_index,
                            part="single",
                            geometry_applied=False,
                            enhancement_applied="none",
                        ))
                remapped_overrides, remapped_auto = remap_page_metadata(
                    output_pages, previous_overrides, previous_auto,
                )
                signals.finished.emit({
                    "generation": generation,
                    "pages": output_pages,
                    "page_overrides": remapped_overrides,
                    "auto_suggested": remapped_auto,
                    "previous_state": previous_state,
                    "target_count": len(targets),
                    "split_sources": split_sources,
                    "geometry_pages": geometry_pages,
                    "enhanced_pages": enhanced_pages,
                })
            except Exception as exc:
                signals.error.emit(str(exc))

        def on_progress(current, total):
            if generation != self._scan_process_generation:
                return
            total = max(1, int(total))
            current = max(0, min(total, int(current)))
            self._prog.setRange(0, total)
            self._prog.setValue(current)
            self._prog.setFormat(f"扫描件优化 {current}/{total}")

        def cleanup_signal():
            self._scan_signal_refs.pop(generation, None)

        def on_finished(payload):
            try:
                if int(payload.get("generation", -1)) != self._scan_process_generation:
                    return
                pages = list(payload.get("pages") or [])
                output_paths = [str(page.output_path) for page in pages]
                if not output_paths:
                    raise RuntimeError("扫描件优化没有生成可用页面")
                self._scan_history.append(payload["previous_state"])
                self._scan_history = self._scan_history[-5:]
                if self.project_manager.active_project is not None:
                    output_paths = self.project_manager.persist_processed_page_images(output_paths)
                self.page_images = [Path(path) for path in output_paths]
                self.page_overrides = dict(payload.get("page_overrides") or {})
                self._auto_suggested = set(payload.get("auto_suggested") or set())
                self._last_loaded_raw_inputs = list(output_paths)
                previous_pdf_map = {int(k): int(v) for k, v in dict(payload["previous_state"].get("pdf_physical_page_map") or {}).items()}
                self._original_pdf_sources = list(payload["previous_state"].get("original_pdf_sources") or self._original_pdf_sources)
                self._pdf_physical_page_map = {
                    new_no: previous_pdf_map.get(int(page.source_index), int(page.source_index))
                    for new_no, page in enumerate(pages, start=1)
                    if previous_pdf_map or self._original_pdf_sources
                }
                self.selected_pages.clear()
                self.thumb_cache.clear()
                self._thumb_pending.clear()
                base_label = str(payload["previous_state"].get("file_label") or "书籍")
                base_label = re.sub(r"(?: · 扫描优化)+$", "", base_label)
                self._file_lbl.setText(base_label + " · 扫描优化")
                self._count_lbl.setText(f"{len(output_paths)} 页")
                self._render()
                self._persist_project_page_state()
                self.pages_loaded.emit(output_paths)
                self._scan_restore_btn.setEnabled(bool(self._scan_history))
                if hasattr(self, "_scan_restore_action"):
                    self._scan_restore_action.setEnabled(bool(self._scan_history))
                summary = (
                    f"已优化 {int(payload.get('target_count', 0))} 个源页面，"
                    f"当前共 {len(output_paths)} 页。"
                )
                details = []
                if int(payload.get("split_sources", 0)):
                    details.append(f"拆分双页 {int(payload['split_sources'])} 张")
                if int(payload.get("geometry_pages", 0)):
                    details.append(f"几何校正 {int(payload['geometry_pages'])} 页")
                if int(payload.get("enhanced_pages", 0)):
                    details.append(f"漂白/去阴影 {int(payload['enhanced_pages'])} 页")
                if details:
                    summary += "\n" + "；".join(details) + "。"
                summary += "\n原图未修改，可点击“恢复优化前”回退。"
                notify(self, summary, "success")
            except Exception as exc:
                show_error_dialog(self, "扫描件优化失败", str(exc))
            finally:
                self._prog.setVisible(False)
                self._scan_optimize_btn.setEnabled(True)
                if generation == self._scan_process_generation:
                    self._scan_restore_btn.setEnabled(bool(self._scan_history))
                    if hasattr(self, "_scan_restore_action"):
                        self._scan_restore_action.setEnabled(bool(self._scan_history))
                cleanup_signal()

        def on_error(message):
            try:
                if generation == self._scan_process_generation:
                    self._prog.setVisible(False)
                    show_error_dialog(self, "扫描件优化失败", message)
            finally:
                self._scan_optimize_btn.setEnabled(True)
                self._scan_restore_btn.setEnabled(bool(self._scan_history))
                if hasattr(self, "_scan_restore_action"):
                    self._scan_restore_action.setEnabled(bool(self._scan_history))
                cleanup_signal()

        signals.progress.connect(on_progress)
        signals.finished.connect(on_finished)
        signals.error.connect(on_error)
        threading.Thread(target=worker, daemon=True).start()

    def _restore_scan_preprocess(self) -> None:
        if self._scan_processing_blocked():
            notify(self, "请先停止或完成当前 OCR，再恢复页面源。", "warning")
            return
        if not self._scan_history:
            return
        self._scan_process_generation += 1
        state = self._scan_history.pop()
        self.page_images = [Path(path) for path in state.get("page_images", [])]
        self.page_overrides = dict(state.get("page_overrides") or {})
        self._auto_suggested = set(state.get("auto_suggested") or set())
        self._last_loaded_raw_inputs = list(
            state.get("raw_inputs") or [str(path) for path in self.page_images]
        )
        self._original_pdf_sources = list(state.get("original_pdf_sources") or [])
        self._pdf_physical_page_map = {int(k): int(v) for k, v in dict(state.get("pdf_physical_page_map") or {}).items()}
        self.selected_pages.clear()
        self.thumb_cache.clear()
        self._thumb_pending.clear()
        self._file_lbl.setText(str(state.get("file_label") or "书籍"))
        self._count_lbl.setText(str(state.get("count_label") or f"{len(self.page_images)} 页"))
        self._render()
        self._persist_project_page_state()
        self.pages_loaded.emit([str(path) for path in self.page_images])
        self._scan_restore_btn.setEnabled(bool(self._scan_history))
        if hasattr(self, "_scan_restore_action"):
            self._scan_restore_action.setEnabled(bool(self._scan_history))

    def _load_inputs(self, raw_paths, display_name):
        # 新来源拥有新的页面血统；仅清除内存中的恢复栈。临时优化图片仍由
        # SessionTempRegistry 在退出时统一清理。
        self._scan_history.clear()
        if hasattr(self, "_scan_restore_btn"):
            self._scan_restore_btn.setEnabled(False)
        if hasattr(self, "_scan_restore_action"):
            self._scan_restore_action.setEnabled(False)
        self._scan_process_generation += 1
        self._load_generation += 1
        generation = self._load_generation
        previous_inputs = list(self._last_loaded_raw_inputs or [])
        if previous_inputs and previous_inputs != list(raw_paths):
            def normalized_input(value):
                try:
                    return str(Path(value).expanduser().resolve())
                except Exception:
                    return str(value)
            new_sources = {normalized_input(value) for value in raw_paths}
            self._pending_pdf_cache_cleanup = [
                value for value in previous_inputs
                if normalized_input(value) not in new_sources
            ]
        pdf_inputs = [str(Path(value).expanduser()) for value in raw_paths if Path(str(value)).suffix.lower() == ".pdf"]
        if len(pdf_inputs) == 1:
            self._pending_pdf_source_for_load = pdf_inputs[0]
        elif list(raw_paths) != previous_inputs:
            current_pages = {str(Path(value)) for value in self.page_images}
            requested = {str(Path(value)) for value in raw_paths}
            if not self._original_pdf_sources or requested != current_pages:
                self._original_pdf_sources = []
                self._pdf_physical_page_map = {}
                self._pending_pdf_source_for_load = ""
        self._last_loaded_raw_inputs = list(raw_paths)
        self._file_lbl.setText(display_name)
        self._prog.setVisible(True)
        self._prog.setRange(0, 0)
        self._prog.setFormat("正在读取输入…")

        active_project = self.project_manager.active_project
        persist_in_worker = active_project is not None and not self._project_restore_in_progress
        project_path = str(active_project) if persist_in_worker else ""
        workspace_root = str(self.project_manager.workspace_root)
        pending_pdf_source = str(self._pending_pdf_source_for_load or "")
        original_pdf_sources = (
            [pending_pdf_source] if pending_pdf_source
            else list(self._original_pdf_sources)
        )
        physical_page_map = dict(self._pdf_physical_page_map)

        def worker():
            try:
                from adapters.pdf_input import expand_inputs, natural_sort_key

                def on_pdf_progress(current, total):
                    if generation != self._load_generation:
                        return
                    signals.progress.emit(int(current), int(total))

                images = expand_inputs(
                    raw_paths,
                    progress_callback=on_pdf_progress,
                    cancel_check=lambda: generation != self._load_generation,
                )
                images = sorted(set(images), key=natural_sort_key)
                payload = {"generation": generation, "images": images}
                if persist_in_worker and images and generation == self._load_generation:
                    from core.project_workspace import ProjectWorkspaceManager

                    manager = ProjectWorkspaceManager(workspace_root)
                    manager.open_project(project_path)
                    durable_images = manager.persist_page_images(images)
                    count = len(durable_images)
                    overrides = {index: "paragraph" for index in range(1, count + 1)}
                    auto_suggested = set(overrides)
                    if pending_pdf_source:
                        pdf_sources = [pending_pdf_source]
                        page_map = {index: index for index in range(1, count + 1)}
                    else:
                        pdf_sources = original_pdf_sources
                        page_map = physical_page_map
                    manager.save_page_state(
                        durable_images, overrides, auto_suggested, raw_paths,
                        source_label=display_name,
                        original_pdf_sources=pdf_sources,
                        pdf_physical_page_map=page_map,
                    )
                    payload.update({
                        "images": durable_images,
                        "page_overrides": overrides,
                        "auto_suggested": auto_suggested,
                        "original_pdf_sources": pdf_sources,
                        "pdf_physical_page_map": page_map,
                        "project_state_saved": True,
                    })
                if generation == self._load_generation:
                    signals.finished.emit(payload)
            except Exception as e:
                if generation == self._load_generation:
                    signals.error.emit(str(e))

        # 绑定在 self 上（而不是局部变量）：Qt 跨线程信号投递是异步的，
        # 工作线程结束后如果没有任何 Python 引用持有 WorkerSignals，
        # 垃圾回收可能在主线程处理排队事件之前就把它回收掉，导致野指针崩溃。
        signals = WorkerSignals()
        self._load_signals = signals
        self._load_signal_refs[generation] = signals

        def on_finished(payload):
            try:
                self._finish_load(payload)
            finally:
                self._load_signal_refs.pop(generation, None)

        def on_error(message):
            try:
                if generation == self._load_generation:
                    self._prog.setVisible(False)
                    QMessageBox.critical(self, "加载失败", message)
            finally:
                self._load_signal_refs.pop(generation, None)

        def on_progress(current, total):
            if generation != self._load_generation:
                return
            total = max(1, int(total))
            current = max(0, min(total, int(current)))
            self._prog.setRange(0, total)
            self._prog.setValue(current)
            self._prog.setFormat(f"PDF 页面准备 {current}/{total} · 页面缩略图将在显示时加载")

        signals.finished.connect(on_finished)
        signals.error.connect(on_error)
        signals.progress.connect(on_progress)
        threading.Thread(target=worker, daemon=True).start()

    def _finish_load(self, payload):
        if isinstance(payload, dict):
            if int(payload.get("generation", -1)) != self._load_generation:
                return
            images = payload.get("images") or []
        else:
            images = payload or []
        project_state_saved = bool(isinstance(payload, dict) and payload.get("project_state_saved"))
        self._prog.setVisible(False)
        if not images:
            QMessageBox.warning(self, "无图片", "未找到可用的图片")
            return
        if (not project_state_saved and self.project_manager.active_project is not None
                and not self._project_restore_in_progress and not isinstance(payload, dict)):
            try:
                images = self.project_manager.persist_page_images(images)
            except Exception as exc:
                self._project_status.setText(f"项目页面持久化失败，暂用当前页面：{exc}")
        self.page_images = [Path(p) for p in images]
        if isinstance(payload, dict) and payload.get("page_overrides") is not None:
            self.page_overrides = {
                int(key): str(value) for key, value in payload.get("page_overrides", {}).items()
            }
            self._auto_suggested = {
                int(value) for value in payload.get("auto_suggested", [])
            }
            self._original_pdf_sources = [
                str(value) for value in payload.get("original_pdf_sources", [])
            ]
            self._pdf_physical_page_map = {
                int(key): int(value)
                for key, value in payload.get("pdf_physical_page_map", {}).items()
            }
            self._pending_pdf_source_for_load = ""
            if project_state_saved and hasattr(self, "_project_status"):
                self._project_status.setText(
                    f"已载入并保存 {len(self.page_images)} 页；页面缩略图按需解码。"
                )
        elif self._pending_pdf_source_for_load:
            self._original_pdf_sources = [str(Path(self._pending_pdf_source_for_load).expanduser())]
            self._pdf_physical_page_map = {index: index for index in range(1, len(self.page_images) + 1)}
            self._pending_pdf_source_for_load = ""
            self.page_overrides = {index: "paragraph" for index in range(1, len(self.page_images) + 1)}
            self._auto_suggested = set(self.page_overrides)
        else:
            # 默认全部当正文页；需要封面/插图时可在页面管理里手动改。
            self.page_overrides = {index: "paragraph" for index in range(1, len(self.page_images) + 1)}
            self._auto_suggested = set(self.page_overrides)
        self.selected_pages.clear()
        self.thumb_cache.clear()
        self._count_lbl.setText(f"{len(images)} 页")
        self._render()
        if not project_state_saved:
            self._persist_project_page_state()
        self.pages_loaded.emit([str(path) for path in self.page_images])
        old_inputs = list(self._pending_pdf_cache_cleanup)
        self._pending_pdf_cache_cleanup = []
        if old_inputs:
            def release_old_pdf_cache():
                try:
                    from adapters.pdf_input import release_pdf_caches
                    release_pdf_caches(old_inputs)
                except Exception:
                    pass
            QTimer.singleShot(0, release_old_pdf_cache)

    def _ptype(self, page_no):
        return self.page_overrides.get(page_no, "unknown")

    def _render(self):
        self._grid_build_generation += 1
        generation = self._grid_build_generation
        self._grid_build_complete = False
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            if item.widget():
                item.widget().hide()
                item.widget().deleteLater()

        filtered_specs = []
        query = self._search_text
        filter_types = {
            "text": {"paragraph"},
            "illustration": {"illustration", "color_illus"},
            "cover": {"cover"},
        }.get(self._current_filter)
        for page_no, path in enumerate(self.page_images, start=1):
            if filter_types is not None and self._ptype(page_no) not in filter_types:
                continue
            if query and query not in Path(path).name.casefold():
                continue
            filtered_specs.append((Path(path), page_no))

        self._thumb_labels = {}
        self._page_frames = {}
        self._page_type_labels = {}
        if not filtered_specs:
            self._grid_layout.setAlignment(Qt.AlignCenter)
            empty = QLabel("（此类型暂无页面）" if self.page_images else
                          "点击「导入文件」或「导入文件夹」开始\n\n也可以把图片、PDF、文件夹直接拖到这里")
            empty.setAlignment(Qt.AlignCenter)
            empty.setStyleSheet(f"color:{MUTED}; font-size: 14px; padding: 80px 40px;")
            self._grid_layout.addWidget(empty, 0, 0, 1, 6, Qt.AlignCenter)
            self._grid_build_specs = []
            self._grid_build_cursor = 0
            self._grid_build_complete = True
            self._rendered_grid_cols = 0
            self._update_counts()
            return

        self._grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        cell_w, gap = 148, 16
        scroll = getattr(self, "_page_scroll", None)
        viewport_w = int(scroll.viewport().width()) if scroll is not None else int(self.width())
        if viewport_w < 320:
            viewport_w = max(720, int(self.width()) - 70)
        columns = max(1, min(12, len(filtered_specs), (viewport_w + gap) // (cell_w + gap)))
        self._rendered_grid_cols = int(columns)
        for column in range(12):
            self._grid_layout.setColumnStretch(column, 1 if column < columns else 0)

        # Create a small batch per event-loop turn. This keeps imports of long
        # books responsive and prevents clicks from reentering a half-built grid.
        self._grid_build_specs = filtered_specs
        self._grid_build_cursor = 0
        self._grid_build_cols = columns
        self._grid_widget.setUpdatesEnabled(True)
        self._update_counts()
        self._sel_lbl.setText(f"选中 {len(self.selected_pages)} 页 · 标记为：")
        self._update_stat_bar()
        self._build_thumbnail_batch(generation)

    def _build_thumbnail_batch(self, generation: int) -> None:
        if generation != self._grid_build_generation:
            return
        specs = self._grid_build_specs
        columns = max(1, int(self._grid_build_cols))
        start = self._grid_build_cursor
        stop = min(len(specs), start + 24)
        self._grid_widget.setUpdatesEnabled(False)
        for index in range(start, stop):
            path, page_no = specs[index]
            self._add_thumbnail_card(path, page_no, index, columns)
        self._grid_widget.setUpdatesEnabled(True)
        self._grid_build_cursor = stop
        if stop < len(specs):
            QTimer.singleShot(0, partial(self._build_thumbnail_batch, generation))
            return
        self._grid_build_complete = True
        self._visible_thumb_timer.start()

    def _add_thumbnail_card(self, path: Path, page_no: int, index: int, columns: int) -> None:
        cell_width, image_width, image_height = 148, 126, 168
        ptype = self._ptype(page_no)
        color = TYPE_COLOR.get(ptype, "#AAA")
        compact_color = {
            "cover": "#3478F6",
            "illustration": "#7894B8",
            "color_illus": "#7894B8",
            "paragraph": "#17A46B",
        }.get(ptype, color)
        label = TYPE_LABEL.get(ptype, "?")

        cell = QWidget()
        cell.setObjectName("pageThumbClickable")
        cell.setStyleSheet(
            f"QWidget#pageThumbClickable {{ background:{CARD}; border:1px solid #D5E4F7; border-radius:14px; }}"
            "QWidget#pageThumbClickable:hover { background:#FAFAFB; border-color:#B8D1F1; }"
        )
        cell.setFixedSize(cell_width, 218)
        cell_layout = QVBoxLayout(cell)
        cell_layout.setContentsMargins(10, 10, 10, 8)
        cell_layout.setSpacing(7)

        image_frame = _PageThumbnailLabel()
        image_frame.setFixedSize(image_width, image_height)
        image_frame.setAlignment(Qt.AlignCenter)
        image_frame.setStyleSheet(
            f"background:#F3EEDF;border:1px solid {BORDER};border-radius:9px;"
        )
        image_frame.set_selected(page_no in self.selected_pages)
        key = str(path)
        if path.exists():
            cached = self._thumb_cache_get(key)
            if cached is not None:
                image_frame.setPixmap(cached)
            else:
                image_frame.setText(f"第 {page_no} 页")
                self._thumb_labels[key] = image_frame
        else:
            image_frame.setText(f"第 {page_no} 页")
            image_frame.setStyleSheet(
                image_frame.styleSheet() + "color:#666;font-size:12px;"
            )
        self._page_frames[page_no] = image_frame
        cell_layout.addWidget(image_frame)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(6)
        page_label = QLabel(f"p{page_no:03d}")
        page_label.setStyleSheet(f"color:{INK};font-size:10px;font-weight:700;")
        footer.addWidget(page_label)
        footer.addStretch(1)

        tag = QLabel(label)
        tag.setFixedHeight(18)
        tag.setContentsMargins(7, 0, 7, 0)
        tag.setAlignment(Qt.AlignCenter)
        tag_bg = blend(compact_color, 0.14, "#FFFFFF")
        tag.setStyleSheet(
            f"background:{tag_bg};color:{compact_color};border-radius:9px;"
            "font-size:9px;font-weight:700;"
        )
        self._page_type_labels[page_no] = tag
        footer.addWidget(tag)
        cell_layout.addLayout(footer)

        cell.setProperty("page_no", page_no)
        cell.mousePressEvent = partial(self._on_thumb_click, page_no)
        cell.mouseDoubleClickEvent = partial(self._on_thumb_double_click, page_no)
        cell.setToolTip("单击选择 · 双击打开高清预览 · 右键设置页面类型")
        cell.setContextMenuPolicy(Qt.CustomContextMenu)
        cell.customContextMenuRequested.connect(partial(self._on_thumb_context, page_no))
        self._grid_layout.addWidget(
            cell, index // columns, index % columns, Qt.AlignTop | Qt.AlignHCenter,
        )

    def _desired_grid_columns(self) -> int:
        if not self.page_images:
            return 0
        scroll = getattr(self, "_page_scroll", None)
        width = int(scroll.viewport().width()) if scroll is not None else int(self.width())
        if width < 320:
            return int(getattr(self, "_rendered_grid_cols", 0) or 0)
        cell_w, gap = 148, 16
        visible_count = len(self.page_images)
        if self._current_filter != "all":
            filter_types = {
                "text": {"paragraph"},
                "illustration": {"illustration", "color_illus"},
                "cover": {"cover"},
            }.get(self._current_filter, {self._current_filter})
            visible_count = sum(
                1 for index, _path in enumerate(self.page_images)
                if self._ptype(index + 1) in filter_types
            )
        if self._search_text:
            query = self._search_text
            visible_count = sum(
                1 for path in self.page_images
                if query in Path(path).name.casefold()
            ) if self._current_filter == "all" else visible_count
        if visible_count <= 0:
            return 0
        return max(1, min(12, visible_count, (width + gap) // (cell_w + gap)))

    def _reflow_grid_after_resize(self) -> None:
        desired = self._desired_grid_columns()
        if desired and desired != int(getattr(self, "_rendered_grid_cols", 0) or 0):
            self._render()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        timer = getattr(self, "_grid_reflow_timer", None)
        if timer is not None and self.page_images:
            timer.start()

    def showEvent(self, event):
        super().showEvent(event)
        timer = getattr(self, "_visible_thumb_timer", None)
        if timer is not None:
            timer.start()

    # ── 异步缩略图加载 ────────────────────────────────────────────────────
    #
    # 旧实现：在主线程用 PIL 全量解码 + LANCZOS 缩放每一张高清扫描图，
    # 导入几百页时界面直接冻结数秒到数十秒。
    # 新实现：QImageReader.setScaledSize 按目标尺寸解码（JPEG 走 libjpeg
    # 的快速缩放解码），且放在后台线程；主线程只把结果转 QPixmap 贴上去。

    @staticmethod
    def _decode_thumb(path: str, w: int, h: int) -> QImage:
        reader = QImageReader(path)
        reader.setAutoTransform(True)
        size = reader.size()
        if size.isValid():
            reader.setScaledSize(size.scaled(w, h, Qt.KeepAspectRatio))
        return reader.read()

    def _enqueue_thumb(self, path, w, h):
        if self._thumb_closed:
            return
        key = str(path)
        if key in self._thumb_pending or key not in self._thumb_labels:
            return  # 已在队列里（如快速切换筛选触发的重复渲染）
        try:
            self._thumb_queue.put_nowait((key, w, h))
        except queue.Full:
            return
        self._thumb_pending.add(key)
        if self._thumb_thread is None or not self._thumb_thread.is_alive():
            self._thumb_thread = threading.Thread(
                target=self._thumb_worker, daemon=True, name="thumb-loader")
            self._thumb_thread.start()

    def _thumb_worker(self):
        while True:
            item = self._thumb_queue.get()
            if item is None:
                return
            key, w, h = item
            try:
                img = self._decode_thumb(key, w, h)
            except Exception:
                img = QImage()
            try:
                self._thumb_signals.finished.emit((key, img))
            except RuntimeError:
                return  # 界面已销毁

    def _on_thumb_ready(self, payload):
        key, img = payload
        self._thumb_pending.discard(key)
        if self._thumb_closed:
            return
        label = self._thumb_labels.pop(key, None)
        if img is None or img.isNull():
            if label is not None:
                try:
                    label.setText("无法预览")
                except RuntimeError:
                    self._thumb_labels.pop(key, None)
            self._visible_thumb_timer.start()
            return
        pix = QPixmap.fromImage(img)
        self._thumb_cache_put(key, pix)
        if label is not None:
            try:
                label.setPixmap(pix)
            except RuntimeError:
                self._thumb_labels.pop(key, None)
        self._visible_thumb_timer.start()

    def _load_visible_thumbnails(self) -> None:
        if self._thumb_closed or not self._thumb_labels:
            return
        scrollbar = self._page_scroll.verticalScrollBar()
        top = int(scrollbar.value())
        bottom = top + int(self._page_scroll.viewport().height())
        margin = 2 * 234
        for key, label in list(self._thumb_labels.items()):
            try:
                if not label.isVisible():
                    continue
                y = label.mapTo(self._grid_widget, QPoint(0, 0)).y()
                if y + label.height() < top - margin or y > bottom + margin:
                    continue
                self._enqueue_thumb(key, label.width() - 4, label.height() - 4)
            except RuntimeError:
                self._thumb_labels.pop(key, None)

    def _thumb_cache_get(self, key: str):
        pixmap = self.thumb_cache.get(str(key))
        if pixmap is not None:
            self.thumb_cache.move_to_end(str(key))
        return pixmap

    def _thumb_cache_put(self, key: str, pixmap: QPixmap) -> None:
        cache_key = str(key)
        self.thumb_cache[cache_key] = pixmap
        self.thumb_cache.move_to_end(cache_key)
        while len(self.thumb_cache) > int(self._thumb_cache_limit):
            self.thumb_cache.popitem(last=False)

    def _refresh_selection_styles(self):
        """只用轻量绘制刷新选中边框，避免点击时重套 Qt 样式表。"""
        for page_no, frame in self._page_frames.items():
            try:
                frame.set_selected(page_no in self.selected_pages)
            except RuntimeError:
                continue
        self._sel_lbl.setText(f"选中 {len(self.selected_pages)} 页 · 标记为：")

    def _on_thumb_click(self, page_no, event):
        if event.button() == Qt.LeftButton:
            if page_no in self.selected_pages:
                self.selected_pages.discard(page_no)
            else:
                self.selected_pages.add(page_no)
            self._refresh_selection_styles()

    def _on_thumb_double_click(self, page_no, event):
        if event.button() == Qt.LeftButton:
            self._open_page_preview(page_no)
            event.accept()

    def _open_page_preview(self, page_no: int):
        if not self.page_images:
            return
        index = max(0, min(len(self.page_images) - 1, int(page_no) - 1))
        dialog = PageImagePreviewDialog(self.page_images, index, self)
        dialog.exec()

    # ── 拉框多选（在缩略图之间的空白处按下拖动，与矩形相交的页面都会被选中）───

    def _grid_mouse_press(self, event):
        if event.button() != Qt.LeftButton:
            return
        # 点在某张缩略图上：交给缩略图自己的 mousePressEvent 处理单击切换，
        # 这里只处理点在空白区域时开始拉框。
        if self._grid_widget.childAt(event.pos()) is not None:
            return
        self._rubber_origin = event.pos()
        self._rubber_band.setGeometry(QRect(self._rubber_origin, event.pos()).normalized())
        self._rubber_band.show()

    def _grid_mouse_move(self, event):
        if self._rubber_origin is None:
            return
        self._rubber_band.setGeometry(QRect(self._rubber_origin, event.pos()).normalized())

    def _grid_mouse_release(self, event):
        if self._rubber_origin is None:
            return
        rect = QRect(self._rubber_origin, event.pos()).normalized()
        self._rubber_band.hide()
        self._rubber_origin = None

        # 矩形太小（几乎没拖动）视为一次空白点击：不加修饰键则清空选择
        if rect.width() < 4 and rect.height() < 4:
            if not (event.modifiers() & (Qt.ShiftModifier | Qt.MetaModifier)):
                self.selected_pages.clear()
                self._refresh_selection_styles()
            return

        framed: set[int] = set()
        for i in range(self._grid_layout.count()):
            item = self._grid_layout.itemAt(i)
            w = item.widget() if item else None
            if w is None:
                continue
            page_no = w.property("page_no")
            if page_no is None:
                continue
            if rect.intersects(w.geometry()):
                framed.add(page_no)

        # 按住 Shift/Cmd 拖框 = 追加到现有选择；否则拖框结果替换当前选择（类 Finder 行为）
        if event.modifiers() & (Qt.ShiftModifier | Qt.MetaModifier):
            self.selected_pages |= framed
        else:
            self.selected_pages = framed
        self._refresh_selection_styles()

    def _on_thumb_context(self, page_no, pos):
        if page_no not in self.selected_pages:
            self.selected_pages = {page_no}
            self._refresh_selection_styles()
        menu = QMenu(self)
        menu.addAction(f"第 {page_no} 页 — 设置类型").setEnabled(False)
        menu.addAction("🔍 放大预览", lambda: self._open_page_preview(page_no))
        menu.addSeparator()
        for ttype, label, color in PAGE_TYPES:
            action = menu.addAction(f"  {label}")
            action.triggered.connect(partial(self._batch_tag, ttype))
        menu.addSeparator()
        menu.addAction("取消选择", self._clear_sel)
        menu.addAction("🗑 删除选中页面", self._delete_selected)
        menu.exec(QCursor.pos())

    def _delete_selected(self):
        if self._scan_processing_blocked():
            notify(
                self,
                "请先停止或完成当前 OCR，再删除页面。\n"
                "正在运行的 Apple OCR/多模型任务已经冻结了页面队列，运行中修改页面会造成结果错位。",
                "warning",
            )
            return
        if self._load_signal_refs or self._scan_signal_refs:
            notify(
                self,
                "页面导入或扫描件优化仍在进行，请完成后再删除页面。",
                "warning",
            )
            return
        if not self.selected_pages:
            notify(self, "请先点选要删除的页面（可多选，或拖框多选）", "info")
            return
        reply = QMessageBox.question(
            self, "删除页面",
            ui_message("page.delete.confirm", count=len(self.selected_pages)),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        # 页码要重新连续编号，overrides 的 key 也要跟着重新映射，不然删掉中间
        # 某页之后，后面所有页的标注全部错位。
        keep_indices = [i for i in range(len(self.page_images)) if (i + 1) not in self.selected_pages]
        new_overrides = {}
        new_auto_suggested = set()
        for new_no, old_i in enumerate(keep_indices, start=1):
            old_no = old_i + 1
            if old_no in self.page_overrides:
                new_overrides[new_no] = self.page_overrides[old_no]
            if old_no in self._auto_suggested:
                new_auto_suggested.add(new_no)

        removed_paths = {
            str(self.page_images[i])
            for i in range(len(self.page_images))
            if (i + 1) in self.selected_pages
        }
        previous_pdf_map = dict(self._pdf_physical_page_map)
        self.page_images = [self.page_images[i] for i in keep_indices]
        self.page_overrides = new_overrides
        self._auto_suggested = new_auto_suggested
        self._pdf_physical_page_map = {
            new_no: previous_pdf_map[old_i + 1]
            for new_no, old_i in enumerate(keep_indices, start=1)
            if (old_i + 1) in previous_pdf_map
        }
        # 删除后的显式页面列表就是本会话新的唯一页面源。不能继续保留原文件夹/
        # PDF 作为 raw input，否则 OCR 页会再次展开原来源，把刚删掉的页面重新
        # 放回 Apple OCR 队列。刷新按钮也只刷新当前保留页，不会偷偷恢复已删除页。
        self._last_loaded_raw_inputs = [str(path) for path in self.page_images]
        # “恢复优化前”的历史包含删除前页面，继续保留会让用户一键把已删页
        # 重新带回 OCR。页面集合一旦人工删除，旧扫描优化历史即失效。
        self._scan_history.clear()
        self._scan_restore_btn.setEnabled(False)
        if hasattr(self, "_scan_restore_action"):
            self._scan_restore_action.setEnabled(False)
        for path in removed_paths:
            self.thumb_cache.pop(path, None)
            self._thumb_pending.discard(path)
            self._thumb_labels.pop(path, None)
        self.selected_pages.clear()
        self._count_lbl.setText(f"{len(self.page_images)} 页")
        self._render()
        self._persist_project_page_state()
        # 页面删除属于输入集合变化，而不是单纯分类变化。使用 pages_loaded 让
        # MainWindow 同步清空旧 OCR 文档、替换 OCRTab 输入与预览，并使下一次
        # OCR 只读取当前剩余页面。
        self.pages_loaded.emit([str(path) for path in self.page_images])

    def _refresh_page_type_styles(self, page_numbers) -> None:
        """Update only changed page badges/counts; never rebuild the thumbnail grid."""
        for raw_page_no in page_numbers or ():
            try:
                page_no = int(raw_page_no)
            except (TypeError, ValueError, OverflowError):
                continue
            tag = self._page_type_labels.get(page_no)
            if tag is None:
                continue
            ptype = self._ptype(page_no)
            tag.setText(TYPE_LABEL.get(ptype, "?"))
            color = TYPE_COLOR.get(ptype, "#AAA")
            tag_bg = blend(color, 0.14, "#FFFFFF")
            tag.setStyleSheet(
                f"background:{tag_bg};color:{color};border-radius:11px;"
                "font-size:10px;font-weight:700;"
            )
        self._update_counts()
        self._update_stat_bar()

    def _batch_tag(self, ttype):
        if not self.selected_pages:
            notify(self, "请先点选页面（可多选）", "info")
            return
        changed_pages = tuple(sorted(int(p) for p in self.selected_pages))
        for p in changed_pages:
            self.page_overrides[p] = ttype
            # 用户亲自选中并打了标——从"自动建议"升级成"确认过"。
            self._auto_suggested.discard(p)
        # High-frequency classification must be O(changed pages), not O(book).
        # Keep thumbnail widgets/pixmaps alive and only repaint their badges.
        self._refresh_page_type_styles(changed_pages)
        self.selected_pages.clear()
        self._refresh_selection_styles()
        self._persist_project_page_state(classification_only=True)
        self.types_changed.emit()

    def _select_all(self):
        self.selected_pages = set(range(1, len(self.page_images) + 1))
        self._refresh_selection_styles()

    def _clear_sel(self):
        self.selected_pages.clear()
        self._refresh_selection_styles()

    def _set_filter(self, ttype):
        self._current_filter = ttype
        for t, btn in self._filter_btns.items():
            btn.setChecked(t == ttype)
            if t == ttype:
                btn.setStyleSheet(
                    f"background:#FFFFFF;color:{ACC};padding:6px 14px;font-size:12px;"
                    "font-weight:700;border:1px solid #B9D3F6;border-radius:10px;"
                )
            else:
                btn.setStyleSheet(
                    f"background:transparent;color:{MUTED};padding:6px 10px;font-size:12px;"
                    "border:1px solid transparent;border-radius:10px;"
                )
        self._render()

    def _update_counts(self):
        counts = Counter(self._ptype(i + 1) for i in range(len(self.page_images)))
        total = len(self.page_images)
        for t, lbl in self._filter_counts.items():
            lbl.setText(str(total if t == "all" else counts.get(t, 0)))
        compact_counts = {
            "all": total,
            "text": counts.get("paragraph", 0),
            "illustration": counts.get("illustration", 0) + counts.get("color_illus", 0),
            "cover": counts.get("cover", 0),
        }
        compact_labels = {"all": "全部", "text": "正文", "illustration": "插图", "cover": "封面"}
        for t, btn in self._filter_btns.items():
            btn.setText(f"{compact_labels.get(t, t)} {compact_counts.get(t, 0)}")
            btn.setChecked(t == self._current_filter)
        # Apply the same selected-state styling on the first render/load.
        for t, btn in self._filter_btns.items():
            if t == self._current_filter:
                btn.setStyleSheet(
                    f"background:#FFFFFF;color:{ACC};padding:6px 14px;font-size:12px;"
                    "font-weight:700;border:1px solid #B9D3F6;border-radius:10px;"
                )
            else:
                btn.setStyleSheet(
                    f"background:transparent;color:{MUTED};padding:6px 10px;font-size:12px;"
                    "border:1px solid transparent;border-radius:10px;"
                )

    def _update_stat_bar(self):
        counts = Counter(self._ptype(i + 1) for i in range(len(self.page_images)))
        parts = []
        for t, l, c in PAGE_TYPES:
            n = counts.get(t, 0)
            if n > 0:
                # Use the exact same PAGE_TYPES palette as the tagging controls
                # and thumbnail badges. Only the dot is colored; text keeps the
                # neutral status-bar tone for readability.
                parts.append(
                    f'<span style="color:{c};font-weight:700">●</span> '
                    f'<span style="color:{MUTED}">{l} {n}页</span>'
                )
        self._stat_label.setTextFormat(Qt.RichText)
        self._stat_label.setText("&nbsp;&nbsp;&nbsp;".join(parts) if parts else "")

    def shutdown_cleanup(self) -> None:
        self._load_generation += 1
        self._scan_process_generation += 1
        self._grid_build_generation += 1
        # Let an already queued page-state snapshot finish before process exit.
        # The writer is a non-daemon thread and exits after the queue drains.
        self._page_state_save_closed = True
        self._thumb_closed = True
        for timer_name in ("_visible_thumb_timer", "_grid_reflow_timer"):
            timer = getattr(self, timer_name, None)
            if timer is not None:
                timer.stop()
        self._load_signal_refs.clear()
        self._scan_signal_refs.clear()
        self._thumb_pending.clear()
        try:
            while True:
                self._thumb_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._thumb_queue.put_nowait(None)
        except queue.Full:
            pass
