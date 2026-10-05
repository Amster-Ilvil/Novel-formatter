from __future__ import annotations

import copy
import os
import re
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, QFrame, QLabel, QPushButton, QComboBox,
    QLineEdit, QListWidget, QPlainTextEdit, QProgressBar, QRadioButton, QStackedWidget, QTabWidget,
    QTextBrowser, QTreeView, QFileSystemModel, QFileDialog, QMessageBox, QSizePolicy, QAbstractItemView, QMenu,
)
from PySide6.QtCore import Qt, Signal, QTimer, QUrl, QDir, QModelIndex
from PySide6.QtGui import QKeySequence, QShortcut, QPixmap

# QtWebEngine is intentionally opt-in.  In packaged macOS standalone builds
# WebEngine can terminate the whole GUI with SIGSEGV after an EPUB build even
# though the EPUB itself was already written successfully.  QTextBrowser is
# sufficient for the default package/content preview and has no helper process.
_ENABLE_WEBENGINE_PREVIEW = os.environ.get("NOVEL_FORMATTER_ENABLE_WEBENGINE_PREVIEW", "0").strip() == "1"
if _ENABLE_WEBENGINE_PREVIEW:
    try:
        from PySide6.QtWebEngineWidgets import QWebEngineView
        HAS_WEBENGINE = True
    except Exception:
        QWebEngineView = None
        HAS_WEBENGINE = False
else:
    QWebEngineView = None
    HAS_WEBENGINE = False

from models.document import UnifiedDocument, BlockType, TocEntry
from models.format_profile import FormatProfileStore
from ui.flow_layout import FlowLayout
from ui.dialogs import show_error_dialog
from ui.formatter.tab import FormatProfileDialog
from ui.common.signals import WorkerSignals
from ui.common.toast import notify
from ui.common.styling import BG, ACC, BORDER, CARD, DANGER, INK, MUTED, SUCCESS, accent_button, link_button, make_separator, wrap_in_card
from utils.clear_manager import ClearManager, create_workspace_clear_button

class EPUBTab(QWidget):
    epub_built = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._doc: Optional[UnifiedDocument] = None
        self._source_kind = ""
        self._source_label = "未选择正文"
        self._selected_preview_path = ""
        self._preview_extract_dir = ""
        self._package_model: Optional[QFileSystemModel] = None
        self._workspace_active = False
        self._preview_pending_while_hidden = False
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._build_generation = 0
        self._document_generation = 0
        self._build_running = False
        self._project_export_dir = ""
        self._page_manager_images: tuple[str, ...] = ()
        self._page_manager_overrides: dict[int, str] = {}
        self._page_manager_cover_path = ""
        self._preview_timer.timeout.connect(self._render_live_preview)
        self._build()
        self.destroyed.connect(lambda *_: self._cleanup_preview_dir())

    def _build(self):
        root = wrap_in_card(self)

        # ── 左侧：结构树 + 设置 ───────────────────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(320)
        left.setMaximumWidth(360)
        left.setStyleSheet(f"background: {BG};")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)

        top = QWidget()
        tv = QVBoxLayout(top)
        tv.setContentsMargins(14, 10, 14, 10)
        tv.setSpacing(8)
        tv.addWidget(QLabel("<b>EPUB 制作</b>"))
        purpose = QLabel("这里仅负责检查、预览与生成 EPUB；OCR 与正文整理请在前面的工作区完成。")
        purpose.setWordWrap(True)
        purpose.setStyleSheet(f"color: {MUTED}; font-size: 10px;")
        tv.addWidget(purpose)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.setContentsMargins(0, 0, 0, 0)

        self._refresh_btn = QPushButton("↻ 刷新检查")
        self._refresh_btn.setProperty("flat", True)
        self._refresh_btn.setStyleSheet("padding: 6px 12px;")
        self._refresh_btn.clicked.connect(self._refresh_document_summary)
        btn_row.addWidget(self._refresh_btn)

        self._build_btn = accent_button("生成 EPUB")
        self._build_btn.clicked.connect(self._build_epub)
        btn_row.addWidget(self._build_btn, 1)
        create_workspace_clear_button(
            self, "EPUB", ClearManager.clear_epub, target_layout=btn_row,
        )
        tv.addLayout(btn_row)

        ll.addWidget(top)
        ll.addWidget(make_separator())

        # 逻辑内容结构与实际 EPUB 包文件分开显示。包文件使用 QFileSystemModel，
        # 不再手工反复创建 QTreeWidgetItem，避免连续处理第二本书时 Qt 内部越界。
        self._left_tabs = QTabWidget()
        self._left_tabs.setDocumentMode(True)
        self._left_tabs.setStyleSheet("QTabWidget::pane { border: 0; }")

        self._structure_list = QListWidget()
        self._structure_list.setStyleSheet(f"margin: 6px 8px; border: 1px solid {BORDER}; border-radius: 7px;")
        self._structure_list.setSelectionMode(QAbstractItemView.NoSelection)
        self._structure_list.addItem("等待正文版本")
        self._left_tabs.addTab(self._structure_list, "内容结构")

        package_page = QWidget()
        package_layout = QVBoxLayout(package_page)
        package_layout.setContentsMargins(8, 6, 8, 6)
        self._package_stack = QStackedWidget()
        self._package_placeholder = QLabel("生成 EPUB 后，可在这里浏览实际 XHTML、CSS、图片和 OPF 文件。")
        self._package_placeholder.setAlignment(Qt.AlignCenter)
        self._package_placeholder.setWordWrap(True)
        self._package_placeholder.setStyleSheet(f"color: {MUTED}; padding: 18px;")
        self._package_tree = QTreeView()
        self._package_tree.setHeaderHidden(True)
        self._package_tree.setAnimated(False)
        self._package_tree.setIndentation(16)
        self._package_tree.setStyleSheet(f"border: 1px solid {BORDER}; border-radius: 7px;")
        self._package_tree.clicked.connect(self._on_package_tree_click)
        self._package_stack.addWidget(self._package_placeholder)
        self._package_stack.addWidget(self._package_tree)
        package_layout.addWidget(self._package_stack)
        self._left_tabs.addTab(package_page, "打包文件")
        ll.addWidget(self._left_tabs, 1)

        ll.addWidget(make_separator())

        # 元数据编辑
        meta_lbl = QLabel("元数据")
        meta_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px; padding: 4px 14px 2px;")
        ll.addWidget(meta_lbl)

        form = QWidget()
        fl = QFormLayout(form)
        fl.setContentsMargins(14, 0, 14, 0)
        fl.setSpacing(4)

        self._title_edit = QLineEdit()
        self._title_edit.setPlaceholderText("书名；可直接粘贴，生成时自动作为 EPUB 文件名")
        title_row = QWidget()
        title_layout = QHBoxLayout(title_row)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(4)
        title_layout.addWidget(self._title_edit, 1)
        paste_title_btn = QPushButton("粘贴")
        paste_title_btn.setProperty("flat", True)
        paste_title_btn.setToolTip("从剪贴板粘贴书名，并用于建议保存文件名")
        paste_title_btn.clicked.connect(self._paste_title_from_clipboard)
        title_layout.addWidget(paste_title_btn)
        fl.addRow("书名:", title_row)
        self._author_edit = QLineEdit()
        self._author_edit.setPlaceholderText("作者")
        fl.addRow("作者:", self._author_edit)
        self._publisher_edit = QLineEdit()
        self._publisher_edit.setPlaceholderText("出版社")
        fl.addRow("出版社:", self._publisher_edit)
        self._volume_edit = QLineEdit()
        self._volume_edit.setPlaceholderText("卷号")
        fl.addRow("卷号:", self._volume_edit)

        template_row = QWidget()
        tr = QHBoxLayout(template_row)
        tr.setContentsMargins(0, 0, 0, 0)
        tr.setSpacing(4)
        self._template_combo = QComboBox()
        # Template/profile names are stable identifiers and may be user-defined;
        # never localize their item text.
        self._template_combo.setProperty("nfNoTranslateItems", True)
        tr.addWidget(self._template_combo, 1)
        self._manage_tpl_btn = QPushButton("⚙ 管理格式")
        self._manage_tpl_btn.setProperty("flat", True)
        self._manage_tpl_btn.setToolTip("管理排版格式（新建/学习/删除/导入导出）")
        self._manage_tpl_btn.clicked.connect(self._open_format_profiles)
        tr.addWidget(self._manage_tpl_btn)
        fl.addRow("CSS 模板:", template_row)
        self._reload_templates()

        mode_row = QWidget()
        mr = QHBoxLayout(mode_row)
        mr.setContentsMargins(0, 0, 0, 0)
        self._vert_radio = QRadioButton("竖排")
        self._horiz_radio = QRadioButton("横排")
        self._horiz_radio.setChecked(True)
        mr.addWidget(self._vert_radio)
        mr.addWidget(self._horiz_radio)
        mr.addStretch()
        fl.addRow("排版:", mode_row)

        ll.addWidget(form)
        meta_hint = QLabel("可复制粘贴书名；生成 EPUB 时会自动把书名填入保存文件名。书名留空时使用导出的文件名。")
        meta_hint.setWordWrap(True)
        meta_hint.setStyleSheet(f"color: {MUTED}; font-size: 10px; padding: 2px 14px 0;")
        ll.addWidget(meta_hint)
        ll.addSpacing(10)
        root.addWidget(left, 0)

        # ── 右侧：元数据条 + 源码/预览 ────────────────────────────────────────
        right = QWidget()
        right.setMinimumWidth(520)
        right.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right.setStyleSheet(f"background: {BG};")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)

        # 顶部只显示会影响导出结果的状态，不重复展示左侧可编辑的书名/作者。
        pills_bar = QWidget()
        pills_bar.setStyleSheet(f"background: {BG};")
        pbl = FlowLayout(pills_bar, hspacing=8, vspacing=8)
        pbl.setContentsMargins(14, 8, 14, 8)
        self._pills: dict[str, QLabel] = {}
        for key, icon, default in [
            ("source", "🧾", "未选择正文"), ("cover", "🖼", "封面未设置"),
            ("chapters", "☰", "0 章"), ("images", "🎨", "0 图"),
            ("css", "CSS", "默认模板"), ("mode", "↕", "横排"),
            ("lang", "🌐", "ja · EPUB3"),
        ]:
            pill = QLabel(f"{icon} {default}")
            pill.setStyleSheet(
                f"background: #F5F4F0; color: #444; border: 1px solid {BORDER}; "
                f"border-radius: 6px; padding: 4px 10px; font-size: 11px;")
            pbl.addWidget(pill)
            self._pills[key] = pill
        rl.addWidget(pills_bar)
        rl.addWidget(make_separator())

        self._prog = QProgressBar()
        self._prog.setVisible(False)
        rl.addWidget(self._prog)

        # 内容切换：源码 / 预览
        view_row = QHBoxLayout()
        view_row.setContentsMargins(14, 6, 14, 4)
        self._preview_title = QLabel("选择左侧文件预览内容")
        self._preview_title.setStyleSheet(f"color: {MUTED}; font-size: 12px;")
        view_row.addWidget(self._preview_title)
        view_row.addStretch()

        self._code_btn = QPushButton("源码")
        self._code_btn.setProperty("flat", True)
        self._code_btn.setStyleSheet(f"background: #E4EEFF; color: {INK}; border: 1px solid #CDD3DA; border-radius: 6px; padding: 6px 12px;")
        self._code_btn.clicked.connect(lambda: self._switch_preview("code"))
        view_row.addWidget(self._code_btn)
        self._preview_btn = QPushButton("预览")
        self._preview_btn.setProperty("flat", True)
        self._preview_btn.clicked.connect(lambda: self._switch_preview("preview"))
        view_row.addWidget(self._preview_btn)
        rl.addLayout(view_row)
        rl.addWidget(make_separator())

        # 源码视图
        self._code_view = QPlainTextEdit()
        self._code_view.setReadOnly(True)
        self._code_view.setStyleSheet(f"border: none; border-radius: 0;")
        rl.addWidget(self._code_view, 1)

        # 真实 XHTML/CSS 预览。安装 Qt WebEngine 时使用完整浏览器内核，
        # 否则回退 QTextBrowser，仍能实时显示正文、标题和大部分 CSS。
        if HAS_WEBENGINE:
            self._preview_widget = QWebEngineView()
            self._preview_is_webengine = True
        else:
            self._preview_widget = QTextBrowser()
            self._preview_widget.setOpenExternalLinks(False)
            self._preview_is_webengine = False
        self._preview_widget.setStyleSheet("border: none; background: #FFFFFF;")
        self._preview_widget.setVisible(False)
        rl.addWidget(self._preview_widget, 1)

        # 底部统计
        rl.addWidget(make_separator())
        stats = QWidget()
        stats.setStyleSheet(f"background: {BG};")
        st_layout = QHBoxLayout(stats)
        st_layout.setContentsMargins(0, 0, 0, 0)
        st_layout.setSpacing(0)
        self._stat_labels: dict[str, tuple[QLabel, QLabel]] = {}
        for key, label in [("files", "文件数"), ("chapters", "章节"), ("images", "图片"),
                           ("size", "估算大小"), ("status", "状态")]:
            cell = QWidget()
            cell.setStyleSheet(f"border-right: 1px solid {BORDER};")
            cl_layout = QVBoxLayout(cell)
            cl_layout.setContentsMargins(8, 4, 8, 4)
            v = QLabel("0" if key != "status" else "待构建")
            v.setStyleSheet(f"font-size: 16px; font-weight: bold;")
            v.setAlignment(Qt.AlignCenter)
            cl_layout.addWidget(v)
            n = QLabel(label)
            n.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
            n.setAlignment(Qt.AlignCenter)
            cl_layout.addWidget(n)
            st_layout.addWidget(cell, 1)
            self._stat_labels[key] = (v, n)
        rl.addWidget(stats)

        root.addWidget(right, 1)

        # ── Phase 20 compact export dashboard ─────────────────────────────
        # The complete package browser/source preview above remains intact and
        # is available through “高级构建”.  The default view mirrors the
        # supplied metadata + export-summary mock-up while delegating every
        # action to the existing EPUB pipeline.
        self._epub_advanced_left = left
        self._epub_advanced_right = right
        compact = QWidget()
        compact.setObjectName("epubCompactView")
        compact_layout = QHBoxLayout(compact)
        compact_layout.setContentsMargins(28, 8, 28, 22)
        compact_layout.setSpacing(16)

        meta_card = QFrame()
        meta_card.setObjectName("epubCompactMetaCard")
        meta_card.setFixedWidth(459)
        meta_card.setStyleSheet(
            f"QFrame#epubCompactMetaCard{{background:{CARD};border:1px solid {BORDER};border-radius:16px;}}"
        )
        meta_box = QVBoxLayout(meta_card)
        meta_box.setContentsMargins(20, 18, 20, 18)
        meta_box.setSpacing(10)
        meta_tabs = QHBoxLayout()
        for i, text in enumerate(("书籍信息", "样式", "目录")):
            label = QLabel(text)
            if i == 0:
                label.setAlignment(Qt.AlignCenter)
                label.setMinimumSize(82, 32)
                label.setStyleSheet(
                    f"color:{ACC};font-size:11px;font-weight:750;background:#FFFFFF;"
                    f"border:1px solid #CDD3DA;border-radius:9px;padding:0 12px;"
                )
            else:
                label.setStyleSheet(
                    f"color:{MUTED};font-size:11px;font-weight:600;padding:7px 8px;"
                )
            meta_tabs.addWidget(label)
        meta_tabs.addStretch(1)
        meta_box.addLayout(meta_tabs)

        def _field(label_text: str, editor: QWidget) -> None:
            label = QLabel(label_text)
            label.setStyleSheet(f"color:{INK};font-size:11px;font-weight:600;")
            meta_box.addWidget(label)
            editor.setMinimumHeight(34)
            meta_box.addWidget(editor)

        self._compact_title_edit = QLineEdit()
        self._compact_author_edit = QLineEdit()
        # Publisher/volume remain live compatibility mirrors for the advanced
        # metadata editor, but no longer consume space in the calm reference
        # surface.
        self._compact_publisher_edit = QLineEdit(meta_card)
        self._compact_volume_edit = QLineEdit(meta_card)
        self._compact_publisher_edit.hide()
        self._compact_volume_edit.hide()
        _field("书名", self._compact_title_edit)
        _field("作者", self._compact_author_edit)

        language_mode = QHBoxLayout()
        language_mode.setSpacing(12)
        language_box = QWidget()
        language_layout = QVBoxLayout(language_box)
        language_layout.setContentsMargins(0, 0, 0, 0)
        language_layout.setSpacing(5)
        language_layout.addWidget(QLabel("语言"))
        self._compact_language_combo = QComboBox()
        self._compact_language_combo.addItem("日语 (ja)", "ja")
        self._compact_language_combo.addItem("简体中文 (zh-CN)", "zh-CN")
        self._compact_language_combo.addItem("English (en)", "en")
        self._compact_language_combo.setMinimumHeight(34)
        language_layout.addWidget(self._compact_language_combo)
        language_mode.addWidget(language_box, 1)

        direction_box = QWidget()
        direction_layout = QVBoxLayout(direction_box)
        direction_layout.setContentsMargins(0, 0, 0, 0)
        direction_layout.setSpacing(5)
        direction_layout.addWidget(QLabel("阅读方向"))
        self._compact_mode_combo = QComboBox()
        self._compact_mode_combo.addItem("横排 · 左开", "horizontal")
        self._compact_mode_combo.addItem("竖排 · 右开", "vertical")
        self._compact_mode_combo.setMinimumHeight(34)
        direction_layout.addWidget(self._compact_mode_combo)
        language_mode.addWidget(direction_box, 1)
        meta_box.addLayout(language_mode)

        for label_text, value_text in (("EPUB3 导航", "始终生成"), ("Ruby 注音", "按正文保留")):
            sep = QFrame()
            sep.setFixedHeight(1)
            sep.setStyleSheet(f"background:{BORDER};border:none;")
            meta_box.addWidget(sep)
            row = QHBoxLayout()
            row.addWidget(QLabel(label_text))
            row.addStretch(1)
            value = QLabel(value_text)
            value.setStyleSheet(f"color:{ACC};font-size:10px;font-weight:700;")
            row.addWidget(value)
            meta_box.addLayout(row)

        advanced_btn = link_button("高级构建 / 包文件预览  ›")
        advanced_btn.clicked.connect(lambda: self._set_epub_compact_mode(False))
        advanced_btn.show()
        self._compact_advanced_btn = advanced_btn
        meta_box.addWidget(advanced_btn)
        meta_box.addStretch(1)

        meta_card.setContextMenuPolicy(Qt.CustomContextMenu)
        def _open_epub_meta_menu(pos):
            menu = QMenu(meta_card)
            menu.addAction("高级构建 / 包文件预览", lambda: self._set_epub_compact_mode(False))
            menu.exec(meta_card.mapToGlobal(pos))
        meta_card.customContextMenuRequested.connect(_open_epub_meta_menu)
        self._epub_advanced_shortcut = QShortcut(QKeySequence("Ctrl+Shift+E"), compact)
        self._epub_advanced_shortcut.activated.connect(lambda: self._set_epub_compact_mode(False))
        compact_layout.addWidget(meta_card, 0)

        right_compact = QWidget()
        right_compact_layout = QVBoxLayout(right_compact)
        right_compact_layout.setContentsMargins(0, 0, 0, 0)
        right_compact_layout.setSpacing(14)

        summary_card = QFrame()
        summary_card.setObjectName("epubCompactSummaryCard")
        summary_card.setStyleSheet(
            f"QFrame#epubCompactSummaryCard{{background:{CARD};border:1px solid {BORDER};border-radius:16px;}}"
        )
        summary_card.setFixedHeight(248)
        summary_box = QHBoxLayout(summary_card)
        summary_box.setContentsMargins(20, 18, 20, 18)
        summary_box.setSpacing(24)
        cover = QLabel("封面\n未设置")
        cover.setAlignment(Qt.AlignCenter)
        cover.setFixedSize(150, 210)
        cover.setStyleSheet(
            f"background:{ACC};color:white;border-radius:8px;font-size:16px;font-weight:700;line-height:1.4;"
        )
        self._compact_cover_preview = cover
        summary_box.addWidget(cover)
        metrics = QWidget()
        mg = QGridLayout(metrics)
        mg.setContentsMargins(0, 0, 0, 0)
        mg.setHorizontalSpacing(34)
        overview_title = QLabel("导出概览")
        overview_title.setStyleSheet(f"color:{MUTED};font-size:10px;font-weight:700;")
        mg.addWidget(overview_title, 0, 0, 1, 3)
        self._compact_stat_values: dict[str, QLabel] = {}
        for col, (key, name) in enumerate((("pages", "页"), ("chapters", "章节"), ("size", "预计大小"))):
            value = QLabel("0")
            value.setStyleSheet(f"color:{INK};font-size:22px;font-weight:800;")
            caption = QLabel(name)
            caption.setStyleSheet(f"color:{MUTED};font-size:10px;")
            mg.addWidget(value, 1, col)
            mg.addWidget(caption, 2, col)
            self._compact_stat_values[key] = value
        self._compact_source_label = QLabel("等待正文")
        self._compact_source_label.hide()
        mg.addWidget(self._compact_source_label, 3, 0, 1, 3)
        actions = QHBoxLayout()
        build2 = accent_button("生成 EPUB")
        build2.clicked.connect(self._build_epub)
        refresh2 = QPushButton("预览")
        def _open_compact_preview():
            self._set_epub_compact_mode(False)
            self._switch_preview("preview")
        refresh2.clicked.connect(_open_compact_preview)
        actions.addWidget(build2)
        actions.addWidget(refresh2)
        actions.addStretch(1)
        mg.addLayout(actions, 4, 0, 1, 3)
        summary_box.addWidget(metrics, 1)
        right_compact_layout.addWidget(summary_card)

        check_card = QFrame()
        check_card.setObjectName("epubCompactCheckCard")
        check_card.setStyleSheet(
            f"QFrame#epubCompactCheckCard{{background:{CARD};border:1px solid {BORDER};border-radius:16px;}}"
        )
        check_box = QVBoxLayout(check_card)
        check_box.setContentsMargins(20, 16, 20, 16)
        check_title = QLabel("检查结果")
        check_title.setStyleSheet("font-size:13px;font-weight:700;")
        check_box.addWidget(check_title)
        self._compact_checks: dict[str, QLabel] = {}
        check_names = ("目录结构", "注音标签", "EPUBCheck")
        for index, text in enumerate(check_names):
            row_widget = QWidget(check_card)
            row_widget.setFixedHeight(38)
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(QLabel(text))
            row.addStretch(1)
            status = QLabel("待检查")
            status.setAlignment(Qt.AlignCenter)
            status.setMinimumWidth(54)
            status.setStyleSheet(
                f"color:{MUTED};font-size:10px;font-weight:700;background:#EEF3FA;"
                "border-radius:9px;padding:3px 8px;"
            )
            row.addWidget(status)
            check_box.addWidget(row_widget)
            self._compact_checks[text] = status
            if index < len(check_names) - 1:
                sep = QFrame(check_card)
                sep.setFixedHeight(1)
                sep.setStyleSheet(f"background:{BORDER};border:none;")
                check_box.addWidget(sep)
        check_box.addStretch(1)
        right_compact_layout.addWidget(check_card, 1)
        compact_layout.addWidget(right_compact, 1)

        root.addWidget(compact, 1)
        self._epub_compact_view = compact
        self._epub_advanced_left.setVisible(False)
        self._epub_advanced_right.setVisible(False)

        # Bidirectional metadata bridge: compact and advanced views are merely
        # two views over the same values.
        for compact_edit, advanced_edit in (
            (self._compact_title_edit, self._title_edit),
            (self._compact_author_edit, self._author_edit),
            (self._compact_publisher_edit, self._publisher_edit),
            (self._compact_volume_edit, self._volume_edit),
        ):
            compact_edit.setText(advanced_edit.text())
            compact_edit.textChanged.connect(
                lambda value, target=advanced_edit: target.setText(value) if target.text() != value else None
            )
            advanced_edit.textChanged.connect(
                lambda value, target=compact_edit: target.setText(value) if target.text() != value else None
            )
        self._compact_mode_combo.currentIndexChanged.connect(self._sync_compact_epub_mode_to_advanced)
        self._compact_language_combo.currentIndexChanged.connect(self._sync_compact_epub_language_to_document)
        self._vert_radio.toggled.connect(lambda _v: self._sync_compact_epub_mode_from_advanced())
        self._horiz_radio.toggled.connect(lambda _v: self._sync_compact_epub_mode_from_advanced())

        compact_return = QPushButton("简洁视图")
        compact_return.setProperty("flat", True)
        compact_return.clicked.connect(lambda: self._set_epub_compact_mode(True))
        tv.insertWidget(0, compact_return, 0, Qt.AlignLeft)
        self._sync_compact_epub_mode_from_advanced()
        self._refresh_compact_epub_summary()

        for edit in (self._title_edit, self._author_edit, self._publisher_edit, self._volume_edit):
            edit.textChanged.connect(self._schedule_live_preview)
        self._template_combo.currentTextChanged.connect(self._schedule_live_preview)
        self._vert_radio.toggled.connect(self._schedule_live_preview)
        self._horiz_radio.toggled.connect(self._schedule_live_preview)

    def _set_epub_compact_mode(self, compact: bool) -> None:
        compact = bool(compact)
        self._epub_compact_view.setVisible(compact)
        self._epub_advanced_left.setVisible(not compact)
        self._epub_advanced_right.setVisible(not compact)
        if compact:
            self._refresh_compact_epub_summary()

    def _sync_compact_epub_mode_to_advanced(self, _index: int = -1) -> None:
        vertical = self._compact_mode_combo.currentData() == "vertical"
        if vertical:
            self._vert_radio.setChecked(True)
        else:
            self._horiz_radio.setChecked(True)

    def _sync_compact_epub_mode_from_advanced(self) -> None:
        desired = "vertical" if self._vert_radio.isChecked() else "horizontal"
        idx = self._compact_mode_combo.findData(desired)
        if idx >= 0 and self._compact_mode_combo.currentIndex() != idx:
            blocked = self._compact_mode_combo.blockSignals(True)
            self._compact_mode_combo.setCurrentIndex(idx)
            self._compact_mode_combo.blockSignals(blocked)

    def _refresh_compact_epub_summary(self) -> None:
        if not hasattr(self, "_compact_stat_values"):
            return
        if "pages" in self._compact_stat_values:
            page_count = len(getattr(self._doc, "pages", ()) or ()) if self._doc is not None else 0
            self._compact_stat_values["pages"].setText(str(page_count))
        for key in ("chapters", "size"):
            source = self._stat_labels.get(key)
            if source and key in self._compact_stat_values:
                self._compact_stat_values[key].setText(source[0].text())
        if hasattr(self, "_compact_source_label"):
            self._compact_source_label.setText(str(self._source_label or "等待正文"))
        if hasattr(self, "_compact_language_combo"):
            language = str(getattr(getattr(self._doc, "metadata", None), "language", "") or "ja")
            idx = self._compact_language_combo.findData(language)
            if idx < 0:
                idx = self._compact_language_combo.findData("ja")
            if idx >= 0 and self._compact_language_combo.currentIndex() != idx:
                blocked = self._compact_language_combo.blockSignals(True)
                self._compact_language_combo.setCurrentIndex(idx)
                self._compact_language_combo.blockSignals(blocked)

    def _sync_compact_epub_language_to_document(self, _index: int = -1) -> None:
        if self._doc is None or not hasattr(self, "_compact_language_combo"):
            return
        language = str(self._compact_language_combo.currentData() or "ja")
        if getattr(self._doc.metadata, "language", "") != language:
            self._doc.metadata.language = language
            self._update_pills()
            self._schedule_live_preview()

    def _reload_templates(self):
        """内置模板 + 自定义 Format Profile 名称一起塞进下拉框，尽量保留原来的选中项。"""
        from builder.epub_builder import CSS_TEMPLATES
        current = self._template_combo.currentText()
        self._template_combo.blockSignals(True)
        self._template_combo.clear()
        self._template_combo.addItems(list(CSS_TEMPLATES.keys()))
        for p in FormatProfileStore().list():
            self._template_combo.addItem(p.name)
        idx = self._template_combo.findText(current)
        self._template_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._template_combo.blockSignals(False)

    def _open_format_profiles(self):
        dlg = FormatProfileDialog(self)
        dlg.exec()
        self._reload_templates()

    def _reset_package_browser(self, message: str = "生成 EPUB 后，可浏览实际打包文件。"):
        """Detach the old filesystem model before deleting its preview directory."""
        self._selected_preview_path = ""
        if getattr(self, "_package_tree", None) is not None:
            try:
                self._package_tree.setRootIndex(QModelIndex())
                self._package_tree.setModel(None)
            except RuntimeError:
                # QWidget teardown can reach here after the child view is gone.
                pass
        old_model = self._package_model
        self._package_model = None
        if old_model is not None:
            try:
                old_model.deleteLater()
            except RuntimeError:
                pass
        if getattr(self, "_package_placeholder", None) is not None:
            try:
                self._package_placeholder.setText(message)
                self._package_stack.setCurrentWidget(self._package_placeholder)
            except RuntimeError:
                pass

    def _install_package_model(self, root_path: str):
        self._reset_package_browser()
        model = QFileSystemModel(self._package_tree)
        model.setFilter(QDir.AllDirs | QDir.Files | QDir.NoDotAndDotDot)
        root_index = model.setRootPath(root_path)
        self._package_tree.setModel(model)
        self._package_tree.setRootIndex(root_index)
        for column in range(1, model.columnCount()):
            self._package_tree.hideColumn(column)
        self._package_tree.expandToDepth(1)
        self._package_model = model
        self._package_stack.setCurrentWidget(self._package_tree)

    def _source_label_for_doc(self, doc: UnifiedDocument, source_kind: str = "") -> str:
        mode = str(getattr(doc.metadata, "ai_processing_mode", "") or "")
        ai_source = str(getattr(doc.metadata, "ai_source_kind", "") or "")
        if source_kind == "ai":
            base = "AI 纠错排版" if mode == "typeset" else "AI 纠错"
            if ai_source == "ocr":
                return f"{base}（来源：OCR）"
            if ai_source == "replacement":
                return f"{base}（来源：替换结果）"
            return base
        labels = {
            "ocr": "OCR 原文", "formatter": "Formatter 结果",
            "pdf_text": "PDF 文字层",
            "ai_image_source": "AI 图文原文",
            "ai_image_translation": "AI 图文译文",
        }
        if source_kind in labels:
            return labels[source_kind]
        if mode == "typeset":
            return "AI 纠错排版"
        if mode == "correction":
            return "AI 纠错"
        return "当前工作文档"

    def _refresh_document_summary(self, *_args):
        self._update_pills()
        self._structure_list.clear()
        if not self._doc:
            self._structure_list.addItem("等待正文版本")
            self._stat_labels["chapters"][0].setText("0")
            self._stat_labels["images"][0].setText("0")
            self._refresh_compact_epub_summary()
            self._schedule_live_preview()
            return

        cover_pages = [p for p in self._doc.pages if p.page_type == BlockType.COVER and p.image_path]
        effective_cover = self._effective_cover_path()
        image_blocks = self._doc.image_blocks()
        chapters = list(self._doc.toc)
        if not chapters:
            chapters = [TocEntry(str(b.text or "").strip(), int(b.chapter_index or 0), i)
                        for i, b in enumerate(self._doc.blocks)
                        if b.type == BlockType.CHAPTER and str(b.text or "").strip()]

        self._structure_list.addItem(f"正文来源：{self._source_label}")
        if self._page_manager_cover_path:
            try:
                page_no = self._page_manager_images.index(self._page_manager_cover_path) + 1
            except ValueError:
                page_no = 0
            self._structure_list.addItem(f"✓ 封面：页面管理第 {page_no} 页" if page_no else "✓ 封面：已同步页面管理")
        elif cover_pages:
            self._structure_list.addItem(f"✓ 封面：第 {cover_pages[0].page_no} 页")
        elif effective_cover:
            self._structure_list.addItem("✓ 封面：已设置")
        else:
            self._structure_list.addItem("⚠ 封面：未设置（可在页面管理中指定）")
        self._structure_list.addItem(f"图片页：{len(image_blocks)}")
        self._structure_list.addItem(f"章节：{len(chapters)}")
        for index, entry in enumerate(chapters[:120], start=1):
            title = str(getattr(entry, "title", "") or f"第 {index} 章").strip()
            self._structure_list.addItem(f"  {index:02d}. {title}")
        if len(chapters) > 120:
            self._structure_list.addItem(f"  … 另有 {len(chapters) - 120} 章")

        self._stat_labels["chapters"][0].setText(str(len(chapters)))
        self._stat_labels["images"][0].setText(str(len(image_blocks)))
        self._refresh_compact_epub_summary()
        self._schedule_live_preview()

    def clear_doc(self):
        self._document_generation += 1
        self._doc = None
        self._source_kind = ""
        self._source_label = "未选择正文"
        self._ai_layout_locked = False
        self._selected_preview_path = ""
        self._cleanup_preview_dir()
        self._structure_list.clear()
        self._structure_list.addItem("等待新书")
        for edit in (self._title_edit, self._author_edit, self._publisher_edit, self._volume_edit):
            edit.clear()
        self._code_view.clear()
        self._preview_title.setText("实时预览当前正文；生成后可从“打包文件”选择具体文件")
        for key, value in {"chapters":"0", "images":"0", "size":"0 KB", "status":"待构建", "files":"0"}.items():
            if key in self._stat_labels:
                self._stat_labels[key][0].setText(value)
        self._update_pills()
        self._stat_labels["status"][0].setStyleSheet("font-size: 16px; font-weight: bold;")
        self._refresh_compact_epub_summary()
        self._render_live_preview()

    def set_doc(self, doc: UnifiedDocument, source_kind: str = ""):
        self._document_generation += 1
        self._doc = doc
        if source_kind:
            self._source_kind = source_kind
        self._source_label = self._source_label_for_doc(doc, self._source_kind)
        self._ai_layout_locked = bool(getattr(doc.metadata, "ai_layout_locked", False))
        ai_css_active = (
            getattr(doc.metadata, "ai_processing_mode", "") == "typeset"
            and self._ai_layout_locked
            and bool(str(getattr(doc.metadata, "ai_epub_css", "") or "").strip())
        )
        self._template_combo.setEnabled(not ai_css_active)
        self._manage_tpl_btn.setEnabled(not ai_css_active)
        if ai_css_active:
            self._template_combo.setToolTip("AI纠错排版版本会自动应用 AI 返回的 CSS；此处模板不参与导出")
        else:
            self._template_combo.setToolTip("")
        self._selected_preview_path = ""

        # 新正文进入时，实际打包文件一定已经过期，立即解除旧模型和旧目录。
        self._cleanup_preview_dir()
        self._left_tabs.setCurrentIndex(0)

        self._title_edit.setText(doc.metadata.title or "")
        self._author_edit.setText(doc.metadata.author or "")
        self._publisher_edit.setText(doc.metadata.publisher or "")
        self._volume_edit.setText(doc.metadata.volume or "")
        # Export layout follows the authoritative document metadata instead of
        # inheriting whichever radio button happened to be selected for the
        # previous book.  This keeps the OCR/Formatter -> EPUB handoff symmetric:
        # explicit horizontal documents select horizontal, while Japanese
        # vertical documents select vertical.  Unknown/legacy documents keep the
        # user's current manual choice.
        writing_direction = str(getattr(doc.metadata, "writing_direction", "") or "").strip()
        ocr_mode = str(getattr(doc.metadata, "ocr_mode", "") or "").strip()
        if writing_direction == "horizontal-tb" or ocr_mode == "zh_hans_horizontal":
            self._horiz_radio.setChecked(True)
        elif writing_direction.startswith("vertical") or ocr_mode == "ja_vertical":
            self._vert_radio.setChecked(True)

        for key, value in {"size": "0 KB", "status": "待构建", "files": "0"}.items():
            if key in self._stat_labels:
                self._stat_labels[key][0].setText(value)
        self._refresh_compact_epub_summary()

        self._code_view.clear()
        self._preview_title.setText("实时预览当前正文；生成后可从“打包文件”选择具体文件")
        self._stat_labels["status"][0].setStyleSheet("font-size: 16px; font-weight: bold;")
        self._refresh_document_summary()

    def _update_pills(self):
        mode = "竖排" if self._vert_radio.isChecked() else "横排"
        self._pills["mode"].setText(f"↕ {mode}")
        self._pills["source"].setText(f"🧾 {self._source_label}")
        if not self._doc:
            self._pills["cover"].setText("🖼 已设置封面" if self._effective_cover_path() else "🖼 封面未设置")
            self._pills["chapters"].setText("☰ 0 章")
            self._pills["images"].setText("🎨 0 图")
            self._pills["css"].setText("CSS 默认模板")
            self._pills["lang"].setText("🌐 ja · EPUB3")
            self._refresh_page_manager_cover_preview()
            return

        has_cover = bool(self._effective_cover_path())
        chapters = len(self._doc.toc) or sum(1 for b in self._doc.blocks if b.type == BlockType.CHAPTER and str(b.text or "").strip())
        images = len(self._doc.image_blocks())
        ai_css_active = (
            getattr(self._doc.metadata, "ai_processing_mode", "") == "typeset"
            and bool(getattr(self._doc.metadata, "ai_layout_locked", False))
            and bool(str(getattr(self._doc.metadata, "ai_epub_css", "") or "").strip())
        )
        css_label = "AI 排版 CSS" if ai_css_active else f"模板 {self._template_combo.currentText()}"
        self._pills["cover"].setText("🖼 已设置封面" if has_cover else "🖼 封面未设置")
        self._pills["chapters"].setText(f"☰ {chapters} 章")
        self._pills["images"].setText(f"🎨 {images} 图")
        self._pills["css"].setText(f"CSS {css_label}")
        language = str(getattr(self._doc.metadata, "language", "") or "ja")
        self._pills["lang"].setText(f"🌐 {language} · EPUB3")
        self._refresh_page_manager_cover_preview()

    def _switch_preview(self, which):
        self._code_view.setVisible(which == "code")
        self._preview_widget.setVisible(which == "preview")
        self._code_btn.setStyleSheet(
            f"background: {'#E4EEFF' if which == 'code' else '#F7F8FA'}; "
            f"color: {INK}; border: 1px solid #CDD3DA; border-radius: 6px; padding: 6px 12px;")
        self._preview_btn.setStyleSheet(
            f"background: {'#E4EEFF' if which == 'preview' else '#F7F8FA'}; "
            f"color: {INK}; border: 1px solid #CDD3DA; border-radius: 6px; padding: 6px 12px;")
        if which == "preview":
            if self._selected_preview_path and Path(self._selected_preview_path).exists():
                self._load_preview_file(self._selected_preview_path)
            else:
                self._render_live_preview()

    def _on_package_tree_click(self, index: QModelIndex):
        model = self._package_model
        if model is None or not index.isValid():
            return
        local = Path(model.filePath(index))
        if not local.is_file():
            return
        try:
            relative = local.relative_to(Path(self._preview_extract_dir)).as_posix()
        except Exception:
            relative = local.name
        self._preview_title.setText(relative)
        text_suffixes = {".xhtml", ".html", ".htm", ".css", ".opf", ".xml", ".ncx", ".txt", ".json"}
        if local.suffix.lower() in text_suffixes:
            self._code_view.setPlainText(local.read_text(encoding="utf-8", errors="replace"))
        else:
            self._code_view.setPlainText("（二进制文件；请切换到“预览”查看图片）")
        if local.suffix.lower() in {".xhtml", ".html", ".htm", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp"}:
            self._selected_preview_path = str(local)
            if self._preview_widget.isVisible():
                self._load_preview_file(str(local))
        else:
            self._selected_preview_path = ""

    def _schedule_live_preview(self, *_args):
        if hasattr(self, "_preview_timer"):
            self._selected_preview_path = ""
            if not self._workspace_active:
                self._preview_pending_while_hidden = True
                return
            self._preview_timer.start(120)

    def _set_preview_html(self, html_text: str, base_dir: str = ""):
        base_url = QUrl.fromLocalFile(str(Path(base_dir).resolve()) + os.sep) if base_dir else QUrl()
        if self._preview_is_webengine:
            self._preview_widget.setHtml(html_text, base_url)
        else:
            if base_dir:
                self._preview_widget.document().setBaseUrl(base_url)
            self._preview_widget.setHtml(html_text)

    def _load_preview_file(self, path: str):
        local = Path(path)
        if not local.exists():
            self._render_live_preview()
            return
        image_suffixes = {".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp"}
        if local.suffix.lower() in image_suffixes:
            uri = QUrl.fromLocalFile(str(local.resolve())).toString()
            html_text = ("<html><body style='margin:0;background:#eee;display:flex;"
                         "align-items:center;justify-content:center;min-height:100vh'>"
                         f"<img src='{uri}' style='max-width:100%;max-height:100vh;object-fit:contain'>"
                         "</body></html>")
            self._set_preview_html(html_text, str(local.parent))
            return
        if self._preview_is_webengine:
            self._preview_widget.load(QUrl.fromLocalFile(str(local.resolve())))
        else:
            try:
                html_text = local.read_text(encoding="utf-8", errors="replace")
                link = re.search(r'<link[^>]+href=["\']([^"\']+\.css)["\'][^>]*/?>', html_text, flags=re.IGNORECASE)
                if link:
                    css_path = (local.parent / link.group(1)).resolve()
                    if css_path.exists():
                        css = css_path.read_text(encoding="utf-8", errors="replace")
                        html_text = html_text[:link.start()] + f"<style>{css}</style>" + html_text[link.end():]
                self._preview_widget.document().setBaseUrl(QUrl.fromLocalFile(str(local.parent.resolve()) + os.sep))
                self._preview_widget.setHtml(html_text)
            except Exception as exc:
                self._preview_widget.setPlainText(f"预览失败：{exc}")

    def _render_live_preview(self):
        if not self._doc:
            self._set_preview_html("<html><body><p style='text-align:center;color:#777'>等待文档</p></body></html>")
            return
        try:
            from builder.epub_builder import resolve_epub_css, _block_to_xhtml, CSS_TEMPLATES, _esc
            template = self._template_combo.currentText()
            custom_css = None
            if template not in CSS_TEMPLATES:
                profile = FormatProfileStore().get_by_name(template)
                custom_css = profile.css if profile else None
            css, _source = resolve_epub_css(self._doc, template, custom_css)
            css += "\nbody{padding:2em;} section{max-width:48em;margin:auto;}"
            if not self._vert_radio.isChecked():
                css = re.sub(r"(?i)(?:-epub-)?writing-mode\s*:\s*[^;}]*(?:;)?", "", css)

            pieces = []
            text_count = 0
            for block in self._doc.blocks:
                if (block.metadata or {}).get("consumed") or block.type == BlockType.IMAGE_REF:
                    continue
                if not str(block.text or "").strip():
                    continue
                pieces.append(_block_to_xhtml(block))
                text_count += 1
                if text_count >= 80:
                    break
            if not pieces:
                pieces = ["<p class='normal'>暂无可预览正文</p>"]
            title = self._title_edit.text() or self._doc.metadata.title or "EPUB Preview"
            author = self._author_edit.text() or self._doc.metadata.author
            safe_title = _esc(title)
            safe_author = _esc(author)
            byline = f"<p class='normal' style='text-indent:0;opacity:.65'>{safe_author}</p>" if author else ""
            html_text = (
                "<!doctype html><html lang='ja'><head><meta charset='utf-8'>"
                f"<title>{safe_title}</title><style>{css}</style></head><body>"
                f"<section><h1>{safe_title}</h1>{byline}{''.join(pieces)}</section></body></html>"
            )
            self._set_preview_html(html_text)
        except Exception as exc:
            self._set_preview_html(f"<html><body><pre>实时预览失败：{exc}</pre></body></html>")

    def _paste_title_from_clipboard(self):
        title = QApplication.clipboard().text().strip()
        if title:
            self._title_edit.setText(title)
            self._title_edit.setFocus()
            self._title_edit.selectAll()
        else:
            notify(self, "剪贴板中没有可粘贴的书名。", "info")

    @staticmethod
    def _safe_epub_filename(title: str) -> str:
        name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", str(title or "")).strip(" .")
        name = re.sub(r"\s+", " ", name).strip()
        return (name[:160] or "未命名图书") + ".epub"

    def _cleanup_preview_dir(self):
        self._reset_package_browser()
        if self._preview_extract_dir:
            shutil.rmtree(self._preview_extract_dir, ignore_errors=True)
            self._preview_extract_dir = ""

    def set_workspace_active(self, active: bool) -> None:
        active = bool(active)
        if active == self._workspace_active:
            return
        self._workspace_active = active
        if not active:
            self._preview_timer.stop()
            return
        if self._preview_pending_while_hidden:
            self._preview_pending_while_hidden = False
            self._preview_timer.start(0)

    def shutdown_cleanup(self) -> None:
        self._workspace_active = False
        self._preview_timer.stop()
        self._build_generation += 1
        self._build_running = False
        self._cleanup_preview_dir()

    def _set_build_running(self, running: bool) -> None:
        self._build_running = bool(running)
        if hasattr(self, "_build_btn"):
            self._build_btn.setEnabled(not self._build_running)
            self._build_btn.setText("正在生成…" if self._build_running else "生成 EPUB")
        if hasattr(self, "_refresh_btn"):
            self._refresh_btn.setEnabled(not self._build_running)

    def set_page_manager_assets(self, page_images, page_overrides) -> None:
        """Mirror Page Manager's currently visible cover into EPUB immediately."""
        images = tuple(str(Path(p)) for p in (page_images or []) if str(p or ""))
        overrides = {}
        for key, value in dict(page_overrides or {}).items():
            try:
                page_no = int(key)
            except (TypeError, ValueError):
                continue
            overrides[page_no] = str(getattr(value, "value", value) or "")
        self._page_manager_images = images
        self._page_manager_overrides = overrides
        cover_path = ""
        for page_no in sorted(overrides):
            if overrides.get(page_no) == "cover" and 1 <= page_no <= len(images):
                cover_path = images[page_no - 1]
                break
        self._page_manager_cover_path = cover_path
        self._refresh_page_manager_cover_preview()
        self._refresh_document_summary()

    def _effective_cover_path(self) -> str:
        if self._page_manager_cover_path:
            return self._page_manager_cover_path
        if self._doc is not None:
            for page in getattr(self._doc, "pages", []) or []:
                if page.page_type == BlockType.COVER and str(page.image_path or ""):
                    return str(page.image_path)
        return ""

    def _refresh_page_manager_cover_preview(self) -> None:
        label = getattr(self, "_compact_cover_preview", None)
        if label is None:
            return
        path = self._effective_cover_path()
        if path and Path(path).is_file():
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                label.setPixmap(pixmap.scaled(150, 210, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                label.setText("")
                label.setStyleSheet(f"background:{CARD};border:1px solid {BORDER};border-radius:8px;")
                label.setToolTip(f"页面管理封面：{Path(path).name}")
                return
        label.setPixmap(QPixmap())
        label.setText("封面\n未设置")
        label.setStyleSheet(
            f"background:{ACC};color:white;border-radius:8px;font-size:16px;font-weight:700;line-height:1.4;"
        )
        label.setToolTip("页面管理尚未设置封面")

    def set_project_export_dir(self, path: str | Path | None) -> None:
        self._project_export_dir = str(Path(path).expanduser().resolve()) if path else ""
        if self._project_export_dir:
            Path(self._project_export_dir).mkdir(parents=True, exist_ok=True)

    def _build_epub(self):
        if self._build_running:
            notify(self, "当前已有一个 EPUB 构建任务，请等待完成后再生成。", "warning")
            return
        # EPUB always builds from a private immutable snapshot. Page Manager is
        # merged as an overlay first, then metadata edits and the background
        # builder operate only on the deep copy. No title/author/export change
        # can leak back into Formatter, OCR comparison, or MainWindow.
        doc = self._doc
        main_win = self.window()

        # Only fall back when the EPUB workspace has never received a document;
        # never replace an explicitly selected AI/document version.
        if doc is None and main_win is not None:
            doc = getattr(main_win, "_doc", None)

        # Page classifications are a lightweight independent overlay. Sync them
        # into a copy before taking the final build snapshot.
        if doc is not None and main_win is not None and hasattr(main_win, "_document_with_current_page_assets"):
            doc, asset_report = main_win._document_with_current_page_assets(doc, copy_document=True)
            if asset_report is not None and hasattr(self, "_status"):
                self._status.setText(
                    f"构建前已同步页面管理：{asset_report.managed_pages} 页，"
                    f"{asset_report.image_pages} 个图片页。"
                )

        if doc is None:
            QMessageBox.warning(self, "错误", "请先完成 OCR 和格式处理")
            return

        doc = copy.deepcopy(doc)

        from utils.publication_preflight import inspect_document_for_publication
        preflight = inspect_document_for_publication(doc)
        critical = preflight.critical_messages
        warnings = preflight.warning_messages
        if critical or warnings:
            lines = [f"• {message}" for message in critical]
            lines.extend(f"◇ {message}" for message in warnings)
            risky = [issue for issue in preflight.text_issues if issue.severity in {"high", "medium"}]
            if risky:
                lines.append("")
                lines.append("部分正文问题：")
                for issue in risky[:8]:
                    lines.append(f"  第 {issue.block_index + 1} 块：{issue.message} · {issue.excerpt}")
                if len(risky) > 8:
                    lines.append(f"  ……另有 {len(risky) - 8} 处，请在文字校对中复核。")
            lines.append("")
            if preflight.translation_ready:
                lines.append("翻译可用性：通过结构检查；剩余问题预计不影响主要句意。")
            else:
                lines.append(
                    f"翻译可用性：未通过。仍有 {preflight.translation_blocker_count} 处"
                    "可能改变句意、漏文或说话人归属的问题。"
                )
            answer = QMessageBox.warning(
                self,
                "翻译/发布前检查发现问题",
                "\n".join(lines) + "\n\n建议返回文字校对逐项处理。是否仍然继续导出？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        img_count = sum(1 for b in doc.blocks if b.type == BlockType.IMAGE_REF)

        suggested_title = self._title_edit.text().strip() or str(doc.metadata.title or "").strip()
        suggested_name = self._safe_epub_filename(suggested_title)
        suggested_path = (
            str(Path(self._project_export_dir) / suggested_name)
            if self._project_export_dir else suggested_name
        )
        path, _ = QFileDialog.getSaveFileName(self, "保存 EPUB", suggested_path, "EPUB (*.epub)")
        if not path:
            return
        if not path.lower().endswith(".epub"):
            path += ".epub"
        self._build_generation += 1
        build_generation = self._build_generation
        document_generation = self._document_generation
        self._set_build_running(True)

        title = self._title_edit.text().strip() or Path(path).stem
        if not self._title_edit.text().strip():
            self._title_edit.setText(title)
        doc.metadata.title = title
        doc.metadata.author = self._author_edit.text().strip()
        doc.metadata.publisher = self._publisher_edit.text().strip()
        doc.metadata.volume = self._volume_edit.text().strip()
        if hasattr(self, "_compact_language_combo"):
            doc.metadata.language = str(self._compact_language_combo.currentData() or doc.metadata.language or "ja")

        self._update_pills()
        self._stat_labels["status"][0].setText("构建中…")
        self._prog.setVisible(True)
        self._prog.setRange(0, 0)

        template = self._template_combo.currentText()
        vertical = self._vert_radio.isChecked()

        def worker():
            try:
                import io, contextlib
                from builder.epub_builder import build_epub, CSS_TEMPLATES
                custom_css = None
                if template not in CSS_TEMPLATES:
                    # 下拉框里内置模板之外的名字都是自定义 Format Profile
                    profile = FormatProfileStore().get_by_name(template)
                    custom_css = profile.css if profile else None
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    build_epub(doc, output_path=path, css_template=template,
                              vertical=vertical, verbose=True, custom_css=custom_css)
                ai_css_path = ""
                if (getattr(doc.metadata,"ai_processing_mode","") == "typeset"
                        and bool(getattr(doc.metadata,"ai_layout_locked",False))
                        and bool(str(getattr(doc.metadata,"ai_epub_css","") or "").strip())):
                    candidate = Path(path).with_name(f"{Path(path).stem}.ai-typeset.css")
                    if candidate.exists():
                        ai_css_path = str(candidate)
                # Always-on internal quality gate first.  Unlike Java epubcheck
                # this has no external runtime dependency and therefore guards
                # every build.  Structural failures are fatal; warnings are
                # surfaced together with epubcheck in the completion dialog.
                from utils.epub_quality_gate import validate_epub_quality
                quality_gate = validate_epub_quality(path, expect_vertical=vertical)
                if not quality_gate.valid:
                    detail = "\n".join(quality_gate.errors[:20])
                    raise RuntimeError(
                        quality_gate.summary() + (f"\n{detail}" if detail else "")
                    )
                # 构建后 EPUBCheck 校验。未安装校验器时允许跳过；一旦
                # 实际运行，任何 ERROR/FATAL 都作为发布失败，避免界面把
                # “生成了 ZIP”误报成“出版包合规”。
                from utils.epub_validator import validate_epub
                check = validate_epub(path)
                if not check.skipped and not check.valid:
                    detail = "\n".join((check.errors + check.warnings)[:20])
                    raise RuntimeError(
                        check.summary() + (f"\n{detail}" if detail else "")
                    )
                project_copy_path = ""
                project_export_dir = str(self._project_export_dir or "").strip()
                if project_export_dir:
                    export_root = Path(project_export_dir)
                    export_root.mkdir(parents=True, exist_ok=True)
                    target = export_root / Path(path).name
                    try:
                        if target.resolve() != Path(path).resolve():
                            shutil.copy2(path, target)
                        project_copy_path = str(target.resolve())
                    except Exception:
                        project_copy_path = ""
                signals.finished.emit({
                    "path": path,
                    "project_copy_path": project_copy_path,
                    "log": buf.getvalue(),
                    "ai_css_path": ai_css_path,
                    "generation": build_generation,
                    "document_generation": document_generation,
                    "chapters": len(doc.toc),
                    "images": len(doc.image_blocks()),
                    "epubcheck": check,
                    "quality_gate": quality_gate,
                })
            except Exception as e:
                import traceback
                signals.error.emit(traceback.format_exc())

        signals = self._build_signals = WorkerSignals()
        signals.finished.connect(self._on_build_done)
        signals.error.connect(
            lambda message, generation=build_generation:
                self._on_build_error(message, generation)
        )
        threading.Thread(target=worker, daemon=True).start()

    def _on_build_done(self, info):
        generation = int(info.get("generation", -1))
        if generation != self._build_generation:
            return
        self._set_build_running(False)
        self._prog.setVisible(False)
        path = info["path"]
        if int(info.get("document_generation", -1)) != self._document_generation:
            self._stat_labels["status"][0].setText("待构建")
            notify(
                self,
                f"EPUB 已从启动任务时的独立正文快照生成：\n{path}\n"
                "构建期间当前正文已经切换，因此未把旧包预览安装到新书工作区。",
                "warning",
            )
            return

        # 更新统计；使用启动构建时的快照，避免期间切换到第二本书后串书。
        size_kb = Path(path).stat().st_size // 1024
        self._stat_labels["chapters"][0].setText(str(info.get("chapters", 0)))
        self._stat_labels["images"][0].setText(str(info.get("images", 0)))
        self._stat_labels["size"][0].setText(f"{size_kb} KB")
        self._stat_labels["status"][0].setText("✓ 完成")
        self._stat_labels["status"][0].setStyleSheet(f"font-size: 16px; font-weight: bold; color: {SUCCESS};")
        self._refresh_compact_epub_summary()
        for name, label in getattr(self, "_compact_checks", {}).items():
            if name != "EPUBCheck":
                label.setText("通过")
                color = SUCCESS
            else:
                report = info.get("epubcheck")
                if report is None or getattr(report, "skipped", False):
                    label.setText("未运行")
                    color = MUTED
                elif getattr(report, "valid", False):
                    label.setText("通过")
                    color = SUCCESS
                else:
                    label.setText("失败")
                    color = DANGER
            label.setStyleSheet(f"color:{color};font-size:10px;font-weight:700;")

        # 显示结构树
        self._show_tree(path)
        css_note = f"\nAI 排版 CSS 已保存：\n{info['ai_css_path']}" if info.get("ai_css_path") else ""
        gate = info.get("quality_gate")
        check = info.get("epubcheck")
        check_note = ""
        if gate is not None:
            check_note += f"\n{gate.summary()}"
            if getattr(gate, "warnings", None):
                check_note += "\n" + "\n".join(list(gate.warnings)[:20])
        if check is not None:
            check_note += f"\n{check.summary()}"
            if not check.skipped and (check.errors or check.warnings):
                detail = "\n".join((check.errors + check.warnings)[:20])
                check_note += f"\n{detail}"
        project_note = ""
        project_copy = str(info.get("project_copy_path") or "")
        if project_copy and Path(project_copy).resolve() != Path(path).resolve():
            project_note = f"\n项目副本：\n{project_copy}"
        notify(self, f"EPUB 已生成:\n{path}\n({size_kb} KB){project_note}{css_note}{check_note}", "success")
        self.epub_built.emit(dict(info))

    def _on_build_error(self, msg, generation: int | None = None):
        if generation is not None and int(generation) != self._build_generation:
            return
        self._set_build_running(False)
        self._prog.setVisible(False)
        self._stat_labels["status"][0].setText("✗ 失败")
        self._stat_labels["status"][0].setStyleSheet(f"font-size: 16px; font-weight: bold; color: {DANGER};")
        self._refresh_compact_epub_summary()
        for label in getattr(self, "_compact_checks", {}).values():
            label.setText("失败")
            label.setStyleSheet(f"color:{DANGER};font-size:10px;font-weight:700;")
        show_error_dialog(self, "生成失败", msg)

    def _show_tree(self, epub_path):
        """Extract into a fresh directory and browse it through QFileSystemModel.

        QFileSystemModel owns the hierarchy and never constructs Python lists of
        QTreeWidgetItem objects, which removes the consecutive-book PySide6
        ``list assignment index out of range`` failure path.
        """
        self._selected_preview_path = ""
        self._cleanup_preview_dir()
        try:
            from utils.epub_inspector import inspect_epub_archive
            preview_dir = tempfile.mkdtemp(prefix="novel_formatter_epub_preview_")
            try:
                inspection = inspect_epub_archive(epub_path, preview_dir)
            except Exception:
                shutil.rmtree(preview_dir, ignore_errors=True)
                raise
            self._preview_extract_dir = inspection.extract_dir
            self._stat_labels["files"][0].setText(str(len(inspection.names)))
            self._install_package_model(self._preview_extract_dir)
            self._left_tabs.setCurrentIndex(1)
            self._preview_title.setText(f"已载入 {Path(epub_path).name}；选择文件查看源码或预览")
        except Exception as exc:
            import traceback
            print("[EPUB PREVIEW TREE ERROR]", traceback.format_exc())
            self._reset_package_browser(f"EPUB 已生成，但预览读取失败：{exc}")

