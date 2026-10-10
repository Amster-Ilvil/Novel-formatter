from __future__ import annotations

import copy
import html
import json
import shutil
import threading
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QFrame, QLabel, QPushButton,
    QPlainTextEdit, QTextEdit, QProgressBar, QScrollArea, QSplitter, QToolButton, QCheckBox, QLineEdit,
    QListWidget, QListWidgetItem, QFileDialog, QMessageBox, QInputDialog, QSizePolicy, QMenu,
    QComboBox,
)
from PySide6.QtCore import Qt, Signal, QTimer, QUrl
from PySide6.QtGui import (
    QCursor, QDesktopServices, QKeySequence, QShortcut, QPainter, QFont, QFontMetrics,
    QPalette, QTextOption, QTextBlockFormat, QTextCharFormat, QTextCursor,
)

from models.document import UnifiedDocument, BlockType, new_temp_repo_path
from models.format_profile import FormatProfile, FormatProfileStore
from ui.flow_layout import FlowLayout
from ui.localized_dialogs import ui_message
from ui.responsive import preserve_button_text
from ui.dialogs import show_error_dialog
from ui.settings.ai_dialog import AISettingsDialog, ensure_ai_settings
from ui.common.editor_controls import MouseWheelPlainTextEdit, EditorPositionSlider, configure_drag_scrollbar, attach_editor_position_slider
from ui.common.signals import WorkerSignals
from ui.common.toast import notify
from ui.common.styling import (
    BG,
    ACC, ACC_BG, BORDER, CARD, CLICKABLE_BG, DANGER, INK, MUTED, SUCCESS, LIGHT_PREVIEW_STYLE,
    accent_button, link_button, blend, make_separator, wrap_in_card,
)
from utils.async_generation import GenerationGuard
from utils.clear_manager import ClearManager, create_workspace_clear_button
from adapters.result_export import (
    ensure_export_extension, export_text_result, format_from_filter, safe_result_filename, save_filter_string,
)
from ui.formatter.contracts import FORMATTER_STEPS, LOW_LEVEL_FORMATTER_STEPS, FORMATTER_RULES
from core.batch_changes import BatchChange, BatchChangeSet, BatchRestoreGuard, content_state_token
from ui.batch_change_dialog import BatchChangePreviewDialog
from ui.design.components import DesignSwitch


class _FormatterBookPreview(QTextEdit):
    """Scrollable book-style Formatter preview with block-level formatting."""

    _SAMPLE = "人間というものを見た。\n記憶している。\nニャーニャー泣いていた事だけは\n何でも薄暗いじめじめした所で\nどこで生れたかとんと見當がつかぬ。\n吾輩は猫である。名前はまだ無い。"
    # Multiple Formatter panes often render the same book.  Font shaping is
    # substantially more expensive than the percentile arithmetic, so share a
    # tiny bounded cache of the representative sample's measured width.  This
    # caches only a visual width estimate; text and layout remain independent.
    _WIDTH_CACHE: OrderedDict[tuple, int] = OrderedDict()
    _WIDTH_CACHE_LIMIT = 32

    def __init__(self, parent=None):
        super().__init__(parent)
        self._text = ""
        self._content_signature = None
        self._vertical = False
        self.setReadOnly(True)
        self.setAcceptRichText(False)
        self.setUndoRedoEnabled(False)
        self.setLineWrapMode(QTextEdit.WidgetWidth)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMinimumSize(320, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Keep the reading measure bounded, but let document content set a
        # useful minimum width.  With only a small size hint, the preview row
        # could collapse to a narrow strip even though the card had room.
        self._reading_width = 1320
        self.setMaximumWidth(self._reading_width)
        self.setStyleSheet(
            "QTextEdit{border:1px solid #E1E7EF;border-radius:8px;"
            "background:#FFFEFC;color:#172033;"
            "padding:28px 34px;font-size:16px;selection-background-color:#D7E7FF;}"
        )

    def set_text(self, text: str) -> None:
        value = str(text or "").replace("\r", "")
        signature = ("text", value)
        if signature == self._content_signature:
            return
        bar = self.verticalScrollBar()
        old_max = max(1, bar.maximum())
        old_ratio = bar.value() / old_max
        self._text = value
        self._content_signature = signature
        display_text = value.strip() or self._SAMPLE
        rows = []
        for line in display_text.splitlines():
            text_line = line.strip()
            if not text_line:
                continue
            if text_line.startswith("▌"):
                rows.append((BlockType.CHAPTER, text_line.lstrip("▌　 ").strip()))
            elif text_line.startswith("──"):
                rows.append((BlockType.SECTION, text_line.strip("─　 ")))
            elif text_line.startswith("[图片:"):
                rows.append((BlockType.IMAGE_REF, text_line))
            elif text_line.startswith("[ruby]"):
                rows.append((BlockType.RUBY, text_line.removeprefix("[ruby]").strip()))
            elif text_line.startswith(("「", "『", "“", "\"")):
                rows.append((BlockType.DIALOGUE, text_line))
            else:
                rows.append((BlockType.PARAGRAPH, text_line))
        self._render_blocks(rows)
        self._update_reading_width([line for _, line in rows])
        QTimer.singleShot(0, lambda ratio=old_ratio: self._restore_scroll_ratio(ratio))

    def set_document(self, doc: Optional[UnifiedDocument]) -> None:
        """Render document blocks with reading styles while preserving scroll."""
        if doc is None:
            self.set_text("")
            return
        blocks = []
        for block in doc.blocks:
            metadata = block.metadata or {}
            if metadata.get("consumed") or block.type == BlockType.HEADER_FOOTER:
                continue
            if block.type == BlockType.IMAGE_REF:
                label = f"[图片: {Path(block.image_path).name}]" if block.image_path else block.text
            else:
                label = str(block.text or "")
            label = label.strip(" \t\r\n\u3000")
            if label:
                blocks.append((block.type, label))
        signature = ("document", tuple((str(kind), text) for kind, text in blocks))
        if signature == self._content_signature:
            return
        bar = self.verticalScrollBar()
        old_max = max(1, bar.maximum())
        old_ratio = bar.value() / old_max
        self._content_signature = signature
        self._text = "\n".join(text for _, text in blocks)
        self._render_blocks(blocks)
        self._update_reading_width([text for kind, text in blocks if kind not in {
            BlockType.CHAPTER, BlockType.SECTION
        }])
        QTimer.singleShot(0, lambda ratio=old_ratio: self._restore_scroll_ratio(ratio))

    def _render_blocks(self, blocks) -> None:
        """Render the read-only book preview in one Qt rich-text transaction.

        Building thousands of QTextBlocks one-by-one through QTextCursor forces
        repeated document-layout bookkeeping.  Qt's HTML importer creates the
        same paragraph structure in one pass and is substantially faster on
        full novels.  All source text is escaped before insertion, so OCR/book
        content can never become active markup.
        """
        style = (
            "p{white-space:pre-wrap;line-height:155%;margin-top:0;"
            "margin-bottom:14px;text-indent:16px;}"
            ".chapter{text-align:center;font-size:19pt;font-weight:700;"
            "margin-top:24px;margin-bottom:22px;text-indent:0;}"
            ".section{text-align:center;font-size:16pt;font-weight:600;"
            "margin-top:18px;margin-bottom:16px;text-indent:0;}"
            ".dialogue{margin-left:32px;margin-right:12px;margin-bottom:10px;text-indent:0;}"
            ".ruby{font-size:14pt;color:#555;margin-bottom:10px;}"
            ".image{text-align:center;font-style:italic;color:#555;"
            "margin-top:10px;margin-bottom:10px;text-indent:0;}"
            ".small{font-size:14pt;margin-bottom:8px;}"
        )
        parts = [f"<html><head><style>{style}</style></head><body>"]
        for kind, text in blocks:
            clean_text = str(text).strip(" \t\r\n\u3000")
            if kind == BlockType.CHAPTER:
                css_class = "chapter"
            elif kind == BlockType.SECTION:
                css_class = "section"
            elif kind == BlockType.DIALOGUE:
                css_class = "dialogue"
            elif kind == BlockType.RUBY:
                css_class = "ruby"
            elif kind == BlockType.IMAGE_REF:
                css_class = "image"
            elif kind in {BlockType.FOOTNOTE, BlockType.TOC_ENTRY}:
                css_class = "small"
            else:
                css_class = "normal"
            parts.append(f'<p class="{css_class}">{html.escape(clean_text)}</p>')
        parts.append("</body></html>")
        self.setHtml("".join(parts))
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.Start)
        self.setTextCursor(cursor)
        self.document().setModified(False)

    def _update_reading_width(self, content: list[str]) -> None:
        """Size the page from a bounded representative sample.

        Font shaping every paragraph is surprisingly expensive on a 4k-block
        novel and this value is only a visual reading-width estimate.  Uniform
        sampling preserves the same 80th-percentile policy while bounding GUI
        work regardless of book length.
        """
        values = [text for text in content if text.strip()]
        sample_limit = 256
        if len(values) > sample_limit:
            last = len(values) - 1
            values = [
                values[round(index * last / (sample_limit - 1))]
                for index in range(sample_limit)
            ]
        metric = self.fontMetrics()
        font = self.font()
        sample_key = tuple(text[:160] for text in values)
        cache_key = (
            font.family(),
            round(float(font.pointSizeF()), 3),
            int(font.weight()),
            bool(font.italic()),
            bool(font.underline()),
            sample_key,
        )
        cache = type(self)._WIDTH_CACHE
        typical = cache.get(cache_key)
        if typical is None:
            widths = sorted(metric.horizontalAdvance(text) for text in sample_key)
            if widths:
                typical = widths[min(len(widths) - 1, int((len(widths) - 1) * 0.8))]
            else:
                typical = 0
            cache[cache_key] = int(typical)
            cache.move_to_end(cache_key)
            while len(cache) > type(self)._WIDTH_CACHE_LIMIT:
                cache.popitem(last=False)
        else:
            cache.move_to_end(cache_key)
        target = max(760, min(self._reading_width, int(typical + 130)))
        parent_width = self.parentWidget().contentsRect().width() if self.parentWidget() else 0
        available_width = max(parent_width, self.width())
        if available_width > 0:
            target = min(target, max(320, available_width - 40))
        self.setMinimumWidth(target)

    def _restore_scroll_ratio(self, ratio: float) -> None:
        bar = self.verticalScrollBar()
        bar.setValue(int(max(0.0, min(1.0, float(ratio))) * bar.maximum()))

    def set_vertical(self, vertical: bool) -> None:
        # Kept for the historical compact-preview contract.  The compact UI is
        # intentionally horizontal; vertical EPUB layout is previewed in EPUB.
        self._vertical = bool(vertical)



class FormatterTab(QWidget):
    doc_formatted = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._original_doc: Optional[UnifiedDocument] = None
        self._ocr_doc: Optional[UnifiedDocument] = None
        self._fmt_doc: Optional[UnifiedDocument] = None
        self._active_step = "split_embedded_titles"
        self._scroll_shortcuts: list[QShortcut] = []
        self._formatter_ai_running = False
        self._formatter_pipeline_running = False
        self._formatter_ai_cancel_requested = False
        self._formatter_ai_cancel_event = threading.Event()
        self._content_generation = GenerationGuard()
        self._formatter_ai_buttons: list[QPushButton] = []
        self._formatter_action_buttons: list[QPushButton] = []
        self._formatter_source_path: str = ""
        self._formatter_ai_checkpoint_root: str = ""
        self._formatter_checkpoint_override: str = ""
        self._last_manual_batch_restore: dict | None = None
        # The advanced editors contain the full book text and are hidden in the
        # default compact workspace.  Hydrate them only when the user actually
        # opens the advanced view; this keeps ordinary book loading responsive.
        self._advanced_editors_dirty = True
        self._hydrating_advanced_editors = False
        self._compact_preview_dirty = True
        self._pdf_preview_dirty = True
        self._build()

    def _build(self):
        root = wrap_in_card(self)

        # ── 左侧：精简步骤选择 ────────────────────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(148)
        left.setMaximumWidth(188)
        left.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        left.setStyleSheet(f"background: {BG};")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)

        self._show_rule_details_cb = QCheckBox("显示具体规则说明")
        self._show_rule_details_cb.setToolTip("显示当前所选步骤的具体判断与处理规则；阅读顺序、清理模块等底层步骤始终隐藏")
        self._show_rule_details_cb.setChecked(False)
        self._show_rule_details_cb.setContentsMargins(12, 10, 12, 8)
        ll.addWidget(self._show_rule_details_cb)

        self._step_checks: dict[str, QCheckBox] = {}
        self._step_cards: dict[str, QWidget] = {}
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        scroll.setStyleSheet("border: none;")
        steps_widget = QWidget()
        steps_widget.setStyleSheet(f"background: {BG};")
        sl = QVBoxLayout(steps_widget)
        sl.setContentsMargins(8, 8, 8, 8)
        sl.setSpacing(4)

        for sid, label, badge_text, desc in FORMATTER_STEPS:
            # 所有规则默认关闭。底层步骤仍保留在程序中，但只有显式选择
            # 对应流程时才执行，不再在打开 Formatter 后自动改写文本。
            cb = QCheckBox()
            cb.setChecked(False)
            self._step_checks[sid] = cb
            if sid in LOW_LEVEL_FORMATTER_STEPS:
                continue

            card = QWidget()
            card.setObjectName("stepCard")
            card.setCursor(QCursor(Qt.PointingHandCursor))
            card.setMinimumHeight(36)
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            # 样式限定到卡片本身（#stepCard），避免级联到内部复选框/文字
            card.setAttribute(Qt.WA_Hover, True)
            card.setStyleSheet(
                f"QWidget#stepCard {{ background: {CLICKABLE_BG}; border: 1px solid #E2E5E9; border-radius: 8px; }}"
                f"QWidget#stepCard:hover {{ border-color: #CDD3DA; background: #E4EEFF; }}")
            cl = QHBoxLayout(card)
            cl.setContentsMargins(8, 5, 8, 5)
            cl.setSpacing(6)

            cl.addWidget(cb)
            nlbl = QLabel(label)
            nlbl.setStyleSheet("font-size: 12px;")
            cl.addWidget(nlbl, 1)

            card.mousePressEvent = partial(self._select_step, sid)
            # 所有可见规则卡片等比例占满列表高度；窗口变矮时仍由滚动区承接。
            sl.addWidget(card, 1)
            self._step_cards[sid] = card

        scroll.setWidget(steps_widget)
        ll.addWidget(scroll, 1)

        # ── 高级模式开关：默认全部关闭，集中放在步骤列表下方 ────────────────
        ll.addWidget(make_separator())

        self._preserve_ocr_layout_cb = QCheckBox("固定原 OCR 排版")
        self._preserve_ocr_layout_cb.setToolTip("保留导入或 OCR 产生的原始块、段落及分页结构；适合希望手动处理版式时使用")
        self._preserve_ocr_layout_cb.setContentsMargins(12, 8, 12, 4)
        ll.addWidget(self._preserve_ocr_layout_cb)

        self._pdf_text_layer_mode_cb = QCheckBox("PDF文字层模式")
        self._pdf_text_layer_mode_cb.setToolTip(
            "仅处理可选择文字的PDF提取结果：启用词中换列、跨页强接续、"
            "提前闭引号回收、双人同时发言引号修复和资源占位符隔离。\n"
            "普通Apple Vision/Paddle等图片OCR不会使用这些规则。"
        )
        self._pdf_text_layer_mode_cb.setContentsMargins(12, 0, 12, 5)
        self._pdf_text_layer_mode_cb.toggled.connect(self._on_pdf_text_layer_mode_toggled)
        ll.addWidget(self._pdf_text_layer_mode_cb)

        self._pdf_keep_afterwords_cb = QCheckBox("保留作者前书/后记")
        self._pdf_keep_afterwords_cb.setChecked(False)
        self._pdf_keep_afterwords_cb.setEnabled(False)
        self._pdf_keep_afterwords_cb.setToolTip(
            "默认关闭。勾选后 PDF 文字层保留真实作者前书/后记；"
            "不勾选时允许删除“数字（前書/後書き）”段落。"
        )
        self._pdf_keep_afterwords_cb.setContentsMargins(28, 0, 12, 10)
        self._pdf_keep_afterwords_cb.toggled.connect(lambda _checked: self._apply_mode_to_documents())
        ll.addWidget(self._pdf_keep_afterwords_cb)
        ll.addSpacing(10)

        root.addWidget(left, 0)

        # ── 右侧：工具栏 + 对比区 ─────────────────────────────────────────────
        right = QWidget()
        right.setMinimumWidth(520)
        right.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right.setStyleSheet(f"background: {BG};")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)

        # 工具栏
        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(14, 8, 14, 8)
        toolbar_layout.setSpacing(10)

        toolbar_left = QWidget()
        tbl = FlowLayout(toolbar_left, hspacing=8, vspacing=8)
        tbl.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.addWidget(toolbar_left, 1)

        toolbar_actions = QWidget()
        toolbar_actions_layout = QHBoxLayout(toolbar_actions)
        toolbar_actions_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_actions_layout.setSpacing(8)

        self._step_title = QLabel("")
        self._step_title.setStyleSheet("font-weight: bold; font-size: 13px;")
        tbl.addWidget(self._step_title)

        load_btn = QPushButton("📂 导入 JSON / MD / TEI")
        load_btn.setProperty("flat", True)
        preserve_button_text(load_btn)
        load_btn.clicked.connect(self._load_json)
        tbl.addWidget(load_btn)

        docx_btn = QPushButton("📄 导入 DOCX")
        docx_btn.setProperty("flat", True)
        preserve_button_text(docx_btn)
        docx_btn.clicked.connect(self._import_docx)
        tbl.addWidget(docx_btn)

        epub_btn = QPushButton("📖 导入 EPUB")
        epub_btn.setProperty("flat", True)
        preserve_button_text(epub_btn)
        epub_btn.clicked.connect(self._import_epub)
        tbl.addWidget(epub_btn)

        # OCR 配准由文字校对工作区独立处理。
        # Formatter 这里只保留格式处理、预览、手动编辑与导出。

        save_btn = QPushButton("💾 保存结果")
        save_btn.setProperty("flat", True)
        preserve_button_text(save_btn)
        save_btn.setToolTip("可选择保存为 Word、JSON、Markdown 或纯文本")
        save_btn.clicked.connect(self._save_result)
        tbl.addWidget(save_btn)

        undo_btn = QPushButton("↩ 撤销")
        undo_btn.setProperty("flat", True)
        preserve_button_text(undo_btn)
        undo_btn.clicked.connect(self._undo)
        tbl.addWidget(undo_btn)

        self._restore_batch_btn = QPushButton("⟲ 恢复批量")
        self._restore_batch_btn.setProperty("flat", True)
        preserve_button_text(self._restore_batch_btn)
        self._restore_batch_btn.setToolTip("恢复最近一次通过预览确认的批量编辑；如果之后又修改过正文则拒绝恢复。")
        self._restore_batch_btn.clicked.connect(self._restore_last_manual_batch)
        self._restore_batch_btn.setEnabled(False)
        tbl.addWidget(self._restore_batch_btn)

        edit_apply_btn = QPushButton("📝 应用编辑")
        edit_apply_btn.setProperty("flat", True)
        preserve_button_text(edit_apply_btn)
        edit_apply_btn.setToolTip("把处理前/处理后文本框里的手动修改写回当前文档")
        edit_apply_btn.clicked.connect(self._apply_manual_edit)
        tbl.addWidget(edit_apply_btn)

        self._version_lbl = QLabel("")
        self._version_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px; padding: 0 8px;")
        tbl.addWidget(self._version_lbl)

        apply_btn = accent_button("✓ 应用", color=SUCCESS)
        preserve_button_text(apply_btn)
        apply_btn.clicked.connect(self._apply_step)
        toolbar_actions_layout.addWidget(apply_btn)

        run_btn = accent_button("▶  全部运行")
        preserve_button_text(run_btn)
        run_btn.clicked.connect(self._run_all)
        toolbar_actions_layout.addWidget(run_btn)
        self._formatter_action_buttons = [
            load_btn, docx_btn, epub_btn, undo_btn, self._restore_batch_btn, edit_apply_btn, apply_btn, run_btn,
        ]
        toolbar_layout.addWidget(toolbar_actions, 0, Qt.AlignRight | Qt.AlignVCenter)

        rl.addWidget(toolbar)
        rl.addWidget(make_separator())

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setMinimumHeight(22)
        self._progress.setTextVisible(True)
        self._progress.setFormat("%p%")
        rl.addWidget(self._progress)

        # 前后对比
        compare_hdr = QHBoxLayout()
        compare_hdr.setContentsMargins(14, 8, 14, 4)
        compare_hdr.addWidget(QLabel("处理前"))
        compare_hdr.addStretch()
        compare_hdr.addWidget(QLabel("处理后"))
        compare_hdr.addStretch()
        rl.addLayout(compare_hdr)

        splitter = QSplitter(Qt.Horizontal)
        self._before = MouseWheelPlainTextEdit()
        self._before.setReadOnly(False)
        self._before.setUndoRedoEnabled(True)
        self._before.setPlaceholderText("处理前文本（可编辑，点击「应用编辑」写回）")
        self._before.setStyleSheet(LIGHT_PREVIEW_STYLE)
        self._after = MouseWheelPlainTextEdit()
        self._after.setReadOnly(False)
        self._after.setUndoRedoEnabled(True)
        self._after.setPlaceholderText("处理后文本（可编辑，点击「应用编辑」写回）")
        self._after.setStyleSheet(LIGHT_PREVIEW_STYLE)
        splitter.addWidget(self._make_editor_pane(self._before))
        splitter.addWidget(self._make_editor_pane(self._after))
        self._install_scroll_shortcuts(self._before)
        self._install_scroll_shortcuts(self._after)
        rl.addWidget(splitter, 1)

        # 底部操作区：本地规则与 Formatter 专用 AI 协同。
        result_bar = QWidget()
        result_bar.setStyleSheet("background: #F7F8FA; border-top: 1px solid #E2E5E9;")
        # 两行布局：状态提示独占一行，操作按钮独占一行——彻底避免窄窗口下按钮被挤压截断
        result_layout = QVBoxLayout(result_bar)
        result_layout.setContentsMargins(14, 6, 14, 8)
        result_layout.setSpacing(4)
        self._formatter_ai_status = QLabel("本地规则处理短文本；Formatter AI 负责长文本和复杂结构")
        self._formatter_ai_status.setStyleSheet(f"color: {MUTED}; font-size: 11px; border: none;")
        result_layout.addWidget(self._formatter_ai_status)
        result_actions = QWidget()
        al = QHBoxLayout(result_actions)
        al.setContentsMargins(0, 0, 0, 0)
        al.setSpacing(8)
        clear_after_btn = QPushButton("清空处理后")
        clear_after_btn.setProperty("flat", True)
        preserve_button_text(clear_after_btn)
        clear_after_btn.setToolTip("只清空右侧处理后文本框和当前处理结果，不影响处理前内容")
        clear_after_btn.clicked.connect(self._clear_after_result)
        al.addWidget(clear_after_btn)
        reload_after_btn = QPushButton("重新载入处理前")
        reload_after_btn.setProperty("flat", True)
        preserve_button_text(reload_after_btn)
        reload_after_btn.setToolTip("把左侧处理前文档重新载入到右侧处理后文本框")
        reload_after_btn.clicked.connect(self._reload_before_into_after)
        al.addWidget(reload_after_btn)
        self._formatter_action_buttons.extend([clear_after_btn, reload_after_btn])

        ai_settings_btn = QPushButton("⚙ AI设置")
        ai_settings_btn.setProperty("flat", True)
        ai_settings_btn.setToolTip("使用全局 AI 服务商、密钥和模型设置。")
        ai_settings_btn.clicked.connect(lambda: AISettingsDialog(self).exec())
        al.addWidget(ai_settings_btn)

        checkpoint_btn = QToolButton()
        checkpoint_btn.setText("断点 ▾")
        checkpoint_btn.setPopupMode(QToolButton.InstantPopup)
        checkpoint_menu = QMenu(checkpoint_btn)
        checkpoint_menu.addAction("打开断点目录", self._open_formatter_checkpoint_directory)
        checkpoint_menu.addAction("选择已有断点目录", self._choose_formatter_checkpoint_directory)
        checkpoint_menu.addSeparator()
        checkpoint_menu.addAction("清除本书断点", self._clear_formatter_checkpoint_directory)
        checkpoint_btn.setMenu(checkpoint_menu)
        al.addWidget(checkpoint_btn)

        ai_layout_btn = accent_button("✨ AI排版", color="#7C3AED")
        ai_layout_btn.setToolTip("先补跑尚未执行的本地规则，再用 Formatter 专用规则处理长段、对白与叙述混排、跨块续接")
        ai_layout_btn.clicked.connect(lambda: self._run_formatter_ai("layout"))
        al.addWidget(ai_layout_btn)

        ai_correct_btn = accent_button("✨ AI纠错", color="#2F6BFF")
        ai_correct_btn.setToolTip("只纠正明确 OCR 错字、缺字、标点和语法，不改变块数量与排版")
        ai_correct_btn.clicked.connect(lambda: self._run_formatter_ai("correction"))
        al.addWidget(ai_correct_btn)

        ai_both_btn = accent_button("✨ AI纠错排版", color="#9D174D")
        ai_both_btn.setToolTip("使用宽松 Formatter 规则同时纠错和整理长文本结构。")
        ai_both_btn.clicked.connect(lambda: self._run_formatter_ai("correction_layout"))
        al.addWidget(ai_both_btn)
        create_workspace_clear_button(
            self, "Formatter", ClearManager.clear_formatter, target_layout=al,
        )
        self._formatter_ai_buttons = [ai_layout_btn, ai_correct_btn, ai_both_btn]

        result_layout.addWidget(result_actions, 0, Qt.AlignRight)
        rl.addWidget(result_bar)

        # 当前步骤的具体规则说明默认折叠；由左侧开关控制。
        self._rules_bar = QFrame()
        self._rules_bar.setStyleSheet(
            f"QFrame {{ background: #F7F8FA; border-top: 1px solid {BORDER}; }}"
        )
        self._rules_layout = QVBoxLayout(self._rules_bar)
        self._rules_layout.setContentsMargins(14, 9, 14, 10)
        self._rules_layout.setSpacing(4)
        self._rules_bar.setVisible(False)
        rl.addWidget(self._rules_bar)
        self._show_rule_details_cb.toggled.connect(self._set_rule_details_visible)

        root.addWidget(right, 1)
        self._advanced_left = left
        self._advanced_right = right

        # ── Phase 20 compact publication view ─────────────────────────────
        # Keep the complete historical editor alive and reachable through the
        # Advanced button, but make the supplied calm two-card layout the
        # default.  Compact controls delegate to the same checkboxes/callbacks,
        # so no formatter function or run path is forked.
        compact = QWidget()
        compact.setObjectName("formatterCompactView")
        compact_layout = QHBoxLayout(compact)
        compact_layout.setContentsMargins(12, 8, 12, 12)
        compact_layout.setSpacing(20)

        compact_left = QWidget()
        compact_left.setMinimumWidth(340)
        compact_left.setMaximumWidth(390)
        compact_left.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        compact_left_layout = QVBoxLayout(compact_left)
        compact_left_layout.setContentsMargins(0, 0, 0, 0)
        compact_left_layout.setSpacing(16)

        steps_card = QFrame()
        steps_card.setObjectName("formatterCompactCard")
        steps_card.setStyleSheet(
            f"QFrame#formatterCompactCard{{background:{CARD};border:1px solid {BORDER};border-radius:16px;}}"
        )
        steps_card.setFixedHeight(352)
        steps_box = QVBoxLayout(steps_card)
        steps_box.setContentsMargins(20, 18, 20, 16)
        steps_box.setSpacing(0)
        steps_title = QLabel("处理步骤")
        steps_title.setStyleSheet("font-size:13px;font-weight:700;margin-bottom:8px;")
        steps_box.addWidget(steps_title)
        # 14 条可见规则全部归入精简视图（原先只覆盖 11 条，短对白闭引号 / 重复删除 / 缩进分节没有入口）
        self._compact_step_groups = {
            "flow": ("cross_page_merge", "merge_overlaps", "merge_sentences"),
            "dialogue": ("repair_dialogue_quotes", "dialogue_restore"),
            "clean": ("remove_duplicates", "fix_dash_artifacts"),
            "indent": ("restore_indents",),
            "ruby": ("recover_ruby",),
            "punct": ("normalize_punct",),
            "chapter": ("split_embedded_titles", "detect_chapters"),
            "notes": ("strip_chapter_notes", "strip_boilerplate"),
        }
        compact_labels = [
            ("flow", "合并断行与整句重排"),
            ("dialogue", "对白与引号修复"),
            ("clean", "重复与破折号清理"),
            ("indent", "段首缩进与分节"),
            ("ruby", "注音（ルビ）保留"),
            ("punct", "标点与全半角统一"),
            ("chapter", "章节标题识别"),
            ("notes", "脚注与前后书处理"),
        ]
        self._compact_step_checks: dict[str, DesignSwitch] = {}
        for number, (group, label) in enumerate(compact_labels, start=1):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row.setFixedHeight(34)
            row_layout.setContentsMargins(0, 4, 0, 4)
            num = QLabel(str(number))
            num.setFixedWidth(20)
            num.setStyleSheet(f"color:{ACC};font-weight:800;")
            row_layout.addWidget(num)
            name = QLabel(label)
            name.setStyleSheet(f"color:{INK};font-size:12px;font-weight:600;")
            row_layout.addWidget(name, 1)
            switch = DesignSwitch()
            switch.setToolTip("切换该组规则；完整逐项控制请打开高级编辑")
            switch.toggled.connect(partial(self._set_compact_step_group, group))
            self._compact_step_checks[group] = switch
            row_layout.addWidget(switch)
            steps_box.addWidget(row)
            if number != len(compact_labels):
                sep = QFrame()
                sep.setFixedHeight(1)
                sep.setStyleSheet(f"background:{BORDER};border:none;")
                steps_box.addWidget(sep)
        compact_left_layout.addWidget(steps_card)

        output_card = QFrame()
        output_card.setObjectName("formatterCompactOutput")
        output_card.setStyleSheet(
            f"QFrame#formatterCompactOutput{{background:{CARD};border:1px solid {BORDER};border-radius:16px;}}"
        )
        output_card.setFixedHeight(150)
        output_box = QVBoxLayout(output_card)
        output_box.setContentsMargins(20, 16, 20, 16)
        output_box.setSpacing(8)
        output_title = QLabel("输出")
        output_title.setStyleSheet("font-size:13px;font-weight:700;")
        output_box.addWidget(output_title)
        # 输出只需要横排：不再提供方向选择。控件保留为隐藏的兼容引用，固定为“横排”。
        self._compact_direction = QComboBox(output_card)
        self._compact_direction.addItem("竖排（右→左）", True)
        self._compact_direction.addItem("横排（左→右）", False)
        self._compact_direction.setCurrentIndex(1)
        self._compact_direction.setVisible(False)
        direction_note = QLabel("文字方向：横排（左 → 右）")
        direction_note.setStyleSheet(f"color:{MUTED};font-size:12px;")
        output_box.addWidget(direction_note)
        compact_run = accent_button("开始处理")
        compact_run.setMinimumHeight(38)
        compact_run.clicked.connect(self._run_all)
        output_box.addWidget(compact_run)
        advanced = link_button("高级编辑 / 全部工具  ›")
        advanced.clicked.connect(lambda: self._set_formatter_compact_mode(False))
        advanced.setVisible(True)
        self._compact_advanced_btn = advanced
        output_box.addWidget(advanced)
        compact.setContextMenuPolicy(Qt.CustomContextMenu)
        def _open_formatter_compact_menu(pos):
            menu = QMenu(compact)
            menu.addAction("高级编辑 / 全部工具", lambda: self._set_formatter_compact_mode(False))
            menu.exec(compact.mapToGlobal(pos))
        compact.customContextMenuRequested.connect(_open_formatter_compact_menu)
        self._formatter_advanced_shortcut = QShortcut(QKeySequence("Ctrl+Shift+E"), compact)
        self._formatter_advanced_shortcut.activated.connect(lambda: self._set_formatter_compact_mode(False))
        compact_left_layout.addWidget(output_card)
        compact_left_layout.addStretch(1)
        compact_layout.addWidget(compact_left, 0)

        preview_card = QFrame()
        preview_card.setObjectName("formatterCompactPreviewCard")
        preview_card.setStyleSheet(
            f"QFrame#formatterCompactPreviewCard{{background:{CARD};border:1px solid {BORDER};border-radius:16px;}}"
        )
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        # The reference sheet is intentionally headerless: the page title
        # already identifies this as Format, so the preview can use the full
        # card height like a real book page.
        self._compact_preview = _FormatterBookPreview()
        self._compact_preview.set_vertical(False)   # 预览与输出一致：横排
        compact_preview_row = QHBoxLayout()
        compact_preview_row.setContentsMargins(0, 0, 0, 0)
        compact_preview_row.setSpacing(0)
        compact_preview_row.addStretch(1)
        compact_preview_row.addWidget(self._compact_preview, 8)
        compact_preview_row.addStretch(1)
        preview_layout.addLayout(compact_preview_row, 1)
        compact_layout.addWidget(preview_card, 1)

        root.addWidget(compact, 1)
        self._formatter_compact_view = compact

        # ── Dedicated selectable-PDF format view ─────────────────────────
        # PDF extraction and PDF publication formatting are deliberately two
        # different phases.  The adapter owns coordinate-dependent filtering
        # (Ruby/footer page numbers); this view owns logical reflow, notes,
        # generated-site matter, chapter reconstruction and the character guard.
        pdf_view = QWidget()
        pdf_view.setObjectName("pdfFormatView")
        pv = QHBoxLayout(pdf_view)
        pv.setContentsMargins(28, 8, 30, 24)
        pv.setSpacing(20)

        pdf_left = QWidget()
        # Preserve Phase35/37 responsive formatter width instead of the fixed 416px branch.
        pdf_left.setMinimumWidth(340)
        pdf_left.setMaximumWidth(390)
        pdf_left.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        pl = QVBoxLayout(pdf_left)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(16)

        info_card = QFrame()
        info_card.setObjectName("pdfFormatInfoCard")
        info_card.setStyleSheet(
            f"QFrame#pdfFormatInfoCard{{background:{CARD};border:1px solid {BORDER};border-radius:16px;}}"
        )
        il = QVBoxLayout(info_card)
        il.setContentsMargins(20, 18, 20, 18)
        il.setSpacing(8)
        title = QLabel("PDF 格式处理")
        title.setStyleSheet("font-size:15px;font-weight:800;")
        il.addWidget(title)
        note = QLabel("PDF文字层只负责完整提取字符与几何；这里统一处理分页/分列接续、前后书、站点页和正文格式。章节 / TOC 交给后续 AI 识别。")
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{MUTED};font-size:12px;")
        il.addWidget(note)
        self._pdf_format_extract_status = QLabel("等待 PDF 文字层结果")
        self._pdf_format_extract_status.setWordWrap(True)
        self._pdf_format_extract_status.setStyleSheet(f"color:{INK};font-size:12px;")
        il.addWidget(self._pdf_format_extract_status)
        pl.addWidget(info_card)

        options_card = QFrame()
        options_card.setObjectName("pdfFormatOptionsCard")
        options_card.setStyleSheet(
            f"QFrame#pdfFormatOptionsCard{{background:{CARD};border:1px solid {BORDER};border-radius:16px;}}"
        )
        ol = QVBoxLayout(options_card)
        ol.setContentsMargins(20, 16, 20, 16)
        ol.setSpacing(4)
        options_title = QLabel("PDF 专用规则")
        options_title.setStyleSheet("font-size:13px;font-weight:700;margin-bottom:4px;")
        ol.addWidget(options_title)

        def add_pdf_option(label_text, auto_text=None):
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 5, 0, 5)
            label = QLabel(label_text)
            label.setStyleSheet(f"color:{INK};font-size:12px;font-weight:600;")
            rl.addWidget(label, 1)
            if auto_text is not None:
                status = QLabel(auto_text)
                status.setStyleSheet(f"color:{SUCCESS};font-size:11px;font-weight:700;")
                rl.addWidget(status)
                switch = None
            else:
                switch = DesignSwitch()
                rl.addWidget(switch)
            ol.addWidget(row)
            return switch

        add_pdf_option("Ruby / 小字号注音", "提取层自动")
        add_pdf_option("页码过滤（仅几何确认 ASCII）", "提取层自动")
        add_pdf_option("物理列 + 跨页接续", "几何自动")
        add_pdf_option("章节 / TOC 识别", "目录待 AI 识别 · 交给 AI")
        self._pdf_remove_notes_switch = add_pdf_option("删除作者前书 / 后书")
        self._pdf_remove_generated_switch = add_pdf_option("清理 PDFNovels 站点前后置页")
        self._pdf_restore_indents_switch = add_pdf_option("恢复出版段首缩进")
        self._pdf_remove_notes_switch.setChecked(True)
        self._pdf_remove_generated_switch.setChecked(True)
        self._pdf_restore_indents_switch.setChecked(True)
        ol.addSpacing(4)
        guard = QLabel("字符守卫：强制执行，发现 missing / extra 会在报告中标红")
        guard.setWordWrap(True)
        guard.setStyleSheet(f"color:{MUTED};font-size:11px;")
        ol.addWidget(guard)
        pl.addWidget(options_card)

        run_card = QFrame()
        run_card.setObjectName("pdfFormatRunCard")
        run_card.setStyleSheet(
            f"QFrame#pdfFormatRunCard{{background:{CARD};border:1px solid {BORDER};border-radius:16px;}}"
        )
        rcl = QVBoxLayout(run_card)
        rcl.setContentsMargins(20, 16, 20, 16)
        self._pdf_format_run_btn = accent_button("开始 PDF 格式处理")
        self._pdf_format_run_btn.setMinimumHeight(38)
        self._pdf_format_run_btn.clicked.connect(self._run_pdf_format)
        rcl.addWidget(self._pdf_format_run_btn)
        pdf_advanced = link_button("通用高级编辑 / 全部工具  ›")
        pdf_advanced.clicked.connect(self._open_pdf_advanced_formatter)
        rcl.addWidget(pdf_advanced)
        pl.addWidget(run_card)
        pl.addStretch(1)
        pv.addWidget(pdf_left, 0)

        pdf_preview_card = QFrame()
        pdf_preview_card.setObjectName("pdfFormatPreviewCard")
        pdf_preview_card.setStyleSheet(
            f"QFrame#pdfFormatPreviewCard{{background:{CARD};border:1px solid {BORDER};border-radius:16px;}}"
        )
        pvl = QVBoxLayout(pdf_preview_card)
        pvl.setContentsMargins(20, 18, 20, 18)
        self._pdf_format_preview_title = QLabel("PDF 提取原文 · 尚未格式处理")
        self._pdf_format_preview_title.setStyleSheet(f"color:{MUTED};font-size:12px;font-weight:600;")
        pvl.addWidget(self._pdf_format_preview_title)
        self._pdf_format_preview = _FormatterBookPreview()
        self._pdf_format_preview.set_vertical(True)
        pdf_preview_row = QHBoxLayout()
        pdf_preview_row.setContentsMargins(0, 0, 0, 0)
        pdf_preview_row.setSpacing(0)
        pdf_preview_row.addStretch(1)
        pdf_preview_row.addWidget(self._pdf_format_preview, 8)
        pdf_preview_row.addStretch(1)
        pvl.addLayout(pdf_preview_row, 1)
        pv.addWidget(pdf_preview_card, 1)
        root.addWidget(pdf_view, 1)
        self._pdf_format_view = pdf_view
        self._pdf_format_view.setVisible(False)

        self._advanced_left.setVisible(False)
        self._advanced_right.setVisible(False)
        self._before.textChanged.connect(self._sync_compact_preview_text)
        self._after.textChanged.connect(self._sync_compact_preview_text)
        self._before.textChanged.connect(
            lambda: self._pdf_format_preview.set_text(
                self._after.toPlainText() if self._fmt_doc is not None else self._before.toPlainText()
            )
        )
        self._after.textChanged.connect(
            lambda: self._pdf_format_preview.set_text(
                self._after.toPlainText() if self._fmt_doc is not None else self._before.toPlainText()
            )
        )
        self._sync_compact_step_state()

        compact_return = QPushButton("简洁视图")
        compact_return.setProperty("flat", True)
        compact_return.clicked.connect(lambda: self._set_formatter_compact_mode(True))
        toolbar_actions_layout.insertWidget(0, compact_return)
        self._select_step("split_embedded_titles")


    def _set_compact_step_group(self, group: str, checked: bool) -> None:
        for sid in self._compact_step_groups.get(str(group), ()): 
            cb = self._step_checks.get(sid)
            if cb is not None and cb.isChecked() != bool(checked):
                cb.setChecked(bool(checked))

    def _sync_compact_step_state(self) -> None:
        for group, switch in getattr(self, "_compact_step_checks", {}).items():
            members = [self._step_checks.get(sid) for sid in self._compact_step_groups.get(group, ())]
            members = [cb for cb in members if cb is not None]
            desired = bool(members) and all(cb.isChecked() for cb in members)
            old = switch.blockSignals(True)
            switch.setChecked(desired)
            switch.blockSignals(old)

    def _advanced_editors_visible(self) -> bool:
        return bool(
            getattr(self, "_advanced_left", None) is not None
            and (self._advanced_left.isVisible() or self._advanced_right.isVisible())
        )

    def _materialize_advanced_editors(self, *, force: bool = False) -> None:
        """Populate the two full-book editors only when the advanced UI needs them."""
        if not force and not getattr(self, "_advanced_editors_dirty", True):
            return
        self._hydrating_advanced_editors = True
        before_blocked = self._before.blockSignals(True)
        after_blocked = self._after.blockSignals(True)
        try:
            if self._ocr_doc is not None:
                self._show_doc(self._ocr_doc, self._before)
            else:
                self._clear_editor(self._before)
            if self._fmt_doc is not None:
                self._show_doc(self._fmt_doc, self._after)
            else:
                self._clear_editor(self._after)
            self._advanced_editors_dirty = False
        finally:
            self._before.blockSignals(before_blocked)
            self._after.blockSignals(after_blocked)
            self._hydrating_advanced_editors = False
        self._sync_compact_preview_text()

    def _set_formatter_compact_mode(self, compact: bool) -> None:
        compact = bool(compact)
        self._formatter_compact_view.setVisible(compact)
        self._advanced_left.setVisible(not compact)
        self._advanced_right.setVisible(not compact)
        if compact:
            self._sync_compact_step_state()
            self._sync_compact_preview_text()
        else:
            self._materialize_advanced_editors()


    def _sync_compact_preview_text(self) -> None:
        preview = getattr(self, "_compact_preview", None)
        if preview is None or getattr(self, "_hydrating_advanced_editors", False):
            return
        compact_view = getattr(self, "_formatter_compact_view", None)
        if compact_view is not None and not compact_view.isVisible():
            self._compact_preview_dirty = True
            return
        # Re-showing the Formatter page must be O(1) when the document has not
        # changed. ``set_document`` still scans thousands of blocks to build its
        # content signature, so honor the view dirty bit before doing that work.
        # Advanced-editor text changes happen while the compact view is hidden
        # and mark this bit above, so unsaved edits remain visible on return.
        if not getattr(self, "_compact_preview_dirty", True):
            return
        # Unsaved edits only exist in the advanced editors.  While they are
        # visible, mirror those edits into the compact preview.  Otherwise use
        # the typed document directly and avoid converting the entire book to
        # plain text merely to feed a hidden editor.
        if self._advanced_editors_visible():
            if self._fmt_doc is not None and self._after.toPlainText().strip():
                preview.set_text(self._after.toPlainText())
                self._compact_preview_dirty = False
                return
            if self._ocr_doc is not None:
                preview.set_text(self._before.toPlainText())
                self._compact_preview_dirty = False
                return
        preview.set_document(self._fmt_doc or self._ocr_doc)
        self._compact_preview_dirty = False

    def _materialize_pdf_preview(self) -> None:
        if not getattr(self, "_pdf_preview_dirty", True):
            return
        preview = getattr(self, "_pdf_format_preview", None)
        view = getattr(self, "_pdf_format_view", None)
        if preview is None or view is None or not view.isVisible():
            return
        preview.set_document(self._fmt_doc or self._ocr_doc)
        self._pdf_preview_dirty = False

    def _set_pdf_format_view_active(self, active: bool) -> None:
        active = bool(active)
        if not hasattr(self, "_pdf_format_view"):
            return
        self._pdf_format_view.setVisible(active)
        if active:
            self._formatter_compact_view.setVisible(False)
            self._advanced_left.setVisible(False)
            self._advanced_right.setVisible(False)
            self._materialize_pdf_preview()
        elif not self._advanced_left.isVisible() and not self._advanced_right.isVisible():
            self._formatter_compact_view.setVisible(True)
            self._sync_compact_preview_text()

    def _open_pdf_advanced_formatter(self) -> None:
        self._pdf_format_view.setVisible(False)
        self._set_formatter_compact_mode(False)

    def _refresh_pdf_format_summary(self, doc: Optional[UnifiedDocument]) -> None:
        if not hasattr(self, "_pdf_format_extract_status"):
            return
        if doc is None:
            self._pdf_format_extract_status.setText("等待 PDF 文字层结果")
            self._pdf_format_preview_title.setText("等待 PDF 文字层结果")
            return
        meta = doc.metadata
        ruby = int(getattr(meta, "pdf_text_furigana_chars_skipped", 0) or 0)
        page_digits = int(getattr(meta, "pdf_text_page_number_chars_skipped", 0) or 0)
        physical = int(getattr(meta, "pdf_text_physical_page_count", 0) or 0)
        logical = int(getattr(meta, "pdf_text_logical_page_count", 0) or 0)
        processed = bool(getattr(meta, "pdf_format_processed", False))
        self._pdf_format_preview_title.setText(
            "PDF 格式处理结果" if processed else "PDF 提取原文 · 尚未格式处理"
        )
        guard = dict(getattr(meta, "pdf_text_guard_report", {}) or {})
        guard_state = "未运行"
        if guard:
            guard_state = "PASS" if guard.get("passed") else "FAIL"
        state = "已完成" if processed else "待处理"
        self._pdf_format_extract_status.setText(
            f"提取：{physical or '—'} 物理页 → {logical or '—'} 逻辑页 · "
            f"提取层几何过滤：Ruby {ruby} 字符 · 页脚页码 {page_digits} 字符\n"
            f"PDF格式：{state} · blocks {len(doc.blocks)} · 只做版面/前后书 · 目录交 AI · 字符守卫 {guard_state}"
        )

    def _run_pdf_format(self) -> None:
        if not self._ensure_editor_state_saved():
            return
        base = self._ocr_doc
        if base is None:
            QMessageBox.warning(self, "错误", "请先在“PDF文字层”完成提取")
            return
        source_engine = str(getattr(base.metadata, "source_engine", "") or "")
        if source_engine != "pdf_text_layer" and not bool(getattr(base.metadata, "pdf_text_layer_mode", False)):
            QMessageBox.warning(self, "错误", "当前文档不是 PDF 文字层结果")
            return
        token = self._begin_formatter_pipeline_operation()
        if token is None:
            return
        # ``run_pdf_format_pipeline`` owns the isolated working copy.  Do not
        # deepcopy the complete selectable-PDF document here as well: a 4k+
        # page book can exceed 100k raw blocks and the redundant GUI copy causes
        # a large memory/GC spike before formatting even starts.
        keep_notes = not self._pdf_remove_notes_switch.isChecked()
        remove_generated = self._pdf_remove_generated_switch.isChecked()
        restore_indents = self._pdf_restore_indents_switch.isChecked()
        self._progress.setVisible(True)
        self._progress.setRange(0, 20)

        def worker():
            try:
                from engine.pdf_format_pipeline import run_pdf_format_pipeline
                def on_progress(_step_name, current, total):
                    signals.progress.emit(current, total)
                result = run_pdf_format_pipeline(
                    base,
                    keep_author_notes=keep_notes,
                    remove_generated_matter=remove_generated,
                    restore_indents=restore_indents,
                    verbose=False,
                    progress_callback=on_progress,
                )
                signals.finished.emit(result)
            except Exception:
                import traceback
                signals.error.emit(traceback.format_exc())

        signals = self._pipeline_signals = WorkerSignals()
        signals.finished.connect(
            lambda result, t=token: self._on_run_done(result)
            if self._finish_formatter_pipeline_operation(t) else None
        )
        signals.error.connect(
            lambda message, t=token: self._on_error(message)
            if self._finish_formatter_pipeline_operation(t) else None
        )
        signals.progress.connect(
            lambda cur, total, t=token: (self._progress.setRange(0, max(1, total)), self._progress.setValue(cur))
            if self._formatter_operation_current(t) else None
        )
        threading.Thread(target=worker, daemon=True).start()

    def reset_for_new_book(self):
        self._content_generation.invalidate()
        self._formatter_ai_cancel_requested = True
        self._formatter_ai_cancel_event.set()
        self._original_doc = None
        self._ocr_doc = None
        self._fmt_doc = None
        self._set_pdf_text_layer_mode_checked(False)
        self._set_pdf_format_view_active(False)
        self._refresh_editor_views()
        self._update_version(None)
        self._progress.setVisible(False)
        self._formatter_ai_running = False
        self._formatter_pipeline_running = False
        self._last_manual_batch_restore = None
        if hasattr(self, "_restore_batch_btn"):
            self._restore_batch_btn.setEnabled(False)
        self._set_formatter_ai_buttons_enabled(True)
        self._set_formatter_action_buttons_enabled(True)
        if hasattr(self, "_formatter_ai_status"):
            self._formatter_ai_status.setText("本地规则处理短文本；Formatter AI 负责长文本和复杂结构")

    def shutdown_cleanup(self) -> None:
        self._content_generation.invalidate()
        self._formatter_ai_cancel_requested = True
        self._formatter_ai_cancel_event.set()

    def set_doc(self, doc: UnifiedDocument):
        # A new upstream snapshot invalidates any formatter/import result still
        # returning from the previous book or previous base document.
        self._content_generation.invalidate()
        self._formatter_ai_cancel_requested = True
        self._formatter_ai_cancel_event.set()
        self._formatter_ai_running = False
        self._formatter_pipeline_running = False
        self._set_formatter_ai_buttons_enabled(True)
        self._set_formatter_action_buttons_enabled(True)
        # MainWindow synchronization must not emit doc_formatted again; doing so
        # created feedback loops between Formatter, Compare and Replacement.
        self._set_base_doc(doc, notify=False)

    def _set_rule_details_visible(self, visible: bool):
        self._rules_bar.setVisible(bool(visible))
        self._refresh_rules(self._active_step)

    def _select_step(self, sid, event=None):
        self._active_step = sid
        for s, card in self._step_cards.items():
            if s == sid:
                card.setStyleSheet(
                    f"QWidget#stepCard {{ background: {ACC_BG}; border: 1.5px solid {ACC}; border-radius: 8px; }}")
            else:
                card.setStyleSheet(
                    f"QWidget#stepCard {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px; }}"
                    f"QWidget#stepCard:hover {{ border-color: #CDD3DA; background: #F7F8FA; }}")
        info = {s: (l, d) for s, l, _, d in FORMATTER_STEPS}
        if sid in info:
            label, _desc = info[sid]
            self._step_title.setText(label)
        self._refresh_rules(sid)

    def _refresh_rules(self, sid):
        while self._rules_layout.count():
            item = self._rules_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        if not self._show_rule_details_cb.isChecked():
            return

        info = {s: (label, desc) for s, label, _badge, desc in FORMATTER_STEPS}
        label, desc = info.get(sid, (sid, ""))
        title = QLabel(f"{label}：具体规则说明")
        title.setStyleSheet(f"color: {INK}; font-weight: 600; font-size: 12px;")
        self._rules_layout.addWidget(title)

        rules = FORMATTER_RULES.get(sid) or ([desc] if desc else [])
        for rule in rules:
            line = QLabel(f"• {rule}")
            line.setWordWrap(True)
            line.setTextInteractionFlags(Qt.TextSelectableByMouse)
            line.setStyleSheet(f"color: {MUTED}; font-size: 11px; padding-left: 4px;")
            self._rules_layout.addWidget(line)

    def _replacement_mode_changed(self, *_args):
        mode=self._replacement_mode.currentData()
        labels={
            "strict_full":"🔁 覆盖并重建排版",
            "strict_literal":"🔁 原样严格覆盖",
            "smart_patch":"🔀 智能替换",
            "compare_only":"🔎 仅比较",
        }
        self._run_btn.setText(labels.get(mode,"开始处理"))

    def _make_editor_pane(self, editor: QPlainTextEdit) -> QWidget:
        configure_drag_scrollbar(editor)
        pane = QWidget()
        layout = QHBoxLayout(pane)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        layout.addWidget(editor, 1)

        rail = QWidget()
        rail.setFixedWidth(26)
        rail_layout = QVBoxLayout(rail)
        rail_layout.setContentsMargins(1, 1, 1, 1)
        rail_layout.setSpacing(2)

        top_btn = QToolButton()
        top_btn.setObjectName("railNavBtn")
        top_btn.setText("▲")
        top_btn.setToolTip("滚动到顶部（⌘↑ / Ctrl+Home）")
        top_btn.setFixedSize(22, 22)
        top_btn.setAutoRaise(True)
        top_btn.clicked.connect(partial(self._scroll_editor, editor, "top"))
        rail_layout.addWidget(top_btn, 0, Qt.AlignHCenter)

        position_slider = EditorPositionSlider()
        attach_editor_position_slider(editor, position_slider)
        rail_layout.addWidget(position_slider, 1, Qt.AlignHCenter)

        bottom_btn = QToolButton()
        bottom_btn.setObjectName("railNavBtn")
        bottom_btn.setText("▼")
        bottom_btn.setToolTip("滚动到底部（⌘↓ / Ctrl+End）")
        bottom_btn.setFixedSize(22, 22)
        bottom_btn.setAutoRaise(True)
        bottom_btn.clicked.connect(partial(self._scroll_editor, editor, "bottom"))
        rail_layout.addWidget(bottom_btn, 0, Qt.AlignHCenter)
        layout.addWidget(rail, 0)
        return pane

    def _install_scroll_shortcuts(self, editor: QPlainTextEdit):
        shortcuts = [
            ("Meta+Up", "top"),
            ("Meta+Down", "bottom"),
            ("Ctrl+Home", "top"),
            ("Ctrl+End", "bottom"),
        ]
        for sequence, action in shortcuts:
            shortcut = QShortcut(QKeySequence(sequence), editor)
            shortcut.setContext(Qt.WidgetShortcut)
            shortcut.activated.connect(partial(self._scroll_editor, editor, action))
            self._scroll_shortcuts.append(shortcut)

    def _scroll_editor(self, editor: QPlainTextEdit, action: str):
        bar = editor.verticalScrollBar()
        if action == "top":
            bar.setValue(bar.minimum())
        elif action == "bottom":
            bar.setValue(bar.maximum())
        elif action == "page_up":
            bar.setValue(max(bar.minimum(), bar.value() - bar.pageStep()))
        elif action == "page_down":
            bar.setValue(min(bar.maximum(), bar.value() + bar.pageStep()))
        editor.setFocus()

    def _clear_editor(self, widget: QPlainTextEdit):
        widget.clear()
        widget.document().setModified(False)

    def _set_pdf_text_layer_mode_checked(self, enabled: bool):
        if not hasattr(self, "_pdf_text_layer_mode_cb"):
            return
        previous = self._pdf_text_layer_mode_cb.blockSignals(True)
        self._pdf_text_layer_mode_cb.setChecked(bool(enabled))
        self._pdf_text_layer_mode_cb.blockSignals(previous)
        self._preserve_ocr_layout_cb.setEnabled(not bool(enabled))
        if hasattr(self, "_pdf_keep_afterwords_cb"):
            self._pdf_keep_afterwords_cb.setEnabled(bool(enabled))
        if enabled:
            self._preserve_ocr_layout_cb.setChecked(False)
            self._preserve_ocr_layout_cb.setToolTip(
                "PDF文字层模式需要重新接续物理列，因此不能同时固定原OCR排版"
            )
        else:
            self._preserve_ocr_layout_cb.setToolTip(
                "保留导入或 OCR 产生的原始块、段落及分页结构；适合希望手动处理版式时使用"
            )
        self._update_formatter_mode_status()

    def _update_formatter_mode_status(self):
        if not hasattr(self, "_formatter_ai_status"):
            return
        enabled = bool(
            hasattr(self, "_pdf_text_layer_mode_cb")
            and self._pdf_text_layer_mode_cb.isChecked()
        )
        self._formatter_ai_status.setText(
            "PDF文字层专用规则已开启；普通图片OCR规则保持独立"
            if enabled
            else "本地规则处理短文本；Formatter AI 负责长文本和复杂结构"
        )

    def _apply_mode_to_documents(self):
        enabled = bool(
            hasattr(self, "_pdf_text_layer_mode_cb")
            and self._pdf_text_layer_mode_cb.isChecked()
        )
        keep_afterwords = bool(
            hasattr(self, "_pdf_keep_afterwords_cb")
            and self._pdf_keep_afterwords_cb.isChecked()
        )
        for doc in (self._original_doc, self._ocr_doc, self._fmt_doc):
            if doc is None:
                continue
            doc.metadata.pdf_text_layer_mode = enabled
            doc.metadata.pdf_keep_afterwords = keep_afterwords
            if hasattr(self, "_pdf_remove_generated_switch"):
                doc.metadata.pdf_remove_generated_matter = self._pdf_remove_generated_switch.isChecked()
                doc.metadata.pdf_restore_indents = self._pdf_restore_indents_switch.isChecked()
            if enabled:
                doc.metadata.preserve_ocr_layout = False

    def _on_pdf_text_layer_mode_toggled(self, enabled: bool):
        self._set_pdf_text_layer_mode_checked(bool(enabled))
        self._apply_mode_to_documents()

    def _current_doc(self) -> Optional[UnifiedDocument]:
        return self._fmt_doc or self._ocr_doc

    @staticmethod
    def _snapshot_clone_document(doc: UnifiedDocument) -> UnifiedDocument:
        """Create an exact detached document snapshot without generic deepcopy."""
        return doc.snapshot_clone()

    def _set_base_doc(self, doc: UnifiedDocument, notify: bool = True):
        self._last_manual_batch_restore = None
        if hasattr(self, "_restore_batch_btn"):
            self._restore_batch_btn.setEnabled(False)
        self._original_doc = self._snapshot_clone_document(doc)
        self._ocr_doc = self._snapshot_clone_document(doc)
        self._fmt_doc = None
        source_engine = str(getattr(doc.metadata, "source_engine", "") or "")
        pdf_mode = bool(getattr(doc.metadata, "pdf_text_layer_mode", False))
        # Backward compatibility for JSON produced before v1.3.7: selectable
        # PDF documents were already tagged by source_engine but had no mode bit.
        if source_engine == "pdf_text_layer":
            pdf_mode = True
        self._set_pdf_text_layer_mode_checked(pdf_mode)
        if hasattr(self, "_preserve_ocr_layout_cb"):
            previous = self._preserve_ocr_layout_cb.blockSignals(True)
            self._preserve_ocr_layout_cb.setChecked(
                bool(getattr(doc.metadata, "preserve_ocr_layout", False)) and not pdf_mode
            )
            self._preserve_ocr_layout_cb.blockSignals(previous)
        if hasattr(self, "_pdf_keep_afterwords_cb"):
            previous = self._pdf_keep_afterwords_cb.blockSignals(True)
            self._pdf_keep_afterwords_cb.setChecked(
                bool(getattr(doc.metadata, "pdf_keep_afterwords", False))
            )
            self._pdf_keep_afterwords_cb.blockSignals(previous)
        if hasattr(self, "_pdf_remove_notes_switch"):
            previous = self._pdf_remove_notes_switch.blockSignals(True)
            self._pdf_remove_notes_switch.setChecked(
                not bool(getattr(doc.metadata, "pdf_keep_afterwords", False))
            )
            self._pdf_remove_notes_switch.blockSignals(previous)
            previous = self._pdf_remove_generated_switch.blockSignals(True)
            self._pdf_remove_generated_switch.setChecked(
                bool(getattr(doc.metadata, "pdf_remove_generated_matter", True))
            )
            self._pdf_remove_generated_switch.blockSignals(previous)
            previous = self._pdf_restore_indents_switch.blockSignals(True)
            self._pdf_restore_indents_switch.setChecked(
                bool(getattr(doc.metadata, "pdf_restore_indents", True))
            )
            self._pdf_restore_indents_switch.blockSignals(previous)
        self._apply_mode_to_documents()
        self._set_pdf_format_view_active(pdf_mode)
        self._refresh_editor_views()
        self._update_version(None)
        if notify:
            self.doc_formatted.emit(self._ocr_doc)

    def _refresh_editor_views(self):
        # Only the currently visible heavy surface is materialized.  Compact,
        # advanced and PDF views each keep a dirty bit and hydrate on demand.
        self._advanced_editors_dirty = True
        self._compact_preview_dirty = True
        self._pdf_preview_dirty = True
        if self._advanced_editors_visible():
            self._materialize_advanced_editors()
        elif getattr(self, "_pdf_format_view", None) is not None and self._pdf_format_view.isVisible():
            self._materialize_pdf_preview()
        else:
            self._sync_compact_preview_text()
        if hasattr(self, "_pdf_format_preview"):
            self._refresh_pdf_format_summary(self._fmt_doc or self._ocr_doc)

    def _operation_steps(self, doc: Optional[UnifiedDocument]) -> set[str]:
        """返回当前文本修订周期内已经执行的步骤。

        正文手动编辑会生成新的版本；更早的 Formatter 日志不能继续
        用来提示“已经执行过”，否则用户会误以为替换后的正文已经完成格式化，
        并被诱导选择危险的强制重复执行。
        """
        if doc is None:
            return set()
        logs = list(getattr(doc, "processing_log", []))
        last_text_mutation = -1
        for i, entry in enumerate(logs):
            if entry.get("step") == "manual_edit":
                last_text_mutation = i
        return {
            entry.get("step", "")
            for entry in logs[last_text_mutation + 1:]
        }

    def _confirm_reapply(self, doc: Optional[UnifiedDocument], steps: list[str], label: str) -> bool:
        repeated = [s for s in steps if s in self._operation_steps(doc)]
        if not repeated:
            return True
        names = "、".join(repeated)
        answer = QMessageBox.question(
            self,
            "重复执行确认",
            f"{label} 已经执行过：{names}\n\n"
            "继续会基于当前文档再次执行，可能改变已有结果。\n"
            "如果想从处理前重新开始，请先使用「清空处理后」或「重新载入处理前」。\n\n"
            "是否继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    def _ensure_editor_state_saved(self) -> bool:
        dirty_before = self._before.document().isModified()
        dirty_after = self._after.document().isModified()
        if not dirty_before and not dirty_after:
            return True

        answer = QMessageBox.question(
            self,
            "文本框有未应用修改",
            "处理前/处理后文本框里有尚未写回文档模型的修改。\n\n"
            "选择「Yes」会先应用编辑再继续；选择「No」会丢弃文本框临时修改并重新按文档模型刷新；选择「Cancel」取消当前操作。",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Yes,
        )
        if answer == QMessageBox.Cancel:
            return False
        if answer == QMessageBox.Yes:
            return self._apply_manual_edit(show_message=False)

        self._refresh_editor_views()
        return True

    def _show_doc(self, doc, widget):
        if doc is None:
            return
        lines = []
        for b in doc.blocks:
            if (b.metadata or {}).get("consumed"):
                continue
            if b.type == BlockType.IMAGE_REF:
                lines.append(f"[图片: {Path(b.image_path).name}]")
            elif b.type == BlockType.CHAPTER:
                lines.append(f"\n▌ {b.text}\n")
            elif b.type == BlockType.DIALOGUE:
                lines.append(f"  {b.text}")
            elif b.type == BlockType.SECTION:
                lines.append(f"\n── {b.text} ──\n")
            elif b.type == BlockType.RUBY:
                lines.append(f"  [ruby] {b.text}")
            else:
                # 可编辑预览只显示正文。旧版用“◼”标记修改块，复制或再次
                # 应用编辑时容易被误认为 OCR 字符，并把连续省略号拆成独立行。
                lines.append(f"  {b.text}")
        widget.setPlainText("\n".join(lines))
        widget.document().setModified(False)

    def _editor_lines(self, widget):
        lines = []
        for raw in widget.toPlainText().splitlines():
            text = raw.strip()
            if not text or text.startswith("[图片:"):
                continue
            for prefix in ("▌", "──", "[ruby]", "◼"):
                if text.startswith(prefix):
                    text = text[len(prefix):].strip()
            if text.endswith("──"):
                text = text[:-2].strip()
            lines.append(text)
        return lines

    @staticmethod
    def _doc_state_token(doc) -> str:
        """Fingerprint complete content so restore cannot overwrite later work."""
        if doc is None:
            return ""
        payload = doc.to_dict() if hasattr(doc, "to_dict") else doc
        return content_state_token(
            payload, lineage=str(getattr(doc, "commit_id", "") or "")
        )

    def _manual_edit_change_set(self) -> BatchChangeSet | None:
        changes: list[BatchChange] = []
        sources = [
            ("before", self._ocr_doc, self._before, "处理前"),
        ]
        if self._fmt_doc is not None and self._after.toPlainText().strip():
            sources.append(("after", self._fmt_doc, self._after, "处理后"))

        source_tokens: list[str] = []
        for prefix, doc, widget, source_name in sources:
            if doc is None:
                continue
            edited = self._editor_lines(widget)
            text_blocks = [
                b for b in doc.blocks
                if b.type in {BlockType.PARAGRAPH, BlockType.DIALOGUE, BlockType.CHAPTER,
                              BlockType.SECTION, BlockType.RUBY}
            ]
            if not edited:
                continue
            if len(edited) != len(text_blocks):
                QMessageBox.warning(
                    self,
                    "无法预览编辑",
                    f"{source_name}文本行数和文档文本块数不一致：{len(edited)} 行 / {len(text_blocks)} 块。\n"
                    "请保持一行对应一个文本块；删除空行可以，但不要合并或拆分块。"
                )
                return None
            source_tokens.append(f"{prefix}:{self._doc_state_token(doc)}")
            for index, (block, new_text) in enumerate(zip(text_blocks, edited)):
                if str(block.text or "") == str(new_text or ""):
                    continue
                block_id = str(getattr(block, "id", "") or f"{index:06d}")
                changes.append(BatchChange.create(
                    f"{prefix}:{block_id}",
                    str(block.text or ""),
                    str(new_text or ""),
                    category=source_name,
                    detail=f"第 {index + 1} 个文本块",
                    metadata={"block_id": block_id, "source": prefix},
                ))
        return BatchChangeSet.from_changes(
            "Formatter 批量编辑预览",
            changes,
            source_token="|".join(source_tokens),
            operation="formatter_manual_edit",
        )

    def _restore_last_manual_batch(self) -> None:
        snapshot = self._last_manual_batch_restore
        if not isinstance(snapshot, dict):
            notify(self, "没有可恢复的批量编辑", "info")
            return
        current = (self._doc_state_token(self._ocr_doc), self._doc_state_token(self._fmt_doc))
        guard_ocr = snapshot.get("guard_ocr")
        guard_fmt = snapshot.get("guard_fmt")
        valid = (
            isinstance(guard_ocr, BatchRestoreGuard)
            and isinstance(guard_fmt, BatchRestoreGuard)
            and guard_ocr.can_restore(current[0])
            and guard_fmt.can_restore(current[1])
        )
        if not valid:
            QMessageBox.warning(
                self,
                "无法恢复",
                "批量编辑之后正文又发生了变化。为避免覆盖后续工作，本次批量恢复已失效。"
            )
            self._last_manual_batch_restore = None
            self._restore_batch_btn.setEnabled(False)
            return
        self._ocr_doc = snapshot.get("before_ocr")
        self._fmt_doc = snapshot.get("before_fmt")
        self._last_manual_batch_restore = None
        self._restore_batch_btn.setEnabled(False)
        self._refresh_editor_views()
        self._update_version(self._fmt_doc)
        self.doc_formatted.emit(self._fmt_doc or self._ocr_doc)
        notify(self, "已恢复最近一次批量编辑；更早历史未受影响。", "success")

    def _apply_editor_to_doc(self, doc, widget, source_name):
        if doc is None:
            return None, 0
        edited = self._editor_lines(widget)
        text_blocks = [
            b for b in doc.blocks
            if b.type in {BlockType.PARAGRAPH, BlockType.DIALOGUE, BlockType.CHAPTER,
                          BlockType.SECTION, BlockType.RUBY}
        ]
        if not edited:
            return doc, 0
        if len(edited) != len(text_blocks):
            QMessageBox.warning(
                self,
                "无法应用编辑",
                f"{source_name}文本行数和文档文本块数不一致：{len(edited)} 行 / {len(text_blocks)} 块。\n"
                "请保持一行对应一个文本块；删除空行可以，但不要合并或拆分块。"
            )
            return None, 0

        changed = 0
        new_doc = self._snapshot_clone_document(doc)
        editable_idx = 0
        for block in new_doc.blocks:
            if block.type not in {BlockType.PARAGRAPH, BlockType.DIALOGUE, BlockType.CHAPTER,
                                  BlockType.SECTION, BlockType.RUBY}:
                continue
            new_text = edited[editable_idx]
            editable_idx += 1
            if block.text != new_text:
                block.ocr_raw = block.ocr_raw or block.text
                block.text = new_text
                block.modified_by = "manual_edit"
                changed += 1
        if changed:
            new_doc.add_log("manual_edit", f"{source_name}手动编辑 {changed} 个文本块", changed)
        return new_doc, changed

    def _apply_manual_edit(self, show_message=True) -> bool:
        change_set = self._manual_edit_change_set()
        if change_set is None:
            return False
        if change_set.count and not BatchChangePreviewDialog.confirm(
            self, change_set, apply_text="应用批量编辑"
        ):
            return False

        before_ocr = self._ocr_doc
        before_fmt = self._fmt_doc
        total_changed = 0

        if self._ocr_doc is not None:
            updated, changed = self._apply_editor_to_doc(self._ocr_doc, self._before, "处理前")
            if updated is None:
                return False
            self._ocr_doc = updated
            total_changed += changed

        if self._fmt_doc is not None and self._after.toPlainText().strip():
            updated, changed = self._apply_editor_to_doc(self._fmt_doc, self._after, "处理后")
            if updated is None:
                return False
            self._fmt_doc = updated
            total_changed += changed
            self._update_version(self._fmt_doc)
            self.doc_formatted.emit(self._fmt_doc)
        elif self._ocr_doc is not None:
            self.doc_formatted.emit(self._ocr_doc)

        self._show_doc(self._ocr_doc, self._before)
        if self._fmt_doc is not None:
            self._show_doc(self._fmt_doc, self._after)

        if total_changed:
            self._last_manual_batch_restore = {
                "before_ocr": before_ocr,
                "before_fmt": before_fmt,
                "guard_ocr": BatchRestoreGuard(
                    operation="formatter_manual_edit:before",
                    after_token=self._doc_state_token(self._ocr_doc),
                    change_set_fingerprint=change_set.fingerprint,
                ),
                "guard_fmt": BatchRestoreGuard(
                    operation="formatter_manual_edit:after",
                    after_token=self._doc_state_token(self._fmt_doc),
                    change_set_fingerprint=change_set.fingerprint,
                ),
            }
            self._restore_batch_btn.setEnabled(True)
        if show_message:
            notify(self, f"已应用手动编辑：{total_changed} 个文本块", "success")
        return True

    def _clear_after_result(self):
        self._fmt_doc = None
        self._clear_editor(self._after)
        self._sync_compact_preview_text()
        if hasattr(self, "_pdf_format_preview"):
            self._pdf_format_preview.set_document(self._ocr_doc)
            self._refresh_pdf_format_summary(self._ocr_doc)
        self._update_version(None)
        if self._ocr_doc is not None:
            self.doc_formatted.emit(self._ocr_doc)

    def _reload_before_into_after(self):
        if self._ocr_doc is None:
            notify(self, "还没有处理前内容可载入", "info")
            return
        self._fmt_doc = self._snapshot_clone_document(self._ocr_doc)
        self._refresh_editor_views()
        self._update_version(self._fmt_doc)
        self.doc_formatted.emit(self._fmt_doc)

    def _load_json(self):
        """Formatter 外部导入入口。

        本项目保存的 UnifiedDocument JSON 与 Novel Formatter TEI P5 XML 可恢复
        完整页面/证据结构；外部 TEI 则通过 facsimile/standOff 尽可能映射。
        PaddleOCR-VL 原始 JSON、Markdown 与 TXT 按段落构造可格式化文档。
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "导入待格式处理文本", "",
            "支持的格式 (*.json *.xml *.tei *.md *.markdown *.txt);;TEI P5 XML (*.xml *.tei);;JSON (*.json);;Markdown (*.md *.markdown);;文本 (*.txt)")
        if not path:
            return
        try:
            suffix = Path(path).suffix.lower()
            doc = None
            if suffix in {".xml", ".tei"}:
                from adapters.tei_adapter import load_tei
                doc = load_tei(path)
            elif suffix == ".json":
                import json
                raw = Path(path).read_text(encoding="utf-8")
                data = json.loads(raw)
                if isinstance(data, dict) and "blocks" in data:
                    doc = UnifiedDocument.from_dict(data)
            if doc is None:
                from adapters.text_extractors import extract_paragraphs
                from models.document import Block
                paragraphs = extract_paragraphs(path)
                doc = UnifiedDocument()
                doc.metadata.source_engine = f"external_{suffix.lstrip('.') or 'text'}"
                for idx, para in enumerate(paragraphs):
                    btype = BlockType.CHAPTER if para.is_title else BlockType.PARAGRAPH
                    doc.blocks.append(Block(
                        type=btype, text=para.text, page=0,
                        reading_order=idx, ocr_raw=para.text,
                        source_format=suffix.lstrip('.') or "text"))
                doc.add_log("external_import", f"从 {Path(path).name} 导入 {len(doc.blocks)} 段", len(doc.blocks))
            self._formatter_source_path = path
            self._formatter_checkpoint_override = ""
            self._set_base_doc(doc)
            notify(self, f"已导入 {len(doc.blocks)} 个块", "success")
        except Exception as e:
            QMessageBox.critical(self, "载入失败", str(e))

    def _import_docx(self):
        """
        导入已经 OCR 识别过的 DOCX（Abbyy/Adobe 等输出），
        直接作为 Formatter 的输入，只做后处理，不进入 OCR 流程。
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "导入 Word 文档（已识别）", "", "Word 文档 (*.docx *.doc)")
        if not path:
            return
        token = self._begin_formatter_pipeline_operation()
        if token is None:
            return
        self._formatter_source_path = path
        self._formatter_checkpoint_override = ""

        def worker():
            try:
                from adapters.docx_adapter import import_docx
                doc = import_docx(path, verbose=True)
                signals.finished.emit(doc)
            except Exception:
                import traceback
                signals.error.emit(traceback.format_exc())

        signals = self._docx_signals = WorkerSignals()
        signals.finished.connect(
            lambda doc, t=token: self._on_docx_imported(doc)
            if self._finish_formatter_pipeline_operation(t) else None
        )
        signals.error.connect(
            lambda message, t=token: self._on_error(message)
            if self._finish_formatter_pipeline_operation(t) else None
        )
        threading.Thread(target=worker, daemon=True).start()

    def _on_docx_imported(self, doc):
        self._set_base_doc(doc)
        notify(
            self,
            ui_message("formatter.import_docx.summary", blocks=len(doc.blocks), chapters=len(doc.toc)),
            "success",
        )

    def _import_epub(self):
        """
        逆向导入已有 EPUB（比如本工具之前生成、又手动编辑过的文件），
        重新回炉跑 Formatter Pipeline——常见场景是用"前后书剥离"步骤清理
        导入时才发现混进正文里的网站样板文字/版权声明。
        """
        path, _ = QFileDialog.getOpenFileName(self, "导入 EPUB", "", "EPUB (*.epub)")
        if not path:
            return
        token = self._begin_formatter_pipeline_operation()
        if token is None:
            return
        self._formatter_source_path = path
        self._formatter_checkpoint_override = ""

        def worker():
            try:
                from adapters.epub_adapter import import_epub
                doc = import_epub(path, verbose=True)
                signals.finished.emit(doc)
            except Exception:
                import traceback
                signals.error.emit(traceback.format_exc())

        signals = self._epub_import_signals = WorkerSignals()
        signals.finished.connect(
            lambda doc, t=token: self._on_epub_imported(doc)
            if self._finish_formatter_pipeline_operation(t) else None
        )
        signals.error.connect(
            lambda message, t=token: self._on_error(message)
            if self._finish_formatter_pipeline_operation(t) else None
        )
        threading.Thread(target=worker, daemon=True).start()

    def _on_epub_imported(self, doc):
        self._set_base_doc(doc)
        notify(
            self,
            ui_message("formatter.import_epub.summary", blocks=len(doc.blocks), toc=len(doc.toc)),
            "success",
        )


    def _current_result_document(self):
        return self._fmt_doc or self._ocr_doc

    def _save_result(self):
        """Save the current Formatter result using one unified format picker."""
        doc = self._current_result_document()
        if not doc:
            notify(self, "还没有处理结果", "info")
            return

        suggested_stem = safe_result_filename(getattr(doc.metadata, "title", ""))
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "保存 Formatter 结果",
            f"{suggested_stem}.docx",
            save_filter_string(),
        )
        if not path:
            return

        fmt = format_from_filter(selected_filter, path)
        path = ensure_export_extension(path, fmt)
        if fmt == "docx":
            self._export_docx_to_path(doc, path)
            return

        try:
            saved_path = export_text_result(doc, path, fmt)
        except Exception:
            import traceback
            show_error_dialog(self, "保存失败", traceback.format_exc())
            return
        notify(self, ui_message("common.saved_path", path=saved_path), "success")

    def _save_json(self):
        """Backward-compatible JSON-only entry point used by older plugins."""
        doc = self._fmt_doc or self._ocr_doc
        if not doc:
            notify(self, "还没有处理结果", "info")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存 JSON", "", "JSON (*.json)")
        if path:
            try:
                saved_path = export_text_result(doc, path, "json")
            except Exception:
                import traceback
                show_error_dialog(self, "保存失败", traceback.format_exc())
                return
            notify(self, f"已保存: {saved_path}", "success")

    def _export_docx(self):
        """Backward-compatible DOCX-only entry point used by older plugins."""
        doc = self._fmt_doc or self._ocr_doc
        if not doc:
            notify(self, "还没有处理结果", "info")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出 DOCX", "", "Word (*.docx)")
        if not path:
            return

        self._export_docx_to_path(doc, ensure_export_extension(path, "docx"))

    def _export_docx_to_path(self, doc, path):
        """Run the potentially slow Word build outside the GUI thread."""

        def worker():
            try:
                from builder.word_builder import build_word
                report = build_word(doc, output_path=path, verbose=False)
                signals.finished.emit(report)
            except Exception:
                import traceback
                signals.error.emit(traceback.format_exc())

        def show_report(report):
            if isinstance(report, dict):
                saved_path = report.get("path", path)
                expected = int(report.get("expected_columns", 0) or 0)
                written = int(report.get("docx_written_columns", 0) or 0)
                if expected:
                    last_ok = bool(report.get("last_page_all_columns_exported"))
                    text = (
                        f"DOCX 已生成:\n{saved_path}\n\n"
                        f"列对账：预计 {expected} · 写入 DOCX {written}\n"
                        f"最后一页：{'全部列已导出' if last_ok else '存在未导出列'}"
                    )
                else:
                    text = f"DOCX 已生成:\n{saved_path}"
            else:
                text = f"DOCX 已生成:\n{report}"
            notify(self, text, "success")

        signals = self._docx_export_signals = WorkerSignals()
        signals.finished.connect(show_report)
        signals.error.connect(lambda msg: show_error_dialog(self, "导出失败", msg))
        threading.Thread(target=worker, daemon=True).start()

    def _undo(self):
        doc = self._fmt_doc
        log = doc.commit_log() if doc else []
        if not doc or len(log) < 2:
            notify(self, "没有可撤销的步骤", "info")
            return
        # log[0] 是当前 commit，log[1] 是上一步
        prev = doc.rollback_to_commit(log[1]["id"])
        self._fmt_doc = prev
        self._show_doc(prev, self._after)
        self._update_version(prev)
        self.doc_formatted.emit(prev)

    def _update_version(self, doc):
        if doc:
            log = doc.commit_log()
            short = doc.commit_id[:8] if doc.commit_id else "—"
            self._version_lbl.setText(f"commit {short} · {len(log)} 步历史")
        else:
            self._version_lbl.setText("")

    def _run_all(self):
        steps = [sid for sid, _, _, _ in FORMATTER_STEPS if self._step_checks[sid].isChecked()]
        if not steps:
            notify(self, "格式规则默认全部关闭，请先勾选需要运行的规则。", "warning")
            return
        self._run_steps(steps, base_on_current=True)

    def _apply_step(self):
        self._run_steps([self._active_step], base_on_current=True)

    def _run_steps(self, steps, base_on_current=True):
        if not self._ensure_editor_state_saved():
            return
        base = (self._fmt_doc if base_on_current and self._fmt_doc else self._ocr_doc)
        if base is None:
            QMessageBox.warning(self, "错误", "请先完成 OCR 或载入 JSON")
            return
        if not self._confirm_reapply(base, steps, "Formatter 步骤"):
            return
        token = self._begin_formatter_pipeline_operation()
        if token is None:
            return

        base = self._snapshot_clone_document(base)
        base.metadata.pdf_text_layer_mode = self._pdf_text_layer_mode_cb.isChecked()
        base.metadata.pdf_keep_afterwords = self._pdf_keep_afterwords_cb.isChecked()
        base.metadata.preserve_ocr_layout = (
            self._preserve_ocr_layout_cb.isChecked()
            and not base.metadata.pdf_text_layer_mode
        )

        self._show_doc(self._ocr_doc or base, self._before)
        self._progress.setVisible(True)
        internal_pdf_steps = 2 if base.metadata.pdf_text_layer_mode and steps else 0
        self._progress.setRange(0, len(steps) + internal_pdf_steps)

        def worker():
            try:
                from engine.formatter import run_pipeline
                def on_progress(step_name, current, total):
                    signals.progress.emit(current, total)
                result = run_pipeline(base, steps=steps, verbose=False, progress_callback=on_progress)
                signals.finished.emit(result)
            except Exception as e:
                import traceback
                signals.error.emit(traceback.format_exc())

        signals = self._pipeline_signals = WorkerSignals()
        signals.finished.connect(
            lambda result, t=token: self._on_run_done(result)
            if self._finish_formatter_pipeline_operation(t) else None
        )
        signals.error.connect(
            lambda message, t=token: self._on_error(message)
            if self._finish_formatter_pipeline_operation(t) else None
        )
        signals.progress.connect(
            lambda cur, _total, t=token: self._progress.setValue(cur)
            if self._formatter_operation_current(t) else None
        )
        threading.Thread(target=worker, daemon=True).start()

    def _commit_operation_result(self, result: UnifiedDocument, step: str):
        current = self._current_doc()
        if current is not None and result.repo is None:
            result.repo = current.repo
            result.commit_id = current.commit_id
        repo_path = str(result.repo.path) if result.repo is not None else new_temp_repo_path()
        message = ""
        if result.processing_log:
            message = result.processing_log[-1].get("message", "")
        result.commit(repo_path, step, message)

    def _on_run_done(self, result, commit_step: Optional[str] = None):
        if commit_step:
            self._commit_operation_result(result, commit_step)
        self._fmt_doc = result
        self._progress.setVisible(False)
        self._refresh_editor_views()
        self._update_version(result)
        if bool(getattr(result.metadata, "pdf_text_layer_mode", False)):
            report = dict(getattr(result.metadata, "pdf_text_guard_report", {}) or {})
            if report.get("passed"):
                self._formatter_ai_status.setText(
                    f"PDF文字层处理完成：字符保全通过（{report.get('source_chars', 0)}→{report.get('output_chars', 0)}）"
                )
            else:
                self._formatter_ai_status.setText(
                    "PDF文字层处理完成，但字符保全未通过："
                    f"疑似丢失 {report.get('missing_chars', 0)} 字，额外 {report.get('extra_chars', 0)} 字；请在文字校对中复核"
                )
        self.doc_formatted.emit(result)

    def _on_error(self, msg):
        self._progress.setVisible(False)
        show_error_dialog(self, "处理失败", msg)

    def _set_formatter_ai_buttons_enabled(self, enabled: bool):
        for button in self._formatter_ai_buttons:
            button.setEnabled(bool(enabled))

    def _set_formatter_action_buttons_enabled(self, enabled: bool):
        for button in self._formatter_action_buttons:
            button.setEnabled(bool(enabled))
        # A worker starts from an immutable copy.  Lock both panes so edits made
        # after launch cannot be silently replaced when the result arrives.
        if hasattr(self, "_before"):
            self._before.setReadOnly(not enabled)
        if hasattr(self, "_after"):
            self._after.setReadOnly(not enabled)

    def _formatter_operation_current(self, token: int) -> bool:
        return self._content_generation.is_current(token)

    def _begin_formatter_pipeline_operation(self) -> int | None:
        if self._formatter_pipeline_running or self._formatter_ai_running:
            notify(self, "请先等待当前 Formatter 任务完成，或停止正在运行的 AI 任务。", "warning")
            return None
        token = self._content_generation.begin()
        self._formatter_pipeline_running = True
        self._set_formatter_action_buttons_enabled(False)
        self._set_formatter_ai_buttons_enabled(False)
        return token

    def _finish_formatter_pipeline_operation(self, token: int) -> bool:
        if not self._formatter_operation_current(token):
            return False
        self._formatter_pipeline_running = False
        self._set_formatter_action_buttons_enabled(True)
        self._set_formatter_ai_buttons_enabled(True)
        return True

    def _on_formatter_ai_progress(self, event: dict):
        stage = str(event.get("stage", "AI处理中"))
        current = int(event.get("current", 0) or 0)
        total = max(1, int(event.get("total", 1) or 1))
        chapter = str(event.get("chapter", "") or "")
        self._progress.setVisible(True)
        self._progress.setRange(0, 100)
        if "overall_percent" in event:
            value = int(float(event.get("overall_percent", 0) or 0))
        else:
            value = int(current * 100 / total)
        self._progress.setValue(max(0, min(100, value)))
        self._progress.setFormat(f"{max(0, min(100, value))}%")

        details = []
        if chapter:
            details.append(chapter)
        completed_chapters = int(event.get("completed_chapters", 0) or 0)
        total_chapters = int(event.get("total_chapters", 0) or 0)
        if total_chapters:
            details.append(f"章节 {completed_chapters}/{total_chapters}")
        token_total = int(event.get("token_total", 0) or 0)
        if token_total:
            token_label = "实际" if event.get("token_actual") else "估算"
            details.append(f"Token {token_total:,}（{token_label}）")
        resumed = int(event.get("resumed_batches", 0) or 0)
        unit = str(event.get("unit", "batch") or "batch")
        unit_zh = "条" if unit == "row" else "批"
        if resumed:
            details.append(f"断点恢复 {resumed} {unit_zh}")
        requests = int(event.get("requests", 0) or 0)
        if requests:
            details.append(f"请求 {requests}")
        suffix = " · " + " · ".join(details) if details else ""
        self._formatter_ai_status.setText(f"{stage}（{unit_zh} {current}/{total}）{suffix}")

    def _formatter_checkpoint_directory(self) -> str:
        """Stable across app replacement and Markdown re-import; migrates legacy roots."""
        from engine.ai_checkpoint_store import prepare_checkpoint_root
        root = prepare_checkpoint_root(self._formatter_source_path, self._current_doc(), self._formatter_checkpoint_override)
        probe = root / ".write_test"
        probe.write_text("ok", encoding="utf-8"); probe.unlink(missing_ok=True)
        self._formatter_ai_checkpoint_root = str(root)
        return str(root)

    def _open_formatter_checkpoint_directory(self):
        try:
            path = self._formatter_checkpoint_directory()
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        except Exception as exc:
            QMessageBox.warning(self, "断点目录", str(exc))

    def _choose_formatter_checkpoint_directory(self):
        path = QFileDialog.getExistingDirectory(self, "选择本书断点目录", self._formatter_ai_checkpoint_root or str(Path.home()))
        if path:
            self._formatter_checkpoint_override = path
            self._formatter_ai_checkpoint_root = path
            self._formatter_ai_status.setText(f"已指定断点目录：{path}")

    def _clear_formatter_checkpoint_directory(self):
        try: path = Path(self._formatter_checkpoint_directory())
        except Exception as exc:
            QMessageBox.warning(self, "断点目录", str(exc)); return
        if QMessageBox.question(self, "清除断点", f"确定删除本书全部 AI 断点？\n{path}") != QMessageBox.Yes: return
        shutil.rmtree(path, ignore_errors=True); path.mkdir(parents=True, exist_ok=True)
        self._formatter_ai_status.setText("本书 AI 断点已清除")

    def _run_formatter_ai(self, action: str):
        """Run Formatter-specific AI with its own isolated prompt/rule contract.

        For layout-related actions, any checked local Formatter steps that have not
        yet run are applied first.  This keeps deterministic short-fragment work
        local, while AI focuses on long/ambiguous passages.
        """
        if self._formatter_ai_running:
            notify(self, "Formatter AI 任务尚未完成", "warning")
            return
        if self._formatter_pipeline_running:
            notify(self, "请先等待当前 Formatter 导入或本地规则任务完成。", "warning")
            return
        if not self._ensure_editor_state_saved():
            return
        source = self._current_doc()
        if source is None:
            QMessageBox.warning(self, "提示", "请先导入文档或完成 OCR")
            return

        settings = ensure_ai_settings(
            self,
            "请先选择服务商、填写模型和 API Key。Formatter 使用全局连接设置。",
        )
        if settings is None:
            return

        labels = {
            "layout": "AI排版",
            "correction": "AI纠错",
            "correction_layout": "AI纠错 + 排版",
        }
        label = labels.get(action, "Formatter AI")
        step_name = f"formatter_ai_{action}"
        if not self._confirm_reapply(source, [step_name], label):
            return

        source = self._snapshot_clone_document(source)
        source.metadata.pdf_text_layer_mode = self._pdf_text_layer_mode_cb.isChecked()
        source.metadata.pdf_keep_afterwords = self._pdf_keep_afterwords_cb.isChecked()
        source.metadata.preserve_ocr_layout = (
            self._preserve_ocr_layout_cb.isChecked()
            and not source.metadata.pdf_text_layer_mode
        )
        selected_steps = [
            sid for sid, _label, _badge, _desc in FORMATTER_STEPS
            if self._step_checks[sid].isChecked()
        ]
        # 宽松 Formatter：AI 前允许重新运行结构规则。布局相关操作执行全部
        # 已勾选规则；单独纠错至少执行翻页、重叠、断句和对白恢复。
        loose_structure_steps = {
            "cross_page_merge", "merge_overlaps", "repair_dialogue_quotes",
            "merge_sentences", "remove_duplicates", "dialogue_restore",
        }
        if action in {"layout", "correction_layout"}:
            missing_local_steps = list(selected_steps)
        else:
            missing_local_steps = [sid for sid in selected_steps if sid in loose_structure_steps]

        token = self._content_generation.begin()
        cancel_event = threading.Event()
        self._formatter_ai_cancel_event = cancel_event
        self._formatter_ai_running = True
        self._formatter_ai_cancel_requested = False
        self._set_formatter_ai_buttons_enabled(False)
        self._set_formatter_action_buttons_enabled(False)
        self._progress.setVisible(True)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._formatter_ai_status.setText(f"{label}：准备中")

        def worker():
            try:
                working = source
                local_steps_done = []
                if missing_local_steps:
                    signals.log.emit(json.dumps({
                        "stage": "本地规则预处理", "current": 0,
                        "total": len(missing_local_steps), "unit": "local",
                    }, ensure_ascii=False))
                    from engine.formatter import run_pipeline
                    def local_progress(step_name, current, total):
                        signals.log.emit(json.dumps({
                            "stage": f"本地规则：{step_name}",
                            "current": current,
                            "total": max(1, total),
                            "unit": "local",
                        }, ensure_ascii=False))
                    working = run_pipeline(
                        working.snapshot_clone(),
                        steps=missing_local_steps,
                        verbose=False,
                        progress_callback=local_progress,
                    )
                    local_steps_done = list(missing_local_steps)

                from ai.provider_factory import create_provider
                from engine.ai_document_processor import run_ai_document
                from engine.formatter_ai import formatter_ai_prompt
                processor_mode, prompt = formatter_ai_prompt(action)

                checkpoint_root = self._formatter_checkpoint_directory()

                def phase_report(phase_label: str, start_percent: int, end_percent: int):
                    def _report(event):
                        event = dict(event or {})
                        current = int(event.get("current", 0) or 0)
                        total = max(1, int(event.get("total", 1) or 1))
                        fraction = current / total
                        event["overall_percent"] = start_percent + fraction * (end_percent - start_percent)
                        event["stage"] = f"{phase_label} · {event.get('stage', 'AI处理中')}"
                        signals.log.emit(json.dumps(event, ensure_ascii=False))
                    return _report

                signals.log.emit(json.dumps({
                    "stage": label, "independent_prompt": True, "independent_prompt_marker": "independent_prompt=true"
                }, ensure_ascii=False))
                with create_provider(settings) as provider:
                    result, changes = run_ai_document(
                        provider,
                        working,
                        prompt_template=prompt,
                        mode=processor_mode,
                        progress_callback=phase_report(label, 2, 98),
                        cancel_check=cancel_event.is_set,
                        request_css=False,
                        layout_lock=False,
                        cleanup_replacement_fragments=True,
                        checkpoint_dir=str(Path(checkpoint_root) / action), resume=True,
                        glossary_path=getattr(settings, "glossary_path", ""),
                    )
                result.metadata.ai_processing_mode = f"formatter_{action}"
                result.metadata.ai_layout_locked = False
                result.metadata.authoritative_text = False
                authority_report = None
                result.add_log(
                    step_name,
                    f"Formatter {label}; local_prepass={','.join(local_steps_done) or 'none'}; loose_structure=true; ocr_repair_mode={getattr(settings, 'ocr_repair_mode', 'readability')}",
                    len(changes),
                )
                signals.finished.emit({
                    "doc": result,
                    "changes": changes,
                    "action": action,
                    "label": label,
                    "local_steps": local_steps_done,
                    "authority_report": authority_report.to_dict() if authority_report else None,
                    "checkpoint_root": checkpoint_root,
                })
            except Exception as exc:
                from ai.config import APIKeyValidationError
                if isinstance(exc, APIKeyValidationError) or "ascii' codec can't encode" in str(exc):
                    signals.error.emit(f"AI 配置错误：{exc}\n\n请打开 AI 设置，清空 API Key 输入框后重新粘贴真实密钥。不要填写中文提示、Bearer 前缀或接口网址。")
                else:
                    import traceback
                    signals.error.emit(traceback.format_exc())

        signals = self._formatter_ai_signals = WorkerSignals()
        signals.log.connect(
            lambda raw, t=token: self._on_formatter_ai_progress(json.loads(raw))
            if self._formatter_operation_current(t) else None
        )
        signals.finished.connect(
            lambda payload, t=token: self._on_formatter_ai_done(payload)
            if self._formatter_operation_current(t) else None
        )
        signals.error.connect(
            lambda details, t=token: self._on_formatter_ai_failed(details)
            if self._formatter_operation_current(t) else None
        )
        threading.Thread(target=worker, daemon=True).start()

    def _on_formatter_ai_done(self, payload: dict):
        self._formatter_ai_running = False
        self._set_formatter_ai_buttons_enabled(True)
        self._set_formatter_action_buttons_enabled(True)
        result = payload["doc"]
        action = payload.get("action", "layout")
        # AI 排版后的简单、可确定结构交给 Formatter 内置规则收尾：
        # 完整的「……」人物对白必须独立成块；不完整引号不猜测、不拆分。
        if action in {"layout", "correction_layout"}:
            from engine.formatter import restore_dialogue_breaks
            result = restore_dialogue_breaks(result)
        step_name = f"formatter_ai_{action}"
        self._on_run_done(result, commit_step=step_name)
        self._progress.setVisible(True)
        self._progress.setRange(0, 100)
        self._progress.setValue(100)
        self._progress.setFormat("100%")
        local_count = len(payload.get("local_steps") or [])
        changes = len(payload.get("changes") or [])
        suffix = "；宽松结构规则已运行"
        self._formatter_ai_status.setText(
            f"✓ {payload.get('label', 'Formatter AI')}完成：本地补跑 {local_count} 步，AI 修改 {changes} 处{suffix}"
        )
        QTimer.singleShot(1800, lambda: self._progress.setVisible(False))

    def _on_formatter_ai_failed(self, details: str):
        self._formatter_ai_running = False
        self._set_formatter_ai_buttons_enabled(True)
        self._set_formatter_action_buttons_enabled(True)
        self._progress.setVisible(False)
        checkpoint_note = f"；断点保存在 {self._formatter_ai_checkpoint_root}" if self._formatter_ai_checkpoint_root else ""
        if "AI任务已停止" in details:
            self._formatter_ai_status.setText("Formatter AI 已停止，已完成批次不会丢失，下次自动续跑" + checkpoint_note)
            return
        self._formatter_ai_status.setText("Formatter AI 处理失败，已完成批次断点已保存，下次自动续跑" + checkpoint_note)
        show_error_dialog(self, "Formatter AI 处理失败", details)


    def _materialize_visible_surface(self) -> None:
        if getattr(self, "_pdf_format_view", None) is not None and self._pdf_format_view.isVisible():
            self._materialize_pdf_preview()
        elif self._advanced_editors_visible():
            self._materialize_advanced_editors()
        else:
            self._sync_compact_preview_text()

    def showEvent(self, event):
        super().showEvent(event)
        # ``set_doc`` is frequently called while the Formatter page is hidden.
        # Defer the expensive surface hydration until Qt has made the page
        # visible so ``isVisible()`` reflects the actual active workspace.
        QTimer.singleShot(0, self._materialize_visible_surface)


class FormatProfileDialog(QDialog):
    """
    排版格式管理：denki/mf/web 三个内置模板只读展示，自定义格式支持
    新建、从参考 EPUB 学习格式、编辑 CSS、保存、删除、导入、导出。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("排版格式管理")
        self.resize(780, 540)
        self.store = FormatProfileStore()
        self._current: Optional[FormatProfile] = None
        self._is_builtin = False
        self._learn_generation = GenerationGuard()
        self._learn_running = False
        self._build()
        self._reload_list()
        if self._list.count():
            self._list.setCurrentRow(0)

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(14)

        # ── 左侧：格式列表 ────────────────────────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(210)
        left.setMaximumWidth(280)
        left.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(8)
        ll.addWidget(QLabel("<b>格式列表</b>"))
        self._list = QListWidget()
        self._list.currentItemChanged.connect(self._on_select)
        ll.addWidget(self._list, 1)
        new_btn = QPushButton("＋ 新建空白格式")
        new_btn.setProperty("flat", True)
        new_btn.clicked.connect(self._new_blank)
        ll.addWidget(new_btn)
        import_btn = QPushButton("📥 导入格式…")
        import_btn.setProperty("flat", True)
        import_btn.clicked.connect(self._import)
        ll.addWidget(import_btn)
        root.addWidget(left, 0)

        vline = QFrame()
        vline.setFrameShape(QFrame.VLine)
        vline.setStyleSheet(f"background-color: {BORDER}; max-width: 1px; border: none;")
        root.addWidget(vline)

        # ── 右侧：详情/编辑 ───────────────────────────────────────────────────
        right = QWidget()
        right.setMinimumWidth(520)
        right.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)

        form = QFormLayout()
        self._name_edit = QLineEdit()
        form.addRow("名称:", self._name_edit)
        rl.addLayout(form)

        self._learn_btn = accent_button("📖 从参考 EPUB 学习格式…", color="#AF52DE")
        self._learn_btn.clicked.connect(self._learn_from_epub)
        rl.addWidget(self._learn_btn)

        self._notes_lbl = QLabel("")
        self._notes_lbl.setWordWrap(True)
        self._notes_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        rl.addWidget(self._notes_lbl)

        rl.addWidget(QLabel("CSS:"))
        self._css_edit = QPlainTextEdit()
        rl.addWidget(self._css_edit, 1)

        btn_row = QHBoxLayout()

        # 保存按钮（绿色主按钮）
        self._save_btn = accent_button("💾 保存", color=SUCCESS)
        self._save_btn.setAutoDefault(False)
        self._save_btn.setDefault(False)
        self._save_btn.clicked.connect(self._save)
        btn_row.addWidget(self._save_btn)

        # 导出按钮（蓝色主按钮，不再透明）
        export_btn = accent_button("📤 导出…", color=ACC)
        export_btn.clicked.connect(self._export)
        btn_row.addWidget(export_btn)

        # 删除按钮（浅红背景 + 红色文字）
        self._delete_btn = QPushButton("🗑 删除")
        self._delete_btn.setStyleSheet(
            f"background-color: {blend(DANGER, 0.15, CARD)}; "
            f"color: {DANGER}; border: 1px solid {DANGER}; "
            f"border-radius: 8px; padding: 8px 16px; font-weight: 600;"
        )
        self._delete_btn.clicked.connect(self._delete)
        btn_row.addWidget(self._delete_btn)

        btn_row.addStretch()

        # 关闭按钮（灰色背景）
        close_btn = accent_button("关闭", color="#E2E5E9", text_color=INK)
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)

        rl.addLayout(btn_row)

        # This is an editor dialog (name + CSS).  Enter belongs to the editor,
        # never to Save/Delete/Export or any automatically focused toolbar button.
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
            button.setDefault(False)

        root.addWidget(right, 1)

    def _enforce_editor_default_policy(self) -> None:
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
            button.setDefault(False)
        if hasattr(self, "_list"):
            self._list.setFocus(Qt.OtherFocusReason)

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._enforce_editor_default_policy)

    # ── 列表 ────────────────────────────────────────────────────────────────

    def _reload_list(self):
        from builder.epub_builder import CSS_TEMPLATES
        self._list.clear()
        BUILTIN_LABELS = {"denki": "denki（電撃文庫・竖排）", "mf": "mf（MF文庫J・竖排）", "web": "web（横排）"}
        for key in CSS_TEMPLATES:
            item = QListWidgetItem(f"🔒 {BUILTIN_LABELS.get(key, key)}")
            item.setData(Qt.UserRole, ("builtin", key))
            self._list.addItem(item)
        for p in self.store.list():
            item = QListWidgetItem(f"📝 {p.name}")
            item.setData(Qt.UserRole, ("custom", p.id))
            self._list.addItem(item)

    def _select_profile_id(self, profile_id: str):
        for i in range(self._list.count()):
            item = self._list.item(i)
            kind, key = item.data(Qt.UserRole)
            if kind == "custom" and key == profile_id:
                self._list.setCurrentItem(item)
                return

    def _on_select(self, current, previous):
        if current is None:
            return
        kind, key = current.data(Qt.UserRole)
        if kind == "builtin":
            from builder.epub_builder import CSS_TEMPLATES
            self._current = None
            self._is_builtin = True
            self._name_edit.setText(key)
            self._name_edit.setEnabled(False)
            self._css_edit.setPlainText(CSS_TEMPLATES[key])
            self._css_edit.setReadOnly(True)
            self._notes_lbl.setText("内置模板，只读——想在它基础上改，先「新建空白格式」再把这段 CSS 复制过去编辑。")
            self._save_btn.setEnabled(False)
            self._delete_btn.setEnabled(False)
        else:
            profile = self.store.get(key)
            self._current = profile
            self._is_builtin = False
            self._name_edit.setEnabled(True)
            self._name_edit.setText(profile.name if profile else "")
            self._css_edit.setReadOnly(False)
            self._css_edit.setPlainText(profile.css if profile else "")
            self._notes_lbl.setText((profile.notes if profile else "") or "手动新建的空白格式，还没有参考来源说明。")
            self._save_btn.setEnabled(True)
            self._delete_btn.setEnabled(True)

    # ── 操作 ────────────────────────────────────────────────────────────────

    def _new_blank(self):
        from builder.epub_builder import CSS_TEMPLATES, DEFAULT_TEMPLATE
        name, ok = QInputDialog.getText(self, "新建格式", "格式名称:")
        if not ok or not name.strip():
            return
        profile = FormatProfile(name=name.strip(), css=CSS_TEMPLATES[DEFAULT_TEMPLATE], source="manual")
        self.store.save(profile)
        self._reload_list()
        self._select_profile_id(profile.id)

    def _learn_from_epub(self):
        if self._learn_running:
            return
        path, _ = QFileDialog.getOpenFileName(self, "选择参考 EPUB（日文/中文）", "", "EPUB (*.epub)")
        if not path:
            return
        default_name = Path(path).stem
        name, ok = QInputDialog.getText(self, "格式名称", "给学到的格式起个名字:", text=default_name)
        if not ok or not name.strip():
            return
        name = name.strip()
        token = self._learn_generation.begin()
        self._learn_running = True
        self._learn_btn.setEnabled(False)
        self._learn_btn.setText("正在学习格式…")

        def worker():
            try:
                profile = FormatProfile.from_reference_epub(path, name)
                signals.finished.emit(profile)
            except Exception:
                import traceback
                signals.error.emit(traceback.format_exc())

        signals = self._learn_signals = WorkerSignals()
        signals.finished.connect(lambda profile, t=token: self._on_learned(profile, t))
        signals.error.connect(lambda msg, t=token: self._on_learn_failed(msg, t))
        threading.Thread(target=worker, daemon=True).start()

    def _finish_learn(self, token: int) -> bool:
        if not self._learn_generation.is_current(token):
            return False
        self._learn_running = False
        self._learn_btn.setEnabled(True)
        self._learn_btn.setText("📖 从参考 EPUB 学习格式…")
        return True

    def _on_learn_failed(self, message: str, token: int):
        if not self._finish_learn(token):
            return
        show_error_dialog(self, "学习格式失败", message)

    def _on_learned(self, profile: FormatProfile, token: int):
        if not self._finish_learn(token):
            return
        self.store.save(profile)
        self._reload_list()
        self._select_profile_id(profile.id)

        css_len = len(profile.css.strip()) if profile.css else 0
        msg = f"已从参考 EPUB 学习格式「{profile.name}」"

        if css_len > 0:
            msg += f"\n\n✅ 排版规则已学习（{css_len} 字符）"
            if "inline style" in profile.css:
                msg += "\n来源：XHTML 内联样式分析"
            elif "structure" in profile.css:
                msg += "\n来源：XHTML 结构分析"
            else:
                msg += "\n来源：EPUB CSS"

            if profile.vertical:
                msg += "\n📖 检测到竖排格式"

            notify(self, msg, "success")
        else:
            msg += (
                "\n\n⚠️ 未发现可用 CSS"
                "\n\n已尝试："
                "\n✓ 独立 CSS 文件"
                "\n✓ XHTML <style> 标签"
                "\n✓ XHTML 内联 style"
                "\n✓ 页面结构分析"
            )
            QMessageBox.warning(self, "未发现 CSS", msg)

    def _save(self):
        if self._is_builtin or self._current is None:
            return
        name = self._name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "提示", "请输入格式名称")
            return
        self._current.name = name
        self._current.css = self._css_edit.toPlainText()
        self.store.save(self._current)
        self._reload_list()
        self._select_profile_id(self._current.id)
        notify(self, "格式已保存", "success")

    def _delete(self):
        if self._is_builtin or self._current is None:
            return
        ret = QMessageBox.question(
            self, "删除格式", ui_message("format.delete.confirm", name=self._current.name)
        )
        if ret != QMessageBox.Yes:
            return
        self.store.delete(self._current.id)
        self._current = None
        self._reload_list()
        self._css_edit.clear()
        self._name_edit.clear()
        self._notes_lbl.clear()

    def _export(self):
        if self._is_builtin or self._current is None:
            notify(self, "内置模板不支持导出，请先选中一个自定义格式", "info")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出格式", f"{self._current.name}.json", "JSON (*.json)")
        if not path:
            return
        self.store.export_to(self._current.id, path)
        notify(self, f"已导出: {path}", "success")

    def _import(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入格式", "", "JSON (*.json)")
        if not path:
            return
        try:
            profile = self.store.import_from(path)
        except Exception as e:
            QMessageBox.critical(self, "导入失败", str(e))
            return
        self._reload_list()
        self._select_profile_id(profile.id)
        notify(self, f"已导入格式「{profile.name}」", "success")
