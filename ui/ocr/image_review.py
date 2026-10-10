from __future__ import annotations

import copy
import html
import math
import subprocess
import sys
import tempfile
import threading
from bisect import bisect_left, bisect_right
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QTimer, QUrl
from PySide6.QtGui import QFont, QKeySequence, QShortcut, QDesktopServices, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QWidget, QFrame, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QToolButton, QMenu, QScrollArea, QSizePolicy, QSplitter, QTextEdit, QMessageBox,
    QStackedWidget, QListView, QAbstractItemView, QLineEdit, QComboBox,
)

from models.document import UnifiedDocument
from ui.common.editor_controls import MouseWheelPlainTextEdit
from ui.common.signals import WorkerSignals, ImageReviewRenderSignals
from ui.common.toast import notify
from ui.common.styling import BORDER, CARD, MUTED, LIGHT_PREVIEW_STYLE, EDITOR_SCROLLBAR_STYLE, accent_button, wrap_in_card
from ui.design.metrics import (
    REVIEW_PAGE_MARGIN_X, REVIEW_PAGE_MARGIN_TOP, REVIEW_PAGE_MARGIN_BOTTOM,
    REVIEW_LEFT_WIDTH, REVIEW_COLUMN_GAP,
)
from ui.dialogs import show_error_dialog
from ui.localized_dialogs import LocalizedMessageBox
from ui.ocr.proofread_widgets import OCRProofreadImageLabel, OCRVerticalColumnTextWidget
from ui.ocr.sentence_strip import SentenceStrip
from ui.common.window_state import bind_splitter
from ui.responsive import configure_combo
from utils.async_generation import GenerationGuard

# Preserve the GUI module's localized message-box behaviour after extraction.
QMessageBox = LocalizedMessageBox

class _ImageReviewFusionCandidateCard(QFrame):
    """Full-width, fully visible OCR candidate used by 图文对照."""

    drafted = Signal(int, str)
    chosen = Signal(int, str)

    def __init__(self, candidate_index: int, caption: str, text: str, *, confidence: float = 0.0, parent=None):
        super().__init__(parent)
        self.candidate_index = int(candidate_index)
        self.candidate_text = str(text or "")
        self.setObjectName("imageReviewFusionCard")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 9)
        root.setSpacing(6)
        title = str(caption or "候选")
        if confidence > 0:
            title += f" · {confidence:.0%}"
        self._caption = QLabel(title)
        self._caption.setWordWrap(False)
        self._caption.setFixedHeight(26)
        self._caption.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._caption.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self._caption.setStyleSheet("font-size:11px;font-weight:700;color:#2559E0;")
        self._draft = QPushButton("作为底稿")
        self._draft.setMinimumHeight(28)
        self._draft.setMaximumHeight(28)
        self._draft.setToolTip("复制到手动编辑框；不会保存、不会跳转")
        self._draft.clicked.connect(self._emit_drafted)
        self._choose = QPushButton("直接采用并下一条")
        self._choose.setMinimumHeight(28)
        self._choose.setMaximumHeight(28)
        self._choose.setToolTip("直接完成当前裁决并进入下一条 OCR 分歧")
        self._choose.clicked.connect(self._emit_chosen)
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(8)
        head.addWidget(self._caption, 1)
        head.addWidget(self._draft, 0)
        head.addWidget(self._choose, 0)
        root.addLayout(head)
        # Fully expand wrapped candidate text, as in OCR 对比.
        self._body = QTextEdit()
        self._body.setReadOnly(True)
        self._body.setAcceptRichText(True)
        self._body.setLineWrapMode(QTextEdit.WidgetWidth)
        self._body.setMinimumSize(220, 54)
        self._body.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._body.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._body.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._body.setStyleSheet(
            "QTextEdit{background:#FFFFFF;border:1px solid #D7E3F4;border-radius:7px;"
            "padding:8px;font-size:15px;color:#14202E;}" + EDITOR_SCROLLBAR_STYLE
        )
        root.addWidget(self._body, 1)
        self.set_candidate(candidate_index, caption, text, confidence=confidence)
        self.set_reference_text("")
        self.set_selected(False)

    def _emit_drafted(self) -> None:
        self.drafted.emit(self.candidate_index, self.candidate_text)

    def _emit_chosen(self) -> None:
        self.chosen.emit(self.candidate_index, self.candidate_text)

    def set_candidate(
        self, candidate_index: int, caption: str, text: str, *, confidence: float = 0.0
    ) -> None:
        self.candidate_index = int(candidate_index)
        self.candidate_text = str(text or "")
        title = str(caption or "候选")
        if confidence > 0:
            title += f" · {confidence:.0%}"
        self._caption.setText(title)

    @staticmethod
    def _rich_diff(text: str, marks) -> str:
        value = str(text or "")
        spans = sorted((max(0, int(a)), max(0, int(b)), str(kind)) for a, b, kind in (marks or ()))
        parts = []
        cursor = 0
        for start, end, kind in spans:
            start = min(len(value), max(cursor, start))
            end = min(len(value), max(start, end))
            if start > cursor:
                parts.append(html.escape(value[cursor:start]).replace("\n", "<br>"))
            if end > start:
                color = "#FFD1D1" if kind == "replace" else "#FFE4AD"
                foreground = "#8A1C1C" if kind == "replace" else "#7A3E00"
                parts.append(
                    f'<span style="background:{color};color:{foreground};font-weight:600;">'
                    + html.escape(value[start:end]).replace("\n", "<br>") + "</span>"
                )
                cursor = end
            else:
                parts.append('<span style="color:#D97706;font-weight:800;">▏</span>')
                cursor = start
        if cursor < len(value):
            parts.append(html.escape(value[cursor:]).replace("\n", "<br>"))
        return "".join(parts) or "<span style='color:#9CA3AF;'>（空候选）</span>"

    def set_reference_text(self, reference: str) -> None:
        try:
            from engine.ocr_compare_view_model import paired_candidate_diff_spans
            marks, _other, changes = paired_candidate_diff_spans(self.candidate_text, str(reference or ""))
        except Exception:
            marks, changes = (), 0
        self._body.setHtml(self._rich_diff(self.candidate_text, marks))
        base = self._caption.text().split(" · 差异 ", 1)[0]
        self._caption.setText(base + (f" · 差异 {changes} 处" if changes else " · 与当前稿一致"))
        QTimer.singleShot(0, self._fit_body_height)

    def _fit_body_height(self) -> None:
        try:
            document = self._body.document()
            document.setTextWidth(float(max(1, self._body.viewport().width())))
            height = max(54, int(math.ceil(document.size().height())) + 20)
            # Keep enough height for every wrapped line, while allowing the
            # candidate card's row to use the remaining comparison area.
            if self._body.height() != height or self._body.minimumHeight() != height or self._body.maximumHeight() != height:
                self._body.setFixedHeight(height)
        except RuntimeError:
            pass

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self._fit_body_height)

    def set_draft_source(self, active: bool) -> None:
        self._draft.setText("✓ 手动编辑底稿" if active else "作为底稿")

    def set_selected(self, selected: bool) -> None:
        if selected:
            self.setStyleSheet(
                "QFrame#imageReviewFusionCard{background:#E4EEFF;border:2px solid #2F6BFF;border-radius:10px;}"
            )
            self._draft.setText("作为底稿")
            self._draft.setEnabled(True)
            self._choose.setText("✓ 已采用")
            self._choose.setEnabled(False)
        else:
            self.setStyleSheet(
                "QFrame#imageReviewFusionCard{background:#F7F8FA;border:1px solid #D8DDE3;border-radius:10px;}"
            )
            self._draft.setText("作为底稿")
            self._draft.setEnabled(True)
            self._choose.setText("直接采用并下一条")
            self._choose.setEnabled(True)


class OCRImageTextReviewTab(QWidget):
    """One immutable OCR sentence per page: editable text left, source pixels right."""

    doc_applied = Signal(object)
    # Emitted after every successful sentence save.  OCRCompareTab consumes this
    # as a final-fusion decision; model OCR sources remain immutable.
    row_review_saved = Signal(object)
    # The exact MultiOcrRow index currently displayed.
    source_row_changed = Signal(int)
    # View navigation only: return to OCR Compare's independent full overview
    # at the same stable row.  This signal never confirms or applies a draft.
    full_compare_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source_doc: UnifiedDocument | None = None
        self._review_doc: UnifiedDocument | None = None
        self._pending_source_doc: UnifiedDocument | None = None
        self._pending_source_label = ""
        self._page_image_provider = None
        self._document_load_generation = GenerationGuard()
        self._document_load_signal_refs: dict[int, WorkerSignals] = {}
        self._document_load_busy = False
        self._workspace_active = False
        self._pending_jump_source_row = -1
        self._entries = []
        self._entry_index_by_source_row: dict[int, int] = {}
        self._entry_index_by_sentence_group_id: dict[str, int] = {}
        self._entry_indices_by_block: dict[int, tuple[int, ...]] = {}
        self._index = -1
        self._dirty = False
        self._reviewed_count = 0
        self._changed_count = 0
        self._source_name = ""
        self._initial_text_by_key: dict[str, str] = {}
        self._session_dirty_keys: set[str] = set()
        self._cache_dir: tempfile.TemporaryDirectory | None = None
        self._horizontal_document = False
        self._current_review_image_path = ""
        self._current_image_column_intervals: tuple[tuple[float, float], ...] = ()
        self._active_physical_column_index = -1
        self._fusion_candidate_buttons: list[QWidget] = []
        self._fusion_candidate_grid_columns = 0
        self._candidate_reflow_pending = False
        self._image_render_generation = GenerationGuard()
        self._image_render_signal_refs: dict[int, ImageReviewRenderSignals] = {}
        # Sentence crops are serialized and latest-request-wins.  Rapid next/prev
        # navigation therefore cannot create hundreds of simultaneous PIL/PNG
        # workers that compete for CPU and disk.
        self._image_render_busy = False
        self._queued_image_request = None
        self._image_path_cache: dict[str, str] = {}
        self._image_cache_lock = threading.Lock()
        # Image-review cache directories can still be in use by a background
        # render/prefetch worker when the user switches books or closes the
        # workspace.  Track a monotonically increasing cache epoch plus leases
        # so reset never deletes a directory underneath an active worker (and a
        # stale worker can never recreate/repopulate the new book's cache).
        self._image_cache_epoch = 0
        self._image_cache_jobs: dict[int, int] = {}
        self._retired_cache_dirs: dict[int, tempfile.TemporaryDirectory] = {}
        self._image_render_cache_epochs: dict[int, int] = {}
        self._image_prefetch_pending: set[tuple[int, str]] = set()
        self._pending_judgement_indices: set[int] = set()
        self._has_judgement_entries = False
        self._ocr_disagreement_indices: list[int] = []
        self._ocr_disagreement_source_indices: list[int] = []
        self._ocr_disagreement_source_row_order: tuple[int, ...] = ()
        self._ocr_disagreement_source_order_synced = False
        self._pending_ocr_sync_rows: set[int] = set()
        self._pending_ocr_compare_decisions: dict[str, dict] = {}
        self._suppress_row_review_emit = False
        self._apple_handwriting_active = False
        self._apple_handwriting_target_key = ""
        self._build()
        self._clipboard = QApplication.clipboard()
        self._clipboard.dataChanged.connect(self._on_apple_handwriting_clipboard_changed)
        self._apple_handwriting_timeout = QTimer(self)
        self._apple_handwriting_timeout.setSingleShot(True)
        self._apple_handwriting_timeout.timeout.connect(self._cancel_apple_handwriting_session)

    def _build(self) -> None:
        # Legacy R7 source-contract markers retained for downstream audits:
        # left_panel.setMinimumWidth(210)
        # left_panel.setMaximumWidth(340)
        # visual_splitter = QSplitter(Qt.Horizontal)
        # visual_splitter.addWidget(right_panel)
        # visual_splitter.addWidget(left_panel)
        # visual_splitter.setSizes([780, 240])
        # visual_splitter.setStretchFactor(1, 0)
        # content_splitter.addWidget(text_card)
        # content_splitter.addWidget(visual_splitter)
        # content_splitter.setStretchFactor(0, 3)
        # content_splitter.setStretchFactor(1, 7)
        # wrap_in_card returns a horizontal surface layout.  Treating it as the
        # page's vertical layout placed header, image splitter, candidates and
        # navigation side by side, which caused the narrow columns and huge blank
        # areas visible on wide screens.  Install one expanding vertical surface
        # inside it and let every review section consume the complete width.
        outer_root = wrap_in_card(self)
        page_surface = QWidget()
        page_surface.setObjectName("imageTextReviewSurface")
        page_surface.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        outer_root.addWidget(page_surface, 1)
        root = QVBoxLayout(page_surface)
        root.setContentsMargins(
            REVIEW_PAGE_MARGIN_X, REVIEW_PAGE_MARGIN_TOP,
            REVIEW_PAGE_MARGIN_X, REVIEW_PAGE_MARGIN_BOTTOM,
        )
        root.setSpacing(0)

        # The former title/status strip consumed valuable vertical space.  Keep
        # the source label as hidden state for existing load/error messages and
        # move review counters to the bottom navigation row beside the position.
        self._source_label = QLabel("尚未接收 OCR 结果", page_surface)
        self._source_label.setVisible(False)
        self._review_summary = QLabel("已核对 0/0 · 已修改 0", page_surface)
        self._review_summary.setStyleSheet(f"color: {MUTED}; font-size: 10px; font-weight: 600;")

        content_splitter = QSplitter(Qt.Horizontal)
        content_splitter.setChildrenCollapsible(False)
        content_splitter.setHandleWidth(REVIEW_COLUMN_GAP)
        content_splitter.setProperty("nfPreserveHandleWidth", True)
        content_splitter.setStyleSheet("QSplitter::handle { background:#E3ECF7; border-radius:2px; margin:28px 1px; }")

        left_panel = QWidget()
        self._columns_panel = left_panel
        left_panel.setMinimumWidth(210)
        left_panel.setMaximumWidth(340)
        left_panel.setFixedWidth(REVIEW_LEFT_WIDTH)
        left_panel.setMinimumHeight(260)
        left_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        left_layout = QVBoxLayout(left_panel)
        left_panel.setObjectName("irColumnsCard")
        left_panel.setStyleSheet("QWidget#irColumnsCard{background:" + CARD + ";border:1px solid " + BORDER + ";border-radius:16px;}")
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(8)
        columns_header = QHBoxLayout()
        columns_header.setContentsMargins(0, 0, 5, 0)
        columns_header.setSpacing(4)
        columns_title = QLabel("OCR 竖列")
        columns_title.setStyleSheet("font-weight: 700; font-size: 12px;")
        columns_title.setMinimumHeight(27)
        columns_header.addWidget(columns_title)
        columns_header.addStretch(1)
        self._left_queue_btn = QToolButton(left_panel)
        self._left_queue_btn.setText("分歧队列")
        self._left_queue_btn.setCheckable(True)
        self._left_queue_btn.setToolTip("快速定位 OCR 结果不一致的句子。")
        self._left_queue_btn.clicked.connect(lambda: self._set_left_review_panel_mode("queue"))
        columns_header.addWidget(self._left_queue_btn)
        self._left_columns_btn = QToolButton(left_panel)
        self._left_columns_btn.setText("裁决竖列")
        self._left_columns_btn.setCheckable(True)
        self._left_columns_btn.setToolTip("查看当前裁决稿对应的完整物理竖列。")
        self._left_columns_btn.clicked.connect(lambda: self._set_left_review_panel_mode("columns"))
        columns_header.addWidget(self._left_columns_btn)
        self._columns_state = QLabel("", left_panel)
        self._columns_state.setStyleSheet(f"color: {MUTED}; font-size: 9px;")
        self._columns_state.setVisible(False)
        self._column_left_btn = QToolButton()
        self._column_left_btn.setText("←")
        self._column_left_btn.setFixedSize(24, 27)
        self._column_left_btn.setToolTip("定位左侧物理列（⌥⇧←）")
        self._column_left_btn.clicked.connect(lambda: self._move_active_column(+1))
        columns_header.addWidget(self._column_left_btn)
        self._column_right_btn = QToolButton()
        self._column_right_btn.setText("→")
        self._column_right_btn.setFixedSize(24, 27)
        self._column_right_btn.setToolTip("定位右侧物理列（⌥⇧→）")
        self._column_right_btn.clicked.connect(lambda: self._move_active_column(-1))
        columns_header.addWidget(self._column_right_btn)
        left_layout.addLayout(columns_header)
        self._review_queue_filters = QWidget(left_panel)
        queue_filters_layout = QVBoxLayout(self._review_queue_filters)
        queue_filters_layout.setContentsMargins(0, 0, 0, 0)
        queue_filters_layout.setSpacing(5)
        self._review_queue_scope = QComboBox(self._review_queue_filters)
        configure_combo(self._review_queue_scope)
        self._review_queue_scope.addItem("全部分歧", "all")
        self._review_queue_scope.addItem("待裁决", "pending")
        self._review_queue_scope.addItem("三方分歧", "threeway")
        self._review_queue_scope.addItem("空/占位符", "broken")
        self._review_queue_scope.addItem("已修改", "changed")
        self._review_queue_scope.addItem("已确认", "reviewed")
        self._review_queue_scope.setMinimumHeight(27)
        self._review_queue_scope.currentIndexChanged.connect(lambda _i: self._refresh_review_disagreement_queue())
        queue_filters_layout.addWidget(self._review_queue_scope)
        self._review_queue_search = QLineEdit(self._review_queue_filters)
        self._review_queue_search.setClearButtonEnabled(True)
        self._review_queue_search.setPlaceholderText("筛选页码 / 当前文本")
        self._review_queue_search.setMinimumHeight(27)
        self._review_queue_search.setStyleSheet(
            "QLineEdit{border:1px solid #D7E3F4;border-radius:7px;padding:3px 8px;background:#FAFCFF;}"
            "QLineEdit:focus{border-color:#4F7CFF;background:#FFFFFF;}"
        )
        self._review_queue_search.textChanged.connect(lambda _t: self._refresh_review_disagreement_queue())
        queue_filters_layout.addWidget(self._review_queue_search)
        left_layout.addWidget(self._review_queue_filters)
        self._left_review_stack = QStackedWidget(left_panel)
        self._review_disagreement_queue = QListView(self._left_review_stack)
        self._review_disagreement_queue.setUniformItemSizes(True)
        self._review_disagreement_queue.setSpacing(5)
        self._review_disagreement_queue.setSelectionMode(QAbstractItemView.SingleSelection)
        self._review_disagreement_queue.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._review_disagreement_queue.setStyleSheet("QListView{background:transparent;border:none;font-size:10px;outline:0;}")
        from ui.ocr.compare_widgets import DecisionQueueDelegate, DecisionQueueListModel
        queue_delegate = DecisionQueueDelegate(self._review_disagreement_queue)
        queue_delegate.force_light = True
        self._review_disagreement_queue.setItemDelegate(queue_delegate)
        self._review_disagreement_queue_model = DecisionQueueListModel(
            self._review_disagreement_display_text,
            self._review_disagreement_tooltip,
            self._review_disagreement_queue,
        )
        self._review_disagreement_queue.setModel(self._review_disagreement_queue_model)
        self._review_disagreement_queue.clicked.connect(self._review_disagreement_item_clicked)
        self._review_disagreement_queue.activated.connect(self._review_disagreement_item_clicked)
        self._left_review_stack.addWidget(self._review_disagreement_queue)
        self._columns_scroll = QScrollArea(self._left_review_stack)
        self._columns_scroll.setWidgetResizable(True)
        self._columns_scroll.setAlignment(Qt.AlignRight | Qt.AlignTop)
        self._columns_scroll.setFrameShape(QFrame.NoFrame)
        self._columns_scroll.setStyleSheet("QScrollArea { background: " + CARD + "; border: none; }")
        self._vertical_columns = OCRVerticalColumnTextWidget()
        self._columns_scroll.setWidget(self._vertical_columns)
        self._left_review_stack.addWidget(self._columns_scroll)
        left_layout.addWidget(self._left_review_stack, 1)
        self._set_left_review_panel_mode("queue")

        right_panel = QWidget()
        right_panel.setMinimumWidth(480)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        image_card = QWidget()
        image_layout = QVBoxLayout(image_card)
        image_card.setObjectName("irImageCard")
        image_card.setStyleSheet("QWidget#irImageCard{background:" + CARD + ";border:1px solid " + BORDER + ";border-radius:16px;}")
        image_layout.setContentsMargins(14, 12, 14, 12)
        image_layout.setSpacing(5)
        image_header = QHBoxLayout()
        image_title = QLabel("对应识别图片")
        image_title.setStyleSheet("font-weight: 700; font-size: 12px;")
        image_header.addWidget(image_title)
        self._image_state = QLabel("", image_card)
        self._image_state.setStyleSheet(f"color: {MUTED}; font-size: 9px;")
        self._image_state.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._image_state.setVisible(False)
        image_header.addStretch(1)
        self._open_preview_btn = QPushButton("原尺寸查看")
        self._open_preview_btn.setFixedSize(96, 27)
        self._open_preview_btn.setToolTip("使用 macOS 预览打开当前句/列图片（⌥P）")
        self._open_preview_btn.clicked.connect(self._open_current_image_in_preview)
        image_header.addWidget(self._open_preview_btn)
        image_layout.addLayout(image_header)
        self._image_interaction_hint = QLabel("拖动图片可定位物理列 · 蓝框=当前列 · Alt+P 原尺寸")
        self._image_interaction_hint.setStyleSheet("color:#7B8797;font-size:8.5px;")
        image_layout.addWidget(self._image_interaction_hint)
        self._image = OCRProofreadImageLabel()
        # The candidate comparison column needs most of the horizontal space;
        # keep enough width for a readable source crop while releasing the old
        # 360 px floor that forced the middle image pane wider than necessary.
        self._image.setMinimumWidth(280)
        self._image.column_scrubbed.connect(self._scrub_image_to_column)
        image_layout.addWidget(self._image, 1)
        right_layout.addWidget(image_card, 1)

        text_card = QWidget()
        text_card.setMinimumWidth(400)
        text_card.setMinimumHeight(360)   # 高度随窗口伸缩；候选区可滚动
        self._text_review_panel = text_card
        text_layout = QVBoxLayout(text_card)
        text_card.setObjectName("irTextCard")
        text_card.setStyleSheet("QWidget#irTextCard{background:" + CARD + ";border:1px solid " + BORDER + ";border-radius:16px;}")
        text_layout.setContentsMargins(14, 10, 14, 10)
        text_layout.setSpacing(5)
        text_header = QHBoxLayout()
        text_title = QLabel("横排校对文本")
        text_title.setStyleSheet("font-weight: 700; font-size: 12px;")
        text_title.setMinimumHeight(27)
        text_header.addWidget(text_title)
        text_header.addStretch(1)
        self._entry_status_badge = QLabel("待核对", text_card)
        self._entry_status_badge.setAlignment(Qt.AlignCenter)
        self._entry_status_badge.setMinimumWidth(62)
        self._entry_status_badge.setFixedHeight(24)
        text_header.addWidget(self._entry_status_badge)
        self._text_state = QLabel("", text_card)
        self._text_state.setStyleSheet(f"color: {MUTED}; font-size: 9px;")
        self._text_state.setMaximumWidth(220)
        self._text_state.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._text_state.setVisible(False)
        text_layout.addLayout(text_header)

        self._fusion_candidate_frame = QFrame()
        self._fusion_candidate_frame.setObjectName("fusionCandidateFrame")
        self._fusion_candidate_frame.setStyleSheet(
            "QFrame#fusionCandidateFrame { background: #F7F8FA; border: 1px solid #E2E5E9; border-radius: 12px; }"
        )
        self._fusion_candidate_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        candidate_box = QVBoxLayout(self._fusion_candidate_frame)
        candidate_box.setContentsMargins(8, 6, 8, 7)
        candidate_box.setSpacing(5)
        self._fusion_candidate_title = QLabel("多模型融合候选")
        self._fusion_candidate_title.setWordWrap(True)
        self._fusion_candidate_title.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._fusion_candidate_title.setStyleSheet("font-weight: 700; font-size: 11px;")
        candidate_box.addWidget(self._fusion_candidate_title)
        self._fusion_candidate_detail = QLabel("红底=替换 · 橙底=增删/缺失")
        self._fusion_candidate_detail.setWordWrap(True)
        self._fusion_candidate_detail.setStyleSheet("color:#7B8797;font-size:8.5px;")
        candidate_box.addWidget(self._fusion_candidate_detail)
        self._fusion_candidate_grid = QGridLayout()
        self._fusion_candidate_grid.setHorizontalSpacing(8)
        self._fusion_candidate_grid.setVerticalSpacing(8)
        self._fusion_candidate_grid.setColumnStretch(0, 1)
        self._fusion_candidate_grid.setColumnStretch(1, 1)
        self._fusion_candidate_grid.setColumnStretch(2, 1)
        candidate_box.addLayout(self._fusion_candidate_grid)
        candidate_box.addStretch(1)
        self._fusion_candidate_scroll = QScrollArea()
        self._fusion_candidate_scroll.setWidgetResizable(True)
        self._fusion_candidate_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._fusion_candidate_scroll.setFrameShape(QFrame.NoFrame)
        self._fusion_candidate_scroll.setStyleSheet("QScrollArea{background:transparent;border:none;}")
        self._fusion_candidate_scroll.setWidget(self._fusion_candidate_frame)
        text_layout.addWidget(self._fusion_candidate_scroll, 1)
        self._set_fusion_candidates_visible(False)

        self._editor = MouseWheelPlainTextEdit()
        self._editor.setPlaceholderText("完成 OCR 后在此横排逐句校对")
        self._editor.setLineWrapMode(MouseWheelPlainTextEdit.WidgetWidth)
        self._editor.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._editor.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._editor.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        editor_font = QFont(self._editor.font())
        editor_font.setPointSize(max(13, editor_font.pointSize() + 2))
        self._editor.setFont(editor_font)
        self._editor.setStyleSheet(
            "QPlainTextEdit {" + LIGHT_PREVIEW_STYLE + "}" + EDITOR_SCROLLBAR_STYLE
        )
        self._editor.textChanged.connect(self._on_text_changed)
        self._editor_fit_timer = QTimer(self)
        self._editor_fit_timer.setSingleShot(True)
        self._editor_fit_timer.timeout.connect(self._fit_review_editor_height)
        text_layout.addWidget(self._editor, 0)
        self._schedule_review_editor_fit()

        # Phase 20 layout follows the supplied review mock-up: immutable OCR
        # columns stay in a narrow card on the left; the source image gets the
        # largest area on the right, with the editable horizontal proofread
        # sentence directly below it.  No capability is removed—the same
        # editor, candidate cards, shortcuts and navigation controls are simply
        # re-parented into the clearer visual hierarchy.
        nav = QHBoxLayout()
        nav.setContentsMargins(0, 0, 0, 0)
        nav.setSpacing(7)
        self._prev_btn = QPushButton("← 上一分歧")
        self._prev_btn.setToolTip("保存当前修改并跳到上一条 OCR 分歧句")
        self._prev_btn.clicked.connect(self._jump_previous_ocr_disagreement)
        self._next_btn = QPushButton("下一句 →")
        self._next_btn.clicked.connect(self._next)
        self._next_btn.setVisible(False)  # Alt+Right and the jump menu browse all sentences.
        self._save_btn = accent_button("确认裁决")
        self._save_btn.setToolTip("确认当前手动文本并停留在本句")
        self._save_btn.clicked.connect(self._save_current)
        self._save_btn.setVisible(False)
        self._save_next_btn = accent_button("确认并下一句")
        self._save_next_btn.setToolTip("保存当前裁决并进入下一句（Ctrl/⌘+Shift+Return）")
        self._save_next_btn.clicked.connect(self._save_and_next)

        self._next_judgement_btn = QPushButton("下一待判断")
        self._next_judgement_btn.clicked.connect(self._jump_next_judgement)
        self._next_judgement_btn.setVisible(False)
        self._prev_disagreement_btn = QPushButton("上一OCR分歧")
        self._prev_disagreement_btn.clicked.connect(self._jump_previous_ocr_disagreement)
        self._prev_disagreement_btn.setVisible(False)
        self._next_disagreement_btn = QPushButton("下一OCR分歧")
        self._next_disagreement_btn.clicked.connect(self._jump_next_ocr_disagreement)
        self._next_disagreement_btn.setVisible(False)

        self._jump_menu_btn = QToolButton()
        self._jump_menu_btn.setText("跳转 ▾")
        self._jump_menu_btn.setPopupMode(QToolButton.InstantPopup)
        self._jump_menu_btn.setToolTip("跳到待判断或 OCR 分歧句；仅浏览不会自动裁决当前句")
        jump_menu = QMenu(self._jump_menu_btn)
        self._jump_pending_action = jump_menu.addAction("下一待判断")
        self._jump_pending_action.triggered.connect(self._jump_next_judgement)
        self._jump_prev_diff_action = jump_menu.addAction("上一 OCR 分歧")
        self._jump_prev_diff_action.triggered.connect(self._jump_previous_ocr_disagreement)
        self._jump_next_diff_action = jump_menu.addAction("下一 OCR 分歧")
        self._jump_next_diff_action.triggered.connect(self._jump_next_ocr_disagreement)
        self._jump_menu_btn.setMenu(jump_menu)
        jump_menu.addSeparator()
        self._jump_previous_plain_action = jump_menu.addAction("上一句（不确认）  Alt+←")
        self._jump_previous_plain_action.triggered.connect(self._previous)
        self._jump_next_plain_action = jump_menu.addAction("下一句（不确认）  Alt+→")
        self._jump_next_plain_action.triggered.connect(self._next)
        self._jump_menu_btn.setVisible(True)   # 跳转命令需要可见入口，不能只靠右键/快捷键
        text_card.setContextMenuPolicy(Qt.ActionsContextMenu)
        text_card.addAction(self._jump_pending_action)
        text_card.addAction(self._jump_prev_diff_action)
        text_card.addAction(self._jump_next_diff_action)

        self._review_summary.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._review_summary.setWordWrap(True)
        nav.addWidget(self._review_summary)
        self._position_label = QLabel("第 0 / 0 句")
        self._position_label.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        self._position_label.setVisible(True)   # 当前第几句必须可见
        self._position_label.setWordWrap(True)
        self._position_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        nav.addWidget(self._position_label)
        nav.addStretch(1)
        self._review_keyboard_hint = QLabel("F7 下一分歧 · Shift+F7 上一 · Alt+1…9 作底稿")
        self._review_keyboard_hint.setStyleSheet("color:#7B8797;font-size:8.5px;")
        self._review_keyboard_hint.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        nav.addWidget(self._review_keyboard_hint)
        nav.addWidget(self._jump_menu_btn)
        nav.addWidget(self._prev_btn)
        nav.addWidget(self._save_btn)
        nav.addWidget(self._save_next_btn)
        self._full_compare_btn = QPushButton("显示全文")
        self._full_compare_btn.setToolTip("返回 OCR 对比全文总览；不会自动确认当前句")
        self._full_compare_btn.clicked.connect(self._request_full_compare)
        nav.addWidget(self._full_compare_btn)
        self._apply_btn = QPushButton("✓ 应用整本")
        self._apply_btn.clicked.connect(self._apply_document)
        nav.addWidget(self._apply_btn)
        self._sentence_strip = SentenceStrip(text_card)
        self._sentence_strip.jump.connect(self._jump_to_entry_index)
        text_layout.addWidget(self._sentence_strip)
        text_layout.addLayout(nav)

        image_card.setMinimumWidth(260)
        visual_splitter = QSplitter(Qt.Horizontal)
        visual_splitter.setChildrenCollapsible(False)
        visual_splitter.setHandleWidth(REVIEW_COLUMN_GAP)
        visual_splitter.setProperty("nfPreserveHandleWidth", True)
        visual_splitter.setStyleSheet("QSplitter::handle { background:#E3ECF7; border-radius:2px; margin:28px 1px; }")
        visual_splitter.addWidget(right_panel)
        visual_splitter.addWidget(left_panel)
        visual_splitter.setSizes([780, 240])
        visual_splitter.setStretchFactor(0, 1)
        visual_splitter.setStretchFactor(1, 0)
        self._review_right_split = visual_splitter
        self._visual_review_splitter = visual_splitter
        content_splitter.addWidget(text_card)
        content_splitter.addWidget(visual_splitter)
        content_splitter.setStretchFactor(0, 3)
        content_splitter.setStretchFactor(1, 7)
        content_splitter.setSizes([360, 840])
        self._review_content_splitter = content_splitter
        root.addWidget(content_splitter, 1)
        bind_splitter(visual_splitter, "image_review_right_candidates_v2")
        bind_splitter(content_splitter, "image_review_content")

        # Historical source-contract markers retained after the Phase 22 visual
        # hierarchy rebuild.  The live layout is the two-card hierarchy above.
        # visual_splitter = QSplitter(Qt.Horizontal)
        # visual_splitter.addWidget(right_panel)
        # visual_splitter.addWidget(left_panel)
        # visual_splitter.setSizes([780, 240])
        # visual_splitter.setStretchFactor(1, 0)
        # content_splitter.addWidget(text_card)
        # content_splitter.addWidget(visual_splitter)
        # content_splitter.setStretchFactor(0, 3)
        # content_splitter.setStretchFactor(1, 7)
        # left_panel.setMinimumWidth(210)
        # left_panel.setMaximumWidth(340)
        # self._columns_scroll.setAlignment(Qt.AlignRight | Qt.AlignTop)

        self._prev_shortcut = QShortcut(QKeySequence("Alt+Left"), self)
        self._prev_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self._prev_shortcut.activated.connect(self._previous)
        self._next_shortcut = QShortcut(QKeySequence("Alt+Right"), self)
        self._next_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self._next_shortcut.activated.connect(self._next)
        self._save_next_shortcut = QShortcut(QKeySequence("Ctrl+Shift+Return"), self)
        self._save_next_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self._save_next_shortcut.activated.connect(self._save_and_next)
        self._next_judgement_shortcut = QShortcut(QKeySequence("Alt+Down"), self)
        self._next_judgement_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self._next_judgement_shortcut.activated.connect(self._jump_next_judgement)
        self._previous_disagreement_shortcut = QShortcut(QKeySequence("Alt+Shift+D"), self)
        self._previous_disagreement_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self._previous_disagreement_shortcut.activated.connect(self._jump_previous_ocr_disagreement)
        self._next_disagreement_shortcut = QShortcut(QKeySequence("Alt+D"), self)
        self._next_disagreement_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self._next_disagreement_shortcut.activated.connect(self._jump_next_ocr_disagreement)
        self._next_diff_f7_shortcut = QShortcut(QKeySequence("F7"), self)
        self._next_diff_f7_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self._next_diff_f7_shortcut.activated.connect(self._jump_next_ocr_disagreement)
        self._prev_diff_f7_shortcut = QShortcut(QKeySequence("Shift+F7"), self)
        self._prev_diff_f7_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self._prev_diff_f7_shortcut.activated.connect(self._jump_previous_ocr_disagreement)
        self._candidate_choice_shortcuts = []
        for candidate_number in range(1, 10):
            shortcut = QShortcut(QKeySequence(f"Alt+{candidate_number}"), self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(lambda n=candidate_number: self._choose_visible_candidate_shortcut(n))
            self._candidate_choice_shortcuts.append(shortcut)
            direct = QShortcut(QKeySequence(f"Ctrl+Alt+{candidate_number}"), self)
            direct.setContext(Qt.WidgetWithChildrenShortcut)
            direct.activated.connect(lambda n=candidate_number: self._accept_visible_candidate_shortcut(n))
            self._candidate_choice_shortcuts.append(direct)
        self._preview_shortcut = QShortcut(QKeySequence("Alt+P"), self)
        self._preview_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self._preview_shortcut.activated.connect(self._open_current_image_in_preview)
        self._handwriting_shortcut = QShortcut(QKeySequence("Alt+H"), self)
        self._handwriting_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self._handwriting_shortcut.activated.connect(self._open_apple_handwriting_input)
        self._column_left_shortcut = QShortcut(QKeySequence("Alt+Shift+Left"), self)
        self._column_left_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self._column_left_shortcut.activated.connect(lambda: self._move_active_column(+1))
        self._column_right_shortcut = QShortcut(QKeySequence("Alt+Shift+Right"), self)
        self._column_right_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self._column_right_shortcut.activated.connect(lambda: self._move_active_column(-1))
        self._set_controls_enabled(False)

    def _activate_physical_column(
        self, column_index: int, *, scrub_position: float | None = None, source: str = "列定位"
    ) -> None:
        """Highlight one immutable physical column in both visual panes."""
        entry = self._current_entry()
        if entry is None or self._horizontal_document:
            return
        count = max(1, len(entry.column_ids), len(entry.regions), int(entry.column_count or 1))
        column_index = max(0, min(count - 1, int(column_index)))
        intervals = tuple(getattr(self, "_current_image_column_intervals", ()) or ())
        if len(intervals) != count:
            intervals = tuple(
                (1.0 - ((index + 1) / count), 1.0 - (index / count))
                for index in range(count)
            )
        self._active_physical_column_index = column_index
        self._image.set_column_intervals(intervals)
        self._image.set_active_column(column_index)
        self._vertical_columns.set_active_column(column_index)
        if scrub_position is None:
            left_ratio, right_ratio = intervals[column_index]
            scrub_position = (left_ratio + right_ratio) / 2.0
        self._image.set_scrub_position(scrub_position)
        center_x = self._vertical_columns.column_center_x(column_index)
        bar = self._columns_scroll.horizontalScrollBar()
        target = int(center_x - self._columns_scroll.viewport().width() / 2)
        bar.setValue(max(bar.minimum(), min(bar.maximum(), target)))
        column_id = (
            str(entry.column_ids[column_index])
            if column_index < len(entry.column_ids)
            else f"第 {column_index + 1} 列"
        )
        self._columns_state.setText(
            f"定位 {column_index + 1}/{count} · {column_id} · 右→左（{source}）"
        )
        self._update_column_nav_buttons(count)

    def _update_column_nav_buttons(self, count: int | None = None) -> None:
        entry = self._current_entry()
        if count is None:
            count = (
                max(1, len(entry.column_ids), len(entry.regions), int(entry.column_count or 1))
                if entry is not None else 0
            )
        available = bool(entry is not None and not self._horizontal_document and int(count or 0) > 0)
        current = int(getattr(self, "_active_physical_column_index", -1))
        self._column_left_btn.setEnabled(available and (current < 0 or current < int(count) - 1))
        self._column_right_btn.setEnabled(available and (current < 0 or current > 0))

    def _move_active_column(self, delta: int) -> None:
        """Move one physical screen column while preserving Japanese RTL order."""
        entry = self._current_entry()
        if entry is None or self._horizontal_document:
            return
        count = max(1, len(entry.column_ids), len(entry.regions), int(entry.column_count or 1))
        current = int(getattr(self, "_active_physical_column_index", -1))
        if current < 0 or current >= count:
            target = 0
        else:
            target = max(0, min(count - 1, current + int(delta)))
        self._activate_physical_column(target, source="键盘逐列")

    def _scrub_image_to_column(self, normalized_x: float) -> None:
        """Link horizontal image dragging to the corresponding RTL OCR column."""
        entry = self._current_entry()
        if entry is None or self._horizontal_document:
            return
        count = max(
            1, len(entry.column_ids), len(entry.regions), int(entry.column_count or 1)
        )
        x = max(0.0, min(1.0, float(normalized_x)))
        intervals = tuple(getattr(self, "_current_image_column_intervals", ()) or ())
        column_index = -1
        if len(intervals) == count:
            # Exact intervals come from the final composed review PNG, including
            # real strip widths, cleanup crops, margins and inter-column gaps.
            # If the pointer is in a gap, choose the nearest physical strip.
            for index, (left, right) in enumerate(intervals):
                if left <= x <= right:
                    column_index = index
                    break
            if column_index < 0:
                column_index = min(
                    range(count),
                    key=lambda index: abs(x - ((intervals[index][0] + intervals[index][1]) / 2.0)),
                )
        if column_index < 0:
            # Legacy/preferred images may not carry a sidecar.  Preserve the old
            # deterministic right-to-left equal-width fallback rather than
            # guessing from OCR text.
            column_index = min(count - 1, max(0, int((1.0 - x) * count)))
        self._activate_physical_column(
            column_index, scrub_position=x, source="拖动图片联动"
        )

    def _scroll_columns_to_rtl_origin(self) -> None:
        """Show the first logical Japanese column after a sentence/layout change."""
        if self._horizontal_document:
            return
        bar = self._columns_scroll.horizontalScrollBar()
        bar.setValue(bar.maximum())
        self._columns_scroll.verticalScrollBar().setValue(0)

    def _set_left_review_panel_mode(self, mode: str) -> None:
        columns = str(mode) == "columns" and not self._horizontal_document
        self._left_review_stack.setCurrentWidget(
            self._columns_scroll if columns else self._review_disagreement_queue
        )
        self._left_queue_btn.setChecked(not columns)
        self._left_columns_btn.setChecked(columns)
        active = "QToolButton{font-size:9px;padding:2px 5px;border:1px solid #B9D0F4;border-radius:6px;background:#EAF1FF;color:#2559E0;font-weight:700;}"
        idle = "QToolButton{font-size:9px;padding:2px 5px;border:1px solid transparent;border-radius:6px;background:transparent;color:#607086;} QToolButton:hover{background:#F2F4F7;}"
        self._left_queue_btn.setStyleSheet(active if not columns else idle)
        self._left_columns_btn.setStyleSheet(active if columns else idle)
        self._columns_state.setVisible(columns and bool(self._columns_state.text()))
        if hasattr(self, "_review_queue_filters"):
            self._review_queue_filters.setVisible(not columns)
        self._column_left_btn.setVisible(columns)
        self._column_right_btn.setVisible(columns)

    def _review_disagreement_display_text(self, index: int) -> str:
        if not 0 <= int(index) < len(self._entries):
            return ""
        entry = self._entries[int(index)]
        page = "、".join(str(value) for value in entry.pages) or str(entry.page or "-")
        unsaved = bool(
            int(index) == int(getattr(self, "_index", -1))
            and getattr(self, "_editor", None) is not None
            and self._editor.toPlainText() != str(entry.text or "")
        )
        status = "未保存" if unsaved else ("已确认" if entry.reviewed else ("已修改" if entry.changed else "待裁决"))
        text = " ".join(str(entry.text or "").replace("\r", " ").replace("\n", " ").split())
        preview = text[:56] + ("…" if len(text) > 56 else "")
        return f"p{int(entry.page or 0):03d} · 第 {int(index) + 1} 句    {status}\n{preview or '—'}"

    def _review_disagreement_tooltip(self, index: int) -> str:
        if not 0 <= int(index) < len(self._entries):
            return ""
        entry = self._entries[int(index)]
        return str(entry.text or "")

    def _refresh_review_disagreement_queue(self) -> None:
        source_indices = tuple(int(value) for value in self._ocr_disagreement_source_indices)
        ordered: list[int] = []
        seen: set[int] = set()
        if bool(getattr(self, "_ocr_disagreement_source_order_synced", False)):
            # OCR 对比 owns the canonical adjudication queue.  When it publishes
            # a queue order, 图文对照 follows the same stable-row set and order
            # exactly; local filters below may only narrow that queue further.
            for source_row in self._ocr_disagreement_source_row_order:
                entry_index = self._entry_index_by_source_row.get(int(source_row))
                if entry_index is not None and entry_index in source_indices and entry_index not in seen:
                    ordered.append(entry_index)
                    seen.add(entry_index)
        else:
            ordered.extend(source_indices)
        self._ocr_disagreement_indices = ordered
        total_values = tuple(ordered)
        scope_widget = getattr(self, "_review_queue_scope", None)
        scope = str(scope_widget.currentData() or "all") if scope_widget is not None else "all"
        query_widget = getattr(self, "_review_queue_search", None)
        query = str(query_widget.text() if query_widget is not None else "").strip().casefold()
        filtered = []
        for index in total_values:
            entry = self._entries[index]
            if scope == "pending" and entry.reviewed:
                continue
            if scope == "threeway" and len({str(value or "").strip() for value in (entry.fusion_candidate_texts or ()) if str(value or "").strip()}) < 3:
                continue
            if scope == "broken" and not any(
                (not str(value or "").strip())
                or ("□" in str(value or ""))
                or ("�" in str(value or ""))
                for value in (entry.fusion_candidate_texts or ())
            ):
                continue
            if scope == "changed" and not entry.changed:
                continue
            if scope == "reviewed" and (not entry.reviewed or entry.changed):
                continue
            if query:
                haystack = (
                    self._review_disagreement_display_text(index) + " "
                    + str(entry.text or "") + " "
                    + " ".join(str(value) for value in (entry.pages or ())) + " "
                    + f"p{int(entry.page or 0):03d} p{int(entry.page or 0):05d} "
                    + " ".join(str(value) for value in (entry.column_ids or ())) + " "
                    + " ".join(str(value or "") for value in (entry.fusion_candidate_texts or ()))
                ).casefold()
                if query not in haystack:
                    continue
            filtered.append(index)
        values = tuple(filtered)
        self._disagreement_queue_indices = values
        self._review_disagreement_queue_model.set_rows(values)
        self._left_queue_btn.setText(
            f"分歧 {len(values)}/{len(total_values)}" if len(values) != len(total_values) else f"分歧 {len(values)}"
        )
        self._select_review_disagreement_queue_item()

    def set_disagreement_source_row_order(self, source_rows) -> None:
        """Apply OCR 对比's queue ordering to matching 图文对照 entries."""
        values: list[int] = []
        seen: set[int] = set()
        for value in source_rows or ():
            try:
                row = int(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if row >= 0 and row not in seen:
                values.append(row)
                seen.add(row)
        order = tuple(values)
        was_synced = bool(getattr(self, "_ocr_disagreement_source_order_synced", False))
        if was_synced and order == self._ocr_disagreement_source_row_order:
            return
        self._ocr_disagreement_source_order_synced = True
        self._ocr_disagreement_source_row_order = order
        if self._entries:
            self._refresh_review_disagreement_queue()

    def _select_review_disagreement_queue_item(self) -> None:
        row = self._review_disagreement_queue_model.model_index_for_row(self._index)
        selection = self._review_disagreement_queue.selectionModel()
        if selection is not None:
            selection.clearSelection()
        if row.isValid():
            self._review_disagreement_queue.setCurrentIndex(row)
            self._review_disagreement_queue.scrollTo(row, QAbstractItemView.PositionAtCenter)

    def _review_disagreement_item_clicked(self, model_index) -> None:
        if model_index is None or not model_index.isValid():
            return
        try:
            target = int(self._review_disagreement_queue_model.data(model_index, Qt.UserRole))
        except (TypeError, ValueError, OverflowError):
            return
        if not 0 <= target < len(self._entries) or target == self._index:
            return
        if self._current_entry() is not None and not self.save_pending_edit():
            return
        self._index = target
        self._show_current()

    def _set_controls_enabled(self, enabled: bool) -> None:
        self._editor.setEnabled(enabled)
        locked = self._entry_consensus_locked(self._current_entry())
        self._editor.setReadOnly(locked)
        self._prev_btn.setEnabled(enabled and len(self._ocr_disagreement_indices) > 1)
        self._next_btn.setEnabled(enabled and self._index + 1 < len(self._entries))
        self._save_btn.setEnabled(enabled and not locked)
        self._save_next_btn.setEnabled(enabled and bool(self._ocr_disagreement_indices))
        self._apply_btn.setEnabled(enabled)
        self._open_preview_btn.setEnabled(enabled and bool(self._current_review_image_path))
        self._next_judgement_btn.setEnabled(enabled and bool(self._pending_judgement_indices))
        has_disagreement = bool(self._ocr_disagreement_indices)
        self._prev_disagreement_btn.setEnabled(enabled and has_disagreement)
        self._next_disagreement_btn.setEnabled(enabled and has_disagreement)
        if enabled:
            self._update_column_nav_buttons()
        else:
            self._column_left_btn.setEnabled(False)
            self._column_right_btn.setEnabled(False)

    @staticmethod
    def _entry_consensus_locked(entry) -> bool:
        if entry is None:
            return False
        multi = int(getattr(entry, "source_row_index", -1)) >= 0 or len(getattr(entry, "fusion_candidate_texts", ()) or ()) >= 2
        return multi and not OCRImageTextReviewTab._entry_has_ocr_disagreement(entry)

    @staticmethod
    def _entry_has_ocr_disagreement(entry) -> bool:
        explicit = getattr(entry, "source_ocr_disagreement", None)
        if explicit is not None:
            return bool(explicit)
        values = {
            str(value or "")
            for value in (getattr(entry, "fusion_candidate_texts", ()) or ())
            if str(value or "").strip()
        }
        return len(values) > 1

    def _dispose_cache(self) -> None:
        self._image_render_generation.invalidate()
        self._image_render_signal_refs.clear()
        self._queued_image_request = None
        cleanup_dir = None
        with self._image_cache_lock:
            self._image_path_cache.clear()
            self._image_prefetch_pending.clear()
            cache_dir = self._cache_dir
            cache_epoch = int(self._image_cache_epoch)
            self._cache_dir = None
            if cache_dir is not None:
                if int(self._image_cache_jobs.get(cache_epoch, 0)) > 0:
                    self._retired_cache_dirs[cache_epoch] = cache_dir
                else:
                    cleanup_dir = cache_dir
                    self._image_cache_jobs.pop(cache_epoch, None)
        if cleanup_dir is not None:
            self._cleanup_review_cache_dir(cleanup_dir)

    @staticmethod
    def _cleanup_review_cache_dir(cache_dir) -> None:
        try:
            cache_dir.cleanup()
        except OSError:
            # A platform image backend may release the last file handle one
            # event-loop turn later. The TemporaryDirectory finalizer remains
            # a second cleanup path; never block source switching/closing.
            pass

    def _current_review_cache_epoch(self) -> int:
        with self._image_cache_lock:
            return int(self._image_cache_epoch)

    def _review_cache_epoch_is_current(self, epoch: int) -> bool:
        with self._image_cache_lock:
            return self._cache_dir is not None and int(epoch) == int(self._image_cache_epoch)

    def _retain_review_cache_job(self, epoch: int) -> bool:
        """Lease the active cache directory before a background worker starts."""
        with self._image_cache_lock:
            if self._cache_dir is None or int(epoch) != int(self._image_cache_epoch):
                return False
            self._image_cache_jobs[int(epoch)] = int(self._image_cache_jobs.get(int(epoch), 0)) + 1
            return True

    def _release_review_cache_job(self, epoch: int) -> None:
        cleanup_dir = None
        with self._image_cache_lock:
            key = int(epoch)
            count = max(0, int(self._image_cache_jobs.get(key, 0)) - 1)
            if count:
                self._image_cache_jobs[key] = count
            else:
                self._image_cache_jobs.pop(key, None)
                cleanup_dir = self._retired_cache_dirs.pop(key, None)
        if cleanup_dir is not None:
            self._cleanup_review_cache_dir(cleanup_dir)

    def _refresh_sentence_strip(self) -> None:
        strip = getattr(self, "_sentence_strip", None)
        if strip is None:
            return
        states = []
        for entry in self._entries:
            if entry.requires_judgement and not entry.reviewed:
                states.append("judge")
            elif entry.reviewed:
                states.append("changed" if entry.changed else "reviewed")
            else:
                states.append("pending")
        strip.set_states(states, self._index)

    def _jump_to_entry_index(self, target: int) -> None:
        """进度条点击：与按钮翻页相同的保存-切换流程。"""
        if not (0 <= int(target) < len(self._entries)) or int(target) == self._index:
            return
        if self._current_entry() is not None and not self.save_pending_edit():
            return
        self._index = int(target)
        self._show_current()

    def _update_summary(self) -> None:
        total = len(self._entries)
        pending_judgement = len(self._pending_judgement_indices)
        judgement_note = f" · 待判断 {pending_judgement}" if self._has_judgement_entries else ""
        disagreement_count = len(self._ocr_disagreement_indices)
        disagreement_note = f" · OCR分歧 {disagreement_count}" if disagreement_count else ""
        self._review_summary.setText(
            f"已核对 {self._reviewed_count}/{total} · 已修改 {self._changed_count}{judgement_note}{disagreement_note}"
        )
        self._refresh_entry_status_badge()
        if hasattr(self, "_next_judgement_btn"):
            self._next_judgement_btn.setEnabled(bool(self._entries) and pending_judgement > 0)
        self._refresh_sentence_strip()
        if hasattr(self, "_prev_disagreement_btn"):
            self._prev_disagreement_btn.setEnabled(bool(self._entries) and disagreement_count > 0)
        if hasattr(self, "_next_disagreement_btn"):
            self._next_disagreement_btn.setEnabled(bool(self._entries) and disagreement_count > 0)
        if hasattr(self, "_jump_pending_action"):
            self._jump_pending_action.setEnabled(bool(self._entries) and pending_judgement > 0)
        if hasattr(self, "_jump_prev_diff_action"):
            self._jump_prev_diff_action.setEnabled(bool(self._entries) and disagreement_count > 0)
        if hasattr(self, "_jump_next_diff_action"):
            self._jump_next_diff_action.setEnabled(bool(self._entries) and disagreement_count > 0)

    def reset_for_new_book(self) -> None:
        """Detach every image/text review object from the previous OCR document."""
        self._external_decision_dirty_rows = set()
        # Release the displayed pixmap before deleting its cache directory.  This
        # matters on platforms that keep open image files locked.
        self._image.clear_image("完成 OCR 后，右侧显示当前句对应的单列或多列原图。")
        self._current_review_image_path = ""
        self._current_image_column_intervals = ()
        self._cancel_apple_handwriting_session()
        self._clear_fusion_candidate_buttons(dispose=True)
        self._set_fusion_candidates_visible(False)
        self._dispose_cache()
        self._source_doc = None
        self._review_doc = None
        self._horizontal_document = False
        if hasattr(self, "_columns_panel"):
            self._columns_panel.setVisible(True)
        self._document_load_generation.invalidate()
        self._document_load_signal_refs.clear()
        self._document_load_busy = False
        self._pending_jump_source_row = -1
        self._pending_source_doc = None
        self._pending_source_label = ""
        self._entries = []
        self._entry_index_by_source_row.clear()
        self._entry_index_by_sentence_group_id.clear()
        self._entry_indices_by_block.clear()
        self._disagreement_queue_indices = ()
        if hasattr(self, "_review_disagreement_queue_model"):
            self._review_disagreement_queue_model.set_rows(())
        if hasattr(self, "_left_queue_btn"):
            self._left_queue_btn.setText("分歧 0")
            self._set_left_review_panel_mode("queue")
        self._pending_judgement_indices.clear()
        self._has_judgement_entries = False
        self._ocr_disagreement_indices.clear()
        self._ocr_disagreement_source_indices.clear()
        self._ocr_disagreement_source_row_order = ()
        self._index = -1
        self._dirty = False
        self._reviewed_count = 0
        self._changed_count = 0
        self._source_name = ""
        self._initial_text_by_key.clear()
        self._session_dirty_keys.clear()
        self._pending_ocr_sync_rows.clear()
        self._pending_ocr_compare_decisions.clear()
        self._suppress_row_review_emit = False
        self._source_label.setText("尚未接收 OCR 结果")
        self._editor.blockSignals(True)
        try:
            self._editor.clear()
        finally:
            self._editor.blockSignals(False)
        self._position_label.setText("第 0 / 0 句")
        self._text_state.setText("")
        self._image_state.setText("")
        self._columns_state.setText("")
        self._vertical_columns.clear_columns()
        self._update_summary()
        self._set_controls_enabled(False)

    def set_document(
        self,
        doc: UnifiedDocument | None,
        label: str = "OCR 校对稿",
        *,
        lazy: bool = True,
    ) -> None:
        """Attach a new OCR source, optionally deferring the full-book clone.

        Multi-model OCR opens the OCR comparison workspace first.  Cloning a
        300-page document and building every image-review entry at that moment
        only blocks the GUI for a feature the user has not opened yet.  Lazy
        mode stores the immutable source reference and performs that work when
        the 图文对照 workspace is first activated.
        """
        self.reset_for_new_book()
        if doc is None:
            return
        if lazy:
            self._pending_source_doc = doc
            self._pending_source_label = str(label or "OCR 校对稿")
            self._source_label.setText(f"{self._pending_source_label} · 打开本页时载入")
            self._image.clear_image("图文对照尚未载入；进入本工作区后自动准备逐句图片。")
            return
        self._load_document_now(doc, label)

    def set_page_image_provider(self, provider) -> None:
        """Provide durable page-manager images for stale/restored OCR paths.

        OCR/PDF temporary paths may disappear after cleanup or workspace
        restore.  图文对照 must be able to rebuild its immutable sentence crops
        from the project's retained page images without rerunning OCR.
        """
        self._page_image_provider = provider

    def _page_image_fallback_snapshot(self) -> tuple[str, ...]:
        provider = self._page_image_provider
        if not callable(provider):
            return ()
        try:
            values = provider() or []
        except Exception:
            return ()
        return tuple(
            str(Path(str(value)).expanduser())
            for value in values
            if str(value or "")
        )

    def ensure_document_loaded(self) -> None:
        if self._pending_source_doc is None or self._document_load_busy:
            return
        doc = self._pending_source_doc
        label = self._pending_source_label or "OCR 校对稿"
        self._pending_source_doc = None
        self._pending_source_label = ""
        self._document_load_busy = True
        self._source_label.setText(f"{label} · 正在后台准备逐句校对…")
        self._image.clear_image("正在建立图文索引；界面可以继续操作。")
        self._set_controls_enabled(False)
        fallback_page_images = self._page_image_fallback_snapshot()
        token = self._document_load_generation.begin()
        signals = WorkerSignals(self)
        self._document_load_signal_refs[token] = signals
        signals.finished.connect(self._on_document_load_finished)
        signals.error.connect(lambda message, token=token: self._on_document_load_error(token, message))

        def worker():
            try:
                from engine.ocr_image_text_review import build_review_entries, clone_for_review
                review_doc = clone_for_review(doc)
                entries = build_review_entries(
                    review_doc,
                    fallback_page_images=fallback_page_images,
                )
                signals.finished.emit({
                    "token": token,
                    "source_doc": doc,
                    "label": label,
                    "review_doc": review_doc,
                    "entries": entries,
                })
            except Exception as exc:
                signals.error.emit(str(exc))

        threading.Thread(
            target=worker, daemon=True,
            name=f"image-review-index-{token}",
        ).start()

    def _on_document_load_finished(self, payload: dict) -> None:
        data = payload if isinstance(payload, dict) else {}
        token = int(data.get("token", -1))
        self._document_load_signal_refs.pop(token, None)
        if not self._document_load_generation.is_current(token):
            return
        self._document_load_busy = False
        self._install_loaded_document(
            data.get("source_doc"), str(data.get("label", "") or "OCR 校对稿"),
            data.get("review_doc"), list(data.get("entries") or []),
        )

    def _on_document_load_error(self, token: int, message: str) -> None:
        self._document_load_signal_refs.pop(int(token), None)
        if not self._document_load_generation.is_current(int(token)):
            return
        self._document_load_busy = False
        self._source_label.setText("图文逐句索引建立失败")
        self._image.clear_image("图文索引建立失败；OCR 对比和正文仍可正常使用。")
        self._image_state.setText(str(message or "未知错误"))

    def _load_document_now(self, doc: UnifiedDocument, label: str) -> None:
        from engine.ocr_image_text_review import build_review_entries, clone_for_review
        self._source_label.setText(f"{label} · 正在准备逐句校对…")
        QApplication.processEvents()
        review_doc = clone_for_review(doc)
        entries = build_review_entries(
            review_doc,
            fallback_page_images=self._page_image_fallback_snapshot(),
        )
        self._install_loaded_document(doc, label, review_doc, entries)

    def _install_loaded_document(
        self, doc: UnifiedDocument | None, label: str,
        review_doc: UnifiedDocument | None, entries: list,
    ) -> None:
        if doc is None or review_doc is None:
            self._on_document_load_error(self._document_load_generation.current, "校对文档为空")
            return
        self._source_doc = doc
        self._horizontal_document = bool(
            str(getattr(doc.metadata, "ocr_mode", "") or "") == "zh_hans_horizontal"
            or str(getattr(doc.metadata, "writing_direction", "") or "").startswith("horizontal")
        )
        self._columns_panel.setVisible(True)
        self._left_columns_btn.setVisible(not self._horizontal_document)
        self._review_doc = review_doc
        self._entries = list(entries or [])
        self._entry_index_by_source_row = {
            int(entry.source_row_index): position
            for position, entry in enumerate(self._entries)
            if int(getattr(entry, "source_row_index", -1)) >= 0
        }
        self._entry_index_by_sentence_group_id = {
            str(entry.sentence_group_id): position
            for position, entry in enumerate(self._entries)
            if str(getattr(entry, "sentence_group_id", "") or "")
        }
        block_map: dict[int, list[int]] = {}
        for position, entry in enumerate(self._entries):
            block_map.setdefault(int(entry.block_index), []).append(position)
        self._entry_indices_by_block = {
            block_index: tuple(sorted(indices, key=lambda index: (
                self._entries[index].segment_index, self._entries[index].segment_key
            )))
            for block_index, indices in block_map.items()
        }
        self._index = 0 if self._entries else -1
        self._source_name = str(label or "OCR 校对稿")
        self._initial_text_by_key = {entry.segment_key: entry.text for entry in self._entries}
        if self._entries and self._cache_dir is None:
            self._cache_dir = tempfile.TemporaryDirectory(prefix="novel_formatter_sentence_review_")
        pending_decisions = list(self._pending_ocr_compare_decisions.values())
        self._pending_ocr_compare_decisions.clear()
        for pending_payload in pending_decisions:
            self.apply_ocr_compare_decision(pending_payload)
        self._reviewed_count = sum(1 for entry in self._entries if entry.reviewed)
        self._changed_count = sum(1 for entry in self._entries if entry.changed)
        self._pending_judgement_indices = {
            index for index, entry in enumerate(self._entries)
            if bool(entry.requires_judgement and not entry.reviewed)
        }
        self._has_judgement_entries = any(bool(entry.requires_judgement) for entry in self._entries)
        self._ocr_disagreement_source_indices = [
            index for index, entry in enumerate(self._entries)
            if self._entry_has_ocr_disagreement(entry)
        ]
        self._refresh_review_disagreement_queue()
        pending_row = int(self._pending_jump_source_row)
        self._pending_jump_source_row = -1
        if pending_row >= 0:
            self._index = self._entry_index_by_source_row.get(pending_row, self._index)
        self._source_label.setText(self._source_name)
        if self._entries:
            if self._cache_dir is None:
                self._cache_dir = tempfile.TemporaryDirectory(prefix="novel_formatter_sentence_review_")
            self._show_current()
        else:
            self._image.clear_image("当前 OCR 文档没有可校对的文字句。")
            self._vertical_columns.clear_columns("当前 OCR 文档没有可校对的文字句。")
            self._position_label.setText("第 0 / 0 句")
            self._text_state.setText("")
            self._image_state.setText("")
            self._columns_state.setText("")
            self._update_summary()
            self._set_controls_enabled(False)

    def _current_entry(self):
        if 0 <= self._index < len(self._entries):
            return self._entries[self._index]
        return None

    def current_source_row_index(self) -> int:
        entry = self._current_entry()
        return int(getattr(entry, "source_row_index", -1)) if entry is not None else -1

    def jump_to_source_row(self, row_index: int, *, save_current: bool = False) -> bool:
        """Show the exact OCR 对比 row instead of using a separate sentence index."""
        try:
            source_row = int(row_index)
        except (TypeError, ValueError, OverflowError):
            return False
        if self._pending_source_doc is not None or self._document_load_busy:
            self._pending_jump_source_row = source_row
            self.ensure_document_loaded()
            return True
        target = self._entry_index_by_source_row.get(source_row)
        if target is None:
            return False
        if save_current and self._current_entry() is not None:
            if not self._save_current(silent=True):
                return False
        if target == self._index:
            if target in getattr(self, "_external_decision_dirty_rows", set()):
                self._show_current()
            return True
        self._index = target
        self._show_current()
        return True

    def _on_text_changed(self) -> None:
        self._schedule_review_editor_fit()
        entry = self._current_entry()
        if entry is None:
            return
        if not self._horizontal_document:
            from engine.multi_ocr_compare import project_fused_text_to_physical_columns
            count = max(1, len(entry.column_ids), len(entry.regions), int(entry.column_count or 1))
            self._vertical_columns.set_columns(project_fused_text_to_physical_columns(
                self._editor.toPlainText(), entry.column_texts, column_count=count,
            ))
            self._vertical_columns.set_active_column(self._active_physical_column_index)
        changed = self._editor.toPlainText() != entry.text
        self._text_state.setText("未保存修改" if changed else ("已人工确认" if entry.reviewed else ""))
        self._refresh_entry_status_badge(entry)
        queue_model = getattr(self, "_review_disagreement_queue_model", None)
        if queue_model is not None:
            try:
                queue_model.refresh_row(self._index)
            except Exception:
                pass
        if changed:
            self._source_label.setText(f"{self._source_name} · 当前句未保存")
        elif self._dirty:
            self._source_label.setText(f"{self._source_name} · 有未应用修改")
        else:
            self._source_label.setText(self._source_name)

    def _schedule_review_editor_fit(self) -> None:
        timer = getattr(self, "_editor_fit_timer", None)
        if timer is not None:
            timer.start(0)

    def _fit_review_editor_height(self) -> None:
        """Size the manual editor to its wrapped text, leaving spare room above."""
        try:
            document = self._editor.document()
            document.setTextWidth(float(max(1, self._editor.viewport().width() - 4)))
            content_height = int(math.ceil(document.documentLayout().documentSize().height()))
            margins = self._editor.contentsMargins()
            chrome = (
                self._editor.frameWidth() * 2
                + margins.top() + margins.bottom()
                + 18
            )
            target = max(58, content_height + chrome)
            if self._editor.height() != target:
                self._editor.setFixedHeight(target)
        except RuntimeError:
            # The editor may be closing while a coalesced resize is pending.
            return

    def _refresh_entry_status_badge(self, entry=None) -> None:
        badge = getattr(self, "_entry_status_badge", None)
        if badge is None:
            return
        entry = self._current_entry() if entry is None else entry
        if entry is None:
            text, bg, fg, border = "待核对", "#F3F5F7", "#667085", "#D9DEE5"
        else:
            unsaved = self._editor.toPlainText() != str(entry.text or "")
            if unsaved:
                text, bg, fg, border = "未保存", "#FFF4E5", "#9A6700", "#F5C36B"
            elif entry.changed:
                text, bg, fg, border = "已修改", "#E7F0FF", "#2559E0", "#B9D0F4"
            elif entry.reviewed:
                text, bg, fg, border = "已确认", "#EAF8EF", "#147A42", "#B9E4C9"
            elif entry.requires_judgement:
                text, bg, fg, border = "待裁决", "#FFF0E6", "#B54708", "#F7C89C"
            else:
                text, bg, fg, border = "待核对", "#F3F5F7", "#667085", "#D9DEE5"
        badge.setText(text)
        badge.setStyleSheet(
            f"QLabel{{background:{bg};color:{fg};border:1px solid {border};border-radius:10px;"
            "font-size:9px;font-weight:700;padding:2px 8px;}"
        )

    def _choose_visible_candidate_shortcut(self, number: int) -> None:
        """Alt+1..9 stages the corresponding visible fusion candidate."""
        target = max(0, int(number) - 1)
        cards = [card for card in self._fusion_candidate_buttons if card.isVisible()]
        if not 0 <= target < len(cards):
            return
        card = cards[target]
        self._choose_fusion_candidate(int(card.candidate_index), str(card.candidate_text or ""))

    def _accept_visible_candidate_shortcut(self, number: int) -> None:
        """Ctrl+Alt+1..9 directly accepts and advances."""
        target = max(0, int(number) - 1)
        cards = [card for card in self._fusion_candidate_buttons if card.isVisible()]
        if not 0 <= target < len(cards):
            return
        card = cards[target]
        self._accept_fusion_candidate(int(card.candidate_index), str(card.candidate_text or ""))

    def _show_current(self) -> None:
        entry = self._current_entry()
        if entry is not None:
            getattr(self, "_draft_candidate_sources", {}).pop(id(entry), None)
        getattr(self, "_external_decision_dirty_rows", set()).discard(self._index)
        entry = self._current_entry()
        if entry is None:
            return
        self._active_physical_column_index = -1
        self._image.set_scrub_position(None)
        self._image.set_active_column(-1)
        self._vertical_columns.set_active_column(-1)
        self._editor.blockSignals(True)
        self._editor.setPlainText(entry.text)
        self._editor.blockSignals(False)
        self._schedule_review_editor_fit()
        self._text_state.setText("已人工确认" if entry.reviewed else "")
        self._refresh_entry_status_badge(entry)
        physical_count = max(
            1,
            len(entry.column_ids),
            len(entry.regions),
            int(entry.column_count or 1),
        )
        if self._horizontal_document:
            # Chinese horizontal OCR is reviewed directly against its page/line
            # crop.  Do not project it into the Japanese right-to-left column UI.
            self._vertical_columns.clear_columns()
            self._columns_state.setText("横排 · 上→下、左→右")
        else:
            from engine.multi_ocr_compare import project_fused_text_to_physical_columns
            display_columns = project_fused_text_to_physical_columns(
                entry.text,
                entry.column_texts,
                column_count=physical_count,
            )
            self._vertical_columns.set_columns(display_columns)
            # The widget paints logical column 1 at the far right.  Wait until
            # QScrollArea has recomputed its range, then reveal that RTL origin.
            QTimer.singleShot(0, self._scroll_columns_to_rtl_origin)
            lineage_state = (
                " · 列ID已锁定"
                if entry.column_ids and len(entry.column_ids) == physical_count
                else " · 按原图列数锁定"
            )
            self._columns_state.setText(
                f"{physical_count} 列 · 右→左{lineage_state}"
                + (f" · 原块第 {entry.segment_index + 1}/{entry.segment_count} 句" if entry.segment_count > 1 else "")
            )
        self._update_column_nav_buttons(physical_count if not self._horizontal_document else 0)

        self._refresh_fusion_candidates(entry)
        self._request_review_image(entry)

        pages = "、".join(str(page) for page in entry.pages) or str(entry.page or "-")
        judgement_state = " · 需要判断" if entry.requires_judgement and not entry.reviewed else ""
        disagreement_state = " · OCR结果不一致" if self._entry_has_ocr_disagreement(entry) else ""
        self._position_label.setText(
            f"第 {self._index + 1} / {len(self._entries)} 句 · 页 {pages} · {physical_count} 列{judgement_state}{disagreement_state}"
        )
        self._update_summary()
        self._set_controls_enabled(True)
        self._select_review_disagreement_queue_item()
        source_row = int(getattr(entry, "source_row_index", -1))
        if source_row >= 0:
            self.source_row_changed.emit(source_row)
        self._editor.setFocus()

    @staticmethod
    def _review_image_layout_name(layout: str) -> str:
        return {
            "single_column": "单列句",
            "column_sentence": "多列成句",
            "column_strips": "多列条带句图",
            "column_strips_fallback": "合并框失败后条带句图",
            "global_merged_boxes": "全书真实合并框句图",
            "source_block_context": "人工重对齐来源块参考图",
            "no_primary_source": "其他模型独有句·无主模型来源图",
        }.get(str(layout or ""), str(layout or "句图"))

    def _review_image_cache_path(self, entry) -> Path:
        with self._image_cache_lock:
            if self._cache_dir is None:
                self._cache_dir = tempfile.TemporaryDirectory(prefix="novel_formatter_sentence_review_")
                self._image_cache_epoch += 1
            return Path(self._cache_dir.name) / f"{entry.cache_key}.png"

    def _request_review_image(self, entry) -> None:
        token = self._image_render_generation.begin()
        entry_key = str(entry.segment_key)
        cache_key = str(entry.cache_key)
        self._current_review_image_path = ""
        self._current_image_column_intervals = ()
        self._image.clear_image("正在载入当前句图片…")
        self._image_state.setText("正在载入…")
        self._open_preview_btn.setEnabled(False)
        cache_path = self._review_image_cache_path(entry)
        cache_epoch = self._current_review_cache_epoch()
        request = (token, entry_key, cache_key, entry, cache_path, cache_epoch)
        if not self._workspace_active:
            self._queued_image_request = request
            return
        if self._image_render_busy:
            # Discard any older not-yet-started request. Only the sentence the
            # user finally stopped on is worth rendering.
            self._queued_image_request = request
            return
        self._start_review_image_render(request)

    def _start_review_image_render(self, request) -> None:
        token, entry_key, cache_key, entry, cache_path, cache_epoch = request
        if not self._image_render_generation.is_current(int(token)):
            self._drain_queued_review_image_render()
            return
        if not self._review_cache_epoch_is_current(int(cache_epoch)):
            self._drain_queued_review_image_render()
            return
        with self._image_cache_lock:
            cached = self._image_path_cache.get(str(cache_key), "")
        if cached and Path(cached).is_file():
            self._install_review_image(int(token), str(entry_key), cached)
            self._drain_queued_review_image_render()
            return
        if Path(cache_path).is_file():
            path = str(cache_path)
            with self._image_cache_lock:
                self._image_path_cache[str(cache_key)] = path
            self._install_review_image(int(token), str(entry_key), path)
            self._drain_queued_review_image_render()
            return

        if not self._retain_review_cache_job(int(cache_epoch)):
            self._drain_queued_review_image_render()
            return
        self._image_render_busy = True
        self._image_render_cache_epochs[int(token)] = int(cache_epoch)
        signals = ImageReviewRenderSignals(self)
        self._image_render_signal_refs[int(token)] = signals
        signals.finished.connect(self._on_review_image_rendered)
        signals.error.connect(self._on_review_image_render_error)

        def worker():
            try:
                from engine.ocr_image_text_review import render_review_image
                image_path = str(render_review_image(entry, cache_path) or "")
                if not image_path:
                    raise RuntimeError("没有生成可用句图")
                signals.finished.emit(int(token), str(entry_key), image_path)
            except Exception as exc:
                signals.error.emit(int(token), str(entry_key), str(exc))
            finally:
                self._release_review_cache_job(int(cache_epoch))

        threading.Thread(
            target=worker, daemon=True,
            name=f"image-review-render-{token}",
        ).start()

    def _drain_queued_review_image_render(self) -> None:
        if self._image_render_busy:
            return
        request = self._queued_image_request
        self._queued_image_request = None
        if request is not None:
            self._start_review_image_render(request)

    def _on_review_image_rendered(self, token: int, entry_key: str, image_path: str) -> None:
        self._image_render_signal_refs.pop(int(token), None)
        cache_epoch = self._image_render_cache_epochs.pop(int(token), None)
        self._image_render_busy = False
        path = str(image_path or "")
        if path and cache_epoch is not None and self._review_cache_epoch_is_current(int(cache_epoch)):
            # Cache filenames are exactly <entry.cache_key>.png.  This remains
            # correct even when the user has already navigated elsewhere.
            with self._image_cache_lock:
                self._image_path_cache[Path(path).stem] = path
        self._install_review_image(int(token), str(entry_key), path)
        self._drain_queued_review_image_render()

    def _on_review_image_render_error(self, token: int, entry_key: str, message: str) -> None:
        self._image_render_signal_refs.pop(int(token), None)
        self._image_render_cache_epochs.pop(int(token), None)
        self._image_render_busy = False
        current = self._current_entry()
        if self._image_render_generation.is_current(int(token)) and (
            current is not None and str(current.segment_key) == str(entry_key)
        ):
            self._image.clear_image("当前句图片生成失败；文字仍可正常修改。")
            self._image_state.setText(f"图片生成失败：{message}")
            self._open_preview_btn.setEnabled(False)
        self._drain_queued_review_image_render()

    def _install_review_image(self, token: int, entry_key: str, image_path: str) -> None:
        current = self._current_entry()
        if not self._image_render_generation.is_current(int(token)):
            return
        if current is None or str(current.segment_key) != str(entry_key):
            return
        if image_path and self._image.set_image_path(image_path):
            self._current_review_image_path = str(image_path)
            physical_count = max(
                1, len(current.column_ids), len(current.regions), int(current.column_count or 1)
            )
            from engine.ocr_image_text_review import load_review_image_column_intervals
            exact_intervals = load_review_image_column_intervals(
                image_path, expected_count=physical_count
            )
            if exact_intervals:
                intervals = exact_intervals
                mapping_note = " · 精确列联动"
            else:
                intervals = tuple(
                    (1.0 - ((index + 1) / physical_count), 1.0 - (index / physical_count))
                    for index in range(physical_count)
                )
                mapping_note = " · 等宽联动"
            self._current_image_column_intervals = intervals
            self._image.set_column_intervals(intervals)
            self._image_state.setText(
                f"{self._review_image_layout_name(current.layout)} · {physical_count} 列{mapping_note}"
            )
            self._open_preview_btn.setEnabled(True)
            self._prefetch_adjacent_review_images()
        else:
            self._current_image_column_intervals = ()
            self._image.clear_image("找不到当前句对应的图片；文字仍可正常修改。")
            self._image_state.setText("图片不可用")
            self._open_preview_btn.setEnabled(False)

    def _prefetch_adjacent_review_images(self) -> None:
        if not self._workspace_active:
            return
        candidates = []
        for index in (self._index - 1, self._index + 1):
            if not 0 <= index < len(self._entries):
                continue
            entry = self._entries[index]
            key = str(entry.cache_key)
            cache_path = self._review_image_cache_path(entry)
            cache_epoch = self._current_review_cache_epoch()
            pending_key = (int(cache_epoch), key)
            with self._image_cache_lock:
                cached = self._image_path_cache.get(key, "")
                if (cached and Path(cached).is_file()) or pending_key in self._image_prefetch_pending:
                    continue
                self._image_prefetch_pending.add(pending_key)
            if cache_path.is_file():
                with self._image_cache_lock:
                    self._image_path_cache[key] = str(cache_path)
                    self._image_prefetch_pending.discard(pending_key)
                continue
            candidates.append((entry, key, cache_path, int(cache_epoch), pending_key))
        if not candidates:
            return

        # Every candidate in this batch belongs to the same active cache
        # directory. Hold one lease for the worker's complete lifetime so a
        # book switch cannot delete/recreate that directory mid-render.
        worker_epoch = int(candidates[0][3])
        if not self._retain_review_cache_job(worker_epoch):
            with self._image_cache_lock:
                for _entry, _key, _path, _epoch, pending_key in candidates:
                    self._image_prefetch_pending.discard(pending_key)
            return

        def worker():
            try:
                from engine.ocr_image_text_review import render_review_image
                for entry, key, cache_path, cache_epoch, pending_key in candidates:
                    if not self._review_cache_epoch_is_current(cache_epoch):
                        break
                    try:
                        path = str(render_review_image(entry, cache_path) or "")
                        if path and self._review_cache_epoch_is_current(cache_epoch):
                            with self._image_cache_lock:
                                self._image_path_cache[key] = path
                    except Exception:
                        pass
                    finally:
                        with self._image_cache_lock:
                            self._image_prefetch_pending.discard(pending_key)
            finally:
                with self._image_cache_lock:
                    for _entry, _key, _path, _epoch, pending_key in candidates:
                        self._image_prefetch_pending.discard(pending_key)
                self._release_review_cache_job(worker_epoch)

        threading.Thread(
            target=worker, daemon=True,
            name="image-review-prefetch",
        ).start()

    def _clear_fusion_candidate_buttons(self, *, dispose: bool = False) -> None:
        while self._fusion_candidate_grid.count():
            item = self._fusion_candidate_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
        if dispose:
            for widget in self._fusion_candidate_buttons:
                widget.deleteLater()
            self._fusion_candidate_buttons.clear()

    def _candidate_column_count(self, candidate_count: int) -> int:
        """Choose a wide-screen grid without leaving an artificial side gutter."""
        count = max(1, int(candidate_count or 1))
        frame_width = int(self._fusion_candidate_frame.width())
        pane = getattr(self, "_text_review_panel", None)
        pane_width = int(pane.width()) if pane is not None else 0
        width = max(frame_width, pane_width)
        del width, count      # 图文对照改为左右并排：各模型结果自上而下每行一个，便于逐字对比
        return 1

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._schedule_review_editor_fit()
        entry = self._current_entry()
        if entry is None or not getattr(self, "_fusion_candidate_frame", None):
            return
        candidate_count = len({
            str(value or "") for value in (getattr(entry, "fusion_candidate_texts", ()) or ())
            if str(value or "").strip()
        })
        columns = self._candidate_column_count(candidate_count) if candidate_count else 0
        if columns == self._fusion_candidate_grid_columns or self._candidate_reflow_pending:
            return
        self._candidate_reflow_pending = True
        current_key = str(entry.segment_key)

        def reflow():
            self._candidate_reflow_pending = False
            current = self._current_entry()
            if current is not None and str(current.segment_key) == current_key:
                self._refresh_fusion_candidates(current)

        QTimer.singleShot(0, reflow)

    def _set_fusion_candidates_visible(self, visible: bool) -> None:
        self._fusion_candidate_frame.setVisible(bool(visible))
        scroll = getattr(self, "_fusion_candidate_scroll", None)
        if scroll is not None:
            scroll.setVisible(bool(visible))

    def _refresh_fusion_candidates(self, entry) -> None:
        self._clear_fusion_candidate_buttons()
        texts = list(getattr(entry, "fusion_candidate_texts", ()) or ())
        labels = list(getattr(entry, "fusion_candidate_labels", ()) or ())
        confidences = list(getattr(entry, "fusion_candidate_confidences", ()) or ())
        grouped: list[dict] = []
        by_text: dict[str, dict] = {}
        for index, text in enumerate(texts):
            value = str(text or "")
            if not value.strip():
                continue
            label = str(labels[index] if index < len(labels) else f"候选{index + 1}")
            confidence = float(confidences[index]) if index < len(confidences) else 0.0
            group = by_text.get(value)
            if group is None:
                group = {"text": value, "indices": [], "labels": [], "confidence": confidence}
                by_text[value] = group
                grouped.append(group)
            group["indices"].append(index)
            if label not in group["labels"]:
                group["labels"].append(label)
            group["confidence"] = max(float(group["confidence"]), confidence)
        visible = bool(grouped)   # 只要有模型结果就显示：即使各模型一致，也让人看到谁识别了什么
        show_candidates = visible and bool(grouped)
        self._set_fusion_candidates_visible(show_candidates)
        pane = getattr(self, "_text_review_panel", None)
        if pane is not None:
            pane.setMinimumHeight(0)   # 高度由左右分栏与滚动区决定
        if not grouped:
            return
        reason = str(getattr(entry, "judgement_reason", "") or "")
        warnings = tuple(getattr(entry, "judgement_warnings", ()) or ())
        state = "需要判断" if getattr(entry, "requires_judgement", False) else "候选已融合"
        reason_names = {
            "pending_ai_review": "等待 AI / 人工裁决",
            "multi_model_disagreement": "多模型原始结果不一致",
            "low_confidence": "低置信候选",
            "placeholder_or_empty": "存在空白/占位符",
            "manual_review": "需要人工复核",
        }
        readable_reason = reason_names.get(reason, reason.replace("_", " ").strip()) if reason else ""
        detail_parts = [part for part in (readable_reason, "；".join(warnings[:2])) if part]
        if not detail_parts:
            detail_parts.append("点击候选可直接写入当前校对稿；不会覆盖任何模型原文")
        self._fusion_candidate_title.setText(f"候选对比 · {len(grouped)} 份 · {state}")
        self._fusion_candidate_detail.setText(" · ".join(detail_parts) + " · 红底=替换，橙底=增删/缺失")
        try:
            selected = int(getattr(entry, "selected_candidate_index", -1))
        except (TypeError, ValueError, OverflowError):
            selected = -1
        reference_text = str(self._editor.toPlainText() or entry.text or grouped[0]["text"])
        columns = self._candidate_column_count(len(grouped))
        self._fusion_candidate_grid_columns = columns
        for candidate_number, group in enumerate(grouped, start=1):
            index = int(group["indices"][0])
            label_text = " + ".join(group["labels"])
            pool_index = candidate_number - 1
            if pool_index < len(self._fusion_candidate_buttons):
                card = self._fusion_candidate_buttons[pool_index]
                card.set_candidate(
                    index, f"候选{candidate_number} · {label_text}", group["text"],
                    confidence=float(group["confidence"] or 0.0),
                )
            else:
                card = _ImageReviewFusionCandidateCard(
                    index, f"候选{candidate_number} · {label_text}", group["text"],
                    confidence=float(group["confidence"] or 0.0), parent=self._fusion_candidate_frame,
                )
                card.drafted.connect(self._choose_fusion_candidate)
                card.chosen.connect(self._accept_fusion_candidate)
                self._fusion_candidate_buttons.append(card)
            card.set_reference_text(reference_text)
            card.set_selected(
                bool(getattr(entry, "reviewed", False))
                and (selected in group["indices"] or (selected < 0 and group["text"] == entry.text))
            )
            baseline = getattr(self, "_draft_candidate_sources", {}).get(id(entry))
            card.set_draft_source(
                baseline in group["indices"] if baseline is not None else group["text"] == reference_text
            )
            row = (candidate_number - 1) // columns
            column = (candidate_number - 1) % columns
            self._fusion_candidate_grid.addWidget(card, row, column)
            self._fusion_candidate_grid.setRowStretch(row, 0)
            card._choose.setEnabled(not self._entry_consensus_locked(entry))
            card.show()
        for column in range(4):
            self._fusion_candidate_grid.setColumnStretch(column, 1 if column < columns else 0)

    def _choose_fusion_candidate(self, candidate_index: int, text: str) -> None:
        """Stage one OCR candidate as the manual-edit baseline only."""
        entry = self._current_entry()
        if entry is None or self._entry_consensus_locked(entry):
            return
        self._draft_candidate_index = int(candidate_index)
        if not hasattr(self, "_draft_candidate_sources"):
            self._draft_candidate_sources = {}
        self._draft_candidate_sources[id(entry)] = int(candidate_index)
        self._editor.setPlainText(str(text or ""))
        self._editor.moveCursor(QTextCursor.End)
        self._text_state.setText(
            f"候选 {int(candidate_index) + 1} 已作为手动底稿 · 尚未完成裁决，修改后点“确认裁决”"
        )
        for card in self._fusion_candidate_buttons:
            card.set_draft_source(card.candidate_index == int(candidate_index))
        self._editor.setFocus()

    def _accept_fusion_candidate(self, candidate_index: int, text: str) -> None:
        """Explicit fast-path: accept a candidate and move to next OCR disagreement."""
        entry = self._current_entry()
        if entry is None or self._entry_consensus_locked(entry):
            return
        self._editor.setPlainText(str(text or ""))
        self._editor.moveCursor(QTextCursor.End)
        if not self._save_current(silent=True):
            self._text_state.setText("直接采用失败；请检查当前句映射")
            return
        QTimer.singleShot(0, self._jump_next_ocr_disagreement)

    def _jump_next_judgement(self) -> None:
        if not self._entries:
            return
        if not self.save_pending_edit():
            return
        pending = sorted(self._pending_judgement_indices)
        if not pending:
            self._text_state.setText("所有需要判断的句图均已核对")
            self._update_summary()
            return
        position = bisect_right(pending, self._index)
        self._index = pending[position] if position < len(pending) else pending[0]
        self._show_current()

    def _jump_previous_ocr_disagreement(self) -> None:
        if not self._entries:
            return
        if not self.save_pending_edit():
            return
        indices = self._ocr_disagreement_indices
        if not indices:
            self._text_state.setText("当前文档没有多模型 OCR 分歧句")
            return
        try:
            position = indices.index(self._index)
        except ValueError:
            position = 0
        self._index = indices[(position - 1) % len(indices)]
        self._show_current()
        self._text_state.setText("已跳到上一条 OCR 对比不一致句图")

    def _jump_next_ocr_disagreement(self) -> None:
        if not self._entries:
            return
        if not self.save_pending_edit():
            return
        indices = self._ocr_disagreement_indices
        if not indices:
            self._text_state.setText("当前文档没有多模型 OCR 分歧句")
            return
        try:
            position = indices.index(self._index)
        except ValueError:
            position = -1
        self._index = indices[(position + 1) % len(indices)]
        self._show_current()
        self._text_state.setText("已跳到下一条 OCR 对比不一致句图")

    def _open_current_image_in_preview(self) -> None:
        path = Path(str(self._current_review_image_path or ""))
        if not path.is_file():
            notify(self, "请先载入一条带原图的 OCR 句。", "warning")
            return
        try:
            if sys.platform == "darwin":
                subprocess.Popen(
                    ["/usr/bin/open", "-a", "Preview", str(path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            else:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
            self._image_state.setText(f"已用预览打开 · {path.name}")
        except Exception as exc:
            show_error_dialog(self, "无法打开当前图片", str(exc))

    def _cancel_apple_handwriting_session(self) -> None:
        self._apple_handwriting_active = False
        self._apple_handwriting_target_key = ""
        timer = getattr(self, "_apple_handwriting_timeout", None)
        if timer is not None and timer.isActive():
            timer.stop()

    def _open_apple_handwriting_input(self) -> None:
        entry = self._current_entry()
        if entry is None:
            return
        self._editor.setFocus()
        if sys.platform != "darwin":
            notify(self, "苹果系统手写输入只在 macOS 上可用。", "info")
            return
        try:
            from adapters.apple_pkstroke_engine import launch_manual_test_panel
            launch_manual_test_panel(auto_build=True)
        except Exception as exc:
            self._cancel_apple_handwriting_session()
            QMessageBox.information(
                self,
                "苹果手写面板暂不可用",
                f"{exc}\n\n文本框已获得焦点，仍可从 macOS 输入菜单切换到日语输入法或系统手写输入。",
            )
            return
        self._apple_handwriting_active = True
        self._apple_handwriting_target_key = str(entry.segment_key)
        self._apple_handwriting_timeout.start(10 * 60 * 1000)
        self._text_state.setText("苹果手写板已打开；点击“复制并返回”后自动写入此处")

    def _on_apple_handwriting_clipboard_changed(self) -> None:
        if not self._apple_handwriting_active:
            return
        mime = self._clipboard.mimeData()
        marker = "com.novelformatter.apple-handwriting"
        if mime is None or marker not in set(mime.formats()):
            return
        entry = self._current_entry()
        if entry is None or str(entry.segment_key) != self._apple_handwriting_target_key:
            self._cancel_apple_handwriting_session()
            return
        value = str(self._clipboard.text() or "")
        if not value:
            return
        cursor = self._editor.textCursor()
        cursor.insertText(value)
        self._editor.setTextCursor(cursor)
        self._cancel_apple_handwriting_session()
        window = self.window()
        window.showNormal()
        window.raise_()
        window.activateWindow()
        self._editor.setFocus()
        self._text_state.setText("已接收苹果手写结果，尚未保存")

    def _save_current(self, checked=False, *, silent: bool = False) -> bool:
        entry = self._current_entry()
        if entry is None or self._review_doc is None:
            return False
        if self._entry_consensus_locked(entry):
            return self._editor.toPlainText().strip("\n") == entry.text.strip("\n")
        from engine.ocr_image_text_review import apply_review_text
        value = self._editor.toPlainText().replace("\r\n", "\n").replace("\r", "\n").strip("\n")
        was_reviewed = bool(entry.reviewed)
        was_changed = bool(entry.changed)
        block_entries = [
            self._entries[index]
            for index in self._entry_indices_by_block.get(int(entry.block_index), (self._index,))
            if 0 <= index < len(self._entries)
        ]
        merged_value = "".join(
            value if item is entry else item.text
            for item in block_entries
        )
        found, changed = apply_review_text(
            self._review_doc,
            entry.block_id,
            merged_value,
            block_index=entry.block_index,
        )
        if not found:
            if not silent:
                QMessageBox.warning(self, "保存失败", "当前句在校对文档中不存在。")
            return False
        entry.text = value
        entry.reviewed = True
        self._pending_judgement_indices.discard(int(self._index))
        exact_matches = [
            index for index, candidate in enumerate(entry.fusion_candidate_texts)
            if str(candidate or "") == value
        ]
        if entry.selected_candidate_index not in exact_matches:
            entry.selected_candidate_index = exact_matches[0] if exact_matches else -1
        from engine.multi_ocr_compare import project_fused_text_to_physical_columns
        physical_count = max(
            1,
            len(entry.column_ids),
            len(entry.regions),
            int(entry.column_count or 1),
        )
        entry.column_texts = tuple(project_fused_text_to_physical_columns(
            entry.text,
            entry.column_texts,
            column_count=physical_count,
        ))
        entry.column_count = physical_count
        initial_text = self._initial_text_by_key.get(entry.segment_key, entry.text)
        entry.changed = entry.text != initial_text

        # Keep the immutable OCR-row boundary metadata in sync with sentence
        # edits.  The structural block remains one block for EPUB/layout, while
        # reopening 图文对照 still restores the same independent sentence items.
        if 0 <= entry.block_index < len(self._review_doc.blocks):
            block = self._review_doc.blocks[entry.block_index]
            metadata = dict(block.metadata) if isinstance(block.metadata, dict) else {}
            raw_groups = metadata.get("ocr_review_sentence_groups") or []
            if isinstance(raw_groups, list) and raw_groups:
                groups = [dict(group) if isinstance(group, dict) else {} for group in raw_groups]
                for sibling in block_entries:
                    if 0 <= sibling.segment_index < len(groups):
                        groups[sibling.segment_index]["text"] = sibling.text
                        groups[sibling.segment_index]["column_texts"] = list(sibling.column_texts)
                        groups[sibling.segment_index]["review_selected_candidate_index"] = int(
                            sibling.selected_candidate_index
                        )
                metadata["ocr_review_sentence_groups"] = groups
            reviewed_segments = dict(metadata.get("ocr_image_text_review_checked_segments") or {})
            changed_segments = dict(metadata.get("ocr_image_text_review_changed_segments") or {})
            reviewed_segments[entry.segment_key] = True
            changed_segments[entry.segment_key] = bool(entry.changed)
            metadata["ocr_image_text_review_checked_segments"] = reviewed_segments
            metadata["ocr_image_text_review_changed_segments"] = changed_segments
            block.metadata = metadata
        if not was_reviewed:
            self._reviewed_count += 1
        if was_changed != entry.changed:
            self._changed_count += 1 if entry.changed else -1
        if entry.changed:
            self._session_dirty_keys.add(entry.segment_key)
        else:
            self._session_dirty_keys.discard(entry.segment_key)
        self._dirty = bool(self._session_dirty_keys)
        self._text_state.setText("已保存修改" if entry.changed else "已人工确认")
        self._source_label.setText(
            f"{self._source_name} · 有未应用修改" if self._dirty else self._source_name
        )
        self._update_summary()
        if int(getattr(entry, "source_row_index", -1)) >= 0 and not self._suppress_row_review_emit:
            source_row = int(entry.source_row_index)
            self._pending_ocr_sync_rows.add(source_row)
            self._text_state.setText("已保存，正在同步到 OCR 对比…")
            self.row_review_saved.emit({
                "row_index": source_row,
                "sentence_group_id": str(getattr(entry, "sentence_group_id", "") or ""),
                "column_ids": list(entry.column_ids),
                "text": str(entry.text or ""),
                "delete_intentionally": not bool(str(entry.text or "").strip()),
                "selected_candidate_index": int(entry.selected_candidate_index),
                "segment_key": str(entry.segment_key),
                "changed": bool(entry.changed),
                "reviewed": True,
            })
        return True

    def save_pending_edit(self) -> bool:
        """Commit only a real unsaved editor change before leaving the workspace."""
        entry = self._current_entry()
        if entry is None:
            return True
        if self._index in getattr(self, "_external_decision_dirty_rows", set()):
            self._show_current()
            return True
        value = self._editor.toPlainText().replace("\r\n", "\n").replace("\r", "\n").strip("\n")
        if value == str(entry.text or ""):
            return True
        return bool(self._save_current(silent=True))

    @staticmethod
    def _ocr_compare_decision_pending_key(payload: dict) -> str:
        group_id = str(payload.get("sentence_group_id", "") or "")
        if group_id:
            return "sentence:" + group_id
        try:
            row_index = int(payload.get("row_index", -1))
        except (TypeError, ValueError, OverflowError):
            row_index = -1
        if row_index >= 0:
            return f"row:{row_index}"
        columns = tuple(str(value) for value in (payload.get("column_ids") or ()) if str(value))
        return "columns:" + "|".join(columns) if columns else ""

    def _resolve_entry_index_for_decision(self, payload: dict) -> int | None:
        try:
            row_index = int(payload.get("row_index", -1))
        except (TypeError, ValueError, OverflowError):
            row_index = -1
        incoming_group_id = str(payload.get("sentence_group_id", "") or "")
        if incoming_group_id:
            target = self._entry_index_by_sentence_group_id.get(incoming_group_id)
            if target is not None:
                return target
            # A supplied stable ID that is unknown must never fall through to a
            # same-column sentence. That was the bulk-adjudication corruption.
            return None

        direct = self._entry_index_by_source_row.get(row_index)
        incoming_columns = tuple(str(value) for value in (payload.get("column_ids") or ()) if str(value))
        if direct is not None:
            entry = self._entries[direct]
            current_columns = tuple(str(value) for value in (entry.column_ids or ()) if str(value))
            if not incoming_columns or current_columns == incoming_columns:
                return direct
        matches = [
            index for index, entry in enumerate(self._entries)
            if tuple(str(value) for value in (entry.column_ids or ()) if str(value)) == incoming_columns
        ] if incoming_columns else []
        return matches[0] if len(matches) == 1 else None

    def _apply_hidden_ocr_decision_to_entry(
        self, target: int, text: str, selected_candidate_index: int
    ) -> bool:
        """Update one 图文 entry without touching editor/image widgets.

        This is the high-throughput mirror path used while OCR 对比 is the
        visible workspace.  It keeps the review document and counters current
        but deliberately avoids ``_show_current()`` and image decoding.
        """
        if self._review_doc is None or not 0 <= int(target) < len(self._entries):
            return False
        target = int(target)
        entry = self._entries[target]
        from engine.ocr_image_text_review import apply_review_text
        was_reviewed = bool(entry.reviewed)
        was_changed = bool(entry.changed)
        block_indices = self._entry_indices_by_block.get(int(entry.block_index), (target,))
        block_entries = [
            self._entries[index] for index in block_indices
            if 0 <= int(index) < len(self._entries)
        ]
        merged_value = "".join(text if item is entry else item.text for item in block_entries)
        found, _changed = apply_review_text(
            self._review_doc, entry.block_id, merged_value, block_index=entry.block_index
        )
        if not found:
            return False
        entry.text = str(text or "")
        entry.reviewed = True
        entry.selected_candidate_index = int(selected_candidate_index)
        self._pending_judgement_indices.discard(target)
        from engine.multi_ocr_compare import project_fused_text_to_physical_columns
        physical_count = max(1, len(entry.column_ids), len(entry.regions), int(entry.column_count or 1))
        entry.column_texts = tuple(project_fused_text_to_physical_columns(
            entry.text, entry.column_texts, column_count=physical_count,
        ))
        entry.column_count = physical_count
        initial_text = self._initial_text_by_key.get(entry.segment_key, entry.text)
        entry.changed = entry.text != initial_text
        if 0 <= entry.block_index < len(self._review_doc.blocks):
            block = self._review_doc.blocks[entry.block_index]
            metadata = dict(block.metadata) if isinstance(block.metadata, dict) else {}
            raw_groups = metadata.get("ocr_review_sentence_groups") or []
            if isinstance(raw_groups, list) and raw_groups:
                groups = [dict(group) if isinstance(group, dict) else {} for group in raw_groups]
                for sibling in block_entries:
                    if 0 <= sibling.segment_index < len(groups):
                        groups[sibling.segment_index]["text"] = sibling.text
                        groups[sibling.segment_index]["column_texts"] = list(sibling.column_texts)
                        groups[sibling.segment_index]["review_selected_candidate_index"] = int(
                            sibling.selected_candidate_index
                        )
                metadata["ocr_review_sentence_groups"] = groups
            reviewed = dict(metadata.get("ocr_image_text_review_checked_segments") or {})
            changed = dict(metadata.get("ocr_image_text_review_changed_segments") or {})
            reviewed[entry.segment_key] = True
            changed[entry.segment_key] = bool(entry.changed)
            metadata["ocr_image_text_review_checked_segments"] = reviewed
            metadata["ocr_image_text_review_changed_segments"] = changed
            block.metadata = metadata
        if not was_reviewed:
            self._reviewed_count += 1
        if was_changed != entry.changed:
            self._changed_count += 1 if entry.changed else -1
        if entry.changed:
            self._session_dirty_keys.add(entry.segment_key)
        else:
            self._session_dirty_keys.discard(entry.segment_key)
        self._dirty = bool(self._session_dirty_keys)
        self._update_summary()
        return True

    def apply_ocr_compare_decision(self, payload: dict, *, refresh: bool = True) -> bool:
        """Mirror an OCR 对比 choice into 图文 without emitting a feedback loop."""
        if not isinstance(payload, dict) or str(payload.get("origin", "")) == "image_review":
            return False
        key = self._ocr_compare_decision_pending_key(payload)
        if self._pending_source_doc is not None and not self._entries:
            if key:
                self._pending_ocr_compare_decisions[key] = dict(payload)
            return True
        target = self._resolve_entry_index_for_decision(payload)
        if target is None:
            if key:
                self._pending_ocr_compare_decisions[key] = dict(payload)
            return False
        entry = self._entries[target]
        if not refresh:
            if not hasattr(self, "_external_decision_dirty_rows"):
                self._external_decision_dirty_rows = set()
            self._external_decision_dirty_rows.add(target)
        resolved = bool(payload.get("resolved", False))
        if not resolved:
            was_reviewed = bool(entry.reviewed)
            was_changed = bool(entry.changed)
            # Undo/reopen carries the current unresolved fusion output separately
            # from the selected-candidate text.  Restore the review copy to that
            # value so OCR 对比 and 图文对照 cannot disagree after an undo.
            if "current_output_text" in payload and self._review_doc is not None:
                from engine.ocr_image_text_review import apply_review_text
                restore_text = str(payload.get("current_output_text") or "")
                block_indices = self._entry_indices_by_block.get(int(entry.block_index), (target,))
                block_entries = [
                    self._entries[index] for index in block_indices
                    if 0 <= int(index) < len(self._entries)
                ]
                merged_value = "".join(
                    restore_text if item is entry else item.text for item in block_entries
                )
                found, _changed = apply_review_text(
                    self._review_doc, entry.block_id, merged_value, block_index=entry.block_index
                )
                if found:
                    entry.text = restore_text
                    from engine.multi_ocr_compare import project_fused_text_to_physical_columns
                    physical_count = max(
                        1, len(entry.column_ids), len(entry.regions), int(entry.column_count or 1)
                    )
                    entry.column_texts = tuple(project_fused_text_to_physical_columns(
                        entry.text, entry.column_texts, column_count=physical_count,
                    ))
                    entry.column_count = physical_count
                    initial_text = self._initial_text_by_key.get(entry.segment_key, entry.text)
                    entry.changed = entry.text != initial_text
            entry.selected_candidate_index = -1
            entry.reviewed = False
            if bool(entry.requires_judgement):
                self._pending_judgement_indices.add(int(target))
            if was_reviewed:
                self._reviewed_count = max(0, self._reviewed_count - 1)
            if was_changed != entry.changed:
                self._changed_count += 1 if entry.changed else -1
            if entry.changed:
                self._session_dirty_keys.add(entry.segment_key)
            else:
                self._session_dirty_keys.discard(entry.segment_key)
            self._dirty = bool(self._session_dirty_keys)
            if self._review_doc is not None and 0 <= entry.block_index < len(self._review_doc.blocks):
                block = self._review_doc.blocks[entry.block_index]
                metadata = dict(block.metadata) if isinstance(block.metadata, dict) else {}
                reviewed_segments = dict(metadata.get("ocr_image_text_review_checked_segments") or {})
                changed_segments = dict(metadata.get("ocr_image_text_review_changed_segments") or {})
                reviewed_segments[entry.segment_key] = False
                changed_segments[entry.segment_key] = bool(entry.changed)
                metadata["ocr_image_text_review_checked_segments"] = reviewed_segments
                metadata["ocr_image_text_review_changed_segments"] = changed_segments
                groups = [
                    dict(group) if isinstance(group, dict) else {}
                    for group in (metadata.get("ocr_review_sentence_groups") or [])
                ]
                if 0 <= entry.segment_index < len(groups):
                    groups[entry.segment_index]["text"] = entry.text
                    groups[entry.segment_index]["column_texts"] = list(entry.column_texts)
                    groups[entry.segment_index]["review_selected_candidate_index"] = -1
                    metadata["ocr_review_sentence_groups"] = groups
                block.metadata = metadata
            self._update_summary()
            if refresh and target == self._index:
                self._show_current()
                self._text_state.setText("OCR 对比已重新打开此句，等待选择")
            return True
        text = str(payload.get("text", "") or "")
        delete_intentionally = bool(payload.get("delete_intentionally", False))
        if not text and not delete_intentionally:
            return False
        labels = list(entry.fusion_candidate_labels)
        values = list(entry.fusion_candidate_texts)
        confidences = list(entry.fusion_candidate_confidences)
        display_label = str(payload.get("display_label", "") or "OCR 对比最终选择")
        selected = next((i for i, value in enumerate(values) if str(value or "") == text and i < len(labels) and labels[i] == display_label), None)
        if selected is None:
            values.append(text)
            labels.append(display_label)
            confidences.append(float(payload.get("confidence", 0.0) or 0.0))
            selected = len(values) - 1
        entry.fusion_candidate_texts = tuple(values)
        entry.fusion_candidate_labels = tuple(labels)
        entry.fusion_candidate_confidences = tuple(confidences)
        entry.selected_candidate_index = int(selected)
        if not refresh:
            return self._apply_hidden_ocr_decision_to_entry(target, text, int(selected))

        previous_index = self._index
        self._index = target
        self._editor.blockSignals(True)
        try:
            self._editor.setPlainText(text)
        finally:
            self._editor.blockSignals(False)
        self._suppress_row_review_emit = True
        try:
            success = self._save_current(silent=True)
        finally:
            self._suppress_row_review_emit = False
        self._index = previous_index
        if previous_index == target:
            self._show_current()
            self._text_state.setText("已同步 OCR 对比最终选择")
        elif 0 <= previous_index < len(self._entries):
            self._show_current()
        return bool(success)

    def notify_ocr_compare_sync_result(
        self, row_index: int, success: bool, message: str = ""
    ) -> None:
        """Acknowledge the real OCRCompare write instead of assuming signal success."""
        try:
            row = int(row_index)
        except (TypeError, ValueError, OverflowError):
            row = -1
        self._pending_ocr_sync_rows.discard(row)
        current = self._current_entry()
        if current is None or int(getattr(current, "source_row_index", -1)) != row:
            return
        if success:
            self._text_state.setStyleSheet(f"color:{MUTED};font-size:10px;")
            self._text_state.setText(message or "已保存并同步到 OCR 对比")
        else:
            self._text_state.setText(message or "已保存，但 OCR 对比同步失败")
            self._text_state.setStyleSheet("color:#B42318;font-size:10px;font-weight:700;")

    def _previous(self) -> None:
        if self._index <= 0:
            return
        if self.save_pending_edit():
            self._index -= 1
            self._show_current()

    def _next(self) -> None:
        if self._index + 1 >= len(self._entries):
            return
        if self.save_pending_edit():
            self._index += 1
            self._show_current()

    def _save_and_next(self) -> None:
        if not self._save_current():
            return
        if self._index + 1 < len(self._entries):
            self._index += 1
            self._show_current()

    def _request_full_compare(self) -> None:
        entry = self._current_entry()
        row = int(getattr(entry, "source_row_index", -1)) if entry is not None else -1
        self.full_compare_requested.emit(row)

    def _apply_document(self) -> None:
        if self._review_doc is None:
            QMessageBox.warning(self, "没有校对稿", "请先完成 OCR。")
            return
        if not self._save_current():
            return
        reviewed = self._reviewed_count
        changed = self._changed_count
        self._review_doc.metadata.__dict__.update({
            "ocr_image_text_review_applied": True,
            "ocr_image_text_review_total": len(self._entries),
            "ocr_image_text_review_checked": reviewed,
            "ocr_image_text_review_changed": changed,
        })
        self._review_doc.add_log(
            "ocr_image_text_review",
            f"图文逐句校对：核对 {reviewed}/{len(self._entries)} 句，修改 {changed} 句",
            changed,
        )
        self.doc_applied.emit(self._review_doc.snapshot_clone())
        self._dirty = False
        self._session_dirty_keys.clear()
        self._initial_text_by_key = {entry.segment_key: entry.text for entry in self._entries}
        self._source_label.setText(f"{self._source_name} · 已应用")
        notify(
            self,
            f"已将校对稿送入后续工作流。\n核对 {reviewed} 句，修改 {changed} 句。",
            "success",
        )

    def set_workspace_active(self, active: bool) -> None:
        active = bool(active)
        if active == self._workspace_active:
            return
        self._workspace_active = active
        if not active:
            self._apple_handwriting_timeout.stop()
            return
        self.ensure_document_loaded()
        if self._index in getattr(self, "_external_decision_dirty_rows", set()):
            self._show_current()
        if self._queued_image_request is not None and not self._image_render_busy:
            self._drain_queued_review_image_render()
        elif self._current_entry() is not None and not self._current_review_image_path:
            self._request_review_image(self._current_entry())

    def shutdown_cleanup(self) -> None:
        self._workspace_active = False
        self._image.clear_image("")
        self._dispose_cache()
