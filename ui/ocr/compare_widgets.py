from __future__ import annotations

import math
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QTimer, QEvent, QModelIndex, QAbstractListModel, QSize, QRectF
from PySide6.QtGui import QFont, QColor, QCursor, QTextCursor, QTextCharFormat, QPainter, QPen
from PySide6.QtWidgets import (
    QWidget, QFrame, QHBoxLayout, QVBoxLayout, QLabel, QCheckBox, QPlainTextEdit, QTextEdit,
    QPushButton, QDialog, QFormLayout, QComboBox, QLineEdit, QFileDialog, QMessageBox,
    QStyledItemDelegate, QStyle, QScrollArea, QSizePolicy,
)

from ui.common.editor_controls import MouseWheelPlainTextEdit, NoWheelSpinBox
from ui.common.styling import EDITOR_SCROLLBAR_STYLE, accent_button
from ui.localized_dialogs import LocalizedFileDialog, LocalizedMessageBox

# Preserve the GUI module's localized native-dialog behaviour after extraction.
QFileDialog = LocalizedFileDialog
QMessageBox = LocalizedMessageBox



class DecisionQueueDelegate(QStyledItemDelegate):
    """Reference-style decision queue renderer.

    The queue model remains fully virtual; this delegate only replaces the
    platform-dependent QListView item chrome with the supplied UI language:
    location on the first line, a compact status pill, and the candidate
    summary on the second line.
    """

    def sizeHint(self, option, index):  # noqa: N802 - Qt API
        return QSize(max(280, option.rect.width()), 60)

    @staticmethod
    def _split_display(value: str) -> tuple[str, str, str]:
        lines = str(value or "").splitlines()
        head = lines[0] if lines else ""
        summary = lines[1] if len(lines) > 1 else ""
        # _decision_queue_display_text separates location/status with four
        # spaces so the existing model contract stays untouched.
        if "    " in head:
            location, status = head.split("    ", 1)
        else:
            location, status = head, "待裁决"
        return location.strip(), status.strip() or "待裁决", summary.strip()

    @staticmethod
    def _status_colors(status: str, *, dark: bool) -> tuple[QColor, QColor]:
        if "已" in status or "本地" in status or "人工" in status or "AI" in status:
            return (QColor("#DDEBFF" if not dark else "#1E3A5F"), QColor("#3165C7" if not dark else "#B8D3FF"))
        if "紧急" in status or "高风险" in status:
            return (QColor("#FDE7E7" if not dark else "#5A2626"), QColor("#B42318" if not dark else "#FFC3BD"))
        if "复核" in status or "草稿" in status or "未保存" in status:
            return (QColor("#FFF2D6" if not dark else "#57401F"), QColor("#9A6700" if not dark else "#FFD98A"))
        return (QColor("#EDF2F7" if not dark else "#2B3542"), QColor("#607086" if not dark else "#C7D0DB"))

    def paint(self, painter: QPainter, option, index):  # noqa: N802 - Qt API
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)

        selected = bool(option.state & QStyle.State_Selected)
        force_light = bool(getattr(self, "force_light", False))
        dark = not force_light and option.palette.window().color().lightness() < 128
        rect = QRectF(option.rect).adjusted(1.0, 1.0, -1.0, -1.0)
        if selected:
            fill = QColor("#FFFFFF" if force_light else ("#EAF1FF" if not dark else "#233A63"))
            border = QColor("#4F7CFF")
            border_width = 1.6
        else:
            fill = QColor("#FFFFFF" if not dark else "#17202A")
            border = QColor("#DCE7F5" if not dark else "#394758")
            border_width = 1.0
        painter.setPen(QPen(border, border_width))
        painter.setBrush(fill)
        painter.drawRoundedRect(rect, 9.0, 9.0)

        location, status, summary = self._split_display(index.data(Qt.DisplayRole))
        # The active row is visually in progress; risk priority remains available
        # from the queue context menu/tooltip instead of crowding the card header.
        if selected and not ("已" in status or "人工" in status or "AI" in status):
            status = "处理中"
        left = rect.left() + 12.0
        right = rect.right() - 10.0
        top = rect.top() + 9.0

        title_font = QFont(option.font)
        title_font.setPointSizeF(max(9.0, option.font.pointSizeF()))
        title_font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(title_font)
        title_color = QColor("#14202E" if not dark else "#E7EDF5")
        painter.setPen(title_color)
        painter.drawText(QRectF(left, top, max(80.0, rect.width() * 0.56), 18.0), Qt.AlignLeft | Qt.AlignVCenter, location)

        badge_font = QFont(option.font)
        badge_font.setPointSizeF(max(7.5, option.font.pointSizeF() - 1.0))
        badge_font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(badge_font)
        metrics = painter.fontMetrics()
        badge_w = max(44.0, float(metrics.horizontalAdvance(status) + 18))
        badge_rect = QRectF(right - badge_w, top - 1.0, badge_w, 20.0)
        badge_bg, badge_fg = self._status_colors(status, dark=dark)
        painter.setPen(Qt.NoPen)
        painter.setBrush(badge_bg)
        painter.drawRoundedRect(badge_rect, 9.0, 9.0)
        painter.setPen(badge_fg)
        painter.drawText(badge_rect, Qt.AlignCenter, status)

        summary_font = QFont(option.font)
        summary_font.setPointSizeF(max(7.5, option.font.pointSizeF() - 1.0))
        painter.setFont(summary_font)
        painter.setPen(QColor("#8190A4" if not dark else "#9EACBC"))
        summary_rect = QRectF(left, top + 24.0, max(20.0, right - left), 18.0)
        summary_text = painter.fontMetrics().elidedText(summary, Qt.ElideRight, max(20, int(summary_rect.width())))
        painter.drawText(summary_rect, Qt.AlignLeft | Qt.AlignVCenter, summary_text)
        painter.restore()

_OCR_MODEL_COLORS = [
    ("#2F6BFF", "#E4EEFF", "#BFDBFE"),
    ("#7C3AED", "#F5F3FF", "#DDD6FE"),
    ("#D97706", "#FFFBEB", "#FDE68A"),
]

class _FusionCandidateCard(QFrame):
    selected = Signal(int, int)
    drafted = Signal(int, int)
    focused = Signal(int)
    text_edited = Signal(int)

    def __init__(
        self,
        row_index: int,
        candidate_index: int,
        text: str,
        model_indices: tuple[int, ...],
        labels: list[str],
        parent=None,
        *,
        candidate_state=None,
    ):
        super().__init__(parent)
        self.row_index = row_index
        self.candidate_index = candidate_index
        self.model_indices = tuple(model_indices)
        self._candidate_state = candidate_state
        self._candidate_state_signature = self._snapshot_candidate_state(candidate_state)
        self._synthetic = bool(getattr(candidate_state, "synthetic", False))
        self._candidate_reason = str(getattr(candidate_state, "reason", "") or "")
        self._candidate_confidence = float(getattr(candidate_state, "confidence", 0.0) or 0.0)
        self._transaction_id = str(getattr(candidate_state, "transaction_id", "") or "")
        self._audit_level = str(getattr(candidate_state, "audit_level", "") or "")
        self._audit_flags = tuple(getattr(candidate_state, "audit_flags", ()) or ())
        self._historical_evidence = self._audit_level == "historical_ocr_evidence"
        if self._historical_evidence:
            color, background, border = "#5B6B80", "#F7F8FA", "#D8DDE3"
        elif self._synthetic:
            color, background, border = "#047857", "#ECFDF5", "#A7F3D0"
        else:
            color_index = self.model_indices[0] if self.model_indices else 0
            color, background, border = _OCR_MODEL_COLORS[color_index % len(_OCR_MODEL_COLORS)]
        # Phase 22 proofing surface: candidates share one quiet blue card
        # treatment; model identity lives in the caption badge.  This matches
        # the supplied master more closely than full-card model colours while
        # preserving every candidate/editor/selection signal.
        neutral_bg = "#F7F8FA" if not self._historical_evidence else "#F8FAFC"
        neutral_border = "#D5E4F7"
        self._normal_style = (
            f"QFrame#fusionCandidate{{background:{neutral_bg};border:1px solid {neutral_border};"
            "border-radius:11px;}"
        )
        self._selected_style = (
            f"QFrame#fusionCandidate{{background:#EEF4FF;border:2px solid #4F7CFF;"
            "border-radius:11px;}"
        )
        self.setObjectName("fusionCandidate")
        self.setStyleSheet(self._normal_style)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(7)

        model_names = [labels[index] if index < len(labels) else f"模型{index + 1}" for index in self.model_indices]
        display_label = str(getattr(candidate_state, "display_label", "") or "")
        self._base_caption = display_label or ("＋".join(model_names) if model_names else "候选")
        if self._synthetic and self._candidate_confidence:
            self._base_caption += f" · 置信 {self._candidate_confidence:.1%}"
        self._caption = QLabel(self._base_caption)
        self._caption.setStyleSheet(
            f"color:{color};font-size:11px;font-weight:700;border:none;"
            "background:transparent;padding:0px;"
        )
        self._caption.setMinimumWidth(112)
        self._caption.setWordWrap(False)
        self._caption.setFixedHeight(24)
        self._caption.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._caption.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(self._caption)

        self.editor = MouseWheelPlainTextEdit()
        self.editor.setPlainText(text)
        # Model/AI candidates are evidence, not the manual editing surface.
        # Keep them immutable and copy the chosen baseline into the dedicated
        # manual-decision editor instead.  This guarantees a "use as draft"
        # action never rewrites OCR evidence before the user confirms.
        self.editor.setReadOnly(True)
        self.editor.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.editor.setMinimumHeight(54)
        self.editor.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # Candidate text is always fully expanded.  The outer fusion workspace
        # owns scrolling, so long sentences never disappear inside a nested
        # editor scrollbar.
        self.editor.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.editor.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.editor.setStyleSheet(
            "QPlainTextEdit{background:#FFFFFF;color:#14202E;border:1px solid #D7E3F4;"
            "border-radius:8px;padding:8px 10px;font-size:15px;}" + EDITOR_SCROLLBAR_STYLE
        )
        tooltip = (
            "待判断候选差异：红底=替换/错字；橙底=仅本候选存在；"
            "橙色波浪线=另一候选在此位置有额外文字。"
        )
        if self._synthetic:
            tooltip += "\n字符级融合只在当前物理列 ID 内逐字符投票，不会跨列拼接。"
        if self._transaction_id:
            tooltip += f"\n原子事务：{self._transaction_id}；选择或取消其中一项会同步整个事务。"
        if self._historical_evidence:
            tooltip += "\n这是纠错前的只读 OCR 分歧证据，不能重新选为正文。"
        elif self._audit_level:
            tooltip += f"\n导入审计：{self._audit_level}"
        if self._audit_flags:
            tooltip += "\n风险标记：" + "、".join(str(value) for value in self._audit_flags)
        if self._candidate_reason:
            tooltip += f"\n{self._candidate_reason}"
        self.editor.setToolTip(tooltip)
        self._last_editor_layout_width = -1
        self.editor.viewport().installEventFilter(self)
        self.editor.textChanged.connect(self._fit_editor_height)
        QTimer.singleShot(0, self._fit_editor_height)
        layout.addWidget(self.editor, 1)

        self._draft = QPushButton("作为底稿")
        self._draft.setCursor(QCursor(Qt.PointingHandCursor))
        self._draft.setStyleSheet(
            "QPushButton{color:#2559E0;font-weight:700;border:1px solid #BFD3F7;"
            "background:#F7FAFF;border-radius:9px;padding:5px 10px;}"
            "QPushButton:hover{background:#EAF1FF;}"
        )
        self._draft.setToolTip(
            f"Alt+{candidate_index + 1}：复制到手动编辑框；不会完成裁决，也不会跳转"
            if candidate_index < 9 else "复制到手动编辑框；不会完成裁决，也不会跳转"
        )
        self._draft.clicked.connect(
            lambda: self.drafted.emit(self.row_index, self.candidate_index)
        )

        self.check = QCheckBox("直接采用")
        self.check.setCursor(QCursor(Qt.PointingHandCursor))
        self.check.setStyleSheet(
            "QCheckBox{color:#2F6BFF;font-weight:700;border:1px solid #E2E5E9;"
            "background:#F4F5F7;border-radius:9px;padding:5px 10px;}"
            "QCheckBox:checked{color:#FFFFFF;background:#2F6BFF;border-color:#2F6BFF;}"
            "QCheckBox::indicator{width:0;height:0;border:none;background:transparent;}"
        )
        self.check.setToolTip(
            f"Ctrl+Alt+{candidate_index + 1}：直接采用并完成当前裁决"
            if candidate_index < 9 else "直接采用并完成当前裁决"
        )
        self.check.clicked.connect(lambda checked: self.selected.emit(self.row_index, self.candidate_index) if checked else None)
        self._draft.setVisible(not self._historical_evidence)
        self._draft.setEnabled(not self._historical_evidence)
        self.check.setVisible(not self._historical_evidence)
        self.check.setEnabled(not self._historical_evidence)
        footer = QHBoxLayout()
        footer.addStretch(1)
        footer.addWidget(self._draft, 0, Qt.AlignRight)
        footer.addWidget(self.check, 0, Qt.AlignRight)
        layout.addLayout(footer)

    @staticmethod
    def _snapshot_candidate_state(candidate) -> tuple:
        return (
            str(getattr(candidate, "text", "") or ""),
            tuple(getattr(candidate, "model_indices", ()) or ()),
            str(getattr(candidate, "display_label", "") or ""),
            bool(getattr(candidate, "synthetic", False)),
            bool(getattr(candidate, "delete_intentionally", False)),
            str(getattr(candidate, "transaction_id", "") or ""),
            str(getattr(candidate, "audit_level", "") or ""),
            tuple(str(value) for value in (getattr(candidate, "audit_flags", ()) or ())),
        )

    def _sync_candidate_text(self):
        if self._candidate_state is not None:
            self._candidate_state.text = self.editor.toPlainText()
            self._candidate_state.delete_intentionally = not bool(self._candidate_state.text.strip())
            self._candidate_state_signature = self._snapshot_candidate_state(self._candidate_state)
        self.text_edited.emit(self.candidate_index)

    @staticmethod
    def _qt_position(text: str, python_index: int) -> int:
        """Convert a Python Unicode offset to Qt's UTF-16 cursor offset."""
        index = max(0, min(len(text), int(python_index)))
        return len(text[:index].encode("utf-16-le")) // 2

    def clear_diff_marks(self):
        self.editor.setExtraSelections([])
        self._caption.setText(self._base_caption)

    def set_diff_marks(self, marks, *, change_count: int = 0):
        """Draw non-destructive character differences over the editable text."""
        text = self.editor.toPlainText()
        selections = []
        for start, end, kind in marks or ():
            start = max(0, min(len(text), int(start)))
            end = max(start, min(len(text), int(end)))
            cursor = QTextCursor(self.editor.document())
            fmt = QTextCharFormat()
            if end > start:
                cursor.setPosition(self._qt_position(text, start))
                cursor.setPosition(self._qt_position(text, end), QTextCursor.KeepAnchor)
                if kind == "replace":
                    fmt.setBackground(QColor("#FFD1D1"))
                    fmt.setForeground(QColor("#8A1C1C"))
                else:
                    fmt.setBackground(QColor("#FFE4AD"))
                    fmt.setForeground(QColor("#7A3E00"))
                fmt.setFontWeight(QFont.Weight.DemiBold)
            elif text:
                # A zero-width difference means the other candidate contains
                # extra text here.  Underline the nearest real character so the
                # missing boundary is visible without changing the OCR string.
                anchor = start if start < len(text) else max(0, start - 1)
                anchor_end = min(len(text), anchor + 1)
                cursor.setPosition(self._qt_position(text, anchor))
                cursor.setPosition(self._qt_position(text, anchor_end), QTextCursor.KeepAnchor)
                fmt.setUnderlineStyle(QTextCharFormat.UnderlineStyle.WaveUnderline)
                fmt.setUnderlineColor(QColor("#D97706"))
                fmt.setFontUnderline(True)
            else:
                continue
            selection = QTextEdit.ExtraSelection()
            selection.cursor = cursor
            selection.format = fmt
            selections.append(selection)
        self.editor.setExtraSelections(selections)
        suffix = f" · 差异 {int(change_count)} 处" if change_count else ""
        self._caption.setText(self._base_caption + suffix)

    def _fit_editor_height(self):
        """Expand the editor to the complete wrapped document, without a height cap."""
        try:
            document = self.editor.document()
            usable_width = max(1, self.editor.viewport().width())
            # Reflow against the actual visible width before reading the layout
            # height.  This follows Qt's wrapping rather than estimating by
            # character count, so punctuation, Latin text and manual line breaks
            # are all fully represented.
            document.setTextWidth(float(usable_width))
            # QPlainTextDocumentLayout.documentSize().height() counts lines,
            # rather than pixels. Measure each laid-out block to include every
            # wrapped line and every explicit newline at the current width.
            document_layout = document.documentLayout()
            block = document.begin()
            block_height = 0.0
            while block.isValid():
                block_height += document_layout.blockBoundingRect(block).height()
                block = block.next()
            document_height = int(math.ceil(block_height + document.documentMargin() * 2))
            chrome = max(0, self.editor.height() - self.editor.viewport().height())
            # Reserve one extra wrapped line because Qt's plain-text document
            # layout can report fractional line heights before the final paint.
            # Without this small guard, the last line of a long candidate may
            # sit below the fixed-height viewport on some font/scale settings.
            line_guard = max(16, self.editor.fontMetrics().lineSpacing())
            target = max(54, document_height + chrome + line_guard)
            # Content sets the minimum; the editor receives all spare card
            # height instead of leaving that space around the model caption.
            if self.editor.height() != target or self.editor.minimumHeight() != target or self.editor.maximumHeight() != target:
                self.editor.setFixedHeight(target)
        except RuntimeError:
            # The card may already be scheduled for deletion from the virtual cache.
            return

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self._fit_editor_height)

    def eventFilter(self, watched, event):
        if watched is self.editor.viewport():
            if event.type() == QEvent.MouseButtonPress:
                self.focused.emit(self.row_index)
            elif event.type() == QEvent.Resize:
                width = int(watched.width())
                if width != self._last_editor_layout_width:
                    self._last_editor_layout_width = width
                    QTimer.singleShot(0, self._fit_editor_height)
        return super().eventFilter(watched, event)

    def set_draft_source(self, active: bool) -> None:
        self._draft.setText("✓ 手动编辑底稿" if active else "作为底稿")

    def set_selected(self, selected: bool):
        self.check.blockSignals(True)
        self.check.setChecked(selected)
        self.check.setText("✓ 已采用" if selected else "直接采用")
        self.check.blockSignals(False)
        self.setStyleSheet(self._selected_style if selected else self._normal_style)

    def set_history_read_only(self, enabled: bool) -> None:
        """Freeze interactive controls while OCR adjudication history is being browsed."""
        read_only = bool(enabled) or self._historical_evidence
        self.editor.setReadOnly(read_only)
        # Keep the chosen mark visible for audit, but make it impossible to
        # change authority while the history filter is active.  Historical
        # evidence candidates remain permanently non-selectable.
        self._draft.setVisible(not self._historical_evidence)
        self._draft.setEnabled((not enabled) and (not self._historical_evidence))
        self.check.setVisible(not self._historical_evidence)
        self.check.setEnabled((not enabled) and (not self._historical_evidence))


class _FusionDecisionRow(QFrame):
    reference_ready = Signal(str, str)
    draft_requested = Signal(int, int, str, str)
    focused = Signal(int)
    about_to_resolve = Signal(int, int)
    resolved = Signal(int)
    about_to_reopen = Signal(int)
    reopened = Signal(int)
    manual_text_changed = Signal(int)

    def __init__(
        self,
        row_index: int,
        texts: list[str],
        labels: list[str],
        parent=None,
        *,
        preferred_model_index: int | None = None,
        decision_state=None,
    ):
        super().__init__(parent)
        from engine.ocr_compare_view_model import FusionDecisionState

        self.row_index = row_index
        self.labels = list(labels)
        self._decision_state = decision_state or FusionDecisionState.from_texts(
            row_index,
            texts,
            preferred_model_index=preferred_model_index,
            auto_choose=False,
        )
        self._cards: list[_FusionCandidateCard] = []
        self._review_only_candidates = False
        self._showing_history = False
        self._consensus_locked = False
        self._reference_enabled = True
        self.setObjectName("fusionDecisionRow")
        self.setStyleSheet(
            "QFrame#fusionDecisionRow{background:#FFFFFF;border:1px solid #D7E3F4;border-radius:12px;}"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 12)
        root.setSpacing(8)
        header = QHBoxLayout()
        self._title = QLabel("候选结果")
        self._title.setStyleSheet("font-size:11px;font-weight:750;color:#14202E;border:none;")
        header.addWidget(self._title)
        self._state = QLabel("")
        self._state.setStyleSheet("font-size:10px;color:#5B6B80;border:none;")
        header.addWidget(self._state)
        header.addStretch(1)
        self._reopen = QPushButton("重新选择")
        self._reopen.setVisible(False)
        self._reopen.setMaximumHeight(25)
        self._reopen.clicked.connect(self.reopen)
        header.addWidget(self._reopen)
        root.addLayout(header)

        body = QHBoxLayout()
        self._body_layout = body
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(14)

        from ui.ocr.proofread_widgets import OCRProofreadImageLabel
        self._reference_preview = QFrame(self)
        self._reference_preview.setMinimumWidth(255)
        self._reference_preview.setMaximumWidth(320)
        self._reference_preview.setMinimumHeight(0)
        self._reference_preview.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        preview_layout = QVBoxLayout(self._reference_preview)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        self._reference_preview_text = OCRProofreadImageLabel(self)
        self._reference_preview_text.setMinimumSize(235, 260)
        self._reference_preview_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._reference_preview_text.setText("正在载入参考原图…")
        self._reference_preview_text.setToolTip("完整参考裁片；点击原尺寸查看可放大阅读。")
        preview_layout.addWidget(self._reference_preview_text)
        self._reference_open = QPushButton("原尺寸查看")
        self._reference_open.setEnabled(False)
        self._reference_open.clicked.connect(self._open_reference_image)
        preview_layout.addWidget(self._reference_open)
        self._reference_path = ""
        self._reference_source_key = ""
        self._reference_inflight_key = ""
        self._reference_pending_request = None
        self._reference_load_timer = QTimer(self)
        self._reference_load_timer.setSingleShot(True)
        self._reference_load_timer.setInterval(70)
        self._reference_load_timer.timeout.connect(self._start_reference_load)
        self.reference_ready.connect(self._install_reference_image)
        body.addWidget(self._reference_preview, 0)
        self._reference_preview.setVisible(False)

        cards_wrap = QWidget(self)
        cards_wrap.setStyleSheet("background:transparent;border:none;")
        self._cards_layout = QVBoxLayout(cards_wrap)
        self._cards_layout.setContentsMargins(0, 0, 0, 0)
        self._cards_layout.setSpacing(8)
        body.addWidget(cards_wrap, 1)
        root.addLayout(body)

        # All decisions live in a lightweight Python state.  The Qt row is only
        # a temporary view for the current virtual window, so a 300-page book no
        # longer creates thousands of nested text editors at once.
        for candidate_index, candidate in enumerate(self._decision_state.candidates):
            card = _FusionCandidateCard(
                row_index,
                candidate_index,
                candidate.text,
                candidate.model_indices,
                self.labels,
                self,
                candidate_state=candidate,
            )
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
            card.selected.connect(self._select_from_signal)
            card.drafted.connect(self._draft_from_signal)
            card.focused.connect(self.focused)
            card.text_edited.connect(self._candidate_text_changed)
            self._cards_layout.addWidget(card)
            self._cards.append(card)

        if not self._cards:
            card = _FusionCandidateCard(row_index, 0, "", (), self.labels, self)
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
            card.selected.connect(self._select_from_signal)
            card.drafted.connect(self._draft_from_signal)
            card.focused.connect(self.focused)
            card.text_edited.connect(self._candidate_text_changed)
            self._cards_layout.addWidget(card)
            self._cards.append(card)

        self._cards_layout.addStretch(1)

        self._review_card_indices = tuple(self._decision_state.review_indices)

        selected = self._decision_state.selected_index
        if selected is not None:
            selected_candidate = self._decision_state.candidates[selected]
            if bool(getattr(selected_candidate, "synthetic", False)):
                self._state.setText("高置信字符级融合 · 已自动采用（可重新选择任一原始 OCR）")
            elif str(getattr(self._decision_state, "review_classification", "")) == "provisional_consensus":
                self._state.setText("两模型共同候选 · 按 v8 规则自动保留")
            elif len(self._cards) == 1 and len(self._decision_state.candidates[0].model_indices) >= 2:
                self._state.setText("所有独立模型完全一致 · 已自动保留")
            elif len(self._cards) == 1:
                self._state.setText("仅一个有效候选 · 已保留")
            keep_evidence = bool(
                getattr(self._decision_state, "preserve_candidates_visible", False)
            )
            self.choose(selected, collapse=not keep_evidence, emit_signals=False)
            if keep_evidence:
                self._state.setText("AI纠错已采用 · 原始OCR分歧保持可见 · 未覆盖任何模型原文")
                self._state.setStyleSheet("font-size:10px;color:#0F766E;font-weight:700;border:none;")
        else:
            if bool(getattr(self._decision_state, "requires_confirmation", False)):
                self._state.setText("两模型共同候选 · 已自动保留，可查看证据")
                self._state.setStyleSheet("font-size:10px;color:#B45309;font-weight:700;border:none;")
            elif bool(getattr(self._decision_state, "local_reocr_recommended", False)):
                self._state.setText(
                    f"{len(self._cards)} 个候选不完全一致 · 建议局部重识别/人工确认"
                )
                self._state.setStyleSheet("font-size:10px;color:#B45309;font-weight:700;border:none;")
                reason = str(getattr(self._decision_state, "fusion_reason", "") or "")
                if reason:
                    self._state.setToolTip(reason)
            else:
                self._state.setText(f"{len(self._cards)} 个候选不完全一致 · 请打钩选择")
        self._refresh_candidate_diffs()

    def _refresh_reference_preview(self) -> None:
        # The reference is immutable source imagery, independent of the chosen
        # candidate. Editing/choosing text must never change these pixels.
        pass

    def set_consensus_locked(self, locked: bool) -> None:
        self._consensus_locked = bool(locked)
        for card in self._cards:
            card.set_history_read_only(self._consensus_locked or self._showing_history)
        if self._consensus_locked:
            self._reopen.hide()
            self._state.setText("多模型结果一致 · 只读")

    def set_reference_target_height(self, height: int) -> None:
        """Keep one adjudication row compact; the outer scroll owns long content."""
        target = 300 if self._reference_preview.isVisible() else 0
        self.setMinimumHeight(target)

    def set_reference_source(self, document, row, page_images=()):
        if not self._reference_enabled:
            return
        key = f"{id(document)}:{id(row)}"
        if key == self._reference_source_key and (
            bool(self._reference_path)
            or self._reference_load_timer.isActive()
            or self._reference_pending_request is not None
            or self._reference_inflight_key == key
        ):
            return
        self._reference_source_key = key
        self._reference_path = ""
        self._reference_open.setEnabled(False)
        self._reference_preview.setVisible(False)
        self._reference_preview_text.clear_image("正在载入参考原图…")
        # Reference crops can be relatively expensive because a restored
        # workspace may need to reconstruct canonical OCR transport pixels.
        # Defer the work briefly: fast F7/next/previous navigation then cancels
        # rows the user never actually stopped on instead of spawning one
        # background crop thread per transient sentence.
        self._reference_pending_request = (
            key, document, row, tuple(page_images or ()),
        )
        self._reference_load_timer.start()

    def cancel_pending_reference_load(self) -> None:
        """Cancel only not-yet-started crop work when a virtual row leaves view."""
        self._reference_load_timer.stop()
        self._reference_pending_request = None

    def _start_reference_load(self) -> None:
        request = self._reference_pending_request
        self._reference_pending_request = None
        if request is None or not self._reference_enabled:
            return
        key, document, row, page_images = request
        if key != self._reference_source_key:
            return

        import tempfile
        import threading
        cache = tempfile.TemporaryDirectory(prefix="novel-ocr-reference-")
        self._reference_cache = cache
        self._reference_inflight_key = key

        def worker():
            path = ""
            try:
                from engine.ocr_image_text_review import build_comparison_reference_entry, render_review_image
                entry = build_comparison_reference_entry(document, row, fallback_page_images=page_images)
                if entry is not None:
                    path = render_review_image(entry, Path(cache.name) / "reference.png")
            except Exception:
                path = ""
            try:
                self.reference_ready.emit(key, str(path or ""))
            except RuntimeError:
                pass  # This virtual row was disposed while its crop was loading.

        threading.Thread(target=worker, daemon=True, name="ocr-reference-crop").start()

    def _install_reference_image(self, key, path):
        if key != self._reference_source_key:
            return
        if self._reference_inflight_key == key:
            self._reference_inflight_key = ""
        self._reference_path = path
        available = bool(path and self._reference_preview_text.set_image_path(path))
        self._reference_open.setEnabled(available)
        self._reference_preview.setVisible(available and self._reference_enabled)
        if not available:
            self._reference_preview_text.clear_image("此句没有可定位的参考裁片，请在图文对照查看源页。")

    def set_reference_visible(self, visible: bool) -> None:
        """Keep asynchronous image loads from reopening the full-text preview."""
        self._reference_enabled = bool(visible)
        self._reference_preview.setVisible(self._reference_enabled and bool(self._reference_path))
        if not self._reference_enabled:
            self.setMinimumHeight(0)

    def _open_reference_image(self):
        if not self._reference_path:
            return
        from PySide6.QtGui import QPixmap
        dialog = QDialog(self)
        dialog.setWindowTitle("OCR 参考原图 · 原尺寸")
        dialog.resize(800, 850)
        layout = QVBoxLayout(dialog)
        scroll = QScrollArea(dialog)
        image = QLabel()
        image.setPixmap(QPixmap(self._reference_path))
        image.setAlignment(Qt.AlignCenter)
        scroll.setWidget(image)
        scroll.setWidgetResizable(False)
        layout.addWidget(scroll)
        dialog.exec()

    @staticmethod
    def _candidate_signature(candidate) -> tuple:
        return (
            str(getattr(candidate, "text", "") or ""),
            tuple(getattr(candidate, "model_indices", ()) or ()),
            str(getattr(candidate, "display_label", "") or ""),
            bool(getattr(candidate, "synthetic", False)),
            bool(getattr(candidate, "delete_intentionally", False)),
            str(getattr(candidate, "transaction_id", "") or ""),
            str(getattr(candidate, "audit_level", "") or ""),
            tuple(str(value) for value in (getattr(candidate, "audit_flags", ()) or ())),
        )

    def is_compatible_with(self, decision_state, labels) -> bool:
        """Return whether a cached Qt row still represents the live state."""
        if self._decision_state is not decision_state or self.labels != list(labels):
            return False
        candidates = list(getattr(decision_state, "candidates", ()) or ())
        if len(candidates) != len(self._cards):
            return False
        for card, candidate in zip(self._cards, candidates):
            if getattr(card, "_candidate_state", None) is not candidate:
                return False
            if card.editor.toPlainText() != str(getattr(candidate, "text", "") or ""):
                return False
            if self._candidate_signature(candidate) != getattr(card, "_candidate_state_signature", ()):
                return False
        return True

    def sync_from_state(self) -> None:
        """Refresh check marks/visibility after off-screen or transactional edits."""
        self._cards_layout.addStretch(1)

        self._review_card_indices = tuple(self._decision_state.review_indices)
        self._showing_history = False
        selected = self._decision_state.selected_index
        keep_evidence = bool(
            getattr(self._decision_state, "preserve_candidates_visible", False)
        )
        if selected is not None and 0 <= selected < len(self._cards):
            for index, card in enumerate(self._cards):
                card.set_selected(index == selected)
                card.setVisible(True if keep_evidence else index == selected)
            self._reopen.setVisible(
                (not keep_evidence)
                and (len(self._cards) > 1 or bool(getattr(self._decision_state, "requires_confirmation", False)))
            )
            if keep_evidence:
                label = str(
                    getattr(self._decision_state.candidates[selected], "display_label", "")
                    or "当前候选"
                )
                self._state.setText(f"已选择 {label} · 原始OCR分歧保持可见 · 源模型未被覆盖")
            else:
                self._state.setText("已选择；未选候选已收起，当前文字仍可编辑")
        else:
            self._reopen.setVisible(False)
            allowed = (
                set(self._review_card_indices)
                if self._review_only_candidates
                else set(range(len(self._cards)))
            )
            for index, card in enumerate(self._cards):
                card.set_selected(False)
                card.setVisible(index in allowed)
            self._state.setText(f"{len(self._cards)} 个候选不完全一致 · 请打钩选择")
        self._refresh_candidate_diffs()
        self._refresh_reference_preview()

    def set_history_mode(self, enabled: bool) -> None:
        """Render an already-adjudicated row as immutable audit history."""
        self._showing_history = bool(enabled)
        for card in self._cards:
            card.set_history_read_only(bool(enabled))
            if self._consensus_locked and not enabled:
                card.set_history_read_only(True)
        if enabled:
            self._reopen.setVisible(False)
            from engine.ocr_compare_view_model import fusion_decision_origin_label
            label = fusion_decision_origin_label(
                str(getattr(self._decision_state, "selection_origin", "") or "")
            ) or "明确裁决"
            self._state.setText(f"已裁决 · 只读历史 · {label}")
        else:
            # Rebuild the normal selected/unresolved presentation from the live
            # state without replacing the state object or candidate texts.
            self.sync_from_state()

    def _candidate_text_changed(self, candidate_index: int):
        # Editing the currently selected text is an explicit human decision even
        # when the candidate was originally auto-selected or imported by AI.
        if self._decision_state.selected_index == int(candidate_index):
            self._decision_state.selection_origin = "human_manual_edit"
        self._refresh_candidate_diffs()
        self._refresh_reference_preview()
        if self._decision_state.selected_index == int(candidate_index):
            self.manual_text_changed.emit(self.row_index)

    @staticmethod
    def _merge_marks(marks):
        grouped = {}
        for start, end, kind in marks or ():
            grouped.setdefault(kind, []).append((int(start), int(end)))
        merged = []
        for kind, ranges in grouped.items():
            for start, end in sorted(ranges):
                if merged and merged[-1][2] == kind and start <= merged[-1][1]:
                    old_start, old_end, _ = merged[-1]
                    merged[-1] = (old_start, max(old_end, end), kind)
                else:
                    merged.append((start, end, kind))
        return tuple(merged)

    def _refresh_candidate_diffs(self):
        from engine.ocr_compare_view_model import paired_candidate_diff_spans

        for card in self._cards:
            card.clear_diff_marks()
        if (
            self._decision_state.selected_index is not None
            and not self._showing_history
            and not bool(getattr(self._decision_state, "preserve_candidates_visible", False))
        ):
            return

        visible = (
            list(range(len(self._cards)))
            if (
                self._showing_history
                or bool(getattr(self._decision_state, "preserve_candidates_visible", False))
            )
            else list(self._decision_state.visible_candidate_indices(self._review_only_candidates))
        )
        visible = [index for index in visible if 0 <= index < len(self._cards)]
        if len(visible) < 2:
            return

        marks_by_index = {index: [] for index in visible}
        total_changes = 0
        if len(visible) == 2:
            left_index, right_index = visible
            left_marks, right_marks, total_changes = paired_candidate_diff_spans(
                self._cards[left_index].editor.toPlainText(),
                self._cards[right_index].editor.toPlainText(),
            )
            marks_by_index[left_index].extend(left_marks)
            marks_by_index[right_index].extend(right_marks)
        else:
            # In the full-candidate view use the preferred/recommended card as
            # a stable reference and mark every alternative against it.
            reference = self._review_card_indices[0] if self._review_card_indices else visible[0]
            if reference not in visible:
                reference = visible[0]
            reference_text = self._cards[reference].editor.toPlainText()
            for index in visible:
                if index == reference:
                    continue
                ref_marks, other_marks, changes = paired_candidate_diff_spans(
                    reference_text, self._cards[index].editor.toPlainText()
                )
                marks_by_index[reference].extend(ref_marks)
                marks_by_index[index].extend(other_marks)
                total_changes += changes

        for index in visible:
            self._cards[index].set_diff_marks(
                self._merge_marks(marks_by_index[index]),
                change_count=total_changes,
            )
        if total_changes:
            if (
                self._decision_state.selected_index is not None
                and bool(getattr(self._decision_state, "preserve_candidates_visible", False))
            ):
                self._state.setText(
                    f"AI纠错已采用 · 原始OCR分歧保持可见 · 差异 {total_changes} 处"
                )
            else:
                self._state.setText(
                    f"{len(self._cards)} 个候选不完全一致 · 差异 {total_changes} 处 · 请打钩选择"
                )
        else:
            self._state.setText("候选文字一致；来源仍分别保留，可重新选择")

    def _build_review_card_indices(self) -> tuple[int, ...]:
        return tuple(self._decision_state.review_indices)

    def set_review_only_candidates(self, enabled: bool):
        self._review_only_candidates = bool(enabled)
        selected_index = self._decision_state.selected_index
        if selected_index is not None:
            keep_evidence = bool(
                getattr(self._decision_state, "preserve_candidates_visible", False)
            )
            for index, card in enumerate(self._cards):
                card.setVisible(True if keep_evidence else index == selected_index)
            self._refresh_candidate_diffs()
            return
        allowed = set(self._review_card_indices) if enabled else set(range(len(self._cards)))
        for index, card in enumerate(self._cards):
            card.setVisible(index in allowed)
        self._refresh_candidate_diffs()

    def _draft_from_signal(self, row_index: int, candidate_index: int):
        if row_index != self.row_index or not 0 <= int(candidate_index) < len(self._cards):
            return
        card = self._cards[int(candidate_index)]
        self.draft_requested.emit(
            self.row_index, int(candidate_index), card.editor.toPlainText(), str(card._base_caption or "候选")
        )
        self.focused.emit(self.row_index)

    def stage_candidate(self, candidate_index: int) -> bool:
        if self._showing_history or self._consensus_locked:
            return False
        if not 0 <= int(candidate_index) < len(self._cards):
            return False
        self._draft_from_signal(self.row_index, int(candidate_index))
        return True

    def _select_from_signal(self, row_index: int, candidate_index: int):
        if row_index == self.row_index:
            self.choose(candidate_index, collapse=True)

    @property
    def unresolved(self) -> bool:
        return self._decision_state.unresolved

    def output_text(self) -> str:
        return self._decision_state.output_text()

    def choose(self, candidate_index: int, *, collapse: bool = True, emit_signals: bool = True):
        if self._showing_history or self._consensus_locked:
            return
        if emit_signals:
            self.about_to_resolve.emit(self.row_index, int(candidate_index))
            if not self._decision_state.choose(candidate_index, origin="human_ocr_compare"):
                return
        else:
            # Constructor/state-sync calls are presentation-only.  Re-selecting
            # an already authoritative imported/manual candidate here used to
            # overwrite selection_origin with human_ocr_compare.
            if self._decision_state.selected_index != int(candidate_index):
                self._decision_state.selected_index = int(candidate_index)
        self._showing_history = False
        keep_evidence = bool(
            getattr(self._decision_state, "preserve_candidates_visible", False)
        )
        collapse = bool(collapse and not keep_evidence)
        for index, card in enumerate(self._cards):
            selected = index == candidate_index
            card.set_selected(selected)
            card.setVisible(selected if collapse else True)
        self._reopen.setVisible(
            collapse and (len(self._cards) > 1 or bool(getattr(self._decision_state, "requires_confirmation", False)))
        )
        self._refresh_candidate_diffs()
        self._refresh_reference_preview()
        if keep_evidence:
            candidate = self._decision_state.candidates[candidate_index]
            label = str(getattr(candidate, "display_label", "") or "当前候选")
            self._state.setText(f"已选择 {label} · 原始OCR分歧保持可见 · 源模型未被覆盖")
        else:
            self._state.setText("已选择；未选候选已收起，当前文字仍可编辑")
        if emit_signals:
            self.resolved.emit(self.row_index)
            self.focused.emit(self.row_index)

    def choose_model(self, model_index: int) -> bool:
        for index, card in enumerate(self._cards):
            if model_index in card.model_indices:
                self.choose(index, collapse=True)
                return True
        return False

    def choose_text(self, text: str) -> bool:
        target = str(text or "").strip()
        for index, card in enumerate(self._cards):
            if card.editor.toPlainText().strip() == target:
                self.choose(index, collapse=True)
                return True
        return False

    def reopen(self):
        if self._showing_history:
            return
        if str(getattr(self._decision_state, "review_classification", "") or "") == "source_correction_resolved":
            self._showing_history = True
            selected = self._decision_state.selected_index
            for index, card in enumerate(self._cards):
                card.setVisible(True)
                card.set_selected(index == selected)
            self._reopen.setVisible(False)
            self._refresh_candidate_diffs()
            self._state.setText("已完成裁决 · 下方显示纠错前只读分歧证据")
            self.focused.emit(self.row_index)
            return
        self.about_to_reopen.emit(self.row_index)
        if not self._decision_state.reopen():
            return
        self._showing_history = False
        allowed = (
            set(self._review_card_indices)
            if self._review_only_candidates
            else set(range(len(self._cards)))
        )
        for index, card in enumerate(self._cards):
            card.setVisible(index in allowed)
            card.set_selected(False)
        self._reopen.setVisible(False)
        self._refresh_candidate_diffs()
        if bool(getattr(self._decision_state, "requires_confirmation", False)) and len(self._cards) == 1:
            self._state.setText("两模型共同候选 · 已重新打开证据")
        else:
            self._state.setText(
                f"{len(self._cards)} 个候选不完全一致 · 差异已标注 · 请重新打钩选择"
            )
        self.reopened.emit(self.row_index)
        self.focused.emit(self.row_index)

    def set_current(self, current: bool):
        # The active sentence is already unambiguous in the left queue.  Avoid
        # a second heavy blue outline around the entire decision card; the
        # chosen candidate itself carries the accent, matching the reference.
        border = "#C9D9EE" if current else "#D7E3F4"
        self.setStyleSheet(
            f"QFrame#fusionDecisionRow{{background:#FFFFFF;border:1px solid {border};border-radius:12px;}}"
        )


class OCRAIAdjudicationDialog(QDialog):
    """Configuration for low-token batched visual OCR adjudication."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("出版级 AI 裁决")
        self.setMinimumWidth(680)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        note = QLabel(
            "出版级模式会先让当前已配置的视觉 AI 在看不到 A/B/C 的情况下独立逐字抄写，"
            "再把视觉结果、全部 OCR 候选、物理列、前后文和已保存术语交给第二阶段上下文裁决，"
            "最后由独立审计器反查。支持 OpenAI、Gemini、DeepSeek、GLM、Claude、OpenRouter 等已配置服务。"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#5B6B80;background:#F7F8FA;border:1px solid #C7D7FE;border-radius:8px;padding:8px;")
        layout.addWidget(note)

        provider_row = QHBoxLayout()
        provider_row.setSpacing(8)
        self._provider_state = QLabel("")
        self._provider_state.setStyleSheet("color:#334155;font-size:11px;font-weight:650;")
        provider_row.addWidget(self._provider_state, 1)
        provider_btn = QPushButton("切换 AI 服务…")
        provider_btn.clicked.connect(self._open_ai_settings)
        provider_row.addWidget(provider_btn)
        layout.addLayout(provider_row)
        self._refresh_provider_state()

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(8)

        self.adjudication_mode = QComboBox()
        self.adjudication_mode.addItem("出版级 AI · 视觉抄写 + 上下文裁决 + 独立审计（推荐）", "gpt_grade")
        self.adjudication_mode.addItem("纯视觉 · 独立抄写 + 候选比较（更快）", "visual_only")
        form.addRow("裁决模式：", self.adjudication_mode)

        preset_row = QWidget(); preset_layout = QHBoxLayout(preset_row)
        preset_layout.setContentsMargins(0,0,0,0); preset_layout.setSpacing(8)
        self.preset = QComboBox()
        self.preset.addItem("出版精细 · 16 条/请求（推荐）", 16)
        self.preset.addItem("均衡 · 24 条/请求", 24)
        self.preset.addItem("快速 · 32 条/请求", 32)
        preset_layout.addWidget(self.preset)
        self.batch_items = NoWheelSpinBox(); self.batch_items.setRange(4,32); self.batch_items.setValue(16)
        preset_layout.addWidget(QLabel("实际批量")); preset_layout.addWidget(self.batch_items)
        preset_layout.addStretch(1)
        self.preset.currentIndexChanged.connect(lambda _=None: self.batch_items.setValue(int(self.preset.currentData() or 16)))
        form.addRow("批量：", preset_row)

        self.routing = QComboBox()
        self.routing.addItem("智能均衡 · 保留实质 2:1 分歧，仅跳过高置信标点/空格差异", "smart")
        self.routing.addItem("出版全部 · 所有分歧/低置信都送视觉裁决", "exhaustive")
        self.routing.addItem("极速高风险 · 可跳过干净高置信 2:1", "lean")
        form.addRow("裁决范围：", self.routing)

        concurrency_row = QWidget(); concurrency_layout = QHBoxLayout(concurrency_row)
        concurrency_layout.setContentsMargins(0,0,0,0); concurrency_layout.setSpacing(8)
        self.concurrency = NoWheelSpinBox(); self.concurrency.setRange(0,16); self.concurrency.setValue(0)
        self.concurrency.setToolTip("0=自动：在线模型从 4 路起步；稳定时逐步增加，检测到传输重试则减半。Ollama 默认单路。")
        concurrency_layout.addWidget(self.concurrency)
        concurrency_layout.addWidget(QLabel("0=自动（推荐）"))
        concurrency_layout.addStretch(1)
        form.addRow("并发请求：", concurrency_row)

        self.reasoning = QComboBox()
        self.reasoning.addItem("low · 快速", "low")
        self.reasoning.addItem("high · 出版裁决推荐", "high")
        self.reasoning.addItem("max · 极少数疑难页", "max")
        self.reasoning.setCurrentIndex(1)
        form.addRow("视觉思考：", self.reasoning)

        context_row = QWidget(); context_layout = QHBoxLayout(context_row)
        context_layout.setContentsMargins(0,0,0,0); context_layout.setSpacing(8)
        self.context_pages = NoWheelSpinBox(); self.context_pages.setRange(10,30); self.context_pages.setValue(20)
        context_layout.addWidget(self.context_pages)
        context_layout.addWidget(QLabel("页/批；自动带前后 2 页锚点"))
        context_layout.addStretch(1)
        form.addRow("上下文窗口：", context_row)

        self.output_dir = QLineEdit()
        desktop = Path.home() / "Desktop"
        self.output_dir.setText(str(desktop if desktop.is_dir() else Path.home()))
        output_row = QWidget(); output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0,0,0,0); output_layout.setSpacing(6)
        output_layout.addWidget(self.output_dir, 1)
        output_btn = QPushButton("选择…")
        output_btn.clicked.connect(self._choose_output)
        output_layout.addWidget(output_btn)
        form.addRow("审计报告：", output_row)
        layout.addLayout(form)

        self.risk_only = QCheckBox("只发送真正分歧、低置信、疑似缺失/重复条目（推荐）")
        self.risk_only.setChecked(True)
        self.allow_novel = QCheckBox("允许 X：所有本地候选都错时，AI 可返回少量精确新字；仍需本地防改写校验")
        self.allow_novel.setChecked(True)
        self.retry_incomplete = QCheckBox("AI 漏回较多编号时，只把缺失项缩小批次自动补跑一次（推荐）")
        self.retry_incomplete.setChecked(True)
        self.independent_audit = QCheckBox("出版级模式完成后再用独立审计器反查缺字、重复、邻列粘连、数字/否定/专名（推荐）")
        self.independent_audit.setChecked(True)
        layout.addWidget(self.risk_only)
        layout.addWidget(self.allow_novel)
        layout.addWidget(self.retry_incomplete)
        layout.addWidget(self.independent_audit)

        warning = QLabel(
            "证据规则：图片必须覆盖该行声明的全部物理列和页面；有图片文件不等于证据完整。"
            "上下文只能用于消歧，不能凭语法润色或续写；模型置信度不参与自动放行。"
            "本地安全闸门或独立审计失败的条目会保留人工复核。"
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("color:#92400E;background:#FFF8E7;border:1px solid #F2D18A;border-radius:8px;padding:8px;")
        layout.addWidget(warning)

        buttons = QHBoxLayout(); buttons.addStretch(1)
        cancel = QPushButton("取消"); cancel.clicked.connect(self.reject); buttons.addWidget(cancel)
        start = accent_button("开始出版级 AI 裁决", color="#7C3AED"); start.clicked.connect(self._accept_checked); buttons.addWidget(start)
        layout.addLayout(buttons)

    def _refresh_provider_state(self):
        try:
            from ai.config import load_ai_settings
            st = load_ai_settings()
            names = {
                "openai": "OpenAI", "anthropic": "Claude / Anthropic", "gemini": "Google Gemini",
                "deepseek": "DeepSeek", "zhipu": "智谱 GLM（国内）", "zai": "Z.AI / GLM（国际）",
                "openrouter": "OpenRouter", "ollama": "Ollama（本地）", "custom": "自定义 OpenAI 兼容接口",
            }
            name = names.get(str(st.provider or "").lower(), str(st.provider or "未配置"))
            self._provider_state.setText(f"当前 AI：{name} · {str(st.model or '未选模型')}")
        except Exception:
            self._provider_state.setText("当前 AI：未配置")

    def _open_ai_settings(self):
        from ui.settings.ai_dialog import AISettingsDialog
        AISettingsDialog(self).exec()
        self._refresh_provider_state()

    def _choose_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择 AI 裁决报告目录", self.output_dir.text())
        if path:
            self.output_dir.setText(path)

    def _accept_checked(self):
        output = Path(self.output_dir.text().strip()).expanduser()
        try:
            output.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            QMessageBox.warning(self, "结果目录不可用", str(exc))
            return
        self.accept()

    def values(self) -> dict:
        return {
            "adjudication_mode": str(self.adjudication_mode.currentData() or "gpt_grade"),
            "output_dir": self.output_dir.text().strip(),
            "batch_items": self.batch_items.value(),
            "risk_only": self.risk_only.isChecked(),
            "allow_novel_text": self.allow_novel.isChecked(),
            "reasoning_effort": str(self.reasoning.currentData() or "low"),
            "routing_mode": str(self.routing.currentData() or "smart"),
            "concurrency": int(self.concurrency.value()),
            "retry_incomplete": self.retry_incomplete.isChecked(),
            "context_pages": int(self.context_pages.value()),
            "independent_audit": self.independent_audit.isChecked(),
        }


class DecisionQueueListModel(QAbstractListModel):
    """Virtual decision queue for thousands of OCR review rows.

    QListWidget eagerly constructs one item object per row and the previous code
    cleared/rebuilt all of them after every decision.  A Qt item model keeps only
    integer row identities; display strings are produced lazily for visible rows,
    and one resolved decision becomes one beginRemoveRows/endRemoveRows pair.
    """

    def __init__(self, display_provider, tooltip_provider, parent=None):
        super().__init__(parent)
        self._rows: list[int] = []
        self._position_by_row: dict[int, int] = {}
        self._display_provider = display_provider
        self._tooltip_provider = tooltip_provider

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._rows)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self._rows):
            return None
        row_index = self._rows[index.row()]
        if role == Qt.DisplayRole:
            return self._display_provider(row_index)
        if role == Qt.ToolTipRole:
            return self._tooltip_provider(row_index)
        if role == Qt.UserRole:
            return row_index
        return None

    @property
    def rows(self) -> tuple[int, ...]:
        return tuple(self._rows)

    def _reindex_from(self, start: int = 0) -> None:
        start = max(0, int(start))
        for position in range(start, len(self._rows)):
            self._position_by_row[self._rows[position]] = position

    def set_rows(self, rows) -> None:
        values = [int(value) for value in rows]
        if values == self._rows:
            return
        self.beginResetModel()
        self._rows = values
        self._position_by_row = {row: pos for pos, row in enumerate(values)}
        self.endResetModel()

    def remove_row_index(self, row_index: int) -> bool:
        row_index = int(row_index)
        position = self._position_by_row.pop(row_index, None)
        if position is None:
            return False
        self.beginRemoveRows(QModelIndex(), position, position)
        self._rows.pop(position)
        self._reindex_from(position)
        self.endRemoveRows()
        return True

    def remove_row_indices(self, row_indices) -> int:
        positions = sorted(
            (self._position_by_row.get(int(value)), int(value)) for value in row_indices
            if self._position_by_row.get(int(value)) is not None
        )
        removed = 0
        # Remove from the end so earlier positions remain stable.
        for position, row_index in reversed(positions):
            self.beginRemoveRows(QModelIndex(), position, position)
            self._rows.pop(position)
            self._position_by_row.pop(row_index, None)
            self.endRemoveRows()
            removed += 1
        if removed:
            self._position_by_row = {row: pos for pos, row in enumerate(self._rows)}
        return removed

    def model_index_for_row(self, row_index: int):
        position = self._position_by_row.get(int(row_index))
        return self.index(position, 0) if position is not None else QModelIndex()

    def refresh_row(self, row_index: int) -> None:
        index = self.model_index_for_row(row_index)
        if index.isValid():
            self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.ToolTipRole])
