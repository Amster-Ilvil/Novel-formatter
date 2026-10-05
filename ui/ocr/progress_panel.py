from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QCheckBox, QSizePolicy, QProgressBar, QPlainTextEdit, QTextEdit,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction

from ui.common.styling import ACC, BORDER, CARD, INK, MUTED, TONAL, LIGHT_LOG_STYLE, accent_button, make_separator
from ui.responsive import preserve_button_text
from ui.design.metrics import OCR_BOTTOM_HEIGHT
from ui.design.components import DesignLogCollapseButton


def build_ocr_progress_panel(tab, main_v):
    self = tab
    # ── 底部：常驻 OCR 日志；加高面板，让日志位置整体上移 ────────────────
    bottom = QWidget()
    self._ocr_log_panel = bottom
    self._ocr_log_expanded_height = OCR_BOTTOM_HEIGHT
    self._ocr_log_collapsed_height = 54
    bottom.setFixedHeight(self._ocr_log_expanded_height)
    bottom.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    bottom.setStyleSheet(f"background: {CARD}; border: none;")
    rl = QVBoxLayout(bottom)
    rl.setContentsMargins(0, 0, 0, 0)
    rl.setSpacing(0)

    # Hidden accessibility mirror of the progress text in the log header.
    self._progress_lbl = QLabel("")
    self._progress_lbl.setVisible(False)

    self._phase_progress_lbl = QLabel("")
    self._phase_progress_lbl.setStyleSheet(
        f"color: #5F7189; font-size: 10px; padding: 1px 14px 3px;"
    )
    self._phase_progress_lbl.setVisible(False)
    self._phase_progress_lbl.setWordWrap(True)

    log_header = QHBoxLayout()
    log_header.setContentsMargins(18, 9, 18, 9)
    log_header.setSpacing(8)
    self._ocr_log_collapse_btn = DesignLogCollapseButton(bottom)
    self._ocr_log_collapse_btn.setObjectName("ocrLogCollapseButton")
    self._ocr_log_collapse_btn.setArrowType(Qt.UpArrow)
    self._ocr_log_collapse_btn.clicked.connect(
        lambda: self._set_ocr_log_collapsed(not getattr(self, "_ocr_log_collapsed", False))
    )
    # Make the redesigned icon the true left-most control, as in the current
    # Phase 22 header contract.  It carries both "log" meaning and the
    # expanded/collapsed chevron, so no platform-native triangle leaks in.
    log_header.addWidget(self._ocr_log_collapse_btn)

    log_title = QLabel("OCR 日志")
    log_title.setStyleSheet(f"color: {INK}; font-size: 14px; font-weight: 700;")
    log_header.addWidget(log_title)
    self._ocr_log_collapse_action = QAction("收起日志", bottom)
    self._ocr_log_collapse_action.triggered.connect(
        lambda: self._set_ocr_log_collapsed(not getattr(self, "_ocr_log_collapsed", False))
    )
    bottom.addAction(self._ocr_log_collapse_action)
    bottom.setContextMenuPolicy(Qt.ActionsContextMenu)
    self._progress_bar_text = QLabel("总进度 0.0% · 已用 0秒 · 预计剩余：计算中")
    self._progress_bar_text.setAlignment(Qt.AlignCenter)
    self._progress_bar_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    self._progress_bar_text.setAttribute(Qt.WA_TransparentForMouseEvents, True)
    self._progress_bar_text.setStyleSheet(
        f"color: {MUTED}; background: none; border: none; padding: 0 8px; "
        "font-size: 11px; font-weight: 600;"
    )
    self._progress_bar_text.setVisible(False)
    # Keep the header's expanding space when progress text is hidden at idle.
    progress_header_space = QWidget(bottom)
    progress_header_layout = QHBoxLayout(progress_header_space)
    progress_header_layout.setContentsMargins(0, 0, 0, 0)
    progress_header_layout.addWidget(self._progress_bar_text)
    log_header.addWidget(progress_header_space, 1)

    self._progress_display_cb = QCheckBox("显示实时进度")
    self._progress_display_cb.setChecked(True)
    self._progress_display_cb.setToolTip(
        "关闭后隐藏蓝色总进度、当前操作与预计时间，并停止高频进度信号和界面刷新；"
        "OCR 模型、分列、整句重识别和结果输出继续正常运行。可在运行中随时重新开启。"
    )
    self._progress_display_cb.toggled.connect(self._toggle_progress_display)
    log_header.addWidget(self._progress_display_cb)

    self._pause_btn = QPushButton("停止 OCR")
    self._pause_btn.setProperty("role", "secondary")
    preserve_button_text(self._pause_btn)
    self._pause_btn.setMinimumWidth(112)
    self._pause_btn.setMinimumHeight(36)
    self._pause_btn.setVisible(True)
    self._pause_btn.setEnabled(False)
    self._pause_btn.clicked.connect(self._toggle_pause)
    log_header.addWidget(self._pause_btn)

    self._run_btn = accent_button("▶  开始 OCR")
    self._run_btn.setObjectName("ocrRunButtonBesideProgress")
    self._run_btn.setMinimumHeight(36)
    preserve_button_text(self._run_btn)
    self._run_btn.clicked.connect(self._run_ocr)
    log_header.addWidget(self._run_btn)

    self._rerun_btn = QPushButton("重新 OCR")
    self._rerun_btn.setProperty("role", "secondary")
    preserve_button_text(self._rerun_btn)
    self._rerun_btn.setVisible(False)
    self._rerun_btn.clicked.connect(self._re_ocr)
    log_header.addWidget(self._rerun_btn)
    self._ocr_log_header = log_header

    rl.addLayout(log_header)

    self._ocr_log_body = QWidget(bottom)
    self._ocr_log_body.setObjectName("ocrLogBody")
    body_layout = QVBoxLayout(self._ocr_log_body)
    body_layout.setContentsMargins(0, 0, 0, 0)
    body_layout.setSpacing(0)

    self._progress_bar_wrap = QWidget()
    self._progress_bar_wrap.setVisible(False)
    progress_bar_layout = QGridLayout(self._progress_bar_wrap)
    progress_bar_layout.setContentsMargins(18, 3, 18, 6)
    progress_bar_layout.setSpacing(0)
    progress_bar_layout.setColumnStretch(0, 1)

    self._prog = QProgressBar()
    self._prog.setFixedHeight(8)
    self._prog.setTextVisible(False)
    self._prog.setStyleSheet(
        f"QProgressBar {{ background-color: {TONAL}; border: none; border-radius: 4px; }} "
        f"QProgressBar::chunk {{ background-color: {ACC}; border-radius: 4px; }}"
    )
    progress_bar_layout.addWidget(self._prog, 0, 0)

    body_layout.addWidget(self._progress_bar_wrap)
    body_layout.addWidget(self._phase_progress_lbl)

    self._log_view = QPlainTextEdit()
    self._log_view.setReadOnly(True)
    self._log_view.setStyleSheet(LIGHT_LOG_STYLE)

    # During OCR the log follows the newest line explicitly. QPlainTextEdit's
    # implicit append scrolling is not reliable after the user switched pages,
    # selected text, or the document was updated in buffered batches.
    def _follow_ocr_log_tail():
        if not bool(getattr(self, "_ocr_run_active", False)):
            return
        bar = self._log_view.verticalScrollBar()
        bar.setValue(bar.maximum())
    self._log_view.textChanged.connect(_follow_ocr_log_tail)
    self._ocr_log_view_wrap = QWidget(self._ocr_log_body)
    log_view_layout = QVBoxLayout(self._ocr_log_view_wrap)
    log_view_layout.setContentsMargins(18, 0, 18, 14)
    log_view_layout.setSpacing(0)
    log_view_layout.addWidget(self._log_view, 1)
    body_layout.addWidget(self._ocr_log_view_wrap, 1)
    rl.addWidget(self._ocr_log_body, 1)

    # 兼容旧插件/完成回调对该成员的访问，但不再显示“OCR结果”页签或面板。
    self._result_view = QTextEdit(bottom)
    self._result_view.setReadOnly(True)
    self._result_view.setVisible(False)

    saved_collapsed = self._ocr_mode_settings.value(
        "ui/ocr_log_collapsed", False, type=bool
    )
    self._set_ocr_log_collapsed(saved_collapsed, persist=False)

    main_v.addWidget(bottom)
    # Single-model and multi-model configurations are intentionally
    # independent.  A fresh installation keeps Hayai as the ordinary single
    # OCR choice, while all six multi-model role slots start empty.  Turning

