from __future__ import annotations

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox, QSizePolicy, QFrame
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut

from ui.common.styling import BG, BORDER, CARD, INK, MUTED, TONAL, TONAL_HOVER
from ui.theme.tokens import DISABLED_BG, DISABLED_FG
from ui.ocr.preview import OCRCropPreview


def build_ocr_preview_panel(tab, top_row):
    self = tab
    # ── 中间：大幅放大的识别区域预览（拖框选定，替代原来的百分比裁剪）───────
    center = QWidget()
    center.setStyleSheet(f"background: {BG};")
    cv = QVBoxLayout(center)
    # Avoid a wide blank gutter between the settings column and the preview;
    # this is especially costly in character-review mode on a narrow window.
    cv.setContentsMargins(10, 15, 12, 15)
    cv.setSpacing(10)

    # Phase 21: match the supplied reference navigation row.  The controls
    # remain the same objects/signals; only their visual order changes.
    preview_nav_row = QHBoxLayout()
    preview_nav_row.setContentsMargins(0, 0, 0, 0)
    preview_nav_row.setSpacing(6)

    preview_button_style = (
        f"QPushButton {{ font-size: 11px; padding: 4px 12px; min-height: 22px; border: none; "
        f"border-radius: 9px; background: {TONAL}; color: {INK}; font-weight: 600; }}"
        f"QPushButton:hover {{ background: {TONAL_HOVER}; }}"
        f"QPushButton:disabled {{ background: {DISABLED_BG}; color: {DISABLED_FG}; }}"
    )

    self._preview_prev_btn = QPushButton("‹  上一页")
    self._preview_prev_btn.setObjectName("ocrPreviewPreviousButton")
    self._preview_prev_btn.setStyleSheet(preview_button_style)
    self._preview_prev_btn.setMinimumHeight(30)
    self._preview_prev_btn.setVisible(True)
    self._preview_prev_btn.setToolTip("返回已经保留的上一张实时预览；OCR 运行中也可使用（快捷键：⌥←）")
    self._preview_prev_btn.clicked.connect(self._show_previous_preview_page)
    preview_nav_row.addWidget(self._preview_prev_btn)

    self._preview_next_btn = QPushButton("下一页 →")
    self._preview_next_btn.setObjectName("ocrPreviewNextButton")
    self._preview_next_btn.setStyleSheet(preview_button_style)
    self._preview_next_btn.setMinimumHeight(30)
    self._preview_next_btn.setVisible(True)
    self._preview_next_btn.setToolTip("前往已经保留的下一张实时预览；OCR 运行中也可使用（快捷键：⌥→）")
    self._preview_next_btn.clicked.connect(self._show_next_preview_page)
    preview_nav_row.addWidget(self._preview_next_btn)

    self._preview_filename_lbl = QLabel("当前图片：尚未载入")
    self._preview_filename_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px; font-weight: 600;")
    self._preview_filename_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
    self._preview_filename_lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    preview_nav_row.addWidget(self._preview_filename_lbl, 1)

    self._preview_detector_box_check = QCheckBox("检测框")
    self._preview_detector_box_check.setChecked(True)
    self._preview_detector_box_check.setToolTip("显示/隐藏右侧实时预览中的检测框")
    preview_nav_row.addWidget(self._preview_detector_box_check)
    self._preview_input_box_check = QCheckBox("输入框")
    self._preview_input_box_check.setChecked(False)
    self._preview_input_box_check.setToolTip("显示/隐藏右侧实时预览中的 OCR 输入原像素框")
    preview_nav_row.addWidget(self._preview_input_box_check)

    self._preview_enabled_cb = QCheckBox("实时预览")
    self._preview_enabled_cb.setChecked(True)
    self._preview_enabled_cb.setStyleSheet("font-size: 11px;")
    self._preview_enabled_cb.setToolTip(
        "可在 OCR 运行过程中随时关闭或重新开启。关闭后停止生成、保存和刷新新的预览图，"
        "OCR、分列、整句重识别和结果输出继续正常运行；开启时会保留本轮全部正文页的缩略预览。"
    )
    self._preview_enabled_cb.toggled.connect(self._on_live_preview_toggled)
    preview_nav_row.addWidget(self._preview_enabled_cb)

    # Phase 23: keep the diagnostic column-preview action in the main preview
    # header.  It remains usable before the OCR split switch is enabled, so the
    # user can inspect physical columns before committing to the pipeline.
    self._preview_columns_btn = QPushButton("预览分列")
    self._preview_columns_btn.setObjectName("ocrPreviewColumnsButton")
    self._preview_columns_btn.setStyleSheet(preview_button_style)
    self._preview_columns_btn.setMinimumHeight(30)
    self._preview_columns_btn.setToolTip("预览当前页的物理分列与实际单列 OCR 输入")
    self._preview_columns_btn.clicked.connect(self._preview_column_split)
    self._preview_columns_btn.setEnabled(True)
    preview_nav_row.addWidget(self._preview_columns_btn)

    self._preview_clear_crop_btn = QPushButton("清除框选")
    self._preview_clear_crop_btn.setObjectName("ocrPreviewClearSelectionButton")
    self._preview_clear_crop_btn.setStyleSheet(preview_button_style)
    self._preview_clear_crop_btn.setMinimumHeight(30)
    self._preview_clear_crop_btn.setToolTip("清除当前 OCR 识别区域框选，恢复使用完整正文区域")
    preview_nav_row.addWidget(self._preview_clear_crop_btn)

    # Keep the historical source ordering required by the preview contract,
    # while inserting the page counter between Previous and Next visually.
    self._preview_page_lbl = QLabel("0 / 0")
    self._preview_page_lbl.setMinimumWidth(54)
    self._preview_page_lbl.setAlignment(Qt.AlignCenter)
    self._preview_page_lbl.setStyleSheet(f"color: {INK}; font-size: 12px; font-weight: 700;")
    preview_nav_row.insertWidget(1, self._preview_page_lbl)

    cv.addLayout(preview_nav_row)

    self._preview_prev_shortcut = QShortcut(QKeySequence("Alt+Left"), self)
    self._preview_prev_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
    self._preview_prev_shortcut.activated.connect(self._show_previous_preview_page)
    self._preview_next_shortcut = QShortcut(QKeySequence("Alt+Right"), self)
    self._preview_next_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
    self._preview_next_shortcut.activated.connect(self._show_next_preview_page)

    self._preview = OCRCropPreview()
    self._preview_clear_crop_btn.clicked.connect(self._preview.clear_rect)
    self._preview_detector_box_check.toggled.connect(
        self._preview.set_detector_boxes_visible
    )
    self._preview_input_box_check.toggled.connect(
        self._preview.set_input_boxes_visible
    )
    self._preview.set_detector_boxes_visible(self._preview_detector_box_check.isChecked())
    self._preview.set_input_boxes_visible(self._preview_input_box_check.isChecked())
    canvas = QFrame()
    canvas.setObjectName("ocrCanvas")
    canvas.setStyleSheet(f"QFrame#ocrCanvas{{background:{TONAL};border:1px solid {BORDER};border-radius:16px;}}")
    canvas_layout = QVBoxLayout(canvas)
    canvas_layout.setContentsMargins(8, 8, 8, 8)
    canvas_layout.addWidget(self._preview, 1)
    cv.addWidget(canvas, 1)

    self._update_preview_navigation()

    top_row.addWidget(center, 2)

