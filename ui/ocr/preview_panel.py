from __future__ import annotations

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox, QSizePolicy, QFrame, QColorDialog, QMenu
from PySide6.QtCore import QSize, Qt, QSettings
from PySide6.QtGui import QColor, QIcon, QKeySequence, QPainter, QPen, QPixmap, QShortcut

from ui.common.styling import ACC, ACC_BG, ACC_TEXT, BG, BORDER, CARD, INK, MUTED, SURFACE, TONAL, TONAL_HOVER
from ui.theme.tokens import DISABLED_BG, DISABLED_FG
from ui.ocr.preview import DETECTOR_BOX_COLOR, INPUT_BOX_COLOR, OCRCropPreview


def _style_overlay_toggle(toggle: QCheckBox, color: str, line_style: Qt.PenStyle) -> None:
    toggle.setStyleSheet(
        f"QCheckBox {{ background: transparent; border: 1px solid transparent; "
        f"border-radius: 7px; padding: 3px 7px; spacing: 5px; font-size: 11px; color: {MUTED}; }}"
        f"QCheckBox:checked {{ background: {CARD}; border-color: {BORDER}; color: {INK}; }}"
        f"QCheckBox:hover {{ background: {TONAL_HOVER}; }}"
        f"QCheckBox:focus {{ border-color: {ACC}; }}"
    )
    swatch = QPixmap(32, 20)
    swatch.setDevicePixelRatio(2)
    swatch.fill(Qt.transparent)
    painter = QPainter(swatch)
    pen = QPen(QColor(color))
    pen.setWidth(2)
    pen.setStyle(line_style)
    pen.setCapStyle(Qt.FlatCap)
    if line_style == Qt.DashLine:
        pen.setDashPattern([2, 1.5])
    painter.setPen(pen)
    painter.drawLine(0, 5, 16, 5)
    painter.end()
    toggle.setIcon(QIcon(swatch))
    toggle.setIconSize(QSize(16, 10))


def build_ocr_preview_panel(tab, top_row):
    self = tab
    # ── 中间：大幅放大的识别区域预览（拖框选定，替代原来的百分比裁剪）───────
    center = QWidget()
    center.setObjectName("ocrPreviewPanel")
    center.setStyleSheet(f"QWidget#ocrPreviewPanel {{ background: {BG}; }}")
    cv = QVBoxLayout(center)
    # Avoid a wide blank gutter between the settings column and the preview;
    # this is especially costly in character-review mode on a narrow window.
    cv.setContentsMargins(10, 15, 12, 15)
    cv.setSpacing(10)

    toolbar = QFrame(center)
    toolbar.setObjectName("ocrPreviewToolbar")
    toolbar.setStyleSheet(
        f"QFrame#ocrPreviewToolbar {{ background: {SURFACE}; border: 1px solid {BORDER}; "
        "border-radius: 12px; }"
    )
    toolbar_layout = QVBoxLayout(toolbar)
    toolbar_layout.setContentsMargins(9, 8, 9, 8)
    toolbar_layout.setSpacing(5)
    preview_nav_row = QHBoxLayout()
    self._preview_nav_row = preview_nav_row
    preview_nav_row.setSpacing(6)
    toolbar_layout.addLayout(preview_nav_row)

    def preview_button_style(background, foreground):
        return (
            f"QPushButton {{ font-size: 12px; padding: 4px 10px; "
            f"border: 1px solid transparent; border-radius: 8px; "
            f"background: {background}; color: {foreground}; font-weight: 600; }}"
            f"QPushButton:hover {{ background: {TONAL_HOVER}; }}"
            f"QPushButton:pressed {{ background: {TONAL}; }}"
            f"QPushButton:focus {{ border-color: {ACC}; }}"
            f"QPushButton:disabled {{ background: {DISABLED_BG}; color: {DISABLED_FG}; }}"
        )

    navigation_style = preview_button_style(CARD, INK)

    self._preview_prev_btn = QPushButton("‹  上一页")
    self._preview_prev_btn.setObjectName("ocrPreviewPreviousButton")
    self._preview_prev_btn.setStyleSheet(navigation_style)
    self._preview_prev_btn.setMinimumHeight(30)
    self._preview_prev_btn.setVisible(True)
    self._preview_prev_btn.setToolTip("返回已经保留的上一张实时预览；OCR 运行中也可使用（快捷键：⌥←）")
    self._preview_prev_btn.clicked.connect(self._show_previous_preview_page)
    preview_nav_row.addWidget(self._preview_prev_btn)

    self._preview_page_lbl = QLabel("0 / 0")
    self._preview_page_lbl.setMinimumWidth(54)
    self._preview_page_lbl.setAlignment(Qt.AlignCenter)
    self._preview_page_lbl.setStyleSheet(
        f"color: {INK}; font-size: 12px; font-weight: 700; padding: 0 4px;"
    )
    preview_nav_row.addWidget(self._preview_page_lbl)

    self._preview_next_btn = QPushButton("下一页 →")
    self._preview_next_btn.setObjectName("ocrPreviewNextButton")
    self._preview_next_btn.setStyleSheet(navigation_style)
    self._preview_next_btn.setMinimumHeight(30)
    self._preview_next_btn.setVisible(True)
    self._preview_next_btn.setToolTip("前往已经保留的下一张实时预览；OCR 运行中也可使用（快捷键：⌥→）")
    self._preview_next_btn.clicked.connect(self._show_next_preview_page)
    preview_nav_row.addWidget(self._preview_next_btn)

    self._preview_clear_crop_btn = QPushButton("清除框选")
    self._preview_clear_crop_btn.setObjectName("ocrPreviewClearSelectionButton")
    self._preview_clear_crop_btn.setStyleSheet(preview_button_style("transparent", MUTED))
    self._preview_clear_crop_btn.setMinimumHeight(30)
    self._preview_clear_crop_btn.setToolTip("清除当前 OCR 识别区域框选，恢复使用完整正文区域")
    preview_nav_row.addWidget(self._preview_clear_crop_btn)
    preview_nav_row.addStretch(1)

    self._preview_columns_btn = QPushButton("预览分列")
    self._preview_columns_btn.setObjectName("ocrPreviewColumnsButton")
    self._preview_columns_btn.setStyleSheet(preview_button_style(ACC_BG, ACC_TEXT))
    self._preview_columns_btn.setMinimumHeight(30)
    self._preview_columns_btn.setToolTip("预览当前页的物理分列与实际单列 OCR 输入")
    self._preview_columns_btn.clicked.connect(self._preview_column_split)
    self._preview_columns_btn.setEnabled(True)
    preview_nav_row.addWidget(self._preview_columns_btn)

    display_row = QHBoxLayout()
    display_row.setSpacing(6)
    self._preview_filename_lbl = QLabel("当前图片：尚未载入")
    self._preview_filename_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px; padding-left: 3px;")
    self._preview_filename_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
    self._preview_filename_lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    display_row.addWidget(self._preview_filename_lbl, 1)

    self._preview_detector_box_check = QCheckBox("检测框")
    self._preview_detector_box_check.setObjectName("ocrPreviewDetectorBoxToggle")
    self._preview_detector_box_check.setChecked(True)
    self._preview_detector_box_check.setToolTip("显示或隐藏虚线检测框")

    self._preview_input_box_check = QCheckBox("输入框")
    self._preview_input_box_check.setObjectName("ocrPreviewInputBoxToggle")
    self._preview_input_box_check.setChecked(True)
    self._preview_input_box_check.setToolTip("显示或隐藏实线 OCR 输入框")
    display_row.addWidget(self._preview_detector_box_check)
    display_row.addWidget(self._preview_input_box_check)
    colors_button = QPushButton("框线颜色")
    colors_button.setStyleSheet(preview_button_style(TONAL, INK))
    colors_menu = QMenu(colors_button)
    detector_action = colors_menu.addAction("检测框颜色…")
    input_action = colors_menu.addAction("输入框颜色…")
    colors_menu.addSeparator()
    reset_action = colors_menu.addAction("恢复默认颜色")
    colors_button.setMenu(colors_menu)
    display_row.addWidget(colors_button)
    display_row.addSpacing(4)

    self._preview_enabled_cb = QCheckBox("实时预览")
    self._preview_enabled_cb.setChecked(True)
    self._preview_enabled_cb.setStyleSheet("font-size: 11px;")
    self._preview_enabled_cb.setToolTip(
        "可在 OCR 运行过程中随时关闭或重新开启。关闭后停止生成、保存和刷新新的预览图，"
        "OCR、分列、整句重识别和结果输出继续正常运行；开启时会保留本轮全部正文页的缩略预览。"
    )
    self._preview_enabled_cb.toggled.connect(self._on_live_preview_toggled)
    display_row.addWidget(self._preview_enabled_cb)
    toolbar_layout.addLayout(display_row)
    cv.addWidget(toolbar)

    self._preview_prev_shortcut = QShortcut(QKeySequence("Alt+Left"), self)
    self._preview_prev_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
    self._preview_prev_shortcut.activated.connect(self._show_previous_preview_page)
    self._preview_next_shortcut = QShortcut(QKeySequence("Alt+Right"), self)
    self._preview_next_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
    self._preview_next_shortcut.activated.connect(self._show_next_preview_page)

    self._preview = OCRCropPreview()
    color_settings = QSettings("NovelFormatter", "NovelFormatter")
    detector_key = "ocr/preview_detector_box_color"
    input_key = "ocr/preview_input_box_color"

    def refresh_overlay_colors():
        detector = QColor(color_settings.value(detector_key, DETECTOR_BOX_COLOR, type=str))
        input_box = QColor(color_settings.value(input_key, INPUT_BOX_COLOR, type=str))
        detector_color = detector.name() if detector.isValid() else DETECTOR_BOX_COLOR
        input_color = input_box.name() if input_box.isValid() else INPUT_BOX_COLOR
        self._preview.set_overlay_colors(detector_color, input_color)
        _style_overlay_toggle(self._preview_detector_box_check, detector_color, Qt.DashLine)
        _style_overlay_toggle(self._preview_input_box_check, input_color, Qt.SolidLine)

    def choose_overlay_color(key, current, title):
        color = QColorDialog.getColor(QColor(current), self, title)
        if color.isValid():
            color_settings.setValue(key, color.name())
            refresh_overlay_colors()

    def reset_overlay_colors():
        color_settings.remove(detector_key)
        color_settings.remove(input_key)
        refresh_overlay_colors()

    detector_action.triggered.connect(lambda: choose_overlay_color(
        detector_key, self._preview.detector_box_color, "选择检测框颜色",
    ))
    input_action.triggered.connect(lambda: choose_overlay_color(
        input_key, self._preview.input_box_color, "选择输入框颜色",
    ))
    reset_action.triggered.connect(reset_overlay_colors)
    refresh_overlay_colors()
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
    canvas.setStyleSheet(f"QFrame#ocrCanvas{{background:{SURFACE};border:1px solid {BORDER};border-radius:12px;}}")
    canvas_layout = QVBoxLayout(canvas)
    canvas_layout.setContentsMargins(8, 8, 8, 8)
    canvas_layout.addWidget(self._preview, 1)
    cv.addWidget(canvas, 1)

    self._update_preview_navigation()

    top_row.addWidget(center, 2)

