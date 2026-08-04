#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scan-cleanup settings and single-page preview dialog."""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from adapters.scan_preprocess import ScanPreprocessOptions, process_scan_page
from utils.session_temp import session_temp_registry


_SETTINGS_KEY = "scan_preprocess/options_v1"


class _PreviewPane(QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "QFrame { background: #F8FAFD; border: 1px solid #CFE2F8; border-radius: 8px; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        heading = QLabel(title)
        heading.setStyleSheet("font-weight: 700; color: #202733; border: none;")
        layout.addWidget(heading)
        self.image_label = QLabel("尚未生成预览")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(250, 320)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet(
            "background: #FFFFFF; color: #667085; border: 1px solid #D7E7F8; border-radius: 6px;"
        )
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(self.image_label)
        layout.addWidget(scroll, 1)
        self.caption = QLabel("")
        self.caption.setWordWrap(True)
        self.caption.setStyleSheet("color: #667085; font-size: 10px; border: none;")
        layout.addWidget(self.caption)

    def set_path(self, path: str, caption: str = "") -> None:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.image_label.setText("预览图读取失败")
            self.image_label.setPixmap(QPixmap())
        else:
            target = self.image_label.size()
            scaled = pixmap.scaled(
                max(180, target.width() - 12),
                max(260, target.height() - 12),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.image_label.setPixmap(scaled)
            self.image_label.setText("")
        self.caption.setText(caption)


class ScanPreprocessDialog(QDialog):
    """Configure scan processing without touching the original page files."""

    def __init__(
        self,
        preview_path: str,
        *,
        source_count: int,
        selected_only: bool,
        parent=None,
    ):
        super().__init__(parent)
        self._preview_path = str(preview_path)
        self._source_count = max(1, int(source_count))
        self._selected_only = bool(selected_only)
        self._preview_dir = session_temp_registry().make_dir("scan-preprocess-preview")
        self._latest_preview_paths: list[str] = []
        self.setWindowTitle("扫描件优化")
        self.resize(1080, 760)
        self.setMinimumSize(900, 620)
        self._build()
        self._load_settings()
        self._show_original()
        self._refresh_control_state()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        intro = QLabel(
            "只生成会话临时副本，不修改原图。处理完成后，页面管理、OCR预览、分列和多模型会统一读取同一批优化页面。"
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #475467; font-size: 11px;")
        root.addWidget(intro)

        body = QHBoxLayout()
        body.setSpacing(12)
        root.addLayout(body, 1)

        settings_panel = QWidget()
        settings_panel.setMaximumWidth(330)
        settings_layout = QVBoxLayout(settings_panel)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(8)

        geometry_group = QGroupBox("页面几何")
        geometry_layout = QVBoxLayout(geometry_group)
        self.auto_crop = QCheckBox("自动检测并裁掉桌面/黑边")
        self.perspective = QCheckBox("透视拉正（检测可靠时才执行）")
        self.deskew = QCheckBox("小角度自动纠偏")
        self.split_spread = QCheckBox("书籍双页自动拆分")
        geometry_layout.addWidget(self.auto_crop)
        geometry_layout.addWidget(self.perspective)
        geometry_layout.addWidget(self.deskew)
        geometry_layout.addWidget(self.split_spread)

        geometry_form = QFormLayout()
        self.margin = QDoubleSpinBox()
        self.margin.setRange(0.0, 5.0)
        self.margin.setDecimals(1)
        self.margin.setSingleStep(0.2)
        self.margin.setSuffix(" %")
        self.margin.setKeyboardTracking(False)
        geometry_form.addRow("保留边距", self.margin)
        self.spread_order = QComboBox()
        self.spread_order.addItem("日文书：右页 → 左页", "right_to_left")
        self.spread_order.addItem("横排书：左页 → 右页", "left_to_right")
        geometry_form.addRow("双页顺序", self.spread_order)
        geometry_layout.addLayout(geometry_form)
        settings_layout.addWidget(geometry_group)

        enhance_group = QGroupBox("漂白与去阴影")
        enhance_form = QFormLayout(enhance_group)
        self.enhancement = QComboBox()
        self.enhancement.addItem("不处理颜色/亮度", "none")
        self.enhancement.addItem("柔和漂白（轻小说推荐）", "soft")
        self.enhancement.addItem("强力文档（阴影较重）", "strong")
        self.enhancement.addItem("OCR专用灰度（不二值化）", "ocr")
        enhance_form.addRow("增强模式", self.enhancement)
        self.preserve_color = QCheckBox("保护彩色插图与印章")
        enhance_form.addRow("", self.preserve_color)
        settings_layout.addWidget(enhance_group)

        scope_text = (
            f"应用范围：当前选中的 {self._source_count} 页"
            if self._selected_only
            else f"应用范围：全部 {self._source_count} 页"
        )
        self.scope_label = QLabel(scope_text)
        self.scope_label.setWordWrap(True)
        self.scope_label.setStyleSheet(
            "background: #EAF3FF; color: #174A7E; border: 1px solid #CFE2F8; "
            "border-radius: 7px; padding: 8px; font-size: 11px;"
        )
        settings_layout.addWidget(self.scope_label)

        warning = QLabel(
            "双页拆分会改变页面数量，但每个新页面会继承原页的正文/封面/插图分类。"
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("color: #667085; font-size: 10px;")
        settings_layout.addWidget(warning)

        self.preview_button = QPushButton("更新当前页预览")
        self.preview_button.clicked.connect(self._generate_preview)
        settings_layout.addWidget(self.preview_button)
        settings_layout.addStretch(1)
        body.addWidget(settings_panel)

        preview_widget = QWidget()
        preview_layout = QHBoxLayout(preview_widget)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(10)
        self.original_pane = _PreviewPane("原图")
        self.processed_pane = _PreviewPane("优化预览")
        preview_layout.addWidget(self.original_pane, 1)
        preview_layout.addWidget(self.processed_pane, 1)
        body.addWidget(preview_widget, 1)

        footer = QHBoxLayout()
        self.restore_defaults_button = QPushButton("恢复推荐设置")
        self.restore_defaults_button.clicked.connect(self._restore_defaults)
        footer.addWidget(self.restore_defaults_button)
        footer.addStretch(1)
        cancel_button = QPushButton("取消")
        cancel_button.clicked.connect(self.reject)
        footer.addWidget(cancel_button)
        self.apply_button = QPushButton(
            f"应用到{'选中' if self._selected_only else '全部'} {self._source_count} 页"
        )
        self.apply_button.setDefault(True)
        self.apply_button.setStyleSheet(
            "QPushButton { background: #1677FF; color: white; border: none; "
            "border-radius: 7px; padding: 8px 16px; font-weight: 700; }"
            "QPushButton:hover { background: #0F6CE8; }"
        )
        self.apply_button.clicked.connect(self._accept_and_save)
        footer.addWidget(self.apply_button)
        root.addLayout(footer)

        self.split_spread.toggled.connect(self._refresh_control_state)
        self.enhancement.currentIndexChanged.connect(self._refresh_control_state)

    def _show_original(self) -> None:
        path = Path(self._preview_path)
        self.original_pane.set_path(
            str(path),
            f"{path.name} · 原文件保持不变",
        )

    def _restore_defaults(self) -> None:
        self._apply_options(ScanPreprocessOptions())
        self._refresh_control_state()

    def _load_settings(self) -> None:
        settings = QSettings("NovelFormatter", "NovelFormatter")
        raw = settings.value(_SETTINGS_KEY, "")
        options = ScanPreprocessOptions()
        if raw:
            try:
                options = ScanPreprocessOptions.from_dict(json.loads(str(raw)))
            except Exception:
                pass
        self._apply_options(options)

    def _apply_options(self, options: ScanPreprocessOptions) -> None:
        options = options.normalized()
        self.auto_crop.setChecked(options.auto_crop)
        self.perspective.setChecked(options.perspective)
        self.deskew.setChecked(options.deskew)
        self.split_spread.setChecked(options.split_spread)
        self.margin.setValue(options.crop_margin_percent)
        order_index = self.spread_order.findData(options.spread_order)
        self.spread_order.setCurrentIndex(max(0, order_index))
        enhancement_index = self.enhancement.findData(options.enhancement)
        self.enhancement.setCurrentIndex(max(0, enhancement_index))
        self.preserve_color.setChecked(options.preserve_color)

    def _refresh_control_state(self, *_args) -> None:
        self.spread_order.setEnabled(self.split_spread.isChecked())
        self.preserve_color.setEnabled(self.enhancement.currentData() not in {"none", "ocr"})

    def options(self) -> ScanPreprocessOptions:
        return ScanPreprocessOptions(
            auto_crop=self.auto_crop.isChecked(),
            perspective=self.perspective.isChecked(),
            deskew=self.deskew.isChecked(),
            split_spread=self.split_spread.isChecked(),
            spread_order=str(self.spread_order.currentData() or "right_to_left"),
            enhancement=str(self.enhancement.currentData() or "soft"),
            preserve_color=self.preserve_color.isChecked(),
            crop_margin_percent=float(self.margin.value()),
        ).normalized()

    def _generate_preview(self) -> None:
        self.preview_button.setEnabled(False)
        self.preview_button.setText("正在生成…")
        try:
            pages = process_scan_page(
                self._preview_path,
                self._preview_dir,
                self.options(),
                source_index=1,
            )
            self._latest_preview_paths = [page.output_path for page in pages]
            if not pages:
                raise RuntimeError("没有生成预览页面")
            first = pages[0]
            caption = Path(first.output_path).name
            if len(pages) > 1:
                caption += f" · 双页已拆为 {len(pages)} 页，右侧显示第 1 页"
            self.processed_pane.set_path(first.output_path, caption)
        except Exception as exc:
            QMessageBox.critical(self, "预览失败", str(exc))
        finally:
            self.preview_button.setText("更新当前页预览")
            self.preview_button.setEnabled(True)

    def _accept_and_save(self) -> None:
        settings = QSettings("NovelFormatter", "NovelFormatter")
        settings.setValue(_SETTINGS_KEY, json.dumps(self.options().to_dict(), ensure_ascii=False))
        self.accept()
