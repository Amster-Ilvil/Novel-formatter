#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Novel Formatter Studio — PySide6 GUI (完整版)
四个工作区：页面管理 / OCR 适配器 / Formatter Engine / EPUB Builder
macOS 简约风格，完整功能移植。

用法: python3 gui_pyside6.py
依赖: pip3 install PySide6 pillow
"""

from __future__ import annotations

VERSION = "1.0.0"

import sys
import os
import json
import copy
import re
import threading
import zipfile
from pathlib import Path
from collections import Counter
from functools import partial
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QStackedWidget,
    QPlainTextEdit, QTextEdit, QLabel, QPushButton, QComboBox, QLineEdit,
    QCheckBox, QSlider, QFileDialog, QMessageBox, QGroupBox,
    QFormLayout, QProgressBar, QFrame, QSizePolicy, QScrollArea,
    QGridLayout, QMenu, QToolButton, QTreeWidget, QTreeWidgetItem,
    QRadioButton, QButtonGroup, QSpacerItem, QRubberBand,
    QDialog, QListWidget, QListWidgetItem, QInputDialog,
)
from PySide6.QtCore import Qt, Signal, QObject, QSize, QTimer, QThread, QRect, QPoint
from PySide6.QtGui import (
    QFont, QColor, QPalette, QPixmap, QImage, QPainter, QIcon,
    QAction, QCursor,
)

from models.document import UnifiedDocument, Block, BlockType, TocEntry
from utils.paddle_importer import import_paddle_json, import_paddle_md
from models.format_profile import FormatProfile, FormatProfileStore

# ── 配色常量（macOS 系统色板）───────────────────────────────────────────────────

BG          = "#F5F5F7"   # 内容区背景（对应 macOS 窗口浅灰背景）
SIDEBAR_BG  = "#EBEBF0"   # 左侧边栏背景（比内容区略深，营造分层感）
CARD        = "#FFFFFF"   # 卡片/输入控件背景
INK         = "#1D1D1F"   # 主文字色（Apple 标准近黑）
MUTED       = "#86868B"   # 次要文字色（Apple 标准灰）
BORDER      = "#E5E5EA"   # 分隔线/边框
ACC         = "#0071E3"   # 强调色（Apple 蓝）
ACC_BG      = "#E8F1FE"   # 强调色浅底
DANGER      = "#FF3B30"   # 系统红
SUCCESS     = "#34C759"   # 系统绿

PAGE_TYPES = [
    ("cover",        "封面",     "#C0542F"),
    ("title_page",   "扉页",     "#6B4FA0"),
    ("color_illus",  "彩色插图", "#B5417C"),
    ("blank",        "空白页",   "#8C8B84"),
    ("toc_page",     "目录",     "#2C6FB5"),
    ("illustration", "插图",     "#127A56"),
    ("paragraph",    "正文",     "#4A3FA3"),
    ("afterword",    "后记",     "#93650D"),
    ("colophon",     "版权页",   "#5B5A54"),
    ("unknown",      "未分类",   "#AFAEA7"),
]
TYPE_LABEL = {t: l for t, l, _ in PAGE_TYPES}
TYPE_COLOR = {t: c for t, l, c in PAGE_TYPES}
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.heic', '.tif', '.tiff', '.bmp', '.gif'}

OCR_ADAPTERS = [
    ("apple_vision", "Apple Vision OCR", "macOS", "#4A3FA3",
     "Apple Live Text / Vision 框架，竖排识别优先", True),
    ("pdf_craft",    "pdf-craft",        "跨平台", "#127A56",
     "pdf-craft 开源工具输出，支持版面分析", False),
    ("paddle_ocr",   "PaddleOCR",        "跨平台", "#C0542F",
     "百度 PaddleOCR，坐标为像素值数组（首次使用会自动创建独立环境并下载模型）", True),
    ("google_vision", "Google Vision API", "云端",   "#93650D",
     "Google Cloud Vision JSON 响应", False),
]

FORMATTER_STEPS = [
    ("reading_order",     "阅读顺序",   "自动",     "GapTree列聚类，竖排右→左排序"),
    ("clean_metadata",    "清理模块",   "自动",     "删除页码、页眉、出版信息"),
    ("split_embedded_titles", "内嵌标题拆分", "规则", "拆出跟正文粘连在一起的章节标题"),
    ("strip_chapter_notes", "逐章备注剥离", "规则",  "删除每话附带的（前書）/（後書き）编辑备注"),
    ("merge_sentences",   "断句修复",   "规则+AI",  "合并OCR错误换行，恢复连续段落"),
    ("remove_duplicates", "重复删除",   "自动",     "删除OCR扫描产生的重复段落和对白"),
    ("fix_dash_artifacts","破折号修复", "自动",     "修复OCR把破折号误读成「/｜的错字"),
    ("dialogue_restore",  "对白恢复",   "规则",     "识别对白行，恢复单独换行排版"),
    ("restore_indents",   "缩进分节",   "规则",     "恢复段首缩进和分节符检测"),
    ("recover_ruby",      "振假名恢复", "规则",     "恢复｜漢字《よみ》ruby标注"),
    ("detect_chapters",   "章节识别",   "规则+AI",  "自动识别章节标题，生成TOC"),
    ("strip_boilerplate", "前后书剥离", "规则",     "删除版权声明/站点介绍等前书后书样板文字"),
    ("normalize_punct",   "标点规范",   "自动",     "统一省略号、破折号、括号格式"),
]
BADGE_COLOR = {"自动": "#127A56", "规则": "#2C6FB5", "规则+AI": "#4A3FA3"}

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def blend(hex_color: str, alpha: float = 0.15, bg: str = BG) -> str:
    def p(h):
        h = h.lstrip("#")
        return int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    fr, fg, fb = p(hex_color)
    br, bg2, bb = p(bg)
    return "#{:02X}{:02X}{:02X}".format(
        int(fr * alpha + br * (1 - alpha)),
        int(fg * alpha + bg2 * (1 - alpha)),
        int(fb * alpha + bb * (1 - alpha)))


def make_separator():
    sep = QFrame()
    sep.setFrameShape(QFrame.HLine)
    sep.setStyleSheet(f"background-color: {BORDER}; max-height: 1px; border: none;")
    return sep


def make_badge(text, color):
    lbl = QLabel(text)
    bg = blend(color, 0.15, CARD)
    lbl.setStyleSheet(
        f"background-color: {bg}; color: {color}; "
        f"border-radius: 4px; padding: 2px 8px; font-size: 11px; font-weight: 600;")
    return lbl


def accent_button(text: str, color: str = ACC, text_color: str = "white") -> QPushButton:
    """
    创建一个强制内联样式的主按钮。
    不依赖全局 QSS 级联（在某些 Qt/平台组合下，纯类型选择器的
    background-color 有时不会正确应用到深层嵌套的 QPushButton 上），
    直接把颜色写在按钮自身的 setStyleSheet 里，确保任何环境下都可见。
    """
    btn = QPushButton(text)
    btn.setStyleSheet(
        f"QPushButton {{ background-color: {color}; color: {text_color}; border: none; "
        f"border-radius: 8px; padding: 8px 16px; font-weight: 600; }}"
        f"QPushButton:hover {{ background-color: {blend(color, 0.85, '#000000')}; }}"
        f"QPushButton:disabled {{ background-color: #C7C7CC; color: #8E8E93; }}"
    )
    return btn


def danger_button(text: str) -> QPushButton:
    btn = QPushButton(text)
    btn.setProperty("danger", True)
    btn.setCursor(QCursor(Qt.PointingHandCursor))
    btn.setStyleSheet(
        f"QPushButton {{ background-color: {DANGER}; color: white; border: none; "
        f"border-radius: 8px; padding: 8px 16px; font-weight: 700; }}"
        f"QPushButton:hover {{ background-color: #E03128; }}"
        f"QPushButton:pressed {{ background-color: #B4231D; }}"
    )
    return btn



def show_error_dialog(parent: QWidget, title: str, traceback_text: str):
    """
    显示错误弹窗：标题栏显示最后一行（实际异常信息），
    完整 traceback 放进"详情"里（可展开、可复制），避免关键报错信息被截断看不到。
    """
    last_line = traceback_text.strip().splitlines()[-1] if traceback_text.strip() else "未知错误"
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Critical)
    box.setWindowTitle(title)
    box.setText(last_line)
    box.setDetailedText(traceback_text)
    box.setStandardButtons(QMessageBox.Ok)
    box.exec()


def wrap_in_card(owner: QWidget) -> QHBoxLayout:
    """
    在 owner (一个 QStackedWidget 页面) 上套一层参照原型的卡片外壳：
    浅褐色背景(BG) + 10px 边距 + 白色圆角卡片(CARD/BORDER)。
    返回卡片内部的 QHBoxLayout，供调用方继续 addWidget。
    """
    owner.setStyleSheet(f"background: {BG};")
    outer = QVBoxLayout(owner)
    outer.setContentsMargins(10, 10, 10, 10)
    card = QFrame()
    card.setStyleSheet(
        f"QFrame {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px; }}")
    outer.addWidget(card)
    inner = QHBoxLayout(card)
    inner.setContentsMargins(0, 0, 0, 0)
    inner.setSpacing(0)
    return inner


# ── 全局样式 ──────────────────────────────────────────────────────────────────

SIDEBAR_BTN_STYLE = f"""
QToolButton {{
    text-align: left;
    background: transparent;
    border: none;
    border-radius: 8px;
    padding: 9px 12px;
    font-size: 13px;
    color: {INK};
}}
QToolButton:hover {{ background: #DCDCE1; }}
QToolButton:checked {{ background: {ACC}; color: white; font-weight: 600; }}
"""

STYLE = f"""
QMainWindow {{ background-color: {BG}; }}
QWidget {{
    font-family: -apple-system, "SF Pro Text", "Helvetica Neue", "Hiragino Sans", sans-serif;
    font-size: 13px; color: {INK};
}}
QPushButton {{
    background-color: {ACC}; color: white; border: none;
    border-radius: 8px; padding: 8px 16px; font-weight: 500;
}}
QPushButton:hover {{ background-color: #0077ED; }}
QPushButton:pressed {{ background-color: #0058B8; }}
QPushButton:disabled {{ background-color: #D2D2D7; color: #98989D; }}
QPushButton[flat="true"] {{
    background-color: transparent; color: {ACC}; font-weight: normal;
}}
QPushButton[flat="true"]:hover {{ background-color: {ACC_BG}; border-radius: 6px; }}
QLineEdit, QComboBox {{
    background-color: {CARD}; border: 1px solid {BORDER};
    border-radius: 8px; padding: 6px 10px; selection-background-color: {ACC};
}}
QLineEdit:focus, QComboBox:focus {{ border: 1.5px solid {ACC}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QPlainTextEdit, QTextEdit {{
    background-color: {CARD}; border: 1px solid {BORDER};
    border-radius: 8px; padding: 10px;
    font-family: "SF Mono", "Menlo", monospace; font-size: 12px;
    selection-background-color: {ACC_BG};
}}
QGroupBox {{
    background-color: {CARD}; border: 1px solid {BORDER}; border-radius: 12px;
    margin-top: 14px; padding: 16px; font-weight: 600;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 14px; padding: 0 6px; color: {INK}; }}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{
    width: 17px; height: 17px; border-radius: 5px; border: 1.5px solid #C7C7CC;
    background: {CARD};
}}
QCheckBox::indicator:checked {{ background-color: {ACC}; border-color: {ACC}; }}
QRadioButton {{ spacing: 8px; }}
QRadioButton::indicator {{
    width: 16px; height: 16px; border-radius: 8px; border: 1.5px solid #C7C7CC;
    background: {CARD};
}}
QRadioButton::indicator:checked {{ background-color: {ACC}; border-color: {ACC}; }}
QProgressBar {{
    background-color: #E5E5EA; border: none; border-radius: 3px; height: 6px; text-align: center;
}}
QProgressBar::chunk {{ background-color: {ACC}; border-radius: 3px; }}
QSlider::groove:horizontal {{ background: #E5E5EA; height: 4px; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {ACC}; width: 16px; height: 16px; margin: -6px 0; border-radius: 8px;
}}
QTreeWidget {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px;
    font-size: 12px; padding: 4px;
}}
QTreeWidget::item {{ padding: 4px 2px; border-radius: 6px; }}
QTreeWidget::item:selected {{ background: {ACC_BG}; color: {INK}; }}
QScrollArea {{ background: transparent; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #C7C7CC; border-radius: 5px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: #ADADB3; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QToolTip {{
    background-color: {INK}; color: white; border: none;
    border-radius: 6px; padding: 6px 10px; font-size: 12px;
}}
QMessageBox QPushButton, QDialogButtonBox QPushButton {{
    background-color: {ACC}; color: white; border: none;
    border-radius: 6px; padding: 6px 14px; font-weight: 600; min-width: 64px;
}}
QMessageBox QPushButton:hover, QDialogButtonBox QPushButton:hover {{ background-color: #0077ED; }}
QMessageBox QPushButton:pressed, QDialogButtonBox QPushButton:pressed {{ background-color: #0058B8; }}
"""


# ══════════════════════════════════════════════════════════════════════════════
#  信号桥
# ══════════════════════════════════════════════════════════════════════════════

class WorkerSignals(QObject):
    finished = Signal(object)
    error = Signal(str)
    log = Signal(str)
    progress = Signal(int, int)
    current_image_data = Signal(QImage)




# ══════════════════════════════════════════════════════════════════════════════
#  Tab 1 — 页面管理器
# ══════════════════════════════════════════════════════════════════════════════

class PageManagerTab(QWidget):
    pages_loaded = Signal(list)
    go_ocr = Signal()
    types_changed = Signal()   # 页面分类被改动（标注/删除），OCR 页据此刷新参照图

    def __init__(self, parent=None):
        super().__init__(parent)
        self.page_images: list[Path] = []
        self.page_overrides: dict[int, str] = {}
        # 哪些页的分类还只是"导入时自动给的建议"，用户没有亲自确认过——
        # 跟 page_overrides 分开跟踪，因为下游 OCR 的"跳过识别"逻辑只应该
        # 信任真人确认过的标注，不能被这里图省事打的默认建议误伤（见
        # _finish_load 的详细说明）。
        self._auto_suggested: set[int] = set()
        self.selected_pages: set[int] = set()
        self.thumb_cache: dict[str, QPixmap] = {}
        self._current_filter = "all"
        self._last_loaded_raw_inputs: list[str] | None = None
        self._build()

    def _build(self):
        root_layout = wrap_in_card(self)

        # ── 左侧：类型筛选列表 ────────────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(180)
        left.setStyleSheet(f"background: {CARD};")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 12, 0, 12)

        hdr = QLabel("页面类型")
        hdr.setStyleSheet(f"color: {MUTED}; font-size: 11px; font-weight: 600; padding-left: 14px;")
        left_layout.addWidget(hdr)
        sub = QLabel("点击筛选页面")
        sub.setStyleSheet(f"color: #B7B6AF; font-size: 10px; padding-left: 14px; padding-bottom: 8px;")
        left_layout.addWidget(sub)

        self._filter_btns: dict[str, QPushButton] = {}
        self._filter_counts: dict[str, QLabel] = {}
        for ttype, label, color in [("all", "全部页面", "#333")] + PAGE_TYPES:
            row = QWidget()
            row.setStyleSheet(f"background: {CARD};")
            row.setCursor(QCursor(Qt.PointingHandCursor))
            rl = QHBoxLayout(row)
            rl.setContentsMargins(14, 4, 8, 4)
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {color}; font-size: 10px;")
            dot.setFixedWidth(16)
            rl.addWidget(dot)
            btn = QPushButton(label)
            btn.setFlat(True)
            btn.setStyleSheet(f"background: transparent; color: {INK}; text-align: left; padding: 4px; font-size: 12px; border: none;")
            btn.clicked.connect(partial(self._set_filter, ttype))
            rl.addWidget(btn, 1)
            cnt = QLabel("0")
            cnt.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
            cnt.setFixedWidth(30)
            cnt.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            rl.addWidget(cnt)
            left_layout.addWidget(row)
            self._filter_btns[ttype] = btn
            self._filter_counts[ttype] = cnt

        left_layout.addStretch()
        left_layout.addWidget(make_separator())

        ocr_btn = accent_button("开始 OCR")
        ocr_btn.setStyleSheet(ocr_btn.styleSheet() + "QPushButton { margin: 4px 10px 8px 10px; }")
        ocr_btn.clicked.connect(self.go_ocr.emit)
        left_layout.addWidget(ocr_btn)

        root_layout.addWidget(left)

        # ── 右侧主区域 ────────────────────────────────────────────────────────
        right = QWidget()
        right.setMinimumWidth(760)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # 顶部工具栏
        toolbar = QWidget()
        toolbar.setStyleSheet(f"background: {CARD};")
        toolbar.setFixedHeight(48)
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(14, 0, 14, 0)
        self._file_icon = QLabel("📄")
        self._file_icon.setStyleSheet("font-size: 16px;")
        tb_layout.addWidget(self._file_icon)
        self._file_lbl = QLabel("未打开文件")
        self._file_lbl.setStyleSheet(f"font-weight: bold; font-size: 13px;")
        tb_layout.addWidget(self._file_lbl)
        self._count_lbl = QLabel("")
        self._count_lbl.setStyleSheet(f"color: {MUTED}; font-size: 12px; padding-left: 8px;")
        tb_layout.addWidget(self._count_lbl)
        tb_layout.addStretch()

        open_btn = accent_button("打开")
        open_btn.clicked.connect(self._open_menu)
        tb_layout.addWidget(open_btn)
        right_layout.addWidget(toolbar)

        self._prog = QProgressBar()
        self._prog.setVisible(False)
        right_layout.addWidget(self._prog)

        right_layout.addWidget(make_separator())

        # 批量标签栏
        tagbar = QWidget()
        tagbar.setFixedHeight(44)
        tagbar.setStyleSheet(f"background: #FAFAF7;")
        tag_layout = QHBoxLayout(tagbar)
        tag_layout.setContentsMargins(10, 0, 10, 0)
        self._sel_lbl = QLabel("选中 0 页 · 标记为：")
        self._sel_lbl.setStyleSheet(f"color: {MUTED}; font-size: 12px;")
        tag_layout.addWidget(self._sel_lbl)
        for ttype, label, color in PAGE_TYPES:
            b = QPushButton(label)
            bg = blend(color, 0.15, "#FAFAF7")
            b.setStyleSheet(
                f"background: {bg}; color: {color}; border: none; border-radius: 4px; "
                f"padding: 4px 8px; font-size: 11px;")
            b.setCursor(QCursor(Qt.PointingHandCursor))
            b.clicked.connect(partial(self._batch_tag, ttype))
            tag_layout.addWidget(b)
        tag_layout.addStretch()
        sel_all = QPushButton("全选")
        sel_all.setProperty("flat", True)
        sel_all.clicked.connect(self._select_all)
        tag_layout.addWidget(sel_all)
        clr = QPushButton("取消选择")
        clr.setProperty("flat", True)
        clr.clicked.connect(self._clear_sel)
        tag_layout.addWidget(clr)
        del_btn = QPushButton("🗑 删除选中")
        del_btn.setStyleSheet(
            f"background: transparent; color: {DANGER}; border: none; "
            f"border-radius: 6px; padding: 6px 10px;")
        del_btn.setCursor(QCursor(Qt.PointingHandCursor))
        del_btn.clicked.connect(self._delete_selected)
        tag_layout.addWidget(del_btn)
        right_layout.addWidget(tagbar)

        right_layout.addWidget(make_separator())

        # 缩略图网格（滚动区域）
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"background: {CARD}; border: none;")
        self._grid_widget = QWidget()
        self._grid_widget.setStyleSheet(f"background: {CARD};")
        self._grid_layout = QGridLayout(self._grid_widget)
        self._grid_layout.setSpacing(8)
        self._grid_layout.setContentsMargins(12, 12, 12, 12)
        scroll.setWidget(self._grid_widget)
        right_layout.addWidget(scroll, 1)

        # 拉框多选（框选空白区域拖出一个矩形，与矩形相交的缩略图都会被选中）
        self._rubber_band = QRubberBand(QRubberBand.Rectangle, self._grid_widget)
        self._rubber_origin: Optional[QPoint] = None
        self._grid_widget.mousePressEvent = self._grid_mouse_press
        self._grid_widget.mouseMoveEvent = self._grid_mouse_move
        self._grid_widget.mouseReleaseEvent = self._grid_mouse_release

        # 空状态提示
        self._empty_label = QLabel("打开图片文件夹 / PDF / 单张图片开始\n\n支持 PNG · JPG · HEIC · TIFF · PDF")
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet(f"color: {MUTED}; font-size: 14px; padding: 60px;")
        self._grid_layout.addWidget(self._empty_label, 0, 0, 1, 6)

        # 底部统计条
        stat_bar = QWidget()
        stat_bar.setFixedHeight(28)
        stat_bar.setStyleSheet(f"background: #FAFAF7; border-top: 1px solid {BORDER};")
        stat_layout = QHBoxLayout(stat_bar)
        stat_layout.setContentsMargins(14, 0, 14, 0)
        self._stat_label = QLabel("")
        self._stat_label.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        stat_layout.addWidget(self._stat_label)
        right_layout.addWidget(stat_bar)

        root_layout.addWidget(right, 1)

    def _open_menu(self):
        menu = QMenu(self)
        menu.addAction("打开文件夹...", self._open_folder)
        menu.addAction("打开图片/PDF文件...", self._open_files)
        menu.exec(QCursor.pos())

    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if folder:
            self._load_inputs([folder], Path(folder).name)

    def _open_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择图片或 PDF（可多选）", "",
            "图片和PDF (*.png *.jpg *.jpeg *.heic *.tif *.tiff *.bmp *.gif *.pdf);;所有文件 (*)")
        if paths:
            name = Path(paths[0]).name if len(paths) == 1 else f"{len(paths)} 个文件"
            self._load_inputs(paths, name)

    def _load_inputs(self, raw_paths, display_name):
        self._last_loaded_raw_inputs = list(raw_paths)
        self._file_lbl.setText(display_name)
        self._prog.setVisible(True)
        self._prog.setRange(0, 0)

        def worker():
            try:
                from adapters.pdf_input import expand_inputs, natural_sort_key
                work_dir = raw_paths[0] if Path(raw_paths[0]).is_dir() else str(Path(raw_paths[0]).parent)
                images = expand_inputs(raw_paths, work_dir=work_dir)
                images = sorted(set(images), key=natural_sort_key)
                signals.finished.emit(images)
            except Exception as e:
                signals.error.emit(str(e))

        # 绑定在 self 上（而不是局部变量）：Qt 跨线程信号投递是异步的，
        # 工作线程结束后如果没有任何 Python 引用持有 WorkerSignals，
        # 垃圾回收可能在主线程处理排队事件之前就把它回收掉，导致野指针崩溃。
        signals = self._load_signals = WorkerSignals()
        signals.finished.connect(self._finish_load)
        signals.error.connect(lambda msg: QMessageBox.critical(self, "加载失败", msg))
        threading.Thread(target=worker, daemon=True).start()

    def _finish_load(self, images):
        self._prog.setVisible(False)
        if not images:
            QMessageBox.warning(self, "无图片", "未找到可用的图片")
            return
        self.page_images = [Path(p) for p in images]
        self.page_overrides.clear()
        # 默认全部当正文页——不再猜第一页=封面/最后一页=版权页。这个"聪明"
        # 默认之前会导致：只要没人手动点开确认，第一页/最后一页在 OCR 阶段
        # 会被当成"已标注非正文"直接跳过识别（哪怕它们其实是正文），单张图
        # 测试、或者首页本来就是正文的书会因此识别不出任何东西。用户明确要求
        # 全部默认正文，需要封面/插图/版权页的话在下面网格里手动右键改。
        n = len(self.page_images)
        self._auto_suggested = set(range(1, n + 1))
        for page_no in range(1, n + 1):
            self.page_overrides[page_no] = "paragraph"
        self.selected_pages.clear()
        self.thumb_cache.clear()
        self._count_lbl.setText(f"{len(images)} 页")
        self._render()
        self.pages_loaded.emit(images)

    def _ptype(self, page_no):
        return self.page_overrides.get(page_no, "unknown")

    def _render(self):
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        pages = self.page_images
        filt = self._current_filter
        if filt != "all":
            pages = [p for i, p in enumerate(pages) if self._ptype(i + 1) == filt]

        if not pages:
            empty = QLabel("（此类型暂无页面）" if self.page_images else
                          "打开图片文件夹 / PDF / 单张图片开始\n\n支持 PNG · JPG · HEIC · TIFF · PDF")
            empty.setAlignment(Qt.AlignCenter)
            empty.setStyleSheet(f"color: {MUTED}; font-size: 13px; padding: 60px;")
            self._grid_layout.addWidget(empty, 0, 0, 1, 6)
            self._update_counts()
            return

        COLS = 6
        W, H = 108, 140

        for idx, path in enumerate(pages):
            i_orig = self.page_images.index(path)
            page_no = i_orig + 1
            ptype = self._ptype(page_no)
            color = TYPE_COLOR.get(ptype, "#AAA")
            label = TYPE_LABEL.get(ptype, "?")
            selected = page_no in self.selected_pages

            col, row = idx % COLS, idx // COLS

            cell = QWidget()
            cell.setFixedSize(W + 10, H + 30)
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(2, 2, 2, 2)
            cell_layout.setSpacing(2)

            # 缩略图区域
            img_frame = QLabel()
            img_frame.setFixedSize(W, H - 24)
            img_frame.setAlignment(Qt.AlignCenter)
            border_color = ACC if selected else BORDER
            border_w = 3 if selected else 1
            img_frame.setStyleSheet(
                f"background: #EAEAE4; border: {border_w}px solid {border_color}; border-radius: 4px;")

            if HAS_PIL and path.exists():
                pix = self._get_thumb(path, W - 4, H - 28)
                if pix:
                    img_frame.setPixmap(pix)
            else:
                img_frame.setText(f"第 {page_no} 页")
                img_frame.setStyleSheet(img_frame.styleSheet() + f"color: #666; font-size: 12px;")

            cell_layout.addWidget(img_frame)

            # 类型标签
            tag = QLabel(label)
            tag.setFixedHeight(22)
            tag.setAlignment(Qt.AlignCenter)
            tag.setStyleSheet(
                f"background: {color}; color: white; border-radius: 3px; font-size: 11px;")
            cell_layout.addWidget(tag)

            # 页码
            page_lbl = QLabel(f"第 {page_no} 页")
            page_lbl.setAlignment(Qt.AlignCenter)
            page_lbl.setStyleSheet(f"color: {MUTED}; font-size: 10px;")
            cell_layout.addWidget(page_lbl)

            cell.setProperty("page_no", page_no)
            cell.mousePressEvent = partial(self._on_thumb_click, page_no)
            cell.setContextMenuPolicy(Qt.CustomContextMenu)
            cell.customContextMenuRequested.connect(partial(self._on_thumb_context, page_no))

            self._grid_layout.addWidget(cell, row, col)

        self._update_counts()
        self._sel_lbl.setText(f"选中 {len(self.selected_pages)} 页 · 标记为：")
        self._update_stat_bar()

    def _get_thumb(self, path, w, h):
        key = str(path)
        if key in self.thumb_cache:
            return self.thumb_cache[key]
        try:
            img = PILImage.open(str(path))
            img.thumbnail((w, h), PILImage.LANCZOS)
            if img.mode != "RGB":
                img = img.convert("RGB")
            data = img.tobytes()
            qimg = QImage(data, img.width, img.height, img.width * 3, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.thumb_cache[key] = pix
            return pix
        except Exception:
            return None

    def _on_thumb_click(self, page_no, event):
        if event.button() == Qt.LeftButton:
            if page_no in self.selected_pages:
                self.selected_pages.discard(page_no)
            else:
                self.selected_pages.add(page_no)
            self._render()

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
                self._render()
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
        self._render()

    def _on_thumb_context(self, page_no, pos):
        if page_no not in self.selected_pages:
            self.selected_pages = {page_no}
        menu = QMenu(self)
        menu.addAction(f"第 {page_no} 页 — 设置类型").setEnabled(False)
        menu.addSeparator()
        for ttype, label, color in PAGE_TYPES:
            action = menu.addAction(f"  {label}")
            action.triggered.connect(partial(self._batch_tag, ttype))
        menu.addSeparator()
        menu.addAction("取消选择", self._clear_sel)
        menu.addAction("🗑 删除选中页面", self._delete_selected)
        menu.exec(QCursor.pos())

    def _delete_selected(self):
        if not self.selected_pages:
            QMessageBox.information(self, "提示", "请先点选要删除的页面（可多选，或拖框多选）")
            return
        reply = QMessageBox.question(
            self, "删除页面",
            f"确定要从当前导入中移除选中的 {len(self.selected_pages)} 页吗？\n"
            "（只是从这次处理里去掉，不会删除原始图片文件）",
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

        self.page_images = [self.page_images[i] for i in keep_indices]
        self.page_overrides = new_overrides
        self._auto_suggested = new_auto_suggested
        self.selected_pages.clear()
        self._count_lbl.setText(f"{len(self.page_images)} 页")
        self._render()
        self.types_changed.emit()

    def _batch_tag(self, ttype):
        if not self.selected_pages:
            QMessageBox.information(self, "提示", "请先点选页面（可多选）")
            return
        for p in self.selected_pages:
            self.page_overrides[p] = ttype
            # 用户亲自选中并打了标——从"自动建议"升级成"确认过"，之后会被
            # 正常转发给 OCR（包括跳过识别，如果标的是非正文类型）。
            self._auto_suggested.discard(p)
        self.selected_pages.clear()
        self._render()
        self.types_changed.emit()

    def _select_all(self):
        self.selected_pages = set(range(1, len(self.page_images) + 1))
        self._render()

    def _clear_sel(self):
        self.selected_pages.clear()
        self._render()

    def _set_filter(self, ttype):
        self._current_filter = ttype
        for t, btn in self._filter_btns.items():
            if t == ttype:
                btn.setStyleSheet(f"background: {ACC_BG}; color: {ACC}; text-align: left; padding: 4px; font-size: 12px; font-weight: bold; border: none; border-radius: 4px;")
            else:
                btn.setStyleSheet(f"background: transparent; color: {INK}; text-align: left; padding: 4px; font-size: 12px; border: none;")
        self._render()

    def _update_counts(self):
        counts = Counter(self._ptype(i + 1) for i in range(len(self.page_images)))
        total = len(self.page_images)
        for t, lbl in self._filter_counts.items():
            lbl.setText(str(total if t == "all" else counts.get(t, 0)))

    def _update_stat_bar(self):
        counts = Counter(self._ptype(i + 1) for i in range(len(self.page_images)))
        parts = []
        for t, l, c in PAGE_TYPES:
            n = counts.get(t, 0)
            if n > 0:
                parts.append(f"● {l} {n}页")
        self._stat_label.setText("  ".join(parts) if parts else "")


# ══════════════════════════════════════════════════════════════════════════════
#  Tab 2 — OCR 适配器
# ══════════════════════════════════════════════════════════════════════════════

class OCRCropPreview(QLabel):
    """
    显示当前图片，支持拖框选定识别区域（替代原来"顶部/底部百分比"的裁剪方式）。
    框选矩形以"相对当前显示图片的归一化坐标 [0,1]"保存，与图片实际分辨率无关；
    OCR 运行时会把每一页的当前处理图片实时换到这里显示，方便对照框选范围。

    交互方式和 PageManagerTab 里缩略图拉框多选一致：QRubberBand 走拖拽过程的
    视觉反馈，松手后用一个半透明 QFrame 常驻显示已确定的框选区域。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            f"background: #1B1B1D; color: {MUTED}; border: 1px solid {BORDER}; border-radius: 6px;")
        self.setText("选择输入后在此显示图片，可拖框选定识别区域")
        self.setWordWrap(True)
        self.setCursor(QCursor(Qt.CrossCursor))

        self._orig_pixmap: QPixmap | None = None
        self._rect_norm: tuple[float, float, float, float] | None = None
        self._rubber_origin: QPoint | None = None
        self._rubber_band = QRubberBand(QRubberBand.Rectangle, self)

        self._overlay = QFrame(self)
        self._overlay.setStyleSheet(f"background: rgba(74,99,211,60); border: 2px solid {ACC};")
        # 让点击穿透到底下的 QLabel，否则已经框选过一次之后，想在旧框内重新拖拽
        # 会被这个纯展示用的覆盖层"吃掉"鼠标事件，看起来就像只能拖一次改不了。
        self._overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._overlay.hide()

    def set_image(self, path: str):
        pm = QPixmap(path)
        if pm.isNull():
            return
        self._orig_pixmap = pm
        self.setText("")
        self._rescale()
        self._update_overlay_geometry()

    def set_image_data(self, qimage: QImage):
        """
        跟 set_image 一样，只是接收一个已经在后台线程里解码/缩小好的 QImage
        （OCR 运行时实时预览用这个），主线程这边只做一次便宜的 QPixmap 转换，
        不用再读一次原图文件。
        """
        if qimage.isNull():
            return
        self._orig_pixmap = QPixmap.fromImage(qimage)
        self.setText("")
        self._rescale()
        self._update_overlay_geometry()

    def clear_rect(self):
        self._rect_norm = None
        self._overlay.hide()

    def get_crop_rect(self):
        return self._rect_norm

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()
        self._update_overlay_geometry()

    def _rescale(self):
        if self._orig_pixmap is None:
            return
        scaled = self._orig_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)

    def _display_geometry(self):
        """当前缩放后图片在 label 内实际绘制的区域 (x, y, w, h)（居中）"""
        pm = self.pixmap()
        if pm is None or pm.isNull():
            return None
        w, h = pm.width(), pm.height()
        return (self.width() - w) / 2, (self.height() - h) / 2, w, h

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton or self._orig_pixmap is None:
            return
        self._rubber_origin = event.pos()
        self._rubber_band.setGeometry(QRect(self._rubber_origin, event.pos()).normalized())
        self._rubber_band.show()

    def mouseMoveEvent(self, event):
        if self._rubber_origin is None:
            return
        self._rubber_band.setGeometry(QRect(self._rubber_origin, event.pos()).normalized())

    def mouseReleaseEvent(self, event):
        if self._rubber_origin is None:
            return
        rect = QRect(self._rubber_origin, event.pos()).normalized()
        self._rubber_band.hide()
        self._rubber_origin = None

        geo = self._display_geometry()
        if not geo or rect.width() < 6 or rect.height() < 6:
            return
        dx, dy, dw, dh = geo
        x0 = max(dx, min(rect.left(), dx + dw))
        x1 = max(dx, min(rect.right(), dx + dw))
        y0 = max(dy, min(rect.top(), dy + dh))
        y1 = max(dy, min(rect.bottom(), dy + dh))
        if x1 - x0 < 6 or y1 - y0 < 6:
            return
        self._rect_norm = ((x0 - dx) / dw, (y0 - dy) / dh, (x1 - dx) / dw, (y1 - dy) / dh)
        self._update_overlay_geometry()

    def _update_overlay_geometry(self):
        if not self._rect_norm:
            self._overlay.hide()
            return
        geo = self._display_geometry()
        if not geo:
            self._overlay.hide()
            return
        dx, dy, dw, dh = geo
        x0, y0, x1, y1 = self._rect_norm
        self._overlay.setGeometry(QRect(
            int(dx + x0 * dw), int(dy + y0 * dh),
            int((x1 - x0) * dw), int((y1 - y0) * dh),
        ))
        self._overlay.show()
        self._overlay.raise_()


class OCRTab(QWidget):
    ocr_done = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pending_inputs: list[str] = []
        self._cancel_event = threading.Event()
        self._active_adapter = "apple_vision"
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
        left_container = QWidget()
        left_container.setFixedWidth(340)
        left_container.setStyleSheet(f"background: {CARD}; border-right: 1px solid {BORDER};")
        left_outer = QVBoxLayout(left_container)
        left_outer.setContentsMargins(0, 0, 0, 0)
        left_outer.setSpacing(0)

        left = QWidget()
        left.setStyleSheet(f"background: {CARD};")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 14, 0, 14)

        hdr = QLabel("OCR 适配器")
        hdr.setStyleSheet(f"font-size: 14px; font-weight: bold; padding-left: 14px;")
        ll.addWidget(hdr)
        sub = QLabel("选择识别引擎")
        sub.setStyleSheet(f"color: {MUTED}; font-size: 11px; padding-left: 14px; padding-bottom: 8px;")
        ll.addWidget(sub)

        self._adapter_cards: dict[str, QWidget] = {}
        for aid, name, badge_text, color, desc, enabled in OCR_ADAPTERS:
            card = QWidget()
            card.setCursor(QCursor(Qt.PointingHandCursor))
            card.setStyleSheet(
                f"background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px; margin: 4px 10px;")
            cl = QVBoxLayout(card)
            cl.setContentsMargins(10, 8, 10, 8)
            top_line = QHBoxLayout()
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {color}; font-size: 10px;")
            dot.setFixedWidth(14)
            top_line.addWidget(dot)
            nlbl = QLabel(name)
            nlbl.setStyleSheet(f"font-weight: bold; font-size: 12px;")
            top_line.addWidget(nlbl)
            top_line.addStretch()
            top_line.addWidget(make_badge(badge_text, color if enabled else "#8C8B84"))
            cl.addLayout(top_line)
            dl = QLabel(desc)
            dl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
            dl.setWordWrap(True)
            cl.addWidget(dl)
            if not enabled:
                na = QLabel("即将支持")
                na.setStyleSheet(f"color: #B7B6AF; font-size: 10px; font-style: italic;")
                na.setAlignment(Qt.AlignRight)
                cl.addWidget(na)
            card.mousePressEvent = partial(self._select_adapter, aid, enabled)
            ll.addWidget(card)
            self._adapter_cards[aid] = card

        ll.addWidget(make_separator())

        # Apple Vision 识别方式（只在选中 apple_vision 适配器时显示）——
        # "快捷指令" App 是最早的实现，ocrmac（pip install ocrmac）不依赖
        # 那层中间 App，直接调 Vision/VisionKit 框架，还能拿到坐标/置信度。
        self._vision_backend_widget = QWidget()
        vbw = QVBoxLayout(self._vision_backend_widget)
        vbw.setContentsMargins(10, 0, 10, 0)
        vbw.setSpacing(2)
        vb_lbl = QLabel("识别方式")
        vb_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        vbw.addWidget(vb_lbl)
        self._vision_backend_combo = QComboBox()
        self._vision_backend_combo.addItem("自动选择（推荐）", "auto")
        self._vision_backend_combo.addItem("ocrmac · accurate 模式（有坐标/置信度）", "ocrmac-vision")
        self._vision_backend_combo.addItem("ocrmac · livetext（竖排文字实验性更好，无置信度）", "ocrmac-livetext")
        self._vision_backend_combo.addItem("快捷指令 App（原方案）", "shortcut")
        self._vision_backend_combo.setCurrentIndex(3)  # 默认原方案（快捷指令 App）
        self._vision_backend_combo.currentIndexChanged.connect(self._on_vision_backend_changed)
        vbw.addWidget(self._vision_backend_combo)
        self._vision_vertical_check = QCheckBox("竖排文字（右→左排序）")
        self._vision_vertical_check.setChecked(False)
        vbw.addWidget(self._vision_vertical_check)
        vb_hint = QLabel("")
        vb_hint.setStyleSheet(f"color: #B7B6AF; font-size: 10px;")
        vb_hint.setWordWrap(True)
        vbw.addWidget(vb_hint)
        self._vision_backend_hint = vb_hint
        ll.addWidget(self._vision_backend_widget)

        # 快捷指令名称（只在识别方式选了"快捷指令 App"时才有意义）
        self._shortcut_widget = QWidget()
        sc_col = QVBoxLayout(self._shortcut_widget)
        sc_col.setContentsMargins(10, 4, 10, 0)
        sc_col.setSpacing(2)
        sc_lbl = QLabel("快捷指令名称")
        sc_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        sc_col.addWidget(sc_lbl)
        self._shortcut_edit = QLineEdit("ExtractText")
        sc_col.addWidget(self._shortcut_edit)
        ll.addWidget(self._shortcut_widget)
        self._on_vision_backend_changed()

        # PaddleOCR 模型选择（只在选中 paddle_ocr 适配器时显示）
        self._paddle_model_widget = QWidget()
        self._paddle_model_widget.setVisible(False)
        pmv = QVBoxLayout(self._paddle_model_widget)
        pmv.setContentsMargins(10, 8, 10, 0)
        pmv.setSpacing(2)
        pm_lbl = QLabel("Paddle 模型")
        pm_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        pmv.addWidget(pm_lbl)
        self._paddle_model_combo = QComboBox()
        self._paddle_model_combo.addItem("PP-OCR（纯文字识别，最快最省资源）", "ocr")
        self._paddle_model_combo.addItem("PP-StructureV3（版面分析，较占内存）", "structure")
        self._paddle_model_combo.addItem("PaddleOCR-VL-1.6（视觉语言模型，识别质量更好，首次用下载约2GB且很占内存）", "vl")
        pmv.addWidget(self._paddle_model_combo)
        pm_hint = QLabel("StructureV3/VL 同时加载多个模型，内存有限的机器上可能会被系统直接杀掉进程；不确定的话先用 PP-OCR")
        pm_hint.setStyleSheet(f"color: #B7B6AF; font-size: 10px;")
        pm_hint.setWordWrap(True)
        pmv.addWidget(pm_hint)
        ll.addWidget(self._paddle_model_widget)

        # 输入选择（纵向排列，避免文字被挤压截断）
        inp_lbl = QLabel("输入（图片 / PDF）")
        inp_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px; padding: 8px 0 4px 14px;")
        ll.addWidget(inp_lbl)
        btn_col = QVBoxLayout()
        btn_col.setContentsMargins(10, 0, 10, 0)
        btn_col.setSpacing(4)
        fb = QPushButton("📂  选择文件夹")
        fb.setStyleSheet(f"background: {CARD}; color: {INK}; border: 1px solid {BORDER}; text-align: left; padding: 8px 12px;")
        fb.clicked.connect(self._pick_folder)
        btn_col.addWidget(fb)
        ib = QPushButton("🖼  选择图片 / PDF")
        ib.setStyleSheet(f"background: {CARD}; color: {INK}; border: 1px solid {BORDER}; text-align: left; padding: 8px 12px;")
        ib.clicked.connect(self._pick_files)
        btn_col.addWidget(ib)
        ll.addLayout(btn_col)
        hint = QLabel("DOCX（已识别文档）请到 Formatter 页导入")
        hint.setStyleSheet(f"color: #B7B6AF; font-size: 10px; padding: 4px 14px 0;")
        hint.setWordWrap(True)
        ll.addWidget(hint)

        self._input_lbl = QLabel("（尚未选择输入）")
        self._input_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px; padding-left: 14px;")
        self._input_lbl.setWordWrap(True)
        ll.addWidget(self._input_lbl)
        ll.addStretch()

        left_outer.addWidget(left, 1)

        # 底部常驻按钮（不随内容滚动，始终可见）
        footer = QWidget()
        footer.setStyleSheet(f"background: {CARD}; border-top: 1px solid {BORDER};")
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(10, 8, 10, 10)
        footer_layout.setSpacing(4)

        self._progress_lbl = QLabel("")
        self._progress_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        self._progress_lbl.setVisible(False)
        footer_layout.addWidget(self._progress_lbl)

        ocr_row = QHBoxLayout()
        self._run_btn = accent_button("▶  开始 OCR")
        self._run_btn.clicked.connect(self._run_ocr)
        ocr_row.addWidget(self._run_btn)
        self._pause_btn = accent_button("⏸ 停止", color="#FF9500")
        self._pause_btn.setVisible(False)
        self._pause_btn.clicked.connect(self._toggle_pause)
        ocr_row.addWidget(self._pause_btn)
        self._rerun_btn = accent_button("🔄 重新OCR", color="#FF9500")
        self._rerun_btn.setVisible(False)
        self._rerun_btn.clicked.connect(self._re_ocr)
        ocr_row.addWidget(self._rerun_btn)
        footer_layout.addLayout(ocr_row)

        left_outer.addWidget(footer)
        top_row.addWidget(left_container)

        # ── 中间：大幅放大的识别区域预览（拖框选定，替代原来的百分比裁剪）───────
        center = QWidget()
        center.setStyleSheet(f"background: {CARD};")
        cv = QVBoxLayout(center)
        cv.setContentsMargins(14, 14, 14, 8)
        cv.setSpacing(6)

        crop_lbl = QLabel("识别区域（拖框选定，留空=整页；只会显示 Page Manager 里标为「正文」的页）")
        crop_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        crop_lbl.setWordWrap(True)
        cv.addWidget(crop_lbl)

        self._preview = OCRCropPreview()
        cv.addWidget(self._preview, 1)

        crop_btn_row = QHBoxLayout()
        clear_crop_btn = QPushButton("清除框选")
        clear_crop_btn.setStyleSheet(f"font-size: 11px; padding: 4px 10px;")
        clear_crop_btn.clicked.connect(self._preview.clear_rect)
        crop_btn_row.addWidget(clear_crop_btn)
        crop_btn_row.addStretch()
        cv.addLayout(crop_btn_row)

        self._preview_enabled_cb = QCheckBox("实时预览")
        self._preview_enabled_cb.setChecked(True)
        crop_btn_row.addWidget(self._preview_enabled_cb)

        crop_hint = QLabel("框选之外的区域完全不会被 OCR 识别；可以随时在图上重新拖拽调整框选范围；OCR 运行时这里会实时切换成当前处理的图片")
        crop_hint.setStyleSheet(f"color: #B7B6AF; font-size: 10px;")
        crop_hint.setWordWrap(True)
        cv.addWidget(crop_hint)

        top_row.addWidget(center, 1)

        # ── 底部：日志/识别结果，收窄成常驻小面板 ───────────────────────────────
        bottom = QWidget()
        bottom.setMaximumHeight(170)
        bottom.setStyleSheet(f"background: {CARD}; border-top: 1px solid {BORDER};")
        rl = QVBoxLayout(bottom)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)

        tab_row = QHBoxLayout()
        tab_row.setContentsMargins(14, 4, 14, 4)
        self._view_log_btn = QPushButton("OCR 日志")
        self._view_log_btn.setProperty("flat", True)
        self._view_log_btn.clicked.connect(lambda: self._switch_view("log"))
        tab_row.addWidget(self._view_log_btn)
        self._view_result_btn = QPushButton("识别结果")
        self._view_result_btn.setProperty("flat", True)
        self._view_result_btn.clicked.connect(lambda: self._switch_view("result"))
        tab_row.addWidget(self._view_result_btn)
        tab_row.addStretch()
        rl.addLayout(tab_row)
        rl.addWidget(make_separator())

        self._prog = QProgressBar()
        self._prog.setVisible(False)
        self._prog.setFixedHeight(6)
        rl.addWidget(self._prog)

        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setStyleSheet(
            f"background: #1B1B1D; color: #CFCFC6; font-family: 'Menlo', monospace; "
            f"font-size: 11px; border: none; border-radius: 0;")
        rl.addWidget(self._log_view, 1)

        self._result_view = QTextEdit()
        self._result_view.setReadOnly(True)
        self._result_view.setStyleSheet(f"border: none; border-radius: 0; padding: 8px 14px; font-size: 11px;")
        self._result_view.setVisible(False)
        rl.addWidget(self._result_view, 1)

        main_v.addWidget(bottom)
        self._highlight_adapter("apple_vision")
        self._vision_backend_widget.setVisible(True)
        self._on_vision_backend_changed()

    def _select_adapter(self, aid, enabled, event):
        if not enabled:
            QMessageBox.information(self, "即将支持", f"该适配器正在开发中，目前请使用 Apple Vision OCR。")
            return
        self._active_adapter = aid
        self._highlight_adapter(aid)
        self._paddle_model_widget.setVisible(aid == "paddle_ocr")
        self._vision_backend_widget.setVisible(aid == "apple_vision")
        if aid == "apple_vision":
            self._on_vision_backend_changed()
        else:
            self._shortcut_widget.setVisible(False)

    def _on_vision_backend_changed(self):
        """切换识别方式：只在选了"快捷指令 App"时才显示快捷指令名称输入框，
        其余方式在下面用一行文字标出这个 backend 实际具备的能力（坐标/
        置信度），能力信息直接从 BackendFactory 里查，不是写死的文案，
        换了实现也不会跟实际能力对不上。"""
        backend_id = self._vision_backend_combo.currentData()
        self._shortcut_widget.setVisible(backend_id == "shortcut")

        if backend_id == "auto":
            self._vision_backend_hint.setText("自动挑选当前环境下可用、能力最好的识别方式")
            return
        try:
            from adapters.vision_backends import BackendFactory
            backend = BackendFactory.create(backend_id)
            available, reason = backend.is_available()
            if not available:
                self._vision_backend_hint.setText(f"⚠️ 当前不可用：{reason}")
                return
            caps = backend.capabilities
            parts = []
            parts.append("坐标✓" if caps.bbox else "无坐标")
            parts.append("置信度✓" if caps.confidence else "无置信度")
            self._vision_backend_hint.setText("　".join(parts))
        except Exception as e:
            self._vision_backend_hint.setText(f"⚠️ {e}")

    def _highlight_adapter(self, aid):
        for a, card in self._adapter_cards.items():
            bc = ACC if a == aid else BORDER
            bw = 2 if a == aid else 1
            card.setStyleSheet(
                f"background: {CARD}; border: {bw}px solid {bc}; border-radius: 8px; margin: 4px 10px;")

    def _switch_view(self, which):
        self._log_view.setVisible(which == "log")
        self._result_view.setVisible(which == "result")
        self._view_log_btn.setStyleSheet(
            f"background: {ACC_BG if which == 'log' else 'transparent'}; "
            f"color: {ACC}; border: none; border-radius: 6px; padding: 6px 12px;")
        self._view_result_btn.setStyleSheet(
            f"background: {ACC_BG if which == 'result' else 'transparent'}; "
            f"color: {ACC}; border: none; border-radius: 6px; padding: 6px 12px;")

    def _pick_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if folder:
            self._pending_inputs = [folder]
            self._input_lbl.setText(f"文件夹: {Path(folder).name}")
            self._load_preview_reference()

    def _pick_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择图片或 PDF（可多选）", "",
            "图片和PDF (*.png *.jpg *.jpeg *.heic *.tif *.tiff *.bmp *.gif *.pdf)")
        if paths:
            self._pending_inputs = paths
            name = Path(paths[0]).name if len(paths) == 1 else f"{len(paths)} 个文件"
            self._input_lbl.setText(f"文件: {name}")
            self._load_preview_reference()

    def set_inputs(self, paths):
        self._pending_inputs = [str(p) for p in paths]
        if paths:
            name = Path(paths[0]).name if len(paths) == 1 else f"{len(paths)} 个文件"
            self._input_lbl.setText(name)
            self._load_preview_reference()

    def _load_preview_reference(self):
        """
        选好输入后，把第一张"正文"页加载进裁剪预览框，供用户拖框选定识别区域——
        故意不用第一张图片（可能是封面/插图，版式和正文页不一样，照着它画框会跑偏）。
        """
        self._preview.clear_rect()
        if not self._pending_inputs:
            return

        image_exts = {'.png', '.jpg', '.jpeg', '.heic', '.tif', '.tiff', '.bmp', '.gif'}
        try:
            from adapters.pdf_input import expand_inputs, natural_sort_key
            work_dir = (self._pending_inputs[0] if Path(self._pending_inputs[0]).is_dir()
                        else str(Path(self._pending_inputs[0]).parent))
            images = sorted(set(expand_inputs(self._pending_inputs, work_dir=work_dir)), key=natural_sort_key)
        except Exception:
            images = []
        images = [p for p in images if Path(p).suffix.lower() in image_exts]
        if not images:
            return

        # 只认页面管理页里用户手动确认过的标注（不再按文件名猜类型）——找第一张
        # 明确标成"正文"的页面作为参照图；没有手动标注就直接用第一张图。
        pm = getattr(self.window(), "_tab_pages", None)
        known = dict(pm.page_overrides) if pm is not None else {}

        candidate = None
        for i, p in enumerate(images):
            if known.get(i + 1) == "paragraph":
                candidate = p
                break
        self._preview.set_image(candidate or images[0])

    def _toggle_pause(self):
        # 请求停止：设置取消标志位，适配器会在处理完当前页后于下一页开始前中止，
        # 并返回已经识别完成的部分结果（而不是丢弃全部进度）。
        self._cancel_event.set()
        self._pause_btn.setEnabled(False)
        self._pause_btn.setText("正在停止…")
        self._log_view.appendPlainText("\n⏸  已请求停止，将在当前页处理完成后中止…")

    def _run_ocr(self):
        inputs = self._pending_inputs
        if not inputs:
            QMessageBox.warning(self, "错误", "请先选择文件夹或文件")
            return

        # 让页面管理页也能看到同一批输入的缩略图（两个页签共享同一次导入结果）
        main_win = self.window()
        if hasattr(main_win, "_tab_pages"):
            main_win._sync_page_manager_inputs(inputs)

        self._switch_view("log")
        self._log_view.clear()
        self._run_btn.setEnabled(False)
        self._rerun_btn.setVisible(False)
        self._pause_btn.setVisible(True)
        self._pause_btn.setEnabled(True)
        self._pause_btn.setText("⏸ 停止")
        self._cancel_event = threading.Event()
        self._prog.setVisible(True)
        self._prog.setRange(0, 1)
        self._prog.setValue(0)
        self._progress_lbl.setVisible(True)
        self._progress_lbl.setText("准备中…")

        def on_page_progress(current, total, filename, image_path=None):
            signals.progress.emit(current, total)
            signals.log.emit(f"  [{current:3d}/{total}] {filename}")
            if image_path and self._preview_enabled_cb.isChecked():
                # 解码 + 缩小放在这个后台线程里做（原图可能是几千像素的高清扫描件，
                # 用 QImage 读取在这里是线程安全的），主线程那边只需要把结果转成
                # QPixmap 显示，不然每页都在主线程读大图会让整个界面跟着卡顿、
                # 感觉 OCR 变慢了（其实是 UI 线程被这步拖住了，不管用哪个识别引擎
                # 都一样会卡）。
                img = QImage(image_path)
                if not img.isNull():
                    img = img.scaled(1000, 1000, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    signals.current_image_data.emit(img)

        def worker():
            try:
                pm = getattr(main_win, "_tab_pages", None)

                # 判断"这一页是不是正文"决定要不要跳过 OCR。不再按文件名猜测——
                # 只认页面管理页里用户手动确认过的标注；没有标注就照旧 OCR 一遍，
                # 交给适配器按识别出的字数/标点事后判断是不是正文。
                #
                # 页面管理页导入时会自动给第一页/最后一页打上"封面"/"版权页"
                # 建议（PageManagerTab._finish_load），但那只是给用户看的初始
                # 猜测，不是"确认过跳过识别"——这里要把还停留在自动建议状态的
                # 非正文标注过滤掉，只转发 paragraph（本来就不跳过 OCR）和用户
                # 真正点选确认过的标注，否则只有一两页的测试输入、或者首页其实
                # 是正文的书会被直接判定成"封面"整页跳过识别，识别不出任何东西。
                page_overrides = {}
                if pm is not None and pm._last_loaded_raw_inputs == inputs:
                    for page_no, ptype in pm.page_overrides.items():
                        if page_no in pm._auto_suggested and ptype != BlockType.PARAGRAPH.value:
                            continue
                        page_overrides[page_no] = ptype

                common_kwargs = dict(
                    input_paths=inputs,
                    page_overrides=page_overrides,
                    verbose=False,
                    progress_callback=on_page_progress,
                    cancel_check=self._cancel_event.is_set,
                    crop_rect=self._preview.get_crop_rect(),
                )
                if self._active_adapter == "paddle_ocr":
                    from adapters.paddle_ocr_adapter import run as ocr_run
                    doc = ocr_run(pipeline=self._paddle_model_combo.currentData(), **common_kwargs)
                else:
                    from adapters.apple_vision_adapter import run as ocr_run
                    doc = ocr_run(
                        shortcut_name=self._shortcut_edit.text().strip(),
                        backend=self._vision_backend_combo.currentData(),
                        vertical=self._vision_vertical_check.isChecked(),
                        **common_kwargs,
                    )
                signals.finished.emit(doc)
            except Exception as e:
                import traceback
                signals.error.emit(traceback.format_exc())

        signals = self._ocr_signals = WorkerSignals()
        signals.finished.connect(self._on_done)
        signals.error.connect(self._on_error)
        signals.log.connect(self._log_view.appendPlainText)
        signals.progress.connect(self._on_progress)
        signals.current_image_data.connect(self._preview.set_image_data)
        threading.Thread(target=worker, daemon=True).start()

    def _on_progress(self, current, total):
        self._prog.setRange(0, max(total, 1))
        self._prog.setValue(current)
        self._progress_lbl.setText(f"{current} / {total} 页")

    def _reset_run_state(self):
        self._run_btn.setEnabled(True)
        self._pause_btn.setVisible(False)
        self._rerun_btn.setVisible(False)
        self._prog.setVisible(False)
        self._progress_lbl.setVisible(False)

    def _on_done(self, doc):
        self._reset_run_state()
        self._rerun_btn.setVisible(True)
        was_cancelled = any("暂停" in log.get("message", "") for log in doc.processing_log)
        status = "⏸ 已停止（部分结果）" if was_cancelled else "✅ OCR 完成"
        self._log_view.appendPlainText(f"\n{status}: {len(doc.blocks)} 块，{len(doc.toc)} 章节")

        # 显示结果摘要
        lines = [f"<b>OCR 引擎:</b> {doc.metadata.source_engine}",
                 f"<b>总页数:</b> {len(doc.pages)}  ·  <b>总块数:</b> {len(doc.blocks)}", "",
                 "<b>章节目录:</b>"]
        for e in doc.toc:
            lines.append(f"  {e.chapter_index}. {e.title}")
        if not doc.toc:
            lines.append("  （未识别到章节）")
        self._result_view.setHtml("<br>".join(lines))

        self.ocr_done.emit(doc)

    def _on_error(self, msg):
        self._reset_run_state()
        self._log_view.appendPlainText(f"\n❌ 错误:\n{msg}")
        show_error_dialog(self, "OCR 失败", msg)

    def _re_ocr(self):
        """复用当前输入、裁剪框和适配器设置，清空日志/结果后重新执行一次 OCR。"""
        self._result_view.clear()
        self._result_view.setHtml("")
        self._run_ocr()


# ══════════════════════════════════════════════════════════════════════════════
#  Tab 3 — Formatter Engine
# ══════════════════════════════════════════════════════════════════════════════

class FormatterTab(QWidget):
    doc_formatted = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ocr_doc: Optional[UnifiedDocument] = None
        self._fmt_doc: Optional[UnifiedDocument] = None
        self._paddle_doc: Optional[UnifiedDocument] = None   # <-- 在这里添加这一行
        self._active_step = FORMATTER_STEPS[0][0]
        self._build()

    def _build(self):
        root = wrap_in_card(self)

        # ── 左侧：步骤卡片 ────────────────────────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(320)
        left.setMaximumWidth(360)
        left.setStyleSheet(f"background: {CARD};")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)

        # 顶部
        top = QWidget()
        top.setFixedHeight(48)
        top.setStyleSheet(f"background: {CARD};")
        tl = QHBoxLayout(top)
        tl.setContentsMargins(14, 0, 10, 0)
        tl.addWidget(QLabel("<b>Formatter Engine</b>"))
        tl.addStretch()
        ll.addWidget(top)
        ll.addWidget(make_separator())

        self._step_checks: dict[str, QCheckBox] = {}
        self._step_cards: dict[str, QWidget] = {}
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none;")
        steps_widget = QWidget()
        steps_widget.setStyleSheet(f"background: {CARD};")
        sl = QVBoxLayout(steps_widget)
        sl.setContentsMargins(8, 8, 8, 8)
        sl.setSpacing(4)

        for sid, label, badge_text, desc in FORMATTER_STEPS:
            card = QWidget()
            card.setCursor(QCursor(Qt.PointingHandCursor))
            card.setStyleSheet(f"background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px; padding: 4px;")
            cl = QVBoxLayout(card)
            cl.setContentsMargins(8, 6, 8, 6)
            cl.setSpacing(2)

            hdr = QHBoxLayout()
            cb = QCheckBox()
            cb.setChecked(True)
            self._step_checks[sid] = cb
            hdr.addWidget(cb)
            nlbl = QLabel(f"<b>{label}</b>")
            nlbl.setStyleSheet("font-size: 12px;")
            hdr.addWidget(nlbl)
            hdr.addStretch()
            hdr.addWidget(make_badge(badge_text, BADGE_COLOR.get(badge_text, "#888")))
            cl.addLayout(hdr)

            dl = QLabel(desc)
            dl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
            dl.setWordWrap(True)
            cl.addWidget(dl)

            card.mousePressEvent = partial(self._select_step, sid)
            sl.addWidget(card)
            self._step_cards[sid] = card

        sl.addStretch()
        scroll.setWidget(steps_widget)
        ll.addWidget(scroll, 1)

        root.addWidget(left, 0)

        # ── 右侧：工具栏 + 对比区 ─────────────────────────────────────────────
        right = QWidget()
        right.setMinimumWidth(760)
        right.setStyleSheet(f"background: {CARD};")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)

        # 工具栏
        toolbar = QWidget()
        toolbar.setFixedHeight(48)
        tbl = QHBoxLayout(toolbar)
        tbl.setContentsMargins(14, 0, 14, 0)

        self._step_title = QLabel("")
        self._step_title.setStyleSheet("font-weight: bold; font-size: 13px;")
        tbl.addWidget(self._step_title)
        tbl.addStretch()

        load_btn = QPushButton("📂 载入JSON")
        load_btn.setProperty("flat", True)
        load_btn.clicked.connect(self._load_json)
        tbl.addWidget(load_btn)

        docx_btn = QPushButton("📄 导入 DOCX")
        docx_btn.setProperty("flat", True)
        docx_btn.clicked.connect(self._import_docx)
        tbl.addWidget(docx_btn)

        epub_btn = QPushButton("📖 导入 EPUB")
        epub_btn.setProperty("flat", True)
        epub_btn.clicked.connect(self._import_epub)
        tbl.addWidget(epub_btn)

        self._preserve_layout_cb = QCheckBox("固定原OCR排版")
        self._preserve_layout_cb.setToolTip("文本替换时只替换文字内容，保留 OCR 原有段落结构，避免替换后乱分段")
        self._preserve_layout_cb.setChecked(False)
        tbl.addWidget(self._preserve_layout_cb)

        replace_btn = QPushButton("🔀 文本替换")   # ← 这里缩进与上面一致（4个空格）
        replace_btn.setProperty("flat", True)
        replace_btn.setToolTip("用外部高质量文本（docx/epub/json/txt/md/html）替换当前 OCR 正文，保留页面结构")
        replace_btn.clicked.connect(self._run_text_replacement)
        tbl.addWidget(replace_btn)

        # ----- 新增：OCR 配准专用按钮 -----
        load_paddle_btn = QPushButton("📂 载入Paddle")
        load_paddle_btn.setProperty("flat", True)
        load_paddle_btn.setToolTip("载入 PaddleOCR 输出的 JSON 结果")
        load_paddle_btn.clicked.connect(self._load_paddle_json)
        tbl.addWidget(load_paddle_btn)

        align_btn = QPushButton("🔄 配准替换")
        align_btn.setProperty("flat", True)
        align_btn.setToolTip("用 PaddleOCR 文本替换当前工作文档（保留 Mac 版式）")
        align_btn.clicked.connect(self._run_ocr_alignment)
        tbl.addWidget(align_btn)
        # ----- 新增结束 -----

        save_btn = QPushButton("💾 保存结果")   # ← 缩进也保持一致
        save_btn.setProperty("flat", True)
        save_btn.clicked.connect(self._save_json)
        tbl.addWidget(save_btn)

        export_docx_btn = QPushButton("📤 导出 DOCX")
        export_docx_btn.setProperty("flat", True)
        export_docx_btn.clicked.connect(self._export_docx)
        tbl.addWidget(export_docx_btn)

        undo_btn = QPushButton("↩ 撤销")
        undo_btn.setProperty("flat", True)
        undo_btn.clicked.connect(self._undo)
        tbl.addWidget(undo_btn)

        edit_apply_btn = QPushButton("📝 应用编辑")
        edit_apply_btn.setProperty("flat", True)
        edit_apply_btn.setToolTip("把处理前/处理后文本框里的手动修改写回当前文档")
        edit_apply_btn.clicked.connect(self._apply_manual_edit)
        tbl.addWidget(edit_apply_btn)

        self._version_lbl = QLabel("")
        self._version_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px; padding: 0 8px;")
        tbl.addWidget(self._version_lbl)

        apply_btn = accent_button("✓ 应用此步", color=SUCCESS)
        apply_btn.clicked.connect(self._apply_step)
        tbl.addWidget(apply_btn)

        run_btn = accent_button("▶  全部运行")
        run_btn.clicked.connect(self._run_all)
        tbl.addWidget(run_btn)

        rl.addWidget(toolbar)
        rl.addWidget(make_separator())

        self._progress = QProgressBar()
        self._progress.setVisible(False)
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
        self._before = QPlainTextEdit()
        self._before.setReadOnly(False)
        self._before.setUndoRedoEnabled(True)
        self._before.setPlaceholderText("OCR 原始文本（可编辑，点击「应用编辑」写回）")
        self._after = QPlainTextEdit()
        self._after.setReadOnly(False)
        self._after.setUndoRedoEnabled(True)
        self._after.setPlaceholderText("处理后文本（可编辑，点击「应用编辑」写回）")
        splitter.addWidget(self._before)
        splitter.addWidget(self._after)
        rl.addWidget(splitter, 1)

        # AI 校正面板
        ai_bar = QWidget()
        ai_bar.setStyleSheet(f"background: #FAFAF7;")
        al = QHBoxLayout(ai_bar)
        al.setContentsMargins(14, 6, 14, 6)
        al.addWidget(QLabel("AI 校正:"))
        self._ai_provider = QComboBox()
        self._ai_provider.addItems(["openai", "deepseek", "gemini"])
        self._ai_provider.setFixedWidth(100)
        al.addWidget(self._ai_provider)
        self._ai_model = QLineEdit()
        self._ai_model.setPlaceholderText("模型 (留空=默认)")
        self._ai_model.setFixedWidth(140)
        al.addWidget(self._ai_model)
        self._ai_key = QLineEdit()
        self._ai_key.setEchoMode(QLineEdit.Password)
        self._ai_key.setPlaceholderText("API Key")
        self._ai_key.setFixedWidth(180)
        al.addWidget(self._ai_key)
        ai_run = accent_button("运行 AI 校正", color="#AF52DE")
        ai_run.clicked.connect(self._run_ai)
        al.addWidget(ai_run)
        ai_reformat = accent_button("AI 重新排版", color="#AF52DE")
        ai_reformat.clicked.connect(self._run_ai_reformat)
        al.addWidget(ai_reformat)
        al.addStretch()
        rl.addWidget(ai_bar)

        # 底部规则提示
        self._rules_bar = QWidget()
        self._rules_bar.setFixedHeight(34)
        self._rules_bar.setStyleSheet(f"background: #FAFAF7; border-top: 1px solid {BORDER};")
        self._rules_layout = QHBoxLayout(self._rules_bar)
        self._rules_layout.setContentsMargins(14, 0, 14, 0)
        rl.addWidget(self._rules_bar)

        root.addWidget(right, 1)
        self._select_step(FORMATTER_STEPS[0][0])


    def _load_paddle_json(self):
        """载入 PaddleOCR-VL 输出的 JSON/Markdown，作为「配准替换」的高质量文本来源。
        不影响当前工作文档（self._ocr_doc / self._fmt_doc），只是把结果暂存到
        self._paddle_doc，供 _run_ocr_alignment() 使用。"""
        main_win = self.window()
        pm = getattr(main_win, "_tab_pages", None)

        image_folder = ""
        overrides: dict = {}
        # offset=0：这份 JSON 覆盖整本书、从第一张本地图片开始（最常见情况）。
        # import_paddle_json() 内部会自动把 JSON 自己的 page_index（不管从
        # 0 还是从 1 开始）归一化，这里不需要、也不应该再去猜本地图片文件
        # 名的起始数字——那样猜反而会导致图片全书错位一张（详见
        # utils/paddle_importer.py 里的说明）。只有导入的是书的一部分、
        # 需要跟已加载的完整图片列表对齐时，才需要手动传非 0 的 offset。
        offset = 0
        if pm is not None and getattr(pm, "page_images", None):
            image_folder = str(pm.page_images[0].parent)
            overrides = getattr(pm, "page_overrides", {})

        fn, _ = QFileDialog.getOpenFileName(
            self, "载入 PaddleOCR 输出", "", "JSON/Markdown (*.json *.md)")
        if not fn:
            return

        try:
            if fn.lower().endswith(".json"):
                doc = import_paddle_json(
                    fn, image_folder, overrides, offset,
                    page_images=getattr(pm, "page_images", None),
                    strip_special_text=True)
            else:
                doc = import_paddle_md(fn, image_folder, overrides, offset)
        except Exception as e:
            show_error_dialog(self, "载入失败", str(e))
            return

        self._paddle_doc = doc
        QMessageBox.information(
            self, "载入完成",
            f"已载入 PaddleOCR 文本，共 {len(doc.blocks)} 个块。\n"
            f"接下来点击「🔄 配准替换」，用它按页/字符级配准替换当前工作文档的正文（保留版式）。")

    def _run_ocr_alignment(self):
        """用已载入的 PaddleOCR 文本（self._paddle_doc），按页分块 + 字符级
        局部序列对齐，回填替换当前工作文档的正文——保留当前文档的块结构/
        坐标/分页（即 Mac OCR 等本地识别结果的版式）。"""
        doc = self._fmt_doc or self._ocr_doc
        if not doc:
            QMessageBox.warning(self, "提示", "请先完成 OCR 或载入 JSON")
            return
        if not self._paddle_doc:
            QMessageBox.warning(self, "提示", "请先点击「📂 载入Paddle」载入 PaddleOCR 输出")
            return

        def worker():
            try:
                from engine.ocr_aligner import OcrAligner
                aligner = OcrAligner(doc, self._paddle_doc)
                new_doc, report = aligner.align()
                signals.finished.emit({"doc": new_doc, "report": report})
            except Exception:
                import traceback
                signals.error.emit(traceback.format_exc())

        signals = self._align_signals = WorkerSignals()
        signals.finished.connect(self._on_align_done)
        signals.error.connect(self._on_align_error)
        threading.Thread(target=worker, daemon=True).start()

    def _on_align_done(self, payload):
        new_doc = payload["doc"]
        report = payload.get("report") or {}
        self._on_run_done(new_doc)
        replaced = report.get("replaced", "?")
        total = report.get("total_text_blocks", "?")
        pages = report.get("pages_processed", "?")
        QMessageBox.information(
            self, "配准替换完成",
            f"已用 PaddleOCR 文本替换 {replaced}/{total} 个文字块（{pages} 页参与配准），\n"
            f"版式（块结构/坐标/分页）保持不变。")

    def _on_align_error(self, msg):
        show_error_dialog(self, "配准失败", msg)

    def set_doc(self, doc: UnifiedDocument):
        self._ocr_doc = doc
        # 接入新 OCR 文档时清理上一本书的格式化缓存
        self._fmt_doc = None
        self._paddle_doc = None
        self._show_doc(doc, self._before)
        self._after.clear()
        if hasattr(self, "_update_version"):
            self._update_version(None)

    def _select_step(self, sid, event=None):
        self._active_step = sid
        for s, card in self._step_cards.items():
            if s == sid:
                card.setStyleSheet(f"background: {ACC_BG}; border: 2px solid {ACC}; border-radius: 8px; padding: 4px;")
            else:
                card.setStyleSheet(f"background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px; padding: 4px;")
        info = {s: (l, d) for s, l, _, d in FORMATTER_STEPS}
        if sid in info:
            label, desc = info[sid]
            self._step_title.setText(f"{label} — {desc}")
        self._refresh_rules(sid)

    def _refresh_rules(self, sid):
        while self._rules_layout.count():
            item = self._rules_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        RULES = {
            "reading_order": ["GapTree 列聚类", "右→左 上→下"],
            "clean_metadata": ["页码正则 /^\\d{1,6}$/", "页眉位置检测 y<0.15", "跨页重复行检测"],
            "split_embedded_titles": ["块内搜索章节关键词", "只在句子边界后才拆分"],
            "strip_chapter_notes": ["数字＋（前書）/（後書き）标记触发", "裸数字块才算备注结束"],
            "merge_sentences": ["末尾无句末符→合并", "接续词感知（て/で/から）", "章节标题不合并"],
            "remove_duplicates": ["相邻规范化匹配", "章节标题模糊去重", "对白全文去重"],
            "fix_dash_artifacts": ["孤立「→——", "非ruby的｜/|→——"],
            "dialogue_restore": ["只拆「…」（不拆『…』）", "迭代拆分混合段落"],
            "restore_indents": ["全角空格缩进", "分节符检测 ◆※☆"],
            "recover_ruby": ["｜漢字《よみ》→ ruby", "EPUB3 ruby 渲染"],
            "detect_chapters": ["序章/第X章/幕間/後記", "正规化后匹配"],
            "strip_boilerplate": ["前书：首章前+命中签名才删", "后书：末尾签名+相邻乱码块"],
            "normalize_punct": ["...→……", "--→——", "半角→全角括号"],
        }
        rl = QLabel("规则:")
        rl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        self._rules_layout.addWidget(rl)
        for r in RULES.get(sid, []):
            pill = QLabel(r)
            pill.setStyleSheet(
                f"background: #EFEEE9; color: #555; font-size: 11px; "
                f"border-radius: 4px; padding: 2px 6px;")
            self._rules_layout.addWidget(pill)
        self._rules_layout.addStretch()

    def _show_doc(self, doc, widget):
        if doc is None:
            return
        lines = []
        for b in doc.blocks:
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
                prefix = "  ◼ " if b.modified_by else "  "
                lines.append(f"{prefix}{b.text}")
        widget.setPlainText("\n".join(lines))

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
        new_doc = copy.deepcopy(doc)
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

    def _apply_manual_edit(self):
        total_changed = 0

        if self._ocr_doc is not None:
            updated, changed = self._apply_editor_to_doc(self._ocr_doc, self._before, "处理前")
            if updated is None:
                return
            self._ocr_doc = updated
            total_changed += changed

        if self._fmt_doc is not None and self._after.toPlainText().strip():
            updated, changed = self._apply_editor_to_doc(self._fmt_doc, self._after, "处理后")
            if updated is None:
                return
            self._fmt_doc = updated
            total_changed += changed
            self._update_version(self._fmt_doc)
            self.doc_formatted.emit(self._fmt_doc)
        elif self._ocr_doc is not None:
            self.doc_formatted.emit(self._ocr_doc)

        self._show_doc(self._ocr_doc, self._before)
        if self._fmt_doc is not None:
            self._show_doc(self._fmt_doc, self._after)

        QMessageBox.information(self, "完成", f"已应用手动编辑：{total_changed} 个文本块")

    def _load_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "载入 JSON", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                doc = UnifiedDocument.from_json(f.read())
            self._ocr_doc = doc
            self._show_doc(doc, self._before)
            QMessageBox.information(self, "完成", f"已载入 {len(doc.blocks)} 个块")
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

        def worker():
            try:
                from adapters.docx_adapter import import_docx
                doc = import_docx(path, verbose=True)
                signals.finished.emit(doc)
            except Exception:
                import traceback
                signals.error.emit(traceback.format_exc())

        signals = self._docx_signals = WorkerSignals()
        signals.finished.connect(self._on_docx_imported)
        signals.error.connect(self._on_error)
        threading.Thread(target=worker, daemon=True).start()

    def _on_docx_imported(self, doc):
        self._ocr_doc = doc
        self._fmt_doc = None
        self._show_doc(doc, self._before)
        self._after.clear()
        self._update_version(None)
        self.doc_formatted.emit(doc)
        QMessageBox.information(self, "导入完成", f"从 DOCX 导入 {len(doc.blocks)} 个块，{len(doc.toc)} 个章节\n可直接勾选步骤运行后处理。")

    def _import_epub(self):
        """
        逆向导入已有 EPUB（比如本工具之前生成、又手动编辑过的文件），
        重新回炉跑 Formatter Pipeline——常见场景是用"前后书剥离"步骤清理
        导入时才发现混进正文里的网站样板文字/版权声明。
        """
        path, _ = QFileDialog.getOpenFileName(self, "导入 EPUB", "", "EPUB (*.epub)")
        if not path:
            return

        def worker():
            try:
                from adapters.epub_adapter import import_epub
                doc = import_epub(path, verbose=True)
                signals.finished.emit(doc)
            except Exception:
                import traceback
                signals.error.emit(traceback.format_exc())

        signals = self._epub_import_signals = WorkerSignals()
        signals.finished.connect(self._on_epub_imported)
        signals.error.connect(self._on_error)
        threading.Thread(target=worker, daemon=True).start()

    def _on_epub_imported(self, doc):
        self._ocr_doc = doc
        self._fmt_doc = None
        self._show_doc(doc, self._before)
        self._after.clear()
        self._update_version(None)
        self.doc_formatted.emit(doc)
        QMessageBox.information(
            self, "导入完成",
            f"从 EPUB 导入 {len(doc.blocks)} 个块，{len(doc.toc)} 个目录条目\n"
            f"勾选「前后书剥离」等步骤运行后处理，可清理残留的样板文字。")

    def _run_text_replacement(self):
        """
        用外部高质量文本来源替换当前 OCR 正文——OCR 结果负责页面结构
        （图片/章节顺序/版面），来源文件负责正文内容，两边通过
        engine/alignment.py 的章节锚点 + Needleman-Wunsch 段落对齐自动配对，
        不是简单按顺序对应（会漏识别/多识别/拆分/合并导致的错位）。
        """
        doc = self._fmt_doc or self._ocr_doc
        if not doc:
            QMessageBox.warning(self, "提示", "请先完成 OCR 或载入 JSON")
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "选择高质量文本来源", "",
            "支持的格式 (*.docx *.epub *.json *.txt *.md *.markdown *.html *.htm)")
        if not path:
            return

        def worker():
            try:
                from adapters.text_extractors import extract_paragraphs
                from engine.replacement_engine import replace_text
                source_paragraphs = extract_paragraphs(path)
                preserve_layout = self._preserve_layout_cb.isChecked()
                new_doc, report = replace_text(
                    doc, source_paragraphs,
                    preserve_ocr_layout=preserve_layout,
                )
                signals.finished.emit({
                    "doc": new_doc,
                    "report": report,
                    "preserve_layout": preserve_layout,
                })
            except Exception:
                import traceback
                signals.error.emit(traceback.format_exc())

        signals = self._replace_signals = WorkerSignals()
        signals.finished.connect(self._on_text_replacement_done)
        signals.error.connect(self._on_error)
        threading.Thread(target=worker, daemon=True).start()

    def _on_text_replacement_done(self, payload):
        new_doc = payload["doc"]
        report = payload["report"]
        preserve_layout = payload.get("preserve_layout", False)
        mode = "固定 OCR 版式" if preserve_layout else "标准替换"
        self._on_run_done(new_doc)
        QMessageBox.information(
            self, "文本替换完成",
            f"模式: {mode}\n"
            f"替换 {report.replaced} 段（低置信度跳过 {report.low_confidence} 段）\n"
            f"OCR 中未找到对应来源: {report.skipped_ocr} 段（原文保留）\n"
            f"来源中未找到对应 OCR 位置: {report.skipped_source} 段（未自动插入，见下方预览）\n"
            f"平均相似度: {report.avg_similarity:.1%}　耗时: {report.execution_seconds:.1f}s\n\n"
            + ("未匹配来源预览:\n" + "\n".join(f"· {t}" for t in report.unmatched_source_preview[:10])
               if report.unmatched_source_preview else ""))

    def _save_json(self):
        doc = self._fmt_doc or self._ocr_doc
        if not doc:
            QMessageBox.information(self, "提示", "还没有处理结果")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存 JSON", "", "JSON (*.json)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(doc.to_json())
            QMessageBox.information(self, "完成", f"已保存: {path}")

    def _export_docx(self):
        doc = self._fmt_doc or self._ocr_doc
        if not doc:
            QMessageBox.information(self, "提示", "还没有处理结果")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出 DOCX", "", "Word (*.docx)")
        if not path:
            return

        def worker():
            try:
                from builder.word_builder import build_word
                build_word(doc, output_path=path, verbose=False)
                signals.finished.emit(path)
            except Exception as e:
                import traceback
                signals.error.emit(traceback.format_exc())

        signals = self._docx_export_signals = WorkerSignals()
        signals.finished.connect(lambda p: QMessageBox.information(self, "完成", f"DOCX 已生成:\n{p}"))
        signals.error.connect(lambda msg: show_error_dialog(self, "导出失败", msg))
        threading.Thread(target=worker, daemon=True).start()

    def _undo(self):
        doc = self._fmt_doc
        log = doc.commit_log() if doc else []
        if not doc or len(log) < 2:
            QMessageBox.information(self, "提示", "没有可撤销的步骤")
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
        self._run_steps(steps, base_on_current=True)

    def _apply_step(self):
        self._run_steps([self._active_step], base_on_current=True)

    def _run_steps(self, steps, base_on_current=True):
        base = (self._fmt_doc if base_on_current and self._fmt_doc else self._ocr_doc)
        if base is None:
            QMessageBox.warning(self, "错误", "请先完成 OCR 或载入 JSON")
            return

        self._show_doc(self._ocr_doc or base, self._before)
        self._progress.setVisible(True)
        self._progress.setRange(0, len(steps))

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
        signals.finished.connect(self._on_run_done)
        signals.error.connect(self._on_error)
        signals.progress.connect(lambda cur, _: self._progress.setValue(cur))
        threading.Thread(target=worker, daemon=True).start()

    def _on_run_done(self, result):
        self._fmt_doc = result
        self._progress.setVisible(False)
        self._show_doc(result, self._after)
        self._update_version(result)
        self.doc_formatted.emit(result)

    def _on_error(self, msg):
        self._progress.setVisible(False)
        show_error_dialog(self, "处理失败", msg)

    def _run_ai(self):
        doc = self._fmt_doc or self._ocr_doc
        if not doc:
            QMessageBox.warning(self, "提示", "请先运行 Pipeline 或载入 JSON")
            return
        key = self._ai_key.text().strip()
        if not key:
            QMessageBox.warning(self, "提示", "请输入 API Key")
            return

        provider_name = self._ai_provider.currentText()
        model = self._ai_model.text().strip()

        def worker():
            try:
                from ai import get_provider, apply_suggestions
                provider = get_provider(provider_name, api_key=key, model=model)
                suggestions = provider.correct_ocr(doc)
                print(f"[AI DEBUG] suggestions={len(suggestions)}")
                for i, item in enumerate(suggestions[:5]):
                    print(f"[AI DEBUG] {i}: block={item.block_index} {item.original[:30]!r} -> {item.suggested[:30]!r}")
                corrected = apply_suggestions(doc, suggestions)
                signals.finished.emit(corrected)
            except Exception as e:
                import traceback
                signals.error.emit(traceback.format_exc())

        signals = self._ai_signals = WorkerSignals()
        signals.finished.connect(self._on_run_done)
        signals.error.connect(self._on_error)
        threading.Thread(target=worker, daemon=True).start()

    def _run_ai_reformat(self):
        """
        按某个 Format Profile 描述的排版风格用 AI 重新排版——和 _run_ai 走
        完全一样的 worker 线程 + apply_suggestions + 前后对比刷新逻辑，
        只是多一步选格式、调用 provider.reformat 而不是 correct_ocr。
        """
        doc = self._fmt_doc or self._ocr_doc
        if not doc:
            QMessageBox.warning(self, "提示", "请先运行 Pipeline 或载入 JSON")
            return
        key = self._ai_key.text().strip()
        if not key:
            QMessageBox.warning(self, "提示", "请输入 API Key")
            return

        profiles = FormatProfileStore().list()
        if not profiles:
            QMessageBox.information(
                self, "提示",
                "还没有自定义排版格式。先去 EPUB Builder 页的模板下拉框旁点「⚙」，"
                "新建一个空白格式或提交参考 EPUB 学习一个格式。")
            return
        names = [p.name for p in profiles]
        name, ok = QInputDialog.getItem(self, "选择排版格式", "按哪个格式重新排版:", names, 0, False)
        if not ok:
            return
        profile = next(p for p in profiles if p.name == name)

        provider_name = self._ai_provider.currentText()
        model = self._ai_model.text().strip()

        def worker():
            try:
                from ai import get_provider, apply_suggestions
                provider = get_provider(provider_name, api_key=key, model=model)
                suggestions = provider.reformat(doc, profile)
                corrected = apply_suggestions(doc, suggestions)
                signals.finished.emit(corrected)
            except Exception:
                import traceback
                signals.error.emit(traceback.format_exc())

        signals = self._ai_reformat_signals = WorkerSignals()
        signals.finished.connect(self._on_run_done)
        signals.error.connect(self._on_error)
        threading.Thread(target=worker, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
#  Format Profile 管理窗口
# ══════════════════════════════════════════════════════════════════════════════

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
        left.setFixedWidth(230)
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
        right.setMinimumWidth(760)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)

        form = QFormLayout()
        self._name_edit = QLineEdit()
        form.addRow("名称:", self._name_edit)
        rl.addLayout(form)

        learn_btn = accent_button("📖 从参考 EPUB 学习格式…", color="#AF52DE")
        learn_btn.clicked.connect(self._learn_from_epub)
        rl.addWidget(learn_btn)

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
        close_btn = accent_button("关闭", color="#E5E5EA", text_color=INK)
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)

        rl.addLayout(btn_row)

        root.addWidget(right, 1)

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
        path, _ = QFileDialog.getOpenFileName(self, "选择参考 EPUB（日文/中文）", "", "EPUB (*.epub)")
        if not path:
            return
        default_name = Path(path).stem
        name, ok = QInputDialog.getText(self, "格式名称", "给学到的格式起个名字:", text=default_name)
        if not ok or not name.strip():
            return
        name = name.strip()

        def worker():
            try:
                profile = FormatProfile.from_reference_epub(path, name)
                signals.finished.emit(profile)
            except Exception:
                import traceback
                signals.error.emit(traceback.format_exc())

        signals = self._learn_signals = WorkerSignals()
        signals.finished.connect(self._on_learned)
        signals.error.connect(lambda msg: show_error_dialog(self, "学习格式失败", msg))
        threading.Thread(target=worker, daemon=True).start()

    def _on_learned(self, profile: FormatProfile):
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

            QMessageBox.information(self, "完成", msg)
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
        QMessageBox.information(self, "完成", "格式已保存")

    def _delete(self):
        if self._is_builtin or self._current is None:
            return
        ret = QMessageBox.question(self, "删除格式", f"确定删除格式「{self._current.name}」？此操作不可撤销。")
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
            QMessageBox.information(self, "提示", "内置模板不支持导出，请先选中一个自定义格式")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出格式", f"{self._current.name}.json", "JSON (*.json)")
        if not path:
            return
        self.store.export_to(self._current.id, path)
        QMessageBox.information(self, "完成", f"已导出: {path}")

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
        QMessageBox.information(self, "完成", f"已导入格式「{profile.name}」")


# ══════════════════════════════════════════════════════════════════════════════
#  Tab 4 — EPUB Builder
# ══════════════════════════════════════════════════════════════════════════════

class EPUBTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._doc: Optional[UnifiedDocument] = None
        self._paddle_button = QPushButton("📥 导入 PaddleOCR-VL")
        self._paddle_button.clicked.connect(self.import_paddle_output)
        self._build()


    def import_paddle_output(self):
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        import re

        main_win = self.window()
        if main_win is None:
            QMessageBox.warning(self, "错误", "无法获取主窗口")
            return

        pm = getattr(main_win, "_tab_pages", None)
        if pm is None or not getattr(pm, "page_images", None):
            QMessageBox.warning(self, "提示", "请先在页面管理中加载图片并标注类型。")
            return

        fn, _ = QFileDialog.getOpenFileName(
            self,
            "选择 PaddleOCR-VL 输出",
            "",
            "JSON/Markdown (*.json *.md)"
        )
        if not fn:
            return

        image_folder = str(pm.page_images[0].parent)
        overrides = getattr(pm, "page_overrides", {})

        # offset=0：完整导入整本书，从第一张本地图片开始。import_paddle_json()
        # 内部会自动归一化 JSON 自己的 page_index 起点，不需要在这里猜本地
        # 图片文件名的起始数字（那样猜会导致全书图片错位一张，详见
        # utils/paddle_importer.py 里的说明）。
        offset = 0

        try:
            from utils.paddle_importer import import_paddle_json, import_paddle_md

            if fn.lower().endswith(".json"):
                doc = import_paddle_json(fn, image_folder, overrides, offset, page_images=pm.page_images, strip_special_text=True)
            else:
                doc = import_paddle_md(fn, image_folder, overrides, offset)

        except Exception as e:
            QMessageBox.critical(self, "导入失败", str(e))
            return

        self._doc = doc

        fmt_tab = getattr(main_win, "_tab_fmt", None)
        if fmt_tab and hasattr(fmt_tab, "set_doc"):
            fmt_tab.set_doc(doc)

        if hasattr(main_win, "_goto"):
            main_win._goto(3)

        QMessageBox.information(
            self,
            "导入成功",
            f"已导入 {len(doc.blocks)} 个块。"
        )

    def _build(self):
        root = wrap_in_card(self)

        # ── 左侧：结构树 + 设置 ───────────────────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(320)
        left.setMaximumWidth(360)
        left.setStyleSheet(f"background: {CARD};")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)

        top = QWidget()
        tv = QVBoxLayout(top)
        tv.setContentsMargins(14, 10, 14, 10)
        tv.setSpacing(8)
        tv.addWidget(QLabel("<b>EPUB Builder</b>"))

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.setContentsMargins(0, 0, 0, 0)

        paddle_btn = QPushButton("📥 Paddle-VL")
        paddle_btn.setProperty("flat", True)
        paddle_btn.setStyleSheet(f"background: {CARD}; color: {INK}; border: 1px solid {BORDER};")
        paddle_btn.clicked.connect(self.import_paddle_output)
        btn_row.addWidget(paddle_btn)

        word_btn = QPushButton("📝 Word")
        word_btn.setProperty("flat", True)
        word_btn.setStyleSheet(f"background: {CARD}; color: {INK}; border: 1px solid {BORDER};")
        word_btn.clicked.connect(self._export_word)
        btn_row.addWidget(word_btn)

        build_btn = accent_button("⚙ Build EPUB")
        build_btn.clicked.connect(self._build_epub)
        btn_row.addWidget(build_btn, 1)
        tv.addLayout(btn_row)

        ll.addWidget(top)
        ll.addWidget(make_separator())

        # 书籍结构树
        tree_lbl = QLabel("Book Structure")
        tree_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px; padding: 8px 14px 4px;")
        ll.addWidget(tree_lbl)
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setStyleSheet(f"margin: 0 8px;")
        placeholder = QTreeWidgetItem(self._tree, ["点击 Build 生成"])
        ll.addWidget(self._tree, 1)

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
        self._title_edit.setPlaceholderText("书名")
        fl.addRow("书名:", self._title_edit)
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
        tr.addWidget(self._template_combo, 1)
        manage_tpl_btn = QPushButton("⚙ 管理格式")
        manage_tpl_btn.setProperty("flat", True)
        manage_tpl_btn.setToolTip("管理排版格式（新建/学习/删除/导入导出）")
        manage_tpl_btn.clicked.connect(self._open_format_profiles)
        tr.addWidget(manage_tpl_btn)
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
        ll.addSpacing(10)
        root.addWidget(left, 0)

        # ── 右侧：元数据条 + 源码/预览 ────────────────────────────────────────
        right = QWidget()
        right.setMinimumWidth(760)
        right.setStyleSheet(f"background: {CARD};")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)

        # 元数据信息条
        pills_bar = QWidget()
        pills_bar.setFixedHeight(44)
        pills_bar.setStyleSheet(f"background: {CARD};")
        pbl = QHBoxLayout(pills_bar)
        pbl.setContentsMargins(14, 0, 14, 0)
        self._pills: dict[str, QLabel] = {}
        for key, icon, default in [
            ("title", "📖", "未命名"), ("author", "👤", "作者未填"),
            ("lang", "🌐", "ja · EPUB3"), ("mode", "✎", "vertical-rl"),
            ("pages", "🖼", "0 页 · 0 图"),
        ]:
            pill = QLabel(f"{icon} {default}")
            pill.setStyleSheet(
                f"background: #F5F4F0; color: #444; border: 1px solid {BORDER}; "
                f"border-radius: 6px; padding: 4px 10px; font-size: 11px;")
            pbl.addWidget(pill)
            self._pills[key] = pill
        pbl.addStretch()
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
        self._code_btn.setStyleSheet(f"background: {ACC_BG}; color: {ACC}; border: none; border-radius: 6px; padding: 6px 12px;")
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

        # 预览占位
        self._preview_widget = QLabel("📖 EPUB 实时预览区域\n\n等待页面选择")
        self._preview_widget.setAlignment(Qt.AlignCenter)
        self._preview_widget.setStyleSheet(f"background: #F0EFEA; color: {MUTED}; font-size: 14px;")
        self._preview_widget.setVisible(False)
        rl.addWidget(self._preview_widget, 1)

        # 底部统计
        rl.addWidget(make_separator())
        stats = QWidget()
        stats.setFixedHeight(54)
        stats.setStyleSheet(f"background: {CARD};")
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

        self._tree.itemClicked.connect(self._on_tree_click)
        self._tree_data: dict[int, dict] = {}

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

    def set_doc(self, doc: UnifiedDocument):
        self._doc = doc

        # 新 OCR 文档进入时，清理上一本书的 EPUB 构建缓存
        self._tree.clear()
        self._tree_data.clear()
        placeholder = QTreeWidgetItem(self._tree, ["点击 Build 生成"])

        if doc.metadata.title:
            self._title_edit.setText(doc.metadata.title)
        if doc.metadata.author:
            self._author_edit.setText(doc.metadata.author)
        if doc.metadata.publisher:
            self._publisher_edit.setText(doc.metadata.publisher)

        for key, value in {
            "chapters": "0",
            "images": "0",
            "size": "0 KB",
            "status": "待构建",
            "files": "0",
        }.items():
            if key in self._stat_labels:
                self._stat_labels[key][0].setText(value)

        self._code_view.clear()
        self._preview_title.setText("选择左侧文件预览内容")
        self._update_pills()

    def _update_pills(self):
        self._pills["title"].setText(f"📖 {self._title_edit.text() or '未命名'}")
        self._pills["author"].setText(f"👤 {self._author_edit.text() or '作者未填'}")
        mode = "vertical-rl" if self._vert_radio.isChecked() else "horizontal"
        self._pills["mode"].setText(f"✎ {mode}")
        if self._doc:
            np = len(self._doc.pages)
            ni = len(self._doc.image_blocks())
            self._pills["pages"].setText(f"🖼 {np} 页 · {ni} 图")

    def _switch_preview(self, which):
        self._code_view.setVisible(which == "code")
        self._preview_widget.setVisible(which == "preview")
        self._code_btn.setStyleSheet(
            f"background: {ACC_BG if which == 'code' else 'transparent'}; "
            f"color: {ACC}; border: none; border-radius: 6px; padding: 6px 12px;")
        self._preview_btn.setStyleSheet(
            f"background: {ACC_BG if which == 'preview' else 'transparent'}; "
            f"color: {ACC}; border: none; border-radius: 6px; padding: 6px 12px;")

    def _on_tree_click(self, item, column):
        idx = id(item)
        if idx in self._tree_data:
            data = self._tree_data[idx]
            self._preview_title.setText(data["name"])
            self._code_view.setPlainText(data.get("content", "(二进制文件，无法预览)"))

    def _export_word(self):
        if self._doc is None:
            QMessageBox.warning(self, "错误", "请先完成 OCR 和格式处理")
            return

        path, _ = QFileDialog.getSaveFileName(self, "导出 Word", "", "Word (*.docx)")
        if not path:
            return

        doc = self._doc
        vertical = self._vert_radio.isChecked()

        def worker():
            try:
                from builder.word_builder import build_word
                build_word(doc, output_path=path, vertical=vertical, verbose=False)
                signals.finished.emit(path)
            except Exception as e:
                import traceback
                signals.error.emit(traceback.format_exc())

        signals = self._word_signals = WorkerSignals()
        signals.finished.connect(lambda p: QMessageBox.information(self, "完成", f"Word 已生成:\n{p}"))
        signals.error.connect(lambda msg: show_error_dialog(self, "导出失败", msg))
        threading.Thread(target=worker, daemon=True).start()

    def _build_epub(self):
        def _debug_doc_state(label, d):
            if d is None:
                print(f"[DOC DEBUG] {label}: NONE")
                return
            try:
                img_count = sum(1 for b in d.blocks if b.type == BlockType.IMAGE_REF)
                samples = []
                for b in d.blocks[:10]:
                    t = getattr(b, "text", "")
                    if t:
                        samples.append(t[:50].replace("\n", "\\n"))
                print("=" * 40)
                print("[DOC DEBUG]", label)
                print("id:", id(d))
                print("blocks:", len(d.blocks))
                print("images:", img_count)
                print("samples:", samples)
                print("=" * 40)
            except Exception as e:
                print("[DOC DEBUG ERROR]", label, e)

        doc = self._doc
        main_win = self.window()
        fmt_tab = getattr(main_win, "_tab_fmt", None) if main_win else None

        _debug_doc_state("EPUBTab._doc", self._doc)
        if fmt_tab is not None:
            _debug_doc_state("Formatter._fmt_doc", getattr(fmt_tab, "_fmt_doc", None))
            _debug_doc_state("Formatter._preview_doc", getattr(fmt_tab, "_preview_doc", None))

            fmt_doc = getattr(fmt_tab, "_fmt_doc", None)
            if fmt_doc is not None:
                doc = fmt_doc
                self._doc = fmt_doc
                print(f"[BUILD] FORCE Formatter 文档: {len(doc.blocks)} blocks")

        if doc is None:
            QMessageBox.warning(self, "错误", "请先完成 OCR 和格式处理")
            return

        img_count = sum(1 for b in doc.blocks if b.type == BlockType.IMAGE_REF)
        print(f"[BUILD] IMAGE_REF 数量: {img_count}")

        path, _ = QFileDialog.getSaveFileName(self, "保存 EPUB", "", "EPUB (*.epub)")
        if not path:
            return

        self._doc = doc
        if self._title_edit.text():
            doc.metadata.title = self._title_edit.text()
        if self._author_edit.text():
            doc.metadata.author = self._author_edit.text()
        if self._publisher_edit.text():
            doc.metadata.publisher = self._publisher_edit.text()
        if self._volume_edit.text():
            doc.metadata.volume = self._volume_edit.text()

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
                signals.finished.emit({"path": path, "log": buf.getvalue()})
            except Exception as e:
                import traceback
                signals.error.emit(traceback.format_exc())

        signals = self._build_signals = WorkerSignals()
        signals.finished.connect(self._on_build_done)
        signals.error.connect(self._on_build_error)
        threading.Thread(target=worker, daemon=True).start()

    def _on_build_done(self, info):
        self._prog.setVisible(False)
        path = info["path"]

        # 更新统计
        size_kb = Path(path).stat().st_size // 1024
        self._stat_labels["chapters"][0].setText(str(len(self._doc.toc)))
        self._stat_labels["images"][0].setText(str(len(self._doc.image_blocks())))
        self._stat_labels["size"][0].setText(f"{size_kb} KB")
        self._stat_labels["status"][0].setText("✓ 完成")
        self._stat_labels["status"][0].setStyleSheet(f"font-size: 16px; font-weight: bold; color: {SUCCESS};")

        # 显示结构树
        self._show_tree(path)
        QMessageBox.information(self, "完成", f"EPUB 已生成:\n{path}\n({size_kb} KB)")

    def _on_build_error(self, msg):
        self._prog.setVisible(False)
        self._stat_labels["status"][0].setText("✗ 失败")
        self._stat_labels["status"][0].setStyleSheet(f"font-size: 16px; font-weight: bold; color: {DANGER};")
        show_error_dialog(self, "生成失败", msg)

    def _show_tree(self, epub_path):
        self._tree.clear()
        self._tree_data.clear()
        try:
            with zipfile.ZipFile(epub_path) as zf:
                names = sorted(zf.namelist())
                contents = {}
                for n in names:
                    if n.endswith(('.xhtml', '.opf', '.css', '.xml')):
                        try:
                            contents[n] = zf.read(n).decode('utf-8', errors='replace')
                        except Exception:
                            pass

            self._stat_labels["files"][0].setText(str(len(names)))

            root_item = QTreeWidgetItem(self._tree, [f"📦 {Path(epub_path).name}"])
            root_item.setExpanded(True)
            dirs: dict[str, QTreeWidgetItem] = {}

            for name in names:
                parts = name.split("/")
                if len(parts) == 1:
                    item = QTreeWidgetItem(root_item, [f"📄 {name}"])
                    self._tree_data[id(item)] = {"name": name, "content": contents.get(name, "(二进制)")}
                else:
                    dname = parts[0]
                    if dname not in dirs:
                        dirs[dname] = QTreeWidgetItem(root_item, [f"📁 {dname}/"])
                        dirs[dname].setExpanded(True)
                    fname = "/".join(parts[1:])
                    if fname:
                        ext = Path(fname).suffix.lower().lstrip(".")
                        icon = {"xhtml": "📝", "css": "🎨", "jpg": "🖼", "jpeg": "🖼",
                                "png": "🖼", "opf": "⚙️", "xml": "⚙️"}.get(ext, "📄")
                        item = QTreeWidgetItem(dirs[dname], [f"{icon} {fname}"])
                        self._tree_data[id(item)] = {"name": name, "content": contents.get(name, "(二进制)")}
        except Exception as e:
            QTreeWidgetItem(self._tree, [f"⚠️ 无法读取: {e}"])


# ══════════════════════════════════════════════════════════════════════════════
#  PDF 文字层直读（独立页签，不经过 OCR 适配器那一套流程）
# ══════════════════════════════════════════════════════════════════════════════

class PdfTextLayerTab(QWidget):
    """
    直接读取带文字层的 PDF（不调用任何 OCR 模型），完全独立于 OCR 适配器页——
    不共享裁剪预览、不共享 Page Manager 同步，避免了把这条路径和图片 OCR
    耦合在一起时反复出现的"莫名其妙把整本 PDF 渲染成图片"的问题。
    提取完的结果和 OCR 结果走同一个信号出口，接入 Formatter / EPUB Builder。
    """
    doc_extracted = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pending_pdf: str | None = None
        self._build()

    def _build(self):
        root = wrap_in_card(self)

        left = QWidget()
        left.setFixedWidth(340)
        left.setStyleSheet(f"background: {CARD}; border-right: 1px solid {BORDER};")
        left_outer = QVBoxLayout(left)
        left_outer.setContentsMargins(0, 0, 0, 0)
        left_outer.setSpacing(0)

        top = QWidget()
        ll = QVBoxLayout(top)
        ll.setContentsMargins(0, 14, 0, 14)

        hdr = QLabel("PDF 文字层直读")
        hdr.setStyleSheet("font-size: 14px; font-weight: bold; padding-left: 14px;")
        ll.addWidget(hdr)
        sub = QLabel("直接读取 PDF 内嵌文字层，不调用任何 OCR 模型，速度快、最准确；"
                      "按字号自动跳过振假名，按位置自动跳过页码")
        sub.setStyleSheet(f"color: {MUTED}; font-size: 11px; padding: 0 14px 8px 14px;")
        sub.setWordWrap(True)
        ll.addWidget(sub)
        ll.addWidget(make_separator())

        inp_lbl = QLabel("输入（单个 PDF 文件）")
        inp_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px; padding: 8px 0 4px 14px;")
        ll.addWidget(inp_lbl)
        pick_btn = QPushButton("📄  选择 PDF 文件")
        pick_btn.setStyleSheet(
            f"background: {CARD}; color: {INK}; border: 1px solid {BORDER}; "
            f"text-align: left; padding: 8px 12px; margin: 0 10px;")
        pick_btn.clicked.connect(self._pick_pdf)
        ll.addWidget(pick_btn)

        self._input_lbl = QLabel("（尚未选择输入）")
        self._input_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px; padding: 6px 14px 0;")
        self._input_lbl.setWordWrap(True)
        ll.addWidget(self._input_lbl)
        ll.addStretch()

        left_outer.addWidget(top, 1)

        footer = QWidget()
        footer.setStyleSheet(f"background: {CARD}; border-top: 1px solid {BORDER};")
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(10, 8, 10, 10)
        self._progress_lbl = QLabel("")
        self._progress_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        self._progress_lbl.setVisible(False)
        footer_layout.addWidget(self._progress_lbl)
        self._run_btn = accent_button("▶  开始提取")
        self._run_btn.clicked.connect(self._run_extract)
        footer_layout.addWidget(self._run_btn)
        left_outer.addWidget(footer)

        root.addWidget(left, 0)

        right = QWidget()
        right.setMinimumWidth(760)
        right.setStyleSheet(f"background: {CARD};")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)

        log_hdr = QLabel("提取日志")
        log_hdr.setStyleSheet("font-weight: bold; font-size: 13px; padding: 8px 14px;")
        rl.addWidget(log_hdr)
        rl.addWidget(make_separator())

        self._prog = QProgressBar()
        self._prog.setVisible(False)
        rl.addWidget(self._prog)

        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setStyleSheet(
            f"background: #1B1B1D; color: #CFCFC6; font-family: 'Menlo', monospace; "
            f"font-size: 12px; border: none; border-radius: 0;")
        rl.addWidget(self._log_view, 1)

        root.addWidget(right, 1)

    def _pick_pdf(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 PDF 文件", "", "PDF (*.pdf)")
        if path:
            self._pending_pdf = path
            self._input_lbl.setText(Path(path).name)

    def _run_extract(self):
        if not self._pending_pdf:
            QMessageBox.warning(self, "错误", "请先选择 PDF 文件")
            return
        pdf_path = self._pending_pdf

        self._log_view.clear()
        self._run_btn.setEnabled(False)
        self._prog.setVisible(True)
        self._prog.setRange(0, 1)
        self._prog.setValue(0)
        self._progress_lbl.setVisible(True)
        self._progress_lbl.setText("准备中…")

        def on_progress(current, total, label):
            signals.progress.emit(current, total)
            signals.log.emit(f"  [{current:3d}/{total}] {label}")

        def worker():
            try:
                from adapters.pdf_text_layer import has_text_layer, extract_pdf_text_layer
                if not has_text_layer(pdf_path):
                    raise ValueError("该 PDF 没有可提取的文字层（可能是扫描版 PDF），请改用 OCR 适配器")
                doc = extract_pdf_text_layer(pdf_path, verbose=False, progress_callback=on_progress)
                signals.finished.emit(doc)
            except Exception as e:
                import traceback
                signals.error.emit(traceback.format_exc())

        signals = self._signals = WorkerSignals()
        signals.finished.connect(self._on_done)
        signals.error.connect(self._on_error)
        signals.log.connect(self._log_view.appendPlainText)
        signals.progress.connect(self._on_progress)
        threading.Thread(target=worker, daemon=True).start()

    def _on_progress(self, current, total):
        self._prog.setRange(0, max(total, 1))
        self._prog.setValue(current)
        self._progress_lbl.setText(f"{current} / {total} 页")

    def _on_done(self, doc):
        self._run_btn.setEnabled(True)
        self._prog.setVisible(False)
        self._progress_lbl.setVisible(False)
        self._log_view.appendPlainText(f"\n✅ 完成: {len(doc.blocks)} 个块，{len(doc.toc)} 章节")
        self.doc_extracted.emit(doc)

    def _on_error(self, msg):
        self._run_btn.setEnabled(True)
        self._prog.setVisible(False)
        self._progress_lbl.setVisible(False)
        show_error_dialog(self, "提取失败", msg)


# ══════════════════════════════════════════════════════════════════════════════
#  左侧导航栏
# ══════════════════════════════════════════════════════════════════════════════

class Sidebar(QWidget):
    section_changed = Signal(int)

    ITEMS = [
        ("📑", "页面管理"),
        ("🔍", "OCR 适配器"),
        ("📄", "PDF 文字层"),
        ("✨", "Formatter"),
        ("📚", "EPUB Builder"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(200)
        self.setStyleSheet(f"background-color: {SIDEBAR_BG};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 22, 14, 14)
        layout.setSpacing(3)

        title = QLabel("Novel Formatter")
        title.setStyleSheet(
            f"font-size: 15px; font-weight: 700; color: {INK}; "
            f"padding: 0 6px 18px 6px;"
        )
        layout.addWidget(title)

        self._buttons: list[QToolButton] = []
        for icon, label in self.ITEMS:
            btn = QToolButton()
            btn.setText(f"  {icon}   {label}")
            btn.setCheckable(True)
            btn.setAutoExclusive(True)
            btn.setCursor(QCursor(Qt.PointingHandCursor))
            btn.setStyleSheet(SIDEBAR_BTN_STYLE)
            btn.setMinimumHeight(34)
            idx = len(self._buttons)
            btn.clicked.connect(partial(self._on_click, idx))
            layout.addWidget(btn)
            self._buttons.append(btn)

        self._buttons[0].setChecked(True)
        layout.addStretch(1)

        footer = QLabel(f"v{VERSION}")
        footer.setStyleSheet(f"color: {MUTED}; font-size: 10px; padding: 0 6px;")
        layout.addWidget(footer)

    def _on_click(self, idx: int):
        self.section_changed.emit(idx)

    def select(self, idx: int):
        """程序化切换（不经过用户点击），同步高亮状态，不重复发出信号。"""
        if 0 <= idx < len(self._buttons):
            self._buttons[idx].setChecked(True)


# ══════════════════════════════════════════════════════════════════════════════
#  主窗口
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Novel Formatter Studio v{VERSION}")
        self.setMinimumSize(1100, 700)
        self.resize(1320, 840)

        self._doc: Optional[UnifiedDocument] = None
        self._paddle_button = QPushButton("📥 导入 PaddleOCR-VL")
        self._paddle_button.clicked.connect(self.import_paddle_output)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 左侧导航栏
        self._sidebar = Sidebar()
        root_layout.addWidget(self._sidebar)

        # 右侧：内容堆叠 + 底部历史滑块
        right = QWidget()
        right.setMinimumWidth(760)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self._stack = QStackedWidget()

        self._tab_pages = PageManagerTab()
        self._tab_ocr = OCRTab()
        self._tab_pdf_text = PdfTextLayerTab()
        self._tab_fmt = FormatterTab()
        self._tab_epub = EPUBTab()

        self._stack.addWidget(self._tab_pages)
        self._stack.addWidget(self._tab_ocr)
        self._stack.addWidget(self._tab_pdf_text)
        self._stack.addWidget(self._tab_fmt)
        self._stack.addWidget(self._tab_epub)

        right_layout.addWidget(self._stack, 1)

        root_layout.addWidget(right, 1)

        # 导航联动：侧边栏点击 → 切换内容；内容内部跳转 → 同步侧边栏高亮
        self._sidebar.section_changed.connect(self._goto)

        # 信号连接
        self._tab_pages.pages_loaded.connect(self._on_pages_loaded)
        self._tab_pages.go_ocr.connect(lambda: self._goto(1))
        self._tab_pages.types_changed.connect(self._on_page_types_changed)
        self._tab_ocr.ocr_done.connect(self._on_ocr_done)
        self._tab_pdf_text.doc_extracted.connect(self._on_ocr_done)
        self._tab_fmt.doc_formatted.connect(self._on_fmt_done)

    def import_paddle_output(self):
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        fn,_=QFileDialog.getOpenFileName(self,"选择 PaddleOCR-VL 输出","","JSON/Markdown (*.json *.md)")
        if not fn: return
        folder=getattr(self._tab_pages,"image_folder","")
        try:
            if fn.endswith(".json"):
                self._doc=import_paddle_json(fn,folder,{},0)
            else:
                self._doc=import_paddle_md(fn,folder,{},0)
            QMessageBox.information(self,"PaddleOCR-VL","导入完成，已生成 UnifiedDocument")
        except Exception as e:
            QMessageBox.critical(self,"导入失败",str(e))

    def _goto(self, idx: int):
        """统一的页面切换入口：同时更新内容堆叠和侧边栏高亮，两者不会不同步。"""
        self._stack.setCurrentIndex(idx)
        self._sidebar.select(idx)

    def _on_pages_loaded(self, images):
        self._tab_ocr.set_inputs(images)

    def _on_page_types_changed(self):
        """
        页面管理页里改了某页的分类（或删了页）之后，OCR 页的裁剪参照图要跟着
        刷新——否则改完类型回到 OCR 页，看到的还是改之前选中的那张参照图
        （可能已经不再是正文页了），拖框裁剪也就没意义了。
        """
        if self._tab_ocr._pending_inputs:
            self._tab_ocr._load_preview_reference()

    def _sync_page_manager_inputs(self, inputs: list[str]):
        """
        当用户直接在「OCR 适配器」页选择输入（而不是先经过「页面管理」页）时，
        把同一批输入同步加载进页面管理页，这样切换过去也能看到缩略图，
        而不是一片空白。
        """
        if self._tab_pages.page_images and self._tab_pages._last_loaded_raw_inputs == inputs:
            # 页面管理页已经加载过同一批输入，不重复加载
            return
        name = Path(inputs[0]).name if len(inputs) == 1 else f"{len(inputs)} 个输入"
        self._tab_pages._load_inputs(inputs, name)

    def _on_ocr_done(self, doc):
        self._doc = doc
        self._tab_fmt.set_doc(doc)
        self._tab_epub.set_doc(doc)
        self._goto(3)

    def _on_fmt_done(self, doc):
        self._doc = doc
        self._tab_epub.set_doc(doc)


# ══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    # v2.4.5 UI refresh: use uploaded mascot icon
    try:
        icon_path = Path(__file__).resolve().parent / "icon.ico"
        if icon_path.exists():
            app.setWindowIcon(QIcon(str(icon_path)))
    except Exception as e:
        print("[icon]", e)
    app.setStyleSheet(STYLE)
    if sys.platform == "darwin":
        app.setStyle("Fusion")
    window = MainWindow()
    try:
        icon_path = Path(__file__).resolve().parent / "icon.ico"
        if icon_path.exists():
            window.setWindowIcon(QIcon(str(icon_path)))
    except Exception:
        pass
    window.show()
    sys.exit(app.exec())



# ---- v2.1 unified clear buttons ----
def _install_clear_buttons():
    try:
        from utils.clear_manager import ClearManager
        from PySide6.QtWidgets import QPushButton, QMessageBox

        def add_clear(tab, title, func):
            btn=danger_button("🧹 "+title)
            def run():
                if QMessageBox.question(tab,"确认","确定清空当前工作区？") == QMessageBox.Yes:
                    func(tab)
            btn.clicked.connect(run)
            return btn

        for cls, name, func in [
            (PageManagerTab,"清空页面",ClearManager.clear_pages),
            (OCRTab,"清空OCR",ClearManager.clear_ocr),
            (FormatterTab,"清空Formatter",ClearManager.clear_formatter),
            (EPUBTab,"清空EPUB",ClearManager.clear_epub),
        ]:
            old=cls._build
            def wrapper(self, old=old, name=name, func=func):
                old(self)
                try:
                    # 找到最外层布局追加按钮
                    lay=self.layout()
                    if lay:
                        lay.addWidget(add_clear(self,name,func))
                except Exception:
                    pass
            cls._build=wrapper
    except Exception as e:
        print("[clear button init]",e)

_install_clear_buttons()

if __name__ == "__main__":
    main()
