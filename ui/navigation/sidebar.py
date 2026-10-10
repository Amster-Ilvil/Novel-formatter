from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

from PySide6.QtWidgets import QWidget, QVBoxLayout, QStackedWidget, QSizePolicy, QPushButton, QButtonGroup
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QCursor, QIcon

from PySide6.QtWidgets import QLabel, QHBoxLayout
from ui.theme.tokens import (
    BG, SIDEBAR_BG, CARD, BORDER, ACC, INK, MUTED, DISABLED_FG, CLICKABLE_HOVER, HERO_A, HERO_B,
)
from ui.design.components import DesignBrandLabel, DesignNavButton, tinted_svg_icon
from ui.design.metrics import SIDEBAR_WIDTH, SIDEBAR_PAD_X, SIDEBAR_PAD_TOP, SIDEBAR_ITEM_GAP, SIDEBAR_BRAND_GAP
from core.command_catalog import PRIMARY_NAVIGATION_SPECS, primary_navigation_shortcut_label


def _tinted_icon(path: Path, color: str) -> str:
    """把线性图标重新着色后写入临时目录，深色轨道上才看得清。"""
    import re, tempfile
    out = Path(tempfile.gettempdir()) / f"nf_nav_{path.stem}_{color.lstrip('#')}.svg"
    try:
        out.write_text(re.sub(r'stroke="#[0-9A-Fa-f]{3,8}"', f'stroke="{color}"',
                              path.read_text(encoding="utf-8")), encoding="utf-8")
        return str(out)
    except Exception:
        return str(path)


REFERENCE_SECTION_ITEMS = tuple(
    (spec.icon_name, spec.title) for spec in PRIMARY_NAVIGATION_SPECS
)
SECTION_WORKSPACE = 0
SECTION_PAGE = 1
SECTION_OCR = 2
SECTION_FORMAT = 3
SECTION_PROOF = 4
SECTION_EPUB = 5
SECTION_SYSTEM = 6


class _ReferenceStack(QStackedWidget):
    """QTabWidget-compatible subset used by the lazy workspace installer."""

    def removeTab(self, index: int) -> None:
        widget = self.widget(int(index))
        if widget is not None:
            self.removeWidget(widget)

    def insertTab(self, index: int, widget: QWidget, _label: str = "") -> int:
        return self.insertWidget(int(index), widget)


class ReferenceSectionHost(QWidget):
    """Flat reference tab host with a full-width content stack.

    The old QTabWidget pane imposed left/right content margins on every page.
    That made the OCR log and PDF canvas physically incapable of matching the
    supplied masters.  This host keeps the public ``set_current_index`` /
    ``current_index`` contract but separates the compact segment bar from a
    true edge-to-edge QStackedWidget.
    """

    current_changed = Signal(int)

    def __init__(self, tabs: list[tuple[str, QWidget]], parent=None, *, overlay_tabs: bool = False, overlay_width: int = 408, show_segment_bar: bool = True):
        super().__init__(parent)
        self._overlay_tabs = bool(overlay_tabs)
        self._overlay_width = max(220, int(overlay_width))
        self._show_segment_bar = bool(show_segment_bar)
        self.setObjectName("referenceSectionHost")
        self.setStyleSheet(f"QWidget#referenceSectionHost{{background:{BG};border:none;}}")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        bar = QWidget(self)
        bar.setObjectName("referenceSegmentBar")
        bar.setFixedHeight(42)
        bar.setStyleSheet(f"QWidget#referenceSegmentBar{{background:{BG};border:none;}}")
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(18, 4, 12, 6)
        bar_layout.setSpacing(8)

        self._segment_group = QButtonGroup(self)
        self._segment_group.setExclusive(True)
        self._segment_buttons: list[QPushButton] = []
        self._stack = _ReferenceStack(self)
        self._stack.setObjectName("referenceSectionStack")
        self._stack.setStyleSheet(f"QStackedWidget#referenceSectionStack{{background:{BG};border:none;}}")
        # Compatibility surface used by legacy tests/callers: QStackedWidget
        # exposes count/currentIndex/setCurrentIndex just like the subset that
        # was consumed from the old QTabWidget.
        self.tabs = self._stack

        for index, (label, widget) in enumerate(tabs):
            button = QPushButton(label, bar)
            button.setObjectName("referenceSegmentButton")
            button.setCheckable(True)
            button.setAutoExclusive(True)
            button.setCursor(QCursor(Qt.PointingHandCursor))
            button.setFixedHeight(32)
            button.setStyleSheet(
                f"QPushButton#referenceSegmentButton{{background:transparent;color:{MUTED};"
                f"border:1px solid transparent;border-radius:10px;padding:5px 16px;"
                f"font-size:12px;font-weight:500;}}"
                f"QPushButton#referenceSegmentButton:hover{{background:{CLICKABLE_HOVER};color:{INK};}}"
                f"QPushButton#referenceSegmentButton:checked{{background:{CARD};color:{ACC};"
                f"border-color:{BORDER};font-weight:700;}}"
            )
            button.clicked.connect(partial(self.set_current_index, index))
            self._segment_group.addButton(button, index)
            self._segment_buttons.append(button)
            bar_layout.addWidget(button)
            self._stack.addWidget(widget)

        bar_layout.addStretch(1)
        self._segment_bar = bar
        if not self._show_segment_bar:
            # Some workspaces provide their own in-page view switch.  Keep the
            # stacked host API and hidden segment buttons for compatibility, but
            # do not reserve the historical 42 px navigation strip.
            bar.setVisible(False)
            layout.addWidget(self._stack, 1)
        elif self._overlay_tabs:
            # Keep the OCR/PDF switcher only over the left control column.
            # The content stack itself starts at y=0, so the right preview/log
            # gains the full height that was previously wasted by a full-width bar.
            bar.setParent(self)
            bar.setFixedHeight(42)
            bar.setFixedWidth(self._overlay_width)
            layout.addWidget(self._stack, 1)
            bar.raise_()
        else:
            layout.addWidget(bar)
            layout.addWidget(self._stack, 1)

        self._stack.currentChanged.connect(self._sync_segment)
        self._stack.currentChanged.connect(self.current_changed.emit)
        if self._segment_buttons:
            self._segment_buttons[0].setChecked(True)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if getattr(self, "_overlay_tabs", False) and getattr(self, "_show_segment_bar", True):
            bar = getattr(self, "_segment_bar", None)
            if bar is not None:
                bar.setGeometry(0, 0, min(self.width(), self._overlay_width), 42)
                bar.raise_()

    def _sync_segment(self, index: int) -> None:
        if 0 <= int(index) < len(self._segment_buttons):
            self._segment_buttons[int(index)].setChecked(True)

    def set_current_index(self, index: int) -> None:
        if 0 <= int(index) < self._stack.count():
            self._stack.setCurrentIndex(int(index))

    def current_index(self) -> int:
        return self._stack.currentIndex()


class Sidebar(QWidget):
    section_changed = Signal(int)
    command_palette_requested = Signal()

    # 七个主功能区；工作区独立于页面管理。OCR/PDF 保留紧凑顶部切换；
    # 文字校对的全文总览/图文对照改为页内互切，不再占用额外顶部栏。
    ITEMS = list(REFERENCE_SECTION_ITEMS)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("mainSidebar")
        # Force the rail to paint its own background when captured/composited by
        # offscreen Qt as well as on macOS/Windows.  Without WA_StyledBackground
        # the child grab was correct but a full-window grab could inherit BG.
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setFixedWidth(SIDEBAR_WIDTH)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setStyleSheet(
            f"QWidget#mainSidebar {{ background:{SIDEBAR_BG}; border-right:1px solid {BORDER}; }}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SIDEBAR_PAD_X, SIDEBAR_PAD_TOP, SIDEBAR_PAD_X, 14)
        layout.setSpacing(SIDEBAR_ITEM_GAP)

        brand = QHBoxLayout()
        brand.setContentsMargins(4, 0, 0, 0)
        brand.setSpacing(0)
        name = DesignBrandLabel("Novel Formatter", self)
        brand.addWidget(name, 1)
        layout.addLayout(brand)
        layout.addSpacing(SIDEBAR_BRAND_GAP)

        self._buttons: list[DesignNavButton] = []
        icon_root = Path(__file__).resolve().parents[2] / "assets" / "ui_icons"
        for icon_name, label in self.ITEMS:
            btn = DesignNavButton(label)
            icon_path = icon_root / f"{icon_name}.svg"
            active_icon_path = icon_root / f"{icon_name}_active.svg"
            if icon_path.exists():
                icon = QIcon()
                # QIcon state variants keep the selected icon blue while preserving
                # the exact same SVG geometry in every navigation row.
                for size in (QSize(18, 18),):
                    # QIcon cannot directly merge QIcon instances, so use their cache files
                    # through the existing helper for deterministic state variants.
                    dim = _tinted_icon(icon_path, "#6F829B")
                    bright = _tinted_icon(icon_path, ACC)
                    icon.addFile(dim, size, QIcon.Mode.Normal, QIcon.State.Off)
                    icon.addFile(bright, size, QIcon.Mode.Normal, QIcon.State.On)
                    icon.addFile(bright, size, QIcon.Mode.Active, QIcon.State.On)
                btn.setIcon(icon)
                btn.setIconSize(QSize(18, 18))
            idx = len(self._buttons)
            btn.setToolTip(f"{label}（{primary_navigation_shortcut_label(idx)}）")
            btn.clicked.connect(partial(self._on_click, idx))
            layout.addWidget(btn)
            self._buttons.append(btn)

        self._buttons[0].setChecked(True)
        layout.addStretch(1)

        command_btn = DesignNavButton("命令")
        command_btn.setCheckable(False)
        command_btn.setAutoExclusive(False)
        command_shortcut = "⌘K" if sys.platform == "darwin" else "Ctrl+K"
        command_btn.setToolTip(f"命令面板（{command_shortcut}）")
        command_btn.clicked.connect(self.command_palette_requested.emit)
        command_icon = icon_root / "command.svg"
        if command_icon.exists():
            icon = QIcon()
            dim = _tinted_icon(command_icon, "#6F829B")
            bright = _tinted_icon(command_icon, ACC)
            icon.addFile(dim, QSize(18, 18), QIcon.Mode.Normal, QIcon.State.Off)
            icon.addFile(bright, QSize(18, 18), QIcon.Mode.Normal, QIcon.State.On)
            command_btn.setIcon(icon)
            command_btn.setIconSize(QSize(18, 18))
        # Phase 26: keep command discovery visible while preserving the clean
        # reference rail.  The preceding stretch anchors this tertiary row at
        # the bottom instead of mixing it with the seven primary sections.
        command_btn.setVisible(True)
        layout.addWidget(command_btn)
        self._command_palette_button = command_btn

    def _on_click(self, idx: int):
        self.section_changed.emit(idx)

    def select(self, idx: int):
        """程序化切换时同步高亮，不重复发出页面切换信号。"""
        if 0 <= idx < len(self._buttons):
            self._buttons[idx].setChecked(True)
