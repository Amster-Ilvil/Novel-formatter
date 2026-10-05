from __future__ import annotations

from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QFrame, QLabel, QPushButton, QToolButton, QCheckBox, QRadioButton, QSplitter, QTabWidget, QScrollArea, QComboBox, QLineEdit, QSizePolicy)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QCursor

from ui.clickable_style_guard import enforce_button_contrast_tree
from ui.flow_layout import FlowLayout

from ui.theme.tokens import (  # noqa: E402,F401
    BG, SIDEBAR_BG, CARD, SURFACE, INK, MUTED, SUBTLE, BORDER, BORDER_STRONG,
    ACC, ACC_HOVER, ACC_PRESSED, ACC_BG, DANGER, SUCCESS, WARN, SEAL,
    TONAL, TONAL_HOVER, ACC_TOP, ACC_FILL, ACC_FILL_HOVER, ACC_TEXT, DISABLED_FG, HERO_A, HERO_B, ON_HERO, ON_HERO_TEXT,
    CLICKABLE_HOVER, CLICKABLE_PRESSED,
)
CLICKABLE_BG = BG

LIGHT_PREVIEW_STYLE = """
background: #FFFFFF;
color: #14202E;
border: 1px solid #E2E5E9;
border-radius: 8px;
padding: 10px;
selection-background-color: #EAEDF1;
selection-color: #14202E;
"""

LIGHT_LOG_STYLE = """
background: #F7F8FA;
color: #5B6B80;
border: 1px solid #E2E5E9;
border-radius: 8px;
padding: 10px;
font-family: "SFMono-Regular", "Menlo", "Monaco", "Courier New";
font-size: 11px;
selection-background-color: #EAEDF1;
selection-color: #14202E;
"""

EDITOR_SCROLLBAR_STYLE = f"""
QScrollBar:vertical {{
    background: #FFFFFF; width: 10px; margin: 2px; border: none; border-radius: 5px;
}}
QScrollBar::handle:vertical {{
    background: #D8DDE3; min-height: 42px; border-radius: 4px; margin: 2px;
}}
QScrollBar::handle:vertical:hover {{ background: #B8BEC7; }}
QScrollBar::handle:vertical:pressed {{ background: #5F7189; }}
QScrollBar::sub-line:vertical, QScrollBar::add-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
QScrollBar:horizontal {{
    background: #FFFFFF; height: 10px; margin: 2px; border: none; border-radius: 5px;
}}
QScrollBar::handle:horizontal {{
    background: #D8DDE3; min-width: 42px; border-radius: 4px; margin: 2px;
}}
QScrollBar::handle:horizontal:hover {{ background: #B8BEC7; }}
QScrollBar::sub-line:horizontal, QScrollBar::add-line:horizontal {{ width: 0; }}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: transparent; }}
QAbstractScrollArea::corner {{ background: #F7F8FA; }}
"""

def blend(hex_color: str, alpha: float = 0.15, bg: str = BG) -> str:
    def p(h, fallback="#888888"):
        h = (h or fallback).strip().lstrip("#")
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        if len(h) != 6:
            h = fallback.lstrip("#")
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
    bg = blend(color, 0.13, CARD)
    lbl.setStyleSheet(
        f"background-color: {bg}; color: {color}; "
        f"border-radius: 9px; padding: 2px 10px; font-size: 11px; font-weight: 600;")
    return lbl


def _valid_button_color(value: str, fallback: str) -> str:
    color = QColor(str(value or ""))
    return color.name().upper() if color.isValid() else fallback


def _darker_button_color(value: str, factor: int) -> str:
    color = QColor(_valid_button_color(value, ACC))
    return color.darker(max(100, int(factor))).name().upper()


def _primary_button_stylesheet(color: str, text_color: str) -> str:
    """Return a complete per-widget style for high-contrast primary buttons.

    A local stylesheet is deliberate here.  On some macOS/Fusion builds the
    dynamic ``role=primary`` selector keeps the white text but loses the fill
    when the button becomes disabled.  That creates a white-on-white control.
    Styling the actual button fixes enabled, hover, pressed and disabled states
    independently of the native palette or a parent stylesheet.
    """
    base = _valid_button_color(color, ACC)
    foreground = _valid_button_color(text_color, "#FFFFFF")
    hover = _darker_button_color(base, 108)
    pressed = _darker_button_color(base, 118)
    disabled_bg = blend(base, 0.16, CARD)
    disabled_border = blend(base, 0.34, CARD)
    return f"""
/* nf-primary-button-contrast */
QPushButton {{
    background-color: {base}; color: {foreground}; border: 1px solid {base};
    border-radius: 10px; padding: 7px 16px; font-weight: 650; min-height: 18px;
}}
QPushButton:hover {{ background-color: {hover}; color: {foreground}; border-color: {hover}; }}
QPushButton:pressed {{ background-color: {pressed}; color: {foreground}; border-color: {pressed}; }}
QPushButton:disabled {{
    background-color: {disabled_bg}; color: #5B6B80; border-color: {disabled_border};
}}
"""


def accent_button(text: str, color: str = ACC_FILL, text_color: str = "#FFFFFF") -> QPushButton:
    """创建始终清晰可读的主操作按钮。"""
    btn = QPushButton(text)
    btn.setProperty("role", "primary")
    btn.setProperty("accentColor", _valid_button_color(color, ACC))
    btn.setProperty("accentTextColor", _valid_button_color(text_color, "#FFFFFF"))
    btn.setCursor(QCursor(Qt.PointingHandCursor))
    btn.setStyleSheet(_primary_button_stylesheet(color, text_color))
    fm = btn.fontMetrics()
    btn.setMinimumWidth(fm.horizontalAdvance(btn.text()) + 30)
    btn.setMinimumHeight(30)
    return btn


def link_button(text: str) -> QPushButton:
    """Low-noise visible entry point for advanced/secondary functionality.

    Phase 26 restores discoverability without bringing back the dense legacy
    toolbars.  These buttons behave like text links, but keep normal keyboard
    focus and a generous click target.
    """
    btn = QPushButton(text)
    btn.setProperty("role", "link")
    btn.setCursor(QCursor(Qt.PointingHandCursor))
    btn.setMinimumHeight(28)
    btn.setStyleSheet(
        f"QPushButton{{text-align:left;background:transparent;border:none;color:{ACC_TEXT};"
        "padding:5px 2px;font-size:11px;font-weight:650;}"
        f"QPushButton:hover{{color:{ACC_HOVER};}}"
        f"QPushButton:pressed{{color:{ACC_PRESSED};}}"
        f"QPushButton:disabled{{color:{DISABLED_FG};}}"
    )
    return btn


class WrapRow(QWidget):
    """可自动折行的工具条行。

    以 QHBoxLayout 的常用 API（addWidget / addSpacing / addStretch）为兼容面，
    内部使用 FlowLayout：窗口变窄时控件整齐换到下一行，而不是把按钮文字
    挤压截断成「AI纠…」。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        from ui.flow_layout import FlowLayout
        self._flow = FlowLayout(self, margin=0, hspacing=8, vspacing=6)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

    def addWidget(self, widget, *args, **kwargs):
        self._flow.addWidget(widget)

    def addSpacing(self, _px):  # 折行布局下无需手工间距
        pass

    def addStretch(self, *_args):  # 折行布局下无右对齐语义
        pass


def wrap_in_card(owner: QWidget) -> QHBoxLayout:
    """建立与参考图一致的无外框工作区表面。

    页面本身使用浅灰底，主工作区为整块白色表面；内部各功能区再通过
    细分隔线和独立卡片形成层级，避免旧版“卡片套卡片”的厚重感。
    """
    owner.setObjectName("workspacePage")
    owner.setStyleSheet("")
    outer = QVBoxLayout(owner)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)
    card = QFrame()
    card.setObjectName("workspaceSurface")
    outer.addWidget(card)
    inner = QHBoxLayout(card)
    inner.setContentsMargins(0, 0, 0, 0)
    inner.setSpacing(0)
    return inner


def polish_reference_workspace(root: QWidget) -> None:
    """统一所有工作区的交互密度和细节，不改变任何业务信号。"""
    for widget_type in (QPushButton, QToolButton, QCheckBox, QRadioButton):
        for button in root.findChildren(widget_type):
            button.setCursor(QCursor(Qt.PointingHandCursor))
    for splitter in root.findChildren(QSplitter):
        if not bool(splitter.property("nfPreserveHandleWidth")):
            splitter.setHandleWidth(1)
        splitter.setChildrenCollapsible(False)
    for tabs in root.findChildren(QTabWidget):
        tabs.setDocumentMode(True)
    for scroll in root.findChildren(QScrollArea):
        scroll.setFrameShape(QFrame.NoFrame)
    for combo in root.findChildren(QComboBox):
        if combo.maximumHeight() >= 30:
            combo.setMinimumHeight(max(30, combo.minimumHeight()))
    for edit in root.findChildren(QLineEdit):
        edit.setMinimumHeight(max(30, edit.minimumHeight()))
    # Run once during construction as well as through the application event
    # filter.  This covers disabled controls that are already visible in the
    # initial empty-document state (EPUB/Formatter/文字校对等工作区).
    enforce_button_contrast_tree(root)

