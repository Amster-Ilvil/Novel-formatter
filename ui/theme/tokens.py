# -*- coding: utf-8 -*-
"""Novel Formatter 设计令牌：白色工作区 + 单一品牌蓝，附带深色模式色值对。

颜色统一写成 6 位大写 hex。深色模式不另写一套 QSS：DARK_PAIRS 会并入
theme_palette._DARK_MAP，由现有的 darken_stylesheet 精确换色。
"""
from __future__ import annotations

BG, SIDEBAR_BG, CARD, SURFACE = "#FFFFFF", "#FFFFFF", "#FFFFFF", "#F7F8FA"
INK, MUTED, SUBTLE = "#14202E", "#5B6B80", "#5F7189"
BORDER, BORDER_STRONG = "#E2E5E9", "#CDD3DA"
ACC, ACC_HOVER, ACC_PRESSED, ACC_BG = "#2F6BFF", "#2559E0", "#1C47B8", "#E4EEFF"
ACC_TOP = "#2D69FA"                       # 主按钮渐变上端
ACC_FILL, ACC_FILL_HOVER, ACC_TEXT = "#2860E8", "#2459DC", "#2250D6"   # 白字按钮底色 / 悬停 / 淡底上的蓝字（对比度 ≥ 4.5）
TONAL, TONAL_HOVER, TONAL_PRESSED = "#F2F4F7", "#EAEDF1", "#E2E6EB"
DANGER, DANGER_BG, DANGER_BORDER = "#D64545", "#FDEEEE", "#F0C9C9"
SUCCESS, SUCCESS_BG, WARN = "#1F8F63", "#E3F5EC", "#B7791F"
DISABLED_BG, DISABLED_FG = "#F2F3F5", "#A1A8B2"
CLICKABLE_HOVER, CLICKABLE_PRESSED = "#F4F5F7", "#EAEDF1"
SCROLL, SCROLL_HOVER = "#D4D8DE", "#B8BEC7"
HERO_A, HERO_B = "#4A86FF", "#2E62E8"     # 项目横幅渐变
ON_HERO, ON_HERO_TEXT = "#FEFEFE", "#2457D6"
SEAL = ACC
RAIL, RAIL_HOVER, RAIL_ON = SIDEBAR_BG, CLICKABLE_HOVER, CARD
FONT_UI = '"PingFang SC","Microsoft YaHei UI","Noto Sans CJK SC","Segoe UI","Helvetica Neue",sans-serif'
FONT_SERIF = '"Songti SC","Noto Serif CJK SC","SimSun",serif'
FONT_MONO = '"SF Mono","Menlo","Consolas","Courier New",monospace'

# 浅色 → 深色
DARK_PAIRS = {
    BG: "#0E151F", SIDEBAR_BG: "#121B27", CARD: "#1B1F24", SURFACE: "#1A2638",
    INK: "#E6EDF6", MUTED: "#9FB0C6", SUBTLE: "#8497B0",
    BORDER: "#26364A", BORDER_STRONG: "#324560",
    ACC_FILL: "#2F5FCB", ACC_FILL_HOVER: "#3A6EDF", ACC_TEXT: "#8DB4FF",
    ACC: "#5B93FF", ACC_HOVER: "#7BA8FF", ACC_PRESSED: "#4A7FE8", ACC_BG: "#1C2E4D", ACC_TOP: "#3468DC",
    TONAL: "#22344F", TONAL_HOVER: "#2B405E", TONAL_PRESSED: "#34496B",
    DANGER: "#FF6B6B", DANGER_BG: "#3A2327", DANGER_BORDER: "#6B3A40",
    SUCCESS: "#4CD39A", SUCCESS_BG: "#163428", WARN: "#E3A94A",
    DISABLED_BG: "#1A2433", DISABLED_FG: "#5E6C80",
    CLICKABLE_HOVER: "#1E2B3E", CLICKABLE_PRESSED: "#25354C",
    SCROLL: "#34465E", SCROLL_HOVER: "#46607F",
    HERO_A: "#27508F", HERO_B: "#1E3F7A", ON_HERO: "#DCE8FF", ON_HERO_TEXT: "#1E4AAE",
}


def build_stylesheet(checkmark: str = "", radio_dot: str = "") -> str:
    chk = f'image: url("{checkmark}");' if checkmark else ""
    dot = f'image: url("{radio_dot}");' if radio_dot else ""
    return f"""
QMainWindow {{ background:{BG}; }}
QWidget {{ font-family:{FONT_UI}; font-size:12px; color:{INK}; }}
QWidget#workspacePage, QWidget#referenceSectionHost, QFrame#workspaceSurface, QWidget#settingsPage {{ background:{BG}; border:none; }}
QStackedWidget#mainSectionStack {{ background:{BG}; border:none; }}
QWidget#bookToolbar {{ background:{BG}; border-bottom:1px solid {BORDER}; }}
QWidget#bookSourceBar {{ background:{SURFACE}; border-bottom:1px solid {BORDER}; }}

/* 页签：选中为白色胶囊，其余无框 */
QTabWidget#referenceSectionTabs::pane {{ background:{CARD}; border:1px solid {BORDER}; border-radius:14px; top:-1px; }}
QTabWidget#referenceSectionTabs > QWidget > QWidget {{ background:{CARD}; }}
QTabBar::tab {{ background:transparent; color:{MUTED}; border:1px solid transparent; border-radius:10px;
    padding:7px 16px; margin-right:4px; min-height:18px; font-weight:500; }}
QTabBar::tab:hover {{ background:{CLICKABLE_HOVER}; color:{INK}; }}
QTabBar::tab:selected {{ background:{CARD}; color:{ACC}; border-color:{BORDER}; font-weight:650; }}
QTabWidget::pane {{ border:none; top:4px; }}

QFrame#settingsCard, QFrame#projectDashboardCard {{ background:{CARD}; border:1px solid {BORDER}; border-radius:16px; }}
QLabel#settingsCardTitle {{ font-size:13px; font-weight:700; }}
QLabel#settingsCardSubtitle {{ color:{MUTED}; font-size:11px; }}
QLabel#enabledBadge {{ background:{SUCCESS_BG}; color:{SUCCESS}; border:none; border-radius:9px; padding:2px 9px; font-size:10px; font-weight:650; }}
QLabel#shortcutKey {{ background:{SURFACE}; color:{MUTED}; border:1px solid {BORDER}; border-radius:6px; padding:2px 7px; font-family:{FONT_MONO}; }}

/* 按钮：默认=淡蓝填充(无描边)；主要=蓝色渐变；危险=淡红填充 */
QPushButton {{ background:{TONAL}; color:{INK}; border:1px solid transparent; border-radius:10px; padding:7px 16px; font-weight:550; min-height:18px; }}
QPushButton:hover {{ background:{TONAL_HOVER}; }}
QPushButton:pressed {{ background:{TONAL_PRESSED}; }}
QPushButton:disabled {{ background:{DISABLED_BG}; color:{DISABLED_FG}; }}
QPushButton:focus {{ outline:none; border-color:{ACC}; }}
QPushButton[role="primary"] {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 {ACC_TOP},stop:1 {ACC_FILL}); color:#FFFFFF; border:1px solid {ACC_FILL_HOVER}; font-weight:650; }}
QPushButton[role="primary"]:hover {{ background:{ACC_FILL_HOVER}; }}
QPushButton[role="primary"]:pressed {{ background:{ACC_FILL}; border-color:{ACC_FILL}; }}
QPushButton[role="primary"]:disabled {{ background:{TONAL}; color:{DISABLED_FG}; border-color:{TONAL}; }}
QPushButton[role="danger"] {{ background:{DANGER_BG}; color:{DANGER}; }}
QPushButton[role="danger"]:hover {{ border-color:{DANGER_BORDER}; }}
QToolButton {{ background:transparent; color:{INK}; border:1px solid transparent; border-radius:9px; padding:5px 9px; }}
QToolButton:hover {{ background:{CLICKABLE_HOVER}; }}
QToolButton:pressed {{ background:{CLICKABLE_PRESSED}; }}
QToolButton:disabled {{ color:{DISABLED_FG}; }}
QToolButton:focus {{ border-color:{ACC}; }}
QToolButton::menu-indicator {{ image:none; }}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{ background:{CARD}; border:1px solid {BORDER_STRONG}; border-radius:10px; padding:6px 11px; min-height:19px; selection-background-color:{ACC}; selection-color:#FFFFFF; }}
QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{ border-color:{ACC_HOVER}; }}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border:1px solid {ACC}; }}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{ background:{DISABLED_BG}; color:{DISABLED_FG}; border-color:{BORDER}; }}
QComboBox::drop-down {{ border:none; width:24px; }}
QComboBox QAbstractItemView {{ background:{CARD}; border:1px solid {BORDER}; border-radius:10px; padding:4px; selection-background-color:{ACC_BG}; selection-color:{ACC_TEXT}; outline:none; }}
QPlainTextEdit, QTextEdit {{ background:{CARD}; border:1px solid {BORDER}; border-radius:12px; padding:10px; font-family:{FONT_MONO}; font-size:11px; selection-background-color:{ACC_BG}; }}
QPlainTextEdit:focus, QTextEdit:focus {{ border-color:{ACC}; }}
QGroupBox {{ background:transparent; border:none; border-top:1px solid {BORDER}; margin-top:10px; padding:14px 0 4px 0; font-weight:650; }}
QGroupBox::title {{ subcontrol-origin:margin; left:0; padding:0 4px 0 0; color:{MUTED}; }}

QCheckBox, QRadioButton {{ spacing:8px; background:transparent; border:none; padding:3px 2px; color:{INK}; }}
QCheckBox::indicator {{ width:15px; height:15px; border-radius:5px; border:1px solid {BORDER_STRONG}; background:{CARD}; }}
QCheckBox::indicator:hover {{ border-color:{ACC}; }}
QCheckBox::indicator:checked {{ background:{ACC}; border-color:{ACC}; {chk} }}
QRadioButton::indicator {{ width:15px; height:15px; border-radius:8px; border:1px solid {BORDER_STRONG}; background:{CARD}; }}
QRadioButton::indicator:hover {{ border-color:{ACC}; }}
QRadioButton::indicator:checked {{ background:{ACC}; border:1px solid {ACC}; {dot} }}

QProgressBar {{ background:{TONAL}; border:none; border-radius:3px; height:6px; text-align:center; color:transparent; }}
QProgressBar::chunk {{ background:{ACC}; border-radius:3px; }}
QSlider::groove:horizontal {{ background:{TONAL}; height:4px; border-radius:2px; }}
QSlider::sub-page:horizontal {{ background:{ACC}; border-radius:2px; }}
QSlider::handle:horizontal {{ background:{CARD}; border:2px solid {ACC}; width:12px; height:12px; margin:-7px 0; border-radius:8px; }}

QTreeWidget, QTreeView, QListWidget {{ background:{CARD}; border:1px solid {BORDER}; border-radius:12px; padding:4px; outline:none; }}
QTreeWidget::item, QTreeView::item, QListWidget::item {{ padding:7px 8px; border-radius:8px; }}
QTreeWidget::item:hover, QTreeView::item:hover, QListWidget::item:hover {{ background:{CLICKABLE_HOVER}; }}
QTreeWidget::item:selected, QTreeView::item:selected, QListWidget::item:selected {{ background:{ACC_BG}; color:{INK}; }}
QHeaderView::section {{ background:{SURFACE}; border:none; border-bottom:1px solid {BORDER}; padding:7px 8px; font-weight:600; color:{MUTED}; }}

QMenu {{ background:{CARD}; border:1px solid {BORDER}; border-radius:12px; padding:5px; }}
QMenu::item {{ padding:7px 22px 7px 12px; border-radius:7px; }}
QMenu::item:selected {{ background:{ACC_BG}; color:{ACC_TEXT}; }}
QMenu::separator {{ height:1px; background:{BORDER}; margin:5px 8px; }}
QScrollArea {{ background:transparent; border:none; }}
QScrollBar:vertical {{ background:transparent; width:10px; margin:2px; }}
QScrollBar::handle:vertical {{ background:{SCROLL}; border-radius:3px; min-height:28px; margin:0 2px; }}
QScrollBar::handle:vertical:hover {{ background:{SCROLL_HOVER}; }}
QScrollBar:horizontal {{ background:transparent; height:10px; margin:2px; }}
QScrollBar::handle:horizontal {{ background:{SCROLL}; border-radius:3px; min-width:28px; margin:2px 0; }}
QScrollBar::handle:horizontal:hover {{ background:{SCROLL_HOVER}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width:0; height:0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background:transparent; }}
QSplitter::handle {{ background:{BORDER}; }}
QToolTip {{ background:#1E293B; color:#FFFFFF; border:none; border-radius:6px; padding:6px 9px; font-size:11px; }}

/* 对话框 / 向导：与主窗口同一套圆角、按钮层级 */
QDialog, QInputDialog, QProgressDialog, QWizard {{ background:{BG}; }}
QDialog QLabel {{ background:transparent; }}
QDialogButtonBox QPushButton, QDialog QPushButton {{ min-width:84px; min-height:20px; }}
QDialogButtonBox QPushButton:default {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 {ACC_TOP},stop:1 {ACC_FILL}); color:#FFFFFF; border:1px solid {ACC_FILL_HOVER}; font-weight:650; }}
QDialogButtonBox QPushButton:default:hover {{ background:{ACC_FILL_HOVER}; }}
QTabBar::close-button {{ subcontrol-position:right; }}
QStatusBar {{ background:{SIDEBAR_BG}; border-top:1px solid {BORDER}; color:{MUTED}; }}
"""
