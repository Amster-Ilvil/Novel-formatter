from __future__ import annotations

from pathlib import Path
import re
import tempfile

from PySide6.QtCore import Property, QEasingCurve, QPropertyAnimation, QSize, Qt
from PySide6.QtGui import QColor, QCursor, QIcon, QPainter, QPen, QBrush, QFont
from PySide6.QtWidgets import (
    QAbstractButton, QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
    QVBoxLayout, QWidget, QToolButton,
)

from ui.design.metrics import (
    BUTTON_HEIGHT, BUTTON_RADIUS, CARD_PAD_X, CARD_PAD_Y, CARD_RADIUS,
    FONT_CARD_TITLE, FONT_SMALL, SIDEBAR_ITEM_HEIGHT, SIDEBAR_ITEM_RADIUS,
)
from ui.theme.tokens import (
    ACC, BORDER, CARD, CLICKABLE_HOVER, DISABLED_FG, INK, MUTED, SIDEBAR_BG,
    TONAL, DARK_PAIRS,
)




def _paint_token(widget: QWidget, light_value: str) -> str:
    """Return the theme-aware colour for custom-painted shared widgets."""
    try:
        dark = widget.palette().window().color().lightness() < 128
    except Exception:
        dark = False
    return DARK_PAIRS.get(light_value, light_value) if dark else light_value


class DesignCard(QFrame):
    """Reference card with one shared geometry contract across every page."""

    def __init__(self, parent=None, *, radius: int = CARD_RADIUS, padded: bool = True):
        super().__init__(parent)
        self.setObjectName("nfDesignCard")
        self.setProperty("nfDesignCard", True)
        self.setStyleSheet(
            f"QFrame#nfDesignCard{{background:{CARD};border:1px solid {BORDER};"
            f"border-radius:{int(radius)}px;}}"
        )
        self._body = QVBoxLayout(self)
        if padded:
            self._body.setContentsMargins(CARD_PAD_X, CARD_PAD_Y, CARD_PAD_X, CARD_PAD_Y)
        else:
            self._body.setContentsMargins(0, 0, 0, 0)
        self._body.setSpacing(8)

    @property
    def body(self) -> QVBoxLayout:
        return self._body


class SectionLabel(QLabel):
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setObjectName("nfSectionLabel")
        self.setStyleSheet(
            f"QLabel#nfSectionLabel{{background:transparent;border:none;color:{MUTED};"
            f"font-size:{FONT_SMALL}px;font-weight:700;}}"
        )


def tinted_svg_icon(path: Path, color: str) -> QIcon:
    """Deterministic monochrome SVG recolor used by the shared navigation."""
    if not path.exists():
        return QIcon()
    cache = Path(tempfile.gettempdir()) / "novel_formatter_icons"
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / f"{path.stem}_{color.lstrip('#')}.svg"
    try:
        source = path.read_text(encoding="utf-8")
        source = re.sub(r'stroke="#[0-9A-Fa-f]{3,8}"', f'stroke="{color}"', source)
        source = re.sub(r'fill="#[0-9A-Fa-f]{3,8}"', f'fill="{color}"', source)
        out.write_text(source, encoding="utf-8")
        return QIcon(str(out))
    except Exception:
        return QIcon(str(path))


class DesignBrandLabel(QWidget):
    """Paint the product name directly so global QLabel styles cannot add a box."""

    def __init__(self, text: str = "Novel Formatter", parent=None):
        super().__init__(parent)
        self._text = str(text)
        self.setFixedHeight(34)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

    def sizeHint(self) -> QSize:
        return QSize(148, 34)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.TextAntialiasing, True)
        font = QFont(self.font())
        font.setPixelSize(13)
        font.setWeight(QFont.Bold)
        p.setFont(font)
        p.setPen(QColor(_paint_token(self, INK)))
        p.drawText(self.rect(), Qt.AlignLeft | Qt.AlignVCenter, self._text)
        p.end()


class DesignNavButton(QAbstractButton):
    """Pixel-stable sidebar row painted independently of platform styles.

    QToolButton/QPushButton both allow native/global style rules to re-centre
    the icon+text block or paint inactive pills.  The supplied masters use a
    fixed left edge, so this control paints the row itself and keeps the same
    geometry on macOS, Windows and the Linux golden-test runner.
    """

    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        self.setText(text)
        self.setCheckable(True)
        self.setAutoExclusive(True)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.setFixedHeight(SIDEBAR_ITEM_HEIGHT)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setAttribute(Qt.WA_Hover, True)

    def sizeHint(self) -> QSize:
        return QSize(150, SIDEBAR_ITEM_HEIGHT)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = self.rect().adjusted(0, 0, -1, -1)
        checked = self.isChecked()
        hovered = self.underMouse() and self.isEnabled()

        if checked:
            p.setBrush(QBrush(QColor(_paint_token(self, CARD))))
            p.setPen(QPen(QColor(_paint_token(self, BORDER)), 1))
            p.drawRoundedRect(rect, SIDEBAR_ITEM_RADIUS, SIDEBAR_ITEM_RADIUS)
        elif hovered:
            p.setBrush(QBrush(QColor(_paint_token(self, CLICKABLE_HOVER))))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(rect, SIDEBAR_ITEM_RADIUS, SIDEBAR_ITEM_RADIUS)

        icon = self.icon()
        icon_w = 16 if not icon.isNull() else 0
        left = 16
        if icon_w:
            y = (self.height() - icon_w) // 2
            mode = QIcon.Mode.Disabled if not self.isEnabled() else QIcon.Mode.Normal
            state = QIcon.State.On if checked else QIcon.State.Off
            icon.paint(p, left, y, icon_w, icon_w, Qt.AlignCenter, mode, state)
            text_x = left + icon_w + 10
        else:
            text_x = left

        font = QFont(self.font())
        font.setPixelSize(12)
        font.setWeight(QFont.Bold if checked else QFont.Medium)
        p.setFont(font)
        color = DISABLED_FG if not self.isEnabled() else (ACC if checked else MUTED)
        p.setPen(QColor(_paint_token(self, color)))
        p.drawText(text_x, 0, max(0, self.width() - text_x - 10), self.height(),
                   Qt.AlignLeft | Qt.AlignVCenter, self.text())
        p.end()


class DesignLogCollapseButton(QToolButton):
    """Compact deterministic chevron for the OCR log header.

    Earlier builds painted a miniature log sheet next to the chevron.  At
    Retina scale its two text rows looked like stray horizontal rules above
    the header.  The header already says “OCR 日志”, so keep only the
    expand/collapse affordance.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.setFixedSize(28, 28)
        self.setAutoRaise(True)
        self.setArrowType(Qt.UpArrow)
        self.setAttribute(Qt.WA_Hover, True)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        if self.underMouse() and self.isEnabled():
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(_paint_token(self, CLICKABLE_HOVER))))
            p.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 7, 7)
        token = ACC if self.underMouse() and self.isEnabled() else MUTED
        if not self.isEnabled():
            token = DISABLED_FG
        pen = QPen(QColor(_paint_token(self, token)), 1.8)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        cx, cy = self.width() // 2, self.height() // 2
        if self.arrowType() == Qt.DownArrow:
            p.drawLine(cx - 4, cy - 2, cx, cy + 2)
            p.drawLine(cx, cy + 2, cx + 4, cy - 2)
        else:
            p.drawLine(cx - 4, cy + 2, cx, cy - 2)
            p.drawLine(cx, cy - 2, cx + 4, cy + 2)
        p.end()


class DesignSwitch(QAbstractButton):
    """Painted switch matching the reference instead of native checkbox indicators.

    It exposes the normal QAbstractButton checked/toggled API so it can proxy an
    existing QCheckBox without introducing a second configuration path.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.setFixedSize(34, 20)
        self._progress = 0.0
        self._anim = QPropertyAnimation(self, b"progress", self)
        self._anim.setDuration(120)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self.toggled.connect(self._animate)

    def _animate(self, checked: bool):
        self._anim.stop()
        self._anim.setStartValue(self._progress)
        self._anim.setEndValue(1.0 if checked else 0.0)
        self._anim.start()

    def get_progress(self) -> float:
        return self._progress

    def set_progress(self, value: float) -> None:
        self._progress = float(value)
        self.update()

    progress = Property(float, get_progress, set_progress)

    def nextCheckState(self):
        super().nextCheckState()
        # A programmatic setChecked() can happen before an event loop exists.
        if self._anim.state() != QPropertyAnimation.Running:
            self._progress = 1.0 if self.isChecked() else 0.0
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        enabled = self.isEnabled()
        on = self._progress if enabled else 0.0
        track = QColor(ACC if on >= .5 else TONAL)
        if not enabled:
            track = QColor("#DCE6F3")
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(track))
        p.drawRoundedRect(0, 1, 34, 18, 9, 9)
        x = 3 + (16 * on)
        p.setBrush(QBrush(QColor("#FFFFFF" if enabled else "#F5F8FC")))
        p.drawEllipse(int(round(x)), 4, 12, 12)
        p.end()


class DesignButton(QPushButton):
    def __init__(self, text: str = "", parent=None, *, primary: bool = False):
        super().__init__(text, parent)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.setMinimumHeight(BUTTON_HEIGHT)
        self.setProperty("role", "primary" if primary else "secondary")
        self.setStyleSheet(
            f"QPushButton{{border-radius:{BUTTON_RADIUS}px;min-height:{BUTTON_HEIGHT-2}px;}}"
        )
