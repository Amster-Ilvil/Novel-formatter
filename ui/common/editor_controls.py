from __future__ import annotations

from PySide6.QtWidgets import QPlainTextEdit, QScrollBar, QSlider, QSpinBox, QDoubleSpinBox, QComboBox, QStyle, QStyleOptionSlider
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCursor

ACC = "#2F6BFF"

class NoWheelSpinBox(QSpinBox):
    """数值框只接受点击、键盘和按钮调整，滚轮交给外层页面。

    macOS 触控板或鼠标滚轮经过数值框时，Qt 默认会直接修改当前值。
    对 OCR 的安全上限这类低频设置很容易造成无意改动，因此明确忽略
    wheelEvent，让事件继续传给父级滚动区域。
    """

    def wheelEvent(self, event):
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """浮点数值框禁用鼠标滚轮/触控板调值。"""

    def wheelEvent(self, event):
        event.ignore()


class NoWheelComboBox(QComboBox):
    """下拉选项禁用鼠标滚轮/触控板切换，避免经过控件时误换模型。"""

    def wheelEvent(self, event):
        event.ignore()


class MouseWheelPlainTextEdit(QPlainTextEdit):
    """使用与 OCR 适配器滚动区一致的 Qt 原生滚轮/触控板行为。

    旧实现手工解释 ``pixelDelta``，但没有正确处理 macOS 的“自然滚动”
    ``event.inverted()``，在滚动条位于顶部时会把向下滑动误算成继续向上，
    看起来就像完全滚不动。这里先交给 QPlainTextEdit/QAbstractScrollArea 的
    原生实现；只有极少数组合下原生事件未改变滚动值时，才按 Qt 的方向语义
    做一次兜底。
    """

    def wheelEvent(self, event):
        if event.modifiers() & (Qt.ControlModifier | Qt.MetaModifier):
            super().wheelEvent(event)
            return

        bar = self.verticalScrollBar()
        before = bar.value()
        super().wheelEvent(event)

        # 原生滚动已经生效，行为与 OCR 适配器的 QScrollArea 完全一致。
        if bar.value() != before or bar.maximum() <= bar.minimum():
            return

        # 已经在顶端/底端：原生不动是正确行为（滚动到头就该停）。
        # 兜底若在这里继续算 delta，方向翻转的组合会把内容反向弹回去，
        # 表现为"滚到尽头不停、往回跳"。兜底只救"中段卡住"的场景。
        if before <= bar.minimum() or before >= bar.maximum():
            return

        pixel_y = event.pixelDelta().y()
        if pixel_y:
            delta = float(pixel_y)
        else:
            angle_y = event.angleDelta().y()
            if not angle_y:
                return
            delta = (angle_y / 120.0) * max(1, bar.singleStep()) * 3

        # Qt 在自然滚动开启时通过 inverted() 告知方向反转。
        if event.inverted():
            delta = -delta
        target = round(before - delta)
        target = max(bar.minimum(), min(bar.maximum(), target))
        if target != before:
            bar.setValue(target)
            event.accept()


class EditorPositionSlider(QSlider):
    """文本定位滑块，只允许按住滑块手柄后拖动。

    点击轨道、轻触滑轨或在滑块上滚动鼠标都不会让正文突然跳页。用户必须
    把指针放在灰色手柄上，按住鼠标左键，再上下拖动。文本框自身的滚轮和
    Mac 触控板滚动仍保持可用。
    """

    def __init__(self, parent=None):
        super().__init__(Qt.Vertical, parent)
        self.setTracking(True)
        self.setInvertedAppearance(True)
        self.setCursor(QCursor(Qt.OpenHandCursor))
        self.setToolTip("请按住灰色滑块手柄后上下拖动；点击轨道不会跳转")
        self.setMinimumWidth(16)
        self.setMaximumWidth(16)
        self._dragging_handle = False
        # 与全局滚动条同一套视觉语言：细浅轨 + 圆角胶囊手柄，拖动时点亮强调色
        self.setStyleSheet(
            "QSlider { background: transparent; }"
            "QSlider::groove:vertical {"
            " background: #EFEFF3; width: 6px; border-radius: 3px; margin: 2px 0; }"
            "QSlider::handle:vertical {"
            " background: #B4B4BC; border: none;"
            " width: 14px; height: 40px; margin: 0 -4px; border-radius: 7px; }"
            "QSlider::handle:vertical:hover { background: #90909A; }"
            f"QSlider::handle:vertical:pressed {{ background: {ACC}; }}"
            "QSlider::sub-page:vertical, QSlider::add-page:vertical { background: transparent; }"
            "QSlider:disabled { background: transparent; }"
            "QSlider::handle:vertical:disabled { background: #E3E3E8; }"
        )
        self.sliderPressed.connect(lambda: self.setCursor(QCursor(Qt.ClosedHandCursor)))
        self.sliderReleased.connect(lambda: self.setCursor(QCursor(Qt.OpenHandCursor)))

    def _handle_rect(self):
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        return self.style().subControlRect(
            QStyle.CC_Slider, option, QStyle.SC_SliderHandle, self
        )

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton or not self._handle_rect().contains(event.position().toPoint()):
            event.ignore()
            return
        self._dragging_handle = True
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self._dragging_handle:
            event.ignore()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if not self._dragging_handle:
            event.ignore()
            return
        super().mouseReleaseEvent(event)
        self._dragging_handle = False

    def wheelEvent(self, event):
        # 防止鼠标刚好停在滑块上时，滚轮造成大幅跳转。
        event.ignore()


def _editor_vertical_scrollbar(editor: QPlainTextEdit) -> QScrollBar:
    """Return a concrete QScrollBar during early PySide widget startup."""
    bar = editor.verticalScrollBar()
    if isinstance(bar, QScrollBar):
        return bar
    bar = QScrollBar(Qt.Vertical, editor)
    editor.setVerticalScrollBar(bar)
    return bar


def configure_drag_scrollbar(editor: QPlainTextEdit) -> None:
    """配置文本编辑器的滚轮基础行为。

    实际可拖拽定位由编辑器右侧的 :class:`EditorPositionSlider` 提供。内部
    QScrollBar 继续保存滚动范围，但隐藏，避免平台主题只显示两端箭头而没有
    可拖动滑块。鼠标滚轮和 Mac 触控板仍直接驱动内部滚动值。
    """
    editor.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    editor.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    editor.setFocusPolicy(Qt.StrongFocus)
    editor.setMouseTracking(True)
    bar = _editor_vertical_scrollbar(editor)
    bar.setTracking(True)
    bar.setSingleStep(3)
    bar.setPageStep(max(10, bar.pageStep()))


def attach_editor_position_slider(editor: QPlainTextEdit, slider: EditorPositionSlider) -> None:
    """双向同步文本编辑器与外置可拖拽定位滑块。"""
    bar = _editor_vertical_scrollbar(editor)

    def sync_range(minimum=None, maximum=None):
        slider.blockSignals(True)
        slider.setRange(bar.minimum(), bar.maximum())
        slider.setSingleStep(max(1, bar.singleStep()))
        slider.setPageStep(max(1, bar.pageStep()))
        slider.setValue(bar.value())
        slider.setEnabled(bar.maximum() > bar.minimum())
        slider.blockSignals(False)

    bar.rangeChanged.connect(sync_range)
    bar.valueChanged.connect(slider.setValue)
    slider.valueChanged.connect(bar.setValue)
    editor.blockCountChanged.connect(lambda _count: QTimer.singleShot(0, sync_range))
    editor._position_slider = slider
    QTimer.singleShot(0, sync_range)

