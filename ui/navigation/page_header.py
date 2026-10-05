# -*- coding: utf-8 -*-
"""各功能页顶部的统一标题栏：标题 + 说明 + 可点击的流程步骤 + 「下一步」。

流程步骤让用户随时知道自己在哪一步，并能一键跳到任意一步；
「下一步」按钮把最常见的操作缩短为一次点击。
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ui.common.styling import accent_button
from ui.navigation.flow import FLOW, flow_position, next_section, step_state
from ui.theme.tokens import ACC_BG, ACC_TEXT, BG, CLICKABLE_HOVER, INK, MUTED, SUBTLE, SUCCESS

PAGE_INFO = {
    1: ("页面管理", ""),
    2: ("OCR 识别", "图片 OCR 逐列识别，或直接读取 PDF 内嵌文字层"),
    3: ("格式处理", "把校对后的文字整理成章节、注音与版式"),
    4: ("文字校对", "OCR 对比裁决分歧，图文对照逐句核对原图"),
    5: ("EPUB 生成", "设置书籍信息与样式，导出可阅读的 EPUB"),
    6: ("设置", "模型、设备、界面与快捷键"),
}
_NEXT_LABEL = {idx: label for idx, label in FLOW}
# 这些页面自带分区页签，不再显示标题栏（OCR 识别）
HEADERLESS = frozenset({2, 4})


def _chip_style(state: str) -> str:
    if state == "current":
        return (f"QPushButton{{background:{ACC_BG};color:{ACC_TEXT};border:none;border-radius:10px;"
                "padding:5px 12px;font-weight:700;}")
    color = SUCCESS if state == "done" else MUTED
    return (f"QPushButton{{background:transparent;color:{color};border:none;border-radius:10px;"
            f"padding:5px 12px;font-weight:600;}}QPushButton:hover{{background:{CLICKABLE_HOVER};color:{INK};}}")


class PageHeader(QWidget):
    navigate = Signal(int)          # 请求跳转到侧边栏索引（由主窗口接到 _goto）

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("pageHeader")
        self.setStyleSheet(f"QWidget#pageHeader{{background:{BG};border:none;}}")
        row = QHBoxLayout(self)
        row.setContentsMargins(28, 18, 28, 6)
        row.setSpacing(12)
        text = QVBoxLayout()
        text.setSpacing(3)
        self._title = QLabel("")
        self._title.setStyleSheet(f"color:{INK};font-size:20px;font-weight:750;background:none;")
        self._sub = QLabel("")
        self._sub.setStyleSheet(f"color:{MUTED};font-size:12px;background:none;")
        text.addWidget(self._title)
        text.addWidget(self._sub)
        row.addLayout(text, 1)

        self._steps_host = QWidget()
        steps = QHBoxLayout(self._steps_host)
        steps.setContentsMargins(0, 0, 0, 0)
        steps.setSpacing(2)
        self._chips: dict[int, QPushButton] = {}
        for pos, (section, label) in enumerate(FLOW):
            if pos:
                sep = QLabel("›")
                sep.setStyleSheet(f"color:{SUBTLE};background:none;font-size:13px;")
                steps.addWidget(sep)
            chip = QPushButton(label)
            chip.setCursor(QCursor(Qt.PointingHandCursor))
            chip.setToolTip(f"跳到「{PAGE_INFO[section][0]}」")
            chip.clicked.connect(lambda _=False, s=section: self.navigate.emit(s))
            self._chips[section] = chip
            steps.addWidget(chip)
        # Phase44: the sidebar already carries navigation.  The duplicated
        # workflow chips in the upper-right consumed horizontal space on every
        # page and made compact workspaces feel cramped, so keep the widgets only
        # as compatibility handles and never render them.
        self._steps_host.setVisible(False)

        self._next_btn = accent_button("下一步 →")
        self._next_btn.setMinimumHeight(34)
        self._next_btn.clicked.connect(self._go_next)
        self._next_btn.setVisible(False)
        self._section = 0
        self.set_section(0)

    def _go_next(self) -> None:
        target = next_section(self._section)
        if target is not None:
            self.navigate.emit(target)

    def set_section(self, index: int) -> None:
        index = int(index)
        self._section = index
        info = PAGE_INFO.get(index)
        self.setVisible(info is not None and index not in HEADERLESS)   # 工作区有项目横幅、OCR 页有分区页签
        if not info:
            return
        self._title.setText(info[0])
        self._sub.setText(info[1])
        self._sub.setVisible(bool(info[1]))
        in_flow = flow_position(index) is not None
        self._steps_host.setVisible(False)
        for section, chip in self._chips.items():
            state = step_state(section, index)
            chip.setStyleSheet(_chip_style(state))
            chip.setText(("✓ " if state == "done" else "") + dict(FLOW)[section])
        target = next_section(index)
        self._next_btn.setVisible(False)
        if target is not None:
            self._next_btn.setText(f"下一步：{_NEXT_LABEL[target]} →")
