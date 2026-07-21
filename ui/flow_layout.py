from __future__ import annotations
from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QLayout, QSizePolicy, QStyle, QWidgetItem

class FlowLayout(QLayout):
    def __init__(self, parent=None, margin=0, hspacing=8, vspacing=8):
        super().__init__(parent)
        self._items = []
        self._hspacing = hspacing
        self._vspacing = vspacing
        self.setContentsMargins(margin, margin, margin, margin)
    def addItem(self, item): self._items.append(item)
    def addWidget(self, widget): self.addItem(QWidgetItem(widget))
    def count(self): return len(self._items)
    def itemAt(self, index): return self._items[index] if 0 <= index < len(self._items) else None
    def takeAt(self, index): return self._items.pop(index) if 0 <= index < len(self._items) else None
    def expandingDirections(self): return Qt.Orientations(Qt.Orientation(0))
    def hasHeightForWidth(self): return True
    def heightForWidth(self, width): return self._do_layout(QRect(0, 0, width, 0), True)
    def setGeometry(self, rect): super().setGeometry(rect); self._do_layout(rect, False)
    def sizeHint(self): return self.minimumSize()
    def minimumSize(self):
        size = QSize()
        for item in self._items: size = size.expandedTo(item.minimumSize())
        l,t,r,b = self.getContentsMargins(); size += QSize(l+r, t+b); return size
    def setHorizontalSpacing(self, spacing): self._hspacing = spacing
    def setVerticalSpacing(self, spacing): self._vspacing = spacing
    def _smart_spacing(self, pm):
        parent = self.parent()
        if parent is None: return -1
        if parent.isWidgetType(): return parent.style().pixelMetric(pm, None, parent)
        return parent.spacing()
    def _do_layout(self, rect, test_only):
        l,t,r,b = self.getContentsMargins(); effective = rect.adjusted(l,t,-r,-b)
        x, y, line_height = effective.x(), effective.y(), 0
        for item in self._items:
            hint = item.sizeHint(); space_x = self._hspacing if self._hspacing >= 0 else self._smart_spacing(QStyle.PM_LayoutHorizontalSpacing)
            space_y = self._vspacing if self._vspacing >= 0 else self._smart_spacing(QStyle.PM_LayoutVerticalSpacing)
            next_x = x + hint.width() + space_x
            if next_x - space_x > effective.right() and line_height > 0:
                x = effective.x(); y += line_height + space_y; next_x = x + hint.width() + space_x; line_height = 0
            if not test_only: item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x; line_height = max(line_height, hint.height())
        return y + line_height - rect.y() + b
