from __future__ import annotations
from PySide6.QtWidgets import QWidget, QSizePolicy, QVBoxLayout, QPushButton, QComboBox, QLineEdit, QPlainTextEdit
from .flow_layout import FlowLayout

LIGHT_LOG_STYLE = '''
QPlainTextEdit {
    background: #FFFFFF;
    color: #34343A;
    border: none;
    border-radius: 0;
    padding: 10px 14px;
    font-family: "SF Mono", "JetBrains Mono", "Menlo", "Consolas", monospace;
    font-size: 12px;
    selection-background-color: #DCE8FF;
    selection-color: #1D1D1F;
}
'''

def repolish(widget):
    widget.style().unpolish(widget); widget.style().polish(widget); widget.update()

def apply_light_log_style(editor: QPlainTextEdit) -> None:
    editor.setStyleSheet(LIGHT_LOG_STYLE)

def preserve_button_text(button: QPushButton) -> None:
    button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
    button.setMinimumHeight(36)
    fm = button.fontMetrics(); text_width = fm.horizontalAdvance(button.text())
    icon_width = button.iconSize().width() if not button.icon().isNull() else 0
    button.setMinimumWidth(text_width + icon_width + 34)

def configure_combo(combo: QComboBox) -> None:
    combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
    combo.setMinimumContentsLength(12)
    combo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

def configure_line_edit(line_edit: QLineEdit) -> None:
    line_edit.setMinimumWidth(140)
    line_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

class ResponsiveToolBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.flow = FlowLayout(self)
        self.flow.setContentsMargins(12, 8, 12, 8)
        self.flow.setHorizontalSpacing(8)
        self.flow.setVerticalSpacing(8)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
    def add_widget(self, widget):
        widget.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.flow.addWidget(widget)
