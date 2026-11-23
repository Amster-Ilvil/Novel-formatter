"""Secondary and selected button colors for the text review workspace."""
from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QPushButton, QToolButton, QWidget

from ui.theme.tokens import ACC, ACC_BG, ACC_HOVER, ACC_TEXT, BORDER, BORDER_STRONG, CARD, INK, SUBTLE


def style_review_button(button: QPushButton | QToolButton | QCheckBox) -> None:
    """Apply the review palette while preserving the control's local sizing."""
    if button.property("role") == "primary":
        return
    if isinstance(button, QToolButton):
        selector = "QToolButton"
    elif isinstance(button, QCheckBox):
        selector = "QCheckBox"
    else:
        selector = "QPushButton"
    button.setStyleSheet(button.styleSheet() + f"""
{selector} {{ background-color:{CARD}; color:{INK}; border:1px solid {BORDER}; }}
{selector}:hover, {selector}:focus {{ border-color:{ACC}; color:{ACC_TEXT}; }}
{selector}:pressed {{ background-color:{ACC_BG}; border-color:{ACC_HOVER}; }}
{selector}:checked {{ background-color:{ACC_BG}; color:{ACC_TEXT}; border-color:{ACC}; font-weight:700; }}
{selector}:disabled {{ background-color:{CARD}; color:{SUBTLE}; border-color:{BORDER}; }}
{selector}:disabled:checked {{ background-color:{ACC_BG}; border-color:{BORDER_STRONG}; }}
""")


def style_review_buttons(root: QWidget) -> None:
    for button in root.findChildren(QPushButton):
        style_review_button(button)
    for button in root.findChildren(QToolButton):
        style_review_button(button)
