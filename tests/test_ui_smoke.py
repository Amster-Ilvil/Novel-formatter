"""Offscreen smoke test: the main window in English shows no CJK chrome.

Builds the real MainWindow in the same order as gui_pyside6.main(), applies
the English UI language and walks every live widget.  QSettings is redirected
to a throw-away INI file so the user's real preferences and last project are
never read or written.
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QMetaMethod, QSettings  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractButton, QApplication, QCheckBox, QComboBox, QGroupBox, QLabel,
    QLineEdit, QPlainTextEdit, QTabWidget, QTextEdit,
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_BUILD_BUDGET_SECONDS = 10.0


@pytest.fixture(scope="module")
def english_window():
    app = QApplication.instance() or QApplication(list())
    import gui_pyside6
    from ui.interface_preferences import InterfacePreferenceManager
    from ui.localization import LANG_EN, LANGUAGE_LABELS

    real_settings = QSettings
    ini_path = Path(tempfile.mkdtemp(prefix="nf_ui_smoke_")) / "settings.ini"

    class IsolatedSettings(real_settings):
        def __init__(self, *args, **kwargs):
            super().__init__(str(ini_path), real_settings.IniFormat)

    patched = [
        module for module in list(sys.modules.values())
        if module is not None and getattr(module, "QSettings", None) is real_settings
    ]
    for module in patched:
        setattr(module, "QSettings", IsolatedSettings)
    window = None
    try:
        started = time.perf_counter()
        window = gui_pyside6.MainWindow()
        manager = InterfacePreferenceManager(app, base_stylesheet=gui_pyside6.STYLE, language=LANG_EN)
        window.show()
        for _ in range(5):
            app.processEvents()
        elapsed = time.perf_counter() - started
        yield app, window, manager, elapsed, frozenset(LANGUAGE_LABELS.values())
    finally:
        if window is not None:
            window.close()
            window.deleteLater()
            app.processEvents()
        for module in patched:
            setattr(module, "QSettings", real_settings)


def _collect_cjk(app, allowed: frozenset) -> list[tuple[str, str, str]]:
    hits: list[tuple[str, str, str]] = []

    def note(owner, kind: str, text: str) -> None:
        if text and _CJK.search(text) and text not in allowed:
            hits.append((type(owner).__name__, kind, text))

    for widget in app.allWidgets():
        if widget.property("nfNoTranslate"):
            continue
        if isinstance(widget, (QLabel, QAbstractButton)):
            note(widget, "text", widget.text())
        if isinstance(widget, QGroupBox):
            note(widget, "title", widget.title())
        if isinstance(widget, QTabWidget):
            for i in range(widget.count()):
                note(widget, "tabText", widget.tabText(i))
        if isinstance(widget, (QLineEdit, QTextEdit, QPlainTextEdit)):
            note(widget, "placeholderText", widget.placeholderText())
        if isinstance(widget, QComboBox) and not widget.property("nfNoTranslateItems"):
            for i in range(widget.count()):
                note(widget, "item", widget.itemText(i))
        note(widget, "toolTip", widget.toolTip())
        if widget.isWindow():
            note(widget, "windowTitle", widget.windowTitle())
        for action in widget.actions():
            if action is None or action.isSeparator():
                continue
            note(action, "action.text", action.text())
            note(action, "action.toolTip", action.toolTip())
    return sorted(set(hits))


def test_main_window_builds_quickly(english_window) -> None:
    _app, _window, _manager, elapsed, _allowed = english_window
    assert elapsed < _BUILD_BUDGET_SECONDS, f"main window took {elapsed:.2f}s"


def test_english_main_window_has_no_cjk_chrome(english_window) -> None:
    app, _window, _manager, _elapsed, allowed = english_window
    hits = _collect_cjk(app, allowed)
    report = "\n".join(f"{owner} {kind}: {text[:100]!r}" for owner, kind, text in hits[:40])
    assert hits == [], f"{len(hits)} CJK UI texts with language=en:\n{report}"


def test_language_switcher_keeps_native_names(english_window) -> None:
    app, _window, _manager, _elapsed, allowed = english_window
    combos = [
        w for w in app.allWidgets()
        if isinstance(w, QComboBox) and w.property("nfNoTranslateItems")
        and {w.itemText(i) for i in range(w.count())} == {"English", "简体中文", "日本語"}
    ]
    assert combos, "language switcher not found"
    switcher = next(iter(combos))
    assert switcher.itemText(0) == "English"
    assert allowed >= {switcher.itemText(i) for i in range(switcher.count())}


def test_epub_ltr_checkbox_is_wired(english_window) -> None:
    app, window, _manager, _elapsed, _allowed = english_window
    from ui.navigation.sidebar import SECTION_EPUB

    # The EPUB workspace is built lazily; navigate to it like the sidebar does.
    window._goto(SECTION_EPUB)
    for _ in range(3):
        app.processEvents()
    epub_tab = window._tab_epub
    checkbox = getattr(epub_tab, "_ltr_check", None)
    assert isinstance(checkbox, QCheckBox), "EPUB tab has no _ltr_check"
    # The checkbox sits in a collapsed panel; chrome is translated on Show, so
    # reveal its hidden ancestors the way expanding the panel would.
    hidden = []
    node = checkbox
    while node is not None and node is not epub_tab:
        if not node.isVisibleTo(epub_tab):
            hidden.append(node)
        node = node.parentWidget()
    for widget in reversed(hidden):
        widget.show()
    for _ in range(3):
        app.processEvents()
    assert checkbox.isVisible()
    assert checkbox.text() == "Convert to left-to-right (horizontal) layout"
    assert checkbox.isSignalConnected(QMetaMethod.fromSignal(checkbox.toggled)), "LTR checkbox has no toggled handler"
    initial = checkbox.isChecked()
    checkbox.setChecked(not initial)
    app.processEvents()
    checkbox.setChecked(initial)
    app.processEvents()
    assert checkbox.isChecked() == initial
