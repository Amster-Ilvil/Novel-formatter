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
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import zipfile

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent, QMetaMethod, QSettings  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractButton, QApplication, QCheckBox, QComboBox, QGroupBox, QLabel,
    QLineEdit, QPlainTextEdit, QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_BUILD_BUDGET_SECONDS = 10.0


@pytest.fixture
def english_window(tmp_path):
    app = QApplication.instance() or QApplication(list())
    import gui_pyside6
    from ui.interface_preferences import InterfacePreferenceManager
    from ui.localization import LANG_EN, LANGUAGE_LABELS

    real_settings = QSettings
    ini_path = tmp_path / "settings.ini"

    class IsolatedSettings(real_settings):
        def __init__(self, *args, **kwargs):
            super().__init__(str(ini_path), real_settings.IniFormat)

    previous_manager = getattr(app, "_nf_interface_preference_manager", None)
    previous_stylesheet = app.styleSheet()
    previous_palette = app.palette()
    previous_properties = {
        name: app.property(name) for name in ("nfLanguage", "nfTheme", "nfDarkMode")
    }
    previous_quit = app.quitOnLastWindowClosed()
    if previous_manager is not None:
        app.removeEventFilter(previous_manager)
    app._nf_interface_preference_manager = None
    app.setQuitOnLastWindowClosed(False)
    for module in list(sys.modules.values()):
        if module is not None and getattr(module, "QSettings", None) is real_settings:
            setattr(module, "QSettings", IsolatedSettings)
    window = None
    manager = None
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
        try:
            if window is not None:
                for _ in range(5):
                    app.processEvents()
                closed = window.close()
                for _ in range(5):
                    app.processEvents()
                window.deleteLater()
                assert closed, "Main window rejected cleanup"
        finally:
            if manager is not None:
                app.removeEventFilter(manager)
                manager.deleteLater()
            app._nf_interface_preference_manager = None
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            app.processEvents()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            # Include modules imported lazily while navigating this window.
            for module in list(sys.modules.values()):
                if module is not None and getattr(module, "QSettings", None) is IsolatedSettings:
                    setattr(module, "QSettings", real_settings)
            app.setStyleSheet(previous_stylesheet)
            app.setPalette(previous_palette)
            for name, value in previous_properties.items():
                app.setProperty(name, value)
            app.setQuitOnLastWindowClosed(previous_quit)
            app._nf_interface_preference_manager = previous_manager
            if previous_manager is not None:
                app.installEventFilter(previous_manager)


def _collect_cjk(window, allowed: frozenset) -> list[tuple[str, str, str]]:
    hits: list[tuple[str, str, str]] = []

    def note(owner, kind: str, text: str) -> None:
        if text and _CJK.search(text) and text not in allowed:
            hits.append((type(owner).__name__, kind, text))

    for widget in [window, *window.findChildren(QWidget)]:
        if not widget.isVisible() or widget.property("nfNoTranslate"):
            continue
        if isinstance(widget, (QLabel, QAbstractButton)):
            note(widget, "text", widget.text())
        if isinstance(widget, QGroupBox):
            note(widget, "title", widget.title())
        if isinstance(widget, QTabWidget):
            for i in range(widget.count()):
                if widget.isTabVisible(i):
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
            if action is None or action.isSeparator() or not action.isVisible():
                continue
            note(action, "action.text", action.text())
            note(action, "action.toolTip", action.toolTip())
    return sorted(set(hits))


def test_main_window_builds_quickly(english_window) -> None:
    _app, _window, _manager, elapsed, _allowed = english_window
    assert elapsed < _BUILD_BUDGET_SECONDS, f"main window took {elapsed:.2f}s"


def test_english_main_window_has_no_cjk_chrome(english_window) -> None:
    app, window, _manager, _elapsed, allowed = english_window
    from ui.navigation.sidebar import ReferenceSectionHost, SECTION_EPUB

    hits = []

    def check_visible(context: str) -> None:
        for _ in range(5):
            app.processEvents()
        hits.extend(
            (context, owner, kind, text)
            for owner, kind, text in _collect_cjk(window, allowed)
        )

    visited = set()
    for section in range(window._stack.count()):
        window._goto(section)
        visited.add(section)
        check_visible(f"section={section}")
        if section == SECTION_EPUB:
            window._tab_epub._set_epub_compact_mode(False)
            check_visible(f"section={section}, advanced")
        for host in window.findChildren(ReferenceSectionHost):
            if not host.isVisible():
                continue
            for index in range(host.tabs.count()):
                host.set_current_index(index)
                check_visible(f"section={section}, subtab={index}")
        for tabs in window.findChildren(QTabWidget):
            if not tabs.isVisible():
                continue
            for index in range(tabs.count()):
                if tabs.isTabVisible(index) and tabs.isTabEnabled(index):
                    tabs.setCurrentIndex(index)
                    check_visible(f"section={section}, tab={tabs.tabText(index)!r}")
    assert SECTION_EPUB in visited
    assert window._lazy_workspace_if_loaded("epub") is not None
    report = "\n".join(
        f"{context}: {owner} {kind}: {text!r}"
        for context, owner, kind, text in sorted(set(hits))
    )
    assert hits == [], f"{len(hits)} CJK UI texts with language=en:\n{report}"


def test_language_switcher_keeps_native_names(english_window) -> None:
    _app, window, _manager, _elapsed, allowed = english_window
    combos = [
        w for w in window.findChildren(QComboBox)
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


def test_page_header_translates_immediately_and_retains_source(english_window) -> None:
    app, window, manager, _elapsed, _allowed = english_window
    from ui.localization import LANG_EN, LANG_JA, LANG_ZH, translate_text
    from ui.navigation.page_header import PAGE_INFO
    from ui.navigation.sidebar import SECTION_EPUB

    header = window._page_header
    assert window._lazy_workspace_if_loaded("epub") is None
    connections = [
        (window._sidebar, "2section_changed(int)"),
        (header, "2navigate(int)"),
        (header._next_btn, "2clicked(bool)"),
        *((chip, "2clicked(bool)") for chip in header._chips.values()),
    ]
    receiver_counts = [obj.receivers(signal) for obj, signal in connections]
    assert all(count > 0 for count in receiver_counts)
    epub_tab = None

    for navigation_language in (LANG_JA, LANG_ZH, LANG_EN):
        manager.apply_language(navigation_language)
        for section in (1, SECTION_EPUB, 3, 6, SECTION_EPUB, 1):
            header.navigate.emit(section)
            assert window._stack.currentIndex() == section
            assert window._page_header is header
            title, subtitle = PAGE_INFO[section]
            assert header._title.text() == translate_text(title, navigation_language)
            assert header._sub.text() == translate_text(subtitle, navigation_language)
            for language in (LANG_EN, LANG_JA, LANG_ZH, navigation_language):
                manager.apply_language(language)
                for _ in range(3):
                    app.processEvents()
                assert header.isVisible()
                assert header._title.text() == translate_text(title, language)
                assert header._sub.text() == translate_text(subtitle, language)
                assert header._title._nf_i18n_source_text == title
                assert header._sub._nf_i18n_source_text == subtitle
            if section == SECTION_EPUB:
                loaded = window._lazy_workspace_if_loaded("epub")
                assert loaded is not None
                if epub_tab is None:
                    epub_tab = loaded
                assert loaded is epub_tab
            assert [obj.receivers(signal) for obj, signal in connections] == receiver_counts
            assert app._nf_interface_preference_manager is manager


def test_dynamic_widget_properties_translate_without_changing_content(english_window) -> None:
    app, window, manager, _elapsed, _allowed = english_window
    from ui.localization import LANG_EN, LANG_JA, translate_text

    panel = QWidget(window)
    layout = QVBoxLayout(panel)
    group = QGroupBox("Initial title")
    group_layout = QVBoxLayout(group)
    label = QLabel("Initial label")
    editor = QLineEdit()
    editor.setText("日本語の本文")
    identifiers = QComboBox()
    identifiers.setProperty("nfNoTranslateItems", True)
    identifiers.addItem("日本語の識別子", "stable-id")
    group_layout.addWidget(label)
    group_layout.addWidget(editor)
    group_layout.addWidget(identifiers)
    layout.addWidget(group)
    panel.resize(400, 200)
    panel.show()
    app.processEvents()

    source = "页面管理"
    label.setText(source)
    group.setTitle(source)
    window.setWindowTitle(source)
    editor.setPlaceholderText(source)
    label.setToolTip(source)
    for _ in range(5):
        app.processEvents()
    label.repaint()
    group.repaint()
    editor.repaint()

    for language in (LANG_EN, LANG_JA, LANG_EN):
        if language != LANG_EN or manager.language != LANG_EN:
            manager.apply_language(language)
        expected = translate_text(source, language)
        assert label.text() == expected
        assert group.title() == expected
        assert window.windowTitle() == expected
        assert editor.placeholderText() == expected
        assert label.toolTip() == expected
        assert editor.text() == "日本語の本文"
        assert identifiers.currentText() == "日本語の識別子"
        assert identifiers.currentData() == "stable-id"


def test_compact_epub_direction_tracks_ltr_override(english_window) -> None:
    app, window, _manager, _elapsed, _allowed = english_window
    from ui.navigation.sidebar import SECTION_EPUB

    window._goto(SECTION_EPUB)
    tab = window._tab_epub
    tab._set_epub_compact_mode(True)
    tab._ltr_check.setChecked(False)
    tab._vert_radio.setChecked(True)
    app.processEvents()
    assert tab._compact_mode_combo.isVisible()
    assert tab._compact_mode_combo.currentData() == "vertical"
    assert tab._pills["mode"].text() == "↕ Vertical"

    tab._ltr_check.setChecked(True)
    app.processEvents()
    assert tab._vert_radio.isChecked()
    assert tab._compact_mode_combo.currentData() == "horizontal"
    assert tab._pills["mode"].text() == "↕ Horizontal"

    tab._ltr_check.setChecked(False)
    app.processEvents()
    assert tab._compact_mode_combo.currentData() == "vertical"
    assert tab._pills["mode"].text() == "↕ Vertical"

    tab._ltr_check.setChecked(True)
    tab._compact_mode_combo.setCurrentIndex(tab._compact_mode_combo.findData("vertical"))
    app.processEvents()
    assert not tab._ltr_check.isChecked()
    assert tab._vert_radio.isChecked()
    assert tab._compact_mode_combo.currentData() == "vertical"


@pytest.mark.parametrize("selected", [False, True], ids=["ltr-off", "ltr-on"])
def test_epub_view_switch_preserves_ltr_and_connections(english_window, monkeypatch, selected) -> None:
    app, window, _manager, _elapsed, _allowed = english_window
    from ui.navigation.sidebar import SECTION_EPUB

    window._goto(SECTION_EPUB)
    tab = window._tab_epub
    tab._vert_radio.setChecked(True)
    tab._ltr_check.setChecked(selected)
    sync = Mock(wraps=tab._sync_compact_epub_mode_from_advanced)
    monkeypatch.setattr(tab, "_sync_compact_epub_mode_from_advanced", sync)
    connections = [
        (tab._ltr_check, "2toggled(bool)"),
        (tab._vert_radio, "2toggled(bool)"),
        (tab._horiz_radio, "2toggled(bool)"),
        (tab._compact_mode_combo, "2currentIndexChanged(int)"),
    ]
    receiver_counts = [obj.receivers(signal) for obj, signal in connections]
    assert all(count > 0 for count in receiver_counts)

    for compact in (False, True, False, True, False, True):
        tab._set_epub_compact_mode(compact)
        app.processEvents()
        assert tab._epub_compact_view.isVisible() == compact
        assert tab._epub_advanced_left.isVisible() == (not compact)
        assert tab._ltr_check.isChecked() == selected
        assert tab._vert_radio.isChecked()
        assert tab._compact_mode_combo.currentData() == ("horizontal" if selected else "vertical")
        assert tab._pills["mode"].text() == ("↕ Horizontal" if selected else "↕ Vertical")
        assert [obj.receivers(signal) for obj, signal in connections] == receiver_counts
        sync.reset_mock()
        tab._ltr_check.setChecked(not selected)
        app.processEvents()
        sync.assert_called_once_with()
        sync.reset_mock()
        tab._ltr_check.setChecked(selected)
        app.processEvents()
        sync.assert_called_once_with()


@pytest.mark.parametrize("compact", [False, True], ids=["advanced", "compact"])
@pytest.mark.parametrize("selected", [False, True], ids=["ltr-off", "ltr-on"])
def test_epub_ui_exports_selected_direction(english_window, monkeypatch, tmp_path, compact, selected) -> None:
    app, window, _manager, _elapsed, _allowed = english_window
    import builder.epub_builder as exporter
    import ui.epub.tab as epub_ui
    import utils.epub_validator as validator
    from models.document import Block, BlockType, UnifiedDocument
    from ui.navigation.sidebar import SECTION_EPUB

    window._goto(SECTION_EPUB)
    tab = window._tab_epub
    doc = UnifiedDocument()
    doc.metadata.title = "Direction regression"
    doc.metadata.author = "Test author"
    doc.blocks = [
        Block(BlockType.CHAPTER, "第一章"),
        Block(BlockType.PARAGRAPH, "これは本文です。"),
        Block(BlockType.PARAGRAPH, "︻完︼"),
    ]
    original_texts = [block.text for block in doc.blocks]
    tab._doc = doc
    tab._source_kind = "format"
    tab._title_edit.setText(doc.metadata.title)
    tab._author_edit.setText(doc.metadata.author)
    template_index = tab._template_combo.findText("denki")
    assert template_index >= 0
    tab._template_combo.setCurrentIndex(template_index)
    tab._vert_radio.setChecked(True)
    tab._ltr_check.setChecked(selected)
    tab._set_epub_compact_mode(compact)
    app.processEvents()
    tab._preview_timer.stop()

    path = tmp_path / "direction.epub"
    monkeypatch.setattr(epub_ui.QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(path), "EPUB (*.epub)"))
    monkeypatch.setattr(epub_ui.QMessageBox, "warning", lambda *args, **kwargs: epub_ui.QMessageBox.Yes)
    thread_factory = Mock()
    monkeypatch.setattr(epub_ui, "threading", SimpleNamespace(Thread=thread_factory))
    build = Mock(wraps=exporter.build_epub)
    monkeypatch.setattr(exporter, "build_epub", build)
    external_check = Mock(return_value=SimpleNamespace(skipped=True, valid=True))
    monkeypatch.setattr(validator, "validate_epub", external_check)
    done, error = Mock(), Mock()
    monkeypatch.setattr(tab, "_on_build_done", done)
    monkeypatch.setattr(tab, "_on_build_error", error)

    try:
        tab._build_epub()
        thread_factory.assert_called_once()
        thread_factory.return_value.start.assert_called_once_with()
        assert tab._build_running
        # The worker must use the selection captured at export time.
        tab._ltr_check.setChecked(not selected)
        tab._preview_timer.stop()
        thread_factory.call_args.kwargs["target"]()
        app.processEvents()
        assert error.call_count == 0, error.call_args_list
        done.assert_called_once()
        build.assert_called_once()
        assert build.call_args.kwargs["text_direction"] == ("ltr" if selected else "keep")
        assert build.call_args.kwargs["vertical"] is True
        exported_doc, = build.call_args.args
        result, = done.call_args.args
        assert exported_doc is not doc
        assert result["quality_gate"].valid
        external_check.assert_called_once_with(str(path))
        assert path.is_file()

        with zipfile.ZipFile(path) as archive:
            opf = archive.read("EPUB/content.opf").decode("utf-8")
            css = archive.read("EPUB/styles/style.css").decode("utf-8")
            chapter = archive.read("EPUB/content/ch001.xhtml").decode("utf-8")
        direction = "ltr" if selected else "rtl"
        assert f'page-progression-direction="{direction}"' in opf
        if selected:
            assert "vertical-rl" not in css
            assert "writing-mode: horizontal-tb" in css
            assert '<body dir="ltr">' in chapter
            assert "【完】" in chapter
            assert "︻" not in chapter
        else:
            assert "writing-mode: vertical-rl" in css
            assert "︻完︼" in chapter
        assert [block.text for block in doc.blocks] == original_texts
        assert doc.metadata.title == "Direction regression"
    finally:
        tab._preview_timer.stop()
        tab._set_build_running(False)


def test_epub_controls_retranslate_to_japanese_and_english(english_window) -> None:
    app, window, manager, _elapsed, _allowed = english_window
    from PySide6.QtWidgets import QPushButton
    from ui.localization import LANG_EN, LANG_JA
    from ui.navigation.sidebar import SECTION_EPUB

    window._goto(SECTION_EPUB)
    tab = window._tab_epub
    tab._set_epub_compact_mode(False)
    tab._ltr_check.setChecked(True)
    clear = tab.findChild(QPushButton, "novelFormatterWorkspaceClear")
    assert clear is not None
    for language, mode_text, clear_text in (
        (LANG_EN, "↕ Horizontal", "Clear EPUB"),
        (LANG_JA, "↕ 横書き", "EPUBをクリア"),
        (LANG_EN, "↕ Horizontal", "Clear EPUB"),
    ):
        manager.apply_language(language)
        app.processEvents()
        assert clear.isVisible()
        assert tab._pills["mode"].text() == mode_text
        assert clear.text() == clear_text
        assert clear._nf_i18n_source_text == "清空EPUB"
