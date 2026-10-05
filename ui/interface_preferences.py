# -*- coding: utf-8 -*-
"""Runtime theme + language management for the PySide6 desktop UI.

This module is intentionally UI-only.  It never touches OCR configuration,
documents, model caches, worker state, project files or export data.
"""
from __future__ import annotations

import re
from typing import Iterable

from PySide6.QtCore import QObject, QEvent, QTimer
from PySide6.QtGui import QAction, QColor, QPalette
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QToolButton, QCheckBox,
    QRadioButton, QLineEdit, QComboBox, QTabWidget, QListWidget, QTableWidget,
    QMenu, QGroupBox, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox,
    QProgressBar, QTreeWidget,
)

from ui.localization import LANG_ZH, normalize_language, translate_text

from ui.theme_palette import (
    THEME_LIGHT, THEME_DARK, SUPPORTED_THEMES, normalize_theme, darken_stylesheet,
)


def dark_palette() -> QPalette:
    p = QPalette()
    p.setColor(QPalette.Window, QColor("#111318"))
    p.setColor(QPalette.WindowText, QColor("#E6EEF8"))
    p.setColor(QPalette.Base, QColor("#171B21"))
    p.setColor(QPalette.AlternateBase, QColor("#1F242B"))
    p.setColor(QPalette.ToolTipBase, QColor("#242A32"))
    p.setColor(QPalette.ToolTipText, QColor("#E6EEF8"))
    p.setColor(QPalette.Text, QColor("#E6EEF8"))
    p.setColor(QPalette.Button, QColor("#1B1F24"))
    p.setColor(QPalette.ButtonText, QColor("#E6EEF8"))
    p.setColor(QPalette.BrightText, QColor("#FFFFFF"))
    p.setColor(QPalette.Link, QColor("#6AAEFF"))
    p.setColor(QPalette.Highlight, QColor("#2F6BFF"))
    p.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    p.setColor(QPalette.PlaceholderText, QColor("#778392"))
    p.setColor(QPalette.Disabled, QPalette.Text, QColor("#778392"))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#778392"))
    p.setColor(QPalette.Disabled, QPalette.WindowText, QColor("#778392"))
    return p


class InterfacePreferenceManager(QObject):
    """Application-wide theme and UI language manager.

    Existing widget instances are retained.  We only change visible Qt chrome,
    so signals, combo ``itemData``, model IDs, OCR settings and document objects
    are unaffected by a language/theme switch.
    """

    _TEXT_TYPES = (QLabel, QPushButton, QToolButton, QCheckBox, QRadioButton)
    # Polish can arrive while a Python QWidget subclass is still being built.
    # Calling back into that temporary wrapper can make PySide cache it as a
    # plain QWidget. Show happens after construction and is the safe boundary.
    _LIFECYCLE_EVENTS = {QEvent.Show}
    _DYNAMIC_TEXT_TYPES = _TEXT_TYPES + (
        QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QListWidget, QTableWidget,
        QGroupBox, QSpinBox, QDoubleSpinBox, QProgressBar, QTreeWidget,
    )

    def __init__(self, app: QApplication, *, base_stylesheet: str = "", language: str = LANG_ZH, theme: str = THEME_LIGHT):
        super().__init__(app)
        self.app = app
        self.base_stylesheet = str(base_stylesheet or app.styleSheet() or "")
        self.language = normalize_language(language)
        self.theme = normalize_theme(theme)
        self._theme_guard: set[int] = set()
        self._translate_guard: set[int] = set()
        self._default_palette = QPalette(app.palette())
        app.installEventFilter(self)
        # Keep the Python manager as a normal attribute; Qt dynamic properties
        # are reserved for simple QVariant-compatible values (theme/language).
        app._nf_interface_preference_manager = self
        self.apply_theme(self.theme)
        self.apply_language(self.language)

    def apply_preferences(self, *, language: str | None = None, theme: str | None = None) -> None:
        if theme is not None:
            self.apply_theme(theme)
        if language is not None:
            self.apply_language(language)

    def apply_theme(self, theme: str) -> None:
        self.theme = normalize_theme(theme)
        self.app.setProperty("nfTheme", self.theme)
        self.app.setProperty("nfDarkMode", self.theme == THEME_DARK)
        if self.theme == THEME_DARK:
            self.app.setPalette(dark_palette())
            self.app.setStyleSheet(darken_stylesheet(self.base_stylesheet))
        else:
            self.app.setPalette(QPalette(self._default_palette))
            self.app.setStyleSheet(self.base_stylesheet)
        for widget in list(self.app.allWidgets()):
            self._apply_widget_theme(widget)

    def apply_language(self, language: str) -> None:
        self.language = normalize_language(language)
        self.app.setProperty("nfLanguage", self.language)
        for widget in list(self.app.allWidgets()):
            self._translate_widget(widget, force=True)

    def eventFilter(self, obj, event):
        try:
            et = event.type()
            if isinstance(obj, QWidget):
                # Leave editor-private scrollbars to Qt; attaching Python state
                # to their base wrapper can corrupt the PySide return type.
                if obj.metaObject().className() == "QScrollBar":
                    return False
                if et in self._LIFECYCLE_EVENTS:
                    if self.theme == THEME_DARK:
                        self._apply_widget_theme(obj)
                    if self.language != LANG_ZH:
                        self._translate_widget(obj, force=False)
                elif et == QEvent.UpdateRequest and self.language != LANG_ZH and isinstance(obj, self._DYNAMIC_TEXT_TYPES):
                    # setText()/setItemText() schedules an update.  Restrict the
                    # dynamic pass to text-bearing widgets so live OCR status
                    # messages switch language without rescanning theme/QSS on
                    # every paint event.
                    self._translate_widget(obj, force=False)
                elif et == QEvent.StyleChange and id(obj) not in self._theme_guard:
                    # Widget code may replace its inline QSS in either theme.
                    # Capture the external style as the new canonical light source
                    # even while the app is light; otherwise a later dark switch
                    # would resurrect the stale startup stylesheet. In dark mode
                    # the deferred pass also converts the just-written light QSS.
                    if obj.isVisible():
                        QTimer.singleShot(0, lambda w=obj: self._apply_widget_theme(w, accept_new_source=True))
        except RuntimeError:
            pass  # QWidget deleted between queued event and callback.
        return False

    def _apply_widget_theme(self, widget: QWidget, *, accept_new_source: bool = False) -> None:
        wid = id(widget)
        if wid in self._theme_guard:
            return
        self._theme_guard.add(wid)
        try:
            current = widget.styleSheet() or ""
            source = getattr(widget, "_nf_theme_source_style", None)
            applied = getattr(widget, "_nf_theme_applied_style", None)
            if source is None:
                source = current
                setattr(widget, "_nf_theme_source_style", source)
            elif applied is not None and current != applied and accept_new_source:
                # A business widget or the clickable contrast guard changed its
                # canonical light QSS after our previous pass. StyleChange is the
                # only path allowed to promote that external QSS to canonical
                # source, so ordinary theme re-application never mistakes stale
                # dark output for a new light stylesheet.
                source = current
                setattr(widget, "_nf_theme_source_style", source)
            target = darken_stylesheet(source) if self.theme == THEME_DARK else source
            if current != target:
                widget.setStyleSheet(target)
            setattr(widget, "_nf_theme_applied_style", target)
        finally:
            self._theme_guard.discard(wid)

    @staticmethod
    def _source_attr(obj, name: str, current: str) -> str:
        attr = f"_nf_i18n_source_{name}"
        applied_attr = f"_nf_i18n_applied_{name}"
        source = getattr(obj, attr, None)
        applied = getattr(obj, applied_attr, None)
        # A business widget may update a status after the language switch.  Use
        # the last text written by *this* manager rather than Unicode ranges to
        # distinguish that real update from our Japanese translation (Japanese
        # naturally contains Han characters too).
        if source is None or (applied is not None and current != applied):
            source = current
            setattr(obj, attr, source)
        return source or ""

    @staticmethod
    def _remember_applied(obj, name: str, value: str) -> None:
        setattr(obj, f"_nf_i18n_applied_{name}", value)

    def _translate_widget(self, widget: QWidget, *, force: bool) -> None:
        wid = id(widget)
        if wid in self._translate_guard:
            return
        self._translate_guard.add(wid)
        try:
            if widget.property("nfNoTranslate"):
                return
            if isinstance(widget, self._TEXT_TYPES):
                current = widget.text()
                source = self._source_attr(widget, "text", current)
                target = translate_text(source, self.language)
                if current != target:
                    widget.setText(target)
                self._remember_applied(widget, "text", target)
            if isinstance(widget, (QLineEdit, QTextEdit, QPlainTextEdit)):
                current = widget.placeholderText()
                source = self._source_attr(widget, "placeholder", current)
                target = translate_text(source, self.language)
                if current != target:
                    widget.setPlaceholderText(target)
                self._remember_applied(widget, "placeholder", target)

            if isinstance(widget, QGroupBox):
                current = widget.title()
                source = self._source_attr(widget, "group_title", current)
                target = translate_text(source, self.language)
                if current != target:
                    widget.setTitle(target)
                self._remember_applied(widget, "group_title", target)

            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                for name, getter, setter in (
                    ("prefix", widget.prefix, widget.setPrefix),
                    ("suffix", widget.suffix, widget.setSuffix),
                ):
                    current = getter()
                    if not current:
                        continue
                    source = self._source_attr(widget, name, current)
                    target = translate_text(source, self.language)
                    if current != target:
                        setter(target)
                    self._remember_applied(widget, name, target)

            if isinstance(widget, QProgressBar):
                current = widget.format()
                if current:
                    source = self._source_attr(widget, "progress_format", current)
                    target = translate_text(source, self.language)
                    if current != target:
                        widget.setFormat(target)
                    self._remember_applied(widget, "progress_format", target)

            current_tip = widget.toolTip()
            if current_tip:
                source_tip = self._source_attr(widget, "tooltip", current_tip)
                target_tip = translate_text(source_tip, self.language)
                if current_tip != target_tip:
                    widget.setToolTip(target_tip)
                self._remember_applied(widget, "tooltip", target_tip)
            current_status = widget.statusTip()
            if current_status:
                source_status = self._source_attr(widget, "statustip", current_status)
                target_status = translate_text(source_status, self.language)
                if current_status != target_status:
                    widget.setStatusTip(target_status)
                self._remember_applied(widget, "statustip", target_status)

            title = widget.windowTitle()
            if title:
                source_title = self._source_attr(widget, "window_title", title)
                target_title = translate_text(source_title, self.language)
                if title != target_title:
                    widget.setWindowTitle(target_title)
                self._remember_applied(widget, "window_title", target_title)

            if isinstance(widget, QTabWidget):
                sources = getattr(widget, "_nf_i18n_tab_sources", {})
                applieds = getattr(widget, "_nf_i18n_tab_applied", {})
                for i in range(widget.count()):
                    current = widget.tabText(i)
                    if i not in sources or (i in applieds and current != applieds[i]):
                        sources[i] = current
                    target = translate_text(sources.get(i, current), self.language)
                    if current != target:
                        widget.setTabText(i, target)
                    applieds[i] = target
                setattr(widget, "_nf_i18n_tab_sources", sources)
                setattr(widget, "_nf_i18n_tab_applied", applieds)

            if isinstance(widget, QComboBox) and not widget.property("nfNoTranslateItems"):
                sources = getattr(widget, "_nf_i18n_item_sources", {})
                applieds = getattr(widget, "_nf_i18n_item_applied", {})
                for i in range(widget.count()):
                    current = widget.itemText(i)
                    if i not in sources or (i in applieds and current != applieds[i]):
                        sources[i] = current
                    target = translate_text(sources.get(i, current), self.language)
                    if current != target:
                        widget.setItemText(i, target)
                    applieds[i] = target
                setattr(widget, "_nf_i18n_item_sources", sources)
                setattr(widget, "_nf_i18n_item_applied", applieds)

            if isinstance(widget, QListWidget) and widget.property("nfTranslateItems"):
                # QListWidget rows are often filenames, chapter names, OCR text
                # or user-defined profile names.  Never translate them unless a
                # widget explicitly opts in; changing item.text() can otherwise
                # alter identifiers consumed by business callbacks.
                for i in range(widget.count()):
                    item = widget.item(i)
                    current = item.text()
                    source = item.data(0x0100 + 51)  # private UserRole slot
                    applied = item.data(0x0100 + 53)
                    if not source or (applied is not None and current != str(applied)):
                        source = current
                        item.setData(0x0100 + 51, source)
                    target = translate_text(str(source), self.language)
                    if current != target:
                        item.setText(target)
                    item.setData(0x0100 + 53, target)

            if isinstance(widget, QTableWidget):
                for i in range(widget.columnCount()):
                    item = widget.horizontalHeaderItem(i)
                    if item is not None:
                        current = item.text()
                        source = item.data(0x0100 + 52)
                        applied = item.data(0x0100 + 54)
                        if not source or (applied is not None and current != str(applied)):
                            source = current
                            item.setData(0x0100 + 52, source)
                        target = translate_text(str(source), self.language)
                        if current != target:
                            item.setText(target)
                        item.setData(0x0100 + 54, target)

            if isinstance(widget, QTreeWidget):
                header = widget.headerItem()
                if header is not None:
                    sources = getattr(widget, "_nf_i18n_tree_header_sources", {})
                    applieds = getattr(widget, "_nf_i18n_tree_header_applied", {})
                    for i in range(widget.columnCount()):
                        current = header.text(i)
                        if i not in sources or (i in applieds and current != applieds[i]):
                            sources[i] = current
                        target = translate_text(sources.get(i, current), self.language)
                        if current != target:
                            header.setText(i, target)
                        applieds[i] = target
                    setattr(widget, "_nf_i18n_tree_header_sources", sources)
                    setattr(widget, "_nf_i18n_tree_header_applied", applieds)

            if isinstance(widget, QMenu):
                current = widget.title()
                if current:
                    source = self._source_attr(widget, "menu_title", current)
                    target = translate_text(source, self.language)
                    if current != target:
                        widget.setTitle(target)
                    self._remember_applied(widget, "menu_title", target)

            self._translate_actions(widget.actions())
        finally:
            self._translate_guard.discard(wid)

    def _translate_actions(self, actions: Iterable[QAction]) -> None:
        for action in actions:
            if action is None or action.isSeparator():
                continue
            current = action.text()
            source = getattr(action, "_nf_i18n_source_text", None)
            applied = getattr(action, "_nf_i18n_applied_text", None)
            if source is None or (applied is not None and current != applied):
                source = current
                setattr(action, "_nf_i18n_source_text", source)
            target = translate_text(source, self.language)
            if current != target:
                action.setText(target)
            setattr(action, "_nf_i18n_applied_text", target)


def manager_for(app: QApplication | None = None) -> InterfacePreferenceManager | None:
    app = app or QApplication.instance()
    if app is None:
        return None
    value = getattr(app, "_nf_interface_preference_manager", None)
    return value if isinstance(value, InterfacePreferenceManager) else None
