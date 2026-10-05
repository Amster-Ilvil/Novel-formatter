# -*- coding: utf-8 -*-
"""Localized wrappers around static Qt dialogs.

Native/static dialogs are created after the application event filter has run and
can bypass ordinary widget-tree translation entirely (especially native file
pickers on macOS/Windows). These wrappers translate only dialog chrome; paths,
filenames, OCR/document data and returned values are untouched.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QApplication,
    QFileDialog as _QFileDialog,
    QMessageBox as _QMessageBox,
    QInputDialog as _QInputDialog,
    QColorDialog as _QColorDialog,
)

from ui.dialog_localization import translate_dialog_text
from ui.localization import LANG_ZH, normalize_language
from ui.message_catalog import format_message


def current_ui_language() -> str:
    app = QApplication.instance()
    if app is None:
        return LANG_ZH
    return normalize_language(str(app.property("nfLanguage") or LANG_ZH))


def _tr(value):
    return translate_dialog_text(value, current_ui_language()) if isinstance(value, str) else value



def ui_message(key: str, /, **values) -> str:
    """Format a stable message-key using the current application language."""
    return format_message(key, current_ui_language(), **values)

def _translated_call_args(args, kwargs, *, text_indices=(), text_keys=()):
    values = list(args)
    for index in text_indices:
        if index < len(values):
            values[index] = _tr(values[index])
    copied = dict(kwargs)
    for key in text_keys:
        if key in copied:
            copied[key] = _tr(copied[key])
    return values, copied


class LocalizedFileDialog(_QFileDialog):
    @staticmethod
    def _source_filter(args, kwargs):
        if len(args) > 3 and isinstance(args[3], str):
            return args[3]
        value = kwargs.get("filter", "")
        return value if isinstance(value, str) else ""

    @staticmethod
    def _restore_selected_filter(selected, source_filter):
        if not isinstance(selected, str) or not source_filter:
            return selected
        translated = _tr(source_filter)
        source_parts = source_filter.split(";;")
        translated_parts = translated.split(";;")
        for source, shown in zip(source_parts, translated_parts):
            if selected == shown:
                return source
        return selected

    @classmethod
    def _file_call(cls, method, args, kwargs):
        source_filter = cls._source_filter(args, kwargs)
        values, copied = _translated_call_args(
            args, kwargs, text_indices=(1, 3, 4), text_keys=("caption", "filter", "selectedFilter")
        )
        result = method(*values, **copied)
        if isinstance(result, tuple) and len(result) == 2:
            return result[0], cls._restore_selected_filter(result[1], source_filter)
        return result

    @classmethod
    def getOpenFileName(cls, *args, **kwargs):
        return cls._file_call(_QFileDialog.getOpenFileName, args, kwargs)

    @classmethod
    def getOpenFileNames(cls, *args, **kwargs):
        return cls._file_call(_QFileDialog.getOpenFileNames, args, kwargs)

    @classmethod
    def getSaveFileName(cls, *args, **kwargs):
        return cls._file_call(_QFileDialog.getSaveFileName, args, kwargs)

    @staticmethod
    def getExistingDirectory(*args, **kwargs):
        args, kwargs = _translated_call_args(
            args, kwargs, text_indices=(1,), text_keys=("caption",)
        )
        return _QFileDialog.getExistingDirectory(*args, **kwargs)


class LocalizedMessageBox(_QMessageBox):
    @staticmethod
    def information(*args, **kwargs):
        args, kwargs = _translated_call_args(args, kwargs, text_indices=(1, 2), text_keys=("title", "text"))
        return _QMessageBox.information(*args, **kwargs)

    @staticmethod
    def warning(*args, **kwargs):
        args, kwargs = _translated_call_args(args, kwargs, text_indices=(1, 2), text_keys=("title", "text"))
        return _QMessageBox.warning(*args, **kwargs)

    @staticmethod
    def critical(*args, **kwargs):
        args, kwargs = _translated_call_args(args, kwargs, text_indices=(1, 2), text_keys=("title", "text"))
        return _QMessageBox.critical(*args, **kwargs)

    @staticmethod
    def question(*args, **kwargs):
        args, kwargs = _translated_call_args(args, kwargs, text_indices=(1, 2), text_keys=("title", "text"))
        return _QMessageBox.question(*args, **kwargs)

    def setText(self, text):
        super().setText(_tr(text))

    def setInformativeText(self, text):
        super().setInformativeText(_tr(text))

    def setDetailedText(self, text):
        super().setDetailedText(_tr(text))

    def setWindowTitle(self, title):
        super().setWindowTitle(_tr(title))


class LocalizedInputDialog(_QInputDialog):
    @staticmethod
    def getText(*args, **kwargs):
        args, kwargs = _translated_call_args(args, kwargs, text_indices=(1, 2), text_keys=("title", "label"))
        return _QInputDialog.getText(*args, **kwargs)

    @staticmethod
    def getItem(*args, **kwargs):
        args, kwargs = _translated_call_args(args, kwargs, text_indices=(1, 2), text_keys=("title", "label"))
        return _QInputDialog.getItem(*args, **kwargs)


class LocalizedColorDialog(_QColorDialog):
    @staticmethod
    def getColor(*args, **kwargs):
        args, kwargs = _translated_call_args(args, kwargs, text_indices=(2,), text_keys=("title",))
        return _QColorDialog.getColor(*args, **kwargs)
