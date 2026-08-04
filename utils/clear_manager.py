# -*- coding: utf-8 -*-
"""Workspace reset helpers used by the GUI.

The previous implementation silently ignored every failure and was injected by
monkey-patching each tab's ``_build`` method at import time.  These helpers are
now called explicitly after tabs are constructed, making installation
idempotent and reset failures visible in the application log.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

from PySide6.QtWidgets import QHBoxLayout, QMessageBox, QPushButton, QWidget

logger = logging.getLogger(__name__)


class ClearManager:
    @staticmethod
    def clear_pages(tab) -> None:
        # Invalidate a still-running thumbnail loader before touching its state.
        tab._load_generation = int(getattr(tab, "_load_generation", 0)) + 1
        old_inputs = list(getattr(tab, "_last_loaded_raw_inputs", None) or [])
        for name in ("page_images", "page_overrides", "selected_pages", "thumb_cache", "_auto_suggested"):
            value = getattr(tab, name, None)
            if hasattr(value, "clear"):
                value.clear()
        signal_refs = getattr(tab, "_load_signal_refs", None)
        if hasattr(signal_refs, "clear"):
            signal_refs.clear()
        tab._last_loaded_raw_inputs = None
        try:
            from adapters.pdf_input import release_pdf_caches
            release_pdf_caches(old_inputs)
        except Exception:
            logger.debug("Failed to release page-manager PDF cache", exc_info=True)
        tab._current_filter = "all"
        if hasattr(tab, "image_folder"):
            tab.image_folder = ""
        for name, text in (
            ("_file_lbl", "未打开文件"),
            ("_count_lbl", ""),
            ("_sel_lbl", "选中 0 页 · 标记为："),
            ("_stat_label", ""),
        ):
            widget = getattr(tab, name, None)
            if widget is not None:
                widget.setText(text)
        progress = getattr(tab, "_prog", None)
        if progress is not None:
            progress.setVisible(False)
        render = getattr(tab, "_render", None)
        if callable(render):
            render()
        changed = getattr(tab, "types_changed", None)
        if changed is not None:
            changed.emit()

    @staticmethod
    def clear_ocr(tab) -> None:
        cancel_event = getattr(tab, "_cancel_event", None)
        if cancel_event is not None:
            cancel_event.set()
        tab._run_generation = int(getattr(tab, "_run_generation", 0)) + 1
        cleanup_temp = getattr(tab, "clear_ocr_temporary_files", None)
        if callable(cleanup_temp):
            cleanup_temp(closing=False)
        pending = getattr(tab, "_pending_inputs", None)
        if hasattr(pending, "clear"):
            pending.clear()
        preview = getattr(tab, "_preview", None)
        if preview is not None:
            clear_rect = getattr(preview, "clear_rect", None)
            if callable(clear_rect):
                clear_rect()
            clear_image = getattr(preview, "clear_preview", None)
            if not callable(clear_image):
                clear_image = getattr(preview, "clear", None)
            if callable(clear_image):
                clear_image()
        for name in ("_log_view", "_result_view"):
            widget = getattr(tab, name, None)
            if widget is not None and hasattr(widget, "clear"):
                widget.clear()
        label = getattr(tab, "_input_lbl", None)
        if label is not None:
            label.setText("（尚未选择输入）")
        reset = getattr(tab, "_reset_run_state", None)
        if callable(reset):
            reset()
        run_btn = getattr(tab, "_run_btn", None)
        if run_btn is not None:
            run_btn.setEnabled(True)
        if hasattr(tab, "_latest_single_doc"):
            tab._latest_single_doc = None
        export_btn = getattr(tab, "_single_roundtrip_export_btn", None)
        if export_btn is not None:
            export_btn.setEnabled(False)

    @staticmethod
    def clear_formatter(tab) -> None:
        reset = getattr(tab, "reset_for_new_book", None)
        if callable(reset):
            reset()
            return
        raise AttributeError("FormatterTab 缺少 reset_for_new_book()")

    @staticmethod
    def clear_epub(tab) -> None:
        clear = getattr(tab, "clear_doc", None)
        if callable(clear):
            clear()
            return
        raise AttributeError("EPUBTab 缺少 clear_doc()")


def attach_workspace_clear_button(
    tab: QWidget,
    label: str,
    callback: Callable[[QWidget], None],
) -> QPushButton:
    """Append one unobtrusive reset button below a workspace card."""
    existing = tab.findChild(QPushButton, "novelFormatterWorkspaceClear")
    if existing is not None:
        return existing

    layout = tab.layout()
    if layout is None or not hasattr(layout, "addLayout"):
        raise RuntimeError(f"无法为 {type(tab).__name__} 添加清空按钮：缺少顶层布局")

    row = QHBoxLayout()
    row.setContentsMargins(10, 0, 10, 8)
    row.addStretch(1)
    button = QPushButton(f"清空{label}")
    button.setObjectName("novelFormatterWorkspaceClear")
    button.setProperty("variant", "danger")
    button.setToolTip(f"清除当前{label}工作区，不影响磁盘上的原始文件")
    button.setStyleSheet(
        "QPushButton { background:#FFF0EF; color:#C4322B; border:1px solid #E9B7B3;"
        "border-radius:8px; padding:6px 12px; min-height:28px; font-weight:600; }"
        "QPushButton:hover { background:#FFF1F0; border-color:#D96C65; }"
        "QPushButton:pressed { background:#FFE4E0; }"
    )

    def run_reset() -> None:
        answer = QMessageBox.question(
            tab,
            "确认清空",
            f"确定清空当前{label}工作区？\n\n磁盘上的原始文件不会被删除。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        try:
            callback(tab)
        except Exception:
            logger.exception("Failed to clear %s workspace", label)
            QMessageBox.critical(tab, "清空失败", f"{label}工作区未能完全清空，请查看终端日志。")

    button.clicked.connect(run_reset)
    row.addWidget(button)
    layout.addLayout(row)
    return button
