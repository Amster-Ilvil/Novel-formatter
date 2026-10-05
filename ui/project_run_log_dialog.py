# -*- coding: utf-8 -*-
"""Read-only project run history browser.

The authoritative records live under ``project/logs/runs`` and are managed by
``ProjectWorkspaceManager``.  This dialog only presents, exports and prunes run
diagnostics; it never edits OCR/adjudication/document artifacts.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QFrame,
)

from core.project_workspace import ProjectWorkspaceManager
from ui.common.toast import notify
from ui.common.window_state import bind_splitter
from ui.common.styling import CARD, BORDER, MUTED


_STATUS_LABELS = {
    "running": "运行中",
    "ok": "完成",
    "cancelled": "已停止",
    "error": "失败",
    "warning": "警告",
}

_STAGE_LABELS = {
    "single_ocr": "单模型 OCR",
    "multi_ocr": "多模型 OCR",
    "ocr": "OCR 正文",
    "formatter": "Formatter",
    "manual_compare": "旧版手动校对",
    "ocr_image_review": "图文校对",
    "cloud_adjudication_import": "云端裁决导入",
    "epub_export": "EPUB 导出",
    "package_export": "校对/裁决包导出",
    "project_backup": "项目备份",
    "project_restore": "项目恢复",
    "pages_loaded": "页面载入",
    "workspace_cleanup": "工作区清理",
}


def _duration_text(value) -> str:
    try:
        seconds = max(0.0, float(value or 0.0))
    except Exception:
        seconds = 0.0
    if seconds < 1:
        return f"{seconds:.2f}s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {rem:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _stage_label(stage: str) -> str:
    return _STAGE_LABELS.get(str(stage or ""), str(stage or "—"))


class ProjectRunLogDialog(QDialog):
    """Browse one project's lightweight history and detailed run attachments."""

    def __init__(self, manager: ProjectWorkspaceManager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self._rows: list[dict] = []
        self.setWindowTitle("项目运行日志")
        self.resize(1120, 720)
        self.setMinimumSize(860, 560)
        self._build()
        self.refresh()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(8)

        filter_card = QFrame()
        filter_card.setObjectName("runLogFilterCard")
        filter_card.setStyleSheet(f"QFrame#runLogFilterCard{{background:{CARD};border:1px solid {BORDER};border-radius:12px;}}")
        filter_layout = QVBoxLayout(filter_card)
        filter_layout.setContentsMargins(12, 10, 12, 10)
        filter_layout.setSpacing(8)
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("状态"))
        self._status_combo = QComboBox()
        self._status_combo.addItem("全部", "")
        for key in ("running", "ok", "cancelled", "error", "warning"):
            self._status_combo.addItem(_STATUS_LABELS[key], key)
        self._status_combo.currentIndexChanged.connect(self.refresh)
        toolbar.addWidget(self._status_combo)

        toolbar.addWidget(QLabel("阶段"))
        self._stage_combo = QComboBox()
        self._stage_combo.addItem("全部", "")
        self._stage_combo.currentIndexChanged.connect(self._apply_filters)
        toolbar.addWidget(self._stage_combo)
        toolbar.addStretch(1)

        refresh = QPushButton("刷新")
        refresh.clicked.connect(self.refresh)
        toolbar.addWidget(refresh)
        export = QPushButton("导出日志")
        export.clicked.connect(self._export)
        toolbar.addWidget(export)
        open_folder = QPushButton("打开日志目录")
        open_folder.clicked.connect(self._open_folder)
        toolbar.addWidget(open_folder)
        prune = QPushButton("清理旧日志")
        prune.setToolTip("只删除旧运行日志附件，不删除 OCR、裁决、正文或导出结果")
        prune.clicked.connect(self._prune)
        toolbar.addWidget(prune)
        filter_layout.addLayout(toolbar)

        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet(f"color:{MUTED};")
        filter_layout.addWidget(self._summary)
        root.addWidget(filter_card)

        splitter = QSplitter(Qt.Vertical)
        self._table = QTableWidget(0, 7)
        self._table.setHorizontalHeaderLabels([
            "开始时间", "阶段", "状态", "耗时", "摘要", "文本日志", "性能明细",
        ])
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        self._table.itemSelectionChanged.connect(self._show_selected)
        splitter.addWidget(self._table)

        detail_host = QWidget()
        detail_layout = QVBoxLayout(detail_host)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_head = QHBoxLayout()
        detail_head.addWidget(QLabel("运行详情（只读）"))
        detail_head.addStretch(1)
        copy_btn = QPushButton("复制详情")
        copy_btn.clicked.connect(self._copy_detail)
        detail_head.addWidget(copy_btn)
        detail_layout.addLayout(detail_head)
        self._detail = QPlainTextEdit()
        self._detail.setReadOnly(True)
        self._detail.setLineWrapMode(QPlainTextEdit.NoWrap)
        detail_layout.addWidget(self._detail, 1)
        splitter.addWidget(detail_host)
        splitter.setSizes([360, 280])
        bind_splitter(splitter, "project_run_log_vertical")
        root.addWidget(splitter, 1)

        bottom = QHBoxLayout()
        bottom.addWidget(QLabel("运行日志属于项目备份的一部分；API Key / Token 常见格式会在文本日志落盘前自动脱敏。"))
        bottom.addStretch(1)
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        bottom.addWidget(close_btn)
        root.addLayout(bottom)

    def _copy_detail(self) -> None:
        QApplication.clipboard().setText(self._detail.toPlainText())
        notify(self, "运行详情已复制到剪贴板。", "success")

    def refresh(self, _index: int = 0) -> None:
        try:
            self._rows = self.manager.list_run_history(limit=1000)
        except Exception as exc:
            QMessageBox.warning(self, "日志读取失败", str(exc))
            self._rows = []
        current_stage = str(self._stage_combo.currentData() or "")
        stages = sorted({str(row.get("stage") or "") for row in self._rows if row.get("stage")})
        self._stage_combo.blockSignals(True)
        self._stage_combo.clear()
        self._stage_combo.addItem("全部", "")
        for stage in stages:
            self._stage_combo.addItem(_stage_label(stage), stage)
        index = self._stage_combo.findData(current_stage)
        self._stage_combo.setCurrentIndex(max(0, index))
        self._stage_combo.blockSignals(False)
        self._apply_filters()

    def _apply_filters(self, _index: int = 0) -> None:
        status = str(self._status_combo.currentData() or "")
        stage = str(self._stage_combo.currentData() or "")
        rows = [
            row for row in self._rows
            if (not status or str(row.get("status") or "") == status)
            and (not stage or str(row.get("stage") or "") == stage)
        ]
        self._table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = [
                str(row.get("started_at") or row.get("timestamp") or ""),
                _stage_label(str(row.get("stage") or "")),
                _STATUS_LABELS.get(str(row.get("status") or ""), str(row.get("status") or "")),
                _duration_text(row.get("duration_seconds")),
                str(row.get("summary") or row.get("error_message") or ""),
                "有" if row.get("has_text_log") else "—",
                "有" if row.get("has_performance") else "—",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(Qt.UserRole, str(row.get("run_id") or ""))
                self._table.setItem(row_index, column, item)
        errors = sum(1 for row in rows if str(row.get("status") or "") == "error")
        running = sum(1 for row in rows if str(row.get("status") or "") == "running")
        self._summary.setText(
            f"显示 {len(rows)} / {len(self._rows)} 条运行记录 · 失败 {errors} · 运行中 {running}"
        )
        if rows:
            self._table.selectRow(0)
        else:
            self._detail.clear()

    def _selected_run_id(self) -> str:
        row = self._table.currentRow()
        if row < 0:
            return ""
        item = self._table.item(row, 0)
        return str(item.data(Qt.UserRole) or "") if item else ""

    def _show_selected(self) -> None:
        run_id = self._selected_run_id()
        if not run_id:
            self._detail.clear()
            return
        try:
            payload = self.manager.load_run(run_id)
            text_log = self.manager.load_run_text_log(run_id)
        except Exception as exc:
            self._detail.setPlainText(f"读取运行详情失败：{exc}")
            return
        pieces = [
            "=== 结构化运行记录 ===",
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        ]
        artifacts = dict(payload.get("artifacts") or {})
        perf_name = str(artifacts.get("performance") or "")
        log_dir = self.manager.run_log_dir
        if perf_name and log_dir is not None:
            perf_path = Path(log_dir) / perf_name
            if perf_path.is_file():
                try:
                    perf = json.loads(perf_path.read_text(encoding="utf-8"))
                    pieces.extend(["", "=== OCR 性能明细 ===", json.dumps(perf, ensure_ascii=False, indent=2, sort_keys=True)])
                except Exception as exc:
                    pieces.extend(["", f"=== OCR 性能明细读取失败：{exc} ==="])
        if text_log:
            pieces.extend(["", "=== OCR / 运行文本日志 ===", text_log.rstrip()])
        self._detail.setPlainText("\n".join(pieces))

    def _export(self) -> None:
        if self.manager.active_project is None:
            return
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        default = Path(self.manager.active_project) / "exports" / "packages" / f"运行日志_{stamp}.csv"
        path, selected = QFileDialog.getSaveFileName(
            self,
            "导出运行日志",
            str(default),
            "CSV (*.csv);;JSON (*.json)",
        )
        if not path:
            return
        if "JSON" in str(selected) and not str(path).lower().endswith(".json"):
            path += ".json"
        try:
            output = self.manager.export_run_history(path)
            notify(self, f"日志已导出：\n{output}", "success")
        except Exception as exc:
            QMessageBox.warning(self, "日志导出失败", str(exc))

    def _open_folder(self) -> None:
        path = self.manager.run_log_dir
        if path is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _prune(self) -> None:
        keep, ok = QInputDialog.getInt(
            self,
            "清理旧日志",
            "保留最近多少条运行记录？\n只清理 logs/runs，不删除 OCR、裁决、正文或 EPUB。",
            200,
            20,
            10000,
            20,
        )
        if not ok:
            return
        reply = QMessageBox.question(
            self,
            "确认清理日志",
            f"将保留最近 {keep} 条运行记录，删除更旧的 JSON / 文本日志 / 性能明细。\n\n继续吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        try:
            result = self.manager.prune_run_history(keep_recent=keep)
            notify(
                self,
                f"已清理 {result.get('runs', 0)} 条旧记录 / {result.get('files', 0)} 个日志文件；"
                f"保留 {result.get('kept', 0)} 条。",
                "success",
            )
            self.refresh()
        except Exception as exc:
            QMessageBox.warning(self, "日志清理失败", str(exc))
