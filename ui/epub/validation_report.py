from __future__ import annotations

from dataclasses import dataclass
import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication, QDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPlainTextEdit, QPushButton, QVBoxLayout,
)


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    text: str
    path: str = ""
    line: int = 0
    column: int = 0


def _items(value):
    return [str(item) for item in (value or ()) if str(item).strip()]


def _split_location(text: str) -> tuple[str, int, int]:
    """Extract an EPUB member path and optional line/column from a message.

    EPUBCheck appends locations as ``(path:line:column)`` or ``(path:line)``.
    Keep parsing deliberately conservative: only a trailing parenthesized value
    can become a navigation target, so normal punctuation in the message cannot
    accidentally open an unrelated package file.
    """
    match = re.search(r"\(([^()\n]+)\)\s*$", str(text or ""))
    if not match:
        return "", 0, 0
    location = match.group(1).strip()
    if not location:
        return "", 0, 0
    path = location
    line = column = 0
    parts = location.rsplit(":", 2)
    if len(parts) >= 2 and parts[-1].isdigit():
        if len(parts) == 3 and parts[-2].isdigit():
            path, line, column = parts[0], int(parts[-2]), int(parts[-1])
        else:
            path, line = ":".join(parts[:-1]), int(parts[-1])
    path = path.strip().replace("\\", "/")
    # A useful EPUB package target is a member path, not a bare number.
    if not path or "/" not in path and "." not in path:
        return "", 0, 0
    return path, line, column


def extract_validation_issues(info: dict | None) -> list[ValidationIssue]:
    data = dict(info or {})
    issues: list[ValidationIssue] = []
    for source in (data.get("quality_gate"), data.get("epubcheck")):
        if source is None:
            continue
        for severity, attr in (("错误", "errors"), ("警告", "warnings"), ("信息", "infos")):
            for text in _items(getattr(source, attr, None)):
                path, line, column = _split_location(text)
                issues.append(ValidationIssue(severity, text, path, line, column))
    return issues


def format_epub_validation_report(info: dict | None) -> str:
    """Format the retained post-build validation snapshot for human review."""
    data = dict(info or {})
    path = str(data.get("path") or "")
    gate = data.get("quality_gate")
    check = data.get("epubcheck")
    lines = ["EPUB 发布检查报告", "=" * 24]
    if path:
        lines.extend([f"输出：{path}", ""])

    lines.append("[内部质量门禁]")
    if gate is None:
        lines.append("未运行")
    else:
        summary = getattr(gate, "summary", None)
        lines.append(summary() if callable(summary) else str(gate))
        errors = _items(getattr(gate, "errors", None))
        warnings = _items(getattr(gate, "warnings", None))
        if errors:
            lines.append(f"错误（{len(errors)}）:")
            lines.extend(f"  - {item}" for item in errors)
        if warnings:
            lines.append(f"警告（{len(warnings)}）:")
            lines.extend(f"  - {item}" for item in warnings)
        if not errors and not warnings:
            lines.append("错误 0 · 警告 0")

    lines.extend(["", "[EPUBCheck]"])
    if check is None:
        lines.append("未运行")
    else:
        summary = getattr(check, "summary", None)
        lines.append(summary() if callable(summary) else str(check))
        engine = str(getattr(check, "engine", "") or "")
        version = str(getattr(check, "version", "") or "")
        spec_target = str(getattr(check, "spec_target", "") or "")
        meta = " · ".join(part for part in (engine, version, spec_target) if part)
        if meta:
            lines.append(f"引擎：{meta}")
        if bool(getattr(check, "skipped", False)):
            reason = str(getattr(check, "skip_reason", "") or "")
            if reason:
                lines.append(f"跳过原因：{reason}")
        errors = _items(getattr(check, "errors", None))
        warnings = _items(getattr(check, "warnings", None))
        infos = _items(getattr(check, "infos", None))
        if errors:
            lines.append(f"错误（{len(errors)}）:")
            lines.extend(f"  - {item}" for item in errors)
        if warnings:
            lines.append(f"警告（{len(warnings)}）:")
            lines.extend(f"  - {item}" for item in warnings)
        if infos:
            lines.append(f"信息（{len(infos)}）:")
            lines.extend(f"  - {item}" for item in infos[:50])
        if not errors and not warnings and not infos and not bool(getattr(check, "skipped", False)):
            lines.append("错误 0 · 警告 0")
    return "\n".join(lines).rstrip() + "\n"


class EPUBValidationReportDialog(QDialog):
    location_requested = Signal(str, int, int)

    _PATH_ROLE = Qt.UserRole
    _LINE_ROLE = Qt.UserRole + 1
    _COLUMN_ROLE = Qt.UserRole + 2

    def __init__(self, report_text: str, parent=None, *, issues: list[ValidationIssue] | None = None):
        super().__init__(parent)
        self.setWindowTitle("检查结果")
        self.resize(860, 620)
        root = QVBoxLayout(self)

        locatable = [issue for issue in (issues or ()) if issue.path]
        self._issue_list: QListWidget | None = None
        if locatable:
            issue_head = QHBoxLayout()
            title = QLabel("可定位问题")
            title.setStyleSheet("font-weight: 700;")
            issue_head.addWidget(title)
            hint = QLabel("双击或按 Enter 跳转到打包文件")
            hint.setStyleSheet("color: #6b7280;")
            issue_head.addWidget(hint)
            issue_head.addStretch(1)
            root.addLayout(issue_head)

            self._issue_list = QListWidget()
            self._issue_list.setMaximumHeight(165)
            self._issue_list.setAlternatingRowColors(True)
            for issue in locatable:
                location = issue.path
                if issue.line:
                    location += f":{issue.line}"
                    if issue.column:
                        location += f":{issue.column}"
                item = QListWidgetItem(f"{issue.severity} · {location} · {issue.text}")
                item.setData(self._PATH_ROLE, issue.path)
                item.setData(self._LINE_ROLE, issue.line)
                item.setData(self._COLUMN_ROLE, issue.column)
                self._issue_list.addItem(item)
            self._issue_list.itemActivated.connect(self._activate_issue)
            root.addWidget(self._issue_list)

        self._view = QPlainTextEdit()
        self._view.setReadOnly(True)
        self._view.setLineWrapMode(QPlainTextEdit.NoWrap)
        self._view.setPlainText(str(report_text or ""))
        root.addWidget(self._view, 1)
        actions = QHBoxLayout()
        actions.addStretch(1)
        copy_btn = QPushButton("复制详情")
        copy_btn.clicked.connect(self._copy)
        actions.addWidget(copy_btn)
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        actions.addWidget(close_btn)
        root.addLayout(actions)

    def _activate_issue(self, item: QListWidgetItem) -> None:
        path = str(item.data(self._PATH_ROLE) or "")
        if not path:
            return
        self.location_requested.emit(
            path,
            int(item.data(self._LINE_ROLE) or 0),
            int(item.data(self._COLUMN_ROLE) or 0),
        )
        self.accept()

    def _copy(self) -> None:
        QApplication.clipboard().setText(self._view.toPlainText())


__all__ = [
    "EPUBValidationReportDialog", "ValidationIssue", "extract_validation_issues",
    "format_epub_validation_report",
]
