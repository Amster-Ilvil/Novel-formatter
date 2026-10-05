from __future__ import annotations

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtWidgets import (
    QApplication, QDialog, QDialogButtonBox, QHeaderView, QLabel, QTableView, QVBoxLayout,
)

from core.batch_changes import BatchChangeSet
from ui.localization import LANG_ZH, normalize_language, translate_text


def _ui_language() -> str:
    app = QApplication.instance()
    return normalize_language(app.property("nfLanguage") if app is not None else LANG_ZH)


def _tr_ui(value: str) -> str:
    return translate_text(str(value or ""), _ui_language())


class _BatchChangeTableModel(QAbstractTableModel):
    HEADERS = ("#", "类型", "修改前", "修改后", "说明")

    def __init__(self, change_set: BatchChangeSet, parent=None):
        super().__init__(parent)
        self._items = change_set.changes

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._items)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal and 0 <= section < len(self.HEADERS):
            return _tr_ui(self.HEADERS[section])
        return super().headerData(section, orientation, role)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self._items):
            return None
        item = self._items[index.row()]
        if role in (Qt.DisplayRole, Qt.ToolTipRole):
            values = (
                str(index.row() + 1),
                _tr_ui(item.category),
                item.before,
                item.after,
                _tr_ui(item.detail),
            )
            return values[index.column()]
        return None


class BatchChangePreviewDialog(QDialog):
    """Scalable Preview → Apply confirmation for existing batch operations.

    This dialog owns no domain mutation.  It renders an immutable ChangeSet and
    returns Accepted/Rejected; the caller remains the single owner of applying
    or restoring the real project state.
    """

    def __init__(self, change_set: BatchChangeSet, parent=None, *, apply_text: str = "应用变更"):
        super().__init__(parent)
        self.setWindowTitle(_tr_ui(change_set.title))
        self.resize(900, 560)

        layout = QVBoxLayout(self)
        summary_text = _tr_ui(f"共 {change_set.count} 项变更")
        categories = change_set.category_counts()
        if categories:
            details = " · ".join(
                f"{_tr_ui(category)} {count}"
                for category, count in sorted(categories.items())
            )
            summary_text += f" · {details}"
        summary = QLabel(
            f"{summary_text}\n"
            + _tr_ui("这里只显示计划中的修改；点击应用前不会写回任何正文、OCR 候选或项目文件。")
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)

        table = QTableView(self)
        table.setModel(_BatchChangeTableModel(change_set, table))
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QTableView.SelectRows)
        table.setWordWrap(False)
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        layout.addWidget(table, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok, parent=self)
        ok = buttons.button(QDialogButtonBox.Ok)
        if ok is not None:
            ok.setText(_tr_ui(str(apply_text or "应用变更")))
        cancel = buttons.button(QDialogButtonBox.Cancel)
        if cancel is not None:
            cancel.setText(_tr_ui("取消"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @classmethod
    def confirm(
        cls,
        parent,
        change_set: BatchChangeSet,
        *,
        apply_text: str = "应用变更",
    ) -> bool:
        if change_set.count <= 0:
            return True
        dialog = cls(change_set, parent, apply_text=apply_text)
        try:
            return dialog.exec() == QDialog.Accepted
        finally:
            dialog.deleteLater()
