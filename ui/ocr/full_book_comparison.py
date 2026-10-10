"""Read-only, *complete* aligned multi-model book viewer.

The ordinary OCR editors may display only the selected sentence.  This view
reads directly from the full comparison rows and never materializes one QWidget
per row. A single QTableView scrollbar keeps models and fusion in lockstep.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from functools import lru_cache

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, QTimer
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor, QTextDocument
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QStyledItemDelegate, QStyle, QTableView


def _safe(value) -> str:
    return str(value or "")


def _fusion_cell(state) -> str:
    if state is None:
        return "〔未提供融合状态〕"
    if state.output_delete_intentionally():
        return "〔有意删除〕"
    if state.unresolved:
        first = next((c.text for c in state.candidates if c.text and c.audit_level != "historical_ocr_evidence"), "")
        return "〔待裁决〕" + _safe(first)
    return _safe(state.output_text())


@lru_cache(maxsize=12288)
def difference_spans(values: tuple[str, ...], index: int) -> tuple[tuple[int, int], ...]:
    """Character ranges that disagree with any other OCR source (not the EPUB).

    Missing sources are separately labeled. Case/character replacements are
    highlighted in BOTH participating models, not just a minority candidate.
    """
    if index >= len(values) or not values[index]:
        return ()
    base = values[index]
    marked = set()
    for peer_i, peer in enumerate(values):
        if peer_i == index or base == peer:
            continue
        for op, i1, i2, _j1, _j2 in SequenceMatcher(None, base, peer, autojunk=False).get_opcodes():
            if op == "equal":
                continue
            if i1 == i2:
                # Visually mark the insertion boundary as well.
                if base:
                    marked.add(min(i1, len(base) - 1))
            else:
                marked.update(range(i1, i2))
    if not marked:
        return ()
    ranges = []
    start = last = -1
    for point in sorted(marked):
        if start < 0:
            start = last = point
        elif point == last + 1:
            last = point
        else:
            ranges.append((start, last + 1))
            start = last = point
    ranges.append((start, last + 1))
    return tuple(ranges)


class FullBookComparisonModel(QAbstractTableModel):
    """One logical row per sentence; ALL model values and fusion are retained."""

    DiffSpansRole = Qt.UserRole + 1
    ConflictRole = Qt.UserRole + 2
    RawValueRole = Qt.UserRole + 3

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = ()
        self.states = ()
        self.labels = ()
        self.model_count = 0

    def set_session(self, rows, states, labels):
        self.beginResetModel()
        self.rows = tuple(rows or ())
        self.states = tuple(states or ())
        self.model_count = max(len(labels or ()), max((len(getattr(r, 'texts', ()) or ()) for r in self.rows), default=0))
        self.labels = tuple(str(v) for v in (labels or ()))
        self._last_fusion_cells = tuple(_fusion_cell(state) for state in self.states)
        difference_spans.cache_clear()
        self.endResetModel()

    def update_states(self, states):
        """Repaint adjudication overlays without resetting the book position."""
        previous = tuple(_fusion_cell(state) for state in self.states)
        # States can mutate in place between updates, so UI preserves its own
        # last text snapshot instead of comparing only state object identity.
        previous = getattr(self, '_last_fusion_cells', previous)
        self.states = tuple(states or ())
        fresh = tuple(_fusion_cell(state) for state in self.states)
        changed = [i for i, (old, new) in enumerate(zip(previous, fresh)) if old != new]
        changed.extend(range(len(previous), len(fresh)))
        self._last_fusion_cells = fresh
        if changed and self.rows:
            for row in changed:
                if row < len(self.rows):
                    idx = self.index(row, self.model_count)
                    self.dataChanged.emit(idx, idx, [Qt.DisplayRole, Qt.UserRole])
        return changed

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self.model_count + 1

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Vertical:
                return str(section + 1)
            return self.labels[section] if section < len(self.labels) else (
                '融合结果（只读）' if section == self.model_count else f'模型 {section + 1}'
            )
        return None

    def raw_values(self, row_index):
        return tuple(_safe(t) for t in (getattr(self.rows[row_index], 'texts', ()) or ())) + tuple(
            '' for _ in range(max(0, self.model_count - len(getattr(self.rows[row_index], 'texts', ()) or ())))
        )

    def row_disagrees(self, row_index):
        values = self.raw_values(row_index)
        if len(set(values)) > 1:
            return True
        state = self.states[row_index] if row_index < len(self.states) else None
        if state is None or state.unresolved:
            return False  # a pending marker is not a model-text difference
        if state.output_delete_intentionally():
            return any(values)
        fused = _safe(state.output_text())
        return bool(values) and fused != values[0]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        row_index, col = index.row(), index.column()
        values = self.raw_values(row_index)
        fusion = _fusion_cell(self.states[row_index] if row_index < len(self.states) else None)
        raw = values[col] if col < self.model_count else fusion
        if role == Qt.DisplayRole:
            return raw or '〔该模型无对应文字〕'
        if role == self.RawValueRole:
            return raw
        if role == self.ConflictRole:
            return self.row_disagrees(row_index)
        if role == self.DiffSpansRole:
            if col < self.model_count:
                return difference_spans(values, col)
            if not fusion or fusion.startswith('〔'):
                return ()
            marks = set()
            for value in values:
                for op, i1, i2, _j1, _j2 in SequenceMatcher(None, fusion, value, autojunk=False).get_opcodes():
                    if op != 'equal':
                        marks.update(range(i1, max(i1 + 1, i2)))
            marks = sorted(i for i in marks if i < len(fusion))
            if not marks:
                return ()
            spans = []
            start = last = marks[0]
            for p in marks[1:]:
                if p == last + 1:
                    last = p
                else:
                    spans.append((start, last + 1))
                    start = last = p
            spans.append((start, last + 1))
            return tuple(spans)
        if role == Qt.ToolTipRole:
            return f'第 {row_index + 1} 行 · ' + ('多模型不一致' if self.row_disagrees(row_index) else '模型一致')
        return None


class FullBookDiffDelegate(QStyledItemDelegate):
    def _document(self, option, index):
        text = str(index.data(Qt.DisplayRole) or '')
        doc = QTextDocument()
        doc.setDocumentMargin(4)
        doc.setDefaultFont(option.font)
        doc.setPlainText(text)
        ranges = index.data(FullBookComparisonModel.DiffSpansRole) or ()
        if ranges:
            cursor = QTextCursor(doc)
            fmt = QTextCharFormat()
            fmt.setBackground(QColor('#F5B1B6'))
            fmt.setForeground(QColor('#8B1027'))
            for start, end in ranges:
                cursor.setPosition(max(0, min(start, doc.characterCount() - 1)))
                cursor.setPosition(max(0, min(end, doc.characterCount() - 1)), QTextCursor.KeepAnchor)
                cursor.mergeCharFormat(fmt)
        return doc

    def paint(self, painter, option, index):
        painter.save()
        painter.fillRect(option.rect, QColor('#FFF0F0') if index.data(FullBookComparisonModel.ConflictRole) else QColor('#FFFFFF'))
        if option.state & QStyle.State_Selected:
            painter.fillRect(option.rect, QColor('#E4EEFF'))
        doc = self._document(option, index)
        doc.setTextWidth(max(80, option.rect.width() - 10))
        painter.translate(option.rect.left() + 5, option.rect.top() + 3)
        doc.drawContents(painter)
        painter.restore()

    def sizeHint(self, option, index):
        doc = self._document(option, index)
        doc.setTextWidth(max(80, option.rect.width() - 10))
        from PySide6.QtCore import QSize
        return QSize(option.rect.width(), int(doc.size().height()) + 8)


class FullBookComparisonTable(QTableView):
    """One shared vertical scrollbar; row heights account for complete text."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('ocrCompareMultiModelFullBook')
        self.setModel(FullBookComparisonModel(self))
        self.setItemDelegate(FullBookDiffDelegate(self))
        self.setWordWrap(True)
        self.setTextElideMode(Qt.ElideNone)
        self.setAlternatingRowColors(False)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.horizontalHeader().setMinimumSectionSize(240)
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setMinimumWidth(62)
        self.verticalHeader().setDefaultSectionSize(56)
        self._resize_generation = 0
        self._last_comparison = None
        self._last_labels = ()
        self.horizontalHeader().sectionResized.connect(self._schedule_resize)
        self.setStyleSheet('QTableView{background:#FFFFFF;gridline-color:#E1E8F2;font-size:12px;}'
                           'QHeaderView::section{background:#EFF4FB;border:1px solid #DCE5F1;padding:7px;font-weight:600;}')

    def set_comparison(self, comparison, states, labels):
        labels = tuple(labels or ())
        if comparison is not None and comparison is self._last_comparison and labels == self._last_labels:
            changed = self.model().update_states(states)
            if changed:
                self._resize_generation += 1
                QTimer.singleShot(0, lambda: self._resize_changed_rows(changed, self._resize_generation))
            return False  # no reset; preserve current scroll / selection
        rows = getattr(comparison, 'rows', ()) if comparison is not None else ()
        self._last_comparison = comparison
        self._last_labels = labels
        self.model().set_session(rows, states, labels)
        for i in range(self.model().columnCount()):
            self.setColumnWidth(i, 330 if i < self.model().model_count else 390)
        self._schedule_resize()
        return True

    def _resize_changed_rows(self, indices, generation):
        if generation != self._resize_generation:
            return
        for row in indices:
            if 0 <= row < self.model().rowCount():
                self._fit_row(row)

    def _schedule_resize(self, *_args):
        self._resize_generation += 1
        generation = self._resize_generation
        QTimer.singleShot(0, lambda: self._resize_rows(generation, 0))

    def _resize_rows(self, generation, offset):
        if generation != self._resize_generation:
            return
        model = self.model()
        if not model or not model.rowCount():
            return
        # Chunked sizing keeps Qt responsive while still making EVERY row fully visible.
        limit = min(model.rowCount(), offset + 80)
        for row in range(offset, limit):
            self._fit_row(row)
        if limit < model.rowCount():
            QTimer.singleShot(0, lambda: self._resize_rows(generation, limit))

    def _fit_row(self, row):
        model = self.model()
        if not model or not 0 <= row < model.rowCount():
            return
        required = 32
        for column in range(model.columnCount()):
            idx = model.index(row, column)
            doc = QTextDocument()
            doc.setDocumentMargin(4)
            doc.setDefaultFont(self.font())
            doc.setPlainText(str(idx.data(Qt.DisplayRole) or ''))
            doc.setTextWidth(max(80, self.columnWidth(column) - 10))
            required = max(required, int(doc.size().height()) + 8)
        if self.rowHeight(row) != required:
            self.setRowHeight(row, required)
