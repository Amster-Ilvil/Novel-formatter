"""Read-only, geometry-backed full-page <-> physical-column <-> sentence navigator.

No OCR, inference, text editing, or evidence file rewriting occurs here.  An
absent/ambiguous geometric lineage is shown as unavailable, never guessed.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import math
from pathlib import Path

from PySide6.QtCore import Qt, QRectF, Signal, QSize, QPointF
from PySide6.QtGui import QColor, QImageReader, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget, QSizePolicy


@dataclass(frozen=True)
class PhysicalRegion:
    page: int
    column_id: str
    bbox: tuple[float, float, float, float]  # normalized x, y, width, height


def _region(raw):
    if not isinstance(raw, dict):
        return None
    try:
        page = int(raw.get('page') or 0)
        column = str(raw.get('column_id') or '')
        x, y, w, h = (float(v) for v in (raw.get('bbox') or ())[:4])
    except (ValueError, TypeError, OverflowError):
        return None
    if page <= 0 or not column or not all(math.isfinite(v) for v in (x, y, w, h)):
        return None
    # Canonical OCR geometry is relative to the actual scanned page.
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > 1.0001 or y + h > 1.0001:
        return None
    return PhysicalRegion(page, column, (x, y, min(1.0 - x, w), min(1.0 - y, h)))


def _real_file(value) -> str:
    try:
        path = Path(str(value or ''))
        return str(path) if path.is_file() else ''
    except (OSError, ValueError, TypeError):
        return ''


class PageColumnIndex:
    """Immutable mapping derived only from stored source geometry and stable IDs."""

    def __init__(self, document, comparison_rows, fallback_page_images=()):
        self.paths: dict[int, str] = {}
        self.regions_by_page: dict[int, tuple[PhysicalRegion, ...]] = {}
        self.rows_by_column: dict[str, tuple[int, ...]] = {}
        self.columns_by_row: dict[int, tuple[str, ...]] = {}
        self.pages_by_row: dict[int, tuple[int, ...]] = {}
        self._regions_by_column: dict[str, tuple[PhysicalRegion, ...]] = {}

        for index, value in enumerate(fallback_page_images or (), 1):
            path = _real_file(value)
            if path:
                self.paths[index] = path
        for page in getattr(document, 'pages', ()) or ():
            try:
                number = int(getattr(page, 'page_no', 0) or 0)
            except (ValueError, TypeError, OverflowError):
                continue
            path = _real_file(getattr(page, 'image_path', ''))
            if number > 0 and path:
                self.paths[number] = path

        dedup = {}
        # The canonical document may store one region at block level and again
        # inside one or more sentence groups.  Deduplicate by exact coordinates.
        for block in getattr(document, 'blocks', ()) or ():
            meta = getattr(block, 'metadata', None)
            if not isinstance(meta, dict):
                continue
            groups = meta.get('ocr_review_sentence_groups') or ()
            if isinstance(groups, dict):
                groups = (groups,)
            sources = [meta.get('ocr_review_regions')]
            if isinstance(groups, (list, tuple)):
                sources += [g.get('regions') for g in groups if isinstance(g, dict)]
            for source in sources:
                if isinstance(source, dict):
                    source = (source,)
                if not isinstance(source, (list, tuple)):
                    continue
                for raw in source:
                    candidate = _region(raw)
                    if candidate is None:
                        continue
                    key = (candidate.page, candidate.column_id, candidate.bbox)
                    dedup[key] = candidate
                    raw_path = _real_file(raw.get('page_path'))
                    if raw_path and candidate.page not in self.paths:
                        self.paths[candidate.page] = raw_path

        pages = defaultdict(list)
        columns = defaultdict(list)
        for region in dedup.values():
            pages[region.page].append(region)
            columns[region.column_id].append(region)
        self.regions_by_page = {p: tuple(sorted(v, key=lambda r: -r.bbox[0])) for p, v in pages.items()}
        self._regions_by_column = {col: tuple(v) for col, v in columns.items()}

        rows_for = defaultdict(list)
        for index, row in enumerate(comparison_rows or ()):
            # Never synthesize a column from similar text: this identity is the
            # only safe link back into the existing adjudication state.
            ids = tuple(dict.fromkeys(str(v) for v in (getattr(row, 'column_ids', ()) or ()) if str(v)))
            self.columns_by_row[index] = ids
            available_pages = list(dict.fromkeys(
                region.page for cid in ids for region in columns.get(cid, ())
            ))
            self.pages_by_row[index] = tuple(available_pages)
            for cid in ids:
                rows_for[cid].append(index)
        self.rows_by_column = {col: tuple(indices) for col, indices in rows_for.items()}

    def location_for_row(self, row_index: int, preferred_page: int = 0):
        pages = self.pages_by_row.get(int(row_index), ())
        if not pages:
            return 0, ()
        selected_page = preferred_page if preferred_page in pages else pages[0]
        ids = set(self.columns_by_row.get(int(row_index), ()))
        return selected_page, tuple(r for r in self.regions_by_page.get(selected_page, ()) if r.column_id in ids)

    def row_for_column(self, column_id: str, current: int, states=()):
        rows = self.rows_by_column.get(str(column_id), ())
        if not rows:
            return None
        if int(current) in rows:
            return int(current)
        return next((i for i in rows if i < len(states) and bool(getattr(states[i], 'unresolved', False))), rows[0])


class PageColumnNavigator(QWidget):
    """Fit-to-window whole page; hit testing uses source coordinates, not crops."""
    column_clicked = Signal(str)
    zoom_changed = Signal(float)  # multiplier relative to fit-to-panel

    def __init__(self, parent=None):
        super().__init__(parent)
        self._path = ''
        self._source_size = QSize()
        self._pixmap = QPixmap()
        self._regions: tuple[PhysicalRegion, ...] = ()
        self._active_ids: frozenset[str] = frozenset()
        self._last_rect = QRectF()
        # Keep a separate, user-controlled full-page magnification across
        # sentence changes and page navigation; never edit the source pixels.
        self._fit_multiplier = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._drag_origin: QPointF | None = None
        self._drag_last: QPointF | None = None
        self._dragging = False
        self._drag_button = Qt.NoButton
        self.setMinimumSize(210, 280)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCursor(Qt.CrossCursor)
        self.setToolTip('点击蓝色列框定位句子 · Ctrl+滚轮缩放 · 放大后拖动图片平移')

    @property
    def zoom_multiplier(self) -> float:
        return self._fit_multiplier

    def fit_to_window(self) -> None:
        self._fit_multiplier = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.zoom_changed.emit(self._fit_multiplier)
        self.update()

    def zoom_by(self, factor: float) -> None:
        self.set_zoom(self._fit_multiplier * float(factor))

    def set_zoom(self, multiplier: float) -> None:
        if not math.isfinite(float(multiplier)):
            return
        previous = self._fit_multiplier
        self._fit_multiplier = max(1.0, min(6.0, float(multiplier)))
        if self._fit_multiplier == 1.0:
            self._pan = QPointF(0.0, 0.0)
        else:
            change = self._fit_multiplier / previous
            self._pan = QPointF(self._pan.x() * change, self._pan.y() * change)
        self._clamp_pan()
        self.zoom_changed.emit(self._fit_multiplier)
        self.update()

    def _available_rect(self):
        return QRectF(self.rect().adjusted(6, 6, -6, -6))

    def _clamp_pan(self) -> None:
        if self._pixmap.isNull():
            self._pan = QPointF(0.0, 0.0)
            return
        available = self._available_rect()
        if available.isEmpty():
            return
        fit = min(available.width() / self._pixmap.width(),
                  available.height() / self._pixmap.height())
        width = self._pixmap.width() * fit * self._fit_multiplier
        height = self._pixmap.height() * fit * self._fit_multiplier
        limit_x = max(0.0, (width - available.width()) / 2)
        limit_y = max(0.0, (height - available.height()) / 2)
        self._pan = QPointF(max(-limit_x, min(limit_x, self._pan.x())),
                            max(-limit_y, min(limit_y, self._pan.y())))

    def _ensure_active_column_visible(self) -> None:
        if self._fit_multiplier <= 1 or not self.active_regions:
            return
        region = self.active_regions[0]
        x, y, w, h = region.bbox
        center_x = x + w / 2
        center_y = y + h / 2
        rect = self._display_rect()
        available = self._available_rect()
        # Center the selected physical column without resetting manual zoom.
        self._pan += QPointF(available.center().x() - (rect.left() + center_x * rect.width()),
                             available.center().y() - (rect.top() + center_y * rect.height()))
        self._clamp_pan()

    def focus_column(self, column_id: str) -> None:
        column_id = str(column_id)
        if any(region.column_id == column_id for region in self._regions):
            self._active_ids = frozenset((column_id,))
            self._ensure_active_column_visible()
            self.update()

    @property
    def image_path(self):
        return self._path

    @property
    def active_regions(self):
        return tuple(r for r in self._regions if r.column_id in self._active_ids)

    def set_page(self, image_path: str, regions=(), active_columns=()):
        path = _real_file(image_path)
        if path != self._path:
            self._path = path
            self._pixmap = QPixmap()
            self._source_size = QSize()
            if path:
                reader = QImageReader(path)
                reader.setAutoTransform(False)  # OCR bbox is measured on stored, unrotated pixels
                size = reader.size()
                if size.isValid():
                    self._source_size = size
                    factor = min(1.0, 2300 / max(1, size.width()), 2300 / max(1, size.height()))
                    if factor < 1:
                        reader.setScaledSize(QSize(max(1, int(size.width() * factor)), max(1, int(size.height() * factor))))
                image = reader.read()
                if not image.isNull():
                    self._pixmap = QPixmap.fromImage(image)
                    if not self._source_size.isValid():
                        self._source_size = image.size()
        previous_ids = self._active_ids
        self._regions = tuple(regions or ())
        self._active_ids = frozenset(str(value) for value in (active_columns or ()))
        if self._active_ids != previous_ids and self._fit_multiplier > 1:
            self._ensure_active_column_visible()
        self._clamp_pan()
        self.update()

    def _display_rect(self):
        if self._pixmap.isNull():
            return QRectF()
        available = self._available_rect()
        width = float(self._pixmap.width())
        height = float(self._pixmap.height())
        ratio = min(available.width() / width, available.height() / height) * self._fit_multiplier
        dw, dh = width * ratio, height * ratio
        return QRectF(available.left() + (available.width() - dw) / 2 + self._pan.x(),
                      available.top() + (available.height() - dh) / 2 + self._pan.y(), dw, dh)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor('#F3F6FB'))
        rect = self._display_rect()
        self._last_rect = rect
        if rect.isEmpty():
            painter.setPen(QColor('#687A92'))
            painter.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap,
                             '此句缺少可验证的全页原图或物理列坐标\n仍可查看右侧原有裁片')
            painter.end()
            return
        # Zoomed pages may extend beyond the widget. Keep the full-page
        # drawing (and clickable boxes) inside the actual evidence canvas.
        painter.setClipRect(self._available_rect())
        painter.drawPixmap(rect.toRect(), self._pixmap)
        for region in self._regions:
            x, y, w, h = region.bbox
            box = QRectF(rect.left() + x * rect.width(), rect.top() + y * rect.height(),
                         w * rect.width(), h * rect.height())
            active = region.column_id in self._active_ids
            painter.fillRect(box, QColor(37, 99, 235, 43 if active else 9))
            painter.setPen(QPen(QColor('#2563EB' if active else '#8296B6'), 2.5 if active else 1.0))
            painter.drawRect(box)
        painter.end()

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            steps = event.angleDelta().y() / 120.0
            if steps:
                self.zoom_by(1.18 ** steps)
                event.accept()
                return
        super().wheelEvent(event)

    def mousePressEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.MiddleButton):
            if self._display_rect().contains(event.position()):
                self._drag_origin = QPointF(event.position())
                self._drag_last = QPointF(event.position())
                self._drag_button = event.button()
                self._dragging = False
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_origin is not None and self._fit_multiplier > 1:
            current = QPointF(event.position())
            if (current - self._drag_origin).manhattanLength() > 5:
                self._dragging = True
            if self._dragging and self._drag_last is not None:
                self._pan += current - self._drag_last
                self._clamp_pan()
                self.update()
            self._drag_last = current
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._drag_origin is None or event.button() != self._drag_button:
            return super().mouseReleaseEvent(event)
        origin = self._drag_origin
        was_dragged = self._dragging or (
            (QPointF(event.position()) - origin).manhattanLength() > 5 and self._fit_multiplier > 1
        )
        self._drag_origin = self._drag_last = None
        self._drag_button = Qt.NoButton
        self._dragging = False
        if was_dragged or event.button() != Qt.LeftButton:
            event.accept()
            return
        rect = self._display_rect()
        point = event.position()
        if not rect.contains(point) or not self._available_rect().contains(point):
            event.accept()
            return
        x = (point.x() - rect.left()) / rect.width()
        y = (point.y() - rect.top()) / rect.height()
        found = []
        for region in self._regions:
            left, top, width, height = region.bbox
            if left <= x <= left + width and top <= y <= top + height:
                found.append(region)
        if found:
            chosen = min(found, key=lambda r: r.bbox[2] * r.bbox[3])
            self.column_clicked.emit(chosen.column_id)
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._clamp_pan()
