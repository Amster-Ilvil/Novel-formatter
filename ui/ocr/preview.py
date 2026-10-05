from __future__ import annotations

import math

from PySide6.QtWidgets import QLabel, QFrame, QRubberBand
from PySide6.QtCore import Qt, QRect, QPoint
from PySide6.QtGui import QPixmap, QImage, QPainter, QCursor

from ui.common.styling import ACC

LIGHT_IMAGE_PREVIEW_STYLE = f"""
background: transparent;
color: #5B6B80;
border: none;
padding: 0px;
"""

class OCRCropPreview(QLabel):
    """
    显示当前图片，支持拖框选定识别区域（替代原来"顶部/底部百分比"的裁剪方式）。
    框选矩形以"相对当前显示图片的归一化坐标 [0,1]"保存，与图片实际分辨率无关；
    OCR 运行时会把每一页的当前处理图片实时换到这里显示，方便对照框选范围。

    交互方式和 PageManagerTab 里缩略图拉框多选一致：QRubberBand 走拖拽过程的
    视觉反馈，松手后用一个半透明 QFrame 常驻显示已确定的框选区域。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(LIGHT_IMAGE_PREVIEW_STYLE)
        self.setText("选择输入后在此显示图片，可拖框选定识别区域")
        self.setWordWrap(True)
        self.setCursor(QCursor(Qt.CrossCursor))

        self._orig_pixmap: QPixmap | None = None
        self._scaled_source_size: tuple[int, int] | None = None
        self._source_display_offset: tuple[float, float] = (0.0, 0.0)
        self._rect_norm: tuple[float, float, float, float] | None = None
        self._rubber_origin: QPoint | None = None
        self._rubber_band = QRubberBand(QRubberBand.Rectangle, self)

        self._overlay = QFrame(self)
        self._overlay.setStyleSheet(f"background: rgba(74,99,211,60); border: 2px solid {ACC};")
        # 让点击穿透到底下的 QLabel，否则已经框选过一次之后，想在旧框内重新拖拽
        # 会被这个纯展示用的覆盖层"吃掉"鼠标事件，看起来就像只能拖一次改不了。
        self._overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._overlay.hide()

        # Real-time masked-column preview.  Boxes are stored in normalized image
        # coordinates so they remain aligned after the QLabel is resized.
        self._column_rects_norm: list[tuple[float, float, float, float]] = []
        self._column_overlays: list[QFrame] = []
        self._column_labels: list[QLabel] = []
        # Final native-pixel OCR reveal boxes are deliberately separate from
        # detector boxes: red marks detected columns, blue marks source pixels
        # preserved for OCR before Ruby is blanked and white context is added.
        self._input_rects_norm: list[tuple[float, float, float, float]] = []
        self._input_overlays: list[QFrame] = []
        self._show_detector_boxes = True
        self._show_input_boxes = True
        # Review-only one-character frames. They are independent from the
        # physical-column boxes and are only populated while the user has
        # explicitly enabled the per-character review controls.
        self._character_rects_norm: list[tuple[float, float, float, float]] = []
        self._character_overlays: list[QFrame] = []

    def _clear_column_overlays(self):
        self._column_rects_norm = []
        for overlay in (*self._column_overlays, *self._column_labels):
            overlay.hide()
            overlay.deleteLater()
        self._column_overlays = []
        self._column_labels = []

    def _clear_input_overlays(self):
        self._input_rects_norm = []
        for overlay in self._input_overlays:
            overlay.hide()
            overlay.deleteLater()
        self._input_overlays = []

    def _clear_character_overlays(self):
        self._character_rects_norm = []
        for overlay in self._character_overlays:
            overlay.hide()
            overlay.deleteLater()
        self._character_overlays = []

    def clear_column_boxes(self):
        self._clear_column_overlays()

    def clear_input_boxes(self):
        self._clear_input_overlays()

    def clear_character_boxes(self):
        self._clear_character_overlays()

    def set_detector_boxes_visible(self, visible: bool):
        self._show_detector_boxes = bool(visible)
        self._update_column_overlay_geometry()

    def set_input_boxes_visible(self, visible: bool):
        self._show_input_boxes = bool(visible)
        self._update_input_overlay_geometry()

    def clear_preview(self):
        """Clear both the displayed image and every non-document overlay."""
        self._orig_pixmap = None
        self._clear_column_overlays()
        self._clear_input_overlays()
        self._clear_character_overlays()
        self.clear()
        self.setText("选择输入后在此显示图片，可拖框选定识别区域")
        self._update_overlay_geometry()

    def _set_column_rects(self, rects):
        self._clear_column_overlays()
        for index, raw in enumerate(rects or [], start=1):
            if not isinstance(raw, (list, tuple)) or len(raw) < 4:
                continue
            x0, y0, x1, y1 = [max(0.0, min(1.0, float(v))) for v in raw[:4]]
            left, right = sorted((x0, x1))
            top, bottom = sorted((y0, y1))
            if right - left <= 0.0001 or bottom - top <= 0.0001:
                continue
            self._column_rects_norm.append((left, top, right, bottom))
            frame = QFrame(self)
            frame.setStyleSheet(
                "background: transparent; border: 2px solid rgb(220,38,38);"
            )
            frame.setAttribute(Qt.WA_TransparentForMouseEvents)
            self._column_overlays.append(frame)

            label = QLabel(str(index), self)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet(
                "background: rgba(255,255,255,230); border: 1px solid rgb(220,38,38); "
                "color: rgb(220,38,38); font-size: 11px; font-weight: 700; padding: 0 3px;"
            )
            label.adjustSize()
            label.setFixedSize(max(18, label.sizeHint().width()), 18)
            label.setAttribute(Qt.WA_TransparentForMouseEvents)
            self._column_labels.append(label)
        self._update_column_overlay_geometry()

    def _set_input_rects(self, rects):
        self._clear_input_overlays()
        for raw in rects or []:
            if not isinstance(raw, (list, tuple)) or len(raw) < 4:
                continue
            try:
                x0, y0, x1, y1 = [
                    max(0.0, min(1.0, float(value))) for value in raw[:4]
                ]
            except (TypeError, ValueError):
                continue
            left, right = sorted((x0, x1))
            top, bottom = sorted((y0, y1))
            if right - left <= 0.0001 or bottom - top <= 0.0001:
                continue
            self._input_rects_norm.append((left, top, right, bottom))
            overlay = QFrame(self)
            overlay.setStyleSheet(
                "background: rgba(22,119,255,3); "
                "border: 2px solid rgb(22,119,255); border-radius: 2px;"
            )
            overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
            self._input_overlays.append(overlay)
        self._update_input_overlay_geometry()

    def _set_character_rects(self, rects):
        self._clear_character_overlays()
        for raw in rects or []:
            if not isinstance(raw, (list, tuple)) or len(raw) < 4:
                continue
            try:
                x0, y0, x1, y1 = [
                    max(0.0, min(1.0, float(value))) for value in raw[:4]
                ]
            except (TypeError, ValueError):
                continue
            left, right = sorted((x0, x1))
            top, bottom = sorted((y0, y1))
            if right - left <= 0.0001 or bottom - top <= 0.0001:
                continue
            self._character_rects_norm.append((left, top, right, bottom))
            overlay = QFrame(self)
            overlay.setStyleSheet(
                "background: rgba(22,119,255,5); "
                "border: 2px solid rgba(22,119,255,225); border-radius: 2px;"
            )
            overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
            self._character_overlays.append(overlay)
        self._update_character_overlay_geometry()

    def set_image(self, path: str):
        pm = QPixmap(path)
        if pm.isNull():
            return
        self._orig_pixmap = pm
        self._clear_column_overlays()
        self._clear_input_overlays()
        self._clear_character_overlays()
        self.setText("")
        self._rescale()
        self._update_overlay_geometry()

    def set_image_with_columns(
        self, path: str, column_rects, character_rects=None, input_rects=None
    ):
        """Load a retained page with detector, OCR-input and character boxes."""
        pm = QPixmap(path)
        if pm.isNull():
            return False
        self._orig_pixmap = pm
        self.setText("")
        self._rescale()
        self._set_input_rects(input_rects or [])
        self._set_column_rects(column_rects)
        self._set_character_rects(character_rects or [])
        self._update_overlay_geometry()
        return True

    def set_image_data(self, qimage: QImage):
        """
        跟 set_image 一样，只是接收一个已经在后台线程里解码/缩小好的 QImage
        （OCR 运行时实时预览用这个），主线程这边只做一次便宜的 QPixmap 转换，
        不用再读一次原图文件。
        """
        if qimage.isNull():
            return
        self._orig_pixmap = QPixmap.fromImage(qimage)
        self._clear_column_overlays()
        self._clear_input_overlays()
        self._clear_character_overlays()
        self.setText("")
        self._rescale()
        self._update_overlay_geometry()

    def set_column_preview_data(
        self, qimage: QImage, column_rects, character_rects=None, input_rects=None
    ):
        """Display one page with detector, OCR-input and review-character boxes."""
        if qimage.isNull():
            return
        self._orig_pixmap = QPixmap.fromImage(qimage)
        self.setText("")
        self._rescale()
        self._set_input_rects(input_rects or [])
        self._set_column_rects(column_rects)
        self._set_character_rects(character_rects or [])
        self._update_overlay_geometry()

    def clear_rect(self):
        self._rect_norm = None
        self._overlay.hide()
        self._rescale()
        self._update_overlay_geometry()

    def get_crop_rect(self):
        return self._rect_norm

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()
        self._update_overlay_geometry()
        self._update_input_overlay_geometry()
        self._update_column_overlay_geometry()
        self._update_character_overlay_geometry()

    def _rescale(self):
        if self._orig_pixmap is None:
            return
        content = self.contentsRect()
        if content.width() <= 0 or content.height() <= 0:
            return
        scaled = self._orig_pixmap.scaled(
            content.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        width, height = scaled.width(), scaled.height()
        offset_x = (content.width() - width) / 2
        offset_y = (content.height() - height) / 2
        if self._rect_norm:
            x0, _, x1, _ = self._rect_norm
            desired = content.width() / 2 - ((x0 + x1) / 2) * width
            # Keep the selected region visible while centering it whenever the
            # viewport is wide enough. The full source image remains available
            # to the OCR pipeline; only its preview position changes.
            min_offset = -x0 * width
            max_offset = content.width() - x1 * width
            offset_x = max(min_offset, min(max_offset, desired))

        canvas = QPixmap(content.size())
        canvas.fill(Qt.transparent)
        painter = QPainter(canvas)
        painter.drawPixmap(int(round(offset_x)), int(round(offset_y)), scaled)
        painter.end()
        self._scaled_source_size = (width, height)
        self._source_display_offset = (
            content.x() + offset_x,
            content.y() + offset_y,
        )
        self.setPixmap(canvas)

    def _display_geometry(self):
        """Scaled source image geometry in label coordinates."""
        if self._orig_pixmap is None or self._scaled_source_size is None:
            return None
        x, y = self._source_display_offset
        w, h = self._scaled_source_size
        return x, y, w, h

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton or self._orig_pixmap is None:
            return
        self._rubber_origin = event.pos()
        self._rubber_band.setGeometry(QRect(self._rubber_origin, event.pos()).normalized())
        self._rubber_band.show()

    def mouseMoveEvent(self, event):
        if self._rubber_origin is None:
            return
        self._rubber_band.setGeometry(QRect(self._rubber_origin, event.pos()).normalized())

    def mouseReleaseEvent(self, event):
        if self._rubber_origin is None:
            return
        rect = QRect(self._rubber_origin, event.pos()).normalized()
        self._rubber_band.hide()
        self._rubber_origin = None

        geo = self._display_geometry()
        if not geo or rect.width() < 6 or rect.height() < 6:
            return
        dx, dy, dw, dh = geo
        x0 = max(dx, min(rect.left(), dx + dw))
        x1 = max(dx, min(rect.right(), dx + dw))
        y0 = max(dy, min(rect.top(), dy + dh))
        y1 = max(dy, min(rect.bottom(), dy + dh))
        if x1 - x0 < 6 or y1 - y0 < 6:
            return
        self._rect_norm = ((x0 - dx) / dw, (y0 - dy) / dh, (x1 - dx) / dw, (y1 - dy) / dh)
        self._rescale()
        self._update_overlay_geometry()

    def _update_overlay_geometry(self):
        if not self._rect_norm:
            self._overlay.hide()
            return
        geo = self._display_geometry()
        if not geo:
            self._overlay.hide()
            return
        dx, dy, dw, dh = geo
        x0, y0, x1, y1 = self._rect_norm
        self._overlay.setGeometry(QRect(
            int(dx + x0 * dw), int(dy + y0 * dh),
            int((x1 - x0) * dw), int((y1 - y0) * dh),
        ))
        self._overlay.show()
        self._overlay.raise_()
        self._update_input_overlay_geometry()
        self._update_column_overlay_geometry()
        self._update_character_overlay_geometry()

    def _update_input_overlay_geometry(self):
        geo = self._display_geometry()
        if not self._show_input_boxes or not geo or not self._input_rects_norm:
            for overlay in self._input_overlays:
                overlay.hide()
            return
        dx, dy, dw, dh = geo
        display_right = dx + dw
        display_bottom = dy + dh
        for overlay, (x0, y0, x1, y1) in zip(self._input_overlays, self._input_rects_norm):
            frame_left = max(dx, dx + x0 * dw)
            frame_top = max(dy, dy + y0 * dh)
            frame_right = min(display_right, dx + x1 * dw)
            frame_bottom = min(display_bottom, dy + y1 * dh)
            overlay.setGeometry(QRect(
                int(frame_left), int(frame_top),
                max(2, int(math.ceil(frame_right - frame_left))),
                max(2, int(math.ceil(frame_bottom - frame_top))),
            ))
            overlay.show()
            overlay.raise_()

    def _update_column_overlay_geometry(self):
        geo = self._display_geometry()
        if not self._show_detector_boxes or not geo or not self._column_rects_norm:
            for overlay in (*self._column_overlays, *self._column_labels):
                overlay.hide()
            return
        dx, dy, dw, dh = geo
        preview_outset = 2  # diagnostic outline stays outside the OCR crop
        display_right = dx + dw
        display_bottom = dy + dh
        common_frame_top = min(
            max(dy, int(dy + y0 * dh) - preview_outset)
            for _x0, y0, _x1, _y1 in self._column_rects_norm
        )
        common_label_y = max(
            0, int(common_frame_top) - max(label.height() for label in self._column_labels)
        )
        for overlay, label, (x0, y0, x1, y1) in zip(
            self._column_overlays, self._column_labels, self._column_rects_norm
        ):
            crop_left = int(dx + x0 * dw)
            crop_top = int(dy + y0 * dh)
            crop_right = int(math.ceil(dx + x1 * dw))
            crop_bottom = int(math.ceil(dy + y1 * dh))
            frame_left = max(dx, crop_left - preview_outset)
            frame_top = max(dy, crop_top - preview_outset)
            frame_right = min(display_right, crop_right + preview_outset)
            frame_bottom = min(display_bottom, crop_bottom + preview_outset)
            overlay.setGeometry(QRect(
                int(frame_left), int(frame_top),
                max(2, int(frame_right - frame_left)),
                max(2, int(frame_bottom - frame_top)),
            ))
            overlay.show()
            overlay.raise_()
            label_x = max(
                dx,
                min(
                    display_right - label.width(),
                    int((frame_left + frame_right - label.width()) / 2),
                ),
            )
            label.setGeometry(label_x, common_label_y, label.width(), label.height())
            label.show()
            label.raise_()

    def _update_character_overlay_geometry(self):
        geo = self._display_geometry()
        if not geo or not self._character_rects_norm:
            for overlay in self._character_overlays:
                overlay.hide()
            return
        dx, dy, dw, dh = geo
        for overlay, (x0, y0, x1, y1) in zip(
            self._character_overlays, self._character_rects_norm
        ):
            overlay.setGeometry(QRect(
                int(dx + x0 * dw), int(dy + y0 * dh),
                max(3, int((x1 - x0) * dw)), max(3, int((y1 - y0) * dh)),
            ))
            overlay.show()
            overlay.raise_()
