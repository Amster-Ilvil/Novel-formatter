from __future__ import annotations

import hashlib
import tempfile
import threading
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor, QImage, QImageReader
from PySide6.QtWidgets import QDialog, QLabel, QMessageBox, QPushButton, QVBoxLayout

from ui.common.styling import MUTED
from ui.dialogs import show_error_dialog
from ui.ocr.preview import OCRCropPreview


class OCRPreviewController:
    """Own OCR preview history, navigation, overlays and diagnostic rendering.

    Recognition geometry remains owned by the OCR runtime.  This controller only
    consumes retained geometry or runs the existing preview-only detector path.
    """

    def __init__(self, tab):
        self._tab = tab

    def _on_live_preview_toggled(self, enabled: bool):
        """Apply the live-preview switch immediately, including during OCR.

        The worker never reads QWidget state.  A threading.Event is safe to
        inspect from adapters/callbacks and the GUI signal handlers perform the
        same check again so previews already queued before the click cannot
        leak through after the switch is turned off.
        """
        self = self._tab
        if bool(enabled):
            self._live_preview_enabled_event.set()
            restored_tip = self._preview_filename_lbl.toolTip()
            for prefix in (
                "实时预览已关闭；OCR 仍在后台正常运行。",
                "实时预览已关闭；不会生成或刷新新的预览图。",
            ):
                if restored_tip.startswith(prefix):
                    restored_tip = restored_tip[len(prefix):]
            self._preview_filename_lbl.setToolTip(restored_tip)
            if "实时预览已关闭" in self._preview_filename_lbl.text():
                self._preview_filename_lbl.setText("当前图片：等待下一张识别图片…")
        else:
            self._live_preview_enabled_event.clear()
            # Do not erase the crop rectangle or retained history: both are
            # independent features.  Merely stop following/adding new pages.
            self._preview_follow_latest = False
            current_tip = self._preview_filename_lbl.toolTip()
            if not current_tip.startswith("实时预览已关闭；"):
                self._preview_filename_lbl.setToolTip(
                    "实时预览已关闭；OCR 仍在后台正常运行。" + current_tip
                )
        self._update_preview_navigation()


    def _live_preview_is_enabled(self) -> bool:
        self = self._tab
        return bool(self._live_preview_enabled_event.is_set())


    def _reset_preview_history(self, *, keep_current_image: bool = True):
        """Reset page navigation without deleting any retained temporary files."""
        self = self._tab
        self._preview_items = []
        self._preview_item_index_by_key = {}
        self._preview_index = -1
        self._preview_follow_latest = True
        self._active_preview_source_path = None
        if not keep_current_image:
            self._preview.clear_preview()
        self._update_preview_navigation()


    def _update_preview_navigation(self):
        self = self._tab
        count = len(getattr(self, "_preview_items", []) or [])
        index = int(getattr(self, "_preview_index", -1))
        previous = getattr(self, "_preview_prev_btn", None)
        following = getattr(self, "_preview_next_btn", None)
        page_label = getattr(self, "_preview_page_lbl", None)
        if previous is not None:
            previous.setEnabled(count > 1 and index > 0)
        if following is not None:
            following.setEnabled(count > 1 and 0 <= index < count - 1)
        if page_label is not None:
            page_label.setText(f"{index + 1} / {count}" if count and index >= 0 else "0 / 0")


    @staticmethod
    def _build_review_character_rects(
        image_path: str,
        column_rects,
        *,
        maximum_boxes: int = 2000,
    ) -> list[tuple[float, float, float, float]]:
        """Build review-only one-glyph boxes for a retained preview page.

        This helper is intentionally outside the OCR recognition pipeline.  It
        reads the already retained page snapshot, crops the existing physical
        column rectangles, and runs the black-ink slider only for visualization.
        OCR text, page order, column masks and document blocks are untouched.
        """
        from PIL import Image
        from adapters.handwriting_image_tools import segment_black_ink_glyphs_slider

        rects = list(column_rects or [])
        if not image_path or not rects:
            return []
        output: list[tuple[float, float, float, float]] = []
        with Image.open(str(image_path)) as opened:
            page = opened.convert("RGB")
        try:
            width, height = page.size
            if width <= 0 or height <= 0:
                return []
            for raw in rects:
                if len(output) >= maximum_boxes:
                    break
                if not isinstance(raw, (list, tuple)) or len(raw) < 4:
                    continue
                try:
                    nx0, ny0, nx1, ny1 = [float(value) for value in raw[:4]]
                except (TypeError, ValueError):
                    continue
                left, right = sorted((
                    max(0.0, min(1.0, nx0)),
                    max(0.0, min(1.0, nx1)),
                ))
                top, bottom = sorted((
                    max(0.0, min(1.0, ny0)),
                    max(0.0, min(1.0, ny1)),
                ))
                x0 = max(0, min(width - 1, int(round(left * width))))
                x1 = max(x0 + 1, min(width, int(round(right * width))))
                y0 = max(0, min(height - 1, int(round(top * height))))
                y1 = max(y0 + 1, min(height, int(round(bottom * height))))
                column = page.crop((x0, y0, x1, y1)).convert("RGB")
                segmented = None
                segments = []
                try:
                    segmented, segments, _info = segment_black_ink_glyphs_slider(
                        column,
                        apply_main_band_mask=True,
                        expected_count=None,
                        character_anchors=None,
                    )
                    crop_w = max(1, x1 - x0)
                    crop_h = max(1, y1 - y0)
                    for segment in segments:
                        if len(output) >= maximum_boxes:
                            break
                        sx0 = max(0, min(crop_w, int(segment.x0)))
                        sx1 = max(0, min(crop_w, int(segment.x1)))
                        sy0 = max(0, min(crop_h, int(segment.y0)))
                        sy1 = max(0, min(crop_h, int(segment.y1)))
                        if sx1 <= sx0 or sy1 <= sy0:
                            continue
                        output.append((
                            (x0 + sx0) / float(width),
                            (y0 + sy0) / float(height),
                            (x0 + sx1) / float(width),
                            (y0 + sy1) / float(height),
                        ))
                finally:
                    for segment in segments:
                        try:
                            segment.image.close()
                        except Exception:
                            pass
                    if segmented is not None:
                        try:
                            segmented.close()
                        except Exception:
                            pass
                    column.close()
        finally:
            page.close()
        output.sort(key=lambda box: (box[1], -box[0], box[3], box[2]))
        return output


    def _apply_review_preview_boxes(self, key: str, payload) -> None:
        self = self._tab
        data = dict(payload or {}) if isinstance(payload, dict) else {}
        generation = int(data.get("generation", -1) or -1)
        if generation != int(getattr(self, "_review_preview_generation", 0)):
            return
        index = self._preview_item_index_by_key.get(str(key))
        if index is None or not (0 <= index < len(self._preview_items)):
            return
        item = self._preview_items[index]
        item["glyph_rects"] = list(data.get("rects") or [])
        item.pop("_glyph_preview_pending_generation", None)
        enabled = bool(
            getattr(self, "_review_preview_run_active", False)
            or (
                hasattr(self, "_handwriting_trace_check")
                and self._handwriting_trace_check.isChecked()
            )
        )
        if enabled and index == self._preview_index:
            self._show_preview_page(index)


    def _refresh_review_preview_boxes(self) -> None:
        """Refresh the visible preview after the review checkbox changes."""
        self = self._tab
        self._review_preview_generation += 1
        generation = self._review_preview_generation
        enabled = bool(
            self._current_ocr_mode() == "ja_vertical"
            and (
                bool(getattr(self, "_review_preview_run_active", False))
                or (
                    hasattr(self, "_handwriting_trace_check")
                    and self._handwriting_trace_check.isChecked()
                )
            )
        )
        if not enabled:
            if hasattr(self, "_preview"):
                self._preview.clear_character_boxes()
            if self._preview_items and 0 <= self._preview_index < len(self._preview_items):
                self._show_preview_page(self._preview_index)
            return
        if not self._preview_items or not (0 <= self._preview_index < len(self._preview_items)):
            return
        item = self._preview_items[self._preview_index]
        rects = list(item.get("rects") or [])
        path = str(item.get("path") or "")
        key = str(item.get("key") or path)
        if not rects or not path or not Path(path).exists():
            self._preview.clear_character_boxes()
            return
        cached = list(item.get("glyph_rects") or [])
        if cached:
            self._show_preview_page(self._preview_index)
            return
        item["_glyph_preview_pending_generation"] = generation

        def build():
            try:
                glyphs = self._build_review_character_rects(path, rects)
            except Exception:
                glyphs = []
            self.review_preview_boxes_ready.emit(
                key, {"generation": generation, "rects": glyphs}
            )

        threading.Thread(
            target=build,
            daemon=True,
            name=f"review-preview-boxes-{generation}",
        ).start()


    def _show_preview_page(self, index: int, *, manual: bool = False):
        self = self._tab
        if not self._preview_items:
            self._update_preview_navigation()
            return
        index = max(0, min(int(index), len(self._preview_items) - 1))
        item = self._preview_items[index]
        path = str(item.get("path") or "")
        rects = list(item.get("rects") or [])
        input_rects = list(item.get("input_rects") or [])
        show_character_boxes = bool(
            self._current_ocr_mode() == "ja_vertical"
            and (
                bool(getattr(self, "_review_preview_run_active", False))
                or (
                    hasattr(self, "_handwriting_trace_check")
                    and self._handwriting_trace_check.isChecked()
                )
            )
        )
        glyph_rects = list(item.get("glyph_rects") or []) if show_character_boxes else []
        if not path or not Path(path).exists():
            return
        if not self._preview.set_image_with_columns(
            path, rects, glyph_rects, input_rects=input_rects
        ):
            return
        self._preview_index = index
        source_path = str(item.get("source_path") or "")
        self._active_preview_source_path = source_path if source_path and Path(source_path).exists() else None
        if manual:
            self._preview_follow_latest = index == len(self._preview_items) - 1
        stage_names = {
            "split": "分列检测",
            "recognition": "识别完成",
            "plain": "识别完成",
        }
        display_name = str(item.get("name") or Path(path).name)
        stage_text = stage_names.get(str(item.get("stage") or ""), "实时预览")
        self._preview_filename_lbl.setText(f"当前图片：{display_name}")
        self._preview_filename_lbl.setToolTip(str(item.get("source_path") or path))
        self._preview_page_lbl.setToolTip(stage_text)
        self._update_preview_navigation()


    def _show_previous_preview_page(self):
        # This handler intentionally stays independent from OCR run state.
        # Once two retained pages exist, the user may browse backwards while the
        # worker continues processing newer pages in the background.
        self = self._tab
        if self._preview_items and self._preview_index > 0:
            self._show_preview_page(self._preview_index - 1, manual=True)


    def _show_next_preview_page(self):
        self = self._tab
        if (
            self._preview_items
            and 0 <= self._preview_index < len(self._preview_items) - 1
        ):
            self._show_preview_page(self._preview_index + 1, manual=True)


    def _record_preview_page(
        self,
        key: str,
        display_name: str,
        snapshot_path: str,
        rects,
        stage: str,
    ):
        """Insert/update one retained page and preserve manual browsing position."""
        self = self._tab
        if not self._live_preview_is_enabled():
            return
        if not snapshot_path or not Path(snapshot_path).exists():
            return
        old_count = len(self._preview_items)
        was_at_latest = self._preview_follow_latest or self._preview_index >= old_count - 1
        current_key = ""
        if 0 <= self._preview_index < old_count:
            current_key = str(self._preview_items[self._preview_index].get("key") or "")
        if isinstance(rects, dict):
            column_rects = list(rects.get("columns") or [])
            input_rects = list(rects.get("input_rects") or [])
            glyph_rects = list(rects.get("glyphs") or [])
            detector_version = str(rects.get("detector_version") or "")
            try:
                page_ordinal = int(rects.get("page_ordinal", 0) or 0)
            except (TypeError, ValueError):
                page_ordinal = 0
        else:
            column_rects = list(rects or [])
            input_rects = []
            glyph_rects = []
            detector_version = ""
            page_ordinal = 0
        item = {
            "key": str(key),
            "name": str(display_name or Path(snapshot_path).name),
            "path": str(snapshot_path),
            "source_path": str(key),
            "rects": column_rects,
            "input_rects": input_rects,
            "glyph_rects": glyph_rects,
            "stage": str(stage or "plain"),
            "detector_version": detector_version,
            "page_ordinal": page_ordinal,
        }
        existing = self._preview_item_index_by_key.get(str(key))
        if existing is None:
            existing = len(self._preview_items)
            self._preview_item_index_by_key[str(key)] = existing
            self._preview_items.append(item)
        else:
            if page_ordinal <= 0:
                item["page_ordinal"] = int(
                    self._preview_items[existing].get("page_ordinal", 0) or 0
                )
            self._preview_items[existing] = item

        # Parallel OCR engines and asynchronous callbacks can arrive out of
        # order.  Navigation must still follow the authoritative input page
        # order rather than event timing.  Unknown ordinals remain stable at the
        # end through Python's stable sort.
        self._preview_items.sort(
            key=lambda value: (
                int(value.get("page_ordinal", 0) or 0) <= 0,
                int(value.get("page_ordinal", 0) or 0)
                if int(value.get("page_ordinal", 0) or 0) > 0
                else 10**9,
            )
        )
        self._preview_item_index_by_key = {
            str(value.get("key") or ""): index
            for index, value in enumerate(self._preview_items)
        }
        existing = self._preview_item_index_by_key.get(str(key), existing)
        if current_key and current_key in self._preview_item_index_by_key:
            self._preview_index = self._preview_item_index_by_key[current_key]

        should_show = (
            self._preview_index < 0
            or (existing == len(self._preview_items) - 1 and was_at_latest)
            or existing == self._preview_index
        )
        if should_show:
            self._preview_follow_latest = was_at_latest
            target = len(self._preview_items) - 1 if was_at_latest else existing
            self._show_preview_page(target)
        else:
            self._update_preview_navigation()


    @staticmethod
    def _write_preview_snapshot(image_path: str, preview_dir: str, key: str) -> tuple[str, QImage]:
        """Decode/scale in the worker thread and persist a navigation-friendly PNG."""
        import hashlib

        source = str(image_path)
        target_dir = Path(preview_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(str(key).encode("utf-8", errors="replace")).hexdigest()[:20]
        target = target_dir / f"{digest}.png"
        # Split and recognition callbacks for the same page reuse the exact
        # snapshot.  Check before decoding so multi-stage OCR does not repeatedly
        # load and rescale hundreds of high-resolution pages.
        if target.exists() and target.stat().st_size > 0:
            return str(target), QImage()
        reader = QImageReader(source)
        reader.setAutoTransform(True)
        size = reader.size()
        if size.isValid():
            reader.setScaledSize(size.scaled(1400, 1400, Qt.KeepAspectRatio))
        image = reader.read()
        if image.isNull():
            return "", QImage()
        if not image.save(str(target), "PNG"):
            return "", QImage()
        return str(target), image



    def clear_ocr_temporary_files(self, *, closing: bool = False):
        """Delete previews/crops only for explicit OCR clear or application close."""
        self = self._tab
        self._preview_temp_store.cleanup(closing=closing)
        self._reset_preview_history(keep_current_image=closing)
        if not closing:
            self._preview_source_path = None
            self._active_preview_source_path = None
            self._preview.clear_preview()
            self._preview_filename_lbl.setText("当前图片：尚未载入")
            self._preview_filename_lbl.setToolTip("")

    def _load_preview_reference(self):
        """
        选好输入后，把第一张"正文"页加载进裁剪预览框，供用户拖框选定识别区域——
        故意不用第一张图片（可能是封面/插图，版式和正文页不一样，照着它画框会跑偏）。
        """
        self = self._tab
        self._preview.clear_rect()
        self._preview_source_path = None
        self._active_preview_source_path = None
        if not self._pending_inputs:
            if not self._preview_items:
                self._preview_filename_lbl.setText("当前图片：尚未载入")
            return

        image_exts = {'.png', '.jpg', '.jpeg', '.heic', '.tif', '.tiff', '.bmp', '.gif'}
        try:
            from adapters.pdf_input import expand_inputs, natural_sort_key
            work_dir = (self._pending_inputs[0] if Path(self._pending_inputs[0]).is_dir()
                        else str(Path(self._pending_inputs[0]).parent))
            images = sorted(set(expand_inputs(self._pending_inputs, work_dir=work_dir)), key=natural_sort_key)
        except Exception:
            images = []
        images = [p for p in images if Path(p).suffix.lower() in image_exts]
        if not images:
            return

        # 只认页面管理页里用户手动确认过的标注（不再按文件名猜类型）——找第一张
        # 明确标成"正文"的页面作为参照图；没有手动标注就直接用第一张图。
        pm = getattr(self.window(), "_tab_pages", None)
        known = dict(pm.page_overrides) if pm is not None else {}

        candidate = None
        for i, p in enumerate(images):
            if known.get(i + 1) == "paragraph":
                candidate = p
                break
        self._preview_source_path = candidate or images[0]
        if not self._preview_items:
            self._preview.set_image(self._preview_source_path)
            reference_name = Path(self._preview_source_path).name
            self._preview_filename_lbl.setText(f"当前图片：{reference_name}")
            self._preview_filename_lbl.setToolTip(self._preview_source_path)
            self._preview_page_lbl.setText("输入参考图")


    def _current_column_preview_context(self) -> tuple[str | None, list, str, str]:
        """Return the exact source/columns represented by the visible preview page."""
        self = self._tab
        if self._preview_items and 0 <= self._preview_index < len(self._preview_items):
            item = self._preview_items[self._preview_index]
            source = str(item.get("source_path") or "")
            if source and Path(source).exists():
                return (
                    source,
                    list(item.get("rects") or []),
                    str(item.get("stage") or ""),
                    str(item.get("detector_version") or ""),
                )
        source = str(self._active_preview_source_path or self._preview_source_path or "")
        return (source if source and Path(source).exists() else None), [], "", ""


    @staticmethod
    def _columns_from_preview_rects(rects, image_size):
        """Rebuild hard physical slots from the live preview's normalized boxes."""
        from adapters.column_ocr_adapter import DetectedColumn

        width, height = image_size
        columns = []
        for raw in rects or []:
            if not isinstance(raw, (list, tuple)) or len(raw) < 4:
                continue
            try:
                x0, y0, x1, y1 = [max(0.0, min(1.0, float(v))) for v in raw[:4]]
            except (TypeError, ValueError):
                continue
            left, right = sorted((round(x0 * width), round(x1 * width)))
            top, bottom = sorted((round(y0 * height), round(y1 * height)))
            left = max(0, min(width - 1, left)); right = max(left + 1, min(width, right))
            top = max(0, min(height - 1, top)); bottom = max(top + 1, min(height, bottom))
            # Retained live-preview rectangles are now tight detector ink
            # envelopes. Rebuild the internal physical slot as full-height while
            # preserving that tight y-span solely for green-box diagnostics.
            columns.append(DetectedColumn(
                left=left, top=0, right=right, bottom=height,
                hard_left=left, hard_right=right, ink_score=0.0,
                content_spans=((top, bottom),), estimated_chars=0, full_height_slot=True,
            ))
        return sorted(columns, key=lambda column: (-(column.left + column.right), column.top))


    def _preview_column_split(self):
        """Render the column/mask preview for the page currently visible on canvas."""
        self = self._tab
        source_path, retained_rects, retained_stage, retained_detector_version = (
            self._current_column_preview_context()
        )
        crop_rect = self._preview.get_crop_rect()
        if not source_path:
            QMessageBox.warning(self, "无法预览", "请先选择图片或 PDF。")
            return
        if crop_rect is None:
            QMessageBox.warning(self, "请先固定正文区域", "请先在右侧图片上拖框选定纯正文区域，再预览分列。")
            return
        try:
            import io
            from PIL import Image, ImageDraw
            from adapters.apple_vision_adapter import crop_for_ocr
            from adapters.column_ocr_adapter import (
                COLUMN_DETECTOR_VERSION,
                detect_vertical_columns,
                _column_visibility_viewport,
                _recognizer_viewport_mode,
                _column_detector_preview_bounds,
                _column_source_union,
                _tighten_ink_framing,
                _center_wide_column_viewport,
            )

            with tempfile.TemporaryDirectory(prefix="novel_formatter_column_preview_") as temp_dir:
                cropped_path = crop_for_ocr(
                    source_path,
                    crop_rect=crop_rect,
                    out_dir=temp_dir,
                )
                with Image.open(cropped_path) as opened_for_size:
                    cropped_size = opened_for_size.size
                # During/after OCR, reuse the exact physical slots painted on the
                # visible retained page.  This makes the dialog, mask and live
                # canvas share one page index and one column lineage.  Before OCR
                # (no retained boxes yet), run the detector normally.
                columns = self._columns_from_preview_rects(retained_rects, cropped_size)
                reused_live_columns = bool(
                    columns
                    and retained_stage in {"split", "recognition"}
                    and retained_detector_version == COLUMN_DETECTOR_VERSION
                )
                if not reused_live_columns:
                    columns = detect_vertical_columns(
                        cropped_path,
                        sensitivity=self._column_sensitivity_spin.value(),
                        padding_percent=self._column_padding_spin.value(),
                        fixed_region_rect=crop_rect,
                        fixed_region_already_masked=True,
                        capture_ruby_candidates=False,
                    )
                if not columns:
                    QMessageBox.warning(
                        self,
                        "未检测到竖列",
                        "当前固定区域没有检测到稳定竖列。请重新框选纯正文区域，或提高分列灵敏度。",
                    )
                    return
                with Image.open(cropped_path) as src:
                    source = src.convert("RGB")
                annotated = source.copy()
                draw = ImageDraw.Draw(annotated)
                # Draw the two geometry layers that matter when diagnosing
                # omissions.  Green = detector box.  Blue = the larger native-
                # pixel source reveal actually preserved for OCR before Ruby is
                # blanked and recognizer-specific white context is added.
                line_width = 1
                preview_outset = max(2, round(min(annotated.size) / 900))
                column_frames = []
                for index, column in enumerate(columns, start=1):
                    in_left, in_top, in_right, in_bottom = _column_source_union(column)
                    input_left = max(0, int(in_left) - preview_outset)
                    input_top = max(0, int(in_top) - preview_outset)
                    input_right = min(annotated.width - 1, int(in_right) - 1 + preview_outset)
                    input_bottom = min(annotated.height - 1, int(in_bottom) - 1 + preview_outset)
                    draw.rectangle(
                        (input_left, input_top, input_right, input_bottom),
                        outline=(22, 119, 255),
                        width=line_width,
                    )
                    det_left, det_top, det_right, det_bottom = _column_detector_preview_bounds(column)
                    frame_left = max(0, int(det_left) - preview_outset)
                    frame_top = max(0, int(det_top) - preview_outset)
                    frame_right = min(annotated.width - 1, int(det_right) - 1 + preview_outset)
                    frame_bottom = min(annotated.height - 1, int(det_bottom) - 1 + preview_outset)
                    column_frames.append((index, frame_left, frame_top, frame_right, frame_bottom))

                # Put every column number on one shared row above the highest
                # detector frame, so uneven column tops do not stagger labels.
                common_label_y = max(
                    0,
                    min(frame_top for _index, _left, frame_top, _right, _bottom in column_frames)
                    - 16,
                )
                for index, frame_left, frame_top, frame_right, frame_bottom in column_frames:
                    draw.rectangle(
                        (frame_left, frame_top, frame_right, frame_bottom),
                        outline=(34, 197, 94),
                        width=line_width,
                    )
                    label_text = str(index)
                    label_box = draw.textbbox((0, 0), label_text)
                    label_width = label_box[2] - label_box[0]
                    label_height = label_box[3] - label_box[1]
                    label_x = max(0, min(
                        annotated.width - label_width,
                        frame_left + (frame_right - frame_left - label_width) // 2,
                    ))
                    label_y = common_label_y
                    draw.rectangle(
                        (label_x - 2, label_y - 1,
                         label_x + label_width + 2, label_y + label_height + 1),
                        fill=(255, 255, 255),
                    )
                    draw.text((label_x, label_y), label_text, fill=(17, 130, 68))
                runtime_options = self._column_runtime_options_snapshot()
                preserve_body_pixels = bool(
                    runtime_options.get("column_preserve_body_pixels", False)
                )
                engine_key = str(getattr(self, "_active_adapter", "apple_vision") or "apple_vision").lower()
                viewport_mode = _recognizer_viewport_mode(engine_key)
                masked = _column_visibility_viewport(
                    source,
                    columns[0],
                    mode=viewport_mode,
                    preserve_body_pixels=preserve_body_pixels,
                )
                if engine_key == "ndlocr_lite":
                    centered = _center_wide_column_viewport(masked, columns[0].width)
                    mode_label = "NDLOCR 整列上下文"
                elif engine_key in {"apple_vision", "macocr", "mac_ocr", "macos_ocr"} or str(
                    runtime_options.get("column_isolation_mode", "mask") or "mask"
                ).strip().lower() == "display":
                    centered = _tighten_ink_framing(masked, columns[0].width)
                    mode_label = "单列原像素视窗"
                else:
                    centered = masked
                    mode_label = f"单列 {viewport_mode} 视窗"
                if centered is not masked:
                    masked.close()
                    masked = centered
                gap = max(12, round(source.width*0.02))
                preview = Image.new("RGB", (source.width*2+gap, source.height), "white")
                preview.paste(annotated,(0,0))
                right_x = source.width + gap + max(0, (source.width - masked.width) // 2)
                right_y = max(0, (source.height - masked.height) // 2)
                preview.paste(masked,(right_x,right_y))
                actual_input_size = masked.size
                preview.thumbnail((1800,1900), Image.Resampling.LANCZOS)
                data = io.BytesIO(); preview.save(data,format="PNG")
                preview.close(); masked.close(); annotated.close(); source.close()

            qimage = QImage.fromData(data.getvalue(), "PNG")
            if qimage.isNull():
                raise RuntimeError("分列预览图生成失败")
            dialog = QDialog(self)
            page_name = Path(str(source_path)).name
            dialog.setWindowTitle(f"分列输入预览 · {page_name} · {len(columns)} 列")
            dialog.resize(980, 760)
            layout = QVBoxLayout(dialog)
            sync_note = (
                "已复用当前实时预览页的同一组物理列；"
                if reused_live_columns else "当前页尚无实时分列结果，已按现有参数重新检测；"
            )
            info = QLabel(
                f"页面：{page_name}。{sync_note}"
                f"左侧绿色实线为检测框、蓝色框为 OCR 输入原像素范围，按日文阅读顺序从右到左编号，共 {len(columns)} 列；"
                f"右侧展示当前实际输入：{mode_label}，尺寸 {actual_input_size[0]}×{actual_input_size[1]}。"
                "蓝框之外仍可存在纯白安全槽上下文；目标正文保持原始像素，其他列与 Ruby 均被纸白色隔离。"
            )
            info.setWordWrap(True)
            info.setStyleSheet(f"color: {MUTED}; padding: 6px;")
            layout.addWidget(info)
            image_label = OCRCropPreview()
            image_label.setCursor(QCursor(Qt.ArrowCursor))
            image_label.setAttribute(Qt.WA_TransparentForMouseEvents)
            image_label.set_image_data(qimage)
            layout.addWidget(image_label, 1)
            close_btn = QPushButton("关闭")
            close_btn.clicked.connect(dialog.accept)
            layout.addWidget(close_btn)
            dialog.exec()
        except Exception as exc:
            show_error_dialog(self, "分列预览失败", str(exc))
