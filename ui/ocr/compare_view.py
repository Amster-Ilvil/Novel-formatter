from __future__ import annotations

from functools import partial

from PySide6.QtCore import Qt, QModelIndex, QTimer
from PySide6.QtGui import QColor, QKeySequence, QShortcut, QTextCursor, QTextCharFormat
from PySide6.QtWidgets import QAbstractItemView, QPlainTextEdit, QTextEdit

from models.document import UnifiedDocument
from ui.ocr.compare_widgets import _FusionDecisionRow
from ui.ocr.review_flow import candidate_for_key, step_pending


class OCRCompareViewMixin:
    def _set_advanced_actions_visible(self, visible: bool):
        self._advanced_actions_panel.setVisible(bool(visible))
        self._more_actions_btn.setText("收起操作 ▴" if visible else "更多操作 ▾")


    def _sync_compact_compare_controls(self) -> None:
        """Mirror backend control state into the Phase 20 compact command bar."""
        full = getattr(self, "_compact_full_compare_btn", None)
        if full is not None:
            backend = getattr(self, "_full_text_compare_check", None)
            enabled = bool(backend is not None and backend.isEnabled())
            full.setEnabled(enabled)
            if backend is not None and full.isChecked() != backend.isChecked():
                full.blockSignals(True)
                full.setChecked(backend.isChecked())
                full.blockSignals(False)
            full.setText("返回逐句" if bool(backend is not None and backend.isChecked()) else "显示全文")
        ai = getattr(self, "_compact_ai_btn", None)
        backend_ai = getattr(self, "_ai_adjudicate_btn", None)
        if ai is not None:
            ai.setEnabled(bool(backend_ai is not None and backend_ai.isEnabled()))
        status = getattr(self, "_compact_compare_state", None)
        if status is not None:
            detail = getattr(self, "_detail", None)
            summary = getattr(self, "_summary", None)
            value = ""
            if detail is not None and detail.text().strip() and detail.text().strip() != "当前句：—":
                value = detail.text().strip()
            elif summary is not None:
                value = summary.text().strip()
            status.setText(value or "等待 OCR 结果")


    def _multi_review_controls_available(self) -> bool:
        """Return whether a real multi-model comparison is currently active."""
        return bool(
            self._mode == "multi"
            and self._comparison is not None
            and len(self._documents) >= 2
        )


    def _sync_review_mode_controls(self) -> bool:
        """Keep every load/restore path in sync with the visible view switch."""
        available = self._multi_review_controls_available()
        self._decision_mode_btn.setEnabled(available)
        self._full_mode_btn.setEnabled(available)
        self._full_text_compare_check.setEnabled(available)
        if not available:
            self._full_text_compare_check.blockSignals(True)
            self._full_text_compare_check.setChecked(False)
            self._full_text_compare_check.blockSignals(False)
        self._sync_compact_compare_controls()
        return available


    def _toggle_full_text_compare(self, checked: bool):
        """Expose the existing full/one-sentence presentation through one switch."""
        # Restored multi-model sessions previously left the visible switch
        # disabled even though the comparison had already been restored.  Heal
        # the control state first and never let a stale disabled flag make the
        # visible button appear inert.
        if not self._sync_review_mode_controls():
            return
        self._set_review_mode("full" if bool(checked) else "decision")


    def _set_review_mode(self, mode: str, _checked=False, *, force: bool = False):
        """Switch view organization only; OCR documents and decisions stay shared."""
        mode = str(mode or "decision")
        if mode not in {"decision", "full"}:
            return
        if not force and mode == self._review_mode:
            return
        self._review_mode = mode
        decision_mode = mode == "decision"
        self._decision_mode_btn.setChecked(decision_mode)
        self._full_mode_btn.setChecked(not decision_mode)
        self._full_text_compare_check.blockSignals(True)
        self._full_text_compare_check.setChecked(not decision_mode)
        self._full_text_compare_check.blockSignals(False)
        self._decision_queue_panel.setVisible(decision_mode)
        self._auto_advance_check.setVisible(False)
        self._review_only_check.setVisible(False)
        # Phase 20: the default adjudication surface mirrors the redesign mockup:
        # queue on the left, candidate cards on the right.  The six raw OCR source
        # editors remain intact and reappear instantly in Full Text mode; nothing
        # is discarded or re-OCRed when the presentation changes.
        if hasattr(self, "_source_area") and self._mode == "multi":
            self._source_area.setVisible(not decision_mode)
        if hasattr(self, "_main_splitter"):
            self._main_splitter.setSizes([0, 820] if decision_mode else [500, 320])

        self._single_card_check.blockSignals(True)
        self._single_card_check.setChecked(decision_mode)
        self._single_card_check.blockSignals(False)
        self._review_only_check.blockSignals(True)
        self._review_only_check.setChecked(decision_mode)
        self._review_only_check.blockSignals(False)
        self._set_single_card_mode(decision_mode)
        self._set_review_only(decision_mode)

        if decision_mode:
            self._result_title.setText("候选结果")
            self._workspace_title.setText("OCR 对比 · 逐句裁决")
            self._refresh_decision_queue(force=True)
        else:
            self._result_title.setText("全文融合结果")
            self._workspace_title.setText("OCR 对比 · 全文对比")
        self._sync_compact_compare_controls()
        if self._comparison is not None and self._fusion_states:
            self._select_row(self._current_row())


    def _request_image_review(self):
        self.image_review_requested.emit(int(self._current_row()))


    @staticmethod
    def _queue_preview_text(state) -> str:
        candidates = list(getattr(state, "candidates", ()) or ())
        text = ""
        if candidates:
            preferred = getattr(state, "preferred_model_index", None)
            for candidate in candidates:
                if preferred is not None and preferred in tuple(getattr(candidate, "model_indices", ()) or ()):
                    text = str(getattr(candidate, "text", "") or "")
                    break
            if not text:
                text = str(getattr(candidates[0], "text", "") or "")
        text = " ".join(text.replace("\r", " ").replace("\n", " ").split())
        return text[:28] + ("…" if len(text) > 28 else "")


    def _resolved_history_rows(self) -> tuple[int, ...]:
        """Return explicitly adjudicated rows matching the active history filter."""
        from engine.ocr_compare_view_model import fusion_decision_origin_group
        wanted = str(getattr(self, "_resolved_history_group", "all") or "all")
        rows: list[int] = []
        for index, state in enumerate(self._fusion_states):
            if state.unresolved or state.selected_index is None:
                continue
            group = fusion_decision_origin_group(str(getattr(state, "selection_origin", "") or ""))
            if not group:
                continue
            if wanted == "all" or group == wanted:
                rows.append(index)
        return tuple(rows)


    def _decision_navigation_rows(self) -> tuple[int, ...]:
        if bool(getattr(self, "_show_resolved_history", False)):
            return self._resolved_history_rows()
        if (
            bool(getattr(getattr(self, "_active_review_check", None), "isChecked", lambda: False)())
            and self._comparison is not None
        ):
            try:
                from engine.ocr_review_priority import prioritized_unresolved_rows
                return prioritized_unresolved_rows(
                    self._comparison.rows, self._fusion_states
                )
            except Exception:
                pass
        return tuple(index for index, state in enumerate(self._fusion_states) if state.unresolved)


    def disagreement_queue_row_order(self) -> tuple[int, ...]:
        """Return the active pending-row order for the synchronized image queue."""
        if bool(getattr(self, "_show_resolved_history", False)):
            return ()
        return tuple(self._decision_navigation_rows())


    def _publish_disagreement_queue_order(self, rows=None) -> None:
        if bool(getattr(self, "_show_resolved_history", False)):
            return
        values = self.disagreement_queue_row_order() if rows is None else tuple(
            int(value) for value in rows
        )
        signal = getattr(self, "disagreement_queue_order_changed", None)
        if signal is not None:
            signal.emit(tuple(values))


    def _toggle_active_review_queue(self, checked: bool) -> None:
        self._decision_queue_signature = ()
        self._refresh_decision_queue(force=True)
        if bool(getattr(self, "_show_resolved_history", False)):
            return
        rows = self._decision_navigation_rows()
        if checked and rows and self._single_card_enabled and self._review_only_enabled:
            self._select_row(rows[0])
            self._summary.setText(
                "已启用高风险优先：只调整待判断顺序，不改变任何 OCR 候选或融合结果。"
            )


    @staticmethod
    def _decision_history_label(state) -> str:
        from engine.ocr_compare_view_model import fusion_decision_origin_label
        return fusion_decision_origin_label(str(getattr(state, "selection_origin", "") or ""))


    def _show_no_pending_decisions(self, message: str = "所有待判断句均已裁决") -> None:
        """Clear the one-card workspace so the last resolved sentence never lingers."""
        self._clear_fusion_widgets(dispose=False)
        self._fusion_window_indices = ()
        if self._single_card_enabled:
            self._single_card_preview_row = -1
            self._loading_text = True
            try:
                for editor in self._source_editors[:len(self._documents)]:
                    editor.setUpdatesEnabled(False)
                    try:
                        editor.clear()
                        editor.setReadOnly(True)
                    finally:
                        editor.setUpdatesEnabled(True)
            finally:
                self._loading_text = False
        self._row_state.setText("当前句：—")
        self._detail.setText(str(message or ""))
        self._detail.setToolTip(self._detail.text())
        self._virtual_hint.setText(
            f"待判断 0 句 · 全部 {len(self._fusion_states)} 句"
            if not self._show_resolved_history
            else str(message or "")
        )
        if hasattr(self, "_next_sentence_btn"):
            self._next_sentence_btn.setEnabled(False)


    def _sync_canonical_authority_from_states(self, row_indices=None) -> int:
        """Mirror explicit UI/fusion decisions into the single canonical store.

        Human, local Paddle, local visual-AI and imported cloud-AI selections all
        end here.  Automatic/provisional candidates are deliberately excluded.
        Modern sentence_group_id identity prevents one physical OCR column from
        fanning a decision out to several sentence groups.
        """
        if self._comparison is None or not self._fusion_states:
            return 0
        from engine.multi_ocr_source_correction import (
            canonical_decision_from_fusion_state, canonical_decision_key,
        )
        if row_indices is None:
            indices = range(min(len(self._comparison.rows), len(self._fusion_states)))
        else:
            indices = sorted({
                int(value) for value in row_indices
                if 0 <= int(value) < min(len(self._comparison.rows), len(self._fusion_states))
            })
        changed = 0
        for row_index in indices:
            row = self._comparison.rows[row_index]
            state = self._fusion_states[row_index]
            key = canonical_decision_key(row)
            if not key:
                continue
            decision = canonical_decision_from_fusion_state(row, state, self._labels)
            if decision is not None:
                existing = self._canonical_source_decisions.get(key)
                # Recovery/import may reconstruct the UI candidate from an
                # already richer canonical record.  If text+source are unchanged,
                # keep that record instead of discarding migration/raw-evidence
                # metadata.  Legacy AI overlays intentionally keep their original
                # source record even though the UI marker itself is ``ai_overlay``.
                state_origin = str(getattr(state, "selection_origin", "") or "")
                if (
                    isinstance(existing, dict)
                    and existing.get("status") == "accepted"
                    and str(existing.get("final_text", "") or "") == str(decision.get("final_text", "") or "")
                    and (
                        state_origin == "ai_overlay"
                        or str(existing.get("source", "") or "") == str(decision.get("source", "") or "")
                    )
                ):
                    continue
                if existing != decision:
                    self._canonical_source_decisions[key] = decision
                    changed += 1
            elif bool(getattr(state, "unresolved", False)):
                # Re-opening an explicitly adjudicated row removes its current
                # authority; the next export must therefore return it to pending.
                if key in self._canonical_source_decisions:
                    self._canonical_source_decisions.pop(key, None)
                    changed += 1
        return changed


    def _sync_after_bulk_resolution(self, preferred_row: int | None = None) -> None:
        """Refresh the adjudication workspace after a bulk/local/AI decision pass.

        Review-only mode must never leave a now-resolved sentence visible.  If
        unresolved rows remain, jump to the nearest next one; if none remain,
        clear the current candidate/source preview completely.  History mode is
        read-only and keeps browsing resolved rows instead.
        """
        self._sync_canonical_authority_from_states()
        self._refresh_decision_queue(force=True)
        self._update_unresolved_summary()
        if not self._fusion_states:
            self._show_no_pending_decisions("当前没有 OCR 对比句")
            return
        if self._show_resolved_history:
            rows = self._resolved_history_rows()
            if not rows:
                self._show_no_pending_decisions("当前筛选没有已裁决句")
                return
            current = self._current_row()
            self._select_row(current if current in rows else rows[0])
            return
        if not self._review_only_enabled:
            current = self._current_row()
            self._select_row(max(0, min(current, len(self._fusion_states) - 1)))
            return
        unresolved = tuple(self._decision_navigation_rows())
        if not unresolved:
            self._show_no_pending_decisions()
            return
        # Active-review mode intentionally owns the navigation sequence.  After
        # a bulk/local/AI pass we jump to the highest-risk unresolved sentence
        # rather than falling back to physical row order.  Without priority mode
        # the helper returns the historical document-order sequence.
        if bool(getattr(getattr(self, "_active_review_check", None), "isChecked", lambda: False)()):
            target = unresolved[0]
        else:
            try:
                start = int(self._current_row() if preferred_row is None else preferred_row)
            except (TypeError, ValueError, OverflowError):
                start = self._current_row()
            after = [value for value in unresolved if value >= start]
            target = after[0] if after else unresolved[0]
        self._select_row(target)


    def _toggle_resolved_history(self, checked: bool) -> None:
        self._show_resolved_history = bool(checked)
        self._resolved_history_filter.setEnabled(bool(checked))
        self._resolved_history_filter.setVisible(bool(checked))
        self._review_only_check.setEnabled(not bool(checked))
        for index, button in enumerate(self._choose_buttons):
            visible = index < len(self._documents) and button.isVisible()
            button.setEnabled((not bool(checked)) and visible)
        self._prev_group_btn.setText("← 上一已裁决" if checked else "← 上一分歧")
        self._next_group_btn.setText("下一已裁决 →" if checked else "下一分歧 →")
        self._refresh_decision_queue(force=True)
        if not self._fusion_states:
            return
        if checked:
            rows = self._resolved_history_rows()
            if not rows:
                self._show_no_pending_decisions("当前筛选没有已裁决句")
                self._update_unresolved_summary()
                return
            target = self._current_row() if self._current_row() in rows else rows[0]
            self._select_row(target)
            self._summary.setText(
                f"已进入裁决历史：显示 {len(rows)} 句；仅浏览，不会改动任何裁决结果。"
            )
        else:
            unresolved = tuple(self._decision_navigation_rows())
            if self._review_only_enabled:
                if unresolved:
                    self._select_row(unresolved[0])
                else:
                    self._show_no_pending_decisions()
            else:
                self._select_row(self._current_row())
        self._update_unresolved_summary()


    def _resolved_history_filter_changed(self, _index: int = -1) -> None:
        self._resolved_history_group = str(self._resolved_history_filter.currentData() or "all")
        if not self._show_resolved_history:
            return
        rows = self._resolved_history_rows()
        self._refresh_decision_queue(force=True)
        if rows:
            target = self._current_row() if self._current_row() in rows else rows[0]
            self._select_row(target)
        else:
            self._show_no_pending_decisions("当前筛选没有已裁决句")
        self._update_unresolved_summary()


    def _decision_queue_display_text(self, row_index: int) -> str:
        if not 0 <= int(row_index) < len(self._fusion_states):
            return ""
        row_index = int(row_index)
        state = self._fusion_states[row_index]
        history_mode = bool(getattr(self, "_show_resolved_history", False))
        active_priority = bool(
            not history_mode
            and getattr(self, "_active_review_check", None) is not None
            and self._active_review_check.isChecked()
        )
        badge = ""
        if history_mode:
            badge = self._decision_history_label(state) or "已裁决"
        elif active_priority and self._comparison is not None:
            try:
                from engine.ocr_review_priority import review_priority
                priority = review_priority(self._comparison.rows[row_index], state)
                badge = {"critical": "紧急", "high": "高风险", "medium": "待复核", "normal": "待裁决"}.get(priority.level, "待裁决")
            except Exception:
                badge = "待裁决"
        else:
            badge = "待裁决"

        page_label = f"第 {row_index + 1} 句"
        column_label = ""
        row = None
        if self._comparison is not None and row_index < len(self._comparison.rows):
            row = self._comparison.rows[row_index]
            page = int(getattr(row, "page", 0) or 0)
            if page > 0:
                page_label = f"p{page:03d}"
            column_ids = tuple(str(value) for value in (getattr(row, "column_ids", ()) or ()) if str(value))
            if len(column_ids) == 1:
                token = column_ids[0].replace("_", ":").split(":")[-1].lower()
                if token.startswith("c") and token[1:].isdigit():
                    column_label = f"列{int(token[1:])}"
                else:
                    column_label = token
            elif column_ids:
                column_label = f"{len(column_ids)} 列"

        candidates = list(getattr(state, "candidates", ()) or ())
        unique_texts = []
        for candidate in candidates:
            value = " ".join(str(getattr(candidate, "text", "") or "").replace("\r", " ").replace("\n", " ").split())
            if value and value not in unique_texts:
                unique_texts.append(value)
        preview = " / ".join(value[:12] + ("…" if len(value) > 12 else "") for value in unique_texts[:2])
        if not preview:
            preview = self._queue_preview_text(state) or "—"
        where = f"{page_label} · {column_label}" if column_label else page_label
        return f"{where}    {badge}\n{preview}  ·  {max(1, len(unique_texts))} 个候选"


    def _decision_queue_tooltip(self, row_index: int) -> str:
        if not 0 <= int(row_index) < len(self._fusion_states):
            return ""
        row_index = int(row_index)
        state = self._fusion_states[row_index]
        if bool(getattr(self, "_show_resolved_history", False)):
            suffix = self._decision_history_label(state) or "已裁决"
        elif (
            getattr(self, "_active_review_check", None) is not None
            and self._active_review_check.isChecked()
            and self._comparison is not None
        ):
            try:
                from engine.ocr_review_priority import review_priority
                priority = review_priority(self._comparison.rows[row_index], state)
                suffix = "风险优先 · " + ("、".join(priority.reasons) if priority.reasons else "普通分歧")
            except Exception:
                suffix = "点击定位"
        else:
            suffix = "点击定位"
        return f"第 {row_index + 1} 句 · {suffix}"


    def _refresh_decision_queue(self, *, force: bool = False):
        if not hasattr(self, "_decision_queue") or self._decision_queue_model is None:
            return
        history_mode = bool(getattr(self, "_show_resolved_history", False))
        rows = self._resolved_history_rows() if history_mode else self._decision_navigation_rows()
        active_priority = bool(
            not history_mode
            and getattr(self, "_active_review_check", None) is not None
            and self._active_review_check.isChecked()
        )
        mode_key = (
            "history" if history_mode else "pending",
            str(getattr(self, "_resolved_history_group", "all") or "all") if history_mode else "",
            "priority" if active_priority else "source_order",
        )
        self._decision_queue_count.setText(str(len(rows)))
        if hasattr(self, "_decision_queue_title"):
            title = "已裁决历史" if history_mode else "分歧队列"
            self._decision_queue_title.setText(f"{title} · {len(rows)}")
        # Qt's item model makes a full row reset cheap and lazy.  Single-row
        # adjudication normally never reaches this branch because the resolved
        # row is removed incrementally by _remove_decision_queue_rows().
        if force or mode_key != self._decision_queue_signature or self._decision_queue_model.rows != tuple(rows):
            self._syncing_decision_queue = True
            try:
                self._decision_queue_model.set_rows(rows)
            finally:
                self._syncing_decision_queue = False
            self._decision_queue_signature = mode_key
        self._select_current_queue_item()
        if not history_mode:
            self._publish_disagreement_queue_order(rows)


    def _remove_decision_queue_rows(self, row_indices) -> int:
        if self._decision_queue_model is None or self._show_resolved_history:
            return 0
        removed = self._decision_queue_model.remove_row_indices(row_indices)
        if removed:
            self._decision_queue_count.setText(str(self._decision_queue_model.rowCount()))
            self._publish_disagreement_queue_order(self._decision_queue_model.rows)
        return removed


    def _select_current_queue_item(self):
        if (
            not hasattr(self, "_decision_queue")
            or self._syncing_decision_queue
            or self._decision_queue_model is None
        ):
            return
        model_index = self._decision_queue_model.model_index_for_row(self._current_row())
        self._syncing_decision_queue = True
        try:
            selection = self._decision_queue.selectionModel()
            if selection is not None:
                selection.clearSelection()
            if model_index.isValid():
                self._decision_queue.setCurrentIndex(model_index)
                self._decision_queue.scrollTo(
                    model_index, QAbstractItemView.ScrollHint.PositionAtCenter
                )
        finally:
            self._syncing_decision_queue = False


    def _decision_queue_item_clicked(self, model_index):
        if self._syncing_decision_queue or model_index is None or self._decision_queue_model is None:
            return
        try:
            row_index = int(self._decision_queue_model.data(model_index, Qt.UserRole))
        except (TypeError, ValueError, OverflowError):
            return
        self._select_row(row_index)


    @staticmethod
    def _editor_lines(editor: QPlainTextEdit) -> list[str]:
        return editor.toPlainText().replace("\r\n", "\n").replace("\r", "\n").split("\n")


    @staticmethod
    def _line_text(editor: QPlainTextEdit, row: int) -> str:
        block = editor.document().findBlockByNumber(row)
        return block.text() if block.isValid() else ""


    @staticmethod
    def _split_source_lines(text: str) -> list[str]:
        return str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")


    def _invalidate_highlight_cache(self) -> None:
        self._highlight_source_revision += 1
        self._highlight_mask_cache.clear()
        self._last_highlight_signature = None


    def _set_full_source_texts(self, texts) -> None:
        values = [str(value or "") for value in (texts or [])]
        self._full_source_texts = values
        self._full_source_lines = [self._split_source_lines(value) for value in values]
        self._invalidate_highlight_cache()


    def _set_full_source_text(self, model_index: int, text: str) -> None:
        index = max(0, int(model_index))
        while len(self._full_source_texts) <= index:
            self._full_source_texts.append("")
            self._full_source_lines.append([""])
        value = str(text or "")
        if self._full_source_texts[index] == value:
            return
        self._full_source_texts[index] = value
        self._full_source_lines[index] = self._split_source_lines(value)
        self._invalidate_highlight_cache()


    def _sync_full_source_cache_from_editors(self) -> None:
        """Persist full editor text before entering the one-card preview mode."""
        if self._single_card_enabled or self._loading_text:
            return
        count = len(self._documents)
        self._set_full_source_texts([
            self._source_editors[index].toPlainText()
            for index in range(min(count, len(self._source_editors)))
        ])


    def _source_text(self, model_index: int) -> str:
        if 0 <= model_index < len(self._full_source_texts):
            return str(self._full_source_texts[model_index] or "")
        if 0 <= model_index < len(self._source_editors):
            return self._source_editors[model_index].toPlainText()
        return ""


    def _source_line_text(self, model_index: int, row: int) -> str:
        if 0 <= model_index < len(self._full_source_lines):
            lines = self._full_source_lines[model_index]
        else:
            lines = self._split_source_lines(self._source_text(model_index))
        return lines[row] if 0 <= row < len(lines) else ""


    def _source_line_count(self, model_index: int) -> int:
        if 0 <= model_index < len(self._full_source_lines):
            return len(self._full_source_lines[model_index])
        return len(self._split_source_lines(self._source_text(model_index)))


    def _set_source_labels_for_mode(self) -> None:
        for index in range(min(len(self._documents), len(self._source_labels))):
            label = self._labels[index] if index < len(self._labels) else f"模型{index + 1}"
            suffix = "（结构底稿）" if index == 0 else "（文字候选）"
            if self._single_card_enabled:
                suffix = "（当前句只读）"
            self._source_labels[index].setText(f"模型{index + 1} · {label} {suffix}")


    def _update_single_card_source_preview(self, row: int, *, force: bool = False) -> None:
        if not self._single_card_enabled or self._comparison is None:
            return
        row = max(0, min(int(row), max(0, len(self._comparison.rows) - 1)))
        if not force and getattr(self, "_single_card_preview_row", -1) == row:
            return
        self._single_card_preview_row = row
        self._loading_text = True
        try:
            for index, editor in enumerate(self._source_editors[:len(self._documents)]):
                editor.setUpdatesEnabled(False)
                try:
                    editor.setReadOnly(True)
                    editor.setPlainText(self._source_line_text(index, row))
                    editor.moveCursor(QTextCursor.Start)
                finally:
                    editor.setUpdatesEnabled(True)
        finally:
            self._loading_text = False
        self._set_source_labels_for_mode()


    def _restore_full_source_editors(self) -> None:
        self._single_card_preview_row = -1
        self._loading_text = True
        try:
            for index, editor in enumerate(self._source_editors[:len(self._documents)]):
                editor.setUpdatesEnabled(False)
                try:
                    editor.setReadOnly(False)
                    editor.setPlainText(self._source_text(index))
                    block = editor.document().findBlockByNumber(self._current_row())
                    if block.isValid():
                        editor.setTextCursor(QTextCursor(block))
                finally:
                    editor.setUpdatesEnabled(True)
        finally:
            self._loading_text = False
        self._set_source_labels_for_mode()


    @staticmethod
    def _single_document_preview_text(doc: UnifiedDocument) -> str:
        lines = []
        for block in doc.text_blocks():
            if isinstance(block.metadata, dict) and block.metadata.get("consumed"):
                continue
            lines.append(str(block.text or ""))
        return "\n".join(lines)


    def _clear_fusion_widgets(self, *, dispose: bool = True):
        mounted = getattr(self, "_mounted_reference_row", None)
        if mounted is not None:
            mounted._reference_preview.hide()
            self._proof_reference_layout.removeWidget(mounted._reference_preview)
            mounted._reference_preview.setParent(mounted)
            mounted._body_layout.insertWidget(0, mounted._reference_preview)
            self._mounted_reference_row = None
        active = dict(self._fusion_widgets)
        while self._fusion_layout.count() > 1:
            item = self._fusion_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
        self._fusion_widgets = {}
        self._fusion_window_indices = ()
        if dispose:
            for widget in active.values():
                widget.deleteLater()
            for widget in self._fusion_widget_cache.values():
                widget.deleteLater()
            self._fusion_widget_cache.clear()
            return
        for row_index, widget in active.items():
            self._fusion_widget_cache[row_index] = widget
            self._fusion_widget_cache.move_to_end(row_index)
        while len(self._fusion_widget_cache) > self._fusion_widget_cache_limit:
            _old_row, old_widget = self._fusion_widget_cache.popitem(last=False)
            old_widget.deleteLater()


    def _obtain_fusion_widget(self, row_index: int, state) -> _FusionDecisionRow:
        widget = self._fusion_widget_cache.pop(int(row_index), None)
        if widget is not None and not widget.is_compatible_with(state, self._labels):
            # Never reuse a row whose check box is bound to an old decision
            # object or an older candidate list.  That stale-widget bug could
            # show “✓ 选择” while export still saw the row as unresolved.
            widget.hide()
            widget.deleteLater()
            widget = None
        if widget is None:
            widget = _FusionDecisionRow(
                row_index,
                [candidate.text for candidate in state.candidates],
                self._labels,
                self._fusion_content,
                preferred_model_index=state.preferred_model_index,
                decision_state=state,
            )
            widget.focused.connect(lambda index, self=self: self._select_row(index))
            if hasattr(self, "_history_service"):
                widget.about_to_resolve.connect(self._history_service.prepare_resolution)
                widget.about_to_reopen.connect(self._history_service.prepare_reopen)
            # Publish selected-candidate edits without rebuilding the editor
            # under the user's cursor or advancing to another sentence.
            widget.manual_text_changed.connect(self._publish_fusion_decision)
            widget.resolved.connect(self._fusion_row_resolved)
            widget.reopened.connect(self._fusion_row_reopened)
            if hasattr(self, "_history_service"):
                widget.resolved.connect(self._history_service.commit_resolution)
                widget.reopened.connect(self._history_service.commit_reopen)
        else:
            widget.sync_from_state()
        if row_index == int(getattr(self, "_current_row_index", -1)):
            viewport = self._fusion_scroll.viewport()
            widget.set_reference_target_height(max(380, viewport.height() - 26))
        else:
            widget.setMinimumHeight(0)
        if self._comparison is not None and 0 <= row_index < len(self._comparison.rows):
            provider = getattr(self, "_recovery_page_image_provider", None)
            try:
                page_images = tuple(provider() or ()) if callable(provider) else ()
            except Exception:
                page_images = ()
            widget.set_reference_source(self._primary_doc, self._comparison.rows[row_index], page_images)
        widget.set_review_only_candidates(self._review_only_enabled)
        widget.set_history_mode(bool(getattr(self, "_show_resolved_history", False)))
        locked = bool(self._comparison is not None and len(self._documents) >= 2
                      and not self._comparison.rows[row_index].is_conflict)
        widget.set_consensus_locked(locked)
        host = getattr(self, "_proof_reference_host", None)
        if host is not None:
            host.setVisible(bool(self._single_card_enabled))
            if self._single_card_enabled:
                widget._body_layout.removeWidget(widget._reference_preview)
                self._proof_reference_layout.addWidget(widget._reference_preview, 1)
                widget._reference_preview.show()
                self._mounted_reference_row = widget
            else:
                # Full-text mode is for comparing complete OCR documents; its
                # per-sentence source crop would only consume horizontal space.
                widget._reference_preview.hide()
        widget.show()
        return widget


    def _rebuild_fusion_rows(self, *, auto_choose: bool):
        from engine.ocr_compare_view_model import build_fusion_states
        if hasattr(self, "_history_service"):
            self._history_service.clear()
        self._clear_fusion_widgets()
        if self._comparison is None:
            self._fusion_states = []
            return
        # The comparison rows already contain exactly the text loaded into the
        # source editors.  Reading every QTextBlock back here would add another
        # O(rows × models) GUI-thread pass for no benefit.
        self._fusion_states = build_fusion_states(
            self._comparison.rows,
            auto_choose=auto_choose,
        )
        self._reapply_image_review_overrides()
        self._render_fusion_window(self._current_row_index, force=True)
        self._refresh_decision_queue(force=True)
        self._update_unresolved_summary()


    def _render_fusion_window(self, current_row: int, *, force: bool = False):
        if self._comparison is None or not self._fusion_states:
            self._clear_fusion_widgets()
            self._virtual_hint.setText("")
            return
        from engine.ocr_compare_view_model import windowed_row_indices
        history_mode = bool(getattr(self, "_show_resolved_history", False))
        if history_mode:
            eligible_rows = list(self._resolved_history_rows())
            if not eligible_rows:
                self._clear_fusion_widgets(dispose=False)
                self._fusion_window_indices = ()
                self._virtual_hint.setText("当前筛选没有已裁决句")
                return
            current_row = int(current_row)
            if current_row not in eligible_rows:
                current_row = eligible_rows[0]
            if self._single_card_enabled:
                indices = (current_row,)
            else:
                position = eligible_rows.index(current_row)
                radius = max(1, self._fusion_window_size // 2)
                start_at = max(0, position - radius)
                end_at = min(len(eligible_rows), start_at + self._fusion_window_size)
                start_at = max(0, end_at - self._fusion_window_size)
                indices = tuple(eligible_rows[start_at:end_at])
            eligible = len(eligible_rows)
        elif self._single_card_enabled:
            current_row = max(0, min(int(current_row), len(self._fusion_states) - 1))
            indices = (current_row,)
            eligible = sum(1 for state in self._fusion_states if (state.unresolved if self._review_only_enabled else True))
        else:
            indices = tuple(windowed_row_indices(
                self._fusion_states,
                current_row,
                review_only=self._review_only_enabled,
                window_size=self._fusion_window_size,
            ))
            eligible = sum(1 for state in self._fusion_states if (state.unresolved if self._review_only_enabled else True))
        if not force and indices == self._fusion_window_indices:
            viewport_height = self._fusion_scroll.viewport().height() - 26
            for visible_index, visible_widget in self._fusion_widgets.items():
                visible_widget.set_reference_target_height(
                    viewport_height if visible_index == int(self._current_row_index) else 0
                )
            return
        self._clear_fusion_widgets(dispose=False)
        self._fusion_window_indices = indices
        for row_index in indices:
            state = self._fusion_states[row_index]
            widget = self._obtain_fusion_widget(row_index, state)
            self._fusion_layout.insertWidget(self._fusion_layout.count() - 1, widget, 1 if len(indices) == 1 else 0)
            self._fusion_widgets[row_index] = widget
        self._fusion_layout.setStretch(self._fusion_layout.count() - 1, 0 if len(indices) == 1 else 1)
        total = len(self._fusion_states)
        if history_mode:
            self._virtual_hint.setText(
                f"裁决历史 {eligible} 句 · 当前 {indices[0] + 1}/{total}" if indices else "当前筛选没有已裁决句"
            )
        elif indices and self._single_card_enabled:
            self._virtual_hint.setText(f"单框逐句 · 当前 {indices[0] + 1}/{total} 句")
        elif indices:
            self._virtual_hint.setText(f"按需显示 {len(indices)}/{eligible} 句 · 全部 {total} 句")
        elif self._review_only_enabled:
            self._virtual_hint.setText(f"待判断 0 句 · 全部 {total} 句")
        else:
            self._virtual_hint.setText(f"全部 {total} 句")


    def clear(self, *, preserve_available_single: bool = True):
        if hasattr(self, "_batch_service"):
            self._batch_service.invalidate_restore()
        if hasattr(self, "_history_service"):
            self._history_service.clear()
        self._result_load_generation += 1
        self._ai_import_generation.invalidate()
        self._ai_import_signal_refs.clear()
        self._pending_ai_import_apply = None
        self._ai_import_busy = False
        # Source-correction export/import/restore workers may still be finishing
        # in background.  A clear, book switch, or mode replacement makes every
        # prior callback stale so old-book decisions can never land in the new
        # OCR workspace.  The worker may finish naturally; GenerationGuard drops it.
        self._source_correction_generation.invalidate()
        self._source_correction_signal_refs.clear()
        self._source_correction_busy = False
        self._source_correction_lock_workspace = False
        if hasattr(self, "_source_correction_progress"):
            self._source_correction_progress.setVisible(False)
        if hasattr(self, "_restore_source_session_btn"):
            self._restore_source_session_btn.setEnabled(True)
        self._full_source_texts = []
        self._full_source_lines = []
        self._single_card_preview_row = -1
        if hasattr(self, "_source_area"):
            self._source_area.setEnabled(True)
        if hasattr(self, "_result_panel"):
            self._result_panel.setEnabled(True)
        self._mode = "empty"
        self._single_doc = None
        self._single_original_doc = None
        self._loaded_single_generation = -1
        self._documents = []
        self._labels = []
        self._comparison = None
        self._primary_doc = None
        self._initial_payload = None
        # A true clear/new-book boundary must never retain Ruby annotations
        # from the previous multi-model session. Temporary multi -> single
        # switching saves/restores this overlay explicitly in _capture_multi_state.
        self._ruby_overlay_doc = None
        self._fusion_states = []
        self._show_resolved_history = False
        self._resolved_history_group = "all"
        if hasattr(self, "_resolved_history_check"):
            self._resolved_history_check.blockSignals(True)
            self._resolved_history_check.setChecked(False)
            self._resolved_history_check.blockSignals(False)
        if hasattr(self, "_resolved_history_filter"):
            self._resolved_history_filter.blockSignals(True)
            self._resolved_history_filter.setCurrentIndex(0)
            self._resolved_history_filter.setVisible(False)
            self._resolved_history_filter.setEnabled(False)
            self._resolved_history_filter.blockSignals(False)
        if hasattr(self, "_review_only_check"):
            self._review_only_check.setEnabled(True)
        if hasattr(self, "_prev_group_btn"):
            self._prev_group_btn.setText("← 上一分歧")
        if hasattr(self, "_next_group_btn"):
            self._next_group_btn.setText("下一分歧 →")
        self._image_review_overrides = {}
        self._canonical_source_decisions = {}
        self._source_correction_import_history = []
        self._source_correction_imported_package_ids = set()
        self._targeted_retry_local_report = None
        self._loading_text = True
        try:
            for index, editor in enumerate(self._source_editors):
                editor.setUpdatesEnabled(False)
                try:
                    editor.clear()
                    editor.setExtraSelections([])
                finally:
                    editor.setUpdatesEnabled(True)
                self._source_labels[index].setText(f"模型{index + 1} · 未载入")
                self._source_editors[index].setReadOnly(False)
                self._source_panels[index].setVisible(False)
                self._choose_buttons[index].setVisible(False)
                self._choose_buttons[index].setEnabled(False)
        finally:
            self._loading_text = False
        self._clear_fusion_widgets()
        self._decision_queue_signature = ()
        if hasattr(self, "_decision_queue"):
            # QListView has no clear(); clear its model and selection instead.
            # This is also O(1)-ish model reset rather than rebuilding thousands
            # of QListWidget items from the pre-virtualized implementation.
            if self._decision_queue_model is not None:
                self._decision_queue_model.set_rows(())
            selection = self._decision_queue.selectionModel()
            if selection is not None:
                selection.clearSelection()
                selection.clearCurrentIndex()
            self._decision_queue.setCurrentIndex(QModelIndex())
            self._decision_queue_count.setText("0")
        self._virtual_hint.setText("")
        self._source_area.setVisible(False)
        self._workspace_title.setText("OCR 对比 · 结果收件箱")
        self._summary.setText(
            "单模型 OCR 完成后请手动点击“从 OCR 识别载入”；多模型 OCR 完成后会自动载入。"
        )
        self._choose_label.setVisible(True)
        self._row_state.setVisible(True)
        self._result_panel.setVisible(True)
        self._result_title.setText("逐句裁决 · 当前候选")
        self._apply_btn.setText("✓ 应用融合稿")
        self._row_state.setText("当前句：—")
        self._detail.setText("当前句：—")
        if hasattr(self, "_source_correction_state"):
            self._source_correction_state.setText("尚未导入逐源纠错")
        self._auto_btn.setEnabled(False)
        self._sync_review_mode_controls()
        self._realign_btn.setEnabled(False)
        self._restore_btn.setEnabled(False)
        self._unicode_normalize_btn.setEnabled(False)
        self._export_texts_btn.setEnabled(False)
        self._single_export_btn.setEnabled(False)
        self._export_ai_package_btn.setEnabled(False)
        self._export_source_correction_btn.setEnabled(False)
        self._import_source_correction_btn.setEnabled(False)
        self._export_fusion_skeleton_btn.setEnabled(False)
        self._export_ai_repair_epub_btn.setEnabled(False)
        self._import_ai_repair_result_btn.setEnabled(False)
        self._ai_adjudicate_btn.setEnabled(False)
        self._import_ai_package_btn.setEnabled(False)
        self._apply_btn.setEnabled(False)
        if hasattr(self, "_next_sentence_btn"):
            self._next_sentence_btn.setEnabled(False)
        for button in self._choose_buttons:
            button.setEnabled(False)
        if not preserve_available_single:
            self._available_single_doc = None
            self._available_single_label = ""
            self._available_single_generation = 0
        self._refresh_single_source_controls()


    def reset_for_new_book(self):
        self._suspended_multi_state = None
        self._reference_path = ""
        self._last_source_correction_report = None
        self._source_correction_export_report = None
        self._canonical_source_decisions = {}
        self._source_correction_import_history = []
        self._source_correction_imported_package_ids = set()
        self.clear(preserve_available_single=False)


    def shutdown_cleanup(self) -> None:
        self._result_load_generation += 1
        self._ai_import_generation.invalidate()
        self._ai_import_signal_refs.clear()
        self._source_correction_generation.invalidate()
        self._source_correction_signal_refs.clear()
        self._source_correction_busy = False
        self._source_correction_lock_workspace = False


    def _current_row(self) -> int:
        return max(0, self._current_row_index)


    def install_keyboard_flow(self) -> None:
        """合并编辑器式键盘流：Alt+1..9 采用候选，Alt+↑/↓ 上/下一个待判断句。

        只在本页获得焦点时生效；采用后自动跳到下一个待判断句（可连按数字键快速过稿）。
        """
        self._keyboard_flow_shortcuts = []
        def bind(seq, slot):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(slot)
            self._keyboard_flow_shortcuts.append(sc)
        for n in range(1, 10):
            bind(f"Alt+{n}", partial(self._choose_candidate_by_number, n))
        bind("Alt+Down", partial(self._goto_pending_row, True))
        bind("Alt+Up", partial(self._goto_pending_row, False))

    def _choose_candidate_by_number(self, number: int) -> None:
        widget = self._fusion_widgets.get(self._current_row())
        if widget is None or getattr(self, "_show_resolved_history", False):
            return
        index = candidate_for_key(number, len(widget._cards))
        if index is None:
            return
        widget.choose(index)
        QTimer.singleShot(140, partial(self._goto_pending_row, True))   # 采用后自动到下一个

    def _goto_pending_row(self, forward: bool = True) -> None:
        model = getattr(self, "_decision_queue_model", None)
        rows = model.rows if model is not None else ()
        target = step_pending(rows, self._current_row(), forward)
        if target is None:
            if hasattr(self, "_summary"):
                self._summary.setText("已经是最后一个待判断句" if forward else "已经是第一个待判断句")
            return
        self._select_row(target)

    def current_row_index(self) -> int:
        """Return the stable multi-model comparison row currently in view."""
        return self._current_row()


    def select_source_row(self, row_index: int) -> None:
        """Select one stable comparison row from another proofreading view."""
        if self._comparison is None or not self._comparison.rows:
            return
        try:
            target = int(row_index)
        except (TypeError, ValueError, OverflowError):
            return
        target = max(0, min(target, len(self._comparison.rows) - 1))
        if target == self._current_row_index:
            return
        self._select_row(target)


    def _source_cursor_changed(self, source_index: int):
        if self._single_card_enabled:
            return
        if self._loading_text or self._syncing_cursor or self._comparison is None:
            return
        row = self._source_editors[source_index].textCursor().blockNumber()
        self._select_row(row, source_index=source_index)


    def _source_text_changed(self):
        if self._loading_text or self._comparison is None or self._single_card_enabled:
            return
        sender = self.sender()
        if sender in self._source_editors:
            index = self._source_editors.index(sender)
            self._set_full_source_text(index, sender.toPlainText())
        self._sources_dirty = True
        self._highlight_timer.start()
        self._summary.setText(
            "源 OCR 已手动修改。字符红绿会自动刷新；若增删了换行，点击“重新自动对齐”后再裁决或应用。"
        )


    def _select_row(self, row: int, source_index: int | None = None):
        if self._comparison is None or not self._comparison.rows:
            return
        row = max(0, min(row, len(self._comparison.rows) - 1))
        if self._show_resolved_history:
            history_rows = self._resolved_history_rows()
            if not history_rows:
                self._show_no_pending_decisions("当前筛选没有已裁决句")
                return
            if row not in history_rows:
                row = history_rows[0]
        timer = getattr(self, "_manual_decision_commit_timer", None)
        if timer is not None and timer.isActive():
            timer.stop()
            if hasattr(self, "_commit_manual_decision_editor"):
                self._commit_manual_decision_editor()
        self._current_row_index = row
        self._active_source_index = source_index
        self._syncing_cursor = True
        self._syncing_scroll = True
        try:
            if self._single_card_enabled:
                self._update_single_card_source_preview(row)
            else:
                editors = self._source_editors[:len(self._documents)]
                if editors:
                    anchor_index = source_index if source_index is not None and 0 <= source_index < len(editors) else 0
                    anchor = editors[anchor_index]
                    block = anchor.document().findBlockByNumber(row)
                    if block.isValid() and anchor.textCursor().blockNumber() != row:
                        anchor.setTextCursor(QTextCursor(block))
                    anchor.ensureCursorVisible()
                    scroll_value = anchor.verticalScrollBar().value()
                    # Keep the three books visually aligned without forcing Qt to
                    # reposition three independent cursors and repaint three full
                    # documents for every navigation click.
                    for index, editor in enumerate(editors):
                        if index != anchor_index and editor.verticalScrollBar().value() != scroll_value:
                            editor.verticalScrollBar().setValue(scroll_value)
        finally:
            self._syncing_scroll = False
            self._syncing_cursor = False

        self._render_fusion_window(row)
        for index, widget in self._fusion_widgets.items():
            widget.set_current(index == row)
        widget = self._fusion_widgets.get(row)
        if widget is not None:
            self._fusion_scroll.ensureWidgetVisible(widget, 0, 70)

        # Character-level painting is intentionally coalesced. Fast repeated
        # next/previous operations now perform one visible-window diff pass.
        self._highlight_timer.start()
        info = self._comparison.rows[row]
        if bool(getattr(info, "is_conflict", False)):
            conflict = "真正分歧"
        elif bool(getattr(info, "provisional_consensus", False)):
            conflict = "两模型共同候选"
        else:
            conflict = "真正一致"
        unresolved = row < len(self._fusion_states) and self._fusion_states[row].unresolved
        state = "待选择" if unresolved else "已确定"
        self._row_state.setText(
            f"当前句：{row + 1}/{len(self._comparison.rows)} · {conflict} · {state}"
        )
        warning = f"；提示：{'、'.join(info.warnings)}" if info.warnings else ""
        self._detail.setText(f"自动分析：{info.reason}{warning}")
        self._detail.setToolTip(self._detail.text())
        if hasattr(self, "_next_sentence_btn"):
            self._next_sentence_btn.setEnabled(row + 1 < len(self._comparison.rows))
        self._select_current_queue_item()
        if hasattr(self, "_sync_manual_decision_editor"):
            self._sync_manual_decision_editor()
        self.current_row_changed.emit(int(row))


    def _visible_source_rows(self, editors: list[QPlainTextEdit]) -> list[int]:
        if self._comparison is None:
            return []
        row_count = len(self._comparison.rows)
        rows: set[int] = set()
        for editor in editors:
            try:
                first = max(0, editor.firstVisibleBlock().blockNumber())
                line_height = max(12, editor.fontMetrics().height())
                visible_count = max(18, editor.viewport().height() // line_height + 8)
            except Exception:
                first = max(0, self._current_row_index - 24)
                visible_count = 48
            rows.update(range(first, min(row_count, first + visible_count)))
        rows.update(range(
            max(0, self._current_row_index - 4),
            min(row_count, self._current_row_index + 5),
        ))
        return sorted(rows)


    def _refresh_source_highlights(self):
        if not self._workspace_active or self._comparison is None or self._loading_text:
            return
        from engine.multi_ocr_compare import intraline_match_masks

        editors = self._source_editors[:len(self._documents)]
        if not editors:
            return
        if self._single_card_enabled:
            visible_rows = (self._current_row(),)
        else:
            row_count = min(
                [len(self._comparison.rows)]
                + [editor.document().blockCount() for editor in editors]
            )
            visible_rows = tuple(row for row in self._visible_source_rows(editors) if row < row_count)
        signature = (
            bool(self._single_card_enabled), int(self._current_row_index),
            int(self._highlight_source_revision), visible_rows,
        )
        if signature == self._last_highlight_signature:
            return
        selections_by_editor: list[list[QTextEdit.ExtraSelection]] = [[] for _ in editors]

        for source_row in visible_rows:
            texts = tuple(self._source_line_text(index, source_row) for index in range(len(editors)))
            cached = self._highlight_mask_cache.get(source_row)
            if cached is not None and cached[0] == texts:
                masks = cached[1]
            else:
                masks = tuple(tuple(bool(value) for value in mask) for mask in intraline_match_masks(list(texts)))
                self._highlight_mask_cache[source_row] = (texts, masks)
            display_row = 0 if self._single_card_enabled else source_row
            for editor_index, (editor, text, mask) in enumerate(zip(editors, texts, masks)):
                block = editor.document().findBlockByNumber(display_row)
                if not block.isValid() or not text:
                    continue
                block_text = block.text()
                # The single-card preview replaces editor contents while row
                # navigation is in progress. Never apply cached offsets to the
                # previous row (or to text that was edited independently).
                if block_text != text:
                    for target_editor in editors:
                        target_editor.setExtraSelections([])
                    self._last_highlight_signature = None
                    return
                max_block_offset = max(0, block.length() - 1)
                document_max_position = max(0, editor.document().characterCount() - 1)
                start_pos = 0
                while start_pos < len(text):
                    same = bool(mask[start_pos]) if start_pos < len(mask) else False
                    end_pos = start_pos + 1
                    while end_pos < len(text) and (
                        bool(mask[end_pos]) if end_pos < len(mask) else False
                    ) == same:
                        end_pos += 1
                    selection = QTextEdit.ExtraSelection()
                    cursor = QTextCursor(editor.document())
                    start_offset = len(text[:start_pos].encode("utf-16-le")) // 2
                    end_offset = len(text[:end_pos].encode("utf-16-le")) // 2
                    start_position = min(
                        document_max_position, block.position() + start_offset,
                        block.position() + max_block_offset,
                    )
                    end_position = min(
                        document_max_position, block.position() + end_offset,
                        block.position() + max_block_offset,
                    )
                    if end_position <= start_position:
                        start_pos = end_pos
                        continue
                    cursor.setPosition(start_position)
                    cursor.setPosition(end_position, QTextCursor.KeepAnchor)
                    selection.cursor = cursor
                    fmt = QTextCharFormat()
                    fmt.setBackground(QColor("#BDEFC9" if same else "#FFC7C7"))
                    fmt.setForeground(QColor("#14202E"))
                    selection.format = fmt
                    selections_by_editor[editor_index].append(selection)
                    start_pos = end_pos

        if not self._single_card_enabled:
            row_count = min([len(self._comparison.rows)] + [editor.document().blockCount() for editor in editors])
            if 0 <= self._current_row_index < row_count:
                for editor_index, editor in enumerate(editors):
                    if self._active_source_index is not None and editor_index == self._active_source_index:
                        continue
                    block = editor.document().findBlockByNumber(self._current_row_index)
                    if not block.isValid():
                        continue
                    selection = QTextEdit.ExtraSelection()
                    cursor = QTextCursor(block)
                    cursor.select(QTextCursor.BlockUnderCursor)
                    selection.cursor = cursor
                    fmt = QTextCharFormat()
                    fmt.setFontUnderline(True)
                    fmt.setUnderlineColor(QColor("#2F6BFF"))
                    selection.format = fmt
                    selections_by_editor[editor_index].append(selection)

        for editor, selections in zip(editors, selections_by_editor):
            editor.setExtraSelections(selections)
        self._last_highlight_signature = signature


    def _source_scroll_changed(self, source_index: int, value: int):
        if self._single_card_enabled:
            return
        if self._loading_text or self._syncing_scroll:
            return
        self._syncing_scroll = True
        try:
            for index, editor in enumerate(self._source_editors[:len(self._documents)]):
                if index != source_index:
                    editor.verticalScrollBar().setValue(value)
        finally:
            self._syncing_scroll = False
        if self._workspace_active:
            self._highlight_timer.start()


    def set_workspace_active(self, active: bool) -> None:
        active = bool(active)
        if active == self._workspace_active:
            return
        self._workspace_active = active
        if not active:
            self._highlight_timer.stop()
            return
        self._last_highlight_signature = None
        if self._comparison is not None:
            self._highlight_timer.start()
