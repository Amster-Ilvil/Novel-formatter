from __future__ import annotations

import copy
import threading
from pathlib import Path

from adapters.result_export import safe_result_filename
from ui.common.signals import WorkerSignals
from ui.common.toast import notify
from ui.dialogs import show_error_dialog
from ui.localized_dialogs import LocalizedFileDialog as QFileDialog, LocalizedMessageBox as QMessageBox


class OCRCompareSourceCorrectionService:
    """Own source-correction package/session orchestration for OCRCompareTab.

    Raw OCR documents remain owned by the tab and immutable; this service only
    orchestrates recovery, canonical-decision overlays and background exchange.
    """

    def __init__(self, tab):
        self._tab = tab

    def _current_comparison_documents_for_source_correction(self):
        """Compatibility helper; heavy synchronisation now runs in a worker."""
        self = self._tab
        if self._comparison is None or len(self._documents) < 2:
            raise ValueError("请先完成并载入多模型 OCR。")
        comparison = self._comparison_with_current_source_texts()
        if comparison is None:
            raise ValueError("当前 OCR 行数与对齐不一致，请先点击“重新对齐”。")
        from engine.multi_ocr_source_correction import documents_with_comparison_texts
        documents = documents_with_comparison_texts(self._documents, comparison)
        return documents, comparison

    def _set_source_correction_busy(self, busy: bool, message: str = "", *, lock_workspace: bool = False):
        self = self._tab
        self._source_correction_busy = bool(busy)
        self._source_correction_lock_workspace = bool(busy and lock_workspace)
        multi_ready = self._mode == "multi" and self._comparison is not None and len(self._documents) >= 2
        enabled = multi_ready and not self._source_correction_busy and not self._ai_import_busy
        for button in (
            self._auto_btn, self._realign_btn, self._unicode_normalize_btn,
            self._export_texts_btn, self._export_ai_package_btn,
            self._export_source_correction_btn, self._import_source_correction_btn,
            self._export_fusion_skeleton_btn, self._export_ai_repair_epub_btn,
            self._import_ai_repair_result_btn,
            self._ai_adjudicate_btn, self._import_ai_package_btn, self._apply_btn,
        ):
            button.setEnabled(enabled)
        self._restore_btn.setEnabled(enabled and self._initial_payload is not None)
        self._restore_source_session_btn.setEnabled(not busy and not self._ai_import_busy)
        self._source_correction_progress.setVisible(bool(busy))
        if busy:
            self._source_correction_progress.setValue(0)
            self._source_correction_progress.setFormat("准备 OCR 裁决任务…")
        workspace_enabled = not self._source_correction_lock_workspace and not self._ai_import_busy
        self._source_area.setEnabled(workspace_enabled)
        self._result_panel.setEnabled(workspace_enabled)
        if message:
            self._summary.setText(message)

    def _on_source_correction_progress(self, token: int, stage: str, current: int, total: int):
        self = self._tab
        if not self._source_correction_generation.is_current(token):
            return
        total = max(1, int(total or 1))
        current = max(0, min(int(current or 0), total))
        value = min(99, int(current * 100 / total))
        self._source_correction_progress.setValue(value)
        self._source_correction_progress.setFormat(f"{stage} {current}/{total}")
        self._source_correction_state.setText(str(stage or "OCR 裁决任务进行中…"))
        self._summary.setText(f"{stage}：{current}/{total}。任务在后台执行，窗口仍可响应。")

    def _on_source_correction_error(self, token: int, title: str, message: str):
        self = self._tab
        self._source_correction_signal_refs.pop(token, None)
        if not self._source_correction_generation.is_current(token):
            return
        self._set_source_correction_busy(False)
        self._source_correction_state.setText("OCR 裁决任务失败")
        if "导出" in str(title or ""):
            self.package_exported.emit({
                "kind": "source_correction_v5",
                "status": "error",
                "path": "",
                "error": str(message or "未知错误"),
                "details": {},
            })
        show_error_dialog(self, title, str(message or "未知错误"))

    def _export_model_source_correction_package(self):
        self = self._tab
        if self._comparison is None or len(self._documents) < 2:
            QMessageBox.warning(self, "没有多模型结果", "请先完成多模型 OCR 并载入 OCR 对比。")
            return
        if self._source_correction_busy or self._ai_import_busy:
            notify(self, "当前已有逐源纠错或 AI 导入任务在后台执行。", "warning")
            return
        comparison = self._comparison_with_current_source_texts()
        if comparison is None:
            QMessageBox.warning(self, "无法准备 AI OCR 裁决包", "当前 OCR 行数与对齐不一致，请先点击“重新对齐”。")
            return
        title = safe_result_filename(
            str(getattr(getattr(self._primary_doc, "metadata", None), "title", "") or ""),
            default="multi_ocr",
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 V5 多模型 AI OCR 裁决包",
            str((Path.home() / "Downloads" / f"{title}_OCR多模型AI裁决包_v5.zip") if (Path.home() / "Downloads").exists() else Path.home() / f"{title}_OCR多模型AI裁决包_v5.zip"), "ZIP 纠错包 (*.zip)",
        )
        if not path:
            return
        token = self._source_correction_generation.begin()
        signals = WorkerSignals()
        self._source_correction_signal_refs[token] = signals
        signals.phase_progress.connect(
            lambda stage, current, total, t=token: self._on_source_correction_progress(t, stage, current, total)
        )
        signals.error.connect(
            lambda message, t=token: self._on_source_correction_error(t, "导出 AI OCR 裁决包失败", message)
        )
        signals.finished.connect(lambda result, t=token: self._on_source_correction_export_ready(t, result))
        documents = list(self._documents)
        labels = list(self._labels)
        self._sync_canonical_authority_from_states()
        recovery_selection_records = self._capture_current_fusion_selection_records()
        recovery_decisions = [dict(item) for item in self._canonical_source_decisions.values()]
        recovery_row_index = int(self._current_row_index)
        recovery_ruby_overlay = self._ruby_overlay_doc
        output_path = str(path)
        self._set_source_correction_busy(
            True, "正在后台生成单一 GPT OCR 裁决包；已完成裁决保持锁定，只导出剩余待判断项…", lock_workspace=False,
        )

        def worker():
            try:
                from engine.multi_ocr_source_correction import (
                    documents_with_comparison_texts, export_source_correction_bundle,
                )
                def progress(stage, current, total):
                    signals.phase_progress.emit(stage, current, total)
                synced = documents_with_comparison_texts(
                    documents, comparison, progress_callback=progress,
                )
                report = export_source_correction_bundle(
                    synced, labels, comparison, output_path, progress_callback=progress,
                    fusion_selection_records=recovery_selection_records,
                    canonical_decisions=recovery_decisions,
                    review_prior_decisions=False,
                    current_row_index=recovery_row_index,
                    ruby_overlay_source=recovery_ruby_overlay,
                    include_recovery_snapshot=False,
                    bundle_profile="ai_quick",
                )
                signals.finished.emit(report)
            except Exception as exc:
                signals.error.emit(str(exc))

        threading.Thread(
            target=worker, name=f"source-correction-export-{token}", daemon=True,
        ).start()

    def _on_source_correction_export_ready(self, token: int, result: object):
        self = self._tab
        self._source_correction_signal_refs.pop(token, None)
        if not self._source_correction_generation.is_current(token):
            return
        if not isinstance(result, dict):
            self._on_source_correction_error(token, "导出 AI OCR 裁决包失败", "后台导出没有返回有效结果。")
            return
        report = dict(result)
        self._source_correction_export_report = report
        self._set_source_correction_busy(False)
        self._source_correction_progress.setValue(100)
        self._source_correction_state.setText(
            f"已导出 · 复审既有 AI 结果 {report.get('prior_decision_review_rows', 0)} · "
            f"共同候选待核验 {report.get('editable_provisional_rows', 0)} / {report.get('provisional_consensus_rows', 0)} · "
            f"本轮待审 {report.get('pending_review_rows', 0)} · "
            f"包ID {str(report.get('package_id', ''))[:8]}"
        )
        self.package_exported.emit({
            "kind": "source_correction_v5",
            "status": "ok",
            "path": str(report.get("path", "") or ""),
            "details": {
                "package_id": str(report.get("package_id", "") or ""),
                "editable_conflict_rows": int(report.get("editable_conflict_rows", 0) or 0),
                "pending_review_rows": int(report.get("pending_review_rows", 0) or 0),
                "image_count": int(report.get("image_count", 0) or 0),
                "recovery_snapshot_included": bool(report.get("recovery_snapshot_included")),
            },
        })
        notify(
            self,
            f"真正分歧总数：{report.get('editable_conflict_rows', 0)} · "
            f"本轮待审：{report.get('pending_review_rows', 0)} · "
            f"视觉证据：{report.get('image_count', 0)} 张\n"
            f"裁决包：{report.get('path', '')}",
            "success",
        )

    def set_recovery_page_image_provider(self, provider) -> None:
        """Provide current Page Manager images for post-crash path rebinding."""
        self = self._tab
        self._recovery_page_image_provider = provider if callable(provider) else None

    def _current_recovery_page_images(self) -> list[str]:
        self = self._tab
        provider = self._recovery_page_image_provider
        if not callable(provider):
            return []
        try:
            return [str(value) for value in (provider() or []) if str(value)]
        except Exception:
            return []

    def _restore_model_source_correction_session(self):
        self = self._tab
        if self._source_correction_busy or self._ai_import_busy:
            notify(self, "当前已有逐源纠错或 AI 导入任务在后台执行。", "warning")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "恢复多模型 OCR 裁决会话", "", "当前 AI OCR 裁决包 (*.zip)",
        )
        if not path:
            return
        if self._comparison is not None and self._documents:
            answer = QMessageBox.question(
                self, "替换当前 OCR 对比会话",
                "恢复会替换当前 OCR 对比中的 2～6 份模型文档和候选状态，但不会重新 OCR。继续吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        token = self._source_correction_generation.begin()
        signals = WorkerSignals()
        self._source_correction_signal_refs[token] = signals
        signals.phase_progress.connect(
            lambda stage, current, total, t=token: self._on_source_correction_progress(t, stage, current, total)
        )
        signals.error.connect(
            lambda message, t=token: self._on_source_correction_error(t, "恢复多模型 OCR 会话失败", message)
        )
        signals.finished.connect(lambda result, t=token: self._on_source_session_restore_ready(t, result))
        page_images = self._current_recovery_page_images()
        source_path = str(path)
        self._set_source_correction_busy(
            True, "正在后台恢复压缩 OCR 文档并重新建立对齐；不会调用任何 OCR 模型…", lock_workspace=True,
        )

        def worker():
            try:
                from engine.multi_ocr_source_correction import load_source_correction_recovery
                from engine.ocr_compare_view_model import build_fusion_states
                def progress(stage, current, total):
                    signals.phase_progress.emit(stage, current, total)
                documents, labels, comparison, report = load_source_correction_recovery(
                    source_path, replacement_page_images=page_images, progress_callback=progress,
                )
                prepared = [
                    "\n".join(str(row.texts[index] if index < len(row.texts) else "") for row in comparison.rows)
                    for index in range(len(documents))
                ]
                states = build_fusion_states(comparison.rows, auto_choose=False)
                from engine.multi_ocr_source_correction import apply_canonical_decisions_to_fusion_states
                decisions = [item for item in (report.get("canonical_decisions") or []) if isinstance(item, dict)]
                apply_canonical_decisions_to_fusion_states(states, comparison, decisions)
                signals.finished.emit({
                    "documents": documents, "labels": labels, "comparison": comparison,
                    "report": report, "prepared": prepared, "fusion_states": states,
                    "canonical_decisions": decisions,
                })
            except Exception as exc:
                signals.error.emit(str(exc))

        threading.Thread(target=worker, name=f"source-session-restore-{token}", daemon=True).start()

    def _on_source_session_restore_ready(self, token: int, result: object):
        self = self._tab
        self._source_correction_signal_refs.pop(token, None)
        if not self._source_correction_generation.is_current(token):
            return
        if not isinstance(result, dict):
            self._on_source_correction_error(token, "恢复多模型 OCR 会话失败", "后台恢复没有返回有效结果。")
            return
        try:
            documents = list(result.get("documents") or [])
            labels = list(result.get("labels") or [])
            comparison = result.get("comparison")
            report = dict(result.get("report") or {})
            restored_ruby_overlay = report.get("ruby_overlay")
            self.clear(preserve_available_single=True)
            self._initial_payload = {
                "documents": tuple(documents), "labels": tuple(labels),
                "comparison": comparison, "source_texts": tuple(result.get("prepared") or ()),
                "ruby_overlay": restored_ruby_overlay,
            }
            self._ruby_overlay_doc = restored_ruby_overlay
            self._labels = labels
            decisions = [item for item in (result.get("canonical_decisions") or report.get("canonical_decisions") or []) if isinstance(item, dict)]
            from engine.multi_ocr_source_correction import canonical_decision_key
            self._canonical_source_decisions = {
                canonical_decision_key(item): dict(item)
                for item in decisions if canonical_decision_key(item)
            }
            decision_keys = set(self._canonical_source_decisions)
            restored_selections = {}
            restored_records = {
                key: value for key, value in (report.get("fusion_selection_records") or {}).items()
                if key not in decision_keys
            }
            self._replace_multi_source_workspace(
                documents, comparison, preserve_selection_records=restored_records,
                prepared=result.get("prepared") or [], fusion_states=result.get("fusion_states") or [],
            )
            self._mode = "multi"
            # _replace_multi_source_workspace runs while clear() still marks
            # the tab as empty.  Re-sync after switching to multi so restored
            # sessions can use the visible "显示全文对比" control immediately.
            self._sync_review_mode_controls()
            self._current_row_index = min(
                max(0, int(report.get("current_row_index", 0) or 0)),
                max(0, len(comparison.rows) - 1),
            )
            self._workspace_title.setText("OCR 对比 · 已恢复纠错会话")
            self._choose_label.setVisible(True)
            self._row_state.setVisible(True)
            self._result_panel.setVisible(True)
            self._auto_btn.setEnabled(True)
            self._realign_btn.setEnabled(True)
            self._restore_btn.setEnabled(True)
            self._unicode_normalize_btn.setEnabled(True)
            self._export_texts_btn.setEnabled(True)
            self._export_ai_package_btn.setEnabled(True)
            self._export_source_correction_btn.setEnabled(True)
            self._import_source_correction_btn.setEnabled(True)
            self._export_fusion_skeleton_btn.setEnabled(True)
            self._export_ai_repair_epub_btn.setEnabled(True)
            self._import_ai_repair_result_btn.setEnabled(True)
            self._ai_adjudicate_btn.setEnabled(True)
            self._import_ai_package_btn.setEnabled(True)
            self._apply_btn.setEnabled(True)
            self._select_row(self._current_row_index)
        except Exception as exc:
            self._on_source_correction_error(token, "载入恢复会话失败", str(exc))
            return
        self._set_source_correction_busy(False)
        rebind = report.get("image_rebind") or {}
        self._source_correction_state.setText(
            f"会话已恢复 · {len(documents)} 模型 · {len(comparison.rows)} 句"
        )
        try:
            from engine.multi_ocr_compare import build_fused_document
            result_lines = [state.output_text() for state in self._fusion_states]
            delete_flags = [state.output_delete_intentionally() for state in self._fusion_states]
            fused = build_fused_document(
                self._primary_doc, comparison, result_lines, delete_flags=delete_flags,
                ruby_overlay_source=self._ruby_overlay_doc,
            )
        except Exception:
            fused = self._primary_doc
        self._ruby_overlay_doc = fused
        self.multi_session_restored.emit({
            "documents": documents, "labels": labels, "comparison": comparison,
            "fused": fused, "report": report,
        })
        notify(
            self,
            f"多模型 OCR 会话已恢复：{len(documents)} 个模型 · {len(comparison.rows)} 个对齐句 · "
            f"重新绑定图片 {rebind.get('rebound', 0)}。没有重新运行 OCR。",
            "success",
        )

    def _capture_current_fusion_selection_records(self) -> dict[tuple[str, ...], dict]:
        """Capture selected text *and* deletion/provenance for crash-safe recovery."""
        self = self._tab
        records: dict[tuple[str, ...], dict] = {}
        if self._comparison is None:
            return records
        from engine.multi_ocr_source_correction import canonical_decision_key
        for row, state in zip(self._comparison.rows, self._fusion_states):
            selected = state.selected_index
            if selected is None or not 0 <= selected < len(state.candidates):
                continue
            candidate = state.candidates[selected]
            key = canonical_decision_key(row)
            column_ids = [str(item) for item in (row.column_ids or ()) if str(item)]
            if not key or not column_ids:
                continue
            records[key] = {
                "sentence_group_id": str(getattr(row, "sentence_group_id", "") or ""),
                "column_ids": list(column_ids),
                "text": str(getattr(candidate, "text", "") or ""),
                "delete_intentionally": bool(getattr(candidate, "delete_intentionally", False)),
                "display_label": str(getattr(candidate, "display_label", "") or ""),
                "reason": str(getattr(candidate, "reason", "") or ""),
                "confidence": float(getattr(candidate, "confidence", 0.0) or 0.0),
                "selection_origin": str(getattr(state, "selection_origin", "") or ""),
            }
        return records

    def _replace_multi_source_workspace(
        self, documents, comparison, *, preserve_selection_records=None,
        prepared=None, fusion_states=None
    ):
        self = self._tab
        self._documents = list(documents)
        self._primary_doc = self._documents[0] if self._documents else None
        self._comparison = comparison
        if prepared is None:
            prepared = [
                "\n".join(str(row.texts[index] if index < len(row.texts) else "") for row in comparison.rows)
                for index in range(len(self._documents))
            ]
        else:
            prepared = list(prepared)
        self._set_full_source_texts(prepared)
        self._loading_text = True
        try:
            for index, editor in enumerate(self._source_editors):
                visible = index < len(self._documents)
                self._source_panels[index].setVisible(visible)
                if not visible:
                    editor.clear()
                    continue
                editor.setUpdatesEnabled(False)
                try:
                    editor.setReadOnly(self._single_card_enabled)
                    editor.setPlainText(
                        self._source_line_text(index, self._current_row_index)
                        if self._single_card_enabled else self._source_text(index)
                    )
                finally:
                    editor.setUpdatesEnabled(True)
        finally:
            self._loading_text = False
        self._sources_dirty = False
        self._set_source_labels_for_mode()
        if fusion_states is None:
            self._rebuild_fusion_rows(auto_choose=False)
        else:
            self._clear_fusion_widgets()
            self._fusion_states = list(fusion_states)
            self._render_fusion_window(self._current_row_index, force=True)
            self._update_unresolved_summary()
        preserved = 0
        from engine.ocr_compare_view_model import upsert_external_candidate
        record_map = dict(preserve_selection_records or {})
        from engine.multi_ocr_source_correction import canonical_decision_key
        for row, state in zip(self._comparison.rows, self._fusion_states):
            key = canonical_decision_key(row)
            record = record_map.get(key)
            if not isinstance(record, dict):
                continue
            selected_text = str(record.get("text", "") or "")
            delete_intentionally = bool(record.get("delete_intentionally", False))
            display_label = str(record.get("display_label", "") or "恢复的人工融合结果")
            reason = str(record.get("reason", "") or "从逐源纠错恢复会话中的已选融合结果重建。")
            confidence = float(record.get("confidence", 1.0) or 0.0)
            selection_origin = str(record.get("selection_origin", "") or "restored_human")
            if selected_text and state.choose_text(selected_text, origin=selection_origin):
                preserved += 1
                continue
            if upsert_external_candidate(
                state, selected_text,
                display_label=display_label,
                select=True,
                reason=reason,
                confidence=confidence,
                allow_empty=delete_intentionally,
                force_role_candidate=True,
                selection_origin=selection_origin,
            ) is not None:
                preserved += 1
        self._render_fusion_window(self._current_row_index, force=True)
        self._refresh_source_highlights()
        self._update_unresolved_summary()
        if self._comparison.rows:
            self._select_row(min(self._current_row_index, len(self._comparison.rows) - 1))
        return preserved

    def _import_model_source_correction_result(self):
        self = self._tab
        if self._comparison is None or len(self._documents) < 2:
            QMessageBox.warning(self, "没有多模型结果", "请先完成多模型 OCR。")
            return
        if self._source_correction_busy or self._ai_import_busy:
            notify(self, "当前已有逐源纠错或 AI 导入任务在后台执行。", "warning")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "导入 AI OCR 裁决结果", "",
            "当前 AI OCR 裁决结果 (*.json *.zip);;JSON (*.json);;ZIP (*.zip)",
        )
        if not path:
            return
        if not self._alignment_counts_valid(show_warning=False):
            QMessageBox.warning(self, "无法导入", "当前 OCR 行数与对齐不一致，请先点击“重新对齐”。")
            return
        # Import is an overlay operation, so it can validate against the existing
        # immutable comparison directly.  Avoid the former GUI-thread deep copy
        # of thousands of rows and full-text editor reads before the worker starts.
        comparison = self._comparison
        token = self._source_correction_generation.begin()
        signals = WorkerSignals()
        self._source_correction_signal_refs[token] = signals
        signals.phase_progress.connect(
            lambda stage, current, total, t=token: self._on_source_correction_progress(t, stage, current, total)
        )
        signals.error.connect(
            lambda message, t=token: self._on_source_correction_error(t, "导入 AI OCR 裁决失败", message)
        )
        signals.finished.connect(lambda result, t=token: self._on_source_correction_import_ready(t, result))
        # Keep the original model objects and the current alignment by reference.
        # The worker only validates the package and derives lightweight overlay
        # decisions; it never clones books, rewrites OCR, or realigns rows.
        documents = list(self._documents)
        labels = list(self._labels)
        source_path = str(path)
        self._set_source_correction_busy(
            True,
            "正在后台校验 AI 纠错并建立非破坏式融合覆盖；原始 OCR 和当前分歧保持不变…",
            lock_workspace=False,
        )

        def worker():
            try:
                from engine.multi_ocr_source_correction import import_source_corrections

                def progress(stage, current, total):
                    signals.phase_progress.emit(stage, current, total)

                _same_documents, _same_comparison, report = import_source_corrections(
                    source_path, documents, labels, comparison, progress_callback=progress,
                )
                decisions = [
                    item for item in (report.get("canonical_decisions") or [])
                    if isinstance(item, dict)
                ]
                signals.finished.emit({
                    "report": report,
                    "canonical_decisions": decisions,
                })
            except Exception as exc:
                signals.error.emit(str(exc))

        threading.Thread(
            target=worker, name=f"source-correction-import-{token}", daemon=True,
        ).start()

    def _on_source_correction_import_ready(self, token: int, result: object):
        self = self._tab
        self._source_correction_signal_refs.pop(token, None)
        if not self._source_correction_generation.is_current(token):
            return
        if not isinstance(result, dict):
            self._on_source_correction_error(token, "导入 AI OCR 裁决失败", "后台导入没有返回有效结果。")
            return
        try:
            report = dict(result.get("report") or {})
            decisions = [
                item for item in (
                    result.get("canonical_decisions")
                    or report.get("canonical_decisions")
                    or []
                ) if isinstance(item, dict)
            ]
            from engine.multi_ocr_source_correction import (
                apply_canonical_decisions_to_fusion_states, canonical_decision_key,
                merge_canonical_decision_overlays,
            )
            existing_decisions = [dict(item) for item in self._canonical_source_decisions.values()]
            merged_decisions, merge_stats = merge_canonical_decision_overlays(existing_decisions, decisions)
            effective_decisions = [
                dict(item) for item in (merge_stats.get("changed_accepted_decisions") or [])
                if isinstance(item, dict)
            ]
            affected_keys = {
                canonical_decision_key(item) for item in effective_decisions
                if canonical_decision_key(item)
            }
            preserved = sum(
                1
                for row, state in zip(self._comparison.rows, self._fusion_states)
                if state.selected_index is not None
                and canonical_decision_key(row) not in affected_keys
            )
            self._canonical_source_decisions = {
                canonical_decision_key(item): dict(item)
                for item in merged_decisions if canonical_decision_key(item)
            }
            # Only newly accepted or genuinely improved rows are selected again.
            # Re-importing an identical package is idempotent and cannot steal a
            # later manual selection; unresolved later rows keep prior good AI.
            overlay_applied = apply_canonical_decisions_to_fusion_states(
                self._fusion_states, self._comparison, effective_decisions,
            )
            # Only the virtualised judgement window is rebuilt.  The multi-model OCR
            # documents, source editors, alignment and full-text buffers are left
            # untouched, eliminating the former post-import freeze.
            self._render_fusion_window(self._current_row_index, force=True)
            self._update_unresolved_summary()
            if self._comparison.rows:
                self._select_row(min(self._current_row_index, len(self._comparison.rows) - 1))
        except Exception as exc:
            self._on_source_correction_error(token, "载入 AI 纠错覆盖失败", str(exc))
            return
        package_id = str(report.get("package_id", "") or "")
        duplicate_package = bool(package_id and package_id in self._source_correction_imported_package_ids)
        if package_id:
            self._source_correction_imported_package_ids.add(package_id)
        history_entry = {
            "package_id": package_id,
            "accepted_in_package": int(report.get("accepted_canonical_decisions", 0) or 0),
            "new_accepted": int(merge_stats.get("new_accepted_rows", 0) or 0),
            "replaced_accepted": int(merge_stats.get("replaced_accepted_rows", 0) or 0),
            "preserved_prior": int(merge_stats.get("preserved_prior_rows", 0) or 0),
            "effective_applied": int(overlay_applied or 0),
            "duplicate_package": duplicate_package,
        }
        if not duplicate_package:
            self._source_correction_import_history.append(history_entry)
        # Persist cumulative audit state for later fusion/skeleton exports.
        report["cumulative_import_count"] = len(self._source_correction_import_history)
        report["cumulative_canonical_decisions"] = [
            dict(item) for item in self._canonical_source_decisions.values()
        ]
        report["cumulative_accepted_canonical_decisions"] = sum(
            1 for item in self._canonical_source_decisions.values()
            if str(item.get("status", "") or "") == "accepted"
        )
        report["cumulative_merge_stats"] = {
            key: value for key, value in merge_stats.items()
            if key != "changed_accepted_decisions"
        }
        report["cumulative_import_history"] = copy.deepcopy(self._source_correction_import_history)
        self._last_source_correction_report = report
        self._set_source_correction_busy(False)
        self._source_correction_progress.setValue(100)
        unresolved_decisions = int(report.get("unresolved_canonical_decisions", 0) or 0)
        resumed_locked = int(report.get("prefilled_resolved_decisions", 0) or 0)
        proposed_cells = int(report.get("proposed_model_cells", 0) or 0)
        proposed_rows = int(report.get("proposed_model_rows", 0) or 0)
        visible_disagreements = int(report.get("resolved_history_rows_annotated", 0) or 0)
        import_rounds = len(self._source_correction_import_history)
        cumulative_accepted = sum(
            1 for item in self._canonical_source_decisions.values()
            if str(item.get("status", "") or "") == "accepted"
        )
        improved = int(merge_stats.get("replaced_accepted_rows", 0) or 0)
        newly_added = int(merge_stats.get("new_accepted_rows", 0) or 0)
        self._source_correction_state.setText(
            f"AI裁决累计 {import_rounds} 包 · 当前有效 {cumulative_accepted} · "
            f"本包新增 {newly_added} / 改进 {improved} · 待判断 {unresolved_decisions}"
        )
        self._summary.setText(
            f"AI OCR 裁决已累积合并：当前保留 {cumulative_accepted} 条有效 AI 裁决；"
            f"本包新增 {newly_added} 条、改进替换 {improved} 条。缺失/未决行不会清掉之前结果；"
            f"原始各模型 OCR、物理列和对齐均未修改。"
        )
        # Notify downstream workspaces without replacing/copying model documents.
        # The MainWindow recognises the overlay flag and deliberately skips the
        # expensive restored-session path.
        self.multi_session_restored.emit({
            "documents": list(self._documents),
            "labels": list(self._labels),
            "comparison": self._comparison,
            "fused": None,
            "report": report,
        })
        validation_mode = str(report.get("identity_validation_mode", "") or "")
        if validation_mode == "sealed_row_evidence_and_alignment":
            identity_note = "\n原始文档侧通道存在无关漂移；已核验全部密封行证据与稳定对齐完全一致后安全导入。"
        else:
            identity_note = "\n已按当前模型文档快照和密封对齐快照做严格一致性校验。"
        proposed_labels = [
            str(value) for value in (report.get("proposed_model_labels") or []) if str(value)
        ]
        proposed_text = "、".join(proposed_labels) or "无逐模型修改"
        merge_conflicts = int(report.get("three_way_merge_conflicts", 0) or 0)
        notify(
            self,
            f"AI OCR 裁决已非破坏式导入：本包新增 {newly_added}、改进 {improved}；"
            f"当前有效 {cumulative_accepted}，仍需判断 {unresolved_decisions}。"
            "原始各模型 OCR、物理列和对齐均保持不变。",
            "success",
        )
