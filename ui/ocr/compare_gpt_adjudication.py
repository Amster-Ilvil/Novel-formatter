from __future__ import annotations

import copy
import json
import threading
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog

from adapters.result_export import safe_result_filename
from ai.gemini_ocr_transport import GeminiRequestError
from ui.common.signals import WorkerSignals
from ui.common.toast import notify
from ui.dialogs import show_error_dialog
from ui.localized_dialogs import LocalizedMessageBox as QMessageBox
from ui.ocr.compare_widgets import OCRAIAdjudicationDialog
from ui.settings.ai_dialog import ensure_ai_settings


def _apply_ai_report_to_fusion_states(fusion_states, lines, package_items, audit_report):
    """Apply accepted AI judgements, including explicit no-change confirmations.

    A reviewed row is a decision even when the final text is byte-for-byte equal
    to the pre-review fusion text.  Marking that existing candidate with an
    explicit AI origin removes it from the pending queue and makes the decision
    durable in the canonical/project snapshot without altering raw OCR evidence.
    """
    from engine.ocr_compare_view_model import upsert_external_candidate

    uncertain_ids = {str(value) for value in (audit_report.get("low_uncertain") or []) if str(value)}
    report_by_id = {
        str(entry.get("item_id", "") or ""): entry
        for entry in (audit_report.get("items") or [])
        if isinstance(entry, dict) and str(entry.get("item_id", "") or "")
    }
    stats = {"resolved": 0, "confirmed_unchanged": 0, "changed": 0, "uncertain": 0}
    if len(lines) != len(fusion_states):
        return stats
    image_only = audit_report.get("mode") == "image_only"

    for row_index, (state, text) in enumerate(zip(fusion_states, lines)):
        row_id = (
            str(package_items[row_index].get("row_id", "") or "")
            if row_index < len(package_items) and isinstance(package_items[row_index], dict)
            else ""
        )
        report_entry = report_by_id.get(row_id, {})
        if row_id in uncertain_ids:
            if image_only:
                state.requires_confirmation = True
                state.reopen()
            state.local_reocr_recommended = True
            state.fusion_reason = str(
                report_entry.get("reason", "视觉证据不足，保留人工复核")
                or "视觉证据不足，保留人工复核"
            )
            stats["uncertain"] += 1
            continue
        if not report_entry or bool(report_entry.get("needs_human_review", False)):
            if report_entry:
                stats["uncertain"] += 1
            continue

        before = str(report_entry.get("before", "") or "")
        after = str(report_entry.get("after", text) or "")
        selected = upsert_external_candidate(
            state,
            after,
            display_label="AI 组合图裁决" if image_only else "出版级 AI 裁决",
            select=True,
            reason=str(report_entry.get("reason", "") or ("AI 已核验原文无需修改" if after == before else "")),
            allow_empty=False,
            selection_origin="ai_visual_batch_adjudication" if image_only else "ai_gpt_grade_adjudication",
        )
        if selected is None:
            continue
        stats["resolved"] += 1
        if after == before:
            stats["confirmed_unchanged"] += 1
        else:
            stats["changed"] += 1
    return stats


class OCRCompareGPTAdjudicationController:
    """Own the batch visual adjudication lifecycle for OCRCompareTab.

    OCR/fusion documents remain owned by the tab.  The controller orchestrates
    provider execution, cancellation/progress, result sealing and Runtime Task
    State events without changing raw OCR evidence.
    """

    def __init__(self, tab):
        self._tab = tab

    def _run_ai_adjudication(self):
        self = self._tab
        if self._comparison is None or len(self._documents) < 2 or self._primary_doc is None:
            QMessageBox.warning(self, "没有多模型结果", "请先完成多模型 OCR 并载入 OCR 对比。")
            return
        if self._ai_import_busy or self._source_correction_busy:
            notify(self, "当前已有 AI 或逐源纠错任务，请稍候。", "warning")
            return
        if not self._alignment_counts_valid(show_warning=False):
            notify(self, "各 OCR 栏行数已改变。请先点击“重新对齐”。", "warning")
            return
        from engine.ocr_compare_view_model import is_explicit_fusion_selection_origin

        resolved_states = {
            index: state for index, state in enumerate(self._fusion_states)
            if state.selected_index is not None and is_explicit_fusion_selection_origin(state.selection_origin)
        }
        if self._fusion_states and len(resolved_states) == len(self._fusion_states):
            notify(self, "当前没有待裁决的分歧。", "info")
            return
        dialog = OCRAIAdjudicationDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        config = dialog.values()
        settings = ensure_ai_settings(
            self,
            "请在 AI 设置中选择支持图片输入的模型，用于识别分歧组合图。",
        )
        if settings is None:
            return
        if settings.provider == "gemini":
            try:
                settings = replace(settings, **dialog.gemini_options())
            except ValueError as exc:
                QMessageBox.warning(self, "Gemini 裁决模型无效", str(exc))
                return
        settings = replace(
            settings,
            json_mode=False,  # Gemini uses a schema; compatible endpoints use the prompt format.
            deepseek_thinking=False,
            max_tokens=max(1500, min(8000, int(getattr(settings, "max_tokens", 8000) or 8000))),
        )
        comparison = self._comparison_with_current_source_texts()
        if comparison is None:
            QMessageBox.warning(self, "无法裁决", "当前 OCR 对齐状态不完整。")
            return
        result_lines: list[str] = []
        delete_flags: list[bool] = []
        for row_index, state in enumerate(self._fusion_states):
            value = state.output_text()
            if state.selected_index is None and row_index < len(comparison.rows):
                value = comparison.rows[row_index].output_text
            result_lines.append(str(value or ""))
            delete_flags.append(state.output_delete_intentionally())

        token = self._ai_import_generation.begin()
        signals = WorkerSignals()
        self._ai_import_signal_refs[token] = signals
        cancel_event = threading.Event()
        self._ai_adjudication_cancel_event = cancel_event
        self._ai_adjudication_report = None
        self._ai_adjudication_output_paths = None
        self._ai_adjudication_task_id = f"ai-adjudication-{token}"
        self.run_log_event.emit({
            "event": "started",
            "session_id": self._ai_adjudication_task_id,
            "stage": "ai_adjudication",
            "details": {
                "summary": "AI 组合图裁决正在运行",
                "provider": str(getattr(settings, "provider", "") or ""),
                "model": str(getattr(settings, "model", "") or ""),
                "adjudication_mode": "image_only",
                "resume_supported": False,
            },
        })
        queued = Qt.ConnectionType.QueuedConnection
        signals.log.connect(lambda raw: self._on_ai_adjudication_progress(token, raw), queued)
        signals.error.connect(lambda details: self._on_ai_adjudication_error(token, details), queued)
        signals.finished.connect(lambda result: self._on_ai_adjudication_ready(token, result), queued)
        self._ai_adjudication_progress.setValue(0)
        self._ai_adjudication_progress.setFormat("准备视觉证据…")
        self._set_ai_import_busy(
            True,
            "正在将待裁决的分歧裁片拼成编号图…",
        )

        documents = list(self._documents)
        labels = list(self._labels)
        current_primary = self._primary_doc
        ruby_overlay_doc = self._ruby_overlay_doc
        title = safe_result_filename(
            str(getattr(getattr(self._primary_doc, "metadata", None), "title", "") or ""),
            default="multi_ocr",
        )
        def worker():
            try:
                from ai.multimodal_client import MultimodalClient
                from engine.ocr_adjudication_checkpoint import save_checkpoint
                from engine.ocr_visual_batch_adjudicator import VisualBatchOptions, adjudicate_visual_batches
                from engine.ocr_roundtrip_package import export_multi_package, import_multi_package
                from engine.ocr_compare_view_model import build_fusion_states

                baseline = export_multi_package(
                    documents,
                    labels,
                    comparison,
                    ruby_overlay_source=ruby_overlay_doc,
                )
                package = {
                    **baseline,
                    "editable_items": [
                        {**item, "edited_text": text, "delete_intentionally": deleted}
                        for item, text, deleted in zip(baseline["editable_items"], result_lines, delete_flags)
                    ],
                }
                package_items = list(package.get("editable_items") or [])
                target_row_ids = tuple(
                    str(item.get("row_id", "") or "")
                    for index, item in enumerate(package_items)
                    if index not in resolved_states and str(item.get("row_id", "") or "")
                )
                # Build a fused document first so the visual layer can reuse the
                # exact immutable row->page/column geometry already used by 图文对照.
                fused_before, _comparison_before, _lines_before = import_multi_package(
                    package,
                    current_primary=current_primary,
                )
                visual_options = VisualBatchOptions(
                    batch_items=int(config["batch_items"]),
                    concurrency=int(config.get("concurrency", 0) or 0),
                    image_only=True,
                    publication_review=True,
                    cell_width=400,
                    cell_height=620,
                    target_row_ids=target_row_ids,
                )

                def report(event):
                    signals.log.emit(json.dumps(event, ensure_ascii=False))

                with MultimodalClient(settings) as vision_client:
                    reviewed_package, audit_report = adjudicate_visual_batches(
                        vision_client, package, fused_before, options=visual_options,
                        progress_callback=report, cancel_check=cancel_event.is_set,
                    )
                if cancel_event.is_set():
                    raise RuntimeError("AI 裁决已取消")
                fused, imported_comparison, lines = import_multi_package(
                    reviewed_package,
                    current_primary=current_primary,
                )
                fusion_states = build_fusion_states(imported_comparison.rows, auto_choose=False)
                for index, state in resolved_states.items():
                    fusion_states[index] = copy.deepcopy(state)
                package_items = list(reviewed_package.get("editable_items") or [])
                state_stats = _apply_ai_report_to_fusion_states(
                    fusion_states, lines, package_items, audit_report
                )
                if cancel_event.is_set():
                    raise RuntimeError("AI 裁决已取消")
                paths = ()
                if audit_report.get("target_rows"):
                    paths = save_checkpoint(
                        baseline, reviewed_package, audit_report, fusion_states,
                        Path(config["output_dir"]).expanduser(), title,
                    )
                source_texts = [
                    "\n".join(row.texts[index] for row in imported_comparison.rows)
                    for index in range(len(imported_comparison.labels))
                ]
                signals.finished.emit({
                    "fused": fused if state_stats["changed"] else None,
                    "comparison": imported_comparison,
                    "lines": list(lines),
                    "fusion_states": fusion_states,
                    "source_texts": source_texts,
                    "path": str(paths[0]) if paths else "",
                    "ai_adjudication_report": audit_report,
                    "ai_adjudication_paths": tuple(map(str, paths)),
                    "ai_adjudication_state_stats": state_stats,
                })
            except GeminiRequestError as exc:
                signals.error.emit(str(exc))
            except Exception:
                import traceback
                signals.error.emit(traceback.format_exc())

        threading.Thread(
            target=worker,
            name=f"ocr-ai-visual-adjudication-{token}",
            daemon=True,
        ).start()
    def _cancel_ai_adjudication(self):
        self = self._tab
        event = self._ai_adjudication_cancel_event
        if event is None:
            return
        event.set()
        if self._ai_adjudication_task_id:
            self.run_log_event.emit({
                "event": "cancelling",
                "session_id": self._ai_adjudication_task_id,
                "stage": "ai_adjudication",
                "message": "正在停止 AI 组合图裁决",
            })
        self._ai_adjudication_cancel_btn.setEnabled(False)
        self._ai_adjudication_progress.setFormat("正在停止裁决，等待当前请求结束…")
        self._set_compare_task_progress(True, text="正在停止裁决，等待当前请求结束…")
        self._summary.setText("正在取消 AI 审定；已完成批次会被丢弃，不会写入半成品融合稿…")
    def _on_ai_adjudication_progress(self, token: int, raw: str) -> None:
        self = self._tab
        if not self._ai_import_generation.is_current(token):
            return
        if self._ai_adjudication_cancel_event is not None and self._ai_adjudication_cancel_event.is_set():
            return
        try:
            event = json.loads(raw)
        except Exception:
            self._summary.setText(str(raw or "AI 审定中…"))
            return
        stage = str(event.get("stage", "AI 审定") or "AI 审定")
        current = int(event.get("current", 0) or 0)
        total = max(1, int(event.get("total", 1) or 1))
        value = min(99, max(0, int(current * 100 / total)))
        self._ai_adjudication_progress.setValue(value)
        self._ai_adjudication_progress.setFormat(f"{stage} {current}/{total}")
        self._set_compare_task_progress(
            True, value=value, text=f"{stage} {current}/{total}", indeterminate=False
        )
        page_start = int(event.get("page_start", 0) or 0)
        page_end = int(event.get("page_end", 0) or 0)
        page_note = f"；第 {page_start}–{page_end} 页" if page_start else ""
        token_estimate = int(event.get("estimated_tokens", 0) or 0)
        token_note = f"；本请求约 {token_estimate / 1000:.1f}k 输入 tokens" if token_estimate else ""
        target_rows = int(event.get("target_rows", 0) or 0)
        target_note = f"；风险条目 {target_rows}" if target_rows else ""
        concurrency = int(event.get("concurrency", 0) or 0)
        peak = int(event.get("peak_concurrency", 0) or 0)
        concurrency_note = f"；并发 {concurrency}" if concurrency else ""
        if peak and peak != concurrency:
            concurrency_note += f"（峰值 {peak}）"
        self._summary.setText(f"AI OCR 审定：{stage} {current}/{total}{page_note}{target_note}{token_note}{concurrency_note}")
    def _on_ai_adjudication_error(self, token: int, details: str) -> None:
        self = self._tab
        self._ai_import_signal_refs.pop(token, None)
        if not self._ai_import_generation.is_current(token):
            return
        cancelled = self._ai_adjudication_cancel_event is not None and self._ai_adjudication_cancel_event.is_set()
        self._ai_adjudication_cancel_event = None
        self._ai_adjudication_progress.setVisible(False)
        self._ai_adjudication_cancel_btn.setVisible(False)
        self._set_ai_import_busy(False)
        task_id = self._ai_adjudication_task_id
        self._ai_adjudication_task_id = ""
        if cancelled or "AdjudicationCancelled" in str(details):
            if task_id:
                self.run_log_event.emit({
                    "event": "finished", "session_id": task_id, "stage": "ai_adjudication",
                    "status": "cancelled",
                    "details": {"summary": "AI 组合图裁决已取消"},
                })
            self._summary.setText("AI 审定已取消；当前融合稿没有被半途改写。")
            return
        if task_id:
            self.run_log_event.emit({
                "event": "finished", "session_id": task_id, "stage": "ai_adjudication",
                "status": "failed", "error": str(details or "未知错误"),
                "details": {"summary": "AI 组合图裁决失败"},
            })
        show_error_dialog(self, "AI 审定融合稿失败", str(details or "未知错误"))
    def _on_ai_adjudication_ready(self, token: int, result: object) -> None:
        self = self._tab
        if not self._ai_import_generation.is_current(token):
            return
        if self._ai_adjudication_cancel_event is not None and self._ai_adjudication_cancel_event.is_set():
            self._on_ai_adjudication_error(token, "AI 裁决已取消")
            return
        if not isinstance(result, dict):
            self._on_ai_adjudication_error(token, "AI 审定后台没有返回有效结果。")
            return
        self._ai_adjudication_cancel_event = None
        self._ai_adjudication_cancel_btn.setVisible(False)
        self._ai_adjudication_report = result.get("ai_adjudication_report")
        report = self._ai_adjudication_report if isinstance(self._ai_adjudication_report, dict) else {}
        stats = report.get("stats") or {}
        state_stats = result.get("ai_adjudication_state_stats") or {}
        network_failed_rows = int(stats.get("network_failed_rows", 0) or 0)
        uncertain = int(stats.get("uncertain_items", 0) or 0)
        requests = int(stats.get("request_count", 0) or 0)
        label = "AI 裁决请求失败" if network_failed_rows and network_failed_rows == stats.get("target_items") else "AI 裁决结束"
        status_text = f"{label} · 请求 {requests} 次 · 待复核 {uncertain} 条"
        self._ai_adjudication_progress.setValue(100)
        self._ai_adjudication_progress.setFormat(status_text)
        self._set_compare_task_progress(
            True, value=100, text=status_text, indeterminate=False
        )
        task_id = self._ai_adjudication_task_id
        self._ai_adjudication_task_id = ""
        if task_id:
            self.run_log_event.emit({
                "event": "finished", "session_id": task_id, "stage": "ai_adjudication",
                "status": "warning" if uncertain else "ok",
                "details": {
                    "summary": status_text,
                    "network_failed_rows": network_failed_rows,
                    "request_errors": list(report.get("request_errors") or []),
                    "quota_blocked_rows": int(stats.get("quota_blocked_rows", 0) or 0),
                    "visual_request_count": requests,
                    "visual_transport_attempts": int(stats.get("transport_attempts", 0) or 0),
                    "visual_transport_retries": int(stats.get("transport_retries", 0) or 0),
                    "target_items": int(stats.get("target_items", 0) or 0),
                    "final_uncertain_items": uncertain,
                    "confirmed_unchanged": int(state_stats.get("confirmed_unchanged", 0) or 0),
                    "visual_applied_changes": int(state_stats.get("changed", 0) or 0),
                },
            })
        paths = result.get("ai_adjudication_paths")
        if isinstance(paths, (list, tuple)) and len(paths) >= 2:
            self._ai_adjudication_output_paths = (str(paths[0]), str(paths[1]))
        self._on_ai_import_ready(token, result)
