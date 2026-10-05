from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path

from PySide6.QtWidgets import QDialog

from adapters.result_export import safe_result_filename
from ui.common.signals import WorkerSignals
from ui.common.toast import notify
from ui.dialogs import show_error_dialog
from ui.localized_dialogs import LocalizedMessageBox as QMessageBox
from ui.ocr.compare_widgets import OCRAIAdjudicationDialog
from ui.settings.ai_dialog import ensure_ai_settings


class OCRCompareGPTAdjudicationController:
    """Own provider-neutral publication-grade AI adjudication lifecycle for OCRCompareTab.

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
        settings = ensure_ai_settings(
            self,
            "请先在 AI 设置中选择支持图片输入的模型并填写 API Key。AI 裁决会把多条 OCR 分歧拼成证据板批量提交。",
        )
        if settings is None:
            return
        from dataclasses import replace
        settings = replace(
            settings,
            json_mode=True,
            deepseek_thinking=False,
            max_tokens=max(1500, min(8000, int(getattr(settings, "max_tokens", 8000) or 8000))),
        )
        dialog = OCRAIAdjudicationDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        config = dialog.values()

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
                "summary": "出版级 AI OCR 裁决正在运行",
                "provider": str(getattr(settings, "provider", "") or ""),
                "model": str(getattr(settings, "model", "") or ""),
                "adjudication_mode": str(config.get("adjudication_mode", "gpt_grade") or "gpt_grade"),
                "resume_supported": False,
            },
        })
        signals.log.connect(lambda raw, t=token: self._on_ai_adjudication_progress(t, raw))
        signals.error.connect(lambda details, t=token: self._on_ai_adjudication_error(t, details))
        signals.finished.connect(lambda result, t=token: self._on_ai_adjudication_ready(t, result))
        self._ai_adjudication_progress.setValue(0)
        self._ai_adjudication_progress.setFormat("准备视觉证据…")
        self._set_ai_import_busy(
            True,
            "正在把真正 OCR 分歧对应的原图裁切拼成低 token 证据板；一致行不会发送给 AI…",
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
                from ai.provider_factory import create_provider
                from engine.ocr_visual_batch_adjudicator import VisualBatchOptions, analyse_routing
                from engine.ocr_ai_adjudicator import AdjudicationOptions
                from engine.ocr_gpt_grade_adjudicator import (
                    GptGradeAdjudicationOptions, adjudicate_gpt_grade,
                )
                from engine.ocr_roundtrip_package import export_multi_package, import_multi_package
                from engine.ocr_compare_view_model import build_fusion_states, upsert_external_candidate

                package = export_multi_package(
                    documents,
                    labels,
                    comparison,
                    result_lines=result_lines,
                    delete_flags=delete_flags,
                    ruby_overlay_source=ruby_overlay_doc,
                )
                package_items = list(package.get("editable_items") or [])
                # Deterministic local targeted-retry decisions are frozen.  Human
                # and previous AI choices stay visible as context and may be
                # revisited by the publication-grade pass, matching the external package
                # workflow while preserving raw OCR evidence.
                local_resolved_indices = {
                    index for index, state in enumerate(self._fusion_states)
                    if str(getattr(state, "selection_origin", "") or "")
                    == "local_targeted_retry_majority_adjudication"
                }
                target_row_ids = tuple(
                    str(item.get("row_id", "") or "")
                    for index, item in enumerate(package_items)
                    if index not in local_resolved_indices and str(item.get("row_id", "") or "")
                )
                # Build a fused document first so the visual layer can reuse the
                # exact immutable row->page/column geometry already used by 图文对照.
                fused_before, _comparison_before, _lines_before = import_multi_package(
                    package,
                    current_primary=current_primary,
                )
                visual_options = VisualBatchOptions(
                    batch_items=int(config["batch_items"]),
                    risk_only=bool(config["risk_only"]),
                    allow_novel_text=bool(config["allow_novel_text"]),
                    reasoning_effort=str(config["reasoning_effort"] or "high"),
                    routing_mode=str(config.get("routing_mode", "smart") or "smart"),
                    concurrency=int(config.get("concurrency", 0) or 0),
                    retry_incomplete=bool(config.get("retry_incomplete", True)),
                    transcription_first=True,
                    targeted_verify=True,
                    target_row_ids=target_row_ids,
                )
                contextual_options = AdjudicationOptions(
                    batch_pages=int(config.get("context_pages", 20) or 20),
                    overlap_pages=2,
                    risk_only=bool(config["risk_only"]),
                    independent_audit=bool(config.get("independent_audit", True)),
                    request_retries=2,
                    max_prompt_tokens=12000,
                    auto_apply_medium=True,
                    target_row_ids=target_row_ids,
                )
                gpt_options = GptGradeAdjudicationOptions(
                    visual=visual_options,
                    contextual=contextual_options,
                    contextual_pass=(str(config.get("adjudication_mode", "gpt_grade") or "gpt_grade") == "gpt_grade"),
                )

                def report(event):
                    signals.log.emit(json.dumps(event, ensure_ascii=False))

                routing_preview = analyse_routing(
                    {**package, "editable_items": [
                        item for idx, item in enumerate(package_items)
                        if idx not in local_resolved_indices
                    ]},
                    routing_mode=str(config.get("routing_mode", "smart") or "smart"),
                    risk_only=bool(config.get("risk_only", True)),
                )
                if not int(routing_preview.get("target_items", 0) or 0):
                    audit_report = {
                        "schema": "novel_formatter.ocr_visual_batch_adjudication.v3",
                        "items": [],
                        "low_uncertain": [],
                        "stats": {
                            "target_items": 0,
                            "request_count": 0,
                            "pre_adjudicated_local": len(local_resolved_indices),
                            "routing_preview": routing_preview,
                        },
                        "summary": "本地裁决后没有剩余需要视觉 AI 的风险正文。",
                    }
                    reviewed_package = package
                else:
                    with MultimodalClient(settings) as vision_client:
                        if gpt_options.contextual_pass:
                            with create_provider(settings) as text_provider:
                                reviewed_package, audit_report = adjudicate_gpt_grade(
                                    vision_client,
                                    text_provider,
                                    package,
                                    fused_before,
                                    options=gpt_options,
                                    # Keep production adjudication blind by default.
                                    # The publication EPUB remains a post-hoc benchmark
                                    # unless a future explicit workflow opts into it.
                                    reference_path=None,
                                    glossary_path=str(getattr(settings, "glossary_path", "") or "") or None,
                                    progress_callback=report,
                                    cancel_check=cancel_event.is_set,
                                )
                        else:
                            reviewed_package, audit_report = adjudicate_gpt_grade(
                                vision_client,
                                None,
                                package,
                                fused_before,
                                options=gpt_options,
                                reference_path=None,
                                glossary_path=None,
                                progress_callback=report,
                                cancel_check=cancel_event.is_set,
                            )
                output_dir = Path(config["output_dir"]).expanduser()
                output_dir.mkdir(parents=True, exist_ok=True)
                package_path = output_dir / f"{title}_AI_GPT级裁决.json"
                report_path = output_dir / f"{title}_AI_GPT级裁决报告.json"
                package_path.write_text(json.dumps(reviewed_package, ensure_ascii=False, indent=2), encoding="utf-8")
                report_path.write_text(json.dumps(audit_report, ensure_ascii=False, indent=2), encoding="utf-8")

                fused, imported_comparison, lines = import_multi_package(
                    reviewed_package,
                    current_primary=current_primary,
                )
                fusion_states = build_fusion_states(imported_comparison.rows, auto_choose=False)
                uncertain_ids = set(str(value) for value in (audit_report.get("low_uncertain") or []))
                report_by_id = {
                    str(entry.get("item_id", "") or ""): entry
                    for entry in (audit_report.get("items") or [])
                    if isinstance(entry, dict)
                }
                package_items = list(reviewed_package.get("editable_items") or [])
                if len(lines) == len(fusion_states):
                    for row_index, (state, text) in enumerate(zip(fusion_states, lines)):
                        row_id = str(package_items[row_index].get("row_id", "") or "") if row_index < len(package_items) else ""
                        report_entry = report_by_id.get(row_id, {})
                        if row_id in uncertain_ids:
                            state.local_reocr_recommended = True
                            state.fusion_reason = str(report_entry.get("reason", "视觉证据不足，保留人工复核") or "视觉证据不足，保留人工复核")
                            continue
                        if not report_entry or bool(report_entry.get("needs_human_review", False)):
                            continue
                        before = str(report_entry.get("before", "") or "")
                        after = str(report_entry.get("after", text) or "")
                        if after == before:
                            continue
                        upsert_external_candidate(
                            state,
                            after,
                            display_label="AI GPT级裁决",
                            select=True,
                            reason=str(report_entry.get("reason", "") or ""),
                            allow_empty=False,
                            selection_origin="ai_gpt_grade_adjudication",
                        )
                source_texts = [
                    "\n".join(row.texts[index] for row in imported_comparison.rows)
                    for index in range(len(imported_comparison.labels))
                ]
                signals.finished.emit({
                    "fused": fused,
                    "comparison": imported_comparison,
                    "lines": list(lines),
                    "fusion_states": fusion_states,
                    "source_texts": source_texts,
                    "path": str(package_path),
                    "ai_adjudication_report": audit_report,
                    "ai_adjudication_paths": (str(package_path), str(report_path)),
                })
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
                "message": "正在停止出版级 AI 裁决",
            })
        self._ai_adjudication_cancel_btn.setEnabled(False)
        self._summary.setText("正在取消 AI 审定；已完成批次会被丢弃，不会写入半成品融合稿…")
    def _on_ai_adjudication_progress(self, token: int, raw: str) -> None:
        self = self._tab
        if not self._ai_import_generation.is_current(token):
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
                    "details": {"summary": "GPT级 AI OCR 裁决已取消"},
                })
            self._summary.setText("AI 审定已取消；当前融合稿没有被半途改写。")
            return
        if task_id:
            self.run_log_event.emit({
                "event": "finished", "session_id": task_id, "stage": "ai_adjudication",
                "status": "failed", "error": str(details or "未知错误"),
                "details": {"summary": "GPT级 AI OCR 裁决失败"},
            })
        show_error_dialog(self, "AI 审定融合稿失败", str(details or "未知错误"))
    def _on_ai_adjudication_ready(self, token: int, result: object) -> None:
        self = self._tab
        if not self._ai_import_generation.is_current(token):
            return
        if not isinstance(result, dict):
            self._on_ai_adjudication_error(token, "AI 审定后台没有返回有效结果。")
            return
        self._ai_adjudication_cancel_event = None
        self._ai_adjudication_cancel_btn.setVisible(False)
        self._ai_adjudication_progress.setValue(100)
        self._ai_adjudication_progress.setFormat("AI 审定与独立审计完成")
        self._ai_adjudication_report = result.get("ai_adjudication_report")
        task_id = self._ai_adjudication_task_id
        self._ai_adjudication_task_id = ""
        if task_id:
            report = self._ai_adjudication_report if isinstance(self._ai_adjudication_report, dict) else {}
            stats = dict(report.get("stats") or {})
            self.run_log_event.emit({
                "event": "finished", "session_id": task_id, "stage": "ai_adjudication",
                "status": "ok",
                "details": {
                    "summary": "GPT级 AI OCR 裁决完成",
                    "target_items": int(stats.get("target_items", 0) or 0),
                    "final_uncertain_items": int(stats.get("final_uncertain_items", 0) or 0),
                    "visual_applied_changes": int(stats.get("visual_applied_changes", 0) or 0),
                    "contextual_applied_changes": int(stats.get("contextual_applied_changes", 0) or 0),
                },
            })
        paths = result.get("ai_adjudication_paths")
        if isinstance(paths, (list, tuple)) and len(paths) >= 2:
            self._ai_adjudication_output_paths = (str(paths[0]), str(paths[1]))
        self._on_ai_import_ready(token, result)
