from __future__ import annotations

import html
import time

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMessageBox

from ui.dialogs import show_error_dialog


class OCRRunLifecycleController:
    """Own terminal OCR UI state, progress rendering and completion/error handoff."""

    def __init__(self, tab):
        self._tab = tab


    def _toggle_progress_display(self, enabled: bool) -> None:
        """Show/hide live OCR progress without pausing recognition."""
        self = self._tab
        if bool(enabled):
            self._progress_display_enabled_event.set()
            running = not self._run_btn.isEnabled()
            if running:
                self._progress_bar_wrap.setVisible(True)
                self._progress_bar_text.setVisible(True)
                self._phase_progress_lbl.setVisible(True)
                snapshot = self._latest_progress_snapshot
                if snapshot is not None:
                    self._on_overall_progress(snapshot)
                else:
                    self._phase_progress_lbl.setText("当前：等待下一次 OCR 进度事件…")
        else:
            self._progress_display_enabled_event.clear()
            self._progress_clock_timer.stop()
            self._progress_bar_wrap.setVisible(False)
            self._progress_bar_text.setVisible(False)
            self._progress_lbl.setVisible(False)
            self._phase_progress_lbl.setVisible(False)

    def _flush_ocr_log_buffer(self, *, force_all: bool = False) -> None:
        """Append buffered OCR logs in a few large Qt document operations."""
        self = self._tab
        if not hasattr(self, "_ocr_log_buffer"):
            return
        packets: list[str] = []
        while True:
            lines = self._ocr_log_buffer.drain(max_lines=1200 if force_all else 320)
            if not lines:
                break
            packets.append("\n".join(lines))
            if not force_all:
                break
        if packets:
            self._log_view.appendPlainText("\n".join(packets))
            # Always follow the newest OCR line while a run is active.  Updating
            # the scrollbar both now and on the next event-loop turn handles
            # QPlainTextEdit's delayed document-layout range update.
            if bool(getattr(self, "_ocr_run_active", False)):
                bar = self._log_view.verticalScrollBar()
                bar.setValue(bar.maximum())
                QTimer.singleShot(0, lambda b=bar: b.setValue(b.maximum()))


    def _emit_run_log_terminal(
        self,
        status: str,
        *,
        extra_details: dict | None = None,
        error: str = "",
    ) -> None:
        """Publish one terminal OCR lifecycle event for durable project logging.

        The project layer owns files and retention; OCRTab only reports an
        immutable summary plus the already-visible human log/performance trace.
        This keeps single/multi OCR independent from workspace persistence.
        """
        self = self._tab
        if self._active_ocr_log_finalized:
            return
        session_id = str(self._active_ocr_log_session_id or "")
        if not session_id:
            return
        self._flush_ocr_log_buffer(force_all=True)
        trace = self._active_ocr_performance_trace
        performance = {}
        if trace is not None:
            try:
                performance = dict(trace.finish() or {})
            except Exception:
                performance = {}
        metadata = dict(performance.get("metadata") or {})
        stages = dict(performance.get("stages") or {})
        gauges = dict(performance.get("gauges") or {})
        engine_ids = list(metadata.get("engine_ids") or [])
        role_slots = list(metadata.get("role_slots") or [])
        model_timings = []
        for index, engine_id in enumerate(engine_ids):
            stage_key = f"model_{index + 1}_{engine_id}"
            metric = dict(stages.get(stage_key) or {})
            slot = dict(role_slots[index] or {}) if index < len(role_slots) else {}
            role_name = str(slot.get("role") or ("single" if len(engine_ids) == 1 else ""))
            model_timings.append({
                "index": index + 1,
                "role": role_name,
                "engine_id": str(engine_id),
                "label": self._engine_label(str(engine_id)),
                "seconds": round(float(metric.get("total_seconds", 0.0) or 0.0), 6),
            })
        elapsed = round(float(performance.get("elapsed_seconds", 0.0) or 0.0), 6)
        ruby_seconds = round(
            float(dict(stages.get("ruby_preservation") or {}).get("total_seconds", 0.0) or 0.0), 6
        )
        details = {
            "pages": int(gauges.get("input_pages", metadata.get("page_count", 0)) or 0),
            "engines": [item["label"] for item in model_timings]
                or [self._engine_label(str(value)) for value in engine_ids],
            "engine_ids": engine_ids,
            "ocr_mode": str(metadata.get("ocr_mode") or ""),
            "elapsed_seconds": elapsed,
            "main_ocr_seconds": round(max(0.0, elapsed - ruby_seconds), 6),
            "findtext_seconds": ruby_seconds,
            "model_timings": model_timings,
            "role_slots": role_slots,
            "image_preparation_seconds": round(sum(
                float(dict(metric or {}).get("total_seconds", 0.0) or 0.0)
                for name, metric in stages.items() if str(name).endswith(".image_preparation")
            ), 6),
            "recognition_seconds": round(sum(
                float(dict(metric or {}).get("total_seconds", 0.0) or 0.0)
                for name, metric in stages.items() if str(name).endswith(".recognition")
            ), 6),
            "fusion_seconds": round(
                float(dict(stages.get("multi_model_alignment_and_fusion") or {}).get("total_seconds", 0.0) or 0.0), 6
            ),
            "cancelled": bool(gauges.get("cancelled", False)) or str(status) == "cancelled",
        }
        details.update(dict(extra_details or {}))
        self._active_ocr_log_finalized = True
        self.run_log_event.emit({
            "event": "finished",
            "session_id": session_id,
            "status": str(status),
            "details": details,
            "error": str(error or ""),
            "log_text": self._log_view.toPlainText(),
            "performance": performance,
            "performance_trace_path": str(self._last_ocr_performance_trace_path or ""),
        })


    def _on_progress(self, current, total):
        self = self._tab
        if not self._progress_display_is_enabled():
            return
        # Legacy adapters still emit page progress.  Whole-run progress is
        # controlled exclusively by _on_overall_progress so a new phase/model
        # cannot reset the bar back to zero.
        self._phase_progress_lbl.setText(f"当前：页面识别 {current} / {max(total, 1)} 页")


    def _on_phase_progress(self, label, current, total):
        self = self._tab
        if not self._progress_display_is_enabled():
            return
        unit = "句组" if "句" in str(label) and "识别" in str(label) else "页"
        if max(total, 1) == 1 and current in {0, 1}:
            self._phase_progress_lbl.setText(f"当前：{label}")
        else:
            self._phase_progress_lbl.setText(
                f"当前：{label} {current} / {max(total, 1)} {unit}"
            )


    def _set_progress_bar_text(self, percent: float, elapsed: float, eta) -> None:
        self = self._tab
        from core.ocr_progress_estimator import OCRProgressEstimator
        elapsed_text = OCRProgressEstimator.format_duration(max(0.0, float(elapsed or 0.0)))
        eta_text = OCRProgressEstimator.format_duration(eta)
        text = f"总进度 {percent:.1f}% · 已用 {elapsed_text} · 预计剩余：{eta_text}"
        self._progress_bar_text.setText(text)
        # Hidden mirror and tooltip keep the same information available to
        # accessibility tools and to narrow-window users.
        self._progress_lbl.setText(text)
        self._prog.setToolTip(text)


    def _refresh_live_progress_clock(self) -> None:
        self = self._tab
        if not self._progress_display_is_enabled():
            return
        state = self._overall_progress_live_state
        if not state or self._progress_bar_text.isHidden():
            return
        delta = max(0.0, time.monotonic() - float(state.get("received_at", 0.0)))
        elapsed = max(0.0, float(state.get("elapsed", 0.0)) + delta)
        eta_base = state.get("eta")
        # Between real work callbacks, show a lightweight countdown from the
        # latest estimator snapshot.  The next callback immediately corrects it.
        eta = None if eta_base is None else max(0.0, float(eta_base) - delta)
        self._set_progress_bar_text(float(state.get("percent", 0.0)), elapsed, eta)


    def _on_overall_progress(self, snapshot):
        self = self._tab
        self._latest_progress_snapshot = snapshot
        if not self._progress_display_is_enabled():
            return
        try:
            percent = max(0.0, min(100.0, float(snapshot.percent)))
            elapsed = max(0.0, float(snapshot.elapsed_seconds))
            eta = snapshot.eta_seconds
            label = str(snapshot.label or "OCR 处理中")
            current = max(0, int(snapshot.current or 0))
            total = max(1, int(snapshot.total or 1))
            unit = str(snapshot.unit or "项")
        except Exception:
            return

        self._overall_progress_live_state = {
            "percent": percent,
            "elapsed": elapsed,
            "eta": eta,
            "received_at": time.monotonic(),
        }
        if not self._progress_clock_timer.isActive():
            self._progress_clock_timer.start()
        self._progress_bar_wrap.setVisible(True)
        self._progress_bar_text.setVisible(True)
        self._phase_progress_lbl.setVisible(True)
        self._prog.setRange(0, 1000)
        self._prog.setValue(int(round(percent * 10.0)))
        self._set_progress_bar_text(percent, elapsed, eta)
        if unit == "内部进度":
            self._phase_progress_lbl.setText(f"当前：{label}")
        elif total == 1 and current in {0, 1}:
            self._phase_progress_lbl.setText(f"当前：{label}")
        else:
            self._phase_progress_lbl.setText(
                f"当前：{label} {current} / {total} {unit}"
            )


    def _reset_run_state(self):
        self = self._tab
        self._flush_ocr_log_buffer(force_all=True)
        self._ocr_log_flush_timer.stop()
        self._ocr_run_active = False
        self._review_preview_run_active = False
        self._ocr_watchdog_timer.stop()
        self._ocr_worker_thread = None
        self._run_btn.setEnabled(True)
        if hasattr(self, "_handwriting_run_btn"):
            self._handwriting_run_btn.setEnabled(True)
        self._pause_btn.setVisible(True)
        self._pause_btn.setEnabled(False)
        self._rerun_btn.setVisible(False)
        self._progress_clock_timer.stop()
        self._overall_progress_live_state = None
        self._latest_progress_snapshot = None
        self._progress_bar_wrap.setVisible(False)
        self._progress_bar_text.setVisible(False)
        self._progress_lbl.setVisible(False)
        self._phase_progress_lbl.setVisible(False)
        self._refresh_review_preview_boxes()


    def _on_multi_ocr_done(self, payload: dict):
        self = self._tab
        self._reset_run_state()
        self._rerun_btn.setText("重新 OCR")
        self._rerun_btn.setVisible(True)
        self._latest_single_doc = None
        self._handwriting_review_context = None
        documents = list(payload.get("documents") or [])
        labels = list(payload.get("labels") or [])
        comparison = payload.get("comparison")
        fused = payload.get("fused")
        summary = getattr(comparison, "summary", "多模型 OCR 对比已完成")
        self._log_view.appendPlainText(f"\n✅ 多模型 OCR 完成：{summary}")
        lines = [
            f"<b>模型数量:</b> {len(documents)}",
            f"<b>模型:</b> {' · '.join(html.escape(label) for label in labels)}",
            f"<b>自动融合:</b> {html.escape(summary)}",
            "",
            "当前全部模型原文均已独立保留；请到左侧“OCR 对比”逐句改选或直接应用自动融合稿。",
        ]
        self._result_view.setHtml("<br>".join(lines))
        self._refresh_ocr_runtime_status(show_dialog=False, deep=False)
        self._emit_run_log_terminal(
            "ok",
            extra_details={
                "models": list(labels),
                "documents": len(documents),
                "rows": len(getattr(comparison, "rows", []) or []),
                "conflicts": int(getattr(comparison, "conflict_rows", 0) or 0),
                "exact_rows": int(getattr(comparison, "exact_rows", 0) or 0),
                "summary": str(summary or "多模型 OCR 对比已完成"),
            },
        )
        self.multi_ocr_done.emit(payload)


    def _on_done(self, doc):
        self = self._tab
        if isinstance(doc, dict) and doc.get("ocr_cancelled"):
            self._handwriting_review_context = None
            self._reset_run_state()
            self._rerun_btn.setText("继续 OCR")
            self._rerun_btn.setToolTip("按当前参数继续；已完成模型阶段与单列/整句缓存会校验后直接复用。")
            self._rerun_btn.setVisible(True)
            self._log_view.appendPlainText(
                "\n⏸ OCR 已停止。当前阶段尚未产生可保留的完整页面结果；"
                "没有继续派发新页面、新列、新模型或重试任务。"
            )
            self._result_view.setHtml(
                "<b>OCR 已停止</b><br>当前阶段尚未产生可保留的完整页面结果。"
            )
            self._emit_run_log_terminal(
                "cancelled",
                extra_details={"summary": "OCR 已停止；没有产生可保留的完整页面结果"},
            )
            return
        if isinstance(doc, dict) and doc.get("multi_ocr"):
            self._on_multi_ocr_done(doc)
            return
        self._reset_run_state()
        self._rerun_btn.setText("继续 OCR" if any(
            any(marker in str(log.get("message", "") or "") for marker in ("暂停", "停止", "取消"))
            for log in doc.processing_log
        ) else "重新 OCR")
        self._rerun_btn.setVisible(True)
        was_cancelled = any(
            any(marker in str(log.get("message", "") or "") for marker in ("暂停", "停止", "取消"))
            for log in doc.processing_log
        )
        review_context = self._handwriting_review_context or {}
        self._handwriting_review_context = None
        audit_before_review = getattr(doc.metadata, "column_ocr_audit", {}) or {}
        totals_before_review = audit_before_review.get("totals", {}) if isinstance(audit_before_review, dict) else {}
        pending_empty_columns = int(totals_before_review.get("pending_manual", 0) or 0)
        need_review = bool(
            review_context.get("enabled")
            and review_context.get("mode") in {"hybrid", "manual"}
        )

        if pending_empty_columns and not need_review:
            self._log_view.appendPlainText(
                f"\n⚠️ 有 {pending_empty_columns} 列 OCR 三次均为空；物理列已保留。"
                "普通 OCR 不会自动打开人工纠错，可单独点击“开始 OCR + 人工纠错”复核。"
            )

        if need_review and not was_cancelled:
            if pending_empty_columns:
                self._log_view.appendPlainText(
                    f"\n⚠️ 有 {pending_empty_columns} 列 OCR 三次均为空；已保留物理列并进入人工输入，任务不会中止。"
                )
            self._log_view.appendPlainText("\n✍️ 打开 OCR 人工纠错工作台：优先定位高风险列…")
            try:
                from adapters.handwriting_trace_review import (
                    OCRManualReviewDialog, apply_review_payload,
                )
                review = OCRManualReviewDialog(
                    self,
                    doc,
                    crop_rect=review_context.get("crop_rect"),
                    mask_main_band=bool(review_context.get("character_mask", True)),
                    recognition_backend=str(review_context.get("backend") or "auto"),
                    strategy=str(review_context.get("strategy") or "balanced"),
                )
                if review.exec() and review.payload is not None:
                    reviewed, changed = apply_review_payload(doc, review.payload)
                    from engine.ocr_manual_review import annotate_ocr_review_risks
                    risk_after = annotate_ocr_review_risks(doc)
                    self._log_view.appendPlainText(
                        f"✍️ 人工纠错完成：明确核对 {reviewed} 列，修改 {changed} 列；"
                        f"仍有 {int(risk_after.get('suspicious_columns', 0) or 0)} 列未确认疑点"
                    )
                else:
                    self._log_view.appendPlainText("✍️ 已跳过人工复核，完整保留普通 OCR 底稿")
            except Exception as exc:
                self._log_view.appendPlainText(f"⚠️ 人工纠错工作台未能打开：{exc}")
                QMessageBox.warning(
                    self, "人工纠错不可用",
                    f"人工纠错工作台未能打开，将继续使用完整的普通 OCR 底稿。\n\n{exc}",
                )

        if need_review and review_context.get("apply_column_reflow"):
            self._log_view.appendPlainText("\n🧩 人工纠错后处理：按列尾标点组句并跨页接续…")
            from engine.column_sentence_reflow import reflow_columns_into_sentences
            doc = reflow_columns_into_sentences(
                doc, max_columns=int(review_context.get("column_reflow_max", 64) or 64)
            )

        status = "⏸ 已停止（部分结果）" if was_cancelled else "✅ OCR 完成"
        chapter_suffix = (
            f"{len(doc.toc)} 章节"
            if doc.toc
            else "正文已输出 · 未检测到章节标题"
        )
        self._log_view.appendPlainText(
            f"\n{status}: {len(doc.blocks)} 块，{chapter_suffix}"
        )

        # 显示结果摘要；精准分列模式额外展示三段式列完整性对账。
        lines = [f"<b>OCR 引擎:</b> {doc.metadata.source_engine}",
                 f"<b>总页数:</b> {len(doc.pages)}  ·  <b>总块数:</b> {len(doc.blocks)}"]
        audit = getattr(doc.metadata, "column_ocr_audit", {}) or {}
        totals = audit.get("totals", {}) if isinstance(audit, dict) else {}
        expected = int(totals.get("expected", 0) or 0)
        if expected:
            recognized = int(totals.get("recognized", 0) or 0)
            text_recognized = int(totals.get("text_recognized", recognized) or 0)
            pending_manual = int(totals.get("pending_manual", 0) or 0)
            model_written = int(totals.get("model_written", 0) or 0)
            integrity = bool(audit.get("model_integrity_passed"))
            last_ok = bool(audit.get("last_page_all_columns_exported"))
            marker = "⚠️" if integrity and last_ok and pending_manual else ("✅" if integrity and last_ok else "❌")
            audit_line = (
                f"{marker} 分列对账：预计 {expected} · 已保全 {recognized} · OCR有字 {text_recognized} · "
                f"待人工 {pending_manual} · 写入文档模型 {model_written} · 最后一页 {'完整' if last_ok else '不完整'}"
            )
            self._log_view.appendPlainText(audit_line)
            lines.extend(["", f"<b>分列完整性:</b> {audit_line}"])
        review_report = getattr(doc.metadata, "ocr_review_report", {}) or {}
        if isinstance(review_report, dict) and int(review_report.get("columns", 0) or 0):
            suspicious = int(review_report.get("suspicious_columns", 0) or 0)
            high_risk = int(review_report.get("high_risk_columns", 0) or 0)
            review_line = f"⚠️ OCR 疑点：待人工确认 {suspicious} 列 · 高风险 {high_risk} 列 · 候选未自动改字"
            self._log_view.appendPlainText(review_line)
            lines.extend(["", f"<b>人工纠错:</b> {review_line}"])
        lines.extend(["", "<b>章节目录:</b>"])
        for e in doc.toc:
            lines.append(f"  {e.chapter_index}. {e.title}")
        if not doc.toc:
            lines.append("  （未识别到章节）")
        self._result_view.setHtml("<br>".join(lines))

        self._refresh_ocr_runtime_status(show_dialog=False, deep=False)
        self._latest_single_doc = doc
        self._emit_run_log_terminal(
            "cancelled" if was_cancelled else "ok",
            extra_details={
                "source_engine": str(getattr(doc.metadata, "source_engine", "") or ""),
                "blocks": len(doc.blocks),
                "chapters": len(doc.toc),
                "pages": len(doc.pages),
                "pending_manual_columns": pending_empty_columns,
                "summary": (
                    f"{'部分结果' if was_cancelled else '完成'} · {len(doc.pages)} 页 · "
                    f"{len(doc.blocks)} 块 · {len(doc.toc)} 章节"
                ),
            },
        )
        self.ocr_done.emit(doc)


    def _on_error(self, msg):
        self = self._tab
        self._handwriting_review_context = None
        self._reset_run_state()
        self._rerun_btn.setText("继续 OCR")
        self._rerun_btn.setToolTip("若项目断点有效，将复用已完成模型阶段和分段识别缓存。")
        self._rerun_btn.setVisible(True)
        self._log_view.appendPlainText(f"\n❌ 错误:\n{msg}")
        self._emit_run_log_terminal(
            "error",
            extra_details={"summary": "OCR 运行失败"},
            error=str(msg or ""),
        )
        show_error_dialog(self, "OCR 失败", msg)


    def _re_ocr(self):
        """复用当前输入、裁剪框和适配器设置，清空日志/结果后重新执行一次 OCR。"""
        self = self._tab
        self._result_view.clear()
        self._result_view.setHtml("")
        self._run_ocr()

