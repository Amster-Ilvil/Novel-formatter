from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QTimer

from ui.common.signals import WorkerSignals
from ui.common.toast import notify
from ui.dialogs import show_error_dialog
from ui.localized_dialogs import LocalizedFileDialog as QFileDialog, LocalizedMessageBox as QMessageBox
from adapters.result_export import safe_result_filename


class OCRCompareExchangeService:
    """Roundtrip/AI package exchange owner for OCRCompareTab.

    The tab remains the Qt presentation/state authority; this service owns the
    import/export orchestration and delegates UI refreshes through the tab's
    stable public/private compatibility methods.
    """

    def __init__(self, tab):
        self._tab = tab

    def _export_single_roundtrip_package(self):
        self = self._tab
        doc = self._single_doc
        if self._mode != "single" or doc is None:
            QMessageBox.warning(self, "没有单 OCR 结果", "请先点击“从 OCR 识别载入”。")
            return
        from engine.ocr_roundtrip_package import export_single_package, save_package
        title = safe_result_filename(str(getattr(doc.metadata, "title", "") or ""), default="single_ocr")
        path, selected_filter = QFileDialog.getSaveFileName(
            self, "导出单 OCR 大模型校对包",
            self._project_package_default(f"{title}_单OCR原格式校对包.json"),
            "JSON 校对包 (*.json);;Markdown 校对包 (*.md)",
        )
        if not path:
            return
        if not Path(path).suffix:
            path += ".md" if "Markdown" in selected_filter else ".json"
        try:
            package = export_single_package(doc)
            written = save_package(package, path)
        except Exception as exc:
            self.package_exported.emit({"kind": "single_roundtrip", "status": "error", "path": str(path), "error": str(exc), "details": {}})
            show_error_dialog(self, "导出单 OCR 校对包失败", str(exc))
            return
        self.package_exported.emit({
            "kind": "single_roundtrip",
            "status": "ok",
            "path": str(written),
            "details": {"editable_items": len(package.get("editable_items", []))},
        })
        notify(
            self,
            f"已导出 {len(package.get('editable_items', []))} 个可校对正文块：\n{written}\n\n"
            "只修改 edited_text 或 Markdown 校对结果区，不要改动 ID 和结构清单。",
            "success",
        )

    def _import_single_roundtrip_package(self):
        self = self._tab
        from engine.ocr_roundtrip_package import import_single_package, load_package
        path, _ = QFileDialog.getOpenFileName(
            self, "导入单 OCR 大模型校对包", "", "OCR 校对包 (*.json *.md *.markdown)",
        )
        if not path:
            return
        current = self._single_doc or self._available_single_doc or self._primary_doc
        try:
            package = load_package(path)
            imported = import_single_package(package, current_document=current)
        except Exception as exc:
            show_error_dialog(self, "导入单 OCR 校对包失败", str(exc))
            return
        self._available_single_doc = imported
        self._available_single_label = "外部 AI 校对后的单 OCR"
        self._available_single_generation += 1
        self._load_single_result(
            imported, self._available_single_label, self._available_single_generation
        )
        self.single_doc_applied.emit(imported)
        notify(
            self,
            "修改文字已按 block ID 完整覆盖并进入 Formatter、OCR 对比、图文对照和 EPUB。\n"
            "同一来源被更新，没有新增重复副本，也没有执行全文查找替换。",
            "success",
        )

    def _export_multi_roundtrip_package(self):
        self = self._tab
        if self._comparison is None or len(self._documents) < 2:
            QMessageBox.warning(self, "没有多模型结果", "请先完成多模型 OCR。")
            return
        if self._sources_dirty and not self._alignment_counts_valid(show_warning=False):
            notify(self, "各 OCR 栏的行数已改变。请先点击“重新对齐”，再导出大模型对比包。", "warning")
            return
        comparison = self._comparison_with_current_source_texts()
        if comparison is None:
            QMessageBox.warning(self, "无法导出", "当前 OCR 对齐状态不完整。")
            return
        result_lines = []
        delete_flags = []
        for row_index, state in enumerate(self._fusion_states):
            value = state.output_text()
            if state.selected_index is None and row_index < len(comparison.rows):
                value = comparison.rows[row_index].output_text
            result_lines.append(str(value or ""))
            delete_flags.append(state.output_delete_intentionally())
        from engine.ocr_roundtrip_package import export_multi_package, save_package
        title = safe_result_filename(
            str(getattr(getattr(self._primary_doc, "metadata", None), "title", "") or ""),
            default="multi_ocr",
        )
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "导出多模型 OCR 大模型对比包",
            self._project_package_default(f"{title}_多模型OCR对比融合包.json"),
            "JSON 校对包 (*.json);;Markdown 校对包 (*.md)",
        )
        if not path:
            return
        if not Path(path).suffix:
            path += ".md" if "Markdown" in selected_filter else ".json"
        try:
            package = export_multi_package(
                self._documents,
                self._labels,
                comparison,
                result_lines=result_lines,
                delete_flags=delete_flags,
                ruby_overlay_source=self._ruby_overlay_doc,
            )
            written = save_package(package, path)
        except Exception as exc:
            self.package_exported.emit({"kind": "multi_roundtrip", "status": "error", "path": str(path), "error": str(exc), "details": {}})
            show_error_dialog(self, "导出多模型 AI 对比包失败", str(exc))
            return
        self.package_exported.emit({
            "kind": "multi_roundtrip",
            "status": "ok",
            "path": str(written),
            "details": {
                "model_count": len(package.get("model_sources", [])),
                "editable_items": len(package.get("editable_items", [])),
            },
        })
        notify(
            self,
            f"已导出 {len(package.get('model_sources', []))} 个 OCR 模型、"
            f"{len(package.get('editable_items', []))} 个对齐句组：\n{written}\n\n"
            "交给大模型时只修改 edited_text 或 Markdown 的“大模型校对结果”区域。",
            "success",
        )

    def _set_ai_import_busy(self, busy: bool, message: str = "") -> None:
        self = self._tab
        self._ai_import_busy = bool(busy)
        multi_ready = self._mode == "multi" and self._comparison is not None
        enabled = multi_ready and not self._ai_import_busy and not self._source_correction_busy
        for button in (
            self._auto_btn, self._realign_btn, self._export_texts_btn,
            self._export_ai_package_btn, self._export_ai_repair_epub_btn, self._import_ai_repair_result_btn,
            self._export_source_correction_btn, self._import_source_correction_btn, self._export_fusion_skeleton_btn,
            self._ai_adjudicate_btn, self._import_ai_package_btn, self._apply_btn,
        ):
            button.setEnabled(enabled)
        self._restore_btn.setEnabled(enabled and self._initial_payload is not None)
        self._single_import_btn.setEnabled(not self._ai_import_busy)
        running_adjudication = self._ai_adjudication_cancel_event is not None
        self._ai_adjudication_cancel_btn.setVisible(running_adjudication)
        self._ai_adjudication_cancel_btn.setEnabled(running_adjudication)
        self._ai_adjudication_progress.setVisible(running_adjudication)
        if busy:
            if running_adjudication:
                self._set_compare_task_progress(
                    True, value=0, text="准备 AI 裁决…", indeterminate=False
                )
            else:
                self._set_compare_task_progress(
                    True, text=message or "正在导入 AI 结果…", indeterminate=True
                )
        elif not self._source_correction_busy:
            self._set_compare_task_progress(False)
        workspace_enabled = not self._ai_import_busy and not self._source_correction_lock_workspace
        self._source_area.setEnabled(workspace_enabled)
        self._result_panel.setEnabled(workspace_enabled)
        if message:
            self._summary.setText(message)

    def _import_multi_roundtrip_package(self):
        self = self._tab
        if self._primary_doc is None:
            QMessageBox.warning(self, "没有结构底稿", "请先载入当前书的多模型 OCR 结果。")
            return
        if self._ai_import_busy or self._source_correction_busy:
            notify(self, "当前已有多模型 AI 或逐源纠错任务，请稍候。", "warning")
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入多模型 OCR 大模型融合包",
            "",
            "OCR 校对包 (*.json *.md *.markdown)",
        )
        if not path:
            return

        token = self._ai_import_generation.begin()
        signals = WorkerSignals()
        self._ai_import_signal_refs[token] = signals
        signals.log.connect(lambda text, t=token: self._on_ai_import_progress(t, text))
        signals.error.connect(lambda text, t=token: self._on_ai_import_error(t, text))
        signals.finished.connect(lambda result, t=token: self._on_ai_import_ready(t, result))
        current_primary = self._primary_doc
        source_path = str(path)
        self._set_ai_import_busy(
            True,
            "正在后台读取并校验多模型 AI 修订包；界面不会再因 20MB 以上 JSON 解析而无响应…",
        )

        def worker():
            try:
                from engine.ocr_roundtrip_package import import_multi_package, load_package
                from engine.ocr_compare_view_model import (
                    build_fusion_states,
                    upsert_external_candidate,
                )

                signals.log.emit("正在后台读取 JSON／Markdown…")
                package = load_package(source_path)
                signals.log.emit("正在校验 row ID、结构哈希、列 ID 和不可变清单…")
                fused, imported_comparison, lines = import_multi_package(
                    package,
                    current_primary=current_primary,
                )
                signals.log.emit("正在后台建立轻量候选状态…")
                fusion_states = build_fusion_states(
                    imported_comparison.rows,
                    auto_choose=False,
                )
                if len(lines) == len(fusion_states):
                    package_items = list(package.get("editable_items") or [])
                    for row_index, (state, text) in enumerate(zip(fusion_states, lines)):
                        delete_intentionally = (
                            package_items[row_index].get("delete_intentionally", False) is True
                        ) if row_index < len(package_items) else False
                        upsert_external_candidate(
                            state,
                            str(text or ""),
                            display_label="外部AI融合结果",
                            select=True,
                            allow_empty=delete_intentionally,
                            selection_origin="external_ai_package",
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
                    "path": source_path,
                })
            except Exception as exc:
                signals.error.emit(str(exc))

        threading.Thread(
            target=worker,
            name=f"multi-ai-package-import-{token}",
            daemon=True,
        ).start()

    def _on_ai_import_progress(self, token: int, text: str) -> None:
        self = self._tab
        if not self._ai_import_generation.is_current(token):
            return
        message = str(text or "正在导入多模型 AI 包…")
        self._summary.setText(message)
        self._set_compare_task_progress(True, text=message, indeterminate=True)

    def _on_ai_import_error(self, token: int, message: str) -> None:
        self = self._tab
        self._ai_import_signal_refs.pop(token, None)
        if not self._ai_import_generation.is_current(token):
            return
        self._set_ai_import_busy(False)
        show_error_dialog(self, "导入多模型 AI 融合包失败", str(message or "未知错误"))

    def _on_ai_import_ready(self, token: int, result: object) -> None:
        self = self._tab
        self._ai_import_signal_refs.pop(token, None)
        if not self._ai_import_generation.is_current(token):
            return
        if not isinstance(result, dict):
            self._on_ai_import_error(token, "后台导入没有返回有效结果。")
            return
        imported_comparison = result.get("comparison")
        lines = list(result.get("lines") or [])
        if imported_comparison is None or not lines:
            self._on_ai_import_error(token, "AI 修订包没有可应用的逐句结果。")
            return

        restore_comparison = (
            len(getattr(imported_comparison, "labels", []) or []) == len(self._documents)
        )
        self._pending_ai_import_apply = {
            "token": token,
            "result": result,
            "restore_comparison": restore_comparison,
            "editor_index": 0,
        }
        if restore_comparison:
            self._comparison = imported_comparison
            self._labels = list(imported_comparison.labels)
            imported_source_texts = [str(value or "") for value in (result.get("source_texts") or [])]
            while len(imported_source_texts) < len(self._documents):
                imported_source_texts.append("")
            self._set_full_source_texts(imported_source_texts)
        self._summary.setText("校验完成，正在分批刷新 OCR 对比界面…")
        QTimer.singleShot(0, self._apply_next_ai_import_editor)

    def _apply_next_ai_import_editor(self) -> None:
        self = self._tab
        pending = self._pending_ai_import_apply
        if not isinstance(pending, dict):
            return
        token = int(pending.get("token", -1))
        if not self._ai_import_generation.is_current(token):
            self._pending_ai_import_apply = None
            return
        if not pending.get("restore_comparison"):
            self._complete_ai_import_apply()
            return

        index = int(pending.get("editor_index", 0) or 0)
        if index >= len(self._documents):
            self._complete_ai_import_apply()
            return
        pending["editor_index"] = index + 1
        editor = self._source_editors[index]
        target_text = (
            self._source_line_text(index, self._current_row())
            if self._single_card_enabled
            else self._source_text(index)
        )
        # The common case (same exported package, only edited_text changed)
        # avoids rebuilding QTextDocument entirely.
        if editor.toPlainText() != target_text:
            self._loading_text = True
            editor.setUpdatesEnabled(False)
            try:
                editor.setReadOnly(self._single_card_enabled)
                editor.setPlainText(target_text)
            finally:
                editor.setUpdatesEnabled(True)
                self._loading_text = False
        self._summary.setText(
            f"正在刷新模型栏 {index + 1}/{len(self._documents)}；其余步骤继续在事件循环间隙执行…"
        )
        QTimer.singleShot(0, self._apply_next_ai_import_editor)

    def _complete_ai_import_apply(self) -> None:
        self = self._tab
        pending = self._pending_ai_import_apply
        self._pending_ai_import_apply = None
        if not isinstance(pending, dict):
            return
        token = int(pending.get("token", -1))
        if not self._ai_import_generation.is_current(token):
            return
        result = pending.get("result") or {}
        imported_comparison = result.get("comparison")
        lines = list(result.get("lines") or [])
        restore_comparison = bool(pending.get("restore_comparison"))

        if restore_comparison:
            self._fusion_states = list(result.get("fusion_states") or [])
            self._sources_dirty = False
            self._set_source_labels_for_mode()
        elif len(lines) == len(self._fusion_states):
            from engine.ocr_compare_view_model import upsert_external_candidate
            for state, text in zip(self._fusion_states, lines):
                upsert_external_candidate(
                    state,
                    str(text or ""),
                    display_label="外部AI融合结果",
                    select=True,
                    selection_origin="external_ai_package",
                )

        current = min(self._current_row(), max(0, len(self._fusion_states) - 1))
        self._clear_fusion_widgets()
        self._refresh_source_highlights()
        self._sync_after_bulk_resolution(current)
        checkpoint_signal = getattr(self, "adjudication_checkpoint_requested", None)
        if checkpoint_signal is not None:
            checkpoint_signal.emit()

        fused = result.get("fused")
        repair_count = int(getattr(imported_comparison, "alignment_shift_repairs", 0) or 0)
        empty_count = int(getattr(imported_comparison, "unresolved_empty_cells", 0) or 0)
        repair_note = f"；已自动复位旧版相邻句错位 {repair_count} 处" if repair_count else ""
        empty_note = f"；仍有 {empty_count} 处单模型物理列为空，已标为低置信复核" if empty_count else ""
        adjudication_report = result.get("ai_adjudication_report")
        adjudication_paths = result.get("ai_adjudication_paths")
        self._set_ai_import_busy(False)
        if isinstance(adjudication_report, dict):
            stats = adjudication_report.get("stats") or {}
            schema = str(adjudication_report.get("schema", "") or "")
            is_gpt_grade = schema == "novel_formatter.ocr_gpt_grade_adjudication.v1"
            if is_gpt_grade:
                visual_report = adjudication_report.get("visual") if isinstance(adjudication_report.get("visual"), dict) else {}
                contextual_report = adjudication_report.get("contextual") if isinstance(adjudication_report.get("contextual"), dict) else {}
                visual_stats = visual_report.get("stats") if isinstance(visual_report.get("stats"), dict) else {}
                contextual_stats = contextual_report.get("stats") if isinstance(contextual_report.get("stats"), dict) else {}
                visual_usage = visual_report.get("usage") if isinstance(visual_report.get("usage"), dict) else {}
                contextual_usage = contextual_report.get("usage") if isinstance(contextual_report.get("usage"), dict) else {}
                targets = int(stats.get("target_items", 0) or 0)
                visual_applied = int(stats.get("visual_applied_changes", 0) or 0)
                contextual_applied = int(stats.get("contextual_applied_changes", 0) or 0)
                uncertain = int(stats.get("final_uncertain_items", 0) or 0)
                state_stats = dict(result.get("ai_adjudication_state_stats") or {})
                confirmed_unchanged = int(state_stats.get("confirmed_unchanged", 0) or 0)
                visual_requests = int(stats.get("visual_request_count", 0) or 0)
                contextual_batches = int(stats.get("contextual_request_batches", 0) or 0)
                peak_concurrency = int(visual_stats.get("concurrency_peak", 0) or 0)
                token_total = int(visual_usage.get("total_tokens", 0) or 0) + int(contextual_usage.get("total_tokens", 0) or 0)
                token_note = f"；API 用量 {token_total / 1000:.1f}k tokens" if token_total else ""
                context_error = str(adjudication_report.get("contextual_error", "") or "")
                pass_note = "出版级·视觉抄写→上下文裁决→独立审计" if contextual_report else "出版级·视觉阶段（上下文阶段未完成）"
                speed_parts = []
                if visual_requests:
                    speed_parts.append(f"视觉 {visual_requests} 次 API")
                if contextual_batches:
                    speed_parts.append(f"上下文 {contextual_batches} 批")
                if peak_concurrency:
                    speed_parts.append(f"视觉峰值并发 {peak_concurrency}")
                speed_note = f"；{'，'.join(speed_parts)}" if speed_parts else ""
                error_note = "；上下文阶段失败，已保留通过安全闸门的视觉结果" if context_error else ""
                self._summary.setText(
                    f"✅ 出版级 AI 裁决完成：风险条目 {targets}，视觉阶段采用 {visual_applied}，"
                    f"上下文二次改判 {contextual_applied}，AI确认原文无需修改 {confirmed_unchanged}，"
                    f"最终保留人工复核 {uncertain}；{pass_note}{speed_note}{token_note}{error_note}。"
                    "原始 OCR、物理列与结构未改变。"
                )
            else:
                usage = adjudication_report.get("usage") or {}
                applied = int(stats.get("applied_changes", 0) or 0)
                uncertain = int(stats.get("uncertain_items", 0) or 0)
                targets = int(stats.get("target_items", 0) or 0)
                skipped = int(stats.get("skipped_low_risk", 0) or 0)
                token_total = int(usage.get("total_tokens", 0) or 0)
                token_note = f"；API 用量 {token_total / 1000:.1f}k tokens" if token_total else ""
                mode = str(adjudication_report.get("mode", "") or "")
                routing_mode = str(adjudication_report.get("routing_mode", "smart") or "smart")
                routing_note = {"smart":"智能均衡", "exhaustive":"出版全部", "lean":"极速高风险"}.get(routing_mode, routing_mode)
                requests = int(stats.get("request_count", 0) or 0)
                peak_concurrency = int(stats.get("concurrency_peak", 0) or 0)
                mode_note = "出版参考真值模式" if mode == "reference_truth" else f"批量视觉裁决·{routing_note}"
                speed_note = f"；{requests} 次 API，请求峰值并发 {peak_concurrency}" if requests else ""
                self._summary.setText(
                    f"✅ AI 审定完成：风险条目 {targets}，自动采用 {applied}，保留人工复核 {uncertain}，"
                    f"跳过低风险项 {skipped}；{mode_note}{speed_note}{token_note}。结构与物理列未改变。"
                )
        else:
            self._summary.setText(
                f"✅ 已严格导入外部大模型融合包：{len(lines)} 句均按 row ID 完整覆盖；"
                f"原始多模型候选、封面、插图、目录、坐标和列 ID 保持不变{repair_note}{empty_note}。"
            )
        if fused is not None:
            self.doc_applied.emit(fused)

        if isinstance(adjudication_report, dict):
            stats = adjudication_report.get("stats") or {}
            paths = list(adjudication_paths or []) if isinstance(adjudication_paths, (list, tuple)) else []
            schema = str(adjudication_report.get("schema", "") or "")
            if schema == "novel_formatter.ocr_gpt_grade_adjudication.v1":
                visual = adjudication_report.get("visual") if isinstance(adjudication_report.get("visual"), dict) else {}
                contextual = adjudication_report.get("contextual") if isinstance(adjudication_report.get("contextual"), dict) else {}
                visual_stats = visual.get("stats") if isinstance(visual.get("stats"), dict) else {}
                context_error = str(adjudication_report.get("contextual_error", "") or "")
                detail_lines = [
                    f"出版级链路处理 {int(stats.get('target_items', 0) or 0)} 个真实分歧：先独立视觉抄写，再做候选/物理列/前后文裁决，最后独立审计。",
                    f"视觉阶段采用 {int(stats.get('visual_applied_changes', 0) or 0)} 处；上下文阶段二次改判 {int(stats.get('contextual_applied_changes', 0) or 0)} 处；最终 {int(stats.get('final_uncertain_items', 0) or 0)} 处保留人工复核。",
                    f"视觉 API {int(stats.get('visual_request_count', 0) or 0)} 次，峰值并发 {int(visual_stats.get('concurrency_peak', 0) or 0)}；上下文请求批次 {int(stats.get('contextual_request_batches', 0) or 0)}；独立审计失败 {int(stats.get('contextual_audit_failures', 0) or 0)} 处。",
                    "第二阶段只允许输出稀疏 edited_text；原始 OCR 候选只读，row ID、block ID、页码、坐标、物理列、封面、插图、目录和 EPUB 结构均由程序锁定。",
                    "默认未读取文库电子版作为答案：参考电子版仍用于事后 benchmark，避免把参考答案泄漏给裁决模型。",
                ]
                if context_error:
                    detail_lines.append(f"上下文阶段未完成：{context_error}；已保留通过本地安全闸门的视觉裁决，不把失败批次伪装成出版级完整结果。")
                elif contextual:
                    detail_lines.append("上下文裁决与独立审计均已完成；仍未通过本地安全闸门的条目不会自动写回。")
            else:
                detail_lines = [
                    f"AI 仅处理了 {int(stats.get('target_items', 0) or 0)} 个路由后风险条目；"
                    f"{int(stats.get('skipped_low_risk', 0) or 0)} 个低风险条目未发送，节省 tokens。",
                    f"API 请求 {int(stats.get('request_count', 0) or 0)} 次；并发 {int(stats.get('concurrency_start', 0) or 0)}→峰值 {int(stats.get('concurrency_peak', 0) or 0)}→结束 {int(stats.get('concurrency_end', 0) or 0)}；"
                    f"漏回缩批补跑 {int(stats.get('incomplete_retry_batches', 0) or 0)} 次。",
                    f"自动写入 {int(stats.get('applied_changes', 0) or 0)} 处；"
                    f"{int(stats.get('uncertain_items', 0) or 0)} 处证据不足或审计失败，仍保留原融合稿等待人工复核。",
                    "校订模型只返回稀疏 edited_text；独立审计不读取第一遍解释，且所有结果均通过本地长度、语言和来源覆盖检查。",
                    "row ID、block ID、页码、坐标、物理列、封面、插图、目录和 EPUB 结构均由程序锁定。",
                ]
            if paths:
                detail_lines.append(f"审定融合包：{paths[0]}")
                if len(paths) > 1:
                    detail_lines.append(f"审计报告：{paths[1]}")
            if adjudication_report.get("mode") != "reference_truth" and schema != "novel_formatter.ocr_gpt_grade_adjudication.v1":
                detail_lines.append("未提供出版版真值：本结果只称为高精度融合候选，不声称出版级或 100% 准确。")
            notify(self, "\n".join(detail_lines), "success")
        else:
            detail_lines = [
                "大模型结果已完全替换融合正文并进入 Formatter、OCR 对比和 EPUB。",
                "JSON 读取、结构校验和候选状态构建均在后台完成；模型栏按事件循环分批刷新。",
                "导入按 row ID 写回，不执行全文替换，因此不会产生重复替换。",
            ]
            if repair_count:
                detail_lines.append(f"检测并修复了 {repair_count} 处旧版相邻句错位。")
            if empty_count:
                detail_lines.append(f"仍有 {empty_count} 处仅一个模型识别到文字，已保留并标记为低置信候选。")
            notify(self, "\n".join(detail_lines), "success")

    def _export_separate_texts(self):
        self = self._tab
        if self._comparison is None or not self._documents:
            QMessageBox.warning(self, "没有结果", "请先运行多模型 OCR。")
            return
        target_dir = QFileDialog.getExistingDirectory(
            self, "选择分别导出 OCR 文本的文件夹", self._project_package_dir
        )
        if not target_dir:
            return
        folder = Path(target_dir)
        title = safe_result_filename(
            str(getattr(getattr(self._primary_doc, "metadata", None), "title", "") or ""),
            default="ocr_compare",
        )
        written: list[Path] = []
        try:
            self._sync_full_source_cache_from_editors()
            for index, editor in enumerate(self._source_editors[:len(self._documents)]):
                label = self._labels[index] if index < len(self._labels) else f"模型{index + 1}"
                label_stem = safe_result_filename(label, default=f"模型{index + 1}")
                path = folder / f"{title}_{index + 1:02d}_{label_stem}.txt"
                text = self._source_text(index).replace("\r\n", "\n").replace("\r", "\n")
                if text and not text.endswith("\n"):
                    text += "\n"
                self._atomic_write_utf8(path, text)
                written.append(path)

            unresolved = [state for state in self._fusion_states if state.unresolved]
            if not unresolved and not self._sources_dirty and self._alignment_counts_valid(show_warning=False):
                fused_text = "\n".join(state.output_text() for state in self._fusion_states)
                if fused_text and not fused_text.endswith("\n"):
                    fused_text += "\n"
                fused_path = folder / f"{title}_融合结果.txt"
                self._atomic_write_utf8(fused_path, fused_text)
                written.append(fused_path)

        except Exception as exc:
            show_error_dialog(self, "分别导出文本失败", str(exc))
            return

        note = ""
        if any(state.unresolved for state in self._fusion_states):
            note = "\n融合稿仍有未选择候选，本次只导出了各 OCR 原文。"
        elif self._sources_dirty:
            note = "\n源 OCR 尚未重新对齐，本次只导出了各 OCR 原文。"
        notify(
            self,
            f"已导出 {len(written)} 个 UTF-8 文本文件到：\n{folder}{note}",
            "success",
        )

