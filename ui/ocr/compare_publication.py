from __future__ import annotations

import copy
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QCheckBox, QDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout

from ui.common.editor_controls import NoWheelComboBox
from ui.common.toast import notify
from ui.dialogs import show_error_dialog
from ui.localized_dialogs import LocalizedFileDialog as QFileDialog, LocalizedMessageBox as QMessageBox


class OCRComparePublicationService:
    """Own external AI publication/repair package IO for OCRCompareTab.

    The tab remains authoritative for fusion state and canonical export blockers;
    this service only owns package preparation dialogs, publication bundle export,
    and strict external repair-result import.
    """

    def __init__(self, tab):
        self._tab = tab

    def _current_multi_package_for_external_repair(self, *, allow_unresolved: bool = False):
        self = self._tab
        """Create the current sealed package without changing active selections."""
        if self._comparison is None or len(self._documents) < 2 or self._primary_doc is None:
            raise ValueError("请先完成并载入多模型 OCR。")
        if self._sources_dirty and not self._alignment_counts_valid(show_warning=False):
            raise ValueError("各 OCR 栏的行数已改变，请先点击“重新对齐”。")
        comparison = self._comparison_with_current_source_texts()
        if comparison is None:
            raise ValueError("当前 OCR 对齐状态不完整。")
        blockers = self._canonical_export_blockers(comparison, allow_unresolved=allow_unresolved)
        if blockers:
            preview = "\n".join(f"• {value}" for value in blockers[:12])
            more = f"\n另有 {len(blockers) - 12} 项。" if len(blockers) > 12 else ""
            first = blockers[0]
            raise ValueError(
                "当前权威正文尚未通过导出前致命检查：\n" + preview + more
                + "\n请在 OCR 对比/图文对照中完成判断后再导出。"
                + f"\n首项阻断：{first}"
            )
        result_lines: list[str] = []
        delete_flags: list[bool] = []
        for row_index, state in enumerate(self._fusion_states):
            value = state.output_text()
            if state.selected_index is None and row_index < len(comparison.rows):
                value = comparison.rows[row_index].output_text
            result_lines.append(str(value or ""))
            delete_flags.append(state.output_delete_intentionally())
        from engine.ocr_roundtrip_package import export_multi_package
        package = export_multi_package(
            self._documents,
            self._labels,
            comparison,
            result_lines=result_lines,
            delete_flags=delete_flags,
            ruby_overlay_source=self._ruby_overlay_doc,
        )
        # AI 修复包只使用当前多模型 OCR、页面结构和原始出版图片。
        # 出版参考 EPUB 属于旧版编辑流程的独立功能，不得被静默附加到
        # OCR 修复包，也不得成为导出的前置条件。
        package["canonical_text_authority"] = {
            "schema": "novel_formatter.canonical_text_authority.v1",
            "raw_ocr_documents_immutable": True,
            "decision_count": len(self._canonical_source_decisions),
            "decisions": [dict(item) for item in self._canonical_source_decisions.values()],
            "derivative_text_rebuilt_from_current_fusion_states": True,
        }
        primary_meta = getattr(getattr(self, "_primary_doc", None), "metadata", None)
        primary_raw = getattr(primary_meta, "__dict__", {}) if primary_meta is not None else {}
        if isinstance(primary_raw, dict):
            if isinstance(primary_raw.get("multi_ocr_role_plan"), dict):
                package["multi_ocr_role_plan"] = copy.deepcopy(primary_raw.get("multi_ocr_role_plan"))
            if primary_raw.get("multi_ocr_roles_executed"):
                package["multi_ocr_roles_executed"] = list(primary_raw.get("multi_ocr_roles_executed") or [])
            if primary_raw.get("multi_ocr_role_schema"):
                package["multi_ocr_role_schema"] = int(primary_raw.get("multi_ocr_role_schema") or 0)
        package.pop("publication_reference", None)
        return package, comparison
    def _choose_ai_repair_reference_option(self, package: dict):
        self = self._tab
        """Choose package profile and optional publication evidence explicitly."""
        dialog = QDialog(self)
        dialog.setWindowTitle("AI 修复包导出选项")
        dialog.setModal(True)
        layout = QVBoxLayout(dialog)
        note = QLabel(
            "当前开发版只导出 V5 多模型分歧裁决包。真正分歧交给外部 AI，"
            "本地已完成裁决会冻结；普通两模型共同候选默认继续 Lean 保留。"
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("AI 修复包模式："))
        mode_combo = NoWheelComboBox()
        mode_combo.addItem("V5 多模型分歧裁决包", "disagreement_v5")
        mode_combo.setEnabled(False)
        mode_row.addWidget(mode_combo, 1)
        layout.addLayout(mode_row)
        estimate_label = QLabel()
        estimate_label.setWordWrap(True)
        layout.addWidget(estimate_label)

        include = QCheckBox("包含出版参考证据（可选，默认关闭）")
        layout.addWidget(include)
        path_row = QHBoxLayout()
        path_edit = QLineEdit()
        path_edit.setReadOnly(True)
        path_edit.setPlaceholderText("未选择出版参考 EPUB")
        browse = QPushButton("选择 EPUB…")
        path_row.addWidget(path_edit, 1)
        path_row.addWidget(browse)
        layout.addLayout(path_row)
        browse.setEnabled(False)
        path_edit.setEnabled(False)
        include.toggled.connect(browse.setEnabled)
        include.toggled.connect(path_edit.setEnabled)

        def choose_file():
            path, _ = QFileDialog.getOpenFileName(dialog, "选择出版参考 EPUB", "", "EPUB 电子书 (*.epub)")
            if path:
                path_edit.setText(path)
        browse.clicked.connect(choose_file)

        estimate_cache = {}
        def update_estimate():
            package_mode = str(mode_combo.currentData() or "disagreement_v5")
            try:
                if package_mode not in estimate_cache:
                    from engine.ai_publication_bundle_v2 import estimate_export_size
                    estimate_cache[package_mode] = estimate_export_size(package, package_mode=package_mode)
                estimate = estimate_cache[package_mode]
                estimate_label.setText(
                    f"预计导出：约 {estimate.get('estimated_zip_size_mb', 0)} MB；"
                    f"文件约 {estimate.get('estimated_file_count', 0)} 个；"
                    f"视觉复核页约 {estimate.get('visual_page_count', 0)} 页。"
                    "实际结果取决于页面几何核验和图片压缩率。"
                )
            except Exception as exc:
                estimate_label.setText(f"暂时无法预测体积：{exc}")
        mode_combo.currentIndexChanged.connect(update_estimate)
        update_estimate()

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton("取消")
        confirm = QPushButton("开始导出")
        buttons.addWidget(cancel)
        buttons.addWidget(confirm)
        layout.addLayout(buttons)
        cancel.clicked.connect(dialog.reject)

        def accept_dialog():
            if include.isChecked() and not path_edit.text().strip():
                QMessageBox.warning(dialog, "尚未选择 EPUB", "请先选择出版参考 EPUB，或取消勾选该选项。")
                return
            dialog.accept()
        confirm.clicked.connect(accept_dialog)
        if dialog.exec() != QDialog.Accepted:
            return None
        return (
            bool(include.isChecked()),
            path_edit.text().strip() or None,
            str(mode_combo.currentData() or "disagreement_v5"),
        )
    def _export_ai_repair_epub(self):
        self = self._tab
        # V5 is the only supported multi-model adjudication package during development.
        if self._comparison is None or self._primary_doc is None or len(self._documents) < 2:
            QMessageBox.warning(self, "没有多模型结果", "请先完成多模型 OCR。")
            return
        mode = "one_pass"
        # V5 multi-model adjudication may retain unresolved rows for the external model.
        try:
            preliminary_package, _comparison = self._current_multi_package_for_external_repair(allow_unresolved=True)
        except Exception as exc:
            show_error_dialog(self, "无法准备 AI 修复包", str(exc))
            return
        reference_option = self._choose_ai_repair_reference_option(preliminary_package)
        if reference_option is None:
            return
        include_reference, reference_path, package_mode = reference_option
        try:
            package, _comparison = self._current_multi_package_for_external_repair(
                allow_unresolved=(package_mode == "disagreement_v5")
            )
        except Exception as exc:
            show_error_dialog(self, "无法准备 AI 修复包", str(exc))
            return
        if package_mode == "disagreement_v5":
            # Freeze every authoritative local retry verdict so external AI
            # cannot overwrite a same-model retry-confirmed strict majority.
            from engine.targeted_retry_adjudicator import build_local_adjudication_freeze_map
            local_map = build_local_adjudication_freeze_map(
                package, self._canonical_source_decisions.values()
            )
            package["local_adjudication"] = local_map
            package["local_adjudication_policy"] = {
                "resolved_rows_are_frozen_for_external_adjudication": True,
                "unresolved_rows_are_exportable_for_llm": True,
                "ruby_is_excluded": True,
                "same_model_retry_never_counts_as_independent_vote": True,
                "targeted_retry_may_confirm_preexisting_strict_majority": True,
            }
            retry_report = self._targeted_retry_local_report
            if retry_report is not None:
                if isinstance(retry_report, dict):
                    package["targeted_retry_adjudication"] = copy.deepcopy(retry_report)
                elif hasattr(retry_report, "to_dict"):
                    package["targeted_retry_adjudication"] = retry_report.to_dict()
        parent = QFileDialog.getExistingDirectory(
            self,
            "选择 AI 修复包的保存位置",
            str(Path.home()),
        )
        if not parent:
            return
        title = "AI修复包"
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            from engine.ai_repair_epub import export_ai_publication_bundle
            report = export_ai_publication_bundle(
                self._primary_doc,
                package,
                parent,
                mode=mode,
                vertical=True,
                bundle_name=title,
                create_zip=True,
                include_publication_reference=include_reference,
                publication_reference_path=reference_path,
                package_mode=package_mode,
            )
        except Exception as exc:
            self.package_exported.emit({"kind": "ai_repair_epub", "status": "error", "path": str(parent), "error": str(exc), "details": {"package_mode": package_mode}})
            show_error_dialog(self, "导出 AI 修复包失败", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        mode_value = str(report.get("package_mode", "") or "")
        mode_label = "V5 多模型分歧裁决包" if mode_value == "disagreement_v5" else (mode_value or "AI 修复包")
        zip_path = Path(str(report.get("zip_path", "") or ""))
        zip_mb = round(zip_path.stat().st_size / (1024 * 1024), 1) if zip_path.is_file() else 0
        self.package_exported.emit({
            "kind": "ai_repair_epub",
            "status": "ok",
            "path": str(report.get("zip_path", "") or report.get("folder", "") or ""),
            "details": {
                "package_mode": mode_value,
                "stable_item_count": int(report.get("stable_item_count", 0) or 0),
                "locked_consensus_count": int(report.get("locked_consensus_count", 0) or 0),
                "review_required_count": int(report.get("review_required_count", 0) or 0),
                "visual_evidence_page_count": int(report.get("visual_evidence_page_count", 0) or 0),
                "zip_mb": zip_mb,
            },
        })
        self._summary.setText(
            f"已导出 {mode_label}：{report.get('stable_item_count', 0)} 个正文条目，"
            f"锁定 {report.get('locked_consensus_count', 0)} 条、需复核 {report.get('review_required_count', 0)} 条，"
            f"视觉复核 {report.get('visual_evidence_page_count', 0)} 页，ZIP {zip_mb} MB。"
        )
        notify(
            self,
            f"已导出 {mode_label}：正文 {report.get('editable_count', 0)} 条 · "
            f"需复核 {report.get('review_required_count', 0)} 条 · ZIP {zip_mb} MB\n"
            f"{report.get('zip_path', '')}",
            "success",
        )
    def _import_ai_repair_result(self):
        self = self._tab
        if self._comparison is None or self._primary_doc is None or len(self._documents) < 2:
            QMessageBox.warning(self, "没有多模型结果", "请先载入与修复文件对应的多模型 OCR。")
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入外部 AI 修复结果（推荐稀疏 JSON）",
            "",
            "当前 V5 裁决结果 (*.json *.zip);;JSON (*.json);;ZIP (*.zip)",
        )
        if not path:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # Current V5 exports intentionally contain unresolved rows; validate
            # the returned decisions against the same unresolved baseline.
            package, _comparison = self._current_multi_package_for_external_repair(allow_unresolved=True)
            from engine.ai_repair_epub import load_ai_repair_result
            updates, report = load_ai_repair_result(path, expected_package=package)
            from engine.ocr_compare_view_model import upsert_external_candidate
            row_to_index = {
                str(item.get("row_id") or item.get("item_id") or ""): index
                for index, item in enumerate(package.get("editable_items") or [])
            }
            applied = 0
            review_only = 0
            for update in updates:
                row_index = row_to_index.get(str(update.get("item_id", "") or ""))
                if row_index is None or row_index >= len(self._fusion_states):
                    continue
                needs_review = bool(update.get("needs_review", False))
                reason = str(update.get("reason", "") or "")
                evidence = update.get("evidence") or []
                audit_flags = list(update.get("audit_flags") or [])
                baseline_status = str(update.get("baseline_status", "") or "")
                transaction_id = str(update.get("transaction_id", "") or "")
                if evidence:
                    reason = (reason + "；证据：" + "、".join(str(value) for value in evidence)).strip("；")
                if baseline_status == "stale_conflict":
                    reason = (reason + "；导出后本地同一区域已修改，结果已过期，禁止自动选择").strip("；")
                elif baseline_status == "baseline_mismatch_no_text":
                    reason = (reason + "；基线哈希已过期且结果未携带 baseline_text，无法三方合并").strip("；")
                elif baseline_status == "auto_merged_non_overlapping":
                    reason = (reason + "；已生成非重叠三方合并候选，请人工确认").strip("；")
                if audit_flags:
                    reason = (reason + "；审计：" + "、".join(str(value) for value in audit_flags)).strip("；")
                state = self._fusion_states[row_index]
                previous_selected = state.selected_index
                confidence_value = update.get("confidence", 0.0)
                confidence_labels = {
                    "exact_reference": 1.0,
                    "high_consensus": 0.95,
                    "medium_context": 0.75,
                    "low_uncertain": 0.4,
                    "external_epub_edit": 0.7,
                }
                if isinstance(confidence_value, str) and confidence_value in confidence_labels:
                    confidence_value = confidence_labels[confidence_value]
                try:
                    confidence_value = float(confidence_value or 0.0)
                except (TypeError, ValueError, OverflowError):
                    confidence_value = 0.0
                audit_level = str(update.get("audit_level", "") or "")
                if transaction_id:
                    display_label = "外部AI跨条目事务（成组选择）"
                elif baseline_status in {"stale_conflict", "baseline_mismatch_no_text"} or audit_level == "high_risk":
                    display_label = "外部AI高风险建议（待人工）"
                elif baseline_status == "auto_merged_non_overlapping":
                    display_label = "外部AI三方合并建议（待人工）"
                else:
                    display_label = "外部AI修复建议（待人工）" if needs_review else "外部AI修复结果"
                upsert_external_candidate(
                    state,
                    str(update.get("edited_text", "") or ""),
                    display_label=display_label,
                    select=not needs_review,
                    confidence=confidence_value,
                    reason=reason,
                    allow_empty=bool(update.get("delete_intentionally", False)),
                    transaction_id=transaction_id,
                    transaction_operation=str(update.get("transaction_operation", "") or ""),
                    transaction_member_ids=list(update.get("transaction_member_ids") or []),
                    audit_level=audit_level,
                    audit_flags=audit_flags,
                    selection_origin=("ai_adjudication_result" if not needs_review else ""),
                )
                if needs_review:
                    state.selected_index = previous_selected
                    state.review_indices = state._build_review_indices()
                if needs_review:
                    review_only += 1
                else:
                    applied += 1
        except Exception as exc:
            show_error_dialog(self, "导入 AI 修复结果失败", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        self._sync_after_bulk_resolution(self._current_row_index)
        self._refresh_source_highlights()
        source_type = str(report.get("source_type", "") or "").upper()
        atomic_count = int(report.get("atomic_transaction_count", 0) or 0)
        stale_conflicts = int(report.get("stale_conflicts", 0) or 0)
        three_way_merges = int(report.get("three_way_merges", 0) or 0)
        self._summary.setText(
            f"已从 {source_type or 'AI修复文件'} 导入 {len(updates)} 条："
            f"自动选择 {applied}，待人工 {review_only}，原子事务 {atomic_count}，"
            f"过期冲突 {stale_conflicts}，三方合并 {three_way_merges}；原始 OCR 和结构未被覆盖。"
        )
        notify(
            self,
            f"AI 修复结果已载入：{len(updates)} 条 · 已选择 {applied} · 待人工 {review_only} · "
            f"原子事务 {atomic_count}。原始 OCR 和结构未被覆盖。",
            "success",
        )
