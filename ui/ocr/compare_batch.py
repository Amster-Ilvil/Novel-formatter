from __future__ import annotations

import copy

from ui.localized_dialogs import LocalizedMessageBox as QMessageBox
from ui.common.toast import notify

from core.batch_changes import BatchChange, BatchChangeSet, BatchRestoreGuard, content_state_token
from ui.batch_change_dialog import BatchChangePreviewDialog


class OCRCompareBatchService:
    """Preview/apply/restore owner for large OCR comparison batch decisions.

    The service does not touch OCR source documents, model transport, alignment
    identity or project package schemas.  It stages the same auto-selection on
    copies, renders the exact proposed selection delta, and swaps the staged
    state into the tab only after explicit confirmation.
    """

    def __init__(self, tab):
        self._tab = tab
        self._last_auto_restore: dict | None = None

    @staticmethod
    def _fusion_state_token(states) -> str:
        payload = []
        for state in list(states or ()):
            payload.append({
                "row_index": int(getattr(state, "row_index", -1)),
                "selected_index": getattr(state, "selected_index", None),
                "selection_origin": str(getattr(state, "selection_origin", "") or ""),
                "requires_confirmation": bool(getattr(state, "requires_confirmation", False)),
                "review_classification": str(getattr(state, "review_classification", "") or ""),
                "preserve_candidates_visible": bool(getattr(state, "preserve_candidates_visible", False)),
                "review_indices": [int(value) for value in tuple(getattr(state, "review_indices", ()) or ())],
                "candidates": [
                    {
                        "text": str(getattr(candidate, "text", "") or ""),
                        "model_indices": [int(value) for value in tuple(getattr(candidate, "model_indices", ()) or ())],
                        "display_label": str(getattr(candidate, "display_label", "") or ""),
                        "synthetic": bool(getattr(candidate, "synthetic", False)),
                        "delete_intentionally": bool(getattr(candidate, "delete_intentionally", False)),
                        "reason": str(getattr(candidate, "reason", "") or ""),
                        "confidence": float(getattr(candidate, "confidence", 0.0) or 0.0),
                        "transaction_id": str(getattr(candidate, "transaction_id", "") or ""),
                        "audit_level": str(getattr(candidate, "audit_level", "") or ""),
                        "audit_flags": [str(value) for value in tuple(getattr(candidate, "audit_flags", ()) or ())],
                    }
                    for candidate in list(getattr(state, "candidates", ()) or ())
                ],
            })
        return content_state_token(payload, lineage="ocr_compare_fusion")

    @staticmethod
    def _state_display(state) -> str:
        selected = getattr(state, "selected_index", None)
        origin = str(getattr(state, "selection_origin", "") or "")
        text = ""
        if selected is not None and 0 <= int(selected) < len(getattr(state, "candidates", ())):
            text = str(state.candidates[int(selected)].text or "")
        status = origin or ("待判断" if bool(getattr(state, "unresolved", False)) else "未选择")
        return f"[{status}] {text}".rstrip()

    def _apply_image_review_overrides_to(self, comparison, states) -> None:
        tab = self._tab
        if comparison is None or not getattr(tab, "_image_review_overrides", None):
            return
        from engine.ocr_compare_view_model import upsert_external_candidate

        for row_index, row in enumerate(comparison.rows):
            override = tab._image_review_overrides.get(tab._image_review_row_identity(row))
            if not isinstance(override, dict) or row_index >= len(states):
                continue
            upsert_external_candidate(
                states[row_index],
                str(override.get("text", "") or ""),
                display_label="图文对照人工校对",
                select=True,
                reason="由图文对照保存并同步；OCR 模型原文保持不变。",
                confidence=1.0,
                allow_empty=bool(override.get("delete_intentionally", False)),
                force_role_candidate=True,
                selection_origin="human_image_review",
            )

    def _stage_auto_select(self):
        tab = self._tab
        from engine.ocr_batch_preview import stage_auto_select

        row_texts = [
            [
                tab._source_line_text(model_index, row_index)
                for model_index in range(len(tab._documents))
            ]
            for row_index in range(len(tab._comparison.rows))
        ]
        staged_comparison, staged_states = stage_auto_select(
            tab._comparison, row_texts, tab._labels
        )
        self._apply_image_review_overrides_to(staged_comparison, staged_states)
        return staged_comparison, staged_states

    def _change_set(self, staged_states) -> BatchChangeSet:
        tab = self._tab
        changes: list[BatchChange] = []
        before_states = list(getattr(tab, "_fusion_states", ()) or ())
        for row_index, after_state in enumerate(staged_states):
            before_state = before_states[row_index] if row_index < len(before_states) else None
            before = self._state_display(before_state) if before_state is not None else "[无]"
            after = self._state_display(after_state)
            if before == after:
                continue
            row = tab._comparison.rows[row_index]
            category = "待判断→自动选择" if bool(getattr(before_state, "unresolved", False)) and not bool(getattr(after_state, "unresolved", False)) else "自动选优更新"
            stable_id = str(getattr(row, "sentence_group_id", "") or "")
            if not stable_id:
                stable_id = "+".join(str(value) for value in (getattr(row, "column_ids", ()) or ()))
            changes.append(BatchChange.create(
                f"row:{stable_id or row_index}",
                before,
                after,
                category=category,
                detail=f"第 {row_index + 1} 句",
                metadata={"row_index": row_index, "sentence_group_id": stable_id},
            ))
        return BatchChangeSet.from_changes(
            "OCR 自动选优 · 批量预览",
            changes,
            source_token=self._fusion_state_token(before_states),
            operation="ocr_compare_auto_select",
        )

    def _render_current_state(self) -> None:
        tab = self._tab
        tab._clear_fusion_widgets()
        tab._render_fusion_window(tab._current_row_index, force=True)
        tab._refresh_decision_queue(force=True)
        tab._update_unresolved_summary()

    def auto_select_all(self):
        tab = self._tab
        if tab._comparison is None:
            return
        if tab._sources_dirty or not tab._alignment_counts_valid(show_warning=False):
            tab._realign_and_auto()
            if tab._sources_dirty or not tab._alignment_counts_valid(show_warning=False):
                return

        staged_comparison, staged_states = self._stage_auto_select()
        change_set = self._change_set(staged_states)
        if change_set.count and not BatchChangePreviewDialog.confirm(
            tab, change_set, apply_text="应用自动选优"
        ):
            tab._summary.setText("已取消自动选优；当前 OCR 候选和裁决状态没有发生变化。")
            return

        if hasattr(tab, "_history_service"):
            tab._history_service.clear()
        before_comparison = tab._comparison
        before_states = tab._fusion_states
        tab._comparison = staged_comparison
        tab._fusion_states = staged_states
        self._render_current_state()
        expected_after = self._fusion_state_token(tab._fusion_states)
        self._last_auto_restore = {
            "before_comparison": before_comparison,
            "before_states": before_states,
            "guard": BatchRestoreGuard(
                operation="ocr_compare_auto_select",
                after_token=expected_after,
                change_set_fingerprint=change_set.fingerprint,
            ),
        }
        if hasattr(tab, "_auto_restore_batch_btn"):
            tab._auto_restore_batch_btn.setEnabled(True)
        tab._summary.setText(
            f"已预览并应用自动选优：{change_set.count} 项状态变化。{tab._comparison.summary} "
            "任何已选候选框中的文字仍可直接手动修改。"
        )
        tab._select_row(tab._current_row())

    def restore_last_auto_select(self):
        tab = self._tab
        snapshot = self._last_auto_restore
        if not isinstance(snapshot, dict):
            notify(tab, "没有可恢复的自动选优批次", "info")
            return
        if bool(getattr(tab, "_sources_dirty", False)):
            QMessageBox.warning(
                tab, "无法恢复",
                "自动选优后 OCR 源文字已经修改但尚未重新对齐。为避免覆盖后续工作，本次恢复已失效。"
            )
            self.invalidate_restore()
            return
        current_token = self._fusion_state_token(tab._fusion_states)
        guard = snapshot.get("guard")
        if not isinstance(guard, BatchRestoreGuard) or not guard.can_restore(current_token):
            QMessageBox.warning(
                tab, "无法恢复",
                "自动选优后已经发生新的人工/AI裁决。为避免覆盖这些后续选择，本次恢复已失效。"
            )
            self.invalidate_restore()
            return
        if hasattr(tab, "_history_service"):
            tab._history_service.clear()
        tab._comparison = snapshot.get("before_comparison")
        tab._fusion_states = snapshot.get("before_states") or []
        self.invalidate_restore()
        self._render_current_state()
        tab._summary.setText("已恢复自动选优前的候选与裁决状态；原始 OCR 文本始终未被改写。")
        if tab._fusion_states:
            tab._select_row(tab._current_row())

    def invalidate_restore(self) -> None:
        self._last_auto_restore = None
        tab = self._tab
        if hasattr(tab, "_auto_restore_batch_btn"):
            tab._auto_restore_batch_btn.setEnabled(False)
