from __future__ import annotations

from typing import Iterable

from ui.localized_dialogs import LocalizedMessageBox as QMessageBox
from ui.common.toast import notify

from core.adjudication_history import (
    AdjudicationDeltaHistory,
    AdjudicationHistoryEntry,
    AdjudicationRowDelta,
)
from engine.ocr_compare_view_model import FusionDecisionState
from engine.ocr_adjudication_history import (
    fusion_state_from_payload,
    fusion_state_payload,
    fusion_state_token,
    fusion_state_token_from_payload,
)


class OCRCompareAdjudicationHistoryService:
    """Session-local, conflict-guarded delta history for manual adjudication.

    Only the rows affected by a manual candidate choice/reopen (including
    atomic-transaction members) are snapshotted.  OCR source documents and the
    whole comparison are never copied.  External AI/import/source-correction
    changes are not rewritten into this stack; if they touch a recorded row,
    optimistic row tokens make the older undo/redo refuse to overwrite them.
    """

    def __init__(self, tab, *, max_entries: int = 100):
        self._tab = tab
        self._history = AdjudicationDeltaHistory(max_entries=max_entries)
        self._pending: dict | None = None

    # ---- stable row identity / serialization ---------------------------------

    def _row_key(self, row_index: int) -> str:
        tab = self._tab
        try:
            row = tab._comparison.rows[int(row_index)]
        except Exception:
            return f"row-index:{int(row_index)}"
        sentence_group_id = str(getattr(row, "sentence_group_id", "") or "")
        if sentence_group_id:
            return "sentence:" + sentence_group_id
        column_ids = tuple(str(value) for value in (getattr(row, "column_ids", ()) or ()))
        if column_ids:
            return "columns:" + "+".join(column_ids)
        return f"row-index:{int(row_index)}"

    @staticmethod
    def _payload(state: FusionDecisionState) -> str:
        return fusion_state_payload(state)

    @staticmethod
    def _state_from_payload(payload: str) -> FusionDecisionState:
        return fusion_state_from_payload(payload)

    @staticmethod
    def _token_from_payload(payload: str) -> str:
        return fusion_state_token_from_payload(payload)

    @staticmethod
    def _state_token(state: FusionDecisionState) -> str:
        return fusion_state_token(state)

    def _current_tokens(self, entry: AdjudicationHistoryEntry) -> dict[str, str]:
        tab = self._tab
        tokens: dict[str, str] = {}
        for delta in entry.deltas:
            index = int(delta.row_index)
            if not (0 <= index < len(tab._fusion_states)):
                tokens[delta.row_key] = ""
                continue
            if self._row_key(index) != delta.row_key:
                tokens[delta.row_key] = ""
                continue
            tokens[delta.row_key] = self._state_token(tab._fusion_states[index])
        return tokens

    # ---- capture --------------------------------------------------------------

    def _transaction_member_rows(self, transaction_ids: Iterable[str]) -> set[int]:
        ids = {str(value or "") for value in transaction_ids if str(value or "")}
        result: set[int] = set()
        if not ids:
            return result
        for index, state in enumerate(self._tab._fusion_states):
            if any(str(getattr(candidate, "transaction_id", "") or "") in ids for candidate in state.candidates):
                result.add(index)
        return result

    def prepare_resolution(self, row_index: int, candidate_index: int) -> None:
        tab = self._tab
        if not (0 <= int(row_index) < len(tab._fusion_states)):
            self._pending = None
            return
        state = tab._fusion_states[int(row_index)]
        selected_tx = ""
        if 0 <= int(candidate_index) < len(state.candidates):
            selected_tx = str(getattr(state.candidates[int(candidate_index)], "transaction_id", "") or "")
        row_transactions = {
            str(getattr(candidate, "transaction_id", "") or "")
            for candidate in state.candidates
            if str(getattr(candidate, "transaction_id", "") or "")
        }
        affected = {int(row_index)}
        affected.update(self._transaction_member_rows(row_transactions | ({selected_tx} if selected_tx else set())))
        self._prepare("manual_select", f"选择第 {int(row_index) + 1} 句候选", affected)

    def prepare_reopen(self, row_index: int) -> None:
        tab = self._tab
        if not (0 <= int(row_index) < len(tab._fusion_states)):
            self._pending = None
            return
        state = tab._fusion_states[int(row_index)]
        transaction_ids = {
            str(getattr(candidate, "transaction_id", "") or "")
            for candidate in state.candidates
            if str(getattr(candidate, "transaction_id", "") or "")
        }
        affected = {int(row_index)}
        affected.update(self._transaction_member_rows(transaction_ids))
        self._prepare("reopen", f"重新打开第 {int(row_index) + 1} 句", affected)

    def _prepare(self, operation: str, label: str, row_indices: Iterable[int]) -> None:
        tab = self._tab
        before: dict[int, tuple[str, str]] = {}
        for index in sorted(set(int(value) for value in row_indices)):
            if not (0 <= index < len(tab._fusion_states)):
                continue
            payload = self._payload(tab._fusion_states[index])
            before[index] = (self._row_key(index), payload)
        self._pending = {"operation": str(operation), "label": str(label), "before": before}

    def commit_resolution(self, _row_index: int) -> None:
        self._commit_pending()

    def commit_reopen(self, _row_index: int) -> None:
        self._commit_pending()

    def cancel_pending(self) -> None:
        self._pending = None

    def _commit_pending(self) -> None:
        tab = self._tab
        pending = self._pending
        self._pending = None
        if not isinstance(pending, dict):
            return
        deltas: list[AdjudicationRowDelta] = []
        for index, (row_key, before_payload) in (pending.get("before") or {}).items():
            if not (0 <= int(index) < len(tab._fusion_states)):
                continue
            if self._row_key(int(index)) != row_key:
                continue
            after_payload = self._payload(tab._fusion_states[int(index)])
            if before_payload == after_payload:
                continue
            deltas.append(AdjudicationRowDelta(
                row_key=row_key,
                row_index=int(index),
                before_payload=before_payload,
                after_payload=after_payload,
                before_token=self._token_from_payload(before_payload),
                after_token=self._token_from_payload(after_payload),
            ))
        if deltas:
            self._history.record(AdjudicationHistoryEntry(
                operation=str(pending.get("operation") or "manual"),
                label=str(pending.get("label") or "裁决操作"),
                deltas=tuple(deltas),
            ))
        self._refresh_controls()

    # ---- apply history --------------------------------------------------------

    def _apply_entry(self, entry: AdjudicationHistoryEntry, *, undo: bool) -> None:
        tab = self._tab
        affected: list[int] = []
        for delta in entry.deltas:
            index = int(delta.row_index)
            payload = delta.before_payload if undo else delta.after_payload
            tab._fusion_states[index] = self._state_from_payload(payload)
            affected.append(index)

        # The history changes only fusion authority. Re-publish exactly those
        # stable rows so image-review/project state stays synchronized.
        affected = sorted(set(affected))
        tab._clear_fusion_widgets()
        tab._render_fusion_window(tab._current_row_index, force=True)
        tab._sync_canonical_authority_from_states(affected)
        for index in affected:
            tab._publish_fusion_decision(index, origin="ocr_compare_history")
        tab._refresh_decision_queue(force=True)
        tab._update_unresolved_summary()
        if tab._fusion_states:
            tab._select_row(min(max(int(tab._current_row()), 0), len(tab._fusion_states) - 1))

    def undo(self) -> None:
        if not self._history.can_undo:
            notify(self._tab, "没有可撤销的裁决操作。", "info")
            return
        # First obtain the newest entry so we know which stable rows to hash.
        candidate = self._history.next_undo_entry
        entry = self._history.peek_undo(self._current_tokens(candidate)) if candidate is not None else None
        if entry is None:
            QMessageBox.warning(
                self._tab,
                "无法撤销",
                "最近一次裁决之后，同一句已经被人工编辑、AI裁决或外部导入修改。为避免覆盖后续结果，本次撤销已失效。",
            )
            self.clear()
            return
        self._apply_entry(entry, undo=True)
        self._history.commit_undo(entry)
        self._tab._summary.setText(f"已撤销：{entry.label}（仅恢复 {entry.count} 个受影响句）。")
        self._refresh_controls()

    def redo(self) -> None:
        if not self._history.can_redo:
            notify(self._tab, "没有可重做的裁决操作。", "info")
            return
        candidate = self._history.next_redo_entry
        entry = self._history.peek_redo(self._current_tokens(candidate)) if candidate is not None else None
        if entry is None:
            QMessageBox.warning(
                self._tab,
                "无法重做",
                "撤销之后，同一句已经发生新的修改。为避免覆盖后续结果，本次重做已失效。",
            )
            # A divergent edit creates a new branch. Keep already-undone older
            # entries inaccessible rather than replaying stale authority.
            self.clear()
            return
        self._apply_entry(entry, undo=False)
        self._history.commit_redo(entry)
        self._tab._summary.setText(f"已重做：{entry.label}（仅更新 {entry.count} 个受影响句）。")
        self._refresh_controls()

    def clear(self) -> None:
        self._pending = None
        self._history.clear()
        self._refresh_controls()

    def _refresh_controls(self) -> None:
        tab = self._tab
        undo = getattr(tab, "_adjudication_undo_btn", None)
        redo = getattr(tab, "_adjudication_redo_btn", None)
        if undo is not None:
            undo.setEnabled(self._history.can_undo)
            suffix = f"：{self._history.undo_label}" if self._history.undo_label else ""
            undo.setToolTip("撤销最近一次人工候选选择/重新打开；若同句之后被 AI 或人工修改，会拒绝覆盖" + suffix)
        if redo is not None:
            redo.setEnabled(self._history.can_redo)
            suffix = f"：{self._history.redo_label}" if self._history.redo_label else ""
            redo.setToolTip("重做刚刚撤销的裁决；只在相关句仍保持撤销后状态时允许" + suffix)
