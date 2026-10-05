from __future__ import annotations

import copy
from pathlib import Path

from models.document import UnifiedDocument


class OCRCompareProjectStateService:
    """Project snapshot / adjudication-journal persistence for OCRCompareTab.

    This is deliberately Qt-free.  The service owns serialization and replay
    orchestration while the tab remains the authoritative owner of live widgets
    and comparison state.
    """

    def __init__(self, tab):
        self._tab = tab

    def set_project_package_dir(self, path) -> None:
        tab = self._tab
        tab._project_package_dir = str(path or "")
        if tab._project_package_dir:
            Path(tab._project_package_dir).mkdir(parents=True, exist_ok=True)

    def project_package_default(self, filename: str) -> str:
        tab = self._tab
        if tab._project_package_dir:
            return str(Path(tab._project_package_dir) / filename)
        return filename

    @staticmethod
    def jsonable(value):
        from dataclasses import asdict, is_dataclass
        if is_dataclass(value):
            return OCRCompareProjectStateService.jsonable(asdict(value))
        if isinstance(value, dict):
            return {str(k): OCRCompareProjectStateService.jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [OCRCompareProjectStateService.jsonable(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def snapshot_state(self) -> dict:
        """Return a JSON-safe multi-OCR/adjudication snapshot for the project."""
        tab = self._tab
        if tab._mode != "multi" or tab._comparison is None or len(tab._documents) < 2:
            return {}
        return self.jsonable({
            "schema_version": 1,
            "mode": "multi",
            "documents": [document.to_dict() for document in tab._documents],
            "labels": list(tab._labels),
            "comparison": tab._comparison,
            "ruby_overlay_doc": (
                tab._ruby_overlay_doc.to_dict() if tab._ruby_overlay_doc is not None else None
            ),
            "fusion_states": list(tab._fusion_states),
            "image_review_overrides": copy.deepcopy(tab._image_review_overrides),
            "canonical_source_decisions": [
                copy.deepcopy(item) for item in tab._canonical_source_decisions.values()
            ],
            "source_correction_import_history": copy.deepcopy(tab._source_correction_import_history),
            "source_correction_imported_package_ids": list(tab._source_correction_imported_package_ids),
            "last_source_correction_report": copy.deepcopy(tab._last_source_correction_report),
            "targeted_retry_local_report": copy.deepcopy(tab._targeted_retry_local_report),
            "source_texts": [tab._source_text(index) for index in range(len(tab._documents))],
            "current_row_index": int(tab._current_row_index),
        })

    @staticmethod
    def prepare_snapshot_restore(snapshot: dict) -> dict | None:
        """Deserialize a large project OCR snapshot without touching Qt widgets."""
        data = dict(snapshot or {})
        if data.get("mode") != "multi":
            return None
        from engine.multi_ocr_compare import MultiOcrComparison, MultiOcrRow
        from engine.ocr_compare_view_model import FusionCandidateState, FusionDecisionState
        from engine.multi_ocr_source_correction import canonical_decision_key

        documents = [
            UnifiedDocument.from_dict(item)
            for item in (data.get("documents") or [])
            if isinstance(item, dict)
        ]
        labels = [str(value) for value in (data.get("labels") or [])]
        raw_comparison = dict(data.get("comparison") or {})
        raw_rows = list(raw_comparison.pop("rows", []) or [])
        rows = [MultiOcrRow(**dict(item)) for item in raw_rows if isinstance(item, dict)]
        comparison = MultiOcrComparison(rows=rows, **raw_comparison)
        ruby_payload = data.get("ruby_overlay_doc")
        ruby_doc = UnifiedDocument.from_dict(ruby_payload) if isinstance(ruby_payload, dict) else None
        restored_states = []
        for item in list(data.get("fusion_states") or []):
            if not isinstance(item, dict):
                continue
            payload = dict(item)
            raw_candidates = list(payload.pop("candidates", []) or [])
            candidates = [
                FusionCandidateState(**dict(candidate))
                for candidate in raw_candidates if isinstance(candidate, dict)
            ]
            restored_states.append(FusionDecisionState(candidates=candidates, **payload))
        raw_decisions = data.get("canonical_source_decisions") or []
        if isinstance(raw_decisions, dict):
            decision_items = [item for item in raw_decisions.values() if isinstance(item, dict)]
        else:
            decision_items = [item for item in raw_decisions if isinstance(item, dict)]
        decisions = {
            canonical_decision_key(item): copy.deepcopy(item)
            for item in decision_items if canonical_decision_key(item)
        }
        return {
            "documents": documents,
            "labels": labels,
            "comparison": comparison,
            "ruby_doc": ruby_doc,
            "fusion_states": restored_states,
            "image_review_overrides": copy.deepcopy(data.get("image_review_overrides") or {}),
            "canonical_source_decisions": decisions,
            "source_correction_import_history": copy.deepcopy(data.get("source_correction_import_history") or []),
            "source_correction_imported_package_ids": set(data.get("source_correction_imported_package_ids") or []),
            "last_source_correction_report": copy.deepcopy(data.get("last_source_correction_report")),
            "targeted_retry_local_report": copy.deepcopy(data.get("targeted_retry_local_report")),
            "source_texts": list(data.get("source_texts") or []),
            "current_row_index": int(data.get("current_row_index", 0) or 0),
        }

    def restore_snapshot(self, snapshot: dict, *, prepared: dict | None = None) -> bool:
        """Restore a project snapshot; heavy deserialization may be precomputed off-thread."""
        tab = self._tab
        try:
            state = prepared or self.prepare_snapshot_restore(snapshot)
            if not state:
                return False
            documents = list(state.get("documents") or [])
            labels = list(state.get("labels") or [])
            comparison = state.get("comparison")
            ruby_doc = state.get("ruby_doc")
            restored_states = list(state.get("fusion_states") or [])
            tab.set_results({
                "documents": documents,
                "labels": labels,
                "comparison": comparison,
                "source_texts": list(state.get("source_texts") or []),
                "fused": ruby_doc,
                "ruby_enabled": ruby_doc is not None,
                "restored_fusion_states": restored_states,
            })
            if len(restored_states) == len(tab._fusion_states):
                tab._fusion_states = restored_states
            tab._image_review_overrides = copy.deepcopy(state.get("image_review_overrides") or {})
            tab._canonical_source_decisions = dict(state.get("canonical_source_decisions") or {})
            tab._source_correction_import_history = copy.deepcopy(state.get("source_correction_import_history") or [])
            tab._source_correction_imported_package_ids = set(
                state.get("source_correction_imported_package_ids") or []
            )
            tab._last_source_correction_report = copy.deepcopy(state.get("last_source_correction_report"))
            tab._targeted_retry_local_report = copy.deepcopy(state.get("targeted_retry_local_report"))
            tab._current_row_index = max(0, min(
                int(state.get("current_row_index", 0) or 0),
                max(0, len(tab._fusion_states) - 1),
            ))
            tab._clear_fusion_widgets(dispose=False)
            tab._reapply_image_review_overrides()
            tab._render_fusion_window(tab._current_row_index, force=True)
            tab._refresh_decision_queue(force=True)
            tab._update_unresolved_summary()
            tab._refresh_source_highlights()
            return True
        except Exception:
            return False

    def apply_adjudication_events(self, events) -> int:
        """Replay the append-only decision journal over the sealed OCR base."""
        tab = self._tab
        if tab._mode != "multi" or tab._comparison is None or not tab._fusion_states:
            return 0
        from engine.ocr_compare_view_model import resolve_stable_row_index, upsert_external_candidate

        compact: dict[tuple, dict] = {}
        for raw in events or ():
            if not isinstance(raw, dict):
                continue
            data = raw.get("decision") if isinstance(raw.get("decision"), dict) else raw
            try:
                raw_row = int(data.get("row_index", -1))
            except (TypeError, ValueError, OverflowError):
                raw_row = -1
            group_id = str(data.get("sentence_group_id", "") or "")
            columns = tuple(str(v) for v in (data.get("column_ids") or ()) if str(v))
            key = ("sentence", group_id) if group_id else (("columns",) + columns if columns else ("row", raw_row))
            compact[key] = dict(data)

        affected: set[int] = set()
        for data in compact.values():
            try:
                raw_row = int(data.get("row_index", -1))
            except (TypeError, ValueError, OverflowError):
                raw_row = -1
            group_id = str(data.get("sentence_group_id", "") or "")
            columns = tuple(str(v) for v in (data.get("column_ids") or ()) if str(v))
            resolved = resolve_stable_row_index(tab._comparison.rows, raw_row, columns, group_id)
            if resolved is None:
                continue
            row_index = int(resolved)
            if not 0 <= row_index < len(tab._fusion_states):
                continue
            state = tab._fusion_states[row_index]
            if not bool(data.get("resolved", False)):
                state.selected_index = None
                state.selection_origin = ""
                state.review_indices = state._build_review_indices()
                if str(data.get("origin", "")) == "image_review":
                    tab._image_review_overrides.pop(tab._image_review_row_identity(tab._comparison.rows[row_index]), None)
                affected.add(row_index)
                continue

            text = str(data.get("text", "") or "")
            delete_intentionally = bool(data.get("delete_intentionally", False))
            try:
                selected_index = int(data.get("selected_candidate_index", -1))
            except (TypeError, ValueError, OverflowError):
                selected_index = -1
            selection_origin = str(data.get("selection_origin", "") or "")
            if not selection_origin:
                selection_origin = "human_image_review" if str(data.get("origin", "")) == "image_review" else "human_ocr_compare"

            matched = None
            if 0 <= selected_index < len(state.candidates):
                candidate = state.candidates[selected_index]
                if str(getattr(candidate, "text", "") or "") == text:
                    matched = selected_index
            if matched is None:
                matched = next((
                    index for index, candidate in enumerate(state.candidates)
                    if str(getattr(candidate, "text", "") or "") == text
                    and bool(getattr(candidate, "delete_intentionally", False)) == delete_intentionally
                ), None)
            if matched is not None:
                state.selected_index = int(matched)
                state.selection_origin = selection_origin
                state.review_indices = state._build_review_indices()
            else:
                matched = upsert_external_candidate(
                    state, text,
                    display_label=str(data.get("display_label", "") or "恢复的人工裁决"),
                    select=True,
                    reason=str(data.get("reason", "") or "从项目裁决日志恢复。"),
                    confidence=float(data.get("confidence", 1.0) or 1.0),
                    allow_empty=delete_intentionally,
                    force_role_candidate=True,
                    selection_origin=selection_origin,
                )
                if matched is None:
                    continue
            if selection_origin == "human_image_review" or str(data.get("origin", "")) == "image_review":
                row = tab._comparison.rows[row_index]
                tab._image_review_overrides[tab._image_review_row_identity(row)] = {
                    "text": text,
                    "sentence_group_id": str(getattr(row, "sentence_group_id", "") or ""),
                    "column_ids": list(getattr(row, "column_ids", ()) or ()),
                    "segment_key": str(data.get("segment_key", "") or ""),
                    "delete_intentionally": delete_intentionally,
                }
            affected.add(row_index)

        if not affected:
            return 0
        tab._sync_canonical_authority_from_states(sorted(affected))
        tab._clear_fusion_widgets(dispose=False)
        tab._render_fusion_window(tab._current_row_index, force=True)
        tab._refresh_decision_queue(force=True)
        tab._update_unresolved_summary(refresh_queue=False)
        return len(affected)
