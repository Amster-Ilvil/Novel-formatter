#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight state and windowing helpers for the large OCR comparison UI.

The Qt workspace used to materialise one nested editor widget for every aligned
sentence.  A 300-page book can contain several thousand sentences, so that
approach creates thousands of QFrames/QPlainTextEdits on the GUI thread.

This module keeps all decisions in small Python objects while the GUI creates
widgets only for a bounded window around the active sentence.  No OCR text,
selection, model membership, or row order is discarded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Iterable, Sequence

from engine.multi_ocr_compare import exact_candidate_groups, exact_consensus_candidate
from engine.adaptive_ocr_ensemble import standard_japanese_key


EXPLICIT_FUSION_SELECTION_ORIGINS = frozenset({
    "human_ocr_compare",
    "human_image_review",
    "human_manual_edit",
    "restored_human",
    "external_ai_package",
    "ai_adjudication_result",
})


def is_explicit_fusion_selection_origin(value: str) -> bool:
    """Return whether a selected fusion candidate is an explicit authority override."""
    return str(value or "") in EXPLICIT_FUSION_SELECTION_ORIGINS


def selection_supersedes_ai_verdict(state, candidate_label: str = "") -> bool:
    """Return whether the current selection may intentionally override imported AI text."""
    return (
        is_explicit_fusion_selection_origin(str(getattr(state, "selection_origin", "") or ""))
        or str(candidate_label or "") in {
            "图文对照人工校对",
            "恢复的人工融合结果",
            "人工最终裁决",
        }
    )


@dataclass(slots=True)
class FusionCandidateState:
    text: str
    model_indices: tuple[int, ...] = ()
    display_label: str = ""
    synthetic: bool = False
    confidence: float = 0.0
    reason: str = ""
    delete_intentionally: bool = False
    transaction_id: str = ""
    transaction_operation: str = ""
    transaction_member_ids: tuple[str, ...] = ()
    audit_level: str = ""
    audit_flags: tuple[str, ...] = ()


def paired_candidate_diff_spans(left: str, right: str):
    """Return character-level marks for two editable OCR candidates.

    Marks use Python string offsets and one of three kinds:

    * ``replace``: characters exist on both sides but differ;
    * ``extra``: characters only exist in this candidate;
    * ``missing``: a zero-width boundary where the other candidate has text.

    The function is deliberately independent from Qt so the large comparison
    workspace can calculate only the currently materialised rows and the logic
    remains directly testable.
    """
    left_text = str(left or "")
    right_text = str(right or "")
    if standard_japanese_key(left_text) == standard_japanese_key(right_text):
        return (), (), 0
    left_marks: list[tuple[int, int, str]] = []
    right_marks: list[tuple[int, int, str]] = []
    change_count = 0

    matcher = SequenceMatcher(None, left_text, right_text, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        change_count += 1
        if tag == "replace":
            if i2 > i1:
                left_marks.append((i1, i2, "replace"))
            else:
                left_marks.append((i1, i1, "missing"))
            if j2 > j1:
                right_marks.append((j1, j2, "replace"))
            else:
                right_marks.append((j1, j1, "missing"))
        elif tag == "delete":
            left_marks.append((i1, i2, "extra"))
            right_marks.append((j1, j1, "missing"))
        elif tag == "insert":
            left_marks.append((i1, i1, "missing"))
            right_marks.append((j1, j2, "extra"))

    return tuple(left_marks), tuple(right_marks), change_count


@dataclass(slots=True)
class FusionDecisionState:
    row_index: int
    candidates: list[FusionCandidateState] = field(default_factory=list)
    selected_index: int | None = None
    preferred_model_index: int | None = None
    review_indices: tuple[int, ...] = ()
    local_reocr_recommended: bool = False
    fusion_reason: str = ""
    requires_confirmation: bool = False
    review_classification: str = ""
    # AI source-correction imports are non-destructive overlays.  Keep every
    # original OCR candidate visible even after the AI suggestion is selected.
    preserve_candidates_visible: bool = False
    # Track *how* the active candidate became authoritative.  The UI must be
    # able to distinguish automatic consensus/AI overlays from an explicit
    # human choice of any original OCR candidate during export validation.
    selection_origin: str = ""

    @classmethod
    def from_texts(
        cls,
        row_index: int,
        texts: Sequence[str],
        *,
        preferred_model_index: int | None = None,
        auto_choose: bool = False,
        synthetic_text: str = "",
        synthetic_auto_selected: bool = False,
        synthetic_confidence: float = 0.0,
        synthetic_reason: str = "",
        local_reocr_recommended: bool = False,
        requires_confirmation: bool = False,
        review_classification: str = "",
    ) -> "FusionDecisionState":
        consensus = exact_consensus_candidate(texts)
        groups = [consensus] if consensus is not None else exact_candidate_groups(texts)
        candidates = [
            FusionCandidateState(str(text or ""), tuple(model_indices))
            for text, model_indices in groups
        ]
        synthetic_value = str(synthetic_text or "").strip()
        synthetic_index: int | None = None
        if synthetic_value and all(candidate.text.strip() != synthetic_value for candidate in candidates):
            synthetic_index = len(candidates)
            candidates.append(FusionCandidateState(
                synthetic_value,
                (),
                display_label="字符级融合建议",
                synthetic=True,
                confidence=float(synthetic_confidence or 0.0),
                reason=str(synthetic_reason or ""),
            ))
        if not candidates:
            candidates = [FusionCandidateState("", ())]

        selected_index: int | None = None
        selection_origin = ""
        # v8-compatible authority rule: a row with only one distinct non-empty
        # candidate is never allowed to become an empty fusion line.  A
        # provisional-consensus marker may remain visible for audit/reopen, but
        # it cannot force the only available sentence back into an unresolved
        # state after AI import or session restore.
        if len(candidates) == 1:
            selected_index = 0
            selection_origin = "auto_single_candidate"
        elif consensus is not None and not requires_confirmation:
            selected_index = 0
            selection_origin = "auto_consensus"
        elif synthetic_index is not None and synthetic_auto_selected:
            selected_index = synthetic_index
            selection_origin = "auto_character_fusion"
        elif auto_choose and preferred_model_index is not None:
            selected_index = next(
                (
                    index
                    for index, candidate in enumerate(candidates)
                    if preferred_model_index in candidate.model_indices
                ),
                None,
            )
            if selected_index is not None:
                selection_origin = "auto_preferred_model"

        state = cls(
            row_index=row_index,
            candidates=candidates,
            selected_index=selected_index,
            preferred_model_index=preferred_model_index,
            local_reocr_recommended=bool(local_reocr_recommended),
            fusion_reason=str(synthetic_reason or ""),
            requires_confirmation=bool(requires_confirmation),
            review_classification=str(review_classification or ""),
            selection_origin=selection_origin,
        )
        state.review_indices = state._build_review_indices()
        return state

    def _build_review_indices(self) -> tuple[int, ...]:
        """Keep every distinct model candidate visible for an unresolved row.

        ``只显示需判断`` is a row filter, not a candidate filter.  Earlier
        versions reduced a three-way disagreement to the recommended candidate
        plus the most different alternative, which hid the third OCR model and
        made manual fusion impossible.  Exact duplicate texts are still grouped
        by ``from_texts``; all genuinely different non-empty sentences remain.
        """
        return tuple(range(len(self.candidates)))

    @property
    def unresolved(self) -> bool:
        return self.selected_index is None and (len(self.candidates) > 1 or self.requires_confirmation)

    def output_text(self) -> str:
        if self.selected_index is None:
            return ""
        if 0 <= self.selected_index < len(self.candidates):
            return self.candidates[self.selected_index].text.strip()
        return ""

    def output_delete_intentionally(self) -> bool:
        if self.selected_index is None or not 0 <= self.selected_index < len(self.candidates):
            return False
        return bool(self.candidates[self.selected_index].delete_intentionally)

    def choose(self, candidate_index: int, *, origin: str = "human_ocr_compare") -> bool:
        if not 0 <= candidate_index < len(self.candidates):
            return False
        if str(self.candidates[candidate_index].audit_level or "") == "historical_ocr_evidence":
            return False
        self.selected_index = candidate_index
        self.selection_origin = str(origin or "human_ocr_compare")
        return True

    def choose_model(self, model_index: int, *, origin: str = "human_ocr_compare") -> bool:
        for index, candidate in enumerate(self.candidates):
            if model_index in candidate.model_indices:
                self.selected_index = index
                self.selection_origin = str(origin or "human_ocr_compare")
                return True
        return False

    def choose_text(self, text: str, *, origin: str = "human_ocr_compare") -> bool:
        target = str(text or "").strip()
        for index, candidate in enumerate(self.candidates):
            if (
                candidate.text.strip() == target
                and str(candidate.audit_level or "") != "historical_ocr_evidence"
            ):
                self.selected_index = index
                self.selection_origin = str(origin or "human_ocr_compare")
                return True
        return False

    def reopen(self) -> bool:
        if len(self.candidates) <= 1 and not self.requires_confirmation:
            return False
        self.selected_index = None
        self.selection_origin = ""
        return True

    def visible_candidate_indices(self, review_only: bool) -> tuple[int, ...]:
        if self.preserve_candidates_visible:
            return tuple(range(len(self.candidates)))
        if self.selected_index is not None:
            return (self.selected_index,)
        # ``review_only`` filters whole rows in ``eligible_row_indices``.  It
        # must never hide one of three different OCR texts inside the active row.
        return tuple(range(len(self.candidates)))


def upsert_external_candidate(
    state: FusionDecisionState,
    text: str,
    *,
    display_label: str,
    select: bool,
    reason: str = "",
    confidence: float = 0.0,
    allow_empty: bool = False,
    transaction_id: str = "",
    transaction_operation: str = "",
    transaction_member_ids: Sequence[str] = (),
    audit_level: str = "",
    audit_flags: Sequence[str] = (),
    force_role_candidate: bool = False,
    selection_origin: str = "",
) -> int | None:
    """Add/select one external candidate without overwriting other synthetic roles.

    Character-level fusion and AI adjudication both use ``model_indices=()``.
    Treating every synthetic candidate as interchangeable used to rename or
    overwrite the character-fusion suggestion when an AI package was imported.
    This helper matches exact text first, then only reuses the same display role.
    """
    target = str(text or "").strip()
    if not target and not allow_empty:
        # A missing/uncertain external proposal is audit information only.  It
        # must never clear a valid selection that already exists in the fusion
        # state (the old behaviour made checked rows reappear as unresolved).
        state.review_indices = state._build_review_indices()
        return None
    if not target and allow_empty:
        index = next(
            (
                i for i, candidate in enumerate(state.candidates)
                if candidate.synthetic and candidate.display_label == display_label
            ),
            None,
        )
        if index is None:
            state.candidates.append(FusionCandidateState(
                "", (), display_label=display_label, synthetic=True,
                confidence=float(confidence or 0.0), reason=str(reason or ""),
                delete_intentionally=True,
                transaction_id=str(transaction_id or ""),
                transaction_operation=str(transaction_operation or ""),
                transaction_member_ids=tuple(str(value) for value in transaction_member_ids),
                audit_level=str(audit_level or ""),
                audit_flags=tuple(str(value) for value in audit_flags),
            ))
            index = len(state.candidates) - 1
        candidate = state.candidates[index]
        candidate.transaction_id = str(transaction_id or "")
        candidate.transaction_operation = str(transaction_operation or "")
        candidate.transaction_member_ids = tuple(str(value) for value in transaction_member_ids)
        candidate.audit_level = str(audit_level or "")
        candidate.audit_flags = tuple(str(value) for value in audit_flags)
        if select:
            state.selected_index = index
            state.selection_origin = str(selection_origin or "external_candidate")
        state.review_indices = state._build_review_indices()
        return index
    index = None
    if not transaction_id and not force_role_candidate:
        index = next(
            (i for i, candidate in enumerate(state.candidates) if candidate.text.strip() == target),
            None,
        )
    if index is None:
        index = next(
            (
                i for i, candidate in enumerate(state.candidates)
                if candidate.synthetic and candidate.display_label == display_label
            ),
            None,
        )
    if index is None:
        state.candidates.append(FusionCandidateState(
            target,
            (),
            display_label=display_label,
            synthetic=True,
            confidence=float(confidence or 0.0),
            reason=str(reason or ""),
            transaction_id=str(transaction_id or ""),
            transaction_operation=str(transaction_operation or ""),
            transaction_member_ids=tuple(str(value) for value in transaction_member_ids),
            audit_level=str(audit_level or ""),
            audit_flags=tuple(str(value) for value in audit_flags),
        ))
        index = len(state.candidates) - 1
    else:
        candidate = state.candidates[index]
        # Preserve the identity of an exact model/character-fusion candidate.
        # Only same-role synthetic candidates are rewritten in place.
        if candidate.synthetic and candidate.display_label == display_label:
            candidate.text = target
            candidate.confidence = float(confidence or 0.0)
            candidate.reason = str(reason or "")
            candidate.delete_intentionally = False
            candidate.transaction_id = str(transaction_id or "")
            candidate.transaction_operation = str(transaction_operation or "")
            candidate.transaction_member_ids = tuple(str(value) for value in transaction_member_ids)
            candidate.audit_level = str(audit_level or "")
            candidate.audit_flags = tuple(str(value) for value in audit_flags)
    if select:
        state.selected_index = index
        state.selection_origin = str(selection_origin or "external_candidate")
    state.review_indices = state._build_review_indices()
    return index


def build_fusion_states(
    rows: Iterable[object],
    *,
    auto_choose: bool = False,
) -> list[FusionDecisionState]:
    states: list[FusionDecisionState] = []
    for row_index, row in enumerate(rows):
        texts = list(getattr(row, "texts", []) or [])
        preferred = getattr(row, "chosen_index", None)
        try:
            preferred_index = int(preferred) if preferred is not None else None
        except (TypeError, ValueError):
            preferred_index = None
        state = FusionDecisionState.from_texts(
            row_index,
            texts,
            preferred_model_index=preferred_index,
            auto_choose=auto_choose,
            synthetic_text=str(getattr(row, "character_fused_text", "") or ""),
            synthetic_auto_selected=bool(getattr(row, "character_fusion_auto_selected", False)),
            synthetic_confidence=float(getattr(row, "character_fusion_confidence", 0.0) or 0.0),
            synthetic_reason=str(getattr(row, "character_fusion_reason", "") or ""),
            local_reocr_recommended=bool(getattr(row, "local_reocr_recommended", False)),
            # v8 semantics: two-model common candidates are usable output,
            # not a second confirmation queue.  The classification remains
            # visible for audit, but never blanks the only sentence.
            requires_confirmation=False,
            review_classification=(
                "source_correction_resolved"
                if bool(getattr(row, "source_correction_resolved", False))
                else str(getattr(row, "review_classification", "") or "")
            ),
        )
        historical_texts = tuple(
            str(value or "") for value in (getattr(row, "historical_ocr_texts", ()) or ())
        )
        historical_labels = tuple(
            str(value or "") for value in (getattr(row, "historical_ocr_labels", ()) or ())
        )
        grouped_history: dict[str, list[str]] = {}
        current_values = {candidate.text.strip() for candidate in state.candidates}
        for history_index, history_text in enumerate(historical_texts):
            value = history_text.strip()
            if not value or value in current_values:
                continue
            label = (
                historical_labels[history_index]
                if history_index < len(historical_labels) and historical_labels[history_index]
                else f"模型{history_index + 1}"
            )
            grouped_history.setdefault(value, []).append(label)
        for value, source_labels in grouped_history.items():
            state.candidates.append(FusionCandidateState(
                value,
                (),
                display_label="纠错前·" + "＋".join(dict.fromkeys(source_labels)),
                reason=str(getattr(row, "historical_resolution_reason", "") or "纠错前 OCR 分歧，只读保留。"),
                confidence=float(getattr(row, "historical_resolution_confidence", 0.0) or 0.0),
                audit_level="historical_ocr_evidence",
                audit_flags=("read_only_pre_correction_candidate",),
            ))
        state.review_indices = state._build_review_indices()
        states.append(state)
    return states


def resolve_stable_row_index(
    rows: Sequence[object],
    requested_row_index: int,
    incoming_columns: Sequence[str] = (),
) -> int | None:
    """Resolve a cross-view row without trusting a stale numeric position.

    The shared row index is fastest when it still refers to the same complete
    physical-column group.  After recovery or re-alignment, the exact ordered
    column identity is authoritative. Ambiguous matches are rejected rather
    than writing a decision to the wrong OCR sentence.
    """
    columns = tuple(str(value) for value in (incoming_columns or ()) if str(value))
    try:
        requested = int(requested_row_index)
    except (TypeError, ValueError, OverflowError):
        requested = -1
    if 0 <= requested < len(rows):
        if not columns:
            return requested
        current = tuple(
            str(value)
            for value in (getattr(rows[requested], "column_ids", ()) or ())
            if str(value)
        )
        if current == columns:
            return requested
    if not columns:
        return None
    matches = [
        index
        for index, row in enumerate(rows)
        if tuple(
            str(value)
            for value in (getattr(row, "column_ids", ()) or ())
            if str(value)
        ) == columns
    ]
    return matches[0] if len(matches) == 1 else None


def eligible_row_indices(
    states: Sequence[FusionDecisionState],
    *,
    review_only: bool,
) -> list[int]:
    if review_only:
        return [state.row_index for state in states if state.unresolved]
    return [state.row_index for state in states]


def windowed_row_indices(
    states: Sequence[FusionDecisionState],
    current_row: int,
    *,
    review_only: bool,
    window_size: int = 36,
    retain_resolved_current: bool = False,
) -> list[int]:
    """Return a bounded, stable window around ``current_row``.

    In ``review_only`` mode resolved rows are hidden immediately.  Earlier
    builds kept the just-resolved active row as a visual acknowledgement, which
    made a checked card appear to remain pending and could leave a stale cached
    widget disconnected from the authoritative fusion state.  Callers may opt
    into one-shot retention explicitly, but the normal OCR comparison workflow
    uses strict filtering.
    """
    if not states:
        return []
    size = max(5, int(window_size or 36))
    current_row = max(0, min(int(current_row), len(states) - 1))
    eligible = eligible_row_indices(states, review_only=review_only)
    if not eligible:
        return [current_row] if (retain_resolved_current and not review_only) else []

    if current_row in eligible:
        position = eligible.index(current_row)
    else:
        position = next(
            (index for index, row_index in enumerate(eligible) if row_index > current_row),
            len(eligible) - 1,
        )

    half = size // 2
    start = max(0, position - half)
    end = min(len(eligible), start + size)
    start = max(0, end - size)
    window = list(eligible[start:end])
    if retain_resolved_current and current_row not in window and not review_only:
        window.append(current_row)
        window.sort()
        if len(window) > size:
            farthest = max(
                (row for row in window if row != current_row),
                key=lambda row: abs(row - current_row),
            )
            window.remove(farthest)
    return window
