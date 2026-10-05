#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conservative local adjudication from alternate-input OCR retries.

A retry is *not* an independent OCR vote.  It may only confirm an already
existing strict majority of independently executed OCR models.  This prevents
one recognizer from manufacturing extra votes by seeing several views of the
same pixels while still allowing alternate framing to verify a majority and
close a local conflict without cloud AI. Whole-row horizontal retries may ignore
unrelated OCR errors, but only when the original dispute is exclusively anchored.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
import unicodedata
from typing import Any, Iterable, Sequence

from engine.ocr_unicode_standardizer import japanese_ocr_comparison_key


@dataclass(slots=True)
class TargetedRetryEvidence:
    row_index: int
    model_index: int
    model_label: str
    original_text: str
    retry_text: str
    retry_kind: str = "alternate_input"
    confidence: float = 0.0
    input_sha256: str = ""
    input_path: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TargetedRetryDecision:
    row_index: int
    state: str  # AUTO_ACCEPT | LLM_REQUIRED | SKIPPED
    chosen_text: str
    confidence: float
    rationale: str
    majority_model_indices: tuple[int, ...] = ()
    majority_model_labels: tuple[str, ...] = ()
    retry_model_index: int = -1
    retry_model_label: str = ""
    retry_kind: str = ""
    retry_text: str = ""
    original_retry_model_text: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TargetedRetryAdjudicationReport:
    decisions: list[TargetedRetryDecision] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def local_resolved_row_indices(self) -> tuple[int, ...]:
        return tuple(d.row_index for d in self.decisions if d.state == "AUTO_ACCEPT")

    @property
    def llm_required_row_indices(self) -> tuple[int, ...]:
        return tuple(d.row_index for d in self.decisions if d.state == "LLM_REQUIRED")

    def to_dict(self) -> dict[str, Any]:
        return {
            "decisions": [asdict(d) for d in self.decisions],
            "summary": dict(self.summary),
        }


def _key(text: str) -> str:
    return str(japanese_ocr_comparison_key(str(text or ""))[0] or "")


def _usable(text: str) -> bool:
    value = str(text or "").strip()
    return bool(value) and "□" not in value and "�" not in value


_MAX_VERIFIED_GRID_BOUNDARY_INK_RATIO = 0.11
_MAX_VERIFIED_GRID_PEAK_BOUNDARY_INK_RATIO = 0.18


_FULL_ROW_RETRY_KINDS = frozenset({
    "single_column",
    "expanded_paper",
    "sentence_2x",
    "sentence_context",
    "sentence_context_retry",
    "ndl_tight_column_conflict_retry",
    "alternate_input",
})


def _retry_is_full_row_reading(item: TargetedRetryEvidence) -> bool:
    """Fail closed for cropped/diagnostic evidence.

    New retry producers must mark ``details["full_row_reading"]`` explicitly.
    A small compatibility allow-list keeps older full-row retry records usable,
    while any diagnostic/masked scope is never allowed to auto-confirm a row.
    """
    details = dict(item.details or {})
    explicit = details.get("full_row_reading")
    if explicit is not None:
        return bool(explicit)
    kind = str(item.retry_kind or "").strip()
    if kind == "masked_phrase_diagnostic" or "diagnostic" in kind:
        return False
    if kind in _FULL_ROW_RETRY_KINDS:
        return True
    return kind.endswith("_to_sentence_context_retry")




def _is_punctuation_char(ch: str) -> bool:
    if not ch:
        return False
    return unicodedata.category(ch).startswith("P") or ch in {"…", "‥"}


def _equal_index_map(reference: str, observed: str) -> dict[int, int]:
    """Map reference indices to observed indices only for exact equal runs."""
    mapping: dict[int, int] = {}
    for tag, i1, i2, j1, j2 in SequenceMatcher(
        None, reference, observed, autojunk=False
    ).get_opcodes():
        if tag != "equal":
            continue
        for offset in range(i2 - i1):
            mapping[i1 + offset] = j1 + offset
    return mapping


def _nearest_content_indices(text: str, start: int, direction: int, limit: int = 2) -> list[int]:
    output: list[int] = []
    index = int(start)
    while 0 <= index < len(text) and len(output) < max(0, int(limit)):
        if not _is_punctuation_char(text[index]) and not text[index].isspace():
            output.append(index)
        index += int(direction)
    if direction < 0:
        output.reverse()
    return output


def _local_candidate_support(
    reference: str, dissent: str, observed: str
) -> tuple[bool, str, dict[str, Any]]:
    """Return whether ``observed`` uniquely reads the local reference dispute.

    This helper deliberately ignores non-target punctuation/character errors.  It
    requires exact equal-run mappings for the disputed reference glyphs plus two
    nearby non-punctuation anchors on each available side.
    """
    changes = [
        op for op in SequenceMatcher(None, reference, dissent, autojunk=False).get_opcodes()
        if op[0] != "equal"
    ]
    if len(changes) != 1:
        return False, "", {}
    tag, i1, i2, j1, j2 = changes[0]
    if max(i2 - i1, j2 - j1) > 2:
        return False, "", {}
    mapping = _equal_index_map(reference, observed)
    left_anchors = _nearest_content_indices(reference, i1 - 1, -1, 2)
    right_anchors = _nearest_content_indices(reference, i2, +1, 2)
    anchor_indices = left_anchors + right_anchors
    if not anchor_indices or any(index not in mapping for index in anchor_indices):
        return False, "", {}

    if i1 < i2:
        target_indices = list(range(i1, i2))
        if any(index not in mapping for index in target_indices):
            return False, "", {}
        target = reference[i1:i2]
        observed_target = "".join(observed[mapping[index]] for index in target_indices)
        if observed_target != target:
            return False, "", {}
    else:
        # Reference omits the dissenting insertion.  Confirm absence only when
        # the immediate physical neighbours remain adjacent in observed OCR.
        if not (0 < i1 < len(reference)):
            return False, "", {}
        left = i1 - 1
        right = i1
        if left not in mapping or right not in mapping:
            return False, "", {}
        if mapping[right] != mapping[left] + 1:
            return False, "", {}
        target = ""
        observed_target = ""

    ordered = [
        index for index in left_anchors + list(range(i1, i2)) + right_anchors
        if index in mapping
    ]
    mapped = [mapping[index] for index in ordered]
    if any(a >= b for a, b in zip(mapped, mapped[1:])):
        return False, "", {}
    return True, target, {
        "dispute_tag": str(tag),
        "majority_span": [int(i1), int(i2)],
        "dissent_span": [int(j1), int(j2)],
        "majority_dispute": reference[i1:i2],
        "dissent_dispute": dissent[j1:j2],
        "retry_dispute": observed_target,
        "left_anchor_indices": left_anchors,
        "right_anchor_indices": right_anchors,
        "ignored_non_dispute_errors": observed != reference,
    }


def _retry_verified_dispute_matches_majority(
    item: TargetedRetryEvidence, majority_key: str, original_key: str
) -> tuple[bool, str, dict[str, Any]]:
    """Verify only the original disagreement inside whole-row horizontal OCR.

    The fixed physical-cell grid proves one source cell per majority-key glyph.
    OCR errors elsewhere are ignored, but confirmation must be *exclusive*: the
    same horizontal OCR may not also satisfy the dissenting candidate's local
    reading.  Ambiguous punctuation runs therefore stay unresolved.
    """
    if str(item.retry_kind or "") != "horizontal_reflow_dispute":
        return False, "", {}
    details = dict(item.details or {})
    if not (
        details.get("grid_verified") is True
        and details.get("glyph_count_matches_majority") is True
        and details.get("ruby_free") is True
        and details.get("full_row_reading") is True
        and str(details.get("verification_scope") or "") == "dispute_only"
    ):
        return False, "", {}
    try:
        boundary_ratio = float(details.get("grid_boundary_ink_ratio", 1.0))
        peak_boundary_ratio = float(details.get("grid_peak_boundary_ink_ratio", 1.0))
        pitches = tuple(int(value) for value in (details.get("grid_pitches") or ()))
    except Exception:
        return False, "", {}
    if (
        not pitches
        or any(value <= 0 for value in pitches)
        or not (0.0 <= boundary_ratio <= _MAX_VERIFIED_GRID_BOUNDARY_INK_RATIO)
        or not (0.0 <= peak_boundary_ratio <= _MAX_VERIFIED_GRID_PEAK_BOUNDARY_INK_RATIO)
    ):
        return False, "", {}
    expected_full = str(details.get("expected_comparison_key") or "")
    if expected_full != majority_key:
        return False, "", {}
    retry_key = _key(item.retry_text)
    if not retry_key:
        return False, "", {}

    ok, target, meta = _local_candidate_support(majority_key, original_key, retry_key)
    if not ok:
        return False, "", {}
    reverse_ok, _reverse_target, _reverse_meta = _local_candidate_support(
        original_key, majority_key, retry_key
    )
    if reverse_ok:
        return False, "", {}
    meta["exclusive_confirmation"] = True
    meta["dissent_candidate_also_supported"] = False
    return True, target, meta


def _retry_verified_span_matches_majority(
    item: TargetedRetryEvidence, majority_key: str
) -> tuple[bool, str]:
    """Validate a geometry-indexed local horizontal re-read.

    This path is intentionally limited to the production
    ``horizontal_reflow_focus`` producer.  The producer first proves that the
    total fixed-grid physical-cell count equals the strict-majority comparison
    key, then emits one deterministic local window.  The OCR result must match
    that *entire* majority window, not merely the changed glyph.
    """
    if str(item.retry_kind or "") != "horizontal_reflow_focus":
        return False, ""
    details = dict(item.details or {})
    if not (
        details.get("verified_span_reading") is True
        and details.get("grid_verified") is True
        and details.get("glyph_count_matches_majority") is True
        and details.get("ruby_free") is True
        and details.get("full_row_reading") is False
    ):
        return False, ""
    try:
        boundary_ratio = float(details.get("grid_boundary_ink_ratio", 1.0))
        peak_boundary_ratio = float(details.get("grid_peak_boundary_ink_ratio", 1.0))
        pitches = tuple(int(value) for value in (details.get("grid_pitches") or ()))
    except Exception:
        return False, ""
    if (
        not pitches
        or any(value <= 0 for value in pitches)
        or not (0.0 <= boundary_ratio <= _MAX_VERIFIED_GRID_BOUNDARY_INK_RATIO)
        or not (0.0 <= peak_boundary_ratio <= _MAX_VERIFIED_GRID_PEAK_BOUNDARY_INK_RATIO)
    ):
        return False, ""
    try:
        start = int(details.get("span_start", -1))
        end = int(details.get("span_end", -1))
    except Exception:
        return False, ""
    expected = str(details.get("expected_comparison_key") or "")
    if not (0 <= start < end <= len(majority_key)):
        return False, ""
    if len(expected) < 3 or majority_key[start:end] != expected:
        return False, ""
    if _key(item.retry_text) != expected:
        return False, ""
    return True, expected


def _strict_majority(row) -> tuple[str, str, tuple[int, ...]] | None:
    """Return (comparison_key, representative_raw_text, model_indices)."""
    seeded = {
        int(index) for index in (getattr(row, "consensus_seeded_models", ()) or ())
        if 0 <= int(index) < len(getattr(row, "texts", ()) or ())
    }
    groups: dict[str, list[tuple[int, str]]] = {}
    for index, raw in enumerate(list(getattr(row, "texts", ()) or [])):
        if index in seeded or not _usable(raw):
            continue
        groups.setdefault(_key(raw), []).append((index, str(raw)))
    usable_count = sum(len(values) for values in groups.values())
    if usable_count < 3 or not groups:
        return None
    ranked = sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)
    majority_key, values = ranked[0]
    count = len(values)
    if count <= usable_count // 2:
        return None
    # A normalized-key tie cannot be a strict majority by construction, but
    # keep this explicit for future weighting changes.
    if len(ranked) > 1 and len(ranked[1][1]) == count:
        return None
    representative = values[0][1]
    return majority_key, representative, tuple(index for index, _raw in values)


def strict_majority_candidate(row) -> tuple[str, str, tuple[int, ...]] | None:
    """Public read-only view of the pre-existing independent-model majority."""
    return _strict_majority(row)


def adjudicate_targeted_retries(
    comparison,
    retry_evidence: Iterable[TargetedRetryEvidence | dict[str, Any]],
    *,
    labels: Sequence[str] = (),
) -> TargetedRetryAdjudicationReport:
    """Confirm existing strict-majority rows with alternate-input retries.

    Safety rules:
    * the original OCR row must already have a strict majority among >=3 usable
      independently executed model outputs;
    * the retrying model must originally disagree with that majority;
    * a complete-row retry may confirm the row only when it normalizes exactly
      to the majority key; geometry-verified horizontal reflow may confirm only the original dispute
      position while ignoring unrelated horizontal OCR errors elsewhere;
    * proportional/masked/local diagnostic crops can never confirm a row;
    * retries never increment the vote count and never create a majority.
    """
    evidence_by_row: dict[int, list[TargetedRetryEvidence]] = {}
    for raw in retry_evidence:
        if isinstance(raw, TargetedRetryEvidence):
            item = raw
        elif isinstance(raw, dict):
            try:
                item = TargetedRetryEvidence(**raw)
            except Exception:
                continue
        else:
            continue
        evidence_by_row.setdefault(int(item.row_index), []).append(item)

    decisions: list[TargetedRetryDecision] = []
    considered = resolved = no_majority = retry_mismatch = invalid = diagnostic_only = 0
    rows = list(getattr(comparison, "rows", ()) or ())
    for row_index, items in sorted(evidence_by_row.items()):
        if not 0 <= row_index < len(rows):
            invalid += 1
            continue
        row = rows[row_index]
        if not bool(getattr(row, "is_conflict", False)):
            continue
        considered += 1
        majority = _strict_majority(row)
        if majority is None:
            no_majority += 1
            decisions.append(TargetedRetryDecision(
                row_index=row_index,
                state="LLM_REQUIRED",
                chosen_text="",
                confidence=0.0,
                rationale="原始独立 OCR 未形成严格多数；同模型重试不能制造额外投票。",
            ))
            continue
        majority_key, majority_text, majority_indices = majority
        majority_labels = tuple(
            str(labels[index] if index < len(labels) else f"模型{index + 1}")
            for index in majority_indices
        )
        accepted: TargetedRetryDecision | None = None
        for item in items:
            model_index = int(item.model_index)
            texts = list(getattr(row, "texts", ()) or ())
            if not 0 <= model_index < len(texts):
                invalid += 1
                continue
            original = str(texts[model_index] or "")
            # Confirmation is meaningful only when the same model originally
            # dissented. Retrying a majority model merely confirms itself.
            if _usable(original) and _key(original) == majority_key:
                continue
            if not _usable(item.retry_text):
                invalid += 1
                continue
            full_row = _retry_is_full_row_reading(item)
            span_ok = False
            expected_span = ""
            dispute_ok = False
            dispute_meta: dict[str, Any] = {}
            if str(item.retry_kind or "") == "horizontal_reflow_dispute":
                dispute_ok, expected_span, dispute_meta = _retry_verified_dispute_matches_majority(
                    item, majority_key, _key(original)
                )
                if not dispute_ok:
                    diagnostic_only += 1
                    continue
            elif not full_row:
                span_ok, expected_span = _retry_verified_span_matches_majority(item, majority_key)
                if not span_ok:
                    diagnostic_only += 1
                    continue
            elif _key(item.retry_text) != majority_key:
                retry_mismatch += 1
                continue
            model_label = str(
                item.model_label
                or (labels[model_index] if model_index < len(labels) else f"模型{model_index + 1}")
            )
            confidence = max(0.90, min(0.999, float(item.confidence or 0.0)))
            if dispute_ok:
                rationale = (
                    f"{len(majority_indices)} 个独立 OCR 模型已形成严格多数；"
                    f"原持异议的 {model_label} 对整句固定物理字格横排复识后，"
                    "仅原始争议位与多数候选一致，且争议前后正文锚点可唯一对齐。"
                    "横排 OCR 在非争议位置的标点或字符误差不参与本次判断；"
                    "复识只作局部确认，不增加模型票数。"
                )
            elif span_ok:
                context_cells = int((item.details or {}).get("horizontal_context_cells", 0) or 0)
                context_note = (f"（争议两侧各 {context_cells} 格上下文）" if context_cells > 0 else "")
                rationale = (
                    f"{len(majority_indices)} 个独立 OCR 模型已形成严格多数；"
                    f"原持异议的 {model_label} 对固定物理字格生成横排争议窗口{context_note}后，"
                    "完整窗口与多数候选对应片段精确一致。字格总数已先与多数文本长度核验；"
                    "该复识只作局部确认，不增加模型票数。"
                )
            else:
                rationale = (
                    f"{len(majority_indices)} 个独立 OCR 模型已形成严格多数；"
                    f"原持异议的 {model_label} 采用 {item.retry_kind or 'alternate_input'} "
                    "完整行重试后收敛到同一比较候选。重试仅作确认，不增加模型票数。"
                )
            accepted = TargetedRetryDecision(
                row_index=row_index,
                state="AUTO_ACCEPT",
                chosen_text=majority_text,
                confidence=confidence,
                rationale=rationale,
                majority_model_indices=tuple(majority_indices),
                majority_model_labels=majority_labels,
                retry_model_index=model_index,
                retry_model_label=model_label,
                retry_kind=str(item.retry_kind or "alternate_input"),
                retry_text=str(item.retry_text or ""),
                original_retry_model_text=original,
                evidence={
                    "retry_input_sha256": str(item.input_sha256 or ""),
                    "retry_input_path": str(item.input_path or ""),
                    "retry_details": dict(item.details or {}),
                    "retry_is_full_row_reading": bool(full_row),
                    "retry_is_verified_span_reading": bool(span_ok),
                    "retry_is_verified_dispute_reading": bool(dispute_ok),
                    "verified_majority_span": str(expected_span or ""),
                    "verified_dispute": dict(dispute_meta),
                    "retry_is_independent_vote": False,
                    "majority_was_preexisting": True,
                },
            )
            break
        if accepted is not None:
            resolved += 1
            decisions.append(accepted)
        else:
            decisions.append(TargetedRetryDecision(
                row_index=row_index,
                state="LLM_REQUIRED",
                chosen_text="",
                confidence=0.0,
                rationale="定向重试未与原始独立模型严格多数收敛，保留给后续复核/AI。",
                majority_model_indices=tuple(majority_indices),
                majority_model_labels=majority_labels,
            ))

    return TargetedRetryAdjudicationReport(
        decisions=decisions,
        summary={
            "rows_considered": considered,
            "local_resolved": resolved,
            "llm_required": sum(d.state == "LLM_REQUIRED" for d in decisions),
            "no_preexisting_strict_majority": no_majority,
            "retry_mismatch": retry_mismatch,
            "diagnostic_only_evidence": diagnostic_only,
            "invalid_retry_evidence": invalid,
            "policy": "preexisting_strict_majority_plus_dissenting_model_retry_confirmation",
            "retry_counts_as_independent_vote": False,
        },
    )


def retry_evidence_from_document(
    base_comparison,
    canonical_document,
    retry_document,
    *,
    model_index: int,
    model_label: str,
    retry_kind: str,
    target_row_indices: Iterable[int] | None = None,
    details: dict[str, Any] | None = None,
) -> list[TargetedRetryEvidence]:
    """Project a selective retry document back onto the existing comparison rows."""
    from engine.multi_ocr_compare import compare_ocr_documents

    retry_comp = compare_ocr_documents(
        [canonical_document, retry_document],
        ["canonical_geometry", str(model_label or "retry")],
    )
    by_group: dict[str, Any] = {}
    by_columns: dict[tuple[str, ...], Any] = {}
    for row in retry_comp.rows:
        group_id = str(getattr(row, "sentence_group_id", "") or "")
        columns = tuple(str(value) for value in (getattr(row, "column_ids", ()) or ()) if str(value))
        if group_id:
            by_group[group_id] = row
        if columns:
            by_columns[columns] = row

    wanted = None if target_row_indices is None else {int(value) for value in target_row_indices}
    output: list[TargetedRetryEvidence] = []
    for row_index, base_row in enumerate(list(getattr(base_comparison, "rows", ()) or ())):
        if wanted is not None and row_index not in wanted:
            continue
        group_id = str(getattr(base_row, "sentence_group_id", "") or "")
        columns = tuple(str(value) for value in (getattr(base_row, "column_ids", ()) or ()) if str(value))
        retry_row = by_group.get(group_id) if group_id else None
        if retry_row is None and columns:
            retry_row = by_columns.get(columns)
        if retry_row is None:
            continue
        retry_texts = list(getattr(retry_row, "texts", ()) or ())
        if len(retry_texts) < 2:
            continue
        base_texts = list(getattr(base_row, "texts", ()) or ())
        if not 0 <= int(model_index) < len(base_texts):
            continue
        confidences = list(getattr(retry_row, "model_confidences", ()) or ())
        output.append(TargetedRetryEvidence(
            row_index=row_index,
            model_index=int(model_index),
            model_label=str(model_label or ""),
            original_text=str(base_texts[int(model_index)] or ""),
            retry_text=str(retry_texts[1] or ""),
            retry_kind=str(retry_kind or "alternate_input"),
            confidence=(float(confidences[1] or 0.0) if len(confidences) > 1 else 0.0),
            details={**dict(details or {}), "full_row_reading": True},
        ))
    return output


def targeted_retry_plan(
    comparison,
    *,
    model_limit: int | None = None,
    exclude_rows: Iterable[int] = (),
) -> dict[int, tuple[int, ...]]:
    """Return dissenting model -> rows where a retry can confirm a strict majority."""
    excluded = {int(value) for value in exclude_rows}
    plan: dict[int, list[int]] = {}
    rows = list(getattr(comparison, "rows", ()) or ())
    for row_index, row in enumerate(rows):
        if row_index in excluded or not bool(getattr(row, "is_conflict", False)):
            continue
        majority = _strict_majority(row)
        if majority is None:
            continue
        majority_key, _text, majority_indices = majority
        majority_set = set(majority_indices)
        seeded = {int(v) for v in (getattr(row, "consensus_seeded_models", ()) or ())}
        for model_index, raw in enumerate(list(getattr(row, "texts", ()) or [])):
            if model_limit is not None and model_index >= int(model_limit):
                continue
            if model_index in seeded or model_index in majority_set:
                continue
            # Empty/failed original output can also be usefully retried, but a
            # non-empty dissent is the primary targeted-retry use case.
            if _usable(raw) and _key(raw) == majority_key:
                continue
            plan.setdefault(model_index, []).append(row_index)
    return {index: tuple(values) for index, values in plan.items() if values}


def build_local_adjudication_freeze_map(
    package: dict[str, Any],
    canonical_decisions: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Map accepted local canonical verdicts to sealed roundtrip row IDs."""
    editable_items = list(package.get("editable_items") or [])
    local_origins = {"local_targeted_retry_majority_adjudication"}
    output: dict[str, dict[str, Any]] = {}
    for decision in canonical_decisions:
        if not isinstance(decision, dict):
            continue
        if str(decision.get("status", "") or "") != "accepted":
            continue
        source = str(decision.get("source", "") or "")
        if source not in local_origins:
            continue
        # Old workspaces may contain historical local-majority overlays created
        # before the Stable Core retry contract was tightened. Do not freeze
        # them into a new AI package unless the canonical decision proves the
        # exact safe path: pre-existing independent majority + dissenting-model
        # retry convergence; retry itself is explicitly not a vote.
        audit_level = str(decision.get("audit_level", "") or "")
        audit_flags = {str(value) for value in (decision.get("audit_flags") or [])}
        required_flags = {
            "retry_is_not_independent_vote",
            "preexisting_strict_model_majority",
            "dissenting_model_retry_converged",
            "raw_ocr_sources_preserved",
        }
        if audit_level != "local_targeted_retry_majority_confirmation":
            continue
        if not required_flags.issubset(audit_flags):
            continue
        try:
            row_index = int(decision.get("row_index", -1))
        except (TypeError, ValueError, OverflowError):
            continue
        if not 0 <= row_index < len(editable_items):
            continue
        item = editable_items[row_index]
        if not isinstance(item, dict):
            continue
        row_id = str(item.get("row_id", "") or "")
        chosen_text = str(decision.get("final_text", "") or "")
        if not row_id or not chosen_text.strip():
            continue
        output[row_id] = {
            "state": "AUTO_ACCEPT",
            "chosen_text": chosen_text,
            "confidence": float(decision.get("confidence", 0.0) or 0.0),
            "rationale": str(decision.get("reason", "") or ""),
            "source": source,
            "audit_level": str(decision.get("audit_level", "") or ""),
            "audit_flags": list(decision.get("audit_flags") or []),
        }
    return output
