from __future__ import annotations

import json

from core.batch_changes import content_state_token
from engine.ocr_compare_view_model import FusionCandidateState, FusionDecisionState


def fusion_state_dict(state: FusionDecisionState) -> dict:
    """Lossless JSON-safe snapshot of one fusion decision row."""
    return {
        "row_index": int(state.row_index),
        "selected_index": state.selected_index,
        "preferred_model_index": state.preferred_model_index,
        "review_indices": [int(value) for value in tuple(state.review_indices or ())],
        "local_reocr_recommended": bool(state.local_reocr_recommended),
        "fusion_reason": str(state.fusion_reason or ""),
        "requires_confirmation": bool(state.requires_confirmation),
        "review_classification": str(state.review_classification or ""),
        "preserve_candidates_visible": bool(state.preserve_candidates_visible),
        "selection_origin": str(state.selection_origin or ""),
        "candidates": [
            {
                "text": str(candidate.text or ""),
                "model_indices": [int(value) for value in tuple(candidate.model_indices or ())],
                "display_label": str(candidate.display_label or ""),
                "synthetic": bool(candidate.synthetic),
                "confidence": float(candidate.confidence or 0.0),
                "reason": str(candidate.reason or ""),
                "delete_intentionally": bool(candidate.delete_intentionally),
                "transaction_id": str(candidate.transaction_id or ""),
                "transaction_operation": str(candidate.transaction_operation or ""),
                "transaction_member_ids": [str(value) for value in tuple(candidate.transaction_member_ids or ())],
                "audit_level": str(candidate.audit_level or ""),
                "audit_flags": [str(value) for value in tuple(candidate.audit_flags or ())],
            }
            for candidate in list(state.candidates or ())
        ],
    }


def fusion_state_payload(state: FusionDecisionState) -> str:
    return json.dumps(
        fusion_state_dict(state),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def fusion_state_from_payload(payload: str) -> FusionDecisionState:
    raw = json.loads(str(payload or "{}"))
    return FusionDecisionState(
        row_index=int(raw.get("row_index", -1)),
        candidates=[
            FusionCandidateState(
                text=str(item.get("text", "") or ""),
                model_indices=tuple(int(value) for value in item.get("model_indices", ()) or ()),
                display_label=str(item.get("display_label", "") or ""),
                synthetic=bool(item.get("synthetic", False)),
                confidence=float(item.get("confidence", 0.0) or 0.0),
                reason=str(item.get("reason", "") or ""),
                delete_intentionally=bool(item.get("delete_intentionally", False)),
                transaction_id=str(item.get("transaction_id", "") or ""),
                transaction_operation=str(item.get("transaction_operation", "") or ""),
                transaction_member_ids=tuple(str(value) for value in item.get("transaction_member_ids", ()) or ()),
                audit_level=str(item.get("audit_level", "") or ""),
                audit_flags=tuple(str(value) for value in item.get("audit_flags", ()) or ()),
            )
            for item in raw.get("candidates", ()) or ()
        ],
        selected_index=raw.get("selected_index"),
        preferred_model_index=raw.get("preferred_model_index"),
        review_indices=tuple(int(value) for value in raw.get("review_indices", ()) or ()),
        local_reocr_recommended=bool(raw.get("local_reocr_recommended", False)),
        fusion_reason=str(raw.get("fusion_reason", "") or ""),
        requires_confirmation=bool(raw.get("requires_confirmation", False)),
        review_classification=str(raw.get("review_classification", "") or ""),
        preserve_candidates_visible=bool(raw.get("preserve_candidates_visible", False)),
        selection_origin=str(raw.get("selection_origin", "") or ""),
    )


def fusion_state_token_from_payload(payload: str) -> str:
    return content_state_token(json.loads(str(payload or "{}")), lineage="ocr_compare_row")


def fusion_state_token(state: FusionDecisionState) -> str:
    return fusion_state_token_from_payload(fusion_state_payload(state))
