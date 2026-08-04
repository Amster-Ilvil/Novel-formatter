from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any


def _metadata_mapping(document: Any) -> MutableMapping[str, Any] | None:
    metadata = getattr(document, "metadata", None)
    if isinstance(metadata, MutableMapping):
        return metadata
    raw = getattr(metadata, "__dict__", None)
    return raw if isinstance(raw, MutableMapping) else None


def assess_column_ocr_health(document: Any) -> dict[str, object]:
    """Assess whether an OCR model decoded enough physical columns to vote.

    Strict column OCR preserves unresolved slots with visible placeholders, so
    structural column-ID completeness does not imply usable recognized text.
    This pure helper deliberately has no Qt dependency and can be reused by the
    GUI, export pipeline, and tests.
    """
    raw_meta = _metadata_mapping(document)
    audit = raw_meta.get("column_ocr_audit") if raw_meta is not None else None
    if not isinstance(audit, dict):
        return {
            "available": False,
            "usable": True,
            "warning": False,
            "expected": 0,
            "recognized": 0,
            "text_recognized": 0,
            "pending_manual": 0,
            "coverage": 1.0,
        }

    totals = audit.get("totals")
    if not isinstance(totals, dict):
        return {
            "available": False,
            "usable": True,
            "warning": False,
            "expected": 0,
            "recognized": 0,
            "text_recognized": 0,
            "pending_manual": 0,
            "coverage": 1.0,
        }

    def safe_nonnegative_int(value: Any) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError, OverflowError):
            return 0

    expected = safe_nonnegative_int(totals.get("expected"))
    recognized = safe_nonnegative_int(totals.get("recognized"))
    text_recognized = safe_nonnegative_int(totals.get("text_recognized"))
    pending_manual = safe_nonnegative_int(totals.get("pending_manual"))
    coverage = min(1.0, text_recognized / max(1, expected)) if expected else 1.0

    # Tiny documents do not provide enough evidence for a hard model-level
    # rejection.  On normal books, less than 35% usable text is generally a
    # failed runtime/input contract rather than ordinary OCR uncertainty.
    enough_evidence = expected >= 20
    usable = not enough_evidence or coverage >= 0.35
    warning = enough_evidence and usable and coverage < 0.90
    result: dict[str, object] = {
        "available": bool(expected),
        "usable": bool(usable),
        "warning": bool(warning),
        "expected": expected,
        "recognized": recognized,
        "text_recognized": text_recognized,
        "pending_manual": pending_manual,
        "coverage": coverage,
    }
    if raw_meta is not None:
        raw_meta["column_ocr_model_health"] = dict(result)
    return result
