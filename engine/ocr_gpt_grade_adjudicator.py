# -*- coding: utf-8 -*-
"""GPT-grade OCR adjudication pipeline.

This module deliberately composes the two mature adjudication paths that already
exist in Novel Formatter instead of inventing a third decision engine:

1. :mod:`ocr_visual_batch_adjudicator` sees source pixels first and produces an
   independent transcription plus guarded visual decision.
2. :mod:`ocr_ai_adjudicator` then receives the complete neighbouring text,
   physical-column/model evidence and the independent visual result, and performs
   a context-aware sparse adjudication followed by an independent audit.

The default mode is *blind*: publication/reference EPUB text is not consulted.
That matches the external GPT adjudication-package workflow, where scan pixels,
OCR candidates and neighbouring context are the evidence used to create the
answer.  A reference can still be supplied explicitly by a future caller, but it
is never discovered or enabled implicitly here.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from engine.ocr_ai_adjudicator import (
    AdjudicationCancelled,
    AdjudicationOptions,
    adjudicate_package,
)
from engine.ocr_visual_batch_adjudicator import VisualBatchOptions, adjudicate_visual_batches

SCHEMA = "novel_formatter.ocr_gpt_grade_adjudication.v1"


@dataclass(slots=True)
class GptGradeAdjudicationOptions:
    """Options for the two-stage production adjudicator.

    ``contextual_pass`` can be disabled for compatibility/benchmarking.  The
    production GUI enables it by default because this is the part that makes the
    in-app workflow behave like the full GPT adjudication package rather than a
    sheet-only visual chooser.
    """

    visual: VisualBatchOptions = field(default_factory=VisualBatchOptions)
    contextual: AdjudicationOptions = field(default_factory=AdjudicationOptions)
    contextual_pass: bool = True

    def normalised(self) -> "GptGradeAdjudicationOptions":
        return GptGradeAdjudicationOptions(
            visual=self.visual.normalised(),
            contextual=self.contextual.normalised(),
            contextual_pass=bool(self.contextual_pass),
        )


def _visual_evidence_map(report: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for raw in list(report.get("items") or []):
        if not isinstance(raw, dict):
            continue
        item_id = str(raw.get("item_id", "") or "")
        if not item_id:
            continue
        out[item_id] = {
            "image_transcription": str(raw.get("image_transcription", "") or ""),
            "proposed_text": str(raw.get("proposed_text", "") or ""),
            "confidence": str(raw.get("confidence", "") or ""),
            "needs_human_review": bool(raw.get("needs_human_review", False)),
            "audit_issues": [str(v) for v in (raw.get("audit_issues") or [])],
            "route_reason": str(raw.get("route_reason", "") or ""),
            "secondary_verdict": str(raw.get("secondary_verdict", "") or ""),
        }
    return out


def _attach_visual_evidence(package: dict, report: dict) -> dict:
    """Attach compact first-pass evidence and reseal the internal roundtrip.

    The evidence becomes immutable package metadata; only ``edited_text`` and
    ``delete_intentionally`` remain editable.  This lets the contextual pass and
    its independent auditor consume the actual first-pass image transcription
    without relying on hidden process state.
    """
    from engine.ocr_roundtrip_package import _editable_structure_hash, _seal_package

    result = copy.deepcopy(package)
    evidence = _visual_evidence_map(report)
    for item in list(result.get("editable_items") or []):
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("row_id", "") or "")
        if item_id in evidence:
            item["visual_evidence"] = copy.deepcopy(evidence[item_id])
    result["editable_structure_sha256"] = _editable_structure_hash(list(result.get("editable_items") or []))
    result["ai_visual_evidence_summary"] = {
        "schema": str(report.get("schema", "") or ""),
        "target_items": int((report.get("stats") or {}).get("target_items", 0) or 0),
        "visual_items": int((report.get("stats") or {}).get("visual_items", 0) or 0),
        "uncertain_items": int((report.get("stats") or {}).get("uncertain_items", 0) or 0),
    }
    _seal_package(result)
    return result


def _prefixed(callback: Callable[[dict], None] | None, prefix: str):
    if callback is None:
        return None

    def emit(event: dict) -> None:
        payload = dict(event or {})
        stage = str(payload.get("stage", "") or "")
        payload["stage"] = f"{prefix} · {stage}" if stage else prefix
        callback(payload)

    return emit


def adjudicate_gpt_grade(
    multimodal_client,
    text_provider,
    package: dict,
    fused_document,
    *,
    options: GptGradeAdjudicationOptions | None = None,
    reference_path: str | Path | None = None,
    glossary_path: str | Path | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[dict, dict]:
    """Run visual-first + contextual + independent-audit OCR adjudication.

    ``reference_path`` is accepted only for explicit callers.  The desktop GPT
    grade mode passes ``None`` so adjudication stays blind to electronic-book
    truth and remains comparable with the exported GPT adjudication workflow.
    """
    opts = (options or GptGradeAdjudicationOptions()).normalised()
    if cancel_check and cancel_check():
        raise AdjudicationCancelled("AI 审定已取消。")

    visual_package, visual_report = adjudicate_visual_batches(
        multimodal_client,
        package,
        fused_document,
        options=opts.visual,
        progress_callback=_prefixed(progress_callback, "视觉证据"),
        cancel_check=cancel_check,
    )
    enriched = _attach_visual_evidence(visual_package, visual_report)

    contextual_report: dict = {}
    contextual_error = ""
    final_package = enriched
    if opts.contextual_pass:
        if cancel_check and cancel_check():
            raise AdjudicationCancelled("AI 审定已取消。")
        try:
            final_package, contextual_report = adjudicate_package(
                text_provider,
                enriched,
                options=opts.contextual,
                reference_path=reference_path,
                glossary_path=glossary_path,
                progress_callback=_prefixed(progress_callback, "上下文裁决"),
                cancel_check=cancel_check,
            )
        except AdjudicationCancelled:
            raise
        except Exception as exc:
            # A successful source-image pass is still valuable evidence.  Do not
            # throw away its safe decisions because a separate text endpoint or
            # JSON protocol failed; expose the failure in the audit and keep the
            # visual result.  The GUI will show that contextual parity was not
            # reached for this run.
            contextual_error = str(exc)
            final_package = enriched

    visual_low = set(str(v) for v in (visual_report.get("low_uncertain") or []) if str(v))
    if contextual_report:
        final_low = set(str(v) for v in (contextual_report.get("low_uncertain") or []) if str(v))
    else:
        final_low = visual_low

    visual_stats = dict(visual_report.get("stats") or {})
    contextual_stats = dict(contextual_report.get("stats") or {}) if contextual_report else {}
    report = {
        "schema": SCHEMA,
        "mode": "visual_context_independent_audit" if opts.contextual_pass else "visual_only",
        "claims": (
            "Blind GPT-grade mode: source pixels, OCR candidates, physical columns and neighbouring context only; "
            "publication-reference text is not used unless explicitly supplied by the caller."
        ),
        "options": {
            "visual": asdict(opts.visual),
            "contextual": asdict(opts.contextual),
            "contextual_pass": bool(opts.contextual_pass),
            "reference_explicitly_supplied": bool(reference_path),
        },
        "stats": {
            "target_items": int(visual_stats.get("target_items", 0) or contextual_stats.get("target_items", 0) or 0),
            "visual_applied_changes": int(visual_stats.get("applied_changes", 0) or 0),
            "contextual_applied_changes": int(contextual_stats.get("applied_changes", 0) or 0),
            "final_uncertain_items": len(final_low),
            "visual_request_count": int(visual_stats.get("request_count", 0) or visual_report.get("request_count", 0) or 0),
            "contextual_request_batches": int(contextual_stats.get("request_batches", 0) or 0),
            "contextual_audit_failures": int(contextual_stats.get("audit_failures", 0) or 0),
        },
        # Compatibility view consumed by OCRCompareTab: final contextual rows
        # when available, otherwise the guarded visual rows.
        "items": list((contextual_report.get("items") if contextual_report else visual_report.get("items")) or []),
        "low_uncertain": sorted(final_low),
        "visual": visual_report,
        "contextual": contextual_report,
        "contextual_error": contextual_error,
    }
    return final_package, report


__all__ = [
    "SCHEMA",
    "GptGradeAdjudicationOptions",
    "adjudicate_gpt_grade",
]
