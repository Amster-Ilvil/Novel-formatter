from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ui.ocr.run_plan import OcrRunPlan


@dataclass(frozen=True, slots=True)
class OcrRuntimeSnapshot:
    """Behavior-preserving GUI-thread snapshot used by OCR orchestration.

    The snapshot owns detached copies of the mutable model/column option maps so
    workers never read QWidget state and later UI changes cannot alter the active
    run.  Multi-model's canonical mask transport is applied only to this detached
    copy, preserving the independent single-model preference.
    """

    plan: OcrRunPlan
    engine_options: dict[str, dict[str, Any]]
    column_runtime: dict[str, Any]


def build_runtime_snapshot(
    tab,
    *,
    inputs: Iterable[str],
    missing_inputs: Iterable[str],
    manual_review_requested: bool,
    runtime_preflight_done: bool,
    ocr_mode: str,
    engine_ids: Iterable[str],
    multi_role_enabled: bool,
    fixed_crop_rect,
) -> OcrRuntimeSnapshot:
    """Freeze mutable OCR controls once on the GUI thread.

    This is intentionally a small extraction from ``OcrRunController``: it does
    not choose engines, inspect files, start workers, or change OCR semantics.
    """

    engines = tuple(str(engine_id) for engine_id in engine_ids)
    engine_options = {
        engine_id: dict(tab._engine_options(engine_id))
        for engine_id in engines
    }
    column_runtime = dict(tab._column_runtime_options_snapshot())
    if multi_role_enabled:
        # Multi-model has its own transport policy.  Never mutate the checkbox,
        # persisted single-model preference, or the dict returned by the view.
        column_runtime["column_isolation_mode"] = "mask"

    plan = OcrRunPlan.capture(
        inputs=inputs,
        missing_inputs=missing_inputs,
        manual_review_requested=manual_review_requested,
        runtime_preflight_done=runtime_preflight_done,
        ocr_mode=ocr_mode,
        engine_ids=engines,
        multi_role_enabled=multi_role_enabled,
        fixed_crop_rect=fixed_crop_rect,
        engine_options=engine_options,
        column_runtime=column_runtime,
    )
    return OcrRuntimeSnapshot(
        plan=plan,
        engine_options=engine_options,
        column_runtime=column_runtime,
    )
