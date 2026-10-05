from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _freeze(value: Any):
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class OcrRunPlan:
    """Immutable GUI-thread snapshot for one OCR run.

    Mutable Qt widgets are deliberately not stored here.  Complex option dictionaries are
    recursively frozen to tuples/frozensets so the plan cannot be changed after the worker
    is launched.
    """

    inputs: tuple[str, ...]
    missing_inputs: tuple[str, ...]
    manual_review_requested: bool
    runtime_preflight_done: bool
    ocr_mode: str
    engine_ids: tuple[str, ...]
    multi_role_enabled: bool
    fixed_crop_rect: tuple[float, float, float, float] | None
    engine_options: tuple
    column_runtime: tuple

    @classmethod
    def capture(
        cls,
        *,
        inputs,
        missing_inputs,
        manual_review_requested: bool,
        runtime_preflight_done: bool,
        ocr_mode: str,
        engine_ids,
        multi_role_enabled: bool,
        fixed_crop_rect,
        engine_options,
        column_runtime,
    ) -> "OcrRunPlan":
        crop = None
        if fixed_crop_rect is not None:
            crop = tuple(float(value) for value in fixed_crop_rect[:4])
        return cls(
            inputs=tuple(str(path) for path in inputs),
            missing_inputs=tuple(str(path) for path in missing_inputs),
            manual_review_requested=bool(manual_review_requested),
            runtime_preflight_done=bool(runtime_preflight_done),
            ocr_mode=str(ocr_mode),
            engine_ids=tuple(str(engine_id) for engine_id in engine_ids),
            multi_role_enabled=bool(multi_role_enabled),
            fixed_crop_rect=crop,
            engine_options=_freeze(dict(engine_options)),
            column_runtime=_freeze(dict(column_runtime)),
        )
