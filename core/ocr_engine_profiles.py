from __future__ import annotations

"""Per-engine OCR transport profiles.

Geometry is shared across engines; final recognizer input is not.  This module
contains only stable capability/transport facts so GUI, cache and adapters can
agree without hard-coding engine names in several places.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OcrEngineProfile:
    engine_id: str
    transport_profile: str
    input_role: str
    viewport_mode: str
    resource_class: str
    pad_x_ratio: float = 0.50
    pad_y_ratio: float = 0.30
    min_pad_x: int = 12
    min_pad_y: int = 10
    short_block_max_columns: int = 2
    min_canvas_width_ratio: float = 0.0
    min_canvas_height_ratio: float = 0.0
    notes: str = ""


# input_role describes the engine-native transport preference.  The role-based
# scheduler may still place an engine in column/page/sentence/review, but the
# final recognizer pixels remain engine-specific rather than slot-specific.
_PROFILES: dict[str, OcrEngineProfile] = {
    "ndlocr_lite": OcrEngineProfile(
        "ndlocr_lite", "ndl-page-projection-v1", "page", "context", "onnx_model",
        0.60, 0.35, 14, 10,
        notes="Native whole-page OCR; shared geometry is used for projection/fallback only.",
    ),
    "apple_vision": OcrEngineProfile(
        "apple_vision", "apple-vision-context-v1", "column", "context", "native_vision",
        0.85, 0.50, 18, 14,
        notes="Native pixels with generous paper context; avoids narrow manual-crop regression.",
    ),
    "macocr": OcrEngineProfile(
        "macocr", "apple-vision-context-v1", "column", "context", "native_vision",
        0.85, 0.50, 18, 14,
    ),
    "mac_ocr": OcrEngineProfile(
        "mac_ocr", "apple-vision-context-v1", "column", "context", "native_vision",
        0.85, 0.50, 18, 14,
    ),
    "macos_ocr": OcrEngineProfile(
        "macos_ocr", "apple-vision-context-v1", "column", "context", "native_vision",
        0.85, 0.50, 18, 14,
    ),
    "hayai_ocr": OcrEngineProfile(
        "hayai_ocr", "hayai-tight-native-v2", "column", "compact", "mps_model",
        0.36, 0.22, 10, 8,
        min_canvas_width_ratio=1.85, min_canvas_height_ratio=4.50,
        notes="Tight isolated ink plus blank paper context; never reintroduce neighbouring glyphs.",
    ),
    "manga_48px": OcrEngineProfile(
        "manga_48px", "48px-line-native-v2", "column", "line", "mps_model",
        0.24, 0.16, 8, 6,
        min_canvas_height_ratio=3.50,
        notes="Tight source column with blank end-context; recognizer performs rotate/48px normalization.",
    ),
    "manga_ocr": OcrEngineProfile(
        "manga_ocr", "manga-ocr-short-block-review-v1", "column", "short_block", "mps_model",
        0.20, 0.14, 8, 6,
        min_canvas_height_ratio=2.0,
        notes="Review-only: split disputed physical columns into short vertical blocks before fixed 224x224 OCR.",
    ),
    "paddle_ocr": OcrEngineProfile(
        "paddle_ocr", "paddle-context-v1", "column", "context", "onnx_model",
        0.70, 0.40, 16, 12,
    ),
    "google_vision": OcrEngineProfile(
        "google_vision", "layout-context-v1", "column", "context", "remote_model",
        0.70, 0.40, 16, 12,
    ),
}

_DEFAULT = OcrEngineProfile(
    "unknown", "generic-context-v1", "column", "context", "cpu_model",
    0.60, 0.35, 14, 10,
)


def get_ocr_engine_profile(engine_id: str) -> OcrEngineProfile:
    key = str(engine_id or "").strip().lower()
    return _PROFILES.get(key, OcrEngineProfile(
        key or _DEFAULT.engine_id,
        _DEFAULT.transport_profile,
        _DEFAULT.input_role,
        _DEFAULT.viewport_mode,
        _DEFAULT.resource_class,
        _DEFAULT.pad_x_ratio,
        _DEFAULT.pad_y_ratio,
        _DEFAULT.min_pad_x,
        _DEFAULT.min_pad_y,
        _DEFAULT.short_block_max_columns,
        _DEFAULT.min_canvas_width_ratio,
        _DEFAULT.min_canvas_height_ratio,
        _DEFAULT.notes,
    ))



def get_ocr_transport_profile(transport_profile: str) -> OcrEngineProfile | None:
    key = str(transport_profile or "").strip()
    if not key:
        return None
    seen: set[str] = set()
    for profile in _PROFILES.values():
        if profile.transport_profile in seen:
            continue
        seen.add(profile.transport_profile)
        if profile.transport_profile == key:
            return profile
    return None

def preferred_input_role(engine_id: str) -> str:
    return get_ocr_engine_profile(engine_id).input_role


def transport_profile_id(engine_id: str) -> str:
    return get_ocr_engine_profile(engine_id).transport_profile


def resource_class(engine_id: str) -> str:
    return get_ocr_engine_profile(engine_id).resource_class
