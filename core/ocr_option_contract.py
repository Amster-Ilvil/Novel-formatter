"""Pure dependency contracts for OCR checkboxes and related controls.

The GUI deliberately delegates enable/disable decisions to this module so a
visible option can never be silently hard-disabled or coupled through a loose
lambda.  The functions contain no Qt imports and are covered by ordinary unit
tests on every platform, including CI environments without PySide6.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MultiOcrOptionState:
    model_select_enabled: bool
    quick_consensus_enabled: bool
    parallel_first_enabled: bool
    sentence_mode_enabled: bool
    quick_consensus_active: bool


@dataclass(frozen=True, slots=True)
class ColumnCleanupOptionState:
    cleanup_controls_enabled: bool
    cleanup_strength_enabled: bool


def resolve_multi_ocr_option_state(
    *,
    multi_enabled: bool,
    column_capable: bool,
    column_split_enabled: bool,
    quick_consensus_checked: bool,
) -> MultiOcrOptionState:
    """Return the exact UI dependency state for multi-model OCR controls.

    Quick consensus and first-round parallelism are independent sibling
    features.  Both require a shared physical-column run, while the model
    selectors only require multi-model mode.  The sentence-stage selector is
    disabled only when quick consensus is *actually active*, not merely checked
    while its required parent control is unavailable.
    """

    multi = bool(multi_enabled)
    column_run = bool(multi and column_capable and column_split_enabled)
    quick_active = bool(column_run and quick_consensus_checked)
    return MultiOcrOptionState(
        model_select_enabled=multi,
        quick_consensus_enabled=column_run,
        parallel_first_enabled=column_run,
        sentence_mode_enabled=bool(multi and not quick_active),
        quick_consensus_active=quick_active,
    )


def resolve_column_cleanup_option_state(
    *,
    column_split_enabled: bool,
    ruby_filter_checked: bool,
    fragment_filter_checked: bool,
) -> ColumnCleanupOptionState:
    """Return availability for destructive column-image cleanup controls."""

    column_active = bool(column_split_enabled)
    return ColumnCleanupOptionState(
        cleanup_controls_enabled=column_active,
        cleanup_strength_enabled=bool(
            column_active and (ruby_filter_checked or fragment_filter_checked)
        ),
    )


_V8_COLUMN_PREPROCESS_DEFAULTS = {
    "ruby_filter": True,
    "fragment_filter": False,
    "smart_crop": True,
    "ruby_strength": "standard",
}


def column_preprocess_defaults() -> dict[str, object]:
    """Return a fresh copy of the hidden stable OCR preprocessing defaults."""

    return dict(_V8_COLUMN_PREPROCESS_DEFAULTS)


def is_default_column_preprocess(
    *,
    ruby_filter: bool,
    fragment_filter: bool,
    smart_crop: bool,
    ruby_strength: str,
) -> bool:
    """Return whether advanced controls still match the hidden default contract."""

    strength = str(ruby_strength or "standard").strip().lower()
    if strength not in {"weak", "standard", "strong"}:
        strength = "standard"
    return bool(
        bool(ruby_filter) is True
        and bool(fragment_filter) is False
        and bool(smart_crop) is True
        and strength == "standard"
    )


def migrate_column_preprocess_state(state: dict | None) -> dict[str, object]:
    """Normalize persisted advanced preprocessing without trusting old profiles.

    Versions before v23 could write ``column_input_profile=custom`` even when
    the user never changed a control.  Only the explicit v23 customization
    marker is therefore authoritative.  Missing markers migrate to the stable
    default and historical profile names remain audit-only metadata.
    """

    source = dict(state or {})
    customized = bool(source.get("column_preprocess_customized", False))
    if not customized:
        result = column_preprocess_defaults()
        result["customized"] = False
        return result

    strength = str(source.get("column_ruby_strength", "standard") or "standard").strip().lower()
    if strength not in {"weak", "standard", "strong"}:
        strength = "standard"
    result = {
        "ruby_filter": bool(source.get("column_auto_filter_ruby", True)),
        "fragment_filter": bool(source.get("column_filter_fragments", False)),
        "smart_crop": bool(source.get("column_smart_crop", True)),
        "ruby_strength": strength,
    }
    result["customized"] = not is_default_column_preprocess(**result)
    return result
