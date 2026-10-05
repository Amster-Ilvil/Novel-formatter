from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from adapters.ocr_profiles import OCRProfile, get_ocr_profile, is_engine_compatible


@dataclass(frozen=True, slots=True)
class OcrPreflightIssue:
    title: str
    message: str


@dataclass(frozen=True, slots=True)
class OcrRunPreflight:
    """GUI-thread OCR launch decision before model/runtime installation.

    The object contains only resolved launch policy and immutable snapshots or
    references to immutable profile/role-plan objects.  It deliberately does
    not start workers, install runtimes, touch model caches, or dispatch OCR.
    """

    inputs: tuple[str, ...]
    missing_inputs: tuple[str, ...]
    ocr_mode: str
    ocr_profile: OCRProfile
    ruby_preserve_enabled: bool
    ruby_scan_mode: str
    multi_role_enabled: bool
    multi_local_retry_enabled: bool
    multi_smart_router_enabled: bool
    multi_role_plan: Any | None
    engine_ids: tuple[str, ...]
    role_index_by_name: tuple[tuple[str, int], ...]
    physical_column_required: bool
    hayai_column_required: bool
    manga_48px_column_required: bool
    fixed_crop_rect: tuple[float, float, float, float] | None
    narrow_single_column_batch: bool
    issue: OcrPreflightIssue | None = None

    @property
    def role_index(self) -> dict[str, int]:
        return dict(self.role_index_by_name)


def _crop_tuple(value):
    if value is None:
        return None
    return tuple(float(item) for item in value[:4])


def build_run_preflight(tab) -> OcrRunPreflight:
    """Resolve inputs, OCR mode, engine roles and fixed-column requirements.

    All QWidget reads happen here on the GUI thread.  Validation failures are
    returned as data so the controller remains the only place that presents
    dialogs.  This keeps the preflight directly unit-testable without starting
    OCR workers or importing model runtimes.
    """

    inputs, missing_inputs = tab._resolve_ocr_run_inputs()
    inputs = tuple(str(path) for path in inputs)
    missing_inputs = tuple(str(path) for path in missing_inputs)
    no_inputs = not inputs

    ocr_mode = str(tab._current_ocr_mode())
    profile = get_ocr_profile(ocr_mode)

    # Defence in depth for legacy/restored settings.  This remains GUI-thread
    # state normalization, not execution policy: unsupported Japanese-only
    # controls are turned off before a run snapshot is created.
    if not profile.allow_column_pipeline:
        tab._column_split_check.setChecked(False)
        tab._column_sentence_reflow_check.setChecked(False)
    if not profile.allow_japanese_handwriting and hasattr(tab, "_handwriting_trace_check"):
        tab._handwriting_trace_check.setChecked(False)

    ruby_preserve_enabled = bool(
        profile.allow_column_pipeline
        and hasattr(tab, "_preserve_ruby_check")
        and tab._preserve_ruby_check.isChecked()
    )
    ruby_scan_mode = str(
        getattr(getattr(tab, "_ruby_scan_mode_combo", None), "currentData", lambda: "smart_roi")()
        or "smart_roi"
    )
    if ruby_scan_mode not in {"smart_roi", "full_page"}:
        ruby_scan_mode = "smart_roi"

    multi_role_enabled = bool(
        hasattr(tab, "_multi_ocr_check") and tab._multi_ocr_check.isChecked()
    )
    multi_local_retry_enabled = bool(
        multi_role_enabled
        and profile.allow_column_pipeline
        and getattr(tab, "_multi_local_retry_check", None) is not None
        and tab._multi_local_retry_check.isChecked()
    )
    multi_smart_router_enabled = bool(
        multi_role_enabled
        and hasattr(tab, "_multi_smart_router_check")
        and tab._multi_smart_router_check.isChecked()
    )
    multi_role_plan = tab._selected_multi_ocr_role_plan() if multi_role_enabled else None

    issue = None
    role_index_by_name: dict[str, int] = {}
    physical_column_required = False

    if multi_role_plan is not None:
        selected_role_engines = list(multi_role_plan.selected_engines)
        incompatible = [
            engine_id
            for engine_id in selected_role_engines
            if not is_engine_compatible(engine_id, ocr_mode)
        ]
        issues = list(multi_role_plan.validate())
        if incompatible:
            issues.append(
                f"当前“{profile.label}”模式不能使用："
                + "、".join(tab._engine_label(item) for item in incompatible)
            )

        from adapters.column_ocr_adapter import SUPPORTED_RECOGNIZERS
        unsupported_roles = [
            (role, engine_id)
            for role, engine_id in multi_role_plan.selected_roles
            if role != "page" and engine_id not in SUPPORTED_RECOGNIZERS
        ]
        if unsupported_roles:
            from core.multi_ocr_roles import ROLE_LABELS
            issues.extend(
                f"{ROLE_LABELS.get(role, role)}不能使用{tab._engine_label(engine_id)}；该引擎仅支持整页输入"
                for role, engine_id in unsupported_roles
            )
        if issues:
            issue = OcrPreflightIssue("多模型 OCR 配置不可运行", "\n".join(issues))

        selected_role_slots = list(multi_role_plan.selected_roles)
        engine_ids = tuple(engine for _role, engine in selected_role_slots)
        role_index_by_name = {
            role: index for index, (role, _engine) in enumerate(selected_role_slots)
        }
        # Multi-role evidence is aligned to one authoritative physical geometry.
        # The page role still receives the original full page.
        physical_column_required = True
    else:
        # Single-model resolution is intentionally independent from all role
        # comboboxes.  No stale multi-role selection can force column transport.
        engine_ids = (str(tab._selected_single_ocr_engine()),)
        incompatible = [
            engine_id for engine_id in engine_ids
            if not is_engine_compatible(engine_id, ocr_mode)
        ]
        if incompatible:
            issue = OcrPreflightIssue(
                "OCR 模式与引擎冲突",
                f"当前“{profile.label}”模式不能使用："
                + "、".join(tab._engine_label(item) for item in incompatible),
            )
        physical_column_required = False

    if issue is None and "windows_snipping_ocr" in engine_ids:
        from adapters.windows_snipping_ocr_adapter import availability
        ready, detail = availability()
        if not ready:
            issue = OcrPreflightIssue("Windows Snipping OCR 不可用", detail)

    if multi_role_plan is not None:
        columnish_engines = {
            engine for role, engine in multi_role_plan.selected_roles
            if role != "page" and engine
        }
    else:
        columnish_engines = set(engine_ids)
    hayai_column_required = "hayai_ocr" in columnish_engines
    manga_48px_column_required = "manga_48px" in columnish_engines

    fixed_crop_rect = _crop_tuple(tab._preview.get_crop_rect())
    narrow_single_column_batch = False
    if (
        multi_role_plan is None
        and tab._column_split_check.isChecked()
        and fixed_crop_rect is None
    ):
        # Sample only a handful of image headers; never scan every input on the
        # GUI thread.  Narrow manually-cropped sources already define the body.
        narrow_single_column_batch = bool(tab._sampled_narrow_single_column_batch(inputs))

    fixed_crop_required = bool(
        physical_column_required
        if multi_role_plan is not None
        else (tab._column_split_check.isChecked() and not narrow_single_column_batch)
    )
    if issue is None and fixed_crop_required and fixed_crop_rect is None:
        title = "日文列 OCR 必须先分列" if physical_column_required else "请先固定正文区域"
        detail = (
            "Hayai OCR 与 48px AR 使用物理列输入；多模型共享固定正文区域和右→左列几何。"
            "请先在右侧预览图上拖框选定纯正文区域，程序会按右到左生成物理列后再逐列识别。"
            if physical_column_required
            else "分列掩膜必须先在右侧预览图上拖框选定纯正文区域。\n"
                 "程序会保持整页尺寸，先遮住框外页眉/页脚，再逐列遮住其他文字。"
        )
        issue = OcrPreflightIssue(title, detail)

    # Preserve the original launch precedence: an empty runnable queue is
    # always reported as the input error, even if stale UI settings would also
    # fail a later engine/crop validation.
    if no_inputs:
        issue = OcrPreflightIssue("错误", "请先在页面管理中导入图片或 PDF")

    return OcrRunPreflight(
        inputs=inputs,
        missing_inputs=missing_inputs,
        ocr_mode=ocr_mode,
        ocr_profile=profile,
        ruby_preserve_enabled=ruby_preserve_enabled,
        ruby_scan_mode=ruby_scan_mode,
        multi_role_enabled=multi_role_enabled,
        multi_local_retry_enabled=multi_local_retry_enabled,
        multi_smart_router_enabled=multi_smart_router_enabled,
        multi_role_plan=multi_role_plan,
        engine_ids=engine_ids,
        role_index_by_name=tuple(role_index_by_name.items()),
        physical_column_required=physical_column_required,
        hayai_column_required=hayai_column_required,
        manga_48px_column_required=manga_48px_column_required,
        fixed_crop_rect=fixed_crop_rect,
        narrow_single_column_batch=narrow_single_column_batch,
        issue=issue,
    )
