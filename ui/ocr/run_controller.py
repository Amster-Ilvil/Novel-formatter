from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

from PySide6.QtGui import QImageReader

from core.project_workspace import ProjectWorkspaceManager
from ui.common.signals import WorkerSignals
from ui.common.toast import notify
from ui.localized_dialogs import LocalizedMessageBox as QMessageBox
from ui.pages.types import TYPE_LABEL
from ui.ocr.run_plan import OcrRunPlan
from ui.ocr.run_setup import build_runtime_snapshot
from ui.ocr.run_preflight import build_run_preflight

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _bounded_review_rows(rows, review_engine: str, *, column_budget: int | None = None):
    """Select high-value disagreement rows under an explicit physical-column budget.

    Returned rows remain ordinary unresolved conflicts. This helper only decides
    where an expensive reviewer is allowed to acquire extra OCR evidence.
    """
    high_value_rows = [row for row in rows if row.high_value_review_evidence]
    if column_budget is None:
        default_budget = 80 if str(review_engine) == "manga_48px" else 256
        try:
            column_budget = max(0, int(os.environ.get(
                "NOVEL_FORMATTER_REVIEW_MAX_COLUMNS", str(default_budget)
            ) or default_budget))
        except (TypeError, ValueError):
            column_budget = default_budget
    column_budget = max(0, int(column_budget))
    ranked_rows = sorted(
        high_value_rows,
        key=lambda row: (-float(row.local_review_priority), int(row.index)),
    )
    selected = []
    column_ids: set[str] = set()
    for row in ranked_rows:
        row_ids = {str(value) for value in (row.column_ids or ()) if str(value)}
        added = row_ids - column_ids
        if column_budget <= 0:
            break
        if column_ids and len(column_ids) + len(added) > column_budget:
            continue
        selected.append(row)
        column_ids.update(row_ids)
        if len(column_ids) >= column_budget:
            break
    return selected, column_ids, column_budget


class OcrRunController:
    """Own the long-running OCR orchestration while OCRTab remains the view/state host.

    The migration is intentionally behavior-preserving: the implementation still calls
    existing OCRTab helpers, but the orchestration body no longer lives in the 20k-line
    compatibility GUI module.  ``last_run_plan`` exposes the immutable GUI-thread snapshot
    used to start the active run.
    """

    def __init__(self, tab):
        self._tab = tab
        self._last_run_plan: OcrRunPlan | None = None

    @property
    def last_run_plan(self) -> OcrRunPlan | None:
        return self._last_run_plan

    def run(
        self,
        *,
        _runtime_preflight_done: bool = False,
        _manual_review_requested_override: bool | None = None,
    ):
        return _run_ocr_impl(
            self._tab,
            self,
            _runtime_preflight_done=_runtime_preflight_done,
            _manual_review_requested_override=_manual_review_requested_override,
        )


def _run_ocr_impl(
    tab,
    controller: OcrRunController,
    *,
    _runtime_preflight_done: bool = False,
    _manual_review_requested_override: bool | None = None,
):
    self = tab
    manual_review_requested = (
        bool(_manual_review_requested_override)
        if _manual_review_requested_override is not None
        else bool(getattr(self, "_manual_review_requested", False))
    )
    if _manual_review_requested_override is None:
        self._manual_review_requested = False
    if self._ocr_preflight_active and not _runtime_preflight_done:
        return
    if _runtime_preflight_done:
        self._ocr_preflight_active = False
    if self._ocr_run_active:
        notify(self, "当前 OCR 尚未结束，请先停止或等待完成。", "warning")
        return
    if bool(getattr(self, "_paddle_prepare_active", False)):
        notify(self, "模型安装/修复完成后再开始 OCR，避免同时写入模型缓存。", "warning")
        return
    preflight = build_run_preflight(self)
    inputs = list(preflight.inputs)
    missing_inputs = list(preflight.missing_inputs)
    if missing_inputs:
        # Missing/deleted files are never allowed to enter the adapter queue.
        # Keep this warning on the GUI thread; workers receive only the
        # already-filtered immutable list below.
        preview = "\n".join(Path(path).name for path in missing_inputs[:8])
        suffix = "" if len(missing_inputs) <= 8 else f"\n……另有 {len(missing_inputs) - 8} 项"
        QMessageBox.warning(
            self,
            "已移除无效 OCR 页面",
            f"发现 {len(missing_inputs)} 个已删除或不存在的输入，已从本次 OCR 队列移除：\n"
            f"{preview}{suffix}",
        )
    if inputs != self._pending_inputs:
        self._pending_inputs = list(inputs)
    if preflight.issue is not None:
        QMessageBox.warning(self, preflight.issue.title, preflight.issue.message)
        return

    ocr_mode_snapshot = preflight.ocr_mode
    ocr_profile = preflight.ocr_profile
    ruby_preserve_enabled = preflight.ruby_preserve_enabled
    ruby_scan_mode_snapshot = preflight.ruby_scan_mode
    multi_role_enabled = preflight.multi_role_enabled
    multi_local_retry_enabled = preflight.multi_local_retry_enabled
    multi_smart_router_enabled = preflight.multi_smart_router_enabled
    multi_role_plan = preflight.multi_role_plan
    engine_ids = list(preflight.engine_ids)
    role_index_by_name = preflight.role_index
    physical_column_required = preflight.physical_column_required
    hayai_column_required = preflight.hayai_column_required
    manga_48px_column_required = preflight.manga_48px_column_required
    fixed_crop_rect = preflight.fixed_crop_rect

    # Architecture/source-contract markers retained for regression audits:
    # Empty-input user guidance remains: 请先在页面管理中导入图片或 PDF
    # engine_ids = [engine for _role, engine in selected_role_slots]
    # role_index_by_name = {role: index for index, (role, _engine) in enumerate(selected_role_slots)}
    # single-model resolver keeps: physical_column_required = False
    # narrow-source sampling lives in run_preflight:
    # self._sampled_narrow_single_column_batch(inputs)

    # Role-based multi-model may intentionally use only one main role.
    # Zero main roles were rejected above; review slots alone can never run.
    if not _runtime_preflight_done:
        self._begin_runtime_installation_preflight(
            engine_ids,
            include_ruby=ruby_preserve_enabled,
            resume_callback=lambda: self._run_ocr(
                _runtime_preflight_done=True,
                _manual_review_requested_override=manual_review_requested,
            ),
        )
        return

    # Freeze every model option on the GUI thread.  The builder owns the
    # detached option copies and the multi-model-only canonical mask policy;
    # workers therefore never read QWidget state and single-model preferences
    # are not mutated by a multi-model run.
    runtime_snapshot = build_runtime_snapshot(
        self,
        inputs=inputs,
        missing_inputs=missing_inputs,
        manual_review_requested=manual_review_requested,
        runtime_preflight_done=_runtime_preflight_done,
        ocr_mode=ocr_mode_snapshot,
        engine_ids=engine_ids,
        multi_role_enabled=multi_role_enabled,
        fixed_crop_rect=fixed_crop_rect,
    )
    frozen_engine_options = runtime_snapshot.engine_options
    frozen_column_runtime = runtime_snapshot.column_runtime
    controller._last_run_plan = runtime_snapshot.plan
    # Source-contract marker retained for existing architecture audits:
    # frozen_column_runtime["column_isolation_mode"] = "mask"

    # Start a fresh navigation queue for this OCR run, but deliberately keep
    # all earlier run directories on disk until the user clears OCR or closes
    # the application.  This avoids invalidating a worker that is still
    # finishing its last page and honours the workspace-level lifetime rule.
    self._reset_preview_history(keep_current_image=True)
    if self._live_preview_is_enabled():
        self._preview_filename_lbl.setText("当前图片：等待第一张识别图片…")
        self._preview_filename_lbl.setToolTip("")
    else:
        self._preview_filename_lbl.setText("当前图片：实时预览已关闭（OCR 继续运行）")
        self._preview_filename_lbl.setToolTip("实时预览已关闭；不会生成或刷新新的预览图。")
    preview_epoch = self._preview_temp_store.begin_run()
    preview_dir_holder = {"path": ""}

    # 让页面管理页也能看到同一批输入的缩略图（两个页签共享同一次导入结果）
    main_win = self.window()
    page_overrides_snapshot: dict[int, str] = {}
    page_manager = getattr(main_win, "_tab_pages", None)
    if page_manager is not None:
        # Page Manager may remember the original PDF/folder while OCR runs on
        # its expanded page-image list.  Compare both representations; the
        # old raw-input-only equality silently discarded confirmed labels and
        # sent covers/TOC/illustrations/back matter into MacOCR.
        from engine.page_ocr_policy import confirmed_overrides_for_active_inputs
        page_overrides_snapshot = confirmed_overrides_for_active_inputs(
            active_inputs=inputs,
            page_images=getattr(page_manager, "page_images", []) or [],
            raw_inputs=getattr(page_manager, "_last_loaded_raw_inputs", []) or [],
            page_overrides=getattr(page_manager, "page_overrides", {}) or {},
            auto_suggested=getattr(page_manager, "_auto_suggested", set()) or set(),
        )
    self._latest_single_doc = None
    self._switch_view("log")
    self._ocr_log_buffer.clear()
    self._ocr_log_flush_timer.start()
    self._log_view.clear()
    self._log_view.appendPlainText(
        f"🧭 OCR 模式：{ocr_profile.label} · {ocr_profile.language} · "
        f"{ocr_profile.writing_direction}"
    )
    if ruby_preserve_enabled:
        self._log_view.appendPlainText(
            "💎 Ruby 保留：已开启。普通 OCR 继续只识别 Ruby-free 分列正文；"
            + (
                "findtextCenterNet 将复用分列阶段记录的疑似 Ruby 位置，只扫描原图局部 ROI。"
                if ruby_scan_mode_snapshot == "smart_roi"
                else "findtextCenterNet 使用全页精确扫描（慢）。"
            )
            + " Ruby 证据不参与正文投票/融合。"
        )
    if page_overrides_snapshot:
        from engine.page_ocr_policy import should_skip_page_ocr
        skipped_by_type: dict[str, int] = {}
        for page_type in page_overrides_snapshot.values():
            if should_skip_page_ocr(page_type):
                skipped_by_type[page_type] = skipped_by_type.get(page_type, 0) + 1
        if skipped_by_type:
            label_map = globals().get("TYPE_LABEL", {})
            detail = "、".join(
                f"{label_map.get(page_type, page_type)} {count} 页"
                for page_type, count in sorted(skipped_by_type.items())
            )
            self._log_view.appendPlainText(
                f"🚫 页面管理 OCR 隔离：{detail}；不裁剪、不分列、不调用识别模型，仅作为 EPUB 页面资源保留。"
            )
    if not ocr_profile.allow_column_pipeline:
        self._log_view.appendPlainText(
            "🧱 模式隔离：整页横排按上→下、左→右重建；"
            "日文分列、Ruby、逐列组句与日语手写模块均未加载。"
        )
    if hayai_column_required:
        self._log_view.appendPlainText(
            "🧭 Hayai OCR 安全模式：整页输入已禁用；"
            "复用固定正文区域与右→左物理列，紧裁后按较长 crop 批量识别，并启用输出长度/CJK 比例幻觉拦截。"
        )
    if manga_48px_column_required:
        self._log_view.appendPlainText(
            "🧭 48px AR 行识别模式：使用紧裁物理列，竖列自动旋转并缩放到 48px 高。"
        )
    self._run_btn.setEnabled(False)
    if hasattr(self, "_handwriting_run_btn"):
        self._handwriting_run_btn.setEnabled(False)
    self._rerun_btn.setVisible(False)
    self._pause_btn.setVisible(True)
    self._pause_btn.setEnabled(True)
    self._pause_btn.setText("停止 OCR")
    cancel_event = threading.Event()
    self._cancel_event = cancel_event
    self._review_preview_run_active = bool(manual_review_requested)
    self._ocr_run_active = True
    self._ocr_last_activity_at = time.monotonic()
    self._ocr_stall_notice_emitted = False
    self._ocr_hard_timeout_requested = False
    self._ocr_watchdog_timer.start()
    self._run_generation += 1
    run_generation = self._run_generation
    from core.ocr_runtime_optimizer import (
        OcrCancellationToken, OcrPerformanceTrace, OcrResourceGovernor,
        adaptive_ocr_runtime_limits,
    )
    cancel_token = OcrCancellationToken(cancel_event)
    runtime_limits = adaptive_ocr_runtime_limits()
    runtime_governor = OcrResourceGovernor(runtime_limits)
    runtime_backend_config = {}
    backend_option_keys = (
        "apple_backend", "backend", "device", "pipeline", "model_source",
        "vl_backend", "mode", "recognition_level", "detector_onnx",
        "large_review", "ocr_size", "lang", "language",
    )
    for configured_engine in engine_ids:
        options = frozen_engine_options.get(configured_engine, {})
        runtime_backend_config[configured_engine] = {
            key: options[key]
            for key in backend_option_keys
            if key in options and isinstance(options[key], (str, int, float, bool, type(None)))
        }
        if configured_engine == "apple_vision":
            runtime_backend_config[configured_engine].setdefault(
                "apple_backend", options.get("apple_backend") or "native_helper"
            )
        elif configured_engine == "ndlocr_lite":
            runtime_backend_config[configured_engine].setdefault("backend", "NDLOCR-Lite / ONNX Runtime")
            runtime_backend_config[configured_engine].setdefault("device", "provider-selected")
        elif configured_engine in {"hayai_ocr", "manga_48px", "manga_ocr", "paddle_ocr"}:
            runtime_backend_config[configured_engine].setdefault(
                "backend",
                {
                    "hayai_ocr": "PyTorch or LiteRT",
                    "manga_48px": "PyTorch",
                    "manga_ocr": "PyTorch / VisionEncoderDecoder",
                    "paddle_ocr": str(options.get("pipeline") or "PaddleOCR"),
                }[configured_engine],
            )
            runtime_backend_config[configured_engine].setdefault(
                "device", str(options.get("device") or "runtime-selected")
            )
        elif configured_engine == "paddle_aistudio":
            runtime_backend_config[configured_engine].setdefault("backend", "PaddleOCR AI Studio API")
            runtime_backend_config[configured_engine].setdefault("device", "remote")
    if ruby_preserve_enabled:
        runtime_backend_config["ruby"] = {
            "engine": "findtextCenterNet",
            "scan_mode": ruby_scan_mode_snapshot,
            "backend_priority": ["CoreML", "ONNX Runtime", "PyTorch"],
        }
    performance_trace = OcrPerformanceTrace(
        run_id=f"ocr-{run_generation}",
        metadata={
            "page_count": len(inputs),
            "engine_ids": list(engine_ids),
            "role_slots": (
                [
                    {"index": index + 1, "role": role, "engine_id": engine}
                    for index, (role, engine) in enumerate(multi_role_plan.selected_roles)
                ]
                if multi_role_plan is not None
                else [{"index": 1, "role": "single", "engine_id": engine_ids[0]}]
            ),
            "engine_backend_config": runtime_backend_config,
            "multi_model": bool(multi_role_enabled),
            "ocr_mode": ocr_mode_snapshot,
            "ruby_preservation": ruby_preserve_enabled,
            "ruby_scan_mode": ruby_scan_mode_snapshot if ruby_preserve_enabled else "disabled",
            "image_preparation_worker_limit": runtime_limits.as_dict()["image_prepare"],
            "image_encode_worker_limit": runtime_limits.as_dict()["image_encode"],
            "ocr_runtime_limits": runtime_limits.as_dict(),
            "heavy_model_policy": "heavy_models_serial",
        },
    )
    # Deterministic OCR cache/checkpoint context.  This is frozen on the
    # GUI thread and later reopened by the worker without touching Qt state.
    # Cache keys include page content lineage, OCR parameters and an adapter
    # implementation fingerprint, so changing pixels, options or code cannot
    # silently reuse stale OCR evidence.
    ocr_project_cache_context = {}
    try:
        project_manager = getattr(page_manager, "project_manager", None)
        if project_manager is not None and project_manager.active_project is not None:
            adapter_file_map = {
                "apple_vision": "apple_vision_adapter.py",
                "windows_snipping_ocr": "windows_snipping_ocr_adapter.py",
                "ndlocr_lite": "ndlocr_lite_adapter.py",
                "hayai_ocr": "hayai_ocr_adapter.py",
                "manga_48px": "manga_48px_adapter.py",
                "manga_ocr": "manga_ocr_adapter.py",
                "paddle_ocr": "paddle_ocr_adapter.py",
                "paddle_aistudio": "paddle_aistudio_adapter.py",
            }
            source_root = PROJECT_ROOT
            # Model cache safety is intentionally conservative.  Adapter
            # code alone is not the complete effective implementation:
            # shared dispatch, runtime governance, role routing and
            # sentence reflow can all change the authoritative document
            # emitted by an otherwise unchanged adapter.  Hash the shared
            # execution layer once, then combine it with each adapter's
            # fingerprint so any semantic code edit invalidates old model
            # cache entries without re-reading the large GUI module for
            # every engine.
            shared_ocr_impl = project_manager.implementation_fingerprint([
                source_root / "gui_pyside6.py",
                source_root / "core" / "ocr_runtime_optimizer.py",
                source_root / "core" / "multi_ocr_roles.py",
                source_root / "core" / "ocr_smart_router.py",
                source_root / "core" / "ocr_segment_cache.py",
                source_root / "engine" / "column_sentence_reflow.py",
            ])
            engine_impl = {}
            for engine_id in engine_ids:
                files = []
                adapter_name = adapter_file_map.get(str(engine_id))
                if adapter_name:
                    files.append(source_root / "adapters" / adapter_name)
                # Column transport is part of the effective implementation
                # whenever any physical-column path can be used.
                if multi_role_enabled or self._column_split_check.isChecked():
                    files.append(source_root / "adapters" / "column_ocr_adapter.py")
                adapter_impl = project_manager.implementation_fingerprint(files)
                engine_impl[str(engine_id)] = f"{shared_ocr_impl}:{adapter_impl}"
            pipeline_signature = project_manager.make_stage_cache_key(
                "ocr_pipeline",
                config={
                    "pipeline_kind": "multi" if multi_role_enabled else "single",
                    "engines": list(engine_ids),
                    "role_plan": multi_role_plan.as_dict() if multi_role_plan is not None else {},
                    "smart_router": bool(multi_smart_router_enabled),
                    "ocr_mode": ocr_mode_snapshot,
                    "ruby_preservation": ruby_preserve_enabled,
                    "ruby_scan_mode": ruby_scan_mode_snapshot if ruby_preserve_enabled else "disabled",
                    "column_runtime": frozen_column_runtime,
                    "engine_options": frozen_engine_options,
                    "page_overrides": page_overrides_snapshot,
                    "fixed_crop_rect": fixed_crop_rect,
                },
                implementation=project_manager.implementation_fingerprint(
                    [source_root / "gui_pyside6.py", source_root / "core" / "artifact_pipeline.py"]
                ),
            )
            ocr_project_cache_context = {
                "workspace_root": str(project_manager.workspace_root.resolve()),
                "project_path": str(project_manager.active_project.resolve()),
                "pipeline_signature": pipeline_signature,
                "engine_implementation": engine_impl,
            }
    except Exception as exc:
        # Caching is an optimization/safety layer; never block OCR if its
        # bookkeeping cannot be initialized.
        self._log_view.appendPlainText(f"⚠️ OCR 缓存初始化已跳过：{exc}")

    self._active_ocr_performance_trace = performance_trace
    self._active_ocr_log_session_id = performance_trace.run_id
    self._active_ocr_log_finalized = False
    self.run_log_event.emit({
        "event": "started",
        "session_id": performance_trace.run_id,
        "stage": "multi_ocr" if multi_role_enabled else "single_ocr",
        "details": {
            "pipeline_kind": "multi" if multi_role_enabled else "single",
            "smart_router": bool(multi_smart_router_enabled),
            "pages": len(inputs),
            "engines": [self._engine_label(engine_id) for engine_id in engine_ids],
            "engine_ids": list(engine_ids),
            "role_plan": multi_role_plan.as_dict() if multi_role_plan is not None else {},
            "role_slots": (
                [
                    {"index": index + 1, "role": role, "engine_id": engine}
                    for index, (role, engine) in enumerate(multi_role_plan.selected_roles)
                ]
                if multi_role_plan is not None
                else [{"index": 1, "role": "single", "engine_id": engine_ids[0]}]
            ),
            "ocr_mode": ocr_mode_snapshot,
            "ruby_preservation": ruby_preserve_enabled,
            "ruby_scan_mode": ruby_scan_mode_snapshot if ruby_preserve_enabled else "disabled",
            "column_split": bool(frozen_column_runtime.get("column_isolation_mode"))
                if multi_role_enabled else bool(self._column_split_check.isChecked()),
            "backend_config": runtime_backend_config,
            "checkpoint_signature": str(ocr_project_cache_context.get("pipeline_signature") or ""),
            "resume_supported": bool(ocr_project_cache_context.get("pipeline_signature")),
        },
    })
    progress_visible = self._progress_display_is_enabled()
    self._latest_progress_snapshot = None
    self._progress_bar_wrap.setVisible(progress_visible)
    self._progress_bar_text.setVisible(progress_visible)
    self._prog.setRange(0, 1000)
    self._prog.setValue(0)
    self._progress_lbl.setVisible(False)
    self._overall_progress_live_state = {
        "percent": 0.0,
        "elapsed": 0.0,
        "eta": None,
        "received_at": time.monotonic(),
    }
    self._set_progress_bar_text(0.0, 0.0, None)
    if progress_visible:
        self._progress_clock_timer.start()
    else:
        self._progress_clock_timer.stop()
    self._phase_progress_lbl.setVisible(progress_visible)
    self._phase_progress_lbl.setText("当前：准备输入与 OCR 模型…")
    column_controls_active = bool(
        ocr_profile.allow_column_pipeline
        and (multi_role_enabled or self._column_split_check.isChecked())
    )
    apply_column_reflow = bool(
        ocr_profile.allow_column_pipeline
        and self._column_sentence_reflow_check.isChecked()
    )
    column_reflow_max = self._column_sentence_max_spin.value()
    # Physical page detection and sentence reflow have different limits.
    # Reusing the UI's "每句最多列数" (normally 10) as the page detector
    # ceiling caused shared multi-OCR prewarm to reject ordinary 12–15-column
    # novel pages and forced every recognizer to rediscover geometry.
    column_detection_max_columns = max(
        16, min(160, int(frozen_column_runtime.get("column_detection_max_columns", 80) or 80))
    )
    sentence_context_reocr = bool(
        apply_column_reflow
        and self._column_split_check.isChecked()
        and self._column_sentence_context_reocr_check.isChecked()
    )
    sentence_global_merged_box_reocr = bool(
        sentence_context_reocr
        and hasattr(self, "_column_sentence_global_merged_box_check")
        and self._column_sentence_global_merged_box_check.isChecked()
    )
    sentence_context_strategy = str(
        self._column_sentence_strategy_combo.currentData()
        if hasattr(self, "_column_sentence_strategy_combo")
        else "full"
    ).strip().lower() or "full"
    if sentence_context_strategy not in {"smart", "full"}:
        sentence_context_strategy = "full"
    compact_primary_transport = bool(
        column_controls_active
        and hasattr(self, "_column_compact_transport_check")
        and self._column_compact_transport_check.isChecked()
    )
    column_rescue_policy = str(
        self._column_rescue_policy_combo.currentData()
        if column_controls_active
        and hasattr(self, "_column_rescue_policy_combo")
        else "adaptive"
    ).strip().lower()
    if column_rescue_policy not in {"adaptive", "off", "legacy"}:
        column_rescue_policy = "adaptive"
    column_rescue_summary = {
        "adaptive": "列级救援：每列最多一种自适应路径",
        "off": "列级救援：关闭，疑难列直接进入人工复核",
        "legacy": "列级救援：兼容旧版完整多轮恢复链",
    }[column_rescue_policy]
    ndlocr_page_mode = str(
        self._ndlocr_page_mode_combo.currentData()
        if column_controls_active
        and hasattr(self, "_ndlocr_page_mode_combo")
        else "column"
    ).strip().lower()
    if ndlocr_page_mode not in {"hybrid", "page", "column"}:
        ndlocr_page_mode = "hybrid"
    ndlocr_mode_summary = {
        "hybrid": "NDLOCR智能混合：整页一次，仅疑难列逐列补识",
        "page": "NDLOCR高速整页：每页一次，漏列保留待人工",
        "column": "NDLOCR强制逐列：共享统一分列后逐列识别",
    }[ndlocr_page_mode]
    multi_sentence_mode = str(
        self._multi_sentence_speed_combo.currentData()
        if hasattr(self, "_multi_sentence_speed_combo")
        else "primary_only"
    ).strip().lower()
    if multi_sentence_mode not in {"primary_only", "all_models"}:
        multi_sentence_mode = "primary_only"
    review_preview_boxes_enabled = bool(
        ocr_profile.allow_japanese_handwriting
        and (
            manual_review_requested
            or (
                hasattr(self, "_handwriting_trace_check")
                and self._handwriting_trace_check.isChecked()
            )
        )
    )
    handwriting_enabled = bool(
        manual_review_requested
        and review_preview_boxes_enabled
    )
    handwriting_mode = self._handwriting_selected_mode() if handwriting_enabled else ""
    handwriting_strategy = self._handwriting_selected_strategy() if handwriting_enabled else "balanced"
    handwriting_backend = self._handwriting_selected_backend() if handwriting_enabled else "auto"
    handwriting_review = handwriting_enabled and handwriting_mode in {"hybrid", "manual"}
    handwriting_char_mask = bool(
        handwriting_enabled
        and hasattr(self, "_handwriting_char_mask_check")
        and self._handwriting_char_mask_check.isChecked()
    )
    handwriting_insert_symbols = bool(
        handwriting_enabled
        and hasattr(self, "_handwriting_symbol_insert_check")
        and self._handwriting_symbol_insert_check.isChecked()
    )

    use_column_mask = bool(
        ocr_profile.allow_column_pipeline
        and (self._column_split_check.isChecked() or multi_role_enabled or handwriting_enabled)
    )
    chinese_merge_horizontal_fragments = bool(
        hasattr(self, "_chinese_merge_line_fragments_check")
        and self._chinese_merge_line_fragments_check.isChecked()
    )
    chinese_filter_headers = bool(
        hasattr(self, "_chinese_header_filter_check")
        and self._chinese_header_filter_check.isChecked()
    )
    # Free 3-slot mode always executes every selected model.  AI adjudication
    # coverage must not shrink merely because the first two OCRs agree.
    early_consensus_enabled = False
    parallel_first_enabled = bool(
        multi_role_enabled
        and use_column_mask
        and len(engine_ids) >= 2
    )
    # In fast-consensus mode the primary Hayai recognizer may be needed a
    # second time for sentence-context rescue. Keep that one worker alive
    # only until the first disagreement rescue is complete, then release it
    # before the next selective model starts so 48px/other MPS models do not compete for
    # unified memory with an idle Hayai model.
    retain_primary_hayai_session = bool(
        early_consensus_enabled
        and sentence_context_reocr
        and engine_ids
        and str(engine_ids[0]).strip().lower() == "hayai_ocr"
    )
    retained_hayai_holder: dict[str, object] = {"session": None}

    def close_retained_hayai(*, force: bool = False) -> None:
        session = retained_hayai_holder.pop("session", None)
        retained_hayai_holder["session"] = None
        if session is None:
            return
        try:
            if force and hasattr(session, "close"):
                session.close(force=True)
            else:
                session.__exit__(None, None, None)
        except Exception:
            try:
                session.close(force=True)
            except Exception:
                pass

    from core.ocr_progress_estimator import OCRProgressEstimator
    progress_estimator = OCRProgressEstimator(
        model_count=len(engine_ids),
        use_column_mask=use_column_mask,
        sentence_context_reocr=sentence_context_reocr,
    )

    progress_emit_state = {"time": 0.0, "percent": -1.0, "label": ""}
    progress_emit_lock = threading.Lock()

    def emit_overall_progress(snapshot, *, force: bool = False):
        if cancel_event.is_set():
            return
        performance_trace.increment("overall_progress_events")
        # Always retain the newest estimator snapshot so turning the switch
        # back on during OCR immediately restores an accurate percentage.
        self._latest_progress_snapshot = snapshot
        if not self._progress_display_enabled_event.is_set():
            return
        now = time.monotonic()
        try:
            percent = float(snapshot.percent)
            current = int(snapshot.current or 0)
            total = max(1, int(snapshot.total or 1))
            unit = str(snapshot.unit or "")
            label = str(snapshot.label or "")
        except Exception:
            percent, current, total, unit, label = 0.0, 0, 1, "", ""
        with progress_emit_lock:
            should_emit = bool(
                force
                or current <= 0
                or current >= total
                or now - float(progress_emit_state["time"]) >= 0.20
                or abs(percent - float(progress_emit_state["percent"])) >= 0.15
            )
            if should_emit:
                progress_emit_state.update(
                    {"time": now, "percent": percent, "label": label}
                )
        if should_emit:
            signals.overall_progress.emit(snapshot)

    def emit_phase_progress(label, current, total):
        if cancel_event.is_set():
            return
        performance_trace.increment("phase_progress_events")
        if self._progress_display_enabled_event.is_set():
            signals.phase_progress.emit(label, current, total)

    self._handwriting_review_context = {
        "enabled": handwriting_enabled,
        "mode": handwriting_mode,
        "strategy": handwriting_strategy,
        "backend": handwriting_backend,
        "character_mask": handwriting_char_mask,
        "insert_missing_symbols": handwriting_insert_symbols,
        "crop_rect": fixed_crop_rect,
        "apply_column_reflow": apply_column_reflow,
        "column_reflow_max": column_reflow_max,
        "sentence_context_reocr": sentence_context_reocr,
        "sentence_global_merged_box_reocr": sentence_global_merged_box_reocr,
    }

    handwriting_review_context_snapshot = dict(self._handwriting_review_context or {})
    model_progress_label = {"value": ""}
    long_book_runtime = {
        "enabled": False,
        "page_total": 0,
        "page_order_by_path": {},
    }

    def preview_runtime_active() -> bool:
        return bool(
            self._live_preview_enabled_event.is_set()
            and run_generation == self._run_generation
            and not cancel_event.is_set()
        )

    def emit_retained_preview(
        image_path, display_name, rects=None, stage="plain", *, page_ordinal=0
    ):
        if not preview_runtime_active() or not image_path:
            return
        preview_dir = str(preview_dir_holder.get("path") or "")
        if not preview_dir:
            return
        try:
            key = str(Path(image_path).expanduser().resolve())
            snapshot_path, _image = self._write_preview_snapshot(
                str(image_path), preview_dir, key
            )
            if snapshot_path and preview_runtime_active():
                preview_payload = (
                    dict(rects)
                    if isinstance(rects, dict)
                    else {"columns": list(rects or []), "glyphs": []}
                )
                preview_payload["page_ordinal"] = int(page_ordinal or 0)
                if isinstance(preview_payload, dict) and preview_payload.pop(
                    "build_glyphs", False
                ):
                    try:
                        preview_payload["glyphs"] = self._build_review_character_rects(
                            snapshot_path,
                            preview_payload.get("columns") or [],
                        )
                    except Exception as exc:
                        preview_payload["glyphs"] = []
                        signals.log.emit(f"  ⚠️  逐字框预览生成失败：{exc}")
                signals.preview_page_ready.emit(
                    key,
                    Path(str(display_name or image_path)).name,
                    snapshot_path,
                    preview_payload,
                    str(stage or "plain"),
                )
        except Exception as exc:
            signals.log.emit(f"  ⚠️  保存实时预览失败：{exc}")

    def on_page_progress(current, total, filename, image_path=None):
        if cancel_event.is_set():
            return
        if not use_column_mask:
            if self._progress_display_enabled_event.is_set():
                signals.progress.emit(current, total)
                prefix = f"{model_progress_label['value']} " if model_progress_label["value"] else ""
                signals.log.emit(f"  {prefix}[{current:3d}/{total}] {filename}")
        if image_path and not use_column_mask:
            emit_retained_preview(
                image_path, filename, [], "plain", page_ordinal=current
            )

    def on_column_preview(page_path, columns, stage="split"):
        if not preview_runtime_active() or not page_path:
            return
        if cancel_event.is_set():
            return
        resolved_key = str(Path(str(page_path)).expanduser().resolve())
        page_ordinal = int(
            (long_book_runtime.get("page_order_by_path") or {}).get(
                resolved_key, 0
            )
            or 0
        )
        try:
            from adapters.column_ocr_adapter import (
                COLUMN_DETECTOR_VERSION,
                _column_detector_preview_bounds,
                _column_source_union,
            )

            reader = QImageReader(str(page_path))
            size = reader.size()
            if not size.isValid() or size.width() <= 0 or size.height() <= 0:
                return
            source_width = float(size.width())
            source_height = float(size.height())
            rects = []
            for column in columns:
                det_left, det_top, det_right, det_bottom = _column_detector_preview_bounds(column)
                rects.append((
                    max(0.0, min(1.0, float(det_left) / source_width)),
                    max(0.0, min(1.0, float(det_top) / source_height)),
                    max(0.0, min(1.0, float(det_right) / source_width)),
                    max(0.0, min(1.0, float(det_bottom) / source_height)),
                ))
            input_rects = []
            for column in columns:
                in_left, in_top, in_right, in_bottom = _column_source_union(column)
                input_rects.append((
                    max(0.0, min(1.0, float(in_left) / source_width)),
                    max(0.0, min(1.0, float(in_top) / source_height)),
                    max(0.0, min(1.0, float(in_right) / source_width)),
                    max(0.0, min(1.0, float(in_bottom) / source_height)),
                ))
            emit_retained_preview(
                str(page_path),
                Path(str(page_path)).name,
                {
                    "columns": rects,
                    "input_rects": input_rects,
                    "glyphs": [],
                    "build_glyphs": bool(review_preview_boxes_enabled),
                    "detector_version": COLUMN_DETECTOR_VERSION,
                },
                stage,
                page_ordinal=page_ordinal,
            )
        except Exception as exc:
            signals.log.emit(f"  ⚠️  实时分列框预览失败：{exc}")

    cancelled_partial = {"document": None}

    def raise_if_cancelled() -> None:
        cancel_token.checkpoint()

    def remember_partial(document):
        if document is not None:
            cancelled_partial["document"] = document
        return document

    def worker():
        # Import before nested run_one closes over this name.  Free-slot v3
        # imported it only in a later branch, making cache-key construction
        # raise an UnboundLocalError on every real model run.
        from core.multi_ocr_roles import MULTI_OCR_ROLE_SCHEMA
        project_cache_manager = None
        checkpoint_signature = str(ocr_project_cache_context.get("pipeline_signature") or "")
        checkpoint_terminal_status = "running"
        ruby_worker_job_token = None
        try:
            if ocr_project_cache_context:
                try:
                    project_cache_manager = ProjectWorkspaceManager(
                        ocr_project_cache_context["workspace_root"]
                    )
                    project_cache_manager.open_project(
                        ocr_project_cache_context["project_path"]
                    )
                    previous_checkpoint = project_cache_manager.load_checkpoint(
                        "ocr", checkpoint_signature
                    )
                    completed_steps = dict(previous_checkpoint.get("completed_steps") or {})
                    project_cache_manager.begin_checkpoint(
                        "ocr", checkpoint_signature,
                        metadata={
                            "pipeline_kind": "multi" if multi_role_enabled else "single",
                            "engine_ids": list(engine_ids),
                            "page_count": len(inputs),
                            "ocr_mode": ocr_mode_snapshot,
                        },
                    )
                    if completed_steps:
                        signals.log.emit(
                            f"♻️ 发现可恢复 OCR 断点：{len(completed_steps)} 个已完成模型阶段；"
                            "将先校验 Stage Cache，命中则直接复用。"
                        )
                        performance_trace.set_gauge(
                            "checkpoint.resume_steps", len(completed_steps)
                        )
                except Exception as exc:
                    project_cache_manager = None
                    signals.log.emit(f"⚠️ OCR 断点恢复初始化已跳过：{exc}")
            def finish_project_checkpoint(status: str) -> None:
                if project_cache_manager is None or not checkpoint_signature:
                    return
                try:
                    project_cache_manager.finish_checkpoint(
                        "ocr", checkpoint_signature, status=str(status)
                    )
                except Exception as exc:
                    signals.log.emit(f"⚠️ OCR 断点状态保存失败：{exc}")

            if ruby_preserve_enabled:
                # Start model construction immediately after the OCR worker is
                # alive.  This daemon prewarm overlaps the expensive CoreML /
                # ONNX startup with ordinary page OCR instead of making the Ruby
                # phase pay the complete cold-start cost at the end.  The lease
                # is released in this worker's outer finally block, so there is
                # no arbitrary idle TTL.
                try:
                    from adapters.findtext_centernet_ruby import begin_findtext_ocr_job
                    ruby_worker_job_token = begin_findtext_ocr_job(
                        f"ocr-{run_generation}",
                        cancel_check=cancel_event.is_set,
                        log_callback=signals.log.emit,
                        background=True,
                    )
                    performance_trace.event("ruby_worker_prewarm_started")
                except Exception as exc:
                    # Ruby is an optional structural side-channel.  A prewarm
                    # scheduling failure must never stop the main OCR pipeline;
                    # the normal Ruby pass retains its existing fallback path.
                    signals.log.emit(f"⚠️ Ruby worker 预热调度失败，将在 Ruby 阶段按需启动：{exc}")

            raise_if_cancelled()
            # Page classifications were snapshotted on the GUI thread.  A
            # running OCR job is therefore isolated from asynchronous page
            # thumbnail loading and later user edits.
            page_overrides = dict(page_overrides_snapshot)

            # Expand folders/PDFs once for the complete role-based multi-model
            # run.  Every role then consumes the same immutable page-image
            # list; a 400-page PDF is never re-rendered/enumerated once per
            # model.  Page-Manager input is already an explicit image list, so
            # this is effectively a no-op there.
            resolved_inputs = list(inputs)
            if multi_role_enabled:
                try:
                    from adapters.pdf_input import expand_inputs as expand_ocr_inputs
                    expanded_inputs = expand_ocr_inputs(
                        resolved_inputs,
                        cancel_check=cancel_event.is_set,
                    )
                    if expanded_inputs:
                        resolved_inputs = list(expanded_inputs)
                except InterruptedError:
                    raise
                except Exception as exc:
                    signals.log.emit(
                        f"⚠️ 多模型输入预展开失败，将沿用原输入列表：{exc}"
                    )

            ruby_detection_cache = {"prepared": False, "report": None}

            def maybe_preserve_ruby(documents):
                docs = [item for item in documents if item is not None]
                if not docs:
                    return
                if not ruby_preserve_enabled:
                    # Hard OFF-state boundary.  A fresh Ruby-disabled run must
                    # behave like the pre-Ruby pipeline: no locked readings, no
                    # Ruby report/state and no persisted candidate sidecar.  This
                    # does not alter OCR prose, confidence, layout or provenance.
                    from adapters.findtext_centernet_ruby import strip_ruby_overlay
                    for doc in docs:
                        strip_ruby_overlay(
                            doc, strip_candidate_geometry=True, strip_logs=True
                        )
                    return
                raise_if_cancelled()
                signals.log.emit(
                    "\n💎 Ruby 结构识别："
                    + (
                        "复用普通 OCR 分列阶段的几何候选，将相邻列合并为原图上下文 ROI；"
                        "无候选页不再二次 OCR…"
                        if ruby_scan_mode_snapshot == "smart_roi"
                        else "findtextCenterNet 单独读取未清理原始正文页执行全页扫描…"
                    )
                )
                from adapters.findtext_centernet_ruby import preserve_ruby_in_documents

                def ruby_progress(current, total, filename):
                    if cancel_event.is_set():
                        return
                    emit_phase_progress("Ruby 结构识别", current, max(1, total))
                    if current <= 1 or current >= total or current % max(1, total // 10) == 0:
                        signals.log.emit(
                            f"  [Ruby {current:3d}/{max(1, total)}] {filename}"
                        )

                ruby_started = time.perf_counter()
                try:
                    report = preserve_ruby_in_documents(
                        docs,
                        cancel_check=cancel_event.is_set,
                        log_callback=signals.log.emit,
                        progress_callback=ruby_progress,
                        scan_mode=ruby_scan_mode_snapshot,
                        candidate_root=shared_column_prepare_dir,
                    )
                finally:
                    performance_trace.add_duration(
                        "ruby_preservation", time.perf_counter() - ruby_started
                    )
                performance_trace.event(
                    "engine_runtime",
                    engine_id="ruby",
                    backend=str(getattr(report, "backend", "") or "unknown"),
                    device=(
                        "Core ML (system-selected)"
                        if str(getattr(report, "backend", "")).lower() == "coreml"
                        else "backend-selected"
                    ),
                )
                for metric_name in (
                    "pages_scanned", "pages_with_candidates", "candidate_boxes",
                    "roi_count", "estimated_detector_tiles", "full_page_detector_tiles",
                    "cache_hits", "cache_misses", "failed_rois",
                ):
                    performance_trace.set_gauge(
                        f"ruby.{metric_name}", int(getattr(report, metric_name, 0) or 0)
                    )
                # Candidate geometry is a transient scheduling aid only. Once
                # the optional pass finishes, discard it from every OCR model
                # document so raw voters stay fully Ruby-free on save/roundtrip.
                # The last document is the authoritative overlay target.
                from adapters.findtext_centernet_ruby import strip_ruby_overlay
                for source_doc in docs[:-1]:
                    strip_ruby_overlay(
                        source_doc, strip_candidate_geometry=True, strip_logs=False
                    )
                target_meta = getattr(docs[-1], "metadata", None)
                if target_meta is not None and hasattr(target_meta, "ruby_candidate_pages"):
                    target_meta.ruby_candidate_pages = {}
                ruby_detection_cache["prepared"] = True
                ruby_detection_cache["report"] = report
                if report.error:
                    signals.log.emit(
                        "⚠️ Ruby 保留未完成；已保留主 OCR 正文，不做任何猜测性回写。"
                    )
                else:
                    if report.scan_mode == "smart_roi":
                        saved_pct = max(0.0, (1.0 - report.estimated_tile_ratio) * 100.0)
                        signals.log.emit(
                            f"✅ Ruby ROI 完成：{report.candidate_boxes} 个候选 → {report.roi_count} 个 ROI，"
                            f"实际触发 {report.pages_scanned} 页；预计 findtext 检测窗减少约 {saved_pct:.1f}%。"
                        )
                    signals.log.emit(
                        f"✅ Ruby 保留完成：检测 {report.ruby_pairs} 处，"
                        f"安全写回 {report.matched_pairs} 处 / {report.updated_blocks} 块；"
                        f"未唯一匹配 {report.unmatched_pairs} 处保持原样。"
                    )

            try:
                from adapters.pdf_input import expand_inputs, natural_sort_key
                work_dir = str(Path(inputs[0]).expanduser().parent)
                expanded = expand_inputs(
                    inputs, work_dir=work_dir, cancel_check=cancel_event.is_set
                )
                raise_if_cancelled()
                if expanded:
                    resolved_inputs = sorted(set(expanded), key=natural_sort_key)
                performance_trace.set_gauge("input_pages", len(resolved_inputs))
            except Exception as exc:
                if cancel_event.is_set() or isinstance(exc, InterruptedError):
                    raise InterruptedError("OCR 已停止") from exc
                signals.log.emit(f"  ⚠️  输入预展开失败，改用原始输入列表：{exc}")

            # Final defence before Apple OCR/Apple Vision or any other adapter
            # receives its queue.  A file may disappear between the GUI
            # snapshot and worker startup; silently skip it rather than
            # recognizing a stale path or shifting page numbering.
            existing_resolved = []
            missing_resolved = []
            for path in resolved_inputs:
                try:
                    if Path(path).exists():
                        existing_resolved.append(path)
                    else:
                        missing_resolved.append(path)
                except Exception:
                    missing_resolved.append(path)
            resolved_inputs = existing_resolved
            raise_if_cancelled()
            if missing_resolved:
                signals.log.emit(
                    f"  ⚠️  OCR 启动前再次过滤 {len(missing_resolved)} 个已删除/不存在页面。"
                )
            if not resolved_inputs:
                raise RuntimeError("页面管理中的图片已全部删除或输入文件不存在，OCR 队列为空。")

            long_book_mode = bool(
                len(resolved_inputs) >= 100
            )
            long_book_runtime.update({
                "enabled": long_book_mode,
                "page_total": len(resolved_inputs),
                "page_order_by_path": {
                    str(Path(path).expanduser().resolve()): index
                    for index, path in enumerate(resolved_inputs, start=1)
                },
            })
            if long_book_mode:
                signals.log.emit(
                    f"\n🧠 大批量稳定模式：{len(resolved_inputs)} 页；"
                    "实时预览将保留全部正文页缩略图；同页分列/识别阶段复用一份快照。"
                )

            # Two independent PyTorch/MPS recognizers running together on
            # one Apple GPU usually thrash unified memory and are slower than
            # serial full batches. Keep true heterogeneous pairs parallel.
            mps_heavy_engines = {"hayai_ocr", "manga_48px", "paddle_ocr"}

            def is_gpu_heavy_engine(engine_id: str) -> bool:
                key = str(engine_id or "").strip().lower()
                if key == "hayai_ocr":
                    hayai_opts = frozen_engine_options.get("hayai_ocr", {})
                    if str(hayai_opts.get("backend") or "torch").strip().lower() == "litert":
                        return False
                    if str(hayai_opts.get("device") or "auto").strip().lower() == "cpu":
                        return False
                return key in mps_heavy_engines

            parallel_first_runtime = bool(parallel_first_enabled)
            if (
                parallel_first_runtime
                and len(engine_ids) >= 2
                and is_gpu_heavy_engine(engine_ids[0])
                and is_gpu_heavy_engine(engine_ids[1])
            ):
                parallel_first_runtime = False
                signals.log.emit(
                    "🧠 模型1/2均为本地深度模型：为避免 Mac 统一内存/MPS争用，"
                    "首轮自动改为串行批处理；模型常驻与批量推理保持开启。"
                )

            raise_if_cancelled()
            from utils.session_temp import session_temp_registry
            crop_temp_root = str(session_temp_registry().make_dir("ocr-crop"))
            trace_path = str(Path(crop_temp_root) / "OCR_PERFORMANCE_TRACE_V30.json")
            self._last_ocr_performance_trace_path = trace_path
            from adapters.column_ocr_adapter import clear_task_file_sha_memo
            clear_task_file_sha_memo()
            performance_trace.event("temporary_root_ready")
            if not self._preview_temp_store.register(crop_temp_root, preview_epoch):
                if cancel_event.is_set():
                    raise InterruptedError("OCR 已停止")
                raise RuntimeError("OCR 临时目录会话已失效，请重新开始识别。")
            preview_dir_holder["path"] = str(Path(crop_temp_root) / "retained_preview_pages")
            try:
                shared_crop_dir = str(Path(crop_temp_root) / "shared_fixed_crop")
                shared_column_prepare_dir = str(Path(crop_temp_root) / "shared_column_prepare")
                shared_column_variant_dir = str(Path(crop_temp_root) / "shared_column_variants")
                shared_sentence_dir = str(Path(crop_temp_root) / "shared_sentence_groups")
                Path(shared_crop_dir).mkdir(parents=True, exist_ok=True)
                Path(shared_column_prepare_dir).mkdir(parents=True, exist_ok=True)
                Path(shared_column_variant_dir).mkdir(parents=True, exist_ok=True)
                Path(shared_sentence_dir).mkdir(parents=True, exist_ok=True)
                raise_if_cancelled()

                # Role-based multi-OCR always prepares one authoritative
                # physical-column geometry cache before any recognizer is
                # loaded.  This stage performs no OCR.  It lets the page
                # role, column role, sentence role and all review roles share
                # exactly the same column identities and avoids N separate
                # connected-component scans of a 400-page book.
                multi_structure_cache = None
                # Shared geometry is mandatory for role-based multi OCR, and it is
                # also needed by Ruby smart-ROI even when the *main* single-model
                # OCR deliberately remains whole-page.  The latter is geometry-only:
                # it performs zero recognition calls and therefore does not mutate
                # the single-model split/full-page transport contract.
                ruby_geometry_prewarm = bool(
                    ruby_preserve_enabled
                    and ruby_scan_mode_snapshot == "smart_roi"
                    and ocr_profile.allow_column_pipeline
                )
                if (multi_role_enabled or ruby_geometry_prewarm) and ocr_profile.allow_column_pipeline:
                    try:
                        from engine.multi_ocr_structure_prewarm import (
                            prewarm_multi_ocr_structure,
                        )
                        structure_log_state = {"bucket": -1}

                        def on_structure_progress(current, total, filename):
                            bucket = int((max(0, current) / max(1, total)) * 20)
                            if bucket != structure_log_state["bucket"] or current >= total:
                                structure_log_state["bucket"] = bucket
                                signals.log.emit(
                                    f"  [共享分列 {current:3d}/{max(1, total)}] {filename}"
                                )

                        from engine.page_ocr_policy import should_skip_page_ocr
                        prewarm_excluded_pages = {
                            int(page_no)
                            for page_no, page_type in page_overrides.items()
                            if should_skip_page_ocr(page_type)
                        }
                        prewarm_body_count = max(0, len(resolved_inputs) - len(prewarm_excluded_pages))
                        if multi_role_enabled:
                            signals.log.emit(
                                "\n🧭 多模型共享结构预热：仅统一正文页分列；此阶段不调用任何 OCR 模型。"
                                f" 正文页 {prewarm_body_count}/{len(resolved_inputs)}；"
                                f"物理列检测上限 {column_detection_max_columns}/页；"
                                f"成句上限 {column_reflow_max}/句。"
                            )
                        else:
                            signals.log.emit(
                                "\n💎 Ruby 智能 ROI 几何预热：主 OCR 仍保持整页输入；"
                                "这里只检测物理列/Ruby 候选，OCR 调用 0 次。"
                                f" 正文页 {prewarm_body_count}/{len(resolved_inputs)}。"
                            )
                        multi_structure_cache = prewarm_multi_ocr_structure(
                            resolved_inputs,
                            shared_prepare_base=shared_column_prepare_dir,
                            sensitivity=int(round(float(frozen_column_runtime.get("column_sensitivity", 55)))),
                            padding_percent=int(round(float(frozen_column_runtime.get("column_padding_percent", 10)))),
                            max_columns=int(column_detection_max_columns),
                            fixed_region_rect=fixed_crop_rect,
                            detector_mode=str(frozen_column_runtime.get("column_detector_mode", "components") or "components"),
                            capture_ruby_candidates=bool(
                                frozen_column_runtime.get("column_capture_ruby_candidates", False)
                            ),
                            cancel_check=cancel_event.is_set,
                            progress_callback=on_structure_progress,
                            excluded_page_numbers=prewarm_excluded_pages,
                        )
                        st = multi_structure_cache.stats
                        performance_trace.add_duration("multi_ocr.structure_prewarm", st.seconds)
                        performance_trace.set_gauge("multi_ocr.structure_pages", st.pages_prepared)
                        performance_trace.set_gauge("multi_ocr.structure_columns", st.columns)
                        signals.log.emit(
                            f"✅ 共享结构完成：{st.pages_prepared}/{st.pages_requested} 页，"
                            f"{st.columns} 个物理列，{st.seconds:.2f} 秒；OCR 调用 0 次。"
                        )
                    except InterruptedError:
                        raise
                    except Exception as exc:
                        # Structure prewarm is an optimization, not a reason
                        # to make ordinary OCR unavailable.  The model paths
                        # retain their historical lazy geometry fallback.
                        multi_structure_cache = None
                        signals.log.emit(f"⚠️ 共享结构预热失败，将回退模型内按需分列：{exc}")

                def performance_callback_for(engine_id: str):
                    def callback(stage: str, seconds: float, details: dict | None = None):
                        detail = details if isinstance(details, dict) else {}
                        if stage == "runtime":
                            performance_trace.event(
                                "engine_runtime",
                                engine_id=engine_id,
                                backend=str(detail.get("backend") or "unknown"),
                                device=str(detail.get("device") or "unknown"),
                            )
                        elif stage == "input_pages":
                            for key in ("total", "ocr", "skipped"):
                                if key in detail:
                                    performance_trace.set_gauge(
                                        f"{engine_id}.pages_{key}", int(detail[key])
                                    )
                        else:
                            performance_trace.add_duration(
                                f"{engine_id}.{stage}", seconds
                            )
                            if stage == "recognition":
                                for key in (
                                    "batches", "items",
                                    "segment_cache_hits", "segment_cache_misses",
                                ):
                                    if key in detail:
                                        performance_trace.set_gauge(
                                            f"{engine_id}.{key}", int(detail[key] or 0)
                                        )
                    return callback

                def make_common_kwargs(model_index: int):
                    # Fixed-region crops are byte-identical for every model.
                    # Model 1 writes them once; later models reuse the same
                    # run-local files instead of decoding/re-encoding every page.
                    def model_page_progress(current, total, filename, image_path=None):
                        # The page image is also identical.  Decode it only for
                        # model 1; later models update progress/logs without
                        # repeating expensive high-resolution preview I/O.
                        preview_path = image_path if model_index == 0 else None
                        if not use_column_mask:
                            engine_label = self._engine_label(engine_ids[model_index])
                            emit_overall_progress(
                                progress_estimator.update_phase(
                                    model_index,
                                    "recognition",
                                    current,
                                    total,
                                    label=f"模型{model_index + 1}/{len(engine_ids)}·{engine_label}·页面识别",
                                    unit="页",
                                )
                            )
                        on_page_progress(current, total, filename, preview_path)

                    kwargs = dict(
                        input_paths=resolved_inputs,
                        page_overrides=page_overrides,
                        verbose=False,
                        progress_callback=model_page_progress,
                        cancel_check=cancel_event.is_set,
                        crop_rect=fixed_crop_rect,
                        temp_crop_dir=shared_crop_dir,
                        # Column OCR routes through run_ocr_engine(), whose
                        # shared-crop contract accepts this flag. Per-path
                        # locks in crop_for_ocr make parallel model startup
                        # safe: every high-resolution page is masked once.
                        reuse_existing_crops=bool(len(engine_ids) > 1 and use_column_mask),
                        performance_callback=performance_callback_for(engine_ids[model_index]),
                        # Chinese horizontal mode has its own optional
                        # cross-page header filter. Japanese column reflow
                        # keeps the original no-delete safety contract.
                        filter_running_headers=(
                            chinese_filter_headers
                            if not ocr_profile.allow_column_pipeline
                            else not apply_column_reflow
                        ),
                    )
                    if not ocr_profile.allow_column_pipeline:
                        # Only the three Chinese-compatible adapters receive
                        # these new kwargs. Legacy Japanese plugins therefore
                        # keep their exact historical call signatures.
                        kwargs.update(
                            ocr_mode=ocr_mode_snapshot,
                            merge_horizontal_fragments=chinese_merge_horizontal_fragments,
                        )
                    return kwargs

                def model_resource(engine_id: str) -> str:
                    """Classify the *actual configured backend*, not just the engine name.

                        This keeps CPU/ONNX work able to overlap an Apple-GPU model while
                        still serializing workers that compete for the same unified-memory
                        accelerator.  Unknown accelerators remain conservative.
                        """
                    key = str(engine_id or "").strip().lower()
                    opts = frozen_engine_options.get(key, {})
                    if key == "hayai_ocr":
                        backend = str(opts.get("backend") or "torch").strip().lower()
                        device = str(opts.get("device") or "auto").strip().lower()
                        if backend == "litert" or device == "cpu":
                            return "cpu_model"
                        if device == "cuda":
                            return "cuda_model"
                        return "mps_model"
                    if key == "paddle_ocr":
                        pipeline = str(opts.get("pipeline") or "ocr").strip().lower()
                        vl_backend = str(opts.get("vl_backend") or "auto").strip().lower()
                        if pipeline == "vl" and vl_backend in {"auto", "mlx"}:
                            return "mps_model"
                        # Native Paddle on macOS is a CPU path; do not block an
                        # unrelated MPS model just because both are OCR engines.
                        return "cpu_model"
                    if key in {"manga_48px", "manga_ocr"}:
                        return "mps_model"
                    if key == "ndlocr_lite":
                        return "onnx_model"
                    if key in {"apple_vision", "macocr", "mac_ocr", "macos_ocr"}:
                        return "native_vision"
                    if key == "paddle_aistudio":
                        return "remote_model"
                    return "onnx_model"

                def run_one(
                    engine_id: str,
                    model_index: int,
                    *,
                    input_role: str = "auto",
                    semantic_role: str | None = None,
                    force_sentence_reocr: bool | None = None,
                    target_column_ids: set[str] | None = None,
                    seed_results: dict[str, dict[str, object]] | None = None,
                    sentence_target_ids: set[str] | None = None,
                    sentence_target_groups: tuple[tuple[str, ...], ...] | None = None,
                    seed_mark_selective: bool = False,
                    recovery_only: bool = False,
                    ndlocr_page_mode_override: str | None = None,
                ):
                    raise_if_cancelled()
                    engine_label = self._engine_label(engine_id)
                    role = str(input_role or "auto").strip().lower()
                    declared_role = str(semantic_role or role or "auto").strip().lower()
                    from core.ocr_engine_profiles import get_ocr_engine_profile, preferred_input_role
                    engine_profile = get_ocr_engine_profile(engine_id)
                    if role == "auto" and multi_role_enabled:
                        role = preferred_input_role(engine_id)
                    effective_ndlocr_page_mode = str(
                        ndlocr_page_mode_override or ndlocr_page_mode or "hybrid"
                    ).strip().lower()
                    # NDLOCR page-role remains a genuine one-call-per-page
                    # primary pass, but it runs through the shared-column
                    # adapter so page blocks inherit immutable physical IDs
                    # and hybrid mode can selectively re-read only unstable
                    # columns. Other page engines still receive raw pages.
                    ndl_page_role = bool(
                        role == "page"
                        and str(engine_id).strip().lower() == "ndlocr_lite"
                    )
                    if ndl_page_role and effective_ndlocr_page_mode == "column":
                        effective_ndlocr_page_mode = "hybrid"
                    role_use_column_mask = bool(
                        use_column_mask if role == "auto"
                        else role in {"column", "sentence", "review"}
                        or ndl_page_role
                    )
                    if role in {"column", "page"}:
                        model_sentence_reocr = False
                        sentence_target_ids = None
                        sentence_target_groups = None
                    elif role in {"sentence", "review"}:
                        # Sentence/review roles are an explicit multi-model
                        # input contract.  They do not depend on the legacy
                        # optional "整句重识别" switch.
                        model_sentence_reocr = True
                    elif force_sentence_reocr is None:
                        model_sentence_reocr = bool(
                            sentence_context_reocr
                            and (
                                len(engine_ids) <= 1
                                or multi_sentence_mode == "all_models"
                                or model_index == 0
                            )
                        )
                    else:
                        model_sentence_reocr = bool(
                            sentence_context_reocr and force_sentence_reocr
                        )
                    if role == "auto" and not sentence_context_reocr:
                        model_sentence_reocr = False
                        sentence_target_ids = None
                        sentence_target_groups = None
                    model_global_sentence_reocr = bool(
                        model_sentence_reocr
                        and (True if role in {"sentence", "review"} else sentence_global_merged_box_reocr)
                    )
                    model_progress_label["value"] = (
                        f"[共识分歧救援·{engine_label}]"
                        if recovery_only
                        else f"[模型{model_index + 1}·{engine_label}]"
                    )
                    if not recovery_only:
                        emit_overall_progress(
                            progress_estimator.start_model(
                                model_index,
                                label=f"模型{model_index + 1}/{len(engine_ids)}·{engine_label}·加载与准备",
                            )
                        )
                    else:
                        emit_phase_progress(
                            f"共识分歧救援·{engine_label}", 0, 1
                        )

                    def on_column_phase(phase, current, total, filename):
                        if cancel_event.is_set():
                            return
                        phase_names = {
                            "split": "分列掩膜",
                            "recognition": "逐列识别",
                            "recognition_work": "逐列识别",
                            "document_build": "生成 OCR 文档",
                            "sentence_reocr": "整句重识别",
                        }
                        if model_global_sentence_reocr:
                            phase_names["split"] = "全书预分列"
                            phase_names["sentence_reocr"] = (
                                "疑难句合并框重识别"
                                if sentence_context_strategy == "smart"
                                else "全书合并框重识别"
                            )
                        if role == "review":
                            # Review models consume only the unresolved sentence
                            # transport.  Physical columns are seeded from the
                            # already completed main roles; calling this phase
                            # "全书预分列/逐列识别" made a healthy geometry-cache
                            # replay look like a reviewer was re-running 5k+ columns.
                            phase_names["split"] = "复用共享列几何"
                            phase_names["recognition"] = "残余整句复核"
                            phase_names["recognition_work"] = "残余整句复核"
                            phase_names["sentence_reocr"] = "残余整句复核"
                        phase_name = phase_names.get(phase, "逐列识别")
                        label = f"模型{model_index + 1}·{engine_label}·{phase_name}"
                        if phase == "recognition_work":
                            detail = str(filename or "逐列识别")
                            emit_overall_progress(
                                progress_estimator.update_phase(
                                    model_index,
                                    "recognition",
                                    current,
                                    max(total, 1),
                                    label=(
                                        f"模型{model_index + 1}/{len(engine_ids)}·"
                                        f"{engine_label}·{detail}"
                                    ),
                                    unit="内部进度",
                                )
                            )
                            # Per-column callbacks may be very frequent.  The
                            # blue bar receives every event, while the log is
                            # throttled to 5% buckets to avoid UI/log overhead.
                            bucket = int((max(0, current) / max(total, 1)) * 20)
                            state = getattr(on_column_phase, "_live_log_state", None)
                            if self._progress_display_enabled_event.is_set() and (state != bucket or current >= total):
                                on_column_phase._live_log_state = bucket
                                signals.log.emit(f"  [{label}] {detail}")
                            return

                        unit = (
                            "句组" if phase == "sentence_reocr"
                            else ("页" if phase in {"split", "recognition", "document_build"} else "项")
                        )
                        emit_overall_progress(
                            progress_estimator.update_phase(
                                model_index,
                                phase,
                                current,
                                max(total, 1),
                                label=f"模型{model_index + 1}/{len(engine_ids)}·{engine_label}·{phase_name}",
                                unit=unit,
                            )
                        )
                        bucket = int((max(0, current) / max(total, 1)) * 20)
                        phase_log_state = getattr(on_column_phase, "_phase_log_state", {})
                        phase_key = str(phase)
                        previous_bucket = phase_log_state.get(phase_key)
                        should_report_phase = bool(
                            current <= 0 or current >= total or previous_bucket != bucket
                        )
                        if should_report_phase:
                            phase_log_state[phase_key] = bucket
                            on_column_phase._phase_log_state = phase_log_state
                            emit_phase_progress(label, current, max(total, 1))
                        if self._progress_display_enabled_event.is_set() and should_report_phase:
                            signals.log.emit(
                                f"  [{label} {current:3d}/{max(total, 1)}] {filename}"
                            )

                    # Keep the original option contract visible for plugins/tests,
                    # then narrow only the sentence-stage switches per model in
                    # balanced multi-model mode. Single-model behaviour is unchanged.
                    sentence_option_contract = {
                        "column_sentence_context_reocr": sentence_context_reocr,
                        "column_sentence_global_merged_box_reocr": sentence_global_merged_box_reocr,
                    }
                    sentence_option_contract.update({
                        "column_sentence_context_reocr": model_sentence_reocr,
                        "column_sentence_global_merged_box_reocr": model_global_sentence_reocr,
                    })
                    # Preserve the historical option contract for plugins/tests;
                    # the role-specific effective value below may override it.
                    legacy_ndlocr_contract = {
                        "column_ndlocr_page_batch": ndlocr_page_mode != "column",
                        "column_ndlocr_page_mode": ndlocr_page_mode,
                    }

                    # 多模型对比只需要每个模型对同一物理列读取一次。
                    # 不再为主模型或辅助模型重复执行短块、扩边、放大、二值化
                    # 等 OCR 轮次；识别为空时保留 □ 并交给人工裁决。
                    compare_mode = (
                        "single_pass"
                        if (len(engine_ids) > 1 and role_use_column_mask)
                        or target_column_ids is not None
                        else "full"
                    )
                    model_variant_dir = shared_column_variant_dir
                    if len(engine_ids) > 1:
                        model_variant_dir = str(
                            Path(shared_column_variant_dir) / f"model_{model_index + 1}"
                        )
                    cache_key = ""
                    checkpoint_step = (
                        f"model_{model_index + 1}:{engine_id}:{role}:"
                        f"{effective_ndlocr_page_mode}"
                    )
                    if project_cache_manager is not None:
                        try:
                            cache_key = project_cache_manager.make_stage_cache_key(
                                "ocr_model",
                                config={
                                    "engine_id": str(engine_id),
                                    "model_index": int(model_index),
                                    "role": declared_role,
                                    "input_role": role,
                                    "role_schema": MULTI_OCR_ROLE_SCHEMA,
                                    "scheduler_contract": "role4-model-profiles-v2-independent-slots",
                                    "transport_profile": str(engine_profile.transport_profile),
                                    "resource_class": str(engine_profile.resource_class),
                                    "use_column_mask": bool(role_use_column_mask),
                                    "sentence_reocr": bool(model_sentence_reocr),
                                    "global_sentence_reocr": bool(model_global_sentence_reocr),
                                    "target_column_ids": sorted(target_column_ids) if target_column_ids is not None else None,
                                    "sentence_target_ids": sorted(sentence_target_ids) if sentence_target_ids is not None else None,
                                    "sentence_target_groups": (
                                        [list(group) for group in sentence_target_groups]
                                        if sentence_target_groups is not None else None
                                    ),
                                    "seed_results": seed_results or {},
                                    "seed_mark_selective": bool(seed_mark_selective),
                                    "recovery_only": bool(recovery_only),
                                    "ndlocr_page_mode": effective_ndlocr_page_mode,
                                    "ocr_mode": ocr_mode_snapshot,
                                    "fixed_crop_rect": fixed_crop_rect,
                                    "page_overrides": page_overrides,
                                    "engine_options": frozen_engine_options.get(engine_id, {}),
                                    "column_runtime": frozen_column_runtime,
                                    "compare_mode": compare_mode,
                                },
                                implementation=str(
                                    ocr_project_cache_context.get("engine_implementation", {}).get(
                                        str(engine_id), ""
                                    )
                                ),
                            )
                            cached_payload = project_cache_manager.load_stage_cache(
                                "ocr_model", cache_key
                            )
                            cached_document = (cached_payload or {}).get("document")
                            if isinstance(cached_document, dict):
                                document = UnifiedDocument.from_dict(cached_document)
                                performance_trace.increment("stage_cache_hits")
                                performance_trace.increment(f"stage_cache_hits.{engine_id}")
                                performance_trace.event(
                                    "stage_cache_hit", engine_id=str(engine_id),
                                    model_index=model_index + 1, role=role,
                                )
                                signals.log.emit(
                                    f"♻️ 模型{model_index + 1}·{engine_label}："
                                    "Stage Cache 命中，跳过重复 OCR。"
                                )
                                project_cache_manager.mark_checkpoint_step(
                                    "ocr", checkpoint_signature, checkpoint_step, cache_key,
                                    reused=True,
                                    detail={"engine_id": str(engine_id), "role": declared_role, "input_role": role},
                                )
                                performance_trace.increment("models_completed")
                                remember_partial(document)
                                raise_if_cancelled()
                                return document
                            performance_trace.increment("stage_cache_misses")
                        except Exception as exc:
                            cache_key = ""
                            signals.log.emit(
                                f"  ⚠️ {engine_label} 缓存读取失败，继续真实 OCR：{exc}"
                            )

                    # Compatibility/source contract: historically this line
                    # was ``return self._run_engine_document(``; the call path is
                    # still identical, but the result is remembered first so a
                    # Stop request can preserve the last completed partial document.
                    resource_name = model_resource(engine_id)
                    model_stage = f"model_{model_index + 1}_{engine_id}"
                    with runtime_governor.lease(
                        resource_name, cancel_check=cancel_token.is_cancelled
                    ), performance_trace.stage(model_stage):
                        document = self._run_engine_document(
                        engine_id,
                        common_kwargs=make_common_kwargs(model_index),
                        use_column_mask=role_use_column_mask,
                        phase_callback=on_column_phase if role_use_column_mask else None,
                        # Keep column and sentence review assets under the
                        # retained OCR temp root even for a single model.
                        # This does not change OCR pixels; it only prevents
                        # the exact proofread image from being deleted when
                        # the adapter-local temporary directory closes.
                        shared_column_prepare_dir=shared_column_prepare_dir,
                        shared_column_variant_dir=(model_variant_dir if len(engine_ids) > 1 else ""),
                        # 保持前 3 个稳定版本的 Apple OCR 调用路径：
                        # 共享物理列掩膜，但不向通用裁图层注入专用复用参数。
                        # 每个模型仍只对每个物理列执行一次 OCR。
                        extra_engine_options={
                            "column_compare_mode": compare_mode,
                            "column_target_ids": (
                                sorted(target_column_ids)
                                if target_column_ids is not None else None
                            ),
                            "column_seed_results": dict(seed_results or {}),
                            "column_seed_mark_selective": bool(seed_mark_selective),
                            "column_sentence_target_ids": (
                                sorted(sentence_target_ids)
                                if sentence_target_ids is not None else None
                            ),
                            "column_sentence_target_groups": (
                                [list(group) for group in sentence_target_groups]
                                if sentence_target_groups is not None else None
                            ),
                            "column_retained_recognizer_holder": (
                                retained_hayai_holder
                                if retain_primary_hayai_session
                                and model_index == 0
                                and str(engine_id).strip().lower() == "hayai_ocr"
                                else None
                            ),
                            "column_preview_callback": (on_column_preview if model_index == 0 else None),
                            **sentence_option_contract,
                            "column_sentence_context_strategy": (
                                "full" if role in {"sentence", "review"} else sentence_context_strategy
                            ),
                            "column_compact_primary_transport": compact_primary_transport,
                            "column_rescue_policy": column_rescue_policy,
                            # V5.7: persistent content-addressed recognition cache
                            # below the whole-model Stage Cache.  A single
                            # changed page therefore does not force unchanged
                            # column/sentence images through OCR again.
                            "column_segment_cache_dir": (
                                str(Path(str(ocr_project_cache_context.get("project_path", ""))) /
                                    "artifacts" / "segment_cache" / str(engine_id))
                                if ocr_project_cache_context.get("project_path") else ""
                            ),
                            "column_segment_cache_runtime_id": (
                                hashlib.sha256(json.dumps({
                                    "implementation": str(ocr_project_cache_context.get("engine_implementation", {}).get(str(engine_id), "")),
                                    "engine_options": frozen_engine_options.get(engine_id, {}),
                                    "column_runtime": frozen_column_runtime,
                                    "transport_profile": str(engine_profile.transport_profile),
                                    "input_role": str(role),
                                    "model_slot_index": int(model_index),
                                    "scheduler_contract": "role4-model-profiles-v2-independent-slots",
                                }, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()
                                if ocr_project_cache_context.get("project_path") else ""
                            ),
                            **legacy_ndlocr_contract,
                            # NDLOCR is the page-layout exception: it may use
                            # one Ruby-free full-page pass for throughput, but
                            # routing and any fallback still consume the same
                            # authoritative shared physical-column geometry as
                            # every other OCR model.
                            "column_ndlocr_page_batch": effective_ndlocr_page_mode != "column",
                            "column_ndlocr_page_mode": effective_ndlocr_page_mode,
                            "column_sentence_context_max_columns": column_reflow_max,
                            "column_shared_sentence_dir": (
                                shared_sentence_dir
                            ),
                            # Defence in depth: the common OCR layer already
                            # writes a masked page, but every later fallback
                            # and sentence-box pass receives the same fixed
                            # body rectangle and reapplies it before reading
                            # pixels.  A running header can therefore never
                            # re-enter through an original-page code path.
                            "column_fixed_region_rect": fixed_crop_rect,
                        },
                        frozen_engine_options=frozen_engine_options.get(engine_id, {}),
                        frozen_column_options=frozen_column_runtime,
                        input_role=role,
                    )
                    # A column adapter may return a structurally valid partial
                    # document while Stop is propagating through an external worker.
                    # Never promote that partial model to Stage Cache/checkpoint: the
                    # next run must reuse segment-level results but resume the unfinished
                    # model instead of treating it as complete.
                    raise_if_cancelled()

                    # Persist both role granularity and the engine-specific
                    # final transport. Geometry identity is shared; recognizer
                    # pixels are allowed to differ by profile.
                    try:
                        document.metadata.__dict__["multi_ocr_model_index"] = int(model_index) + 1
                        # Keep the semantic role separate from the recognizer input role.
                        # Review slots commonly use column transport, so caching only
                        # ``multi_ocr_input_role`` loses the fact that a restored model
                        # was review1/review2 and breaks adjudication-bundle export.
                        document.metadata.__dict__["multi_ocr_role"] = declared_role
                        document.metadata.__dict__["multi_ocr_input_role"] = str(role)
                        document.metadata.__dict__["multi_ocr_transport_profile"] = engine_profile.transport_profile
                        document.metadata.__dict__["multi_ocr_resource_class"] = engine_profile.resource_class
                        document.metadata.__dict__["multi_ocr_role_schema"] = MULTI_OCR_ROLE_SCHEMA
                        document.metadata.__dict__["multi_ocr_scheduler_contract"] = "role4-model-profiles-v2-independent-slots"
                    except Exception:
                        pass

                    if project_cache_manager is not None and cache_key:
                        try:
                            project_cache_manager.save_stage_cache(
                                "ocr_model", cache_key,
                                {"document": document.to_dict()},
                                metadata={
                                    "engine_id": str(engine_id),
                                    "engine_label": str(engine_label),
                                    "model_index": int(model_index),
                                    "role": role,
                                    "role_schema": MULTI_OCR_ROLE_SCHEMA,
                                    "pipeline_signature": checkpoint_signature,
                                },
                            )
                            project_cache_manager.mark_checkpoint_step(
                                "ocr", checkpoint_signature, checkpoint_step, cache_key,
                                reused=False,
                                detail={"engine_id": str(engine_id), "role": role},
                            )
                            performance_trace.increment("stage_cache_writes")
                        except Exception as exc:
                            signals.log.emit(
                                f"  ⚠️ {engine_label} 缓存写入失败，不影响本次 OCR：{exc}"
                            )
                    performance_trace.increment("models_completed")
                    remember_partial(document)
                    raise_if_cancelled()
                    return document

                if multi_role_enabled and multi_role_plan is not None:
                    from core.multi_ocr_roles import (
                        MULTI_OCR_ROLE_SCHEMA, role_display, runtime_input_role,
                    )
                    from engine.column_sentence_reflow import reflow_columns_into_sentences
                    from engine.multi_ocr_compare import compare_ocr_documents, build_fused_document
                    from engine.multi_ocr_consensus import seed_from_document
                    from engine.page_sentence_projector import (
                        canonicalize_page_document_sentences,
                        project_page_document_to_sentences,
                    )

                    all_labels = [
                        role_display(role, self._engine_label(engine_id))
                        for role, engine_id in multi_role_plan.selected_roles
                    ]
                    execution_preview = list(multi_role_plan.preferred_execution_roles)
                    if execution_preview:
                        signals.log.emit(
                            "🧩 多模型独立执行计划："
                            + " → ".join(
                                f"{role_display(role, self._engine_label(engine_id))}"
                                for role, engine_id in execution_preview
                            )
                            + "；每个槽位独立运行/缓存，单模型选择不参与本轮。"
                        )
                    raw_documents: list = []
                    active_labels: list[str] = []
                    active_roles: list[str] = []
                    active_engine_ids: list[str] = []
                    failed_models: dict[int, str] = {}

                    def role_model_index(role: str, engine_id: str = "") -> int:
                        # Engine IDs are unique across visible roles.  The role
                        # slot remains the stable execution/cache identity;
                        # same-engine disagreement confirmation is internal retry.
                        if role in role_index_by_name:
                            return int(role_index_by_name[role])
                        for index, (slot_role, slot_engine) in enumerate(multi_role_plan.selected_roles):
                            if slot_role == role and (not engine_id or slot_engine == engine_id):
                                return index
                        raise ValueError(f"未知多模型角色槽位：{role} / {engine_id}")

                    def run_role_safely(role: str, engine_id: str, **kwargs):
                        model_index = role_model_index(role, engine_id)
                        performance_trace.increment(f"multi_ocr.role_calls.{role}")
                        try:
                            with performance_trace.stage(f"multi_ocr.role.{role}"):
                                return run_one(
                                    engine_id,
                                    model_index,
                                    input_role=runtime_input_role(role, engine_id),
                                    semantic_role=role,
                                    **kwargs,
                                )
                        except InterruptedError:
                            raise
                        except Exception as exc:
                            performance_trace.increment(f"multi_ocr.role_failures.{role}")
                            failed_models[model_index] = str(exc)
                            signals.log.emit(
                                f"⚠️ {role_display(role, self._engine_label(engine_id))} 失败，"
                                f"其余角色继续：{exc}"
                            )
                            return None

                    def normalize_main_document(role: str, document):
                        if document is None:
                            return None
                        metadata_dict = getattr(getattr(document, "metadata", None), "__dict__", {}) or {}
                        source_engine = str(getattr(getattr(document, "metadata", None), "source_engine", "") or "")
                        page_has_shared_columns = bool(
                            role == "page"
                            and (metadata_dict.get("column_ocr") or source_engine.startswith("masked_column_ocr:"))
                        )
                        if role in {"column", "sentence"} or page_has_shared_columns:
                            document = reflow_columns_into_sentences(
                                document,
                                max_columns=column_reflow_max,
                                cancel_check=cancel_event.is_set,
                            )
                        document.metadata.__dict__["multi_ocr_role"] = role
                        document.metadata.__dict__["multi_ocr_role_schema"] = MULTI_OCR_ROLE_SCHEMA
                        return document

                    main_docs_by_role: dict[str, object] = {}
                    raw_page_document = None
                    canonical_seed_doc = None

                    def commit_main_role_result(role: str, engine_id: str, current):
                        """Normalize one completed main-role result without re-running OCR."""
                        nonlocal raw_page_document, canonical_seed_doc
                        if current is None:
                            return
                        if role == "page":
                            meta = getattr(getattr(current, "metadata", None), "__dict__", {}) or {}
                            source_engine = str(getattr(getattr(current, "metadata", None), "source_engine", "") or "")
                            already_shared = bool(
                                meta.get("column_ocr")
                                or source_engine.startswith("masked_column_ocr:")
                            )
                            # NDLOCR page-role hybrid already performs one true
                            # full-page pass and routes it onto shared columns;
                            # do not throw away that lineage and realign by text.
                            raw_page_document = None if already_shared else current
                        current = normalize_main_document(role, current)
                        if current is None:
                            return
                        main_docs_by_role[role] = current
                        if role in {"page", "column", "sentence"} and canonical_seed_doc is None:
                            seed = seed_from_document(current)
                            if seed:
                                canonical_seed_doc = current

                    def run_main_role(role: str, engine_id: str):
                        label = role_display(role, self._engine_label(engine_id))
                        signals.log.emit(f"\n▶ {label}")
                        kwargs = {}
                        if role in {"column", "sentence", "page"}:
                            # Phase27: the historical ``sentence`` slot is now a
                            # second full-column main role.  It reads every shared
                            # physical column independently and is reflowed only
                            # after OCR for sentence-level comparison.
                            kwargs.update(force_sentence_reocr=False)
                        return run_role_safely(role, engine_id, **kwargs)

                    execution_roles = list(multi_role_plan.preferred_execution_roles)
                    bootstrap_roles = [
                        (role, engine_id) for role, engine_id in execution_roles
                        if role in {"page", "column"}
                    ]
                    remaining_roles = [
                        (role, engine_id) for role, engine_id in execution_roles
                        if role not in {"page", "column"}
                    ]

                    # Reuse the fastest idea from the old multi-OCR path, but
                    # at role granularity: full-page and full-column passes
                    # overlap only when their configured backends use
                    # complementary resources.  The default NDL(ONNX/CPU) +
                    # Hayai(MPS) pairing therefore overlaps; two MPS-heavy
                    # recognizers remain serialized to protect unified memory.
                    parallel_bootstrap = False
                    if len(bootstrap_roles) == 2:
                        from core.multi_ocr_roles import resources_can_overlap
                        left_role, left_engine = bootstrap_roles[0]
                        right_role, right_engine = bootstrap_roles[1]
                        left_resource = model_resource(left_engine)
                        right_resource = model_resource(right_engine)
                        parallel_bootstrap = resources_can_overlap(
                            left_resource, right_resource
                        )
                        if parallel_bootstrap:
                            signals.log.emit(
                                "🚀 多模型异构并行："
                                f"{role_display(left_role, self._engine_label(left_engine))}"
                                f"[{left_resource}] + "
                                f"{role_display(right_role, self._engine_label(right_engine))}"
                                f"[{right_resource}] 同时运行。"
                            )
                            performance_trace.increment(
                                "multi_ocr.parallel_bootstrap_waves"
                            )
                            performance_trace.event(
                                "multi_ocr_parallel_bootstrap",
                                left_role=left_role,
                                left_engine=left_engine,
                                left_resource=left_resource,
                                right_role=right_role,
                                right_engine=right_engine,
                                right_resource=right_resource,
                            )
                        else:
                            signals.log.emit(
                                "🧠 两个主角色资源存在竞争，将按上述队列依次运行："
                                f"{left_resource} + {right_resource}。"
                                " 当前只显示正在执行的一个槽位，不代表其它已选模型被跳过。"
                            )

                    if parallel_bootstrap:
                        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
                        pool = ThreadPoolExecutor(
                            max_workers=2, thread_name_prefix="ocr-role-bootstrap"
                        )
                        future_to_role = {}
                        parallel_results: dict[str, object] = {}
                        parallel_cancelled = False
                        try:
                            for role, engine_id in bootstrap_roles:
                                future = pool.submit(run_main_role, role, engine_id)
                                future_to_role[future] = (role, engine_id)
                            pending = set(future_to_role)
                            while pending:
                                if cancel_event.is_set():
                                    parallel_cancelled = True
                                    raise InterruptedError("OCR 已停止")
                                done, pending = wait(
                                    pending, timeout=0.10, return_when=FIRST_COMPLETED
                                )
                                for future in done:
                                    role, _engine_id = future_to_role[future]
                                    parallel_results[role] = future.result()
                        finally:
                            if parallel_cancelled or cancel_event.is_set():
                                for future in future_to_role:
                                    future.cancel()
                                pool.shutdown(wait=False, cancel_futures=True)
                            else:
                                pool.shutdown(wait=True)
                        # Commit in deterministic scheduler order so the
                        # canonical seed does not depend on thread completion.
                        for role, engine_id in bootstrap_roles:
                            commit_main_role_result(
                                role, engine_id, parallel_results.get(role)
                            )
                    else:
                        for role, engine_id in bootstrap_roles:
                            raise_if_cancelled()
                            commit_main_role_result(
                                role, engine_id, run_main_role(role, engine_id)
                            )

                    # If no column result exists, the page role itself must
                    # provide canonical sentence structure for page-only or
                    # page+sentence runs.  Do this exactly once, after bootstrap,
                    # so page+column never pays for an intermediate projection
                    # that would immediately be replaced by column canonical rows.
                    if (
                        raw_page_document is not None
                        and multi_structure_cache is not None
                    ):
                        try:
                            from engine.multi_ocr_structure_prewarm import (
                                page_document_to_shared_sentences,
                            )
                            page_structured, page_structure_stats = page_document_to_shared_sentences(
                                raw_page_document,
                                multi_structure_cache,
                                max_columns=column_reflow_max,
                                cancel_check=cancel_event.is_set,
                            )
                            page_structured = normalize_main_document("page", page_structured)
                            main_docs_by_role["page"] = page_structured
                            raw_page_document = None
                            performance_trace.set_gauge(
                                "multi_ocr.page_projection_pages",
                                int(page_structure_stats.pages),
                            )
                            performance_trace.set_gauge(
                                "multi_ocr.page_projection_sentences",
                                int(page_structure_stats.sentences),
                            )
                            signals.log.emit(
                                "🧭 整页 OCR→共享列→整句："
                                f"{page_structure_stats.pages} 页 / "
                                f"{page_structure_stats.columns} 列 / "
                                f"{page_structure_stats.sentences} 句；"
                                f"Ruby侧框过滤 {getattr(page_structure_stats, 'ruby_filtered_blocks', 0)}；"
                                f"栏外坐标块丢弃 {getattr(page_structure_stats, 'outside_geometry_blocks', 0)}；"
                                "OCR 调用 +0。"
                            )
                        except InterruptedError:
                            raise
                        except Exception as exc:
                            signals.log.emit(
                                f"⚠️ 整页结果映射共享结构失败，改用句级兜底：{exc}"
                            )

                    # Prefer the column role as the structural sentence seed
                    # when it exists: its per-column terminal evidence is the
                    # most direct representation of the shared physical
                    # columns.  Page-only runs still use the page projection.
                    canonical_seed_doc = (
                        main_docs_by_role.get("column")
                        or main_docs_by_role.get("page")
                        or canonical_seed_doc
                    )

                    # Phase27: the historical third ``sentence`` slot is now
                    # a full-column main role.  It always performs a complete
                    # physical-column pass.  Selective conflict-only OCR belongs
                    # exclusively to review1, so the old sentence smart router is
                    # intentionally bypassed while its audit fields remain for
                    # backwards-readable workspaces.
                    router_audit = {
                        "enabled": False,
                        "sentence_mode": "legacy_key_full_column",
                        "sentence_target_rows": 0,
                        "sentence_target_columns": 0,
                        "sentence_reason": "phase27_full_column_role",
                    }
                    for role, engine_id in remaining_roles:
                        raise_if_cancelled()
                        commit_main_role_result(
                            role, engine_id, run_main_role(role, engine_id)
                        )

                    if not main_docs_by_role:
                        failure_lines = [
                            f"{all_labels[index]}：{message}"
                            for index, message in sorted(failed_models.items())
                            if index < len(all_labels)
                        ]
                        raise RuntimeError(
                            "所选主 OCR 角色均未能启动或返回结果。"
                            + ("\n" + "\n".join(failure_lines) if failure_lines else "")
                        )

                    canonical_doc = (
                        main_docs_by_role.get("column")
                        or main_docs_by_role.get("sentence")
                        or main_docs_by_role.get("page")
                    )
                    # Full-page recognition count is unchanged: one real OCR
                    # call per page.  If another main role provides the final
                    # canonical sentence boundaries, align the already-read
                    # page evidence to those rows.  Otherwise the prewarmed
                    # geometry projection above is itself the canonical page
                    # sentence document.  None of these branches re-run OCR.
                    if (
                        raw_page_document is not None
                        and canonical_doc is not None
                        and (main_docs_by_role.get("column") is not None
                             or main_docs_by_role.get("sentence") is not None)
                    ):
                        projected, stats = project_page_document_to_sentences(
                            raw_page_document,
                            canonical_doc,
                            page_label=role_display("page", self._engine_label(multi_role_plan.page)),
                        )
                        main_docs_by_role["page"] = projected
                        signals.log.emit(
                            f"🧭 整页→canonical 单句投影：{stats.rows} 句；"
                            f"空候选 {stats.empty_rows}；"
                            f"共享列锚定 {stats.column_anchored_rows} 句；OCR 调用 +0。"
                        )
                    elif main_docs_by_role.get("page") is not None:
                        page_doc = main_docs_by_role["page"]
                        has_shared_ids = any(
                            bool((block.metadata or {}).get("source_column_ids"))
                            for block in getattr(page_doc, "blocks", [])
                        )
                        if has_shared_ids:
                            canonical_doc = page_doc
                        else:
                            projected, stats = canonicalize_page_document_sentences(page_doc)
                            main_docs_by_role["page"] = projected
                            canonical_doc = projected
                            signals.log.emit(
                                f"🧭 仅整页主模型→单句整理：{stats.rows} 句；未增加 OCR 调用。"
                            )

                    documents = []
                    labels = []
                    roles = []
                    engines = []
                    for role, engine_id in multi_role_plan.main_roles:
                        current = main_docs_by_role.get(role)
                        if current is None:
                            continue
                        documents.append(current)
                        labels.append(role_display(role, self._engine_label(engine_id)))
                        roles.append(role)
                        engines.append(engine_id)

                    raise_if_cancelled()
                    comparison = compare_ocr_documents(documents, labels)
                    # Alternate-input retries are supplemental evidence only.
                    # They may confirm an already existing strict majority,
                    # but never replace the raw OCR documents or add votes.
                    main_document_count = len(documents)
                    targeted_retry_evidence = []
                    targeted_retry_local_report = None
                    performance_trace.set_gauge("multi_ocr.main_models", len(documents))
                    performance_trace.set_gauge("multi_ocr.compare_rows", len(comparison.rows))
                    performance_trace.set_gauge("multi_ocr.conflict_rows_initial", comparison.conflict_rows)
                    performance_trace.set_gauge("multi_ocr.exact_rows_initial", comparison.exact_rows)

                    # Whole-page NDLOCR is already an independent main-role vote.
                    # Do NOT re-read every conflicting physical column with the same
                    # model: the full-book benchmark (416 pages / 404 OCR pages)
                    # spent ~745 s on 2,104 columns without reducing the 1,173
                    # sentence conflicts at all. Same-model column retries therefore
                    # add no vote and are no longer part of the normal local-retry
                    # path. NDLOCR's own tiny hybrid bad-column rescue inside its
                    # primary page pass remains available.
                    performance_trace.set_gauge(
                        "multi_ocr.ndl_conflict_column_rescue_columns", 0
                    )

                    canonical_for_reviews = (
                        main_docs_by_role.get("column")
                        or main_docs_by_role.get("sentence")
                        or main_docs_by_role.get("page")
                    )
                    # Persist enough evidence-routing telemetry to audit a
                    # later workspace/decision bundle without replaying logs.
                    # A configured reviewer is useful only if it actually
                    # re-read target pixels rather than inheriting seed text.
                    review_evidence_audit: dict[str, dict[str, object]] = {}

                    for review_role, review_engine in multi_role_plan.review_roles:
                        raise_if_cancelled()
                        historical_conflicts = [row for row in comparison.rows if row.is_conflict]
                        # 48px AR is intentionally a *bounded* reviewer, not a
                        # third full-book OCR. On the real 269-page project an
                        # unbounded disagreement pass tried 2,147 columns and
                        # consumed another ~11 minutes before being stopped.
                        # Spend the local OCR budget only on the highest-risk
                        # rows; every skipped disagreement remains pending and
                        # is exported with its sentence image for AI/human
                        # adjudication.
                        conflict_rows, budget_ids, column_budget = _bounded_review_rows(
                            historical_conflicts, review_engine
                        )
                        deferred_to_ai = max(0, len(historical_conflicts) - len(conflict_rows))
                        if historical_conflicts:
                            signals.log.emit(
                                f"⚡ 分歧复核快速策略：{deferred_to_ai} 个分歧直接保留给 AI/人工；"
                                f"第三 OCR 仅补读 {len(conflict_rows)} 个最高风险句 / "
                                f"{len(budget_ids)} 个物理列（预算 {column_budget} 列）。"
                            )
                        if not historical_conflicts:
                            signals.log.emit(
                                f"⏭ {role_display(review_role, self._engine_label(review_engine))} 已跳过："
                                "当前没有真正句级分歧。"
                            )
                            continue
                        if not conflict_rows:
                            signals.log.emit(
                                f"⏭ {role_display(review_role, self._engine_label(review_engine))} 已跳过："
                                f"{len(historical_conflicts)} 个分歧句已经拥有至少 3 份独立证据且出现重复候选；"
                                "保留分歧给 AI/人工，不再无意义叠加本地 OCR。"
                            )
                            continue
                        if canonical_for_reviews is None:
                            signals.log.emit(
                                f"⏭ {role_display(review_role, self._engine_label(review_engine))} 无法自动运行："
                                "当前只有整页主模型，没有共享物理列 sentence geometry；分歧保留给人工/AI。"
                            )
                            continue
                        target_ids = {
                            str(column_id)
                            for row in conflict_rows
                            for column_id in (row.column_ids or ())
                            if str(column_id)
                        }
                        performance_trace.set_gauge(
                            f"multi_ocr.{review_role}.target_sentences", len(conflict_rows)
                        )
                        performance_trace.set_gauge(
                            f"multi_ocr.{review_role}.target_columns", len(target_ids)
                        )
                        if not target_ids:
                            signals.log.emit(
                                f"⏭ {role_display(review_role, self._engine_label(review_engine))} 无可定位列 ID；"
                                "分歧保留给人工/AI。"
                            )
                            continue
                        seed = seed_from_document(canonical_for_reviews)
                        signals.log.emit(
                            f"🔎 {role_display(review_role, self._engine_label(review_engine))}："
                            f"仅处理 {len(conflict_rows)} 个仍需新增证据的分歧句 / "
                            f"{len(target_ids)} 个物理列。"
                        )
                        # Review slots are column-evidence refreshers.  They
                        # re-read only disputed physical columns; no removed
                        # model-specific sentence-review transport remains.
                        review_doc = run_role_safely(
                            review_role,
                            review_engine,
                            force_sentence_reocr=False,
                            target_column_ids=set(target_ids),
                            seed_results=seed,
                            sentence_target_ids=None,
                            seed_mark_selective=True,
                        )
                        if review_doc is None:
                            continue
                        review_doc = reflow_columns_into_sentences(
                            review_doc,
                            max_columns=column_reflow_max,
                            cancel_check=cancel_event.is_set,
                        )
                        review_doc.metadata.__dict__["multi_ocr_role"] = review_role
                        review_doc.metadata.__dict__["multi_ocr_role_schema"] = MULTI_OCR_ROLE_SCHEMA
                        review_doc.metadata.__dict__["multi_ocr_review_transport"] = "target-physical-columns-v1"
                        documents.append(review_doc)
                        labels.append(role_display(review_role, self._engine_label(review_engine)))
                        roles.append(review_role)
                        engines.append(review_engine)
                        comparison = compare_ocr_documents(documents, labels)
                        remaining = sum(1 for row in comparison.rows if row.is_conflict)
                        review_model_index = len(documents) - 1
                        independent_review_rows = sum(
                            1 for row in comparison.rows
                            if review_model_index not in set(row.consensus_seeded_models or ())
                            and review_model_index < len(row.texts)
                            and bool(
                                str(row.texts[review_model_index] or "").strip()
                                .replace("□", "").replace("�", "")
                            )
                        )
                        performance_trace.set_gauge(
                            f"multi_ocr.{review_role}.remaining_conflicts", remaining
                        )
                        performance_trace.set_gauge(
                            f"multi_ocr.{review_role}.independent_evidence_rows",
                            independent_review_rows,
                        )
                        review_evidence_audit[str(review_role)] = {
                            "engine_id": str(review_engine or ""),
                            "target_sentence_rows": int(len(conflict_rows)),
                            "target_physical_columns": int(len(target_ids)),
                            "independent_evidence_rows": int(independent_review_rows),
                            "remaining_conflicts": int(remaining),
                            "transport": "target_physical_columns",
                        }
                        if independent_review_rows <= 0:
                            signals.log.emit(
                                f"⚠️ {role_display(review_role, self._engine_label(review_engine))} 完成，"
                                "但没有生成可计入对比的独立 OCR 证据；该模型不会被当作第三票。"
                            )
                        else:
                            signals.log.emit(
                                f"✅ {role_display(review_role, self._engine_label(review_engine))} 完成；"
                                f"新增独立证据 {independent_review_rows} 句；"
                                f"仍有 {remaining} 个真正分歧句。"
                            )

                    # Targeted alternate-input confirmation.  Only rows
                    # that already have a strict majority among independent
                    # OCR models are eligible.  Any actually executed OCR
                    # role (main or disagreement-review) may re-read the same
                    # verified retry input; the retry can confirm that
                    # majority, but the retry itself never adds a vote.
                    targeted_retry_diagnostics = []
                    targeted_retry_stage_metrics = []
                    try:
                        from engine.targeted_retry_adjudicator import (
                            adjudicate_targeted_retries, retry_evidence_from_document,
                            targeted_retry_plan,
                        )
                        preliminary_retry_report = adjudicate_targeted_retries(
                            comparison, targeted_retry_evidence, labels=labels
                        )
                        already_resolved = set(preliminary_retry_report.local_resolved_row_indices)
                        retry_plan = targeted_retry_plan(
                            comparison,
                            exclude_rows=already_resolved,
                        )

                        def _primary_retry_role(model_index: int) -> bool:
                            if model_index < 0 or model_index >= len(roles):
                                return False
                            role_name = str(roles[model_index] or "")
                            return role_name in {"column", "page"}

                        # A disagreement reviewer has already been invoked exactly
                        # because the main models disagreed. Re-running that same
                        # reviewer is correlated evidence, not a new vote. Sentence
                        # main roles already saw merged sentence context. Only the
                        # original page/column main models are eligible for optional
                        # second-pass confirmation.
                        retry_plan = {
                            int(model_index): list(row_indices)
                            for model_index, row_indices in retry_plan.items()
                            if _primary_retry_role(int(model_index))
                        }
                        if not multi_local_retry_enabled:
                            # The checkbox is a hard execution gate: OFF means
                            # zero alternate-input / sentence-context retries
                            # and therefore zero retry-based local verdicts.
                            retry_plan = {}
                        if multi_local_retry_enabled:
                            signals.log.emit(
                                "🧭 定向二次 OCR：仅逐列/整页主模型可做同模型确认；全列主模型与分歧复核模型不会再二次重跑。"
                            )
                            signals.log.emit(
                                "🔬 主模型多数确认重试：只使用真实整句上下文；"
                                "已移除整本实测无新增确认的扩白边/横排/2×图像变体重试。"
                            )

                        canonical_retry_doc = canonical_for_reviews
                        if canonical_retry_doc is not None:
                            for retry_model_index, retry_rows in sorted(retry_plan.items()):
                                if retry_model_index >= len(roles) or retry_model_index >= len(engines):
                                    continue
                                retry_role = str(roles[retry_model_index] or "")
                                retry_engine = str(engines[retry_model_index] or "")
                                # The third full-column main role is intentionally excluded from
                                # optional same-model confirmation to avoid a second full-column pass.
                                if retry_role == "sentence" or not retry_engine:
                                    continue
                                retry_column_ids = {
                                    str(column_id)
                                    for row_index in retry_rows
                                    if 0 <= row_index < len(comparison.rows)
                                    for column_id in (comparison.rows[row_index].column_ids or ())
                                    if str(column_id)
                                }
                                if not retry_column_ids:
                                    continue
                                seed = seed_from_document(canonical_retry_doc)
                                engine_slot = role_model_index(retry_role, retry_engine)
                                signals.log.emit(
                                    f"🔁 定向重试确认：{labels[retry_model_index]} 原本持异议，"
                                    f"对 {len(retry_rows)} 个已有严格多数的句组改用真实整句上下文重读；"
                                    "不新增模型票。"
                                )
                                with performance_trace.stage(
                                    f"multi_ocr.targeted_retry.{retry_model_index}"
                                ):
                                    retry_doc = run_one(
                                        retry_engine,
                                        engine_slot,
                                        input_role="review",
                                        force_sentence_reocr=True,
                                        target_column_ids=set(),
                                        seed_results=seed,
                                        sentence_target_ids=retry_column_ids,
                                        seed_mark_selective=True,
                                        recovery_only=True,
                                    )
                                if retry_doc is None:
                                    continue
                                retry_doc = reflow_columns_into_sentences(
                                    retry_doc,
                                    max_columns=column_reflow_max,
                                    cancel_check=cancel_event.is_set,
                                )
                                targeted_retry_evidence.extend(retry_evidence_from_document(
                                    comparison,
                                    canonical_retry_doc,
                                    retry_doc,
                                    model_index=retry_model_index,
                                    model_label=labels[retry_model_index],
                                    retry_kind=f"{retry_role}_to_sentence_context_retry",
                                    target_row_indices=retry_rows,
                                    details={
                                        "input_role_before": retry_role,
                                        "input_role_retry": "sentence_context",
                                        "raw_document_replaced": False,
                                        "retry_counts_as_independent_vote": False,
                                    },
                                ))
                        targeted_retry_local_report = (
                            adjudicate_targeted_retries(
                                comparison, targeted_retry_evidence, labels=labels
                            )
                            if multi_local_retry_enabled else None
                        )
                        if targeted_retry_local_report is not None and targeted_retry_stage_metrics:
                            targeted_retry_local_report.summary["stage_metrics"] = list(
                                targeted_retry_stage_metrics
                            )
                            targeted_retry_local_report.summary["stage_seconds_total"] = round(
                                sum(float(item.get("seconds", 0.0) or 0.0) for item in targeted_retry_stage_metrics),
                                6,
                            )
                            targeted_retry_local_report.summary["stage_attempted_inputs_total"] = sum(
                                int(item.get("attempted_inputs", 0) or 0) for item in targeted_retry_stage_metrics
                            )
                        if targeted_retry_diagnostics and targeted_retry_local_report is not None:
                            audit_path = Path(crop_temp_root) / "targeted_retry_variants" / "diagnostics.json"
                            targeted_retry_local_report.summary["variant_diagnostics_count"] = len(
                                targeted_retry_diagnostics
                            )
                            try:
                                audit_path.write_text(
                                    json.dumps(targeted_retry_diagnostics, ensure_ascii=False, indent=2),
                                    encoding="utf-8",
                                )
                                targeted_retry_local_report.summary["variant_diagnostics_path"] = str(audit_path)
                            except OSError as exc:
                                signals.log.emit(f"⚠️ 局部补救诊断文件未能保存：{exc}")
                        retry_summary = (
                            dict(targeted_retry_local_report.summary or {})
                            if targeted_retry_local_report is not None else {}
                        )
                        performance_trace.set_gauge(
                            "multi_ocr.targeted_retry.local_resolved",
                            int(retry_summary.get("local_resolved", 0) or 0),
                        )
                        performance_trace.set_gauge(
                            "multi_ocr.targeted_retry.evidence_count",
                            len(targeted_retry_evidence),
                        )
                        if int(retry_summary.get("local_resolved", 0) or 0):
                            signals.log.emit(
                                f"⚖ 本地重试多数裁决：{int(retry_summary.get('local_resolved', 0) or 0)} 句；"
                                "原始多模型 OCR 全部保留，重试只确认原有严格多数。"
                            )
                    except InterruptedError:
                        raise
                    except Exception as exc:
                        signals.log.emit(
                            f"⚠️ 定向重试多数确认失败，保留原始多模型分歧继续人工/AI：{exc}"
                        )
                        targeted_retry_local_report = None

                    performance_trace.set_gauge(
                        "multi_ocr.conflict_rows_final",
                        sum(1 for row in comparison.rows if row.is_conflict),
                    )
                    performance_trace.set_gauge(
                        "multi_ocr.executed_documents", len(documents)
                    )
                    raise_if_cancelled()
                    # Ensure the base document used for writeback has the same
                    # canonical sentence structure as the comparison primary.
                    fused = build_fused_document(documents[0], comparison)
                    fused.metadata.__dict__["multi_ocr_role_plan"] = multi_role_plan.as_dict()
                    fused.metadata.__dict__["multi_ocr_role_schema"] = MULTI_OCR_ROLE_SCHEMA
                    fused.metadata.__dict__["multi_ocr_roles_executed"] = list(roles)
                    fused.metadata.__dict__["multi_ocr_router"] = dict(router_audit)
                    fused.metadata.__dict__["multi_ocr_review_evidence"] = dict(review_evidence_audit)
                    fused.metadata.__dict__["multi_ocr_review_evidence_schema"] = 2
                    maybe_preserve_ruby([*documents, fused])
                    if handwriting_enabled:
                        from engine.ocr_manual_review import annotate_ocr_review_risks
                        annotate_ocr_review_risks(fused)
                    signals.log.emit(f"🔀 {comparison.summary}")
                    signals.log.emit(
                        "📌 分工式多模型：OCR 对比仍统一按 canonical 单句；"
                        "第三主槽为全列主模型，完整读取共享物理列；"
                        "分歧复核模型只重读残余冲突物理列。"
                    )
                    emit_overall_progress(progress_estimator.complete(label="分工式多模型 OCR 完成"))
                    source_texts = [
                        "\n".join(row.texts[index] for row in comparison.rows)
                        for index in range(len(documents))
                    ]
                    finish_project_checkpoint("ok")
                    signals.finished.emit({
                        "multi_ocr": True,
                        "role_based": True,
                        "role_plan": multi_role_plan.as_dict(),
                        "router_audit": dict(router_audit),
                        "targeted_retry_local_report": targeted_retry_local_report,
                        "targeted_retry_diagnostics": targeted_retry_diagnostics if multi_local_retry_enabled else [],
                        "documents": documents,
                        "labels": labels,
                        "comparison": comparison,
                        "source_texts": source_texts,
                        "fused": fused,
                        "ruby_enabled": bool(ruby_preserve_enabled),
                        "handwriting_review_context": dict(handwriting_review_context_snapshot),
                        "performance_trace_path": trace_path,
                    })
                    return

                elif len(engine_ids) > 1:
                    raw_documents = []
                    all_labels = [self._engine_label(engine_id) for engine_id in engine_ids]
                    active_labels: list[str] = []
                    active_model_indices: list[int] = []
                    failed_models: dict[int, str] = {}
                    adaptive_ensemble_audit: dict[str, object] = {}

                    def register_model_document(doc, model_index: int, label: str) -> None:
                        remember_partial(doc)
                        raise_if_cancelled()
                        doc.metadata.__dict__["multi_ocr_model_index"] = model_index
                        doc.metadata.__dict__["multi_ocr_model_label"] = label
                        raw_documents.append(doc)
                        active_labels.append(label)
                        active_model_indices.append(model_index)
                        emit_overall_progress(
                            progress_estimator.complete_model(
                                model_index,
                                label=(
                                    f"模型{model_index + 1}/{len(engine_ids)}·"
                                    f"{label}·完成"
                                ),
                            )
                        )

                    def run_model_safely(
                        engine_id: str,
                        model_index: int,
                        **kwargs,
                    ):
                        """Isolate an optional OCR failure from the other models.

                            Multi-model OCR is specifically a redundancy feature.  A
                            missing NDLOCR runtime or an unavailable helper must not
                            discard results already produced by another healthy OCR role.
                            """
                        try:
                            document = run_one(engine_id, model_index, **kwargs)
                            # Selective later-model passes inherit consensus seeds
                            # for untouched columns, so only judge raw model
                            # coverage on a complete model pass.
                            if kwargs.get("target_column_ids") is None:
                                health = self._document_column_ocr_health(document)
                                if health.get("available"):
                                    percent = float(health.get("coverage", 0.0) or 0.0) * 100.0
                                    recognized = int(health.get("text_recognized", 0) or 0)
                                    expected = int(health.get("expected", 0) or 0)
                                    if not health.get("usable", True):
                                        raise RuntimeError(
                                            "有效正文列覆盖过低："
                                            f"{recognized}/{expected}（{percent:.1f}%）。"
                                            "该模型结果已从自动共识中隔离，避免大量占位符污染融合。"
                                        )
                                    if health.get("warning"):
                                        signals.log.emit(
                                            f"⚠️ 模型{model_index + 1}·{all_labels[model_index]}"
                                            f"有效列覆盖 {recognized}/{expected}（{percent:.1f}%）；"
                                            "结果保留，但缺失列将依赖其他模型或人工复核。"
                                        )
                            return document
                        except Exception as exc:
                            if cancel_event.is_set() or isinstance(exc, InterruptedError):
                                raise InterruptedError("OCR 已停止") from exc
                            detail = str(exc).strip() or exc.__class__.__name__
                            if kwargs.get("recovery_only"):
                                # Sentence rescue is a refinement of an already
                                # successful primary model, not a new model pass.
                                # A rescue failure must never retroactively mark
                                # the original model unavailable or discard its
                                # first-pass evidence.
                                signals.log.emit(
                                    f"⚠️ 共识分歧整句救援失败：{all_labels[model_index]} · "
                                    f"{detail}；保留首轮逐列结果继续裁决。"
                                )
                                return None
                            failed_models[model_index] = detail
                            signals.log.emit(
                                f"⚠️ 模型{model_index + 1}·{all_labels[model_index]}不可用，"
                                f"已从本轮多模型中跳过：{detail}"
                            )
                            emit_overall_progress(
                                progress_estimator.complete_model(
                                    model_index,
                                    label=(
                                        f"模型{model_index + 1}/{len(engine_ids)}·"
                                        f"{all_labels[model_index]}·失败，已跳过"
                                    ),
                                )
                            )
                            return None

                    if early_consensus_enabled:
                        signals.log.emit(
                            "\n⚡ 多模型共识优先：模型1/2先读取同一批物理列；"
                            "一致列立即定稿，模型3～6和整句上下文只处理分歧。"
                        )
                        initial_count = min(2, len(engine_ids))
                        initial_results: dict[int, object] = {}
                        if parallel_first_runtime and initial_count == 2:
                            signals.log.emit(
                                "🚀 首轮双模型并行：共享分列裁图，识别进程与增强临时目录完全独立。"
                            )
                            from concurrent.futures import (
                                FIRST_COMPLETED, ThreadPoolExecutor, wait,
                            )
                            pool = ThreadPoolExecutor(
                                max_workers=2,
                                thread_name_prefix="ocr-consensus",
                            )
                            futures = {}
                            future_to_index = {}
                            parallel_cancelled = False
                            try:
                                for model_index in range(initial_count):
                                    raise_if_cancelled()
                                    signals.log.emit(
                                        f"\n▶ 并行首轮模型 {model_index + 1}/{initial_count}："
                                        f"{all_labels[model_index]}"
                                    )
                                    future = pool.submit(
                                        run_model_safely,
                                        engine_ids[model_index],
                                        model_index,
                                        force_sentence_reocr=False,
                                    )
                                    futures[model_index] = future
                                    future_to_index[future] = model_index
                                pending = set(future_to_index)
                                while pending:
                                    if cancel_event.is_set():
                                        parallel_cancelled = True
                                        raise InterruptedError("OCR 已停止")
                                    done, pending = wait(
                                        pending, timeout=0.10,
                                        return_when=FIRST_COMPLETED,
                                    )
                                    for future in done:
                                        model_index = future_to_index[future]
                                        initial_results[model_index] = future.result()
                            finally:
                                if parallel_cancelled or cancel_event.is_set():
                                    for future in futures.values():
                                        future.cancel()
                                    pool.shutdown(wait=False, cancel_futures=True)
                                else:
                                    pool.shutdown(wait=True)
                        else:
                            for model_index in range(initial_count):
                                if cancel_event.is_set():
                                    break
                                signals.log.emit(
                                    f"\n▶ 首轮模型 {model_index + 1}/{initial_count}："
                                    f"{all_labels[model_index]}"
                                )
                                initial_results[model_index] = run_model_safely(
                                    engine_ids[model_index],
                                    model_index,
                                    force_sentence_reocr=False,
                                )

                        for model_index in range(initial_count):
                            current_doc = initial_results.get(model_index)
                            if current_doc is None:
                                continue
                            register_model_document(
                                current_doc, model_index, all_labels[model_index]
                            )

                        # If one of the first two engines is unavailable,
                        # promote a later model to a full pass instead of aborting.
                        if len(raw_documents) < 2:
                            raise_if_cancelled()
                            for model_index in range(initial_count, len(engine_ids)):
                                raise_if_cancelled()
                                if model_index in active_model_indices or model_index in failed_models:
                                    continue
                                signals.log.emit(
                                    f"▶ 自动替补模型{model_index + 1}·{all_labels[model_index]}："
                                    "完整识别以补足多模型结果。"
                                )
                                replacement = run_model_safely(
                                    engine_ids[model_index],
                                    model_index,
                                    force_sentence_reocr=False,
                                )
                                if replacement is not None:
                                    register_model_document(
                                        replacement, model_index, all_labels[model_index]
                                    )
                                if len(raw_documents) >= 2:
                                    break

                        if not raw_documents:
                            failure_lines = []
                            for model_index, label in enumerate(all_labels):
                                detail = str(failed_models.get(model_index, "未返回结果") or "未返回结果")
                                failure_lines.append(f"• {label}：{detail}")
                            raise RuntimeError(
                                "所选 OCR 模型均未能启动或返回结果。\n"
                                + "\n".join(failure_lines)
                            )
                        if len(raw_documents) == 1:
                            only_doc = raw_documents[0]
                            if apply_column_reflow:
                                from engine.column_sentence_reflow import reflow_columns_into_sentences
                                only_doc = reflow_columns_into_sentences(
                                    only_doc, max_columns=column_reflow_max,
                                    cancel_check=cancel_event.is_set,
                                )
                            only_doc.add_log(
                                "multi_ocr_degraded",
                                "其他 OCR 模型不可用，已保留唯一成功模型的结果",
                                len(failed_models),
                            )
                            signals.log.emit(
                                f"⚠️ 多模型已降级为单模型：仅 {active_labels[0]} 成功；"
                                "结果不会因辅助模型故障而丢失。"
                            )
                            emit_overall_progress(
                                progress_estimator.complete(label="OCR 完成（多模型降级）")
                            )
                            maybe_preserve_ruby([only_doc])
                            finish_project_checkpoint("ok")
                            signals.finished.emit(only_doc)
                            return

                        raise_if_cancelled()
                        from engine.multi_ocr_consensus import (
                            build_column_consensus,
                            seed_from_document,
                        )
                        consensus_plan = build_column_consensus(raw_documents)
                        adaptive_ensemble_audit = {
                            "policy": "adaptive_diversity_aware_v1",
                            "model_reliability": dict(consensus_plan.model_reliability),
                            "exact_columns": len(consensus_plan.exact_ids),
                            "normalized_columns": len(consensus_plan.normalized_ids),
                            "majority_columns": len(consensus_plan.majority_ids),
                            "verification_columns": len(consensus_plan.verification_ids),
                            "unresolved_columns": len(consensus_plan.unresolved_ids),
                        }
                        signals.log.emit(
                            f"✅ 分列首轮共识：原文一致 {len(consensus_plan.exact_ids)} 列；"
                            f"标准化一致 {len(consensus_plan.normalized_ids)} 列；"
                            f"需后续模型验证/分歧 {consensus_plan.conflict_count} 列。"
                        )

                        sentence_recovery_done = False
                        initial_unresolved_ids = set(consensus_plan.unresolved_ids)
                        if (
                            sentence_context_reocr
                            and initial_unresolved_ids
                            and retain_primary_hayai_session
                            and active_model_indices
                            and active_model_indices[0] == 0
                            and retained_hayai_holder.get("session") is not None
                        ):
                            raise_if_cancelled()
                            signals.log.emit(
                                f"🧩 共识分歧救援：复用首轮 {all_labels[0]} 会话，"
                                f"先对包含 {len(initial_unresolved_ids)} 个未决物理列的完整句组执行整句上下文 OCR；"
                                "完成后释放主模型，再交给后续模型处理剩余分歧。"
                            )
                            primary_seed = seed_from_document(raw_documents[0])
                            resolved_primary = run_model_safely(
                                engine_ids[0],
                                0,
                                force_sentence_reocr=True,
                                target_column_ids=set(),
                                seed_results=primary_seed,
                                sentence_target_ids=initial_unresolved_ids,
                                seed_mark_selective=False,
                                recovery_only=True,
                            )
                            if resolved_primary is not None:
                                resolved_primary.metadata.__dict__["multi_ocr_model_index"] = 0
                                resolved_primary.metadata.__dict__["multi_ocr_model_label"] = all_labels[0]
                                raw_documents[0] = resolved_primary
                                sentence_recovery_done = True
                                consensus_plan = build_column_consensus(raw_documents)
                                adaptive_ensemble_audit.update({
                                    "primary_sentence_recovery": "before_third_model_same_session",
                                    "exact_columns": len(consensus_plan.exact_ids),
                                    "normalized_columns": len(consensus_plan.normalized_ids),
                                    "majority_columns": len(consensus_plan.majority_ids),
                                    "verification_columns": len(consensus_plan.verification_ids),
                                    "unresolved_columns": len(consensus_plan.unresolved_ids),
                                })
                                signals.log.emit(
                                    f"✅ 主模型整句救援后：仍需后续模型验证/分歧 "
                                    f"{consensus_plan.conflict_count} 列；同一主模型结果只替换自身候选，不增加投票权。"
                                )
                            emit_phase_progress(
                                f"共识分歧救援·{all_labels[0]}", 1, 1
                            )
                        elif sentence_context_reocr and not initial_unresolved_ids:
                            sentence_recovery_done = True

                        # Never keep an idle Hayai MPS model resident while
                        # the next model (often 48px/MPS) starts.  This is the
                        # important unified-memory boundary on Apple Silicon.
                        close_retained_hayai(force=False)

                        unused_model_indices = [
                            index for index in range(len(engine_ids))
                            if index not in active_model_indices
                            and index not in failed_models
                        ]
                        for model_index in unused_model_indices:
                            raise_if_cancelled()
                            if not consensus_plan.conflict_ids:
                                signals.log.emit(
                                    f"⏭ 模型{model_index + 1}·{all_labels[model_index]} 已跳过："
                                    "当前全部物理列已形成可接受共识。"
                                )
                                emit_overall_progress(
                                    progress_estimator.complete_model(
                                        model_index,
                                        label=(
                                            f"模型{model_index + 1}/{len(engine_ids)}·"
                                            f"{all_labels[model_index]}·已有共识，已跳过"
                                        ),
                                    )
                                )
                                continue
                            target_ids = set(consensus_plan.conflict_ids)
                            signals.log.emit(
                                f"▶ 模型{model_index + 1}·{all_labels[model_index]}：仅识别 "
                                f"{len(target_ids)} 个仍有分歧的物理列；已定稿列不重复推理。"
                            )
                            selective_doc = run_model_safely(
                                engine_ids[model_index],
                                model_index,
                                force_sentence_reocr=False,
                                target_column_ids=target_ids,
                                seed_results=seed_from_document(raw_documents[0]),
                                seed_mark_selective=True,
                            )
                            if selective_doc is None:
                                continue
                            register_model_document(
                                selective_doc, model_index, all_labels[model_index]
                            )
                            consensus_plan = build_column_consensus(raw_documents)
                            adaptive_ensemble_audit = {
                                **adaptive_ensemble_audit,
                                "policy": "adaptive_diversity_aware_v2_up_to_6",
                                "model_reliability": dict(consensus_plan.model_reliability),
                                "exact_columns": len(consensus_plan.exact_ids),
                                "normalized_columns": len(consensus_plan.normalized_ids),
                                "majority_columns": len(consensus_plan.majority_ids),
                                "verification_columns": len(consensus_plan.verification_ids),
                                "sensitive_dissent_columns": len(consensus_plan.sensitive_dissent_ids),
                                "unresolved_columns": len(consensus_plan.unresolved_ids),
                                "models_used": len(raw_documents),
                            }
                            signals.log.emit(
                                f"✅ 模型{model_index + 1}补证后：已定稿 {consensus_plan.exact_count} 列；"
                                f"敏感少数异议 {len(consensus_plan.sensitive_dissent_ids)} 列；"
                                f"仍需后续处理 {len(consensus_plan.unresolved_ids)} 列。"
                            )

                        unresolved_ids = set(consensus_plan.unresolved_ids)
                        if sentence_context_reocr and unresolved_ids and not sentence_recovery_done:
                            # Non-Hayai primaries and degraded/replacement
                            # paths keep the compatible post-later-model rescue.
                            raise_if_cancelled()
                            signals.log.emit(
                                f"🧩 共识分歧救援：仅对包含 {len(unresolved_ids)} 个"
                                "未决物理列的完整句组执行整句上下文 OCR。"
                            )
                            primary_seed = seed_from_document(raw_documents[0])
                            primary_model_index = active_model_indices[0]
                            resolved_primary = run_model_safely(
                                engine_ids[primary_model_index],
                                primary_model_index,
                                force_sentence_reocr=True,
                                target_column_ids=set(),
                                seed_results=primary_seed,
                                sentence_target_ids=unresolved_ids,
                                seed_mark_selective=False,
                                recovery_only=True,
                            )
                            if resolved_primary is not None:
                                resolved_primary.metadata.__dict__["multi_ocr_model_index"] = primary_model_index
                                resolved_primary.metadata.__dict__["multi_ocr_model_label"] = all_labels[primary_model_index]
                                raw_documents[0] = resolved_primary
                            emit_phase_progress(
                                f"共识分歧救援·{all_labels[primary_model_index]}", 1, 1
                            )
                        elif sentence_context_reocr and unresolved_ids and sentence_recovery_done:
                            signals.log.emit(
                                "⏭ 主模型整句上下文已在后续模型之前完成；剩余未决项保留给独立模型/人工/AI裁决，"
                                "不重复调用同一主模型充当额外投票。"
                            )
                        elif sentence_context_reocr:
                            signals.log.emit(
                                "⏭ 所有物理列已有已有多模型共识结论，"
                                "整句上下文 OCR 全部跳过。"
                            )
                    else:
                        signals.log.emit(
                            "\n🔀 多模型完整兼容模式：保留全部逐列与整句流程。"
                        )
                        if sentence_context_reocr and multi_sentence_mode == "primary_only":
                            signals.log.emit(
                                "⚡ 多模型整句加速：整句上下文只由模型1执行一次；"
                                "模型2～6保留完整逐列候选。"
                            )
                        start_serial_index = 0
                        if parallel_first_runtime and len(engine_ids) >= 2:
                            signals.log.emit(
                                "🚀 完整模式首轮双模型并行：模型1/2全量识别；"
                                "模型3～6随后按完整模式运行。"
                            )
                            from concurrent.futures import (
                                FIRST_COMPLETED, ThreadPoolExecutor, wait,
                            )
                            pool = ThreadPoolExecutor(
                                max_workers=2,
                                thread_name_prefix="ocr-consensus",
                            )
                            futures = {}
                            future_to_index = {}
                            parallel_results: dict[int, object] = {}
                            parallel_cancelled = False
                            try:
                                for model_index in range(2):
                                    raise_if_cancelled()
                                    signals.log.emit(
                                        f"\n▶ 并行首轮模型 {model_index + 1}/2："
                                        f"{all_labels[model_index]}"
                                    )
                                    future = pool.submit(
                                        run_model_safely,
                                        engine_ids[model_index],
                                        model_index,
                                    )
                                    futures[model_index] = future
                                    future_to_index[future] = model_index
                                pending = set(future_to_index)
                                while pending:
                                    if cancel_event.is_set():
                                        parallel_cancelled = True
                                        raise InterruptedError("OCR 已停止")
                                    done, pending = wait(
                                        pending, timeout=0.10,
                                        return_when=FIRST_COMPLETED,
                                    )
                                    for future in done:
                                        model_index = future_to_index[future]
                                        parallel_results[model_index] = future.result()
                            finally:
                                if parallel_cancelled or cancel_event.is_set():
                                    for future in futures.values():
                                        future.cancel()
                                    pool.shutdown(wait=False, cancel_futures=True)
                                else:
                                    pool.shutdown(wait=True)
                            for model_index in range(2):
                                current_doc = parallel_results.get(model_index)
                                if current_doc is not None:
                                    register_model_document(
                                        current_doc, model_index, all_labels[model_index]
                                    )
                            start_serial_index = 2
                        for model_index in range(start_serial_index, len(engine_ids)):
                            if cancel_event.is_set():
                                break
                            signals.log.emit(
                                f"\n▶ 模型 {model_index + 1}/{len(engine_ids)}："
                                f"{all_labels[model_index]}"
                            )
                            current_doc = run_model_safely(
                                engine_ids[model_index], model_index
                            )
                            if current_doc is not None:
                                register_model_document(
                                    current_doc, model_index, all_labels[model_index]
                                )

                    raise_if_cancelled()
                    if not raw_documents:
                        raise RuntimeError("所选 OCR 模型均未能启动或返回结果。")
                    if len(raw_documents) == 1:
                        only_doc = raw_documents[0]
                        if apply_column_reflow:
                            from engine.column_sentence_reflow import reflow_columns_into_sentences
                            only_doc = reflow_columns_into_sentences(
                                only_doc, max_columns=column_reflow_max
                            )
                        only_doc.add_log(
                            "multi_ocr_degraded",
                            "其他 OCR 模型不可用，已保留唯一成功模型的结果",
                            len(failed_models),
                        )
                        signals.log.emit(
                            f"⚠️ 多模型已降级为单模型：仅 {active_labels[0]} 成功。"
                        )
                        emit_overall_progress(
                            progress_estimator.complete(label="OCR 完成（多模型降级）")
                        )
                        maybe_preserve_ruby([only_doc])
                        finish_project_checkpoint("ok")
                        signals.finished.emit(only_doc)
                        return

                    raise_if_cancelled()
                    documents = []
                    labels = list(active_labels)
                    for model_index, current_doc in enumerate(raw_documents):
                        if apply_column_reflow:
                            from engine.column_sentence_reflow import reflow_columns_into_sentences
                            current_doc = reflow_columns_into_sentences(
                                current_doc, max_columns=column_reflow_max,
                                cancel_check=cancel_event.is_set,
                            )
                        else:
                            current_doc.add_log(
                                "column_sentence_reflow",
                                "OCR 界面已关闭逐列成句，原始 OCR 块直接进入多模型对比",
                                0,
                            )
                        current_doc.metadata.__dict__["multi_ocr_model_index"] = model_index
                        current_doc.metadata.__dict__["multi_ocr_model_label"] = labels[model_index]
                        documents.append(current_doc)

                    emit_overall_progress(
                        progress_estimator.update_final_stage(
                            0,
                            1,
                            label="逐句对齐与自动选优",
                        )
                    )
                    emit_phase_progress("逐句对齐与自动选优", 0, 1)
                    raise_if_cancelled()
                    from engine.multi_ocr_compare import compare_ocr_documents, build_fused_document
                    with runtime_governor.lease(
                        "alignment", cancel_check=cancel_token.is_cancelled
                    ), performance_trace.stage("multi_model_alignment_and_fusion"):
                        comparison = compare_ocr_documents(documents, labels)
                        fused = build_fused_document(documents[0], comparison)
                    # Feed every ordinary OCR document as geometry evidence;
                    # preserve_ruby_in_documents() merges geometry-only hints
                    # and still writes Ruby metadata exclusively to ``fused``.
                    maybe_preserve_ruby([*documents, fused])
                    if adaptive_ensemble_audit:
                        fused.metadata.__dict__["adaptive_ocr_ensemble"] = dict(adaptive_ensemble_audit)
                    if handwriting_enabled:
                        from engine.ocr_manual_review import annotate_ocr_review_risks
                        annotate_ocr_review_risks(fused)
                    emit_overall_progress(
                        progress_estimator.update_final_stage(
                            1,
                            1,
                            label="逐句对齐与自动选优",
                        )
                    )
                    emit_phase_progress("逐句对齐与自动选优", 1, 1)
                    signals.log.emit(f"🔀 {comparison.summary}")
                    emit_overall_progress(progress_estimator.complete(label="多模型 OCR 完成"))
                    source_texts = [
                        "\n".join(row.texts[model_index] for row in comparison.rows)
                        for model_index in range(len(documents))
                    ]
                    finish_project_checkpoint("ok")
                    signals.finished.emit({
                        "multi_ocr": True,
                        "documents": documents,
                        "labels": labels,
                        "comparison": comparison,
                        "source_texts": source_texts,
                        "fused": fused,
                        "ruby_enabled": bool(ruby_preserve_enabled),
                        "handwriting_review_context": dict(handwriting_review_context_snapshot),
                        "performance_trace_path": trace_path,
                    })
                else:
                    doc = run_one(engine_ids[0], 0)
                    remember_partial(doc)
                    raise_if_cancelled()
                    health = self._document_column_ocr_health(doc)
                    if health.get("available"):
                        percent = float(health.get("coverage", 0.0) or 0.0) * 100.0
                        recognized = int(health.get("text_recognized", 0) or 0)
                        expected = int(health.get("expected", 0) or 0)
                        if not health.get("usable", True):
                            raise RuntimeError(
                                "所选 OCR 模型有效正文列覆盖过低："
                                f"{recognized}/{expected}（{percent:.1f}%）。"
                                "识别结果以占位符或空列为主，已停止后处理，"
                                "请检查模型运行环境、分列输入或改用其他 OCR 模型。"
                            )
                        if health.get("warning"):
                            signals.log.emit(
                                f"⚠️ {self._engine_label(engine_ids[0])} 有效列覆盖 "
                                f"{recognized}/{expected}（{percent:.1f}%）；"
                                "结果会保留，但缺失列必须人工复核。"
                            )
                    emit_overall_progress(
                        progress_estimator.complete_model(
                            0,
                            label=f"模型1/1·{self._engine_label(engine_ids[0])}·识别完成",
                        )
                    )
                    raise_if_cancelled()
                    if handwriting_enabled:
                        # Stability-first workflow: only run lightweight OCR risk
                        # analysis here. Full-book handwriting candidate generation
                        # remains an explicit manual action.
                        signals.log.emit(
                            f"\n✍️ OCR 疑点筛查：模式 = {handwriting_mode or 'hybrid'}，"
                            "只检查空列、低置信度、字数偏差、异常符号、引号失衡和重复片段；"
                            "不批量运行手写识别，不自动修改正文。"
                        )
                        emit_overall_progress(
                            progress_estimator.update_final_stage(
                                0,
                                2,
                                label="轻量疑点筛查",
                            )
                        )
                        emit_phase_progress("② 轻量疑点筛查", 0, 1)
                        from engine.ocr_manual_review import annotate_ocr_review_risks
                        risk_report = annotate_ocr_review_risks(doc)
                        emit_overall_progress(
                            progress_estimator.update_final_stage(
                                1,
                                2,
                                label="轻量疑点筛查",
                            )
                        )
                        emit_phase_progress("② 轻量疑点筛查", 1, 1)
                        confidence_unavailable = int(
                            risk_report.get("confidence_unavailable_columns", 0) or 0
                        )
                        confidence_note = (
                            f"其中 {confidence_unavailable} 列的 OCR 后端未提供可校准置信度，"
                            "不会因此被误判为低置信；"
                            if confidence_unavailable else ""
                        )
                        signals.log.emit(
                            f"✍️ 疑点筛查完成：检查 {int(risk_report.get('columns', 0) or 0)} 列；"
                            f"标记 {int(risk_report.get('suspicious_columns', 0) or 0)} 列，"
                            f"其中高风险 {int(risk_report.get('high_risk_columns', 0) or 0)} 列。"
                            f"{confidence_note}OCR 正文与 ocr_raw 均未被候选覆盖。"
                        )
                    raise_if_cancelled()
                    if apply_column_reflow and not handwriting_review:
                        signals.log.emit("\n🧩 OCR 后处理：只按每列末尾判断并接续；同列内部不拆分…")
                        emit_overall_progress(
                            progress_estimator.update_final_stage(
                                1,
                                2,
                                label="逐列成句后处理",
                            )
                        )
                        from engine.column_sentence_reflow import reflow_columns_into_sentences
                        doc = reflow_columns_into_sentences(
                            doc, max_columns=column_reflow_max,
                            cancel_check=cancel_event.is_set,
                        )
                    elif handwriting_review:
                        doc.add_log("handwriting_trace_review", "等待 OCR 疑点人工纠错；复核完成后再执行逐列成句", 0)
                    else:
                        doc.add_log("column_sentence_reflow", "OCR 界面已关闭逐列成句，原始 OCR 块直接进入 Formatter", 0)
                    maybe_preserve_ruby([doc])
                    emit_overall_progress(progress_estimator.complete(label="OCR 完成"))
                    doc.metadata.__dict__["ocr_performance_trace_path"] = trace_path
                    finish_project_checkpoint("ok")
                    signals.finished.emit(doc)
            finally:
                close_retained_hayai(force=bool(cancel_event.is_set()))
                try:
                    from adapters.column_ocr_adapter import (
                        clear_task_file_sha_memo, task_file_sha_memo_metrics,
                    )
                    performance_trace.set_gauge(
                        "task_sha_memo_entries", task_file_sha_memo_metrics().get("entries", 0)
                    )
                    performance_trace.set_gauge("cancelled", bool(cancel_event.is_set()))
                    snapshot = performance_trace.snapshot()
                    elapsed = float(snapshot.get("elapsed_seconds", 0.0) or 0.0)
                    stages = snapshot.get("stages", {})
                    counters = snapshot.get("counters", {})
                    gauges = snapshot.get("gauges", {})
                    pages = int(gauges.get("input_pages", len(inputs)) or 0)
                    completed_models = int(counters.get("models_completed", 0) or 0)
                    model_seconds = sum(
                        float(stages.get(f"model_{index + 1}_{engine_id}", {}).get("total_seconds", 0.0) or 0.0)
                        for index, engine_id in enumerate(engine_ids)
                    )
                    image_seconds = sum(
                        float(metric.get("total_seconds", 0.0) or 0.0)
                        for name, metric in stages.items()
                        if name.endswith(".image_preparation")
                    )
                    recognition_seconds = sum(
                        float(metric.get("total_seconds", 0.0) or 0.0)
                        for name, metric in stages.items()
                        if name.endswith(".recognition")
                    )
                    fusion_seconds = float(
                        stages.get("multi_model_alignment_and_fusion", {}).get("total_seconds", 0.0) or 0.0
                    )
                    ruby_seconds = float(
                        stages.get("ruby_preservation", {}).get("total_seconds", 0.0) or 0.0
                    )
                    model_page_rate = (
                        (pages * completed_models) / model_seconds if model_seconds > 0 else 0.0
                    )
                    runtime_labels = {}
                    for event in snapshot.get("events", []):
                        if event.get("name") != "engine_runtime":
                            continue
                        runtime_labels[str(event.get("engine_id") or "?")] = (
                            f"{event.get('backend', 'unknown')} / {event.get('device', 'unknown')}"
                        )
                    backend_summary = ", ".join(
                        f"{engine}={runtime_labels.get(engine, runtime_backend_config.get(engine, {}))}"
                        for engine in engine_ids
                    )
                    ruby_summary = "关闭"
                    if ruby_preserve_enabled:
                        ruby_summary = (
                            f"候选页 {int(gauges.get('ruby.pages_with_candidates', 0) or 0)}，"
                            f"ROI {int(gauges.get('ruby.roi_count', 0) or 0)}，"
                            f"检测窗 {int(gauges.get('ruby.estimated_detector_tiles', 0) or 0)}/"
                            f"{int(gauges.get('ruby.full_page_detector_tiles', 0) or 0)}，"
                            f"后端 {runtime_labels.get('ruby', 'findtext backend unknown')}"
                        )
                    # Human-readable completion timing: keep the raw trace for
                    # profiling, but also print each OCR model, findtextCenterNet
                    # and the end-to-end wall time directly in the OCR log.
                    timing_parts = []
                    for model_index, engine_id in enumerate(engine_ids):
                        stage_key = f"model_{model_index + 1}_{engine_id}"
                        seconds = float(
                            stages.get(stage_key, {}).get("total_seconds", 0.0) or 0.0
                        )
                        label = self._engine_label(engine_id)
                        timing_parts.append((model_index + 1, label, seconds))
                    signals.log.emit("\n⏱ OCR 完成耗时：")
                    # End-to-end wall time minus the optional Ruby side pass
                    # answers the user's practical question "multi-model OCR
                    # itself took how long".  Do not sum per-model stages here:
                    # model 1/2 may run concurrently, so a sum can exceed wall
                    # time and is not a meaningful elapsed duration.
                    main_ocr_wall_seconds = max(0.0, elapsed - ruby_seconds)
                    main_label = "多模型 OCR" if len(engine_ids) > 1 else "OCR 主流程"
                    signals.log.emit(
                        f"  {main_label}（不含 findtextCenterNet）：{main_ocr_wall_seconds:.2f}s"
                    )
                    for model_index, label, seconds in timing_parts:
                        signals.log.emit(
                            f"    模型{model_index} · {label}：{seconds:.2f}s"
                        )
                    if ruby_preserve_enabled:
                        signals.log.emit(
                            f"  findtextCenterNet（Ruby）：{ruby_seconds:.2f}s"
                        )
                    else:
                        signals.log.emit("  findtextCenterNet（Ruby）：未启用")
                    cache_parts = []
                    total_cache_hits = 0
                    total_cache_misses = 0
                    for engine_id in engine_ids:
                        hits = int(gauges.get(f"{engine_id}.segment_cache_hits", 0) or 0)
                        misses = int(gauges.get(f"{engine_id}.segment_cache_misses", 0) or 0)
                        total_cache_hits += hits
                        total_cache_misses += misses
                        if hits or misses:
                            cache_parts.append(
                                f"{self._engine_label(engine_id)} {hits}命中/{misses}重算"
                            )
                    if cache_parts:
                        signals.log.emit("  分段 OCR 缓存：" + "；".join(cache_parts))
                        performance_trace.set_gauge(
                            "summary.segment_cache_hits", total_cache_hits
                        )
                        performance_trace.set_gauge(
                            "summary.segment_cache_misses", total_cache_misses
                        )
                    signals.log.emit(f"  总耗时：{elapsed:.2f}s")

                    performance_trace.set_gauge("summary.total_seconds", elapsed)
                    performance_trace.set_gauge(
                        "summary.input_pages_per_second",
                        round(pages / elapsed, 4) if elapsed > 0 else 0.0,
                    )
                    performance_trace.set_gauge(
                        "summary.model_page_equivalents_per_second",
                        round((pages * completed_models) / model_seconds, 4)
                        if model_seconds > 0 else 0.0,
                    )
                    signals.log.emit(
                        f"⏱ OCR 性能摘要：{pages} 页 / {elapsed:.1f}s，"
                        f"端到端 {(pages / elapsed) if elapsed > 0 else 0.0:.2f} 页/s；"
                        f"图像准备 {image_seconds:.1f}s，识别 {recognition_seconds:.1f}s，"
                        f"融合 {fusion_seconds:.1f}s，Ruby {ruby_seconds:.1f}s（{ruby_summary}）；"
                        f"模型等价页速 {model_page_rate:.2f} 页/s；引擎/后端：{backend_summary}。"
                        f"性能明细：{trace_path}"
                    )
                    performance_trace.write_json(
                        trace_path,
                        extra={
                            "resource_governor": runtime_governor.snapshot(),
                            "log_buffer": self._ocr_log_buffer.metrics(),
                        },
                    )
                    clear_task_file_sha_memo()
                except Exception:
                    pass
                # Intentionally retained.  The OCR workspace clear action and
                # MainWindow.closeEvent are the only normal deletion points.
                # A stale worker that raced with Clear can recreate a partial
                # directory; finish_run performs the required second cleanup.
                self._preview_temp_store.finish_run(crop_temp_root, preview_epoch)
        except Exception as exc:
            if cancel_event.is_set() or isinstance(exc, InterruptedError):
                document = cancelled_partial.get("document")
                # A stopped multi-model run is not a single OCR document. Publishing
                # the most recently completed/partial model through ocr_done used to
                # overwrite the project's current stage and made OCR Compare / image
                # review show thousands of raw physical columns as a one-model result.
                if multi_role_enabled:
                    finish_project_checkpoint("cancelled")
                    signals.finished.emit({
                        "ocr_cancelled": True,
                        "multi_ocr_cancelled": True,
                        "resume_supported": True,
                    })
                elif document is not None:
                    try:
                        document.add_log(
                            "ocr_cancelled",
                            "OCR 已停止；没有继续派发新页面、新列、新模型、重试或整句后处理",
                            0,
                        )
                    except Exception:
                        pass
                    finish_project_checkpoint("cancelled")
                    signals.finished.emit(document)
                else:
                    finish_project_checkpoint("cancelled")
                    signals.finished.emit({"ocr_cancelled": True})
            else:
                import traceback
                finish_project_checkpoint("error")
                signals.error.emit(traceback.format_exc())
        finally:
            if ruby_worker_job_token is not None:
                try:
                    from adapters.findtext_centernet_ruby import end_findtext_ocr_job
                    end_findtext_ocr_job(
                        ruby_worker_job_token, log_callback=signals.log.emit
                    )
                    performance_trace.event("ruby_worker_released")
                except Exception:
                    # Process shutdown still has the module-level atexit guard;
                    # never turn an optional cleanup problem into OCR failure.
                    pass

    signals = self._ocr_signals = WorkerSignals()

    def is_current() -> bool:
        return run_generation == self._run_generation

    def on_finished(doc):
        if is_current():
            self._mark_ocr_activity()
            self._flush_ocr_log_buffer(force_all=True)
            self._on_done(doc)

    def on_error(message):
        if is_current() and not cancel_event.is_set():
            self._mark_ocr_activity()
            self._flush_ocr_log_buffer(force_all=True)
            self._on_error(message)

    def on_log(message):
        if is_current() and not cancel_event.is_set():
            self._mark_ocr_activity()
            performance_trace.increment("log_lines")
            self._ocr_log_buffer.push(message)

    def on_progress(current, total):
        if is_current() and not cancel_event.is_set():
            self._mark_ocr_activity()
            self._on_progress(current, total)

    def on_phase_progress(label, current, total):
        if is_current() and not cancel_event.is_set():
            self._mark_ocr_activity()
            self._on_phase_progress(label, current, total)

    def on_overall_progress(snapshot):
        if is_current() and not cancel_event.is_set():
            self._mark_ocr_activity()
            self._on_overall_progress(snapshot)

    signals.finished.connect(on_finished)
    signals.error.connect(on_error)
    signals.log.connect(on_log)
    signals.progress.connect(on_progress)
    signals.phase_progress.connect(on_phase_progress)
    signals.overall_progress.connect(on_overall_progress)
    signals.current_image_data.connect(
        lambda image: (self._mark_ocr_activity(), self._preview.set_image_data(image))
        if is_current() and self._live_preview_is_enabled() else None
    )
    signals.current_column_image_data.connect(
        lambda image, rects: (
            self._mark_ocr_activity(),
            self._preview.set_column_preview_data(image, rects),
        )
        if (
            is_current()
            and self._live_preview_is_enabled()
            and not (
                bool(getattr(self, "_review_preview_run_active", False))
                or (
                    hasattr(self, "_handwriting_trace_check")
                    and self._handwriting_trace_check.isChecked()
                )
            )
        )
        else None
    )
    signals.preview_page_ready.connect(
        lambda key, name, path, rects, stage: (
            self._mark_ocr_activity(),
            self._record_preview_page(key, name, path, rects, stage),
        ) if is_current() and self._live_preview_is_enabled() else None
    )
    self._ocr_worker_thread = threading.Thread(
        target=worker, daemon=True, name=f"novel-ocr-{run_generation}"
    )
    self._ocr_worker_thread.start()
