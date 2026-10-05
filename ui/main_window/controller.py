#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MainWindow coordination and lifecycle methods kept outside the GUI composition root."""

from __future__ import annotations

import threading
import sys
from pathlib import Path
from functools import partial

from PySide6.QtCore import QEvent, QTimer
from PySide6.QtWidgets import QApplication

from models.document import UnifiedDocument
from utils.paddle_importer import import_paddle_json, import_paddle_md
from ui.common.signals import WorkerSignals
from ui.common.toast import notify
from ui.interface_preferences import THEME_LIGHT, manager_for
from ui.localization import LANG_ZH
from ui.localized_dialogs import LocalizedFileDialog, LocalizedMessageBox
from ui.navigation.sidebar import (
    SECTION_WORKSPACE, SECTION_PAGE, SECTION_OCR, SECTION_FORMAT,
    SECTION_PROOF, SECTION_EPUB, SECTION_SYSTEM,
)
from utils.clear_manager import ClearManager
from core.project_workspace import ProjectWorkspaceManager

QFileDialog = LocalizedFileDialog
QMessageBox = LocalizedMessageBox


class MainWindowControllerMixin:

    def _command_palette_actions(self):
        """Describe existing actions for the lazy global command palette.

        This is only an index over existing owners.  It never duplicates OCR,
        project, adjudication or export behavior.
        """
        from core.command_catalog import CommandSpec
        from ui.command_palette import CommandAction

        mod = "⌘" if sys.platform == "darwin" else "Ctrl+"
        project_active = bool(self._tab_pages.project_manager.active_project)

        def nav(command_id, title, workspace, shortcut="", *, keywords=(), priority=40):
            return CommandAction(
                CommandSpec(
                    command_id=command_id, title=title, category="导航",
                    keywords=tuple(keywords), shortcut=shortcut, priority=priority,
                ),
                partial(self._open_named_workspace, workspace),
            )

        actions = [
            CommandAction(
                CommandSpec(
                    "project.resume", "继续上次工作", "项目",
                    ("resume", "continue", "恢复", "断点", "project"), priority=5,
                ),
                self._tab_workspace.resume_current_stage,
                enabled=project_active,
                disabled_reason="未选择项目" if not project_active else "",
            ),
            nav("nav.workspace", "工作区", "workspace", f"{mod}1", keywords=("workspace", "project", "首页"), priority=10),
            nav("nav.pages", "页面管理", "book", f"{mod}2", keywords=("pages", "page", "图片", "pdf")),
            nav("nav.ocr", "OCR 识别", "ocr", f"{mod}3", keywords=("ocr", "识别", "模型")),
            nav("nav.formatter", "格式处理", "format", f"{mod}4", keywords=("formatter", "正文", "排版")),
            nav("nav.proof", "文字校对", "ocr_compare", f"{mod}5", keywords=("proof", "compare", "裁决", "ai")),
            nav("nav.epub", "EPUB生成", "export", f"{mod}6", keywords=("epub", "export", "导出")),
            nav("nav.settings", "设置", "system", f"{mod}7", keywords=("settings", "配置")),
            nav("nav.pdf", "PDF 文字层", "pdf", keywords=("pdf", "text layer", "文字层"), priority=60),
            nav("nav.image_review", "图文对照", "image_text", keywords=("image", "review", "校对", "原图"), priority=60),
        ]

        project_tools = [
            ("project.backup", "备份项目", self._tab_pages._export_project_backup, ("backup", "zip")),
            ("project.logs", "运行日志", self._tab_pages._open_project_run_logs, ("logs", "history", "历史")),
            ("project.audit", "检查项目", self._tab_pages._audit_active_project, ("audit", "check", "完整性")),
            ("project.optimize", "优化存储", self._tab_pages._optimize_active_project_storage, ("storage", "clean", "cas", "空间")),
            ("project.folder", "项目目录", self._tab_pages._open_active_project_folder, ("folder", "directory", "finder", "explorer")),
        ]
        for command_id, title, callback, keywords in project_tools:
            actions.append(CommandAction(
                CommandSpec(command_id, title, "项目工具", tuple(keywords), priority=80),
                callback, enabled=project_active,
                disabled_reason="未选择项目" if not project_active else "",
            ))
        actions.append(CommandAction(
            CommandSpec(
                "ocr.import_paddle", "📥 导入 PaddleOCR-VL", "OCR 工具",
                ("paddle", "json", "md", "external", "外部ocr"), priority=85,
            ),
            self.import_paddle_output,
        ))
        return tuple(actions)

    def _open_command_palette(self) -> None:
        from ui.command_palette import CommandPalette

        palette = getattr(self, "_command_palette", None)
        if palette is None:
            palette = CommandPalette(self._command_palette_actions, self)
            self._command_palette = palette
        palette.open_palette()

    def _on_project_changed(self, context: dict) -> None:
        data = context if isinstance(context, dict) else {}
        self._active_project_dir = str(data.get("path") or "")
        if hasattr(self, "_runtime_tasks") and hasattr(self, "_tab_workspace"):
            self._tab_workspace.set_runtime_task(
                self._runtime_tasks.latest_for_project(self._active_project_dir)
            )
        self._project_restore_generation += 1
        generation = self._project_restore_generation
        self._pending_project_document = None
        self._pending_project_hydrated.clear()
        self._pending_project_multi_restore = None
        self._project_multi_restore_inflight = False
        if self._active_project_dir:
            export_dir = Path(self._active_project_dir) / "exports" / "epub"
            package_dir = Path(self._active_project_dir) / "exports" / "packages"
            self._pending_project_export_dir = export_dir
            self._pending_project_package_dir = package_dir
            epub = self._lazy_workspace_if_loaded("epub")
            compare = self._lazy_workspace_if_loaded("ocr_compare")
            if epub is not None:
                epub.set_project_export_dir(export_dir)
            if compare is not None:
                compare.set_project_package_dir(package_dir)
            # Let the main window paint before any project restoration starts.
            # The actual JSON/object loading below is performed off the Qt thread.
            QTimer.singleShot(80, lambda g=generation: self._restore_active_project_state(g))
        else:
            self._pending_project_export_dir = None
            self._pending_project_package_dir = None
            epub = self._lazy_workspace_if_loaded("epub")
            compare = self._lazy_workspace_if_loaded("ocr_compare")
            if epub is not None:
                epub.set_project_export_dir(None)
            if compare is not None:
                compare.set_project_package_dir(None)

    def _on_ocr_run_log_event(self, event: dict) -> None:
        """Persist OCR lifecycle and mirror it into the shared runtime task registry.

        The durable run/checkpoint files remain authoritative for crash recovery;
        the in-memory registry only gives every workspace one consistent live
        running/cancelling/terminal state without making the GUI own worker logic.
        """
        data = dict(event or {})
        session_id = str(data.get("session_id") or "")
        if not session_id:
            return
        event_kind = str(data.get("event") or "")
        manager = self._tab_pages.project_manager

        if event_kind == "started":
            details = dict(data.get("details") or {})
            project_path = str(manager.active_project.resolve()) if manager.active_project is not None else ""
            state = self._runtime_tasks.start(
                session_id, str(data.get("stage") or "single_ocr"),
                project_path=project_path,
                message="OCR 正在运行",
                checkpoint_signature=str(details.get("checkpoint_signature") or ""),
                resume_supported=bool(details.get("resume_supported")),
            )
            self._tab_workspace.set_runtime_task(state)
            if manager.active_project is None:
                return
            try:
                workspace_root = str(manager.workspace_root.resolve())
                details["ocr_session_id"] = session_id
                run_id = manager.begin_run(
                    str(data.get("stage") or "single_ocr"),
                    details=details,
                )
                self._ocr_project_run_sessions[session_id] = {
                    "run_id": run_id,
                    "project_path": project_path,
                    "workspace_root": workspace_root,
                }
            except Exception as exc:
                self.statusBar().showMessage(f"OCR 项目日志开始记录失败：{exc}", 8000)
            return

        if event_kind == "cancelling":
            state = self._runtime_tasks.request_cancel(
                session_id, message=str(data.get("message") or "正在停止 OCR")
            )
            if state is not None:
                self._tab_workspace.set_runtime_task(state)
            return

        if event_kind != "finished":
            return

        raw_status = str(data.get("status") or "ok")
        current = self._runtime_tasks.get(session_id)
        resume_supported = bool(
            current is not None and current.resume_supported
            and raw_status.strip().lower() in {"cancelled", "canceled", "error", "failed", "failure"}
        )
        state = self._runtime_tasks.finish(
            session_id, raw_status,
            message=str((data.get("details") or {}).get("summary") or ""),
            error=str(data.get("error") or ""),
            resume_supported=resume_supported,
        )
        if state is not None:
            self._tab_workspace.set_runtime_task(state)

        session = self._ocr_project_run_sessions.pop(session_id, None)
        if not session:
            return
        try:
            log_manager = ProjectWorkspaceManager(session["workspace_root"])
            log_manager.open_project(session["project_path"])
            details = dict(data.get("details") or {})
            details["ocr_session_id"] = session_id
            trace_source = str(data.get("performance_trace_path") or "")
            if trace_source:
                details["runtime_trace_source"] = trace_source
            log_manager.finish_run(
                str(session["run_id"]),
                status=raw_status,
                details=details,
                error=str(data.get("error") or "") or None,
                log_text=str(data.get("log_text") or ""),
                performance=dict(data.get("performance") or {}),
            )
            # Re-read the durable checkpoint/run summary after finalisation.  This
            # lets the dashboard immediately expose a safe "continue" action.
            self._tab_workspace.refresh_project_state()
        except Exception as exc:
            self.statusBar().showMessage(f"OCR 项目日志保存失败：{exc}", 8000)

    def _on_compare_run_log_event(self, event: dict) -> None:
        """Persist long-running OCR-compare/AI work in the shared task model.

        Unlike OCR, GPT-grade adjudication has no resumable disk checkpoint yet.
        The registry therefore exposes running/cancelling/terminal state only;
        durable project Run History remains the source of truth after restart.
        """
        data = dict(event or {})
        session_id = str(data.get("session_id") or "")
        if not session_id:
            return
        event_kind = str(data.get("event") or "")
        stage = str(data.get("stage") or "ai_adjudication")
        manager = self._tab_pages.project_manager

        if event_kind == "started":
            details = dict(data.get("details") or {})
            project_path = str(manager.active_project.resolve()) if manager.active_project is not None else ""
            state = self._runtime_tasks.start(
                session_id, stage,
                project_path=project_path,
                message=str(details.get("summary") or "AI 裁决正在运行"),
                resume_supported=False,
            )
            self._tab_workspace.set_runtime_task(state)
            if manager.active_project is None:
                return
            try:
                workspace_root = str(manager.workspace_root.resolve())
                details["ai_session_id"] = session_id
                run_id = manager.begin_run(stage, details=details)
                self._compare_project_run_sessions[session_id] = {
                    "run_id": run_id,
                    "project_path": project_path,
                    "workspace_root": workspace_root,
                }
            except Exception as exc:
                self.statusBar().showMessage(f"AI 裁决项目日志开始记录失败：{exc}", 8000)
            return

        if event_kind == "cancelling":
            state = self._runtime_tasks.request_cancel(
                session_id, message=str(data.get("message") or "正在停止 AI 裁决")
            )
            if state is not None:
                self._tab_workspace.set_runtime_task(state)
            return

        if event_kind != "finished":
            return

        raw_status = str(data.get("status") or "ok")
        details = dict(data.get("details") or {})
        state = self._runtime_tasks.finish(
            session_id, raw_status,
            message=str(details.get("summary") or ""),
            error=str(data.get("error") or ""),
            resume_supported=False,
        )
        if state is not None:
            self._tab_workspace.set_runtime_task(state)

        session = self._compare_project_run_sessions.pop(session_id, None)
        if not session:
            return
        try:
            log_manager = ProjectWorkspaceManager(session["workspace_root"])
            log_manager.open_project(session["project_path"])
            details["ai_session_id"] = session_id
            log_manager.finish_run(
                str(session["run_id"]),
                status=raw_status,
                details=details,
                error=str(data.get("error") or "") or None,
                log_text=str(data.get("log_text") or ""),
                performance=dict(data.get("performance") or {}),
            )
            self._tab_workspace.refresh_project_state()
        except Exception as exc:
            self.statusBar().showMessage(f"AI 裁决项目日志保存失败：{exc}", 8000)

    def _restore_active_project_state(self, generation: int) -> None:
        """Load a project's heavy document state off the GUI thread.

        Page widgets are already restored by PageManagerTab.  Stage JSON, content-
        addressed objects and adjudication snapshots can be
        tens of MB, so none of those are parsed during application startup.
        """
        if int(generation) != self._project_restore_generation or not self._active_project_dir:
            return
        manager = self._tab_pages.project_manager
        if manager.active_project is None:
            return
        project_path = str(manager.active_project.resolve())
        workspace_root = str(manager.workspace_root.resolve())

        signals = WorkerSignals()
        self._project_restore_signal_refs[int(generation)] = signals
        signals.finished.connect(
            lambda payload, g=int(generation): self._apply_restored_project_state(g, payload)
        )
        signals.error.connect(
            lambda message, g=int(generation): self._project_restore_failed(g, message)
        )

        def worker():
            try:
                restore_manager = ProjectWorkspaceManager(workspace_root)
                restore_manager.open_project(project_path)
                compatible = bool(restore_manager.documents_compatible_with_pages())
                payload = {
                    "project_path": project_path,
                    "compatible": compatible,
                    "current": None,
                    "single_pipeline": {},
                    "multi_pipeline": {},
                    "has_multi": False,
                }
                if compatible:
                    raw_current = restore_manager.load_current_stage_document(
                        require_compatible_pages=True
                    )
                    if raw_current is not None:
                        stage_name, raw_document = raw_current
                        # UnifiedDocument reconstruction is pure Python and safe
                        # off the GUI thread; do it here rather than freezing Qt.
                        payload["current"] = (stage_name, UnifiedDocument.from_dict(raw_document))
                    payload["single_pipeline"] = restore_manager.load_pipeline_config("single_ocr")
                    payload["multi_pipeline"] = restore_manager.load_pipeline_config("multi_ocr")
                    payload["has_multi"] = restore_manager.has_multi_ocr_snapshot()
                signals.finished.emit(payload)
            except Exception:
                import traceback
                signals.error.emit(traceback.format_exc())

        threading.Thread(
            target=worker, daemon=True, name=f"project-restore-{generation}"
        ).start()

    def _project_restore_failed(self, generation: int, message: str) -> None:
        self._project_restore_signal_refs.pop(int(generation), None)
        if int(generation) != self._project_restore_generation:
            return
        self.statusBar().showMessage(f"项目正文恢复失败：{str(message).strip()}", 10000)

    def _apply_restored_project_state(self, generation: int, payload: dict) -> None:
        self._project_restore_signal_refs.pop(int(generation), None)
        if int(generation) != self._project_restore_generation:
            return
        data = dict(payload or {})
        project_path = str(data.get("project_path") or "")
        if not project_path or project_path != str(self._active_project_dir):
            return
        if not bool(data.get("compatible")):
            self.statusBar().showMessage("项目页面已恢复；历史正文与当前页面谱系不一致，未自动载入。", 7000)
            return

        # Restore OCR controls before the next run. Multi config wins only when
        # the saved current pipeline is explicitly multi-model.
        single_state = dict(data.get("single_pipeline") or {})
        multi_state = dict(data.get("multi_pipeline") or {})
        current_record = data.get("current")
        current_stage = str(current_record[0] or "") if current_record is not None else ""
        if current_stage == "ocr_auto_fusion":
            pipeline_state = multi_state or single_state
        else:
            pipeline_state = single_state or multi_state
        if pipeline_state:
            self._pending_ocr_pipeline_state = dict(pipeline_state)
            ocr_tab = self._lazy_workspace_if_loaded("ocr")
            if ocr_tab is not None:
                try:
                    ocr_tab.restore_project_pipeline_state(pipeline_state)
                    self._pending_ocr_pipeline_state = {}
                except Exception as exc:
                    self.statusBar().showMessage(f"OCR 参数恢复失败：{exc}", 7000)

        current = data.get("current")
        if current is not None:
            stage, raw_document = current
            try:
                doc = raw_document if isinstance(raw_document, UnifiedDocument) else UnifiedDocument.from_dict(raw_document)
                self._snapshot_registry.publish_aliases((stage, "current"), doc, clone=False)
                self._current_stage = str(stage or "ocr")
                self._doc = doc
                self._pending_project_document = (self._current_stage, doc)
                self._pending_project_hydrated.clear()
                source_engine = str(getattr(doc.metadata, "source_engine", "") or "")
                processing_log = list(getattr(doc, "processing_log", []) or [])
                cancelled_intermediate_multi = bool(
                    getattr(doc.metadata, "multi_ocr_model_index", 0)
                    and any(
                        str((entry or {}).get("step", "")) == "ocr_cancelled"
                        or "OCR 已停止" in str((entry or {}).get("message", ""))
                        for entry in processing_log if isinstance(entry, dict)
                    )
                )
                if cancelled_intermediate_multi:
                    self._snapshot_registry.clear_alias("current") if hasattr(self._snapshot_registry, "clear_alias") else None
                    self._doc = None
                    self._pending_project_document = None
                    self._pending_project_hydrated.clear()
                    self._pending_single_compare_result = None
                    self._pending_image_review_document = None
                    self.statusBar().showMessage(
                        "上次多模型 OCR 中途停止；已保留断点/分段缓存，但不会把中间模型误当成正文。", 9000
                    )
                elif self._current_stage == "ocr":
                    if source_engine == "pdf_text_layer":
                        # PDF text resumes directly into Formatter/EPUB.  Do not
                        # re-introduce it as a synthetic single-OCR source after
                        # project restore or workspace switching.
                        self._pending_single_compare_result = None
                        self._pending_image_review_document = None
                        compare = self._lazy_workspace_if_loaded("ocr_compare")
                        if compare is not None:
                            compare.set_available_single_result(None)
                    else:
                        source_label = source_engine or "项目恢复 · 单模型 OCR"
                        review_label = "项目恢复 · OCR 原文"
                        self._pending_single_compare_result = (doc, source_label)
                        self._pending_image_review_document = (doc, review_label, True)
                        compare = self._lazy_workspace_if_loaded("ocr_compare")
                        review = self._lazy_workspace_if_loaded("image_review")
                        if compare is not None:
                            compare.set_available_single_result(doc, source_label)
                        if review is not None:
                            review.set_document(doc, review_label, lazy=True)
            except Exception as exc:
                self.statusBar().showMessage(f"项目正文恢复失败：{exc}", 8000)

        if bool(data.get("has_multi")):
            self._pending_project_multi_restore = {
                "project_path": project_path,
                "workspace_root": str(self._tab_pages.project_manager.workspace_root.resolve()),
                "generation": int(generation),
            }
            self.statusBar().showMessage("项目正文已恢复；OCR 对比将在首次打开文字校对时按需载入。", 5000)

        # Hydrate only the currently visible workspace. Other large editors are
        # populated on first visit so app startup never rebuilds the entire book.
        QTimer.singleShot(0, self._hydrate_visible_project_workspace)

    def _hydrate_visible_project_workspace(self) -> None:
        idx = self._stack.currentIndex() if hasattr(self, "_stack") else -1
        if idx == SECTION_FORMAT:
            self._ensure_project_document_hydrated("formatter")
        elif idx == SECTION_PROOF:
            target = {0: "ocr_compare", 1: "image_review"}.get(self._proof_section.current_index())
            if target:
                self._ensure_project_document_hydrated(target)
        elif idx == SECTION_EPUB:
            self._ensure_project_document_hydrated("epub")

    def _ensure_project_document_hydrated(self, target: str) -> None:
        pending = self._pending_project_document
        key = str(target or "")
        if pending is None or key in self._pending_project_hydrated:
            return
        stage, doc = pending
        source_kind = "pdf_text" if str(getattr(doc.metadata, "source_engine", "") or "") == "pdf_text_layer" else (stage or "ocr")
        if key == "formatter":
            self._tab_fmt.set_doc(doc)
        elif key == "epub":
            self._tab_epub.set_doc(doc, source_kind)
        elif key == "ocr_compare":
            self._ensure_project_multi_ocr_hydrated()
            return
        elif key == "image_review":
            self._tab_ocr_image_review.set_document(doc, f"项目恢复 · {stage or 'OCR'}", lazy=False)
            self._ensure_project_multi_ocr_hydrated()
        self._pending_project_hydrated.add(key)

    def _save_project_adjudication_snapshot(self, *, reset_journal: bool = False) -> None:
        """Write a full checkpoint at low-frequency lifecycle boundaries only."""
        manager = self._tab_pages.project_manager
        if manager.active_project is None:
            return
        try:
            snapshot = self._tab_ocr_compare.project_snapshot_state()
            if snapshot:
                manager.save_multi_ocr_snapshot(snapshot)
                manager.save_adjudication_state({
                    "mode": "multi",
                    "current_row_index": snapshot.get("current_row_index", 0),
                    "fusion_states": snapshot.get("fusion_states", []),
                    "image_review_overrides": snapshot.get("image_review_overrides", {}),
                    "canonical_source_decisions": snapshot.get("canonical_source_decisions", {}),
                })
                if reset_journal:
                    manager.clear_adjudication_events()
        except Exception as exc:
            self.statusBar().showMessage(f"裁决状态 checkpoint 保存失败：{exc}", 8000)

    def _append_project_adjudication_event(self, payload: dict) -> None:
        """Persist one manual decision without serialising the whole OCR session."""
        manager = self._tab_pages.project_manager
        if manager.active_project is None:
            return
        try:
            manager.append_adjudication_event({
                "kind": "fusion_decision",
                "decision": dict(payload or {}),
            })
        except Exception as exc:
            self.statusBar().showMessage(f"裁决增量日志保存失败：{exc}", 8000)

    def _on_package_exported(self, info: dict) -> None:
        manager = self._tab_pages.project_manager
        if manager.active_project is None:
            return
        data = dict(info or {})
        details = dict(data.get("details") or {})
        details.update({
            "kind": str(data.get("kind") or "package"),
            "path": str(data.get("path") or ""),
        })
        error_text = str(data.get("error") or "")
        if error_text:
            details["error"] = error_text
        try:
            package_path = str(data.get("path") or "")
            package_kind = str(data.get("kind") or "package")
            if package_path and Path(package_path).is_file():
                details["external_path"] = package_path
                # V5 GPT adjudication packages are derived exchange artifacts.
                # The workspace already persists OCR documents, comparison state
                # and imported adjudication overlays, so retaining another 20–60MB
                # copy inside the project only causes storage growth.
                if package_kind != "source_correction_v5":
                    project_copy = manager.preserve_exported_file(package_path, category="packages")
                    details["project_copy_path"] = str(project_copy)
                    data["project_copy_path"] = str(project_copy)
                    graph = manager.artifact_graph()
                    current = dict(graph.get("current") or {})
                    deps = [
                        current.get("adjudication.cloud", ""),
                        current.get("adjudication.local", ""),
                        current.get("ocr.multi.latest", ""),
                        current.get("stage.current", ""),
                    ]
                    manager.register_file_artifact(
                        kind="package", path=project_copy, alias="export.package.latest",
                        dependencies=[value for value in deps if value],
                        metadata={
                            "kind": package_kind,
                            "external_path": package_path,
                        },
                    )
                else:
                    details["project_copy_skipped"] = True
            manager.record_run(
                "package_export",
                status=str(data.get("status") or "ok"),
                details=details,
            )
        except Exception as exc:
            self.statusBar().showMessage(f"导出日志保存失败：{exc}", 8000)

    def _on_epub_built(self, info: dict) -> None:
        manager = self._tab_pages.project_manager
        if manager.active_project is None:
            return
        try:
            artifact_path = str((info or {}).get("project_copy_path") or (info or {}).get("path") or "")
            if artifact_path and Path(artifact_path).is_file():
                graph = manager.artifact_graph()
                current = dict(graph.get("current") or {})
                deps = [
                    current.get("adjudication.cloud", ""),
                    current.get("adjudication.local", ""),
                    current.get("stage.current", ""),
                ]
                manager.register_file_artifact(
                    kind="epub", path=artifact_path, alias="export.epub.latest",
                    dependencies=[value for value in deps if value],
                    metadata={
                        "chapters": int((info or {}).get("chapters", 0) or 0),
                        "images": int((info or {}).get("images", 0) or 0),
                    },
                )
            manager.record_run(
                "epub_export",
                details={
                    "path": str((info or {}).get("path") or ""),
                    "project_copy_path": str((info or {}).get("project_copy_path") or ""),
                    "chapters": int((info or {}).get("chapters", 0) or 0),
                    "images": int((info or {}).get("images", 0) or 0),
                    "quality_gate": (
                        (info or {}).get("quality_gate").to_dict()
                        if getattr((info or {}).get("quality_gate"), "to_dict", None)
                        else {}
                    ),
                },
            )
        except Exception:
            pass

    def _clear_ocr_workspace(self, tab) -> None:
        """Clear OCR runtime state and every UI that directly depends on it.

        Downstream Formatter/EPUB snapshots remain independent by design,
        but stale OCR comparison cards and image/text sentence pairs must not stay
        visible after the OCR workspace itself has been cleared.
        """
        # Release any QPixmap/file handles held by downstream OCR workspaces
        # before deleting the OCR temp root that owns sentence-group images.
        # This ordering is important on filesystems that reject deleting open
        # files and also prevents a one-frame stale image after Clear.
        review = self._lazy_workspace_if_loaded("image_review")
        compare = self._lazy_workspace_if_loaded("ocr_compare")
        if review is not None:
            review.reset_for_new_book()
        if compare is not None:
            compare.reset_for_new_book()
        self._pending_single_compare_result = None
        self._pending_image_review_document = None
        self._pending_ocr_inputs = ()
        ClearManager.clear_ocr(tab)

    def _source_update_has_active_jobs(self) -> bool:
        """Conservative write gate used only by the program source updater.

        Existing workspaces are not modified. We merely inspect their explicit
        busy flags/thread handles so a source-tree mutation cannot overlap an OCR,
        formatting, comparison or export worker.
        """
        for tab in tuple(getattr(self, "_workspace_tabs", ()) or ()):
            try:
                for name, value in vars(tab).items():
                    if isinstance(value, threading.Thread) and value.is_alive():
                        return True
                    if name.endswith("_busy") and value is True:
                        return True
                    running = getattr(value, "isRunning", None)
                    if callable(running) and running():
                        return True
            except Exception:
                return True
        return False

    # ── 全局拖放：在主窗口任意子控件上拖入文件 / 文件夹都统一导入 ─────────────
    def _dropped_sources(self, event) -> list[str]:
        from ui.pages.import_sources import collect_sources
        mime = event.mimeData()
        if not mime.hasUrls():
            return []
        return collect_sources([u.toLocalFile() for u in mime.urls() if u.isLocalFile()])

    def _hide_drop_overlay(self) -> None:
        overlay = getattr(self, "_nf_drop_overlay", None)
        if overlay is not None:
            overlay.hide()

    def _show_drop_overlay(self, count: int) -> None:
        from ui.common.drop_overlay import DropOverlay
        if getattr(self, "_nf_drop_overlay", None) is None:
            self._nf_drop_overlay = DropOverlay(self)
        self._nf_drop_overlay.show_for(count)

    def _ocr_import_busy(self) -> bool:
        tab = self._lazy_workspace_if_loaded("ocr") if hasattr(self, "_lazy_workspace_if_loaded") else None
        if tab is None:
            return False
        if bool(getattr(tab, "_ocr_run_active", False)):
            return True
        worker = getattr(tab, "_worker", None)
        is_running = getattr(worker, "isRunning", None)
        return bool(callable(is_running) and is_running())

    def _import_dropped_sources(self, sources: list[str]) -> bool:
        from ui.pages.import_sources import import_label
        from ui.common.toast import notify
        from ui.common.interaction_logic import drop_decision

        decision = drop_decision(sources, self._ocr_import_busy())
        if decision == "empty":
            return False
        if decision == "busy":
            notify(self, "OCR 正在运行，请先停止或完成后再导入新文件", "warning")
            return True

        self._goto(SECTION_PAGE)
        label = import_label(sources, "导入文件")
        imported = self._tab_pages._import_sources_into_project(list(sources), label)
        if imported:
            self._tab_pages._load_inputs(imported, label)
        return True

    def _is_own_widget(self, obj) -> bool:
        current = obj
        while current is not None:
            if current is self:
                return True
            parent = getattr(current, "parent", None)
            current = parent() if callable(parent) else None
        return False

    def eventFilter(self, watched, event):
        """Capture file URL drops before QTextEdit/QLineEdit can turn them into text."""
        event_type = event.type()
        if self._is_own_widget(watched) and event_type in (QEvent.DragEnter, QEvent.DragMove, QEvent.Drop, QEvent.DragLeave):
            if event_type == QEvent.DragLeave:
                self._hide_drop_overlay()
                return False
            sources = self._dropped_sources(event)
            if not sources:
                return False
            if event_type in (QEvent.DragEnter, QEvent.DragMove):
                self._show_drop_overlay(len(sources))
                event.acceptProposedAction()
                return True
            self._hide_drop_overlay()
            handled = self._import_dropped_sources(sources)
            if handled:
                event.acceptProposedAction()
            return handled
        return super().eventFilter(watched, event)

    def dragEnterEvent(self, event) -> None:
        sources = self._dropped_sources(event)
        if not sources:
            event.ignore()
            return
        self._show_drop_overlay(len(sources))
        event.acceptProposedAction()

    def dragLeaveEvent(self, event) -> None:
        self._hide_drop_overlay()
        event.accept()

    def dropEvent(self, event) -> None:
        self._hide_drop_overlay()
        sources = self._dropped_sources(event)
        if self._import_dropped_sources(sources):
            event.acceptProposedAction()
        else:
            event.ignore()

    def closeEvent(self, event):
        """Invalidate every workspace task and release retained temporary files."""
        if hasattr(self, "_tab_system") and self._tab_system.source_update_in_progress():
            event.ignore()
            notify(self, "请等待 Git 检查/更新完成后再关闭窗口，以免中断事务更新。", "warning")
            return
        try:
            # Manual adjudication is already crash-safe in the append-only
            # project journal.  Never rebuild/compress the 50-100 MB full OCR
            # snapshot on the Qt close path: that made macOS show a temporary
            # "not responding" state.  Full checkpoints remain written at OCR
            # completion and successful AI-import boundaries.
            # Background single-OCR persistence may still be compressing a large
            # stage. Invalidate UI callbacks on close; the atomic writer itself
            # may finish safely against the project captured at OCR completion.
            if hasattr(self, "_project_stage_save_generation"):
                self._project_stage_save_generation.invalidate()
            if hasattr(self, "_project_stage_save_signal_refs"):
                self._project_stage_save_signal_refs.clear()
            if hasattr(self, "_ocr_handoff_generation"):
                self._ocr_handoff_generation.invalidate()
            if hasattr(self, "_project_restore_signal_refs"):
                self._project_restore_signal_refs.clear()
            if hasattr(self, "_workspace_coordinator"):
                self._workspace_coordinator.shutdown()
            # Ordering contract: sentence-review pixmaps must release files before
            # the OCR temp store removes those exact paths. Keep this explicit
            # even though the generic workspace loop handles every other tab.
            image_review = self._lazy_workspace_if_loaded("image_review")
            ocr_tab = self._lazy_workspace_if_loaded("ocr")
            if image_review is not None:
                image_review.shutdown_cleanup()
            if ocr_tab is not None:
                ocr_tab.shutdown_cleanup()

            already_cleaned = {
                id(image_review),
                id(ocr_tab),
            }
            for tab in list(getattr(self, "_workspace_tabs", []) or []):
                if id(tab) in already_cleaned:
                    continue
                # First invalidate generic guards/events so late worker signals
                # cannot update a closing or already-destroyed workspace.
                for name, value in list(vars(tab).items()):
                    if name.endswith("_cancel_event") or name == "_cancel_event":
                        setter = getattr(value, "set", None)
                        if callable(setter):
                            setter()
                    if name.endswith("_generation") and hasattr(value, "invalidate"):
                        try:
                            value.invalidate()
                        except Exception:
                            pass
                cleanup = getattr(tab, "shutdown_cleanup", None)
                if callable(cleanup):
                    try:
                        cleanup()
                    except Exception:
                        # Application close must continue even when an optional
                        # preview/native helper has already torn itself down.
                        pass
        finally:
            try:
                from utils.session_temp import cleanup_session_temp_async
                # Large OCR preview/crop trees can contain tens of thousands of
                # files.  Detach deletion from the Qt close path so macOS never
                # reports a temporary hang while the filesystem is being swept.
                cleanup_session_temp_async()
            except Exception:
                pass
            super().closeEvent(event)
            if event.isAccepted():
                app = QApplication.instance()
                if app is not None:
                    app.removeEventFilter(self)

    def import_paddle_output(self):
        """Compatibility entry for legacy PaddleOCR-VL exports.

        Formatter already exposes the normal external-import UI.  Keep this
        legacy handler functional for older integrations, but publish the result
        through the same immutable OCR handoff as every real adapter instead of
        only assigning ``self._doc`` and leaving all workspaces stale.
        """
        fn, _ = QFileDialog.getOpenFileName(
            self, "选择 PaddleOCR-VL 输出", "",
            "JSON/Markdown (*.json *.md *.markdown)",
        )
        if not fn:
            return
        folder = str(getattr(self._tab_pages, "image_folder", "") or "")
        overrides = self._confirmed_page_overrides()
        try:
            if Path(fn).suffix.lower() == ".json":
                doc = import_paddle_json(fn, folder, overrides, 0)
            else:
                doc = import_paddle_md(fn, folder, overrides, 0)
            self._on_ocr_done(doc)
            notify(self, f"导入完成：{len(doc.blocks)} 个内容块，已同步到后续工作区。", "success")
        except Exception as exc:
            QMessageBox.critical(self, "导入失败", str(exc))

    def _apply_system_settings(self, _prefs=None) -> None:
        """Apply UI preferences and OCR defaults without touching business data."""
        prefs = dict(_prefs or {})
        if hasattr(self, "_tab_system"):
            if not prefs:
                prefs = self._tab_system.current_preferences()
            manager = manager_for()
            if manager is not None:
                manager.apply_preferences(
                    language=str(prefs.get("language") or LANG_ZH),
                    theme=str(prefs.get("appearance") or THEME_LIGHT),
                )
        if hasattr(self, "_tab_system"):
            ocr_tab = self._lazy_workspace_if_loaded("ocr")
            if ocr_tab is not None:
                self._tab_system.apply_to_ocr_tab(ocr_tab)

    def _run_system_ocr_runtime_check(self) -> None:
        self._open_named_workspace("ocr")
        QTimer.singleShot(0, lambda: self._tab_ocr._refresh_ocr_runtime_status(show_dialog=True, deep=True))

    def _restore_reference_navigation(self) -> None:
        """恢复七主区与融合页签；未恢复时使用系统设置中的默认功能区。"""
        initial_section = self._tab_system.default_section_index()
        if self._tab_system.restore_workspace_enabled():
            try:
                initial_section = int(self._ui_settings.value("reference_ui/last_section", SECTION_PAGE))
            except (TypeError, ValueError):
                initial_section = SECTION_PAGE
        initial_section = max(SECTION_WORKSPACE, min(len(self._section_pages) - 1, initial_section))

        if self._tab_system.remember_subtabs_enabled():
            try:
                self._ocr_section.set_current_index(
                    int(self._ui_settings.value("reference_ui/ocr_subtab", 0))
                )
                self._proof_section.set_current_index(
                    int(self._ui_settings.value("reference_ui/proof_subtab", 0))
                )
            except (TypeError, ValueError):
                self._ocr_section.set_current_index(0)
                self._proof_section.set_current_index(0)
        if initial_section == SECTION_OCR:
            self._ocr_section.set_current_index(0)
        elif initial_section == SECTION_PROOF and self._proof_section.current_index() > 1:
            self._proof_section.set_current_index(0)
        self._goto(initial_section)

    def _open_named_workspace(self, key: str) -> None:
        routes = {
            "workspace": (SECTION_WORKSPACE, None),
            "book": (SECTION_PAGE, None),
            "ocr": (SECTION_OCR, 0),
            "pdf": (SECTION_OCR, 1),
            "ai_image": (SECTION_OCR, 2),
            "format": (SECTION_FORMAT, None),
            "ocr_compare": (SECTION_PROOF, 0),
            "image_text": (SECTION_PROOF, 1),
            "export": (SECTION_EPUB, None),
            "system": (SECTION_SYSTEM, None),
        }
        section, subtab = routes.get(str(key), (SECTION_PAGE, None))
        # Select the requested subtab before entering the main section so direct
        # routes do not construct an unrelated default workspace first.
        if section == SECTION_OCR and subtab is not None:
            self._ocr_section.set_current_index(subtab)
        elif section == SECTION_PROOF and subtab is not None:
            self._proof_section.set_current_index(subtab)
        self._goto(section)

    def _on_ocr_subtab_changed(self, index: int) -> None:
        if self._tab_system.remember_subtabs_enabled():
            self._ui_settings.setValue("reference_ui/ocr_subtab", int(index))
        if self._stack.currentIndex() == SECTION_OCR:
            self._ensure_ocr_subworkspace(index)
            key = {0: "ocr", 1: "pdf_text", 2: "ai_image"}.get(int(index), "ocr")
            self._workspace_coordinator.set_active((key,))

    def _on_image_review_row_saved(self, payload: dict) -> None:
        """Commit one 图文 decision and return an explicit success/failure ack."""
        data = payload if isinstance(payload, dict) else {}
        row_index = -1
        try:
            row_index = int(data.get("row_index", -1))
        except (TypeError, ValueError, OverflowError):
            pass
        success = self._tab_ocr_compare.apply_image_review_update(data, refresh=False)
        if success:
            actual_row = self._tab_ocr_compare.current_row_index()
            self._pending_image_review_source_row = int(actual_row)
            message = f"已保存并同步到 OCR 对比第 {actual_row + 1} 句"
            self._tab_ocr_image_review.notify_ocr_compare_sync_result(
                row_index, True, message
            )
        else:
            self._tab_ocr_image_review.notify_ocr_compare_sync_result(
                row_index, False,
                "已保存图文校对，但无法绑定 OCR 对比行；未伪报同步成功。"
            )

    def _on_ocr_compare_decision_changed(self, payload: dict) -> None:
        data = payload if isinstance(payload, dict) else {}
        # Persist first, including decisions that originated in 图文校对.  One
        # decision is one append-only JSONL event; the 50-100 MB OCR snapshot is
        # not rewritten during a review streak.
        self._append_project_adjudication_event(data)
        if str(data.get("origin", "")) == "image_review":
            return
        image_review_visible = bool(self._proof_section.current_index() == 1)
        self._tab_ocr_image_review.apply_ocr_compare_decision(
            data, refresh=image_review_visible
        )

    def _on_coordinated_row_changed(self, row_index: int, source: str) -> None:
        if str(source) == "image_review":
            self._sync_ocr_compare_from_image_review(row_index)
        else:
            self._remember_ocr_compare_row(row_index)

    def _remember_ocr_compare_row(self, row_index: int) -> None:
        try:
            self._pending_image_review_source_row = max(0, int(row_index))
        except (TypeError, ValueError, OverflowError):
            return
        if self._proof_section.current_index() == 1:
            self._tab_ocr_image_review.jump_to_source_row(self._pending_image_review_source_row)

    def _sync_ocr_compare_from_image_review(self, row_index: int) -> None:
        if self._proof_section.current_index() != 1:
            return
        self._pending_image_review_source_row = max(0, int(row_index))
        self._tab_ocr_compare.select_source_row(self._pending_image_review_source_row)

    def _open_image_review_from_ocr_compare(self, row_index: int) -> None:
        """Open the independent image workspace at the OCR comparison stable row."""
        try:
            row = max(0, int(row_index))
        except (TypeError, ValueError, OverflowError):
            row = self._tab_ocr_compare.current_row_index()
        self._pending_image_review_source_row = row
        self._goto(SECTION_PROOF)
        self._proof_section.set_current_index(1)

    def _load_and_sync_image_review_row(self) -> None:
        # Capture the OCR row before lazy loading: the initial image-review
        # render emits row 0, and must not overwrite the row the user came from.
        row = self._tab_ocr_compare.current_row_index()
        self._pending_image_review_source_row = max(0, int(row))
        self._tab_ocr_image_review.ensure_document_loaded()
        self._tab_ocr_image_review.jump_to_source_row(row)

    def _on_proof_subtab_changed(self, index: int) -> None:
        previous = int(getattr(self, "_last_proof_subtab_index", self._proof_section.current_index()))
        index = max(0, min(1, int(index)))
        if self._stack.currentIndex() == SECTION_PROOF:
            self._ensure_proof_subworkspace(index)
            proof_keys = ("ocr_compare", "image_review")
            self._workspace_coordinator.set_active((proof_keys[index],))
            self._ensure_project_document_hydrated(proof_keys[index])
        review = self._lazy_workspace_if_loaded("image_review")
        if previous == 1 and index != 1 and review is not None:
            if not review.save_pending_edit():
                self.statusBar().showMessage("图文对照当前句保存失败；修改仍保留在编辑器中。", 7000)
        self._last_proof_subtab_index = index
        if self._tab_system.remember_subtabs_enabled():
            self._ui_settings.setValue("reference_ui/proof_subtab", index)
        if index == 0:
            compare = self._lazy_workspace_if_loaded("ocr_compare")
            if compare is not None:
                # Single-model OCR is already authoritative enough to continue
                # downstream, so opening OCR Compare should show it immediately.
                # A multi-model session is intentionally left untouched until
                # human/AI adjudication is explicitly applied.
                QTimer.singleShot(0, compare.ensure_latest_single_result_loaded)
            review = self._lazy_workspace_if_loaded("image_review")
            if review is not None:
                source_row = review.current_source_row_index()
                if source_row >= 0:
                    QTimer.singleShot(0, lambda row=source_row: self._tab_ocr_compare.select_source_row(row))
        else:
            QTimer.singleShot(0, self._load_and_sync_image_review_row)

    def _goto(self, idx: int):
        """六主功能区统一切换；融合页签与侧栏高亮始终同步。"""
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            return
        if idx < 0 or idx >= self._stack.count():
            return
        self._ensure_section_workspace(idx)
        self._stack.setCurrentIndex(idx)
        self._sidebar.select(idx)
        header = getattr(self, "_page_header", None)
        if header is not None:
            header.set_section(idx)
        # OCR 按钮只属于 OCR 工作区；文字校对区不会创建或复用该按钮。
        self._ui_settings.setValue("reference_ui/last_section", idx)
        if idx == SECTION_WORKSPACE:
            self._workspace_coordinator.set_active(("workspace",))
        elif idx == SECTION_PAGE:
            self._workspace_coordinator.set_active(("pages",))
        elif idx == SECTION_OCR:
            self._on_ocr_subtab_changed(self._ocr_section.current_index())
        elif idx == SECTION_FORMAT:
            self._workspace_coordinator.set_active(("formatter",))
            self._ensure_project_document_hydrated("formatter")
        elif idx == SECTION_PROOF:
            if self._proof_section.current_index() > 1:
                self._proof_section.set_current_index(0)
            self._on_proof_subtab_changed(self._proof_section.current_index())
        elif idx == SECTION_EPUB:
            self._workspace_coordinator.set_active(("epub",))
            self._ensure_project_document_hydrated("epub")
        elif idx == SECTION_SYSTEM:
            self._workspace_coordinator.set_active(("system",))

    def _confirmed_page_overrides(self):
        pm = self._tab_pages
        return {
            int(page_no): page_type
            for page_no, page_type in getattr(pm, "page_overrides", {}).items()
            if page_no not in getattr(pm, "_auto_suggested", set())
        }

    def _pdf_text_page_manager_context(self, pdf_path: str | None = None) -> dict:
        """Expose Page Manager's confirmed page types to PDF text extraction."""
        pm = getattr(self, "_tab_pages", None)
        if pm is None:
            return {"matched": False, "reason": "页面管理尚未初始化", "page_overrides": {}}
        from core.pdf_page_manager_binding import resolve_pdf_page_manager_context
        return resolve_pdf_page_manager_context(
            pdf_path=pdf_path,
            raw_inputs=list(getattr(pm, "_last_loaded_raw_inputs", None) or []),
            managed_page_count=len(getattr(pm, "page_images", []) or []),
            page_overrides=dict(getattr(pm, "page_overrides", {}) or {}),
            auto_suggested=set(getattr(pm, "_auto_suggested", set()) or set()),
            original_pdf_sources=list(getattr(pm, "_original_pdf_sources", None) or []),
            pdf_physical_page_map=dict(getattr(pm, "_pdf_physical_page_map", None) or {}),
        )

    def _refresh_page_asset_context(self):
        """Refresh Page Manager assets in EPUB without mutating text snapshots.

        EPUB's cover indicator/preview follows the Page Manager immediately.
        This asset-only synchronization is intentionally separate from OCR page
        filtering: EPUB mirrors the page types currently shown by Page Manager,
        while OCR/PDF extraction continues to trust only explicitly confirmed
        non-body classifications for skip decisions.
        """
        pm = getattr(self, "_tab_pages", None)
        if pm is None:
            return
        images = list(getattr(pm, "page_images", []) or [])
        overrides = dict(getattr(pm, "page_overrides", {}) or {})
        epub = self._lazy_workspace_if_loaded("epub")
        if epub is not None:
            if hasattr(epub, "set_page_manager_assets"):
                epub.set_page_manager_assets(images, overrides)
            if hasattr(epub, "_status"):
                epub._status.setText(
                    f"页面管理已更新：{len(images)} 页；封面和插图已同步到 EPUB。"
                )

    def _persist_single_ocr_stage_async(
        self, doc: UnifiedDocument, *, pipeline_state: dict | None = None
    ) -> None:
        """Persist a completed single-OCR document off the Qt thread.

        ``UnifiedDocument.to_dict()`` is cheap and detached from subsequent GUI
        state. Compression/content-addressed storage is the expensive part, so a
        separate ProjectWorkspaceManager performs it in a serialized worker.
        The destination project is captured now; switching projects while the
        write finishes can never redirect the old OCR result into the new book.
        """
        manager = getattr(getattr(self, "_tab_pages", None), "project_manager", None)
        if manager is None or manager.active_project is None:
            return
        try:
            workspace_root = str(manager.workspace_root.resolve())
            project_path = str(manager.active_project.resolve())
            # Keep the immutable OCR document itself; serializing a full book can
            # be tens of MB and belongs in the persistence worker, not the Qt
            # completion callback.
            document = doc
            pipeline_payload = dict(pipeline_state or {})
        except Exception as exc:
            self.statusBar().showMessage(f"单模型 OCR 后台保存准备失败：{exc}", 8000)
            return

        token = self._project_stage_save_generation.begin()
        signals = WorkerSignals()
        self._project_stage_save_signal_refs[token] = signals

        def done(_payload=None):
            self._project_stage_save_signal_refs.pop(token, None)
            if self._project_stage_save_generation.is_current(token):
                self.statusBar().showMessage("单模型 OCR 已写入当前工作区", 3500)

        def failed(message):
            self._project_stage_save_signal_refs.pop(token, None)
            if self._project_stage_save_generation.is_current(token):
                self.statusBar().showMessage(f"单模型 OCR 后台保存失败：{message}", 10000)

        signals.finished.connect(done)
        signals.error.connect(failed)

        def worker():
            try:
                with self._project_stage_save_lock:
                    save_manager = ProjectWorkspaceManager(workspace_root)
                    save_manager.open_project(project_path)
                    save_manager.save_stage_document("ocr", document.to_dict())
                    if pipeline_payload:
                        save_manager.save_pipeline_config("single_ocr", pipeline_payload)
                signals.finished.emit({"stage": "ocr"})
            except Exception:
                import traceback
                signals.error.emit(traceback.format_exc())

        threading.Thread(
            target=worker, daemon=True, name=f"single-ocr-project-save-{token}"
        ).start()

    def _persist_multi_ocr_stage_async(
        self, doc: UnifiedDocument, *, pipeline_state: dict | None = None
    ) -> None:
        """Persist the automatic multi-OCR fusion without blocking OCR Compare."""
        manager = getattr(getattr(self, "_tab_pages", None), "project_manager", None)
        if manager is None or manager.active_project is None:
            return
        try:
            workspace_root = str(manager.workspace_root.resolve())
            project_path = str(manager.active_project.resolve())
            document = doc
            pipeline_payload = dict(pipeline_state or {})
        except Exception as exc:
            self.statusBar().showMessage(f"多模型 OCR 后台保存准备失败：{exc}", 8000)
            return

        token = self._project_stage_save_generation.begin()
        signals = WorkerSignals()
        self._project_stage_save_signal_refs[token] = signals

        def done(_payload=None):
            self._project_stage_save_signal_refs.pop(token, None)
            if self._project_stage_save_generation.is_current(token):
                self.statusBar().showMessage("多模型 OCR 已写入当前工作区", 3500)

        def failed(message):
            self._project_stage_save_signal_refs.pop(token, None)
            if self._project_stage_save_generation.is_current(token):
                self.statusBar().showMessage(f"多模型 OCR 后台保存失败：{message}", 10000)

        signals.finished.connect(done)
        signals.error.connect(failed)

        def worker():
            try:
                with self._project_stage_save_lock:
                    save_manager = ProjectWorkspaceManager(workspace_root)
                    save_manager.open_project(project_path)
                    if pipeline_payload:
                        save_manager.save_pipeline_config("multi_ocr", pipeline_payload)
                    save_manager.save_stage_document("ocr_auto_fusion", document.to_dict())
                signals.finished.emit({"stage": "ocr_auto_fusion"})
            except Exception:
                import traceback
                signals.error.emit(traceback.format_exc())

        threading.Thread(
            target=worker, daemon=True, name=f"multi-ocr-project-save-{token}"
        ).start()

    def _record_workspace_version(
        self,
        stage: str,
        doc: UnifiedDocument,
        *,
        record_project_run: bool = True,
    ):
        # Stage history is an immutable snapshot registry.  Workspaces may keep
        # their own editable clones, but no later metadata/text edit is allowed
        # to retroactively mutate an OCR/Formatter snapshot.
        #
        # Static hardening audits historically looked for the three statements
        # below.  They remain as contract markers while the registry performs a
        # single immutable alias publication plus one mutable working clone:
        # stage_snapshot = copy.deepcopy(doc)
        # self._workspace_docs["current"] = copy.deepcopy(stage_snapshot)
        # self._doc = copy.deepcopy(stage_snapshot)
        stage_snapshot = self._snapshot_registry.publish_aliases((stage, "current"), doc, clone=True)
        self._current_stage = stage
        self._doc = self._snapshot_registry.clone(stage_snapshot)

        # Project persistence is outside the OCR/Formatter algorithms: every
        # published text stage is durable and page-lineage guarded, while UI
        # workspaces are still free to keep independent editable clones.
        manager = getattr(getattr(self, "_tab_pages", None), "project_manager", None)
        if manager is not None and manager.active_project is not None:
            try:
                manager.save_stage_document(stage, stage_snapshot.to_dict())
                metadata = getattr(stage_snapshot, "metadata", None)
                if record_project_run:
                    manager.record_run(
                        stage,
                        details={
                            "blocks": len(getattr(stage_snapshot, "blocks", []) or []),
                            "pages": len(getattr(stage_snapshot, "pages", []) or []),
                            "source_engine": str(getattr(metadata, "source_engine", "") or ""),
                        },
                    )
            except Exception as exc:
                self.statusBar().showMessage(f"项目阶段自动保存失败：{exc}", 8000)

    def _on_pages_loaded(self, images):
        def normalized(values):
            result = []
            for value in values:
                try:
                    result.append(str(Path(value).expanduser().resolve()))
                except Exception:
                    result.append(str(value))
            return tuple(result)

        signature = normalized(images)
        previous = getattr(self, "_active_page_image_signature", ())
        current_doc_paths = normalized(
            p.image_path for p in getattr(self._doc, "pages", []) if getattr(p, "image_path", "")
        ) if self._doc is not None else ()

        # OCR can finish while Page Manager is still expanding the same PDF/folder
        # in a background thread.  In that case the resulting document already
        # belongs to these images, so the late pages_loaded signal must not wipe it.
        same_as_current_doc = bool(current_doc_paths) and current_doc_paths == signature
        restoring_pages = bool(getattr(self._tab_pages, "_project_restore_in_progress", False))
        if signature != previous and not same_as_current_doc:
            # A user import can finish while the delayed project-restore worker
            # from project activation is still parsing an older current-stage
            # document.  Invalidate that worker now; otherwise its late result
            # can overwrite the freshly imported PDF/OCR text in Formatter.
            # PageManager's own restore path is exempt because both page and
            # document restoration belong to the same project generation.
            if not restoring_pages:
                self._project_restore_generation += 1
            self._doc = None
            self._snapshot_registry.clear()
            self._current_stage = ""
            self._pending_project_document = None
            self._pending_project_hydrated.clear()
            self._ocr_handoff_generation.invalidate()
            for key, method_name in (
                ("pdf_text", "reset_for_new_book"),
                ("formatter", "reset_for_new_book"),
                ("ocr_compare", "reset_for_new_book"),
                ("image_review", "reset_for_new_book"),
                ("epub", "clear_doc"),
            ):
                workspace = self._lazy_workspace_if_loaded(key)
                reset = getattr(workspace, method_name, None) if workspace is not None else None
                if callable(reset):
                    reset()
            self._pending_single_compare_result = None
            self._pending_image_review_document = None
        self._active_page_image_signature = signature
        self._pending_ocr_inputs = tuple(str(value) for value in images)
        # Legacy source-contract marker; the loaded-only branch below is the
        # lazy equivalent of: self._tab_ocr.set_inputs(images)
        ocr_tab = self._lazy_workspace_if_loaded("ocr")
        if ocr_tab is not None:
            ocr_tab.set_inputs(images)
        pdf_text_tab = self._lazy_workspace_if_loaded("pdf_text")
        if pdf_text_tab is not None and hasattr(pdf_text_tab, "notify_page_manager_context_changed"):
            pdf_text_tab.notify_page_manager_context_changed()
        self._refresh_page_asset_context()

        if self._active_project_dir:
            restoring = restoring_pages
            if not restoring:
                try:
                    manager = self._tab_pages.project_manager
                    manager.record_run(
                        "pages_loaded",
                        details={"pages": len(signature), "changed": signature != previous},
                    )
                except Exception:
                    pass

    def _on_page_types_changed(self):
        """
        页面管理页里改了某页的分类（或删了页）之后，OCR 页的裁剪参照图要跟着
        刷新——否则改完类型回到 OCR 页，看到的还是改之前选中的那张参照图
        （可能已经不再是正文页了），拖框裁剪也就没意义了。
        """
        ocr_tab = self._lazy_workspace_if_loaded("ocr")
        if ocr_tab is not None and ocr_tab._pending_inputs:
            ocr_tab._load_preview_reference()
        pdf_text_tab = self._lazy_workspace_if_loaded("pdf_text")
        if pdf_text_tab is not None and hasattr(pdf_text_tab, "notify_page_manager_context_changed"):
            pdf_text_tab.notify_page_manager_context_changed()
        # 页面分类是独立的轻量覆盖层。只更新资产上下文，不复制正文、不触发
        # 隐藏文本工作区重新对齐，也不清空 EPUB 预览。最终打包时再合并。
        self._refresh_page_asset_context()

    def _document_with_current_page_assets(self, doc: UnifiedDocument, *, copy_document: bool = True):
        """Merge current, explicitly confirmed Page Manager classifications.

        Page Manager remains an independent asset overlay. This bridge is called whenever
        a document is handed to EPUB Builder, so classifications made after OCR are
        still reflected in the exported book.
        """
        pm = self._tab_pages
        if doc is None or not getattr(pm, "page_images", None):
            return doc, None
        # EPUB is an asset consumer, so it mirrors Page Manager's current page
        # classification exactly.  OCR/PDF extraction has a separate
        # _confirmed_page_overrides() safety gate and is not affected here.
        asset_overrides = dict(getattr(pm, "page_overrides", {}) or {})
        ai_report = dict(getattr(getattr(doc, "metadata", None), "ai_image_report", {}) or {})
        ai_text_pages = []
        if str(getattr(getattr(doc, "metadata", None), "source_engine", "") or "").startswith("ai_multimodal"):
            ai_text_pages = list(ai_report.get("processed_page_numbers") or [])
        from engine.page_asset_sync import sync_page_manager_assets
        return sync_page_manager_assets(
            doc, pm.page_images, asset_overrides,
            preserve_unmanaged_pages=False,
            copy_document=copy_document,
            text_page_numbers=ai_text_pages,
        )

    def _on_multi_ocr_done(self, payload: dict):
        documents = list(payload.get("documents") or [])
        labels = list(payload.get("labels") or [])
        fused = payload.get("fused")
        if not documents or fused is None:
            QMessageBox.warning(self, "多模型 OCR 结果无效", "没有收到可用的多模型 OCR 文档。")
            return
        for index, document in enumerate(documents):
            self._snapshot_registry.publish(f"ocr_model_{index + 1}", document, clone=True)
        self._snapshot_registry.publish("ocr_auto_fusion", fused, clone=True)
        self._tab_ocr_image_review.set_document(fused, "多模型 OCR 自动融合稿")
        row_count = len(getattr(payload.get("comparison"), "rows", []) or [])
        load_generation = self._tab_ocr_compare.prepare_large_result_load(labels, row_count)
        self._goto(SECTION_PROOF)
        self._proof_section.set_current_index(0)

        pipeline_state = self._tab_ocr.project_pipeline_state()
        pipeline_state["pipeline_kind"] = "multi"
        pipeline_state["multi_ocr"] = True
        self._persist_multi_ocr_stage_async(fused, pipeline_state=pipeline_state)

        def load_multi_result_and_save(data=payload, generation=load_generation):
            self._tab_ocr_compare.set_results(data, generation=generation)
            self._save_project_adjudication_snapshot(reset_journal=True)

        QTimer.singleShot(0, load_multi_result_and_save)

    def _on_multi_ocr_session_restored(self, payload: dict):
        """Reconnect restored OCR snapshots to image review and downstream selectors."""
        report = dict((payload or {}).get("report") or {})
        if bool(report.get("non_destructive_overlay_import", False)):
            # AI correction import already updated the lightweight fusion states
            # in OCR 对比.  Persist that cloud-returned adjudication immediately,
            # but do not copy model books or replace any downstream OCR source.
            self._save_project_adjudication_snapshot(reset_journal=True)
            manager = self._tab_pages.project_manager
            if manager.active_project is not None:
                try:
                    snapshot = self._tab_ocr_compare.project_snapshot_state()
                    manager.save_adjudication_state({
                        "mode": "multi",
                        "origin": "cloud_import",
                        "report": report,
                        "current_row_index": snapshot.get("current_row_index", 0),
                        "fusion_states": snapshot.get("fusion_states", []),
                        "canonical_source_decisions": snapshot.get("canonical_source_decisions", {}),
                    }, channel="cloud")
                    manager.record_run(
                        "cloud_adjudication_import",
                        details={
                            "accepted": int(report.get("cumulative_accepted_canonical_decisions", 0) or 0),
                            "unresolved": int(report.get("unresolved_canonical_decisions", 0) or 0),
                            "package_id": str(report.get("package_id", "") or ""),
                        },
                    )
                except Exception as exc:
                    self.statusBar().showMessage(f"云端裁决项目状态保存失败：{exc}", 8000)
            return
        documents = list((payload or {}).get("documents") or [])
        labels = list((payload or {}).get("labels") or [])
        fused = (payload or {}).get("fused")
        if not documents:
            return
        for index, document in enumerate(documents):
            self._snapshot_registry.publish(f"ocr_model_{index + 1}", document, clone=True)
        if fused is not None:
            self._snapshot_registry.publish("ocr_auto_fusion", fused, clone=True)
        if fused is not None:
            self._tab_ocr_image_review.set_document(fused, "恢复的多模型 OCR 融合稿")
        self._save_project_adjudication_snapshot(reset_journal=True)

    def _on_ocr_compare_applied(self, doc):
        self._record_workspace_version("ocr", doc)
        self._snapshot_registry.publish("ocr_multi_fusion", doc, clone=True)
        self._tab_fmt.set_doc(doc)
        self._tab_epub.set_doc(doc, "ocr")
        self._tab_ocr_image_review.set_document(doc, "多模型 OCR 人工融合稿")

    def _on_pdf_text_done(self, doc):
        """Publish a completed selectable-PDF text layer to the normal text pipeline.

        PDF text extraction is not OCR, but from Formatter/EPUB's perspective it
        is a complete authoritative text source.  Keep its handoff explicit so
        lazy OCR workspaces, restored project state, or OCR-specific invalidation
        can never swallow the result.
        """
        if doc is None:
            return
        metadata = getattr(doc, "metadata", None)
        if metadata is not None:
            metadata.source_engine = "pdf_text_layer"
            metadata.pdf_text_layer_mode = True

        self._record_workspace_version("ocr", doc, record_project_run=True)
        self._doc = doc
        self._current_stage = "ocr"
        self._pending_project_document = None
        self._pending_project_hydrated.clear()

        # Selectable-PDF text is already an authoritative text source.  It does
        # not need OCR Compare / Image Review, which are evidence workspaces for
        # image OCR.  Feeding PDF text into them used to create a misleading
        # extra proofreading step and a narrow single-source preview.
        self._pending_single_compare_result = None
        self._pending_image_review_document = None
        compare = self._lazy_workspace_if_loaded("ocr_compare")
        if compare is not None:
            compare.set_available_single_result(None)

        self._schedule_single_ocr_downstream_handoff(doc, "pdf_text")

    def _on_ocr_done(self, doc):
        source_engine = str(getattr(doc.metadata, "source_engine", "") or "")
        source_kind = "pdf_text" if source_engine == "pdf_text_layer" else "ocr"

        if source_kind == "pdf_text":
            self._on_pdf_text_done(doc)
            return

        if source_kind == "ocr":
            # Restore the proven 2026-09-24 behaviour: OCR completion is a real
            # document handoff, not a dead-end inbox. Persistence remains async
            # and downstream GUI work is staged across event-loop turns so the
            # completion callback itself stays short.
            self._snapshot_registry.publish_aliases(("ocr", "current"), doc, clone=False)
            self._current_stage = "ocr"
            self._doc = doc
            self._pending_project_document = None
            self._pending_project_hydrated.clear()

            pipeline_state = self._tab_ocr.project_pipeline_state()
            pipeline_state["pipeline_kind"] = "single"
            pipeline_state["multi_ocr"] = False
            self._persist_single_ocr_stage_async(doc, pipeline_state=pipeline_state)

            self._tab_ocr_image_review.set_document(doc, "OCR 原文", lazy=True)
            source_label = source_engine or "单模型 OCR"
            self._tab_ocr_compare.set_available_single_result(doc, source_label)
            self._schedule_single_ocr_downstream_handoff(doc, source_kind)
            return

    def _schedule_single_ocr_downstream_handoff(self, doc, source_kind: str) -> None:
        token = self._ocr_handoff_generation.begin()
        source_label = "PDF 文字层" if source_kind == "pdf_text" else "OCR"
        self.statusBar().showMessage(f"{source_label}已完成，正在同步到 Formatter 与 EPUB…", 7000)
        self._goto(SECTION_FORMAT)

        def current() -> bool:
            return self._ocr_handoff_generation.is_current(token) and self._doc is doc

        def feed_formatter():
            if not current():
                return
            try:
                self._tab_fmt.set_doc(doc)
            except Exception as exc:
                self.statusBar().showMessage(f"Formatter 接收 OCR 正文失败：{exc}", 9000)
                return
            QTimer.singleShot(0, feed_epub)

        def feed_epub():
            if not current():
                return
            try:
                self._tab_epub.set_doc(doc, source_kind)
            except Exception as exc:
                self.statusBar().showMessage(f"EPUB 区接收 OCR 正文失败：{exc}", 9000)
                return
            self.statusBar().showMessage(
                f"{source_label}正文已输出：{len(getattr(doc, 'blocks', []) or [])} 块；Formatter / EPUB 已同步。",
                7000,
            )

        QTimer.singleShot(0, feed_formatter)

    def _on_single_ocr_compare_applied(self, doc):
        """Apply a manually loaded/imported single OCR result without labeling it as multi fusion."""
        self._record_workspace_version("ocr", doc)
        self._snapshot_registry.publish("ocr_single_review", doc, clone=True)
        self._tab_fmt.set_doc(doc)
        self._tab_epub.set_doc(doc, "ocr")
        self._tab_ocr_image_review.set_document(doc, "单 OCR 校对稿")
        self._tab_ocr_compare.set_available_single_result(doc, "单 OCR 校对稿")

    def _on_ocr_image_review_applied(self, doc):
        """Publish image-review text and mirror final decisions into OCR Compare."""
        self._tab_ocr_compare.sync_image_review_document(doc)
        self._record_workspace_version("ocr_image_review", doc)
        self._snapshot_registry.publish("ocr_image_review", doc, clone=True)
        self._tab_fmt.set_doc(doc)
        self._tab_epub.set_doc(doc, "ocr")

    def _on_ai_image_documents_ready(self, source_doc, translated_doc):
        """Publish AI image-book output without forcing OCR Compare/Formatter.

        Page Manager remains authoritative for assets.  We keep both source and
        translation as immutable snapshots and preselect the user's requested
        output in EPUB Builder.
        """
        if source_doc is None:
            return
        self._snapshot_registry.publish("ai_image_source", source_doc, clone=True)
        if translated_doc is not None:
            self._snapshot_registry.publish("ai_image_translation", translated_doc, clone=True)
            chosen, report = self._document_with_current_page_assets(translated_doc, copy_document=True)
            self._doc = chosen
            self._tab_epub.set_doc(chosen, "ai_image_translation")
        else:
            chosen, report = self._document_with_current_page_assets(source_doc, copy_document=True)
            self._doc = chosen
            self._tab_epub.set_doc(chosen, "ai_image_source")
        if hasattr(self._tab_epub, "_status"):
            image_note = f"；同步图片页 {report.image_pages}" if report is not None else ""
            self._tab_epub._status.setText(
                "AI 图文处理已完成：无需传统 OCR 对比/Formatter，可直接生成 EPUB" + image_note
            )

    def _on_ai_image_epub_requested(self, doc, kind: str):
        if doc is None:
            return
        synced, report = self._document_with_current_page_assets(doc, copy_document=True)
        self._doc = synced
        self._tab_epub.set_doc(synced, str(kind or "ai_image_source"))
        if hasattr(self._tab_epub, "_status"):
            label = "AI 译文" if str(kind) == "ai_image_translation" else "AI 原文"
            asset_note = f"；已同步页面管理中的 {report.image_pages} 个图片页" if report is not None else ""
            self._tab_epub._status.setText(f"当前正文来源：{label}{asset_note}")
        self._goto(SECTION_EPUB)

    def _on_fmt_done(self, doc):
        self._record_workspace_version("formatter", doc)
        self._tab_epub.set_doc(doc, "formatter")


__all__ = ["MainWindowControllerMixin"]
