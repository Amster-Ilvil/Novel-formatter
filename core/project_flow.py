# -*- coding: utf-8 -*-
"""Lightweight durable project-flow summaries.

The desktop workspace should be able to answer two questions without loading a
40-100 MB OCR/adjudication snapshot:

* what has this book completed already?
* where should the user continue?

The summary is intentionally derived from tiny project metadata, stage refs and
run-history indexes.  It never resolves content-addressed document payloads and
contains no Qt imports.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any, Mapping

from core.project_workspace import PAGE_STATE_FILE, PROJECT_FILE, SCHEMA_VERSION
from core.runtime_task_state import RuntimeTaskSnapshot, restore_project_runtime_task


@dataclass(frozen=True, slots=True)
class FlowStage:
    key: str
    state: str
    detail: str = ""
    workspace: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProjectFlowSnapshot:
    project_id: str = ""
    project_name: str = ""
    project_path: str = ""
    updated_at: str = ""
    active_stage: str = ""
    resume_workspace: str = "book"
    resume_reason: str = ""
    page_count: int = 0
    unresolved_count: int | None = None
    last_run_stage: str = ""
    last_run_status: str = ""
    last_run_finished_at: str = ""
    runtime_task_status: str = ""
    resumable_ocr: bool = False
    resume_completed_steps: int = 0
    stages: tuple[FlowStage, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["stages"] = [stage.to_dict() for stage in self.stages]
        return payload


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _page_summary(project: Path) -> tuple[int, str]:
    raw = _read_json(project / PAGE_STATE_FILE)
    if not raw:
        return 0, ""
    images = raw.get("page_images") or []
    count = len(images) if isinstance(images, list) else 0
    return count, str(raw.get("page_lineage_hash") or "")


def _multi_summary(project: Path) -> tuple[bool, int | None, int | None, str]:
    raw = _read_json(project / "artifacts" / "ocr" / "multi" / "latest.json")
    if not raw:
        return False, None, None, ""
    if int(raw.get("schema_version") or 0) != SCHEMA_VERSION:
        return False, None, None, ""
    summary = raw.get("summary") if isinstance(raw.get("summary"), Mapping) else {}
    unresolved = summary.get("unresolved")
    total = summary.get("total_rows")
    try:
        unresolved_i = max(0, int(unresolved)) if unresolved is not None else None
    except (TypeError, ValueError, OverflowError):
        unresolved_i = None
    try:
        total_i = max(0, int(total)) if total is not None else None
    except (TypeError, ValueError, OverflowError):
        total_i = None
    return True, unresolved_i, total_i, str(raw.get("page_lineage_hash") or "")


def _has_epub(project: Path) -> bool:
    target = project / "exports" / "epub"
    try:
        return any(path.is_file() and path.suffix.lower() == ".epub" for path in target.iterdir())
    except OSError:
        return False


def build_project_flow_snapshot(manager, *, runtime_task: RuntimeTaskSnapshot | None = None) -> ProjectFlowSnapshot:
    """Return a cheap continuation summary for ``manager.active_project``.

    The function deliberately avoids ``load_current_stage_document()`` and
    ``load_multi_ocr_snapshot()`` because those resolve potentially very large
    compressed objects.  It uses only logical reference wrappers and metadata.
    """
    project = getattr(manager, "active_project", None)
    if project is None:
        return ProjectFlowSnapshot(
            resume_workspace="book",
            resume_reason="尚未选择项目",
            stages=(
                FlowStage("pages", "pending", "先导入图片或 PDF", "book"),
                FlowStage("ocr", "pending", "等待页面", "ocr"),
                FlowStage("proof", "pending", "等待 OCR", "ocr_compare"),
                FlowStage("format", "pending", "等待正文", "format"),
                FlowStage("export", "pending", "等待正文", "export"),
            ),
        )

    project = Path(project).expanduser().resolve()
    project_payload = _read_json(project / PROJECT_FILE)
    page_count, page_lineage = _page_summary(project)
    active_stage = str(project_payload.get("active_stage") or "")
    document_lineage = str(project_payload.get("document_page_lineage_hash") or "")
    stale_document = bool(page_lineage and document_lineage and page_lineage != document_lineage)
    has_multi, unresolved_count, multi_total, multi_lineage = _multi_summary(project)
    stale_multi = bool(page_lineage and multi_lineage and page_lineage != multi_lineage)
    has_epub = _has_epub(project)
    last_run = project_payload.get("last_run") if isinstance(project_payload.get("last_run"), Mapping) else {}
    if runtime_task is None:
        runtime_task = restore_project_runtime_task(
            project, project_payload=project_payload, current_page_lineage_hash=page_lineage,
        )
    elif runtime_task.project_path and str(Path(runtime_task.project_path).expanduser().resolve()) != str(project):
        runtime_task = None
    elif runtime_task.status not in {"running", "cancelling"}:
        # Terminal in-memory state is only a hint. Revalidate the durable run +
        # exact checkpoint against current page lineage before offering resume.
        runtime_task = restore_project_runtime_task(
            project, project_payload=project_payload, current_page_lineage_hash=page_lineage,
        ) or runtime_task
        if runtime_task.resume_supported and runtime_task.checkpoint_signature:
            durable = restore_project_runtime_task(
                project, project_payload=project_payload, current_page_lineage_hash=page_lineage,
            )
            if durable is None or not durable.resume_supported:
                runtime_task = durable
    resumable_ocr = bool(
        runtime_task is not None
        and runtime_task.resume_supported
        and runtime_task.kind in {"single_ocr", "multi_ocr", "ocr"}
    )

    stages: list[FlowStage] = []
    if page_count:
        stages.append(FlowStage("pages", "done", f"{page_count} 页已持久化", "book"))
    else:
        stages.append(FlowStage("pages", "pending", "尚未导入页面", "book"))

    if runtime_task is not None and runtime_task.status in {"running", "cancelling"}:
        detail = "OCR 正在运行" if runtime_task.status == "running" else "OCR 正在停止并保存断点"
        stages.append(FlowStage("ocr", "active", detail, "ocr"))
    elif resumable_ocr:
        count = int(runtime_task.completed_steps or 0) if runtime_task is not None else 0
        detail = f"上次 OCR 可继续；已记录 {count} 个完成阶段"
        stages.append(FlowStage("ocr", "active", detail, "ocr"))
    elif stale_document:
        stages.append(FlowStage("ocr", "stale", "页面已变化，现有正文需重新确认", "ocr"))
    elif active_stage:
        label = {
            "ocr": "OCR 正文已保存",
            "ocr_auto_fusion": "多模型自动融合已保存",
            "ocr_image_review": "图文校对稿已保存",
            "formatter": "Formatter 正文已保存",
        }.get(active_stage, f"正文阶段：{active_stage}")
        stages.append(FlowStage("ocr", "done", label, "ocr"))
    else:
        stages.append(FlowStage("ocr", "pending", "尚未生成 OCR 正文", "ocr"))

    if stale_document or stale_multi:
        proof_state, proof_detail = "stale", "上游页面已变化，旧裁决快照仅保留为历史证据"
    elif has_multi:
        if unresolved_count is None:
            proof_state, proof_detail = "active", "存在多模型会话，可继续裁决"
        elif unresolved_count > 0:
            suffix = f" / {multi_total}" if multi_total is not None else ""
            proof_state, proof_detail = "active", f"待裁决 {unresolved_count}{suffix} 句"
        else:
            proof_state, proof_detail = "done", "多模型裁决已无待判断句"
    elif active_stage:
        proof_state, proof_detail = "optional", "单模型流程：校对可选"
    else:
        proof_state, proof_detail = "pending", "等待 OCR"
    stages.append(FlowStage("proof", proof_state, proof_detail, "ocr_compare"))

    if stale_document:
        format_state, format_detail = "stale", "上游页面已变化，需先重新确认 OCR 正文"
    elif active_stage == "formatter":
        format_state, format_detail = "done", "正文整理已保存"
    elif active_stage:
        format_state, format_detail = "active", "可继续正文整理"
    else:
        format_state, format_detail = "pending", "等待正文"
    stages.append(FlowStage("format", format_state, format_detail, "format"))

    if stale_document and has_epub:
        export_state, export_detail = "stale", "已有 EPUB，但上游页面已变化"
    elif stale_document:
        export_state, export_detail = "stale", "上游页面已变化，暂不建议导出"
    elif has_epub:
        export_state, export_detail = "done", "已有 EPUB 导出"
    elif active_stage:
        export_state, export_detail = "active", "可生成 EPUB"
    else:
        export_state, export_detail = "pending", "等待正文"
    stages.append(FlowStage("export", export_state, export_detail, "export"))

    if page_count <= 0:
        resume_workspace, resume_reason = "book", "继续导入和整理页面"
    elif runtime_task is not None and runtime_task.status in {"running", "cancelling"}:
        resume_workspace, resume_reason = "ocr", "查看当前 OCR 任务"
    elif resumable_ocr:
        resume_workspace, resume_reason = "ocr", "继续上次 OCR（复用断点）"
    elif stale_document or not active_stage:
        resume_workspace, resume_reason = "ocr", "继续 OCR"
    elif has_multi and (unresolved_count is None or unresolved_count > 0):
        resume_workspace, resume_reason = "ocr_compare", "继续多模型 OCR 裁决"
    elif active_stage != "formatter":
        resume_workspace, resume_reason = "format", "继续正文整理"
    elif not has_epub:
        resume_workspace, resume_reason = "export", "继续生成 EPUB"
    else:
        resume_workspace, resume_reason = "export", "查看或重新导出 EPUB"

    return ProjectFlowSnapshot(
        project_id=str(project_payload.get("project_id") or ""),
        project_name=str(project_payload.get("name") or project.name),
        project_path=str(project),
        updated_at=str(project_payload.get("updated_at") or ""),
        active_stage=active_stage,
        resume_workspace=resume_workspace,
        resume_reason=resume_reason,
        page_count=page_count,
        unresolved_count=unresolved_count,
        last_run_stage=str(last_run.get("stage") or ""),
        last_run_status=(runtime_task.status if runtime_task is not None else str(last_run.get("status") or "")),
        last_run_finished_at=str(last_run.get("finished_at") or ""),
        runtime_task_status=(runtime_task.status if runtime_task is not None else ""),
        resumable_ocr=resumable_ocr,
        resume_completed_steps=(int(runtime_task.completed_steps or 0) if runtime_task is not None else 0),
        stages=tuple(stages),
    )


__all__ = ["FlowStage", "ProjectFlowSnapshot", "build_project_flow_snapshot"]
