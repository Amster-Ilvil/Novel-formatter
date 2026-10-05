# -*- coding: utf-8 -*-
"""Small, shared runtime-task state model for long-running desktop work.

The project already persists durable run logs and resumable checkpoints.  This
module deliberately does *not* duplicate them.  It provides one lightweight
in-process state owner plus a cold-start adapter that turns those durable files
into a user-facing "running / cancelling / resumable / failed" task summary.

Qt is intentionally absent so OCR, AI adjudication, EPUB and future workers can
share the same state contract without importing the GUI.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any, Mapping


ACTIVE_STATUSES = {"running", "cancelling"}
TERMINAL_STATUSES = {"succeeded", "cancelled", "failed", "interrupted"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def normalize_task_status(value: Any, *, cold_start: bool = False) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "ok": "succeeded", "success": "succeeded", "done": "succeeded", "completed": "succeeded",
        "error": "failed", "failure": "failed", "exception": "failed",
        "stopped": "cancelled", "canceled": "cancelled", "cancel": "cancelled",
        "warning": "failed", "warn": "failed",
    }
    text = aliases.get(text, text)
    if text == "running" and cold_start:
        return "interrupted"
    if text in ACTIVE_STATUSES | TERMINAL_STATUSES:
        return text
    return "failed" if text else "interrupted"


@dataclass(frozen=True, slots=True)
class RuntimeTaskSnapshot:
    task_id: str
    kind: str
    project_path: str = ""
    status: str = "running"
    message: str = ""
    started_at: str = ""
    updated_at: str = ""
    finished_at: str = ""
    progress_current: int = 0
    progress_total: int = 0
    checkpoint_signature: str = ""
    checkpoint_status: str = ""
    completed_steps: int = 0
    resume_supported: bool = False
    error: str = ""

    @property
    def progress_percent(self) -> float | None:
        if self.progress_total <= 0:
            return None
        return max(0.0, min(100.0, self.progress_current * 100.0 / self.progress_total))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RuntimeTaskRegistry:
    """Thread-safe owner for current long-running task state.

    The registry is intentionally tiny: it owns state transitions, not workers,
    cancellation primitives, logs or checkpoints.  Those remain with the feature
    that actually performs the work.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tasks: dict[str, RuntimeTaskSnapshot] = {}

    def start(
        self,
        task_id: str,
        kind: str,
        *,
        project_path: str = "",
        message: str = "",
        checkpoint_signature: str = "",
        resume_supported: bool = False,
    ) -> RuntimeTaskSnapshot:
        now = _now_iso()
        state = RuntimeTaskSnapshot(
            task_id=str(task_id), kind=str(kind), project_path=str(project_path),
            status="running", message=str(message), started_at=now, updated_at=now,
            checkpoint_signature=str(checkpoint_signature),
            resume_supported=bool(resume_supported),
        )
        with self._lock:
            self._tasks[state.task_id] = state
        return state

    def update(
        self,
        task_id: str,
        *,
        message: str | None = None,
        progress_current: int | None = None,
        progress_total: int | None = None,
        checkpoint_status: str | None = None,
        completed_steps: int | None = None,
        resume_supported: bool | None = None,
    ) -> RuntimeTaskSnapshot | None:
        with self._lock:
            current = self._tasks.get(str(task_id))
            if current is None:
                return None
            state = replace(
                current,
                message=current.message if message is None else str(message),
                progress_current=current.progress_current if progress_current is None else max(0, int(progress_current)),
                progress_total=current.progress_total if progress_total is None else max(0, int(progress_total)),
                checkpoint_status=current.checkpoint_status if checkpoint_status is None else str(checkpoint_status),
                completed_steps=current.completed_steps if completed_steps is None else max(0, int(completed_steps)),
                resume_supported=current.resume_supported if resume_supported is None else bool(resume_supported),
                updated_at=_now_iso(),
            )
            self._tasks[state.task_id] = state
            return state

    def request_cancel(self, task_id: str, *, message: str = "") -> RuntimeTaskSnapshot | None:
        with self._lock:
            current = self._tasks.get(str(task_id))
            if current is None:
                return None
            state = replace(
                current, status="cancelling", message=str(message or current.message), updated_at=_now_iso()
            )
            self._tasks[state.task_id] = state
            return state

    def finish(
        self,
        task_id: str,
        status: str,
        *,
        message: str = "",
        error: str = "",
        resume_supported: bool | None = None,
    ) -> RuntimeTaskSnapshot | None:
        with self._lock:
            current = self._tasks.get(str(task_id))
            if current is None:
                return None
            final_status = normalize_task_status(status)
            now = _now_iso()
            state = replace(
                current,
                status=final_status,
                message=str(message or current.message),
                updated_at=now,
                finished_at=now,
                error=str(error or ""),
                resume_supported=(
                    current.resume_supported if resume_supported is None else bool(resume_supported)
                ),
            )
            self._tasks[state.task_id] = state
            return state

    def get(self, task_id: str) -> RuntimeTaskSnapshot | None:
        with self._lock:
            return self._tasks.get(str(task_id))

    def latest_for_project(self, project_path: str) -> RuntimeTaskSnapshot | None:
        target = str(project_path or "")
        with self._lock:
            rows = [state for state in self._tasks.values() if state.project_path == target]
        if not rows:
            return None
        return max(rows, key=lambda item: (item.updated_at, item.task_id))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(raw) if isinstance(raw, Mapping) else {}


def restore_project_runtime_task(
    project_path: str | Path,
    *,
    project_payload: Mapping[str, Any] | None = None,
    current_page_lineage_hash: str = "",
) -> RuntimeTaskSnapshot | None:
    """Build a cold-start task summary without resolving any large artifact.

    A persisted ``running`` run is necessarily interrupted when observed during a
    fresh process.  For OCR, the exact checkpoint signature saved in the run
    details is inspected.  Resume is offered only when the checkpoint belongs to
    the current page lineage and is not already terminal-successful.
    """
    project = Path(project_path).expanduser().resolve()
    payload = dict(project_payload or {})
    if not payload:
        payload = _read_json(project / "project.json")
    last_run = payload.get("last_run") if isinstance(payload.get("last_run"), Mapping) else {}
    if not last_run:
        return None
    stage = str(last_run.get("stage") or "")
    run_id = str(last_run.get("run_id") or "")
    if not run_id or not stage:
        return None
    # project.json intentionally keeps only a tiny last-run summary.  Resolve
    # the matching small run-log JSON (never OCR payloads) for checkpoint identity.
    run_payload = _read_json(project / "logs" / "runs" / f"{run_id}.json")
    if run_payload:
        last_run = run_payload
        stage = str(last_run.get("stage") or stage)
    details = last_run.get("details") if isinstance(last_run.get("details"), Mapping) else {}
    signature = str(details.get("checkpoint_signature") or "")
    checkpoint: dict[str, Any] = {}
    if signature and stage in {"single_ocr", "multi_ocr", "ocr"}:
        checkpoint = _read_json(project / "artifacts" / "checkpoints" / "ocr" / f"{signature}.json")
    completed_steps = len(dict(checkpoint.get("completed_steps") or {})) if checkpoint else 0
    checkpoint_lineage = str(checkpoint.get("page_lineage_hash") or "") if checkpoint else ""
    lineage_ok = not (
        current_page_lineage_hash and checkpoint_lineage and current_page_lineage_hash != checkpoint_lineage
    )
    checkpoint_status = str(checkpoint.get("status") or "") if checkpoint else ""
    raw_status = str(last_run.get("status") or "")
    status = normalize_task_status(raw_status, cold_start=True)
    resume_supported = bool(
        signature
        and checkpoint
        and lineage_ok
        and checkpoint_status not in {"ok", "succeeded"}
        and status in {"cancelled", "failed", "interrupted"}
    )
    summary = str(details.get("summary") or "")
    if resume_supported:
        resume_message = f"可从 OCR 断点继续；已记录 {completed_steps} 个完成阶段"
    elif status == "interrupted":
        resume_message = "上次任务异常中断；当前断点不可安全复用"
    else:
        resume_message = summary
    return RuntimeTaskSnapshot(
        task_id=run_id,
        kind=stage,
        project_path=str(project),
        status=status,
        message=resume_message,
        started_at=str(last_run.get("started_at") or last_run.get("timestamp") or ""),
        updated_at=str(last_run.get("finished_at") or last_run.get("started_at") or ""),
        finished_at=str(last_run.get("finished_at") or ""),
        checkpoint_signature=signature,
        checkpoint_status=checkpoint_status,
        completed_steps=completed_steps,
        resume_supported=resume_supported,
        error=str((last_run.get("error") or {}).get("message") if isinstance(last_run.get("error"), Mapping) else ""),
    )


__all__ = [
    "ACTIVE_STATUSES", "TERMINAL_STATUSES", "RuntimeTaskSnapshot", "RuntimeTaskRegistry",
    "normalize_task_status", "restore_project_runtime_task",
]
