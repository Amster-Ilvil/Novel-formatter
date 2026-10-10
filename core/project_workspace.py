# -*- coding: utf-8 -*-
"""Persistent project/workspace layer for Novel Formatter.

The GUI workspaces remain independent UI surfaces.  This module owns durable book
projects: source copies, page lineage, OCR/adjudication artifacts, pipeline
snapshots, exports and append-only run history.  It intentionally contains no Qt
imports so it can be tested and reused by CLI tools.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
import uuid
import zipfile
import csv
import io
from typing import Iterable, Mapping, Any

from utils.atomic_io import atomic_write_bytes, atomic_write_text
from utils.safe_archive import safe_extract_zip
from core.ocr_segment_cache import OcrSegmentCache
from core.artifact_pipeline import (
    ArtifactGraphStore, StageCacheStore, CheckpointStore, ContentAddressedJsonStore,
    is_object_ref, stable_json_hash, implementation_fingerprint,
)

SCHEMA_VERSION = 4
PROJECT_FILE = "project.json"
PAGE_STATE_FILE = "pages/state.json"
PROJECT_FORMAT = "novel-formatter-project"
RUN_LOG_FORMAT = "novel-formatter-run-log"
RUN_LOG_SCHEMA_VERSION = 4
RUN_STATUSES = {"running", "ok", "cancelled", "error", "warning"}
ADJUDICATION_EVENT_FILE = "adjudication/local/events.jsonl"
ADJUDICATION_EVENT_SCHEMA_VERSION = 2

# Multiple ProjectWorkspaceManager instances may target the same active project:
# the GUI owns one while low-frequency checkpoint workers create short-lived
# managers.  Keep append + checkpoint-compaction operations serialized per
# journal path so a decision appended while a 35 MB checkpoint is being written
# can never be erased by a later journal reset.
_ADJUDICATION_JOURNAL_LOCKS_GUARD = threading.Lock()
_ADJUDICATION_JOURNAL_LOCKS: dict[str, threading.RLock] = {}


def _adjudication_journal_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _ADJUDICATION_JOURNAL_LOCKS_GUARD:
        lock = _ADJUDICATION_JOURNAL_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _ADJUDICATION_JOURNAL_LOCKS[key] = lock
        return lock


class LegacyWorkspaceCompatibilityUnavailable(ValueError):
    """Stable compatibility seam for pre-current workspace schemas.

    The pre-stable build deliberately ships no migration implementation.  The
    exception and migration entry point remain stable so a future stable release
    can restore compatibility without changing GUI/caller APIs.
    """


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _safe_component(value: str, fallback: str = "project") -> str:
    text = re.sub(r"[\\/:*?\"<>|\x00-\x1f]+", "_", str(value or "")).strip(" .")
    text = re.sub(r"\s+", " ", text).strip()
    return (text[:120] or fallback)


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _multi_ocr_snapshot_summary(payload: Mapping[str, Any]) -> dict[str, int]:
    """Return tiny review counters without resolving the snapshot later."""
    raw_states = payload.get("fusion_states") if isinstance(payload, Mapping) else None
    states = raw_states if isinstance(raw_states, list) else []
    unresolved = 0
    selected = 0
    for raw in states:
        if not isinstance(raw, Mapping):
            continue
        candidates = raw.get("candidates") if isinstance(raw.get("candidates"), list) else []
        selected_index = raw.get("selected_index")
        requires_confirmation = bool(raw.get("requires_confirmation", False))
        is_selected = selected_index is not None
        if is_selected:
            selected += 1
        elif len(candidates) > 1 or requires_confirmation:
            unresolved += 1
    return {
        "total_rows": len(states),
        "selected_rows": selected,
        "unresolved": unresolved,
    }


def _require_run_log_schema(payload: Any, label: str = "run log") -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} 不是 JSON 对象")
    if int(payload.get("schema_version") or 0) != RUN_LOG_SCHEMA_VERSION:
        raise ValueError(
            f"{label} schema={payload.get('schema_version')!r} 不受当前开发版支持；"
            f"当前只接受 schema {RUN_LOG_SCHEMA_VERSION}。"
        )
    if str(payload.get("format") or "") != RUN_LOG_FORMAT:
        raise ValueError(f"{label} format 不受当前开发版支持")
    return payload


def _require_workspace_schema(payload: Any, label: str) -> dict:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} 顶层必须是 JSON 对象。")
    version = int(payload.get("schema_version") or 0)
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{label} schema={version} 不受当前开发版支持；当前只接受 {SCHEMA_VERSION}。"
        )
    return payload


@dataclass(frozen=True)
class ProjectInfo:
    project_id: str
    name: str
    path: str
    created_at: str = ""
    updated_at: str = ""


class ProjectWorkspaceManager:
    """Manage one workspace root and at most one active book project."""

    LAYOUT = (
        "source/original",
        "pages/source",
        "pages/processed",
        "artifacts/stages",
        "artifacts/cache",
        "artifacts/cache_assets",
        "artifacts/checkpoints",
        "artifacts/ocr/single",
        "artifacts/ocr/multi",
        "artifacts/compare",
        "adjudication/local",
        "adjudication/cloud",
        "documents",
        "revisions/ocr_history",
        "revisions/ai_history",
        "revisions/objects",
        "assets/cover",
        "assets/images",
        "exports/epub",
        "exports/packages",
        "logs/runs",
        "pipeline",
        "metadata",
        "recovery",
    )

    def __init__(self, workspace_root: str | Path | None = None):
        default = Path.home() / "Documents" / "Novel Formatter Workspace"
        self.workspace_root = Path(workspace_root).expanduser() if workspace_root else default
        self.active_project: Path | None = None
        self._checked_adjudication_journals: set[str] = set()
        # Recent-project dashboards refresh frequently.  Cache parsed project.json
        # metadata by exact file identity so refreshes pay only a cheap stat call.
        self._project_info_cache: dict[str, tuple[int, int, ProjectInfo | None]] = {}

    # ------------------------------------------------------------------
    # workspace/project discovery
    # ------------------------------------------------------------------
    @property
    def projects_root(self) -> Path:
        return self.workspace_root / "projects"

    def set_workspace_root(self, path: str | Path, *, create: bool = True) -> Path:
        root = Path(path).expanduser().resolve()
        if create:
            (root / "projects").mkdir(parents=True, exist_ok=True)
        self.workspace_root = root
        if self.active_project is not None:
            try:
                self.active_project.relative_to(root)
            except ValueError:
                self.active_project = None
        return root

    def list_projects(self) -> list[ProjectInfo]:
        root = self.projects_root
        if not root.exists():
            return []
        result: list[ProjectInfo] = []
        seen_cache_keys: set[str] = set()
        for folder in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name.casefold()):
            try:
                seen_cache_keys.add(str(folder.resolve()))
            except OSError:
                seen_cache_keys.add(str(folder))
            info = self._read_project_info(folder)
            if info is not None:
                result.append(info)
        # External folder deletion/renames must not leave an unbounded metadata cache.
        for cache_key in tuple(self._project_info_cache):
            if cache_key not in seen_cache_keys:
                self._project_info_cache.pop(cache_key, None)
        if len(self._project_info_cache) > 512:
            # list_projects() reads in deterministic folder order; retaining the most
            # recent 512 entries is ample while keeping memory bounded.
            for cache_key in tuple(self._project_info_cache)[:-512]:
                self._project_info_cache.pop(cache_key, None)
        result.sort(key=lambda item: (item.updated_at, item.name.casefold()), reverse=True)
        return result

    def create_project(self, name: str) -> ProjectInfo:
        self.projects_root.mkdir(parents=True, exist_ok=True)
        display = str(name or "").strip() or "未命名项目"
        base = _safe_component(display, "project")
        path = self.projects_root / base
        serial = 2
        while path.exists():
            path = self.projects_root / f"{base} {serial}"
            serial += 1
        path.mkdir(parents=True, exist_ok=False)
        self._ensure_layout(path)
        now = _now_iso()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "format": PROJECT_FORMAT,
            "project_id": uuid.uuid4().hex,
            "name": display,
            "created_at": now,
            "updated_at": now,
            "active_stage": "",
            "document_page_lineage_hash": "",
            "source_imports": [],
        }
        self._write_project_payload(path, payload)
        self.active_project = path
        return self._project_info_from_payload(path, payload)

    def open_project(self, path: str | Path) -> ProjectInfo:
        project = Path(path).expanduser().resolve()
        self._validate_current_project(project)
        info = self._read_project_info(project)
        if info is None:
            raise ValueError(f"不是有效的 Novel Formatter 项目：{project}")
        self._ensure_layout(project)
        self.active_project = project
        return info

    def import_project_folder(self, path: str | Path, *, copy_into_workspace: bool = True) -> ProjectInfo:
        """Import an existing project folder.

        By default the project is copied into the current workspace so deleting or
        moving the external backup cannot break the restored book.  The current
        development build accepts only the exact current workspace schema.
        """
        source = Path(path).expanduser().resolve()
        self._validate_current_project(source)
        info = self._read_project_info(source)
        if info is None:
            raise ValueError(f"不是有效的 Novel Formatter 项目：{source}")
        if not copy_into_workspace:
            return self.open_project(source)
        try:
            source.relative_to(self.projects_root.resolve())
            return self.open_project(source)
        except ValueError:
            pass

        self.projects_root.mkdir(parents=True, exist_ok=True)
        target = self._unique_project_dir(info.name)
        shutil.copytree(source, target, copy_function=shutil.copy2)
        self._normalize_imported_project_identity(target, source_project_id=info.project_id)
        return self.open_project(target)

    def export_project_archive(self, target_zip: str | Path) -> Path:
        """Create a complete portable project backup ZIP."""
        project = self._require_project()
        target = Path(target_zip).expanduser()
        if target.suffix.lower() != ".zip":
            target = target.with_suffix(".zip")
        target.parent.mkdir(parents=True, exist_ok=True)

        # Audit first so obviously broken projects are not silently presented as
        # healthy backups.  Missing transient exports are warnings; missing pages
        # remain recorded in the archive manifest for later repair.
        audit = self.audit_project(deep=True)
        manifest = {
            "format": "novel-formatter-project-backup",
            "schema_version": SCHEMA_VERSION,
            "created_at": _now_iso(),
            "project_id": self.project_context().get("project_id", ""),
            "project_name": self.project_context().get("name", project.name),
            "audit": audit,
        }
        temp_target = target.with_name(f".{target.name}.{uuid.uuid4().hex[:8]}.tmp")
        compression = zipfile.ZIP_DEFLATED
        try:
            with zipfile.ZipFile(temp_target, "w", compression=compression, allowZip64=True) as archive:
                root_name = _safe_component(project.name, "project")
                for file in sorted(project.rglob("*")):
                    if not file.is_file():
                        continue
                    try:
                        if file.resolve() == target.resolve() or file.resolve() == temp_target.resolve():
                            continue
                    except OSError:
                        pass
                    rel = file.relative_to(project).as_posix()
                    if rel == "metadata/backup_manifest.json":
                        continue
                    archive.write(
                        file,
                        f"{root_name}/{rel}",
                        compress_type=self._zip_compression_for_path(file),
                    )
                archive.writestr(
                    f"{root_name}/metadata/backup_manifest.json",
                    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                )
            os.replace(temp_target, target)
        finally:
            temp_target.unlink(missing_ok=True)
        self.record_run("project_backup", details={"archive": str(target), "audit_ok": bool(audit.get("ok"))})
        return target

    def import_project_archive(self, archive_path: str | Path) -> ProjectInfo:
        """Safely restore a project backup ZIP into the current workspace."""
        archive_path = Path(archive_path).expanduser().resolve()
        if not archive_path.exists():
            raise FileNotFoundError(str(archive_path))
        self.projects_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="nf-project-import-", dir=str(self.workspace_root)) as tmp:
            temp_root = Path(tmp)
            safe_extract_zip(archive_path, temp_root)
            candidates = [p for p in temp_root.rglob(PROJECT_FILE) if p.is_file()]
            if len(candidates) != 1:
                raise ValueError(f"项目备份中应恰好包含 1 个 project.json，实际找到 {len(candidates)} 个")
            source = candidates[0].parent
            self._validate_current_project(source)
            backup_manifest = source / "metadata" / "backup_manifest.json"
            if not backup_manifest.is_file():
                raise ValueError("当前项目备份缺少 metadata/backup_manifest.json。")
            manifest_payload = _require_workspace_schema(
                json.loads(backup_manifest.read_text(encoding="utf-8")),
                "metadata/backup_manifest.json",
            )
            if str(manifest_payload.get("format") or "") != "novel-formatter-project-backup":
                raise ValueError("项目备份 format 标记无效。")
            info = self._read_project_info(source)
            if info is None:
                raise ValueError("项目备份中的 project.json 无效")
            target = self._unique_project_dir(info.name)
            shutil.copytree(source, target, copy_function=shutil.copy2)
            self._normalize_imported_project_identity(target, source_project_id=info.project_id)
        restored = self.open_project(target)
        self.record_run("project_restore", details={"archive": str(archive_path)})
        return restored

    def open_project_by_id(self, project_id: str) -> ProjectInfo:
        for info in self.list_projects():
            if info.project_id == str(project_id):
                return self.open_project(info.path)
        raise FileNotFoundError(f"找不到项目：{project_id}")

    def delete_project(self, path: str | Path) -> None:
        project = Path(path).expanduser().resolve()
        # Destructive delete is intentionally restricted to the current workspace's
        # projects/ directory.  External imported project folders are never removed.
        root = self.projects_root.resolve()
        try:
            project.relative_to(root)
        except ValueError as exc:
            raise ValueError("只能删除当前工作区 projects 目录中的项目") from exc
        if self._read_project_info(project) is None:
            raise ValueError("目标目录不是有效项目")
        if self.active_project and self.active_project.resolve() == project:
            self.active_project = None
        shutil.rmtree(project)

    def project_context(self) -> dict[str, str]:
        if self.active_project is None:
            return {}
        info = self._read_project_info(self.active_project)
        if info is None:
            return {}
        return asdict(info)


    @staticmethod
    def _collect_object_ref_digests(value: Any, output: set[str]) -> None:
        if is_object_ref(value):
            digest = str(value.get("sha256") or "")
            if re.fullmatch(r"[0-9a-f]{64}", digest):
                output.add(digest)
            return
        if isinstance(value, Mapping):
            for child in value.values():
                ProjectWorkspaceManager._collect_object_ref_digests(child, output)
        elif isinstance(value, (list, tuple)):
            for child in value:
                ProjectWorkspaceManager._collect_object_ref_digests(child, output)

    def prune_unreferenced_json_objects(self) -> dict[str, int]:
        """Delete only compressed JSON objects with no live logical reference."""
        project = self._require_project()
        live: set[str] = set()
        scan_roots = [
            project / "artifacts" / "stages",
            project / "artifacts" / "ocr",
            project / "artifacts" / "cache",
            project / "adjudication",
            project / "revisions",
        ]
        object_root = project / "revisions" / "objects"
        invalid_reference_files: list[str] = []
        for root in scan_roots:
            if not root.exists():
                continue
            for path in root.rglob("*.json"):
                # revisions/ may later contain logical revision refs, but never parse
                # compressed object files themselves here.
                if object_root in path.parents:
                    continue
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    invalid_reference_files.append(path.relative_to(project).as_posix())
                    continue
                self._collect_object_ref_digests(raw, live)
        removed = kept = 0
        if invalid_reference_files:
            # Fail closed. A malformed logical file may still contain the only
            # reference to a valid immutable object. Never delete objects until
            # every potential reference file has been parsed successfully.
            if object_root.exists():
                kept = sum(1 for path in object_root.rglob("*.json.gz") if path.is_file())
            return {
                "live": len(live),
                "kept": kept,
                "removed": 0,
                "skipped": 1,
                "invalid_reference_files": len(invalid_reference_files),
            }
        if object_root.exists():
            for path in object_root.rglob("*.json.gz"):
                digest = path.name.split(".", 1)[0]
                if digest in live:
                    kept += 1
                    continue
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    kept += 1
            for directory in sorted(
                [p for p in object_root.rglob("*") if p.is_dir()],
                key=lambda p: len(p.parts), reverse=True,
            ):
                try:
                    directory.rmdir()
                except OSError:
                    pass
        return {
            "live": len(live), "kept": kept, "removed": removed,
            "skipped": 0, "invalid_reference_files": 0,
        }

    def project_storage_stats(self) -> dict[str, Any]:
        """Return logical workspace usage grouped by durable/cache categories."""
        project = self._require_project()

        def tree_bytes(root: Path) -> int:
            total = 0
            if not root.exists():
                return 0
            for item in root.rglob("*"):
                if not item.is_file():
                    continue
                try:
                    total += int(item.stat().st_size)
                except OSError:
                    pass
            return total

        groups = {
            "source": project / "source",
            "pages": project / "pages",
            "ocr": project / "artifacts" / "ocr",
            "stages": project / "artifacts" / "stages",
            "stage_cache": project / "artifacts" / "cache",
            "segment_cache": project / "artifacts" / "segment_cache",
            "cache_assets": project / "artifacts" / "cache_assets",
            "objects": project / "revisions" / "objects",
            "adjudication": project / "adjudication",
            "exports": project / "exports",
            "logs": project / "logs",
        }
        by_group = {key: tree_bytes(path) for key, path in groups.items()}
        total = tree_bytes(project)
        return {
            "project": str(project),
            "total_bytes": total,
            "groups": by_group,
            "measured_at": _now_iso(),
        }

    def compact_project_storage(self) -> dict[str, Any]:
        """Losslessly compact a project while preserving every public path.

        Large current-schema JSON payloads are moved into the compressed immutable
        object store and their original files become lightweight references.  This
        maintenance operation never upgrades or interprets older workspace schemas.
        """
        project = self._require_project()
        before = self.project_storage_stats()
        object_store = ContentAddressedJsonStore(project)
        converted = {
            "stage_documents": 0,
            "multi_ocr_snapshots": 0,
            "adjudication_states": 0,
            "stage_cache_entries": 0,
            "segment_cache_scanned": 0,
            "segment_cache_imported": 0,
            "segment_cache_removed": 0,
        }
        errors: list[str] = []

        def compact_wrapper(path: Path, inline_key: str, ref_key: str, kind: str) -> bool:
            if not path.is_file():
                return False
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    return False
                if is_object_ref(raw.get(ref_key)):
                    # Verify the object now; leave a healthy existing reference alone.
                    object_store.resolve(raw[ref_key])
                    return False
                payload = raw.get(inline_key)
                if not isinstance(payload, dict):
                    return False
                reference = object_store.put(payload, kind=kind)
                raw.pop(inline_key, None)
                raw[ref_key] = reference
                atomic_write_text(path, json.dumps(raw, ensure_ascii=False, indent=2) + "\n")
                return True
            except Exception as exc:
                errors.append(f"{path.relative_to(project).as_posix()}: {exc}")
                return False

        for path in sorted((project / "artifacts" / "stages").glob("*.json")):
            if compact_wrapper(path, "document", "document_ref", f"stage_document:{path.stem}"):
                converted["stage_documents"] += 1

        multi_path = project / "artifacts" / "ocr" / "multi" / "latest.json"
        if compact_wrapper(multi_path, "payload", "payload_ref", "multi_ocr_snapshot"):
            converted["multi_ocr_snapshots"] += 1

        for channel in ("local", "cloud"):
            state_path = project / "adjudication" / channel / "state.json"
            if compact_wrapper(state_path, "state", "state_ref", f"adjudication:{channel}"):
                converted["adjudication_states"] += 1

        cache_root = project / "artifacts" / "cache"
        if cache_root.exists():
            for path in sorted(cache_root.rglob("*.json")):
                if compact_wrapper(path, "payload", "payload_ref", f"stage_cache:{path.parent.name}"):
                    converted["stage_cache_entries"] += 1

        # Current-schema projects write OCR segment caches directly to SQLite.
        # Older shard-based caches belong to older workspace schemas and are not
        # migrated by this development build.

        object_gc = self.prune_unreferenced_json_objects()
        converted["json_objects_removed"] = int(object_gc.get("removed") or 0)
        after = self.project_storage_stats()
        payload = self._read_project_payload(project)
        payload["storage_format"] = "content-addressed-json-v1"
        payload["storage_compacted_at"] = _now_iso()
        payload["storage_last_compaction"] = {
            "before_bytes": int(before.get("total_bytes") or 0),
            "after_bytes": int(after.get("total_bytes") or 0),
            "saved_bytes": max(0, int(before.get("total_bytes") or 0) - int(after.get("total_bytes") or 0)),
            "converted": dict(converted),
            "errors": len(errors),
        }
        self._touch_project_payload(project, payload)
        result = {
            "ok": not errors,
            "before": before,
            "after": after,
            "saved_bytes": max(0, int(before.get("total_bytes") or 0) - int(after.get("total_bytes") or 0)),
            "converted": converted,
            "errors": errors,
        }
        self.record_run("workspace_compaction", status="ok" if not errors else "warning", details={
            "before_bytes": int(before.get("total_bytes") or 0),
            "after_bytes": int(after.get("total_bytes") or 0),
            "saved_bytes": int(result["saved_bytes"]),
            "converted": converted,
            "error_count": len(errors),
        })
        return result

    # ------------------------------------------------------------------
    # source/page persistence
    # ------------------------------------------------------------------
    def import_sources(self, paths: Iterable[str | Path]) -> list[str]:
        project = self._require_project()
        payload = self._read_project_payload(project)
        mappings = {
            str(item.get("source_path") or ""): str(item.get("project_path") or "")
            for item in (payload.get("source_imports") or [])
            if isinstance(item, dict)
        }
        imported: list[str] = []
        imports = list(payload.get("source_imports") or [])
        for raw in paths:
            source = Path(raw).expanduser().resolve()
            if not source.exists():
                raise FileNotFoundError(str(source))
            if self._is_within_project(source, project):
                imported.append(str(source))
                continue
            source_key = str(source)
            previous_rel = mappings.get(source_key, "")
            if previous_rel:
                previous = project / previous_rel
                if previous.exists():
                    imported.append(str(previous))
                    continue
            target = self._unique_target(project / "source" / "original", source.name)
            if source.is_dir():
                shutil.copytree(source, target, copy_function=shutil.copy2)
                size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
                digest = ""
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                size = target.stat().st_size
                digest = _sha256_file(target)
            rel = target.relative_to(project).as_posix()
            imports.append({
                "source_path": source_key,
                "project_path": rel,
                "imported_at": _now_iso(),
                "kind": "folder" if source.is_dir() else "file",
                "size": int(size),
                "sha256": digest,
            })
            imported.append(str(target))
        payload["source_imports"] = imports
        self._touch_project_payload(project, payload)
        return imported

    def persist_page_images(self, paths: Iterable[str | Path]) -> list[str]:
        """Persist source/PDF-rendered pages under ``pages/source``."""
        return self._persist_page_files(paths, "source")

    def persist_processed_page_images(self, paths: Iterable[str | Path]) -> list[str]:
        """Persist scan-preprocessed pages under ``pages/processed``."""
        return self._persist_page_files(paths, "processed", keep_project_sources=False)

    def _persist_page_files(
        self, paths: Iterable[str | Path], bucket: str, *, keep_project_sources: bool = True
    ) -> list[str]:
        project = self._require_project()
        result: list[str] = []
        destination_root = project / "pages" / bucket
        destination_root.mkdir(parents=True, exist_ok=True)
        for index, raw in enumerate(paths, start=1):
            source = Path(raw).expanduser().resolve()
            if not source.exists():
                raise FileNotFoundError(str(source))
            if keep_project_sources and self._is_within_project(source, project):
                result.append(str(source))
                continue
            # Already in the requested durable bucket.
            try:
                source.relative_to(destination_root.resolve())
                result.append(str(source))
                continue
            except ValueError:
                pass
            suffix = source.suffix.lower() or ".png"
            target = destination_root / f"{index:06d}{suffix}"
            if target.exists():
                try:
                    if os.path.samefile(target, source):
                        result.append(str(target.resolve()))
                        continue
                except OSError:
                    pass
                same = target.stat().st_size == source.stat().st_size
                if same:
                    try:
                        same = _sha256_file(target) == _sha256_file(source)
                    except OSError:
                        same = False
                if not same:
                    target.unlink(missing_ok=True)
            if not target.exists():
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copy2(source, target)
            result.append(str(target.resolve()))
        return result

    def save_page_state(
        self,
        page_images: Iterable[str | Path],
        page_overrides: Mapping[int | str, str] | None = None,
        auto_suggested: Iterable[int] | None = None,
        raw_inputs: Iterable[str | Path] | None = None,
        *,
        source_label: str = "",
        original_pdf_sources: Iterable[str | Path] | None = None,
        pdf_physical_page_map: Mapping[int | str, int | str] | None = None,
    ) -> dict:
        project = self._require_project()
        image_paths = [Path(p).expanduser().resolve() for p in page_images]
        raw_input_paths = [Path(p).expanduser() for p in (raw_inputs or [])]
        images = [self._portable_path(project, p) for p in image_paths]
        raws = [self._portable_path(project, p) for p in raw_input_paths]
        original_pdfs = [self._portable_path(project, Path(p)) for p in (original_pdf_sources or [])]
        physical_map = {str(int(k)): int(v) for k, v in dict(pdf_physical_page_map or {}).items() if int(k) > 0 and int(v) > 0}
        overrides = {str(int(k)): str(v) for k, v in dict(page_overrides or {}).items()}
        page_fingerprints = self._page_fingerprints(project, image_paths)
        state = {
            "schema_version": SCHEMA_VERSION,
            "saved_at": _now_iso(),
            "page_images": images,
            "page_fingerprints": page_fingerprints,
            "page_overrides": overrides,
            "auto_suggested": sorted({int(v) for v in (auto_suggested or [])}),
            "raw_inputs": raws,
            "original_pdf_sources": original_pdfs,
            "pdf_physical_page_map": physical_map,
            "source_label": str(source_label or ""),
        }
        state["page_lineage_hash"] = self.page_lineage_hash_from_state(state)
        target = project / PAGE_STATE_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(target, json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        graph = ArtifactGraphStore(project)
        source_nodes: list[str] = []
        page_path_set = {path.resolve() for path in image_paths}
        registrations: list[dict[str, Any]] = []
        for raw_path in raw_input_paths:
            try:
                candidate = Path(raw_path).expanduser().resolve()
                # When a page deletion snapshots the remaining page paths as
                # raw inputs, the page_image nodes below already represent
                # those files. Avoid hashing and registering every page a
                # second time as an independent source file.
                if not candidate.is_file() or candidate in page_path_set:
                    continue
                digest = _sha256_file(candidate)
                source_nodes.append(f"source_file:{digest}")
                registrations.append({
                    "kind": "source_file",
                    "digest": digest,
                    "path": self._portable_path(project, candidate),
                    "metadata": {
                        "name": candidate.name,
                        "size": int(candidate.stat().st_size),
                        "sha256": digest,
                    },
                })
            except Exception:
                continue

        page_nodes: list[str] = []
        for ordinal, (image_path, fingerprint) in enumerate(zip(image_paths, page_fingerprints), start=1):
            try:
                page_digest = stable_json_hash({
                    "sha256": str(fingerprint.get("sha256") or ""),
                    "path": str(fingerprint.get("path") or ""),
                })
                page_nodes.append(f"page_image:{page_digest}")
                registrations.append({
                    "kind": "page_image",
                    "digest": page_digest,
                    "path": self._portable_path(project, image_path),
                    "dependencies": source_nodes,
                    "metadata": {
                        "ordinal": ordinal,
                        "size": int(fingerprint.get("size") or 0),
                        "sha256": str(fingerprint.get("sha256") or ""),
                    },
                })
            except Exception:
                continue

        registrations.append({
            "kind": "page_set",
            "digest": str(state["page_lineage_hash"]),
            "alias": "pages.current",
            "path": self._portable_path(project, target),
            "dependencies": page_nodes or source_nodes,
            "metadata": {"page_count": len(images), "source_label": str(source_label or "")},
        })
        graph.register_many(registrations)
        payload = self._read_project_payload(project)
        payload["page_count"] = len(images)
        payload["page_lineage_hash"] = state["page_lineage_hash"]
        self._touch_project_payload(project, payload)
        return state

    def save_page_overrides(
        self,
        page_overrides: Mapping[int | str, str] | None = None,
        auto_suggested: Iterable[int] | None = None,
        *,
        source_label: str | None = None,
    ) -> dict:
        """Persist classification-only page edits without re-hashing page images.

        Manual page tagging is a high-frequency UI action.  The ordered source
        pages and their fingerprints do not change when a page is relabelled as
        cover/illustration/paragraph, so recomputing every page SHA and
        re-registering hundreds of ArtifactGraph nodes is pure work.  Reuse the
        already sealed page lineage and atomically update only the lightweight
        overlay fields.
        """
        project = self._require_project()
        target = project / PAGE_STATE_FILE
        if not target.exists():
            raise FileNotFoundError(str(target))
        raw = _require_workspace_schema(
            json.loads(target.read_text(encoding="utf-8")), "pages/state.json"
        )
        raw["saved_at"] = _now_iso()
        raw["page_overrides"] = {
            str(int(k)): str(v) for k, v in dict(page_overrides or {}).items()
        }
        raw["auto_suggested"] = sorted({int(v) for v in (auto_suggested or [])})
        if source_label is not None:
            raw["source_label"] = str(source_label or "")
        # Classification is deliberately excluded from page lineage. Preserve
        # the existing content fingerprint hash byte-for-byte where possible.
        raw["page_lineage_hash"] = str(
            raw.get("page_lineage_hash") or self.page_lineage_hash_from_state(raw)
        )
        atomic_write_text(target, json.dumps(raw, ensure_ascii=False, indent=2) + "\n")
        return raw

    def load_page_state(self) -> dict:
        project = self._require_project()
        path = project / PAGE_STATE_FILE
        if not path.exists():
            return {}
        raw = _require_workspace_schema(
            json.loads(path.read_text(encoding="utf-8")), "pages/state.json"
        )
        images = [str(self._resolve_portable(project, p)) for p in raw.get("page_images", [])]
        # An interrupted progressive PDF import could save session paths after
        # durable copies had already been created. Resolve those copies using
        # the saved ordinal and content identity, before the GUI drops missing
        # pages and shifts all subsequent page numbers.
        fingerprints = list(raw.get("page_fingerprints") or [])
        session_path_indexes = [
            index for index, value in enumerate(images)
            if "novel_formatter_sessions" in Path(value).parts
        ]
        recovered_path_indexes: set[int] = set()
        for index, value in enumerate(images):
            source = Path(value)
            if "novel_formatter_sessions" not in source.parts:
                continue
            candidate = project / "pages" / "source" / f"{index + 1:06d}{source.suffix.lower()}"
            if not candidate.is_file():
                continue
            try:
                if source.exists():
                    matches = source.samefile(candidate)
                else:
                    fingerprint = fingerprints[index] if index < len(fingerprints) else {}
                    if not isinstance(fingerprint, Mapping):
                        continue
                    matches = (
                        int(fingerprint.get("size") or -1) == candidate.stat().st_size
                        and bool(fingerprint.get("sha256"))
                        and _sha256_file(candidate) == str(fingerprint["sha256"])
                    )
                if matches:
                    images[index] = str(candidate.resolve())
                    recovered_path_indexes.add(index)
            except OSError:
                continue
        raws = [str(self._resolve_portable(project, p)) for p in raw.get("raw_inputs", [])]
        original_pdfs = [str(self._resolve_portable(project, p)) for p in raw.get("original_pdf_sources", [])]
        return {
            **raw,
            "page_images": images,
            "_page_paths_recovered": bool(
                session_path_indexes
                and recovered_path_indexes == set(session_path_indexes)
                and all(Path(value).is_file() for value in images)
            ),
            "raw_inputs": raws,
            "original_pdf_sources": original_pdfs,
            "pdf_physical_page_map": {int(k): int(v) for k, v in dict(raw.get("pdf_physical_page_map") or {}).items()},
            "page_overrides": {int(k): str(v) for k, v in dict(raw.get("page_overrides") or {}).items()},
            "auto_suggested": {int(v) for v in raw.get("auto_suggested", [])},
        }

    @staticmethod
    def page_lineage_hash_from_state(state: Mapping[str, Any] | None) -> str:
        # Classification overlays do not invalidate OCR text.  Only ordered page
        # identities matter for document compatibility.  Schema v2 uses content
        # fingerprints so replacing an image in place invalidates old OCR/corrections.
        raw = dict(state or {})
        fingerprints = raw.get("page_fingerprints") or []
        if isinstance(fingerprints, list) and fingerprints:
            compact = [
                {
                    "path": str(item.get("path") or ""),
                    "size": int(item.get("size") or 0),
                    "sha256": str(item.get("sha256") or ""),
                }
                for item in fingerprints
                if isinstance(item, dict)
            ]
            return _json_hash({"page_fingerprints": compact})
        images = [str(v) for v in raw.get("page_images", [])]
        return _json_hash({"page_images": images})

    def current_page_lineage_hash(self) -> str:
        project = self._require_project()
        path = project / PAGE_STATE_FILE
        if not path.exists():
            return ""
        try:
            raw = _require_workspace_schema(
                json.loads(path.read_text(encoding="utf-8")), "pages/state.json"
            )
        except Exception as exc:
            raise ValueError(f"当前工作区页面状态损坏：{exc}") from exc
        recovered = self.load_page_state()
        if recovered.get("_page_paths_recovered"):
            # The fallback verifies each durable image against the saved
            # fingerprint (or confirms it is the same inode), so path recovery
            # preserves the page set and its existing OCR lineage.
            return str(raw.get("page_lineage_hash") or self.page_lineage_hash_from_state(raw))
        saved_fingerprints = raw.get("page_fingerprints") or []
        if isinstance(saved_fingerprints, list) and saved_fingerprints:
            page_paths = [self._resolve_portable(project, p) for p in raw.get("page_images", [])]
            try:
                actual = self._page_fingerprints(project, page_paths)
            except OSError:
                return ""
            return self.page_lineage_hash_from_state({"page_fingerprints": actual})
        return str(raw.get("page_lineage_hash") or self.page_lineage_hash_from_state(raw))

    # ------------------------------------------------------------------
    # pipeline/artifact/adjudication persistence
    # ------------------------------------------------------------------
    def save_pipeline_config(self, pipeline: str, state: Mapping[str, Any]) -> Path:
        project = self._require_project()
        key = _safe_component(pipeline, "pipeline").replace(" ", "_").lower()
        path = project / "pipeline" / f"{key}.json"
        payload = {
            "schema_version": SCHEMA_VERSION,
            "pipeline": str(pipeline),
            "saved_at": _now_iso(),
            "state": dict(state or {}),
        }
        atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        graph = ArtifactGraphStore(project)
        page_node = graph.current_node("pages.current")
        graph.register(
            kind="pipeline_config", digest=stable_json_hash(payload["state"]),
            alias=f"pipeline.{key}", path=self._portable_path(project, path),
            dependencies=[page_node] if page_node else [],
            metadata={"pipeline": str(pipeline)},
        )
        return path

    def load_pipeline_config(self, pipeline: str) -> dict:
        project = self._require_project()
        key = _safe_component(pipeline, "pipeline").replace(" ", "_").lower()
        path = project / "pipeline" / f"{key}.json"
        if not path.exists():
            return {}
        raw = _require_workspace_schema(
            json.loads(path.read_text(encoding="utf-8")), f"pipeline/{key}.json"
        )
        return dict(raw.get("state") or {})

    def save_stage_document(self, stage: str, document_payload: Mapping[str, Any]) -> Path:
        project = self._require_project()
        safe_stage = _safe_component(stage, "stage").replace(" ", "_")
        lineage = self.current_page_lineage_hash()
        wrapper = {
            "schema_version": SCHEMA_VERSION,
            "stage": str(stage),
            "saved_at": _now_iso(),
            "page_lineage_hash": lineage,
            "document": dict(document_payload),
        }
        stage_path = project / "artifacts" / "stages" / f"{safe_stage}.json"
        current_path = project / "artifacts" / "stages" / "current.json"
        document_ref = ContentAddressedJsonStore(project).put(
            wrapper.pop("document"), kind=f"stage_document:{safe_stage}"
        )
        wrapper["document_ref"] = document_ref
        data = json.dumps(wrapper, ensure_ascii=False, indent=2) + "\n"
        atomic_write_text(stage_path, data)
        # current.json is intentionally another tiny logical reference, not a
        # second 40-80 MB document copy.
        atomic_write_text(current_path, data)
        payload = self._read_project_payload(project)
        payload["active_stage"] = str(stage)
        payload["document_page_lineage_hash"] = lineage
        self._touch_project_payload(project, payload)
        graph = ArtifactGraphStore(project)
        dependencies = []
        page_node = graph.current_node("pages.current")
        if page_node:
            dependencies.append(page_node)
        for alias in ("pipeline.multi_ocr", "pipeline.single_ocr"):
            node = graph.current_node(alias)
            if node and node not in dependencies:
                dependencies.append(node)
        graph.register(
            kind="stage_document", digest=str(document_ref.get("sha256") or stable_json_hash(document_payload)),
            alias=f"stage.{safe_stage}", path=self._portable_path(project, stage_path),
            dependencies=dependencies, metadata={"stage": str(stage), "page_lineage_hash": lineage},
        )
        graph.register(
            kind="stage_document", digest=str(document_ref.get("sha256") or stable_json_hash(document_payload)),
            alias="stage.current", path=self._portable_path(project, current_path),
            dependencies=dependencies, metadata={"stage": str(stage), "page_lineage_hash": lineage},
        )
        return stage_path

    def load_current_stage_document(self, *, require_compatible_pages: bool = True) -> tuple[str, dict] | None:
        project = self._require_project()
        path = project / "artifacts" / "stages" / "current.json"
        if not path.exists():
            return None
        raw = _require_workspace_schema(
            json.loads(path.read_text(encoding="utf-8")), "artifacts/stages/current.json"
        )
        if require_compatible_pages:
            saved = str(raw.get("page_lineage_hash") or "")
            current = self.current_page_lineage_hash()
            if saved and current and saved != current:
                return None
        reference = raw.get("document_ref")
        if not is_object_ref(reference):
            raise ValueError("当前 stage 必须使用 schema 4 document_ref；不兼容旧内联 document。")
        try:
            resolved = ContentAddressedJsonStore(project).resolve(reference)
        except Exception as exc:
            raise ValueError(f"当前 stage 对象缺失或损坏：{exc}") from exc
        doc = resolved if isinstance(resolved, dict) else None
        if not isinstance(doc, dict):
            raise ValueError("当前 stage document_ref 未解析为 JSON 对象。")
        # Stage documents may be created on a different Mac/Windows/Linux path.
        # Rebind only a matching copy already owned by this project.  Never
        # modify the content-addressed payload, source hashes, or OCR decisions.
        self._rebind_stage_image_paths(doc, project)
        return str(raw.get("stage") or "ocr"), doc

    @staticmethod
    def _rebind_stage_image_paths(doc: dict, project: Path) -> int:
        source_dir = project / "source" / "original"
        if not source_dir.is_dir():
            return 0
        changed = 0
        for group in (doc.get("pages"), doc.get("blocks")):
            if not isinstance(group, list):
                continue
            for item in group:
                if not isinstance(item, dict):
                    continue
                original = item.get("image_path")
                if not isinstance(original, str) or not original:
                    continue
                # Path separators can differ between the source and target OS.
                name = original.replace("\\", "/").rsplit("/", 1)[-1]
                if not name or name in (".", ".."):
                    continue
                local_copy = source_dir / name
                if local_copy.is_file() and original != str(local_copy):
                    item["image_path"] = str(local_copy)
                    changed += 1
        return changed

    def documents_compatible_with_pages(self) -> bool:
        project = self._require_project()
        payload = self._read_project_payload(project)
        saved = str(payload.get("document_page_lineage_hash") or "")
        current = self.current_page_lineage_hash()
        return bool(saved and current and saved == current)

    def save_multi_ocr_snapshot(self, payload: Mapping[str, Any]) -> Path:
        project = self._require_project()
        path = project / "artifacts" / "ocr" / "multi" / "latest.json"
        payload_dict = dict(payload)
        payload_ref = ContentAddressedJsonStore(project).put(
            payload_dict, kind="multi_ocr_snapshot"
        )
        wrapper = {
            "schema_version": SCHEMA_VERSION,
            "saved_at": _now_iso(),
            "page_lineage_hash": self.current_page_lineage_hash(),
            "summary": _multi_ocr_snapshot_summary(payload_dict),
            "payload_ref": payload_ref,
        }
        atomic_write_text(path, json.dumps(wrapper, ensure_ascii=False, indent=2) + "\n")
        graph = ArtifactGraphStore(project)
        deps = [node for node in (
            graph.current_node("pages.current"), graph.current_node("pipeline.multi_ocr")
        ) if node]
        graph.register(
            kind="multi_ocr_snapshot", digest=str(payload_ref.get("sha256") or stable_json_hash(payload_dict)),
            alias="ocr.multi.latest", path=self._portable_path(project, path),
            dependencies=deps, metadata={"page_lineage_hash": wrapper["page_lineage_hash"]},
        )
        return path

    def has_multi_ocr_snapshot(self) -> bool:
        """Cheap existence/lineage probe that never resolves the large payload."""
        project = self._require_project()
        path = project / "artifacts" / "ocr" / "multi" / "latest.json"
        if not path.exists():
            return False
        try:
            raw = _require_workspace_schema(
                json.loads(path.read_text(encoding="utf-8")), "artifacts/ocr/multi/latest.json"
            )
        except Exception:
            return False
        saved = str(raw.get("page_lineage_hash") or "")
        current = self.current_page_lineage_hash()
        return bool((not saved or not current or saved == current) and is_object_ref(raw.get("payload_ref")))

    def load_multi_ocr_snapshot(self) -> dict:
        project = self._require_project()
        path = project / "artifacts" / "ocr" / "multi" / "latest.json"
        if not path.exists():
            return {}
        raw = _require_workspace_schema(
            json.loads(path.read_text(encoding="utf-8")), "artifacts/ocr/multi/latest.json"
        )
        saved = str(raw.get("page_lineage_hash") or "")
        current = self.current_page_lineage_hash()
        if saved and current and saved != current:
            return {}
        reference = raw.get("payload_ref")
        if not is_object_ref(reference):
            raise ValueError("当前 multi OCR 快照必须使用 schema 4 payload_ref；不兼容旧内联 payload。")
        try:
            resolved = ContentAddressedJsonStore(project).resolve(reference)
        except Exception as exc:
            raise ValueError(f"当前 multi OCR 对象缺失或损坏：{exc}") from exc
        if not isinstance(resolved, dict):
            raise ValueError("当前 multi OCR 对象不是 JSON 对象。")
        return dict(resolved)

    def save_adjudication_state(self, state: Mapping[str, Any], *, channel: str = "local") -> Path:
        project = self._require_project()
        channel_key = "cloud" if str(channel).lower() == "cloud" else "local"
        path = project / "adjudication" / channel_key / "state.json"
        state_dict = dict(state or {})
        state_ref = ContentAddressedJsonStore(project).put(
            state_dict, kind=f"adjudication:{channel_key}"
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "saved_at": _now_iso(),
            "page_lineage_hash": self.current_page_lineage_hash(),
            "state_ref": state_ref,
        }
        atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        graph = ArtifactGraphStore(project)
        deps = [node for node in (
            graph.current_node("ocr.multi.latest"), graph.current_node("stage.current")
        ) if node]
        graph.register(
            kind="adjudication", digest=str(state_ref.get("sha256") or stable_json_hash(state_dict)),
            alias=f"adjudication.{channel_key}", path=self._portable_path(project, path),
            dependencies=deps, metadata={"channel": channel_key, "page_lineage_hash": payload["page_lineage_hash"]},
        )
        return path

    @staticmethod
    def _repair_adjudication_journal_tail(path: Path) -> None:
        """Repair only an interrupted final JSONL record, never middle corruption.

        The final event may legitimately exceed a small fixed tail window, so
        search backwards in chunks until the preceding newline is found.  This
        keeps repair bounded in memory without guessing at a maximum event size.
        """
        if not path.is_file() or path.stat().st_size <= 0:
            return
        with path.open("r+b") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(size - 1)
            if fh.read(1) == b"\n":
                return

            chunk_size = 64 * 1024
            cursor = size
            last_newline_offset = -1
            while cursor > 0:
                start = max(0, cursor - chunk_size)
                fh.seek(start)
                chunk = fh.read(cursor - start)
                idx = chunk.rfind(b"\n")
                if idx >= 0:
                    last_newline_offset = start + idx
                    break
                cursor = start

            fragment_start = last_newline_offset + 1
            fh.seek(fragment_start)
            fragment = fh.read(size - fragment_start)
            try:
                parsed = json.loads(fragment.decode("utf-8"))
            except Exception:
                # Only the final interrupted record may be discarded. If no
                # newline exists, the file consists solely of that partial row.
                fh.truncate(fragment_start)
                fh.flush()
                return
            if not isinstance(parsed, dict):
                raise ValueError("人工裁决 journal 最后一条完整记录不是 JSON 对象。")
            fh.seek(0, os.SEEK_END)
            fh.write(b"\n")
            fh.flush()

    def append_adjudication_event(self, event: Mapping[str, Any]) -> Path:
        """Append one tiny crash-recoverable adjudication event.

        Thousands of manual decisions must never rewrite the 50-100 MB OCR
        snapshot.  The immutable OCR snapshot is the base state; this journal is
        the mutable overlay.  One decision therefore costs one JSON line and no
        ArtifactGraph traversal.
        """
        project = self._require_project()
        path = project / ADJUDICATION_EVENT_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        journal_key = str(path.resolve())
        if journal_key not in self._checked_adjudication_journals:
            self._repair_adjudication_journal_tail(path)
            self._checked_adjudication_journals.add(journal_key)
        payload = dict(event or {})
        payload.setdefault("timestamp", _now_iso())
        payload.setdefault("schema_version", ADJUDICATION_EVENT_SCHEMA_VERSION)
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        # A checkpoint worker can compact the journal concurrently with manual
        # review.  Serialize the tiny append/compaction critical sections while
        # leaving the expensive checkpoint compression outside the lock.
        with _adjudication_journal_lock(path):
            # Open/append/close intentionally flushes the Python and libc buffers
            # on every decision while avoiding a full-project fsync.
            with path.open("a", encoding="utf-8", newline="") as fh:
                fh.write(line)
                fh.flush()
        return path

    def load_adjudication_events(self) -> list[dict]:
        project = self._require_project()
        path = project / ADJUDICATION_EVENT_FILE
        if not path.exists():
            return []
        events: list[dict] = []
        raw_lines = path.read_text(encoding="utf-8").splitlines()
        nonempty = [(index, line.strip()) for index, line in enumerate(raw_lines) if line.strip()]
        final_nonempty_index = nonempty[-1][0] if nonempty else -1
        for line_index, line in nonempty:
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                if line_index == final_nonempty_index:
                    # Only a final interrupted write is recoverable. Middle
                    # corruption must never be silently reinterpreted or skipped.
                    continue
                raise ValueError(
                    f"人工裁决 journal 第 {line_index + 1} 行损坏；为避免静默丢裁决已停止恢复。"
                ) from exc
            if not isinstance(item, dict):
                raise ValueError(f"人工裁决 journal 第 {line_index + 1} 行不是 JSON 对象。")
            version = int(item.get("schema_version") or 0)
            if version != ADJUDICATION_EVENT_SCHEMA_VERSION:
                raise ValueError(
                    f"人工裁决事件 schema={version} 不受当前开发版支持；"
                    f"当前只接受 {ADJUDICATION_EVENT_SCHEMA_VERSION}。"
                )
            events.append(item)
        return events

    def adjudication_event_checkpoint_marker(self) -> dict[str, Any]:
        """Return a stable prefix marker for the current adjudication journal.

        The marker is captured on the GUI thread together with the OCR snapshot.
        A background checkpoint may later remove exactly this prefix while
        preserving decisions appended after the snapshot was captured.
        """
        project = self._require_project()
        path = project / ADJUDICATION_EVENT_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with _adjudication_journal_lock(path):
            raw = path.read_bytes() if path.exists() else b""
        return {
            "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    def compact_adjudication_events_through(self, marker: Mapping[str, Any] | None) -> bool:
        """Drop only journal bytes already represented by a saved checkpoint.

        New decisions may be appended while the 35 MB checkpoint is compressed.
        Under the shared per-path lock we verify that the saved prefix is still
        byte-identical, then atomically replace the file with only the suffix.
        A mismatch is conservative: nothing is deleted.
        """
        if not isinstance(marker, Mapping):
            return False
        try:
            prefix_size = max(0, int(marker.get("size") or 0))
        except (TypeError, ValueError, OverflowError):
            return False
        expected = str(marker.get("sha256") or "")
        project = self._require_project()
        path = project / ADJUDICATION_EVENT_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with _adjudication_journal_lock(path):
            raw = path.read_bytes() if path.exists() else b""
            if prefix_size > len(raw):
                return False
            prefix = raw[:prefix_size]
            if hashlib.sha256(prefix).hexdigest() != expected:
                return False
            atomic_write_bytes(path, raw[prefix_size:])
            self._checked_adjudication_journals.discard(str(path.resolve()))
        return True

    def clear_adjudication_events(self) -> None:
        project = self._require_project()
        path = project / ADJUDICATION_EVENT_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with _adjudication_journal_lock(path):
            atomic_write_text(path, "")
            self._checked_adjudication_journals.discard(str(path.resolve()))

    def load_adjudication_state(self, *, channel: str = "local") -> dict:
        project = self._require_project()
        channel_key = "cloud" if str(channel).lower() == "cloud" else "local"
        path = project / "adjudication" / channel_key / "state.json"
        if not path.exists():
            return {}
        raw = _require_workspace_schema(
            json.loads(path.read_text(encoding="utf-8")), f"adjudication/{channel_key}/state.json"
        )
        saved = str(raw.get("page_lineage_hash") or "")
        current = self.current_page_lineage_hash()
        if saved and current and saved != current:
            return {}
        reference = raw.get("state_ref")
        if not is_object_ref(reference):
            raise ValueError("当前裁决状态必须使用 schema 4 state_ref；不兼容旧内联 state。")
        try:
            resolved = ContentAddressedJsonStore(project).resolve(reference)
        except Exception as exc:
            raise ValueError(f"当前裁决状态对象缺失或损坏：{exc}") from exc
        if not isinstance(resolved, dict):
            raise ValueError("当前裁决状态对象不是 JSON 对象。")
        return dict(resolved)

    def record_run(self, stage: str, *, status: str = "ok", details: Mapping[str, Any] | None = None) -> Path:
        """Record an already-completed lightweight run.

        Older callers use this convenience API.  Long-running jobs should use
        ``begin_run`` + ``finish_run`` so one run id survives success, stop and
        failure and can carry text/performance diagnostics without bloating the
        history index.
        """
        run_id = self.begin_run(stage, details=details)
        return self.finish_run(run_id, status=status, details=details)

    def begin_run(
        self,
        stage: str,
        *,
        details: Mapping[str, Any] | None = None,
        run_id: str | None = None,
    ) -> str:
        """Create a crash-visible running record and return its stable run id."""
        project = self._require_project()
        context = self.project_context()
        run_id = str(run_id or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}")
        now = _now_iso()
        payload = {
            "schema_version": RUN_LOG_SCHEMA_VERSION,
            "format": RUN_LOG_FORMAT,
            "run_id": run_id,
            "timestamp": now,
            "started_at": now,
            "finished_at": "",
            "duration_seconds": 0.0,
            "stage": str(stage),
            "status": "running",
            "project_id": str(context.get("project_id") or ""),
            "project_name": str(context.get("name") or project.name),
            "page_lineage_hash": self.current_page_lineage_hash(),
            "details": dict(details or {}),
            "error": {},
            "artifacts": {},
        }
        self._write_run_payload(project, payload)
        self.rebuild_run_history_index()
        self._update_last_run(project, payload)
        return run_id

    def finish_run(
        self,
        run_id: str,
        *,
        status: str = "ok",
        details: Mapping[str, Any] | None = None,
        error: str | Mapping[str, Any] | None = None,
        log_text: str | None = None,
        performance: Mapping[str, Any] | None = None,
        artifacts: Mapping[str, Any] | None = None,
    ) -> Path:
        """Finalize a run atomically and persist optional human/perf diagnostics."""
        project = self._require_project()
        path = project / "logs" / "runs" / f"{run_id}.json"
        if path.exists():
            try:
                payload = _require_run_log_schema(
                    json.loads(path.read_text(encoding="utf-8")), f"logs/runs/{path.name}"
                )
            except Exception as exc:
                raise ValueError(f"当前 run log 损坏或版本不匹配：{exc}") from exc
        else:
            payload = {}
        now = _now_iso()
        started_at = str(payload.get("started_at") or payload.get("timestamp") or now)
        payload.update({
            "schema_version": RUN_LOG_SCHEMA_VERSION,
            "format": RUN_LOG_FORMAT,
            "run_id": str(run_id),
            "timestamp": str(payload.get("timestamp") or started_at),
            "started_at": started_at,
            "finished_at": now,
            "stage": str(payload.get("stage") or "unknown"),
            "status": self._normalize_run_status(status),
            "page_lineage_hash": str(payload.get("page_lineage_hash") or self.current_page_lineage_hash()),
        })
        merged_details = dict(payload.get("details") or {})
        merged_details.update(dict(details or {}))
        payload["details"] = merged_details
        payload["duration_seconds"] = self._run_duration_seconds(started_at, now)

        if error:
            if isinstance(error, Mapping):
                error_payload = dict(error)
            else:
                error_text = self._redact_log_text(str(error))
                error_payload = {
                    "message": error_text.splitlines()[-1] if error_text.splitlines() else error_text,
                    "traceback": error_text,
                }
            payload["error"] = error_payload
        elif payload["status"] != "error":
            payload["error"] = {}

        run_artifacts = dict(payload.get("artifacts") or {})
        run_artifacts.update(dict(artifacts or {}))
        if log_text is not None:
            log_path = project / "logs" / "runs" / f"{run_id}.log"
            atomic_write_text(log_path, self._redact_log_text(log_text).rstrip() + "\n")
            run_artifacts["text_log"] = log_path.name
        if performance is not None:
            perf_path = project / "logs" / "runs" / f"{run_id}.performance.json"
            atomic_write_text(
                perf_path,
                json.dumps(dict(performance), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
            run_artifacts["performance"] = perf_path.name
            # Prefer the performance trace wall time when available: it is
            # monotonic and therefore immune to wall-clock jumps/timezone edits.
            try:
                perf_elapsed = float(performance.get("elapsed_seconds", 0.0) or 0.0)
            except Exception:
                perf_elapsed = 0.0
            if perf_elapsed > 0:
                payload["duration_seconds"] = round(perf_elapsed, 6)
        payload["artifacts"] = run_artifacts
        self._write_run_payload(project, payload)
        self.rebuild_run_history_index()
        self._update_last_run(project, payload)
        return path

    def list_run_history(
        self,
        *,
        limit: int | None = 300,
        stages: Iterable[str] | None = None,
        statuses: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return newest-first run summaries, repairing a broken index on demand."""
        project = self._require_project()
        wanted_stages = {str(v) for v in (stages or []) if str(v)}
        wanted_statuses = {self._normalize_run_status(v) for v in (statuses or []) if str(v)}
        index_path = project / "logs" / "runs" / "history.jsonl"
        rows: list[dict[str, Any]] = []
        invalid = False
        if index_path.exists():
            for line in index_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    invalid = True
                    continue
                if isinstance(row, dict) and row.get("run_id"):
                    rows.append(row)
        if invalid:
            self.rebuild_run_history_index()
            return self.list_run_history(limit=limit, stages=stages, statuses=statuses)
        if wanted_stages:
            rows = [row for row in rows if str(row.get("stage") or "") in wanted_stages]
        if wanted_statuses:
            rows = [row for row in rows if self._normalize_run_status(row.get("status")) in wanted_statuses]
        rows.sort(key=lambda item: (str(item.get("started_at") or item.get("timestamp") or ""), str(item.get("run_id") or "")), reverse=True)
        if limit is not None and int(limit) >= 0:
            rows = rows[: int(limit)]
        return rows

    def load_run(self, run_id: str) -> dict[str, Any]:
        project = self._require_project()
        path = project / "logs" / "runs" / f"{_safe_component(run_id, 'run')}.json"
        if not path.exists():
            return {}
        raw = _require_run_log_schema(
            json.loads(path.read_text(encoding="utf-8")), f"logs/runs/{path.name}"
        )
        return dict(raw)

    def load_run_text_log(self, run_id: str) -> str:
        project = self._require_project()
        path = project / "logs" / "runs" / f"{_safe_component(run_id, 'run')}.log"
        return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""

    def export_run_history(self, target: str | Path) -> Path:
        """Export all current run summaries as CSV or JSON without source text."""
        target = Path(target).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = self.list_run_history(limit=None)
        if target.suffix.lower() == ".json":
            atomic_write_text(target, json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
            return target
        if target.suffix.lower() != ".csv":
            target = target.with_suffix(".csv")
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(["run_id", "started_at", "finished_at", "stage", "status", "duration_seconds", "summary"])
        for row in rows:
            writer.writerow([
                row.get("run_id", ""), row.get("started_at", row.get("timestamp", "")),
                row.get("finished_at", ""), row.get("stage", ""), row.get("status", ""),
                row.get("duration_seconds", 0), row.get("summary", ""),
            ])
        atomic_write_text(target, "\ufeff" + out.getvalue())
        return target

    def prune_run_history(self, *, keep_recent: int = 200) -> dict[str, int]:
        """Remove old run diagnostics only; project OCR/adjudication artifacts stay intact."""
        project = self._require_project()
        rows = self.list_run_history(limit=None)
        keep_recent = max(0, int(keep_recent))
        remove = rows[keep_recent:]
        removed_files = 0
        for row in remove:
            run_id = _safe_component(str(row.get("run_id") or ""), "run")
            for suffix in (".json", ".log", ".performance.json"):
                path = project / "logs" / "runs" / f"{run_id}{suffix}"
                if path.exists():
                    path.unlink(missing_ok=True)
                    removed_files += 1
        self.rebuild_run_history_index()
        return {"runs": len(remove), "files": removed_files, "kept": min(len(rows), keep_recent)}

    def rebuild_run_history_index(self) -> Path:
        """Rebuild the JSONL run index from authoritative per-run JSON files."""
        project = self._require_project()
        runs_dir = project / "logs" / "runs"
        rows: list[dict[str, Any]] = []
        for path in sorted(runs_dir.glob("*.json")):
            if path.name.endswith(".performance.json"):
                continue
            try:
                raw = _require_run_log_schema(
                    json.loads(path.read_text(encoding="utf-8")), f"logs/runs/{path.name}"
                )
            except Exception as exc:
                raise ValueError(f"当前 run log 损坏或版本不匹配：{exc}") from exc
            if raw.get("run_id"):
                rows.append(self._run_history_summary(raw))
        rows.sort(key=lambda item: (str(item.get("started_at") or item.get("timestamp") or ""), str(item.get("run_id") or "")))
        text = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
        target = runs_dir / "history.jsonl"
        atomic_write_text(target, text)
        return target

    def audit_project(self, *, deep: bool = False) -> dict[str, Any]:
        """Return a non-destructive project integrity report.

        ``deep=True`` verifies SHA-256 hashes for imported source files and current
        page content.  Normal UI checks stay fast and only verify structure/paths.
        """
        project = self._require_project()
        issues: list[str] = []
        warnings: list[str] = []
        payload = self._read_project_payload(project)
        schema = int(payload.get("schema_version") or 0)
        if schema != SCHEMA_VERSION:
            issues.append(f"project.json schema={schema}，当前需要 {SCHEMA_VERSION}")
        if str(payload.get("format") or "") != PROJECT_FORMAT:
            issues.append("project.json format 标记与当前开发版不一致")

        state_path = project / PAGE_STATE_FILE
        page_count = 0
        missing_pages: list[str] = []
        page_lineage_matches = True
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                page_values = list(state.get("page_images") or [])
                page_count = len(page_values)
                for value in page_values:
                    resolved = self._resolve_portable(project, value)
                    if not resolved.is_file():
                        missing_pages.append(str(value))
                if deep and not missing_pages and state.get("page_fingerprints"):
                    actual = self._page_fingerprints(
                        project, [self._resolve_portable(project, value) for value in page_values]
                    )
                    actual_hash = self.page_lineage_hash_from_state({"page_fingerprints": actual})
                    saved_hash = str(state.get("page_lineage_hash") or "")
                    page_lineage_matches = not saved_hash or actual_hash == saved_hash
                    if not page_lineage_matches:
                        issues.append("页面内容指纹已变化；旧 OCR/裁决结果应视为不兼容")
            except Exception as exc:
                issues.append(f"pages/state.json 无法读取：{exc}")
        if missing_pages:
            issues.append(f"缺失页面文件 {len(missing_pages)} 个")

        missing_sources: list[str] = []
        changed_sources: list[str] = []
        for item in payload.get("source_imports") or []:
            if not isinstance(item, dict):
                continue
            rel = str(item.get("project_path") or "")
            if not rel:
                continue
            source = project / rel
            if not source.exists():
                missing_sources.append(rel)
                continue
            expected = str(item.get("sha256") or "")
            if deep and expected and source.is_file():
                try:
                    if _sha256_file(source) != expected:
                        changed_sources.append(rel)
                except OSError:
                    changed_sources.append(rel)
        if missing_sources:
            issues.append(f"缺失源文件/文件夹 {len(missing_sources)} 个")
        if changed_sources:
            warnings.append(f"有 {len(changed_sources)} 个源文件内容与导入时 SHA-256 不同")

        current_stage = project / "artifacts" / "stages" / "current.json"
        stage_compatible: bool | None = None
        if current_stage.exists():
            try:
                stage_raw = json.loads(current_stage.read_text(encoding="utf-8"))
                # A compact stage stores the large document in revisions/objects;
                # audit only needs the lightweight lineage metadata here.
                saved = str(stage_raw.get("page_lineage_hash") or "")
                current = self.current_page_lineage_hash()
                stage_compatible = bool(not saved or not current or saved == current)
                if stage_compatible is False:
                    warnings.append("当前正文阶段与页面 lineage 不兼容，将不会自动恢复")
            except Exception as exc:
                issues.append(f"当前正文阶段无法读取：{exc}")

        history_path = project / "logs" / "runs" / "history.jsonl"
        history_invalid_lines = 0
        run_count = 0
        running_runs = 0
        error_runs = 0
        if history_path.exists():
            try:
                for line in history_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        if isinstance(row, dict) and row.get("run_id"):
                            run_count += 1
                            status = self._normalize_run_status(row.get("status"))
                            if status == "running":
                                running_runs += 1
                            elif status == "error":
                                error_runs += 1
                    except Exception:
                        history_invalid_lines += 1
            except OSError as exc:
                warnings.append(f"运行历史索引无法读取：{exc}")
        if history_invalid_lines:
            warnings.append(f"运行历史索引有 {history_invalid_lines} 行损坏，可重建")
        if running_runs:
            warnings.append(f"有 {running_runs} 条运行记录仍为 running；可能是上次异常退出，可在项目日志中查看")

        graph_raw = ArtifactGraphStore(project).load()
        artifact_nodes = len(dict(graph_raw.get("nodes") or {}))
        cache_entries = len(list((project / "artifacts" / "cache").rglob("*.json")))
        checkpoint_entries = len(list((project / "artifacts" / "checkpoints").rglob("*.json")))
        current_page_node = str(dict(graph_raw.get("current") or {}).get("pages.current") or "")
        if current_page_node and current_page_node not in dict(graph_raw.get("nodes") or {}):
            warnings.append("Artifact Graph 的 pages.current 指针损坏，可通过重新保存页面状态修复")

        return {
            "ok": not issues,
            "schema_version": schema,
            "project_id": str(payload.get("project_id") or ""),
            "project_name": str(payload.get("name") or project.name),
            "page_count": page_count,
            "missing_pages": missing_pages,
            "missing_sources": missing_sources,
            "changed_sources": changed_sources,
            "page_lineage_matches": page_lineage_matches,
            "stage_compatible": stage_compatible,
            "history_invalid_lines": history_invalid_lines,
            "run_count": run_count,
            "running_runs": running_runs,
            "error_runs": error_runs,
            "artifact_nodes": artifact_nodes,
            "cache_entries": cache_entries,
            "checkpoint_entries": checkpoint_entries,
            "issues": issues,
            "warnings": warnings,
        }

    def find_orphan_page_files(self) -> list[Path]:
        """List durable page files not referenced by the current page state."""
        project = self._require_project()
        referenced: set[Path] = set()
        state_path = project / PAGE_STATE_FILE
        if state_path.exists():
            try:
                raw = json.loads(state_path.read_text(encoding="utf-8"))
                for value in raw.get("page_images", []):
                    referenced.add(self._resolve_portable(project, value))
            except Exception:
                return []
        orphans: list[Path] = []
        for bucket in (project / "pages" / "source", project / "pages" / "processed"):
            for path in bucket.rglob("*") if bucket.exists() else ():
                if path.is_file() and path.resolve() not in referenced:
                    orphans.append(path.resolve())
        return sorted(orphans)

    def cleanup_orphan_page_files(self) -> list[Path]:
        """Move orphan durable pages to ``recovery/orphan-pages`` instead of deleting them."""
        project = self._require_project()
        orphans = self.find_orphan_page_files()
        if not orphans:
            return []
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        recovery = project / "recovery" / "orphan-pages" / stamp
        moved: list[Path] = []
        for source in orphans:
            try:
                rel = source.relative_to(project / "pages")
            except ValueError:
                continue
            target = recovery / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            moved.append(target)
        self.record_run("workspace_cleanup", details={"orphan_pages_moved": len(moved)})
        return moved

    @property
    def epub_export_dir(self) -> Path | None:
        if self.active_project is None:
            return None
        path = self.active_project / "exports" / "epub"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def package_export_dir(self) -> Path | None:
        if self.active_project is None:
            return None
        path = self.active_project / "exports" / "packages"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def preserve_exported_file(self, source: str | Path, *, category: str = "packages") -> Path:
        """Keep an exported artifact inside the active project as a durable copy.

        User-facing exports may be saved anywhere.  The project itself must still
        remain self-contained, so a byte-identical copy is retained under
        ``exports/<category>`` and registered through its portable project path.
        Identical content is deduplicated; a same-name/different-content export
        receives a short SHA suffix instead of overwriting history.
        """
        project = self._require_project()
        src = Path(source).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(str(src))
        safe_category = _safe_component(str(category or "packages"), "packages")
        dest_dir = project / "exports" / safe_category
        dest_dir.mkdir(parents=True, exist_ok=True)
        digest = _sha256_file(src)
        target = dest_dir / _safe_component(src.name, "export.bin")
        try:
            # If the user already saved the artifact anywhere inside this project,
            # it is already durable. Do not copy it again into exports/<category>.
            src.relative_to(project.resolve())
            return src
        except ValueError:
            pass
        except Exception:
            pass
        if target.exists():
            if _sha256_file(target) == digest:
                return target
            target = target.with_name(f"{target.stem}-{digest[:8]}{target.suffix}")
            if target.exists() and _sha256_file(target) == digest:
                return target
        fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(dest_dir))
        os.close(fd)
        temp = Path(temp_name)
        try:
            shutil.copy2(src, temp)
            if _sha256_file(temp) != digest:
                raise IOError(f"项目导出副本 SHA-256 校验失败：{src}")
            os.replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)
        return target

    @property
    def run_log_dir(self) -> Path | None:
        if self.active_project is None:
            return None
        path = self.active_project / "logs" / "runs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    # artifact graph / deterministic cache / checkpoint resume
    # ------------------------------------------------------------------
    def artifact_graph(self) -> dict[str, Any]:
        project = self._require_project()
        return ArtifactGraphStore(project).load()

    def artifact_lineage(self, node_id: str) -> list[str]:
        project = self._require_project()
        return ArtifactGraphStore(project).lineage(node_id)

    def artifact_descendants(self, node_id: str) -> list[str]:
        project = self._require_project()
        return ArtifactGraphStore(project).descendants(node_id)

    def register_artifact(
        self, *, kind: str, digest: str, alias: str = "", path: str | Path = "",
        dependencies: Iterable[str] = (), metadata: Mapping[str, Any] | None = None,
    ) -> str:
        project = self._require_project()
        portable = self._portable_path(project, Path(path)) if path else ""
        return ArtifactGraphStore(project).register(
            kind=kind, digest=digest, alias=alias, path=portable,
            dependencies=dependencies, metadata=metadata,
        )

    def register_file_artifact(
        self, *, kind: str, path: str | Path, alias: str = "",
        dependencies: Iterable[str] = (), metadata: Mapping[str, Any] | None = None,
    ) -> str:
        project = self._require_project()
        target = Path(path).expanduser().resolve()
        if not target.is_file():
            raise FileNotFoundError(str(target))
        digest = _sha256_file(target)
        return ArtifactGraphStore(project).register(
            kind=str(kind), digest=digest, alias=str(alias or ""),
            path=self._portable_path(project, target),
            dependencies=dependencies, metadata={
                **dict(metadata or {}),
                "size": int(target.stat().st_size),
                "sha256": digest,
            },
        )

    def make_stage_cache_key(
        self, stage: str, *, config: Any, inputs: Any = None,
        implementation: str = "",
    ) -> str:
        project = self._require_project()
        return StageCacheStore(project).make_key(
            stage=str(stage),
            page_lineage_hash=self.current_page_lineage_hash(),
            config=config, inputs=inputs, implementation=implementation,
        )

    @staticmethod
    def implementation_fingerprint(paths: Iterable[str | Path]) -> str:
        return implementation_fingerprint(paths)

    _CACHE_PROJECT_URI = "project://"
    _CACHE_EVIDENCE_KEYS = {
        "ocr_review_sentence_image_path",
        "ocr_review_preferred_image_path",
    }

    def _cache_portable_payload(self, project: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Make cached OCR documents relocatable and preserve transient review evidence.

        OCR sentence evidence is normally created under the per-run temporary
        directory.  A text-only cache hit after restart would otherwise point at
        deleted images and break image/text review.  Known evidence files are
        copied into a content-addressed project cache and all project-local paths
        are stored with a portable ``project://`` URI so project backup/restore can
        move to a different workspace without invalidating the cached document.
        """
        project = project.resolve()
        asset_root = project / "artifacts" / "cache_assets"

        def portable_string(value: str, field_name: str = "") -> str:
            text = str(value or "")
            if not text or text.startswith(self._CACHE_PROJECT_URI):
                return text
            try:
                candidate = Path(text).expanduser()
                if not candidate.is_absolute():
                    return text
                candidate = candidate.resolve()
            except Exception:
                return text

            # Existing project-local files can simply become portable URIs.
            try:
                relative = candidate.relative_to(project)
                return self._CACHE_PROJECT_URI + relative.as_posix()
            except ValueError:
                pass

            # Only known OCR review evidence is copied from session-temp paths.
            if field_name not in self._CACHE_EVIDENCE_KEYS or not candidate.is_file():
                return text
            try:
                digest = _sha256_file(candidate)
                suffix = candidate.suffix.lower() or ".bin"
                target = asset_root / digest[:2] / f"{digest}{suffix}"
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    temp = target.with_name(target.name + f".{uuid.uuid4().hex}.tmp")
                    shutil.copy2(candidate, temp)
                    os.replace(temp, target)
                relative = target.relative_to(project)
                return self._CACHE_PROJECT_URI + relative.as_posix()
            except Exception:
                # Cache persistence is an optimization: never break OCR because a
                # best-effort evidence copy failed.  The original value remains.
                return text

        def walk(value: Any, field_name: str = "") -> Any:
            if isinstance(value, dict):
                return {str(k): walk(v, str(k)) for k, v in value.items()}
            if isinstance(value, list):
                return [walk(v, field_name) for v in value]
            if isinstance(value, tuple):
                return [walk(v, field_name) for v in value]
            if isinstance(value, str):
                return portable_string(value, field_name)
            return value

        result = walk(dict(payload))
        return result if isinstance(result, dict) else dict(payload)

    def _cache_resolve_payload(self, project: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
        project = project.resolve()

        def walk(value: Any) -> Any:
            if isinstance(value, dict):
                return {str(k): walk(v) for k, v in value.items()}
            if isinstance(value, list):
                return [walk(v) for v in value]
            if isinstance(value, str) and value.startswith(self._CACHE_PROJECT_URI):
                relative = value[len(self._CACHE_PROJECT_URI):].lstrip("/")
                try:
                    candidate = (project / relative).resolve()
                    candidate.relative_to(project)
                    return str(candidate)
                except Exception:
                    return value
            return value

        result = walk(dict(payload))
        return result if isinstance(result, dict) else dict(payload)

    def load_stage_cache(self, stage: str, key: str) -> dict[str, Any] | None:
        project = self._require_project()
        payload = StageCacheStore(project).load(
            stage=str(stage), key=str(key),
            page_lineage_hash=self.current_page_lineage_hash(),
        )
        if payload is None:
            return None
        return self._cache_resolve_payload(project, payload)

    def save_stage_cache(
        self, stage: str, key: str, payload: Mapping[str, Any], *,
        metadata: Mapping[str, Any] | None = None,
        dependencies: Iterable[str] = (),
    ) -> Path:
        project = self._require_project()
        lineage = self.current_page_lineage_hash()
        portable_payload = self._cache_portable_payload(project, payload)
        path = StageCacheStore(project).save(
            stage=str(stage), key=str(key), page_lineage_hash=lineage,
            payload=portable_payload, metadata=metadata,
        )
        graph = ArtifactGraphStore(project)
        deps = [str(v) for v in dependencies if str(v)]
        page_node = graph.current_node("pages.current")
        if page_node and page_node not in deps:
            deps.append(page_node)
        graph.register(
            kind=f"cache.{stage}", digest=str(key),
            alias=f"cache.{stage}.latest", path=self._portable_path(project, path),
            dependencies=deps, metadata=dict(metadata or {}),
        )
        return path

    def prune_stage_cache(self, *, keep_recent: int = 300) -> int:
        project = self._require_project()
        removed = StageCacheStore(project).prune(keep_recent=keep_recent)
        # Stage-cache payloads are immutable shared objects.  Prune only objects
        # no longer referenced by any live stage/snapshot/adjudication/cache ref.
        self.prune_unreferenced_json_objects()
        return removed

    def begin_checkpoint(
        self, stage: str, signature: str, *, metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        project = self._require_project()
        return CheckpointStore(project).begin(
            stage=str(stage), signature=str(signature),
            page_lineage_hash=self.current_page_lineage_hash(),
            metadata=metadata,
        )

    def load_checkpoint(self, stage: str, signature: str) -> dict[str, Any]:
        project = self._require_project()
        raw = CheckpointStore(project).load(stage=str(stage), signature=str(signature))
        if not raw:
            return {}
        saved = str(raw.get("page_lineage_hash") or "")
        current = self.current_page_lineage_hash()
        if saved and current and saved != current:
            return {}
        return raw

    def mark_checkpoint_step(
        self, stage: str, signature: str, step: str, cache_key: str, *,
        reused: bool = False, detail: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        project = self._require_project()
        return CheckpointStore(project).mark(
            stage=str(stage), signature=str(signature), step=str(step),
            cache_key=str(cache_key), reused=reused, detail=detail,
        )

    def finish_checkpoint(self, stage: str, signature: str, *, status: str = "ok") -> dict[str, Any]:
        project = self._require_project()
        return CheckpointStore(project).finish(
            stage=str(stage), signature=str(signature), status=str(status),
        )

    # ------------------------------------------------------------------
    # run log internals
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_run_status(value: Any) -> str:
        text = str(value or "ok").strip().lower()
        aliases = {
            "success": "ok", "done": "ok", "completed": "ok",
            "failed": "error", "failure": "error", "exception": "error",
            "stopped": "cancelled", "canceled": "cancelled", "cancel": "cancelled",
            "warn": "warning",
        }
        text = aliases.get(text, text)
        return text if text in RUN_STATUSES else "warning"

    @staticmethod
    def _run_duration_seconds(started_at: str, finished_at: str) -> float:
        try:
            start = datetime.fromisoformat(str(started_at))
            finish = datetime.fromisoformat(str(finished_at))
            return round(max(0.0, (finish - start).total_seconds()), 6)
        except Exception:
            return 0.0

    @staticmethod
    def _redact_log_text(value: str) -> str:
        """Remove common credential spellings before logs become portable backups."""
        text = str(value or "")
        patterns = (
            (r"(?i)(api[_ -]?key\s*[:=]\s*)([^\s,;]+)", r"\1***REDACTED***"),
            (r"(?i)(authorization\s*[:=]\s*bearer\s+)([^\s]+)", r"\1***REDACTED***"),
            (r"(?i)(access[_ -]?token\s*[:=]\s*)([^\s,;]+)", r"\1***REDACTED***"),
            (r"(?i)(secret\s*[:=]\s*)([^\s,;]+)", r"\1***REDACTED***"),
        )
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text)
        return text

    def _write_run_payload(self, project: Path, payload: Mapping[str, Any]) -> Path:
        run_id = _safe_component(str(payload.get("run_id") or "run"), "run")
        path = project / "logs" / "runs" / f"{run_id}.json"
        atomic_write_text(path, json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n")
        return path

    @staticmethod
    def _run_summary_text(payload: Mapping[str, Any]) -> str:
        details = dict(payload.get("details") or {})
        if details.get("summary"):
            return str(details.get("summary"))[:300]
        stage = str(payload.get("stage") or "")
        pieces: list[str] = []
        if stage in {"ocr", "single_ocr", "multi_ocr"}:
            engines = details.get("engines") or details.get("models") or []
            if isinstance(engines, (list, tuple)) and engines:
                pieces.append(" / ".join(str(v) for v in engines[:6]))
            pages = details.get("pages")
            if pages is not None:
                pieces.append(f"{pages} 页")
            conflicts = details.get("conflicts")
            if conflicts is not None:
                pieces.append(f"分歧 {conflicts}")
        elif stage == "epub_export":
            chapters = details.get("chapters")
            if chapters is not None:
                pieces.append(f"{chapters} 章")
        elif stage == "package_export":
            kind = str(details.get("kind") or "")
            if kind:
                pieces.append(kind)
            for key, label in (("editable_items", "条目"), ("editable_conflict_rows", "分歧"), ("pending_review_rows", "待审"), ("row_count", "正文")):
                if details.get(key) is not None:
                    pieces.append(f"{label} {details.get(key)}")
        elif stage == "cloud_adjudication_import":
            if details.get("accepted") is not None:
                pieces.append(f"接受 {details.get('accepted')}")
            if details.get("unresolved") is not None:
                pieces.append(f"未决 {details.get('unresolved')}")
        elif details.get("blocks") is not None:
            pieces.append(f"{details.get('blocks')} 块")
        return " · ".join(pieces)[:300]

    def _run_history_summary(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        raw_artifacts = payload.get("artifacts")
        artifacts = dict(raw_artifacts) if isinstance(raw_artifacts, Mapping) else {}
        raw_error = payload.get("error")
        error = dict(raw_error) if isinstance(raw_error, Mapping) else {"message": str(raw_error or "")}
        try:
            duration = max(0.0, float(payload.get("duration_seconds", 0.0) or 0.0))
        except Exception:
            duration = 0.0
        return {
            "schema_version": RUN_LOG_SCHEMA_VERSION,
            "run_id": str(payload.get("run_id") or ""),
            "timestamp": str(payload.get("timestamp") or payload.get("started_at") or ""),
            "started_at": str(payload.get("started_at") or payload.get("timestamp") or ""),
            "finished_at": str(payload.get("finished_at") or ""),
            "stage": str(payload.get("stage") or ""),
            "status": self._normalize_run_status(payload.get("status")),
            "duration_seconds": round(duration, 6),
            "summary": self._run_summary_text(payload),
            "error_message": str(error.get("message") or "")[:300],
            "has_text_log": bool(artifacts.get("text_log")),
            "has_performance": bool(artifacts.get("performance")),
            "page_lineage_hash": str(payload.get("page_lineage_hash") or ""),
        }

    def _update_last_run(self, project: Path, payload: Mapping[str, Any]) -> None:
        project_payload = self._read_project_payload(project)
        project_payload["last_run"] = {
            "run_id": str(payload.get("run_id") or ""),
            "stage": str(payload.get("stage") or ""),
            "status": self._normalize_run_status(payload.get("status")),
            "finished_at": str(payload.get("finished_at") or ""),
        }
        self._touch_project_payload(project, project_payload)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _require_project(self) -> Path:
        if self.active_project is None:
            raise RuntimeError("当前没有活动项目")
        project = self.active_project.expanduser().resolve()
        self._ensure_layout(project)
        return project

    def _ensure_layout(self, project: Path) -> None:
        project.mkdir(parents=True, exist_ok=True)
        for rel in self.LAYOUT:
            (project / rel).mkdir(parents=True, exist_ok=True)

    def _migrate_project_if_needed(self, project: Path) -> None:
        """Compatibility interface retained for the future stable release.

        No legacy migration code is shipped during the pre-stable cycle.  Current
        projects are a no-op; every older/different workspace contract fails
        explicitly through this stable hook.
        """
        marker = Path(project) / PROJECT_FILE
        if not marker.is_file():
            raise LegacyWorkspaceCompatibilityUnavailable(
                f"旧工作区兼容接口暂不可用：项目缺少 {PROJECT_FILE}。"
            )
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except Exception as exc:
            raise LegacyWorkspaceCompatibilityUnavailable(
                f"旧工作区兼容接口暂不可用，且无法读取项目清单：{exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise LegacyWorkspaceCompatibilityUnavailable(
                "旧工作区兼容接口暂不可用：project.json 顶层不是对象。"
            )
        version = int(payload.get("schema_version") or 0)
        fmt = str(payload.get("format") or "")
        if version == SCHEMA_VERSION and fmt == PROJECT_FORMAT:
            return
        raise LegacyWorkspaceCompatibilityUnavailable(
            f"旧工作区兼容接口已保留但开发版未启用迁移实现："
            f"schema={version}, format={fmt or '<missing>'}；"
            f"当前只接受 schema {SCHEMA_VERSION} / {PROJECT_FORMAT}。"
        )

    def _validate_current_project(self, project: Path) -> dict:
        """Require the exact current workspace contract.

        During the development cycle we intentionally do not migrate older project
        layouts.  Silent migration made OCR lineage and adjudication provenance hard
        to reason about.  A project created by another schema must be opened with the
        build that created it or explicitly re-imported after the stable format ships.
        """
        marker = project / PROJECT_FILE
        if not marker.is_file():
            raise ValueError(f"项目缺少 {PROJECT_FILE}：{project}")
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"无法读取项目清单：{exc}") from exc
        if not isinstance(payload, dict) or not str(payload.get("project_id") or ""):
            raise ValueError("project.json 缺少有效 project_id。")
        version = int(payload.get("schema_version") or 0)
        fmt = str(payload.get("format") or "")
        if version != SCHEMA_VERSION or fmt != PROJECT_FORMAT:
            # Keep one stable seam for future compatibility work; there is no
            # migration implementation in pre-stable builds.
            self._migrate_project_if_needed(project)
            payload = json.loads(marker.read_text(encoding="utf-8"))
            version = int(payload.get("schema_version") or 0)
            fmt = str(payload.get("format") or "")
        if version != SCHEMA_VERSION or fmt != PROJECT_FORMAT:
            raise LegacyWorkspaceCompatibilityUnavailable(
                "旧工作区兼容接口未能产生当前格式项目。"
            )
        self._ensure_layout(project)
        self._validate_current_workspace_files(project)
        return payload

    def _validate_current_workspace_files(self, project: Path) -> None:
        """Reject stale internal envelopes inside a current-schema project."""
        candidates: list[tuple[Path, str]] = []
        direct = [
            (project / PAGE_STATE_FILE, PAGE_STATE_FILE),
            (project / "artifacts/ocr/multi/latest.json", "artifacts/ocr/multi/latest.json"),
            (project / "adjudication/local/state.json", "adjudication/local/state.json"),
            (project / "adjudication/cloud/state.json", "adjudication/cloud/state.json"),
            (project / "metadata/page_hash_cache.json", "metadata/page_hash_cache.json"),
        ]
        candidates.extend((path, label) for path, label in direct if path.is_file())
        for root, pattern in ((project / "pipeline", "*.json"), (project / "artifacts/stages", "*.json")):
            if root.exists():
                candidates.extend((path, path.relative_to(project).as_posix()) for path in root.glob(pattern) if path.is_file())
        for path, label in candidates:
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                _require_workspace_schema(raw, label)
            except Exception as exc:
                raise ValueError(f"当前项目包含不兼容或损坏的内部状态 {label}：{exc}") from exc

    def _normalize_imported_project_identity(self, project: Path, *, source_project_id: str) -> None:
        payload = self._read_project_payload(project)
        existing_ids = {
            info.project_id
            for info in self.list_projects()
            if Path(info.path).resolve() != project.resolve()
        }
        if str(payload.get("project_id") or "") in existing_ids:
            payload["restored_from_project_id"] = str(source_project_id or payload.get("project_id") or "")
            payload["project_id"] = uuid.uuid4().hex
        payload["schema_version"] = SCHEMA_VERSION
        payload["format"] = PROJECT_FORMAT
        payload["restored_at"] = _now_iso()
        self._touch_project_payload(project, payload)

    def _unique_project_dir(self, name: str) -> Path:
        base = _safe_component(name, "project")
        target = self.projects_root / base
        serial = 2
        while target.exists():
            target = self.projects_root / f"{base} {serial}"
            serial += 1
        return target

    @staticmethod
    def _zip_compression_for_path(path: Path) -> int:
        """Avoid wasting CPU recompressing media and already-compressed containers."""
        if path.suffix.lower() in {
            ".png", ".jpg", ".jpeg", ".webp", ".heic", ".avif", ".gif",
            ".pdf", ".epub", ".zip", ".7z", ".rar", ".mp3", ".mp4", ".mov",
            ".gz", ".sqlite", ".sqlite3",
        }:
            return zipfile.ZIP_STORED
        return zipfile.ZIP_DEFLATED

    def _page_fingerprints(self, project: Path, paths: Iterable[Path]) -> list[dict[str, Any]]:
        """Return content fingerprints with a size/mtime backed SHA-256 cache."""
        cache_path = project / "metadata" / "page_hash_cache.json"
        try:
            cache_raw = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
        except Exception:
            cache_raw = {}
        cache = dict(cache_raw.get("files") or {}) if isinstance(cache_raw, dict) else {}
        changed = False
        result: list[dict[str, Any]] = []
        for raw in paths:
            path = Path(raw).expanduser().resolve()
            portable = self._portable_path(project, path)
            stat_result = path.stat()
            size = int(stat_result.st_size)
            mtime_ns = int(getattr(stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1_000_000_000)))
            cached = cache.get(portable) if isinstance(cache.get(portable), dict) else {}
            digest = ""
            if (
                int(cached.get("size") or -1) == size
                and int(cached.get("mtime_ns") or -1) == mtime_ns
                and str(cached.get("sha256") or "")
            ):
                digest = str(cached.get("sha256"))
            else:
                digest = _sha256_file(path)
                cache[portable] = {"size": size, "mtime_ns": mtime_ns, "sha256": digest}
                changed = True
            result.append({"path": portable, "size": size, "sha256": digest})
        if changed or not cache_path.exists():
            atomic_write_text(
                cache_path,
                json.dumps({"schema_version": SCHEMA_VERSION, "files": cache}, ensure_ascii=False, indent=2) + "\n",
            )
        return result

    @staticmethod
    def _project_info_from_payload(project: Path, payload: Mapping[str, Any]) -> ProjectInfo:
        return ProjectInfo(
            project_id=str(payload.get("project_id") or ""),
            name=str(payload.get("name") or project.name),
            path=str(project.resolve()),
            created_at=str(payload.get("created_at") or ""),
            updated_at=str(payload.get("updated_at") or ""),
        )

    def _read_project_info(self, project: Path) -> ProjectInfo | None:
        path = project / PROJECT_FILE
        try:
            stat = path.stat()
        except OSError:
            self._project_info_cache.pop(str(project), None)
            return None
        cache_key = str(project.resolve())
        identity = (int(stat.st_mtime_ns), int(stat.st_size))
        cached = self._project_info_cache.get(cache_key)
        if cached is not None and cached[:2] == identity:
            return cached[2]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if int(payload.get("schema_version", 0)) != SCHEMA_VERSION:
                info = None
            elif str(payload.get("format") or "") != PROJECT_FORMAT:
                info = None
            elif not str(payload.get("project_id") or ""):
                info = None
            else:
                info = self._project_info_from_payload(project, payload)
        except Exception:
            info = None
        self._project_info_cache[cache_key] = (identity[0], identity[1], info)
        return info

    @staticmethod
    def _read_project_payload(project: Path) -> dict:
        path = project / PROJECT_FILE
        if not path.exists():
            raise FileNotFoundError(str(path))
        raw = _require_workspace_schema(
            json.loads(path.read_text(encoding="utf-8")), "project.json"
        )
        if str(raw.get("format") or "") != PROJECT_FORMAT:
            raise ValueError(f"project.json format 必须为 {PROJECT_FORMAT}")
        return raw

    def _write_project_payload(self, project: Path, payload: Mapping[str, Any]) -> None:
        atomic_write_text(project / PROJECT_FILE, json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n")
        self._project_info_cache.pop(str(project.resolve()), None)

    def _touch_project_payload(self, project: Path, payload: dict) -> None:
        payload["updated_at"] = _now_iso()
        self._write_project_payload(project, payload)

    @staticmethod
    def _unique_target(root: Path, name: str) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        clean = _safe_component(name, "source")
        target = root / clean
        if not target.exists():
            return target
        stem, suffix = Path(clean).stem, Path(clean).suffix
        serial = 2
        while True:
            candidate = root / f"{stem}_{serial}{suffix}"
            if not candidate.exists():
                return candidate
            serial += 1

    @staticmethod
    def _is_within_project(path: Path, project: Path) -> bool:
        try:
            path.resolve().relative_to(project.resolve())
            return True
        except ValueError:
            return False

    @staticmethod
    def _portable_path(project: Path, path: Path) -> str:
        resolved = path.expanduser().resolve()
        try:
            return resolved.relative_to(project.resolve()).as_posix()
        except ValueError:
            return str(resolved)

    @staticmethod
    def _resolve_portable(project: Path, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path.expanduser().resolve()
        return (project / path).resolve()
