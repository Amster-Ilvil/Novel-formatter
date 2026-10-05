# -*- coding: utf-8 -*-
"""Content-addressed artifact graph, stage cache and resumable checkpoints.

This module deliberately has no Qt dependency.  It borrows the useful ideas
from asset-oriented workflow systems (explicit lineage, deterministic cache
keys and resumable task ledgers) without pulling an orchestration framework
into the desktop application.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
import re
from pathlib import Path
import tempfile
import threading
from typing import Any, Iterable, Mapping

from utils.atomic_io import atomic_write_text

ARTIFACT_GRAPH_FORMAT = "novel-formatter-artifact-graph"
ARTIFACT_GRAPH_SCHEMA = 1
STAGE_CACHE_FORMAT = "novel-formatter-stage-cache"
STAGE_CACHE_SCHEMA = 1
CHECKPOINT_FORMAT = "novel-formatter-checkpoint"
CHECKPOINT_SCHEMA = 1
OBJECT_REF_FORMAT = "novel-formatter-json-object-ref"
OBJECT_REF_SCHEMA = 1


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def normalize_for_hash(value: Any) -> Any:
    """Convert heterogeneous runtime config into deterministic JSON data.

    Cache keys may receive Paths, sets, dataclasses and small custom objects.
    Secrets are never written to cache metadata by callers; this normalized form
    is consumed only by SHA-256 calculation unless explicitly requested.
    """
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        # Avoid platform-specific repr tails while preserving practical config.
        return round(value, 12)
    if isinstance(value, Path):
        return str(value.expanduser())
    if is_dataclass(value):
        return normalize_for_hash(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): normalize_for_hash(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [normalize_for_hash(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [normalize_for_hash(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True, default=str))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return normalize_for_hash(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            public = {
                str(k): v for k, v in vars(value).items()
                if not str(k).startswith("_")
            }
            if public:
                return normalize_for_hash(public)
        except Exception:
            pass
    return {"__type__": type(value).__name__, "repr": str(value)}


def stable_json_hash(value: Any) -> str:
    raw = json.dumps(
        normalize_for_hash(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), default=str,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def implementation_fingerprint(paths: Iterable[str | Path]) -> str:
    """Fingerprint implementation files so code edits invalidate stale caches."""
    rows: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw)
        try:
            stat = path.stat()
            rows.append({
                "name": path.name,
                "size": int(stat.st_size),
                "sha256": file_sha256(path),
            })
        except OSError:
            rows.append({"name": path.name, "missing": True})
    return stable_json_hash(rows)


class ContentAddressedJsonStore:
    """Compressed immutable JSON object store shared by workspace artifacts.

    Large OCR/stage/cache snapshots are stored once as immutable compressed
    objects under ``revisions/objects``. Current logical files contain only
    schema-bound object references; older inline layouts are intentionally not
    accepted during pre-stable development.
    """

    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root)
        self.root = self.project_root / "revisions" / "objects"

    @staticmethod
    def _serialize(payload: Any) -> bytes:
        # Storage must preserve the exact JSON data model.  Do not reuse
        # ``normalize_for_hash`` here: that helper intentionally rounds floats
        # for cache-key stability, while OCR confidence/provenance values are
        # authoritative persisted data.
        return json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), default=str,
        ).encode("utf-8")

    def put(self, payload: Any, *, kind: str = "json") -> dict[str, Any]:
        raw = self._serialize(payload)
        digest = hashlib.sha256(raw).hexdigest()
        relative = Path(digest[:2]) / f"{digest}.json.gz"
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)

        def existing_object_is_valid() -> bool:
            if not target.is_file():
                return False
            try:
                existing_raw = gzip.decompress(target.read_bytes())
            except Exception:
                return False
            return hashlib.sha256(existing_raw).hexdigest() == digest

        if not existing_object_is_valid():
            compressed = gzip.compress(raw, compresslevel=6, mtime=0)
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
            )
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(compressed)
                    fh.flush()
                    os.fsync(fh.fileno())
                # Content-addressed writes are idempotent. Replacing an invalid or
                # concurrently-created target with the same digest is safe and also
                # repairs interrupted/corrupted prior writes.
                os.replace(tmp_name, target)
            finally:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
        try:
            stored_bytes = int(target.stat().st_size)
        except OSError:
            stored_bytes = 0
        return {
            "format": OBJECT_REF_FORMAT,
            "schema_version": OBJECT_REF_SCHEMA,
            "sha256": digest,
            "codec": "gzip-json",
            "path": f"project://revisions/objects/{relative.as_posix()}",
            "kind": str(kind or "json"),
            "logical_bytes": len(raw),
            "stored_bytes": stored_bytes,
        }

    def _validated_reference_path(self, reference: Mapping[str, Any]) -> tuple[str, Path]:
        if not isinstance(reference, Mapping):
            raise ValueError("invalid object reference")
        if str(reference.get("format") or "") != OBJECT_REF_FORMAT:
            raise ValueError("unsupported object reference")
        if int(reference.get("schema_version") or 0) != OBJECT_REF_SCHEMA:
            raise ValueError("unsupported object reference schema")
        if str(reference.get("codec") or "") != "gzip-json":
            raise ValueError("unsupported object reference codec")
        digest = str(reference.get("sha256") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid object digest")

        canonical_relative = Path("revisions") / "objects" / digest[:2] / f"{digest}.json.gz"
        canonical_uri = f"project://{canonical_relative.as_posix()}"
        raw_path = str(reference.get("path") or "")
        if raw_path != canonical_uri:
            raise ValueError("non-canonical workspace object reference path")
        candidate = (self.project_root / canonical_relative).resolve()
        candidate.relative_to(self.project_root.resolve())
        return digest, candidate

    def resolve(self, reference: Mapping[str, Any]) -> Any:
        digest, candidate = self._validated_reference_path(reference)
        compressed = candidate.read_bytes()
        raw = gzip.decompress(compressed)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"workspace object digest mismatch: {digest}")
        return json.loads(raw.decode("utf-8"))

    def referenced_path(self, reference: Mapping[str, Any]) -> Path:
        _digest, path = self._validated_reference_path(reference)
        return path


def is_object_ref(value: Any) -> bool:
    return isinstance(value, Mapping) and str(value.get("format") or "") == OBJECT_REF_FORMAT


class ArtifactGraphStore:
    """Small content-addressed graph persisted as one atomic JSON file."""

    _write_lock = threading.RLock()

    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root)
        self.path = self.project_root / "metadata" / "artifact_graph.json"

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "format": ARTIFACT_GRAPH_FORMAT,
                "schema_version": ARTIFACT_GRAPH_SCHEMA,
                "updated_at": "",
                "nodes": {},
                "current": {},
            }
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        raw.setdefault("format", ARTIFACT_GRAPH_FORMAT)
        raw.setdefault("schema_version", ARTIFACT_GRAPH_SCHEMA)
        raw.setdefault("nodes", {})
        raw.setdefault("current", {})
        return raw

    def register(
        self,
        *,
        kind: str,
        digest: str,
        alias: str = "",
        path: str = "",
        dependencies: Iterable[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        return self.register_many([{
            "kind": kind,
            "digest": digest,
            "alias": alias,
            "path": path,
            "dependencies": list(dependencies),
            "metadata": dict(metadata or {}),
        }])[0]

    def register_many(self, registrations: Iterable[Mapping[str, Any]]) -> list[str]:
        """Register related nodes and write the graph once for the whole batch."""
        # Page-state and OCR/stage saves may finish on different threads.
        # Atomic replacement alone does not protect their read-modify-write
        # cycle from losing one writer's newly registered nodes.
        with self._write_lock:
            return self._register_many_locked(registrations)

    def _register_many_locked(self, registrations: Iterable[Mapping[str, Any]]) -> list[str]:
        graph = self.load()
        nodes = graph.setdefault("nodes", {})
        registered: list[str] = []
        for registration in registrations:
            safe_kind = str(registration.get("kind") or "artifact").strip() or "artifact"
            path = str(registration.get("path") or "")
            safe_digest = str(
                registration.get("digest")
                or stable_json_hash({"kind": safe_kind, "path": path})
            )
            node_id = f"{safe_kind}:{safe_digest}"
            existing = dict(nodes.get(node_id) or {})
            deps = [str(v) for v in registration.get("dependencies", ()) if str(v)]
            node = {
                "id": node_id,
                "kind": safe_kind,
                "digest": safe_digest,
                "path": str(path or existing.get("path") or ""),
                "created_at": str(existing.get("created_at") or now_iso()),
                "last_seen_at": now_iso(),
                "dependencies": sorted(set(deps or existing.get("dependencies") or [])),
                "metadata": dict(registration.get("metadata") or existing.get("metadata") or {}),
            }
            nodes[node_id] = node
            alias = str(registration.get("alias") or "")
            if alias:
                graph.setdefault("current", {})[alias] = node_id
            registered.append(node_id)
        graph["updated_at"] = now_iso()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.path, json.dumps(graph, ensure_ascii=False, indent=2) + "\n")
        return registered

    def current_node(self, alias: str) -> str:
        return str(self.load().get("current", {}).get(str(alias), "") or "")

    def lineage(self, node_id: str) -> list[str]:
        graph = self.load()
        nodes = dict(graph.get("nodes") or {})
        result: list[str] = []
        seen: set[str] = set()

        def visit(current: str) -> None:
            if not current or current in seen:
                return
            seen.add(current)
            node = nodes.get(current)
            if not isinstance(node, dict):
                return
            for dep in node.get("dependencies", []) or []:
                visit(str(dep))
            result.append(current)

        visit(str(node_id))
        return result

    def descendants(self, node_id: str) -> list[str]:
        graph = self.load()
        nodes = dict(graph.get("nodes") or {})
        reverse: dict[str, set[str]] = {}
        for child_id, node in nodes.items():
            if not isinstance(node, dict):
                continue
            for dep in node.get("dependencies", []) or []:
                reverse.setdefault(str(dep), set()).add(str(child_id))
        result: list[str] = []
        queue = list(sorted(reverse.get(str(node_id), set())))
        seen: set[str] = set()
        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            result.append(current)
            queue.extend(sorted(reverse.get(current, set())))
        return result


class StageCacheStore:
    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root)
        self.root = self.project_root / "artifacts" / "cache"

    @staticmethod
    def make_key(
        *, stage: str,
        page_lineage_hash: str,
        config: Any,
        inputs: Any = None,
        implementation: str = "",
    ) -> str:
        return stable_json_hash({
            "stage": str(stage),
            "page_lineage_hash": str(page_lineage_hash or ""),
            "inputs": inputs,
            "config": config,
            "implementation": str(implementation or ""),
        })

    def _path(self, stage: str, key: str) -> Path:
        stage_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(stage or "stage"))
        return self.root / stage_name / f"{key}.json"

    def load(self, *, stage: str, key: str, page_lineage_hash: str = "") -> dict[str, Any] | None:
        path = self._path(stage, key)
        if not path.exists():
            return None
        try:
            wrapper = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(wrapper, dict) or wrapper.get("format") != STAGE_CACHE_FORMAT:
            return None
        if str(wrapper.get("key") or "") != str(key):
            return None
        saved_lineage = str(wrapper.get("page_lineage_hash") or "")
        if page_lineage_hash and saved_lineage and saved_lineage != page_lineage_hash:
            return None
        payload = wrapper.get("payload")
        if isinstance(payload, dict):
            return dict(payload)
        reference = wrapper.get("payload_ref")
        if is_object_ref(reference):
            try:
                resolved = ContentAddressedJsonStore(self.project_root).resolve(reference)
            except Exception:
                return None
            return dict(resolved) if isinstance(resolved, dict) else None
        return None

    def save(
        self,
        *,
        stage: str,
        key: str,
        page_lineage_hash: str,
        payload: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        path = self._path(stage, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload_dict = dict(payload)
        reference = ContentAddressedJsonStore(self.project_root).put(
            payload_dict, kind=f"stage_cache:{stage}"
        )

        portable_paths: set[str] = set()
        def collect_paths(value: Any) -> None:
            if isinstance(value, Mapping):
                for child in value.values():
                    collect_paths(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    collect_paths(child)
            elif isinstance(value, str) and value.startswith("project://"):
                portable_paths.add(value)
        collect_paths(payload_dict)

        wrapper = {
            "format": STAGE_CACHE_FORMAT,
            "schema_version": STAGE_CACHE_SCHEMA,
            "stage": str(stage),
            "key": str(key),
            "created_at": now_iso(),
            "page_lineage_hash": str(page_lineage_hash or ""),
            "metadata": dict(metadata or {}),
            "portable_paths": sorted(portable_paths),
            "payload_ref": reference,
        }
        atomic_write_text(path, json.dumps(wrapper, ensure_ascii=False, indent=2) + "\n")
        return path

    def prune(self, *, keep_recent: int = 300) -> int:
        files = [p for p in self.root.rglob("*.json") if p.is_file()] if self.root.exists() else []
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        removed = 0
        for path in files[max(0, int(keep_recent)):]:
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
        return removed


class CheckpointStore:
    """Atomic resumable ledger; results live in StageCacheStore, not the ledger."""

    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root)
        self.root = self.project_root / "artifacts" / "checkpoints"

    def _path(self, stage: str, signature: str) -> Path:
        stage_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(stage or "stage"))
        return self.root / stage_name / f"{signature}.json"

    def load(self, *, stage: str, signature: str) -> dict[str, Any]:
        path = self._path(stage, signature)
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return raw if isinstance(raw, dict) else {}

    def begin(
        self,
        *,
        stage: str,
        signature: str,
        page_lineage_hash: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        previous = self.load(stage=stage, signature=signature)
        completed = dict(previous.get("completed_steps") or {}) if isinstance(previous, dict) else {}
        payload = {
            "format": CHECKPOINT_FORMAT,
            "schema_version": CHECKPOINT_SCHEMA,
            "stage": str(stage),
            "signature": str(signature),
            "page_lineage_hash": str(page_lineage_hash or ""),
            "status": "running",
            "created_at": str(previous.get("created_at") or now_iso()) if isinstance(previous, dict) else now_iso(),
            "updated_at": now_iso(),
            "completed_steps": completed,
            "metadata": dict(metadata or {}),
        }
        path = self._path(stage, signature)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return payload

    def mark(
        self,
        *,
        stage: str,
        signature: str,
        step: str,
        cache_key: str,
        reused: bool = False,
        detail: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self.load(stage=stage, signature=signature)
        if not payload:
            payload = self.begin(stage=stage, signature=signature, page_lineage_hash="")
        completed = dict(payload.get("completed_steps") or {})
        completed[str(step)] = {
            "cache_key": str(cache_key),
            "completed_at": now_iso(),
            "reused": bool(reused),
            "detail": dict(detail or {}),
        }
        payload["completed_steps"] = completed
        payload["updated_at"] = now_iso()
        path = self._path(stage, signature)
        atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return payload

    def finish(self, *, stage: str, signature: str, status: str) -> dict[str, Any]:
        payload = self.load(stage=stage, signature=signature)
        if not payload:
            return {}
        payload["status"] = str(status or "ok")
        payload["updated_at"] = now_iso()
        payload["finished_at"] = now_iso()
        path = self._path(stage, signature)
        atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return payload
