# -*- coding: utf-8 -*-
"""Content-addressed block/page history snapshots for DocumentVersionStore.

Current development workspaces use one strict snapshot format. Historical
revisions deduplicate unchanged pages/blocks through immutable objects; older
complete-JSON history entries are intentionally unsupported before stable.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from utils.atomic_io import atomic_write_text

SCHEMA = "novel_formatter.version_snapshot.v2"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


class VersionDeltaStore:
    def __init__(self, revisions_dir: str | Path):
        self.root = Path(revisions_dir)
        self.objects = self.root / "objects"
        self.objects.mkdir(parents=True, exist_ok=True)

    def _object_path(self, object_id: str) -> Path:
        return self.objects / object_id[:2] / f"{object_id[2:]}.json"

    def store_object(self, value: Any) -> str:
        raw = _canonical_bytes(value)
        object_id = hashlib.sha256(raw).hexdigest()
        target = self._object_path(object_id)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(target, raw.decode("utf-8") + "\n")
        return object_id

    def read_object(self, object_id: str) -> Any:
        if not object_id or len(object_id) != 64:
            raise ValueError("无效的版本对象 ID")
        target = self._object_path(object_id)
        if not target.is_file():
            raise FileNotFoundError(f"版本对象缺失：{object_id}")
        raw = target.read_bytes().strip()
        actual = hashlib.sha256(raw).hexdigest()
        # Files are written as canonical JSON plus one trailing newline. strip()
        # recovers the exact canonical bytes used to derive the object id.
        if actual != object_id:
            raise ValueError(f"版本对象校验失败：{object_id}")
        return json.loads(raw.decode("utf-8"))

    def snapshot_payload(
        self,
        payload: dict,
        history_dir: str | Path,
        *,
        kind: str,
        stamp_ns: int | None = None,
    ) -> Path:
        """Persist one historical payload as a small manifest + shared objects."""
        if not isinstance(payload, dict):
            raise TypeError("版本快照必须是 JSON 对象")
        history = Path(history_dir)
        history.mkdir(parents=True, exist_ok=True)
        pages = list(payload.get("pages") or [])
        blocks = list(payload.get("blocks") or [])
        shell = {k: v for k, v in payload.items() if k not in {"pages", "blocks"}}

        manifest = {
            "schema": SCHEMA,
            "kind": str(kind),
            "created_at_ns": int(stamp_ns or time.time_ns()),
            "shell": self.store_object(shell),
            "pages": [self.store_object(page) for page in pages],
            "blocks": [self.store_object(block) for block in blocks],
        }
        # The manifest itself is intentionally readable and tiny; object hashes make
        # corruption/missing-object failures explicit instead of silently restoring
        # a partial book.
        manifest_id = hashlib.sha256(_canonical_bytes(manifest)).hexdigest()
        manifest["manifest_id"] = manifest_id
        revision = int(((payload.get("version") or {}).get("revision", 0)) or 0)
        name = f"{manifest['created_at_ns']}_{revision:06d}_{manifest_id[:12]}.snapshot.json"
        target = history / name
        atomic_write_text(target, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        return target

    def load_snapshot(self, snapshot_path: str | Path) -> dict:
        source = Path(snapshot_path)
        manifest = json.loads(source.read_text(encoding="utf-8"))
        if manifest.get("schema") != SCHEMA:
            raise ValueError(f"不支持的历史快照格式：{source}")
        manifest_for_hash = {k: v for k, v in manifest.items() if k != "manifest_id"}
        expected = hashlib.sha256(_canonical_bytes(manifest_for_hash)).hexdigest()
        if manifest.get("manifest_id") != expected:
            raise ValueError(f"历史快照清单校验失败：{source.name}")

        shell = self.read_object(str(manifest.get("shell") or ""))
        if not isinstance(shell, dict):
            raise ValueError("历史快照 shell 不是 JSON 对象")
        payload = dict(shell)
        payload["pages"] = [self.read_object(str(value)) for value in manifest.get("pages") or []]
        payload["blocks"] = [self.read_object(str(value)) for value in manifest.get("blocks") or []]
        return payload

    def list_history(self, history_dir: str | Path) -> list[Path]:
        history = Path(history_dir)
        if not history.is_dir():
            return []
        snapshots = list(history.glob("*.snapshot.json"))
        return sorted(snapshots, key=lambda p: (p.stat().st_mtime_ns, p.name), reverse=True)

    def reachable_object_ids(self) -> set[str]:
        """Return every object referenced by every valid snapshot manifest.

        Corrupt/unreadable manifests abort the scan. Garbage collection must fail
        closed rather than delete objects that might still be referenced.
        """
        reachable: set[str] = set()
        for manifest_path in sorted(self.root.glob("*_history/*.snapshot.json")):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema") != SCHEMA:
                raise ValueError(f"无法安全扫描历史快照：{manifest_path}")
            manifest_for_hash = {k: v for k, v in manifest.items() if k != "manifest_id"}
            expected = hashlib.sha256(_canonical_bytes(manifest_for_hash)).hexdigest()
            if manifest.get("manifest_id") != expected:
                raise ValueError(f"历史快照清单校验失败：{manifest_path.name}")
            shell = str(manifest.get("shell") or "")
            if shell:
                reachable.add(shell)
            reachable.update(str(value) for value in (manifest.get("pages") or []) if value)
            reachable.update(str(value) for value in (manifest.get("blocks") or []) if value)
        return reachable

    def collect_garbage(self, *, dry_run: bool = True) -> dict:
        """Find/delete unreachable immutable objects; defaults to non-destructive dry-run."""
        reachable = self.reachable_object_ids()
        candidates: list[tuple[str, Path, int]] = []
        if self.objects.is_dir():
            for path in self.objects.glob("*/*.json"):
                object_id = path.parent.name + path.stem
                if len(object_id) != 64 or object_id in reachable:
                    continue
                candidates.append((object_id, path, int(path.stat().st_size)))
        deleted = 0
        reclaimed = 0
        if not dry_run:
            for _object_id, path, size in candidates:
                path.unlink(missing_ok=True)
                if not path.exists():
                    deleted += 1
                    reclaimed += size
            for bucket in self.objects.iterdir() if self.objects.exists() else []:
                if bucket.is_dir():
                    try:
                        bucket.rmdir()
                    except OSError:
                        pass
        return {
            "dry_run": bool(dry_run),
            "reachable": len(reachable),
            "unreachable": len(candidates),
            "deleted": deleted,
            "reclaimable_bytes": sum(item[2] for item in candidates),
            "reclaimed_bytes": reclaimed,
            "object_ids": [item[0] for item in candidates],
        }
