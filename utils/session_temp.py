# -*- coding: utf-8 -*-
"""Session-owned temporary storage with crash-recovery cleanup.

All large PDF rasters, OCR preview files, and ephemeral formatter repositories
must live under one application-owned root.  The root is removed on normal
shutdown and stale roots left by a crash are reclaimed on the next start.
"""
from __future__ import annotations

import atexit
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

_APP_DIR_NAME = "novel_formatter_sessions"
_MARKER_NAME = ".novel_formatter_session.json"
_DEFAULT_STALE_SECONDS = 24 * 60 * 60


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _safe_remove_tree(path: Path) -> bool:
    try:
        shutil.rmtree(path)
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return False


class SessionTempRegistry:
    """Own every run-local directory and delete it deterministically."""

    def __init__(self, *, stale_seconds: int = _DEFAULT_STALE_SECONDS) -> None:
        self._lock = threading.RLock()
        self._closed = False
        self._base = Path(tempfile.gettempdir()) / _APP_DIR_NAME
        self._base.mkdir(parents=True, exist_ok=True)
        self._cleanup_stale_sessions(stale_seconds=max(3600, int(stale_seconds)))
        token = f"session-{os.getpid()}-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        self.root = self._base / token
        self.root.mkdir(parents=True, exist_ok=False)
        self._owned: set[Path] = {self.root}
        marker = {
            "pid": os.getpid(),
            "created": time.time(),
            "version": 1,
        }
        (self.root / _MARKER_NAME).write_text(json.dumps(marker), encoding="utf-8")
        atexit.register(self.cleanup)

    def _cleanup_stale_sessions(self, *, stale_seconds: int) -> None:
        now = time.time()
        try:
            children = list(self._base.iterdir())
        except OSError:
            return
        for child in children:
            if not child.is_dir():
                continue
            marker = child / _MARKER_NAME
            try:
                payload = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else {}
            except Exception:
                payload = {}
            try:
                pid = int(payload.get("pid", 0) or 0)
            except (TypeError, ValueError):
                pid = 0
            try:
                created = float(payload.get("created", child.stat().st_mtime) or child.stat().st_mtime)
            except OSError:
                created = 0.0
            # Never remove an active process' root.  A marked root whose PID
            # is gone is known to be ours and can be reclaimed immediately; an
            # unknown/unmarked directory still waits for the age threshold.
            if _pid_is_alive(pid):
                continue
            if marker.exists() and pid > 0:
                _safe_remove_tree(child)
            elif now - created >= stale_seconds:
                _safe_remove_tree(child)

    def make_dir(self, prefix: str) -> Path:
        clean = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(prefix or "tmp"))
        clean = clean.strip("._-")[:48] or "tmp"
        with self._lock:
            if self._closed:
                raise RuntimeError("temporary session has already closed")
            path = self.root / f"{clean}-{uuid.uuid4().hex[:10]}"
            path.mkdir(parents=True, exist_ok=False)
            self._owned.add(path)
            return path

    def path(self, *parts: str) -> Path:
        with self._lock:
            if self._closed:
                raise RuntimeError("temporary session has already closed")
            target = self.root.joinpath(*map(str, parts))
            resolved = target.resolve(strict=False)
            root = self.root.resolve(strict=False)
            if resolved != root and root not in resolved.parents:
                raise ValueError("temporary path escapes the session root")
            resolved.parent.mkdir(parents=True, exist_ok=True)
            return resolved

    def register(self, path: str | Path) -> Path:
        """Register a directory already created by legacy code.

        Only descendants of the system temporary directory are accepted.  User
        documents and source folders can never be deleted through this API.
        """
        candidate = Path(path).expanduser().resolve(strict=False)
        temp_root = Path(tempfile.gettempdir()).resolve(strict=False)
        if candidate != temp_root and temp_root not in candidate.parents:
            raise ValueError(f"refusing to own non-temporary path: {candidate}")
        with self._lock:
            if self._closed:
                _safe_remove_tree(candidate)
            else:
                self._owned.add(candidate)
        return candidate

    def release(self, path: str | Path, *, delete: bool = True) -> bool:
        candidate = Path(path).expanduser().resolve(strict=False)
        root = self.root.resolve(strict=False)
        if candidate != root and root not in candidate.parents:
            raise ValueError(f"refusing to release path outside this session: {candidate}")
        with self._lock:
            self._owned.discard(candidate)
        return _safe_remove_tree(candidate) if delete else True

    def cleanup(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            # Deleting the common root is both faster and safer than walking
            # every nested preview/crop directory separately.
            root = self.root
            self._owned.clear()
        _safe_remove_tree(root)


_registry_lock = threading.Lock()
_registry: SessionTempRegistry | None = None


def session_temp_registry() -> SessionTempRegistry:
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = SessionTempRegistry()
        return _registry


def cleanup_session_temp() -> None:
    global _registry
    with _registry_lock:
        registry = _registry
        _registry = None
    if registry is not None:
        registry.cleanup()
