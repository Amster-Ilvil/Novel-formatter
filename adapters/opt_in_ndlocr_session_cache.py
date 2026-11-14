#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experimental opt-in NDLOCR-Lite *worker* cache (not the PR global LRU).

Keeps one successful, healthy NDLOCR JSONL worker between independent OCR
jobs.  Default OFF; never downloads models or alters inference outputs.
Strictly discards on Stop, protocol errors, worker death and changed runtime.
Does not purport to work without a complete official NDLOCR runtime installed.
"""
from __future__ import annotations

import atexit
import json
import os
import sys
from pathlib import Path

from adapters.opt_in_48px_session_cache import Single48pxCache

ENV_ENABLED = "NOVEL_FORMATTER_NDLOCR_SESSION_CACHE"


def enabled() -> bool:
    return os.environ.get(ENV_ENABLED, "").strip().lower() in {"1", "true", "yes", "on"}


def _stat(path: Path):
    try:
        s = path.stat()
        return (s.st_size, s.st_mtime_ns)
    except OSError:
        return (-1, -1)


def session_key(options: dict | None = None) -> str:
    """Conservative identity includes model files, source, provider and options."""
    from adapters.ndlocr_lite_adapter import SOURCE_DIR

    raw = os.environ.get("NOVEL_FORMATTER_NDLOCR_SOURCE_DIR", "").strip()
    source = Path(raw).expanduser().resolve() if raw else SOURCE_DIR.resolve()
    if source.name == "src" and (source / "ocr.py").exists():
        source = source.parent
    parts = ["src/ocr.py", "src/config/NDLmoji.yaml", "src/config/ndl.yaml"]
    if (source / "src/model").is_dir():
        parts.extend("src/model/" + p.name for p in sorted((source / "src/model").glob("*.onnx")))
    envs = (
        "NOVEL_FORMATTER_NDLOCR_SOURCE_DIR", "NOVEL_FORMATTER_NDLOCR_PYTHON",
        "NOVEL_FORMATTER_NDLOCR_REF", "NOVEL_FORMATTER_NDLOCR_PROVIDER",
        "NOVEL_FORMATTER_NDLOCR_TCY", "NOVEL_FORMATTER_NDLOCR_COREML_CACHE",
        "NOVEL_FORMATTER_NDLOCR_COREML_UNITS", "NOVEL_FORMATTER_OCR_OFFLINE",
        "NOVEL_FORMATTER_OCR_STARTUP_TIMEOUT", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    )
    try:
        opt = json.dumps(dict(options or {}), ensure_ascii=False, sort_keys=True)
    except (ValueError, TypeError):
        opt = repr(sorted((str(k), id(v)) for k, v in (options or {}).items()))
    return json.dumps({
        "source": str(source), "files": [(name, _stat(source / name)) for name in parts],
        "env": {k: os.environ.get(k, "") for k in envs},
        "options": opt, "python": sys.executable,
    }, ensure_ascii=False, sort_keys=True)


class NDLOcrWorkerHandle:
    """Adapt NDL's `.process` / `close()` to the existing tested lease API."""

    _FATAL_ERRORS = ("用户取消", "已终止", "常驻进程未", "常驻进程已", "无法启动", "模型加载失败")

    def __init__(self, session):
        self.session = session
        self.poisoned = False

    @property
    def proc(self):
        process = self.session.process
        return None if self.poisoned else process

    @property
    def cancel_check(self):
        return self.session.cancel_check

    @cancel_check.setter
    def cancel_check(self, callback):
        self.session.cancel_check = callback

    @property
    def verbose(self):
        return self.session.verbose

    @verbose.setter
    def verbose(self, value):
        self.session.verbose = bool(value)

    def iter_recognize(self, *args, **kwargs):
        try:
            for item in self.session.iter_recognize(*args, **kwargs):
                error = item[2] if len(item) >= 3 else None
                if error and any(marker in str(error) for marker in self._FATAL_ERRORS):
                    self.poisoned = True
                yield item
        except BaseException:
            self.poisoned = True
            raise

    def close(self, *, force: bool = False):
        self.poisoned = True
        self.session.close()

    def __getattr__(self, name):
        return getattr(self.session, name)


class SingleNDLOcrCache(Single48pxCache):
    """Independent one-slot cache; cannot evict another model's worker."""

    def snapshot(self) -> dict:
        """Report the NDLOCR opt-in flag, not the 48px flag from the base class."""
        state = super().snapshot()
        state["enabled"] = enabled()
        return state


_CACHE = SingleNDLOcrCache()
atexit.register(_CACHE.clear)


def default_cache() -> SingleNDLOcrCache:
    return _CACHE


def invalidate_ndlocr_session_cache() -> None:
    _CACHE.clear()
