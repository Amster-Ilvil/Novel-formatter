#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Opt-in single-worker Hayai cache, built on the tested exclusive 48px lease.

Default OFF. This is intentionally *not* the PR's multi-engine LRU. The
underlying worker can occupy ~1.4 GiB RSS. Stop, error and dead workers are
not kept; a normal job releases the exclusive lease without closing the model.
"""
from __future__ import annotations

import atexit
import json
import os
import sys
from pathlib import Path

from adapters.opt_in_48px_session_cache import Single48pxCache

ENV_ENABLED = "NOVEL_FORMATTER_HAYAI_SESSION_CACHE"


def enabled() -> bool:
    return os.environ.get(ENV_ENABLED, "").strip().lower() in {"1", "true", "yes", "on"}


def session_key(options: dict | None = None) -> str:
    """Include all startup options and the full local-model identity.

    Conservative extra misses are preferable to cross-job reuse with stale
    device, quantization, model or OCR segmentation settings.
    """
    values = dict(options or {})
    try:
        serialized = json.dumps(values, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        serialized = repr(sorted((str(key), id(value)) for key, value in values.items()))

    model_name = os.environ.get("NOVEL_FORMATTER_HAYAI_OCR_MODEL", "JustANormalTinkerer/hayai-ocr-v2.5-nova")
    local = Path(model_name).expanduser()
    signatures = []
    if local.is_dir():
        for name in ("model.safetensors", "config.json", "modeling_hayai.py", "configuration_hayai.py"):
            path = local / name
            try:
                stat = path.stat()
                signatures.append((name, stat.st_size, stat.st_mtime_ns))
            except OSError:
                signatures.append((name, -1, -1))
    relevant_env = (
        "NOVEL_FORMATTER_HAYAI_OCR_MODEL", "NOVEL_FORMATTER_HAYAI_OCR_PYTHON",
        "NOVEL_FORMATTER_HAYAI_OCR_CACHE_DIR", "NOVEL_FORMATTER_HAYAI_OCR_BATCH",
        "NOVEL_FORMATTER_HAYAI_OCR_THREADS", "NOVEL_FORMATTER_HAYAI_OCR_SOURCE_WINDOW",
        "NOVEL_FORMATTER_HAYAI_OCR_PACKAGE", "NOVEL_FORMATTER_HAYAI_TRANSFORMERS_PACKAGE",
        "NOVEL_FORMATTER_HAYAI_OCR_LITERT_REPO", "NOVEL_FORMATTER_HAYAI_OCR_LITERT_MODEL_PATH",
        "HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
    )
    return json.dumps({
        "model": str(local.resolve()) if local.exists() else model_name,
        "files": signatures,
        "options": serialized,
        "environment": {name: os.environ.get(name, "") for name in relevant_env},
        "python": sys.executable,
    }, sort_keys=True, ensure_ascii=False)


class SingleHayaiCache(Single48pxCache):
    """Separate cache instance; never shares a slot with the 48px model."""

    def snapshot(self) -> dict:
        """Report the Hayai opt-in flag, not the 48px flag from the base class."""
        state = super().snapshot()
        state["enabled"] = enabled()
        return state


_CACHE = SingleHayaiCache()
atexit.register(_CACHE.clear)


def default_cache() -> SingleHayaiCache:
    return _CACHE


def invalidate_hayai_session_cache() -> None:
    _CACHE.clear()


class HayaiLease:
    """Session-shaped object so the existing retained-holder lifecycle works.

    A regular __exit__ gives the loaded worker back to the cache; explicit
    close(force=True), errors and Stop make the cache discard it instead.
    """
    def __init__(self, loader, *, options=None, cancel_check=None, verbose=True, load_progress_callback=None):
        self._loader = loader
        self._options = dict(options or {})
        self._cancel_check = cancel_check
        self._verbose = verbose
        self._load_progress_callback = load_progress_callback
        self._lease = None
        self._session = None

    def __enter__(self):
        if self._session is not None:
            raise RuntimeError("Hayai cache lease already acquired")
        lease = _CACHE.lease(
            session_key(self._options), self._loader,
            cancel_check=self._cancel_check, verbose=self._verbose,
            load_progress_callback=self._load_progress_callback,
        )
        session = lease.__enter__()
        session.options = dict(self._options)
        self._lease = lease
        self._session = session
        return self

    def __getattr__(self, attr):
        session = self._session
        if session is None:
            raise AttributeError(attr)
        return getattr(session, attr)

    @property
    def cancel_check(self):
        return self._session.cancel_check if self._session is not None else self._cancel_check

    @cancel_check.setter
    def cancel_check(self, callback):
        self._cancel_check = callback
        if self._session is not None:
            self._session.cancel_check = callback

    @property
    def verbose(self):
        return self._session.verbose if self._session is not None else self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = bool(value)
        if self._session is not None:
            self._session.verbose = bool(value)

    @property
    def load_progress_callback(self):
        return self._session.load_progress_callback if self._session is not None else self._load_progress_callback

    @load_progress_callback.setter
    def load_progress_callback(self, value):
        self._load_progress_callback = value
        if self._session is not None:
            self._session.load_progress_callback = value

    def __exit__(self, exc_type, exc, tb):
        lease, self._lease = self._lease, None
        self._session = None
        if lease is not None:
            return bool(lease.__exit__(exc_type, exc, tb))
        return False

    def close(self, *, force: bool = False):
        if force:
            _CACHE.clear()
        return self.__exit__(None, None, None)
