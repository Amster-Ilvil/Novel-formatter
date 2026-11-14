#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Opt-in, single-entry 48px OCR worker cache, not the PR #12 global LRU.

The default is OFF.  Set NOVEL_FORMATTER_48PX_SESSION_CACHE=1 to keep exactly
one healthy worker loaded between sequential column OCR jobs.  A cache entry
owns hundreds of MiB of memory, so an explicit opt-in is required.  Concurrent
jobs serialize and may cancel while waiting. Stop/error/dead workers are never
reused, and explicit invalidation/exit closes idle workers.

The existing recognizer and all of its inference/output logic are unchanged.
"""
from __future__ import annotations

import atexit
import contextlib
import json
import os
import threading
from pathlib import Path
from typing import Callable, Iterator, Any

from core.ocr_runtime_optimizer import OcrCancelled

ENV_ENABLED = "NOVEL_FORMATTER_48PX_SESSION_CACHE"


def enabled() -> bool:
    return os.environ.get(ENV_ENABLED, "").strip().lower() in {"1", "true", "yes", "on"}


def _check_cancel(callback) -> None:
    if callback and callback():
        raise OcrCancelled("等待 48px 共享 Worker 时用户取消")


def session_key(options: dict | None = None) -> str:
    # Treat an unknown option type as a cache miss rather than assuming it
    # has no influence on startup or inference. The rare callable option is
    # stringified with identity, so it cannot alias a different callback.
    raw_options = dict(options or {})
    relevant_options = {
        str(k): v for k, v in raw_options.items()
        if not str(k).startswith("column_")
    }
    try:
        serialized = json.dumps(relevant_options, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        serialized = repr(sorted((str(k), id(v)) for k, v in relevant_options.items()))
    from utils.model_inventory import manga_48px_cache_dir
    model_dir = manga_48px_cache_dir().resolve()
    fingerprints: list[tuple[str, int, int]] = []
    for name in ("ocr_ar_48px.ckpt", "alphabet-all-v7.txt",
                 "upstream-source/model_48px.py", "upstream-source/xpos_relative_position.py"):
        candidate = model_dir / name
        try:
            stat = candidate.stat()
            fingerprints.append((name, stat.st_size, stat.st_mtime_ns))
        except OSError:
            fingerprints.append((name, -1, -1))
    worker_env = {
        name: os.environ.get(name, "")
        for name in (
            "NOVEL_FORMATTER_MANGA_48PX_PYTHON",
            "NOVEL_FORMATTER_MANGA_48PX_THREADS",
            "NOVEL_FORMATTER_MANGA_48PX_MAX_REFLOW_CHARS",
            "NOVEL_FORMATTER_MANGA_48PX_MODEL_FILE",
            "NOVEL_FORMATTER_MANGA_48PX_DICT_FILE",
        )
    }
    return json.dumps([str(model_dir), fingerprints, serialized, worker_env],
                      sort_keys=True, ensure_ascii=False)


def _alive(session) -> bool:
    proc = getattr(session, "proc", None)
    return proc is not None and proc.poll() is None


def _close(session) -> None:
    try:
        session.close(force=True)
    except Exception:
        pass


class Single48pxCache:
    """A single leased process with cancel-aware sequential handoff."""

    def __init__(self):
        self._condition = threading.Condition(threading.RLock())
        self._key: str | None = None
        self._session: Any = None
        self._busy = False
        self._owner: int | None = None
        self._invalidated = False
        self.hits = 0
        self.loads = 0

    @contextlib.contextmanager
    def lease(self, key: str, loader: Callable[[], Any], *, cancel_check=None,
              verbose: bool = True, load_progress_callback=None) -> Iterator[Any]:
        _check_cancel(cancel_check)
        me = threading.get_ident()
        with self._condition:
            while self._busy:
                if self._owner == me:
                    raise RuntimeError("48px cache cannot be nested in the same thread")
                _check_cancel(cancel_check)
                self._condition.wait(timeout=0.10)
            _check_cancel(cancel_check)
            self._busy = True
            self._owner = me
            previous, previous_key = self._session, self._key
            self._session = None
            self._key = None
            invalidated = self._invalidated
            self._invalidated = False

        worker = None
        failed = True
        try:
            if not invalidated and previous_key == key and previous is not None and _alive(previous):
                worker = previous
                self.hits += 1
            else:
                if previous is not None:
                    _close(previous)
                worker = loader()
                self.loads += 1
            # Every job must use its own cancel signal and progress callback.
            # Do not allow an old Stop event to affect the next recognition run.
            worker.cancel_check = cancel_check
            worker.verbose = verbose
            worker.load_progress_callback = load_progress_callback
            _check_cancel(cancel_check)
            yield worker
            _check_cancel(cancel_check)
            failed = not _alive(worker)
        finally:
            # Keep the exclusive lease until the old process has exited.
            # Otherwise a waiting job could start another 500 MB worker while
            # the failed/evicted process is still shutting down.
            try:
                with self._condition:
                    if failed or self._invalidated or not _alive(worker):
                        discarded = worker
                        worker = None
                    else:
                        discarded = None
                    self._session = worker
                    self._key = key if worker is not None else None
                    self._invalidated = False
                if discarded is not None:
                    _close(discarded)
            finally:
                with self._condition:
                    self._busy = False
                    self._owner = None
                    self._condition.notify_all()

    def clear(self) -> None:
        with self._condition:
            if self._busy:
                self._invalidated = True
                return
            worker, self._session = self._session, None
            self._key = None
        if worker is not None:
            _close(worker)

    def snapshot(self) -> dict:
        with self._condition:
            return {"enabled": enabled(), "busy": self._busy, "resident": self._session is not None,
                    "hits": self.hits, "loads": self.loads}


_DEFAULT_CACHE = Single48pxCache()
atexit.register(_DEFAULT_CACHE.clear)


def default_cache() -> Single48pxCache:
    return _DEFAULT_CACHE


def invalidate_48px_session_cache() -> None:
    _DEFAULT_CACHE.clear()
