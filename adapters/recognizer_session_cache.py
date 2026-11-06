#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Process-wide LRU cache of loaded OCR recognizer sessions.

Every heavyweight recognizer (48px AR, NDLOCR-lite, PaddleOCR, Hayai, the
Apple Vision helper) runs its model inside a long-lived worker process owned by
a session object.  Before this cache those sessions lived only as long as one
``column_ocr_adapter.run()`` call, so every OCR job paid the full worker start
and model load again.  The cache keeps up to ``capacity`` sessions (default 2,
``NOVEL_FORMATTER_MODEL_CACHE_SIZE`` overrides, ``0`` disables caching) keyed by
:class:`SessionKey`: engine, model location from
:class:`utils.model_manager.ModelManager`, device/provider and a fingerprint of
the engine options that a worker reads at start-up.

Guarantees:

* one loader call per key even when many threads ask at once;
* exclusive leases: a second job using the same model waits (cancel-aware)
  instead of starting a second copy of the model; nested leases from the same
  thread share the session, so recursive passes cannot dead-lock;
* ``cancel_check``/``verbose`` are rebound on every lease, so a reused worker
  obeys the *current* job's Stop button;
* dead workers are reloaded, and a lease that ends with a real exception
  discards the session (fail closed);
* Stop is not a failure: a lease that ends with a cancellation exception
  (:class:`InterruptedError`, which includes ``OcrCancelled``, or
  ``ProcessCancelled``) keeps the session when its worker is still alive.
  Every bundled engine is consistent after Stop: JSON-line workers either
  match responses by ``request_id`` or are terminated by the watchdog (then the
  liveness check reloads them), and the Apple Vision backends recreate their
  helper client on the next request.  A session class can opt out with a
  truthy ``discard_on_cancel`` attribute;
* :func:`invalidate_recognizer_sessions` closes sessions explicitly, e.g. when
  the models folder or the default engine changes in Settings.
"""
from __future__ import annotations

import atexit
import contextlib
import hashlib
import json
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from core.ocr_runtime_optimizer import OcrCancelled


def _check_cancel(cancel_check) -> None:
    if cancel_check is not None and cancel_check():
        raise OcrCancelled("OCR stopped while waiting for a shared recognizer")


ENV_CACHE_SIZE = "NOVEL_FORMATTER_MODEL_CACHE_SIZE"
DEFAULT_CAPACITY = 2
_MAX_CAPACITY = 16
_LEASE_POLL_SECONDS = 0.1
_DEVICE_OPTION_KEYS = ("device", "provider", "execution_provider", "backend", "compute_units", "precision")


@dataclass(frozen=True)
class SessionKey:
    """Identity of one loaded model runtime."""

    engine: str
    model_path: str = ""
    device: str = ""
    options: str = ""


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        items = [_json_safe(item) for item in value]
        return None if any(item is _SKIP for item in items) else items
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            safe = _json_safe(item)
            if safe is not _SKIP:
                out[str(key)] = safe
        return out
    return _SKIP


_SKIP = object()


def options_fingerprint(engine_options: dict | None) -> str:
    """Stable digest of the start-up relevant engine options.

    ``column_*`` keys belong to the masked-column orchestration layer (targets,
    callbacks, cache holders) and never reach a worker's model start-up, so
    they are excluded; callables and other non-JSON values are skipped too.
    """
    relevant = {}
    for key, value in sorted((engine_options or {}).items(), key=lambda item: str(item)):
        name = str(key)
        if name.startswith("column_"):
            continue
        safe = _json_safe(value)
        if safe is not _SKIP:
            relevant[name] = safe
    payload = json.dumps(relevant, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _device_label(engine_options: dict | None) -> str:
    options = engine_options or {}
    parts = [f"{key}={options.get(key)}" for key in _DEVICE_OPTION_KEYS if options.get(key) not in (None, "")]
    return ",".join(parts)


def model_location(engine: str) -> str:
    """Model path for ``engine`` as resolved by the shared ModelManager."""
    try:
        from utils.model_manager import default_manager, specs_for_engine

        manager = default_manager()
        specs = specs_for_engine(str(engine))
        if specs:
            return "|".join(sorted({str(manager.target_dir(spec)) for spec in specs}))
        return str(manager.models_dir)
    except Exception:
        return ""


def session_key_for(engine: str, engine_options: dict | None = None) -> SessionKey:
    """Build the cache key for one engine/options combination."""
    return SessionKey(
        engine=str(engine or "").strip().lower(),
        model_path=model_location(engine),
        device=_device_label(engine_options),
        options=options_fingerprint(engine_options),
    )


def close_session(session: Any) -> None:
    """Close a session through its context-manager or ``close`` contract."""
    exit_method = getattr(session, "__exit__", None)
    try:
        if callable(exit_method):
            exit_method(None, None, None)
            return
        close = getattr(session, "close", None)
        if callable(close):
            close()
    except Exception:
        pass


def session_alive(session: Any) -> bool:
    """Best-effort liveness check of the worker process behind ``session``."""
    for attribute in ("proc", "process"):
        if hasattr(session, attribute):
            process = getattr(session, attribute)
            if process is None:
                return False
            poll = getattr(process, "poll", None)
            return not callable(poll) or poll() is None
    if hasattr(session, "backend"):
        return getattr(session, "backend") is not None
    return True


def rebind_session(session: Any, *, cancel_check=None, verbose: bool | None = None) -> None:
    """Point a reused session (and Apple helper backends) at the current job."""
    targets = [session]
    backend = getattr(session, "backend", None)
    if backend is not None:
        targets.append(backend)
        targets.extend(
            candidate for candidate in (
                getattr(backend, "_live_text", None),
                getattr(backend, "_native", None),
                getattr(backend, "_shortcut", None),
            ) if candidate is not None
        )
    for target in targets:
        if hasattr(target, "cancel_check"):
            try:
                setattr(target, "cancel_check", cancel_check)
            except Exception:
                pass
    if verbose is not None and hasattr(session, "verbose"):
        try:
            setattr(session, "verbose", bool(verbose))
        except Exception:
            pass


def _cancellation_types() -> tuple[type[BaseException], ...]:
    types: list[type[BaseException]] = [InterruptedError]
    try:
        from adapters.subprocess_watchdog import ProcessCancelled

        types.append(ProcessCancelled)
    except Exception:
        pass
    return tuple(types)


def is_cancellation(exc: BaseException | None) -> bool:
    """True when ``exc`` (or an explicit ``raise ... from`` cause) is a Stop."""
    cancel_types = _cancellation_types()
    seen = 0
    while exc is not None and seen < 8:
        if isinstance(exc, cancel_types):
            return True
        exc = exc.__cause__
        seen += 1
    return False


def _capacity_from_env() -> int:
    raw = str(os.environ.get(ENV_CACHE_SIZE, "") or "").strip()
    if not raw:
        return DEFAULT_CAPACITY
    try:
        return max(0, min(_MAX_CAPACITY, int(raw)))
    except ValueError:
        return DEFAULT_CAPACITY


class _Entry:
    __slots__ = ("key", "session", "ready", "error", "evicted", "lease_lock", "owner", "depth")

    def __init__(self, key: SessionKey) -> None:
        self.key = key
        self.session: Any = None
        self.ready = threading.Event()
        self.error: BaseException | None = None
        self.evicted = False
        self.lease_lock = threading.Lock()
        self.owner: int | None = None
        self.depth = 0


class RecognizerSessionCache:
    """Thread-safe LRU of loaded sessions; see the module docstring."""

    def __init__(self, capacity: int | None = None, *,
                 closer: Callable[[Any], None] = close_session,
                 is_alive: Callable[[Any], bool] = session_alive) -> None:
        self._capacity = _capacity_from_env() if capacity is None else max(0, int(capacity))
        self._closer = closer
        self._is_alive = is_alive
        self._guard = threading.RLock()
        self._entries: OrderedDict[SessionKey, _Entry] = OrderedDict()
        self._stats = {"hits": 0, "loads": 0, "evictions": 0, "invalidations": 0, "cancel_reuses": 0}

    @property
    def capacity(self) -> int:
        return self._capacity

    def set_capacity(self, capacity: int) -> None:
        with self._guard:
            self._capacity = max(0, int(capacity))
        self._evict_over_capacity()

    def __len__(self) -> int:
        with self._guard:
            return len(self._entries)

    def keys(self) -> list[SessionKey]:
        with self._guard:
            return list(self._entries)

    def metrics(self) -> dict[str, int]:
        with self._guard:
            return {**self._stats, "entries": len(self._entries), "capacity": self._capacity}

    # -- loading -----------------------------------------------------------

    def get(self, key: SessionKey, loader: Callable[[], Any], *, cancel_check=None) -> Any:
        """Return a shared session; use ``lease`` for exclusive access."""
        entry = self._entry_for(key, loader, cancel_check=cancel_check)
        _check_cancel(cancel_check)
        return entry.session

    def _entry_for(self, key: SessionKey, loader: Callable[[], Any], *,
                   cancel_check=None) -> _Entry:
        _check_cancel(cancel_check)
        with self._guard:
            entry = self._entries.get(key)
            if entry is None:
                entry = _Entry(key)
                self._entries[key] = entry
                owner = True
            else:
                self._entries.move_to_end(key)
                owner = False
        if not owner:
            while True:
                _check_cancel(cancel_check)
                if entry.ready.wait(timeout=_LEASE_POLL_SECONDS):
                    break
            _check_cancel(cancel_check)
            with self._guard:
                if entry.error is not None:
                    raise entry.error
                self._stats["hits"] += 1
            return entry
        try:
            _check_cancel(cancel_check)
            # Arbitrary loaders cannot be interrupted here. A completed load
            # remains shared even if this client's cancellation arrived meanwhile.
            session = loader()
        except BaseException as exc:
            with self._guard:
                if entry.error is None:
                    entry.error = exc
                if self._entries.get(key) is entry:
                    del self._entries[key]
                entry.ready.set()
            raise
        with self._guard:
            self._stats["loads"] += 1
            invalidated = entry.evicted
            if not invalidated:
                entry.session = session
            entry.ready.set()
        if invalidated:
            self._closer(session)
            raise entry.error or OcrCancelled("OCR recognizer cache was invalidated")
        self._evict_over_capacity(keep=key)
        _check_cancel(cancel_check)
        return entry

    # -- leasing -----------------------------------------------------------

    @contextlib.contextmanager
    def lease(self, key: SessionKey, loader: Callable[[], Any], *, cancel_check=None,
              rebind: Callable[[Any], None] | None = None) -> Iterator[Any]:
        """Exclusively use a session, cooperatively cancelling waits and handoff."""
        _check_cancel(cancel_check)
        if self._capacity <= 0:
            session = loader()
            try:
                _check_cancel(cancel_check)
                if rebind is not None:
                    rebind(session)
                _check_cancel(cancel_check)
                yield session
            finally:
                self._closer(session)
            return

        me = threading.get_ident()
        while True:
            entry = self._entry_for(key, loader, cancel_check=cancel_check)
            _check_cancel(cancel_check)
            if entry.owner == me:
                entry.depth += 1
                try:
                    _check_cancel(cancel_check)
                    yield entry.session
                finally:
                    entry.depth -= 1
                return
            while True:
                _check_cancel(cancel_check)
                if entry.error is not None:
                    raise entry.error
                if entry.lease_lock.acquire(timeout=_LEASE_POLL_SECONDS):
                    break
            usable = False
            try:
                _check_cancel(cancel_check)
                if entry.error is not None:
                    raise entry.error
                usable = not entry.evicted and self._is_alive(entry.session)
            finally:
                if not usable:
                    entry.lease_lock.release()
            if not usable:
                self._discard(entry)
                continue
            break

        entry.owner = me
        entry.depth = 1
        failed = False
        entered = False
        try:
            _check_cancel(cancel_check)
            if rebind is not None:
                rebind(entry.session)
            _check_cancel(cancel_check)
            if entry.error is not None:
                raise entry.error
            entered = True
            yield entry.session
        except BaseException as exc:
            # Cancellation before handoff must not discard another client's
            # healthy worker, including sessions opting out of in-body reuse.
            failed = (
                not self._reusable_after_cancel(entry.session, exc)
                if entered else not is_cancellation(exc)
            )
            raise
        finally:
            entry.owner = None
            entry.depth = 0
            discard = failed or entry.evicted
            if discard:
                # Mark unavailable before unlocking, so a waiter cannot enter
                # a failed session between release and removal.
                self._discard(entry)
            entry.lease_lock.release()
            if discard:
                self._discard(entry)

    # -- eviction / invalidation ------------------------------------------

    def _reusable_after_cancel(self, session: Any, exc: BaseException) -> bool:
        if not is_cancellation(exc) or getattr(session, "discard_on_cancel", False):
            return False
        try:
            alive = bool(self._is_alive(session))
        except Exception:
            alive = False
        if alive:
            with self._guard:
                self._stats["cancel_reuses"] += 1
        return alive

    def _discard(self, entry: _Entry) -> None:
        with self._guard:
            if self._entries.get(entry.key) is entry:
                del self._entries[entry.key]
            already = entry.evicted and entry.session is None
            entry.evicted = True
        if already:
            return
        if entry.lease_lock.acquire(blocking=False):
            try:
                session, entry.session = entry.session, None
            finally:
                entry.lease_lock.release()
            if session is not None:
                self._closer(session)

    def _evict_over_capacity(self, keep: SessionKey | None = None) -> None:
        victims: list[_Entry] = []
        with self._guard:
            ready = [key for key, entry in self._entries.items() if entry.ready.is_set() and key != keep]
            while len(self._entries) > self._capacity and ready:
                victim = self._entries.pop(ready.pop(0))
                victim.evicted = True
                victims.append(victim)
                self._stats["evictions"] += 1
        for victim in victims:
            self._discard(victim)

    def invalidate(self, engine: str | None = None,
                   predicate: Callable[[SessionKey], bool] | None = None) -> int:
        """Invalidate without waiting for loaders or active leases.

        Waiters wake with ``OcrCancelled``. A running loader is not interrupted;
        its result is closed when it returns. New clients may load fresh entries.
        """
        wanted = str(engine or "").strip().lower()
        with self._guard:
            victims = [
                entry for key, entry in self._entries.items()
                if (not wanted or key.engine == wanted) and (predicate is None or predicate(key))
            ]
            for entry in victims:
                del self._entries[entry.key]
                entry.evicted = True
                entry.error = OcrCancelled("OCR recognizer cache was invalidated")
                entry.ready.set()
            self._stats["invalidations"] += len(victims)
        for entry in victims:
            self._discard(entry)
        return len(victims)

    def clear(self) -> int:
        return self.invalidate()


_DEFAULT_CACHE: RecognizerSessionCache | None = None
_DEFAULT_CACHE_LOCK = threading.Lock()


def default_session_cache() -> RecognizerSessionCache:
    """The process-wide cache shared by every OCR worker thread."""
    global _DEFAULT_CACHE
    with _DEFAULT_CACHE_LOCK:
        if _DEFAULT_CACHE is None:
            _DEFAULT_CACHE = RecognizerSessionCache()
            atexit.register(_DEFAULT_CACHE.clear)
        return _DEFAULT_CACHE


def invalidate_recognizer_sessions(engine: str | None = None) -> int:
    """Close cached sessions (all, or one engine); safe to call from the GUI."""
    with _DEFAULT_CACHE_LOCK:
        cache = _DEFAULT_CACHE
    return cache.invalidate(engine) if cache is not None else 0


__all__ = [
    "DEFAULT_CAPACITY",
    "ENV_CACHE_SIZE",
    "RecognizerSessionCache",
    "SessionKey",
    "close_session",
    "default_session_cache",
    "invalidate_recognizer_sessions",
    "is_cancellation",
    "model_location",
    "options_fingerprint",
    "rebind_session",
    "session_alive",
    "session_key_for",
]
