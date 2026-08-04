# -*- coding: utf-8 -*-
"""Thread-safe, one-clone snapshot alias registry.

A publication may have several semantic aliases (for example ``formatter`` and
``current``). The source object is cloned exactly once, aliases share that
frozen snapshot, and mutable consumers explicitly request their own copy.
"""
from __future__ import annotations

import copy
from threading import RLock
from typing import Generic, Iterable, MutableMapping, TypeVar

T = TypeVar("T")
_MISSING = object()


class WorkspaceSnapshotRegistry(Generic[T]):
    def __init__(self) -> None:
        self.aliases: MutableMapping[str, T] = {}
        self._lock = RLock()
        self._revision = 0
        self._alias_revisions: dict[str, int] = {}

    @staticmethod
    def _normalise_key(key: object) -> str:
        value = str(key or "").strip()
        if not value:
            raise ValueError("snapshot key cannot be empty")
        return value

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    def clear(self) -> None:
        with self._lock:
            self.aliases.clear()
            self._alias_revisions.clear()
            self._revision += 1

    def publish(self, key: str, value: T, *, clone: bool = True) -> T:
        snapshot_key = self._normalise_key(key)
        snapshot = copy.deepcopy(value) if clone else value
        with self._lock:
            self._revision += 1
            self.aliases[snapshot_key] = snapshot
            self._alias_revisions[snapshot_key] = self._revision
        return snapshot

    def publish_aliases(self, keys: Iterable[str], value: T, *, clone: bool = True) -> T:
        normalised = tuple(dict.fromkeys(self._normalise_key(key) for key in keys))
        if not normalised:
            raise ValueError("at least one snapshot alias is required")
        # Keep the expensive/deep operation outside the registry lock. The
        # resulting snapshot is committed to every alias in one critical section.
        snapshot = copy.deepcopy(value) if clone else value
        with self._lock:
            self._revision += 1
            revision = self._revision
            for key in normalised:
                self.aliases[key] = snapshot
                self._alias_revisions[key] = revision
        return snapshot

    def get(self, key: str, default=None):
        snapshot_key = self._normalise_key(key)
        with self._lock:
            return self.aliases.get(snapshot_key, default)

    def require(self, key: str) -> T:
        snapshot_key = self._normalise_key(key)
        with self._lock:
            value = self.aliases.get(snapshot_key, _MISSING)
        if value is _MISSING:
            raise KeyError(snapshot_key)
        return value  # type: ignore[return-value]

    def clone_for(self, key: str) -> T:
        return self.clone(self.require(key))

    def remove(self, key: str) -> T | None:
        snapshot_key = self._normalise_key(key)
        with self._lock:
            value = self.aliases.pop(snapshot_key, None)
            self._alias_revisions.pop(snapshot_key, None)
            if value is not None:
                self._revision += 1
            return value

    def alias_revision(self, key: str) -> int | None:
        snapshot_key = self._normalise_key(key)
        with self._lock:
            return self._alias_revisions.get(snapshot_key)

    @staticmethod
    def clone(value: T) -> T:
        return copy.deepcopy(value)
