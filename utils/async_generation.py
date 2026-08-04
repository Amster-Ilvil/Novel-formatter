# -*- coding: utf-8 -*-
"""Thread-safe generation tokens for discard-on-stale background work."""
from __future__ import annotations

from threading import Lock


class GenerationGuard:
    """Monotonic token source used to isolate late background results."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._generation = 0

    def begin(self) -> int:
        """Invalidate older work and return the token for a new operation."""
        with self._lock:
            self._generation += 1
            return self._generation

    def invalidate(self) -> int:
        """Invalidate all previously issued tokens without starting new work."""
        return self.begin()

    def invalidate_if_current(self, token: int) -> bool:
        """Invalidate ``token`` only when it still owns the active generation."""
        try:
            candidate = int(token)
        except (TypeError, ValueError, OverflowError):
            return False
        with self._lock:
            if candidate != self._generation:
                return False
            self._generation += 1
            return True

    def is_current(self, token: int) -> bool:
        try:
            candidate = int(token)
        except (TypeError, ValueError, OverflowError):
            return False
        with self._lock:
            return candidate == self._generation

    @property
    def current(self) -> int:
        with self._lock:
            return self._generation
