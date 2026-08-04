# -*- coding: utf-8 -*-
"""Thread-safe rolling RPM/TPM limiter for concurrent AI requests."""
from __future__ import annotations

import threading
import time
from collections import deque



class RequestLimiter:
    def __init__(self, rpm: int = 0, tpm: int = 0):
        self.rpm = max(0, int(rpm or 0))
        self.tpm = max(0, int(tpm or 0))
        self._requests: deque[float] = deque()
        self._tokens: deque[tuple[float, int]] = deque()
        self._token_total = 0
        self._condition = threading.Condition()

    def acquire(self, estimated_tokens: int = 0, cancel_check=None) -> None:
        if not self.rpm and not self.tpm:
            return

        # A single prompt can legitimately be larger than the configured TPM
        # estimate (for example after a model/provider limit is lowered).  The
        # old condition could never become true in that case and waited forever
        # even with an empty window.  Account it as one full TPM window instead:
        # the request proceeds once, then subsequent calls wait normally.
        requested_tokens = max(0, int(estimated_tokens or 0))
        accounted_tokens = min(requested_tokens, self.tpm) if self.tpm else requested_tokens
        while True:
            if cancel_check and cancel_check():
                raise RuntimeError("AI任务已停止")
            with self._condition:
                now = time.monotonic()
                cutoff = now - 60.0
                while self._requests and self._requests[0] <= cutoff:
                    self._requests.popleft()
                while self._tokens and self._tokens[0][0] <= cutoff:
                    _, value = self._tokens.popleft()
                    self._token_total -= value
                rpm_ok = not self.rpm or len(self._requests) < self.rpm
                tpm_ok = not self.tpm or self._token_total + accounted_tokens <= self.tpm
                if rpm_ok and tpm_ok:
                    self._requests.append(now)
                    if accounted_tokens:
                        self._tokens.append((now, accounted_tokens))
                        self._token_total += accounted_tokens
                    return
                deadlines = []
                if not rpm_ok and self._requests:
                    deadlines.append(self._requests[0] + 60.0)
                if not tpm_ok and self._tokens:
                    deadlines.append(self._tokens[0][0] + 60.0)
                timeout = max(0.05, min(deadlines) - now) if deadlines else 0.25
                self._condition.wait(timeout=min(timeout, 1.0))
