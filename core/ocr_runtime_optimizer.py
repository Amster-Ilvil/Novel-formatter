# -*- coding: utf-8 -*-
"""Non-semantic OCR runtime optimization helpers.

The classes in this module only coordinate work, cancellation, UI batching and
performance telemetry.  They never inspect or transform OCR pixels/text, and
therefore can be used without changing model input/output contracts.
"""
from __future__ import annotations

from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
import threading
import time
from typing import Callable, Deque, Iterable, Iterator, Mapping, MutableMapping, TypeVar
from concurrent.futures import TimeoutError as FutureTimeout

T = TypeVar("T")
R = TypeVar("R")


class OcrCancelled(InterruptedError):
    """Raised at an explicit cancellation checkpoint."""


class OcrCancellationToken:
    """Thread-safe cancellation facade backed by the GUI's existing Event."""

    def __init__(self, event: threading.Event | None = None):
        self._event = event or threading.Event()
        self._reason = "OCR 已停止"
        self._lock = threading.Lock()

    @property
    def event(self) -> threading.Event:
        return self._event

    def cancel(self, reason: str = "OCR 已停止") -> None:
        with self._lock:
            self._reason = str(reason or "OCR 已停止")
            self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def checkpoint(self) -> None:
        if self._event.is_set():
            with self._lock:
                reason = self._reason
            raise OcrCancelled(reason)


@dataclass(frozen=True)
class OcrRuntimeLimits:
    """Conservative resource limits for one application process."""

    mps_model: int = 1
    onnx_model: int = 1
    native_vision: int = 1
    remote_model: int = 2
    alignment: int = 1
    image_prepare: int = 4
    image_encode: int = 2

    def as_dict(self) -> dict[str, int]:
        return {
            "mps_model": max(1, int(self.mps_model)),
            "onnx_model": max(1, int(self.onnx_model)),
            "native_vision": max(1, int(self.native_vision)),
            "remote_model": max(1, int(self.remote_model)),
            "alignment": max(1, int(self.alignment)),
            "image_prepare": max(1, int(self.image_prepare)),
            "image_encode": max(1, int(self.image_encode)),
        }


class OcrResourceGovernor:
    """Bounded resource leases with cancellation-aware waits and metrics."""

    def __init__(self, limits: OcrRuntimeLimits | Mapping[str, int] | None = None):
        if limits is None:
            capacities = OcrRuntimeLimits().as_dict()
        elif isinstance(limits, OcrRuntimeLimits):
            capacities = limits.as_dict()
        else:
            capacities = {str(k): max(1, int(v)) for k, v in limits.items()}
        self._capacities = capacities
        self._semaphores = {name: threading.BoundedSemaphore(value) for name, value in capacities.items()}
        self._lock = threading.Lock()
        self._active: MutableMapping[str, int] = defaultdict(int)
        self._max_active: MutableMapping[str, int] = defaultdict(int)
        self._lease_count: MutableMapping[str, int] = defaultdict(int)
        self._wait_seconds: MutableMapping[str, float] = defaultdict(float)

    @contextmanager
    def lease(
        self,
        resource: str,
        *,
        cancel_check: Callable[[], bool] | None = None,
        poll_seconds: float = 0.05,
    ) -> Iterator[None]:
        name = str(resource or "").strip() or "default"
        semaphore = self._semaphores.get(name)
        if semaphore is None:
            with self._lock:
                capacity = self._capacities.setdefault(name, 1)
                semaphore = self._semaphores.setdefault(name, threading.BoundedSemaphore(capacity))
        started = time.perf_counter()
        while not semaphore.acquire(timeout=max(0.01, float(poll_seconds))):
            if callable(cancel_check) and cancel_check():
                raise OcrCancelled("等待 OCR 资源时已停止")
        waited = max(0.0, time.perf_counter() - started)
        with self._lock:
            self._wait_seconds[name] += waited
            self._lease_count[name] += 1
            self._active[name] += 1
            self._max_active[name] = max(self._max_active[name], self._active[name])
        try:
            if callable(cancel_check) and cancel_check():
                raise OcrCancelled("取得 OCR 资源后已停止")
            yield
        finally:
            with self._lock:
                self._active[name] = max(0, self._active[name] - 1)
            semaphore.release()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            names = sorted(set(self._capacities) | set(self._lease_count))
            return {
                "capacities": dict(self._capacities),
                "resources": {
                    name: {
                        "lease_count": int(self._lease_count.get(name, 0)),
                        "wait_seconds": float(self._wait_seconds.get(name, 0.0)),
                        "max_active": int(self._max_active.get(name, 0)),
                    }
                    for name in names
                },
            }


class CoalescedLineBuffer:
    """Thread-safe line buffer drained by one low-frequency GUI timer."""

    def __init__(self, *, max_lines: int = 4000):
        self.max_lines = max(32, int(max_lines))
        self._lines: Deque[str] = deque()
        self._lock = threading.Lock()
        self.dropped_lines = 0
        self.pushed_lines = 0
        self.flush_count = 0

    def push(self, line: object) -> None:
        value = str(line or "")
        if not value:
            return
        with self._lock:
            self.pushed_lines += 1
            self._lines.append(value)
            while len(self._lines) > self.max_lines:
                self._lines.popleft()
                self.dropped_lines += 1

    def drain(self, *, max_lines: int = 320) -> list[str]:
        limit = max(1, int(max_lines))
        with self._lock:
            if not self._lines:
                return []
            result = [self._lines.popleft() for _ in range(min(limit, len(self._lines)))]
            self.flush_count += 1
            return result

    def clear(self) -> None:
        with self._lock:
            self._lines.clear()

    def pending(self) -> int:
        with self._lock:
            return len(self._lines)

    def metrics(self) -> dict[str, int]:
        with self._lock:
            return {
                "pushed_lines": self.pushed_lines,
                "flush_count": self.flush_count,
                "pending_lines": len(self._lines),
                "dropped_lines": self.dropped_lines,
            }


@dataclass
class _StageMetric:
    calls: int = 0
    total_seconds: float = 0.0
    max_seconds: float = 0.0


class OcrPerformanceTrace:
    """Low-overhead per-run timing/counter report."""

    def __init__(self, *, run_id: str, metadata: Mapping[str, object] | None = None):
        self.run_id = str(run_id)
        self.started_at = time.time()
        self.started_monotonic = time.perf_counter()
        self.metadata = dict(metadata or {})
        self._stages: MutableMapping[str, _StageMetric] = defaultdict(_StageMetric)
        self._counters: MutableMapping[str, int] = defaultdict(int)
        self._gauges: dict[str, float | int | str | bool | None] = {}
        self._events: list[dict[str, object]] = []
        self._lock = threading.RLock()
        self._finished_at: float | None = None

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        key = str(name or "stage")
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = max(0.0, time.perf_counter() - started)
            with self._lock:
                metric = self._stages[key]
                metric.calls += 1
                metric.total_seconds += elapsed
                metric.max_seconds = max(metric.max_seconds, elapsed)

    def add_duration(self, name: str, seconds: float) -> None:
        elapsed = max(0.0, float(seconds))
        with self._lock:
            metric = self._stages[str(name)]
            metric.calls += 1
            metric.total_seconds += elapsed
            metric.max_seconds = max(metric.max_seconds, elapsed)

    def increment(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[str(name)] += int(value)

    def set_gauge(self, name: str, value) -> None:
        with self._lock:
            self._gauges[str(name)] = value

    def event(self, name: str, **detail) -> None:
        with self._lock:
            if len(self._events) >= 256:
                return
            self._events.append({
                "at_seconds": max(0.0, time.perf_counter() - self.started_monotonic),
                "name": str(name),
                **detail,
            })

    def finish(self) -> dict[str, object]:
        with self._lock:
            if self._finished_at is None:
                self._finished_at = time.time()
            return self.snapshot()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            ended = self._finished_at or time.time()
            return {
                "schema": "novel_formatter.ocr_performance_trace.v1",
                "run_id": self.run_id,
                "started_at_unix": self.started_at,
                "finished_at_unix": ended,
                "elapsed_seconds": max(0.0, time.perf_counter() - self.started_monotonic),
                "metadata": dict(self.metadata),
                "stages": {
                    name: {
                        "calls": metric.calls,
                        "total_seconds": metric.total_seconds,
                        "max_seconds": metric.max_seconds,
                    }
                    for name, metric in sorted(self._stages.items())
                },
                "counters": dict(sorted(self._counters.items())),
                "gauges": dict(sorted(self._gauges.items())),
                "events": list(self._events),
            }

    def write_json(self, path: str | Path, *, extra: Mapping[str, object] | None = None) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.finish()
        if extra:
            payload["runtime"] = dict(extra)
        tmp = target.with_name(f".{target.name}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(target)
        return target


def iter_bounded_ordered(
    executor,
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    max_pending: int,
    cancel_check: Callable[[], bool] | None = None,
) -> Iterator[R]:
    """Submit at most ``max_pending`` jobs while yielding in input order.

    Unlike ``[executor.submit(...) for item in all_items]``, this keeps memory
    bounded and prevents hundreds of already-queued page preparations from
    delaying cancellation.  Result ordering remains exactly the input ordering.
    """

    iterator = iter(items)
    pending: Deque[object] = deque()
    limit = max(1, int(max_pending))

    def cancelled() -> bool:
        return bool(callable(cancel_check) and cancel_check())

    for _ in range(limit):
        if cancelled():
            raise OcrCancelled("提交 OCR 预处理任务时已停止")
        try:
            item = next(iterator)
        except StopIteration:
            break
        pending.append(executor.submit(fn, item))

    while pending:
        if cancelled():
            for future in pending:
                future.cancel()
            raise OcrCancelled("等待 OCR 预处理任务时已停止")
        future = pending.popleft()
        while True:
            if cancelled():
                future.cancel()
                for queued in pending:
                    queued.cancel()
                raise OcrCancelled("等待 OCR 预处理任务时已停止")
            try:
                result = future.result(timeout=0.10)
                break
            except FutureTimeout:
                continue
        yield result
        if cancelled():
            for item in pending:
                item.cancel()
            raise OcrCancelled("OCR 预处理任务已停止")
        try:
            item = next(iterator)
        except StopIteration:
            continue
        pending.append(executor.submit(fn, item))
