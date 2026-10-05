#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded one-window-ahead preprocessing for accelerator OCR sessions."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def windowed(values: Sequence[T], size: int) -> list[tuple[int, Sequence[T]]]:
    step = max(1, int(size))
    return [(start, values[start:start + step]) for start in range(0, len(values), step)]


def iter_prefetched_windows(
    windows: Sequence[tuple[int, Sequence[T]]],
    prepare: Callable[[int, Sequence[T]], R],
    *,
    enabled: bool,
) -> Iterator[tuple[int, Sequence[T], R]]:
    """Yield prepared windows while preparing at most one future window.

    Preparation order and exceptions are identical to the serial path.  The
    single background future bounds memory and avoids oversubscribing unified
    memory while MPS/CUDA is busy with the current inference window.
    """
    if not windows:
        return
    if not enabled or len(windows) <= 1:
        for start, values in windows:
            yield start, values, prepare(start, values)
        return
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr-preprocess") as executor:
        start, values = windows[0]
        future: Future[R] = executor.submit(prepare, start, values)
        for position, (start, values) in enumerate(windows):
            prepared = future.result()
            if position + 1 < len(windows):
                next_start, next_values = windows[position + 1]
                future = executor.submit(prepare, next_start, next_values)
            yield start, values, prepared


def accelerator_prefetch_enabled(device: str, env_value: str | None = None) -> bool:
    raw = str(env_value or "auto").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    accelerated = any(token in str(device or "").lower() for token in ("mps", "cuda"))
    if raw in {"1", "true", "yes", "on"}:
        return accelerated
    return accelerated


__all__ = ["accelerator_prefetch_enabled", "iter_prefetched_windows", "windowed"]
