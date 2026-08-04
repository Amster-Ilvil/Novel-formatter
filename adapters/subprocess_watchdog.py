#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small, dependency-free watchdog helpers for OCR subprocess pipes.

The OCR GUI runs native/ML engines in child processes.  A blocking ``readline``
or an undrained stderr pipe can otherwise leave both parent and child waiting
forever.  These helpers keep pipe reads off the OCR orchestration thread, poll
cancellation, enforce an idle timeout, and terminate a stalled child cleanly.
"""
from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
from typing import Callable, TextIO


class ProcessCancelled(RuntimeError):
    """Raised when a cooperative cancellation request stops a child process."""


class ProcessStalledError(TimeoutError):
    """Raised when a child process produces no protocol output before timeout."""


def env_seconds(name: str, default: float, *, minimum: float = 1.0, maximum: float = 86400.0) -> float:
    try:
        value = float(os.environ.get(name, str(default)) or default)
    except (TypeError, ValueError):
        value = float(default)
    return max(float(minimum), min(float(maximum), value))


def isolated_process_kwargs() -> dict:
    """Return kwargs that let us terminate a worker and its descendants on POSIX."""
    if os.name == "posix":
        return {"start_new_session": True}
    return {}


def _poll(proc) -> int | None:
    try:
        return proc.poll()
    except Exception:
        return None


def _wait(proc, timeout: float | None = None) -> int | None:
    try:
        if timeout is None:
            return proc.wait()
        return proc.wait(timeout=timeout)
    except TypeError:
        # Lightweight test doubles often expose wait() without a timeout arg.
        return proc.wait()


def terminate_process(proc, *, grace: float = 2.0) -> int | None:
    """Terminate a subprocess without ever waiting indefinitely.

    Workers are normally launched in a dedicated session, so on POSIX we first
    signal the complete process group.  This also cleans up helper/model child
    processes that would otherwise survive after the OCR worker is stopped.
    """
    if proc is None:
        return None
    current = _poll(proc)
    if current is not None:
        return current

    signalled = False
    if os.name == "posix":
        try:
            import signal
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            signalled = True
        except Exception:
            pass
    if not signalled:
        try:
            proc.terminate()
        except Exception:
            pass

    try:
        return _wait(proc, max(0.1, float(grace)))
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        current = _poll(proc)
        if current is not None:
            return current

    killed = False
    if os.name == "posix":
        try:
            import signal
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGKILL)
            killed = True
        except Exception:
            pass
    if not killed:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        return _wait(proc, 2.0)
    except Exception:
        return _poll(proc)


class LinePump:
    """Read a text stream on a daemon thread and expose cancellable timed reads."""

    _EOF = object()

    def __init__(self, stream: TextIO | None, *, name: str):
        self.stream = stream
        self.name = name
        self.queue: queue.Queue[object] = queue.Queue()
        self._closed = threading.Event()
        self.thread: threading.Thread | None = None
        if stream is not None:
            self.thread = threading.Thread(target=self._reader, daemon=True, name=name)
            self.thread.start()
        else:
            self.queue.put(self._EOF)

    def _reader(self) -> None:
        try:
            assert self.stream is not None
            while not self._closed.is_set():
                try:
                    line = self.stream.readline()
                except (ValueError, OSError) as exc:
                    if not self._closed.is_set():
                        self.queue.put(exc)
                    break
                if not line:
                    break
                self.queue.put(line)
        except BaseException as exc:  # Preserve diagnostics from unexpected readers.
            if not self._closed.is_set():
                self.queue.put(exc)
        finally:
            self.queue.put(self._EOF)

    def get_nowait_lines(self, *, limit: int = 200) -> list[str]:
        lines: list[str] = []
        for _ in range(max(1, int(limit))):
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                break
            if item is self._EOF:
                # Keep EOF observable for a later blocking reader.
                self.queue.put(self._EOF)
                break
            if isinstance(item, BaseException):
                continue
            lines.append(str(item))
        return lines

    def readline(
        self,
        *,
        proc,
        timeout: float,
        cancel_check: Callable[[], bool] | None = None,
        label: str = "OCR worker",
        on_wait: Callable[[], None] | None = None,
    ) -> str | None:
        deadline = time.monotonic() + max(0.1, float(timeout))
        while True:
            if callable(cancel_check) and cancel_check():
                terminate_process(proc)
                raise ProcessCancelled(f"{label} 已取消")
            if callable(on_wait):
                try:
                    on_wait()
                except Exception:
                    pass
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                terminate_process(proc)
                raise ProcessStalledError(
                    f"{label} 连续 {max(1, int(timeout))} 秒没有返回协议数据，已自动终止以避免永久卡死"
                )
            try:
                item = self.queue.get(timeout=min(0.20, remaining))
            except queue.Empty:
                if _poll(proc) is not None and self.queue.empty():
                    return None
                continue
            if item is self._EOF:
                return None
            if isinstance(item, BaseException):
                raise RuntimeError(f"{label} 管道读取失败：{item}") from item
            return str(item)

    def close(self) -> None:
        self._closed.set()
        try:
            if self.stream is not None:
                self.stream.close()
        except Exception:
            pass
        thread = self.thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
