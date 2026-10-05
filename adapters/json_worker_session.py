#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reusable JSON-line subprocess session primitives for local OCR workers."""
from __future__ import annotations

import json
import subprocess
import threading


class JsonWorkerSessionBase:
    def __init__(self, *, cancel_check=None, verbose: bool = True, worker_label: str = "OCR"):
        self.cancel_check = cancel_check
        self.verbose = verbose
        self.worker_label = str(worker_label or "OCR")
        self.proc: subprocess.Popen[str] | None = None
        self._stderr_lines: list[str] = []
        self._stderr_thread: threading.Thread | None = None
        self._stderr_stop = threading.Event()
        self._stdout_pump = None
        self.device = ""
        self._request_id = 0

    def _drain_stderr(self, stderr_pipe) -> None:
        while not self._stderr_stop.is_set():
            try:
                line = stderr_pipe.readline()
            except (ValueError, OSError):
                break
            if not line:
                break
            self._stderr_lines.append(str(line).rstrip())
            if len(self._stderr_lines) > 200:
                del self._stderr_lines[:100]

    def _read_response(self, timeout: float | None = None, *, on_wait=None) -> dict:
        if self.proc is None or self.proc.stdout is None:
            raise RuntimeError(f"{self.worker_label} worker 尚未启动")
        from adapters.subprocess_watchdog import LinePump, env_seconds
        if self._stdout_pump is None:
            self._stdout_pump = LinePump(
                self.proc.stdout, name=self.worker_label.lower().replace(" ", "-") + "-stdout"
            )
        wait_seconds = (
            float(timeout) if timeout is not None
            else env_seconds("NOVEL_FORMATTER_OCR_REQUEST_TIMEOUT", 300.0, minimum=30.0)
        )
        line = self._stdout_pump.readline(
            proc=self.proc,
            timeout=wait_seconds,
            cancel_check=self.cancel_check,
            label=self.worker_label,
            on_wait=on_wait,
        )
        if line is None:
            ret = self.proc.poll()
            tail = "\n".join(self._stderr_lines[-30:])
            raise RuntimeError(f"{self.worker_label} worker 提前退出 (code={ret})\n{tail}")
        try:
            return json.loads(line)
        except Exception as exc:
            raise RuntimeError(
                f"{self.worker_label} worker 返回无效 JSON: {line[:300]} ({exc})"
            ) from exc

    def close(self, *, force: bool = False) -> None:
        proc = self.proc
        self.proc = None
        if proc is None:
            return
        initial_ret = proc.poll()
        shutdown_requested = False
        termination_requested = False
        ret = initial_ret
        try:
            if not force and initial_ret is None and proc.stdin is not None:
                proc.stdin.write(json.dumps({"command": "close"}) + "\n")
                proc.stdin.flush()
                shutdown_requested = True
        except Exception:
            pass
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        try:
            if force and proc.poll() is None:
                proc.terminate()
                termination_requested = True
            ret = proc.wait(timeout=12 if shutdown_requested else 5 if force else 1)
        except subprocess.TimeoutExpired:
            if proc.poll() is None:
                try:
                    proc.terminate()
                    termination_requested = True
                    ret = proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:
                        proc.kill()
                        termination_requested = True
                    except Exception:
                        pass
                    ret = proc.wait()
        except Exception:
            if proc.poll() is None:
                try:
                    proc.kill()
                    termination_requested = True
                except Exception:
                    pass
            ret = proc.wait()

        stderr_thread = self._stderr_thread
        self._stderr_stop.set()
        if stderr_thread is not None and stderr_thread.is_alive():
            stderr_thread.join(timeout=1.5)
        # Never close a TextIO wrapper while its background reader may still
        # be blocked in readline(); that can deadlock on TextIO's internal lock.
        # The worker process has already exited/been terminated above, so EOF
        # should release the daemon reader shortly without blocking teardown.
        if stderr_thread is None or not stderr_thread.is_alive():
            try:
                if proc.stderr:
                    proc.stderr.close()
            except Exception:
                pass
        self._stderr_thread = None

        stdout_pump = self._stdout_pump
        self._stdout_pump = None
        if stdout_pump is not None:
            stdout_pump.close()
        stdout_thread = getattr(stdout_pump, "thread", None) if stdout_pump is not None else None
        if stdout_thread is None or not stdout_thread.is_alive():
            try:
                if proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass

        intentional_shutdown = bool(force or shutdown_requested or termination_requested)
        if ret not in (0, -15) and not intentional_shutdown:
            tail = "\n".join(self._stderr_lines[-30:])
            raise RuntimeError(f"{self.worker_label} worker 异常退出 (code={ret}):\n{tail}")

    def __exit__(self, exc_type, exc, tb):
        try:
            self.close(force=exc is not None)
        except Exception:
            if exc is None:
                raise
        return False
