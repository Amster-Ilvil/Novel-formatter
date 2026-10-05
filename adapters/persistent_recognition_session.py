#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistent JSONL recognition workers for targeted glyph rescue.

NDLOCR-Lite and PaddleOCR model startup dominates inference when a worker is
created once per column.  This session keeps one worker alive for the complete
handwriting-fusion run and sends small batches over stdin.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import threading
import os
from typing import Iterator


class PersistentRecognitionSession:
    def __init__(self, *, engine: str, engine_options: dict | None = None):
        self.engine = str(engine or "")
        self.options = dict(engine_options or {})
        self._proc: subprocess.Popen | None = None
        self._stdout_pump = None
        self._stderr_file = None
        self._request_id = 0
        self._lock = threading.Lock()
        self._paddle_source_index = 0
        self._idle_timer: threading.Timer | None = None
        self._idle_timer_lock = threading.Lock()

    @property
    def supported(self) -> bool:
        return self.engine in {"ndlocr_lite", "paddle_ocr"}

    def _cancel_idle_unload(self) -> None:
        with self._idle_timer_lock:
            timer = self._idle_timer
            self._idle_timer = None
        if timer is not None:
            try:
                timer.cancel()
            except Exception:
                pass

    def _schedule_idle_unload(self) -> None:
        """Release heavyweight worker models after a configurable idle period."""
        self._cancel_idle_unload()
        try:
            seconds = float(os.environ.get("NOVEL_FORMATTER_OCR_IDLE_UNLOAD_SECONDS", "300") or 300)
        except (TypeError, ValueError):
            seconds = 300.0
        if seconds <= 0:
            return
        seconds = max(0.05, min(seconds, 86400.0))

        def unload() -> None:
            # Never kill a worker while recognize() owns the session lock.  If a
            # new request started just before the timer fired, reschedule.
            acquired = self._lock.acquire(blocking=False)
            if not acquired:
                self._schedule_idle_unload()
                return
            try:
                self._close_process_only()
            finally:
                self._lock.release()
                with self._idle_timer_lock:
                    self._idle_timer = None

        timer = threading.Timer(seconds, unload)
        timer.daemon = True
        with self._idle_timer_lock:
            self._idle_timer = timer
        timer.start()

    def _command(self) -> list[str]:
        if self.engine == "ndlocr_lite":
            from adapters.ndlocr_lite_adapter import setup_venv, WORKER_SCRIPT
            python, source = setup_venv(verbose=False)
            return [
                str(python), str(WORKER_SCRIPT),
                "--source-root", str(source), "--server",
            ]
        if self.engine == "paddle_ocr":
            from adapters.paddle_ocr_adapter import (
                setup_venv, VENV_PYTHON, WORKER_SCRIPT, _prepare_vl_runtime,
                recommended_paddle_batch, _paddle_worker_env,
            )
            pipeline = str(self.options.get("pipeline") or "ocr")
            if pipeline not in {"ocr", "structure", "vl"}:
                pipeline = "ocr"
            setup_venv(verbose=False, pipeline=pipeline)
            vl_runtime = _prepare_vl_runtime(
                pipeline,
                str(self.options.get("vl_backend") or "auto"),
                verbose=False,
            )
            command = [
                str(VENV_PYTHON), str(WORKER_SCRIPT),
                "--lang", str(self.options.get("lang") or "japan"),
                "--pipeline", pipeline,
                "--batch-size", str(recommended_paddle_batch(pipeline)),
            ]
            if pipeline == "vl":
                command.extend([
                    "--vl-backend", "mlx" if vl_runtime.get("backend") == "mlx" else "paddle",
                    "--vl-server-url", str(vl_runtime.get("server_url") or ""),
                    "--vl-api-model-name", str(
                        vl_runtime.get("model") or "PaddlePaddle/PaddleOCR-VL-1.6"
                    ),
                ])
            command.append("--server")
            return command
        raise RuntimeError(f"不支持持久会话的 OCR：{self.engine}")

    def _process_environment(self) -> dict[str, str] | None:
        if self.engine != "paddle_ocr":
            return None
        from adapters.paddle_ocr_models import (
            paddle_model_source_attempts,
            paddle_source_environment,
        )
        sources = paddle_model_source_attempts(str(self.options.get("model_source") or "auto"))
        source = sources[min(self._paddle_source_index, len(sources) - 1)]
        return _paddle_worker_env(source)

    def _ensure_started(self) -> None:
        self._cancel_idle_unload()
        if self._proc is not None and self._proc.poll() is None:
            return
        if not self.supported:
            raise RuntimeError(f"OCR {self.engine} 不支持持久字框会话")
        self.close()
        self._stderr_file = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
        from adapters.subprocess_watchdog import LinePump, isolated_process_kwargs
        self._proc = subprocess.Popen(
            self._command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file,
            text=True,
            bufsize=1,
            env=self._process_environment(),
            **isolated_process_kwargs(),
        )
        self._stdout_pump = LinePump(self._proc.stdout, name=f"{self.engine}-persistent-stdout")

    def recognize(self, image_paths: list[str]) -> Iterator[tuple[str, list[dict] | None, str | None]]:
        if not image_paths:
            return iter(())

        def iterator():
            with self._lock:
                attempts = 1
                if self.engine == "paddle_ocr":
                    from adapters.paddle_ocr_models import paddle_model_source_attempts
                    attempts = len(paddle_model_source_attempts(str(self.options.get("model_source") or "auto")))
                last_error = ""
                for attempt in range(attempts):
                    emitted = False
                    try:
                        self._ensure_started()
                        assert self._proc is not None and self._proc.stdin is not None and self._proc.stdout is not None
                        from adapters.subprocess_watchdog import (
                            LinePump, env_seconds, workload_timeout,
                        )
                        if self._stdout_pump is None:
                            self._stdout_pump = LinePump(
                                self._proc.stdout, name=f"{self.engine}-persistent-stdout"
                            )
                        self._request_id += 1
                        request_id = self._request_id
                        payload = {"request_id": request_id, "images": [str(Path(p)) for p in image_paths]}
                        self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
                        self._proc.stdin.flush()
                        if (
                            self.engine == "paddle_ocr"
                            and str(self.options.get("pipeline") or "ocr") == "vl"
                            and request_id == 1
                        ):
                            # The first MLX-VLM request may also download/load
                            # PaddleOCR-VL-1.6. Reuse the normal model-startup
                            # deadline instead of the shorter steady-state page deadline.
                            request_timeout = env_seconds(
                                "NOVEL_FORMATTER_OCR_STARTUP_TIMEOUT", 900.0, minimum=60.0
                            )
                        else:
                            base_timeout = env_seconds(
                                "NOVEL_FORMATTER_OCR_REQUEST_TIMEOUT", 300.0, minimum=30.0
                            )
                            per_mp = env_seconds(
                                "NOVEL_FORMATTER_OCR_TIMEOUT_PER_MEGAPIXEL",
                                2.0, minimum=0.1, maximum=60.0,
                            )
                            max_mp = 0.0
                            try:
                                from adapters.ocr_engine_common import page_size
                                for image_path in image_paths:
                                    width, height = page_size(image_path)
                                    max_mp = max(max_mp, width * height / 1_000_000.0)
                            except Exception:
                                pass
                            request_timeout = workload_timeout(
                                base_timeout,
                                megapixels=max_mp,
                                seconds_per_megapixel=per_mp,
                            )
                        while True:
                            line = self._stdout_pump.readline(
                                proc=self._proc,
                                timeout=request_timeout,
                                label=f"{self.engine} 持久 worker",
                            )
                            if line is None:
                                tail = self.stderr_tail()
                                raise RuntimeError(
                                    f"{self.engine} 持久 worker 提前退出"
                                    + (f"：{tail}" if tail else "")
                                )
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                data = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if data.get("type") == "ready":
                                continue
                            if int(data.get("request_id", request_id) or request_id) != request_id:
                                continue
                            if data.get("batch_done") or data.get("type") == "request_done":
                                self._schedule_idle_unload()
                                return
                            emitted = True
                            path = str(data.get("path") or "")
                            if data.get("ok"):
                                yield path, list(data.get("blocks") or []), None
                            else:
                                yield path, None, str(data.get("error") or "未知错误")
                    except Exception as exc:
                        last_error = str(exc)
                        if emitted or self.engine != "paddle_ocr" or attempt + 1 >= attempts:
                            raise
                        self.close()
                        self._paddle_source_index += 1
                        continue
                if last_error:
                    raise RuntimeError(last_error)

        return iterator()

    def stderr_tail(self, limit: int = 2000) -> str:
        if self._stderr_file is None:
            return ""
        try:
            self._stderr_file.flush()
            self._stderr_file.seek(0)
            return self._stderr_file.read()[-limit:]
        except Exception:
            return ""

    def _close_process_only(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                if proc.poll() is None and proc.stdin is not None:
                    proc.stdin.write(json.dumps({"command": "close"}) + "\n")
                    proc.stdin.flush()
                    proc.wait(timeout=3)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass
            stdout_pump = self._stdout_pump
            self._stdout_pump = None
            if stdout_pump is not None:
                stdout_pump.close()
            for stream in (proc.stdin, proc.stdout):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass
        if self._stderr_file is not None:
            try:
                self._stderr_file.close()
            except Exception:
                pass
            self._stderr_file = None

    def close(self) -> None:
        self._cancel_idle_unload()
        self._close_process_only()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
