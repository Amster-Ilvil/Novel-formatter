#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YomiToku OCR adapter for Japanese printed novels.

The adapter deliberately uses only YomiToku's text detector and text
recognizer. Novel Formatter remains responsible for physical column splitting,
reading order, page continuation and formatter repairs.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Iterator

from adapters.runtime_env import ensure_venv

ROOT = Path(__file__).parent.parent
VENV_DIR = ROOT / ".venv-yomitoku"
WORKER_SCRIPT = Path(__file__).parent / "yomitoku_worker.py"
MODEL_CACHE = ROOT / ".model-cache" / "yomitoku"
YOMITOKU_VERSION = "0.13.1"
YOMITOKU_PACKAGE = os.environ.get(
    "NOVEL_FORMATTER_YOMITOKU_PACKAGE", f"yomitoku=={YOMITOKU_VERSION}"
)


def setup_venv(verbose: bool = True) -> Path:
    """Create or repair the isolated YomiToku runtime."""
    return ensure_venv(
        VENV_DIR,
        label="YomiToku OCR",
        marker_code=(
            "import importlib.metadata as m; import yomitoku, torch, cv2; "
            "from yomitoku.text_detector import TextDetector; "
            "from yomitoku.text_recognizer import TextRecognizer; "
            f"assert m.version('yomitoku') == {YOMITOKU_VERSION!r}"
        ),
        packages=[YOMITOKU_PACKAGE],
        verbose=verbose,
        min_minor=10,
        max_minor=13,
    )


def _worker_env() -> dict[str, str]:
    env = os.environ.copy()
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    hf_home = MODEL_CACHE / "huggingface"
    torch_home = MODEL_CACHE / "torch"
    hf_home.mkdir(parents=True, exist_ok=True)
    torch_home.mkdir(parents=True, exist_ok=True)
    env.setdefault("HF_HOME", str(hf_home))
    env.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
    env.setdefault("TRANSFORMERS_CACHE", str(hf_home / "hub"))
    env.setdefault("TORCH_HOME", str(torch_home))
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    env.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    env.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _normalise_mode(value: str) -> str:
    value = str(value or "fast").strip().lower()
    return value if value in {"fast", "accurate"} else "fast"


class YomiTokuSession:
    """Keep DBNet and PARSeq models alive for the complete OCR run."""

    def __init__(
        self,
        *,
        mode: str = "fast",
        device: str = "auto",
        detector_onnx: bool = True,
        large_review: bool = True,
        review_threshold: float = 0.82,
        cancel_check=None,
        verbose: bool = True,
    ):
        self.mode = _normalise_mode(mode)
        self.device_request = str(device or "auto").strip().lower()
        self.detector_onnx = bool(detector_onnx)
        self.large_review = bool(large_review) and self.mode == "fast"
        self.review_threshold = max(0.0, min(1.0, float(review_threshold)))
        self.cancel_check = cancel_check
        self.verbose = verbose
        self.proc: subprocess.Popen | None = None
        self._stderr_file = None
        self._stdout_pump = None
        self._request_id = 0
        self._lock = threading.Lock()
        self.device = ""
        self.detector_backend = ""
        self.recognizer_model = ""
        self.package_version = ""
        self.fallback = ""

    def _stderr_tail(self, limit: int = 4000) -> str:
        if self._stderr_file is None:
            return ""
        try:
            self._stderr_file.flush()
            self._stderr_file.seek(0)
            return self._stderr_file.read()[-limit:]
        except Exception:
            return ""

    def _read_json(self, timeout: float | None = None) -> dict:
        if self.proc is None or self.proc.stdout is None:
            raise RuntimeError("YomiToku worker 尚未启动")
        from adapters.subprocess_watchdog import LinePump, env_seconds
        if self._stdout_pump is None:
            self._stdout_pump = LinePump(self.proc.stdout, name="yomitoku-stdout")
        wait_seconds = (
            float(timeout) if timeout is not None
            else env_seconds("NOVEL_FORMATTER_OCR_REQUEST_TIMEOUT", 300.0, minimum=30.0)
        )
        while True:
            line = self._stdout_pump.readline(
                proc=self.proc,
                timeout=wait_seconds,
                cancel_check=self.cancel_check,
                label="YomiToku",
            )
            if line is None:
                tail = self._stderr_tail()
                code = self.proc.poll()
                raise RuntimeError(
                    "YomiToku worker 提前退出"
                    + (f"（退出码 {code}）" if code is not None else "")
                    + (f"：{tail}" if tail else "")
                )
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    def __enter__(self):
        python = setup_venv(verbose=self.verbose)
        self._stderr_file = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
        command = [
            str(python), str(WORKER_SCRIPT), "--server",
            "--mode", self.mode,
            "--device", self.device_request,
            "--review-threshold", str(self.review_threshold),
        ]
        if self.detector_onnx:
            command.append("--detector-onnx")
        if self.large_review:
            command.append("--large-review")
        from adapters.subprocess_watchdog import LinePump, env_seconds, isolated_process_kwargs
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file,
            text=True,
            bufsize=1,
            env=_worker_env(),
            **isolated_process_kwargs(),
        )
        self._stdout_pump = LinePump(self.proc.stdout, name="yomitoku-stdout")
        ready = self._read_json(
            timeout=env_seconds("NOVEL_FORMATTER_OCR_STARTUP_TIMEOUT", 900.0, minimum=60.0)
        )
        if not ready.get("ready"):
            error = str(ready.get("error") or "YomiToku worker 未就绪")
            self.close(force=True)
            raise RuntimeError(error)
        self.device = str(ready.get("device") or "")
        self.detector_backend = str(ready.get("detector_backend") or "")
        self.recognizer_model = str(ready.get("recognizer_model") or "")
        self.package_version = str(ready.get("version") or "")
        self.fallback = str(ready.get("fallback") or "")
        return self

    def iter_recognize(
        self, image_paths: list[str], *, progress_callback=None
    ) -> Iterator[tuple[str, list[dict] | None, str | None]]:
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("YomiToku worker 尚未启动")
        total = max(1, len(image_paths))
        completed = 0
        # Small requests keep cancellation responsive and bound source-image RAM.
        for offset in range(0, len(image_paths), 24):
            if self.cancel_check is not None and self.cancel_check():
                break
            chunk = [str(Path(path)) for path in image_paths[offset:offset + 24]]
            with self._lock:
                self._request_id += 1
                request_id = self._request_id
                payload = {
                    "request_id": request_id,
                    "images": chunk,
                    # Every path in this session is produced by Novel
                    # Formatter's authoritative physical-column layer.  DBNet
                    # must not re-segment a narrow printed column into radicals,
                    # punctuation fragments or Ruby-like side boxes.
                    "already_isolated": True,
                }
                self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
                self.proc.stdin.flush()
                seen: set[str] = set()
                batch_error = ""
                while True:
                    response = self._read_json()
                    if int(response.get("request_id", request_id) or request_id) != request_id:
                        continue
                    # Device/backend may change after an MPS or ONNX runtime
                    # fallback. Keep the session state and ready marker accurate.
                    self.device = str(response.get("device") or self.device)
                    self.detector_backend = str(
                        response.get("detector_backend") or self.detector_backend
                    )
                    self.fallback = str(response.get("fallback") or self.fallback)
                    if response.get("batch_done"):
                        if not response.get("ok", True):
                            batch_error = str(
                                response.get("error") or "YomiToku 批处理失败"
                            )
                        break
                    path = str(response.get("path") or "")
                    if not path:
                        continue
                    seen.add(path)
                    completed += 1
                    if response.get("ok"):
                        yield path, list(response.get("blocks") or []), None
                    else:
                        yield path, None, str(response.get("error") or "YomiToku 未知错误")
                    if callable(progress_callback):
                        progress_callback(completed, total, path)
                # A worker crash/bug must not silently lose physical columns.
                for path in chunk:
                    if path in seen:
                        continue
                    completed += 1
                    yield path, None, batch_error or "YomiToku worker 未返回该图片"
                    if callable(progress_callback):
                        progress_callback(completed, total, path)

    def recognize(
        self, image_paths: list[str], *, progress_callback=None
    ) -> dict[str, tuple[str, float, str | None]]:
        results: dict[str, tuple[str, float, str | None]] = {}
        for path, blocks, error in self.iter_recognize(
            image_paths, progress_callback=progress_callback
        ):
            if error:
                results[str(path)] = ("", 0.0, str(error))
                continue
            items = list(blocks or [])
            text = "".join(str(item.get("text") or "").strip() for item in items).strip()
            weights = [max(1, len(str(item.get("text") or ""))) for item in items]
            confidence = (
                sum(
                    float(item.get("confidence", 0.0) or 0.0) * weight
                    for item, weight in zip(items, weights)
                ) / max(1, sum(weights))
            )
            results[str(path)] = (
                text,
                confidence,
                None if text else "YomiToku 未检测到有效日文文字",
            )
        return results

    def close(self, *, force: bool = False) -> None:
        proc = self.proc
        self.proc = None
        if proc is not None:
            try:
                if not force and proc.poll() is None and proc.stdin is not None:
                    proc.stdin.write(json.dumps({"command": "close"}) + "\n")
                    proc.stdin.flush()
                    proc.wait(timeout=4)
                elif proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
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

    def __exit__(self, exc_type, exc, tb):
        try:
            self.close(force=exc is not None)
        finally:
            if exc is None:
                from adapters.ocr_runtime_catalog import mark_runtime_ready
                mark_runtime_ready(
                    "yomitoku",
                    version=self.package_version or "0.13.1",
                    detector="dbnetv2_1",
                    recognizer=self.recognizer_model,
                    device=self.device,
                    detector_backend=self.detector_backend,
                    fallback=self.fallback,
                )
        return False


def recognize_crops(
    crop_paths: list[str],
    manifest_path: str,
    *,
    cancel_check=None,
    verbose: bool = True,
    mode: str = "fast",
    device: str = "auto",
    detector_onnx: bool = True,
    large_review: bool = True,
    review_threshold: float = 0.82,
) -> Iterator[tuple[str, list[dict] | None, str | None]]:
    del manifest_path
    with YomiTokuSession(
        mode=mode,
        device=device,
        detector_onnx=detector_onnx,
        large_review=large_review,
        review_threshold=review_threshold,
        cancel_check=cancel_check,
        verbose=verbose,
    ) as session:
        yield from session.iter_recognize([str(path) for path in crop_paths])


def run(*, verbose: bool = True, **kwargs):
    """Run through Novel Formatter's authoritative physical-column pipeline."""
    from adapters.column_ocr_adapter import run as run_column_ocr

    recognition_engine = str(kwargs.pop("recognition_engine", "yomitoku") or "yomitoku")
    if recognition_engine != "yomitoku":
        raise ValueError("YomiToku 适配器只能使用 recognition_engine='yomitoku'")
    return run_column_ocr(recognition_engine="yomitoku", verbose=verbose, **kwargs)
