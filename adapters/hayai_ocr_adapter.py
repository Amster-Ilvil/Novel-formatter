#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hayai OCR v2.1 adapter using the existing physical-column pipeline.

Hayai is deliberately treated as a crop recognizer, never as a page-layout
engine.  Novel Formatter owns page geometry, reading order, Ruby cleanup,
segment validation, multi-model fusion and AI adjudication.
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
from adapters.manga_ocr_adapter import (
    _compact_text,
    _japanese_ratio,
    _looks_like_full_page,
    prepare_manga_ocr_segments,
)

ROOT = Path(__file__).parent.parent
VENV_DIR = ROOT / ".venv-hayai-ocr"
WORKER_SCRIPT = Path(__file__).parent / "hayai_ocr_worker.py"
MODEL_CACHE = ROOT / ".model-cache" / "hayai-ocr"
HAYAI_OCR_VERSION = "2.1.0"
HAYAI_OCR_PACKAGE = os.environ.get(
    "NOVEL_FORMATTER_HAYAI_OCR_PACKAGE", f"hayai-ocr=={HAYAI_OCR_VERSION}"
)
# Hayai v2 uses SigLIP2/Transformers 4 APIs. Keep the independent runtime out
# of a future Transformers major-version migration until this adapter is
# explicitly revalidated; this cannot affect any other OCR venv.
HAYAI_TRANSFORMERS_PACKAGE = os.environ.get(
    "NOVEL_FORMATTER_HAYAI_TRANSFORMERS_PACKAGE", "transformers>=4.49,<5"
)


def _normalise_backend(value: str | None) -> str:
    return "litert" if str(value or "").strip().lower() in {"litert", "tflite", "lite_rt"} else "torch"


def _safe_int(value, default: int, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(minimum, min(maximum, parsed))


def _safe_float(value, default: float, *, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    return max(minimum, min(maximum, parsed))


def _normalise_litert_quant(value: str | None) -> str:
    raw = str(value or "wi4").strip().lower().replace("-", "_")
    aliases = {
        "int4": "wi4",
        "int8": "wi8_afp32",
        "wi8": "wi8_afp32",
        "float": "none",
        "fp32": "none",
        "dynamic_int4": "dynamic_wi4",
        "dynamic_int8": "dynamic_wi8",
    }
    raw = aliases.get(raw, raw)
    return raw if raw in {"none", "wi4", "wi8_afp32", "dynamic_wi4", "dynamic_wi8"} else "wi4"


def _resolved_model_cache() -> Path:
    override = os.environ.get("NOVEL_FORMATTER_HAYAI_OCR_CACHE_DIR", "").strip()
    return Path(override).expanduser() if override else MODEL_CACHE


def setup_venv(*, verbose: bool = True, backend: str = "torch") -> Path:
    backend = _normalise_backend(backend)
    package = HAYAI_OCR_PACKAGE
    if backend == "litert" and "[litert]" not in package:
        if "==" in package:
            name, version = package.split("==", 1)
            package = f"{name}[litert]=={version}"
        else:
            package = "hayai-ocr[litert]"
    marker = (
        "from hayai_ocr import HayaiOcr; "
        "from importlib.metadata import version; "
        f"assert version('hayai-ocr') == {HAYAI_OCR_VERSION!r}, version('hayai-ocr')"
    )
    if backend == "litert":
        marker += "; import ai_edge_litert"
    return ensure_venv(
        VENV_DIR,
        label="Hayai OCR v2.1",
        marker_code=marker,
        packages=[package, HAYAI_TRANSFORMERS_PACKAGE],
        verbose=verbose,
        min_minor=10,
        max_minor=13,
    )


def _offline_cache_ready(backend: str) -> bool:
    component_id = "hayai_ocr_litert" if _normalise_backend(backend) == "litert" else "hayai_ocr"
    try:
        from adapters.ocr_runtime_catalog import _state_marker_ready
        return bool(_state_marker_ready(component_id))
    except Exception:
        return False


def _worker_env(backend: str = "torch") -> dict[str, str]:
    env = os.environ.copy()
    cache = _resolved_model_cache()
    cache.mkdir(parents=True, exist_ok=True)
    # Always isolate Hayai from the process-wide Hugging Face cache. This avoids
    # one OCR backend changing another backend's cache state and makes runtime
    # detection deterministic. A dedicated Hayai override remains available.
    env["HF_HOME"] = str(cache)
    env["HUGGINGFACE_HUB_CACHE"] = str(cache / "hub")
    env["TRANSFORMERS_CACHE"] = str(cache / "transformers")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    # After one successful inference, keep the verified cached snapshot immutable
    # during ordinary OCR starts.  This enforces the project's "no silent model
    # update on startup" contract; explicit cache removal/manual maintenance is
    # required before a new upstream snapshot can be fetched.
    if _offline_cache_ready(backend):
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    return env


def validate_hayai_ocr_text(text: str, expected_chars: int) -> tuple[bool, float, str]:
    """Reject obvious generative hallucination while preserving mixed Japanese text."""
    value = _compact_text(text)
    if not value:
        return False, 0.0, "Hayai OCR 返回空文本"
    expected = max(1, int(expected_chars or 1))
    ratio = len(value) / expected
    if len(value) > max(56, int(expected * 2.35 + 10)):
        return False, 0.0, f"Hayai OCR 输出字数异常（识别 {len(value)} / 字形估计 {expected}）"
    if ratio < 0.26:
        return False, 0.0, f"Hayai OCR 严重缺字（识别 {len(value)} / 字形估计 {expected}）"
    if _japanese_ratio(value) < 0.42:
        return False, 0.0, "Hayai OCR 输出的日文/CJK 字符比例异常"
    # Heuristic confidence only.  It is intentionally not presented as a model probability.
    confidence = 0.80 if 0.56 <= ratio <= 1.65 else 0.60
    return True, confidence, ""


class HayaiOcrSession:
    """One persistent Hayai model shared by primary, rescue and sentence crops."""

    def __init__(self, *, engine_options: dict | None = None, cancel_check=None, verbose: bool = True):
        self.options = dict(engine_options or {})
        self.cancel_check = cancel_check
        self.verbose = verbose
        self.proc: subprocess.Popen[str] | None = None
        self._stderr_lines: list[str] = []
        self._stderr_thread: threading.Thread | None = None
        self._stderr_stop = threading.Event()
        self._stdout_pump = None
        self.device = ""
        self.backend = _normalise_backend(self.options.get("backend"))
        self.requested_quantize = "none"
        self.effective_quantize = "none"
        self.startup_warning = ""
        self._request_id = 0

    def __enter__(self):
        python = setup_venv(verbose=self.verbose, backend=self.backend)
        quantize = str(self.options.get("quantize") or "none").strip().lower()
        if quantize not in {"none", "int8", "int4"}:
            quantize = "none"
        self.requested_quantize = quantize
        device = str(self.options.get("device") or "auto").strip().lower()
        if device not in {"auto", "cpu", "cuda", "mps"}:
            device = "auto"
        litert_quant = _normalise_litert_quant(self.options.get("litert_quant"))
        litert_threads = _safe_int(self.options.get("litert_threads"), 0, minimum=0, maximum=256)
        max_token_cap = 64 if self.backend == "litert" else 192
        max_new_tokens = _safe_int(self.options.get("max_new_tokens"), 128, minimum=32, maximum=max_token_cap)
        cmd = [
            str(python), str(WORKER_SCRIPT), "--stream",
            "--backend", self.backend,
            "--device", device,
            "--quantize", quantize,
            "--litert-quant", litert_quant,
            "--litert-threads", str(litert_threads),
            "--max-new-tokens", str(max_new_tokens),
        ]
        from adapters.subprocess_watchdog import LinePump, isolated_process_kwargs
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=_worker_env(self.backend),
            **isolated_process_kwargs(),
        )
        self._stdout_pump = LinePump(self.proc.stdout, name="hayai-ocr-stdout")
        assert self.proc.stderr is not None
        self._stderr_stop.clear()
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(self.proc.stderr,),
            daemon=True,
            name="hayai-ocr-stderr",
        )
        self._stderr_thread.start()
        from adapters.subprocess_watchdog import env_seconds
        ready = self._read_response(
            timeout=env_seconds("NOVEL_FORMATTER_OCR_STARTUP_TIMEOUT", 900.0, minimum=60.0)
        )
        if not ready.get("ready"):
            self.close(force=True)
            raise RuntimeError(str(ready.get("error") or "Hayai OCR worker 未就绪"))
        self.device = str(ready.get("device") or "")
        self.backend = _normalise_backend(ready.get("backend") or self.backend)
        self.effective_quantize = str(ready.get("effective_quantize") or ready.get("quantize") or "none")
        self.startup_warning = str(ready.get("warning") or "").strip()
        if self.startup_warning and self.verbose:
            print(f"[Hayai OCR] {self.startup_warning}")
        return self

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

    def _read_response(self, timeout: float | None = None) -> dict:
        if self.proc is None or self.proc.stdout is None:
            raise RuntimeError("Hayai OCR worker 尚未启动")
        from adapters.subprocess_watchdog import LinePump, env_seconds
        if self._stdout_pump is None:
            self._stdout_pump = LinePump(self.proc.stdout, name="hayai-ocr-stdout")
        wait_seconds = float(timeout) if timeout is not None else env_seconds(
            "NOVEL_FORMATTER_OCR_REQUEST_TIMEOUT", 300.0, minimum=30.0
        )
        line = self._stdout_pump.readline(
            proc=self.proc,
            timeout=wait_seconds,
            cancel_check=self.cancel_check,
            label="Hayai OCR",
        )
        if line is None:
            tail = "\n".join(self._stderr_lines[-30:])
            raise RuntimeError(f"Hayai OCR worker 提前退出 (code={self.proc.poll()})\n{tail}")
        try:
            return json.loads(line)
        except Exception as exc:
            raise RuntimeError(f"Hayai OCR worker 返回无效 JSON: {line[:300]} ({exc})") from exc

    def recognize(self, crop_paths: list[str], *, progress_callback=None) -> dict[str, tuple[str, float, str | None]]:
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("Hayai OCR worker 尚未启动")
        ordered_paths = [str(path) for path in crop_paths]
        results: dict[str, tuple[str, float, str | None]] = {}
        total = max(1, len(ordered_paths))
        try:
            source_window = int(os.environ.get("NOVEL_FORMATTER_HAYAI_OCR_SOURCE_WINDOW", "24") or 24)
        except ValueError:
            source_window = 24
        source_window = max(2, min(48, source_window))
        default_batch = 8 if "mps" in self.device.lower() else 16 if "cuda" in self.device.lower() else 4
        if self.backend == "litert":
            default_batch = 4
        try:
            segment_batch = int(os.environ.get("NOVEL_FORMATTER_HAYAI_OCR_BATCH", str(default_batch)) or default_batch)
        except ValueError:
            segment_batch = default_batch
        segment_batch = max(1, min(32, segment_batch))
        max_chars = _safe_int(self.options.get("segment_max_chars"), 24, minimum=12, maximum=40)
        max_aspect = _safe_float(self.options.get("segment_max_aspect"), 16.0, minimum=6.0, maximum=18.0)

        with tempfile.TemporaryDirectory(prefix="novel_formatter_hayai_chunks_") as temp_dir:
            chunk_root = Path(temp_dir)
            completed = 0
            for window_start in range(0, len(ordered_paths), source_window):
                if self.cancel_check is not None and self.cancel_check():
                    break
                window = ordered_paths[window_start:window_start + source_window]
                prepared: list[tuple[str, list, int, str]] = []
                flat_segments = []
                for local_index, source_path in enumerate(window, start=1):
                    global_index = window_start + local_index
                    try:
                        segments, column_count = prepare_manga_ocr_segments(
                            source_path,
                            chunk_root / f"i{global_index:05d}",
                            already_isolated=True,
                            estimate_isolated_chars=True,
                            max_chars=max_chars,
                            max_aspect=max_aspect,
                        )
                        error = "" if segments else "Hayai OCR 输入区域没有检测到印刷文字"
                    except Exception as exc:
                        segments, column_count = [], 0
                        error = f"Hayai OCR 输入分段失败: {exc}"
                    prepared.append((source_path, list(segments), int(column_count), error))
                    flat_segments.extend(segments)

                returned: dict[str, dict] = {}
                transport_error = ""
                for offset in range(0, len(flat_segments), segment_batch):
                    if self.cancel_check is not None and self.cancel_check():
                        transport_error = "用户取消"
                        break
                    batch = flat_segments[offset:offset + segment_batch]
                    self._request_id += 1
                    request_id = self._request_id
                    try:
                        self.proc.stdin.write(json.dumps({
                            "request_id": request_id,
                            "paths": [segment.path for segment in batch],
                        }, ensure_ascii=False) + "\n")
                        self.proc.stdin.flush()
                        data = self._read_response()
                        if int(data.get("request_id", -1) or -1) != request_id:
                            raise RuntimeError("Hayai OCR 请求响应串位")
                        if not data.get("ok"):
                            raise RuntimeError(str(data.get("error") or "未知错误"))
                        items = data.get("items")
                        if not isinstance(items, list):
                            raise RuntimeError("Hayai OCR 批量响应缺少 items")
                        expected_paths = [str(segment.path) for segment in batch]
                        if len(items) != len(expected_paths):
                            raise RuntimeError(
                                f"Hayai OCR 批量响应数量异常：输入 {len(expected_paths)}，返回 {len(items)}"
                            )
                        batch_returned: set[str] = set()
                        for item in items:
                            if not isinstance(item, dict):
                                raise RuntimeError("Hayai OCR 批量响应包含非对象条目")
                            item_path = str(item.get("path") or "")
                            if item_path not in expected_paths or item_path in batch_returned:
                                raise RuntimeError("Hayai OCR 批量响应路径异常或重复")
                            batch_returned.add(item_path)
                            returned[item_path] = item
                    except Exception as exc:
                        transport_error = str(exc)
                        break

                for source_path, segments, column_count, prepare_error in prepared:
                    # A later batch transport failure must not discard source crops
                    # whose complete segment set was already returned successfully.
                    failure = prepare_error
                    column_texts: dict[int, list[str]] = {}
                    confidences: list[float] = []
                    if not failure:
                        for segment in segments:
                            data = returned.get(segment.path)
                            if data is None:
                                failure = transport_error or f"Hayai OCR 未返回分段: {Path(segment.path).name}"
                                break
                            if not data.get("ok"):
                                failure = str(data.get("error") or "未知错误")
                                break
                            blocks = data.get("blocks") or []
                            text = "".join(
                                str(item.get("text") or "").strip()
                                for item in blocks
                                if str(item.get("text") or "").strip()
                            ).strip()
                            valid, confidence, reason = validate_hayai_ocr_text(text, segment.expected_chars)
                            if not valid:
                                failure = reason
                                break
                            column_texts.setdefault(segment.column_index, []).append(_compact_text(text))
                            confidences.append(confidence)
                    if failure:
                        results[source_path] = ("", 0.0, failure)
                    else:
                        ordered_columns = [
                            "".join(column_texts[index])
                            for index in sorted(column_texts)
                            if column_texts.get(index)
                        ]
                        text = ("\n" if column_count > 1 else "").join(ordered_columns).strip()
                        results[source_path] = (
                            text,
                            min(confidences) if confidences else 0.0,
                            None if text else "Hayai OCR 未返回有效文字",
                        )
                    completed += 1
                    if callable(progress_callback):
                        progress_callback(completed, total, source_path)
        return results

    def close(self, *, force: bool = False) -> None:
        proc = self.proc
        self.proc = None
        if proc is None:
            return
        shutdown_requested = False
        termination_requested = False
        initial_ret = proc.poll()
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
                    proc.kill()
                    termination_requested = True
                    ret = proc.wait()
        except Exception:
            if proc.poll() is None:
                try:
                    proc.kill()
                    termination_requested = True
                except Exception:
                    pass
            ret = proc.wait()

        self._stderr_stop.set()
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1.5)
        try:
            if proc.stderr:
                proc.stderr.close()
        except Exception:
            pass
        self._stderr_thread = None
        if self._stdout_pump is not None:
            self._stdout_pump.close()
            self._stdout_pump = None
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass
        intentional = bool(force or shutdown_requested or termination_requested)
        if ret not in (0, -15) and not intentional:
            tail = "\n".join(self._stderr_lines[-30:])
            raise RuntimeError(f"Hayai OCR worker 异常退出 (code={ret}):\n{tail}")

    def __exit__(self, exc_type, exc, tb):
        try:
            self.close(force=exc is not None)
        except Exception:
            if exc is None:
                raise
        if exc is None:
            from adapters.ocr_runtime_catalog import mark_runtime_ready
            component_id = "hayai_ocr_litert" if self.backend == "litert" else "hayai_ocr"
            mark_runtime_ready(
                component_id,
                backend=self.backend,
                device=self.device,
                version=HAYAI_OCR_VERSION,
                requested_quantize=self.requested_quantize,
                effective_quantize=self.effective_quantize,
                warning=self.startup_warning,
            )
        return False


def recognize_crops(
    crop_paths: list[str],
    manifest_path: str,
    *,
    cancel_check=None,
    verbose: bool = True,
    engine_options: dict | None = None,
) -> Iterator[tuple[str, list[dict] | None, str | None]]:
    del manifest_path
    ordered = [str(path) for path in crop_paths]
    page_like = {path for path in ordered if _looks_like_full_page(path)}
    safe_paths = [path for path in ordered if path not in page_like]
    results: dict[str, tuple[str, float, str | None]] = {}
    if safe_paths:
        with HayaiOcrSession(
            engine_options=engine_options,
            cancel_check=cancel_check,
            verbose=verbose,
        ) as session:
            results = session.recognize(safe_paths)
    for path in ordered:
        if path in page_like:
            yield path, None, (
                "Hayai OCR 收到疑似整页图片，已拒绝直接识别。"
                "请使用 Hayai OCR 页面入口，由程序先做物理分列后再识别。"
            )
            continue
        text, confidence, error = results.get(path, ("", 0.0, "识字进程未返回该区域"))
        if error:
            yield path, None, error
        else:
            blocks = [{
                "text": text,
                "confidence": confidence,
                "confidence_kind": "heuristic",
                "box": None,
            }] if text else []
            yield path, blocks, None


def run(*, verbose: bool = True, **kwargs):
    from adapters.column_ocr_adapter import run as run_column_ocr

    recognition_engine = str(kwargs.pop("recognition_engine", "hayai_ocr") or "hayai_ocr")
    if recognition_engine != "hayai_ocr":
        raise ValueError("Hayai OCR 适配器只能使用 recognition_engine='hayai_ocr'")
    return run_column_ocr(
        recognition_engine="hayai_ocr",
        verbose=verbose,
        **kwargs,
    )
