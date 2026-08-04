#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Manga Image Translator 48px autoregressive OCR adapter."""
from __future__ import annotations

import json
import os
import queue
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Iterator

from adapters.manga_ocr_adapter import (
    MangaOcrSession,
    _compact_text,
    _japanese_ratio,
    _looks_like_full_page,
    prepare_manga_ocr_segments,
)
from adapters.runtime_env import ensure_venv
from adapters.manga_48px_runtime import ensure_runtime_files

ROOT = Path(__file__).parent.parent
VENV_DIR = ROOT / ".venv-manga-48px"
WORKER_SCRIPT = Path(__file__).parent / "manga_48px_worker.py"
MODEL_CACHE = ROOT / ".model-cache" / "manga-48px-ar"


def setup_venv(verbose: bool = True) -> Path:
    return ensure_venv(
        VENV_DIR,
        label="Manga 48px AR OCR",
        marker_code="import torch, einops, numpy; from PIL import Image; assert torch.__version__",
        packages=[
            os.environ.get("NOVEL_FORMATTER_MANGA_48PX_TORCH", "torch>=2.3,<3"),
            "numpy>=1.26,<3",
            "Pillow>=10,<13",
            "einops>=0.8,<1",
        ],
        verbose=verbose,
    )


def _worker_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def validate_manga_48px_text(text: str, expected_chars: int, model_confidence: float) -> tuple[bool, float, str]:
    value = _compact_text(text)
    if not value:
        return False, 0.0, "48px AR OCR 返回空文本"
    if len(value) >= 250:
        return False, 0.0, "48px AR OCR 输出触及序列上限，已拒绝异常结果"
    if _japanese_ratio(value) < 0.48:
        return False, 0.0, "48px AR OCR 输出的日文字符比例异常"
    expected = max(1, int(expected_chars or 1))
    ratio = len(value) / expected
    if len(value) > max(48, expected * 2.45 + 10):
        return False, 0.0, f"48px AR OCR 输出字数异常（识别 {len(value)} / 字形估计 {expected}）"
    if expected >= 5 and ratio < 0.22:
        return False, 0.0, f"48px AR OCR 严重缺字（识别 {len(value)} / 字形估计 {expected}）"
    confidence = max(0.0, min(1.0, float(model_confidence or 0.0)))
    if confidence < 0.05:
        return False, confidence, f"48px AR OCR 模型置信度过低（{confidence:.3f}）"
    return True, confidence, ""


class Manga48pxSession(MangaOcrSession):
    """Persistent official 48px AR model shared across all column passes.

    The large model is prepared in the parent OCR thread before the worker is
    spawned.  This is intentional: downloading inside the worker left the GUI
    blocked on ``stdout.readline()`` with no progress or actionable error.
    """

    def __init__(self, *, cancel_check=None, verbose: bool = True, load_progress_callback=None):
        super().__init__(cancel_check=cancel_check, verbose=verbose)
        self.load_progress_callback = load_progress_callback

    def _emit_load(self, stage: str, current: int, total: int, detail: str) -> None:
        callback = self.load_progress_callback
        if callable(callback):
            callback(stage, current, total, detail)

    def _read_startup_response(self, timeout: float = 300.0) -> dict:
        """Wait for model initialization without allowing an infinite GUI hang."""
        responses: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

        def reader() -> None:
            try:
                responses.put((True, self._read_response()))
            except BaseException as exc:  # propagate exact worker diagnostics
                responses.put((False, exc))

        thread = threading.Thread(target=reader, daemon=True, name="manga-48px-ready")
        thread.start()
        deadline = time.monotonic() + max(30.0, float(timeout))
        while True:
            try:
                ok, value = responses.get(timeout=0.20)
                if ok:
                    return value  # type: ignore[return-value]
                raise value  # type: ignore[misc]
            except queue.Empty:
                if self.cancel_check is not None and self.cancel_check():
                    self.close(force=True)
                    raise RuntimeError("用户取消 48px AR OCR 模型加载")
                if time.monotonic() >= deadline:
                    tail = "\n".join(self._stderr_lines[-30:])
                    self.close(force=True)
                    raise RuntimeError(
                        "48px AR OCR 权重已下载，但模型初始化超过 5 分钟。"
                        "请查看下方诊断；程序已停止等待，不会永久卡住。\n" + tail
                    )

    def __enter__(self):
        self._emit_load("environment", 0, 1, "检查 48px AR 独立运行环境")
        python = setup_venv(verbose=self.verbose)
        self._emit_load("environment", 1, 1, "48px AR 运行环境已就绪 · 检查官方权重")
        MODEL_CACHE.mkdir(parents=True, exist_ok=True)
        ensure_runtime_files(
            MODEL_CACHE,
            progress_callback=self._emit_load,
            cancel_check=self.cancel_check,
        )
        self._emit_load("model", 0, 1, "权重下载与校验完成 · 正在创建识字模型")
        from adapters.subprocess_watchdog import isolated_process_kwargs
        self.proc = subprocess.Popen(
            [str(python), str(WORKER_SCRIPT), "--stream", "--cache-dir", str(MODEL_CACHE)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=_worker_env(),
            **isolated_process_kwargs(),
        )
        assert self.proc.stderr is not None
        stderr_pipe = self.proc.stderr
        self._stderr_stop.clear()
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(stderr_pipe,),
            daemon=True,
            name="manga-48px-stderr",
        )
        self._stderr_thread.start()
        ready = self._read_startup_response()
        if not ready.get("ready"):
            self.close(force=True)
            raise RuntimeError(ready.get("error", "48px AR OCR worker 未就绪"))
        self.device = str(ready.get("device", ""))
        fallback = str(ready.get("fallback", "") or "").strip()
        detail = f"48px AR 模型已加载 · device={self.device or 'unknown'}"
        if fallback:
            detail += f" · {fallback}"
        self._emit_load("model", 1, 1, detail)
        return self

    def recognize(self, crop_paths: list[str], *, progress_callback=None) -> dict[str, tuple[str, float, str | None]]:
        """Recognize physical columns in bounded multi-column batches.

        The old path sent one JSON request per column, so a 400-page book could
        perform several thousand pipe flushes while the worker's native 16-item
        tensor batch was almost always fed only one segment.  This windowed path
        preserves the exact segmenting/beam-search/validation rules but groups
        many columns into each request so the official model actually receives
        full batches.
        """
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("48px AR OCR worker 尚未启动")
        results: dict[str, tuple[str, float, str | None]] = {}
        ordered_paths = [str(path) for path in crop_paths]
        total = max(1, len(ordered_paths))
        try:
            source_window = int(os.environ.get("NOVEL_FORMATTER_MANGA_48PX_SOURCE_WINDOW", "32") or 32)
        except ValueError:
            source_window = 32
        source_window = max(4, min(64, source_window))

        with tempfile.TemporaryDirectory(prefix="novel_formatter_48px_chunks_") as temp_dir:
            chunk_root = Path(temp_dir)
            completed = 0
            for window_start in range(0, len(ordered_paths), source_window):
                if self.cancel_check is not None and self.cancel_check():
                    break
                window = ordered_paths[window_start:window_start + source_window]
                prepared: list[tuple[str, list, int, str]] = []
                flat_paths: list[str] = []

                for local_index, source_path in enumerate(window, start=1):
                    global_index = window_start + local_index
                    try:
                        segments, column_count = prepare_manga_ocr_segments(
                            source_path,
                            chunk_root / f"i{global_index:05d}",
                            max_aspect=15.0,
                            max_chars=30,
                            # source_path is already the common column layer's
                            # authoritative Ruby-free compact physical column.
                            # Never run page-level column detection on it again.
                            already_isolated=True,
                        )
                        error = "" if segments else "48px AR OCR 输入区域没有检测到印刷文字"
                    except Exception as exc:
                        segments, column_count = [], 0
                        error = f"48px AR OCR 输入分段失败: {exc}"
                    prepared.append((source_path, list(segments), int(column_count), error))
                    flat_paths.extend(segment.path for segment in segments)

                returned: dict[str, dict] = {}
                request_error = ""
                if flat_paths:
                    self._request_id += 1
                    request_id = self._request_id
                    request = {
                        "request_id": request_id,
                        "paths": flat_paths,
                        "beams_k": 5,
                        "max_seq_length": 255,
                    }
                    try:
                        self.proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
                        self.proc.stdin.flush()
                        data = self._read_response()
                        if int(data.get("request_id", -1) or -1) != request_id:
                            raise RuntimeError("48px AR OCR 请求响应串位")
                        if not data.get("ok"):
                            raise RuntimeError(str(data.get("error", "未知错误")))
                        returned = {
                            str(item.get("path", "")): item
                            for item in data.get("items", [])
                        }
                    except Exception as exc:
                        request_error = str(exc)

                for source_path, segments, column_count, prepare_error in prepared:
                    failure = prepare_error or request_error
                    column_texts: dict[int, list[str]] = {}
                    confidences: list[float] = []
                    if not failure:
                        for segment in segments:
                            item = returned.get(segment.path)
                            if item is None:
                                failure = f"48px AR OCR 未返回分段: {Path(segment.path).name}"
                                break
                            text = str(item.get("text", "") or "").strip()
                            valid, confidence, reason = validate_manga_48px_text(
                                text,
                                segment.expected_chars,
                                float(item.get("confidence", 0.0) or 0.0),
                            )
                            if not valid:
                                failure = reason
                                break
                            column_texts.setdefault(segment.column_index, []).append(
                                _compact_text(text)
                            )
                            confidences.append(confidence)

                    if failure:
                        results[source_path] = ("", 0.0, failure)
                    else:
                        columns = [
                            "".join(column_texts[index])
                            for index in sorted(column_texts)
                            if column_texts.get(index)
                        ]
                        text = ("\n" if column_count > 1 else "").join(columns).strip()
                        results[source_path] = (
                            text,
                            min(confidences) if confidences else 0.0,
                            None if text else "48px AR OCR 未返回有效文字",
                        )
                    completed += 1
                    if callable(progress_callback):
                        progress_callback(completed, total, source_path)
        return results

    def __exit__(self, exc_type, exc, tb):
        try:
            self.close(force=exc is not None)
        except Exception:
            if exc is None:
                raise
        if exc is None:
            from adapters.ocr_runtime_catalog import mark_runtime_ready
            mark_runtime_ready("manga_48px", model="ocr_ar_48px.ckpt", device=self.device)
        return False


def recognize_crops(
    crop_paths: list[str], manifest_path: str, *, cancel_check=None, verbose: bool = True
) -> Iterator[tuple[str, list[dict] | None, str | None]]:
    del manifest_path
    ordered = [str(path) for path in crop_paths]
    page_like = {path for path in ordered if _looks_like_full_page(path)}
    safe_paths = [path for path in ordered if path not in page_like]
    results: dict[str, tuple[str, float, str | None]] = {}
    if safe_paths:
        with Manga48pxSession(cancel_check=cancel_check, verbose=verbose) as session:
            results = session.recognize(safe_paths)
    for path in ordered:
        if path in page_like:
            yield path, None, "48px AR OCR 收到疑似整页图片；请先启用物理分列后逐列识别。"
            continue
        text, confidence, error = results.get(path, ("", 0.0, "识字进程未返回该区域"))
        if error:
            yield path, None, error
        else:
            yield path, ([{"text": text, "confidence": confidence, "box": None}] if text else []), None


def run(*, verbose: bool = True, **kwargs):
    from adapters.column_ocr_adapter import run as run_column_ocr
    recognition_engine = str(kwargs.pop("recognition_engine", "manga_48px") or "manga_48px")
    if recognition_engine != "manga_48px":
        raise ValueError("48px AR OCR 适配器只能使用 recognition_engine='manga_48px'")
    return run_column_ocr(recognition_engine="manga_48px", verbose=verbose, **kwargs)
