#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Manga Image Translator 48px autoregressive OCR adapter."""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Iterator

from adapters.json_worker_session import JsonWorkerSessionBase
from adapters.vertical_crop_segments import (
    OcrCropSegment,
    compact_text,
    japanese_ratio,
    looks_like_full_page,
    prepare_vertical_ocr_segments,
)
from adapters.runtime_env import ensure_venv
from adapters.ocr_prefetch import accelerator_prefetch_enabled, iter_prefetched_windows, windowed
from adapters.manga_48px_runtime import ensure_runtime_files
from utils.apple_silicon_runtime import is_m6, recommended_cpu_threads, torch_worker_env

ROOT = Path(__file__).parent.parent
VENV_DIR = ROOT / ".venv-manga-48px"
WORKER_SCRIPT = Path(__file__).parent / "manga_48px_worker.py"
MODEL_CACHE = Path(os.environ.get("NOVEL_FORMATTER_CLOUD_48PX_CACHE", "").strip() or (ROOT / ".model-cache" / "manga-48px-ar")).expanduser()
MANGA_48PX_TORCH_PACKAGE = os.environ.get(
    "NOVEL_FORMATTER_MANGA_48PX_TORCH",
    "torch>=2.14,<2.15" if is_m6() else "torch>=2.3,<3",
)

# Real-book ablation: horizontal Sentence-Reflow remains strong through roughly
# 60 glyphs, then AR decoding quality collapses sharply.  Fail closed instead
# of injecting a truncated third-model candidate into adjudication.
MANGA_48PX_MAX_REFLOW_CHARS = max(16, min(96, int(
    os.environ.get("NOVEL_FORMATTER_MANGA_48PX_MAX_REFLOW_CHARS", "60") or 60
)))

# AR beam decoding can enter a pathological long-running branch on an otherwise
# ordinary short column.  Keep CPU batches deliberately small and bound every
# request by a per-segment watchdog budget; a timed-out worker is killed and
# restarted so one bad crop cannot stall a full book.
MANGA_48PX_CPU_REQUEST_BATCH = 2
MANGA_48PX_ACCEL_REQUEST_BATCH = 16
MANGA_48PX_ITEM_WATCHDOG_SECONDS = 6.0


def _decode_cap_for_expected(expected_chars: int) -> int:
    need = max(16, int(expected_chars or 1) + 12)
    for cap in (24, 32, 48, 64, 80, 96, 128, 160, 192, 224, 255):
        if need <= cap:
            return cap
    return 255


def setup_venv(verbose: bool = True) -> Path:
    # Advanced/offline deployments may point at an already-provisioned Python.
    # This prevents a valid cloud/local runtime from touching pip or the network
    # merely because the project-private venv was created with another interpreter.
    explicit = os.environ.get("NOVEL_FORMATTER_MANGA_48PX_PYTHON", "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            return path
    marker = "import torch, einops, numpy; from PIL import Image; assert torch.__version__"
    if is_m6():
        marker += (
            "; v=torch.__version__.split('+',1)[0].split('.'); "
            "n=tuple(int(''.join(c for c in p if c.isdigit()) or 0) for p in v[:2]); "
            "assert n >= (2,14), 'M6 48px requires PyTorch 2.14+'"
        )
    # Prefer an already-provisioned runtime.  The recognizer must never stall an
    # OCR run by silently trying to upgrade pip/setuptools or download PyTorch.
    # This is especially important for offline production Macs and cloud QA.
    def _runtime_ready(python: Path) -> bool:
        if not python.is_file():
            return False
        try:
            proc = subprocess.run(
                [str(python), "-c", marker],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
            )
            return proc.returncode == 0
        except Exception:
            return False

    private_python = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if _runtime_ready(private_python):
        return private_python
    current_python = Path(sys.executable).resolve()
    if _runtime_ready(current_python):
        return current_python

    auto_install = str(os.environ.get("NOVEL_FORMATTER_MANGA_48PX_AUTO_INSTALL", "0") or "0").strip().lower()
    if auto_install not in {"1", "true", "yes", "on"}:
        raise RuntimeError(
            "48px AR OCR 运行依赖尚未准备。为避免 OCR 启动时隐式联网，程序不会自动升级 pip/下载 PyTorch。"
            "请先在设置中的 OCR 运行环境安装/修复依赖，或设置 NOVEL_FORMATTER_MANGA_48PX_PYTHON 指向已安装 torch/einops/Pillow 的 Python。"
        )

    torch_package = os.environ.get(
        "NOVEL_FORMATTER_MANGA_48PX_TORCH",
        "torch>=2.14,<2.15" if is_m6() else "torch>=2.3,<3",
    )
    return ensure_venv(
        VENV_DIR,
        label="Manga 48px AR OCR",
        marker_code=marker,
        packages=[
            torch_package,
            "numpy>=1.26,<3",
            "Pillow>=10,<13",
            "einops>=0.8,<1",
        ],
        verbose=verbose,
    )


def _worker_env() -> dict[str, str]:
    env = torch_worker_env(os.environ.copy())
    env.setdefault("NOVEL_FORMATTER_MANGA_48PX_THREADS", str(recommended_cpu_threads()))
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def validate_manga_48px_text(text: str, expected_chars: int, model_confidence: float) -> tuple[bool, float, str]:
    value = compact_text(text)
    if not value:
        return False, 0.0, "48px AR OCR 返回空文本"
    if len(value) >= 250:
        return False, 0.0, "48px AR OCR 输出触及序列上限，已拒绝异常结果"
    if japanese_ratio(value) < 0.48:
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


class Manga48pxSession(JsonWorkerSessionBase):
    """Persistent official 48px AR model shared across all column passes.

    The large model is prepared in the parent OCR thread before the worker is
    spawned.  This is intentional: downloading inside the worker left the GUI
    blocked on ``stdout.readline()`` with no progress or actionable error.
    """

    def __init__(self, *, cancel_check=None, verbose: bool = True, load_progress_callback=None, engine_options=None):
        super().__init__(cancel_check=cancel_check, verbose=verbose, worker_label="48px AR OCR")
        self.load_progress_callback = load_progress_callback
        self.options = dict(engine_options or {})

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

    def _start_worker(self, *, announce_environment: bool = False) -> None:
        if announce_environment:
            self._emit_load("environment", 0, 1, "检查 48px AR 独立运行环境")
        python = setup_venv(verbose=self.verbose)
        if announce_environment:
            self._emit_load("environment", 1, 1, "48px AR 运行环境已就绪 · 检查官方权重")
        # Strictly offline runs must never fall through to the model/source
        # downloader if an externally provisioned cache is incomplete.
        if any(os.environ.get(flag, "").strip().lower() in {"1", "true", "yes"}
               for flag in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "NOVEL_FORMATTER_OCR_OFFLINE")):
            required = (
                MODEL_CACHE / "ocr_ar_48px.ckpt",
                MODEL_CACHE / "alphabet-all-v7.txt",
                MODEL_CACHE / "upstream-source" / "model_48px.py",
                MODEL_CACHE / "upstream-source" / "xpos_relative_position.py",
            )
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "48px AR 离线资源不完整，已禁止联网下载：" + ", ".join(missing)
                )
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

    def _restart_worker_after_timeout(self) -> None:
        self.close(force=True)
        if self.cancel_check is not None and self.cancel_check():
            raise RuntimeError("用户取消 48px AR OCR")
        self._start_worker(announce_environment=False)

    def __enter__(self):
        self._start_worker(announce_environment=True)
        return self

    def recognize(
        self, crop_paths: list[str], *, progress_callback=None,
        input_metadata: dict[str, dict] | None = None,
    ) -> dict[str, tuple[str, float, str | None]]:
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
        metadata = {str(k): dict(v or {}) for k, v in (input_metadata or {}).items()}
        total = max(1, len(ordered_paths))
        try:
            source_window = int(os.environ.get("NOVEL_FORMATTER_MANGA_48PX_SOURCE_WINDOW", "32") or 32)
        except ValueError:
            source_window = 32
        source_window = max(4, min(64, source_window))

        with tempfile.TemporaryDirectory(prefix="novel_formatter_48px_chunks_") as temp_dir:
            chunk_root = Path(temp_dir)
            completed = 0

            def prepare_window(window_start: int, window):
                prepared: list[tuple[str, list, int, str]] = []
                flat_paths: list[str] = []
                for local_index, source_path in enumerate(window, start=1):
                    global_index = window_start + local_index
                    try:
                        item_meta = metadata.get(str(source_path), {})
                        if str(item_meta.get("layout") or "") == "horizontal_reflow":
                            # The native 48px worker already consumes horizontal
                            # strips and only auto-rotates tall vertical crops.
                            # Real-book ablation shows a hard quality cliff above
                            # ~60 glyphs, so overlong sentence evidence fails closed
                            # and the comparison keeps the independent column/page
                            # models instead of accepting a truncated AR result.
                            expected_chars = max(1, int(item_meta.get("expected_chars") or 1))
                            if expected_chars > MANGA_48PX_MAX_REFLOW_CHARS:
                                segments = []
                                column_count = 1
                                error = (
                                    "48px AR OCR 整句超过安全长度"
                                    f"（{expected_chars} > {MANGA_48PX_MAX_REFLOW_CHARS}），已跳过该证据"
                                )
                                prepared.append((source_path, list(segments), int(column_count), error))
                                continue
                            segments = [OcrCropSegment(str(source_path), expected_chars, 0, 0)]
                            column_count = 1
                        else:
                            segments, column_count = prepare_vertical_ocr_segments(
                                source_path,
                                chunk_root / f"i{global_index:05d}",
                                max_aspect=15.0,
                                max_chars=30,
                                already_isolated=True,
                                physical_column_boxes=item_meta.get("physical_column_boxes") or (),
                            )
                        error = "" if segments else "48px AR OCR 输入区域没有检测到印刷文字"
                    except Exception as exc:
                        segments, column_count = [], 0
                        error = f"48px AR OCR 输入分段失败: {exc}"
                    prepared.append((source_path, list(segments), int(column_count), error))
                    flat_paths.extend(segment.path for segment in segments)
                return prepared, flat_paths

            # If any multi-item AR batch times out, keep the worker alive but
            # degrade the remainder of this recognition run to batch=1.  This
            # avoids repeatedly pairing a pathological segment with healthy peers
            # across later pages while preserving fast batching on runs that never
            # exhibit the long-tail branch.
            adaptive_batch_cap: int | None = None

            windows = windowed(ordered_paths, source_window)
            prefetch = accelerator_prefetch_enabled(
                self.device, os.environ.get("NOVEL_FORMATTER_OCR_DOUBLE_BUFFER", "auto")
            )
            for window_start, window, prepared_payload in iter_prefetched_windows(
                windows, prepare_window, enabled=prefetch
            ):
                if self.cancel_check is not None and self.cancel_check():
                    break
                prepared, flat_paths = prepared_payload

                returned: dict[str, dict] = {}
                segment_errors: dict[str, str] = {}
                if flat_paths:
                    segment_lookup = {
                        segment.path: segment
                        for _source, segments, _count, _error in prepared
                        for segment in segments
                    }
                    cap_buckets: dict[int, list[str]] = {}
                    for segment_path in flat_paths:
                        segment = segment_lookup[segment_path]
                        cap = _decode_cap_for_expected(segment.expected_chars)
                        cap_buckets.setdefault(cap, []).append(segment_path)

                    try:
                        configured_batch = int(os.environ.get(
                            "NOVEL_FORMATTER_MANGA_48PX_REQUEST_BATCH", "0"
                        ) or 0)
                    except ValueError:
                        configured_batch = 0
                    default_batch = (
                        MANGA_48PX_CPU_REQUEST_BATCH
                        if str(self.device or "").lower().startswith("cpu")
                        else MANGA_48PX_ACCEL_REQUEST_BATCH
                    )
                    request_batch = max(1, min(32, configured_batch or default_batch))
                    # Sparse conflict review values failure isolation over raw
                    # throughput. One pathological AR beam must not force healthy
                    # neighbour columns through recursive retries and model reloads.
                    selective_review = bool(
                        self.options.get("review_only")
                        or self.options.get("disagreement_review")
                        or any(bool(metadata.get(path, {}).get("disagreement_review")) for path in ordered_paths)
                    )
                    if selective_review and "NOVEL_FORMATTER_MANGA_48PX_REQUEST_BATCH" not in os.environ:
                        # The former review path forced batch=1. On an M-series GPU this
                        # turned ~2k disagreement columns into thousands of tiny beam
                        # requests and was slower than the two full main models combined.
                        # Keep small CPU batches, but let accelerators amortize the AR
                        # decoder. Pathological beams are isolated by recursive bisection.
                        request_batch = min(request_batch, 2 if str(self.device or "").lower().startswith("cpu") else 8)
                    try:
                        per_item_timeout = float(os.environ.get(
                            "NOVEL_FORMATTER_MANGA_48PX_ITEM_TIMEOUT",
                            str(MANGA_48PX_ITEM_WATCHDOG_SECONDS),
                        ) or MANGA_48PX_ITEM_WATCHDOG_SECONDS)
                    except ValueError:
                        per_item_timeout = MANGA_48PX_ITEM_WATCHDOG_SECONDS
                    per_item_timeout = max(5.0, min(120.0, per_item_timeout))

                    def run_bounded_batch(batch_paths: list[str], cap: int) -> None:
                        """Run one 48px request and isolate pathological crops on failure.

                        A timeout in a multi-item request used to discard the whole batch.
                        Instead, kill/restart the worker and bisect the batch recursively.
                        Normal siblings are therefore recovered while only the crop that
                        repeatedly times out is failed closed.
                        """
                        nonlocal adaptive_batch_cap
                        if not batch_paths:
                            return
                        if self.cancel_check is not None and self.cancel_check():
                            for path in batch_paths:
                                segment_errors[path] = "用户取消"
                            return
                        if self.proc is None or self.proc.stdin is None:
                            try:
                                self._start_worker(announce_environment=False)
                            except Exception as exc:
                                for path in batch_paths:
                                    segment_errors[path] = f"48px AR OCR worker 重启失败: {exc}"
                                return
                        self._request_id += 1
                        request_id = self._request_id
                        request = {
                            "request_id": request_id,
                            "paths": batch_paths,
                            "beams_k": 5,
                            "max_seq_length": cap,
                        }
                        # The worker evaluates a tensor batch as one inference, not
                        # N serial inferences. Multiplying the watchdog by batch size
                        # lets one pathological beam hold a 4-item CPU batch for 48s
                        # before isolation even starts. Give the batch a bounded base
                        # budget plus small scheduling slack instead.
                        timeout = per_item_timeout + 2.0 * max(0, len(batch_paths) - 1)
                        try:
                            assert self.proc is not None and self.proc.stdin is not None
                            self.proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
                            self.proc.stdin.flush()
                            try:
                                data = self._read_response(timeout=timeout)
                            except TypeError as exc:
                                # Backward-compatible with tiny test/fake sessions that
                                # override _read_response() without the optional timeout.
                                # Production JsonWorkerSessionBase always accepts timeout.
                                if "timeout" not in str(exc):
                                    raise
                                data = self._read_response()
                            if int(data.get("request_id", -1) or -1) != request_id:
                                raise RuntimeError("48px AR OCR 请求响应串位")
                            if not data.get("ok"):
                                raise RuntimeError(str(data.get("error", "未知错误")))
                            for item in data.get("items", []):
                                returned[str(item.get("path", ""))] = item
                            missing = [path for path in batch_paths if path not in returned]
                            if missing:
                                raise RuntimeError(
                                    "48px AR OCR 批请求缺少返回项: "
                                    + ", ".join(Path(path).name for path in missing[:4])
                                )
                        except Exception as exc:
                            message = (
                                f"48px AR OCR 单条 watchdog 触发"
                                f"（batch={len(batch_paths)}, cap={cap}, timeout={timeout:.0f}s）: {exc}"
                            )
                            try:
                                self._restart_worker_after_timeout()
                            except Exception as restart_exc:
                                self.close(force=True)
                                message += f"; worker 重启失败: {restart_exc}"
                                for path in batch_paths:
                                    segment_errors[path] = message
                                return
                            if len(batch_paths) > 1:
                                # Isolate the current long-tail request, but do not
                                # collapse the remainder of a 2k-column review to batch=1.
                                # Learn a smaller cap and retain useful accelerator batching.
                                adaptive_batch_cap = max(2, min(
                                    int(adaptive_batch_cap or request_batch),
                                    max(2, len(batch_paths) // 2),
                                ))
                                mid = max(1, len(batch_paths) // 2)
                                run_bounded_batch(batch_paths[:mid], cap)
                                run_bounded_batch(batch_paths[mid:], cap)
                            else:
                                segment_errors[batch_paths[0]] = message

                    for cap in sorted(cap_buckets):
                        bucket = cap_buckets[cap]
                        offset = 0
                        while offset < len(bucket):
                            batch_size = max(1, min(request_batch, int(adaptive_batch_cap or request_batch)))
                            run_bounded_batch(bucket[offset:offset + batch_size], cap)
                            offset += batch_size

                for source_path, segments, column_count, prepare_error in prepared:
                    failure = prepare_error
                    column_texts: dict[int, list[str]] = {}
                    confidences: list[float] = []
                    if not failure:
                        for segment in segments:
                            if segment.path in segment_errors:
                                failure = segment_errors[segment.path]
                                break
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
                                compact_text(text)
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
    crop_paths: list[str], manifest_path: str, *, cancel_check=None, verbose: bool = True,
    input_metadata: dict[str, dict] | None = None,
) -> Iterator[tuple[str, list[dict] | None, str | None]]:
    del manifest_path
    ordered = [str(path) for path in crop_paths]
    metadata = {str(k): dict(v or {}) for k, v in (input_metadata or {}).items()}
    page_like = {
        path for path in ordered
        if str(metadata.get(path, {}).get("layout") or "") != "horizontal_reflow"
        and looks_like_full_page(path)
    }
    safe_paths = [path for path in ordered if path not in page_like]
    results: dict[str, tuple[str, float, str | None]] = {}
    if safe_paths:
        with Manga48pxSession(cancel_check=cancel_check, verbose=verbose) as session:
            results = session.recognize(safe_paths, input_metadata=metadata)
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
