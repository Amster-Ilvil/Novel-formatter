#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Review-only Manga OCR adapter for kha-white/manga-ocr-base.

Unlike the retired whole-column integration, this adapter deliberately splits a
conflicting physical column into short vertical blocks before the model's fixed
224x224 image processor.  It is intended for disagreement review only; it does
not download or update model weights.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Iterator

from adapters.json_worker_session import JsonWorkerSessionBase
from adapters.runtime_env import ensure_venv, persistent_runtime_root, persistent_venv_dir
from adapters.vertical_crop_segments import (
    OcrCropSegment,
    compact_text,
    japanese_ratio,
    looks_like_full_page,
    prepare_vertical_ocr_segments,
)
from utils.apple_silicon_runtime import is_m6, recommended_cpu_threads, torch_worker_env

ROOT = Path(__file__).parent.parent
WORKER_SCRIPT = Path(__file__).parent / "manga_ocr_worker.py"
METADATA_DIR = ROOT / "resources" / "manga_ocr_base"
MODEL_CACHE = ROOT / ".model-cache" / "manga-ocr-base"
VENV_DIR = persistent_venv_dir("manga-ocr-review")
MANGA_OCR_SHA256 = "c63e0bb5b3ff798c5991de18a8e0956c7ee6d1563aca6729029815eda6f5c2eb"
MANGA_OCR_CPU_BATCH = 4
MANGA_OCR_ACCEL_BATCH = 8
MANGA_OCR_ITEM_TIMEOUT_SECONDS = 10.0


def _metadata_ready(path: Path) -> bool:
    required = (
        "config.json", "preprocessor_config.json", "generation_config.json",
        "tokenizer_config.json", "special_tokens_map.json", "vocab.txt",
    )
    return all((path / name).is_file() for name in required)


def resolve_model_assets() -> tuple[Path, Path]:
    """Resolve local weight + metadata paths without network access."""
    candidates: list[Path] = []
    env = os.environ.get("NOVEL_FORMATTER_MANGA_OCR_MODEL", "").strip()
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend([
        MODEL_CACHE,
        persistent_runtime_root() / "manga-ocr-base" / "model",
    ])
    for candidate in candidates:
        weight = candidate / "pytorch_model.bin" if candidate.is_dir() else candidate
        if not weight.is_file():
            continue
        local_metadata = weight.parent
        metadata = local_metadata if _metadata_ready(local_metadata) else METADATA_DIR
        if not _metadata_ready(metadata):
            raise RuntimeError("Manga OCR 小型配置/词表文件不完整")
        return weight.resolve(), metadata.resolve()
    raise RuntimeError(
        "未找到 Manga OCR 本地权重。请设置 NOVEL_FORMATTER_MANGA_OCR_MODEL 指向"
        " kha-white/manga-ocr-base 的 pytorch_model.bin（或所在目录）；程序不会自动下载模型。"
    )


def setup_venv(*, verbose: bool = True) -> Path:
    explicit = os.environ.get("NOVEL_FORMATTER_MANGA_OCR_PYTHON", "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            return path
    marker = (
        "import torch; from PIL import Image; "
        "from transformers import VisionEncoderDecoderModel, ViTImageProcessor"
    )
    torch_pkg = "torch>=2.14,<2.15" if is_m6() else "torch>=2.3,<3"
    return ensure_venv(
        VENV_DIR,
        label="Manga OCR 分歧复核",
        marker_code=marker,
        packages=[torch_pkg, "transformers>=4.49,<5", "Pillow>=10,<13"],
        verbose=verbose,
        min_minor=10,
        max_minor=13,
    )


def _worker_env() -> dict[str, str]:
    env = torch_worker_env(os.environ.copy())
    env.setdefault("NOVEL_FORMATTER_MANGA_OCR_THREADS", str(recommended_cpu_threads()))
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def _decode_cap(expected_chars: int) -> int:
    need = max(16, int(expected_chars or 1) + 10)
    for cap in (24, 32, 48, 64, 96, 128):
        if need <= cap:
            return cap
    return 128


def validate_manga_ocr_text(text: str, expected_chars: int) -> tuple[bool, float, str]:
    value = compact_text(text)
    if not value:
        return False, 0.0, "Manga OCR 返回空文本"
    if japanese_ratio(value) < 0.42:
        return False, 0.0, "Manga OCR 输出的日文字符比例异常"
    expected = max(1, int(expected_chars or 1))
    if len(value) > max(32, int(expected * 2.6) + 8):
        return False, 0.0, f"Manga OCR 输出字数异常（识别 {len(value)} / 估计 {expected}）"
    if expected >= 5 and len(value) < max(1, int(expected * 0.28)):
        return False, 0.0, f"Manga OCR 严重缺字（识别 {len(value)} / 估计 {expected}）"
    return True, 0.88, ""


class MangaOcrSession(JsonWorkerSessionBase):
    def __init__(self, *, cancel_check=None, verbose: bool = True, load_progress_callback=None):
        super().__init__(cancel_check=cancel_check, verbose=verbose, worker_label="Manga OCR")
        self.load_progress_callback = load_progress_callback
        self.weights = Path()
        self.metadata_dir = Path()

    def _emit_load(self, stage: str, current: int, total: int, detail: str) -> None:
        callback = self.load_progress_callback
        if callable(callback):
            callback(stage, current, total, detail)

    def __enter__(self):
        self._emit_load("environment", 0, 1, "检查 Manga OCR 离线运行环境")
        python = setup_venv(verbose=self.verbose)
        self._emit_load("environment", 1, 1, "Manga OCR 运行环境已就绪 · 校验本地权重")
        self.weights, self.metadata_dir = resolve_model_assets()
        from adapters.subprocess_watchdog import isolated_process_kwargs
        self.proc = subprocess.Popen(
            [
                str(python), str(WORKER_SCRIPT), "--stream",
                "--weights", str(self.weights),
                "--metadata-dir", str(self.metadata_dir),
                "--device", str(os.environ.get("NOVEL_FORMATTER_MANGA_OCR_DEVICE", "auto") or "auto"),
            ],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, env=_worker_env(), **isolated_process_kwargs(),
        )
        assert self.proc.stderr is not None
        self._stderr_stop.clear()
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, args=(self.proc.stderr,), daemon=True,
            name="manga-ocr-stderr",
        )
        self._stderr_thread.start()
        self._emit_load("model", 0, 1, "正在加载 Manga OCR 模型")
        ready = self._read_response(timeout=180.0)
        if not ready.get("ready"):
            self.close(force=True)
            raise RuntimeError(str(ready.get("error") or "Manga OCR worker 未就绪"))
        self.device = str(ready.get("device") or "")
        self._emit_load("model", 1, 1, f"Manga OCR 已就绪 · {self.device or 'unknown'}")
        return self

    def _restart(self) -> None:
        self.close(force=True)
        if self.cancel_check is not None and self.cancel_check():
            raise RuntimeError("用户取消 Manga OCR")
        self.__enter__()

    def recognize(
        self,
        crop_paths: list[str],
        *,
        progress_callback=None,
        input_metadata: dict[str, dict] | None = None,
    ) -> dict[str, tuple[str, float, str | None]]:
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("Manga OCR worker 尚未启动")
        paths = [str(path) for path in crop_paths]
        metadata = {str(k): dict(v or {}) for k, v in (input_metadata or {}).items()}
        results: dict[str, tuple[str, float, str | None]] = {}
        total = max(1, len(paths))

        try:
            configured_batch = int(os.environ.get("NOVEL_FORMATTER_MANGA_OCR_REQUEST_BATCH", "0") or 0)
        except ValueError:
            configured_batch = 0
        default_batch = MANGA_OCR_CPU_BATCH if str(self.device).startswith("cpu") else MANGA_OCR_ACCEL_BATCH
        request_batch = max(1, min(16, configured_batch or default_batch))
        try:
            per_item_timeout = float(os.environ.get(
                "NOVEL_FORMATTER_MANGA_OCR_ITEM_TIMEOUT", str(MANGA_OCR_ITEM_TIMEOUT_SECONDS)
            ) or MANGA_OCR_ITEM_TIMEOUT_SECONDS)
        except ValueError:
            per_item_timeout = MANGA_OCR_ITEM_TIMEOUT_SECONDS
        per_item_timeout = max(4.0, min(120.0, per_item_timeout))
        try:
            num_beams = int(os.environ.get("NOVEL_FORMATTER_MANGA_OCR_BEAMS", "4") or 4)
        except ValueError:
            num_beams = 4
        num_beams = max(1, min(4, num_beams))

        with tempfile.TemporaryDirectory(prefix="novel_formatter_manga_ocr_review_") as td:
            root = Path(td)
            prepared: list[tuple[str, list[OcrCropSegment], str]] = []
            for index, source_path in enumerate(paths):
                if self.cancel_check is not None and self.cancel_check():
                    results[source_path] = ("", 0.0, "用户取消")
                    continue
                try:
                    item_meta = metadata.get(source_path, {})
                    if str(item_meta.get("layout") or "") == "horizontal_reflow":
                        # Not expected for a review role, but keep a safe direct
                        # route for diagnostics instead of re-segmenting horizontal text.
                        expected = max(1, int(item_meta.get("expected_chars") or 1))
                        segments = [OcrCropSegment(source_path, expected, 0, 0)]
                    else:
                        segments, _ = prepare_vertical_ocr_segments(
                            source_path,
                            root / f"i{index:05d}",
                            max_aspect=7.2,
                            max_chars=12,
                            already_isolated=True,
                            estimate_isolated_chars=True,
                        )
                    error = "" if segments else "Manga OCR 分歧列未检测到印刷文字"
                except Exception as exc:
                    segments = []
                    error = f"Manga OCR 短块切分失败: {exc}"
                prepared.append((source_path, list(segments), error))

            segment_lookup = {
                segment.path: segment
                for _source, segments, _error in prepared
                for segment in segments
            }
            cap_buckets: dict[int, list[str]] = {}
            for segment in segment_lookup.values():
                cap_buckets.setdefault(_decode_cap(segment.expected_chars), []).append(segment.path)
            returned: dict[str, dict] = {}
            errors: dict[str, str] = {}

            def run_bounded(batch_paths: list[str], cap: int) -> None:
                if not batch_paths:
                    return
                if self.cancel_check is not None and self.cancel_check():
                    for path in batch_paths:
                        errors[path] = "用户取消"
                    return
                if self.proc is None or self.proc.stdin is None:
                    try:
                        self.__enter__()
                    except Exception as exc:
                        for path in batch_paths:
                            errors[path] = f"Manga OCR worker 重启失败: {exc}"
                        return
                self._request_id += 1
                request_id = self._request_id
                timeout = per_item_timeout * max(1, len(batch_paths))
                request = {
                    "request_id": request_id,
                    "paths": batch_paths,
                    "max_length": cap,
                    "num_beams": num_beams,
                }
                try:
                    assert self.proc is not None and self.proc.stdin is not None
                    self.proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
                    self.proc.stdin.flush()
                    data = self._read_response(timeout=timeout)
                    if int(data.get("request_id", -1) or -1) != request_id:
                        raise RuntimeError("Manga OCR 请求响应串位")
                    if not data.get("ok"):
                        raise RuntimeError(str(data.get("error") or "未知错误"))
                    for item in data.get("items", []):
                        returned[str(item.get("path") or "")] = item
                    missing = [path for path in batch_paths if path not in returned]
                    if missing:
                        raise RuntimeError("Manga OCR 批请求缺少返回项")
                except Exception as exc:
                    message = (
                        f"Manga OCR 单条 watchdog 触发（batch={len(batch_paths)}, "
                        f"cap={cap}, beams={num_beams}, timeout={timeout:.0f}s）: {exc}"
                    )
                    try:
                        self._restart()
                    except Exception as restart_exc:
                        message += f"; worker 重启失败: {restart_exc}"
                        for path in batch_paths:
                            errors[path] = message
                        return
                    if len(batch_paths) > 1:
                        mid = max(1, len(batch_paths) // 2)
                        run_bounded(batch_paths[:mid], cap)
                        run_bounded(batch_paths[mid:], cap)
                    else:
                        errors[batch_paths[0]] = message

            for cap in sorted(cap_buckets):
                bucket = cap_buckets[cap]
                for offset in range(0, len(bucket), request_batch):
                    run_bounded(bucket[offset:offset + request_batch], cap)

            completed = 0
            for source_path, segments, prepare_error in prepared:
                if source_path in results:
                    completed += 1
                    continue
                failure = prepare_error
                parts: list[str] = []
                confidences: list[float] = []
                if not failure:
                    for segment in segments:
                        if segment.path in errors:
                            failure = errors[segment.path]
                            break
                        item = returned.get(segment.path)
                        if item is None:
                            failure = "Manga OCR 未返回短块"
                            break
                        text = str(item.get("text") or "").strip()
                        valid, confidence, reason = validate_manga_ocr_text(text, segment.expected_chars)
                        if not valid:
                            failure = reason
                            break
                        parts.append(compact_text(text))
                        confidences.append(confidence)
                if failure:
                    results[source_path] = ("", 0.0, failure)
                else:
                    text = "".join(parts).strip()
                    results[source_path] = (
                        text,
                        min(confidences) if confidences else 0.0,
                        None if text else "Manga OCR 未返回有效文字",
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
        return False


def recognize_crops(
    crop_paths: list[str], manifest_path: str, *, cancel_check=None, verbose: bool = True,
    input_metadata: dict[str, dict] | None = None,
) -> Iterator[tuple[str, list[dict] | None, str | None]]:
    del manifest_path
    ordered = [str(path) for path in crop_paths]
    metadata = {str(k): dict(v or {}) for k, v in (input_metadata or {}).items()}
    page_like = {path for path in ordered if looks_like_full_page(path)}
    safe = [path for path in ordered if path not in page_like]
    result: dict[str, tuple[str, float, str | None]] = {}
    if safe:
        with MangaOcrSession(cancel_check=cancel_check, verbose=verbose) as session:
            result = session.recognize(safe, input_metadata=metadata)
    for path in ordered:
        if path in page_like:
            yield path, None, "Manga OCR 仅用于分歧物理列复核，不接受整页图片。"
            continue
        text, confidence, error = result.get(path, ("", 0.0, "Manga OCR 未返回该区域"))
        if error:
            yield path, None, error
        else:
            yield path, ([{
                "text": text,
                "confidence": confidence,
                "confidence_kind": "heuristic",
                "box": None,
            }] if text else []), None


def run(*, verbose: bool = True, **kwargs):
    from adapters.column_ocr_adapter import run as run_column_ocr
    recognition_engine = str(kwargs.pop("recognition_engine", "manga_ocr") or "manga_ocr")
    if recognition_engine != "manga_ocr":
        raise ValueError("Manga OCR 复核适配器只能使用 recognition_engine='manga_ocr'")
    return run_column_ocr(recognition_engine="manga_ocr", verbose=verbose, **kwargs)
