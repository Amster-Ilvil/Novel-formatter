#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hayai OCR adapter using the existing physical-column pipeline.

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
import time
from pathlib import Path
from typing import Iterator

from adapters.runtime_env import ensure_venv, persistent_venv_dir
from adapters.ocr_prefetch import accelerator_prefetch_enabled, iter_prefetched_windows, windowed
from adapters.vertical_crop_segments import (
    OcrCropSegment,
    compact_text,
    japanese_ratio,
    looks_like_full_page,
    prepare_vertical_ocr_segments,
)
from utils.apple_silicon_runtime import (
    is_m6,
    recommended_cpu_threads,
    recommended_hayai_batch,
    recommended_hayai_segment_chars,
    torch_worker_env,
)

ROOT = Path(__file__).parent.parent
VENV_DIR = persistent_venv_dir("hayai-ocr-v2.1")
WORKER_SCRIPT = Path(__file__).parent / "hayai_ocr_worker.py"
MODEL_CACHE = ROOT / ".model-cache" / "hayai-ocr"
HAYAI_OCR_VERSION = "2.3.0"
# Phase26 role-fit contract: Hayai v2.5-nova always uses the proven 512-patch
# NaFlex budget.  Upstream nova currently defaults to 512 too, but making the
# value explicit prevents an inherited HAYAI_MAX_NUM_PATCHES environment value
# or a future upstream default change from silently altering OCR quality.
HAYAI_TORCH_MAX_NUM_PATCHES = 512
HAYAI_OCR_PACKAGE = os.environ.get(
    "NOVEL_FORMATTER_HAYAI_OCR_PACKAGE", f"hayai-ocr=={HAYAI_OCR_VERSION}"
)
# Hayai v2 uses SigLIP2/Transformers 4 APIs. Keep the independent runtime out
# of a future Transformers major-version migration until this adapter is
# explicitly revalidated; this cannot affect any other OCR venv.
HAYAI_TRANSFORMERS_PACKAGE = os.environ.get(
    "NOVEL_FORMATTER_HAYAI_TRANSFORMERS_PACKAGE", "transformers>=4.49,<5"
)
# PyTorch 2.14 adds current Apple-Silicon/MPS kernel improvements. Keep the
# pin scoped to the isolated Hayai environment and allow an explicit override.
HAYAI_TORCH_PACKAGE = os.environ.get(
    "NOVEL_FORMATTER_HAYAI_TORCH_PACKAGE", "torch>=2.14,<2.15"
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


def _resolved_model_cache(backend: str = "torch") -> Path:
    """Use the same already-populated cache root that runtime detection found.

    Prefer the project-local cache for new downloads, but transparently reuse a
    complete HF_HOME / ~/.cache/huggingface installation.  This keeps the model
    detector and the actual Hayai worker from disagreeing about where the model
    lives.
    """
    override = os.environ.get("NOVEL_FORMATTER_HAYAI_OCR_CACHE_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    try:
        from adapters.ocr_runtime_catalog import _hayai_cache_roots, _hayai_litert_cache_complete, _hayai_processor_cache_complete, _hayai_repo_candidates, _hayai_torch_cache_complete, _hf_repo_dir
        roots = _hayai_cache_roots()
        if _normalise_backend(backend) == "litert":
            repo_id = os.environ.get(
                "NOVEL_FORMATTER_HAYAI_OCR_LITERT_REPO", "JustANormalTinkerer/hayai-ocr-v2-tflite"
            ).strip() or "JustANormalTinkerer/hayai-ocr-v2-tflite"
            repo_dir = _hf_repo_dir(repo_id)
            for root in roots:
                if any(_hayai_litert_cache_complete(path) for path in _hayai_repo_candidates(root, repo_dir)):
                    return root
        else:
            model_name = os.environ.get(
                "NOVEL_FORMATTER_HAYAI_OCR_MODEL", "JustANormalTinkerer/hayai-ocr-v2.5-nova"
            ).strip() or "JustANormalTinkerer/hayai-ocr-v2.5-nova"
            local_model = Path(model_name).expanduser()
            repo_dir = _hf_repo_dir(model_name) if not local_model.exists() else ""
            for root in roots:
                processor_ready = _hayai_processor_cache_complete(root)
                model_ready = local_model.exists() and _hayai_torch_cache_complete(local_model)
                if repo_dir:
                    model_ready = any(
                        _hayai_torch_cache_complete(path)
                        for path in _hayai_repo_candidates(root, repo_dir)
                    )
                if model_ready and processor_ready:
                    return root
    except Exception:
        pass
    return MODEL_CACHE


def setup_venv(*, verbose: bool = True, backend: str = "torch") -> Path:
    backend = _normalise_backend(backend)
    # Offline/cloud acceptance can reuse an already-provisioned exact Hayai
    # runtime instead of invoking pip. Normal desktop installs keep the private
    # venv path unless this advanced override is explicitly set.
    explicit = os.environ.get("NOVEL_FORMATTER_HAYAI_OCR_PYTHON", "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            return path
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
    if is_m6() and backend == "torch":
        marker += (
            "; v=version('torch').split('+',1)[0].split('.'); "
            "n=tuple(int(''.join(c for c in p if c.isdigit()) or 0) for p in v[:2]); "
            "assert n >= (2,14), 'M6 Hayai requires PyTorch 2.14+'"
        )
    if backend == "litert":
        marker += "; import ai_edge_litert"
    packages = [package, HAYAI_TRANSFORMERS_PACKAGE]
    if is_m6() and backend == "torch":
        packages.insert(0, HAYAI_TORCH_PACKAGE)
    return ensure_venv(
        VENV_DIR,
        label="Hayai OCR / PyTorch",
        marker_code=marker,
        packages=packages,
        verbose=verbose,
        min_minor=10,
        max_minor=13,
    )


def _offline_cache_ready(backend: str) -> bool:
    component_id = "hayai_ocr_litert" if _normalise_backend(backend) == "litert" else "hayai_ocr"
    try:
        from adapters.ocr_runtime_catalog import (
            _environment_installed, _model_cache_ready, _read_ready_payload,
            _state_marker_ready, mark_runtime_ready,
        )
        # The successful-run marker suppresses the GUI install prompt, but only
        # force HF offline mode when a fresh static scan can still see reusable
        # assets.  If the user manually deleted/moved the cache, allow upstream
        # to repair/download instead of trapping the worker in offline mode.
        cache_ready = bool(_model_cache_ready(component_id)[0])
        if _state_marker_ready(component_id) and cache_ready:
            return True

        # r3 and older column pipelines could overwrite Hayai's rich success
        # marker with a bare {ready:true}.  Migrate that known legacy shape on
        # first start, but only after independently verifying BOTH the exact
        # pinned Hayai package and the complete local model/processor cache.
        # This lets an already-good installation become offline immediately
        # after upgrade without weakening the no-silent-download contract.
        payload = _read_ready_payload(component_id) or {}
        legacy_marker = bool(payload.get("ready")) and not str(payload.get("version") or "").strip()
        environment_ok = bool(_environment_installed(component_id, deep=False)[0]) if legacy_marker else False
        if legacy_marker and environment_ok and cache_ready:
            normalized_backend = _normalise_backend(backend)
            mark_runtime_ready(
                component_id,
                backend=normalized_backend,
                version=HAYAI_OCR_VERSION,
                cache_root=str(_resolved_model_cache(normalized_backend)),
                verified_stage="legacy_ready_marker_migrated",
            )
            return True
        return False
    except Exception:
        return False


def _worker_env(backend: str = "torch") -> dict[str, str]:
    env = torch_worker_env(os.environ.copy()) if backend == "torch" else os.environ.copy()
    cache = _resolved_model_cache(backend)
    cache.mkdir(parents=True, exist_ok=True)
    # Pin the worker to the exact cache root selected by runtime detection.
    # New downloads still prefer the project-local cache, while an already
    # complete HF_HOME / ~/.cache/huggingface snapshot is reused without a
    # second download or a repeated install-confirmation dialog.
    env["HF_HOME"] = str(cache)
    env["HUGGINGFACE_HUB_CACHE"] = str(cache / "hub")
    env["TRANSFORMERS_CACHE"] = str(cache / "transformers")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if backend == "torch":
        # All platforms use the live hardware detector. Apple Silicon gets MPS/MLX
        # tuning from the same helper, while Windows/Linux may choose CUDA/CPU.
        env.setdefault("NOVEL_FORMATTER_HAYAI_OCR_BATCH", str(recommended_hayai_batch()))
        env.setdefault("NOVEL_FORMATTER_HAYAI_OCR_THREADS", str(recommended_cpu_threads()))
    # After one successful inference, keep the verified cached snapshot immutable
    # during ordinary OCR starts.  This enforces the project's "no silent model
    # update on startup" contract; explicit cache removal/manual maintenance is
    # required before a new upstream snapshot can be fetched.
    if _offline_cache_ready(backend):
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["NOVEL_FORMATTER_HAYAI_OFFLINE_ACTIVE"] = "1"
    return env


def validate_hayai_ocr_text(text: str, expected_chars: int) -> tuple[bool, float, str]:
    """Reject obvious generative hallucination while preserving mixed Japanese text."""
    value = compact_text(text)
    if not value:
        return False, 0.0, "Hayai OCR 返回空文本"
    expected = max(1, int(expected_chars or 1))
    ratio = len(value) / expected
    if len(value) > max(56, int(expected * 2.35 + 10)):
        return False, 0.0, f"Hayai OCR 输出字数异常（识别 {len(value)} / 字形估计 {expected}）"
    if ratio < 0.26:
        return False, 0.0, f"Hayai OCR 严重缺字（识别 {len(value)} / 字形估计 {expected}）"
    if japanese_ratio(value) < 0.42:
        return False, 0.0, "Hayai OCR 输出的日文/CJK 字符比例异常"
    # Heuristic confidence only.  It is intentionally not presented as a model probability.
    confidence = 0.80 if 0.56 <= ratio <= 1.65 else 0.60
    return True, confidence, ""


class HayaiOcrSession:
    """One persistent Hayai model shared by primary, rescue and sentence crops."""

    def __init__(self, *, engine_options: dict | None = None, cancel_check=None, verbose: bool = True,
                 load_progress_callback=None):
        self.options = dict(engine_options or {})
        self.cancel_check = cancel_check
        self.verbose = verbose
        self.load_progress_callback = load_progress_callback
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
        # Snapshot the transport batch before the worker/model consumes unified
        # memory.  Recomputing from live free memory after model load creates a
        # self-throttling loop on Apple Silicon: the model's own expected memory
        # footprint makes every later batch look like memory pressure.  Real OOM
        # protection remains in the worker and can still reduce effective_batch.
        self._startup_segment_batch = 4 if self.backend == "litert" else max(1, recommended_hayai_batch())
        self._batch_was_explicit = "NOVEL_FORMATTER_HAYAI_OCR_BATCH" in os.environ

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
            "--max-num-patches", str(HAYAI_TORCH_MAX_NUM_PATCHES if self.backend == "torch" else 256),
        ]
        from adapters.subprocess_watchdog import LinePump, isolated_process_kwargs
        worker_env = _worker_env(self.backend)
        worker_env["NOVEL_FORMATTER_HAYAI_STATUS"] = "1"
        try:
            self._startup_segment_batch = max(
                1, min(32, int(worker_env.get("NOVEL_FORMATTER_HAYAI_OCR_BATCH", self._startup_segment_batch)))
            )
        except (TypeError, ValueError):
            pass
        if callable(self.load_progress_callback):
            if worker_env.get("NOVEL_FORMATTER_HAYAI_OFFLINE_ACTIVE") == "1":
                self.load_progress_callback("model", 0, 1, "Hayai 本地缓存已完整验证 · 强制离线加载")
            else:
                self.load_progress_callback("model", 0, 1, "Hayai 缓存未通过离线锁定条件 · 允许上游解析/修复缓存")
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=worker_env,
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
            timeout=env_seconds("NOVEL_FORMATTER_OCR_STARTUP_TIMEOUT", 900.0, minimum=60.0),
            status_callback=self._on_loading_status,
        )
        if not ready.get("ready"):
            self.close(force=True)
            raise RuntimeError(str(ready.get("error") or "Hayai OCR worker 未就绪"))
        self.device = str(ready.get("device") or "")
        self.backend = _normalise_backend(ready.get("backend") or self.backend)
        # Match the proven fast pre-regression MPS operating point unless the
        # user explicitly overrides it.  A lower pre-load recommendation is
        # retained; worker-side OOM backoff can still lower it further.
        if "mps" in self.device.lower() and not self._batch_was_explicit and self.backend != "litert":
            self._startup_segment_batch = min(self._startup_segment_batch, 8)
        self.effective_quantize = str(ready.get("effective_quantize") or ready.get("quantize") or "none")
        self.startup_warning = str(ready.get("warning") or "").strip()
        if self.startup_warning and self.verbose:
            print(f"[Hayai OCR] {self.startup_warning}")

        # The model is already loaded at this point.  Record that fact now, not
        # only when the context manager exits: a later user cancel, downstream
        # crop error, or GUI close must not make the next run ask to "install"
        # a Hayai model that has just proven it can load successfully.
        try:
            from adapters.ocr_runtime_catalog import mark_runtime_ready
            component_id = "hayai_ocr_litert" if self.backend == "litert" else "hayai_ocr"
            mark_runtime_ready(
                component_id,
                backend=self.backend,
                device=self.device,
                version=HAYAI_OCR_VERSION,
                cache_root=str(_resolved_model_cache(self.backend)),
                requested_quantize=self.requested_quantize,
                effective_quantize=self.effective_quantize,
                warning=self.startup_warning,
                verified_stage="worker_model_loaded",
            )
        except Exception:
            pass
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

    def _on_loading_status(self, status: dict) -> None:
        if callable(self.load_progress_callback):
            elapsed = int(status.get("elapsed_seconds", 0) or 0)
            phase = str(status.get("status") or "").strip()
            labels = {
                "startup": "Hayai worker 启动",
                "import_torch": "导入 PyTorch",
                "resolve_device": "检测 CUDA/MPS/CPU",
                "import_hayai": "导入 Hayai/Transformers",
                "device_resolved": "Hayai 推理设备已解析",
                "load_model": "解析本地缓存并加载 Hayai 权重",
                "load_model_unquantized_retry": "量化回退后重新加载 Hayai 权重",
                "configure_runtime": "配置推理运行时",
                "model_loading": "加载 Hayai 模型",
            }
            label = labels.get(phase, phase or "Hayai 启动")
            if phase == "device_resolved":
                effective = str(status.get("effective_device") or "cpu").upper()
                torch_version = str(status.get("torch_version") or "未知")
                mps = "可用" if status.get("mps_available") else ("已编译但不可用" if status.get("mps_built") else "未编译")
                offline = "离线" if status.get("offline_mode") else "在线解析允许"
                machine = str(status.get("machine") or "").strip()
                probe = str(status.get("mps_probe_detail") or "").strip()
                arch = f" · Python {machine}" if machine else ""
                probe_note = f" · {probe}" if probe and not status.get("mps_available") else ""
                label = f"Hayai 设备：{effective} · PyTorch {torch_version}{arch} · MPS {mps}{probe_note} · {offline}"
            suffix = f" · 已等待 {elapsed} 秒" if elapsed > 0 else ""
            self.load_progress_callback("model", 0, 1, f"{label}{suffix}")

    def _read_response(self, timeout: float | None = None, *, status_callback=None) -> dict:
        if self.proc is None or self.proc.stdout is None:
            raise RuntimeError("Hayai OCR worker 尚未启动")
        from adapters.subprocess_watchdog import LinePump, env_seconds
        if self._stdout_pump is None:
            self._stdout_pump = LinePump(self.proc.stdout, name="hayai-ocr-stdout")
        wait_seconds = float(timeout) if timeout is not None else env_seconds(
            "NOVEL_FORMATTER_OCR_REQUEST_TIMEOUT", 300.0, minimum=30.0
        )
        deadline = time.monotonic() + wait_seconds
        while True:
            line = self._stdout_pump.readline(
                proc=self.proc,
                timeout=max(0.01, deadline - time.monotonic()),
                cancel_check=self.cancel_check,
                label="Hayai OCR",
            )
            if line is None:
                tail = "\n".join(self._stderr_lines[-30:])
                raise RuntimeError(f"Hayai OCR worker 提前退出 (code={self.proc.poll()})\n{tail}")
            try:
                data = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"Hayai OCR worker 返回无效 JSON: {line[:300]} ({exc})") from exc
            if isinstance(data, dict) and data.get("status") in {
                "startup", "import_torch", "resolve_device", "device_resolved", "import_hayai",
                "load_model", "load_model_unquantized_retry", "configure_runtime",
                "model_loading", "recognition",
            }:
                if callable(status_callback):
                    status_callback(data)
                continue
            return data

    def _restart_worker_after_timeout(self) -> None:
        """Kill a stuck Hayai generation request and start a clean worker."""
        self.close(force=True)
        if self.cancel_check is not None and self.cancel_check():
            raise RuntimeError("用户取消 Hayai OCR")
        self.__enter__()

    def recognize(
        self, crop_paths: list[str], *, progress_callback=None,
        input_metadata: dict[str, dict] | None = None,
    ) -> dict[str, tuple[str, float, str | None]]:
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("Hayai OCR worker 尚未启动")
        ordered_paths = [str(path) for path in crop_paths]
        metadata = {str(k): dict(v or {}) for k, v in (input_metadata or {}).items()}
        results: dict[str, tuple[str, float, str | None]] = {}
        total = max(1, len(ordered_paths))
        try:
            source_window = int(os.environ.get("NOVEL_FORMATTER_HAYAI_OCR_SOURCE_WINDOW", "24") or 24)
        except ValueError:
            source_window = 24
        if "mps" in self.device.lower() and "NOVEL_FORMATTER_HAYAI_OCR_SOURCE_WINDOW" not in os.environ:
            source_window = max(24, recommended_hayai_segment_chars())
        source_window = max(2, min(48, source_window))
        default_batch = 4 if self.backend == "litert" else self._startup_segment_batch
        try:
            segment_batch = int(os.environ.get("NOVEL_FORMATTER_HAYAI_OCR_BATCH", str(default_batch)) or default_batch)
        except ValueError:
            segment_batch = default_batch
        segment_batch = max(1, min(32, segment_batch))
        # Conflict-only review is sparse and must be failure-isolated. A single
        # pathological crop can otherwise poison a large Hayai batch, forcing
        # the parent watchdog to kill/reload the model and re-run healthy peers.
        # Keep ordinary full-column OCR batched, but default selective review to
        # one segment/request unless the user explicitly overrides the batch env.
        selective_review = bool(self.options.get("column_target_ids")) or bool(
            self.options.get("review_only") or self.options.get("disagreement_review")
        )
        if selective_review and "NOVEL_FORMATTER_HAYAI_OCR_BATCH" not in os.environ:
            segment_batch = 1
        max_chars = _safe_int(
            self.options.get("segment_max_chars"),
            recommended_hayai_segment_chars(),
            minimum=12,
            maximum=40,
        )
        max_aspect = _safe_float(self.options.get("segment_max_aspect"), 16.0, minimum=6.0, maximum=18.0)

        with tempfile.TemporaryDirectory(prefix="novel_formatter_hayai_chunks_") as temp_dir:
            chunk_root = Path(temp_dir)
            completed = 0

            def prepare_window(window_start: int, window):
                prepared: list[tuple[str, list, int, str]] = []
                flat_segments = []
                for local_index, source_path in enumerate(window, start=1):
                    global_index = window_start + local_index
                    try:
                        item_meta = metadata.get(str(source_path), {})
                        if str(item_meta.get("layout") or "") == "horizontal_reflow":
                            # This line has already been rebuilt from verified
                            # fixed physical cells.  Keep the exact horizontal
                            # geometry and send it directly through the existing
                            # Hayai worker instead of running the vertical splitter
                            # a second time.
                            expected_chars = max(1, int(item_meta.get("expected_chars") or 1))
                            segments = [OcrCropSegment(str(source_path), expected_chars, 0, 0)]
                            column_count = 1
                        else:
                            segments, column_count = prepare_vertical_ocr_segments(
                                source_path,
                                chunk_root / f"i{global_index:05d}",
                                already_isolated=True,
                                estimate_isolated_chars=True,
                                physical_column_boxes=item_meta.get("physical_column_boxes") or (),
                                max_chars=max_chars,
                                max_aspect=max_aspect,
                            )
                        error = "" if segments else "Hayai OCR 输入区域没有检测到印刷文字"
                    except Exception as exc:
                        segments, column_count = [], 0
                        error = f"Hayai OCR 输入分段失败: {exc}"
                    prepared.append((source_path, list(segments), int(column_count), error))
                    flat_segments.extend(segments)
                return prepared, flat_segments

            windows = windowed(ordered_paths, source_window)
            prefetch = accelerator_prefetch_enabled(
                self.device, os.environ.get("NOVEL_FORMATTER_OCR_DOUBLE_BUFFER", "auto")
            )
            for window_start, window, prepared_payload in iter_prefetched_windows(
                windows, prepare_window, enabled=prefetch
            ):
                if self.cancel_check is not None and self.cancel_check():
                    break
                prepared, flat_segments = prepared_payload

                returned: dict[str, dict] = {}
                segment_errors: dict[str, str] = {}
                try:
                    per_item_timeout = float(os.environ.get(
                        "NOVEL_FORMATTER_HAYAI_OCR_ITEM_TIMEOUT", "20"
                    ) or 20)
                except ValueError:
                    per_item_timeout = 20.0
                per_item_timeout = max(5.0, min(120.0, per_item_timeout))

                def run_bounded_batch(batch) -> None:
                    nonlocal segment_batch
                    if not batch:
                        return
                    batch_paths = [str(segment.path) for segment in batch]
                    if self.cancel_check is not None and self.cancel_check():
                        for path in batch_paths:
                            segment_errors[path] = "用户取消"
                        return
                    if self.proc is None or self.proc.stdin is None:
                        try:
                            self._restart_worker_after_timeout()
                        except Exception as exc:
                            for path in batch_paths:
                                segment_errors[path] = f"Hayai OCR worker 重启失败: {exc}"
                            return
                    self._request_id += 1
                    request_id = self._request_id
                    timeout = per_item_timeout * max(1, len(batch_paths))
                    try:
                        assert self.proc is not None and self.proc.stdin is not None
                        self.proc.stdin.write(json.dumps({
                            "request_id": request_id,
                            "paths": batch_paths,
                        }, ensure_ascii=False) + "\n")
                        self.proc.stdin.flush()
                        data = self._read_response(
                            timeout=timeout,
                            status_callback=(
                                lambda _status: progress_callback(completed, total, "")
                                if callable(progress_callback) else None
                            ),
                        )
                        if int(data.get("request_id", -1) or -1) != request_id:
                            raise RuntimeError("Hayai OCR 请求响应串位")
                        if not data.get("ok"):
                            raise RuntimeError(str(data.get("error") or "未知错误"))
                        items = data.get("items")
                        if not isinstance(items, list):
                            raise RuntimeError("Hayai OCR 批量响应缺少 items")
                        if len(items) != len(batch_paths):
                            raise RuntimeError(
                                f"Hayai OCR 批量响应数量异常：输入 {len(batch_paths)}，返回 {len(items)}"
                            )
                        batch_returned: set[str] = set()
                        for item in items:
                            if not isinstance(item, dict):
                                raise RuntimeError("Hayai OCR 批量响应包含非对象条目")
                            item_path = str(item.get("path") or "")
                            if item_path not in batch_paths or item_path in batch_returned:
                                raise RuntimeError("Hayai OCR 批量响应路径异常或重复")
                            batch_returned.add(item_path)
                            returned[item_path] = item
                        effective = int(data.get("effective_batch", 0) or 0)
                        if effective > 0:
                            segment_batch = max(1, min(segment_batch, effective))
                    except Exception as exc:
                        message = (
                            f"Hayai OCR 单条 watchdog 触发"
                            f"（batch={len(batch_paths)}, timeout={timeout:.0f}s）: {exc}"
                        )
                        try:
                            self._restart_worker_after_timeout()
                        except Exception as restart_exc:
                            message += f"; worker 重启失败: {restart_exc}"
                            for path in batch_paths:
                                segment_errors[path] = message
                            return
                        if len(batch) > 1:
                            mid = max(1, len(batch) // 2)
                            run_bounded_batch(batch[:mid])
                            run_bounded_batch(batch[mid:])
                        else:
                            segment_errors[batch_paths[0]] = message

                offset = 0
                while offset < len(flat_segments):
                    batch_size = max(1, min(segment_batch, len(flat_segments) - offset))
                    batch = flat_segments[offset:offset + batch_size]
                    run_bounded_batch(batch)
                    offset += len(batch)

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
                                failure = segment_errors.get(
                                    segment.path, f"Hayai OCR 未返回分段: {Path(segment.path).name}"
                                )
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
                            column_texts.setdefault(segment.column_index, []).append(compact_text(text))
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
                cache_root=str(_resolved_model_cache(self.backend)),
                requested_quantize=self.requested_quantize,
                effective_quantize=self.effective_quantize,
                warning=self.startup_warning,
                verified_stage="session_completed",
            )
        return False


def recognize_crops(
    crop_paths: list[str],
    manifest_path: str,
    *,
    cancel_check=None,
    verbose: bool = True,
    engine_options: dict | None = None,
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
        with HayaiOcrSession(
            engine_options=engine_options,
            cancel_check=cancel_check,
            verbose=verbose,
        ) as session:
            results = session.recognize(safe_paths, input_metadata=metadata)
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
