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
import re
import subprocess
import tempfile
import threading
import time
import sys
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
SUPERNOVA_VENV_DIR = persistent_venv_dir("hayai-ocr-v2.7-supernova")
SUPERNOVA_MODEL = "JustANormalTinkerer/hayai-ocr-v2.7-supernova"
SUPERNOVA_REVISION = "517bdeeb64b68c08939e12f7211e99943c1e1640"
SUPERNOVA_RUNTIME_VERSION = f"v2.7-supernova@{SUPERNOVA_REVISION}"
WORKER_SCRIPT = Path(__file__).parent / "hayai_ocr_worker.py"
SUPERNOVA_TORCH_MAX_NUM_PATCHES = 256
SUPERNOVA_M6_TORCH_PACKAGE = os.environ.get("NOVEL_FORMATTER_HAYAI_TORCH_PACKAGE", "torch>=2.14,<2.15")


def _normalise_backend(value: str | None) -> str:
    # Legacy saved backend values migrate to the supported v2.7 model.
    return "supernova"


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


def _resolved_model_cache(backend: str = "supernova") -> Path:
    """Reuse a complete official v2.7 cache, never inspect old Hayai models."""
    from adapters.ocr_runtime_catalog import (
        _hayai_cache_roots, _hayai_repo_candidates, _hf_repo_dir,
        _supernova_cache_complete, _supernova_siglip_config_complete,
    )
    model = _supernova_model_location()
    local_model = Path(model).expanduser()
    if local_model.is_dir():
        return local_model.parent
    repo_dir = _hf_repo_dir(model)
    for root in _hayai_cache_roots():
        if (any(_supernova_cache_complete(path, revision=SUPERNOVA_REVISION)
                for path in _hayai_repo_candidates(root, repo_dir))
                and _supernova_siglip_config_complete(root)):
            return root
    override = os.environ.get("NOVEL_FORMATTER_HAYAI_OCR_CACHE_DIR", "").strip()
    return Path(override).expanduser() if override else ROOT / ".model-cache" / "hayai-supernova"


def _supernova_model_location() -> str:
    """Only the official upstream repo or a local folder is permitted."""
    model = os.environ.get("NOVEL_FORMATTER_HAYAI_SUPERNOVA_MODEL", SUPERNOVA_MODEL).strip() or SUPERNOVA_MODEL
    if model != SUPERNOVA_MODEL and not Path(model).expanduser().is_dir():
        raise ValueError("Supernova 模型来源必须是官方仓库或显式本地目录，拒绝未知远端代码：" + model)
    return model


def _project_supernova_python() -> Path:
    return ROOT / ".venv-hayai-supernova" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _probe_supernova_runtime(python: Path) -> bool:
    if not python.is_file():
        return False
    marker = (
        "import sys; assert (3,10) <= sys.version_info[:2] <= (3,14); "
        "import torch; from PIL import Image; from transformers import AutoModel, AutoProcessor; "
        "from importlib.metadata import version; "
        "assert 5 <= int(version('transformers').split('.')[0]) < 6"
    )
    # Runtime imports determine CPU availability; device selection checks MPS separately.
    # Do not mistake a Rosetta x86_64 interpreter for a working Apple MPS runtime.
    from adapters.runtime_env import _host_is_apple_silicon
    if _host_is_apple_silicon():
        marker += "; import platform; assert platform.machine().lower() in ('arm64','aarch64')"
    try:
        return subprocess.run(
            [str(python), "-c", marker], capture_output=True, timeout=60,
            env=dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PIP_NO_INDEX="1"),
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _supernova_effective_python() -> tuple[Path | None, str]:
    """Side-effect-free, fail-closed runtime selection for the v5 model."""
    explicit = os.environ.get("NOVEL_FORMATTER_HAYAI_SUPERNOVA_PYTHON", "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        return (path, "explicit") if _probe_supernova_runtime(path) else (None, "invalid-explicit")
    project = _project_supernova_python()
    if _probe_supernova_runtime(project):
        return project, "project"
    private = SUPERNOVA_VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if _probe_supernova_runtime(private):
        return private, "private"
    # Avoid treating an unrelated system installation as a managed OCR runtime.
    return None, ""


def _setup_supernova_venv(*, verbose: bool = True) -> Path:
    _supernova_model_location()  # Reject unapproved custom model URLs before installs.
    python, source = _supernova_effective_python()
    if python is not None:
        if verbose:
            print(f"[Hayai v2.7] reuse {source} runtime: {python}", flush=True)
        return python
    if source == "invalid-explicit":
        raise RuntimeError("Hayai v2.7 显式 Python 环境不可用；拒绝自动切换或安装")
    if _offline_runtime_requested():
        raise RuntimeError("Hayai v2.7 离线模式缺少可用 Transformers 5 独立环境；拒绝自动安装")
    packages = ["torch>=2.2", "transformers>=5,<6", "accelerate", "Pillow", "safetensors", "huggingface_hub"]
    if is_m6():
        packages[0] = SUPERNOVA_M6_TORCH_PACKAGE
    return ensure_venv(
        SUPERNOVA_VENV_DIR, label="Hayai OCR v2.7 Supernova / Transformers 5",
        marker_code=(
            "import torch; from transformers import AutoModel, AutoProcessor; "
            "from importlib.metadata import version; "
            "assert 5 <= int(version('transformers').split('.')[0]) < 6"
        ), packages=packages, verbose=verbose, min_minor=10, max_minor=14,
    )


def _offline_runtime_requested() -> bool:
    """One policy for dependency installation and Hugging Face worker access.

    PIP_NO_INDEX is treated as an explicit no-network request for the OCR
    component, even if the caller forgot to set the other offline flags.
    """
    yes = {"1", "true", "yes", "on"}
    no = {"0", "false", "no", "off"}
    return any(
        os.environ.get(name, "").strip().lower() in yes
        for name in ("NOVEL_FORMATTER_OCR_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PIP_NO_INDEX")
    ) or os.environ.get("NOVEL_FORMATTER_MODEL_AUTO_DOWNLOAD", "").strip().lower() in no


def setup_venv(*, verbose: bool = True, backend: str = "supernova") -> Path:
    """Old saved backend options migrate to Supernova without old installs."""
    return _setup_supernova_venv(verbose=verbose)


def _offline_cache_ready(backend: str) -> bool:
    component_id = "hayai_ocr_supernova"
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

        # Older column pipelines could overwrite a versioned v2.7 marker with
        # bare {ready:true}. Migrate only after independently verifying the
        # current Transformers 5 runtime and complete Supernova cache.
        # This lets an already-good installation become offline immediately
        # after upgrade without weakening the no-silent-download contract.
        payload = _read_ready_payload(component_id) or {}
        legacy_marker = bool(payload.get("ready")) and not str(payload.get("version") or "").strip()
        environment_ok = bool(_environment_installed(component_id, deep=False)[0]) if legacy_marker else False
        if legacy_marker and environment_ok and cache_ready:
            normalized_backend = "supernova"
            mark_runtime_ready(
                component_id,
                backend=normalized_backend,
                version=SUPERNOVA_RUNTIME_VERSION,
                cache_root=str(_resolved_model_cache(normalized_backend)),
                verified_stage="legacy_ready_marker_migrated",
            )
            return True
        return False
    except Exception:
        return False


def _worker_env(backend: str = "supernova") -> dict[str, str]:
    """Configure the sole supported v2.7 worker and its verified HF cache."""
    env = torch_worker_env(os.environ.copy())
    cache = _resolved_model_cache()
    cache.mkdir(parents=True, exist_ok=True)
    from adapters.ocr_runtime_catalog import (
        _hayai_repo_candidates, _hf_repo_dir, _supernova_cache_complete,
    )
    model_repo = _hf_repo_dir(SUPERNOVA_MODEL)
    verified_snapshot = next((path for path in _hayai_repo_candidates(cache, model_repo)
                              if _supernova_cache_complete(path, revision=SUPERNOVA_REVISION)), None)
    # Reuse the verified cache namespace (hub/ or transformers/) containing
    # the actual snapshot. Pinning a different empty namespace makes an offline
    # installation look missing to Hugging Face even though readiness passed.
    hub_cache = verified_snapshot.parent if verified_snapshot else cache / "hub"
    env["HF_HOME"] = str(cache)
    env["HF_HUB_CACHE"] = str(hub_cache)
    env["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    env["TRANSFORMERS_CACHE"] = str(hub_cache)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    env.setdefault("NOVEL_FORMATTER_HAYAI_OCR_BATCH", str(recommended_hayai_batch()))
    env.setdefault("NOVEL_FORMATTER_HAYAI_OCR_THREADS", str(recommended_cpu_threads()))
    explicit = str(env.get("NOVEL_FORMATTER_HAYAI_SUPERNOVA_MODEL") or "").strip()
    is_local = bool(explicit and Path(explicit).expanduser().is_dir())
    if _offline_runtime_requested():
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   PIP_NO_INDEX="1", NOVEL_FORMATTER_HAYAI_OFFLINE_ACTIVE="1")
    elif _offline_cache_ready("supernova") or is_local:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   NOVEL_FORMATTER_HAYAI_OFFLINE_ACTIVE="1")
    return env


def validate_hayai_ocr_text(text: str, expected_chars: int) -> tuple[bool, float, str]:
    """Reject obvious generative hallucination while preserving mixed Japanese text."""
    value = compact_text(text)
    if not value:
        return False, 0.0, "Hayai OCR 返回空文本"
    expected = max(1, int(expected_chars or 1))
    ratio = len(value) / expected
    # This model can emit a plausible Japanese prefix followed by a repeated
    # hallucination. Keep that prefix for visual review instead of accepting
    # the repeated tail as book text. Length checks alone miss shorter loops.
    if len(value) >= 10 and re.search(r"(.{1,3})\1{5,}", value):
        return False, 0.0, "Hayai OCR 疑似循环重复幻觉"
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
        # Source-crop keyed rejected model responses. These are untrusted OCR
        # candidates, NEVER automatically promoted to canonical book text.
        # The orchestration layer retains them in column review metadata.
        self.review_evidence: dict[str, list[dict]] = {}
        # Snapshot the transport batch before the worker/model consumes unified
        # memory.  Recomputing from live free memory after model load creates a
        # self-throttling loop on Apple Silicon: the model's own expected memory
        # footprint makes every later batch look like memory pressure.  Real OOM
        # protection remains in the worker and can still reduce effective_batch.
        self._startup_segment_batch = max(1, recommended_hayai_batch())
        self._batch_was_explicit = "NOVEL_FORMATTER_HAYAI_OCR_BATCH" in os.environ

    def __enter__(self):
        python = setup_venv(verbose=self.verbose, backend=self.backend)
        # v2.7 always uses non-quantized Supernova weights; old profile
        # quantization values are deliberately ignored during migration.
        self.requested_quantize = "none"
        device = str(self.options.get("device") or "auto").strip().lower()
        if device not in {"auto", "cpu", "cuda", "mps"}:
            device = "auto"
        max_new_tokens = _safe_int(self.options.get("max_new_tokens"), 128, minimum=32, maximum=192)
        cmd = [
            str(python), str(WORKER_SCRIPT), "--stream", "--backend", "supernova",
            "--device", device,
            "--max-new-tokens", str(max_new_tokens),
            "--max-num-patches", str(SUPERNOVA_TORCH_MAX_NUM_PATCHES),
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
        if "mps" in self.device.lower() and not self._batch_was_explicit:
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
            component_id = "hayai_ocr_supernova"
            mark_runtime_ready(
                component_id,
                backend=self.backend,
                device=self.device,
                version=SUPERNOVA_RUNTIME_VERSION,
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
        default_batch = self._startup_segment_batch
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
                    # v2.7 evidence-preserving guard: rejection of ONE segment
                    # must not erase successfully decoded neighboring segments.
                    # Unsafe output is always marked for review, not accepted as
                    # authoritative text. A visible square marks missing text.
                    self.review_evidence.pop(source_path, None)
                    failure = prepare_error
                    column_texts: dict[int, list[str]] = {}
                    confidences: list[float] = []
                    rejected: list[dict] = []
                    if not failure:
                        for segment in segments:
                            data = returned.get(segment.path)
                            if data is None:
                                reason = segment_errors.get(
                                    segment.path, f"Hayai OCR 未返回分段: {Path(segment.path).name}"
                                )
                                raw_text = ""
                            elif not data.get("ok"):
                                reason = str(data.get("error") or "未知错误")
                                raw_text = ""
                            else:
                                blocks = data.get("blocks") or []
                                raw_text = "".join(
                                    str(item.get("text") or "").strip()
                                    for item in blocks
                                    if str(item.get("text") or "").strip()
                                ).strip()
                                valid, confidence, reason = validate_hayai_ocr_text(
                                    raw_text, segment.expected_chars
                                )
                                if valid:
                                    column_texts.setdefault(segment.column_index, []).append(compact_text(raw_text))
                                    confidences.append(confidence)
                                    continue
                            compact = compact_text(raw_text)
                            # Keep the unmodified model response for original-image
                            # AI adjudication. An arbitrary length cap is reported,
                            # not silently hidden. No private page data in stderr.
                            rejected.append({
                                "segment_index": int(segment.segment_index),
                                "column_index": int(segment.column_index),
                                "expected_chars": int(segment.expected_chars),
                                "reason": str(reason),
                                "raw_text": raw_text[:4096],
                                "raw_truncated": len(raw_text) > 4096,
                            })
                            # A short but Japanese-looking response can contain a
                            # valid beginning. Preserve it with explicit missing-
                            # text marker, *never* as a complete high-confidence
                            # candidate. Overlong/foreign/repetitive hallucination
                            # is evidence only, not injected into output prose.
                            plausible_prefix = bool(
                                compact and len(compact) <= max(56, int(segment.expected_chars * 2.35 + 10))
                                and japanese_ratio(compact) >= 0.42
                                and 0 < len(compact) < max(1, segment.expected_chars * 0.26)
                            )
                            safe_prefix = compact if plausible_prefix else ""
                            if "循环重复幻觉" in str(reason) and japanese_ratio(compact) >= 0.42:
                                repeat = re.search(r"(.{1,3})\1{5,}", compact)
                                if repeat is not None and repeat.start() >= 3:
                                    safe_prefix = compact[:repeat.start()]
                            guarded = safe_prefix + "□"
                            column_texts.setdefault(segment.column_index, []).append(guarded)
                            confidences.append(0.0)
                    if rejected:
                        self.review_evidence[source_path] = rejected
                    if failure:
                        results[source_path] = ("", 0.0, failure)
                    else:
                        ordered_columns = [
                            "".join(column_texts[index])
                            for index in sorted(column_texts)
                            if column_texts.get(index)
                        ]
                        text = ("\n" if column_count > 1 else "").join(ordered_columns).strip()
                        # Placeholder-only crops must remain unresolved even
                        # after a model returned an obviously unsafe response.
                        if text and any(ch not in "□\n " for ch in text):
                            results[source_path] = (
                                text,
                                0.0 if rejected else (min(confidences) if confidences else 0.0),
                                None,
                            )
                        else:
                            results[source_path] = ("", 0.0, rejected[0]["reason"] if rejected else "Hayai OCR 未返回有效文字")
                    completed += 1
                    if callable(progress_callback):
                        progress_callback(completed, total, source_path)
        return results

    def close(self, *, force: bool = False) -> None:
        """Stop the persistent worker with strictly bounded teardown.

        Pipe reader threads can be blocked inside ``readline()``. Closing their
        TextIOWrapper from another thread may then wait forever on the wrapper's
        internal lock, which used to make Stop/close look like a frozen GUI.
        Terminate the process (or request graceful shutdown) first, wait only for
        bounded intervals, and close a read pipe only after its reader exited.
        """
        proc = self.proc
        self.proc = None
        if proc is None:
            return

        from adapters.subprocess_watchdog import terminate_process
        shutdown_requested = False
        termination_requested = False
        initial_ret = proc.poll()
        ret = initial_ret

        if initial_ret is None and not force and proc.stdin is not None:
            try:
                proc.stdin.write(json.dumps({"command": "close"}) + "\n")
                proc.stdin.flush()
                shutdown_requested = True
            except Exception:
                pass

        if initial_ret is None:
            if force:
                termination_requested = True
                ret = terminate_process(proc, grace=1.5)
            elif shutdown_requested:
                try:
                    ret = proc.wait(timeout=4.0)
                except subprocess.TimeoutExpired:
                    termination_requested = True
                    ret = terminate_process(proc, grace=1.5)
                except Exception:
                    termination_requested = True
                    ret = terminate_process(proc, grace=1.5)
            else:
                termination_requested = True
                ret = terminate_process(proc, grace=1.5)

        # Writers are safe to close once shutdown has completed/been signalled.
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass

        self._stderr_stop.set()
        stderr_thread = self._stderr_thread
        if stderr_thread is not None and stderr_thread.is_alive():
            stderr_thread.join(timeout=1.0)
        # Never close a TextIOWrapper underneath a thread still blocked in
        # readline(); leave that daemon reader to observe EOF instead.
        if stderr_thread is None or not stderr_thread.is_alive():
            try:
                if proc.stderr:
                    proc.stderr.close()
            except Exception:
                pass
        self._stderr_thread = None

        if self._stdout_pump is not None:
            self._stdout_pump.close()
            self._stdout_pump = None
        else:
            try:
                if proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass

        intentional = bool(force or shutdown_requested or termination_requested)
        if ret not in (0, -15, -9, None) and not intentional:
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
            component_id = "hayai_ocr_supernova"
            mark_runtime_ready(
                component_id,
                backend=self.backend,
                device=self.device,
                version=SUPERNOVA_RUNTIME_VERSION,
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
