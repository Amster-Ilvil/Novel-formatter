#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small persistent calibration store for OCR runtime choices.

The store only records non-sensitive performance/resource observations keyed by
an anonymized local hardware fingerprint.  It never stores OCR text or images.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path
import threading
from typing import Any

_LOCK = threading.RLock()
_MAX_ENGINES = 8


def _state_root() -> Path:
    override = os.environ.get("NOVEL_FORMATTER_OCR_CALIBRATION_HOME", "").strip()
    if override:
        return Path(override).expanduser()
    home = Path.home()
    if sys_platform() == "darwin":
        return home / "Library" / "Caches" / "NovelFormatter" / "ocr-calibration"
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA", "").strip()
        return (Path(base) if base else home / "AppData" / "Local") / "NovelFormatter" / "ocr-calibration"
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    return (Path(xdg).expanduser() if xdg else home / ".cache") / "novel-formatter" / "ocr-calibration"


def sys_platform() -> str:
    return platform.system().lower()


def _hardware_fingerprint_payload() -> dict[str, Any]:
    try:
        from utils.hardware_runtime import detect_system_hardware
        profile = detect_system_hardware(refresh=False)
        return {
            "platform": profile.platform,
            "arch": profile.arch,
            "cpu_threads": profile.cpu_threads,
            "total_memory_gb": profile.total_memory_gb,
            "apple_chip": profile.apple_chip,
            "apple_family": profile.apple_family,
            "gpu_name": profile.gpu_name,
            "gpu_backend": profile.gpu_backend,
            "gpu_memory_gb": profile.gpu_memory_gb,
            "onnx_providers": list(profile.onnx_providers),
        }
    except Exception:
        return {
            "platform": platform.system().lower(),
            "arch": platform.machine().lower(),
            "cpu_threads": int(os.cpu_count() or 1),
        }


def hardware_fingerprint() -> str:
    payload = json.dumps(_hardware_fingerprint_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _path() -> Path:
    return _state_root() / f"{hardware_fingerprint()}.json"


def load() -> dict[str, Any]:
    path = _path()
    try:
        with _LOCK:
            data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _bounded_number(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return default
    return parsed


def record(
    engine: str,
    *,
    batch_size: int,
    elapsed_seconds: float,
    items: int,
    available_memory_gb: float,
    success: bool,
    resource_error: bool = False,
) -> None:
    """Record a batch observation and remember a safe ceiling after a resource error."""
    name = str(engine or "ocr").strip().lower() or "ocr"
    size = max(1, int(batch_size))
    elapsed = max(0.001, _bounded_number(elapsed_seconds, 0.001))
    count = max(1, int(items))
    throughput = count / elapsed
    available = max(0.0, _bounded_number(available_memory_gb))
    with _LOCK:
        data = load()
        data.setdefault("schema", "novel_formatter.ocr_runtime_calibration.v1")
        data.setdefault("hardware", _hardware_fingerprint_payload())
        engines = data.setdefault("engines", {})
        if not isinstance(engines, dict):
            engines = {}
            data["engines"] = engines
        if len(engines) >= _MAX_ENGINES and name not in engines:
            oldest = sorted(engines.items(), key=lambda item: item[1].get("last_seen", 0.0))[:1]
            if oldest:
                engines.pop(oldest[0][0], None)
        state = engines.setdefault(name, {"samples": 0, "best_throughput": 0.0})
        state["samples"] = min(10000, int(state.get("samples", 0)) + 1)
        state["last_seen"] = __import__("time").time()
        state["last_batch"] = size
        state["last_available_memory_gb"] = round(available, 3)
        state["last_success"] = bool(success)
        current_best = _bounded_number(state.get("best_throughput", 0.0))
        if success:
            if size >= int(state.get("safe_batch_ceiling", 0) or 0):
                state["safe_batch_ceiling"] = size
            if not state.get("best_batch") or throughput > current_best * 1.03:
                state["best_batch"] = size
                state["best_throughput"] = throughput
            else:
                state["best_throughput"] = current_best
        elif current_best:
            state["best_throughput"] = current_best
        if resource_error:
            ceiling = max(1, size // 2)
            old = int(state.get("safe_batch_ceiling", 0) or 0)
            state["safe_batch_ceiling"] = ceiling if old == 0 else min(old, ceiling)
            state["resource_backoffs"] = min(1000, int(state.get("resource_backoffs", 0)) + 1)
        _path().parent.mkdir(parents=True, exist_ok=True)
        tmp = _path().with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(_path())


def safe_batch_ceiling(engine: str) -> int | None:
    try:
        state = load().get("engines", {}).get(str(engine or "ocr").strip().lower(), {})
        value = int(state.get("safe_batch_ceiling", 0) or 0)
        return value if value > 0 else None
    except Exception:
        return None


def best_batch(engine: str) -> int | None:
    try:
        state = load().get("engines", {}).get(str(engine or "ocr").strip().lower(), {})
        value = int(state.get("best_batch", 0) or 0)
        return value if value > 0 else None
    except Exception:
        return None


def clear() -> None:
    try:
        _path().unlink(missing_ok=True)
    except Exception:
        pass


__all__ = ["hardware_fingerprint", "load", "record", "safe_batch_ceiling", "best_batch", "clear"]
