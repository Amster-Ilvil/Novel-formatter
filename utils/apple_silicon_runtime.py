#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apple Silicon/M6-aware runtime tuning without changing OCR semantics.

The application deliberately does not hard-code a single accelerator for every
engine.  Apple Vision/Core ML can use CPU/GPU/Neural Engine as selected by the
OS, PyTorch OCR workers can use MPS, and MLX-VLM uses Apple's GPU/unified-memory
path.  This module only supplies conservative, hardware-aware defaults and
process environment hints; every engine keeps its existing fallback behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import platform
import subprocess
import sys
from pathlib import Path

from .hardware_runtime import (
    current_available_memory_gb,
    memory_bytes,
    memory_budget_gb,
    detect_system_hardware,
    recommended_cpu_threads as _generic_cpu_threads,
    recommended_encode_workers as _generic_encode_workers,
    recommended_hayai_batch as _generic_hayai_batch,
    recommended_hayai_segment_chars as _generic_hayai_segment_chars,
    recommended_manga_batch as _generic_manga_batch,
    recommended_paddle_batch as _generic_paddle_batch,
    recommended_prepare_workers as _generic_prepare_workers,
)


@dataclass(frozen=True, slots=True)
class AppleSiliconProfile:
    available: bool
    chip: str
    family: str
    is_m6: bool
    memory_gb: float
    cpu_cores: int
    gpu_cores: int
    neural_engine_cores: int
    memory_bandwidth_gbps: int

    @property
    def label(self) -> str:
        if not self.available:
            return "非 Apple Silicon"
        suffix = " · M6 性能档" if self.is_m6 else " · Apple Silicon"
        return f"{self.chip or 'Apple Silicon'}{suffix}"

    def as_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "chip": self.chip,
            "family": self.family,
            "is_m6": self.is_m6,
            "memory_gb": self.memory_gb,
            "cpu_cores": self.cpu_cores,
            "gpu_cores": self.gpu_cores,
            "neural_engine_cores": self.neural_engine_cores,
            "memory_bandwidth_gbps": self.memory_bandwidth_gbps,
        }


def _run_text(command: list[str], timeout: float = 5.0) -> str:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return (result.stdout or "").strip()


def _sysctl(name: str) -> str:
    if sys.platform != "darwin":
        return ""
    return _run_text(["/usr/sbin/sysctl", "-n", name], timeout=4.0)


def _memory_gb() -> float:
    raw = _sysctl("hw.memsize")
    try:
        return round(int(raw) / (1024 ** 3), 1)
    except (TypeError, ValueError):
        total_bytes, _ = memory_bytes()
        return round(total_bytes / (1024 ** 3), 1) if total_bytes else 0.0


def _cpu_name() -> str:
    value = _sysctl("machdep.cpu.brand_string")
    if value:
        return value
    profiler = "/usr/sbin/system_profiler"
    raw = _run_text([profiler, "SPHardwareDataType", "-json"], timeout=10.0)
    if raw:
        try:
            payload = __import__("json").loads(raw)
            rows = payload.get("SPHardwareDataType", []) if isinstance(payload, dict) else []
            for row in rows:
                if isinstance(row, dict):
                    chip = str(row.get("chip_type") or row.get("cpu_type") or "").strip()
                    if chip:
                        return chip
        except Exception:
            pass
    return platform.processor() or "Apple Silicon"


def detect_apple_silicon() -> AppleSiliconProfile:
    override = os.environ.get("NOVEL_FORMATTER_TEST_APPLE_CHIP", "").strip()
    chip = override or (_cpu_name() if sys.platform == "darwin" else "")
    machine = platform.machine().strip().lower()
    available = sys.platform == "darwin" and machine in {"arm64", "aarch64"}
    normalized = " ".join(chip.lower().split())
    family = ""
    for candidate in ("m6", "m5", "m4", "m3", "m2", "m1"):
        if candidate in normalized:
            family = candidate.upper()
            break
    is_m6 = bool(available and family == "M6")
    is_base_m6 = bool(is_m6 and not any(v in normalized for v in ("pro", "max", "ultra")))
    if is_base_m6:
        # Apple M6 as documented for the current M6 Mac mini: 12 CPU cores,
        # 12 GPU cores and a dual 16-core Neural Engine.  Memory bandwidth is
        # configuration-dependent: the 16 GB base configuration is 153 GB/s,
        # while higher-memory configurations can expose 170 GB/s.  Keep this
        # descriptor accurate instead of advertising the maximum on every M6.
        memory_gb = _memory_gb()
        bandwidth = 170 if memory_gb >= 23.0 else 153
        return AppleSiliconProfile(
            available=True,
            chip=chip or "Apple M6",
            family="M6",
            is_m6=True,
            memory_gb=memory_gb,
            cpu_cores=12,
            gpu_cores=12,
            neural_engine_cores=32,
            memory_bandwidth_gbps=bandwidth,
        )
    return AppleSiliconProfile(
        available=available,
        chip=chip,
        family=family,
        is_m6=is_m6,
        memory_gb=_memory_gb() if available else 0.0,
        cpu_cores=int(os.cpu_count() or 0) if available else 0,
        gpu_cores=0,
        neural_engine_cores=0,
        memory_bandwidth_gbps=0,
    )


_PROFILE: AppleSiliconProfile | None = None
_PROFILE_KEY: tuple[str, str, str] | None = None


def _profile_cache_key() -> tuple[str, str, str]:
    """Return the stable hardware identity that owns the cached profile.

    Normal application sessions never change CPU family, but tests, diagnostic
    tools and child-process launchers may override the detected Apple chip.  A
    process-global cache that ignores those inputs can incorrectly keep a stale
    non-Apple profile and silently skip MPS safety settings.
    """
    return (
        str(sys.platform or ""),
        str(platform.machine() or "").strip().lower(),
        os.environ.get("NOVEL_FORMATTER_TEST_APPLE_CHIP", "").strip(),
    )


def profile(*, refresh: bool = False) -> AppleSiliconProfile:
    global _PROFILE, _PROFILE_KEY
    key = _profile_cache_key()
    if _PROFILE is None or refresh or _PROFILE_KEY != key:
        _PROFILE = detect_apple_silicon()
        _PROFILE_KEY = key
    return _PROFILE


def is_m6() -> bool:
    return profile().is_m6


def recommended_hayai_batch() -> int:
    return _generic_hayai_batch()


def recommended_manga_batch() -> int:
    return _generic_manga_batch()


def recommended_hayai_segment_chars() -> int:
    return _generic_hayai_segment_chars()


def recommended_paddle_batch(pipeline: str = "ocr") -> int:
    return _generic_paddle_batch(pipeline)


def recommended_cpu_threads() -> int:
    return _generic_cpu_threads()


def recommended_prepare_workers() -> int:
    return _generic_prepare_workers()


def recommended_encode_workers() -> int:
    return _generic_encode_workers()



def probe_torch_mps(torch_module) -> tuple[bool, str]:
    """Verify MPS with a real tensor operation, not only capability flags."""
    try:
        backends = getattr(torch_module, "backends", None)
        mps_backend = getattr(backends, "mps", None) if backends is not None else None
        if mps_backend is None:
            return False, "当前 PyTorch 没有 MPS backend"
        try:
            built = bool(mps_backend.is_built())
        except Exception as exc:
            return False, f"MPS build probe 失败：{exc}"
        if not built:
            return False, "当前 PyTorch wheel 未编译 MPS"
        try:
            available = bool(mps_backend.is_available())
        except Exception as exc:
            return False, f"MPS availability probe 失败：{exc}"
        if not available:
            return False, "MPS 已编译但 is_available=False"
        if not hasattr(torch_module, "ones"):
            # Lightweight test doubles/wrappers may expose only backend flags.
            # Real PyTorch always has ``ones`` and therefore takes the tensor path.
            return True, "MPS capability flags PASS（wrapper 无 tensor API）"
        try:
            probe = torch_module.ones((8, 8), dtype=getattr(torch_module, "float32", None), device="mps")
            probe = (probe * 1.125).sum()
            mps = getattr(torch_module, "mps", None)
            if mps is not None and hasattr(mps, "synchronize"):
                mps.synchronize()
            _ = float(probe.detach().cpu().item())
            del probe
        except Exception as exc:
            return False, f"MPS tensor 实测失败：{exc}"
        return True, "MPS tensor probe PASS"
    except Exception as exc:
        return False, f"MPS probe 异常：{exc}"

def configure_torch_memory_safety(torch_module) -> float | None:
    """Set a live MPS per-process allocation fraction when available.

    The fraction is derived from the current free unified-memory headroom and
    Metal/PyTorch's recommended working-set size; it is intentionally not based
    on a hard-coded 8/16/24/32 GB machine tier.
    """
    try:
        mps = getattr(torch_module, "mps", None)
        backends = getattr(torch_module, "backends", None)
        if mps is None or not getattr(backends, "mps", None) or not mps.is_available():
            return None
        recommended = int(mps.recommended_max_memory()) if hasattr(mps, "recommended_max_memory") else 0
        available_gb = current_available_memory_gb()
        if recommended <= 0:
            return None
        # Keep a healthy reserve for macOS/GUI/NDL and transient image buffers.
        # Under pressure, shrink the allowed MPS working set rather than letting
        # the allocator consume the whole unified-memory pool.
        reserve_fraction = 0.90
        if available_gb and available_gb < 3.0:
            reserve_fraction = 0.58
        elif available_gb and available_gb < 5.0:
            reserve_fraction = 0.70
        elif available_gb and available_gb < 8.0:
            reserve_fraction = 0.82
        available_bytes = available_gb * (1024 ** 3) if available_gb else recommended * 0.90
        target = min(recommended * reserve_fraction, available_bytes * 0.82)
        fraction = max(0.55, min(1.0, target / recommended))
        setter = getattr(mps, "set_per_process_memory_fraction", None)
        if callable(setter):
            setter(float(fraction))
        return float(fraction)
    except Exception:
        # Memory safety is an optimization hint. Never prevent OCR startup.
        return None


def torch_worker_env(env: dict[str, str] | None = None) -> dict[str, str]:
    out = dict(env or os.environ)
    if profile().available:
        # Keep MPS allocator behavior bounded on unified-memory Macs. PyTorch's
        # documented 1.0 high watermark corresponds to Metal's recommended max
        # working-set size; do not disable the limit on a user's machine.
        out.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0")
        out.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.82")
        out.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        out.setdefault("TOKENIZERS_PARALLELISM", "false")
        out.setdefault("PYTHONUNBUFFERED", "1")
    return out


def mlx_worker_env(env: dict[str, str] | None = None) -> dict[str, str]:
    out = dict(env or os.environ)
    if profile().available:
        out.setdefault("TOKENIZERS_PARALLELISM", "false")
        out.setdefault("PYTHONUNBUFFERED", "1")
    return out


def diagnostic_lines() -> list[str]:
    p = profile(refresh=True)
    system = detect_system_hardware(refresh=True)
    available = current_available_memory_gb()
    lines = [
        f"运行平台：{system.platform} / {system.arch}",
        f"CPU 线程：{system.cpu_threads}",
        f"内存：{system.total_memory_gb:.1f} GB 总计 / {available:.1f} GB 当前可用" if system.total_memory_gb else "内存：未知",
        f"GPU/加速后端：{system.gpu_backend or 'cpu'}{(' · ' + system.gpu_name) if system.gpu_name else ''}",
        f"OCR 动态内存预算：{__import__('utils.hardware_runtime', fromlist=['memory_budget_gb']).memory_budget_gb():.2f} GB",
    ]
    if p.available:
        lines.extend([
            f"Apple：{p.label}",
            f"MPS 可用：{'是' if system.gpu_backend == 'mps' else '否'}；MLX 可用：{'是' if system.mlx_available else '否'}",
        ])
    lines.extend([
        f"Hayai 推荐批量：{recommended_hayai_batch()}",
        f"48px 推荐批量：{recommended_manga_batch()}",
        f"Hayai 竖排分段：{recommended_hayai_segment_chars()} 字",
        f"CPU 线程：{recommended_cpu_threads()}；预处理：{recommended_prepare_workers()}；编码：{recommended_encode_workers()}",
    ])
    return lines


__all__ = [
    "AppleSiliconProfile", "detect_apple_silicon", "diagnostic_lines", "is_m6",
    "mlx_worker_env", "profile", "recommended_cpu_threads", "recommended_encode_workers",
    "recommended_hayai_batch", "recommended_hayai_segment_chars", "recommended_manga_batch", "recommended_paddle_batch", "recommended_prepare_workers",
    "torch_worker_env", "probe_torch_mps",
]
