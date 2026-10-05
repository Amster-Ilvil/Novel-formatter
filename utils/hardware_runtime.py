#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-platform hardware/resource detection for OCR runtime tuning.

The detector deliberately separates *hardware facts* from *policy*.  It reads
current system memory pressure on every recommendation call and caches only
stable-ish hardware facts.  Apple Silicon gets an MPS/MLX-aware path, while
Windows/Linux use the same API with CUDA/DirectML/CPU fallbacks where present.

No OCR semantics are changed here.  The module only controls safe batch sizes,
thread counts, and bounded concurrency.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import ctypes
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SystemHardwareProfile:
    platform: str
    arch: str
    total_memory_gb: float
    cpu_threads: int
    apple_silicon: bool
    apple_chip: str
    apple_family: str
    is_m6: bool
    gpu_name: str
    gpu_backend: str
    gpu_memory_gb: float
    gpu_free_memory_gb: float
    directml_available: bool
    cuda_available: bool
    mlx_available: bool
    onnx_providers: tuple[str, ...]

    @property
    def unified_memory(self) -> bool:
        return self.apple_silicon

    @property
    def accelerator_available(self) -> bool:
        return self.gpu_backend not in {"", "cpu"} or self.mlx_available

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["onnx_providers"] = list(self.onnx_providers)
        payload["unified_memory"] = self.unified_memory
        payload["accelerator_available"] = self.accelerator_available
        return payload


_STABLE: SystemHardwareProfile | None = None
_STABLE_TS = 0.0


def _run(command: list[str], timeout: float = 4.0) -> str:
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return ""
    if proc.returncode != 0:
        return ""
    return (proc.stdout or "").strip()


def _linux_meminfo() -> tuple[float, float]:
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace")
    except Exception:
        return 0.0, 0.0
    values: dict[str, float] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].replace('.', '', 1).isdigit():
            values[parts[0].rstrip(':')] = float(parts[1]) * 1024.0
    total = values.get("MemTotal", 0.0)
    available = values.get("MemAvailable", values.get("MemFree", 0.0))
    return total, available


def _windows_memory() -> tuple[float, float]:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sAvailExtendedVirtual", ctypes.c_ulonglong),
        ]
    try:
        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        if not ok:
            return 0.0, 0.0
        return float(status.ullTotalPhys), float(status.ullAvailPhys)
    except Exception:
        return 0.0, 0.0


def memory_bytes() -> tuple[float, float]:
    """Return ``(total_bytes, available_bytes)`` using psutil/OS fallbacks."""
    # Test-only injection keeps CI deterministic without changing production
    # detection. It is never set by the application itself.
    override = os.environ.get("NOVEL_FORMATTER_TEST_TOTAL_MEMORY_GB", "").strip()
    if override:
        try:
            total = float(override) * (1024 ** 3)
            avail_override = os.environ.get("NOVEL_FORMATTER_TEST_AVAILABLE_MEMORY_GB", "").strip()
            available = float(avail_override) * (1024 ** 3) if avail_override else total * 0.60
            return total, min(total, max(0.0, available))
        except ValueError:
            pass
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        return float(vm.total), float(vm.available)
    except Exception:
        pass
    if os.name == "nt":
        return _windows_memory()
    if sys.platform == "darwin":
        raw = _run(["/usr/sbin/sysctl", "-n", "hw.memsize"])
        try:
            total = float(int(raw))
        except Exception:
            total = 0.0
        # vm_stat is intentionally best-effort.  If unavailable, use total only.
        vm = _run(["/usr/bin/vm_stat"])
        if total and vm:
            page_size = 4096.0
            for line in vm.splitlines()[:3]:
                if "page size of" in line.lower():
                    digits = "".join(ch for ch in line.split("page size of", 1)[1] if ch.isdigit())
                    if digits:
                        try:
                            page_size = float(int(digits))
                        except ValueError:
                            pass
                    break
            free_pages = 0.0
            for key in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
                for line in vm.splitlines():
                    if line.startswith(key + ":"):
                        try:
                            value = line.split(":", 1)[1].strip().rstrip(".").replace(",", "")
                            free_pages += float(value)
                        except Exception:
                            pass
                        break
            if free_pages > 0:
                return total, free_pages * page_size
        if total <= 0:
            try:
                pages = int(os.sysconf("SC_PHYS_PAGES"))
                page_size = int(os.sysconf("SC_PAGE_SIZE"))
                total = float(pages * page_size)
            except (AttributeError, OSError, TypeError, ValueError):
                pass
        return total, 0.0
    return _linux_meminfo()


def _windows_gpu_info() -> tuple[str, float]:
    if os.name != "nt":
        return "", 0.0
    ps = shutil.which("powershell") or shutil.which("pwsh")
    if not ps:
        return "", 0.0
    script = "Get-CimInstance Win32_VideoController | Select-Object -First 1 Name,AdapterRAM | ConvertTo-Csv -NoTypeInformation"
    raw = _run([ps, "-NoProfile", "-Command", script], timeout=6.0)
    lines = raw.splitlines()
    if len(lines) < 2:
        return "", 0.0
    try:
        import csv
        row = next(csv.reader([lines[1]]))
        name = str(row[0]).strip() if row else ""
        ram = float(row[1]) / (1024 ** 3) if len(row) > 1 and row[1].strip().isdigit() else 0.0
        return name, ram
    except Exception:
        return "", 0.0


def _nvidia_info() -> tuple[str, float, float]:
    tool = shutil.which("nvidia-smi")
    if not tool:
        return "", 0.0, 0.0
    raw = _run([
        tool,
        "--query-gpu=name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ], timeout=5.0)
    if not raw:
        return "", 0.0, 0.0
    first = raw.splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    try:
        total = float(parts[1]) / 1024.0 if len(parts) > 1 else 0.0
        free = float(parts[2]) / 1024.0 if len(parts) > 2 else 0.0
    except Exception:
        total = free = 0.0
    return (parts[0] if parts else "NVIDIA GPU"), total, free


def _apple_chip() -> tuple[bool, str, str, bool]:
    if sys.platform != "darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
        return False, "", "", False
    chip = os.environ.get("NOVEL_FORMATTER_TEST_APPLE_CHIP", "").strip() or _run(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], timeout=3.0)
    if not chip:
        raw = _run(["/usr/sbin/system_profiler", "SPHardwareDataType", "-json"], timeout=20.0)
        try:
            rows = __import__("json").loads(raw).get("SPHardwareDataType", [])
            chip = next(
                (str(row.get("chip_type") or row.get("cpu_type") or "").strip()
                 for row in rows if isinstance(row, dict)),
                "",
            )
        except (ValueError, AttributeError, TypeError):
            chip = ""
    if not chip:
        processor = platform.processor().strip()
        chip = processor if processor.lower() not in {"arm", "arm64", "aarch64"} else "Apple Silicon"
    normalized = " ".join(chip.lower().split())
    family = next((name.upper() for name in ("m6", "m5", "m4", "m3", "m2", "m1") if name in normalized), "")
    # Generation-level tuning applies to the complete M6 family. Exact core
    # counts are discovered separately; do not exclude Pro/Max/Ultra variants.
    is_m6 = family == "M6"
    return True, chip, family, is_m6


def _probe_onnx_providers() -> tuple[str, ...]:
    try:
        import onnxruntime as ort  # type: ignore
        return tuple(str(v) for v in ort.get_available_providers())
    except Exception:
        return ()


def _probe_directml() -> bool:
    try:
        import torch_directml  # type: ignore
        return bool(torch_directml)
    except Exception:
        return False


def _probe_mps() -> bool:
    if sys.platform != "darwin":
        return False
    try:
        import torch  # type: ignore
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def _probe_mlx() -> bool:
    if sys.platform != "darwin":
        return False
    try:
        import mlx.core as mx  # type: ignore
        return bool(mx)
    except Exception:
        return False


def detect_system_hardware(*, refresh: bool = False) -> SystemHardwareProfile:
    global _STABLE, _STABLE_TS
    now = time.monotonic()
    if _STABLE is not None and not refresh and now - _STABLE_TS < 30.0:
        return _STABLE

    total_bytes, _ = memory_bytes()
    total_gb = round(total_bytes / (1024 ** 3), 1) if total_bytes else 0.0
    apple, chip, family, m6 = _apple_chip()
    nvidia_name, nvidia_vram, nvidia_free_vram = _nvidia_info()
    windows_gpu_name, windows_gpu_vram = _windows_gpu_info()
    onnx = _probe_onnx_providers()
    cuda = "CUDAExecutionProvider" in onnx or bool(nvidia_name)
    directml = os.name == "nt" and ("DmlExecutionProvider" in onnx or _probe_directml())
    xpu = "OpenVINOExecutionProvider" in onnx or "IntelExecutionProvider" in onnx
    mps = apple and _probe_mps()
    mlx = apple and _probe_mlx()

    backend = "cpu"
    gpu_name = ""
    gpu_memory = 0.0
    gpu_free_memory = 0.0
    if mps or apple:
        backend = "mps" if mps else "apple_gpu"
        gpu_name = chip or "Apple GPU"
    elif cuda:
        backend = "cuda"
        gpu_name = nvidia_name or "NVIDIA GPU"
        gpu_memory = nvidia_vram
        gpu_free_memory = nvidia_free_vram
    elif directml:
        backend = "directml"
        gpu_name = nvidia_name or windows_gpu_name or ("DirectML GPU" if os.name == "nt" else "")
        gpu_memory = nvidia_vram or windows_gpu_vram
        gpu_free_memory = nvidia_free_vram if nvidia_vram else 0.0
    elif "MIGraphXExecutionProvider" in onnx:
        backend = "migraphx"
        gpu_name = "AMD GPU"
    elif xpu:
        backend = "openvino"
        gpu_name = windows_gpu_name or "Intel/OpenVINO accelerator"
        gpu_memory = windows_gpu_vram
        gpu_free_memory = windows_gpu_vram

    _STABLE = SystemHardwareProfile(
        platform=sys.platform,
        arch=platform.machine().lower(),
        total_memory_gb=total_gb,
        cpu_threads=max(1, int(os.cpu_count() or 1)),
        apple_silicon=apple,
        apple_chip=chip,
        apple_family=family,
        is_m6=m6,
        gpu_name=gpu_name,
        gpu_backend=backend,
        gpu_memory_gb=round(gpu_memory, 1),
        gpu_free_memory_gb=round(gpu_free_memory, 1),
        directml_available=directml,
        cuda_available=cuda,
        mlx_available=mlx,
        onnx_providers=onnx,
    )
    _STABLE_TS = now
    return _STABLE


def current_available_memory_gb() -> float:
    _, available = memory_bytes()
    return round(available / (1024 ** 3), 1) if available else 0.0


def memory_budget_gb() -> float:
    """Compute a live budget for OCR allocations rather than using RAM tiers.

    On unified-memory Macs both CPU and GPU share this pool, so the budget is
    intentionally tighter. On discrete GPUs, the system RAM budget and VRAM
    budget are treated separately and the lower live headroom wins.
    """
    p = detect_system_hardware()
    available = current_available_memory_gb()
    if p.total_memory_gb <= 0:
        return 2.0
    if available <= 0:
        available = p.total_memory_gb * 0.55
    if p.unified_memory:
        # Keep enough room for macOS/GUI and page buffers; GPU allocations come
        # from the same pool as Python/NDL/IO.
        return max(1.5, min(p.total_memory_gb * 0.50, available * 0.78))
    # Dedicated GPU: CPU RAM and VRAM are separate pools. Use whichever live
    # headroom is tighter so CUDA/DirectML batches do not silently overfill VRAM.
    system_budget = max(1.5, min(p.total_memory_gb * 0.60, available * 0.82))
    if p.gpu_free_memory_gb > 0 and p.gpu_backend in {"cuda", "directml", "migraphx", "openvino"}:
        gpu_budget = max(0.75, p.gpu_free_memory_gb * 0.72)
        return max(1.5, min(system_budget, gpu_budget))
    return system_budget


def recommended_cpu_threads() -> int:
    p = detect_system_hardware()
    reserve = 4 if p.apple_silicon else 2
    if current_available_memory_gb() < 3.0:
        reserve += 1
    return max(2, min(12, p.cpu_threads - reserve))


def recommended_prepare_workers() -> int:
    p = detect_system_hardware()
    available = current_available_memory_gb()
    cap = 8 if p.apple_silicon else 6
    workers = min(cap, max(2, p.cpu_threads - (4 if p.apple_silicon else 2)))
    if available and available < 3.0:
        workers = min(workers, 2)
    elif available and available < 5.0:
        workers = min(workers, 4)
    return workers


def recommended_encode_workers() -> int:
    p = detect_system_hardware()
    available = current_available_memory_gb()
    cap = 4 if p.apple_silicon else 3
    workers = min(cap, max(2, p.cpu_threads // 4))
    if available and available < 3.0:
        workers = 1
    return workers


def _accelerator_batch(memory_cost_per_item: float, *, cpu_default: int = 4, cap: int = 12) -> int:
    p = detect_system_hardware()
    budget = memory_budget_gb()
    if p.gpu_backend == "cpu":
        base = max(cpu_default, min(8, recommended_cpu_threads() // 2))
        return max(1, min(cap, base))
    # Runtime-measured budget, not total-RAM tiers.
    value = int(budget / max(0.25, memory_cost_per_item))
    value = max(1, min(cap, value))
    if current_available_memory_gb() and current_available_memory_gb() < 2.5:
        value = min(value, 2)
    return value


def _benchmark_override(engine: str, value: int, maximum: int) -> int:
    if os.environ.get("NOVEL_FORMATTER_OCR_BENCHMARK", "0").strip() != "1":
        return int(value)
    raw = os.environ.get("NOVEL_FORMATTER_OCR_BENCHMARK_BATCH", "").strip()
    try:
        requested = int(raw)
    except ValueError:
        return int(value)
    if requested <= 0:
        return int(value)
    # Benchmark mode may intentionally probe above the normal live-memory
    # recommendation, but remains bounded by the engine's hard ceiling. Worker
    # level OOM backoff is the safety net for that probe.
    return max(1, min(int(maximum), requested))


def _apply_calibrated_ceiling(engine: str, value: int) -> int:
    try:
        from utils.ocr_runtime_calibration import safe_batch_ceiling
        ceiling = safe_batch_ceiling(engine)
    except Exception:
        ceiling = None
    if ceiling:
        return max(1, min(int(value), int(ceiling)))
    return max(1, int(value))


def recommended_hayai_batch() -> int:
    p = detect_system_hardware()
    # Hayai's working set is much larger than 48px strips; keep the model as one
    # instance and adapt request batch to the *currently available* memory.
    if p.gpu_backend in {"mps", "cuda", "directml", "migraphx"}:
        value = _accelerator_batch(0.72, cpu_default=2, cap=12)
    else:
        value = max(2, min(6, recommended_cpu_threads() // 2))
    value = _benchmark_override("hayai", value, 12)
    return _apply_calibrated_ceiling("hayai", value)


def recommended_manga_batch() -> int:
    p = detect_system_hardware()
    if p.gpu_backend in {"mps", "cuda", "directml", "migraphx"}:
        value = _accelerator_batch(0.22, cpu_default=2, cap=16)
    else:
        value = max(2, min(8, recommended_cpu_threads() // 2))
    value = _benchmark_override("48px", value, 16)
    return _apply_calibrated_ceiling("48px", value)


def recommended_hayai_segment_chars() -> int:
    p = detect_system_hardware()
    budget = memory_budget_gb()
    if p.apple_silicon:
        # Keep the existing 40-char hard cap but adapt the normal segment size.
        return max(20, min(32, int(round(22 + budget * 1.2))))
    return max(20, min(30, int(round(20 + budget * 1.0))))


def recommended_paddle_batch(pipeline: str = "ocr") -> int:
    pipe = str(pipeline or "ocr").strip().lower()
    cost = 0.55 if pipe != "vl" else 1.10
    return _accelerator_batch(cost, cpu_default=1 if pipe == "vl" else 2, cap=8 if pipe != "vl" else 4)


def accelerator_recommended_working_set_gb() -> float:
    """Return the accelerator working-set limit when the active runtime exposes it."""
    p = detect_system_hardware()
    if p.gpu_backend != "mps":
        return 0.0
    try:
        import torch  # type: ignore
        value = int(torch.mps.recommended_max_memory())
        return round(value / (1024 ** 3), 2)
    except Exception:
        return 0.0


def diagnostic_snapshot() -> dict[str, Any]:
    p = detect_system_hardware(refresh=True)
    payload = p.as_dict()
    payload["available_memory_gb"] = current_available_memory_gb()
    payload["ocr_memory_budget_gb"] = round(memory_budget_gb(), 2)
    try:
        from utils.ocr_runtime_calibration import hardware_fingerprint, safe_batch_ceiling, best_batch
        payload["calibration"] = {
            "hardware_fingerprint": hardware_fingerprint(),
            "hayai_safe_batch_ceiling": safe_batch_ceiling("hayai"),
            "hayai_best_batch": best_batch("hayai"),
            "manga_48px_safe_batch_ceiling": safe_batch_ceiling("48px"),
            "manga_48px_best_batch": best_batch("48px"),
        }
    except Exception:
        payload["calibration"] = {}
    payload["accelerator_recommended_working_set_gb"] = accelerator_recommended_working_set_gb()
    payload["recommended"] = {
        "cpu_threads": recommended_cpu_threads(),
        "prepare_workers": recommended_prepare_workers(),
        "encode_workers": recommended_encode_workers(),
        "hayai_batch": recommended_hayai_batch(),
        "hayai_segment_chars": recommended_hayai_segment_chars(),
        "manga_48px_batch": recommended_manga_batch(),
        "paddle_ocr_batch": recommended_paddle_batch("ocr"),
        "paddle_vl_batch": recommended_paddle_batch("vl"),
    }
    return payload


__all__ = [
    "SystemHardwareProfile",
    "current_available_memory_gb",
    "detect_system_hardware",
    "diagnostic_snapshot",
    "accelerator_recommended_working_set_gb",
    "memory_budget_gb",
    "recommended_cpu_threads",
    "recommended_encode_workers",
    "recommended_hayai_batch",
    "recommended_hayai_segment_chars",
    "recommended_manga_batch",
    "recommended_paddle_batch",
    "recommended_prepare_workers",
]
