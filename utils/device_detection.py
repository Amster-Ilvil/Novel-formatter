# -*- coding: utf-8 -*-
"""User-initiated, cross-platform device and accelerator diagnostics.

Importing this module is side-effect free. Hardware commands and runtime probes
run only when :func:`detect_devices` is explicitly called from the settings UI.
No detected backend is applied automatically to OCR settings.
"""
from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
ProgressCallback = Callable[[str, int, int, str], None]


@dataclass(frozen=True, slots=True)
class GPUDevice:
    name: str
    vendor: str = ""
    memory_mb: int = 0
    driver: str = ""
    source: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RuntimeBackend:
    runtime: str
    python: str
    torch_version: str = ""
    cuda_available: bool = False
    cuda_devices: tuple[str, ...] = ()
    mps_available: bool = False
    onnxruntime_version: str = ""
    onnx_providers: tuple[str, ...] = ()
    detail: str = ""

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["cuda_devices"] = list(self.cuda_devices)
        payload["onnx_providers"] = list(self.onnx_providers)
        return payload


@dataclass(frozen=True, slots=True)
class DeviceReport:
    platform_name: str
    platform_release: str
    architecture: str
    cpu: str
    logical_cores: int
    memory_gb: float
    gpus: tuple[GPUDevice, ...]
    runtimes: tuple[RuntimeBackend, ...]
    acceleration_summary: str
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "platform_name": self.platform_name,
            "platform_release": self.platform_release,
            "architecture": self.architecture,
            "cpu": self.cpu,
            "logical_cores": self.logical_cores,
            "memory_gb": self.memory_gb,
            "gpus": [item.to_dict() for item in self.gpus],
            "runtimes": [item.to_dict() for item in self.runtimes],
            "acceleration_summary": self.acceleration_summary,
            "notes": list(self.notes),
        }


def _emit(callback: ProgressCallback | None, stage: str, current: int, total: int, detail: str) -> None:
    if callback is None:
        return
    try:
        callback(str(stage), max(0, int(current)), max(0, int(total)), str(detail))
    except Exception:
        pass


def _hidden_process_kwargs() -> dict:
    if os.name != "nt":
        return {}
    return {"creationflags": int(getattr(subprocess, "CREATE_NO_WINDOW", 0))}


def _run_text(command: list[str], *, timeout: float = 15.0) -> str:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            **_hidden_process_kwargs(),
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return (result.stdout or "").strip()


def _powershell_executable() -> str:
    for name in ("powershell.exe", "pwsh.exe", "powershell", "pwsh"):
        value = shutil.which(name)
        if value:
            return value
    return ""


def _windows_video_controllers() -> list[GPUDevice]:
    powershell = _powershell_executable()
    if not powershell:
        return []
    script = (
        "$ErrorActionPreference='Stop';"
        "$items=Get-CimInstance -ClassName Win32_VideoController | "
        "Select-Object Name,AdapterRAM,DriverVersion,PNPDeviceID;"
        "$items | ConvertTo-Json -Compress"
    )
    raw = _run_text(
        [powershell, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script],
        timeout=20.0,
    )
    if not raw:
        return []
    try:
        payload = json.loads(raw.lstrip("\ufeff"))
    except json.JSONDecodeError:
        return []
    items = payload if isinstance(payload, list) else [payload]
    result: list[GPUDevice] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("Name") or "").strip()
        if not name:
            continue
        pnp = str(item.get("PNPDeviceID") or "").upper()
        lowered = name.lower()
        if "nvidia" in lowered or "VEN_10DE" in pnp:
            vendor = "NVIDIA"
        elif "amd" in lowered or "radeon" in lowered or "VEN_1002" in pnp:
            vendor = "AMD"
        elif "intel" in lowered or "VEN_8086" in pnp:
            vendor = "Intel"
        else:
            vendor = ""
        try:
            memory_mb = max(0, int(item.get("AdapterRAM") or 0) // (1024 * 1024))
        except (TypeError, ValueError):
            memory_mb = 0
        result.append(
            GPUDevice(
                name=name,
                vendor=vendor,
                memory_mb=memory_mb,
                driver=str(item.get("DriverVersion") or "").strip(),
                source="Windows CIM",
            )
        )
    return result


def _nvidia_smi_devices() -> list[GPUDevice]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        common = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"
        if common.is_file():
            executable = str(common)
    if not executable:
        return []
    raw = _run_text(
        [
            executable,
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        timeout=15.0,
    )
    result: list[GPUDevice] = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if not parts or not parts[0]:
            continue
        try:
            memory_mb = int(float(parts[1])) if len(parts) > 1 else 0
        except (TypeError, ValueError):
            memory_mb = 0
        result.append(
            GPUDevice(
                name=parts[0],
                vendor="NVIDIA",
                memory_mb=memory_mb,
                driver=parts[2] if len(parts) > 2 else "",
                source="nvidia-smi",
            )
        )
    return result


def _mac_video_controllers() -> list[GPUDevice]:
    profiler = shutil.which("system_profiler") or "/usr/sbin/system_profiler"
    if not Path(profiler).exists():
        return []
    raw = _run_text([profiler, "SPDisplaysDataType", "-json"], timeout=30.0)
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    result: list[GPUDevice] = []
    for item in payload.get("SPDisplaysDataType", []) if isinstance(payload, dict) else []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("sppci_model") or item.get("_name") or "").strip()
        if not name:
            continue
        vram_text = str(item.get("spdisplays_vram") or item.get("spdisplays_vram_shared") or "")
        memory_mb = 0
        digits = "".join(ch if ch.isdigit() or ch == "." else " " for ch in vram_text).split()
        if digits:
            try:
                value = float(digits[0])
                memory_mb = int(value * 1024) if "GB" in vram_text.upper() else int(value)
            except ValueError:
                memory_mb = 0
        vendor = "Apple" if "apple" in name.lower() else ""
        result.append(GPUDevice(name=name, vendor=vendor, memory_mb=memory_mb, source="system_profiler"))
    return result


def _linux_video_controllers() -> list[GPUDevice]:
    lspci = shutil.which("lspci")
    if not lspci:
        return []
    raw = _run_text([lspci], timeout=10.0)
    result: list[GPUDevice] = []
    for line in raw.splitlines():
        lowered = line.lower()
        if "vga compatible controller" not in lowered and "3d controller" not in lowered:
            continue
        name = line.split(": ", 1)[-1].strip()
        vendor = "NVIDIA" if "nvidia" in lowered else ("AMD" if "amd" in lowered or "radeon" in lowered else ("Intel" if "intel" in lowered else ""))
        result.append(GPUDevice(name=name, vendor=vendor, source="lspci"))
    return result


def _merge_gpus(primary: list[GPUDevice], precise: list[GPUDevice]) -> tuple[GPUDevice, ...]:
    merged: list[GPUDevice] = []
    for candidate in [*precise, *primary]:
        key = " ".join(candidate.name.lower().split())
        existing_index = next((i for i, item in enumerate(merged) if key in " ".join(item.name.lower().split()) or " ".join(item.name.lower().split()) in key), -1)
        if existing_index < 0:
            merged.append(candidate)
            continue
        existing = merged[existing_index]
        merged[existing_index] = GPUDevice(
            name=existing.name if len(existing.name) >= len(candidate.name) else candidate.name,
            vendor=existing.vendor or candidate.vendor,
            memory_mb=max(existing.memory_mb, candidate.memory_mb),
            driver=existing.driver or candidate.driver,
            source=" + ".join(dict.fromkeys(filter(None, [existing.source, candidate.source]))),
        )
    return tuple(merged)


def _memory_bytes() -> int:
    if os.name == "nt":
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
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        state = MEMORYSTATUSEX()
        state.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
                return int(state.ullTotalPhys)
        except Exception:
            return 0
    if sys.platform == "darwin":
        raw = _run_text(["/usr/sbin/sysctl", "-n", "hw.memsize"], timeout=5.0)
        try:
            return int(raw)
        except ValueError:
            return 0
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return pages * page_size
    except (AttributeError, OSError, TypeError, ValueError):
        return 0


def _cpu_name() -> str:
    if sys.platform == "darwin":
        value = _run_text(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], timeout=5.0)
        if value:
            return value
    if os.name == "nt":
        powershell = _powershell_executable()
        if powershell:
            value = _run_text(
                [
                    powershell,
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)",
                ],
                timeout=10.0,
            )
            if value:
                return value
    return platform.processor() or platform.machine() or "未知 CPU"


def _runtime_python_candidates() -> list[tuple[str, Path]]:
    candidates = [("主程序", Path(sys.executable))]
    for label, folder in (
        ("Manga OCR", ".venv-manga-ocr"),
        ("48px AR", ".venv-manga-48px"),
        ("YomiToku", ".venv-yomitoku"),
        ("NDLOCR-Lite", ".venv-ndlocr-lite"),
        ("PaddleOCR", ".venv-paddle"),
        ("PDF Craft", ".venv-pdf-craft"),
    ):
        root = ROOT / folder
        python = root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        candidates.append((label, python))
    result: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for label, python in candidates:
        try:
            key = str(python.resolve())
        except OSError:
            key = str(python)
        if key in seen or not python.is_file():
            continue
        seen.add(key)
        result.append((label, python))
    return result


_RUNTIME_PROBE = r'''
import json
out = {
    "torch_version": "", "cuda_available": False, "cuda_devices": [],
    "mps_available": False, "onnxruntime_version": "", "onnx_providers": [],
    "errors": []
}
try:
    import torch
    out["torch_version"] = str(getattr(torch, "__version__", ""))
    try:
        out["cuda_available"] = bool(torch.cuda.is_available())
        if out["cuda_available"]:
            out["cuda_devices"] = [str(torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
    except Exception as exc:
        out["errors"].append("CUDA: " + str(exc))
    try:
        out["mps_available"] = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    except Exception as exc:
        out["errors"].append("MPS: " + str(exc))
except Exception as exc:
    out["errors"].append("PyTorch: " + str(exc))
try:
    import onnxruntime as ort
    out["onnxruntime_version"] = str(getattr(ort, "__version__", ""))
    out["onnx_providers"] = list(ort.get_available_providers())
except Exception as exc:
    out["errors"].append("ONNX Runtime: " + str(exc))
print(json.dumps(out, ensure_ascii=False))
'''


def _probe_runtime(label: str, python: Path) -> RuntimeBackend:
    try:
        result = subprocess.run(
            [str(python), "-c", _RUNTIME_PROBE],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=35.0,
            **_hidden_process_kwargs(),
        )
    except subprocess.TimeoutExpired:
        return RuntimeBackend(runtime=label, python=str(python), detail="运行时探测超时")
    except Exception as exc:
        return RuntimeBackend(runtime=label, python=str(python), detail=f"无法启动：{exc}")
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or f"退出码 {result.returncode}").strip()
        return RuntimeBackend(runtime=label, python=str(python), detail=detail[-500:])
    lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    try:
        payload = json.loads(lines[-1]) if lines else {}
    except json.JSONDecodeError:
        return RuntimeBackend(runtime=label, python=str(python), detail="探测输出不是有效 JSON")
    errors = [str(item) for item in payload.get("errors", []) if str(item)]
    return RuntimeBackend(
        runtime=label,
        python=str(python),
        torch_version=str(payload.get("torch_version") or ""),
        cuda_available=bool(payload.get("cuda_available")),
        cuda_devices=tuple(str(item) for item in payload.get("cuda_devices", []) if str(item)),
        mps_available=bool(payload.get("mps_available")),
        onnxruntime_version=str(payload.get("onnxruntime_version") or ""),
        onnx_providers=tuple(str(item) for item in payload.get("onnx_providers", []) if str(item)),
        detail="；".join(errors),
    )


def _acceleration_summary(runtimes: tuple[RuntimeBackend, ...], gpus: tuple[GPUDevice, ...]) -> str:
    cuda = [item.runtime for item in runtimes if item.cuda_available]
    mps = [item.runtime for item in runtimes if item.mps_available]
    directml = [item.runtime for item in runtimes if "DmlExecutionProvider" in item.onnx_providers]
    ort_cuda = [item.runtime for item in runtimes if "CUDAExecutionProvider" in item.onnx_providers]
    parts: list[str] = []
    if cuda:
        parts.append("PyTorch CUDA 可用：" + "、".join(cuda))
    if ort_cuda:
        parts.append("ONNX CUDA 可用：" + "、".join(ort_cuda))
    if directml:
        parts.append("ONNX DirectML 可用：" + "、".join(directml))
    if mps:
        parts.append("Apple MPS 可用：" + "、".join(mps))
    if not parts:
        parts.append("当前已安装 OCR 运行时仅确认 CPU 可用")
        if gpus:
            parts.append("检测到 GPU，但对应 CUDA/DirectML/MPS 运行库可能尚未安装")
    return "；".join(parts)


def detect_devices(*, progress_callback: ProgressCallback | None = None) -> DeviceReport:
    """Inspect hardware and installed OCR runtimes after a manual UI action."""
    _emit(progress_callback, "system", 0, 3, "读取操作系统、CPU 与内存")
    if os.name == "nt":
        platform_name = "Windows"
        base_gpus = _windows_video_controllers()
    elif sys.platform == "darwin":
        platform_name = "macOS"
        base_gpus = _mac_video_controllers()
    else:
        platform_name = platform.system() or sys.platform
        base_gpus = _linux_video_controllers()
    gpus = _merge_gpus(base_gpus, _nvidia_smi_devices())
    _emit(progress_callback, "gpu", 1, 3, f"检测到 {len(gpus)} 个图形设备")

    candidates = _runtime_python_candidates()
    runtimes: list[RuntimeBackend] = []
    total = max(1, len(candidates))
    for index, (label, python) in enumerate(candidates, start=1):
        _emit(progress_callback, "runtime", index - 1, total, f"检测 {label} 加速后端")
        runtimes.append(_probe_runtime(label, python))
        _emit(progress_callback, "runtime", index, total, f"已检测 {label}")

    notes = [
        "检测结果只用于诊断，不会自动更改任何 OCR 引擎或设备选项。",
        "Windows 的独立 OCR 环境可能分别安装 CPU、CUDA 或 DirectML 运行库，因此同一台电脑上各模型结果可以不同。",
    ]
    if os.name == "nt" and not _powershell_executable():
        notes.append("未找到 PowerShell，Windows CIM 设备信息可能不完整。")
    memory = _memory_bytes()
    report = DeviceReport(
        platform_name=platform_name,
        platform_release=platform.release() or platform.version(),
        architecture=platform.machine() or "未知",
        cpu=_cpu_name(),
        logical_cores=int(os.cpu_count() or 0),
        memory_gb=round(memory / (1024 ** 3), 1) if memory else 0.0,
        gpus=gpus,
        runtimes=tuple(runtimes),
        acceleration_summary=_acceleration_summary(tuple(runtimes), gpus),
        notes=tuple(notes),
    )
    _emit(progress_callback, "done", 3, 3, "设备与 GPU 检测完成")
    return report


__all__ = [
    "DeviceReport",
    "GPUDevice",
    "RuntimeBackend",
    "detect_devices",
]
