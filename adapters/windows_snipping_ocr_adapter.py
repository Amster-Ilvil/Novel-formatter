#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Windows Snipping Tool OneOCR adapter.

The adapter reuses OneOCR assets from the Microsoft Snipping Tool package
already installed on the user's Windows machine.  No Microsoft binaries or
models are downloaded, copied into the project, or redistributed.

OneOCR is not a documented public Microsoft API.  Runtime discovery and the
worker are deliberately isolated so a future Snipping Tool ABI change fails
closed without affecting the other OCR engines.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from adapters.ocr_engine_common import iter_worker_jsonl, run_ocr_engine

PACKAGE_NAME = "Microsoft.ScreenSketch"
REQUIRED_FILES = ("oneocr.dll", "oneocr.onemodel", "onnxruntime.dll")
WORKER_SCRIPT = Path(__file__).with_name("windows_snipping_ocr_worker.py")


def _valid_runtime_dir(path: str | Path | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path).expanduser()
    if candidate.name.lower() != "snippingtool" and (candidate / "SnippingTool").is_dir():
        candidate = candidate / "SnippingTool"
    try:
        if candidate.is_dir() and all((candidate / name).is_file() for name in REQUIRED_FILES):
            return candidate.resolve()
    except OSError:
        return None
    return None


def _powershell_executable() -> str:
    root = os.environ.get("SystemRoot", r"C:\Windows")
    candidate = Path(root) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    return str(candidate) if candidate.is_file() else "powershell.exe"


def _query_screen_sketch_install_location(timeout: float = 5.0) -> Path | None:
    if sys.platform != "win32":
        return None
    script = (
        "$p = Get-AppxPackage -Name Microsoft.ScreenSketch | "
        "Sort-Object {[version]$_.Version} -Descending | Select-Object -First 1; "
        "if ($p) { [Console]::Out.Write($p.InstallLocation) }"
    )
    try:
        result = subprocess.run(
            [
                _powershell_executable(),
                "-NoLogo", "-NoProfile", "-NonInteractive",
                "-ExecutionPolicy", "Bypass",
                "-Command", script,
            ],
            capture_output=True,
            text=True,
            timeout=max(1.0, float(timeout)),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    value = (result.stdout or "").strip().strip('"')
    return Path(value) if value else None


def find_runtime_dir() -> Path | None:
    """Locate Snipping Tool OneOCR without downloading or copying anything."""
    override = os.environ.get("NOVEL_FORMATTER_SNIPPING_OCR_DIR", "").strip()
    if override:
        return _valid_runtime_dir(override)
    if sys.platform != "win32":
        return None
    install = _query_screen_sketch_install_location()
    return _valid_runtime_dir(install)


def availability() -> tuple[bool, str]:
    if sys.platform != "win32":
        return False, "Windows Snipping OCR 仅支持 Windows 10/11。"
    runtime = find_runtime_dir()
    if runtime is None:
        return False, (
            "未检测到 Windows“截图工具 / Snipping Tool”的 OneOCR 运行资源。"
            "请先通过 Microsoft Store 更新“截图工具”，然后重试；"
            "高级用户也可设置 NOVEL_FORMATTER_SNIPPING_OCR_DIR 指向其 SnippingTool 目录。"
            "Novel Formatter 不会自行下载或打包 Microsoft DLL/模型。"
        )
    return True, f"已检测到系统 Snipping Tool OneOCR：{runtime}"


def runtime_status() -> str:
    ready, detail = availability()
    return "系统 OneOCR 可用" if ready else detail


def _run_worker(
    image_paths: list[str],
    *,
    cancel_check=None,
    verbose: bool = True,
):
    del verbose
    ready, detail = availability()
    if not ready:
        raise RuntimeError(detail)
    runtime = find_runtime_dir()
    if runtime is None:
        raise RuntimeError(detail)
    cmd = [
        sys.executable,
        str(WORKER_SCRIPT),
        "--runtime-dir", str(runtime),
        *[str(path) for path in image_paths],
    ]
    yield from iter_worker_jsonl(
        cmd,
        cancel_check=cancel_check,
        engine_label="Windows Snipping OCR",
    )


def run(
    image_folder: str | None = None,
    page_overrides: dict[int, str] | None = None,
    verbose: bool = True,
    input_paths: list[str] | None = None,
    progress_callback=None,
    cancel_check=None,
    crop_top: float = 0.0,
    crop_bottom: float = 0.0,
    crop_rect: tuple[float, float, float, float] | None = None,
    temp_crop_dir: str | None = None,
    reuse_existing_crops: bool = False,
    filter_running_headers: bool = True,
    force_text_pages: bool = False,
    strict_column_audit: bool = False,
    ocr_mode: str = "ja_vertical",
    merge_horizontal_fragments: bool = True,
    performance_callback=None,
):
    """Run system OneOCR through Novel Formatter's normal document pipeline."""

    def worker(paths, cancel_check=None):
        yield from _run_worker(
            paths,
            cancel_check=cancel_check,
            verbose=verbose,
        )

    return run_ocr_engine(
        worker,
        source_engine="windows_snipping_ocr",
        image_folder=image_folder,
        page_overrides=page_overrides,
        verbose=verbose,
        input_paths=input_paths,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
        crop_top=crop_top,
        crop_bottom=crop_bottom,
        crop_rect=crop_rect,
        temp_crop_dir=temp_crop_dir,
        reuse_existing_crops=reuse_existing_crops,
        filter_running_headers=filter_running_headers,
        force_text_pages=force_text_pages,
        strict_column_audit=strict_column_audit,
        ocr_mode=ocr_mode,
        merge_horizontal_fragments=merge_horizontal_fragments,
        performance_callback=performance_callback,
        worker_timing_name="windows_snipping_oneocr",
    )


__all__ = [
    "PACKAGE_NAME", "REQUIRED_FILES", "WORKER_SCRIPT",
    "availability", "find_runtime_dir", "runtime_status", "run",
]


if __name__ == "__main__":
    ready, detail = availability()
    print(detail)
    raise SystemExit(0 if ready else 1)
