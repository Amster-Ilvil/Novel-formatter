#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Windows Snipping Tool OneOCR adapter.

Discovers OneOCR inside the user's installed Microsoft Snipping Tool package.
No Microsoft binaries/models are downloaded, copied into the project, or
redistributed. OneOCR is an undocumented compatibility ABI and is therefore
kept behind a fail-closed subprocess boundary.
"""
from __future__ import annotations

import os
import hashlib
import platform
import re
import shutil
import statistics
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from adapters.ocr_engine_common import iter_server_worker_jsonl, run_ocr_engine
from adapters.runtime_env import persistent_runtime_root

PACKAGE_NAME = "Microsoft.ScreenSketch"
PACKAGE_FAMILY = "Microsoft.ScreenSketch_8wekyb3d8bbwe"
REQUIRED_FILES = ("oneocr.dll", "oneocr.onemodel", "onnxruntime.dll")
EXECUTABLE_FILES = ("oneocr.dll", "onnxruntime.dll")
MODEL_FILE = "oneocr.onemodel"
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
                _powershell_executable(), "-NoLogo", "-NoProfile", "-NonInteractive",
                "-ExecutionPolicy", "Bypass", "-Command", script,
            ],
            capture_output=True, text=True, timeout=max(1.0, float(timeout)),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    value = (result.stdout or "").strip().strip('"')
    return Path(value) if value else None


def _version_key(value: str) -> tuple[int, ...]:
    match = re.search(r"\d+(?:\.\d+){1,3}", str(value or ""))
    return tuple(int(part) for part in match.group(0).split(".")) if match else ()


def _platform_package_architecture() -> str:
    machine = platform.machine().strip().lower()
    if machine in {"amd64", "x86_64"}:
        return "x64"
    if machine in {"arm64", "aarch64"}:
        return "arm64"
    if machine in {"x86", "i386", "i686"}:
        return "x86"
    return machine


def _query_provisioned_screen_sketch_install_location() -> Path | None:
    """Find a staged/provisioned Snipping Tool when Get-AppxPackage is blocked.

    Windows keeps the bundle manifest path in AppxAllUserStore. Reading that
    manifest is enough to identify the architecture-specific application
    package without changing package registration or requiring a Store update.
    """
    if sys.platform != "win32":
        return None
    try:
        import winreg
    except Exception:
        return None

    key_path = (
        r"SOFTWARE\Microsoft\Windows\CurrentVersion\Appx\AppxAllUserStore"
        r"\Applications"
    )
    bundle_prefix = f"{PACKAGE_NAME}_"
    bundle_suffix = "_neutral_~_8wekyb3d8bbwe"
    expected_arch = _platform_package_architecture()
    candidates: list[tuple[tuple[int, ...], Path]] = []
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as root:
            index = 0
            while True:
                try:
                    name = winreg.EnumKey(root, index)
                except OSError:
                    break
                index += 1
                if not (name.startswith(bundle_prefix) and name.endswith(bundle_suffix)):
                    continue
                try:
                    with winreg.OpenKey(root, name) as package_key:
                        manifest_value, _ = winreg.QueryValueEx(package_key, "Path")
                    manifest = Path(str(manifest_value))
                    tree = ET.parse(manifest)
                    windows_apps = manifest.parents[2]
                    for element in tree.getroot().iter():
                        if not element.tag.endswith("Package"):
                            continue
                        if str(element.attrib.get("Type", "")).lower() != "application":
                            continue
                        arch = str(element.attrib.get("Architecture", "")).lower()
                        if expected_arch and arch != expected_arch:
                            continue
                        version = str(element.attrib.get("Version", ""))
                        package_root = windows_apps / (
                            f"{PACKAGE_NAME}_{version}_{arch}__8wekyb3d8bbwe"
                        )
                        runtime = _valid_runtime_dir(package_root)
                        if runtime is not None:
                            candidates.append((_version_key(version), runtime))
                except (OSError, ValueError, ET.ParseError, IndexError):
                    continue
    except OSError:
        return None
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def find_runtime_dir() -> Path | None:
    """Locate Snipping Tool OneOCR without downloading or copying anything."""
    override = os.environ.get("NOVEL_FORMATTER_SNIPPING_OCR_DIR", "").strip()
    if override:
        return _valid_runtime_dir(override)
    if sys.platform != "win32":
        return None
    registered = _valid_runtime_dir(_query_screen_sketch_install_location())
    if registered is not None:
        return registered
    return _query_provisioned_screen_sketch_install_location()


def _needs_private_dll_cache(runtime: Path) -> bool:
    package_root_name = runtime.parent.name.lower()
    return (
        package_root_name.startswith("microsoft.screensketch_")
        or any(part.lower() == "windowsapps" for part in runtime.parts)
    )


def _private_dll_cache_root() -> Path:
    override = os.environ.get("NOVEL_FORMATTER_SNIPPING_OCR_CACHE_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return persistent_runtime_root() / "windows-snipping-ocr"


def _cache_key(runtime: Path) -> str:
    details = [str(runtime.resolve()).casefold()]
    for name in REQUIRED_FILES:
        stat = (runtime / name).stat()
        details.append(f"{name}:{stat.st_size}:{stat.st_mtime_ns}")
    digest = hashlib.sha256("\n".join(details).encode("utf-8")).hexdigest()[:16]
    label = "-".join(
        part for part in re.split(r"[^A-Za-z0-9._-]+", runtime.parent.name) if part
    )
    return f"{label[:64] or 'snipping-tool'}-{digest}"


def _cached_dlls_match(source: Path, cache: Path) -> bool:
    def digest(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    try:
        return all(
            (cache / name).is_file()
            and (cache / name).stat().st_size == (source / name).stat().st_size
            and digest(cache / name) == digest(source / name)
            for name in EXECUTABLE_FILES
        )
    except OSError:
        return False


def prepare_loadable_runtime(runtime: str | Path) -> tuple[Path, Path]:
    """Return a DLL directory and model path usable by an unpackaged process.

    Windows can allow normal reads from another MSIX package while refusing to
    map its DLLs into an unpackaged process (WinError 5). Copy only the two
    executable DLLs into Novel Formatter's per-user cache. The large OneOCR
    model remains in the installed Snipping Tool package and is never copied or
    downloaded.
    """
    source = _valid_runtime_dir(runtime)
    if source is None:
        raise RuntimeError("OneOCR 运行目录不完整")
    model_path = source / MODEL_FILE
    if not _needs_private_dll_cache(source):
        return source, model_path

    cache = _private_dll_cache_root() / _cache_key(source)
    if _cached_dlls_match(source, cache):
        return cache.resolve(), model_path.resolve()

    cache.mkdir(parents=True, exist_ok=True)
    for name in EXECUTABLE_FILES:
        source_file = source / name
        target_file = cache / name
        temporary = cache / f".{name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        try:
            shutil.copy2(source_file, temporary)
            if temporary.stat().st_size != source_file.stat().st_size:
                raise RuntimeError(f"缓存 {name} 时文件大小校验失败")
            os.replace(temporary, target_file)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    if not _cached_dlls_match(source, cache):
        raise RuntimeError("Windows Snipping OCR 本地 DLL 缓存不完整")
    return cache.resolve(), model_path.resolve()


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


def _item_rect(item: dict) -> tuple[float, float, float, float] | None:
    """Return a conservative axis-aligned rectangle for one OneOCR item."""
    points = item.get("box")
    if not isinstance(points, (list, tuple)) or len(points) < 2:
        return None
    try:
        xs = [float(point[0]) for point in points if len(point) >= 2]
        ys = [float(point[1]) for point in points if len(point) >= 2]
    except (TypeError, ValueError, OverflowError):
        return None
    if not xs or not ys:
        return None
    left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _normalise_vertical_text(text: str) -> str:
    """Remove OneOCR's artificial ASCII spacing around vertical ellipses."""
    value = str(text or "").strip()
    return re.sub(r"\s*([…‥]+)\s*", r"\1", value)


def _order_vertical_page_items(items: list[dict]) -> list[dict]:
    """Recover Japanese vertical full-page order and suppress side Ruby.

    OneOCR returns full-page vertical lines in backend order (commonly left to
    right) and exposes furigana as separate narrow lines.  Novel text reads by
    physical column from right to left, while fragments within one column read
    top to bottom.  Keep isolated narrow captions, but discard narrow blocks
    that overlap and sit immediately beside a normal-width body column.
    """
    values = [dict(item) for item in items]
    measured: list[tuple[int, dict, tuple[float, float, float, float]]] = []
    unboxed: list[tuple[int, dict]] = []
    for index, item in enumerate(values):
        rect = _item_rect(item)
        if rect is None:
            unboxed.append((index, item))
        else:
            measured.append((index, item, rect))
    if len(measured) < 3:
        return values

    widths = [rect[2] - rect[0] for _index, _item, rect in measured]
    reference_width = float(statistics.median_high(sorted(widths)))
    narrow_limit = max(10.0, reference_width * 0.72)
    ordinary = [entry for entry in measured if entry[2][2] - entry[2][0] >= narrow_limit]
    if len(ordinary) < 2:
        ordinary = measured

    kept: list[tuple[int, dict, tuple[float, float, float, float]]] = []
    for entry in measured:
        _index, _item, rect = entry
        width = rect[2] - rect[0]
        if width >= narrow_limit:
            kept.append(entry)
            continue
        center_x = (rect[0] + rect[2]) / 2.0
        is_side_ruby = False
        for _main_index, _main_item, main in ordinary:
            main_center_x = (main[0] + main[2]) / 2.0
            vertical_overlap = min(rect[3], main[3]) - max(rect[1], main[1])
            if (
                vertical_overlap > 0
                and abs(center_x - main_center_x) <= reference_width * 1.65
            ):
                is_side_ruby = True
                break
        if not is_side_ruby:
            kept.append(entry)

    kept.sort(key=lambda entry: (
        -((entry[2][0] + entry[2][2]) / 2.0),
        entry[2][1],
        entry[0],
    ))
    return [entry[1] for entry in kept] + [item for _index, item in unboxed]


def _run_worker(
    image_paths: list[str], *, cancel_check=None, verbose: bool = True,
    ocr_mode: str = "ja_vertical",
):
    del verbose
    ready, detail = availability()
    if not ready:
        raise RuntimeError(detail)
    runtime = find_runtime_dir()
    if runtime is None:
        raise RuntimeError(detail)
    dll_runtime, model_path = prepare_loadable_runtime(runtime)
    cmd = [
        sys.executable, str(WORKER_SCRIPT),
        "--runtime-dir", str(dll_runtime),
        "--model-path", str(model_path),
        "--server",
    ]
    for path, blocks, error in iter_server_worker_jsonl(
        cmd, [str(path) for path in image_paths],
        cancel_check=cancel_check,
        engine_label="Windows Snipping OCR",
        batch_size=128,
    ):
        if not error and str(ocr_mode or "").strip().lower() == "ja_vertical":
            normalized = []
            for raw in list(blocks or []):
                item = dict(raw)
                item["text"] = _normalise_vertical_text(item.get("text", ""))
                normalized.append(item)
            blocks = _order_vertical_page_items(normalized)
        yield path, blocks, error


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
    def worker(paths, cancel_check=None):
        yield from _run_worker(
            paths, cancel_check=cancel_check, verbose=verbose, ocr_mode=ocr_mode
        )

    return run_ocr_engine(
        worker, source_engine="windows_snipping_ocr",
        image_folder=image_folder, page_overrides=page_overrides,
        verbose=verbose, input_paths=input_paths,
        progress_callback=progress_callback, cancel_check=cancel_check,
        crop_top=crop_top, crop_bottom=crop_bottom, crop_rect=crop_rect,
        temp_crop_dir=temp_crop_dir, reuse_existing_crops=reuse_existing_crops,
        filter_running_headers=filter_running_headers,
        force_text_pages=force_text_pages,
        strict_column_audit=strict_column_audit,
        ocr_mode=ocr_mode,
        merge_horizontal_fragments=merge_horizontal_fragments,
        performance_callback=performance_callback,
        worker_timing_name="windows_snipping_oneocr",
    )


__all__ = [
    "PACKAGE_NAME", "PACKAGE_FAMILY", "REQUIRED_FILES", "EXECUTABLE_FILES",
    "MODEL_FILE", "WORKER_SCRIPT", "availability", "find_runtime_dir",
    "prepare_loadable_runtime", "runtime_status", "run",
]


if __name__ == "__main__":
    ready, detail = availability()
    print(detail)
    raise SystemExit(0 if ready else 1)
