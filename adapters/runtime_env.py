#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Helpers for isolated OCR runtimes.

Torch/ONNX OCR packages often lag behind the newest CPython release.  The main
application may run on Python 3.14, while some local OCR engines need a
3.10-3.13 interpreter.  This module locates a genuinely executable compatible
Python instead of trusting that a hard-coded path merely exists.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Iterable



def persistent_runtime_root() -> Path:
    """Per-user OCR runtime root shared by application/source upgrades."""
    override = os.environ.get("NOVEL_FORMATTER_OCR_RUNTIME_HOME", "").strip()
    if override:
        return Path(override).expanduser()
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Caches" / "NovelFormatter" / "ocr-runtimes"
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA", "").strip()
        return (Path(base) if base else home / "AppData" / "Local") / "NovelFormatter" / "ocr-runtimes"
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    return (Path(xdg).expanduser() if xdg else home / ".cache") / "novel-formatter" / "ocr-runtimes"


def persistent_venv_dir(name: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in str(name or "ocr"))
    return persistent_runtime_root() / safe / "venv"

def _candidate_paths() -> Iterable[str]:
    seen: set[str] = set()

    def add(value: str | None):
        if not value:
            return
        value = str(value)
        if value in seen:
            return
        seen.add(value)
        yield value

    for env_name in (
        "NOVEL_FORMATTER_OCR_PYTHON",
        "NOVEL_FORMATTER_PYTHON313",
        "NOVEL_FORMATTER_PYTHON",
    ):
        yield from add(os.environ.get(env_name))

    yield from add(sys.executable)

    for name in ("python3.13", "python3.12", "python3.11", "python3.10", "python3"):
        yield from add(shutil.which(name))

    for path in (
        "/opt/homebrew/bin/python3.13",
        "/opt/homebrew/bin/python3.12",
        "/opt/homebrew/bin/python3.11",
        "/usr/local/bin/python3.13",
        "/usr/local/bin/python3.12",
        "/usr/local/bin/python3.11",
        "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3",
        "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3",
        "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3",
        "/opt/local/bin/python3.13",
        "/opt/local/bin/python3.12",
        "/opt/local/bin/python3.11",
    ):
        yield from add(path)

    # Windows py launcher: resolve it to the real executable before returning.
    py_launcher = shutil.which("py")
    if py_launcher:
        for version in ("3.13", "3.12", "3.11", "3.10"):
            try:
                proc = subprocess.run(
                    [py_launcher, f"-{version}", "-c", "import sys;print(sys.executable)"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except Exception:
                continue
            if proc.returncode == 0 and proc.stdout.strip():
                yield from add(proc.stdout.strip().splitlines()[-1])


def _host_is_apple_silicon() -> bool:
    """Detect Apple Silicon even when the parent app itself is under Rosetta."""
    if sys.platform != "darwin":
        return False
    if platform.machine().strip().lower() in {"arm64", "aarch64"}:
        return True
    try:
        proc = subprocess.run(
            ["/usr/sbin/sysctl", "-in", "sysctl.proc_translated"],
            capture_output=True, text=True, timeout=5,
        )
        return proc.returncode == 0 and (proc.stdout or "").strip() == "1"
    except Exception:
        return False


def probe_python(
    path: str, min_minor: int = 10, max_minor: int = 13,
    *, require_native_apple_silicon: bool | None = None,
) -> tuple[bool, str]:
    """Return ``(usable, detail)`` for a Python candidate.

    On Apple Silicon, an x86_64/Rosetta interpreter can create a perfectly
    importable venv while making PyTorch MPS unavailable.  Treat that runtime
    as incompatible before installing hundreds of MB of OCR dependencies.
    """
    candidate = Path(path).expanduser()
    if not candidate.exists():
        return False, "文件不存在"
    if require_native_apple_silicon is None:
        require_native_apple_silicon = _host_is_apple_silicon()
    code = (
        "import json,platform,sys,venv; "
        "v=f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'; "
        "print(json.dumps({'version':v,'machine':platform.machine().lower(),"
        "'bits':64 if sys.maxsize>2**32 else 32})); "
        "raise SystemExit(0 if sys.version_info.major==3 and "
        f"{min_minor}<=sys.version_info.minor<={max_minor} else 8)"
    )
    try:
        proc = subprocess.run(
            [str(candidate), "-c", code], capture_output=True, text=True, timeout=15,
        )
    except Exception as exc:
        return False, f"无法启动: {exc}"
    raw = (proc.stdout or proc.stderr or "").strip().splitlines()
    payload = {}
    if raw:
        try:
            payload = json.loads(raw[-1])
        except Exception:
            payload = {"version": raw[-1]}
    version = str(payload.get("version") or "未知版本")
    machine = str(payload.get("machine") or "").strip().lower()
    bits = int(payload.get("bits") or 0)
    detail = version + (f" · {machine}" if machine else "") + (f" · {bits}-bit" if bits else "")
    if proc.returncode != 0:
        return False, f"版本不兼容或 venv 不可用: {detail}"
    if require_native_apple_silicon and machine not in {"arm64", "aarch64"}:
        return False, f"Apple Silicon 需要原生 arm64 Python，当前为 {detail}"
    return True, detail


def find_compatible_python(
    *, min_minor: int = 10, max_minor: int = 13, label: str = "OCR",
    require_native_apple_silicon: bool | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    failures: list[str] = []
    for candidate in _candidate_paths():
        ok, detail = probe_python(
            candidate, min_minor=min_minor, max_minor=max_minor,
            require_native_apple_silicon=require_native_apple_silicon,
        )
        if ok:
            return Path(candidate).resolve()
        failures.append(f"- {candidate}: {detail}")

    details = "\n".join(failures[-12:]) or "（没有发现候选解释器）"
    raise RuntimeError(
        f"{label} 需要可执行的 Python 3.{min_minor}～3.{max_minor}。\n"
        "请安装 Python 3.13（Mac 推荐 `brew install python@3.13`），或设置：\n"
        "NOVEL_FORMATTER_OCR_PYTHON=/完整路径/python3.13\n\n"
        f"已检查：\n{details}"
    )


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def ensure_venv(
    venv_dir: Path,
    *,
    label: str,
    marker_code: str,
    packages: list[str] | None = None,
    requirements: Path | None = None,
    verbose: bool = True,
    min_minor: int = 10,
    max_minor: int = 13,
    require_native_apple_silicon: bool | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """Create/repair a venv and install dependencies idempotently.

    ``progress_callback`` is optional and is used by heavyweight OCR runtimes
    to emit a heartbeat while Python/venv/pip subprocesses are still healthy.
    Existing callers keep the exact blocking ``subprocess.run`` behavior when
    no callback is supplied.
    """
    py = venv_python(venv_dir)

    def emit_progress(detail: str) -> None:
        if callable(progress_callback):
            try:
                progress_callback(str(detail or ""))
            except Exception:
                pass

    def run_checked(cmd, *, timeout: float, detail: str) -> None:
        if not callable(progress_callback):
            subprocess.run(cmd, check=True, timeout=timeout)
            return
        emit_progress(detail)
        proc = subprocess.Popen(cmd)
        deadline = time.monotonic() + max(1.0, float(timeout))
        last_emit = 0.0
        while proc.poll() is None:
            now = time.monotonic()
            if now >= deadline:
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass
                raise subprocess.TimeoutExpired(cmd, timeout)
            if now - last_emit >= 5.0:
                last_emit = now
                emit_progress(detail)
            time.sleep(0.25)
        if proc.returncode:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
        emit_progress(detail)

    def valid_existing() -> bool:
        if not py.exists():
            return False
        try:
            proc = subprocess.run(
                [str(py), "-c", marker_code], capture_output=True, timeout=30
            )
            return proc.returncode == 0
        except Exception:
            return False

    # Existing venvs can outlive or point at a removed framework Python.
    # Verify the interpreter itself before trying to repair packages inside it.
    if py.exists():
        alive, _detail = probe_python(
            str(py), min_minor=min_minor, max_minor=max_minor,
            require_native_apple_silicon=require_native_apple_silicon,
        )
        if not alive:
            if verbose:
                print(f"♻️  {label} 运行环境需要重建：{_detail}")
            shutil.rmtree(venv_dir, ignore_errors=True)

    if not py.exists():
        base = find_compatible_python(
            min_minor=min_minor, max_minor=max_minor, label=label,
            require_native_apple_silicon=require_native_apple_silicon,
        )
        if verbose:
            print(f"🔧  首次使用 {label}：用 {base} 创建 {venv_dir} ...")
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
        run_checked(
            [str(base), "-m", "venv", str(venv_dir)],
            timeout=300,
            detail=f"正在创建 {label} 独立运行环境",
        )
        run_checked(
            [str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
            timeout=1800,
            detail=f"正在准备 {label} Python 基础依赖",
        )

    if not valid_existing():
        if verbose:
            print(f"📦  安装/修复 {label} 依赖，首次使用需要下载模型或运行库 ...")
        cmd = [str(py), "-m", "pip", "install"]
        if requirements is not None:
            cmd.extend(["-r", str(requirements)])
        else:
            cmd.extend(packages or [])
        run_checked(
            cmd,
            timeout=3600,
            detail=f"正在安装/修复 {label} 依赖（首次使用可能下载较大运行库）",
        )
        proc = subprocess.run([str(py), "-c", marker_code], capture_output=True, text=True, timeout=60)
        if proc.returncode != 0:
            raise RuntimeError(
                f"{label} 环境安装后仍无法导入：\n{(proc.stderr or proc.stdout)[-3000:]}"
            )

    return py
