#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Helpers for isolated OCR runtimes.

Torch/ONNX OCR packages often lag behind the newest CPython release.  The main
application may run on Python 3.14, while Manga-OCR/NDLOCR-Lite need a
3.10-3.13 interpreter.  This module locates a genuinely executable compatible
Python instead of trusting that a hard-coded path merely exists.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable



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


def probe_python(path: str, min_minor: int = 10, max_minor: int = 13) -> tuple[bool, str]:
    """Return ``(usable, detail)`` for a Python candidate."""
    candidate = Path(path).expanduser()
    if not candidate.exists():
        return False, "文件不存在"
    try:
        proc = subprocess.run(
            [str(candidate), "-c", (
                "import sys,venv; "
                "print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'); "
                "raise SystemExit(0 if sys.version_info.major==3 and "
                f"{min_minor}<=sys.version_info.minor<={max_minor} else 8)"
            )],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as exc:
        return False, f"无法启动: {exc}"
    version = (proc.stdout or proc.stderr or "未知版本").strip().splitlines()[-1]
    if proc.returncode != 0:
        return False, f"版本不兼容或 venv 不可用: {version}"
    return True, version


def find_compatible_python(
    *, min_minor: int = 10, max_minor: int = 13, label: str = "OCR"
) -> Path:
    failures: list[str] = []
    for candidate in _candidate_paths():
        ok, detail = probe_python(candidate, min_minor=min_minor, max_minor=max_minor)
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
) -> Path:
    """Create/repair a venv and install dependencies idempotently."""
    py = venv_python(venv_dir)

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
        try:
            alive = subprocess.run(
                [str(py), "-c", (
                    "import sys; "
                    f"raise SystemExit(0 if sys.version_info.major==3 and {min_minor}<=sys.version_info.minor<={max_minor} else 8)"
                )],
                capture_output=True,
                timeout=15,
            ).returncode == 0
        except Exception:
            alive = False
        if not alive:
            shutil.rmtree(venv_dir, ignore_errors=True)

    if not py.exists():
        base = find_compatible_python(
            min_minor=min_minor, max_minor=max_minor, label=label
        )
        if verbose:
            print(f"🔧  首次使用 {label}：用 {base} 创建 {venv_dir} ...")
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(base), "-m", "venv", str(venv_dir)], check=True, timeout=300)
        subprocess.run(
            [str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
            check=True,
            timeout=1800,
        )

    if not valid_existing():
        if verbose:
            print(f"📦  安装/修复 {label} 依赖，首次使用需要下载模型或运行库 ...")
        cmd = [str(py), "-m", "pip", "install"]
        if requirements is not None:
            cmd.extend(["-r", str(requirements)])
        else:
            cmd.extend(packages or [])
        subprocess.run(cmd, check=True, timeout=3600)
        proc = subprocess.run([str(py), "-c", marker_code], capture_output=True, text=True, timeout=60)
        if proc.returncode != 0:
            raise RuntimeError(
                f"{label} 环境安装后仍无法导入：\n{(proc.stderr or proc.stdout)[-3000:]}"
            )

    return py
