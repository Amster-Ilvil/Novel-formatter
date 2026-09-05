#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Official PaddleOCR-VL 1.6 MLX-VLM runtime for Apple Silicon.

PaddleOCR officially supports ``mlx-vlm-server`` as the VLM recognition backend
on Apple Silicon.  Novel Formatter keeps the PaddleOCR client/layout pipeline in
``.venv-paddle`` and isolates MLX-VLM in its own environment so MLX/Transformers
packages cannot disturb the already verified Paddle/PP-OCR runtimes.

This module only manages the local MLX-VLM service.  It does not reimplement or
convert PaddleOCR-VL weights, and it never changes PP-OCRv6/PP-Structure paths.
"""
from __future__ import annotations

import atexit
import os
import platform
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from adapters.runtime_env import ensure_venv, venv_python

ROOT = Path(__file__).parent.parent
VENV_DIR = ROOT / ".venv-mlx-vlm"
VENV_PYTHON = venv_python(VENV_DIR)
MLX_VLM_PACKAGE_SPEC = os.environ.get(
    "NOVEL_FORMATTER_MLX_VLM_PACKAGE", "mlx-vlm>=0.3.11"
).strip() or "mlx-vlm>=0.3.11"
DEFAULT_MODEL = os.environ.get(
    "NOVEL_FORMATTER_PADDLE_VL_MLX_MODEL", "PaddlePaddle/PaddleOCR-VL-1.6"
).strip() or "PaddlePaddle/PaddleOCR-VL-1.6"
SERVER_RUNNER = Path(__file__).parent / "paddle_vl_mlx_server_runner.py"


def is_apple_silicon() -> bool:
    machine = platform.machine().strip().lower()
    return sys.platform == "darwin" and machine in {"arm64", "aarch64"}


def normalize_vl_backend(value: str | None) -> str:
    raw = str(value or "auto").strip().lower().replace("_", "-")
    aliases = {
        "": "auto",
        "default": "auto",
        "automatic": "auto",
        "mlx-vlm": "mlx",
        "mlx-vlm-server": "mlx",
        "native": "paddle",
        "paddlepaddle": "paddle",
        "cpu": "paddle",
    }
    raw = aliases.get(raw, raw)
    return raw if raw in {"auto", "mlx", "paddle"} else "auto"


def _marker_code() -> str:
    # Keep the probe self-contained because it runs inside the isolated venv.
    return (
        "import platform, sys, mlx, mlx_vlm; "
        "from importlib.metadata import version; "
        "parts=version('mlx-vlm').split('+',1)[0].split('.'); "
        "nums=tuple(int(''.join(c for c in p if c.isdigit()) or 0) for p in parts[:3]); "
        "assert nums >= (0,3,11), 'mlx-vlm>=0.3.11 required'; "
        "assert sys.platform=='darwin' and platform.machine().lower() in ('arm64','aarch64'), "
        "'MLX-VLM requires Apple Silicon macOS'"
    )


def setup_mlx_venv(*, verbose: bool = True) -> Path:
    if not is_apple_silicon():
        raise RuntimeError("PaddleOCR-VL 的官方 MLX-VLM 后端仅支持 Apple Silicon macOS。")
    return ensure_venv(
        VENV_DIR,
        label="PaddleOCR-VL · MLX-VLM",
        marker_code=_marker_code(),
        packages=[MLX_VLM_PACKAGE_SPEC],
        verbose=verbose,
        min_minor=10,
        max_minor=13,
    )


def probe_mlx_runtime(*, deep: bool = False) -> tuple[bool, str]:
    if not is_apple_silicon():
        return False, "当前不是 Apple Silicon macOS；MLX 后端不会启用"
    if not VENV_PYTHON.exists():
        return False, "MLX-VLM 独立环境尚未安装"
    if not deep:
        return True, "已检测到 MLX-VLM 独立环境"
    try:
        proc = subprocess.run(
            [str(VENV_PYTHON), "-c", _marker_code()],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        return False, f"MLX-VLM 环境检测失败：{exc}"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "导入失败").strip().splitlines()
        return False, f"MLX-VLM 环境不可用：{detail[-1] if detail else '未知错误'}"
    try:
        version_proc = subprocess.run(
            [str(VENV_PYTHON), "-c", "from importlib.metadata import version; print(version('mlx-vlm'))"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        version_text = version_proc.stdout.strip() if version_proc.returncode == 0 else ""
    except Exception:
        version_text = ""
    return True, f"MLX-VLM {version_text or '>=0.3.11'} 可用"


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _tcp_ready(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=0.35):
            return True
    except OSError:
        return False


class _MlxVlmServer:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._proc: subprocess.Popen | None = None
        self._log_file = None
        self._port = 0
        self._python = ""

    @property
    def running(self) -> bool:
        proc = self._proc
        return bool(proc is not None and proc.poll() is None and self._port and _tcp_ready(self._port))

    @property
    def server_url(self) -> str:
        return f"http://127.0.0.1:{self._port}/" if self._port else ""

    def log_tail(self, limit: int = 3000) -> str:
        handle = self._log_file
        if handle is None:
            return ""
        try:
            handle.flush()
            handle.seek(0)
            return handle.read()[-limit:]
        except Exception:
            return ""

    def start(self, *, verbose: bool = False, progress_callback=None) -> str:
        with self._lock:
            if self.running:
                return self.server_url
            self.stop()
            python = setup_mlx_venv(verbose=verbose)
            port_override = str(os.environ.get("NOVEL_FORMATTER_PADDLE_VL_MLX_PORT") or "").strip()
            try:
                port = int(port_override) if port_override else _free_local_port()
            except ValueError:
                port = _free_local_port()
            if not (1 <= port <= 65535):
                port = _free_local_port()

            self._log_file = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
            cmd = [
                str(python), str(SERVER_RUNNER),
                "--host", "127.0.0.1",
                "--port", str(port),
            ]
            env = dict(os.environ)
            env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            env.setdefault("TOKENIZERS_PARALLELISM", "false")
            env["NOVEL_FORMATTER_MLX_PARENT_PID"] = str(os.getpid())
            from adapters.subprocess_watchdog import isolated_process_kwargs
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                **isolated_process_kwargs(),
            )
            self._port = port
            self._python = str(python)
            if progress_callback is not None:
                progress_callback(f"正在启动 PaddleOCR-VL 官方 MLX-VLM 服务 · 127.0.0.1:{port}")

            timeout = float(os.environ.get("NOVEL_FORMATTER_MLX_VLM_STARTUP_TIMEOUT", "120") or 120)
            deadline = time.monotonic() + max(15.0, timeout)
            while time.monotonic() < deadline:
                proc = self._proc
                if proc is None:
                    break
                if proc.poll() is not None:
                    detail = self.log_tail()
                    self.stop()
                    raise RuntimeError(
                        "MLX-VLM 服务启动失败" + (f"：\n{detail}" if detail else "")
                    )
                if _tcp_ready(port):
                    if progress_callback is not None:
                        progress_callback("MLX-VLM 服务已启动；PaddleOCR-VL 将通过官方 mlx-vlm-server 接口调用")
                    return self.server_url
                time.sleep(0.15)

            detail = self.log_tail()
            self.stop()
            raise RuntimeError(
                "MLX-VLM 服务启动超时" + (f"：\n{detail}" if detail else "")
            )

    def stop(self) -> None:
        with self._lock:
            proc = self._proc
            self._proc = None
            if proc is not None:
                try:
                    from adapters.subprocess_watchdog import terminate_process
                    terminate_process(proc, grace=3.0)
                except Exception:
                    try:
                        if proc.poll() is None:
                            proc.kill()
                    except Exception:
                        pass
            if self._log_file is not None:
                try:
                    self._log_file.close()
                except Exception:
                    pass
            self._log_file = None
            self._port = 0
            self._python = ""


_SERVER = _MlxVlmServer()
atexit.register(_SERVER.stop)


def prepare_vl_backend(
    requested: str | None = "auto",
    *,
    verbose: bool = False,
    progress_callback=None,
) -> dict:
    """Resolve/start the PaddleOCR-VL backend without disturbing other OCRs.

    ``auto`` means MLX on Apple Silicon and native Paddle everywhere else.  Any
    MLX setup/start failure in auto mode is deliberately converted to a Paddle
    fallback so an OCR job remains functional.  Explicit ``mlx`` keeps errors
    visible to users who specifically requested that backend.
    """
    requested_backend = normalize_vl_backend(requested)
    if requested_backend == "paddle":
        return {
            "requested": requested_backend,
            "backend": "paddle",
            "server_url": "",
            "model": DEFAULT_MODEL,
            "detail": "使用 PaddleOCR-VL 原生 Paddle 后端",
        }

    if not is_apple_silicon():
        detail = "当前平台不支持 MLX；已使用 PaddleOCR-VL 原生 Paddle 后端"
        if requested_backend == "mlx":
            raise RuntimeError(detail)
        if progress_callback is not None:
            progress_callback(detail)
        return {
            "requested": requested_backend,
            "backend": "paddle",
            "server_url": "",
            "model": DEFAULT_MODEL,
            "detail": detail,
        }

    try:
        url = _SERVER.start(verbose=verbose, progress_callback=progress_callback)
        return {
            "requested": requested_backend,
            "backend": "mlx",
            "server_url": url,
            "model": DEFAULT_MODEL,
            "detail": "Apple Silicon · 官方 MLX-VLM",
        }
    except Exception as exc:
        if requested_backend == "mlx":
            raise
        detail = f"MLX-VLM 不可用，已自动回退 Paddle：{exc}"
        if progress_callback is not None:
            progress_callback(detail)
        return {
            "requested": requested_backend,
            "backend": "paddle",
            "server_url": "",
            "model": DEFAULT_MODEL,
            "detail": detail,
        }


def stop_mlx_server() -> None:
    _SERVER.stop()
