#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NDLOCR-Lite adapter with automatic source/model/dependency installation.

The official release title may be prefixed with ``v`` while the Git tag itself
may omit it. Do not hard-code the visible release title as a Git branch: resolve
the latest release tag through GitHub's API and gracefully fall back to compatible
tag spellings or the default branch.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import queue
import threading
import time
import uuid
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from utils.safe_archive import safe_extract_zip
from utils.apple_silicon_runtime import is_m6, profile, recommended_cpu_threads

from adapters.ocr_engine_common import iter_worker_jsonl, run_ocr_engine
from adapters.runtime_env import ensure_venv, persistent_runtime_root

ROOT = Path(__file__).parent.parent
RUNTIME_ROOT = ROOT / ".ocr-runtimes"
SOURCE_DIR = RUNTIME_ROOT / "ndlocr-lite"
VENV_DIR = ROOT / ".venv-ndlocr-lite"
WORKER_SCRIPT = Path(__file__).parent / "ndlocr_lite_worker.py"
REPO_URL = "https://github.com/ndl-lab/ndlocr-lite.git"
LATEST_RELEASE_API = "https://api.github.com/repos/ndl-lab/ndlocr-lite/releases/latest"
# Offline fallback only. Online first installation resolves releases/latest.
FALLBACK_REF = "1.3.1"
REF_MARKER = ".novel-formatter-ndlocr-ref"
MODEL_MIN_BYTES = 1_000_000
USER_AGENT = "Novel-Formatter-NDLOCR-Installer/1.1"


def _ndlocr_worker_env(engine_options: dict | None = None) -> dict[str, str]:
    env = os.environ.copy()
    options = dict(engine_options or {})
    threads = str(recommended_cpu_threads())
    if threads:
        env.setdefault("OMP_NUM_THREADS", threads)
        env.setdefault("OPENBLAS_NUM_THREADS", threads)
        env.setdefault("MKL_NUM_THREADS", threads)
        env.setdefault("VECLIB_MAXIMUM_THREADS", threads)
        env.setdefault("BLIS_NUM_THREADS", threads)
        env.setdefault("NUMEXPR_NUM_THREADS", threads)
        env.setdefault("OMP_WAIT_POLICY", "PASSIVE")
        env.setdefault("PYTHONUNBUFFERED", "1")
    enable_tcy = options.get("enable_tcy")
    if enable_tcy is not None:
        env["NOVEL_FORMATTER_NDLOCR_TCY"] = "1" if bool(enable_tcy) else "0"
    else:
        env["NOVEL_FORMATTER_NDLOCR_TCY"] = "0"

    # NDLOCR is ONNX Runtime, not a PyTorch/MPS model.  On Apple Silicon use a
    # conservative hybrid by default: page detector may use CoreML/ANE while
    # PARSeq recognizers stay on CPU so character logits retain the stable path.
    requested_provider = str(
        options.get("execution_provider")
        or options.get("provider")
        or env.get("NOVEL_FORMATTER_NDLOCR_PROVIDER")
        or ("auto" if profile().available else "cpu")
    ).strip().lower()
    env["NOVEL_FORMATTER_NDLOCR_PROVIDER"] = requested_provider
    cache_dir = persistent_runtime_root() / "ndlocr-lite" / "coreml-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    env.setdefault("NOVEL_FORMATTER_NDLOCR_COREML_CACHE", str(cache_dir))
    env.setdefault("NOVEL_FORMATTER_NDLOCR_COREML_UNITS", "CPUAndNeuralEngine")
    return env


def _is_real_model(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size < MODEL_MIN_BYTES:
            return False
        with path.open("rb") as fh:
            return not fh.read(128).startswith(b"version https://git-lfs")
    except OSError:
        return False


def _models_ready(source_dir: Path | None = None) -> bool:
    source_dir = source_dir or SOURCE_DIR
    model_dir = source_dir / "src" / "model"
    models = list(model_dir.glob("*.onnx"))
    return len(models) >= 4 and all(_is_real_model(path) for path in models)


def _ref_candidates(ref: str) -> list[str]:
    """Return tag/branch spellings in safe retry order.

    GitHub release titles and Git tags have not always used the same ``v`` prefix.
    Supporting both forms keeps environment overrides robust when users copy the
    visible release title.
    """
    value = (ref or "").strip()
    value = re.sub(r"^refs/tags/", "", value)
    if not value:
        return []
    out: list[str] = []

    def add(item: str):
        if item and item not in out:
            out.append(item)

    add(value)
    if re.fullmatch(r"v?\d+(?:\.\d+){1,3}(?:[-+][0-9A-Za-z.-]+)?", value):
        if value.startswith("v"):
            add(value[1:])
        else:
            add("v" + value)
    return out


def _request(url: str, *, accept: str | None = None) -> urllib.request.Request:
    headers = {"User-Agent": USER_AGENT}
    if accept:
        headers["Accept"] = accept
    return urllib.request.Request(url, headers=headers)


def _resolve_latest_ref(verbose: bool = True) -> str:
    override = os.environ.get("NOVEL_FORMATTER_NDLOCR_REF", "").strip()
    if override:
        return override
    try:
        with urllib.request.urlopen(
            _request(LATEST_RELEASE_API, accept="application/vnd.github+json"),
            timeout=20,
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
        tag = str(payload.get("tag_name") or "").strip()
        if tag:
            if verbose:
                print(f"🔎  NDLOCR-Lite 最新正式版标签：{tag}")
            return tag
    except Exception as exc:
        if verbose:
            print(f"⚠️  无法查询 NDLOCR-Lite 最新版本，回退 {FALLBACK_REF}: {exc}")
    return FALLBACK_REF


def _download_to(url: str, destination: Path, *, timeout: int = 120) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".part")
    temp.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(_request(url), timeout=timeout) as response, temp.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
        temp.replace(destination)
    finally:
        temp.unlink(missing_ok=True)


def _discover_model_names(source_dir: Path) -> list[str]:
    model_dir = source_dir / "src" / "model"
    names = [path.name for path in model_dir.glob("*.onnx")]
    ocr_py = source_dir / "src" / "ocr.py"
    if ocr_py.exists():
        try:
            text = ocr_py.read_text(encoding="utf-8", errors="ignore")
            for name in re.findall(r'["\']([^"\']+\.onnx)["\']', text):
                base = Path(name).name
                if base not in names:
                    names.append(base)
        except OSError:
            pass
    return names


def _try_git_lfs(source_dir: Path, verbose: bool) -> None:
    git = shutil.which("git")
    if not git:
        return
    # ``git lfs`` may be available even when no standalone git-lfs binary is in PATH.
    check = subprocess.run([git, "lfs", "version"], capture_output=True, text=True, timeout=30)
    if check.returncode != 0:
        return
    if verbose:
        print("⬇️  使用 Git LFS 补齐 NDLOCR-Lite ONNX 模型 ...")
    subprocess.run(
        [git, "-C", str(source_dir), "lfs", "pull"],
        capture_output=True,
        text=True,
        timeout=1800,
    )


def _download_models_direct(source_dir: Path, ref: str, verbose: bool) -> list[str]:
    """Download actual LFS model objects without requiring git-lfs."""
    failures: list[str] = []
    model_dir = source_dir / "src" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    names = _discover_model_names(source_dir)
    if not names:
        return ["无法从 src/ocr.py 或 src/model 发现 ONNX 文件名"]

    quoted_ref = urllib.parse.quote(ref, safe="")
    for name in names:
        target = model_dir / name
        if _is_real_model(target):
            continue
        quoted_name = urllib.parse.quote(name, safe="")
        # media.githubusercontent.com serves the actual Git LFS object rather than
        # the small pointer text returned by ordinary source archives.
        url = (
            "https://media.githubusercontent.com/media/ndl-lab/ndlocr-lite/"
            f"{quoted_ref}/src/model/{quoted_name}"
        )
        try:
            if verbose:
                print(f"⬇️  下载 NDLOCR-Lite 模型：{name}")
            _download_to(url, target, timeout=300)
            if not _is_real_model(target):
                raise RuntimeError("下载结果仍是 Git LFS 指针或文件过小")
        except Exception as exc:
            target.unlink(missing_ok=True)
            failures.append(f"{name}: {exc}")
    return failures


def _ensure_models(source_dir: Path, ref: str, verbose: bool) -> tuple[bool, list[str]]:
    if _models_ready(source_dir):
        return True, []
    _try_git_lfs(source_dir, verbose)
    if _models_ready(source_dir):
        return True, []
    failures = _download_models_direct(source_dir, ref, verbose)
    return _models_ready(source_dir), failures


def _write_ref_marker(source_dir: Path, ref: str) -> None:
    try:
        (source_dir / REF_MARKER).write_text(ref + "\n", encoding="utf-8")
    except OSError:
        pass


def _activate_source(staged_dir: Path) -> None:
    """Replace the active runtime only after its source and weights validate."""
    backup = RUNTIME_ROOT / f".ndlocr-lite-previous-{uuid.uuid4().hex}"
    moved_old = False
    if SOURCE_DIR.exists() or SOURCE_DIR.is_symlink():
        os.replace(SOURCE_DIR, backup)
        moved_old = True
    try:
        os.replace(staged_dir, SOURCE_DIR)
    except Exception:
        if moved_old and not SOURCE_DIR.exists():
            os.replace(backup, SOURCE_DIR)
        raise
    if moved_old:
        shutil.rmtree(backup, ignore_errors=True)


def _clone_source(ref: str, verbose: bool) -> tuple[bool, str]:
    git = shutil.which("git")
    if not git:
        return False, "系统未找到 git"
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"⬇️  git 下载 NDLOCR-Lite {ref} ...")
    with tempfile.TemporaryDirectory(prefix="ndlocr-clone-", dir=RUNTIME_ROOT) as tmp_name:
        staged = Path(tmp_name) / "source"
        try:
            proc = subprocess.run(
                [git, "clone", "--depth", "1", "--branch", ref, REPO_URL, str(staged)],
                capture_output=True,
                text=True,
                timeout=900,
            )
        except subprocess.TimeoutExpired:
            return False, "git clone 超过 15 分钟，已终止"
        if proc.returncode != 0:
            return False, (proc.stderr or proc.stdout)[-2000:]
        ready, model_failures = _ensure_models(staged, ref, verbose)
        if not ready:
            return False, "\n".join(model_failures[-6:]) or "ONNX 模型不完整"
        _write_ref_marker(staged, ref)
        try:
            _activate_source(staged)
        except OSError as exc:
            return False, f"无法切换到已验证的 NDLOCR-Lite 安装：{exc}"
        return True, ""


def _archive_url(ref: str) -> str:
    quoted = urllib.parse.quote(ref, safe="")
    if ref in {"main", "master"}:
        return f"https://codeload.github.com/ndl-lab/ndlocr-lite/zip/refs/heads/{quoted}"
    return f"https://codeload.github.com/ndl-lab/ndlocr-lite/zip/refs/tags/{quoted}"


def _download_archive_source(ref: str, verbose: bool) -> tuple[bool, str]:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="ndlocr-download-", dir=RUNTIME_ROOT) as tmp_name:
            tmp = Path(tmp_name)
            archive = tmp / "source.zip"
            extract_dir = tmp / "extract"
            extract_dir.mkdir()
            if verbose:
                print(f"⬇️  压缩包下载 NDLOCR-Lite {ref} ...")
            _download_to(_archive_url(ref), archive, timeout=180)
            safe_extract_zip(archive, extract_dir)
            roots = [p for p in extract_dir.iterdir() if p.is_dir()]
            if len(roots) != 1:
                return False, "GitHub压缩包目录结构异常"
            staged = tmp / "source"
            shutil.move(str(roots[0]), str(staged))
            ready, model_failures = _ensure_models(staged, ref, verbose)
            if not ready:
                return False, "\n".join(model_failures[-6:]) or "ONNX 模型不完整"
            _write_ref_marker(staged, ref)
            _activate_source(staged)
            return True, ""
    except Exception as exc:
        return False, str(exc)


def _download_source(verbose: bool = True) -> Path:
    # A complete cached installation remains usable offline. An explicit ref
    # override intentionally forces a version check/redownload.
    override = os.environ.get("NOVEL_FORMATTER_NDLOCR_REF", "").strip()
    if (SOURCE_DIR / "src" / "ocr.py").exists() and _models_ready():
        if not override:
            return SOURCE_DIR
        try:
            installed = (SOURCE_DIR / REF_MARKER).read_text(encoding="utf-8").strip()
        except OSError:
            installed = ""
        if installed in _ref_candidates(override):
            return SOURCE_DIR

    desired = _resolve_latest_ref(verbose=verbose)
    refs = _ref_candidates(desired)
    if not override:
        # If a release tag is temporarily inaccessible, current master is a
        # better final fallback than a permanent 404.
        if "master" not in refs:
            refs.append("master")

    errors: list[str] = []
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    for ref in refs:
        ok, detail = _clone_source(ref, verbose)
        if ok:
            return SOURCE_DIR
        errors.append(f"git {ref}: {detail}")

        ok, detail = _download_archive_source(ref, verbose)
        if ok:
            return SOURCE_DIR
        errors.append(f"zip {ref}: {detail}")

    summary = "\n".join(errors[-10:])
    raise RuntimeError(
        "NDLOCR-Lite 自动下载失败。程序已尝试最新正式版的真实 Git 标签、"
        "兼容的 v/无v 标签、GitHub压缩包和默认分支。\n"
        "请检查网络、GitHub访问权限或代理设置。\n\n"
        f"尝试记录：\n{summary}"
    )


def setup_venv(verbose: bool = True) -> tuple[Path, Path]:
    source = _download_source(verbose=verbose)
    requirements = source / "requirements.txt"
    python = ensure_venv(
        VENV_DIR,
        label="NDLOCR-Lite",
        marker_code="import onnxruntime, cv2, numpy, PIL, yaml",
        requirements=requirements,
        verbose=verbose,
    )
    return python, source


class NDLOcrLiteSession:
    """Keep the official NDLOCR ONNX models alive across OCR phases.

    The worker now has an explicit ready handshake and can restart itself once
    when a previous request/cancellation closed the process.  A failed optional
    NDLOCR model therefore returns per-image errors instead of crashing the
    complete multi-model OCR run with an ambiguous ``未启动`` exception.
    """

    STARTUP_TIMEOUT_SECONDS = 300.0

    def __init__(self, *, cancel_check=None, verbose: bool = True, engine_options: dict | None = None):
        self.cancel_check = cancel_check
        self.verbose = verbose
        self.engine_options = dict(engine_options or {})
        self.process = None
        self._stderr_file = None
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._reader = None
        self._lock = threading.RLock()
        self._ready = False
        self.provider_info: dict = {}

    def _stderr_text(self) -> str:
        handle = self._stderr_file
        if handle is None:
            return ""
        try:
            handle.flush()
            handle.seek(0)
            return handle.read().strip()
        except Exception:
            return ""

    def _start_process(self) -> None:
        # Do not reuse a queue containing the terminal ``None`` marker from an
        # earlier worker.  That stale marker was one cause of immediate failures
        # after an interrupted multi-model pass.
        self.close()
        self._queue = queue.Queue()
        self._ready = False
        python, source = setup_venv(verbose=self.verbose)
        self._stderr_file = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
        from adapters.subprocess_watchdog import isolated_process_kwargs
        self.process = subprocess.Popen(
            [str(python), str(WORKER_SCRIPT), "--source-root", str(source), "--server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=_ndlocr_worker_env(self.engine_options),
            **isolated_process_kwargs(),
        )
        process = self.process
        output_queue = self._queue

        def read_stdout():
            try:
                if process is None or process.stdout is None:
                    return
                for line in process.stdout:
                    output_queue.put(line)
            finally:
                # Keep the terminal marker tied to the worker that created this
                # queue. A late reader thread from a closed worker must never
                # poison the replacement worker's fresh queue.
                output_queue.put(None)

        self._reader = threading.Thread(
            target=read_stdout,
            name="ndlocr-jsonl-reader",
            daemon=True,
        )
        self._reader.start()

    def _wait_until_ready(self) -> None:
        from adapters.subprocess_watchdog import env_seconds
        startup_timeout = env_seconds(
            "NOVEL_FORMATTER_OCR_STARTUP_TIMEOUT",
            max(self.STARTUP_TIMEOUT_SECONDS, 900.0),
            minimum=60.0,
        )
        deadline = time.monotonic() + startup_timeout
        while True:
            if self.cancel_check is not None and self.cancel_check():
                self.close()
                raise RuntimeError("NDLOCR-Lite 模型加载已取消")
            process = self.process
            if process is None:
                raise RuntimeError("NDLOCR-Lite 常驻识别进程未创建")
            if process.poll() is not None and self._queue.empty():
                detail = self._stderr_text()
                raise RuntimeError(
                    "NDLOCR-Lite 模型加载失败："
                    + (detail or f"worker 退出码 {process.returncode}")
                )
            if time.monotonic() >= deadline:
                detail = self._stderr_text()
                self.close()
                raise RuntimeError(
                    f"NDLOCR-Lite 模型加载超过 {max(1, int(startup_timeout // 60))} 分钟，已停止。"
                    + (f"\n{detail}" if detail else "")
                )
            try:
                line = self._queue.get(timeout=0.20)
            except queue.Empty:
                continue
            if line is None:
                detail = self._stderr_text()
                raise RuntimeError(
                    "NDLOCR-Lite 模型加载进程提前结束："
                    + (detail or str(process.returncode))
                )
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if payload.get("type") == "ready" and payload.get("ok", True):
                self._ready = True
                self.provider_info = dict(payload.get("provider") or {})
                if self.verbose and self.provider_info:
                    effective = str(self.provider_info.get("effective") or "cpu")
                    print(f"🚀  NDLOCR-Lite ONNX provider：{effective}")
                return
            # Startup errors are emitted as an ordinary error packet with an
            # empty path before the worker exits.
            if payload.get("ok") is False and not str(payload.get("path") or ""):
                raise RuntimeError(
                    "NDLOCR-Lite 模型加载失败："
                    + str(payload.get("error") or "未知错误")
                )

    def _ensure_started(self) -> None:
        process = self.process
        if process is not None and process.poll() is None and self._ready:
            return
        self._start_process()
        self._wait_until_ready()

    def __enter__(self):
        self._ensure_started()
        return self

    def iter_recognize(self, image_paths: list[str], *, on_wait=None):
        paths = [str(path) for path in image_paths]
        if not paths:
            return
        with self._lock:
            try:
                self._ensure_started()
            except Exception as exc:
                detail = str(exc) or self._stderr_text() or "常驻进程未启动"
                for path in paths:
                    yield path, None, detail
                return

            process = self.process
            if process is None or process.poll() is not None:
                detail = self._stderr_text() or "常驻进程未启动"
                for path in paths:
                    yield path, None, f"NDLOCR-Lite 常驻识别进程不可用：{detail}"
                return

            request_id = uuid.uuid4().hex
            packet = json.dumps(
                {"request_id": request_id, "images": paths},
                ensure_ascii=False,
            ) + "\n"
            try:
                assert process.stdin is not None
                process.stdin.write(packet)
                process.stdin.flush()
            except Exception as exc:
                detail = self._stderr_text()
                for path in paths:
                    yield path, None, f"NDLOCR-Lite 常驻进程写入失败：{detail or exc}"
                return

            from adapters.subprocess_watchdog import env_seconds
            request_timeout = env_seconds(
                "NOVEL_FORMATTER_OCR_REQUEST_TIMEOUT", 300.0, minimum=30.0
            )
            deadline = time.monotonic() + request_timeout
            next_heartbeat = time.monotonic() + 20.0
            returned: set[str] = set()
            while True:
                if self.cancel_check is not None and self.cancel_check():
                    self.close()
                    for path in paths:
                        if path not in returned:
                            yield path, None, "用户取消"
                    return
                if time.monotonic() >= deadline:
                    detail = self._stderr_text()
                    self.close()
                    timeout_message = (
                        f"NDLOCR-Lite 连续 {max(1, int(request_timeout))} 秒未返回结果，"
                        "已自动终止以避免永久卡死"
                        + (f"：{detail[-1200:]}" if detail else "")
                    )
                    for path in paths:
                        if path not in returned:
                            yield path, None, timeout_message
                    return
                if process.poll() is not None and self._queue.empty():
                    detail = self._stderr_text()
                    for path in paths:
                        if path not in returned:
                            yield path, None, f"NDLOCR-Lite 常驻进程已退出：{detail or process.returncode}"
                    return
                try:
                    line = self._queue.get(timeout=0.20)
                except queue.Empty:
                    now = time.monotonic()
                    if now >= next_heartbeat:
                        if callable(on_wait):
                            try:
                                on_wait(len(returned), len(paths))
                            except Exception:
                                pass
                        next_heartbeat = now + 20.0
                    continue
                if line is None:
                    detail = self._stderr_text()
                    for path in paths:
                        if path not in returned:
                            yield path, None, f"NDLOCR-Lite 常驻进程未返回完整结果：{detail or process.returncode}"
                    return
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if payload.get("type") == "ready":
                    # A restarted worker can race with the request write.  The
                    # ready marker is informational and not part of a request.
                    self._ready = True
                    continue
                if str(payload.get("request_id") or "") != request_id:
                    continue
                deadline = time.monotonic() + request_timeout
                next_heartbeat = time.monotonic() + 20.0
                if payload.get("type") in {"request_done", "batch_done"}:
                    break
                path = str(payload.get("path") or "")
                if not path:
                    continue
                returned.add(path)
                if payload.get("ok"):
                    yield path, payload.get("blocks") or [], None
                else:
                    yield path, None, str(payload.get("error") or "NDLOCR-Lite 识别失败")
            for path in paths:
                if path not in returned:
                    yield path, None, "NDLOCR-Lite 常驻进程未返回该图片"

    def close(self):
        process = self.process
        self.process = None
        self._ready = False
        if process is not None:
            try:
                if process.poll() is None and process.stdin is not None:
                    packet = json.dumps(
                        {"request_id": uuid.uuid4().hex, "command": "close"},
                        ensure_ascii=False,
                    ) + "\n"
                    process.stdin.write(packet)
                    process.stdin.flush()
            except Exception:
                pass
            from adapters.subprocess_watchdog import terminate_process
            terminate_process(process, grace=2.0)
        reader = self._reader
        self._reader = None
        if reader is not None and reader.is_alive():
            reader.join(timeout=1.0)
        if self._stderr_file is not None:
            try:
                self._stderr_file.close()
            except Exception:
                pass
            self._stderr_file = None

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

def _run_worker(image_paths: list[str], cancel_check=None, verbose: bool = True):
    python, source = setup_venv(verbose=verbose)
    cmd = [str(python), str(WORKER_SCRIPT), "--source-root", str(source), *image_paths]
    return iter_worker_jsonl(cmd, cancel_check=cancel_check, engine_label="NDLOCR-Lite")


def run(verbose: bool = True, **kwargs):
    def worker_fn(ocr_paths, cancel_check):
        return _run_worker(list(ocr_paths), cancel_check=cancel_check, verbose=verbose)

    doc = run_ocr_engine(
        worker_fn,
        source_engine="ndlocr_lite",
        verbose=verbose,
        **kwargs,
    )
    from adapters.ocr_runtime_catalog import mark_runtime_ready
    mark_runtime_ready("ndlocr_lite")
    return doc
