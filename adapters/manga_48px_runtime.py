#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bootstrap the official Manga Image Translator 48px AR OCR runtime.

The 48px checkpoint is large enough that a silent ``urlopen`` inside the OCR
worker looks exactly like a frozen model load.  This module therefore provides
an explicit, resumable preparation step with progress callbacks and two
*official* download origins carrying the same verified SHA-256:

* the upstream author's Hugging Face repository (preferred),
* the upstream GitHub beta-0.3 release (fallback).

The network source is fetched as an exact Git blob, not an unpinned ``main``
file, so a later upstream edit cannot silently make the verified checkpoint
unloadable.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Iterable

HF_REVISION = "3e29cd63a0ce7d1b4013b0a6e56da4cddaf4fe5b"
GITHUB_RELEASE_ROOT = (
    "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3"
)
HF_RESOLVE_ROOT = (
    "https://huggingface.co/zyddnys/manga-image-translator/resolve/"
    + HF_REVISION
)

DEFAULT_MODEL_URLS = (
    f"{HF_RESOLVE_ROOT}/ocr_ar_48px.ckpt?download=true",
    f"{GITHUB_RELEASE_ROOT}/ocr_ar_48px.ckpt",
)
DEFAULT_DICT_URLS = (
    f"{HF_RESOLVE_ROOT}/alphabet-all-v7.txt?download=true",
    f"{GITHUB_RELEASE_ROOT}/alphabet-all-v7.txt",
)

# Backward-compatible public constants used by older plugins/tests.
MODEL_URL = os.environ.get(
    "NOVEL_FORMATTER_MANGA_48PX_MODEL_URL", DEFAULT_MODEL_URLS[0]
)
DICT_URL = os.environ.get(
    "NOVEL_FORMATTER_MANGA_48PX_DICT_URL", DEFAULT_DICT_URLS[0]
)
SOURCE_URL = os.environ.get(
    "NOVEL_FORMATTER_MANGA_48PX_SOURCE_URL",
    "https://raw.githubusercontent.com/zyddnys/manga-image-translator/main/manga_translator/ocr/model_48px.py",
)
XPOS_URL = os.environ.get(
    "NOVEL_FORMATTER_MANGA_48PX_XPOS_URL",
    "https://raw.githubusercontent.com/zyddnys/manga-image-translator/main/manga_translator/ocr/xpos_relative_position.py",
)

# GitHub blob IDs of the exact source files used by the verified official
# checkpoint. The blob API remains stable even if the main branch changes.
UPSTREAM_MODEL_SOURCE_SHA = "8a410854407f258a1bf5a5027beda09785cdcdd5"
UPSTREAM_XPOS_SOURCE_SHA = "cf2d9a7cb219e6590afb23b6fce6261cca134b10"
SOURCE_API_URL = (
    "https://api.github.com/repos/zyddnys/manga-image-translator/git/blobs/"
    + UPSTREAM_MODEL_SOURCE_SHA
)
XPOS_API_URL = (
    "https://api.github.com/repos/zyddnys/manga-image-translator/git/blobs/"
    + UPSTREAM_XPOS_SOURCE_SHA
)

MODEL_SHA256 = "29daa46d080818bb4ab239a518a88338cbccff8f901bef8c9db191a7cb97671d"
DICT_SHA256 = "f5722368146aa0fbcc9f4726866e4efc3203318ebb66c811d8cbbe915576538a"
MODEL_SIZE = 204_290_192

ProgressCallback = Callable[[str, int, int, str], None]
CancelCheck = Callable[[], bool]


class DownloadCancelled(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha_bytes(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _git_blob_sha(path: Path) -> str:
    return _git_blob_sha_bytes(path.read_bytes())


def _emit(
    callback: ProgressCallback | None,
    stage: str,
    current: int,
    total: int,
    detail: str,
) -> None:
    if callback is None:
        return
    try:
        callback(stage, max(0, int(current)), max(0, int(total)), str(detail))
    except Exception:
        # UI progress must never make a verified download fail.
        pass


def _is_cancelled(cancel_check: CancelCheck | None) -> bool:
    try:
        return bool(cancel_check and cancel_check())
    except Exception:
        return False


def _request(url: str, *, offset: int = 0) -> urllib.request.Request:
    headers = {
        "User-Agent": "Novel-Formatter-48px-AR/1.3",
        "Accept": "application/octet-stream,application/vnd.github+json;q=0.9,*/*;q=0.8",
    }
    if offset > 0:
        headers["Range"] = f"bytes={offset}-"
    return urllib.request.Request(url, headers=headers)


def _configured_urls(env_name: str, defaults: Iterable[str]) -> list[str]:
    """Return custom URL(s) followed by verified official fallbacks.

    Multiple custom mirrors may be separated with a newline, comma or the OS
    path separator.  Keeping the official origins afterwards prevents a stale
    environment variable from permanently disabling model installation.
    """
    values: list[str] = []
    raw = str(os.environ.get(env_name, "") or "").strip()
    if raw:
        # URL schemes contain ':', so do not use POSIX os.pathsep as a divider.
        # Newline/comma/semicolon are unambiguous for one or more mirror URLs.
        for item in re.split(r"[\n,;]+", raw):
            item = item.strip()
            if item.startswith(("http://", "https://", "file://")):
                values.append(item)
    for url in defaults:
        if url not in values:
            values.append(url)
    return values


def _origin_label(url: str) -> str:
    lowered = url.lower()
    if "huggingface.co" in lowered:
        return "Hugging Face 官方仓库"
    if "github.com" in lowered or "githubusercontent.com" in lowered:
        return "GitHub 官方发布"
    if lowered.startswith("file://"):
        return "本地文件"
    return "自定义下载源"


def _response_total(response, offset: int) -> int:
    content_range = str(response.headers.get("Content-Range") or "")
    match = re.search(r"/(\d+)$", content_range)
    if match:
        return int(match.group(1))
    try:
        length = int(response.headers.get("Content-Length") or 0)
    except (TypeError, ValueError):
        length = 0
    status = int(getattr(response, "status", 200) or 200)
    return (offset + length) if status == 206 and length else length


def _download_one(
    url: str,
    part_path: Path,
    *,
    label: str,
    expected_size: int,
    progress_callback: ProgressCallback | None,
    cancel_check: CancelCheck | None,
) -> None:
    if _is_cancelled(cancel_check):
        raise DownloadCancelled("用户取消 48px 模型下载")

    offset = part_path.stat().st_size if part_path.exists() else 0
    if expected_size and offset > expected_size:
        part_path.unlink(missing_ok=True)
        offset = 0

    request = _request(url, offset=offset)
    with urllib.request.urlopen(request, timeout=45) as response:
        status = int(getattr(response, "status", 200) or 200)
        # A server that ignored Range returned the complete file. Restart rather
        # than appending a second copy to the partial download.
        if offset and status != 206:
            offset = 0
            mode = "wb"
        else:
            mode = "ab" if offset else "wb"
        total = _response_total(response, offset) or expected_size
        downloaded = offset
        origin = _origin_label(url)
        _emit(
            progress_callback,
            "download",
            downloaded,
            total,
            f"{label} · {origin} · 建立连接",
        )
        last_emit_time = 0.0
        last_emit_bytes = downloaded
        with part_path.open(mode) as output:
            while True:
                if _is_cancelled(cancel_check):
                    raise DownloadCancelled("用户取消 48px 模型下载")
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_emit_time >= 0.25 or downloaded - last_emit_bytes >= 4 * 1024 * 1024:
                    _emit(
                        progress_callback,
                        "download",
                        downloaded,
                        total,
                        f"{label} · {origin}",
                    )
                    last_emit_time = now
                    last_emit_bytes = downloaded
        _emit(
            progress_callback,
            "download",
            downloaded,
            total,
            f"{label} · {origin} · 下载完成",
        )





def _download_one_curl(
    url: str,
    part_path: Path,
    *,
    label: str,
    expected_size: int,
    progress_callback: ProgressCallback | None,
    cancel_check: CancelCheck | None,
) -> None:
    """Download through macOS/system curl when Python TLS cannot connect."""
    curl = shutil.which("curl")
    if not curl:
        raise RuntimeError("系统未找到 curl")
    origin = _origin_label(url)
    command = [
        curl,
        "--location",
        "--fail",
        "--silent",
        "--show-error",
        "--retry",
        "3",
        "--retry-delay",
        "1",
        "--connect-timeout",
        "20",
        "--max-time",
        "1800",
    ]
    if part_path.exists() and part_path.stat().st_size > 0:
        command.extend(["--continue-at", "-"])
    command.extend(["--output", str(part_path), url])
    from adapters.subprocess_watchdog import isolated_process_kwargs, terminate_process

    # curl can produce repeated TLS/retry diagnostics.  Keep them in a file so
    # an unread stderr pipe can never block a long model download.
    stderr_file = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
    proc = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=stderr_file,
        text=True,
        **isolated_process_kwargs(),
    )
    try:
        while proc.poll() is None:
            if _is_cancelled(cancel_check):
                terminate_process(proc)
                raise DownloadCancelled("用户取消 48px 模型下载")
            current = part_path.stat().st_size if part_path.exists() else 0
            _emit(
                progress_callback,
                "download",
                current,
                expected_size,
                f"{label} · {origin} · 系统 curl",
            )
            time.sleep(0.25)
        stderr = ""
        try:
            stderr_file.flush()
            stderr_file.seek(0)
            stderr = stderr_file.read().strip()
        except Exception:
            pass
        if proc.returncode != 0:
            raise RuntimeError(stderr[-1200:] or f"curl 退出码 {proc.returncode}")
        current = part_path.stat().st_size if part_path.exists() else 0
        _emit(
            progress_callback,
            "download",
            current,
            expected_size,
            f"{label} · {origin} · 系统 curl 下载完成",
        )
    finally:
        if proc.poll() is None:
            terminate_process(proc)
        stderr_file.close()

def _seed_from_local(
    destination: Path,
    *,
    env_name: str,
    sha256: str,
    minimum_size: int,
    expected_size: int,
    label: str,
    progress_callback: ProgressCallback | None,
) -> bool:
    """Import a manually downloaded official file after cryptographic checks."""
    project_root = destination.parent.parent.parent
    candidates: list[Path] = []
    configured = str(os.environ.get(env_name, "") or "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            project_root / destination.name,
            project_root / "models" / destination.name,
            Path.home() / "Downloads" / destination.name,
            Path.home() / "Desktop" / destination.name,
        ]
    )
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        try:
            if not candidate.is_file() or candidate.resolve() == destination.resolve():
                continue
            size = candidate.stat().st_size
            if size < minimum_size or (expected_size and size != expected_size):
                continue
            _emit(progress_callback, "verify", 0, 1, f"发现本地 {label} · 正在校验")
            if sha256 and _sha256(candidate) != sha256:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temp = destination.with_name(destination.name + ".importing")
            shutil.copy2(candidate, temp)
            os.replace(temp, destination)
            _emit(progress_callback, "verify", 1, 1, f"已导入本地 {label}")
            return True
        except OSError:
            continue
    return False

def _download(
    urls: str | Iterable[str],
    destination: Path,
    *,
    sha256: str = "",
    minimum_size: int = 1,
    expected_size: int = 0,
    label: str = "文件",
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size >= minimum_size:
        _emit(
            progress_callback,
            "verify",
            0,
            1,
            f"正在校验已存在的 {label}",
        )
        if not sha256 or _sha256(destination) == sha256:
            _emit(progress_callback, "verify", 1, 1, f"{label} 已完整下载并通过校验")
            return destination
        destination.unlink(missing_ok=True)

    local_env = (
        "NOVEL_FORMATTER_MANGA_48PX_MODEL_FILE"
        if destination.name == "ocr_ar_48px.ckpt"
        else "NOVEL_FORMATTER_MANGA_48PX_DICT_FILE"
    )
    if _seed_from_local(
        destination,
        env_name=local_env,
        sha256=sha256,
        minimum_size=minimum_size,
        expected_size=expected_size,
        label=label,
        progress_callback=progress_callback,
    ):
        return destination

    url_list = [urls] if isinstance(urls, str) else [str(item) for item in urls]
    url_list = [item for item in url_list if item]
    if not url_list:
        raise RuntimeError(f"{label} 没有可用下载地址")

    part_path = destination.with_name(destination.name + ".part")
    errors: list[str] = []
    for url in url_list:
        transports = [("Python HTTPS", _download_one)]
        if shutil.which("curl"):
            transports.append(("系统 curl", _download_one_curl))
        for transport_name, transport in transports:
            try:
                _emit(
                    progress_callback,
                    "connect",
                    part_path.stat().st_size if part_path.exists() else 0,
                    expected_size,
                    f"准备下载 {label} · {_origin_label(url)} · {transport_name}",
                )
                transport(
                    url,
                    part_path,
                    label=label,
                    expected_size=expected_size,
                    progress_callback=progress_callback,
                    cancel_check=cancel_check,
                )
                if part_path.stat().st_size < minimum_size:
                    raise RuntimeError(
                        f"下载文件过小：{part_path.stat().st_size} bytes"
                    )
                if expected_size and part_path.stat().st_size != expected_size:
                    raise RuntimeError(
                        f"下载大小不完整：{part_path.stat().st_size}/{expected_size} bytes"
                    )
                _emit(progress_callback, "verify", 0, 1, f"正在校验 {label} SHA-256")
                if sha256 and _sha256(part_path) != sha256:
                    # A checksum mismatch is not resumable; start clean on the
                    # next official origin.
                    part_path.unlink(missing_ok=True)
                    raise RuntimeError("SHA-256 校验失败")
                os.replace(part_path, destination)
                _emit(progress_callback, "verify", 1, 1, f"{label} 校验完成")
                return destination
            except DownloadCancelled:
                raise
            except Exception as exc:
                errors.append(f"{_origin_label(url)}·{transport_name}: {exc}")
                _emit(
                    progress_callback,
                    "retry",
                    part_path.stat().st_size if part_path.exists() else 0,
                    expected_size,
                    f"{label} 下载失败，正在切换下载通道：{exc}",
                )
                time.sleep(0.35)

    attempted = "\n".join(f"- {item}" for item in errors[-8:])
    raise RuntimeError(
        f"{label} 无法自动下载。已保留断点文件：{part_path}\n"
        f"可手动把官方文件放到：{destination}\n"
        f"下载尝试：\n{attempted}"
    )




def _curl_read_bytes(url: str, *, timeout: int = 120) -> bytes:
    curl = shutil.which("curl")
    if not curl:
        raise RuntimeError("系统未找到 curl")
    proc = subprocess.run(
        [
            curl,
            "--location",
            "--fail",
            "--silent",
            "--show-error",
            "--retry",
            "2",
            "--connect-timeout",
            "20",
            "--max-time",
            str(timeout),
            url,
        ],
        capture_output=True,
        timeout=timeout + 10,
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(detail[-1200:] or f"curl 退出码 {proc.returncode}")
    return bytes(proc.stdout)

def _write_verified_blob(
    data: bytes,
    destination: Path,
    *,
    expected_blob_sha: str,
    minimum_size: int,
) -> Path:
    if len(data) < minimum_size:
        raise RuntimeError(f"官方源码文件过小：{destination.name}")
    actual = _git_blob_sha_bytes(data)
    if actual != expected_blob_sha:
        raise RuntimeError(
            f"官方源码版本校验失败：{destination.name}，"
            f"expected={expected_blob_sha}, actual={actual}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".source", dir=destination.parent
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        temp_path.write_bytes(data)
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)
    return destination


def _download_github_text(
    raw_url: str,
    blob_api_url: str,
    destination: Path,
    *,
    expected_blob_sha: str,
    minimum_size: int,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> Path:
    """Fetch an exact GitHub source blob with raw/API fallback."""
    if destination.exists() and destination.stat().st_size >= minimum_size:
        try:
            if _git_blob_sha(destination) == expected_blob_sha:
                return destination
        except Exception:
            pass
        destination.unlink(missing_ok=True)

    if _is_cancelled(cancel_check):
        raise DownloadCancelled("用户取消 48px 模型准备")
    _emit(progress_callback, "source", 0, 1, f"获取固定版本源码：{destination.name}")
    errors: list[str] = []
    for transport_name, fetcher in (
        ("Python HTTPS", lambda url: urllib.request.urlopen(_request(url), timeout=45).read()),
        ("系统 curl", _curl_read_bytes),
    ):
        if transport_name == "系统 curl" and not shutil.which("curl"):
            continue
        try:
            if _is_cancelled(cancel_check):
                raise DownloadCancelled("用户取消 48px 模型准备")
            data = fetcher(raw_url)
            result = _write_verified_blob(
                data,
                destination,
                expected_blob_sha=expected_blob_sha,
                minimum_size=minimum_size,
            )
            _emit(progress_callback, "source", 1, 1, f"固定版本源码已就绪：{destination.name}")
            return result
        except DownloadCancelled:
            raise
        except Exception as exc:
            errors.append(f"raw·{transport_name}: {exc}")

    for transport_name, fetcher in (
        ("Python HTTPS", lambda url: urllib.request.urlopen(_request(url), timeout=45).read()),
        ("系统 curl", _curl_read_bytes),
    ):
        if transport_name == "系统 curl" and not shutil.which("curl"):
            continue
        try:
            if _is_cancelled(cancel_check):
                raise DownloadCancelled("用户取消 48px 模型准备")
            raw_payload = fetcher(blob_api_url)
            payload = json.loads(raw_payload.decode("utf-8"))
            if str(payload.get("sha") or "") != expected_blob_sha:
                raise RuntimeError("GitHub blob API 返回了错误版本")
            encoded = str(payload.get("content") or "").replace("\n", "")
            data = base64.b64decode(encoded, validate=True)
            result = _write_verified_blob(
                data,
                destination,
                expected_blob_sha=expected_blob_sha,
                minimum_size=minimum_size,
            )
            _emit(progress_callback, "source", 1, 1, f"固定版本源码已就绪：{destination.name}")
            return result
        except DownloadCancelled:
            raise
        except Exception as exc:
            errors.append(f"blob_api·{transport_name}: {exc}")

    raise RuntimeError(
        "48px OCR 权重已经下载，但匹配该权重的官方网络源码无法取得。\n"
        + "\n".join(f"- {item}" for item in errors[-8:])
    )


def extract_model_core(source: str) -> str:
    """Extract the dependency-light network/beam-search section upstream."""
    start_marker = "class ConvNeXtBlock"
    end_marker = "\ndef convert_pl_model"
    start = source.find(start_marker)
    end = source.find(end_marker, start)
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError("上游 48px OCR 源码结构发生变化，无法提取模型核心")
    core = source[start:end].rstrip() + "\n"
    header = '''# Generated from zyddnys/manga-image-translator model_48px.py.
# Exact upstream Git blob: %s
import math
from typing import Callable, List, Optional, Tuple, Union
from collections import defaultdict
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from manga_48px_xpos import XPOS

''' % UPSTREAM_MODEL_SOURCE_SHA
    return header + core


def ensure_runtime_files(
    cache_dir: Path,
    *,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> tuple[Path, Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_urls = _configured_urls(
        "NOVEL_FORMATTER_MANGA_48PX_MODEL_URL", DEFAULT_MODEL_URLS
    )
    dict_urls = _configured_urls(
        "NOVEL_FORMATTER_MANGA_48PX_DICT_URL", DEFAULT_DICT_URLS
    )
    model_path = _download(
        model_urls,
        cache_dir / "ocr_ar_48px.ckpt",
        sha256=MODEL_SHA256,
        minimum_size=100_000_000,
        expected_size=MODEL_SIZE,
        label="48px AR 官方权重（204 MB）",
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )
    dict_path = _download(
        dict_urls,
        cache_dir / "alphabet-all-v7.txt",
        sha256=DICT_SHA256,
        minimum_size=1_000,
        label="48px AR 字符表",
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )

    source_dir = cache_dir / "upstream-source"
    raw_model = _download_github_text(
        SOURCE_URL,
        SOURCE_API_URL,
        source_dir / "model_48px.py",
        expected_blob_sha=UPSTREAM_MODEL_SOURCE_SHA,
        minimum_size=25_000,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )
    raw_xpos = _download_github_text(
        XPOS_URL,
        XPOS_API_URL,
        source_dir / "xpos_relative_position.py",
        expected_blob_sha=UPSTREAM_XPOS_SOURCE_SHA,
        minimum_size=2_000,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )

    generated = source_dir / "manga_48px_core.py"
    stamp = source_dir / ".generated-source-blob"
    source_blob = _git_blob_sha(raw_model)
    if (
        not generated.exists()
        or not stamp.exists()
        or stamp.read_text(encoding="utf-8").strip() != source_blob
    ):
        generated.write_text(
            extract_model_core(raw_model.read_text(encoding="utf-8")),
            encoding="utf-8",
        )
        stamp.write_text(source_blob, encoding="utf-8")

    xpos_target = source_dir / "manga_48px_xpos.py"
    if (
        not xpos_target.exists()
        or _git_blob_sha(xpos_target) != UPSTREAM_XPOS_SOURCE_SHA
    ):
        xpos_target.write_bytes(raw_xpos.read_bytes())
    _emit(progress_callback, "ready", 1, 1, "48px AR 模型文件全部就绪")
    return model_path, dict_path, generated


def load_ocr_class(cache_dir: Path):
    model_path, dict_path, generated = ensure_runtime_files(cache_dir)
    source_dir = generated.parent
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    module_name = "novel_formatter_manga_48px_core"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, generated)
        if spec is None or spec.loader is None:
            raise RuntimeError("无法加载 48px OCR 模型核心")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    if not hasattr(module, "OCR"):
        raise RuntimeError("48px OCR 模型核心中缺少 OCR 类")
    return module.OCR, model_path, dict_path


def _console_progress(stage: str, current: int, total: int, detail: str) -> None:
    if total > 0:
        percent = min(100.0, max(0.0, current * 100.0 / total))
        current_mb = current / (1024 * 1024)
        total_mb = total / (1024 * 1024)
        print(
            f"\r[{stage}] {percent:6.2f}%  {current_mb:7.1f}/{total_mb:7.1f} MB  {detail:<48}",
            end="",
            flush=True,
        )
        if current >= total:
            print()
    else:
        print(f"[{stage}] {detail}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="准备 Manga 48px AR OCR 模型")
    parser.add_argument("--cache-dir", required=True)
    args = parser.parse_args()
    try:
        model, dictionary, source = ensure_runtime_files(
            Path(args.cache_dir).expanduser().resolve(),
            progress_callback=_console_progress,
        )
    except Exception as exc:
        print(f"\n准备失败：{exc}", file=sys.stderr)
        raise SystemExit(1)
    print("\n48px AR OCR 已准备完成：")
    print(f"- 权重：{model} ({model.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"- 字符表：{dictionary}")
    print(f"- 固定模型源码：{source}")


if __name__ == "__main__":
    main()
