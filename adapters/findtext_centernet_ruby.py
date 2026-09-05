#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional findtextCenterNet Ruby (furigana) preservation pass.

This module is deliberately *not* an OCR replacement.  The normal OCR engine
continues to own prose recognition; findtextCenterNet is only invoked when the
user explicitly enables Ruby preservation.  Its structured ``aozora``/``noruby``
output is then merged back into already-recognised blocks without changing the
plain prose characters.

First use is isolated in ``.venv-findtext-centernet`` and
``.ocr-runtimes/findtext-centernet``.  No source/model download occurs unless
``prepare_runtime``/``preserve_ruby_in_documents`` is called.
"""
from __future__ import annotations

import copy
import hashlib
import http.client
import json
import os
import queue
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import tarfile
import time
import unicodedata
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image

from adapters.runtime_env import ensure_venv
from adapters.ruby_roi_planner import build_ruby_roi_plans, estimate_findtext_tiles
from adapters.ruby_result_cache import (
    RubyResultCache, file_sha256, make_cache_key, runtime_fingerprint,
)
from models.document import BlockType, UnifiedDocument
from engine.ruby_anchor_core import resolve_annotation_for_text, resolve_exact

ROOT = Path(__file__).resolve().parent.parent
RUNTIME_ROOT = ROOT / ".ocr-runtimes" / "findtext-centernet"
DEFAULT_SOURCE_DIR = RUNTIME_ROOT / "src"
DEFAULT_VENV_DIR = ROOT / ".venv-findtext-centernet"
WORKER_SCRIPT = Path(__file__).parent / "findtext_centernet_worker.py"
UPSTREAM_COMMIT = "295bc88703039f9a83eef8146c743c94cf728b96"
UPSTREAM_REPO_URL = "https://github.com/lithium0003/findtextCenterNet.git"
# Prefer codeload directly so a transient github.com -> codeload redirect cannot
# lose a response half way through. Keep the normal archive URL as an official
# fallback. Downloads are resumable and validated before extraction.
SOURCE_ARCHIVE_URLS = (
    f"https://codeload.github.com/lithium0003/findtextCenterNet/zip/{UPSTREAM_COMMIT}",
    f"https://github.com/lithium0003/findtextCenterNet/archive/{UPSTREAM_COMMIT}.zip",
)
SOURCE_ARCHIVE_URL = SOURCE_ARCHIVE_URLS[0]  # backward-compatible diagnostic name
# Git blob SHA-1s of inference-critical files at the pinned source commit.
# This verifies that a syntactically valid ZIP is also the exact upstream code
# we reviewed, without relying on a generated archive's byte-for-byte hash.
SOURCE_BLOB_SHA1 = {
    "run_ocr.py": "3f2fed9134532b8fc528c6233498f92c12821320",
    "process_ocr_base.py": "08cd7851f155d19b1614357dbf3168983dd6328c",
    "process_ocr_torch.py": "27bb0f01ab71d5a9e893f558d1d850c167e7648b",
    "models/detector.py": "bd8f3d08353157c94bf19aafc6b988cd394183e3",
    "const.py": "5a50214669f7516dff9117a21bfeba83a7e395e7",
    "util_func.py": "30fa12ddc8d900902ca57032cbb0ac6005e1350d",
}

# Pinned v3 weights. Exact LFS size/hash prevent a >1 MB partial file from being
# mistaken for a valid model after a broken connection.
UPSTREAM_WEIGHT_REVISION = "b426acc08a5976ec804084eefb1c987709f2e34d"
MODEL_SPECS = {
    "model.pt": {
        "size": 1053713502,
        "sha256": "fe71aaee3ee9dc2b0364021ac4907d997ea3b43c7a8df996788c958b039752e9",
        "urls": (
            f"https://huggingface.co/lithium0003/findtextCenterNet/resolve/{UPSTREAM_WEIGHT_REVISION}/model.pt?download=true",
            "https://huggingface.co/datasets/lithium0003/findtextCenterNet_dataset/resolve/main/model.pt?download=true",
        ),
    },
    "model3.pt": {
        "size": 437420605,
        "sha256": "16cbe15e60c44edf48090046723ed942665a6bc243593a0f833682d0c195e062",
        "urls": (
            f"https://huggingface.co/lithium0003/findtextCenterNet/resolve/{UPSTREAM_WEIGHT_REVISION}/model3.pt?download=true",
            "https://huggingface.co/datasets/lithium0003/findtextCenterNet_dataset/resolve/main/model3.pt?download=true",
        ),
    },
}
MODEL_URLS = {name: spec["urls"][0] for name, spec in MODEL_SPECS.items()}

# Upstream run_ocr.py selects CoreML -> ONNX -> Torch solely from the files
# present in its own project root.  Match that contract instead of forcing the
# 1.49 GB Torch checkpoint path on every platform.
COREML_SPECS = {
    "TextDetector.mlpackage.tar.gz": {
        # Hugging Face currently serves the pinned artifact at 450,029,932
        # bytes; keep the live object metadata aligned so a valid download is
        # not rejected as the obsolete 44-byte-short variant.
        "size": 450_029_932,
        "sha256": "bf172ec0b78f69e9becbdcde630486254acadad48e523171eaca53c804b274a2",
        "output": "TextDetector.mlpackage",
    },
    "TransformerEncoder.mlpackage.tar.gz": {
        "size": 79_177_068,
        # Current Hugging Face object hash for the pinned encoder archive.
        "sha256": "eefc3c4a06b8a86e49edfdb77c9c9e850d3fbca13ec7fc980a564de6b6c55d1b",
        "output": "TransformerEncoder.mlpackage",
    },
    "TransformerDecoder.mlpackage.tar.gz": {
        # The pinned Hugging Face object currently resolves to this 18-byte
        # shorter revision of the decoder archive.
        "size": 119_118_317,
        "sha256": "65e4408c3b1ad1575b96ab98e933b13c108f5b12402bf6def0b3fc02c9227440",
        "output": "TransformerDecoder.mlpackage",
    },
}
# The pinned upstream source keeps a 400-token transformer window, while the
# official CoreML packages currently published for findtextCenterNet declare a
# 100-token encoder/decoder window.  Keep this value scoped to CoreML; ONNX and
# Torch continue to use the upstream constants unchanged.
COREML_TRANSFORMER_LENGTH = 100
ONNX_SPECS = {
    "TextDetector.quant.onnx": {
        "size": 246_826_507,
        "sha256": "f24daa7cd2280fd4db8b1ba387c7f777bdef805c166a489733690cac283eb3ac",
    },
    "TransformerEncoder.onnx": {
        "size": 175_284_069,
        "sha256": "7a78c1bd6b721c4919bf016a4a83c158f2639fd0d1eee929b93b92d8f92de27c",
    },
    "TransformerDecoder.onnx": {
        "size": 264_681_314,
        "sha256": "5c07594e1caad88a5934361610e9a34cc55ee558e941f8feb8559fc61be8bb0f",
    },
}
HF_MODEL_REPO = "lithium0003/findtextCenterNet"
CACHE_RUNTIME_ID = hashlib.sha256(
    (UPSTREAM_COMMIT + "|" + "|".join(SOURCE_ARCHIVE_URLS) + "|" + "|".join(
        f"{name}={spec['sha256']}:{spec['size']}" for name, spec in sorted(MODEL_SPECS.items())
    )).encode("utf-8")
).hexdigest()[:24]
DEFAULT_CACHE_DIR = RUNTIME_ROOT / "cache-v1"
_RUNTIME_PREPARE_LOCK = threading.RLock()
RUBY_RE = re.compile(r"[｜|]([^《\n]+)《([^》\n]+)》")
_TEXT_TYPES = {
    BlockType.PARAGRAPH,
    BlockType.DIALOGUE,
    BlockType.CHAPTER,
    BlockType.SECTION,
    BlockType.RUBY,
}


@dataclass(frozen=True)
class RubyLine:
    page: int
    aozora: str
    plain: str
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    # Optional per-Ruby-base boxes in original-page pixel coordinates.  The
    # tuple order follows ``pairs``.  Older/upstream payloads without character
    # boxes simply leave this empty; text matching still works as before.
    pair_boxes: tuple[tuple[float, float, float, float], ...] = ()

    @property
    def pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple((m.group(1), m.group(2)) for m in RUBY_RE.finditer(self.aozora))


@dataclass(frozen=True)
class RubyPreservationReport:
    enabled: bool
    pages_scanned: int = 0
    ruby_lines: int = 0
    ruby_pairs: int = 0
    matched_lines: int = 0
    matched_pairs: int = 0
    updated_blocks: int = 0
    unmatched_pairs: int = 0
    scan_mode: str = ""
    pages_with_candidates: int = 0
    candidate_boxes: int = 0
    roi_count: int = 0
    roi_coverage_ratio: float = 0.0
    estimated_detector_tiles: int = 0
    full_page_detector_tiles: int = 0
    estimated_tile_ratio: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    failed_rois: int = 0
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "pages_scanned": self.pages_scanned,
            "ruby_lines": self.ruby_lines,
            "ruby_pairs": self.ruby_pairs,
            "matched_lines": self.matched_lines,
            "matched_pairs": self.matched_pairs,
            "updated_blocks": self.updated_blocks,
            "unmatched_pairs": self.unmatched_pairs,
            "scan_mode": self.scan_mode,
            "pages_with_candidates": self.pages_with_candidates,
            "candidate_boxes": self.candidate_boxes,
            "roi_count": self.roi_count,
            "roi_coverage_ratio": round(float(self.roi_coverage_ratio), 6),
            "estimated_detector_tiles": self.estimated_detector_tiles,
            "full_page_detector_tiles": self.full_page_detector_tiles,
            "estimated_tile_ratio": round(float(self.estimated_tile_ratio), 6),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "failed_rois": self.failed_rois,
            "error": self.error,
        }


def _emit(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        try:
            callback(str(message))
        except Exception:
            pass


def _cache_dir() -> Path:
    override = os.environ.get("NOVEL_FORMATTER_FINDTEXT_CACHE_DIR", "").strip()
    return Path(override).expanduser().resolve() if override else DEFAULT_CACHE_DIR


def _venv_python(venv_dir: Path = DEFAULT_VENV_DIR) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _linedetect_path(source_dir: Path) -> Path:
    return source_dir / "textline_detect" / ("linedetect.exe" if os.name == "nt" else "linedetect")


def _source_dir() -> Path:
    override = os.environ.get("NOVEL_FORMATTER_FINDTEXT_CENTERNET_DIR", "").strip()
    return Path(override).expanduser().resolve() if override else DEFAULT_SOURCE_DIR


def _runtime_python(source_dir: Path) -> Path:
    override = os.environ.get("NOVEL_FORMATTER_FINDTEXT_CENTERNET_PYTHON", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    # External source overrides may intentionally share the current Python.
    if source_dir != DEFAULT_SOURCE_DIR and os.environ.get(
        "NOVEL_FORMATTER_FINDTEXT_CENTERNET_USE_CURRENT_PYTHON", ""
    ).strip().lower() in {"1", "true", "yes", "on"}:
        return Path(sys.executable)
    return _venv_python()


def _source_tree_ready(source: Path) -> bool:
    required = (
        source / "run_ocr.py",
        source / "process_ocr_base.py",
        source / "models" / "detector.py",
        source / "models" / "transformer.py",
        source / "textline_detect" / "Makefile",
    )
    return all(path.is_file() for path in required)


def runtime_ready(source_dir: Path | None = None) -> tuple[bool, str]:
    source = Path(source_dir or _source_dir())
    python = _runtime_python(source)
    if not _source_tree_ready(source):
        return False, "findtextCenterNet 源码未准备完整"
    backend = _ready_backend(source)
    if not backend:
        return False, "缺少完整的 CoreML / ONNX / Torch 任一上游推理后端"
    if not _linedetect_path(source).exists():
        return False, f"缺少 {_linedetect_path(source).name}"
    if not python.exists():
        return False, f"缺少 Python 运行环境：{python}"
    return True, f"findtextCenterNet Ruby 运行时已就绪（{backend.upper()}）"


def _download_request(url: str, *, offset: int = 0) -> urllib.request.Request:
    headers = {
        "User-Agent": "NovelFormatter/FindtextRuby",
        "Accept": "application/octet-stream,*/*;q=0.8",
        "Accept-Encoding": "identity",
    }
    if offset > 0:
        headers["Range"] = f"bytes={offset}-"
    return urllib.request.Request(url, headers=headers)


def _response_total(response, offset: int) -> int:
    content_range = str(response.headers.get("Content-Range", "") or "")
    match = re.search(r"/(\d+)\s*$", content_range)
    if match:
        return int(match.group(1))
    value = str(response.headers.get("Content-Length", "") or "").strip()
    if value.isdigit():
        length = int(value)
        return offset + length if int(getattr(response, "status", 200) or 200) == 206 else length
    return 0


def _human_bytes(value: int) -> str:
    amount = float(max(0, int(value)))
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    idx = 0
    while amount >= 1024.0 and idx < len(units) - 1:
        amount /= 1024.0
        idx += 1
    return f"{amount:.1f} {units[idx]}" if idx else f"{int(amount)} {units[idx]}"


def _download_progress_line(label: str, downloaded: int, total: int, *, elapsed: float, offset: int) -> str:
    session_bytes = max(0, int(downloaded) - int(offset))
    speed = session_bytes / max(0.001, float(elapsed))
    if total > 0:
        pct = max(0.0, min(100.0, downloaded * 100.0 / total))
        remaining = max(0, total - downloaded)
        eta = remaining / speed if speed > 0 else 0.0
        eta_text = f"，预计剩余 {int(eta // 60)}分{int(eta % 60):02d}秒" if eta > 1 else ""
        return (
            f"⬇️ {label}：{_human_bytes(downloaded)} / {_human_bytes(total)} "
            f"({pct:.1f}%) · {_human_bytes(int(speed))}/s{eta_text}"
        )
    return f"⬇️ {label}：已下载 {_human_bytes(downloaded)} · {_human_bytes(int(speed))}/s"


def _download_python_once(
    url: str, partial: Path, *, label: str, expected_size: int = 0, log_callback=None,
) -> None:
    offset = partial.stat().st_size if partial.exists() else 0
    if expected_size and offset >= expected_size:
        # Do not truncate/delete a complete-ish artifact here.  The caller's
        # size+SHA validator may be able to recover a tiny transport tail.
        return
    req = _download_request(url, offset=offset)
    socket_timeout = max(15.0, min(75.0, float(os.environ.get("NOVEL_FORMATTER_FINDTEXT_HTTP_TIMEOUT", "45"))))
    _emit(log_callback, f"🌐 {label}：正在建立下载连接…")
    with urllib.request.urlopen(req, timeout=socket_timeout) as response:
        status = int(getattr(response, "status", 200) or 200)
        # Some CDNs ignore Range. Never append a complete response to a .part.
        if offset and status != 206:
            offset = 0
            mode = "wb"
        else:
            mode = "ab" if offset else "wb"
        total = _response_total(response, offset) or expected_size
        downloaded = offset
        started = time.monotonic()
        last_report = started
        _emit(log_callback, _download_progress_line(label, downloaded, total, elapsed=0.001, offset=offset))
        with partial.open(mode) as out:
            while True:
                try:
                    chunk = response.read(256 * 1024)
                except http.client.IncompleteRead as exc:
                    if exc.partial:
                        out.write(exc.partial)
                        downloaded += len(exc.partial)
                    raise RuntimeError(
                        f"连接提前结束，已保存 {downloaded} bytes，稍后自动续传"
                    ) from exc
                except (TimeoutError, socket.timeout) as exc:
                    raise RuntimeError(
                        f"下载连接 {int(socket_timeout)} 秒无数据，已保存 {downloaded} bytes，稍后自动续传"
                    ) from exc
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_report >= 5.0:
                    _emit(
                        log_callback,
                        _download_progress_line(label, downloaded, total, elapsed=now - started, offset=offset),
                    )
                    last_report = now
        _emit(
            log_callback,
            _download_progress_line(
                label, downloaded, total, elapsed=max(0.001, time.monotonic() - started), offset=offset
            ),
        )
        if total and partial.stat().st_size != total:
            raise RuntimeError(
                f"下载未完整：{partial.stat().st_size}/{total} bytes（将自动续传）"
            )
        if expected_size and partial.stat().st_size < expected_size:
            raise RuntimeError(
                f"文件大小不完整：{partial.stat().st_size}/{expected_size} bytes（将自动续传）"
            )


def _download_curl_once(
    url: str, partial: Path, *, label: str, expected_size: int = 0, log_callback=None,
) -> None:
    curl = shutil.which("curl")
    if not curl:
        raise RuntimeError("系统未找到 curl")
    # Hugging Face/CDN connections on some macOS networks repeatedly fail with
    # curl error 92 (HTTP/2 stream CANCEL) after making substantial progress.
    # Force HTTP/1.1 for the resumable fallback: curl negotiates HTTP/2 by
    # default for HTTPS, while --http1.1 keeps the same Range semantics without
    # the flaky multiplexed stream on those routes.
    command = [
        curl, "--http1.1", "--location", "--fail", "--silent", "--show-error",
        "--connect-timeout", "20",
        "--max-time", "7200",
    ]
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > 0:
        command.extend(["--continue-at", "-"])
    command.extend(["--output", str(partial), url])
    started = time.monotonic()
    last_report = 0.0
    _emit(log_callback, f"🌐 {label}：系统 curl 已启动，正在接收数据…")
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        while process.poll() is None:
            now = time.monotonic()
            if now - last_report >= 5.0:
                current = partial.stat().st_size if partial.exists() else offset
                _emit(
                    log_callback,
                    _download_progress_line(
                        label, current, expected_size, elapsed=max(0.001, now - started), offset=offset
                    ),
                )
                last_report = now
            time.sleep(0.5)
        stdout, stderr = process.communicate(timeout=5)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass
        raise
    current = partial.stat().st_size if partial.exists() else 0
    _emit(
        log_callback,
        _download_progress_line(
            label, current, expected_size, elapsed=max(0.001, time.monotonic() - started), offset=offset
        ),
    )
    if process.returncode != 0:
        raise RuntimeError((stderr or stdout or f"curl 退出码 {process.returncode}")[-1200:])
    if expected_size and current < expected_size:
        raise RuntimeError(
            f"文件大小不完整：{current}/{expected_size} bytes（将自动续传）"
        )
    if expected_size and current > expected_size:
        _emit(
            log_callback,
            f"⚠️ {label}：收到 {current} bytes，比官方对象多 {current - expected_size} bytes；"
            "先保留原文件，交由 SHA-256 校验尝试安全恢复。",
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_prefix(path: Path, length: int) -> str:
    """SHA-256 exactly the first *length* bytes without copying the artifact."""
    remaining = max(0, int(length))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while remaining > 0:
            chunk = stream.read(min(4 * 1024 * 1024, remaining))
            if not chunk:
                break
            digest.update(chunk)
            remaining -= len(chunk)
    if remaining:
        raise RuntimeError(f"文件短于待校验前缀：缺少 {remaining} bytes")
    return digest.hexdigest()


def _recover_small_trailing_bytes(
    path: Path, *, expected_size: int, validate: Callable[[Path], None] | None,
    label: str = "文件", log_callback=None, max_tail_bytes: int = 1024 * 1024,
) -> bool:
    """Safely recover a valid artifact with a small transport-added tail.

    Some resumable CDN/curl sessions can leave a tiny suffix after the exact
    LFS object (observed: 44 bytes after a 450,029,888-byte CoreML tarball).
    Never throw away a nearly-complete artifact merely because it is slightly
    oversized.  Temporarily trim only a small suffix, run the authoritative
    validator (size + SHA-256), and restore the exact suffix if validation
    fails.
    """
    if not path.exists() or not expected_size or validate is None:
        return False
    actual = path.stat().st_size
    extra = actual - int(expected_size)
    if extra <= 0 or extra > max(1, int(max_tail_bytes)):
        return False
    with path.open("r+b") as stream:
        stream.seek(expected_size)
        tail = stream.read()
        if len(tail) != extra:
            return False
        stream.truncate(expected_size)
        stream.flush()
        try:
            os.fsync(stream.fileno())
        except OSError:
            pass
    try:
        validate(path)
    except Exception:
        # Restore byte-for-byte original state before returning failure.
        with path.open("ab") as stream:
            stream.write(tail)
            stream.flush()
            try:
                os.fsync(stream.fileno())
            except OSError:
                pass
        return False
    _emit(
        log_callback,
        f"♻️ {label}：检测到传输尾部多出 {extra} bytes；前 {expected_size} bytes 已通过官方 SHA-256，"
        "已安全裁除尾部并复用完整文件。",
    )
    return True


def _quarantine_partial(path: Path, *, label: str = "文件", log_callback=None) -> Path | None:
    """Preserve an invalid complete-ish artifact instead of deleting it."""
    if not path.exists():
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    candidate = path.with_name(f"{path.name}.invalid-{stamp}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.invalid-{stamp}-{counter}")
        counter += 1
    os.replace(path, candidate)
    _emit(log_callback, f"🧯 {label}：完整性校验未通过，已隔离保留为 {candidate.name}，不会删除原始字节。")
    return candidate


def _source_archive_root(names: set[str]) -> str:
    """Return the one directory prefix that contains the inference source tree.

    GitHub/codeload archives contain a top-level directory, while a manually
    repacked local archive may place the project directly at ZIP root.  Do not
    locate critical files with basename/suffix matching: upstream also contains
    training-side paths such as ``make_traindata/util_func.py`` and those must
    never make the runtime validator ambiguous.
    """
    files = {str(name).replace("\\", "/").lstrip("./") for name in names if name and not name.endswith("/")}
    marker = "run_ocr.py"
    roots: set[str] = set()
    for name in files:
        if name == marker:
            roots.add("")
        elif name.endswith("/" + marker):
            roots.add(name[: -len(marker)])

    essentials = (
        "run_ocr.py",
        "process_ocr_base.py",
        "models/detector.py",
        "models/transformer.py",
        "textline_detect/Makefile",
    )
    complete = [root for root in sorted(roots) if all(root + rel in files for rel in essentials)]
    if len(complete) != 1:
        detail = ", ".join(repr(root or "<zip-root>") for root in complete) or "none"
        raise RuntimeError(f"源码 ZIP 无法唯一定位推理源码根目录（候选：{detail}）")
    return complete[0]


def _validate_source_archive(path: Path) -> None:
    if not path.is_file() or path.stat().st_size < 50_000:
        raise RuntimeError(f"源码压缩包过小：{path.stat().st_size if path.exists() else 0} bytes")
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
            if bad:
                raise RuntimeError(f"ZIP CRC 校验失败：{bad}")
            names = set(zf.namelist())
            root = _source_archive_root(names)
            member_map = {
                str(name).replace("\\", "/").lstrip("./"): name
                for name in names
            }
            normalized = set(member_map)

            # Verify exact Git blobs only at the selected project root.  The
            # repository intentionally has duplicate basenames in training
            # helpers (notably make_traindata/util_func.py), so endswith() is
            # unsafe here.
            for relpath, expected_blob in SOURCE_BLOB_SHA1.items():
                member = root + relpath
                if member not in normalized:
                    raise RuntimeError(f"源码 ZIP 缺少根目录文件 {relpath}")
                data = zf.read(member_map[member])
                header = f"blob {len(data)}\0".encode("ascii")
                actual_blob = hashlib.sha1(header + data).hexdigest()
                if actual_blob != expected_blob:
                    raise RuntimeError(
                        f"源码版本校验失败：{relpath} {actual_blob[:12]}… != {expected_blob[:12]}…"
                    )
    except zipfile.BadZipFile as exc:
        raise RuntimeError("findtextCenterNet 源码 ZIP 不完整或已损坏") from exc


def _validate_model(path: Path, *, expected_size: int, expected_sha256: str) -> None:
    actual_size = path.stat().st_size if path.exists() else 0
    if actual_size != expected_size:
        raise RuntimeError(f"模型大小不完整：{actual_size}/{expected_size} bytes")
    actual_hash = _sha256(path)
    if actual_hash.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"模型 SHA-256 校验失败：{actual_hash[:16]}… != {expected_sha256[:16]}…"
        )


def _validate_artifact(path: Path, spec: dict) -> None:
    _validate_model(
        path,
        expected_size=int(spec["size"]),
        expected_sha256=str(spec["sha256"]),
    )


def _backend_ready(source_dir: Path, backend: str) -> bool:
    backend = str(backend or "").strip().lower()
    try:
        if backend == "coreml":
            return all(
                (source_dir / str(spec["output"])).is_dir()
                and (source_dir / str(spec["output"]) / "Manifest.json").is_file()
                for spec in COREML_SPECS.values()
            )
        if backend == "onnx":
            return all(
                (source_dir / name).is_file()
                and (source_dir / name).stat().st_size == int(spec["size"])
                for name, spec in ONNX_SPECS.items()
            )
        if backend == "torch":
            return all(
                (source_dir / name).is_file()
                and (source_dir / name).stat().st_size == int(spec["size"])
                for name, spec in MODEL_SPECS.items()
            )
    except OSError:
        return False
    return False


def _ready_backend(source_dir: Path) -> str:
    # Keep the exact priority used by upstream run_ocr.py.
    for backend in ("coreml", "onnx", "torch"):
        if _backend_ready(source_dir, backend):
            return backend
    return ""


def _preferred_new_backend() -> str:
    override = str(os.environ.get("NOVEL_FORMATTER_FINDTEXT_BACKEND", "auto") or "auto").strip().lower()
    aliases = {"ml": "coreml", "core_ml": "coreml", "ort": "onnx", "pytorch": "torch"}
    override = aliases.get(override, override)
    if override in {"coreml", "onnx", "torch"}:
        return override
    # Upstream itself prefers CoreML first. On macOS, use the official
    # pre-converted CoreML packages rather than forcing 1.49 GB of Torch
    # checkpoints. Other platforms prefer the official quantized ONNX path.
    return "coreml" if sys.platform == "darwin" else "onnx"


def _safe_extract_mlpackage(archive: Path, source_dir: Path, output_name: str) -> None:
    """Extract one official .mlpackage tarball without trusting archive paths."""
    target = source_dir / output_name
    if target.is_dir():
        return
    staging = source_dir / (output_name + ".extracting")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    keep_staging = True
    try:
        with tarfile.open(archive, "r:gz") as tf:
            members = tf.getmembers()
            for member in members:
                member_path = Path(member.name)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise RuntimeError(f"不安全的 CoreML 归档路径：{member.name}")
            tf.extractall(staging, filter="data")
        candidates = [p for p in staging.rglob(output_name) if p.is_dir()]
        if len(candidates) != 1:
            if (staging / "Manifest.json").is_file():
                package_root = staging
            else:
                raise RuntimeError(f"CoreML 归档无法唯一定位 {output_name}")
        else:
            package_root = candidates[0]
        incoming = source_dir / (output_name + ".incoming")
        shutil.rmtree(incoming, ignore_errors=True)
        if package_root == staging:
            os.replace(staging, incoming)
            keep_staging = False
        else:
            shutil.move(str(package_root), str(incoming))
        if not (incoming / "Manifest.json").is_file():
            raise RuntimeError(f"{output_name} 缺少 Manifest.json")
        shutil.rmtree(target, ignore_errors=True)
        os.replace(incoming, target)
    finally:
        if keep_staging:
            shutil.rmtree(staging, ignore_errors=True)


def _env_true(name: str) -> bool:
    return str(os.environ.get(name, "") or "").strip().lower() in {"1", "true", "yes", "on"}


def _xet_health_path() -> Path:
    return RUNTIME_ROOT / "xet-health.json"


def _xet_cooldown_seconds() -> float:
    try:
        value = float(os.environ.get("NOVEL_FORMATTER_FINDTEXT_XET_COOLDOWN", "21600") or 21600)
    except Exception:
        value = 21600.0
    return max(0.0, min(value, 7 * 24 * 3600.0))


def _xet_cooldown_active() -> tuple[bool, str]:
    if _env_true("NOVEL_FORMATTER_FINDTEXT_FORCE_XET"):
        return False, ""
    path = _xet_health_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        until = float(data.get("degraded_until", 0) or 0)
        if until > time.time():
            remaining = max(1, int(until - time.time()))
            reason = str(data.get("reason", "上次 Xet 下载停滞") or "上次 Xet 下载停滞")
            return True, f"{reason}；{remaining // 60} 分钟内优先使用可续传 HTTP"
    except Exception:
        pass
    return False, ""


def _mark_xet_degraded(reason: str) -> None:
    cooldown = _xet_cooldown_seconds()
    if cooldown <= 0:
        return
    path = _xet_health_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "degraded_until": time.time() + cooldown,
            "reason": str(reason or "Xet 下载停滞"),
            "updated_at": time.time(),
        }
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        pass


def _clear_xet_degraded() -> None:
    try:
        _xet_health_path().unlink(missing_ok=True)
    except Exception:
        pass


def _hf_repo_cache_roots() -> tuple[Path, ...]:
    return tuple(root / "models--lithium0003--findtextCenterNet" for root in _huggingface_hub_roots())


def _xet_progress_snapshot(*, started_wall: float = 0.0) -> tuple[int, int, int]:
    """Return (bytes, newest_mtime_ns, file_count) for active HF incomplete files.

    huggingface_hub passes its normal cache ``*.incomplete`` path directly to
    ``xet_get``.  Watching file growth/mtime gives us a backend-independent
    liveness signal even when the Rust Xet client only prints "connection
    struggling" messages.  Existing stale partials are included in the initial
    snapshot but do not count as progress unless their size/mtime changes.
    """
    total = 0
    newest = 0
    count = 0
    for repo_root in _hf_repo_cache_roots():
        blobs = repo_root / "blobs"
        if not blobs.is_dir():
            continue
        try:
            candidates = list(blobs.glob("*.incomplete"))
        except OSError:
            continue
        for path in candidates:
            try:
                st = path.stat()
            except OSError:
                continue
            # Do not exclude pre-existing incomplete files: they are precisely
            # what Xet may resume.  We compare snapshots instead of absolute age.
            total += int(st.st_size)
            newest = max(newest, int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))))
            count += 1
    return total, newest, count


def _xet_line_expected_probe(line: str) -> bool:
    text = str(line or "").lower()
    return "416" in text and (
        "range" in text or "requested range not satisfiable" in text or "status code" in text
    )


def _terminate_download_process(proc) -> None:
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=5)
        return
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def _hf_xet_download(python: Path, *, filename: str, target: Path, spec: dict, log_callback=None) -> bool:
    """Try the official Hugging Face Xet path, but never let it block Ruby setup.

    Xet is a fast-path only.  If its standard HF cache stops changing for the
    configured stall window, or the attempt exceeds its wall-clock budget, the
    subprocess is terminated and the caller immediately falls back to the
    resumable HTTP transport.  Xet's own partial cache is left intact for a
    future retry; Novel-formatter's separate ``.part`` file is also untouched.
    """
    if _env_true("NOVEL_FORMATTER_FINDTEXT_DISABLE_XET") or _env_true("HF_HUB_DISABLE_XET"):
        _emit(log_callback, f"↪️ {filename}：Xet 已禁用，直接使用可续传 HTTP。")
        return False
    cooled, detail = _xet_cooldown_active()
    if cooled:
        _emit(log_callback, f"↪️ {filename}：{detail}。")
        return False

    script = (
        'import os\n'
        'from huggingface_hub import hf_hub_download\n'
        'path = hf_hub_download(repo_id=os.environ["NF_REPO"], filename=os.environ["NF_FILE"], revision=os.environ["NF_REV"])\n'
        'print("__NF_HF_RESULT__=" + path, flush=True)\n'
    )
    env = os.environ.copy()
    env["NF_REPO"] = HF_MODEL_REPO
    env["NF_FILE"] = filename
    env["NF_REV"] = UPSTREAM_WEIGHT_REVISION
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # Do not enable HIGH_PERFORMANCE automatically.  Xet already adapts its
    # concurrency, and on weak links aggressive concurrency can make recovery
    # worse.  User-provided HF_TOKEN/HF_XET_* settings are inherited unchanged.
    _emit(log_callback, f"⚡ {filename}：尝试 Hugging Face Xet；若停滞将自动切换可续传 HTTP…")
    proc = subprocess.Popen(
        [str(python), "-u", "-c", script],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env,
    )
    q = queue.Queue()
    result_path = ""

    def reader() -> None:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                q.put(line.rstrip("\r\n"))
        finally:
            q.put(None)

    threading.Thread(target=reader, daemon=True, name="findtext-hf-xet").start()
    started = time.monotonic()
    started_wall = time.time()
    try:
        stall_timeout = float(os.environ.get("NOVEL_FORMATTER_FINDTEXT_XET_STALL_TIMEOUT", "45") or 45)
    except Exception:
        stall_timeout = 45.0
    try:
        max_seconds = float(os.environ.get("NOVEL_FORMATTER_FINDTEXT_XET_MAX_SECONDS", "300") or 300)
    except Exception:
        max_seconds = 300.0
    stall_timeout = max(15.0, min(stall_timeout, 600.0))
    max_seconds = max(stall_timeout + 5.0, min(max_seconds, 3600.0))
    last_heartbeat = 0.0
    finished_stream = False
    snapshot = _xet_progress_snapshot(started_wall=started_wall)
    last_progress = started
    struggling_seen = 0
    expected_416_reported = False
    stalled = False
    stall_reason = ""

    while proc.poll() is None or not finished_stream:
        try:
            item = q.get(timeout=1.0)
            if item is None:
                finished_stream = True
            elif item.startswith("__NF_HF_RESULT__="):
                result_path = item.split("=", 1)[1].strip()
            elif item.strip():
                line = item.strip()
                low = line.lower()
                if _xet_line_expected_probe(line):
                    if not expected_416_reported:
                        _emit(log_callback, f"   HF/Xet · 416 Range 探测响应已忽略（非下载失败）")
                        expected_416_reported = True
                else:
                    if "connection struggling" in low:
                        struggling_seen += 1
                    _emit(log_callback, "   HF/Xet · " + line)
        except queue.Empty:
            pass

        now = time.monotonic()
        new_snapshot = _xet_progress_snapshot(started_wall=started_wall)
        if new_snapshot != snapshot:
            snapshot = new_snapshot
            last_progress = now

        if proc.poll() is None:
            if now - last_progress >= stall_timeout:
                stalled = True
                stall_reason = f"Xet 缓存连续 {int(stall_timeout)} 秒无字节/文件活动"
            elif now - started >= max_seconds:
                stalled = True
                stall_reason = f"Xet 快速尝试超过 {int(max_seconds)} 秒预算"
            if stalled:
                _emit(log_callback, f"⚠️ {filename}：{stall_reason}，正在终止 Xet 并切换可续传 HTTP…")
                _terminate_download_process(proc)
                _mark_xet_degraded(stall_reason)
                break

        if now - last_heartbeat >= 5.0 and proc.poll() is None:
            elapsed = max(1, int(now - started))
            cached_bytes, _mtime, incomplete_count = snapshot
            cache_text = (
                f"，HF incomplete 缓存 {_human_bytes(cached_bytes)} / {incomplete_count} 个文件"
                if incomplete_count else ""
            )
            struggling_text = f"，connection struggling×{struggling_seen}" if struggling_seen else ""
            _emit(
                log_callback,
                f"⚡ {filename}：HF/Xet 活动中（{elapsed} 秒{cache_text}{struggling_text}）…",
            )
            last_heartbeat = now

        if proc.poll() is not None and finished_stream:
            break

    code = proc.wait() if getattr(proc, "returncode", None) is None else proc.returncode
    if stalled:
        return False
    if code != 0 or not result_path:
        reason = f"Xet 子进程退出码 {code}" if code else "Xet 未返回完整文件路径"
        _mark_xet_degraded(reason)
        _emit(log_callback, f"⚠️ {filename}：HF/Xet 下载未完成，立即回退到可续传 HTTP。")
        return False
    candidate = Path(result_path)
    try:
        _validate_artifact(candidate, spec)
        target.parent.mkdir(parents=True, exist_ok=True)
        importing = target.with_name(target.name + ".hf-importing")
        importing.unlink(missing_ok=True)
        try:
            os.link(candidate, importing)
            method = "硬链接"
        except OSError:
            shutil.copy2(candidate, importing)
            method = "复制"
        _validate_artifact(importing, spec)
        os.replace(importing, target)
        _clear_xet_degraded()
        _emit(log_callback, f"✅ {filename}：HF/Xet 下载完成并已{method}到 findtextCenterNet 运行目录")
        return True
    except Exception as exc:
        _mark_xet_degraded(f"Xet 完成文件校验失败：{exc}")
        _emit(log_callback, f"⚠️ {filename}：HF/Xet 文件校验失败：{exc}；立即回退到可续传 HTTP。")
        return False

def _huggingface_hub_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    explicit = (
        os.environ.get("HF_HUB_CACHE", ""),
        os.environ.get("HUGGINGFACE_HUB_CACHE", ""),
    )
    for value in explicit:
        if str(value or "").strip():
            roots.append(Path(str(value)).expanduser())
    hf_home = str(os.environ.get("HF_HOME", "") or "").strip()
    if hf_home:
        roots.append(Path(hf_home).expanduser() / "hub")
    xdg = str(os.environ.get("XDG_CACHE_HOME", "") or "").strip()
    if xdg:
        roots.append(Path(xdg).expanduser() / "huggingface" / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            key = str(root.resolve())
        except Exception:
            key = str(root)
        if key not in seen:
            seen.add(key)
            out.append(root)
    return tuple(out)


def _huggingface_cached_model_candidates(name: str) -> tuple[Path, ...]:
    """Return already-downloaded official HF cache paths without networking.

    ``huggingface_hub`` stores snapshots as symlinks/hardlinks into ``blobs``.
    We prefer the exact pinned revision but also scan other snapshots; the
    mandatory size+SHA validator prevents a stale revision from being reused.
    """
    candidates: list[Path] = []
    repo_dirs = (
        "models--lithium0003--findtextCenterNet",
        "datasets--lithium0003--findtextCenterNet_dataset",
    )
    for hub in _huggingface_hub_roots():
        for repo_dir in repo_dirs:
            snapshots = hub / repo_dir / "snapshots"
            exact = snapshots / UPSTREAM_WEIGHT_REVISION / name
            candidates.append(exact)
            if snapshots.is_dir():
                try:
                    candidates.extend(sorted(snapshots.glob(f"*/{name}")))
                except OSError:
                    pass

    # If huggingface_hub is already available because another OCR installed it,
    # ask it for a local-only resolution as well.  This performs zero network I/O
    # and makes us resilient to future cache-layout changes.
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
        local = hf_hub_download(
            repo_id="lithium0003/findtextCenterNet",
            filename=name,
            revision=UPSTREAM_WEIGHT_REVISION,
            local_files_only=True,
        )
        candidates.insert(0, Path(local))
    except Exception:
        pass

    out: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            if not candidate.exists():
                continue
            resolved = candidate.resolve()
            key = str(resolved)
            if key in seen or not resolved.is_file():
                continue
            seen.add(key)
            out.append(resolved)
        except OSError:
            continue
    return tuple(out)


def _seed_huggingface_cached_model(
    target: Path, *, name: str, validate: Callable[[Path], None], log_callback=None,
) -> bool:
    for candidate in _huggingface_cached_model_candidates(name):
        try:
            validate(candidate)
            target.parent.mkdir(parents=True, exist_ok=True)
            importing = target.with_name(target.name + ".hf-importing")
            importing.unlink(missing_ok=True)
            try:
                os.link(candidate, importing)
                method = "硬链接"
            except OSError:
                shutil.copy2(candidate, importing)
                method = "复制"
            validate(importing)
            os.replace(importing, target)
            _emit(log_callback, f"♻️ 已复用 Hugging Face 缓存中的 {name}（{method}，无需下载）")
            return True
        except Exception:
            try:
                target.with_name(target.name + ".hf-importing").unlink(missing_ok=True)
            except Exception:
                pass
            continue
    return False


def _seed_local_file(
    target: Path, *, label: str, validate: Callable[[Path], None] | None = None,
    env_name: str = "", log_callback=None,
) -> bool:
    candidates: list[Path] = []
    configured = str(os.environ.get(env_name, "") or "").strip() if env_name else ""
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend([
        ROOT / target.name,
        ROOT / "models" / target.name,
        Path.home() / "Downloads" / target.name,
    ])
    if target.suffix.lower() == ".zip" and target.name.startswith("findtextCenterNet-"):
        # Common filenames produced when the pinned GitHub URL is downloaded
        # manually in a browser.  This lets users recover from restricted or
        # unstable networks without renaming the official archive.
        for base in (ROOT, Path.home() / "Downloads", Path.home() / "Desktop"):
            candidates.extend([
                base / f"{UPSTREAM_COMMIT}.zip",
                base / f"findtextCenterNet-{UPSTREAM_COMMIT}.zip",
                base / "findtextCenterNet.zip",
            ])
    seen: set[str] = set()
    for candidate in candidates:
        try:
            key = str(candidate.resolve())
            if key in seen or not candidate.is_file() or candidate.resolve() == target.resolve():
                continue
            seen.add(key)
            if validate is not None:
                validate(candidate)
            target.parent.mkdir(parents=True, exist_ok=True)
            importing = target.with_name(target.name + ".importing")
            shutil.copy2(candidate, importing)
            os.replace(importing, target)
            _emit(log_callback, f"✅ 已导入本地 {label}：{candidate}")
            return True
        except Exception:
            continue
    return False


def _download(
    urls: str | Iterable[str], target: Path, *, label: str, log_callback=None,
    expected_size: int = 0, validate: Callable[[Path], None] | None = None,
    local_env: str = "", attempts_per_transport: int = 3,
) -> Path:
    """Resumable, multi-origin optional-runtime downloader.

    A short network read is not fatal: bytes stay in ``.part`` and the next
    attempt sends a Range request. Completed files are atomically promoted only
    after size/hash/ZIP validation.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    if target.exists():
        try:
            if validate is not None:
                validate(target)
            elif expected_size and target.stat().st_size != expected_size:
                raise RuntimeError("已有文件大小不完整")
            return target
        except Exception:
            target.unlink(missing_ok=True)

    # A previous version may have downloaded a complete artifact into .part but
    # rejected it because of an overly strict/buggy validator. Re-validate the
    # existing bytes before touching the network. This is especially important
    # for the pinned GitHub source ZIP, where upstream has duplicate basenames
    # (e.g. root util_func.py and make_traindata/util_func.py).
    if partial.exists() and partial.stat().st_size > 0:
        try:
            if validate is not None:
                validate(partial)
            elif expected_size and partial.stat().st_size != expected_size:
                raise RuntimeError("临时文件尚未完整")
            else:
                raise RuntimeError("临时文件需要继续下载")
            os.replace(partial, target)
            _emit(log_callback, f"♻️ 已恢复并复用完整的 {label} 临时文件，无需重新下载")
            return target
        except Exception:
            # Never delete a complete-ish artifact.  First try the common
            # transport-tail recovery (authoritative validator decides), then
            # quarantine any still-invalid complete file byte-for-byte.
            if expected_size and partial.stat().st_size > expected_size:
                if _recover_small_trailing_bytes(
                    partial, expected_size=expected_size, validate=validate,
                    label=label, log_callback=log_callback,
                ):
                    os.replace(partial, target)
                    return target
            complete_invalid = bool(expected_size and partial.stat().st_size >= expected_size)
            if not complete_invalid and target.suffix.lower() == ".zip":
                try:
                    with zipfile.ZipFile(partial) as zf:
                        zf.infolist()
                    complete_invalid = True
                except zipfile.BadZipFile:
                    complete_invalid = False
            if complete_invalid:
                _quarantine_partial(partial, label=label, log_callback=log_callback)

    if _seed_local_file(
        target, label=label, validate=validate, env_name=local_env, log_callback=log_callback
    ):
        return target

    url_list = [urls] if isinstance(urls, str) else [str(item) for item in urls if str(item)]
    errors: list[str] = []
    # ``attempts_per_transport`` is a *consecutive no-progress* budget, not a
    # lifetime connection-count budget. Large HF artifacts commonly make tens
    # of MiB of valid progress before a CDN/socket reset; counting every such
    # reset as a fatal retry caused a 94%-complete file to be abandoned.
    no_progress_limit = max(1, int(attempts_per_transport))
    max_connections = max(
        no_progress_limit,
        int(os.environ.get("NOVEL_FORMATTER_FINDTEXT_HTTP_MAX_CONNECTIONS", "64") or "64"),
    )
    for url_index, url in enumerate(url_list):
        if url_index:
            _emit(log_callback, f"↪️ {label} 切换官方备用下载源：{url}")
        curl_available = bool(shutil.which("curl"))
        # After Xet falls back, Hugging Face large-file routes are more robust on
        # affected macOS networks with curl forced to HTTP/1.1 than with a long
        # urllib session. Prefer that path for HF artifacts, while keeping the
        # Python transport as a portable fallback and for non-HF origins.
        if curl_available and "huggingface.co/" in url.lower():
            transports = [
                ("系统 curl · HTTP/1.1", _download_curl_once),
                ("Python HTTPS", _download_python_once),
            ]
        else:
            transports = [("Python HTTPS", _download_python_once)]
            if curl_available:
                transports.append(("系统 curl · HTTP/1.1", _download_curl_once))
        for transport_name, transport in transports:
            consecutive_no_progress = 0
            connection_no = 0
            while consecutive_no_progress < no_progress_limit and connection_no < max_connections:
                connection_no += 1
                download_completed = False
                before = partial.stat().st_size if partial.exists() else 0
                try:
                    suffix = f"（续传 {before} bytes）" if before else ""
                    _emit(
                        log_callback,
                        f"⬇️ {label} · {transport_name} · 连接 {connection_no}{suffix}"
                        f"（连续无进展 {consecutive_no_progress}/{no_progress_limit}）：{url}",
                    )
                    transport(
                        url, partial, label=label, expected_size=expected_size,
                        log_callback=log_callback,
                    )
                    download_completed = True
                    if validate is not None:
                        _emit(log_callback, f"🔐 {label}：下载完成，正在校验完整性…")
                        try:
                            validate(partial)
                        except Exception:
                            if not (
                                expected_size
                                and partial.exists()
                                and partial.stat().st_size > expected_size
                                and _recover_small_trailing_bytes(
                                    partial, expected_size=expected_size, validate=validate,
                                    label=label, log_callback=log_callback,
                                )
                            ):
                                raise
                    elif expected_size and partial.stat().st_size != expected_size:
                        raise RuntimeError(f"文件大小不完整：{partial.stat().st_size}/{expected_size}")
                    os.replace(partial, target)
                    _emit(log_callback, f"✅ {label} 下载完整并通过校验")
                    return target
                except Exception as exc:
                    after = partial.stat().st_size if partial.exists() else 0
                    gained = max(0, after - before)
                    errors.append(f"{transport_name}#{connection_no}: {exc}")

                    # A completed-but-invalid artifact cannot be repaired by appending,
                    # but it must never be silently deleted.  Tiny oversize tails
                    # are recoverable if the authoritative validator accepts the
                    # exact expected-size prefix; otherwise quarantine the bytes.
                    if partial.exists() and (
                        download_completed or (expected_size and partial.stat().st_size >= expected_size)
                    ):
                        recovered = False
                        if expected_size and partial.stat().st_size > expected_size:
                            recovered = _recover_small_trailing_bytes(
                                partial, expected_size=expected_size, validate=validate,
                                label=label, log_callback=log_callback,
                            )
                        if recovered:
                            os.replace(partial, target)
                            _emit(log_callback, f"✅ {label} 已从完整下载尾部异常中恢复并通过校验")
                            return target
                        _quarantine_partial(partial, label=label, log_callback=log_callback)
                        after = 0
                        gained = 0

                    if gained > 0:
                        consecutive_no_progress = 0
                        _emit(
                            log_callback,
                            f"⚠️ {label} 连接中断但本次已新增 {_human_bytes(gained)}；"
                            f"有效进展不计失败预算，将从 {_human_bytes(after)} 自动续传。",
                        )
                    else:
                        consecutive_no_progress += 1
                        _emit(
                            log_callback,
                            f"⚠️ {label} 下载中断且本次无新增字节：{exc}；"
                            f"连续无进展 {consecutive_no_progress}/{no_progress_limit}。",
                        )
                    time.sleep(min(2.0, 0.35 * max(1, consecutive_no_progress)))

            if connection_no >= max_connections and partial.exists():
                _emit(
                    log_callback,
                    f"⚠️ {label} · {transport_name} 已达到单传输层 {max_connections} 次连接安全上限；"
                    "保留现有 .part，并尝试下一传输层/备用源。",
                )
    detail = " | ".join(errors[-8:])
    current = partial.stat().st_size if partial.exists() else 0
    progress = f"；已保留 {_human_bytes(current)} 临时文件供下次继续" if current else ""
    raise RuntimeError(
        f"{label} 连续多次无有效进展，当前下载仍未完成{progress}；"
        f"可重新点击开始续传，或手动放入运行目录。{detail}"
    )


def _install_source_with_git(source_dir: Path, log_callback=None) -> tuple[bool, str]:
    """Install the pinned upstream source with git, matching other OCR adapters.

    Git is the primary transport because it verifies the requested commit and
    avoids archive-root/layout ambiguity.  The resumable ZIP installer remains
    as a fallback for systems without git or networks that block git smart HTTP.
    """
    git = shutil.which("git")
    if not git:
        return False, "系统未找到 git"
    stage = RUNTIME_ROOT / "_git_source"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True, exist_ok=True)
    try:
        _emit(log_callback, f"⬇️ 准备 findtextCenterNet 源码（git · {UPSTREAM_COMMIT[:12]}）…")
        subprocess.run([git, "init", "-q", str(stage)], check=True, capture_output=True, text=True, timeout=60)
        subprocess.run([git, "-C", str(stage), "remote", "add", "origin", UPSTREAM_REPO_URL], check=True, capture_output=True, text=True, timeout=30)
        fetched = subprocess.run(
            [git, "-C", str(stage), "fetch", "--depth", "1", "origin", UPSTREAM_COMMIT],
            capture_output=True, text=True, timeout=900,
        )
        if fetched.returncode != 0:
            return False, (fetched.stderr or fetched.stdout or "git fetch 失败")[-1600:]
        checked = subprocess.run(
            [git, "-C", str(stage), "checkout", "-q", "--detach", "FETCH_HEAD"],
            capture_output=True, text=True, timeout=60,
        )
        if checked.returncode != 0 or not _source_tree_ready(stage):
            return False, (checked.stderr or checked.stdout or "git checkout 后源码不完整")[-1600:]
        rev = subprocess.run(
            [git, "-C", str(stage), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30
        )
        if rev.returncode != 0 or rev.stdout.strip().lower() != UPSTREAM_COMMIT.lower():
            return False, f"git 源码 commit 校验失败：{rev.stdout.strip() or 'unknown'}"
        shutil.rmtree(stage / ".git", ignore_errors=True)
        shutil.rmtree(source_dir, ignore_errors=True)
        source_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(stage), str(source_dir))
        marker = {"upstream": "lithium0003/findtextCenterNet", "commit": UPSTREAM_COMMIT, "transport": "git"}
        (source_dir / ".novel-formatter-source.json").write_text(
            json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _emit(log_callback, "✅ findtextCenterNet 固定源码已通过 git commit 校验")
        return True, ""
    except Exception as exc:
        return False, str(exc)
    finally:
        if stage.exists() and stage != source_dir:
            shutil.rmtree(stage, ignore_errors=True)


def _install_source(source_dir: Path, log_callback=None) -> None:
    if _source_tree_ready(source_dir):
        return
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    git_ok, git_detail = _install_source_with_git(source_dir, log_callback=log_callback)
    if git_ok:
        return
    _emit(log_callback, f"ℹ️ findtextCenterNet git 获取不可用，回退可续传 ZIP：{git_detail}")
    archive = RUNTIME_ROOT / f"findtextCenterNet-{UPSTREAM_COMMIT[:12]}.zip"
    partial = archive.with_name(archive.name + ".part")
    # Migrate the previous downloader's partial file so the exact bytes from an
    # IncompleteRead (for example 295418 bytes) are not thrown away after an
    # upgrade. The new downloader will issue a Range request from that offset.
    legacy_archive = RUNTIME_ROOT / "findtextCenterNet-main.zip"
    legacy_partial = legacy_archive.with_name(legacy_archive.name + ".part")
    if not archive.exists() and not partial.exists():
        if legacy_archive.exists():
            try:
                _validate_source_archive(legacy_archive)
                os.replace(legacy_archive, archive)
                _emit(log_callback, "♻️ 已复用旧版已下载的 findtextCenterNet 源码包")
            except Exception:
                legacy_archive.unlink(missing_ok=True)
        if not archive.exists() and legacy_partial.exists() and legacy_partial.stat().st_size > 0:
            os.replace(legacy_partial, partial)
            _emit(
                log_callback,
                f"♻️ 检测到旧版中断下载 {partial.stat().st_size} bytes，将从断点继续",
            )
    _download(
        SOURCE_ARCHIVE_URLS,
        archive,
        label="下载 findtextCenterNet 源码",
        log_callback=log_callback,
        validate=_validate_source_archive,
        local_env="NOVEL_FORMATTER_FINDTEXT_CENTERNET_ARCHIVE",
    )
    extract_root = RUNTIME_ROOT / "_extract"
    shutil.rmtree(extract_root, ignore_errors=True)
    extract_root.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extract_root)
        candidates = [p for p in extract_root.iterdir() if p.is_dir() and _source_tree_ready(p)]
        if len(candidates) != 1:
            raise RuntimeError("findtextCenterNet 源码压缩包结构异常：无法唯一定位完整源码目录")
        staged = candidates[0]
        shutil.rmtree(source_dir, ignore_errors=True)
        shutil.move(str(staged), str(source_dir))
        marker = {"upstream": "lithium0003/findtextCenterNet", "commit": UPSTREAM_COMMIT}
        (source_dir / ".novel-formatter-source.json").write_text(
            json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    finally:
        shutil.rmtree(extract_root, ignore_errors=True)

def _ensure_venv(source_dir: Path, log_callback=None) -> Path:
    python = _runtime_python(source_dir)
    if source_dir != DEFAULT_SOURCE_DIR and python.exists():
        return python
    # Use the same idempotent venv manager as Hayai/48px/Paddle adapters.
    # A valid existing environment is reused silently; package installation only
    # runs when the import marker is actually missing.
    packages = [
        os.environ.get("NOVEL_FORMATTER_FINDTEXT_TORCH", "torch>=2.3,<3"),
        os.environ.get("NOVEL_FORMATTER_FINDTEXT_TORCHVISION", "torchvision>=0.18,<1"),
        "numpy>=1.26,<3",
        "Pillow>=10,<13",
        # Official HF model repo is Xet-backed. huggingface_hub >=0.32 pulls
        # hf_xet automatically; keeping hf_xet explicit also repairs older
        # environments upgraded in-place.
        "huggingface_hub>=0.32,<2",
        "hf_xet>=1.1,<2",
    ]
    marker = "import torch, torchvision, numpy, huggingface_hub; from PIL import Image; assert torch.__version__"
    if sys.platform == "darwin":
        packages.append("coremltools>=8,<10")
        marker += "; import coremltools"
    else:
        packages.append("onnxruntime>=1.18,<2")
        marker += "; import onnxruntime"
    return ensure_venv(
        DEFAULT_VENV_DIR,
        label="findtextCenterNet Ruby",
        marker_code=marker,
        packages=packages,
        verbose=bool(log_callback),
        min_minor=10,
        max_minor=13,
    )

def _compile_linedetect(source_dir: Path, log_callback=None) -> None:
    target = _linedetect_path(source_dir)
    if target.exists():
        return
    work = source_dir / "textline_detect"
    _emit(log_callback, "🧩 编译 findtextCenterNet 文本行/Ruby 结构解析器…")
    if os.name == "nt":
        nmake = shutil.which("nmake")
        if not nmake:
            raise RuntimeError(
                "Windows 首次启用 findtextCenterNet Ruby 需要 Microsoft C++ Build Tools（nmake/cl）；"
                "安装后重新启用，或通过 NOVEL_FORMATTER_FINDTEXT_CENTERNET_DIR 指向已编译的上游运行目录。"
            )
        subprocess.run([nmake, "/f", "Makefile.mak"], cwd=work, check=True, timeout=1200)
    else:
        make = shutil.which("make")
        if not make:
            raise RuntimeError("未找到 make，无法编译 findtextCenterNet/textline_detect")
        subprocess.run([make], cwd=work, check=True, timeout=1200)
    if not target.exists():
        raise RuntimeError(f"findtextCenterNet linedetect 编译完成但未生成：{target}")


def _align_coreml_transformer_length(source_dir: Path, log_callback=None) -> None:
    """Align the copied upstream window with the installed CoreML specs.

    The source commit is verified before this compatibility edit.  Only the
    generated local runtime copy is changed, so the upstream source archive
    and its Git blob checks remain reproducible.
    """
    const_path = source_dir / "const.py"
    try:
        original = const_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"无法读取 findtextCenterNet 常量文件：{const_path}") from exc

    replacements = {
        "max_encoderlen": COREML_TRANSFORMER_LENGTH,
        "max_decoderlen": COREML_TRANSFORMER_LENGTH,
    }
    updated = original
    counts: dict[str, int] = {}
    for name, value in replacements.items():
        pattern = rf"(?m)^{re.escape(name)}[ \t]*=[ \t]*\d+[ \t]*$"
        updated, count = re.subn(pattern, f"{name} = {value}", updated)
        counts[name] = count

    if any(count != 1 for count in counts.values()):
        raise RuntimeError(
            "findtextCenterNet CoreML 兼容处理未找到唯一的 Transformer 长度定义："
            + ", ".join(f"{name}={count}" for name, count in counts.items())
        )
    if updated == original:
        return

    temporary = const_path.with_name(const_path.name + ".coreml-aligning")
    try:
        temporary.unlink(missing_ok=True)
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            stream.write(updated)
            stream.flush()
            try:
                os.fsync(stream.fileno())
            except OSError:
                pass
        shutil.copymode(const_path, temporary)
        os.replace(temporary, const_path)
    finally:
        temporary.unlink(missing_ok=True)
    _emit(
        log_callback,
        f"✅ findtextCenterNet CoreML 兼容：Transformer 编码/解码长度已对齐为 "
        f"{COREML_TRANSFORMER_LENGTH}（官方模型规格）",
    )


def _findtext_worker_environment() -> dict[str, str]:
    """Give CoreML a stable writable directory for its compiled model cache."""
    environment = os.environ.copy()
    if sys.platform != "darwin":
        return environment
    configured = str(os.environ.get("NOVEL_FORMATTER_FINDTEXT_TMPDIR", "") or "").strip()
    temporary = Path(configured).expanduser() if configured else RUNTIME_ROOT / "tmp"
    try:
        temporary.mkdir(parents=True, exist_ok=True)
        environment["TMPDIR"] = str(temporary.resolve())
    except OSError:
        # Keep the inherited value as a fallback; the actual worker error will
        # still be reported through the existing startup diagnostics.
        pass
    return environment


def prepare_runtime(*, log_callback=None) -> tuple[Path, Path]:
    """Prepare the optional upstream runtime after explicit opt-in.

    The lock prevents two OCR jobs from racing while creating the venv,
    downloading multi-GB weights, or compiling ``linedetect``.
    """
    with _RUNTIME_PREPARE_LOCK:
        source = _source_dir()
        if source == DEFAULT_SOURCE_DIR:
            _install_source(source, log_callback=log_callback)
        elif not _source_tree_ready(source):
            raise RuntimeError(f"指定的 findtextCenterNet 目录不完整或无效：{source}")
        python = _ensure_venv(source, log_callback=log_callback)

        # If any upstream-supported backend is already complete, keep it. This
        # avoids downloading CoreML/ONNX merely because the user previously
        # finished the Torch weights. Otherwise prepare the platform-native
        # backend that upstream run_ocr.py already knows how to select.
        backend = _ready_backend(source)
        if not backend:
            backend = _preferred_new_backend()
            _emit(log_callback, f"🧭 findtextCenterNet：按原项目后端优先级准备 {backend.upper()} 运行文件")

            if backend == "coreml":
                download_dir = RUNTIME_ROOT / "downloads"
                for name, spec in COREML_SPECS.items():
                    output = source / str(spec["output"])
                    if output.is_dir():
                        continue
                    archive = download_dir / name
                    if not archive.exists() or archive.stat().st_size != int(spec["size"]):
                        if not _hf_xet_download(python, filename=name, target=archive, spec=spec, log_callback=log_callback):
                            _download(
                                f"https://huggingface.co/{HF_MODEL_REPO}/resolve/{UPSTREAM_WEIGHT_REVISION}/{name}?download=true",
                                archive, label=f"下载 {name}", log_callback=log_callback,
                                expected_size=int(spec["size"]),
                                validate=lambda path, spec=spec: _validate_artifact(path, spec),
                                attempts_per_transport=4,
                            )
                    _validate_artifact(archive, spec)
                    _safe_extract_mlpackage(archive, source, str(spec["output"]))

            elif backend == "onnx":
                for name, spec in ONNX_SPECS.items():
                    target = source / name
                    if target.exists() and target.stat().st_size == int(spec["size"]):
                        continue
                    if not _hf_xet_download(python, filename=name, target=target, spec=spec, log_callback=log_callback):
                        _download(
                            f"https://huggingface.co/{HF_MODEL_REPO}/resolve/{UPSTREAM_WEIGHT_REVISION}/{name}?download=true",
                            target, label=f"下载 {name}", log_callback=log_callback,
                            expected_size=int(spec["size"]),
                            validate=lambda path, spec=spec: _validate_artifact(path, spec),
                            attempts_per_transport=4,
                        )

            else:  # torch
                for name, spec in MODEL_SPECS.items():
                    target = source / name
                    expected_size = int(spec["size"])
                    expected_hash = str(spec["sha256"])
                    if target.exists() and target.stat().st_size == expected_size:
                        continue
                    validator = lambda path, size=expected_size, digest=expected_hash: _validate_model(
                        path, expected_size=size, expected_sha256=digest
                    )
                    reused_hf = _seed_huggingface_cached_model(
                        target, name=name, validate=validator, log_callback=log_callback
                    )
                    if not reused_hf and not _hf_xet_download(
                        python, filename=name, target=target, spec=spec, log_callback=log_callback
                    ):
                        _download(
                            spec["urls"], target, label=f"下载 {name}", log_callback=log_callback,
                            expected_size=expected_size, validate=validator,
                            local_env=(
                                "NOVEL_FORMATTER_FINDTEXT_MODEL_FILE" if name == "model.pt"
                                else "NOVEL_FORMATTER_FINDTEXT_MODEL3_FILE"
                            ),
                            attempts_per_transport=4,
                        )

        if _ready_backend(source) == "coreml":
            _align_coreml_transformer_length(source, log_callback=log_callback)
        _compile_linedetect(source, log_callback=log_callback)
        ready, detail = runtime_ready(source)
        if not ready:
            raise RuntimeError(detail)
        active_backend = _ready_backend(source) or backend
        try:
            from adapters.ocr_runtime_catalog import mark_runtime_ready
            mark_runtime_ready(
                "findtext_centernet_ruby",
                upstream="lithium0003/findtextCenterNet", commit=UPSTREAM_COMMIT,
                backend=active_backend,
            )
        except Exception:
            pass
        return source, python


def _normalise_context(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.replace("\uFFF9", "").replace("\uFFFA", "").replace("\uFFFB", "")
    return re.sub(r"\s+", "", value)


def _line_pair_boxes(
    item: dict, payload: dict, pairs: tuple[tuple[str, str], ...], *,
    offset_x: float = 0.0, offset_y: float = 0.0,
) -> tuple[tuple[float, float, float, float], ...]:
    """Recover per-base geometry from findtextCenterNet character boxes.

    Upstream emits every decoded character with ``blockidx``/``lineidx`` and
    ``rubybase`` flags.  Group consecutive ruby-base characters, then map the
    groups to the Aozora pairs from the corresponding line.  Geometry is used
    only as alignment evidence; it never becomes OCR text or a fusion vote.
    """
    if not pairs:
        return ()
    raw_boxes = payload.get("box") or []
    if not isinstance(raw_boxes, list):
        return ()
    blockidx = item.get("blockidx")
    lineidx = item.get("lineidx")
    if blockidx is None or lineidx is None:
        return ()

    ordered: list[dict] = []
    for raw in raw_boxes:
        if not isinstance(raw, dict):
            continue
        if raw.get("blockidx") != blockidx or raw.get("lineidx") != lineidx:
            continue
        ordered.append(raw)

    groups: list[list[dict]] = []
    current: list[dict] = []
    for raw in ordered:
        is_base = bool(raw.get("rubybase"))
        if is_base:
            current.append(raw)
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    if not groups:
        return ()

    def group_text(group: list[dict]) -> str:
        return "".join(str(raw.get("text") or "") for raw in group)

    def group_box(group: list[dict]) -> tuple[float, float, float, float] | None:
        coords: list[tuple[float, float, float, float]] = []
        for raw in group:
            try:
                cx = float(raw.get("cx") or 0.0) + float(offset_x)
                cy = float(raw.get("cy") or 0.0) + float(offset_y)
                w = float(raw.get("w") or 0.0)
                h = float(raw.get("h") or 0.0)
            except (TypeError, ValueError):
                continue
            if w <= 0 or h <= 0:
                continue
            coords.append((cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0))
        if not coords:
            return None
        return (
            min(v[0] for v in coords), min(v[1] for v in coords),
            max(v[2] for v in coords), max(v[3] for v in coords),
        )

    unused = set(range(len(groups)))
    mapped: list[tuple[float, float, float, float]] = []
    for pair_index, (base, _reading) in enumerate(pairs):
        base_norm = _normalise_context(base)
        chosen = None
        for group_index in sorted(unused):
            if _normalise_context(group_text(groups[group_index])) == base_norm:
                chosen = group_index
                break
        # When upstream character text and Aozora text differ only enough to
        # defeat exact matching, preserve deterministic reading order iff the
        # group cardinality itself is unambiguous.
        if chosen is None and len(groups) == len(pairs) and pair_index < len(groups):
            chosen = pair_index
        if chosen is None or chosen not in unused:
            return ()
        box = group_box(groups[chosen])
        if box is None:
            return ()
        unused.remove(chosen)
        mapped.append(box)
    return tuple(mapped)


def _parse_payload(
    page: int, payload: dict, *, offset_x: float = 0.0, offset_y: float = 0.0,
) -> list[RubyLine]:
    """Parse upstream JSON and map ROI-local boxes back to original-page pixels."""
    entries = payload.get("line") or payload.get("block") or []
    out: list[RubyLine] = []
    seen: set[tuple] = set()
    for item in entries:
        if not isinstance(item, dict):
            continue
        aozora = str(item.get("aozora") or "")
        plain = str(item.get("noruby") or "")
        pairs = tuple((m.group(1), m.group(2)) for m in RUBY_RE.finditer(aozora))
        if not pairs:
            continue
        if not plain:
            plain = RUBY_RE.sub(lambda m: m.group(1), aozora)
        x1 = float(item.get("x1") or 0.0) + float(offset_x)
        y1 = float(item.get("y1") or 0.0) + float(offset_y)
        x2 = float(item.get("x2") or 0.0) + float(offset_x)
        y2 = float(item.get("y2") or 0.0) + float(offset_y)
        pair_boxes = _line_pair_boxes(
            item, payload, pairs, offset_x=offset_x, offset_y=offset_y
        )
        key = (aozora, plain, round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2))
        if key in seen:
            continue
        seen.add(key)
        out.append(RubyLine(
            page=page, aozora=aozora, plain=plain,
            x1=x1, y1=y1, x2=x2, y2=y2, pair_boxes=pair_boxes,
        ))
    return out


def _dedupe_ruby_lines(lines: Iterable[RubyLine]) -> list[RubyLine]:
    """Deduplicate the same Ruby line seen in overlapping context ROIs."""
    kept: list[RubyLine] = []
    for line in sorted(lines, key=lambda item: (item.page, item.y1, item.x1, item.aozora)):
        duplicate = False
        for previous in kept:
            if previous.page != line.page or previous.aozora != line.aozora:
                continue
            # Exact/nearby detections from overlapping crops should count once.
            pcx = (previous.x1 + previous.x2) / 2.0
            pcy = (previous.y1 + previous.y2) / 2.0
            lcx = (line.x1 + line.x2) / 2.0
            lcy = (line.y1 + line.y2) / 2.0
            if abs(pcx - lcx) <= 48 and abs(pcy - lcy) <= 64:
                duplicate = True
                break
        if not duplicate:
            kept.append(line)
    return kept


def _text_page_images(document: UnifiedDocument) -> list[tuple[int, str]]:
    pages_with_text = {
        int(block.page or block.page_index or block.page_number or 0)
        for block in document.blocks
        if block.type in _TEXT_TYPES and str(block.text or "").strip()
    }
    out: list[tuple[int, str]] = []
    for page in document.pages:
        page_no = int(page.page_no or 0)
        path = str(page.image_path or "")
        if page_no in pages_with_text and path and Path(path).exists():
            out.append((page_no, path))
    return out


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _run_process(
    command: list[str], *, cwd: Path, cancel_check=None, timeout: float = 3600.0
) -> tuple[int, str, str]:
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    started = time.monotonic()
    while proc.poll() is None:
        if callable(cancel_check) and cancel_check():
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise InterruptedError("Ruby 识别已停止")
        if time.monotonic() - started > timeout:
            proc.kill()
            raise TimeoutError("findtextCenterNet Ruby 识别超时")
        time.sleep(0.10)
    stdout, stderr = proc.communicate()
    return int(proc.returncode or 0), stdout, stderr




class FindtextCenterNetSession:
    """Persistent adapter around the *unmodified* upstream ``run_ocr.py``.

    The parent prepares source/models/linedetect once.  This process then imports
    upstream ``run_ocr`` once, so Detector + Transformer weights remain resident
    while all Ruby Smart ROIs are processed.  This mirrors the persistent-worker
    architecture used by the other heavy local OCR adapters.
    """

    def __init__(self, source_dir: Path, python: Path, *, cancel_check=None, log_callback=None):
        self.source_dir = Path(source_dir)
        self.python = Path(python)
        self.cancel_check = cancel_check
        self.log_callback = log_callback
        self.proc = None
        self._stdout_pump = None
        self._stderr_pump = None
        self._request_id = 0
        self.backend = ""
        self._last_heartbeat = 0.0

    def _drain_upstream_log(self) -> None:
        pump = self._stderr_pump
        if pump is None:
            return
        for line in pump.get_nowait_lines(limit=100):
            line = str(line).strip()
            if line:
                _emit(self.log_callback, "findtext · " + line)

    def _read_protocol(self, *, timeout: float, label: str) -> dict:
        assert self.proc is not None
        assert self._stdout_pump is not None

        def on_wait() -> None:
            self._drain_upstream_log()
            now = time.monotonic()
            if now - self._last_heartbeat >= 12.0:
                self._last_heartbeat = now
                _emit(self.log_callback, f"⏳ {label}仍在运行；上游模型/ROI 进程保持响应…")

        while True:
            line = self._stdout_pump.readline(
                proc=self.proc,
                timeout=timeout,
                cancel_check=self.cancel_check,
                label=label,
                on_wait=on_wait,
            )
            if line is None:
                self._drain_upstream_log()
                raise RuntimeError("findtextCenterNet 上游 worker 提前退出")
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                # Protocol stdout should contain only JSON, but keep fail-open
                # tolerance in case an upstream dependency writes to fd=1.
                _emit(self.log_callback, "findtext · " + line)
                continue
            if isinstance(data, dict):
                return data

    def start(self) -> "FindtextCenterNetSession":
        if self.proc is not None and self.proc.poll() is None:
            return self
        self.close()
        from adapters.subprocess_watchdog import LinePump, isolated_process_kwargs, env_seconds
        command = [
            str(self.python), str(WORKER_SCRIPT),
            "--source-root", str(self.source_dir),
        ]
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=_findtext_worker_environment(),
            **isolated_process_kwargs(),
        )
        self._stdout_pump = LinePump(self.proc.stdout, name="findtext-upstream-stdout")
        self._stderr_pump = LinePump(self.proc.stderr, name="findtext-upstream-stderr")
        ready = self._read_protocol(
            timeout=env_seconds("NOVEL_FORMATTER_OCR_STARTUP_TIMEOUT", 900.0, minimum=60.0),
            label="findtextCenterNet 上游模型初始化",
        )
        if not ready.get("ready"):
            error = str(ready.get("error") or "上游 worker 未就绪")
            self.close()
            raise RuntimeError(error)
        self.backend = str(ready.get("backend") or "")
        _emit(
            self.log_callback,
            f"✅ findtextCenterNet 上游原生 worker 已就绪 · backend={self.backend or 'unknown'} · 模型仅加载一次",
        )
        return self

    def recognize(self, image_paths: list[str], *, timeout: float | None = None) -> dict[str, tuple[dict | None, str | None]]:
        if not image_paths:
            return {}
        self.start()
        assert self.proc is not None and self.proc.stdin is not None
        from adapters.subprocess_watchdog import env_seconds
        self._request_id += 1
        request_id = self._request_id
        request = {
            "request_id": request_id,
            "images": [str(Path(path).resolve()) for path in image_paths],
            "resize": 1.0,
        }
        self.proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        data = self._read_protocol(
            timeout=(
                float(timeout) if timeout is not None else
                env_seconds("NOVEL_FORMATTER_FINDTEXT_REQUEST_TIMEOUT", 3600.0, minimum=60.0)
            ),
            label="findtextCenterNet Ruby ROI",
        )
        if int(data.get("request_id", -1) or -1) != request_id:
            raise RuntimeError("findtextCenterNet worker 请求响应串位")
        out: dict[str, tuple[dict | None, str | None]] = {}
        for item in data.get("items", []) or []:
            path = str(item.get("path") or "")
            if item.get("ok") and isinstance(item.get("payload"), dict):
                out[path] = (dict(item["payload"]), None)
            else:
                out[path] = (None, str(item.get("error") or "上游未返回有效 JSON"))
        self._drain_upstream_log()
        return out

    def close(self) -> None:
        proc = self.proc
        self.proc = None
        if proc is not None:
            try:
                if proc.poll() is None and proc.stdin is not None:
                    proc.stdin.write(json.dumps({"command": "close"}) + "\n")
                    proc.stdin.flush()
                    proc.wait(timeout=3)
            except Exception:
                try:
                    from adapters.subprocess_watchdog import terminate_process
                    terminate_process(proc)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass
        for pump in (self._stdout_pump, self._stderr_pump):
            try:
                if pump is not None:
                    pump.close()
            except Exception:
                pass
        self._stdout_pump = None
        self._stderr_pump = None

    def __enter__(self) -> "FindtextCenterNetSession":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _merge_ruby_candidate_payloads(documents: Iterable[UnifiedDocument]) -> dict[str, dict]:
    """Merge geometry-only Ruby hints from all ordinary OCR documents.

    Columns are matched by original-page geometry rather than list index.  This
    matters when two OCR adapters segment the same page into a slightly
    different number of columns: index-based merging can otherwise attach a
    candidate from one physical column to its neighbour.  Recognized text is
    never read or copied, so this ledger still cannot become an OCR vote.
    """
    merged: dict[str, dict] = {}
    source_counts: dict[str, int] = {}

    def page_key(raw_key, payload: dict) -> str:
        value = payload.get("page_no", raw_key)
        try:
            return str(int(value))
        except (TypeError, ValueError):
            return str(raw_key)

    def as_box(item) -> list[int] | None:
        if not isinstance(item, (list, tuple)) or len(item) < 4:
            return None
        try:
            box = [int(round(float(v))) for v in item[:4]]
        except (TypeError, ValueError):
            return None
        return box if box[2] > box[0] and box[3] > box[1] else None

    def box_iou(a: list[int], b: list[int]) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        aa = max(1, (a[2] - a[0]) * (a[3] - a[1]))
        ba = max(1, (b[2] - b[0]) * (b[3] - b[1]))
        return inter / max(1, aa + ba - inter)

    def append_unique_box(target: list[list[int]], box: list[int]) -> None:
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        for old in target:
            ocx, ocy = (old[0] + old[2]) / 2.0, (old[1] + old[3]) / 2.0
            if box_iou(old, box) >= 0.72 or (abs(cx - ocx) <= 3 and abs(cy - ocy) <= 3):
                return
        target.append(box)

    def normalise_column(raw: dict) -> dict:
        boxes: list[list[int]] = []
        for item in raw.get("ruby_candidate_boxes", []) or []:
            box = as_box(item)
            if box is not None:
                append_unique_box(boxes, box)
        return {
            "left": int(raw.get("left", 0) or 0),
            "top": int(raw.get("top", 0) or 0),
            "right": int(raw.get("right", 0) or 0),
            "bottom": int(raw.get("bottom", 0) or 0),
            "hard_left": int(raw.get("hard_left", raw.get("left", 0)) or 0),
            "hard_right": int(raw.get("hard_right", raw.get("right", 0)) or 0),
            "ruby_candidate_boxes": boxes,
            "ruby_candidate_confidence": float(raw.get("ruby_candidate_confidence", 0.0) or 0.0),
        }

    def column_match_score(a: dict, b: dict) -> float:
        try:
            al, ar = float(a["left"]), float(a["right"])
            bl, br = float(b["left"]), float(b["right"])
            at, ab = float(a.get("top", 0)), float(a.get("bottom", 0))
            bt, bb = float(b.get("top", 0)), float(b.get("bottom", 0))
        except (TypeError, ValueError, KeyError):
            return 0.0
        aw, bw = max(1.0, ar - al), max(1.0, br - bl)
        ax, bx = (al + ar) / 2.0, (bl + br) / 2.0
        overlap_x = max(0.0, min(ar, br) - max(al, bl)) / max(1.0, min(aw, bw))
        centre = max(0.0, 1.0 - abs(ax - bx) / max(18.0, 1.8 * max(aw, bw)))
        ah, bh = max(1.0, ab - at), max(1.0, bb - bt)
        overlap_y = max(0.0, min(ab, bb) - max(at, bt)) / max(1.0, min(ah, bh))
        return 0.62 * max(overlap_x, centre) + 0.38 * overlap_y

    for doc in documents:
        if doc is None:
            continue
        pages = getattr(getattr(doc, "metadata", None), "ruby_candidate_pages", {}) or {}
        if not isinstance(pages, dict):
            continue
        for raw_key, payload in pages.items():
            if not isinstance(payload, dict):
                continue
            key = page_key(raw_key, payload)
            raw_columns = payload.get("columns", []) or []
            if not isinstance(raw_columns, list):
                continue
            columns = [normalise_column(raw) for raw in raw_columns if isinstance(raw, dict)]
            if key not in merged:
                merged[key] = {
                    "schema": "novel-formatter-ruby-candidates-v3-geometry-matched",
                    "page_no": int(payload.get("page_no", key) or int(key) if key.isdigit() else 0),
                    "page_path": str(payload.get("page_path") or ""),
                    "columns": columns,
                }
                source_counts[key] = 1
                continue

            target = merged[key]
            target_columns = target.get("columns", [])
            for raw in columns:
                # Find the same *physical* column.  A moderate threshold allows
                # slightly different crop widths while rejecting neighbour
                # columns when one engine inserted/omitted a split.
                ranked = sorted(
                    ((column_match_score(dst, raw), idx, dst) for idx, dst in enumerate(target_columns)),
                    key=lambda item: (-item[0], item[1]),
                )
                if ranked and ranked[0][0] >= 0.52:
                    dst = ranked[0][2]
                    for box in raw.get("ruby_candidate_boxes", []) or []:
                        append_unique_box(dst["ruby_candidate_boxes"], box)
                    dst["ruby_candidate_confidence"] = max(
                        float(dst.get("ruby_candidate_confidence", 0.0) or 0.0),
                        float(raw.get("ruby_candidate_confidence", 0.0) or 0.0),
                    )
                elif raw.get("ruby_candidate_boxes"):
                    # Keep an unmatched candidate-bearing column rather than
                    # forcing it into a neighbour.  Candidate-free alternate
                    # segmentation adds no Ruby evidence and is ignored.
                    target_columns.append(raw)

            # Restore deterministic Japanese vertical reading order (right to
            # left).  ROI neighbour selection can now safely use list indices.
            target_columns.sort(
                key=lambda c: (-(float(c.get("left", 0)) + float(c.get("right", 0))) / 2.0,
                               float(c.get("top", 0)))
            )
            if not target.get("page_path") and payload.get("page_path"):
                target["page_path"] = str(payload.get("page_path"))
            source_counts[key] = source_counts.get(key, 1) + 1

    for key, payload in merged.items():
        columns = payload.get("columns", []) or []
        payload["ruby_candidate_summary"] = {
            "columns": sum(1 for column in columns if column.get("ruby_candidate_boxes")),
            "boxes": sum(len(column.get("ruby_candidate_boxes", []) or []) for column in columns),
            "max_confidence": max(
                (float(column.get("ruby_candidate_confidence", 0.0) or 0.0) for column in columns),
                default=0.0,
            ),
            "geometry_sources": int(source_counts.get(key, 1)),
            "merge_strategy": "original_page_geometry",
        }
    return merged


def detect_ruby_lines(
    document: UnifiedDocument,
    *,
    cancel_check=None,
    log_callback=None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    batch_size: int = 16,
    runner: Callable[[int, str], dict] | None = None,
    scan_mode: str = "auto",
    candidate_root: str | Path | None = None,
    candidate_payloads: dict | None = None,
    roi_neighbor_columns: int = 1,
    roi_max_columns: int = 10,
    diagnostics: dict | None = None,
) -> tuple[list[RubyLine], int]:
    """Return structured Ruby lines without making findtextCenterNet scan blindly.

    Modes:
      ``smart_roi`` -- consume geometry recorded by normal OCR and OCR only
      original-page context ROIs. Pages without candidates are not scanned.
      ``full_page`` -- legacy diagnostic/maximum-recall path.
      ``auto`` -- smart ROI when geometry exists, otherwise legacy full-page.

    ``runner`` remains a test/embedding hook and receives the exact image that
    findtextCenterNet would receive (full page or an original-page ROI crop).
    """
    page_images = _text_page_images(document)
    if diagnostics is not None:
        diagnostics.clear()
    if not page_images:
        if diagnostics is not None:
            diagnostics.update({"scan_mode": scan_mode, "text_pages": 0})
        return [], 0

    document_candidate_payloads = getattr(document.metadata, "ruby_candidate_pages", {}) or {}
    if candidate_payloads is None:
        candidate_payloads = document_candidate_payloads
    elif not isinstance(candidate_payloads, dict):
        candidate_payloads = {}
    requested = str(scan_mode or "auto").strip().lower()
    if requested not in {"auto", "smart_roi", "full_page"}:
        requested = "auto"
    has_geometry = bool(candidate_payloads) or bool(candidate_root)
    mode = "smart_roi" if requested == "auto" and has_geometry else (
        "full_page" if requested == "auto" else requested
    )

    staged_specs: list[tuple[int, str, int, int, str]] = []
    plans = []
    full_page_tiles = 0
    page_area_total = 0
    if mode == "smart_roi":
        plans = build_ruby_roi_plans(
            page_images, candidate_root=candidate_root,
            candidate_payloads=candidate_payloads,
            neighbor_columns=roi_neighbor_columns,
            max_columns_per_roi=roi_max_columns,
        )
        full_page_tiles = sum(plan.full_page_detector_tiles for plan in plans)
        page_area_total = sum(plan.page_width * plan.page_height for plan in plans)
        roi_tiles = sum(plan.estimated_detector_tiles for plan in plans)
        roi_area = sum(plan.roi_area for plan in plans)
        candidate_boxes = sum(plan.candidate_boxes for plan in plans)
        pages_with_candidates = sum(1 for plan in plans if plan.rois)
        roi_count = sum(len(plan.rois) for plan in plans)
        if diagnostics is not None:
            diagnostics.update({
                "scan_mode": mode,
                "text_pages": len(page_images),
                "pages_with_candidates": pages_with_candidates,
                "candidate_boxes": candidate_boxes,
                "roi_count": roi_count,
                "roi_area": roi_area,
                "page_area": page_area_total,
                "roi_coverage_ratio": roi_area / max(1, page_area_total),
                "estimated_detector_tiles": roi_tiles,
                "full_page_detector_tiles": full_page_tiles,
                "estimated_tile_ratio": roi_tiles / max(1, full_page_tiles),
            })
        _emit(
            log_callback,
            f"🔎 Ruby 智能 ROI：{len(page_images)} 个正文页中 {pages_with_candidates} 页有候选，"
            f"合并为 {roi_count} 个上下文框；预计 findtext 检测窗 {roi_tiles}/{max(1, full_page_tiles)}。"
        )
        if not any(plan.rois for plan in plans):
            _emit(log_callback, "ℹ️ 普通 OCR 分列阶段未记录 Ruby 候选；智能模式不执行任何整页 findtext OCR。")
            return [], 0
    elif diagnostics is not None:
        diagnostics.update({
            "scan_mode": mode, "text_pages": len(page_images),
            "pages_with_candidates": len(page_images), "roi_count": 0,
        })

    # Preserve the original embedding/test hook contract in legacy full-page
    # mode: callers receive the authoritative source image path, not a staged
    # hardlink. Smart ROI mode intentionally receives generated ROI crops.
    if runner is not None and mode == "full_page":
        lines: list[RubyLine] = []
        for index, (page_no, image_path) in enumerate(page_images, start=1):
            if callable(cancel_check) and cancel_check():
                raise InterruptedError("Ruby 识别已停止")
            payload = runner(page_no, image_path) or {}
            lines.extend(_parse_payload(page_no, payload))
            if progress_callback:
                progress_callback(index, len(page_images), Path(image_path).name)
        if diagnostics is not None:
            diagnostics.update({
                "scan_mode": "full_page", "text_pages": len(page_images),
                "pages_with_candidates": len(page_images), "roi_count": 0,
            })
        return _dedupe_ruby_lines(lines), len(page_images)

    all_lines: list[RubyLine] = []
    with tempfile.TemporaryDirectory(prefix="novel-ruby-findtext-") as tmp:
        tmpdir = Path(tmp)
        if mode == "smart_roi":
            serial = 0
            for plan in plans:
                if not plan.rois:
                    continue
                try:
                    page_image = Image.open(plan.page_path).convert("RGB")
                except Exception as exc:
                    _emit(log_callback, f"⚠️ Ruby ROI 无法读取第 {plan.page_no} 页原图：{exc}")
                    continue
                try:
                    for roi_index, roi in enumerate(plan.rois, start=1):
                        serial += 1
                        target = tmpdir / f"p{plan.page_no:05d}_r{roi_index:03d}_{serial:05d}.png"
                        crop = page_image.crop((roi.x1, roi.y1, roi.x2, roi.y2))
                        crop.save(target, format="PNG", compress_level=1)
                        crop.close()
                        staged_specs.append((
                            plan.page_no, str(target), int(roi.x1), int(roi.y1),
                            f"{Path(plan.page_path).name} · ROI {roi_index}/{len(plan.rois)}",
                        ))
                finally:
                    page_image.close()
        else:
            for serial, (page_no, image_path) in enumerate(page_images, start=1):
                source = Path(image_path)
                suffix = source.suffix if source.suffix else ".png"
                target = tmpdir / f"p{page_no:05d}_{serial:05d}{suffix}"
                _link_or_copy(source, target)
                staged_specs.append((page_no, str(target), 0, 0, source.name))

        if not staged_specs:
            return [], 0

        if runner is not None:
            for index, (page_no, image_path, offset_x, offset_y, display_name) in enumerate(staged_specs, start=1):
                if callable(cancel_check) and cancel_check():
                    raise InterruptedError("Ruby 识别已停止")
                payload = runner(page_no, image_path) or {}
                all_lines.extend(_parse_payload(
                    page_no, payload, offset_x=offset_x, offset_y=offset_y
                ))
                if progress_callback:
                    progress_callback(index, len(staged_specs), display_name)
        else:
            # Persistent ROI cache: re-running Fusion/AI roundtrips must not
            # re-OCR unchanged page pixels.  The key contains source image
            # digest + exact ROI coordinates + pinned upstream/runtime identity.
            cache = RubyResultCache(_cache_dir())
            page_digests: dict[str, str] = {}
            cache_keys: dict[str, str] = {}
            cache_source = _source_dir()
            if _ready_backend(cache_source):
                cache_runtime_id = runtime_fingerprint(
                    cache_source, upstream_commit=UPSTREAM_COMMIT
                )
            else:
                cache_runtime_id = CACHE_RUNTIME_ID
            cache_hits = 0
            cache_misses = 0
            failed_rois = 0
            pending_specs = []

            # Map staged files back to the authoritative source pixels.  Smart
            # ROI filenames encode the crop but we retain explicit metadata so
            # caching never depends on a temporary path/name.
            stage_sources: dict[str, tuple[str, tuple[int, int, int, int]]] = {}
            # Build a robust mapping directly from stage order and ROI order.
            flat_origins: list[tuple[str, tuple[int, int, int, int]]] = []
            if mode == "smart_roi":
                for plan in plans:
                    for roi in plan.rois:
                        flat_origins.append((plan.page_path, (roi.x1, roi.y1, roi.x2, roi.y2)))
            else:
                for _page_no, image_path in page_images:
                    flat_origins.append((image_path, (0, 0, 0, 0)))
            for spec, origin in zip(staged_specs, flat_origins):
                stage_sources[spec[1]] = origin

            for spec in staged_specs:
                page_no, staged_path, offset_x, offset_y, display_name = spec
                source_path, roi_box = stage_sources.get(staged_path, (staged_path, (0, 0, 0, 0)))
                try:
                    digest = page_digests.get(source_path)
                    if digest is None:
                        digest = file_sha256(source_path)
                        page_digests[source_path] = digest
                    key = make_cache_key(page_digest=digest, roi_box=roi_box, runtime_id=cache_runtime_id)
                    cache_keys[staged_path] = key
                    payload = cache.get(key)
                except Exception:
                    payload = None
                if isinstance(payload, dict):
                    all_lines.extend(_parse_payload(
                        page_no, payload, offset_x=offset_x, offset_y=offset_y
                    ))
                    cache_hits += 1
                    if progress_callback:
                        progress_callback(cache_hits, len(staged_specs), display_name + " · cache")
                else:
                    pending_specs.append(spec)
                    cache_misses += 1

            if diagnostics is not None:
                diagnostics["cache_hits"] = cache_hits
                diagnostics["cache_misses"] = cache_misses

            if pending_specs:
                source_dir, python = prepare_runtime(log_callback=log_callback)
                # Model files now exist.  Re-key pending entries with a cheap
                # model stat fingerprint so manually updated weights cannot
                # accidentally reuse stale Ruby JSON.  Check the final key once
                # more before inference (useful after runtime restoration).
                final_runtime_id = runtime_fingerprint(
                    source_dir, upstream_commit=UPSTREAM_COMMIT
                )
                if final_runtime_id != cache_runtime_id:
                    still_pending = []
                    for spec in pending_specs:
                        page_no, staged_path, offset_x, offset_y, display_name = spec
                        source_path, roi_box = stage_sources.get(
                            staged_path, (staged_path, (0, 0, 0, 0))
                        )
                        digest = page_digests[source_path]
                        key = make_cache_key(
                            page_digest=digest, roi_box=roi_box,
                            runtime_id=final_runtime_id,
                        )
                        cache_keys[staged_path] = key
                        payload = cache.get(key)
                        if isinstance(payload, dict):
                            all_lines.extend(_parse_payload(
                                page_no, payload, offset_x=offset_x, offset_y=offset_y
                            ))
                            cache_hits += 1
                            cache_misses = max(0, cache_misses - 1)
                            if progress_callback:
                                progress_callback(
                                    cache_hits, len(staged_specs), display_name + " · cache"
                                )
                        else:
                            still_pending.append(spec)
                    pending_specs = still_pending
                    if diagnostics is not None:
                        diagnostics["cache_hits"] = cache_hits
                        diagnostics["cache_misses"] = cache_misses
                # Upstream-native persistent worker: import the original
                # run_ocr.py once, then reuse its processer for the complete
                # Ruby pass.  This keeps Novel Formatter out of Detector /
                # Transformer internals and avoids reloading ~1.5 GB of weights
                # for every small ROI batch.
                batch_size = max(1, min(64, int(batch_size or 16)))
                completed = cache_hits

                def consume_payload(spec, payload: dict | None, error: str | None = None) -> bool:
                    nonlocal completed, failed_rois
                    page_no, staged_path, offset_x, offset_y, display_name = spec
                    if not isinstance(payload, dict):
                        failed_rois += 1
                        completed += 1
                        _emit(log_callback, f"⚠️ 跳过失败 Ruby ROI：{display_name}：{error or '上游未返回有效 JSON'}")
                        if progress_callback:
                            progress_callback(completed, len(staged_specs), display_name + " · skipped")
                        return False
                    all_lines.extend(_parse_payload(
                        page_no, payload, offset_x=offset_x, offset_y=offset_y
                    ))
                    key = cache_keys.get(staged_path)
                    if key:
                        try:
                            cache.put(key, payload)
                        except Exception:
                            pass
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, len(staged_specs), display_name)
                    return True

                worker_failed = False
                try:
                    with FindtextCenterNetSession(
                        source_dir, python,
                        cancel_check=cancel_check,
                        log_callback=log_callback,
                    ) as session:
                        for start in range(0, len(pending_specs), batch_size):
                            if callable(cancel_check) and cancel_check():
                                raise InterruptedError("Ruby 识别已停止")
                            batch = pending_specs[start:start + batch_size]
                            returned = session.recognize([item[1] for item in batch])
                            for spec in batch:
                                staged_path = str(Path(spec[1]).resolve())
                                payload, error = returned.get(
                                    staged_path,
                                    returned.get(spec[1], (None, "上游 worker 未返回该 ROI")),
                                )
                                consume_payload(spec, payload, error)
                except InterruptedError:
                    raise
                except Exception as exc:
                    worker_failed = True
                    _emit(
                        log_callback,
                        "⚠️ findtextCenterNet 长驻上游 worker 异常；仅对尚未完成 ROI 回退原项目 run_ocr.py 单图调用："
                        + str(exc),
                    )

                if worker_failed:
                    # Compatibility fallback uses the upstream CLI exactly as
                    # documented.  It is intentionally slower because each call
                    # reloads the model, but Ruby is optional and must fail open.
                    already_done = completed - cache_hits
                    remaining = pending_specs[max(0, already_done):]
                    for spec in remaining:
                        if callable(cancel_check) and cancel_check():
                            raise InterruptedError("Ruby 识别已停止")
                        page_no, staged_path, _ox, _oy, display_name = spec
                        result_path = Path(staged_path + ".json")
                        result_path.unlink(missing_ok=True)
                        one = [str(python), "run_ocr.py", staged_path]
                        one_code, one_stdout, one_stderr = _run_process(
                            one, cwd=source_dir, cancel_check=cancel_check
                        )
                        if one_code == 0 and result_path.is_file():
                            try:
                                payload = json.loads(result_path.read_text(encoding="utf-8"))
                            except Exception as json_exc:
                                payload = None
                                error = f"上游 JSON 无法解析：{json_exc}"
                            else:
                                error = None
                            finally:
                                result_path.unlink(missing_ok=True)
                            consume_payload(spec, payload, error)
                            continue
                        detail = " | ".join(
                            (one_stderr or one_stdout or "无输出").strip().splitlines()[-4:]
                        )
                        consume_payload(spec, None, detail or "上游单图调用失败")

                try:
                    cache.prune()
                except Exception:
                    pass
                if failed_rois >= len(pending_specs) and not all_lines:
                    raise RuntimeError("findtextCenterNet Ruby ROI 全部失败；正文 OCR 未受影响")

            if diagnostics is not None:
                diagnostics["failed_rois"] = failed_rois

    lines = _dedupe_ruby_lines(all_lines)
    pages_scanned = len({page_no for page_no, *_rest in staged_specs})
    if diagnostics is not None and mode == "full_page":
        # Best-effort cost estimate for diagnostics only; recognition does not
        # depend on opening the image here.
        tiles = 0
        area = 0
        for _page_no, image_path in page_images:
            try:
                with Image.open(image_path) as image:
                    w, h = image.size
                tiles += estimate_findtext_tiles(w, h)
                area += w * h
            except Exception:
                pass
        diagnostics.update({
            "pages_with_candidates": len(page_images),
            "candidate_boxes": 0, "roi_count": 0,
            "roi_area": area, "page_area": area, "roi_coverage_ratio": 1.0,
            "estimated_detector_tiles": tiles, "full_page_detector_tiles": tiles,
            "estimated_tile_ratio": 1.0 if tiles else 0.0,
        })
    return lines, pages_scanned


def _block_score(block_text: str, line: RubyLine) -> float:
    block_norm = _normalise_context(block_text)
    plain_norm = _normalise_context(line.plain)
    if not block_norm or not plain_norm:
        return 0.0
    if plain_norm in block_norm:
        return 1.0 + min(0.1, len(plain_norm) / 1000.0)
    if block_norm in plain_norm and len(block_norm) >= 4:
        return 0.94
    if min(len(block_norm), len(plain_norm)) < 4:
        return 0.0
    return SequenceMatcher(None, block_norm, plain_norm, autojunk=False).ratio() * 0.80


def _page_pixel_size(document: UnifiedDocument, page_no: int) -> tuple[int, int]:
    for page in document.pages:
        if int(page.page_no or 0) != int(page_no):
            continue
        width = int(getattr(page, "width", 0) or 0)
        height = int(getattr(page, "height", 0) or 0)
        if width > 0 and height > 0:
            return width, height
        path = str(getattr(page, "image_path", "") or "")
        if path and Path(path).exists():
            try:
                with Image.open(path) as image:
                    return image.size
            except Exception:
                pass
    return 0, 0


def _block_pixel_box(document: UnifiedDocument, block) -> tuple[float, float, float, float] | None:
    """Return authoritative block geometry in original-page pixels when available."""
    page_no = int(block.page or block.page_index or block.page_number or 0)
    width, height = _page_pixel_size(document, page_no)
    bbox = getattr(block, "bbox", None)
    if bbox is not None and width > 0 and height > 0:
        try:
            x1 = float(bbox.x) * width
            y1 = float(bbox.y) * height
            x2 = float(bbox.x + bbox.w) * width
            y2 = float(bbox.y + bbox.h) * height
            if x2 > x1 and y2 > y1:
                return x1, y1, x2, y2
        except (TypeError, ValueError):
            pass
    metadata = block.metadata if isinstance(getattr(block, "metadata", None), dict) else {}
    try:
        x1 = float(metadata.get("column_left"))
        y1 = float(metadata.get("column_top"))
        x2 = float(metadata.get("column_right"))
        y2 = float(metadata.get("column_bottom"))
        if x2 > x1 and y2 > y1:
            return x1, y1, x2, y2
    except (TypeError, ValueError):
        pass
    return None


def _spatial_alignment_score(
    line: RubyLine, block_box: tuple[float, float, float, float] | None,
) -> float:
    """Score how well a findtext vertical line overlaps one OCR block.

    Text remains the primary criterion.  Geometry is a tie-break/guard that
    prevents a Ruby reading from being attached to an identical word in another
    physical column on the same page.
    """
    if block_box is None or line.x2 <= line.x1 or line.y2 <= line.y1:
        return 0.0
    lx1, ly1, lx2, ly2 = line.x1, line.y1, line.x2, line.y2
    bx1, by1, bx2, by2 = block_box
    lw, lh = max(1.0, lx2 - lx1), max(1.0, ly2 - ly1)
    bw, bh = max(1.0, bx2 - bx1), max(1.0, by2 - by1)
    overlap_x = max(0.0, min(lx2, bx2) - max(lx1, bx1))
    overlap_y = max(0.0, min(ly2, by2) - max(ly1, by1))
    x_ratio = overlap_x / max(1.0, min(lw, bw))
    y_ratio = overlap_y / max(1.0, min(lh, bh))
    lcx, bcx = (lx1 + lx2) / 2.0, (bx1 + bx2) / 2.0
    centre_dx = abs(lcx - bcx)
    x_near = max(0.0, 1.0 - centre_dx / max(24.0, 2.5 * max(lw, bw)))
    # Vertical Japanese lines are narrow in X and long in Y: X correspondence
    # is more discriminative than exact line height.
    return min(1.0, 0.58 * max(x_ratio, x_near) + 0.42 * y_ratio)


def _safe_inject(
    source: str, line: RubyLine, *, allow_base_correction: bool = False,
) -> tuple[str, int]:
    """Inject markers without changing prose characters.

    Exact matching remains the default.  When the surrounding findtext line is
    already a strong match to the same physical OCR column, an optional second
    stage may map a *small local OCR edit* inside the Ruby base (substitution,
    missing glyph/kana, or one extra glyph; for example ``椅于 -> 椅子`` or
    ``水姿見 -> 水の姿見``).  The marker wraps the
    authoritative OCR spelling that is already present; no prose character is
    ever replaced here.  Later AI correction can migrate the locked reading to
    the corrected base through the same edit-span map.
    """
    if not source:
        return source, 0
    if line.aozora in source:
        return source, len(line.pairs)
    if line.plain and source.count(line.plain) == 1:
        return source.replace(line.plain, line.aozora, 1), len(line.pairs)

    # If at least one Ruby base itself differs by a safely mappable OCR glyph,
    # resolve *all* pairs against the original plain source before inserting any
    # markers.  This avoids the first exact pair changing offsets for a later
    # typo-corrected pair on the same line.
    if allow_base_correction and line.plain and line.aozora:
        annotations = _annotations_from_marked(line.aozora, line.plain)
        placements: list[tuple[int, int, str]] = []
        used: list[tuple[int, int]] = []
        saw_migration = False
        for annotation in annotations:
            resolved = resolve_annotation_for_text(
                source, annotation, used, source_text=line.plain,
                require_base_evidence=False,
            )
            if not resolved:
                continue
            migrated = resolved["annotation"]
            pos = int(resolved["position"])
            base_now = str(migrated.get("base", "") or "")
            reading = str(migrated.get("reading", "") or "")
            if not base_now or not reading:
                continue
            end = pos + len(base_now)
            used.append((pos, end))
            placements.append((pos, end, f"｜{base_now}《{reading}》"))
            saw_migration = saw_migration or str(resolved.get("mode", "")) == "base_edit_span"
        if saw_migration:
            out = source
            for start, end, marker in sorted(placements, key=lambda item: item[0], reverse=True):
                out = out[:start] + marker + out[end:]
            return out, len(placements)

    out = source
    inserted = 0
    for base, reading in line.pairs:
        marked = f"｜{base}《{reading}》"
        if marked in out:
            inserted += 1
            continue
        # Never guess between repeated kanji occurrences.  This is the guard
        # that prevents the optional Ruby pass from corrupting main OCR prose.
        if out.count(base) != 1:
            continue
        out = out.replace(base, marked, 1)
        inserted += 1
    return out, inserted

def _annotations_from_marked(marked: str, plain: str, *, context_chars: int = 18) -> list[dict]:
    """Build immutable Ruby anchors without changing authoritative prose.

    ``marked`` is guaranteed to be the same prose as ``plain`` plus Aozora
    markers.  Scanning it therefore yields the exact occurrence offset even
    when the same base appears multiple times in one block.
    """
    annotations: list[dict] = []
    marked_cursor = 0
    plain_cursor = 0
    for match in RUBY_RE.finditer(marked):
        between = marked[marked_cursor:match.start()]
        # Between complete Ruby markers there should be ordinary prose only;
        # the fallback substitution keeps this robust to legacy nested strings.
        between_plain = RUBY_RE.sub(lambda m: m.group(1), between)
        plain_cursor += len(between_plain)
        base, reading = match.group(1), match.group(2)
        start = plain_cursor
        end = start + len(base)
        # Trust the position only if it maps back to the actual authoritative
        # text.  Otherwise retain the pair but omit positional anchors.
        anchored = plain[start:end] == base
        record = {"base": base, "reading": reading}
        if anchored:
            record.update({
                "source_offset": start,
                "source_occurrence": plain[:start].count(base),
                "left_context": plain[max(0, start - context_chars):start],
                "right_context": plain[end:end + context_chars],
                "anchor_version": 1,
            })
        annotations.append(record)
        plain_cursor = end
        marked_cursor = match.end()
    return annotations


def _annotation_anchor_key(annotation: dict) -> tuple:
    return (
        str(annotation.get("base", "") or ""),
        str(annotation.get("reading", "") or ""),
        annotation.get("source_offset"),
    )


def _merge_findtext_annotation_evidence(
    previous: list[dict], generated: list[dict], line: RubyLine,
) -> list[dict]:
    """Preserve old evidence and attach current findtext bases only to new markers.

    ``apply_ruby_lines`` may add several findtext lines to the same OCR block.
    Rebuilding annotations from the whole marked block on every line used to make
    a same-reading pair from an earlier line eligible for evidence belonging to
    the current line.  That can later authorize a wrong base migration.

    Existing anchors are therefore merged back by stable plain-text position,
    and only annotations newly introduced by this line consume ``line.pairs``.
    Reading order is used as a secondary discriminator; if it is ambiguous we
    leave correction evidence unset and remain fail-closed.
    """
    previous_by_key = {
        _annotation_anchor_key(item): copy.deepcopy(item)
        for item in (previous or []) if isinstance(item, dict)
    }
    out: list[dict] = []
    new_indices: list[int] = []
    for generated_item in generated:
        item = copy.deepcopy(generated_item)
        old = previous_by_key.get(_annotation_anchor_key(item))
        if old is not None:
            merged = old
            # Fresh offset/context fields reflect the authoritative current prose;
            # richer findtext/migration evidence from the old record survives.
            merged.update(item)
            out.append(merged)
        else:
            new_indices.append(len(out))
            out.append(item)

    source_annotations = _annotations_from_marked(line.aozora, line.plain)
    unmatched = set(range(len(source_annotations)))
    last_source_index = -1
    for out_index in new_indices:
        item = out[out_index]
        reading = str(item.get("reading", "") or "")
        base_now = str(item.get("base", "") or "")
        if not reading:
            continue
        candidates = [
            i for i in sorted(unmatched)
            if i > last_source_index
            and str(source_annotations[i].get("reading", "") or "") == reading
        ]
        if not candidates:
            candidates = [
                i for i in sorted(unmatched)
                if str(source_annotations[i].get("reading", "") or "") == reading
            ]
        if not candidates:
            continue

        # Prefer an exact base, otherwise require a single ordered candidate.
        exact = [
            i for i in candidates
            if str(source_annotations[i].get("base", "") or "") == base_now
        ]
        if len(exact) == 1:
            chosen = exact[0]
        elif len(candidates) == 1:
            chosen = candidates[0]
        else:
            # Multiple same-reading candidates are unsafe without another
            # discriminator.  The immutable reading remains, but no base-edit
            # authorization is recorded.
            continue
        unmatched.discard(chosen)
        last_source_index = chosen
        detected_base = str(source_annotations[chosen].get("base", "") or "")
        if not detected_base:
            continue
        item["findtext_detected_base"] = detected_base
        item["findtext_detected_reading"] = reading
        item["findtext_page"] = int(line.page or 0)
        if chosen < len(line.pair_boxes):
            item["findtext_pair_bbox"] = [round(float(v), 3) for v in line.pair_boxes[chosen]]
        if detected_base != base_now:
            item["base_correction_candidates"] = [detected_base]
            evidence = ["findtextCenterNet"]
            if item.get("findtext_pair_bbox"):
                evidence.append("original_page_pair_bbox")
            item["base_correction_evidence"] = evidence
    return out


def _all_occurrences(text: str, needle: str) -> list[int]:
    if not needle:
        return []
    out: list[int] = []
    pos = 0
    while True:
        found = text.find(needle, pos)
        if found < 0:
            break
        out.append(found)
        pos = found + max(1, len(needle))
    return out


def _resolve_annotation_position(
    current: str, annotation: dict, used: list[tuple[int, int]],
) -> int | None:
    # Keep the historical private helper/API while delegating exact matching to
    # the central dependency-free anchor module.  Base-spelling migration needs
    # the old revision text and is intentionally handled by apply_ruby_overlay.
    return resolve_exact(current, annotation, used)


def _render_locked_annotations(current: str, annotations: list[dict]) -> tuple[str, int]:
    """Render only uniquely resolvable annotations, right-to-left."""
    placements: list[tuple[int, int, str]] = []
    used: list[tuple[int, int]] = []
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        base = str(annotation.get("base", "") or "")
        reading = str(annotation.get("reading", "") or "")
        if not base or not reading:
            continue
        pos = _resolve_annotation_position(current, annotation, used)
        if pos is None:
            continue
        end = pos + len(base)
        used.append((pos, end))
        placements.append((pos, end, f"｜{base}《{reading}》"))
    marked = current
    for start, end, marker in sorted(placements, key=lambda item: item[0], reverse=True):
        marked = marked[:start] + marker + marked[end:]
    return marked, len(placements)


def apply_ruby_lines(document: UnifiedDocument, lines: Iterable[RubyLine]) -> RubyPreservationReport:
    grouped: dict[int, list[RubyLine]] = {}
    for line in lines:
        grouped.setdefault(int(line.page), []).append(line)
    ruby_lines = sum(len(items) for items in grouped.values())
    ruby_pairs = sum(len(line.pairs) for items in grouped.values() for line in items)
    matched_lines = matched_pairs = updated_blocks = 0

    for page_no, page_lines in grouped.items():
        candidates = [
            block for block in document.blocks
            if int(block.page or block.page_index or block.page_number or 0) == page_no
            and block.type in _TEXT_TYPES
            and str(block.text or "").strip()
        ]
        if not candidates:
            continue
        for line in page_lines:
            ranked = []
            for index, block in enumerate(candidates):
                text_score = _block_score(str(block.text or ""), line)
                spatial_score = _spatial_alignment_score(
                    line, _block_pixel_box(document, block)
                )
                # Text is authoritative; page geometry only resolves physically
                # distinct candidates and must never rescue unrelated prose.
                combined = text_score + (0.18 * spatial_score if text_score >= 0.55 else 0.0)
                ranked.append((combined, text_score, spatial_score, index, block))
            ranked.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
            if not ranked or ranked[0][1] < 0.55:
                continue
            _combined_score, text_score, spatial_score, _index, block = ranked[0]
            metadata = block.metadata if isinstance(block.metadata, dict) else {}
            # Ruby is a metadata overlay only.  Never consume/replace ``ocr_raw``:
            # that field remains evidence from the ordinary OCR/fusion channel.
            existing = str(metadata.get("ruby_aozora") or block.text or "")
            # Base-spelling recovery is permitted only after the ordinary OCR
            # block has already won a strong text match.  Original-page geometry
            # strengthens the decision when char/pair boxes are available.
            allow_base_correction = bool(
                text_score >= 0.70
                and (spatial_score >= 0.35 or not line.pair_boxes)
            )
            marked, inserted = _safe_inject(
                existing, line, allow_base_correction=allow_base_correction,
            )
            if inserted <= 0 or marked == existing:
                # Existing markers still count as matched, but don't increment
                # updated_blocks twice when multiple lines target one block.
                if inserted > 0:
                    matched_lines += 1
                    matched_pairs += inserted
                continue
            was_marked = bool(RUBY_RE.search(str(metadata.get("ruby_aozora") or "")))
            metadata = dict(metadata)
            metadata["ruby_preserved"] = True
            metadata["ruby_source"] = "findtextCenterNet"
            metadata["ruby_aozora"] = marked
            previous_annotations = [
                copy.deepcopy(item)
                for item in (metadata.get("ruby_annotations") or [])
                if isinstance(item, dict)
            ]
            generated_annotations = _annotations_from_marked(
                marked, str(block.text or "")
            )
            generated_annotations = _merge_findtext_annotation_evidence(
                previous_annotations, generated_annotations, line,
            )
            metadata["ruby_annotations"] = generated_annotations
            metadata["ruby_annotation_schema"] = "locked-context-v4-position-map"
            metadata["ruby_pair_count"] = len(metadata["ruby_annotations"])
            metadata["ruby_structure_locked"] = True
            metadata["ruby_overlay_only"] = True
            metadata["ruby_original_block_type"] = str(getattr(block.type, "value", block.type))
            evidence = list(metadata.get("ruby_alignment_evidence") or [])
            block_box = _block_pixel_box(document, block)
            evidence.append({
                "method": "text_plus_original_page_geometry" if block_box else "text_only",
                "line_bbox": [round(line.x1, 2), round(line.y1, 2), round(line.x2, 2), round(line.y2, 2)],
                "pair_bboxes": [[round(v, 2) for v in box] for box in line.pair_boxes],
                "block_bbox": [round(v, 2) for v in block_box] if block_box else [],
                "text_score": round(float(text_score), 6),
                "spatial_score": round(float(spatial_score), 6),
            })
            metadata["ruby_alignment_evidence"] = evidence
            block.metadata = metadata
            # Intentionally do not touch block.text, block.ocr_raw or block.type.
            # Ordinary OCR evidence and semantic block classification stay pure.
            matched_lines += 1
            matched_pairs += inserted
            if not was_marked:
                updated_blocks += 1

    return RubyPreservationReport(
        enabled=True,
        ruby_lines=ruby_lines,
        ruby_pairs=ruby_pairs,
        matched_lines=matched_lines,
        matched_pairs=matched_pairs,
        updated_blocks=updated_blocks,
        unmatched_pairs=max(0, ruby_pairs - matched_pairs),
    )



def _metadata_ruby_pairs(metadata: dict) -> list[tuple[str, str]]:
    """Return immutable Ruby pairs stored on a block, with legacy fallback."""
    raw = metadata.get("ruby_annotations") if isinstance(metadata, dict) else None
    pairs: list[tuple[str, str]] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            base = str(item.get("base", "") or "")
            reading = str(item.get("reading", "") or "")
            if base and reading:
                pairs.append((base, reading))
    if pairs:
        return pairs
    aozora = str((metadata or {}).get("ruby_aozora", "") or "")
    return [(m.group(1), m.group(2)) for m in RUBY_RE.finditer(aozora)]


def refresh_preserved_ruby(document: UnifiedDocument) -> RubyPreservationReport:
    """Re-attach locked Ruby readings to the current authoritative prose.

    Repeated bases are supported only when their immutable surrounding-context
    anchors uniquely identify the original occurrence.  Legacy annotations
    without anchors retain the old conservative rule: a base must occur once.
    ``block.text`` is never changed.
    """
    matched_pairs = updated_blocks = unmatched_pairs = 0
    for block in document.blocks:
        metadata = block.metadata if isinstance(block.metadata, dict) else {}
        raw_annotations = metadata.get("ruby_annotations") if isinstance(metadata, dict) else None
        annotations: list[dict] = []
        if isinstance(raw_annotations, list):
            for item in raw_annotations:
                if not isinstance(item, dict):
                    continue
                base = str(item.get("base", "") or "")
                reading = str(item.get("reading", "") or "")
                if base and reading:
                    annotations.append(dict(item))
        if not annotations:
            annotations = [
                {"base": base, "reading": reading}
                for base, reading in _metadata_ruby_pairs(metadata)
            ]
        if not annotations:
            continue

        current = str(block.text or "")
        marked, matched = _render_locked_annotations(current, annotations)
        unmatched = max(0, len(annotations) - matched)
        metadata = dict(metadata)
        metadata["ruby_structure_locked"] = True
        metadata["ruby_annotations"] = annotations
        metadata.setdefault("ruby_annotation_schema", "legacy-pairs")
        metadata["ruby_overlay_only"] = True
        if matched:
            metadata["ruby_preserved"] = True
            metadata["ruby_aozora"] = marked
            metadata["ruby_pair_count"] = matched
            matched_pairs += matched
            updated_blocks += 1
        else:
            metadata["ruby_preserved"] = False
            metadata.pop("ruby_aozora", None)
            metadata["ruby_pair_count"] = 0
        unmatched_pairs += unmatched
        block.metadata = metadata

    return RubyPreservationReport(
        enabled=True,
        ruby_pairs=matched_pairs + unmatched_pairs,
        matched_pairs=matched_pairs,
        updated_blocks=updated_blocks,
        unmatched_pairs=unmatched_pairs,
    )



RUBY_BLOCK_METADATA_KEYS = (
    "ruby_preserved",
    "ruby_source",
    "ruby_aozora",
    "ruby_annotations",
    "ruby_annotation_schema",
    "ruby_pair_count",
    "ruby_structure_locked",
    "ruby_overlay_only",
    "ruby_original_block_type",
    "ruby_alignment_evidence",
)

RUBY_DOCUMENT_METADATA_KEYS = (
    "ruby_preservation_enabled",
    "ruby_overlay_scope",
    "ruby_input_contract",
    "ruby_preservation_engine",
    "ruby_preservation_report",
    "ruby_overlay_transfer_report",
)

_RUBY_LINEAGE_KEYS = (
    "multi_ocr_source_block_ids",
    "source_block_ids",
    "manual_compare_source_block_ids",
    "formatter_source_block_ids",
    "ruby_source_block_ids",
)
_RUBY_LINEAGE_SINGLE_KEYS = (
    "multi_ocr_split_from_block",
    "split_from_block",
)
_RUBY_COLUMN_KEYS = (
    "source_column_ids",
    "multi_ocr_column_ids",
)


def _ruby_overlay_column_ids(block) -> list[str]:
    metadata = getattr(block, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    out: list[str] = []
    for key in _RUBY_COLUMN_KEYS:
        value = metadata.get(key)
        values = [value] if isinstance(value, str) else value if isinstance(value, (list, tuple, set)) else []
        for item in values:
            token = str(item or "")
            if token and token not in out:
                out.append(token)
    value = str(metadata.get("column_id", "") or "")
    if value and value not in out:
        out.append(value)
    return out


def _ruby_overlay_lineage_ids(block) -> list[str]:
    metadata = getattr(block, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    out: list[str] = []
    own = str(getattr(block, "id", "") or "")
    if own:
        out.append(own)
    for key in _RUBY_LINEAGE_SINGLE_KEYS:
        token = str(metadata.get(key, "") or "")
        if token and token not in out:
            out.append(token)
    for key in _RUBY_LINEAGE_KEYS:
        value = metadata.get(key)
        values = [value] if isinstance(value, str) else value if isinstance(value, (list, tuple, set)) else []
        for item in values:
            token = str(item or "")
            if token and token not in out:
                out.append(token)
    return out


def _ruby_overlay_bbox(block) -> list[float]:
    bbox = getattr(block, "bbox", None)
    if bbox is None:
        return []
    try:
        return [float(bbox.x), float(bbox.y), float(bbox.w), float(bbox.h)]
    except (TypeError, ValueError, AttributeError):
        return []


def strip_ruby_overlay(
    document: UnifiedDocument | None,
    *,
    strip_candidate_geometry: bool = False,
    strip_logs: bool = False,
) -> UnifiedDocument | None:
    """Remove every Ruby *result* side-channel without touching authoritative prose.

    This is the hard OFF-state boundary.  It removes locked readings and document-
    level preservation state while leaving OCR text, confidence, provenance and
    ordinary column metadata byte-for-byte unchanged.  Geometry-only candidate
    telemetry can optionally be removed as well so a Ruby-disabled OCR result
    serialises exactly like a normal OCR result and cannot accidentally seed a
    later Ruby pass.
    """
    if document is None:
        return None
    for block in getattr(document, "blocks", []) or []:
        metadata = getattr(block, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        if any(key in metadata for key in RUBY_BLOCK_METADATA_KEYS):
            cleaned = dict(metadata)
            for key in RUBY_BLOCK_METADATA_KEYS:
                cleaned.pop(key, None)
            block.metadata = cleaned

    meta = getattr(document, "metadata", None)
    if meta is not None:
        # Explicit dataclass fields must be reset rather than deleted. Metadata.to_dict()
        # omits false/empty values, preserving the pre-Ruby serialisation shape.
        defaults = {
            "ruby_preservation_enabled": False,
            "ruby_overlay_scope": "",
            "ruby_input_contract": "",
            "ruby_preservation_engine": "",
            "ruby_preservation_report": {},
            "ruby_overlay_transfer_report": {},
        }
        for key, value in defaults.items():
            if hasattr(meta, key):
                setattr(meta, key, copy.deepcopy(value))
            else:
                getattr(meta, "__dict__", {}).pop(key, None)
        if strip_candidate_geometry and hasattr(meta, "ruby_candidate_pages"):
            meta.ruby_candidate_pages = {}

    if strip_logs and hasattr(document, "processing_log"):
        document.processing_log = [
            item for item in (document.processing_log or [])
            if str((item or {}).get("step", "")) != "ruby_preservation"
        ]
    return document


def ruby_overlay_is_enabled(document: UnifiedDocument | None) -> bool:
    """Return True only for an explicitly enabled authoritative Ruby overlay.

    A stray/stale ``ruby_aozora`` key is intentionally insufficient.  Exporters use
    this document-level gate so switching Ruby OFF cannot render residual metadata.
    """
    if document is None:
        return False
    return bool(getattr(getattr(document, "metadata", None), "ruby_preservation_enabled", False))


def _capture_prose_guard(documents: Iterable[UnifiedDocument]) -> list[tuple]:
    guard: list[tuple] = []
    for doc in documents:
        for block in getattr(doc, "blocks", []) or []:
            guard.append((
                block,
                getattr(block, "type", None),
                str(getattr(block, "text", "") or ""),
                str(getattr(block, "ocr_raw", "") or ""),
            ))
    return guard


def _restore_and_check_prose_guard(guard: list[tuple]) -> int:
    """Restore accidental Ruby-side prose mutations and return mutation count."""
    changed = 0
    for block, block_type, text, ocr_raw in guard:
        if (
            getattr(block, "type", None) != block_type
            or str(getattr(block, "text", "") or "") != text
            or str(getattr(block, "ocr_raw", "") or "") != ocr_raw
        ):
            changed += 1
            block.type = block_type
            block.text = text
            block.ocr_raw = ocr_raw
    return changed


def extract_ruby_overlay(document: UnifiedDocument | dict | None) -> dict:
    """Create a compact immutable Ruby side-channel detached from OCR prose.

    Raw OCR model documents intentionally stay Ruby-free.  This object is safe
    to carry through rebuild/round-trip/recovery paths because it contains only
    locked Ruby annotations plus matching provenance; it never becomes an OCR
    voting candidate and never overwrites ``Block.text``.
    """
    if document is None:
        return {"schema": "novel_formatter.ruby_overlay.v1", "blocks": [], "document_metadata": {}}
    if isinstance(document, dict) and str(document.get("schema", "")).startswith("novel_formatter.ruby_overlay."):
        copied = copy.deepcopy(document)
        enabled = bool((copied.get("document_metadata") or {}).get("ruby_preservation_enabled"))
        if not enabled:
            return {"schema": "novel_formatter.ruby_overlay.v1", "blocks": [], "document_metadata": {}}
        return copied
    if not isinstance(document, UnifiedDocument):
        raise TypeError("Ruby overlay source must be UnifiedDocument or ruby overlay dict")
    if not ruby_overlay_is_enabled(document):
        return {"schema": "novel_formatter.ruby_overlay.v1", "blocks": [], "document_metadata": {}}

    blocks: list[dict] = []
    for block in document.blocks:
        metadata = getattr(block, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        if not (metadata.get("ruby_annotations") or metadata.get("ruby_aozora")):
            continue
        ruby_metadata = {
            key: copy.deepcopy(metadata[key])
            for key in RUBY_BLOCK_METADATA_KEYS
            if key in metadata
        }
        blocks.append({
            "block_id": str(getattr(block, "id", "") or ""),
            "page": int(getattr(block, "page", 0) or getattr(block, "page_index", 0) or getattr(block, "page_number", 0) or 0),
            "bbox": _ruby_overlay_bbox(block),
            "column_ids": _ruby_overlay_column_ids(block),
            "lineage_ids": _ruby_overlay_lineage_ids(block),
            # Source text is matching evidence only.  It is immutable and is
            # never written back into the target document.
            "source_text": str(getattr(block, "text", "") or ""),
            "metadata": ruby_metadata,
        })
    source_meta = getattr(getattr(document, "metadata", None), "__dict__", {})
    document_metadata = {
        key: copy.deepcopy(source_meta[key])
        for key in RUBY_DOCUMENT_METADATA_KEYS
        if key in source_meta and key != "ruby_overlay_transfer_report"
    }
    return {
        "schema": "novel_formatter.ruby_overlay.v1",
        "blocks": blocks,
        "document_metadata": document_metadata,
    }


def _ruby_context_quality(text: str, annotation: dict) -> tuple[int, int, int]:
    """Return (best exact context sides, occurrence count, best position)."""
    base = str(annotation.get("base", "") or "")
    if not base:
        return (-1, 0, -1)
    positions = _all_occurrences(text, base)
    if not positions:
        return (-1, 0, -1)
    left = str(annotation.get("left_context", "") or "")
    right = str(annotation.get("right_context", "") or "")
    best_exact = -1
    best_pos = positions[0]
    for pos in positions:
        exact = 0
        if left and text[max(0, pos - len(left)):pos] == left:
            exact += 1
        if right and text[pos + len(base):pos + len(base) + len(right)] == right:
            exact += 1
        if exact > best_exact:
            best_exact = exact
            best_pos = pos
    return (max(0, best_exact), len(positions), best_pos)


def _normalise_overlay_annotations(entry: dict) -> list[dict]:
    metadata = entry.get("metadata") if isinstance(entry, dict) else None
    metadata = metadata if isinstance(metadata, dict) else {}
    raw = metadata.get("ruby_annotations")
    annotations: list[dict] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            base = str(item.get("base", "") or "")
            reading = str(item.get("reading", "") or "")
            if base and reading:
                annotations.append(copy.deepcopy(item))
    if annotations:
        return annotations
    aozora = str(metadata.get("ruby_aozora", "") or "")
    source_text = str(entry.get("source_text", "") or "")
    if aozora:
        generated = _annotations_from_marked(aozora, source_text)
        if generated:
            return generated
    return [
        {"base": match.group(1), "reading": match.group(2)}
        for match in RUBY_RE.finditer(aozora)
    ]


def _ruby_annotation_identity(annotation: dict) -> tuple:
    return (
        str(annotation.get("base", "") or ""),
        str(annotation.get("reading", "") or ""),
        annotation.get("source_offset"),
        str(annotation.get("left_context", "") or ""),
        str(annotation.get("right_context", "") or ""),
    )


def apply_ruby_overlay(
    target: UnifiedDocument,
    overlay_source: UnifiedDocument | dict | None,
) -> dict:
    """Safely carry locked Ruby through document rebuild/split/merge operations.

    Matching priority is stable block identity -> explicit lineage -> physical
    column IDs -> unique page/context evidence.  Ambiguous annotations are not
    guessed.  Only Ruby metadata is copied; target prose/OCR evidence/type are
    never changed.
    """
    overlay = extract_ruby_overlay(overlay_source)
    entries = [item for item in (overlay.get("blocks") or []) if isinstance(item, dict)]
    report = {
        "schema": "novel_formatter.ruby_overlay_transfer.v1",
        "source_blocks": len(entries),
        "source_pairs": 0,
        "matched_pairs": 0,
        "unmatched_pairs": 0,
        "target_blocks": 0,
        "exact_id_pairs": 0,
        "lineage_pairs": 0,
        "column_pairs": 0,
        "context_pairs": 0,
        "base_migrated_pairs": 0,
        "ambiguous_pairs": 0,
    }
    if target is None:
        return report
    overlay_enabled = bool(
        (overlay.get("document_metadata") or {}).get("ruby_preservation_enabled")
    )
    if not overlay_enabled:
        # Explicit OFF overlay is authoritative. Remove any stale target readings
        # rather than letting old block metadata survive a rebuild.
        strip_ruby_overlay(target, strip_candidate_geometry=False, strip_logs=False)
        report["disabled"] = True
        return report

    # When an overlay is present, remove stale copies first.  This is essential
    # after split operations where shallow/deep copies can otherwise duplicate
    # all readings onto every child block.
    if entries:
        for block in target.blocks:
            metadata = getattr(block, "metadata", None)
            if not isinstance(metadata, dict):
                continue
            if any(key in metadata for key in RUBY_BLOCK_METADATA_KEYS):
                metadata = dict(metadata)
                for key in RUBY_BLOCK_METADATA_KEYS:
                    metadata.pop(key, None)
                block.metadata = metadata

    text_targets = [
        block for block in target.blocks
        if getattr(block, "type", None) in _TEXT_TYPES
        and not (getattr(block, "metadata", None) or {}).get("consumed")
        and str(getattr(block, "text", "") or "")
    ]
    target_lineage = [_ruby_overlay_lineage_ids(block) for block in text_targets]
    target_columns = [_ruby_overlay_column_ids(block) for block in text_targets]
    assigned: dict[int, list[dict]] = {}
    # Reserve resolved target spans as soon as an annotation is assigned.  This
    # prevents two source annotations from independently claiming the same
    # surviving homograph before the final renderer gets a chance to de-duplicate.
    assigned_used: dict[int, list[tuple[int, int]]] = {}
    assigned_evidence: dict[int, list] = {}
    assigned_source_ids: dict[int, list[str]] = {}

    for entry in entries:
        annotations = _normalise_overlay_annotations(entry)
        report["source_pairs"] += len(annotations)
        source_id = str(entry.get("block_id", "") or "")
        source_lineage = {str(value) for value in (entry.get("lineage_ids") or []) if str(value)}
        if source_id:
            source_lineage.add(source_id)
        source_columns = {str(value) for value in (entry.get("column_ids") or []) if str(value)}
        try:
            source_page = int(entry.get("page", 0) or 0)
        except (TypeError, ValueError):
            source_page = 0
        source_metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        source_text = str(entry.get("source_text", "") or "")

        for annotation in annotations:
            base = str(annotation.get("base", "") or "")
            reading = str(annotation.get("reading", "") or "")
            if not base or not reading:
                report["unmatched_pairs"] += 1
                continue
            resolved_by_target: dict[int, dict | None] = {}

            def _resolved(index: int) -> dict | None:
                if index not in resolved_by_target:
                    text = str(getattr(text_targets[index], "text", "") or "")
                    resolved_by_target[index] = resolve_annotation_for_text(
                        text, annotation, assigned_used.get(index, ()),
                        source_text=source_text,
                    )
                return resolved_by_target[index]

            def _can_render(index: int) -> bool:
                return _resolved(index) is not None

            # Stable identity is strongest evidence only when the annotation can
            # still be rendered in that exact target.  A common split operation
            # keeps the source id on the first child even when the Ruby-bearing
            # text moved to a later child; blindly accepting the id would strand
            # the annotation on the wrong block and prevent safe context fallback.
            exact = [
                i for i, block in enumerate(text_targets)
                if source_id
                and str(getattr(block, "id", "") or "") == source_id
                and _can_render(i)
            ]
            chosen = None
            method = ""
            if len(exact) == 1:
                chosen, method = exact[0], "exact_id"
            else:
                lineage = [
                    i for i, ids in enumerate(target_lineage)
                    if source_lineage and source_lineage.intersection(ids) and _can_render(i)
                ]
                if len(lineage) == 1:
                    chosen, method = lineage[0], "lineage"
                else:
                    pool = lineage if lineage else [i for i in range(len(text_targets)) if _can_render(i)]
                    # Prefer the source page when it still exists after rebuild.
                    page_pool = [
                        i for i in pool
                        if source_page > 0 and int(getattr(text_targets[i], "page", 0) or 0) == source_page
                    ]
                    if page_pool:
                        pool = page_pool
                    column_pool = [
                        i for i in pool
                        if source_columns and source_columns.intersection(target_columns[i])
                    ]
                    if len(column_pool) == 1:
                        chosen, method = column_pool[0], "column"
                    else:
                        if column_pool:
                            pool = column_pool
                        ranked: list[tuple[int, int, int]] = []
                        for i in pool:
                            text = str(getattr(text_targets[i], "text", "") or "")
                            exact_context, occurrences, _pos = _ruby_context_quality(text, annotation)
                            # More exact context is better; a single occurrence
                            # is safer than multiple unresolved-looking copies.
                            ranked.append((exact_context, -occurrences, i))
                        ranked.sort(reverse=True)
                        if ranked:
                            best = ranked[0]
                            ties = [item for item in ranked if item[:2] == best[:2]]
                            if len(ties) == 1:
                                chosen, method = best[2], "context"
                            else:
                                report["ambiguous_pairs"] += 1
            if chosen is None:
                report["unmatched_pairs"] += 1
                continue
            resolved = _resolved(chosen)
            if not resolved:
                report["unmatched_pairs"] += 1
                continue
            resolved_annotation = copy.deepcopy(resolved.get("annotation") or annotation)
            resolved_pos = int(resolved.get("position", -1))
            resolved_base = str(resolved_annotation.get("base", "") or "")
            if resolved_pos < 0 or not resolved_base:
                report["unmatched_pairs"] += 1
                continue
            assigned_used.setdefault(chosen, []).append(
                (resolved_pos, resolved_pos + len(resolved_base))
            )
            assigned.setdefault(chosen, []).append(resolved_annotation)
            if str(resolved.get("mode", "")) == "base_edit_span":
                report["base_migrated_pairs"] += 1
            if source_id:
                assigned_source_ids.setdefault(chosen, []).append(source_id)
            evidence = source_metadata.get("ruby_alignment_evidence")
            if isinstance(evidence, list):
                assigned_evidence.setdefault(chosen, []).extend(copy.deepcopy(evidence))
            report["matched_pairs"] += 1
            report[f"{method}_pairs"] += 1

    for index, annotations in assigned.items():
        block = text_targets[index]
        metadata = dict(getattr(block, "metadata", None) or {})
        deduped: list[dict] = []
        seen: set[tuple] = set()
        for annotation in annotations:
            key = _ruby_annotation_identity(annotation)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(annotation)
        metadata["ruby_annotations"] = deduped
        metadata["ruby_annotation_schema"] = "locked-context-v4-position-map"
        metadata["ruby_structure_locked"] = True
        metadata["ruby_overlay_only"] = True
        metadata["ruby_source"] = "findtextCenterNet"
        metadata["ruby_preserved"] = True
        metadata["ruby_pair_count"] = len(deduped)
        metadata["ruby_source_block_ids"] = list(dict.fromkeys(assigned_source_ids.get(index, [])))
        if assigned_evidence.get(index):
            metadata["ruby_alignment_evidence"] = assigned_evidence[index]
        block.metadata = metadata

    source_doc_meta = overlay.get("document_metadata") if isinstance(overlay.get("document_metadata"), dict) else {}
    target_meta = getattr(getattr(target, "metadata", None), "__dict__", {})
    for key in RUBY_DOCUMENT_METADATA_KEYS:
        if key == "ruby_overlay_transfer_report":
            continue
        if key in source_doc_meta:
            target_meta[key] = copy.deepcopy(source_doc_meta[key])
    refresh = refresh_preserved_ruby(target) if assigned else RubyPreservationReport(enabled=True)
    report["rendered_pairs"] = int(refresh.matched_pairs or 0)
    report["render_unmatched_pairs"] = int(refresh.unmatched_pairs or 0)
    report["target_blocks"] = len(assigned)
    target_meta["ruby_overlay_transfer_report"] = copy.deepcopy(report)
    return report


def carry_ruby_overlay(source: UnifiedDocument, target: UnifiedDocument) -> dict:
    """Convenience wrapper used by formatter/replacement transformations."""
    return apply_ruby_overlay(target, source)


def has_ruby_overlay(
    document: UnifiedDocument | None, *, allow_legacy_unscoped: bool = False
) -> bool:
    """Return whether Ruby is explicitly enabled on this authoritative document.

    Candidate geometry and stray block metadata are never sufficient to re-enable
    Ruby.  This explicit document-level gate is what makes the OFF state durable
    across rebuilds, exports and stale snapshots.  ``allow_legacy_unscoped`` exists
    only for deliberate migration tools that need to inspect pre-gate documents.
    """
    if document is None:
        return False
    if bool(getattr(getattr(document, "metadata", None), "ruby_preservation_enabled", False)):
        return True
    if not allow_legacy_unscoped:
        return False
    return any(
        bool((getattr(block, "metadata", None) or {}).get("ruby_annotations"))
        or bool((getattr(block, "metadata", None) or {}).get("ruby_aozora"))
        for block in getattr(document, "blocks", [])
    )


def _clear_ruby_candidate_geometry(documents: Iterable[UnifiedDocument]) -> None:
    """Drop transient scheduling telemetry without touching prose or Ruby output."""
    for document in documents:
        meta = getattr(document, "metadata", None)
        if meta is not None and hasattr(meta, "ruby_candidate_pages"):
            meta.ruby_candidate_pages = {}


def preserve_ruby_in_documents(
    documents: Iterable[UnifiedDocument],
    *,
    cancel_check=None,
    log_callback=None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    runner: Callable[[int, str], dict] | None = None,
    scan_mode: str = "auto",
    candidate_root: str | Path | None = None,
    roi_neighbor_columns: int = 1,
    roi_max_columns: int = 10,
) -> RubyPreservationReport:
    docs = [doc for doc in documents if doc is not None]
    if not docs:
        return RubyPreservationReport(enabled=True, scan_mode=str(scan_mode or "auto"))
    # Isolation rule: ordinary OCR documents are geometry/image-source carriers
    # and the last document is the sole Ruby-overlay target.  Geometry hints from
    # every ordinary OCR are merged without recognized text, so engine ordering
    # cannot make Ruby ROI discovery disappear. Model documents remain pure
    # evidence and only fused output gains Ruby metadata.
    source_doc = docs[0]
    geometry_docs = docs if len(docs) == 1 else docs[:-1]
    merged_candidate_payloads = _merge_ruby_candidate_payloads(geometry_docs)
    target_docs = [docs[-1]]
    # A retry must never inherit readings from a previous Ruby pass.  Clear only
    # the result side-channel; keep geometry until the new pass finishes.
    for doc in target_docs:
        strip_ruby_overlay(doc, strip_candidate_geometry=False, strip_logs=True)
    # Defence in depth: findtextCenterNet is allowed to add metadata only.  Capture
    # prose/type/raw OCR for every participant so a future upstream or integration
    # regression cannot silently mutate any ordinary OCR result or the fused text.
    prose_guard = _capture_prose_guard(docs)
    diagnostics: dict = {}
    try:
        lines, pages_scanned = detect_ruby_lines(
            source_doc,
            cancel_check=cancel_check,
            log_callback=log_callback,
            progress_callback=progress_callback,
            runner=runner,
            scan_mode=scan_mode,
            candidate_root=candidate_root,
            candidate_payloads=merged_candidate_payloads,
            roi_neighbor_columns=roi_neighbor_columns,
            roi_max_columns=roi_max_columns,
            diagnostics=diagnostics,
        )
        reports = [apply_ruby_lines(doc, lines) for doc in target_docs]
        prose_mutations = _restore_and_check_prose_guard(prose_guard)
        if prose_mutations:
            for doc in target_docs:
                strip_ruby_overlay(doc, strip_candidate_geometry=False, strip_logs=True)
            raise RuntimeError(
                f"Ruby 隔离保护触发：检测到 {prose_mutations} 个正文块被 Ruby 支路意外修改，已恢复正文并丢弃 Ruby 结果"
            )
        report = RubyPreservationReport(
            enabled=True,
            pages_scanned=pages_scanned,
            ruby_lines=len(lines),
            ruby_pairs=sum(len(line.pairs) for line in lines),
            matched_lines=max((item.matched_lines for item in reports), default=0),
            matched_pairs=max((item.matched_pairs for item in reports), default=0),
            updated_blocks=max((item.updated_blocks for item in reports), default=0),
            unmatched_pairs=max((item.unmatched_pairs for item in reports), default=0),
            scan_mode=str(diagnostics.get("scan_mode") or scan_mode or "auto"),
            pages_with_candidates=int(diagnostics.get("pages_with_candidates", 0) or 0),
            candidate_boxes=int(diagnostics.get("candidate_boxes", 0) or 0),
            roi_count=int(diagnostics.get("roi_count", 0) or 0),
            roi_coverage_ratio=float(diagnostics.get("roi_coverage_ratio", 0.0) or 0.0),
            estimated_detector_tiles=int(diagnostics.get("estimated_detector_tiles", 0) or 0),
            full_page_detector_tiles=int(diagnostics.get("full_page_detector_tiles", 0) or 0),
            estimated_tile_ratio=float(diagnostics.get("estimated_tile_ratio", 0.0) or 0.0),
            cache_hits=int(diagnostics.get("cache_hits", 0) or 0),
            cache_misses=int(diagnostics.get("cache_misses", 0) or 0),
            failed_rois=int(diagnostics.get("failed_rois", 0) or 0),
        )
        for doc in target_docs:
            doc.metadata.__dict__["ruby_preservation_enabled"] = True
            doc.metadata.__dict__["ruby_overlay_scope"] = "authoritative_output_only"
            doc.metadata.__dict__["ruby_input_contract"] = (
                "original_page_candidate_rois"
                if report.scan_mode == "smart_roi" else "untouched_original_page"
            )
            doc.metadata.__dict__["ruby_preservation_engine"] = "findtextCenterNet"
            doc.metadata.__dict__["ruby_preservation_report"] = report.to_dict()
            if report.scan_mode == "smart_roi":
                saved_pct = max(0.0, (1.0 - report.estimated_tile_ratio) * 100.0)
                message = (
                    f"findtextCenterNet Ruby ROI：候选 {report.candidate_boxes} 框 → "
                    f"{report.roi_count} 个上下文 ROI / {report.pages_scanned} 页；"
                    f"预计检测窗 {report.estimated_detector_tiles}/{max(1, report.full_page_detector_tiles)} "
                    f"（减少约 {saved_pct:.1f}%）；缓存命中 {report.cache_hits}/{report.cache_hits + report.cache_misses}，"
                    f"失败 ROI {report.failed_rois}；检测 {report.ruby_pairs} 处 Ruby，"
                    f"安全回写 {report.matched_pairs} 处 / {report.updated_blocks} 块"
                )
            else:
                message = (
                    f"findtextCenterNet：全页扫描 {pages_scanned} 页，检测 {report.ruby_pairs} 处 Ruby，"
                    f"安全回写 {report.matched_pairs} 处 / {report.updated_blocks} 块"
                )
            doc.add_log("ruby_preservation", message, report.matched_pairs)
        # Candidate boxes are only a scheduler input.  Once the side pass is
        # complete they must not remain on raw OCR voters or the fused result.
        _clear_ruby_candidate_geometry(docs)
        return report
    except InterruptedError:
        _clear_ruby_candidate_geometry(docs)
        raise
    except Exception as exc:
        prose_mutations = _restore_and_check_prose_guard(prose_guard)
        for doc in geometry_docs:
            # Ordinary OCR evidence is never allowed to retain Ruby result metadata.
            strip_ruby_overlay(doc, strip_candidate_geometry=False, strip_logs=False)
        message = str(exc)
        if prose_mutations and "Ruby 隔离保护触发" not in message:
            message = f"{message}；Ruby 隔离保护另恢复 {prose_mutations} 个正文块"
        _emit(log_callback, f"⚠️ Ruby 保留失败，正文 OCR 保持原样：{message}")
        report = RubyPreservationReport(
            enabled=True,
            scan_mode=str(diagnostics.get("scan_mode") or scan_mode or "auto"),
            pages_with_candidates=int(diagnostics.get("pages_with_candidates", 0) or 0),
            candidate_boxes=int(diagnostics.get("candidate_boxes", 0) or 0),
            roi_count=int(diagnostics.get("roi_count", 0) or 0),
            roi_coverage_ratio=float(diagnostics.get("roi_coverage_ratio", 0.0) or 0.0),
            estimated_detector_tiles=int(diagnostics.get("estimated_detector_tiles", 0) or 0),
            full_page_detector_tiles=int(diagnostics.get("full_page_detector_tiles", 0) or 0),
            estimated_tile_ratio=float(diagnostics.get("estimated_tile_ratio", 0.0) or 0.0),
            cache_hits=int(diagnostics.get("cache_hits", 0) or 0),
            cache_misses=int(diagnostics.get("cache_misses", 0) or 0),
            failed_rois=int(diagnostics.get("failed_rois", 0) or 0),
            error=message,
        )
        for doc in target_docs:
            doc.metadata.__dict__["ruby_preservation_enabled"] = True
            doc.metadata.__dict__["ruby_overlay_scope"] = "authoritative_output_only"
            doc.metadata.__dict__["ruby_input_contract"] = (
                "original_page_candidate_rois"
                if report.scan_mode == "smart_roi" else "untouched_original_page"
            )
            doc.metadata.__dict__["ruby_preservation_engine"] = "findtextCenterNet"
            doc.metadata.__dict__["ruby_preservation_report"] = report.to_dict()
            doc.add_log("ruby_preservation", f"Ruby 保留失败；正文 OCR 未修改：{message}", 0)
        _clear_ruby_candidate_geometry(docs)
        return report
