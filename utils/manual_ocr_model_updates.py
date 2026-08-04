# -*- coding: utf-8 -*-
"""Explicit, user-initiated OCR model update management.

This module is deliberately isolated from OCR recognition code. Importing it is
side-effect free: it never performs a network request, starts a timer, downloads
files, or changes a model. Network access only occurs through ``check_updates``
and model replacement only occurs through ``update_component`` after the GUI has
received an explicit user action.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from utils.atomic_io import atomic_write_json
from utils.safe_archive import safe_extract_zip

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / ".manual-model-updates"
BACKUP_DIR = STATE_DIR / "backups"
HISTORY_PATH = STATE_DIR / "history.jsonl"
LOCK_PATH = STATE_DIR / "update.lock"
USER_AGENT = "Novel-Formatter-Manual-Model-Updater/1.0"

ProgressCallback = Callable[[str, int, int, str], None]
CancelCheck = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class ModelUpdateStatus:
    component_id: str
    label: str
    management: str
    local_revision: str
    remote_revision: str
    state: str
    detail: str
    can_update: bool
    action_label: str
    source_url: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


_MODEL_ORDER = (
    "apple_vision",
    "ndlocr_lite",
    "manga_48px",
    "manga_ocr",
    "yomitoku",
    "paddle_ocr",
    "pdf_craft",
    "google_vision",
)

_LABELS = {
    "apple_vision": "Apple Vision / Live Text",
    "ndlocr_lite": "NDLOCR-Lite",
    "manga_48px": "48px AR OCR",
    "manga_ocr": "Manga OCR",
    "yomitoku": "YomiToku OCR",
    "paddle_ocr": "PaddleOCR / PP-OCR",
    "pdf_craft": "PDF Craft / DeepSeek-OCR",
    "google_vision": "Google Vision API",
}

_SOURCE_URLS = {
    "apple_vision": "x-apple.systempreferences:com.apple.Software-Update-Settings.extension",
    "ndlocr_lite": "https://github.com/ndl-lab/ndlocr-lite/releases",
    "manga_48px": "https://github.com/zyddnys/manga-image-translator/releases",
    "manga_ocr": "https://huggingface.co/kha-white/manga-ocr-base",
    "yomitoku": "https://pypi.org/project/yomitoku/",
    "paddle_ocr": "https://github.com/PaddlePaddle/PaddleOCR/releases",
    "pdf_craft": "https://pypi.org/project/pdf-craft/",
    "google_vision": "https://cloud.google.com/vision/docs",
}

_GITHUB_LATEST = {
    "ndlocr_lite": "https://api.github.com/repos/ndl-lab/ndlocr-lite/releases/latest",
    "manga_48px": "https://api.github.com/repos/zyddnys/manga-image-translator/releases/latest",
    "paddle_ocr": "https://api.github.com/repos/PaddlePaddle/PaddleOCR/releases/latest",
}

_HF_MODEL_API = {
    "manga_ocr": "https://huggingface.co/api/models/kha-white/manga-ocr-base",
}

_PYPI_API = {
    "yomitoku": "https://pypi.org/pypi/yomitoku/json",
    "pdf_craft": "https://pypi.org/pypi/pdf-craft/json",
}


class ModelUpdateError(RuntimeError):
    pass


class ModelUpdateCancelled(ModelUpdateError):
    pass


class _UpdateLock:
    """Process-level and filesystem marker preventing concurrent replacements."""

    _thread_lock = threading.Lock()

    def __enter__(self):
        if not self._thread_lock.acquire(blocking=False):
            raise ModelUpdateError("已有 OCR 模型更新任务正在运行")
        try:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            if LOCK_PATH.exists():
                try:
                    payload = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
                    pid = int(payload.get("pid") or 0)
                except Exception:
                    pid = 0
                if pid and _pid_is_running(pid):
                    raise ModelUpdateError(f"另一个程序实例正在更新 OCR 模型（PID {pid}）")
                LOCK_PATH.unlink(missing_ok=True)
            atomic_write_json(
                LOCK_PATH,
                {
                    "pid": os.getpid(),
                    "started_at": _utc_now(),
                },
            )
            return self
        except Exception:
            self._thread_lock.release()
            raise

    def __exit__(self, exc_type, exc, tb):
        LOCK_PATH.unlink(missing_ok=True)
        self._thread_lock.release()


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
        callback(str(stage), max(0, int(current)), max(0, int(total)), str(detail))
    except Exception:
        pass


def _cancelled(cancel_check: CancelCheck | None) -> bool:
    try:
        return bool(cancel_check and cancel_check())
    except Exception:
        return False


def _request(url: str, *, accept: str = "application/json") -> urllib.request.Request:
    return urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": accept,
            "Cache-Control": "no-cache",
        },
    )


def _read_json_url(url: str, *, timeout: float = 20.0) -> dict:
    with urllib.request.urlopen(_request(url), timeout=timeout) as response:
        raw = response.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise ModelUpdateError("版本信息响应过大，已拒绝处理")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ModelUpdateError("版本信息格式不是 JSON 对象")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_assignment(path: Path, name: str) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    match = re.search(rf"^{re.escape(name)}\s*=\s*[\"']([^\"']+)[\"']", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _dist_version(venv_dir: Path, distribution: str) -> str:
    normalized = distribution.replace("-", "_").lower()
    roots = [venv_dir / "Lib" / "site-packages"]
    lib = venv_dir / "lib"
    if lib.is_dir():
        roots.extend(lib.glob("python*/site-packages"))
    candidates: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.glob("*.dist-info"):
            stem = path.name[:-10].replace("-", "_").lower()
            if stem == normalized or stem.startswith(normalized + "_"):
                candidates.append(path)
    for path in sorted(candidates, reverse=True):
        metadata = path / "METADATA"
        try:
            for line in metadata.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("Version:"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            continue
    return "未安装"


def _read_json_marker(path: Path, key: str = "revision") -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return str(payload.get(key) or "").strip()
    except Exception:
        return ""


def _manga_ocr_local_revision() -> str:
    marker = STATE_DIR / "manga_ocr.json"
    revision = _read_json_marker(marker)
    if revision:
        return revision
    roots = (
        ROOT / ".model-cache" / "manga-ocr" / "hub" / "models--kha-white--manga-ocr-base" / "refs" / "main",
        ROOT / ".model-cache" / "manga-ocr" / "models--kha-white--manga-ocr-base" / "refs" / "main",
        Path.home() / ".cache" / "huggingface" / "hub" / "models--kha-white--manga-ocr-base" / "refs" / "main",
    )
    for ref in roots:
        try:
            value = ref.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            return value
    return "未安装"


def _ndlocr_local_revision() -> str:
    marker = ROOT / ".ocr-runtimes" / "ndlocr-lite" / ".novel-formatter-ndlocr-ref"
    try:
        value = marker.read_text(encoding="utf-8").strip()
    except OSError:
        value = ""
    return value or "未安装"


def _manga_48px_local_revision() -> tuple[str, bool, str]:
    cache = ROOT / ".model-cache" / "manga-48px-ar"
    model = cache / "ocr_ar_48px.ckpt"
    dictionary = cache / "alphabet-all-v7.txt"
    runtime = ROOT / "adapters" / "manga_48px_runtime.py"
    revision = _read_assignment(runtime, "HF_REVISION") or "兼容锁定版"
    expected_size = 204_290_192
    try:
        ready = model.stat().st_size == expected_size and dictionary.stat().st_size >= 1_000
    except OSError:
        ready = False
    if ready:
        return revision, True, "已安装经 SHA-256 约束的兼容权重"
    part = cache / "ocr_ar_48px.ckpt.part"
    try:
        partial = part.stat().st_size / 1024 / 1024
    except OSError:
        partial = 0.0
    if partial:
        return "未完成", False, f"存在 {partial:.1f} MiB 未完成下载，可手动继续/修复"
    return "未安装", False, "未检测到完整 48px AR 权重与字符表"


def _apple_local_revision() -> str:
    if sys_platform() != "darwin":
        return "仅 macOS 可用"
    return platform.mac_ver()[0] or "macOS 系统组件"


def sys_platform() -> str:
    # Kept as a function to make local-status tests deterministic without
    # modifying global ``sys.platform``.
    import sys

    return sys.platform


def local_statuses() -> list[ModelUpdateStatus]:
    """Return local metadata only. This function never accesses the network."""
    statuses: list[ModelUpdateStatus] = []

    statuses.append(
        ModelUpdateStatus(
            "apple_vision",
            _LABELS["apple_vision"],
            "系统托管",
            _apple_local_revision(),
            "由 macOS 软件更新管理",
            "system_managed",
            "Apple Vision 模型随 macOS 更新，应用内不会下载或替换系统模型。",
            False,
            "打开系统更新",
            _SOURCE_URLS["apple_vision"],
        )
    )

    ndl_local = _ndlocr_local_revision()
    statuses.append(
        ModelUpdateStatus(
            "ndlocr_lite",
            _LABELS["ndlocr_lite"],
            "可手动更新",
            ndl_local,
            "未检查",
            "not_checked" if ndl_local != "未安装" else "not_installed",
            "点击“检查更新”后查询 NDLOCR-Lite 官方最新正式版。",
            ndl_local == "未安装",
            "安装/修复" if ndl_local == "未安装" else "更新所选模型",
            _SOURCE_URLS["ndlocr_lite"],
        )
    )

    px_local, px_ready, px_detail = _manga_48px_local_revision()
    statuses.append(
        ModelUpdateStatus(
            "manga_48px",
            _LABELS["manga_48px"],
            "兼容权重锁定",
            px_local,
            "未检查",
            "current_compatible" if px_ready else "repair_required",
            px_detail + "；只允许手动重装当前 OCR 代码验证过的权重。",
            True,
            "重装兼容模型" if px_ready else "安装/修复",
            _SOURCE_URLS["manga_48px"],
        )
    )

    manga_local = _manga_ocr_local_revision()
    statuses.append(
        ModelUpdateStatus(
            "manga_ocr",
            _LABELS["manga_ocr"],
            "可手动更新",
            manga_local,
            "未检查",
            "not_checked" if manga_local != "未安装" else "not_installed",
            "点击“检查更新”后读取 Hugging Face 官方模型仓库提交版本。",
            manga_local == "未安装",
            "安装/修复" if manga_local == "未安装" else "更新所选模型",
            _SOURCE_URLS["manga_ocr"],
        )
    )

    yomi_local = _dist_version(ROOT / ".venv-yomitoku", "yomitoku")
    yomi_pinned = _read_assignment(ROOT / "adapters" / "yomitoku_adapter.py", "YOMITOKU_VERSION")
    statuses.append(
        ModelUpdateStatus(
            "yomitoku",
            _LABELS["yomitoku"],
            "与 OCR 代码绑定",
            yomi_local,
            "未检查",
            "compatibility_locked",
            f"当前 OCR 代码固定兼容 YomiToku {yomi_pinned or '指定版本'}；只显示上游版本，不盲目升级。",
            False,
            "不可单独更新",
            _SOURCE_URLS["yomitoku"],
        )
    )

    paddle_local = _dist_version(ROOT / ".venv-paddle", "paddleocr")
    statuses.append(
        ModelUpdateStatus(
            "paddle_ocr",
            _LABELS["paddle_ocr"],
            "与运行时绑定",
            paddle_local,
            "未检查",
            "compatibility_locked",
            "PaddleOCR 模型名、Paddle/PaddleX 版本与现有 worker 共同构成运行合同；只显示上游版本。",
            False,
            "不可单独更新",
            _SOURCE_URLS["paddle_ocr"],
        )
    )

    pdf_local = _dist_version(ROOT / ".venv-pdf-craft", "pdf-craft")
    statuses.append(
        ModelUpdateStatus(
            "pdf_craft",
            _LABELS["pdf_craft"],
            "与运行时绑定",
            pdf_local,
            "未检查",
            "compatibility_locked",
            "DeepSeek-OCR 权重与 PDF Craft/CUDA 运行时绑定；只显示上游版本，避免单独替换权重。",
            False,
            "不可单独更新",
            _SOURCE_URLS["pdf_craft"],
        )
    )

    statuses.append(
        ModelUpdateStatus(
            "google_vision",
            _LABELS["google_vision"],
            "云端托管",
            "无本地模型",
            "Google 云端服务",
            "cloud_managed",
            "Google Vision API 模型由云端服务维护，本机没有可更新的权重。",
            False,
            "无需更新",
            _SOURCE_URLS["google_vision"],
        )
    )
    return statuses


def _remote_version(component_id: str, timeout: float) -> str:
    if component_id in _GITHUB_LATEST:
        payload = _read_json_url(_GITHUB_LATEST[component_id], timeout=timeout)
        return str(payload.get("tag_name") or payload.get("name") or "").strip()
    if component_id in _HF_MODEL_API:
        payload = _read_json_url(_HF_MODEL_API[component_id], timeout=timeout)
        return str(payload.get("sha") or payload.get("lastModified") or "").strip()
    if component_id in _PYPI_API:
        payload = _read_json_url(_PYPI_API[component_id], timeout=timeout)
        info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
        return str(info.get("version") or "").strip()
    return ""


def check_updates(
    component_ids: Iterable[str] | None = None,
    *,
    timeout: float = 20.0,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> list[ModelUpdateStatus]:
    """Check official upstream metadata after an explicit user action."""
    selected = set(component_ids or _MODEL_ORDER)
    current = {item.component_id: item for item in local_statuses()}
    result: list[ModelUpdateStatus] = []
    ordered = [cid for cid in _MODEL_ORDER if cid in selected]
    total = len(ordered)
    for index, component_id in enumerate(ordered, start=1):
        if _cancelled(cancel_check):
            raise ModelUpdateCancelled("用户取消检查 OCR 模型更新")
        item = current[component_id]
        _emit(progress_callback, "check", index - 1, total, f"检查 {item.label}")
        if component_id in {"apple_vision", "google_vision"}:
            result.append(item)
            _emit(progress_callback, "check", index, total, item.detail)
            continue
        try:
            remote = _remote_version(component_id, timeout)
            if not remote:
                raise ModelUpdateError("上游未返回版本号")
            local = item.local_revision
            if component_id == "manga_48px":
                detail = (
                    f"上游最新发布为 {remote}。当前识别代码只允许使用已验证的 48px AR 兼容权重；"
                    "不会自动替换为未知检查点。"
                )
                state = "current_compatible" if local not in {"未安装", "未完成"} else "repair_required"
                updated = replace(
                    item,
                    remote_revision=remote,
                    state=state,
                    detail=detail,
                    can_update=True,
                    action_label="重装兼容模型" if state == "current_compatible" else "安装/修复",
                )
            elif component_id in {"yomitoku", "paddle_ocr", "pdf_craft"}:
                pinned_detail = item.detail
                if local != "未安装" and local == remote:
                    state = "current_compatible"
                    detail = f"本地版本与上游一致。{pinned_detail}"
                else:
                    state = "upstream_differs_locked"
                    detail = f"上游版本为 {remote}；{pinned_detail}"
                updated = replace(
                    item,
                    remote_revision=remote,
                    state=state,
                    detail=detail,
                    can_update=False,
                )
            else:
                installed = local not in {"", "未安装", "未完成"}
                same = installed and _normalise_revision(local) == _normalise_revision(remote)
                state = "current" if same else ("not_installed" if not installed else "update_available")
                detail = (
                    "已是官方最新版本。"
                    if same
                    else ("未安装，可由用户手动下载安装。" if not installed else "检测到官方新版本，可由用户手动更新。")
                )
                updated = replace(
                    item,
                    remote_revision=remote,
                    state=state,
                    detail=detail,
                    can_update=not same,
                    action_label="安装/修复" if not installed else "更新所选模型",
                )
            result.append(updated)
        except Exception as exc:
            result.append(
                replace(
                    item,
                    remote_revision="检查失败",
                    state="check_error",
                    detail=f"无法查询官方版本：{exc}",
                    can_update=item.state in {"not_installed", "repair_required"},
                )
            )
        _emit(progress_callback, "check", index, total, result[-1].detail)
    return result


def _normalise_revision(value: str) -> str:
    return str(value or "").strip().lower().removeprefix("v")


def _download_file(
    url: str,
    destination: Path,
    *,
    label: str,
    progress_callback: ProgressCallback | None,
    cancel_check: CancelCheck | None,
    timeout: float = 120.0,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")
    part.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(
            _request(url, accept="application/octet-stream,*/*;q=0.8"),
            timeout=timeout,
        ) as response, part.open("wb") as output:
            try:
                total = int(response.headers.get("Content-Length") or 0)
            except (TypeError, ValueError):
                total = 0
            current = 0
            while True:
                if _cancelled(cancel_check):
                    raise ModelUpdateCancelled(f"用户取消下载 {label}")
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                current += len(chunk)
                _emit(progress_callback, "download", current, total, label)
        os.replace(part, destination)
        return destination
    finally:
        part.unlink(missing_ok=True)


def _record_history(component_id: str, from_revision: str, to_revision: str, result: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "time": _utc_now(),
        "component_id": component_id,
        "from_revision": from_revision,
        "to_revision": to_revision,
        "result": result,
    }
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    with HISTORY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass


def _backup_path(name: str) -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return BACKUP_DIR / f"{name}-{stamp}-{uuid.uuid4().hex[:6]}"


def _prune_backups(prefix: str, keep: int = 1) -> None:
    try:
        paths = sorted(
            (path for path in BACKUP_DIR.glob(prefix + "-*") if path.exists()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return
    for path in paths[max(0, keep):]:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def _invalidate_runtime_state(component_id: str) -> None:
    (ROOT / ".ocr-runtime-state" / f"{component_id}.json").unlink(missing_ok=True)


def _update_manga_48px(
    *,
    progress_callback: ProgressCallback | None,
    cancel_check: CancelCheck | None,
) -> str:
    from adapters.manga_48px_runtime import HF_REVISION, ensure_runtime_files

    cache_parent = ROOT / ".model-cache"
    cache_parent.mkdir(parents=True, exist_ok=True)
    live = cache_parent / "manga-48px-ar"
    stage = cache_parent / f".manga-48px-ar.update-{uuid.uuid4().hex}"
    backup: Path | None = None
    try:
        _emit(progress_callback, "prepare", 0, 1, "下载并校验 48px AR 兼容权重")
        ensure_runtime_files(
            stage,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )
        if _cancelled(cancel_check):
            raise ModelUpdateCancelled("用户取消 48px AR 模型更新")
        _emit(progress_callback, "replace", 0, 1, "原子替换 48px AR 模型目录")
        if live.exists():
            backup = _backup_path("manga-48px-ar")
            os.replace(live, backup)
        try:
            os.replace(stage, live)
        except Exception:
            if backup is not None and backup.exists() and not live.exists():
                os.replace(backup, live)
            raise
        atomic_write_json(
            STATE_DIR / "manga_48px.json",
            {"revision": HF_REVISION, "updated_at": _utc_now(), "mode": "manual"},
        )
        _invalidate_runtime_state("manga_48px")
        _prune_backups("manga-48px-ar", keep=1)
        _emit(progress_callback, "done", 1, 1, "48px AR 兼容模型已手动安装并校验")
        return HF_REVISION
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def _find_hf_snapshot_python() -> Path:
    venv = ROOT / ".venv-manga-ocr"
    for path in (venv / "bin" / "python", venv / "Scripts" / "python.exe"):
        if path.exists():
            return path
    # This call is user-initiated and only prepares the already pinned Manga OCR
    # runtime when it does not yet exist.
    from adapters.manga_ocr_adapter import setup_venv

    return Path(setup_venv(verbose=False))


def _update_manga_ocr(
    target_revision: str | None,
    *,
    progress_callback: ProgressCallback | None,
    cancel_check: CancelCheck | None,
) -> str:
    if _cancelled(cancel_check):
        raise ModelUpdateCancelled("用户取消 Manga OCR 模型更新")
    revision = str(target_revision or "").strip()
    if not revision:
        revision = _remote_version("manga_ocr", 20.0)
    if not revision:
        raise ModelUpdateError("Hugging Face 未返回 Manga OCR 提交版本")
    python = _find_hf_snapshot_python()
    cache = ROOT / ".model-cache" / "manga-ocr"
    cache.mkdir(parents=True, exist_ok=True)
    script = (
        "from huggingface_hub import snapshot_download; "
        "import sys; "
        "path=snapshot_download(repo_id='kha-white/manga-ocr-base', "
        "revision=sys.argv[1], cache_dir=sys.argv[2], resume_download=True); "
        "print(path)"
    )
    _emit(progress_callback, "download", 0, 0, f"下载 Manga OCR {revision[:12]}")
    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    try:
        process = subprocess.Popen(
            [str(python), "-c", script, revision, str(cache)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    except OSError as exc:
        raise ModelUpdateError(f"无法启动 Manga OCR 更新环境：{exc}") from exc
    output: list[str] = []
    assert process.stdout is not None
    while True:
        if _cancelled(cancel_check):
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            raise ModelUpdateCancelled("用户取消 Manga OCR 模型更新")
        line = process.stdout.readline()
        if line:
            output.append(line.rstrip())
            _emit(progress_callback, "download", 0, 0, line.rstrip())
        elif process.poll() is not None:
            break
        else:
            time.sleep(0.1)
    code = process.wait()
    if code != 0:
        raise ModelUpdateError("Manga OCR 下载失败：\n" + "\n".join(output[-20:]))
    atomic_write_json(
        STATE_DIR / "manga_ocr.json",
        {"revision": revision, "updated_at": _utc_now(), "mode": "manual"},
    )
    _invalidate_runtime_state("manga_ocr")
    _emit(progress_callback, "done", 1, 1, f"Manga OCR 已更新到 {revision[:12]}")
    return revision


def _ndlocr_model_names(source_dir: Path) -> list[str]:
    names: list[str] = []
    model_dir = source_dir / "src" / "model"
    for path in model_dir.glob("*.onnx"):
        if path.name not in names:
            names.append(path.name)
    ocr_py = source_dir / "src" / "ocr.py"
    try:
        text = ocr_py.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    for name in re.findall(r"[\"']([^\"']+\.onnx)[\"']", text):
        base = Path(name).name
        if base not in names:
            names.append(base)
    return names


def _real_onnx(path: Path) -> bool:
    try:
        if path.stat().st_size < 1_000_000:
            return False
        with path.open("rb") as handle:
            return not handle.read(128).startswith(b"version https://git-lfs")
    except OSError:
        return False


def _prepare_ndlocr_source(
    revision: str,
    *,
    progress_callback: ProgressCallback | None,
    cancel_check: CancelCheck | None,
) -> Path:
    runtime_root = ROOT / ".ocr-runtimes"
    runtime_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="ndlocr-manual-update-", dir=runtime_root))
    archive = work / "source.zip"
    extract = work / "extract"
    extract.mkdir()
    quoted = urllib.parse.quote(revision.removeprefix("refs/tags/"), safe="")
    url = f"https://codeload.github.com/ndl-lab/ndlocr-lite/zip/refs/tags/{quoted}"
    try:
        _download_file(
            url,
            archive,
            label=f"NDLOCR-Lite {revision} 源码",
            progress_callback=progress_callback,
            cancel_check=cancel_check,
            timeout=240.0,
        )
        safe_extract_zip(archive, extract)
        roots = [path for path in extract.iterdir() if path.is_dir()]
        if len(roots) != 1:
            raise ModelUpdateError("NDLOCR-Lite 官方压缩包目录结构异常")
        source = roots[0]
        names = _ndlocr_model_names(source)
        if len(names) < 4:
            raise ModelUpdateError("无法从 NDLOCR-Lite 源码发现完整 ONNX 模型列表")
        model_dir = source / "src" / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        for index, name in enumerate(names, start=1):
            target = model_dir / name
            if _real_onnx(target):
                continue
            model_url = (
                "https://media.githubusercontent.com/media/ndl-lab/ndlocr-lite/"
                f"{urllib.parse.quote(revision, safe='')}/src/model/{urllib.parse.quote(name, safe='')}"
            )
            _download_file(
                model_url,
                target,
                label=f"NDLOCR-Lite 模型 {index}/{len(names)}：{name}",
                progress_callback=progress_callback,
                cancel_check=cancel_check,
                timeout=600.0,
            )
            if not _real_onnx(target):
                target.unlink(missing_ok=True)
                raise ModelUpdateError(f"NDLOCR-Lite 模型无效或仍是 Git LFS 指针：{name}")
        if len([path for path in model_dir.glob("*.onnx") if _real_onnx(path)]) < 4:
            raise ModelUpdateError("NDLOCR-Lite ONNX 模型数量不足")
        prepared = runtime_root / f".ndlocr-lite.prepared-{uuid.uuid4().hex}"
        os.replace(source, prepared)
        return prepared
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _swap_ndlocr_runtime(
    prepared_source: Path,
    revision: str,
    *,
    progress_callback: ProgressCallback | None,
) -> None:
    from adapters.runtime_env import ensure_venv

    runtime_root = ROOT / ".ocr-runtimes"
    live_source = runtime_root / "ndlocr-lite"
    live_venv = ROOT / ".venv-ndlocr-lite"
    source_backup: Path | None = None
    venv_backup: Path | None = None
    try:
        _emit(progress_callback, "replace", 0, 3, "备份当前 NDLOCR-Lite 运行时")
        if live_source.exists():
            source_backup = _backup_path("ndlocr-lite-source")
            os.replace(live_source, source_backup)
        if live_venv.exists():
            venv_backup = _backup_path("ndlocr-lite-venv")
            os.replace(live_venv, venv_backup)
        os.replace(prepared_source, live_source)
        (live_source / ".novel-formatter-ndlocr-ref").write_text(revision + "\n", encoding="utf-8")
        _emit(progress_callback, "replace", 1, 3, "创建新的独立 NDLOCR-Lite 环境")
        ensure_venv(
            live_venv,
            label="NDLOCR-Lite",
            marker_code="import onnxruntime, cv2, numpy, PIL, yaml",
            requirements=live_source / "requirements.txt",
            verbose=False,
        )
        _emit(progress_callback, "replace", 2, 3, "验证 NDLOCR-Lite 模型文件")
        model_dir = live_source / "src" / "model"
        if len([path for path in model_dir.glob("*.onnx") if _real_onnx(path)]) < 4:
            raise ModelUpdateError("更新后的 NDLOCR-Lite 模型验证失败")
        _emit(progress_callback, "replace", 3, 3, "NDLOCR-Lite 运行时替换完成")
    except Exception:
        shutil.rmtree(live_source, ignore_errors=True)
        shutil.rmtree(live_venv, ignore_errors=True)
        if source_backup is not None and source_backup.exists():
            os.replace(source_backup, live_source)
        if venv_backup is not None and venv_backup.exists():
            os.replace(venv_backup, live_venv)
        raise
    _prune_backups("ndlocr-lite-source", keep=1)
    _prune_backups("ndlocr-lite-venv", keep=1)


def _update_ndlocr(
    target_revision: str | None,
    *,
    progress_callback: ProgressCallback | None,
    cancel_check: CancelCheck | None,
) -> str:
    revision = str(target_revision or "").strip()
    if not revision:
        revision = _remote_version("ndlocr_lite", 20.0)
    if not revision:
        raise ModelUpdateError("GitHub 未返回 NDLOCR-Lite 最新正式版标签")
    prepared = _prepare_ndlocr_source(
        revision,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )
    try:
        if _cancelled(cancel_check):
            raise ModelUpdateCancelled("用户取消 NDLOCR-Lite 模型更新")
        _swap_ndlocr_runtime(prepared, revision, progress_callback=progress_callback)
    finally:
        if prepared.exists():
            shutil.rmtree(prepared, ignore_errors=True)
    atomic_write_json(
        STATE_DIR / "ndlocr_lite.json",
        {"revision": revision, "updated_at": _utc_now(), "mode": "manual"},
    )
    _invalidate_runtime_state("ndlocr_lite")
    _emit(progress_callback, "done", 1, 1, f"NDLOCR-Lite 已更新到 {revision}")
    return revision


def update_component(
    component_id: str,
    *,
    target_revision: str | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> ModelUpdateStatus:
    """Manually install/update one supported model with rollback boundaries."""
    component_id = str(component_id or "").strip()
    if component_id not in {"ndlocr_lite", "manga_48px", "manga_ocr"}:
        raise ModelUpdateError("该模型由系统、云端或当前 OCR 运行合同管理，不能单独替换")
    before_map = {item.component_id: item for item in local_statuses()}
    before = before_map[component_id]
    with _UpdateLock():
        try:
            if component_id == "ndlocr_lite":
                revision = _update_ndlocr(
                    target_revision,
                    progress_callback=progress_callback,
                    cancel_check=cancel_check,
                )
            elif component_id == "manga_48px":
                revision = _update_manga_48px(
                    progress_callback=progress_callback,
                    cancel_check=cancel_check,
                )
            else:
                revision = _update_manga_ocr(
                    target_revision,
                    progress_callback=progress_callback,
                    cancel_check=cancel_check,
                )
            _record_history(component_id, before.local_revision, revision, "success")
        except Exception as exc:
            _record_history(component_id, before.local_revision, target_revision or "", f"failed: {exc}")
            raise
    after = {item.component_id: item for item in local_statuses()}[component_id]
    return replace(
        after,
        remote_revision=revision,
        state="current",
        detail=f"已由用户手动更新/修复到 {revision}；程序未启用自动检查或自动更新。",
        can_update=False if component_id != "manga_48px" else True,
        action_label="重装兼容模型" if component_id == "manga_48px" else "已是当前版本",
    )


def open_source_url(component_id: str) -> str:
    return _SOURCE_URLS.get(str(component_id or ""), "")


__all__ = [
    "ModelUpdateCancelled",
    "ModelUpdateError",
    "ModelUpdateStatus",
    "check_updates",
    "local_statuses",
    "open_source_url",
    "update_component",
]
