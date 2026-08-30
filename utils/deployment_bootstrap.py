# -*- coding: utf-8 -*-
"""Privacy-safe application bootstrap support for Novel Formatter.

Only the GUI/application dependencies may be prepared during startup. OCR
dependencies and model resources are deliberately deferred until the user
starts OCR and confirms the installation prompt. No OCR update check or model
replacement is performed here.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT / ".runtime"
STATE_PATH = RUNTIME_DIR / "bootstrap-state.json"
LOG_PATH = RUNTIME_DIR / "logs" / "bootstrap.log"
MODEL_BUNDLE_VERSION = "public-bootstrap-v1"

PIP_INDEXES: tuple[tuple[str, str], ...] = (
    ("清华大学 TUNA", "https://pypi.tuna.tsinghua.edu.cn/simple"),
    ("阿里云", "https://mirrors.aliyun.com/pypi/simple"),
    ("华为云", "https://repo.huaweicloud.com/repository/pypi/simple"),
    ("PyPI 官方", "https://pypi.org/simple"),
)
HF_ENDPOINTS: tuple[tuple[str, str], ...] = (
    ("HF-Mirror", "https://hf-mirror.com"),
    ("Hugging Face 官方", "https://huggingface.co"),
)
PADDLE_SOURCES: tuple[str, ...] = ("modelscope", "bos", "aistudio", "huggingface")

ProgressCallback = Callable[[str, int, int, str], None]


@dataclass(frozen=True, slots=True)
class HardwareSummary:
    system: str
    architecture: str
    memory_gb: float
    gpu_vendor: str
    gpu_name: str
    accelerator: str

    def to_public_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DownloadSources:
    pip_name: str
    pip_index: str
    hf_name: str
    hf_endpoint: str
    paddle_source: str

    def to_public_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ComponentResult:
    component: str
    status: str
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


_HOME_PATTERNS = (
    re.compile(r"(?i)(?:[A-Z]:\\Users\\)[^\\\s]+"),
    re.compile(r"/Users/[^/\s]+"),
    re.compile(r"/home/[^/\s]+"),
)
_SECRET_QUERY = re.compile(r"(?i)([?&](?:token|key|signature|sig|credential)=)[^&\s]+")


def _redact(value: object) -> str:
    text = str(value or "")
    try:
        home = str(Path.home())
    except Exception:
        home = ""
    if home:
        text = text.replace(home, "<HOME>")
    for pattern in _HOME_PATTERNS:
        text = pattern.sub("<HOME>", text)
    text = _SECRET_QUERY.sub(r"\1<REDACTED>", text)
    text = re.sub(r"(?i)\b(?:sk|hf|AIza)[-_A-Za-z0-9]{12,}\b", "<REDACTED>", text)
    return text[-4000:]


def _log(message: object) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {_redact(message)}\n"
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line)


def _emit(callback: ProgressCallback | None, stage: str, current: int, total: int, detail: str) -> None:
    detail = _redact(detail)
    _log(f"{stage}: {detail}")
    if callback is not None:
        try:
            callback(stage, current, total, detail)
        except Exception:
            pass
    print(detail, flush=True)


def _run_text(command: list[str], timeout: float = 12.0) -> str:
    kwargs: dict = {}
    if os.name == "nt":
        kwargs["creationflags"] = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            **kwargs,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return (result.stdout or "").strip()


def _memory_gb() -> float:
    if os.name == "nt":
        # PowerShell is present on supported Windows versions. Do not query the
        # computer name, user profile, serial number or network interfaces.
        ps = shutil.which("powershell.exe") or shutil.which("pwsh.exe")
        if ps:
            raw = _run_text(
                [
                    ps,
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)",
                ]
            )
            try:
                return float(raw.splitlines()[-1].replace(",", "."))
            except (ValueError, IndexError):
                pass
    if sys.platform == "darwin":
        raw = _run_text(["/usr/sbin/sysctl", "-n", "hw.memsize"], timeout=5.0)
        try:
            return round(int(raw) / (1024 ** 3), 1)
        except ValueError:
            pass
    try:
        return round(int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE")) / (1024 ** 3), 1)
    except (AttributeError, OSError, TypeError, ValueError):
        return 0.0


def _gpu_summary() -> tuple[str, str, str]:
    smi = shutil.which("nvidia-smi")
    if smi:
        raw = _run_text([smi, "--query-gpu=name", "--format=csv,noheader"], timeout=12.0)
        name = next((line.strip() for line in raw.splitlines() if line.strip()), "")
        if name:
            return "NVIDIA", name, "CUDA（驱动已检测）"
    if os.name == "nt":
        ps = shutil.which("powershell.exe") or shutil.which("pwsh.exe")
        if ps:
            raw = _run_text(
                [
                    ps,
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
                ],
                timeout=15.0,
            )
            names = [line.strip() for line in raw.splitlines() if line.strip()]
            for name in names:
                lowered = name.lower()
                if "nvidia" in lowered:
                    return "NVIDIA", name, "CUDA（需运行时支持）"
                if "amd" in lowered or "radeon" in lowered:
                    return "AMD", name, "DirectML/CPU"
                if "intel" in lowered:
                    return "Intel", name, "DirectML/CPU"
            if names:
                return "", names[0], "CPU"
    if sys.platform == "darwin":
        machine = platform.machine().lower()
        if machine == "arm64":
            raw = _run_text(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], timeout=5.0)
            return "Apple", raw or "Apple Silicon", "MPS"
        return "Apple/AMD", "macOS 图形设备", "Metal/CPU"
    lspci = shutil.which("lspci")
    if lspci:
        raw = _run_text([lspci], timeout=10.0)
        for line in raw.splitlines():
            lower = line.lower()
            if "vga" not in lower and "3d controller" not in lower:
                continue
            if "nvidia" in lower:
                return "NVIDIA", line.split(": ", 1)[-1], "CUDA（需运行时支持）"
            if "amd" in lower or "radeon" in lower:
                return "AMD", line.split(": ", 1)[-1], "ROCm/CPU"
            if "intel" in lower:
                return "Intel", line.split(": ", 1)[-1], "CPU"
    return "", "未检测到独立 GPU", "CPU"


def detect_hardware() -> HardwareSummary:
    vendor, name, accelerator = _gpu_summary()
    return HardwareSummary(
        system=platform.system() or sys.platform,
        architecture=platform.machine() or "unknown",
        memory_gb=_memory_gb(),
        gpu_vendor=vendor,
        gpu_name=name,
        accelerator=accelerator,
    )


def _reachable(url: str, timeout: float = 4.0) -> bool:
    request = urllib.request.Request(
        url.rstrip("/") + "/",
        headers={"User-Agent": "Novel-Formatter-Deployment/1.0", "Range": "bytes=0-0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(getattr(response, "status", 200) or 200) < 500
    except urllib.error.HTTPError as exc:
        # 401/403/405 proves the endpoint is reachable even if it rejects the
        # lightweight probe method.
        return exc.code in {401, 403, 405, 416}
    except Exception:
        return False


def _first_reachable(items: Iterable[tuple[str, str]], override: str = "") -> tuple[str, str]:
    if override:
        return "用户指定", override.rstrip("/")
    fallback = None
    for name, url in items:
        fallback = fallback or (name, url)
        if _reachable(url):
            return name, url.rstrip("/")
    assert fallback is not None
    return fallback[0], fallback[1].rstrip("/")


def select_download_sources() -> DownloadSources:
    pip_name, pip_index = _first_reachable(
        PIP_INDEXES,
        os.environ.get("NOVEL_FORMATTER_PIP_INDEX", "").strip(),
    )
    hf_name, hf_endpoint = _first_reachable(
        HF_ENDPOINTS,
        os.environ.get("NOVEL_FORMATTER_HF_ENDPOINT", "").strip(),
    )
    paddle = os.environ.get("NOVEL_FORMATTER_PADDLE_SOURCE", "").strip().lower()
    if paddle not in PADDLE_SOURCES:
        paddle = "modelscope"
    return DownloadSources(pip_name, pip_index, hf_name, hf_endpoint, paddle)


def select_application_download_sources() -> DownloadSources:
    """Select only the PyPI source needed by the application bootstrap.

    OCR model endpoints are not probed during application startup. Their source
    selection is deferred until the user confirms an OCR installation.
    """
    pip_name, pip_index = _first_reachable(
        PIP_INDEXES,
        os.environ.get("NOVEL_FORMATTER_PIP_INDEX", "").strip(),
    )
    hf_override = os.environ.get("NOVEL_FORMATTER_HF_ENDPOINT", "").strip()
    paddle = os.environ.get("NOVEL_FORMATTER_PADDLE_SOURCE", "").strip().lower()
    if paddle not in PADDLE_SOURCES:
        paddle = "modelscope"
    return DownloadSources(
        pip_name,
        pip_index,
        "用户指定" if hf_override else "延后选择",
        hf_override.rstrip("/") if hf_override else "https://huggingface.co",
        paddle,
    )


def apply_application_download_environment(sources: DownloadSources) -> dict[str, str]:
    """Apply only pip settings; do not initialize OCR model download settings."""
    env = os.environ
    env["PIP_INDEX_URL"] = sources.pip_index
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    env.setdefault("PIP_NO_INPUT", "1")
    return dict(env)


def apply_download_environment(sources: DownloadSources) -> dict[str, str]:
    env = os.environ
    env["PIP_INDEX_URL"] = sources.pip_index
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    env.setdefault("PIP_NO_INPUT", "1")
    env["HF_ENDPOINT"] = sources.hf_endpoint
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    env["PADDLE_PDX_MODEL_SOURCE"] = sources.paddle_source

    # The 48px runtime verifies the official files by SHA-256, so a mirror can
    # safely be tried first while the official origins remain fallbacks.
    try:
        from adapters.manga_48px_runtime import HF_REVISION
    except Exception:
        HF_REVISION = "3e29cd63a0ce7d1b4013b0a6e56da4cddaf4fe5b"
    if sources.hf_endpoint.rstrip("/") != "https://huggingface.co":
        mirror_root = f"{sources.hf_endpoint.rstrip('/')}/zyddnys/manga-image-translator/resolve/{HF_REVISION}"
        model_urls = [
            f"{mirror_root}/ocr_ar_48px.ckpt?download=true",
            f"https://huggingface.co/zyddnys/manga-image-translator/resolve/{HF_REVISION}/ocr_ar_48px.ckpt?download=true",
        ]
        dict_urls = [
            f"{mirror_root}/alphabet-all-v7.txt?download=true",
            f"https://huggingface.co/zyddnys/manga-image-translator/resolve/{HF_REVISION}/alphabet-all-v7.txt?download=true",
        ]
        env["NOVEL_FORMATTER_MANGA_48PX_MODEL_URLS"] = ",".join(model_urls)
        env["NOVEL_FORMATTER_MANGA_48PX_DICT_URLS"] = ",".join(dict_urls)
        # Existing runtime names are singular but accept comma-separated values.
        env["NOVEL_FORMATTER_MANGA_48PX_MODEL_URL"] = env["NOVEL_FORMATTER_MANGA_48PX_MODEL_URLS"]
        env["NOVEL_FORMATTER_MANGA_48PX_DICT_URL"] = env["NOVEL_FORMATTER_MANGA_48PX_DICT_URLS"]
    return dict(env)


def _requirements_hash(requirements: Path) -> str:
    return hashlib.sha256(requirements.read_bytes()).hexdigest()


def install_main_dependencies(
    *,
    python_executable: Path | None = None,
    requirements: Path | None = None,
    sources: DownloadSources | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ComponentResult:
    python_executable = Path(python_executable or sys.executable)
    requirements = Path(requirements or (ROOT / "requirements.txt"))
    sources = sources or select_download_sources()
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    interpreter_fingerprint = hashlib.sha256(
        (str(python_executable.resolve()) + "|" + platform.python_version()).encode("utf-8", "replace")
    ).hexdigest()[:12]
    stamp = RUNTIME_DIR / f"requirements-{interpreter_fingerprint}-{_requirements_hash(requirements)}.ok"
    probe = subprocess.run(
        [
            str(python_executable),
            "-c",
            "import PySide6, PIL, fitz, docx, httpx; print('ready')",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
    )
    if stamp.is_file() and probe.returncode == 0:
        return ComponentResult("main_dependencies", "ready", "主程序依赖已准备")

    indexes = [(sources.pip_name, sources.pip_index)] + [
        item for item in PIP_INDEXES if item[1].rstrip("/") != sources.pip_index.rstrip("/")
    ]
    errors: list[str] = []
    for index, (name, url) in enumerate(indexes, start=1):
        _emit(progress_callback, "dependencies", index - 1, len(indexes), f"使用{name}安装主程序依赖")
        env = dict(os.environ)
        env["PIP_INDEX_URL"] = url
        # Standalone Python builds normally include pip. ``ensurepip`` is a
        # harmless fallback for minimal installations.
        subprocess.run(
            [str(python_executable), "-m", "ensurepip", "--upgrade"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=300,
            check=False,
        )
        command = [
            str(python_executable),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--prefer-binary",
            "--index-url",
            url,
            "-r",
            str(requirements),
        ]
        try:
            result = subprocess.run(command, env=env, timeout=7200)
        except Exception as exc:
            errors.append(f"{name}: {_redact(exc)}")
            continue
        if result.returncode == 0:
            for old_stamp in RUNTIME_DIR.glob("requirements-*.ok"):
                old_stamp.unlink(missing_ok=True)
            stamp.write_text(MODEL_BUNDLE_VERSION + "\n", encoding="ascii")
            os.environ["PIP_INDEX_URL"] = url
            _emit(progress_callback, "dependencies", index, len(indexes), f"主程序依赖安装完成（{name}）")
            return ComponentResult("main_dependencies", "installed", name)
        errors.append(f"{name}: pip exit {result.returncode}")
    raise RuntimeError("主程序依赖安装失败；已依次尝试国内镜像和官方源。" + "；".join(errors[-4:]))


def _real_onnx(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size < 1_000_000:
            return False
        with path.open("rb") as handle:
            return not handle.read(128).startswith(b"version https://git-lfs")
    except OSError:
        return False


def _ndlocr_ready() -> bool:
    model_dir = ROOT / ".ocr-runtimes" / "ndlocr-lite" / "src" / "model"
    return len([p for p in model_dir.glob("*.onnx") if _real_onnx(p)]) >= 4


def _hayai_ocr_ready() -> bool:
    # Keep the legacy deployment helper aligned with the authoritative runtime
    # catalog.  A model cache alone is not sufficient: the isolated environment
    # must contain the pinned Hayai package and the Torch-specific cache must be
    # complete as well.
    try:
        from adapters.ocr_runtime_catalog import probe_runtime
        return bool(probe_runtime("hayai_ocr", refresh=True).ready)
    except Exception:
        return False


def _manga_ocr_ready() -> bool:
    roots = (
        ROOT / ".model-cache" / "manga-ocr",
        ROOT / ".model-cache" / "manga-ocr" / "hub",
    )
    for root in roots:
        try:
            if any(path.is_file() and path.stat().st_size >= 500_000 for path in root.rglob("*")):
                return True
        except OSError:
            continue
    return False


def _manga_48px_ready() -> bool:
    cache = ROOT / ".model-cache" / "manga-48px-ar"
    try:
        return (cache / "ocr_ar_48px.ckpt").stat().st_size == 204_290_192 and (cache / "alphabet-all-v7.txt").stat().st_size >= 1_000
    except OSError:
        return False


def _install_ndlocr(progress_callback: ProgressCallback | None) -> ComponentResult:
    if _ndlocr_ready():
        return ComponentResult("ndlocr_lite", "ready", "已安装；未检查更新")
    _emit(progress_callback, "model", 0, 1, "准备 NDLOCR-Lite 依赖和模型")
    from adapters.ndlocr_lite_adapter import setup_venv

    setup_venv(verbose=True)
    if not _ndlocr_ready():
        raise RuntimeError("NDLOCR-Lite 安装结束后模型仍不完整")
    return ComponentResult("ndlocr_lite", "installed", "首次部署安装完成")


def _install_hayai_ocr(progress_callback: ProgressCallback | None) -> ComponentResult:
    if _hayai_ocr_ready():
        return ComponentResult("hayai_ocr", "ready", "已安装；未检查更新")
    _emit(progress_callback, "model", 0, 1, "准备 Hayai OCR v2.1 依赖和模型")
    from adapters.hayai_ocr_adapter import WORKER_SCRIPT, _resolved_model_cache, setup_venv

    python = setup_venv(verbose=True, backend="torch")
    cache = _resolved_model_cache()
    cache.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["HF_HOME"] = str(cache)
    env["HUGGINGFACE_HUB_CACHE"] = str(cache / "hub")
    env["TRANSFORMERS_CACHE"] = str(cache / "transformers")
    proc = subprocess.run(
        [str(python), str(WORKER_SCRIPT), "--stream", "--backend", "torch", "--device", "auto"],
        input=json.dumps({"command": "close"}, ensure_ascii=False) + "\n",
        text=True,
        capture_output=True,
        env=env,
        timeout=5400,
    )
    if proc.returncode != 0 or '"ready": true' not in (proc.stdout or "").lower():
        raise RuntimeError("Hayai OCR v2.1 模型初始化失败：" + _redact(proc.stderr or proc.stdout))
    if not _hayai_ocr_ready():
        raise RuntimeError("Hayai OCR v2.1 下载结束后未检测到完整权重")
    return ComponentResult("hayai_ocr", "installed", "首次部署安装完成")


def _install_manga_ocr(progress_callback: ProgressCallback | None) -> ComponentResult:
    if _manga_ocr_ready():
        return ComponentResult("manga_ocr", "ready", "已安装；未检查更新")
    _emit(progress_callback, "model", 0, 1, "准备 Manga OCR 依赖和模型")
    from adapters.manga_ocr_adapter import MODEL_CACHE, WORKER_SCRIPT, setup_venv

    python = setup_venv(verbose=True)
    cache = Path(MODEL_CACHE)
    cache.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["HF_HOME"] = str(cache)
    env["HUGGINGFACE_HUB_CACHE"] = str(cache / "hub")
    env["TRANSFORMERS_CACHE"] = str(cache / "transformers")
    proc = subprocess.run(
        [str(python), str(WORKER_SCRIPT), "--stream"],
        input=json.dumps({"command": "close"}, ensure_ascii=False) + "\n",
        text=True,
        capture_output=True,
        env=env,
        timeout=5400,
    )
    if proc.returncode != 0 or '"ready": true' not in (proc.stdout or "").lower():
        raise RuntimeError("Manga OCR 模型初始化失败：" + _redact(proc.stderr or proc.stdout))
    if not _manga_ocr_ready():
        raise RuntimeError("Manga OCR 下载结束后未检测到完整权重")
    return ComponentResult("manga_ocr", "installed", "首次部署安装完成")


def _install_manga_48px(progress_callback: ProgressCallback | None) -> ComponentResult:
    if _manga_48px_ready():
        return ComponentResult("manga_48px", "ready", "已安装固定兼容权重；未检查更新")
    _emit(progress_callback, "model", 0, 1, "准备 48px AR OCR 依赖和固定兼容权重")
    from adapters.manga_48px_adapter import MODEL_CACHE, setup_venv
    from adapters.manga_48px_runtime import (
        DEFAULT_DICT_URLS,
        DEFAULT_MODEL_URLS,
        DICT_SHA256,
        MODEL_SHA256,
        MODEL_SIZE,
        ensure_runtime_files,
    )

    setup_venv(verbose=True)
    temporary_environment: dict[str, str] = {}
    seeds: list[Path] = []
    if os.name == "nt":
        # Route the two large verified files through the Windows transport
        # stack (BITS -> curl.exe -> Python HTTPS), then feed them into the
        # unchanged OCR runtime's existing local-import path.
        from utils.platform_download import download_first

        seed_dir = RUNTIME_DIR / "downloads" / "manga-48px-seed"
        seed_dir.mkdir(parents=True, exist_ok=True)
        model_seed = seed_dir / "ocr_ar_48px.ckpt"
        dict_seed = seed_dir / "alphabet-all-v7.txt"
        seeds.extend((model_seed, dict_seed))

        def configured(name: str, defaults: tuple[str, ...]) -> list[str]:
            values = [item.strip() for item in os.environ.get(name, "").split(",") if item.strip()]
            for item in defaults:
                if item not in values:
                    values.append(item)
            return values

        download_first(
            configured("NOVEL_FORMATTER_MANGA_48PX_MODEL_URL", DEFAULT_MODEL_URLS),
            model_seed,
            label="48px AR 兼容权重（Windows）",
            progress_callback=progress_callback,
            timeout=1800.0,
        )
        download_first(
            configured("NOVEL_FORMATTER_MANGA_48PX_DICT_URL", DEFAULT_DICT_URLS),
            dict_seed,
            label="48px AR 字符表（Windows）",
            progress_callback=progress_callback,
            timeout=300.0,
        )
        def file_sha256(path: Path) -> str:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        if model_seed.stat().st_size != MODEL_SIZE or file_sha256(model_seed) != MODEL_SHA256:
            raise RuntimeError("Windows 下载的 48px AR 权重大小或 SHA-256 校验失败")
        if dict_seed.stat().st_size < 1_000 or file_sha256(dict_seed) != DICT_SHA256:
            raise RuntimeError("Windows 下载的 48px AR 字符表 SHA-256 校验失败")
        temporary_environment = {
            "NOVEL_FORMATTER_MANGA_48PX_MODEL_FILE": str(model_seed),
            "NOVEL_FORMATTER_MANGA_48PX_DICT_FILE": str(dict_seed),
        }

    previous = {name: os.environ.get(name) for name in temporary_environment}
    try:
        os.environ.update(temporary_environment)
        ensure_runtime_files(Path(MODEL_CACHE), progress_callback=progress_callback)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        for seed in seeds:
            seed.unlink(missing_ok=True)
    if not _manga_48px_ready():
        raise RuntimeError("48px AR OCR 安装结束后校验未通过")
    return ComponentResult("manga_48px", "installed", "首次部署安装完成并通过 SHA-256 校验")


def _install_paddle(progress_callback: ProgressCallback | None) -> ComponentResult:
    # Full profile only. The default public deployment remains the stable
    # three-model bundle above; Paddle can also install lazily on first use.
    _emit(progress_callback, "model", 0, len(PADDLE_SOURCES), "准备 PaddleOCR 依赖和 PP-OCRv6 模型")
    from adapters.paddle_ocr_adapter import VENV_DIR, WORKER_SCRIPT, setup_venv
    from adapters.runtime_env import venv_python

    setup_venv(verbose=True, pipeline="ocr")
    python = venv_python(Path(VENV_DIR))
    errors: list[str] = []
    for index, source in enumerate(PADDLE_SOURCES, start=1):
        env = dict(os.environ)
        env["PADDLE_PDX_MODEL_SOURCE"] = source
        _emit(progress_callback, "model", index - 1, len(PADDLE_SOURCES), f"PaddleOCR 模型源：{source}")
        proc = subprocess.run(
            [str(python), str(WORKER_SCRIPT), "--probe", "--pipeline", "ocr", "--lang", "japan"],
            text=True,
            capture_output=True,
            env=env,
            timeout=5400,
        )
        if proc.returncode == 0 and '"probe": true' in (proc.stdout or "").lower():
            return ComponentResult("paddle_ocr", "installed", f"首次部署安装完成（{source}）")
        errors.append(f"{source}: {_redact(proc.stderr or proc.stdout)}")
    raise RuntimeError("PaddleOCR 模型准备失败：" + "；".join(errors[-2:]))


def _install_yomitoku(progress_callback: ProgressCallback | None) -> ComponentResult:
    _emit(progress_callback, "model", 0, 1, "准备 YomiToku 依赖和模型")
    from adapters.runtime_env import venv_python
    from adapters.yomitoku_adapter import MODEL_CACHE, VENV_DIR, WORKER_SCRIPT, setup_venv

    setup_venv(verbose=True)
    cache = Path(MODEL_CACHE)
    env = dict(os.environ)
    env["HF_HOME"] = str(cache / "huggingface")
    env["TORCH_HOME"] = str(cache / "torch")
    proc = subprocess.run(
        [str(venv_python(Path(VENV_DIR))), str(WORKER_SCRIPT), "--server", "--mode", "fast", "--device", "auto"],
        input=json.dumps({"command": "close"}, ensure_ascii=False) + "\n",
        text=True,
        capture_output=True,
        env=env,
        timeout=5400,
    )
    if proc.returncode != 0 or '"ready": true' not in (proc.stdout or "").lower():
        raise RuntimeError("YomiToku 模型初始化失败：" + _redact(proc.stderr or proc.stdout))
    return ComponentResult("yomitoku", "installed", "首次部署安装完成")


def resolve_profile(hardware: HardwareSummary, requested: str = "auto") -> tuple[str, tuple[str, ...]]:
    value = (requested or "auto").strip().lower()
    if value not in {"auto", "lite", "standard", "full", "none"}:
        value = "auto"
    if value == "auto":
        # Keep low-memory systems usable. Machines with at least 16 GB and a
        # detected accelerator receive the complete local bundle; other modern
        # systems receive the stable three-model bundle.
        if hardware.memory_gb and hardware.memory_gb < 8:
            value = "lite"
        elif hardware.memory_gb >= 16 and hardware.accelerator not in {"CPU", ""}:
            value = "full"
        else:
            value = "standard"
    plans = {
        "none": (),
        "lite": ("ndlocr_lite",),
        "standard": ("ndlocr_lite", "manga_ocr", "manga_48px"),
        # Keep the historical profile byte-for-byte in meaning. Hayai is installed
        # only after the user explicitly selects it in the GUI; adding a new OCR
        # engine must never silently enlarge an existing deployment profile.
        "full": ("ndlocr_lite", "manga_ocr", "manga_48px", "paddle_ocr", "yomitoku"),
    }
    return value, plans[value]


_INSTALLERS = {
    "ndlocr_lite": _install_ndlocr,
    "hayai_ocr": _install_hayai_ocr,
    "manga_ocr": _install_manga_ocr,
    "manga_48px": _install_manga_48px,
    "paddle_ocr": _install_paddle,
    "yomitoku": _install_yomitoku,
}


def _load_state() -> dict:
    try:
        payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _write_state(payload: dict) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    safe = {
        "schema": "novel_formatter.public_deployment.v1",
        "bundle_version": MODEL_BUNDLE_VERSION,
        "profile": str(payload.get("profile") or ""),
        "sources": dict(payload.get("sources") or {}),
        "hardware": dict(payload.get("hardware") or {}),
        "components": list(payload.get("components") or []),
        "completed": bool(payload.get("completed")),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    temp = STATE_PATH.with_suffix(".json.tmp")
    temp.write_text(json.dumps(safe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(STATE_PATH)


def prepare_first_deployment(
    *,
    profile: str = "none",
    progress_callback: ProgressCallback | None = None,
    continue_on_model_error: bool = True,
) -> dict:
    """Record a deferred OCR state without downloading OCR resources.

    This legacy API remains for compatibility with older launch commands. It
    intentionally does not call any OCR installer, create OCR virtualenvs, or
    download model files. OCR installation is triggered only from the OCR UI
    after explicit user confirmation.
    """
    del profile, continue_on_model_error
    hardware = detect_hardware()
    sources = select_application_download_sources()
    apply_application_download_environment(sources)
    _emit(
        progress_callback,
        "ocr_deferred",
        1,
        1,
        "启动阶段不安装 OCR 模型或专用依赖；首次开始 OCR 时将先请求确认。",
    )
    payload = {
        "profile": "deferred",
        "sources": sources.to_public_dict(),
        "hardware": hardware.to_public_dict(),
        "components": [],
        "completed": True,
        "previous_bundle": _load_state().get("bundle_version", ""),
    }
    _write_state(payload)
    return payload


__all__ = [
    "ComponentResult",
    "DownloadSources",
    "HardwareSummary",
    "apply_application_download_environment",
    "apply_download_environment",
    "detect_hardware",
    "install_main_dependencies",
    "prepare_first_deployment",
    "resolve_profile",
    "select_application_download_sources",
    "select_download_sources",
]
