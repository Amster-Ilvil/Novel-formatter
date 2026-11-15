#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Side-effect-free OCR environment/model detection.

Older releases only trusted ``.ocr-runtime-state/*.json`` markers written after
a successful run.  That made already-installed environments appear missing
when the project directory was copied or upgraded.  This catalog now combines:

* the successful-run marker (strongest signal),
* the actual virtualenv and installed package,
* known local model-cache locations,
* an optional deep import probe used by the GUI's “重新检测” button.

Detection never creates a venv and never downloads a model.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from adapters.runtime_env import persistent_venv_dir

from adapters.paddle_ocr_models import (
    PADDLE_DETECTION_MODEL,
    PADDLE_RECOGNITION_MODEL,
    PADDLE_RUNTIME_SIGNATURE,
)
ROOT = Path(__file__).parent.parent
STATE_DIR = ROOT / ".ocr-runtime-state"
HAYAI_SUPERNOVA_VENV_DIR = persistent_venv_dir("hayai-ocr-v2.7-supernova")


@dataclass(frozen=True)
class RuntimeComponent:
    id: str
    label: str
    kind: str
    note: str
    venv_dir: str = ""


@dataclass(frozen=True)
class RuntimeProbe:
    component_id: str
    ready: bool
    installed: bool
    detail: str
    source: str = ""


COMPONENTS: dict[str, RuntimeComponent] = {
    "paddle_ocr": RuntimeComponent(
        "paddle_ocr", "PaddleOCR / PP-OCR", "本地模型",
        "会创建独立 Python 环境；日文竖排使用 japan，简体中文横排使用 ch，模型缓存彼此按上游语言配置加载。",
        ".venv-paddle",
    ),
    "paddle_structure": RuntimeComponent(
        "paddle_structure", "PP-StructureV3", "本地模型",
        "会安装 paddlex[ocr] 并下载版面分析模型，通常比纯 OCR 更大。",
        ".venv-paddle",
    ),
    "paddle_vl": RuntimeComponent(
        "paddle_vl", "PaddleOCR-VL-1.6", "本地大模型",
        "Apple Silicon 默认使用 PaddleOCR 官方 MLX-VLM 后端（独立 .venv-mlx-vlm）；"
        "其它平台或 MLX 失败自动回退原生 Paddle，PP-OCRv6 / PP-Structure 不受影响。",
        ".venv-paddle",
    ),
    "ndlocr_lite": RuntimeComponent(
        "ndlocr_lite", "NDLOCR-Lite", "本地模型",
        "会下载官方源码、ONNX 模型和独立运行环境。",
        ".venv-ndlocr-lite",
    ),
    "findtext_centernet_ruby": RuntimeComponent(
        "findtext_centernet_ruby", "findtextCenterNet Ruby 专家", "本地模型",
        "与其它本地 OCR 使用同一套运行环境检测/安装确认；固定 commit 上游源码保持原样，"
        "由长期存活的 adapter worker 直接 import 原项目 run_ocr.py，Detector/Transformer 仅加载一次。"
        "Novel Formatter 只提供 Smart ROI 与解析上游 Ruby JSON，不改主 OCR、OCR 对比或 Fusion。",
        ".venv-findtext-centernet",
    ),
    "hayai_ocr_supernova": RuntimeComponent(
        "hayai_ocr_supernova", "Hayai OCR v2.7 Supernova · PyTorch", "本地模型",
        "官方 v2.7 纯 Transformers 5 推理；独立虚拟环境和独立模型缓存；"
        "首次确认后下载 Hugging Face 官方权重及执行其定制模型代码。",
        str(HAYAI_SUPERNOVA_VENV_DIR),
    ),
    "manga_48px": RuntimeComponent(
        "manga_48px", "Manga 48px AR OCR", "本地模型",
        "首次使用下载 Manga Image Translator 官方 48px 自回归权重（约 195 MB）、字符表和独立 PyTorch 环境。",
        ".venv-manga-48px",
    ),
}

_PACKAGE_MARKERS = {
    "paddle_ocr": ("paddle", "paddleocr"),
    "paddle_structure": ("paddle", "paddleocr", "paddlex"),
    "paddle_vl": ("paddle", "paddleocr", "paddlex"),
    "ndlocr_lite": ("onnxruntime", "cv2", "yaml"),
    "findtext_centernet_ruby": ("torch", "torchvision", "PIL"),
    "hayai_ocr_supernova": ("torch", "transformers", "PIL", "huggingface_hub"),
    "manga_48px": ("torch", "einops", "PIL"),
}
_DEEP_IMPORTS = {
    "paddle_ocr": "import paddle, paddleocr",
    "paddle_structure": "import paddle, paddleocr, paddlex",
    "paddle_vl": "import paddle, paddleocr, paddlex",
    "ndlocr_lite": "import onnxruntime, cv2, yaml",
    "findtext_centernet_ruby": "import torch, torchvision; from PIL import Image",
    "hayai_ocr_supernova": "import torch; from transformers import AutoModel, AutoProcessor; from PIL import Image",
    "manga_48px": "import torch, einops; from PIL import Image",
}
_PROBE_CACHE: dict[tuple[str, bool], RuntimeProbe] = {}


def _venv_root(component_id: str) -> Path:
    component = COMPONENTS[component_id]
    if not component.venv_dir:
        return ROOT
    path = Path(component.venv_dir).expanduser()
    return path if path.is_absolute() else ROOT / path


def _venv_python_path(component_id: str) -> Path | None:
    root = _venv_root(component_id)
    for path in (root / "bin" / "python", root / "Scripts" / "python.exe"):
        if path.exists():
            return path
    return None


def _venv_python_exists(venv_dir: str) -> bool:
    if not venv_dir:
        return True
    value = Path(venv_dir).expanduser()
    root = value if value.is_absolute() else ROOT / value
    return any(path.exists() for path in (root / "bin" / "python", root / "Scripts" / "python.exe"))


def _site_package_roots(component_id: str) -> list[Path]:
    root = _venv_root(component_id)
    candidates = [root / "Lib" / "site-packages"]
    candidates.extend((root / "lib").glob("python*/site-packages") if (root / "lib").exists() else [])
    return [path for path in candidates if path.is_dir()]


def _module_marker_exists(component_id: str, module: str) -> bool:
    normalized = module.replace(".", "/")
    package_leaf = normalized.split("/")[0]
    for site in _site_package_roots(component_id):
        if (site / normalized).exists() or (site / f"{normalized}.py").exists():
            return True
        prefix = package_leaf.replace("_", "-").lower()
        for item in site.iterdir():
            name = item.name.lower().replace("_", "-")
            if name.startswith(prefix + "-") and (name.endswith(".dist-info") or name.endswith(".egg-info")):
                return True
    return False


def _distribution_version_from_site(component_id: str, distribution: str) -> str:
    """Read an installed distribution version without importing its runtime."""
    prefix = str(distribution or "").strip().lower().replace("-", "_") + "_"
    for site in _site_package_roots(component_id):
        try:
            for item in site.iterdir():
                normalized = item.name.lower().replace("-", "_")
                if not (normalized.startswith(prefix) and normalized.endswith(".dist_info")):
                    continue
                metadata = item / "METADATA"
                if metadata.is_file():
                    for line in metadata.read_text(encoding="utf-8", errors="ignore").splitlines():
                        if line.lower().startswith("version:"):
                            return line.split(":", 1)[1].strip()
                # Fall back to the canonical dist-info directory name.
                raw = item.name[:-len(".dist-info")] if item.name.endswith(".dist-info") else item.name
                if "-" in raw:
                    return raw.rsplit("-", 1)[1].strip()
        except OSError:
            continue
    return ""


def _manga_48px_effective_python() -> tuple[Path | None, str]:
    """Return a verified 48px Python without installing or downloading.

    Runtime readiness must match the adapter's actual launch policy. A stale
    project-local venv marker used to report READY even when that venv had no
    torch, while the adapter later fell back to another interpreter or failed.
    Probe explicit override, private venv, then the current interpreter and
    accept only a process that can import every required package (and PyTorch
    2.14+ on M6).
    """
    candidates: list[tuple[Path, str]] = []
    explicit = os.environ.get("NOVEL_FORMATTER_MANGA_48PX_PYTHON", "").strip()
    if explicit:
        candidates.append((Path(explicit).expanduser(), "explicit"))
    private = _venv_python_path("manga_48px")
    if private is not None:
        candidates.append((private, "private"))
    candidates.append((Path(sys.executable).resolve(), "current"))
    marker = "import torch, einops, numpy; from PIL import Image; assert torch.__version__"
    try:
        from utils.apple_silicon_runtime import is_m6
        m6 = bool(is_m6())
    except Exception:
        m6 = False
    if m6:
        marker += (
            "; v=torch.__version__.split('+',1)[0].split('.'); "
            "n=tuple(int(''.join(c for c in p if c.isdigit()) or 0) for p in v[:2]); "
            "assert n >= (2,14)"
        )
    seen: set[str] = set()
    for python, source in candidates:
        key = str(python)
        if key in seen or not python.is_file():
            continue
        seen.add(key)
        try:
            proc = subprocess.run(
                [str(python), "-c", marker],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=20,
            )
        except Exception:
            continue
        if proc.returncode == 0:
            return python, source
    return None, ""


def _environment_installed(component_id: str, *, deep: bool = False) -> tuple[bool, str]:
    if component_id == "manga_48px":
        python, source = _manga_48px_effective_python()
        if python is None:
            return False, "48px 运行环境存在标记，但 torch/einops/Pillow 无法实际导入"
        if deep:
            try:
                result = subprocess.run(
                    [str(python), "-c", _DEEP_IMPORTS[component_id]],
                    capture_output=True, text=True, timeout=20,
                )
            except Exception as exc:
                return False, f"48px 环境导入检测失败：{exc}"
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "导入失败").strip().splitlines()[-1]
                return False, f"48px 环境存在但无法导入：{detail}"
        return True, f"48px 运行环境已验证（{source}: {python}）"
    python = _venv_python_path(component_id)
    python_source = "private"
    if component_id == "hayai_ocr_supernova":
        from adapters.hayai_ocr_adapter import _supernova_effective_python
        python, python_source = _supernova_effective_python()
        if python is None:
            return False, ("v2.7 指定的运行环境无效" if python_source == "invalid-explicit"
                           else "缺少 v2.7 独立 Transformers 5 环境")
        # The same probe is used by the worker before installing dependencies.
        return True, f"Hayai v2.7 Transformers 5 环境已验证（{python_source}: {python}）"
    if python is None:
        return False, "未找到独立运行环境"
    markers = _PACKAGE_MARKERS.get(component_id, ())
    if markers and not all(_module_marker_exists(component_id, marker) for marker in markers):
        return False, "运行环境存在，但所需 OCR 包不完整"
    if deep:
        try:
            result = subprocess.run(
                [str(python), "-c", _DEEP_IMPORTS[component_id]],
                capture_output=True, text=True, timeout=15,
            )
        except Exception as exc:
            return False, f"环境导入检测失败：{exc}"
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "导入失败").strip().splitlines()[-1]
            return False, f"环境存在但无法导入：{detail}"
    return True, "OCR 运行环境已安装"


def _has_large_file(root: Path, minimum: int = 500_000, name_tokens: tuple[str, ...] = ()) -> bool:
    if not root.exists():
        return False
    try:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.stat().st_size < minimum:
                continue
            lowered = str(path).lower()
            if not name_tokens or any(token.lower() in lowered for token in name_tokens):
                return True
    except OSError:
        return False
    return False


def _hf_repo_dir(repo_id: str) -> str:
    return "models--" + "--".join(part for part in str(repo_id or "").strip().split("/") if part)


def _hf_repo_weight_ready(hf_home: Path, repo_id: str) -> bool:
    """Return true only when one complete snapshot exposes a real weight file."""
    repo_dir = _hf_repo_dir(repo_id)
    roots = (hf_home / "hub" / repo_dir, hf_home / repo_dir)
    for root in roots:
        snapshots = root / "snapshots"
        if not snapshots.is_dir():
            continue
        try:
            for snapshot in snapshots.iterdir():
                if not snapshot.is_dir():
                    continue
                for name in ("model.safetensors", "pytorch_model.bin"):
                    weight = snapshot / name
                    if weight.is_file() and weight.stat().st_size > 1_000_000:
                        return True
        except OSError:
            continue
    return False


def _normalise_hf_home(path: Path) -> Path:
    """Normalize HF cache-related paths to a Hugging Face home directory.

    ``HF_HOME`` points at the home itself while ``HUGGINGFACE_HUB_CACHE`` often
    points at its ``hub`` child.  Older Transformers installations may point
    directly at a cache directory.  Keeping this normalization in the runtime
    catalog makes readiness checks and the worker consume the same cache.
    """
    value = Path(path).expanduser()
    if value.name.lower() in {"hub", "transformers"}:
        return value.parent
    return value


def _hayai_cache_roots() -> list[Path]:
    """Return reusable Hayai/Hugging Face cache roots in priority order.

    The project-local cache remains preferred for isolation, but pre-existing
    standard Hugging Face caches are valid and must not trigger an installation
    prompt on every OCR run.  No directory is created here.
    """
    roots: list[Path] = []

    def add(path: Path | str | None) -> None:
        if not path:
            return
        candidate = _normalise_hf_home(Path(path))
        try:
            key = str(candidate.resolve())
        except OSError:
            key = str(candidate)
        if key not in {str(item) for item in roots}:
            roots.append(Path(key))

    override = os.environ.get("NOVEL_FORMATTER_HAYAI_OCR_CACHE_DIR", "").strip()
    add(override or (ROOT / ".model-cache" / "hayai-supernova"))
    add(os.environ.get("HF_HOME", "").strip())
    add(os.environ.get("HUGGINGFACE_HUB_CACHE", "").strip())
    add(os.environ.get("TRANSFORMERS_CACHE", "").strip())
    add(Path.home() / ".cache" / "huggingface")
    return roots


def _hayai_repo_candidates(root: Path, repo_dir: str) -> tuple[Path, ...]:
    """Return the cache layouts used by Hugging Face Hub and Transformers.

    Both Hugging Face hub and older Transformers cache layouts are accepted
    for a fully verified v2.7 Supernova snapshot.
    """
    return (
        root / "hub" / repo_dir,
        root / "transformers" / repo_dir,
        root / repo_dir,
    )


def _supernova_cache_complete(root: Path, *, revision: str | None = None) -> bool:
    """Require a complete v2.7 inference bundle, not a bare weight file.

    The processor resides in the official v2.7 model repository. Never
    import or execute remote model code during a cache probe.
    """
    if not root.exists():
        return False
    required = (
        "model.safetensors", "config.json", "tokenizer.json",
        "preprocessor_config.json", "processor_config.json",
        "modeling_hayai.py", "configuration_hayai.py", "processing_hayai.py",
    )
    try:
        snapshots = root / "snapshots"
        if revision and root.name.startswith("models--"):
            folders = [snapshots / revision]
        elif snapshots.is_dir():
            folders = [path for path in snapshots.iterdir() if path.is_dir()]
        else:
            folders = [root]
        for folder in folders:
            if all((folder / name).is_file() and (folder / name).stat().st_size > 0 for name in required):
                if (folder / "model.safetensors").stat().st_size >= 1_000_000:
                    return True
    except OSError:
        return False
    return False


def _supernova_siglip_config_complete(root: Path) -> bool:
    """Check the tiny SigLIP2 vision config imported by upstream v2.7 code."""
    repo_name = _hf_repo_dir("google/siglip2-base-patch16-naflex")
    try:
        for candidate in _hayai_repo_candidates(root, repo_name):
            if not candidate.exists():
                continue
            for config in candidate.rglob("config.json"):
                if config.is_file() and config.stat().st_size >= 2:
                    return True
    except OSError:
        return False
    return False


def _model_cache_ready(component_id: str) -> tuple[bool, str]:
    home = Path.home()
    if component_id == "findtext_centernet_ruby":
        override = os.environ.get("NOVEL_FORMATTER_FINDTEXT_CENTERNET_DIR", "").strip()
        source = Path(override).expanduser() if override else ROOT / ".ocr-runtimes" / "findtext-centernet" / "src"
        linedetect = source / "textline_detect" / ("linedetect.exe" if os.name == "nt" else "linedetect")
        source_required = [
            source / "run_ocr.py",
            source / "process_ocr_base.py",
            source / "models" / "detector.py",
            source / "models" / "transformer.py",
        ]
        try:
            source_ready = all(path.is_file() for path in source_required) and linedetect.is_file()
            coreml_ready = all(
                (source / name).is_dir() and (source / name / "Manifest.json").is_file()
                for name in ("TextDetector.mlpackage", "TransformerEncoder.mlpackage", "TransformerDecoder.mlpackage")
            )
            onnx_sizes = {
                "TextDetector.quant.onnx": 246_826_507,
                "TransformerEncoder.onnx": 175_284_069,
                "TransformerDecoder.onnx": 264_681_314,
            }
            onnx_ready = all(
                (source / name).is_file() and (source / name).stat().st_size == size
                for name, size in onnx_sizes.items()
            )
            model = source / "model.pt"
            model3 = source / "model3.pt"
            torch_ready = (
                model.is_file() and model.stat().st_size == 1_053_713_502
                and model3.is_file() and model3.stat().st_size == 437_420_605
            )
        except OSError:
            source_ready = coreml_ready = onnx_ready = torch_ready = False
        backend = "CoreML" if coreml_ready else "ONNX" if onnx_ready else "Torch" if torch_ready else ""
        ready = source_ready and bool(backend)
        return (True, f"检测到完整 findtextCenterNet {backend} 上游运行时") if ready else (False, "环境已安装，但 findtextCenterNet Ruby 源码/后端/linedetect 尚未准备完整")
    if component_id == "manga_48px":
        cache = ROOT / ".model-cache" / "manga-48px-ar"
        model = cache / "ocr_ar_48px.ckpt"
        dictionary = cache / "alphabet-all-v7.txt"
        part = cache / "ocr_ar_48px.ckpt.part"
        try:
            ready = model.stat().st_size == 204_290_192 and dictionary.stat().st_size >= 1_000
        except OSError:
            ready = False
        if ready:
            return True, "检测到完整 48px AR 官方权重与字符表"
        try:
            partial_mb = part.stat().st_size / 1024 / 1024
        except OSError:
            partial_mb = 0.0
        if partial_mb > 0:
            return False, f"48px AR 权重已下载 {partial_mb:.1f}/194.8 MiB，下次会断点续传"
        return False, "环境已安装，但 48px AR 权重尚未下载完整"

    if component_id == "hayai_ocr_supernova":
        from adapters.hayai_ocr_adapter import SUPERNOVA_REVISION, _supernova_model_location
        try:
            model_name = _supernova_model_location()
        except ValueError as exc:
            return False, str(exc)
        local_model = Path(model_name).expanduser()
        if local_model.is_dir():
            ready = (_supernova_cache_complete(local_model)
                     and _supernova_siglip_config_complete(local_model.parent))
            return (True, "完整 v2.7 本地模型、处理器和 SigLIP2 配置") if ready else (False, "v2.7 本地模型文件或 SigLIP2 配置不完整")
        repo_dir = _hf_repo_dir(model_name)
        for cache_root in _hayai_cache_roots():
            model_ready = any(
                _supernova_cache_complete(candidate, revision=SUPERNOVA_REVISION)
                for candidate in _hayai_repo_candidates(cache_root, repo_dir)
            )
            if model_ready and _supernova_siglip_config_complete(cache_root):
                return True, f"完整 v2.7 Supernova 官方模型和 SigLIP2 配置缓存（{cache_root}）"
        return False, "v2.7 Supernova 模型、处理器、模型代码或 SigLIP2 配置尚未完整缓存"

    if component_id == "ndlocr_lite":
        model_dir = ROOT / ".ocr-runtimes" / "ndlocr-lite" / "src" / "model"
        try:
            models = [p for p in model_dir.glob("*.onnx") if p.stat().st_size >= 1_000_000]
        except OSError:
            models = []
        return (True, "检测到 NDLOCR-Lite ONNX 模型") if len(models) >= 4 else (False, "环境已安装，但 NDLOCR-Lite ONNX 模型不完整")

    if component_id.startswith("paddle"):
        roots = [
            home / ".paddlex" / "official_models",
            home / ".paddleocr",
            home / ".cache" / "paddle",
            home / ".cache" / "PaddleX",
        ]
        if component_id == "paddle_vl":
            tokens = ("vl", "vision-language", "paddleocr-vl")
            ready = any(_has_large_file(root, 1_000_000, tokens) for root in roots)
            # Official MLX-VLM downloads the model through Hugging Face rather
            # than PaddleX caches. Recognize that cache as a valid reusable VL
            # model without requiring a project-local state marker.
            mlx_hf = (
                home / ".cache" / "huggingface" / "hub"
                / "models--PaddlePaddle--PaddleOCR-VL-1.6"
            )
            ready = ready or _has_large_file(mlx_hf, 1_000_000)
        else:
            v6_ready = all(
                any(_has_large_file(root, 1_000_000, (model_name,)) for root in roots)
                for model_name in (PADDLE_DETECTION_MODEL, PADDLE_RECOGNITION_MODEL)
            )
            default_ready = any(
                _has_large_file(root, 1_000_000, ("ocr", "det", "rec", "japan", "PP-OCR"))
                for root in roots
            )
            ready = v6_ready or default_ready
            if component_id == "paddle_structure":
                ready = ready and any(
                    _has_large_file(root, 1_000_000, ("layout", "doclayout", "structure"))
                    for root in roots
                )
        label = {"paddle_ocr": "PaddleOCR", "paddle_structure": "PP-StructureV3 + PaddleOCR", "paddle_vl": "PaddleOCR-VL"}[component_id]
        return (True, f"检测到 {label} 本地模型缓存") if ready else (False, f"环境已安装，但未检测到 {label} 模型缓存")


    return True, "本地组件可用"


def _user_state_dir() -> Path:
    """Stable per-user state that survives replacing/upgrading the source tree.

    The project-local marker is kept for backwards compatibility, but GUI
    readiness must not regress merely because the application directory was
    replaced.  This directory contains only tiny readiness metadata, never
    model files or user content.
    """
    override = os.environ.get("NOVEL_FORMATTER_RUNTIME_STATE_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Caches" / "NovelFormatter" / "ocr-runtime-state"
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA", "").strip()
        return (Path(base) if base else home / "AppData" / "Local") / "NovelFormatter" / "ocr-runtime-state"
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    return (Path(xdg).expanduser() if xdg else home / ".cache") / "novel-formatter" / "ocr-runtime-state"


def _state_path(component_id: str) -> Path:
    """Legacy/project-local marker path kept for compatibility/tests."""
    return STATE_DIR / f"{component_id}.json"


def _state_paths(component_id: str) -> list[Path]:
    paths = [_state_path(component_id), _user_state_dir() / f"{component_id}.json"]
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        try:
            key = str(path.expanduser().resolve())
        except OSError:
            key = str(path.expanduser())
        if key not in seen:
            seen.add(key)
            unique.append(Path(key))
    return unique


def _read_ready_payload(component_id: str) -> dict | None:
    candidates: list[tuple[int, dict]] = []
    for path in _state_paths(component_id):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if bool(payload.get("ready")):
                candidates.append((path.stat().st_mtime_ns, payload))
        except Exception:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _runtime_python_exists(component_id: str) -> bool:
    """Check the effective runtime while preserving legacy test/plugin hooks."""
    if component_id == "manga_48px":
        return _manga_48px_effective_python()[0] is not None
    if component_id == "hayai_ocr_supernova":
        from adapters.hayai_ocr_adapter import _supernova_effective_python
        return _supernova_effective_python()[0] is not None
    return _venv_python_exists(COMPONENTS[component_id].venv_dir)


def _state_marker_ready(component_id: str) -> bool:
    try:
        payload = _read_ready_payload(component_id)
        if not payload or not _runtime_python_exists(component_id):
            return False
        if component_id == "manga_48px":
            # A READY marker is only advisory. Verify the effective Python and
            # local weights every time so an empty/stale venv cannot masquerade
            # as a usable 48px runtime after copying/upgrading the app.
            env_ready, _detail = _environment_installed(component_id, deep=False)
            return bool(env_ready and _model_cache_ready(component_id)[0])
        if component_id == "ndlocr_lite":
            source_dir = ROOT / ".ocr-runtimes" / "ndlocr-lite"
            if not (source_dir / "src" / "ocr.py").is_file():
                return False
            # A previous successful run is not enough after the source tree or
            # ONNX files have been removed/replaced. NDLOCR's worker needs both.
            return _model_cache_ready(component_id)[0]
        # PaddleOCR v6 medium 在部分网络环境中不可下载；当前运行时允许
        # worker 自动回退到 PaddleOCR 默认日文模型，因此只核对“新版
        # PaddleOCR 运行时”签名，不再把 v6 medium 权重作为唯一 ready 条件。
        if component_id in {"paddle_ocr", "paddle_structure"}:
            return payload.get("runtime_signature") == PADDLE_RUNTIME_SIGNATURE
        if component_id == "hayai_ocr_supernova":
            from adapters.hayai_ocr_adapter import SUPERNOVA_RUNTIME_VERSION
            return (payload.get("version") == SUPERNOVA_RUNTIME_VERSION and
                    payload.get("backend") == "supernova" and
                    _model_cache_ready(component_id)[0])
        return True
    except Exception:
        return False


def mark_runtime_ready(component_id: str, **details) -> None:
    if component_id not in COMPONENTS:
        return
    component = COMPONENTS[component_id]
    if not _runtime_python_exists(component_id):
        return
    payload = {"component": component_id, "ready": True, **details}
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    wrote = False
    for path in _state_paths(component_id):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(encoded, encoding="utf-8")
            tmp.replace(path)
            wrote = True
        except OSError:
            continue
    if wrote:
        _PROBE_CACHE.clear()


def clear_runtime_ready(component_id: str) -> None:
    """Invalidate successful-run markers after a verified runtime failure."""
    for path in _state_paths(component_id):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    _PROBE_CACHE.clear()


def probe_runtime(component_id: str, *, deep: bool = False, refresh: bool = False) -> RuntimeProbe:
    if component_id not in COMPONENTS:
        return RuntimeProbe(component_id, False, False, "未知或已移除的 OCR 组件", "unknown")
    key = (component_id, deep)
    if not refresh and key in _PROBE_CACHE:
        return _PROBE_CACHE[key]

    override = os.environ.get(f"NOVEL_FORMATTER_{component_id.upper()}_READY", "").strip().lower()
    if override in {"1", "true", "yes", "on"}:
        probe = RuntimeProbe(component_id, True, True, "由环境变量标记为可用", "override")
    elif _state_marker_ready(component_id):
        probe = RuntimeProbe(component_id, True, True, "已通过实际 OCR 推理验证", "state")
    else:
        installed, env_detail = _environment_installed(component_id, deep=deep)
        if not installed:
            probe = RuntimeProbe(component_id, False, False, env_detail, "environment")
        else:
            model_ready, model_detail = _model_cache_ready(component_id)
            probe = RuntimeProbe(component_id, model_ready, True, model_detail if not model_ready else model_detail, "scan")
    _PROBE_CACHE[key] = probe
    return probe


def runtime_ready(component_id: str) -> bool:
    probe = probe_runtime(component_id)
    # Negative probes are intentionally short-lived.  A first OCR attempt may
    # have downloaded the model after the confirmation dialog was shown but
    # before a success marker could be written.  Re-scan real files on the next
    # start instead of serving a stale False forever from _PROBE_CACHE.
    if not probe.ready:
        probe = probe_runtime(component_id, refresh=True)
    return probe.ready


def runtime_status_text(component_id: str, *, deep: bool = False, refresh: bool = False) -> str:
    if component_id == "windows_snipping_ocr":
        try:
            from adapters.windows_snipping_ocr_adapter import runtime_status
            return runtime_status()
        except Exception as exc:
            return f"Windows Snipping OCR 检测失败：{exc}"
    if component_id == "apple_vision":
        try:
            import platform
            from adapters.vision_backends import BackendFactory
            native_ready, _ = BackendFactory.create("native_helper").is_available()
            live_ready, _ = BackendFactory.create("live_text").is_available()
            shortcut_ready, _ = BackendFactory.create("shortcut").is_available()
            arch = platform.machine() or "unknown"
            ready = []
            if native_ready:
                ready.append("原生 Vision")
            if live_ready:
                ready.append("Live Text")
            if shortcut_ready:
                ready.append("快捷指令")
            return (f"{arch} · " + " / ".join(ready) + " 可用") if ready else f"{arch} · Apple OCR 通道均不可用"
        except Exception:
            return "Apple Vision 检测失败"
    probe = probe_runtime(component_id, deep=deep, refresh=refresh)
    if not probe.ready and not refresh:
        # Keep status labels consistent with the install-confirmation path: a
        # model that appeared on disk after the previous probe should become
        # "本地可用" without requiring the user to press a separate refresh button.
        probe = probe_runtime(component_id, deep=deep, refresh=True)
    if probe.ready:
        return "本地可用"
    if probe.installed:
        return "环境已安装·模型待确认"
    return "未安装·运行时先确认"


def clear_probe_cache() -> None:
    _PROBE_CACHE.clear()


def required_components(
    adapter_id: str,
    *,
    paddle_pipeline: str = "ocr",
    layout_engine: str = "",
    recognition_engine: str = "",
    engine_options: dict | None = None,
) -> list[str]:
    del layout_engine
    engine = recognition_engine or adapter_id
    if engine == "paddle_ocr":
        return [{"ocr":"paddle_ocr","structure":"paddle_structure","vl":"paddle_vl"}.get(paddle_pipeline,"paddle_ocr")]
    if engine == "hayai_ocr":
        return ["hayai_ocr_supernova"]
    if engine in {"ndlocr_lite","manga_48px"}:
        return [engine]
    return []


def missing_components(component_ids: list[str]) -> list[RuntimeComponent]:
    missing: list[RuntimeComponent] = []
    for cid in component_ids:
        if cid not in COMPONENTS:
            continue
        if runtime_ready(cid):
            continue
        if cid == "hayai_ocr_supernova":
            # Cached weights are more important than source-local environment
            # location.  A new source ZIP may only need its persistent venv
            # created/repaired; never prompt as if already-local weights must be
            # downloaded again.
            # Hayai weights are much larger/more important than the small Python
            # environment.  Once the required model assets are already cached,
            # do not show a misleading "install model" confirmation just
            # because a newly unpacked Novel Formatter version has not yet
            # created its persistent venv.  ensure_venv() will idempotently
            # prepare/repair the shared per-user venv when OCR actually starts.
            try:
                if _model_cache_ready(cid)[0]:
                    continue
            except Exception:
                pass
        missing.append(COMPONENTS[cid])
    return missing


def confirmation_text(components: list[RuntimeComponent]) -> str:
    lines = ["检测到以下 OCR 运行环境或模型尚未完成首次安装/加载：", ""]
    any_cached_model = False
    for component in components:
        probe = probe_runtime(component.id, refresh=True)
        lines.append(f"• {component.label}（{component.kind}）")
        if probe.installed and not probe.ready:
            lines.append(f"  已检测到运行环境，但模型尚未确认完整：{probe.detail}")
        else:
            # Do not hide the actual environment failure behind a generic model
            # description.  In particular, Hayai weights can already be cached
            # globally while a newly unpacked app folder is only missing its
            # small Python venv.
            lines.append(f"  运行环境状态：{probe.detail}")
            try:
                model_ready, model_detail = _model_cache_ready(component.id)
            except Exception:
                model_ready, model_detail = False, ""
            if model_ready:
                any_cached_model = True
                lines.append(f"  模型已在本地：{model_detail}；不会重复下载模型。")
            else:
                lines.append(f"  {component.note}")
    if any_cached_model:
        lines.extend(["", "本地模型已存在的项目只会修复/创建缺失运行环境，不会重新下载权重。"] )
    else:
        lines.extend(["", "只有点击“安装并继续”后才会创建环境、联网下载并启动识别；点击取消不会下载任何模型。"] )
    return "\n".join(lines)


def installed_local_layout_engines() -> list[str]:
    candidates = ["ndlocr_lite", "paddle_ocr", "paddle_structure"]
    return [item for item in candidates if runtime_ready(item)]


def installed_local_recognition_engines() -> list[str]:
    # The UI engine remains ``hayai_ocr``; the actual runtime is Supernova.
    candidates = ["manga_48px", "ndlocr_lite", "paddle_ocr"]
    installed = [item for item in candidates if runtime_ready(item)]
    if runtime_ready("hayai_ocr_supernova"):
        installed.insert(0, "hayai_ocr")
    try:
        from adapters.windows_snipping_ocr_adapter import availability
        if availability()[0]:
            installed.append("windows_snipping_ocr")
    except Exception:
        pass
    return installed
