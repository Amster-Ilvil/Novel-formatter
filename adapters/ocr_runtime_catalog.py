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
from dataclasses import dataclass
from pathlib import Path

from adapters.paddle_ocr_models import (
    PADDLE_DETECTION_MODEL,
    PADDLE_RECOGNITION_MODEL,
    PADDLE_RUNTIME_SIGNATURE,
)

ROOT = Path(__file__).parent.parent
STATE_DIR = ROOT / ".ocr-runtime-state"


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
        "paddle_vl", "PaddleOCR-VL", "本地大模型",
        "实验性视觉语言模型，首次运行可能下载数 GB。",
        ".venv-paddle",
    ),
    "ndlocr_lite": RuntimeComponent(
        "ndlocr_lite", "NDLOCR-Lite", "本地模型",
        "会下载官方源码、ONNX 模型和独立运行环境。",
        ".venv-ndlocr-lite",
    ),
    "manga_ocr": RuntimeComponent(
        "manga_ocr", "Manga OCR", "本地模型",
        "日文漫画与小说印刷体识别；页面输入会先做物理分列。",
        ".venv-manga-ocr",
    ),
    "manga_48px": RuntimeComponent(
        "manga_48px", "Manga 48px AR OCR", "本地模型",
        "首次使用下载 Manga Image Translator 官方 48px 自回归权重（约 195 MB）、字符表和独立 PyTorch 环境。",
        ".venv-manga-48px",
    ),
    "yomitoku": RuntimeComponent(
        "yomitoku", "YomiToku OCR", "本地模型",
        "仅安装 YomiToku OCR 模块；下载 dbnetv2_1 与 PARSeq 日文识别权重。快速模式按需再下载 large 复核模型。非商业用途遵循上游 CC BY-NC-SA 4.0。",
        ".venv-yomitoku",
    ),
    "pdf_craft": RuntimeComponent(
        "pdf_craft", "PDF Craft / DeepSeek-OCR", "本地大模型",
        "会安装 PDF Craft 并下载 DeepSeek-OCR 权重。当前上游推理主要面向 CUDA；Mac/CPU 可能无法运行。",
        ".venv-pdf-craft",
    ),
}

_PACKAGE_MARKERS = {
    "paddle_ocr": ("paddle", "paddleocr"),
    "paddle_structure": ("paddle", "paddleocr", "paddlex"),
    "paddle_vl": ("paddle", "paddleocr", "paddlex"),
    "ndlocr_lite": ("onnxruntime", "cv2", "yaml"),
    "manga_ocr": ("manga_ocr",),
    "manga_48px": ("torch", "einops", "PIL"),
    "yomitoku": ("yomitoku", "torch", "cv2"),
    "pdf_craft": ("pdf_craft", "torch"),
}
_DEEP_IMPORTS = {
    "paddle_ocr": "import paddle, paddleocr",
    "paddle_structure": "import paddle, paddleocr, paddlex",
    "paddle_vl": "import paddle, paddleocr, paddlex",
    "ndlocr_lite": "import onnxruntime, cv2, yaml",
    "manga_ocr": "from manga_ocr import MangaOcr",
    "manga_48px": "import torch, einops; from PIL import Image",
    "yomitoku": "from yomitoku.text_detector import TextDetector; from yomitoku.text_recognizer import TextRecognizer",
    "pdf_craft": "from pdf_craft import transform_markdown; import torch",
}
_PROBE_CACHE: dict[tuple[str, bool], RuntimeProbe] = {}


def _venv_root(component_id: str) -> Path:
    component = COMPONENTS[component_id]
    return ROOT / component.venv_dir if component.venv_dir else ROOT


def _venv_python_path(component_id: str) -> Path | None:
    root = _venv_root(component_id)
    for path in (root / "bin" / "python", root / "Scripts" / "python.exe"):
        if path.exists():
            return path
    return None


def _venv_python_exists(venv_dir: str) -> bool:
    if not venv_dir:
        return True
    root = ROOT / venv_dir
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


def _environment_installed(component_id: str, *, deep: bool = False) -> tuple[bool, str]:
    python = _venv_python_path(component_id)
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


def _model_cache_ready(component_id: str) -> tuple[bool, str]:
    home = Path.home()
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

    if component_id == "yomitoku":
        cache = ROOT / ".model-cache" / "yomitoku" / "huggingface"
        ready = _has_large_file(cache, 1_000_000, ("dbnet", "parseq", "yomitoku"))
        return (True, "检测到 YomiToku Hugging Face 模型缓存") if ready else (False, "环境已安装，但 YomiToku 模型尚未完成首次下载/推理")

    if component_id == "manga_ocr":
        candidates = [
            ROOT / ".model-cache" / "manga-ocr",
            home / ".cache" / "huggingface" / "hub" / "models--kha-white--manga-ocr-base",
        ]
        for env_name in ("HF_HOME", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE"):
            value = os.environ.get(env_name, "").strip()
            if value:
                candidates.append(Path(value).expanduser())
        return (True, "检测到 Manga OCR 本地权重") if any(_has_large_file(p) for p in candidates) else (False, "环境已安装，但未检测到 Manga OCR 权重")

    if component_id == "pdf_craft":
        path = ROOT / ".model-cache" / "pdf-craft"
        return (True, "检测到 PDF Craft 本地权重") if _has_large_file(path, 1_000_000) else (False, "环境已安装，但未检测到 PDF Craft 权重")

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


def _state_path(component_id: str) -> Path:
    return STATE_DIR / f"{component_id}.json"


def _state_marker_ready(component_id: str) -> bool:
    try:
        payload = json.loads(_state_path(component_id).read_text(encoding="utf-8"))
        if not (bool(payload.get("ready")) and _venv_python_exists(COMPONENTS[component_id].venv_dir)):
            return False
        # PaddleOCR v6 medium 在部分网络环境中不可下载；当前运行时允许
        # worker 自动回退到 PaddleOCR 默认日文模型，因此只核对“新版
        # PaddleOCR 运行时”签名，不再把 v6 medium 权重作为唯一 ready 条件。
        if component_id in {"paddle_ocr", "paddle_structure"}:
            return payload.get("runtime_signature") == PADDLE_RUNTIME_SIGNATURE
        return True
    except Exception:
        return False


def mark_runtime_ready(component_id: str, **details) -> None:
    if component_id not in COMPONENTS:
        return
    component = COMPONENTS[component_id]
    if not _venv_python_exists(component.venv_dir):
        return
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"component": component_id, "ready": True, **details}
    _state_path(component_id).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _PROBE_CACHE.clear()


def probe_runtime(component_id: str, *, deep: bool = False, refresh: bool = False) -> RuntimeProbe:
    if component_id not in COMPONENTS:
        return RuntimeProbe(component_id, True, True, "系统组件可用", "system")
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
    return probe_runtime(component_id).ready


def runtime_status_text(component_id: str, *, deep: bool = False, refresh: bool = False) -> str:
    if component_id == "apple_vision":
        try:
            from adapters.vision_backends import BackendFactory
            helper_ready, _ = BackendFactory.create("native_helper").is_available()
            shortcut_ready, _ = BackendFactory.create("shortcut").is_available()
            if helper_ready and shortcut_ready:
                return "Swift Helper 与快捷指令均可用"
            if helper_ready:
                return "Swift Helper 可用"
            if shortcut_ready:
                return "快捷指令可用"
            return "Apple Vision 两种通道均不可用"
        except Exception:
            return "Apple Vision 检测失败"
    probe = probe_runtime(component_id, deep=deep, refresh=refresh)
    if probe.ready:
        return "本地可用"
    if probe.installed:
        return "环境已安装·模型待确认"
    return "未安装·运行时先确认"


def clear_probe_cache() -> None:
    _PROBE_CACHE.clear()


def required_components(adapter_id: str, *, paddle_pipeline: str = "ocr", layout_engine: str = "", recognition_engine: str = "") -> list[str]:
    del layout_engine
    engine = recognition_engine or adapter_id
    if engine == "paddle_ocr":
        return [{"ocr":"paddle_ocr","structure":"paddle_structure","vl":"paddle_vl"}.get(paddle_pipeline,"paddle_ocr")]
    if engine in {"ndlocr_lite","pdf_craft","manga_ocr","manga_48px","yomitoku"}:
        return [engine]
    return []


def missing_components(component_ids: list[str]) -> list[RuntimeComponent]:
    return [COMPONENTS[cid] for cid in component_ids if cid in COMPONENTS and not runtime_ready(cid)]


def confirmation_text(components: list[RuntimeComponent]) -> str:
    lines = ["检测到以下 OCR 运行环境或模型尚未完成首次安装/加载：", ""]
    for component in components:
        probe = probe_runtime(component.id)
        lines.append(f"• {component.label}（{component.kind}）")
        if probe.installed and not probe.ready:
            lines.append(f"  已检测到运行环境，但模型尚未确认完整：{probe.detail}")
        else:
            lines.append(f"  {component.note}")
    lines.extend(["", "只有点击“安装并继续”后才会创建环境、联网下载并启动识别；点击取消不会下载任何模型。"])
    return "\n".join(lines)


def installed_local_layout_engines() -> list[str]:
    candidates = ["ndlocr_lite", "paddle_ocr", "paddle_structure"]
    return [item for item in candidates if runtime_ready(item)]


def installed_local_recognition_engines() -> list[str]:
    candidates = ["yomitoku", "manga_48px", "manga_ocr", "ndlocr_lite", "paddle_ocr", "pdf_craft"]
    return [item for item in candidates if runtime_ready(item)]
