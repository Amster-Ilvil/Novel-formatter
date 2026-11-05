#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional independent Japanese handwriting recognizer based on OpenVINO.

The model is Open Model Zoo's handwritten-japanese-recognition-0001. Model files
are downloaded from official locations through utils.model_manager (SHA-256 and
SHA-384 checked). They are not bundled.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

from utils.model_manager import ModelManager, ModelStatus, default_manager, get_spec, user_requested_download

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MODEL_DIR = _PROJECT_ROOT / "models" / "handwriting_openvino"
_MODEL_NAME = "handwritten-japanese-recognition-0001"
_MODEL_XML = _MODEL_DIR / f"{_MODEL_NAME}.xml"
_MODEL_BIN = _MODEL_DIR / f"{_MODEL_NAME}.bin"
_CHARLIST = _MODEL_DIR / "kondate_nakayosi.txt"

_MODEL_IDS = (
    "openvino_handwriting.xml",
    "openvino_handwriting.bin",
    "openvino_handwriting.charlist",
)

def runtime_available() -> tuple[bool, str]:
    try:
        import openvino  # noqa: F401
        import numpy  # noqa: F401
        return True, "OpenVINO Runtime 可用"
    except Exception as exc:
        return False, f"缺少 OpenVINO Runtime：{exc}"


def _model_dir() -> Path:
    """Folder holding a complete model set.

    The legacy ``models/handwriting_openvino`` folder is reused when it already
    contains all files; new installs go to the configured models folder.
    """
    manager = default_manager()
    spec = get_spec("openvino_handwriting.xml")
    target = manager.target_dir(spec)
    for candidate in [target] + [_PROJECT_ROOT / item for item in spec.legacy_dirs]:
        if all((candidate / get_spec(model_id).filename).is_file() for model_id in _MODEL_IDS):
            return candidate
    return target


def _model_paths() -> tuple[Path, Path, Path]:
    directory = _model_dir()
    xml_path, bin_path, charlist_path = (directory / get_spec(model_id).filename for model_id in _MODEL_IDS)
    return xml_path, bin_path, charlist_path


def model_available() -> tuple[bool, str]:
    manager = default_manager()
    directory = _model_dir()
    missing = [
        get_spec(model_id).filename
        for model_id in _MODEL_IDS
        if manager.status(model_id, target_dir=directory, deep=False) is not ModelStatus.READY
    ]
    if missing:
        return False, "缺少模型文件：" + "、".join(missing)
    ok, detail = runtime_available()
    return (ok, "模型与运行环境可用" if ok else detail)


def download_model(progress_callback=None, *, cancel_check=None) -> Path:
    """Download/verify all files after an explicit user action."""
    manager: ModelManager = default_manager()
    directory = _model_dir()
    total = len(_MODEL_IDS)
    with user_requested_download():
        for index, model_id in enumerate(_MODEL_IDS, start=1):
            name = get_spec(model_id).filename
            if manager.status(model_id, target_dir=directory) is ModelStatus.READY:
                if progress_callback:
                    progress_callback(index, total, f"已存在：{name}")
                continue
            if progress_callback:
                progress_callback(index - 1, total, f"下载：{name}")
            manager.ensure(model_id, target_dir=directory, cancel_check=cancel_check)
            if progress_callback:
                progress_callback(index, total, f"完成：{name}")
    return directory


class OpenVINOJapaneseHandwritingRecognizer:
    def __init__(self, device: str = "CPU"):
        ok, detail = model_available()
        if not ok:
            raise RuntimeError(detail)
        import numpy as np
        from openvino import Core

        self._np = np
        core = Core()
        xml_path, _bin_path, charlist_path = _model_paths()
        model = core.read_model(str(xml_path))
        self._compiled = core.compile_model(model, device)
        self._input = model.inputs[0]
        self._output = model.outputs[0]
        self._height = int(self._input.shape[2])
        self._width = int(self._input.shape[3])
        characters = "".join(charlist_path.read_text(encoding="utf-8").splitlines())
        self._characters = ["[blank]", *list(characters)]
        if len(self._characters) != int(self._output.shape[2]):
            raise RuntimeError("OpenVINO 手写模型与字符表不匹配")

    def _preprocess(self, image: Image.Image):
        gray = ImageOps.grayscale(image)
        ratio = gray.width / max(1.0, float(gray.height))
        target_width = max(1, min(self._width, int(round(self._height * ratio))))
        resized = gray.resize((target_width, self._height), Image.Resampling.BOX)
        array = self._np.asarray(resized, dtype=self._np.float32)
        if target_width < self._width:
            edge = array[:, -1:]
            pad = self._np.repeat(edge, self._width - target_width, axis=1)
            array = self._np.concatenate([array, pad], axis=1)
        return array[None, None, :, :]

    def recognize(self, image: Image.Image) -> str:
        tensor = self._preprocess(image)
        result = self._compiled([tensor])[self._output]
        indices = self._np.argmax(result, axis=2).transpose(1, 0).reshape(-1)
        chars: list[str] = []
        previous = None
        for raw in indices:
            index = int(raw)
            if index != 0 and index != previous and index < len(self._characters):
                chars.append(self._characters[index])
            previous = index
        return "".join(chars)
