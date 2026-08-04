#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional independent Japanese handwriting recognizer based on OpenVINO.

The model is Open Model Zoo's handwritten-japanese-recognition-0001. Model files
are downloaded from official locations and SHA-384 checked. They are not bundled.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

from PIL import Image, ImageOps

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MODEL_DIR = _PROJECT_ROOT / "models" / "handwriting_openvino"
_MODEL_NAME = "handwritten-japanese-recognition-0001"
_MODEL_XML = _MODEL_DIR / f"{_MODEL_NAME}.xml"
_MODEL_BIN = _MODEL_DIR / f"{_MODEL_NAME}.bin"
_CHARLIST = _MODEL_DIR / "kondate_nakayosi.txt"

_FILES = {
    _MODEL_XML: (
        "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/"
        f"{_MODEL_NAME}/FP16-INT8/{_MODEL_NAME}.xml",
        "b8cac8a7a57ec8e741d336517e7406333aef866cf5976e28ef44f8f5afb43b116fc657050592229d28a932c9f1ed77c2",
    ),
    _MODEL_BIN: (
        "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/"
        f"{_MODEL_NAME}/FP16-INT8/{_MODEL_NAME}.bin",
        "2120d799e261f6abbf46215de79592c3c334d955a8220c6e7c74791ce7d6662292080d90df593865c145d40516df6c7b",
    ),
    _CHARLIST: (
        "https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/"
        "7cc29a91472b4cb1289a11e655ba3e188e1d4a31/data/dataset_classes/kondate_nakayosi.txt",
        None,
    ),
}


def runtime_available() -> tuple[bool, str]:
    try:
        import openvino  # noqa: F401
        import numpy  # noqa: F401
        return True, "OpenVINO Runtime 可用"
    except Exception as exc:
        return False, f"缺少 OpenVINO Runtime：{exc}"


def model_available() -> tuple[bool, str]:
    missing = [path.name for path in _FILES if not path.exists()]
    if missing:
        return False, "缺少模型文件：" + "、".join(missing)
    ok, detail = runtime_available()
    return (ok, "模型与运行环境可用" if ok else detail)


def _sha384(path: Path) -> str:
    digest = hashlib.sha384()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(progress_callback=None) -> Path:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    total = len(_FILES)
    for index, (target, (url, checksum)) in enumerate(_FILES.items(), start=1):
        if target.exists() and (not checksum or _sha384(target) == checksum):
            if progress_callback:
                progress_callback(index, total, f"已存在：{target.name}")
            continue
        if progress_callback:
            progress_callback(index - 1, total, f"下载：{target.name}")
        fd, temp_name = tempfile.mkstemp(
            prefix=target.name + ".", suffix=".part", dir=str(_MODEL_DIR),
        )
        os.close(fd)
        temp = Path(temp_name)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "NovelFormatter/2.0"})
            with urllib.request.urlopen(request, timeout=120) as response, temp.open("wb") as out:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    out.write(chunk)
            if checksum and _sha384(temp) != checksum:
                raise RuntimeError(f"{target.name} 校验失败")
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)
        if progress_callback:
            progress_callback(index, total, f"完成：{target.name}")
    return _MODEL_DIR


class OpenVINOJapaneseHandwritingRecognizer:
    def __init__(self, device: str = "CPU"):
        ok, detail = model_available()
        if not ok:
            raise RuntimeError(detail)
        import numpy as np
        from openvino import Core

        self._np = np
        core = Core()
        model = core.read_model(str(_MODEL_XML))
        self._compiled = core.compile_model(model, device)
        self._input = model.inputs[0]
        self._output = model.outputs[0]
        self._height = int(self._input.shape[2])
        self._width = int(self._input.shape[3])
        characters = "".join(_CHARLIST.read_text(encoding="utf-8").splitlines())
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
