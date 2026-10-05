#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Windows Snipping Tool OneOCR worker.

This worker talks to the OneOCR runtime already installed with Microsoft's
Snipping Tool.  It does not download, copy, bundle or modify Microsoft files.

The OneOCR export surface used here is an undocumented compatibility ABI and
may change with a future Snipping Tool update.  All calls therefore fail closed
with explicit diagnostics instead of silently falling back to another OCR.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from ctypes import (
    POINTER, Structure, byref, c_char, c_char_p, c_float, c_int32, c_int64,
    c_ubyte,
)
from pathlib import Path

from PIL import Image

MODEL_KEY = b'kj)TGtrK>f]b[Piow.gU+nC@s""""""4'
MIN_IMAGE_SIDE = 50
MAX_IMAGE_SIDE = 10_000

c_int64_p = POINTER(c_int64)
c_float_p = POINTER(c_float)
c_ubyte_p = POINTER(c_ubyte)


class ImageStructure(Structure):
    _fields_ = [
        ("type", c_int32),
        ("width", c_int32),
        ("height", c_int32),
        ("reserved", c_int32),
        ("step_size", c_int64),
        ("data_ptr", c_ubyte_p),
    ]


class BoundingBox(Structure):
    _fields_ = [
        ("x1", c_float), ("y1", c_float),
        ("x2", c_float), ("y2", c_float),
        ("x3", c_float), ("y3", c_float),
        ("x4", c_float), ("y4", c_float),
    ]


BoundingBox_p = POINTER(BoundingBox)


_REQUIRED_EXPORTS = (
    ("CreateOcrInitOptions", [c_int64_p], c_int64),
    ("OcrInitOptionsSetUseModelDelayLoad", [c_int64, c_char], c_int64),
    ("CreateOcrPipeline", [c_char_p, c_char_p, c_int64, c_int64_p], c_int64),
    ("CreateOcrProcessOptions", [c_int64_p], c_int64),
    ("OcrProcessOptionsSetMaxRecognitionLineCount", [c_int64, c_int64], c_int64),
    ("RunOcrPipeline", [c_int64, POINTER(ImageStructure), c_int64, c_int64_p], c_int64),
    ("GetOcrLineCount", [c_int64, c_int64_p], c_int64),
    ("GetOcrLine", [c_int64, c_int64, c_int64_p], c_int64),
    ("GetOcrLineContent", [c_int64, POINTER(c_char_p)], c_int64),
    ("GetOcrLineBoundingBox", [c_int64, POINTER(BoundingBox_p)], c_int64),
    ("GetOcrLineWordCount", [c_int64, c_int64_p], c_int64),
    ("GetOcrWord", [c_int64, c_int64, c_int64_p], c_int64),
    ("ReleaseOcrResult", [c_int64], None),
)

_OPTIONAL_EXPORTS = (
    ("GetOcrWordConfidence", [c_int64, c_float_p], c_int64),
    ("ReleaseOcrInitOptions", [c_int64], None),
    ("ReleaseOcrPipeline", [c_int64], None),
    ("ReleaseOcrProcessOptions", [c_int64], None),
)


def _check(code: int, stage: str) -> None:
    if int(code) != 0:
        raise RuntimeError(f"{stage}失败（OneOCR code={int(code)}）")


def _decode_utf8(pointer: c_char_p | None) -> str:
    if not pointer:
        return ""
    raw = pointer.value
    if not raw:
        return ""
    return raw.decode("utf-8", errors="replace").strip()


def _box_points(pointer: BoundingBox_p | None):
    if not pointer:
        return None
    box = pointer.contents
    values = (
        [float(box.x1), float(box.y1)],
        [float(box.x2), float(box.y2)],
        [float(box.x3), float(box.y3)],
        [float(box.x4), float(box.y4)],
    )
    if not all(all(abs(v) < 1e9 for v in point) for point in values):
        return None
    return list(values)


class OneOcrEngine:
    def __init__(self, runtime_dir: str | Path):
        if sys.platform != "win32":
            raise RuntimeError("Windows Snipping OCR 仅支持 Windows 10/11")
        self.runtime_dir = Path(runtime_dir).resolve()
        self._dll_dir_handle = None
        self._ort = None
        self._dll = None
        self._init_options = c_int64()
        self._pipeline = c_int64()
        self._process_options = c_int64()
        self._load()

    def _bind(self, name, argtypes, restype, *, required=True):
        try:
            fn = getattr(self._dll, name)
        except AttributeError:
            if required:
                raise RuntimeError(
                    f"当前 Snipping Tool 的 OneOCR 缺少导出 {name}；"
                    "系统组件 ABI 可能已经变化"
                )
            return None
        fn.argtypes = argtypes
        fn.restype = restype
        return fn

    def _load(self) -> None:
        required_files = ("oneocr.dll", "oneocr.onemodel", "onnxruntime.dll")
        missing = [name for name in required_files if not (self.runtime_dir / name).is_file()]
        if missing:
            raise RuntimeError("OneOCR 运行目录不完整，缺少：" + "、".join(missing))

        if hasattr(os, "add_dll_directory"):
            self._dll_dir_handle = os.add_dll_directory(str(self.runtime_dir))

        # Load ORT first so oneocr.dll resolves the sibling runtime deterministically.
        self._ort = ctypes.WinDLL(str(self.runtime_dir / "onnxruntime.dll"))
        self._dll = ctypes.WinDLL(str(self.runtime_dir / "oneocr.dll"))

        for spec in _REQUIRED_EXPORTS:
            self._bind(*spec, required=True)
        for spec in _OPTIONAL_EXPORTS:
            self._bind(*spec, required=False)

        _check(self._dll.CreateOcrInitOptions(byref(self._init_options)), "创建初始化参数")
        _check(
            self._dll.OcrInitOptionsSetUseModelDelayLoad(self._init_options, c_char(b"\0")),
            "配置模型加载",
        )
        model_bytes = os.fsencode(str(self.runtime_dir / "oneocr.onemodel"))
        _check(
            self._dll.CreateOcrPipeline(
                c_char_p(model_bytes), c_char_p(MODEL_KEY),
                self._init_options, byref(self._pipeline),
            ),
            "创建 OneOCR Pipeline",
        )
        _check(self._dll.CreateOcrProcessOptions(byref(self._process_options)), "创建识别参数")
        _check(
            self._dll.OcrProcessOptionsSetMaxRecognitionLineCount(
                self._process_options, 1000
            ),
            "设置最大识别行数",
        )

    def _word_confidence(self, line_handle: int) -> tuple[float, str]:
        confidence_fn = getattr(self._dll, "GetOcrWordConfidence", None)
        if confidence_fn is None:
            return 0.0, "unavailable"

        count = c_int64()
        if self._dll.GetOcrLineWordCount(line_handle, byref(count)) != 0:
            return 0.0, "unavailable"
        values: list[float] = []
        for index in range(max(0, int(count.value))):
            word = c_int64()
            if self._dll.GetOcrWord(line_handle, index, byref(word)) != 0 or not word.value:
                continue
            score = c_float()
            if confidence_fn(word, byref(score)) != 0:
                continue
            value = float(score.value)
            if 0.0 <= value <= 1.0:
                values.append(value)
        if not values:
            return 0.0, "unavailable"
        return sum(values) / len(values), "model"

    def recognize(self, image_path: str | Path) -> list[dict]:
        with Image.open(image_path) as opened:
            rgba = opened.convert("RGBA")
            width, height = rgba.size
            if (
                width < MIN_IMAGE_SIDE or height < MIN_IMAGE_SIDE
                or width > MAX_IMAGE_SIDE or height > MAX_IMAGE_SIDE
            ):
                raise ValueError(
                    f"OneOCR 输入尺寸必须在 {MIN_IMAGE_SIDE}–{MAX_IMAGE_SIDE}pt；"
                    f"当前为 {width}×{height}"
                )
            # OneOCR type=3 expects BGRA byte order (the native Windows image
            # convention used by Snipping Tool), not Pillow's default RGBA order.
            red, green, blue, alpha = rgba.split()
            bgra = Image.merge("RGBA", (blue, green, red, alpha))
            raw = bgra.tobytes()
            stride = width * 4

        buffer = (c_ubyte * len(raw)).from_buffer_copy(raw)
        image = ImageStructure(
            type=3,
            width=width,
            height=height,
            reserved=0,
            step_size=stride,
            data_ptr=ctypes.cast(buffer, c_ubyte_p),
        )
        result_handle = c_int64()
        _check(
            self._dll.RunOcrPipeline(
                self._pipeline, byref(image), self._process_options, byref(result_handle)
            ),
            "执行 OneOCR",
        )
        if not result_handle.value:
            return []

        try:
            line_count = c_int64()
            _check(
                self._dll.GetOcrLineCount(result_handle, byref(line_count)),
                "读取识别行数",
            )
            blocks: list[dict] = []
            for index in range(max(0, int(line_count.value))):
                line_handle = c_int64()
                _check(
                    self._dll.GetOcrLine(result_handle, index, byref(line_handle)),
                    "读取识别行",
                )
                if not line_handle.value:
                    continue

                text_ptr = c_char_p()
                _check(
                    self._dll.GetOcrLineContent(line_handle, byref(text_ptr)),
                    "读取识别文字",
                )
                text = _decode_utf8(text_ptr)
                if not text:
                    continue

                box_ptr = BoundingBox_p()
                box = None
                if self._dll.GetOcrLineBoundingBox(line_handle, byref(box_ptr)) == 0:
                    box = _box_points(box_ptr)

                confidence, confidence_kind = self._word_confidence(line_handle)
                item = {
                    "text": text,
                    "confidence": confidence,
                    "box": box,
                    "confidence_kind": confidence_kind,
                }
                blocks.append(item)
            return blocks
        finally:
            self._dll.ReleaseOcrResult(result_handle)

    def close(self) -> None:
        if self._dll is not None:
            for export_name, handle in (
                ("ReleaseOcrProcessOptions", self._process_options),
                ("ReleaseOcrPipeline", self._pipeline),
                ("ReleaseOcrInitOptions", self._init_options),
            ):
                fn = getattr(self._dll, export_name, None)
                if fn is not None and getattr(handle, "value", 0):
                    try:
                        fn(handle)
                    except Exception:
                        pass
        self._process_options = c_int64()
        self._pipeline = c_int64()
        self._init_options = c_int64()
        if self._dll_dir_handle is not None:
            try:
                self._dll_dir_handle.close()
            except Exception:
                pass
            self._dll_dir_handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def _payload(engine: OneOcrEngine, path: str) -> dict:
    try:
        return {"ok": True, "path": path, "blocks": engine.recognize(path)}
    except Exception as exc:
        return {"ok": False, "path": path, "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("images", nargs="*")
    args = parser.parse_args()

    try:
        with OneOcrEngine(args.runtime_dir) as engine:
            if args.probe:
                print(json.dumps({"ok": True, "probe": True}, ensure_ascii=False), flush=True)
                return
            if not args.images:
                raise ValueError("没有输入图片")
            for path in args.images:
                print(json.dumps(_payload(engine, path), ensure_ascii=False), flush=True)
    except Exception as exc:
        print(json.dumps({
            "ok": False,
            "path": "",
            "probe": bool(args.probe),
            "error": str(exc),
        }, ensure_ascii=False), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
