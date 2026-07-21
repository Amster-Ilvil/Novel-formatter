#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vision Backend 注册表 + 工厂；Apple Vision 仅保留快捷指令后端。"""

from __future__ import annotations

from typing import Callable

from .base import VisionBackend, OCRResult, OCRBlock, OCRConfig, BackendCapabilities
from .shortcut_backend import ShortcutBackend
from .stub_backends import (
    SimpleOCRBackend, MacOCRBackend, SeeVBackend, OCRmyPDFAppleOCRBackend,
)

_REGISTRY: dict[str, Callable[[], VisionBackend]] = {
    "shortcut": ShortcutBackend,
    "simpleocr": SimpleOCRBackend,
    "mac-ocr": MacOCRBackend,
    "seeV": SeeVBackend,
    "ocrmypdf-appleocr": OCRmyPDFAppleOCRBackend,
}

_AUTO_PRIORITY = ["shortcut"]


class BackendFactory:

    @staticmethod
    def create(name: str = "shortcut", vertical: bool = True) -> VisionBackend:
        if name in {"auto", "", None}:
            name = "shortcut"
        if name != "shortcut":
            raise ValueError(f"不支持的 Apple Vision backend: {name}")
        return _REGISTRY[name]()

    @staticmethod
    def auto(vertical: bool = True) -> VisionBackend:
        return BackendFactory.create("shortcut")

    @staticmethod
    def available_backends() -> list[str]:
        return list(_REGISTRY.keys())
