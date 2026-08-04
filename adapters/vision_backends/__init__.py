#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apple OCR backend registry.

Four modes are kept side by side:
* live_text: Swift VisionKit ImageAnalyzer / Live Text transcript
* native_helper: Swift Vision RecognizeTextRequest (bbox/confidence/candidates)
* shortcut: original macOS Shortcuts route
* auto: Live Text first, shortcut only on helper infrastructure/availability failure
"""
from __future__ import annotations

from typing import Callable

from .base import VisionBackend, OCRResult, OCRBlock, OCRConfig, BackendCapabilities
from .shortcut_backend import ShortcutBackend
from .native_helper_backend import NativeVisionHelperBackend
from .live_text_backend import LiveTextHelperBackend
from .auto_backend import AutoVisionBackend

_REGISTRY: dict[str, Callable[[], VisionBackend]] = {
    "auto": AutoVisionBackend,
    "live_text": LiveTextHelperBackend,
    "native_helper": NativeVisionHelperBackend,
    "shortcut": ShortcutBackend,
}


class BackendFactory:
    @staticmethod
    def create(name: str = "auto", vertical: bool = True) -> VisionBackend:
        normalized = str(name or "auto").strip().lower()
        aliases = {
            "livetext": "live_text",
            "live-text": "live_text",
            "visionkit": "live_text",
            "image_analyzer": "live_text",
            "native": "native_helper",
            "swift": "native_helper",
            "helper": "native_helper",
            "shortcuts": "shortcut",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in _REGISTRY:
            raise ValueError(f"不支持的 Apple Vision backend: {name}")
        return _REGISTRY[normalized]()

    @staticmethod
    def auto(vertical: bool = True) -> VisionBackend:
        return _REGISTRY["auto"]()

    @staticmethod
    def available_backends() -> list[str]:
        return ["auto", "live_text", "native_helper", "shortcut"]
