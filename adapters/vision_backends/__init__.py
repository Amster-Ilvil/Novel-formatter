#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apple OCR backend registry.

Three explicit modes are kept side by side:
* live_text: Swift VisionKit ImageAnalyzer / Live Text transcript
* native_helper: Swift Vision RecognizeTextRequest (bbox/confidence/candidates)
* shortcut: original macOS Shortcuts route

``auto`` remains accepted only as a migration alias for old saved settings. It
must never be presented as a selectable mode because silent backend switching
makes OCR failures difficult to diagnose and can change reading order/results.
"""
from __future__ import annotations

from typing import Callable

from .base import VisionBackend, OCRResult, OCRBlock, OCRConfig, BackendCapabilities
from .shortcut_backend import ShortcutBackend
from .native_helper_backend import NativeVisionHelperBackend
from .live_text_backend import LiveTextHelperBackend

_REGISTRY: dict[str, Callable[[], VisionBackend]] = {
    "live_text": LiveTextHelperBackend,
    "native_helper": NativeVisionHelperBackend,
    "shortcut": ShortcutBackend,
}


class BackendFactory:
    @staticmethod
    def create(name: str = "live_text", vertical: bool = True) -> VisionBackend:
        normalized = str(name or "live_text").strip().lower()
        aliases = {
            # Settings written by older releases are migrated to the stable
            # explicit route instead of silently choosing a new backend.
            "auto": "live_text",
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
        # Compatibility for plugins that still call BackendFactory.auto().
        # This is deliberately deterministic and is not a UI option.
        return _REGISTRY["live_text"]()

    @staticmethod
    def available_backends() -> list[str]:
        return ["live_text", "native_helper", "shortcut"]
