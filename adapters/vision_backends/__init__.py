#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apple OCR backend registry.

Three explicit modes are kept side by side:
* native_helper: Swift Vision OCR (stable default, bbox/confidence/candidates)
* live_text: Swift VisionKit ImageAnalyzer / Live Text transcript
* shortcut: original macOS Shortcuts route

``auto`` remains accepted as a migration alias and resolves deterministically to
the native Vision backend. The GUI keeps every route explicit so OCR evidence
does not silently change between machines.
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
    def create(name: str = "native_helper", vertical: bool = True) -> VisionBackend:
        normalized = str(name or "native_helper").strip().lower()
        aliases = {
            # Settings written by older releases are migrated to the stable
            # explicit route instead of silently choosing a new backend.
            "auto": "native_helper",
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
        # Compatibility for plugins that explicitly ask for a resilient auto
        # chain. The GUI itself keeps the concrete backend explicit.
        from .auto_backend import AutoVisionBackend
        return AutoVisionBackend()

    @staticmethod
    def available_backends() -> list[str]:
        return ["native_helper", "live_text", "shortcut"]
