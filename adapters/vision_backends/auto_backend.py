#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prefer Swift VisionKit Live Text; retain original Shortcuts fallback."""
from __future__ import annotations

from .base import VisionBackend, OCRResult, OCRConfig, BackendCapabilities
from .live_text_backend import LiveTextHelperBackend
from .native_helper_backend import HelperInfrastructureError
from .shortcut_backend import ShortcutBackend


class AutoVisionBackend(VisionBackend):
    def __init__(self):
        self._live_text = LiveTextHelperBackend()
        # Compatibility alias for older tests/extensions that inspected _native.
        self._native = self._live_text
        self._shortcut = ShortcutBackend()
        self._live_text_disabled = False
        self._native_disabled = False

    @property
    def name(self) -> str:
        return "auto"

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._live_text.capabilities

    def is_available(self) -> tuple[bool, str]:
        live_ok, live_reason = self._live_text.is_available()
        shortcut_ok, shortcut_reason = self._shortcut.is_available()
        if live_ok or shortcut_ok:
            return True, ""
        return False, f"Live Text Helper: {live_reason}；快捷指令: {shortcut_reason}"

    def recognize(self, image_path: str, config: OCRConfig) -> OCRResult:
        if not self._live_text_disabled and not self._native_disabled:
            available, _ = self._live_text.is_available()
            if available:
                try:
                    # Empty but successful Live Text output is final: do not run
                    # a second OCR merely because no text was found.
                    return self._live_text.recognize(image_path, config)
                except HelperInfrastructureError as exc:
                    self._live_text_disabled = True
                    self._native_disabled = True
                    self._live_text.close()
                    print(f"  ⚠️ Swift Live Text 基础设施/可用性失败，本次任务回退快捷指令：{exc}")
        return self._shortcut.recognize(image_path, config)

    def close(self) -> None:
        self._live_text.close()
        self._shortcut.close()
