#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility auto backend: native Vision -> Live Text -> Shortcuts.

The normal GUI keeps every Apple OCR route explicit for reproducibility.  This
backend exists for older plugins/callers that explicitly request ``auto``.  It
prefers the cross-architecture native Vision helper, then tries VisionKit Live
Text, and finally preserves the original Shortcuts route.
"""
from __future__ import annotations

from .base import VisionBackend, OCRResult, OCRConfig, BackendCapabilities
from .native_helper_backend import (
    NativeVisionHelperBackend,
    HelperInfrastructureError,
    VisionRecognitionError,
)
from .live_text_backend import LiveTextHelperBackend
from .shortcut_backend import ShortcutBackend


class AutoVisionBackend(VisionBackend):
    def __init__(self):
        self._native = NativeVisionHelperBackend()
        self._live_text = LiveTextHelperBackend()
        self._shortcut = ShortcutBackend()
        self._disabled: set[str] = set()

    @property
    def name(self) -> str:
        return "auto"

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._native.capabilities

    def is_available(self) -> tuple[bool, str]:
        reasons: list[str] = []
        for label, backend in (
            ("Native Vision", self._native),
            ("Live Text", self._live_text),
            ("快捷指令", self._shortcut),
        ):
            ok, reason = backend.is_available()
            if ok:
                return True, ""
            reasons.append(f"{label}: {reason}")
        return False, "；".join(reasons)

    def recognize(self, image_path: str, config: OCRConfig) -> OCRResult:
        failures: list[str] = []
        for key, backend in (
            ("native_helper", self._native),
            ("live_text", self._live_text),
            ("shortcut", self._shortcut),
        ):
            if key in self._disabled:
                continue
            available, reason = backend.is_available()
            if not available:
                failures.append(f"{key}: {reason}")
                continue
            try:
                return backend.recognize(image_path, config)
            except (HelperInfrastructureError, VisionRecognitionError, RuntimeError) as exc:
                self._disabled.add(key)
                try:
                    backend.close()
                except Exception:
                    pass
                failures.append(f"{key}: {exc}")
        raise HelperInfrastructureError(
            "Apple OCR 自动兼容链全部不可用：" + "；".join(failures[-3:])
        )

    def close(self) -> None:
        for backend in (self._native, self._live_text, self._shortcut):
            try:
                backend.close()
            except Exception:
                pass
