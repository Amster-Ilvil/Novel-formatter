#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Swift VisionKit ImageAnalyzer (Live Text) backend.

This is a separate OCR route from RecognizeTextRequest. It submits the original
masked-column image exactly once and consumes ImageAnalysis.transcript, which is
closer to the system Live Text/Shortcuts behaviour. It does not fake bbox,
confidence or alternative candidates that ImageAnalysis.transcript doesn't
expose through this helper.
"""
from __future__ import annotations

import platform
import shutil
import uuid
from pathlib import Path

from .base import VisionBackend, OCRResult, OCRBlock, OCRConfig, BackendCapabilities
from .native_helper_backend import (
    SOURCE, BINARY, ensure_helper_binary, _make_client, _mac_version_major,
    HelperInfrastructureError, VisionRecognitionError,
)


class LiveTextHelperBackend(VisionBackend):
    def __init__(self):
        import threading
        self._client = None
        self._client_lock = threading.RLock()

    @property
    def name(self) -> str:
        return "live_text"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            language=True,
            vertical_text=True,
            batch=True,
        )

    def is_available(self) -> tuple[bool, str]:
        if platform.system() != "Darwin":
            return False, "仅支持 macOS"
        if _mac_version_major() < 13:
            return False, "VisionKit ImageAnalyzer 需要 macOS 13 或更高版本"
        if not SOURCE.exists():
            return False, "缺少 AppleVisionOCRHelper.swift"
        if BINARY.exists() or shutil.which("xcrun"):
            return True, ""
        return False, "未找到 xcrun，请安装 Xcode 或 Xcode Command Line Tools"

    def recognize(self, image_path: str, config: OCRConfig) -> OCRResult:
        binary = ensure_helper_binary()
        payload = {
            "id": uuid.uuid4().hex,
            "api": "live_text",
            "image": str(Path(image_path).resolve()),
            "languages": list(config.languages or ["ja-JP"]),
            # Live Text reads the original full-size masked column. Do not
            # rotate/crop or submit a second OCR request behind the user's back.
            "orientation": str(config.orientation or "auto"),
            "vertical": bool(config.vertical),
        }
        try:
            with self._client_lock:
                if self._client is None:
                    self._client = _make_client(binary)
                try:
                    response = self._client.request(
                        payload,
                        max(1.0, float(config.timeout)),
                        cancel_check=getattr(self, "cancel_check", None),
                    )
                except TypeError as exc:
                    # Compatibility with older plugin/test clients that
                    # still expose request(payload, timeout).
                    if "cancel_check" not in str(exc):
                        raise
                    response = self._client.request(
                        payload, max(1.0, float(config.timeout))
                    )
        except HelperInfrastructureError:
            raise
        except Exception as exc:
            self.close()
            raise HelperInfrastructureError(f"Swift Live Text Helper 通信失败：{exc}") from exc

        if not response.get("success"):
            error = str(response.get("error") or "Swift Live Text OCR 失败")
            if error == "live_text_unsupported":
                raise HelperInfrastructureError("当前 Mac 不支持 VisionKit ImageAnalyzer Live Text")
            raise VisionRecognitionError(error)

        text = str(response.get("text") or "").strip()
        blocks = [OCRBlock(text=line) for line in text.splitlines() if line.strip()]
        return OCRResult(
            full_text=text,
            blocks=blocks,
            language=(config.languages[0] if config.languages else ""),
        )

    def close(self) -> None:
        with self._client_lock:
            if self._client is not None:
                try:
                    self._client.close()
                finally:
                    self._client = None
