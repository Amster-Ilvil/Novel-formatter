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
import uuid
from pathlib import Path

from .base import VisionBackend, OCRResult, OCRBlock, OCRConfig, BackendCapabilities
from .native_helper_backend import (
    SOURCE, BINARY, ensure_helper_binary, _mac_version_major,
    _request_with_resilient_client, _swift_toolchain_available,
    HelperInfrastructureError, VisionRecognitionError,
)


class LiveTextHelperBackend(VisionBackend):
    def __init__(self):
        import threading
        self._client = None
        self._client_binary_signature = None
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
        if BINARY.exists() or _swift_toolchain_available():
            return True, ""
        return False, "首次使用需要 Swift 编译器；请安装 Xcode Command Line Tools 或 Xcode"

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
            response = _request_with_resilient_client(
                self,
                binary,
                payload,
                max(1.0, float(config.timeout)),
                cancel_check=getattr(self, "cancel_check", None),
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
        # ImageAnalysis.transcript exposes neither per-line confidence nor
        # alternatives. Keep confidence explicitly unknown instead of inheriting
        # OCRBlock's numeric default and presenting it as measured certainty.
        blocks = [OCRBlock(text=line, confidence=0.0) for line in text.splitlines() if line.strip()]
        return OCRResult(
            full_text=text,
            blocks=blocks,
            language=(config.languages[0] if config.languages else ""),
            metadata=dict(response.get("metadata") or {}),
        )

    def close(self) -> None:
        with self._client_lock:
            if self._client is not None:
                try:
                    self._client.close()
                finally:
                    self._client = None
                    self._client_binary_signature = None
