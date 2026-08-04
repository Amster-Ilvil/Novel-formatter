#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native Swift Vision OCR backend using RecognizeTextRequest.

The helper is built on the user's Mac from the bundled Swift source and kept
alive as a JSON-lines process.  GUI runs use QProcess when PySide6 is present;
CLI/tests fall back to subprocess.Popen.  The original Shortcuts backend remains
independent and selectable.
"""
from __future__ import annotations

import json
import os
import platform
import select
import shutil
import subprocess
import threading
import time
import uuid
import tempfile
from pathlib import Path

from .base import VisionBackend, OCRResult, OCRBlock, OCRConfig, BackendCapabilities

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tools" / "apple_vision_helper" / "AppleVisionOCRHelper.swift"
BINARY = ROOT / "tools" / "apple_vision_helper" / "bin" / "apple_vision_helper"
BUILD_SCRIPT = ROOT / "build_apple_vision_helper.command"


class HelperInfrastructureError(RuntimeError):
    """The helper could not be built, started, or communicated with."""


class VisionRecognitionError(RuntimeError):
    """Vision executed but rejected the image/request."""


def _mac_version_major() -> int:
    try:
        return int((platform.mac_ver()[0] or "0").split(".")[0])
    except Exception:
        return 0


def ensure_helper_binary() -> Path:
    if platform.system() != "Darwin":
        raise HelperInfrastructureError("Swift Vision Helper 仅支持 macOS")
    if _mac_version_major() < 15:
        raise HelperInfrastructureError("RecognizeTextRequest 需要 macOS 15 或更高版本")
    if not SOURCE.exists():
        raise HelperInfrastructureError(f"缺少 Swift Helper 源码：{SOURCE}")
    needs_build = not BINARY.exists()
    if BINARY.exists():
        try:
            needs_build = BINARY.stat().st_mtime < SOURCE.stat().st_mtime
        except OSError:
            needs_build = True
    if not needs_build:
        return BINARY
    if not shutil.which("xcrun"):
        raise HelperInfrastructureError("未找到 xcrun，请安装 Xcode 或 Xcode Command Line Tools")
    result = subprocess.run(
        [str(BUILD_SCRIPT)], cwd=str(ROOT), capture_output=True, text=True, timeout=180,
    )
    if result.returncode != 0 or not BINARY.exists():
        detail = (result.stderr or result.stdout or "Swift Helper 编译失败").strip()
        raise HelperInfrastructureError(f"Swift Vision Helper 编译失败：{detail}")
    return BINARY


class _SubprocessJSONClient:
    def __init__(self, binary: Path):
        from adapters.subprocess_watchdog import isolated_process_kwargs
        self._stderr_file = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
        self.process = subprocess.Popen(
            [str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr_file,
            text=True, encoding="utf-8", bufsize=1,
            **isolated_process_kwargs(),
        )

    def _stderr_tail(self) -> str:
        try:
            self._stderr_file.flush()
            self._stderr_file.seek(0)
            return self._stderr_file.read()[-4000:].strip()
        except Exception:
            return ""

    def request(self, payload: dict, timeout: float, cancel_check=None) -> dict:
        if self.process.poll() is not None:
            detail = self._stderr_tail()
            raise RuntimeError(f"Swift Vision Helper 已退出：{detail or self.process.returncode}")
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        deadline = time.monotonic() + max(1.0, float(timeout))
        while True:
            if callable(cancel_check) and cancel_check():
                from adapters.subprocess_watchdog import terminate_process
                terminate_process(self.process)
                raise InterruptedError("Apple Vision OCR 已停止，卡住的 Helper 已终止")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                from adapters.subprocess_watchdog import terminate_process
                terminate_process(self.process)
                raise TimeoutError("Swift Vision Helper OCR 超时，已自动终止卡住的 Helper")
            ready, _, _ = select.select(
                [self.process.stdout], [], [], min(0.25, remaining)
            )
            if ready:
                break
            if self.process.poll() is not None:
                detail = self._stderr_tail()
                raise RuntimeError(f"Swift Vision Helper 异常退出：{detail or self.process.returncode}")
        line = self.process.stdout.readline()
        if not line:
            detail = self._stderr_tail()
            raise RuntimeError(f"Swift Vision Helper 未返回结果：{detail}")
        return json.loads(line)

    def close(self):
        from adapters.subprocess_watchdog import terminate_process
        terminate_process(self.process)
        try:
            self._stderr_file.close()
        except Exception:
            pass


class _QtJSONClient:
    def __init__(self, binary: Path):
        from PySide6.QtCore import QProcess
        self.process = QProcess()
        self._stderr_tail_text = ""
        self.process.setProgram(str(binary))
        self.process.start()
        if not self.process.waitForStarted(10000):
            raise RuntimeError(self.process.errorString() or "QProcess 无法启动 Swift Vision Helper")

    def _drain_stderr(self) -> str:
        try:
            text = bytes(self.process.readAllStandardError()).decode("utf-8", "replace")
            if text:
                self._stderr_tail_text = (self._stderr_tail_text + text)[-4000:]
        except Exception:
            pass
        return self._stderr_tail_text.strip()

    def request(self, payload: dict, timeout: float, cancel_check=None) -> dict:
        from PySide6.QtCore import QProcess
        if self.process.state() == QProcess.ProcessState.NotRunning:
            detail = self._drain_stderr()
            raise RuntimeError(f"Swift Vision Helper 已退出：{detail or self.process.exitCode()}")
        packet = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        if self.process.write(packet) < 0 or not self.process.waitForBytesWritten(5000):
            raise RuntimeError("无法向 Swift Vision Helper 写入 OCR 请求")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if callable(cancel_check) and cancel_check():
                self.process.terminate()
                if not self.process.waitForFinished(750):
                    self.process.kill()
                raise InterruptedError("Apple Vision OCR 已停止，卡住的 Helper 已终止")
            if self.process.canReadLine():
                line = bytes(self.process.readLine()).decode("utf-8", "replace")
                return json.loads(line)
            remaining = max(1, int((deadline - time.monotonic()) * 1000))
            self.process.waitForReadyRead(min(remaining, 500))
            self._drain_stderr()
            if self.process.state() == QProcess.ProcessState.NotRunning:
                detail = self._drain_stderr()
                raise RuntimeError(f"Swift Vision Helper 异常退出：{detail or self.process.exitCode()}")
        self.process.terminate()
        if not self.process.waitForFinished(1500):
            self.process.kill()
        raise TimeoutError("Swift Vision Helper OCR 超时，已自动终止卡住的 Helper")

    def close(self):
        try:
            self.process.closeWriteChannel()
            self.process.terminate()
            if not self.process.waitForFinished(2000):
                self.process.kill()
        except Exception:
            pass


def _make_client(binary: Path):
    try:
        from PySide6.QtCore import QCoreApplication
        if QCoreApplication.instance() is not None:
            return _QtJSONClient(binary)
    except Exception:
        pass
    return _SubprocessJSONClient(binary)




def _prepare_vertical_column_image(image_path: str, mode: str) -> tuple[str, dict | None]:
    """Create one OCR input optimized for an isolated vertical column.

    The masked-column pipeline normally keeps the original full-page canvas and
    whites out every other column. Public Vision text recognition has no
    dedicated vertical-Japanese switch; feeding that very tall vertical strip
    directly is often worse than the Shortcuts/Live Text action.  In
    crop_rotate_left mode we detect an isolated narrow ink band, crop it with a
    conservative margin, rotate it 90° counter-clockwise so top→bottom becomes
    left→right, and OCR that image exactly once.

    Returns (request_path, transform). transform is used to map Vision boxes
    back to the original image coordinates. If the image is not clearly one
    isolated vertical column, the original path is returned unchanged.
    """
    if str(mode or "none") != "crop_rotate_left":
        return image_path, None
    try:
        from PIL import Image, ImageOps
        with Image.open(image_path) as opened:
            source = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = source.size
        if width < 8 or height < 8:
            source.close()
            return image_path, None
        gray = source.convert("L")
        # Very light threshold is intentional: preserve antialiasing and faint
        # printed strokes, but ignore the pure-white mask canvas.
        ink_mask = gray.point(lambda value: 255 if value < 246 else 0)
        bbox = ink_mask.getbbox()
        gray.close(); ink_mask.close()
        if not bbox:
            source.close()
            return image_path, None
        left, top, right, bottom = bbox
        ink_w = max(1, right - left)
        ink_h = max(1, bottom - top)
        narrow_on_page = ink_w <= width * 0.38 and ink_h >= ink_w * 1.35
        narrow_input = width <= height * 0.46 and ink_h >= ink_w * 1.25
        if not (narrow_on_page or narrow_input):
            source.close()
            return image_path, None
        # Do not rotate tiny specks or page numbers accidentally left alone.
        if ink_h < max(40, int(height * 0.12)):
            source.close()
            return image_path, None
        margin_x = max(10, int(ink_w * 0.22))
        margin_y = max(10, int(min(ink_h * 0.04, 48)))
        crop_left = max(0, left - margin_x)
        crop_top = max(0, top - margin_y)
        crop_right = min(width, right + margin_x)
        crop_bottom = min(height, bottom + margin_y)
        cropped = source.crop((crop_left, crop_top, crop_right, crop_bottom))
        source.close()
        crop_w, crop_h = cropped.size
        rotated = cropped.transpose(Image.Transpose.ROTATE_90)
        cropped.close()
        handle = tempfile.NamedTemporaryFile(prefix="nf_apple_vertical_", suffix=".png", delete=False)
        temp_path = handle.name
        handle.close()
        rotated.save(temp_path, format="PNG", optimize=False)
        rotated.close()
        return temp_path, {
            "original_width": width,
            "original_height": height,
            "crop_left": crop_left,
            "crop_top": crop_top,
            "crop_width": crop_w,
            "crop_height": crop_h,
            "rotation": "left",
        }
    except Exception:
        return image_path, None


def _map_bbox_from_vertical_preprocess(
    bbox: tuple[float, float, float, float], transform: dict | None,
) -> tuple[float, float, float, float]:
    """Map a Vision lower-left normalized box from rotated crop to original."""
    if not transform or transform.get("rotation") != "left":
        return bbox
    x, y, w, h = bbox
    original_w = float(transform["original_width"])
    original_h = float(transform["original_height"])
    crop_left = float(transform["crop_left"])
    crop_top = float(transform["crop_top"])
    crop_w = float(transform["crop_width"])
    crop_h = float(transform["crop_height"])
    # Rotated image dimensions after 90° CCW: width=crop_h, height=crop_w.
    rotated_w, rotated_h = crop_h, crop_w
    rx0 = x * rotated_w
    rx1 = (x + w) * rotated_w
    ry_top = (1.0 - (y + h)) * rotated_h
    ry_bottom = (1.0 - y) * rotated_h
    # Inverse of CCW rotation in upper-left pixel coordinates:
    # original_crop_x = crop_w - rotated_y; original_crop_y = rotated_x.
    ox0 = crop_left + (crop_w - ry_bottom)
    ox1 = crop_left + (crop_w - ry_top)
    oy0 = crop_top + rx0
    oy1 = crop_top + rx1
    ox0, ox1 = sorted((max(0.0, ox0), min(original_w, ox1)))
    oy0, oy1 = sorted((max(0.0, oy0), min(original_h, oy1)))
    mapped_x = ox0 / original_w
    mapped_y = (original_h - oy1) / original_h
    mapped_w = max(0.0, ox1 - ox0) / original_w
    mapped_h = max(0.0, oy1 - oy0) / original_h
    return (mapped_x, mapped_y, mapped_w, mapped_h)

class NativeVisionHelperBackend(VisionBackend):
    def __init__(self):
        self._client = None
        self._client_lock = threading.RLock()

    @property
    def name(self) -> str:
        return "native_helper"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            accurate=True,
            fast=True,
            bbox=True,
            confidence=True,
            language=True,
            language_correction=True,
            batch=True,
        )

    def is_available(self) -> tuple[bool, str]:
        if platform.system() != "Darwin":
            return False, "仅支持 macOS"
        if _mac_version_major() < 15:
            return False, "RecognizeTextRequest 需要 macOS 15 或更高版本"
        if not SOURCE.exists():
            return False, "缺少 AppleVisionOCRHelper.swift"
        if BINARY.exists() or shutil.which("xcrun"):
            return True, ""
        return False, "未找到 xcrun，请安装 Xcode 或 Xcode Command Line Tools"

    def recognize(self, image_path: str, config: OCRConfig) -> OCRResult:
        binary = ensure_helper_binary()
        # Character-box mode must be handled inside the Swift helper because
        # Vision's Character ranges and geometry must share the same request.
        # All existing Novel-formatter-1 calls continue to use the original
        # Python crop_rotate_left compatibility path.
        helper_vertical_mode = bool(
            config.vertical and (config.vertical_compatibility_mode or config.character_boxes)
        )
        request_path, transform = _prepare_vertical_column_image(
            image_path,
            "none" if helper_vertical_mode else (
                config.vertical_preprocess if config.vertical else "none"
            ),
        )
        transformed = transform is not None
        payload = {
            "id": uuid.uuid4().hex,
            "api": "recognize_text",
            "image": str(Path(request_path).resolve()),
            "languages": list(config.languages or ["ja-JP"]),
            "recognitionLevel": config.recognition_level,
            "automaticallyDetectsLanguage": bool(config.automatically_detect_language),
            "usesLanguageCorrection": bool(config.use_language_correction),
            "minimumTextHeightFraction": float(config.minimum_text_height_fraction),
            "candidateCount": int(config.candidate_count),
            # The generated PNG is already physically rotated and has no EXIF.
            "orientation": "up" if transformed else str(config.orientation or "auto"),
            # After Python-side CCW rotation the former vertical column is a
            # horizontal line. Helper-native mode receives the original column.
            "vertical": False if transformed else bool(config.vertical),
            "verticalCompatibilityMode": bool(helper_vertical_mode),
            "characterBoxes": bool(config.character_boxes),
        }
        try:
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
                raise HelperInfrastructureError(f"Swift Vision Helper 通信失败：{exc}") from exc
        finally:
            if transformed:
                try:
                    Path(request_path).unlink(missing_ok=True)
                except Exception:
                    pass
        if not response.get("success"):
            raise VisionRecognitionError(str(response.get("error") or "Swift Vision OCR 失败"))
        blocks: list[OCRBlock] = []
        for item in response.get("items") or []:
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            bbox = _map_bbox_from_vertical_preprocess((
                float(item.get("x", 0.0)), float(item.get("y", 0.0)),
                float(item.get("width", 0.0)), float(item.get("height", 0.0)),
            ), transform)
            candidates = [
                (str(candidate.get("text") or ""), float(candidate.get("confidence", 0.0)))
                for candidate in (item.get("candidates") or [])
                if str(candidate.get("text") or "").strip()
            ]
            blocks.append(OCRBlock(
                text=text,
                confidence=float(item.get("confidence", 0.0)),
                bbox=bbox,
                language=(config.languages[0] if config.languages else ""),
                candidates=candidates,
            ))
        return OCRResult(
            full_text=str(response.get("text") or "").strip(),
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

