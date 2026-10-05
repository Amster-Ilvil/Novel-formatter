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
import hashlib
import platform
import select
import shutil
import stat
import struct
import subprocess
import threading
import time
import uuid
import tempfile
import zlib
from pathlib import Path

from .base import VisionBackend, OCRResult, OCRBlock, OCRConfig, BackendCapabilities
from adapters.geometry_transform import AffineMatrix

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tools" / "apple_vision_helper" / "AppleVisionOCRHelper.swift"
BUILD_SCRIPT = ROOT / "build_apple_vision_helper.command"
_RUNTIME_HELPER_ROOT = Path(
    os.environ.get(
        "NOVEL_FORMATTER_APPLE_VISION_HELPER_DIR",
        str(Path.home() / "Library" / "Caches" / "NovelFormatter" / "apple_vision_helper"),
    )
).expanduser()
BINARY = _RUNTIME_HELPER_ROOT / "apple_vision_helper"
BUILD_STAMP = BINARY.with_suffix(".sdk-version")
BUILD_ARCH_STAMP = BINARY.with_suffix(".arch")
BUILD_SOURCE_STAMP = BINARY.with_suffix(".source-sha256")


class HelperInfrastructureError(RuntimeError):
    """The helper could not be built, started, or communicated with."""


class VisionRecognitionError(RuntimeError):
    """Vision executed but rejected the image/request."""


def _mac_version_major() -> int:
    try:
        return int((platform.mac_ver()[0] or "0").split(".")[0])
    except Exception:
        return 0


def _current_sdk_version() -> str:
    """Return the active macOS SDK version when xcrun is available."""
    if not shutil.which("xcrun"):
        return ""
    try:
        result = subprocess.run(
            ["xcrun", "--sdk", "macosx", "--show-sdk-version"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return str(result.stdout or "").strip()
    except Exception:
        pass
    return ""


def _helper_source_sha256() -> str:
    try:
        return hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    except OSError:
        return ""


def _helper_binary_signature(binary: Path) -> tuple[int, int, int]:
    stat_result = binary.stat()
    return (int(stat_result.st_ino), int(stat_result.st_size), int(stat_result.st_mtime_ns))


def _helper_update_available(*, sdk_version: str = "", architecture: str = "", source_hash: str = "") -> bool:
    """Return whether bundled source/SDK metadata differs from the active helper.

    This is deliberately informational.  Source ZIP updates and Xcode/SDK
    updates must never replace a helper that has already proven usable on the
    user's Mac.  A new build is promoted only when there is no runnable helper,
    the CPU architecture changed, or the user explicitly requests a rebuild.
    """
    if not BINARY.exists():
        return True
    if source_hash:
        try:
            if BUILD_SOURCE_STAMP.read_text(encoding="utf-8").strip() != source_hash:
                return True
        except OSError:
            return True
    if sdk_version:
        try:
            if BUILD_STAMP.read_text(encoding="utf-8").strip() != sdk_version:
                return True
        except OSError:
            return True
    if architecture:
        try:
            built_arch = BUILD_ARCH_STAMP.read_text(encoding="utf-8").strip().lower()
        except OSError:
            return True
        expected = str(architecture or "").strip().lower()
        aliases = {"aarch64": "arm64", "amd64": "x86_64"}
        if aliases.get(built_arch, built_arch) != aliases.get(expected, expected):
            return True
    return False


def _helper_needs_build(*, sdk_version: str = "", architecture: str = "", source_hash: str = "") -> bool:
    """Return only *hard* rebuild requirements.

    Historically this function rebuilt the persistent helper whenever the
    source hash or active SDK changed.  That made replacing the application ZIP
    capable of replacing a known-good local Apple OCR binary.  The persistent
    helper is now a last-known-good runtime: source/SDK drift alone is not an
    automatic rebuild trigger.
    """
    if not BINARY.exists():
        return True
    if architecture:
        try:
            built_arch = BUILD_ARCH_STAMP.read_text(encoding="utf-8").strip().lower()
        except OSError:
            # Missing old stamp is not enough reason to destroy a working helper.
            return False
        expected = str(architecture or "").strip().lower()
        aliases = {"aarch64": "arm64", "amd64": "x86_64"}
        if aliases.get(built_arch, built_arch) != aliases.get(expected, expected):
            return True
    return False


def _force_helper_rebuild_requested() -> bool:
    value = os.environ.get("NOVEL_FORMATTER_APPLE_VISION_HELPER_REBUILD", "").strip().lower()
    return value in {"1", "true", "yes", "on", "force"}


def _ensure_helper_executable(binary: Path) -> Path:
    """Restore the executable bit when an updater/ZIP extractor stripped it.

    Python's :mod:`zipfile` does not reliably restore POSIX executable bits on
    extraction.  Portable source updates can therefore leave the bundled Swift
    helper as ``0644`` even though the archive stored it as ``0755``.  Repair
    only the execute bits in place; never alter contents or clear macOS security
    metadata such as quarantine attributes.
    """
    try:
        current_mode = binary.stat().st_mode
    except OSError as exc:
        raise HelperInfrastructureError(f"无法读取 Swift Vision Helper 权限：{exc}") from exc
    if not stat.S_ISREG(current_mode):
        raise HelperInfrastructureError(f"Swift Vision Helper 不是普通文件：{binary}")
    if os.access(binary, os.X_OK):
        return binary
    try:
        # Match the build script's chmod +x semantics while preserving every
        # existing read/write bit from the extracted file.
        binary.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except OSError as exc:
        raise HelperInfrastructureError(
            f"Swift Vision Helper 缺少执行权限且自动修复失败：{binary}（{exc}）"
        ) from exc
    if not os.access(binary, os.X_OK):
        raise HelperInfrastructureError(
            f"Swift Vision Helper 仍不可执行：{binary}。请确认项目目录不是 noexec 文件系统。"
        )
    return binary


def _write_probe_png(path: Path, width: int = 64, height: int = 64) -> None:
    """Write a tiny valid RGB PNG using only the standard library."""
    width = max(8, int(width)); height = max(8, int(height))
    raw = b"".join(b"\x00" + (b"\xff\xff\xff" * width) for _ in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    payload = b"\x89PNG\r\n\x1a\n"
    payload += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += chunk(b"IDAT", zlib.compress(raw, 9))
    payload += chunk(b"IEND", b"")
    path.write_bytes(payload)


def _probe_helper_binary(binary: Path) -> None:
    """Require a candidate helper to start and execute the supported OCR APIs.

    The probe uses a generated blank PNG, so it does not depend on project
    fixtures or user files.  macOS 13/14 validate the VisionKit path; macOS 15+
    additionally validate native Vision.  Empty OCR text is acceptable, but an
    API/process error is not.
    """
    binary = _ensure_helper_executable(Path(binary))
    with tempfile.TemporaryDirectory(prefix="novel_formatter_apple_helper_probe_") as tmp:
        image = Path(tmp) / "probe.png"
        _write_probe_png(image)
        client = _SubprocessJSONClient(binary)
        try:
            apis = ["live_text"]
            if _mac_version_major() >= 15:
                apis.append("recognize_text")
            for api in apis:
                response = client.request({
                    "id": f"selftest-{api}",
                    "api": api,
                    "image": str(image),
                    "languages": ["ja-JP"],
                    "recognitionLevel": "accurate",
                    "automaticallyDetectsLanguage": True,
                    "usesLanguageCorrection": False,
                    "minimumTextHeightFraction": 0.0,
                    "candidateCount": 1,
                    "orientation": "up",
                    "vertical": True,
                    "characterBoxes": False,
                }, timeout=45.0)
                if not bool(response.get("success")):
                    raise HelperInfrastructureError(
                        f"候选 Apple Vision Helper 自检失败（{api}）："
                        f"{response.get('error') or 'unknown error'}"
                    )
        finally:
            client.close()


def _promote_candidate_helper(candidate_root: Path, *, sdk_version: str, architecture: str, source_hash: str) -> Path:
    candidate = candidate_root / "apple_vision_helper"
    if not candidate.exists():
        raise HelperInfrastructureError("候选 Apple Vision Helper 编译后不存在")
    _probe_helper_binary(candidate)

    BINARY.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the current binary as an emergency last-known-good backup before
    # any explicit/architecture rebuild is promoted.  Promotion itself is
    # atomic at the executable path; ordinary source ZIP updates never enter
    # this path at all.
    backup = BINARY.with_suffix(".last-known-good")
    if BINARY.exists():
        try:
            shutil.copy2(BINARY, backup)
            _ensure_helper_executable(backup)
        except OSError:
            pass

    staged = BINARY.with_name(BINARY.name + f".promote-{uuid.uuid4().hex}")
    shutil.copy2(candidate, staged)
    _ensure_helper_executable(staged)
    os.replace(staged, BINARY)
    try:
        if sdk_version:
            BUILD_STAMP.write_text(sdk_version + "\n", encoding="utf-8")
        if architecture:
            BUILD_ARCH_STAMP.write_text(architecture + "\n", encoding="utf-8")
        if source_hash:
            BUILD_SOURCE_STAMP.write_text(source_hash + "\n", encoding="utf-8")
    except OSError:
        pass
    return _ensure_helper_executable(BINARY)


def _build_candidate_helper(*, sdk_version: str, architecture: str, source_hash: str) -> Path:
    if not shutil.which("xcrun"):
        raise HelperInfrastructureError("Swift Helper 需要构建，但未找到 xcrun；请安装 Xcode 或 Xcode Command Line Tools")
    BINARY.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="novel_formatter_apple_helper_candidate_", dir=str(BINARY.parent)) as tmp:
        candidate_root = Path(tmp)
        env = os.environ.copy()
        env["NOVEL_FORMATTER_APPLE_VISION_HELPER_DIR"] = str(candidate_root)
        env["NOVEL_FORMATTER_APPLE_VISION_HELPER_SOURCE_SHA256"] = source_hash
        result = subprocess.run(
            ["/bin/bash", str(BUILD_SCRIPT)], cwd=str(ROOT), env=env,
            capture_output=True, text=True, timeout=180,
        )
        candidate = candidate_root / "apple_vision_helper"
        if result.returncode != 0 or not candidate.exists():
            detail = (result.stderr or result.stdout or "Swift Helper 编译失败").strip()
            raise HelperInfrastructureError(f"Swift Vision Helper 候选编译失败：{detail}")
        return _promote_candidate_helper(
            candidate_root, sdk_version=sdk_version, architecture=architecture, source_hash=source_hash,
        )


def ensure_helper_binary() -> Path:
    if platform.system() != "Darwin":
        raise HelperInfrastructureError("Swift Vision Helper 仅支持 macOS")
    if _mac_version_major() < 13:
        raise HelperInfrastructureError("Apple Vision Helper 需要 macOS 13 或更高版本")
    if not SOURCE.exists():
        raise HelperInfrastructureError(f"缺少 Swift Helper 源码：{SOURCE}")

    sdk_version = _current_sdk_version()
    architecture = platform.machine().strip().lower()
    source_hash = _helper_source_sha256()
    force = _force_helper_rebuild_requested()

    # Normal application/source updates are intentionally read-only with
    # respect to the persistent Apple OCR runtime.  A known-good binary remains
    # selected even when the bundled Swift source or active SDK changed.
    if BINARY.exists() and not force and not _helper_needs_build(architecture=architecture):
        return _ensure_helper_executable(BINARY)

    return _build_candidate_helper(
        sdk_version=sdk_version, architecture=architecture, source_hash=source_hash,
    )


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
    # Defensive second check: permissions may change after the build check
    # (for example a source updater swapped the project tree).
    binary = _ensure_helper_executable(binary)
    try:
        from PySide6.QtCore import QCoreApplication
        if QCoreApplication.instance() is not None:
            return _QtJSONClient(binary)
    except Exception:
        pass
    return _SubprocessJSONClient(binary)




def _request_with_resilient_client(owner, binary: Path, payload: dict, timeout: float, cancel_check=None) -> dict:
    """Reconnect after helper replacement/broken pipe and self-heal hard runtime failures.

    OCR/vision errors returned as JSON are not infrastructure failures and never
    trigger a rebuild.  Only process/pipe failures can reach the candidate-build
    recovery, and the candidate must pass both API probes before promotion.
    """
    binary = Path(binary)

    def signature_for(path: Path):
        if path.exists():
            return _helper_binary_signature(path)
        return getattr(owner, "_client_binary_signature", None)

    signature = signature_for(binary)
    with owner._client_lock:
        if getattr(owner, "_client_binary_signature", None) != signature:
            if getattr(owner, "_client", None) is not None:
                try:
                    owner._client.close()
                except Exception:
                    pass
            owner._client = None
            owner._client_binary_signature = None

        last_exc = None
        for attempt in range(3):
            if owner._client is None:
                owner._client = _make_client(binary)
                owner._client_binary_signature = signature_for(binary)
            try:
                try:
                    return owner._client.request(
                        payload, max(1.0, float(timeout)), cancel_check=cancel_check,
                    )
                except TypeError as exc:
                    if "cancel_check" not in str(exc):
                        raise
                    return owner._client.request(payload, max(1.0, float(timeout)))
            except (InterruptedError, TimeoutError):
                raise
            except Exception as exc:
                last_exc = exc
                try:
                    owner._client.close()
                except Exception:
                    pass
                owner._client = None
                owner._client_binary_signature = None
                if attempt == 0:
                    # First retry is cheap and handles a stale pipe/client after
                    # an external replacement without touching the runtime.
                    continue
                if (
                    attempt == 1
                    and platform.system() == "Darwin"
                    and binary == BINARY
                    and SOURCE.exists()
                    and shutil.which("xcrun")
                ):
                    # The selected last-known-good helper itself is no longer
                    # launchable. Build a candidate separately; promotion occurs
                    # only after the full helper probe succeeds.
                    try:
                        binary = _build_candidate_helper(
                            sdk_version=_current_sdk_version(),
                            architecture=platform.machine().strip().lower(),
                            source_hash=_helper_source_sha256(),
                        )
                        signature = signature_for(binary)
                        continue
                    except Exception:
                        pass
                break
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Swift Vision Helper 通信失败")


def _prepare_vertical_column_image(image_path: str, mode: str) -> tuple[str, dict | None]:
    """Create one OCR input optimized for an isolated vertical column.

    The masked-column pipeline normally keeps the original full-page canvas and
    whites out every other column. Public Vision text recognition has no
    dedicated vertical-Japanese switch; feeding that very tall vertical strip
    directly is often worse than the Shortcuts/Live Text action.  In
    crop_rotate_left mode we detect an isolated narrow ink band, preserve a
    deliberately wide paper context around it, rotate that viewport 90°
    counter-clockwise so top→bottom becomes left→right, and OCR that image
    exactly once.  The wider context avoids recreating the historical 40--70 px
    "needle crop" after the column pipeline has already isolated the body.

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
        # Keep enough white paper around the isolated column for Vision's
        # recognizer.  A human crop that works well is usually several glyph
        # widths wide; the old 22% side margin could collapse a 42 px body band
        # to roughly 60 px even when the upstream visibility viewport was 280+
        # px wide.  Preserve up to 256 px of the input context without resizing.
        target_width = min(
            width,
            max(
                96,
                int(round(ink_w * 5.0)),
                min(256, int(round(width * 0.72))),
            ),
        )
        center_x = (left + right) / 2.0
        crop_left = int(round(center_x - target_width / 2.0))
        crop_left = max(0, min(width - target_width, crop_left))
        crop_right = crop_left + target_width
        margin_y = max(10, int(min(ink_h * 0.04, 48)))
        crop_top = max(0, top - margin_y)
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
        # Pixel-space mapping from the physically rotated OCR image back to
        # the original page (upper-left coordinates):
        #   original_x = crop_left + crop_width - rotated_y
        #   original_y = crop_top + rotated_x
        to_original = AffineMatrix(
            0.0, -1.0, float(crop_left + crop_w),
            1.0,  0.0, float(crop_top),
        )
        return temp_path, {
            "original_width": width,
            "original_height": height,
            "crop_left": crop_left,
            "crop_top": crop_top,
            "crop_width": crop_w,
            "crop_height": crop_h,
            "rotation": "left",
            "to_original_affine": to_original.to_list(),
            "coordinate_contract": "pixel-upper-left/preprocessed-to-original",
        }
    except Exception:
        return image_path, None


def _map_bbox_from_vertical_preprocess(
    bbox: tuple[float, float, float, float], transform: dict | None,
) -> tuple[float, float, float, float]:
    """Map a Vision lower-left normalized box from preprocessed image to original.

    New preprocessors serialize ``to_original_affine``.  Legacy crop/rotation
    fields remain supported so old cached metadata and tests keep working.
    """
    if not transform or transform.get("rotation") != "left":
        return bbox
    x, y, w, h = (float(value) for value in bbox)
    original_w = float(transform["original_width"])
    original_h = float(transform["original_height"])
    crop_w = float(transform["crop_width"])
    crop_h = float(transform["crop_height"])
    rotated_w, rotated_h = crop_h, crop_w

    # Vision boxes use normalized lower-left coordinates. Convert once to the
    # common upper-left pixel contract before applying adapter-independent affine.
    rx0 = x * rotated_w
    rx1 = (x + w) * rotated_w
    ry0 = (1.0 - (y + h)) * rotated_h
    ry1 = (1.0 - y) * rotated_h

    matrix_value = transform.get("to_original_affine")
    if matrix_value is not None:
        to_original = AffineMatrix.from_value(matrix_value)
    else:
        # Backward-compatible reconstruction for metadata created before the
        # generic affine contract was introduced.
        crop_left = float(transform["crop_left"])
        crop_top = float(transform["crop_top"])
        to_original = AffineMatrix(
            0.0, -1.0, crop_left + crop_w,
            1.0,  0.0, crop_top,
        )

    ox0, oy0, ox1, oy1 = to_original.map_bbox((rx0, ry0, rx1, ry1))
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
        self._client_binary_signature = None
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
        # The project already has a tested crop/rotate + affine-backmapping path.
        # Reuse it for helper compatibility / character anchors instead of keeping
        # a second, hidden geometry implementation in Swift.  Character ranges and
        # their boundingBox(for:) calls still happen inside one Vision request.
        helper_vertical_mode = bool(
            config.vertical and (config.vertical_compatibility_mode or config.character_boxes)
        )
        preprocess_mode = (
            "crop_rotate_left" if helper_vertical_mode
            else (config.vertical_preprocess if config.vertical else "none")
        )
        request_path, transform = _prepare_vertical_column_image(
            image_path, preprocess_mode
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
            # horizontal line.  No duplicate Swift-side rotation is performed.
            "vertical": False if transformed else bool(config.vertical),
            "characterBoxes": bool(config.character_boxes),
        }
        try:
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
            observation_languages = [
                str(value) for value in (item.get("recognitionLanguages") or []) if str(value)
            ]
            blocks.append(OCRBlock(
                text=text,
                confidence=float(item.get("confidence", 0.0)),
                bbox=bbox,
                language=(observation_languages[0] if observation_languages else (
                    config.languages[0] if config.languages else ""
                )),
                candidates=candidates,
                text_direction=str(item.get("textDirection") or ""),
                should_wrap_to_next_line=(
                    bool(item.get("shouldWrapToNextLine"))
                    if item.get("shouldWrapToNextLine") is not None else None
                ),
                recognition_languages=observation_languages,
                is_title=(bool(item.get("isTitle")) if item.get("isTitle") is not None else None),
            ))
        return OCRResult(
            full_text=str(response.get("text") or "").strip(),
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
