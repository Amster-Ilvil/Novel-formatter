#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native Apple PencilKit PKStrokeRecognizer bridge.

The Python application extracts black-ink skeleton paths and sends only one
glyph at a time as point sequences plus pen-up boundaries. The protocol-11
helper keeps one AppKit process alive for the whole OCR run,
but each recognition request still creates a fresh Apple recognizer, submits
exactly one PKDrawing, reads the first Japanese result, and clears/discards
that drawing before the next glyph. This mirrors the working single-character
manual input panel while removing repeated process/framework startup.

PKStrokeRecognizer is an OS 27 API. This module performs strict runtime and SDK
checks and never pretends that the Apple backend is available on older systems.
The GUI keeps unresolved equal-grid cells as placeholders when unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import platform
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
import selectors
import threading
from typing import Any, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_NATIVE_ROOT = _PROJECT_ROOT / "native" / "AppleStrokeRecognizer"
_APP_BUNDLE = _NATIVE_ROOT / "AppleStrokeRecognizer.app"
_BINARY = _NATIVE_ROOT / "bin" / "apple-stroke-recognizer"
_BUILD_SCRIPT = _NATIVE_ROOT / "build.command"
_DEBUG_ROOT = _PROJECT_ROOT / "debug" / "apple_pkstroke"
_LATEST_PAYLOAD = _DEBUG_ROOT / "latest-auto-input.payload.json"


@dataclass(slots=True)
class AppleStrokeStatus:
    available: bool
    detail: str
    os_version: str = ""
    sdk_version: str = ""
    binary_path: str = ""
    japanese_supported: bool = False
    supported_languages: tuple[str, ...] = ()
    recognition_version: int | None = None
    protocol_version: int = 0


def _major(version: str) -> int:
    match = re.match(r"\s*(\d+)", str(version or ""))
    return int(match.group(1)) if match else 0


def _run_text(args: Sequence[str], *, timeout: float = 20.0) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            list(args), capture_output=True, text=True, timeout=timeout, check=False,
        )
        return int(proc.returncode), str(proc.stdout or "").strip(), str(proc.stderr or "").strip()
    except Exception as exc:
        return 127, "", str(exc)


def macos_version() -> str:
    if platform.system() != "Darwin":
        return ""
    code, stdout, _stderr = _run_text(["/usr/bin/sw_vers", "-productVersion"])
    return stdout if code == 0 else platform.mac_ver()[0]


def sdk_version() -> str:
    if platform.system() != "Darwin":
        return ""
    code, stdout, _stderr = _run_text(["/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-version"])
    return stdout if code == 0 else ""


def _parse_bridge_json(stdout: str) -> dict[str, Any]:
    lines = [line.strip() for line in str(stdout or "").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Apple PKStrokeRecognizer 桥接没有返回 JSON")
    try:
        payload = json.loads(lines[-1])
    except Exception as exc:
        raise RuntimeError(f"Apple PKStrokeRecognizer 返回内容无法解析：{lines[-1][:300]}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Apple PKStrokeRecognizer 返回的不是 JSON 对象")
    return payload


def build_bridge() -> Path:
    if platform.system() != "Darwin":
        raise RuntimeError("Apple PKStrokeRecognizer 只能在 macOS 上编译和运行")
    if not _BUILD_SCRIPT.exists():
        raise RuntimeError(f"缺少 Swift 构建脚本：{_BUILD_SCRIPT}")
    current_sdk = sdk_version()
    if _major(current_sdk) < 27:
        raise RuntimeError(
            f"当前 macOS SDK 为 {current_sdk or '未知'}；需要安装 Xcode 27 或更高版本"
        )
    proc = subprocess.run(
        [str(_BUILD_SCRIPT)], cwd=str(_NATIVE_ROOT), capture_output=True,
        text=True, timeout=180, check=False,
    )
    if proc.returncode != 0 or not _BINARY.exists():
        detail = (proc.stderr or proc.stdout or "未知编译错误").strip()
        raise RuntimeError(f"Apple PKStrokeRecognizer Swift 桥接编译失败：\n{detail}")
    _BINARY.chmod(_BINARY.stat().st_mode | 0o111)
    return _BINARY


def bridge_status(*, auto_build: bool = False) -> AppleStrokeStatus:
    os_ver = macos_version()
    sdk_ver = sdk_version()
    if platform.system() != "Darwin":
        return AppleStrokeStatus(False, "仅支持 macOS", os_version=os_ver, sdk_version=sdk_ver)
    if _major(os_ver) < 27:
        return AppleStrokeStatus(
            False, f"当前 macOS {os_ver or '未知'}；PKStrokeRecognizer 需要 macOS 27 或更高版本",
            os_version=os_ver, sdk_version=sdk_ver,
        )
    if not _BINARY.exists() and auto_build:
        try:
            build_bridge()
        except Exception as exc:
            return AppleStrokeStatus(False, str(exc), os_version=os_ver, sdk_version=sdk_ver)
    if not _BINARY.exists():
        sdk_note = "" if _major(sdk_ver) >= 27 else f"；当前 SDK {sdk_ver or '未知'}，还需 Xcode 27"
        return AppleStrokeStatus(
            False, f"尚未编译 Apple Swift 桥接{sdk_note}",
            os_version=os_ver, sdk_version=sdk_ver, binary_path=str(_BINARY),
        )
    code, stdout, stderr = _run_text([str(_BINARY), "--status"], timeout=30)
    if code != 0:
        return AppleStrokeStatus(
            False, f"Apple Swift 桥接状态检查失败：{stderr or stdout}",
            os_version=os_ver, sdk_version=sdk_ver, binary_path=str(_BINARY),
        )
    try:
        payload = _parse_bridge_json(stdout)
    except Exception as exc:
        return AppleStrokeStatus(
            False, str(exc), os_version=os_ver, sdk_version=sdk_ver,
            binary_path=str(_BINARY),
        )
    protocol_version = int(payload.get("bridgeProtocolVersion") or 0)
    if protocol_version < 11:
        if auto_build:
            try:
                build_bridge()
                return bridge_status(auto_build=False)
            except Exception as exc:
                return AppleStrokeStatus(
                    False, f"检测到旧版 Apple 桥接，自动重编译失败：{exc}",
                    os_version=os_ver, sdk_version=sdk_ver, binary_path=str(_BINARY),
                )
        return AppleStrokeStatus(
            False,
            "检测到旧版 Apple 桥接，请点击“编译 Apple 桥接”重新生成含常驻逐字识别服务、结果轮询与复核增量笔画支持的新版 App",
            os_version=os_ver, sdk_version=sdk_ver, binary_path=str(_BINARY),
        )
    languages = tuple(str(item) for item in payload.get("supportedLanguages", []) if item)
    japanese = bool(payload.get("japaneseSupported", False))
    ok = bool(payload.get("ok", False)) and japanese
    detail = "Apple PKStrokeRecognizer 可用，日语模型已由系统提供" if ok else str(
        payload.get("error") or "设备未提供日语手写识别"
    )
    version = payload.get("recognitionVersion")
    try:
        version = int(version) if version is not None else None
    except Exception:
        version = None
    return AppleStrokeStatus(
        ok, detail, os_version=os_ver, sdk_version=sdk_ver,
        binary_path=str(_BINARY), japanese_supported=japanese,
        supported_languages=languages, recognition_version=version,
        protocol_version=protocol_version,
    )


def runtime_available(*, auto_build: bool = False) -> tuple[bool, str]:
    status = bridge_status(auto_build=auto_build)
    return status.available, status.detail

def launch_manual_test_panel(*, auto_build: bool = True, payload_path: str | os.PathLike[str] | None = None) -> Path:
    """Launch the native macOS PKStrokeRecognizer handwriting test panel.

    The panel captures manual mouse/trackpad strokes, renders the exact
    ``PKDrawing`` that is submitted to Apple, and displays ``recognizedText``
    plus ``indexableContent``.  It is intentionally separate from OCR so the
    user can verify the system recognizer before testing generated skeletons.
    """
    if platform.system() != "Darwin":
        raise RuntimeError("Apple 手写测试面板只能在 macOS 上运行")
    status = bridge_status(auto_build=auto_build)
    if not status.available:
        raise RuntimeError(status.detail)
    if not _APP_BUNDLE.exists():
        if auto_build:
            build_bridge()
        if not _APP_BUNDLE.exists():
            raise RuntimeError(f"缺少 Apple 手写测试 App：{_APP_BUNDLE}")
    try:
        args = ["/usr/bin/open", "-n", str(_APP_BUNDLE), "--args", "--manual-panel"]
        selected_payload = Path(payload_path) if payload_path else None
        if selected_payload is not None and selected_payload.exists():
            args.extend(["--payload", str(selected_payload.resolve())])
        env = os.environ.copy()
        env["NOVEL_FORMATTER_ROOT"] = str(_PROJECT_ROOT)
        subprocess.Popen(
            args, cwd=str(_NATIVE_ROOT), env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as exc:
        raise RuntimeError(f"无法打开 Apple 手写测试面板：{exc}") from exc
    return _APP_BUNDLE


def latest_auto_payload_path() -> Path | None:
    """Return the newest automatic PKStroke payload available for inspection."""
    if _LATEST_PAYLOAD.exists():
        return _LATEST_PAYLOAD
    if not _DEBUG_ROOT.exists():
        return None
    candidates = sorted(
        _DEBUG_ROOT.glob("failure-*.payload.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


class ApplePKStrokeRecognizer:
    def __init__(self, *, auto_build: bool = True, binary_path: str | os.PathLike[str] | None = None):
        self.binary_path = Path(binary_path) if binary_path else _BINARY
        self.last_result: dict[str, Any] = {}
        self.last_debug_path: str = ""
        self._server_proc: subprocess.Popen | None = None
        self._server_stderr_file = None
        self._server_lock = threading.Lock()
        self._server_disabled = False
        if binary_path is None:
            status = bridge_status(auto_build=auto_build)
            if not status.available:
                raise RuntimeError(status.detail)
            self.status = status
        else:
            if not self.binary_path.exists():
                raise RuntimeError(f"Apple PKStrokeRecognizer 桥接不存在：{self.binary_path}")
            self.status = AppleStrokeStatus(
                True, "使用指定的 Apple PKStrokeRecognizer 桥接",
                binary_path=str(self.binary_path), japanese_supported=True,
            )

    @property
    def persistent_server_supported(self) -> bool:
        """Whether the compiled bridge supports the JSONL persistent mode.

        Protocol 11 keeps one initialized AppKit process alive while still
        creating a fresh PKStrokeRecognizer and one single-glyph PKDrawing for
        every request.  Recognition behaviour therefore stays identical; only
        repeated process/AppKit startup is removed.
        """
        return bool(
            not self._server_disabled
            and int(getattr(self.status, "protocol_version", 0) or 0) >= 11
        )

    def _close_server_unlocked(self) -> None:
        from adapters.subprocess_watchdog import terminate_process

        proc = self._server_proc
        stderr_file = self._server_stderr_file
        self._server_proc = None
        self._server_stderr_file = None
        if proc is not None:
            try:
                if proc.poll() is None and proc.stdin is not None:
                    proc.stdin.write('{"command":"close"}\n')
                    proc.stdin.flush()
                    proc.wait(timeout=2)
            except Exception:
                terminate_process(proc)
            for stream in (proc.stdin, proc.stdout):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass
        try:
            if stderr_file is not None:
                stderr_file.close()
        except Exception:
            pass

    def close(self) -> None:
        with self._server_lock:
            self._close_server_unlocked()

    def _ensure_server_unlocked(self) -> subprocess.Popen:
        proc = self._server_proc
        if proc is not None and proc.poll() is None:
            return proc
        self._close_server_unlocked()
        from adapters.subprocess_watchdog import isolated_process_kwargs

        # The native bridge may write diagnostics for every glyph.  A PIPE that
        # is never drained eventually fills and deadlocks the recognizer.  A
        # temporary seekable file preserves diagnostics without backpressure.
        stderr_file = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
        proc = subprocess.Popen(
            [str(self.binary_path), "--server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            text=True,
            bufsize=1,
            **isolated_process_kwargs(),
        )
        self._server_stderr_file = stderr_file
        self._server_proc = proc
        return proc

    def _recognize_payload_persistent(
        self, payload: dict[str, Any], *, timeout: float,
    ) -> dict[str, Any]:
        with self._server_lock:
            proc = self._ensure_server_unlocked()
            if proc.stdin is None or proc.stdout is None:
                raise RuntimeError("Apple PKStrokeRecognizer 常驻桥接管道不可用")
            proc.stdin.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            proc.stdin.flush()

            selector = selectors.DefaultSelector()
            try:
                selector.register(proc.stdout, selectors.EVENT_READ)
                ready = selector.select(max(0.1, float(timeout)))
            finally:
                selector.close()
            if not ready:
                self._close_server_unlocked()
                raise TimeoutError("Apple PKStrokeRecognizer 常驻桥接响应超时")
            line = proc.stdout.readline()
            if not line:
                error = ""
                try:
                    stderr_file = self._server_stderr_file
                    if stderr_file is not None:
                        stderr_file.flush()
                        stderr_file.seek(0)
                        error = stderr_file.read()[-4000:].strip()
                except Exception:
                    pass
                self._close_server_unlocked()
                raise RuntimeError(
                    "Apple PKStrokeRecognizer 常驻桥接提前退出"
                    + (f"：{error}" if error else "")
                )
            return _parse_bridge_json(line)

    @staticmethod
    def _svg_from_payload(payload: dict[str, Any]) -> str:
        width = max(1.0, float(payload.get("canvasWidth") or 842.0))
        height = max(1.0, float(payload.get("canvasHeight") or 595.0))
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:g}" height="{height:g}" viewBox="0 0 {width:g} {height:g}">',
            '<rect width="100%" height="100%" fill="white"/>',
        ]
        for stroke in payload.get("strokes", []):
            points = stroke.get("points", [])
            if len(points) < 2:
                continue
            coords = " ".join(f'{float(p.get("x", 0)):g},{float(p.get("y", 0)):g}' for p in points)
            glyph = stroke.get("glyphIndex", "")
            parts.append(
                f'<polyline data-glyph="{glyph}" points="{coords}" fill="none" stroke="black" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>'
            )
        parts.append('</svg>')
        return "\n".join(parts)

    @staticmethod
    def _write_latest_payload(payload: dict[str, Any]) -> str:
        try:
            _DEBUG_ROOT.mkdir(parents=True, exist_ok=True)
            _LATEST_PAYLOAD.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return str(_LATEST_PAYLOAD)
        except Exception:
            return ""

    def _write_debug_bundle(self, payload: dict[str, Any], result: dict[str, Any] | None, error: str) -> str:
        try:
            root = _DEBUG_ROOT
            root.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            base = root / f"failure-{stamp}"
            (base.with_suffix(".payload.json")).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (base.with_suffix(".result.json")).write_text(
                json.dumps({"error": error, "result": result or {}}, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (base.with_suffix(".svg")).write_text(self._svg_from_payload(payload), encoding="utf-8")
            # Keep the directory bounded.
            files = sorted(root.glob("failure-*"), key=lambda item: item.stat().st_mtime, reverse=True)
            for old_file in files[60:]:
                try:
                    old_file.unlink()
                except Exception:
                    pass
            self.last_debug_path = str(base.with_suffix(".svg"))
            return self.last_debug_path
        except Exception:
            return ""

    def recognize_payload(
        self, payload: dict[str, Any], *, timeout: float = 120.0,
        save_latest_payload: bool = True,
    ) -> dict[str, Any]:
        if not payload.get("strokes"):
            return {"ok": False, "text": "", "error": "没有可提交的笔画"}
        if save_latest_payload:
            self._write_latest_payload(payload)
        result: dict[str, Any]
        if self.persistent_server_supported:
            try:
                result = self._recognize_payload_persistent(payload, timeout=timeout)
            except Exception:
                # A stale/unrebuilt helper must never break OCR. Disable the
                # optimization for this recognizer instance and use the exact
                # historical one-shot path for this and all later requests.
                self._server_disabled = True
                self.close()
                result = {}
            if result:
                self.last_result = dict(result)
            else:
                result = self._recognize_payload_oneshot(payload, timeout=timeout)
        else:
            result = self._recognize_payload_oneshot(payload, timeout=timeout)

        self.last_result = dict(result)
        result["text"] = str(result.get("text") or result.get("perGlyphText") or result.get("indexableContent") or "").strip()
        if not bool(result.get("ok", False)) or not result["text"]:
            message = str(result.get("error") or "Apple PKStrokeRecognizer 未返回文字")
            debug = self._write_debug_bundle(payload, result, message)
            diagnostics = []
            if result.get("recognizerLanguages"):
                diagnostics.append("识别器语言=" + ",".join(map(str, result["recognizerLanguages"])))
            if result.get("strokeCount") is not None:
                diagnostics.append(f"笔画={result.get('strokeCount')}")
            if result.get("pointCount") is not None:
                diagnostics.append(f"采样点={result.get('pointCount')}")
            if result.get("drawingUpdateCount") is not None:
                diagnostics.append(f"逐点更新={result.get('drawingUpdateCount')}")
            if result.get("playbackMode"):
                diagnostics.append(f"写入模式={result.get('playbackMode')}")
            suffix = ("；" + "，".join(diagnostics)) if diagnostics else ""
            if debug:
                suffix += f"；调试图：{debug}"
            raise RuntimeError(message + suffix)
        return result

    def _recognize_payload_oneshot(
        self, payload: dict[str, Any], *, timeout: float,
    ) -> dict[str, Any]:
        try:
            proc = subprocess.run(
                [str(self.binary_path)],
                input=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                capture_output=True, text=True, timeout=timeout, check=False,
            )
        except Exception as exc:
            debug = self._write_debug_bundle(payload, None, str(exc))
            raise RuntimeError(f"Apple PKStrokeRecognizer 桥接执行失败：{exc}" + (f"；调试图：{debug}" if debug else "")) from exc
        if proc.returncode != 0:
            message = (proc.stderr or proc.stdout).strip()
            debug = self._write_debug_bundle(payload, None, message)
            raise RuntimeError(
                f"Apple PKStrokeRecognizer 桥接执行失败：{message}" + (f"；调试图：{debug}" if debug else "")
            )
        return _parse_bridge_json(proc.stdout)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def recognize_strokes(
        self,
        strokes: Sequence[Sequence[tuple[float, float]]],
        *,
        canvas_width: float,
        canvas_height: float,
        preferred_languages: Sequence[str] = ("ja-JP",),
        glyph_indices: Sequence[int] | None = None,
        glyph_count: int | None = None,
        single_glyph_only: bool = False,
        debug_images: dict[str, str] | None = None,
        playback_mode: str | None = None,
        timeout: float = 120.0,
        save_latest_payload: bool = True,
    ) -> str:
        serialized: list[dict[str, Any]] = []
        for stroke_index, stroke in enumerate(strokes):
            if len(stroke) < 2:
                continue
            points = []
            total_length = 0.0
            previous: tuple[float, float] | None = None
            for point_index, (x, y) in enumerate(stroke):
                current = (float(x), float(y))
                if previous is not None:
                    total_length += ((current[0] - previous[0]) ** 2 + (current[1] - previous[1]) ** 2) ** 0.5
                previous = current
                # Use distance-derived time, approximating a stable handwriting
                # speed rather than making short and long strokes last equally.
                points.append({
                    "x": round(current[0], 4),
                    "y": round(current[1], 4),
                    "time": round(total_length / 180.0, 4),
                    "width": 3.2,
                    "opacity": 1.0,
                    "force": 0.55,
                })
            item: dict[str, Any] = {"id": f"stroke-{stroke_index + 1}", "points": points}
            if glyph_indices is not None and stroke_index < len(glyph_indices):
                item["glyphIndex"] = int(glyph_indices[stroke_index])
            serialized.append(item)
        result = self.recognize_payload({
            "preferredLanguages": list(preferred_languages),
            "canvasWidth": float(canvas_width),
            "canvasHeight": float(canvas_height),
            "glyphCount": int(glyph_count) if glyph_count is not None else None,
            "singleGlyphOnly": bool(single_glyph_only),
            # Match the working native manual panel exactly: one glyph only,
            # each pen-down point array becomes one stable PKStroke, and the
            # complete single-character PKDrawing is submitted once. A fresh
            # bridge invocation/recognizer is used for the next glyph.
            "playbackMode": str(playback_mode or (
                "single_glyph_manual_equivalent" if single_glyph_only
                else "manual_equivalent"
            )),
            "pointInterval": 0.010,
            "strokeGap": 0.065,
            "updateEveryPoints": 3,
            "debugImages": dict(debug_images or {}),
            "strokes": serialized,
        }, timeout=float(timeout), save_latest_payload=bool(save_latest_payload))
        return str(result.get("text") or "").strip()
