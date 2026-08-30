#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Manga-OCR runtime and persistent recognition session."""
from __future__ import annotations

import json
import os
import subprocess
import threading
import tempfile
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from PIL import Image, ImageOps

from adapters.runtime_env import ensure_venv

ROOT = Path(__file__).parent.parent
VENV_DIR = ROOT / ".venv-manga-ocr"
WORKER_SCRIPT = Path(__file__).parent / "manga_ocr_worker.py"
MANGA_OCR_PACKAGE = os.environ.get("NOVEL_FORMATTER_MANGA_OCR_PACKAGE", "manga-ocr==0.1.16")
MANGA_TRANSFORMERS_PACKAGE = os.environ.get(
    "NOVEL_FORMATTER_MANGA_TRANSFORMERS_PACKAGE", "transformers==4.55.0"
)
MANGA_HF_HUB_PACKAGE = os.environ.get(
    "NOVEL_FORMATTER_MANGA_HF_HUB_PACKAGE", "huggingface-hub==0.34.4"
)
MODEL_CACHE = ROOT / ".model-cache" / "manga-ocr"



@dataclass(frozen=True, slots=True)
class MangaOcrSegment:
    path: str
    expected_chars: int
    column_index: int
    segment_index: int


def _compact_text(text: str) -> str:
    return "".join(str(text or "").split())


def _japanese_ratio(text: str) -> float:
    value = _compact_text(text)
    if not value:
        return 0.0
    count = 0
    for ch in value:
        code = ord(ch)
        if (
            0x3040 <= code <= 0x30FF
            or 0x31F0 <= code <= 0x31FF
            or 0x3400 <= code <= 0x4DBF
            or 0x4E00 <= code <= 0x9FFF
            or 0xF900 <= code <= 0xFAFF
            or ch in "、。！？…―ー・「」『』（）［］【】〈〉《》〜～＝0123456789０１２３４５６７８９"
        ):
            count += 1
    return count / max(1, len(value))


def _ink_bbox(image: Image.Image, threshold: int = 242) -> tuple[int, int, int, int] | None:
    gray = ImageOps.grayscale(image)
    try:
        mask = gray.point(lambda value: 255 if value < threshold else 0, mode="1")
        return mask.getbbox()
    finally:
        gray.close()


def _horizontal_projection(image: Image.Image, threshold: int = 242) -> list[int]:
    gray = ImageOps.grayscale(image)
    try:
        width, height = gray.size
        pixels = gray.load()
        return [sum(1 for x in range(width) if pixels[x, y] < threshold) for y in range(height)]
    finally:
        gray.close()


def _choose_split(projection: list[int], start: int, ideal: int, end: int, guard: int) -> int:
    low = max(start + guard, ideal - max(guard, (ideal - start) // 3))
    high = min(end - guard, ideal + max(guard, (ideal - start) // 3))
    if high <= low:
        return min(end, max(start + guard, ideal))
    return min(range(low, high + 1), key=lambda y: (projection[y], abs(y - ideal)))


def _split_column_image(
    image: Image.Image,
    *,
    output_dir: Path,
    stem: str,
    column_index: int,
    estimated_chars: int = 0,
    max_aspect: float = 7.2,
    max_chars: int = 12,
) -> list[MangaOcrSegment]:
    """Split one long printed vertical column into Manga-OCR-sized chunks.

    The upstream ViT image processor resizes each complete input.  A 40–50
    character light-novel column therefore makes every glyph far smaller than
    the manga model saw during training.  Keep the original vertical direction
    and cut only between glyph rows at low-ink horizontal valleys.
    """
    bbox = _ink_bbox(image)
    if bbox is None:
        return []
    left, top, right, bottom = bbox
    ink_width = max(1, right - left)
    margin_x = max(8, round(ink_width * 0.18))
    margin_y = max(7, round(ink_width * 0.24))
    crop_left = max(0, left - margin_x)
    crop_right = min(image.width, right + margin_x)
    base = image.crop((crop_left, top, crop_right, bottom)).convert("RGB")
    try:
        projection = _horizontal_projection(base)
        total_height = base.height
        import math
        count_by_aspect = max(1, math.ceil(total_height / max(96.0, base.width * max_aspect)))
        count_by_chars = max(1, math.ceil(max(1, int(estimated_chars or 0)) / max_chars)) if estimated_chars else 1
        segment_count = max(count_by_aspect, count_by_chars)
        target_height = max(72, round(total_height / segment_count))
        min_height = max(54, round(max(1, base.width) * 1.65))
        ranges: list[tuple[int, int]] = []
        cursor = 0
        while len(ranges) + 1 < segment_count and total_height - cursor > min_height * 1.2:
            ideal = min(total_height, cursor + target_height)
            split = _choose_split(
                projection, cursor, ideal, total_height,
                guard=max(10, round(base.width * 0.38)),
            )
            if split - cursor < min_height:
                split = min(total_height, cursor + target_height)
            ranges.append((cursor, split))
            cursor = split
        if cursor < total_height:
            if ranges and total_height - cursor < min_height * 0.55:
                ranges[-1] = (ranges[-1][0], total_height)
            else:
                ranges.append((cursor, total_height))
        if not ranges:
            ranges = [(0, total_height)]

        output: list[MangaOcrSegment] = []
        for segment_index, (seg_top, seg_bottom) in enumerate(ranges):
            if seg_bottom <= seg_top:
                continue
            region = base.crop((0, seg_top, base.width, seg_bottom)).convert("RGB")
            try:
                segment_bbox = _ink_bbox(region)
                if segment_bbox is None:
                    continue
                sl, st, sr, sb = segment_bbox
                clean = region.crop((
                    max(0, sl - margin_x), max(0, st - margin_y),
                    min(region.width, sr + margin_x), min(region.height, sb + margin_y),
                )).convert("RGB")
                try:
                    digest = hashlib.sha1(
                        f"{stem}:{column_index}:{segment_index}:{seg_top}:{seg_bottom}".encode("utf-8")
                    ).hexdigest()[:10]
                    path = output_dir / f"{stem}_c{column_index:03d}_s{segment_index:03d}_{digest}.png"
                    clean.save(path, format="PNG", compress_level=1)
                finally:
                    clean.close()
            finally:
                region.close()
            if estimated_chars > 0:
                expected = max(1, round(estimated_chars * (seg_bottom - seg_top) / max(1, total_height)))
            else:
                expected = max(1, round((seg_bottom - seg_top) / max(12.0, ink_width * 0.92)))
            output.append(MangaOcrSegment(str(path), expected, column_index, segment_index))
        return output
    finally:
        base.close()


def prepare_manga_ocr_segments(
    image_path: str,
    output_dir: Path,
    *,
    max_aspect: float = 7.2,
    max_chars: int = 12,
    already_isolated: bool = False,
    estimate_isolated_chars: bool = False,
) -> tuple[list[MangaOcrSegment], int]:
    """Create short, unrotated vertical chunks in Japanese reading order.

    ``already_isolated`` is the contract used by the 48px AR path.  The common
    column-OCR layer has already selected one authoritative physical column,
    removed neighbouring columns/Ruby, and added paper-colour margins.  Running
    the page-level column detector a second time on that narrow crop is unsafe:
    detached radicals and long punctuation can be mistaken for separate columns,
    causing a second isolation pass to keep only the widest stroke band.

    In isolated mode we therefore preserve the complete supplied glyph envelope
    and only split it along horizontal whitespace.  No page-layout detection or
    second Ruby filter is performed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    try:
        bbox = _ink_bbox(image)
        if bbox is None:
            return [], 0

        stem = Path(image_path).stem.replace(" ", "_")[:48] or "manga"
        if already_isolated:
            # Historical Manga OCR / 48px callers keep estimated_chars=0 exactly
            # as before.  Hayai can opt into a conservative ink-envelope estimate
            # so its larger max_chars target actually participates in splitting,
            # without changing any pre-Hayai recognizer's segmentation contract.
            estimated_chars = 0
            if estimate_isolated_chars:
                left, top, right, bottom = bbox
                ink_width = max(1, right - left)
                ink_height = max(1, bottom - top)
                estimated_chars = max(1, round(ink_height / max(12.0, ink_width * 0.92)))
            segments = _split_column_image(
                image,
                output_dir=output_dir,
                stem=stem,
                column_index=0,
                estimated_chars=estimated_chars,
                max_aspect=max_aspect,
                max_chars=max_chars,
            )
            return segments, 1 if segments else 0

        from adapters.column_ocr_adapter import detect_vertical_columns, _isolated_column_image
        detected = detect_vertical_columns(image, sensitivity=48, padding_percent=6, max_columns=48)

        # Direct Manga-OCR callers may still pass an unprepared page or region.
        # Keep the historical layout path for those calls.
        if image.width <= 420:
            main_band = max(
                detected,
                key=lambda item: (
                    int(getattr(item, "width", 0) or 0),
                    float(getattr(item, "ink_score", 0.0) or 0.0),
                ),
                default=None,
            )
            if main_band is not None:
                columns = [main_band]
            else:
                class _CompactColumn:
                    hard_left, top, hard_right, bottom = bbox
                    left, right = bbox[0], bbox[2]
                    width = max(1, bbox[2] - bbox[0])
                    full_height_slot = False
                    estimated_chars = max(1, round((bbox[3] - bbox[1]) / max(12.0, width * 0.72)))
                columns = [_CompactColumn()]
        elif detected:
            columns = detected
        else:
            class _FallbackColumn:
                hard_left, top, hard_right, bottom = bbox
                left, right = bbox[0], bbox[2]
                width = max(1, bbox[2] - bbox[0])
                full_height_slot = False
                estimated_chars = max(1, round((bbox[3] - bbox[1]) / max(12.0, width * 0.72)))
            columns = [_FallbackColumn()]

        segments: list[MangaOcrSegment] = []
        for column_index, column in enumerate(columns):
            compact = _isolated_column_image(image, column, retry=False, background=(255, 255, 255))
            try:
                segments.extend(_split_column_image(
                    compact,
                    output_dir=output_dir,
                    stem=stem,
                    column_index=column_index,
                    estimated_chars=int(getattr(column, "estimated_chars", 0) or 0),
                    max_aspect=max_aspect,
                    max_chars=max_chars,
                ))
            finally:
                compact.close()
        return segments, len(columns)
    finally:
        image.close()

def validate_manga_ocr_text(text: str, expected_chars: int) -> tuple[bool, float, str]:
    """Reject obvious generative runaway instead of publishing it as OCR."""
    value = _compact_text(text)
    if not value:
        return False, 0.0, "Manga OCR 返回空文本"
    if len(value) >= 280:
        return False, 0.0, "Manga OCR 输出触及 300 字生成上限，已判定为幻觉"
    if _japanese_ratio(value) < 0.55:
        return False, 0.0, "Manga OCR 输出的日文字符比例异常"
    expected = max(1, int(expected_chars or 1))
    ratio = len(value) / expected
    if len(value) > max(32, expected * 2.20 + 7):
        return False, 0.0, f"Manga OCR 输出字数异常（识别 {len(value)} / 字形估计 {expected}）"
    if ratio < 0.28:
        return False, 0.0, f"Manga OCR 严重缺字（识别 {len(value)} / 字形估计 {expected}）"
    confidence = 0.72 if 0.55 <= ratio <= 1.70 else 0.54
    return True, confidence, ""

def setup_venv(verbose: bool = True) -> Path:
    # manga-translator-ui pins Transformers 4.55 for this model family.  Older
    # manga-ocr packages combined with an unconstrained future Transformers
    # release can load an incompatible Japanese tokenizer and produce fluent but
    # unrelated text.  The marker repairs existing venvs as well as new ones.
    marker = (
        "from manga_ocr import MangaOcr; import transformers; "
        "assert transformers.__version__.startswith('4.55.'), transformers.__version__"
    )
    return ensure_venv(
        VENV_DIR,
        label="Manga OCR",
        marker_code=marker,
        packages=[
            MANGA_OCR_PACKAGE,
            MANGA_TRANSFORMERS_PACKAGE,
            MANGA_HF_HUB_PACKAGE,
        ],
        verbose=verbose,
    )


def _worker_env() -> dict[str, str]:
    env = os.environ.copy()
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)

    def has_weights(root: Path) -> bool:
        try:
            return any(path.is_file() and path.stat().st_size >= 500_000 for path in root.rglob("*"))
        except OSError:
            return False

    if "HF_HOME" not in env and "TRANSFORMERS_CACHE" not in env and has_weights(MODEL_CACHE):
        env["HF_HOME"] = str(MODEL_CACHE)
        env["TRANSFORMERS_CACHE"] = str(MODEL_CACHE / "transformers")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def _looks_like_full_page(path: str) -> bool:
    """Conservatively identify page-sized inputs for the crop-only API.

    Manga OCR is a recognizer, not a page-layout engine.  The normal ``run``
    entry point now performs physical-column segmentation first.  This helper is
    only a defence for plugins that accidentally call ``recognize_crops`` with
    complete book pages.
    """
    try:
        from PIL import Image
        with Image.open(path) as image:
            width, height = image.size
    except Exception:
        return False
    if width < 700 or height < 900:
        return False
    area = width * height
    aspect = width / max(1, height)
    return area >= 900_000 and 0.42 <= aspect <= 1.35


class MangaOcrSession:
    """One worker/model shared by any number of crop batches."""

    def __init__(self, *, cancel_check=None, verbose: bool = True):
        self.cancel_check = cancel_check
        self.verbose = verbose
        self.proc: subprocess.Popen[str] | None = None
        self._stderr_lines: list[str] = []
        self._stderr_thread: threading.Thread | None = None
        self._stderr_stop = threading.Event()
        self._stdout_pump = None
        self.device = ""
        self._request_id = 0

    def __enter__(self):
        python = setup_venv(verbose=self.verbose)
        cmd = [str(python), str(WORKER_SCRIPT), "--stream"]
        from adapters.subprocess_watchdog import LinePump, isolated_process_kwargs
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=_worker_env(),
            **isolated_process_kwargs(),
        )
        self._stdout_pump = LinePump(self.proc.stdout, name="manga-ocr-stdout")
        assert self.proc.stderr is not None
        stderr_pipe = self.proc.stderr
        self._stderr_stop.clear()
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(stderr_pipe,),
            daemon=True,
            name="manga-ocr-stderr",
        )
        self._stderr_thread.start()
        from adapters.subprocess_watchdog import env_seconds
        ready = self._read_response(
            timeout=env_seconds("NOVEL_FORMATTER_OCR_STARTUP_TIMEOUT", 900.0, minimum=60.0)
        )
        if not ready.get("ready"):
            self.close(force=True)
            raise RuntimeError(ready.get("error", "Manga OCR worker 未就绪"))
        self.device = str(ready.get("device", ""))
        return self

    def _drain_stderr(self, stderr_pipe) -> None:
        """Collect worker diagnostics without leaking shutdown tracebacks.

        ``subprocess`` pipes are closed by the owner thread during teardown.
        A daemon reader may still be blocked in ``readline()`` at that exact
        moment, especially while macOS/MPS is finalizing PyTorch.  Reading a
        pipe that has just been closed raises ``ValueError``; this is a normal
        shutdown race, not an OCR failure, so the reader must end quietly.
        """
        while not self._stderr_stop.is_set():
            try:
                line = stderr_pipe.readline()
            except (ValueError, OSError):
                break
            if not line:
                break
            self._stderr_lines.append(str(line).rstrip())
            if len(self._stderr_lines) > 200:
                del self._stderr_lines[:100]

    def _read_response(self, timeout: float | None = None) -> dict:
        if self.proc is None or self.proc.stdout is None:
            raise RuntimeError("Manga OCR worker 尚未启动")
        from adapters.subprocess_watchdog import LinePump, env_seconds
        if self._stdout_pump is None:
            self._stdout_pump = LinePump(self.proc.stdout, name="manga-ocr-stdout")
        wait_seconds = (
            float(timeout) if timeout is not None
            else env_seconds("NOVEL_FORMATTER_OCR_REQUEST_TIMEOUT", 300.0, minimum=30.0)
        )
        line = self._stdout_pump.readline(
            proc=self.proc,
            timeout=wait_seconds,
            cancel_check=self.cancel_check,
            label="Manga OCR",
        )
        if line is None:
            ret = self.proc.poll()
            tail = "\n".join(self._stderr_lines[-30:])
            raise RuntimeError(f"Manga OCR worker 提前退出 (code={ret})\n{tail}")
        try:
            return json.loads(line)
        except Exception as exc:
            raise RuntimeError(f"Manga OCR worker 返回无效 JSON: {line[:300]} ({exc})") from exc

    def recognize(
        self,
        crop_paths: list[str],
        *,
        progress_callback=None,
    ) -> dict[str, tuple[str, float, str | None]]:
        """Recognize many physical columns with bounded model batches.

        Segmentation and validation remain byte-for-byte compatible with the
        previous path.  Only transport changes: several prepared segments are
        sent in one JSON request and the worker performs one ViT/generation
        batch instead of thousands of single-image requests on long books.
        """
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("Manga OCR worker 尚未启动")
        results: dict[str, tuple[str, float, str | None]] = {}
        ordered_paths = [str(path) for path in crop_paths]
        total = max(1, len(ordered_paths))
        try:
            source_window = int(os.environ.get("NOVEL_FORMATTER_MANGA_OCR_SOURCE_WINDOW", "16") or 16)
        except ValueError:
            source_window = 16
        source_window = max(2, min(32, source_window))
        default_segment_batch = 4 if "mps" in str(self.device or "").lower() else 8
        try:
            segment_batch = int(
                os.environ.get(
                    "NOVEL_FORMATTER_MANGA_OCR_BATCH", str(default_segment_batch)
                ) or default_segment_batch
            )
        except ValueError:
            segment_batch = default_segment_batch
        segment_batch = max(1, min(16, segment_batch))

        with tempfile.TemporaryDirectory(prefix="novel_formatter_manga_chunks_") as temp_dir:
            chunk_root = Path(temp_dir)
            completed = 0
            for window_start in range(0, len(ordered_paths), source_window):
                if self.cancel_check is not None and self.cancel_check():
                    break
                window = ordered_paths[window_start:window_start + source_window]
                prepared: list[tuple[str, list, int, str]] = []
                flat_segments: list = []

                for local_index, source_path in enumerate(window, start=1):
                    global_index = window_start + local_index
                    try:
                        segments, column_count = prepare_manga_ocr_segments(
                            source_path,
                            chunk_root / f"i{global_index:05d}",
                            # The shared column layer already supplied one
                            # authoritative, Ruby-free physical column.  Manga
                            # OCR used to run the page detector again here,
                            # which could mistake detached radicals or long
                            # punctuation for separate columns and crop away
                            # part of the glyph.  Only split the long column
                            # horizontally into recognizer-sized chunks.
                            already_isolated=True,
                        )
                        error = "" if segments else "Manga OCR 输入区域没有检测到印刷文字"
                    except Exception as exc:
                        segments, column_count = [], 0
                        error = f"Manga OCR 输入分段失败: {exc}"
                    prepared.append((source_path, list(segments), int(column_count), error))
                    flat_segments.extend(segments)

                returned: dict[str, dict] = {}
                transport_error = ""
                for offset in range(0, len(flat_segments), segment_batch):
                    if self.cancel_check is not None and self.cancel_check():
                        transport_error = "用户取消"
                        break
                    batch = flat_segments[offset:offset + segment_batch]
                    self._request_id += 1
                    request_id = self._request_id
                    request = {
                        "request_id": request_id,
                        "paths": [segment.path for segment in batch],
                    }
                    try:
                        self.proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
                        self.proc.stdin.flush()
                        data = self._read_response()
                        response_id = int(data.get("request_id", -1) or -1)
                        if response_id != request_id:
                            raise RuntimeError(
                                "Manga OCR 请求响应串位："
                                f"request_id={request_id}/{response_id}"
                            )
                        if not data.get("ok"):
                            raise RuntimeError(str(data.get("error", "未知错误")))
                        items = data.get("items")
                        # Compatibility with older/third-party stream workers for
                        # a one-segment request.
                        if not isinstance(items, list) and len(batch) == 1 and data.get("path"):
                            items = [data]
                        if not isinstance(items, list):
                            raise RuntimeError("Manga OCR 批量响应缺少 items")
                        for item in items:
                            returned[str(item.get("path", ""))] = item
                    except Exception as exc:
                        transport_error = str(exc)
                        break

                for source_path, segments, column_count, prepare_error in prepared:
                    failure = prepare_error or transport_error
                    column_texts: dict[int, list[str]] = {}
                    confidences: list[float] = []
                    if not failure:
                        for segment in segments:
                            data = returned.get(segment.path)
                            if data is None:
                                failure = f"Manga OCR 未返回分段: {Path(segment.path).name}"
                                break
                            if not data.get("ok"):
                                failure = str(data.get("error", "未知错误"))
                                break
                            blocks = data.get("blocks") or []
                            text = "".join(
                                str(item.get("text", "")).strip()
                                for item in blocks
                                if str(item.get("text", "")).strip()
                            ).strip()
                            valid, confidence, reason = validate_manga_ocr_text(
                                text, segment.expected_chars
                            )
                            if not valid:
                                failure = reason
                                break
                            column_texts.setdefault(segment.column_index, []).append(
                                _compact_text(text)
                            )
                            confidences.append(confidence)

                    if failure:
                        results[source_path] = ("", 0.0, failure)
                    else:
                        ordered_columns = [
                            "".join(column_texts[index])
                            for index in sorted(column_texts)
                            if column_texts.get(index)
                        ]
                        text = ("\n" if column_count > 1 else "").join(ordered_columns).strip()
                        confidence = min(confidences) if confidences else 0.0
                        results[source_path] = (
                            text,
                            confidence,
                            None if text else "Manga OCR 未返回有效文字",
                        )
                    completed += 1
                    if callable(progress_callback):
                        progress_callback(completed, total, source_path)
        return results

    def close(self, *, force: bool = False) -> None:
        """Close the persistent worker without turning cleanup into OCR failure.

        On macOS/MPS the Python interpreter can spend a long time finalizing
        PyTorch and ``multiprocessing.resource_tracker`` objects after all OCR
        responses have already been returned.  The old implementation killed
        that lingering process after 15 seconds and then raised on ``-9``, so a
        successful book was reported as failed during ``__exit__``.

        A non-zero code is still fatal when the worker was already dead before
        shutdown began.  Codes caused by our own graceful/forced shutdown are
        cleanup outcomes and must not replace valid OCR results or an upstream
        exception.
        """
        proc = self.proc
        self.proc = None
        if proc is None:
            return

        initial_ret = proc.poll()
        shutdown_requested = False
        termination_requested = False
        ret = initial_ret

        try:
            if not force and initial_ret is None and proc.stdin is not None:
                proc.stdin.write(json.dumps({"command": "close"}) + "\n")
                proc.stdin.flush()
                shutdown_requested = True
        except Exception:
            # Broken stdin normally means the child has already exited.  Its
            # actual return code is checked below instead of hiding it here.
            pass
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass

        try:
            if force and proc.poll() is None:
                proc.terminate()
                termination_requested = True
            ret = proc.wait(timeout=12 if shutdown_requested else 5 if force else 1)
        except subprocess.TimeoutExpired:
            # The recognition work is complete at this point.  Stop a worker
            # stuck only in MPS/Python interpreter teardown, first politely and
            # then decisively, without treating our own signal as OCR failure.
            if proc.poll() is None:
                try:
                    proc.terminate()
                    termination_requested = True
                    ret = proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:
                        proc.kill()
                        termination_requested = True
                    except Exception:
                        pass
                    ret = proc.wait()
        except Exception:
            if proc.poll() is None:
                try:
                    proc.kill()
                    termination_requested = True
                except Exception:
                    pass
            ret = proc.wait()

        # Let the stderr reader consume EOF before closing its pipe.  If a
        # platform keeps the reader blocked, close the pipe only after the first
        # join attempt; ``_drain_stderr`` treats that ValueError/OSError as a
        # normal shutdown signal rather than printing a daemon-thread traceback.
        stderr_thread = self._stderr_thread
        self._stderr_stop.set()
        if stderr_thread is not None:
            stderr_thread.join(timeout=1.5)
        if stderr_thread is not None and stderr_thread.is_alive():
            try:
                if proc.stderr:
                    proc.stderr.close()
            except Exception:
                pass
            stderr_thread.join(timeout=1.5)
        else:
            try:
                if proc.stderr:
                    proc.stderr.close()
            except Exception:
                pass
        self._stderr_thread = None

        stdout_pump = self._stdout_pump
        self._stdout_pump = None
        if stdout_pump is not None:
            stdout_pump.close()

        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass

        intentional_shutdown = bool(force or shutdown_requested or termination_requested)
        if ret not in (0, -15) and not intentional_shutdown:
            tail = "\n".join(self._stderr_lines[-30:])
            raise RuntimeError(f"Manga OCR worker 异常退出 (code={ret}):\n{tail}")

    def __exit__(self, exc_type, exc, tb):
        try:
            self.close(force=exc is not None)
        except Exception:
            if exc is None:
                raise
        if exc is None:
            from adapters.ocr_runtime_catalog import mark_runtime_ready
            mark_runtime_ready("manga_ocr")
        return False


def recognize_crops(
    crop_paths: list[str],
    manifest_path: str,
    *,
    cancel_check=None,
    verbose: bool = True,
) -> Iterator[tuple[str, list[dict] | None, str | None]]:
    """Recognize already-isolated text crops with a persistent Manga OCR model.

    Complete pages are rejected here because Manga OCR has no reliable page
    layout or reading-order stage.  Call :func:`run` for pages; it performs the
    project's physical-column segmentation before recognition.
    """
    del manifest_path
    ordered = [str(path) for path in crop_paths]
    page_like = {path for path in ordered if _looks_like_full_page(path)}
    safe_paths = [path for path in ordered if path not in page_like]
    results: dict[str, tuple[str, float, str | None]] = {}
    if safe_paths:
        with MangaOcrSession(cancel_check=cancel_check, verbose=verbose) as session:
            results = session.recognize(safe_paths)
    for path in ordered:
        if path in page_like:
            yield path, None, (
                "Manga OCR 收到疑似整页图片，已拒绝直接识别。"
                "请使用 Manga OCR 页面入口，由程序先做物理分列后再逐列识别。"
            )
            continue
        text, confidence, error = results.get(path, ("", 0.0, "识字进程未返回该区域"))
        if error:
            yield path, None, error
        else:
            blocks = [{"text": text, "confidence": confidence, "box": None}] if text else []
            yield path, blocks, None


def run(*, verbose: bool = True, **kwargs):
    """Recognize pages safely by segmenting them into physical vertical columns.

    Manga OCR is never allowed to consume a complete light-novel page directly.
    The existing column adapter preserves the page geometry, reading order,
    placeholders and review metadata, then reuses one persistent Manga OCR model
    for all isolated columns and optional sentence crops.
    """
    from adapters.column_ocr_adapter import run as run_column_ocr

    recognition_engine = str(kwargs.pop("recognition_engine", "manga_ocr") or "manga_ocr")
    if recognition_engine != "manga_ocr":
        raise ValueError("Manga OCR 适配器只能使用 recognition_engine='manga_ocr'")
    return run_column_ocr(
        recognition_engine="manga_ocr",
        verbose=verbose,
        **kwargs,
    )
