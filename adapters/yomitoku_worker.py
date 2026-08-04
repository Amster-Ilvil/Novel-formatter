#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistent YomiToku text-detector + recognizer JSONL worker.

Only YomiToku's OCR modules are loaded. Novel Formatter keeps ownership of
physical column segmentation, page reading order, headings and cross-page text
repair.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def _read_image(path: str) -> np.ndarray:
    # cv2.imread can fail on some non-ASCII paths. imdecode is path-safe.
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取图片：{path}")
    return image


def _resolve_device(requested: str) -> str:
    value = str(requested or "auto").strip().lower()
    if value not in {"auto", "mps", "cpu", "cuda"}:
        value = "auto"
    if value == "auto":
        if torch.cuda.is_available():
            return "cuda"
        try:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                return "mps"
        except Exception:
            pass
        return "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        return "cpu"
    if value == "mps":
        try:
            if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
                return "cpu"
        except Exception:
            return "cpu"
    return value


def _to_points(value) -> list[list[float]]:
    array = np.asarray(value, dtype=np.float32).reshape(-1, 2)
    return [[float(x), float(y)] for x, y in array]


def _rect(points: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _sort_blocks(blocks: list[dict], image_shape: tuple[int, int, int]) -> list[dict]:
    if not blocks:
        return []
    image_h, image_w = image_shape[:2]
    vertical_votes = sum(
        1 for block in blocks if str(block.get("direction") or "") == "vertical"
    )
    vertical = image_h > image_w * 1.15 or vertical_votes > len(blocks) / 2

    def key(block: dict):
        x1, y1, x2, y2 = _rect(block["box"])
        if vertical:
            # Japanese vertical reading: columns right-to-left, then top-to-bottom.
            return (-(x1 + x2) / 2.0, y1, x1)
        return (y1, x1, y2)

    return sorted(blocks, key=key)


def _weighted_confidence(blocks: list[dict]) -> float:
    total = sum(max(1, len(str(block.get("text") or ""))) for block in blocks)
    if total <= 0:
        return 0.0
    return sum(
        float(block.get("confidence", 0.0) or 0.0)
        * max(1, len(str(block.get("text") or "")))
        for block in blocks
    ) / total


def _clear_accelerator_cache() -> None:
    gc.collect()
    try:
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class YomiTokuRecognizer:
    def __init__(
        self,
        *,
        mode: str,
        requested_device: str,
        detector_onnx: bool,
        large_review: bool,
        review_threshold: float,
    ):
        from yomitoku.text_detector import TextDetector
        from yomitoku.text_recognizer import TextRecognizer

        self.TextDetector = TextDetector
        self.TextRecognizer = TextRecognizer
        self.package_version = importlib.metadata.version("yomitoku")
        self.mode = "accurate" if mode == "accurate" else "fast"
        self.device = _resolve_device(requested_device)
        self.detector_onnx_requested = bool(detector_onnx)
        self.large_review = bool(large_review) and self.mode == "fast"
        self.review_threshold = max(0.0, min(1.0, float(review_threshold)))
        self.detector_backend = ""
        self.device_fallback_reason = ""

        try:
            self.detector = self._construct_detector(self.device)
        except Exception as exc:
            if self.device == "cpu":
                raise
            previous = self.device
            self.device = "cpu"
            self.device_fallback_reason = (
                f"DBNet 在 {previous} 初始化失败，已回退 CPU：{exc}"
            )
            self.detector = self._construct_detector("cpu")

        self.recognizer_model = (
            "parseq-large-v4_1"
            if self.mode == "accurate"
            else "parseq-tiny-dynw-v4"
        )
        try:
            self.recognizer = self._construct_recognizer(
                self.recognizer_model, self.device
            )
        except Exception as exc:
            if self.device == "cpu":
                raise
            self._move_primary_to_cpu(
                f"{self.device} 识别模型加载失败，已回退 CPU：{exc}"
            )
        self.large_recognizer = None
        self.large_recognizer_device = None

    def _construct_detector(self, device: str):
        detector_class = getattr(self, "TextDetector", None)
        if detector_class is None:
            from yomitoku.text_detector import TextDetector as detector_class
            self.TextDetector = detector_class
        use_onnx = self.detector_onnx_requested and device == "cpu"
        if use_onnx:
            try:
                detector = detector_class(
                    model_name="dbnetv2_1",
                    device="cpu",
                    visualize=False,
                    infer_onnx=True,
                )
                self.detector_backend = "onnxruntime"
                return detector
            except Exception as exc:
                self.device_fallback_reason = (
                    self.device_fallback_reason + "; "
                    if self.device_fallback_reason
                    else ""
                ) + f"DBNet ONNX 初始化失败，改用 PyTorch CPU：{exc}"
        detector = detector_class(
            model_name="dbnetv2_1",
            device=device,
            visualize=False,
            infer_onnx=False,
        )
        self.detector_backend = "pytorch" if device != "cpu" else "pytorch-cpu"
        return detector

    def _construct_recognizer(self, model_name: str, device: str):
        recognizer_class = getattr(self, "TextRecognizer", None)
        if recognizer_class is None:
            from yomitoku.text_recognizer import TextRecognizer as recognizer_class
            self.TextRecognizer = recognizer_class
        tiny = model_name == "parseq-tiny-dynw-v4"
        parallel_batches = (
            2 if tiny and device == "cpu" and (os.cpu_count() or 1) >= 4 else 1
        )
        return recognizer_class(
            model_name=model_name,
            device=device,
            visualize=False,
            infer_onnx=False,
            dynamic_width=True if tiny else False,
            batch_bucketing=True if tiny else False,
            num_parallel_batches=parallel_batches,
            rec_orientation_fallback=True,
            rec_orientation_fallback_thresh=0.76,
            source_downscale=tiny,
        )

    def _move_primary_to_cpu(self, reason: str) -> None:
        self.device_fallback_reason = reason
        self.device = "cpu"
        _clear_accelerator_cache()
        self.detector = self._construct_detector("cpu")
        self.recognizer = self._construct_recognizer(
            self.recognizer_model, "cpu"
        )
        self.large_recognizer = None
        self.large_recognizer_device = None

    def _ensure_large(self):
        if self.large_recognizer is None:
            try:
                self.large_recognizer = self._construct_recognizer(
                    "parseq-large-v4_1", self.device
                )
                self.large_recognizer_device = self.device
            except Exception as exc:
                if self.device == "cpu":
                    raise
                _clear_accelerator_cache()
                self.large_recognizer = self._construct_recognizer(
                    "parseq-large-v4_1", "cpu"
                )
                self.large_recognizer_device = "cpu"
                self.device_fallback_reason = (
                    self.device_fallback_reason + "; "
                    if self.device_fallback_reason
                    else ""
                ) + f"large 复核模型在 {self.device} 加载失败，复核改用 CPU：{exc}"
        return self.large_recognizer

    def _run_large_with_fallback(self, image, points):
        large = self._ensure_large()
        try:
            return self._run_recognizer(large, image, points)
        except Exception as exc:
            if self.large_recognizer_device == "cpu":
                raise
            previous = self.large_recognizer_device or self.device
            _clear_accelerator_cache()
            self.large_recognizer = self._construct_recognizer(
                "parseq-large-v4_1", "cpu"
            )
            self.large_recognizer_device = "cpu"
            self.device_fallback_reason = (
                self.device_fallback_reason + "; "
                if self.device_fallback_reason
                else ""
            ) + f"large 复核模型在 {previous} 推理失败，复核改用 CPU：{exc}"
            return self._run_recognizer(self.large_recognizer, image, points)

    @staticmethod
    def _run_recognizer(recognizer, image, points):
        outputs, _ = recognizer(image, points, vis=None)
        return (
            [str(value or "") for value in outputs.contents],
            [float(value or 0.0) for value in outputs.scores],
            [str(value or "") for value in outputs.directions],
        )

    def _detect_with_device_fallback(self, image):
        try:
            return self.detector(image)
        except Exception as exc:
            if self.device != "cpu":
                previous = self.device
                self._move_primary_to_cpu(
                    f"DBNet 在 {previous} 推理失败，整套 YomiToku 已回退 CPU：{exc}"
                )
                return self.detector(image)
            if self.detector_backend == "onnxruntime":
                self.device_fallback_reason = (
                    self.device_fallback_reason + "; "
                    if self.device_fallback_reason
                    else ""
                ) + f"DBNet ONNX 推理失败，改用 PyTorch CPU：{exc}"
                self.detector_onnx_requested = False
                self.detector = self._construct_detector("cpu")
                self.detector_backend = "pytorch-cpu-fallback"
                return self.detector(image)
            raise

    # Compatibility name kept for older plugin/tests that called the earlier
    # backend-only fallback helper directly.
    def _detect_with_backend_fallback(self, image):
        return self._detect_with_device_fallback(image)

    def _recognize_with_device_fallback(self, image, points):
        try:
            return self._run_recognizer(self.recognizer, image, points)
        except Exception as exc:
            if self.device == "cpu":
                raise
            previous = self.device
            self._move_primary_to_cpu(
                f"PARSeq 在 {previous} 推理失败，整套 YomiToku 已回退 CPU：{exc}"
            )
            return self._run_recognizer(self.recognizer, image, points)

    def recognize_image(self, path: str, *, already_isolated: bool = False) -> list[dict]:
        image = _read_image(path)
        detector_fallback = bool(already_isolated)
        if already_isolated:
            # The caller has already isolated one complete physical column and
            # removed neighbouring text/Ruby.  Recognize the exact supplied
            # glyph envelope as one polygon instead of invoking DBNet again.
            h, w = image.shape[:2]
            points = [[[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]]
            det_scores = [1.0]
        else:
            det_outputs, _ = self._detect_with_device_fallback(image)
            points = list(det_outputs.points) if det_outputs.points is not None else []
            raw_det_scores = (
                list(det_outputs.scores) if det_outputs.scores is not None else []
            )
            det_scores = [float(value or 0.0) for value in raw_det_scores]
        if not points:
            # DBNet occasionally misses an already-isolated narrow column. The
            # recognizer can still consume the exact crop as one polygon.
            h, w = image.shape[:2]
            points = [[[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]]
            det_scores = [0.0]
            detector_fallback = True

        texts, scores, directions = self._recognize_with_device_fallback(
            image, points
        )
        models = [self.recognizer_model] * len(texts)
        reviewed = [False] * len(texts)

        if self.large_review:
            retry_indices = [
                index
                for index, (text, score) in enumerate(zip(texts, scores))
                if not str(text).strip() or float(score) < self.review_threshold
            ]
            if retry_indices:
                retry_points = [points[index] for index in retry_indices]
                large_texts, large_scores, large_directions = (
                    self._run_large_with_fallback(image, retry_points)
                )
                for local_index, original_index in enumerate(retry_indices):
                    if local_index >= len(large_texts):
                        continue
                    candidate = str(large_texts[local_index] or "")
                    candidate_score = float(
                        large_scores[local_index]
                        if local_index < len(large_scores)
                        else 0.0
                    )
                    current = str(texts[original_index] or "")
                    current_score = float(scores[original_index] or 0.0)
                    if candidate.strip() and (
                        not current.strip()
                        or candidate_score >= current_score + 0.015
                    ):
                        texts[original_index] = candidate
                        scores[original_index] = candidate_score
                        if local_index < len(large_directions):
                            directions[original_index] = large_directions[local_index]
                        models[original_index] = "parseq-large-v4_1"
                        reviewed[original_index] = True

        raw_blocks: list[dict] = []
        for index, point in enumerate(points):
            text = str(texts[index] if index < len(texts) else "").strip()
            if not text:
                continue
            box = _to_points(point)
            raw_blocks.append(
                {
                    "text": text,
                    "confidence": float(
                        scores[index] if index < len(scores) else 0.0
                    ),
                    "det_score": float(
                        det_scores[index] if index < len(det_scores) else 0.0
                    ),
                    "direction": str(
                        directions[index] if index < len(directions) else ""
                    ),
                    "box": box,
                    "model": models[index]
                    if index < len(models)
                    else self.recognizer_model,
                    "large_reviewed": bool(
                        reviewed[index] if index < len(reviewed) else False
                    ),
                }
            )

        raw_blocks = _sort_blocks(raw_blocks, image.shape)
        if not raw_blocks:
            return []
        text = "".join(block["text"] for block in raw_blocks).strip()
        confidence = _weighted_confidence(raw_blocks)
        all_points = [point for block in raw_blocks for point in block["box"]]
        x1, y1, x2, y2 = _rect(all_points)
        return [
            {
                "text": text,
                "confidence": confidence,
                "box": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "segments": raw_blocks,
                "detector": "dbnetv2_1",
                "recognizer": self.recognizer_model,
                "detector_backend": self.detector_backend,
                "detector_fallback": detector_fallback,
                "large_review_count": sum(
                    1 for block in raw_blocks if block["large_reviewed"]
                ),
            }
        ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true", required=True)
    parser.add_argument("--mode", choices=("fast", "accurate"), default="fast")
    parser.add_argument(
        "--device", choices=("auto", "mps", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--detector-onnx", action="store_true")
    parser.add_argument("--large-review", action="store_true")
    parser.add_argument("--review-threshold", type=float, default=0.82)
    args = parser.parse_args()

    try:
        try:
            torch.set_num_threads(min(8, max(1, os.cpu_count() or 4)))
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        recognizer = YomiTokuRecognizer(
            mode=args.mode,
            requested_device=args.device,
            detector_onnx=args.detector_onnx,
            large_review=args.large_review,
            review_threshold=args.review_threshold,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "ready": False,
                    "error": f"YomiToku 初始化失败：{exc}",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        raise SystemExit(1)

    print(
        json.dumps(
            {
                "ok": True,
                "ready": True,
                "version": recognizer.package_version,
                "device": recognizer.device,
                "detector": "dbnetv2_1",
                "detector_backend": recognizer.detector_backend,
                "recognizer_model": recognizer.recognizer_model,
                "dynamic_width": recognizer.mode == "fast",
                "batch_bucketing": recognizer.mode == "fast",
                "large_review": recognizer.large_review,
                "fallback": recognizer.device_fallback_reason,
                "platform": f"{platform.system()} {platform.machine()}",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for line in sys.stdin:
        request: dict = {}
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if request.get("command") == "close":
                print(
                    json.dumps({"ok": True, "closed": True}, ensure_ascii=False),
                    flush=True,
                )
                return
            request_id = int(request.get("request_id", 0) or 0)
            already_isolated = bool(request.get("already_isolated", False))
            images = [
                str(path) for path in request.get("images", []) if str(path)
            ]
            if not images:
                raise ValueError("缺少 images")
            for path in images:
                try:
                    blocks = recognizer.recognize_image(
                        path, already_isolated=already_isolated
                    )
                    print(
                        json.dumps(
                            {
                                "ok": True,
                                "request_id": request_id,
                                "path": path,
                                "blocks": blocks,
                                "device": recognizer.device,
                                "detector_backend": recognizer.detector_backend,
                                "fallback": recognizer.device_fallback_reason,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                except Exception as exc:
                    print(
                        json.dumps(
                            {
                                "ok": False,
                                "request_id": request_id,
                                "path": path,
                                "error": str(exc),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
            print(
                json.dumps(
                    {
                        "ok": True,
                        "request_id": request_id,
                        "batch_done": True,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "request_id": int(request.get("request_id", 0) or 0)
                        if isinstance(request, dict)
                        else 0,
                        "error": str(exc),
                        "batch_done": True,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
