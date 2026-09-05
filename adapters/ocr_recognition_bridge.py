#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Direct recognition bridge used by masked-column OCR.

This module intentionally contains no layout-composition pipeline. It accepts
already prepared images and sends them directly to one selected OCR engine.
"""
from __future__ import annotations

import os
from typing import Iterator

RECOGNITION_ENGINES = {
    "apple_vision": "macOS OCR（Apple Vision）",
    "macocr": "macOS OCR（Apple Vision）",
    "mac_ocr": "macOS OCR（Apple Vision）",
    "macos_ocr": "macOS OCR（Apple Vision）",
    "manga_ocr": "Manga OCR",
    "hayai_ocr": "Hayai OCR v2.1",
    "manga_48px": "48px AR OCR",
    "yomitoku": "YomiToku OCR",
    "ndlocr_lite": "NDLOCR-Lite",
    "paddle_ocr": "PaddleOCR",
    "pdf_craft": "PDF Craft",
    "google_vision": "Google Vision API",
}


def _apple_config_and_backend(shortcut_name: str, options: dict):
    from adapters.vision_backends import BackendFactory, OCRConfig
    # Apple OCR is explicit.  ``auto`` is a legacy saved-setting alias handled
    # by BackendFactory and maps to the explicit Live Text route.
    backend_id = str(options.get("apple_backend") or options.get("backend") or "live_text")
    backend = BackendFactory.create(backend_id)
    available, reason = backend.is_available()
    if not available:
        raise RuntimeError(f"Apple Vision {backend_id} 不可用：{reason}")
    raw_languages = options.get("recognition_languages") or options.get("languages") or ["ja-JP"]
    if isinstance(raw_languages, str):
        languages = [part.strip() for part in raw_languages.replace(";", ",").split(",") if part.strip()]
    else:
        languages = [str(part).strip() for part in raw_languages if str(part).strip()]
    config = OCRConfig(
        shortcut_name=shortcut_name,
        vertical=bool(options.get("vertical", True)),
        recognition_level=str(options.get("recognition_level") or "accurate"),
        languages=languages or ["ja-JP"],
        use_language_correction=bool(options.get("use_language_correction", True)),
        automatically_detect_language=bool(options.get("automatically_detect_language", False)),
        minimum_text_height_fraction=float(options.get("minimum_text_height_fraction", 0.005)),
        candidate_count=int(options.get("candidate_count", 3)),
        orientation=str(options.get("orientation") or "auto"),
        vertical_preprocess=str(options.get("vertical_preprocess") or "none"),
        timeout=max(10.0, min(300.0, float(options.get("request_timeout", 90.0) or 90.0))),
    )
    return backend, config


def _apple_blocks(result) -> list[dict]:
    blocks = [
        {
            "text": item.text,
            "confidence": float(item.confidence or 0.0),
            "bbox": item.bbox,
            "language": item.language,
            "candidates": [
                {"text": text, "confidence": confidence}
                for text, confidence in item.candidates
            ],
        }
        for item in result.blocks if str(item.text or "").strip()
    ]
    if not blocks and str(result.full_text or "").strip():
        blocks = [{"text": result.full_text.strip(), "confidence": 1.0}]
    return blocks


class AppleVisionRecognitionSession:
    """Keep one Vision backend/helper alive for primary and sentence OCR."""

    def __init__(self, *, shortcut_name: str = "ExtractText", engine_options: dict | None = None,
                 cancel_check=None):
        self.shortcut_name = shortcut_name
        self.options = dict(engine_options or {})
        self.cancel_check = cancel_check
        self.backend = None
        self.config = None

    def __enter__(self):
        self.backend, self.config = _apple_config_and_backend(self.shortcut_name, self.options)
        # Propagate the GUI stop event into persistent Swift helpers and the
        # automatic Live Text/Shortcut wrapper.  The helper poll loop checks it
        # every 250 ms and kills the complete helper process group.
        for candidate in (
            self.backend,
            getattr(self.backend, "_live_text", None),
            getattr(self.backend, "_native", None),
            getattr(self.backend, "_shortcut", None),
        ):
            if candidate is not None:
                try:
                    setattr(candidate, "cancel_check", self.cancel_check)
                except Exception:
                    pass
        return self

    def iter_recognize(self, image_paths: list[str]):
        if self.backend is None or self.config is None:
            raise RuntimeError("Apple Vision 常驻会话尚未启动")
        for image_path in image_paths:
            if self.cancel_check is not None and self.cancel_check():
                break
            try:
                result = self.backend.recognize(image_path, self.config)
                yield image_path, _apple_blocks(result), None
            except Exception as exc:
                yield image_path, None, str(exc)

    def close(self):
        if self.backend is not None:
            try:
                self.backend.close()
            finally:
                self.backend = None
                self.config = None

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def recognizer_iterator(
    engine: str,
    image_paths: list[str],
    manifest_path: str,
    *,
    shortcut_name: str = "ExtractText",
    cancel_check=None,
    verbose: bool = True,
    engine_options: dict | None = None,
) -> Iterator[tuple[str, list[dict] | None, str | None]]:
    """Recognize prepared images with exactly one selected engine."""
    engine = {
        "macocr": "apple_vision",
        "mac_ocr": "apple_vision",
        "macos_ocr": "apple_vision",
    }.get(str(engine or "").strip().lower(), str(engine or "").strip())
    options = dict(engine_options or {})
    if engine == "apple_vision":
        with AppleVisionRecognitionSession(
            shortcut_name=shortcut_name, engine_options=options, cancel_check=cancel_check
        ) as session:
            yield from session.iter_recognize(image_paths)
        return
    if engine == "manga_ocr":
        from adapters.manga_ocr_adapter import recognize_crops
        yield from recognize_crops(image_paths, manifest_path, cancel_check=cancel_check, verbose=verbose)
        return
    if engine == "hayai_ocr":
        from adapters.hayai_ocr_adapter import recognize_crops
        yield from recognize_crops(
            image_paths, manifest_path, cancel_check=cancel_check, verbose=verbose, engine_options=options
        )
        return
    if engine == "manga_48px":
        from adapters.manga_48px_adapter import recognize_crops
        yield from recognize_crops(image_paths, manifest_path, cancel_check=cancel_check, verbose=verbose)
        return
    if engine == "yomitoku":
        from adapters.yomitoku_adapter import recognize_crops
        yield from recognize_crops(
            image_paths,
            manifest_path,
            cancel_check=cancel_check,
            verbose=verbose,
            mode=str(options.get("mode") or "fast"),
            device=str(options.get("device") or "auto"),
            detector_onnx=bool(options.get("detector_onnx", True)),
            large_review=bool(options.get("large_review", True)),
            review_threshold=float(options.get("review_threshold", 0.82) or 0.82),
        )
        return
    if engine == "ndlocr_lite":
        from adapters.ndlocr_lite_adapter import _run_worker
        yield from _run_worker(image_paths, cancel_check=cancel_check, verbose=verbose)
        return
    if engine == "paddle_ocr":
        from adapters.paddle_ocr_adapter import setup_venv, _run_worker
        pipeline = str(options.get("pipeline") or "ocr")
        if pipeline not in {"ocr", "structure", "vl"}:
            pipeline = "ocr"
        setup_venv(verbose=verbose, pipeline=pipeline)
        yield from _run_worker(
            image_paths,
            lang=str(options.get("lang") or "japan"),
            pipeline=pipeline,
            cancel_check=cancel_check,
            model_source=str(options.get("model_source") or "auto"),
            vl_backend=str(options.get("vl_backend") or "auto"),
        )
        return
    if engine == "pdf_craft":
        from adapters.pdf_craft_adapter import setup_venv, WORKER_SCRIPT, MODEL_CACHE
        from adapters.ocr_engine_common import iter_worker_jsonl
        from adapters.ocr_runtime_catalog import runtime_ready
        prepare_models = not runtime_ready("pdf_craft")
        python = setup_venv(verbose=verbose)
        MODEL_CACHE.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(python), str(WORKER_SCRIPT),
            "--model-cache", str(MODEL_CACHE),
            "--ocr-size", str(options.get("ocr_size") or "base"),
            *(["--prepare-models"] if prepare_models else []),
            *image_paths,
        ]
        yield from iter_worker_jsonl(cmd, cancel_check=cancel_check, engine_label="PDF Craft")
        return
    if engine == "google_vision":
        from adapters.google_vision_adapter import _annotate_image, DEFAULT_ENDPOINT
        api_key = str(options.get("api_key") or os.environ.get("GOOGLE_CLOUD_VISION_API_KEY", "")).strip()
        if not api_key:
            raise ValueError("请填写 Google Cloud Vision API Key，或设置 GOOGLE_CLOUD_VISION_API_KEY。")
        raw_hints = options.get("language_hints") or ""
        hints = ([part.strip() for part in raw_hints.replace(";", ",").split(",") if part.strip()]
                 if isinstance(raw_hints, str)
                 else [str(part).strip() for part in raw_hints if str(part).strip()])
        endpoint = str(options.get("endpoint") or DEFAULT_ENDPOINT)
        for image_path in image_paths:
            if cancel_check is not None and cancel_check():
                break
            try:
                yield image_path, _annotate_image(
                    image_path, api_key=api_key, language_hints=hints, endpoint=endpoint
                ), None
            except Exception as exc:
                yield image_path, None, str(exc)
        return
    raise ValueError(f"不支持的识字引擎: {engine}")
