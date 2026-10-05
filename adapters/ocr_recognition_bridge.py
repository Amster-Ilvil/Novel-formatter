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
    "apple_vision": "Apple OCR",
    "macocr": "Apple OCR",
    "mac_ocr": "Apple OCR",
    "macos_ocr": "Apple OCR",
    "hayai_ocr": "Hayai OCR",
    "manga_48px": "48px AR OCR",
    "ndlocr_lite": "NDLOCR-Lite",
    "paddle_ocr": "PaddleOCR",
    "google_vision": "Google Vision API",
}


def _normalise_engine_id(engine: str) -> str:
    return {
        "macocr": "apple_vision",
        "mac_ocr": "apple_vision",
        "macos_ocr": "apple_vision",
    }.get(str(engine or "").strip().lower(), str(engine or "").strip())


def _simple_result_blocks(text: str, confidence: float, *, heuristic: bool = False) -> list[dict]:
    if not str(text or "").strip():
        return []
    block = {
        "text": str(text),
        "confidence": float(confidence or 0.0),
        "box": None,
    }
    if heuristic:
        block["confidence_kind"] = "heuristic"
    return [block]


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
        blocks = [{"text": result.full_text.strip(), "confidence": 0.0}]
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

    def iter_recognize(self, image_paths: list[str], *, on_wait=None):
        if self.backend is None or self.config is None:
            raise RuntimeError("Apple Vision 常驻会话尚未启动")
        total = len(image_paths)
        for completed, image_path in enumerate(image_paths):
            if self.cancel_check is not None and self.cancel_check():
                break
            if callable(on_wait):
                try:
                    on_wait(completed, total)
                except Exception:
                    pass
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


class ReusableRecognitionSession:
    """Keep one OCR runtime alive across staged retry batches when possible.

    Targeted disagreement retries are intentionally staged: cheap canonical
    views run first, and horizontal reflow is only sent for rows that remain
    unresolved.  Calling :func:`recognizer_iterator` separately for those
    stages would reload Hayai/48px/NDLOCR models.  This small
    bridge exposes the persistent session objects those adapters already own,
    while falling back to the ordinary iterator for engines without a reusable
    runtime.

    Apple Vision is orientation-configured per request.  We therefore keep at
    most one helper alive and restart it only when a batch switches between
    vertical/standard and horizontal input, rather than keeping two helpers in
    memory simultaneously.
    """

    def __init__(
        self,
        engine: str,
        manifest_path: str,
        *,
        shortcut_name: str = "ExtractText",
        cancel_check=None,
        verbose: bool = True,
        engine_options: dict | None = None,
    ):
        self.engine = _normalise_engine_id(engine)
        self.manifest_path = str(manifest_path)
        self.shortcut_name = shortcut_name
        self.cancel_check = cancel_check
        self.verbose = verbose
        self.options = dict(engine_options or {})
        self._session = None
        self._session_kind = ""
        self._apple_vertical: bool | None = None

    def __enter__(self):
        engine = self.engine
        if engine == "hayai_ocr":
            from adapters.hayai_ocr_adapter import HayaiOcrSession
            self._session = HayaiOcrSession(
                engine_options=self.options,
                cancel_check=self.cancel_check,
                verbose=self.verbose,
            ).__enter__()
            self._session_kind = "hayai"
        elif engine == "manga_48px":
            from adapters.manga_48px_adapter import Manga48pxSession
            self._session = Manga48pxSession(
                cancel_check=self.cancel_check,
                verbose=self.verbose,
                engine_options=self.options,
            ).__enter__()
            self._session_kind = "manga_48px"
        elif engine == "manga_ocr":
            from adapters.manga_ocr_adapter import MangaOcrSession
            self._session = MangaOcrSession(
                cancel_check=self.cancel_check,
                verbose=self.verbose,
            ).__enter__()
            self._session_kind = "manga_ocr"
        elif engine == "ndlocr_lite":
            from adapters.ndlocr_lite_adapter import NDLOcrLiteSession
            self._session = NDLOcrLiteSession(
                cancel_check=self.cancel_check,
                verbose=self.verbose,
                engine_options=self.options,
            ).__enter__()
            self._session_kind = "ndlocr_lite"
        # Apple is opened lazily because the first stage determines vertical
        # orientation. Other engines use the existing one-shot bridge.
        return self

    def _close_current(self, exc_type=None, exc=None, tb=None) -> None:
        session = self._session
        self._session = None
        self._session_kind = ""
        self._apple_vertical = None
        if session is None:
            return
        exit_fn = getattr(session, "__exit__", None)
        if callable(exit_fn):
            exit_fn(exc_type, exc, tb)
            return
        close_fn = getattr(session, "close", None)
        if callable(close_fn):
            close_fn()

    def _ensure_apple(self, options: dict):
        vertical = bool(options.get("vertical", True))
        if self._session is not None and self._session_kind == "apple" and self._apple_vertical == vertical:
            return self._session
        self._close_current()
        session = AppleVisionRecognitionSession(
            shortcut_name=self.shortcut_name,
            engine_options=options,
            cancel_check=self.cancel_check,
        )
        self._session = session.__enter__()
        self._session_kind = "apple"
        self._apple_vertical = vertical
        return self._session

    def iter_recognize(
        self,
        image_paths: list[str],
        *,
        engine_options: dict | None = None,
        input_metadata: dict[str, dict] | None = None,
    ) -> Iterator[tuple[str, list[dict] | None, str | None]]:
        paths = [str(path) for path in image_paths]
        if not paths:
            return
        options = dict(self.options)
        options.update(dict(engine_options or {}))
        metadata = {str(k): dict(v or {}) for k, v in (input_metadata or {}).items()}

        if self.engine == "apple_vision":
            session = self._ensure_apple(options)
            yield from session.iter_recognize(paths)
            return

        if self._session_kind in {"hayai", "manga_48px", "manga_ocr"} and self._session is not None:
            results = self._session.recognize(paths, input_metadata=metadata)
            for path in paths:
                text, confidence, error = results.get(path, ("", 0.0, "识字进程未返回该区域"))
                if error:
                    yield path, None, str(error)
                else:
                    yield path, _simple_result_blocks(
                        text, confidence, heuristic=self._session_kind in {"hayai", "manga_ocr"}
                    ), None
            return

        if self._session_kind == "ndlocr_lite" and self._session is not None:
            yield from self._session.iter_recognize(paths)
            return

        # Engines without a reusable local session (for example Paddle/PDF
        # Craft/cloud providers) still participate in every retry stage.  They
        # use the existing bridge and may pay their own startup cost.
        yield from recognizer_iterator(
            self.engine,
            paths,
            self.manifest_path,
            shortcut_name=self.shortcut_name,
            cancel_check=self.cancel_check,
            verbose=self.verbose,
            engine_options=options,
            input_metadata=metadata,
        )

    def close(self, exc_type=None, exc=None, tb=None) -> None:
        self._close_current(exc_type, exc, tb)

    def __exit__(self, exc_type, exc, tb):
        self.close(exc_type, exc, tb)
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
    input_metadata: dict[str, dict] | None = None,
) -> Iterator[tuple[str, list[dict] | None, str | None]]:
    """Recognize prepared images with exactly one selected engine."""
    engine = _normalise_engine_id(engine)
    options = dict(engine_options or {})
    if engine == "apple_vision":
        with AppleVisionRecognitionSession(
            shortcut_name=shortcut_name, engine_options=options, cancel_check=cancel_check
        ) as session:
            yield from session.iter_recognize(image_paths)
        return
    if engine == "hayai_ocr":
        from adapters.hayai_ocr_adapter import recognize_crops
        yield from recognize_crops(
            image_paths, manifest_path, cancel_check=cancel_check, verbose=verbose,
            engine_options=options, input_metadata=input_metadata,
        )
        return
    if engine == "manga_48px":
        from adapters.manga_48px_adapter import recognize_crops
        yield from recognize_crops(
            image_paths, manifest_path, cancel_check=cancel_check, verbose=verbose,
            engine_options=options, input_metadata=input_metadata,
        )
        return
    if engine == "manga_ocr":
        from adapters.manga_ocr_adapter import recognize_crops
        yield from recognize_crops(
            image_paths, manifest_path, cancel_check=cancel_check, verbose=verbose,
            input_metadata=input_metadata,
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
