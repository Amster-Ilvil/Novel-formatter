#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pinned PaddleOCR model/runtime identifiers and download-source helpers."""
from __future__ import annotations

import os

PADDLE_OCR_VERSION = "PP-OCRv6"
PADDLE_DETECTION_MODEL = "PP-OCRv6_medium_det"
PADDLE_RECOGNITION_MODEL = "PP-OCRv6_medium_rec"
PADDLEOCR_MIN_VERSION = (3, 7, 0)
PADDLEOCR_PACKAGE_SPEC = "paddleocr>=3.7,<4"
PADDLE_RUNTIME_SIGNATURE = "paddleocr>=3.7:ppocr_v6_medium_or_default_japan:multi_source_v2"

# PaddleX/PaddleOCR 3.x accepts these values through PADDLE_PDX_MODEL_SOURCE.
# Keep the identifiers lower-case because upstream normalises them that way.
PADDLE_MODEL_SOURCES = ("huggingface", "modelscope", "bos", "aistudio")
PADDLE_MODEL_SOURCE_LABELS = {
    "auto": "自动重试",
    "huggingface": "Hugging Face",
    "modelscope": "ModelScope",
    "bos": "百度 BOS",
    "aistudio": "AIStudio",
}


def normalize_paddle_model_source(value: str | None) -> str:
    source = str(value or "auto").strip().lower()
    aliases = {
        "hf": "huggingface",
        "hugging_face": "huggingface",
        "model_scope": "modelscope",
        "baidu": "bos",
        "baidu_bos": "bos",
        "ai_studio": "aistudio",
        "default": "auto",
        "": "auto",
    }
    source = aliases.get(source, source)
    return source if source in PADDLE_MODEL_SOURCES else "auto"


def paddle_model_source_attempts(preferred: str | None = "auto") -> tuple[str, ...]:
    """Return a deterministic, duplicate-free download-source retry order.

    An explicit source is respected exactly.  Auto mode first honours an
    existing PADDLE_PDX_MODEL_SOURCE, then tries the official default
    HuggingFace and the three other sources supported by PaddleX.  Model
    initialisation happens before any page is processed, so retrying a failed
    source cannot duplicate OCR output.
    """
    normalized = normalize_paddle_model_source(preferred)
    if normalized != "auto":
        return (normalized,)

    ordered: list[str] = []
    env_source = normalize_paddle_model_source(os.environ.get("PADDLE_PDX_MODEL_SOURCE"))
    if env_source != "auto":
        ordered.append(env_source)
    # Official default first; regional mirrors follow automatically.
    ordered.extend(("huggingface", "modelscope", "bos", "aistudio"))
    return tuple(dict.fromkeys(ordered))


def paddle_source_environment(source: str | None, base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base or os.environ)
    normalized = normalize_paddle_model_source(source)
    if normalized == "auto":
        env.pop("PADDLE_PDX_MODEL_SOURCE", None)
    else:
        env["PADDLE_PDX_MODEL_SOURCE"] = normalized
    # Avoid noisy telemetry/progress bars corrupting the JSONL stdout protocol.
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    return env


def paddleocr_version_marker() -> str:
    """Return a self-contained import/version probe for the isolated venv."""
    minimum = ",".join(str(part) for part in PADDLEOCR_MIN_VERSION)
    return (
        "import paddle, paddleocr; "
        "from importlib.metadata import version; "
        "v=version('paddleocr').split('+',1)[0].split('.'); "
        "n=tuple(int(''.join(c for c in p if c.isdigit()) or 0) for p in v[:3]); "
        f"assert n>=({minimum}), 'PaddleOCR 3.7+ required, got '+version('paddleocr')"
    )
