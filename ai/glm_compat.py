from __future__ import annotations

"""Compatibility helpers for GLM OpenAI-compatible routes.

The domestic BigModel and international Z.AI routes are nominally OpenAI-
compatible, but GLM-5.3 Flash has a few deployment-specific quirks:

* some routes reject ``thinking=disabled`` and require a reasoning level such
  as ``low``/``high``/``max`` instead;
* server-side error code ``1210`` is overloaded across both image-parse and
  thinking-policy failures.

These helpers keep the matching rules explicit and reusable across both the
text provider and the multimodal image-book client.
"""

from typing import Any


GLM_PROVIDERS = {"zhipu", "zai"}


def is_glm53_flash(provider: str | None, model: str | None) -> bool:
    return str(provider or "").strip().lower() in GLM_PROVIDERS and str(model or "").strip().lower().startswith("glm-5.3-flash")


def _lower_text(exc: Any) -> str:
    return str(exc).strip().lower()


def looks_always_thinking_error(exc: Any) -> bool:
    lowered = _lower_text(exc)
    direct_markers = (
        "不支持关闭思考",
        "始终思考",
        "always thinking",
        "does not support disabling thinking",
        "thinking.type",
        "reasoning level",
        "reasoning effort",
    )
    if any(marker in lowered for marker in direct_markers):
        return True
    # Some gateways only say that the value must be low/high/max. Pair that
    # with the service's overloaded 1210 or an explicit thinking keyword.
    level_markers = (
        "low/high/max",
        "low、high 或 max",
        "low、high、max",
        "low, high or max",
        "low, high, or max",
        "只支持 low",
        "must be low",
    )
    return ("1210" in lowered or "thinking" in lowered or "思考" in lowered) and any(
        marker in lowered for marker in level_markers
    )


def looks_image_payload_error(exc: Any) -> bool:
    lowered = _lower_text(exc)
    image_markers = (
        "image parse",
        "image parsing",
        "failed to parse image",
        "invalid image",
        "image decode",
        "decode image",
        "image payload",
        "image input",
        "图片解析",
        "图片输入格式",
        "图片格式",
        "无法解析图片",
        "解析图片",
        "unsupported image format",
    )
    # Error code 1210 alone is not enough because GLM also reuses it for the
    # always-thinking policy violation.
    return any(marker in lowered for marker in image_markers)
