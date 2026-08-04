# -*- coding: utf-8 -*-
"""Retrieve model IDs for the AI settings model selector.

The implementation intentionally uses the Python standard library so the GUI
can populate the model list without introducing another runtime dependency.
API keys are sent only in request headers/query parameters required by the
selected provider and are never persisted or logged here.
"""
from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from .config import provider_defaults, validate_api_key


class ModelCatalogError(RuntimeError):
    """Raised when a provider model list cannot be retrieved or parsed."""


def _append_path(base_url: str, suffix: str) -> str:
    base = str(base_url or "").strip().rstrip("/")
    suffix = "/" + suffix.lstrip("/")
    if base.endswith(suffix):
        return base
    return base + suffix


def _ollama_tags_url(base_url: str) -> str:
    raw = str(base_url or "http://127.0.0.1:11434/v1").strip()
    parts = urlsplit(raw)
    path = parts.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    if path.endswith("/api"):
        path += "/tags"
    else:
        path += "/api/tags"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _request_json(url: str, headers: dict[str, str], timeout: int) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "NovelFormatter/1.0 model-catalog",
            **headers,
        },
        method="GET",
    )
    try:
        with urlopen(request, timeout=max(5, int(timeout))) as response:
            raw = response.read()
    except HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            detail = ""
        suffix = f"：{detail}" if detail else ""
        raise ModelCatalogError(f"模型列表请求失败（HTTP {exc.code}）{suffix}") from exc
    except URLError as exc:
        raise ModelCatalogError(f"无法连接模型列表接口：{exc.reason}") from exc
    except TimeoutError as exc:
        raise ModelCatalogError("读取模型列表超时") from exc

    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ModelCatalogError("模型列表接口返回的不是有效 JSON") from exc
    if not isinstance(payload, dict):
        raise ModelCatalogError("模型列表接口返回格式不正确")
    return payload


def _normalise_ids(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return sorted(result, key=lambda value: value.lower())


def _generic_models(payload: dict[str, Any]) -> list[str]:
    data = payload.get("data", [])
    if not isinstance(data, list):
        return []
    values: list[str] = []
    for item in data:
        if isinstance(item, str):
            values.append(item)
        elif isinstance(item, dict):
            values.append(str(item.get("id") or item.get("name") or ""))
    return _normalise_ids(values)


def _gemini_models(payload: dict[str, Any]) -> list[str]:
    data = payload.get("models", [])
    if not isinstance(data, list):
        return []
    values: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        methods = item.get("supportedGenerationMethods")
        if isinstance(methods, list) and methods and "generateContent" not in methods:
            continue
        name = str(item.get("name") or "")
        if name.startswith("models/"):
            name = name[len("models/"):]
        values.append(name)
    return _normalise_ids(values)


def _ollama_models(payload: dict[str, Any]) -> list[str]:
    data = payload.get("models", [])
    if not isinstance(data, list):
        return []
    values: list[str] = []
    for item in data:
        if isinstance(item, dict):
            values.append(str(item.get("name") or item.get("model") or ""))
        elif isinstance(item, str):
            values.append(item)
    return _normalise_ids(values)


def fetch_available_models(
    provider: str,
    api_key: str = "",
    base_url: str = "",
    timeout: int = 30,
) -> list[str]:
    """Return model IDs exposed by the selected provider.

    Supported providers mirror :mod:`ai.config`: OpenAI, Anthropic, Gemini,
    DeepSeek, OpenRouter, Ollama, and arbitrary OpenAI-compatible endpoints.
    """
    provider = str(provider or "").strip().lower()
    api_key = str(api_key or "").strip()
    if provider != "ollama":
        api_key = validate_api_key(api_key)
    default_model, default_url = provider_defaults(provider)
    base_url = str(base_url or default_url or "").strip()

    if provider != "ollama" and not api_key:
        raise ModelCatalogError("请先输入 API Key")

    if provider == "gemini":
        root = base_url or "https://generativelanguage.googleapis.com/v1beta"
        url = _append_path(root, "models")
        separator = "&" if "?" in url else "?"
        url += separator + urlencode({"key": api_key})
        models = _gemini_models(_request_json(url, {}, timeout))
    elif provider == "anthropic":
        root = base_url or "https://api.anthropic.com/v1"
        url = _append_path(root, "models")
        models = _generic_models(_request_json(url, {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }, timeout))
    elif provider == "ollama":
        models = _ollama_models(_request_json(_ollama_tags_url(base_url), {}, timeout))
    else:
        if not base_url:
            raise ModelCatalogError("请先填写 Base URL")
        url = _append_path(base_url, "models")
        models = _generic_models(_request_json(url, {
            "Authorization": f"Bearer {api_key}",
        }, timeout))

    if not models:
        provider_name = provider or "当前服务"
        fallback = f"；可继续手动填写模型名称（默认 {default_model}）" if default_model else "；可继续手动填写模型名称"
        raise ModelCatalogError(f"{provider_name} 未返回可选择的模型{fallback}")
    return models
