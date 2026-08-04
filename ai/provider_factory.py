# -*- coding: utf-8 -*-
from __future__ import annotations
from .config import AISettings, validate_api_key
from .openai_provider import OpenAIProvider
from .deepseek_provider import DeepSeekProvider
from .gemini_provider import GeminiProvider
from .anthropic_provider import AnthropicProvider


def create_provider(settings: AISettings):
    name = settings.provider.lower().strip()
    kwargs = settings.provider_kwargs()
    api_key = settings.api_key
    if settings.requires_key:
        api_key = validate_api_key(api_key)
    if name == "openai":
        return OpenAIProvider(api_key, settings.model, **kwargs)
    if name == "deepseek":
        return DeepSeekProvider(api_key, settings.model, **kwargs)
    if name == "gemini":
        return GeminiProvider(api_key, settings.model, **kwargs)
    if name == "anthropic":
        return AnthropicProvider(api_key, settings.model, **kwargs)
    if name == "openrouter":
        kwargs.setdefault("base_url", "https://openrouter.ai/api/v1")
        return OpenAIProvider(api_key, settings.model, **kwargs)
    if name == "ollama":
        kwargs.setdefault("base_url", "http://127.0.0.1:11434/v1")
        return OpenAIProvider(api_key or "ollama", settings.model, **kwargs)
    if name == "custom":
        if not settings.base_url:
            raise ValueError("自定义 Provider 必须填写 Base URL")
        return OpenAIProvider(api_key or "not-required", settings.model, **kwargs)
    raise ValueError(f"不支持的 AI Provider: {settings.provider}")


def test_provider(settings: AISettings) -> str:
    with create_provider(settings) as provider:
        reply = provider._call_llm("Reply with exactly: OK", 0.0).strip()
        if not reply:
            raise RuntimeError("服务已响应，但返回内容为空")
        return reply[:200]
