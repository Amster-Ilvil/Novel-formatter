#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Provider 插件系统
工厂函数自动选择对应的 Provider。
"""

from .base import AIProvider, Suggestion
from .openai_provider import OpenAIProvider
from .deepseek_provider import DeepSeekProvider
from .gemini_provider import GeminiProvider
from .diff import apply_suggestions

PROVIDER_REGISTRY: dict[str, type[AIProvider]] = {
    "openai": OpenAIProvider,
    "deepseek": DeepSeekProvider,
    "gemini": GeminiProvider,
}


def get_provider(name: str, api_key: str, model: str = "", **kwargs) -> AIProvider:
    """
    获取 AI Provider 实例。

    Args:
        name: provider 名称（openai / deepseek / gemini）
        api_key: API Key
        model: 模型名（留空使用默认）
        **kwargs: 额外参数（如 base_url）

    Returns:
        AIProvider 实例
    """
    cls = PROVIDER_REGISTRY.get(name.lower())
    if cls is None:
        available = ", ".join(PROVIDER_REGISTRY.keys())
        raise ValueError(f"未知 AI Provider: {name}。可用: {available}")
    return cls(api_key=api_key, model=model, **kwargs)
