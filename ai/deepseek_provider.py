#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek API 校正器
通过 OpenAI SDK 连接 DeepSeek API。
"""

from __future__ import annotations

from .openai_provider import OpenAIProvider


class DeepSeekProvider(OpenAIProvider):

    @property
    def name(self) -> str:
        return "deepseek"

    def __init__(self, api_key: str, model: str = "", **kwargs):
        kwargs.setdefault("base_url", "https://api.deepseek.com/v1")
        super().__init__(api_key, model or "deepseek-chat", **kwargs)
