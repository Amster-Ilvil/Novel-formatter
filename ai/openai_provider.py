#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI API 校正器
支持自定义 base_url（兼容 API 代理）。
"""

from __future__ import annotations

from .base import AIProvider


class OpenAIProvider(AIProvider):

    @property
    def name(self) -> str:
        return "openai"

    def __init__(self, api_key: str, model: str = "", **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.model = model or "gpt-4o"
        self.base_url = kwargs.get("base_url", None)

    def _call_llm(self, prompt: str, temperature: float) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")

        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        client = OpenAI(**client_kwargs)

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        msg = response.choices[0].message
        content = getattr(msg, "content", None)
        if not content:
            content = getattr(msg, "reasoning_content", None)
        return content or ""
