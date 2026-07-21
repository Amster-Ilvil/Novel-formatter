#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google Gemini API 校正器
"""

from __future__ import annotations

from .base import AIProvider


class GeminiProvider(AIProvider):

    @property
    def name(self) -> str:
        return "gemini"

    def __init__(self, api_key: str, model: str = "", **kwargs):
        super().__init__(api_key, model or "gemini-2.0-flash", **kwargs)

    def _call_llm(self, prompt: str, temperature: float) -> str:
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("请安装 google-generativeai: pip install google-generativeai")

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model)

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=temperature),
        )
        return response.text or ""
