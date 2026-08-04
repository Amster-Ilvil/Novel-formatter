#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google Gemini API 校正器
"""

from __future__ import annotations

from .base import AIProvider
import threading


class GeminiProvider(AIProvider):

    @property
    def name(self) -> str:
        return "gemini"

    def __init__(self, api_key: str, model: str = "", **kwargs):
        super().__init__(api_key, model or "gemini-2.0-flash", **kwargs)
        self._model = None
        self._model_lock = threading.Lock()

    def _get_model(self):
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    try:
                        import google.generativeai as genai
                    except ImportError:
                        raise ImportError("请安装 google-generativeai: pip install google-generativeai")
                    genai.configure(api_key=self.api_key)
                    self._model = genai.GenerativeModel(self.model)
        return self._model

    def _request(self, prompt: str, temperature: float, json_mode: bool = False) -> str:
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("请安装 google-generativeai: pip install google-generativeai")

        config_kwargs = {
            "temperature": temperature,
            "max_output_tokens": int(self.kwargs.get("max_tokens", 32000)),
        }
        if self.kwargs.get("top_p") is not None:
            config_kwargs["top_p"] = float(self.kwargs.get("top_p"))
        if json_mode and self.kwargs.get("json_mode", True):
            config_kwargs["response_mime_type"] = "application/json"
        response = self._get_model().generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(**config_kwargs),
        )
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
        completion_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
        self._record_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=getattr(usage, "total_token_count", 0) if usage else 0,
            cached_tokens=getattr(usage, "cached_content_token_count", 0) if usage else 0,
            reasoning_tokens=getattr(usage, "thoughts_token_count", 0) if usage else 0,
        )
        return response.text or ""

    def _call_llm(self, prompt: str, temperature: float) -> str:
        return self._request(prompt, temperature, False)

    def call_json(self, prompt: str, temperature: float) -> str:
        try:
            return self._request(prompt, temperature, True)
        except Exception as exc:
            message = str(exc).lower()
            if any(marker in message for marker in ("response_mime_type", "mime type", "unsupported", "invalid argument")):
                return self._request(prompt, temperature, False)
            raise
