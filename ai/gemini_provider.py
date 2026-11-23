#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Google Gemini text provider via official REST; no obsolete SDK dependency.

Text and vision transports now share the same HTTPS endpoint, timeout,
rate limiter and JSON-mode contract, with no credential persistence here.
"""
from __future__ import annotations

import threading
from .base import AIProvider
from .gemini_ocr_transport import GeminiJsonTransport, DEFAULT_MODEL


class GeminiProvider(AIProvider):
    @property
    def name(self) -> str:
        return "gemini"

    def __init__(self, api_key: str, model: str = "", **kwargs):
        super().__init__(api_key, model or DEFAULT_MODEL, **kwargs)
        self._model_lock = threading.Lock()
        self._transport: GeminiJsonTransport | None = None

    def _get_transport(self) -> GeminiJsonTransport:
        if self._transport is None:
            with self._model_lock:
                if self._transport is None:
                    self._transport = GeminiJsonTransport(
                        api_key=self.api_key, model=self.model,
                        request_timeout=float(self.kwargs.get("request_timeout", 180)),
                        rpm=int(self.kwargs.get("rpm_limit", 0)),
                        tpm=int(self.kwargs.get("tpm_limit", 0)),
                        rpd=int(self.kwargs.get("gemini_rpd_limit", 100)),
                        free_mode=bool(self.kwargs.get("gemini_free_mode", True)),
                    )
        return self._transport

    def _request(self, prompt: str, temperature: float, json_mode: bool = False) -> str:
        transport = self._get_transport()
        def record_usage(*, prompt: int, completion: int, total: int) -> None:
            self._record_usage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)

        return transport.generate(
            prompt, [], schema=None, text_only=True,
            json_mode=bool(json_mode and self.kwargs.get("json_mode", True)),
            max_output_tokens=int(self.kwargs.get("max_tokens", 32000)),
            temperature=temperature, usage_callback=record_usage,
        )

    def _call_llm(self, prompt: str, temperature: float) -> str:
        return self._request(prompt, temperature, False)

    def call_json(self, prompt: str, temperature: float) -> str:
        return self._request(prompt, temperature, True)
