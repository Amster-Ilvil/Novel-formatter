#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI API 校正器
支持自定义 base_url（兼容 API 代理）。
"""

from __future__ import annotations

from .base import AIProvider
from .config import validate_api_key
import itertools
import logging
import threading


class OpenAIProvider(AIProvider):

    @property
    def name(self) -> str:
        return "openai"

    def __init__(self, api_key: str, model: str = "", **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.model = model or "gpt-4o"
        self.base_url = kwargs.get("base_url", None)
        self._clients = []
        self._client_lock = threading.Lock()
        self._client_cycle = None
        self._cycle_lock = threading.Lock()
        self._json_mode_supported = bool(kwargs.get("json_mode", True))

    def _build_clients(self):
        try:
            import httpx
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")
        raw_key = str(self.api_key or "")
        if raw_key and raw_key not in {"not-required", "ollama"}:
            raw_key = validate_api_key(raw_key)
        keys = [x.strip() for x in raw_key.split(",") if x.strip()] or ["not-required"]
        timeout = float(self.kwargs.get("request_timeout", 180) or 180)
        limits = httpx.Limits(max_connections=128, max_keepalive_connections=64, keepalive_expiry=60.0)
        for key in keys:
            client_kwargs = {
                "api_key": key,
                "timeout": timeout,
                "max_retries": 0,
                "http_client": httpx.Client(limits=limits, timeout=timeout),
            }
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self._clients.append(OpenAI(**client_kwargs))
        self._client_cycle = itertools.cycle(self._clients)

    def _next_client(self):
        if not self._clients:
            with self._client_lock:
                if not self._clients:
                    self._build_clients()
        with self._cycle_lock:
            return next(self._client_cycle)

    def _request(self, prompt: str, temperature: float, json_mode: bool = False) -> str:
        client = self._next_client()
        params = dict(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=int(self.kwargs.get("max_tokens", 32000)),
        )
        if self.kwargs.get("top_p") is not None:
            params["top_p"] = float(self.kwargs.get("top_p"))
        if json_mode and self._json_mode_supported:
            params["response_format"] = {"type": "json_object"}
        try:
            response = client.chat.completions.create(**params)
        except Exception as exc:
            message = str(exc).lower()
            status = getattr(exc, "status_code", None)
            unsupported_json = status in (400, 404, 422) and any(
                marker in message for marker in ("response_format", "json mode", "json_object", "unsupported parameter")
            )
            if json_mode and "response_format" in params and unsupported_json:
                self._json_mode_supported = False
                params.pop("response_format", None)
                response = client.chat.completions.create(**params)
            else:
                raise
        usage = getattr(response, "usage", None)
        prompt_details = getattr(usage, "prompt_tokens_details", None) if usage else None
        completion_details = getattr(usage, "completion_tokens_details", None) if usage else None
        self._record_usage(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            total_tokens=getattr(usage, "total_tokens", 0) if usage else 0,
            cached_tokens=getattr(prompt_details, "cached_tokens", 0) if prompt_details else 0,
            reasoning_tokens=getattr(completion_details, "reasoning_tokens", 0) if completion_details else 0,
        )
        msg = response.choices[0].message
        content = getattr(msg, "content", None)
        if not content:
            content = getattr(msg, "reasoning_content", None)
        return content or ""

    def _call_llm(self, prompt: str, temperature: float) -> str:
        return self._request(prompt, temperature, json_mode=False)

    def call_json(self, prompt: str, temperature: float) -> str:
        return self._request(prompt, temperature, json_mode=True)

    def close(self) -> None:
        """Close all SDK/httpx clients and make cleanup safe to call repeatedly."""
        with self._client_lock:
            clients = self._clients
            self._clients = []
            self._client_cycle = None
        for client in clients:
            try:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
            except Exception:
                logging.getLogger(__name__).warning(
                    "Failed to close an OpenAI-compatible client", exc_info=True
                )
