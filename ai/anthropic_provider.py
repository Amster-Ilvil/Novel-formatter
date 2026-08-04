# -*- coding: utf-8 -*-
from __future__ import annotations

import itertools
import logging
import threading

from .base import AIProvider


class AnthropicProvider(AIProvider):
    @property
    def name(self) -> str:
        return "anthropic"

    def __init__(self, api_key: str, model: str = "", **kwargs):
        super().__init__(api_key, model or "claude-sonnet-4-5", **kwargs)
        self._clients = []
        self._client_cycle = None
        self._client_lock = threading.Lock()
        self._cycle_lock = threading.Lock()

    def _build_clients(self):
        try:
            import httpx
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError("请安装 anthropic: pip install anthropic") from exc
        keys = [x.strip() for x in str(self.api_key or "").split(",") if x.strip()]
        if not keys:
            raise ValueError("Anthropic API Key 不能为空")
        timeout = float(self.kwargs.get("request_timeout", 180) or 180)
        base_url = str(self.kwargs.get("base_url", "") or "").strip() or None
        for key in keys:
            client_kwargs = {
                "api_key": key,
                "timeout": timeout,
                "max_retries": 0,
                "http_client": httpx.Client(
                    limits=httpx.Limits(
                        max_connections=128,
                        max_keepalive_connections=64,
                        keepalive_expiry=60.0,
                    ),
                    timeout=timeout,
                ),
            }
            if base_url:
                client_kwargs["base_url"] = base_url
            self._clients.append(Anthropic(**client_kwargs))
        self._client_cycle = itertools.cycle(self._clients)

    def _next_client(self):
        if not self._clients:
            with self._client_lock:
                if not self._clients:
                    self._build_clients()
        with self._cycle_lock:
            return next(self._client_cycle)

    def _call_llm(self, prompt: str, temperature: float) -> str:
        params = {
            "model": self.model,
            "max_tokens": int(self.kwargs.get("max_tokens", 24000)),
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.kwargs.get("top_p") is not None:
            params["top_p"] = float(self.kwargs.get("top_p"))
        response = self._next_client().messages.create(**params)
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage else 0
        self._record_usage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cached_tokens=(
                (getattr(usage, "cache_read_input_tokens", 0) or 0)
                + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
            ) if usage else 0,
        )
        return "".join(getattr(item, "text", "") for item in response.content)

    def close(self) -> None:
        """Close all Anthropic/httpx clients; safe after success or failure."""
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
                    "Failed to close an Anthropic client", exc_info=True
                )
