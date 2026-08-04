#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DeepSeek-optimised OpenAI-compatible provider.

The official DeepSeek API has several behaviours that differ from a generic
OpenAI-compatible endpoint:

* V4 thinking mode is enabled by default, which is wasteful for deterministic
  OCR correction/typesetting unless explicitly requested.
* Context caching benefits from a stable message prefix, so the static
  instructions and variable document payload are sent as separate messages.
* Cache hit/miss usage fields use DeepSeek-specific names.
* The legacy ``deepseek-chat`` / ``deepseek-reasoner`` aliases are retired in
  July 2026 and are normalised to V4-Flash while preserving thinking intent.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from .base import AIOutputTruncatedError
from .openai_provider import OpenAIProvider


LEGACY_MODEL_MAP = {
    "deepseek-chat": ("deepseek-v4-flash", False),
    "deepseek-reasoner": ("deepseek-v4-flash", True),
}


def normalise_deepseek_model(model: str, thinking: bool = False) -> tuple[str, bool]:
    """Return a current model name and preserve legacy reasoning semantics."""
    raw = str(model or "").strip()
    mapped = LEGACY_MODEL_MAP.get(raw.lower())
    if mapped:
        return mapped
    return raw or "deepseek-v4-flash", bool(thinking)


class DeepSeekProvider(OpenAIProvider):

    @property
    def name(self) -> str:
        return "deepseek"

    def __init__(self, api_key: str, model: str = "", **kwargs):
        kwargs.setdefault("base_url", "https://api.deepseek.com")
        requested_thinking = bool(kwargs.get("deepseek_thinking", False))
        normalised_model, normalised_thinking = normalise_deepseek_model(
            model, requested_thinking
        )
        kwargs["deepseek_thinking"] = normalised_thinking
        kwargs.setdefault("deepseek_reasoning_effort", "high")
        kwargs.setdefault("deepseek_user_id", "novel_formatter")
        super().__init__(api_key, normalised_model, **kwargs)
        self.model = normalised_model
        self.deepseek_thinking = normalised_thinking
        self.deepseek_reasoning_effort = str(
            kwargs.get("deepseek_reasoning_effort", "high") or "high"
        ).strip().lower()
        if self.deepseek_reasoning_effort not in {"high", "max"}:
            self.deepseek_reasoning_effort = "high"
        self.deepseek_user_id = self._normalise_user_id(
            kwargs.get("deepseek_user_id", "novel_formatter")
        )

    @staticmethod
    def _normalise_user_id(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "").strip())
        return (cleaned or "novel_formatter")[:512]

    def _is_official_endpoint(self) -> bool:
        try:
            host = (urlparse(str(self.base_url or "")).hostname or "").lower()
        except Exception:
            return False
        return host == "api.deepseek.com" or host.endswith(".api.deepseek.com")

    @staticmethod
    def _split_cached_messages(prompt: str, json_mode: bool) -> list[dict]:
        """Separate static instructions from changing payload for KV-cache hits."""
        text = str(prompt or "")
        marker = "INPUT:\n"
        position = text.rfind(marker)
        if position < 0:
            return [{"role": "user", "content": text}]
        system = text[:position].strip()
        payload = text[position + len(marker):].strip()
        # Keep a constant user-prefix too. DeepSeek can persist common prefixes
        # after repeated requests, reducing both latency and billed miss tokens.
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": "JSON INPUT:\n" + payload},
        ]

    @staticmethod
    def _unsupported_parameter(exc: Exception, *names: str) -> bool:
        status = getattr(exc, "status_code", None)
        message = str(exc).lower()
        return status in (400, 404, 422) and any(name.lower() in message for name in names)

    def _record_deepseek_usage(self, response) -> None:
        usage = getattr(response, "usage", None)
        prompt_details = getattr(usage, "prompt_tokens_details", None) if usage else None
        completion_details = getattr(usage, "completion_tokens_details", None) if usage else None
        cache_hit = 0
        cache_miss = 0
        if usage:
            cache_hit = int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0)
            cache_miss = int(getattr(usage, "prompt_cache_miss_tokens", 0) or 0)
        if not cache_hit and prompt_details:
            cache_hit = int(getattr(prompt_details, "cached_tokens", 0) or 0)
        reasoning_tokens = 0
        if completion_details:
            reasoning_tokens = int(getattr(completion_details, "reasoning_tokens", 0) or 0)
        if usage and not reasoning_tokens:
            reasoning_tokens = int(getattr(usage, "reasoning_tokens", 0) or 0)
        self._record_usage(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            total_tokens=getattr(usage, "total_tokens", 0) if usage else 0,
            cached_tokens=cache_hit,
            cache_miss_tokens=cache_miss,
            reasoning_tokens=reasoning_tokens,
        )

    def _request(self, prompt: str, temperature: float, json_mode: bool = False) -> str:
        client = self._next_client()
        messages = self._split_cached_messages(prompt, json_mode)
        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(self.kwargs.get("max_tokens", 48000) or 48000),
        }
        if self.kwargs.get("top_p") is not None and not self.deepseek_thinking:
            params["top_p"] = float(self.kwargs.get("top_p"))
        if not self.deepseek_thinking:
            params["temperature"] = temperature
        if json_mode and self._json_mode_supported:
            params["response_format"] = {"type": "json_object"}

        extra_body = {
            "thinking": {"type": "enabled" if self.deepseek_thinking else "disabled"},
            "user_id": self.deepseek_user_id,
        }
        params["extra_body"] = extra_body
        if self.deepseek_thinking:
            # Put it in extra_body so older OpenAI SDK versions can still forward
            # the current DeepSeek parameter without knowing its typed signature.
            extra_body["reasoning_effort"] = self.deepseek_reasoning_effort

        def send(current_params: dict):
            response = client.chat.completions.create(**current_params)
            self._record_deepseek_usage(response)
            return response

        try:
            response = send(params)
        except Exception as exc:
            # Some third-party DeepSeek-compatible gateways lag behind the official
            # API. Keep official behaviour strict; for custom gateways, remove only
            # parameters that the error explicitly reports as unsupported.
            if json_mode and "response_format" in params and self._unsupported_parameter(
                exc, "response_format", "json mode", "json_object"
            ):
                self._json_mode_supported = False
                params.pop("response_format", None)
                response = send(params)
            elif (not self._is_official_endpoint()) and self._unsupported_parameter(
                exc, "thinking", "reasoning_effort", "user_id"
            ):
                message = str(exc).lower()
                compatible_body = dict(params.get("extra_body") or {})
                if "reasoning_effort" in message:
                    compatible_body.pop("reasoning_effort", None)
                if "user_id" in message:
                    compatible_body.pop("user_id", None)
                if "thinking" in message:
                    compatible_body.pop("thinking", None)
                # Unknown gateway wording: remove the extension block as the final
                # compatibility fallback. Otherwise preserve thinking=disabled when
                # the gateway only rejected user_id or reasoning_effort.
                if len(compatible_body) == len(params.get("extra_body") or {}):
                    compatible_body = {}
                if compatible_body:
                    params["extra_body"] = compatible_body
                else:
                    params.pop("extra_body", None)
                response = send(params)
            else:
                raise

        choice = response.choices[0]
        message = choice.message
        content = getattr(message, "content", None) or ""
        finish_reason = str(getattr(choice, "finish_reason", "") or "").lower()
        if finish_reason == "length":
            raise AIOutputTruncatedError(content)

        # DeepSeek documents an occasional empty content response in JSON mode.
        # Retry only once, without recursively splitting the whole batch, and add a
        # short explicit reminder. The first empty response usually has no output
        # tokens, so this is cheaper than immediately creating many child requests.
        if not content.strip() and json_mode:
            retry_params = dict(params)
            retry_messages = [dict(item) for item in messages]
            retry_messages[0] = dict(retry_messages[0])
            retry_messages[0]["content"] = (
                str(retry_messages[0].get("content", ""))
                + "\nYou must return one non-empty JSON object."
            )
            retry_params["messages"] = retry_messages
            response = send(retry_params)
            choice = response.choices[0]
            message = choice.message
            content = getattr(message, "content", None) or ""
            finish_reason = str(getattr(choice, "finish_reason", "") or "").lower()
            if finish_reason == "length":
                raise AIOutputTruncatedError(content)

        # Never use reasoning_content as the final document patch. It is internal
        # reasoning, not the requested JSON answer, and can be extremely large.
        return content or ""
