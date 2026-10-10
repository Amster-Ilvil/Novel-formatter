# -*- coding: utf-8 -*-
"""Small multimodal JSON client shared by the AI image-book workflow.

This deliberately sits beside the existing text AI provider stack instead of
changing it.  The image workflow can therefore use any configured provider whose
selected model actually accepts image inputs, while Formatter keeps its proven
text-only contract unchanged.
"""
from __future__ import annotations

import base64
import itertools
import logging
import threading
import time
from pathlib import Path
from typing import Iterable

from .config import AISettings, validate_api_key
from .glm_compat import is_glm53_flash, looks_always_thinking_error, looks_image_payload_error
from .request_limiter import retry_delay_seconds


class MultimodalCapabilityError(RuntimeError):
    """The endpoint/model rejected image input or cannot be called as configured."""


class MultimodalImageInputError(RuntimeError):
    """The endpoint is reachable but rejected or failed to decode the image payload."""


class MultimodalClient:
    """Provider-neutral image + text caller with usage accounting.

    Supported transport families:
      * OpenAI-compatible: OpenAI, OpenRouter, DeepSeek, Ollama, custom endpoints
      * Anthropic Messages
      * Google Gemini ``generate_content``

    Model capability is intentionally *not* inferred from the model name.  A
    provider may expose text-only and vision models through the same endpoint;
    the real request is the source of truth.
    """

    def __init__(self, settings: AISettings):
        self.settings = settings
        self._clients = []
        self._client_cycle = None
        self._client_lock = threading.Lock()
        self._cycle_lock = threading.Lock()
        self._anthropic_clients = []
        self._anthropic_cycle = None
        self._anthropic_client_lock = threading.Lock()
        self._anthropic_cycle_lock = threading.Lock()
        self._gemini_model = None
        self._gemini_model_lock = threading.Lock()
        self._usage_lock = threading.Lock()
        self._usage = {
            "requests": 0,
            "transport_attempts": 0,
            "transport_retries": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
        }
        self._image_cache_lock = threading.Lock()
        self._image_cache: dict[tuple[str, int, int], tuple[str, str]] = {}
        # Some GLM-5.3 Flash deployments advertise the thinking switch but
        # reject ``disabled`` at runtime.  Keep this override request-local so
        # the user's saved default remains disabled for compatible models.
        self._glm_thinking_override: str | None = None

    @property
    def provider(self) -> str:
        return str(self.settings.provider or "").strip().lower()

    def usage_snapshot(self) -> dict:
        with self._usage_lock:
            return dict(self._usage)

    def _record_usage(self, *, prompt=0, completion=0, total=0, cached=0, reasoning=0) -> None:
        prompt = max(0, int(prompt or 0))
        completion = max(0, int(completion or 0))
        total = max(0, int(total or 0)) or prompt + completion
        with self._usage_lock:
            self._usage["requests"] += 1
            self._usage["prompt_tokens"] += prompt
            self._usage["completion_tokens"] += completion
            self._usage["total_tokens"] += total
            self._usage["cached_tokens"] += max(0, int(cached or 0))
            self._usage["reasoning_tokens"] += max(0, int(reasoning or 0))

    def _image_data(self, path: str | Path) -> tuple[str, str]:
        """Read/base64 an image once per unchanged file.

        Image-book requests may retry transient network failures or re-use the same
        prepared page in a targeted audit.  Caching the encoded payload avoids
        repeatedly reading and base64-encoding large scans without changing the
        provider request or billing semantics.
        """
        p = Path(path)
        st = p.stat()
        key = (str(p.resolve()), int(st.st_mtime_ns), int(st.st_size))
        with self._image_cache_lock:
            cached = self._image_cache.get(key)
        if cached is not None:
            return cached
        suffix = p.suffix.lower()
        mime = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }.get(suffix, "image/png")
        value = (mime, base64.b64encode(p.read_bytes()).decode("ascii"))
        with self._image_cache_lock:
            # Keep the cache deliberately small; a run only needs recent prepared
            # pages/review crops and this prevents a whole large book being pinned.
            if len(self._image_cache) >= 16:
                self._image_cache.pop(next(iter(self._image_cache)))
            self._image_cache[key] = value
        return value

    @staticmethod
    def _looks_capability_error(exc: Exception) -> bool:
        lowered = str(exc).lower()
        markers = (
            "unsupported modality", "unsupported image", "image input is not supported",
            "does not support image", "doesn't support image", "only text input",
            "image_url is not supported", "multimodal input is not supported",
            "invalid content type: image", "vision is not supported",
        )
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _looks_image_payload_error(exc: Exception) -> bool:
        return looks_image_payload_error(exc)

    @staticmethod
    def _looks_always_thinking_error(exc: Exception) -> bool:
        return looks_always_thinking_error(exc)

    @staticmethod
    def _looks_transient_error(exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        if status is None:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None) if response is not None else None
        try:
            if int(status) in {408, 409, 425, 429, 500, 502, 503, 504}:
                return True
        except Exception:
            pass
        lowered = str(exc).lower()
        markers = (
            "timeout", "timed out", "rate limit", "too many requests", "temporarily unavailable",
            "connection reset", "connection aborted", "connection refused", "server disconnected",
            "service unavailable", "bad gateway", "gateway timeout", "internal server error",
            "overloaded", "try again", "network error",
        )
        return any(marker in lowered for marker in markers)

    def _record_transport_attempt(self, *, retry: bool) -> None:
        with self._usage_lock:
            self._usage["transport_attempts"] += 1
            if retry:
                self._usage["transport_retries"] += 1

    def call_json(
        self, prompt: str, image_paths: Iterable[str | Path], *,
        temperature: float = 0.0, reasoning_effort: str | None = None,
        cancel_check=None,
    ) -> str:
        images = [str(Path(p)) for p in image_paths]
        if not images:
            raise ValueError("multimodal request requires at least one image")
        provider = self.provider
        max_attempts = 3
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            if cancel_check and cancel_check():
                raise RuntimeError("AI 图文处理已停止")
            self._record_transport_attempt(retry=attempt > 0)
            try:
                if provider == "anthropic":
                    return self._call_anthropic(prompt, images, temperature)
                if provider == "gemini":
                    return self._call_gemini(prompt, images, temperature)
                if reasoning_effort is None:
                    return self._call_openai_compatible(prompt, images, temperature)
                try:
                    return self._call_openai_compatible(
                        prompt, images, temperature, reasoning_effort=reasoning_effort
                    )
                except TypeError as exc:
                    if "reasoning_effort" not in str(exc):
                        raise
                    return self._call_openai_compatible(prompt, images, temperature)
            except MultimodalCapabilityError:
                raise
            except Exception as exc:
                last_error = exc
                if self._looks_capability_error(exc):
                    raise MultimodalCapabilityError(
                        "当前接口或模型拒绝了图片输入。请在 AI 设置中改用真正支持图片的模型，"
                        "或使用自定义 OpenAI 兼容接口。原始错误：" + str(exc)
                    ) from exc
                # GLM-5.3 Flash is already sent with always-on thinking and an
                # explicit reasoning_effort. If a gateway still emits the stock
                # "cannot disable thinking" message, expose the real route/model
                # compatibility issue instead of suggesting a disabled mode that
                # this model does not support.
                if (
                    self.provider in {"zhipu", "zai"}
                    and is_glm53_flash(self.provider, self.settings.model)
                    and self._looks_always_thinking_error(exc)
                ):
                    effort = "high" if bool(getattr(self.settings, "glm_thinking", False)) else "low"
                    raise RuntimeError(
                        "GLM-5.3 Flash 本身无法关闭思考。Novel Formatter 已按当前模型要求发送 "
                        f"thinking=enabled + reasoning_effort={effort}，但服务端仍拒绝请求。"
                        "请确认模型 ID、Base URL 与网关是否支持 GLM-5.3 Flash 的 reasoning_effort。"
                        "原始错误：" + str(exc)
                    ) from exc
                if self._looks_image_payload_error(exc):
                    raise MultimodalImageInputError(
                        "接口鉴权/路由已到达服务端，但图片内容未被成功解析。"
                        "Novel Formatter 会发送本地 PNG/JPEG 的 Base64 Data URL；"
                        "请优先使用“测试图片能力”验证当前 Base URL + 模型组合。原始错误：" + str(exc)
                    ) from exc
                if attempt + 1 >= max_attempts or not self._looks_transient_error(exc):
                    raise
                # Share the same Retry-After-aware bounded backoff used by the
                # text provider stack. Re-entering a pooled provider path also
                # advances the configured API-key cycle where applicable.
                remaining = max(0.0, float(retry_delay_seconds(exc, attempt)))
                if remaining <= 0.0:
                    # Keep the retry path scheduler-friendly even when a provider
                    # explicitly allows an immediate retry (Retry-After: 0).
                    time.sleep(0.0)
                    continue
                deadline = time.monotonic() + remaining
                while True:
                    if cancel_check and cancel_check():
                        raise RuntimeError("AI 图文处理已停止")
                    left = deadline - time.monotonic()
                    if left <= 0:
                        break
                    time.sleep(min(0.20, left))
        assert last_error is not None
        raise last_error

    # ------------------------------------------------------------------
    # OpenAI-compatible
    # ------------------------------------------------------------------
    def _build_openai_clients(self):
        try:
            import httpx
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("请安装 openai/httpx：pip install openai httpx") from exc
        settings = self.settings
        key = settings.api_key
        if settings.requires_key:
            key = validate_api_key(key)
        elif key:
            key = validate_api_key(key, required=False)
        else:
            key = "local"
        keys = [x.strip() for x in str(key or "").split(",") if x.strip()] or ["local"]
        base_url = str(settings.base_url or "").strip()
        if self.provider == "zhipu" and not base_url:
            base_url = "https://open.bigmodel.cn/api/paas/v4"
        elif self.provider == "zai" and not base_url:
            base_url = "https://api.z.ai/api/paas/v4/"
        elif self.provider == "openrouter" and not base_url:
            base_url = "https://openrouter.ai/api/v1"
        elif self.provider == "ollama" and not base_url:
            base_url = "http://127.0.0.1:11434/v1"
        elif self.provider == "openai" and not base_url:
            base_url = "https://api.openai.com/v1"
        elif self.provider == "deepseek" and not base_url:
            base_url = "https://api.deepseek.com"
        if self.provider == "custom" and not base_url:
            raise ValueError("自定义 Provider 必须填写 Base URL")

        timeout = float(settings.request_timeout or 180)
        for item in keys:
            kwargs = {
                "api_key": item,
                "timeout": timeout,
                "max_retries": 0,
                "http_client": httpx.Client(
                    limits=httpx.Limits(max_connections=32, max_keepalive_connections=16, keepalive_expiry=60.0),
                    timeout=timeout,
                ),
            }
            if base_url:
                kwargs["base_url"] = base_url
            self._clients.append(OpenAI(**kwargs))
        self._client_cycle = itertools.cycle(self._clients)

    def _next_openai_client(self):
        if not self._clients:
            with self._client_lock:
                if not self._clients:
                    self._build_openai_clients()
        with self._cycle_lock:
            return next(self._client_cycle)

    def _openai_request_params(
        self, prompt: str, images: list[str], temperature: float, *,
        reasoning_effort: str | None = None,
    ) -> dict:
        image_parts = []
        for image in images:
            mime, payload = self._image_data(image)
            image_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{payload}"},
            })

        model = str(self.settings.model or "").strip().lower()
        is_glm = str(self.provider or "") in {"zhipu", "zai"}
        # BigModel/Z.AI examples and some gateways are happiest when media comes
        # before the instruction.  Other OpenAI-compatible providers retain the
        # usual text-first order.
        if is_glm:
            content = [*image_parts, {"type": "text", "text": prompt}]
        else:
            content = [{"type": "text", "text": prompt}, *image_parts]

        params = {
            "model": self.settings.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": int(self.settings.max_tokens or 24000),
        }
        # GLM-5.3 Flash is an always-thinking model. Current BigModel/Z.AI
        # deployments accept ``reasoning_effort`` = low/high/max and reject
        # ``thinking.type=disabled``.  For book OCR we map the existing boolean
        # setting to the two useful latency presets: unchecked = low, checked =
        # high.  Keep the field in ``extra_body`` so older OpenAI SDK versions
        # still send it as a top-level JSON member.
        if is_glm53_flash(self.provider, model):
            deep_thinking = bool(getattr(self.settings, "glm_thinking", False))
            requested_effort = str(reasoning_effort or "").strip().lower()
            if requested_effort not in {"low", "high", "max"}:
                requested_effort = ""
            effort = requested_effort or self._glm_thinking_override or ("high" if deep_thinking else "low")
            params["temperature"] = 1.0
            params["extra_body"] = {
                "thinking": {"type": "enabled"},
                "reasoning_effort": effort,
            }
        else:
            params["temperature"] = float(temperature)
        if bool(self.settings.json_mode):
            params["response_format"] = {"type": "json_object"}
        return params

    def _call_openai_compatible(
        self, prompt: str, images: list[str], temperature: float, *,
        reasoning_effort: str | None = None,
    ) -> str:
        params = self._openai_request_params(prompt, images, temperature, reasoning_effort=reasoning_effort)
        client = self._next_openai_client()
        try:
            response = client.chat.completions.create(**params)
        except Exception as exc:
            # OpenAI-compatible vendors vary in support for JSON mode and
            # temperature/reasoning parameters.  Retry only by removing optional
            # controls; never mutate the image payload or silently fall back to a
            # text-only request.
            lowered = str(exc).lower()
            retry = dict(params)
            changed = False
            if "response_format" in retry and any(x in lowered for x in ("response_format", "json mode", "json_object", "unsupported parameter")):
                retry.pop("response_format", None); changed = True
            if "temperature" in retry and any(x in lowered for x in ("temperature", "unsupported parameter")):
                retry.pop("temperature", None); changed = True
            if not changed:
                raise
            response = client.chat.completions.create(**retry)

        usage = getattr(response, "usage", None)
        details = getattr(usage, "prompt_tokens_details", None) if usage else None
        completion_details = getattr(usage, "completion_tokens_details", None) if usage else None
        self._record_usage(
            prompt=getattr(usage, "prompt_tokens", 0) if usage else 0,
            completion=getattr(usage, "completion_tokens", 0) if usage else 0,
            total=getattr(usage, "total_tokens", 0) if usage else 0,
            cached=getattr(details, "cached_tokens", 0) if details else 0,
            reasoning=getattr(completion_details, "reasoning_tokens", 0) if completion_details else 0,
        )
        message = response.choices[0].message
        text = getattr(message, "content", None) or getattr(message, "reasoning_content", None) or ""
        return str(text)

    # ------------------------------------------------------------------
    # Anthropic
    # ------------------------------------------------------------------
    def _build_anthropic_clients(self) -> None:
        try:
            import httpx
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError("请安装 anthropic/httpx：pip install anthropic httpx") from exc
        keys = [
            validate_api_key(item)
            for item in str(self.settings.api_key or "").split(",")
            if item.strip()
        ]
        if not keys:
            raise ValueError("Anthropic API Key 不能为空")
        timeout = float(self.settings.request_timeout or 180)
        base_url = str(self.settings.base_url or "").strip()
        clients = []
        try:
            for key in keys:
                kwargs = {
                    "api_key": key,
                    "timeout": timeout,
                    "max_retries": 0,
                    "http_client": httpx.Client(
                        limits=httpx.Limits(
                            max_connections=32,
                            max_keepalive_connections=16,
                            keepalive_expiry=60.0,
                        ),
                        timeout=timeout,
                    ),
                }
                if base_url:
                    kwargs["base_url"] = base_url
                clients.append(Anthropic(**kwargs))
        except Exception:
            for client in clients:
                try:
                    close = getattr(client, "close", None)
                    if callable(close):
                        close()
                except Exception:
                    pass
            raise
        self._anthropic_clients.extend(clients)
        self._anthropic_cycle = itertools.cycle(self._anthropic_clients)

    def _next_anthropic_client(self):
        if not self._anthropic_clients:
            with self._anthropic_client_lock:
                if not self._anthropic_clients:
                    self._build_anthropic_clients()
        with self._anthropic_cycle_lock:
            return next(self._anthropic_cycle)

    def _call_anthropic(self, prompt: str, images: list[str], temperature: float) -> str:
        client = self._next_anthropic_client()
        content = [{"type": "text", "text": prompt}]
        for image in images:
            mime, payload = self._image_data(image)
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": payload},
            })
        response = client.messages.create(
            model=self.settings.model,
            max_tokens=int(self.settings.max_tokens or 24000),
            temperature=float(temperature),
            messages=[{"role": "user", "content": content}],
        )
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "output_tokens", 0) if usage else 0
        self._record_usage(
            prompt=prompt_tokens,
            completion=completion_tokens,
            total=prompt_tokens + completion_tokens,
            cached=(
                (getattr(usage, "cache_read_input_tokens", 0) or 0)
                + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
            ) if usage else 0,
        )
        return "".join(str(getattr(item, "text", "") or "") for item in response.content)

    # ------------------------------------------------------------------
    # Gemini
    # ------------------------------------------------------------------
    def _get_gemini_model(self):
        if self._gemini_model is None:
            with self._gemini_model_lock:
                if self._gemini_model is None:
                    try:
                        import google.generativeai as genai
                    except ImportError as exc:
                        raise ImportError("请安装 google-generativeai：pip install google-generativeai") from exc
                    genai.configure(api_key=validate_api_key(self.settings.api_key))
                    self._gemini_model = genai.GenerativeModel(self.settings.model)
        return self._gemini_model

    def _call_gemini(self, prompt: str, images: list[str], temperature: float) -> str:
        try:
            import google.generativeai as genai
            from PIL import Image
        except ImportError as exc:
            raise ImportError("请安装 google-generativeai 和 Pillow") from exc
        model = self._get_gemini_model()
        opened = []
        try:
            for path in images:
                image = Image.open(path)
                image.load()
                opened.append(image)
            config_kwargs = {
                "temperature": float(temperature),
                "max_output_tokens": int(self.settings.max_tokens or 24000),
            }
            if bool(self.settings.json_mode):
                config_kwargs["response_mime_type"] = "application/json"
            try:
                response = model.generate_content(
                    [prompt, *opened],
                    generation_config=genai.types.GenerationConfig(**config_kwargs),
                )
            except Exception as exc:
                if "response_mime_type" not in str(exc).lower():
                    raise
                config_kwargs.pop("response_mime_type", None)
                response = model.generate_content(
                    [prompt, *opened],
                    generation_config=genai.types.GenerationConfig(**config_kwargs),
                )
            usage = getattr(response, "usage_metadata", None)
            self._record_usage(
                prompt=getattr(usage, "prompt_token_count", 0) if usage else 0,
                completion=getattr(usage, "candidates_token_count", 0) if usage else 0,
                total=getattr(usage, "total_token_count", 0) if usage else 0,
                cached=getattr(usage, "cached_content_token_count", 0) if usage else 0,
                reasoning=getattr(usage, "thoughts_token_count", 0) if usage else 0,
            )
            return str(getattr(response, "text", "") or "")
        finally:
            for image in opened:
                try:
                    image.close()
                except Exception:
                    pass

    def close(self) -> None:
        with self._client_lock:
            clients = self._clients
            self._clients = []
            self._client_cycle = None
        with self._anthropic_client_lock:
            anthropic_clients = self._anthropic_clients
            self._anthropic_clients = []
            self._anthropic_cycle = None
        with self._gemini_model_lock:
            self._gemini_model = None
        with self._image_cache_lock:
            self._image_cache.clear()
        for client in [*clients, *anthropic_clients]:
            try:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
            except Exception:
                logging.getLogger(__name__).debug("multimodal client close failed", exc_info=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
