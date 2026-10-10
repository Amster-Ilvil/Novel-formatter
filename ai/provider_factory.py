# -*- coding: utf-8 -*-
from __future__ import annotations
from .config import AISettings, validate_api_key
from .openai_provider import OpenAIProvider
from .deepseek_provider import DeepSeekProvider
from .gemini_provider import GeminiProvider
from .anthropic_provider import AnthropicProvider


def create_provider(settings: AISettings):
    name = settings.provider.lower().strip()
    kwargs = settings.provider_kwargs()
    api_key = settings.api_key
    if settings.requires_key:
        api_key = validate_api_key(api_key)
    elif api_key:
        api_key = validate_api_key(api_key, required=False)
    if name == "openai":
        return OpenAIProvider(api_key, settings.model, **kwargs)
    if name == "deepseek":
        return DeepSeekProvider(api_key, settings.model, **kwargs)
    if name == "gemini":
        return GeminiProvider(api_key, settings.model, **kwargs)
    if name == "anthropic":
        return AnthropicProvider(api_key, settings.model, **kwargs)
    if name == "zhipu":
        kwargs.setdefault("base_url", "https://open.bigmodel.cn/api/paas/v4")
        return OpenAIProvider(api_key, settings.model, **kwargs)
    if name == "zai":
        kwargs.setdefault("base_url", "https://api.z.ai/api/paas/v4/")
        return OpenAIProvider(api_key, settings.model, **kwargs)
    if name == "openrouter":
        kwargs.setdefault("base_url", "https://openrouter.ai/api/v1")
        return OpenAIProvider(api_key, settings.model, **kwargs)
    if name == "ollama":
        kwargs.setdefault("base_url", "http://127.0.0.1:11434/v1")
        return OpenAIProvider(api_key or "ollama", settings.model, **kwargs)
    if name == "custom":
        if not settings.base_url:
            raise ValueError("自定义 Provider 必须填写 Base URL")
        return OpenAIProvider(api_key or "not-required", settings.model, **kwargs)
    raise ValueError(f"不支持的 AI Provider: {settings.provider}")


def test_provider(settings: AISettings) -> str:
    with create_provider(settings) as provider:
        reply = provider._call_llm("Reply with exactly: OK", 0.0).strip()
        if not reply:
            raise RuntimeError("服务已响应，但返回内容为空")
        return reply[:200]


def probe_multimodal_provider(settings: AISettings, *, client_factory=None) -> str:
    """Verify that the configured model really consumes image input.

    The probe is generated locally and the API key remains only in ``settings``.
    A text-only connectivity test can succeed even when the selected model rejects
    images, so the AI image workspace exposes this as a separate explicit check.
    """
    import tempfile
    from pathlib import Path

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise ImportError("测试图片能力需要 Pillow") from exc
    if client_factory is None:
        from .multimodal_client import MultimodalClient
        client_factory = MultimodalClient

    with tempfile.TemporaryDirectory(prefix="novel-formatter-vision-probe-") as td:
        path = Path(td) / "vision-probe.png"
        image = Image.new("RGB", (360, 140), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((8, 8, 352, 132), outline="black", width=4)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 54)
        except Exception:
            font = ImageFont.load_default()
        draw.text((82, 38), "NF42", fill="black", font=font)
        image.save(path, format="PNG")

        prompt = (
            "这是图片输入能力测试。请读取图中黑框里的字符。"
            "只返回紧凑JSON，例如 {\"seen\":\"NF42\"}，不要解释。"
        )
        with client_factory(settings) as client:
            reply = str(client.call_json(prompt, [path], temperature=0.0) or "").strip()
        if "42" not in reply:
            raise RuntimeError(
                "接口已返回，但未能确认模型读取到了测试图片中的 NF42。"
                "这通常表示当前模型是文本模型，或图片输入被网关忽略。返回摘要：" + reply[:160]
            )
        return reply[:200]
