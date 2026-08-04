# -*- coding: utf-8 -*-
"""Persistent AI settings with environment-variable fallback."""
from __future__ import annotations

import json
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path

from utils.atomic_io import atomic_write_json

APP_DIR_NAME = "NovelFormatter"


class APIKeyValidationError(ValueError):
    pass


def normalise_api_key(value: str) -> str:
    """Trim harmless paste artefacts without ever inventing a key."""
    raw = str(value or "").strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"\"", "'"}:
        raw = raw[1:-1].strip()
    if raw.lower().startswith("bearer "):
        raw = raw[7:].strip()
    return ",".join(part.strip() for part in raw.split(",") if part.strip())


def api_key_validation_error(value: str) -> str:
    key = normalise_api_key(value)
    if not key:
        return "API Key 为空"
    lowered = key.lower()
    placeholder_markers = ("请输入", "密钥", "api key", "apikey", "your_key", "your-key", "粘贴")
    if any(marker in lowered for marker in placeholder_markers):
        return "API Key 输入框中仍是提示文字，请粘贴真实密钥"
    if key.startswith(("http://", "https://")):
        return "API Key 输入框里填入了网址；接口地址应填写在 Base URL"
    for part in key.split(","):
        if not part:
            return "API Key 格式为空"
        if any(ord(ch) < 33 or ord(ch) > 126 for ch in part):
            return "API Key 含中文、全角字符、空格或不可见字符，请重新粘贴纯 ASCII 密钥"
    return ""


def validate_api_key(value: str, *, required: bool = True) -> str:
    key = normalise_api_key(value)
    if not key and not required:
        return ""
    error = api_key_validation_error(key)
    if error:
        raise APIKeyValidationError(error)
    return key


def _config_dir() -> Path:
    override = os.environ.get("NOVEL_FORMATTER_CONFIG_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    if sys_platform() == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_DIR_NAME
    if os.name == "nt":
        return Path(os.environ.get("APPDATA", Path.home())) / APP_DIR_NAME
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "novel-formatter"


def sys_platform() -> str:
    import sys
    return sys.platform


CONFIG_PATH = _config_dir() / "ai_settings.json"
_SESSION_API_KEY = ""


@dataclass
class AISettings:
    provider: str = "openai"
    api_key: str = ""
    model: str = "gpt-4o"
    base_url: str = ""
    temperature: float = 0.2
    max_tokens: int = 24000
    concurrency: int = 0
    rpm_limit: int = 0
    tpm_limit: int = 0
    batch_chars: int = 0
    batch_tokens: int = 0
    request_timeout: int = 180
    json_mode: bool = True
    deepseek_thinking: bool = False
    deepseek_reasoning_effort: str = "high"
    deepseek_user_id: str = "novel_formatter"
    ocr_repair_mode: str = "readability"  # readability | strict
    performance_profile_version: int = 4

    @property
    def requires_key(self) -> bool:
        return self.provider not in {"ollama"}

    @property
    def configured(self) -> bool:
        if not self.provider or not self.model:
            return False
        if not self.requires_key:
            return True
        return bool(self.api_key and not api_key_validation_error(self.api_key))

    def provider_kwargs(self) -> dict:
        kwargs = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "concurrency": self.concurrency,
            "rpm_limit": self.rpm_limit,
            "tpm_limit": self.tpm_limit,
            "ai_batch_chars": self.batch_chars,
            "ai_batch_tokens": self.batch_tokens,
            "request_timeout": self.request_timeout,
            "json_mode": self.json_mode,
            "provider_name": self.provider,
            "deepseek_thinking": self.deepseek_thinking,
            "deepseek_reasoning_effort": self.deepseek_reasoning_effort,
            "deepseek_user_id": self.deepseek_user_id,
            "ocr_repair_mode": self.ocr_repair_mode,
        }
        if self.base_url.strip():
            kwargs["base_url"] = self.base_url.strip()
        return kwargs


def provider_defaults(provider: str) -> tuple[str, str]:
    return {
        "openai": ("gpt-4o", "https://api.openai.com/v1"),
        "anthropic": ("claude-sonnet-4-5", ""),
        "gemini": ("gemini-2.0-flash", ""),
        "deepseek": ("deepseek-v4-flash", "https://api.deepseek.com"),
        "openrouter": ("openai/gpt-4o", "https://openrouter.ai/api/v1"),
        "ollama": ("qwen3:8b", "http://127.0.0.1:11434/v1"),
        "custom": ("", ""),
    }.get(provider, ("", ""))


def set_session_api_key(api_key: str) -> None:
    """Keep a key in memory for this process only; never writes it to disk."""
    global _SESSION_API_KEY
    _SESSION_API_KEY = normalise_api_key(api_key)


def clear_session_api_key() -> None:
    set_session_api_key("")


def load_ai_settings() -> AISettings:
    data: dict = {}
    try:
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        data = {}

    provider = str(data.get("provider") or os.environ.get("NOVEL_FORMATTER_AI_PROVIDER") or "openai").lower()
    default_model, default_url = provider_defaults(provider)
    # Session-only key has priority. A persisted key is used only when the JSON
    # actually contains it. Environment variables remain the final fallback.
    persisted_key = str(data.get("api_key", "")) if "api_key" in data else ""
    api_key = _SESSION_API_KEY or persisted_key or os.environ.get("NOVEL_FORMATTER_AI_KEY", "")
    legacy_performance_defaults = (
        int(data.get("performance_profile_version", 0) or 0) < 2
        and int(data.get("concurrency", 4) or 0) == 4
        and int(data.get("rpm_limit", 60) or 0) == 60
        and "batch_chars" not in data
    )
    model = str(data.get("model") or os.environ.get("NOVEL_FORMATTER_AI_MODEL") or default_model)
    deepseek_thinking = bool(data.get("deepseek_thinking", False))
    # DeepSeek retired the legacy aliases in July 2026. Migrate transparently while
    # preserving whether the old alias represented thinking mode.
    base_url = str(data.get("base_url") or os.environ.get("NOVEL_FORMATTER_AI_BASE_URL") or default_url)
    max_tokens = int(data.get("max_tokens", os.environ.get("NOVEL_FORMATTER_AI_MAX_TOKENS", 24000)))
    if provider == "deepseek":
        lowered_model = model.strip().lower()
        if lowered_model == "deepseek-chat":
            model = "deepseek-v4-flash"
            deepseek_thinking = False
        elif lowered_model == "deepseek-reasoner":
            model = "deepseek-v4-flash"
            deepseek_thinking = True
        if base_url.rstrip("/") == "https://api.deepseek.com/v1":
            base_url = "https://api.deepseek.com"
        # V4 allows much larger output. Raising the ceiling prevents a changed-heavy
        # typesetting batch from being cut off; unused max_tokens are not billed.
        if int(data.get("performance_profile_version", 0) or 0) < 4 and max_tokens == 24000:
            max_tokens = 48000

    return AISettings(
        provider=provider,
        api_key=normalise_api_key(api_key),
        model=model,
        base_url=base_url,
        temperature=float(data.get("temperature", os.environ.get("NOVEL_FORMATTER_AI_TEMPERATURE", 0.2))),
        max_tokens=max_tokens,
        concurrency=0 if legacy_performance_defaults else int(data.get("concurrency", os.environ.get("NOVEL_FORMATTER_AI_CONCURRENCY", 0))),
        rpm_limit=0 if legacy_performance_defaults else int(data.get("rpm_limit", os.environ.get("NOVEL_FORMATTER_AI_RPM", 0))),
        tpm_limit=int(data.get("tpm_limit", os.environ.get("NOVEL_FORMATTER_AI_TPM", 0))),
        batch_chars=int(data.get("batch_chars", os.environ.get("NOVEL_FORMATTER_AI_BATCH_CHARS", 0))),
        batch_tokens=int(data.get("batch_tokens", os.environ.get("NOVEL_FORMATTER_AI_BATCH_TOKENS", 0))),
        request_timeout=int(data.get("request_timeout", os.environ.get("NOVEL_FORMATTER_AI_TIMEOUT", 180))),
        json_mode=bool(data.get("json_mode", True)),
        deepseek_thinking=deepseek_thinking,
        deepseek_reasoning_effort=str(data.get("deepseek_reasoning_effort", "high") or "high"),
        deepseek_user_id=str(data.get("deepseek_user_id", "novel_formatter") or "novel_formatter"),
        ocr_repair_mode=(str(data.get("ocr_repair_mode", "readability") or "readability") if str(data.get("ocr_repair_mode", "readability") or "readability") in {"readability", "strict"} else "readability"),
        performance_profile_version=4,
    )

def save_ai_settings(settings: AISettings, persist_api_key: bool = True) -> Path:
    """Save non-secret settings and optionally the API key.

    When persist_api_key is False, any previously stored key is removed from the
    JSON file and the supplied key is kept only in process memory. Saving an
    empty key always removes the persisted key and clears the session key.
    """
    payload = asdict(settings)
    key = normalise_api_key(payload.pop("api_key", ""))
    if key:
        key = validate_api_key(key)
    if persist_api_key and key:
        payload["api_key"] = key
        clear_session_api_key()
    elif key:
        set_session_api_key(key)
    else:
        clear_session_api_key()

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Same-directory temporary file + fsync + replace keeps the settings valid
    # even if the app or Mac is interrupted during save.
    atomic_write_json(CONFIG_PATH, payload, ensure_ascii=False, indent=2)
    try:
        CONFIG_PATH.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    return CONFIG_PATH
