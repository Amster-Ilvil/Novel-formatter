# -*- coding: utf-8 -*-
"""Small, dependency-free redaction helpers for AI diagnostics.

The formatter intentionally keeps OCR/book text in review artefacts where that text is
required evidence.  This module therefore redacts *credentials*, not arbitrary prose.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

_REDACTED = "<REDACTED>"

_SECRET_NAMES = (
    r"authorization|proxy[-_ ]?authorization|"
    r"(?:(?:x[-_ ]?(?:goog[-_ ]?)?|(?:gemini|google|openai|anthropic|deepseek)[-_ ]+))?api[-_ ]?key|"
    r"(?:access|refresh|id|oauth|session)[-_ ]?token|token|credential(?:s)?|"
    r"(?:client[-_ ]?)?secret|password|passwd|cookie|set[-_ ]?cookie|private[-_ ]?key"
)
_SECRET_FIELD = re.compile(rf"(?:{_SECRET_NAMES})", re.I)
_QUERY_SECRET = re.compile(
    rf"(?i)([?&](?:{_SECRET_NAMES}|key|signature|sig|"
    r"x-(?:amz|goog)-(?:signature|credential|security-token))=)[^&\s\"'<>#]+"
)
_ASSIGNED_SECRET = re.compile(
    rf"""(?i)(\b(?:{_SECRET_NAMES})["']?\s*[:=][ \t]*)"""
    r"""(?:"(?:\\.|[^"\\\r\n])*"|'(?:\\.|[^'\\\r\n])*'|"""
    r"""(?:(?:bearer|basic)[ \t]+)?[^\s,;\]}<>"']+)"""
)
_COOKIE_HEADER = re.compile(r"(?im)^([ \t]*(?:set-cookie|cookie)[ \t]*:[ \t]*)[^\r\n]+")
_BEARER = re.compile(r"""(?i)(\bbearer[ \t]+)([^\s,;\]}<>"']+)""")
_URL_USERINFO = re.compile(r"(?i)(\bhttps?://)[^/?#\s@<>]+@")

# Common public API key shapes.  Keep the minimum length high to avoid masking normal
# Japanese/English text that happens to start with the same two letters.
_KEY_SHAPES = (
    re.compile(r"\bsk-(?:ant-|or-v1-)?[A-Za-z0-9_-]{16,}\b", re.I),
    re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\b(?:gsk_|hf_)[A-Za-z0-9_-]{16,}\b", re.I),
    re.compile(r"\b(?:gh[pousr]_|github_pat_|xox[baprs]-)[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"(?<![\w.-])(?:AQ\.|ya29\.|1//)[A-Za-z0-9_.-]{20,}"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
)


def redact_secrets(value: object, *, secrets: Iterable[str] = ()) -> str:
    """Return diagnostic text with credentials removed.

    ``secrets`` lets callers provide the exact active credential as an extra defence;
    it is replaced literally before regex-based provider-agnostic patterns run.
    """
    text = str(value or "")
    exact = sorted(
        {str(secret) for secret in secrets if str(secret or "")},
        key=len,
        reverse=True,
    )
    for secret in exact:
        text = text.replace(secret, _REDACTED)
    text = _COOKIE_HEADER.sub(lambda m: m.group(1) + _REDACTED, text)
    text = _QUERY_SECRET.sub(lambda m: m.group(1) + _REDACTED, text)
    text = _ASSIGNED_SECRET.sub(lambda m: m.group(1) + _REDACTED, text)
    text = _BEARER.sub(lambda m: m.group(1) + _REDACTED, text)
    text = _URL_USERINFO.sub(lambda m: m.group(1) + _REDACTED + "@", text)
    for pattern in _KEY_SHAPES:
        text = pattern.sub(_REDACTED, text)
    return text


def redact_json_credentials(value: object) -> object:
    """Redact named credential fields and credential strings within JSON data."""
    if isinstance(value, Mapping):
        return {
            redact_secrets(key) if isinstance(key, str) else key: (
                _REDACTED if item and _SECRET_FIELD.fullmatch(str(key))
                else redact_json_credentials(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_json_credentials(item) for item in value]
    if isinstance(value, str):
        return redact_secrets(value)
    return value
