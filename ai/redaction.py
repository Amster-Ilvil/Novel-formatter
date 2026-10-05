# -*- coding: utf-8 -*-
"""Small, dependency-free redaction helpers for AI diagnostics.

The formatter intentionally keeps OCR/book text in review artefacts where that text is
required evidence.  This module therefore redacts *credentials*, not arbitrary prose.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

_REDACTED = "<REDACTED>"

# Query strings and common header renderings used by providers/proxies.
_QUERY_SECRET = re.compile(
    r"(?i)([?&](?:key|api[_-]?key|token|access[_-]?token|credential|signature|sig)=)[^&\s]+"
)
_HEADER_SECRET = re.compile(
    r"(?i)((?:authorization\s*[:=]\s*(?:bearer\s+)?)|(?:x-api-key|api-key)\s*[:=]\s*)"
    r"([^\s,;\]}]+)"
)
_BEARER = re.compile(r"(?i)(\bbearer\s+)([^\s,;\]}]+)")

# Common public API key shapes.  Keep the minimum length high to avoid masking normal
# Japanese/English text that happens to start with the same two letters.
_KEY_SHAPES = (
    re.compile(r"\bsk-(?:ant-|or-v1-)?[A-Za-z0-9_-]{16,}\b", re.I),
    re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\b(?:gsk_|hf_)[A-Za-z0-9_-]{16,}\b", re.I),
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
    text = _QUERY_SECRET.sub(lambda m: m.group(1) + _REDACTED, text)
    text = _HEADER_SECRET.sub(lambda m: m.group(1) + _REDACTED, text)
    text = _BEARER.sub(lambda m: m.group(1) + _REDACTED, text)
    for pattern in _KEY_SHAPES:
        text = pattern.sub(_REDACTED, text)
    return text
