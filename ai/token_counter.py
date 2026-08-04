# -*- coding: utf-8 -*-
"""Fast token estimation with optional tiktoken support.

The application must run without adding another mandatory dependency, so this module
uses tiktoken when it is already installed and falls back to a Japanese-aware estimate.
The fallback deliberately errs slightly high for CJK/kana text to keep API requests
inside their intended input budget.
"""
from __future__ import annotations

import functools
import re

_CJK_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]"
)
_ASCII_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


@functools.lru_cache(maxsize=32)
def _encoding_for_model(model: str):
    try:
        import tiktoken  # type: ignore
    except Exception:
        return None
    try:
        return tiktoken.encoding_for_model(model or "gpt-4o")
    except Exception:
        try:
            return tiktoken.get_encoding("o200k_base")
        except Exception:
            return None


def estimate_tokens(text: str, model: str = "") -> int:
    """Estimate tokens for batching/rate limiting without requiring tiktoken.

    Japanese kana/kanji are commonly close to one token per character on modern
    tokenizers. ASCII words are cheaper, while JSON punctuation and whitespace still
    carry some cost. The fallback therefore combines these groups rather than using a
    single characters-per-token ratio that badly underestimates Japanese OCR.
    """
    value = str(text or "")
    if not value:
        return 0
    encoding = _encoding_for_model(str(model or ""))
    if encoding is not None:
        try:
            return len(encoding.encode(value, disallowed_special=()))
        except Exception:
            pass

    cjk = len(_CJK_RE.findall(value))
    ascii_word_chars = sum(len(match.group(0)) for match in _ASCII_WORD_RE.finditer(value))
    remainder = max(0, len(value) - cjk - ascii_word_chars)
    # CJK ~= 1 token/char, ASCII words ~= 4 chars/token, punctuation/JSON ~= 2 chars/token.
    return max(1, cjk + (ascii_word_chars + 3) // 4 + (remainder + 1) // 2)
