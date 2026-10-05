#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publication-safe Unicode normalization for Japanese OCR text.

Two policies are intentionally separated:

``publication`` (default)
    Safe for authoritative/book text.  Compose kana/combining marks, widen
    half-width Japanese kana/punctuation, normalize vertical presentation forms,
    and drop illegal/invisible control garbage.  CJK radicals, compatibility
    ideographs and IVS remain byte-for-byte intact.

``compatibility``
    Reader/search compatibility mode.  In addition to the publication rules,
    Kangxi/CJK radical code points that have a standard NFKC ideograph mapping
    are rewritten to that ideograph, and obvious kana-internal dash aliases are
    normalized to the prolonged-sound mark.

The previous formatter always rewrote radicals.  That improved searchability but
could silently erase a deliberately chosen historical glyph.  Making the policy
explicit keeps publication output conservative while still offering the older
high-compatibility behaviour when requested.
"""
from __future__ import annotations

import re
import unicodedata

from engine.ocr_unicode_standardizer import normalize_japanese_ocr_text

_RADICAL_CHAR = re.compile(r'[\u2E80-\u2EF3\u2F00-\u2FD5]')
_CONTROL_CHARS = re.compile(
    r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F\u200B\u200E\u200F\uFEFF]'
)
_KANA = r'[ぁ-ゟァ-ヺー]'
_DASH_BETWEEN_KANA = re.compile(rf'(?<={_KANA})[‐‑‒–−─](?={_KANA})')


def _compatibility_radicals(text: str) -> tuple[str, int]:
    changed = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal changed
        source = match.group(0)
        target = unicodedata.normalize("NFKC", source)
        if target != source:
            changed += 1
            return target
        return source

    return _RADICAL_CHAR.sub(replace, text), changed


def normalize_ocr_codepoints(
    text: str,
    *,
    policy: str = "publication",
) -> tuple[str, dict[str, int]]:
    """Normalize OCR code points according to a non-semantic policy.

    ``policy`` accepts ``publication``/``conservative`` and
    ``compatibility``/``reader``.  Unknown values deliberately fall back to the
    conservative publication policy rather than performing a destructive NFKC.
    """
    counts: dict[str, int] = {}
    if not text:
        return text, counts

    normalized, report = normalize_japanese_ocr_text(str(text))
    for key, value in report.counts.items():
        if value:
            counts[key] = counts.get(key, 0) + int(value)
    text = normalized

    cleaned = _CONTROL_CHARS.sub("", text)
    if cleaned != text:
        counts["control_char"] = len(text) - len(cleaned)
        text = cleaned

    mode = str(policy or "publication").strip().lower()
    if mode in {"compatibility", "compatible", "reader", "high_compat"}:
        text, radical_count = _compatibility_radicals(text)
        if radical_count:
            counts["kangxi_radical"] = radical_count
        dashed = _DASH_BETWEEN_KANA.sub("ー", text)
        if dashed != text:
            counts["dash_variant"] = len(_DASH_BETWEEN_KANA.findall(text))
            text = dashed

    return text, counts
