#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Helpers for keeping OCR output valid UTF-8.

Apple Vision can occasionally surface an isolated UTF-16 surrogate in a
candidate string. Python can hold that value internally, but writing it to an
UTF-8 HTML/CSV/JSON file raises ``UnicodeEncodeError``. These helpers remove
only invalid surrogate code points and preserve normal Japanese combining marks
and variation selectors.
"""
from __future__ import annotations

import json
import unicodedata
from typing import Any


def clean_text(value: Any) -> str:
    text = str(value or "")
    text = "".join(ch for ch in text if not (0xD800 <= ord(ch) <= 0xDFFF))
    try:
        text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    except Exception:
        pass
    try:
        return unicodedata.normalize("NFC", text)
    except Exception:
        return text


def clean_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, dict):
        return {clean_text(key): clean_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json_value(item) for item in value]
    return value


def dumps(value: Any, *, indent: int | None = None, ensure_ascii: bool = False) -> str:
    return json.dumps(clean_json_value(value), ensure_ascii=ensure_ascii, indent=indent)
