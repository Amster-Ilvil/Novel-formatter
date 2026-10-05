# -*- coding: utf-8 -*-
"""Shared, low-token glossary support for AI proofreading workflows.

The glossary is deliberately advisory/protective rather than a translation engine.
Only terms actually present (or plausibly OCR-corrupted) in the current batch are
sent to the provider so a book-sized glossary does not get repeated on every API
request.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Sequence

_SPACE_RE = re.compile(r"[\s\u3000]+")


def _compact(value: str) -> str:
    return _SPACE_RE.sub("", unicodedata.normalize("NFKC", str(value or ""))).casefold()


def load_glossary(path: str | Path | None) -> list[tuple[str, str]]:
    raw_path = str(path or "").strip()
    if not raw_path:
        return []
    source = Path(raw_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"术语表不存在：{source}")

    pairs: list[tuple[str, str]] = []
    if source.suffix.lower() == ".json":
        data = json.loads(source.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):
            values = list(data.items())
        elif isinstance(data, list):
            values = []
            for item in data:
                if isinstance(item, dict):
                    values.append((
                        item.get("term") or item.get("source") or item.get("original"),
                        item.get("preferred") or item.get("target") or item.get("translation"),
                    ))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    values.append((item[0], item[1]))
        else:
            values = []
        for left, right in values:
            left, right = str(left or "").strip(), str(right or "").strip()
            if left and right:
                pairs.append((left, right))
    else:
        for raw in source.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"\s*(?:\t|=>|=|：|:)\s*", line, maxsplit=1)
            if len(parts) == 2 and parts[0] and parts[1]:
                pairs.append((parts[0], parts[1]))
            else:
                # A one-column whitelist is useful for coined names that must only
                # be preserved, not mapped to a different spelling.
                pairs.append((line, line))

    dedup: dict[str, str] = {}
    for left, right in pairs:
        dedup.setdefault(left, right)
    return list(dedup.items())


def glossary_fingerprint(path: str | Path | None, pairs: Sequence[tuple[str, str]] | None = None) -> str:
    raw_path = str(path or "").strip()
    if raw_path:
        source = Path(raw_path).expanduser()
        if source.is_file():
            return hashlib.sha256(source.read_bytes()).hexdigest()
    canonical = json.dumps(list(pairs or []), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest() if canonical != "[]" else ""


def select_relevant_glossary(
    glossary: Sequence[tuple[str, str]],
    texts: Iterable[str],
    *,
    max_entries: int = 48,
    max_chars: int = 4096,
) -> list[list[str]]:
    """Return a compact, deterministic subset relevant to this request.

    Exact source/target occurrences rank first.  A conservative two-character
    overlap admits likely OCR variants for longer terms, while one-character terms
    are never fuzzy-matched because that would flood every batch with noise.
    """
    if not glossary:
        return []
    haystack = _compact("".join(str(value or "") for value in texts))
    if not haystack:
        return []

    exact: list[tuple[str, str]] = []
    fuzzy: list[tuple[str, str]] = []
    for left, right in glossary:
        left_s, right_s = str(left or "").strip(), str(right or "").strip()
        if not left_s or not right_s:
            continue
        left_key, right_key = _compact(left_s), _compact(right_s)
        if (left_key and left_key in haystack) or (right_key and right_key in haystack):
            exact.append((left_s, right_s))
            continue
        # OCR corruption usually preserves at least a short contiguous fragment of
        # a longer proper noun. Avoid fuzzy matching short/common words.
        candidates = [key for key in (left_key, right_key) if len(key) >= 4]
        if any(any(key[i:i + 2] in haystack for i in range(len(key) - 1)) for key in candidates):
            fuzzy.append((left_s, right_s))

    result: list[list[str]] = []
    char_budget = 0
    for left, right in exact + fuzzy:
        cost = len(left) + len(right) + 8
        if result and (len(result) >= max_entries or char_budget + cost > max_chars):
            break
        result.append([left, right])
        char_budget += cost
    return result
