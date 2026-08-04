# -*- coding: utf-8 -*-
from __future__ import annotations

import concurrent.futures
import copy
import difflib
import hashlib
import json
import os
import math
import re
import threading
import time
import uuid
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ai.request_limiter import RequestLimiter
from ai.token_counter import estimate_tokens
from engine.document_versions import ai_request_payload, build_ai_document, cleanup_ai_covered_fragments

# Protocol v3 follows the same efficiency principle as AiNiee: ordered numbered text,
# minimal repeated metadata, token-aware batching, and local reuse of unchanged content.
TYPESET_PROMPT = """Correct and typeset Japanese light-novel OCR without changing its style or facts.
Input is one ordered chapter part: {"b":[[id,type,text],...],"g":0|1}. Types: p=paragraph,d=dialogue,c=chapter,s=section,r=ruby,f=footnote,t=toc,u=other.
Fix clear OCR errors, missing characters, punctuation and grammar. Merge/split blocks when needed, repair cross-page sentences, put each dialogue turn in its own block, and separate dialogue from narration. Preserve chapter/section order and apply consistent light-novel formatting across batches.
Return ONLY changed contiguous ranges as compact JSON: {"o":[[[source_ids...],[[type,text],...]],...]}. Each operation replaces exactly those contiguous input ids; operations must not overlap. Omit every unchanged range.
When g=1, also return one complete EPUB stylesheet in top-level key "s". The CSS must be plain CSS (no Markdown), suitable for Japanese light novels, and style body, h1, h2, p.normal, p.dialogue, ruby, rt, .cover-page, .illus-page and img. It must preserve readable spacing in both vertical and horizontal export; writing-mode may be included because the exporter removes it for horizontal mode. When g=0, omit "s".
If nothing changes return {"o":[],"s":"..."} when g=1, otherwise {"o":[]}.
Never invent, summarize, translate, rename, delete content, or return explanations/Markdown. Every replacement text must be non-empty.
INPUT:\n{{INPUT}}"""

CORRECTION_PROMPT = """Proofread Japanese light-novel OCR while preserving wording, style, block order, block count and block type.
Input: {"b":[[id,type,text],...]}. Correct only clear OCR character errors, obvious missing characters, punctuation and grammar. Do not merge, split or reformat.
Return ONLY changed blocks as compact JSON: {"c":[[id,corrected_text],...]}. Omit unchanged blocks. If nothing changes return {"c":[]}.
No explanations, unchanged text or Markdown.
INPUT:\n{{INPUT}}"""

_RETRY_SUFFIX = """
Your response was not valid JSON. Return exactly one compact JSON object matching the schema, with no Markdown or commentary.
"""

AI_TYPESET_FALLBACK_CSS = """html {
    writing-mode: vertical-rl;
    -epub-writing-mode: vertical-rl;
}
body {
    margin: 5%;
    line-height: 1.9;
    font-family: serif;
    color: #222;
}
h1 {
    font-size: 1.4em;
    margin: 1.5em 0 1em;
    break-before: page;
}
h2 {
    font-size: 1.15em;
    margin: 1.2em 0 0.8em;
}
p.normal {
    margin: 0 0 0.55em;
    text-indent: 1em;
}
p.dialogue {
    margin: 0 0 0.55em;
    text-indent: 0;
}
ruby { ruby-position: over; }
rt { font-size: 0.55em; }
.cover-page, .illus-page {
    text-align: center;
    page-break-after: always;
}
.cover-page img, .illus-page img, img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
}
"""

_TYPE_TO_CODE = {
    "paragraph": "p",
    "dialogue": "d",
    "chapter": "c",
    "section": "s",
    "ruby": "r",
    "footnote": "f",
    "toc_entry": "t",
}
_CODE_TO_TYPE = {value: key for key, value in _TYPE_TO_CODE.items()}
_CODE_TO_TYPE["u"] = "paragraph"


class AIReplyFormatError(ValueError):
    """The API returned content, but it was not a complete parseable JSON reply."""

    def __init__(self, message: str, *, truncated: bool = False):
        super().__init__(message)
        self.truncated = bool(truncated)


class AIRequestError(ValueError):
    """The provider request failed after bounded transient retries."""


@dataclass
class _UsageTracker:
    requests: int = 0
    api_retries: int = 0
    format_errors: int = 0
    split_events: int = 0
    singleton_fallbacks: int = 0
    estimated_prompt_tokens: int = 0
    estimated_completion_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def request(self, prompt: str, model: str = "", retry: bool = False) -> None:
        with self._lock:
            self.requests += 1
            if retry:
                self.api_retries += 1
            self.estimated_prompt_tokens += estimate_tokens(prompt, model)

    def response(self, reply, model: str = "") -> None:
        if isinstance(reply, (dict, list)):
            text = json.dumps(reply, ensure_ascii=False, separators=(",", ":"))
        else:
            text = str(reply or "")
        with self._lock:
            self.estimated_completion_tokens += estimate_tokens(text, model)

    def format_error(self) -> None:
        with self._lock:
            self.format_errors += 1

    def split(self) -> None:
        with self._lock:
            self.split_events += 1

    def singleton_fallback(self) -> None:
        with self._lock:
            self.singleton_fallbacks += 1

    def snapshot(self, provider=None) -> dict:
        with self._lock:
            data = {
                "requests": self.requests,
                "api_retries": self.api_retries,
                "format_errors": self.format_errors,
                "split_events": self.split_events,
                "singleton_fallbacks": self.singleton_fallbacks,
                "estimated_prompt_tokens": self.estimated_prompt_tokens,
                "estimated_completion_tokens": self.estimated_completion_tokens,
            }
        provider_usage = {}
        getter = getattr(provider, "usage_snapshot", None)
        if callable(getter):
            try:
                provider_usage = getter() or {}
            except Exception:
                provider_usage = {}
        actual_total = int(provider_usage.get("total_tokens", 0) or 0)
        estimated_total = data["estimated_prompt_tokens"] + data["estimated_completion_tokens"]
        data.update({
            "provider_requests": int(provider_usage.get("requests", 0) or 0),
            "prompt_tokens": int(provider_usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(provider_usage.get("completion_tokens", 0) or 0),
            "total_tokens": actual_total,
            "cached_tokens": int(provider_usage.get("cached_tokens", 0) or 0),
            "cache_miss_tokens": int(provider_usage.get("cache_miss_tokens", 0) or 0),
            "reasoning_tokens": int(provider_usage.get("reasoning_tokens", 0) or 0),
            "display_total_tokens": actual_total or estimated_total,
            "usage_is_actual": bool(actual_total),
        })
        return data


def _balanced_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _light_json_cleanup(text: str) -> str:
    text = text.lstrip("\ufeff").strip()
    return re.sub(r",\s*([}\]])", r"\1", text)


def parse_json_reply(text) -> dict:
    if isinstance(text, dict):
        return text
    if isinstance(text, list):
        # Legacy providers sometimes return the requested full blocks array directly.
        return {"blocks": text}
    raw = str(text or "").strip()
    if not raw:
        raise AIReplyFormatError("AI 返回了空内容。请检查模型、API 配额或接口兼容性。")
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S | re.I)
    candidates = [fence.group(1).strip()] if fence else []
    candidates.append(raw)
    balanced = _balanced_json_object(raw)
    if balanced:
        candidates.append(balanced)
    seen: set[str] = set()
    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        candidate = _light_json_cleanup(candidate)
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                return {"blocks": data}
            if not isinstance(data, dict):
                raise AIReplyFormatError("AI JSON 顶层必须是对象或数组")
            return data
        except json.JSONDecodeError as exc:
            last_error = exc
    if last_error:
        near = raw[max(0, last_error.pos - 80):last_error.pos + 80].replace("\n", "\\n")
        truncated = _balanced_json_object(raw) is None
        hint = "，响应可能被截断" if truncated else ""
        raise AIReplyFormatError(
            f"AI 返回的 JSON 无效{hint}：第 {last_error.lineno} 行、第 {last_error.colno} 列；错误附近：{near!r}",
            truncated=truncated,
        ) from last_error
    raise AIReplyFormatError("AI 返回内容中没有找到 JSON 对象。")


def _type_code(value) -> str:
    return _TYPE_TO_CODE.get(str(getattr(value, "value", value) or "paragraph"), "u")


def _type_name(value, fallback: str = "paragraph") -> str:
    raw = str(getattr(value, "value", value) or "").strip().lower()
    if raw == "u":
        return str(fallback or "paragraph")
    if raw in _CODE_TO_TYPE:
        return _CODE_TO_TYPE[raw]
    if raw in _TYPE_TO_CODE:
        return raw
    return str(fallback or "paragraph")


def _wire_row(alias: str, block: dict) -> list[str]:
    return [alias, _type_code(block.get("type", "paragraph")), str(block.get("text", ""))]


def _batch_blocks(
    blocks: list[dict],
    max_chars: int = 0,
    max_tokens: int = 0,
    prompt_template: str = "",
    model: str = "",
) -> Iterable[list[dict]]:
    """Batch ordered blocks by actual serialized prompt-token cost.

    ``max_chars`` remains a compatible optional source-text ceiling. Auto mode primarily
    uses ``max_tokens`` so hundreds of short paragraphs cannot hide large JSON overhead.
    """
    max_chars = max(0, int(max_chars or 0))
    max_tokens = max(0, int(max_tokens or 0))
    empty_payload = json.dumps({"b": []}, ensure_ascii=False, separators=(",", ":"))
    base_prompt = (prompt_template or "{{INPUT}}").replace("{{INPUT}}", empty_payload)
    base_tokens = estimate_tokens(base_prompt, model)

    batch: list[dict] = []
    source_chars = 0
    payload_tokens = 0
    for block in blocks:
        alias = f"b{len(batch) + 1}"
        row_json = json.dumps(_wire_row(alias, block), ensure_ascii=False, separators=(",", ":"))
        row_tokens = estimate_tokens(row_json, model) + 1
        text_chars = len(str(block.get("text", "")))
        exceeds_chars = bool(batch and max_chars and source_chars + text_chars > max_chars)
        exceeds_tokens = bool(batch and max_tokens and base_tokens + payload_tokens + row_tokens > max_tokens)
        if exceeds_chars or exceeds_tokens:
            yield batch
            batch = []
            source_chars = 0
            payload_tokens = 0
            alias = "b1"
            row_json = json.dumps(_wire_row(alias, block), ensure_ascii=False, separators=(",", ":"))
            row_tokens = estimate_tokens(row_json, model) + 1
        batch.append(block)
        source_chars += text_chars
        payload_tokens += row_tokens
    if batch:
        yield batch


def _chapter_batches(
    payload: dict,
    max_chars: int = 0,
    max_tokens: int = 16000,
    prompt_template: str = TYPESET_PROMPT,
    model: str = "",
) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    order: list[str] = []
    for block in payload.get("blocks", []):
        chapter_id = str(block.get("chapter_id") or "chapter_001")
        if chapter_id not in grouped:
            grouped[chapter_id] = []
            order.append(chapter_id)
        # Keep all page/image/document metadata local. Order is implicit in the list.
        grouped[chapter_id].append({
            "id": str(block.get("id", "")),
            "type": str(block.get("type", "paragraph")),
            "text": str(block.get("text", "")),
        })
    result: list[dict] = []
    for chapter_id in order:
        parts = _batch_blocks(
            grouped[chapter_id],
            max_chars=max_chars,
            max_tokens=max_tokens,
            prompt_template=prompt_template,
            model=model,
        )
        for part_no, blocks in enumerate(parts, 1):
            result.append({
                "target_chapter_id": chapter_id,
                "chapter_part": part_no,
                "blocks": blocks,
            })
    return result


def _wire_batch(batch: dict) -> tuple[dict, dict[str, str]]:
    """Create the token-light AiNiee-style wire payload and retain UUIDs locally."""
    aliases: dict[str, str] = {}
    rows: list[list[str]] = []
    for index, block in enumerate(batch.get("blocks", []), 1):
        alias = f"b{index}"
        aliases[alias] = str(block.get("id", ""))
        rows.append(_wire_row(alias, block))
    payload = {"b": rows}
    if batch.get("request_css"):
        payload["g"] = 1
    return payload, aliases


def _extract_ai_css(parsed: dict) -> str:
    """Extract and lightly validate a stylesheet returned with a typeset batch."""
    if not isinstance(parsed, dict):
        return ""
    parsed = _unwrap_compact_payload(parsed)
    value = parsed.get("s")
    if value is None:
        value = parsed.get("css")
    if isinstance(value, dict):
        value = value.get("content") or value.get("stylesheet") or value.get("css")
    css = str(value or "").strip()
    css = re.sub(r"^```(?:css)?\s*", "", css, flags=re.IGNORECASE)
    css = re.sub(r"\s*```$", "", css).strip()
    if len(css) < 40 or len(css) > 100_000:
        return ""
    if "{" not in css or "}" not in css:
        return ""
    # A stylesheet must never smuggle executable markup into the EPUB.
    if re.search(r"<\s*/?\s*(?:script|style|html|body)\b", css, flags=re.IGNORECASE):
        return ""
    if re.search(r"(?:javascript\s*:|expression\s*\(|behavior\s*:|@import\b)", css, flags=re.IGNORECASE):
        return ""
    return css


def _is_transient_api_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status in (408, 409, 425, 429) or (isinstance(status, int) and status >= 500):
        return True
    message = str(exc).lower()
    return any(marker in message for marker in (
        "timeout", "timed out", "temporarily unavailable", "connection reset",
        "connection aborted", "connection error", "rate limit", "too many requests",
        "service unavailable", "bad gateway", "gateway timeout",
    ))


def _retry_delay(exc: Exception, attempt: int) -> float:
    """Respect provider Retry-After headers, otherwise use bounded backoff."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or getattr(exc, "headers", None) or {}
    retry_after = None
    try:
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
    except Exception:
        retry_after = None
    if retry_after is not None:
        try:
            return max(0.25, min(float(retry_after), 30.0))
        except (TypeError, ValueError):
            pass
    return min(0.75 * (2 ** max(0, attempt)), 6.0)


def _call_and_parse(
    provider,
    prompt: str,
    temperature: float,
    api_retries: int = 1,
    cancel_check=None,
    limiter: RequestLimiter | None = None,
    usage: _UsageTracker | None = None,
) -> dict:
    """Retry transient API failures only; never repeat a large malformed response."""
    last_error: Exception | None = None
    model = str(getattr(provider, "model", "") or "")
    for attempt in range(max(0, int(api_retries)) + 1):
        if cancel_check and cancel_check():
            raise RuntimeError("AI任务已停止")
        if limiter:
            limiter.acquire(estimate_tokens(prompt, model), cancel_check=cancel_check)
        if usage:
            usage.request(prompt, model, retry=attempt > 0)
        try:
            caller = getattr(provider, "call_json", None) or provider._call_llm
            reply = caller(prompt, temperature)
        except Exception as exc:
            if getattr(exc, "output_truncated", False):
                partial = str(getattr(exc, "partial_content", "") or "")
                if usage:
                    if partial:
                        usage.response(partial, model)
                    usage.format_error()
                raise AIReplyFormatError(
                    "AI 输出达到 max_tokens，已直接拆分当前批次。",
                    truncated=True,
                ) from exc
            if isinstance(exc, RuntimeError):
                raise
            if isinstance(exc, UnicodeEncodeError) or "ascii codec can't encode" in str(exc).lower():
                raise AIRequestError(
                    "AI 请求头包含中文、全角字符或不可见字符。请清空 API Key 输入框后，"
                    "重新粘贴服务商提供的纯 ASCII 密钥；不要填写“请输入密钥”、Bearer 前缀或接口网址。"
                ) from exc
            last_error = exc
            if attempt < api_retries and _is_transient_api_error(exc):
                time.sleep(_retry_delay(exc, attempt))
                continue
            raise AIRequestError(f"AI API 请求失败：{exc}") from exc
        if cancel_check and cancel_check():
            raise RuntimeError("AI任务已停止")
        if usage:
            usage.response(reply, model)
        try:
            return parse_json_reply(reply)
        except AIReplyFormatError:
            if usage:
                usage.format_error()
            # A parse failure after a generated response may already be billed. Returning
            # immediately lets the caller split once instead of paying for the same large
            # prompt and output a second time.
            raise
    raise AIRequestError(f"AI API 请求失败：{last_error}") from last_error


def _unwrap_compact_payload(parsed: dict) -> dict:
    """Unwrap common gateway/model envelopes without accepting arbitrary prose."""
    current = parsed
    for _ in range(4):
        if not isinstance(current, dict):
            break
        if any(key in current for key in ("o", "c", "ops", "blocks", "changes", "chapters")):
            return current
        next_value = None
        for key in ("result", "data", "output", "response", "content"):
            value = current.get(key)
            if isinstance(value, dict):
                next_value = value
                break
            if isinstance(value, list):
                return {"blocks": value}
            if isinstance(value, str) and "{" in value:
                try:
                    return parse_json_reply(value)
                except ValueError:
                    pass
        if next_value is None:
            break
        current = next_value
    return current if isinstance(current, dict) else parsed


def _resolve_source_id(value, source_by_id: dict[str, dict], aliases: dict[str, str]) -> str:
    candidate = str(value or "")
    if candidate in source_by_id:
        return candidate
    return aliases.get(candidate, "")


def _extract_correction_changes(parsed: dict) -> tuple[list, bool]:
    parsed = _unwrap_compact_payload(parsed)
    if "c" in parsed:
        return parsed.get("c") if isinstance(parsed.get("c"), list) else [], False
    changes = parsed.get("changes")
    if isinstance(changes, list):
        return changes, True
    blocks = parsed.get("blocks")
    if isinstance(blocks, list):
        return blocks, True
    return [], True


def _extract_typeset_ops(parsed: dict) -> tuple[list, bool]:
    parsed = _unwrap_compact_payload(parsed)
    if "o" in parsed:
        return parsed.get("o") if isinstance(parsed.get("o"), list) else [], True
    if "ops" in parsed:
        return parsed.get("ops") if isinstance(parsed.get("ops"), list) else [], True
    return [], False


def _normalise_output_blocks(value, fallback_type: str) -> list[dict]:
    if isinstance(value, str):
        value = [[fallback_type, value]]
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []
    # Common near-miss: ["d", "text"] instead of [["d", "text"]].
    if len(value) >= 2 and isinstance(value[0], str) and isinstance(value[1], str):
        first = value[0].strip().lower()
        if first in _CODE_TO_TYPE or first in _TYPE_TO_CODE:
            value = [value]
    output: list[dict] = []
    for item in value:
        block_type = fallback_type
        text = ""
        if isinstance(item, (list, tuple)):
            if len(item) >= 2:
                block_type, text = item[0], item[1]
            elif len(item) == 1:
                text = item[0]
        elif isinstance(item, dict):
            block_type = item.get("type", fallback_type)
            text = item.get("text", "")
        elif isinstance(item, str):
            text = item
        text = str(text or "").strip()
        if not text:
            continue
        output.append({"type": _type_name(block_type, fallback_type), "text": text})
    return output


def _compact_guard_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text or "")
    value = re.sub(r"[\s　]+", "", value)
    return value.translate(str.maketrans({
        "『": "「", "』": "」", "“": "「", "”": "」",
        "—": "ー", "―": "ー", "−": "ー", "ｰ": "ー",
    }))


_PROTOCOL_ONLY_RE = re.compile(r"^(?:[pcdsrftu]|b\d{1,6})$", re.I)


def _looks_like_protocol_leak(text: str, aliases: dict[str, str] | None = None) -> bool:
    """Return True for compact-protocol aliases accidentally emitted as prose."""
    value = str(text or "").strip()
    if not value or "\n" in value or "\r" in value:
        return False
    if _PROTOCOL_ONLY_RE.fullmatch(value):
        return True
    return bool(aliases and value in aliases)


def _dynamic_member_coverage_minimum(length: int) -> float:
    if length >= 80:
        return 0.44
    if length >= 40:
        return 0.40
    if length >= 12:
        return 0.34
    if length >= 4:
        return 0.22
    return 0.0


def _correction_replacement_is_safe(source_text: str, output_text: str, aliases: dict[str, str] | None = None) -> tuple[bool, str]:
    """Accept only genuinely local OCR corrections.

    Correction mode is a differential protocol: a returned value must be a full
    replacement for one source block, not a changed fragment, summary, or explanation.
    Unsafe replies are ignored and the original block is retained.  This turns model
    mistakes into recoverable no-ops instead of failing the whole novel after all API
    calls have completed.
    """
    source = _compact_guard_text(source_text)
    output = _compact_guard_text(output_text)
    if not source or not output:
        return False, "empty"
    if _looks_like_protocol_leak(output_text, aliases):
        return False, "protocol_leak"
    if source == output:
        return True, ""

    source_len = len(source)
    output_len = len(output)
    ratio = output_len / max(source_len, 1)

    # OCR proofreading should be character-local.  Be permissive for tiny captions and
    # dialogue, but never accept a long paragraph collapsed into a sentence/fragment.
    if source_len >= 160 and not (0.80 <= ratio <= 1.22):
        return False, "length"
    if 40 <= source_len < 160 and not (0.70 <= ratio <= 1.32):
        return False, "length"
    if 12 <= source_len < 40 and not (0.50 <= ratio <= 1.65):
        return False, "length"
    if 4 <= source_len < 12 and not (0.35 <= ratio <= 2.20):
        return False, "length"
    if source_len < 4 and output_len > max(6, source_len * 3):
        return False, "length"

    if source_len >= 4:
        matcher = difflib.SequenceMatcher(None, source, output, autojunk=False)
        common = sum(m.size for m in matcher.get_matching_blocks() if m.size)
        source_coverage = common / max(source_len, 1)
        output_coverage = common / max(output_len, 1)
        if source_len >= 80:
            minimum = 0.62
        elif source_len >= 40:
            minimum = 0.54
        elif source_len >= 12:
            minimum = 0.42
        else:
            minimum = 0.25
        if source_coverage < minimum or output_coverage < min(0.50, minimum):
            return False, "low_coverage"

    # A common model/protocol mistake is returning only the corrected fragment.
    if source_len >= 40 and output in source and output_len < source_len * 0.88:
        return False, "fragment_only"
    return True, ""


def _typeset_replacement_is_safe(
    ordered_ids: list[str],
    replacement: list[dict],
    source_by_id: dict[str, dict],
    aliases: dict[str, str] | None = None,
) -> tuple[bool, str]:
    """Reject local AI operations that summarize or silently delete source prose."""
    if any(_looks_like_protocol_leak(str(x.get("text", "")), aliases) for x in replacement):
        return False, "protocol_leak"
    source_parts = [_compact_guard_text(str(source_by_id[sid].get("text", ""))) for sid in ordered_ids]
    output_text = _compact_guard_text("".join(str(x.get("text", "")) for x in replacement))
    source_text = "".join(source_parts)
    if not source_text or not output_text:
        return False, "empty"

    source_len = len(source_text)
    output_len = len(output_text)
    if source_len >= 160:
        if output_len < source_len * 0.76 or output_len > source_len * 1.35:
            return False, "length"
    elif source_len >= 40:
        if output_len < source_len * 0.68 or output_len > source_len * 1.45:
            return False, "length"
    elif source_len >= 12:
        if output_len < source_len * 0.45 or output_len > source_len * 1.90:
            return False, "length"
    elif source_len >= 4:
        if output_len < source_len * 0.28 or output_len > source_len * 2.50:
            return False, "length"
    elif output_len > max(6, source_len * 3):
        return False, "length"

    if source_len >= 4:
        matcher = difflib.SequenceMatcher(None, source_text, output_text, autojunk=False)
        common = sum(m.size for m in matcher.get_matching_blocks() if m.size)
        if source_len >= 40:
            minimum_total = 0.55
        elif source_len >= 12:
            minimum_total = 0.40
        else:
            minimum_total = 0.24
        if common / max(source_len, 1) < minimum_total:
            return False, "low_source_coverage"

    # A large merged operation must still retain every substantial source block.
    # This catches one paragraph disappearing inside an otherwise long, plausible reply.
    for part in source_parts:
        minimum = _dynamic_member_coverage_minimum(len(part))
        if minimum <= 0:
            continue
        matcher = difflib.SequenceMatcher(None, part, output_text, autojunk=False)
        common = sum(m.size for m in matcher.get_matching_blocks() if m.size)
        if common / max(len(part), 1) < minimum:
            return False, "missing_member_block"
    return True, ""


def _apply_typeset_ops(
    ops: list,
    batch: dict,
    aliases: dict[str, str],
    recovery: dict[str, int],
) -> list[dict]:
    source_blocks = list(batch.get("blocks", []))
    source_by_id = {str(b.get("id")): b for b in source_blocks}
    source_rank = {str(b.get("id")): i for i, b in enumerate(source_blocks)}

    def bump(key: str, amount: int = 1) -> None:
        recovery[key] = int(recovery.get(key, 0)) + amount

    normalized: list[dict] = []
    occupied: set[int] = set()
    for item in ops:
        ids = None
        outputs = None
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            ids, outputs = item[0], item[1]
            # Common singleton shorthand: ["b1", "p", "corrected text"].
            if len(item) >= 3 and isinstance(item[0], str) and isinstance(item[1], str):
                maybe_type = item[1].strip().lower()
                if maybe_type in _CODE_TO_TYPE or maybe_type in _TYPE_TO_CODE:
                    ids, outputs = [item[0]], [[item[1], item[2]]]
        elif isinstance(item, dict):
            ids = item.get("ids") or item.get("source_block_ids") or item.get("source_ids")
            outputs = item.get("blocks") if "blocks" in item else item.get("output")
            if outputs is None and "text" in item:
                outputs = [{"type": item.get("type"), "text": item.get("text")}]
        if isinstance(ids, str):
            ids = [ids]
        if not isinstance(ids, list):
            bump("malformed_ops_ignored")
            continue
        resolved = [
            _resolve_source_id(value, source_by_id, aliases)
            for value in ids
        ]
        resolved = list(dict.fromkeys(x for x in resolved if x in source_by_id))
        if not resolved:
            bump("unknown_output_blocks_ignored")
            continue
        ranks = sorted(source_rank[x] for x in resolved)
        if ranks != list(range(ranks[0], ranks[-1] + 1)):
            bump("noncontiguous_merges_recovered")
            continue
        rank_set = set(ranks)
        if occupied.intersection(rank_set):
            bump("overlapping_ops_ignored")
            continue
        ordered_ids = [str(source_blocks[rank].get("id")) for rank in ranks]
        fallback_type = str(source_by_id[ordered_ids[0]].get("type", "paragraph"))
        replacement = _normalise_output_blocks(outputs, fallback_type)
        if not replacement:
            # Empty replacement cannot mean deletion; preserve the local source range.
            bump("empty_ops_recovered")
            continue
        safe, reason = _typeset_replacement_is_safe(ordered_ids, replacement, source_by_id, aliases)
        if not safe:
            bump(f"unsafe_typeset_{reason}_recovered")
            continue
        occupied.update(rank_set)
        normalized.append({
            "start": ranks[0],
            "end": ranks[-1],
            "ids": ordered_ids,
            "blocks": replacement,
        })

    normalized.sort(key=lambda x: x["start"])
    by_start = {op["start"]: op for op in normalized}
    result: list[dict] = []
    index = 0
    while index < len(source_blocks):
        op = by_start.get(index)
        if op is None:
            src = source_blocks[index]
            sid = str(src.get("id"))
            text = str(src.get("text", "")).strip()
            if text:
                result.append({
                    "id": f"ai_{sid}",
                    "source_block_ids": [sid],
                    "type": str(src.get("type", "paragraph")),
                    "text": text,
                })
            index += 1
            continue
        for output in op["blocks"]:
            result.append({
                "id": f"ai_{uuid.uuid4().hex}",
                "source_block_ids": list(op["ids"]),
                "type": output["type"],
                "text": output["text"],
            })
        index = op["end"] + 1
    return result


def _extract_typeset_items(parsed: dict) -> tuple[list[dict], bool]:
    """Recover legacy full-document/full-batch schemas for backward compatibility."""
    parsed = _unwrap_compact_payload(parsed)
    items = parsed.get("blocks")
    if isinstance(items, list):
        return items, False
    legacy = parsed.get("chapters")
    if isinstance(legacy, list):
        flattened = [
            item
            for chapter in legacy if isinstance(chapter, dict)
            for item in (chapter.get("blocks", []) or []) if isinstance(item, dict)
        ]
        return flattened, True
    changes = parsed.get("changes")
    if isinstance(changes, list):
        converted = []
        for item in changes:
            if not isinstance(item, dict):
                continue
            converted.append({
                "source_block_id": item.get("source_block_id") or item.get("id"),
                "type": item.get("type"),
                "text": item.get("text", ""),
            })
        return converted, True
    if isinstance(parsed, dict) and "text" in parsed and any(
        key in parsed for key in ("source_block_ids", "source_block_id", "id")
    ):
        return [parsed], True
    return [], True


def _legacy_full_typeset_to_blocks(
    parsed: dict,
    batch: dict,
    aliases: dict[str, str],
    recovery: dict[str, int],
) -> list[dict]:
    """Convert old full blocks[] replies while locally restoring harmless omissions."""
    source_blocks = list(batch.get("blocks", []))
    source_by_id = {str(b.get("id")): b for b in source_blocks}
    source_rank = {str(b.get("id")): i for i, b in enumerate(source_blocks)}

    def bump(key: str, amount: int = 1) -> None:
        recovery[key] = int(recovery.get(key, 0)) + amount

    items, schema_recovered = _extract_typeset_items(parsed)
    if schema_recovered:
        bump("schema_fallback_batches")
    entries: list[dict] = []
    covered: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        ids = item.get("source_block_ids")
        if isinstance(ids, str):
            ids = [ids]
        elif not isinstance(ids, list):
            single = item.get("source_block_id") or item.get("id")
            ids = [single] if single else []
        ids = [_resolve_source_id(x, source_by_id, aliases) for x in ids]
        ids = list(dict.fromkeys(x for x in ids if x in source_by_id))
        if not ids:
            bump("unknown_output_blocks_ignored")
            continue
        ranks = sorted(source_rank[x] for x in ids)
        if len(ranks) > 1 and ranks != list(range(ranks[0], ranks[-1] + 1)):
            bump("noncontiguous_merges_recovered")
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            text = "\n".join(str(source_by_id[x].get("text", "")).strip() for x in ids).strip()
            bump("empty_blocks_recovered")
        if not text:
            covered.update(ids)
            continue
        safe, reason = _typeset_replacement_is_safe(
            ids, [{"type": _type_name(item.get("type"), str(source_by_id[ids[0]].get("type", "paragraph"))), "text": text}], source_by_id, aliases
        )
        if not safe:
            bump(f"unsafe_typeset_{reason}_recovered")
            continue
        covered.update(ids)
        entries.append({
            "min_rank": min(ranks),
            "max_rank": max(ranks),
            "block": {
                "id": str(item.get("output_id") or f"ai_{uuid.uuid4().hex}"),
                "source_block_ids": ids,
                "type": _type_name(item.get("type"), str(source_by_id[ids[0]].get("type", "paragraph"))),
                "text": text,
            },
        })

    for sid in [str(b.get("id")) for b in source_blocks if str(b.get("id")) not in covered]:
        needle = re.sub(r"\s+", "", str(source_by_id[sid].get("text", "")))
        if len(needle) < 8:
            continue
        matches = [entry for entry in entries if needle in re.sub(r"\s+", "", entry["block"]["text"])]
        if len(matches) == 1:
            entry = matches[0]
            entry["block"]["source_block_ids"].append(sid)
            rank = source_rank[sid]
            entry["min_rank"] = min(entry["min_rank"], rank)
            entry["max_rank"] = max(entry["max_rank"], rank)
            covered.add(sid)
            bump("coverage_inferred")

    missing_ids = [str(b.get("id")) for b in source_blocks if str(b.get("id")) not in covered]
    if missing_ids:
        bump("missing_blocks_recovered", len(missing_ids))
    for sid in missing_ids:
        src = source_by_id[sid]
        text = str(src.get("text", "")).strip()
        if not text:
            continue
        rank = source_rank[sid]
        fallback = {
            "min_rank": rank,
            "max_rank": rank,
            "block": {
                "id": f"ai_recovered_{sid}",
                "source_block_ids": [sid],
                "type": str(src.get("type", "paragraph")),
                "text": text,
            },
        }
        insert_at = len(entries)
        for pos, entry in enumerate(entries):
            if entry["min_rank"] > rank:
                insert_at = pos
                break
        entries.insert(insert_at, fallback)
    return [entry["block"] for entry in entries]


def _compact_to_chapters(
    parsed: dict,
    batch: dict,
    mode: str,
    aliases: dict[str, str] | None = None,
    recovery: dict[str, int] | None = None,
) -> list[dict]:
    """Apply compact patches locally, retaining legacy full-output compatibility."""
    parsed = _unwrap_compact_payload(parsed)
    recovery = recovery if recovery is not None else {}
    aliases = dict(aliases or {})
    source_blocks = list(batch.get("blocks", []))
    source_by_id = {str(b.get("id")): b for b in source_blocks}
    chapter_id = str(batch.get("target_chapter_id") or "chapter_001")

    def bump(key: str, amount: int = 1) -> None:
        recovery[key] = int(recovery.get(key, 0)) + amount

    if mode == "correction":
        changes, legacy = _extract_correction_changes(parsed)
        if legacy:
            bump("legacy_schema_batches")
        changed: dict[str, str] = {}
        for item in changes:
            sid = ""
            text = ""
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                sid = _resolve_source_id(item[0], source_by_id, aliases)
                text = str(item[1] or "").strip()
            elif isinstance(item, dict):
                sid = _resolve_source_id(item.get("source_block_id") or item.get("id"), source_by_id, aliases)
                text = str(item.get("text", "")).strip()
            if sid not in source_by_id:
                bump("unknown_output_blocks_ignored")
                continue
            if not text:
                bump("empty_blocks_recovered")
                continue
            original = str(source_by_id[sid].get("text", ""))
            safe, reason = _correction_replacement_is_safe(original, text, aliases)
            if not safe:
                bump(f"unsafe_correction_{reason}_recovered")
                continue
            changed[sid] = text
        blocks = []
        for src in source_blocks:
            sid = str(src.get("id"))
            text = changed.get(sid) or str(src.get("text", ""))
            if not str(text).strip():
                continue
            blocks.append({
                "id": f"ai_{sid}",
                "source_block_ids": [sid],
                "type": str(src.get("type", "paragraph")),
                "text": text,
            })
        return [{"id": chapter_id, "title": "", "blocks": blocks}]

    ops, recognized_patch = _extract_typeset_ops(parsed)
    if recognized_patch:
        blocks = _apply_typeset_ops(ops, batch, aliases, recovery)
    else:
        # Models/custom adapters using protocol v2 continue to work unchanged.
        blocks = _legacy_full_typeset_to_blocks(parsed, batch, aliases, recovery)
    return [{"id": chapter_id, "title": "", "blocks": blocks}]


def _resolve_workers(provider, batch_count: int) -> int:
    kwargs = getattr(provider, "kwargs", {})
    provider_name = str(kwargs.get("provider_name", getattr(provider, "name", ""))).lower()
    configured = int(kwargs.get("concurrency", 0) or 0)
    if provider_name == "ollama":
        workers = configured if configured > 0 else 1
    elif configured > 0:
        workers = configured
    elif provider_name == "deepseek":
        # DeepSeek limits concurrency per account rather than per API key. Flash is
        # designed for high throughput; Pro and thinking mode use a more conservative
        # default to avoid needless reasoning-token bursts.
        model = str(getattr(provider, "model", "") or "").lower()
        thinking = bool(kwargs.get("deepseek_thinking", False))
        workers = 6 if thinking else (12 if "pro" in model else 20)
        rpm = int(kwargs.get("rpm_limit", 0) or 0)
        if rpm > 0:
            workers = min(workers, max(1, math.ceil(rpm / 10)))
    else:
        rpm = int(kwargs.get("rpm_limit", 0) or 0)
        workers = 8 if rpm <= 0 else max(1, min(32, math.ceil(rpm / 15)))
        key_count = len([x for x in str(getattr(provider, "api_key", "")).split(",") if x.strip()])
        if key_count > 1:
            workers = min(64, max(workers, key_count * 8))
    return max(1, min(int(workers), 64, max(1, batch_count)))


def run_ai_document(
    provider,
    doc,
    prompt_template: str = "",
    mode: str = "typeset",
    progress_callback=None,
    cancel_check=None,
    *,
    request_css: bool = True,
    layout_lock: bool | None = None,
    cleanup_replacement_fragments: bool = True,
    checkpoint_dir: str | os.PathLike | None = None,
    resume: bool = True,
):
    """Run token-light, bounded, concurrent and order-preserving AI processing."""
    # Replacement documents created by older versions may still contain tiny OCR
    # continuation columns after a complete replaced sentence. Clean a private copy
    # before batching so the model never sees and "corrects" those fragments into
    # more convincing duplicates.
    doc = copy.deepcopy(doc)
    pre_ai_fragment_cleanup = 0
    if cleanup_replacement_fragments:
        from engine.replacement_engine import cleanup_covered_replacement_fragments
        pre_ai_fragment_cleanup = cleanup_covered_replacement_fragments(doc)
    payload = ai_request_payload(doc)
    mode = "correction" if mode == "correction" else "typeset"
    template = prompt_template or (CORRECTION_PROMPT if mode == "correction" else TYPESET_PROMPT)
    kwargs = getattr(provider, "kwargs", {})
    temperature = float(kwargs.get("temperature", 0.2))
    configured_chars = int(kwargs.get("ai_batch_chars", 0) or 0)
    configured_tokens = int(kwargs.get("ai_batch_tokens", 0) or 0)
    # Differential replies make output tiny. DeepSeek V4 has a 1M context window and
    # benefits from fewer, larger batches, while other providers keep conservative
    # cross-platform defaults.
    provider_name = str(kwargs.get("provider_name", getattr(provider, "name", ""))).lower()
    deepseek_thinking = bool(kwargs.get("deepseek_thinking", False))
    if provider_name == "deepseek" and not deepseek_thinking:
        default_tokens = 48000 if mode == "correction" else 32000
    else:
        default_tokens = 24000 if mode == "correction" else 16000
    batch_tokens = configured_tokens if configured_tokens > 0 else default_tokens
    batches = _chapter_batches(
        payload,
        max_chars=configured_chars,
        max_tokens=batch_tokens,
        prompt_template=template,
        model=str(getattr(provider, "model", "") or ""),
    )
    if mode == "typeset" and batches and request_css:
        # Only one batch requests the global stylesheet. Formatter-workspace AI
        # disables this because its output is a document revision, not an EPUB theme.
        batches[0]["request_css"] = True

    checkpoint_job_dir: Path | None = None
    checkpoint_signature = ""
    if checkpoint_dir:
        canonical_batches = [{
            "target_chapter_id": b.get("target_chapter_id"),
            "chapter_part": b.get("chapter_part"),
            "blocks": [{"type": x.get("type"), "text": x.get("text")} for x in b.get("blocks", [])],
            "request_css": bool(b.get("request_css", False)),
        } for b in batches]
        signature_payload = {
            "version": 2,
            "mode": mode,
            "model": str(getattr(provider, "model", "") or ""),
            "prompt_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
            "batches": canonical_batches,
        }
        checkpoint_signature = hashlib.sha256(
            json.dumps(signature_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        checkpoint_job_dir = Path(checkpoint_dir).expanduser() / f"{mode}_{checkpoint_signature[:20]}"
        checkpoint_job_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "version": 1,
            "signature": checkpoint_signature,
            "mode": mode,
            "model": str(getattr(provider, "model", "") or ""),
            "total_batches": len(batches),
            "updated_at": time.time(),
        }
        tmp = checkpoint_job_dir / "manifest.json.tmp"
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, checkpoint_job_dir / "manifest.json")

    workers = _resolve_workers(provider, len(batches))
    limiter = RequestLimiter(int(kwargs.get("rpm_limit", 0) or 0), int(kwargs.get("tpm_limit", 0) or 0))
    usage_tracker = _UsageTracker()

    chapter_ids = list(dict.fromkeys(str(x.get("target_chapter_id", "chapter_001")) for x in batches))
    completed_batches = 0
    completed_chapters = 0
    chapter_parts = {cid: sum(1 for x in batches if x.get("target_chapter_id") == cid) for cid in chapter_ids}
    chapter_done = {cid: 0 for cid in chapter_ids}
    progress_lock = threading.Lock()
    recovery_lock = threading.Lock()
    css_lock = threading.Lock()
    recovery_stats: dict[str, int] = {}
    generated_css = {"value": "", "source": ""}
    resumed_batches = 0
    started = time.monotonic()

    def emit(stage: str, chapter: str = ""):
        if not progress_callback:
            return
        elapsed = max(0.001, time.monotonic() - started)
        rate = completed_batches / elapsed
        remaining = max(0, len(batches) - completed_batches)
        usage_view = usage_tracker.snapshot(provider)
        progress_callback({
            "stage": stage,
            "current": completed_batches,
            "total": len(batches),
            "completed_chapters": completed_chapters,
            "total_chapters": len(chapter_ids),
            "chapter": chapter,
            "workers": workers,
            "batches": len(batches),
            "rate": rate,
            "eta_seconds": (remaining / rate) if rate > 0 else None,
            "unit": "batch",
            "token_total": usage_view["display_total_tokens"],
            "token_actual": usage_view["usage_is_actual"],
            "requests": usage_view.get("provider_requests") or usage_view["requests"],
            "resumed_batches": resumed_batches,
            "checkpoint_dir": str(checkpoint_job_dir) if checkpoint_job_dir is not None else "",
        })

    emit("初始化任务")

    def merge_recovery(local_recovery: dict[str, int]) -> None:
        if not local_recovery:
            return
        with recovery_lock:
            for key, value in local_recovery.items():
                recovery_stats[key] = recovery_stats.get(key, 0) + value

    def capture_css(parsed: dict, batch: dict) -> None:
        if mode != "typeset" or not batch.get("request_css"):
            return
        css = _extract_ai_css(parsed)
        if not css:
            return
        with css_lock:
            if not generated_css["value"]:
                generated_css["value"] = css
                generated_css["source"] = "ai"

    def local_noop(batch: dict, aliases: dict[str, str], key: str) -> list[dict]:
        local_recovery = {key: 1}
        parsed = {"c": []} if mode == "correction" else {"o": []}
        converted = _compact_to_chapters(parsed, batch, mode, aliases=aliases, recovery=local_recovery)
        merge_recovery(local_recovery)
        return converted

    def process_batch(batch: dict, label: str, split_depth: int = 0) -> list[dict]:
        if cancel_check and cancel_check():
            raise RuntimeError("AI任务已停止")
        wire_batch, aliases = _wire_batch(batch)
        prompt_payload = json.dumps(wire_batch, ensure_ascii=False, separators=(",", ":"))
        prompt = template.replace("{{INPUT}}", prompt_payload)
        try:
            parsed = _call_and_parse(
                provider,
                prompt,
                temperature,
                api_retries=1,
                cancel_check=cancel_check,
                limiter=limiter,
                usage=usage_tracker,
            )
            capture_css(parsed, batch)
            local_recovery: dict[str, int] = {}
            converted = _compact_to_chapters(parsed, batch, mode, aliases=aliases, recovery=local_recovery)
            merge_recovery(local_recovery)
            return converted
        except AIReplyFormatError as exc:
            blocks = list(batch.get("blocks", []))
            if len(blocks) > 1 and split_depth < 8:
                # Do not repeat the same billed malformed output. Split immediately.
                usage_tracker.split()
                mid = max(1, len(blocks) // 2)
                output: list[dict] = []
                for part_no, part in enumerate((blocks[:mid], blocks[mid:]), 1):
                    child = dict(batch)
                    child["blocks"] = part
                    child["chapter_part"] = f"{batch.get('chapter_part', 1)}.{part_no}"
                    output.extend(process_batch(child, f"{label}.{part_no}", split_depth + 1))
                return output
            # A single block cannot be split further. One tiny strict repair request is
            # cheap; if it still fails, preserve local source instead of aborting the book.
            try:
                parsed = _call_and_parse(
                    provider,
                    prompt + _RETRY_SUFFIX,
                    temperature,
                    api_retries=0,
                    cancel_check=cancel_check,
                    limiter=limiter,
                    usage=usage_tracker,
                )
                capture_css(parsed, batch)
                local_recovery = {"singleton_json_repair": 1}
                converted = _compact_to_chapters(parsed, batch, mode, aliases=aliases, recovery=local_recovery)
                merge_recovery(local_recovery)
                return converted
            except AIReplyFormatError:
                usage_tracker.singleton_fallback()
                return local_noop(batch, aliases, "singleton_invalid_json_recovered")

    def _remap_checkpoint_result(returned, batch):
        if not isinstance(returned, list): return None
        old=[]
        for chapter in returned:
            for block in (chapter.get("blocks", []) if isinstance(chapter, dict) else []):
                ids=block.get("source_block_ids", [])
                if isinstance(ids, str): ids=[ids]
                for sid in ids:
                    sid=str(sid)
                    if sid and sid not in old: old.append(sid)
        current=[str(x.get("id", "")) for x in batch.get("blocks", [])]
        if len(old) != len(current): return None
        mapping=dict(zip(old,current))
        cloned=copy.deepcopy(returned)
        for chapter in cloned:
            for block in chapter.get("blocks", []):
                ids=block.get("source_block_ids", [])
                if isinstance(ids,str): ids=[ids]
                block["source_block_ids"]=[mapping.get(str(x),str(x)) for x in ids]
        return cloned

    ordered_results: list[list[dict] | None] = [None] * len(batches)
    if checkpoint_job_dir is not None and resume:
        for index in range(len(batches)):
            batch_path = checkpoint_job_dir / f"batch_{index:06d}.json"
            if not batch_path.exists():
                continue
            try:
                saved = json.loads(batch_path.read_text(encoding="utf-8"))
                if saved.get("signature") != checkpoint_signature or int(saved.get("index", -1)) != index:
                    continue
                returned = _remap_checkpoint_result(saved.get("result"), batches[index])
                if isinstance(returned, list):
                    ordered_results[index] = returned
                    resumed_batches += 1
                    cid = str(batches[index].get("target_chapter_id", "chapter_001"))
                    chapter_done[cid] += 1
            except Exception:
                continue
        # Import old random-ID checkpoints from sibling job folders. Results are
        # remapped by source order, while current integrity gates remain authoritative.
        for legacy_dir in sorted(Path(checkpoint_dir).expanduser().glob(f"{mode}_*"), key=lambda p: p.stat().st_mtime, reverse=True):
            if legacy_dir == checkpoint_job_dir: continue
            try:
                manifest=json.loads((legacy_dir/"manifest.json").read_text(encoding="utf-8"))
                if manifest.get("mode") != mode or str(manifest.get("model", "")) != str(getattr(provider,"model","") or ""): continue
                if int(manifest.get("total_batches",-1)) != len(batches): continue
            except Exception: continue
            for index in range(len(batches)):
                if ordered_results[index] is not None: continue
                try:
                    saved=json.loads((legacy_dir/f"batch_{index:06d}.json").read_text(encoding="utf-8"))
                    returned=_remap_checkpoint_result(saved.get("result"), batches[index])
                    if returned is not None:
                        ordered_results[index]=returned; resumed_batches += 1
                        save_target=checkpoint_job_dir/f"batch_{index:06d}.json"
                        save_target.write_text(json.dumps({"version":2,"signature":checkpoint_signature,"index":index,"result":returned,"saved_at":time.time()},ensure_ascii=False,separators=(",",":")),encoding="utf-8")
                except Exception: pass

        completed_batches = resumed_batches
        completed_chapters = sum(1 for cid in chapter_ids if chapter_done[cid] >= chapter_parts[cid])
        if resumed_batches:
            emit("已恢复断点")

    def save_checkpoint(index: int, returned: list[dict]) -> None:
        if checkpoint_job_dir is None:
            return
        payload = {
            "version": 1,
            "signature": checkpoint_signature,
            "index": index,
            "result": returned,
            "saved_at": time.time(),
        }
        target = checkpoint_job_dir / f"batch_{index:06d}.json"
        tmp = checkpoint_job_dir / f"batch_{index:06d}.json.tmp"
        tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, target)

    def task(index: int, batch: dict):
        cid = str(batch.get("target_chapter_id", "chapter_001"))
        emit("AI生成中", cid)
        return index, cid, process_batch(batch, str(index + 1))

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="novel-ai")
    futures = []
    try:
        futures = [
            executor.submit(task, i, batch)
            for i, batch in enumerate(batches)
            if ordered_results[i] is None
        ]
        for future in concurrent.futures.as_completed(futures):
            if cancel_check and cancel_check():
                raise RuntimeError("AI任务已停止")
            index, cid, returned = future.result()
            ordered_results[index] = returned
            save_checkpoint(index, returned)
            with progress_lock:
                completed_batches += 1
                chapter_done[cid] += 1
                if chapter_done[cid] >= chapter_parts[cid]:
                    completed_chapters += 1
                emit("已保存", cid)
    except Exception:
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    chapters_by_id: dict[str, dict] = {}
    chapter_order: list[str] = []
    for batch, returned in zip(batches, ordered_results):
        if returned is None:
            raise RuntimeError("AI任务结果不完整")
        fallback_id = str(batch.get("target_chapter_id", "chapter_001"))
        for chapter in returned:
            chapter_id = str(chapter.get("id") or fallback_id)
            if chapter_id not in chapters_by_id:
                chapters_by_id[chapter_id] = {"id": chapter_id, "title": chapter.get("title", ""), "blocks": []}
                chapter_order.append(chapter_id)
            chapters_by_id[chapter_id]["blocks"].extend(chapter.get("blocks", []) or [])

    merged = {
        "document_id": payload.get("document_id", "project"),
        "source_version": payload.get("source_version", "replacement"),
        "chapters": [chapters_by_id[cid] for cid in chapter_order],
        "changes": [],
        "complete_document": True,
    }
    result, changes = build_ai_document(doc, merged)
    post_ai_fragment_cleanup = cleanup_ai_covered_fragments(doc, result) if mode == "typeset" else 0
    result.metadata.ai_processing_mode = mode
    result.metadata.ai_layout_locked = (mode == "typeset") if layout_lock is None else bool(layout_lock)
    if mode == "typeset" and request_css:
        css = generated_css["value"] or AI_TYPESET_FALLBACK_CSS
        result.metadata.ai_epub_css = css
        result.metadata.ai_epub_css_name = "ai_typeset.css"
        result.metadata.ai_epub_css_source = generated_css["source"] or "fallback"
    else:
        # AI 纠错以及 Formatter 专用 AI 都不携带 EPUB 排版 CSS。
        result.metadata.ai_epub_css = ""
        result.metadata.ai_epub_css_name = ""
        result.metadata.ai_epub_css_source = ""
    usage = usage_tracker.snapshot(provider)
    actual_or_estimated = "actual" if usage["usage_is_actual"] else "estimated"
    result.add_log(
        "ai_mode",
        (
            f"AI mode={mode}; compact_protocol=3-patch; batches={len(batches)}; "
            f"concurrency={workers}; batch_tokens={batch_tokens}; source_char_cap={configured_chars}; "
            f"requests={usage['requests']}; api_retries={usage['api_retries']}; "
            f"split_events={usage['split_events']}; tokens_{actual_or_estimated}={usage['display_total_tokens']}; "
            f"prompt_tokens={usage['prompt_tokens']}; completion_tokens={usage['completion_tokens']}; "
            f"cached_tokens={usage['cached_tokens']}; cache_miss_tokens={usage['cache_miss_tokens']}; "
            f"reasoning_tokens={usage['reasoning_tokens']}; provider_requests={usage['provider_requests']}; "
            f"deepseek_thinking={deepseek_thinking if provider_name == 'deepseek' else 'n/a'}; "
            f"recovery={recovery_stats}; pre_ai_fragment_cleanup={pre_ai_fragment_cleanup}; "
            f"post_ai_fragment_cleanup={post_ai_fragment_cleanup}; "
            f"checkpoint={str(checkpoint_job_dir) if checkpoint_job_dir is not None else 'off'}; "
            f"resumed_batches={resumed_batches}"
        ),
        sum(recovery_stats.values()) + pre_ai_fragment_cleanup + post_ai_fragment_cleanup,
    )
    emit("完成")
    return result, changes
