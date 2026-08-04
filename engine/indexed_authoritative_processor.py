# -*- coding: utf-8 -*-
"""AiNiee-style indexed proofreading for authoritative novel text.

The model is never allowed to rebuild document structure. Each source block owns one
stable mutable slot. A batch is committed only when every requested index is present,
in order, non-empty, and a safe full replacement of its own source block. Invalid
batches are split and retried; an invalid singleton falls back to its source text.

After proofreading, structure is restored only by deterministic, character-preserving
local operations. This makes omission, duplication, context write-back and protocol
leakage impossible by construction rather than attempting to repair them afterwards.
"""
from __future__ import annotations

import concurrent.futures
import copy
import difflib
import hashlib
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Iterable

from ai.request_limiter import RequestLimiter
from ai.token_counter import estimate_tokens
from engine.ai_document_processor import (
    AIReplyFormatError,
    _UsageTracker,
    _call_and_parse,
    _correction_replacement_is_safe,
    _resolve_workers,
)
from engine.authoritative_text import (
    _TEXT_TYPES,
    _adjacent_pair_issue,
    _body_blocks,
    apply_conservative_ocr_fixes,
    apply_post_indexed_ocr_fixes,
    mark_authoritative,
    promote_authoritative_chapters,
    sanitize_authoritative_source,
    validate_authoritative_transition,
)
from models.document import Block, BlockType, UnifiedDocument

PROTOCOL_VERSION = 8
RECONSTRUCTED_REVIEW_PROTOCOL = 1
READABILITY_REVIEW_PROTOCOL = 2
PROBLEM_PATCH_PROTOCOL = 2

INDEXED_PROOFREAD_PROMPT = """Proofread Japanese light-novel OCR. The model is editing fixed rows, not a document.

INPUT is JSON:
{"context":["read-only previous text",...],"rows":[[1,"p","full source text"],...]}
Types: p=paragraph, d=dialogue, c=chapter, s=section, r=ruby, f=footnote, t=toc, u=other.

Correct only clear OCR character errors, unmistakably broken punctuation, and obvious OCR-caused grammar errors. Preserve wording, facts, names, viewpoint and style. Never translate, summarize, continue an incomplete sentence from imagination, copy context, merge rows, split rows, reorder rows, or omit text.

Return EVERY requested row exactly once and in the same order as compact JSON:
{"rows":[[1,"full corrected text"],[2,"full corrected text"]]}
The text must be the complete replacement for that row, even when unchanged. Do not return type codes, source IDs, explanations, Markdown, or extra keys.
INPUT:\n{{INPUT}}"""


RECONSTRUCTED_REVIEW_PROMPT = """Proofread reconstructed Japanese light-novel OCR sentences. Each row is now one complete logical paragraph or person dialogue assembled from OCR columns.

INPUT is JSON:
{"context":["read-only previous text",...],"rows":[[1,"p","full reconstructed text"],...]}

Correct clear OCR character mistakes, duplicated/missing kana, ruby-reading debris, unmistakable punctuation errors, and obvious OCR-caused grammar defects such as a missing particle or conjugation. Preserve every fact, name, tone, viewpoint and sentence meaning. Do not translate, summarize, rewrite style, add events, complete a genuinely missing clause from imagination, merge rows, split rows, copy context, or omit text.

Return every row exactly once in order as compact JSON:
{"rows":[[1,"full corrected reconstructed text"],[2,"full corrected reconstructed text"]]}
No explanations, Markdown or extra keys.
INPUT:\n{{INPUT}}"""


READABILITY_REVIEW_PROMPT = """Repair Japanese light-novel OCR for natural readability. The rows are fixed output slots inside one chapter-sized semantic window.

INPUT is JSON:
{"context":["read-only previous text",...],"next_context":["read-only following text",...],"rows":[[1,"p","full current text"],...]}

Core goal: make the current Japanese read naturally and coherently, not archaeological character-by-character restoration. Correct OCR glyph errors, ruby/furigana debris, duplicated or missing kana, broken words, punctuation, particles, conjugation, and OCR-caused grammar. You MAY infer and supply a short missing kana, particle, word, or clause when the surrounding rows make it strongly supported. Preserve the original plot, facts, character relationships, names, first-person voice, tone, humour and level of formality. Do not add a new event, new dialogue, new description, or change who did what. Do not translate or summarize.

The local layout pass has already reconstructed most column wraps. Keep one output for each input row, in the same order. Never copy an entire neighbouring row into another row, never merge/split/omit rows, and never return commentary. If a row is already natural, return it unchanged.

Return compact JSON only:
{"rows":[[1,"full repaired text"],[2,"full repaired text"]]}
INPUT:\n{{INPUT}}"""

_RETRY_SUFFIX = """
The previous reply failed strict validation. Return exactly the requested rows once each,
in order, with complete non-empty replacement text. Do not copy context or add commentary.
"""

_TYPE_CODE = {
    BlockType.PARAGRAPH: "p",
    BlockType.DIALOGUE: "d",
    BlockType.CHAPTER: "c",
    BlockType.SECTION: "s",
    BlockType.RUBY: "r",
    BlockType.FOOTNOTE: "f",
    BlockType.TOC_ENTRY: "t",
}

_PROTOCOL_RE = re.compile(r"^(?:[pcdsrftu]|b\d{1,6}|row\d{1,6})$", re.I)


def remove_demonstrable_duplicate_runs(doc: UnifiedDocument) -> int:
    """Remove only exact repeated runs large enough to be OCR/batch duplication.

    A minimum of four consecutive blocks and 160 normalized characters must repeat.
    The later copy is removed. The scan repeats because deleting one overlap copy can
    expose another longer shifted copy from the same malformed source export.
    """
    from engine.authoritative_text import _duplicate_runs
    removed_runs = 0
    removed_blocks = 0
    for _ in range(32):
        runs = _duplicate_runs(doc, min_blocks=4, min_chars=160)
        if not runs:
            break
        body = _body_blocks(doc)
        # Remove the nearest/largest later copy first, then rescan against the changed
        # sequence. This avoids stale body indices when overlap duplicates interleave.
        _start, repeat_start, length = sorted(runs, key=lambda x: (x[1], -x[2]))[0]
        remove_ids = {body[i].id for i in range(repeat_start, min(len(body), repeat_start + length))}
        if not remove_ids:
            break
        doc.blocks = [b for b in doc.blocks if b.id not in remove_ids]
        removed_runs += 1
        removed_blocks += len(remove_ids)
    if removed_blocks:
        doc.add_log(
            "authoritative_exact_run_dedup",
            f"确定性删除 {removed_runs} 组完整重复片段（{removed_blocks} 个正文块）",
            removed_blocks,
        )
    return removed_runs

def _stable_item_key(index: int, block: Block) -> str:
    digest = hashlib.sha256(
        (f"{index}\0{block.type.value}\0{block.text}").encode("utf-8")
    ).hexdigest()[:24]
    return f"{index:08d}-{digest}"


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)


def _canonical_signature(doc: UnifiedDocument, provider, prompt: str) -> str:
    rows = [[b.type.value, str(b.text or "")] for b in _body_blocks(doc)]
    payload = {
        "protocol": PROTOCOL_VERSION,
        "model": str(getattr(provider, "model", "") or ""),
        "prompt": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "rows": rows,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_state(path: Path, signature: str, expected: list[dict], *, protocol: int = PROTOCOL_VERSION) -> tuple[list[dict], int]:
    if not path.exists():
        return copy.deepcopy(expected), 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("signature") != signature or payload.get("protocol") != protocol:
            return copy.deepcopy(expected), 0
        saved = payload.get("items")
        if not isinstance(saved, list) or len(saved) != len(expected):
            return copy.deepcopy(expected), 0
        resumed = 0
        restored: list[dict] = []
        for exp, got in zip(expected, saved):
            if not isinstance(got, dict) or got.get("key") != exp["key"] or got.get("source_hash") != exp["source_hash"]:
                restored.append(copy.deepcopy(exp))
                continue
            item = copy.deepcopy(exp)
            status = str(got.get("status", "pending"))
            text = str(got.get("corrected_text", "") or "")
            if status in {"done", "review"} and text.strip():
                item.update({
                    "status": status,
                    "corrected_text": text,
                    "attempts": int(got.get("attempts", 0) or 0),
                    "error": str(got.get("error", "") or ""),
                })
                resumed += 1
            restored.append(item)
        return restored, resumed
    except Exception:
        return copy.deepcopy(expected), 0


def _save_state(path: Path, signature: str, provider, items: list[dict], *, protocol: int = PROTOCOL_VERSION) -> None:
    _atomic_json(path, {
        "protocol": protocol,
        "signature": signature,
        "model": str(getattr(provider, "model", "") or ""),
        "updated_at": time.time(),
        "items": items,
    })


def _build_batches(items: list[dict], max_tokens: int, prompt: str, model: str) -> list[list[int]]:
    pending = [i for i, item in enumerate(items) if item["status"] == "pending"]
    if not pending:
        return []
    base = estimate_tokens(prompt.replace("{{INPUT}}", '{"context":[],"rows":[]}'), model)
    batches: list[list[int]] = []
    current: list[int] = []
    current_tokens = base
    # Keep batches relatively small. Strict retry is cheap and local, while very large
    # proofreading batches make row drift and truncated replies more likely.
    max_tokens = max(1200, int(max_tokens or 8000))
    for idx in pending:
        item = items[idx]
        row = [len(current) + 1, item["type_code"], item["source_text"]]
        cost = estimate_tokens(json.dumps(row, ensure_ascii=False, separators=(",", ":")), model) + 4
        if current and current_tokens + cost > max_tokens:
            batches.append(current)
            current = []
            current_tokens = base
            row = [1, item["type_code"], item["source_text"]]
            cost = estimate_tokens(json.dumps(row, ensure_ascii=False, separators=(",", ":")), model) + 4
        current.append(idx)
        current_tokens += cost
    if current:
        batches.append(current)
    return batches


def _build_readability_batches(items: list[dict], max_tokens: int, prompt: str, model: str, max_chars: int = 3000) -> list[list[int]]:
    """Create contiguous chapter-aware semantic windows for readability repair."""
    pending = [i for i, item in enumerate(items) if item["status"] == "pending"]
    if not pending:
        return []
    base = estimate_tokens(prompt.replace("{{INPUT}}", '{"context":[],"next_context":[],"rows":[]}'), model)
    token_limit = max(1800, min(int(max_tokens or 6000), 9000))
    char_limit = max(1000, min(int(max_chars or 3000), 5000))
    batches: list[list[int]] = []
    current: list[int] = []
    current_tokens = base
    current_chars = 0
    current_chapter = None
    for idx in pending:
        item = items[idx]
        chapter = item.get("chapter_index")
        row = [len(current) + 1, item["type_code"], item["source_text"]]
        cost = estimate_tokens(json.dumps(row, ensure_ascii=False, separators=(",", ":")), model) + 4
        chars = len(str(item["source_text"] or ""))
        chapter_changed = bool(current) and chapter != current_chapter
        if current and (chapter_changed or current_tokens + cost > token_limit or current_chars + chars > char_limit):
            batches.append(current)
            current = []
            current_tokens = base
            current_chars = 0
            row = [1, item["type_code"], item["source_text"]]
            cost = estimate_tokens(json.dumps(row, ensure_ascii=False, separators=(",", ":")), model) + 4
        if not current:
            current_chapter = chapter
        current.append(idx)
        current_tokens += cost
        current_chars += chars
    if current:
        batches.append(current)
    return batches


def _parse_rows(parsed: dict, count: int) -> list[str]:
    if not isinstance(parsed, dict) or set(parsed.keys()) - {"rows"}:
        raise ValueError("unexpected_top_level")
    rows = parsed.get("rows")
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError("row_count")
    result: list[str] = []
    seen: set[int] = set()
    for expected, row in enumerate(rows, 1):
        if isinstance(row, dict):
            row_id = row.get("id")
            text = row.get("text")
        elif isinstance(row, (list, tuple)) and len(row) == 2:
            row_id, text = row
        else:
            raise ValueError("row_schema")
        try:
            row_id = int(row_id)
        except (TypeError, ValueError):
            raise ValueError("row_id")
        if row_id != expected or row_id in seen:
            raise ValueError("row_order")
        seen.add(row_id)
        text = str(text or "")
        if not text.strip() or _PROTOCOL_RE.fullmatch(text.strip()):
            raise ValueError("empty_or_protocol")
        result.append(text)
    return result


def _readability_replacement_is_safe(source: str, output: str) -> tuple[bool, str]:
    """Permissive but bounded guard for readability-first OCR repair."""
    original = re.sub(r"[\s　]+", "", str(source or ""))
    revised = re.sub(r"[\s　]+", "", str(output or ""))
    if not revised or _PROTOCOL_RE.fullmatch(revised):
        return False, "empty_or_protocol"
    if not original:
        return False, "empty_source"
    old_len, new_len = len(original), len(revised)
    if old_len >= 24 and new_len < max(6, int(old_len * 0.35)):
        return False, "too_short"
    if new_len > max(old_len + 180, int(old_len * 2.2)):
        return False, "too_long"
    if old_len >= 10:
        ratio = difflib.SequenceMatcher(None, original, revised, autojunk=False).ratio()
        if ratio < 0.20:
            return False, "unrelated_rewrite"
    return True, ""


def _validate_batch(items: list[dict], indices: list[int], outputs: list[str], *, repair_mode: str = "strict") -> None:
    if len(outputs) != len(indices):
        raise ValueError("row_count")
    output_norms: dict[str, list[int]] = {}
    for local, (idx, output) in enumerate(zip(indices, outputs), 1):
        source = items[idx]["source_text"]
        if repair_mode == "readability":
            safe, reason = _readability_replacement_is_safe(source, output)
        else:
            safe, reason = _correction_replacement_is_safe(source, output)
        if not safe:
            raise ValueError(f"unsafe_row_{local}_{reason}")
        compact = re.sub(r"[\s　]+", "", output)
        output_norms.setdefault(compact, []).append(local)

    if repair_mode == "readability":
        # A readability model may try to heal a boundary by copying the whole next row
        # into the current row. The fixed-slot protocol forbids that because the next
        # slot still exists and would become duplicated. Short common phrases are ignored.
        source_norms = [re.sub(r"[\s　]+", "", items[idx]["source_text"]) for idx in indices]
        output_norms_list = [re.sub(r"[\s　]+", "", value) for value in outputs]
        for pos, output_norm in enumerate(output_norms_list):
            for other, source_norm in enumerate(source_norms):
                if pos == other or len(source_norm) < 20:
                    continue
                own_source = source_norms[pos]
                if source_norm in output_norm and source_norm not in own_source:
                    raise ValueError(f"absorbed_row_{pos + 1}_{other + 1}")

    # A model sometimes duplicates one requested row into another. Even a short row can
    # pass loose length checks, so reject duplicate outputs unless the source rows were
    # already exactly identical.
    for positions in output_norms.values():
        if len(positions) < 2:
            continue
        source_values = {
            re.sub(r"[\s　]+", "", items[indices[pos - 1]]["source_text"])
            for pos in positions
        }
        if len(source_values) > 1:
            raise ValueError("duplicated_output_rows")


def _recover_indexed_correction_adjacency(
    source: UnifiedDocument, corrected: UnifiedDocument, items: list[dict]
) -> tuple[int, list[str]]:
    """Roll back only indexed rows that introduce adjacent duplication/coverage.

    The indexed protocol locks one source block to one output slot, but a model can
    still return ``current row + next row`` for a short source and pass a permissive
    per-row similarity threshold.  Compare aligned source/output boundaries.  When a
    boundary was clean in the source but becomes an exact/tail duplicate, restore only
    the changed participant(s), mark those cache items for review, and preserve every
    other paid correction.
    """
    source_blocks = _body_blocks(source)
    output_blocks = _body_blocks(corrected)
    if len(source_blocks) != len(output_blocks):
        return 0, []

    item_by_block_id = {str(item.get("block_id", "")): item for item in items}
    recovered_ids: set[str] = set()
    recovered_issues: list[str] = []

    # Restoring one side may expose/resolve the neighbouring boundary, so rescan a
    # handful of times.  Each pass can only move rows back to their immutable source.
    for _ in range(8):
        changed_this_pass = False
        for index in range(len(output_blocks) - 1):
            after_issue = _adjacent_pair_issue(output_blocks[index], output_blocks[index + 1])
            if not after_issue:
                continue
            before_issue = _adjacent_pair_issue(source_blocks[index], source_blocks[index + 1])
            if before_issue:
                # Intentional/source-existing repetition is not an AI error.
                continue

            positions = [
                pos for pos in (index, index + 1)
                if str(output_blocks[pos].text or "") != str(source_blocks[pos].text or "")
            ]
            # Defensive fallback: an alignment anomaly should still never abort the
            # full book. Restoring both source rows is lossless and deterministic.
            if not positions:
                positions = [index, index + 1]

            for pos in positions:
                src = source_blocks[pos]
                out = output_blocks[pos]
                if out.id in recovered_ids and out.text == src.text:
                    continue
                out.text = src.text
                out.type = src.type
                out.ocr_raw = src.ocr_raw
                out.modified_by = "indexed_source_recovery"
                item = item_by_block_id.get(str(out.id))
                item_key = str(item.get("key", "")) if item else ""
                out.metadata = {
                    **(src.metadata or {}),
                    "source_block_ids": list((src.metadata or {}).get("source_block_ids") or [src.id]),
                    "indexed_status": "review",
                    "indexed_item_key": item_key,
                    "indexed_recovery_reason": after_issue,
                }
                if item is not None:
                    item["corrected_text"] = item["source_text"]
                    item["status"] = "review"
                    item["error"] = f"introduced_adjacent_{after_issue}"
                recovered_ids.add(str(out.id))
                changed_this_pass = True
            recovered_issues.append(after_issue)
        if not changed_this_pass:
            break

    if recovered_ids:
        corrected.add_log(
            "indexed_adjacent_recovery",
            f"逐条纠错局部回退 {len(recovered_ids)} 条，消除 {len(recovered_issues)} 处新增相邻重复/覆盖",
            len(recovered_ids),
        )
    return len(recovered_ids), recovered_issues


def _context_for(items: list[dict], first_index: int, count: int = 3) -> list[str]:
    explicit = items[first_index].get("context_before") if 0 <= first_index < len(items) else None
    if isinstance(explicit, list):
        return [str(value) for value in explicit[-count:] if str(value).strip()]
    start = max(0, first_index - count)
    return [items[i]["corrected_text"] or items[i]["source_text"] for i in range(start, first_index)]


def _request_batch(provider, items: list[dict], indices: list[int], temperature: float,
                   limiter: RequestLimiter, usage: _UsageTracker, cancel_check=None,
                   retry_suffix: bool = False, prompt_template: str = INDEXED_PROOFREAD_PROMPT,
                   repair_mode: str = "strict") -> list[str]:
    rows = [[n, items[idx]["type_code"], items[idx]["source_text"]] for n, idx in enumerate(indices, 1)]
    payload = {"context": _context_for(items, indices[0]), "rows": rows}
    if repair_mode == "readability":
        last = indices[-1]
        payload["next_context"] = [
            str(items[i].get("corrected_text") or items[i]["source_text"])
            for i in range(last + 1, min(len(items), last + 3))
            if str(items[i].get("corrected_text") or items[i]["source_text"]).strip()
        ]
    prompt = prompt_template.replace(
        "{{INPUT}}", json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
    if retry_suffix:
        prompt += _RETRY_SUFFIX
    parsed = _call_and_parse(
        provider, prompt, temperature, api_retries=1 if not retry_suffix else 0,
        cancel_check=cancel_check, limiter=limiter, usage=usage,
    )
    outputs = _parse_rows(parsed, len(indices))
    _validate_batch(items, indices, outputs, repair_mode=repair_mode)
    if repair_mode == "readability":
        contexts = list(payload.get("context") or []) + list(payload.get("next_context") or [])
        context_norms = [re.sub(r"[\s　]+", "", str(value or "")) for value in contexts]
        for item_index, output in zip(indices, outputs):
            own = re.sub(r"[\s　]+", "", items[item_index]["source_text"])
            revised = re.sub(r"[\s　]+", "", output)
            for context in context_norms:
                if len(context) >= 20 and context in revised and context not in own:
                    raise ValueError("copied_readonly_context")
    return outputs


def _append_modified_by(current: str, step: str) -> str:
    values = [x for x in str(current or "").split(",") if x]
    if step not in values:
        values.append(step)
    return ",".join(values)


def _has_unbalanced_quote(text: str) -> bool:
    return text.count("「") != text.count("」") or text.count("『") != text.count("』")


_JP_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff々〆ヵヶー]")
_INTERNAL_WRAP_RE = re.compile(
    r"(?P<left>[\u3040-\u30ff\u3400-\u9fff々〆ヵヶー])"
    r"(?P<gap>[ \t　]*\r?\n[ \t　]*)"
    r"(?P<right>[\u3040-\u30ff\u3400-\u9fff々〆ヵヶー])"
)


def _is_japanese_char(value: str) -> bool:
    return bool(value and _JP_CHAR_RE.fullmatch(value[0]))


def _unwrap_internal_ocr_wraps(doc: UnifiedDocument) -> int:
    """Remove only OCR hard-wrap whitespace inside a logical text block.

    A single newline between Japanese characters is layout, not prose. Blank lines,
    sentence-final boundaries and non-Japanese/Latin spacing are preserved.
    """
    changed = 0
    for block in doc.blocks:
        if block.type not in _TEXT_TYPES or not str(block.text or ""):
            continue
        original = str(block.text or "")

        def repl(match: re.Match) -> str:
            left = match.group("left")
            right = match.group("right")
            # Never consume a deliberate paragraph/dialogue boundary after a terminator.
            if left in "。！？!?」』":
                return match.group(0)
            return left + right

        text = _INTERNAL_WRAP_RE.sub(repl, original)
        if text == original:
            continue
        block.ocr_raw = block.ocr_raw or original
        block.text = text
        block.modified_by = _append_modified_by(block.modified_by, "authoritative_internal_wrap")
        changed += 1
    if changed:
        doc.add_log("authoritative_internal_wrap", f"清除 {changed} 个块内 OCR 硬换行", changed)
    return changed


def _layout_canonical_literal(doc: UnifiedDocument) -> str:
    """Canonical body for layout validation, ignoring only proven OCR hard wraps."""
    parts: list[str] = []
    for block in doc.blocks:
        if block.type not in _TEXT_TYPES:
            continue
        text = str(block.text or "")
        # Apply the exact same one-newline Japanese hard-wrap normalization used above.
        previous = None
        while previous != text:
            previous = text
            text = _INTERNAL_WRAP_RE.sub(
                lambda m: m.group(0) if m.group("left") in "。！？!?」』" else m.group("left") + m.group("right"),
                text,
            )
        parts.append(text)
    return "".join(parts)


def _dialogue_depth(text: str) -> int:
    return text.count("「") - text.count("」")


def _clear_wrap_boundary(left: str, right: str) -> bool:
    """High-confidence Japanese continuation boundary used by safe layout."""
    from engine.formatter import CONTINUATION_PREFIXES, _SPLIT_COMPOUND_BOUNDARIES
    l = str(left or "").rstrip(" \t\r\n　")
    r = str(right or "").lstrip(" \t\r\n　")
    if not l or not r:
        return False
    if l[-1] in "。！？!?」』":
        return False
    if r.startswith(CONTINUATION_PREFIXES):
        return True
    if (l[-1], r[0]) in _SPLIT_COMPOUND_BOUNDARIES:
        return True
    # Within an already-open person dialogue, a column break commonly occurs between
    # an ordinary noun/verb and the next Japanese character. Closure must still be
    # found within the bounded look-ahead before this broad signal is accepted.
    return _is_japanese_char(l[-1]) and _is_japanese_char(r[0])


def _find_safe_dialogue_close(blocks: list[Block], start: int, *, max_blocks: int = 4, max_chars: int = 360) -> int | None:
    """Find a nearby closing 」 for a block that starts an unclosed person dialogue."""
    current = str(blocks[start].text or "")
    stripped = current.lstrip(" \t\r\n　")
    if not stripped.startswith("「") or _dialogue_depth(current) != 1:
        return None
    total = len(stripped)
    for end in range(start + 1, min(len(blocks), start + max_blocks + 1)):
        nxt = blocks[end]
        if nxt.type not in {BlockType.PARAGRAPH, BlockType.DIALOGUE}:
            return None
        right = str(nxt.text or "")
        compact = right.strip(" \t\r\n　")
        if not compact or compact.startswith("「"):
            return None
        if not _clear_wrap_boundary(current, right):
            return None
        total += len(compact)
        if total > max_chars:
            return None
        current += right
        depth = _dialogue_depth(current)
        if depth == 0:
            return end
        if depth != 1:
            return None
    return None


def _merge_block_range(blocks: list[Block], start: int, end: int, step: str) -> Block:
    current = copy.deepcopy(blocks[start])
    lineage = list((current.metadata or {}).get("source_block_ids") or [current.id])
    for index in range(start + 1, end + 1):
        nxt = blocks[index]
        current.text = str(current.text or "") + str(nxt.text or "")
        current.ocr_raw = (current.ocr_raw or str(blocks[start].text or "")) + (nxt.ocr_raw or str(nxt.text or ""))
        lineage.extend((nxt.metadata or {}).get("source_block_ids") or [nxt.id])
    current.metadata = {**(current.metadata or {}), "source_block_ids": list(dict.fromkeys(map(str, lineage)))}
    current.modified_by = _append_modified_by(current.modified_by, step)
    return current


def _safe_merge_clear_wraps(doc: UnifiedDocument) -> int:
    """Merge bounded, high-confidence OCR wraps, including split person dialogue.

    Unclosed 「 dialogue is merged only when a matching 」 is found within four
    adjacent text blocks and every boundary looks like Japanese continuation. This
    repairs column wraps without reviving the legacy behaviour that could swallow
    pages after one missing quote.
    """
    from engine.formatter import _should_merge_pair, CHAPTER_RE, BARE_NUMBER_RE

    mergeable = {BlockType.PARAGRAPH, BlockType.DIALOGUE}
    blocks = list(doc.blocks)
    result: list[Block] = []
    merged = 0
    i = 0
    while i < len(blocks):
        current = copy.deepcopy(blocks[i])
        if current.type not in mergeable:
            result.append(current)
            i += 1
            continue

        # Special bounded recovery for a person dialogue split by OCR columns/pages.
        dialogue_end = _find_safe_dialogue_close(blocks, i)
        if dialogue_end is not None:
            current = _merge_block_range(blocks, i, dialogue_end, "authoritative_dialogue_wrap_merge")
            merged += dialogue_end - i
            i = dialogue_end

        while i + 1 < len(blocks):
            nxt = blocks[i + 1]
            left = str(current.text or "")
            right = str(nxt.text or "")
            compact_left = left.strip(" \t\r\n　")
            compact_right = right.strip(" \t\r\n　")
            if nxt.type not in mergeable or not compact_left or not compact_right:
                break
            if CHAPTER_RE.match(compact_left) or CHAPTER_RE.match(compact_right):
                break
            if BARE_NUMBER_RE.match(compact_left) or BARE_NUMBER_RE.match(compact_right):
                break
            # Remaining unbalanced quotes are review boundaries. Balanced 『terms』 and
            # balanced 「dialogue」 do not by themselves forbid a normal wrap merge.
            if _has_unbalanced_quote(left) or _has_unbalanced_quote(right):
                break
            if len(compact_left) + len(compact_right) > 360:
                break
            if not _should_merge_pair(left, right):
                break
            current = _merge_block_range([current, nxt], 0, 1, "authoritative_safe_merge")
            i += 1
            merged += 1
        result.append(current)
        i += 1
    doc.blocks = result
    if merged:
        doc.add_log("authoritative_safe_merge", f"安全合并 {merged} 处明确断列/对白跨列", merged)
    return merged

def _dialogue_spans(text: str) -> list[tuple[int, int]]:
    """Return balanced top-level 「...」 spans. 『...』 is never dialogue."""
    spans: list[tuple[int, int]] = []
    start = None
    depth = 0
    for index, char in enumerate(text):
        if char == "「":
            if depth == 0:
                start = index
            depth += 1
        elif char == "」" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                spans.append((start, index + 1))
                start = None
    if depth:
        return []
    return spans


def _safe_dialogue_layout(doc: UnifiedDocument) -> int:
    """Classify/split only balanced 「人物对白」 while preserving every character."""
    result: list[Block] = []
    changed = 0
    for block in doc.blocks:
        if block.type not in _TEXT_TYPES:
            result.append(block)
            continue
        text = str(block.text or "")
        stripped = text.strip(" \t\r\n　")
        if not stripped:
            result.append(block)
            continue

        # A block beginning with an unclosed dialogue is preserved and flagged. It must
        # not be merged or repaired by inserting punctuation automatically.
        if stripped.startswith("「") and text.count("「") != text.count("」"):
            kept = copy.deepcopy(block)
            kept.type = BlockType.DIALOGUE
            kept.metadata = {**(kept.metadata or {}), "quote_needs_review": True}
            result.append(kept)
            continue

        spans = _dialogue_spans(text)
        if not spans:
            result.append(block)
            continue

        # A complete standalone dialogue only needs a type change; keep literal spaces.
        leading = len(text) - len(text.lstrip(" \t\r\n　"))
        trailing = len(text) - len(text.rstrip(" \t\r\n　"))
        content_end = len(text) - trailing if trailing else len(text)
        if len(spans) == 1 and spans[0] == (leading, content_end):
            dialogue = copy.deepcopy(block)
            dialogue.type = BlockType.DIALOGUE
            dialogue.modified_by = _append_modified_by(dialogue.modified_by, "authoritative_dialogue_layout")
            result.append(dialogue)
            if block.type != BlockType.DIALOGUE:
                changed += 1
            continue

        # Mixed narration/dialogue: only split a quote at block start, after a sentence
        # terminator, or immediately after another complete dialogue. Sentence-internal
        # quotation and every 『术语/内心引用』 remain inside the paragraph.
        accepted: list[tuple[int, int]] = []
        previous_end = None
        for start, end in spans:
            prefix = text[:start]
            prev = prefix.rstrip(" \t\r\n　")[-1:] if prefix.rstrip(" \t\r\n　") else ""
            at_start = not prefix.strip(" \t\r\n　")
            after_sentence = prev in "。！？!?"
            after_dialogue = previous_end is not None and not text[previous_end:start].strip(" \t\r\n　")
            if at_start or after_sentence or after_dialogue:
                accepted.append((start, end))
                previous_end = end
        if not accepted:
            result.append(block)
            continue

        cursor = 0
        pieces: list[tuple[str, BlockType]] = []
        for start, end in accepted:
            if start > cursor:
                pieces.append((text[cursor:start], BlockType.PARAGRAPH))
            pieces.append((text[start:end], BlockType.DIALOGUE))
            cursor = end
        if cursor < len(text):
            pieces.append((text[cursor:], BlockType.PARAGRAPH))
        pieces = [(value, kind) for value, kind in pieces if value]
        if len(pieces) < 2 or "".join(value for value, _ in pieces) != text:
            result.append(block)
            continue

        lineage = list((block.metadata or {}).get("source_block_ids") or [block.id])
        for piece_index, (value, kind) in enumerate(pieces):
            new_block = copy.deepcopy(block)
            if piece_index:
                new_block.id = uuid.uuid4().hex
            new_block.text = value
            new_block.type = kind
            new_block.metadata = {**(new_block.metadata or {}), "source_block_ids": lineage}
            new_block.modified_by = _append_modified_by(new_block.modified_by, "authoritative_dialogue_layout")
            result.append(new_block)
        changed += 1

    doc.blocks = result
    if changed:
        doc.add_log("authoritative_dialogue_layout", f"安全整理 {changed} 个对白块", changed)
    return changed



def _repair_misclassified_term_dialogues(doc: UnifiedDocument) -> int:
    """Undo legacy classification of standalone 『term/inner quote』 as dialogue."""
    changed = 0
    for block in doc.blocks:
        if block.type != BlockType.DIALOGUE:
            continue
        text = str(block.text or "").strip(" \t\r\n　")
        if re.fullmatch(r"『[^』]+』", text, re.DOTALL):
            block.type = BlockType.PARAGRAPH
            block.modified_by = _append_modified_by(block.modified_by, "authoritative_term_type_repair")
            changed += 1
    if changed:
        doc.add_log("authoritative_term_type_repair", f"恢复 {changed} 个被误标为对白的『术语/内心引用』", changed)
    return changed

def _safe_section_types(doc: UnifiedDocument) -> int:
    from engine.formatter import SECTION_RE
    changed = 0
    for block in doc.blocks:
        if block.type != BlockType.PARAGRAPH:
            continue
        if SECTION_RE.match(str(block.text or "").strip()):
            block.type = BlockType.SECTION
            block.modified_by = _append_modified_by(block.modified_by, "authoritative_section_detection")
            changed += 1
    return changed


def apply_lossless_layout(doc: UnifiedDocument) -> tuple[UnifiedDocument, int]:
    """Automatic authoritative layout that is literally character preserving.

    Layout means block boundaries and types only. Paragraph indentation is rendered by
    EPUB CSS (``p.normal { text-indent: 1em }``), never by inserting full-width spaces.
    Quote repair, dash repair, punctuation normalization and fuzzy dedup are excluded.
    """
    source = copy.deepcopy(doc)
    before = _layout_canonical_literal(source)
    result = copy.deepcopy(source)
    unwrap_count = _unwrap_internal_ocr_wraps(result)
    term_type_count = _repair_misclassified_term_dialogues(result)
    merge_count = _safe_merge_clear_wraps(result)
    dialogue_count = _safe_dialogue_layout(result)
    section_count = _safe_section_types(result)
    chapter_count = promote_authoritative_chapters(result)
    after = _layout_canonical_literal(result)
    if after != before:
        source.add_log("lossless_layout_rollback", "安全排版改变了非布局正文字符，已整阶段回滚", 1)
        return source, 0
    total = unwrap_count + term_type_count + merge_count + dialogue_count + section_count + chapter_count
    result.add_log(
        "lossless_authoritative_layout",
        (
            f"自动安全排版：块内硬换行 {unwrap_count}，术语类型恢复 {term_type_count}，"
            f"断列/对白跨列合并 {merge_count}，对白整理 {dialogue_count}，"
            f"分节 {section_count}，章节 {chapter_count}；"
            f"非布局正文字符完全一致，缩进交给 EPUB CSS"
        ),
        total,
    )
    return result, total



_RECONSTRUCTED_SUSPICIOUS_RE = re.compile(
    r"(?:さげす|ほんりゅう|あまた|賛沢|冒渉|閤を|モンスターグ|なぜかかりータ|"
    r"みつともない|あっとという間|やんなざい|[ぁ-んァ-ヶ一-龯]([ぁ-んァ-ヶ])\1{2,})"
)


def _needs_reconstructed_review(block: Block, repair_mode: str = "strict") -> bool:
    if block.type not in {BlockType.PARAGRAPH, BlockType.DIALOGUE}:
        return False
    text = str(block.text or "")
    if repair_mode == "readability":
        return True
    lineage = list((block.metadata or {}).get("source_block_ids") or [])
    return (
        len(lineage) > 1
        or "\n" in text
        or bool((block.metadata or {}).get("quote_needs_review"))
        or bool(_RECONSTRUCTED_SUSPICIOUS_RE.search(text))
    )


def _review_signature(items: list[dict], provider, *, prompt: str, protocol: int, repair_mode: str) -> str:
    payload = {
        "protocol": protocol,
        "repair_mode": repair_mode,
        "model": str(getattr(provider, "model", "") or ""),
        "prompt": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "rows": [[item["type"], item["source_text"]] for item in items],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _run_reconstructed_review(
    provider,
    doc: UnifiedDocument,
    *,
    job_dir: Path,
    temperature: float,
    batch_tokens: int,
    limiter: RequestLimiter,
    usage: _UsageTracker,
    progress_callback=None,
    cancel_check=None,
    resume: bool = True,
    repair_mode: str = "strict",
) -> tuple[UnifiedDocument, list[dict], int, int]:
    """AI-review only complete reconstructed/suspicious logical text units.

    The pass keeps the AiNiee transaction rule: one block owns one slot and no result
    is written unless the whole requested batch passes index/content validation.
    """
    body = _body_blocks(doc)
    body_position = {id(block): index for index, block in enumerate(body)}
    expected: list[dict] = []
    for block in body:
        if not _needs_reconstructed_review(block, repair_mode):
            continue
        ordinal = body_position[id(block)]
        previous = [str(value.text or "") for value in body[max(0, ordinal - 3):ordinal]]
        source_text = str(block.text or "")
        digest = hashlib.sha256(
            (f"review\0{ordinal}\0{block.type.value}\0{source_text}").encode("utf-8")
        ).hexdigest()[:24]
        expected.append({
            "key": f"{ordinal:08d}-{digest}",
            "ordinal": ordinal,
            "source_hash": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
            "source_text": source_text,
            "corrected_text": "",
            "type": block.type.value,
            "type_code": _TYPE_CODE.get(block.type, "u"),
            "block_id": block.id,
            "chapter_index": getattr(block, "chapter_index", None),
            "status": "pending",
            "attempts": 0,
            "error": "",
            "context_before": previous,
        })
    if not expected:
        return copy.deepcopy(doc), [], 0, 0

    review_prompt = READABILITY_REVIEW_PROMPT if repair_mode == "readability" else RECONSTRUCTED_REVIEW_PROMPT
    review_protocol = READABILITY_REVIEW_PROTOCOL if repair_mode == "readability" else RECONSTRUCTED_REVIEW_PROTOCOL
    signature = _review_signature(expected, provider, prompt=review_prompt, protocol=review_protocol, repair_mode=repair_mode)
    review_dir = job_dir / f"reconstructed_review_{repair_mode}_v{review_protocol}_{signature[:20]}"
    state_path = review_dir / "items.json"
    review_dir.mkdir(parents=True, exist_ok=True)
    items, resumed = (
        _load_state(state_path, signature, expected, protocol=review_protocol)
        if resume else (copy.deepcopy(expected), 0)
    )
    _save_state(state_path, signature, provider, items, protocol=review_protocol)
    model = str(getattr(provider, "model", "") or "")
    if repair_mode == "readability":
        batches = _build_readability_batches(items, batch_tokens, review_prompt, model, max_chars=3000)
    else:
        batches = _build_batches(items, min(max(1600, batch_tokens), 8000), review_prompt, model)
    workers = _resolve_workers(provider, len(batches))
    lock = threading.RLock()
    completed = sum(1 for item in items if item["status"] in {"done", "review"})

    def emit(stage: str) -> None:
        if not progress_callback:
            return
        view = usage.snapshot(provider)
        progress_callback({
            "stage": stage,
            "current": completed,
            "total": max(1, len(items)),
            "unit": "sentence",
            "workers": workers,
            "token_total": view["display_total_tokens"],
            "token_actual": view["usage_is_actual"],
            "requests": view.get("provider_requests") or view["requests"],
            "resumed_batches": resumed,
            "checkpoint_dir": str(review_dir),
        })

    emit("恢复句级复核缓存" if resumed else "建立句级复核缓存")

    def commit(indices: list[int], outputs: list[str], status: str, error: str = "") -> None:
        nonlocal completed
        with lock:
            for item_index, output in zip(indices, outputs):
                item = items[item_index]
                if item["status"] in {"done", "review"}:
                    continue
                item["corrected_text"] = output
                item["status"] = status
                item["attempts"] = int(item.get("attempts", 0)) + 1
                item["error"] = error
                completed += 1
            _save_state(state_path, signature, provider, items, protocol=review_protocol)
            emit("句级结果已保存")

    def process(indices: list[int]) -> None:
        if cancel_check and cancel_check():
            raise RuntimeError("AI任务已停止")
        if not indices:
            return
        try:
            outputs = _request_batch(
                provider, items, indices, temperature, limiter, usage,
                cancel_check=cancel_check, prompt_template=review_prompt,
                repair_mode=repair_mode,
            )
            commit(indices, outputs, "done")
        except Exception as exc:
            if isinstance(exc, RuntimeError) and "停止" in str(exc):
                raise
            if len(indices) > 1:
                usage.split()
                mid = len(indices) // 2
                process(indices[:mid])
                process(indices[mid:])
                return
            try:
                outputs = _request_batch(
                    provider, items, indices, temperature, limiter, usage,
                    cancel_check=cancel_check, retry_suffix=True,
                    prompt_template=review_prompt,
                repair_mode=repair_mode,
                )
                commit(indices, outputs, "done")
            except Exception as final_exc:
                usage.singleton_fallback()
                index = indices[0]
                commit(indices, [items[index]["source_text"]], "review", str(final_exc)[:300])

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sentence-review")
    futures = []
    try:
        futures = [executor.submit(process, batch) for batch in batches]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    except Exception:
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    reviewed = copy.deepcopy(doc)
    by_id = {item["block_id"]: item for item in items}
    changes: list[dict] = []
    for block in reviewed.blocks:
        item = by_id.get(block.id)
        if not item:
            continue
        output = item["corrected_text"] or item["source_text"]
        block.metadata = {
            **(block.metadata or {}),
            "sentence_review_status": item["status"],
            "sentence_review_key": item["key"],
        }
        if output != block.text:
            original = str(block.text or "")
            block.ocr_raw = block.ocr_raw or original
            block.text = output
            block.modified_by = _append_modified_by(block.modified_by, "readability_sentence_review" if repair_mode == "readability" else "reconstructed_sentence_review")
            changes.append({"block_id": block.id, "before": original, "after": output})

    recovered, _issues = _recover_indexed_correction_adjacency(doc, reviewed, items)
    if recovered:
        _save_state(state_path, signature, provider, items, protocol=review_protocol)
        changes = [
            change for change in changes
            if (by_id.get(change["block_id"], {}).get("corrected_text") or "")
            != by_id.get(change["block_id"], {}).get("source_text")
        ]
    report = validate_authoritative_transition(doc, reviewed, phase="correction")
    # Every reviewed row has already passed strict per-row full-replacement checks.
    # Whole-document percentage thresholds are unreliable for a tiny candidate set
    # (one deleted duplicate kana can be >4% of a 20-character synthetic document).
    # Keep only structural/protocol/duplication/quote failures as rollback reasons.
    hard_reasons = [
        reason for reason in report.reasons
        if not reason.startswith("字符数量异常") and not reason.startswith("正文相似度过低")
    ]
    if hard_reasons:
        fallback = copy.deepcopy(doc)
        fallback.add_log(
            "reconstructed_sentence_review_rollback",
            "句级 AI 复核未通过结构完整性校验，已只回滚句级复核并保留首轮纠错/排版：" + "；".join(hard_reasons),
            1,
        )
        return fallback, [], sum(1 for item in items if item["status"] == "review"), resumed
    reviewed.add_log(
        "readability_sentence_review" if repair_mode == "readability" else "reconstructed_sentence_review",
        f"复核 {len(items)} 个重组/可疑句，修改 {len(changes)} 个，待复核 {sum(1 for item in items if item['status'] == 'review')} 个",
        len(changes),
    )
    return reviewed, changes, sum(1 for item in items if item["status"] == "review"), resumed



PROBLEM_PATCH_PROMPT = """You are a Japanese light-novel OCR problem-patch proofreader.

INPUT is strict JSON:
{"protocol":"ocr_problem_patch_v2","batch_id":"...","targets":[{"id":"stable id","original_text":"target","previous_text":"read-only context","next_text":"read-only context","block_type":"paragraph|dialogue","issue_types":[...]}]}

Fix only the target text. Context is read-only and must never be copied into revised_text. Correct clear OCR glyph errors, duplicated/missing kana, ruby-reading debris, obvious grammar/conjugation defects, broken words, punctuation, and quote errors. Preserve names, facts, tone, viewpoint and style. Never translate, summarize, rewrite, invent missing story text, merge targets, split targets, or add commentary. If missing source text cannot be uniquely recovered, keep the original and return source_check_required.

Return every target exactly once, in the same order, as compact JSON:
{"protocol":"ocr_problem_patch_v2","batch_id":"same","results":[{"id":"same","original_text":"exact original","revised_text":"complete replacement","status":"fixed|unchanged|source_check_required","issue_types":["ocr_typo|ruby_fragment|duplicate_text|grammar|broken_word|broken_sentence|quote|punctuation|mixed_dialogue|missing_source_text|advertisement_noise"],"confidence":0.0}]}

Rules: fixed must actually change text; unchanged/source_check_required must preserve original_text exactly. No Markdown or extra keys.
INPUT:\n{{INPUT}}"""


READABILITY_PROBLEM_PATCH_PROMPT = """You are a Japanese light-novel OCR readability repairer.

INPUT is strict JSON:
{"protocol":"ocr_problem_patch_v2","batch_id":"...","targets":[{"id":"stable id","original_text":"target","previous_text":"read-only context","next_text":"read-only context","block_type":"paragraph|dialogue","issue_types":[...]}]}

Make each target read as natural Japanese light-novel prose. Fix OCR glyph errors, furigana/ruby debris, duplicated or missing kana, broken words, punctuation, quotes, particles, conjugation and awkward OCR-caused grammar. You MAY infer a short missing word or clause from previous_text/next_text when it is strongly supported. Preserve the existing plot, facts, names, character relationships, viewpoint, personality, humour and tone. Never create a new event, new dialogue, or unrelated description. Context is read-only: do not copy an entire neighbouring sentence into revised_text. Keep one result per target; never merge, split, omit or reorder targets.

Return every target exactly once as compact JSON:
{"protocol":"ocr_problem_patch_v2","batch_id":"same","results":[{"id":"same","original_text":"exact original","revised_text":"complete repaired target","status":"fixed|unchanged|source_check_required","issue_types":["ocr_typo|ruby_fragment|duplicate_text|grammar|broken_word|broken_sentence|quote|punctuation|mixed_dialogue|missing_source_text|advertisement_noise"],"confidence":0.0}]}

Use fixed when you can produce a coherent repair. Use unchanged when no repair is needed. source_check_required is allowed only when even a readability repair would require inventing a new event or fact. No Markdown or extra keys.
INPUT:\n{{INPUT}}"""

_PROBLEM_PATCH_RETRY = """\nThe previous reply failed validation. Return the exact schema, every target once in order. Do not copy context. If uncertain, use source_check_required and preserve original text.\n"""

_PROBLEM_ALLOWED_STATUS = {"fixed", "unchanged", "source_check_required"}
_PROBLEM_ALLOWED_ISSUES = {
    "ocr_typo", "ruby_fragment", "duplicate_text", "grammar", "broken_word",
    "broken_sentence", "quote", "punctuation", "mixed_dialogue",
    "missing_source_text", "advertisement_noise",
}
_PROBLEM_PATTERNS = {
    "さげすもちろん": ("ruby_fragment", "ocr_typo"),
    "まさにしく": ("ocr_typo", "grammar"),
    "それによって更なる。": ("broken_sentence", "missing_source_text"),
    "たた褒": ("broken_sentence", "missing_source_text", "ocr_typo"),
    "待ってよりガーラス": ("ocr_typo", "punctuation"),
    "帰ってくれてくれ": ("duplicate_text", "grammar"),
    "踵俺は踵": ("ruby_fragment", "duplicate_text"),
    "みつともない": ("ocr_typo",),
    "むさぼ「": ("ruby_fragment", "mixed_dialogue"),
    "「あれ」。これは": ("quote", "punctuation"),
    "めざわ": ("ruby_fragment",), "さんれい": ("ruby_fragment",),
    "がくぜん": ("ruby_fragment",), "咆哮をほうこう": ("ruby_fragment", "duplicate_text"),
    "どくろ髑髏": ("ruby_fragment", "duplicate_text"),
    "りようがきょうじん": ("ruby_fragment",), "とりこ「": ("ruby_fragment", "mixed_dialogue"),
    "戻りはんすうつつ": ("ruby_fragment",), "ふんまん憤懣": ("ruby_fragment", "duplicate_text"),
    "ぼくねんじん": ("ruby_fragment",), "へつづく": ("advertisement_noise",),
    "Bmmu": ("advertisement_noise",),
}
_PROBLEM_PROTOCOL_LEAK = re.compile(r"(?i)^(?:[pcdsrftu]|b\d{1,6}|row\d{1,6}|target\d{1,6})$")


def _problem_quote_delta(text: str) -> tuple[int, int]:
    return text.count("「") - text.count("」"), text.count("『") - text.count("』")


def _problem_normalize(text: str) -> str:
    return re.sub(r"[\s　]+", "", str(text or ""))


def _problem_detect(block: Block) -> list[str]:
    if block.type not in {BlockType.PARAGRAPH, BlockType.DIALOGUE}:
        return []
    text = str(block.text or "")
    metadata = block.metadata or {}
    issues: set[str] = set()
    if metadata.get("sentence_review_status") == "review":
        issues.add("broken_sentence")
    if metadata.get("indexed_status") == "review":
        issues.add("ocr_typo")
    if metadata.get("quote_needs_review") is True or _has_unbalanced_quote(text):
        issues.add("quote")
    for pattern, names in _PROBLEM_PATTERNS.items():
        if pattern in text:
            issues.update(names)
    return sorted(issues)


def _problem_signature(items: list[dict], provider, *, repair_mode: str = "strict") -> str:
    prompt = READABILITY_PROBLEM_PATCH_PROMPT if repair_mode == "readability" else PROBLEM_PATCH_PROMPT
    payload = {
        "protocol": PROBLEM_PATCH_PROTOCOL,
        "repair_mode": repair_mode,
        "model": str(getattr(provider, "model", "") or ""),
        "prompt": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "rows": [[x["id"], x["source_text"], x["issues"]] for x in items],
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _problem_load(path: Path, signature: str, expected: list[dict]) -> tuple[list[dict], int]:
    if not path.exists():
        return copy.deepcopy(expected), 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol") != PROBLEM_PATCH_PROTOCOL or payload.get("signature") != signature:
            return copy.deepcopy(expected), 0
        saved = payload.get("items")
        if not isinstance(saved, list) or len(saved) != len(expected):
            return copy.deepcopy(expected), 0
        restored, resumed = [], 0
        for exp, got in zip(expected, saved):
            item = copy.deepcopy(exp)
            if not isinstance(got, dict) or got.get("key") != exp["key"] or got.get("source_hash") != exp["source_hash"]:
                restored.append(item); continue
            status = str(got.get("status", "pending"))
            revised = str(got.get("revised_text", "") or "")
            if status in _PROBLEM_ALLOWED_STATUS and revised:
                item.update(status=status, revised_text=revised, confidence=float(got.get("confidence", 0) or 0),
                            returned_issues=list(got.get("returned_issues") or []), error=str(got.get("error", "") or ""),
                            attempts=int(got.get("attempts", 0) or 0))
                resumed += 1
            restored.append(item)
        return restored, resumed
    except Exception:
        return copy.deepcopy(expected), 0


def _problem_save(path: Path, signature: str, provider, items: list[dict]) -> None:
    _atomic_json(path, {"protocol": PROBLEM_PATCH_PROTOCOL, "signature": signature,
                        "model": str(getattr(provider, "model", "") or ""), "updated_at": time.time(), "items": items})


def _problem_context_copied(revised: str, context: str) -> bool:
    revised_n, context_n = _problem_normalize(revised), _problem_normalize(context)
    if len(context_n) < 24:
        return False
    if revised_n == context_n:
        return True
    for length in (80, 60, 40, 24):
        if len(context_n) >= length and (context_n[:length] in revised_n or context_n[-length:] in revised_n):
            return True
    return False


def _problem_validate_result(item: dict, result: dict, *, repair_mode: str = "strict") -> None:
    if not isinstance(result, dict) or set(result) - {"id", "original_text", "revised_text", "status", "issue_types", "confidence"}:
        raise ValueError("result_schema")
    if str(result.get("id", "")) != item["id"] or result.get("original_text") != item["source_text"]:
        raise ValueError("identity")
    revised = result.get("revised_text")
    status = result.get("status")
    issues = result.get("issue_types")
    confidence = result.get("confidence")
    if not isinstance(revised, str) or not revised.strip() or status not in _PROBLEM_ALLOWED_STATUS:
        raise ValueError("status_or_text")
    if not isinstance(issues, list) or any(x not in _PROBLEM_ALLOWED_ISSUES for x in issues):
        raise ValueError("issues")
    if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
        raise ValueError("confidence")
    if status in {"unchanged", "source_check_required"} and revised != item["source_text"]:
        raise ValueError("uncertain_changed")
    if status == "fixed" and revised == item["source_text"]:
        raise ValueError("fixed_unchanged")
    if _PROBLEM_PROTOCOL_LEAK.fullmatch(revised.strip()):
        raise ValueError("protocol_leak")
    if _problem_context_copied(revised, item["previous_text"]) or _problem_context_copied(revised, item["next_text"]):
        raise ValueError("context_copy")
    if status == "fixed":
        old, new = _problem_normalize(item["source_text"]), _problem_normalize(revised)
        delta = abs(len(new) - len(old))
        if repair_mode == "readability":
            if delta > max(80, int(len(old) * 0.85)):
                raise ValueError("length")
            if len(old) >= 10:
                ratio = difflib.SequenceMatcher(None, old, new, autojunk=False).ratio()
                if ratio < 0.20:
                    raise ValueError("low_similarity")
        else:
            if delta > max(20, int(len(old) * 0.35)):
                raise ValueError("length")
            if len(old) >= 12:
                ratio = difflib.SequenceMatcher(None, old, new, autojunk=False).ratio()
                if ratio < 0.48:
                    raise ValueError("low_similarity")
        if "quote" in set(item["issues"]) | set(issues):
            before = sum(abs(x) for x in _problem_quote_delta(item["source_text"]))
            after = sum(abs(x) for x in _problem_quote_delta(revised))
            if after > before:
                raise ValueError("quote_worse")


def _problem_request(provider, items: list[dict], indices: list[int], temperature: float,
                     limiter: RequestLimiter, usage: _UsageTracker, *, retry=False, cancel_check=None,
                     repair_mode: str = "strict") -> list[dict]:
    batch_id = "patch_" + hashlib.sha256("|".join(items[i]["key"] for i in indices).encode()).hexdigest()[:16]
    targets = []
    for i in indices:
        item = items[i]
        targets.append({"id": item["id"], "original_text": item["source_text"],
                        "previous_text": item["previous_text"], "next_text": item["next_text"],
                        "block_type": item["type"], "issue_types": item["issues"]})
    payload = {"protocol": "ocr_problem_patch_v2", "batch_id": batch_id, "targets": targets}
    prompt_template = READABILITY_PROBLEM_PATCH_PROMPT if repair_mode == "readability" else PROBLEM_PATCH_PROMPT
    prompt = prompt_template.replace("{{INPUT}}", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    if retry:
        prompt += _PROBLEM_PATCH_RETRY
    parsed = _call_and_parse(provider, prompt, temperature, api_retries=1 if not retry else 0,
                             cancel_check=cancel_check, limiter=limiter, usage=usage)
    if not isinstance(parsed, dict) or parsed.get("protocol") != "ocr_problem_patch_v2" or parsed.get("batch_id") != batch_id:
        raise ValueError("envelope")
    results = parsed.get("results")
    if not isinstance(results, list) or len(results) != len(indices):
        raise ValueError("result_count")
    for item_index, result in zip(indices, results):
        _problem_validate_result(items[item_index], result, repair_mode=repair_mode)
    return results


def _run_problem_patch_review(provider, doc: UnifiedDocument, *, job_dir: Path, temperature: float,
                              batch_tokens: int, limiter: RequestLimiter, usage: _UsageTracker,
                              progress_callback=None, cancel_check=None, resume=True,
                              repair_mode: str = "strict"):
    body = _body_blocks(doc)
    expected = []
    for ordinal, block in enumerate(body):
        issues = _problem_detect(block)
        if not issues:
            continue
        source_text = str(block.text or "")
        digest = hashlib.sha256(f"patch\0{ordinal}\0{block.type.value}\0{source_text}".encode()).hexdigest()[:24]
        expected.append({"key": f"{ordinal:08d}-{digest}", "id": str(block.id), "block_id": block.id,
                         "source_hash": hashlib.sha256(source_text.encode()).hexdigest(), "source_text": source_text,
                         "previous_text": str(body[ordinal-1].text or "") if ordinal > 0 else "",
                         "next_text": str(body[ordinal+1].text or "") if ordinal + 1 < len(body) else "",
                         "type": block.type.value, "issues": issues, "status": "pending", "revised_text": "",
                         "returned_issues": [], "confidence": 0.0, "attempts": 0, "error": ""})
    if not expected:
        return copy.deepcopy(doc), [], 0, 0, 0
    signature = _problem_signature(expected, provider, repair_mode=repair_mode)
    patch_dir = job_dir / f"problem_patch_{repair_mode}_v{PROBLEM_PATCH_PROTOCOL}_{signature[:20]}"
    patch_dir.mkdir(parents=True, exist_ok=True)
    state_path = patch_dir / "items.json"
    items, resumed = _problem_load(state_path, signature, expected) if resume else (copy.deepcopy(expected), 0)
    _problem_save(state_path, signature, provider, items)
    pending = [i for i,x in enumerate(items) if x["status"] == "pending"]
    batches = [pending[i:i+8] for i in range(0, len(pending), 8)]
    workers = _resolve_workers(provider, len(batches))
    lock = threading.RLock()
    completed = len(items) - len(pending)

    def emit(stage):
        if progress_callback:
            view=usage.snapshot(provider)
            progress_callback({"stage":stage,"current":completed,"total":len(items),"unit":"problem",
                               "workers":workers,"token_total":view["display_total_tokens"],"token_actual":view["usage_is_actual"],
                               "requests":view.get("provider_requests") or view["requests"],"resumed_batches":resumed,
                               "checkpoint_dir":str(patch_dir)})
    emit("恢复问题句补丁缓存" if resumed else "建立问题句补丁缓存")

    def commit(indices, results=None, fallback_error=""):
        nonlocal completed
        with lock:
            if results is None:
                results=[{"revised_text":items[i]["source_text"],
                          "status":"unchanged" if repair_mode == "readability" else "source_check_required",
                          "issue_types":items[i]["issues"],"confidence":0.0} for i in indices]
            for i,result in zip(indices,results):
                if items[i]["status"] != "pending": continue
                items[i].update(revised_text=result["revised_text"], status=result["status"],
                                returned_issues=list(result.get("issue_types") or []), confidence=float(result.get("confidence",0) or 0),
                                attempts=int(items[i].get("attempts",0))+1, error=fallback_error)
                completed += 1
            _problem_save(state_path, signature, provider, items); emit("问题句补丁已保存")

    def process(indices):
        if not indices: return
        try:
            commit(indices, _problem_request(provider, items, indices, temperature, limiter, usage, cancel_check=cancel_check, repair_mode=repair_mode))
        except Exception as exc:
            if isinstance(exc, RuntimeError) and "停止" in str(exc): raise
            if len(indices)>1:
                usage.split(); mid=len(indices)//2; process(indices[:mid]); process(indices[mid:]); return
            try:
                commit(indices, _problem_request(provider, items, indices, temperature, limiter, usage, retry=True, cancel_check=cancel_check, repair_mode=repair_mode))
            except Exception as final_exc:
                usage.singleton_fallback(); commit(indices, None, str(final_exc)[:300])

    executor=concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="problem-patch")
    futures=[]
    try:
        futures=[executor.submit(process,b) for b in batches]
        for f in concurrent.futures.as_completed(futures): f.result()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    output=copy.deepcopy(doc); by_id={x["block_id"]:x for x in items}; changes=[]; pending_count=0; fixed_count=0
    for block in output.blocks:
        item=by_id.get(block.id)
        if not item: continue
        md={**(block.metadata or {}), "ocr_patch_status":item["status"], "ocr_patch_issue_types":item["returned_issues"] or item["issues"],
            "ocr_patch_confidence":item["confidence"], "ocr_patch_key":item["key"]}
        if item["status"]=="fixed" and item["revised_text"] != block.text:
            original=str(block.text or ""); block.ocr_raw=block.ocr_raw or original; block.text=item["revised_text"]
            block.modified_by=_append_modified_by(block.modified_by,"ocr_problem_patch"); md["sentence_review_status"]="done"
            if not _has_unbalanced_quote(block.text): md["quote_needs_review"]=False
            changes.append({"block_id":block.id,"before":original,"after":block.text}); fixed_count+=1
        elif item["status"]=="source_check_required":
            md["source_check_required"]=True; pending_count+=1
        block.metadata=md
    # Locally roll back only patch rows that introduce a new adjacent duplicate/coverage.
    source_body = _body_blocks(doc)
    output_body = _body_blocks(output)
    recovered = 0
    if len(source_body) == len(output_body):
        item_by_id = {str(x["block_id"]): x for x in items}
        for index in range(len(output_body) - 1):
            issue = _adjacent_pair_issue(output_body[index], output_body[index + 1])
            if not issue or _adjacent_pair_issue(source_body[index], source_body[index + 1]):
                continue
            for pos in (index, index + 1):
                src, out = source_body[pos], output_body[pos]
                if str(src.text or "") == str(out.text or ""):
                    continue
                item = item_by_id.get(str(out.id))
                out.text, out.type, out.ocr_raw = src.text, src.type, src.ocr_raw
                out.modified_by = _append_modified_by(out.modified_by, "ocr_problem_patch_recovery")
                out.metadata = {**(src.metadata or {}), "ocr_patch_status": "source_check_required",
                                "source_check_required": True, "ocr_patch_recovery_reason": issue}
                if item is not None:
                    item.update(status="source_check_required", revised_text=item["source_text"], confidence=0.0, error=f"introduced_adjacent_{issue}")
                recovered += 1
        if recovered:
            _problem_save(state_path,signature,provider,items)
            pending_count += recovered
            fixed_count = max(0, fixed_count - recovered)
            changes = [c for c in changes if str(next((b.text for b in output_body if b.id == c["block_id"]), "")) != c["before"]]
            output.add_log("ocr_problem_patch_recovery", f"局部回退 {recovered} 个会新增相邻重复/覆盖的问题句补丁", recovered)
    output.add_log("ocr_problem_patch",
                   (f"可读性问题句复核 {len(items)} 项，修复 {fixed_count} 项，保持原文/待确认 {pending_count} 项"
                    if repair_mode == "readability"
                    else f"问题句补丁复核 {len(items)} 项，修复 {fixed_count} 项，需扫描页核对 {pending_count} 项"),
                   fixed_count)
    return output,changes,pending_count,resumed,len(items)


def run_indexed_authoritative(
    provider,
    doc: UnifiedDocument,
    *,
    progress_callback=None,
    cancel_check=None,
    checkpoint_dir: str | os.PathLike | None = None,
    resume: bool = True,
    repair_mode: str = "strict",
) -> tuple[UnifiedDocument, list[dict], object]:
    """Create authoritative text with indexed safety and strict/readability OCR repair."""
    repair_mode = str(repair_mode or "strict").lower()
    if repair_mode not in {"strict", "readability"}:
        repair_mode = "readability"
    source = sanitize_authoritative_source(doc)
    source = copy.deepcopy(source)
    apply_conservative_ocr_fixes(source)
    remove_demonstrable_duplicate_runs(source)
    body = _body_blocks(source)
    source_positions = {id(block): i for i, block in enumerate(source.blocks)}
    expected: list[dict] = []
    for ordinal, block in enumerate(body):
        key = _stable_item_key(ordinal, block)
        expected.append({
            "key": key,
            "ordinal": ordinal,
            "source_hash": hashlib.sha256(str(block.text or "").encode("utf-8")).hexdigest(),
            "source_text": str(block.text or ""),
            "corrected_text": "",
            "type": block.type.value,
            "type_code": _TYPE_CODE.get(block.type, "u"),
            "block_id": block.id,
            "status": "pending",
            "attempts": 0,
            "error": "",
        })

    signature = _canonical_signature(source, provider, INDEXED_PROOFREAD_PROMPT)
    checkpoint_root = Path(checkpoint_dir or (Path.home() / ".novel_formatter" / "ai_checkpoints")).expanduser()
    job_dir = checkpoint_root / f"indexed_v{PROTOCOL_VERSION}_{signature[:20]}"
    state_path = job_dir / "items.json"
    job_dir.mkdir(parents=True, exist_ok=True)
    items, resumed = _load_state(state_path, signature, expected) if resume else (copy.deepcopy(expected), 0)
    _save_state(state_path, signature, provider, items)

    kwargs = getattr(provider, "kwargs", {}) or {}
    model = str(getattr(provider, "model", "") or "")
    temperature = float(kwargs.get("temperature", 0.1) or 0.1)
    configured = int(kwargs.get("ai_batch_tokens", 0) or 0)
    batch_tokens = min(configured, 12000) if configured > 0 else 8000
    batches = _build_batches(items, batch_tokens, INDEXED_PROOFREAD_PROMPT, model)
    workers = _resolve_workers(provider, len(batches))
    limiter = RequestLimiter(int(kwargs.get("rpm_limit", 0) or 0), int(kwargs.get("tpm_limit", 0) or 0))
    usage = _UsageTracker()
    state_lock = threading.RLock()
    completed = sum(1 for item in items if item["status"] in {"done", "review"})
    started = time.monotonic()

    def emit(stage: str):
        if not progress_callback:
            return
        view = usage.snapshot(provider)
        elapsed = max(0.001, time.monotonic() - started)
        rate = max(0, completed - resumed) / elapsed
        progress_callback({
            "stage": stage,
            "current": completed,
            "total": max(1, len(items)),
            "unit": "row",
            "workers": workers,
            "rate": rate,
            "eta_seconds": ((len(items) - completed) / rate) if rate > 0 else None,
            "token_total": view["display_total_tokens"],
            "token_actual": view["usage_is_actual"],
            "requests": view.get("provider_requests") or view["requests"],
            "resumed_batches": resumed,
            "checkpoint_dir": str(job_dir),
        })

    emit("恢复逐条缓存" if resumed else "建立逐条缓存")

    def commit(indices: list[int], outputs: list[str], status: str, error: str = "") -> None:
        nonlocal completed
        with state_lock:
            for idx, output in zip(indices, outputs):
                item = items[idx]
                if item["status"] in {"done", "review"}:
                    continue
                item["corrected_text"] = output
                item["status"] = status
                item["attempts"] = int(item.get("attempts", 0)) + 1
                item["error"] = error
                completed += 1
            _save_state(state_path, signature, provider, items)
            emit("逐条结果已保存")

    def process(indices: list[int], depth: int = 0) -> None:
        if cancel_check and cancel_check():
            raise RuntimeError("AI任务已停止")
        if not indices:
            return
        try:
            outputs = _request_batch(
                provider, items, indices, temperature, limiter, usage,
                cancel_check=cancel_check, retry_suffix=False,
            )
            commit(indices, outputs, "done")
            return
        except Exception as exc:
            if isinstance(exc, RuntimeError) and "停止" in str(exc):
                raise
            if len(indices) > 1:
                usage.split()
                mid = len(indices) // 2
                process(indices[:mid], depth + 1)
                process(indices[mid:], depth + 1)
                return
            # One strict retry for a singleton. If it still fails, preserve the source
            # and mark it for review. No invalid model output ever enters the document.
            try:
                outputs = _request_batch(
                    provider, items, indices, temperature, limiter, usage,
                    cancel_check=cancel_check, retry_suffix=True,
                )
                commit(indices, outputs, "done")
            except Exception as final_exc:
                usage.singleton_fallback()
                idx = indices[0]
                commit(indices, [items[idx]["source_text"]], "review", str(final_exc)[:300])

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="indexed-proofread")
    futures = []
    try:
        futures = [executor.submit(process, batch) for batch in batches]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    except Exception:
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    corrected = copy.deepcopy(source)
    by_id = {item["block_id"]: item for item in items}
    changes: list[dict] = []
    for block in corrected.blocks:
        item = by_id.get(block.id)
        if not item:
            continue
        output = item["corrected_text"] or item["source_text"]
        block.metadata = {
            **(block.metadata or {}),
            "source_block_ids": list((block.metadata or {}).get("source_block_ids") or [block.id]),
            "indexed_status": item["status"],
            "indexed_item_key": item["key"],
        }
        if output != block.text:
            original = block.text
            block.ocr_raw = block.ocr_raw or original
            block.text = output
            block.modified_by = "indexed_ai_proofread"
            changes.append({"block_id": block.id, "before": original, "after": output})

    post_fix_count = apply_post_indexed_ocr_fixes(corrected)

    recovered_adjacent_rows, recovered_adjacent_issues = _recover_indexed_correction_adjacency(
        source, corrected, items
    )
    if recovered_adjacent_rows:
        # Persist the local rollback into the item cache. Re-running the same book will
        # resume these rows as review/source text and will not spend tokens again.
        _save_state(state_path, signature, provider, items)
        changes = [
            change for change in changes
            if (by_id.get(change["block_id"], {}).get("corrected_text") or
                by_id.get(change["block_id"], {}).get("source_text"))
               != by_id.get(change["block_id"], {}).get("source_text")
        ]

    correction_report = validate_authoritative_transition(source, corrected, phase="correction")
    if not correction_report.passed:
        # Structural/protocol corruption remains a hard failure. A local adjacent
        # duplicate is recovered above and can no longer discard an otherwise valid
        # paid full-book run.
        raise RuntimeError("逐条 AI 纠错校验失败：" + "；".join(correction_report.reasons))

    emit("本地句子重组与安全排版")
    laid_out, first_layout_changes = apply_lossless_layout(corrected)

    # Re-segmentation can expose an exact duplicated passage that was already present
    # in the OCR/Markdown source but hidden by different block boundaries (for example,
    # narration+dialogue joined in one copy and split in the other).  The literal layout
    # did not create any characters; remove only the later copy of demonstrable exact
    # runs after boundaries have been normalized, then validate the actual result.
    first_layout_dedup_runs = remove_demonstrable_duplicate_runs(laid_out)
    first_layout_report = validate_authoritative_transition(corrected, laid_out, phase="layout")
    if first_layout_dedup_runs:
        # Exact-run deletion is independently proven by four-or-more identical blocks
        # and a 160-character minimum. In a short document that legitimate deletion can
        # exceed global ratio thresholds, so do not let percentage heuristics override
        # the stronger deterministic proof. All other integrity failures remain fatal.
        first_layout_report.reasons = [
            reason for reason in first_layout_report.reasons
            if not reason.startswith("字符数量异常") and not reason.startswith("正文相似度过低")
        ]
        first_layout_report.passed = not first_layout_report.reasons
        if first_layout_report.passed and first_layout_report.warnings:
            first_layout_report.publish_ready = False
    if not first_layout_report.passed:
        raise RuntimeError("本地安全排版完整性校验失败：" + "；".join(first_layout_report.reasons))

    emit("章节语义可读性修复" if repair_mode == "readability" else "重组句级 AI 复核")
    reviewed, sentence_changes, sentence_review_count, sentence_resumed = _run_reconstructed_review(
        provider, laid_out, job_dir=job_dir, temperature=temperature, batch_tokens=batch_tokens,
        limiter=limiter, usage=usage, progress_callback=progress_callback,
        cancel_check=cancel_check, resume=resume, repair_mode=repair_mode,
    )
    changes.extend(sentence_changes)

    emit("残留问题句可读性修复" if repair_mode == "readability" else "问题句补丁 AI 复核")
    reviewed, patch_changes, source_check_count, patch_resumed, patch_target_count = _run_problem_patch_review(
        provider, reviewed, job_dir=job_dir, temperature=temperature, batch_tokens=batch_tokens,
        limiter=limiter, usage=usage, progress_callback=progress_callback, cancel_check=cancel_check, resume=resume,
        repair_mode=repair_mode,
    )
    changes.extend(patch_changes)

    # Sentence/problem review may repair a quote/punctuation and make a previously unsplittable
    # logical unit classifiable. Re-run the character-safe layout idempotently.
    result, second_layout_changes = apply_lossless_layout(reviewed)
    second_layout_dedup_runs = remove_demonstrable_duplicate_runs(result)
    layout_changes = first_layout_changes + second_layout_changes
    layout_dedup_runs = first_layout_dedup_runs + second_layout_dedup_runs
    report = validate_authoritative_transition(corrected, result, phase="layout")
    # First-pass rows and sentence-review rows were each validated transactionally.
    # A small legitimate OCR correction can exceed a global percentage threshold in
    # short documents; it is not structural corruption. Preserve all other reasons.
    remaining_reasons = [
        reason for reason in report.reasons
        if not reason.startswith("字符数量异常") and not reason.startswith("正文相似度过低")
    ]
    if len(remaining_reasons) != len(report.reasons):
        report.reasons = remaining_reasons
        report.passed = not remaining_reasons
        if report.passed and report.warnings:
            report.publish_ready = False
    review_count = sum(1 for item in items if item["status"] == "review")
    if review_count:
        report.warnings.append(f"有 {review_count} 个原始单条未通过 AI 校验，已保留原文并标记待复核")
        report.publish_ready = False
    if sentence_review_count:
        if repair_mode == "readability":
            report.warnings.append(f"有 {sentence_review_count} 个正文单元未通过可读性返回校验，已保留进入本轮前文本")
        else:
            report.warnings.append(f"有 {sentence_review_count} 个重组句未通过第一轮句级校验，已转入问题句补丁复核")
        report.publish_ready = False
    if source_check_count:
        report.warnings.append(
            f"有 {source_check_count} 个问题句仍无法在不编造新事件的前提下修复，已保留原文"
            if repair_mode == "readability"
            else f"有 {source_check_count} 个问题句信息不足，必须核对扫描页"
        )
        report.publish_ready = False
    result = mark_authoritative(result, report)
    result.metadata.authoritative_indexed_protocol = PROTOCOL_VERSION
    result.metadata.authoritative_checkpoint_dir = str(job_dir)
    result.metadata.ocr_repair_mode = repair_mode
    usage_view = usage.snapshot(provider)
    result.add_log(
        "indexed_authoritative_ai",
        (
            f"AiNiee-style indexed proofreading; protocol={PROTOCOL_VERSION}; repair_mode={repair_mode}; rows={len(items)}; "
            f"changed={len(changes)}; deterministic_post_fixes={post_fix_count}; row_review={review_count}; "
            f"sentence_review={sentence_review_count}; problem_patch_targets={patch_target_count}; "
            f"problem_patch_changes={len(patch_changes)}; source_check={source_check_count}; "
            f"resumed_rows={resumed}; resumed_sentences={sentence_resumed}; resumed_problem_patches={patch_resumed}; "
            f"requests={usage_view['requests']}; split_events={usage_view['split_events']}; "
            f"singleton_fallbacks={usage_view['singleton_fallbacks']}; "
            f"tokens={usage_view['display_total_tokens']}; lossless_layout_changes={layout_changes}; "
            f"layout_exposed_duplicate_runs_removed={layout_dedup_runs}; checkpoint={job_dir}"
        ),
        len(changes) + post_fix_count + review_count + sentence_review_count + source_check_count,
    )
    emit("完成")
    return result, changes, report
