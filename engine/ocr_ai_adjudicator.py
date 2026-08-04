# -*- coding: utf-8 -*-
"""Low-token, evidence-ordered AI adjudication for multi-model OCR packages.

The module operates on ``novel_formatter.ocr_roundtrip.v1`` multi-model packages.
It never asks the model to rewrite the full package.  Only risky rows are sent as
compact records; immutable IDs, page geometry, column lineage, assets and EPUB
structure stay local and are validated by the normal round-trip importer.

Two independent passes are used:

1. adjudicator: returns sparse text patches only;
2. auditor: sees the proposed text but not the adjudicator's explanation and
   returns failures only.

Rows rejected by local safety guards or the auditor are left unchanged and are
reported as ``low_uncertain`` instead of being silently applied.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Iterable, Sequence

from ai.token_counter import estimate_tokens
from engine.ai_document_processor import parse_json_reply
from engine.reference_assist import ReferenceCorpus, normalise_reference_content


SCHEMA = "novel_formatter.ocr_ai_adjudication.v1"
AUDIT_SCHEMA = "novel_formatter.ocr_ai_audit.v1"
_ALLOWED_CONFIDENCE = {
    "exact_reference",
    "high_consensus",
    "medium_context",
    "low_uncertain",
}
_ALLOWED_CHANGE_TYPES = {
    "unchanged",
    "single_char",
    "kana",
    "punctuation",
    "missing_text",
    "duplicate",
    "cross_column",
    "order",
    "title_noise",
    "other",
}
_JAPANESE_RE = re.compile(r"[ぁ-ゖァ-ヺー々〆ヵヶ]")
_CJK_RE = re.compile(r"[一-龯豈-﫿]")
_NOISE_RE = re.compile(r"[□�]|(?:[A-Za-z]{5,})|(?:\d{5,})")
_RUBY_PIPE_RE = re.compile(r"([^\s|]{1,24})\|[ぁ-ゖァ-ヺー]{1,24}")
_SPACE_RE = re.compile(r"[\s\u3000]+")
_SYMBOL_FOLD = str.maketrans({
    "“": "「", "”": "」", "‘": "『", "’": "』",
    "﹁": "「", "﹂": "」", "﹃": "『", "﹄": "』",
})


@dataclass(slots=True)
class AdjudicationOptions:
    batch_pages: int = 20
    overlap_pages: int = 2
    risk_only: bool = True
    independent_audit: bool = True
    request_retries: int = 2
    max_prompt_tokens: int = 12000
    reference_min_score: float = 0.82
    reference_strong_score: float = 0.96
    auto_apply_medium: bool = True

    def normalised(self) -> "AdjudicationOptions":
        return AdjudicationOptions(
            batch_pages=max(10, min(30, int(self.batch_pages or 20))),
            overlap_pages=max(0, min(4, int(self.overlap_pages or 2))),
            risk_only=bool(self.risk_only),
            independent_audit=bool(self.independent_audit),
            request_retries=max(0, min(2, int(self.request_retries or 0))),
            max_prompt_tokens=max(3000, min(48000, int(self.max_prompt_tokens or 12000))),
            reference_min_score=max(0.70, min(0.99, float(self.reference_min_score or 0.82))),
            reference_strong_score=max(0.85, min(0.999, float(self.reference_strong_score or 0.96))),
            auto_apply_medium=bool(self.auto_apply_medium),
        )


@dataclass(slots=True)
class ReferenceSource:
    path: str = ""
    label: str = ""
    fingerprint: str = ""
    line_count: int = 0
    corpus: ReferenceCorpus | None = None

    @property
    def available(self) -> bool:
        return self.corpus is not None and self.line_count > 0


class AdjudicationCancelled(RuntimeError):
    pass


class AdjudicationProtocolError(ValueError):
    pass


ADJUDICATOR_PROMPT = r"""You are a Japanese commercial-publication OCR adjudicator. Restore the text that is actually supported by evidence; never polish, translate, continue, summarize, modernize, or invent.
Evidence order: R=aligned publication reference > agreement of 2/3 OCRs > physical columns > neighbouring blocks/pages > Japanese grammar/terminology > one OCR.
Only IDs in t are editable. x is read-only context. Preserve wording, names, numbers, negation and exact symbols. Distinguish 一/ー/―/—/─/‐/－, …/‥/・・・, all quote/bracket forms, small kana, dakuten and full/half width. Delete repetition only with at least two independent signals. A shorter OCR must not erase a supported sentence/column; a longer OCR must be checked for duplication or neighbour-column adhesion.
When R is confidently aligned, reproduce the matching reference text exactly. When evidence is insufficient, do not guess: put the ID in q.
Input compact keys: t=[{i:short integer alias,p:page,y:type,o:current,c:[[models,text,max_confidence,agreement_count]],v:[[models,[column texts]]],r:[score,line1,line2,reference excerpt] or null,b:previous text,a:next text,k:risk codes}], x={b:previous-overlap anchors,a:next-overlap anchors}, g:[[term,preferred]], m:reference|ocr_only.
The i value is the only editable identifier. Copy that exact integer alias into every u/q entry; never invent an ID and never use the array position unless it equals i.
Return one compact JSON object only. Omit unchanged rows. Schema:
{"u":[[i,edited_text,confidence,change_type,[evidence_codes],delete_intentionally]],"q":[[i,short_reason]]}
confidence is exact_reference|high_consensus|medium_context|low_uncertain. change_type is single_char|kana|punctuation|missing_text|duplicate|cross_column|order|title_noise|other. evidence_codes use R,C2,C3,V,X,G. delete_intentionally is 0 unless the source row must intentionally become empty. Never return explanations, Markdown, unknown IDs, changed IDs, coordinates or structure.
INPUT:
{{INPUT}}"""


AUDITOR_PROMPT = r"""Independently audit proposed Japanese OCR corrections. You do not receive the first model's explanation. Check only: missing text, repetition, neighbour-column adhesion, order, reference mismatch, symbol/number/name/negation changes, unsupported rewriting, and accidental empty text. Reference R outranks all OCRs. Do not rewrite or propose prose.
Input: a=[{i:short integer alias,o:before,n:proposal,c:[[model,text]],v:[[model,[columns]]],r:[score,excerpt] or null,b:previous,a:next}]. Copy the exact i alias into failures; never invent an ID.
Return failures only as compact JSON: {"f":[[i,"high"|"medium",issue_code,short_reason]]}. Omit passes. issue_code: missing|duplicate|cross_column|order|reference_mismatch|symbol|name_number_negation|rewrite|empty|other.
INPUT:
{{INPUT}}"""


_RETRY_SUFFIX = "\nPrevious response was invalid. Return exactly one compact JSON object matching the stated schema; no Markdown or commentary."


def _compact_text(value: str) -> str:
    return _SPACE_RE.sub("", unicodedata.normalize("NFKC", str(value or "")).translate(_SYMBOL_FOLD))


def _strict_text_key(value: str) -> str:
    """Whitespace-insensitive but glyph-preserving identity key.

    Unlike NFKC matching, this deliberately keeps full/half-width forms, quote
    shapes, dashes, ellipses and compatibility glyphs distinct.
    """
    return _SPACE_RE.sub("", unicodedata.normalize("NFC", str(value or "")))


def _head_text(value: str, limit: int = 240) -> str:
    text = str(value or "")
    size = max(32, int(limit or 240))
    return text if len(text) <= size else text[:size] + "…"


def _tail_text(value: str, limit: int = 240) -> str:
    text = str(value or "")
    size = max(32, int(limit or 240))
    return text if len(text) <= size else "…" + text[-size:]


def _coerce_bool_flag(value) -> tuple[bool, bool]:
    """Return ``(parsed_value, valid)`` for model-produced boolean flags.

    ``bool("false")`` is true in Python, which is unsafe for
    ``delete_intentionally``.  Accept only explicit JSON-like spellings and
    report everything else as an invalid protocol value.
    """
    if isinstance(value, bool):
        return value, True
    if value is None:
        return False, True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if float(value) == 0.0:
            return False, True
        if float(value) == 1.0:
            return True, True
        return False, False
    raw = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    if raw in {"", "0", "false", "no", "n", "off", "否", "いいえ"}:
        return False, True
    if raw in {"1", "true", "yes", "y", "on", "是", "はい"}:
        return True, True
    return False, False


def _normalise_text_field(value) -> tuple[str, bool]:
    if isinstance(value, str):
        return value, True
    if value is None:
        return "", True
    # Numbers are occasionally returned for a numeric-only source line.  They
    # are safe to stringify; arrays/objects are almost always schema leakage.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value), True
    return "", False


def _strip_ruby_markup(value: str) -> str:
    text = str(value or "")
    previous = None
    while previous != text:
        previous = text
        text = _RUBY_PIPE_RE.sub(r"\1", text)
    return text


def _looks_japanese(value: str, *, is_title: bool = False) -> bool:
    text = _strip_ruby_markup(str(value or "")).strip()
    if not text:
        return False
    kana = len(_JAPANESE_RE.findall(text))
    cjk = len(_CJK_RE.findall(text))
    if kana:
        return True
    # Kanji-only chapter headings are common; long kanji-only paragraphs in a
    # bilingual EPUB are much more likely to be the Chinese translation layer.
    return bool(is_title and 1 <= cjk <= 30 and len(text) <= 40)


def load_reference_source(path: str | Path | None) -> ReferenceSource:
    raw_path = str(path or "").strip()
    if not raw_path:
        return ReferenceSource()
    source = Path(raw_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"参考文件不存在：{source}")
    from adapters.text_extractors import extract_paragraphs

    lines: list[str] = []
    if source.suffix.lower() in {".txt", ".md", ".markdown"}:
        raw_text = source.read_text(encoding="utf-8-sig")
        for raw in raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            text = re.sub(r"^#{1,6}\s*", "", raw.strip()).strip()
            text = _strip_ruby_markup(text)
            if _looks_japanese(text, is_title=bool(re.match(r"^(?:第.+章|序章|終章|プロローグ|エピローグ)", text))):
                lines.append(text)
    else:
        paragraphs = extract_paragraphs(str(source))
        for paragraph in paragraphs:
            text = _strip_ruby_markup(str(getattr(paragraph, "text", "") or "")).strip()
            if not _looks_japanese(text, is_title=bool(getattr(paragraph, "is_title", False))):
                continue
            lines.append(text)
    if len(lines) < 10:
        raise ValueError("参考文件中可用于日文对齐的文本过少。双语 EPUB 会自动过滤无假名的中文正文层。")
    joined = "\n".join(lines)
    return ReferenceSource(
        path=str(source),
        label=source.name,
        fingerprint=hashlib.sha256(source.read_bytes()).hexdigest(),
        line_count=len(lines),
        corpus=ReferenceCorpus.from_lines(lines, max_window=3),
    )


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
            values = data.items()
        elif isinstance(data, list):
            values = []
            for item in data:
                if isinstance(item, dict):
                    values.append((item.get("term") or item.get("source"), item.get("preferred") or item.get("target")))
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
                pairs.append((line, line))
    dedup: dict[str, str] = {}
    for left, right in pairs:
        dedup.setdefault(left, right)
    return list(dedup.items())


def _candidate_texts(item: dict) -> list[str]:
    values = []
    for candidate in item.get("candidates") or []:
        if isinstance(candidate, dict):
            values.append(str(candidate.get("text", "") or ""))
    return values


def _unique_nonempty(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    keys: set[str] = set()
    for value in values:
        text = str(value or "")
        key = _strict_text_key(text)
        if not key or key in keys:
            continue
        keys.add(key)
        result.append(text)
    return result


def _risk_codes(item: dict) -> list[str]:
    texts = _candidate_texts(item)
    compact = [_strict_text_key(value) for value in texts]
    nonempty = [value for value in compact if value]
    unique = set(nonempty)
    codes: list[str] = []
    if len(unique) >= 3:
        codes.append("3way")
    elif len(unique) == 2:
        codes.append("diff")
    if any(not value for value in compact) and nonempty:
        codes.append("empty")
    if nonempty and max(map(len, nonempty)) - min(map(len, nonempty)) >= 15:
        codes.append("len15")
    if bool(item.get("local_reocr_recommended")):
        codes.append("reocr")
    if float(item.get("confidence", 0.0) or 0.0) < 0.90:
        codes.append("low")
    if item.get("warnings") or item.get("character_fusion_warnings"):
        codes.append("warn")
    all_text = "".join(texts)
    if _NOISE_RE.search(all_text):
        codes.append("noise")
    if any(text.count("「") != text.count("」") or text.count("『") != text.count("』") for text in texts):
        codes.append("quote")
    column_variants = {
        tuple(_compact_text(str(fragment or "")) for fragment in (candidate.get("column_texts") or []))
        for candidate in (item.get("physical_column_candidates") or [])
        if isinstance(candidate, dict)
    }
    column_variants.discard(tuple())
    if len(column_variants) > 1:
        codes.append("cols")
    if str(item.get("block_type", "")) in {"chapter", "section", "toc_entry"}:
        codes.append("title")
    return list(dict.fromkeys(codes))


def _is_risky(item: dict) -> bool:
    return bool(_risk_codes(item))


def _best_query(item: dict) -> str:
    candidates = _unique_nonempty([
        str(item.get("edited_text", "") or ""),
        str(item.get("character_fused_text", "") or ""),
        *_candidate_texts(item),
    ])
    if not candidates:
        return ""
    # Prefer the median-length non-noisy candidate: longest can be an Apple
    # duplicate/cross-column paste, shortest can be a 48px omission.
    clean = [x for x in candidates if not _NOISE_RE.search(x)] or candidates
    ordered = sorted(clean, key=lambda x: len(_compact_text(x)))
    return ordered[len(ordered) // 2]


def _reference_evidence(item: dict, reference: ReferenceSource, minimum: float) -> dict | None:
    if not reference.available or reference.corpus is None:
        return None
    query = _best_query(item)
    if len(_compact_text(query)) < 4:
        return None
    match = reference.corpus.search(query)
    if match.score < minimum or not match.text:
        return None
    excerpt = str(match.text or "")
    # Bound pathological multi-line windows while retaining enough text to
    # recover one missing sentence. Japanese is close to one token/character.
    max_chars = max(240, min(900, len(query) * 3 + 120))
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars]
    return {
        "score": round(float(match.score), 4),
        "structure_score": round(float(match.structure_score), 4),
        "line_start": int(match.line_start),
        "line_end": int(match.line_end),
        "text": excerpt,
        "exact": bool(match.exact),
        "containment": bool(match.containment),
    }


def _compact_columns(item: dict) -> list[list]:
    grouped: dict[tuple[str, ...], dict] = {}
    order: list[tuple[str, ...]] = []
    for candidate in item.get("physical_column_candidates") or []:
        if not isinstance(candidate, dict):
            continue
        fragments = tuple(str(value or "") for value in (candidate.get("column_texts") or []))
        key = tuple(_strict_text_key(value) for value in fragments)
        if not any(key):
            continue
        if key not in grouped:
            grouped[key] = {"labels": [], "fragments": list(fragments)}
            order.append(key)
        label = str(candidate.get("model_label", "") or "").strip()
        if label and label not in grouped[key]["labels"]:
            grouped[key]["labels"].append(label)
    return [
        ["+".join(grouped[key]["labels"]), grouped[key]["fragments"]]
        for key in order
    ]


def _compact_candidates(item: dict) -> list[list]:
    # Identical model texts are grouped rather than discarded.  This preserves
    # the strongest low-token signal: whether two or three independent OCRs
    # agree, without repeating the same Japanese string in the prompt.
    grouped: dict[str, dict] = {}
    order: list[str] = []
    for candidate in item.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        text = str(candidate.get("text", "") or "")
        key = _strict_text_key(text)
        if key not in grouped:
            grouped[key] = {"labels": [], "text": text, "confidence": 0.0, "count": 0}
            order.append(key)
        entry = grouped[key]
        label = str(candidate.get("model_label", "") or "").strip()
        if label and label not in entry["labels"]:
            entry["labels"].append(label)
        entry["confidence"] = max(entry["confidence"], float(candidate.get("confidence", 0.0) or 0.0))
        entry["count"] += 1
    return [
        [
            "+".join(grouped[key]["labels"]),
            grouped[key]["text"],
            round(float(grouped[key]["confidence"]), 3),
            int(grouped[key]["count"]),
        ]
        for key in order
    ]


def _relevant_glossary(
    glossary: Sequence[tuple[str, str]],
    wires: Sequence[dict],
    anchors: dict | None = None,
    *,
    max_entries: int = 64,
) -> list[list[str]]:
    """Select only glossary entries plausibly related to one AI chunk.

    Sending a full book glossary with every 10–30-page batch can cost more
    tokens than the OCR evidence itself.  Exact occurrences rank first; a
    conservative two-character overlap keeps likely OCR variants without
    flooding the prompt with unrelated names.
    """
    if not glossary:
        return []
    text_parts: list[str] = []
    for wire in wires:
        text_parts.extend([str(wire.get("o", "") or ""), str(wire.get("b", "") or ""), str(wire.get("a", "") or "")])
        for candidate in wire.get("c") or []:
            if isinstance(candidate, (list, tuple)) and len(candidate) > 1:
                text_parts.append(str(candidate[1] or ""))
        for variant in wire.get("v") or []:
            if isinstance(variant, (list, tuple)) and len(variant) > 1:
                text_parts.extend(str(value or "") for value in (variant[1] or []))
    if anchors:
        text_parts.extend(str(value or "") for value in (anchors.get("b") or []))
        text_parts.extend(str(value or "") for value in (anchors.get("a") or []))
    haystack = _compact_text("".join(text_parts))
    if not haystack:
        return []
    exact: list[list[str]] = []
    fuzzy: list[tuple[int, list[str]]] = []
    for left, right in glossary:
        left_text, right_text = str(left or "").strip(), str(right or "").strip()
        if not left_text or not right_text:
            continue
        left_key, right_key = _compact_text(left_text), _compact_text(right_text)
        if (left_key and left_key in haystack) or (right_key and right_key in haystack):
            exact.append([left_text, right_text])
            continue
        key = left_key or right_key
        if len(key) < 3:
            continue
        grams = {key[i:i + 2] for i in range(len(key) - 1)}
        overlap = sum(1 for gram in grams if gram in haystack)
        if overlap >= max(1, min(2, len(grams))):
            fuzzy.append((overlap, [left_text, right_text]))
    fuzzy.sort(key=lambda item: (-item[0], len(_compact_text(item[1][0]))))
    combined = exact + [entry for _score, entry in fuzzy]
    dedup: list[list[str]] = []
    seen: set[tuple[str, str]] = set()
    for entry in combined:
        key = (_compact_text(entry[0]), _compact_text(entry[1]))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(entry)
        if len(dedup) >= max(1, int(max_entries or 64)):
            break
    return dedup


def _page_number(item: dict, fallback: int) -> int:
    try:
        value = int(item.get("page", 0) or 0)
    except Exception:
        value = 0
    return value if value > 0 else max(1, fallback)


def _annotate_pages(items: Sequence[dict]) -> list[int]:
    pages: list[int] = []
    current = 1
    for item in items:
        value = _page_number(item, current)
        current = max(current, value)
        pages.append(value)
    return pages


def _make_page_windows(items: Sequence[dict], target_indices: Sequence[int], options: AdjudicationOptions) -> list[tuple[int, int, list[int]]]:
    if not target_indices:
        return []
    pages = _annotate_pages(items)
    by_page: dict[int, list[int]] = {}
    for index in target_indices:
        by_page.setdefault(pages[index], []).append(index)
    sorted_pages = sorted(by_page)
    windows: list[tuple[int, int, list[int]]] = []
    cursor = 0
    while cursor < len(sorted_pages):
        start = sorted_pages[cursor]
        end = start + options.batch_pages - 1
        selected_pages = []
        while cursor < len(sorted_pages) and sorted_pages[cursor] <= end:
            selected_pages.append(sorted_pages[cursor])
            cursor += 1
        indices = [index for page in selected_pages for index in by_page[page]]
        windows.append((start, end, sorted(indices)))
    return windows


def _overlap_anchors(items: Sequence[dict], pages: Sequence[int], start: int, end: int, overlap: int) -> dict:
    before = [
        _tail_text(str(items[index].get("edited_text", "") or ""), 180)
        for index, page in enumerate(pages)
        if start - overlap <= page < start
    ][-6:]
    after = [
        _head_text(str(items[index].get("edited_text", "") or ""), 180)
        for index, page in enumerate(pages)
        if end < page <= end + overlap
    ][:6]
    return {"b": before, "a": after}


def _wire_item(
    items: Sequence[dict],
    index: int,
    reference: ReferenceSource,
    options: AdjudicationOptions,
) -> tuple[dict, dict | None, str]:
    item = items[index]
    actual_id = str(item.get("row_id", "") or "")
    evidence = _reference_evidence(item, reference, options.reference_min_score)
    previous = _tail_text(str(items[index - 1].get("edited_text", "") or ""), 240) if index > 0 else ""
    following = _head_text(str(items[index + 1].get("edited_text", "") or ""), 240) if index + 1 < len(items) else ""
    # The immutable row_id is deliberately kept local.  Sending a short integer
    # alias saves tokens and prevents models from truncating or fabricating the
    # long hash-bearing ID.
    wire = {
        "p": int(item.get("page", 0) or 0),
        "y": str(item.get("block_type", "") or ""),
        "o": str(item.get("edited_text", item.get("original_fused_text", "")) or ""),
        "c": _compact_candidates(item),
        "v": _compact_columns(item),
        "r": (
            [evidence["score"], evidence["line_start"], evidence["line_end"], evidence["text"]]
            if evidence else None
        ),
        "b": previous,
        "a": following,
        "k": _risk_codes(item),
    }
    return wire, evidence, actual_id


def _alias_batch(entries: Sequence[tuple[str, dict, dict | None]]) -> tuple[list[dict], dict[str, str], dict[str, dict | None]]:
    wires: list[dict] = []
    actual_ids: list[str] = []
    evidence_map: dict[str, dict | None] = {}
    for position, (actual_id, base_wire, evidence) in enumerate(entries, 1):
        wires.append({"i": position, **base_wire})
        actual_ids.append(actual_id)
        evidence_map[actual_id] = evidence
    return wires, _id_alias_map(actual_ids), evidence_map


def _split_for_token_budget(
    items: Sequence[dict],
    indices: Sequence[int],
    reference: ReferenceSource,
    options: AdjudicationOptions,
    anchors: dict,
    glossary: Sequence[tuple[str, str]],
    model: str,
) -> list[tuple[list[dict], dict[str, dict | None], dict[str, str], list[list[str]]]]:
    chunks: list[tuple[list[dict], dict[str, dict | None], dict[str, str], list[list[str]]]] = []
    current: list[tuple[str, dict, dict | None]] = []

    def pack(entries: Sequence[tuple[str, dict, dict | None]]):
        wires, aliases, evidence_map = _alias_batch(entries)
        glossary_wire = _relevant_glossary(glossary, wires, anchors)
        return wires, evidence_map, aliases, glossary_wire

    def token_count(entries: Sequence[tuple[str, dict, dict | None]]) -> int:
        wires, _evidence_map, _aliases, glossary_wire = pack(entries)
        payload = {
            "t": wires,
            "x": anchors,
            "g": glossary_wire,
            "m": "reference" if reference.available else "ocr_only",
        }
        prompt = ADJUDICATOR_PROMPT.replace(
            "{{INPUT}}",
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        )
        return estimate_tokens(prompt, model)

    for index in indices:
        base_wire, evidence, actual_id = _wire_item(items, index, reference, options)
        trial = current + [(actual_id, base_wire, evidence)]
        if current and token_count(trial) > options.max_prompt_tokens:
            chunks.append(pack(current))
            current = [(actual_id, base_wire, evidence)]
        else:
            current = trial
    if current:
        chunks.append(pack(current))
    return chunks


def _call_json(provider, prompt: str, *, retries: int, cancel_check: Callable[[], bool] | None = None) -> dict:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        if cancel_check and cancel_check():
            raise AdjudicationCancelled("AI 审定已取消。")
        current = prompt if attempt == 0 else prompt + _RETRY_SUFFIX
        try:
            reply = provider.call_json(current, 0.0)
            return parse_json_reply(reply)
        except Exception as exc:
            last_error = exc
    raise AdjudicationProtocolError(f"AI 返回无法解析，已重试 {retries} 次：{last_error}") from last_error


def _id_alias_map(actual_ids: Sequence[str]) -> dict[str, str]:
    """Build a forgiving alias map for one compact AI batch.

    The prompt sends short 1-based integer aliases to save tokens.  Providers
    sometimes return ``1``, ``"1"``, ``#1`` or ``row-1``; all are resolved
    locally to the immutable package row_id.  The actual row_id remains accepted
    for backward compatibility with older prompts and cached replies.
    """
    aliases: dict[str, str] = {}
    for position, actual in enumerate(actual_ids, 1):
        actual = str(actual or "").strip()
        if not actual:
            continue
        aliases[actual] = actual
        for alias in (
            str(position),
            f"#{position}",
            f"i{position}",
            f"item{position}",
            f"item-{position}",
            f"item_{position}",
            f"row{position}",
            f"row-{position}",
            f"row_{position}",
        ):
            aliases.setdefault(alias.casefold(), actual)
    return aliases


def _extract_numeric_alias(value) -> int | None:
    raw = unicodedata.normalize("NFKC", str(value if value is not None else "")).strip()
    if not raw:
        return None
    simplified = raw.casefold().strip("[](){}<> \t\r\n")
    match = re.fullmatch(r"(?:(?:id|i|item|row|t)\s*[:#_\-]?\s*)?(\d+)(?:\.0+)?", simplified)
    return int(match.group(1)) if match else None


def _uses_zero_based_aliases(values: Sequence[object], aliases: dict[str, str] | None) -> bool:
    numbers = [number for number in (_extract_numeric_alias(value) for value in values) if number is not None]
    if not numbers or 0 not in numbers:
        return False
    lookup = aliases or {}
    positive = [int(key) for key in lookup if str(key).isdigit() and int(key) > 0]
    count = max(positive, default=0)
    return bool(count and all(0 <= number < count for number in numbers))


def _resolve_ai_id(
    value,
    allowed_ids: set[str],
    aliases: dict[str, str] | None = None,
    *,
    zero_based: bool = False,
) -> str | None:
    raw = unicodedata.normalize("NFKC", str(value if value is not None else "")).strip()
    if not raw:
        return None
    if raw in allowed_ids:
        return raw
    lookup = aliases or {}
    if zero_based:
        numeric = _extract_numeric_alias(raw)
        if numeric is not None:
            direct = lookup.get(str(numeric + 1)) or lookup.get(f"#{numeric + 1}")
            if direct in allowed_ids:
                return direct
    direct = lookup.get(raw) or lookup.get(raw.casefold())
    if direct in allowed_ids:
        return direct
    # Accept harmless wrappers such as "ID: 1", "row 1" or a JSON number
    # rendered as 1.0.  Do not extract arbitrary digits from immutable row IDs.
    simplified = raw.casefold().strip("[](){}<> \t\r\n")
    match = re.fullmatch(r"(?:(?:id|i|item|row|t)\s*[:#_\-]?\s*)?(\d+)(?:\.0+)?", simplified)
    if match:
        number = int(match.group(1))
        if zero_based:
            number += 1
        direct = lookup.get(str(number)) or lookup.get(f"#{number}")
        if direct in allowed_ids:
            return direct
    return None


def _parse_updates(
    data,
    allowed_ids: set[str],
    aliases: dict[str, str] | None = None,
) -> tuple[dict[str, dict], dict[str, str], list[dict]]:
    # Common provider compatibility: some models return the update array
    # directly, or one update object, despite JSON-mode instructions.
    if isinstance(data, list):
        raw_updates, raw_uncertain = data, []
    elif isinstance(data, dict):
        nested = data.get("result") if isinstance(data.get("result"), dict) else data
        if any(key in nested for key in ("u", "updates", "q", "uncertain", "blocks")):
            raw_updates = nested.get("u", nested.get("updates", nested.get("blocks", [])))
            raw_uncertain = nested.get("q", nested.get("uncertain", []))
        elif any(key in nested for key in ("i", "id", "item_id", "row_id", "edited_text", "text")):
            raw_updates, raw_uncertain = [nested], []
        else:
            raw_updates, raw_uncertain = [], []
    else:
        raise AdjudicationProtocolError("AI JSON 必须是对象或更新数组。")
    if isinstance(raw_updates, dict):
        # Accept compact mappings such as {"1":"修正文"}.
        raw_updates = [[key, value] for key, value in raw_updates.items()]
    if isinstance(raw_uncertain, dict):
        raw_uncertain = [[key, value] for key, value in raw_uncertain.items()]
    if not isinstance(raw_updates, list) or not isinstance(raw_uncertain, list):
        raise AdjudicationProtocolError("AI JSON 缺少数组 u/q。")

    updates: dict[str, dict] = {}
    uncertain: dict[str, str] = {}
    warnings: list[dict] = []

    def reject(section: str, returned_id, reason: str, raw_entry=None) -> None:
        warnings.append({
            "section": section,
            "returned_id": str(returned_id if returned_id is not None else ""),
            "reason": reason,
            "entry_preview": str(raw_entry)[:240] if raw_entry is not None else "",
        })

    returned_ids: list[object] = []
    for raw in [*raw_updates, *raw_uncertain]:
        if isinstance(raw, dict):
            returned_ids.append(raw.get("id", raw.get("item_id", raw.get("row_id", raw.get("i")))))
        elif isinstance(raw, (list, tuple)) and raw:
            returned_ids.append(raw[0])
    zero_based = _uses_zero_based_aliases(returned_ids, aliases)

    for raw in raw_updates:
        invalid_text = False
        invalid_delete = False
        if isinstance(raw, dict):
            returned_id = raw.get("id", raw.get("item_id", raw.get("row_id", raw.get("i"))))
            text, text_valid = _normalise_text_field(raw.get("edited_text", raw.get("text", "")))
            confidence = str(raw.get("confidence", "medium_context") or "medium_context")
            change_type = str(raw.get("change_type", "other") or "other")
            evidence = raw.get("evidence", [])
            delete, delete_valid = _coerce_bool_flag(raw.get("delete_intentionally", False))
            invalid_text = not text_valid
            invalid_delete = not delete_valid
        elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
            returned_id = raw[0]
            text, text_valid = _normalise_text_field(raw[1])
            confidence = str(raw[2] if len(raw) > 2 else "medium_context")
            change_type = str(raw[3] if len(raw) > 3 else "other")
            evidence = raw[4] if len(raw) > 4 else []
            delete, delete_valid = _coerce_bool_flag(raw[5] if len(raw) > 5 else False)
            invalid_text = not text_valid
            invalid_delete = not delete_valid
        else:
            reject("u", "", "u 中存在格式错误的更新", raw)
            continue
        item_id = _resolve_ai_id(returned_id, allowed_ids, aliases, zero_based=zero_based)
        if item_id is None:
            reject("u", returned_id, "AI 返回未知或不可编辑 ID", raw)
            continue
        if invalid_text:
            uncertain[item_id] = "AI 返回的 edited_text 不是字符串"
            reject("u", returned_id, "edited_text 类型错误", raw)
            continue
        if invalid_delete:
            uncertain[item_id] = "AI 返回的 delete_intentionally 不是明确布尔值"
            reject("u", returned_id, "delete_intentionally 类型错误", raw)
            continue
        if confidence not in _ALLOWED_CONFIDENCE:
            confidence = "low_uncertain"
        if change_type not in _ALLOWED_CHANGE_TYPES:
            change_type = "other"
        if isinstance(evidence, str):
            evidence = [evidence]
        if not isinstance(evidence, list):
            evidence = []
        if not text.strip() and not delete:
            uncertain[item_id] = "AI 将文字置空但未声明 delete_intentionally"
            reject("u", returned_id, "空文本缺少 delete_intentionally", raw)
            continue
        proposal = {
            "text": text,
            "confidence": confidence,
            "change_type": change_type,
            "evidence": [str(value) for value in evidence if str(value)],
            "delete_intentionally": delete,
        }
        if item_id in uncertain:
            reject("u", returned_id, "同一条目同时出现在不确定列表", raw)
            continue
        if item_id in updates:
            if updates[item_id] == proposal:
                # Harmless duplicate caused by provider streaming/JSON repair.
                continue
            updates.pop(item_id, None)
            uncertain[item_id] = "AI 对同一条目重复或冲突返回，已转人工复核"
            reject("u", returned_id, "同一条目重复或冲突返回", raw)
            continue
        updates[item_id] = proposal

    for raw in raw_uncertain:
        if isinstance(raw, dict):
            returned_id = raw.get("id", raw.get("item_id", raw.get("row_id", raw.get("i"))))
            reason = str(raw.get("reason", "证据不足") or "证据不足")
        elif isinstance(raw, (list, tuple)) and raw:
            returned_id = raw[0]
            reason = str(raw[1] if len(raw) > 1 else "证据不足")
        else:
            reject("q", "", "q 中存在格式错误的条目", raw)
            continue
        item_id = _resolve_ai_id(returned_id, allowed_ids, aliases, zero_based=zero_based)
        if item_id is None:
            reject("q", returned_id, "AI 返回未知不确定 ID", raw)
            continue
        updates.pop(item_id, None)
        uncertain[item_id] = reason
    return updates, uncertain, warnings


def _source_similarity(proposed: str, sources: Sequence[str]) -> tuple[float, float]:
    p = _compact_text(proposed)
    if not p:
        return 0.0, 0.0
    best_ratio = 0.0
    best_coverage = 0.0
    for source in sources:
        s = _compact_text(source)
        if not s:
            continue
        matcher = SequenceMatcher(None, s, p, autojunk=False)
        common = sum(block.size for block in matcher.get_matching_blocks() if block.size)
        best_ratio = max(best_ratio, matcher.ratio())
        best_coverage = max(best_coverage, common / max(1, len(p)))
    return best_ratio, best_coverage


def _proposal_support_count(item: dict, proposed: str) -> int:
    target = _strict_text_key(proposed)
    if not target:
        return 0
    count = 0
    for source in _candidate_texts(item):
        key = _strict_text_key(source)
        if key and key == target:
            count += 1
    return count


def _calibrate_proposal_confidence(item: dict, proposal: dict, reference: dict | None, *, reference_strong_score: float) -> dict:
    calibrated = dict(proposal)
    confidence = str(calibrated.get("confidence", "medium_context") or "medium_context")
    text = str(calibrated.get("text", "") or "")
    reference_supported = False
    if reference:
        ref_text = str(reference.get("text", "") or "")
        ref_key = _strict_text_key(ref_text)
        proposal_key = _strict_text_key(text)
        score = float(reference.get("score", 0.0) or 0.0)
        reference_supported = bool(
            score >= reference_strong_score
            and proposal_key
            and (proposal_key in ref_key or ref_key in proposal_key)
        )
    if confidence == "exact_reference" and not reference_supported:
        calibrated["confidence"] = "low_uncertain"
        calibrated.setdefault("calibration_reason", "exact_reference 无强参考对齐支持")
    elif confidence == "high_consensus" and _proposal_support_count(item, text) < 2 and not reference_supported:
        calibrated["confidence"] = "medium_context"
        calibrated.setdefault("calibration_reason", "未检测到两个 OCR 对该完整结果的一致支持")
    return calibrated


def _proposal_safe(item: dict, proposal: dict, reference: dict | None, *, reference_strong_score: float = 0.96) -> tuple[bool, str]:
    text = str(proposal.get("text", "") or "")
    if not text.strip():
        return (bool(proposal.get("delete_intentionally")), "intentional_delete" if proposal.get("delete_intentionally") else "empty")
    before = str(item.get("edited_text", item.get("original_fused_text", "")) or "")
    column_sources: list[str] = []
    for candidate in item.get("physical_column_candidates") or []:
        if isinstance(candidate, dict):
            column_sources.extend(str(value or "") for value in (candidate.get("column_texts") or []))
    sources = [before, str(item.get("character_fused_text", "") or ""), *_candidate_texts(item), *column_sources]
    reference_supported = False
    if reference:
        ref_text = str(reference.get("text", "") or "")
        ref_key = _strict_text_key(ref_text)
        proposal_key = _strict_text_key(text)
        score = float(reference.get("score", 0.0) or 0.0)
        reference_supported = bool(
            score >= reference_strong_score
            and proposal_key
            and (proposal_key in ref_key or ref_key in proposal_key)
        )
    if proposal.get("confidence") == "exact_reference" and not reference_supported:
        return False, "false_exact_reference"
    p = _compact_text(text)
    b = _compact_text(before)
    similarity, coverage = _source_similarity(text, sources)
    if not b:
        # Empty fused rows are the easiest place for an LLM to hallucinate.
        # Require either strong publication evidence or meaningful overlap with
        # an OCR/physical-column source before filling them.
        if reference_supported or similarity >= 0.50 or coverage >= 0.62:
            return True, "fills_empty_supported"
        return False, "unsupported_fill_empty"
    ratio = len(p) / max(1, len(b))
    if reference_supported:
        # A strong local reference alignment may restore a missing sentence,
        # but it must not let the model paste an entire neighbouring paragraph.
        if len(b) >= 80 and not (0.50 <= ratio <= 1.85):
            return False, "reference_length"
        if 20 <= len(b) < 80 and not (0.35 <= ratio <= 2.20):
            return False, "reference_length"
        if 5 <= len(b) < 20 and not (0.25 <= ratio <= 3.00):
            return False, "reference_length"
    else:
        if len(b) >= 80 and not (0.62 <= ratio <= 1.45):
            return False, "length"
        if 20 <= len(b) < 80 and not (0.48 <= ratio <= 1.75):
            return False, "length"
        if 5 <= len(b) < 20 and not (0.30 <= ratio <= 2.40):
            return False, "length"
    threshold = 0.46 if len(b) >= 40 else 0.30
    if not reference_supported and similarity < threshold and coverage < threshold:
        return False, "unsupported_rewrite"
    # Prevent a model from translating Japanese into Chinese or English.
    if _JAPANESE_RE.search(before) and not _JAPANESE_RE.search(text) and len(p) >= 8:
        return False, "language_changed"
    return True, "source_supported"


def _audit_wire(item: dict, proposal: dict, reference: dict | None, previous: str, following: str) -> dict:
    return {
        "o": str(item.get("edited_text", item.get("original_fused_text", "")) or ""),
        "n": str(proposal.get("text", "") or ""),
        "c": [[entry[0], entry[1]] for entry in _compact_candidates(item)],
        "v": _compact_columns(item),
        "r": [reference.get("score"), reference.get("text")] if reference else None,
        "b": previous,
        "a": following,
    }


def _parse_audit_failures(
    data,
    allowed_ids: set[str],
    aliases: dict[str, str] | None = None,
) -> tuple[dict[str, dict], list[dict]]:
    if isinstance(data, list):
        raw = data
    elif isinstance(data, dict):
        nested = data.get("result") if isinstance(data.get("result"), dict) else data
        if any(key in nested for key in ("f", "failures")):
            raw = nested.get("f", nested.get("failures", []))
        elif any(key in nested for key in ("i", "id", "item_id", "row_id", "issue")):
            raw = [nested]
        else:
            raw = []
    else:
        raise AdjudicationProtocolError("审计 JSON 必须是对象或失败数组。")
    if isinstance(raw, dict):
        raw = [[key, "high", "other", value] for key, value in raw.items()]
    if not isinstance(raw, list):
        raise AdjudicationProtocolError("审计 JSON 缺少数组 f。")
    failures: dict[str, dict] = {}
    warnings: list[dict] = []
    returned_ids: list[object] = []
    for entry in raw:
        if isinstance(entry, dict):
            returned_ids.append(entry.get("id", entry.get("item_id", entry.get("row_id", entry.get("i")))))
        elif isinstance(entry, (list, tuple)) and entry:
            returned_ids.append(entry[0])
    zero_based = _uses_zero_based_aliases(returned_ids, aliases)
    for entry in raw:
        if isinstance(entry, dict):
            returned_id = entry.get("id", entry.get("item_id", entry.get("row_id", entry.get("i"))))
            severity = str(entry.get("severity", "high") or "high").casefold()
            issue = str(entry.get("issue", "other") or "other")
            reason = str(entry.get("reason", "") or "")
        elif isinstance(entry, (list, tuple)) and entry:
            returned_id = entry[0]
            severity = str(entry[1] if len(entry) > 1 else "high").casefold()
            issue = str(entry[2] if len(entry) > 2 else "other")
            reason = str(entry[3] if len(entry) > 3 else "")
        else:
            warnings.append({"section": "f", "returned_id": "", "reason": "审计 f 中存在格式错误的条目", "entry_preview": str(entry)[:240]})
            continue
        item_id = _resolve_ai_id(returned_id, allowed_ids, aliases, zero_based=zero_based)
        if item_id is None:
            warnings.append({
                "section": "f",
                "returned_id": str(returned_id if returned_id is not None else ""),
                "reason": "审计返回未知 ID",
                "entry_preview": str(entry)[:240],
            })
            continue
        failure = {
            "severity": severity if severity in {"high", "medium"} else "high",
            "issue": issue,
            "reason": reason,
        }
        if item_id in failures and failures[item_id] != failure:
            failures[item_id] = {
                "severity": "high",
                "issue": "conflict",
                "reason": "独立审计对同一条目返回相互冲突的问题",
            }
            warnings.append({
                "section": "f",
                "returned_id": str(returned_id),
                "reason": "审计对同一条目重复或冲突返回",
                "entry_preview": str(entry)[:240],
            })
        else:
            failures[item_id] = failure
    return failures, warnings


def _report_event(callback: Callable[[dict], None] | None, **event) -> None:
    if callback:
        callback(event)


def adjudicate_package(
    provider,
    package: dict,
    *,
    options: AdjudicationOptions | None = None,
    reference_path: str | Path | None = None,
    glossary_path: str | Path | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[dict, dict]:
    """Adjudicate a multi-model OCR package and return (package, audit report)."""
    from engine.ocr_roundtrip_package import MODE_MULTI, _seal_package

    if str(package.get("mode", "") or "") != MODE_MULTI:
        raise ValueError("AI 审定只支持多模型 OCR 融合包。")
    opts = (options or AdjudicationOptions()).normalised()
    reference = load_reference_source(reference_path)
    glossary = load_glossary(glossary_path)
    result = copy.deepcopy(package)
    items = list(result.get("editable_items") or [])
    if not items:
        raise ValueError("OCR 对比包没有 editable_items。")
    id_to_index = {str(item.get("row_id", "") or ""): index for index, item in enumerate(items)}
    if not all(id_to_index) or len(id_to_index) != len(items):
        raise ValueError("OCR 对比包 row_id 缺失或重复。")

    target_indices = [
        index for index, item in enumerate(items)
        if (not opts.risk_only) or _is_risky(item)
    ]
    pages = _annotate_pages(items)
    windows = _make_page_windows(items, target_indices, opts)
    model = str(getattr(provider, "model", "") or "")
    # Ask compatible providers for deterministic nucleus sampling when supported.
    try:
        provider.kwargs["top_p"] = 0.1
    except Exception:
        pass

    report_items: list[dict] = []
    pending_audit: dict[str, tuple[dict, dict, dict | None]] = {}
    uncertain_ids: set[str] = set()
    protocol_warnings: list[dict] = []
    failed_request_batches = 0
    failed_audit_batches = 0
    total_chunks = 0
    for start, end, indices in windows:
        anchors = _overlap_anchors(items, pages, start, end, opts.overlap_pages)
        total_chunks += len(_split_for_token_budget(items, indices, reference, opts, anchors, glossary, model))
    completed_chunks = 0
    _report_event(
        progress_callback,
        stage="准备证据",
        current=0,
        total=max(1, total_chunks),
        target_rows=len(target_indices),
        reference_rows=reference.line_count,
    )

    for start, end, indices in windows:
        if cancel_check and cancel_check():
            raise AdjudicationCancelled("AI 审定已取消。")
        anchors = _overlap_anchors(items, pages, start, end, opts.overlap_pages)
        chunks = _split_for_token_budget(items, indices, reference, opts, anchors, glossary, model)
        for wires, evidence_map, aliases, glossary_wire in chunks:
            allowed = set(aliases.values())
            payload = {
                "t": wires,
                "x": anchors,
                "g": glossary_wire,
                "m": "reference" if reference.available else "ocr_only",
            }
            prompt = ADJUDICATOR_PROMPT.replace(
                "{{INPUT}}",
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            )
            estimated_prompt_tokens = estimate_tokens(prompt, model)
            if len(wires) == 1 and estimated_prompt_tokens > opts.max_prompt_tokens:
                failed_request_batches += 1
                item_id = next(iter(allowed))
                item = items[id_to_index[item_id]]
                reason = (
                    f"单条 OCR 证据约 {estimated_prompt_tokens} tokens，超过本次上限 "
                    f"{opts.max_prompt_tokens}；为避免截断或高额消耗，已保留原文并转人工复核"
                )
                uncertain_ids.add(item_id)
                protocol_warnings.append({
                    "section": "budget",
                    "returned_id": "",
                    "reason": reason,
                    "entry_preview": "",
                    "stage": "adjudication",
                    "page_start": start,
                    "page_end": end,
                    "valid_aliases": [wire["i"] for wire in wires],
                })
                report_items.append({
                    "item_id": item_id,
                    "page": int(item.get("page", 0) or 0),
                    "before": str(item.get("edited_text", "") or ""),
                    "after": str(item.get("edited_text", "") or ""),
                    "evidence_source": [],
                    "change_type": "unchanged",
                    "confidence": "low_uncertain",
                    "needs_human_review": True,
                    "reason": reason,
                    "audit_issues": [],
                })
                completed_chunks += 1
                _report_event(
                    progress_callback,
                    stage="证据超出 Token 上限·已转人工",
                    current=completed_chunks,
                    total=max(1, total_chunks),
                    page_start=start,
                    page_end=end,
                )
                continue
            _report_event(
                progress_callback,
                stage="AI 校订",
                current=completed_chunks,
                total=max(1, total_chunks),
                page_start=start,
                page_end=end,
                batch_rows=len(wires),
                estimated_tokens=estimated_prompt_tokens,
            )
            try:
                response = _call_json(
                    provider,
                    prompt,
                    retries=opts.request_retries,
                    cancel_check=cancel_check,
                )
            except AdjudicationCancelled:
                raise
            except Exception as exc:
                failed_request_batches += 1
                warning = {
                    "section": "request",
                    "returned_id": "",
                    "reason": f"AI 批次请求失败，已保留原文并转人工复核：{exc}",
                    "entry_preview": "",
                    "stage": "adjudication",
                    "page_start": start,
                    "page_end": end,
                    "valid_aliases": [wire["i"] for wire in wires],
                }
                protocol_warnings.append(warning)
                for item_id in allowed:
                    uncertain_ids.add(item_id)
                    item = items[id_to_index[item_id]]
                    report_items.append({
                        "item_id": item_id,
                        "page": int(item.get("page", 0) or 0),
                        "before": str(item.get("edited_text", "") or ""),
                        "after": str(item.get("edited_text", "") or ""),
                        "evidence_source": [],
                        "change_type": "unchanged",
                        "confidence": "low_uncertain",
                        "needs_human_review": True,
                        "reason": warning["reason"],
                        "audit_issues": [],
                    })
                completed_chunks += 1
                _report_event(
                    progress_callback,
                    stage="AI 批次失败·已转人工",
                    current=completed_chunks,
                    total=max(1, total_chunks),
                    page_start=start,
                    page_end=end,
                )
                continue
            try:
                updates, uncertain, parse_warnings = _parse_updates(response, allowed, aliases)
            except AdjudicationProtocolError as exc:
                updates, uncertain = {}, {}
                parse_warnings = [{
                    "section": "batch",
                    "returned_id": "",
                    "reason": str(exc),
                    "entry_preview": "",
                }]

            # If every returned patch used an unusable ID, spend one small retry
            # to repair the protocol rather than throwing away the whole batch.
            if parse_warnings and not updates and not uncertain and opts.request_retries > 0:
                valid_aliases = [wire["i"] for wire in wires]
                repair_prompt = (
                    prompt
                    + "\nPROTOCOL REPAIR: Your previous IDs could not be mapped. "
                    + f"Use only these exact integer i aliases: {valid_aliases}. "
                    + "Return the same compact u/q JSON schema; no prose."
                )
                try:
                    repaired_response = _call_json(
                        provider,
                        repair_prompt,
                        retries=0,
                        cancel_check=cancel_check,
                    )
                    repaired_updates, repaired_uncertain, repaired_warnings = _parse_updates(
                        repaired_response, allowed, aliases
                    )
                    if repaired_updates or repaired_uncertain or not repaired_warnings:
                        updates, uncertain = repaired_updates, repaired_uncertain
                    parse_warnings.extend(repaired_warnings)
                except AdjudicationCancelled:
                    raise
                except Exception as exc:
                    parse_warnings.append({
                        "section": "repair",
                        "returned_id": "",
                        "reason": f"ID 修复重试失败：{exc}",
                        "entry_preview": "",
                    })

            for warning in parse_warnings:
                contextual = dict(warning)
                contextual.update({
                    "stage": "adjudication",
                    "page_start": start,
                    "page_end": end,
                    "valid_aliases": [wire["i"] for wire in wires],
                })
                protocol_warnings.append(contextual)
            if parse_warnings:
                _report_event(
                    progress_callback,
                    stage="AI 协议兼容",
                    current=completed_chunks,
                    total=max(1, total_chunks),
                    page_start=start,
                    page_end=end,
                    warnings=len(parse_warnings),
                )
            # An entirely unmappable response is never silently accepted.  The
            # rows stay unchanged and enter manual review, while later batches
            # continue normally instead of crashing the 400-page job.
            if parse_warnings and not updates and not uncertain:
                uncertain = {
                    item_id: "AI 返回的批次 ID 无法对应，已保留原文并转人工复核"
                    for item_id in allowed
                }

            uncertain_ids.update(uncertain)
            for item_id, reason in uncertain.items():
                item = items[id_to_index[item_id]]
                report_items.append({
                    "item_id": item_id,
                    "page": int(item.get("page", 0) or 0),
                    "before": str(item.get("edited_text", "") or ""),
                    "after": str(item.get("edited_text", "") or ""),
                    "evidence_source": [],
                    "change_type": "unchanged",
                    "confidence": "low_uncertain",
                    "needs_human_review": True,
                    "reason": reason,
                    "audit_issues": [],
                })
            for item_id, proposal in updates.items():
                item = items[id_to_index[item_id]]
                reference_evidence = evidence_map.get(item_id)
                proposal = _calibrate_proposal_confidence(
                    item, proposal, reference_evidence,
                    reference_strong_score=opts.reference_strong_score,
                )
                before = str(item.get("edited_text", item.get("original_fused_text", "")) or "")
                # The prompt asks the model to omit unchanged rows, but some
                # providers echo them anyway.  Do not spend audit tokens or add
                # a synthetic AI candidate for an actual no-op.
                if (
                    _strict_text_key(proposal.get("text", "")) == _strict_text_key(before)
                    and not proposal.get("delete_intentionally")
                ):
                    continue
                safe, safety_reason = _proposal_safe(
                    item, proposal, reference_evidence,
                    reference_strong_score=opts.reference_strong_score,
                )
                if not safe or proposal["confidence"] == "low_uncertain":
                    uncertain_ids.add(item_id)
                    report_items.append({
                        "item_id": item_id,
                        "page": int(item.get("page", 0) or 0),
                        "before": before,
                        "after": before,
                        "proposed_text": proposal["text"],
                        "delete_intentionally": bool(proposal.get("delete_intentionally", False)),
                        "evidence_source": proposal["evidence"],
                        "change_type": proposal["change_type"],
                        "confidence": "low_uncertain",
                        "needs_human_review": True,
                        "reason": (
                            f"本地安全拦截：{safety_reason}"
                            if not safe else str(proposal.get("calibration_reason", "AI 标记证据不足") or "AI 标记证据不足")
                        ),
                        "audit_issues": [],
                        "reference": copy.deepcopy(reference_evidence),
                    })
                    continue
                pending_audit[item_id] = (item, proposal, reference_evidence)
            completed_chunks += 1
            _report_event(
                progress_callback,
                stage="AI 校订",
                current=completed_chunks,
                total=max(1, total_chunks),
                page_start=start,
                page_end=end,
            )

    audit_failures: dict[str, dict] = {}
    if opts.independent_audit and pending_audit:
        audit_entries: list[tuple[str, dict]] = []
        for item_id, (item, proposal, reference_evidence) in pending_audit.items():
            index = id_to_index[item_id]
            previous = _tail_text(str(items[index - 1].get("edited_text", "") or ""), 240) if index > 0 else ""
            following = _head_text(str(items[index + 1].get("edited_text", "") or ""), 240) if index + 1 < len(items) else ""
            audit_entries.append((item_id, _audit_wire(item, proposal, reference_evidence, previous, following)))

        def audit_wire(entries: Sequence[tuple[str, dict]]) -> tuple[list[dict], dict[str, str]]:
            actual_ids = [item_id for item_id, _wire in entries]
            return [
                {"i": position, **wire}
                for position, (_item_id, wire) in enumerate(entries, 1)
            ], _id_alias_map(actual_ids)

        # Keep the audit independent and compact. Split on the same token ceiling.
        audit_chunks: list[list[tuple[str, dict]]] = []
        current: list[tuple[str, dict]] = []
        for entry in audit_entries:
            trial = current + [entry]
            trial_wires, _trial_aliases = audit_wire(trial)
            prompt = AUDITOR_PROMPT.replace(
                "{{INPUT}}",
                json.dumps({"a": trial_wires}, ensure_ascii=False, separators=(",", ":")),
            )
            if current and estimate_tokens(prompt, model) > opts.max_prompt_tokens:
                audit_chunks.append(current)
                current = [entry]
            else:
                current = trial
        if current:
            audit_chunks.append(current)
        for audit_index, chunk in enumerate(audit_chunks, 1):
            chunk_wires, aliases = audit_wire(chunk)
            allowed = set(aliases.values())
            prompt = AUDITOR_PROMPT.replace(
                "{{INPUT}}",
                json.dumps({"a": chunk_wires}, ensure_ascii=False, separators=(",", ":")),
            )
            _report_event(
                progress_callback,
                stage="独立审计",
                current=audit_index - 1,
                total=len(audit_chunks),
                batch_rows=len(chunk_wires),
                estimated_tokens=estimate_tokens(prompt, model),
            )
            try:
                response = _call_json(
                    provider,
                    prompt,
                    retries=opts.request_retries,
                    cancel_check=cancel_check,
                )
            except AdjudicationCancelled:
                raise
            except Exception as exc:
                failed_audit_batches += 1
                audit_warnings = [{
                    "section": "request",
                    "returned_id": "",
                    "reason": f"独立审计批次请求失败：{exc}",
                    "entry_preview": "",
                }]
                parsed_failures = {}
            else:
                try:
                    parsed_failures, audit_warnings = _parse_audit_failures(response, allowed, aliases)
                except AdjudicationProtocolError as exc:
                    parsed_failures = {}
                    audit_warnings = [{
                        "section": "f",
                        "returned_id": "",
                        "reason": str(exc),
                        "entry_preview": "",
                    }]
            audit_failures.update(parsed_failures)
            if audit_warnings:
                for warning in audit_warnings:
                    contextual = dict(warning)
                    contextual.update({
                        "stage": "audit",
                        "audit_batch": audit_index,
                        "valid_aliases": [wire["i"] for wire in chunk_wires],
                    })
                    protocol_warnings.append(contextual)
                # An unresolved auditor warning might refer to any proposal in
                # the chunk.  Conservatively block auto-application for all of
                # them rather than letting a potentially bad rewrite pass.
                for item_id in allowed:
                    audit_failures.setdefault(item_id, {
                        "severity": "high",
                        "issue": "protocol",
                        "reason": "独立审计返回无法对应的 ID，已转人工复核",
                    })
                _report_event(
                    progress_callback,
                    stage="审计协议兼容",
                    current=audit_index,
                    total=len(audit_chunks),
                    warnings=len(audit_warnings),
                )
            _report_event(progress_callback, stage="独立审计", current=audit_index, total=len(audit_chunks))

    applied = 0
    for item_id, (item, proposal, reference_evidence) in pending_audit.items():
        before = str(item.get("edited_text", item.get("original_fused_text", "")) or "")
        failure = audit_failures.get(item_id)
        confidence = proposal["confidence"]
        if failure or (confidence == "medium_context" and not opts.auto_apply_medium):
            uncertain_ids.add(item_id)
            report_items.append({
                "item_id": item_id,
                "page": int(item.get("page", 0) or 0),
                "before": before,
                "after": before,
                "proposed_text": proposal["text"],
                "delete_intentionally": bool(proposal.get("delete_intentionally", False)),
                "evidence_source": proposal["evidence"],
                "change_type": proposal["change_type"],
                "confidence": "low_uncertain",
                "needs_human_review": True,
                "reason": (
                    f"独立审计未通过：{failure.get('reason') or failure.get('issue')}"
                    if failure else "设置要求 medium_context 只进入人工复核"
                ),
                "audit_issues": [failure] if failure else [],
                "reference": copy.deepcopy(reference_evidence),
            })
            continue
        after = str(proposal["text"] or "")
        item["edited_text"] = after
        item["delete_intentionally"] = bool(proposal.get("delete_intentionally", False))
        if after != before or item["delete_intentionally"]:
            applied += 1
            report_items.append({
                "item_id": item_id,
                "page": int(item.get("page", 0) or 0),
                "before": before,
                "after": after,
                "delete_intentionally": bool(proposal.get("delete_intentionally", False)),
                "evidence_source": proposal["evidence"],
                "change_type": proposal["change_type"],
                "confidence": confidence,
                "needs_human_review": False,
                "reason": "独立审计通过" if opts.independent_audit else "本地安全校验通过",
                "audit_issues": [],
                "reference": copy.deepcopy(reference_evidence),
            })

    unchanged = max(0, len(items) - applied - len(uncertain_ids))
    usage_getter = getattr(provider, "usage_snapshot", None)
    try:
        usage = dict(usage_getter() or {}) if callable(usage_getter) else {}
    except Exception:
        usage = {}
    report = {
        "schema": AUDIT_SCHEMA,
        "created_at_unix": int(time.time()),
        "package_id": str(package.get("package_id", "") or ""),
        "provider": str(getattr(provider, "name", "") or ""),
        "model": model,
        "mode": "reference_truth" if reference.available else "high_precision_candidate",
        "reference": {
            "path": reference.path,
            "label": reference.label,
            "sha256": reference.fingerprint,
            "japanese_line_count": reference.line_count,
        },
        "glossary": {
            "path": str(glossary_path or ""),
            "entry_count": len(glossary),
        },
        "options": asdict(opts),
        "stats": {
            "total_items": len(items),
            "target_items": len(target_indices),
            "skipped_low_risk": len(items) - len(target_indices),
            "applied_changes": applied,
            "uncertain_items": len(uncertain_ids),
            "audit_failures": len(audit_failures),
            "unchanged_items": unchanged,
            "request_batches": total_chunks,
            "failed_request_batches": failed_request_batches,
            "failed_audit_batches": failed_audit_batches,
            "protocol_warnings": len(protocol_warnings),
        },
        "usage": usage,
        "items": report_items,
        "protocol_warnings": protocol_warnings,
        "low_uncertain": sorted(uncertain_ids),
        "claims": (
            "出版参考文本仅在可靠局部对齐时作为最高级证据；低置信条目未静默写入。"
            if reference.available else
            "未提供出版版真值；结果仅为高精度融合候选，不代表出版级或100%准确。"
        ),
    }
    result["ai_adjudication_summary"] = {
        "schema": SCHEMA,
        "created_at_unix": report["created_at_unix"],
        "provider": report["provider"],
        "model": model,
        "mode": report["mode"],
        "reference_sha256": reference.fingerprint,
        "applied_changes": applied,
        "uncertain_items": len(uncertain_ids),
        "audit_failures": len(audit_failures),
        "protocol_warnings": len(protocol_warnings),
        "failed_request_batches": failed_request_batches,
        "failed_audit_batches": failed_audit_batches,
        "usage": usage,
    }
    # Top-level summary is new immutable metadata; reseal after adding it.
    _seal_package(result)
    return result, report


def save_adjudication_outputs(
    package: dict,
    report: dict,
    output_dir: str | Path,
    *,
    base_name: str = "multi_ocr",
) -> tuple[Path, Path]:
    from engine.ocr_roundtrip_package import save_package

    folder = Path(output_dir).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w\-一-龯ぁ-ゖァ-ヺ]+", "_", str(base_name or "multi_ocr"), flags=re.UNICODE).strip("_") or "multi_ocr"
    package_path = save_package(package, folder / f"{safe}_AI审定融合包.json")
    report_path = folder / f"{safe}_audit_report.json"
    temp = report_path.with_name(f".{report_path.name}.tmp")
    try:
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(report_path)
    finally:
        temp.unlink(missing_ok=True)
    return package_path, report_path
