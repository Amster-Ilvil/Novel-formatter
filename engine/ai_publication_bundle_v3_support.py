#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""High-value evidence and audit helpers for AI repair package v3.

This module is intentionally independent from the package writer.  It only
produces deterministic JSON-compatible records and optional selected crops, so
legacy OCR, formatter and EPUB code paths remain untouched.
"""
from __future__ import annotations

import copy
import io
import hashlib
import json
import math
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence
import xml.etree.ElementTree as ET

from models.document import BlockType, UnifiedDocument


TERMINAL = tuple("。！？!?」』）)]】》〉〕〗〙〛…‥")
STATUS_LABEL_RE = re.compile(
    r"(?P<label>職業|詳細|技能|スキル|レベル|LEVEL|Lv\.?|HP|MP|称号|種族|名前|NAME|属性|装備|ランク|年齢|性別)"
    r"\s*(?P<separator>[:：・･·‥…]{1,3})\s*",
    re.I,
)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
LATIN_RUN_RE = re.compile(r"[A-Za-z]{4,}")
NUMBER_RE = re.compile(r"[0-9０-９]+(?:[.,．，～〜~\-－][0-9０-９]+)*")
KANA_RE = re.compile(r"[ぁ-ゖァ-ヺー]")
HAN_RE = re.compile(r"[一-龯々〆ヵヶ]")
SIMPLIFIED_HINTS = set("这们来对为发后里国过还个从时会学书见说现没让与并种应实关开长门问间气东车马鱼鸟龙体级术数万叶边处当无")
BLOCK_TAGS = {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "dt", "dd", "blockquote", "pre"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: Any) -> str:
    return sha256_bytes(str(text or "").encode("utf-8"))


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def normalise_text(value: Any) -> str:
    text = unicodedata.normalize("NFC", str(value or ""))
    return re.sub(r"[\s\u3000]+", "", text)


def candidate_texts(source_item: dict | None) -> list[str]:
    if not isinstance(source_item, dict):
        return []
    values: list[str] = []
    for candidate in source_item.get("candidates") or []:
        if isinstance(candidate, dict):
            text = str(candidate.get("text", "") or "")
        else:
            text = str(candidate or "")
        if text and text not in values:
            values.append(text)
    for key in ("edited_text", "original_fused_text", "character_fused_text", "ai_adjudicated_text"):
        text = str(source_item.get(key, "") or "")
        if text and text not in values:
            values.append(text)
    return values




def model_candidate_texts(source_item: dict | None) -> list[str]:
    """Return one text per OCR model; unlike candidate_texts, keep duplicates."""
    if not isinstance(source_item, dict):
        return []
    values: list[str] = []
    for candidate in source_item.get("candidates") or []:
        if isinstance(candidate, dict):
            text = str(candidate.get("text", "") or "")
        else:
            text = str(candidate or "")
        if text:
            values.append(text)
    return values or candidate_texts(source_item)


_RUBY_MARKER_RE = re.compile(r"([一-龯々〆ヵヶァ-ヶーA-Za-z0-9０-９]{1,24})\|([ぁ-ゖァ-ヺー]{1,32})")


def strip_ruby_readings(text: str) -> str:
    """Return plain OCR text, keeping only the printed base characters."""
    value = str(text or "")
    particles = set("をがにへとはもので")
    output: list[str] = []
    last = 0
    for match in _RUBY_MARKER_RE.finditer(value):
        reading = match.group(2)
        effective_end = match.end()
        following = value[effective_end:effective_end + 1]
        if (
            len(reading) >= 3
            and reading[-1] in particles
            and following
            and re.match(r"[一-龯々〆ヵヶァ-ヶーA-Za-z0-9０-９、。！？!?]", following)
        ):
            effective_end -= 1
        output.append(value[last:match.start()])
        output.append(match.group(1))
        last = effective_end
    output.append(value[last:])
    return "".join(output)


def parse_inline_tokens(text: str) -> list[dict]:
    value = strip_ruby_readings(text)
    return [{"type": "text", "value": value}]


def _source_line_records(item: dict) -> list[dict]:
    """Return OCR line records without assuming one producer-specific schema."""
    candidates: list[Any] = []
    for key in ("ocr_lines", "line_boxes", "source_lines", "recognized_lines"):
        value = item.get(key)
        if isinstance(value, list):
            candidates.extend(value)
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    for key in ("ocr_lines", "line_boxes", "source_lines", "recognized_lines"):
        value = metadata.get(key)
        if isinstance(value, list):
            candidates.extend(value)
    result: list[dict] = []
    for raw in candidates:
        if isinstance(raw, str):
            value = raw.strip()
            if value:
                result.append({"text": value, "bbox": None})
            continue
        if not isinstance(raw, dict):
            continue
        value = str(raw.get("text", raw.get("value", raw.get("recognized_text", ""))) or "").strip()
        if not value:
            continue
        bbox = raw.get("bbox", raw.get("box", raw.get("polygon")))
        result.append({
            "text": value,
            "bbox": copy.deepcopy(bbox),
            "font_size": raw.get("font_size", raw.get("estimated_font_size")),
            "line_order": raw.get("line_order", raw.get("order")),
            "column_id": raw.get("column_id"),
        })
    return result


def _split_status_value_and_narrative(value: str) -> tuple[str, str, int | None]:
    """Conservatively separate a flattened status value from resumed prose."""
    text = str(value or "").strip()
    if not text:
        return "", "", None
    # A new physical line is the strongest signal.  The first compact line is
    # retained as the value and the remaining sentence-like lines become prose.
    if "\n" in text:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) >= 2:
            first = lines[0]
            rest = "\n".join(lines[1:])
            if len(first) <= 48 and len(rest) >= 8:
                return first, rest, text.find(lines[1])
    # Flattened OCR often resumes prose with a conventional sentence opener.
    # Require a reasonably compact status prefix so ordinary long values are
    # not split merely because they contain one of these words.
    opener = re.search(
        r"(?P<prose>(?:今|現在|その時|そして|しかし|だが|けれど|一方|やがて|すると|そこへ|その頃|その後)[、，].{6,})$",
        text,
    )
    if opener and 1 <= opener.start() <= 64:
        return text[:opener.start()].rstrip(), opener.group("prose"), opener.start()
    # When a compact value is followed by a full sentence boundary, preserve
    # only the first sentence as value if the remainder is clearly narrative.
    terminal = re.search(r"[。！？!?]", text)
    if terminal and terminal.end() <= 64 and len(text) - terminal.end() >= 10:
        suffix = text[terminal.end():].lstrip()
        if suffix:
            return text[:terminal.end()].rstrip(), suffix, terminal.end()
    return text, "", None


def special_layout(item: dict, text: str) -> dict | None:
    block_type = str(item.get("block_type", "") or "")
    source_lines = _source_line_records(item)
    groups: list[dict] = []
    narrative_suffix = ""
    narrative_suffix_char_start: int | None = None

    # Prefer original OCR line geometry.  This prevents the final status value
    # from swallowing the first narrative line after flattening.
    if source_lines:
        pending_label: str | None = None
        pending_separator = ""
        for line in source_lines:
            line_text = str(line.get("text", "") or "")
            match = STATUS_LABEL_RE.match(line_text)
            if match:
                value = line_text[match.end():].strip()
                groups.append({
                    "label": match.group("label"),
                    "separator": match.group("separator"),
                    "value": value,
                    "source_bbox": copy.deepcopy(line.get("bbox")),
                    "source_line_text": line_text,
                    "source_column_id": line.get("column_id"),
                })
                pending_label = match.group("label")
                pending_separator = match.group("separator")
            elif groups and pending_label and len(line_text) <= 48 and not re.search(r"[。！？!?]", line_text):
                groups[-1]["value"] = (str(groups[-1].get("value", "")) + line_text).strip()
                groups[-1].setdefault("continuation_bboxes", []).append(copy.deepcopy(line.get("bbox")))
            elif len(groups) >= 2:
                narrative_suffix = line_text
                narrative_suffix_char_start = max(0, str(text or "").find(line_text))
                break
        if len(groups) >= 2:
            return {
                "layout_type": "status_table",
                "line_groups": groups,
                "narrative_suffix": narrative_suffix or None,
                "narrative_suffix_char_start": narrative_suffix_char_start,
                "narrative_resume_item_id": str(item.get("next_item_id", "") or "") or None,
                "must_split_suffix_to_next_paragraph": bool(narrative_suffix),
                "must_not_merge_with_narrative": True,
                "preserve_line_breaks": True,
                "requires_structure_review": False,
                "source_geometry_used": True,
            }

    matches = list(STATUS_LABEL_RE.finditer(text))
    if len(matches) >= 2:
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            value = text[match.end():end].strip(" \t\n　:：・･·‥…")
            suffix = ""
            suffix_start = None
            if index + 1 == len(matches):
                value, suffix, local_start = _split_status_value_and_narrative(value)
                if suffix and local_start is not None:
                    narrative_suffix = suffix
                    narrative_suffix_char_start = match.end() + local_start
                    suffix_start = narrative_suffix_char_start
            groups.append({
                "label": match.group("label"),
                "separator": match.group("separator"),
                "value": value,
                "source_char_start": match.start(),
                "source_char_end": suffix_start if suffix_start is not None else end,
                "source_bbox": None,
            })
        return {
            "layout_type": "status_table",
            "line_groups": groups,
            "narrative_suffix": narrative_suffix or None,
            "narrative_suffix_char_start": narrative_suffix_char_start,
            "narrative_resume_item_id": str(item.get("next_item_id", "") or "") or None,
            "must_split_suffix_to_next_paragraph": bool(narrative_suffix),
            "must_not_merge_with_narrative": True,
            "preserve_line_breaks": True,
            "requires_structure_review": "\n" not in text or bool(narrative_suffix),
            "source_geometry_used": False,
        }
    if block_type == BlockType.FOOTNOTE.value:
        return {
            "layout_type": "footnote",
            "line_groups": [],
            "must_not_merge_with_narrative": True,
            "preserve_line_breaks": True,
            "requires_structure_review": False,
        }
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    hinted = str(metadata.get("layout_type", "") or metadata.get("semantic_type", "") or "").lower()
    if hinted in {"poem", "chant", "letter", "profile", "system_message", "status_table"}:
        return {
            "layout_type": hinted,
            "line_groups": [{"line": line} for line in text.splitlines()],
            "must_not_merge_with_narrative": True,
            "preserve_line_breaks": True,
            "requires_structure_review": False,
        }
    if "\n" in text and len(text.splitlines()) >= 3 and all(len(line.strip()) <= 40 for line in text.splitlines() if line.strip()):
        return {
            "layout_type": "multiline_special",
            "line_groups": [{"line": line} for line in text.splitlines()],
            "must_not_merge_with_narrative": True,
            "preserve_line_breaks": True,
            "requires_structure_review": False,
        }
    return None


def decide_edit_policy(
    item: dict,
    proposed_text: str,
    *,
    image_anchor_before: Sequence[str] | None = None,
    image_anchor_after: Sequence[str] | None = None,
    physical_column_coverage_complete: bool | None = None,
    source_block_coverage_complete: bool | None = None,
) -> dict:
    """Decide whether the current edited text may be frozen.

    Locking is deliberately strict: the current edited text, the mapped
    baseline and the actual multi-model consensus must all be identical, and
    both physical-column and source-block coverage must be complete.
    """
    baseline = str(item.get("baseline_text", proposed_text) or "")
    proposed = str(proposed_text or "")
    consensus = item.get("candidate_consensus") if isinstance(item.get("candidate_consensus"), dict) else {}
    nonempty_model_count = int(consensus.get("nonempty_model_count", 0) or 0)
    unique_text_count = int(consensus.get("unique_text_count", 0) or 0)
    status_exact = bool(
        consensus.get("status") == "all_candidates_match_baseline"
        or str(consensus.get("consensus_type", "") or "") in {"all_models", "exact"}
    )
    consensus_text = str(consensus.get("consensus_text", "") or "")
    if not consensus_text and status_exact and unique_text_count == 1:
        consensus_text = baseline
    proposed_matches_baseline = proposed == baseline
    proposed_matches_consensus = bool(consensus_text) and proposed == consensus_text
    baseline_hash = sha256_text(baseline)
    proposed_hash = sha256_text(proposed)

    if physical_column_coverage_complete is None:
        physical_column_coverage_complete = bool(item.get("physical_column_coverage_complete", True))
    if source_block_coverage_complete is None:
        source_block_coverage_complete = bool(item.get("source_block_coverage_complete", True))

    exact_consensus = bool(
        status_exact
        and nonempty_model_count >= 2
        and unique_text_count == 1
        and proposed_matches_baseline
        and proposed_matches_consensus
        and proposed_hash == baseline_hash
        and physical_column_coverage_complete
        and source_block_coverage_complete
    )

    risk_reasons = [str(value) for value in (item.get("risk_reasons") or []) if str(value)]
    textual_reasons = list(dict.fromkeys(risk_reasons))
    if not proposed_matches_baseline:
        textual_reasons.append("proposed_text_differs_from_current_edited_text")
    if not proposed_matches_consensus:
        textual_reasons.append("proposed_text_differs_from_model_consensus")
    if nonempty_model_count < 2 or unique_text_count != 1 or not status_exact:
        textual_reasons.append("model_candidates_not_exact_consensus")
    if not physical_column_coverage_complete:
        textual_reasons.append("physical_column_coverage_incomplete")
    if not source_block_coverage_complete:
        textual_reasons.append("source_block_coverage_incomplete")
    if CONTROL_RE.search(proposed):
        textual_reasons.append("control_character")
    if "□" in proposed:
        textual_reasons.append("placeholder_square")
    if "�" in proposed:
        textual_reasons.append("replacement_character")
    if LATIN_RUN_RE.search(proposed) and not re.search(r"(?:https?://|ISBN|URL)", proposed, re.I):
        textual_reasons.append("abnormal_latin_run")
    if str(item.get("risk_level", "none") or "none") in {"high", "medium"} and not textual_reasons:
        textual_reasons.append("declared_text_risk")

    structure_reasons: list[str] = []
    chapter = item.get("chapter_candidate") if isinstance(item.get("chapter_candidate"), dict) else {}
    if chapter.get("requires_type_repair"):
        structure_reasons.append("chapter_type_repair")
    if str(item.get("block_type", "") or "") in {BlockType.RUBY.value, BlockType.FOOTNOTE.value}:
        structure_reasons.append("structured_inline_or_footnote")
    layout = special_layout(item, proposed)
    if layout:
        structure_reasons.append(str(layout.get("layout_type", "special_layout")))
    if image_anchor_before or image_anchor_after:
        structure_reasons.append("image_boundary_structure")

    textual_reasons = list(dict.fromkeys(textual_reasons))
    if textual_reasons:
        policy = "review_required"
        model_action_required = True
        text_locked = False
        unlock_reasons = list(dict.fromkeys(textual_reasons + structure_reasons))
    elif structure_reasons:
        policy = "structure_only"
        model_action_required = True
        text_locked = True
        unlock_reasons = []
    else:
        policy = "locked_consensus"
        model_action_required = False
        text_locked = True
        unlock_reasons = []

    return {
        "edit_policy": policy,
        "model_action_required": model_action_required,
        "text_locked": text_locked,
        "unlock_reasons": unlock_reasons,
        "structure_reasons": list(dict.fromkeys(structure_reasons)),
        "layout_structure": layout,
        "consensus_lock_eligible": exact_consensus and not textual_reasons,
        "lock_validation": {
            "proposed_matches_baseline": proposed_matches_baseline,
            "proposed_matches_consensus": proposed_matches_consensus,
            "proposed_text_sha256": proposed_hash,
            "baseline_text_sha256": baseline_hash,
            "hashes_match": proposed_hash == baseline_hash,
            "physical_column_coverage_complete": bool(physical_column_coverage_complete),
            "source_block_coverage_complete": bool(source_block_coverage_complete),
            "candidate_support": nonempty_model_count,
            "unique_text_count": unique_text_count,
            "candidate_status_exact": status_exact,
        },
    }

def _split_spans(text: str, maximum: int = 48) -> list[tuple[int, int, str]]:
    value = str(text or "")
    if not value:
        return []
    boundaries = {0, len(value)}
    for match in re.finditer(r"[。！？!?」』）】…‥\n]", value):
        boundaries.add(match.end())
    ordered = sorted(boundaries)
    raw: list[tuple[int, int]] = []
    for left, right in zip(ordered, ordered[1:]):
        if right > left:
            raw.append((left, right))
    if not raw:
        raw = [(0, len(value))]
    result: list[tuple[int, int, str]] = []
    for left, right in raw:
        cursor = left
        while cursor < right:
            end = min(right, cursor + maximum)
            result.append((cursor, end, value[cursor:end]))
            cursor = end
    return result


def _verified_manual_text(source: dict) -> str:
    flags = (
        source.get("manual_review_confirmed"),
        source.get("human_verified"),
        source.get("edited_by_user"),
        source.get("manual_text_verified"),
    )
    if any(bool(value) for value in flags):
        return str(source.get("edited_text", "") or "")
    provenance = str(source.get("edited_text_source", source.get("text_source", "")) or "").lower()
    if provenance in {"manual", "human", "user_confirmed", "manual_review"}:
        return str(source.get("edited_text", "") or "")
    return ""


def _consensus_candidate_source(source: dict) -> tuple[str, int]:
    values = [value for value in model_candidate_texts(source) if str(value or "")]
    if len(values) < 2:
        return "", 0
    grouped: dict[str, list[str]] = defaultdict(list)
    for value in values:
        grouped[normalise_text(value)].append(value)
    best_key, best_values = max(grouped.items(), key=lambda pair: (len(pair[1]), len(pair[0])))
    if not best_key or len(best_values) < 2:
        return "", len(best_values)
    return max(best_values, key=len), len(best_values)


def build_atomic_span_map(stable_records: Sequence[dict], full_items_by_id: dict[str, dict]) -> list[dict]:
    """Build source spans only from evidence that predates proposed_text.

    A missing source is represented as an explicit unverified record.  The
    proposed text is never recycled as evidence, avoiding circular proof of a
    truncated or already-corrupted target.
    """
    records: list[dict] = []
    global_order = 0
    for item in stable_records:
        item_id = str(item.get("item_id", "") or "")
        source = full_items_by_id.get(item_id) or {}
        columns = [str(value) for value in (item.get("source_column_ids") or source.get("column_ids") or []) if str(value)]
        geometry = source.get("column_geometry") if isinstance(source.get("column_geometry"), list) else []
        geometry_by_id = {str(g.get("column_id", "") or ""): g for g in geometry if isinstance(g, dict)}
        all_candidates = candidate_texts(source)
        model_candidates = model_candidate_texts(source)
        physical = source.get("physical_column_candidates") if isinstance(source.get("physical_column_candidates"), list) else []
        column_texts: dict[str, list[str]] = defaultdict(list)
        for model in physical:
            if not isinstance(model, dict):
                continue
            values = list(model.get("column_texts") or [])
            model_column_ids = [str(value) for value in (model.get("column_ids") or columns) if str(value)]
            ids = model_column_ids or columns
            for index, column_id in enumerate(ids):
                text_value = str(values[index] if index < len(values) else "" or "")
                if text_value and text_value not in column_texts[column_id]:
                    column_texts[column_id].append(text_value)

        source_units: list[tuple[str, str, list[str], str, int, bool]] = []
        usable_columns = [column_id for column_id in columns if column_texts.get(column_id)]
        if usable_columns:
            for column_id in usable_columns:
                variants = column_texts[column_id]
                grouped: dict[str, list[str]] = defaultdict(list)
                for value in variants:
                    grouped[normalise_text(value)].append(value)
                _key, support_values = max(grouped.items(), key=lambda pair: (len(pair[1]), len(pair[0])))
                selected = max(support_values, key=len)
                source_units.append((column_id, selected, variants, "physical_column", len(support_values), False))
        else:
            manual = _verified_manual_text(source)
            consensus_text, consensus_support = _consensus_candidate_source(source)
            original_block = str(
                source.get("original_ocr_block", "")
                or source.get("original_fused_text", "")
                or source.get("ocr_raw", "")
                or source.get("source_block_text", "")
                or ""
            )
            fallback_column = columns[0] if columns else f"item-{int(item.get('reading_order', 0) or 0):06d}"
            if manual:
                source_units.append((fallback_column, manual, [manual], "verified_manual", max(1, len(model_candidates)), True))
            elif consensus_text and consensus_support >= 2:
                source_units.append((fallback_column, consensus_text, model_candidates, "consensus_candidate", consensus_support, False))
            elif original_block:
                source_units.append((fallback_column, original_block, all_candidates, "original_ocr_block", _candidate_support(source, original_block), False))
            else:
                records.append({
                    "schema": "novel_formatter.ai_publication_atomic_span.v3",
                    "source_span_id": f"unverified-{re.sub(r'[^A-Za-z0-9_-]+', '_', item_id)[:64]}-{global_order:06d}",
                    "source_order": global_order,
                    "page": int(item.get("page", source.get("page", 0)) or 0),
                    "physical_column_id": fallback_column,
                    "source_scope": "unverified_no_source",
                    "source_bbox": copy.deepcopy(geometry_by_id.get(fallback_column, {}).get("bbox")),
                    "source_char_start": None,
                    "source_char_end": None,
                    "selected_source_text": "",
                    "selected_source_text_sha256": sha256_text(""),
                    "candidate_texts": copy.deepcopy(all_candidates),
                    "candidate_support": 0,
                    "candidate_count": len(model_candidates),
                    "expected_item_id": item_id,
                    "expected_reading_order": int(item.get("reading_order", 0) or 0),
                    "expected_chapter_id": str(item.get("chapter_id", "") or ""),
                    "coverage_policy": "unverified",
                    "reliability": "unverified",
                    "unverified_reason": "no_physical_column_candidate_consensus_original_block_or_verified_manual_text",
                })
                global_order += 1
                continue

        for column_id, selected, variants, scope, source_support, verified_manual in source_units:
            spans = _split_spans(selected)
            if not spans:
                spans = [(0, 0, "")]
            for local_index, (start, end, span_text) in enumerate(spans, start=1):
                normalised_span = normalise_text(span_text)
                support = sum(1 for candidate in model_candidates if normalised_span and normalised_span in normalise_text(candidate))
                if scope == "physical_column":
                    support = max(support, source_support)
                bbox = geometry_by_id.get(column_id, {}).get("bbox")
                page = int(item.get("page", source.get("page", 0)) or 0)
                span_id = f"page-{page:04d}-col-{re.sub(r'[^A-Za-z0-9_-]+', '_', column_id)[:48]}-span-{local_index:04d}-{sha256_text(span_text)[:10]}"
                target_text = str(item.get("proposed_text", "") or "")
                normalised_length = len(normalised_span)
                informative_characters = sum(
                    bool(char.isalnum() or KANA_RE.match(char) or HAN_RE.match(char))
                    for char in normalised_span
                )
                target_occurrences = target_text.count(span_text) if span_text else 0
                # Exact substring coverage is executable only for a sufficiently
                # informative source span that occurs once in the frozen target.
                # One-character punctuation (especially …, 」 and 。) previously
                # produced thousands of false duplicate/order failures in real
                # books.  Short or corrected spans remain recorded for review,
                # while the stable item hash still protects the complete target.
                exactly_once = bool(
                    span_text
                    and (support >= 2 or verified_manual)
                    and normalised_length >= 5
                    and informative_characters >= 3
                    and target_occurrences == 1
                )
                records.append({
                    "schema": "novel_formatter.ai_publication_atomic_span.v3",
                    "source_span_id": span_id,
                    "source_order": global_order,
                    "page": page,
                    "physical_column_id": column_id,
                    "source_scope": scope,
                    "source_bbox": copy.deepcopy(bbox),
                    "source_char_start": start,
                    "source_char_end": end,
                    "selected_source_text": span_text,
                    "selected_source_text_sha256": sha256_text(span_text),
                    "candidate_texts": copy.deepcopy(variants or all_candidates),
                    "candidate_support": support,
                    "candidate_count": len(model_candidates),
                    "expected_item_id": item_id,
                    "expected_reading_order": int(item.get("reading_order", 0) or 0),
                    "expected_chapter_id": str(item.get("chapter_id", "") or ""),
                    "coverage_policy": "exactly_once" if exactly_once else "review_required" if span_text else "unverified",
                    "reliability": "high" if support >= 2 or verified_manual else "medium" if support == 1 else "unverified",
                    "evidence_is_independent_of_proposed_text": True,
                    "exact_match_occurrences_in_proposed": target_occurrences,
                    "exactly_once_eligibility": {
                        "normalised_length": normalised_length,
                        "informative_character_count": informative_characters,
                        "source_support_sufficient": bool(support >= 2 or verified_manual),
                        "unique_exact_target_match": target_occurrences == 1,
                    },
                })
                global_order += 1
    return records

def _candidate_support(source: dict, span: str) -> int:
    needle = normalise_text(span)
    if not needle:
        return 0
    return sum(1 for candidate in model_candidate_texts(source) if needle in normalise_text(candidate))


def _interesting_gram(value: str) -> bool:
    if len(value) < 12:
        return False
    unique = len(set(value))
    alnum = sum(bool(ch.isalnum() or KANA_RE.match(ch) or HAN_RE.match(ch)) for ch in value)
    return unique >= 6 and alnum >= max(8, int(len(value) * 0.55))


def build_global_anomalies(stable_records: Sequence[dict], full_items_by_id: dict[str, dict]) -> dict:
    ordered = sorted(stable_records, key=lambda item: int(item.get("reading_order", 0) or 0))
    normalised = [normalise_text(item.get("proposed_text", "")) for item in ordered]
    gram_index: dict[str, list[int]] = defaultdict(list)
    for index, text in enumerate(normalised):
        if len(text) < 12:
            continue
        seen: set[str] = set()
        for pos in range(0, len(text) - 11):
            gram = text[pos:pos + 12]
            if gram in seen or not _interesting_gram(gram):
                continue
            seen.add(gram)
            gram_index[gram].append(index)

    pairs: set[tuple[int, int]] = set()
    for indices in gram_index.values():
        distinct = sorted(set(indices))
        if len(distinct) < 2 or len(distinct) > 20:
            continue
        for left_pos, left in enumerate(distinct):
            for right in distinct[left_pos + 1:]:
                if abs(right - left) > 1:
                    pairs.add((left, right))
                    if len(pairs) >= 100_000:
                        break
            if len(pairs) >= 100_000:
                break
        if len(pairs) >= 100_000:
            break

    clusters: dict[str, dict] = {}
    seen_pair_span: set[tuple[int, int, str]] = set()
    for left, right in sorted(pairs):
        a, b = normalised[left], normalised[right]
        matcher = SequenceMatcher(None, a, b, autojunk=False)
        blocks = [block for block in matcher.get_matching_blocks() if block.size >= 12]
        if not blocks:
            continue
        block = max(blocks, key=lambda value: value.size)
        span = a[block.a:block.a + block.size]
        key = sha256_text(span)
        if (left, right, key) in seen_pair_span:
            continue
        seen_pair_span.add((left, right, key))
        cluster = clusters.setdefault(key, {
            "cluster_id": f"duplicate-{key[:12]}",
            "text": span,
            "length": len(span),
            "occurrences": {},
        })
        for idx in (left, right):
            item = ordered[idx]
            item_id = str(item.get("item_id", "") or "")
            if item_id not in cluster["occurrences"]:
                cluster["occurrences"][item_id] = {
                    "item_id": item_id,
                    "reading_order": int(item.get("reading_order", 0) or 0),
                    "page": int(item.get("page", 0) or 0),
                    "chapter_id": str(item.get("chapter_id", "") or ""),
                    "source_candidate_support": _candidate_support(full_items_by_id.get(item_id) or {}, span),
                }

    duplicate_clusters: list[dict] = []
    moved: list[dict] = []
    for cluster in clusters.values():
        occurrences = sorted(cluster.pop("occurrences").values(), key=lambda value: value["reading_order"])
        if len(occurrences) < 2:
            continue
        length = int(cluster["length"])
        supports = [int(value.get("source_candidate_support", 0) or 0) for value in occurrences]
        severity = "fatal" if length >= 80 else "high" if length >= 40 else "review"
        record = {
            **cluster,
            "occurrence_count": len(occurrences),
            "occurrences": occurrences,
            "non_adjacent": True,
            "severity": severity,
            "all_occurrences_source_supported": all(value > 0 for value in supports),
        }
        duplicate_clusters.append(record)
        if max(supports, default=0) >= 2 and min(supports, default=0) == 0:
            source_occurrence = max(occurrences, key=lambda value: value.get("source_candidate_support", 0))
            for destination in occurrences:
                if int(destination.get("source_candidate_support", 0) or 0) != 0:
                    continue
                moved.append({
                    "anomaly_type": "possible_span_move",
                    "source_item_id": source_occurrence["item_id"],
                    "destination_item_id": destination["item_id"],
                    "text": record["text"],
                    "length": record["length"],
                    "source_candidate_support": source_occurrence["source_candidate_support"],
                    "destination_candidate_support": 0,
                    "severity": "fatal" if length >= 40 else "high",
                })

    unsupported: list[dict] = []
    orphan: list[dict] = []
    for item, proposed in zip(ordered, normalised):
        item_id = str(item.get("item_id", "") or "")
        source = full_items_by_id.get(item_id) or {}
        candidates = candidate_texts(source)
        if not proposed or not candidates:
            continue
        best = max((normalise_text(value) for value in candidates), key=lambda value: SequenceMatcher(None, value, proposed, autojunk=False).ratio())
        matcher = SequenceMatcher(None, best, proposed, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag in {"insert", "replace"} and j2 - j1 >= 12:
                text = proposed[j1:j2]
                unsupported.append({
                    "anomaly_type": "unsupported_insertion",
                    "item_id": item_id,
                    "reading_order": int(item.get("reading_order", 0) or 0),
                    "text": text,
                    "length": len(text),
                    "candidate_support": _candidate_support(source, text),
                    "severity": "fatal" if len(text) >= 40 else "high",
                })
            if tag in {"delete", "replace"} and i2 - i1 >= 12:
                text = best[i1:i2]
                support = _candidate_support(source, text)
                if support >= 2:
                    orphan.append({
                        "anomaly_type": "orphan_source_span",
                        "item_id": item_id,
                        "reading_order": int(item.get("reading_order", 0) or 0),
                        "text": text,
                        "length": len(text),
                        "candidate_support": support,
                        "severity": "fatal" if len(text) >= 40 else "high",
                    })

    # A true move may disappear from its source item and appear only at the
    # destination, so it will not necessarily form a duplicate in proposed
    # text. Correlate source-supported deletions with unsupported insertions.
    existing_moves = {(value.get("source_item_id"), value.get("destination_item_id"), normalise_text(value.get("text", ""))) for value in moved}
    for source_span in orphan:
        source_text = normalise_text(source_span.get("text", ""))
        if len(source_text) < 12:
            continue
        for destination_span in unsupported:
            destination_text = normalise_text(destination_span.get("text", ""))
            if len(destination_text) < 12:
                continue
            exact = source_text == destination_text
            matcher = SequenceMatcher(None, source_text, destination_text, autojunk=False)
            ratio = matcher.ratio()
            matching = max(matcher.get_matching_blocks(), key=lambda value: value.size)
            common_ratio = matching.size / max(1, min(len(source_text), len(destination_text)))
            if not exact and ratio < 0.90 and not (matching.size >= 12 and common_ratio >= 0.72):
                continue
            source_id = str(source_span.get("item_id", "") or "")
            destination_id = str(destination_span.get("item_id", "") or "")
            if not source_id or not destination_id or source_id == destination_id:
                continue
            if exact:
                text = source_span.get("text") if len(source_text) <= len(destination_text) else destination_span.get("text")
            else:
                text = source_text[matching.a:matching.a + matching.size]
            move_key = (source_id, destination_id, normalise_text(text))
            if move_key in existing_moves:
                continue
            existing_moves.add(move_key)
            length = min(len(source_text), len(destination_text))
            moved.append({
                "anomaly_type": "possible_span_move",
                "source_item_id": source_id,
                "destination_item_id": destination_id,
                "text": text,
                "length": length,
                "source_candidate_support": int(source_span.get("candidate_support", 0) or 0),
                "destination_candidate_support": int(destination_span.get("candidate_support", 0) or 0),
                "match_ratio": round(ratio, 6),
                "common_span_ratio": round(common_ratio, 6),
                "correlation": "orphan_source_span_to_unsupported_insertion",
                "severity": "fatal" if length >= 40 else "high",
            })

    order_conflicts: list[dict] = []
    previous_index = -1
    previous_item = ""
    for item in ordered:
        indices = [int(value) for value in (item.get("primary_block_indices") or []) if isinstance(value, int) or str(value).isdigit()]
        if not indices and item.get("primary_block_index") is not None:
            try:
                indices = [int(item.get("primary_block_index"))]
            except (TypeError, ValueError):
                indices = []
        if not indices:
            continue
        current = min(indices)
        if previous_index >= 0 and current < previous_index:
            order_conflicts.append({
                "anomaly_type": "cross_item_order_conflict",
                "previous_item_id": previous_item,
                "current_item_id": str(item.get("item_id", "") or ""),
                "previous_source_block_index": previous_index,
                "current_source_block_index": current,
                "severity": "fatal",
            })
        previous_index = max(indices)
        previous_item = str(item.get("item_id", "") or "")

    duplicate_clusters.sort(key=lambda value: (-int(value["length"]), value["cluster_id"]))
    return {
        "schema": "novel_formatter.ai_publication_global_text_anomalies.v3",
        "normalisation": "NFC; remove whitespace only; preserve punctuation",
        "minimum_span_length": 12,
        "duplicate_clusters": duplicate_clusters,
        "moved_span_clusters": moved,
        "unsupported_insertions": unsupported,
        "orphan_source_spans": orphan,
        "cross_item_order_conflicts": order_conflicts,
        "summary": {
            "duplicate_cluster_count": len(duplicate_clusters),
            "moved_span_count": len(moved),
            "unsupported_insertion_count": len(unsupported),
            "orphan_source_span_count": len(orphan),
            "cross_item_order_conflict_count": len(order_conflicts),
            "fatal_or_high_count": sum(
                1 for group in (duplicate_clusters, moved, unsupported, orphan, order_conflicts)
                for value in group if value.get("severity") in {"fatal", "high"}
            ),
        },
    }


def _quote_balance(text: str) -> dict:
    return {
        "dialogue": text.count("「") - text.count("」"),
        "citation": text.count("『") - text.count("』"),
        "parenthesis": text.count("（") - text.count("）"),
    }


def _window_item(item: dict) -> dict:
    return {
        "item_id": str(item.get("item_id", "") or ""),
        "page": int(item.get("page", 0) or 0),
        "chapter_id": str(item.get("chapter_id", "") or ""),
        "block_type": str(item.get("block_type", "") or ""),
        "baseline_text": str(item.get("baseline_text", "") or ""),
        "candidate_consensus": copy.deepcopy(item.get("candidate_consensus") or {}),
        "column_ids": copy.deepcopy(item.get("column_ids") or []),
        "column_geometry": copy.deepcopy(item.get("column_geometry") or []),
        "risk_reasons": copy.deepcopy(item.get("risk_reasons") or []),
    }


def _build_dynamic_boundary_windows_full(items: Sequence[dict], assets_manifest: dict) -> dict:
    ordered = sorted(items, key=lambda item: int(item.get("reading_order", item.get("row_index", 0)) or 0))
    image_after = {str(asset.get("stable_item_before", "") or "") for asset in assets_manifest.get("assets", []) if asset.get("include_in_final_epub")}
    image_before = {str(asset.get("stable_item_after", "") or "") for asset in assets_manifest.get("assets", []) if asset.get("include_in_final_epub")}
    trigger_reasons = {
        "possible_missing_column", "possible_column_order_error", "cross_item_boundary", "duplicate_suffix",
        "unbalanced_quote", "model_length_disagreement", "model_text_disagreement", "numeric_disagreement",
    }
    windows: list[dict] = []
    seen: set[tuple[str, ...]] = set()
    for focus, item in enumerate(ordered):
        reasons = set(str(value) for value in (item.get("risk_reasons") or []))
        if str(item.get("risk_level", "none") or "none") not in {"high", "medium"} and not reasons.intersection(trigger_reasons):
            continue
        chapter = str(item.get("chapter_id", "") or "")
        left = focus
        chars = len(str(item.get("baseline_text", "") or ""))
        while left > 0 and focus - left < 8 and chars < 1600:
            previous = ordered[left - 1]
            if str(previous.get("chapter_id", "") or "") != chapter:
                break
            previous_text = str(previous.get("baseline_text", "") or "")
            left -= 1
            chars += len(previous_text)
            balance = _quote_balance("".join(str(value.get("baseline_text", "") or "") for value in ordered[left:focus + 1]))
            if previous_text.rstrip().endswith(TERMINAL) and not any(balance.values()):
                break
        right = focus + 1
        while right < len(ordered) and right - focus <= 8 and chars < 1600:
            current = ordered[right - 1]
            combined = "".join(str(value.get("baseline_text", "") or "") for value in ordered[left:right])
            if combined.rstrip().endswith(TERMINAL) and not any(_quote_balance(combined).values()) and right > focus + 1:
                break
            if right >= len(ordered) or str(ordered[right].get("chapter_id", "") or "") != chapter:
                break
            chars += len(str(ordered[right].get("baseline_text", "") or ""))
            right += 1
        members = ordered[left:right]
        ids = tuple(str(value.get("item_id", "") or "") for value in members)
        if not ids or ids in seen:
            continue
        seen.add(ids)
        combined = "\n\n".join(str(value.get("baseline_text", "") or "") for value in members)
        windows.append({
            "window_id": f"syntax-boundary-{len(windows)+1:05d}",
            "focus_item_id": str(item.get("item_id", "") or ""),
            "trigger_reasons": sorted(reasons.intersection(trigger_reasons) or reasons),
            "window_strategy": "expand_to_sentence_and_balanced_quote_boundary",
            "item_ids": list(ids),
            "combined_baseline": combined,
            "quote_balance": _quote_balance(combined),
            "contains_image_boundary": any(value in image_after or value in image_before for value in ids),
            "items": [_window_item(value) for value in members],
        })

    pages: dict[int, list[dict]] = defaultdict(list)
    for item in ordered:
        pages[int(item.get("page", 0) or 0)].append(item)
    page_numbers = sorted(page for page in pages if page > 0)
    page_windows: list[dict] = []
    publication_image_pages = {
        int(source.get("page_no", 0) or 0)
        for asset in assets_manifest.get("assets", []) if asset.get("include_in_final_epub")
        for source in (asset.get("source_records") or []) if isinstance(source, dict)
    }
    for left_page, right_page in zip(page_numbers, page_numbers[1:]):
        left_items, right_items = pages[left_page], pages[right_page]
        if not left_items or not right_items:
            continue
        members = left_items[-3:] + right_items[:3]
        gap_pages = list(range(left_page + 1, right_page))
        page_windows.append({
            "window_id": f"page-boundary-{left_page:05d}-{right_page:05d}",
            "left_page": left_page,
            "right_page": right_page,
            "is_adjacent_physical_page": right_page == left_page + 1,
            "page_gap": max(0, right_page - left_page - 1),
            "intervening_pages": gap_pages,
            "intervening_publication_image_pages": [page for page in gap_pages if page in publication_image_pages],
            "left_last_item_id": str(left_items[-1].get("item_id", "") or ""),
            "right_first_item_id": str(right_items[0].get("item_id", "") or ""),
            "item_ids": [str(value.get("item_id", "") or "") for value in members],
            "combined_baseline": "\n\n".join(str(value.get("baseline_text", "") or "") for value in members),
            "requires_continuation_review": not str(left_items[-1].get("baseline_text", "") or "").rstrip().endswith(TERMINAL),
            "items": [_window_item(value) for value in members],
        })
    return {
        "schema": "novel_formatter.ai_publication_boundary_windows.v3",
        "window_strategy": "syntax_aware_dynamic",
        "maximum_items_each_side": 8,
        "maximum_character_window": 1600,
        "risk_windows": windows,
        "page_boundary_windows": page_windows,
    }




def build_dynamic_boundary_windows(items: Sequence[dict], assets_manifest: dict, *, compact: bool = False) -> dict:
    full = _build_dynamic_boundary_windows_full(items, assets_manifest)
    if not compact:
        return full
    high_value = {
        "unbalanced_quote", "cross_item_boundary", "possible_missing_column",
        "possible_column_order_error", "duplicate_suffix", "status_boundary",
        "image_boundary", "chapter_title_glued_to_body", "sentence_move",
    }
    risk_windows = []
    for window in full.get("risk_windows") or []:
        reasons = set(window.get("trigger_reasons") or [])
        if not (reasons & high_value or window.get("contains_image_boundary")):
            continue
        item_ids = list(window.get("item_ids") or [])
        members = list(window.get("items") or [])
        centre_id = str(window.get("focus_item_id", "") or "")
        centre_index = item_ids.index(centre_id) if centre_id in item_ids else max(0, len(item_ids) // 2)
        left = members[:centre_index]
        centre = members[centre_index] if centre_index < len(members) else {}
        right = members[centre_index + 1:]
        left_text = "\n".join(str(value.get("baseline_text", "") or "") for value in left)[-120:]
        centre_text = str(centre.get("baseline_text", "") or "")[:240]
        right_text = "\n".join(str(value.get("baseline_text", "") or "") for value in right)[:120]
        risk_windows.append({
            "window_id": window.get("window_id"),
            "center_item_id": centre_id,
            "left_item_ids": item_ids[:centre_index],
            "right_item_ids": item_ids[centre_index + 1:],
            "left_excerpt": left_text,
            "center_excerpt": centre_text,
            "right_excerpt": right_text,
            "reason": sorted(reasons),
            "quote_balance": window.get("quote_balance"),
            "contains_image_boundary": bool(window.get("contains_image_boundary")),
        })
    page_windows = []
    for window in full.get("page_boundary_windows") or []:
        if not window.get("requires_continuation_review") and not window.get("intervening_publication_image_pages"):
            continue
        combined = str(window.get("combined_baseline", "") or "")
        page_windows.append({
            "window_id": window.get("window_id"),
            "left_page": window.get("left_page"),
            "right_page": window.get("right_page"),
            "is_adjacent_physical_page": window.get("is_adjacent_physical_page"),
            "page_gap": window.get("page_gap"),
            "intervening_publication_image_pages": window.get("intervening_publication_image_pages") or [],
            "left_last_item_id": window.get("left_last_item_id"),
            "right_first_item_id": window.get("right_first_item_id"),
            "excerpt": combined[:360],
            "requires_continuation_review": bool(window.get("requires_continuation_review")),
        })
    return {
        "schema": "novel_formatter.ai_publication_boundary_windows.v4",
        "package_mode": "compact",
        "window_strategy": "id_indexed_high_value_excerpts",
        "risk_windows": risk_windows,
        "page_boundary_windows": page_windows,
        "omitted_low_value_window_count": max(0, len(full.get("risk_windows") or []) - len(risk_windows)),
    }

def build_term_graph(repair_map: dict, full_items_by_id: dict[str, dict], reference_alignment: Sequence[dict] | None = None) -> dict:
    variant_sources: dict[str, dict] = defaultdict(lambda: {
        "occurrences": 0,
        "model_support": Counter(),
        "reference_support": 0,
        "contexts": [],
    })
    term_re = re.compile(r"[ァ-ヶー・]{3,32}|[一-龯々〆ヵヶ]{2,12}|[A-Za-z][A-Za-z0-9_.+-]{2,32}")
    for item in repair_map.get("items") or []:
        item_id = str(item.get("item_id", "") or "")
        baseline = str(item.get("baseline_text", "") or "")
        for term in term_re.findall(baseline):
            entry = variant_sources[term]
            entry["occurrences"] += 1
            if len(entry["contexts"]) < 5:
                entry["contexts"].append({"item_id": item_id, "excerpt": baseline[:160]})
        source = full_items_by_id.get(item_id) or {}
        for model_index, candidate in enumerate(source.get("candidates") or []):
            text = str(candidate.get("text", "") if isinstance(candidate, dict) else candidate or "")
            label = str(candidate.get("model_label", model_index) if isinstance(candidate, dict) else model_index)
            for term in set(term_re.findall(text)):
                variant_sources[term]["model_support"][label] += 1
    for alignment in reference_alignment or []:
        text = str(alignment.get("reference_text", "") or "")
        for term in set(term_re.findall(text)):
            variant_sources[term]["reference_support"] += 1

    terms = sorted(variant_sources)
    # The old implementation compared every term with every other term.  A
    # 400-page book can easily contain 4,000+ distinct candidates, making this
    # stage tens of millions of SequenceMatcher calls and blocking export for
    # minutes.  A ratio >= 0.78 necessarily shares substantial contiguous
    # material, so use a character-bigram inverted index and the mathematical
    # maximum ratio from string lengths before invoking SequenceMatcher.
    gram_index: dict[str, set[int]] = defaultdict(set)
    term_grams: list[set[str]] = []
    for index, value in enumerate(terms):
        grams = {value[pos:pos + 2] for pos in range(max(0, len(value) - 1))}
        term_grams.append(grams)
        for gram in grams:
            gram_index[gram].add(index)

    groups: list[dict] = []
    used: set[str] = set()
    for term_index, term in enumerate(terms):
        if term in used:
            continue
        candidate_indices: set[int] = set()
        for gram in term_grams[term_index]:
            candidate_indices.update(gram_index.get(gram, ()))
        related = []
        for other_index in sorted(candidate_indices):
            if other_index == term_index:
                continue
            other = terms[other_index]
            if other in used:
                continue
            maximum_ratio = (2.0 * min(len(term), len(other))) / max(1, len(term) + len(other))
            if maximum_ratio < 0.78:
                continue
            if SequenceMatcher(None, term, other, autojunk=False).ratio() >= 0.78:
                related.append(other)
        group_terms = [term] + related
        if len(group_terms) < 2:
            continue
        candidates = []
        for variant in group_terms:
            entry = variant_sources[variant]
            model_total = sum(entry["model_support"].values())
            score = entry["occurrences"] + model_total * 1.5 + entry["reference_support"] * 5.0
            candidates.append({
                "variant": variant,
                "occurrences": entry["occurrences"],
                "model_support": dict(entry["model_support"]),
                "reference_support": entry["reference_support"],
                "context_consistency": min(1.0, 0.5 + 0.1 * len(entry["contexts"])),
                "canonical_score": round(score, 4),
                "contexts": entry["contexts"],
            })
        candidates.sort(key=lambda value: (-value["canonical_score"], value["variant"]))
        canonical = candidates[0]["variant"]
        for candidate in candidates[1:]:
            candidate["canonical_target"] = canonical
        groups.append({"canonical": canonical, "variants": candidates})
        used.update(group_terms)

    all_text = "\n".join(str(item.get("baseline_text", "") or "") for item in repair_map.get("items") or [])
    numbers = Counter(NUMBER_RE.findall(all_text))
    punctuation = {
        "horizontal_bar": {char: all_text.count(char) for char in "―—─━‐‑‒–" if all_text.count(char)},
        "ellipsis": {char: all_text.count(char) for char in ("……", "…", "‥") if all_text.count(char)},
        "question_exclamation": {char: all_text.count(char) for char in ("！", "!", "？", "?", "!?", "？！", "!!") if all_text.count(char)},
    }
    return {
        "schema": "novel_formatter.ai_publication_term_consistency.v3",
        "canonical_candidate_graph": groups,
        "term_count": len(variant_sources),
        "numbers_levels_units": [{"form": key, "count": count} for key, count in numbers.most_common(1000)],
        "punctuation_style": punctuation,
        "scoring_policy": ["publication_reference", "model_support", "context_consistency", "frequency", "unicode_shape"],
    }


def enrich_asset_display_plan(assets_manifest: dict) -> None:
    for asset in assets_manifest.get("assets") or []:
        if not asset.get("include_in_final_epub"):
            continue
        asset_id = str(asset.get("asset_id", "asset") or "asset")
        role = str(asset.get("publication_role", asset.get("role", "illustration")) or "illustration")
        asset.update({
            "display_mode": "standalone_page",
            "force_split_before": role != "cover",
            "force_split_after": role != "cover",
            "planned_asset_xhtml": f"EPUB/text/{asset_id}.xhtml",
            "spine_position_after_item": str(asset.get("stable_item_before", "") or "") or None,
            "spine_position_before_item": str(asset.get("stable_item_after", "") or "") or None,
            "nav_entry_required": False,
            "page_spread": "center",
            "preserve_original_bytes": True,
        })


def _page_sources(package: dict) -> dict[int, dict]:
    result: dict[int, dict] = {}
    for asset in package.get("assets") or []:
        if not isinstance(asset, dict) or str(asset.get("kind", "") or "") != "page":
            continue
        try:
            page = int(asset.get("page_no", asset.get("page", 0)) or 0)
        except (TypeError, ValueError):
            continue
        path = str(asset.get("image_path", "") or "")
        if page and path:
            result[page] = asset
    return result



def _normalise_bbox(value: Any, width: int = 0, height: int = 0) -> list[float] | None:
    bbox = value
    if isinstance(bbox, dict):
        bbox = [bbox.get("x", 0), bbox.get("y", 0), bbox.get("w", bbox.get("width", 0)), bbox.get("h", bbox.get("height", 0))]
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x, y, w, h = map(float, bbox)
    except (TypeError, ValueError, OverflowError):
        return None
    if width > 0 and height > 0 and max(abs(x), abs(y), abs(w), abs(h)) > 1.5:
        x, w = x / width, w / width
        y, h = y / height, h / height
    if w < 0:
        x, w = x + w, -w
    if h < 0:
        y, h = y + h, -h
    return [max(0.0, min(1.0, x)), max(0.0, min(1.0, y)), max(0.0, min(1.0, w)), max(0.0, min(1.0, h))]


def _bbox_match_score(left: Sequence[float], right: Sequence[float]) -> float:
    """Score vertical-column geometry with horizontal alignment as authority.

    OCR engines often disagree substantially about the top/bottom extent of the
    same Japanese vertical column.  Full rectangle IoU therefore creates false
    "missing column" alarms.  The score follows the production rule: horizontal
    overlap 65%, horizontal centre distance 25%, vertical overlap 10%.
    """
    lx, ly, lw, lh = map(float, left)
    rx, ry, rw, rh = map(float, right)
    l2, lb = lx + lw, ly + lh
    r2, rb = rx + rw, ry + rh
    ix = max(0.0, min(l2, r2) - max(lx, rx))
    iy = max(0.0, min(lb, rb) - max(ly, ry))
    horizontal = ix / max(1e-9, min(lw, rw))
    vertical = iy / max(1e-9, min(lh, rh))
    centre_gap = abs((lx + lw / 2.0) - (rx + rw / 2.0))
    centre_scale = max(1e-9, max(lw, rw) * 1.6)
    centre = max(0.0, 1.0 - centre_gap / centre_scale)
    return horizontal * 0.65 + centre * 0.25 + vertical * 0.10


def _bbox_x_gap(left: Sequence[float], right: Sequence[float]) -> float:
    lx, _ly, lw, _lh = map(float, left)
    rx, _ry, rw, _rh = map(float, right)
    return max(0.0, max(lx, rx) - min(lx + lw, rx + rw))


def _bbox_vertical_overlap(left: Sequence[float], right: Sequence[float]) -> float:
    _lx, ly, _lw, lh = map(float, left)
    _rx, ry, _rw, rh = map(float, right)
    overlap = max(0.0, min(ly + lh, ry + rh) - max(ly, ry))
    return overlap / max(1e-9, min(lh, rh))


def pair_ruby_side_column(slot: dict, production_columns: Sequence[dict]) -> dict | None:
    """Pair a narrow secondary slot with a neighbouring production body column."""
    bbox = _normalise_bbox(slot.get("bbox"))
    if not bbox:
        return None
    sx, sy, sw, sh = map(float, bbox)
    best: dict | None = None
    best_score = 0.0
    for body in production_columns:
        body_bbox = _normalise_bbox(body.get("bbox"))
        if not body_bbox:
            continue
        bx, by, bw, bh = map(float, body_bbox)
        width_ratio = sw / max(1e-9, bw)
        height_ratio = sh / max(1e-9, bh)
        gap = _bbox_x_gap(bbox, body_bbox) / max(1e-9, bw)
        vertical = _bbox_vertical_overlap(bbox, body_bbox)
        # Ruby in vertical Japanese is normally a short, narrow side band.  We
        # accept either side because scans can be mirrored/rotated upstream.
        narrow = max(0.0, min(1.0, (0.82 - width_ratio) / 0.55))
        short = max(0.0, min(1.0, (0.92 - height_ratio) / 0.70))
        close = max(0.0, 1.0 - gap / 1.15)
        score = narrow * 0.42 + short * 0.18 + close * 0.22 + vertical * 0.18
        # A normal-width, full-height column must never be suppressed merely
        # because it is close to another column.
        if width_ratio > 0.70 or height_ratio > 0.94 or gap > 1.25:
            continue
        if score > best_score:
            best_score = score
            best = {
                "paired_body_slot_id": body.get("column_id"),
                "paired_item_ids": copy.deepcopy(body.get("item_ids") or []),
                "pairing_score": round(score, 6),
                "width_ratio": round(width_ratio, 6),
                "height_ratio": round(height_ratio, 6),
                "horizontal_gap_ratio": round(gap, 6),
                "vertical_overlap": round(vertical, 6),
                "side": "left" if sx + sw / 2 < bx + bw / 2 else "right",
            }
    return best if best and best_score >= 0.62 else None


def classify_column_slot(
    slot: dict,
    *,
    production_columns: Sequence[dict],
    calibrated_minimum: float,
) -> dict:
    """Classify a re-detected slot without letting it overrule production data."""
    result = copy.deepcopy(slot)
    pairing = pair_ruby_side_column(result, production_columns)
    body = float(result.get("body_column_probability", 0.0) or 0.0)
    ruby = float(result.get("ruby_probability", 0.0) or 0.0)
    ink = float(result.get("ink_density", 0.0) or 0.0)
    support = int(result.get("independent_support", 1) or 1)
    if pairing:
        result.update({
            "classification": "ruby_side_column",
            "ruby_pairing": pairing,
            "body_obligation": False,
            "severity": "none",
            "requires_visual_review": bool(
                float((pairing or {}).get("width_ratio", 0.0) or 0.0) >= 0.52
                or float((pairing or {}).get("height_ratio", 0.0) or 0.0) >= 0.72
                or float((pairing or {}).get("pairing_score", 1.0) or 1.0) < 0.78
            ),
        })
    elif body >= 0.90 and ruby <= 0.25 and ink >= calibrated_minimum and support >= 2:
        result.update({
            "classification": "fatal_body_column_gap",
            "body_obligation": True,
            "severity": "fatal",
        })
    elif body >= 0.76 and ruby <= 0.45 and ink >= calibrated_minimum * 0.65:
        result.update({
            "classification": "suspected_gap",
            "body_obligation": True,
            "severity": "review",
        })
    else:
        result.update({
            "classification": "ruby_or_noise",
            "body_obligation": False,
            "severity": "none",
        })
    result["classification_inputs"] = {
        "body_column_probability": round(body, 6),
        "ruby_probability": round(ruby, 6),
        "ink_density": round(ink, 6),
        "calibrated_minimum": round(calibrated_minimum, 6),
        "independent_support": support,
    }
    return result


def _synthetic_detected_columns(package: dict, page: int) -> list[dict]:
    raw = package.get("detected_body_columns")
    if isinstance(raw, dict):
        raw = raw.get(str(page), raw.get(page, []))
    if not isinstance(raw, list):
        return []
    result = []
    for index, value in enumerate(raw, start=1):
        if not isinstance(value, dict):
            continue
        bbox = _normalise_bbox(value.get("bbox"))
        if bbox:
            result.append({
                "slot_id": str(value.get("slot_id", f"page-{page:04d}-slot-{index:03d}")),
                "bbox": bbox,
                "ink_density": float(value.get("ink_density", 0.2) or 0.0),
                "ruby_probability": float(value.get("ruby_probability", 0.0) or 0.0),
                "body_column_probability": float(value.get("body_column_probability", 1.0) or 0.0),
                "estimated_chars": int(value.get("estimated_chars", 0) or 0),
                # Explicit fixture slots represent a deliberately asserted
                # second signal, preserving deterministic tests for true gaps.
                "independent_support": int(value.get("independent_support", 2) or 1),
                "detector_source": "package_fixture",
            })
    return result


def _image_ink_density(image: Any, bbox: Sequence[float]) -> float:
    try:
        width, height = image.size
        x, y, w, h = map(float, bbox)
        left = max(0, int(x * width))
        top = max(0, int(y * height))
        right = min(width, max(left + 1, int((x + w) * width)))
        bottom = min(height, max(top + 1, int((y + h) * height)))
        crop = image.crop((left, top, right, bottom)).convert("L")
        histogram = crop.histogram()
        dark = sum(histogram[:192])
        return round(dark / max(1, crop.width * crop.height), 6)
    except Exception:
        return 0.0


def _production_support(source: dict) -> int:
    physical = source.get("physical_column_candidates") if isinstance(source.get("physical_column_candidates"), list) else []
    supported = 0
    for candidate in physical:
        if not isinstance(candidate, dict):
            continue
        texts = candidate.get("column_texts") if isinstance(candidate.get("column_texts"), list) else []
        if any(str(value or "").strip() for value in texts) or candidate.get("column_geometry"):
            supported += 1
    if supported:
        return supported
    candidates = source.get("candidates") if isinstance(source.get("candidates"), list) else []
    return max(1, sum(bool(str((value or {}).get("text", "") if isinstance(value, dict) else value).strip()) for value in candidates))


def _column_audit_inventory(package: dict) -> dict[int, dict[str, dict]]:
    """Collect runtime column IDs independently reported by OCR engines.

    These IDs are production evidence even when a later sentence-alignment step
    failed to create an editable item.  Model audit lists never supply text or
    geometry by themselves, but two or more engines reporting the same ID are
    an independent signal that the physical column existed.
    """
    inventory: dict[int, dict[str, dict]] = defaultdict(dict)
    sources = package.get("model_sources") if isinstance(package.get("model_sources"), list) else []
    if not sources:
        sources = [{
            "model_index": 0,
            "model_label": "structure_document",
            "metadata": (package.get("structure_document") or {}).get("metadata") or {},
        }]
    for source_index, source in enumerate(sources):
        if not isinstance(source, dict):
            continue
        metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
        audit = metadata.get("column_ocr_audit") if isinstance(metadata.get("column_ocr_audit"), dict) else {}
        page_map = audit.get("pages") if isinstance(audit.get("pages"), dict) else {}
        model_key = str(source.get("model_label") or source.get("source_engine") or source.get("model_index", source_index))
        for raw_page, page_record in page_map.items():
            if not isinstance(page_record, dict):
                continue
            try:
                page = int(raw_page)
            except (TypeError, ValueError):
                continue
            for raw_id in page_record.get("column_ids") or []:
                column_id = str(raw_id or "").strip()
                if not column_id:
                    continue
                target = inventory[page].setdefault(column_id, {
                    "column_id": column_id, "model_keys": set(), "page": page,
                })
                target["model_keys"].add(model_key)
    for page_map in inventory.values():
        for target in page_map.values():
            target["independent_support"] = len(target.pop("model_keys", set()))
    return inventory


def _column_order_key(column_id: str) -> tuple[int, str]:
    match = re.search(r"(?:c|col(?:umn)?[-_:]?)(\d+)$", str(column_id or ""), re.I)
    if match:
        return int(match.group(1)), str(column_id)
    numbers = re.findall(r"\d+", str(column_id or ""))
    return (int(numbers[-1]) if numbers else 10**9), str(column_id)


def _promote_regular_grid_support(slots: Sequence[dict], production_columns: Sequence[dict]) -> None:
    """Add a second geometric signal only for convincing body-column grids."""
    body_boxes = [
        _normalise_bbox(value.get("bbox")) for value in production_columns
        if _normalise_bbox(value.get("bbox"))
    ]
    candidate_slots = [
        value for value in slots
        if float(value.get("body_column_probability", 0.0) or 0.0) >= 0.88
        and float(value.get("ruby_probability", 1.0) or 1.0) <= 0.28
    ]
    if not candidate_slots:
        return
    if body_boxes:
        widths = sorted(value[2] for value in body_boxes)
        heights = sorted(value[3] for value in body_boxes)
        median_w = widths[len(widths) // 2]
        median_h = heights[len(heights) // 2]
        centers = sorted(value[0] + value[2] / 2 for value in body_boxes)
        gaps = sorted(abs(centers[index] - centers[index - 1]) for index in range(1, len(centers)))
        median_gap = gaps[len(gaps) // 2] if gaps else median_w * 1.8
        for slot in candidate_slots:
            bbox = _normalise_bbox(slot.get("bbox"))
            if not bbox:
                continue
            width_ok = 0.62 <= bbox[2] / max(1e-9, median_w) <= 1.48
            height_ok = 0.55 <= bbox[3] / max(1e-9, median_h) <= 1.35
            center = bbox[0] + bbox[2] / 2
            grid_distance = min((abs(center - value) for value in centers), default=99.0)
            gap_ok = 0.55 * median_gap <= grid_distance <= 1.55 * median_gap
            if width_ok and height_ok and gap_ok:
                slot["independent_support"] = max(2, int(slot.get("independent_support", 1) or 1))
                slot["support_signals"] = sorted(set(slot.get("support_signals") or []) | {
                    "component_geometry_detector", "production_column_grid",
                })
        return

    # An entirely omitted body page can still be recognised when several
    # mutually consistent full-height columns form a regular grid.  One lone
    # component is never promoted.
    if len(candidate_slots) < 3:
        return
    boxes = [_normalise_bbox(value.get("bbox")) for value in candidate_slots]
    boxes = [value for value in boxes if value]
    if len(boxes) < 3:
        return
    widths = sorted(value[2] for value in boxes)
    heights = sorted(value[3] for value in boxes)
    median_w = widths[len(widths) // 2]
    median_h = heights[len(heights) // 2]
    centers = sorted(value[0] + value[2] / 2 for value in boxes)
    gaps = [centers[index] - centers[index - 1] for index in range(1, len(centers))]
    median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 0.0
    consistent = (
        median_w > 0 and median_h > 0 and median_gap > 0
        and sum(0.58 <= value[2] / median_w <= 1.55 and 0.62 <= value[3] / median_h <= 1.35 for value in boxes) >= max(3, int(len(boxes) * 0.75))
        and sum(0.55 <= gap / median_gap <= 1.55 for gap in gaps) >= max(2, int(len(gaps) * 0.70))
    )
    if consistent:
        for slot in candidate_slots:
            slot["independent_support"] = max(2, int(slot.get("independent_support", 1) or 1))
            slot["support_signals"] = sorted(set(slot.get("support_signals") or []) | {
                "component_geometry_detector", "regular_page_column_grid",
            })


def _adjacent_item_ids(slot_bbox: Sequence[float], production_columns: Sequence[dict], *, limit: int = 2) -> list[str]:
    sx = float(slot_bbox[0]) + float(slot_bbox[2]) / 2.0
    ranked = []
    for body in production_columns:
        bbox = _normalise_bbox(body.get("bbox"))
        if not bbox:
            continue
        bx = bbox[0] + bbox[2] / 2.0
        ranked.append((abs(sx - bx), list(body.get("item_ids") or [])))
    result: list[str] = []
    for _distance, ids in sorted(ranked, key=lambda value: value[0]):
        for item_id in ids:
            item_id = str(item_id or "")
            if item_id and item_id not in result:
                result.append(item_id)
                if len(result) >= limit:
                    return result
    return result


def _compact_ledger_page(record: dict) -> dict:
    status = str(record.get("status", "") or "")
    base = {
        "page": record.get("page"),
        "status": status,
        "production_column_count": record.get("production_column_count", 0),
        "mapped_item_count": record.get("mapped_item_count", 0),
        "geometry_sha256": record.get("geometry_sha256", ""),
        "coverage_complete": record.get("coverage_complete", False),
        "page_review_required": record.get("page_review_required", False),
        "page_review_reasons": copy.deepcopy(record.get("page_review_reasons") or []),
        "text_items_unlocked": copy.deepcopy(record.get("text_items_unlocked") or []),
    }
    if status in {"fatal_body_column_gap", "suspected_gap", "unverifiable"}:
        for key in (
            "fatal_slots", "review_slots", "ruby_side_columns", "suspicious_ruby_side_columns", "detector_error",
            "mapped_columns_missing_from_redetection", "production_columns_without_stable_items", "related_item_ids",
        ):
            if record.get(key):
                base[key] = copy.deepcopy(record[key])
    elif status == "detector_disagreement":
        base["detector_disagreement_count"] = len(record.get("mapped_columns_missing_from_redetection") or [])
    return base


def build_page_column_ledger(
    package: dict,
    stable_records: Sequence[dict],
    full_items_by_id: dict[str, dict],
    *,
    compact: bool = False,
) -> dict:
    """Audit physical columns while treating production segmentation as authority.

    Re-detection is a secondary cross-check only.  Failure to reproduce a
    production column is recorded as detector disagreement and never unlocks an
    otherwise supported text item.  A fatal gap requires at least two
    independent signals plus strong body geometry.
    """
    page_sources = _page_sources(package)
    stable_by_page: dict[int, list[dict]] = defaultdict(list)
    for record in stable_records:
        stable_by_page[int(record.get("page", 0) or 0)].append(record)
    mapped_by_page: dict[int, dict[str, dict]] = defaultdict(dict)
    for item_id, source in full_items_by_id.items():
        page = int(source.get("page", 0) or 0)
        geometries = source.get("column_geometry") if isinstance(source.get("column_geometry"), list) else []
        geometry_by_id = {
            str(value.get("column_id", "") or ""): value
            for value in geometries if isinstance(value, dict) and str(value.get("column_id", "") or "")
        }
        ids = [str(value) for value in (source.get("column_ids") or []) if str(value)] or list(geometry_by_id)
        support = _production_support(source)
        for column_id in ids:
            target = mapped_by_page[page].setdefault(column_id, {
                "column_id": column_id,
                "bbox": copy.deepcopy((geometry_by_id.get(column_id) or {}).get("bbox")),
                "item_ids": [],
                "independent_support": support,
                "production_source": "ocr_runtime_column_geometry",
            })
            target["independent_support"] = max(int(target.get("independent_support", 1) or 1), support)
            if item_id not in target["item_ids"]:
                target["item_ids"].append(item_id)
            if target.get("bbox") is None and geometry_by_id.get(column_id):
                target["bbox"] = copy.deepcopy(geometry_by_id[column_id].get("bbox"))

    audit_inventory = _column_audit_inventory(package)
    for page, columns in audit_inventory.items():
        for column_id, audit_record in columns.items():
            target = mapped_by_page[page].setdefault(column_id, {
                "column_id": column_id, "bbox": None, "item_ids": [],
                "independent_support": 0, "production_source": "ocr_runtime_column_audit",
            })
            target["audit_independent_support"] = max(
                int(target.get("audit_independent_support", 0) or 0),
                int(audit_record.get("independent_support", 0) or 0),
            )
            target["independent_support"] = max(
                int(target.get("independent_support", 0) or 0),
                int(audit_record.get("independent_support", 0) or 0),
            )
            if target.get("production_source") != "ocr_runtime_column_geometry":
                target["production_source"] = "ocr_runtime_column_audit"

    # Include body pages that vanished before item alignment; non-body pages
    # are included only when a stable/production item explicitly references
    # them (for example text immediately after an illustration anchor).
    base_pages = set(stable_by_page) | set(mapped_by_page)
    page_records: list[dict] = []
    fatal_count = review_count = 0
    detector_errors: list[dict] = []
    status_counts: Counter[str] = Counter()
    non_body_page_types = {
        "cover", "title", "title_page", "frontispiece", "toc", "table_of_contents",
        "illustration", "color_illustration", "monochrome_illustration", "image",
        "afterword", "postscript", "colophon", "copyright", "blank", "advertisement",
        "map", "character_profile",
    }
    pages = sorted(base_pages | {
        page for page, source in page_sources.items()
        if str((source or {}).get("page_type", "") or "").strip().lower() not in non_body_page_types
    })
    for page in pages:
        if page <= 0:
            continue
        page_source = page_sources.get(page) or {}
        page_type = str(page_source.get("page_type", "") or "").strip().lower()
        skip_non_body_page = page_type in non_body_page_types
        source_path = Path(str(page_source.get("image_path", "") or "")).expanduser()
        width = height = 0
        image = None
        detected: list[dict] = _synthetic_detected_columns(package, page)
        detector_source = "package_fixture" if detected else "component_geometry_detector"
        detector_error = ""
        if not detected and source_path.is_file() and not skip_non_body_page:
            try:
                from PIL import Image
                from adapters.column_ocr_adapter import detect_vertical_columns, column_detector_version
                image = Image.open(source_path).convert("RGB")
                width, height = image.size
                fixed_region = page_source.get("fixed_region_rect") or package.get("column_fixed_region_rect")
                columns = detect_vertical_columns(
                    image,
                    sensitivity=int(package.get("column_sensitivity", 55) or 55),
                    padding_percent=0,
                    max_columns=int(package.get("max_columns", 80) or 80),
                    fixed_region_rect=fixed_region,
                    fixed_region_already_masked=False,
                    detector_mode="components",
                )
                widths = sorted(max(1, int(value.width)) for value in columns)
                heights = sorted(max(1, int(value.height)) for value in columns)
                median_width = widths[len(widths) // 2] if widths else 1
                median_height = heights[len(heights) // 2] if heights else 1
                for index, column in enumerate(columns, start=1):
                    bbox = [column.left / width, column.top / height, column.width / width, column.height / height]
                    width_ratio = column.width / max(1.0, median_width)
                    height_ratio = column.height / max(1.0, median_height)
                    ruby_probability = max(0.0, min(1.0, (0.72 - width_ratio) * 1.7 + (0.62 - height_ratio) * 0.8))
                    body_probability = max(0.0, min(1.0, 0.54 + min(1.0, height_ratio) * 0.30 + min(1.0, width_ratio) * 0.16 - ruby_probability * 0.45))
                    detected.append({
                        "slot_id": f"page-{page:04d}-slot-{index:03d}",
                        "bbox": [round(value, 7) for value in bbox],
                        "ink_density": _image_ink_density(image, bbox),
                        "ruby_probability": round(ruby_probability, 6),
                        "body_column_probability": round(body_probability, 6),
                        "estimated_chars": int(getattr(column, "estimated_chars", 0) or 0),
                        "independent_support": 1,
                        "detector_source": "component_geometry_detector",
                        "detector_version": column_detector_version("components"),
                    })
            except Exception as exc:
                detector_error = f"{type(exc).__name__}: {exc}"
                detector_errors.append({"page": page, "error": detector_error})
            finally:
                if image is not None:
                    try:
                        image.close()
                    except Exception:
                        pass
        elif source_path.is_file():
            try:
                from PIL import Image
                with Image.open(source_path) as probe:
                    width, height = probe.size
            except Exception:
                width = height = 0

        production = list(mapped_by_page.get(page, {}).values())
        for value in production:
            value["bbox"] = _normalise_bbox(value.get("bbox"), width, height)
        unmatched_production = set(range(len(production)))
        matched_densities: list[float] = []
        for slot in detected:
            best_index = None
            best_score = 0.0
            for index in list(unmatched_production):
                body_bbox = production[index].get("bbox")
                if not body_bbox:
                    continue
                score = _bbox_match_score(slot["bbox"], body_bbox)
                if score > best_score:
                    best_score = score
                    best_index = index
            if best_index is not None and best_score >= 0.56:
                body = production[best_index]
                slot["mapped_column_id"] = body["column_id"]
                slot["mapped_item_ids"] = copy.deepcopy(body["item_ids"])
                slot["mapping_score"] = round(best_score, 6)
                slot["independent_support"] = max(int(slot.get("independent_support", 1) or 1), int(body.get("independent_support", 1) or 1))
                matched_densities.append(float(slot.get("ink_density", 0.0) or 0.0))
                unmatched_production.discard(best_index)
            else:
                slot["mapped_column_id"] = None
                slot["mapped_item_ids"] = []
                slot["mapping_score"] = round(best_score, 6)

        # Audit-only production columns have stable IDs but may not carry a
        # bbox.  Pair them to body-like re-detected slots by vertical reading
        # order.  This restores columns that vanished during sentence
        # alignment without allowing Ruby/noise slots to become obligations.
        bboxless_indices = [
            index for index in unmatched_production
            if not _normalise_bbox(production[index].get("bbox"))
        ]
        slot_candidates = [
            slot for slot in detected
            if not slot.get("mapped_column_id")
            and float(slot.get("body_column_probability", 0.0) or 0.0) >= 0.76
            and float(slot.get("ruby_probability", 1.0) or 1.0) <= 0.45
        ]
        if bboxless_indices and slot_candidates:
            ordered_production = sorted(bboxless_indices, key=lambda index: _column_order_key(production[index].get("column_id", "")))
            # Vertical Japanese c001 is normally the rightmost column.
            ordered_slots = sorted(
                slot_candidates,
                key=lambda value: (_normalise_bbox(value.get("bbox")) or [0, 0, 0, 0])[0],
                reverse=True,
            )
            if len(ordered_slots) > len(ordered_production):
                ordered_slots = sorted(
                    ordered_slots,
                    key=lambda value: (
                        float(value.get("body_column_probability", 0.0) or 0.0)
                        - float(value.get("ruby_probability", 0.0) or 0.0),
                        float(value.get("ink_density", 0.0) or 0.0),
                    ),
                    reverse=True,
                )[:len(ordered_production)]
                ordered_slots.sort(key=lambda value: (_normalise_bbox(value.get("bbox")) or [0, 0, 0, 0])[0], reverse=True)
            for production_index, slot in zip(ordered_production, ordered_slots):
                body = production[production_index]
                slot["mapped_column_id"] = body["column_id"]
                slot["mapped_item_ids"] = copy.deepcopy(body.get("item_ids") or [])
                slot["mapping_score"] = 0.60
                slot["mapping_method"] = "runtime_audit_column_order"
                slot["independent_support"] = max(
                    int(slot.get("independent_support", 1) or 1),
                    int(body.get("audit_independent_support", body.get("independent_support", 1)) or 1),
                )
                slot["support_signals"] = sorted(set(slot.get("support_signals") or []) | {
                    "component_geometry_detector", "multi_engine_runtime_column_audit",
                })
                if not body.get("item_ids"):
                    slot["production_column_unmapped_to_item"] = True
                matched_densities.append(float(slot.get("ink_density", 0.0) or 0.0))
                unmatched_production.discard(production_index)

        positive_density = sorted(value for value in matched_densities if value > 0)
        median_density = positive_density[len(positive_density) // 2] if positive_density else 0.04
        calibrated_minimum = max(0.004, min(0.035, median_density * 0.28))
        unmatched_raw = [
            slot for slot in detected
            if not slot.get("mapped_column_id") or slot.get("production_column_unmapped_to_item")
        ]
        _promote_regular_grid_support(unmatched_raw, production)
        unmatched = [
            classify_column_slot(slot, production_columns=production, calibrated_minimum=calibrated_minimum)
            for slot in unmatched_raw
        ]
        for slot in unmatched:
            slot["adjacent_item_ids"] = _adjacent_item_ids(slot.get("bbox") or [], production)
        fatal_slots = [copy.deepcopy(slot) for slot in unmatched if slot.get("classification") == "fatal_body_column_gap"]
        review_slots = [copy.deepcopy(slot) for slot in unmatched if slot.get("classification") == "suspected_gap"]
        ruby_slots = [copy.deepcopy(slot) for slot in unmatched if slot.get("classification") == "ruby_side_column"]
        suspicious_ruby_slots = [copy.deepcopy(slot) for slot in ruby_slots if slot.get("requires_visual_review")]
        noise_slots = [copy.deepcopy(slot) for slot in unmatched if slot.get("classification") == "ruby_or_noise"]
        missing_production = [copy.deepcopy(production[index]) for index in sorted(unmatched_production)]
        production_without_items = [copy.deepcopy(value) for value in production if not (value.get("item_ids") or [])]

        coverage_verifiable = bool((detected or production) and not skip_non_body_page)
        if skip_non_body_page or (not source_path.is_file() and not detected and not production):
            coverage_verifiable = False
        if fatal_slots:
            status = "fatal_body_column_gap"
        elif review_slots or suspicious_ruby_slots:
            status = "suspected_gap"
        elif production_without_items:
            # Runtime audits prove that a physical column existed, but without
            # matching geometry/text it cannot be declared fatal.
            status = "unverifiable"
        elif missing_production:
            status = "detector_disagreement"
        elif not coverage_verifiable:
            status = "unverifiable"
        else:
            status = "verified_complete"
        # Production columns with stable items remain covered even when the
        # secondary detector cannot reproduce their exact height.
        coverage_complete = not fatal_slots and all(bool(value.get("item_ids")) for value in production)
        page_review_reasons = []
        if status in {"suspected_gap", "detector_disagreement", "unverifiable"}:
            page_review_reasons.append(status)
        unlocked = sorted({
            str(item_id)
            for slot in fatal_slots
            for item_id in (slot.get("adjacent_item_ids") or [])
            if str(item_id)
        })
        geometry_payload = [
            {"column_id": value.get("column_id"), "bbox": value.get("bbox"), "item_ids": value.get("item_ids")}
            for value in production
        ]
        record = {
            "page": page,
            "status": status,
            "source_page_filename": source_path.name if source_path else "",
            "page_type": page_type or None,
            "skipped_non_body_page": skip_non_body_page,
            "source_page_path_omitted": True,
            "detector_source": detector_source,
            "detector_error": detector_error or None,
            "coverage_verifiable": coverage_verifiable,
            "production_column_count": len(production),
            "mapped_item_count": len({item_id for value in production for item_id in value.get("item_ids") or []}),
            "detected_body_column_count": len(detected),
            "mapped_column_count": len(production) - len(unmatched_production),
            "mapped_column_ids": [slot.get("mapped_column_id") for slot in detected if slot.get("mapped_column_id")],
            "detected_body_slots": detected,
            "fatal_slots": fatal_slots,
            "review_slots": review_slots,
            "ruby_side_columns": ruby_slots,
            "suspicious_ruby_side_columns": suspicious_ruby_slots,
            "noise_slots": noise_slots,
            # Compatibility aliases retained for existing validators.
            "unmapped_body_slots": fatal_slots,
            "suspicious_unmapped_slots": review_slots + ruby_slots,
            "mapped_columns_missing_from_redetection": missing_production,
            "production_columns_without_stable_items": production_without_items,
            "fatal_unmapped_body_slot_count": len(fatal_slots),
            "review_issue_count": len(review_slots) + len(suspicious_ruby_slots) + len(missing_production) + len(production_without_items),
            "coverage_complete": coverage_complete,
            "page_review_required": status in {"suspected_gap", "detector_disagreement", "unverifiable"},
            "page_review_reasons": page_review_reasons,
            "text_items_unlocked": unlocked,
            "related_item_ids": [str(value.get("item_id", "") or "") for value in stable_by_page.get(page, [])],
            "geometry_sha256": sha256_bytes(json.dumps(geometry_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")),
            "calibrated_ink_density_minimum": round(calibrated_minimum, 6),
        }
        fatal_count += len(fatal_slots)
        review_count += record["review_issue_count"]
        status_counts[status] += 1
        page_records.append(_compact_ledger_page(record) if compact else record)

    return {
        "schema": "novel_formatter.ai_publication_page_column_ledger.v4",
        "detector_mode": "production_authority_plus_secondary_components_no_ocr_no_review_projection",
        "authority_order": ["production_ocr_columns", "secondary_geometry_redetection"],
        "compact": bool(compact),
        "page_count": len(page_records),
        "pages": page_records,
        "summary": {
            "coverage_complete": all(bool(value.get("coverage_complete", False)) for value in page_records),
            "fatal_unmapped_body_slot_count": fatal_count,
            "review_issue_count": review_count,
            "unverifiable_page_count": int(status_counts.get("unverifiable", 0)),
            "detector_disagreement_page_count": int(status_counts.get("detector_disagreement", 0)),
            "suspected_gap_page_count": int(status_counts.get("suspected_gap", 0)),
            "verified_complete_page_count": int(status_counts.get("verified_complete", 0)),
            "fatal_page_count": int(status_counts.get("fatal_body_column_gap", 0)),
            "detector_error_count": len(detector_errors),
            "status_counts": dict(status_counts),
        },
        "detector_errors": detector_errors,
    }

def _bbox_union(geometries: Sequence[dict]) -> list[float] | None:
    boxes = []
    for geometry in geometries:
        bbox = geometry.get("bbox") if isinstance(geometry, dict) else None
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                x, y, w, h = map(float, bbox)
            except (TypeError, ValueError):
                continue
            boxes.append((x, y, x + w, y + h))
    if not boxes:
        return None
    x1 = min(value[0] for value in boxes)
    y1 = min(value[1] for value in boxes)
    x2 = max(value[2] for value in boxes)
    y2 = max(value[3] for value in boxes)
    return [max(0.0, x1), max(0.0, y1), min(1.0, x2) - max(0.0, x1), min(1.0, y2) - max(0.0, y1)]


def _context_bbox_with_neighbors(source: dict, target_bbox: Sequence[float] | None) -> list[float] | None:
    if not target_bbox:
        return None
    geometries = source.get("column_geometry") if isinstance(source.get("column_geometry"), list) else []
    boxes = []
    for geometry in geometries:
        if not isinstance(geometry, dict):
            continue
        bbox = _normalise_bbox(geometry.get("bbox"))
        if bbox:
            boxes.append(bbox)
    if not boxes:
        return list(target_bbox)
    tx, ty, tw, th = map(float, target_bbox)
    center = tx + tw / 2
    boxes.sort(key=lambda value: value[0] + value[2] / 2)
    nearest_index = min(range(len(boxes)), key=lambda index: abs((boxes[index][0] + boxes[index][2] / 2) - center))
    selected = boxes[max(0, nearest_index - 1):min(len(boxes), nearest_index + 2)]
    x1 = min(value[0] for value in selected)
    y1 = min(value[1] for value in selected)
    x2 = max(value[0] + value[2] for value in selected)
    y2 = max(value[1] + value[3] for value in selected)
    return [x1, y1, x2 - x1, y2 - y1]


def _export_visual_evidence_forensic(
    primary_doc: UnifiedDocument,
    package: dict,
    repair_map: dict,
    stable_records: Sequence[dict],
    global_anomalies: dict,
    folder: Path,
    page_column_ledger: dict | None = None,
) -> dict:
    root = folder / "visual_evidence"
    categories = ["item_crops", "boundary_crops", "status_blocks", "chapter_titles", "page_context", "unmapped_columns"]
    for category in categories:
        (root / category).mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except Exception:
        Image = None

    stable_by_id = {str(value.get("item_id", "") or ""): value for value in stable_records}
    source_by_id = {str(value.get("item_id", "") or ""): value for value in repair_map.get("items") or []}
    full_by_id = {
        str(value.get("row_id") or value.get("item_id") or ""): value
        for value in package.get("editable_items") or [] if isinstance(value, dict)
    }
    reasons: dict[str, set[str]] = defaultdict(set)
    by_page: dict[int, list[dict]] = defaultdict(list)
    for record in stable_records:
        item_id = str(record.get("item_id", "") or "")
        page = int(record.get("page", 0) or 0)
        by_page[page].append(record)
        if record.get("edit_policy") == "review_required":
            reasons[item_id].add("review_required")
        if record.get("layout_type") == "status_table":
            reasons[item_id].add("status_block")
            groups = record.get("line_groups") or []
            if groups and len(str(groups[-1].get("value", "") or "")) > 60:
                reasons[item_id].add("status_last_value_overlong")
        if "chapter_type_repair" in (record.get("structure_reasons") or []):
            reasons[item_id].add("chapter_title")
        if record.get("image_anchor_before") or record.get("image_anchor_after"):
            reasons[item_id].add("image_boundary")
        flags = set(str(value) for value in (record.get("risk_flags") or []))
        if flags & {"abnormal_latin_run", "control_character", "placeholder_square", "replacement_character"}:
            reasons[item_id].add("abnormal_character_sequence")
        source = full_by_id.get(item_id) or {}
        proposed = str(record.get("proposed_text", "") or "")
        char_fused = str(source.get("character_fused_text", "") or "")
        candidates = model_candidate_texts(source)
        longest = max((len(str(value or "")) for value in candidates), default=0)
        if char_fused and len(char_fused) < max(1, int(len(proposed) * 0.92)):
            reasons[item_id].add("character_fused_shorter_than_current_edited_text")
        if longest and len(proposed) < max(1, int(longest * 0.86)):
            reasons[item_id].add("proposed_shorter_than_longest_model_candidate")

    for page, records_on_page in by_page.items():
        lengths = sorted(len(str(value.get("proposed_text", "") or "")) for value in records_on_page if str(value.get("proposed_text", "") or ""))
        if lengths:
            median = lengths[len(lengths) // 2]
            for record in records_on_page:
                item_id = str(record.get("item_id", "") or "")
                length = len(str(record.get("proposed_text", "") or ""))
                if median >= 20 and 0 < length < median * 0.42:
                    reasons[item_id].add("item_text_significantly_shorter_than_page_median")

    for key in ("duplicate_clusters", "moved_span_clusters", "unsupported_insertions", "orphan_source_spans", "cross_item_order_conflicts"):
        for anomaly in global_anomalies.get(key) or []:
            for field in ("item_id", "source_item_id", "destination_item_id", "previous_item_id", "current_item_id"):
                if anomaly.get(field):
                    reasons[str(anomaly[field])].add(key)
            for occurrence in anomaly.get("occurrences") or []:
                if occurrence.get("item_id"):
                    reasons[str(occurrence["item_id"])].add(key)

    ledger_pages = {int(value.get("page", 0) or 0): value for value in (page_column_ledger or {}).get("pages") or []}
    for page, ledger in ledger_pages.items():
        if ledger.get("unmapped_body_slots"):
            for record in by_page.get(page, []):
                reasons[str(record.get("item_id", "") or "")].add("page_column_ledger_incomplete")
        if ledger.get("suspicious_unmapped_slots"):
            for record in by_page.get(page, []):
                reasons[str(record.get("item_id", "") or "")].add("suspicious_ruby_sized_side_column")

    page_sources = _page_sources(package)
    records: list[dict] = []
    page_hash_cache: dict[str, str] = {}

    def export_record(*, item_id: str, item_reasons: set[str], bbox: list[float] | None,
                      context_bbox: list[float] | None, page: int, category: str,
                      related_item_ids: list[str]) -> None:
        page_source = page_sources.get(page) or {}
        page_type = str(page_source.get("page_type", "") or "").strip().lower()
        non_body_page_types = {
            "cover", "title", "title_page", "frontispiece", "toc", "table_of_contents",
            "illustration", "color_illustration", "monochrome_illustration", "image",
            "afterword", "postscript", "colophon", "copyright", "blank", "advertisement",
            "map", "character_profile",
        }
        skip_non_body_page = page_type in non_body_page_types
        source_path = Path(str(page_source.get("image_path", "") or "")).expanduser()
        safe_id = re.sub(r"[^A-Za-z0-9_-]+", "_", item_id)[:80] or sha256_text(item_id)[:16]
        crop_relative = f"visual_evidence/{category}/{safe_id}.png"
        context_relative = f"visual_evidence/page_context/{safe_id}.png"
        record = {
            "crop_path": crop_relative,
            "page_context_path": context_relative,
            "source_page": page,
            "source_filename": source_path.name if source_path else "",
            "source_path_omitted": True,
            "source_page_sha256": "",
            "source_bbox": bbox,
            "context_bbox": context_bbox,
            "padding_ratio": 0.06,
            "context_padding_ratio": 0.08,
            "rotation": 0,
            "crop_sha256": "",
            "page_context_sha256": "",
            "related_item_ids": related_item_ids,
            "reason": sorted(item_reasons),
            "export_status": "unavailable",
        }
        if Image is not None and source_path.is_file() and bbox and float(bbox[2]) > 0 and float(bbox[3]) > 0:
            try:
                raw = source_path.read_bytes()
                page_digest = page_hash_cache.setdefault(str(source_path), sha256_bytes(raw))
                with Image.open(source_path) as image:
                    image.load()
                    width, height = image.size
                    def crop_box(source_bbox: Sequence[float], padding: float):
                        x, y, w, h = map(float, source_bbox)
                        normalized = max(abs(x), abs(y), abs(w), abs(h)) <= 1.5
                        if normalized:
                            x, y, w, h = x * width, y * height, w * width, h * height
                        px = w * padding
                        py = h * padding
                        left = max(0, math.floor(x - px))
                        top = max(0, math.floor(y - py))
                        right = min(width, math.ceil(x + w + px))
                        bottom = min(height, math.ceil(y + h + py))
                        return image.crop((left, top, right, bottom))
                    crop = crop_box(bbox, 0.06)
                    context = crop_box(context_bbox or bbox, 0.08)
                    crop_path = folder / crop_relative
                    context_path = folder / context_relative
                    crop.save(crop_path, format="PNG", optimize=True)
                    context.save(context_path, format="PNG", optimize=True)
                record.update({
                    "source_page_sha256": page_digest,
                    "crop_sha256": sha256_bytes((folder / crop_relative).read_bytes()),
                    "page_context_sha256": sha256_bytes((folder / context_relative).read_bytes()),
                    "export_status": "exported",
                })
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)

    for item_id, item_reasons in sorted(reasons.items(), key=lambda pair: int(stable_by_id.get(pair[0], {}).get("reading_order", 0) or 0)):
        if not item_id:
            continue
        stable = stable_by_id.get(item_id) or {}
        source = source_by_id.get(item_id) or full_by_id.get(item_id) or {}
        page = int(stable.get("page", source.get("page", 0)) or 0)
        geometry = source.get("column_geometry") if isinstance(source.get("column_geometry"), list) else []
        bbox = _bbox_union(geometry)
        if bbox is None:
            raw_bbox = source.get("bbox") or stable.get("bbox")
            bbox = _normalise_bbox(raw_bbox)
        category = "item_crops"
        if "status_block" in item_reasons:
            category = "status_blocks"
        elif "chapter_title" in item_reasons:
            category = "chapter_titles"
        elif "image_boundary" in item_reasons:
            category = "boundary_crops"
        context_bbox = _context_bbox_with_neighbors(source, bbox)
        export_record(
            item_id=item_id,
            item_reasons=item_reasons,
            bbox=bbox,
            context_bbox=context_bbox,
            page=page,
            category=category,
            related_item_ids=[item_id],
        )

    # Unmapped page slots have no stable item by definition; export them as
    # first-class evidence so a shared bad segmentation cannot remain invisible.
    for page, ledger in sorted(ledger_pages.items()):
        related = [str(value) for value in (ledger.get("related_item_ids") or []) if str(value)]
        for index, slot in enumerate(ledger.get("unmapped_body_slots") or [], start=1):
            bbox = _normalise_bbox(slot.get("bbox"))
            if not bbox:
                continue
            x, y, w, h = bbox
            context_bbox = [max(0.0, x - w * 1.5), y, min(1.0 - max(0.0, x - w * 1.5), w * 4.0), h]
            export_record(
                item_id=f"page-{page:04d}-unmapped-{index:03d}",
                item_reasons={"unmapped_body_column", "fatal_page_column_ledger_gap"},
                bbox=bbox,
                context_bbox=context_bbox,
                page=page,
                category="unmapped_columns",
                related_item_ids=related,
            )

    manifest = {
        "schema": "novel_formatter.ai_publication_visual_evidence.v3",
        "selection_policy": "risk, text-length, page-column-ledger and structural triggers; current column plus adjacent-column context; never export all scan pages",
        "record_count": len(records),
        "exported_crop_count": sum(value.get("export_status") == "exported" for value in records),
        "source_page_count_touched": len({value.get("source_page") for value in records if value.get("export_status") == "exported"}),
        "unmapped_column_record_count": sum("unmapped_body_column" in (value.get("reason") or []) for value in records),
        "records": records,
    }
    (root / "visual_evidence_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest



def _union_bbox_values(boxes: Sequence[Sequence[float]]) -> list[float] | None:
    valid = [_normalise_bbox(value) for value in boxes]
    valid = [value for value in valid if value]
    if not valid:
        return None
    x1 = min(value[0] for value in valid)
    y1 = min(value[1] for value in valid)
    x2 = max(value[0] + value[2] for value in valid)
    y2 = max(value[1] + value[3] for value in valid)
    return [x1, y1, x2 - x1, y2 - y1]


def cluster_visual_regions(regions: Sequence[dict]) -> list[dict]:
    """Merge adjacent problem boxes on one page without losing item/reason links."""
    clusters: list[dict] = []
    for raw in sorted(regions, key=lambda value: (_normalise_bbox(value.get("bbox")) or [9, 9, 0, 0])[0]):
        bbox = _normalise_bbox(raw.get("bbox"))
        if not bbox:
            continue
        merged = False
        for cluster in clusters:
            cb = cluster["bbox"]
            x_gap = _bbox_x_gap(bbox, cb)
            vertical = _bbox_vertical_overlap(bbox, cb)
            close = x_gap <= max(bbox[2], cb[2]) * 0.70
            same_band = vertical >= 0.30 or abs((bbox[1] + bbox[3] / 2) - (cb[1] + cb[3] / 2)) <= max(bbox[3], cb[3]) * 0.30
            if close and same_band:
                cluster["bbox"] = _union_bbox_values([cluster["bbox"], bbox]) or cluster["bbox"]
                cluster["reasons"] = sorted(set(cluster["reasons"]) | set(raw.get("reasons") or []))
                cluster["related_item_ids"] = sorted(set(cluster["related_item_ids"]) | set(raw.get("related_item_ids") or []))
                cluster["slot_ids"] = sorted(set(cluster.get("slot_ids") or []) | set(raw.get("slot_ids") or []))
                merged = True
                break
        if not merged:
            clusters.append({
                "bbox": bbox,
                "reasons": sorted(set(raw.get("reasons") or [])),
                "related_item_ids": sorted(set(raw.get("related_item_ids") or [])),
                "slot_ids": sorted(set(raw.get("slot_ids") or [])),
            })
    return clusters


def write_deduplicated_image(image: Any, requested_path: Path, cache: dict[str, str], root: Path) -> tuple[str, str, bool]:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True, compress_level=9)
    data = buffer.getvalue()
    digest = sha256_bytes(data)
    existing = cache.get(digest)
    if existing:
        return existing, digest, True
    requested_path.parent.mkdir(parents=True, exist_ok=True)
    requested_path.write_bytes(data)
    relative = requested_path.relative_to(root).as_posix()
    cache[digest] = relative
    return relative, digest, False


def _visual_bbox_for_item(source: dict, stable: dict) -> list[float] | None:
    geometry = source.get("column_geometry") if isinstance(source.get("column_geometry"), list) else []
    bbox = _bbox_union(geometry)
    if bbox is None:
        bbox = _normalise_bbox(source.get("bbox") or stable.get("bbox"))
    return bbox


def _compact_visual_reasons(record: dict, source: dict) -> set[str]:
    reasons: set[str] = set()
    flags = set(str(value) for value in (record.get("risk_flags") or []))
    unlock = set(str(value) for value in (record.get("unlock_reasons") or []))
    high_value = {
        "numeric_disagreement", "abnormal_latin_run", "control_character", "placeholder_square",
        "replacement_character", "chapter_title_glued_to_body", "status_narrative_glued",
        "possible_missing_column", "possible_column_order_error", "publication_reference_conflict",
    }
    reasons.update(flags.intersection(high_value))
    reasons.update(unlock.intersection(high_value))
    if record.get("layout_type") == "status_table":
        reasons.add("status_block")
    if "chapter_type_repair" in (record.get("structure_reasons") or []):
        reasons.add("chapter_title")
    if record.get("image_anchor_before") or record.get("image_anchor_after"):
        reasons.add("image_boundary")
    proposed = str(record.get("proposed_text", "") or "")
    char_fused = str(source.get("character_fused_text", "") or "")
    if char_fused and len(char_fused) < max(1, int(len(proposed) * 0.92)):
        reasons.add("character_fused_shorter_than_current_edited_text")
    # A failed OCR backend that returned only placeholders is not an
    # independent textual opinion.  Counting it as a third disagreement would
    # turn a two-engine textual review into a page image for almost every row.
    # Keep placeholder conflicts visible through the dedicated risk flag, but
    # exclude unusable candidates from consensus/length visual triggers.
    candidates = []
    raw_candidates = source.get("candidates") if isinstance(source.get("candidates"), list) else []
    for candidate in raw_candidates:
        if isinstance(candidate, dict):
            text = str(candidate.get("text", "") or "")
            confidence = candidate.get("confidence")
        else:
            text = str(candidate or "")
            confidence = None
        compact = re.sub(r"[\s□■�]+", "", text)
        if not compact:
            continue
        try:
            if confidence is not None and float(confidence) <= 0.0:
                continue
        except (TypeError, ValueError, OverflowError):
            pass
        candidates.append(text)
    if not raw_candidates:
        candidates = [
            value for value in model_candidate_texts(source)
            if re.sub(r"[\s□■�]+", "", str(value or ""))
        ]
    normalized = {normalise_text(value) for value in candidates if normalise_text(value)}
    if len(normalized) >= 3:
        reasons.add("all_models_disagree")
    longest = max((len(str(value or "")) for value in candidates), default=0)
    if longest and len(proposed) < max(1, int(longest * 0.86)):
        reasons.add("proposed_shorter_than_longest_model_candidate")
    if any(marker in proposed for marker in ("□", "■", "�")):
        reasons.add("placeholder_or_replacement_character")
    return reasons


def _export_visual_evidence_compact(
    primary_doc: UnifiedDocument,
    package: dict,
    repair_map: dict,
    stable_records: Sequence[dict],
    global_anomalies: dict,
    folder: Path,
    page_column_ledger: dict | None = None,
) -> dict:
    del primary_doc
    root = folder / "visual_evidence"
    pages_root = root / "pages"
    pages_root.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image, ImageDraw, ImageOps
    except Exception:
        Image = ImageDraw = ImageOps = None

    stable_by_id = {str(value.get("item_id", "") or ""): value for value in stable_records}
    source_by_id = {str(value.get("item_id", "") or ""): value for value in repair_map.get("items") or []}
    full_by_id = {
        str(value.get("row_id") or value.get("item_id") or ""): value
        for value in package.get("editable_items") or [] if isinstance(value, dict)
    }
    regions_by_page: dict[int, list[dict]] = defaultdict(list)
    omitted_text_only = 0
    for record in stable_records:
        item_id = str(record.get("item_id", "") or "")
        source = full_by_id.get(item_id) or source_by_id.get(item_id) or {}
        reasons = _compact_visual_reasons(record, source)
        if not reasons:
            if record.get("edit_policy") == "review_required":
                omitted_text_only += 1
            continue
        bbox = _visual_bbox_for_item(source, record)
        if bbox:
            regions_by_page[int(record.get("page", source.get("page", 0)) or 0)].append({
                "bbox": bbox,
                "reasons": sorted(reasons),
                "related_item_ids": [item_id],
                "slot_ids": [],
            })

    anomaly_keys = ("duplicate_clusters", "moved_span_clusters", "unsupported_insertions", "orphan_source_spans", "cross_item_order_conflicts")
    for key in anomaly_keys:
        for anomaly in global_anomalies.get(key) or []:
            if anomaly.get("severity") not in {"fatal", "high"}:
                continue
            ids = []
            for field in ("item_id", "source_item_id", "destination_item_id", "previous_item_id", "current_item_id"):
                if anomaly.get(field):
                    ids.append(str(anomaly[field]))
            ids.extend(str(value.get("item_id")) for value in anomaly.get("occurrences") or [] if value.get("item_id"))
            for item_id in dict.fromkeys(ids):
                stable = stable_by_id.get(item_id) or {}
                source = full_by_id.get(item_id) or source_by_id.get(item_id) or {}
                bbox = _visual_bbox_for_item(source, stable)
                if bbox:
                    regions_by_page[int(stable.get("page", source.get("page", 0)) or 0)].append({
                        "bbox": bbox, "reasons": [key], "related_item_ids": [item_id], "slot_ids": [],
                    })

    ledger_pages = {int(value.get("page", 0) or 0): value for value in (page_column_ledger or {}).get("pages") or []}
    for page, ledger in ledger_pages.items():
        for slot in ledger.get("fatal_slots") or ledger.get("unmapped_body_slots") or []:
            bbox = _normalise_bbox(slot.get("bbox"))
            if bbox:
                regions_by_page[page].append({
                    "bbox": bbox,
                    "reasons": ["fatal_body_column_gap"],
                    "related_item_ids": list(slot.get("adjacent_item_ids") or ledger.get("related_item_ids") or []),
                    "slot_ids": [str(slot.get("slot_id", "") or "")],
                })
        for slot in ledger.get("review_slots") or []:
            bbox = _normalise_bbox(slot.get("bbox"))
            if bbox:
                regions_by_page[page].append({
                    "bbox": bbox,
                    "reasons": ["suspected_gap"],
                    "related_item_ids": list(slot.get("adjacent_item_ids") or []),
                    "slot_ids": [str(slot.get("slot_id", "") or "")],
                })
        for slot in ledger.get("suspicious_ruby_side_columns") or []:
            bbox = _normalise_bbox(slot.get("bbox"))
            if bbox:
                regions_by_page[page].append({
                    "bbox": bbox,
                    "reasons": ["suspicious_ruby_side_column"],
                    "related_item_ids": list((slot.get("ruby_pairing") or {}).get("paired_item_ids") or []),
                    "slot_ids": [str(slot.get("slot_id", "") or "")],
                })

    page_sources = _page_sources(package)
    records: list[dict] = []
    image_cache: dict[str, str] = {}
    file_paths: set[str] = set()
    for page, raw_regions in sorted(regions_by_page.items()):
        if page <= 0 or not raw_regions:
            continue
        clusters = cluster_visual_regions(raw_regions)
        source_path = Path(str((page_sources.get(page) or {}).get("image_path", "") or "")).expanduser()
        page_records = []
        if Image is None or not source_path.is_file():
            for index, cluster in enumerate(clusters, start=1):
                records.append({
                    "evidence_id": f"ev-page-{page:04d}-{index:03d}", "page": page,
                    "reason": cluster["reasons"], "source_bbox": cluster["bbox"],
                    "related_item_ids": cluster["related_item_ids"], "export_status": "unavailable",
                })
            continue
        try:
            with Image.open(source_path) as opened:
                original = ImageOps.grayscale(opened)
                width, height = original.size
                overview = original.copy()
                max_edge = max(overview.size)
                scale = min(1.0, 1600.0 / max(1, max_edge))
                if scale < 1.0:
                    overview = overview.resize((max(1, round(width * scale)), max(1, round(height * scale))))
                overview_rgb = overview.convert("RGB")
                draw = ImageDraw.Draw(overview_rgb)
                crop_images = []
                sheet_positions = []
                for index, cluster in enumerate(clusters, start=1):
                    x, y, w, h = map(float, cluster["bbox"])
                    left = max(0, int(x * width))
                    top = max(0, int(y * height))
                    right = min(width, max(left + 1, int((x + w) * width)))
                    bottom = min(height, max(top + 1, int((y + h) * height)))
                    pad_x = max(2, int((right - left) * 0.08))
                    pad_y = max(2, int((bottom - top) * 0.04))
                    box = (max(0, left - pad_x), max(0, top - pad_y), min(width, right + pad_x), min(height, bottom + pad_y))
                    crop_images.append(original.crop(box))
                    scaled_box = tuple(round(value * scale) for value in box)
                    draw.rectangle(scaled_box, outline=(220, 20, 60), width=max(1, round(2 * scale)))
                    label_x = max(0, scaled_box[0])
                    label_y = max(0, scaled_box[1] - 12)
                    draw.text((label_x, label_y), str(index), fill=(0, 0, 0))
                    page_records.append((index, cluster, box))

                label_height = 24
                sheet_width = max((image.width for image in crop_images), default=1) + 20
                sheet_height = sum(image.height + label_height + 10 for image in crop_images) + 10
                sheet = Image.new("L", (sheet_width, max(1, sheet_height)), 255)
                sheet_draw = ImageDraw.Draw(sheet)
                cursor_y = 10
                for (index, cluster, _box), crop in zip(page_records, crop_images):
                    sheet_draw.text((10, cursor_y + 4), f"#{index}  {'; '.join(cluster['reasons'])}", fill=0)
                    paste_y = cursor_y + label_height
                    sheet.paste(crop, (10, paste_y))
                    sheet_positions.append([10, paste_y, crop.width, crop.height])
                    cursor_y = paste_y + crop.height + 10

                overview_path, overview_sha, overview_reused = write_deduplicated_image(
                    overview_rgb, pages_root / f"page_{page:04d}_overview.png", image_cache, folder,
                )
                sheet_path, sheet_sha, sheet_reused = write_deduplicated_image(
                    sheet, pages_root / f"page_{page:04d}_crops.png", image_cache, folder,
                )
                file_paths.update([overview_path, sheet_path])
                source_sha = sha256_bytes(source_path.read_bytes())
                for record_index, ((index, cluster, _box), sheet_bbox) in enumerate(zip(page_records, sheet_positions), start=1):
                    records.append({
                        "evidence_id": f"ev-page-{page:04d}-{record_index:03d}",
                        "page": page,
                        "reason": cluster["reasons"],
                        "source_bbox": cluster["bbox"],
                        "overview_path": overview_path,
                        "overview_sha256": overview_sha,
                        "crop_sheet_path": sheet_path,
                        "crop_sheet_sha256": sheet_sha,
                        "crop_sheet_bbox": sheet_bbox,
                        "related_item_ids": cluster["related_item_ids"],
                        "slot_ids": cluster.get("slot_ids") or [],
                        "source_filename": source_path.name,
                        "source_page_sha256": source_sha,
                        "overview_reused_by_hash": overview_reused,
                        "crop_sheet_reused_by_hash": sheet_reused,
                        "export_status": "exported",
                    })
        except Exception as exc:
            for index, cluster in enumerate(clusters, start=1):
                records.append({
                    "evidence_id": f"ev-page-{page:04d}-{index:03d}", "page": page,
                    "reason": cluster["reasons"], "source_bbox": cluster["bbox"],
                    "related_item_ids": cluster["related_item_ids"], "export_status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                })

    manifest = {
        "schema": "novel_formatter.ai_publication_visual_evidence.v4",
        "package_mode": "compact",
        "selection_policy": "text review and visual review are separate; page overview plus original-pixel crop sheet only for geometry or visually irresolvable risk",
        "record_count": len(records),
        "exported_crop_count": sum(value.get("export_status") == "exported" for value in records),
        "visual_page_count": len({value.get("page") for value in records if value.get("export_status") == "exported"}),
        "visual_file_count": len(file_paths),
        "source_page_count_touched": len({value.get("page") for value in records if value.get("export_status") == "exported"}),
        "fatal_visual_evidence_count": sum("fatal_body_column_gap" in (value.get("reason") or []) for value in records),
        "review_visual_evidence_count": sum("fatal_body_column_gap" not in (value.get("reason") or []) for value in records),
        "text_only_review_items_without_images": omitted_text_only,
        "image_hash_deduplication_enabled": True,
        "records": records,
    }
    (root / "visual_evidence_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return manifest


def export_visual_evidence(
    primary_doc: UnifiedDocument,
    package: dict,
    repair_map: dict,
    stable_records: Sequence[dict],
    global_anomalies: dict,
    folder: Path,
    page_column_ledger: dict | None = None,
    *,
    package_mode: str = "forensic",
) -> dict:
    if str(package_mode or "forensic").lower() == "compact":
        return _export_visual_evidence_compact(
            primary_doc, package, repair_map, stable_records, global_anomalies, folder,
            page_column_ledger=page_column_ledger,
        )
    manifest = _export_visual_evidence_forensic(
        primary_doc, package, repair_map, stable_records, global_anomalies, folder,
        page_column_ledger=page_column_ledger,
    )
    manifest["package_mode"] = "forensic"
    return manifest

def _reference_path(package: dict) -> tuple[Path | None, dict]:
    value = package.get("publication_reference") or package.get("reference_epub")
    metadata: dict = {}
    if isinstance(value, dict):
        metadata = copy.deepcopy(value)
        raw = value.get("epub_path") or value.get("path") or value.get("source_path")
    else:
        raw = value
    if not raw:
        return None, metadata
    path = Path(str(raw)).expanduser()
    return (path if path.is_file() else None), metadata


def _safe_member(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"unsafe EPUB member: {name}")


def _find_opf(archive: zipfile.ZipFile) -> str:
    root = ET.fromstring(archive.read("META-INF/container.xml"))
    for element in root.iter():
        if local_name(element.tag) == "rootfile" and element.attrib.get("full-path"):
            return element.attrib["full-path"]
    raise ValueError("container.xml has no rootfile")


def _element_plain_text(element: ET.Element) -> str:
    parts: list[str] = []
    def walk(node: ET.Element):
        if local_name(node.tag) == "rt":
            return
        if node.text:
            parts.append(node.text)
        for child in list(node):
            walk(child)
            if child.tail:
                parts.append(child.tail)
    walk(element)
    return re.sub(r"[\t\r ]+", " ", "".join(parts)).strip()


def _ruby_groups(element: ET.Element) -> list[dict]:
    result = []
    for ruby in element.iter():
        if local_name(ruby.tag) != "ruby":
            continue
        readings = ["".join(rt.itertext()).strip() for rt in ruby.iter() if local_name(rt.tag) == "rt"]
        base_parts = []
        if ruby.text:
            base_parts.append(ruby.text)
        for child in list(ruby):
            if local_name(child.tag) != "rt":
                base_parts.extend(child.itertext())
        base = "".join(base_parts).strip()
        reading = "".join(readings).strip()
        if base and reading:
            result.append({"base": base, "reading": reading})
    return result


def _element_inline_tokens(element: ET.Element) -> list[dict]:
    tokens: list[dict] = []

    def add_text(value: str | None) -> None:
        text = str(value or "")
        if not text:
            return
        if tokens and tokens[-1].get("type") == "text":
            tokens[-1]["value"] = str(tokens[-1].get("value", "")) + text
        else:
            tokens.append({"type": "text", "value": text})

    def walk(node: ET.Element) -> None:
        add_text(node.text)
        for child in list(node):
            tag = local_name(child.tag)
            if tag == "rt":
                pass
            elif tag == "ruby":
                groups = _ruby_groups(child)
                if groups:
                    group = groups[0]
                    tokens.append({"type": "ruby", "base": group["base"], "reading": group["reading"]})
                else:
                    add_text(_element_plain_text(child))
            else:
                walk(child)
            add_text(child.tail)

    walk(element)
    return tokens


def _reference_language_decision(text: str, element: ET.Element, html_lang: str, ocr_texts: Sequence[str]) -> tuple[bool, str, float]:
    lang = str(element.attrib.get("{http://www.w3.org/XML/1998/namespace}lang") or element.attrib.get("lang") or html_lang or "").lower()
    class_id = (str(element.attrib.get("class", "")) + " " + str(element.attrib.get("id", ""))).lower()
    if lang.startswith("zh") or re.search(r"(?:^|[-_ ])(?:zh|cn|chinese|translation|translated)(?:$|[-_ ])", class_id):
        return False, "explicit_chinese", 1.0
    if lang.startswith("ja") or re.search(r"(?:^|[-_ ])(?:ja|jp|japanese|original)(?:$|[-_ ])", class_id):
        return True, "explicit_japanese", 1.0
    kana = len(KANA_RE.findall(text))
    han = len(HAN_RE.findall(text))
    if kana:
        return True, "contains_kana", min(1.0, 0.7 + kana / max(1, len(text)))
    if not text.strip():
        return False, "empty", 1.0
    simplified = sum(ch in SIMPLIFIED_HINTS for ch in text)
    if simplified >= 2 and kana == 0:
        return False, "simplified_chinese_signals", min(1.0, 0.7 + simplified / max(1, len(text)))
    norm = normalise_text(text)
    best = 0.0
    if norm and ocr_texts:
        grams = {norm[i:i+3] for i in range(max(0, len(norm)-2))}
        for ocr in ocr_texts:
            other = normalise_text(ocr)
            if not other:
                continue
            if norm in other or other in norm:
                best = max(best, min(len(norm), len(other)) / max(len(norm), len(other)))
            elif grams:
                other_grams = {other[i:i+3] for i in range(max(0, len(other)-2))}
                if other_grams:
                    best = max(best, len(grams & other_grams) / len(grams | other_grams))
    if best >= 0.28:
        return True, "ocr_similarity", best
    if han and len(text) <= 30 and local_name(element.tag) in {"h1", "h2", "h3", "h4"}:
        return True, "short_heading_default_japanese", 0.55
    return False, "ambiguous_han_without_ocr_support", 0.5


def export_reference_evidence(package: dict, stable_records: Sequence[dict], folder: Path) -> dict:
    """Export an explicitly selected publication EPUB as optional evidence.

    The local absolute path is never written to the package.  Ruby is preserved
    only as evidence-backed inline tokens; OCR target text remains plain base
    text and is never replaced by reference text during export.
    """
    root = folder / "reference"
    root.mkdir(parents=True, exist_ok=True)
    path, metadata = _reference_path(package)
    identity = {
        "schema": "novel_formatter.ai_publication_reference_identity.v3",
        "available": bool(path),
        "source_path": None,
        "source_path_omitted": True,
        "source_filename": path.name if path else "",
        "authority": str(metadata.get("authority", "publication_reference" if path else "none") or "none"),
        "contains_bilingual_content": False,
        "chinese_exclusion_policy": ["lang_and_xml_lang", "class_and_id", "kana_ratio", "ocr_similarity", "simplified_chinese_signals"],
        "ruby_policy": "evidence_backed_optional; OCR text stays base-only; never infer readings",
    }
    paragraphs: list[dict] = []
    ruby_records: list[dict] = []
    structure = {"schema": "novel_formatter.ai_publication_reference_structure.v3", "available": False, "spine": [], "nav": []}
    images: list[dict] = []
    css_semantics: list[dict] = []
    alignment: list[dict] = []
    excluded_count = 0
    if path:
        try:
            with zipfile.ZipFile(path, "r") as archive:
                for name in archive.namelist():
                    _safe_member(name)
                if archive.testzip() is not None:
                    raise ValueError("reference EPUB CRC failure")
                opf_path = _find_opf(archive)
                opf_root = ET.fromstring(archive.read(opf_path))
                base_dir = str(PurePosixPath(opf_path).parent)
                manifest: dict[str, dict] = {}
                spine_ids: list[str] = []
                for element in opf_root.iter():
                    tag = local_name(element.tag)
                    if tag == "item":
                        manifest[str(element.attrib.get("id", ""))] = dict(element.attrib)
                    elif tag == "itemref" and element.attrib.get("idref"):
                        spine_ids.append(element.attrib["idref"])
                for nav_item in manifest.values():
                    properties = str(nav_item.get("properties", "") or "")
                    href = str(nav_item.get("href", "") or "")
                    if "nav" not in properties.split() or not href:
                        continue
                    nav_path = str(PurePosixPath(base_dir, href)) if base_dir not in {"", "."} else href
                    nav_path = str(PurePosixPath(nav_path))
                    if nav_path not in archive.namelist():
                        continue
                    try:
                        nav_root = ET.fromstring(archive.read(nav_path))
                    except ET.ParseError:
                        continue
                    for anchor in nav_root.iter():
                        if local_name(anchor.tag) != "a" or not anchor.attrib.get("href"):
                            continue
                        label = re.sub(r"\s+", " ", "".join(anchor.itertext())).strip()
                        if label:
                            structure["nav"].append({"label": label, "href": anchor.attrib.get("href"), "nav_path": nav_path})
                if not structure["nav"]:
                    for ncx_item in manifest.values():
                        if str(ncx_item.get("media-type", "") or "") != "application/x-dtbncx+xml":
                            continue
                        href = str(ncx_item.get("href", "") or "")
                        ncx_path = str(PurePosixPath(base_dir, href)) if base_dir not in {"", "."} else href
                        ncx_path = str(PurePosixPath(ncx_path))
                        if ncx_path not in archive.namelist():
                            continue
                        try:
                            ncx_root = ET.fromstring(archive.read(ncx_path))
                        except ET.ParseError:
                            continue
                        for point in ncx_root.iter():
                            if local_name(point.tag) != "navPoint":
                                continue
                            label = ""
                            href_value = ""
                            for descendant in point.iter():
                                tag = local_name(descendant.tag)
                                if tag == "text" and not label:
                                    label = re.sub(r"\s+", " ", "".join(descendant.itertext())).strip()
                                elif tag == "content" and not href_value:
                                    href_value = str(descendant.attrib.get("src", "") or "")
                            if label and href_value:
                                structure["nav"].append({"label": label, "href": href_value, "nav_path": ncx_path, "source": "ncx"})
                ocr_texts = [str(value.get("proposed_text", "") or "") for value in stable_records]
                structure["available"] = True
                for spine_index, idref in enumerate(spine_ids):
                    item = manifest.get(idref) or {}
                    href = str(item.get("href", "") or "")
                    if not href:
                        continue
                    archive_path = str(PurePosixPath(base_dir, href)) if base_dir not in {"", "."} else href
                    archive_path = str(PurePosixPath(archive_path))
                    if archive_path not in archive.namelist():
                        continue
                    structure["spine"].append({"spine_index": spine_index, "idref": idref, "path": archive_path})
                    try:
                        html = ET.fromstring(archive.read(archive_path))
                    except ET.ParseError:
                        continue
                    html_lang = str(html.attrib.get("{http://www.w3.org/XML/1998/namespace}lang") or html.attrib.get("lang") or "")
                    element_index = 0
                    for element in html.iter():
                        tag = local_name(element.tag)
                        if tag == "img":
                            images.append({
                                "reference_xhtml": archive_path,
                                "spine_index": spine_index,
                                "src": str(element.attrib.get("src", "") or ""),
                                "alt": str(element.attrib.get("alt", "") or ""),
                                "element_index": element_index,
                            })
                        if tag not in BLOCK_TAGS:
                            continue
                        if any(local_name(child.tag) in BLOCK_TAGS for child in list(element)):
                            continue
                        plain_text = _element_plain_text(element)
                        if not plain_text:
                            continue
                        include, reason, confidence = _reference_language_decision(plain_text, element, html_lang, ocr_texts)
                        selector = f"#{element.attrib['id']}" if element.attrib.get("id") else f"{tag}:nth-source({element_index + 1})"
                        element_index += 1
                        if not include:
                            excluded_count += 1
                            continue
                        record_id = f"ref-{spine_index:04d}-{element_index:05d}"
                        inline_tokens = _element_inline_tokens(element)
                        ruby_groups = _ruby_groups(element)
                        paragraph = {
                            "reference_id": record_id,
                            "plain_text": plain_text,
                            "reference_text": plain_text,
                            "reference_text_sha256": sha256_text(plain_text),
                            "inline_tokens": inline_tokens,
                            "ruby_groups": ruby_groups,
                            "ruby_group_count": len(ruby_groups),
                            "reference_xhtml": archive_path,
                            "reference_selector": selector,
                            "spine_index": spine_index,
                            "element_tag": tag,
                            "language_decision": reason,
                            "language_confidence": round(float(confidence), 4),
                            "contains_chinese_translation": False,
                            "ruby_readings_stripped": False,
                            "ruby_evidence_authority": "publication_reference" if ruby_groups else None,
                        }
                        paragraphs.append(paragraph)
                        for ruby_index, group in enumerate(ruby_groups, start=1):
                            ruby_records.append({
                                "ruby_id": f"{record_id}-ruby-{ruby_index:03d}",
                                "reference_id": record_id,
                                "base": group["base"],
                                "reading": group["reading"],
                                "reference_xhtml": archive_path,
                                "reference_selector": selector,
                                "spine_index": spine_index,
                                "authority": "publication_reference",
                            })
                for item in manifest.values():
                    media = str(item.get("media-type", "") or "")
                    href = str(item.get("href", "") or "")
                    if media == "text/css" and href:
                        archive_path = str(PurePosixPath(base_dir, href)) if base_dir not in {"", "."} else href
                        archive_path = str(PurePosixPath(archive_path))
                        if archive_path in archive.namelist():
                            css = archive.read(archive_path).decode("utf-8", errors="replace")
                            css_semantics.append({
                                "path": archive_path,
                                "sha256": sha256_text(css),
                                "writing_modes": sorted(set(re.findall(r"writing-mode\s*:\s*([^;}{]+)", css, re.I))),
                                "fixed_color_rules": len(re.findall(r"(?:^|[;{])\s*(?:color|background(?:-color)?)\s*:", css, re.I)),
                            })
                identity["sha256"] = sha256_bytes(path.read_bytes())
                identity["japanese_record_count"] = len(paragraphs)
                identity["ruby_group_count"] = len(ruby_records)
                identity["excluded_non_japanese_record_count"] = excluded_count
                identity["contains_bilingual_content"] = excluded_count > 0
        except Exception as exc:
            identity["available"] = False
            identity["error"] = f"{type(exc).__name__}: {exc}"

    ref_norm = [normalise_text(value.get("reference_text", "")) for value in paragraphs]
    previous_ref = -1
    for record in stable_records:
        current_text = str(record.get("proposed_text", "") or "")
        text = normalise_text(current_text)
        if not text or not paragraphs:
            continue
        candidates: set[int] = set()
        grams = {text[i:i+3] for i in range(max(0, len(text)-2))}
        for index, other in enumerate(ref_norm):
            if not other:
                continue
            if text in other or other in text:
                candidates.add(index)
            elif grams and any(gram in other for gram in list(grams)[:40]):
                candidates.add(index)
        if not candidates:
            continue
        def score(index: int) -> float:
            ratio = SequenceMatcher(None, text, ref_norm[index], autojunk=False).ratio()
            monotonic_penalty = 0.0 if index >= previous_ref else min(0.25, (previous_ref - index) * 0.01)
            return ratio - monotonic_penalty
        best_index = max(candidates, key=score)
        confidence = max(0.0, score(best_index))
        if confidence < 0.35:
            continue
        ref = paragraphs[best_index]
        previous_ref = max(previous_ref, best_index)
        exact_plain_text_match = normalise_text(ref["reference_text"]) == text
        alignment.append({
            "item_id": str(record.get("item_id", "") or ""),
            "reference_id": ref["reference_id"],
            "reference_text": ref["reference_text"],
            "alignment_confidence": round(confidence, 6),
            "exact_plain_text_match": exact_plain_text_match,
            "reference_xhtml": ref["reference_xhtml"],
            "reference_selector": ref["reference_selector"],
            "reference_inline_tokens": copy.deepcopy(ref.get("inline_tokens") or []),
            "reference_ruby_groups": copy.deepcopy(ref.get("ruby_groups") or []),
            "authority": identity.get("authority", "publication_reference"),
            "contains_chinese_translation": False,
        })

    structure["image_count"] = len(images)
    structure["japanese_text_record_count"] = len(paragraphs)
    identity["alignment_count"] = len(alignment)
    files = {
        "reference_identity.json": identity,
        "reference_japanese_text.jsonl": paragraphs,
        "reference_alignment.jsonl": alignment,
        "reference_ruby.jsonl": ruby_records,
        "reference_structure.json": structure,
        "reference_image_positions.json": {"schema": "novel_formatter.ai_publication_reference_images.v3", "images": images},
        "reference_css_semantics.json": {"schema": "novel_formatter.ai_publication_reference_css.v3", "stylesheets": css_semantics},
    }
    for name, value in files.items():
        path_out = root / name
        if name.endswith(".jsonl"):
            path_out.write_text("\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in value) + ("\n" if value else ""), encoding="utf-8")
        else:
            path_out.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "available": bool(identity.get("available")),
        "identity": identity,
        "alignment": alignment,
        "ruby": ruby_records,
        "ruby_policy": "evidence_backed_optional; base text must match; no inferred readings",
        "japanese_record_count": len(paragraphs),
        "ruby_group_count": len(ruby_records),
        "excluded_non_japanese_record_count": excluded_count,
    }
