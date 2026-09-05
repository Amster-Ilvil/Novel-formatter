#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Structure-preserving EPUB exchange format for external AI OCR repair.

The exported EPUB is a readable context carrier, while sparse JSON is the
preferred write-back channel.  Every editable row has a stable ID, a baseline
text hash, adjacent context, structured OCR disagreement evidence, and a safe
three-way import path.  External EPUB/JSON data is treated as untrusted input.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import posixpath
import re
import shutil
import time
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Sequence
import xml.etree.ElementTree as ET

from builder.epub_builder import build_epub
from models.document import Block, BlockType, UnifiedDocument
from engine.text_compare import looks_like_chapter_title

AI_REPAIR_SCHEMA = "novel_formatter.ai_repair_epub.v2"
AI_REPAIR_EDITS_SCHEMA = "novel_formatter.ai_repair_edits.v2"
AI_DISAGREEMENT_DECISIONS_SCHEMA = "novel_formatter.ai_disagreement_decisions.v1"
_LEGACY_AI_REPAIR_SCHEMAS = {"novel_formatter.ai_repair_epub.v1", AI_REPAIR_SCHEMA}
_LEGACY_AI_REPAIR_EDIT_SCHEMAS = {"novel_formatter.ai_repair_edits.v1", AI_REPAIR_EDITS_SCHEMA}

# Compatibility files retained for older clients.
MAP_PATH = "META-INF/ai-repair-map.json"
GUIDE_PATH = "META-INF/AI_REPAIR_GUIDE.md"
TEMPLATE_PATH = "META-INF/ai-repair-result-template.json"

# Primary v2 layout.
AI_REPAIR_ROOT = "META-INF/ai-repair"
MANIFEST_PATH = f"{AI_REPAIR_ROOT}/manifest.json"
TASKS_PATH = f"{AI_REPAIR_ROOT}/tasks.jsonl"
RESULT_SCHEMA_PATH = f"{AI_REPAIR_ROOT}/result-schema.json"
V2_GUIDE_PATH = f"{AI_REPAIR_ROOT}/guide.md"
V2_TEMPLATE_PATH = f"{AI_REPAIR_ROOT}/result-template.json"
CHAPTERS_DIR = f"{AI_REPAIR_ROOT}/chapters"

_ALLOWED_MODES = {"light", "standard", "expert", "one_pass"}
_ALLOWED_WORKFLOWS = {"exchange", "publication"}

AI_PUBLICATION_FUSION_SCHEMA = "novel_formatter.ai_publication_fusion.v2"
AI_PUBLICATION_MANIFEST_SCHEMA = "novel_formatter.ai_publication_bundle.v2"
AI_PUBLICATION_ROOT = "META-INF/ai-publication"
AI_PUBLICATION_MANIFEST_PATH = f"{AI_PUBLICATION_ROOT}/manifest.json"
AI_PUBLICATION_FUSION_PATH = f"{AI_PUBLICATION_ROOT}/fusion-evidence.json"
AI_PUBLICATION_GUIDE_PATH = f"{AI_PUBLICATION_ROOT}/MODEL_INSTRUCTIONS.md"
_TEXT_TYPES = {
    BlockType.PARAGRAPH,
    BlockType.DIALOGUE,
    BlockType.CHAPTER,
    BlockType.SECTION,
    BlockType.RUBY,
    BlockType.FOOTNOTE,
    BlockType.TOC_ENTRY,
}

# Untrusted input limits. They are intentionally generous for illustrated novels.
_MAX_ZIP_MEMBERS = 50_000
_MAX_ZIP_TOTAL_UNCOMPRESSED = 2 * 1024 * 1024 * 1024
_MAX_ZIP_SINGLE_UNCOMPRESSED = 256 * 1024 * 1024
_MAX_XML_BYTES = 64 * 1024 * 1024
_MAX_JSON_BYTES = 128 * 1024 * 1024
_MAX_JSON_DEPTH = 80
_MAX_JSON_ARRAY_ITEMS = 500_000
_MAX_DISAGREEMENT_SPANS = 256
_DANGEROUS_SCHEMES = {"javascript", "file", "vbscript"}
_SIMPLIFIED_ONLY_CHARS = set("这发为后里个们从东书车门见气云电国学体无与业开关台万广乐长叶号边达过进还远应当会动产种总线图画听说读写买卖让给问间时点对错压龙马风鸟鱼爱欢亲头脸声宝实难")
_SMALL_KANA = "ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ"
_LARGE_KANA = "あいうえおつやゆよわアイウエオツヤユヨワカケ"
_SMALL_KANA_TABLE = str.maketrans(dict(zip(_SMALL_KANA, _LARGE_KANA)))
_NEGATION_TOKENS = ("ない", "ぬ", "ず", "ません", "じゃない", "ではない", "無", "未", "非")
_DASH_CHARS = "─━―—‐‑‒–ー一"


class AIRepairEpubError(ValueError):
    """The external repair artifact is malformed, stale, or structurally unsafe."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value))


def _text_sha256(text: Any) -> str:
    return _sha256_bytes(str(text or "").encode("utf-8"))


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _explicit_bool(value: Any, *, field: str) -> bool:
    if value is True or value is False:
        return bool(value)
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no", ""}:
            return False
    raise AIRepairEpubError(f"{field} 必须是明确的 true/false。")


def _safe_block_type(value: Any, fallback: BlockType) -> BlockType:
    raw = str(value or "").strip()
    try:
        return BlockType(raw)
    except Exception:
        return fallback if fallback in _TEXT_TYPES else BlockType.PARAGRAPH


def _bbox_list(block: Block | None) -> list[float]:
    bbox = getattr(block, "bbox", None) if block is not None else None
    if bbox is None:
        return []
    return [
        _finite_float(getattr(bbox, "x", 0.0)),
        _finite_float(getattr(bbox, "y", 0.0)),
        _finite_float(getattr(bbox, "w", 0.0)),
        _finite_float(getattr(bbox, "h", 0.0)),
    ]


def _normalise_plain_text(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip(" \t\n")


def _html_id(item_id: str, row_index: int) -> str:
    digest = hashlib.sha1(item_id.encode("utf-8")).hexdigest()[:10]
    return f"row-{row_index:06d}-{digest}"


def _content_format(block_type: str) -> str:
    return "inline_tokens_v1" if str(block_type) == BlockType.RUBY.value else "plain_text_with_newlines_v1"


def _inline_tokens(text: str) -> list[dict]:
    """Convert the project's 漢字|よみ convention into controlled tokens."""
    value = str(text or "")
    tokens: list[dict] = []
    last = 0
    for match in re.finditer(r"([^\s|]+)\|([^\s|]+)", value):
        if match.start() > last:
            tokens.append({"type": "text", "value": value[last:match.start()]})
        tokens.append({"type": "ruby", "base": match.group(1), "reading": match.group(2)})
        last = match.end()
    if last < len(value):
        tokens.append({"type": "text", "value": value[last:]})
    if not tokens:
        tokens.append({"type": "text", "value": value})
    return tokens


def _tokens_to_text(tokens: Any, *, field: str) -> str:
    if not isinstance(tokens, list) or len(tokens) > 10_000:
        raise AIRepairEpubError(f"{field} 必须是受控 token 数组。")
    parts: list[str] = []
    for index, token in enumerate(tokens):
        if not isinstance(token, dict):
            raise AIRepairEpubError(f"{field}[{index}] 不是对象。")
        token_type = str(token.get("type", "") or "")
        if token_type == "text":
            if set(token) - {"type", "value"}:
                raise AIRepairEpubError(f"{field}[{index}] 包含未知字段。")
            parts.append(str(token.get("value", "") or ""))
        elif token_type == "ruby":
            if set(token) - {"type", "base", "reading"}:
                raise AIRepairEpubError(f"{field}[{index}] 包含未知字段。")
            base = str(token.get("base", "") or "")
            reading = str(token.get("reading", "") or "")
            if not base or not reading or "|" in base or "|" in reading:
                raise AIRepairEpubError(f"{field}[{index}] 的 Ruby 不完整。")
            parts.append(f"{base}|{reading}")
        else:
            raise AIRepairEpubError(f"{field}[{index}] 的 type 仅允许 text/ruby。")
    return "".join(parts)


def _repair_block_from_item(item: dict, source: Block | None, *, export_revision: int = 0) -> Block:
    row_index = int(item.get("row_index", 0) or 0)
    item_id = str(item.get("row_id") or item.get("item_id") or "").strip()
    if not item_id:
        raise AIRepairEpubError(f"第 {row_index + 1} 条缺少稳定 row_id。")
    fallback_type = source.type if source is not None else BlockType.PARAGRAPH
    block_type = _safe_block_type(item.get("block_type"), fallback_type)
    block = copy.deepcopy(source) if source is not None else Block(type=block_type)
    block.type = block_type
    block.id = f"ai-repair-{hashlib.sha1(item_id.encode('utf-8')).hexdigest()[:24]}"
    baseline = _normalise_plain_text(item.get("edited_text", item.get("original_fused_text", "")))
    block.text = baseline
    block.ocr_raw = _normalise_plain_text(item.get("original_fused_text", baseline))
    block.page = int(item.get("page", getattr(block, "page", 0)) or 0)
    block.source_format = "ai_repair_epub"
    block.modified_by = (str(getattr(block, "modified_by", "") or "") + ",ai_repair_export").strip(",")
    bbox = _bbox_list(source)
    column_ids = [str(value) for value in (item.get("column_ids") or []) if str(value)]
    delete_intentionally = _explicit_bool(
        item.get("delete_intentionally", False),
        field=f"条目 {item_id} 的 delete_intentionally",
    )
    block.metadata = {
        **(copy.deepcopy(getattr(block, "metadata", {}) or {})),
        "ai_repair_item_id": item_id,
        "ai_repair_html_id": _html_id(item_id, row_index),
        "ai_repair_keep_empty": True,
        "ai_repair_delete_intentionally": delete_intentionally,
        "ai_repair_column_ids": column_ids,
        "ai_repair_bbox": bbox,
        "ai_repair_row_index": row_index,
        "ai_repair_content_format": _content_format(block_type.value),
        "ai_repair_baseline_sha256": _text_sha256(baseline),
        "ai_repair_export_revision": int(export_revision or 0),
        "source_column_ids": column_ids,
        "multi_ocr_column_ids": column_ids,
    }
    return block


def build_repair_document(
    primary_doc: UnifiedDocument,
    package: dict,
    *,
    export_revision: int = 0,
) -> UnifiedDocument:
    """Build a row-addressable document while preserving primary block/image order."""
    if not isinstance(package, dict):
        raise AIRepairEpubError("AI 修复 EPUB 缺少有效的多模型包。")
    items = list(package.get("editable_items") or [])
    if not items:
        raise AIRepairEpubError("多模型包没有可编辑条目。")
    ordered = sorted(items, key=lambda item: int(item.get("row_index", 0) or 0))
    ids = [str(item.get("row_id") or item.get("item_id") or "") for item in ordered]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise AIRepairEpubError("AI 修复条目 ID 缺失或重复。")

    anchored: dict[int, list[dict]] = {}
    insertions: dict[int, list[dict]] = {}
    covered: set[int] = set()
    for item in ordered:
        indices: list[int] = []
        raw_indices = item.get("primary_block_indices") or []
        if isinstance(raw_indices, (list, tuple)):
            for value in raw_indices:
                try:
                    index = int(value)
                except (TypeError, ValueError, OverflowError):
                    continue
                if index >= 0 and index not in indices:
                    indices.append(index)
        if not indices and item.get("primary_block_index") is not None:
            try:
                index = int(item.get("primary_block_index"))
            except (TypeError, ValueError, OverflowError):
                index = -1
            if index >= 0:
                indices = [index]
        if indices:
            anchored.setdefault(indices[0], []).append(item)
            covered.update(indices)
        else:
            try:
                target = int(item.get("insert_before_block_index"))
            except (TypeError, ValueError, OverflowError):
                target = len(primary_doc.blocks)
            target = min(max(target, 0), len(primary_doc.blocks))
            insertions.setdefault(target, []).append(item)

    result = copy.deepcopy(primary_doc)
    rebuilt: list[Block] = []

    def emit_items(rows: Iterable[dict], source: Block | None) -> None:
        for item in sorted(rows, key=lambda raw: int(raw.get("row_index", 0) or 0)):
            rebuilt.append(_repair_block_from_item(item, source, export_revision=export_revision))

    for block_index, block in enumerate(primary_doc.blocks):
        emit_items(insertions.get(block_index, ()), None)
        rows = anchored.get(block_index)
        if rows:
            emit_items(rows, block)
            continue
        if block_index in covered and block.type in _TEXT_TYPES:
            continue
        rebuilt.append(copy.deepcopy(block))
    emit_items(insertions.get(len(primary_doc.blocks), ()), None)

    result.blocks = rebuilt
    for index, block in enumerate(result.blocks):
        block.reading_order = index
    result.metadata.source_engine = f"{result.metadata.source_engine or 'multi_ocr'}+ai_repair_epub"
    try:
        from adapters.findtext_centernet_ruby import (
            apply_ruby_overlay, refresh_preserved_ruby, strip_ruby_overlay,
        )
        ruby_overlay = package.get("ruby_overlay") if isinstance(package, dict) else None
        overlay_enabled = bool(
            isinstance(ruby_overlay, dict)
            and (ruby_overlay.get("document_metadata") or {}).get("ruby_preservation_enabled")
            and ruby_overlay.get("blocks")
        )
        if overlay_enabled:
            apply_ruby_overlay(result, ruby_overlay)
        elif bool(getattr(getattr(result, "metadata", None), "ruby_preservation_enabled", False)):
            refresh_preserved_ruby(result)
        else:
            strip_ruby_overlay(result, strip_candidate_geometry=False, strip_logs=False)
    except Exception:
        pass
    return result



def _normalised_image_path(value: str | Path) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return os.path.normcase(str(Path(text).expanduser().resolve(strict=False)))
    except Exception:
        return os.path.normcase(text)


def _retain_only_publication_images(doc: UnifiedDocument, allowed_image_paths: set[str]) -> UnifiedDocument:
    """Remove OCR source-page images without disturbing text-item mapping."""
    allowed = {_normalised_image_path(value) for value in allowed_image_paths if str(value or "").strip()}
    result = copy.deepcopy(doc)
    result.pages = [
        page for page in result.pages
        if _normalised_image_path(getattr(page, "image_path", "")) in allowed
    ]
    result.blocks = [
        block for block in result.blocks
        if block.type != BlockType.IMAGE_REF
        or _normalised_image_path(getattr(block, "image_path", "")) in allowed
    ]
    for index, block in enumerate(result.blocks):
        block.reading_order = index
    return result

def _warning_code(value: str) -> str:
    token = str(value or "").strip().lower()
    mappings = (
        (("missing", "漏", "empty", "缺失"), "possible_missing_column"),
        # Generic pipeline messages such as ``duplicate cleanup checked`` used
        # to mark almost every row as duplicate_suffix.  Only warnings that
        # explicitly describe a repeated tail/boundary are publication risks.
        (("duplicate_suffix", "repeated_suffix", "repeated_tail", "重复尾", "尾部重复", "跨页重复"), "duplicate_suffix"),
        (("cross", "跨", "粘连", "boundary"), "cross_item_boundary"),
        (("order", "错序", "顺序"), "possible_column_order_error"),
        (("quote", "引号"), "unbalanced_quote"),
        (("length", "长度"), "model_length_disagreement"),
        (("single_char", "字符", "glyph"), "character_disagreement"),
    )
    for needles, code in mappings:
        if any(needle in token for needle in needles):
            return code
    # Unknown warnings remain available in the raw evidence, but they are not
    # promoted to publication risk flags without a machine-understood meaning.
    return ""


def _quote_unbalanced(text: str) -> bool:
    pairs = (("「", "」"), ("『", "』"), ("（", "）"), ("(", ")"), ("【", "】"))
    return any(text.count(left) != text.count(right) for left, right in pairs)


def _has_reliable_repeated_suffix(value: str) -> bool:
    """Detect a genuinely duplicated tail without flagging titles/punctuation.

    The old 6-character check treated short headings, ellipses and ordinary
    Japanese repetition as duplicated OCR.  A publication blocker now requires
    a longer, information-bearing segment repeated verbatim at the very end.
    """
    text = re.sub(r"[\s　]+", "", str(value or ""))
    if len(text) < 24:
        return False
    limit = min(40, len(text) // 2)
    for size in range(8, limit + 1):
        segment = text[-size:]
        if segment != text[-2 * size:-size]:
            continue
        informative = re.sub(r"[。！？!?、，,.…‥―—─━‐‑‒–「」『』（）()【】\[\]]", "", segment)
        if len(informative) < 6 or len(set(informative)) < 4:
            continue
        return True
    return False


def _risk_reasons(item: dict) -> list[str]:
    reasons: list[str] = []
    warnings = list(item.get("warnings") or []) + list(item.get("character_fusion_warnings") or [])
    reasons.extend(_warning_code(str(value)) for value in warnings if str(value))
    candidates = [
        str(candidate.get("text", "") or "")
        for candidate in (item.get("candidates") or [])
        if isinstance(candidate, dict)
    ]
    nonempty = [value for value in candidates if value]
    lengths = [len(value) for value in nonempty]
    if len(set(candidates)) > 1:
        reasons.append("model_text_disagreement")
    if lengths and max(lengths) - min(lengths) > max(4, int(max(lengths) * 0.18)):
        reasons.append("model_length_disagreement")
    if nonempty and len(nonempty) != len(candidates):
        reasons.append("possible_missing_column")
    baseline = _normalise_plain_text(item.get("edited_text", item.get("original_fused_text", "")))
    if _quote_unbalanced(baseline):
        reasons.append("unbalanced_quote")
    if bool(item.get("local_reocr_recommended")):
        reasons.append("local_reocr_recommended")
    if bool(item.get("alignment_repaired")):
        reasons.append("alignment_repaired")
    number_sets = {tuple(re.findall(r"\d+(?:[.,]\d+)?", value)) for value in nonempty}
    if len(number_sets) > 1:
        reasons.append("numeric_disagreement")
    # Repeated suffix inside one candidate can indicate cross-page duplication,
    # but only accept a long information-bearing repeat.  Candidate consensus
    # alone is not evidence that a short title is duplicated.
    if any(_has_reliable_repeated_suffix(value) for value in nonempty):
        reasons.append("duplicate_suffix")
    return list(dict.fromkeys(reason for reason in reasons if reason))


def _risk_level_from_reasons(reasons: Sequence[str], confidence: float) -> str:
    high = {
        "possible_missing_column", "duplicate_suffix", "cross_item_boundary",
        "possible_column_order_error", "local_reocr_recommended",
    }
    if any(reason in high for reason in reasons):
        return "high"
    if reasons or confidence < 0.8:
        return "medium"
    return "none"


def _disagreement_spans(item: dict, baseline: str) -> list[dict]:
    spans: list[dict] = []
    for candidate in (item.get("candidates") or []):
        if not isinstance(candidate, dict):
            continue
        text = str(candidate.get("text", "") or "")
        if text == baseline:
            continue
        matcher = SequenceMatcher(None, baseline, text, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            spans.append({
                "model_index": int(candidate.get("model_index", 0) or 0),
                "model_label": str(candidate.get("model_label", "") or ""),
                "change_type": tag,
                "baseline_start": i1,
                "baseline_end": i2,
                "candidate_start": j1,
                "candidate_end": j2,
                "baseline_text": baseline[i1:i2],
                "candidate_text": text[j1:j2],
            })
            if len(spans) >= _MAX_DISAGREEMENT_SPANS:
                return spans
    return spans


def _candidate_consensus_summary(item: dict, baseline: str) -> dict:
    candidates = [candidate for candidate in (item.get("candidates") or []) if isinstance(candidate, dict)]
    texts = [str(candidate.get("text", "") or "") for candidate in candidates]
    nonempty = [value for value in texts if value]
    unique = list(dict.fromkeys(nonempty))
    all_match = bool(candidates) and len(nonempty) == len(candidates) and len(unique) == 1 and unique[0] == baseline
    labels = [str(candidate.get("model_label", "") or f"model_{index}") for index, candidate in enumerate(candidates)]
    hashes = {_label: _text_sha256(text) for _label, text in zip(labels, texts)}
    return {
        "status": "all_candidates_match_baseline" if all_match else ("disagreement" if len(unique) > 1 else "partial_or_empty"),
        "consensus_type": "all_models" if all_match else ("conflict" if len(unique) > 1 else "partial"),
        "model_count": len(candidates),
        "nonempty_model_count": len(nonempty),
        "unique_text_count": len(unique),
        "consensus_text": baseline if all_match else "",
        "consensus_models": labels if all_match else [],
        "candidate_hashes": hashes,
        "candidate_text_sha256": list(hashes.values()),
    }


def _one_pass_evidence_triggers(item: dict, source: Block | None, baseline: str, risk_reasons: Sequence[str]) -> list[str]:
    triggers: list[str] = []
    columns = [str(value) for value in (item.get("column_ids") or []) if str(value)]
    if len(columns) > 1:
        triggers.append("multiple_physical_columns")
    candidates = [str(value.get("text", "") or "") for value in (item.get("candidates") or []) if isinstance(value, dict)]
    nonempty = [value for value in candidates if value]
    if len(set(nonempty)) > 1 or len(nonempty) != len(candidates):
        triggers.append("model_disagreement")
    confidence = _finite_float(item.get("confidence", 0.0))
    if confidence <= 0.0 and _candidate_consensus_summary(item, baseline).get("status") == "all_candidates_match_baseline":
        confidence = 1.0
    level = _risk_level_from_reasons(risk_reasons, confidence)
    if level in {"high", "medium"}:
        triggers.append(f"{level}_risk")
    block_type = str(item.get("block_type", "") or "")
    if block_type in {BlockType.CHAPTER.value, BlockType.SECTION.value, BlockType.RUBY.value, BlockType.FOOTNOTE.value}:
        triggers.append(f"special_block_type:{block_type}")
    metadata = copy.deepcopy(getattr(source, "metadata", {}) or {}) if source is not None else {}
    metadata_blob = " ".join(str(key) + "=" + str(value) for key, value in metadata.items()).lower()
    if any(token in metadata_blob for token in ("status", "poem", "chant", "incant", "ruby", "chapter_title")):
        triggers.append("special_layout_metadata")
    if baseline.count("\n") >= 2:
        triggers.append("multi_line_layout")
    if looks_like_chapter_title(baseline):
        triggers.append("chapter_title_candidate")
    if bool(item.get("local_reocr_recommended")):
        triggers.append("local_reocr_recommended")
    if re.search(r"[A-Za-z]{3,}", baseline) and not re.search(r"(?:URL|ISBN|HTTP|HTTPS)", baseline, flags=re.I):
        triggers.append("abnormal_latin_sequence")
    return list(dict.fromkeys(triggers))


def _attach_full_physical_evidence(result: dict, item: dict, source: Block | None, triggers: Sequence[str]) -> None:
    result.update({
        "evidence_tier": "full_physical",
        "candidate_storage": "full_physical",
        "evidence_triggers": list(dict.fromkeys(str(value) for value in triggers if str(value))),
        "bbox": _bbox_list(source),
        "primary_block_index": item.get("primary_block_index"),
        "primary_block_indices": list(item.get("primary_block_indices") or []),
        "insert_before_block_index": item.get("insert_before_block_index"),
        "candidates": copy.deepcopy(item.get("candidates") or []),
        "physical_column_candidates": copy.deepcopy(item.get("physical_column_candidates") or []),
        "column_geometry": copy.deepcopy(item.get("column_geometry") or []),
        "model_confidences": copy.deepcopy(item.get("model_confidences") or []),
        "alignment_status": str(item.get("alignment_status", "") or ""),
        "alignment_notes": copy.deepcopy(item.get("alignment_notes") or []),
        "character_fusion_reason": str(item.get("character_fusion_reason", "") or ""),
        "character_fusion_warnings": [str(value) for value in (item.get("character_fusion_warnings") or [])],
        "character_fusion_evidence": copy.deepcopy(item.get("character_fusion_evidence") or {}),
    })


def _refresh_immutable_item_hash(result: dict) -> None:
    result["immutable_item_sha256"] = _sha256({
        key: value for key, value in result.items()
        if key not in {"original_text", "baseline_text", "delete_intentionally", "baseline_tokens", "immutable_item_sha256"}
    })


def _map_item(
    item: dict,
    *,
    target: str,
    html_id: str,
    mode: str,
    primary_doc: UnifiedDocument,
    export_revision: int,
) -> dict:
    row_index = int(item.get("row_index", 0) or 0)
    source: Block | None = None
    try:
        source_index = int(item.get("primary_block_index"))
    except (TypeError, ValueError, OverflowError):
        source_index = -1
    if 0 <= source_index < len(primary_doc.blocks):
        source = primary_doc.blocks[source_index]
    baseline = _normalise_plain_text(item.get("edited_text", item.get("original_fused_text", "")))
    block_type = str(item.get("block_type", "paragraph") or "paragraph")
    confidence = _finite_float(item.get("confidence", 0.0))
    consensus_probe = _candidate_consensus_summary(item, baseline)
    if confidence <= 0.0 and consensus_probe.get("status") == "all_candidates_match_baseline":
        confidence = 1.0
    risk_reasons = _risk_reasons(item)
    result = {
        "item_id": str(item.get("row_id") or item.get("item_id") or ""),
        "row_index": row_index,
        "epub_target": target,
        "html_id": html_id,
        "page": int(item.get("page", 0) or 0),
        "block_type": block_type,
        "content_format": _content_format(block_type),
        "column_ids": [str(value) for value in (item.get("column_ids") or []) if str(value)],
        "original_text": baseline,
        "baseline_text": baseline,
        "baseline_text_sha256": _text_sha256(baseline),
        "export_revision": int(export_revision),
        "delete_intentionally": _explicit_bool(
            item.get("delete_intentionally", False),
            field=f"条目 {row_index + 1} 的 delete_intentionally",
        ),
        "risk_level": _risk_level_from_reasons(risk_reasons, confidence),
        "risk_reasons": risk_reasons,
    }
    # Ruby is an immutable side-channel only when the authoritative document
    # explicitly says the feature is enabled.  Stray/stale block metadata from
    # a previous run must never leak into an AI package created while Ruby is OFF.
    from adapters.findtext_centernet_ruby import has_ruby_overlay
    if (
        source is not None
        and has_ruby_overlay(primary_doc)
        and isinstance(getattr(source, "metadata", None), dict)
    ):
        ruby_annotations = copy.deepcopy(source.metadata.get("ruby_annotations") or [])
        if not ruby_annotations:
            ruby_source = str(source.metadata.get("ruby_aozora", "") or "")
            ruby_annotations = [
                {"base": m.group(1), "reading": m.group(2)}
                for m in re.finditer(r"[｜|]([^《\n]+)《([^》\n]+)》", ruby_source)
            ]
        if ruby_annotations:
            # Immutable evidence: AI edits only plain base text; readings are
            # re-attached by the app after import and are never model-generated.
            result["ruby_locked_annotations"] = ruby_annotations
            result["ruby_preservation_policy"] = "locked_reading_reapply_after_text_edit"
    if result["content_format"] == "inline_tokens_v1":
        result["baseline_tokens"] = _inline_tokens(baseline)
    if mode in {"standard", "expert", "one_pass"}:
        result.update({
            "confidence": confidence,
            "reason": str(item.get("reason", "") or ""),
            "disagreement_spans": _disagreement_spans(item, baseline),
            "character_fused_text": str(item.get("character_fused_text", "") or ""),
            "character_fusion_confidence": _finite_float(item.get("character_fusion_confidence", 0.0)),
            "local_reocr_recommended": bool(item.get("local_reocr_recommended", False)),
        })
    if mode in {"standard", "expert"}:
        result["candidates"] = copy.deepcopy(item.get("candidates") or [])
    if mode == "one_pass":
        result["candidate_consensus"] = consensus_probe
        result["evidence_tier"] = "collapsed_consensus"
        result["candidate_storage"] = "collapsed_consensus"
        result["consensus_type"] = str(consensus_probe.get("consensus_type", "partial") or "partial")
        result["auto_fused_text"] = baseline
        result["ai_adjudicated_text"] = str(item.get("ai_adjudicated_text", "") or "") or None
        result["has_ai_adjudication"] = bool(item.get("has_ai_adjudication") and item.get("ai_adjudicated_text"))
        result["evidence_triggers"] = []
        triggers = _one_pass_evidence_triggers(item, source, baseline, risk_reasons)
        if triggers:
            _attach_full_physical_evidence(result, item, source, triggers)
    if mode == "expert":
        _attach_full_physical_evidence(result, item, source, ["expert_mode"])
    _refresh_immutable_item_hash(result)
    return result


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise AIRepairEpubError(f"JSON 包含重复键：{key}")
        result[key] = value
    return result


def _validate_json_shape(value: Any, *, depth: int = 0) -> None:
    if depth > _MAX_JSON_DEPTH:
        raise AIRepairEpubError("JSON 嵌套深度超过安全限制。")
    if isinstance(value, dict):
        if len(value) > _MAX_JSON_ARRAY_ITEMS:
            raise AIRepairEpubError("JSON 对象字段数量超过安全限制。")
        for child in value.values():
            _validate_json_shape(child, depth=depth + 1)
    elif isinstance(value, list):
        if len(value) > _MAX_JSON_ARRAY_ITEMS:
            raise AIRepairEpubError("JSON 数组长度超过安全限制。")
        for child in value:
            _validate_json_shape(child, depth=depth + 1)


def _json_loads_strict(data: str | bytes, *, source: str = "JSON") -> Any:
    raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
    if len(raw) > _MAX_JSON_BYTES:
        raise AIRepairEpubError(f"{source} 超过安全大小限制。")
    try:
        value = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_strict_json_object)
    except AIRepairEpubError:
        raise
    except Exception as exc:
        raise AIRepairEpubError(f"无法解析 {source}：{exc}") from exc
    _validate_json_shape(value)
    return value


def _validate_zip_member_name(name: str) -> None:
    if not name or "\x00" in name or "\\" in name:
        raise AIRepairEpubError(f"EPUB 包含非法 ZIP 路径：{name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise AIRepairEpubError(f"EPUB 包含路径穿越或绝对路径：{name!r}")
    if re.match(r"^[A-Za-z]:", name):
        raise AIRepairEpubError(f"EPUB 包含 Windows 绝对路径：{name!r}")


def _validate_zip_archive(archive: zipfile.ZipFile) -> None:
    infos = archive.infolist()
    if len(infos) > _MAX_ZIP_MEMBERS:
        raise AIRepairEpubError("EPUB 文件数量超过安全限制。")
    names: set[str] = set()
    total = 0
    for info in infos:
        _validate_zip_member_name(info.filename)
        if info.filename in names:
            raise AIRepairEpubError(f"EPUB 包含重复 ZIP 成员：{info.filename}")
        names.add(info.filename)
        if info.flag_bits & 0x1:
            raise AIRepairEpubError(f"EPUB 包含加密成员，拒绝读取：{info.filename}")
        unix_mode = (info.external_attr >> 16) & 0o170000
        if unix_mode == 0o120000:
            raise AIRepairEpubError(f"EPUB 包含符号链接，拒绝读取：{info.filename}")
        if info.is_dir():
            continue
        if info.file_size > _MAX_ZIP_SINGLE_UNCOMPRESSED:
            raise AIRepairEpubError(f"EPUB 单文件解压大小超过限制：{info.filename}")
        total += int(info.file_size)
        if total > _MAX_ZIP_TOTAL_UNCOMPRESSED:
            raise AIRepairEpubError("EPUB 总解压大小超过安全限制，疑似 ZIP Bomb。")
        if info.file_size > 1024 * 1024:
            compressed = max(1, int(info.compress_size))
            if info.file_size / compressed > 5000:
                raise AIRepairEpubError(f"EPUB 压缩比异常，疑似 ZIP Bomb：{info.filename}")


@contextmanager
def _validated_epub(path: Path) -> Iterator[zipfile.ZipFile]:
    try:
        archive = zipfile.ZipFile(path, "r")
    except Exception as exc:
        raise AIRepairEpubError(f"无法打开 EPUB：{exc}") from exc
    try:
        _validate_zip_archive(archive)
        yield archive
    finally:
        archive.close()


def _safe_xml_root(data: bytes, *, source: str) -> ET.Element:
    if len(data) > _MAX_XML_BYTES:
        raise AIRepairEpubError(f"XML/XHTML 超过安全大小限制：{source}")
    lowered = data.lower()
    if b"<!entity" in lowered:
        raise AIRepairEpubError(f"XML/XHTML 含实体声明，拒绝解析：{source}")
    doctype_matches = re.findall(br"<!doctype\s+([^>]+)>", lowered, flags=re.DOTALL)
    if any(match.strip() != b"html" for match in doctype_matches):
        raise AIRepairEpubError(f"XML/XHTML 含外部或复杂 DTD，拒绝解析：{source}")
    try:
        root = ET.fromstring(data)
    except Exception as exc:
        raise AIRepairEpubError(f"XML/XHTML 无法解析：{source}：{exc}") from exc
    for element in root.iter():
        local = _local(element.tag).lower()
        if local == "script":
            raise AIRepairEpubError(f"外部 XHTML 含 script，拒绝读取：{source}")
        for attr in ("href", "src", "xlink:href"):
            value = str(element.attrib.get(attr, "") or "").strip()
            if not value:
                continue
            match = re.match(r"^([A-Za-z][A-Za-z0-9+.-]*):", value)
            if match and match.group(1).lower() in _DANGEROUS_SCHEMES:
                raise AIRepairEpubError(f"外部 XHTML 含危险链接 {value!r}：{source}")
    return root


def _scan_tagged_elements(archive: zipfile.ZipFile) -> dict[str, tuple[str, str, ET.Element]]:
    found: dict[str, tuple[str, str, ET.Element]] = {}
    for name in archive.namelist():
        if not name.lower().endswith((".xhtml", ".html", ".htm")):
            continue
        root = _safe_xml_root(archive.read(name), source=name)
        for element in root.iter():
            item_id = str(element.attrib.get("data-item-id", "") or "").strip()
            if not item_id:
                continue
            html_id = str(element.attrib.get("id", "") or "").strip()
            if not html_id:
                raise AIRepairEpubError(f"{name} 中的 {item_id} 缺少 XHTML id。")
            if item_id in found:
                raise AIRepairEpubError(f"EPUB 中出现重复稳定段落 ID：{item_id}")
            found[item_id] = (name, html_id, element)
    return found


def _archive_image_manifest(archive: zipfile.ZipFile) -> list[dict]:
    images: list[dict] = []
    for name in archive.namelist():
        lowered = name.lower()
        if not lowered.startswith("epub/images/") or lowered.endswith("/"):
            continue
        data = archive.read(name)
        images.append({"path": name, "size": len(data), "sha256": _sha256_bytes(data)})
    return images


def _spine_order(archive: zipfile.ZipFile) -> dict[str, int]:
    opf_names = [name for name in archive.namelist() if name.lower().endswith(".opf")]
    if not opf_names:
        return {}
    opf_name = sorted(opf_names, key=lambda name: (name.lower() != "epub/content.opf", len(name)))[0]
    root = _safe_xml_root(archive.read(opf_name), source=opf_name)
    manifest: dict[str, str] = {}
    for element in root.iter():
        if _local(element.tag).lower() == "item":
            item_id = str(element.attrib.get("id", "") or "")
            href = str(element.attrib.get("href", "") or "")
            if item_id and href:
                archive_path = posixpath.normpath(posixpath.join(posixpath.dirname(opf_name), href))
                if archive_path.startswith("../"):
                    continue
                manifest[item_id] = archive_path
    order: dict[str, int] = {}
    index = 0
    for element in root.iter():
        if _local(element.tag).lower() != "itemref":
            continue
        href = manifest.get(str(element.attrib.get("idref", "") or ""))
        if href and href not in order:
            order[href] = index
            index += 1
    return order


def _detect_chapter_candidates(primary_doc: UnifiedDocument, items: Sequence[dict] | None = None) -> list[dict]:
    item_list = list(items or [])
    item_by_block: dict[int, dict] = {}
    for item in item_list:
        indices = list(item.get("primary_block_indices") or [])
        if item.get("primary_block_index") is not None:
            indices.append(item.get("primary_block_index"))
        for raw_index in indices:
            try:
                item_by_block.setdefault(int(raw_index), item)
            except (TypeError, ValueError, OverflowError):
                continue
    toc_titles = {str(getattr(entry, "title", "") or "").strip() for entry in getattr(primary_doc, "toc", []) if str(getattr(entry, "title", "") or "").strip()}
    candidates: list[dict] = []
    for block_index, block in enumerate(primary_doc.blocks):
        if block.type == BlockType.IMAGE_REF or (block.metadata or {}).get("consumed"):
            continue
        text = _normalise_plain_text(block.text)
        if not text:
            continue
        reasons: list[str] = []
        score = 0.0
        if block.type == BlockType.CHAPTER:
            reasons.append("typed_chapter")
            score += 1.0
        if looks_like_chapter_title(text):
            reasons.append("chapter_title_pattern")
            score += 0.82
        if text in toc_titles:
            reasons.append("matches_toc_title")
            score += 0.72
        metadata = block.metadata or {}
        if bool(metadata.get("chapter_title_atomic")) or bool(metadata.get("is_chapter_title")):
            reasons.append("chapter_metadata")
            score += 0.75
        if block.type == BlockType.SECTION and len(text) <= 80:
            reasons.append("short_section_heading")
            score += 0.35
        if len(text) <= 48 and "\n" not in text and not text.endswith(tuple("。！？!?」』）】")):
            if re.search(r"(?:第[一二三四五六七八九十百千万0-9０-９]+[章話節幕]|プロローグ|エピローグ|序章|終章|幕間|外伝|間章)", text):
                reasons.append("compact_heading_shape")
                score += 0.55
        if not reasons or score < 0.55:
            continue
        item = item_by_block.get(block_index)
        candidates.append({
            "candidate_id": f"chapter-candidate-{len(candidates) + 1:04d}",
            "block_index": block_index,
            "block_id": str(block.id or ""),
            "item_id": str((item or {}).get("item_id", "") or ""),
            "row_index": int((item or {}).get("row_index", -1) or -1),
            "page": int(getattr(block, "page", 0) or 0),
            "text": text,
            "confidence": round(min(1.0, score), 3),
            "reasons": list(dict.fromkeys(reasons)),
            "source_block_type": block.type.value,
            "requires_type_repair": block.type != BlockType.CHAPTER,
        })
    return candidates


def _apply_one_pass_context_evidence(mapped_items: list[dict], source_items: Sequence[dict], primary_doc: UnifiedDocument) -> None:
    by_id = {str(item.get("row_id") or item.get("item_id") or ""): item for item in source_items}
    ordered = sorted(mapped_items, key=lambda item: int(item.get("row_index", 0) or 0))
    page_groups: dict[int, list[dict]] = {}
    for item in ordered:
        page_groups.setdefault(int(item.get("page", 0) or 0), []).append(item)
    context_triggers: dict[str, list[str]] = {}
    for page, group in page_groups.items():
        if not group:
            continue
        context_triggers.setdefault(str(group[0].get("item_id", "")), []).append("page_first_item")
        context_triggers.setdefault(str(group[-1].get("item_id", "")), []).append("page_last_item")
    chapter_candidates = _detect_chapter_candidates(primary_doc, ordered)
    for candidate in chapter_candidates:
        if candidate.get("item_id"):
            context_triggers.setdefault(str(candidate["item_id"]), []).append("chapter_candidate")
    item_positions: list[tuple[int, dict]] = []
    for item in ordered:
        raw = by_id.get(str(item.get("item_id", ""))) or {}
        indices = list(raw.get("primary_block_indices") or [])
        if raw.get("primary_block_index") is not None:
            indices.append(raw.get("primary_block_index"))
        parsed = []
        for value in indices:
            try: parsed.append(int(value))
            except (TypeError, ValueError, OverflowError): pass
        if parsed:
            item_positions.append((min(parsed), item))
    for block_index, block in enumerate(primary_doc.blocks):
        if block.type != BlockType.IMAGE_REF:
            continue
        before = [pair for pair in item_positions if pair[0] < block_index]
        after = [pair for pair in item_positions if pair[0] > block_index]
        if before:
            context_triggers.setdefault(str(before[-1][1].get("item_id", "")), []).append("before_image")
        if after:
            context_triggers.setdefault(str(after[0][1].get("item_id", "")), []).append("after_image")
    for item in ordered:
        item_id = str(item.get("item_id", ""))
        triggers = context_triggers.get(item_id) or []
        if not triggers:
            continue
        raw = by_id.get(item_id) or {}
        source = None
        try:
            idx = int(raw.get("primary_block_index"))
        except (TypeError, ValueError, OverflowError):
            idx = -1
        if 0 <= idx < len(primary_doc.blocks):
            source = primary_doc.blocks[idx]
        merged = list(item.get("evidence_triggers") or []) + triggers
        _attach_full_physical_evidence(item, raw, source, merged)
        _refresh_immutable_item_hash(item)


def _enrich_reading_context(
    items: list[dict],
    spine: dict[str, int],
    *,
    context_size: int = 2,
    chapter_candidates: Sequence[dict] | None = None,
) -> None:
    ordered = sorted(items, key=lambda item: int(item.get("row_index", 0) or 0))
    candidate_by_item = {
        str(candidate.get("item_id", "")): candidate
        for candidate in (chapter_candidates or [])
        if str(candidate.get("item_id", ""))
    }
    chapter_no = 0
    chapter_id = "front-matter"
    chapter_title = "前书页"
    paragraph_index = 0
    for reading_order, item in enumerate(ordered):
        baseline = str(item.get("baseline_text", item.get("original_text", "")) or "")
        candidate = candidate_by_item.get(str(item.get("item_id", "")))
        is_chapter = bool(candidate) or str(item.get("block_type", "")) == BlockType.CHAPTER.value
        if is_chapter:
            chapter_no += 1
            chapter_id = f"chapter-{chapter_no:03d}"
            chapter_title = str((candidate or {}).get("text", "") or baseline or f"章节 {chapter_no}")
            paragraph_index = 0
            item["chapter_candidate"] = copy.deepcopy(candidate or {
                "text": chapter_title,
                "confidence": 1.0,
                "reasons": ["typed_chapter"],
                "requires_type_repair": False,
            })
        item["chapter_id"] = chapter_id
        item["chapter_title"] = chapter_title
        item["reading_order"] = reading_order
        item["paragraph_index"] = paragraph_index
        paragraph_index += 1
        target_path = str(item.get("epub_target", "") or "").split("#", 1)[0]
        item["spine_index"] = int(spine.get(target_path, -1))
    for index, item in enumerate(ordered):
        item["prev_item_id"] = ordered[index - 1]["item_id"] if index > 0 else ""
        item["next_item_id"] = ordered[index + 1]["item_id"] if index + 1 < len(ordered) else ""
        before = ordered[max(0, index - context_size):index]
        after = ordered[index + 1:index + 1 + context_size]
        item["context_before"] = "\n\n".join(str(value.get("baseline_text", "") or "") for value in before)
        item["context_after"] = "\n\n".join(str(value.get("baseline_text", "") or "") for value in after)


def _baseline_book_sha256(items: Sequence[dict]) -> str:
    projection = [
        {
            "item_id": str(item.get("item_id") or item.get("row_id") or ""),
            "baseline_text_sha256": str(item.get("baseline_text_sha256") or _text_sha256(item.get("baseline_text", item.get("edited_text", "")))),
            "delete_intentionally": bool(item.get("delete_intentionally", False)),
        }
        for item in items
    ]
    return _sha256(projection)


def _guide_text(mode: str) -> str:
    return f"""# Novel Formatter 外部 AI 日文 OCR 修复说明

此 EPUB 是结构锁定的校订底稿，证据模式为 `{mode}`。EPUB 用于完整阅读上下文；**首选回写方式是稀疏 JSON**，直接修改 EPUB 仅作兼容。

## 推荐工作流

1. 阅读正文、插图与 `META-INF/ai-repair/chapters/*.json`。
2. 只处理有风险或确实需要修改的条目。
3. 返回 `META-INF/ai-repair/result-template.json` 同结构的稀疏 JSON。
4. 每条更新原样携带 `expected_baseline_sha256`；发生跨条目重排时使用原子事务。

## 允许操作

- 只修复日文 OCR 错字、漏字、重复、断句和错序；AI 可编辑内容始终是普通正文底字。
- Ruby/振假名由程序作为锁定结构层独立保存；`ruby_locked_annotations` 只读，禁止修改、删除、猜测或新增读音。
- 保留稳定 `item_id`，不得翻译、润色、续写或概括。
- 普通多行文本使用 `\\n`；程序会安全回写为 `<br/>`。
- 不得生成 `ruby` token 或把振假名混入 `edited_text`；导入后程序会按最新正文安全重新挂接原 Ruby。

## 稀疏 JSON 示例

```json
{{
  "schema": "{AI_REPAIR_EDITS_SCHEMA}",
  "package_id": "从 manifest.json 原样复制",
  "structure_sha256": "从 manifest.json 原样复制",
  "map_sha256": "从 manifest.json 原样复制",
  "updates": [
    {{
      "item_id": "row:000123:...",
      "baseline_text": "导出时的融合文本",
      "expected_baseline_sha256": "该段 baseline_text_sha256",
      "edited_text": "校订后的日文",
      "delete_intentionally": false,
      "confidence": "high_consensus",
      "reason": "两路 OCR 与上下文一致"
    }}
  ],
  "transactions": [
    {{
      "transaction_id": "boundary-fix-00017",
      "operation": "rebalance_adjacent_items",
      "item_ids": ["row:000123:...", "row:000124:..."],
      "reason": "跨条目断句并包含重复后缀",
      "updates": [
        {{"item_id": "row:000123:...", "baseline_text": "彼は扉を開", "expected_baseline_sha256": "...", "edited_text": "彼は扉を開けた。"}},
        {{"item_id": "row:000124:...", "baseline_text": "開けた。彼は扉を開けた。", "expected_baseline_sha256": "...", "edited_text": "", "delete_intentionally": true}}
      ]
    }}
  ]
}}
```

原子事务必须覆盖连续条目；程序要么完整接收，要么完整拒绝。若本地正文已变化，程序会做逐条三方冲突检测：非重叠改动生成待确认合并候选；重叠冲突标记“结果已过期”，绝不自动选择。

## 直接编辑 EPUB（兼容）

只修改带 `data-item-id` 的 h1/h2/p 内容。保留全部属性、元素、资源和顺序。确需清空时保留元素并设置 `data-delete-intentionally="true"`。不得修改映射、图片、CSS、NAV/NCX、OPF 或其他文件。
"""


def _publication_guide_text(
    mode: str,
    *,
    framework_name: str = "framework/resource_mapping_framework.epub",
    final_name: str = "AI精校出版版.epub",
    publication_reference_available: bool = False,
) -> str:
    reference_note = (
        "本包包含用户显式选择的 `reference/` 出版参考证据。它可提供日文层、结构和可靠 Ruby，"
        "但不得静默覆盖 `proposed_text`；正文不一致时必须转为复核。"
        if publication_reference_available else
        "本包未包含出版参考 EPUB。不得从文字对比页旧路径、网络文本或其他版本静默补入正文。"
    )
    ruby_note = (
        "有 `reference/` 中可靠对齐且底字完全一致的 Ruby 证据时，允许恢复对应 `<ruby><rt>`；"
        "无证据时只保留底字，绝不猜读音。"
        if publication_reference_available else
        "OCR 正文只含底字；没有可靠出版参考证据，不得生成 Ruby 读音。"
    )
    return f"""# Novel Formatter：AI 修复包 v3

这是一个**只导出、不回传 Formatter**的日文 OCR 出版修复包。当前界面已经确认的 `edited_text` 是唯一默认文字母本；任何 `character_fused_text`、隐藏 AI 结果或单模型候选都只是证据，导出阶段不得再次替用户选择正文。不要返回稀疏 JSON。

最终必须交付：

1. `{final_name}`；
2. 完整 `audit/` 审计文件，或包含该目录的 ZIP。

## 强制证据顺序

1. `12_stable_text_map.jsonl`：唯一默认正文权威，读取 `proposed_text` 和 `edit_policy`；
2. `16_page_column_ledger.json`：页面物理正文列总账，任何未映射正文列都是 fatal preflight；
3. `13_atomic_span_map.jsonl`：只从物理列、模型候选、原始 OCR block 或已验证人工文字生成，不允许用 proposed_text 循环证明；每个可靠 atomic span 必须在成品中恰好覆盖一次；
4. `14_global_text_anomalies.json`：重复、移动、无支持插入、orphan span 与顺序冲突；
5. `15_output_structure_plan.json`：章节、分片、独立插图页和 spine；
6. `full_fusion_evidence.json`：完整 OCR 候选证据，不是默认正文；
7. `reading/`、`evidence/`：都必须是稳定表的同一文字投影；
8. `visual_evidence/`：高风险列、相邻列上下文、未映射物理列的选择性裁切图。不得声称未导出的整页扫描已经被逐字核验。

{reference_note}

## 三类编辑策略

- `locked_consensus`：`proposed_text`、当前 edited baseline 和至少两模型唯一共识完全一致，且物理列与源 block 覆盖完整。默认逐字保留；
- `review_required`：任一文字、覆盖、顺序、边界或参考证据冲突，必须复核；
- `structure_only`：文字锁定，只允许调整段落、状态栏、换行、插图和 Ruby 标记，不得改变纯底字序列。

任何锁定条目修改都必须记录明确 `unlock_reason`。不能因为语言更自然、统一表记或模型偏好而改写。

## OCR Ruby 与出版 Ruby 分层

分列和普通 OCR 阶段，Ruby 小字不得进入正文物理列，正文候选只识别底字。{ruby_note}

合法 Ruby 必须同时满足：

- 读音来自包内显式参考证据；
- Ruby 底字纯文本与 `proposed_text` 对应片段完全一致；
- reading 不为空；
- 纯文本回读只出现一次底字。

`ruby_without_evidence`、`ruby_base_text_mismatch`、`ruby_reading_empty` 都是 fatal。

## 特殊排版

状态栏优先使用 OCR 原始行框、坐标和行距。标签分隔符包括 `：`、`:`、`・・`、`･`、`·`、`‥`、`…`。最后一个状态值不得吞入后续叙述正文；按 `narrative_suffix` 与 `must_split_suffix_to_next_paragraph` 拆分。

诗歌、咏唱、人物资料、脚注、信件和系统信息必须保持硬边界。数字、等级、小假名、`〝〟`、`──`、全角标点和破折号样式必须审计。

## 结构、图片与元数据

`{framework_name}` 是干净资源映射框架，只提供资源和映射，不是最终文字真值。必须按预检选择 `preserve_and_patch`、`hybrid_rebuild` 或 `full_rebuild`；不得把框架原样当成品。独立插图 spine 节点必须生成独立图片 XHTML，图片前后强制分片。技术分片不得增加额外 NAV 章节。

修复包导出时书名、作者可以为空；最终出版构建时：

- 已有正确 title、author、identifier 必须保留；
- language 必须为 `ja`；
- 确实没有标题时才使用通用文件名；
- 不得通过封面 OCR 猜测缺失元数据。

## 两遍构建与致命审计

第一遍生成完整文字与结构映射；第二遍在构建 EPUB 后重新打开，并执行下方本地审计器。

## 必须执行本地审计器

构建后在包根目录运行：

```bash
python tools/validate_final_epub.py \\
  --epub "{final_name}" \\
  --stable-map 12_stable_text_map.jsonl \\
  --atomic-map 13_atomic_span_map.jsonl \\
  --structure-plan 15_output_structure_plan.json \\
  --column-ledger 16_page_column_ledger.json \\
  --global-anomalies 14_global_text_anomalies.json \\
  --book-identity 02_book_identity.json \\
  --assets-manifest 05_assets_manifest.json
```

必须生成：

- `audit/audit_report.json`
- `audit/audit_report.md`
- `audit/text_changes.csv`
- `audit/atomic_span_coverage.csv`
- `audit/unresolved_anomalies.csv`
- `audit/epub_integrity.json`

只写一份“审计通过”说明不算验证。任一 `11_final_audit_rules.json` fatal check 失败时，不得交付或宣称完成。

## 最终 EPUB 禁止项

不得嵌入 `META-INF/ai-repair/`、`META-INF/ai-publication/`、`evidence/`、`reading/`、`framework/`、`visual_evidence/`、`reference/`、`tools/`、`audit/` 或 `full_fusion_evidence.json`；不得保留 `data-item-id`、`data-row-id`、`data-block-id`。不得出现 `□`、`�`、NUL、断链、重复 HTML id、NAV 进入 spine 或无法解析的 XML/XHTML。
"""

def _publication_review_priority(item: dict) -> int:
    level = str(item.get("risk_level", "none") or "none")
    base = {"high": 0, "medium": 1000, "low": 2000, "none": 3000}.get(level, 1500)
    reasons = set(str(value) for value in (item.get("risk_reasons") or []))
    weights = {
        "possible_missing_column": -300,
        "possible_column_order_error": -260,
        "cross_item_boundary": -240,
        "duplicate_suffix": -220,
        "unbalanced_quote": -180,
        "model_length_disagreement": -120,
        "numeric_disagreement": -80,
    }
    return base + sum(weights.get(reason, 0) for reason in reasons) + int(item.get("reading_order", 0) or 0)


def _publication_boundary_windows(items: Sequence[dict]) -> list[dict]:
    ordered = sorted(items, key=lambda item: int(item.get("reading_order", item.get("row_index", 0)) or 0))
    trigger_reasons = {
        "possible_missing_column", "possible_column_order_error", "cross_item_boundary",
        "duplicate_suffix", "unbalanced_quote", "model_length_disagreement",
    }
    windows: list[dict] = []
    seen: set[tuple[str, ...]] = set()
    for index, item in enumerate(ordered):
        reasons = set(str(value) for value in (item.get("risk_reasons") or []))
        if str(item.get("risk_level", "")) != "high" and not (reasons & trigger_reasons):
            continue
        start = max(0, index - 2)
        end = min(len(ordered), index + 3)
        members = ordered[start:end]
        ids = tuple(str(member.get("item_id", "") or "") for member in members)
        if not ids or ids in seen:
            continue
        seen.add(ids)
        windows.append({
            "window_id": f"boundary-window-{len(windows) + 1:05d}",
            "focus_item_id": str(item.get("item_id", "") or ""),
            "trigger_reasons": sorted(reasons & trigger_reasons) or list(item.get("risk_reasons") or []),
            "item_ids": list(ids),
            "combined_baseline": "\n\n".join(str(member.get("baseline_text", "") or "") for member in members),
            "items": [
                {
                    "item_id": str(member.get("item_id", "") or ""),
                    "block_type": str(member.get("block_type", "") or ""),
                    "baseline_text": str(member.get("baseline_text", "") or ""),
                    "epub_target": str(member.get("epub_target", "") or ""),
                }
                for member in members
            ],
        })
    return windows


def _publication_page_boundary_windows(items: Sequence[dict]) -> list[dict]:
    ordered = sorted(items, key=lambda item: int(item.get("reading_order", item.get("row_index", 0)) or 0))
    pages: dict[int, list[dict]] = {}
    for item in ordered:
        pages.setdefault(int(item.get("page", 0) or 0), []).append(item)
    page_numbers = sorted(pages)
    windows: list[dict] = []
    for left_page, right_page in zip(page_numbers, page_numbers[1:]):
        left = pages[left_page]
        right = pages[right_page]
        if not left or not right:
            continue
        members = left[-2:] + right[:2]
        windows.append({
            "window_id": f"page-boundary-{left_page:05d}-{right_page:05d}",
            "left_page": left_page,
            "right_page": right_page,
            "left_last_item_id": str(left[-1].get("item_id", "") or ""),
            "right_first_item_id": str(right[0].get("item_id", "") or ""),
            "item_ids": [str(item.get("item_id", "") or "") for item in members],
            "combined_baseline": "\n\n".join(str(item.get("baseline_text", "") or "") for item in members),
            "requires_continuation_review": not str(left[-1].get("baseline_text", "") or "").rstrip().endswith(tuple("。！？!?」』）】…‥")),
            "items": [
                {
                    "item_id": str(item.get("item_id", "") or ""),
                    "page": int(item.get("page", 0) or 0),
                    "block_type": str(item.get("block_type", "") or ""),
                    "baseline_text": str(item.get("baseline_text", "") or ""),
                }
                for item in members
            ],
        })
    return windows


def _publication_compact_item(item: dict) -> dict:
    """Keep publication-useful evidence while removing repeated exchange metadata."""
    keep_keys = {
        "item_id", "row_id", "block_id", "row_index", "reading_order", "paragraph_index", "spine_index",
        "chapter_id", "chapter_title", "chapter_candidate_id",
        "epub_target", "html_id", "page", "block_type", "content_format",
        "column_ids", "baseline_text", "baseline_text_sha256", "baseline_tokens", "prev_item_id", "next_item_id",
        "context_before", "context_after",
        "risk_level", "risk_reasons", "confidence", "reason", "disagreement_spans",
        "character_fused_text", "character_fusion_confidence", "local_reocr_recommended",
        "bbox", "primary_block_index", "primary_block_indices", "insert_before_block_index",
        "physical_column_candidates", "column_geometry", "model_confidences", "alignment_status",
        "alignment_notes", "character_fusion_reason", "character_fusion_warnings",
        "character_fusion_evidence", "candidate_consensus", "evidence_tier",
        "candidate_storage", "consensus_type", "auto_fused_text", "ai_adjudicated_text",
        "has_ai_adjudication", "evidence_triggers", "chapter_candidate",
        "final_html_id", "planned_final_xhtml", "planned_final_target",
        "framework_epub_target", "technical_part_index",
    }
    compact = {key: copy.deepcopy(value) for key, value in item.items() if key in keep_keys}
    baseline = str(item.get("baseline_text", "") or "")
    candidates = [candidate for candidate in (item.get("candidates") or []) if isinstance(candidate, dict)]
    candidate_texts = [str(candidate.get("text", "") or "") for candidate in candidates]
    nonempty_unique = list(dict.fromkeys(value for value in candidate_texts if value))
    if candidates and len(nonempty_unique) == 1 and nonempty_unique[0] == baseline:
        existing = copy.deepcopy(compact.get("candidate_consensus") or {})
        existing.update({
            "status": "all_candidates_match_baseline",
            "consensus_type": "all_models",
            "model_count": len(candidates),
            "nonempty_model_count": len(candidates),
            "unique_text_count": 1,
            "consensus_text": baseline,
            "consensus_models": [str(candidate.get("model_label", "") or f"model_{index}") for index, candidate in enumerate(candidates)],
            "candidate_hashes": {
                str(candidate.get("model_label", "") or f"model_{index}"): _text_sha256(str(candidate.get("text", "") or ""))
                for index, candidate in enumerate(candidates)
            },
        })
        compact["candidate_consensus"] = existing
    elif candidates:
        compact["candidate_consensus"] = {
            **copy.deepcopy(compact.get("candidate_consensus") or {}),
            "status": "disagreement" if len(nonempty_unique) > 1 else "partial_or_empty",
            "consensus_type": "conflict" if len(nonempty_unique) > 1 else "partial",
            "model_count": len(candidates),
            "unique_text_count": len(nonempty_unique),
        }
        compact["candidates"] = copy.deepcopy(candidates)
    elif not compact.get("candidate_consensus"):
        compact["candidate_consensus"] = {"status": "unavailable", "consensus_type": "unknown", "model_count": 0}
    if str(compact.get("character_fused_text", "") or "") == baseline:
        compact.pop("character_fused_text", None)
    if not compact.get("disagreement_spans"):
        compact.pop("disagreement_spans", None)
    for key in list(compact):
        if compact[key] in (None, "", [], {}):
            if key not in {"baseline_text", "item_id", "epub_target", "html_id", "block_type"}:
                compact.pop(key, None)
    return compact

def _publication_fusion_payload(
    repair_map: dict,
    package: dict,
    *,
    framework_name: str,
    final_name: str,
    guide_name: str,
) -> dict:
    raw_items = [copy.deepcopy(item) for item in (repair_map.get("items") or [])]
    compact_by_id = {str(item.get("item_id", "") or ""): _publication_compact_item(item) for item in raw_items}
    items = [compact_by_id[str(item.get("item_id", "") or "")] for item in raw_items]
    chapters: list[dict] = []
    chapter_groups: dict[str, list[dict]] = {}
    chapter_titles: dict[str, str] = {}
    raw_by_id = {str(item.get("item_id", "") or ""): item for item in raw_items}
    for raw in sorted(raw_items, key=lambda value: int(value.get("reading_order", value.get("row_index", 0)) or 0)):
        chapter_id = str(raw.get("chapter_id", "front-matter") or "front-matter")
        item_id = str(raw.get("item_id", "") or "")
        chapter_groups.setdefault(chapter_id, []).append(compact_by_id[item_id])
        chapter_titles[chapter_id] = str(raw.get("chapter_title", "") or "")
    for chapter_id, chapter_items in chapter_groups.items():
        chapters.append({
            "chapter_id": chapter_id,
            "chapter_title": chapter_titles.get(chapter_id, ""),
            "item_count": len(chapter_items),
            "high_risk_count": sum(1 for item in chapter_items if item.get("risk_level") == "high"),
            "medium_risk_count": sum(1 for item in chapter_items if item.get("risk_level") == "medium"),
            "items": chapter_items,
        })
    risk_items = sorted(
        [item for item in raw_items if str(item.get("risk_level", "none")) != "none"],
        key=_publication_review_priority,
    )
    risk_queue = [
        {
            "priority": index + 1,
            "item_id": str(item.get("item_id", "") or ""),
            "chapter_id": str(item.get("chapter_id", "") or ""),
            "chapter_title": str(item.get("chapter_title", "") or ""),
            "reading_order": int(item.get("reading_order", 0) or 0),
            "risk_level": str(item.get("risk_level", "") or ""),
            "risk_reasons": list(item.get("risk_reasons") or []),
            "epub_target": str(item.get("epub_target", "") or ""),
            "baseline_text": str(item.get("baseline_text", "") or ""),
            "context_before": str(item.get("context_before", "") or ""),
            "context_after": str(item.get("context_after", "") or ""),
            "candidate_consensus": copy.deepcopy(compact_by_id[str(item.get("item_id", "") or "")].get("candidate_consensus") or {}),
        }
        for index, item in enumerate(risk_items)
    ]
    all_text = "\n\n".join(str(item.get("baseline_text", "") or "") for item in raw_items)
    model_sources = copy.deepcopy(package.get("model_sources") or [])
    return {
        "schema": AI_PUBLICATION_FUSION_SCHEMA,
        "purpose": "repair_japanese_ocr_and_generate_final_publication_epub",
        "evidence_profile": {
            "format": "layered_publication_v2",
            "identical_candidates_collapsed": True,
            "repeated_item_context_moved_to_risk_queue": True,
            "full_uncompressed_evidence_embedded_in_framework_epub": True,
        },
        "workflow": {
            "return_to_formatter": False,
            "primary_structure_base": framework_name,
            "evidence_file": "self",
            "instruction_file": guide_name,
            "required_final_output": final_name,
            "process": [
                "read_the_book_and_evidence_chapter_by_chapter",
                "repair_all_confirmed_ocr_errors_without_rewriting_style",
                "edit_the_framework_epub_directly",
                "run_whole_book_publication_quality_audit",
                "output_one_final_epub",
            ],
        },
        "publication_policy": {
            "language": "Japanese",
            "translation_forbidden": True,
            "style_rewriting_forbidden": True,
            "hallucinated_completion_forbidden": True,
            "preserve": [
                "cover", "illustrations", "spine_order", "toc", "css", "vertical_writing",
                "footnotes", "chapter_boundaries", "intentional_line_breaks", "metadata",
            ],
            "repair_scope": [
                "wrong_glyphs", "missing_text", "duplicated_text", "column_order",
                "cross_page_continuation", "paragraph_boundaries", "punctuation",
                "proper_noun_consistency", "numeric_consistency",
            ],
            "uncertainty_rule": "Do not invent text. Preserve the most strongly supported candidate when evidence is insufficient.",
        },
        "files": {
            "framework_epub": framework_name,
            "instructions": guide_name,
            "final_epub_name": final_name,
        },
        "book": copy.deepcopy(repair_map.get("book") or package.get("book") or {}),
        "mode": str(repair_map.get("mode", "standard") or "standard"),
        "model_sources": model_sources,
        "statistics": {
            "chapter_count": len(chapters),
            "item_count": len(items),
            "high_risk_count": sum(1 for item in items if item.get("risk_level") == "high"),
            "medium_risk_count": sum(1 for item in items if item.get("risk_level") == "medium"),
            "low_risk_count": sum(1 for item in items if item.get("risk_level") == "low"),
            "no_risk_count": sum(1 for item in items if item.get("risk_level") == "none"),
            "image_count": int(repair_map.get("image_count", 0) or 0),
            "model_count": len(model_sources),
        },
        "integrity": {
            "structure_sha256": str(repair_map.get("structure_sha256", "") or ""),
            "layout_sha256": str(repair_map.get("layout_sha256", "") or ""),
            "baseline_book_sha256": str(repair_map.get("baseline_book_sha256", "") or ""),
            "full_reading_text_sha256": _text_sha256(all_text),
        },
        "risk_queue": risk_queue,
        "boundary_windows": _publication_boundary_windows(raw_items),
        "page_boundary_windows": _publication_page_boundary_windows(raw_items),
        "chapter_candidates": copy.deepcopy(repair_map.get("chapter_candidates") or []),
        "chapters": chapters,
        "final_quality_checklist": [
            "all_chapters_and_items_reviewed_in_reading_order",
            "no_missing_or_duplicated_paragraphs",
            "quotes_and_brackets_balanced",
            "cross_item_boundaries_read_naturally",
            "proper_nouns_numbers_and_levels_consistent",
            "no_new_chinese_or_abnormal_latin_characters",
            "cover_illustrations_toc_spine_css_and_metadata_preserved",
            "all_xhtml_xml_files_parse_successfully",
            "epub_mimetype_is_first_and_uncompressed",
            "ai_work_metadata_removed_from_final_epub",
        ],
    }


def _safe_publication_filename(value: str, default: str = "novel") -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "_", str(value or "")).strip(" ._")
    return (cleaned or default)[:120]


def _write_bundle_zip(folder: Path, zip_path: Path) -> None:
    """Write atomically with type-aware compression.

    PNG/JPEG/WebP/EPUB are already compressed and are stored directly; text
    evidence uses maximum Deflate.  This avoids wasting CPU while preserving
    every byte and every file.
    """
    temp = zip_path.with_suffix(zip_path.suffix + ".tmp")
    media_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".epub", ".zip"}
    with zipfile.ZipFile(temp, "w") as archive:
        for path in sorted(folder.rglob("*"), key=lambda value: value.relative_to(folder).as_posix()):
            if not path.is_file():
                continue
            relative = path.relative_to(folder).as_posix()
            if path.suffix.lower() in media_suffixes:
                archive.write(path, arcname=relative, compress_type=zipfile.ZIP_STORED)
            else:
                archive.write(path, arcname=relative, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    os.replace(temp, zip_path)


def _export_ai_publication_bundle_legacy(
    primary_doc: UnifiedDocument,
    package: dict,
    output_directory: str | Path,
    *,
    mode: str = "standard",
    vertical: bool = True,
    css_template: str = "denki",
    custom_css: str | None = None,
    bundle_name: str | None = None,
    create_zip: bool = True,
) -> dict:
    """Export an upload-ready JSON + framework EPUB package for direct AI publication repair."""
    mode = str(mode or "standard").strip().lower()
    if mode not in _ALLOWED_MODES:
        raise AIRepairEpubError(f"不支持的 AI 修复证据模式：{mode}")
    root = Path(output_directory).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    raw_title = str(
        bundle_name
        or getattr(getattr(primary_doc, "metadata", None), "title", "")
        or (package.get("book") or {}).get("title", "")
        or "novel"
    )
    base = _safe_publication_filename(raw_title)
    folder = root / f"{base}_AI修复包"
    if folder.exists():
        suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
        folder = root / f"{base}_AI修复包_{suffix}"
    folder.mkdir(parents=True, exist_ok=False)

    framework_name = f"{base}_AI出版修复框架.epub"
    fusion_name = f"{base}_AI融合证据.json"
    guide_name = f"{base}_交给大模型的说明.md"
    final_name = f"{base}_AI精校出版版.epub"
    framework_path = folder / framework_name
    guide_text = _publication_guide_text(mode, framework_name=framework_name, final_name=final_name)

    try:
        epub_report = export_ai_repair_epub(
            primary_doc,
            package,
            framework_path,
            mode=mode,
            vertical=vertical,
            css_template=css_template,
            custom_css=custom_css,
            workflow="publication",
            publication_guide=guide_text,
            publication_final_name=final_name,
        )
        repair_map = read_ai_repair_map(framework_path)
        fusion_payload = _publication_fusion_payload(
            repair_map,
            package,
            framework_name=framework_name,
            final_name=final_name,
            guide_name=guide_name,
        )
        fusion_bytes = json.dumps(fusion_payload, ensure_ascii=False, indent=2).encode("utf-8")
        fusion_path = folder / fusion_name
        fusion_path.write_bytes(fusion_bytes)
        guide_path = folder / guide_name
        guide_path.write_text(guide_text, encoding="utf-8")

        publication_manifest = {
            "schema": AI_PUBLICATION_MANIFEST_SCHEMA,
            "purpose": "direct_ai_publication_repair",
            "return_to_formatter": False,
            "framework_epub": framework_name,
            "fusion_evidence_json": fusion_name,
            "model_instructions": guide_name,
            "required_final_output": final_name,
            "mode": mode,
            "item_count": int(epub_report.get("editable_count", 0) or 0),
            "chapter_count": int(epub_report.get("chapter_count", 0) or 0),
            "image_count": int(epub_report.get("image_count", 0) or 0),
            "embedded_payload_sha256": {
                fusion_name: _sha256_bytes(fusion_bytes),
                guide_name: _sha256_bytes(guide_path.read_bytes()),
            },
        }
        manifest_bytes = json.dumps(publication_manifest, ensure_ascii=False, indent=2).encode("utf-8")
        _rewrite_archive_with_metadata(framework_path, {
            AI_PUBLICATION_MANIFEST_PATH: manifest_bytes,
            AI_PUBLICATION_FUSION_PATH: fusion_bytes,
            AI_PUBLICATION_GUIDE_PATH: guide_text.encode("utf-8"),
        })
        with _validated_epub(framework_path) as archive:
            if archive.testzip() is not None:
                raise AIRepairEpubError("出版修复框架 EPUB CRC 完整性检查失败。")

        zip_path = folder.with_suffix(".zip")
        if create_zip:
            _write_bundle_zip(folder, zip_path)
        else:
            zip_path = Path("")
        return {
            "folder": str(folder),
            "zip_path": str(zip_path) if create_zip else "",
            "framework_epub": str(framework_path),
            "fusion_json": str(fusion_path),
            "guide": str(guide_path),
            "final_output_name": final_name,
            "mode": mode,
            "editable_count": int(epub_report.get("editable_count", 0) or 0),
            "chapter_count": int(epub_report.get("chapter_count", 0) or 0),
            "image_count": int(epub_report.get("image_count", 0) or 0),
            "high_risk_count": int(fusion_payload["statistics"]["high_risk_count"]),
            "medium_risk_count": int(fusion_payload["statistics"]["medium_risk_count"]),
            "boundary_window_count": len(fusion_payload["boundary_windows"]),
            "framework_sha256": _sha256_bytes(framework_path.read_bytes()),
            "fusion_sha256": _sha256_bytes(fusion_bytes),
        }
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        zip_candidate = folder.with_suffix(".zip")
        if zip_candidate.exists():
            zip_candidate.unlink(missing_ok=True)
        raise


def export_ai_publication_bundle(
    primary_doc: UnifiedDocument,
    package: dict,
    output_directory: str | Path,
    *,
    mode: str = "one_pass",
    vertical: bool = True,
    css_template: str = "denki",
    custom_css: str | None = None,
    bundle_name: str | None = None,
    create_zip: bool = True,
    include_publication_reference: bool = False,
    publication_reference_path: str | Path | None = None,
    package_mode: str = "forensic",
) -> dict:
    """Export an AI repair package. V4 adjudicates only real disagreements."""
    normalized_package_mode = str(package_mode or "forensic").strip().lower()
    if normalized_package_mode in {"disagreement_v4", "v4", "adjudication", "conflict_only"}:
        from engine.ai_disagreement_package_v4 import export_ai_disagreement_package_v4
        return export_ai_disagreement_package_v4(
            primary_doc,
            package,
            output_directory,
            vertical=vertical,
            css_template=css_template,
            custom_css=custom_css,
            bundle_name=bundle_name,
            create_zip=create_zip,
            include_publication_reference=include_publication_reference,
            publication_reference_path=publication_reference_path,
        )
    from engine.ai_publication_bundle_v2 import export_ai_publication_bundle_v2
    return export_ai_publication_bundle_v2(
        primary_doc,
        package,
        output_directory,
        mode=mode,
        vertical=vertical,
        css_template=css_template,
        custom_css=custom_css,
        bundle_name=bundle_name,
        create_zip=create_zip,
        include_publication_reference=include_publication_reference,
        publication_reference_path=publication_reference_path,
        package_mode=package_mode,
    )

def _result_schema() -> dict:
    update_properties = {
        "item_id": {"type": "string", "minLength": 1},
        "row_id": {"type": "string", "minLength": 1},
        "baseline_text": {"type": "string"},
        "expected_baseline_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "edited_text": {"type": "string"},
        "edited_tokens": {"type": "array"},
        "delete_intentionally": {"type": "boolean"},
        "confidence": {"type": ["string", "number"]},
        "reason": {"type": "string"},
        "evidence": {"type": "array"},
        "needs_review": {"type": "boolean"},
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "novel_formatter.ai_repair_edits.v2",
        "type": "object",
        "required": ["schema", "structure_sha256"],
        "additionalProperties": False,
        "properties": {
            "schema": {"const": AI_REPAIR_EDITS_SCHEMA},
            "package_id": {"type": "string"},
            "structure_sha256": {"type": "string"},
            "map_sha256": {"type": "string"},
            "baseline_book_sha256": {"type": "string"},
            "export_revision": {"type": "integer"},
            "updates": {
                "type": "array",
                "items": {"type": "object", "additionalProperties": False, "properties": update_properties},
            },
            "transactions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["transaction_id", "operation", "item_ids", "updates"],
                    "additionalProperties": False,
                    "properties": {
                        "transaction_id": {"type": "string", "minLength": 1},
                        "operation": {"const": "rebalance_adjacent_items"},
                        "item_ids": {"type": "array", "minItems": 2, "items": {"type": "string"}},
                        "updates": {"type": "array", "minItems": 2, "items": {"type": "object", "additionalProperties": False, "properties": update_properties}},
                        "reason": {"type": "string"},
                        "confidence": {"type": ["string", "number"]},
                        "needs_review": {"type": "boolean"},
                    },
                },
            },
        },
    }


def _rewrite_archive_with_metadata(epub_path: Path, additions: dict[str, bytes]) -> None:
    temp_path = epub_path.with_suffix(epub_path.suffix + ".tmp")
    skip = set(additions) | {"mimetype"}
    with zipfile.ZipFile(epub_path, "r") as source, zipfile.ZipFile(temp_path, "w") as target:
        mime_info = zipfile.ZipInfo("mimetype")
        mime_info.compress_type = zipfile.ZIP_STORED
        target.writestr(mime_info, b"application/epub+zip")
        for info in source.infolist():
            if info.filename in skip:
                continue
            target.writestr(info, source.read(info.filename))
        for name, data in additions.items():
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            target.writestr(info, data)
    os.replace(temp_path, epub_path)


def _chapter_payloads(items: Sequence[dict]) -> tuple[list[dict], dict[str, bytes]]:
    groups: dict[str, list[dict]] = {}
    titles: dict[str, str] = {}
    for item in items:
        chapter_id = str(item.get("chapter_id", "front-matter") or "front-matter")
        groups.setdefault(chapter_id, []).append(item)
        titles[chapter_id] = str(item.get("chapter_title", "") or "")
    index: list[dict] = []
    files: dict[str, bytes] = {}
    for number, (chapter_id, chapter_items) in enumerate(groups.items(), start=1):
        filename = f"{CHAPTERS_DIR}/chapter-{number:03d}.json"
        payload = {
            "schema": "novel_formatter.ai_repair_chapter.v1",
            "chapter_id": chapter_id,
            "chapter_title": titles.get(chapter_id, ""),
            "item_count": len(chapter_items),
            "items": chapter_items,
        }
        payload["chapter_sha256"] = _sha256({key: value for key, value in payload.items() if key != "chapter_sha256"})
        raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        files[filename] = raw
        index.append({
            "chapter_id": chapter_id,
            "chapter_title": titles.get(chapter_id, ""),
            "path": filename,
            "item_count": len(chapter_items),
            "sha256": _sha256_bytes(raw),
        })
    return index, files


def export_ai_repair_epub(
    primary_doc: UnifiedDocument,
    package: dict,
    output_path: str | Path,
    *,
    mode: str = "standard",
    vertical: bool = False,
    css_template: str = "denki",
    custom_css: str | None = None,
    workflow: str = "exchange",
    publication_guide: str | None = None,
    publication_final_name: str = "AI精校出版版.epub",
    publication_image_paths: set[str] | None = None,
) -> dict:
    """Export a complete EPUB with chapter-split OCR evidence.

    ``workflow=publication`` turns the EPUB into a direct final-book framework
    for an external model; it does not advertise any Formatter write-back step.
    """
    mode = str(mode or "standard").strip().lower()
    if mode not in _ALLOWED_MODES:
        raise AIRepairEpubError(f"不支持的 AI 修复证据模式：{mode}")
    workflow = str(workflow or "exchange").strip().lower()
    if workflow not in _ALLOWED_WORKFLOWS:
        raise AIRepairEpubError(f"不支持的 AI 修复工作流：{workflow}")
    output = Path(output_path).expanduser()
    if output.suffix.lower() != ".epub":
        output = output.with_suffix(".epub")
    output.parent.mkdir(parents=True, exist_ok=True)

    export_revision = int(package.get("export_revision", 0) or 0)
    if export_revision <= 0:
        export_revision = int(time.time_ns() // 1_000_000)
    exported_at = datetime.now(timezone.utc).isoformat()

    repair_doc = build_repair_document(primary_doc, package, export_revision=export_revision)
    if publication_image_paths is not None:
        repair_doc = _retain_only_publication_images(repair_doc, publication_image_paths)
    build_epub(
        repair_doc,
        str(output),
        css_template=css_template,
        vertical=vertical,
        verbose=False,
        custom_css=custom_css,
        preserve_image_bytes=True,
    )

    with _validated_epub(output) as archive:
        tagged = _scan_tagged_elements(archive)
        expected_items = list(package.get("editable_items") or [])
        expected_ids = [str(item.get("row_id") or item.get("item_id") or "") for item in expected_items]
        missing = [item_id for item_id in expected_ids if item_id not in tagged]
        extra = [item_id for item_id in tagged if item_id not in set(expected_ids)]
        if missing or extra:
            raise AIRepairEpubError(f"稳定段落 ID 映射不完整：缺少 {len(missing)}，额外 {len(extra)}。")
        spine = _spine_order(archive)
        mapped_items = []
        for item in expected_items:
            item_id = str(item.get("row_id") or item.get("item_id") or "")
            archive_path, html_id, _element = tagged[item_id]
            mapped_items.append(_map_item(
                item,
                target=f"{archive_path}#{html_id}",
                html_id=html_id,
                mode=mode,
                primary_doc=primary_doc,
                export_revision=export_revision,
            ))
        chapter_candidates = _detect_chapter_candidates(primary_doc, mapped_items)
        _enrich_reading_context(mapped_items, spine, chapter_candidates=chapter_candidates)
        if mode == "one_pass":
            _apply_one_pass_context_evidence(mapped_items, expected_items, primary_doc)
        images = _archive_image_manifest(archive)
        xhtml_paths = [name for name in archive.namelist() if name.lower().endswith(".xhtml")]

    baseline_book_hash = _baseline_book_sha256(mapped_items)
    map_core = {
        "schema": AI_REPAIR_SCHEMA,
        "package_id": str(package.get("package_id", "") or ""),
        "structure_sha256": str(package.get("structure_sha256", "") or ""),
        "layout_sha256": str(package.get("layout_sha256", "") or ""),
        "baseline_book_sha256": baseline_book_hash,
        "export_revision": export_revision,
        "exported_at": exported_at,
        "mode": mode,
        "book": copy.deepcopy(package.get("book") or {}),
        "editable_count": len(mapped_items),
        "xhtml_count": len(xhtml_paths),
        "image_count": len(images),
        "images": images,
        "items": mapped_items,
        "chapter_candidates": chapter_candidates,
    }
    map_core["items_sha256"] = _sha256(mapped_items)
    map_core["map_sha256"] = _sha256({key: value for key, value in map_core.items() if key != "map_sha256"})

    template = {
        "schema": AI_REPAIR_EDITS_SCHEMA,
        "package_id": map_core["package_id"],
        "structure_sha256": map_core["structure_sha256"],
        "map_sha256": map_core["map_sha256"],
        "baseline_book_sha256": baseline_book_hash,
        "export_revision": export_revision,
        "updates": [],
        "transactions": [],
    }
    chapter_index, chapter_files = _chapter_payloads(mapped_items)
    task_name = (
        "repair_japanese_ocr_and_publish_final_epub"
        if workflow == "publication"
        else "repair_japanese_ocr_without_rewriting_style"
    )
    tasks_lines = "\n".join(json.dumps({
        "task": task_name,
        **item,
    }, ensure_ascii=False, separators=(",", ":")) for item in mapped_items) + "\n"
    manifest_core = {
        "schema": AI_REPAIR_SCHEMA,
        "package_id": map_core["package_id"],
        "structure_sha256": map_core["structure_sha256"],
        "layout_sha256": map_core["layout_sha256"],
        "map_sha256": map_core["map_sha256"],
        "baseline_book_sha256": baseline_book_hash,
        "export_revision": export_revision,
        "exported_at": exported_at,
        "mode": mode,
        "workflow": workflow,
        "return_to_formatter": workflow != "publication",
        "preferred_writeback": "direct_final_epub" if workflow == "publication" else "sparse_json",
        "compatibility_writeback": "none" if workflow == "publication" else "edited_epub",
        "required_final_output": publication_final_name if workflow == "publication" else "",
        "paths": {
            "legacy_map": MAP_PATH,
            "tasks_jsonl": TASKS_PATH,
            "guide": V2_GUIDE_PATH,
            **({} if workflow == "publication" else {
                "result_schema": RESULT_SCHEMA_PATH,
                "result_template": V2_TEMPLATE_PATH,
            }),
        },
        "editable_count": len(mapped_items),
        "chapter_count": len(chapter_index),
        "chapters": chapter_index,
        "chapter_candidates": chapter_candidates,
        "image_count": len(images),
        "images": images,
    }
    manifest_core["manifest_sha256"] = _sha256({key: value for key, value in manifest_core.items() if key != "manifest_sha256"})

    if workflow == "publication":
        guide_text = publication_guide or _publication_guide_text(
            mode,
            framework_name=output.name,
            final_name=publication_final_name,
        )
    else:
        guide_text = _guide_text(mode)
    guide = guide_text.encode("utf-8")
    template_bytes = json.dumps(template, ensure_ascii=False, indent=2).encode("utf-8")
    additions = {
        MAP_PATH: json.dumps(map_core, ensure_ascii=False, indent=2).encode("utf-8"),
        GUIDE_PATH: guide,
        MANIFEST_PATH: json.dumps(manifest_core, ensure_ascii=False, indent=2).encode("utf-8"),
        V2_GUIDE_PATH: guide,
        TASKS_PATH: tasks_lines.encode("utf-8"),
        **chapter_files,
    }
    if workflow != "publication":
        additions.update({
            TEMPLATE_PATH: template_bytes,
            V2_TEMPLATE_PATH: template_bytes,
            RESULT_SCHEMA_PATH: json.dumps(_result_schema(), ensure_ascii=False, indent=2).encode("utf-8"),
        })
    _rewrite_archive_with_metadata(output, additions)

    with _validated_epub(output) as archive:
        if archive.namelist()[0] != "mimetype":
            raise AIRepairEpubError("EPUB 重封装后 mimetype 不是首项。")
        if archive.getinfo("mimetype").compress_type != zipfile.ZIP_STORED:
            raise AIRepairEpubError("EPUB mimetype 必须不压缩存储。")
        if archive.testzip() is not None:
            raise AIRepairEpubError("EPUB CRC 完整性检查失败。")
        loaded = _json_loads_strict(archive.read(MAP_PATH), source=MAP_PATH)
        if loaded.get("map_sha256") != map_core["map_sha256"]:
            raise AIRepairEpubError("AI 修复映射写入后校验失败。")

    return {
        "path": str(output),
        "mode": mode,
        "editable_count": len(mapped_items),
        "image_count": len(images),
        "xhtml_count": len(xhtml_paths),
        "chapter_count": len(chapter_index),
        "map_sha256": map_core["map_sha256"],
        "baseline_book_sha256": baseline_book_hash,
        "export_revision": export_revision,
        "package_id": map_core["package_id"],
        "workflow": workflow,
        "preferred_writeback": "direct_final_epub" if workflow == "publication" else "sparse_json",
        "required_final_output": publication_final_name if workflow == "publication" else "",
    }


def read_ai_repair_map(epub_path: str | Path) -> dict:
    path = Path(epub_path).expanduser()
    try:
        with _validated_epub(path) as archive:
            value = _json_loads_strict(archive.read(MAP_PATH), source=MAP_PATH)
    except KeyError as exc:
        raise AIRepairEpubError(f"该 EPUB 不包含 {MAP_PATH}。") from exc
    if value.get("schema") not in _LEGACY_AI_REPAIR_SCHEMAS:
        raise AIRepairEpubError(f"不支持的 AI 修复 EPUB schema：{value.get('schema')!r}")
    expected_hash = _sha256({key: item for key, item in value.items() if key != "map_sha256"})
    if str(value.get("map_sha256", "")) != expected_hash:
        raise AIRepairEpubError("AI 修复映射已被修改或损坏。")
    return value


def _element_text(element: ET.Element) -> str:
    """Convert tagged XHTML back to controlled plain/ruby text."""
    if _local(element.tag).lower() == "ruby":
        base_parts = [element.text or ""]
        readings: list[str] = []
        for child in list(element):
            if _local(child.tag).lower() == "rt":
                readings.append("".join(child.itertext()))
            else:
                base_parts.append(_element_text(child))
            if child.tail:
                base_parts.append(child.tail)
        base = "".join(base_parts)
        reading = "".join(readings)
        return f"{base}|{reading}" if reading else base
    parts = [element.text or ""]
    for child in list(element):
        if _local(child.tag).lower() == "br":
            parts.append("\n")
        else:
            parts.append(_element_text(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def _validate_tagged_inline(element: ET.Element, *, item_id: str, content_format: str) -> None:
    """Reject arbitrary external HTML inside editable nodes."""
    allowed = {"br"}
    if content_format == "inline_tokens_v1":
        allowed.update({"ruby", "rt", "rp"})
    for child in element.iter():
        if child is element:
            continue
        local = _local(child.tag).lower()
        if local not in allowed:
            raise AIRepairEpubError(
                f"条目 {item_id} 含不受控内联 XHTML <{local}>；请改用稀疏 JSON 或受控 Ruby token。"
            )
        if local in {"br", "rt", "rp", "ruby"}:
            unsafe_attrs = [key for key in child.attrib if _local(key).lower() not in {"class"}]
            if unsafe_attrs:
                raise AIRepairEpubError(f"条目 {item_id} 的内联元素包含未知属性。")


def _diff_changes(base: str, other: str, source: str) -> list[dict]:
    changes: list[dict] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, base, other, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        changes.append({"start": i1, "end": i2, "replacement": other[j1:j2], "source": source, "tag": tag})
    return changes


def _changes_conflict(left: dict, right: dict) -> bool:
    if left["start"] == left["end"] and right["start"] == right["end"]:
        return left["start"] == right["start"] and left["replacement"] != right["replacement"]
    if left["start"] == left["end"]:
        return right["start"] <= left["start"] <= right["end"]
    if right["start"] == right["end"]:
        return left["start"] <= right["start"] <= left["end"]
    return max(left["start"], right["start"]) < min(left["end"], right["end"])


def _three_way_merge(baseline: str, local: str, remote: str) -> tuple[str | None, list[dict]]:
    local_changes = _diff_changes(baseline, local, "local")
    remote_changes = _diff_changes(baseline, remote, "remote")
    conflicts: list[dict] = []
    for local_change in local_changes:
        for remote_change in remote_changes:
            if _changes_conflict(local_change, remote_change):
                if (local_change["start"], local_change["end"], local_change["replacement"]) == (
                    remote_change["start"], remote_change["end"], remote_change["replacement"]
                ):
                    continue
                conflicts.append({"local": local_change, "remote": remote_change})
    if conflicts:
        return None, conflicts
    combined: dict[tuple[int, int, str], dict] = {}
    for change in local_changes + remote_changes:
        combined[(change["start"], change["end"], change["replacement"])] = change
    merged = baseline
    for change in sorted(combined.values(), key=lambda value: (value["start"], value["end"]), reverse=True):
        merged = merged[:change["start"]] + change["replacement"] + merged[change["end"]:]
    return merged, []


def _common_overlap(left: str, right: str, *, minimum: int = 6, maximum: int = 48) -> int:
    limit = min(len(left), len(right), maximum)
    for size in range(limit, minimum - 1, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


def _candidate_evidence_blob(item: dict) -> str:
    parts = [str(item.get("original_fused_text", "") or ""), str(item.get("edited_text", "") or "")]
    for candidate in item.get("candidates") or []:
        if isinstance(candidate, dict):
            parts.append(str(candidate.get("text", "") or ""))
    return "\n".join(parts)


def _audit_text_change(
    current: str,
    edited: str,
    *,
    item: dict,
    previous_text: str = "",
    next_text: str = "",
    transaction_id: str = "",
    baseline_status: str = "matched",
) -> dict:
    flags: list[str] = []
    current = str(current or "")
    edited = str(edited or "")
    if current:
        ratio = abs(len(edited) - len(current)) / max(1, len(current))
        if ratio > 0.20:
            flags.append("length_change_over_20_percent")
    matcher = SequenceMatcher(None, current, edited, autojunk=False)
    inserted_fragments: list[str] = []
    deleted_fragments: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in {"delete", "replace"} and i2 > i1:
            fragment = current[i1:i2]
            deleted_fragments.append(fragment)
            if len(fragment) > 15:
                flags.append("deleted_more_than_15_continuous_chars")
        if tag in {"insert", "replace"} and j2 > j1:
            inserted_fragments.append(edited[j1:j2])
    evidence_blob = _candidate_evidence_blob(item)
    if any(len(fragment.strip()) >= 12 and fragment.strip() not in evidence_blob for fragment in inserted_fragments):
        flags.append("large_new_text_absent_from_ocr_evidence")
    if any(char in _SIMPLIFIED_ONLY_CHARS and char not in current for fragment in inserted_fragments for char in fragment):
        flags.append("new_simplified_chinese_character")
    if any(re.search(r"[A-Za-z0-9]{4,}", fragment) and fragment not in evidence_blob for fragment in inserted_fragments):
        flags.append("new_abnormal_latin_or_numeric_sequence")
    if any(token in fragment for fragment in deleted_fragments for token in _NEGATION_TOKENS):
        flags.append("negation_removed")
    current_numbers = re.findall(r"\d+(?:[.,]\d+)?", current)
    edited_numbers = re.findall(r"\d+(?:[.,]\d+)?", edited)
    if current_numbers != edited_numbers:
        flags.append("number_or_level_changed")
    current_katakana = re.findall(r"[ァ-ヶー]{3,}", current)
    edited_katakana = re.findall(r"[ァ-ヶー]{3,}", edited)
    if current_katakana != edited_katakana:
        flags.append("name_or_skill_like_token_changed")
    for left, right in (("「", "」"), ("『", "』"), ("（", "）"), ("【", "】")):
        if (current.count(left), current.count(right)) != (edited.count(left), edited.count(right)):
            flags.append("quote_count_changed")
            break
    if (current.count("!"), current.count("?"), current.count("！"), current.count("？")) != (
        edited.count("!"), edited.count("?"), edited.count("！"), edited.count("？")
    ):
        flags.append("exclamation_question_punctuation_changed")
    if current.translate(_SMALL_KANA_TABLE) == edited.translate(_SMALL_KANA_TABLE) and current != edited:
        flags.append("small_kana_changed")
    if [char for char in current if char in _DASH_CHARS] != [char for char in edited if char in _DASH_CHARS]:
        flags.append("dash_character_changed")
    if previous_text and _common_overlap(previous_text, edited) >= 8:
        flags.append("new_duplicate_with_previous_item")
    if next_text and _common_overlap(edited, next_text) >= 8:
        flags.append("new_duplicate_with_next_item")
    terminal = set("。！？!?…」』）】〉》")
    if next_text and bool(current and current[-1] in terminal) != bool(edited and edited[-1] in terminal):
        flags.append("next_item_continuity_changed")
    if baseline_status in {"stale_conflict", "baseline_mismatch_no_text"}:
        flags.append("stale_baseline_conflict")
    if baseline_status == "auto_merged_non_overlapping":
        flags.append("three_way_merge_requires_confirmation")
    flags = list(dict.fromkeys(flags))
    severe = {
        "deleted_more_than_15_continuous_chars",
        "large_new_text_absent_from_ocr_evidence",
        "new_simplified_chinese_character",
        "negation_removed",
        "number_or_level_changed",
        "stale_baseline_conflict",
    }
    if any(flag in severe for flag in flags):
        level = "high_risk"
    elif transaction_id:
        level = "cross_item_review"
    elif flags:
        level = "semantic_review"
    else:
        level = "high_confidence"
    return {"audit_level": level, "audit_flags": flags, "needs_review": level != "high_confidence"}


def _current_item_maps(expected_package: dict) -> tuple[dict[str, dict], dict[str, int], list[dict]]:
    items = list(expected_package.get("editable_items") or [])
    item_map: dict[str, dict] = {}
    row_map: dict[str, int] = {}
    for position, item in enumerate(items):
        item_id = str(item.get("row_id") or item.get("item_id") or "")
        if not item_id or item_id in item_map:
            raise AIRepairEpubError("当前融合稿稳定 ID 缺失或重复。")
        item_map[item_id] = item
        row_map[item_id] = int(item.get("row_index", position) or position)
    return item_map, row_map, items


def _current_text(item: dict) -> str:
    return _normalise_plain_text(item.get("edited_text", item.get("original_fused_text", "")))


def _resolve_update_against_current(
    *,
    item_id: str,
    current_item: dict,
    edited_text: str,
    delete_intentionally: bool,
    expected_baseline_sha256: str = "",
    baseline_text: str | None = None,
    confidence: Any = "",
    reason: str = "",
    evidence: Any = None,
    needs_review: bool = False,
    previous_text: str = "",
    next_text: str = "",
    transaction_id: str = "",
    transaction_operation: str = "",
    transaction_member_ids: Sequence[str] = (),
) -> dict:
    current = _current_text(current_item)
    edited = _normalise_plain_text(edited_text)
    expected_hash = str(expected_baseline_sha256 or "").strip().lower()
    baseline_value = _normalise_plain_text(baseline_text) if baseline_text is not None else None
    if baseline_value is not None:
        baseline_hash = _text_sha256(baseline_value)
        if expected_hash and baseline_hash != expected_hash:
            raise AIRepairEpubError(f"条目 {item_id} 的 baseline_text 与 expected_baseline_sha256 不一致。")
        expected_hash = expected_hash or baseline_hash

    baseline_status = "unverified_legacy"
    original_ai_text = edited
    conflict_details: list[dict] = []
    if expected_hash:
        if _text_sha256(current) == expected_hash:
            baseline_status = "matched"
        elif baseline_value is None:
            baseline_status = "baseline_mismatch_no_text"
            needs_review = True
        else:
            merged, conflicts = _three_way_merge(baseline_value, current, edited)
            if merged is None:
                baseline_status = "stale_conflict"
                conflict_details = conflicts
                needs_review = True
            else:
                baseline_status = "auto_merged_non_overlapping"
                edited = merged
                needs_review = True
    audit = _audit_text_change(
        current,
        edited,
        item=current_item,
        previous_text=previous_text,
        next_text=next_text,
        transaction_id=transaction_id,
        baseline_status=baseline_status,
    )
    needs_review = bool(needs_review or audit["needs_review"] or transaction_id)
    update = {
        "item_id": item_id,
        "edited_text": edited,
        "delete_intentionally": bool(delete_intentionally),
        "confidence": confidence,
        "reason": str(reason or ""),
        "evidence": copy.deepcopy(evidence or []),
        "needs_review": needs_review,
        "expected_baseline_sha256": expected_hash,
        "current_text_sha256": _text_sha256(current),
        "baseline_status": baseline_status,
        "audit_level": audit["audit_level"],
        "audit_flags": audit["audit_flags"],
    }
    if baseline_value is not None:
        update["baseline_text"] = baseline_value
    if edited != original_ai_text:
        update["original_ai_text"] = original_ai_text
    if conflict_details:
        update["conflict_details"] = conflict_details
        update["result_expired"] = True
    if transaction_id:
        update.update({
            "transaction_id": transaction_id,
            "transaction_operation": transaction_operation,
            "transaction_member_ids": list(transaction_member_ids),
        })
    return update


def extract_edits_from_repaired_epub(
    epub_path: str | Path,
    *,
    expected_package: dict | None = None,
) -> tuple[list[dict], dict]:
    path = Path(epub_path).expanduser()
    repair_map = read_ai_repair_map(path)
    if expected_package is not None:
        expected_structure = str(expected_package.get("structure_sha256", "") or "")
        if expected_structure and repair_map.get("structure_sha256") != expected_structure:
            raise AIRepairEpubError("修复 EPUB 不属于当前书：结构哈希不一致。")
        current_map, _row_map, current_items = _current_item_maps(expected_package)
        expected_ids = set(current_map)
    else:
        current_items = []
        current_map = {}
        expected_ids = {str(item.get("item_id", "") or "") for item in repair_map.get("items", [])}

    map_items = {str(item.get("item_id", "") or ""): item for item in repair_map.get("items", [])}
    if not expected_ids or set(map_items) != expected_ids:
        raise AIRepairEpubError("修复 EPUB 的稳定 ID 集合与当前融合稿不一致。")

    with _validated_epub(path) as archive:
        tagged = _scan_tagged_elements(archive)
        if set(tagged) != expected_ids:
            missing = expected_ids - set(tagged)
            extra = set(tagged) - expected_ids
            raise AIRepairEpubError(f"修复 EPUB 的段落结构被改变：缺少 {len(missing)}，额外 {len(extra)}。")
        current_images = {item["path"]: item for item in _archive_image_manifest(archive)}
        for expected in repair_map.get("images", []):
            path_name = str(expected.get("path", "") or "")
            current = current_images.get(path_name)
            if current is None or current.get("sha256") != expected.get("sha256"):
                raise AIRepairEpubError(f"插图或封面被修改/缺失：{path_name}")

        sorted_ids = sorted(expected_ids, key=lambda value: int(map_items[value].get("row_index", 0) or 0))
        updates: list[dict] = []
        for position, item_id in enumerate(sorted_ids):
            archive_name, html_id, element = tagged[item_id]
            map_item = map_items[item_id]
            expected_target = str(map_item.get("epub_target", "") or "")
            actual_target = f"{archive_name}#{html_id}"
            if expected_target and actual_target != expected_target:
                raise AIRepairEpubError(f"条目 {item_id} 被移动或锚点改变：{actual_target}")
            _validate_tagged_inline(
                element,
                item_id=item_id,
                content_format=str(map_item.get("content_format", "plain_text_with_newlines_v1") or "plain_text_with_newlines_v1"),
            )
            baseline = _normalise_plain_text(map_item.get("baseline_text", map_item.get("original_text", "")))
            baseline_hash = str(map_item.get("baseline_text_sha256", "") or _text_sha256(baseline))
            edited = _normalise_plain_text(_element_text(element))
            delete_intentionally = _explicit_bool(
                element.attrib.get("data-delete-intentionally", "false"),
                field=f"条目 {item_id} 的 data-delete-intentionally",
            )
            if not edited and baseline and not delete_intentionally:
                raise AIRepairEpubError(f"条目 {item_id} 被清空但没有 data-delete-intentionally=true。")
            if edited == baseline and delete_intentionally == bool(map_item.get("delete_intentionally", False)):
                continue
            if expected_package is None:
                current_item = {"edited_text": baseline, "candidates": map_item.get("candidates") or []}
                previous_text = str(map_items[sorted_ids[position - 1]].get("baseline_text", "") or "") if position > 0 else ""
                next_text = str(map_items[sorted_ids[position + 1]].get("baseline_text", "") or "") if position + 1 < len(sorted_ids) else ""
            else:
                current_item = current_map[item_id]
                previous_text = _current_text(current_map[sorted_ids[position - 1]]) if position > 0 else ""
                next_text = _current_text(current_map[sorted_ids[position + 1]]) if position + 1 < len(sorted_ids) else ""
            updates.append(_resolve_update_against_current(
                item_id=item_id,
                current_item=current_item,
                edited_text=edited,
                delete_intentionally=delete_intentionally,
                expected_baseline_sha256=baseline_hash,
                baseline_text=baseline,
                confidence="external_epub_edit",
                reason="从稳定段落 ID 的修复 EPUB 导入",
                previous_text=previous_text,
                next_text=next_text,
            ))
    return updates, _import_report(
        source=str(path),
        source_type="epub",
        total_items=len(expected_ids),
        updates=updates,
        metadata=repair_map,
    )


def _normalise_update_payload(payload: Any) -> tuple[list[dict], dict, list[dict]]:
    metadata: dict = {}
    transactions: list[dict] = []
    if isinstance(payload, list):
        return payload, metadata, transactions
    if not isinstance(payload, dict):
        raise AIRepairEpubError("AI 修复结果必须是 JSON 对象或数组。")
    structural_keys = {
        "schema", "package_id", "structure_sha256", "map_sha256",
        "baseline_book_sha256", "export_revision", "updates", "editable_items", "transactions", "decisions",
    }
    looks_structured = any(key in payload for key in structural_keys)
    if looks_structured:
        forbidden_top = sorted(set(payload) - structural_keys)
        if forbidden_top:
            raise AIRepairEpubError(f"AI 修复 JSON 顶层包含未知字段：{', '.join(forbidden_top)}")
    metadata = {
        key: payload.get(key)
        for key in (
            "schema", "package_id", "structure_sha256", "map_sha256",
            "baseline_book_sha256", "export_revision",
        )
        if key in payload
    }
    if payload.get("transactions") is not None:
        if not isinstance(payload.get("transactions"), list):
            raise AIRepairEpubError("transactions 必须是数组。")
        transactions = list(payload.get("transactions") or [])
    if isinstance(payload.get("decisions"), list):
        converted = []
        for decision in payload.get("decisions") or []:
            if not isinstance(decision, dict):
                converted.append(decision)
                continue
            converted.append({
                "item_id": decision.get("item_id"),
                "edited_text": decision.get("selected_text", ""),
                "confidence": decision.get("confidence", ""),
                "reason": decision.get("reason_code", ""),
                "evidence": decision.get("evidence") or [],
                "needs_review": False,
                "_v4_source": decision.get("source", ""),
            })
        return converted, metadata, transactions
    if isinstance(payload.get("updates"), list):
        return list(payload["updates"]), metadata, transactions
    if isinstance(payload.get("editable_items"), list):
        return list(payload["editable_items"]), metadata, transactions
    if transactions and not any(key in payload for key in ("updates", "editable_items")):
        return [], metadata, transactions
    # Compact {item_id: edited_text} is accepted only for exact known IDs later.
    compact_exclusions = set(metadata) | {"transactions"}
    compact = {key: value for key, value in payload.items() if key not in compact_exclusions}
    if compact and all(isinstance(value, str) for value in compact.values()):
        return [{"item_id": str(item_id), "edited_text": text} for item_id, text in compact.items()], metadata, transactions
    if any(key in payload for key in ("item_id", "row_id", "edited_text", "edited_tokens")):
        return [payload], metadata, transactions
    raise AIRepairEpubError("AI 修复 JSON 中找不到 updates 或 transactions。")


def _flatten_transactions(transactions: list[dict], *, row_map: dict[str, int]) -> tuple[list[dict], int]:
    flattened: list[dict] = []
    seen_transaction_ids: set[str] = set()
    for tx_index, transaction in enumerate(transactions):
        if not isinstance(transaction, dict):
            raise AIRepairEpubError(f"第 {tx_index + 1} 个事务不是对象。")
        allowed = {"transaction_id", "operation", "item_ids", "updates", "reason", "confidence", "needs_review"}
        forbidden = sorted(set(transaction) - allowed)
        if forbidden:
            raise AIRepairEpubError(f"事务第 {tx_index + 1} 项包含越权字段：{', '.join(forbidden)}")
        transaction_id = str(transaction.get("transaction_id", "") or "").strip()
        if not transaction_id or transaction_id in seen_transaction_ids:
            raise AIRepairEpubError("原子事务 transaction_id 缺失或重复。")
        seen_transaction_ids.add(transaction_id)
        operation = str(transaction.get("operation", "") or "")
        if operation != "rebalance_adjacent_items":
            raise AIRepairEpubError(f"不支持的原子事务 operation：{operation!r}")
        item_ids = [str(value or "").strip() for value in (transaction.get("item_ids") or [])]
        updates = transaction.get("updates")
        if not isinstance(updates, list) or len(item_ids) < 2 or len(updates) != len(item_ids):
            raise AIRepairEpubError(f"事务 {transaction_id} 的 item_ids/updates 数量不一致或少于 2。")
        if len(item_ids) > 20 or len(set(item_ids)) != len(item_ids):
            raise AIRepairEpubError(f"事务 {transaction_id} 的 item_ids 重复或数量过多。")
        if any(item_id not in row_map for item_id in item_ids):
            raise AIRepairEpubError(f"事务 {transaction_id} 包含未知 ID。")
        positions = [row_map[item_id] for item_id in item_ids]
        if positions != sorted(positions) or any(right != left + 1 for left, right in zip(positions, positions[1:])):
            raise AIRepairEpubError(f"事务 {transaction_id} 只能覆盖按阅读顺序连续的条目。")
        update_ids = [str(update.get("item_id") or update.get("row_id") or "") if isinstance(update, dict) else "" for update in updates]
        if update_ids != item_ids:
            raise AIRepairEpubError(f"事务 {transaction_id} 的 updates 必须与 item_ids 同序且完整覆盖。")
        for update in updates:
            enriched = copy.deepcopy(update)
            enriched["_transaction_id"] = transaction_id
            enriched["_transaction_operation"] = operation
            enriched["_transaction_member_ids"] = list(item_ids)
            enriched["_transaction_reason"] = str(transaction.get("reason", "") or "")
            enriched["_transaction_confidence"] = transaction.get("confidence", "")
            enriched["_transaction_needs_review"] = bool(transaction.get("needs_review", True))
            flattened.append(enriched)
    return flattened, len(seen_transaction_ids)


def _import_report(*, source: str, source_type: str, total_items: int, updates: list[dict], metadata: dict, atomic_count: int = 0, package_id_mismatch: bool = False, baseline_book_mismatch: bool = False) -> dict:
    levels: dict[str, int] = {}
    baseline_statuses: dict[str, int] = {}
    for update in updates:
        level = str(update.get("audit_level", "") or "unknown")
        levels[level] = levels.get(level, 0) + 1
        status = str(update.get("baseline_status", "") or "unknown")
        baseline_statuses[status] = baseline_statuses.get(status, 0) + 1
    return {
        "source": source,
        "source_type": source_type,
        "total_items": total_items,
        "changed_items": len(updates),
        "package_id": metadata.get("package_id", ""),
        "structure_sha256": metadata.get("structure_sha256", ""),
        "baseline_book_sha256": metadata.get("baseline_book_sha256", ""),
        "export_revision": metadata.get("export_revision", 0),
        "sealed": bool(metadata.get("structure_sha256")),
        "package_id_mismatch": package_id_mismatch,
        "baseline_book_mismatch": baseline_book_mismatch,
        "atomic_transaction_count": atomic_count,
        "audit_counts": levels,
        "baseline_status_counts": baseline_statuses,
        "stale_conflicts": baseline_statuses.get("stale_conflict", 0) + baseline_statuses.get("baseline_mismatch_no_text", 0),
        "three_way_merges": baseline_statuses.get("auto_merged_non_overlapping", 0),
    }


def load_ai_repair_json(
    source: str | Path | dict | list,
    *,
    expected_package: dict,
) -> tuple[list[dict], dict]:
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        try:
            raw = path.read_bytes()
        except Exception as exc:
            raise AIRepairEpubError(f"无法读取 AI 修复 JSON：{exc}") from exc
        payload = _json_loads_strict(raw, source=str(path))
        source_name = str(path)
    else:
        payload = copy.deepcopy(source)
        _validate_json_shape(payload)
        source_name = "memory"
    raw_updates, metadata, raw_transactions = _normalise_update_payload(payload)
    is_v4_decisions = metadata.get("schema") == AI_DISAGREEMENT_DECISIONS_SCHEMA
    if metadata.get("schema") not in (None, "", *_LEGACY_AI_REPAIR_EDIT_SCHEMAS, AI_DISAGREEMENT_DECISIONS_SCHEMA):
        raise AIRepairEpubError(f"不支持的修复结果 schema：{metadata.get('schema')!r}")
    expected_structure = str(expected_package.get("structure_sha256", "") or "")
    if metadata.get("structure_sha256") and str(metadata["structure_sha256"]) != expected_structure:
        raise AIRepairEpubError("修复 JSON 不属于当前书：结构哈希不一致。")
    package_id_mismatch = bool(
        metadata.get("package_id")
        and str(metadata["package_id"]) != str(expected_package.get("package_id", "") or "")
    )

    item_map, row_map, expected_items = _current_item_maps(expected_package)
    current_book_hash = _baseline_book_sha256([
        {
            "item_id": item_id,
            "baseline_text_sha256": _text_sha256(_current_text(item_map[item_id])),
            "delete_intentionally": bool(item_map[item_id].get("delete_intentionally", False)),
        }
        for item_id in sorted(item_map, key=lambda value: row_map[value])
    ])
    baseline_book_mismatch = bool(
        metadata.get("baseline_book_sha256")
        and str(metadata.get("baseline_book_sha256")) != current_book_hash
    )
    transaction_updates, atomic_count = _flatten_transactions(raw_transactions, row_map=row_map)
    raw_updates = list(raw_updates) + transaction_updates
    expected_ids = set(item_map)
    v4_editable_ids = None
    if is_v4_decisions:
        from engine.ai_disagreement_package_v4 import build_disagreement_records
        v4_records, _v4_summary = build_disagreement_records(expected_package)
        v4_editable_ids = {record["item_id"] for record in v4_records if record.get("model_action_required")}
    allowed = {
        "item_id", "row_id", "baseline_text", "expected_baseline_sha256",
        "edited_text", "edited_tokens", "delete_intentionally", "confidence",
        "reason", "evidence", "needs_review",
        "_transaction_id", "_transaction_operation", "_transaction_member_ids",
        "_transaction_reason", "_transaction_confidence", "_transaction_needs_review", "_v4_source",
    }
    seen: set[str] = set()
    prepared: list[dict] = []
    ordered_ids = [str(item.get("row_id") or item.get("item_id") or "") for item in expected_items]
    position_by_id = {item_id: position for position, item_id in enumerate(ordered_ids)}

    for index, raw in enumerate(raw_updates):
        if not isinstance(raw, dict):
            raise AIRepairEpubError(f"第 {index + 1} 条修复结果不是对象。")
        forbidden = sorted(set(raw) - allowed)
        if forbidden:
            raise AIRepairEpubError(f"第 {index + 1} 条包含越权字段：{', '.join(forbidden)}")
        item_id = str(raw.get("item_id") or raw.get("row_id") or "").strip()
        if item_id not in expected_ids:
            raise AIRepairEpubError(f"AI 返回未知或不可编辑 ID：{item_id or '(空)'}")
        if v4_editable_ids is not None and item_id not in v4_editable_ids:
            raise AIRepairEpubError(f"V4 决策试图修改已冻结一致条目：{item_id}")
        if item_id in seen:
            raise AIRepairEpubError(f"AI 修复结果包含重复 ID：{item_id}")
        seen.add(item_id)
        edited_text = str(raw.get("edited_text", "") or "")
        if raw.get("edited_tokens") is not None:
            token_text = _tokens_to_text(raw.get("edited_tokens"), field=f"条目 {item_id} 的 edited_tokens")
            if edited_text and _normalise_plain_text(edited_text) != _normalise_plain_text(token_text):
                raise AIRepairEpubError(f"条目 {item_id} 的 edited_text 与 edited_tokens 不一致。")
            edited_text = token_text
        delete_intentionally = _explicit_bool(
            raw.get("delete_intentionally", False),
            field=f"条目 {item_id} 的 delete_intentionally",
        )
        if not _normalise_plain_text(edited_text) and not delete_intentionally:
            raise AIRepairEpubError(f"条目 {item_id} 的 edited_text 为空；确需删除请设置 delete_intentionally=true。")
        position = position_by_id[item_id]
        previous_text = _current_text(item_map[ordered_ids[position - 1]]) if position > 0 else ""
        next_text = _current_text(item_map[ordered_ids[position + 1]]) if position + 1 < len(ordered_ids) else ""
        transaction_id = str(raw.get("_transaction_id", "") or "")
        reason = str(raw.get("reason", "") or "")
        transaction_reason = str(raw.get("_transaction_reason", "") or "")
        if transaction_reason:
            reason = (reason + "；" + transaction_reason).strip("；")
        confidence = raw.get("confidence", "")
        if confidence in (None, "") and raw.get("_transaction_confidence") not in (None, ""):
            confidence = raw.get("_transaction_confidence")
        prepared.append(_resolve_update_against_current(
            item_id=item_id,
            current_item=item_map[item_id],
            edited_text=edited_text,
            delete_intentionally=delete_intentionally,
            expected_baseline_sha256=str(raw.get("expected_baseline_sha256", "") or ""),
            baseline_text=(str(raw.get("baseline_text", "") or "") if "baseline_text" in raw else None),
            confidence=confidence,
            reason=reason,
            evidence=raw.get("evidence") or [],
            needs_review=bool(raw.get("needs_review", False) or raw.get("_transaction_needs_review", False)),
            previous_text=previous_text,
            next_text=next_text,
            transaction_id=transaction_id,
            transaction_operation=str(raw.get("_transaction_operation", "") or ""),
            transaction_member_ids=list(raw.get("_transaction_member_ids") or []),
        ))

    return prepared, _import_report(
        source=source_name,
        source_type="json",
        total_items=len(expected_ids),
        updates=prepared,
        metadata=metadata,
        atomic_count=atomic_count,
        package_id_mismatch=package_id_mismatch,
        baseline_book_mismatch=baseline_book_mismatch,
    )


def load_ai_repair_result(
    source_path: str | Path,
    *,
    expected_package: dict,
) -> tuple[list[dict], dict]:
    path = Path(source_path).expanduser()
    if path.suffix.lower() == ".epub":
        return extract_edits_from_repaired_epub(path, expected_package=expected_package)
    return load_ai_repair_json(path, expected_package=expected_package)
