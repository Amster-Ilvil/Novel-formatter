#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hybrid per-model OCR correction and canonical AI adjudication exchange.

Each conflict segment may carry sparse ``model_edits`` that correct only the
OCR sources which are wrong.  A row may independently carry one whole-row
``ai_verdict`` for final fusion.  Immutable base evidence, model identities,
physical-column IDs and locked consensus remain sealed in every schema.
"""
from __future__ import annotations

import copy
from difflib import SequenceMatcher
import gzip
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Callable, Iterable, Sequence
import uuid
import zipfile

from models.document import BlockType, UnifiedDocument
from engine.column_sentence_reflow import join_column_parts, has_sentence_terminal
from engine.multi_ocr_compare import (
    MultiOcrComparison,
    compare_ocr_documents,
    physical_column_text_snapshot,
    project_fused_text_to_physical_columns,
)
from engine.ocr_roundtrip_package import structure_hash, layout_hash
from utils.safe_archive import (
    UnsafeArchiveError,
    ZipExtractionLimits,
    validate_zip,
)

SCHEMA = "novel_formatter.multi_ocr_source_correction.v1"
CORRECTIONS_SCHEMA = "novel_formatter.multi_ocr_source_corrections.v1"
CANONICAL_CORRECTIONS_SCHEMA_V2 = "novel_formatter.multi_ocr_canonical_adjudication.v2"
CANONICAL_CORRECTIONS_SCHEMA = "novel_formatter.multi_ocr_canonical_adjudication.v3"
SUPPORTED_CANONICAL_CORRECTIONS_SCHEMAS = {
    CANONICAL_CORRECTIONS_SCHEMA_V2,
    CANONICAL_CORRECTIONS_SCHEMA,
}
RECOVERY_SCHEMA = "novel_formatter.multi_ocr_recovery_snapshot.v1"
_TEXT_TYPES = {
    BlockType.PARAGRAPH, BlockType.DIALOGUE, BlockType.CHAPTER,
    BlockType.SECTION, BlockType.RUBY, BlockType.FOOTNOTE, BlockType.TOC_ENTRY,
}

ProgressCallback = Callable[[str, int, int], None]

_SOURCE_CORRECTION_ZIP_LIMITS = ZipExtractionLimits(
    max_members=20_000,
    max_total_uncompressed=2 * 1024 * 1024 * 1024,
    max_single_file=512 * 1024 * 1024,
    max_compression_ratio=1_000.0,
)
_MAX_RECOVERY_DOCUMENT_BYTES = 512 * 1024 * 1024
_MAX_CORRECTION_JSON_BYTES = 256 * 1024 * 1024


def _validate_source_archive(archive: zipfile.ZipFile) -> None:
    try:
        validate_zip(archive, limits=_SOURCE_CORRECTION_ZIP_LIMITS)
    except (UnsafeArchiveError, zipfile.BadZipFile, OSError) as exc:
        raise SourceCorrectionError(f"逐源纠错 ZIP 安全校验失败：{exc}") from exc


def _gzip_declared_size(data: bytes) -> int:
    if len(data) < 4:
        return 0
    return int.from_bytes(data[-4:], "little", signed=False)


def _report_progress(callback: ProgressCallback | None, stage: str, current: int, total: int) -> None:
    if callback is None:
        return
    try:
        callback(str(stage), max(0, int(current)), max(1, int(total)))
    except Exception:
        # Progress reporting must never make an otherwise valid exchange fail.
        pass


class SourceCorrectionError(ValueError):
    """The source-correction package is stale, malformed or unsafe."""


def _json_bytes(value, *, pretty: bool = False) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=not pretty,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    ).encode("utf-8")


def _sha256(value) -> str:
    raw = value if isinstance(value, (bytes, bytearray)) else _json_bytes(value)
    return hashlib.sha256(raw).hexdigest()


def _safe_engine(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value or "ocr")).strip("_")
    return token[:64] or "ocr"


def _metadata(block) -> dict:
    value = getattr(block, "metadata", None)
    return value if isinstance(value, dict) else {}


def _column_ids(metadata: dict) -> list[str]:
    values = metadata.get("source_column_ids") or metadata.get("multi_ocr_column_ids") or []
    if isinstance(values, str):
        values = [values]
    if not values:
        value = str(metadata.get("column_id", "") or "")
        values = [value] if value else []
    return [str(value) for value in values if str(value)]


def _column_texts(metadata: dict, ids: Sequence[str], block_text: str) -> list[str]:
    for key in ("source_column_primary_texts", "source_column_texts"):
        values = metadata.get(key)
        if isinstance(values, list) and len(values) == len(ids):
            return [str(value or "") for value in values]
    if len(ids) == 1:
        return [str(block_text or "")]
    return []




def _source_structure_hash(doc: UnifiedDocument) -> str:
    """Stable structure identity that excludes every mutable OCR text field."""
    value = copy.deepcopy(doc.to_dict())
    value.pop("processing_log", None)
    metadata = value.get("metadata")
    if isinstance(metadata, dict):
        for key in list(metadata):
            if key.endswith("_report") or "correction" in key or "snapshot" in key:
                metadata.pop(key, None)
    for block in value.get("blocks", []) if isinstance(value.get("blocks"), list) else []:
        if not isinstance(block, dict):
            continue
        block["text"] = ""
        block.pop("ocr_raw", None)
        block.pop("modified_by", None)
        meta = block.get("metadata")
        if isinstance(meta, dict):
            for key in list(meta):
                lowered = key.lower()
                if (
                    "text" in lowered
                    or "candidate" in lowered
                    or "audit" in lowered
                    or "correction" in lowered
                    or lowered in {"last_column_text", "sentence_context_reocr_applied", "sentence_context_reocr_accepted"}
                ):
                    meta.pop(key, None)
    return _sha256(value)

def _identity_structure_hash(doc: UnifiedDocument) -> str:
    value = str(getattr(doc.metadata, "multi_ocr_source_correction_original_structure_sha256", "") or "")
    return value or _source_structure_hash(doc)


def _identity_layout_hash(doc: UnifiedDocument) -> str:
    value = str(getattr(doc.metadata, "multi_ocr_source_correction_original_layout_sha256", "") or "")
    return value or layout_hash(doc)


def _document_snapshot(doc: UnifiedDocument) -> str:
    columns, source = physical_column_text_snapshot(doc)
    return _sha256({
        "structure_sha256": _identity_structure_hash(doc),
        "layout_sha256": _identity_layout_hash(doc),
        "physical_column_source": source,
        "columns": columns,
    })


def _document_ocr_input_records(doc: UnifiedDocument) -> dict[str, dict[str, object]]:
    """Collect one non-destructive OCR-input audit record per physical column.

    The document may contain sentence-reflow blocks that reference several
    source columns.  We therefore normalise scalar and array metadata back to
    stable ``column_id`` keys.  Empty hashes stay explicitly unavailable; the
    exporter never reconstructs or guesses model inputs from scan evidence.
    """
    records: dict[str, dict[str, object]] = {}

    def values(metadata: dict, scalar_key: str, array_key: str, count: int) -> list[str]:
        raw = metadata.get(array_key, []) or []
        if isinstance(raw, str):
            raw = [raw]
        result = [str(value or "") for value in list(raw)]
        scalar = str(metadata.get(scalar_key, "") or "")
        if not result and scalar:
            result = [scalar] * max(1, count)
        if len(result) < count:
            result.extend([""] * (count - len(result)))
        return result[:count]

    for block in list(getattr(doc, "blocks", []) or []):
        metadata = getattr(block, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        column_ids = _column_ids(metadata)
        if not column_ids:
            continue
        count = len(column_ids)
        column_hashes = values(
            metadata, "column_ocr_input_sha256", "source_column_ocr_input_sha256", count
        )
        profiles = values(
            metadata, "column_ocr_input_profile", "source_column_ocr_input_profile", count
        )
        profile_hashes = values(
            metadata,
            "column_ocr_input_profile_sha256",
            "source_column_ocr_input_profile_sha256",
            count,
        )
        contracts = values(
            metadata, "column_ocr_input_contract", "source_column_ocr_input_contract", count
        )
        transports = values(
            metadata, "column_ocr_transport", "source_column_ocr_transport", count
        )
        page_input_hash = str(metadata.get("column_ndlocr_page_input_sha256", "") or "")
        for index, column_id in enumerate(column_ids):
            record = records.setdefault(
                column_id,
                {
                    "column_id": column_id,
                    "column_input_sha256": "",
                    "page_input_sha256": "",
                    "input_profile": "",
                    "input_profile_sha256": "",
                    "input_contract": "",
                    "transport": "",
                    "metadata_conflict": False,
                },
            )
            incoming = {
                "column_input_sha256": column_hashes[index],
                "page_input_sha256": page_input_hash,
                "input_profile": profiles[index],
                "input_profile_sha256": profile_hashes[index],
                "input_contract": contracts[index],
                "transport": transports[index],
            }
            for key, value in incoming.items():
                value = str(value or "")
                current = str(record.get(key, "") or "")
                if value and current and current != value:
                    record["metadata_conflict"] = True
                elif value and not current:
                    record[key] = value
    for record in records.values():
        column_hash = str(record.get("column_input_sha256", "") or "")
        page_hash = str(record.get("page_input_sha256", "") or "")
        if column_hash:
            record["final_input_sha256"] = column_hash
            record["input_hash_scope"] = "physical_column"
        elif page_hash:
            record["final_input_sha256"] = page_hash
            record["input_hash_scope"] = "page_routed"
        else:
            record["final_input_sha256"] = ""
            record["input_hash_scope"] = "unavailable"
    return records


def _document_ocr_input_audit(doc: UnifiedDocument) -> dict[str, object]:
    """Summarise actual OCR-input metadata without overstating availability."""
    records = _document_ocr_input_records(doc)
    profiles = {str(item.get("input_profile", "") or "") for item in records.values()}
    profile_hashes = {
        str(item.get("input_profile_sha256", "") or "") for item in records.values()
    }
    transports = {str(item.get("transport", "") or "") for item in records.values()}
    column_hashes = {
        str(item.get("column_input_sha256", "") or "") for item in records.values()
    }
    page_hashes = {str(item.get("page_input_sha256", "") or "") for item in records.values()}
    profiles.discard(""); profile_hashes.discard(""); transports.discard("")
    column_hashes.discard(""); page_hashes.discard("")
    column_hash_count = sum(bool(item.get("column_input_sha256")) for item in records.values())
    page_hash_count = sum(
        not bool(item.get("column_input_sha256")) and bool(item.get("page_input_sha256"))
        for item in records.values()
    )
    total = len(records)
    hashed = column_hash_count + page_hash_count
    if total and hashed == total:
        level = "full_hash"
    elif hashed:
        level = "partial_hash"
    elif profiles or profile_hashes or transports:
        level = "profile_only"
    else:
        level = "unavailable"
    return {
        "ocr_input_profiles": sorted(profiles),
        "ocr_input_profile_sha256": sorted(profile_hashes),
        "ocr_input_transports": sorted(transports),
        "physical_columns_with_input_sha256": int(column_hash_count),
        "physical_columns_with_page_input_sha256": int(page_hash_count),
        "physical_columns_without_input_sha256": max(0, total - hashed),
        "unique_input_sha256_count": len(column_hashes | page_hashes),
        "ocr_input_audit_level": level,
        "ocr_input_profile_metadata_available": bool(profiles or profile_hashes or transports),
        "ocr_input_hash_audit_available": bool(hashed),
        # Backward-compatible broad flag; consumers should prefer the two
        # explicit availability fields above.
        "ocr_input_audit_available": bool(profiles or profile_hashes or transports or hashed),
    }


def _build_detailed_ocr_input_audit(
    documents: Sequence[UnifiedDocument], registry: Sequence[dict]
) -> tuple[list[dict], dict[str, object]]:
    by_model: list[dict[str, dict[str, object]]] = [
        _document_ocr_input_records(document) for document in documents
    ]
    all_column_ids = sorted({column_id for rows in by_model for column_id in rows})
    rows: list[dict] = []
    shared_exact = 0
    unavailable = 0
    for column_id in all_column_ids:
        model_rows: list[dict] = []
        exact_groups: dict[str, list[str]] = {}
        for model_index, records in enumerate(by_model):
            item = dict(records.get(column_id) or {"column_id": column_id})
            model = registry[model_index] if model_index < len(registry) else {}
            item.update({
                "model_id": str(model.get("model_id", "") or ""),
                "model_index": model_index,
                "display_label": str(model.get("display_label", "") or ""),
            })
            input_hash = str(item.get("final_input_sha256", "") or "")
            scope = str(item.get("input_hash_scope", "unavailable") or "unavailable")
            if input_hash and scope == "physical_column":
                exact_groups.setdefault(input_hash, []).append(item["model_id"])
            model_rows.append(item)
        for item in model_rows:
            input_hash = str(item.get("final_input_sha256", "") or "")
            scope = str(item.get("input_hash_scope", "unavailable") or "unavailable")
            if scope == "page_routed":
                status = "page_routed"
                shared_ids: list[str] = []
            elif input_hash:
                shared_ids = exact_groups.get(input_hash, [])
                status = "shared_exact" if len(shared_ids) > 1 else "distinct_exact"
            else:
                status = "unavailable"
                shared_ids = []
                unavailable += 1
            if status == "shared_exact":
                shared_exact += 1
            item["shared_input_status"] = status
            item["shared_with_model_ids"] = shared_ids
            rows.append(item)
    summary = {
        "schema": "novel_formatter.multi_ocr_input_audit.v1",
        "record_count": len(rows),
        "physical_column_count": len(all_column_ids),
        "shared_exact_records": shared_exact,
        "unavailable_records": unavailable,
        "authority_rule": (
            "仅报告 OCR 文档中实际保存的输入哈希；缺失时标记 unavailable，"
            "不得从导出证据图反推或伪造模型输入。"
        ),
    }
    return rows, summary


def _build_model_registry_and_snapshots(
    documents: Sequence[UnifiedDocument], labels: Sequence[str]
) -> tuple[list[dict], list[dict[str, str]]]:
    registry: list[dict] = []
    snapshots: list[dict[str, str]] = []
    for index, doc in enumerate(documents):
        engine = str(getattr(getattr(doc, "metadata", None), "source_engine", "") or f"ocr_{index + 1}")
        doc_layout_hash = _identity_layout_hash(doc)
        doc_structure_hash = _identity_structure_hash(doc)
        columns, column_source = physical_column_text_snapshot(doc)
        structural_identity = _sha256({
            "engine": engine,
            "model_index": index,
            "layout_sha256": doc_layout_hash,
        })
        model_id = f"model:{_safe_engine(engine)}:{index}:{structural_identity[:12]}"
        registry.append({
            "model_id": model_id,
            "model_index": index,
            "display_label": str(labels[index] if index < len(labels) else f"OCR 模型 {index + 1}"),
            "source_engine": engine,
            "layout_sha256": doc_layout_hash,
            "structure_sha256": doc_structure_hash,
            "document_snapshot_sha256": _sha256({
                "structure_sha256": doc_structure_hash,
                "layout_sha256": doc_layout_hash,
                "physical_column_source": column_source,
                "columns": columns,
            }),
            "physical_column_source": column_source,
            "physical_column_count": len(columns),
            **_document_ocr_input_audit(doc),
        })
        snapshots.append(columns)
    return registry, snapshots


def build_model_registry(
    documents: Sequence[UnifiedDocument], labels: Sequence[str]
) -> list[dict]:
    registry, _snapshots = _build_model_registry_and_snapshots(documents, labels)
    return registry


def _nw_align(left: str, right: str) -> tuple[list[str | None], list[str | None]]:
    """Deterministic character alignment used only to expose locked/diff spans."""
    a, b = str(left or ""), str(right or "")
    n, m = len(a), len(b)
    gap, mismatch, match = -2, -1, 2
    score = [[0] * (m + 1) for _ in range(n + 1)]
    trace = [[0] * (m + 1) for _ in range(n + 1)]  # 0 diag, 1 up, 2 left
    for i in range(1, n + 1):
        score[i][0] = i * gap
        trace[i][0] = 1
    for j in range(1, m + 1):
        score[0][j] = j * gap
        trace[0][j] = 2
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            diag = score[i - 1][j - 1] + (match if ai == b[j - 1] else mismatch)
            up = score[i - 1][j] + gap
            left_score = score[i][j - 1] + gap
            best = max(diag, up, left_score)
            score[i][j] = best
            # Prefer exact/mismatch diagonal, then deletion, then insertion.
            trace[i][j] = 0 if diag == best else 1 if up == best else 2
    aligned_a: list[str | None] = []
    aligned_b: list[str | None] = []
    i, j = n, m
    while i or j:
        direction = trace[i][j]
        if i and j and direction == 0:
            aligned_a.append(a[i - 1]); aligned_b.append(b[j - 1]); i -= 1; j -= 1
        elif i and (not j or direction == 1):
            aligned_a.append(a[i - 1]); aligned_b.append(None); i -= 1
        else:
            aligned_a.append(None); aligned_b.append(b[j - 1]); j -= 1
    aligned_a.reverse(); aligned_b.reverse()
    return aligned_a, aligned_b


def _multi_align(texts: Sequence[str]) -> list[tuple[str | None, ...]]:
    values = [str(value or "") for value in texts]
    if not values:
        return []
    if len(values) == 1:
        return [(char,) for char in values[0]]
    a, b = _nw_align(values[0], values[1])
    columns: list[list[str | None]] = [[ca, cb] for ca, cb in zip(a, b)]
    for value in values[2:]:
        representative = "".join(next((char for char in col if char is not None), "") for col in columns)
        rep_aligned, value_aligned = _nw_align(representative, value)
        rebuilt: list[list[str | None]] = []
        old_index = 0
        for rep_char, new_char in zip(rep_aligned, value_aligned):
            if rep_char is None:
                rebuilt.append([None] * len(columns[0]) + [new_char])
            else:
                if old_index >= len(columns):
                    raise SourceCorrectionError("多模型字符对齐内部越界。")
                rebuilt.append(list(columns[old_index]) + [new_char])
                old_index += 1
        while old_index < len(columns):
            rebuilt.append(list(columns[old_index]) + [None])
            old_index += 1
        columns = rebuilt
    return [tuple(column) for column in columns]


def split_conflict_segments(texts: Sequence[str], row_id: str, model_ids: Sequence[str]) -> list[dict]:
    columns = _multi_align(texts)
    if not columns:
        return []
    segments: list[dict] = []
    current_locked: bool | None = None
    current_columns: list[tuple[str | None, ...]] = []

    def flush() -> None:
        nonlocal current_columns, current_locked
        if not current_columns or current_locked is None:
            return
        index = len(segments)
        segment_id = f"seg:{row_id}:{index:03d}"
        model_texts = {
            model_id: "".join(column[model_index] or "" for column in current_columns)
            for model_index, model_id in enumerate(model_ids)
        }
        if current_locked:
            consensus = next(iter(model_texts.values()), "")
            segment = {
                "segment_id": segment_id,
                "type": "locked_consensus",
                "consensus_text": consensus,
                "segment_sha256": _sha256({"type": "locked_consensus", "text": consensus}),
            }
        else:
            segment = {
                "segment_id": segment_id,
                "type": "editable_conflict",
                "model_texts": model_texts,
                "model_edits": {},
                "reason": "",
                "confidence": 0.0,
                "segment_sha256": _sha256({"type": "editable_conflict", "model_texts": model_texts}),
            }
        segments.append(segment)
        current_columns = []
        current_locked = None

    for column in columns:
        locked = bool(column and all(char is not None for char in column) and len(set(column)) == 1)
        if current_locked is None:
            current_locked = locked
        elif current_locked != locked:
            flush()
            current_locked = locked
        current_columns.append(column)
    flush()
    return segments



_PLACEHOLDER_CHARS = frozenset("□�\ufffd")
_JAPANESE_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_SUSPICIOUS_INLINE_LATIN_RE = re.compile(
    r"(?<=[\u3040-\u30ff\u3400-\u9fff])[a-z](?=[\u3040-\u30ff\u3400-\u9fff。！？、」』）\s]|$)"
    r"|(?<![A-Za-z])[a-z](?=[。！？、」』）])"
)


def _contains_placeholder(text: str) -> bool:
    value = str(text or "")
    return any(char in value for char in _PLACEHOLDER_CHARS)


def _suspicious_inline_latin(text: str) -> list[str]:
    """Return isolated lower-case Latin glyphs embedded in Japanese prose.

    Upper-case ranks/skills (A/S/D) and ordinary ASCII words remain allowed.
    The check is intentionally narrow so ``慟哭すd。`` is blocked without
    rejecting legitimate status text or romanised names.
    """
    value = str(text or "")
    return [match.group(0) for match in _SUSPICIOUS_INLINE_LATIN_RE.finditer(value)]


def _pair_has_reordered_multiset(left: str, right: str) -> bool:
    a, b = str(left or ""), str(right or "")
    if not a or not b or a == b:
        return False
    if len(a) != len(b):
        return False
    # Same characters but a different order is the exact class that the old
    # locked-LCS splicer could corrupt (e.g. 一歩。二歩。 / 二歩。一歩。).
    return sorted(a) == sorted(b)


def _legacy_row_requires_whole_verdict(texts: Sequence[str]) -> bool:
    values = [str(value or "") for value in texts if str(value or "")]
    clean = [value for value in values if not _contains_placeholder(value)]
    for index, left in enumerate(clean):
        for right in clean[index + 1:]:
            if _pair_has_reordered_multiset(left, right):
                return True
            matcher = SequenceMatcher(None, left, right, autojunk=False)
            edits = [opcode for opcode in matcher.get_opcodes() if opcode[0] != "equal"]
            max_len = max(len(left), len(right), 1)
            if abs(len(left) - len(right)) > max(4, int(max_len * 0.22)):
                return True
            if len(edits) >= 4 and matcher.ratio() < 0.82:
                return True
    return False


def _canonical_decision_id(column_ids: Sequence[str]) -> str:
    ids = [str(value) for value in column_ids if str(value)]
    return f"decision:{_sha256(ids)[:20]}"


def _clean_candidate(text: str) -> bool:
    value = str(text or "")
    return bool(value.strip()) and not _contains_placeholder(value) and not _suspicious_inline_latin(value)


def _decision_matches_current_evidence(decision: dict, model_ids: Sequence[str], texts: Sequence[str]) -> bool:
    """Return whether a stored verdict still belongs to the current raw OCR row.

    Imported decisions carry ``raw_model_texts``.  Re-exporting may happen after
    harmless UI work, but a decision must never be silently prefilled when the
    underlying OCR evidence changed or the model registry was replaced.
    Decisions from very early snapshots without raw evidence are accepted only
    when their stable physical-column group still matches; this is the legacy
    compatibility path and is recorded in the exported migration metadata.
    """
    def compatible(left_values: Sequence[str], right_values: Sequence[str]) -> bool:
        left = [str(value or "") for value in left_values]
        right = [str(value or "") for value in right_values]
        if left == right:
            return True
        if len(left) != len(right):
            return False
        weighted = 0.0
        total = 0.0
        minimum = 1.0
        for old, current in zip(left, right):
            if old == current:
                ratio = 1.0
            elif not old or not current:
                ratio = 0.0
            else:
                ratio = SequenceMatcher(None, old, current, autojunk=False).ratio()
            weight = float(max(1, len(old), len(current)))
            weighted += ratio * weight
            total += weight
            minimum = min(minimum, ratio)
        aggregate = weighted / total if total else 0.0
        # Minor OCR normalisation and restored Apple-Vision punctuation may
        # alter raw evidence after a crash/rebind.  Stable column identity plus
        # very high row similarity is sufficient to resume; material changes
        # still force the row back into the pending queue.
        return aggregate >= 0.90 and minimum >= 0.72

    indexed = decision.get("raw_model_texts_by_index")
    raw = decision.get("raw_model_texts")
    if isinstance(indexed, list) and indexed and isinstance(raw, dict) and raw:
        if [str(value or "") for value in raw.values()] != [str(value or "") for value in indexed]:
            return False
    if isinstance(indexed, list) and indexed:
        return compatible(indexed, texts)
    if not isinstance(raw, dict) or not raw:
        return True
    expected = {str(model_ids[index]): str(texts[index] if index < len(texts) else "")
                for index in range(len(model_ids))}
    if all(model_id in raw for model_id in expected):
        return compatible([raw.get(model_id, "") for model_id in expected], list(expected.values()))
    # Model IDs include layout-derived identity and may legitimately change
    # after recovery/rebinding.  Import already validates engine slot order, so
    # the original insertion order is the safe compatibility fallback.
    if len(raw) == len(texts):
        return compatible(list(raw.values()), texts)
    return False


def _accepted_decision_for_export(
    decision: dict | None,
    *,
    column_ids: Sequence[str],
    model_ids: Sequence[str],
    texts: Sequence[str],
) -> tuple[dict | None, str]:
    """Validate an already imported verdict against the current OCR evidence.

    Returns ``(resolved_verdict, state)`` where state is one of ``prefilled``,
    ``stale`` or ``none``.  Callers may either seal a validated verdict as
    resolved history or expose it as read-only context for an explicit second
    review pass; the raw OCR evidence itself is never rewritten.
    """
    if not isinstance(decision, dict):
        return None, "none"
    if str(decision.get("status", "") or "") != "accepted":
        return None, "none"
    expected_ids = [str(value) for value in column_ids if str(value)]
    decision_ids = [str(value) for value in (decision.get("column_ids") or []) if str(value)]
    if decision_ids != expected_ids:
        return None, "stale"
    final_text = str(decision.get("final_text", "") or "")
    delete_intentionally = bool(decision.get("delete_intentionally", False))
    if not final_text and not delete_intentionally:
        return None, "stale"
    if _contains_placeholder(final_text) or _suspicious_inline_latin(final_text):
        return None, "stale"
    if not _decision_matches_current_evidence(decision, model_ids, texts):
        return None, "stale"
    resolved = {
        "decision_id": str(decision.get("decision_id", "") or _canonical_decision_id(expected_ids)),
        "final_text": final_text,
        "reason": str(decision.get("reason", "") or "已从此前导入的 OCR 裁决安全迁移。"),
        "confidence": float(decision.get("confidence", 0.0) or 0.0),
        "delete_intentionally": delete_intentionally,
        "source": str(decision.get("source", "") or "prior_canonical_decision"),
        "derivation": str(decision.get("derivation", "") or "prior_accepted_verdict"),
        "audit_flags": [str(value) for value in (decision.get("audit_flags") or [])],
        "migrated_from_prior_session": True,
        "raw_evidence_verified": bool(
            (isinstance(decision.get("raw_model_texts"), dict) and decision.get("raw_model_texts"))
            or (isinstance(decision.get("raw_model_texts_by_index"), list) and decision.get("raw_model_texts_by_index"))
        ),
        "historical_raw_model_texts": copy.deepcopy(
            decision.get("historical_raw_model_texts") or {}
        ),
        "historical_raw_model_texts_by_index": [
            str(value or "") for value in (decision.get("historical_raw_model_texts_by_index") or [])
        ],
        "historical_disagreement": bool(decision.get("historical_disagreement", False)),
        "resolution_kind": str(decision.get("resolution_kind", "") or ""),
    }
    return resolved, "prefilled"


def _derive_legacy_canonical_verdict(row: dict, model_ids: Sequence[str]) -> dict:
    """Migrate a V1 per-model edit file into one explicit AI verdict.

    The V1 format did not contain ``final_text``.  Its only explicit AI output
    is ``model_edits`` on each editable segment, so migration uses those edit
    values directly and never counts unchanged OCR candidates as votes.  This
    is important: two identical OCR strings are evidence only, not a standard
    that can overrule the AI decision.

    Rows with reordered/large alignment differences are deliberately left
    unresolved because the old locked-LCS segment splice can lose or duplicate
    text (for example ``一歩。二歩。`` versus ``二歩。一歩。``).
    """
    raw_texts = [
        str((row.get("base_model_texts") or {}).get(model_id, "") or "")
        for model_id in model_ids
    ]
    corrected = [_incoming_model_text(row, model_id) for model_id in model_ids]
    flags: list[str] = []
    complex_row = _legacy_row_requires_whole_verdict(raw_texts)
    if complex_row:
        flags.append("legacy_complex_alignment_requires_whole_row_verdict")

    pieces: list[str] = []
    explicit_segments = 0
    unresolved_segment = False
    unsafe_edit = False
    if not complex_row:
        for segment in row.get("segments", []) or []:
            if not isinstance(segment, dict):
                unresolved_segment = True
                break
            if segment.get("type") == "locked_consensus":
                pieces.append(str(segment.get("consensus_text", "") or ""))
                continue
            edits = segment.get("model_edits") or {}
            if not isinstance(edits, dict) or not edits:
                unresolved_segment = True
                flags.append("legacy_conflict_segment_without_explicit_ai_edit")
                break
            values = [str(value or "") for value in edits.values()]
            unique_values = list(dict.fromkeys(values))
            if len(unique_values) != 1:
                unresolved_segment = True
                flags.append("legacy_conflict_segment_has_multiple_ai_results")
                break
            chosen = unique_values[0]
            explicit_segments += 1
            if _contains_placeholder(chosen):
                unsafe_edit = True
                flags.append("legacy_ai_edit_contains_placeholder")
            if _suspicious_inline_latin(chosen):
                unsafe_edit = True
                flags.append("legacy_ai_edit_contains_suspicious_inline_latin")
            pieces.append(chosen)

    final_text = "".join(pieces)
    accepted = bool(
        not complex_row
        and not unresolved_segment
        and explicit_segments > 0
        and not unsafe_edit
        and not _contains_placeholder(final_text)
        and not _suspicious_inline_latin(final_text)
    )
    if final_text and (_contains_placeholder(final_text) or _suspicious_inline_latin(final_text)):
        flags.append("derived_verdict_failed_text_safety")
        accepted = False

    return {
        "decision_id": _canonical_decision_id(row.get("column_ids") or []),
        "row_id": str(row.get("row_id", "") or ""),
        "row_index": int(row.get("row_index", 0) or 0),
        "column_ids": [str(value) for value in (row.get("column_ids") or []) if str(value)],
        "final_text": final_text if accepted else "",
        "status": "accepted" if accepted else "unresolved",
        "source": "legacy_model_edits_migrated",
        "derivation": "legacy_explicit_ai_segment_edits" if accepted else "",
        "confidence": 0.95 if accepted else 0.0,
        "reason": "旧版逐模型修改已按 AI 明确填写的 model_edits 迁移为唯一权威正文；未使用两模型相同作为标准。",
        "audit_flags": sorted(set(flags)),
        "raw_model_texts": {model_id: raw_texts[index] for index, model_id in enumerate(model_ids)},
        "raw_model_texts_by_index": list(raw_texts),
        "legacy_corrected_model_texts": {model_id: corrected[index] for index, model_id in enumerate(model_ids)},
        "delete_intentionally": bool(accepted and final_text == ""),
    }


def _read_canonical_verdict(row: dict, model_ids: Sequence[str], schema: str) -> dict:
    column_ids = [str(value) for value in (row.get("column_ids") or []) if str(value)]
    if schema in SUPPORTED_CANONICAL_CORRECTIONS_SCHEMAS:
        verdict = row.get("ai_verdict") or row.get("resolved_verdict") or {}
        final_text = str(verdict.get("final_text", "") or "")
        delete_intentionally = bool(verdict.get("delete_intentionally", False))
        flags: list[str] = []
        if _contains_placeholder(final_text):
            flags.append("final_text_contains_placeholder")
        if _suspicious_inline_latin(final_text):
            flags.append("final_text_contains_suspicious_inline_latin")
        accepted = bool((final_text or delete_intentionally) and not flags)
        return {
            "decision_id": str(verdict.get("decision_id", "") or _canonical_decision_id(column_ids)),
            "row_id": str(row.get("row_id", "") or ""),
            "row_index": int(row.get("row_index", 0) or 0),
            "column_ids": column_ids,
            "final_text": final_text if accepted else "",
            "status": "accepted" if accepted else "unresolved",
            "source": str(
                verdict.get("source", "")
                or ("ai_canonical_verdict_v3" if schema == CANONICAL_CORRECTIONS_SCHEMA else "ai_canonical_verdict_v2")
            ),
            "derivation": str(verdict.get("derivation", "") or "explicit_final_text"),
            "confidence": float(verdict.get("confidence", 0.0) or 0.0),
            "reason": str(verdict.get("reason", "") or ""),
            "audit_flags": flags,
            "raw_model_texts": dict(row.get("base_model_texts") or {}),
            "raw_model_texts_by_index": [
                str((row.get("base_model_texts") or {}).get(model_id, "") or "")
                for model_id in model_ids
            ],
            "historical_raw_model_texts": copy.deepcopy(
                verdict.get("historical_raw_model_texts") or {}
            ),
            "historical_raw_model_texts_by_index": [
                str(value or "") for value in (verdict.get("historical_raw_model_texts_by_index") or [])
            ],
            "historical_disagreement": bool(verdict.get("historical_disagreement", False)),
            "resolution_kind": str(verdict.get("resolution_kind", "") or ""),
            "delete_intentionally": delete_intentionally,
        }
    return _derive_legacy_canonical_verdict(row, model_ids)


def merge_canonical_decision_overlays(
    existing: Sequence[dict] | None,
    incoming: Sequence[dict] | None,
) -> tuple[list[dict], dict]:
    """Merge repeated AI adjudication imports without losing earlier good rows.

    Stable physical-column IDs are the authority.  A later *accepted* verdict
    replaces the earlier verdict for the same row (so a better second-pass AI
    result can win), while an omitted or unresolved later row never erases an
    already accepted verdict.  This makes repeated package imports cumulative
    and idempotent instead of treating every import as a whole-session replace.
    """
    by_key: dict[tuple[str, ...], dict] = {}
    order: list[tuple[str, ...]] = []

    def key_for(item: dict) -> tuple[str, ...]:
        return tuple(str(value) for value in (item.get("column_ids") or []) if str(value))

    for item in existing or ():
        if not isinstance(item, dict):
            continue
        key = key_for(item)
        if not key:
            continue
        if key not in by_key:
            order.append(key)
        by_key[key] = copy.deepcopy(item)

    new_rows = 0
    replaced_rows = 0
    preserved_rows = 0
    unresolved_new_rows = 0
    changed_accepted: list[dict] = []
    for item in incoming or ():
        if not isinstance(item, dict):
            continue
        key = key_for(item)
        if not key:
            continue
        current = by_key.get(key)
        incoming_status = str(item.get("status", "") or "")
        current_status = str((current or {}).get("status", "") or "")
        if incoming_status == "accepted":
            candidate = copy.deepcopy(item)
            changed = True
            if current is None:
                order.append(key)
                new_rows += 1
            elif current_status == "accepted":
                # Later accepted output is authoritative for this stable row,
                # but an identical re-import is idempotent and must not steal a
                # later manual selection in the GUI.
                changed = (
                    str(current.get("final_text", "") or "") != str(candidate.get("final_text", "") or "")
                    or bool(current.get("delete_intentionally", False)) != bool(candidate.get("delete_intentionally", False))
                    or float(current.get("confidence", 0.0) or 0.0) != float(candidate.get("confidence", 0.0) or 0.0)
                    or str(current.get("reason", "") or "") != str(candidate.get("reason", "") or "")
                )
                if changed:
                    replaced_rows += 1
                else:
                    preserved_rows += 1
            else:
                replaced_rows += 1
            by_key[key] = candidate
            if changed:
                changed_accepted.append(candidate)
            continue

        if current is not None and current_status == "accepted":
            # A blank/unresolved second pass is not evidence that the previous
            # accepted decision became wrong.  Preserve it until a later pass
            # supplies a concrete accepted replacement.
            preserved_rows += 1
            continue

        candidate = copy.deepcopy(item)
        if current is None:
            order.append(key)
            unresolved_new_rows += 1
        by_key[key] = candidate

    merged = [by_key[key] for key in order if key in by_key]
    return merged, {
        "existing_rows": sum(1 for item in existing or () if isinstance(item, dict) and key_for(item)),
        "incoming_rows": sum(1 for item in incoming or () if isinstance(item, dict) and key_for(item)),
        "merged_rows": len(merged),
        "new_accepted_rows": new_rows,
        "replaced_accepted_rows": replaced_rows,
        "preserved_prior_rows": preserved_rows,
        "new_unresolved_rows": unresolved_new_rows,
        "changed_accepted_decisions": changed_accepted,
    }


def apply_canonical_decisions_to_fusion_states(
    fusion_states,
    comparison: MultiOcrComparison,
    decisions: Sequence[dict],
) -> int:
    """Overlay AI results on fusion states without rewriting any OCR source.

    Every original model candidate remains in the judgement box.  Accepted AI
    output is represented by its own synthetic card, even when its text equals
    one existing OCR candidate, so provenance stays explicit and the user can
    still compare/select the pre-import disagreement.  Blank or unresolved AI
    rows never clear an existing manual/automatic selection.
    """
    from engine.ocr_compare_view_model import upsert_external_candidate

    by_columns = {
        tuple(str(value) for value in (item.get("column_ids") or [])): item
        for item in decisions if isinstance(item, dict) and item.get("column_ids")
    }
    applied = 0
    for row, state in zip(comparison.rows, fusion_states):
        decision = by_columns.get(tuple(str(value) for value in (row.column_ids or ())))
        if not decision:
            continue
        status = str(decision.get("status", "") or "")
        final_text = str(decision.get("final_text", "") or "")
        if status != "accepted" or (not final_text and not decision.get("delete_intentionally")):
            # Non-results are audit information only.  Never reopen or blank a
            # previously valid fusion line merely because the imported JSON left
            # this row unresolved.
            state.review_indices = state._build_review_indices()
            continue
        source = str(decision.get("source", "") or "")
        per_model = source.startswith("ai_per_model_source_correction")

        # Older destructive builds may already have collapsed the active model
        # documents.  Recover the immutable export-time OCR disagreement from
        # the JSON itself and show any missing original candidates beside the
        # current candidates.  These evidence cards are selectable fusion
        # alternatives but never write back to a model document.
        original_values = [
            str(value or "")
            for value in (
                decision.get("historical_raw_model_texts_by_index")
                or decision.get("raw_model_texts_by_index")
                or []
            )
        ]
        if original_values and len(set(original_values)) > 1:
            grouped_original: dict[str, list[int]] = {}
            for model_index, original_text in enumerate(original_values):
                if original_text.strip():
                    grouped_original.setdefault(original_text, []).append(model_index)
            for original_text, original_indices in grouped_original.items():
                exact_live_candidate = any(
                    candidate.text.strip() == original_text.strip()
                    and tuple(candidate.model_indices) == tuple(original_indices)
                    for candidate in state.candidates
                )
                if exact_live_candidate:
                    continue
                source_labels = [
                    str(
                        comparison.labels[model_index]
                        if model_index < len(comparison.labels)
                        else f"模型{model_index + 1}"
                    )
                    for model_index in original_indices
                ]
                selected_before = state.selected_index
                selection_origin_before = str(getattr(state, "selection_origin", "") or "")
                upsert_external_candidate(
                    state,
                    original_text,
                    display_label="导出时原OCR·" + "＋".join(source_labels),
                    select=False,
                    reason="裁决包中密封的导出时原始 OCR 证据；用于恢复此前分歧，不改写当前模型。",
                    confidence=0.0,
                    transaction_id=(
                        str(decision.get("decision_id", "") or "")
                        + ":original:"
                        + ",".join(str(value) for value in original_indices)
                    ),
                    transaction_operation="original_ocr_evidence_overlay",
                    transaction_member_ids=tuple(str(value) for value in (decision.get("column_ids") or [])),
                    audit_level="original_ocr_evidence_overlay",
                    audit_flags=("sealed_export_time_raw_ocr", "selectable_without_source_writeback"),
                    force_role_candidate=True,
                )
                state.selected_index = selected_before
                state.selection_origin = selection_origin_before

        label = "AI逐模型纠错结果" if per_model else "AI最终裁决"
        index = upsert_external_candidate(
            state,
            final_text,
            display_label=label,
            select=True,
            reason=str(decision.get("reason", "") or "AI 纠错结果，仅作为融合覆盖层；原 OCR 证据未修改。"),
            confidence=float(decision.get("confidence", 0.0) or 0.0),
            allow_empty=bool(decision.get("delete_intentionally", False)),
            transaction_id=str(decision.get("decision_id", "") or ""),
            transaction_operation=(
                "per_model_correction_overlay" if per_model else "canonical_text_verdict_overlay"
            ),
            transaction_member_ids=tuple(str(value) for value in (decision.get("column_ids") or [])),
            audit_level="non_destructive_ai_correction_overlay",
            audit_flags=tuple(dict.fromkeys([
                *(str(value) for value in (decision.get("audit_flags") or [])),
                "raw_ocr_sources_preserved",
                "original_disagreement_visible",
            ])),
            force_role_candidate=True,
            selection_origin="ai_overlay",
        )
        if index is not None:
            state.requires_confirmation = False
            state.review_classification = "ai_correction_overlay"
            state.preserve_candidates_visible = True
            state.review_indices = state._build_review_indices()
            applied += 1
    return applied


def canonical_text_safety_issues(text: str) -> list[str]:
    value = str(text or "")
    issues: list[str] = []
    if _contains_placeholder(value):
        issues.append("包含 OCR 占位符 □/�")
    suspicious = _suspicious_inline_latin(value)
    if suspicious:
        issues.append(f"日文正文夹有可疑小写拉丁字母：{''.join(suspicious[:6])}")
    return issues


def _row_id(row, row_index: int) -> str:
    column_ids = [str(value) for value in (getattr(row, "column_ids", ()) or ())]
    identity = {
        "column_ids": column_ids,
        "primary_block_id": str(getattr(row, "primary_block_id", "") or ""),
        "primary_segment_index": int(getattr(row, "primary_segment_index", 0) or 0),
        "page": int(getattr(row, "page", 0) or 0),
    }
    return f"row:{row_index:06d}:{_sha256(identity)[:16]}"


def _geometry_snapshot(doc: UnifiedDocument) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for block in doc.blocks:
        if block.type not in _TEXT_TYPES:
            continue
        metadata = _metadata(block)
        regions = metadata.get("ocr_review_regions") or []
        if isinstance(regions, list):
            for region in regions:
                if not isinstance(region, dict):
                    continue
                column_id = str(region.get("column_id", "") or "")
                bbox = region.get("bbox")
                if column_id and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    result[column_id] = {
                        "column_id": column_id,
                        "page": int(region.get("page", getattr(block, "page", 0)) or 0),
                        "bbox": [float(value or 0.0) for value in bbox],
                    }
        ids = _column_ids(metadata)
        bbox = getattr(block, "bbox", None)
        if ids and bbox is not None:
            for column_id in ids:
                result.setdefault(column_id, {
                    "column_id": column_id,
                    "page": int(getattr(block, "page", 0) or 0),
                    "bbox": [float(bbox.x), float(bbox.y), float(bbox.w), float(bbox.h)],
                })
    return result


def _page_paths(doc: UnifiedDocument) -> dict[int, str]:
    return {
        int(getattr(page, "page_no", 0) or 0): str(getattr(page, "image_path", "") or "")
        for page in doc.pages
        if str(getattr(page, "image_path", "") or "")
    }


def _immutable_projection(payload: dict) -> dict:
    """Return the sealed package view while excluding only explicit AI output.

    ``model_edits`` and editable verdict values are explicit AI output in all
    supported schemas.  Model identities, physical-column mapping, original
    model text and locked consensus remain sealed and cannot be altered.
    """
    value = {key: item for key, item in payload.items() if key != "immutable_manifest_sha256"}
    sealed_rows: list[dict] = []
    for row in payload.get("rows", []) if isinstance(payload.get("rows"), list) else []:
        if not isinstance(row, dict):
            sealed_rows.append(row)
            continue
        sealed_row = {key: item for key, item in row.items() if key not in {"segments", "ai_verdict"}}
        verdict = row.get("ai_verdict")
        if isinstance(verdict, dict):
            sealed_row["ai_verdict"] = {
                key: item for key, item in verdict.items()
                if key not in {"final_text", "reason", "confidence", "delete_intentionally"}
            }
        sealed_segments: list[dict] = []
        for segment in row.get("segments", []) if isinstance(row.get("segments"), list) else []:
            if not isinstance(segment, dict):
                sealed_segments.append(segment)
                continue
            sealed_segments.append({
                key: item for key, item in segment.items()
                if key not in {"model_edits", "reason", "confidence"}
            })
        sealed_row["segments"] = sealed_segments
        sealed_rows.append(sealed_row)
    value["rows"] = sealed_rows
    return value

def _alignment_snapshot(comparison: MultiOcrComparison) -> list[dict]:
    result = []
    for index, row in enumerate(comparison.rows):
        result.append({
            "row_id": _row_id(row, index),
            "row_index": index,
            "column_ids": [str(value) for value in (row.column_ids or ())],
            "page": int(row.page or 0),
            "primary_block_id": str(row.primary_block_id or ""),
            "primary_segment_index": int(row.primary_segment_index or 0),
            "block_type": str(row.block_type or "paragraph"),
            "atomic": bool(row.atomic),
        })
    return result


def build_source_correction_payload(
    documents: Sequence[UnifiedDocument],
    labels: Sequence[str],
    comparison: MultiOcrComparison,
    *,
    canonical_decisions: Sequence[dict] | None = None,
    review_prior_decisions: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    docs = list(documents)
    if not 2 <= len(docs) <= 3:
        raise SourceCorrectionError("逐源纠错只支持 2～3 个 OCR 模型。")
    if comparison.alignment_mode != "column_id_consensus":
        raise SourceCorrectionError("逐源纠错要求共享物理列 ID；请使用固定分列多模型 OCR 后再导出。")
    _report_progress(progress_callback, "建立模型与物理列索引", 0, len(docs))
    registry, _snapshots = _build_model_registry_and_snapshots(docs, labels)
    _report_progress(progress_callback, "建立模型与物理列索引", len(docs), len(docs))
    model_ids = [item["model_id"] for item in registry]
    alignment = _alignment_snapshot(comparison)
    rows: list[dict] = []
    conflict_count = 0
    provisional_count = 0
    locked_count = 0
    prefilled_count = 0
    prefilled_legacy_count = 0
    prefilled_native_count = 0
    stale_prior_count = 0
    reviewable_prior_count = 0
    pending_conflict_count = 0
    pending_provisional_count = 0
    prior_by_columns = {
        tuple(str(value) for value in (item.get("column_ids") or []) if str(value)): copy.deepcopy(item)
        for item in (canonical_decisions or [])
        if isinstance(item, dict) and item.get("column_ids")
    }
    seen_columns: set[str] = set()
    total_rows = max(1, len(comparison.rows))
    for row_index, row in enumerate(comparison.rows):
        if row_index == 0 or (row_index + 1) % 100 == 0 or row_index + 1 == total_rows:
            _report_progress(progress_callback, "生成锁定段与冲突段", row_index + 1, total_rows)
        column_ids = [str(value) for value in (row.column_ids or ())]
        if not column_ids:
            raise SourceCorrectionError(f"第 {row_index + 1} 行没有物理列 ID，不能安全逐源回写。")
        duplicated = seen_columns.intersection(column_ids)
        if duplicated:
            raise SourceCorrectionError(f"物理列被多个比较行重复占用：{sorted(duplicated)[:3]}")
        seen_columns.update(column_ids)
        row_id = alignment[row_index]["row_id"]
        texts = [str(value or "") for value in row.texts[:len(model_ids)]]
        while len(texts) < len(model_ids):
            texts.append("")
        segments = split_conflict_segments(texts, row_id, model_ids)
        actual_conflict = any(segment["type"] == "editable_conflict" for segment in segments)
        provisional = bool(getattr(row, "provisional_consensus", False))
        # v8-compatible queue semantics: a two-independent-model agreement is
        # an automatically usable fusion candidate, not a new AI/manual task.
        # It remains explicitly labelled as provisional evidence, but only
        # genuine differing text enters the editable adjudication queue.
        review_required = bool(actual_conflict)
        if actual_conflict:
            conflict_count += 1
        elif provisional:
            provisional_count += 1
        else:
            locked_count += 1
        # Preserve accepted adjudication history even after corrections turn a
        # formerly conflicting row into exact/provisional consensus.
        prior_decision = prior_by_columns.get(tuple(column_ids))
        resolved_verdict, prior_state = _accepted_decision_for_export(
            prior_decision,
            column_ids=column_ids,
            model_ids=model_ids,
            texts=texts,
        )
        # OCR compare exports can explicitly re-open prior accepted conflict
        # verdicts for a second AI pass.  Exact/provisional consensus remains
        # locked.  The previous verdict is supplied as read-only context; if
        # the new pass returns nothing, cumulative import preserves the old one.
        prior_reviewable = bool(
            review_prior_decisions and actual_conflict and resolved_verdict is not None
        )
        prefilled = bool(resolved_verdict is not None and not prior_reviewable)
        editable = bool(review_required and not prefilled)
        if prior_reviewable:
            reviewable_prior_count += 1
        elif prefilled:
            prefilled_count += 1
            if str(resolved_verdict.get("source", "") or "").startswith("legacy_"):
                prefilled_legacy_count += 1
            else:
                prefilled_native_count += 1
        elif prior_state == "stale":
            stale_prior_count += 1
        if editable and actual_conflict:
            pending_conflict_count += 1
        elif editable and provisional:
            pending_provisional_count += 1
        row_payload = {
            **alignment[row_index],
            "editable": editable,
            "status": (
                "resolved_prior_canonical" if prefilled
                else "conflict" if actual_conflict
                else "provisional_consensus_auto" if provisional
                else "exact_consensus"
            ),
            "review_required": review_required,
            "decision_state": (
                "prior_canonical_reopened_for_review" if prior_reviewable
                else "resolved_prefilled" if prefilled
                else "pending_ai_review" if editable
                else "auto_selected_provisional_consensus" if provisional
                else "locked_exact_consensus"
            ),
            "provisional_consensus": provisional,
            "consensus_seeded_models": [
                int(value) for value in (getattr(row, "consensus_seeded_models", ()) or ())
            ],
            "base_model_texts": {model_id: texts[index] for index, model_id in enumerate(model_ids)},
            "base_row_sha256": _sha256({model_id: texts[index] for index, model_id in enumerate(model_ids)}),
            "segments": segments,
        }
        if prior_reviewable and isinstance(resolved_verdict, dict):
            row_payload["prior_decision_context"] = {
                "status": "accepted_reopened_for_review",
                "final_text": str(resolved_verdict.get("final_text", "") or ""),
                "reason": str(resolved_verdict.get("reason", "") or ""),
                "confidence": float(resolved_verdict.get("confidence", 0.0) or 0.0),
                "delete_intentionally": bool(resolved_verdict.get("delete_intentionally", False)),
                "source": str(resolved_verdict.get("source", "") or ""),
                "instruction": "这是上一轮已接受结果，仅供参考；若有更好文本可直接改写 ai_verdict.final_text。",
            }
        elif prior_state == "stale":
            row_payload["prior_decision_context"] = {
                "status": "stale_not_prefilled",
                "source": str((prior_decision or {}).get("source", "") or ""),
                "audit_flags": [str(value) for value in ((prior_decision or {}).get("audit_flags") or [])],
                "reason": "此前裁决与当前 OCR 证据或安全规则不再完全匹配，已重新进入待审队列。",
            }
        elif isinstance(prior_decision, dict) and not prefilled:
            row_payload["prior_decision_context"] = {
                "status": str(prior_decision.get("status", "") or "unresolved"),
                "source": str(prior_decision.get("source", "") or ""),
                "audit_flags": [str(value) for value in (prior_decision.get("audit_flags") or [])],
                "reason": str(prior_decision.get("reason", "") or ""),
            }
        if editable:
            row_payload["decision_mode"] = "replace_whole_column_group"
            row_payload["ai_verdict"] = {
                "decision_id": _canonical_decision_id(column_ids),
                "final_text": "",
                "reason": "",
                "confidence": 0.0,
                "delete_intentionally": False,
            }
        elif prefilled:
            row_payload["resolved_verdict"] = resolved_verdict
        rows.append(row_payload)
    payload = {
        "schema": CANONICAL_CORRECTIONS_SCHEMA,
        "package_schema": SCHEMA,
        "package_id": uuid.uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "instructions": {
            "editable_field": (
                "rows[editable=true].segments[type=editable_conflict].model_edits "
                "and/or rows[editable=true].ai_verdict.final_text"
            ),
            "editable_fields": [
                "segments[].model_edits", "segments[].reason", "segments[].confidence",
                "ai_verdict.final_text", "ai_verdict.reason", "ai_verdict.confidence",
                "ai_verdict.delete_intentionally",
            ],
            "allowed_model_ids": model_ids,
            "per_model_rule": (
                "只在错误 OCR 模型的 model_edits 中填写修正文字；正确模型必须省略。"
                "导入后这些修改只用于生成独立 AI 融合候选，不回写任何 OCR 模型，也不重新对齐。"
            ),
            "raw_ocr_rule": (
                "base_model_texts、model_texts 与 locked_consensus 永久只读；"
                "逐模型修正只能写入 model_edits，不得改写原始证据字段。"
            ),
            "locked_rule": (
                "exact_consensus 与 provisional_consensus_auto 不可修改；此前已接受的冲突裁决在本包中作为 prior_decision_context 重新开放复审。"
                if review_prior_decisions else
                "exact_consensus、provisional_consensus_auto 与 resolved_prior_canonical 均不可修改；共同候选按 v8 规则自动保留，已接受裁决会重新校验后锁定。"
            ),
            "resume_rule": (
                "处理 editable=true 的 pending_ai_review / prior_canonical_reopened_for_review；上一轮结果只是参考，可保留也可改进。"
                if review_prior_decisions else
                "只处理 editable=true 的 pending_ai_review；resolved_verdict 已完成并锁定，不得重复改写。"
            ),
            "whole_row_rule": (
                "ai_verdict.final_text 是可选的独立最终融合裁决；调序、增删、跨列差异可用它整体裁决。"
                "仅做逐模型纠错时可保持 final_text 为空。"
            ),
            "empty_rule": "只有确认整行应删除时才设置 delete_intentionally=true。",
            "do_not_change": [
                "schema", "package_schema", "package_id", "model_registry",
                "alignment_snapshot_sha256", "row_id", "row_index", "column_ids",
                "base_model_texts", "segments[].segment_id", "segments[].type",
                "segments[].model_texts", "segments[].segment_sha256",
                "decision_id", "immutable_manifest_sha256",
            ],
        },
        "book": {
            "title": str(getattr(docs[0].metadata, "title", "") or ""),
            "author": str(getattr(docs[0].metadata, "author", "") or ""),
            "language": str(getattr(docs[0].metadata, "language", "ja") or "ja"),
            "page_count": len(docs[0].pages),
        },
        "primary_structure_sha256": registry[0]["structure_sha256"],
        "primary_layout_sha256": registry[0]["layout_sha256"],
        "model_registry": registry,
        "alignment_snapshot": alignment,
        "alignment_snapshot_sha256": _sha256(alignment),
        "row_count": len(rows),
        "editable_conflict_rows": conflict_count,
        "provisional_consensus_rows": provisional_count,
        "editable_provisional_rows": 0,
        "editable_review_rows": conflict_count,
        "pending_conflict_rows": pending_conflict_count,
        "pending_provisional_rows": pending_provisional_count,
        "pending_review_rows": pending_conflict_count + pending_provisional_count,
        "prefilled_prior_decision_rows": prefilled_count,
        "prefilled_legacy_migration_rows": prefilled_legacy_count,
        "prefilled_native_decision_rows": prefilled_native_count,
        "stale_prior_decision_rows": stale_prior_count,
        "prior_decision_review_enabled": bool(review_prior_decisions),
        "prior_decision_review_rows": reviewable_prior_count,
        "locked_consensus_rows": locked_count,
        "rows": rows,
    }
    payload["immutable_manifest_sha256"] = _sha256(_immutable_projection(payload))
    return payload


def _write_jsonl(path: Path, values: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def _export_conflict_images(
    folder: Path,
    primary: UnifiedDocument,
    rows: Sequence[dict],
    *,
    progress_callback: ProgressCallback | None = None,
    max_height: int = 1600,
) -> int:
    """Export compact, lossless review evidence without touching OCR inputs.

    Only the package evidence copy is converted to grayscale and, when needed,
    proportionally downscaled.  Source scans, physical-column crops and every
    recogniser input remain byte-for-byte untouched.
    """
    try:
        from PIL import Image
    except Exception:
        return 0
    geometry = _geometry_snapshot(primary)
    pages = _page_paths(primary)
    grouped: dict[int, list[tuple[dict, list[dict]]]] = {}
    for row in rows:
        if not row.get("editable"):
            continue
        regions = [geometry.get(str(column_id)) for column_id in row.get("column_ids", [])]
        regions = [region for region in regions if isinstance(region, dict)]
        if not regions:
            continue
        page = int(regions[0].get("page", row.get("page", 0)) or 0)
        same_page = [region for region in regions if int(region.get("page", page) or page) == page]
        if same_page:
            grouped.setdefault(page, []).append((row, same_page))
    written = 0
    total_pages = max(1, len(grouped))
    max_height = max(800, int(max_height or 1600))
    for page_index, (page, page_rows) in enumerate(sorted(grouped.items()), start=1):
        _report_progress(progress_callback, "导出紧凑冲突证据", page_index, total_pages)
        source_path = Path(pages.get(page, "")).expanduser()
        if not source_path.is_file():
            continue
        try:
            with Image.open(source_path) as opened:
                image = opened.convert("L")
                for row, regions in page_rows:
                    boxes = []
                    for region in regions:
                        x, y, w, h = [float(value or 0.0) for value in region.get("bbox", [0, 0, 0, 0])]
                        boxes.append((x * image.width, y * image.height, (x + w) * image.width, (y + h) * image.height))
                    left = max(0, int(min(box[0] for box in boxes) - image.width * .012))
                    top = max(0, int(min(box[1] for box in boxes) - image.height * .012))
                    right = min(image.width, int(max(box[2] for box in boxes) + image.width * .012))
                    bottom = min(image.height, int(max(box[3] for box in boxes) + image.height * .012))
                    if right <= left or bottom <= top:
                        continue
                    crop = image.crop((left, top, right, bottom))
                    original_size = tuple(int(value) for value in crop.size)
                    scale = 1.0
                    if crop.height > max_height:
                        scale = max_height / float(crop.height)
                        resized_width = max(1, int(round(crop.width * scale)))
                        resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
                        resized = crop.resize((resized_width, max_height), resampling)
                        crop.close()
                        crop = resized
                    target = folder / "images" / f"page_{page:04d}" / f"{row['row_id'].replace(':', '_')}.png"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        crop.save(target, format="PNG", optimize=False, compress_level=6)
                        payload = target.read_bytes()
                        row["evidence_image"] = target.relative_to(folder).as_posix()
                        row["evidence_image_meta"] = {
                            "profile": "grayscale_review_max_height_v1",
                            "source_crop_size": list(original_size),
                            "exported_size": [int(crop.width), int(crop.height)],
                            "scale": round(float(scale), 8),
                            "max_height": max_height,
                            "mode": "L",
                            "format": "PNG",
                            "png_encoding_lossless": True,
                            "resampled": bool(scale < 1.0),
                            "review_copy_only": True,
                            "file_sha256": hashlib.sha256(payload).hexdigest(),
                            "file_bytes": len(payload),
                            "ocr_input_unchanged": True,
                        }
                        written += 1
                    finally:
                        crop.close()
        except Exception:
            continue
    return written



def _replace_exact_paths(value, mapping: dict[str, str]):
    """Recursively replace only exact absolute source-image paths."""
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [_replace_exact_paths(item, mapping) for item in value]
    if isinstance(value, tuple):
        return tuple(_replace_exact_paths(item, mapping) for item in value)
    if isinstance(value, dict):
        return {key: _replace_exact_paths(item, mapping) for key, item in value.items()}
    return value


def _rebind_recovery_page_images(
    documents: Sequence[UnifiedDocument], replacement_page_images: Sequence[str] | None
) -> dict:
    """Rebind restored OCR documents to pages reloaded after an application restart.

    OCR text, block IDs, column IDs and geometry stay untouched.  Only exact old
    image paths are replaced, in source-page order, so a PDF extracted into a new
    temporary directory can still drive 图文对照 without re-running OCR.
    """
    replacements = [str(Path(value).expanduser()) for value in (replacement_page_images or []) if str(value)]
    if not documents or not replacements:
        return {"requested": len(replacements), "rebound": 0, "mode": "original_paths"}
    reference_pages = list(getattr(documents[0], "pages", []) or [])
    image_indices = [index for index, page in enumerate(reference_pages) if str(getattr(page, "image_path", "") or "")]
    if len(replacements) == len(reference_pages):
        targets = list(range(len(reference_pages)))
        mode = "all_pages"
    elif len(replacements) == len(image_indices):
        targets = image_indices
        mode = "image_pages"
    else:
        return {
            "requested": len(replacements), "rebound": 0, "mode": "count_mismatch",
            "expected_all_pages": len(reference_pages), "expected_image_pages": len(image_indices),
        }
    mapping: dict[str, str] = {}
    for target_index, replacement in zip(targets, replacements):
        old = str(getattr(reference_pages[target_index], "image_path", "") or "")
        if old:
            mapping[old] = replacement
    for document in documents:
        pages = list(getattr(document, "pages", []) or [])
        for target_index, replacement in zip(targets, replacements):
            if target_index < len(pages):
                pages[target_index].image_path = replacement
        for block in getattr(document, "blocks", []) or []:
            metadata = getattr(block, "metadata", None)
            if isinstance(metadata, dict) and mapping:
                block.metadata = _replace_exact_paths(metadata, mapping)
    return {"requested": len(replacements), "rebound": len(mapping), "mode": mode}


def _write_recovery_snapshot(
    folder: Path,
    documents: Sequence[UnifiedDocument],
    labels: Sequence[str],
    *,
    package_id: str,
    fusion_selections: dict[tuple[str, ...], str] | None = None,
    fusion_selection_records: dict[tuple[str, ...], dict] | None = None,
    canonical_decisions: Sequence[dict] | None = None,
    current_row_index: int = 0,
    ruby_overlay_source: UnifiedDocument | dict | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    recovery = folder / "RECOVERY"
    recovery.mkdir(parents=True, exist_ok=True)
    model_files = []
    total = max(1, len(documents))
    for index, document in enumerate(documents):
        _report_progress(progress_callback, "保存可恢复 OCR 会话", index + 1, total)
        raw = _json_bytes(document.to_dict())
        compressed = gzip.compress(raw, compresslevel=6, mtime=0)
        relative = f"RECOVERY/model_{index + 1:02d}.json.gz"
        target = folder / relative
        target.write_bytes(compressed)
        model_files.append({
            "model_index": index,
            "label": str(labels[index] if index < len(labels) else f"OCR 模型 {index + 1}"),
            "path": relative,
            "compressed_sha256": _sha256(compressed),
            "document_sha256": _sha256(raw),
            "source_engine": str(getattr(document.metadata, "source_engine", "") or ""),
            "structure_sha256": _source_structure_hash(document),
            "layout_sha256": layout_hash(document),
        })
    selections = [
        {"column_ids": list(column_ids), "text": str(text or "")}
        for column_ids, text in sorted((fusion_selections or {}).items(), key=lambda item: item[0])
        if column_ids and str(text or "")
    ]
    selection_records = []
    for column_ids, record in sorted((fusion_selection_records or {}).items(), key=lambda item: item[0]):
        if not column_ids or not isinstance(record, dict):
            continue
        value = {
            "column_ids": list(column_ids),
            "text": str(record.get("text", "") or ""),
            "delete_intentionally": bool(record.get("delete_intentionally", False)),
            "display_label": str(record.get("display_label", "") or ""),
            "reason": str(record.get("reason", "") or ""),
            "confidence": float(record.get("confidence", 0.0) or 0.0),
            "selection_origin": str(record.get("selection_origin", "") or ""),
        }
        if value["text"] or value["delete_intentionally"]:
            selection_records.append(value)
    source_pages = []
    if documents:
        for page in getattr(documents[0], "pages", []) or []:
            path = str(getattr(page, "image_path", "") or "")
            source_pages.append({
                "page_no": int(getattr(page, "page_no", 0) or 0),
                "image_path": path,
                "file_name": Path(path).name if path else "",
            })
    ruby_overlay_file = None
    if ruby_overlay_source is not None:
        from adapters.findtext_centernet_ruby import extract_ruby_overlay
        ruby_overlay = extract_ruby_overlay(ruby_overlay_source)
        if ruby_overlay.get("blocks"):
            raw_overlay = _json_bytes(ruby_overlay)
            compressed_overlay = gzip.compress(raw_overlay, compresslevel=6, mtime=0)
            relative_overlay = "RECOVERY/ruby_overlay.json.gz"
            (folder / relative_overlay).write_bytes(compressed_overlay)
            ruby_overlay_file = {
                "path": relative_overlay,
                "compressed_sha256": _sha256(compressed_overlay),
                "document_sha256": _sha256(raw_overlay),
                "block_count": len(ruby_overlay.get("blocks") or []),
            }
    manifest = {
        "schema": RECOVERY_SCHEMA,
        "package_id": str(package_id or ""),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_count": len(model_files),
        "models": model_files,
        "fusion_selections": selections,
        "fusion_selection_records": selection_records,
        "canonical_decisions": [copy.deepcopy(item) for item in (canonical_decisions or []) if isinstance(item, dict)],
        "current_row_index": max(0, int(current_row_index or 0)),
        "source_pages": source_pages,
        "ruby_overlay": ruby_overlay_file,
        "instructions": (
            "重新打开程序后可直接选择本纠错 ZIP 恢复三份 OCR 文档；若 PDF/图片临时路径改变，"
            "先在页面管理重新载入同一批页面，程序只重绑图片路径，不重新 OCR。"
        ),
    }
    (recovery / "session_manifest.json").write_bytes(_json_bytes(manifest, pretty=True))
    return manifest


def load_source_correction_recovery(
    path: str | Path,
    *,
    replacement_page_images: Sequence[str] | None = None,
    progress_callback: ProgressCallback | None = None,
):
    """Restore a complete multi-model OCR session from a source-correction ZIP."""
    source = Path(path).expanduser()
    if not source.is_file() or source.suffix.lower() != ".zip":
        raise SourceCorrectionError("恢复多模型会话需要原始逐模型纠错 ZIP。")
    with zipfile.ZipFile(source, "r") as archive:
        _validate_source_archive(archive)
        names = set(archive.namelist())
        manifest_name = "RECOVERY/session_manifest.json"
        if manifest_name not in names:
            raise SourceCorrectionError("该纠错包没有可恢复 OCR 会话；请使用新版导出的原始纠错 ZIP。")
        try:
            manifest_info = archive.getinfo(manifest_name)
            if int(manifest_info.file_size) > 16 * 1024 * 1024:
                raise SourceCorrectionError("恢复清单异常过大。")
            manifest = json.loads(archive.read(manifest_name).decode("utf-8-sig"))
        except Exception as exc:
            raise SourceCorrectionError(f"无法读取恢复清单：{exc}") from exc
        if not isinstance(manifest, dict) or manifest.get("schema") != RECOVERY_SCHEMA:
            raise SourceCorrectionError("纠错包中的恢复清单版本不受支持。")
        model_items = manifest.get("models")
        if not isinstance(model_items, list) or not 2 <= len(model_items) <= 3:
            raise SourceCorrectionError("恢复清单必须包含 2～3 份 OCR 文档。")
        documents: list[UnifiedDocument] = []
        labels: list[str] = []
        for position, item in enumerate(model_items, start=1):
            _report_progress(progress_callback, "恢复压缩 OCR 文档", position, len(model_items))
            if not isinstance(item, dict):
                raise SourceCorrectionError("恢复清单中的模型记录无效。")
            relative = str(item.get("path", "") or "")
            if relative not in names:
                raise SourceCorrectionError(f"恢复文档缺失：{relative}")
            info = archive.getinfo(relative)
            if int(info.file_size) > 128 * 1024 * 1024:
                raise SourceCorrectionError(f"恢复文档压缩数据异常过大：{relative}")
            compressed = archive.read(relative)
            declared_size = _gzip_declared_size(compressed)
            if declared_size > _MAX_RECOVERY_DOCUMENT_BYTES:
                raise SourceCorrectionError(f"恢复文档解压大小超过安全上限：{relative}")
            if _sha256(compressed) != str(item.get("compressed_sha256", "") or ""):
                raise SourceCorrectionError(f"恢复文档压缩哈希不一致：{relative}")
            try:
                raw = gzip.decompress(compressed)
                if len(raw) > _MAX_RECOVERY_DOCUMENT_BYTES:
                    raise SourceCorrectionError(f"恢复文档解压大小超过安全上限：{relative}")
                if _sha256(raw) != str(item.get("document_sha256", "") or ""):
                    raise SourceCorrectionError(f"恢复文档内容哈希不一致：{relative}")
                document = UnifiedDocument.from_dict(json.loads(raw.decode("utf-8")))
            except SourceCorrectionError:
                raise
            except Exception as exc:
                raise SourceCorrectionError(f"无法恢复 OCR 文档 {relative}：{exc}") from exc
            document.metadata.__dict__["multi_ocr_source_correction_original_structure_sha256"] = str(
                item.get("structure_sha256", "") or ""
            )
            document.metadata.__dict__["multi_ocr_source_correction_original_layout_sha256"] = str(
                item.get("layout_sha256", "") or ""
            )
            documents.append(document)
            labels.append(str(item.get("label", "") or f"OCR 模型 {position}"))
        ruby_overlay = None
        ruby_item = manifest.get("ruby_overlay")
        if isinstance(ruby_item, dict) and ruby_item.get("path"):
            relative = str(ruby_item.get("path") or "")
            if relative not in names:
                raise SourceCorrectionError(f"恢复 Ruby 侧通道缺失：{relative}")
            compressed = archive.read(relative)
            if _sha256(compressed) != str(ruby_item.get("compressed_sha256", "") or ""):
                raise SourceCorrectionError("恢复 Ruby 侧通道压缩哈希不一致。")
            try:
                raw_overlay = gzip.decompress(compressed)
                if _sha256(raw_overlay) != str(ruby_item.get("document_sha256", "") or ""):
                    raise SourceCorrectionError("恢复 Ruby 侧通道内容哈希不一致。")
                ruby_overlay = json.loads(raw_overlay.decode("utf-8"))
            except SourceCorrectionError:
                raise
            except Exception as exc:
                raise SourceCorrectionError(f"无法恢复 Ruby 侧通道：{exc}") from exc
    rebind_report = _rebind_recovery_page_images(documents, replacement_page_images)
    _report_progress(progress_callback, "重新对齐恢复的 OCR 文档", 1, 2)
    comparison = compare_ocr_documents(documents, labels)
    _report_progress(progress_callback, "重新对齐恢复的 OCR 文档", 2, 2)
    selections = {
        tuple(str(value) for value in (item.get("column_ids") or [])): str(item.get("text", "") or "")
        for item in manifest.get("fusion_selections", []) if isinstance(item, dict)
    }
    selection_records = {
        tuple(str(value) for value in (item.get("column_ids") or [])): {
            "column_ids": list(item.get("column_ids") or []),
            "text": str(item.get("text", "") or ""),
            "delete_intentionally": bool(item.get("delete_intentionally", False)),
            "display_label": str(item.get("display_label", "") or ""),
            "reason": str(item.get("reason", "") or ""),
            "confidence": float(item.get("confidence", 0.0) or 0.0),
            "selection_origin": str(item.get("selection_origin", "") or ""),
        }
        for item in manifest.get("fusion_selection_records", []) if isinstance(item, dict) and item.get("column_ids")
    }
    # Backward compatibility: old snapshots only stored non-empty text.
    for key, text in selections.items():
        selection_records.setdefault(key, {"column_ids": list(key), "text": text})
    report = {
        "schema": "novel_formatter.multi_ocr_recovery_report.v1",
        "path": str(source),
        "package_id": str(manifest.get("package_id", "") or ""),
        "model_count": len(documents),
        "row_count": len(comparison.rows),
        "current_row_index": max(0, int(manifest.get("current_row_index", 0) or 0)),
        "fusion_selections": selections,
        "fusion_selection_records": selection_records,
        "canonical_decisions": [copy.deepcopy(item) for item in (manifest.get("canonical_decisions") or []) if isinstance(item, dict)],
        "ruby_overlay": copy.deepcopy(ruby_overlay) if isinstance(ruby_overlay, dict) else None,
        "image_rebind": rebind_report,
    }
    return documents, labels, comparison, report


def export_source_correction_bundle(
    documents: Sequence[UnifiedDocument],
    labels: Sequence[str],
    comparison: MultiOcrComparison,
    output_path: str | Path,
    *,
    progress_callback: ProgressCallback | None = None,
    include_images: bool = True,
    include_recovery_snapshot: bool = True,
    fusion_selections: dict[tuple[str, ...], str] | None = None,
    fusion_selection_records: dict[tuple[str, ...], dict] | None = None,
    canonical_decisions: Sequence[dict] | None = None,
    review_prior_decisions: bool = False,
    current_row_index: int = 0,
    ruby_overlay_source: UnifiedDocument | dict | None = None,
) -> dict:
    output = Path(output_path).expanduser()
    if output.suffix.lower() != ".zip":
        output = output.with_suffix(".zip")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_source_correction_payload(
        documents, labels, comparison,
        canonical_decisions=canonical_decisions,
        review_prior_decisions=review_prior_decisions,
        progress_callback=progress_callback,
    )
    input_audit_rows, input_audit_summary = _build_detailed_ocr_input_audit(
        documents, payload.get("model_registry") or []
    )
    input_audit_summary = {
        **input_audit_summary,
        "path": "07_ocr_input_audit.jsonl",
    }
    payload["ocr_input_audit"] = input_audit_summary
    payload["evidence_export_profile"] = {
        "name": "grayscale_review_max_height_v1",
        "max_height": 1600,
        "mode": "L",
        "format": "PNG",
        "ocr_input_unchanged": True,
    }
    with tempfile.TemporaryDirectory(prefix="nf_source_correction_") as temp:
        folder = Path(temp) / output.stem
        folder.mkdir(parents=True, exist_ok=True)
        image_count = _export_conflict_images(
            folder, documents[0], payload["rows"], progress_callback=progress_callback
        ) if include_images else 0
        _report_progress(progress_callback, "封装不可变清单", 1, 3)
        payload["immutable_manifest_sha256"] = _sha256(_immutable_projection(payload))
        manifest = {key: value for key, value in payload.items() if key not in {"rows", "alignment_snapshot"}}
        (folder / "00_manifest.json").write_bytes(_json_bytes(manifest, pretty=True))
        (folder / "01_model_registry.json").write_bytes(_json_bytes({
            "models": payload["model_registry"],
            "input_audit": input_audit_summary,
        }, pretty=True))
        _write_jsonl(folder / "07_ocr_input_audit.jsonl", input_audit_rows)
        (folder / "02_alignment_snapshot.json").write_bytes(_json_bytes({
            "alignment_snapshot_sha256": payload["alignment_snapshot_sha256"],
            "rows": payload["alignment_snapshot"],
        }))

        def summaries(editable: bool | None = None):
            for row in payload["rows"]:
                if editable is not None and bool(row["editable"]) != editable:
                    continue
                yield {
                    "row_id": row["row_id"], "row_index": row["row_index"], "page": row["page"],
                    "column_ids": row["column_ids"], "status": row["status"], "editable": row["editable"],
                    "decision_state": row.get("decision_state", ""),
                    "base_model_texts": row["base_model_texts"],
                    "evidence_image": row.get("evidence_image", ""),
                    "evidence_image_meta": row.get("evidence_image_meta", {}),
                }

        _write_jsonl(folder / "03_all_model_results.jsonl", summaries())
        _write_jsonl(
            folder / "05_locked_consensus.jsonl",
            (
                item for item in summaries(False)
                if item.get("status") == "exact_consensus"
            ),
        )
        _write_jsonl(
            folder / "06_resolved_prior_decisions.jsonl",
            (
                {
                    **item,
                    "resolved_verdict": payload["rows"][int(item["row_index"])].get("resolved_verdict", {}),
                }
                for item in summaries(False)
                if item.get("status") == "resolved_prior_canonical"
            ),
        )
        # Human-readable conflict index only.  The sole import authority remains
        # AI_OUTPUT/model_corrections.json.  Earlier builds accidentally copied
        # every full editable row into both 04 files, adding another 10–20 MB of
        # JSON before compression.  Keep both compatibility filenames, but make
        # them compact indexes that point to the single sealed authority file.
        compact_conflict_rows = [
            {
                "row_id": row["row_id"],
                "row_index": row["row_index"],
                "page": row["page"],
                "column_ids": list(row.get("column_ids") or []),
                "status": row.get("status", ""),
                "decision_state": row.get("decision_state", ""),
                "base_row_sha256": row.get("base_row_sha256", ""),
                "evidence_image": row.get("evidence_image", ""),
                "evidence_image_meta": row.get("evidence_image_meta", {}),
            }
            for row in payload["rows"]
            if row.get("editable")
        ]
        conflict_view = {
            "schema": "novel_formatter.multi_ocr_source_correction_view.v2",
            "package_id": payload["package_id"],
            "authoritative_payload": "AI_OUTPUT/model_corrections.json",
            "authority_rule": "本文件仅用于快速浏览；不得编辑或导入。所有 AI 修改只写入 authoritative_payload。",
            "evidence_profile": "true_pending_conflicts_only",
            "models": [
                {"model_id": item.get("model_id", ""), "label": item.get("display_label", "")}
                for item in payload["model_registry"]
            ],
            "editable_conflict_rows": payload["editable_conflict_rows"],
            "editable_provisional_rows": payload.get("editable_provisional_rows", 0),
            "editable_review_rows": payload.get(
                "editable_review_rows", payload["editable_conflict_rows"]
            ),
            "pending_conflict_rows": payload.get("pending_conflict_rows", 0),
            "pending_provisional_rows": payload.get("pending_provisional_rows", 0),
            "pending_review_rows": payload.get("pending_review_rows", 0),
            "prefilled_prior_decision_rows": payload.get("prefilled_prior_decision_rows", 0),
            "rows": compact_conflict_rows,
        }
        (folder / "04_editable_conflicts.json").write_bytes(_json_bytes(conflict_view))
        pending_alias = {
            "schema": "novel_formatter.multi_ocr_pending_review_alias.v1",
            "package_id": payload["package_id"],
            "authoritative_payload": "AI_OUTPUT/model_corrections.json",
            "conflict_index": "04_editable_conflicts.json",
            "pending_review_rows": payload.get("pending_review_rows", 0),
            "evidence_profile": "true_pending_conflicts_only",
        }
        (folder / "04_pending_ai_review.json").write_bytes(_json_bytes(pending_alias, pretty=True))
        ai_dir = folder / "AI_OUTPUT"
        ai_dir.mkdir(parents=True, exist_ok=True)
        (ai_dir / "model_corrections.json").write_bytes(_json_bytes(payload))
        (folder / "schemas").mkdir(parents=True, exist_ok=True)
        schema_note = {
            "schema": CANONICAL_CORRECTIONS_SCHEMA,
            "per_model_edit_path": "rows[editable=true].segments[type=editable_conflict].model_edits",
            "canonical_edit_path": "rows[editable=true].ai_verdict.final_text",
            "editable_fields": [
                "model_edits", "segment reason", "segment confidence",
                "final_text", "verdict reason", "verdict confidence", "delete_intentionally",
            ],
            "base_ocr_evidence_is_read_only": True,
            "per_model_corrections_are_applied_to_working_documents": False,
            "per_model_corrections_are_non_destructive_fusion_overlays": True,
            "original_ocr_candidates_remain_visible": True,
            "resolved_prior_path": "rows[status=resolved_prior_canonical].resolved_verdict",
            "resolved_prior_is_read_only": True,
            "validation": "all identity, base evidence and locked fields are sealed by immutable_manifest_sha256",
        }
        (folder / "schemas" / "model_corrections.schema.json").write_bytes(_json_bytes(schema_note, pretty=True))
        recovery_manifest = None
        if include_recovery_snapshot:
            recovery_manifest = _write_recovery_snapshot(
                folder, documents, labels, package_id=payload["package_id"],
                fusion_selections=fusion_selections,
                fusion_selection_records=fusion_selection_records,
                canonical_decisions=canonical_decisions,
                current_row_index=current_row_index,
                ruby_overlay_source=ruby_overlay_source,
                progress_callback=progress_callback,
            )
        readme = f"""# 多模型 OCR 逐源纠错包

本包同时支持“逐模型纠错”和“最终融合裁决”。{("此前已接受的冲突裁决会作为 prior_decision_context 重新开放复审；AI 可保留，也可提交更好的结果。" if review_prior_decisions else "此前已安全接受的最终裁决会写入 resolved_verdict 并锁定。")}
只有 `editable=true` 的行需要提交给 AI。

逐模型纠错：只在 `editable_conflict` 段的 `model_edits` 中填写错误模型的修正文字，
正确模型保持省略。导入后程序仅据此生成“AI逐模型纠错结果”融合候选，不回写任何
OCR 模型、不改变原物理列、不重新对齐；原三模型分歧继续显示并可重新选择。

最终融合裁决（可选）：在 `ai_verdict.final_text` 填写完整整行正文。仅做逐模型纠错时
可保持 final_text 为空。调序、漏句、重复、跨列差异可使用 final_text 整体裁决。

证据图与输入审计：`images/` 是只供 AI/人工查看的无损灰度 PNG；超过 1600px 时按比例缩小。
这不会修改扫描原图、共享物理列图或任何模型输入。每行 `evidence_image_meta` 记录原尺寸、
导出尺寸、缩放率和文件 SHA-256。`07_ocr_input_audit.jsonl` 记录各模型稳定物理列的实际
输入哈希、传输方式与共享关系；旧会话缺少哈希时明确标记 unavailable，不会猜测。

不得修改 `base_model_texts`、`model_texts`、锁定一致段、模型 ID、物理列 ID 或哈希字段。

包 ID：`{payload['package_id']}`  
真正分歧总数：{payload['editable_conflict_rows']}  
两模型共同候选（按 v8 自动保留）：{payload.get('provisional_consensus_rows', 0)}  
此前已完成并锁定：{payload.get('prefilled_prior_decision_rows', 0)}  
此前结果重新开放复审：{payload.get('prior_decision_review_rows', 0)}
本轮真正分歧待审：{payload.get('pending_conflict_rows', 0)}  
本轮共同候选待审：{payload.get('pending_provisional_rows', 0)}  
本轮合计待审：{payload.get('pending_review_rows', 0)}  
过期裁决重新待审：{payload.get('stale_prior_decision_rows', 0)}  
锁定一致行：{payload['locked_consensus_rows']}  
视觉证据：{image_count} 张  
可恢复 OCR 会话：{"是" if recovery_manifest else "否"}

程序崩溃或重启后，可在 OCR 对比页选择“恢复纠错会话”，直接载入本 ZIP。无需重新 OCR。
"""
        (folder / "README_AI.md").write_text(readme, encoding="utf-8")
        _report_progress(progress_callback, "压缩逐源纠错包", 2, 3)
        temp_zip = output.with_name(f".{output.name}.tmp")
        try:
            with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
                for path in sorted(folder.rglob("*")):
                    if not path.is_file():
                        continue
                    compression = zipfile.ZIP_STORED if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gz"} else zipfile.ZIP_DEFLATED
                    archive.write(path, path.relative_to(folder).as_posix(), compress_type=compression)
            os.replace(temp_zip, output)
        finally:
            if temp_zip.exists():
                temp_zip.unlink(missing_ok=True)
        _report_progress(progress_callback, "压缩逐源纠错包", 3, 3)
    return {
        "path": str(output),
        "package_id": payload["package_id"],
        "row_count": payload["row_count"],
        "editable_conflict_rows": payload["editable_conflict_rows"],
        "provisional_consensus_rows": payload.get("provisional_consensus_rows", 0),
        "editable_provisional_rows": payload.get("editable_provisional_rows", 0),
        "editable_review_rows": payload.get("editable_review_rows", payload["editable_conflict_rows"]),
        "pending_conflict_rows": payload.get("pending_conflict_rows", 0),
        "pending_provisional_rows": payload.get("pending_provisional_rows", 0),
        "pending_review_rows": payload.get("pending_review_rows", 0),
        "prefilled_prior_decision_rows": payload.get("prefilled_prior_decision_rows", 0),
        "prefilled_legacy_migration_rows": payload.get("prefilled_legacy_migration_rows", 0),
        "prefilled_native_decision_rows": payload.get("prefilled_native_decision_rows", 0),
        "stale_prior_decision_rows": payload.get("stale_prior_decision_rows", 0),
        "prior_decision_review_enabled": bool(payload.get("prior_decision_review_enabled", False)),
        "prior_decision_review_rows": payload.get("prior_decision_review_rows", 0),
        "locked_consensus_rows": payload["locked_consensus_rows"],
        "image_count": image_count,
        "evidence_export_profile": payload.get("evidence_export_profile", {}),
        "ocr_input_audit": input_audit_summary,
        "immutable_manifest_sha256": payload["immutable_manifest_sha256"],
        "recovery_snapshot_included": bool(include_recovery_snapshot),
        "recovery_model_count": len(documents) if include_recovery_snapshot else 0,
    }


def load_correction_payload(path: str | Path) -> dict:
    source = Path(path).expanduser()
    if not source.is_file():
        raise SourceCorrectionError(f"纠错文件不存在：{source}")
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source, "r") as archive:
            _validate_source_archive(archive)
            names = set(archive.namelist())
            target = next((name for name in (
                "AI_OUTPUT/model_corrections.json",
                "04_editable_conflicts.json",
                "model_corrections.json",
            ) if name in names), None)
            if target is None:
                raise SourceCorrectionError("ZIP 中没有 model_corrections.json。")
            if int(archive.getinfo(target).file_size) > _MAX_CORRECTION_JSON_BYTES:
                raise SourceCorrectionError("model_corrections.json 超过安全大小上限。")
            raw = archive.read(target)
    else:
        if source.stat().st_size > _MAX_CORRECTION_JSON_BYTES:
            raise SourceCorrectionError("纠错 JSON 超过安全大小上限。")
        raw = source.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8-sig"))
    except Exception as exc:
        raise SourceCorrectionError(f"无法解析纠错 JSON：{exc}") from exc
    if not isinstance(payload, dict):
        raise SourceCorrectionError("纠错结果顶层必须是 JSON 对象。")
    return payload


def _validate_payload(payload: dict) -> None:
    schema = str(payload.get("schema", "") or "")
    if schema not in {CORRECTIONS_SCHEMA, *SUPPORTED_CANONICAL_CORRECTIONS_SCHEMAS}:
        raise SourceCorrectionError("不是受支持的多模型 OCR 裁决结果。")
    expected = str(payload.get("immutable_manifest_sha256", "") or "")
    actual = _sha256(_immutable_projection(payload))
    if not expected or expected != actual:
        raise SourceCorrectionError("不可变清单已被修改；原始 OCR、模型身份或物理列映射不可信。")
    rows = payload.get("rows")
    registry = payload.get("model_registry")
    if not isinstance(rows, list) or not isinstance(registry, list):
        raise SourceCorrectionError("裁决结果缺少 rows 或 model_registry。")
    model_ids = [str(item.get("model_id", "") or "") for item in registry if isinstance(item, dict)]
    if not model_ids or len(set(model_ids)) != len(model_ids):
        raise SourceCorrectionError("模型 ID 缺失或重复。")
    seen_rows: set[str] = set()
    seen_segments: set[str] = set()
    seen_decisions: set[str] = set()
    for expected_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SourceCorrectionError("rows 中存在非对象条目。")
        row_id = str(row.get("row_id", "") or "")
        if not row_id or row_id in seen_rows or int(row.get("row_index", -1)) != expected_index:
            raise SourceCorrectionError("行 ID 重复、缺失或顺序已改变。")
        seen_rows.add(row_id)
        editable = bool(row.get("editable", False))
        if schema in SUPPORTED_CANONICAL_CORRECTIONS_SCHEMAS:
            verdict = row.get("ai_verdict")
            resolved_verdict = row.get("resolved_verdict")
            if editable:
                if not isinstance(verdict, dict):
                    raise SourceCorrectionError(f"{row_id} 缺少 ai_verdict。")
                if resolved_verdict not in (None, {}):
                    raise SourceCorrectionError(f"待审行 {row_id} 不允许同时包含 resolved_verdict。")
                decision_id = str(verdict.get("decision_id", "") or "")
                if not decision_id or decision_id in seen_decisions:
                    raise SourceCorrectionError("AI 裁决 ID 缺失或重复。")
                seen_decisions.add(decision_id)
                for key in ("final_text", "reason"):
                    if not isinstance(verdict.get(key, ""), str):
                        raise SourceCorrectionError(f"{row_id} 的 ai_verdict.{key} 必须是字符串。")
                try:
                    float(verdict.get("confidence", 0.0) or 0.0)
                except (TypeError, ValueError):
                    raise SourceCorrectionError(f"{row_id} 的 ai_verdict.confidence 必须是数字。")
            else:
                if verdict not in (None, {}):
                    raise SourceCorrectionError(f"锁定行 {row_id} 不允许出现可编辑 ai_verdict。")
                status = str(row.get("status", "") or "")
                if status == "resolved_prior_canonical":
                    if schema != CANONICAL_CORRECTIONS_SCHEMA:
                        raise SourceCorrectionError("resolved_prior_canonical 只允许出现在 V3 裁决包。")
                    if not isinstance(resolved_verdict, dict):
                        raise SourceCorrectionError(f"已完成行 {row_id} 缺少 resolved_verdict。")
                    decision_id = str(resolved_verdict.get("decision_id", "") or "")
                    if not decision_id or decision_id in seen_decisions:
                        raise SourceCorrectionError("已完成裁决 ID 缺失或重复。")
                    seen_decisions.add(decision_id)
                    final_text = resolved_verdict.get("final_text", "")
                    delete_intentionally = bool(resolved_verdict.get("delete_intentionally", False))
                    if not isinstance(final_text, str) or (not final_text and not delete_intentionally):
                        raise SourceCorrectionError(f"已完成行 {row_id} 的 resolved_verdict 无有效正文。")
                    if _contains_placeholder(final_text) or _suspicious_inline_latin(final_text):
                        raise SourceCorrectionError(f"已完成行 {row_id} 的 resolved_verdict 未通过文字安全检查。")
                    try:
                        float(resolved_verdict.get("confidence", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        raise SourceCorrectionError(f"{row_id} 的 resolved_verdict.confidence 必须是数字。")
                elif resolved_verdict not in (None, {}):
                    raise SourceCorrectionError(f"锁定一致行 {row_id} 不允许出现 resolved_verdict。")
        for segment in row.get("segments", []) or []:
            if not isinstance(segment, dict):
                raise SourceCorrectionError(f"{row_id} 的段不是对象。")
            segment_id = str(segment.get("segment_id", "") or "")
            if not segment_id or segment_id in seen_segments:
                raise SourceCorrectionError("冲突段 ID 缺失或重复。")
            seen_segments.add(segment_id)
            segment_type = str(segment.get("type", "") or "")
            if segment_type == "locked_consensus":
                if "model_edits" in segment:
                    raise SourceCorrectionError(f"锁定一致段 {segment_id} 出现 model_edits。")
            elif segment_type == "editable_conflict":
                base = segment.get("model_texts")
                edits = segment.get("model_edits", {})
                if not isinstance(base, dict) or not isinstance(edits, dict):
                    raise SourceCorrectionError(f"冲突段 {segment_id} 的 model_texts/model_edits 格式错误。")
                if edits and not editable:
                    raise SourceCorrectionError(f"锁定行 {row_id} 的 {segment_id} 不允许填写 model_edits。")
                unknown = set(str(key) for key in edits).difference(base)
                if unknown:
                    raise SourceCorrectionError(f"冲突段 {segment_id} 包含未知模型：{sorted(unknown)}")
                for key, value in edits.items():
                    if not isinstance(value, str):
                        raise SourceCorrectionError(f"{segment_id} 的 {key} 修改值必须是字符串。")
            else:
                raise SourceCorrectionError(f"未知冲突段类型：{segment_type}")

def _current_registry_map(documents: Sequence[UnifiedDocument], labels: Sequence[str]) -> dict[str, dict]:
    registry, _snapshots = _build_model_registry_and_snapshots(documents, labels)
    return {item["model_id"]: item for item in registry}


def _row_text_from_columns(doc: UnifiedDocument, column_ids: Sequence[str]) -> str:
    snapshot, _source = physical_column_text_snapshot(doc)
    return join_column_parts(snapshot.get(str(column_id), "") for column_id in column_ids)


def _incoming_model_text(row: dict, model_id: str) -> str:
    output: list[str] = []
    for segment in row.get("segments", []) or []:
        if segment.get("type") == "locked_consensus":
            output.append(str(segment.get("consensus_text", "") or ""))
        else:
            base = segment.get("model_texts") or {}
            edits = segment.get("model_edits") or {}
            output.append(str(edits[model_id] if model_id in edits else base.get(model_id, "") or ""))
    return "".join(output)


def _apply_column_updates(doc: UnifiedDocument, updates: dict[str, str], *, audit: dict) -> int:
    changed = 0
    seen: set[str] = set()
    for block in doc.blocks:
        if block.type not in _TEXT_TYPES:
            continue
        metadata = _metadata(block)
        ids = _column_ids(metadata)
        if not ids:
            continue
        texts = _column_texts(metadata, ids, block.text)
        if len(texts) != len(ids):
            continue
        new_texts = list(texts)
        block_changed = False
        for position, column_id in enumerate(ids):
            if column_id not in updates:
                continue
            seen.add(column_id)
            value = str(updates[column_id] or "")
            if new_texts[position] != value:
                new_texts[position] = value
                changed += 1
                block_changed = True
        if not block_changed:
            continue
        metadata.setdefault("source_column_original_texts", list(texts))
        metadata["source_column_primary_texts"] = list(new_texts)
        metadata["source_column_texts"] = list(new_texts)
        raw_flags = metadata.get("source_column_terminal_flags")
        if isinstance(raw_flags, list) and len(raw_flags) == len(new_texts):
            metadata["source_column_terminal_flags"] = [has_sentence_terminal(value) for value in new_texts]
        block.text = join_column_parts(new_texts)
        block.modified_by = (str(block.modified_by or "") + ",external_ai_source_correction").strip(",")
        metadata.setdefault("multi_ocr_source_correction_audit", []).append(copy.deepcopy(audit))
    missing = set(updates).difference(seen)
    if missing:
        raise SourceCorrectionError(f"目标 OCR 文档缺少 {len(missing)} 个物理列，已取消导入。")
    return changed


def _exported_alignment_and_columns(payload: dict) -> tuple[list[dict], list[str]]:
    exported_alignment = payload.get("alignment_snapshot") or []
    if not isinstance(exported_alignment, list):
        raise SourceCorrectionError("导出时的对齐快照格式错误。")
    if _sha256(exported_alignment) != str(payload.get("alignment_snapshot_sha256", "") or ""):
        raise SourceCorrectionError("导出时的对齐快照已损坏。")
    ordered_columns: list[str] = []
    seen: set[str] = set()
    for row in exported_alignment:
        if not isinstance(row, dict):
            raise SourceCorrectionError("导出时的对齐快照包含非对象条目。")
        ids = [str(value) for value in (row.get("column_ids") or []) if str(value)]
        if not ids:
            raise SourceCorrectionError("导出时的对齐快照包含没有物理列 ID 的行。")
        duplicated = seen.intersection(ids)
        if duplicated:
            raise SourceCorrectionError(f"纠错包物理列重复：{sorted(duplicated)[:3]}")
        seen.update(ids)
        ordered_columns.extend(ids)
    return exported_alignment, ordered_columns


def _model_text_compatibility_score(
    rows: Sequence[dict], snapshot: dict[str, str], model_id: str, *, max_samples: int = 384,
) -> float:
    """Score whether a current OCR source is the same textual session.

    Geometry hashes are intentionally not used here: block regrouping, restored
    image paths, review metadata and harmless bbox refinements may all alter a
    document layout hash without changing the stable physical-column identity.
    A wrong book with coincidentally similar column IDs is rejected by comparing
    current row text against both the exported base and the incoming corrected
    text on an evenly distributed sample.
    """
    values = [row for row in rows if isinstance(row, dict)]
    if not values:
        return 0.0
    step = max(1, len(values) // max(1, int(max_samples)))
    sampled = values[::step][:max_samples]
    weighted_score = 0.0
    total_weight = 0.0
    for row in sampled:
        column_ids = [str(value) for value in (row.get("column_ids") or []) if str(value)]
        current = join_column_parts(snapshot.get(column_id, "") for column_id in column_ids)
        base = str((row.get("base_model_texts") or {}).get(model_id, "") or "")
        incoming = _incoming_model_text(row, model_id)
        weight = float(max(1, len(current), len(base), len(incoming)))
        if current == base or current == incoming:
            score = 1.0
        else:
            score = max(
                SequenceMatcher(None, current, base, autojunk=False).ratio(),
                SequenceMatcher(None, current, incoming, autojunk=False).ratio(),
            )
        weighted_score += score * weight
        total_weight += weight
    return weighted_score / total_weight if total_weight else 0.0


def _row_has_explicit_model_edits(row: dict) -> bool:
    for segment in row.get("segments", []) or []:
        if isinstance(segment, dict) and segment.get("type") == "editable_conflict":
            edits = segment.get("model_edits") or {}
            if isinstance(edits, dict) and edits:
                return True
    return False


def _canonical_verdict_has_output(row: dict) -> bool:
    verdict = row.get("ai_verdict") or row.get("resolved_verdict") or {}
    return bool(
        isinstance(verdict, dict)
        and (str(verdict.get("final_text", "") or "") or bool(verdict.get("delete_intentionally", False)))
    )


def _derive_per_model_correction_decision(row: dict, model_ids: Sequence[str]) -> dict:
    """Turn a complete sparse per-model correction into a resolved row decision.

    The source-correction contract says every wrong model is corrected and every
    correct model is left unchanged.  Therefore an edited row is finished when
    the effective full-row text of all model slots converges exactly.  Treating
    that result as unresolved forced users to confirm the same sentence again
    and discarded the only complete fusion output.
    """
    column_ids = [str(value) for value in (row.get("column_ids") or []) if str(value)]
    corrected = {model_id: _incoming_model_text(row, model_id) for model_id in model_ids}
    values = [str(corrected.get(model_id, "") or "") for model_id in model_ids]
    unique = set(values)
    final_text = values[0] if len(unique) == 1 and values else ""
    flags: list[str] = []
    if len(unique) != 1:
        flags.append("per_model_edits_did_not_converge")
    if final_text and _contains_placeholder(final_text):
        flags.append("final_text_contains_placeholder")
    if final_text and _suspicious_inline_latin(final_text):
        flags.append("final_text_contains_suspicious_inline_latin")
    accepted = bool(final_text and not flags)
    reasons: list[str] = []
    confidences: list[float] = []
    for segment in row.get("segments", []) or []:
        if not isinstance(segment, dict) or segment.get("type") != "editable_conflict":
            continue
        if not isinstance(segment.get("model_edits"), dict) or not segment.get("model_edits"):
            continue
        reason = str(segment.get("reason", "") or "").strip()
        if reason and reason not in reasons:
            reasons.append(reason)
        try:
            confidence = float(segment.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError, OverflowError):
            confidence = 0.0
        if confidence > 0:
            confidences.append(confidence)
    historical = {
        model_id: str((row.get("base_model_texts") or {}).get(model_id, "") or "")
        for model_id in model_ids
    }
    historical_values = [historical[model_id] for model_id in model_ids]
    return {
        "decision_id": _canonical_decision_id(column_ids),
        "row_id": str(row.get("row_id", "") or ""),
        "row_index": int(row.get("row_index", 0) or 0),
        "column_ids": column_ids,
        "final_text": final_text if accepted else "",
        "status": "accepted" if accepted else "unresolved",
        "source": "ai_per_model_source_correction_v3",
        "derivation": "all_model_texts_converged_after_sparse_model_edits",
        "resolution_kind": "per_model_source_correction",
        "confidence": min(confidences) if confidences else (1.0 if accepted else 0.0),
        "reason": "；".join(reasons) or (
            "逐模型纠错后所有 OCR 模型在该物理列组完全一致。"
            if accepted else "逐模型纠错后模型文字仍未完全一致。"
        ),
        "audit_flags": flags,
        # Re-export compatibility must compare against the *current corrected*
        # OCR evidence, while the original disagreement remains separately sealed.
        "raw_model_texts": corrected,
        "raw_model_texts_by_index": values,
        "historical_raw_model_texts": historical,
        "historical_raw_model_texts_by_index": historical_values,
        "historical_disagreement": len(set(historical_values)) > 1,
        "delete_intentionally": False,
    }


def _annotate_imported_correction_history(
    comparison: MultiOcrComparison,
    payload_rows: Sequence[dict],
    decisions: Sequence[dict],
    labels: Sequence[str],
) -> int:
    """Attach resolved pre-correction candidates to stable comparison rows."""
    payload_by_columns = {
        tuple(str(value) for value in (row.get("column_ids") or []) if str(value)): row
        for row in payload_rows if isinstance(row, dict) and row.get("column_ids")
    }
    decision_by_columns = {
        tuple(str(value) for value in (item.get("column_ids") or []) if str(value)): item
        for item in decisions if isinstance(item, dict) and item.get("column_ids")
    }
    annotated = 0
    for current_row in comparison.rows:
        key = tuple(str(value) for value in (current_row.column_ids or ()) if str(value))
        decision = decision_by_columns.get(key)
        payload_row = payload_by_columns.get(key)
        if not isinstance(decision, dict) or str(decision.get("status", "") or "") != "accepted":
            continue
        history_map = decision.get("historical_raw_model_texts") or {}
        history_indexed = decision.get("historical_raw_model_texts_by_index") or []
        if isinstance(history_indexed, list) and history_indexed:
            history = tuple(str(value or "") for value in history_indexed)
        elif isinstance(history_map, dict) and history_map:
            history = tuple(str(value or "") for value in history_map.values())
        elif isinstance(payload_row, dict) and _row_has_explicit_model_edits(payload_row):
            history = tuple(
                str(value or "") for value in (payload_row.get("base_model_texts") or {}).values()
            )
        else:
            history = ()
        if not history:
            continue
        current_row.source_correction_resolved = True
        current_row.historical_ocr_texts = history
        current_row.historical_ocr_labels = tuple(
            str(labels[index] if index < len(labels) else f"模型{index + 1}")
            for index in range(len(history))
        )
        current_row.historical_ocr_disagreement = bool(
            decision.get("historical_disagreement", len(set(history)) > 1)
        )
        current_row.historical_resolution_reason = str(decision.get("reason", "") or "")
        current_row.historical_resolution_confidence = float(decision.get("confidence", 0.0) or 0.0)
        annotated += 1
    return annotated


def _collect_per_model_updates(
    rows: Sequence[dict],
    model_ids: Sequence[str],
    snapshot_by_model: dict[str, dict[str, str]],
    expected_column_set: set[str],
) -> tuple[dict[str, dict[str, str]], dict]:
    """Build sparse, transaction-safe per-model column updates.

    The merge contract is deliberately conservative.  An incoming model edit is
    applied when the current row still equals the exported baseline, ignored if
    it is already present locally, and reported as a conflict when both local
    and AI changed the same model row differently.
    """
    updates_by_model: dict[str, dict[str, str]] = {model_id: {} for model_id in model_ids}
    requested_values = 0
    requested_model_rows = 0
    already_applied_model_rows = 0
    applied_model_rows = 0
    conflicts: list[dict] = []
    affected_groups: set[tuple[str, ...]] = set()
    corrected_rows: set[str] = set()

    for row in rows:
        if not isinstance(row, dict) or not _row_has_explicit_model_edits(row):
            continue
        row_id = str(row.get("row_id", "") or "")
        column_ids = tuple(str(value) for value in (row.get("column_ids") or []) if str(value))
        if not column_ids or not set(column_ids).issubset(expected_column_set):
            raise SourceCorrectionError(f"{row_id} 的逐模型修改缺少有效物理列映射。")
        base_model_texts = row.get("base_model_texts") or {}
        for model_id in model_ids:
            explicit_count = 0
            for segment in row.get("segments", []) or []:
                if not isinstance(segment, dict) or segment.get("type") != "editable_conflict":
                    continue
                edits = segment.get("model_edits") or {}
                if isinstance(edits, dict) and model_id in edits:
                    explicit_count += 1
            if not explicit_count:
                continue
            requested_values += explicit_count
            requested_model_rows += 1
            baseline = str(base_model_texts.get(model_id, "") or "")
            incoming = _incoming_model_text(row, model_id)
            snapshot = snapshot_by_model[model_id]
            current_parts = [str(snapshot.get(column_id, "") or "") for column_id in column_ids]
            local = join_column_parts(current_parts)

            if incoming == baseline:
                # Explicit no-op edits are harmless but should not create audit noise.
                continue
            if local == incoming:
                already_applied_model_rows += 1
                affected_groups.add(column_ids)
                corrected_rows.add(row_id)
                continue
            if local != baseline:
                conflicts.append({
                    "row_id": row_id,
                    "row_index": int(row.get("row_index", 0) or 0),
                    "column_ids": list(column_ids),
                    "model_id": model_id,
                    "baseline_text": baseline,
                    "local_text": local,
                    "incoming_text": incoming,
                    "reason": "本地 OCR 与 AI 在导出后对同一模型行产生了不同修改；已保留本地文字。",
                })
                continue

            projected = project_fused_text_to_physical_columns(
                incoming, current_parts, column_count=len(column_ids),
            )
            if len(projected) != len(column_ids):
                raise SourceCorrectionError(f"{row_id} 的逐模型修改无法投影回物理列。")
            changed_this_row = False
            model_updates = updates_by_model[model_id]
            for column_id, value in zip(column_ids, projected):
                value = str(value or "")
                previous = model_updates.get(column_id)
                if previous is not None and previous != value:
                    raise SourceCorrectionError(
                        f"同一模型物理列收到互相冲突的逐源修改：{model_id} / {column_id}"
                    )
                if snapshot.get(column_id, "") != value:
                    model_updates[column_id] = value
                    changed_this_row = True
            if changed_this_row:
                applied_model_rows += 1
                affected_groups.add(column_ids)
                corrected_rows.add(row_id)

    return updates_by_model, {
        "requested_model_edit_values": requested_values,
        "requested_model_rows": requested_model_rows,
        "already_applied_model_rows": already_applied_model_rows,
        "applied_model_rows": applied_model_rows,
        "merge_conflicts": conflicts,
        "affected_column_groups": [list(value) for value in sorted(affected_groups)],
        "corrected_row_ids": sorted(corrected_rows),
    }


def import_source_corrections(
    source: str | Path | dict,
    current_documents: Sequence[UnifiedDocument],
    labels: Sequence[str],
    current_comparison: MultiOcrComparison,
    *,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[UnifiedDocument], MultiOcrComparison, dict]:
    """Import AI corrections as a non-destructive fusion overlay.

    ``model_edits`` are validated against stable model/column identities and are
    used to derive an AI correction candidate, but they never rewrite any OCR
    document, comparison row, physical-column text, or alignment.  The original
    multi-model disagreement therefore remains visible and selectable beside the
    AI result.  Whole-row ``ai_verdict`` values use the same overlay channel.
    """
    payload = source if isinstance(source, dict) else load_correction_payload(source)
    _report_progress(progress_callback, "校验裁决包与不可变清单", 1, 6)
    _validate_payload(payload)
    schema = str(payload.get("schema", "") or "")
    docs = list(current_documents)
    if not 2 <= len(docs) <= 3:
        raise SourceCorrectionError("当前 OCR 对比没有 2～3 个模型。")

    registry_list, snapshots = _build_model_registry_and_snapshots(docs, labels)
    exported_registry_list = [
        item for item in (payload.get("model_registry") or []) if isinstance(item, dict)
    ]
    if len(exported_registry_list) != len(docs):
        raise SourceCorrectionError("当前 OCR 模型数量与裁决包不一致。")

    exported_registry: dict[str, dict] = {}
    snapshot_by_model: dict[str, dict[str, str]] = {}
    model_index_by_id: dict[str, int] = {}
    layout_matches: dict[str, bool] = {}
    structure_matches: dict[str, bool] = {}
    snapshot_matches: dict[str, bool] = {}
    for exported_item in exported_registry_list:
        model_id = str(exported_item.get("model_id", "") or "")
        index = int(exported_item.get("model_index", -1))
        if not model_id or model_id in exported_registry:
            raise SourceCorrectionError("裁决包模型 ID 缺失或重复。")
        if not 0 <= index < len(docs):
            raise SourceCorrectionError(f"模型索引越界：{model_id}")
        current_item = registry_list[index]
        current_engine = str(current_item.get("source_engine", "") or "")
        exported_engine = str(exported_item.get("source_engine", "") or "")
        if current_engine != exported_engine:
            raise SourceCorrectionError(
                f"OCR 模型槽位不一致：第 {index + 1} 个模型当前为 {current_engine}，"
                f"裁决包要求 {exported_engine}。"
            )
        exported_registry[model_id] = exported_item
        model_index_by_id[model_id] = index
        snapshot_by_model[model_id] = snapshots[index]
        layout_matches[model_id] = (
            str(current_item.get("layout_sha256", "") or "")
            == str(exported_item.get("layout_sha256", "") or "")
        )
        structure_matches[model_id] = (
            str(current_item.get("structure_sha256", "") or "")
            == str(exported_item.get("structure_sha256", "") or "")
        )
        snapshot_matches[model_id] = (
            str(current_item.get("document_snapshot_sha256", "") or "")
            == str(exported_item.get("document_snapshot_sha256", "") or "")
        )

    exported_alignment, expected_column_order = _exported_alignment_and_columns(payload)
    expected_column_set = set(expected_column_order)
    expected_count = len(expected_column_order)
    if not expected_count:
        raise SourceCorrectionError("裁决包没有物理列。")

    rows = payload.get("rows", [])
    compatibility_scores: dict[str, float] = {}
    for model_id, exported_item in exported_registry.items():
        snapshot = snapshot_by_model[model_id]
        current_columns = set(snapshot)
        missing = expected_column_set.difference(current_columns)
        extra = current_columns.difference(expected_column_set)
        exported_count = int(exported_item.get("physical_column_count", -1))
        if exported_count != expected_count:
            raise SourceCorrectionError(f"裁决包模型 {model_id} 的物理列数量与自身快照不一致。")
        if missing or extra:
            detail = []
            if missing:
                detail.append(f"缺少 {len(missing)} 列（如 {sorted(missing)[:3]}）")
            if extra:
                detail.append(f"多出 {len(extra)} 列（如 {sorted(extra)[:3]}）")
            raise SourceCorrectionError(
                f"当前 OCR 模型 {model_id} 的稳定物理列 ID 与裁决包不一致：" + "；".join(detail)
            )
        score = _model_text_compatibility_score(rows, snapshot, model_id)
        compatibility_scores[model_id] = score
        if score < 0.55:
            raise SourceCorrectionError(
                f"当前 OCR 模型 {model_id} 与裁决包的文本兼容度仅 {score:.1%}，"
                "疑似不是同一次 OCR 或不是同一本书；已取消导入。"
            )

    current_alignment = _alignment_snapshot(current_comparison)
    current_alignment_matches = current_alignment == exported_alignment
    model_ids = [str(item.get("model_id", "") or "") for item in exported_registry_list]

    before = {
        "exact_rows": int(current_comparison.exact_rows),
        "provisional_consensus_rows": int(getattr(current_comparison, "provisional_consensus_rows", 0) or 0),
        "conflict_rows": int(current_comparison.conflict_rows),
        "low_confidence_rows": int(current_comparison.low_confidence_rows),
        "row_count": len(current_comparison.rows),
    }

    _report_progress(progress_callback, "校验逐模型稀疏修改（不回写 OCR）", 2, 6)
    updates_by_model, source_stats = _collect_per_model_updates(
        rows, model_ids, snapshot_by_model, expected_column_set,
    )
    audit_id = uuid.uuid4().hex
    # Non-destructive contract: preserve object identity as well as text.  The
    # calculated sparse updates are audit/proposal data only and are never
    # projected into model documents or followed by an expensive realignment.
    output_docs = docs
    refreshed_comparison = current_comparison
    proposed_model_cells = sum(len(updates) for updates in updates_by_model.values())
    proposed_model_ids = [model_id for model_id, updates in updates_by_model.items() if updates]
    touched_model_ids: list[str] = []
    changed_model_cells = 0
    _report_progress(progress_callback, "保留原始 OCR 与现有对齐", 4, 6)

    decisions: list[dict] = []
    accepted_count = 0
    unresolved_count = 0
    rejected_placeholder = 0
    rejected_ascii = 0
    legacy_complex = 0
    legacy_migrated_accepted = 0
    native_accepted = 0
    prefilled_resolved = 0
    normalized_legacy_provisional_rows = 0
    total_rows = max(1, len(rows))
    for row_index, row in enumerate(rows):
        if row_index == 0 or (row_index + 1) % 100 == 0 or row_index + 1 == total_rows:
            _report_progress(progress_callback, "生成可选最终融合裁决", row_index + 1, total_rows)
        if not isinstance(row, dict):
            continue
        is_editable = bool(row.get("editable", False))
        is_prefilled = str(row.get("status", "") or "") == "resolved_prior_canonical"
        if not is_editable and not is_prefilled:
            continue
        # V23/V24 exported quick-consensus rows as editable pending work even
        # when no AI/model output was provided.  Under v8 semantics those blank
        # legacy rows are automatically retained and must not become thousands
        # of unresolved decisions merely because the old package flag says
        # editable=true.  Explicit model_edits/final_text are still honoured.
        if (
            not is_prefilled
            and str(row.get("status", "") or "") in {
                "provisional_consensus", "provisional_consensus_auto"
            }
            and bool(row.get("provisional_consensus", False))
            and not _row_has_explicit_model_edits(row)
            and not _canonical_verdict_has_output(row)
        ):
            normalized_legacy_provisional_rows += 1
            continue
        column_ids = [str(value) for value in (row.get("column_ids") or []) if str(value)]
        if not column_ids or not set(column_ids).issubset(expected_column_set):
            raise SourceCorrectionError(f"{row.get('row_id', '')} 的稳定物理列映射无效。")
        # A complete V3 per-model correction is itself a finished adjudication
        # when all effective model texts converge.  Do not throw that state away
        # merely because the optional whole-row ai_verdict was intentionally blank.
        if (
            schema in SUPPORTED_CANONICAL_CORRECTIONS_SCHEMAS
            and _row_has_explicit_model_edits(row)
            and not _canonical_verdict_has_output(row)
        ):
            decision = _derive_per_model_correction_decision(row, model_ids)
        else:
            decision = _read_canonical_verdict(row, model_ids, schema)
        flags = set(str(value) for value in (decision.get("audit_flags") or []))
        if "final_text_contains_placeholder" in flags or "derived_verdict_failed_text_safety" in flags:
            rejected_placeholder += 1
        if "final_text_contains_suspicious_inline_latin" in flags:
            rejected_ascii += 1
        if "legacy_complex_alignment_requires_whole_row_verdict" in flags:
            legacy_complex += 1
        if decision.get("status") == "accepted":
            accepted_count += 1
            if is_prefilled:
                prefilled_resolved += 1
            if str(decision.get("source", "") or "").startswith("legacy_"):
                legacy_migrated_accepted += 1
            else:
                native_accepted += 1
        else:
            unresolved_count += 1
        decisions.append(decision)

    # Count every accepted row whose sealed export-time OCR evidence disagreed.
    # This remains accurate even when the current session came from an older
    # destructive import that had already collapsed the live comparison texts.
    historical_rows_annotated = 0
    for decision in decisions:
        if str(decision.get("status", "") or "") != "accepted":
            continue
        original_values = [
            str(value or "")
            for value in (
                decision.get("historical_raw_model_texts_by_index")
                or decision.get("raw_model_texts_by_index")
                or []
            )
        ]
        if bool(decision.get("historical_disagreement", False)) or (
            original_values and len(set(original_values)) > 1
        ):
            historical_rows_annotated += 1

    after = {
        "exact_rows": int(refreshed_comparison.exact_rows),
        "provisional_consensus_rows": int(getattr(refreshed_comparison, "provisional_consensus_rows", 0) or 0),
        "conflict_rows": int(refreshed_comparison.conflict_rows),
        "low_confidence_rows": int(refreshed_comparison.low_confidence_rows),
        "row_count": len(refreshed_comparison.rows),
    }
    all_snapshot_match = all(snapshot_matches.values())
    all_layout_match = all(layout_matches.values())
    if all_snapshot_match and current_alignment_matches:
        validation_mode = "strict_document_snapshot"
    elif all_layout_match and current_alignment_matches:
        validation_mode = "stable_model_slot_and_layout"
    else:
        validation_mode = "stable_model_slot_column_ids_and_text_compatible"

    report = {
        "schema": "novel_formatter.multi_ocr_hybrid_correction_import_report.v5",
        "package_id": str(payload.get("package_id", "") or ""),
        "audit_id": audit_id,
        "source_schema": schema,
        "base_ocr_evidence_immutable": True,
        "raw_ocr_documents_immutable": True,
        "original_comparison_immutable": True,
        "non_destructive_overlay_import": True,
        "source_model_corrections_enabled": True,
        "source_model_corrections_imported_as_overlay": True,
        "changed_model_cells": changed_model_cells,
        "applied_model_rows": 0,
        "proposed_model_cells": int(proposed_model_cells),
        "proposed_model_rows": int(source_stats["applied_model_rows"]),
        "requested_model_rows": int(source_stats["requested_model_rows"]),
        "requested_model_edit_values": int(source_stats["requested_model_edit_values"]),
        "already_applied_model_rows": int(source_stats["already_applied_model_rows"]),
        "source_corrected_row_ids": list(source_stats["corrected_row_ids"]),
        "source_corrected_column_groups": list(source_stats["affected_column_groups"]),
        "resolved_history_rows_annotated": historical_rows_annotated,
        "normalized_legacy_provisional_rows": normalized_legacy_provisional_rows,
        "accepted_canonical_decisions": accepted_count,
        "unresolved_canonical_decisions": unresolved_count,
        "prefilled_resolved_decisions": prefilled_resolved,
        "legacy_migrated_accepted_decisions": legacy_migrated_accepted,
        "native_accepted_decisions": native_accepted,
        "rejected_placeholder_decisions": rejected_placeholder,
        "rejected_suspicious_ascii_decisions": rejected_ascii,
        "legacy_complex_rows_requiring_whole_verdict": legacy_complex,
        "canonical_decisions": decisions,
        "three_way_merge_conflicts": len(source_stats["merge_conflicts"]),
        "merge_conflicts": list(source_stats["merge_conflicts"]),
        "touched_model_ids": touched_model_ids,
        "touched_model_labels": [],
        "proposed_model_ids": proposed_model_ids,
        "proposed_model_labels": [
            str(exported_registry[model_id].get("display_label", "") or model_id)
            for model_id in proposed_model_ids
        ],
        "identity_validation_mode": validation_mode,
        "current_alignment_changed": not current_alignment_matches,
        "model_layout_matches": layout_matches,
        "model_structure_matches": structure_matches,
        "model_snapshot_matches": snapshot_matches,
        "model_text_compatibility_scores": compatibility_scores,
        "expected_physical_column_count": expected_count,
        "before": before,
        "after": after,
        "derivative_state_must_rebuild": False,
        "skip_realign_after_import": True,
        "preserve_resolved_disagreement_history": True,
        "original_disagreement_remains_live": True,
        "forbid_false_model_consensus": True,
    }
    _report_progress(progress_callback, "完成非破坏式 AI 纠错覆盖导入", 6, 6)
    return output_docs, refreshed_comparison, report

def export_fusion_and_skeleton_bundle(
    primary_doc: UnifiedDocument,
    fusion_package: dict,
    output_path: str | Path,
    *,
    correction_audit: dict | None = None,
    vertical: bool = True,
) -> dict:
    """Export current complete fusion JSON plus a clean stable-ID skeleton EPUB."""
    output = Path(output_path).expanduser()
    if output.suffix.lower() != ".zip":
        output = output.with_suffix(".zip")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nf_fusion_skeleton_") as temp:
        folder = Path(temp) / output.stem
        folder.mkdir(parents=True, exist_ok=True)
        fusion_path = folder / "01_multi_ocr_fusion_result.json"
        fusion_path.write_bytes(_json_bytes(fusion_package, pretty=True))
        fusion_sha256 = _sha256(fusion_path.read_bytes())
        audit = copy.deepcopy(correction_audit or {})
        (folder / "02_model_correction_audit.json").write_bytes(_json_bytes(audit, pretty=True))
        comparison = fusion_package.get("comparison") or {}
        alignment_report = {
            "schema": "novel_formatter.final_alignment_report.v1",
            "package_id": str(fusion_package.get("package_id", "") or ""),
            "row_count": len(fusion_package.get("editable_items") or []),
            "alignment_mode": str(comparison.get("alignment_mode", "") or ""),
            "exact_rows": int(comparison.get("exact_rows", 0) or 0),
            "provisional_consensus_rows": int(comparison.get("provisional_consensus_rows", 0) or 0),
            "conflict_rows": int(comparison.get("conflict_rows", 0) or 0),
            "low_confidence_rows": int(comparison.get("low_confidence_rows", 0) or 0),
            "physical_column_source": str(comparison.get("physical_column_source", "") or ""),
        }
        (folder / "03_final_alignment_report.json").write_bytes(_json_bytes(alignment_report, pretty=True))
        framework = folder / "framework"
        framework.mkdir(parents=True, exist_ok=True)
        skeleton = framework / "structure_skeleton.epub"
        from engine.ai_repair_epub import export_ai_repair_epub
        epub_report = export_ai_repair_epub(
            primary_doc,
            fusion_package,
            skeleton,
            mode="one_pass",
            vertical=vertical,
            workflow="exchange",
        )
        try:
            from engine.ai_publication_bundle_v2 import _strip_framework_work_payloads
            _strip_framework_work_payloads(skeleton)
        except Exception:
            pass

        # External AI edits plain text only.  Ruby is frozen separately and the
        # provided builder re-attaches only uniquely resolvable readings.
        from engine.ruby_exchange_bundle import (
            build_edit_template, build_locked_ruby_payload, model_command_text,
            write_exchange_tools,
        )
        ruby_lock = build_locked_ruby_payload(primary_doc, fusion_package)
        ruby_lock_path = folder / "04_ruby_overlay.locked.json"
        ruby_lock_path.write_bytes(_json_bytes(ruby_lock, pretty=True))
        ruby_lock_sha256 = _sha256(ruby_lock_path.read_bytes())
        ai_output = folder / "AI_OUTPUT"
        ai_output.mkdir(parents=True, exist_ok=True)
        edit_template = build_edit_template(
            fusion_package, fusion_sha256=fusion_sha256,
            ruby_lock_sha256=ruby_lock_sha256,
        )
        (ai_output / "edited_text.json").write_bytes(_json_bytes(edit_template, pretty=True))
        (folder / "00_AGENTS.md").write_text(model_command_text(), encoding="utf-8")
        tool_paths = write_exchange_tools(folder)
        tool_sha256 = {
            relative: _sha256((folder / relative).read_bytes())
            for relative in tool_paths
        }

        skeleton_sha256 = _sha256(skeleton.read_bytes())
        manifest = {
            "schema": "novel_formatter.fusion_skeleton_bundle.v2",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "fusion_json": fusion_path.name,
            "skeleton_epub": skeleton.relative_to(folder).as_posix(),
            "ruby_lock": ruby_lock_path.name,
            "edit_template": "AI_OUTPUT/edited_text.json",
            "builder": "tools/build_final_epub.py",
            "ruby_validator": "tools/validate_ruby.py",
            "fusion_json_sha256": fusion_sha256,
            "skeleton_epub_sha256": skeleton_sha256,
            "ruby_lock_sha256": ruby_lock_sha256,
            "ruby_enabled": bool(ruby_lock.get("ruby_preservation_enabled")),
            "ruby_pair_count": int(ruby_lock.get("ruby_pair_count", 0) or 0),
            "ruby_anchor_policy": str(ruby_lock.get("anchor_policy", "") or ""),
            "ruby_anchor_policy_version": int(ruby_lock.get("anchor_policy_version", 1) or 1),
            "row_count": alignment_report["row_count"],
            "tool_paths": tool_paths,
            "tool_sha256": tool_sha256,
            "epub_report": epub_report,
        }
        (folder / "00_manifest.json").write_bytes(_json_bytes(manifest, pretty=True))
        temp_zip = output.with_name(f".{output.name}.tmp")
        try:
            with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                for path in sorted(folder.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(folder).as_posix())
            os.replace(temp_zip, output)
        finally:
            if temp_zip.exists():
                temp_zip.unlink(missing_ok=True)
    return {
        "path": str(output),
        "fusion_json_sha256": manifest["fusion_json_sha256"],
        "skeleton_epub_sha256": manifest["skeleton_epub_sha256"],
        "ruby_lock_sha256": manifest.get("ruby_lock_sha256", ""),
        "ruby_enabled": bool(manifest.get("ruby_enabled")),
        "ruby_pair_count": int(manifest.get("ruby_pair_count", 0) or 0),
        "row_count": manifest["row_count"],
    }


def documents_with_comparison_texts(
    documents: Sequence[UnifiedDocument],
    comparison: MultiOcrComparison,
    *,
    progress_callback: ProgressCallback | None = None,
) -> list[UnifiedDocument]:
    """Synchronise compare-editor text in linear time without mutating active docs."""
    source_docs = list(documents)
    snapshots = [physical_column_text_snapshot(doc)[0] for doc in source_docs]
    updates_by_model: list[dict[str, str]] = [dict() for _ in source_docs]
    total_rows = max(1, len(comparison.rows))
    for row_index, row in enumerate(comparison.rows):
        if row_index == 0 or (row_index + 1) % 100 == 0 or row_index + 1 == total_rows:
            _report_progress(progress_callback, "同步当前 OCR 对比文字", row_index + 1, total_rows)
        column_ids = [str(value) for value in (row.column_ids or ())]
        if not column_ids:
            raise SourceCorrectionError("当前比较包含没有物理列 ID 的行，无法同步逐源文本。")
        for model_index, _doc in enumerate(source_docs):
            text = str(row.texts[model_index] if model_index < len(row.texts) else "")
            snapshot = snapshots[model_index]
            source_parts = [snapshot.get(column_id, "") for column_id in column_ids]
            projected = project_fused_text_to_physical_columns(text, source_parts, column_count=len(column_ids))
            for column_id, value in zip(column_ids, projected):
                previous = updates_by_model[model_index].get(column_id)
                if previous is not None and previous != value:
                    raise SourceCorrectionError(f"同一物理列在当前对齐中出现冲突文本：{column_id}")
                if snapshot.get(column_id, "") != value:
                    updates_by_model[model_index][column_id] = value
    docs = list(source_docs)
    active = [(index, updates) for index, updates in enumerate(updates_by_model) if updates]
    for position, (model_index, updates) in enumerate(active, start=1):
        doc_copy = copy.deepcopy(source_docs[model_index])
        _apply_column_updates(doc_copy, updates, audit={
            "audit_id": "comparison_editor_sync",
            "package_id": "",
            "model_id": "",
            "imported_at": datetime.now(timezone.utc).isoformat(),
            "column_count": len(updates),
        })
        docs[model_index] = doc_copy
        _report_progress(progress_callback, "应用当前 OCR 对比文字", position, max(1, len(active)))
    return docs
