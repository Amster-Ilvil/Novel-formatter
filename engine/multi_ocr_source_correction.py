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
import io
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
from core.multi_ocr_roles import MULTI_OCR_ROLE_SCHEMA, SUPPORTED_MULTI_OCR_ROLE_SCHEMAS
from engine.column_sentence_reflow import join_column_parts, has_sentence_terminal
from engine.multi_ocr_compare import (
    MultiOcrComparison,
    MultiOcrRow,
    compare_ocr_documents,
    physical_column_text_snapshot,
    project_fused_text_to_physical_columns,
)
from engine.ocr_roundtrip_package import _json_detached, structure_hash, layout_hash
from utils.safe_archive import (
    UnsafeArchiveError,
    ZipExtractionLimits,
    validate_zip,
)
from utils.atomic_io import atomic_output_path

SCHEMA = "novel_formatter.multi_ocr_source_correction.v3"
CANONICAL_CORRECTIONS_SCHEMA = "novel_formatter.multi_ocr_canonical_adjudication.v5"
SUPPORTED_CANONICAL_CORRECTIONS_SCHEMAS = {CANONICAL_CORRECTIONS_SCHEMA}
RECOVERY_SCHEMA = "novel_formatter.multi_ocr_recovery_snapshot.v3"
EXCHANGE_VERSION = 7
EXCHANGE_PROFILE = "current_role_aware_multi_ocr_v2"
MODEL_REGISTRY_SCHEMA = "novel_formatter.multi_ocr_model_registry.v2"
ALIGNMENT_SNAPSHOT_SCHEMA = "novel_formatter.multi_ocr_alignment_snapshot.v2"
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


class LegacyAdjudicationCompatibilityUnavailable(SourceCorrectionError):
    """Compatibility hook exists, but legacy package parsing is disabled."""


def _legacy_adjudication_compatibility_interface(*_args, operation: str = "import", **_kwargs):
    """Reserved compatibility seam for the future stable release.

    Keep this callable and the historical helper names stable, but do not carry
    any old V1/V2/V3/V4 migration heuristics in pre-stable builds.
    """
    raise LegacyAdjudicationCompatibilityUnavailable(
        f"旧裁决包兼容接口已保留，但当前开发版未启用旧格式{operation}实现；"
        f"请使用当前 {CANONICAL_CORRECTIONS_SCHEMA} / {SCHEMA} 包。"
    )


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
    value = _json_detached(doc.to_dict())
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
    """Collect durable OCR-input provenance per physical column.

    New sessions persist a document-level provenance map before sentence reflow or
    page-role projection can discard block metadata.  Legacy sessions are still
    reconstructed conservatively from block metadata.  ``final_input_sha256`` is
    only populated when the exact selected recognizer input is known; a prepared
    canonical crop is never substituted for an NDLOCR page input or an untracked
    rescue variant.
    """
    records: dict[str, dict[str, object]] = {}

    durable = getattr(getattr(doc, "metadata", None), "column_ocr_input_records", {}) or {}
    if isinstance(durable, dict):
        for column_id, raw in durable.items():
            if not str(column_id) or not isinstance(raw, dict):
                continue
            raw = dict(raw)
            prepared = str(
                raw.get("prepared_column_sha256")
                or raw.get("column_input_sha256")
                or ""
            )
            actual = str(raw.get("actual_input_sha256") or raw.get("final_input_sha256") or "")
            actual_list = [
                str(value) for value in (raw.get("actual_input_sha256s") or raw.get("final_input_sha256s") or [])
                if str(value)
            ]
            if actual and actual not in actual_list:
                actual_list.insert(0, actual)
            scope = str(raw.get("input_hash_scope") or "unavailable")
            records[str(column_id)] = {
                "column_id": str(column_id),
                "column_input_sha256": prepared,
                "prepared_column_sha256": prepared,
                "page_input_sha256": str(raw.get("page_input_sha256") or ""),
                "sentence_input_sha256": str(raw.get("sentence_input_sha256") or ""),
                "actual_input_sha256": actual,
                "actual_input_sha256s": actual_list,
                "final_input_sha256": actual,
                "final_input_sha256s": actual_list,
                "input_hash_scope": scope if actual or scope != "unavailable" else "unavailable",
                "input_profile": str(raw.get("input_profile") or ""),
                "input_profile_sha256": str(raw.get("input_profile_sha256") or ""),
                "input_contract": str(raw.get("input_contract") or ""),
                "transport": str(raw.get("transport") or ""),
                "transport_version": str(raw.get("transport_version") or ""),
                "isolation_mode": str(raw.get("isolation_mode") or ""),
                "selected_variant": str(raw.get("selected_variant") or ""),
                "seeded_reuse": bool(raw.get("seeded_reuse", False)),
                "metadata_conflict": bool(raw.get("metadata_conflict", False)),
            }

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
        prepared_hashes = values(
            metadata, "column_ocr_input_sha256", "source_column_ocr_input_sha256", count
        )
        actual_hashes = values(
            metadata, "column_ocr_actual_input_sha256", "source_column_ocr_actual_input_sha256", count
        )
        raw_actual_hash_lists = metadata.get("source_column_ocr_actual_input_sha256s", []) or []
        if not raw_actual_hash_lists and metadata.get("column_ocr_actual_input_sha256s"):
            raw_actual_hash_lists = [metadata.get("column_ocr_actual_input_sha256s")] * max(1, count)
        actual_hash_lists: list[list[str]] = []
        for value in list(raw_actual_hash_lists)[:count]:
            if isinstance(value, str):
                value = [value]
            actual_hash_lists.append([str(item) for item in (value or []) if str(item)])
        while len(actual_hash_lists) < count:
            actual_hash_lists.append([])
        actual_scopes = values(
            metadata, "column_ocr_actual_input_scope", "source_column_ocr_actual_input_scope", count
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
        sentence_input_hash = str(metadata.get("sentence_context_reocr_input_sha256", "") or "")
        for index, column_id in enumerate(column_ids):
            record = records.setdefault(
                column_id,
                {
                    "column_id": column_id,
                    "column_input_sha256": "",
                    "prepared_column_sha256": "",
                    "page_input_sha256": "",
                    "sentence_input_sha256": "",
                    "actual_input_sha256": "",
                    "actual_input_sha256s": [],
                    "final_input_sha256": "",
                    "final_input_sha256s": [],
                    "input_hash_scope": "unavailable",
                    "input_profile": "",
                    "input_profile_sha256": "",
                    "input_contract": "",
                    "transport": "",
                    "metadata_conflict": False,
                },
            )
            incoming = {
                "column_input_sha256": prepared_hashes[index],
                "prepared_column_sha256": prepared_hashes[index],
                "page_input_sha256": page_input_hash,
                "sentence_input_sha256": sentence_input_hash,
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

            incoming_actual = actual_hashes[index]
            incoming_actual_list = list(actual_hash_lists[index])
            if incoming_actual and incoming_actual not in incoming_actual_list:
                incoming_actual_list.insert(0, incoming_actual)
            incoming_scope = actual_scopes[index]
            if incoming_actual:
                current = str(record.get("actual_input_sha256", "") or "")
                if current and current != incoming_actual:
                    record["metadata_conflict"] = True
                elif not current:
                    record["actual_input_sha256"] = incoming_actual
                    record["final_input_sha256"] = incoming_actual
                    record["input_hash_scope"] = incoming_scope or "physical_column"
            if incoming_actual_list:
                current_list = [
                    str(value) for value in (record.get("actual_input_sha256s") or []) if str(value)
                ]
                for value in incoming_actual_list:
                    if value not in current_list:
                        current_list.append(value)
                record["actual_input_sha256s"] = current_list
                record["final_input_sha256s"] = list(current_list)
                if not incoming_actual and incoming_scope:
                    record["input_hash_scope"] = incoming_scope

    # Legacy sessions lack explicit actual-input fields.  Only infer when the
    # transport unambiguously identifies the selected bytes.  Never prefer a
    # canonical crop over a known page-routed NDLOCR input.
    for record in records.values():
        actual = str(record.get("actual_input_sha256", "") or record.get("final_input_sha256", "") or "")
        actual_list = [
            str(value) for value in (record.get("actual_input_sha256s") or record.get("final_input_sha256s") or [])
            if str(value)
        ]
        if actual and actual not in actual_list:
            actual_list.insert(0, actual)
        scope = str(record.get("input_hash_scope", "") or "")
        prepared = str(record.get("prepared_column_sha256", "") or record.get("column_input_sha256", "") or "")
        page_hash = str(record.get("page_input_sha256", "") or "")
        transport = str(record.get("transport", "") or "")
        if not actual:
            if page_hash and "full_page_routed" in transport and "fallback" not in transport:
                actual, scope = page_hash, "page_routed"
            elif prepared and page_hash and "fallback" in transport:
                # The final choice is ambiguous in mixed metadata; preserve the
                # prepared hash but do not call it the actual recognizer input.
                actual, scope = "", "mixed_unavailable"
            elif prepared and transport and "full_page_routed" not in transport:
                actual, scope = prepared, "physical_column"
        if actual and actual not in actual_list:
            actual_list.insert(0, actual)
        record["actual_input_sha256"] = actual
        record["actual_input_sha256s"] = actual_list
        record["final_input_sha256"] = actual
        record["final_input_sha256s"] = list(actual_list)
        record["input_hash_scope"] = scope or "unavailable"
    return records

def _document_ocr_input_audit(doc: UnifiedDocument) -> dict[str, object]:
    """Summarise exact actual-input provenance without overstating availability."""
    records = _document_ocr_input_records(doc)
    profiles = {str(item.get("input_profile", "") or "") for item in records.values()}
    profile_hashes = {
        str(item.get("input_profile_sha256", "") or "") for item in records.values()
    }
    transports = {str(item.get("transport", "") or "") for item in records.values()}
    prepared_hashes = {
        str(item.get("prepared_column_sha256", "") or "") for item in records.values()
    }
    actual_hashes = {
        str(value)
        for item in records.values()
        for value in (item.get("final_input_sha256s") or ([item.get("final_input_sha256")] if item.get("final_input_sha256") else []))
        if str(value)
    }
    profiles.discard(""); profile_hashes.discard(""); transports.discard("")
    prepared_hashes.discard(""); actual_hashes.discard("")
    physical_hash_count = sum(
        bool(item.get("final_input_sha256") or item.get("final_input_sha256s"))
        and item.get("input_hash_scope") in {"physical_column", "rescue_variant"}
        for item in records.values()
    )
    page_hash_count = sum(
        bool(item.get("final_input_sha256")) and item.get("input_hash_scope") == "page_routed"
        for item in records.values()
    )
    sentence_hash_count = sum(
        bool(item.get("final_input_sha256")) and item.get("input_hash_scope") == "sentence_group"
        for item in records.values()
    )
    total = len(records)
    hashed = sum(bool(item.get("final_input_sha256") or item.get("final_input_sha256s")) for item in records.values())
    if total and hashed == total:
        level = "full_hash"
    elif hashed:
        level = "partial_hash"
    elif profiles or profile_hashes or transports or prepared_hashes:
        level = "profile_only"
    else:
        level = "unavailable"
    return {
        "ocr_input_profiles": sorted(profiles),
        "ocr_input_profile_sha256": sorted(profile_hashes),
        "ocr_input_transports": sorted(transports),
        "physical_columns_with_input_sha256": int(physical_hash_count),
        "physical_columns_with_page_input_sha256": int(page_hash_count),
        "physical_columns_with_sentence_input_sha256": int(sentence_hash_count),
        "physical_columns_with_prepared_sha256": sum(bool(item.get("prepared_column_sha256")) for item in records.values()),
        "physical_columns_without_input_sha256": max(0, total - hashed),
        "unique_input_sha256_count": len(actual_hashes),
        "unique_prepared_sha256_count": len(prepared_hashes),
        "ocr_input_audit_level": level,
        "ocr_input_profile_metadata_available": bool(profiles or profile_hashes or transports or prepared_hashes),
        "ocr_input_hash_audit_available": bool(hashed),
        "ocr_input_audit_available": bool(profiles or profile_hashes or transports or prepared_hashes or hashed),
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
            input_hashes = [str(value) for value in (item.get("final_input_sha256s") or []) if str(value)]
            if input_hash and input_hash not in input_hashes:
                input_hashes.insert(0, input_hash)
            item["final_input_sha256s"] = input_hashes
            scope = str(item.get("input_hash_scope", "unavailable") or "unavailable")
            if input_hash and scope in {"physical_column", "rescue_variant"}:
                exact_groups.setdefault(input_hash, []).append(item["model_id"])
            model_rows.append(item)
        for item in model_rows:
            input_hash = str(item.get("final_input_sha256", "") or "")
            input_hashes = [str(value) for value in (item.get("final_input_sha256s") or []) if str(value)]
            scope = str(item.get("input_hash_scope", "unavailable") or "unavailable")
            if scope == "page_routed":
                status = "page_routed"
                shared_ids: list[str] = []
            elif scope == "rescue_composite" and input_hashes:
                status = "composite_exact"
                shared_ids = []
            elif input_hash:
                shared_ids = exact_groups.get(input_hash, [])
                status = "shared_exact" if len(shared_ids) > 1 else "distinct_exact"
            elif input_hashes:
                status = "composite_exact"
                shared_ids = []
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



def _source_document_role_metadata(document: UnifiedDocument, model_index: int, label: str) -> dict:
    metadata = getattr(document, "metadata", None)
    raw = getattr(metadata, "__dict__", {}) if metadata is not None else {}
    if not isinstance(raw, dict):
        raw = {}

    # Schema 3 is the free-slot contract: slot number is identity/display order
    # only; input granularity and transport profile come from the engine profile.
    try:
        slot_schema = int(raw.get("multi_ocr_slot_schema", 0) or 0)
    except (TypeError, ValueError):
        slot_schema = 0
    if slot_schema >= 3:
        slot_index = int(raw.get("multi_ocr_slot_index", model_index + 1) or (model_index + 1))
        granularity = str(raw.get("multi_ocr_input_role", "") or "column").strip()
        return {
            "role": f"slot{slot_index}",
            "role_label": f"模型 {slot_index}",
            "input_granularity": granularity,
            "role_schema": 3,
            "transport_profile": str(raw.get("multi_ocr_transport_profile", "") or ""),
            "resource_class": str(raw.get("multi_ocr_resource_class", "") or ""),
            "model_label": str(label or f"OCR 模型 {model_index + 1}"),
        }

    role = str(raw.get("multi_ocr_role", "") or "").strip()
    try:
        from core.multi_ocr_roles import ROLE_LABELS
        role_label = str(ROLE_LABELS.get(role, role) or role)
    except Exception:
        role_label = role
    granularity = {
        "column": "column",
        "page": "page",
        "sentence": "column",
        "review1": "column_review",
        "review2": "column_review",
        "review3": "column_review",
    }.get(role, "unknown")
    return {
        "role": role,
        "role_label": role_label,
        "input_granularity": granularity,
        "role_schema": int(raw.get("multi_ocr_role_schema", 0) or 0),
        "transport_profile": str(raw.get("multi_ocr_transport_profile", "") or ""),
        "resource_class": str(raw.get("multi_ocr_resource_class", "") or ""),
        "model_label": str(label or f"OCR 模型 {model_index + 1}"),
    }


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
        display_label = str(labels[index] if index < len(labels) else f"OCR 模型 {index + 1}")
        role_meta = _source_document_role_metadata(doc, index, display_label)
        registry.append({
            "model_id": model_id,
            "model_index": index,
            "display_label": display_label,
            "source_engine": engine,
            **role_meta,
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


def _validate_current_model_registry(registry: Sequence[dict], *, context: str = "当前 OCR 会话") -> None:
    allowed_roles = {"column", "page", "sentence", "review1", "review2", "review3", "slot1", "slot2", "slot3"}
    if not registry:
        raise SourceCorrectionError(f"{context}没有模型注册信息。")
    seen_roles: set[str] = set()
    seen_ids: set[str] = set()
    for index, item in enumerate(registry):
        if not isinstance(item, dict):
            raise SourceCorrectionError(f"{context}的 model_registry 第 {index + 1} 项不是对象。")
        model_id = str(item.get("model_id", "") or "").strip()
        role = str(item.get("role", "") or "").strip()
        granularity = str(item.get("input_granularity", "") or "").strip()
        source_engine = str(item.get("source_engine", "") or "").strip()
        try:
            role_schema = int(item.get("role_schema", 0) or 0)
        except (TypeError, ValueError):
            role_schema = 0
        if not model_id or model_id in seen_ids:
            raise SourceCorrectionError(f"{context}的模型 ID 缺失或重复。")
        if role_schema not in SUPPORTED_MULTI_OCR_ROLE_SCHEMAS:
            supported = "/".join(str(value) for value in sorted(SUPPORTED_MULTI_OCR_ROLE_SCHEMAS))
            raise SourceCorrectionError(
                f"{context}的模型 {item.get('display_label') or model_id} 缺少受支持的显式角色 "
                f"schema（支持 {supported}，当前 {role_schema or '<missing>'}）；"
                "旧的无角色 schema=0 会话不能安全推断模型职责。"
            )
        if role not in allowed_roles:
            raise SourceCorrectionError(
                f"{context}的模型角色无效或缺失：{role or '<missing>'}；"
                "稳定版前不兼容旧的无角色多模型会话。"
            )
        if role in seen_roles:
            raise SourceCorrectionError(f"{context}存在重复模型角色：{role}。")
        if not source_engine:
            raise SourceCorrectionError(f"{context}的模型 {model_id} 缺少 source_engine。")
        if not granularity or granularity == "unknown":
            raise SourceCorrectionError(f"{context}的模型 {model_id} 缺少当前输入粒度。")
        seen_ids.add(model_id)
        seen_roles.add(role)


def build_model_registry(
    documents: Sequence[UnifiedDocument], labels: Sequence[str]
) -> list[dict]:
    registry, _snapshots = _build_model_registry_and_snapshots(documents, labels)
    _validate_current_model_registry(registry)
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


def canonical_decision_key(value) -> tuple[str, ...]:
    """Return the stable identity for one adjudication row.

    Current comparison rows carry ``sentence_group_id`` because one physical
    OCR column may legitimately contain multiple sentence groups.  Package
    import/export requires this identity; the column-only fallback is retained
    only for internal transient callers and is never accepted as exchange authority.
    """
    if isinstance(value, dict):
        group_id = str(value.get("sentence_group_id", "") or "")
        columns = value.get("column_ids") or ()
    else:
        group_id = str(getattr(value, "sentence_group_id", "") or "")
        columns = getattr(value, "column_ids", ()) or ()
    if group_id:
        return ("sentence_group_id", group_id)
    ids = tuple(str(item) for item in columns if str(item))
    return (("column_ids",) + ids) if ids else ()


def _canonical_decision_id(
    column_ids: Sequence[str],
    sentence_group_id: str = "",
) -> str:
    identity = {
        "sentence_group_id": str(sentence_group_id or ""),
        "column_ids": [str(value) for value in column_ids if str(value)],
    }
    return f"decision:{_sha256(identity)[:20]}"


def canonical_decision_from_fusion_state(row, state, labels: Sequence[str] = ()) -> dict | None:
    """Promote one explicit fusion selection into the canonical decision store."""
    from engine.ocr_compare_view_model import is_explicit_fusion_selection_origin

    selected = getattr(state, "selected_index", None)
    candidates = list(getattr(state, "candidates", ()) or ())
    origin = str(getattr(state, "selection_origin", "") or "")
    if (
        selected is None
        or not 0 <= int(selected) < len(candidates)
        or not is_explicit_fusion_selection_origin(origin)
    ):
        return None
    candidate = candidates[int(selected)]
    final_text = str(getattr(candidate, "text", "") or "").strip()
    delete_intentionally = bool(getattr(candidate, "delete_intentionally", False))
    if not final_text and not delete_intentionally:
        return None

    column_ids = [str(value) for value in (getattr(row, "column_ids", ()) or ()) if str(value)]
    sentence_group_id = str(getattr(row, "sentence_group_id", "") or "")
    raw_texts = [str(value or "") for value in (getattr(row, "texts", ()) or ())]
    raw_by_label = {
        str(labels[index] if index < len(labels) else f"model_{index + 1}"): raw_texts[index]
        for index in range(len(raw_texts))
    }
    reason = str(
        getattr(candidate, "reason", "")
        or getattr(state, "fusion_reason", "")
        or "当前 OCR 融合界面的显式裁决已写入统一 canonical 状态。"
    )
    candidate_audit_flags = [
        str(value) for value in (getattr(candidate, "audit_flags", ()) or ()) if str(value)
    ]
    audit_flags = list(dict.fromkeys(["canonical_single_writeback_chain", *candidate_audit_flags]))
    return {
        "decision_id": _canonical_decision_id(column_ids, sentence_group_id),
        "row_id": sentence_group_id or f"row:{int(getattr(row, 'index', 0) or 0):06d}",
        "row_index": int(getattr(row, "index", 0) or 0),
        "sentence_group_id": sentence_group_id,
        "column_ids": column_ids,
        "final_text": final_text,
        "status": "accepted",
        "source": origin,
        "derivation": "explicit_fusion_selection",
        "confidence": float(getattr(candidate, "confidence", 0.0) or 0.0),
        "reason": reason,
        "audit_level": str(getattr(candidate, "audit_level", "") or ""),
        "audit_flags": audit_flags,
        "raw_model_texts": raw_by_label,
        "raw_model_texts_by_index": raw_texts,
        "historical_raw_model_texts_by_index": [
            str(value or "") for value in (getattr(row, "historical_ocr_texts", ()) or ())
        ],
        "historical_disagreement": bool(getattr(row, "historical_ocr_disagreement", False)),
        "resolution_kind": origin,
        "delete_intentionally": delete_intentionally,
    }


def _clean_candidate(text: str) -> bool:
    value = str(text or "")
    return bool(value.strip()) and not _contains_placeholder(value) and not _suspicious_inline_latin(value)


def _decision_matches_current_evidence(decision: dict, model_ids: Sequence[str], texts: Sequence[str]) -> bool:
    """Require byte-for-byte logical OCR evidence equality for current packages.

    Development builds no longer reuse adjudication across changed OCR sessions.
    Every accepted verdict must carry the exact ordered raw model texts from the
    current model registry; fuzzy similarity and insertion-order fallbacks are
    deliberately forbidden.
    """
    indexed = decision.get("raw_model_texts_by_index")
    raw = decision.get("raw_model_texts")
    if not isinstance(indexed, list) or len(indexed) != len(model_ids):
        return False
    indexed_values = [str(value or "") for value in indexed]
    current_values = [str(value or "") for value in texts]
    if indexed_values != current_values:
        return False
    # ``raw_model_texts_by_index`` is the authoritative current-session evidence.
    # Canonical decisions are created inside the live comparison before the
    # source-correction exporter allocates its registry model IDs, so the optional
    # mapping may legitimately be keyed by display labels.  Accepting the exact
    # ordered vector is strict within the same session and does not reintroduce
    # any cross-version/model fuzzy matching.
    if raw is not None and not isinstance(raw, dict):
        return False
    if isinstance(raw, dict) and raw:
        raw_values = [str(value or "") for value in raw.values()]
        if len(raw_values) != len(current_values) or raw_values != current_values:
            return False
    return True


def _canonical_decision_can_be_reopened_for_ai_review(decision: dict | None) -> bool:
    """Allow explicit second-pass review only for prior AI-origin verdicts.

    Human/local decisions are already authoritative outcomes.  They must never
    be pushed back into ``pending_ai_review`` merely because a later AI package
    is exported with prior-AI review enabled.
    """
    if not isinstance(decision, dict):
        return False
    source = str(decision.get("source", "") or "").strip().lower()
    if not source:
        return False
    if source in {
        "local_targeted_retry_majority_adjudication",
        "human_ocr_compare",
        "human_image_review",
        "human_manual_edit",
        "restored_human",
        "ai_visual_batch_adjudication",
    }:
        return False
    return (
        source in {"external_ai_package", "ai_adjudication_result", "ai_overlay", "ai_final_verdict"}
        or source.startswith("external_ai")
        or source.startswith("cloud_ai")
        or source.startswith("ai_import")
    )


def _accepted_decision_for_export(
    decision: dict | None,
    *,
    column_ids: Sequence[str],
    sentence_group_id: str = "",
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
    current_group_id = str(sentence_group_id or "")
    decision_group_id = str(decision.get("sentence_group_id", "") or "")
    if not current_group_id or decision_group_id != current_group_id:
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
        "decision_id": str(decision.get("decision_id", "") or _canonical_decision_id(
            expected_ids, current_group_id
        )),
        "sentence_group_id": current_group_id or decision_group_id,
        "column_ids": list(expected_ids),
        "final_text": final_text,
        "reason": str(decision.get("reason", "") or "此前当前格式裁决已按原始 OCR 证据重新校验。"),
        "confidence": float(decision.get("confidence", 0.0) or 0.0),
        "delete_intentionally": delete_intentionally,
        "source": str(decision.get("source", "") or "prior_canonical_decision"),
        "derivation": str(decision.get("derivation", "") or "prior_accepted_verdict"),
        "audit_level": str(decision.get("audit_level", "") or ""),
        "audit_flags": [str(value) for value in (decision.get("audit_flags") or [])],
        "revalidated_from_prior_round": True,
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


def _read_canonical_verdict(row: dict, model_ids: Sequence[str], schema: str) -> dict:
    if schema != CANONICAL_CORRECTIONS_SCHEMA:
        raise SourceCorrectionError(
            f"裁决 schema={schema or '<missing>'} 不受当前开发版支持。"
        )
    column_ids = [str(value) for value in (row.get("column_ids") or []) if str(value)]
    sentence_group_id = str(row.get("sentence_group_id", "") or "")
    if not sentence_group_id:
        raise SourceCorrectionError(f"{row.get('row_id', '')} 缺少 sentence_group_id。")
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
        "decision_id": str(verdict.get("decision_id", "") or _canonical_decision_id(column_ids, sentence_group_id)),
        "row_id": str(row.get("row_id", "") or ""),
        "row_index": int(row.get("row_index", 0) or 0),
        "sentence_group_id": sentence_group_id,
        "column_ids": column_ids,
        "final_text": final_text if accepted else "",
        "status": "accepted" if accepted else "unresolved",
        "source": str(verdict.get("source", "") or "ai_canonical_verdict_current"),
        "derivation": str(verdict.get("derivation", "") or "explicit_final_text"),
        "confidence": float(verdict.get("confidence", 0.0) or 0.0),
        "reason": str(verdict.get("reason", "") or ""),
        "audit_flags": flags,
        "raw_model_texts": dict(row.get("base_model_texts") or {}),
        "raw_model_texts_by_index": [
            str((row.get("base_model_texts") or {}).get(model_id, "") or "")
            for model_id in model_ids
        ],
        "historical_raw_model_texts": copy.deepcopy(verdict.get("historical_raw_model_texts") or {}),
        "historical_raw_model_texts_by_index": [
            str(value or "") for value in (verdict.get("historical_raw_model_texts_by_index") or [])
        ],
        "historical_disagreement": bool(verdict.get("historical_disagreement", False)),
        "resolution_kind": str(verdict.get("resolution_kind", "") or ""),
        "delete_intentionally": delete_intentionally,
    }


def merge_canonical_decision_overlays(
    existing: Sequence[dict] | None,
    incoming: Sequence[dict] | None,
) -> tuple[list[dict], dict]:
    """Merge repeated current-format adjudication imports by sentence identity.

    Every persisted decision must carry ``sentence_group_id``. Column-only keys
    are rejected instead of guessed or upgraded from older package layouts.
    """
    by_key: dict[tuple[str, ...], dict] = {}
    order: list[tuple[str, ...]] = []

    def current_key(item: dict, *, source: str) -> tuple[str, ...]:
        group_id = str(item.get("sentence_group_id", "") or "")
        if not group_id:
            raise SourceCorrectionError(
                f"{source}裁决缺少 sentence_group_id；当前开发版不兼容列级旧裁决。"
            )
        return ("sentence_group_id", group_id)

    existing_count = 0
    for item in existing or ():
        if not isinstance(item, dict):
            continue
        key = current_key(item, source="现有")
        existing_count += 1
        if key not in by_key:
            order.append(key)
        by_key[key] = copy.deepcopy(item)

    incoming_count = 0
    new_rows = 0
    replaced_rows = 0
    preserved_rows = 0
    unresolved_new_rows = 0
    changed_accepted: list[dict] = []
    for item in incoming or ():
        if not isinstance(item, dict):
            continue
        incoming_count += 1
        key = current_key(item, source="导入")
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
                changed = (
                    str(current.get("final_text", "") or "") != str(candidate.get("final_text", "") or "")
                    or bool(current.get("delete_intentionally", False)) != bool(candidate.get("delete_intentionally", False))
                    or float(current.get("confidence", 0.0) or 0.0) != float(candidate.get("confidence", 0.0) or 0.0)
                    or str(current.get("reason", "") or "") != str(candidate.get("reason", "") or "")
                    or str(current.get("source", "") or "") != str(candidate.get("source", "") or "")
                )
                if changed:
                    replaced_rows += 1
                else:
                    preserved_rows += 1
            else:
                replaced_rows += 1
            by_key[key] = candidate
            if changed:
                changed_accepted.append(copy.deepcopy(candidate))
            continue

        if current is not None and current_status == "accepted":
            preserved_rows += 1
            continue
        candidate = copy.deepcopy(item)
        if current is None:
            order.append(key)
            unresolved_new_rows += 1
        by_key[key] = candidate

    merged = [by_key[key] for key in order if key in by_key]
    return merged, {
        "existing_rows": existing_count,
        "incoming_rows": incoming_count,
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
    from engine.ocr_pipeline_diagnostics import (
        audit_canonical_decisions,
        audit_comparison,
        audit_fusion_states,
    )

    audit_comparison(comparison).raise_for_errors("无法导入 AI 裁决")
    audit_fusion_states(comparison, fusion_states).raise_for_errors("无法导入 AI 裁决")
    audit_canonical_decisions(comparison, decisions).raise_for_errors("无法导入 AI 裁决")

    by_group: dict[str, dict] = {}
    for item in decisions:
        if not isinstance(item, dict):
            continue
        group_id = str(item.get("sentence_group_id", "") or "")
        if not group_id:
            raise SourceCorrectionError("当前裁决缺少 sentence_group_id；拒绝列级旧裁决。")
        if group_id in by_group:
            raise SourceCorrectionError(f"裁决 sentence_group_id 重复：{group_id}")
        by_group[group_id] = item

    applied = 0
    for row, state in zip(comparison.rows, fusion_states):
        group_id = str(getattr(row, "sentence_group_id", "") or "")
        if not group_id:
            raise SourceCorrectionError("当前 OCR 对比行缺少 sentence_group_id。")
        decision = by_group.get(group_id)
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

        # Keep sealed export-time raw evidence available as explicit candidate
        # provenance; these evidence cards never write back to an OCR model.
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

        # Preserve the authoritative provenance across recovery/import.  The
        # history UI groups decisions by selection_origin; collapsing every
        # restored decision to ``ai_overlay`` would turn a local Paddle or human
        # verdict into a fake cloud-AI decision after restart.
        from engine.ocr_compare_view_model import (
            fusion_decision_origin_group, is_explicit_fusion_selection_origin,
        )
        restored_origin = source if is_explicit_fusion_selection_origin(source) else "ai_overlay"
        origin_group = fusion_decision_origin_group(restored_origin)
        if restored_origin == "local_targeted_retry_majority_adjudication":
            label = "本地重试多数裁决"
        elif origin_group == "human":
            label = "人工最终裁决"
        elif origin_group == "local_ai":
            label = "本地 AI 裁决"
        elif restored_origin == "ai_overlay":
            # Legacy AI sources such as ai_final_verdict are grouped as cloud
            # history through ai_overlay but keep the familiar candidate label.
            label = "AI逐模型纠错结果" if per_model else "AI最终裁决"
        elif origin_group == "cloud_ai":
            label = "云端 AI 裁决"
        else:
            label = "AI逐模型纠错结果" if per_model else "AI最终裁决"
        decision_audit_level = str(decision.get("audit_level", "") or "")
        index = upsert_external_candidate(
            state,
            final_text,
            display_label=label,
            select=True,
            reason=str(decision.get("reason", "") or "统一 canonical 裁决恢复；原 OCR 证据未修改。"),
            confidence=float(decision.get("confidence", 0.0) or 0.0),
            allow_empty=bool(decision.get("delete_intentionally", False)),
            transaction_id=str(decision.get("decision_id", "") or ""),
            transaction_operation=(
                "per_model_correction_overlay" if per_model else "canonical_text_verdict_overlay"
            ),
            transaction_member_ids=tuple(str(value) for value in (decision.get("column_ids") or [])),
            audit_level=decision_audit_level or "non_destructive_canonical_overlay",
            audit_flags=tuple(dict.fromkeys([
                *(str(value) for value in (decision.get("audit_flags") or [])),
                "raw_ocr_sources_preserved",
                "original_disagreement_visible",
                "canonical_provenance_preserved",
            ])),
            force_role_candidate=True,
            selection_origin=restored_origin,
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
            "sentence_group_id": str(getattr(row, "sentence_group_id", "") or ""),
            "column_ids": [str(value) for value in (row.column_ids or ())],
            "page": int(row.page or 0),
            "primary_block_id": str(row.primary_block_id or ""),
            "primary_segment_index": int(row.primary_segment_index or 0),
            "block_type": str(row.block_type or "paragraph"),
            "atomic": bool(row.atomic),
        })
    return result



def _repair_comparison_column_lineage(
    documents: Sequence[UnifiedDocument],
    comparison: MultiOcrComparison,
) -> tuple[int, list[int]]:
    """Recover missing row column IDs from exact primary-block lineage only.

    Role-based page/sentence OCR may transiently enter the text aligner when one
    model emits an extra fragment.  Export must not guess geometry, but it also
    should not fail when the canonical primary block already carries immutable
    ``source_column_ids``.  This helper performs only exact metadata recovery;
    ambiguous/reused IDs remain failures and are reported to the caller.
    """
    docs = list(documents)
    if not docs:
        return 0, [index for index, row in enumerate(comparison.rows) if not (row.column_ids or ())]
    primary = docs[0]
    used: set[str] = set()
    for row in comparison.rows:
        for value in (row.column_ids or ()):
            if str(value):
                used.add(str(value))
    repaired = 0
    unresolved: list[int] = []
    for row_index, row in enumerate(comparison.rows):
        if row.column_ids:
            continue
        block_indices = list(getattr(row, "primary_block_indices", ()) or ())
        if not block_indices and getattr(row, "primary_block_index", None) is not None:
            block_indices = [int(row.primary_block_index)]
        candidates: list[str] = []
        for block_index in block_indices:
            if not 0 <= int(block_index) < len(primary.blocks):
                continue
            block = primary.blocks[int(block_index)]
            metadata = _metadata(block)
            ids = _column_ids(metadata)
            if ids:
                candidates.extend(str(value) for value in ids if str(value))
        if not candidates and str(getattr(row, "primary_block_id", "") or ""):
            wanted = str(row.primary_block_id)
            matches = [block for block in primary.blocks if str(getattr(block, "id", "") or "") == wanted]
            if len(matches) == 1:
                candidates.extend(_column_ids(_metadata(matches[0])))
        ordered = list(dict.fromkeys(candidates))
        # Never steal a physical column already owned by another comparison row.
        if not ordered or any(value in used for value in ordered):
            unresolved.append(row_index)
            continue
        row.column_ids = tuple(ordered)
        used.update(ordered)
        repaired += 1
    remaining = [index for index, row in enumerate(comparison.rows) if not (row.column_ids or ())]
    unresolved = sorted(set(unresolved + remaining))
    if not unresolved and comparison.rows:
        # Exact physical lineage is now complete; downstream source correction
        # can safely use the strict column-consensus contract.
        comparison.alignment_mode = "column_id_consensus"
    return repaired, unresolved


def build_source_correction_payload(
    documents: Sequence[UnifiedDocument],
    labels: Sequence[str],
    comparison: MultiOcrComparison,
    *,
    canonical_decisions: Sequence[dict] | None = None,
    review_prior_decisions: bool = False,
    review_provisional_consensus: bool = False,
    review_common_mode_risk: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    docs = list(documents)
    if not 2 <= len(docs) <= 6:
        raise SourceCorrectionError("逐源纠错支持 2～6 个 OCR 模型。")
    _repaired_lineage, _unresolved_lineage = _repair_comparison_column_lineage(docs, comparison)
    if comparison.alignment_mode != "column_id_consensus" or _unresolved_lineage:
        detail = "、".join(str(index + 1) for index in _unresolved_lineage[:8])
        suffix = f"；仍缺列 ID 的行：{detail}" if detail else ""
        raise SourceCorrectionError("逐源纠错要求共享物理列 ID；请使用固定分列多模型 OCR 后再导出" + suffix + "。")
    _report_progress(progress_callback, "建立模型与物理列索引", 0, len(docs))
    registry, _snapshots = _build_model_registry_and_snapshots(docs, labels)
    _validate_current_model_registry(registry)
    _report_progress(progress_callback, "建立模型与物理列索引", len(docs), len(docs))
    model_ids = [item["model_id"] for item in registry]
    alignment = _alignment_snapshot(comparison)
    rows: list[dict] = []
    conflict_count = 0
    provisional_count = 0
    locked_count = 0
    prefilled_count = 0
    prefilled_native_count = 0
    stale_prior_count = 0
    reviewable_prior_count = 0
    pending_conflict_count = 0
    pending_provisional_count = 0
    prior_by_identity: dict[tuple[str, ...], dict] = {}
    for item in (canonical_decisions or []):
        if not isinstance(item, dict):
            continue
        copied = copy.deepcopy(item)
        identity = canonical_decision_key(copied)
        if identity:
            prior_by_identity[identity] = copied
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
        # The live comparison is authoritative for whether independently
        # executed OCR actually disagrees.  ``split_conflict_segments`` works on
        # raw strings so it can preserve precise per-model edits, but raw-only
        # layout differences (for example an NDL-inserted ASCII/full-width
        # space) may still produce an ``editable_conflict`` segment even when
        # the compare-only Unicode key says the sentence is equivalent.  Using
        # the raw segment type here used to re-open those rows during V5 export
        # and made the package contain more conflicts than the OCR Compare UI.
        actual_conflict = bool(getattr(row, "is_conflict", False))
        provisional = bool(getattr(row, "provisional_consensus", False))
        # A two-independent-model agreement with a seeded/skipped reviewer is
        # useful as a Lean Fast Path candidate, but it is not independent 3-way
        # consensus.  The default Lean export keeps these provisional rows
        # local; explicit strict/risk-audit callers can set review_provisional_consensus=True.
        review_required = bool(actual_conflict or (provisional and review_provisional_consensus))
        if actual_conflict:
            conflict_count += 1
        elif provisional:
            provisional_count += 1
        else:
            locked_count += 1
        # Preserve accepted adjudication history even after corrections turn a
        # formerly conflicting row into exact/provisional consensus.
        row_identity = canonical_decision_key(row)
        if not str(getattr(row, "sentence_group_id", "") or ""):
            raise SourceCorrectionError(
                f"第 {row_index + 1} 行缺少 sentence_group_id；当前裁决包不接受旧式列级身份。"
            )
        prior_decision = prior_by_identity.get(row_identity)
        resolved_verdict, prior_state = _accepted_decision_for_export(
            prior_decision,
            column_ids=column_ids,
            sentence_group_id=str(getattr(row, "sentence_group_id", "") or ""),
            model_ids=model_ids,
            texts=texts,
        )
        # Explicit re-review is limited to *prior AI* verdicts. Human/local
        # canonical decisions stay locked, so UI state, recovery state and
        # AI_OUTPUT can never disagree about whether the row is resolved.
        prior_reviewable = bool(
            review_prior_decisions
            and actual_conflict
            and resolved_verdict is not None
            and _canonical_decision_can_be_reopened_for_ai_review(resolved_verdict)
        )
        prefilled = bool(resolved_verdict is not None and not prior_reviewable)
        editable = bool(review_required and not prefilled)
        if prior_reviewable:
            reviewable_prior_count += 1
        elif prefilled:
            prefilled_count += 1
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
            "model_evidence": [
                {
                    "model_id": model_id,
                    "model_index": index,
                    "display_label": str(registry[index].get("display_label", "") or ""),
                    "source_engine": str(registry[index].get("source_engine", "") or ""),
                    "role": str(registry[index].get("role", "") or ""),
                    "role_label": str(registry[index].get("role_label", "") or ""),
                    "input_granularity": str(registry[index].get("input_granularity", "unknown") or "unknown"),
                    "seeded_reuse": index in set(getattr(row, "consensus_seeded_models", ()) or ()),
                    "independently_executed": index not in set(getattr(row, "consensus_seeded_models", ()) or ()),
                    "text": texts[index],
                }
                for index, model_id in enumerate(model_ids)
            ],
            "independent_model_ids": [
                model_id for index, model_id in enumerate(model_ids)
                if index not in set(getattr(row, "consensus_seeded_models", ()) or ())
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
                "decision_id": _canonical_decision_id(
                    column_ids, str(getattr(row, "sentence_group_id", "") or "")
                ),
                "final_text": "",
                "reason": "",
                "confidence": 0.0,
                "delete_intentionally": False,
            }
        elif prefilled:
            row_payload["resolved_verdict"] = resolved_verdict
        rows.append(row_payload)

    # Final common-mode audit: ordinary conflict routing cannot see rows where
    # the independently executed OCRs agree on the same mistake.  Re-open only
    # a small deterministic risk queue; never rewrite these rows locally.
    pending_common_mode_count = 0
    common_mode_risk_items: list[dict] = []
    if review_common_mode_risk:
        from engine.ocr_common_mode_guard import select_common_mode_risks
        common_mode_risk_items = select_common_mode_risks(rows)
        for risk in common_mode_risk_items:
            row_index = int(risk.get("row_index", -1))  # 0 is a valid first-row index
            if not 0 <= row_index < len(rows):
                continue
            item = rows[row_index]
            if bool(item.get("editable")):
                continue
            if str(item.get("status", "") or "") not in {"exact_consensus", "provisional_consensus_auto"}:
                continue
            column_ids = [str(value) for value in (item.get("column_ids") or []) if str(value)]
            sentence_group_id = str(item.get("sentence_group_id", "") or "")
            if not column_ids or not sentence_group_id:
                continue
            item["editable"] = True
            item["review_required"] = True
            item["status"] = "common_mode_risk"
            item["decision_state"] = "pending_common_mode_review"
            item["decision_mode"] = "replace_whole_column_group"
            item["common_mode_risk"] = copy.deepcopy(risk)
            item["ai_verdict"] = {
                "decision_id": _canonical_decision_id(column_ids, sentence_group_id),
                "final_text": "",
                "reason": "",
                "confidence": 0.0,
                "delete_intentionally": False,
            }
            pending_common_mode_count += 1

    locked_count = sum(
        1 for item in rows
        if (not bool(item.get("editable"))) and str(item.get("status", "") or "") == "exact_consensus"
    )
    payload = {
        "schema": CANONICAL_CORRECTIONS_SCHEMA,
        "package_schema": SCHEMA,
        "package_id": uuid.uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "exchange_version": EXCHANGE_VERSION,
        "exchange_profile": EXCHANGE_PROFILE,
        "multi_ocr_role_schema": max([int(item.get("role_schema", 0) or 0) for item in registry] or [0]),
        "multi_ocr_roles_executed": [
            str(item.get("role", "") or "") for item in registry if str(item.get("role", "") or "")
        ],
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
            "role_evidence_rule": (
                "model_evidence 中 role/input_granularity 表示逐列、整页、全列或分歧列复核来源；"
                "seeded_reuse=true 仅用于保持句结构，不是独立 OCR 证据，严禁计票。"
            ),
            "locked_rule": (
                (
                    "exact_consensus 不可修改；provisional_consensus 是两份独立 OCR 一致、其余模型 seeded/skipped 的共同候选，"
                    "本 V5 严格包要求 AI/人工核验，不得把 seeded_reuse 当独立票；此前已接受的可复审 AI 裁决作为 prior_decision_context 提供。"
                )
                if review_provisional_consensus else
                (
                    "exact_consensus 与 provisional_consensus_auto 不可修改；此前已接受的冲突裁决在本包中作为 prior_decision_context 重新开放复审。"
                    if review_prior_decisions else
                    "exact_consensus、provisional_consensus_auto 与 resolved_prior_canonical 均不可修改；共同候选按 v8 规则自动保留，已接受裁决会重新校验后锁定。"
                )
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
                "alignment_snapshot_sha256", "row_id", "row_index", "sentence_group_id", "column_ids",
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
        "review_provisional_consensus": bool(review_provisional_consensus),
        "editable_provisional_rows": pending_provisional_count,
        "editable_review_rows": pending_conflict_count + pending_provisional_count + pending_common_mode_count + reviewable_prior_count,
        "pending_conflict_rows": pending_conflict_count,
        "pending_provisional_rows": pending_provisional_count,
        "pending_common_mode_rows": pending_common_mode_count,
        "pending_review_rows": pending_conflict_count + pending_provisional_count + pending_common_mode_count,
        "common_mode_review_enabled": bool(review_common_mode_risk),
        "prefilled_prior_decision_rows": prefilled_count,
        "prefilled_native_decision_rows": prefilled_native_count,
        "stale_prior_decision_rows": stale_prior_count,
        "prior_decision_review_enabled": bool(review_prior_decisions),
        "prior_decision_review_rows": reviewable_prior_count,
        "locked_consensus_rows": locked_count,
        "coverage_contract": {
            "profile": "strict_provisional_review" if review_provisional_consensus else "lean_provisional_auto",
            "exact_consensus_locked": True,
            "provisional_consensus_reviewed": bool(review_provisional_consensus),
            "seeded_reuse_is_independent_vote": False,
            "common_mode_error_can_remain_in_exact_consensus": True,
            "common_mode_risk_reviewed": bool(review_common_mode_risk),
            "common_mode_risk_is_advisory_not_autocorrect": True,
            "guarantees_all_ocr_errors_fixed": False,
        },
        "rows": rows,
    }
    payload["immutable_manifest_sha256"] = _sha256(_immutable_projection(payload))
    return payload


def _ai_compact_candidate_key(text: str) -> str:
    """Stable Unicode-only candidate key for GPT-facing deduplication."""
    import unicodedata
    return unicodedata.normalize("NFC", str(text or "")).strip()


def _compact_ai_candidates(row: dict) -> list[dict]:
    """Return only genuinely independent OCR observations, merged by text."""
    merged: list[dict] = []
    by_key: dict[tuple[str, bool], int] = {}
    for evidence in row.get("model_evidence", []) or []:
        if not isinstance(evidence, dict):
            continue
        if not bool(evidence.get("independently_executed", True)):
            continue
        text = str(evidence.get("text", "") or "")
        failed = (not text.strip()) or any(marker in text for marker in ("□", "�"))
        display = str(evidence.get("display_label", "") or evidence.get("model_id", ""))
        granularity = str(evidence.get("input_granularity", "") or "")
        key = ("" if failed else _ai_compact_candidate_key(text), bool(failed))
        if key in by_key:
            item = merged[by_key[key]]
            if display and display not in item["m"]:
                item["m"].append(display)
            if granularity and granularity not in item["g"]:
                item["g"].append(granularity)
            continue
        by_key[key] = len(merged)
        merged.append({
            "m": [display] if display else [],
            "t": "" if failed else text,
            "fail": bool(failed),
            "g": [granularity] if granularity else [],
        })
    return merged


_AI_QUICK_CLI = r'''#!/usr/bin/env python3
import argparse, gzip, json, os, pathlib, zipfile
ROOT = pathlib.Path(__file__).resolve().parent
TASKS = ROOT / "04_ai_tasks_compact.jsonl"
ANSWERS = ROOT / "AI_OUTPUT" / "answers.jsonl"
AUTH = ROOT / "_BINDINGS" / "authority.json.gz"
BIND = ROOT / "_BINDINGS" / "tasks.json.gz"
OUT = ROOT / "AI_IMPORT.zip"

def read_jsonl(path):
    if not path.exists(): return []
    out=[]
    for no,line in enumerate(path.read_text("utf-8").splitlines(),1):
        if not line.strip(): continue
        try: obj=json.loads(line)
        except Exception as exc: raise SystemExit(f"{path.name}:{no}: invalid JSON: {exc}")
        if not isinstance(obj,dict): raise SystemExit(f"{path.name}:{no}: object required")
        out.append(obj)
    return out

def load_gz(path):
    return json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))

def status():
    tasks=read_jsonl(TASKS); answers=read_jsonl(ANSWERS)
    ids={str(x.get("id")) for x in answers}
    batches=[]
    batch_size=40
    for start in range(0,len(tasks),batch_size):
        chunk=tasks[start:start+batch_size]
        answered=sum(str(t.get("id")) in ids for t in chunk)
        batches.append({
            "batch":len(batches)+1,
            "first_id":str(chunk[0].get("id")) if chunk else "",
            "last_id":str(chunk[-1].get("id")) if chunk else "",
            "tasks":len(chunk),
            "answered":answered,
            "remaining":len(chunk)-answered,
        })
    print(json.dumps({
        "tasks":len(tasks),
        "answered":sum(str(t.get("id")) in ids for t in tasks),
        "remaining":sum(str(t.get("id")) not in ids for t in tasks),
        "batches":batches,
    },ensure_ascii=False))

def _validate_answers(require_complete=False):
    tasks={str(x.get("id")):x for x in read_jsonl(TASKS)}
    answers=read_jsonl(ANSWERS)
    seen=set(); errors=[]
    for ans in answers:
        tid=str(ans.get("id") or "")
        if not tid or tid not in tasks:
            errors.append(f"unknown task id: {tid!r}"); continue
        if tid in seen:
            errors.append(f"duplicate task id: {tid}"); continue
        seen.add(tid)
        if "delete_intentionally" in ans and ans["delete_intentionally"] is not True:
            errors.append(f"delete_intentionally must be literal true when supplied: {tid}"); continue
        has_delete=ans.get("delete_intentionally") is True
        if bool(ans.get("unresolved",False)):
            if "pick" in ans or "text" in ans or has_delete:
                errors.append(f"unresolved answer must not also contain pick/text/delete: {tid}")
            continue
        has_pick="pick" in ans; has_text="text" in ans
        if sum((has_pick,has_text,has_delete)) != 1:
            errors.append(f"answer needs exactly one of pick/text/delete_intentionally/unresolved: {tid}"); continue
        if has_delete:
            pass  # Deliberate whole-row omission, with a sealed row binding.
        elif has_pick:
            raw_pick=ans["pick"]
            if type(raw_pick) is not int:
                errors.append(f"pick must be integer: {tid}"); continue
            pick=raw_pick
            candidates=tasks[tid].get("c") or []
            if pick<0 or pick>=len(candidates):
                errors.append(f"pick out of range: {tid}"); continue
            if (bool(candidates[pick].get("f"))
                    or bool(candidates[pick].get("fail"))
                    or not str(candidates[pick].get("t") or "").strip()):
                errors.append(f"cannot pick failed OCR candidate: {tid}")
        else:
            if not str(ans.get("text") or "").strip():
                errors.append(f"empty final text: {tid}")
        try:
            conf=float(ans.get("confidence",0.98))
            if not 0.0<=conf<=1.0: errors.append(f"confidence out of range: {tid}")
        except Exception:
            errors.append(f"invalid confidence: {tid}")
    missing=[tid for tid in tasks if tid not in seen]
    if require_complete and missing:
        errors.append(f"missing answers: {len(missing)} (first: {', '.join(missing[:8])})")
    if errors:
        raise SystemExit("\n".join(errors[:50]))
    return tasks,answers,missing

def validate():
    tasks,answers,missing=_validate_answers(require_complete=False)
    print(json.dumps({"valid":True,"tasks":len(tasks),"answers":len(answers),"remaining":len(missing)},ensure_ascii=False))

def finish():
    tasks,answers,missing=_validate_answers(require_complete=True)
    bindings={str(x.get("id")):x for x in load_gz(BIND)}
    seen=set(); auth=load_gz(AUTH); rows=auth.get("rows") or []
    for ans in answers:
        tid=str(ans.get("id") or "")
        if tid in seen: raise SystemExit(f"duplicate task id: {tid}")
        seen.add(tid)
        if bool(ans.get("unresolved",False)): continue
        task=tasks[tid]; binding=bindings.get(tid)
        if binding is None: raise SystemExit(f"missing sealed task binding: {tid}")
        idx=int(binding["row"])
        if idx<0 or idx>=len(rows): raise SystemExit(f"row out of range: {tid}")
        row=rows[idx]
        if str(row.get("row_id"))!=str(binding.get("row_id")): raise SystemExit(f"row binding mismatch: {tid}")
        if str(row.get("base_row_sha256"))!=str(binding.get("base_row_sha256")): raise SystemExit(f"base hash mismatch: {tid}")
        delete_intentionally=ans.get("delete_intentionally") is True
        if delete_intentionally:
            text=""
        elif "pick" in ans:
            pick=int(ans["pick"]); chosen=(task.get("c") or [])[pick]; text=str(chosen.get("t") or "")
        else:
            text=str(ans.get("text") or "")
        verdict=row.setdefault("ai_verdict",{})
        verdict["final_text"]=text
        verdict["reason"]=str(ans.get("reason") or "GPT adjudication")[:1000]
        try: conf=float(ans.get("confidence",0.98))
        except Exception: conf=0.98
        verdict["confidence"]=max(0.0,min(1.0,conf))
        verdict["delete_intentionally"]=delete_intentionally
    payload=json.dumps(auth,ensure_ascii=False,separators=(",",":")).encode("utf-8")
    tmp=OUT.with_name(f".{OUT.name}.tmp")
    try:
        with zipfile.ZipFile(tmp,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            z.writestr("AI_OUTPUT/model_corrections.json",payload)
            if ANSWERS.exists(): z.write(ANSWERS,"AI_OUTPUT/answers.jsonl")
        with zipfile.ZipFile(tmp,"r") as check:
            bad=check.testzip()
            if bad: raise SystemExit(f"AI_IMPORT ZIP CRC failed: {bad}")
        os.replace(tmp,OUT)
    finally:
        try: tmp.unlink()
        except FileNotFoundError: pass
    print(json.dumps({"output":str(OUT),"answers":len(answers),"tasks":len(tasks)},ensure_ascii=False))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("command",choices=["status","validate","finish"]); ns=ap.parse_args()
    status() if ns.command=="status" else (validate() if ns.command=="validate" else finish())
if __name__=="__main__": main()
'''


def _ai_short_model_label(label: str) -> str:
    value = str(label or "")
    folded = value.casefold()
    if "hayai" in folded:
        return "Hayai"
    if "ndlocr" in folded or "ndl" in folded:
        return "NDL"
    if "48px" in folded:
        return "48px"
    if "apple" in folded or "macos ocr" in folded:
        return "Apple"
    if "paddle" in folded:
        return "Paddle"
    # Remove role prefixes such as "整页主模型 · " without guessing model identity.
    return value.split("·")[-1].strip() or value


_AI_CONTEXT_TRIM = " \t\r\n　、。！？!?「」『』（）()［］[]【】〈〉《》…‥・：:；;—―-"
_AI_KATAKANA_TERM_RE = re.compile(r"[ァ-ヶー]{2,16}")
_AI_HIRAGANA_ONLY_TERM_RE = re.compile(r"^[ぁ-ゖ]+$")
_AI_KATAKANA_ONLY_TERM_RE = re.compile(r"^[ァ-ヶー]+$")
_AI_INFORMATIONAL_TERM_RE = re.compile(r"[一-龯々〆ヵヶァ-ヶー0-9０-９]")


def _ai_stable_context_text(row: dict) -> str:
    """Return only high-trust in-workspace text for web-GPT context anchors.

    The exported web bundle must not manufacture a reference truth.  Only exact
    independently executed OCR consensus can contribute global context.  Fresh
    conflict/provisional rows and prior adjudication results are excluded.
    """
    status = str(row.get("status", "") or "")
    if status == "exact_consensus" and not bool(row.get("editable")):
        evidence = [
            str(item.get("text", "") or "")
            for item in (row.get("model_evidence") or [])
            if isinstance(item, dict) and bool(item.get("independently_executed", True))
            and str(item.get("text", "") or "").strip()
        ]
        if evidence:
            normalized = {_ai_compact_candidate_key(value) for value in evidence}
            if len(normalized) == 1:
                return evidence[0]
    # Do not use prior AI/human adjudications as context truth.  The web-GPT
    # bundle is meant to improve *fresh* adjudication from OCR evidence, so
    # global anchors come from independently executed exact-consensus OCR only.
    return ""


def _ai_variant_windows(candidate_texts: Sequence[str], *, limit: int = 20) -> list[str]:
    """Extract only ambiguity-bearing terms from candidate differences.

    Do not export every proper name that merely happens to occur in a disputed
    sentence.  Whole-book web packages become much smaller when shared context
    is attached only to the actual differing span.  If a changed span touches a
    Katakana token, export the full token (e.g. ニノ, グディオン), not fragments
    such as オン/ール.  Non-Katakana changes keep a short one/two-character
    window so 目/日 or 魔王/魔土 can still find stable book evidence.
    """
    texts = [str(value or "") for value in candidate_texts if str(value or "").strip()]
    unique = list(dict.fromkeys(texts))
    terms: list[str] = []

    def add(value: str, *, from_diff: bool = False) -> None:
        value = str(value or "").strip(_AI_CONTEXT_TRIM)
        if len(value) < 2 or len(value) > 18:
            return
        if _AI_HIRAGANA_ONLY_TERM_RE.fullmatch(value):
            return
        if not _AI_INFORMATIONAL_TERM_RE.search(value):
            return
        if from_diff and _AI_KATAKANA_ONLY_TERM_RE.fullmatch(value):
            # Full Katakana tokens are added explicitly from the changed span.
            return
        if value not in terms:
            terms.append(value)

    def add_overlapping_katakana(text: str, start: int, end: int) -> None:
        if not text:
            return
        probe_start = max(0, start - 1)
        probe_end = min(len(text), max(end, start + 1) + 1)
        for match in _AI_KATAKANA_TERM_RE.finditer(text):
            if match.end() <= probe_start or match.start() >= probe_end:
                continue
            add(match.group(0))

    for left_index in range(len(unique)):
        for right_index in range(left_index + 1, len(unique)):
            left, right = unique[left_index], unique[right_index]
            matcher = SequenceMatcher(None, left, right, autojunk=False)
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == "equal":
                    continue
                add_overlapping_katakana(left, i1, i2)
                add_overlapping_katakana(right, j1, j2)
                for text, start, end in ((left, i1, i2), (right, j1, j2)):
                    if not text:
                        continue
                    if start == end:
                        start = max(0, start - 1)
                        end = min(len(text), end + 1)
                    for radius in (1, 2):
                        window = text[max(0, start - radius):min(len(text), end + radius)]
                        add(window, from_diff=True)
                        if len(terms) >= limit:
                            return terms
    return terms

def _ai_context_snippet(text: str, term: str, *, radius: int = 42) -> str:
    value = str(text or "")
    pos = value.find(term)
    if pos < 0:
        return value[: max(32, radius * 2)]
    start = max(0, pos - radius)
    end = min(len(value), pos + len(term) + radius)
    return value[start:end]


def _build_ai_shared_context(
    rows: Sequence[dict], tasks: Sequence[dict]
) -> tuple[dict[str, list[str]], list[dict]]:
    """Build a de-duplicated book-context index for web GPT review.

    Batch 41 embedded up to six full anchor objects in every task.  On a real
    1,715-task book that tripled the visible JSONL size because popular names
    such as ``ニノ`` and ``シオン`` repeated the same examples hundreds of
    times.  Store each anchor once and let tasks reference compact ``Axxxxx``
    IDs instead.
    """
    stable_rows: list[tuple[str, int, str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = _ai_stable_context_text(row)
        if not text.strip():
            continue
        stable_rows.append((
            str(row.get("row_id", "") or ""),
            int(row.get("page", 0) or 0),
            text,
            "locked_exact_consensus" if str(row.get("status", "") or "") == "exact_consensus" else "locked_canonical",
        ))

    task_terms: dict[str, list[str]] = {}
    all_terms: list[str] = []
    for task in tasks:
        task_id = str(task.get("id", "") or "")
        candidate_texts = [
            str(item.get("t", "") or "")
            for item in (task.get("c") or [])
            if isinstance(item, dict) and not bool(item.get("fail", False))
        ]
        terms = _ai_variant_windows(candidate_texts)
        task_terms[task_id] = terms
        for term in terms:
            if term not in all_terms:
                all_terms.append(term)

    raw_anchors: dict[str, dict] = {}
    for term in all_terms:
        count = 0
        examples: list[dict] = []
        for row_id, page, text, source in stable_rows:
            occurrences = text.count(term)
            if occurrences <= 0:
                continue
            count += occurrences
            if len(examples) < 3:
                examples.append({
                    "row_id": row_id,
                    "page": page,
                    "text": _ai_context_snippet(text, term),
                    "source": source,
                })
        if count > 0:
            raw_anchors[term] = {"t": term, "n": count, "e": examples}

    # Stable deterministic IDs make package diffs and resumable web review easy.
    ordered_terms = sorted(
        raw_anchors,
        key=lambda term: (-int(raw_anchors[term].get("n", 0) or 0), len(term), term),
    )
    anchors: list[dict] = []
    id_by_term: dict[str, str] = {}
    for index, term in enumerate(ordered_terms, start=1):
        anchor_id = f"A{index:05d}"
        id_by_term[term] = anchor_id
        anchors.append({"id": anchor_id, **raw_anchors[term]})

    context_by_task: dict[str, list[str]] = {}
    for task_id, terms in task_terms.items():
        ids = [id_by_term[term] for term in terms if term in id_by_term]
        if ids:
            context_by_task[task_id] = ids[:6]
    return context_by_task, anchors


def _build_ai_task_context(rows: Sequence[dict], tasks: Sequence[dict]) -> dict[str, list[dict]]:
    """Backward-compatible expanded view used by older internal callers/tests."""
    context_by_task, anchors = _build_ai_shared_context(rows, tasks)
    by_id = {str(item.get("id", "")): item for item in anchors}
    return {
        task_id: [copy.deepcopy(by_id[anchor_id]) for anchor_id in ids if anchor_id in by_id]
        for task_id, ids in context_by_task.items()
    }


def _build_ai_consistency_groups(
    context_by_task: dict[str, list[str]], anchors: Sequence[dict]
) -> list[dict]:
    by_id = {str(item.get("id", "")): item for item in anchors}
    grouped: dict[str, dict] = {}
    for task_id, anchor_ids in context_by_task.items():
        for anchor_id in anchor_ids:
            anchor = by_id.get(str(anchor_id))
            if not anchor:
                continue
            bucket = grouped.setdefault(str(anchor_id), {
                "anchor_id": str(anchor_id),
                "term": str(anchor.get("t", "") or ""),
                "task_ids": [],
                "stable_count": int(anchor.get("n", 0) or 0),
            })
            if task_id not in bucket["task_ids"]:
                bucket["task_ids"].append(task_id)
    groups = [item for item in grouped.values() if len(item["task_ids"]) >= 2]
    groups.sort(key=lambda item: (-len(item["task_ids"]), -int(item["stable_count"] or 0), item["term"]))
    return groups[:200]


def _ai_group_member_hint(task: dict, term: str) -> dict:
    """Describe how a strong term anchor applies to one visible review row.

    This is deliberately a *hint*, not an automatic answer.  A web reviewer can
    settle a repeated proper-name/term once at group level, then reuse that
    decision for rows whose only candidate difference lies inside that term.
    Rows with any residual punctuation/lexical difference remain in the normal
    row-review pass.
    """
    term = str(term or "")
    valid_candidates: list[tuple[int, str]] = []
    for index, candidate in enumerate(task.get("c") or []):
        if not isinstance(candidate, dict) or bool(candidate.get("f", False)):
            continue
        text = str(candidate.get("t", "") or "")
        if text:
            valid_candidates.append((index, text))

    anchor_picks = [index for index, text in valid_candidates if term and term in text]
    anchor_only = False
    if len(anchor_picks) == 1 and not task.get("r"):
        anchor_index = anchor_picks[0]
        anchor_text = next(text for index, text in valid_candidates if index == anchor_index)
        spans: list[tuple[int, int]] = []
        start = 0
        while term:
            position = anchor_text.find(term, start)
            if position < 0:
                break
            spans.append((position, position + len(term)))
            start = position + len(term)

        if spans:
            anchor_only = True
            for other_index, other_text in valid_candidates:
                if other_index == anchor_index:
                    continue
                matcher = SequenceMatcher(None, anchor_text, other_text, autojunk=False)
                for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
                    if tag == "equal":
                        continue
                    if i1 == i2:
                        inside_term = any(left <= i1 <= right for left, right in spans)
                    else:
                        inside_term = any(i1 < right and i2 > left for left, right in spans)
                    if not inside_term:
                        anchor_only = False
                        break
                if not anchor_only:
                    break

    hint = {
        "id": str(task.get("id", "") or ""),
        "anchor_picks": anchor_picks,
        "anchor_only": bool(anchor_only),
    }
    if anchor_only and len(anchor_picks) == 1:
        hint["direct_pick"] = anchor_picks[0]
    return hint


_AI_VISUAL_NUMERIC_GLYPHS = frozenset("0123456789０１２３４５６７８９一二三四五六七八九十百千万億兆零〇")
_AI_VISUAL_NEGATION_TOKENS = (
    "ない", "無い", "ぬ", "ず", "ません", "じゃない", "ではない", "なく", "なかった", "なければ",
)
_AI_VISUAL_PUNCT_RE = re.compile(r"[\s、。！？!?…‥・「」『』（）()［］【】〈〉《》〔〕—―─ー〜～:：;；,.，．\-]+")
_AI_SMALL_KANA = frozenset("ぁぃぅぇぉゃゅょっゎァィゥェォャュョッヮヵヶ")
_AI_REPEAT_KANA_RE = re.compile(r"([ぁ-んァ-ヶー])\1{2,}")
_AI_STATUS_LEADING_NUMBER_RE = re.compile(r"^[\s「『【]*(?:[0-9０-９]{1,3})(?=名前|種族|ランク|レベル|職業)")
_AI_NUMERIC_MEASUREMENT_RE = re.compile(
    r"[0-9０-９一二三四五六七八九十百千万億兆零〇]+"
    r"(?:年|歳|人|匹|個|回|階|日|時間|時|分|秒|枚|本|体|頭|羽|冊|巻|章|レベル|以上|以下|程度)"
)
_AI_DIALOGUE_MARKS = frozenset("「」『』!?！？")


def _ai_candidate_diff_hints(task: dict, *, limit: int = 6) -> list[dict]:
    """Return compact candidate-only diff hints for visual focus.

    Hints contain no reference truth and never choose an answer.  They merely
    expose the exact snippets that differ so a web reviewer does not miss a
    one-glyph conflict inside a long vertical crop.
    """
    candidates = [
        candidate for candidate in (task.get("c") or [])
        if isinstance(candidate, dict) and not bool(candidate.get("f", False))
        and str(candidate.get("t", "") or "")
    ]
    hints: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for left_index, left in enumerate(candidates):
        left_text = str(left.get("t", "") or "")
        for right in candidates[left_index + 1:]:
            right_text = str(right.get("t", "") or "")
            matcher = SequenceMatcher(None, left_text, right_text, autojunk=False)
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == "equal":
                    continue
                a = left_text[i1:i2]
                b = right_text[j1:j2]
                if not a and not b:
                    continue
                key = (a, b)
                reverse = (b, a)
                if key in seen or reverse in seen:
                    continue
                seen.add(key)
                hints.append({
                    "a": a, "b": b,
                    "a_models": list(left.get("m") or []),
                    "b_models": list(right.get("m") or []),
                    "micro": max(len(a), len(b)) <= 2,
                })
                if len(hints) >= limit:
                    return hints
    return hints


def _ai_single_lexical_substitution(task: dict) -> tuple[list[str], int, str, str] | None:
    """Return the sole lexical substitution after punctuation stripping.

    The helper is deliberately structural: it never evaluates which glyph is
    correct. It is used only to decide whether one tiny disagreement deserves
    pixel review.
    """
    forms: list[str] = []
    for candidate in (task.get("c") or []):
        if not isinstance(candidate, dict) or bool(candidate.get("f", False)):
            continue
        text = str(candidate.get("t", "") or "")
        if not text:
            continue
        lexical = _AI_VISUAL_PUNCT_RE.sub("", text)
        if lexical not in forms:
            forms.append(lexical)
    if len(forms) != 2 or len(forms[0]) != len(forms[1]):
        return None
    changes = [(index, left, right) for index, (left, right) in enumerate(zip(forms[0], forms[1])) if left != right]
    if len(changes) != 1:
        return None
    index, left, right = changes[0]
    return forms, index, left, right


def _ai_is_cjk_glyph(value: str) -> bool:
    if len(value) != 1:
        return False
    code = ord(value)
    return 0x3400 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF


def _ai_is_hiragana(value: str) -> bool:
    return len(value) == 1 and 0x3040 <= ord(value) <= 0x309F


def _ai_hard_visual_guard_reasons(task: dict) -> list[str]:
    """Candidate-only hard-error guards learned from failure *types*, not truth.

    These guards are intentionally generic.  They do not embed any corrected
    text from post-hoc references; they only detect evidence shapes that proved
    unsafe to settle from model count or language plausibility alone.
    """
    candidates = [
        candidate for candidate in (task.get("c") or [])
        if isinstance(candidate, dict)
    ]
    good = [candidate for candidate in candidates if not bool(candidate.get("f", False)) and str(candidate.get("t", "") or "")]
    texts = list(dict.fromkeys(str(candidate.get("t", "") or "") for candidate in good))
    if len(texts) < 2:
        return []

    reasons: list[str] = []
    hints = _ai_candidate_diff_hints(task, limit=12)
    lexical_hints = [
        hint for hint in hints
        if _AI_VISUAL_PUNCT_RE.sub("", str(hint.get("a", "") or "") + str(hint.get("b", "") or ""))
    ]

    # A 2:1 majority is not image truth, but forcing every one/two-character
    # disagreement to pixels is too expensive. Escalate only the narrow class
    # that repeatedly caused real book errors: one CJK substitution with a
    # majority-backed candidate, or a one-kana passive/causative verb-stem
    # substitution immediately before ``される`` / related forms.
    supports = [len(candidate.get("m") or []) for candidate in good]
    substitution = _ai_single_lexical_substitution(task)
    if supports and max(supports) >= 2 and substitution is not None:
        forms, index, left, right = substitution
        if _ai_is_cjk_glyph(left) and _ai_is_cjk_glyph(right):
            reasons.append("majority_single_cjk_substitution")
        elif _ai_is_hiragana(left) and _ai_is_hiragana(right):
            left_tail = forms[0][index + 1:index + 7]
            right_tail = forms[1][index + 1:index + 7]
            passive_suffixes = ("される", "された", "されて", "されれ", "させる", "させた")
            if left_tail.startswith(passive_suffixes) or right_tail.startswith(passive_suffixes):
                reasons.append("passive_verb_stem_micro_conflict")

    # If one recognizer failed entirely, the remaining disagreement has less
    # independent redundancy than its row count suggests.
    if any(bool(candidate.get("f", False)) or not str(candidate.get("t", "") or "") for candidate in candidates):
        reasons.append("failed_model_with_surviving_conflict")

    # Small-kana and emphatic dialogue length are semantically/voice relevant
    # but easy to miss in compact crops.
    dialogue_like = any(any(mark in text for mark in _AI_DIALOGUE_MARKS) for text in texts)
    if dialogue_like and any(any(char in _AI_SMALL_KANA for char in (str(h.get("a", "")) + str(h.get("b", "")))) for h in hints):
        reasons.append("dialogue_small_kana_conflict")
    if dialogue_like and any(_AI_REPEAT_KANA_RE.search(text) for text in texts):
        reasons.append("dialogue_repeated_kana_emphasis")

    # Status cards frequently share the crop with a printed page number.  The
    # number must not be copied into body text just because multiple OCR engines
    # saw it.
    if any(_AI_STATUS_LEADING_NUMBER_RE.search(text) for text in texts):
        reasons.append("leading_status_page_number_risk")

    # Common-mode numeric errors are invisible to a candidate-diff-only gate:
    # every recognizer can agree on the same wrong number while disagreeing on
    # an unrelated name. If all surviving candidates contain the same explicit
    # measurement/counter token, require pixels for that measurement too.
    measurement_sets = [set(_AI_NUMERIC_MEASUREMENT_RE.findall(text)) for text in texts]
    if measurement_sets and set.intersection(*measurement_sets):
        reasons.append("shared_numeric_measurement_in_disagreement")

    return list(dict.fromkeys(reasons))


def _ai_visual_focus_hints(task: dict, reasons: Sequence[str]) -> list[str]:
    """Human/GPT-facing attention hints that never encode an answer."""
    hints: list[str] = []
    reason_set = set(str(value) for value in reasons)
    if "failed_model_with_surviving_conflict" in reason_set:
        hints.append("One OCR engine failed; do not treat the two surviving candidates as a reliable majority.")
    if "majority_single_cjk_substitution" in reason_set or "passive_verb_stem_micro_conflict" in reason_set:
        hints.append("Inspect the exact one-glyph lexical substitution shown in diff_hints; model count is not image truth.")
    if "dialogue_small_kana_conflict" in reason_set or "dialogue_repeated_kana_emphasis" in reason_set:
        hints.append("Verify small kana, repeated-kana count, ellipsis and any trailing prolonged-sound mark (ー) at the dialogue tail.")
    if "leading_status_page_number_risk" in reason_set:
        hints.append("Check whether leading digits are a printed page number outside the status-card text; do not copy them into body text.")
    if "shared_numeric_measurement_in_disagreement" in reason_set:
        hints.append("Verify every numeric measurement/counter directly from pixels even when all candidates agree on the digits.")
    if "number_or_numeric_lookalike" in reason_set:
        hints.append("Verify the differing number/numeric-lookalike glyph directly from pixels.")
    if "punctuation_or_geometry" in reason_set:
        hints.append("Verify punctuation/long-line geometry from pixels; Unicode lookalikes may be layout-equivalent but are not automatically interchangeable.")
    return list(dict.fromkeys(hints))


def _ai_visual_review_reasons(task: dict) -> list[str]:
    """Return conservative reasons that make source pixels mandatory.

    This does *not* auto-resolve a row.  It only moves rows with known
    high-risk evidence patterns to the visual pass so web review can handle
    cheap text/context rows first.  False positives are acceptable here;
    false negatives must still be allowed to open the original row image.
    """
    reasons: list[str] = []
    if task.get("r"):
        reasons.append("common_mode")

    texts = [
        str(candidate.get("t", "") or "")
        for candidate in (task.get("c") or [])
        if isinstance(candidate, dict) and not bool(candidate.get("f", False)) and str(candidate.get("t", "") or "")
    ]
    texts = list(dict.fromkeys(texts))
    if len(texts) < 2:
        return reasons

    diff_chunks: list[tuple[str, str]] = []
    base = texts[0]
    for other in texts[1:]:
        matcher = SequenceMatcher(None, base, other, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                diff_chunks.append((base[i1:i2], other[j1:j2]))

    if any(
        any(char in _AI_VISUAL_NUMERIC_GLYPHS for char in (left + right))
        for left, right in diff_chunks
    ):
        reasons.append("number_or_numeric_lookalike")

    for token in _AI_VISUAL_NEGATION_TOKENS:
        if len({token in text for text in texts}) > 1:
            reasons.append("negation")
            break

    lexical_forms = {_AI_VISUAL_PUNCT_RE.sub("", text) for text in texts}
    if len(lexical_forms) == 1 and len(texts) > 1:
        reasons.append("punctuation_or_geometry")

    reasons.extend(_ai_hard_visual_guard_reasons(task))
    return list(dict.fromkeys(reasons))


def _build_ai_evidence_cost_plan(batch: Sequence[dict], review_passes: dict) -> dict:
    """Order remaining row review by evidence cost without changing answers."""
    task_by_id = {str(task.get("id", "") or ""): task for task in batch}
    row_ids = [
        str(task_id or "") for task_id in (review_passes.get("row_review_task_ids") or [])
        if str(task_id or "") in task_by_id
    ]
    anchor_text_first: list[str] = []
    text_first: list[str] = []
    visual_required: list[dict] = []
    for task_id in row_ids:
        task = task_by_id[task_id]
        reasons = _ai_visual_review_reasons(task)
        if reasons:
            visual_required.append({
                "id": task_id,
                "image": str(task.get("i", "") or ""),
                "page": int(task.get("p", 0) or 0),
                "reasons": reasons,
                "diff_hints": _ai_candidate_diff_hints(task),
                "focus_hints": _ai_visual_focus_hints(task, reasons),
            })
        elif task.get("x"):
            anchor_text_first.append(task_id)
        else:
            text_first.append(task_id)
    visual_ids = [item["id"] for item in visual_required]
    return {
        "strategy": "lowest_evidence_cost_first",
        "anchor_text_first_task_ids": anchor_text_first,
        "text_first_task_ids": text_first,
        "visual_required": visual_required,
        "visual_required_task_ids": visual_ids,
        "ordered_row_review_task_ids": anchor_text_first + text_first + visual_ids,
        "counts": {
            "anchor_text_first": len(anchor_text_first),
            "text_first": len(text_first),
            "visual_required": len(visual_required),
            "row_review": len(row_ids),
        },
        "contract": "Priority changes review order only. Any text-first row may still open its original image when evidence is insufficient; every visual_required row must inspect source pixels before answering.",
    }


def _ai_visual_contact_sheet_bytes(items: Sequence[dict], folder: Path) -> bytes | None:
    """Pack native-resolution row crops into one labeled lossless WebP without resampling."""
    if not items:
        return None
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None

    opened: list[tuple[dict, object]] = []
    for item in items:
        rel = str(item.get("image", "") or "")
        source = folder / rel
        if not rel or not source.is_file():
            continue
        try:
            image = Image.open(source).convert("L")
        except Exception:
            continue
        opened.append((item, image))
    if not opened:
        return None

    label_h = 26
    gap = 10
    min_cell_width = 220
    max_row_width = 2400
    reason_codes = {
        "number_or_numeric_lookalike": "N",
        "negation": "NEG",
        "punctuation_or_geometry": "P",
        "common_mode": "CM",
        "majority_single_cjk_substitution": "CJK",
        "passive_verb_stem_micro_conflict": "PV",
        "failed_model_with_surviving_conflict": "F",
        "dialogue_small_kana_conflict": "SK",
        "dialogue_repeated_kana_emphasis": "REP",
        "leading_status_page_number_risk": "PN",
        "shared_numeric_measurement_in_disagreement": "NM",
    }

    def cell_width(pair) -> int:
        return max(min_cell_width, int(pair[1].width))

    rows: list[list[tuple[dict, object]]] = []
    current: list[tuple[dict, object]] = []
    current_width = 0
    for pair in opened:
        width = cell_width(pair)
        add_width = width + (gap if current else 0)
        if current and current_width + add_width > max_row_width:
            rows.append(current)
            current = []
            current_width = 0
            add_width = width
        current.append(pair)
        current_width += add_width
    if current:
        rows.append(current)

    row_sizes: list[tuple[int, int]] = []
    for row in rows:
        width = sum(cell_width(pair) for pair in row) + gap * max(0, len(row) - 1)
        height = max(int(pair[1].height) for pair in row) + label_h
        row_sizes.append((width, height))
    canvas_w = max(width for width, _height in row_sizes)
    canvas_h = sum(height for _width, height in row_sizes) + gap * max(0, len(rows) - 1)
    canvas = Image.new("L", (canvas_w, canvas_h), 255)
    draw = ImageDraw.Draw(canvas)
    y = 0
    for row, (_row_w, row_h) in zip(rows, row_sizes):
        x = 0
        for item, image in row:
            width = cell_width((item, image))
            codes = "/".join(reason_codes.get(reason, "?") for reason in (item.get("reasons") or []))
            label = f"{item.get('id', '')} p{item.get('page', '')} [{codes}]"
            draw.text((x + 4, y + 5), label, fill="black")
            image_x = x + max(0, (width - int(image.width)) // 2)
            canvas.paste(image, (image_x, y + label_h))
            x += width + gap
        y += row_h + gap

    output = io.BytesIO()
    canvas.save(output, format="WEBP", lossless=True, quality=100, method=2)
    return output.getvalue()


def _build_ai_batch_review_passes(
    batch: Sequence[dict],
    batch_groups: Sequence[dict],
    batch_anchors: Sequence[dict],
) -> dict:
    """Build a non-overlapping strong-group-first web review plan.

    ``consistency_groups`` can overlap heavily (for example ``ルーティ`` and
    the fragment ``ティ``).  Assign each task to at most one primary strong
    group, preferring the group that covers more rows and the longer term.  The
    full consistency-group list remains in the package for audit; this plan is
    only the efficient execution order for the current web batch.
    """
    task_by_id = {str(task.get("id", "") or ""): task for task in batch}
    anchor_by_id = {str(anchor.get("id", "") or ""): anchor for anchor in batch_anchors}

    eligible: list[tuple[dict, list[str]]] = []
    for group in batch_groups:
        anchor_id = str(group.get("anchor_id", "") or "")
        anchor = anchor_by_id.get(anchor_id) or {}
        stable_count = int(group.get("stable_count", anchor.get("n", 0)) or 0)
        term = str(group.get("term", "") or "")
        # Keep the reusable fast path deliberately narrower than the global
        # consistency index.  Batch 44 promotes only strong Katakana lexical
        # terms (proper names / named entities / stable loanword terms).  Mixed
        # prose fragments such as ``魔王を`` or generic Kanji words remain
        # available in consistency_groups but must stay in ordinary row review.
        if not _AI_KATAKANA_ONLY_TERM_RE.fullmatch(term) or len(term) < 2:
            continue
        local_task_ids = [
            str(task_id) for task_id in (group.get("task_ids") or [])
            if str(task_id) in task_by_id
        ]
        if stable_count < 3 or len(local_task_ids) < 2:
            continue
        eligible.append((group, local_task_ids))

    eligible.sort(key=lambda pair: (
        -len(pair[1]),
        -len(str(pair[0].get("term", "") or "")),
        -int(pair[0].get("stable_count", 0) or 0),
        str(pair[0].get("term", "") or ""),
    ))

    assigned: set[str] = set()
    direct: set[str] = set()
    group_first: list[dict] = []
    for group, local_task_ids in eligible:
        task_ids = [task_id for task_id in local_task_ids if task_id not in assigned]
        if len(task_ids) < 2:
            continue
        term = str(group.get("term", "") or "")
        member_hints = [_ai_group_member_hint(task_by_id[task_id], term) for task_id in task_ids]
        direct_task_ids = [
            str(item.get("id", "") or "")
            for item in member_hints
            if bool(item.get("anchor_only")) and isinstance(item.get("direct_pick"), int)
        ]
        residual_task_ids = [task_id for task_id in task_ids if task_id not in set(direct_task_ids)]
        group_first.append({
            "anchor_id": str(group.get("anchor_id", "") or ""),
            "term": term,
            "stable_count": int(group.get("stable_count", 0) or 0),
            "task_ids": task_ids,
            "direct_task_ids": direct_task_ids,
            "residual_task_ids": residual_task_ids,
            "members": member_hints,
        })
        assigned.update(task_ids)
        direct.update(direct_task_ids)

    row_review_task_ids = [
        str(task.get("id", "") or "") for task in batch
        if str(task.get("id", "") or "") not in direct
    ]
    return {
        "strategy": "strong_group_first_then_row",
        "group_first": group_first,
        "grouped_task_count": len(assigned),
        "direct_after_group_count": len(direct),
        "row_review_task_ids": row_review_task_ids,
        "row_review_count": len(row_review_task_ids),
        "contract": "A group decision never replaces row output: write one independent answer per stable task id. direct_pick is usable only after accepting that strong in-book anchor for the group; residual rows still require normal row evidence review.",
    }

def _ai_visible_quick_tasks(
    tasks: Sequence[dict],
    *,
    context_by_task: dict[str, list[str]] | None = None,
) -> list[dict]:
    visible = []
    context_by_task = context_by_task or {}
    for task in tasks:
        candidates = []
        for candidate in task.get("c", []) or []:
            models = [_ai_short_model_label(item) for item in (candidate.get("m") or []) if str(item or "").strip()]
            entry = {"m": list(dict.fromkeys(models)), "t": str(candidate.get("t", "") or "")}
            if bool(candidate.get("fail", False)):
                entry["f"] = 1
            candidates.append(entry)
        task_id = str(task.get("id", ""))
        entry = {
            "id": task_id,
            "p": int(task.get("page", 0) or 0),
            "cols": list(task.get("cols") or []),
            "s": str(task.get("status", "") or ""),
            "b": str(task.get("before", "") or ""),
            "c": candidates,
            "a": str(task.get("after", "") or ""),
            "i": str(task.get("img", "") or ""),
        }
        if str(task.get("before2", "") or "").strip():
            entry["b2"] = str(task.get("before2", "") or "")
        if task.get("before_candidates"):
            entry["bc"] = list(task.get("before_candidates") or [])
        if task.get("before2_candidates"):
            entry["b2c"] = list(task.get("before2_candidates") or [])
        if str(task.get("after2", "") or "").strip():
            entry["a2"] = str(task.get("after2", "") or "")
        if task.get("after_candidates"):
            entry["ac"] = list(task.get("after_candidates") or [])
        if task.get("after2_candidates"):
            entry["a2c"] = list(task.get("after2_candidates") or [])
        anchors = context_by_task.get(task_id) or []
        if anchors:
            entry["x"] = anchors
        if task.get("risk"):
            entry["r"] = task.get("risk")
        visible.append(entry)
    return visible


def _write_ai_quick_bundle(
    folder: Path,
    *,
    output: Path,
    payload: dict,
    compact_ai_tasks: Sequence[dict],
    direct_output: bool = False,
) -> dict:
    """Create the GPT-facing adjudication exchange ZIP.

    ``direct_output`` writes the compact package directly to ``output`` instead
    of creating a sibling ``*_GPT.zip``.  The GUI uses this mode because the
    project workspace already owns recovery state, so a second full recovery ZIP
    is redundant.
    """
    quick_output = output if direct_output else output.with_name(f"{output.stem}_GPT.zip")
    authority = (folder / "AI_OUTPUT" / "model_corrections.json").read_bytes()
    rows = payload.get("rows") or []
    context_by_task, shared_context = _build_ai_shared_context(rows, compact_ai_tasks)
    consistency_groups = _build_ai_consistency_groups(context_by_task, shared_context)
    visible_tasks = _ai_visible_quick_tasks(
        compact_ai_tasks, context_by_task=context_by_task,
    )
    bindings = []
    for task in compact_ai_tasks:
        # Row zero is a valid authority row.  ``value or -1`` used to turn the
        # first task's row=0 into -1, silently omitting T00001 from the sealed
        # binding table and making ``adjudicate.py finish`` crash after a fully
        # completed review.  Preserve numeric zero exactly.
        try:
            row_index = int(task.get("row", -1))
        except (TypeError, ValueError, OverflowError):
            row_index = -1
        if row_index < 0 or row_index >= len(rows):
            continue
        row = rows[row_index]
        bindings.append({
            "id": str(task.get("id", "")),
            "row": row_index,
            "row_id": str(row.get("row_id", "")),
            "base_row_sha256": str(row.get("base_row_sha256", "")),
        })
    review_batches = []
    batch_size = 40
    shared_context_by_id = {str(item.get("id", "")): item for item in shared_context}
    visible_task_by_id = {str(item.get("id", "")): item for item in visible_tasks}
    book_term_aggregate: dict[tuple[str, str], dict] = {}
    for start in range(0, len(visible_tasks), batch_size):
        batch = visible_tasks[start:start + batch_size]
        if not batch:
            continue
        batch_number = len(review_batches) + 1
        task_ids = {str(task.get("id", "") or "") for task in batch}
        anchor_ids: list[str] = []
        for task in batch:
            for anchor_id in task.get("x") or []:
                anchor_id = str(anchor_id or "")
                if anchor_id and anchor_id not in anchor_ids:
                    anchor_ids.append(anchor_id)
        batch_anchors = [
            shared_context_by_id[anchor_id]
            for anchor_id in anchor_ids
            if anchor_id in shared_context_by_id
        ]
        batch_groups = []
        for group in consistency_groups:
            local_task_ids = [
                str(task_id) for task_id in (group.get("task_ids") or [])
                if str(task_id) in task_ids
            ]
            if not local_task_ids:
                continue
            batch_groups.append({
                "anchor_id": str(group.get("anchor_id", "") or ""),
                "term": str(group.get("term", "") or ""),
                "stable_count": int(group.get("stable_count", 0) or 0),
                "task_ids": local_task_ids,
                "global_task_count": len(group.get("task_ids") or []),
            })
        batch_anchors_compact = [
            {
                "id": str(anchor.get("id", "") or ""),
                "t": str(anchor.get("t", "") or ""),
                "n": int(anchor.get("n", 0) or 0),
                "strong": int(anchor.get("n", 0) or 0) >= 3,
                "e": [
                    str(example.get("text", "") or "")
                    for example in (anchor.get("e") or [])
                    if isinstance(example, dict) and str(example.get("text", "") or "").strip()
                ],
            }
            for anchor in batch_anchors
        ]
        review_passes = _build_ai_batch_review_passes(batch, batch_groups, batch_anchors_compact)
        evidence_cost = _build_ai_evidence_cost_plan(batch, review_passes)
        for group in review_passes.get("group_first") or []:
            anchor_id = str(group.get("anchor_id", "") or "")
            term = str(group.get("term", "") or "")
            key = (anchor_id, term)
            aggregate = book_term_aggregate.setdefault(key, {
                "anchor_id": anchor_id,
                "term": term,
                "stable_count": int(group.get("stable_count", 0) or 0),
                "batches": [],
                "task_ids": [],
                "direct_task_ids": [],
                "residual_task_ids": [],
                "direct_answers": [],
            })
            aggregate["stable_count"] = max(
                int(aggregate.get("stable_count", 0) or 0),
                int(group.get("stable_count", 0) or 0),
            )
            if batch_number not in aggregate["batches"]:
                aggregate["batches"].append(batch_number)
            for field in ("task_ids", "direct_task_ids", "residual_task_ids"):
                for task_id in group.get(field) or []:
                    task_id = str(task_id or "")
                    if task_id and task_id not in aggregate[field]:
                        aggregate[field].append(task_id)
            member_by_id = {
                str(member.get("id", "") or ""): member
                for member in (group.get("members") or [])
                if isinstance(member, dict) and str(member.get("id", "") or "")
            }
            known_direct_ids = {
                str(item.get("id", "") or "")
                for item in aggregate.get("direct_answers") or []
                if isinstance(item, dict)
            }
            for task_id in group.get("direct_task_ids") or []:
                task_id = str(task_id or "")
                member = member_by_id.get(task_id) or {}
                direct_pick = member.get("direct_pick")
                if task_id and isinstance(direct_pick, int) and task_id not in known_direct_ids:
                    aggregate["direct_answers"].append({"id": task_id, "pick": direct_pick})
                    known_direct_ids.add(task_id)
        review_batches.append({
            "batch": batch_number,
            "path": f"BATCHES/B{batch_number:03d}_REVIEW.json",
            "first_id": str(batch[0].get("id", "") or ""),
            "last_id": str(batch[-1].get("id", "") or ""),
            "count": len(batch),
            "with_book_context": sum(bool(item.get("x")) for item in batch),
            "with_images": sum(bool(str(item.get("i", "") or "")) for item in batch),
            "common_mode": sum(bool(item.get("r")) for item in batch),
            "grouped_tasks": int(review_passes.get("grouped_task_count", 0) or 0),
            "direct_after_group": int(review_passes.get("direct_after_group_count", 0) or 0),
            "row_review": int(review_passes.get("row_review_count", len(batch)) or 0),
            "anchor_text_first": int((evidence_cost.get("counts") or {}).get("anchor_text_first", 0) or 0),
            "text_first": int((evidence_cost.get("counts") or {}).get("text_first", 0) or 0),
            "visual_required": int((evidence_cost.get("counts") or {}).get("visual_required", 0) or 0),
            "visual_sheet_count": (int((evidence_cost.get("counts") or {}).get("visual_required", 0) or 0) + 7) // 8,
        })

    ordered_book_terms = sorted(
        book_term_aggregate.values(),
        key=lambda item: (
            -len(item.get("task_ids") or []),
            -len(str(item.get("term", "") or "")),
            str(item.get("term", "") or ""),
            str(item.get("anchor_id", "") or ""),
        ),
    )
    book_term_id_by_key: dict[tuple[str, str], str] = {}
    book_term_entries: list[dict] = []
    for index, aggregate in enumerate(ordered_book_terms, start=1):
        anchor_id = str(aggregate.get("anchor_id", "") or "")
        term = str(aggregate.get("term", "") or "")
        term_id = f"K{index:04d}"
        book_term_id_by_key[(anchor_id, term)] = term_id
        anchor = shared_context_by_id.get(anchor_id) or {}
        representative_tasks = []
        sample_ids = list(aggregate.get("direct_task_ids") or [])[:2]
        for task_id in aggregate.get("residual_task_ids") or []:
            if task_id not in sample_ids:
                sample_ids.append(task_id)
            if len(sample_ids) >= 3:
                break
        for task_id in sample_ids[:3]:
            task = visible_task_by_id.get(str(task_id)) or {}
            representative_tasks.append({
                "id": str(task_id),
                "c": [
                    {
                        "t": str(candidate.get("t", "") or ""),
                        "m": list(candidate.get("m") or []),
                        "f": bool(candidate.get("f", False)),
                    }
                    for candidate in (task.get("c") or [])
                    if isinstance(candidate, dict)
                ],
                "i": str(task.get("i", "") or ""),
            })
        book_term_entries.append({
            "id": term_id,
            "anchor_id": anchor_id,
            "term": term,
            "stable_count": int(aggregate.get("stable_count", 0) or 0),
            "batches": list(aggregate.get("batches") or []),
            "task_count": len(aggregate.get("task_ids") or []),
            "direct_task_count": len(aggregate.get("direct_task_ids") or []),
            "residual_task_count": len(aggregate.get("residual_task_ids") or []),
            "copy_ready_direct_answer_count": len(aggregate.get("direct_answers") or []),
            "direct_answers_if_accepted": list(aggregate.get("direct_answers") or []),
            "fallback_task_ids_if_not_accepted": list(aggregate.get("task_ids") or []),
            "task_ids": list(aggregate.get("task_ids") or []),
            "anchor_examples": [
                str(example.get("text", "") or "")
                for example in (anchor.get("e") or [])
                if isinstance(example, dict) and str(example.get("text", "") or "").strip()
            ],
            "representative_tasks": representative_tasks,
        })

    # Count the Batch-44-style per-batch group decisions directly from the
    # aggregated batch membership: one decision for every term/batch pair.
    batch_group_decision_count = sum(len(item.get("batches") or []) for item in book_term_entries)
    cross_batch_reused_decisions = max(0, batch_group_decision_count - len(book_term_entries))
    copy_ready_direct_answer_count = sum(
        len(item.get("direct_answers_if_accepted") or []) for item in book_term_entries
    )
    evidence_cost_totals = {
        "anchor_text_first": sum(int(item.get("anchor_text_first", 0) or 0) for item in review_batches),
        "text_first": sum(int(item.get("text_first", 0) or 0) for item in review_batches),
        "visual_required": sum(int(item.get("visual_required", 0) or 0) for item in review_batches),
        "visual_sheet_count": sum(int(item.get("visual_sheet_count", 0) or 0) for item in review_batches),
    }

    manifest = {
        "schema": "novel_formatter.ai_quick_adjudication.v4",
        "package_id": str(payload.get("package_id", "")),
        "task_count": len(compact_ai_tasks),
        "conflict_task_count": int(payload.get("pending_conflict_rows", 0) or 0),
        "common_mode_task_count": int(payload.get("pending_common_mode_rows", 0) or 0),
        "answer_path": "AI_OUTPUT/answers.jsonl",
        "status_command": "python adjudicate.py status",
        "validate_command": "python adjudicate.py validate",
        "finish_command": "python adjudicate.py finish",
        "full_package": "" if direct_output else output.name,
        "authority_hidden": True,
        "physical_column_evidence_in_full_package": False if direct_output else True,
        "start_here": "00_GPT_START_HERE.md",
        "task_path": "04_ai_tasks_compact.jsonl",
        "consistency_groups_path": "02_consistency_groups.json",
        "book_context_path": "03_BOOK_CONTEXT.json",
        "batch_dir": "BATCHES",
        "tasks_with_book_context": len(context_by_task),
        "book_context_anchor_count": len(shared_context),
        "consistency_group_count": len(consistency_groups),
        "book_context_policy": "locked_exact_consensus_only_shared_index",
        "review_strategy": "book_term_copy_ready_then_b49_hard_error_guard_visual_sheet_dedup",
        "hard_error_guard_version": "b49.v1",
        "hard_error_guard_contract": "candidate-only escalation; no reference truth or corrected text embedded",
        "book_term_review_path": "BOOK_TERM_PASS.json",
        "book_term_count": len(book_term_entries),
        "copy_ready_direct_answer_count": copy_ready_direct_answer_count,
        "batch_group_decision_count": batch_group_decision_count,
        "cross_batch_reused_term_decisions": cross_batch_reused_decisions,
        "evidence_cost_totals": evidence_cost_totals,
        "visual_sheet_max_items": 8,
        "visual_storage_policy": "lossless_sheet_is_canonical_for_visual_required; duplicate_single_crop_omitted; text_first_single_crop_retained",
        "book": copy.deepcopy(payload.get("book") or {}),
    }
    instructions = """# GPT OCR 裁决包 — 完整操作命令

## 目标

只裁决本包任务，把扫描图能够支持的**原作品日文**还原出来。网页端先读一次 `BOOK_TERM_PASS.json`，把整本重复 strong 片假名专名/术语先做一次书级词形判断；随后按 `BATCHES/Bxxx_REVIEW.json` 分批处理。每个 REVIEW 文件仍内含该批任务、所需书内锚点与一致性组。`04_ai_tasks_compact.jsonl` 仅作为完整任务总表与兼容入口。
这不是润色任务：不得为了语法更顺、现代写法或个人偏好改写原文。

## 文件含义

- `04_ai_tasks_compact.jsonl`：唯一待处理任务集。`p` 是页码，`cols` 是稳定物理列 ID，`s` 是送审状态。
- `b` / `a`：紧邻当前句且已经稳定的前文 / 后文；`b2` / `a2` 是再向外一条。若邻接行本身也有分歧，不会偷偷选第一模型，而会放进 `bc` / `ac`（以及 `b2c` / `a2c`）作为“邻接未决候选”。
- `c`：真正独立执行过的 OCR 候选；相同文字已合并。`m` 是支持该候选的模型。
- `f=1`：该 OCR 失败/占位，**不是原文字符**，不能选。
- `i`：对应扫描句图。文字证据不足时必须看图。
- `r`：Common-Mode 风险原因。它只解释为什么送审，绝不是答案。
- `x`：共享书内上下文 anchor ID 列表。具体词形、稳定出现次数和例句只在 `03_BOOK_CONTEXT.json` 保存一次，避免每条任务重复携带同样证据。
- `02_consistency_groups.json`：多个待审任务共享同一稳定专名/术语时的分组索引。先看这些组，可以保持跨任务一致。
- `03_BOOK_CONTEXT.json`：仅来自锁定 exact-consensus OCR 行的共享书内证据；不是外部真值。
- `BOOK_TERM_PASS.json`：整本一次性的 strong 片假名专名/术语预审；同一 term 只判断一次，并列出稳定例句、代表性候选、涉及批次，以及**仅在接受该 term 后才可复制**的逐 row `direct_answers_if_accepted`。
- `BATCHES/Bxxx_REVIEW.json`：网页端首选的**单文件批次入口**；每批默认 40 条，内含 `tasks + anchors + consistency_groups + evidence_cost`，其中 group 会引用 `book_term_id`。
- `VISUAL_REVIEW/Bxxx_Sxx.webp`：把本批明确需要像素证据的原始 row 裁片按 stable ID 拼成联系表，**不缩放、不重采样**。对已成功写入联系表的 `visual_required` 行，任务 `i` 直接指向该无损 sheet，重复的单张裁片不再写入 ZIP；文字优先行仍保留原 `i` 单图用于必要时 fallback。
- 全局 `02_consistency_groups.json` / `03_BOOK_CONTEXT.json`：只用于跨批最终一致性审计，普通逐批裁决不需要加载。
- `_BINDINGS/`：稳定 ID/hash 密封数据，只供脚本使用；不要读取、修改或重写。
- `AI_OUTPUT/answers.jsonl`：唯一需要写入的答案文件。

## 必须执行的工作流

1. 先检查任务数：

   ```bash
   python adjudicate.py status
   ```

2. 先只做一次 `BOOK_TERM_PASS.json`：每个 `Kxxxx` strong 片假名专名/术语只判断一次。若稳定 anchor 与代表性候选不冲突，就把该词形视为本轮书级锁；此时可直接复制该 term 的 `direct_answers_if_accepted` 到最终逐 row 答案（它们已经是独立 stable-ID `{id,pick}` 记录，不是组级答案）。若证据不足或冲突，**绝不能复制这组 direct answers**，而必须按 `fallback_task_ids_if_not_accepted` 把相关 row 全部送回普通逐行/看图复核。

3. 再按 `BATCHES/Bxxx_REVIEW.json` 分批判断；其中 `review_passes.group_first` 的每个组通过 `book_term_id` 引用上一步书级决定。已接受的 term 不再跨批重复判断；`anchor_only=true` 且给出 `direct_pick` 的 row 可直接按各自 stable ID 写答案。`residual_task_ids` 仍逐 row 复核，因为这些行还有标点、其他字符或其他未决差异。

   每批随后按 `review_passes.evidence_cost` 的顺序处理：先 `anchor_text_first_task_ids`，再 `text_first_task_ids`，最后 `visual_required_task_ids`。前两类只是“先尝试文字证据”，证据不足仍可打开原图；`visual_required` 必须看像素。若提供 `visual_sheets`，优先打开联系表一次查看多条原裁片，stable ID 已写在每个裁片上；成功 sheet 化的 `visual_required` 行其 `i` 本身就指向该无损 sheet，不再重复携带单张裁片。文字优先行若证据不足仍可打开其原 `i` 单图。只有跨批一致性审计时才需要全局 `02_consistency_groups.json` / `03_BOOK_CONTEXT.json`。逐 row 时优先看 `b2/b2c + b/bc + c + a/ac + a2/a2c`；任务出现 `x` 时查本批 context 的对应 anchor；以下情况必须打开像素证据：
   - 人名、地名、技能名若**没有** `strong=true` 的同书稳定 anchor，或 anchor 与当前候选冲突；
   - 数字、等级、否定词；
   - `目/日`、`ニ/二`、`カ/力` 等形近字，除非对应专名已有 `strong=true` 的同书稳定 anchor；
   - 小假名、促音、长音、引号、粘句/漏句；
   - Common-Mode (`r` 存在)；
   - 候选都不自然或无法仅靠上下文确定。

4. **不要按票数裁决。** 2:1、3:1 只表示模型数量，不代表图像真值。
   seeded/copied 证据不会出现在 `c`，不要自行把缺失模型补成一票。

5. 写 `AI_OUTPUT/answers.jsonl`，一行一个 JSON，ID 必须与任务一致：

   - 原候选正确：
     `{"id":"T00001","pick":0,"confidence":0.99,"reason":"image+context"}`
   - 所有候选都错，图像能确定完整正文：
     `{"id":"T00002","text":"完整正确正文","confidence":0.98,"reason":"corrected_from_image"}`
   - 确认是独立页码/页眉等非正文、应从正文有意剔除（必须核对扫描图）：
     `{"id":"T00003","delete_intentionally":true,"confidence":0.99,"reason":"scan_confirmed_running_page_number"}`
   - 图像仍不足以确定：
     `{"id":"T00004","unresolved":true}`

   `text` 必须是**完整当前句/当前行正文**，不能只写差异字符。

6. 中途或完成后验证答案格式：

   ```bash
   python adjudicate.py validate
   python adjudicate.py status
   ```

   `validate` 必须通过；最终 `remaining` 应为 0。

7. 生成 Novel Formatter 可直接导回的文件：

   ```bash
   python adjudicate.py finish
   ```

   输出：`AI_IMPORT.zip`。脚本会再次检查任务 ID、候选索引、密封 row/hash 绑定和完整性。

8. 可选再做 ZIP CRC 检查：

   ```bash
   python -m zipfile -t AI_IMPORT.zip
   ```

## 裁决原则

- **包内扫描句图 + OCR 候选 + 前后文 + `x` 稳定工作区上下文是唯一裁决依据。不要联网寻找、不要读取或依赖电子版/参考稿来生成答案。** `x` 只是当前 OCR 工作区的内部一致性证据，不是出版社真值。参考版若由用户另行提供，只能在全部裁决完成并生成 `AI_IMPORT.zip` 之后做事后质量评估，不能反向修改本轮答案。
- 保留作者语气、异体/口语写法；纯全半角、装饰符号等不影响含义的差异不要为了机械一致而过度改。
- 对会影响翻译的项目优先严格核对：专名、数字、否定、助词导致的主客体变化、漏字、粘句、句界。
- Common-Mode 任务即使只有一个候选也必须看图；本地三个模型可能共同识别错。
- 证据不足就 `unresolved=true`，禁止猜。

最终只交回 `AI_IMPORT.zip`；不要修改原 OCR JSON、图片、manifest 或 `_BINDINGS`。
"""
    with atomic_output_path(quick_output) as quick_tmp:
        with zipfile.ZipFile(quick_tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            archive.writestr("manifest.json", _json_bytes(manifest, pretty=True))
            start_here = (
                "# Novel Formatter · GPT 网页裁决入口\n\n"
                "请先读本文件，再按 `BATCHES/Bxxx_REVIEW.json` 分批处理。\n\n"
                "推荐顺序：\n"
                "1. 先读 `01_REVIEW_PLAN.json`，确认当前批次。\n"
                "2. 先只打开一次 `BOOK_TERM_PASS.json`，把 strong 的片假名专名/术语做书级词形锁定；接受后可直接复制该 term 的逐 row `direct_answers_if_accepted`，证据不足则按 fallback 列表回普通复核。\n"
                "3. 再一次只打开当前 `BATCHES/Bxxx_REVIEW.json`；其中 `review_passes.group_first` 通过 `book_term_id` 复用已接受的书级 term，再按 `review_passes.evidence_cost` 从低成本文字证据到强制视觉证据处理。每条仍必须按 stable task ID 独立写答案。\n"
                "   全局 `02_consistency_groups.json` / `03_BOOK_CONTEXT.json` 只用于跨批最终一致性审计，普通逐批裁决不需要打开。\n"
                "4. 只写 `AI_OUTPUT/answers.jsonl`，不要修改 OCR 原始候选或 `_BINDINGS/`。\n"
                "5. 完成后运行 `python adjudicate.py validate` 与 `python adjudicate.py finish`，交回 `AI_IMPORT.zip`。\n\n"
                "详细规则见 `AGENTS.md`。\n"
            )
            archive.writestr("00_GPT_START_HERE.md", start_here.encode("utf-8"))
            archive.writestr("AGENTS.md", instructions.encode("utf-8"))
            review_plan = {
                "schema": "novel_formatter.ai_web_review_plan.v8",
                "book": copy.deepcopy(payload.get("book") or {}),
                "task_count": len(visible_tasks),
                "tasks_with_book_context": [task["id"] for task in visible_tasks if task.get("x")],
                "tasks_with_images": [task["id"] for task in visible_tasks if str(task.get("i", "") or "")],
                "common_mode_tasks": [task["id"] for task in visible_tasks if task.get("r")],
                "book_context_path": "03_BOOK_CONTEXT.json",
                "consistency_groups_path": "02_consistency_groups.json",
                "book_term_review_path": "BOOK_TERM_PASS.json",
                "book_term_count": len(book_term_entries),
                "copy_ready_direct_answer_count": copy_ready_direct_answer_count,
                "batch_group_decision_count": batch_group_decision_count,
                "cross_batch_reused_term_decisions": cross_batch_reused_decisions,
                "evidence_cost_totals": evidence_cost_totals,
                "visual_sheet_max_items": 8,
                "batches": review_batches,
                "recommended_order": [
                    "book_term_pass_copy_direct_answers",
                    "anchor_text_first",
                    "text_first",
                    "visual_required_contact_sheets",
                    "global_consistency_audit",
                ],
                "unresolved_policy": "evidence_insufficient_only",
            }
            archive.writestr("01_REVIEW_PLAN.json", _json_bytes(review_plan, pretty=True))
            archive.writestr("BOOK_TERM_PASS.json", _json_bytes({
                "schema": "novel_formatter.ai_book_term_pass.v2",
                "source": "strong Katakana groups already admitted by the Batch 44 safe group-first gate",
                "term_count": len(book_term_entries),
                "copy_ready_direct_answer_count": copy_ready_direct_answer_count,
                "batch_group_decision_count": batch_group_decision_count,
                "cross_batch_reused_term_decisions": cross_batch_reused_decisions,
                "rules": [
                    "Judge each Kxxxx term once for the whole book from locked exact-consensus anchor evidence and representative OCR candidates.",
                    "Accept a term only when the strong in-book anchor is applicable and non-conflicting; otherwise leave that term unlocked and review every related row normally.",
                    "Only after accepting a Kxxxx term may direct_answers_if_accepted be copied; those entries are already independent stable-ID row answers, never one group-level answer.",
                    "If a Kxxxx term is not accepted, do not copy its direct answers; review fallback_task_ids_if_not_accepted normally.",
                    "Numbers, negation, common-mode risk and residual punctuation/geometry remain row/image review even when a term is accepted.",
                ],
                "terms": book_term_entries,
            }, pretty=True))
            archive.writestr("02_consistency_groups.json", _json_bytes({
                "schema": "novel_formatter.ai_consistency_groups.v2",
                "source": "locked exact-consensus OCR rows in this workspace only",
                "groups": consistency_groups,
            }, pretty=True))
            archive.writestr("03_BOOK_CONTEXT.json", _json_bytes({
                "schema": "novel_formatter.ai_shared_book_context.v1",
                "source": "locked exact-consensus OCR rows in this workspace only",
                "anchors": shared_context,
            }, pretty=True))
            archive.writestr("adjudicate.py", _AI_QUICK_CLI.encode("utf-8"))
            # Batch 48: successful lossless visual sheets become the canonical
            # image entry for visual-required rows.  Their duplicate single-row
            # crops are omitted from the web package, while text-first rows keep
            # the original crop for fallback.  Book-term representative images
            # are retained when referenced directly by BOOK_TERM_PASS.json.
            protected_image_paths = {
                str(task.get("i", "") or "")
                for term in book_term_entries
                for task in (term.get("representative_tasks") or [])
                if str(task.get("i", "") or "")
            }
            deduplicated_image_paths: set[str] = set()
            visual_sheet_ref_by_task: dict[str, dict] = {}
            for batch_index, start in enumerate(range(0, len(visible_tasks), batch_size), start=1):
                batch = visible_tasks[start:start + batch_size]
                anchor_ids: list[str] = []
                task_ids = {str(task.get("id", "") or "") for task in batch}
                for task in batch:
                    for anchor_id in task.get("x") or []:
                        anchor_id = str(anchor_id or "")
                        if anchor_id and anchor_id not in anchor_ids:
                            anchor_ids.append(anchor_id)
                batch_anchors = [shared_context_by_id[anchor_id] for anchor_id in anchor_ids if anchor_id in shared_context_by_id]
                batch_groups = []
                for group in consistency_groups:
                    local_task_ids = [
                        str(task_id) for task_id in (group.get("task_ids") or [])
                        if str(task_id) in task_ids
                    ]
                    if not local_task_ids:
                        continue
                    batch_groups.append({
                        "anchor_id": str(group.get("anchor_id", "") or ""),
                        "term": str(group.get("term", "") or ""),
                        "stable_count": int(group.get("stable_count", 0) or 0),
                        "task_ids": local_task_ids,
                        "global_task_count": len(group.get("task_ids") or []),
                    })
                batch_anchors_compact = [
                    {
                        "id": str(anchor.get("id", "") or ""),
                        "t": str(anchor.get("t", "") or ""),
                        "n": int(anchor.get("n", 0) or 0),
                        "strong": int(anchor.get("n", 0) or 0) >= 3,
                        "e": [
                            str(example.get("text", "") or "")
                            for example in (anchor.get("e") or [])
                            if isinstance(example, dict) and str(example.get("text", "") or "").strip()
                        ],
                    }
                    for anchor in batch_anchors
                ]
                review_passes = _build_ai_batch_review_passes(batch, batch_groups, batch_anchors_compact)
                for group in review_passes.get("group_first") or []:
                    book_term_id = book_term_id_by_key.get((
                        str(group.get("anchor_id", "") or ""),
                        str(group.get("term", "") or ""),
                    ))
                    if book_term_id:
                        group["book_term_id"] = book_term_id
                        group["decision_source"] = "BOOK_TERM_PASS.json"
                        group["fallback_task_ids_if_not_accepted"] = list(group.get("task_ids") or [])
                review_passes["strategy"] = "book_term_copy_ready_then_b49_hard_error_guard_visual_sheet_dedup"
                review_passes["book_term_review_path"] = "BOOK_TERM_PASS.json"
                review_passes["row_review_task_ids_if_all_terms_rejected"] = [
                    str(task.get("id", "") or "") for task in batch if str(task.get("id", "") or "")
                ]
                evidence_cost = _build_ai_evidence_cost_plan(batch, review_passes)
                visual_items = list(evidence_cost.get("visual_required") or [])
                visual_sheets = []
                for sheet_number, offset in enumerate(range(0, len(visual_items), 8), start=1):
                    sheet_items = visual_items[offset:offset + 8]
                    sheet_path = f"VISUAL_REVIEW/B{batch_index:03d}_S{sheet_number:02d}.webp"
                    sheet_bytes = _ai_visual_contact_sheet_bytes(sheet_items, folder)
                    if sheet_bytes:
                        archive.writestr(sheet_path, sheet_bytes, compress_type=zipfile.ZIP_STORED)
                        visual_sheets.append({
                            "path": sheet_path,
                            "task_ids": [str(item.get("id", "") or "") for item in sheet_items],
                            "count": len(sheet_items),
                            "native_resolution": True,
                            "lossless": True,
                            "resampled": False,
                            "single_crop_deduplicated": True,
                        })
                        batch_task_by_id = {str(task.get("id", "") or ""): task for task in batch}
                        for tile_index, item in enumerate(sheet_items):
                            task_id = str(item.get("id", "") or "")
                            original_image = str(item.get("image", "") or "")
                            task = batch_task_by_id.get(task_id)
                            if task is not None:
                                task["i"] = sheet_path
                                task["vi"] = {
                                    "sheet": sheet_path,
                                    "tile_index": tile_index,
                                    "source_image": original_image,
                                    "native_resolution": True,
                                    "lossless": True,
                                }
                            item["source_image"] = original_image
                            item["image"] = sheet_path
                            item["sheet_index"] = tile_index
                            visual_sheet_ref_by_task[task_id] = {
                                "sheet": sheet_path,
                                "tile_index": tile_index,
                                "source_image": original_image,
                            }
                            if original_image and original_image not in protected_image_paths:
                                deduplicated_image_paths.add(original_image)
                evidence_cost["visual_sheets"] = visual_sheets
                review_passes["evidence_cost"] = evidence_cost
                archive.writestr(f"BATCHES/B{batch_index:03d}_REVIEW.json", _json_bytes({
                    "schema": "novel_formatter.ai_web_batch_review.v9",
                    "batch": batch_index,
                    "task_count": len(batch),
                    "answer_path": "AI_OUTPUT/answers.jsonl",
                    "source": "OCR candidates + local neighbourhood + locked exact-consensus in-workspace anchors only",
                    "rules": [
                        "Do not vote by model count; judge original Japanese from evidence.",
                        "Use anchors for in-book name/term consistency; anchors are OCR consensus evidence, not external truth.",
                        "A strong anchor (strong=true, >=3 locked exact-consensus occurrences) may settle a repeated in-book name/term without reopening the image; otherwise open i for names/lookalikes, and always for numbers, negation, common-mode risk, conflicting anchors, or insufficient text evidence.",
                        "Run BOOK_TERM_PASS.json once before batches. If a Kxxxx term is accepted, its copy-ready direct answers may be copied as independent stable-ID row answers. If rejected/uncertain, do not use direct_pick and review fallback_task_ids_if_not_accepted normally. residual_task_ids always remain row review.",
                        "Even when a group decision is reused, output one independent answer for every stable task id; never emit one group-level answer in place of row answers.",
                        "Use review_passes.evidence_cost in order: anchor_text_first, text_first, then visual_required. Text-first is only an ordering hint and retains its original i crop for fallback; visual_required must inspect source pixels and, when a lossless visual sheet exists, its i points directly to that sheet so the duplicate single crop is omitted from the ZIP.",
                        "For visual_required rows, inspect focus_hints and diff_hints first. They only direct attention and never identify the correct answer. B49 explicitly escalates single-CJK majority conflicts, passive-verb micro-conflicts, failed-model residual disagreements, dialogue small-kana/emphasis tails, shared numeric measurements and status-card leading-number risk to pixels.",
                        "For leading_status_page_number_risk, treat an isolated margin/page number as layout metadata unless the scan clearly places it inside the status text.",
                        "If still uncertain, write unresolved=true rather than guessing.",
                    ],
                    "review_passes": review_passes,
                    "tasks": batch,
                    "anchors": batch_anchors_compact,
                    "consistency_groups": batch_groups,
                }))
            task_bytes = b"".join(_json_bytes(task) + b"\n" for task in visible_tasks)
            archive.writestr("04_ai_tasks_compact.jsonl", task_bytes)
            archive.writestr("AI_OUTPUT/answers.jsonl", b"")
            archive.writestr("_BINDINGS/authority.json.gz", gzip.compress(authority, compresslevel=1, mtime=0), compress_type=zipfile.ZIP_STORED)
            archive.writestr("_BINDINGS/tasks.json.gz", gzip.compress(_json_bytes(bindings), compresslevel=1, mtime=0), compress_type=zipfile.ZIP_STORED)
            for task in compact_ai_tasks:
                rel = str(task.get("img", "") or "")
                if not rel:
                    continue
                if rel in deduplicated_image_paths:
                    continue
                source = folder / rel
                if source.is_file():
                    archive.write(source, rel, compress_type=zipfile.ZIP_STORED)
        with zipfile.ZipFile(quick_tmp, "r") as check:
            bad = check.testzip()
            if bad:
                raise SourceCorrectionError(f"GPT 裁决交换包 ZIP CRC 失败：{bad}")
    return {"path": str(quick_output), "bytes": quick_output.stat().st_size, "tasks": len(compact_ai_tasks)}


def _write_jsonl(path: Path, values: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def _export_conflict_images(
    folder: Path,
    primary: UnifiedDocument | Sequence[UnifiedDocument],
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
    # Evidence geometry is a session-level contract, not a primary-model-only
    # contract.  Page/column roles can carry a column in a secondary OCR
    # document even when the canonical primary document lacks that exact
    # review-region record (especially cross-page sentence groups and restored
    # role-routed sessions). Merge geometry/page paths from every model while
    # keeping the first model authoritative on conflicts.
    if isinstance(primary, UnifiedDocument):
        evidence_documents = [primary]
    else:
        evidence_documents = [doc for doc in primary if isinstance(doc, UnifiedDocument)]
    if not evidence_documents:
        return 0
    geometry: dict[str, dict] = {}
    pages: dict[int, str] = {}
    for document in evidence_documents:
        for column_id, region in _geometry_snapshot(document).items():
            geometry.setdefault(column_id, region)
        for page_no, image_path in _page_paths(document).items():
            if image_path:
                pages.setdefault(page_no, image_path)
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
    # GUI progress callbacks may cross threads/signals.  Reporting once per page
    # for a full book (hundreds of pages) can dominate the actual image work,
    # even though evidence generation itself only takes seconds.  Cap this stage
    # to roughly 80 updates while always reporting the first and last page.
    evidence_progress_stride = max(1, total_pages // 80)
    max_height = max(800, int(max_height or 1600))
    for page_index, (page, page_rows) in enumerate(sorted(grouped.items()), start=1):
        if page_index == 1 or page_index == total_pages or page_index % evidence_progress_stride == 0:
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
                        crop.save(target, format="PNG", optimize=False, compress_level=1)
                        payload = target.read_bytes()
                        row["evidence_image"] = target.relative_to(folder).as_posix()
                        row["evidence_image_meta"] = {
                            "profile": "canonical_role_evidence_v5",
                            "source_crop_size": list(original_size),
                            "exported_size": [int(crop.width), int(crop.height)],
                            "scale": round(float(scale), 8),
                            "max_height": max_height,
                            "mode": "L",
                            "format": "PNG",
                            "png_encoding_lossless": True,
                            "png_compress_level": 1,
                            "resampled": bool(scale < 1.0),
                            "review_copy_only": True,
                            "file_sha256": hashlib.sha256(payload).hexdigest(),
                            "file_bytes": len(payload),
                            "ocr_input_unchanged": True,
                        }
                        written += 1

                        # Keep the combined row crop, but also export each
                        # physical OCR column independently.  Long sentences can
                        # span several columns; local/external OCR should inspect
                        # these narrow evidence units instead of re-reading one
                        # 280x1600 multi-column strip.
                        column_evidence = []
                        safe_row_id = str(row.get("row_id", "row") or "row").replace(":", "_")
                        for column_index, column_id in enumerate(row.get("column_ids", []) or []):
                            region = geometry.get(str(column_id))
                            if not isinstance(region, dict) or int(region.get("page", page) or page) != page:
                                continue
                            x, y, w, h = [float(value or 0.0) for value in region.get("bbox", [0, 0, 0, 0])]
                            c_left = max(0, int(x * image.width - image.width * .006))
                            c_top = max(0, int(y * image.height - image.height * .006))
                            c_right = min(image.width, int((x + w) * image.width + image.width * .006))
                            c_bottom = min(image.height, int((y + h) * image.height + image.height * .006))
                            if c_right <= c_left or c_bottom <= c_top:
                                continue
                            col_crop = image.crop((c_left, c_top, c_right, c_bottom))
                            col_source_size = tuple(int(value) for value in col_crop.size)
                            col_scale = 1.0
                            try:
                                if col_crop.height > max_height:
                                    col_scale = max_height / float(col_crop.height)
                                    resized_width = max(1, int(round(col_crop.width * col_scale)))
                                    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
                                    resized = col_crop.resize((resized_width, max_height), resampling)
                                    col_crop.close()
                                    col_crop = resized
                                col_target = (
                                    folder / "images" / f"page_{page:04d}" / "columns" / safe_row_id
                                    / f"column_{column_index + 1:02d}.png"
                                )
                                col_target.parent.mkdir(parents=True, exist_ok=True)
                                col_crop.save(col_target, format="PNG", optimize=False, compress_level=1)
                                col_payload = col_target.read_bytes()
                                column_evidence.append({
                                    "column_id": str(column_id),
                                    "path": col_target.relative_to(folder).as_posix(),
                                    "normalized_bbox": [x, y, w, h],
                                    "source_crop_size": list(col_source_size),
                                    "exported_size": [int(col_crop.width), int(col_crop.height)],
                                    "scale": round(float(col_scale), 8),
                                    "mode": "L",
                                    "format": "PNG",
                                    "png_encoding_lossless": True,
                                    "png_compress_level": 1,
                                    "review_copy_only": True,
                                    "file_sha256": hashlib.sha256(col_payload).hexdigest(),
                                    "file_bytes": len(col_payload),
                                    "ocr_input_unchanged": True,
                                })
                            except Exception as exc:
                                row.setdefault("evidence_export_errors", []).append({
                                    "stage": "physical_column_evidence",
                                    "page": int(page),
                                    "column_id": str(column_id),
                                    "error_type": type(exc).__name__,
                                    "message": str(exc)[:240],
                                })
                            finally:
                                col_crop.close()
                        if column_evidence:
                            row["physical_column_evidence"] = column_evidence
                    finally:
                        crop.close()
        except Exception as exc:
            # Evidence export is non-authoritative and must never abort OCR
            # package creation, but silent loss makes missing crops impossible
            # to diagnose.  Record the affected rows and failure stage instead.
            for row, _regions in page_rows:
                row.setdefault("evidence_export_errors", []).append({
                    "stage": "page_evidence_export",
                    "page": int(page),
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:240],
                })
            continue

    # A sentence may span two source pages.  The combined row crop above is
    # intentionally anchored to the first page, but physical-column evidence
    # must be complete across *all* pages represented by the row.  Earlier V5
    # exports silently omitted the trailing-page columns because they were not in
    # the currently opened image.  Fill only the missing column evidence here.
    missing_by_page: dict[int, list[tuple[dict, int, str, dict]]] = {}
    for row in rows:
        if not row.get("editable"):
            continue
        existing = {
            str(item.get("column_id", "") or "")
            for item in (row.get("physical_column_evidence") or [])
            if isinstance(item, dict)
        }
        for column_index, column_id in enumerate(row.get("column_ids", []) or []):
            column_id = str(column_id)
            if not column_id or column_id in existing:
                continue
            region = geometry.get(column_id)
            if not isinstance(region, dict):
                continue
            column_page = int(region.get("page", row.get("page", 0)) or 0)
            if column_page <= 0:
                continue
            missing_by_page.setdefault(column_page, []).append(
                (row, column_index, column_id, region)
            )

    for column_page, specs in sorted(missing_by_page.items()):
        source_path = Path(pages.get(column_page, "")).expanduser()
        if not source_path.is_file():
            for row, _index, column_id, _region in specs:
                row.setdefault("evidence_export_errors", []).append({
                    "stage": "cross_page_physical_column_evidence",
                    "page": int(column_page),
                    "column_id": column_id,
                    "error_type": "FileNotFoundError",
                    "message": str(source_path),
                })
            continue
        try:
            with Image.open(source_path) as opened:
                image = opened.convert("L")
                for row, column_index, column_id, region in specs:
                    try:
                        x, y, w, h = [float(value or 0.0) for value in region.get("bbox", [0, 0, 0, 0])]
                        c_left = max(0, int(x * image.width - image.width * .006))
                        c_top = max(0, int(y * image.height - image.height * .006))
                        c_right = min(image.width, int((x + w) * image.width + image.width * .006))
                        c_bottom = min(image.height, int((y + h) * image.height + image.height * .006))
                        if c_right <= c_left or c_bottom <= c_top:
                            raise ValueError("empty physical-column evidence crop")
                        col_crop = image.crop((c_left, c_top, c_right, c_bottom))
                        col_source_size = tuple(int(value) for value in col_crop.size)
                        col_scale = 1.0
                        try:
                            if col_crop.height > max_height:
                                col_scale = max_height / float(col_crop.height)
                                resized_width = max(1, int(round(col_crop.width * col_scale)))
                                resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
                                resized = col_crop.resize((resized_width, max_height), resampling)
                                col_crop.close()
                                col_crop = resized
                            safe_row_id = str(row.get("row_id", "row") or "row").replace(":", "_")
                            col_target = (
                                folder / "images" / f"page_{column_page:04d}" / "columns" / safe_row_id
                                / f"column_{column_index + 1:02d}.png"
                            )
                            col_target.parent.mkdir(parents=True, exist_ok=True)
                            col_crop.save(col_target, format="PNG", optimize=False, compress_level=1)
                            col_payload = col_target.read_bytes()
                            row.setdefault("physical_column_evidence", []).append({
                                "column_id": column_id,
                                "path": col_target.relative_to(folder).as_posix(),
                                "normalized_bbox": [x, y, w, h],
                                "source_crop_size": list(col_source_size),
                                "exported_size": [int(col_crop.width), int(col_crop.height)],
                                "scale": round(float(col_scale), 8),
                                "mode": "L",
                                "format": "PNG",
                                "png_encoding_lossless": True,
                                "png_compress_level": 1,
                                "review_copy_only": True,
                                "file_sha256": hashlib.sha256(col_payload).hexdigest(),
                                "file_bytes": len(col_payload),
                                "ocr_input_unchanged": True,
                                "cross_page_completion": True,
                            })
                        finally:
                            col_crop.close()
                    except Exception as exc:
                        row.setdefault("evidence_export_errors", []).append({
                            "stage": "cross_page_physical_column_evidence",
                            "page": int(column_page),
                            "column_id": column_id,
                            "error_type": type(exc).__name__,
                            "message": str(exc)[:240],
                        })
        except Exception as exc:
            for row, _index, column_id, _region in specs:
                row.setdefault("evidence_export_errors", []).append({
                    "stage": "cross_page_physical_column_evidence",
                    "page": int(column_page),
                    "column_id": column_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:240],
                })

    # Cross-page sentence groups cannot be represented by a bounding-box union
    # on one source page.  The old exporter still wrote a syntactically valid
    # PNG, but it contained only the first page's strip and therefore looked
    # like a failed/empty sentence image to GPT and 图文对照.  Recompose those
    # rows from their already-exported immutable physical-column evidence in
    # exact reading order.  No OCR is rerun and source pixels remain unchanged.
    for row in rows:
        if not row.get("editable"):
            continue
        expected_ids = [str(value) for value in (row.get("column_ids") or []) if str(value)]
        if len(expected_ids) < 2:
            continue
        page_ids = []
        for column_id in expected_ids:
            region = geometry.get(column_id)
            page_id = int(region.get("page", 0) or 0) if isinstance(region, dict) else 0
            if page_id > 0 and page_id not in page_ids:
                page_ids.append(page_id)
        if len(page_ids) <= 1:
            continue

        evidence_by_id = {
            str(item.get("column_id", "") or ""): item
            for item in (row.get("physical_column_evidence") or [])
            if isinstance(item, dict) and str(item.get("column_id", "") or "")
        }
        missing = [column_id for column_id in expected_ids if column_id not in evidence_by_id]
        if missing:
            row.setdefault("evidence_export_errors", []).append({
                "stage": "cross_page_sentence_composition",
                "error_type": "MissingPhysicalColumnEvidence",
                "message": "missing columns: " + ", ".join(missing[:20]),
            })
            continue

        strips = []
        try:
            for column_id in expected_ids:
                rel = str(evidence_by_id[column_id].get("path", "") or "")
                source = folder / rel
                if not source.is_file():
                    raise FileNotFoundError(str(source))
                with Image.open(source) as opened:
                    strips.append(opened.convert("L"))
            if len(strips) != len(expected_ids):
                raise RuntimeError("physical-column evidence count mismatch")

            widths = [strip.width for strip in strips]
            typical = max(1, sorted(widths)[len(widths) // 2])
            gap = max(5, round(typical * 0.35))
            page_gap = max(gap * 2, round(typical * 1.15))
            margin_x = max(8, round(typical * 0.40))
            margin_y = max(6, round(typical * 0.25))
            inter_gaps = []
            for column_id in expected_ids[:-1]:
                region = geometry.get(column_id)
                current_page = int(region.get("page", 0) or 0) if isinstance(region, dict) else 0
                next_id = expected_ids[len(inter_gaps) + 1]
                next_region = geometry.get(next_id)
                next_page = int(next_region.get("page", 0) or 0) if isinstance(next_region, dict) else 0
                inter_gaps.append(page_gap if current_page and next_page and current_page != next_page else gap)
            canvas_width = sum(widths) + sum(inter_gaps) + margin_x * 2
            canvas_height = max(strip.height for strip in strips) + margin_y * 2
            canvas = Image.new("L", (canvas_width, canvas_height), 255)
            try:
                cursor = canvas_width - margin_x
                for index, strip in enumerate(strips):
                    cursor -= strip.width
                    y = margin_y + max(0, (canvas_height - margin_y * 2 - strip.height) // 2)
                    canvas.paste(strip, (cursor, y))
                    if index < len(inter_gaps):
                        cursor -= inter_gaps[index]

                first_page = page_ids[0] if page_ids else int(row.get("page", 0) or 0)
                target = folder / "images" / f"page_{first_page:04d}" / f"{row['row_id'].replace(':', '_')}.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                had_evidence = bool(str(row.get("evidence_image", "") or ""))
                canvas.save(target, format="PNG", optimize=False, compress_level=1)
                payload_bytes = target.read_bytes()
                row["evidence_image"] = target.relative_to(folder).as_posix()
                row["evidence_image_meta"] = {
                    "profile": "canonical_role_evidence_v5_cross_page_columns",
                    "layout": "rtl_physical_column_strips",
                    "covered_column_ids": list(expected_ids),
                    "coverage_complete": True,
                    "source_pages": list(page_ids),
                    "cross_page": True,
                    "exported_size": [int(canvas.width), int(canvas.height)],
                    "mode": "L",
                    "format": "PNG",
                    "png_encoding_lossless": True,
                    "png_compress_level": 1,
                    "review_copy_only": True,
                    "file_sha256": hashlib.sha256(payload_bytes).hexdigest(),
                    "file_bytes": len(payload_bytes),
                    "ocr_input_unchanged": True,
                }
                if not had_evidence:
                    written += 1
            finally:
                canvas.close()
        except Exception as exc:
            row.setdefault("evidence_export_errors", []).append({
                "stage": "cross_page_sentence_composition",
                "error_type": type(exc).__name__,
                "message": str(exc)[:240],
            })
        finally:
            for strip in strips:
                try:
                    strip.close()
                except Exception:
                    pass
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
        compressed = gzip.compress(raw, compresslevel=1, mtime=0)
        relative = f"RECOVERY/model_{index + 1:02d}.json.gz"
        target = folder / relative
        target.write_bytes(compressed)
        role_meta = _source_document_role_metadata(
            document, index, str(labels[index] if index < len(labels) else f"OCR 模型 {index + 1}")
        )
        model_files.append({
            "model_index": index,
            "label": str(labels[index] if index < len(labels) else f"OCR 模型 {index + 1}"),
            "path": relative,
            "compressed_sha256": _sha256(compressed),
            "document_sha256": _sha256(raw),
            "source_engine": str(getattr(document.metadata, "source_engine", "") or ""),
            "structure_sha256": _source_structure_hash(document),
            "layout_sha256": layout_hash(document),
            "role": str(role_meta.get("role", "") or ""),
            "role_label": str(role_meta.get("role_label", "") or ""),
            "input_granularity": str(role_meta.get("input_granularity", "unknown") or "unknown"),
            "role_schema": int(role_meta.get("role_schema", 0) or 0),
        })
    selection_records = []
    for _identity_key, record in sorted((fusion_selection_records or {}).items(), key=lambda item: item[0]):
        if not isinstance(record, dict):
            continue
        record_columns = [str(value) for value in (record.get("column_ids") or []) if str(value)]
        sentence_group_id = str(record.get("sentence_group_id", "") or "")
        if not record_columns or not sentence_group_id:
            raise SourceCorrectionError(
                "当前恢复记录必须同时包含 sentence_group_id 与 column_ids；不再推断旧列级 key。"
            )
        value = {
            "sentence_group_id": sentence_group_id,
            "column_ids": record_columns,
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
            compressed_overlay = gzip.compress(raw_overlay, compresslevel=1, mtime=0)
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
        "fusion_selection_records": selection_records,
        "canonical_decisions": [copy.deepcopy(item) for item in (canonical_decisions or []) if isinstance(item, dict)],
        "current_row_index": max(0, int(current_row_index or 0)),
        "source_pages": source_pages,
        "ruby_overlay": ruby_overlay_file,
        "instructions": (
            "重新打开程序后可直接选择本纠错 ZIP 恢复全部 OCR 文档；若 PDF/图片临时路径改变，"
            "先在页面管理重新载入同一批页面，程序只重绑图片路径，不重新 OCR。"
        ),
    }
    (recovery / "session_manifest.json").write_bytes(_json_bytes(manifest, pretty=True))
    return manifest



def _comparison_from_sealed_package_rows(
    labels: Sequence[str],
    alignment_rows: Sequence[dict],
    result_rows: Sequence[dict],
    registry_models: Sequence[dict],
) -> MultiOcrComparison | None:
    """Restore the exact exported row grouping instead of re-aligning it.

    Recovery packages are intended to be authoritative session snapshots.  A
    newer alignment algorithm may legitimately split/merge the same physical
    columns differently, so recomputing comparison rows during recovery can
    change stable row/sentence identities.  The sealed alignment snapshot plus
    raw model texts is sufficient to restore the original grouping exactly.
    """
    if not alignment_rows or len(alignment_rows) != len(result_rows):
        return None
    model_ids = ["" for _ in labels]
    for item in registry_models or ():
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("model_index", -1))
        except Exception:
            continue
        if 0 <= index < len(model_ids):
            model_ids[index] = str(item.get("model_id", "") or "")
    if any(not value for value in model_ids):
        return None

    restored_rows: list[MultiOcrRow] = []
    for index, (alignment, exported) in enumerate(zip(alignment_rows, result_rows)):
        if not isinstance(alignment, dict) or not isinstance(exported, dict):
            return None
        base = exported.get("base_model_texts") or {}
        if not isinstance(base, dict):
            return None
        texts = [str(base.get(model_id, "") or "") for model_id in model_ids]
        seeded = []
        for evidence in exported.get("model_evidence") or ():
            if not isinstance(evidence, dict) or not evidence.get("seeded_reuse"):
                continue
            try:
                model_index = int(evidence.get("model_index", -1))
            except Exception:
                continue
            if 0 <= model_index < len(texts):
                seeded.append(model_index)
        if not seeded:
            seeded = [
                int(value) for value in (exported.get("consensus_seeded_models") or [])
                if isinstance(value, int) or str(value).isdigit()
            ]
        independent = [
            (model_index, text) for model_index, text in enumerate(texts)
            if model_index not in set(seeded) and str(text or "").strip()
        ]
        chosen_index = independent[0][0] if independent else 0
        if independent:
            counts: dict[str, list[int]] = {}
            for model_index, text in independent:
                counts.setdefault(text, []).append(model_index)
            chosen_group = max(counts.values(), key=lambda values: (len(values), -min(values)))
            chosen_index = min(chosen_group)
        restored_rows.append(MultiOcrRow(
            index=index,
            texts=texts,
            chosen_index=chosen_index,
            confidence=0.0,
            reason="恢复裁决包密封的原始 OCR 对齐与候选。",
            primary_block_id=str(alignment.get("primary_block_id", "") or ""),
            primary_segment_index=int(alignment.get("primary_segment_index", 0) or 0),
            block_type=str(alignment.get("block_type", BlockType.PARAGRAPH.value) or BlockType.PARAGRAPH.value),
            page=int(alignment.get("page", exported.get("page", 0)) or 0),
            column_ids=tuple(str(value) for value in (alignment.get("column_ids") or []) if str(value)),
            atomic=bool(alignment.get("atomic", False)),
            alignment_status=str(exported.get("status", "restored") or "restored"),
            sentence_group_id=str(alignment.get("sentence_group_id", "") or ""),
            consensus_seeded_models=tuple(sorted(set(seeded))),
        ))
    comparison = MultiOcrComparison(
        labels=list(labels),
        rows=restored_rows,
        alignment_mode="column_id_consensus",
        column_anchored_rows=len(restored_rows),
        chapter_atomic_rows=sum(bool(row.atomic) for row in restored_rows),
    )
    for row in restored_rows:
        if row.is_conflict:
            comparison.conflict_rows += 1
        elif row.provisional_consensus:
            comparison.provisional_consensus_rows += 1
        else:
            comparison.exact_rows += 1
        if row.single_model_result:
            comparison.single_model_only_rows += 1
    return comparison


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
        authority_name = "AI_OUTPUT/model_corrections.json"
        if authority_name not in names:
            raise SourceCorrectionError("该 ZIP 不是当前裁决包：缺少 AI_OUTPUT/model_corrections.json。")
        if int(archive.getinfo(authority_name).file_size) > _MAX_CORRECTION_JSON_BYTES:
            raise SourceCorrectionError("裁决 authority 文件超过安全大小上限。")
        try:
            authority_payload = json.loads(archive.read(authority_name).decode("utf-8-sig"))
        except Exception as exc:
            raise SourceCorrectionError(f"无法读取当前裁决 authority：{exc}") from exc
        if not isinstance(authority_payload, dict):
            raise SourceCorrectionError("当前裁决 authority 顶层必须是对象。")
        _validate_payload(authority_payload)
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
            raise SourceCorrectionError("纠错包中的恢复清单版本不受当前开发版支持。")
        allowed_manifest_fields = {
            "schema", "package_id", "created_at", "model_count", "models",
            "fusion_selection_records", "canonical_decisions", "current_row_index",
            "source_pages", "ruby_overlay", "instructions",
        }
        unknown_manifest_fields = sorted(set(manifest) - allowed_manifest_fields)
        if unknown_manifest_fields:
            raise SourceCorrectionError(
                "当前恢复清单包含已废弃或未知字段：" + ", ".join(unknown_manifest_fields)
            )
        if str(manifest.get("package_id") or "") != str(authority_payload.get("package_id") or ""):
            raise SourceCorrectionError("恢复清单 package_id 与裁决 authority 不一致。")
        registry_name = "01_model_registry.json"
        if registry_name not in names:
            raise SourceCorrectionError("当前裁决包缺少 01_model_registry.json。")
        try:
            registry_payload = json.loads(archive.read(registry_name).decode("utf-8-sig"))
        except Exception as exc:
            raise SourceCorrectionError(f"无法读取模型注册表：{exc}") from exc
        if not isinstance(registry_payload, dict) or registry_payload.get("schema") != MODEL_REGISTRY_SCHEMA:
            raise SourceCorrectionError("模型注册表 schema 不受当前开发版支持。")
        if str(registry_payload.get("package_id") or "") != str(authority_payload.get("package_id") or ""):
            raise SourceCorrectionError("模型注册表 package_id 与裁决 authority 不一致。")
        allowed_registry_fields = {"schema", "package_id", "models", "input_audit"}
        unknown_registry_fields = sorted(set(registry_payload) - allowed_registry_fields)
        if unknown_registry_fields:
            raise SourceCorrectionError("模型注册表包含未知字段：" + ", ".join(unknown_registry_fields))
        registry_models = registry_payload.get("models", [])
        if not isinstance(registry_models, list):
            raise SourceCorrectionError("模型注册表格式无效。")
        recovery_registry_by_index: dict[int, dict] = {}
        for registry_item in registry_models:
            if not isinstance(registry_item, dict):
                raise SourceCorrectionError("模型注册表包含非对象条目。")
            raw_registry_index = registry_item.get("model_index", -1)
            registry_index = int(-1 if raw_registry_index is None else raw_registry_index)
            if registry_index < 0 or registry_index in recovery_registry_by_index:
                raise SourceCorrectionError("模型注册表索引缺失或重复。")
            recovery_registry_by_index[registry_index] = registry_item

        model_items = manifest.get("models")
        if not isinstance(model_items, list) or not 2 <= len(model_items) <= 6:
            raise SourceCorrectionError("恢复清单必须包含 2～6 份 OCR 文档。")
        for required_name in ("02_alignment_snapshot.json", "03_all_model_results.jsonl"):
            if required_name not in names:
                raise SourceCorrectionError(f"当前裁决包缺少密封恢复文件：{required_name}")
        try:
            alignment_payload = json.loads(archive.read("02_alignment_snapshot.json").decode("utf-8-sig"))
            if not isinstance(alignment_payload, dict) or alignment_payload.get("schema") != ALIGNMENT_SNAPSHOT_SCHEMA:
                raise SourceCorrectionError("密封对齐快照 schema 不受当前开发版支持。")
            if str(alignment_payload.get("package_id") or "") != str(authority_payload.get("package_id") or ""):
                raise SourceCorrectionError("密封对齐快照 package_id 与裁决 authority 不一致。")
            allowed_alignment_fields = {"schema", "package_id", "alignment_snapshot_sha256", "rows"}
            unknown_alignment_fields = sorted(set(alignment_payload) - allowed_alignment_fields)
            if unknown_alignment_fields:
                raise SourceCorrectionError("密封对齐快照包含未知字段：" + ", ".join(unknown_alignment_fields))
            sealed_alignment_rows = alignment_payload.get("rows", [])
            sealed_result_rows = [
                json.loads(line)
                for line in archive.read("03_all_model_results.jsonl").decode("utf-8-sig").splitlines()
                if line.strip()
            ]
        except Exception as exc:
            raise SourceCorrectionError(f"无法读取密封 OCR 对齐快照：{exc}") from exc
        if not isinstance(sealed_alignment_rows, list) or not sealed_alignment_rows:
            raise SourceCorrectionError("当前裁决包的密封对齐快照为空或无效。")
        if not isinstance(sealed_result_rows, list) or not sealed_result_rows:
            raise SourceCorrectionError("当前裁决包的密封模型结果为空或无效。")
        authority_row_count = int(authority_payload.get("row_count", -1) or -1)
        if authority_row_count < 0 or len(sealed_alignment_rows) != authority_row_count:
            raise SourceCorrectionError(
                "密封对齐快照行数与裁决 authority 不一致；当前包已损坏或不是同一导出会话。"
            )
        if len(sealed_result_rows) != authority_row_count:
            raise SourceCorrectionError(
                "密封模型结果行数与裁决 authority 不一致；当前包已损坏或不是同一导出会话。"
            )
        expected_alignment_sha = str(authority_payload.get("alignment_snapshot_sha256", "") or "")
        if not expected_alignment_sha or _sha256(sealed_alignment_rows) != expected_alignment_sha:
            raise SourceCorrectionError("密封对齐快照哈希与裁决 authority 不一致。")
        if str((alignment_payload or {}).get("alignment_snapshot_sha256", "") or "") != expected_alignment_sha:
            raise SourceCorrectionError("02_alignment_snapshot.json 的声明哈希与裁决 authority 不一致。")
        if len(registry_models) != len(model_items):
            raise SourceCorrectionError("恢复清单模型数量与 01_model_registry.json 不一致。")
        if int(manifest.get("model_count", -1) or -1) != len(model_items):
            raise SourceCorrectionError("恢复清单 model_count 与模型记录数量不一致。")
        if sorted(recovery_registry_by_index) != list(range(len(model_items))):
            raise SourceCorrectionError("模型注册表索引必须从 0 连续到 model_count-1。")

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
            registry_item = recovery_registry_by_index.get(position - 1)
            if not isinstance(registry_item, dict):
                raise SourceCorrectionError(f"恢复模型 {position} 缺少对应模型注册表条目。")
            restored_role = str(item.get("role", "") or "")
            restored_role_schema = int(item.get("role_schema", 0) or 0)
            restored_role_label = str(item.get("role_label", "") or "")
            restored_granularity = str(item.get("input_granularity", "") or "")
            if not restored_role or not restored_role_schema or not restored_granularity:
                raise SourceCorrectionError(f"恢复模型 {position} 缺少当前格式必需的角色/输入粒度。")
            if restored_role != str(registry_item.get("role", "") or ""):
                raise SourceCorrectionError(f"恢复模型 {position} 的 role 与模型注册表不一致。")
            if restored_granularity != str(registry_item.get("input_granularity", "") or ""):
                raise SourceCorrectionError(f"恢复模型 {position} 的 input_granularity 与模型注册表不一致。")
            document.metadata.__dict__["multi_ocr_role"] = restored_role
            document.metadata.__dict__["multi_ocr_role_schema"] = restored_role_schema
            if restored_role_label:
                document.metadata.__dict__["multi_ocr_role_label"] = restored_role_label
            document.metadata.__dict__["multi_ocr_input_granularity"] = restored_granularity
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
    _report_progress(progress_callback, "恢复 OCR 对齐快照", 1, 2)
    registry_models = [recovery_registry_by_index[index] for index in sorted(recovery_registry_by_index)]
    comparison = _comparison_from_sealed_package_rows(
        labels, sealed_alignment_rows, sealed_result_rows, registry_models
    )
    if comparison is None:
        raise SourceCorrectionError("当前裁决包的密封 OCR 对齐快照无法恢复；不再回退重新对齐。")
    _report_progress(progress_callback, "恢复 OCR 对齐快照", 2, 2)
    selection_records = {}
    for item in manifest.get("fusion_selection_records", []):
        if not isinstance(item, dict) or not item.get("column_ids"):
            continue
        record = {
            "sentence_group_id": str(item.get("sentence_group_id", "") or ""),
            "column_ids": list(item.get("column_ids") or []),
            "text": str(item.get("text", "") or ""),
            "delete_intentionally": bool(item.get("delete_intentionally", False)),
            "display_label": str(item.get("display_label", "") or ""),
            "reason": str(item.get("reason", "") or ""),
            "confidence": float(item.get("confidence", 0.0) or 0.0),
            "selection_origin": str(item.get("selection_origin", "") or ""),
        }
        key = canonical_decision_key(record)
        if key:
            selection_records[key] = record
    report = {
        "schema": "novel_formatter.multi_ocr_recovery_report.v2",
        "path": str(source),
        "package_id": str(manifest.get("package_id", "") or ""),
        "model_count": len(documents),
        "row_count": len(comparison.rows),
        "current_row_index": max(0, int(manifest.get("current_row_index", 0) or 0)),
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
    fusion_selection_records: dict[tuple[str, ...], dict] | None = None,
    canonical_decisions: Sequence[dict] | None = None,
    review_prior_decisions: bool = False,
    review_provisional_consensus: bool = False,
    review_common_mode_risk: bool = True,
    current_row_index: int = 0,
    ruby_overlay_source: UnifiedDocument | dict | None = None,
    bundle_profile: str = "full_with_quick",
) -> dict:
    bundle_profile = str(bundle_profile or "full_with_quick").strip().lower()
    if bundle_profile not in {"full_with_quick", "ai_quick"}:
        raise SourceCorrectionError(f"未知裁决包 profile：{bundle_profile}")
    if bundle_profile == "ai_quick":
        include_recovery_snapshot = False
    output = Path(output_path).expanduser()
    if output.suffix.lower() != ".zip":
        output = output.with_suffix(".zip")
    output.parent.mkdir(parents=True, exist_ok=True)
    # The GUI callback may cross a worker/thread boundary and can be orders of
    # magnitude more expensive than the underlying export work.  Keep the
    # public bundle exporter deliberately stage-granular: inner helpers still
    # support detailed progress for diagnostics/tests, but a normal GUI export
    # emits only a handful of heartbeats.  This prevents a full-book package
    # from spending minutes dispatching hundreds/thousands of progress events.
    _report_progress(progress_callback, "建立裁决数据", 0, 1)
    payload = build_source_correction_payload(
        documents, labels, comparison,
        canonical_decisions=canonical_decisions,
        review_prior_decisions=review_prior_decisions,
        review_provisional_consensus=review_provisional_consensus,
        review_common_mode_risk=review_common_mode_risk,
        progress_callback=None,
    )
    _report_progress(progress_callback, "建立裁决数据", 1, 1)
    input_audit_rows, input_audit_summary = _build_detailed_ocr_input_audit(
        documents, payload.get("model_registry") or []
    )
    input_audit_summary = {
        **input_audit_summary,
        "path": "07_ocr_input_audit.jsonl",
    }
    payload["ocr_input_audit"] = input_audit_summary
    payload["evidence_export_profile"] = {
        "name": "canonical_role_evidence_v5_strict",
        "max_height": 1600,
        "mode": "L",
        "format": "PNG",
        "ocr_input_unchanged": True,
        "physical_column_evidence": True,
        "adjudication_hint": "长句优先按 physical_column_evidence / 冲突段核验，不要把多列整句重新 OCR 后做全字符串匹配。",
    }
    with tempfile.TemporaryDirectory(prefix="nf_source_correction_") as temp:
        folder = Path(temp) / output.stem
        folder.mkdir(parents=True, exist_ok=True)
        if include_images:
            _report_progress(progress_callback, "生成裁决证据图", 0, 1)
            image_count = _export_conflict_images(
                folder, documents, payload["rows"], progress_callback=None
            )
            _report_progress(progress_callback, "生成裁决证据图", 1, 1)
        else:
            image_count = 0
        evidence_error_rows = [
            row for row in payload["rows"]
            if isinstance(row, dict) and row.get("evidence_export_errors")
        ]
        payload["evidence_export_diagnostics"] = {
            "row_images_written": int(image_count),
            "rows_with_errors": len(evidence_error_rows),
            "error_count": sum(len(row.get("evidence_export_errors") or []) for row in evidence_error_rows),
            "row_ids_with_errors": [str(row.get("row_id", "") or "") for row in evidence_error_rows[:100]],
        }
        editable_rows = [row for row in payload["rows"] if isinstance(row, dict) and row.get("editable")]
        missing_row_evidence = [
            str(row.get("row_id", "") or "") for row in editable_rows
            if include_images and not str(row.get("evidence_image", "") or "")
        ]
        missing_column_evidence: list[dict[str, object]] = []
        incomplete_cross_page_row_evidence: list[dict[str, object]] = []
        expected_column_evidence = 0
        actual_column_evidence = 0
        for row in editable_rows:
            expected = [str(value) for value in (row.get("column_ids") or []) if str(value)]
            actual = {
                str(item.get("column_id", "") or "")
                for item in (row.get("physical_column_evidence") or [])
                if isinstance(item, dict) and str(item.get("column_id", "") or "")
            }
            expected_column_evidence += len(expected)
            actual_column_evidence += len(actual)
            missing = [value for value in expected if value not in actual]
            if include_images and missing:
                missing_column_evidence.append({
                    "row_id": str(row.get("row_id", "") or ""),
                    "column_ids": missing,
                })
            page_prefixes = {
                value.split(":", 1)[0]
                for value in expected
                if ":" in value and value.split(":", 1)[0]
            }
            if include_images and len(page_prefixes) > 1:
                meta = row.get("evidence_image_meta") or {}
                covered = {
                    str(value) for value in (meta.get("covered_column_ids") or []) if str(value)
                } if isinstance(meta, dict) else set()
                if not bool(isinstance(meta, dict) and meta.get("coverage_complete")) or covered != set(expected):
                    incomplete_cross_page_row_evidence.append({
                        "row_id": str(row.get("row_id", "") or ""),
                        "expected_column_ids": expected,
                        "covered_column_ids": sorted(covered),
                    })
        primary_columns, _primary_column_source = physical_column_text_snapshot(documents[0])
        primary_column_ids = set(primary_columns)
        comparison_column_ids = {
            str(column_id)
            for row in payload["rows"] if isinstance(row, dict)
            for column_id in (row.get("column_ids") or []) if str(column_id)
        }
        missing_from_comparison = sorted(primary_column_ids - comparison_column_ids)
        unknown_comparison_columns = sorted(comparison_column_ids - primary_column_ids)
        payload["evidence_coverage"] = {
            "schema": "novel_formatter.multi_ocr_evidence_coverage.v1",
            "path": "08_evidence_coverage.json",
            "strict_provisional_review": bool(payload.get("review_provisional_consensus", False)),
            "comparison_row_count": len(payload["rows"]),
            "editable_row_count": len(editable_rows),
            "row_evidence_expected": len(editable_rows) if include_images else 0,
            "row_evidence_present": sum(bool(row.get("evidence_image")) for row in editable_rows),
            "missing_row_evidence_count": len(missing_row_evidence),
            "missing_row_evidence_ids": missing_row_evidence[:200],
            "physical_column_evidence_expected": expected_column_evidence if include_images else 0,
            "physical_column_evidence_present": actual_column_evidence,
            "missing_physical_column_evidence_rows": len(missing_column_evidence),
            "missing_physical_column_evidence": missing_column_evidence[:200],
            "incomplete_cross_page_row_evidence_count": len(incomplete_cross_page_row_evidence),
            "incomplete_cross_page_row_evidence": incomplete_cross_page_row_evidence[:200],
            "primary_physical_column_count": len(primary_column_ids),
            "comparison_physical_column_count": len(comparison_column_ids),
            "primary_columns_missing_from_comparison_count": len(missing_from_comparison),
            "primary_columns_missing_from_comparison": missing_from_comparison[:200],
            "comparison_unknown_column_count": len(unknown_comparison_columns),
            "comparison_unknown_columns": unknown_comparison_columns[:200],
            "ocr_input_audit_record_count": int(input_audit_summary.get("record_count", 0) or 0),
            "ocr_input_audit_unavailable_records": int(input_audit_summary.get("unavailable_records", 0) or 0),
            "coverage_complete_for_known_columns": bool(
                (not include_images or (
                    not missing_row_evidence
                    and not missing_column_evidence
                    and not incomplete_cross_page_row_evidence
                ))
                and not missing_from_comparison
                and not unknown_comparison_columns
            ),
            "detector_coverage_guaranteed": False,
            "all_ocr_errors_guaranteed_fixed_after_ai": False,
            "limitation": (
                "该清单验证已知物理列/句级证据是否完整，不证明页面检测器从扫描页中发现了所有正文；"
                "三模型共同错、整列漏检、错误阅读顺序或 AI 自身误判仍需页面级/终校检查。"
            ),
        }
        _report_progress(progress_callback, "封装裁决清单", 0, 1)
        payload["immutable_manifest_sha256"] = _sha256(_immutable_projection(payload))
        manifest = {key: value for key, value in payload.items() if key not in {"rows", "alignment_snapshot"}}
        (folder / "00_manifest.json").write_bytes(_json_bytes(manifest, pretty=True))
        (folder / "01_model_registry.json").write_bytes(_json_bytes({
            "schema": MODEL_REGISTRY_SCHEMA,
            "package_id": payload["package_id"],
            "models": payload["model_registry"],
            "input_audit": input_audit_summary,
        }, pretty=True))
        _write_jsonl(folder / "07_ocr_input_audit.jsonl", input_audit_rows)
        (folder / "08_evidence_coverage.json").write_bytes(
            _json_bytes(payload.get("evidence_coverage", {}), pretty=True)
        )
        (folder / "02_alignment_snapshot.json").write_bytes(_json_bytes({
            "schema": ALIGNMENT_SNAPSHOT_SCHEMA,
            "package_id": payload["package_id"],
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
                    "model_evidence": row.get("model_evidence", []),
                    "independent_model_ids": row.get("independent_model_ids", []),
                    "evidence_image": row.get("evidence_image", ""),
                    "evidence_image_meta": row.get("evidence_image_meta", {}),
                    "physical_column_evidence": row.get("physical_column_evidence", []),
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
        # JSON before compression.  Keep the human-readable index compact that point to the single sealed authority file.
        compact_conflict_rows = [
            {
                "row_id": row["row_id"],
                "row_index": row["row_index"],
                "page": row["page"],
                "column_ids": list(row.get("column_ids") or []),
                "status": row.get("status", ""),
                "decision_state": row.get("decision_state", ""),
                "base_row_sha256": row.get("base_row_sha256", ""),
                "model_evidence": row.get("model_evidence", []),
                "independent_model_ids": row.get("independent_model_ids", []),
                "evidence_image": row.get("evidence_image", ""),
                "evidence_image_meta": row.get("evidence_image_meta", {}),
                "physical_column_evidence": row.get("physical_column_evidence", []),
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
        # AI-facing compact task stream.  Coverage is identical to the editable
        # queue; only redundant internal metadata is removed.  Stable bindings,
        # hashes and the authoritative writable payload remain in AI_OUTPUT.
        compact_ai_tasks = []
        editable_payload_rows = [row for row in payload["rows"] if row.get("editable")]
        for task_number, row in enumerate(editable_payload_rows, start=1):
            try:
                row_index = int(row.get("row_index", -1))
            except (TypeError, ValueError, OverflowError):
                row_index = -1
            def _context_at(index: int) -> tuple[str, list[str]]:
                if not 0 <= index < len(payload["rows"]):
                    return "", []
                item = payload["rows"][index]
                stable_text = _ai_stable_context_text(item)
                if stable_text:
                    return stable_text, []
                verdict = item.get("resolved_verdict") if isinstance(item.get("resolved_verdict"), dict) else {}
                final_text = str(verdict.get("final_text", "") or "")
                if final_text:
                    return final_text, []
                candidates = []
                for candidate in _compact_ai_candidates(item):
                    text = str(candidate.get("t", "") or "")
                    if bool(candidate.get("fail", False)) or not text.strip() or text in candidates:
                        continue
                    candidates.append(text)
                if len(candidates) == 1:
                    return candidates[0], []
                return "", candidates[:4]

            previous_text, previous_candidates = _context_at(row_index - 1)
            previous_text_2, previous_candidates_2 = _context_at(row_index - 2)
            next_text, next_candidates = _context_at(row_index + 1)
            next_text_2, next_candidates_2 = _context_at(row_index + 2)
            candidates = _compact_ai_candidates(row)
            compact_ai_tasks.append({
                "id": f"T{task_number:05d}",
                "row": row_index,
                "page": int(row.get("page", 0) or 0),
                "before": previous_text,
                "before2": previous_text_2,
                "before_candidates": previous_candidates,
                "before2_candidates": previous_candidates_2,
                "c": candidates,
                "after": next_text,
                "after2": next_text_2,
                "after_candidates": next_candidates,
                "after2_candidates": next_candidates_2,
                "img": str(row.get("evidence_image", "") or ""),
                "cols": list(row.get("column_ids") or []),
                "status": str(row.get("status", "") or ""),
                "risk": copy.deepcopy(row.get("common_mode_risk")) if row.get("common_mode_risk") else None,
            })
        _write_jsonl(folder / "04_ai_tasks_compact.jsonl", compact_ai_tasks)
        (folder / "AI_INSTRUCTIONS.md").write_text(
            "# AI OCR 裁决快捷入口\n\n"
            "按 `04_ai_tasks_compact.jsonl` 顺序处理全部任务；不要因为 2:1 或 3:3 就跳过任务。"
            "每条只需结合 before/after、各模型候选 c 与 img 判断；c[].fail=true 表示该 OCR 在该句失败/占位，"
            "不是原文字符 □。\n\n"
            "推荐直接使用同目录自动生成的 `_GPT.zip`：GPT 只写 `AI_OUTPUT/answers.jsonl`，"
            "再运行 `python adjudicate.py finish` 生成 `AI_IMPORT.zip`，无需读取或修改 14MB 级 authority JSON。\n"
            "status=common_mode_risk 的任务是本地一致后的小规模高风险审计；必须以原图为准，risk 只解释为何送审，绝不是自动纠错答案。\n"
            "若图片与候选都不足以确定，返回 unresolved，不要猜。\n",
            encoding="utf-8",
        )

        conflict_view_bytes = _json_bytes(conflict_view)
        (folder / "04_editable_conflicts.json").write_bytes(conflict_view_bytes)
        # Backwards-compatible read-only alias kept for V4/V5 consumers that
        # still look for the historical filename.  It is byte-identical to the
        # current browse view and is never authoritative for import.
        (folder / "04_pending_ai_review.json").write_bytes(conflict_view_bytes)
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
            _report_progress(progress_callback, "保存可恢复 OCR 会话", 0, 1)
            recovery_manifest = _write_recovery_snapshot(
                folder, documents, labels, package_id=payload["package_id"],
                fusion_selection_records=fusion_selection_records,
                canonical_decisions=canonical_decisions,
                current_row_index=current_row_index,
                ruby_overlay_source=ruby_overlay_source,
                progress_callback=None,
            )
            _report_progress(progress_callback, "保存可恢复 OCR 会话", 1, 1)
        _report_progress(progress_callback, "封装裁决清单", 1, 1)
        readme = f"""# 多模型 OCR 逐源纠错包

本包同时支持“逐模型纠错”和“最终融合裁决”。{("此前已接受的冲突裁决会作为 prior_decision_context 重新开放复审；AI 可保留，也可提交更好的结果。" if review_prior_decisions else "此前已安全接受的最终裁决会写入 resolved_verdict 并锁定。")}
只有 `editable=true` 的行需要提交给 AI。

逐模型纠错：只在 `editable_conflict` 段的 `model_edits` 中填写错误模型的修正文字，
正确模型保持省略。导入后程序仅据此生成“AI逐模型纠错结果”融合候选，不回写任何
OCR 模型、不改变原物理列、不重新对齐；原始各模型分歧继续显示并可重新选择。

最终融合裁决（可选）：在 `ai_verdict.final_text` 填写完整整行正文。仅做逐模型纠错时
可保持 final_text 为空。调序、漏句、重复、跨列差异可使用 final_text 整体裁决。

证据图与输入审计：`images/` 是只供 AI/人工查看的无损灰度 PNG；超过 1600px 时按比例缩小。
这不会修改扫描原图、共享物理列图或任何模型输入。每行 `evidence_image_meta` 记录原尺寸、
导出尺寸、缩放率和文件 SHA-256。`07_ocr_input_audit.jsonl` 记录各模型稳定物理列的实际
输入哈希、传输方式与共享关系；若当前会话缺少哈希则明确标记 unavailable，不会猜测。
`08_evidence_coverage.json` 另外核对待审行/物理列证据覆盖和 comparison lineage；它明确区分
“已知列证据完整”与“扫描页检测绝对无漏列”这两个不同命题。

不得修改 `base_model_texts`、`model_texts`、锁定一致段、模型 ID、物理列 ID 或哈希字段。

包 ID：`{payload['package_id']}`  
真正分歧总数：{payload['editable_conflict_rows']}  
两模型共同候选：{payload.get('provisional_consensus_rows', 0)}（{'全部进入核验' if payload.get('review_provisional_consensus') else 'Lean 模式自动保留'}）  
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
        if bundle_profile == "full_with_quick":
            package_files = [path for path in sorted(folder.rglob("*")) if path.is_file()]
            _report_progress(progress_callback, "压缩逐源纠错包", 0, 1)
            local_zip = Path(temp) / f".{output.name}.building"
            try:
                with zipfile.ZipFile(local_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
                    for path in package_files:
                        relative = path.relative_to(folder).as_posix()
                        compression = (
                            zipfile.ZIP_STORED
                            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gz"}
                            else zipfile.ZIP_DEFLATED
                        )
                        archive.write(path, relative, compress_type=compression)
                with zipfile.ZipFile(local_zip, "r") as check:
                    bad = check.testzip()
                    if bad:
                        raise SourceCorrectionError(f"逐源纠错包 ZIP CRC 失败：{bad}")
                _report_progress(progress_callback, "写入裁决包目标位置", 0, 1)
                with atomic_output_path(output) as destination_tmp:
                    shutil.copyfile(local_zip, destination_tmp)
                    with zipfile.ZipFile(destination_tmp, "r") as check:
                        bad = check.testzip()
                        if bad:
                            raise SourceCorrectionError(f"目标裁决包 ZIP CRC 失败：{bad}")
                _report_progress(progress_callback, "写入裁决包目标位置", 1, 1)
            finally:
                local_zip.unlink(missing_ok=True)
            _report_progress(progress_callback, "压缩逐源纠错包", 1, 1)
            ai_quick_report = _write_ai_quick_bundle(
                folder, output=output, payload=payload, compact_ai_tasks=compact_ai_tasks
            )
        else:
            _report_progress(progress_callback, "压缩 GPT 裁决交换包", 0, 1)
            ai_quick_report = _write_ai_quick_bundle(
                folder, output=output, payload=payload, compact_ai_tasks=compact_ai_tasks, direct_output=True
            )
            _report_progress(progress_callback, "压缩 GPT 裁决交换包", 1, 1)
    return {
        "path": str(output if bundle_profile == "ai_quick" else output),
        "bundle_profile": bundle_profile,
        "package_id": payload["package_id"],
        "row_count": payload["row_count"],
        "editable_conflict_rows": payload["editable_conflict_rows"],
        "provisional_consensus_rows": payload.get("provisional_consensus_rows", 0),
        "review_provisional_consensus": bool(payload.get("review_provisional_consensus", False)),
        "editable_provisional_rows": payload.get("editable_provisional_rows", 0),
        "editable_review_rows": payload.get("editable_review_rows", payload["editable_conflict_rows"]),
        "pending_conflict_rows": payload.get("pending_conflict_rows", 0),
        "pending_provisional_rows": payload.get("pending_provisional_rows", 0),
        "pending_review_rows": payload.get("pending_review_rows", 0),
        "prefilled_prior_decision_rows": payload.get("prefilled_prior_decision_rows", 0),
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
        "ai_quick_path": str(ai_quick_report.get("path", "")),
        "ai_quick_bytes": int(ai_quick_report.get("bytes", 0) or 0),
        "ai_quick_tasks": int(ai_quick_report.get("tasks", 0) or 0),
    }


def load_correction_payload(path: str | Path) -> dict:
    source = Path(path).expanduser()
    if not source.is_file():
        raise SourceCorrectionError(f"纠错文件不存在：{source}")
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source, "r") as archive:
            _validate_source_archive(archive)
            names = set(archive.namelist())
            target = "AI_OUTPUT/model_corrections.json"
            if target not in names:
                raise SourceCorrectionError(
                    "该 ZIP 不是当前裁决包：缺少 AI_OUTPUT/model_corrections.json。"
                )
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
    if schema != CANONICAL_CORRECTIONS_SCHEMA:
        _legacy_adjudication_compatibility_interface(
            payload, operation=f"schema {schema or '<missing>'} 导入"
        )
    package_schema = str(payload.get("package_schema", "") or "")
    if package_schema != SCHEMA:
        _legacy_adjudication_compatibility_interface(
            payload, operation=f"package_schema {package_schema or '<missing>'} 导入"
        )
    if int(payload.get("exchange_version") or 0) != EXCHANGE_VERSION:
        raise SourceCorrectionError(
            f"裁决包 exchange_version={payload.get('exchange_version')} 不受支持；"
            f"当前只接受 {EXCHANGE_VERSION}。"
        )
    if str(payload.get("exchange_profile") or "") != EXCHANGE_PROFILE:
        raise SourceCorrectionError("裁决包 exchange_profile 与当前开发版不一致。")
    expected = str(payload.get("immutable_manifest_sha256", "") or "")
    actual = _sha256(_immutable_projection(payload))
    if not expected or expected != actual:
        raise SourceCorrectionError("不可变清单已被修改；原始 OCR、模型身份或物理列映射不可信。")
    rows = payload.get("rows")
    registry = payload.get("model_registry")
    if not isinstance(rows, list) or not isinstance(registry, list):
        raise SourceCorrectionError("裁决结果缺少 rows 或 model_registry。")
    _validate_current_model_registry(registry, context="当前裁决包")
    model_ids = [str(item.get("model_id", "") or "") for item in registry]
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
        sentence_group_id = str(row.get("sentence_group_id", "") or "")
        if not sentence_group_id:
            raise SourceCorrectionError(f"{row_id} 缺少当前格式必需的 sentence_group_id。")
        editable = bool(row.get("editable", False))
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

def _legacy_row_requires_whole_verdict(*args, **kwargs):
    """Compatibility interface only; legacy alignment heuristics are not shipped."""
    return _legacy_adjudication_compatibility_interface(*args, operation="行级裁决迁移", **kwargs)


def _derive_legacy_canonical_verdict(*args, **kwargs):
    """Compatibility interface only; legacy verdict derivation is disabled."""
    return _legacy_adjudication_compatibility_interface(*args, operation="canonical 裁决迁移", **kwargs)


def _model_text_compatibility_score(*args, **kwargs):
    """Compatibility interface only; cross-session fuzzy matching is disabled."""
    return _legacy_adjudication_compatibility_interface(*args, operation="跨会话模糊匹配", **kwargs)


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
        "decision_id": _canonical_decision_id(
            column_ids, str(row.get("sentence_group_id", "") or "")
        ),
        "row_id": str(row.get("row_id", "") or ""),
        "row_index": int(row.get("row_index", 0) or 0),
        "sentence_group_id": str(row.get("sentence_group_id", "") or ""),
        "column_ids": column_ids,
        "final_text": final_text if accepted else "",
        "status": "accepted" if accepted else "unresolved",
        "source": "ai_per_model_source_correction_current",
        "derivation": "all_model_texts_converged_after_sparse_model_edits",
        "resolution_kind": "per_model_source_correction",
        "confidence": min(confidences) if confidences else (1.0 if accepted else 0.0),
        "reason": "；".join(reasons) or (
            "逐模型纠错后所有 OCR 模型在该物理列组完全一致。"
            if accepted else "逐模型纠错后模型文字仍未完全一致。"
        ),
        "audit_flags": flags,
        # Re-export validation must compare against the *current corrected*
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
    if not 2 <= len(docs) <= 6:
        raise SourceCorrectionError("当前 OCR 对比没有 2～6 个模型。")

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

    current_alignment = _alignment_snapshot(current_comparison)
    current_alignment_matches = current_alignment == exported_alignment
    model_ids = [str(item.get("model_id", "") or "") for item in exported_registry_list]
    # Some workspace compaction/transport revisions can change raw document
    # side-channel structure while leaving the sealed row evidence byte-for-byte
    # identical.  Overlay import operates on stable comparison rows, so accept
    # that benign drift only when *every* row identity/alignment and every model
    # text still exactly matches the export-time base evidence.  This remains
    # fail-closed for any changed OCR text or remapped row.
    current_row_evidence_matches = bool(current_alignment_matches and len(rows) == len(current_comparison.rows))
    if current_row_evidence_matches:
        for current_row, exported_row in zip(current_comparison.rows, rows):
            if not isinstance(exported_row, dict):
                current_row_evidence_matches = False
                break
            sealed = exported_row.get("base_model_texts") or {}
            current_values = [str(value or "") for value in (getattr(current_row, "texts", ()) or ())]
            sealed_values = [str(sealed.get(model_id, "") or "") for model_id in model_ids]
            if current_values[:len(model_ids)] != sealed_values:
                current_row_evidence_matches = False
                break

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
    native_accepted = 0
    prefilled_resolved = 0
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
        column_ids = [str(value) for value in (row.get("column_ids") or []) if str(value)]
        if not column_ids or not set(column_ids).issubset(expected_column_set):
            raise SourceCorrectionError(f"{row.get('row_id', '')} 的稳定物理列映射无效。")
        # A complete current per-model correction is itself a finished adjudication
        # when all effective model texts converge.  Do not throw that state away
        # merely because the optional whole-row ai_verdict was intentionally blank.
        if _row_has_explicit_model_edits(row) and not _canonical_verdict_has_output(row):
            decision = _derive_per_model_correction_decision(row, model_ids)
        else:
            decision = _read_canonical_verdict(row, model_ids, schema)
        flags = set(str(value) for value in (decision.get("audit_flags") or []))
        if "final_text_contains_placeholder" in flags or "derived_verdict_failed_text_safety" in flags:
            rejected_placeholder += 1
        if "final_text_contains_suspicious_inline_latin" in flags:
            rejected_ascii += 1
        if decision.get("status") == "accepted":
            accepted_count += 1
            if is_prefilled:
                prefilled_resolved += 1
            native_accepted += 1
        else:
            unresolved_count += 1
        decisions.append(decision)

    # Count every accepted row whose sealed export-time OCR evidence disagreed.
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

    # Overlay import deliberately leaves the live OCR comparison immutable.
    # Keep one authoritative snapshot instead of pretending that an "after"
    # comparison reflects applied model edits.
    live_comparison_snapshot = dict(before)
    all_snapshot_match = all(snapshot_matches.values())
    recomputed_document_evidence_matches = True
    if not current_alignment_matches:
        raise SourceCorrectionError(
            "当前 OCR 对齐快照与裁决包不完全一致；当前开发版不重新映射旧裁决。"
        )
    if all_snapshot_match:
        validation_mode = "strict_document_snapshot_and_alignment"
    elif current_row_evidence_matches:
        # A live comparison object can itself be stale if somebody mutates an
        # OCR document after comparison.  Before accepting document-snapshot
        # drift, rebuild the comparison once from the *current documents* and
        # require the same sealed alignment/text.  This costs a few seconds only
        # on the rare fallback path and prevents a stale comparison from making
        # changed OCR evidence look safe.
        recomputed = compare_ocr_documents(docs, labels)
        recomputed_document_evidence_matches = (
            _alignment_snapshot(recomputed) == exported_alignment
            and len(recomputed.rows) == len(rows)
        )
        if recomputed_document_evidence_matches:
            for recomputed_row, exported_row in zip(recomputed.rows, rows):
                sealed = (exported_row or {}).get("base_model_texts") or {}
                current_values = [str(value or "") for value in (getattr(recomputed_row, "texts", ()) or ())]
                sealed_values = [str(sealed.get(model_id, "") or "") for model_id in model_ids]
                if current_values[:len(model_ids)] != sealed_values:
                    recomputed_document_evidence_matches = False
                    break
        if not recomputed_document_evidence_matches:
            raise SourceCorrectionError(
                "当前 OCR 文档快照已变化，重新从当前文档构建的密封行证据也不一致；拒绝跨 OCR 会话复用裁决。"
            )
        # Safe round-trip fallback: raw document metadata/unused structural
        # side-channels drifted, but the complete sealed row evidence and stable
        # alignment are identical.  Decisions can therefore be applied to the
        # same sentence identities without pretending the raw documents match.
        validation_mode = "sealed_row_evidence_and_alignment"
    else:
        mismatched = [model_id for model_id, matches in snapshot_matches.items() if not matches]
        raise SourceCorrectionError(
            "当前 OCR 文档快照与裁决包不完全一致，且密封行证据也已变化；拒绝跨 OCR 会话复用裁决。"
            f" 不一致模型：{', '.join(mismatched[:6])}"
        )

    # Post-import textual plausibility is separate from sealed structural
    # validation.  Never reject or silently rewrite an accepted verdict here.
    from engine.ocr_adjudication_quality_risks import audit_import_quality_rows
    quality_review_warnings = audit_import_quality_rows(payload.get("rows", []))

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
        "accepted_canonical_decisions": accepted_count,
        "unresolved_canonical_decisions": unresolved_count,
        "quality_review_warning_count": len(quality_review_warnings),
        "quality_review_warnings": quality_review_warnings,
        "prefilled_resolved_decisions": prefilled_resolved,
        "native_accepted_decisions": native_accepted,
        "rejected_placeholder_decisions": rejected_placeholder,
        "rejected_suspicious_ascii_decisions": rejected_ascii,
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
        "sealed_row_evidence_matches": bool(current_row_evidence_matches),
        "recomputed_document_evidence_matches": bool(recomputed_document_evidence_matches),
        "document_snapshot_drift_accepted": bool(
            not all_snapshot_match and current_row_evidence_matches and recomputed_document_evidence_matches
        ),
        "model_layout_matches": layout_matches,
        "model_structure_matches": structure_matches,
        "model_snapshot_matches": snapshot_matches,
        "expected_physical_column_count": expected_count,
        "live_comparison_snapshot": live_comparison_snapshot,
        "comparison_stats_unchanged_by_design": True,
        "overlay_changes_live_comparison": False,
        # The snapshots are intentionally equal because this import never mutates
        # the live OCR/comparison objects.
        "before": dict(live_comparison_snapshot),
        "after": dict(live_comparison_snapshot),
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
        except Exception as exc:
            raise SourceCorrectionError(f"无法清理框架 EPUB 工作负载：{exc}") from exc

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
        with atomic_output_path(output) as temp_zip:
            with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                for path in sorted(folder.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(folder).as_posix())
            with zipfile.ZipFile(temp_zip, "r") as check:
                bad = check.testzip()
                if bad:
                    raise SourceCorrectionError(f"融合骨架包 ZIP CRC 失败：{bad}")
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
    _repaired_lineage, _unresolved_lineage = _repair_comparison_column_lineage(source_docs, comparison)
    if _unresolved_lineage:
        detail = "、".join(str(index + 1) for index in _unresolved_lineage[:8])
        raise SourceCorrectionError(
            f"当前比较仍有无法从 canonical 主文档恢复物理列 ID 的行：{detail}；为避免错误回写，已取消导出。"
        )
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
        doc_copy = source_docs[model_index].snapshot_clone()
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
