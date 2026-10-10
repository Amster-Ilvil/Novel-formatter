#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qt-free OCR adjudication service for CLI / Codex automation.

The service deliberately keeps the external agent on the *decision* side of a
strict boundary:

* the sealed round-trip package is read-only evidence;
* Hayai / NDL / 48px candidate text is never rewritten;
* an external agent writes only a small decisions JSON file;
* decisions are bound to package/hash/row IDs before they are applied;
* unresolved rows are retained unchanged and can be exported separately;
* EPUB structure is produced only by Novel Formatter's own builder.

No module in this file imports PySide6 or any UI package.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from models.document import UnifiedDocument
from engine.ocr_roundtrip_package import (
    MODE_MULTI,
    RoundtripPackageError,
    import_multi_package,
    load_package,
    save_package,
)

TASK_SCHEMA = "novel_formatter.headless_adjudication_task.v1"
DECISIONS_SCHEMA = "novel_formatter.headless_adjudication_decisions.v1"
REPORT_SCHEMA = "novel_formatter.headless_adjudication_report.v1"
UNRESOLVED_SCHEMA = "novel_formatter.headless_unresolved.v1"


class HeadlessAdjudicationError(ValueError):
    """Raised when an external decision file violates the headless contract."""


def _json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(value) -> str:
    data = value if isinstance(value, (bytes, bytearray)) else _json_bytes(value)
    return hashlib.sha256(data).hexdigest()


def _finite_float(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _explicit_bool(value, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes"}:
            return True
        if token in {"false", "0", "no", ""}:
            return False
    raise HeadlessAdjudicationError(f"{field} 必须是明确的 true/false。")


def _package_binding(package: dict) -> dict:
    return {
        "package_id": str(package.get("package_id", "") or ""),
        "schema": str(package.get("schema", "") or ""),
        "immutable_manifest_sha256": str(package.get("immutable_manifest_sha256", "") or ""),
        "editable_structure_sha256": str(package.get("editable_structure_sha256", "") or ""),
        "structure_sha256": str(package.get("structure_sha256", "") or ""),
        "layout_sha256": str(package.get("layout_sha256", "") or ""),
        "editable_count": len(package.get("editable_items") or []),
    }


def _candidate_projection(package: dict) -> list[dict]:
    rows = []
    for item in package.get("editable_items") or []:
        rows.append({
            "row_id": str(item.get("row_id", "") or ""),
            "row_index": int(item.get("row_index", 0) or 0),
            "candidates": [
                {
                    "model_index": int(candidate.get("model_index", 0) or 0),
                    "model_label": str(candidate.get("model_label", "") or ""),
                    "model_role": str(candidate.get("model_role", "") or ""),
                    "source_engine": str(candidate.get("source_engine", "") or ""),
                    "text": str(candidate.get("text", "") or ""),
                }
                for candidate in (item.get("candidates") or [])
                if isinstance(candidate, dict)
            ],
        })
    return rows


def candidate_sha256(package: dict) -> str:
    """Fingerprint immutable raw OCR candidate evidence."""
    return _sha256(_candidate_projection(package))


def _validate_package_shape(package: dict) -> None:
    if not isinstance(package, dict):
        raise HeadlessAdjudicationError("OCR 校对包必须是 JSON 对象。")
    if str(package.get("mode", "") or "") != MODE_MULTI:
        raise HeadlessAdjudicationError("Headless Codex 裁决当前只接受多模型 OCR round-trip 包。")
    items = package.get("editable_items")
    if not isinstance(items, list) or not items:
        raise HeadlessAdjudicationError("OCR 校对包没有 editable_items。")
    row_ids = [str(item.get("row_id", "") or "") for item in items if isinstance(item, dict)]
    if len(row_ids) != len(items) or any(not value for value in row_ids) or len(set(row_ids)) != len(row_ids):
        raise HeadlessAdjudicationError("OCR 校对包的 stable row ID 缺失或重复。")
    binding = _package_binding(package)
    for key in ("package_id", "immutable_manifest_sha256", "editable_structure_sha256", "structure_sha256", "layout_sha256"):
        if not binding[key]:
            raise HeadlessAdjudicationError(f"OCR 校对包缺少安全绑定字段：{key}。")


def load_multi_package(path: str | Path) -> dict:
    package = load_package(path)
    _validate_package_shape(package)
    return package


def _is_true_conflict(item: dict) -> bool:
    classification = str(item.get("review_classification", "") or "").strip().lower()
    if classification == "conflict":
        return True
    status = str(item.get("alignment_status", "") or "").strip().lower()
    return status in {"conflict", "local_reocr_recommended", "character_fused"}


def _is_existing_edit(item: dict) -> bool:
    return str(item.get("edited_text", "") or "") != str(item.get("original_fused_text", "") or "") or bool(item.get("delete_intentionally", False))


def _page_image_map(package: dict) -> dict[int, str]:
    result: dict[int, str] = {}
    for asset in package.get("assets") or []:
        if not isinstance(asset, dict) or str(asset.get("kind", "") or "") != "page":
            continue
        try:
            page = int(asset.get("page_no", 0) or 0)
        except (TypeError, ValueError, OverflowError):
            continue
        path = str(asset.get("image_path", "") or "")
        if page > 0 and path:
            result[page] = path
    return result


def _row_context(items: list[dict], index: int, radius: int) -> list[dict]:
    start = max(0, index - radius)
    end = min(len(items), index + radius + 1)
    result = []
    for cursor in range(start, end):
        if cursor == index:
            continue
        item = items[cursor]
        result.append({
            "relative": cursor - index,
            "row_id": str(item.get("row_id", "") or ""),
            "page": int(item.get("page", 0) or 0),
            "block_type": str(item.get("block_type", "") or ""),
            "text": str(item.get("edited_text", item.get("original_fused_text", "")) or ""),
        })
    return result


def build_adjudication_task(
    package: dict,
    *,
    scope: str = "conflicts",
    context_rows: int = 2,
    include_existing_edits: bool = False,
) -> dict:
    """Build a compact, read-only task for Codex/another external agent."""
    _validate_package_shape(package)
    scope = str(scope or "conflicts").strip().lower()
    if scope not in {"conflicts", "all"}:
        raise HeadlessAdjudicationError("scope 只支持 conflicts 或 all。")
    context_rows = max(0, min(int(context_rows), 8))
    items = list(package.get("editable_items") or [])
    page_images = _page_image_map(package)
    selected = []
    skipped_existing = 0
    for index, item in enumerate(items):
        if scope == "conflicts" and not _is_true_conflict(item):
            continue
        if not include_existing_edits and _is_existing_edit(item):
            skipped_existing += 1
            continue
        page = int(item.get("page", 0) or 0)
        candidates = []
        for candidate in item.get("candidates") or []:
            if not isinstance(candidate, dict):
                continue
            candidates.append({
                "model_index": int(candidate.get("model_index", 0) or 0),
                "model_label": str(candidate.get("model_label", "") or ""),
                "model_role": str(candidate.get("model_role", "") or ""),
                "source_engine": str(candidate.get("source_engine", "") or ""),
                "confidence": _finite_float(candidate.get("confidence", 0.0)),
                "text": str(candidate.get("text", "") or ""),
            })
        selected.append({
            "row_id": str(item.get("row_id", "") or ""),
            "row_index": int(item.get("row_index", index) or 0),
            "sentence_group_id": str(item.get("sentence_group_id", "") or ""),
            "page": page,
            "block_type": str(item.get("block_type", "") or ""),
            "column_ids": [str(value) for value in (item.get("column_ids") or []) if str(value)],
            "review_classification": str(item.get("review_classification", "") or ""),
            "alignment_status": str(item.get("alignment_status", "") or ""),
            "confidence": _finite_float(item.get("confidence", 0.0)),
            "reason": str(item.get("reason", "") or ""),
            "warnings": [str(value) for value in (item.get("warnings") or [])],
            "current_text": str(item.get("edited_text", "") or ""),
            "original_fused_text": str(item.get("original_fused_text", "") or ""),
            "recommended_model_index": int(item.get("recommended_model_index", -1) if item.get("recommended_model_index") is not None else -1),
            "candidates": candidates,
            "image_path": page_images.get(page, ""),
            "source_bbox": copy.deepcopy(item.get("source_bbox") or []),
            "column_geometry": copy.deepcopy(item.get("column_geometry") or []),
            "physical_column_candidates": copy.deepcopy(item.get("physical_column_candidates") or []),
            "context": _row_context(items, index, context_rows),
        })
    binding = _package_binding(package)
    return {
        "schema": TASK_SCHEMA,
        "source": {
            **binding,
            "candidate_sha256": candidate_sha256(package),
            "book": copy.deepcopy(package.get("book") or {}),
            "model_labels": [str(value) for value in (package.get("model_labels") or [])],
        },
        "policy": {
            "scope": scope,
            "selected_rows": len(selected),
            "skipped_existing_edits": skipped_existing,
            "context_rows": context_rows,
            "raw_ocr_candidates_are_read_only": True,
            "do_not_guess": True,
            "unreliable_rows_must_set_needs_manual_review": True,
            "delete_requires_delete_intentionally": True,
        },
        "output_contract": {
            "schema": DECISIONS_SCHEMA,
            "source": binding,
            "decisions": [{
                "row_id": "stable row id from this task",
                "corrected_text": "final Japanese text, or empty only for intentional deletion",
                "confidence": 0.0,
                "reason": "brief evidence-based reason",
                "needs_manual_review": False,
                "delete_intentionally": False,
            }],
        },
        "rows": selected,
    }


def task_prompt(task: dict, *, decisions_path: str = "codex_decisions.json") -> str:
    """Render a deterministic instruction file suitable for ``codex exec``."""
    if str(task.get("schema", "") or "") != TASK_SCHEMA:
        raise HeadlessAdjudicationError("不是有效的 headless adjudication task。")
    source = task.get("source") or {}
    selected = int((task.get("policy") or {}).get("selected_rows", len(task.get("rows") or [])) or 0)
    return f"""# Novel Formatter · Headless OCR adjudication

你只负责 OCR 文本裁决，不负责 EPUB 结构、页码、坐标、图片、目录或排版。

## 强制规则
1. 读取任务 JSON 中的 {selected} 个 rows；只处理这些 stable row。
2. Hayai / NDL / 48px 等 candidates 全部是只读证据，严禁修改任务 JSON 或原 round-trip package。
3. 结合候选、前后文、page/column 几何和可访问的 image_path 判断最终日文。
4. 不确定时不要猜：`needs_manual_review=true`，并保留 `corrected_text` 为当前最安全文本。
5. 非空原文只有在明确应该整句删除时才能 `delete_intentionally=true`。
6. 输出只能写到 `{decisions_path}`；不要修改源码、OCR 原始文件或 EPUB。
7. 每个 decision 必须使用任务里的原始 `row_id`；不得新增、重排或复制 stable row。

## 输出 JSON
- schema: `{DECISIONS_SCHEMA}`
- source.package_id: `{source.get('package_id','')}`
- source.immutable_manifest_sha256: `{source.get('immutable_manifest_sha256','')}`
- source.editable_structure_sha256: `{source.get('editable_structure_sha256','')}`
- source.structure_sha256: `{source.get('structure_sha256','')}`
- source.layout_sha256: `{source.get('layout_sha256','')}`
- source.editable_count: {int(source.get('editable_count',0) or 0)}
- decisions: 对任务 rows 一一给出裁决对象。

完成后不要生成 EPUB；Novel Formatter CLI 会验证 decisions 后自行写回和构建 skeleton EPUB。
"""


def write_task(task: dict, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def load_decisions(path: str | Path) -> dict:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise HeadlessAdjudicationError(f"无法读取 decisions JSON：{exc}") from exc
    if not isinstance(value, dict) or str(value.get("schema", "") or "") != DECISIONS_SCHEMA:
        raise HeadlessAdjudicationError(f"decisions.schema 必须为 {DECISIONS_SCHEMA}。")
    return value


def _validate_source_binding(package: dict, decisions: dict) -> None:
    expected = _package_binding(package)
    actual = decisions.get("source")
    if not isinstance(actual, dict):
        raise HeadlessAdjudicationError("decisions 缺少 source 安全绑定。")
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if key == "editable_count":
            try:
                actual_value = int(actual_value)
            except (TypeError, ValueError, OverflowError):
                actual_value = -1
        else:
            actual_value = str(actual_value or "")
        if actual_value != expected_value:
            raise HeadlessAdjudicationError(f"decisions 与当前 OCR 包不匹配：{key}。")


def validate_decisions(
    package: dict,
    decisions: dict,
    *,
    allowed_row_ids: Iterable[str] | None = None,
    require_complete: bool = False,
) -> dict:
    """Validate decisions without mutating ``package``."""
    _validate_package_shape(package)
    if str(decisions.get("schema", "") or "") != DECISIONS_SCHEMA:
        raise HeadlessAdjudicationError(f"decisions.schema 必须为 {DECISIONS_SCHEMA}。")
    _validate_source_binding(package, decisions)
    items = list(package.get("editable_items") or [])
    item_by_id = {str(item.get("row_id", "") or ""): item for item in items}
    allowed = set(str(value) for value in allowed_row_ids) if allowed_row_ids is not None else set(item_by_id)
    raw = decisions.get("decisions")
    if not isinstance(raw, list):
        raise HeadlessAdjudicationError("decisions.decisions 必须是数组。")
    seen: set[str] = set()
    normalized: list[dict] = []
    unresolved: list[dict] = []
    for position, decision in enumerate(raw):
        if not isinstance(decision, dict):
            raise HeadlessAdjudicationError(f"第 {position + 1} 个 decision 不是对象。")
        row_id = str(decision.get("row_id", "") or "")
        if not row_id or row_id not in item_by_id:
            raise HeadlessAdjudicationError(f"第 {position + 1} 个 decision 的 row_id 无效：{row_id or '<empty>'}。")
        if row_id not in allowed:
            raise HeadlessAdjudicationError(f"decision 越权修改未分配的 stable row：{row_id}。")
        if row_id in seen:
            raise HeadlessAdjudicationError(f"decision 重复 stable row：{row_id}。")
        seen.add(row_id)
        corrected = decision.get("corrected_text")
        if corrected is None:
            corrected = str(item_by_id[row_id].get("edited_text", "") or "")
        if not isinstance(corrected, str):
            raise HeadlessAdjudicationError(f"decision {row_id} 的 corrected_text 必须是字符串。")
        needs_manual = _explicit_bool(decision.get("needs_manual_review", False), field=f"decision {row_id} 的 needs_manual_review")
        delete_intentionally = _explicit_bool(decision.get("delete_intentionally", False), field=f"decision {row_id} 的 delete_intentionally")
        baseline_nonempty = bool(str(item_by_id[row_id].get("edited_text", item_by_id[row_id].get("original_fused_text", "")) or "").strip())
        if baseline_nonempty and not corrected.strip() and not delete_intentionally:
            raise HeadlessAdjudicationError(f"decision {row_id} 把非空正文改为空，但未声明 delete_intentionally=true。")
        normalized_decision = {
            "row_id": row_id,
            "corrected_text": corrected,
            "confidence": _finite_float(decision.get("confidence", 0.0)),
            "reason": str(decision.get("reason", "") or ""),
            "needs_manual_review": needs_manual,
            "delete_intentionally": delete_intentionally,
        }
        normalized.append(normalized_decision)
        if needs_manual:
            unresolved.append(normalized_decision)
    missing = sorted(allowed - seen)
    if require_complete and missing:
        raise HeadlessAdjudicationError(f"decisions 缺少 {len(missing)} 个要求裁决的 stable row；首个为 {missing[0]}。")
    missing_unresolved = []
    for row_id in missing:
        item = item_by_id[row_id]
        missing_unresolved.append({
            "row_id": row_id,
            "corrected_text": str(item.get("edited_text", item.get("original_fused_text", "")) or ""),
            "confidence": 0.0,
            "reason": "missing_decision",
            "needs_manual_review": True,
            "delete_intentionally": bool(item.get("delete_intentionally", False)),
        })
    return {
        "normalized": normalized,
        "unresolved": [*unresolved, *missing_unresolved],
        "missing": missing,
        "seen": seen,
        "candidate_sha256": candidate_sha256(package),
    }


@dataclass
class ApplyResult:
    package: dict
    document: UnifiedDocument
    applied: int
    unresolved: list[dict]
    skipped_unresolved: int
    candidate_sha256_before: str
    candidate_sha256_after: str


def apply_decisions(
    package: dict,
    decisions: dict,
    *,
    allowed_row_ids: Iterable[str] | None = None,
    require_complete: bool = False,
    current_primary: UnifiedDocument | None = None,
) -> ApplyResult:
    """Apply only resolved decision text, then run the canonical package importer."""
    validated = validate_decisions(
        package, decisions,
        allowed_row_ids=allowed_row_ids,
        require_complete=require_complete,
    )
    before = candidate_sha256(package)
    reviewed = copy.deepcopy(package)
    item_by_id = {str(item.get("row_id", "") or ""): item for item in reviewed.get("editable_items") or []}
    applied = 0
    skipped_unresolved = 0
    for decision in validated["normalized"]:
        if decision["needs_manual_review"]:
            skipped_unresolved += 1
            continue
        item = item_by_id[decision["row_id"]]
        item["edited_text"] = decision["corrected_text"]
        item["delete_intentionally"] = decision["delete_intentionally"]
        applied += 1
    after = candidate_sha256(reviewed)
    if before != after:
        raise HeadlessAdjudicationError("原始 OCR candidate 哈希发生变化，已中止写回。")
    try:
        document, _comparison, _lines = import_multi_package(reviewed, current_primary=current_primary)
    except (RoundtripPackageError, ValueError) as exc:
        raise HeadlessAdjudicationError(f"裁决写回后 canonical import 验证失败：{exc}") from exc
    return ApplyResult(
        package=reviewed,
        document=document,
        applied=applied,
        unresolved=list(validated["unresolved"]),
        skipped_unresolved=skipped_unresolved,
        candidate_sha256_before=before,
        candidate_sha256_after=after,
    )


def save_document(document: UnifiedDocument, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(document.to_json(indent=2) + "\n", encoding="utf-8")
    return target


def save_unresolved(package: dict, unresolved: list[dict], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": UNRESOLVED_SCHEMA,
        "source": _package_binding(package),
        "count": len(unresolved),
        "rows": unresolved,
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def export_skeleton_epub(
    package: dict,
    output_path: str | Path,
    *,
    vertical: bool | None = None,
    current_primary: UnifiedDocument | None = None,
) -> dict:
    """Validate the reviewed package and build a lean stable-ID skeleton EPUB.

    The headless CLI only needs a skeleton EPUB.  It must not spend minutes
    generating the full AI-repair exchange payload (chapter evidence JSON, task
    metadata, etc.).  We therefore reuse the same stable-row document builder
    and the normal EPUB builder directly, then verify that every OCR stable row
    is present exactly once.
    """
    _validate_package_shape(package)
    try:
        fused, _comparison, _lines = import_multi_package(package, current_primary=current_primary)
    except (RoundtripPackageError, ValueError) as exc:
        raise HeadlessAdjudicationError(f"导出 skeleton 前 package 验证失败：{exc}") from exc

    primary = (
        UnifiedDocument.from_dict(copy.deepcopy(current_primary.to_dict()))
        if current_primary is not None
        else UnifiedDocument.from_dict(copy.deepcopy(package.get("structure_document") or {}))
    )
    if vertical is None:
        vertical = str(getattr(primary.metadata, "writing_direction", "") or "").startswith("vertical")

    from engine.ai_repair_epub import build_repair_document
    from builder.epub_builder import build_epub
    import zipfile
    import xml.etree.ElementTree as ET

    output = Path(output_path).expanduser()
    if output.suffix.lower() != ".epub":
        output = output.with_suffix(".epub")
    output.parent.mkdir(parents=True, exist_ok=True)

    export_revision = int(package.get("export_revision", 0) or 0) or 1
    repair_doc = build_repair_document(primary, package, export_revision=export_revision)
    build_epub(
        repair_doc,
        str(output),
        css_template="denki",
        vertical=bool(vertical),
        verbose=False,
        preserve_image_bytes=True,
    )

    expected_ids = {
        str(item.get("row_id") or item.get("item_id") or "")
        for item in (package.get("editable_items") or [])
    }
    expected_ids.discard("")
    found_ids: list[str] = []
    xhtml_count = 0
    image_count = 0
    try:
        with zipfile.ZipFile(output, "r") as archive:
            names = archive.namelist()
            if not names or names[0] != "mimetype":
                raise HeadlessAdjudicationError("skeleton EPUB 的 mimetype 不是首项。")
            if archive.getinfo("mimetype").compress_type != zipfile.ZIP_STORED:
                raise HeadlessAdjudicationError("skeleton EPUB 的 mimetype 必须不压缩。")
            bad = archive.testzip()
            if bad is not None:
                raise HeadlessAdjudicationError(f"skeleton EPUB CRC 失败：{bad}")
            for name in names:
                lowered = name.lower()
                if lowered.endswith(".xhtml"):
                    xhtml_count += 1
                    try:
                        root = ET.fromstring(archive.read(name))
                    except Exception as exc:
                        raise HeadlessAdjudicationError(f"skeleton XHTML 无法解析：{name}: {exc}") from exc
                    for element in root.iter():
                        stable_id = str(element.attrib.get("data-item-id", "") or "")
                        if stable_id:
                            found_ids.append(stable_id)
                elif lowered.endswith((".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg")):
                    image_count += 1
    except zipfile.BadZipFile as exc:
        raise HeadlessAdjudicationError(f"skeleton EPUB 不是有效 ZIP：{exc}") from exc

    found_set = set(found_ids)
    duplicates = len(found_ids) - len(found_set)
    missing = expected_ids - found_set
    extra = found_set - expected_ids
    if duplicates or missing or extra:
        raise HeadlessAdjudicationError(
            "skeleton stable row 映射不完整："
            f"missing={len(missing)} / extra={len(extra)} / duplicates={duplicates}"
        )

    return {
        "path": str(output),
        "output_path": str(output),
        "mode": "skeleton",
        "editable_count": len(expected_ids),
        "image_count": image_count,
        "xhtml_count": xhtml_count,
        "stable_id_count": len(found_ids),
        "fused_block_count": len(fused.blocks),
        "candidate_sha256": candidate_sha256(package),
        "zip_crc_passed": True,
        "mimetype_first": True,
        "mimetype_uncompressed": True,
    }

def run_from_decisions(
    package_path: str | Path,
    decisions_path: str | Path,
    output_dir: str | Path,
    *,
    scope: str = "conflicts",
    include_existing_edits: bool = False,
    require_complete: bool = False,
    vertical: bool | None = None,
    skeleton_template: str | Path | None = None,
) -> dict:
    package = load_multi_package(package_path)
    task = build_adjudication_task(
        package,
        scope=scope,
        include_existing_edits=include_existing_edits,
    )
    allowed = [str(row.get("row_id", "") or "") for row in task.get("rows") or []]
    decisions = load_decisions(decisions_path)
    current_primary = None
    if skeleton_template:
        from engine.headless_workspace_source import materialize_skeleton_template_assets
        current_primary = materialize_skeleton_template_assets(
            package, skeleton_template, Path(output_dir) / "publication_assets"
        )
    result = apply_decisions(
        package,
        decisions,
        allowed_row_ids=allowed,
        require_complete=require_complete,
        current_primary=current_primary,
    )
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    reviewed_path = save_package(result.package, folder / "reviewed_roundtrip.json")
    document_path = save_document(result.document, folder / "fused_document.json")
    unresolved_path = save_unresolved(result.package, result.unresolved, folder / "unresolved.json")
    skeleton_path = folder / "structure_skeleton.epub"
    epub_report = export_skeleton_epub(
        result.package, skeleton_path, vertical=vertical, current_primary=current_primary
    )
    report = {
        "schema": REPORT_SCHEMA,
        "source_package": str(Path(package_path)),
        "decisions": str(Path(decisions_path)),
        "scope": scope,
        "selected_rows": len(allowed),
        "applied_decisions": result.applied,
        "unresolved_rows": len(result.unresolved),
        "candidate_sha256_before": result.candidate_sha256_before,
        "candidate_sha256_after": result.candidate_sha256_after,
        "raw_candidates_unchanged": result.candidate_sha256_before == result.candidate_sha256_after,
        "reviewed_roundtrip": str(reviewed_path),
        "fused_document": str(document_path),
        "unresolved": str(unresolved_path),
        "skeleton_epub": str(skeleton_path),
        "epub": epub_report,
    }
    (folder / "headless_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
