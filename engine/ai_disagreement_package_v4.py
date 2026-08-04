#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI repair package v4: adjudicate only genuine multi-model disagreements.

The v4 package deliberately separates three responsibilities:

* OCR engines provide immutable raw candidates.
* deterministic code classifies consensus/disagreement and freezes consensus text.
* an external model may return decisions only for explicitly unlocked item IDs.

EPUB structure is never delegated to the model.  A row-addressable skeleton EPUB
is exported and the bundled standalone tool applies validated decisions by
``data-item-id`` before removing all AI work metadata.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import stat
import textwrap
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from engine import ai_repair_epub as repair
from engine.ocr_unicode_standardizer import (
    japanese_ocr_comparison_key,
    normalize_japanese_ocr_text,
)
from engine.adaptive_ocr_ensemble import (
    ModelReliability, decide_ensemble, estimate_model_reliability, model_family,
)
from models.document import UnifiedDocument

PACKAGE_SCHEMA = "novel_formatter.ai_disagreement_package.v4"
DECISIONS_SCHEMA = "novel_formatter.ai_disagreement_decisions.v1"
SUMMARY_SCHEMA = "novel_formatter.ai_disagreement_summary.v1"
ITEM_INDEX_SCHEMA = "novel_formatter.ai_disagreement_item_index.v1"
CONFLICT_SCHEMA = "novel_formatter.ai_disagreement_conflict_item.v1"
TERMS_SCHEMA = "novel_formatter.ai_disagreement_terms.v1"

PLACEHOLDERS = {"□", "■", "�", "\x00"}
STATUS_TOKENS = ("レベル", "LEVEL", "LV", "技能", "スキル", "装備", "称号", "HP", "MP", "職業")
JAPANESE_RE = re.compile(r"[ぁ-ゟ゠-ヿ一-龯々〆ヵヶ]", re.UNICODE)
JAPANESE_OR_PUNCT_RE = re.compile(r"[ぁ-ゟ゠-ヿ一-龯々〆ヵヶ、。！？：；（）「」『』【】〈〉《》]", re.UNICODE)
JAPANESE_SPACE_RE = re.compile(
    r"(?<=[ぁ-ゟ゠-ヿ一-龯々〆ヵヶ、。！？：；（）「」『』【】〈〉《》])"
    r"[ \t\u3000]+"
    r"(?=[ぁ-ゟ゠-ヿ一-龯々〆ヵヶ、。！？：；（）「」『』【】〈〉《》])"
)
ASCII_JP_PUNCT = str.maketrans({
    "?": "？", "!": "！", ":": "：", ";": "；",
    "(": "（", ")": "）", "[": "［", "]": "］", "{": "｛", "}": "｝",
})
TERM_RE = re.compile(
    r"(?:[ァ-ヶー・]{3,32}|[一-龯々〆ヵヶ]{2,10}(?:・[一-龯々〆ヵヶ]{1,10})?|"
    r"[A-Za-z][A-Za-z0-9_./:+-]{1,30}|(?:レベル|Lv\.?|LEVEL)\s*[0-9０-９]+)",
    re.I,
)
SENSITIVE_NUMBER_RE = re.compile(r"[0-9０-９]+(?:[.,．，][0-9０-９]+)?")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(str(text or "").encode("utf-8"))


def _json_bytes(value: Any, *, pretty: bool = True) -> bytes:
    if pretty:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False).encode("utf-8")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=False).encode("utf-8")


def _safe_name(value: str, default: str = "AI修复包") -> str:
    return repair._safe_publication_filename(value, default=default)


def _item_id(item: dict, index: int = 0) -> str:
    value = str(item.get("row_id") or item.get("item_id") or "").strip()
    return value or f"item-{index + 1:06d}"


def _candidate_model(candidate: dict, index: int) -> str:
    return str(
        candidate.get("model_label")
        or candidate.get("model")
        or candidate.get("source_engine")
        or f"model_{index + 1}"
    ).strip()


def _candidate_text(candidate: dict) -> str:
    return str(candidate.get("raw_text", candidate.get("text", "")) or "")


def _is_placeholder(text: str) -> bool:
    value = str(text or "")
    stripped = re.sub(r"\s+", "", value)
    return bool(stripped) and all(char in PLACEHOLDERS for char in stripped)


def _usable_text(text: str) -> bool:
    value = str(text or "").strip()
    return bool(value) and not _is_placeholder(value)


def canonicalize_standard_japanese(text: str) -> tuple[str, list[str]]:
    """Return a conservative standard-Japanese copy plus event labels.

    Raw model output is never overwritten.  This function only composes Unicode
    Japanese characters, normalises line endings, removes layout-only spaces
    between Japanese glyphs, and converts ASCII punctuation when it appears in
    Japanese prose.  Ideographs, kana choice, digits, dashes and vocabulary are
    never rewritten.
    """
    source = str(text or "")
    value, report = normalize_japanese_ocr_text(source)
    events = list(report.counts)
    newline = value.replace("\r\n", "\n").replace("\r", "\n")
    if newline != value:
        events.append("line_endings")
        value = newline
    if JAPANESE_RE.search(value):
        converted = value.translate(ASCII_JP_PUNCT)
        if converted != value:
            events.append("ascii_japanese_punctuation")
            value = converted
        compact = JAPANESE_SPACE_RE.sub("", value)
        if compact != value:
            events.append("layout_space_between_japanese")
            value = compact
    return value, sorted(set(events))


def comparison_key_standard_japanese(text: str) -> str:
    canonical, _events = canonicalize_standard_japanese(text)
    key, _report = japanese_ocr_comparison_key(canonical)
    # Whitespace is layout evidence, not textual disagreement, in physical
    # Japanese columns.  It remains preserved in raw/canonical text.
    return re.sub(r"\s+", "", key)


def _risk_flags(texts: Sequence[str]) -> list[str]:
    joined = "\n".join(str(value or "") for value in texts)
    flags: list[str] = []
    if SENSITIVE_NUMBER_RE.search(joined):
        flags.append("numeric_or_level_content")
    if any(token.casefold() in joined.casefold() for token in STATUS_TOKENS):
        flags.append("status_or_structured_content")
    if re.search(r"[A-Za-z]", joined):
        flags.append("latin_identifier_or_rank")
    if any(char in joined for char in ("□", "■", "�")):
        flags.append("placeholder_candidate")
    if re.search(r"[「『][^」』]{0,240}$", joined):
        flags.append("possibly_unbalanced_quote")
    return flags


def _candidate_health(items: Sequence[dict]) -> dict[str, dict]:
    """Return availability plus book-local agreement calibration per model."""
    labels: list[str] = []
    totals: Counter[str] = Counter()
    usable: Counter[str] = Counter()
    placeholders: Counter[str] = Counter()
    row_maps: list[dict[str, str]] = []
    for item in items:
        row: dict[str, str] = {}
        for index, candidate in enumerate(item.get("candidates") or []):
            if not isinstance(candidate, dict):
                continue
            model = _candidate_model(candidate, index)
            if model not in labels:
                labels.append(model)
            totals[model] += 1
            text = _candidate_text(candidate)
            row[model] = text
            if _usable_text(text):
                usable[model] += 1
            elif _is_placeholder(text):
                placeholders[model] += 1
        row_maps.append(row)

    reliability_rows = [[row.get(label, "") for label in labels] for row in row_maps]
    calibrated = estimate_model_reliability(reliability_rows, labels)
    result: dict[str, dict] = {}
    for model in labels:
        total = totals[model]
        ratio = usable[model] / total if total else 0.0
        rel = calibrated.get(model, ModelReliability(label=model, family=model_family(model)))
        result[model] = {
            "model": model,
            "family": rel.family,
            "candidate_count": total,
            "usable_count": usable[model],
            "placeholder_count": placeholders[model],
            "usable_ratio": round(ratio, 6),
            "anchor_count": rel.anchor_count,
            "anchor_correct": rel.anchor_correct,
            "anchor_accuracy": round(rel.anchor_accuracy, 6),
            "anchor_wilson_lower": round(rel.wilson_lower, 6),
            "reliability": round(rel.reliability, 6),
            "voting_enabled": bool(rel.voting_enabled),
            "health_status": (
                "healthy_calibrated" if rel.voting_enabled
                else "unhealthy_excluded_from_vote"
            ),
            "health_reason": rel.reason,
        }
    return result


def _context_text(items: Sequence[dict], index: int, direction: int) -> str:
    cursor = index + direction
    collected: list[str] = []
    while 0 <= cursor < len(items) and len("".join(collected)) < 320:
        text = str(items[cursor].get("edited_text", items[cursor].get("original_fused_text", "")) or "").strip()
        if text:
            if direction < 0:
                collected.insert(0, text)
            else:
                collected.append(text)
        if len(collected) >= 2:
            break
        cursor += direction
    return "\n".join(collected)[-320:] if direction < 0 else "\n".join(collected)[:320]


def build_disagreement_records(package: dict) -> tuple[list[dict], dict]:
    """Classify every editable item and return records plus summary."""
    items = [item for item in (package.get("editable_items") or []) if isinstance(item, dict)]
    health = _candidate_health(items)
    records: list[dict] = []
    status_counts: Counter[str] = Counter()
    priority_counts: Counter[str] = Counter()

    for index, item in enumerate(items):
        candidates: list[dict] = []
        voting: list[dict] = []
        for candidate_index, candidate in enumerate(item.get("candidates") or []):
            if not isinstance(candidate, dict):
                continue
            model = _candidate_model(candidate, candidate_index)
            raw = _candidate_text(candidate)
            canonical, events = canonicalize_standard_japanese(raw)
            key = comparison_key_standard_japanese(canonical) if _usable_text(raw) else ""
            candidate_record = {
                "model_index": int(candidate.get("model_index", candidate_index) or candidate_index),
                "model": model,
                "source_engine": str(candidate.get("source_engine", "") or ""),
                "raw_text": raw,
                "canonical_text": canonical,
                "compare_key": key,
                "confidence": float(candidate.get("confidence", 0.0) or 0.0),
                "raw_text_sha256": _sha256_text(raw),
                "canonical_text_sha256": _sha256_text(canonical),
                "normalization_events": events,
                "placeholder": _is_placeholder(raw),
                "usable": _usable_text(raw),
                "eligible_for_vote": bool(health.get(model, {}).get("voting_enabled", True)) and _usable_text(raw),
            }
            candidates.append(candidate_record)
            if candidate_record["eligible_for_vote"]:
                voting.append(candidate_record)

        reliability_map = {
            model: ModelReliability(
                label=model,
                family=str(info.get("family") or model_family(model)),
                usable_ratio=float(info.get("usable_ratio", 1.0) or 0.0),
                anchor_count=int(info.get("anchor_count", 0) or 0),
                anchor_correct=int(info.get("anchor_correct", 0) or 0),
                anchor_accuracy=float(info.get("anchor_accuracy", 1.0) or 0.0),
                wilson_lower=float(info.get("anchor_wilson_lower", 0.5) or 0.0),
                reliability=float(info.get("reliability", 1.0) or 1.0),
                voting_enabled=bool(info.get("voting_enabled", True)),
                reason=str(info.get("health_reason", "") or ""),
            )
            for model, info in health.items()
        }
        ensemble = decide_ensemble(
            [candidate["raw_text"] for candidate in candidates],
            [candidate["model"] for candidate in candidates],
            [candidate["confidence"] for candidate in candidates],
            reliability_map,
            verify_sensitive_two_model_agreement=True,
        )
        expected_models = len(health) or len(candidates)
        usable_models = len(voting)
        ignored_models = [c["model"] for c in candidates if c["usable"] and not c["eligible_for_vote"]]
        missing_models = [c["model"] for c in candidates if not c["usable"]]

        accepted_text: str | None = None
        status = "missing_candidate"
        action_required = True
        priority = "high"
        reason_codes: list[str] = []

        if ensemble.status in {"exact_consensus", "normalized_consensus"}:
            accepted_text = ensemble.chosen_text
            status = ensemble.status
            action_required = bool(ensemble.requires_more_models or ensemble.requires_review)
            priority = "critical" if ensemble.sensitive and action_required else ("high" if action_required else "none")
            reason_codes.append(ensemble.status)
            if action_required:
                reason_codes.append("sensitive_two_model_agreement_requires_independent_check")
            if missing_models:
                reason_codes.append("other_model_missing_but_reliable_models_agree")
            if ignored_models:
                reason_codes.append("unhealthy_model_excluded_but_reliable_models_agree")
        elif ensemble.status == "majority_consensus":
            accepted_text = ensemble.chosen_text
            status = "majority_consensus"
            action_required = True
            priority = "critical" if ensemble.sensitive else "medium"
            reason_codes.append("minority_candidate_disagrees")
        elif ensemble.status in {"single_candidate", "no_usable_candidate"}:
            status = "missing_candidate"
            action_required = True
            priority = "high"
            reason_codes.append("fewer_than_two_reliable_candidates")
        else:
            status = "full_conflict"
            action_required = True
            priority = "critical" if ensemble.sensitive else "high"
            reason_codes.append("no_independent_model_majority")

        risk_flags = _risk_flags([candidate["raw_text"] for candidate in candidates])
        if action_required and risk_flags:
            priority = "critical" if any(flag in risk_flags for flag in ("numeric_or_level_content", "status_or_structured_content")) else priority
        provisional = accepted_text or str(item.get("edited_text", item.get("original_fused_text", "")) or "")
        canonical_provisional, provisional_events = canonicalize_standard_japanese(provisional)
        record = {
            "schema": ITEM_INDEX_SCHEMA,
            "item_id": _item_id(item, index),
            "row_index": int(item.get("row_index", index) or index),
            "page": int(item.get("page", 0) or 0),
            "block_type": str(item.get("block_type", item.get("type", "paragraph")) or "paragraph"),
            "column_ids": [str(value) for value in (item.get("column_ids") or []) if str(value)],
            "status": status,
            "accepted_text": accepted_text,
            "provisional_text": canonical_provisional,
            "provisional_text_sha256": _sha256_text(canonical_provisional),
            "model_action_required": action_required,
            "review_priority": priority,
            "reason_codes": reason_codes,
            "risk_flags": risk_flags,
            "ensemble_confidence": round(float(ensemble.confidence or 0.0), 6),
            "ensemble_score_margin": round(float(ensemble.score_margin or 0.0), 6),
            "ensemble_family_support_count": int(ensemble.family_support_count or 0),
            "ensemble_reason": str(ensemble.reason or ""),
            "ensemble_warnings": list(ensemble.warnings or ()),
            "usable_voting_candidate_count": usable_models,
            "expected_model_count": expected_models,
            "missing_models": sorted(set(missing_models)),
            "ignored_unhealthy_models": sorted(set(ignored_models)),
            "candidates": candidates,
            "context_before": _context_text(items, index, -1),
            "context_after": _context_text(items, index, 1),
            "source_bbox": item.get("bbox") or item.get("source_bbox") or [],
            "normalization_events": provisional_events,
            "decision_contract": {
                "editable": action_required,
                "allowed_fields": ["item_id", "selected_text", "source", "confidence", "reason_code", "evidence"],
                "must_not_move_or_merge_item": True,
            },
        }
        records.append(record)
        status_counts[status] += 1
        priority_counts[priority] += 1

    frozen = [record for record in records if not record["model_action_required"]]
    conflicts = [record for record in records if record["model_action_required"]]
    summary = {
        "schema": SUMMARY_SCHEMA,
        "package_schema": PACKAGE_SCHEMA,
        "total_items": len(records),
        "frozen_item_count": len(frozen),
        "model_action_required_count": len(conflicts),
        "status_counts": dict(status_counts),
        "priority_counts": dict(priority_counts),
        "model_health": health,
        "frozen_item_ids_sha256": _sha256_text("\n".join(record["item_id"] for record in frozen)),
        "editable_item_ids_sha256": _sha256_text("\n".join(record["item_id"] for record in conflicts)),
        "classification_policy": {
            "exact_or_normalized_two_model_agreement_with_failed_model": "freeze",
            "healthy_model_majority_with_real_minority_text": "external_adjudication",
            "all_reliable_candidates_disagree": "external_adjudication",
            "fewer_than_two_reliable_candidates": "external_adjudication",
            "globally_unhealthy_model_threshold": "at least 50 candidates and usable ratio below 0.35",
            "book_local_reliability_gate": "exclude when at least 30 independent anchors and anchor accuracy below 0.55",
            "correlated_family_vote": "second recognizer from the same OCR family contributes only 0.35 vote",
            "sensitive_two_model_agreement": "requires independent verification before freezing",
        },
    }
    return records, summary


def _structure_page_paths(package: dict) -> dict[int, str]:
    structure = package.get("structure_document") if isinstance(package.get("structure_document"), dict) else {}
    output: dict[int, str] = {}
    for index, page in enumerate(structure.get("pages") or []):
        if not isinstance(page, dict):
            continue
        page_no = int(page.get("page_no", page.get("page", index + 1)) or index + 1)
        path = str(page.get("image_path", "") or "")
        if path:
            output[page_no] = path
    return output


def _bbox_pixels(record: dict, width: int, height: int) -> tuple[int, int, int, int] | None:
    bbox = record.get("source_bbox") or []
    if isinstance(bbox, dict):
        bbox = [bbox.get("x", 0), bbox.get("y", 0), bbox.get("w", 0), bbox.get("h", 0)]
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        # Candidate geometry is often more reliable than the fused-item bbox.
        for candidate in record.get("candidates") or []:
            raw = candidate.get("bbox") if isinstance(candidate, dict) else None
            if isinstance(raw, (list, tuple)) and len(raw) >= 4:
                bbox = raw
                break
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x, y, w, h = [float(value) for value in bbox[:4]]
    except (TypeError, ValueError):
        return None
    if max(abs(x), abs(y), abs(w), abs(h)) <= 2.0:
        x, y, w, h = x * width, y * height, w * width, h * height
    margin = max(2, int(round(min(width, height) * 0.003)))
    left = max(0, int(x) - margin)
    top = max(0, int(y) - margin)
    right = min(width, int(x + w + 0.999) + margin)
    bottom = min(height, int(y + h + 0.999) + margin)
    if right - left < 2 or bottom - top < 2:
        return None
    return left, top, right, bottom


def _export_conflict_crops(folder: Path, package: dict, records: Sequence[dict]) -> tuple[dict[str, str], list[dict]]:
    output: dict[str, str] = {}
    omissions: list[dict] = []
    page_paths = _structure_page_paths(package)
    try:
        from PIL import Image
    except Exception:
        return output, [{"reason": "pillow_unavailable", "item_count": sum(1 for r in records if r["model_action_required"])}]
    evidence_dir = folder / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        if not record["model_action_required"]:
            continue
        item_id = record["item_id"]
        page = int(record.get("page", 0) or 0)
        image_path = Path(page_paths.get(page, "")).expanduser()
        if not image_path.is_file():
            omissions.append({"item_id": item_id, "page": page, "reason": "source_page_image_unavailable"})
            continue
        try:
            with Image.open(image_path) as image:
                image.load()
                box = _bbox_pixels(record, image.width, image.height)
                if box is None:
                    omissions.append({"item_id": item_id, "page": page, "reason": "bbox_unavailable"})
                    continue
                crop = image.crop(box).convert("L")
                filename = f"{record['row_index']:06d}_{hashlib.sha1(item_id.encode('utf-8')).hexdigest()[:10]}.png"
                target = evidence_dir / filename
                crop.save(target, format="PNG", optimize=True, compress_level=9)
                output[item_id] = f"evidence/{filename}"
        except Exception as exc:
            omissions.append({"item_id": item_id, "page": page, "reason": f"crop_failed:{type(exc).__name__}"})
    return output, omissions


def _terms_payload(records: Sequence[dict]) -> dict:
    counts: Counter[str] = Counter()
    disagreement: Counter[str] = Counter()
    for record in records:
        texts = [str(record.get("provisional_text", "") or "")]
        texts.extend(str(candidate.get("canonical_text", "") or "") for candidate in record.get("candidates") or [])
        found = set()
        for text in texts:
            found.update(match.group(0) for match in TERM_RE.finditer(text))
        for term in found:
            counts[term] += 1
            if record["model_action_required"]:
                disagreement[term] += 1
    terms = [
        {"term": term, "item_count": count, "conflict_item_count": disagreement[term]}
        for term, count in counts.most_common(1500)
    ]
    return {"schema": TERMS_SCHEMA, "term_count": len(terms), "terms": terms}


def _instructions() -> str:
    return textwrap.dedent(f"""\
    # AI 修复包 V4：多模型分歧裁决

    ## 唯一任务

    只处理 `03_conflict_items.jsonl` 中 `model_action_required=true` 的条目。
    `exact_consensus` 与 `normalized_consensus` 已冻结，禁止改写、润色、移动、合并或拆分。

    ## 工作顺序

    1. 阅读 `02_consensus_summary.json`，确认模型健康状态。
    2. 逐条查看 `03_conflict_items.jsonl` 的原始候选、标准日文副本、前后文和可用裁切图。
    3. 创建 `decisions.json`，schema 必须是 `{DECISIONS_SCHEMA}`。
    4. 每个冲突 ID 恰好给出一次决定；不得返回冻结 ID。
    5. 运行：

       `python3 tools/apply_decisions.py --package . --decisions decisions.json --output 最终出版版.epub --audit final_audit.json`

    6. 再运行：

       `python3 tools/validate_epub.py --package . --epub 最终出版版.epub --audit final_validation.json`

    ## 决策格式

    ```json
    {{
      "schema": "{DECISIONS_SCHEMA}",
      "package_id": "从 01_manifest.json 复制",
      "structure_sha256": "从 01_manifest.json 复制",
      "decisions": [
        {{
          "item_id": "row:...",
          "selected_text": "最终日文",
          "source": "candidate|corrected_from_image|contextual_reconstruction",
          "confidence": 0.98,
          "reason_code": "glyph_and_context_support",
          "evidence": ["evidence/....png"]
        }}
      ]
    }}
    ```

    ## 硬性禁止

    - 不得修改骨架 EPUB 的章节、NAV、spine、图片、CSS、元数据和锚点。
    - 不得凭语言习惯改写一致正文。
    - 不得把别处句子替换进当前条目。
    - 数字、等级、人名、技能名和状态栏冲突必须结合图像证据判断。
    - 无可靠 Ruby 证据时只保留底字，不猜读音。
    """)


def _standalone_apply_tool() -> str:
    # Kept self-contained so a model/runtime does not need Novel Formatter.
    return r'''#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, hashlib, json, re, zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

DECISIONS_SCHEMA = "novel_formatter.ai_disagreement_decisions.v1"
AI_PREFIXES = ("META-INF/ai-repair/", "META-INF/ai-publication/")

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def local(tag):
    return tag.rsplit("}", 1)[-1]

def text_sha(value):
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()

def set_text(element, value):
    for child in list(element):
        element.remove(child)
    element.text = None
    parts = str(value or "").split("\n")
    cursor = element
    for i, part in enumerate(parts):
        if i:
            br = ET.SubElement(element, "{http://www.w3.org/1999/xhtml}br")
            br.tail = part
            cursor = br
        elif part:
            element.text = part

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default=".")
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--audit", required=True)
    args = ap.parse_args()
    root = Path(args.package)
    manifest = load_json(root / "01_manifest.json")
    decisions = load_json(args.decisions)
    if decisions.get("schema") != DECISIONS_SCHEMA:
        raise SystemExit("decisions schema 不正确")
    if decisions.get("package_id") != manifest.get("package_id") or decisions.get("structure_sha256") != manifest.get("structure_sha256"):
        raise SystemExit("decisions 不属于当前包")
    all_items = {}
    with (root / "04_all_items_index.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                item = json.loads(line); all_items[item["item_id"]] = item
    editable = {k for k,v in all_items.items() if v.get("model_action_required")}
    frozen = set(all_items) - editable
    rows = decisions.get("decisions")
    if not isinstance(rows, list):
        raise SystemExit("decisions 必须是数组")
    by_id = {}
    allowed = {"item_id","selected_text","source","confidence","reason_code","evidence"}
    for row in rows:
        if not isinstance(row, dict) or set(row) - allowed:
            raise SystemExit("decision 包含未知字段")
        item_id = str(row.get("item_id", ""))
        if item_id in frozen: raise SystemExit(f"冻结条目禁止修改：{item_id}")
        if item_id not in editable: raise SystemExit(f"未知冲突 ID：{item_id}")
        if item_id in by_id: raise SystemExit(f"重复决定：{item_id}")
        selected = str(row.get("selected_text", ""))
        if not selected.strip(): raise SystemExit(f"空决定：{item_id}")
        by_id[item_id] = row
    missing = sorted(editable - set(by_id))
    if missing: raise SystemExit(f"尚有 {len(missing)} 个冲突未裁决，首项：{missing[0]}")
    skeleton = root / manifest["paths"]["skeleton_epub"]
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    changed = []; found = set(); entries = []
    with zipfile.ZipFile(skeleton) as zin:
        for info in zin.infolist():
            if info.filename.startswith(AI_PREFIXES): continue
            data = zin.read(info.filename)
            if info.filename.lower().endswith((".xhtml", ".html", ".htm")):
                try: tree = ET.fromstring(data)
                except ET.ParseError: entries.append((info, data)); continue
                touched = False
                for elem in tree.iter():
                    item_id = elem.attrib.get("data-item-id", "")
                    if item_id:
                        found.add(item_id)
                        if item_id in by_id:
                            selected = str(by_id[item_id]["selected_text"])
                            set_text(elem, selected); changed.append({"item_id":item_id,"text_sha256":text_sha(selected)}); touched = True
                        for key in list(elem.attrib):
                            if local(key) in {"data-item-id","data-row-id","data-block-id","data-delete-intentionally"}:
                                del elem.attrib[key]; touched = True
                if touched: data = ET.tostring(tree, encoding="utf-8", xml_declaration=True)
            entries.append((info, data))
    if editable - found: raise SystemExit(f"骨架缺少 {len(editable-found)} 个冲突锚点")
    with zipfile.ZipFile(output, "w") as zout:
        mime = next((pair for pair in entries if pair[0].filename == "mimetype"), None)
        if mime: zout.writestr("mimetype", mime[1], compress_type=zipfile.ZIP_STORED)
        for info, data in entries:
            if info.filename == "mimetype": continue
            zi = copy.copy(info); zi.compress_type = zipfile.ZIP_DEFLATED
            zout.writestr(zi, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    audit = {"schema":"novel_formatter.ai_disagreement_apply_audit.v1","package_id":manifest["package_id"],"output":str(output),"decision_count":len(by_id),"changed":changed,"frozen_item_count":len(frozen),"unresolved_count":0}
    Path(args.audit).write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
if __name__ == "__main__": main()
'''


def _standalone_validate_tool() -> str:
    return r'''#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--package",default="."); ap.add_argument("--epub",required=True); ap.add_argument("--audit",required=True); args=ap.parse_args()
    path=Path(args.epub); errors=[]; stats={}
    try:
        with zipfile.ZipFile(path) as z:
            names=z.namelist(); stats["member_count"]=len(names)
            if not names or names[0] != "mimetype": errors.append("mimetype_not_first")
            elif z.getinfo("mimetype").compress_type != zipfile.ZIP_STORED: errors.append("mimetype_compressed")
            bad=z.testzip()
            if bad: errors.append(f"crc:{bad}")
            for name in names:
                if name.startswith(("META-INF/ai-repair/","META-INF/ai-publication/")): errors.append(f"ai_work_payload:{name}")
                if name.lower().endswith((".xhtml",".html",".htm",".opf",".ncx")):
                    data=z.read(name)
                    if b"data-item-id" in data or b"data-row-id" in data or b"data-block-id" in data: errors.append(f"work_attribute:{name}")
                    try: ET.fromstring(data)
                    except ET.ParseError as exc: errors.append(f"xml:{name}:{exc}")
                    if name.lower().endswith((".xhtml",".html",".htm")) and any(token in data for token in ("□".encode(),"�".encode())): errors.append(f"placeholder:{name}")
    except Exception as exc: errors.append(f"open:{exc}")
    audit={"schema":"novel_formatter.ai_disagreement_final_validation.v1","epub":str(path),"passed":not errors,"errors":errors,"statistics":stats}
    Path(args.audit).write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(audit,ensure_ascii=False,indent=2))
    raise SystemExit(0 if not errors else 2)
if __name__=="__main__": main()
'''


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _zip_folder(folder: Path, target: Path) -> None:
    with zipfile.ZipFile(target, "w") as archive:
        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(folder).as_posix()
            suffix = path.suffix.lower()
            if suffix in {".png", ".jpg", ".jpeg", ".webp", ".epub", ".gif"}:
                archive.write(path, relative, compress_type=zipfile.ZIP_STORED)
            else:
                archive.write(path, relative, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def export_ai_disagreement_package_v4(
    primary_doc: UnifiedDocument,
    package: dict,
    output_directory: str | Path,
    *,
    vertical: bool = True,
    css_template: str = "denki",
    custom_css: str | None = None,
    bundle_name: str | None = None,
    create_zip: bool = True,
    include_publication_reference: bool = False,
    publication_reference_path: str | Path | None = None,
) -> dict:
    """Export the compact disagreement-only adjudication package."""
    records, summary = build_disagreement_records(package)
    root = Path(output_directory).expanduser(); root.mkdir(parents=True, exist_ok=True)
    base = _safe_name(str(bundle_name or "AI修复包").strip() or "AI修复包")
    folder = root / (f"{base}_V4_多模型分歧裁决包" if not base.endswith("裁决包") else base)
    if folder.exists():
        folder = root / f"{folder.name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    folder.mkdir(parents=True)
    (folder / "source").mkdir(); (folder / "framework").mkdir(); (folder / "tools").mkdir()

    try:
        skeleton_package = copy.deepcopy(package)
        by_id = {record["item_id"]: record for record in records}
        for index, item in enumerate(skeleton_package.get("editable_items") or []):
            record = by_id.get(_item_id(item, index))
            if record:
                item["edited_text"] = record["provisional_text"]
                item["original_fused_text"] = str(item.get("original_fused_text", item["edited_text"]) or item["edited_text"])
        skeleton_path = folder / "framework" / "structure_skeleton.epub"
        # Reuse the mature V3 structure builder (chapter recovery, independent
        # illustration pages, NAV/spine and resource-byte preservation), then
        # discard its verbose evidence package.  V4 changes only the text
        # adjudication contract, not the proven EPUB structure pipeline.
        structure_temp = folder / ".structure_build"
        structure_temp.mkdir()
        try:
            from engine.ai_publication_bundle_v2 import export_ai_publication_bundle_v2
            structure_report = export_ai_publication_bundle_v2(
                primary_doc,
                skeleton_package,
                structure_temp,
                mode="one_pass",
                vertical=vertical,
                css_template=css_template,
                custom_css=custom_css,
                bundle_name="V4结构骨架临时构建",
                create_zip=False,
                include_publication_reference=False,
                publication_reference_path=None,
                package_mode="compact",
            )
            source_framework = Path(str(structure_report.get("primary_framework_epub") or structure_report.get("framework_epub") or ""))
            if not source_framework.is_file():
                raise repair.AIRepairEpubError("V3 结构构建器没有生成框架 EPUB。")
            shutil.copy2(source_framework, skeleton_path)
            skeleton_report = {
                "path": str(skeleton_path),
                "editable_count": int(structure_report.get("editable_count", len(records)) or len(records)),
                "chapter_count": int(structure_report.get("chapter_count", 0) or 0),
                "image_count": int(structure_report.get("publication_resource_count", 0) or 0),
                "structure_builder": "v3_mature_framework_pipeline",
                "standalone_image_xhtml_count": int(structure_report.get("standalone_image_xhtml_count", 0) or 0),
            }
        except Exception:
            # A narrow fallback keeps V4 export available for minimal or legacy
            # documents that cannot satisfy the richer V3 publication preflight.
            skeleton_report = repair.export_ai_repair_epub(
                primary_doc, skeleton_package, skeleton_path,
                mode="one_pass", vertical=vertical, css_template=css_template,
                custom_css=custom_css, workflow="exchange",
            )
            skeleton_report["structure_builder"] = "direct_repair_epub_fallback"
        finally:
            shutil.rmtree(structure_temp, ignore_errors=True)

        crop_paths, crop_omissions = _export_conflict_crops(folder, package, records)
        for record in records:
            if record["item_id"] in crop_paths:
                record["evidence_paths"] = [crop_paths[record["item_id"]]]
            else:
                record["evidence_paths"] = []

        conflicts = []
        for record in records:
            if not record["model_action_required"]:
                continue
            conflict = copy.deepcopy(record)
            conflict["schema"] = CONFLICT_SCHEMA
            conflict["decision_template"] = {
                "item_id": record["item_id"], "selected_text": "", "source": "",
                "confidence": 0.0, "reason_code": "", "evidence": record["evidence_paths"],
            }
            conflicts.append(conflict)

        package_id = str(package.get("package_id", "") or hashlib.sha256(os.urandom(32)).hexdigest()[:24])
        structure_sha = str(package.get("structure_sha256", "") or "")
        summary.update({
            "package_id": package_id,
            "structure_sha256": structure_sha,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "conflict_crop_count": len(crop_paths),
            "conflict_crop_omission_count": len(crop_omissions),
        })
        source_payload = copy.deepcopy(package)
        source_payload["v4_standardization"] = {
            "raw_candidate_text_preserved": True,
            "canonical_text_exported_separately": True,
            "compare_key_never_written_back": True,
        }
        (folder / "00_INSTRUCTIONS.md").write_text(_instructions(), encoding="utf-8")
        (folder / "02_consensus_summary.json").write_bytes(_json_bytes(summary))
        _write_jsonl(folder / "03_conflict_items.jsonl", conflicts)
        _write_jsonl(folder / "04_all_items_index.jsonl", records)
        (folder / "05_terms.json").write_bytes(_json_bytes(_terms_payload(records)))
        (folder / "source" / "full_multi_model_ocr.json").write_bytes(_json_bytes(source_payload, pretty=False))
        (folder / "evidence" / "omissions.json").parent.mkdir(exist_ok=True)
        (folder / "evidence" / "omissions.json").write_bytes(_json_bytes(crop_omissions))
        (folder / "tools" / "apply_decisions.py").write_text(_standalone_apply_tool(), encoding="utf-8")
        (folder / "tools" / "validate_epub.py").write_text(_standalone_validate_tool(), encoding="utf-8")
        for script in (folder / "tools").glob("*.py"):
            script.chmod(script.stat().st_mode | stat.S_IXUSR)

        decision_template = {
            "schema": DECISIONS_SCHEMA,
            "package_id": package_id,
            "structure_sha256": structure_sha,
            "decisions": [conflict["decision_template"] for conflict in conflicts],
        }
        (folder / "decisions.template.json").write_bytes(_json_bytes(decision_template))
        manifest = {
            "schema": PACKAGE_SCHEMA,
            "package_id": package_id,
            "structure_sha256": structure_sha,
            "layout_sha256": str(package.get("layout_sha256", "") or ""),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "book": copy.deepcopy(package.get("book") or {}),
            "paths": {
                "instructions": "00_INSTRUCTIONS.md",
                "summary": "02_consensus_summary.json",
                "conflicts": "03_conflict_items.jsonl",
                "all_items": "04_all_items_index.jsonl",
                "terms": "05_terms.json",
                "source_multi_model_ocr": "source/full_multi_model_ocr.json",
                "skeleton_epub": "framework/structure_skeleton.epub",
                "decision_template": "decisions.template.json",
                "apply_tool": "tools/apply_decisions.py",
                "validate_tool": "tools/validate_epub.py",
            },
            "counts": {
                "total_items": len(records),
                "frozen_items": summary["frozen_item_count"],
                "model_action_required": summary["model_action_required_count"],
                "conflict_crops": len(crop_paths),
            },
            "invariants": {
                "frozen_items_must_not_change": True,
                "external_model_may_edit_only_conflict_ids": True,
                "epub_structure_is_program_owned": True,
                "raw_ocr_candidates_are_immutable": True,
                "canonical_text_is_safe_standard_japanese_copy": True,
                "compare_key_is_ephemeral": True,
            },
            "skeleton": skeleton_report,
            "publication_reference_included": False,
        }
        if include_publication_reference:
            reference = Path(publication_reference_path or "").expanduser()
            if not reference.is_file() or reference.suffix.lower() != ".epub":
                raise repair.AIRepairEpubError("已要求包含出版参考，但没有有效 EPUB。")
            target = folder / "reference" / reference.name
            target.parent.mkdir(); shutil.copy2(reference, target)
            manifest["publication_reference_included"] = True
            manifest["paths"]["publication_reference"] = f"reference/{reference.name}"
        core_hashes = {}
        for relative in manifest["paths"].values():
            path = folder / relative
            if path.is_file(): core_hashes[relative] = _sha256_bytes(path.read_bytes())
        manifest["core_files_sha256"] = core_hashes
        manifest["manifest_sha256"] = _sha256_bytes(_json_bytes({k:v for k,v in manifest.items() if k != "manifest_sha256"}, pretty=False))
        (folder / "01_manifest.json").write_bytes(_json_bytes(manifest))

        zip_path = folder.with_suffix(".zip")
        if create_zip:
            _zip_folder(folder, zip_path)
            with zipfile.ZipFile(zip_path) as archive:
                if archive.testzip() is not None:
                    raise repair.AIRepairEpubError("V4 ZIP CRC 检查失败。")
        return {
            "schema": PACKAGE_SCHEMA,
            "package_mode": "disagreement_v4",
            "folder": str(folder),
            "zip_path": str(zip_path) if create_zip else "",
            "guide": str(folder / "00_INSTRUCTIONS.md"),
            "fusion_json": str(folder / "source" / "full_multi_model_ocr.json"),
            "framework_epub": str(skeleton_path),
            "primary_framework_epub": str(skeleton_path),
            "editable_count": len(records),
            "stable_item_count": len(records),
            "locked_consensus_count": summary["frozen_item_count"],
            "review_required_count": summary["model_action_required_count"],
            "model_action_required_count": summary["model_action_required_count"],
            "exact_consensus_count": summary["status_counts"].get("exact_consensus", 0),
            "normalized_consensus_count": summary["status_counts"].get("normalized_consensus", 0),
            "majority_consensus_count": summary["status_counts"].get("majority_consensus", 0),
            "full_conflict_count": summary["status_counts"].get("full_conflict", 0),
            "missing_candidate_count": summary["status_counts"].get("missing_candidate", 0),
            "visual_evidence_file_count": len(crop_paths),
            "visual_evidence_page_count": len({record["page"] for record in records if record["item_id"] in crop_paths}),
            "publication_reference_included": bool(manifest["publication_reference_included"]),
            "chapter_count": int(skeleton_report.get("chapter_count", 0) or 0),
            "final_output_name": repair._safe_publication_filename(str((package.get("book") or {}).get("title", "") or "AI精校出版版"), default="AI精校出版版") + ".epub",
        }
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        zip_candidate = folder.with_suffix(".zip")
        zip_candidate.unlink(missing_ok=True)
        raise
