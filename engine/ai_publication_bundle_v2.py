#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Layered, rebuild-capable AI repair package.

The framework EPUB remains useful evidence, but export-time deterministic
preflight decides whether an external model should patch, hybrid-rebuild, or
fully rebuild the final EPUB.  Large books are split into reading text and
chapter JSONL evidence so a model does not need to parse one monolithic JSON.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence
import xml.etree.ElementTree as ET

from engine.text_compare import looks_like_chapter_title
from models.document import BlockType, UnifiedDocument
from utils.publication_preflight import inspect_document_for_publication

from engine import ai_repair_epub as repair
from engine import ocr_roundtrip_package as roundtrip
from engine import ai_publication_bundle_v3_support as v3


BUNDLE_SCHEMA = "novel_formatter.ai_repair_package.v3"
REBUILD_SCHEMA = "novel_formatter.ai_publication_rebuild_source.v3"
PREFLIGHT_SCHEMA = "novel_formatter.ai_publication_structure_preflight.v3"
ASSETS_SCHEMA = "novel_formatter.ai_publication_assets.v3"
RISK_SCHEMA = "novel_formatter.ai_publication_risk_queue.v3"
BOUNDARY_SCHEMA = "novel_formatter.ai_publication_boundary_windows.v3"
TERM_SCHEMA = "novel_formatter.ai_publication_term_consistency.v3"
STYLE_SCHEMA = "novel_formatter.ai_publication_style_profile.v3"
OUTPUT_CONTRACT_SCHEMA = "novel_formatter.ai_publication_output_contract.v3"
FINAL_AUDIT_SCHEMA = "novel_formatter.ai_publication_final_audit_rules.v3"
STABLE_TEXT_MAP_SCHEMA = "novel_formatter.ai_publication_stable_text_map.v3"
FRAMEWORK_AUDIT_SCHEMA = "novel_formatter.ai_publication_framework_audit.v3"
FULL_FUSION_FILENAME = "full_fusion_evidence.json"

_TERMINAL = tuple("。！？!?」』）)]】》〉〕〗〙〛…‥")
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".bmp", ".tif", ".tiff"}
_PUBLICATION_PAGE_TYPES = {
    "cover", "color_illus", "illustration", "half_illustration", "title_page",
    "frontispiece", "insert", "map", "character_sheet", "toc_page",
    "colophon", "index", "appendix",
}
_NON_PUBLICATION_PAGE_TYPES = {"paragraph", "blank", "advertisement", "unknown", "header_footer"}
_PUBLICATION_ROLE_BY_PAGE_TYPE = {
    "cover": "cover",
    "color_illus": "illustration",
    "illustration": "illustration",
    "half_illustration": "illustration",
    "frontispiece": "frontispiece",
    "insert": "insert",
    "title_page": "title_page",
    "toc_page": "toc_image",
    "map": "map",
    "character_sheet": "character_sheet",
    "colophon": "colophon",
    "index": "index_image",
    "appendix": "appendix_image",
}
PACKAGE_PROFILE = "selective_visual_locked_text"
_STATUS_TOKENS = ("status", "ステータス", "level", "レベル", "hp", "mp", "skill", "スキル", "職業", "称号")
_UNIT_RE = re.compile(r"(?P<number>[0-9０-９]+(?:[.,．，][0-9０-９]+)?)(?P<unit>歳|年|月|日|時|分|秒|人|名|体|匹|枚|本|個|階|級|位|点|％|%|km|cm|mm|kg|g|m)", re.I)
_KATAKANA_TERM_RE = re.compile(r"[ァ-ヶー・]{3,32}")
_LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_./:+-]{2,40}")
_JAPANESE_NAME_RE = re.compile(r"[一-龯々〆ヵヶ]{2,8}(?:・[一-龯々〆ヵヶ]{1,8})?")
_IDENTITY_PLACEHOLDERS = {
    "untitled", "unknown", "unknown author", "unknown title", "novel", "book",
    "无题", "未知", "不明", "作者不詳", "作者不明", "タイトル不明", "題名不明",
}
_FORBIDDEN_WORK_PREFIXES = ("META-INF/ai-repair/", "META-INF/ai-publication/", "evidence/", "reading/", "framework/")
_FORBIDDEN_WORK_ATTRIBUTES = ("data-item-id", "data-row-id", "data-block-id")
_MAX_ITEMS_PER_XHTML = 500
_MAX_TEXT_CHARACTERS_PER_XHTML = 80_000
_PUBLISHER_HINTS = (
    "アルファポリス", "KADOKAWA", "角川", "講談社", "集英社", "小学館",
    "双葉社", "ホビージャパン", "オーバーラップ", "SBクリエイティブ",
    "TOブックス", "一迅社", "主婦の友社", "新紀元社", "アース・スター",
)
_COVER_NOISE_RE = re.compile(r"(?:ISBN|定価|価格|発行|発売|文庫|ノベルス|コミックス|VOL\.?|VOLUME|第[0-9０-９]+巻)", re.I)
_AUTHOR_MARKER_RE = re.compile(r"^(?:著者|作者|原作|著|作)\s*[:：]?[\s　]*(.+)$")
_TITLE_MARKER_RE = re.compile(r"^(?:書名|題名|タイトル)\s*[:：]?[\s　]*(.+)$")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False).encode("utf-8")


def _redact_absolute_paths(value: Any, *, key: str = "") -> Any:
    """Remove machine-local absolute paths from exported JSON artifacts.

    The OCR text, hashes, geometry and stable IDs are preserved.  Only local
    filesystem locations are reduced to portable filenames so the exchange
    package cannot leak or depend on the exporting Mac's directory layout.
    """
    if isinstance(value, dict):
        return {str(k): _redact_absolute_paths(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_absolute_paths(v, key=key) for v in value]
    if isinstance(value, tuple):
        return [_redact_absolute_paths(v, key=key) for v in value]
    if isinstance(value, str):
        text = value.strip()
        pathish_key = any(token in key.lower() for token in ("path", "file", "image", "epub", "source"))
        if text.startswith("file://"):
            text = text[7:]
        try:
            candidate = Path(text).expanduser()
            if pathish_key and candidate.is_absolute():
                return candidate.name
        except (OSError, ValueError, RuntimeError):
            pass
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(_redact_absolute_paths(value)))


def _safe_name(value: str, default: str = "novel") -> str:
    return repair._safe_publication_filename(value, default=default)


def _clean_identity_value(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text or text.casefold() in _IDENTITY_PLACEHOLDERS:
        return None
    return text


def _cover_source_path(primary_doc: UnifiedDocument, package: dict) -> str:
    for page in getattr(primary_doc, "pages", []) or []:
        page_type = getattr(getattr(page, "page_type", None), "value", str(getattr(page, "page_type", "")))
        path = str(getattr(page, "image_path", "") or "")
        if page_type == BlockType.COVER.value and path:
            return path
    for asset in package.get("assets") or []:
        if not isinstance(asset, dict):
            continue
        role, include_final, _is_scan = _classify_asset(asset)
        if role == "cover" and include_final and str(asset.get("image_path", "") or ""):
            return str(asset.get("image_path", "") or "")
    return ""


def _cover_text_from_existing_evidence(primary_doc: UnifiedDocument, package: dict) -> tuple[str, str]:
    explicit = str(package.get("cover_ocr_text", "") or "").strip()
    if explicit:
        return explicit, "package_cover_ocr_text"
    for asset in package.get("assets") or []:
        if not isinstance(asset, dict):
            continue
        role, include_final, _is_scan = _classify_asset(asset)
        text = str(asset.get("ocr_text", "") or asset.get("recognized_text", "") or "").strip()
        if role == "cover" and include_final and text:
            return text, "package_cover_asset_ocr"
    cover_pages = {
        int(getattr(page, "page_no", 0) or 0)
        for page in getattr(primary_doc, "pages", []) or []
        if getattr(getattr(page, "page_type", None), "value", str(getattr(page, "page_type", "")))
        in {BlockType.COVER.value, BlockType.TITLE_PAGE.value}
    }
    lines = [
        str(getattr(block, "text", "") or "").strip()
        for block in getattr(primary_doc, "blocks", []) or []
        if int(getattr(block, "page", 0) or 0) in cover_pages and str(getattr(block, "text", "") or "").strip()
    ]
    return ("\n".join(lines), "document_cover_blocks") if lines else ("", "")


def _ocr_cover_text(path: str) -> tuple[str, float, str]:
    """Read the cover once in both common layouts and keep the stronger result.

    Light-novel covers frequently mix horizontal title/credit lines with
    vertical Japanese display text.  Testing both reading orders is still a
    single cover-only operation and avoids forcing the downstream model to
    recover metadata from the image again.
    """
    if not path or not Path(path).is_file() or platform.system() != "Darwin":
        return "", 0.0, ""
    try:
        from adapters.vision_backends.native_helper_backend import NativeVisionHelperBackend
        from adapters.vision_backends.base import OCRConfig
        backend = NativeVisionHelperBackend()
        available, _reason = backend.is_available()
        if not available:
            return "", 0.0, ""
        results: list[tuple[str, float, str, float]] = []
        try:
            for vertical in (False, True):
                result = backend.recognize(path, OCRConfig(
                    recognition_level="accurate",
                    languages=["ja-JP", "en-US"],
                    vertical=vertical,
                    timeout=35.0,
                    use_language_correction=True,
                    automatically_detect_language=False,
                    candidate_count=3,
                ))
                text = str(result.full_text or "").strip()
                if not text:
                    continue
                confidences = [float(block.confidence or 0.0) for block in result.blocks if str(block.text or "").strip()]
                confidence = sum(confidences) / len(confidences) if confidences else 0.82
                parsed = _parse_cover_identity_text(
                    text,
                    source="apple_vision_cover_ocr_vertical" if vertical else "apple_vision_cover_ocr_horizontal",
                    ocr_confidence=confidence,
                )
                title_score = max((float(value.get("confidence", 0.0) or 0.0) for value in parsed.get("title_candidates", [])), default=0.0)
                author_score = max((float(value.get("confidence", 0.0) or 0.0) for value in parsed.get("author_candidates", [])), default=0.0)
                quality = title_score * 1.4 + author_score + min(0.2, len(text) / 500.0)
                results.append((text, round(max(0.0, min(1.0, confidence)), 3), "apple_vision_cover_ocr_vertical" if vertical else "apple_vision_cover_ocr_horizontal", quality))
        finally:
            backend.close()
        if not results:
            return "", 0.0, ""
        text, confidence, source, _quality = max(results, key=lambda value: (value[3], value[1], len(value[0])))
        return text, confidence, source
    except Exception:
        return "", 0.0, ""


def _clean_cover_line(value: str) -> str:
    text = re.sub(r"[\t\r]+", " ", str(value or "")).strip(" \u3000|｜")
    return re.sub(r"[ ]{2,}", " ", text)


def _parse_cover_identity_text(text: str, *, source: str, ocr_confidence: float = 0.0) -> dict:
    lines = [_clean_cover_line(value) for value in re.split(r"[\n\f]+", str(text or ""))]
    lines = [value for value in lines if value and not _COVER_NOISE_RE.search(value)]
    title_candidates: list[dict] = []
    author_candidates: list[dict] = []
    for order, line in enumerate(lines):
        title_match = _TITLE_MARKER_RE.match(line)
        if title_match:
            value = _clean_identity_value(title_match.group(1))
            if value:
                title_candidates.append({"value": value, "confidence": max(0.94, ocr_confidence), "source": source, "method": "explicit_title_marker", "line_order": order})
            continue
        author_match = _AUTHOR_MARKER_RE.match(line)
        if author_match:
            value = _clean_identity_value(author_match.group(1))
            if value:
                author_candidates.append({"value": value, "confidence": max(0.94, ocr_confidence), "source": source, "method": "explicit_author_marker", "line_order": order})
            continue
        if any(hint.casefold() in line.casefold() for hint in _PUBLISHER_HINTS):
            continue
        japanese_count = len(re.findall(r"[一-龯々ぁ-んァ-ヶー]", line))
        if 5 <= len(line) <= 80 and japanese_count >= 4:
            score = 0.70 + min(0.16, len(line) / 240.0) + (0.04 if re.search(r"[のにをへがはでと]", line) else 0.0)
            title_candidates.append({"value": line, "confidence": round(max(score, ocr_confidence * 0.9), 3), "source": source, "method": "cover_line_title_heuristic", "line_order": order})
        if 2 <= len(line) <= 16 and japanese_count >= 2 and not re.search(r"[。！？!?、，,:：]", line):
            score = 0.72 + (0.07 if re.search(r"[一-龯々].*[ぁ-んァ-ヶー]|[ぁ-んァ-ヶー].*[一-龯々]", line) else 0.0)
            author_candidates.append({"value": line, "confidence": round(max(score, ocr_confidence * 0.86), 3), "source": source, "method": "cover_line_author_heuristic", "line_order": order})
    title_candidates.sort(key=lambda value: (-float(value["confidence"]), -len(str(value["value"])), int(value["line_order"])))
    title_value = title_candidates[0]["value"] if title_candidates else None
    author_candidates = [value for value in author_candidates if value.get("value") != title_value]
    author_candidates.sort(key=lambda value: (-float(value["confidence"]), int(value["line_order"]), len(str(value["value"]))))
    return {"title_candidates": title_candidates[:8], "author_candidates": author_candidates[:8], "raw_text_sha256": _sha256_bytes(str(text or "").encode("utf-8")) if text else ""}


def _resolve_cover_identity(primary_doc: UnifiedDocument, package: dict) -> dict:
    existing_text, source = _cover_text_from_existing_evidence(primary_doc, package)
    confidence = 0.98 if existing_text else 0.0
    if not existing_text:
        existing_text, confidence, source = _ocr_cover_text(_cover_source_path(primary_doc, package))
    parsed = _parse_cover_identity_text(existing_text, source=source or "cover_ocr", ocr_confidence=confidence) if existing_text else {"title_candidates": [], "author_candidates": [], "raw_text_sha256": ""}
    parsed["attempted"] = bool(existing_text or platform.system() == "Darwin")
    parsed["source"] = source or None
    return parsed


def _confidence_band(value: Any) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError):
        score = 0.0
    if score >= 0.9:
        return "high"
    if score >= 0.65:
        return "medium"
    return "low"


def _all_regular_files(folder: Path) -> list[Path]:
    return sorted(
        (path for path in folder.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(folder).as_posix(),
    )


def _file_hash_inventory(folder: Path, *, exclude: set[str] | None = None) -> dict[str, str]:
    excluded = set(exclude or set())
    return {
        path.relative_to(folder).as_posix(): _sha256_bytes(path.read_bytes())
        for path in _all_regular_files(folder)
        if path.relative_to(folder).as_posix() not in excluded
    }


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _framework_inventory(epub_path: Path) -> dict:
    inventory: dict[str, Any] = {
        "epub_path": f"framework/{epub_path.name}",
        "opf_path": "",
        "manifest": [],
        "spine": [],
        "nav": {"path": "", "in_spine": False, "entries": []},
        "css": [],
        "xhtml": [],
        "images": [],
        "metadata": {},
    }
    with zipfile.ZipFile(epub_path, "r") as archive:
        names = archive.namelist()
        opf_names = [name for name in names if name.lower().endswith(".opf")]
        if not opf_names:
            return inventory
        opf_name = sorted(opf_names, key=lambda name: (name.lower() != "epub/content.opf", len(name)))[0]
        inventory["opf_path"] = opf_name
        root = ET.fromstring(archive.read(opf_name))
        opf_dir = PurePosixPath(opf_name).parent
        manifest_by_id: dict[str, dict] = {}
        for element in root.iter():
            local = _local(element.tag).lower()
            if local == "item":
                item_id = str(element.attrib.get("id", "") or "")
                href = str(element.attrib.get("href", "") or "")
                full = str((opf_dir / href).as_posix()) if href else ""
                entry = {
                    "id": item_id,
                    "href": href,
                    "archive_path": full,
                    "media_type": str(element.attrib.get("media-type", "") or ""),
                    "properties": str(element.attrib.get("properties", "") or ""),
                }
                manifest_by_id[item_id] = entry
                inventory["manifest"].append(entry)
            elif local in {"title", "creator", "language", "publisher", "identifier"}:
                text = "".join(element.itertext()).strip()
                if text:
                    inventory["metadata"].setdefault(local, []).append(text)
        for element in root.iter():
            if _local(element.tag).lower() != "itemref":
                continue
            item_id = str(element.attrib.get("idref", "") or "")
            entry = manifest_by_id.get(item_id) or {"id": item_id, "archive_path": ""}
            inventory["spine"].append({
                "idref": item_id,
                "archive_path": entry.get("archive_path", ""),
                "linear": str(element.attrib.get("linear", "yes") or "yes"),
            })
        nav_entry = next((entry for entry in inventory["manifest"] if "nav" in str(entry.get("properties", "")).split()), None)
        if nav_entry:
            nav_path = str(nav_entry.get("archive_path", "") or "")
            inventory["nav"]["path"] = nav_path
            inventory["nav"]["in_spine"] = any(item.get("idref") == nav_entry.get("id") for item in inventory["spine"])
            if nav_path in names:
                try:
                    nav_root = ET.fromstring(archive.read(nav_path))
                    for anchor in nav_root.iter():
                        if _local(anchor.tag).lower() != "a":
                            continue
                        inventory["nav"]["entries"].append({
                            "label": "".join(anchor.itertext()).strip(),
                            "href": str(anchor.attrib.get("href", "") or ""),
                        })
                except ET.ParseError:
                    inventory["nav"]["parse_error"] = True
        for name in names:
            lowered = name.lower()
            if lowered.endswith(".css"):
                raw = archive.read(name)
                inventory["css"].append({"path": name, "size": len(raw), "sha256": _sha256_bytes(raw)})
            elif lowered.endswith((".xhtml", ".html", ".htm")):
                raw = archive.read(name)
                text = raw.decode("utf-8", errors="replace")
                inventory["xhtml"].append({
                    "path": name,
                    "size": len(raw),
                    "sha256": _sha256_bytes(raw),
                    "character_count": len(re.sub(r"<[^>]+>", "", text)),
                })
            elif Path(lowered).suffix in _IMAGE_EXTENSIONS:
                raw = archive.read(name)
                inventory["images"].append({"path": name, "size": len(raw), "sha256": _sha256_bytes(raw)})
    return inventory



def _strip_framework_work_payloads(epub_path: Path) -> None:
    """Remove embedded AI evidence while preserving XHTML stable-ID mapping."""
    temp_path = epub_path.with_suffix(epub_path.suffix + ".clean.tmp")
    prefixes = (repair.AI_REPAIR_ROOT + "/", repair.AI_PUBLICATION_ROOT + "/")
    exact = {
        repair.MAP_PATH,
        repair.GUIDE_PATH,
        repair.TEMPLATE_PATH,
        repair.AI_PUBLICATION_MANIFEST_PATH,
        repair.AI_PUBLICATION_FUSION_PATH,
        repair.AI_PUBLICATION_GUIDE_PATH,
    }
    with zipfile.ZipFile(epub_path, "r") as source, zipfile.ZipFile(temp_path, "w") as target:
        mime = zipfile.ZipInfo("mimetype")
        mime.compress_type = zipfile.ZIP_STORED
        mime.external_attr = 0o644 << 16
        target.writestr(mime, b"application/epub+zip")
        for info in source.infolist():
            name = info.filename
            if name == "mimetype" or name in exact or any(name.startswith(prefix) for prefix in prefixes):
                continue
            target.writestr(info, source.read(name))
    os.replace(temp_path, epub_path)

def _item_source_indices(item: dict) -> list[int]:
    values = list(item.get("primary_block_indices") or [])
    if item.get("primary_block_index") is not None:
        values.append(item.get("primary_block_index"))
    result: list[int] = []
    for value in values:
        try:
            result.append(int(value))
        except (TypeError, ValueError, OverflowError):
            continue
    return sorted(set(result))


def _nearest_item_ids_for_block(block_order: int, items: Sequence[dict]) -> tuple[str, str]:
    positioned: list[tuple[int, str]] = []
    for item in items:
        indices = _item_source_indices(item)
        if indices:
            positioned.append((min(indices), str(item.get("item_id", "") or "")))
    positioned.sort()
    before = [item_id for index, item_id in positioned if index < block_order]
    after = [item_id for index, item_id in positioned if index > block_order]
    return (before[-1] if before else "", after[0] if after else "")


def _image_dimensions(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:
        return 0, 0


def _normalised_file_key(value: str | Path) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return os.path.normcase(str(Path(text).expanduser().resolve(strict=False)))
    except Exception:
        return os.path.normcase(text)


def _asset_page_type(asset: dict) -> str:
    metadata = asset.get("metadata") if isinstance(asset.get("metadata"), dict) else {}
    return str(asset.get("page_type") or metadata.get("page_type") or "").strip().lower()


def _classify_asset(asset: dict) -> tuple[str, bool, bool]:
    """Return ``(publication_role, include_in_final_epub, is_scan_page)``.

    Page-manager text pages and other OCR source pages are intentionally omitted.
    IMAGE_REF records without an explicit non-publication page type remain final
    publication images because they represent manually placed book artwork.
    """
    page_type = _asset_page_type(asset)
    explicit_role = str(asset.get("publication_role") or "").strip().lower()
    explicit_include = asset.get("include_in_final_epub")
    if bool(asset.get("is_cover")) or page_type == BlockType.COVER.value:
        return "cover", True, False
    if explicit_include is False:
        return explicit_role or "ocr_source_page", False, True
    if page_type in _PUBLICATION_PAGE_TYPES:
        return explicit_role or _PUBLICATION_ROLE_BY_PAGE_TYPE.get(page_type, "illustration"), True, False
    if explicit_include is True:
        return explicit_role or "illustration", True, False
    if str(asset.get("kind", "") or "") == "block":
        if page_type in _NON_PUBLICATION_PAGE_TYPES:
            return "ocr_source_page", False, True
        return explicit_role or "illustration", True, False
    return "ocr_source_page", False, True


def _publication_image_source_paths(package: dict) -> set[str]:
    result: set[str] = set()
    for asset in package.get("assets") or []:
        if not isinstance(asset, dict):
            continue
        _role, include, _is_scan = _classify_asset(asset)
        if not include:
            continue
        key = _normalised_file_key(asset.get("image_path", ""))
        if key:
            result.add(key)
    return result


def _asset_anchor(asset: dict, items: Sequence[dict]) -> tuple[str, str]:
    before_id = after_id = ""
    if asset.get("kind") == "block":
        try:
            block_order = int(asset.get("block_order", -1))
        except (TypeError, ValueError, OverflowError):
            block_order = -1
        before_id, after_id = _nearest_item_ids_for_block(block_order, items)
    elif asset.get("kind") == "page":
        try:
            page = int(asset.get("page_no", 0) or 0)
        except (TypeError, ValueError, OverflowError):
            page = 0
        same_page = [item for item in items if int(item.get("page", 0) or 0) == page]
        if same_page:
            before_id = str(same_page[0].get("prev_item_id", "") or "")
            after_id = str(same_page[-1].get("next_item_id", "") or "")
    return before_id, after_id


def _copy_assets(
    package: dict,
    items: Sequence[dict],
    resources_dir: Path,
    *,
    storage_mode: str = "copy",
    framework_inventory: dict | None = None,
    framework_name: str = "resource_mapping_framework.epub",
) -> dict:
    """Register final-publication artwork without duplicating it in compact mode.

    ``copy`` preserves the legacy/full-forensic package.  ``framework`` points
    each publication asset at the byte-identical member already stored inside
    the clean mapping EPUB; the final builder extracts and verifies that member.
    """
    storage_mode = "framework" if str(storage_mode).lower() == "framework" else "copy"
    if storage_mode == "copy":
        resources_dir.mkdir(parents=True, exist_ok=True)
    framework_by_digest = {
        str(value.get("sha256", "") or ""): value
        for value in (framework_inventory or {}).get("images", [])
        if str(value.get("sha256", "") or "")
    }
    source_assets = [copy.deepcopy(value) for value in (package.get("assets") or []) if isinstance(value, dict)]
    publication_by_digest: dict[str, dict] = {}
    omitted_by_source: dict[str, dict] = {}
    missing_publication: list[dict] = []
    source_available_keys: set[str] = set()
    publication_source_records = 0
    scan_source_records = 0

    for order, asset in enumerate(source_assets):
        role, include_final, is_scan = _classify_asset(asset)
        source_text = str(asset.get("image_path", "") or "")
        source = Path(source_text).expanduser() if source_text else Path()
        source_key = _normalised_file_key(source_text) or f"record:{order}"
        before_id, after_id = _asset_anchor(asset, items)
        source_record = {
            "source_record_order": order,
            "kind": str(asset.get("kind", "") or ""),
            "page_no": int(asset.get("page_no", asset.get("page", 0)) or 0),
            "block_id": str(asset.get("block_id", "") or ""),
            "page_type": _asset_page_type(asset),
            "source_path": source_text,
        }

        if not include_final:
            scan_source_records += 1
            record = omitted_by_source.get(source_key)
            if record is None:
                record = {
                    "asset_id": "", "source_path": source_text, "source_records": [],
                    "export_path": "", "resource_path": "", "publication_role": role, "role": role,
                    "include_in_final_epub": False, "include_in_ai_package": False,
                    "is_scan_page": bool(is_scan), "export_status": "intentionally_omitted",
                    "available_at_export": bool(source_text and source.is_file()), "sha256": "",
                    "sha256_omitted_reason": "scan_or_nonpublication_source_not_exported",
                    "size": int(source.stat().st_size) if source_text and source.is_file() else 0,
                    "actual_width": int(asset.get("width", 0) or 0),
                    "actual_height": int(asset.get("height", 0) or 0),
                    "anchor_before_item_id": before_id or None, "anchor_after_item_id": after_id or None,
                    "stable_item_before": before_id, "stable_item_after": after_id,
                }
                omitted_by_source[source_key] = record
            record["source_records"].append(source_record)
            continue

        publication_source_records += 1
        if not source_text or not source.is_file():
            missing_publication.append({
                "asset_id": "", "source_path": source_text, "source_records": [source_record],
                "export_path": "", "resource_path": "", "publication_role": role, "role": role,
                "include_in_final_epub": True, "include_in_ai_package": True,
                "is_scan_page": False, "export_status": "missing_source", "available_at_export": False,
                "available": False, "sha256": "", "size": 0,
                "actual_width": int(asset.get("width", 0) or 0), "actual_height": int(asset.get("height", 0) or 0),
                "anchor_before_item_id": before_id or None, "anchor_after_item_id": after_id or None,
                "stable_item_before": before_id, "stable_item_after": after_id,
            })
            continue

        source_available_keys.add(source_key)
        raw = source.read_bytes()
        digest = _sha256_bytes(raw)
        record = publication_by_digest.get(digest)
        if record is None:
            actual_width, actual_height = _image_dimensions(source)
            ext = source.suffix.lower() if source.suffix.lower() in _IMAGE_EXTENSIONS else ".bin"
            if role == "cover" and not any(value.get("publication_role") == "cover" for value in publication_by_digest.values()):
                resource_name = "cover" + ext
            else:
                illustration_number = 1 + sum(1 for value in publication_by_digest.values() if value.get("publication_role") != "cover")
                resource_name = f"illustration_{illustration_number:03d}{ext}"
            framework_member = framework_by_digest.get(digest)
            if storage_mode == "framework" and framework_member:
                export_path = ""
                resource_path = str(framework_member.get("path", "") or "")
                storage = "framework_epub"
                internal_path = resource_path
            else:
                target = resources_dir / resource_name
                resources_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                export_path = resource_path = f"resources/{resource_name}"
                storage = "resources_directory"
                internal_path = ""
            record = {
                "asset_id": "", "source_path": source_text, "source_records": [],
                "export_path": export_path, "resource_path": resource_path,
                "storage": storage, "framework_path": f"framework/{framework_name}" if storage == "framework_epub" else "",
                "internal_path": internal_path,
                "publication_role": role, "role": role, "publication_roles": [role],
                "include_in_final_epub": True, "include_in_ai_package": True, "is_scan_page": False,
                "export_status": "exported", "available_at_export": True, "available": True,
                "format": ext.lstrip("."), "source_filename": source.name, "sha256": digest, "size": len(raw),
                "actual_width": actual_width or int(asset.get("width", 0) or 0),
                "actual_height": actual_height or int(asset.get("height", 0) or 0),
                "anchor_before_item_id": before_id or None, "anchor_after_item_id": after_id or None,
                "stable_item_before": before_id, "stable_item_after": after_id, "anchors": [],
            }
            publication_by_digest[digest] = record
        elif role not in record["publication_roles"]:
            record["publication_roles"].append(role)
            if role == "cover":
                record["publication_role"] = record["role"] = "cover"
        record["source_records"].append(source_record)
        anchor = {"before_item_id": before_id or None, "after_item_id": after_id or None}
        if anchor not in record["anchors"] and (before_id or after_id):
            record["anchors"].append(anchor)
        if not record.get("anchor_before_item_id") and before_id:
            record["anchor_before_item_id"] = before_id
            record["stable_item_before"] = before_id
        if not record.get("anchor_after_item_id") and after_id:
            record["anchor_after_item_id"] = after_id
            record["stable_item_after"] = after_id

    publication_records = list(publication_by_digest.values()) + missing_publication
    publication_records.sort(key=lambda value: (value.get("publication_role") != "cover", value.get("resource_path", ""), value.get("source_path", "")))
    omitted_records = sorted(omitted_by_source.values(), key=lambda value: value.get("source_path", ""))
    for index, record in enumerate(publication_records, start=1):
        record["asset_id"] = f"asset-{index:03d}"
        record["asset_order"] = index - 1
    for index, record in enumerate(omitted_records, start=1):
        record["asset_id"] = f"scan-page-{index:03d}"
        record["asset_order"] = len(publication_records) + index - 1
        record["available"] = False

    records = publication_records + omitted_records
    exported_count = sum(1 for record in publication_records if record.get("export_status") == "exported")
    expected_count = len(publication_records)
    missing_count = expected_count - exported_count
    unique_source_file_count = len(source_available_keys) + sum(1 for record in omitted_records if record.get("available_at_export"))
    return {
        "schema": ASSETS_SCHEMA, "package_profile": PACKAGE_PROFILE,
        "asset_count_declared": int((package.get("book") or {}).get("asset_count", len(source_assets)) or 0),
        "source_asset_record_count": len(source_assets), "asset_record_count": len(records),
        "deduplicated_asset_record_count": len(records), "unique_source_file_count": unique_source_file_count,
        "publication_source_record_count": publication_source_records, "publication_image_count": expected_count,
        "publication_resource_count": exported_count, "publication_image_count_exported": exported_count,
        "final_epub_expected_image_count": expected_count, "scan_source_record_count": scan_source_records,
        "scan_page_count_detected": len(omitted_records), "scan_page_count_exported": 0,
        "source_page_scan_count_detected": len(omitted_records), "source_page_scan_count": 0,
        "scan_pages_required": False, "scan_evidence_absence_is_error": False,
        "publication_images_required": True, "publication_image_count_must_match_manifest": True,
        "missing_publication_image_count": missing_count, "missing_resource_count": missing_count,
        "duplicate_asset_records": max(0, len(source_assets) - len(records)),
        "resources_directory_policy": "framework_epub_single_copy" if storage_mode == "framework" else "publication_images_only",
        "publication_image_storage": storage_mode,
        "assets": records,
    }

def _longest_overlap(left: str, right: str, maximum: int = 80) -> str:
    left = str(left or "").rstrip()
    right = str(right or "").lstrip()
    limit = min(maximum, len(left), len(right))
    for size in range(limit, 1, -1):
        if left[-size:] == right[:size]:
            return right[:size]
    return ""


def _quote_balance(text: str) -> dict:
    return {
        "dialogue": text.count("「") - text.count("」"),
        "citation": text.count("『") - text.count("』"),
        "parenthesis": text.count("（") - text.count("）"),
    }


def _structure_preflight(
    primary_doc: UnifiedDocument,
    repair_map: dict,
    framework_inventory: dict,
    assets_manifest: dict,
    book_identity: dict | None = None,
    chapter_manifest: Sequence[dict] | None = None,
) -> dict:
    deterministic = inspect_document_for_publication(primary_doc)
    items = sorted(list(repair_map.get("items") or []), key=lambda value: int(value.get("reading_order", value.get("row_index", 0)) or 0))
    by_id = {str(item.get("item_id", "") or ""): item for item in items}
    candidates = list(repair_map.get("chapter_candidates") or repair._detect_chapter_candidates(primary_doc, items))
    actual_chapters = [item for item in items if item.get("chapter_candidate") or str(item.get("block_type", "")) == BlockType.CHAPTER.value]
    title_glue: list[dict] = []
    for item in items:
        text = str(item.get("baseline_text", "") or "").strip()
        first_line, _, rest = text.partition("\n")
        if rest and looks_like_chapter_title(first_line) and len(rest.strip()) > 20:
            title_glue.append({"item_id": item.get("item_id"), "title": first_line, "body_excerpt": rest.strip()[:120]})
        elif looks_like_chapter_title(text[:80]) and len(text) > 180:
            title_glue.append({"item_id": item.get("item_id"), "title": text[:80], "body_excerpt": text[80:200]})

    overlong_xhtml = [
        entry for entry in framework_inventory.get("xhtml", [])
        if int(entry.get("size", 0) or 0) > 512_000 or int(entry.get("character_count", 0) or 0) > 120_000
    ]
    adjacent_overlaps: list[dict] = []
    three_short_fragments: list[dict] = []
    abnormal_latin: list[dict] = []
    status_one_line: list[dict] = []
    placeholder_items: list[dict] = []
    replacement_items: list[dict] = []
    for index, item in enumerate(items):
        text = str(item.get("baseline_text", "") or "")
        if "□" in text:
            placeholder_items.append({"item_id": item.get("item_id"), "count": text.count("□"), "excerpt": text[:180]})
        if "�" in text:
            replacement_items.append({"item_id": item.get("item_id"), "count": text.count("�"), "excerpt": text[:180]})
        if index:
            overlap = _longest_overlap(str(items[index - 1].get("baseline_text", "") or ""), text)
            if len(overlap) >= 4:
                adjacent_overlaps.append({
                    "left_item_id": items[index - 1].get("item_id"),
                    "right_item_id": item.get("item_id"),
                    "overlap": overlap,
                    "length": len(overlap),
                })
        if re.search(r"[A-Za-z]{4,}", text) and not re.search(r"(?:https?://|ISBN|URL)", text, re.I):
            abnormal_latin.append({"item_id": item.get("item_id"), "matches": _LATIN_RE.findall(text)[:8]})
        lower = text.lower()
        status_hits = sum(1 for token in _STATUS_TOKENS if token in lower)
        if status_hits >= 3 and "\n" not in text and len(text) >= 30:
            status_one_line.append({"item_id": item.get("item_id"), "excerpt": text[:180]})
        if index >= 2:
            trio = items[index - 2:index + 1]
            lengths = [len(str(value.get("baseline_text", "") or "").strip()) for value in trio]
            if all(0 < length <= 12 for length in lengths):
                three_short_fragments.append({"item_ids": [value.get("item_id") for value in trio], "texts": [value.get("baseline_text") for value in trio]})

    image_boundary_issues: list[dict] = []
    for asset in assets_manifest.get("assets", []):
        if not asset.get("include_in_final_epub") or asset.get("export_status") != "exported":
            continue
        before_id = str(asset.get("stable_item_before", "") or "")
        after_id = str(asset.get("stable_item_after", "") or "")
        if not before_id or not after_id:
            continue
        left = str((by_id.get(before_id) or {}).get("baseline_text", "") or "")
        right = str((by_id.get(after_id) or {}).get("baseline_text", "") or "")
        balance = _quote_balance(left)
        if (left and not left.rstrip().endswith(_TERMINAL)) or any(value != 0 for value in balance.values()):
            image_boundary_issues.append({
                "resource_path": asset.get("resource_path"),
                "before_item_id": before_id,
                "after_item_id": after_id,
                "before_ends_incomplete": bool(left and not left.rstrip().endswith(_TERMINAL)),
                "quote_balance_before": balance,
                "combined_excerpt": (left[-100:] + " [IMAGE] " + right[:100]),
            })

    nav = framework_inventory.get("nav") or {}
    nav_in_spine = bool(nav.get("in_spine"))
    nav_entries = list(nav.get("entries") or [])
    nav_count = len(nav_entries)
    nav_labels = [str(value.get("label", "") or "").strip() for value in nav_entries]
    nav_only_epilogue = nav_count == 1 and bool(re.search(r"(?:エピローグ|終章|尾声|後日談)", nav_labels[0], re.I))
    chapter_candidate_count = len(candidates)
    actual_chapter_count = len(actual_chapters)
    spine_content = [
        entry for entry in framework_inventory.get("spine", [])
        if str(entry.get("archive_path", "") or "") != str(nav.get("path", "") or "")
    ]
    framework_chapter_file_count = len(spine_content)
    chapter_manifest = list(chapter_manifest or [])
    logical_chapter_count = len(chapter_manifest) or max(1, chapter_candidate_count)
    planned_content_xhtml_count = sum(int(chapter.get("planned_content_xhtml_count", chapter.get("technical_part_count", 1)) or 1) for chapter in chapter_manifest) or max(1, logical_chapter_count)
    must_partition_long_xhtml = any(
        int(chapter.get("technical_part_count", 1) or 1) > 1
        and any(not bool(part.get("forced_by_image_boundary")) for part in (chapter.get("parts") or []))
        for chapter in chapter_manifest
    )
    must_split_at_image_anchors = any(
        bool(part.get("forced_by_image_boundary"))
        for chapter in chapter_manifest
        for part in (chapter.get("parts") or [])
    )
    total_text_character_count = sum(len(str(item.get("baseline_text", "") or "")) for item in items)

    identity = book_identity or {}
    identity_unresolved = bool(identity.get("identity_resolution_required"))
    blockers: list[str] = []
    if chapter_candidate_count and actual_chapter_count != chapter_candidate_count:
        blockers.append("chapter_candidate_actual_count_mismatch")
    if title_glue:
        blockers.append("chapter_title_glued_to_body")
    if overlong_xhtml:
        blockers.append("overlong_xhtml")
    if must_partition_long_xhtml:
        blockers.append("long_chapter_requires_technical_partition")
    if must_split_at_image_anchors:
        blockers.append("standalone_image_requires_spine_partition")
    if nav_in_spine:
        blockers.append("nav_in_spine")
    if nav_only_epilogue:
        blockers.append("nav_only_contains_epilogue")
    if image_boundary_issues:
        blockers.append("image_inside_unfinished_text_boundary")
    if deterministic.unplaced_image_count:
        blockers.append("unplaced_images")
    if deterministic.duplicate_runs:
        blockers.append("continuous_duplicate_runs")
    if placeholder_items:
        blockers.append("placeholder_square_found")
    if replacement_items:
        blockers.append("replacement_character_found")
    if identity_unresolved:
        blockers.append("book_identity_unresolved")
    if int(assets_manifest.get("missing_publication_image_count", 0) or 0) > 0:
        blockers.append("missing_publication_images")

    severe_structure = bool(
        not items
        or not framework_inventory.get("opf_path")
        or (
            int(assets_manifest.get("publication_image_count", 0) or 0) > 0
            and int(assets_manifest.get("publication_resource_count", 0) or 0) == 0
        )
    )
    chapter_nav_mismatch = bool(chapter_candidate_count and nav_count != chapter_candidate_count)
    if severe_structure:
        recommended = "full_rebuild"
    elif blockers or deterministic.high_text_issue_count or chapter_nav_mismatch:
        recommended = "hybrid_rebuild"
    else:
        recommended = "preserve_and_patch"
    framework_authoritative = recommended == "preserve_and_patch"

    must_rebuild_nav = bool(recommended != "preserve_and_patch" or nav_in_spine or nav_only_epilogue or chapter_nav_mismatch)
    must_rebuild_ncx = bool(recommended != "preserve_and_patch" or chapter_nav_mismatch)
    must_rebuild_spine = bool(recommended != "preserve_and_patch" or nav_in_spine or chapter_nav_mismatch)
    must_split_chapter_xhtml = bool(
        must_partition_long_xhtml
        or must_split_at_image_anchors
        or title_glue
        or overlong_xhtml
        or (chapter_candidate_count and framework_chapter_file_count < chapter_candidate_count)
    )
    passed = bool(
        recommended == "preserve_and_patch"
        and not blockers
        and not deterministic.high_text_issue_count
        and not chapter_nav_mismatch
    )

    return {
        "schema": PREFLIGHT_SCHEMA,
        "passed": passed,
        "recommended_build_strategy": recommended,
        "recommended_build_strategy_is_mandatory": True,
        "framework_structure_authoritative": framework_authoritative,
        "framework_is_primary_evidence_not_technical_truth": True,
        "package_profile": PACKAGE_PROFILE,
        "scan_pages_required": False,
        "scan_page_count_expected": 0,
        "scan_evidence_absence_is_error": False,
        "publication_images_required": True,
        "publication_image_count_must_match_manifest": True,
        "detected_chapter_count": chapter_candidate_count,
        "logical_chapter_count": logical_chapter_count,
        "framework_chapter_file_count": framework_chapter_file_count,
        "must_partition_long_xhtml": must_partition_long_xhtml,
        "must_split_at_image_anchors": must_split_at_image_anchors,
        "max_items_per_xhtml": _MAX_ITEMS_PER_XHTML,
        "max_text_characters_per_xhtml": _MAX_TEXT_CHARACTERS_PER_XHTML,
        "minimum_content_xhtml_count": planned_content_xhtml_count,
        "planned_content_xhtml_count": planned_content_xhtml_count,
        "total_text_character_count": total_text_character_count,
        "nav_in_spine": nav_in_spine,
        "nav_only_contains_epilogue": nav_only_epilogue,
        "chapter_titles_embedded_in_paragraphs": bool(title_glue),
        "must_rebuild_nav": must_rebuild_nav,
        "must_rebuild_ncx": must_rebuild_ncx,
        "must_rebuild_spine": must_rebuild_spine,
        "must_split_chapter_xhtml": must_split_chapter_xhtml,
        "must_resolve_book_identity": identity_unresolved,
        "must_remove_placeholder_squares": bool(placeholder_items),
        "must_remove_replacement_characters": bool(replacement_items),
        "prohibited_actions": [
            action for enabled, action in [
                (must_rebuild_nav, "reuse_framework_nav_without_rebuild"),
                (must_rebuild_ncx, "reuse_framework_ncx_without_rebuild"),
                (must_rebuild_spine, "reuse_framework_spine_without_rebuild"),
                (must_split_chapter_xhtml, "reuse_framework_xhtml_partitioning_without_split"),
                (must_partition_long_xhtml, "place_all_long_chapter_items_in_one_xhtml"),
            ] if enabled
        ],
        "strategies": {
            "preserve_and_patch": "结构可靠：保留现有 OPF/NAV/spine/CSS，只按稳定 item_id 修正文。",
            "hybrid_rebuild": "保留图片、稳定映射和可用样式，重建章节、NAV、NCX、OPF 与 spine。",
            "full_rebuild": "忽略不可靠 EPUB 结构，根据 structure_document、assets、reading 与 evidence 从零构建。",
        },
        "summary": {
            "chapter_candidate_count": chapter_candidate_count,
            "logical_chapter_count": logical_chapter_count,
            "actual_chapter_start_count": actual_chapter_count,
            "planned_content_xhtml_count": planned_content_xhtml_count,
            "must_partition_long_xhtml": must_partition_long_xhtml,
            "must_split_at_image_anchors": must_split_at_image_anchors,
            "max_items_per_xhtml": _MAX_ITEMS_PER_XHTML,
            "max_text_characters_per_xhtml": _MAX_TEXT_CHARACTERS_PER_XHTML,
            "nav_entry_count": nav_count,
            "nav_in_spine": nav_in_spine,
            "xhtml_count": len(framework_inventory.get("xhtml", [])),
            "overlong_xhtml_count": len(overlong_xhtml),
            "image_boundary_issue_count": len(image_boundary_issues),
            "high_text_issue_count": deterministic.high_text_issue_count,
            "medium_text_issue_count": deterministic.medium_text_issue_count,
            "quote_imbalance_count": deterministic.quote_imbalance_count,
            "placeholder_square_count": sum(value["count"] for value in placeholder_items),
            "replacement_character_count": sum(value["count"] for value in replacement_items),
            "source_asset_record_count": assets_manifest.get("source_asset_record_count", 0),
            "deduplicated_asset_record_count": assets_manifest.get("deduplicated_asset_record_count", 0),
            "publication_image_count": assets_manifest.get("publication_image_count", 0),
            "publication_image_count_exported": assets_manifest.get("publication_image_count_exported", 0),
            "scan_page_count_detected": assets_manifest.get("scan_page_count_detected", 0),
            "scan_page_count_exported": 0,
            "final_epub_expected_image_count": assets_manifest.get("final_epub_expected_image_count", 0),
            "framework_image_count": len(framework_inventory.get("images", [])),
        },
        "blockers": blockers,
        "chapter_candidates": candidates,
        "chapter_title_glued_to_body": title_glue,
        "overlong_xhtml": overlong_xhtml,
        "image_boundary_issues": image_boundary_issues,
        "adjacent_longest_overlaps": adjacent_overlaps,
        "three_consecutive_short_fragments": three_short_fragments,
        "status_bar_single_line_candidates": status_one_line,
        "abnormal_latin_or_garbage": abnormal_latin,
        "placeholder_square_items": placeholder_items,
        "replacement_character_items": replacement_items,
        "metadata_cover_conflicts": _metadata_cover_conflicts(primary_doc, assets_manifest, framework_inventory),
        "publication_preflight": {
            "chapter_count": deterministic.chapter_count,
            "chapter_like_count": deterministic.chapter_like_count,
            "image_count": deterministic.image_count,
            "unplaced_image_count": deterministic.unplaced_image_count,
            "duplicate_runs": [asdict(value) for value in deterministic.duplicate_runs],
            "quote_imbalance_count": deterministic.quote_imbalance_count,
            "embedded_dialogue_count": deterministic.embedded_dialogue_count,
            "text_issues": [issue.to_dict() for issue in deterministic.text_issues],
            "critical_messages": deterministic.critical_messages,
            "warning_messages": deterministic.warning_messages,
        },
    }


def _metadata_cover_conflicts(primary_doc: UnifiedDocument, assets_manifest: dict, framework_inventory: dict) -> list[dict]:
    conflicts: list[dict] = []
    cover_assets = [item for item in assets_manifest.get("assets", []) if item.get("role") == "cover" and item.get("available")]
    framework_cover = [item for item in framework_inventory.get("manifest", []) if "cover-image" in str(item.get("properties", ""))]
    if len(cover_assets) != 1:
        conflicts.append({"code": "source_cover_count", "count": len(cover_assets)})
    if len(framework_cover) != 1:
        conflicts.append({"code": "framework_cover_count", "count": len(framework_cover)})
    title = str(getattr(primary_doc.metadata, "title", "") or "")
    opf_titles = list((framework_inventory.get("metadata") or {}).get("title", []))
    if title and opf_titles and title not in opf_titles:
        conflicts.append({"code": "title_mismatch", "document_title": title, "opf_titles": opf_titles})
    return conflicts


def _risk_queue(repair_map: dict) -> dict:
    raw_items = list(repair_map.get("items") or [])
    risk_items = sorted(
        [item for item in raw_items if str(item.get("risk_level", "none")) != "none"],
        key=repair._publication_review_priority,
    )
    queue = []
    for index, item in enumerate(risk_items, start=1):
        queue.append({
            "priority": index,
            "item_id": item.get("item_id"),
            "chapter_id": item.get("chapter_id"),
            "chapter_title": item.get("chapter_title"),
            "reading_order": item.get("reading_order"),
            "page": item.get("page"),
            "risk_level": item.get("risk_level"),
            "risk_reasons": item.get("risk_reasons", []),
            "evidence_tier": item.get("evidence_tier", ""),
            "evidence_triggers": item.get("evidence_triggers", []),
            "baseline_text": item.get("baseline_text", ""),
            "context_before": item.get("context_before", ""),
            "context_after": item.get("context_after", ""),
            "candidate_consensus": item.get("candidate_consensus", {}),
        })
    return {"schema": RISK_SCHEMA, "count": len(queue), "items": queue}


def _boundary_windows(repair_map: dict, assets_manifest: dict, *, package_mode: str = "forensic") -> dict:
    return v3.build_dynamic_boundary_windows(
        list(repair_map.get("items") or []), assets_manifest,
        compact=str(package_mode or "forensic").lower() == "compact",
    )



def _normalise_variant(value: str) -> str:
    return re.sub(r"[・ー\s　]+", "", value or "").casefold()


def _similar_variant_groups(counter: Counter[str]) -> list[dict]:
    terms = [term for term, count in counter.items() if count >= 1 and 3 <= len(term) <= 24]
    groups: list[dict] = []
    used: set[str] = set()
    for index, term in enumerate(terms):
        if term in used:
            continue
        variants = []
        left = _normalise_variant(term)
        for other in terms[index + 1:]:
            if other in used:
                continue
            right = _normalise_variant(other)
            if not left or not right or abs(len(left) - len(right)) > 2:
                continue
            ratio = SequenceMatcher(None, left, right, autojunk=False).ratio()
            if ratio >= 0.78 and term != other:
                variants.append({"term": other, "count": counter[other], "similarity": round(ratio, 3)})
        if variants:
            used.add(term)
            used.update(item["term"] for item in variants)
            groups.append({"canonical_candidate": term, "count": counter[term], "variants": variants})
    return groups[:500]


def _term_consistency(
    repair_map: dict,
    full_items_by_id: dict[str, dict] | None = None,
    reference_alignment: Sequence[dict] | None = None,
) -> dict:
    return v3.build_term_graph(repair_map, full_items_by_id or {}, reference_alignment)



def _style_profile(
    repair_map: dict,
    framework_inventory: dict,
    vertical: bool,
    stable_records: Sequence[dict] | None = None,
) -> dict:
    items = list(repair_map.get("items") or [])
    texts = [str(item.get("baseline_text", "") or "") for item in items]
    stable = list(stable_records or [])
    dialogue = sum(1 for value in texts if value.strip().startswith("「"))
    indented = sum(1 for value in texts if value.startswith(("　", " ")))
    multiline = sum(1 for value in texts if "\n" in value)
    special = [
        {
            "item_id": record.get("item_id"),
            "layout_type": record.get("layout_type"),
            "line_groups": record.get("line_groups", []),
            "narrative_resume_item_id": record.get("narrative_resume_item_id"),
            "must_not_merge_with_narrative": record.get("must_not_merge_with_narrative", False),
            "preserve_line_breaks": record.get("preserve_line_breaks", False),
        }
        for record in stable if record.get("layout_type")
    ]
    return {
        "schema": STYLE_SCHEMA,
        "writing_mode": "vertical-rl" if vertical else "horizontal-tb",
        "css_policy": {
            "lock_background_color": False,
            "lock_text_color": False,
            "lock_body_font_size": False,
            "dark_mode_compatible": True,
            "reader_font_scaling_compatible": True,
        },
        "paragraph_statistics": {
            "item_count": len(items),
            "dialogue_item_count": dialogue,
            "leading_indent_item_count": indented,
            "multiline_item_count": multiline,
            "special_layout_item_count": len(special),
        },
        "annotation_policy": "ocr_text_base_only; evidence_backed_publication_ruby_allowed; never_infer_readings",
        "special_layout_blocks": special,
        "framework_css_inventory": framework_inventory.get("css", []),
        "recommended_base_css": "Use writing mode, margins, line height and indentation only; do not force colors or body font size. Ruby is allowed only when an exported reference alignment supplies matching base text and a non-empty reading.",
    }



_FALLBACK_CHAPTER_TITLE_RE = re.compile(
    r"^(?P<title>(?:序章|プロローグ|フロローグ|ブロローグ|エピローグ|終章|幕間|間章|"
    r"第[0-9０-９一二三四五六七八九十百千万〇零]+[章話節部巻]|"
    r"[0-9０-９一二三四五六七八九十百]{1,6}(?:章|話|節)))"
    r"(?:[\s　:：―—─\-]|$)"
)
_STANDALONE_CHAPTER_NUMBER_RE = re.compile(r"^[0-9０-９]{1,3}[.．。]?")
_GLUED_CHAPTER_PREFIX_RE = re.compile(
    r"^(?P<title>プロローグ|フロローグ|フロローク|ブロローグ|序章|エピローグ|終章|幕間|間章|"
    r"第[0-9０-９一二三四五六七八九十百千万〇零]+[章話節部巻])"
    r"(?=[ぁ-んァ-ヶ一-龥「『])"
)


def _chapter_marker(item: dict) -> tuple[str, str] | None:
    block_type = str(item.get("block_type", "") or "")
    text = str(item.get("proposed_text", item.get("baseline_text", "")) or "").strip()
    candidate = item.get("chapter_candidate") if isinstance(item.get("chapter_candidate"), dict) else {}
    candidate_title = str(candidate.get("title", candidate.get("text", "")) or item.get("chapter_title", "") or "").strip()
    candidate_id = str(item.get("chapter_candidate_id", "") or candidate.get("candidate_id", "") or "").strip()
    if block_type in {BlockType.CHAPTER.value, BlockType.SECTION.value}:
        title = candidate_title or text[:160] or "章"
        return candidate_id or f"chapter-marker-{item.get('reading_order', 0)}", title
    candidate_boundary = bool(candidate) or bool(item.get("is_chapter_start")) or "chapter_type_repair" in (item.get("structure_reasons") or [])
    if candidate_id and candidate_title and candidate_boundary:
        return candidate_id, candidate_title
    source_marker = item.get("source_column_chapter_marker") if isinstance(item.get("source_column_chapter_marker"), dict) else {}
    if source_marker.get("title"):
        return str(source_marker.get("candidate_id") or f"source-column-chapter-{item.get('reading_order', 0)}"), str(source_marker["title"])
    first_line = text.splitlines()[0].strip() if text else ""
    glued = _GLUED_CHAPTER_PREFIX_RE.match(first_line)
    if glued:
        title = glued.group("title")
        explicit_glue = "chapter_title_glued_to_body" in set(item.get("risk_flags") or []) | set(item.get("structure_reasons") or [])
        prologue_at_start = int(item.get("reading_order", 0) or 0) == 0 and title in {"プロローグ", "フロローグ", "フロローク", "ブロローグ", "序章"}
        if explicit_glue or prologue_at_start:
            return candidate_id or f"chapter-marker-{item.get('reading_order', 0)}", title
    match = _FALLBACK_CHAPTER_TITLE_RE.match(first_line)
    if match:
        title = match.group("title")
        return candidate_id or f"chapter-marker-{item.get('reading_order', 0)}", title
    if looks_like_chapter_title(first_line):
        return candidate_id or f"chapter-marker-{item.get('reading_order', 0)}", first_line[:160]
    # A bare number in body OCR is frequently a page number, status value, or
    # a damaged short utterance.  Accept it only when an upstream chapter
    # boundary/candidate explicitly confirms the role; physical leading-column
    # numbers are handled above by ``source_column_chapter_marker``.
    if (
        _STANDALONE_CHAPTER_NUMBER_RE.fullmatch(first_line)
        and len(text) <= 6
        and (candidate_boundary or bool(item.get("is_chapter_start")))
    ):
        return candidate_id or f"chapter-marker-{item.get('reading_order', 0)}", first_line
    return None


def _chapters_from_items(items: Sequence[dict]) -> list[dict]:
    ordered = sorted(items, key=lambda value: int(value.get("reading_order", value.get("row_index", 0)) or 0))
    explicit_ids = [str(item.get("chapter_id", "") or "") for item in ordered]
    unique_explicit = list(dict.fromkeys(value for value in explicit_ids if value))
    marker_count_hint = sum(_chapter_marker(item) is not None for item in ordered)
    synthetic_single = len(unique_explicit) == 1 and unique_explicit[0] in {"front-matter", "chapter-001", "chapter-1"}
    # Existing NAV/NCX and confirmed GUI chapter IDs are authoritative.  A
    # single generic chapter-001 emitted by an earlier OCR pass is treated as
    # synthetic only when multiple independent chapter/title markers exist;
    # that is the real-world failure mode where a 4,000-item book collapsed to
    # one logical chapter.  Genuine one-chapter books and legacy fixtures with
    # zero/one marker keep their explicit ID.
    if unique_explicit and not (synthetic_single and marker_count_hint > 1):
        by_id: dict[str, list[dict]] = defaultdict(list)
        titles: dict[str, str] = {}
        order: list[str] = []
        for item in ordered:
            chapter_id = str(item.get("chapter_id", "front-matter") or "front-matter")
            if chapter_id not in by_id:
                order.append(chapter_id)
            by_id[chapter_id].append(item)
            titles[chapter_id] = str(item.get("chapter_title", "") or titles.get(chapter_id, ""))
        return [{
            "number": index, "chapter_id": chapter_id, "chapter_title": titles.get(chapter_id, ""),
            "items": by_id[chapter_id], "detection_source": "confirmed_chapter_id",
        } for index, chapter_id in enumerate(order, start=1)]

    groups: list[dict] = []
    current: dict | None = None
    marker_count = 0
    for item in ordered:
        marker = _chapter_marker(item)
        if marker:
            marker_count += 1
            marker_id, title = marker
            current = {
                "number": len(groups) + 1,
                "chapter_id": marker_id or f"chapter-{len(groups)+1:03d}",
                "chapter_title": title,
                "items": [],
                "detection_source": "block_or_title_fallback",
            }
            groups.append(current)
        if current is None:
            current = {
                "number": 1, "chapter_id": "front-matter", "chapter_title": "",
                "items": [], "detection_source": "fallback_front_matter",
            }
            groups.append(current)
        current["items"].append(item)
    if not groups:
        return []
    # If the first marker was the first item, avoid an empty front-matter group.
    groups = [group for group in groups if group.get("items")]
    for index, group in enumerate(groups, start=1):
        group["number"] = index
        group["chapter_detection_warning"] = bool(len(ordered) > 500 and marker_count == 0)
    return groups


def _chapter_sequence_diagnostics(chapter_manifest: Sequence[dict]) -> dict:
    """Describe numeric chapter coverage without inventing missing boundaries."""
    observed: list[int] = []
    has_prologue = False
    for chapter in chapter_manifest or []:
        title = str(chapter.get("chapter_title", "") or "").strip()
        if title in {"序章", "プロローグ", "フロローグ", "フロローク", "ブロローグ"}:
            has_prologue = True
            continue
        match = re.fullmatch(r"(?:第)?([0-9０-９]{1,3})(?:章|話|節)?", title)
        if not match:
            continue
        observed.append(int(match.group(1).translate(str.maketrans("０１２３４５６７８９", "0123456789"))))
    unique = list(dict.fromkeys(observed))
    monotonic = all(right > left for left, right in zip(unique, unique[1:]))
    missing: list[int] = []
    if len(unique) >= 3 and max(unique) <= 999:
        missing = [value for value in range(1, max(unique) + 1) if value not in set(unique)]
    return {
        "has_prologue": has_prologue,
        "observed_numeric_chapters": unique,
        "missing_numeric_chapters": missing,
        "numeric_sequence_monotonic": monotonic,
        "sequence_complete": bool(unique) and monotonic and not missing,
        "requires_review": bool(len(unique) >= 3 and (missing or not monotonic)),
    }

def _physical_leading_chapter_options(source: dict) -> tuple[list[int], list[dict]]:
    options: list[int] = []
    evidence: list[dict] = []
    for candidate in source.get("physical_column_candidates") or []:
        if not isinstance(candidate, dict):
            continue
        texts = candidate.get("column_texts") if isinstance(candidate.get("column_texts"), list) else []
        if len(texts) < 2:
            continue
        token = str(texts[0] or "").strip()
        match = re.fullmatch(r"([0-9０-９]{1,2})[.．。]?", token)
        if not match:
            continue
        number = int(match.group(1).translate(str.maketrans("０１２３４５６７８９", "0123456789")))
        if not 1 <= number <= 99:
            continue
        options.append(number)
        evidence.append({
            "model": candidate.get("model_label", candidate.get("source_engine", "")),
            "token": token,
            "number": number,
        })
    # A standalone edited title is also evidence, but never enough by itself
    # to split a large book unless the sequence contains several markers.
    text = str(source.get("edited_text", source.get("original_fused_text", "")) or "").strip()
    match = re.fullmatch(r"([0-9０-９]{1,2})[.．。]?", text)
    if match:
        number = int(match.group(1).translate(str.maketrans("０１２３４５６７８９", "0123456789")))
        if 1 <= number <= 99:
            options.append(number)
            evidence.append({"model": "current_edited_text", "token": text, "number": number})
    return sorted(set(options)), evidence


def _select_monotonic_source_chapter_markers(source_items: Sequence[dict]) -> dict[str, dict]:
    probes: list[tuple[int, str, list[int], list[dict]]] = []
    for order, source in enumerate(source_items):
        if not isinstance(source, dict):
            continue
        item_id = str(source.get("row_id") or source.get("item_id") or "")
        options, evidence = _physical_leading_chapter_options(source)
        if item_id and options:
            probes.append((order, item_id, options, evidence))
    selected: dict[str, dict] = {}
    previous = 0
    for order, item_id, options, evidence in probes:
        if previous == 0:
            viable = [value for value in options if 1 <= value <= 5]
        else:
            viable = [value for value in options if previous < value <= previous + 5]
        if not viable:
            continue
        number = min(viable)
        previous = number
        selected[item_id] = {
            "candidate_id": f"source-column-chapter-{number:03d}-{order:05d}",
            "title": str(number),
            "number": number,
            "source": "multi_model_physical_leading_column",
            "evidence": evidence,
            "confidence": 0.96 if len({value.get('model') for value in evidence if value.get('number') in {number, int(str(number)[-1])}}) >= 2 else 0.82,
        }
    # One or two isolated numbers are much more likely to be page/status noise.
    return selected if len(selected) >= 3 else {}


def _enrich_repair_items(primary_doc: UnifiedDocument, repair_map: dict, package: dict | None = None) -> None:
    candidates = list(repair_map.get("chapter_candidates") or [])
    candidate_by_item = {str(value.get("item_id", "") or ""): value for value in candidates if value.get("item_id")}
    source_by_id = {
        str(value.get("row_id") or value.get("item_id") or ""): value
        for value in ((package or {}).get("editable_items") or []) if isinstance(value, dict)
    }
    chapter_ids = {str(value.get("chapter_id", "") or "") for value in repair_map.get("items", []) if str(value.get("chapter_id", "") or "")}
    source_markers = {}
    if len(candidates) <= 1 and len(chapter_ids) <= 1 and source_by_id:
        source_markers = _select_monotonic_source_chapter_markers(list(source_by_id.values()))
    for item in repair_map.get("items", []):
        item_id = str(item.get("item_id", "") or "")
        item.setdefault("row_id", item_id)
        indices = _item_source_indices(item)
        source = primary_doc.blocks[indices[0]] if indices and 0 <= indices[0] < len(primary_doc.blocks) else None
        item.setdefault("block_id", str(getattr(source, "id", "") or ""))
        candidate = candidate_by_item.get(item_id) or item.get("chapter_candidate") or {}
        item.setdefault("chapter_candidate_id", str(candidate.get("candidate_id", "") or item.get("chapter_id", "") or ""))
        if item_id in source_markers:
            item["source_column_chapter_marker"] = copy.deepcopy(source_markers[item_id])
            item["structure_reasons"] = list(dict.fromkeys([
                *(item.get("structure_reasons") or []), "chapter_marker_recovered_from_physical_column",
            ]))



def _stable_text_records(repair_map: dict, assets_manifest: dict, chapter_manifest: Sequence[dict] | None = None) -> list[dict]:
    """Freeze the current GUI-edited text as the sole default target.

    Character fusion, AI adjudication and individual model outputs remain
    evidence only.  Export must never re-decide or silently replace text that
    the user already accepted in the current comparison workspace.
    """
    del chapter_manifest
    before_assets: dict[str, list[str]] = defaultdict(list)
    after_assets: dict[str, list[str]] = defaultdict(list)
    for asset in assets_manifest.get("assets", []):
        resource = str(asset.get("resource_path", "") or "")
        if not resource:
            continue
        before = str(asset.get("stable_item_before", "") or "")
        after = str(asset.get("stable_item_after", "") or "")
        if before:
            after_assets[before].append(resource)
        if after:
            before_assets[after].append(resource)
    records: list[dict] = []
    for item in sorted(repair_map.get("items", []), key=lambda value: int(value.get("reading_order", value.get("row_index", 0)) or 0)):
        item_id = str(item.get("item_id", "") or "")
        # ai_repair_epub._map_item creates baseline_text directly from the
        # current edited_text.  It is the unique target authority here.
        baseline = str(item.get("baseline_text", "") or "")
        proposed = baseline
        character_fused = str(item.get("character_fused_text", "") or "")
        ai_text = str(item.get("ai_adjudicated_text", "") or "")
        has_ai = bool(item.get("has_ai_adjudication") and ai_text)
        chapter_id = str(item.get("chapter_id", "front-matter") or "front-matter")
        consensus = item.get("candidate_consensus") or {}
        image_before = before_assets.get(item_id) or []
        image_after = after_assets.get(item_id) or []
        source_indices = _item_source_indices(item)
        source_columns = [str(value) for value in (item.get("column_ids") or []) if str(value)]
        source_block_ids = [str(value) for value in (item.get("source_block_ids") or [item.get("block_id")]) if str(value)]
        source_block_complete = bool(source_indices or source_block_ids)
        physical_complete = bool(source_columns or str(item.get("block_type", "") or "") in {BlockType.CHAPTER.value, BlockType.FOOTNOTE.value})
        item["physical_column_coverage_complete"] = physical_complete
        item["source_block_coverage_complete"] = source_block_complete
        policy = v3.decide_edit_policy(
            item,
            proposed,
            image_anchor_before=image_before,
            image_anchor_after=image_after,
            physical_column_coverage_complete=physical_complete,
            source_block_coverage_complete=source_block_complete,
        )
        layout = policy.get("layout_structure") or {}
        record = {
            "schema": STABLE_TEXT_MAP_SCHEMA,
            "reading_order": int(item.get("reading_order", item.get("row_index", 0)) or 0),
            "item_id": item_id,
            "row_id": str(item.get("row_id", "") or item_id),
            "block_id": str(item.get("block_id", "") or ""),
            "page": int(item.get("page", 0) or 0),
            "chapter_candidate_id": str(item.get("chapter_candidate_id", "") or chapter_id),
            "chapter_id": chapter_id,
            "chapter_title": str(item.get("chapter_title", "") or ""),
            "technical_part_index": int(item.get("technical_part_index", 1) or 1),
            "block_type": str(item.get("block_type", "paragraph") or "paragraph"),
            "primary_block_index": source_indices[0] if source_indices else item.get("primary_block_index"),
            "primary_block_indices": source_indices,
            "source_block_ids": source_block_ids,
            "source_column_ids": source_columns,
            "source_block_coverage_complete": source_block_complete,
            "physical_column_coverage_complete": physical_complete,
            "baseline_text_sha256": _sha256_bytes(baseline.encode("utf-8")),
            "current_edited_text_sha256": _sha256_bytes(baseline.encode("utf-8")),
            "auto_fused_text_sha256": _sha256_bytes(str(item.get("auto_fused_text", "") or baseline).encode("utf-8")),
            "character_fused_text": character_fused or None,
            "character_fused_text_sha256": _sha256_bytes(character_fused.encode("utf-8")) if character_fused else None,
            "character_fused_text_is_evidence_only": True,
            "character_fusion_confidence": float(item.get("character_fusion_confidence", 0.0) or 0.0),
            "has_ai_adjudication": has_ai,
            "ai_adjudicated_text": ai_text or None,
            "ai_adjudicated_text_sha256": _sha256_bytes(ai_text.encode("utf-8")) if ai_text else None,
            "ai_adjudicated_text_is_evidence_only": True,
            "source_text_omitted_from_target_map": False,
            "baseline_is_reference_only": False,
            "current_edited_text_is_unique_default_authority": True,
            "proposed_text": proposed,
            "proposed_text_source": "current_edited_text",
            "proposed_text_sha256": _sha256_bytes(proposed.encode("utf-8")),
            "recommended_text": proposed,
            "recommended_text_sha256": _sha256_bytes(proposed.encode("utf-8")),
            "final_text": None,
            "final_text_requires_model_decision": bool(policy.get("model_action_required")),
            "exported_map_role": "stable_target_with_edit_policy",
            "source_text_authority": FULL_FUSION_FILENAME,
            "target_text_authority": "12_stable_text_map.jsonl",
            "source_evidence_item_id": item_id,
            "edit_policy": policy.get("edit_policy"),
            "model_action_required": policy.get("model_action_required"),
            "text_locked": policy.get("text_locked"),
            "unlock_reasons": policy.get("unlock_reasons", []),
            "lock_validation": policy.get("lock_validation", {}),
            "allowed_unlock_reasons": [
                "model_candidate_disagreement", "low_character_fusion_confidence", "context_contradiction",
                "status_bar_glue", "cross_page_unfinished_sentence", "global_duplicate_or_move",
                "abnormal_latin_numeric_or_control", "publication_reference_conflict", "visual_evidence_conflict",
                "physical_column_coverage_incomplete", "source_block_coverage_incomplete",
            ],
            "structure_reasons": list(dict.fromkeys([
                *(policy.get("structure_reasons", []) or []), *(item.get("structure_reasons", []) or []),
            ])),
            "source_column_chapter_marker": copy.deepcopy(item.get("source_column_chapter_marker")) if item.get("source_column_chapter_marker") else None,
            "confidence": _confidence_band(item.get("confidence", 0.0)),
            "confidence_score": float(item.get("confidence", 0.0) or 0.0),
            "risk_level": str(item.get("risk_level", "none") or "none"),
            "risk_flags": list(item.get("risk_reasons") or []),
            "evidence_tier": str(item.get("evidence_tier", "") or ""),
            "candidate_storage": str(item.get("candidate_storage", "") or item.get("evidence_tier", "") or ""),
            "consensus_type": str(item.get("consensus_type", "") or consensus.get("consensus_type", "partial") or "partial"),
            "candidate_consensus": copy.deepcopy(consensus),
            "image_anchor_before": image_before or None,
            "image_anchor_after": image_after or None,
            "expected_evidence_path": str(item.get("expected_evidence_path", "") or ""),
            "expected_reading_path": str(item.get("expected_reading_path", "") or ""),
            "framework_epub_target": str(item.get("framework_epub_target", "") or item.get("epub_target", "") or ""),
            "final_html_id": str(item.get("final_html_id", "") or _final_html_id(item)),
            "planned_final_xhtml": str(item.get("planned_final_xhtml", "") or ""),
            "planned_final_target": str(item.get("planned_final_target", "") or ""),
            "expected_epub_target": str(item.get("planned_final_target", "") or item.get("epub_target", "") or ""),
            "plain_text": proposed,
            "inline_tokens": [{"type": "text", "value": proposed}],
            "ruby_status": "base_text_only_no_reliable_reference",
            "ruby_group_count": 0,
            "layout_type": layout.get("layout_type"),
            "line_groups": layout.get("line_groups", []),
            "narrative_suffix": layout.get("narrative_suffix"),
            "narrative_suffix_char_start": layout.get("narrative_suffix_char_start"),
            "must_split_suffix_to_next_paragraph": bool(layout.get("must_split_suffix_to_next_paragraph", False)),
            "narrative_resume_item_id": layout.get("narrative_resume_item_id"),
            "must_not_merge_with_narrative": bool(layout.get("must_not_merge_with_narrative", False)),
            "preserve_line_breaks": bool(layout.get("preserve_line_breaks", False)),
        }
        records.append(record)
    return records


def build_compact_stable_record(record: dict) -> dict:
    """Return the minimum executable stable-map record without changing text."""
    result = {
        "item_id": record.get("item_id"),
        "row_id": record.get("row_id") or record.get("item_id"),
        "reading_order": record.get("reading_order"),
        "page": record.get("page"),
        "chapter_id": record.get("chapter_id"),
        "chapter_title": record.get("chapter_title"),
        "block_type": record.get("block_type"),
        "proposed_text": str(record.get("proposed_text", "") or ""),
        "proposed_text_sha256": record.get("proposed_text_sha256"),
        "edit_policy": record.get("edit_policy"),
        "source_block_ids": copy.deepcopy(record.get("source_block_ids") or []),
        "source_column_ids": copy.deepcopy(record.get("source_column_ids") or []),
        "final_html_id": record.get("final_html_id"),
        "planned_final_xhtml": record.get("planned_final_xhtml"),
        "planned_final_target": record.get("planned_final_target"),
    }
    if record.get("inline_tokens") and any(token.get("type") == "ruby" for token in record.get("inline_tokens") or [] if isinstance(token, dict)):
        result["inline_tokens"] = copy.deepcopy(record.get("inline_tokens"))
        result["ruby_status"] = record.get("ruby_status")
    if str(record.get("edit_policy", "") or "") != "locked_consensus":
        for key in (
            "unlock_reasons", "risk_flags", "layout_type", "line_groups", "narrative_suffix",
            "must_split_suffix_to_next_paragraph", "lock_validation", "page_review_required",
            "page_review_reasons",
        ):
            if record.get(key) not in (None, [], {}, False, ""):
                result[key] = copy.deepcopy(record.get(key))
    return result


def build_compact_atomic_span(span: dict) -> dict:
    result = {
        "source_span_id": span.get("source_span_id"),
        "expected_item_id": span.get("expected_item_id"),
        "source_type": span.get("source_scope", span.get("source_type")),
        "source_ref": span.get("physical_column_id") or span.get("source_block_id") or span.get("source_span_id"),
        "physical_column_id": span.get("physical_column_id"),
        # source_order is global within the frozen item.  source_char_start may
        # reset to zero for every physical column and therefore cannot be used
        # alone to audit cross-column order.
        "source_order": span.get("source_order"),
        "source_text_sha256": (
            span.get("selected_source_text_sha256")
            or span.get("source_text_sha256")
            or span.get("text_sha256")
        ),
        "coverage_policy": span.get("coverage_policy"),
        "source_char_start": span.get("source_char_start"),
        "source_char_end": span.get("source_char_end"),
    }
    # Executable validation needs the reliable source text; unverified spans do
    # not repeat proposed text and therefore remain tiny.
    if str(span.get("coverage_policy", "") or "") == "exactly_once":
        value = str(span.get("selected_source_text", span.get("source_text", "")) or "")
        if value:
            result["selected_source_text"] = value
    return result


def _write_stable_text_map(path: Path, records: Sequence[dict], *, compact: bool = False, atomic: bool = False) -> None:
    if compact:
        projection = [build_compact_atomic_span(value) if atomic else build_compact_stable_record(value) for value in records]
    else:
        projection = list(records)
    path.write_text(
        "\n".join(
            json.dumps(_redact_absolute_paths(record), ensure_ascii=False, separators=(",", ":"))
            for record in projection
        ) + ("\n" if projection else ""),
        encoding="utf-8",
    )


def _deduplicated_structure_document(package: dict, stable_records: Sequence[dict], assets_manifest: dict) -> dict:
    """Keep deterministic structure while storing editable text in one map."""
    structure = copy.deepcopy(package.get("structure_document") or {})
    by_block_id: dict[str, list[str]] = defaultdict(list)
    by_block_index: dict[int, list[str]] = defaultdict(list)
    for record in stable_records:
        item_id = str(record.get("item_id", "") or "")
        block_id = str(record.get("block_id", "") or "")
        if block_id and item_id:
            by_block_id[block_id].append(item_id)
        for block_index in record.get("primary_block_indices") or []:
            try:
                by_block_index[int(block_index)].append(item_id)
            except (TypeError, ValueError):
                continue
    source_to_asset: dict[str, dict] = {}
    for asset in assets_manifest.get("assets", []):
        for source in asset.get("source_records") or []:
            key = _normalised_file_key(source.get("source_path", ""))
            if key:
                source_to_asset[key] = asset
        key = _normalised_file_key(asset.get("source_path", ""))
        if key:
            source_to_asset.setdefault(key, asset)

    blocks = structure.get("blocks") if isinstance(structure.get("blocks"), list) else []
    uncovered_text_blocks: list[dict] = []
    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue
        block_id = str(block.get("id", "") or "")
        item_ids = list(dict.fromkeys(by_block_id.get(block_id, []) + by_block_index.get(index, [])))
        original_type = str(block.get("type", "") or "")
        block.pop("text", None)
        block.pop("ocr_raw", None)
        if item_ids:
            block["stable_item_ids"] = item_ids
            block["text_source"] = "12_stable_text_map.jsonl"
        elif original_type not in {BlockType.IMAGE_REF.value, BlockType.HEADER_FOOTER.value, ""}:
            uncovered_text_blocks.append({"block_index": index, "block_id": block_id, "block_type": original_type})
        image_key = _normalised_file_key(block.get("image_path", ""))
        if image_key:
            asset = source_to_asset.get(image_key)
            block["asset_id"] = (asset or {}).get("asset_id")
            block["image_path"] = (asset or {}).get("export_path", "")
            if asset and not asset.get("include_in_ai_package"):
                block["image_export_status"] = "intentionally_omitted"

    pages = structure.get("pages") if isinstance(structure.get("pages"), list) else []
    for page in pages:
        if not isinstance(page, dict):
            continue
        image_key = _normalised_file_key(page.get("image_path", ""))
        if not image_key:
            continue
        asset = source_to_asset.get(image_key)
        page["asset_id"] = (asset or {}).get("asset_id")
        page["image_path"] = (asset or {}).get("export_path", "")
        page["image_export_status"] = (asset or {}).get("export_status", "unmapped")

    structure["text_storage"] = {
        "stable_text_map": "12_stable_text_map.jsonl",
        "atomic_span_map": "13_atomic_span_map.jsonl",
        "reading_text": "reading/full_book_with_item_markers.txt",
        "chapter_evidence": "evidence/chapter_*.jsonl",
        "body_text_intentionally_deduplicated": True,
    }
    structure["stable_mapping_audit"] = {
        "source_text_block_count": sum(1 for block in blocks if isinstance(block, dict) and str(block.get("type", "") or "") not in {BlockType.IMAGE_REF.value, BlockType.HEADER_FOOTER.value, ""}),
        "uncovered_text_blocks": uncovered_text_blocks,
        "uncovered_text_block_count": len(uncovered_text_blocks),
        "block_index_fallback_enabled": True,
    }
    return structure



def _deduplicated_format_manifest(package: dict, assets_manifest: dict) -> dict:
    manifest = copy.deepcopy(package.get("format_manifest") or {})
    source_to_asset: dict[str, dict] = {}
    for asset in assets_manifest.get("assets", []):
        for source in asset.get("source_records") or []:
            key = _normalised_file_key(source.get("source_path", ""))
            if key:
                source_to_asset[key] = asset
    for collection_name in ("page_order", "block_order"):
        collection = manifest.get(collection_name)
        if not isinstance(collection, list):
            continue
        for entry in collection:
            if not isinstance(entry, dict):
                continue
            key = _normalised_file_key(entry.get("image_path", ""))
            if not key:
                continue
            asset = source_to_asset.get(key)
            entry["asset_id"] = (asset or {}).get("asset_id")
            entry["image_path"] = (asset or {}).get("export_path", "")
            entry["image_export_status"] = (asset or {}).get("export_status", "unmapped")
    manifest["text_storage"] = "12_stable_text_map.jsonl"
    return manifest

def _technical_item_parts(
    items: Sequence[dict],
    *,
    split_before_item_ids: set[str] | None = None,
    split_after_item_ids: set[str] | None = None,
) -> list[list[dict]]:
    split_before_item_ids = set(split_before_item_ids or set())
    split_after_item_ids = set(split_after_item_ids or set())
    parts: list[list[dict]] = []
    current: list[dict] = []
    characters = 0
    for item in items:
        item_id = str(item.get("item_id", "") or "")
        text_length = len(str(item.get("baseline_text", "") or ""))
        if current and (
            item_id in split_before_item_ids
            or len(current) >= _MAX_ITEMS_PER_XHTML
            or characters + text_length > _MAX_TEXT_CHARACTERS_PER_XHTML
        ):
            parts.append(current)
            current = []
            characters = 0
        current.append(item)
        characters += text_length
        if item_id in split_after_item_ids:
            parts.append(current)
            current = []
            characters = 0
    if current:
        parts.append(current)
    return parts or [[]]



def _final_html_id(item: dict) -> str:
    order = int(item.get("reading_order", item.get("row_index", 0)) or 0)
    item_id = str(item.get("item_id", "") or "")
    suffix_match = re.search(r"([0-9a-fA-F]{8,})$", item_id)
    suffix = suffix_match.group(1)[:8].lower() if suffix_match else _sha256_bytes(item_id.encode("utf-8"))[:8]
    return f"item-{order:06d}-{suffix}"


def _validate_lossless_fusion_package(package: dict) -> dict:
    """Reject a publication export when the canonical fusion source is incomplete."""
    try:
        # Validate schema, mode, structure_document and layout hashes.  The GUI
        # may append cover OCR candidates or update review fields after the
        # original seal, so stale package-seal hashes are reported rather than
        # treated as evidence loss.
        roundtrip._validate_common(
            package,
            expected_mode=roundtrip.MODE_MULTI,
            validate_editable_structure=False,
            validate_immutable_manifest=False,
        )
    except Exception as exc:
        raise repair.AIRepairEpubError(f"完整融合 JSON 结构完整性校验失败：{exc}") from exc
    items = list(package.get("editable_items") or [])
    model_count = len(package.get("model_sources") or package.get("model_labels") or [])
    if model_count < 2:
        raise repair.AIRepairEpubError("完整融合 JSON 至少需要两路 OCR 模型证据。")
    missing_candidates: list[str] = []
    missing_physical: list[str] = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        item_id = str(raw.get("row_id") or raw.get("item_id") or "")
        if len(raw.get("candidates") or []) != model_count:
            missing_candidates.append(item_id)
        if len(raw.get("physical_column_candidates") or []) != model_count:
            missing_physical.append(item_id)
    if missing_candidates or missing_physical:
        raise repair.AIRepairEpubError(
            "完整融合 JSON 缺少逐模型证据："
            f"候选不完整 {len(missing_candidates)} 条，物理列不完整 {len(missing_physical)} 条。"
        )
    editable_expected = str(package.get("editable_structure_sha256", "") or "")
    immutable_expected = str(package.get("immutable_manifest_sha256", "") or "")
    editable_actual = roundtrip._editable_structure_hash(items)
    immutable_actual = roundtrip._immutable_manifest_hash(package)
    return {
        "model_count": model_count,
        "item_count": len(items),
        "candidate_complete_count": len(items) - len(missing_candidates),
        "physical_column_complete_count": len(items) - len(missing_physical),
        "structure_and_layout_valid": True,
        "editable_structure_seal_present": bool(editable_expected),
        "editable_structure_seal_matches": bool(editable_expected and editable_expected == editable_actual),
        "immutable_manifest_seal_present": bool(immutable_expected),
        "immutable_manifest_seal_matches": bool(immutable_expected and immutable_expected == immutable_actual),
        "seal_mismatch_is_not_evidence_loss": True,
    }


def _full_fusion_items_by_id(package: dict) -> dict[str, dict]:
    """Return the canonical, lossless OCR fusion records keyed by stable row ID.

    The round-trip package is the exact same object exported by the standalone
    “融合JSON” action.  It contains every model candidate and all physical
    column evidence.  Publication indexes may add navigation fields, but may
    never replace or weaken this source evidence.
    """
    result: dict[str, dict] = {}
    for raw in package.get("editable_items") or []:
        if not isinstance(raw, dict):
            continue
        item_id = str(raw.get("row_id") or raw.get("item_id") or "")
        if item_id:
            result[item_id] = copy.deepcopy(raw)
    return result


def _lossless_chapter_evidence_item(mapped: dict, source: dict | None, stable: dict) -> dict:
    """Merge lossless OCR evidence with one authoritative stable target text."""
    result = copy.deepcopy(source or {})
    item_id = str(stable.get("item_id", "") or mapped.get("item_id", "") or result.get("row_id", "") or result.get("item_id", ""))
    proposed = str(stable.get("proposed_text", "") or "")
    result["item_id"] = item_id
    result.setdefault("row_id", item_id)
    result["source_text_authority"] = "12_stable_text_map.jsonl"
    result["source_evidence_complete"] = bool(source)
    result["candidate_storage"] = "full_lossless"
    result["evidence_tier"] = "full_fusion"
    # Compatibility baseline is intentionally identical to proposed_text.
    # Older consumers may read it, but there is still only one target body.
    result["baseline_text"] = proposed
    result["baseline_text_sha256"] = str(stable.get("proposed_text_sha256", "") or _sha256_bytes(proposed.encode("utf-8")))
    result["proposed_text"] = proposed
    result["proposed_text_sha256"] = result["baseline_text_sha256"]
    result["proposed_text_source"] = "12_stable_text_map.jsonl:current_edited_text"
    result["source_baseline_text"] = str(mapped.get("baseline_text", "") or "")
    result["source_baseline_text_sha256"] = _sha256_bytes(result["source_baseline_text"].encode("utf-8"))
    result["character_fused_text_is_evidence_only"] = True
    result["ai_adjudicated_text_is_evidence_only"] = True
    result["edit_policy"] = stable.get("edit_policy")
    result["lock_validation"] = copy.deepcopy(stable.get("lock_validation") or {})
    result["auto_fused_text"] = str(result.get("edited_text", "") or result.get("original_fused_text", "") or proposed)
    result["ai_adjudicated_text"] = mapped.get("ai_adjudicated_text") or result.get("ai_adjudicated_text") or None
    result["has_ai_adjudication"] = bool(result.get("ai_adjudicated_text"))
    for key in (
        "reading_order", "paragraph_index", "spine_index", "chapter_id", "chapter_title",
        "chapter_candidate_id", "chapter_candidate", "epub_target", "html_id", "page",
        "block_type", "content_format", "column_ids", "baseline_tokens",
        "prev_item_id", "next_item_id", "context_before", "context_after", "risk_level",
        "risk_reasons", "disagreement_spans", "bbox", "final_html_id", "planned_final_xhtml",
        "planned_final_target", "framework_epub_target", "technical_part_index",
        "expected_evidence_path", "expected_reading_path",
    ):
        if key in stable:
            result[key] = copy.deepcopy(stable[key])
        elif key in mapped:
            result[key] = copy.deepcopy(mapped[key])
    for key in (
        "candidates", "physical_column_candidates", "column_geometry", "model_confidences",
        "alignment_status", "alignment_notes", "character_fusion_reason",
        "character_fusion_warnings", "character_fusion_evidence", "warnings",
        "recommended_model_index", "confidence", "reason", "local_reocr_recommended",
        "character_fused_text", "character_fusion_confidence", "original_fused_text",
        "edited_text", "delete_intentionally",
    ):
        if key not in result and key in mapped:
            result[key] = copy.deepcopy(mapped[key])
    result["final_text"] = None
    result["final_text_requires_model_decision"] = bool(stable.get("model_action_required"))
    return result



def build_chapter_evidence_projection(stable: dict, source: dict | None, full_index: int | None) -> dict:
    item_id = str(stable.get("item_id", "") or "")
    policy = str(stable.get("edit_policy", "") or "")
    result = {
        "item_id": item_id,
        "proposed_text_sha256": stable.get("proposed_text_sha256"),
        "edit_policy": policy,
        "full_evidence_ref": f"/editable_items/{full_index}" if full_index is not None else None,
        "page": stable.get("page"),
        "chapter_id": stable.get("chapter_id"),
        "planned_final_target": stable.get("planned_final_target"),
    }
    if policy != "locked_consensus":
        result["proposed_text"] = str(stable.get("proposed_text", "") or "")
        result["risk_reasons"] = list(dict.fromkeys([
            *(stable.get("risk_flags") or []), *(stable.get("unlock_reasons") or []),
        ]))
        result["candidate_summary"] = [
            {
                "model": value.get("model_label", value.get("source_engine", f"model-{index}")),
                "text": str(value.get("text", "") or ""),
                "confidence": value.get("confidence"),
            }
            for index, value in enumerate((source or {}).get("candidates") or []) if isinstance(value, dict)
        ]
    return result

def _write_reading_and_evidence(
    folder: Path,
    repair_map: dict,
    stable_records: list[dict],
    full_items_by_id: dict[str, dict] | None = None,
    assets_manifest: dict | None = None,
    *,
    package_mode: str = "forensic",
) -> tuple[list[dict], dict, dict]:
    """Write all human/model views from the already frozen stable records."""
    reading_dir = folder / "reading"
    evidence_dir = folder / "evidence"
    reading_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    assets_manifest = assets_manifest or {"assets": []}
    v3.enrich_asset_display_plan(assets_manifest)
    mapped_by_id = {str(item.get("item_id", "") or ""): item for item in repair_map.get("items") or []}
    compact_mode = str(package_mode or "forensic").lower() == "compact"
    full_index_by_id = {item_id: index for index, item_id in enumerate((full_items_by_id or {}).keys())}
    exported_assets = [
        asset for asset in assets_manifest.get("assets", [])
        if asset.get("include_in_final_epub") and asset.get("export_status") == "exported"
    ]
    split_before_ids = {
        str(asset.get("stable_item_after", "") or "") for asset in exported_assets
        if asset.get("force_split_before") and asset.get("stable_item_after")
    }
    split_after_ids = {
        str(asset.get("stable_item_before", "") or "") for asset in exported_assets
        if asset.get("force_split_after") and asset.get("stable_item_before")
    }
    assets_after: dict[str, list[dict]] = defaultdict(list)
    assets_before: dict[str, list[dict]] = defaultdict(list)
    for asset in exported_assets:
        before = str(asset.get("stable_item_before", "") or "")
        after = str(asset.get("stable_item_after", "") or "")
        if before:
            assets_after[before].append(asset)
        if after:
            assets_before[after].append(asset)

    chapters = _chapters_from_items(stable_records)
    full_parts: list[str] = []
    chapter_manifest: list[dict] = []
    spine_nodes: list[dict] = []
    placed_asset_ids: set[str] = set()
    authority_projection: list[dict] = []
    evidence_projection: list[dict] = []
    reading_projection: list[dict] = []
    for asset in exported_assets:
        role = str(asset.get("publication_role", "") or "")
        if role == "cover" or (not asset.get("stable_item_before") and not asset.get("stable_item_after") and role in {"frontispiece", "title_page", "toc_image", "character_sheet", "map"}):
            spine_nodes.append({
                "type": "standalone_image", "path": asset.get("planned_asset_xhtml"),
                "asset_id": asset.get("asset_id"), "resource_path": asset.get("resource_path"),
                "nav_entry_required": bool(asset.get("nav_entry_required", False)), "placement": "before_body",
            })
            placed_asset_ids.add(str(asset.get("asset_id", "") or ""))

    for chapter in chapters:
        number = int(chapter["number"])
        title = str(chapter.get("chapter_title", "") or "")
        # Chapter recovery is performed after the stable text authority has
        # been frozen.  Propagate the recovered logical identity back into the
        # same records before writing stable/evidence/rebuild views so every
        # exported projection names the same chapter.
        for chapter_item in chapter.get("items") or []:
            chapter_item["chapter_id"] = str(chapter.get("chapter_id", "") or f"chapter-{number:03d}")
            chapter_item["chapter_title"] = title
            chapter_item["logical_chapter_number"] = number
        text_path = reading_dir / f"chapter_{number:03d}.txt"
        text_parts = [f"# {title or chapter['chapter_id']}"]
        for item in chapter["items"]:
            proposed = str(item.get("proposed_text", "") or "")
            marker = (
                f"[[ITEM {item.get('item_id', '')} | page={item.get('page', 0)} | "
                f"type={item.get('block_type', '')} | policy={item.get('edit_policy', '')}]]"
            )
            text_parts.extend([marker, proposed, ""])
            reading_projection.append({"item_id": item.get("item_id"), "sha256": _sha256_bytes(proposed.encode("utf-8"))})
        content = "\n".join(text_parts).rstrip() + "\n"
        text_path.write_text(content, encoding="utf-8")
        full_parts.append(content)

        technical_parts = _technical_item_parts(
            chapter["items"], split_before_item_ids=split_before_ids, split_after_item_ids=split_after_ids,
        )
        part_manifest: list[dict] = []
        high = medium = full = 0
        for part_index, part_items in enumerate(technical_parts, start=1):
            planned_xhtml = f"EPUB/text/chapter_{number:03d}_part_{part_index:03d}.xhtml"
            evidence_name = f"chapter_{number:03d}.jsonl" if len(technical_parts) == 1 else f"chapter_{number:03d}_part_{part_index:03d}.jsonl"
            evidence_path = evidence_dir / evidence_name
            lines: list[str] = []
            part_chars = 0
            for item in part_items:
                item["technical_part_index"] = part_index
                item["final_html_id"] = str(item.get("final_html_id", "") or _final_html_id(item))
                item["framework_epub_target"] = str(item.get("framework_epub_target", "") or mapped_by_id.get(str(item.get("item_id", "") or ""), {}).get("epub_target", "") or "")
                item["planned_final_xhtml"] = planned_xhtml
                item["planned_final_target"] = f"{planned_xhtml}#{item['final_html_id']}"
                item["expected_epub_target"] = item["planned_final_target"]
                item["expected_evidence_path"] = f"evidence/{evidence_name}"
                item["expected_reading_path"] = f"reading/{text_path.name}"
                mapped = mapped_by_id.get(str(item.get("item_id", "") or ""), {})
                source_item = (full_items_by_id or {}).get(str(item.get("item_id", "") or ""))
                if compact_mode:
                    evidence_item = build_chapter_evidence_projection(
                        item, source_item, full_index_by_id.get(str(item.get("item_id", "") or "")),
                    )
                else:
                    evidence_item = _lossless_chapter_evidence_item(mapped, source_item, item)
                lines.append(json.dumps(_redact_absolute_paths(evidence_item), ensure_ascii=False, separators=(",", ":")))
                proposed = str(item.get("proposed_text", "") or "")
                digest = _sha256_bytes(proposed.encode("utf-8"))
                evidence_projection.append({"item_id": item.get("item_id"), "sha256": digest})
                authority_projection.append({"item_id": item.get("item_id"), "sha256": digest})
                part_chars += len(proposed)
                high += str(item.get("risk_level", "")) == "high"
                medium += str(item.get("risk_level", "")) == "medium"
                full += str(item.get("evidence_tier", "")) in {"full_physical", "full_fusion"}
            evidence_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            first_item_id = str(part_items[0].get("item_id", "") or "") if part_items else ""
            last_item_id = str(part_items[-1].get("item_id", "") or "") if part_items else ""
            part_record = {
                "part_index": part_index, "item_count": len(part_items), "text_character_count": part_chars,
                "first_item_id": first_item_id, "last_item_id": last_item_id,
                "evidence_path": f"evidence/{evidence_name}", "planned_final_xhtml": planned_xhtml,
                "evidence_sha256": _sha256_bytes(evidence_path.read_bytes()),
                "forced_by_image_boundary": bool(first_item_id in split_before_ids or last_item_id in split_after_ids),
            }
            part_manifest.append(part_record)
            spine_nodes.append({
                "type": "content", "path": planned_xhtml, "chapter_id": chapter["chapter_id"],
                "chapter_title": title, "technical_part_index": part_index,
                "first_item_id": first_item_id, "last_item_id": last_item_id,
                "nav_entry_required": part_index == 1,
            })
            for asset in assets_after.get(last_item_id, []):
                asset_id = str(asset.get("asset_id", "") or "")
                if asset_id in placed_asset_ids:
                    continue
                spine_nodes.append({
                    "type": "standalone_image", "path": asset.get("planned_asset_xhtml"),
                    "asset_id": asset_id, "resource_path": asset.get("resource_path"),
                    "after_item_id": last_item_id, "before_item_id": asset.get("stable_item_after") or None,
                    "nav_entry_required": bool(asset.get("nav_entry_required", False)), "placement": "anchored",
                })
                placed_asset_ids.add(asset_id)
            for asset in assets_before.get(first_item_id, []):
                asset_id = str(asset.get("asset_id", "") or "")
                if asset_id in placed_asset_ids:
                    continue
                insert_at = max(0, len(spine_nodes) - 1)
                spine_nodes.insert(insert_at, {
                    "type": "standalone_image", "path": asset.get("planned_asset_xhtml"),
                    "asset_id": asset_id, "resource_path": asset.get("resource_path"),
                    "after_item_id": asset.get("stable_item_before") or None, "before_item_id": first_item_id,
                    "nav_entry_required": bool(asset.get("nav_entry_required", False)), "placement": "anchored_before_item",
                })
                placed_asset_ids.add(asset_id)
        chapter_manifest.append({
            "chapter_id": chapter["chapter_id"], "chapter_title": title,
            "reading_path": f"reading/{text_path.name}",
            "evidence_path": part_manifest[0]["evidence_path"] if part_manifest else "",
            "evidence_paths": [part["evidence_path"] for part in part_manifest],
            "item_count": len(chapter["items"]),
            "text_character_count": sum(len(str(item.get("proposed_text", "") or "")) for item in chapter["items"]),
            "technical_part_count": len(part_manifest), "planned_content_xhtml_count": len(part_manifest),
            "parts": part_manifest, "high_risk_count": high, "medium_risk_count": medium,
            "full_physical_evidence_count": full, "reading_sha256": _sha256_bytes(text_path.read_bytes()),
            "detection_source": chapter.get("detection_source"),
            "chapter_detection_warning": bool(chapter.get("chapter_detection_warning", False)),
        })
    for asset in exported_assets:
        asset_id = str(asset.get("asset_id", "") or "")
        if asset_id in placed_asset_ids:
            continue
        spine_nodes.append({
            "type": "standalone_image", "path": asset.get("planned_asset_xhtml"),
            "asset_id": asset_id, "resource_path": asset.get("resource_path"),
            "nav_entry_required": bool(asset.get("nav_entry_required", False)), "placement": "unresolved_anchor_append",
        })
        placed_asset_ids.add(asset_id)
    full_path = reading_dir / "full_book_with_item_markers.txt"
    full_path.write_text("\n\n".join(full_parts).rstrip() + "\n", encoding="utf-8")
    structure_plan = {
        "schema": "novel_formatter.ai_publication_output_structure_plan.v3",
        "split_priority": ["chapter_boundary", "standalone_image", "special_page", "character_limit", "item_limit"],
        "spine_nodes": spine_nodes,
        "content_xhtml_count": sum(node.get("type") == "content" for node in spine_nodes),
        "standalone_image_xhtml_count": sum(node.get("type") == "standalone_image" for node in spine_nodes),
        "nav_entry_count": sum(bool(node.get("nav_entry_required")) for node in spine_nodes),
        "technical_parts_must_not_create_extra_nav_entries": True,
        "all_publication_assets_placed": len(placed_asset_ids) == len(exported_assets),
        "placed_asset_ids": sorted(placed_asset_ids),
    }
    stable_projection = [{"item_id": item.get("item_id"), "sha256": item.get("proposed_text_sha256")} for item in stable_records]
    authority_validation = {
        "schema": "novel_formatter.ai_publication_text_authority_validation.v3",
        "same_text_authority_across_views": stable_projection == authority_projection == evidence_projection == reading_projection,
        "stable_projection_sha256": _sha256_bytes(_json_bytes(stable_projection)),
        "reading_projection_sha256": _sha256_bytes(_json_bytes(reading_projection)),
        "chapter_evidence_projection_sha256": _sha256_bytes(_json_bytes(evidence_projection)),
        "reading_text_sha256_matches_stable_projection": stable_projection == reading_projection,
        "chapter_evidence_sha256_matches_stable_projection": stable_projection == evidence_projection,
        "item_count": len(stable_projection),
    }
    if not authority_validation["same_text_authority_across_views"]:
        raise repair.AIRepairEpubError("reading、evidence 与 stable map 的正文权威不一致，禁止完成导出。")
    return chapter_manifest, structure_plan, authority_validation

def _book_identity(
    primary_doc: UnifiedDocument,
    package: dict,
    repair_map: dict,
    final_name: str,
    framework_inventory: dict | None = None,
    assets_manifest: dict | None = None,
    cover_identity: dict | None = None,
) -> dict:
    """Record optional identity evidence without making it an export gate.

    OCR repair can start before the user has entered a reliable title or author.
    Existing metadata is preserved as optional evidence only; the exporter never
    OCRs the cover to invent missing identity and never blocks on blank fields.
    """
    del assets_manifest, cover_identity
    metadata = getattr(primary_doc, "metadata", None)
    book = copy.deepcopy(package.get("book") or repair_map.get("book") or {})
    framework_metadata = (framework_inventory or {}).get("metadata") or {}
    raw_candidates = {
        "title": [
            ("document_metadata", getattr(metadata, "title", "")),
            ("package_book", book.get("title", "")),
            ("framework_opf", (framework_metadata.get("title") or [""])[0]),
        ],
        "author": [
            ("document_metadata", getattr(metadata, "author", "")),
            ("package_book", book.get("author", "")),
            ("framework_opf", (framework_metadata.get("creator") or [""])[0]),
        ],
        "illustrator": [
            ("document_metadata", getattr(metadata, "illustrator", "")),
            ("package_book", book.get("illustrator", "")),
        ],
        "publisher": [
            ("document_metadata", getattr(metadata, "publisher", "")),
            ("package_book", book.get("publisher", "")),
            ("framework_opf", (framework_metadata.get("publisher") or [""])[0]),
        ],
        "volume": [
            ("document_metadata", getattr(metadata, "volume", "")),
            ("package_book", book.get("volume", "")),
        ],
        "series": [
            ("document_metadata", getattr(metadata, "series", "")),
            ("package_book", book.get("series", "")),
        ],
        "isbn": [
            ("document_metadata", getattr(metadata, "isbn", "")),
            ("package_book", book.get("isbn", "")),
        ],
        "identifier": [
            ("document_metadata", getattr(metadata, "identifier", "")),
            ("package_book", book.get("identifier", "")),
            ("framework_opf", (framework_metadata.get("identifier") or [""])[0]),
        ],
    }
    resolved: dict[str, str | None] = {}
    evidence: dict[str, list[dict]] = {}
    conflicts: list[dict] = []
    for field, candidates in raw_candidates.items():
        clean = [(source, _clean_identity_value(value)) for source, value in candidates]
        clean = [(source, value) for source, value in clean if value]
        evidence[field] = [{"source": source, "value": value} for source, value in clean]
        unique = list(dict.fromkeys(value for _source, value in clean))
        resolved[field] = unique[0] if unique else None
        if len(unique) > 1:
            conflicts.append({
                "field": field,
                "values": unique,
                "sources": [source for source, _value in clean],
                "blocking": False,
            })

    language = (
        _clean_identity_value(getattr(metadata, "language", ""))
        or _clean_identity_value(book.get("language"))
        or "ja"
    )
    resolved["language"] = language
    return {
        "schema": "novel_formatter.ai_publication_book_identity.v3_optional",
        "metadata_required": False,
        "title_optional": True,
        "author_optional": True,
        "cover_ocr_attempted": False,
        "resolved": resolved,
        "evidence": evidence,
        "candidates": {"title": [], "author": []},
        "conflicts": conflicts,
        "metadata_must_not_be_blank": False,
        "metadata_must_not_use_placeholders": False,
        "forbidden_placeholders": [],
        "title_resolution_required": False,
        "author_resolution_required": False,
        "identity_resolution_required": False,
        "writing_direction": str(getattr(metadata, "writing_direction", "") or "vertical-rl"),
        "required_final_filename": final_name,
        "structure_sha256": str(repair_map.get("structure_sha256", "") or ""),
        "layout_sha256": str(repair_map.get("layout_sha256", "") or ""),
        "baseline_book_sha256": str(repair_map.get("baseline_book_sha256", "") or ""),
        "title": resolved.get("title"),
        "author": resolved.get("author"),
        "publisher": resolved.get("publisher"),
        "series": resolved.get("series"),
        "volume": resolved.get("volume"),
        "isbn": resolved.get("isbn"),
        "identifier": resolved.get("identifier") or resolved.get("isbn") or str(repair_map.get("baseline_book_sha256", "") or ""),
        "language": language,
    }


def _framework_audit(preflight: dict, book_identity: dict, assets_manifest: dict, framework_inventory: dict) -> dict:
    preserve = preflight.get("recommended_build_strategy") == "preserve_and_patch"
    identity_resolved = not bool(book_identity.get("identity_resolution_required"))
    resources_complete = int(assets_manifest.get("missing_publication_image_count", 0) or 0) == 0
    expected_images = int(assets_manifest.get("publication_image_count_exported", 0) or 0)
    framework_images = len(framework_inventory.get("images", []))
    return {
        "schema": FRAMEWORK_AUDIT_SCHEMA,
        "package_profile": PACKAGE_PROFILE,
        "framework_role": "clean_resource_mapping_only",
        "contains_scan_pages": False,
        "contains_embedded_ai_evidence": False,
        "must_not_ship_unchanged": True,
        "must_remove_all_work_metadata_before_delivery": True,
        "metadata_trusted": bool(identity_resolved and not book_identity.get("conflicts")),
        "images_trusted": bool(resources_complete and framework_images == expected_images),
        "image_bytes_trusted": bool(resources_complete),
        "paragraph_ids_trusted": True,
        "chapter_boundaries_trusted": bool(preserve and not preflight.get("must_split_chapter_xhtml")),
        "nav_trusted": bool(preserve and not preflight.get("must_rebuild_nav")),
        "ncx_trusted": bool(preserve and not preflight.get("must_rebuild_ncx")),
        "spine_trusted": bool(preserve and not preflight.get("must_rebuild_spine")),
        "css_trusted": False,
        "forbidden_authority_claims": [
            "chapter_count", "xhtml_partitioning", "nav", "ncx", "spine", "metadata", "final_css"
        ],
        "framework_inventory_summary": {
            "opf_path": framework_inventory.get("opf_path", ""),
            "xhtml_count": len(framework_inventory.get("xhtml", [])),
            "image_count": framework_images,
            "expected_publication_image_count": expected_images,
            "scan_page_count": 0,
            "nav_in_spine": bool((framework_inventory.get("nav") or {}).get("in_spine")),
        },
    }

def _existing_final_title(primary_doc: UnifiedDocument, package: dict) -> str | None:
    metadata = getattr(primary_doc, "metadata", None)
    book = package.get("book") if isinstance(package.get("book"), dict) else {}
    for value in (getattr(metadata, "title", ""), book.get("title", "")):
        clean = _clean_identity_value(value)
        if clean:
            return clean
    return None


def _final_output_filename(primary_doc: UnifiedDocument, package: dict) -> str:
    title = _existing_final_title(primary_doc, package)
    if title:
        return f"{_safe_name(title, default='AI精校出版版')}_高精度OCR融合校订版.epub"
    return "AI精校出版版.epub"


def _apply_reference_alignment(stable_records: list[dict], reference_report: dict) -> None:
    """Attach optional publication Ruby/structure evidence without replacing text."""
    by_id = {str(record.get("item_id", "") or ""): record for record in stable_records}
    for alignment in reference_report.get("alignment") or []:
        item_id = str(alignment.get("item_id", "") or "")
        record = by_id.get(item_id)
        if not record:
            continue
        reference_text = str(alignment.get("reference_text", "") or "")
        confidence = float(alignment.get("alignment_confidence", 0.0) or 0.0)
        exact = bool(reference_text and reference_text == str(record.get("proposed_text", "") or ""))
        tokens = copy.deepcopy(alignment.get("inline_tokens") or [])
        ruby_groups = copy.deepcopy(alignment.get("ruby_groups") or [])
        record["publication_reference_alignment"] = {
            "reference_id": alignment.get("reference_id"),
            "alignment_confidence": confidence,
            "plain_text_exact_match": exact,
            "authority": alignment.get("authority", "publication_reference_evidence"),
        }
        if exact and confidence >= 0.92 and ruby_groups:
            record["inline_tokens"] = tokens
            record["reference_ruby_groups"] = ruby_groups
            record["ruby_group_count"] = len(ruby_groups)
            record["ruby_status"] = "evidence_backed_allowed"
            record["ruby_source"] = "explicit_publication_reference"
            record["ruby_confidence"] = confidence
        elif confidence >= 0.85 and reference_text and not exact:
            record.update({
                "edit_policy": "review_required",
                "model_action_required": True,
                "text_locked": False,
                "final_text_requires_model_decision": True,
            })
            record["unlock_reasons"] = list(dict.fromkeys([
                *(record.get("unlock_reasons") or []), "publication_reference_conflict",
            ]))
            lock = record.setdefault("lock_validation", {})
            lock["publication_reference_plain_text_matches"] = False


def _apply_page_ledger_rules(stable_records: list[dict], ledger: dict) -> None:
    """Unlock only text items adjacent to independently confirmed fatal gaps."""
    by_id = {str(record.get("item_id", "") or ""): record for record in stable_records}
    for page in ledger.get("pages") or []:
        status = str(page.get("status", "") or "")
        page_review = status in {"suspected_gap", "detector_disagreement", "unverifiable"}
        if page_review:
            # Page-level review is recorded without changing otherwise stable
            # text policy.  Secondary detector disagreement cannot overrule the
            # columns that were actually used by OCR.
            for item_id in page.get("related_item_ids") or []:
                record = by_id.get(str(item_id))
                if record is not None:
                    record["page_review_required"] = True
                    record["page_review_reasons"] = list(dict.fromkeys([
                        *(record.get("page_review_reasons") or []),
                        *(page.get("page_review_reasons") or [status]),
                    ]))
        fatal_slots = page.get("fatal_slots") or page.get("unmapped_body_slots") or []
        affected = {
            str(item_id)
            for slot in fatal_slots
            for item_id in (slot.get("adjacent_item_ids") or [])
            if str(item_id)
        }
        for item_id in affected:
            record = by_id.get(item_id)
            if not record:
                continue
            record.update({
                "edit_policy": "review_required",
                "model_action_required": True,
                "text_locked": False,
                "final_text_requires_model_decision": True,
                "physical_column_coverage_complete": False,
            })
            record["unlock_reasons"] = list(dict.fromkeys([
                *(record.get("unlock_reasons") or []), "physical_column_coverage_incomplete",
            ]))
            record["risk_flags"] = list(dict.fromkeys([
                *(record.get("risk_flags") or []), "adjacent_fatal_unmapped_body_column",
            ]))
            record.setdefault("lock_validation", {})["physical_column_coverage_complete"] = False

def _output_contract(
    final_name: str,
    language: str,
    assets_manifest: dict,
    preflight: dict,
    *,
    atomic_span_count: int = 0,
    global_anomalies: dict | None = None,
    structure_plan: dict | None = None,
    reference_available: bool = False,
    page_column_ledger: dict | None = None,
    existing_metadata: dict | None = None,
) -> dict:
    expected_images = int(assets_manifest.get("final_epub_expected_image_count", 0) or 0)
    anomalies = global_anomalies or {}
    ledger = page_column_ledger or {}
    metadata = existing_metadata or {}
    return {
        "schema": OUTPUT_CONTRACT_SCHEMA,
        "package_profile": PACKAGE_PROFILE,
        "required_output_name": final_name,
        "build_mode": "auto",
        "allowed_build_strategies": ["preserve_and_patch", "hybrid_rebuild", "full_rebuild"],
        "must_follow_preflight_rebuild_flags": True,
        "must_run_second_pass": True,
        "must_reopen_final_epub": True,
        "must_run_local_executable_auditor": True,
        "local_audit_command": f"python tools/validate_final_epub.py --epub {final_name} --stable-map 12_stable_text_map.jsonl --atomic-map 13_atomic_span_map.jsonl --structure-plan 15_output_structure_plan.json --column-ledger 16_page_column_ledger.json --global-anomalies 14_global_text_anomalies.json --book-identity 02_book_identity.json --assets-manifest 05_assets_manifest.json",
        "must_compare_with_final_text_map": True,
        "final_text_map_must_be_created_by_model": True,
        "final_text_map_source_evidence": ["12_stable_text_map.jsonl", "13_atomic_span_map.jsonl", FULL_FUSION_FILENAME],
        "stable_text_map_path": "12_stable_text_map.jsonl",
        "atomic_span_map_path": "13_atomic_span_map.jsonl",
        "global_text_anomalies_path": "14_global_text_anomalies.json",
        "output_structure_plan_path": "15_output_structure_plan.json",
        "page_column_ledger_path": "16_page_column_ledger.json",
        "source_text_authority_path": "12_stable_text_map.jsonl",
        "source_text_evidence_path": FULL_FUSION_FILENAME,
        "stable_text_map_role": "stable_target_with_edit_policy",
        "stable_text_map_is_sole_default_text_authority": True,
        "character_fused_text_is_evidence_only": True,
        "ai_adjudicated_text_is_evidence_only": True,
        "stable_item_target_fields": ["item_id", "final_html_id", "planned_final_xhtml", "planned_final_target"],
        "must_use_planned_final_targets": True,
        "must_use_output_structure_plan": True,
        "must_preserve_framework_target_as_evidence_only": True,
        "risk_queue_is_navigation_only": True,
        "all_items_require_final_text_decision": False,
        "locked_consensus_items_default_to_proposed_text": True,
        "locked_items_may_change_only_with_explicit_unlock_reason": True,
        "structure_only_items_must_preserve_proposed_text": True,
        "review_required_items_need_model_decision": True,
        "edit_policy_values": ["locked_consensus", "review_required", "structure_only"],
        "atomic_span_count": atomic_span_count,
        "all_reliable_atomic_spans_must_be_covered_exactly_once": True,
        "unverified_atomic_spans_must_not_be_claimed_as_covered": True,
        "page_column_coverage_complete": bool((ledger.get("summary") or {}).get("coverage_complete", True)),
        "unmapped_body_column_count": int((ledger.get("summary") or {}).get("fatal_unmapped_body_slot_count", 0) or 0),
        "unmapped_body_columns_are_fatal_preflight": True,
        "unresolved_high_risk_global_anomalies_are_fatal": True,
        "global_anomaly_summary": copy.deepcopy(anomalies.get("summary") or {}),
        "reference_evidence_available": bool(reference_available),
        "publication_reference_required": False,
        "publication_reference_is_used_only_when_explicitly_enabled": True,
        "publication_reference_must_never_silently_replace_proposed_text": True,
        "publication_reference_is_higher_authority_when_alignment_confident": bool(reference_available),
        "valid_evidence_backed_ruby_allowed": bool(reference_available),
        "ruby_without_evidence_is_fatal": True,
        "must_partition_long_xhtml": bool(preflight.get("must_partition_long_xhtml")),
        "max_items_per_xhtml": int(preflight.get("max_items_per_xhtml", _MAX_ITEMS_PER_XHTML) or _MAX_ITEMS_PER_XHTML),
        "max_text_characters_per_xhtml": int(preflight.get("max_text_characters_per_xhtml", _MAX_TEXT_CHARACTERS_PER_XHTML) or _MAX_TEXT_CHARACTERS_PER_XHTML),
        "minimum_content_xhtml_count": int(preflight.get("minimum_content_xhtml_count", 1) or 1),
        "minimum_spine_node_count": len((structure_plan or {}).get("spine_nodes") or []),
        "standalone_images_must_have_own_xhtml": True,
        "technical_parts_must_not_create_extra_nav_entries": True,
        "must_remove_work_metadata": True,
        "forbidden_final_paths": [*list(_FORBIDDEN_WORK_PREFIXES), FULL_FUSION_FILENAME, "visual_evidence/", "reference/", "tools/", "audit/"],
        "forbidden_final_attributes": list(_FORBIDDEN_WORK_ATTRIBUTES),
        "nav_in_spine": False,
        "language": language or "ja",
        "preserve_image_bytes": True,
        "metadata_title_required": False,
        "metadata_author_required": False,
        "metadata_is_optional_during_repair": True,
        "generic_final_metadata_allowed": not bool(metadata.get("title")),
        "placeholder_metadata_forbidden": False,
        "repair_package_metadata_may_be_missing": True,
        "final_build_must_preserve_existing_metadata": True,
        "existing_metadata": copy.deepcopy(metadata),
        "metadata_title_required_when_existing_title_present": bool(metadata.get("title")),
        "metadata_identifier_required": True,
        "metadata_author_must_be_preserved_when_present": bool(metadata.get("author")),
        "scan_pages_required": False,
        "scan_page_count_expected": 0,
        "scan_evidence_absence_is_error": False,
        "selective_visual_evidence_allowed_in_package_only": True,
        "publication_images_required": True,
        "publication_image_count_must_match_manifest": True,
        "final_epub_expected_image_count": expected_images,
        "forbidden_package_content": ["source_page_*", "scan_evidence/full_pages/", "page_thumbnails/all_pages/", "absolute_local_paths"],
        "required_reports": ["audit/audit_report.json", "audit/audit_report.md", "audit/text_changes.csv", "audit/atomic_span_coverage.csv", "audit/unresolved_anomalies.csv", "audit/epub_integrity.json"],
        "report_delivery_policy": "Deliver the final EPUB together with the complete audit directory or a ZIP containing it; do not embed audit files in EPUB.",
        "delivery_gate": "No fatal check may fail. A readable EPUB or a model-authored claim is not sufficient.",
        "edition_label_without_publication_reference": "高精度 OCR 融合校订版／适合 AI 翻译的日文母本",
    }


def _final_audit_rules(
    preflight: dict,
    assets_manifest: dict,
    *,
    atomic_span_count: int = 0,
    global_anomalies: dict | None = None,
    structure_plan: dict | None = None,
    page_column_ledger: dict | None = None,
    reference_available: bool = False,
) -> dict:
    detected = int(preflight.get("logical_chapter_count", preflight.get("detected_chapter_count", 0)) or 0)
    minimum_xhtml = int(preflight.get("minimum_content_xhtml_count", max(1, detected)) or max(1, detected))
    anomaly_summary = copy.deepcopy((global_anomalies or {}).get("summary") or {})
    ledger = page_column_ledger or {}
    return {
        "schema": FINAL_AUDIT_SCHEMA,
        "audit_result_required": "pass",
        "package_profile": PACKAGE_PROFILE,
        "executable_auditor": "tools/validate_final_epub.py",
        "audit_output_directory": "audit/",
        "scan_pages_required": False,
        "scan_page_count_expected": 0,
        "publication_images_required": True,
        "publication_image_count_must_match_manifest": True,
        "final_epub_expected_image_count": int(assets_manifest.get("final_epub_expected_image_count", 0) or 0),
        "on_failure": "continue_repair_and_rebuild; do_not_claim_completion; do_not_deliver_epub",
        "fatal_checks": {
            "metadata_title_lost_when_source_had_title": True,
            "metadata_author_lost_when_source_had_author": True,
            "metadata_identifier_empty": True,
            "language_not_ja": True,
            "nav_only_contains_epilogue": True,
            "nav_inside_spine": True,
            "replacement_character_found": True,
            "placeholder_square_found": True,
            "missing_manifest_resource": True,
            "publication_image_count_mismatch": True,
            "scan_page_found_in_final_epub": True,
            "broken_internal_link": True,
            "duplicate_html_id": True,
            "work_files_embedded": True,
            "full_fusion_evidence_embedded": True,
            "work_attributes_embedded": True,
            "stable_item_count_mismatch": True,
            "stable_item_missing": True,
            "stable_item_duplicated": True,
            "locked_item_text_changed_without_unlock_reason": True,
            "structure_only_item_text_changed": True,
            "reliable_atomic_span_missing": True,
            "reliable_atomic_span_duplicated": True,
            "atomic_span_order_conflict": True,
            "unverified_atomic_span_claimed_as_reliable": True,
            "unmapped_body_column_remaining": True,
            "unresolved_fatal_duplicate_or_span_move": True,
            "unsupported_insertion_remaining": True,
            "orphan_source_span_remaining": True,
            "chapter_count_below_preflight_minimum": True,
            "content_xhtml_count_below_preflight_minimum": True,
            "stable_final_target_mismatch": True,
            "standalone_image_xhtml_missing": True,
            "image_anchor_order_mismatch": True,
            "ruby_without_evidence": True,
            "ruby_base_text_mismatch": True,
            "ruby_reading_empty": True,
            "mimetype_not_first_or_compressed": True,
            "xml_or_xhtml_parse_failure": True,
        },
        "valid_evidence_backed_ruby_allowed": bool(reference_available),
        "thresholds": {
            "unmapped_text_items": 0,
            "duplicated_text_items": 0,
            "locked_item_changes_without_reason": 0,
            "structure_only_text_changes": 0,
            "reliable_atomic_spans_expected": atomic_span_count,
            "reliable_atomic_spans_missing": 0,
            "reliable_atomic_spans_duplicated": 0,
            "atomic_span_order_conflicts": 0,
            "unmapped_body_columns": int((ledger.get("summary") or {}).get("fatal_unmapped_body_slot_count", 0) or 0),
            "required_unmapped_body_columns": 0,
            "unresolved_fatal_global_anomalies": 0,
            "unsupported_insertions": 0,
            "orphan_source_spans": 0,
            "missing_images": 0,
            "scan_pages_in_final_epub": 0,
            "publication_image_count_delta": 0,
            "standalone_image_xhtml_missing": 0,
            "broken_links": 0,
            "replacement_characters": 0,
            "placeholder_squares": 0,
            "control_characters": 0,
            "work_file_count": 0,
            "full_fusion_work_file_count": 0,
            "work_attribute_count": 0,
            "minimum_final_chapter_count": detected,
            "minimum_content_xhtml_count": minimum_xhtml,
            "minimum_spine_node_count": len((structure_plan or {}).get("spine_nodes") or []),
            "maximum_items_per_content_xhtml": int(preflight.get("max_items_per_xhtml", _MAX_ITEMS_PER_XHTML) or _MAX_ITEMS_PER_XHTML),
            "maximum_text_characters_per_content_xhtml": int(preflight.get("max_text_characters_per_xhtml", _MAX_TEXT_CHARACTERS_PER_XHTML) or _MAX_TEXT_CHARACTERS_PER_XHTML),
        },
        "source_anomaly_summary": anomaly_summary,
        "page_column_ledger_summary": copy.deepcopy(ledger.get("summary") or {}),
        "forbidden_characters": {"replacement_character": "\uFFFD", "placeholder_square": "□", "nul": "\u0000"},
        "abnormal_change_checks": [
            "new_latin_run", "numeric_value_changed", "range_corrupted", "dash_style_changed",
            "new_simplified_chinese_character", "small_kana_deleted", "emphasis_quote_deleted", "duplicate_sentence_count_increased",
        ],
        "required_comparisons": [
            "same_text_authority_across_stable_reading_and_evidence_views",
            "stable_item_id_presence_and_uniqueness",
            "locked_and_structure_only_text_policy",
            "stable_item_text_against_model_generated_final_text_map",
            "atomic_source_span_exactly_once_coverage_without_proposed_text_fallback",
            "page_physical_column_ledger_coverage",
            "global_duplicate_move_insertion_orphan_resolution",
            "full_fusion_item_coverage_and_candidate_integrity",
            "item_id_to_final_html_id_to_planned_xhtml",
            "content_xhtml_partition_limits",
            "output_structure_plan_spine_order",
            "chapter_title_and_count",
            "nav_targets_and_order",
            "spine_order",
            "image_sha256_standalone_xhtml_and_anchor_order",
            "ruby_is_absent_without_evidence_or_exactly_matches_evidence_backed_tokens",
            "existing_metadata_preservation",
        ],
    }

def _write_audit_tools(folder: Path) -> list[str]:
    """Ship standard-library-only structure-aware build/audit helpers."""
    audit_common = r'''#!/usr/bin/env python3
from __future__ import annotations
import csv
import json
import posixpath
import zipfile
from pathlib import Path, PurePosixPath
import xml.etree.ElementTree as ET

WORK_PREFIXES = (
    "META-INF/ai-repair/", "META-INF/ai-publication/", "evidence/", "reading/",
    "framework/", "visual_evidence/", "reference/", "tools/", "audit/",
)
WORK_BASENAMES = {"full_fusion_evidence.json"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    result = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            result.append(json.loads(line))
    return result


def local_name(tag):
    return str(tag).rsplit("}", 1)[-1]


def resolve_href(base_path, href):
    value = str(href or "").split("#", 1)[0]
    if not value:
        return ""
    return str(PurePosixPath(posixpath.normpath(posixpath.join(posixpath.dirname(base_path), value))))


def plain_text(node):
    parts = []
    def walk(current):
        if local_name(current.tag) in {"rt", "rp"}:
            return
        if current.text:
            parts.append(current.text)
        for child in list(current):
            walk(child)
            if child.tail:
                parts.append(child.tail)
    walk(node)
    return "".join(parts)


def ruby_groups(node):
    groups = []
    for ruby in node.iter():
        if local_name(ruby.tag) != "ruby":
            continue
        base_parts = []
        readings = []
        if ruby.text:
            base_parts.append(ruby.text)
        for child in list(ruby):
            name = local_name(child.tag)
            if name == "rt":
                value = "".join(child.itertext()).strip()
                if value:
                    readings.append(value)
            elif name != "rp":
                base_parts.append(plain_text(child))
            if child.tail:
                base_parts.append(child.tail)
        base = "".join(base_parts).strip()
        reading = "".join(readings).strip()
        groups.append({"base": base, "reading": reading})
    return groups


def _find_opf(archive):
    root = ET.fromstring(archive.read("META-INF/container.xml"))
    for element in root.iter():
        if local_name(element.tag) == "rootfile" and element.attrib.get("full-path"):
            return element.attrib["full-path"]
    raise ValueError("container.xml has no rootfile")


def epub_snapshot(epub):
    epub = Path(epub)
    out = {
        "path": epub.name, "crc_ok": False, "mimetype_ok": False,
        "xml_errors": [], "work_paths": [], "names": [], "opf_path": "",
        "metadata": {}, "manifest_missing": [], "spine_paths": [],
        "nav_in_spine": False, "elements": [], "duplicate_ids": [],
        "broken_links": [], "work_attributes": [], "image_count": 0,
        "spine_text": "", "ruby_group_count": 0,
    }
    with zipfile.ZipFile(epub, "r") as archive:
        names = archive.namelist()
        name_set = set(names)
        out["names"] = names
        out["crc_ok"] = archive.testzip() is None
        out["mimetype_ok"] = bool(
            names and names[0] == "mimetype"
            and archive.getinfo("mimetype").compress_type == zipfile.ZIP_STORED
        )
        out["work_paths"] = [
            name for name in names
            if name.startswith(WORK_PREFIXES) or PurePosixPath(name).name in WORK_BASENAMES
        ]
        opf_path = _find_opf(archive)
        out["opf_path"] = opf_path
        opf = ET.fromstring(archive.read(opf_path))
        manifest = {}
        spine_ids = []
        for element in opf.iter():
            name = local_name(element.tag)
            if name == "item":
                item_id = str(element.attrib.get("id", "") or "")
                href = resolve_href(opf_path, element.attrib.get("href", ""))
                manifest[item_id] = {
                    "path": href,
                    "media_type": str(element.attrib.get("media-type", "") or ""),
                    "properties": str(element.attrib.get("properties", "") or ""),
                }
            elif name == "itemref" and element.attrib.get("idref"):
                spine_ids.append(str(element.attrib["idref"]))
            elif name in {"title", "creator", "language", "identifier"}:
                value = "".join(element.itertext()).strip()
                if value and name not in out["metadata"]:
                    out["metadata"][name] = value
        out["manifest_missing"] = [value["path"] for value in manifest.values() if value["path"] not in name_set]
        out["spine_paths"] = [manifest[value]["path"] for value in spine_ids if value in manifest]
        out["nav_in_spine"] = any("nav" in manifest.get(value, {}).get("properties", "").split() for value in spine_ids)
        out["image_count"] = sum(value["media_type"].startswith("image/") for value in manifest.values())

        id_locations = {}
        spine_text_parts = []
        for path in out["spine_paths"]:
            if path not in name_set:
                continue
            try:
                root = ET.fromstring(archive.read(path))
            except Exception as exc:
                out["xml_errors"].append({"path": path, "error": str(exc)})
                continue
            spine_text_parts.append(plain_text(root))
            for element in root.iter():
                element_id = str(element.attrib.get("id", "") or "")
                for attr in element.attrib:
                    if local_name(attr) in {"data-item-id", "data-row-id", "data-block-id"}:
                        out["work_attributes"].append({"path": path, "id": element_id, "attribute": local_name(attr)})
                if element_id:
                    key = (path, element_id)
                    id_locations.setdefault((path, element_id), 0)
                    id_locations[(path, element_id)] += 1
                    groups = ruby_groups(element)
                    out["elements"].append({
                        "path": path, "id": element_id, "text": plain_text(element),
                        "ruby_groups": groups,
                    })
                    out["ruby_group_count"] += len(groups)
                if local_name(element.tag) == "a":
                    href = str(element.attrib.get("href", "") or "")
                    if href and not href.startswith(("http://", "https://", "mailto:", "tel:")):
                        target_path = resolve_href(path, href)
                        if target_path and target_path not in name_set:
                            out["broken_links"].append({"path": path, "href": href, "resolved": target_path})
        out["duplicate_ids"] = [
            {"path": path, "id": element_id, "count": count}
            for (path, element_id), count in id_locations.items() if count > 1
        ]
        out["spine_text"] = "\n".join(spine_text_parts)
    return out


def count_occurrences(text, needle):
    return text.count(needle) if needle else 0


def write_csv(path, fieldnames, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
'''

    validator = r'''#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
from audit_common import read_json, read_jsonl, epub_snapshot, count_occurrences, write_csv


def _normal_path(value):
    return str(value or "").lstrip("/")


def _expected_ruby(item):
    groups = item.get("reference_ruby_groups") or []
    if groups:
        return [{"base": str(v.get("base", "") or ""), "reading": str(v.get("reading", "") or "")} for v in groups]
    result = []
    for token in item.get("inline_tokens") or []:
        if token.get("type") == "ruby":
            result.append({"base": str(token.get("base", "") or ""), "reading": str(token.get("reading", "") or "")})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epub", required=True)
    parser.add_argument("--stable-map", default="12_stable_text_map.jsonl")
    parser.add_argument("--atomic-map", default="13_atomic_span_map.jsonl")
    parser.add_argument("--structure-plan", default="15_output_structure_plan.json")
    parser.add_argument("--column-ledger", default="16_page_column_ledger.json")
    parser.add_argument("--global-anomalies", default="14_global_text_anomalies.json")
    parser.add_argument("--book-identity", default="02_book_identity.json")
    parser.add_argument("--assets-manifest", default="05_assets_manifest.json")
    parser.add_argument("--output-dir", default="audit")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stable = read_jsonl(args.stable_map)
    atomic = read_jsonl(args.atomic_map)
    structure = read_json(args.structure_plan)
    ledger = read_json(args.column_ledger)
    anomalies = read_json(args.global_anomalies) if Path(args.global_anomalies).is_file() else {"summary": {}}
    identity = read_json(args.book_identity) if Path(args.book_identity).is_file() else {}
    assets = read_json(args.assets_manifest) if Path(args.assets_manifest).is_file() else {}
    snapshot = epub_snapshot(args.epub)
    spine_text = snapshot.pop("spine_text")
    fatal = []
    change_rows = []
    anomaly_rows = []

    elements_by_target = {}
    elements_by_id = {}
    for element in snapshot.get("elements") or []:
        key = (_normal_path(element.get("path")), str(element.get("id", "") or ""))
        elements_by_target.setdefault(key, []).append(element)
        elements_by_id.setdefault(key[1], []).append(element)

    actual_by_item = {}
    for item in stable:
        item_id = str(item.get("item_id", "") or "")
        element_id = str(item.get("final_html_id", "") or "")
        planned = _normal_path(item.get("planned_final_xhtml"))
        matches = elements_by_target.get((planned, element_id), [])
        id_matches = elements_by_id.get(element_id, [])
        status = "ok"
        actual = ""
        if not matches:
            status = "target_mismatch" if id_matches else "missing"
            fatal.append(f"stable:{item_id}:{status}")
        elif len(matches) > 1:
            status = "duplicated"
            fatal.append(f"stable:{item_id}:duplicated")
        else:
            element = matches[0]
            actual_by_item[item_id] = element
            actual = str(element.get("text", "") or "")
            expected = str(item.get("proposed_text", "") or "")
            policy = str(item.get("edit_policy", "") or "")
            if policy in {"locked_consensus", "structure_only"} and actual != expected:
                status = "locked_text_changed" if policy == "locked_consensus" else "structure_only_text_changed"
                fatal.append(f"stable:{item_id}:{status}")
            elif not actual and expected:
                status = "empty"
                fatal.append(f"stable:{item_id}:empty")
            expected_ruby = _expected_ruby(item)
            actual_ruby = element.get("ruby_groups") or []
            if actual_ruby != expected_ruby:
                status = "ruby_mismatch"
                fatal.append(f"stable:{item_id}:ruby_mismatch")
        change_rows.append({
            "item_id": item_id, "policy": item.get("edit_policy"),
            "planned_xhtml": planned, "final_html_id": element_id,
            "expected_sha256": item.get("proposed_text_sha256"),
            "actual_text": actual, "status": status,
        })

    span_rows = []
    span_positions = {}
    for span in atomic:
        reliable = str(span.get("coverage_policy", "") or "") == "exactly_once"
        span_id = str(span.get("source_span_id", "") or "")
        item_id = str(span.get("expected_item_id", "") or "")
        value = str(span.get("selected_source_text", span.get("source_text", span.get("text", ""))) or "")
        element = actual_by_item.get(item_id)
        text = str((element or {}).get("text", "") or "")
        count = count_occurrences(text, value) if value else 0
        status = "unverified" if not reliable else ("ok" if count == 1 else ("missing" if count == 0 else "duplicated"))
        position = text.find(value) if reliable and value else -1
        if reliable and count != 1:
            fatal.append(f"atomic:{span_id}:{status}")
        if reliable and count == 1:
            # source_char_start is local to one physical column.  Use the
            # globally assigned source_order when present so multiple columns
            # whose local offsets all start at zero are not reported as a
            # false order conflict.
            source_order = int(span.get("source_order", span.get("source_char_start", 0)) or 0)
            span_positions.setdefault(item_id, []).append((source_order, position, span_id))
        span_rows.append({
            "source_span_id": span_id, "item_id": item_id,
            "coverage_policy": span.get("coverage_policy"),
            "occurrences": count, "position": position, "status": status,
        })
    for item_id, values in span_positions.items():
        by_source = [value[1] for value in sorted(values)]
        if by_source != sorted(by_source):
            fatal.append(f"atomic:{item_id}:order_conflict")

    ledger_summary = ledger.get("summary") or {}
    unmapped = int(ledger_summary.get("fatal_unmapped_body_slot_count", 0) or 0)
    if unmapped:
        fatal.append("unmapped_body_columns")

    planned_spine = [_normal_path(node.get("path")) for node in structure.get("spine_nodes") or []]
    actual_spine = [_normal_path(value) for value in snapshot.get("spine_paths") or []]
    if planned_spine and planned_spine != actual_spine:
        fatal.append("spine_plan_mismatch")
    expected_images = int(assets.get("final_epub_expected_image_count", 0) or 0)
    if expected_images and int(snapshot.get("image_count", 0) or 0) != expected_images:
        fatal.append("publication_image_count_mismatch")

    resolved = identity.get("resolved") or identity
    metadata = snapshot.get("metadata") or {}
    for field, epub_field in (("title", "title"), ("author", "creator"), ("identifier", "identifier")):
        expected = str(resolved.get(field, "") or "")
        if expected and str(metadata.get(epub_field, "") or "") != expected:
            fatal.append(f"metadata_{field}_mismatch")
    if str(metadata.get("language", "") or "") != "ja":
        fatal.append("language_not_ja")

    if not snapshot.get("crc_ok"):
        fatal.append("epub_crc")
    if not snapshot.get("mimetype_ok"):
        fatal.append("mimetype")
    if snapshot.get("xml_errors"):
        fatal.append("xml_parse")
    if snapshot.get("work_paths"):
        fatal.append("work_files_embedded")
    if snapshot.get("work_attributes"):
        fatal.append("work_attributes_embedded")
    if snapshot.get("manifest_missing"):
        fatal.append("missing_manifest_resource")
    if snapshot.get("broken_links"):
        fatal.append("broken_internal_link")
    if snapshot.get("duplicate_ids"):
        fatal.append("duplicate_html_id")
    if snapshot.get("nav_in_spine"):
        fatal.append("nav_inside_spine")
    for marker, name in (("\ufffd", "replacement_character"), ("□", "placeholder_square"), ("\x00", "nul")):
        if marker in spine_text:
            fatal.append(name)

    # Re-evaluate anomalies that expose a concrete text span.  Entries without
    # a concrete span remain visible in the CSV instead of being silently lost.
    for category in ("duplicate_clusters", "moved_span_clusters", "unsupported_insertions", "orphan_source_spans", "cross_item_order_conflicts"):
        for entry in anomalies.get(category) or []:
            value = str(entry.get("text", "") or "")
            count = count_occurrences(spine_text, value) if value else -1
            severity = str(entry.get("severity", "") or "")
            status = "manual_review" if not value else "observed_%s" % count
            if severity == "fatal" and category in {"duplicate_clusters", "moved_span_clusters"} and value and count > 1:
                fatal.append(f"anomaly:{category}:still_duplicated")
                status = "unresolved"
            anomaly_rows.append({"type": category, "severity": severity, "text": value, "occurrences": count, "status": status})

    fatal = list(dict.fromkeys(fatal))
    report = {
        "result": "pass" if not fatal else "fail",
        "fatal_errors": fatal,
        "stable_item_count": len(stable),
        "atomic_span_count": len(atomic),
        "epub_integrity": snapshot,
        "page_column_coverage_complete": unmapped == 0,
        "planned_spine_nodes": len(planned_spine),
        "actual_spine_nodes": len(actual_spine),
        "evidence_backed_ruby_group_count": sum(len(_expected_ruby(item)) for item in stable),
    }
    (output / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "audit_report.md").write_text(
        "# EPUB Audit\n\nResult: **%s**\n\nFatal errors: %s\n" % (report["result"], ", ".join(fatal) or "none"),
        encoding="utf-8",
    )
    (output / "epub_integrity.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output / "text_changes.csv", ["item_id", "policy", "planned_xhtml", "final_html_id", "expected_sha256", "actual_text", "status"], change_rows)
    write_csv(output / "atomic_span_coverage.csv", ["source_span_id", "item_id", "coverage_policy", "occurrences", "position", "status"], span_rows)
    write_csv(output / "unresolved_anomalies.csv", ["type", "severity", "text", "occurrences", "status"], anomaly_rows)
    raise SystemExit(0 if not fatal else 2)


if __name__ == "__main__":
    main()
'''

    builder = r'''#!/usr/bin/env python3
from __future__ import annotations
import argparse
import hashlib
import html
import mimetypes
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from audit_common import read_json, read_jsonl


def render_tokens(item):
    tokens = item.get("inline_tokens") or []
    if not tokens:
        return html.escape(str(item.get("proposed_text", "") or "")).replace("\n", "<br/>")
    parts = []
    for token in tokens:
        if token.get("type") == "ruby" and token.get("base") and token.get("reading"):
            parts.append("<ruby>%s<rt>%s</rt></ruby>" % (
                html.escape(str(token["base"])), html.escape(str(token["reading"])),
            ))
        else:
            parts.append(html.escape(str(token.get("value", "") or "")).replace("\n", "<br/>"))
    return "".join(parts)


def relative_from_epub(path):
    value = str(path or "").lstrip("/")
    return value[5:] if value.startswith("EPUB/") else value


def media_type(path):
    return mimetypes.guess_type(str(path))[0] or "application/octet-stream"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="")
    parser.add_argument("--stable-map", default="12_stable_text_map.jsonl")
    parser.add_argument("--structure-plan", default="15_output_structure_plan.json")
    parser.add_argument("--book-identity", default="02_book_identity.json")
    parser.add_argument("--assets-manifest", default="05_assets_manifest.json")
    parser.add_argument("--resources-dir", default="resources")
    parser.add_argument("--framework-epub", default="framework/resource_mapping_framework.epub")
    args = parser.parse_args()

    stable = read_jsonl(args.stable_map)
    structure = read_json(args.structure_plan)
    identity = read_json(args.book_identity)
    assets = read_json(args.assets_manifest)
    resolved = identity.get("resolved") or identity
    title = str(resolved.get("title") or "AI精校出版版")
    author = str(resolved.get("author") or "")
    language = str(resolved.get("language") or "ja")
    if language != "ja":
        raise SystemExit("language must be ja")
    identifier = str(resolved.get("identifier") or uuid.uuid4())
    default_name = str(identity.get("required_final_filename") or "AI精校出版版.epub")
    output = Path(args.output or default_name)

    temporary = Path(tempfile.mkdtemp(prefix="novel-formatter-final-"))
    try:
        (temporary / "META-INF").mkdir(parents=True)
        (temporary / "EPUB/styles").mkdir(parents=True)
        (temporary / "EPUB/images").mkdir(parents=True)
        (temporary / "mimetype").write_text("application/epub+zip", encoding="ascii")
        (temporary / "META-INF/container.xml").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            '<rootfiles><rootfile full-path="EPUB/package.opf" media-type="application/oebps-package+xml"/>'
            '</rootfiles></container>', encoding="utf-8",
        )
        (temporary / "EPUB/styles/style.css").write_text(
            'html { writing-mode: vertical-rl; -epub-writing-mode: vertical-rl; }\n'
            'body { line-height: 1.8; } p { margin: 0; }\n'
            '.image-page { margin: 0; padding: 0; text-align: center; }\n'
            '.image-page img { max-width: 100%; max-height: 100%; object-fit: contain; }\n'
            'rt { font-size: 0.5em; }\n', encoding="utf-8",
        )

        stable_by_path = {}
        for item in stable:
            path = relative_from_epub(item.get("planned_final_xhtml"))
            if not path:
                raise SystemExit("stable item has no planned_final_xhtml: %s" % item.get("item_id"))
            stable_by_path.setdefault(path, []).append(item)
        asset_by_id = {str(value.get("asset_id", "") or ""): value for value in assets.get("assets") or []}
        spine_nodes = structure.get("spine_nodes") or []
        if not spine_nodes:
            spine_nodes = [{"type": "content", "path": "EPUB/text/book.xhtml", "nav_entry_required": True, "chapter_title": title}]

        manifest = []
        spine_ids = []
        nav_entries = []
        copied_images = {}
        used_content_paths = set()
        for index, node in enumerate(spine_nodes, start=1):
            node_type = str(node.get("type", "content") or "content")
            relative_path = relative_from_epub(node.get("path"))
            if not relative_path:
                raise SystemExit("spine node has no path")
            target = temporary / "EPUB" / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            item_id = "spine-%03d" % index
            if node_type == "content":
                items = sorted(stable_by_path.get(relative_path, []), key=lambda value: int(value.get("reading_order", 0) or 0))
                used_content_paths.add(relative_path)
                body = []
                for item in items:
                    element_id = html.escape(str(item.get("final_html_id") or ""), quote=True)
                    if not element_id:
                        raise SystemExit("stable item has no final_html_id: %s" % item.get("item_id"))
                    tag = "h1" if str(item.get("block_type", "") or "") == "chapter" else "p"
                    body.append('<%s id="%s">%s</%s>' % (tag, element_id, render_tokens(item), tag))
                page_title = str(node.get("chapter_title") or title)
                target.write_text(
                    '<?xml version="1.0" encoding="utf-8"?>'
                    '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="ja"><head>'
                    '<title>%s</title><link rel="stylesheet" type="text/css" href="%s"/>'
                    '</head><body>%s</body></html>' % (
                        html.escape(page_title),
                        html.escape(str(PurePosixPath(*([".."] * max(1, len(PurePosixPath(relative_path).parts) - 1)), "styles/style.css"))),
                        "\n".join(body),
                    ), encoding="utf-8",
                )
                if node.get("nav_entry_required"):
                    nav_entries.append((relative_path, page_title))
            elif node_type == "standalone_image":
                asset_id = str(node.get("asset_id", "") or "")
                asset = asset_by_id.get(asset_id) or {}
                resource_path = Path(str(asset.get("resource_path") or node.get("resource_path") or ""))
                storage = str(asset.get("storage", "") or "")
                internal_path = str(asset.get("internal_path", "") or "")
                if storage == "framework_epub":
                    framework = Path(str(asset.get("framework_path") or args.framework_epub))
                    if not framework.is_file():
                        raise SystemExit("missing framework EPUB: %s" % framework)
                    if not internal_path:
                        raise SystemExit("framework asset has no internal_path: %s" % asset_id)
                    with zipfile.ZipFile(framework, "r") as source_epub:
                        try:
                            raw_image = source_epub.read(internal_path)
                        except KeyError:
                            raise SystemExit("missing framework image: %s" % internal_path)
                    expected_sha = str(asset.get("sha256", "") or "")
                    actual_sha = hashlib.sha256(raw_image).hexdigest()
                    if expected_sha and actual_sha != expected_sha:
                        raise SystemExit("framework image SHA-256 mismatch: %s" % asset_id)
                    suffix = Path(internal_path).suffix.lower() or ".bin"
                    image_name = "%s%s" % (asset_id or ("asset-%03d" % index), suffix)
                    image_target = temporary / "EPUB/images" / image_name
                    image_target.write_bytes(raw_image)
                else:
                    source = Path(args.resources_dir) / resource_path.name
                    if not source.is_file():
                        source = resource_path
                    if not source.is_file():
                        raise SystemExit("missing publication image: %s" % resource_path)
                    image_name = "%s%s" % (asset_id or ("asset-%03d" % index), source.suffix.lower())
                    image_target = temporary / "EPUB/images" / image_name
                    shutil.copyfile(source, image_target)
                image_href = "images/%s" % image_name
                copied_images[asset_id] = image_href
                manifest.append({"id": "img-%03d" % index, "href": image_href, "media": media_type(image_target), "properties": "cover-image" if str(asset.get("role", "")) == "cover" else ""})
                target.write_text(
                    '<?xml version="1.0" encoding="utf-8"?>'
                    '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="ja"><head>'
                    '<title>%s</title><link rel="stylesheet" type="text/css" href="%s"/>'
                    '</head><body class="image-page"><div><img src="%s" alt=""/></div></body></html>' % (
                        html.escape(title),
                        html.escape(str(PurePosixPath(*([".."] * max(1, len(PurePosixPath(relative_path).parts) - 1)), "styles/style.css"))),
                        html.escape(str(PurePosixPath(*([".."] * max(1, len(PurePosixPath(relative_path).parts) - 1)), image_href))),
                    ), encoding="utf-8",
                )
            else:
                raise SystemExit("unsupported spine node type: %s" % node_type)
            manifest.append({"id": item_id, "href": relative_path, "media": "application/xhtml+xml", "properties": ""})
            spine_ids.append(item_id)

        unused = sorted(set(stable_by_path) - used_content_paths)
        if unused:
            raise SystemExit("stable items target XHTML absent from structure plan: %s" % ", ".join(unused))

        nav_items = "".join('<li><a href="%s">%s</a></li>' % (html.escape(path), html.escape(label)) for path, label in nav_entries)
        if not nav_items:
            first = relative_from_epub(spine_nodes[0].get("path"))
            nav_items = '<li><a href="%s">%s</a></li>' % (html.escape(first), html.escape(title))
        (temporary / "EPUB/nav.xhtml").write_text(
            '<?xml version="1.0" encoding="utf-8"?>'
            '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="ja">'
            '<head><title>%s</title></head><body><nav epub:type="toc"><ol>%s</ol></nav></body></html>' % (html.escape(title), nav_items),
            encoding="utf-8",
        )
        manifest.extend([
            {"id": "nav", "href": "nav.xhtml", "media": "application/xhtml+xml", "properties": "nav"},
            {"id": "css", "href": "styles/style.css", "media": "text/css", "properties": ""},
        ])
        creator = '<dc:creator>%s</dc:creator>' % html.escape(author) if author else ""
        manifest_xml = "".join(
            '<item id="%s" href="%s" media-type="%s"%s/>' % (
                html.escape(value["id"], quote=True), html.escape(value["href"], quote=True),
                html.escape(value["media"], quote=True),
                (' properties="%s"' % html.escape(value["properties"], quote=True)) if value["properties"] else "",
            ) for value in manifest
        )
        spine_xml = "".join('<itemref idref="%s"/>' % html.escape(value, quote=True) for value in spine_ids)
        (temporary / "EPUB/package.opf").write_text(
            '<?xml version="1.0" encoding="utf-8"?>'
            '<package xmlns="http://www.idpf.org/2007/opf" unique-identifier="bookid" version="3.0">'
            '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
            '<dc:identifier id="bookid">%s</dc:identifier><dc:title>%s</dc:title>%s<dc:language>ja</dc:language>'
            '</metadata><manifest>%s</manifest><spine page-progression-direction="rtl">%s</spine></package>' % (
                html.escape(identifier), html.escape(title), creator, manifest_xml, spine_xml,
            ), encoding="utf-8",
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "w") as archive:
            archive.write(temporary / "mimetype", "mimetype", compress_type=zipfile.ZIP_STORED)
            for file in sorted(temporary.rglob("*")):
                if file.is_file() and file.name != "mimetype":
                    archive.write(file, file.relative_to(temporary).as_posix(), compress_type=zipfile.ZIP_DEFLATED)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


if __name__ == "__main__":
    main()
'''

    files = {
        "tools/audit_common.py": audit_common,
        "tools/validate_final_epub.py": validator,
        "tools/build_final_epub.py": builder,
    }
    for relative, content in files.items():
        target = folder / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        try:
            target.chmod(0o755)
        except OSError:
            pass
    return list(files)


def build_compact_rebuild_source(
    *,
    stable_records: Sequence[dict],
    atomic_spans: Sequence[dict],
    assets_manifest: dict,
    structure_plan: dict,
    ledger_summary: dict,
    global_anomalies: dict,
    authority_validation: dict,
    reference_report: dict,
    preflight: dict,
    full_fusion_sha256: str,
) -> dict:
    return {
        "schema": "novel_formatter.ai_publication_rebuild_source.v4_compact",
        "package_mode": "compact",
        "export_configuration": {
            "include_scan_pages": False,
            "scan_evidence_mode": "page_aggregated_required_visual_only",
            "include_publication_images": True,
            "include_publication_reference_evidence": bool(reference_report.get("available")),
            "publication_image_storage": assets_manifest.get("publication_image_storage"),
            "full_fusion_mode": "stored_once_as_lossless_evidence",
            "duplicate_full_text_views": False,
            "absolute_local_paths_exported": False,
        },
        "text_authority": {
            "path": "12_stable_text_map.jsonl",
            "meta_path": "12_stable_text_map.meta.json",
            "role": "sole_default_target_authority",
            "item_count": len(stable_records),
            "validation": authority_validation,
        },
        "full_fusion_evidence_ref": {
            "path": FULL_FUSION_FILENAME,
            "sha256": full_fusion_sha256,
            "structure_document_json_pointer": "/structure_document",
            "format_manifest_json_pointer": "/format_manifest",
        },
        "atomic_span_map": {
            "path": "13_atomic_span_map.jsonl",
            "span_count": len(atomic_spans),
            "coverage_policy": "exactly_once",
        },
        "assets_manifest_ref": "05_assets_manifest.json",
        "output_structure_plan_ref": "15_output_structure_plan.json",
        "page_column_ledger": {"path": "16_page_column_ledger.json", "summary": ledger_summary},
        "global_text_anomalies": {"path": "14_global_text_anomalies.json", "summary": global_anomalies.get("summary", {})},
        "publication_reference": {
            "available": bool(reference_report.get("available")),
            "directory": "reference" if reference_report.get("available") else None,
            "may_not_replace_proposed_text": True,
        },
        "recommended_build_strategy": preflight.get("recommended_build_strategy"),
        "final_two_pass_audit_contract": {
            "validator": "tools/validate_final_epub.py",
            "must_reopen_generated_epub": True,
            "must_compare_stable_items_exactly_once": True,
            "must_compare_atomic_spans_exactly_once": True,
            "must_compare_page_column_ledger": True,
            "delivery_prohibited_on_any_fatal_failure": True,
        },
    }


def estimate_export_size(package: dict, *, package_mode: str = "compact") -> dict:
    """Fast conservative prediction used by the GUI before writing files."""
    requested = str(package_mode or "compact").strip().lower()
    if requested in {"disagreement_v4", "v4", "adjudication", "conflict_only"}:
        from engine.ai_disagreement_package_v4 import build_disagreement_records
        records, summary = build_disagreement_records(package)
        conflict_count = int(summary.get("model_action_required_count", 0) or 0)
        pages = {int(record.get("page", 0) or 0) for record in records if record.get("model_action_required")}
        raw_json = len(_json_bytes(_redact_absolute_paths(package)))
        estimated_zip = int(raw_json * 0.10 + conflict_count * 14_000 + 7_000_000)
        return {
            "package_mode": "disagreement_v4",
            "estimated_zip_size_mb": round(estimated_zip / (1024 * 1024), 1),
            "estimated_file_count": int(12 + conflict_count),
            "visual_page_count": len({page for page in pages if page > 0}),
            "fatal_visual_evidence_count": int(summary.get("status_counts", {}).get("full_conflict", 0)),
            "review_visual_evidence_count": conflict_count,
            "publication_image_bytes": 0,
            "prediction_is_conservative": True,
        }
    mode = "forensic" if requested == "forensic" else "compact"
    editable = len(package.get("editable_items") or [])
    raw_json = len(_json_bytes(_redact_absolute_paths(package)))
    publication_bytes = 0
    visual_pages: set[int] = set()
    fatal_estimate = 0
    review_estimate = 0
    for asset in package.get("assets") or []:
        if not isinstance(asset, dict):
            continue
        _role, include, _scan = _classify_asset(asset)
        path = Path(str(asset.get("image_path", "") or "")).expanduser()
        if include and path.is_file():
            publication_bytes += int(path.stat().st_size)
    for item in package.get("editable_items") or []:
        if not isinstance(item, dict):
            continue
        candidates = [str(value.get("text", "") or "") for value in item.get("candidates") or [] if isinstance(value, dict)]
        unique = {re.sub(r"\s+", "", value) for value in candidates if value}
        text = str(item.get("edited_text", "") or "")
        reasons = set(item.get("risk_reasons") or [])
        visually_risky = len(unique) >= 3 or bool(reasons & {"numeric_disagreement", "possible_missing_column", "possible_column_order_error"}) or any(value in text for value in ("□", "■", "�"))
        if visually_risky:
            visual_pages.add(int(item.get("page", 0) or 0))
            review_estimate += 1
    if mode == "forensic":
        visual_file_count = max(0, editable * 2)
        visual_bytes = max(2_000_000, editable * 22_000)
        duplicate_publication = publication_bytes
    else:
        visual_file_count = len({value for value in visual_pages if value > 0}) * 2
        visual_bytes = visual_file_count * 45_000
        duplicate_publication = 0
    estimated_zip = int(raw_json * 0.10 + publication_bytes + duplicate_publication + visual_bytes + 7_000_000)
    return {
        "package_mode": mode,
        "estimated_zip_size_mb": round(estimated_zip / (1024 * 1024), 1),
        "estimated_file_count": int(40 + visual_file_count + min(editable, 500)),
        "visual_page_count": len({value for value in visual_pages if value > 0}),
        "fatal_visual_evidence_count": fatal_estimate,
        "review_visual_evidence_count": review_estimate,
        "publication_image_bytes": publication_bytes,
        "prediction_is_conservative": True,
    }

def export_ai_publication_bundle_v2(
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
    """Export a v3 repair package with one frozen text authority.

    The current GUI edited text is the only default target.  Publication EPUB
    evidence is opt-in and may add structure/Ruby evidence, but can never replace
    proposed_text silently.
    """
    source_package = copy.deepcopy(package)
    package = copy.deepcopy(package)
    package_mode = "forensic" if str(package_mode or "forensic").strip().lower() in {"forensic", "full", "complete"} else "compact"
    compact_mode = package_mode == "compact"
    for key in ("publication_reference", "reference_epub", "publication_reference_epub"):
        package.pop(key, None)
    reference_package = copy.deepcopy(package)
    reference_path = Path(publication_reference_path).expanduser() if publication_reference_path else None
    if include_publication_reference:
        if not reference_path or not reference_path.is_file() or reference_path.suffix.lower() != ".epub":
            raise repair.AIRepairEpubError("已勾选出版参考证据，但没有选择有效的 EPUB 文件。")
        reference_package["publication_reference"] = {
            "epub_path": str(reference_path),
            "authority": "explicit_user_selected_publication_reference",
        }

    requested_mode = str(mode or "one_pass").strip().lower()
    if requested_mode not in repair._ALLOWED_MODES:
        raise repair.AIRepairEpubError(f"不支持的 AI 修复证据模式：{requested_mode}")
    mode = "one_pass"
    root = Path(output_directory).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    base = _safe_name(str(bundle_name or "AI修复包").strip() or "AI修复包")
    folder_label = f"{base}_v3" if base.endswith("AI修复包") else f"{base}_AI修复包_v3"
    folder = root / folder_label
    if folder.exists():
        folder = root / f"{folder_label}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    folder.mkdir(parents=True, exist_ok=False)
    framework_dir = folder / "framework"
    framework_dir.mkdir(parents=True, exist_ok=True)
    resources_dir = folder / "resources"
    framework_name = "resource_mapping_framework.epub"
    final_name = _final_output_filename(primary_doc, package)
    framework_path = framework_dir / framework_name
    framework_audit_path = framework_dir / "framework_audit.json"
    guide_path = folder / "00_MODEL_COMMAND.md"
    full_fusion_path = folder / FULL_FUSION_FILENAME
    stable_map_path = folder / "12_stable_text_map.jsonl"
    atomic_map_path = folder / "13_atomic_span_map.jsonl"
    anomaly_path = folder / "14_global_text_anomalies.json"
    structure_plan_path = folder / "15_output_structure_plan.json"
    ledger_path = folder / "16_page_column_ledger.json"

    mandatory_paths = [
        "00_MODEL_COMMAND.md", "01_manifest.json", "02_book_identity.json",
        "03_rebuild_source.json", "04_structure_preflight.json", "05_assets_manifest.json",
        "06_risk_queue.json", "07_boundary_windows.json", "08_term_consistency.json",
        "09_style_profile.json", "10_output_contract.json", "11_final_audit_rules.json",
        "12_stable_text_map.jsonl", "13_atomic_span_map.jsonl",
        "14_global_text_anomalies.json", "15_output_structure_plan.json",
        "16_page_column_ledger.json", "reading/full_book_with_item_markers.txt",
        f"framework/{framework_name}", "framework/framework_audit.json",
        FULL_FUSION_FILENAME, "visual_evidence/visual_evidence_manifest.json",
        "tools/audit_common.py", "tools/build_final_epub.py", "tools/validate_final_epub.py",
    ]

    def apply_global_anomaly_rules(stable_records: list[dict], anomalies: dict) -> None:
        by_id = {str(record.get("item_id", "") or ""): record for record in stable_records}
        affected: dict[str, set[str]] = defaultdict(set)
        for key in ("duplicate_clusters", "moved_span_clusters", "unsupported_insertions", "orphan_source_spans", "cross_item_order_conflicts"):
            for anomaly in anomalies.get(key) or []:
                if anomaly.get("severity") not in {"fatal", "high"}:
                    continue
                for field in ("item_id", "source_item_id", "destination_item_id", "previous_item_id", "current_item_id"):
                    if anomaly.get(field): affected[str(anomaly[field])].add(key)
                for occurrence in anomaly.get("occurrences") or []:
                    if occurrence.get("item_id"): affected[str(occurrence["item_id"])].add(key)
        for item_id, reason_set in affected.items():
            record = by_id.get(item_id)
            if not record: continue
            record.update({"edit_policy": "review_required", "model_action_required": True, "text_locked": False, "final_text_requires_model_decision": True})
            record["unlock_reasons"] = list(dict.fromkeys([*(record.get("unlock_reasons") or []), "global_duplicate_or_move"]))
            record["risk_flags"] = list(dict.fromkeys([*(record.get("risk_flags") or []), *sorted(reason_set)]))

    try:
        fusion_validation = _validate_lossless_fusion_package(package)
        portable_package = _redact_absolute_paths(package)
        full_fusion_path.write_bytes(_json_bytes(portable_package))
        restored_fusion = json.loads(full_fusion_path.read_text(encoding="utf-8"))
        if restored_fusion != portable_package:
            raise repair.AIRepairEpubError("完整融合 JSON 写出后语义校验失败。")
        full_items_by_id = _full_fusion_items_by_id(package)
        expected_full_count = int(package.get("editable_count", len(package.get("editable_items") or [])) or 0)
        if len(full_items_by_id) != expected_full_count:
            raise repair.AIRepairEpubError(f"完整融合 JSON 的稳定 row_id 不完整或重复：应有 {expected_full_count}，实际唯一 {len(full_items_by_id)}。")

        publication_image_paths = _publication_image_source_paths(package)
        # Keep the original block sequence until ``build_repair_document`` has
        # resolved every primary_block_index.  Removing scan-page IMAGE_REF
        # blocks here shifts later indices and can make valid row IDs vanish
        # from the framework.  ``export_ai_repair_epub`` performs the image
        # filtering only after row-addressable blocks have been rebuilt.
        publication_doc = primary_doc
        guide_text = repair._publication_guide_text(
            mode,
            framework_name=f"framework/{framework_name}",
            final_name=final_name,
            publication_reference_available=bool(include_publication_reference),
        )
        guide_path.write_text(guide_text, encoding="utf-8")
        epub_report = repair.export_ai_repair_epub(
            publication_doc, package, framework_path, mode=mode, vertical=vertical,
            css_template=css_template, custom_css=custom_css, workflow="publication",
            publication_guide=guide_text, publication_final_name=final_name,
            publication_image_paths=publication_image_paths,
        )
        repair_map = repair.read_ai_repair_map(framework_path)
        _enrich_repair_items(primary_doc, repair_map, package)
        _strip_framework_work_payloads(framework_path)
        framework_inventory = _framework_inventory(framework_path)
        assets_manifest = _copy_assets(
            package,
            list(repair_map.get("items") or []),
            resources_dir,
            storage_mode="framework" if compact_mode else "copy",
            framework_inventory=framework_inventory,
            framework_name=framework_name,
        )

        # P0 authority chain: freeze current edited text before creating any view.
        stable_records = _stable_text_records(repair_map, assets_manifest, None)
        page_column_ledger = v3.build_page_column_ledger(
            package, stable_records, full_items_by_id, compact=compact_mode,
        )
        _apply_page_ledger_rules(stable_records, page_column_ledger)

        reference_report = {"available": False, "alignment": [], "ruby": [], "ruby_group_count": 0}
        if include_publication_reference:
            reference_report = v3.export_reference_evidence(reference_package, stable_records, folder)
            if not reference_report.get("available"):
                raise repair.AIRepairEpubError("出版参考 EPUB 解析失败，未生成可用参考证据。")
            # v3 support uses reference_* names; normalize for the stable map helper.
            normalized_alignment = []
            for alignment in reference_report.get("alignment") or []:
                value = copy.deepcopy(alignment)
                value["inline_tokens"] = copy.deepcopy(value.get("reference_inline_tokens") or [])
                value["ruby_groups"] = copy.deepcopy(value.get("reference_ruby_groups") or [])
                normalized_alignment.append(value)
            reference_report["alignment"] = normalized_alignment
            _apply_reference_alignment(stable_records, reference_report)

        global_anomalies = v3.build_global_anomalies(stable_records, full_items_by_id)
        apply_global_anomaly_rules(stable_records, global_anomalies)

        # Reading/evidence/structure are projections of the already frozen map.
        chapter_manifest, structure_plan, authority_validation = _write_reading_and_evidence(
            folder, repair_map, stable_records, full_items_by_id, assets_manifest=assets_manifest,
            package_mode=package_mode,
        )
        atomic_spans = v3.build_atomic_span_map(stable_records, full_items_by_id)
        _write_stable_text_map(stable_map_path, stable_records, compact=compact_mode)
        _write_stable_text_map(atomic_map_path, atomic_spans, compact=compact_mode, atomic=True)
        if compact_mode:
            _write_json(folder / "12_stable_text_map.meta.json", {
                "schema": "novel_formatter.ai_publication_stable_text_map_meta.v4",
                "text_authority": "current_edited_text",
                "character_fused_text_is_evidence_only": True,
                "ai_adjudicated_text_is_evidence_only": True,
                "unlock_reason_vocabulary": sorted({
                    str(reason) for record in stable_records for reason in (record.get("unlock_reasons") or []) if str(reason)
                }),
                "item_count": len(stable_records),
            })
        _write_json(anomaly_path, global_anomalies)
        _write_json(structure_plan_path, structure_plan)
        _write_json(ledger_path, page_column_ledger)

        book_identity = _book_identity(primary_doc, source_package, repair_map, final_name, framework_inventory, assets_manifest=assets_manifest, cover_identity=None)
        preflight = _structure_preflight(publication_doc, repair_map, framework_inventory, assets_manifest, book_identity, chapter_manifest=chapter_manifest)
        ledger_summary = page_column_ledger.get("summary") or {}
        chapter_sequence = _chapter_sequence_diagnostics(chapter_manifest)
        preflight.update({
            "planned_spine_node_count": len(structure_plan.get("spine_nodes") or []),
            "standalone_image_xhtml_count": int(structure_plan.get("standalone_image_xhtml_count", 0) or 0),
            "output_structure_plan_path": "15_output_structure_plan.json",
            "page_column_ledger_path": "16_page_column_ledger.json",
            "page_column_coverage_complete": bool(ledger_summary.get("coverage_complete", True)),
            "fatal_unmapped_body_slot_count": int(ledger_summary.get("fatal_unmapped_body_slot_count", 0) or 0),
            "unverifiable_page_count": int(ledger_summary.get("unverifiable_page_count", 0) or 0),
            "fatal_preflight": int(ledger_summary.get("fatal_unmapped_body_slot_count", 0) or 0) > 0,
            "text_authority_validation": authority_validation,
            "chapter_structure_review_required": bool(
                (len(stable_records) > 500 and len(chapter_manifest) == 1)
                or chapter_sequence.get("requires_review")
            ),
            "chapter_detection_warning": (
                "large_book_single_logical_chapter"
                if len(stable_records) > 500 and len(chapter_manifest) == 1
                else "incomplete_numeric_chapter_sequence"
                if chapter_sequence.get("requires_review")
                else None
            ),
            "chapter_sequence_diagnostics": chapter_sequence,
        })
        if preflight["fatal_preflight"]:
            preflight.setdefault("fatal_blockers", []).append("page_column_ledger_has_unmapped_body_columns")
            preflight["recommended_build_strategy"] = "hybrid_rebuild"

        risk_queue = _risk_queue(repair_map)
        risk_queue["global_anomaly_summary"] = copy.deepcopy(global_anomalies.get("summary") or {})
        risk_queue["page_column_ledger_summary"] = copy.deepcopy(ledger_summary)
        risk_queue["review_required_item_count"] = sum(record.get("edit_policy") == "review_required" for record in stable_records)
        boundary_windows = _boundary_windows(repair_map, assets_manifest, package_mode=package_mode)
        terms = _term_consistency(repair_map, full_items_by_id, reference_report.get("alignment") or [])
        style = _style_profile(repair_map, framework_inventory, vertical, stable_records)
        visual_manifest = v3.export_visual_evidence(
            primary_doc, package, repair_map, stable_records, global_anomalies, folder,
            page_column_ledger=page_column_ledger, package_mode=package_mode,
        )
        metadata_snapshot = {
            "title": book_identity.get("title"), "author": book_identity.get("author"),
            "identifier": book_identity.get("identifier") or book_identity.get("isbn") or book_identity.get("baseline_book_sha256"),
            "language": book_identity.get("language", "ja"),
        }
        output_contract = _output_contract(
            final_name, str(book_identity.get("language", "ja") or "ja"), assets_manifest, preflight,
            atomic_span_count=len(atomic_spans), global_anomalies=global_anomalies,
            structure_plan=structure_plan, reference_available=bool(reference_report.get("available")),
            page_column_ledger=page_column_ledger, existing_metadata=metadata_snapshot,
        )
        audit_rules = _final_audit_rules(
            preflight, assets_manifest, atomic_span_count=len(atomic_spans),
            global_anomalies=global_anomalies, structure_plan=structure_plan,
            page_column_ledger=page_column_ledger, reference_available=bool(reference_report.get("available")),
        )
        framework_audit = _framework_audit(preflight, book_identity, assets_manifest, framework_inventory)
        tool_paths = _write_audit_tools(folder)

        image_anchors = [
            {key: asset.get(key) for key in (
                "asset_id", "resource_path", "role", "sha256", "stable_item_before", "stable_item_after",
                "display_mode", "planned_asset_xhtml", "force_split_before", "force_split_after", "asset_order",
            )}
            for asset in assets_manifest.get("assets", [])
            if asset.get("include_in_final_epub") and asset.get("export_status") == "exported"
        ]
        item_to_chapter_file = {
            str(record.get("item_id", "")): {
                "chapter_id": record.get("chapter_id"), "technical_part_index": record.get("technical_part_index"),
                "evidence_path": record.get("expected_evidence_path"), "framework_epub_target": record.get("framework_epub_target"),
                "final_html_id": record.get("final_html_id"), "planned_final_xhtml": record.get("planned_final_xhtml"),
                "planned_final_target": record.get("planned_final_target"), "edit_policy": record.get("edit_policy"),
            }
            for record in stable_records
        }
        rebuild_source = {
            "schema": REBUILD_SCHEMA,
            "package_profile": PACKAGE_PROFILE,
            "export_configuration": {
                "include_scan_pages": False, "scan_evidence_mode": "selected_high_value_crops_only",
                "include_risk_crops": True, "include_page_thumbnails": "selected_context_only",
                "include_publication_images": True,
                "include_publication_reference_evidence": bool(reference_report.get("available")),
                "publication_reference_selection": "explicit_user_opt_in_only",
                "framework_mode": "clean_mapping_only",
                "physical_evidence_mode": "lossless_full_fusion_plus_selective_visual",
                "boundary_window_mode": "syntax_aware_dynamic_windows",
                "full_fusion_mode": "evidence_only_not_target_authority",
                "archive_mode": "none", "deduplicate_rebuild_text": True, "deduplicate_asset_records": True,
                "absolute_local_paths_exported": False,
            },
            "text_authority": {
                "path": "12_stable_text_map.jsonl", "role": "sole_default_target_authority",
                "item_count": len(stable_records), "contains_proposed_text": True,
                "evidence_path": FULL_FUSION_FILENAME, "evidence_item_count": len(full_items_by_id),
                "character_fused_text_is_evidence_only": True, "ai_adjudicated_text_is_evidence_only": True,
                "validation": authority_validation,
                "atomic_coverage_path": "13_atomic_span_map.jsonl", "page_column_ledger_path": "16_page_column_ledger.json",
            },
            "recommended_build_strategy": preflight["recommended_build_strategy"],
            "framework_structure_authoritative": preflight["framework_structure_authoritative"],
            "stable_item_count": len(stable_records), "atomic_span_count": len(atomic_spans),
            "structure_document": _deduplicated_structure_document(package, stable_records, assets_manifest),
            "format_manifest": _deduplicated_format_manifest(package, assets_manifest),
            "assets": assets_manifest.get("assets", []), "image_anchors": image_anchors,
            "output_structure_plan": structure_plan, "page_column_ledger": {"path": "16_page_column_ledger.json", "summary": ledger_summary},
            "publication_reference": {
                "available": bool(reference_report.get("available")), "directory": "reference" if reference_report.get("available") else None,
                "alignment_count": len(reference_report.get("alignment") or []), "ruby_group_count": int(reference_report.get("ruby_group_count", 0) or 0),
                "may_not_replace_proposed_text": True,
            },
            "chapter_candidates": copy.deepcopy(preflight.get("chapter_candidates") or []), "chapter_boundaries": chapter_manifest,
            "reading_order": [record.get("item_id") for record in stable_records],
            "stable_text_map": {
                "path": "12_stable_text_map.jsonl", "item_count": len(stable_records), "role": "sole_default_text_authority",
                "locked_consensus_count": sum(record.get("edit_policy") == "locked_consensus" for record in stable_records),
                "review_required_count": sum(record.get("edit_policy") == "review_required" for record in stable_records),
                "structure_only_count": sum(record.get("edit_policy") == "structure_only" for record in stable_records),
            },
            "atomic_span_map": {"path": "13_atomic_span_map.jsonl", "span_count": len(atomic_spans), "coverage_policy": "exactly_once", "reliable_spans_only": True, "proposed_text_fallback_forbidden": True},
            "global_text_anomalies": {"path": "14_global_text_anomalies.json", "summary": global_anomalies.get("summary", {})},
            "visual_evidence": {"directory": "visual_evidence", "record_count": visual_manifest.get("record_count", 0), "exported_crop_count": visual_manifest.get("exported_crop_count", 0)},
            "item_to_chapter_file": item_to_chapter_file,
            "paragraph_map": [{key: record.get(key) for key in (
                "reading_order", "item_id", "row_id", "block_id", "page", "chapter_candidate_id", "chapter_id", "chapter_title",
                "block_type", "primary_block_index", "primary_block_indices", "source_block_ids", "source_column_ids",
                "final_html_id", "planned_final_xhtml", "planned_final_target", "edit_policy",
            )} for record in stable_records],
            "final_two_pass_audit_contract": {
                "must_run_executable_auditor": True, "validator": "tools/validate_final_epub.py",
                "must_generate_complete_final_mapping": True, "must_reopen_generated_epub": True,
                "must_compare_stable_items_exactly_once": True, "must_compare_atomic_spans_exactly_once": True,
                "must_compare_page_column_ledger": True, "must_resolve_global_anomalies": True,
                "deliver_only_second_pass": True,
                "deliver_final_epub_and_audit_files": True, "delivery_prohibited_on_any_fatal_failure": True,
            },
            "final_validation_requirements": audit_rules.get("required_comparisons", []),
        }
        if compact_mode:
            rebuild_source = build_compact_rebuild_source(
                stable_records=stable_records,
                atomic_spans=atomic_spans,
                assets_manifest=assets_manifest,
                structure_plan=structure_plan,
                ledger_summary=ledger_summary,
                global_anomalies=global_anomalies,
                authority_validation=authority_validation,
                reference_report=reference_report,
                preflight=preflight,
                full_fusion_sha256=_sha256_bytes(full_fusion_path.read_bytes()),
            )

        _write_json(folder / "02_book_identity.json", book_identity)
        _write_json(folder / "03_rebuild_source.json", rebuild_source)
        _write_json(folder / "04_structure_preflight.json", preflight)
        _write_json(folder / "05_assets_manifest.json", assets_manifest)
        _write_json(folder / "06_risk_queue.json", risk_queue)
        _write_json(folder / "07_boundary_windows.json", boundary_windows)
        _write_json(folder / "08_term_consistency.json", terms)
        _write_json(folder / "09_style_profile.json", style)
        _write_json(folder / "10_output_contract.json", output_contract)
        _write_json(folder / "11_final_audit_rules.json", audit_rules)
        _write_json(framework_audit_path, framework_audit)

        edit_counts = Counter(str(record.get("edit_policy", "") or "") for record in stable_records)
        publication_manifest = {
            "schema": BUNDLE_SCHEMA, "package_version": 3,
            "package_display_name": f"AI修复包 v3（{'标准紧凑包' if compact_mode else '完整取证包'}）",
            "purpose": "ai_epub_repair_package_with_locked_text_atomic_coverage",
            "purpose_v3_detail": "single_text_authority_and_page_column_ledger",
            "legacy_purpose_alias": "ai_publication_master_with_lossless_fusion_evidence",
            "package_profile": PACKAGE_PROFILE, "package_mode": package_mode,
            "export_configuration": copy.deepcopy(rebuild_source["export_configuration"]),
            "return_to_formatter": False, "mode": mode, "requested_mode_compatibility_alias": requested_mode,
            "package_id": str(repair_map.get("package_id", "") or _sha256_bytes((base + str(repair_map.get("structure_sha256", ""))).encode("utf-8"))[:24]),
            "book_id": str(repair_map.get("baseline_book_sha256", "") or repair_map.get("structure_sha256", "") or ""),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "recommended_build_strategy": preflight["recommended_build_strategy"], "framework_structure_authoritative": preflight["framework_structure_authoritative"],
            "required_final_output": final_name, "entrypoint": "00_MODEL_COMMAND.md",
            "files": {
                "book_identity": "02_book_identity.json", "rebuild_source": "03_rebuild_source.json", "structure_preflight": "04_structure_preflight.json",
                "assets_manifest": "05_assets_manifest.json", "risk_queue": "06_risk_queue.json", "boundary_windows": "07_boundary_windows.json",
                "term_consistency": "08_term_consistency.json", "style_profile": "09_style_profile.json", "output_contract": "10_output_contract.json",
                "final_audit_rules": "11_final_audit_rules.json", "stable_text_map": "12_stable_text_map.jsonl", "atomic_span_map": "13_atomic_span_map.jsonl",
                "global_text_anomalies": "14_global_text_anomalies.json", "output_structure_plan": "15_output_structure_plan.json",
                "page_column_ledger": "16_page_column_ledger.json", "full_reading_text": "reading/full_book_with_item_markers.txt",
                "framework_epub": f"framework/{framework_name}", "framework_audit": "framework/framework_audit.json",
                "full_fusion_evidence": FULL_FUSION_FILENAME, "visual_evidence_manifest": "visual_evidence/visual_evidence_manifest.json",
                "audit_tools": tool_paths, "reference_directory": "reference" if reference_report.get("available") else None,
            },
            "text_authority": {
                "path": "12_stable_text_map.jsonl", "item_count": len(stable_records), "sole_default_authority": True,
                "lossless": True, "validation": authority_validation, "evidence_path": FULL_FUSION_FILENAME, "evidence_item_count": len(full_items_by_id),
                "reading_and_chapter_evidence_match_stable_projection": bool(authority_validation.get("same_text_authority_across_views")),
                "atomic_span_exactly_once_coverage_required": True, "page_column_ledger_required": True,
            },
            "publication_reference": {
                "available": bool(reference_report.get("available")), "explicit_opt_in": bool(include_publication_reference),
                "alignment_count": len(reference_report.get("alignment") or []), "ruby_group_count": int(reference_report.get("ruby_group_count", 0) or 0),
            },
            "page_column_ledger_summary": ledger_summary,
            "chapters": chapter_manifest, "stable_item_count": len(stable_records), "atomic_span_count": len(atomic_spans),
            "edit_policy_counts": dict(edit_counts), "global_anomaly_summary": global_anomalies.get("summary", {}),
            "image_count": int(assets_manifest.get("publication_resource_count", 0) or 0),
            "publication_image_count": int(assets_manifest.get("publication_image_count", 0) or 0),
            "scan_page_count_detected": int(assets_manifest.get("scan_page_count_detected", 0) or 0), "scan_page_count_exported": 0,
            "final_epub_expected_image_count": int(assets_manifest.get("final_epub_expected_image_count", 0) or 0),
            "visual_evidence_record_count": int(visual_manifest.get("record_count", 0) or 0),
            "visual_evidence_exported_crop_count": int(visual_manifest.get("exported_crop_count", 0) or 0),
            "visual_evidence_file_count": int(visual_manifest.get("visual_file_count", visual_manifest.get("exported_crop_count", 0) * 2) or 0),
            "visual_evidence_page_count": int(visual_manifest.get("visual_page_count", visual_manifest.get("source_page_count_touched", 0)) or 0),
            "has_scan_truth": bool(visual_manifest.get("exported_crop_count")), "source_scan_assets_detected": bool(assets_manifest.get("scan_page_count_detected", 0)),
            "statistics": {
                "item_count": len(repair_map.get("items", [])), "logical_chapter_count": len(chapter_manifest), "chapter_count": len(chapter_manifest),
                "planned_content_xhtml_count": int(structure_plan.get("content_xhtml_count", len(chapter_manifest)) or len(chapter_manifest)),
                "planned_spine_node_count": len(structure_plan.get("spine_nodes") or []),
                "standalone_image_xhtml_count": int(structure_plan.get("standalone_image_xhtml_count", 0) or 0),
                "high_risk_count": sum(1 for item in repair_map.get("items", []) if item.get("risk_level") == "high"),
                "medium_risk_count": sum(1 for item in repair_map.get("items", []) if item.get("risk_level") == "medium"),
                "low_risk_count": sum(1 for item in repair_map.get("items", []) if item.get("risk_level") == "low"),
                "no_risk_count": sum(1 for item in repair_map.get("items", []) if item.get("risk_level") == "none"),
                "locked_consensus_count": edit_counts.get("locked_consensus", 0), "review_required_count": edit_counts.get("review_required", 0),
                "structure_only_count": edit_counts.get("structure_only", 0), "atomic_span_count": len(atomic_spans),
                "lossless_full_fusion_item_count": len(full_items_by_id), "framework_image_count": len(framework_inventory.get("images", [])),
                "publication_image_count": assets_manifest.get("publication_image_count", 0), "publication_resource_count": assets_manifest.get("publication_resource_count", 0),
                "scan_page_count_detected": assets_manifest.get("scan_page_count_detected", 0), "scan_page_count_exported": 0,
                "fatal_unmapped_body_slot_count": int(ledger_summary.get("fatal_unmapped_body_slot_count", 0) or 0),
                "unverifiable_page_count": int(ledger_summary.get("unverifiable_page_count", 0) or 0),
            },
            "integrity": {
                "hash_algorithm": "sha256", "hash_scope_excludes": ["01_manifest.json"],
                "structure_sha256": repair_map.get("structure_sha256", ""), "layout_sha256": repair_map.get("layout_sha256", ""),
                "baseline_book_sha256": repair_map.get("baseline_book_sha256", ""), "framework_contains_embedded_manifest": False,
                "absolute_local_paths_redacted": True,
            },
        }
        _write_json(folder / "01_manifest.json", publication_manifest)

        with zipfile.ZipFile(framework_path, "r") as archive:
            framework_names = archive.namelist()
            if not framework_names or framework_names[0] != "mimetype" or archive.getinfo("mimetype").compress_type != zipfile.ZIP_STORED:
                raise repair.AIRepairEpubError("出版框架 EPUB 的 mimetype 顺序或压缩方式不合规。")
            if archive.testzip() is not None:
                raise repair.AIRepairEpubError("出版框架 EPUB CRC 完整性检查失败。")
            forbidden_work = [name for name in framework_names if name.startswith(repair.AI_REPAIR_ROOT + "/") or name.startswith(repair.AI_PUBLICATION_ROOT + "/") or name in {repair.MAP_PATH, repair.GUIDE_PATH, repair.TEMPLATE_PATH}]
            if forbidden_work: raise repair.AIRepairEpubError("干净资源框架仍残留 AI 工作证据。")
            if any(Path(name).name.lower().startswith("source_page_") for name in framework_names):
                raise repair.AIRepairEpubError("干净资源框架仍残留扫描页图片。")

        framework_audit["framework_sha256"] = _sha256_bytes(framework_path.read_bytes())
        framework_audit["embedded_work_paths_present_in_framework"] = []
        framework_audit["final_output_must_not_match_framework_sha256"] = True
        _write_json(framework_audit_path, framework_audit)
        forbidden_resource_files = [path.name for path in resources_dir.iterdir() if path.is_file() and path.name.lower().startswith("source_page_")] if resources_dir.exists() else []
        if forbidden_resource_files: raise repair.AIRepairEpubError("resources/ 中发现不应导出的扫描页图片。")

        if compact_mode and "12_stable_text_map.meta.json" not in mandatory_paths:
            mandatory_paths.append("12_stable_text_map.meta.json")
        hashes = _file_hash_inventory(folder, exclude={"01_manifest.json"})
        missing = [relative for relative in mandatory_paths if not (folder / relative).is_file()]
        if compact_mode:
            core_hashes = {key: value for key, value in hashes.items() if not key.startswith("visual_evidence/pages/")}
            publication_manifest["all_files_sha256"] = core_hashes
            visual_manifest_path = folder / "visual_evidence/visual_evidence_manifest.json"
            publication_manifest["visual_evidence_hash_tree"] = {
                "manifest_path": "visual_evidence/visual_evidence_manifest.json",
                "manifest_sha256": _sha256_bytes(visual_manifest_path.read_bytes()) if visual_manifest_path.is_file() else "",
                "file_count": int(visual_manifest.get("visual_file_count", 0) or 0),
                "image_sha256": sorted({
                    str(record.get("overview_sha256", "") or "") for record in visual_manifest.get("records") or [] if record.get("overview_sha256")
                } | {
                    str(record.get("crop_sheet_sha256", "") or "") for record in visual_manifest.get("records") or [] if record.get("crop_sheet_sha256")
                }),
            }
        else:
            publication_manifest["all_files_sha256"] = hashes
        publication_manifest["file_count"] = len(_all_regular_files(folder))
        publication_manifest["mandatory_files"] = mandatory_paths
        publication_manifest["mandatory_files_present"] = not missing
        publication_manifest["missing_mandatory_files"] = missing
        publication_manifest["integrity"].update({
            "framework_sha256": hashes.get(f"framework/{framework_name}", ""), "stable_text_map_sha256": hashes.get("12_stable_text_map.jsonl", ""),
            "atomic_span_map_sha256": hashes.get("13_atomic_span_map.jsonl", ""), "global_text_anomalies_sha256": hashes.get("14_global_text_anomalies.json", ""),
            "output_structure_plan_sha256": hashes.get("15_output_structure_plan.json", ""), "page_column_ledger_sha256": hashes.get("16_page_column_ledger.json", ""),
            "full_reading_text_sha256": hashes.get("reading/full_book_with_item_markers.txt", ""), "full_fusion_evidence_sha256": hashes.get(FULL_FUSION_FILENAME, ""),
            "validator_sha256": hashes.get("tools/validate_final_epub.py", ""),
        })
        _write_json(folder / "01_manifest.json", publication_manifest)
        if missing: raise repair.AIRepairEpubError(f"AI 修复包缺少强制文件：{', '.join(missing)}")

        zip_path = folder.with_suffix(".zip")
        if create_zip: repair._write_bundle_zip(folder, zip_path)
        else: zip_path = Path("")
        return {
            "folder": str(folder), "zip_path": str(zip_path) if create_zip else "", "framework_epub": str(framework_path),
            "package_mode": package_mode,
            "primary_framework_epub": str(framework_path), "fusion_json": str(full_fusion_path), "guide": str(guide_path),
            "manifest": str(folder / "01_manifest.json"), "preflight": str(folder / "04_structure_preflight.json"),
            "rebuild_source": str(folder / "03_rebuild_source.json"), "output_contract": str(folder / "10_output_contract.json"),
            "final_audit_rules": str(folder / "11_final_audit_rules.json"), "stable_text_map": str(stable_map_path),
            "atomic_span_map": str(atomic_map_path), "global_text_anomalies": str(anomaly_path), "output_structure_plan": str(structure_plan_path),
            "page_column_ledger": str(ledger_path), "visual_evidence_manifest": str(folder / "visual_evidence/visual_evidence_manifest.json"),
            "final_output_name": final_name, "mode": mode, "recommended_build_strategy": preflight["recommended_build_strategy"],
            "framework_structure_authoritative": preflight["framework_structure_authoritative"], "editable_count": int(epub_report.get("editable_count", 0) or 0),
            "stable_item_count": len(stable_records), "atomic_span_count": len(atomic_spans), "chapter_count": len(chapter_manifest),
            "image_count": len(framework_inventory.get("images", [])), "publication_resource_count": int(assets_manifest.get("publication_resource_count", 0) or 0),
            "publication_image_count": int(assets_manifest.get("publication_image_count", 0) or 0), "scan_page_count_detected": int(assets_manifest.get("scan_page_count_detected", 0) or 0),
            "scan_page_count_exported": 0, "source_asset_record_count": int(assets_manifest.get("source_asset_record_count", 0) or 0),
            "unique_source_file_count": int(assets_manifest.get("unique_source_file_count", 0) or 0),
            "high_risk_count": int(publication_manifest["statistics"]["high_risk_count"]), "medium_risk_count": int(publication_manifest["statistics"]["medium_risk_count"]),
            "low_risk_count": int(publication_manifest["statistics"]["low_risk_count"]), "no_risk_count": int(publication_manifest["statistics"]["no_risk_count"]),
            "locked_consensus_count": int(edit_counts.get("locked_consensus", 0)), "review_required_count": int(edit_counts.get("review_required", 0)),
            "structure_only_count": int(edit_counts.get("structure_only", 0)), "lossless_full_fusion_item_count": len(full_items_by_id),
            "collapsed_consensus_count": sum(1 for item in repair_map.get("items", []) if item.get("evidence_tier") == "collapsed_consensus"),
            "full_physical_evidence_count": sum(1 for item in repair_map.get("items", []) if item.get("evidence_tier") == "full_physical"),
            "planned_content_xhtml_count": int(structure_plan.get("content_xhtml_count", 0) or 0), "standalone_image_xhtml_count": int(structure_plan.get("standalone_image_xhtml_count", 0) or 0),
            "boundary_window_count": len(boundary_windows.get("risk_windows", [])), "page_boundary_window_count": len(boundary_windows.get("page_boundary_windows", [])),
            "visual_evidence_exported_crop_count": int(visual_manifest.get("exported_crop_count", 0) or 0),
            "visual_evidence_file_count": int(visual_manifest.get("visual_file_count", visual_manifest.get("exported_crop_count", 0) * 2) or 0),
            "visual_evidence_page_count": int(visual_manifest.get("visual_page_count", visual_manifest.get("source_page_count_touched", 0)) or 0),
            "reference_alignment_count": len(reference_report.get("alignment") or []), "reference_ruby_group_count": int(reference_report.get("ruby_group_count", 0) or 0),
            "publication_reference_included": bool(reference_report.get("available")), "page_column_coverage_complete": bool(ledger_summary.get("coverage_complete", True)),
            "fatal_unmapped_body_slot_count": int(ledger_summary.get("fatal_unmapped_body_slot_count", 0) or 0),
            "global_anomaly_summary": global_anomalies.get("summary", {}), "framework_sha256": _sha256_bytes(framework_path.read_bytes()),
            "fusion_sha256": _sha256_bytes(full_fusion_path.read_bytes()), "mandatory_files_present": True,
        }
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        folder.with_suffix(".zip").unlink(missing_ok=True)
        raise

