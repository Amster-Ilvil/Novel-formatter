#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolve an already-OCR'd Novel Formatter project into a fresh adjudication package.

This module is intentionally Qt-free.  The CLI must never start OCR engines or
GUI widgets; it only consumes an existing multi-OCR project snapshot/package.

Fresh adjudication means previous human/AI fusion selections are discarded while
raw Hayai / NDL / 48px candidates, stable rows, geometry and structure remain
unchanged.  A project snapshot is rebuilt through ``export_multi_package`` from
its immutable OCR documents/comparison.  Existing exported packages are reset to
their automatic recommended candidate baseline and resealed.
"""
from __future__ import annotations

import copy
import json
import tempfile
import zipfile
import posixpath
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from engine.headless_adjudication import HeadlessAdjudicationError, candidate_sha256
from engine.ocr_roundtrip_package import (
    _editable_structure_hash,
    _json_detached,
    _seal_package,
    export_multi_package,
)


@dataclass(slots=True)
class ResolvedOcrSource:
    package: dict
    kind: str
    source_path: str
    skeleton_template: str = ""
    previous_authority_count: int = 0
    reset_rows: int = 0


def _fresh_baseline_text(item: dict) -> str:
    candidates = [value for value in (item.get("candidates") or []) if isinstance(value, dict)]
    try:
        index = int(item.get("recommended_model_index", -1))
    except (TypeError, ValueError, OverflowError):
        index = -1
    if 0 <= index < len(candidates):
        text = str(candidates[index].get("text", "") or "")
        if text or not any(str(v.get("text", "") or "") for v in candidates):
            return text
    # Never fall back to original_fused_text here: exported fusion text may be a
    # previous human/AI decision.  Candidate evidence is the only fresh source.
    for candidate in candidates:
        text = str(candidate.get("text", "") or "")
        if text:
            return text
    return ""


def reset_exported_package_for_fresh_adjudication(package: dict) -> tuple[dict, dict]:
    """Discard previous fusion decisions without touching raw OCR candidates."""
    if not isinstance(package, dict):
        raise HeadlessAdjudicationError("多模型 OCR 来源必须是 JSON 对象。")
    result = _json_detached(package)
    before = candidate_sha256(result)
    authority = result.get("canonical_text_authority")
    previous_authority_count = 0
    if isinstance(authority, dict):
        previous_authority_count = int(authority.get("decision_count", 0) or 0)
    reset_rows = 0
    for item in result.get("editable_items") or []:
        if not isinstance(item, dict):
            continue
        fresh = _fresh_baseline_text(item)
        if str(item.get("edited_text", "") or "") != fresh or str(item.get("original_fused_text", "") or "") != fresh:
            reset_rows += 1
        item["original_fused_text"] = fresh
        item["edited_text"] = fresh
        item["delete_intentionally"] = False
        # Old external visual/context evidence is itself a previous adjudication
        # artifact.  Fresh Codex runs must reason from candidates/context again.
        item.pop("visual_evidence", None)
    result.pop("canonical_text_authority", None)
    result.pop("ai_visual_evidence_summary", None)
    result.pop("ai_adjudication", None)
    result["editable_structure_sha256"] = _editable_structure_hash(result.get("editable_items") or [])
    _seal_package(result)
    after = candidate_sha256(result)
    if before != after:
        raise HeadlessAdjudicationError("重置旧裁决时原始 OCR candidate 哈希发生变化。")
    return result, {
        "previous_authority_count": previous_authority_count,
        "reset_rows": reset_rows,
        "candidate_sha256": after,
    }


def _from_project(project: Path) -> ResolvedOcrSource:
    from core.project_workspace import ProjectWorkspaceManager
    from engine.multi_ocr_compare import MultiOcrComparison, MultiOcrRow
    from models.document import UnifiedDocument

    manager = ProjectWorkspaceManager(project.parent)
    manager.open_project(project)
    snapshot = manager.load_multi_ocr_snapshot()
    if not snapshot or snapshot.get("mode") != "multi":
        raise HeadlessAdjudicationError("工作区没有可用的多模型 OCR snapshot；请先在 GUI 完成 OCR。")
    documents = [
        UnifiedDocument.from_dict(item)
        for item in (snapshot.get("documents") or [])
        if isinstance(item, dict)
    ]
    labels = [str(value) for value in (snapshot.get("labels") or [])]
    raw_comparison = dict(snapshot.get("comparison") or {})
    raw_rows = list(raw_comparison.pop("rows", []) or [])
    rows = [MultiOcrRow(**dict(item)) for item in raw_rows if isinstance(item, dict)]
    comparison = MultiOcrComparison(rows=rows, **raw_comparison) if rows else None
    ruby_payload = snapshot.get("ruby_overlay_doc")
    ruby_doc = UnifiedDocument.from_dict(ruby_payload) if isinstance(ruby_payload, dict) else None
    if len(documents) < 2 or comparison is None:
        raise HeadlessAdjudicationError("工作区缺少完整的多模型 OCR 文档或对齐结果。")
    # Intentionally DO NOT consume fusion_states / local/cloud adjudication state.
    # Re-exporting from raw OCR documents + comparison creates a fresh baseline.
    package = export_multi_package(
        documents,
        labels,
        comparison,
        ruby_overlay_source=ruby_doc,
    )
    prior = snapshot.get("canonical_source_decisions") or []
    return ResolvedOcrSource(
        package=package,
        kind="workspace_project",
        source_path=str(project),
        previous_authority_count=len(prior),
        reset_rows=len(package.get("editable_items") or []),
    )


def _from_json(path: Path) -> ResolvedOcrSource:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    package, report = reset_exported_package_for_fresh_adjudication(value)
    return ResolvedOcrSource(
        package=package,
        kind="roundtrip_json",
        source_path=str(path),
        previous_authority_count=report["previous_authority_count"],
        reset_rows=report["reset_rows"],
    )


def _from_bundle(path: Path, work_dir: Path) -> ResolvedOcrSource:
    work_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "r") as archive:
        names = set(archive.namelist())
        member = "01_multi_ocr_fusion_result.json"
        if member not in names:
            raise HeadlessAdjudicationError("ZIP 不是 Novel Formatter 融合/骨架包：缺少 01_multi_ocr_fusion_result.json。")
        package_raw = json.loads(archive.read(member).decode("utf-8-sig"))
        skeleton_template = ""
        skeleton_member = "framework/structure_skeleton.epub"
        if skeleton_member in names:
            target = work_dir / "source_structure_skeleton.epub"
            target.write_bytes(archive.read(skeleton_member))
            skeleton_template = str(target)
    package, report = reset_exported_package_for_fresh_adjudication(package_raw)
    return ResolvedOcrSource(
        package=package,
        kind="fusion_skeleton_bundle",
        source_path=str(path),
        skeleton_template=skeleton_template,
        previous_authority_count=report["previous_authority_count"],
        reset_rows=report["reset_rows"],
    )


def resolve_fresh_ocr_source(source: str | Path, work_dir: str | Path) -> ResolvedOcrSource:
    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(str(path))
    work = Path(work_dir)
    if path.is_dir():
        if not (path / "project.json").is_file():
            raise HeadlessAdjudicationError("目录不是 Novel Formatter 项目：缺少 project.json。")
        return _from_project(path)
    if path.suffix.lower() == ".json":
        return _from_json(path)
    if path.suffix.lower() == ".zip":
        return _from_bundle(path, work)
    raise HeadlessAdjudicationError("CLI 裁决只接受已有 OCR 工作区目录、round-trip JSON 或融合/骨架 ZIP。")


__all__ = [
    "ResolvedOcrSource",
    "resolve_fresh_ocr_source",
    "reset_exported_package_for_fresh_adjudication",
    "materialize_skeleton_template_assets",
]



def materialize_skeleton_template_assets(
    package: dict,
    skeleton_epub: str | Path,
    output_dir: str | Path,
):
    """Build a layout-identical current document whose publication assets are local.

    Fusion/skeleton bundles can be moved away from the Mac that produced them.
    Their round-trip JSON deliberately retains original absolute image paths for
    auditability, while the bundled skeleton EPUB contains byte-preserved cover /
    illustration assets.  For headless re-adjudication we rebind *only* those
    asset paths in a detached ``UnifiedDocument``; row IDs, OCR candidates,
    geometry and package hashes remain untouched.
    """
    from models.document import UnifiedDocument
    from engine.ocr_roundtrip_package import layout_hash

    source = Path(skeleton_epub).expanduser().resolve()
    if not source.is_file():
        raise HeadlessAdjudicationError(f"骨架 EPUB 不存在：{source}")
    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    primary = UnifiedDocument.from_dict(copy.deepcopy(package.get("structure_document") or {}))
    image_blocks = [block for block in primary.blocks if getattr(getattr(block, "type", None), "value", "") == "image_ref"]
    if not image_blocks:
        return primary

    with zipfile.ZipFile(source, "r") as archive:
        names = set(archive.namelist())
        try:
            container = ET.fromstring(archive.read("META-INF/container.xml"))
            rootfile = next(
                elem for elem in container.iter()
                if elem.tag.rsplit("}", 1)[-1] == "rootfile"
            )
            opf_path = str(rootfile.attrib.get("full-path", "") or "")
            if not opf_path or opf_path not in names:
                raise ValueError("container.xml 未指向有效 OPF")
            opf = ET.fromstring(archive.read(opf_path))
        except Exception as exc:
            raise HeadlessAdjudicationError(f"无法读取骨架 EPUB manifest/spine：{exc}") from exc

        manifest = {}
        spine = []
        for elem in opf.iter():
            local = elem.tag.rsplit("}", 1)[-1]
            if local == "item" and elem.attrib.get("id") and elem.attrib.get("href"):
                manifest[str(elem.attrib["id"])] = str(elem.attrib["href"])
            elif local == "itemref" and elem.attrib.get("idref"):
                spine.append(str(elem.attrib["idref"]))
        opf_dir = posixpath.dirname(opf_path)
        ordered_images: list[str] = []
        seen: set[str] = set()
        img_re = re.compile(r"<img\b[^>]*?\bsrc=[\"']([^\"']+)[\"']", re.I)
        for idref in spine:
            href = manifest.get(idref, "")
            if not href.lower().endswith((".xhtml", ".html", ".htm")):
                continue
            xhtml_path = posixpath.normpath(posixpath.join(opf_dir, href))
            if xhtml_path not in names:
                continue
            text = archive.read(xhtml_path).decode("utf-8", errors="replace")
            for raw_src in img_re.findall(text):
                clean_src = raw_src.split("#", 1)[0].split("?", 1)[0]
                member = posixpath.normpath(posixpath.join(posixpath.dirname(xhtml_path), clean_src))
                if member in names and member not in seen:
                    seen.add(member)
                    ordered_images.append(member)
        if len(ordered_images) != len(image_blocks):
            raise HeadlessAdjudicationError(
                "骨架 EPUB 图片数量与 round-trip image_ref 不一致，拒绝猜测绑定："
                f"EPUB={len(ordered_images)} / image_ref={len(image_blocks)}。"
            )

        page_by_no = {int(getattr(page, "page_no", 0) or 0): page for page in primary.pages}
        for index, (block, member) in enumerate(zip(image_blocks, ordered_images)):
            suffix = Path(member).suffix.lower() or ".bin"
            stem = "cover" if index == 0 else f"illustration_{index:03d}"
            target = target_dir / f"{stem}{suffix}"
            target.write_bytes(archive.read(member))
            block.image_path = str(target)
            page = page_by_no.get(int(getattr(block, "page", 0) or 0))
            if page is not None:
                page.image_path = str(target)

    expected_layout = str(package.get("layout_sha256", "") or "")
    if expected_layout and layout_hash(primary) != expected_layout:
        raise HeadlessAdjudicationError("骨架资产重绑定意外改变了 OCR 物理布局，已中止。")
    return primary
