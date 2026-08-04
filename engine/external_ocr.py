#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Import and conservatively fuse an external OCR text version.

The built-in OCR remains authoritative for pages, chapters and image anchors.
External OCR contributes text only. No text is invented by this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import copy
import json
import re
import unicodedata
from typing import Iterable, Sequence

from models.document import Block, BlockType, UnifiedDocument
from engine.text_compare import (
    AlignedLine, CompareLine, align_lines, apply_compare_records, document_lines,
    inherit_right_structure_from_alignment, looks_like_chapter_title,
    normalise_for_alignment, parse_image_marker,
)

_JP = r"一-龯々〆ヵヶぁ-ゖァ-ヺー"
_DIGIT_NOISE = re.compile(rf"(?<=[{_JP}])[0-9](?=[{_JP}])")
_ASCII_NOISE = re.compile(rf"(?<=[{_JP}])[A-Za-z|](?=[{_JP}])")
_REPLACEMENT = re.compile(r"[�□■◼]|\x00")
_DECORATION = re.compile(r"^[#=_*\-—―\s]+$")
_MD_IMAGE = re.compile(r"^!\[[^\]]*\]\([^)]*\)\s*$", re.I)


@dataclass(slots=True)
class OcrLineQuality:
    score: float
    warnings: tuple[str, ...] = ()


@dataclass(slots=True)
class ExternalOcrDecision:
    index: int
    choice: str
    internal_text: str = ""
    external_text: str = ""
    output_text: str = ""
    internal_quality: float = 0.0
    external_quality: float = 0.0
    similarity: float = 0.0
    reference_internal: float = 0.0
    reference_external: float = 0.0
    confidence: float = 0.0
    reason: str = ""
    warnings: tuple[str, ...] = ()
    page: int = 0


@dataclass(slots=True)
class ExternalOcrFusionReport:
    internal_label: str = "内置 OCR"
    external_label: str = "外部 OCR"
    policy: str = "balanced"
    aligned_rows: int = 0
    exact_rows: int = 0
    chose_internal: int = 0
    chose_external: int = 0
    kept_both: int = 0
    image_rows: int = 0
    external_insertions: int = 0
    internal_omissions_kept: int = 0
    merged_external_rows: int = 0
    merged_internal_rows: int = 0
    reference_supported_external: int = 0
    reference_supported_internal: int = 0
    low_confidence: int = 0
    changed_blocks: int = 0
    decisions: list[ExternalOcrDecision] = field(default_factory=list)

    @property
    def summary(self) -> str:
        return (
            f"对齐 {self.aligned_rows} 行；完全一致 {self.exact_rows}；"
            f"采用内置 {self.chose_internal}，采用外部 {self.chose_external}，"
            f"保留双方独有内容 {self.kept_both}；低置信 {self.low_confidence}。"
        )


def line_quality(text: str) -> OcrLineQuality:
    value = unicodedata.normalize("NFKC", text or "").strip()
    if not value:
        return OcrLineQuality(0.0, ("空文本",))
    warnings: list[str] = []
    penalty = 0.0
    if _REPLACEMENT.search(value):
        penalty += 0.38; warnings.append("替代字符/黑块")
    n = len(_DIGIT_NOISE.findall(value))
    if n:
        penalty += min(0.34, 0.14 * n); warnings.append("数字混入日文词")
    n = len(_ASCII_NOISE.findall(value))
    if n:
        penalty += min(0.28, 0.10 * n); warnings.append("ASCII字符混入日文词")
    if "1°" in value or re.search(r"[ァ-ヺ]°", value):
        penalty += 0.20; warnings.append("异常角度符号")
    if value.count("「") != value.count("」"):
        penalty += 0.13; warnings.append("会话引号不平衡")
    if value.count("『") != value.count("』"):
        penalty += 0.10; warnings.append("引用引号不平衡")
    if value.startswith(("。", "、", "，", "」", "』")) and len(value) <= 8:
        penalty += 0.18; warnings.append("疑似句首残片")
    score = max(0.0, min(1.0, 0.94 + min(0.06, len(value) / 1000.0) - penalty))
    return OcrLineQuality(score, tuple(warnings))


def _block_type(text: str, title: bool = False) -> BlockType:
    s = (text or "").strip()
    if title or looks_like_chapter_title(s):
        return BlockType.CHAPTER
    if s.startswith("「") and s.endswith("」"):
        return BlockType.DIALOGUE
    return BlockType.PARAGRAPH


def _doc_from_lines(lines: Iterable[str], source: str) -> UnifiedDocument:
    doc = UnifiedDocument()
    doc.metadata.source_engine = f"external_ocr:{source}"
    doc.metadata.__dict__["external_ocr"] = True
    doc.metadata.__dict__["external_ocr_source"] = source
    for raw in lines:
        s = str(raw).replace("\r", "").rstrip("\n").strip()
        if not s or _DECORATION.fullmatch(s) or _MD_IMAGE.fullmatch(s):
            continue
        m = re.match(r"^#{1,6}\s*(.+?)\s*#*$", s)
        title = bool(m)
        if m:
            s = m.group(1).strip()
        block = Block(type=_block_type(s, title), text=s, ocr_raw=s, source_format="external_ocr")
        block.metadata["external_ocr_source"] = source
        doc.blocks.append(block)
    for i, block in enumerate(doc.blocks):
        block.reading_order = i
    return doc


def import_external_ocr(path: str) -> UnifiedDocument:
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    ext = src.suffix.lower()
    if ext == ".json":
        raw = src.read_text(encoding="utf-8-sig")
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and "blocks" in data:
                doc = UnifiedDocument.from_dict(data)
                doc.blocks = [copy.deepcopy(b) for b in doc.blocks if b.type != BlockType.IMAGE_REF]
                doc.pages = []
                doc.metadata.source_engine = f"external_ocr:{src.name}"
                doc.metadata.__dict__["external_ocr"] = True
                doc.metadata.__dict__["external_ocr_source"] = src.name
                return doc
        except Exception:
            pass
    if ext in {".txt", ".md", ".markdown"}:
        text = src.read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
        return _doc_from_lines(text.split("\n"), src.name)
    from adapters.text_extractors import extract_paragraphs
    paragraphs = extract_paragraphs(str(src))
    doc = UnifiedDocument()
    doc.metadata.source_engine = f"external_ocr:{src.name}"
    doc.metadata.__dict__["external_ocr"] = True
    doc.metadata.__dict__["external_ocr_source"] = src.name
    for para in paragraphs:
        s = (para.text or "").strip()
        if s:
            doc.blocks.append(Block(type=_block_type(s, bool(getattr(para, "is_title", False))), text=s, ocr_raw=s, source_format="external_ocr"))
    return doc


def _reference_scores(corpus, a: str, b: str) -> tuple[float, float]:
    if corpus is None:
        return 0.0, 0.0
    try:
        return float(corpus.search(a).score), float(corpus.search(b).score)
    except Exception:
        return 0.0, 0.0


def _similarity(a: str, b: str) -> float:
    from engine.text_compare import line_similarity
    return line_similarity(a, b)


def _choose(a: CompareLine, b: CompareLine, corpus=None):
    qa, qb = line_quality(a.text), line_quality(b.text)
    sim = _similarity(a.text, b.text)
    if normalise_for_alignment(a.text) == normalise_for_alignment(b.text):
        return "internal", 1.0, "两份 OCR 内容一致，保留结构底稿", (), 0.0, 0.0, qa, qb, sim
    ra, rb = _reference_scores(corpus, a.text, b.text)
    if max(ra, rb) >= 0.90 and abs(ra - rb) >= 0.075:
        if rb > ra:
            return "external", min(0.99, .78 + rb-ra), "参考原文明显支持外部 OCR", qb.warnings, ra, rb, qa, qb, sim
        return "internal", min(0.99, .78 + ra-rb), "参考原文明显支持内置 OCR", qa.warnings, ra, rb, qa, qb, sim
    delta = qb.score - qa.score
    if sim >= .55 and delta >= .08:
        return "external", min(.96, .70+delta), "两侧内容接近，外部 OCR 噪声更少", qb.warnings, ra, rb, qa, qb, sim
    if sim >= .55 and delta <= -.08:
        return "internal", min(.96, .70-delta), "两侧内容接近，内置 OCR 噪声更少", qa.warnings, ra, rb, qa, qb, sim
    na, nb = normalise_for_alignment(a.text), normalise_for_alignment(b.text)
    if na and na in nb and len(nb) <= max(len(na)+120, int(len(na)*2.4)) and qb.score >= qa.score-.02:
        return "external", .78, "外部 OCR 完整包含内置文本，疑似修复断句/漏字", qb.warnings, ra, rb, qa, qb, sim
    if nb and nb in na and len(na) <= max(len(nb)+120, int(len(nb)*2.4)) and qa.score >= qb.score-.02:
        return "internal", .78, "内置 OCR 完整包含外部文本，外部可能漏字", qa.warnings, ra, rb, qa, qb, sim
    if sim < .45:
        warnings = tuple(sorted(set(qa.warnings + qb.warnings + ("两侧差异过大",))))
        return "internal", .42, "两份 OCR 差异过大，保留结构底稿并标记复核", warnings, ra, rb, qa, qb, sim
    if delta > .14:
        return "external", .66, "外部 OCR 质量分明显更高", qb.warnings, ra, rb, qa, qb, sim
    return "internal", .58, "证据不足，保守保留内置 OCR", tuple(sorted(set(qa.warnings+qb.warnings))), ra, rb, qa, qb, sim


def _apply_policy(choice: str, confidence: float, reason: str, *, policy: str,
                  qa: OcrLineQuality, qb: OcrLineQuality, ra: float, rb: float,
                  similarity: float) -> tuple[str, float, str]:
    """Apply the user's preferred text authority without weakening structure safety."""
    if similarity >= .999999:
        return choice, confidence, reason
    if policy == "external_primary":
        # A strong reference match for the built-in text can still veto external
        # primary mode because the external OCR may have hallucinated/reordered.
        if ra >= .92 and ra >= rb + .10:
            return "internal", max(confidence, .84), "参考原文明显支持内置 OCR，否决外部优先"
        severe_external = any(w in qb.warnings for w in ("替代字符/黑块", "数字混入日文词", "ASCII字符混入日文词"))
        if similarity >= .30 and qb.score >= .50 and not (severe_external and qa.score >= qb.score + .08):
            return "external", max(confidence, .70), "已选择外部 OCR 作为文字主稿"
    elif policy == "internal_primary":
        if rb >= .92 and rb >= ra + .10 and qb.score >= .58:
            return "external", max(confidence, .84), "参考原文明显支持外部 OCR，覆盖内置优先"
        return "internal", max(confidence, .68), "已选择内置 OCR 作为文字主稿"
    return choice, confidence, reason


def _copy(line: CompareLine, text=None, ids=None, indices=None) -> CompareLine:
    return CompareLine(
        text=line.text if text is None else text,
        block_ids=list(line.block_ids if ids is None else ids),
        block_indices=list(line.block_indices if indices is None else indices),
        block_type=line.block_type,
        page=int(line.page or 0),
    )


def _covered_left(rows: Sequence[AlignedLine], i: int, external_text: str, max_extra: int = 3):
    """Return the longest consecutive internal sequence covered by one external row."""
    row = rows[i]
    if row.left is None:
        return row.left.text if row.left else "", [], [], 0
    target = normalise_for_alignment(external_text)
    texts, ids, indices = [row.left.text], list(row.left.block_ids), list(row.left.block_indices)
    best = (row.left.text, list(ids), list(indices), 0)
    for off in range(1, max_extra + 1):
        if i + off >= len(rows):
            break
        nxt = rows[i + off]
        if nxt.left is None or nxt.right is not None or parse_image_marker(nxt.left.text):
            break
        texts.append(nxt.left.text)
        ids += nxt.left.block_ids
        indices += nxt.left.block_indices
        candidate = "".join(texts)
        key = normalise_for_alignment(candidate)
        if key and key in target:
            best = (candidate, list(ids), list(indices), off)
            continue
        if target and target in key:
            break
    return best


def _covered_right(rows: Sequence[AlignedLine], i: int, internal_text: str, max_extra: int = 3):
    """Return the longest consecutive external sequence covered by one internal row."""
    row = rows[i]
    if row.right is None:
        return "", 0
    target = normalise_for_alignment(internal_text)
    texts = [row.right.text]
    best = (row.right.text, 0)
    for off in range(1, max_extra + 1):
        if i + off >= len(rows):
            break
        nxt = rows[i + off]
        if nxt.right is None or nxt.left is not None or parse_image_marker(nxt.right.text):
            break
        texts.append(nxt.right.text)
        candidate = "".join(texts)
        key = normalise_for_alignment(candidate)
        if key and key in target:
            best = (candidate, off)
            continue
        if target and target in key:
            break
    return best


def fuse_external_ocr(structure_doc: UnifiedDocument, external_doc: UnifiedDocument, *, reference_corpus=None,
                      internal_label="内置 OCR", external_label="外部 OCR", policy="balanced"):
    rows = inherit_right_structure_from_alignment(align_lines(document_lines(structure_doc), document_lines(external_doc)))
    policy = policy if policy in {"balanced", "external_primary", "internal_primary"} else "balanced"
    report = ExternalOcrFusionReport(internal_label=internal_label, external_label=external_label, policy=policy, aligned_rows=len(rows))
    records: list[CompareLine] = []
    i = 0; idx = 0
    while i < len(rows):
        row = rows[i]; left, right = row.left, row.right
        if left is not None and parse_image_marker(left.text):
            records.append(_copy(left)); report.image_rows += 1
            report.decisions.append(ExternalOcrDecision(idx, "image", left.text, right.text if right else "", left.text, confidence=1, reason="图片锚点由内置 OCR / 页面管理保留", page=int(left.page or 0)))
            idx += 1; i += 1; continue
        if left is not None and right is not None:
            combined, ids, indices, consume_left = _covered_left(rows, i, right.text)
            if consume_left:
                nk, rk = normalise_for_alignment(combined), normalise_for_alignment(right.text)
                if nk and nk in rk and len(rk) <= int(len(nk)*1.65)+12:
                    synthetic = _copy(left, text=combined, ids=ids, indices=indices)
                    if nk == rk:
                        choice, conf, reason, warnings, ra, rb, qa, qb, sim = (
                            "external", .96, "外部 OCR 将多个内置断行完整接回", (), 0.0, 0.0,
                            line_quality(combined), line_quality(right.text), 1.0,
                        )
                    else:
                        choice, conf, reason, warnings, ra, rb, qa, qb, sim = _choose(synthetic, right, reference_corpus)
                    choice, conf, reason = _apply_policy(
                        choice, conf, reason, policy=policy, qa=qa, qb=qb, ra=ra, rb=rb, similarity=sim,
                    )
                    if choice == "external":
                        records.append(_copy(right, ids=ids, indices=indices))
                        report.chose_external += 1; report.merged_external_rows += consume_left
                        if rb > ra+.075: report.reference_supported_external += 1
                        if conf < .60: report.low_confidence += 1
                        report.decisions.append(ExternalOcrDecision(idx, choice, combined, right.text, right.text, qa.score, qb.score, sim, ra, rb, conf, reason, warnings, int(left.page or 0)))
                        idx += 1; i += consume_left+1; continue
            combined_r, consume_right = _covered_right(rows, i, left.text)
            if consume_right:
                nk, rk = normalise_for_alignment(left.text), normalise_for_alignment(combined_r)
                if rk and rk in nk and len(nk) <= int(len(rk)*1.65)+12:
                    records.append(_copy(left)); report.chose_internal += 1; report.merged_internal_rows += consume_right
                    report.decisions.append(ExternalOcrDecision(idx, "internal", left.text, combined_r, left.text, line_quality(left.text).score, line_quality(combined_r).score, row.similarity, confidence=.82, reason="内置 OCR 已完整包含外部拆分行，避免重复插入", page=int(left.page or 0)))
                    idx += 1; i += consume_right+1; continue
            choice, conf, reason, warnings, ra, rb, qa, qb, sim = _choose(left, right, reference_corpus)
            choice, conf, reason = _apply_policy(
                choice, conf, reason, policy=policy, qa=qa, qb=qb, ra=ra, rb=rb, similarity=sim,
            )
            if normalise_for_alignment(left.text) == normalise_for_alignment(right.text): report.exact_rows += 1
            if choice == "external":
                records.append(_copy(right)); report.chose_external += 1; output = right.text
                if rb > ra+.075: report.reference_supported_external += 1
            else:
                records.append(_copy(left)); report.chose_internal += 1; output = left.text
                if ra > rb+.075: report.reference_supported_internal += 1
            if conf < .60: report.low_confidence += 1
            report.decisions.append(ExternalOcrDecision(idx, choice, left.text, right.text, output, qa.score, qb.score, sim, ra, rb, conf, reason, warnings, int(left.page or 0)))
            idx += 1; i += 1; continue
        if left is not None:
            records.append(_copy(left)); report.chose_internal += 1; report.internal_omissions_kept += 1
            report.decisions.append(ExternalOcrDecision(idx, "internal", left.text, "", left.text, line_quality(left.text).score, confidence=.72, reason="外部 OCR 缺少该段，保留内置 OCR", page=int(left.page or 0)))
            idx += 1; i += 1; continue
        if right is not None:
            q = line_quality(right.text)
            if right.text.strip() and q.score >= .48:
                records.append(_copy(right)); report.kept_both += 1; report.external_insertions += 1
                conf = .54 if q.score >= .75 else .42
                if conf < .60: report.low_confidence += 1
                report.decisions.append(ExternalOcrDecision(idx, "both", "", right.text, right.text, external_quality=q.score, confidence=conf, reason="外部 OCR 独有段落，可能是内置 OCR 漏识；保留并标记复核", warnings=q.warnings, page=int(right.page or 0)))
                idx += 1
            i += 1; continue
        i += 1
    fused, changed = apply_compare_records(structure_doc, records)
    fused.metadata.source_engine = "ocr_fusion"
    fused.metadata.__dict__.update({
        "ocr_fusion": True,
        "ocr_fusion_internal_label": internal_label,
        "ocr_fusion_external_label": external_label,
        "ocr_fusion_summary": report.summary,
        "ocr_fusion_low_confidence": report.low_confidence,
        "ocr_fusion_policy": policy,
    })
    report.changed_blocks = changed
    fused.add_log("external_ocr_fusion", report.summary, changed)
    return fused, report
