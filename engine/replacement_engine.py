#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replacement Engine —— 用 alignment.py 算出的对齐结果，把 OCR 文档里的正文文字
替换成高质量来源文本，产出一份新的 UnifiedDocument。

职责边界（刻意收得很窄）：
    只替换 Block.text（把原文字备份进 Block.ocr_raw），不碰其它任何东西——
    IMAGE_REF / 页面信息 / reading_order / TOC / metadata 全部原样保留，
    页面结构永远来自 OCR 那一份文档。
    低置信度的匹配（相似度低于阈值）不执行替换，只计入报告——宁可保留
    OCR 原文，也不用一个不确定的匹配去覆盖它。
    来源里有、OCR 完全没识别到的段落，不会自动插入新 Block——插入意味着
    要凭空决定它属于哪一页、什么位置、什么锚点，这些信息来源文本给不了，
    交给用户在报告里看到"漏了什么"，自己决定要不要处理。

用法（命令行）：
    python -m engine.replacement_engine ocr.json source.docx output.json
    python -m engine.replacement_engine ocr.json source.epub output.json --threshold 0.4 --report report.json
    python -m engine.replacement_engine ocr.json source.txt output.json --force  # 强制替换所有匹配
"""

from __future__ import annotations

import copy
import sys
import time
import re
try:
    import diff_match_patch as dmp_module
except ImportError:
    dmp_module = None
from engine.ngram_index import NGramIndex
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.document import UnifiedDocument, BlockType
from models.paragraph import Paragraph
from engine.alignment_v2 import align
from engine.japanese_normalizer import compare_key






# ── Safe diff guard: prevent destructive long text loss ─────────────────────
SAFE_DIFF_CONFIG = {
    "protect_long_delete": True,
    "delete_limit": 30,
    "protect_japanese_sentence": True,
    "anchor_guard": True,
}

def _has_japanese_sentence_boundary(text: str) -> bool:
    return any(mark in text for mark in ("。", "！", "？", "!", "?"))

def _safe_replace_result(old: str, new: str) -> tuple[str, bool]:
    """防止 diff/patch 阶段把完整日文长句误判为删除。"""
    if not old or not new:
        return new, False
    import difflib
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new).get_opcodes():
        if tag == "delete":
            deleted = old[i1:i2]
            if len(deleted) > SAFE_DIFF_CONFIG["delete_limit"]:
                return old, True
            if SAFE_DIFF_CONFIG["protect_japanese_sentence"] and _has_japanese_sentence_boundary(deleted):
                return old, True
    if len(new) < len(old) * 0.85 and len(old) > 80:
        return old, True
    return new, False

class DiffMatchPatchReplacer:
    """字符级替换器：用高质量来源覆盖 OCR，同时保留失败回退。"""
    def __init__(self, threshold=0.6):
        self.threshold = threshold
        self.dmp = dmp_module.diff_match_patch() if dmp_module else None
        if self.dmp:
            self.dmp.Diff_Timeout = 1.0

    def similarity(self, a, b):
        a = re.sub(r"\s+", "", a or "")
        b = re.sub(r"\s+", "", b or "")
        if not a and not b:
            return 1.0
        if self.dmp:
            diffs = self.dmp.diff_main(a, b)
            common = sum(len(x) for op, x in diffs if op == 0)
            return max(common / max(len(a), len(b), 1), self._normalized_similarity(a, b))
        import difflib
        return difflib.SequenceMatcher(None, a, b).ratio()

    def _normalized_similarity(self, a, b):
        import difflib
        return difflib.SequenceMatcher(None, compare_key(a), compare_key(b)).ratio()

    def replace(self, old, new):
        if self.dmp:
            patches = self.dmp.patch_make(old, new)
            result, ok = self.dmp.patch_apply(patches, old)
            if all(ok):
                return result
        return new

# OCR 常见日文小说误识别修正。
# 只作用于高质量来源文本写入 OCR 文档之前，不改变 alignment 流程。
OCR_CORRECTIONS = {
    "ブロローグ": "プロローグ",
    "魘王": "魔王",
    "鷹王": "魔王",
    "驚王": "魔王",
    "魔驚王": "魔王",
    "成信": "威信",
    "安者": "安堵",
    "微態": "微塵",
    "僧しみ": "憎しみ",
    "隊き": "呟き",
    "旨流": "主流",
    "麗王": "魔王",
    "勇男者": "勇者",
}


def apply_ocr_corrections(text: str) -> str:
    if not text:
        return text
    for wrong, right in OCR_CORRECTIONS.items():
        text = text.replace(wrong, right)
    return text

_TEXT_TYPES = {BlockType.PARAGRAPH, BlockType.DIALOGUE, BlockType.CHAPTER, BlockType.SECTION, BlockType.RUBY}


def _ocr_paragraphs_from_doc(doc: UnifiedDocument) -> list[Paragraph]:
    """从 UnifiedDocument 里抽出 Paragraph 列表，index 用的是 doc.blocks 里的
    真实下标（不是第几个段落）——replacement 阶段要直接按这个下标写回
    doc.blocks[index].text，不能用顺序计数，两者在有非文字块（图片等）
    穿插时并不相等。"""
    paragraphs: list[Paragraph] = []
    current_chapter = ""
    for i, b in enumerate(doc.blocks):
        if b.type not in _TEXT_TYPES:
            continue
        text = b.text.strip()
        if not text:
            continue
        is_title = b.type in (BlockType.CHAPTER, BlockType.SECTION)
        if b.type == BlockType.CHAPTER:
            current_chapter = text
        paragraphs.append(Paragraph(text=text, index=i, chapter=current_chapter, source="ocr", is_title=is_title))
    return paragraphs


@dataclass
class ReplacementReport:
    ocr_paragraph_count: int = 0
    source_paragraph_count: int = 0
    matched: int = 0
    replaced: int = 0             # matched 里相似度达标、实际执行了替换的段数
    low_confidence: int = 0       # matched 但相似度低于阈值，跳过没替换
    skipped_ocr: int = 0          # OCR 有、来源没有对应内容——原文原样保留
    skipped_source: int = 0       # 来源有、OCR 没有对应内容——不会自动插入
    avg_similarity: float = 0.0
    unmatched_source_preview: list[str] = field(default_factory=list)
    matched_pairs_preview: list[dict] = field(default_factory=list)  # 前10对匹配示例
    execution_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "ocr_paragraph_count": self.ocr_paragraph_count,
            "source_paragraph_count": self.source_paragraph_count,
            "matched": self.matched,
            "replaced": self.replaced,
            "low_confidence": self.low_confidence,
            "skipped_ocr": self.skipped_ocr,
            "skipped_source": self.skipped_source,
            "avg_similarity": round(self.avg_similarity, 4),
            "unmatched_source_preview": self.unmatched_source_preview[:20],
            "matched_pairs_preview": self.matched_pairs_preview[:10],
            "execution_seconds": round(self.execution_seconds, 2),
        }


def replace_text(
    ocr_doc: UnifiedDocument,
    source_paragraphs: list[Paragraph],
    match_threshold: float = 0.3,
    force_replace: bool = False,
) -> tuple[UnifiedDocument, ReplacementReport]:
    t0 = time.time()

    ocr_paragraphs = _ocr_paragraphs_from_doc(ocr_doc)
    result = align(ocr_paragraphs, source_paragraphs, match_threshold=match_threshold)

    new_doc = copy.deepcopy(ocr_doc)
    replaced = 0
    low_confidence = 0
    replacer = DiffMatchPatchReplacer(match_threshold)
    unmatched_source_preview: list[str] = []
    matched_pairs_preview: list[dict] = []

    for pair in result.pairs:
        if pair.ocr_index is not None and pair.source_index is not None:
            # 收集匹配示例（前10个）
            if len(matched_pairs_preview) < 10:
                matched_pairs_preview.append({
                    "ocr_text": ocr_paragraphs[pair.ocr_index].text[:60],
                    "source_text": source_paragraphs[pair.source_index].text[:60],
                    "similarity": round(pair.similarity, 4),
                })

            if pair.similarity >= match_threshold or force_replace:
                # pair.ocr_index 是 ocr_paragraphs 列表里的位置，不是 doc.blocks
                # 的下标——两者只在文档里完全没有非文字块（图片等）穿插时才
                # 恰好相等，一旦有 IMAGE_REF 混在正文里就会错位，必须经过
                # ocr_paragraphs[...].index 转换回真正的 doc.blocks 下标。
                block_index = ocr_paragraphs[pair.ocr_index].index
                block = new_doc.blocks[block_index]
                src_text = apply_ocr_corrections(source_paragraphs[pair.source_index].text)
                if block.text != src_text:
                    if not block.ocr_raw:
                        block.ocr_raw = block.text
                    candidate = replacer.replace(block.text, src_text)
                    candidate, protected = _safe_replace_result(block.text, candidate)
                    if protected:
                        new_doc.add_log("safe_diff_guard", "检测到疑似长句误删除，保留 OCR 原文", 1)
                    block.text = candidate
                    block.modified_by = "text_replacement"
                    block.confidence = pair.similarity
                    replaced += 1
            else:
                low_confidence += 1
        elif pair.source_index is not None:
            if len(unmatched_source_preview) < 50:
                unmatched_source_preview.append(source_paragraphs[pair.source_index].text[:40])

    new_doc.add_log(
        "text_replacement",
        f"替换 {replaced} 段（低置信度跳过 {low_confidence} 段），"
        f"{result.skipped_source} 段来源内容未在 OCR 中找到对应位置（未自动插入）",
        replaced,
    )

    report = ReplacementReport(
        ocr_paragraph_count=len(ocr_paragraphs),
        source_paragraph_count=len(source_paragraphs),
        matched=result.matched,
        replaced=replaced,
        low_confidence=low_confidence,
        skipped_ocr=result.skipped_ocr,
        skipped_source=result.skipped_source,
        avg_similarity=result.avg_similarity,
        unmatched_source_preview=unmatched_source_preview,
        matched_pairs_preview=matched_pairs_preview,
        execution_seconds=time.time() - t0,
    )
    return new_doc, report


if __name__ == "__main__":
    import argparse
    import json as jsonlib

    parser = argparse.ArgumentParser(description="Text Replacement Engine —— 用高质量来源文本替换 OCR 正文")
    parser.add_argument("ocr_json", help="OCR 产出的 UnifiedDocument JSON")
    parser.add_argument("source", help="高质量来源文件（docx/epub/json/txt/md/html）")
    parser.add_argument("output_json", help="替换后的 UnifiedDocument JSON 输出路径")
    parser.add_argument("--threshold", type=float, default=0.3, help="匹配相似度阈值（默认 0.3）")
    parser.add_argument("--force", action="store_true", help="强制替换所有匹配对，忽略阈值")
    parser.add_argument("--report", help="替换报告 JSON 输出路径")
    args = parser.parse_args()

    from adapters.text_extractors import extract_paragraphs

    with open(args.ocr_json, encoding="utf-8") as f:
        ocr_doc = UnifiedDocument.from_json(f.read())

    source_paragraphs = extract_paragraphs(args.source)
    print(f"📥  OCR: {len(ocr_doc.text_blocks())} 个文字块　来源: {len(source_paragraphs)} 段")

    new_doc, report = replace_text(
        ocr_doc, source_paragraphs,
        match_threshold=args.threshold,
        force_replace=args.force
    )

    print(
        f"✅  替换 {report.replaced} 段，低置信度跳过 {report.low_confidence} 段，"
        f"OCR 无对应来源 {report.skipped_ocr} 段，来源无对应 OCR {report.skipped_source} 段，"
        f"平均相似度 {report.avg_similarity:.2%}，耗时 {report.execution_seconds:.1f}s"
    )

    # 打印前几个匹配示例
    if report.matched_pairs_preview:
        print("匹配示例（前5对）：")
        for i, pair in enumerate(report.matched_pairs_preview[:5], 1):
            print(f"  {i}. 相似度 {pair['similarity']:.2f}: OCR: {pair['ocr_text']} ... -> 来源: {pair['source_text']} ...")

    with open(args.output_json, "w", encoding="utf-8") as f:
        f.write(new_doc.to_json())
    print(f"💾  已写入: {args.output_json}")

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            jsonlib.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"📊  报告已写入: {args.report}")