#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replacement Engine —— 用 alignment_v2.py 算出的对齐结果，把 OCR 文档里的正文文字
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
import difflib
import hashlib
import sys
import time
import re
import unicodedata
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
from engine.vertical_ocr_canonicalizer import VerticalOCRCanonicalizer, normalize_for_alignment
from engine.strict_reflow import (
    ReflowSegment, analyze_layout_texts, layout_neutral_text, reflow_source_paragraphs,
)




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
    "発王": "魔王",
    "燃王": "魔王",
    "勇男者": "勇者",
    "彼がにある": "彼方にある",
    "情戦": "情報",
    "曖味模糊": "曖昧模糊",
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


def _source_match_paragraphs(source_paragraphs: list[Paragraph]) -> list[Paragraph]:
    return [
        Paragraph(
            text=normalize_for_alignment(p.text),
            index=i,
            chapter=p.chapter,
            source=p.source,
            is_title=p.is_title,
        )
        for i, p in enumerate(source_paragraphs)
    ]


def _split_by_lengths(text: str, lengths: list[int]) -> list[str]:
    if not lengths:
        return []
    if len(lengths) == 1:
        return [text]
    total = max(sum(max(1, n) for n in lengths), 1)
    chunks: list[str] = []
    cursor = 0
    for index, length in enumerate(lengths):
        if index == len(lengths) - 1:
            chunks.append(text[cursor:])
            break
        end = round(len(text) * sum(max(1, n) for n in lengths[:index + 1]) / total)
        end = max(cursor, min(len(text), end))
        chunks.append(text[cursor:end])
        cursor = end
    return chunks


def _split_by_ocr_boundaries(source_text: str, original_texts: list[str]) -> list[str]:
    """按 OCR 原块边界切分来源文本，而不是简单按长度比例硬切。

    固定 OCR 排版要求保留物理块数量及段落边界。先把 OCR 各块拼接后与
    来源文本做字符级对齐，再将每个 OCR 边界投影到来源文本位置。这样
    「その輪」/「郭すら」之类跨块词不会因比例切分而丢字或错位。
    """
    if not original_texts:
        return []
    if len(original_texts) == 1:
        return [source_text]

    def compact(text: str) -> str:
        return re.sub(r"[\s　]+", "", apply_ocr_corrections(text or ""))

    source = compact(source_text)
    originals = [compact(text) for text in original_texts]
    joined = "".join(originals)
    if not source:
        return [""] * len(originals)
    if not joined:
        return _split_by_lengths(source_text, [1] * len(originals))

    boundaries = []
    total = 0
    for text in originals[:-1]:
        total += len(text)
        boundaries.append(total)

    matcher = difflib.SequenceMatcher(None, joined, source, autojunk=False)
    matching = [block for block in matcher.get_matching_blocks() if block.size]

    def project(boundary: int) -> int:
        # 边界落在公共匹配块中时可精确投影。
        for block in matching:
            if block.a <= boundary <= block.a + block.size:
                return block.b + (boundary - block.a)
        # 否则使用边界前后最近的匹配锚点插值；没有锚点才退回比例。
        left = None
        right = None
        for block in matching:
            if block.a + block.size <= boundary:
                left = (block.a + block.size, block.b + block.size)
            elif block.a >= boundary:
                right = (block.a, block.b)
                break
        if left and right and right[0] > left[0]:
            ratio = (boundary - left[0]) / (right[0] - left[0])
            return round(left[1] + ratio * (right[1] - left[1]))
        if left:
            return min(len(source), left[1] + (boundary - left[0]))
        if right:
            return max(0, right[1] - (right[0] - boundary))
        return round(len(source) * boundary / max(len(joined), 1))

    cuts = []
    last = 0
    for boundary in boundaries:
        cut = max(last, min(len(source), project(boundary)))
        cuts.append(cut)
        last = cut

    chunks = []
    start = 0
    for cut in cuts:
        chunks.append(source[start:cut])
        start = cut
    chunks.append(source[start:])
    return chunks


def _correct_unreplaced_ocr_blocks(doc: UnifiedDocument) -> int:
    """
    对文本替换未覆盖到的物理 OCR 块执行保守词典纠错。

    固定 OCR 排版时，来源段落可能只匹配到重复块中的一个；另一个未匹配块
    仍会保留 OCR 错字。这里不重建、不合并、不删除块，只修改 Block.text，
    因而不会破坏页码、坐标、图片锚点或 order_in_page。
    """
    changed = 0
    for block in doc.blocks:
        if block.type not in _TEXT_TYPES or not (block.text or ""):
            continue
        corrected = apply_ocr_corrections(block.text)
        if corrected == block.text:
            continue
        if not block.ocr_raw:
            block.ocr_raw = block.text
        block.text = corrected
        if block.modified_by != "text_replacement":
            block.modified_by = "post_replacement_ocr_correction"
        changed += 1
    if changed:
        doc.add_log("post_replacement_ocr_correction", f"替换后修正 {changed} 个未覆盖 OCR 块", changed)
    return changed




def _trim_compact_prefix(text: str, compact_chars: int) -> str:
    """从原文裁掉指定数量的非空白字符，并清理残留句末符号/空白。"""
    if compact_chars <= 0:
        return text
    consumed = 0
    cut = 0
    for i, ch in enumerate(text or ""):
        if not ch.isspace() and ch != "　":
            consumed += 1
        cut = i + 1
        if consumed >= compact_chars:
            break
    remainder = (text or "")[cut:]
    return remainder.lstrip(" \t\r\n　、，,。．")


def _covered_prefix_length(previous: str, current: str) -> int:
    """识别 current 开头被 previous 覆盖的长度；返回可裁剪的紧凑字符数。"""
    if len(previous) < 15 or len(current) < 20:
        return 0

    # 最可靠路径：当前段以完整上一段开头，随后还有新正文。
    if current.startswith(previous) and len(current) - len(previous) >= 6:
        return len(previous)

    matcher = difflib.SequenceMatcher(None, previous, current, autojunk=False)
    blocks = [b for b in matcher.get_matching_blocks() if b.size]
    if not blocks:
        return 0

    # 从 current 起点向后收集由 previous 覆盖的连续匹配区域；允许 OCR 在中间
    # 漏/多少量字符，但不允许跳过很长的新正文再继续匹配。
    prefix_end = 0
    covered = 0
    last_b_end = 0
    for b in blocks:
        if b.b > prefix_end + 8:
            break
        if b.a < last_b_end - 3:
            continue
        prefix_end = max(prefix_end, b.b + b.size)
        covered += b.size
        last_b_end = b.a + b.size

    if prefix_end < 20 or len(current) - prefix_end < 6:
        return 0
    if covered / max(prefix_end, 1) < 0.72:
        return 0
    if covered / max(min(len(previous), prefix_end), 1) < 0.65:
        return 0
    return prefix_end


def _post_replacement_prefix_dedup(doc: UnifiedDocument) -> int:
    """替换后删除被前文覆盖的残块，并裁掉后续段落的重复前缀。

    三类处理：
      1. 完整重复/短块被相邻长块包含：整块 consumed；
      2. 丢失句首字符、但结尾与上一段一致的后缀残块：整块 consumed；
      3. 当前段开头重述上一段、后面还有新正文：只裁掉重复前缀。
    固定 OCR 排版下不删除 Block，只置空并记录 consumed。
    """
    changed = 0
    previous_index: int | None = None

    def key(text: str) -> str:
        corrected = apply_ocr_corrections(text or "")
        normalized = re.sub(r"[\s　]+", "", normalize_for_alignment(corrected))
        # OCR frequently mixes corner brackets and long-dash glyphs across adjacent
        # vertical columns.  Comparison-only canonicalisation lets a tail such as
        # ``王』といった。`` match ``「魔王」といった。`` without altering the
        # visible replacement text.
        return normalized.translate(str.maketrans({
            "『": "「", "』": "」", "“": "「", "”": "」",
            "—": "ー", "―": "ー", "−": "ー", "ｰ": "ー",
        }))

    def consume(shorter_index: int, longer_index: int, reason: str) -> None:
        nonlocal changed
        shorter = doc.blocks[shorter_index]
        if not shorter.ocr_raw:
            shorter.ocr_raw = shorter.text
        shorter.text = ""
        shorter.modified_by = reason
        shorter.metadata = {
            **(shorter.metadata or {}),
            "consumed": True,
            "consumed_by": longer_index,
            "consumed_reason": reason,
        }
        changed += 1

    for index, block in enumerate(doc.blocks):
        if block.type not in _TEXT_TYPES or not (block.text or "").strip():
            continue
        if previous_index is None:
            previous_index = index
            continue

        previous = doc.blocks[previous_index]
        if abs(int(getattr(previous, "page", 0) or 0) - int(getattr(block, "page", 0) or 0)) > 1:
            previous_index = index
            continue

        left = key(previous.text)
        right = key(block.text)

        # 当前短块是上一段的后缀残片；允许少量句首丢字及结尾措辞变化。
        current_is_suffix = False
        if 4 <= len(right) < len(left):
            # 4～11 字的短尾列最容易被旧逻辑漏掉。只有在上一块确实由
            # 高质量替换文本写入、当前块仍是未替换 OCR 时，才允许用严格的
            # “完整后缀”规则消费，避免误删小说中故意重复的短句。
            strict_short_tail = (
                len(right) < 12
                and previous.modified_by == "text_replacement"
                and block.modified_by != "text_replacement"
                and left.endswith(right)
            )
            if strict_short_tail:
                current_is_suffix = True
            elif len(right) >= 12:
                if right in left:
                    current_is_suffix = True
                elif len(right) >= 20 and left[-20:] == right[-20:]:
                    current_is_suffix = True
                else:
                    tail = left[-max(len(right) + 12, 32):]
                    ratio = difflib.SequenceMatcher(None, right, tail, autojunk=False).ratio()
                    current_is_suffix = ratio >= 0.90
                    # “術』を極めた者である” → “術』を極めた者であったという事実である”
                    # 这类残块中间有改写，但开头锚定在上一段末尾且句尾一致。
                    if not current_is_suffix and len(right) <= 36:
                        head = right[:min(8, len(right))]
                        current_is_suffix = head in left[-48:] and right[-4:] == left[-4:]
        if current_is_suffix:
            consume(index, previous_index, "post_replacement_covered_fragment")
            continue

        # 上一短块被当前完整块覆盖。
        previous_is_fragment = False
        if 12 <= len(left) <= len(right):
            if left == right or left in right:
                previous_is_fragment = True
            elif len(left) >= 20 and left[-20:] == right[-20:]:
                previous_is_fragment = True
            elif len(left) >= 30:
                matcher = difflib.SequenceMatcher(None, left, right, autojunk=False)
                matches = [m for m in matcher.get_matching_blocks() if m.size]
                coverage = sum(m.size for m in matches) / max(len(left), 1)
                anchored = bool(matches and matches[0].a == 0 and matches[0].b == 0 and matches[0].size >= 20)
                previous_is_fragment = anchored and coverage >= 0.90
        if previous_is_fragment:
            consume(previous_index, index, "post_replacement_covered_fragment")
            previous_index = index
            continue

        # 当前段开头重复上一段，但后面还有真正的新内容：只裁掉前缀。
        trim_len = _covered_prefix_length(left, right)
        if trim_len:
            original = block.text
            trimmed = _trim_compact_prefix(original, trim_len)
            if trimmed and trimmed != original:
                if not block.ocr_raw:
                    block.ocr_raw = original
                block.text = trimmed
                block.modified_by = "post_replacement_overlap_trim"
                block.metadata = {
                    **(block.metadata or {}),
                    "trimmed_prefix_chars": trim_len,
                    "overlap_with": previous_index,
                }
                changed += 1
                right = key(trimmed)

        previous_index = index

    if changed:
        doc.add_log("post_replacement_prefix_dedup", f"替换后清理/裁剪 {changed} 个被覆盖残块", changed)
    return changed

def _post_replacement_long_run_dedup(
    doc: UnifiedDocument,
    source_paragraphs: list[Paragraph] | None,
) -> int:
    """Consume a nearby repeated multi-paragraph OCR run absent from the source.

    Page overlap can duplicate two or three scanned pages.  Short-fragment cleanup
    cannot see it because each paragraph is complete.  This detector requires a
    cluster of exact normalized paragraph anchors, a high whole-span similarity,
    nearby page numbers, and a unique anchor in the trusted replacement source.
    The conservative source-uniqueness gate protects intentional repeated dialogue.
    """
    if not source_paragraphs:
        return 0

    def key(text: str) -> str:
        return re.sub(r"[\s　]+", "", normalize_for_alignment(apply_ocr_corrections(text or "")))

    eligible: list[tuple[int, str, int]] = []
    for block_index, block in enumerate(doc.blocks):
        if block.type not in (BlockType.PARAGRAPH, BlockType.DIALOGUE, BlockType.RUBY):
            continue
        if (block.metadata or {}).get("consumed"):
            continue
        compact = key(block.text)
        if len(compact) >= 16:
            eligible.append((block_index, compact, int(getattr(block, "page", 0) or 0)))
    if len(eligible) < 6:
        return 0

    source_text = "".join(key(p.text) for p in source_paragraphs if (p.text or "").strip())
    seen: dict[str, list[int]] = {}
    pairs: list[tuple[int, int]] = []
    for pos, (_block_index, compact, page) in enumerate(eligible):
        for previous_pos in seen.get(compact, [])[-8:]:
            prev_block, _prev_text, prev_page = eligible[previous_pos]
            current_block = eligible[pos][0]
            if current_block - prev_block > 180:
                continue
            page_gap = page - prev_page
            if 1 <= page_gap <= 10:
                pairs.append((previous_pos, pos))
        seen.setdefault(compact, []).append(pos)
    if not pairs:
        return 0

    pairs.sort(key=lambda x: (x[1], x[0]))
    clusters: list[list[tuple[int, int]]] = []
    for pair in pairs:
        attached = False
        for cluster in reversed(clusters[-8:]):
            last_a, last_b = cluster[-1]
            a, b = pair
            if (
                a > last_a and b > last_b
                and a - last_a <= 6 and b - last_b <= 6
                and abs((b - a) - (last_b - last_a)) <= 3
            ):
                cluster.append(pair)
                attached = True
                break
        if not attached:
            clusters.append([pair])

    changed = 0
    consumed_ranges: list[tuple[int, int]] = []
    for cluster in clusters:
        if len(cluster) < 3:
            continue
        matched_chars = sum(len(eligible[b][1]) for _a, b in cluster)
        if matched_chars < 160:
            continue

        first_start = min(eligible[a][0] for a, _b in cluster)
        first_end = max(eligible[a][0] for a, _b in cluster)
        second_start = min(eligible[b][0] for _a, b in cluster)
        second_end = max(eligible[b][0] for _a, b in cluster)
        if second_start <= first_end:
            continue
        if any(not (second_end < start or second_start > end) for start, end in consumed_ranges):
            continue

        first_text = "".join(
            key(block.text) for block in doc.blocks[first_start:first_end + 1]
            if block.type in _TEXT_TYPES and not (block.metadata or {}).get("consumed")
        )
        second_text = "".join(
            key(block.text) for block in doc.blocks[second_start:second_end + 1]
            if block.type in _TEXT_TYPES and not (block.metadata or {}).get("consumed")
        )
        if min(len(first_text), len(second_text)) < 180:
            continue
        ratio = difflib.SequenceMatcher(None, first_text, second_text, autojunk=False).ratio()
        if ratio < 0.88:
            continue

        # Use the longest exact paragraph as a source anchor.  The source must
        # contain it exactly once; otherwise repetition may be intentional.
        anchor = max((eligible[b][1] for _a, b in cluster), key=len, default="")
        if len(anchor) < 24 or source_text.count(anchor) != 1:
            continue

        local_changed = 0
        for block_index in range(second_start, second_end + 1):
            block = doc.blocks[block_index]
            if block.type not in _TEXT_TYPES or not (block.text or "").strip():
                continue
            if (block.metadata or {}).get("consumed"):
                continue
            if not block.ocr_raw:
                block.ocr_raw = block.text
            block.text = ""
            block.modified_by = "post_replacement_long_duplicate"
            block.metadata = {
                **(block.metadata or {}),
                "consumed": True,
                "consumed_by_span": [first_start, first_end],
                "consumed_reason": "nearby_repeated_page_run_not_repeated_in_source",
                "duplicate_similarity": round(ratio, 4),
            }
            local_changed += 1
        if local_changed:
            changed += local_changed
            consumed_ranges.append((second_start, second_end))

    if changed:
        doc.add_log("post_replacement_long_dedup", f"替换后清理 {changed} 个跨页长重复块", changed)
    return changed


def cleanup_covered_replacement_fragments(
    doc: UnifiedDocument,
    source_paragraphs: list[Paragraph] | None = None,
) -> int:
    """Safely rerun cleanup for patch-mode documents only.

    Strict full replacement is already composed exclusively from the trusted source.
    Running OCR-fragment heuristics on it could delete an intentional repeated sentence,
    so strict documents are immutable to this cleanup stage.
    """
    if str(getattr(doc.metadata, "replacement_mode", "") or "") in {"strict_full", "strict_literal"}:
        return 0
    return _post_replacement_prefix_dedup(doc) + _post_replacement_long_run_dedup(doc, source_paragraphs)


def _replace_vertical_logical_text(
    ocr_doc: UnifiedDocument,
    source_paragraphs: list[Paragraph],
    match_threshold: float,
    force_replace: bool,
) -> tuple[UnifiedDocument, ReplacementReport]:
    t0 = time.time()
    canonicalizer = VerticalOCRCanonicalizer()
    match_doc = canonicalizer.build_logical_document(ocr_doc)
    logical = match_doc.logical_paragraphs
    logical_match = [
        Paragraph(text=p.match_text, index=i, source="vertical_ocr", is_title=p.is_title)
        for i, p in enumerate(logical)
    ]
    source_match = _source_match_paragraphs(source_paragraphs)
    result = align(logical_match, source_match, match_threshold=match_threshold)

    new_doc = copy.deepcopy(ocr_doc)
    preserve_layout = bool(getattr(new_doc.metadata, "preserve_ocr_layout", False))
    replaced = 0
    low_confidence = 0
    unmatched_source_preview: list[str] = []
    matched_pairs_preview: list[dict] = []
    consumed_blocks: set[int] = set()
    direct_deduped = 0

    for pair in result.pairs:
        if pair.ocr_index is not None and pair.source_index is not None:
            lp = logical[pair.ocr_index]
            source_para = source_paragraphs[pair.source_index]
            src_text = apply_ocr_corrections(source_para.text)

            if len(matched_pairs_preview) < 10:
                matched_pairs_preview.append({
                    "ocr_text": lp.display_text[:60],
                    "source_text": source_para.text[:60],
                    "similarity": round(pair.similarity, 4),
                })

            # 低阈值只用于“寻找候选对齐”，不能直接等同于“允许覆盖正文”。
            # 0.3 左右的弱匹配会把来源中完全不同的人称/段落写入 OCR 正文，
            # 造成 EPUB 出现逻辑跳跃。写回时采用独立的安全门槛，并检查长度比。
            ocr_text = lp.display_text
            source_text = source_para.text
            compact_ocr = re.sub(r"[\s　]+", "", normalize_for_alignment(ocr_text))
            compact_src = re.sub(r"[\s　]+", "", normalize_for_alignment(source_text))
            length_ratio = min(len(compact_ocr), len(compact_src)) / max(len(compact_ocr), len(compact_src), 1)
            safe_write_threshold = max(match_threshold, 0.52)
            safe_pair = pair.similarity >= safe_write_threshold and length_ratio >= 0.45

            if safe_pair or force_replace:
                refs = [ref.block_index for ref in lp.block_refs if ref.block_index not in consumed_blocks]
                if not refs:
                    continue

                if preserve_layout:
                    physical_keys = [normalize_for_alignment(new_doc.blocks[idx].text) for idx in refs]
                    duplicate_refs = (
                        len(refs) > 1
                        and physical_keys[0]
                        and all(key == physical_keys[0] for key in physical_keys[1:])
                    )
                    if duplicate_refs:
                        # 同一逻辑段由多个完全重复的物理 OCR 块组成时，不能按长度
                        # 把来源正文切成几份，否则会留下两个“半段”。保留第一个块的
                        # 完整文本，其余块只清空文字，坐标/页码/图片锚点均不变。
                        first_idx = refs[0]
                        first = new_doc.blocks[first_idx]
                        if not first.ocr_raw:
                            first.ocr_raw = first.text
                        first.text = src_text
                        first.modified_by = "text_replacement"
                        first.confidence = pair.similarity
                        for idx in refs[1:]:
                            block = new_doc.blocks[idx]
                            if not block.ocr_raw:
                                block.ocr_raw = block.text
                            block.text = ""
                            block.modified_by = "consumed_duplicate_by_text_replacement"
                            block.metadata = {**(block.metadata or {}), "consumed": True, "consumed_by": first_idx}
                            direct_deduped += 1
                    else:
                        original_texts = [new_doc.blocks[idx].text for idx in refs]
                        chunks = _split_by_ocr_boundaries(src_text, original_texts)
                        for idx, chunk in zip(refs, chunks):
                            block = new_doc.blocks[idx]
                            if block.text != chunk:
                                if not block.ocr_raw:
                                    block.ocr_raw = block.text
                                block.text = chunk
                            block.modified_by = "text_replacement"
                            block.confidence = pair.similarity
                else:
                    first = new_doc.blocks[refs[0]]
                    if not first.ocr_raw:
                        first.ocr_raw = first.text
                    first.text = src_text
                    first.modified_by = "text_replacement"
                    first.confidence = pair.similarity
                    first.metadata = {**(first.metadata or {}), "logical_replacement_span": refs}
                    for idx in refs[1:]:
                        block = new_doc.blocks[idx]
                        if not block.ocr_raw:
                            block.ocr_raw = block.text
                        block.text = ""
                        block.modified_by = "consumed_by_text_replacement"
                        block.metadata = {**(block.metadata or {}), "consumed": True, "consumed_by": refs[0]}

                consumed_blocks.update(refs)
                replaced += 1
            else:
                low_confidence += 1
        elif pair.source_index is not None and len(unmatched_source_preview) < 50:
            unmatched_source_preview.append(source_paragraphs[pair.source_index].text[:40])

    _correct_unreplaced_ocr_blocks(new_doc)
    post_deduped = direct_deduped + _post_replacement_prefix_dedup(new_doc) + _post_replacement_long_run_dedup(new_doc, source_paragraphs)

    new_doc.add_log(
        "text_replacement",
        f"竖排逻辑横排化后替换 {replaced} 段（低置信度跳过 {low_confidence} 段），"
        f"{result.skipped_source} 段来源内容未在 OCR 中找到对应位置（未自动插入）",
        replaced,
    )

    report = ReplacementReport(
        ocr_paragraph_count=len(logical),
        source_paragraph_count=len(source_paragraphs),
        matched=result.matched,
        replaced=replaced,
        low_confidence=low_confidence,
        skipped_ocr=result.skipped_ocr,
        skipped_source=result.skipped_source,
        avg_similarity=result.avg_similarity,
        unmatched_source_preview=unmatched_source_preview,
        matched_pairs_preview=matched_pairs_preview,
        post_deduped=post_deduped,
        execution_seconds=time.time() - t0,
    )
    return new_doc, report




def _canonical_body_text_from_paragraphs(paragraphs: list[Paragraph]) -> str:
    """Canonical structured-body stream used by strict replacement validation.

    Paragraph boundaries are significant and represented by a single ``\n``.  The
    only normalization is Unicode NFC plus platform newline normalization.  We do
    *not* ignore punctuation, ordinary spaces, or characters: strict mode must fail
    if even one visible source character is missing or added.
    """
    values: list[str] = []
    for paragraph in paragraphs:
        text = str(paragraph.text or "").replace("\r\n", "\n").replace("\r", "\n")
        values.append(unicodedata.normalize("NFC", text))
    return "\n".join(values)


def _canonical_body_text_from_doc(doc: UnifiedDocument) -> str:
    paragraphs = [
        Paragraph(text=block.text)
        for block in doc.blocks
        if block.type in _TEXT_TYPES
        and not (block.metadata or {}).get("consumed")
        and str(block.text or "") != ""
    ]
    return _canonical_body_text_from_paragraphs(paragraphs)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _diff_character_counts(left: str, right: str) -> tuple[int, int, float]:
    """Return chars only in left, chars only in right, and similarity ratio."""
    if left == right:
        return 0, 0, 1.0
    # Whole novels can contain hundreds of thousands of characters.  ``autojunk``
    # avoids quadratic behaviour in repeated punctuation/whitespace while this
    # report remains diagnostic only; strict acceptance uses exact hashes below.
    matcher = difflib.SequenceMatcher(None, left, right, autojunk=(max(len(left), len(right)) > 100_000))
    left_only = 0
    right_only = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("delete", "replace"):
            left_only += i2 - i1
        if tag in ("insert", "replace"):
            right_only += j2 - j1
    return left_only, right_only, matcher.ratio()




def _nearest_value(values: list[int], target: float) -> int:
    if not values:
        return max(0, round(target))
    return min(values, key=lambda value: abs(value - target))


def _map_source_pages(
    ocr_doc: UnifiedDocument,
    source_paragraphs: list[Paragraph],
) -> tuple[list[int], set[int]]:
    """Map authoritative source paragraphs to body pages with a fast n-gram pass.

    Strict replacement never needs a full dynamic-programming alignment because page
    numbers are only approximate illustration anchors.  A source n-gram index finds a
    few candidates for each sufficiently long OCR block; unmatched source paragraphs
    are then interpolated between reliable anchors.  This keeps whole-book replacement
    close to linear time even for several thousand paragraphs.
    """
    if not source_paragraphs:
        return [], set()

    source_texts = [normalize_for_alignment(p.text) for p in source_paragraphs]
    index = NGramIndex(source_texts, n=3)
    best_by_source: dict[int, tuple[float, int]] = {}

    for paragraph in _ocr_paragraphs_from_doc(ocr_doc):
        raw = normalize_for_alignment(paragraph.text)
        compact = re.sub(r"[\s　]+", "", raw)
        if len(compact) < 18:
            continue
        page = int(getattr(ocr_doc.blocks[paragraph.index], "page", 0) or 0)
        if page <= 0:
            continue
        for candidate in index.query(raw, top_k=4):
            source_compact = re.sub(r"[\s　]+", "", source_texts[candidate])
            if not source_compact:
                continue
            matcher = difflib.SequenceMatcher(None, compare_key(compact), compare_key(source_compact), autojunk=False)
            common = sum(block.size for block in matcher.get_matching_blocks() if block.size)
            full_ratio = matcher.ratio()
            containment = common / max(min(len(compact), len(source_compact)), 1)
            length_ratio = min(len(compact), len(source_compact)) / max(len(compact), len(source_compact), 1)
            score = max(full_ratio, containment * min(1.0, 0.55 + length_ratio))
            if containment < 0.68 or score < 0.58:
                continue
            previous = best_by_source.get(candidate)
            if previous is None or score > previous[0]:
                best_by_source[candidate] = (score, page)

    # Repeated prose can produce individually strong but globally impossible matches
    # (a later source paragraph matched to an earlier OCR page).  Illustration
    # placement requires a monotonic source-index -> page map, so keep the
    # highest-scoring non-decreasing anchor chain before interpolation.
    candidates = sorted(
        (source_index, page, score)
        for source_index, (score, page) in best_by_source.items()
    )
    if candidates:
        page_values = sorted({page for _idx, page, _score in candidates})
        page_rank = {page: rank + 1 for rank, page in enumerate(page_values)}
        size = len(page_values) + 2
        tree = [(0.0, -1)] * size
        predecessor = [-1] * len(candidates)
        chain_score = [0.0] * len(candidates)

        def query(pos):
            best = (0.0, -1)
            while pos > 0:
                if tree[pos][0] > best[0]:
                    best = tree[pos]
                pos -= pos & -pos
            return best

        def update(pos, value):
            while pos < size:
                if value[0] > tree[pos][0]:
                    tree[pos] = value
                pos += pos & -pos

        for candidate_index, (_source_index, page, score) in enumerate(candidates):
            previous_score, previous_index = query(page_rank[page])
            chain_score[candidate_index] = previous_score + max(float(score), 0.01)
            predecessor[candidate_index] = previous_index
            update(page_rank[page], (chain_score[candidate_index], candidate_index))

        end = max(range(len(candidates)), key=lambda i: chain_score[i])
        kept_indexes = set()
        while end >= 0:
            kept_indexes.add(end)
            end = predecessor[end]
        monotonic = [candidates[i] for i in sorted(kept_indexes)]
    else:
        monotonic = []

    mapped = {source_index: page for source_index, page, _score in monotonic}
    confident = {source_index for source_index, _page, score in monotonic if score >= 0.72}

    text_pages = sorted({
        int(p.page_no)
        for p in ocr_doc.pages
        if p.page_type == BlockType.PARAGRAPH and int(p.page_no) > 0
    })
    if not text_pages:
        text_pages = sorted({
            int(getattr(block, "page", 0) or 0)
            for block in ocr_doc.blocks
            if block.type in _TEXT_TYPES and int(getattr(block, "page", 0) or 0) > 0
        })
    if not text_pages:
        text_pages = [1]

    known = sorted(mapped)
    output: list[int] = []
    for source_index in range(len(source_paragraphs)):
        if source_index in mapped:
            output.append(_nearest_value(text_pages, mapped[source_index]))
            continue

        position = __import__('bisect').bisect_left(known, source_index)
        left = known[position - 1] if position > 0 else None
        right = known[position] if position < len(known) else None
        if left is not None and right is not None and right > left:
            ratio = (source_index - left) / (right - left)
            estimate = mapped[left] + ratio * (mapped[right] - mapped[left])
        elif left is not None:
            estimate = mapped[left]
        elif right is not None:
            estimate = mapped[right]
        else:
            ratio = source_index / max(len(source_paragraphs) - 1, 1)
            estimate = text_pages[0] + ratio * (text_pages[-1] - text_pages[0])
        output.append(_nearest_value(text_pages, estimate))
    return output, confident


def _source_block_type(paragraph: Paragraph) -> BlockType:
    if paragraph.is_title:
        return BlockType.CHAPTER
    stripped = str(paragraph.text or "").strip()
    if (
        len(stripped) >= 2
        and stripped[0] in "「『“\""
        and stripped[-1] in "」』”\""
    ):
        return BlockType.DIALOGUE
    return BlockType.PARAGRAPH


def _authoritative_source_blocks(
    source_paragraphs: list[Paragraph],
    page_map: list[int],
    confident_pages: set[int],
) -> tuple[list, list]:
    from models.document import Block, TocEntry

    blocks: list[Block] = []
    toc: list[TocEntry] = []
    chapter_index = 0
    for index, paragraph in enumerate(source_paragraphs):
        block_type = _source_block_type(paragraph)
        if block_type == BlockType.CHAPTER:
            chapter_index += 1
        block = Block(
            type=block_type,
            text=str(paragraph.text or ""),
            page=page_map[index] if index < len(page_map) else 0,
            reading_order=len(blocks),
            confidence=1.0,
            source_format=Path(paragraph.source).suffix.lstrip(".") if paragraph.source else "replacement",
            modified_by="strict_full_replacement",
            chapter_index=chapter_index,
            metadata={
                "authoritative_replacement": True,
                "source_paragraph_index": index,
                "source_page_alignment": "confident" if index in confident_pages else "interpolated",
            },
        )
        blocks.append(block)
        if block_type == BlockType.CHAPTER:
            toc.append(TocEntry(block.text, chapter_index, len(blocks) - 1))
    return blocks, toc


def _authoritative_reflow_blocks(
    segments: list[ReflowSegment],
) -> tuple[list, list]:
    """Build authoritative blocks from reflowed source segments."""
    from models.document import Block, TocEntry

    blocks: list[Block] = []
    toc: list[TocEntry] = []
    chapter_index = 0
    for segment_index, segment in enumerate(segments):
        block_type = segment.type
        if block_type == BlockType.CHAPTER:
            chapter_index += 1
        block = Block(
            type=block_type,
            text=str(segment.text or ""),
            page=int(segment.page or 0),
            reading_order=len(blocks),
            confidence=float(segment.confidence or 0.0),
            source_format="replacement",
            modified_by="strict_full_reflow",
            chapter_index=chapter_index,
            metadata={
                "authoritative_replacement": True,
                "source_paragraph_index": int(segment.source_index),
                "source_page_alignment": "ocr_boundary" if segment.reference_blocks else "interpolated",
                "layout_reflow_method": segment.method,
                "layout_reference_blocks": list(segment.reference_blocks),
                "layout_quote_insertions": int(segment.quote_insertions),
                "layout_quote_moves": int(segment.quote_moves),
            },
        )
        blocks.append(block)
        if block_type == BlockType.CHAPTER:
            toc.append(TocEntry(block.text, chapter_index, len(blocks) - 1))
    return blocks, toc


def _strict_image_blocks(ocr_doc: UnifiedDocument, text_blocks: list) -> tuple[list, int, int, int]:
    """Rebuild image references without copying any OCR text block."""
    from models.document import Block

    page_images: dict[tuple[int, str], tuple[BlockType, str, dict]] = {}
    for page in ocr_doc.pages:
        if not page.image_path or page.page_type in (BlockType.PARAGRAPH, BlockType.BLANK):
            continue
        page_images[(int(page.page_no), str(page.image_path))] = (
            page.page_type,
            str(page.image_path),
            {"source": "strict_full_replacement", "page_type": page.page_type.value},
        )
    for block in ocr_doc.blocks:
        if block.type != BlockType.IMAGE_REF or not block.image_path:
            continue
        page_type_value = (block.metadata or {}).get("page_type", BlockType.ILLUSTRATION.value)
        try:
            page_type = BlockType(page_type_value)
        except Exception:
            page_type = BlockType.ILLUSTRATION
        page_images.setdefault(
            (int(getattr(block, "page", 0) or 0), str(block.image_path)),
            (page_type, str(block.image_path), copy.deepcopy(block.metadata or {})),
        )

    combined = list(text_blocks)
    confident = 0
    approximate = 0
    pending = 0
    front_types = {BlockType.COVER, BlockType.TITLE_PAGE, BlockType.FRONTISPIECE, BlockType.COLOR_ILLUS, BlockType.TOC_PAGE}
    end_types = {BlockType.COLOPHON, BlockType.ADVERTISEMENT}

    for (page_no, image_path), (page_type, _path, metadata) in sorted(page_images.items(), key=lambda item: item[0][0]):
        if page_type in front_types:
            insert_at = next((i for i, block in enumerate(combined) if block.type != BlockType.IMAGE_REF), len(combined))
            confident += 1
        elif page_type in end_types:
            insert_at = len(combined)
            confident += 1
        else:
            insert_at = len(combined)
            for i, block in enumerate(combined):
                if block.type == BlockType.IMAGE_REF:
                    continue
                if int(getattr(block, "page", 0) or 0) > page_no:
                    insert_at = i
                    break
            nearby = [
                block for block in combined[max(0, insert_at - 2):insert_at + 2]
                if block.type != BlockType.IMAGE_REF
            ]
            if nearby:
                confident += 1 if any((block.metadata or {}).get("source_page_alignment") == "confident" for block in nearby) else 0
                approximate += 0 if any((block.metadata or {}).get("source_page_alignment") == "confident" for block in nearby) else 1
            else:
                pending += 1

        previous_text = next(
            (block for block in reversed(combined[:insert_at]) if block.type != BlockType.IMAGE_REF),
            None,
        )
        image = Block(
            type=BlockType.IMAGE_REF,
            page=page_no,
            reading_order=insert_at,
            image_path=image_path,
            image_anchor=previous_text.id if previous_text is not None else "start",
            confidence=1.0,
            metadata={**metadata, "strict_anchor": "page_order"},
            chapter_index=int(getattr(previous_text, "chapter_index", 0) or 0) if previous_text else 0,
        )
        combined.insert(insert_at, image)

    for index, block in enumerate(combined):
        block.reading_order = index
    return combined, confident, approximate, pending


def strict_replace_text(
    ocr_doc: UnifiedDocument,
    source_paragraphs: list[Paragraph],
    *,
    reflow: bool = True,
) -> tuple[UnifiedDocument, "ReplacementReport"]:
    """Rebuild all body text from the trusted replacement source.

    ``reflow=True`` keeps the source words authoritative while projecting the OCR
    document's paragraph/dialogue/page boundaries back onto damaged merged source
    blocks. ``reflow=False`` is the old literal block-for-block behaviour.
    """
    t0 = time.time()
    source_copy = [copy.deepcopy(p) for p in source_paragraphs if str(p.text or "") != ""]
    page_map, confident_pages = _map_source_pages(ocr_doc, source_copy)

    if reflow:
        segments, reflow_stats = reflow_source_paragraphs(ocr_doc, source_copy, page_map)
        text_blocks, toc = _authoritative_reflow_blocks(segments)
        mode = "strict_full"
    else:
        reflow_stats = None
        text_blocks, toc = _authoritative_source_blocks(source_copy, page_map, confident_pages)
        mode = "strict_literal"

    blocks, image_confident, image_approximate, image_pending = _strict_image_blocks(ocr_doc, text_blocks)
    out = copy.deepcopy(ocr_doc)
    out.blocks = blocks
    out.toc = toc
    out.metadata.preserve_ocr_layout = False
    out.metadata.replacement_mode = mode

    literal_source_text = _canonical_body_text_from_paragraphs(source_copy)
    literal_output_text = _canonical_body_text_from_doc(out)
    literal_missing, literal_extra, literal_ratio = _diff_character_counts(literal_source_text, literal_output_text)
    literal_exact = literal_source_text == literal_output_text

    if reflow:
        source_text = layout_neutral_text("".join(str(p.text or "") for p in source_copy))
        output_text = layout_neutral_text("".join(
            str(block.text or "") for block in out.blocks if block.type in _TEXT_TYPES
        ))
    else:
        source_text = literal_source_text
        output_text = literal_output_text

    missing, extra, ratio = _diff_character_counts(source_text, output_text)
    source_hash = _sha256_text(source_text)
    output_hash = _sha256_text(output_text)
    exact = source_hash == output_hash and source_text == output_text

    before_layout = reflow_stats.before if reflow_stats else analyze_layout_texts([p.text for p in source_copy])
    after_layout = reflow_stats.after if reflow_stats else analyze_layout_texts([b.text for b in text_blocks])

    out.metadata.replacement_source_hash = source_hash
    out.metadata.replacement_output_hash = output_hash
    out.metadata.replacement_exact_match = exact
    out.metadata.replacement_literal_exact_match = literal_exact
    out.metadata.replacement_source_chars = len(source_text)
    out.metadata.replacement_output_chars = len(output_text)
    out.metadata.replacement_missing_chars = missing
    out.metadata.replacement_extra_chars = extra
    out.metadata.replacement_pending_images = image_pending
    out.metadata.replacement_layout_passed = after_layout.passed
    out.metadata.replacement_overlong_blocks = after_layout.overlong_blocks
    out.metadata.replacement_mixed_dialogue_blocks = after_layout.mixed_dialogue_blocks
    out.metadata.replacement_unbalanced_dialogue_blocks = after_layout.unbalanced_dialogue_blocks
    out.metadata.replacement_reflowed_blocks = int(reflow_stats.reflowed_source_blocks if reflow_stats else 0)
    out.metadata.replacement_unresolved_layout_blocks = int(reflow_stats.unresolved_blocks if reflow_stats else 0)
    out.metadata.replacement_quote_repairs = int(
        (reflow_stats.quote_insertions + reflow_stats.quote_moves) if reflow_stats else 0
    )
    out.add_log(
        "strict_full_replacement",
        f"authoritative source rebuilt {len(text_blocks)} text blocks; reflow={reflow}; "
        f"content_exact={exact}; literal_exact={literal_exact}; missing={missing}; "
        f"extra={extra}; layout_passed={after_layout.passed}; pending_images={image_pending}",
        len(text_blocks),
    )

    report = ReplacementReport(
        mode=mode,
        ocr_paragraph_count=len(_ocr_paragraphs_from_doc(ocr_doc)),
        source_paragraph_count=len(source_copy),
        matched=len(confident_pages),
        replaced=len(text_blocks),
        skipped_ocr=0,
        skipped_source=0,
        avg_similarity=ratio,
        execution_seconds=time.time() - t0,
        exact_match=exact,
        literal_exact_match=literal_exact,
        source_chars=len(source_text),
        output_chars=len(output_text),
        missing_chars=missing,
        extra_chars=extra,
        source_hash=source_hash,
        output_hash=output_hash,
        image_blocks=image_confident + image_approximate + image_pending,
        image_anchors_confident=image_confident,
        image_anchors_approximate=image_approximate,
        image_anchors_pending=image_pending,
        chapter_count=len(toc),
        layout_passed=after_layout.passed,
        layout_overlong_before=before_layout.overlong_blocks,
        layout_overlong_after=after_layout.overlong_blocks,
        layout_mixed_before=before_layout.mixed_dialogue_blocks,
        layout_mixed_after=after_layout.mixed_dialogue_blocks,
        layout_unbalanced_before=before_layout.unbalanced_dialogue_blocks,
        layout_unbalanced_after=after_layout.unbalanced_dialogue_blocks,
        reflowed_blocks=int(reflow_stats.reflowed_source_blocks if reflow_stats else 0),
        unresolved_layout_blocks=int(reflow_stats.unresolved_blocks if reflow_stats else 0),
        quote_repairs=int((reflow_stats.quote_insertions + reflow_stats.quote_moves) if reflow_stats else 0),
        literal_missing_chars=literal_missing,
        literal_extra_chars=literal_extra,
        literal_similarity=literal_ratio,
    )
    return out, report


def compare_text_only(
    ocr_doc: UnifiedDocument,
    source_paragraphs: list[Paragraph],
) -> "ReplacementReport":
    t0 = time.time()
    ocr_text = _canonical_body_text_from_doc(ocr_doc)
    source_text = _canonical_body_text_from_paragraphs(source_paragraphs)
    ocr_only, source_only, ratio = _diff_character_counts(ocr_text, source_text)
    return ReplacementReport(
        mode="compare_only",
        ocr_paragraph_count=len(_ocr_paragraphs_from_doc(ocr_doc)),
        source_paragraph_count=len(source_paragraphs),
        avg_similarity=ratio,
        execution_seconds=time.time() - t0,
        exact_match=ocr_text == source_text,
        source_chars=len(source_text),
        output_chars=len(ocr_text),
        missing_chars=source_only,
        extra_chars=ocr_only,
        source_hash=_sha256_text(source_text),
        output_hash=_sha256_text(ocr_text),
    )


def format_replacement_report(report: "ReplacementReport") -> str:
    mode_labels = {
        "strict_full": "严格覆盖（重建小说排版）",
        "strict_literal": "严格覆盖（完全原样）",
        "smart_patch": "局部智能替换",
        "compare_only": "仅比较差异",
    }
    lines = [f"模式: {mode_labels.get(report.mode, report.mode or '局部智能替换')}"]
    if report.mode in ("strict_full", "strict_literal", "compare_only"):
        lines.extend([
            f"来源正文字符: {report.source_chars}",
            f"当前/输出正文字符: {report.output_chars}",
            f"一致率: {report.avg_similarity:.6%}",
            f"缺失字符: {report.missing_chars}",
            f"额外字符: {report.extra_chars}",
            f"来源 SHA-256: {report.source_hash}",
            f"输出 SHA-256: {report.output_hash}",
            f"正文内容完整: {'是' if report.exact_match else '否'}",
            f"块边界逐字原样: {'是' if report.literal_exact_match else '否'}",
        ])
    if report.mode in ("strict_full", "strict_literal"):
        lines.extend([
            f"重建正文块: {report.replaced}",
            f"章节: {report.chapter_count}",
            f"图片: {report.image_blocks}",
            f"图片定位（高置信）: {report.image_anchors_confident}",
            f"图片定位（近似）: {report.image_anchors_approximate}",
            f"待确认图片: {report.image_anchors_pending}",
            "OCR 正文残留: 0（正文块已全部重建）",
            f"排版重建块: {report.reflowed_blocks}",
            f"引号结构修复: {report.quote_repairs}",
            f"超长块（前/后）: {report.layout_overlong_before} / {report.layout_overlong_after}",
            f"混合对白块（前/后）: {report.layout_mixed_before} / {report.layout_mixed_after}",
            f"未平衡对白块（前/后）: {report.layout_unbalanced_before} / {report.layout_unbalanced_after}",
            f"仍需人工确认的排版块: {report.unresolved_layout_blocks}",
            f"排版校验: {'通过' if report.layout_passed else '存在异常'}",
        ])
    elif report.mode == "smart_patch" or not report.mode:
        lines.extend([
            f"实际替换: {report.replaced}",
            f"低置信度跳过: {report.low_confidence}",
            f"来源无 OCR 对应: {report.skipped_source}",
            f"OCR 无来源对应: {report.skipped_ocr}",
            f"替换后清理: {report.post_deduped}",
        ])
    lines.append(f"耗时: {report.execution_seconds:.2f}s")
    return "\n".join(lines)


def run_replacement_mode(
    ocr_doc: UnifiedDocument,
    source_paragraphs: list[Paragraph],
    mode: str = "strict_full",
) -> tuple[UnifiedDocument | None, "ReplacementReport"]:
    if mode == "strict_full":
        return strict_replace_text(ocr_doc, source_paragraphs, reflow=True)
    if mode == "strict_literal":
        return strict_replace_text(ocr_doc, source_paragraphs, reflow=False)
    if mode == "compare_only":
        return None, compare_text_only(ocr_doc, source_paragraphs)
    doc, report = replace_text(ocr_doc, source_paragraphs)
    report.mode = "smart_patch"
    doc.metadata.replacement_mode = "smart_patch"
    return doc, report

@dataclass
class ReplacementReport:
    mode: str = "smart_patch"
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
    post_deduped: int = 0
    exact_match: bool = False
    literal_exact_match: bool = False
    source_chars: int = 0
    output_chars: int = 0
    missing_chars: int = 0
    extra_chars: int = 0
    source_hash: str = ""
    output_hash: str = ""
    image_blocks: int = 0
    image_anchors_confident: int = 0
    image_anchors_approximate: int = 0
    image_anchors_pending: int = 0
    chapter_count: int = 0
    layout_passed: bool = True
    layout_overlong_before: int = 0
    layout_overlong_after: int = 0
    layout_mixed_before: int = 0
    layout_mixed_after: int = 0
    layout_unbalanced_before: int = 0
    layout_unbalanced_after: int = 0
    reflowed_blocks: int = 0
    unresolved_layout_blocks: int = 0
    quote_repairs: int = 0
    literal_missing_chars: int = 0
    literal_extra_chars: int = 0
    literal_similarity: float = 0.0
    execution_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
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
            "post_deduped": self.post_deduped,
            "exact_match": self.exact_match,
            "literal_exact_match": self.literal_exact_match,
            "source_chars": self.source_chars,
            "output_chars": self.output_chars,
            "missing_chars": self.missing_chars,
            "extra_chars": self.extra_chars,
            "source_hash": self.source_hash,
            "output_hash": self.output_hash,
            "image_blocks": self.image_blocks,
            "image_anchors_confident": self.image_anchors_confident,
            "image_anchors_approximate": self.image_anchors_approximate,
            "image_anchors_pending": self.image_anchors_pending,
            "chapter_count": self.chapter_count,
            "layout_passed": self.layout_passed,
            "layout_overlong_before": self.layout_overlong_before,
            "layout_overlong_after": self.layout_overlong_after,
            "layout_mixed_before": self.layout_mixed_before,
            "layout_mixed_after": self.layout_mixed_after,
            "layout_unbalanced_before": self.layout_unbalanced_before,
            "layout_unbalanced_after": self.layout_unbalanced_after,
            "reflowed_blocks": self.reflowed_blocks,
            "unresolved_layout_blocks": self.unresolved_layout_blocks,
            "quote_repairs": self.quote_repairs,
            "literal_missing_chars": self.literal_missing_chars,
            "literal_extra_chars": self.literal_extra_chars,
            "literal_similarity": round(self.literal_similarity, 4),
            "execution_seconds": round(self.execution_seconds, 2),
        }


def replace_text(
    ocr_doc: UnifiedDocument,
    source_paragraphs: list[Paragraph],
    match_threshold: float = 0.3,
    force_replace: bool = False,
) -> tuple[UnifiedDocument, ReplacementReport]:
    t0 = time.time()

    canonicalizer = VerticalOCRCanonicalizer()
    if canonicalizer.is_applicable(ocr_doc):
        return _replace_vertical_logical_text(
            ocr_doc,
            source_paragraphs,
            match_threshold=match_threshold,
            force_replace=force_replace,
        )

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

            # 低阈值只用于“寻找候选对齐”，不能直接等同于“允许覆盖正文”。
            # 0.3 左右的弱匹配会把来源中完全不同的人称/段落写入 OCR 正文，
            # 造成 EPUB 出现逻辑跳跃。写回时采用独立的安全门槛，并检查长度比。
            ocr_text = ocr_paragraphs[pair.ocr_index].text
            source_text = source_paragraphs[pair.source_index].text
            compact_ocr = re.sub(r"[\s　]+", "", normalize_for_alignment(ocr_text))
            compact_src = re.sub(r"[\s　]+", "", normalize_for_alignment(source_text))
            length_ratio = min(len(compact_ocr), len(compact_src)) / max(len(compact_ocr), len(compact_src), 1)
            safe_write_threshold = max(match_threshold, 0.52)
            safe_pair = pair.similarity >= safe_write_threshold and length_ratio >= 0.45

            if safe_pair or force_replace:
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
                    block.text = replacer.replace(block.text, src_text)
                    block.modified_by = "text_replacement"
                    block.confidence = pair.similarity
                    replaced += 1
            else:
                low_confidence += 1
        elif pair.source_index is not None:
            if len(unmatched_source_preview) < 50:
                unmatched_source_preview.append(source_paragraphs[pair.source_index].text[:40])

    _correct_unreplaced_ocr_blocks(new_doc)
    post_deduped = _post_replacement_prefix_dedup(new_doc) + _post_replacement_long_run_dedup(new_doc, source_paragraphs)

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
        post_deduped=post_deduped,
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
