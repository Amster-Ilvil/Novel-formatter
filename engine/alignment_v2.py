#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Alignment Engine v2 —— 四层混合对齐。

engine/replacement_engine.py 一直依赖 `from engine.alignment_v2 import align`，
但这个文件之前并不存在（只有旧的 engine/alignment.py），"🔀 文本替换"按钮
一点就直接抛 ModuleNotFoundError。本文件是补齐的正式实现。

设计针对旧版 engine/alignment.py（全量两两比较、无章节锚点）的三个已知问题：

    1. 全局逐段比对 → O(n²)，大文档卡顿
       解决：按章节把文档切成小段，每段内部再用 N-gram 索引做候选筛选
       （engine/ngram_index.py），整体复杂度接近 O(n+m)，不是 O(n*m)。

    2. 缺乏模糊容错 → 一个标点不对就整段判失败
       解决：候选筛选用 n-gram 重叠度（容忍局部差异），精确打分用
       difflib.SequenceMatcher（连续子序列比例，不是全等判断）。

    3. 依赖前后顺序 → 一处错位后续全部错位（连锁反应）
       解决：先按章节标题做锚点匹配，把大文档切成互相独立的小段——
       一段内的误匹配不会扩散到下一段。段内匹配也不强制严格递增，
       而是从候选集里选内容最相似的一个，对局部乱序更宽容。

四层对应关系：
    第一层 文本清洗与归一化   → engine/text_similarity.normalize_for_match
    第二层 章节锚点分段        → _chapter_segments / _match_segments（本文件）
    第三层 N-gram 候选筛选     → engine/ngram_index.NGramIndex
    第四层 精确相似度打分       → engine/text_similarity.similarity（在候选集内做）

真正把文字写回 Block 的字符级替换（diff-match-patch）在
engine/replacement_engine.py 里做，本文件只负责"谁跟谁匹配、匹配得
有多像"，职责边界跟旧版 alignment.py 保持一致，replacement_engine.py
不用改。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.ngram_index import NGramIndex
from engine.text_similarity import similarity as char_similarity
from engine.japanese_normalizer import compare_key


@dataclass
class MatchPair:
    ocr_index: "int | None"
    source_index: "int | None"
    similarity: float


@dataclass
class AlignResult:
    pairs: list
    matched: int
    skipped_ocr: int
    skipped_source: int
    avg_similarity: float


# 段内候选筛选时，n-gram 索引返回的候选数量上限。太小会漏掉真正的匹配，
# 太大会让精确打分（SequenceMatcher）跑得更慢——5~10 是经验值，足够
# 覆盖"OCR 把一句拆成两句/合成一句"这类常见错位。
_CANDIDATE_TOP_K = 8

# 章节标题相似度低于这个值就不认为是同一章，转入位置兜底匹配。
_TITLE_MATCH_THRESHOLD = 0.35


def _chapter_segments(paragraphs: list):
    """按 is_title 把段落切成章节段。

    返回 [(title_or_None, start, end), ...]，start/end 是该段在
    paragraphs 列表里的下标范围 [start, end)。整份文档从来没有标题时，
    退化成一个大段（title=None），对齐逻辑自动退化成"整书一次性
    n-gram 索引匹配"——比旧版的全量两两比较依然快得多。
    """
    if not paragraphs:
        return []

    segments = []
    seg_start = 0
    seg_title = None
    seen_title = False

    for i, p in enumerate(paragraphs):
        if p.is_title:
            if i > seg_start or seen_title:
                segments.append((seg_title, seg_start, i))
            seg_start = i
            seg_title = p.text
            seen_title = True

    segments.append((seg_title, seg_start, len(paragraphs)))
    return [s for s in segments if s[2] > s[1]]


def _match_segments(ocr_segments: list, src_segments: list):
    """给 OCR 章节段和来源章节段配对，返回 [(ocr_seg_idx, src_seg_idx_or_None), ...]。

    第一轮：按标题相似度做贪心最优匹配——章节标题即使被 OCR 识别得
    不完全准，字符相似度也通常够高（"第三章"vs"第3章"这种差异
    SequenceMatcher 容忍得住）。
    第二轮：标题没匹配上（或本来就没有标题）的段，按剩余来源段的
    相对顺序依次兜底对应——保证章节数对不上（漏检/多检）时仍有一个
    大致合理的映射，而不是直接放弃整段、全部转成"未匹配"。
    """
    used_src = set()
    pairs = []

    for oi, (o_title, _, _) in enumerate(ocr_segments):
        if not o_title:
            continue
        best_j, best_score = None, 0.0
        for sj, (s_title, _, _) in enumerate(src_segments):
            if sj in used_src or not s_title:
                continue
            score = char_similarity(o_title, s_title)
            if score > best_score:
                best_score = score
                best_j = sj
        if best_j is not None and best_score >= _TITLE_MATCH_THRESHOLD:
            used_src.add(best_j)
            pairs.append((oi, best_j))

    matched_oi = {oi for oi, _ in pairs}
    remaining_src = sorted(sj for sj in range(len(src_segments)) if sj not in used_src)
    cursor = 0
    for oi in range(len(ocr_segments)):
        if oi in matched_oi:
            continue
        if cursor < len(remaining_src):
            pairs.append((oi, remaining_src[cursor]))
            cursor += 1
        else:
            pairs.append((oi, None))

    pairs.sort(key=lambda pr: pr[0])
    return pairs


def clean_alignment_blocks(items):
    """匹配阶段过滤非正文节点。"""
    result=[]
    for x in items:
        t=getattr(x, "text", "") or ""
        btype=getattr(x, "type", None)
        name=str(btype)
        if "IMAGE" in name or "PAGE" in name or "META" in name:
            continue
        if t.strip():
            result.append(x)
    return result


def cross_block_windows(items, start, window=3):
    """生成跨 block 合并文本窗口，用于后续增强匹配。"""
    out=[]
    for size in range(1, window+1):
        for i in range(start, min(len(items)-size+1, start+window)):
            text="".join(getattr(x, "text", "") for x in items[i:i+size])
            if text:
                out.append((i, size, text))
    return out


class HybridAligner:
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

    def align(self, ocr: list, src: list) -> AlignResult:
        ocr = clean_alignment_blocks(ocr)
        src = clean_alignment_blocks(src)
        if not ocr and not src:
            return AlignResult(pairs=[], matched=0, skipped_ocr=0, skipped_source=0, avg_similarity=0.0)

        ocr_segments = _chapter_segments(ocr)
        src_segments = _chapter_segments(src)
        seg_pairs = _match_segments(ocr_segments, src_segments)

        pairs: list = []

        for oi, sj in seg_pairs:
            _, o_start, o_end = ocr_segments[oi]
            ocr_slice = list(range(o_start, o_end))

            if sj is None:
                # 这一段在来源里完全找不到对应内容，原文全部保留（不替换）。
                for oidx in ocr_slice:
                    pairs.append(MatchPair(oidx, None, 0.0))
                continue

            _, s_start, s_end = src_segments[sj]
            src_slice = list(range(s_start, s_end))
            src_texts = [src[i].text for i in src_slice]

            # 第三层：该章节段内部建 n-gram 索引，快速筛候选
            index = NGramIndex(src_texts, n=3)
            used_local = set()

            for oidx in ocr_slice:
                o_text = ocr[oidx].text
                candidates = index.query(o_text, top_k=_CANDIDATE_TOP_K)

                best_local, best_score = None, 0.0
                for c in candidates:
                    if c in used_local:
                        continue
                    # 第四层：候选集内的精确字符级相似度
                    score = max(char_similarity(o_text, src_texts[c]), char_similarity(compare_key(o_text), compare_key(src_texts[c])))
                    if score > best_score:
                        best_score = score
                        best_local = c

                if best_local is None and oidx + 1 < len(ocr):
                    # 跨 block 断裂补偿：尝试把当前及下一段合并后寻找来源。
                    merged = o_text + ocr[oidx + 1].text
                    for c in index.query(merged, top_k=_CANDIDATE_TOP_K):
                        score=max(char_similarity(compare_key(merged), compare_key(src_texts[c])), char_similarity(merged, src_texts[c]))
                        if score > best_score:
                            best_score=score
                            best_local=c

                if best_local is not None:
                    used_local.add(best_local)
                    src_idx = src_slice[best_local]
                    pairs.append(MatchPair(oidx, src_idx, best_score))
                else:
                    pairs.append(MatchPair(oidx, None, best_score))

            # 该段内没被用上的来源段落 —— 来源有、OCR 没对应，不自动插入
            for local_i, sidx in enumerate(src_slice):
                if local_i not in used_local:
                    pairs.append(MatchPair(None, sidx, 0.0))

        matched = sum(1 for p in pairs if p.ocr_index is not None and p.source_index is not None)
        skipped_ocr = sum(1 for p in pairs if p.ocr_index is not None and p.source_index is None)
        skipped_source = sum(1 for p in pairs if p.source_index is not None and p.ocr_index is None)
        sims = [p.similarity for p in pairs if p.ocr_index is not None and p.source_index is not None]
        avg_similarity = sum(sims) / len(sims) if sims else 0.0

        return AlignResult(
            pairs=pairs,
            matched=matched,
            skipped_ocr=skipped_ocr,
            skipped_source=skipped_source,
            avg_similarity=avg_similarity,
        )


def align(ocr: list, source: list, match_threshold: float = 0.3) -> AlignResult:
    return HybridAligner(match_threshold).align(ocr, source)
