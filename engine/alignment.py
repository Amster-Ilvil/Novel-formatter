#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alignment Engine —— 把 OCR 段落序列跟高质量来源段落序列对齐，得到一份
"OCR 第几段对应来源第几段"的映射，供 replacement_engine.py 做替换。

不能假设"第 N 个 OCR 块 = 第 N 个来源段落"——真实 OCR 会漏识别、多识别、
把一段拆成两段、把两段并成一段，直接按位置对应会导致后面全部错位。

两层策略（对应"章节定位 → Sliding Window/Fuzzy Match"）：
    1. 章节锚点层：先把两边都按章节标题切成区间，用标题相似度做一次
       Needleman-Wunsch 对齐，找出哪些章节能对上——章节数量少，这一层
       几乎不会算错，而且能大幅缩小下一层要处理的范围，避免误差累积到
       后面章节。
    2. 段落层：在每一对匹配上的章节区间内部，再用段落文本相似度做一次
       Needleman-Wunsch，处理漏识别/多识别/拆分/合并——不是简单顺序
       对应，也不只是拿 SequenceMatcher 扫一遍完事，是真正的全局序列
       对齐（DP，允许插入/删除，找全局最优解，而不是贪心局部匹配）。

相似度函数目前用 difflib.SequenceMatcher.ratio()（标准库，不引入新依赖）；
如果以后想换成 Levenshtein/其它更精细的相似度算法，只需要替换
_text_similarity，NW 框架本身不用动。
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from models.paragraph import Paragraph


def _text_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _normalize_title(title: str) -> str:
    """去除常见无意义字符，只保留核心关键词，提高匹配鲁棒性"""
    # 去除括号内容、数字编号、常见分隔符
    cleaned = re.sub(r'[（(][^）)]*[）)]', '', title)
    cleaned = re.sub(r'[\[【].*?[\]】]', '', cleaned)
    cleaned = re.sub(r'[第].*?[章話節]', '', cleaned)  # 去掉“第X章”等编号，但保留后面的标题文本
    cleaned = re.sub(r'[・：:．\-—~]', ' ', cleaned)
    cleaned = re.sub(r'\s+', '', cleaned)  # 去除所有空白
    return cleaned.strip()


def _title_similarity(a: str, b: str) -> float:
    """规范化后计算标题相似度（若规范化后为空则返回原文本相似度）"""
    na, nb = _normalize_title(a), _normalize_title(b)
    if na and nb:
        return _text_similarity(na, nb)
    return _text_similarity(a, b)


def _needleman_wunsch(
    len_a: int,
    len_b: int,
    similarity_fn: Callable[[int, int], float],
    gap_penalty: float = -0.4,
    match_threshold: float = 0.3,
) -> list[tuple[Optional[int], Optional[int], float]]:
    """
    经典 Needleman-Wunsch 全局序列对齐（DP，O(len_a * len_b)）。

    similarity_fn(i, j) 返回 a[i] 跟 b[j] 的相似度（0~1）。相似度低于
    match_threshold 时按"强制不匹配"扣分（比空位对齐还差），宁可让算法
    选择跳过（插入/删除）也不要把明显对不上的两段硬凑在一起。

    返回按顺序排列的 (a_idx 或 None, b_idx 或 None, similarity) 列表；
    一侧为 None 表示这一段在另一边没有对应内容（漏识别/多识别）。
    """
    dp = [[0.0] * (len_b + 1) for _ in range(len_a + 1)]
    for i in range(1, len_a + 1):
        dp[i][0] = dp[i - 1][0] + gap_penalty
    for j in range(1, len_b + 1):
        dp[0][j] = dp[0][j - 1] + gap_penalty

    trace = [[""] * (len_b + 1) for _ in range(len_a + 1)]
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            sim = similarity_fn(i - 1, j - 1)
            match_score = sim if sim >= match_threshold else -1.0
            diag = dp[i - 1][j - 1] + match_score
            up = dp[i - 1][j] + gap_penalty
            left = dp[i][j - 1] + gap_penalty
            best = max(diag, up, left)
            dp[i][j] = best
            trace[i][j] = "diag" if best == diag else ("up" if best == up else "left")

    pairs: list[tuple[Optional[int], Optional[int], float]] = []
    i, j = len_a, len_b
    while i > 0 or j > 0:
        if i > 0 and j > 0 and trace[i][j] == "diag":
            pairs.append((i - 1, j - 1, similarity_fn(i - 1, j - 1)))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or trace[i][j] == "up"):
            pairs.append((i - 1, None, 0.0))
            i -= 1
        else:
            pairs.append((None, j - 1, 0.0))
            j -= 1
    pairs.reverse()
    return pairs


def _chapter_spans(paragraphs: list[Paragraph]) -> list[tuple[str, int, int]]:
    """按 paragraph.chapter 把段落列表切成 (章节标题, start, end) 区间（end 不含）。
    没有任何章节信息时整份当一个区间处理，退化成纯段落层对齐。"""
    if not paragraphs:
        return []
    spans: list[tuple[str, int, int]] = []
    current = paragraphs[0].chapter or "（无章节标题）"
    start = 0
    for i, p in enumerate(paragraphs):
        title = p.chapter or "（无章节标题）"
        if title != current:
            spans.append((current, start, i))
            current = title
            start = i
    spans.append((current, start, len(paragraphs)))
    return spans


@dataclass
class AlignmentPair:
    ocr_index: Optional[int]
    source_index: Optional[int]
    similarity: float = 0.0


@dataclass
class AlignmentResult:
    pairs: list[AlignmentPair] = field(default_factory=list)
    matched: int = 0
    skipped_ocr: int = 0        # OCR 有、来源没有对应内容
    skipped_source: int = 0     # 来源有、OCR 没有对应内容
    conflicts: int = 0          # 配对上了但相似度低于阈值（低置信度匹配，仍然算 matched）
    avg_similarity: float = 0.0
    chapter_matches: list[tuple[str, str, float]] = field(default_factory=list)  # (ocr_title, src_title, similarity)


def align(
    ocr_paragraphs: list[Paragraph],
    source_paragraphs: list[Paragraph],
    match_threshold: float = 0.3,
    debug: bool = False,
) -> AlignmentResult:
    """两层 Needleman-Wunsch 对齐入口。debug=True 时打印详细的匹配信息。"""
    ocr_spans = _chapter_spans(ocr_paragraphs)
    source_spans = _chapter_spans(source_paragraphs)

    if debug:
        print(f"🔍 OCR 章节数: {len(ocr_spans)}，来源章节数: {len(source_spans)}")
        if len(ocr_spans) <= 20:
            print("OCR 章节标题:", [t for t, _, _ in ocr_spans])
            print("来源章节标题:", [t for t, _, _ in source_spans])

    def title_sim(i: int, j: int) -> float:
        return _title_similarity(ocr_spans[i][0], source_spans[j][0])

    chapter_pairs = _needleman_wunsch(
        len(ocr_spans), len(source_spans), title_sim,
        gap_penalty=-0.3, match_threshold=0.5,
    )

    # 收集章节匹配情况
    chapter_matches: list[tuple[str, str, float]] = []
    for oi, si, sim in chapter_pairs:
        if oi is not None and si is not None:
            ocr_title = ocr_spans[oi][0]
            src_title = source_spans[si][0]
            chapter_matches.append((ocr_title, src_title, sim))
            if debug:
                print(f"  章节匹配: {ocr_title[:30]} ↔ {src_title[:30]} (相似度 {sim:.2f})")
        elif oi is not None:
            if debug:
                print(f"  OCR 独有章节: {ocr_spans[oi][0][:30]}")
        else:
            if debug:
                print(f"  来源独有章节: {source_spans[si][0][:30]}")

    all_pairs: list[AlignmentPair] = []

    for ocr_span_idx, src_span_idx, _ in chapter_pairs:
        if ocr_span_idx is not None and src_span_idx is not None:
            _, o_start, o_end = ocr_spans[ocr_span_idx]
            _, s_start, s_end = source_spans[src_span_idx]
            seg_ocr = ocr_paragraphs[o_start:o_end]
            seg_src = source_paragraphs[s_start:s_end]

            if debug:
                print(f"   对齐章节区间: OCR[{o_start}:{o_end}] ↔ 来源[{s_start}:{s_end}] (段落数 {len(seg_ocr)} vs {len(seg_src)})")

            def text_sim(oi: int, si: int, _seg_ocr=seg_ocr, _seg_src=seg_src) -> float:
                return _text_similarity(_seg_ocr[oi].text, _seg_src[si].text)

            local_pairs = _needleman_wunsch(
                len(seg_ocr), len(seg_src), text_sim,
                gap_penalty=-0.4, match_threshold=match_threshold,
            )
            # 收集局部匹配示例（前5对）
            matched_in_this_chapter = 0
            for oi, si, sim in local_pairs:
                if oi is not None and si is not None:
                    matched_in_this_chapter += 1
                    if debug and matched_in_this_chapter <= 5:
                        print(f"      段落匹配示例: OCR[{o_start+oi}] ↔ 来源[{s_start+si}] (相似度 {sim:.2f})")
                all_pairs.append(AlignmentPair(
                    ocr_index=(o_start + oi) if oi is not None else None,
                    source_index=(s_start + si) if si is not None else None,
                    similarity=sim,
                ))
            if debug:
                print(f"      本章节匹配 {matched_in_this_chapter} 对，总段落 {len(local_pairs)}")
        elif ocr_span_idx is not None:
            _, o_start, o_end = ocr_spans[ocr_span_idx]
            for oi in range(o_start, o_end):
                all_pairs.append(AlignmentPair(ocr_index=oi, source_index=None))
        else:
            _, s_start, s_end = source_spans[src_span_idx]
            for si in range(s_start, s_end):
                all_pairs.append(AlignmentPair(ocr_index=None, source_index=si))

    matched_pairs = [p for p in all_pairs if p.ocr_index is not None and p.source_index is not None]
    sims = [p.similarity for p in matched_pairs]

    return AlignmentResult(
        pairs=all_pairs,
        matched=len(matched_pairs),
        skipped_ocr=sum(1 for p in all_pairs if p.ocr_index is not None and p.source_index is None),
        skipped_source=sum(1 for p in all_pairs if p.ocr_index is None and p.source_index is not None),
        conflicts=sum(1 for p in matched_pairs if p.similarity < match_threshold),
        avg_similarity=(sum(sims) / len(sims)) if sims else 0.0,
        chapter_matches=chapter_matches,
    )