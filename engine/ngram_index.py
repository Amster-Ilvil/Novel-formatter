#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NGramIndex —— 字符 n-gram 倒排索引。

用途：在"文本替换"里，OCR 段落要跟高质量来源段落做匹配。如果对每个 OCR
段落都去跟来源里的全部段落跑一次 SequenceMatcher，就是 O(n*m)，几百页的
小说会有几千个段落，两两比较会很卡。

做法参考 passim（n-gram 索引 + 字符级对齐）：先对来源段落建一次 n-gram
倒排索引（O(m)），之后每个 OCR 段落只需要查一次索引（均摊 O(1) 到
O(k)，k 是该段落命中的候选数），就能拿到"n-gram 有重叠的候选段落"，
候选数远小于 m。真正精确的字符级相似度（SequenceMatcher）只需要在这
一小撮候选里跑，整体从 O(n*m) 降到接近 O(n+m)。
"""

from __future__ import annotations

import re
from collections import defaultdict


def _normalize(text: str) -> str:
    """n-gram 切分前的最小清洗：去掉空白（含全角空格）。更完整的标点/
    异体字归一化交给 text_similarity.normalize_for_match，这里不重复。"""
    if not text:
        return ""
    return re.sub(r"[\s\u3000]+", "", text)


def _ngrams(text: str, n: int = 3) -> set:
    t = _normalize(text)
    if not t:
        return set()
    if len(t) < n:
        # 短句（比如只有一两个字的对白"……"）退化成整句当一个 gram，
        # 否则短句永远没有 n-gram、永远查不到候选。
        return {t}
    return {t[i:i + n] for i in range(len(t) - n + 1)}


class NGramIndex:
    """对一批段落文本建立 n-gram 倒排索引，支持快速查询候选段落下标。"""

    def __init__(self, texts: list, n: int = 3):
        self.n = n
        self.texts = texts
        self._grams: list = [_ngrams(t, n) for t in texts]
        self._index: dict = defaultdict(list)
        for idx, grams in enumerate(self._grams):
            for g in grams:
                self._index[g].append(idx)

    def query(self, text: str, top_k: int = 8) -> list:
        """返回按 n-gram Jaccard 重叠度从高到低排序的候选段落下标
        （最多 top_k 个）。重叠度为 0 的段落根本不会进入候选集
        （倒排索引天然跳过），这是比暴力扫描全部段落快的关键。"""
        q = _ngrams(text, self.n)
        if not q:
            return []

        overlap: dict = defaultdict(int)
        for g in q:
            for idx in self._index.get(g, ()):
                overlap[idx] += 1
        if not overlap:
            return []

        def jaccard(idx: int, ov: int) -> float:
            union = len(q) + len(self._grams[idx]) - ov
            return ov / union if union else 0.0

        ranked = sorted(overlap.items(), key=lambda kv: jaccard(kv[0], kv[1]), reverse=True)
        return [idx for idx, _ in ranked[:top_k]]
