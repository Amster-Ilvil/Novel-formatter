#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OCR 输出的码位级规范化。

规则移植来源（只搬表和判定，不引依赖）：
- jaconv：半角片假名→全角（含浊点/半浊点两码位合一，ｶ+ﾞ→ガ）
- textlint-ja：NFD 分解假名（か+゛）、康熙部首码位混入（⾨ vs 門）、
  非法控制字符——三条零误报字符规则

全部规则均为"码位错误"级别修正：不改变文本语义，只把 OCR/复制粘贴
产生的错误码位换成正规码位，对小说正文零风险。
"""
from __future__ import annotations

import re
import unicodedata

# 半角片假名/半角日文标点区（含 ｰ 长音、｡｢｣､ 标点、浊点半浊点）
_HALFWIDTH_KANA_RUN = re.compile(r'[｡-ﾟ]+')

# 康熙部首（U+2F00–2FD5）与 CJK 部首补充（U+2E80–2EF3）：OCR 常把普通汉字
# 识别成这些"长得一样"的部首码位；NFKC 对它们有到统一汉字的标准映射。
_RADICAL_CHAR = re.compile(r'[⺀-⻳⼀-⿕]')

# C0 控制符（保留 \n \t）、C1 控制符、BOM/零宽垃圾
_CONTROL_CHARS = re.compile(
    r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-​‎‏﻿]'
)

# 夹在假名之间的连字符/减号/制表线变体 → 长音符。
# 不碰 —(U+2014)/―(U+2015)：那是小说里正当的破折号。
_KANA = r'[ぁ-ゟァ-ヺー]'
_DASH_BETWEEN_KANA = re.compile(rf'(?<={_KANA})[‐‑‒–−─](?={_KANA})')


def normalize_ocr_codepoints(text: str) -> tuple[str, dict[str, int]]:
    """修正 OCR 常见的错误码位。返回 (修正后文本, 各类修正计数)。"""
    counts: dict[str, int] = {}
    if not text:
        return text, counts

    # 1. NFD 分解假名合成（か+゛→が）。NFC 是纯正规组合，不动兼容字符。
    composed = unicodedata.normalize("NFC", text)
    if composed != text:
        counts["nfd_kana"] = sum(1 for a, b in zip(text, composed) if a != b) or 1
        text = composed

    # 2. 半角片假名/半角日文标点 → 全角（NFKC 对该区做标准映射并自动合成浊点）
    def _widen(m: re.Match) -> str:
        return unicodedata.normalize("NFKC", m.group(0))

    widened = _HALFWIDTH_KANA_RUN.sub(_widen, text)
    if widened != text:
        counts["halfwidth_kana"] = len(_HALFWIDTH_KANA_RUN.findall(text))
        text = widened

    # 3. 康熙部首码位 → 统一汉字
    def _to_ideograph(m: re.Match) -> str:
        mapped = unicodedata.normalize("NFKC", m.group(0))
        return mapped if mapped != m.group(0) else m.group(0)

    fixed = _RADICAL_CHAR.sub(_to_ideograph, text)
    if fixed != text:
        counts["kangxi_radical"] = len(_RADICAL_CHAR.findall(text))
        text = fixed

    # 4. 控制字符/零宽垃圾
    cleaned = _CONTROL_CHARS.sub("", text)
    if cleaned != text:
        counts["control_char"] = len(text) - len(cleaned)
        text = cleaned

    # 5. 假名间的连字符变体 → 长音符
    dashed = _DASH_BETWEEN_KANA.sub("ー", text)
    if dashed != text:
        counts["dash_variant"] = len(_DASH_BETWEEN_KANA.findall(text))
        text = dashed

    return text, counts
