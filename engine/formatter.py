#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Novel Formatter Engine
对 UnifiedDocument 中的 blocks 进行后处理。

每个步骤都是独立的函数，接收并返回 UnifiedDocument，
以便 Pipeline 灵活组合、单步预览或回滚。

步骤：
    1.  reading_order            — GapTree 阅读顺序恢复（竖排日文右→左）
    2.  clean_metadata_blocks    — 删除残留的页眉/页脚块（位置+频率双重检测）
    3.  merge_broken_sentences   — 合并竖排 OCR 产生的跨行断句（日语接续词感知）
    4.  remove_semantic_duplicates — 近邻语义去重，保留更完整/更自然文本
    5.  fix_ocr_dash_artifacts   — 修复破折号被误读成「/｜的 OCR 错字
    6.  restore_dialogue_breaks  — 对白独立换行（迭代拆分混合段落）
    7.  restore_indents_and_breaks — 缩进和分节符恢复
    8.  recover_ruby             — 振假名恢复（｜漢字《よみ》→ ruby 标注）
    9.  detect_chapters          — 章节识别（正规化后匹配，灵活模式）
    10. normalize_punctuation    — 标点规范（省略号、破折号、全半角）
"""

from __future__ import annotations

import re
import copy
import difflib
import unicodedata
import uuid
from pathlib import Path
from typing import Optional, Callable
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.document import UnifiedDocument, Block, BlockType, TocEntry, new_temp_repo_path
from engine.ai_formatter import AIFormatterStep, ai_correction_step
from engine.text_similarity import similarity as _text_similarity
from engine.column_sentence_reflow import reflow_columns_into_sentences

AI_STEP_ENABLED = True

# ── 常量 ──────────────────────────────────────────────────────────────────────

# 章节/小节标题正则（正规化后匹配：先去除多余空格）
#
# 轻小说既可能用“第十二話”，也常用“12話 / １２話 / 12話 前編”。
# 带“第”的格式允许直接跟副标题；不带“第”的短格式只有在行尾、空格、
# 括号或明确的前后篇标记前才视为标题，避免把正文里的“一話だけ”误判。
JP_CHAPTER_NUMBER = r'[一二三四五六七八九十百千〇零\d０-９]+'
CHAPTER_UNIT = r'[章話節巻回幕篇編]'
CHAPTER_CONTINUATION = r'(?:は|が|を|に|で|と|も|の|です|だ|という)'
CHAPTER_RE = re.compile(
    rf'^(序章|終章|プロローグ|フロローグ|ブロローグ|エピローグ|後記|あとがき|'
    rf'幕間(?:[\s　:：・—―-].*)?|'
    rf'第[\s　]*{JP_CHAPTER_NUMBER}[\s　]*{CHAPTER_UNIT}(?!{CHAPTER_CONTINUATION})|'
    rf'{JP_CHAPTER_NUMBER}[\s　]*[話章節回](?=$|[\s　:：・—―「『【（(]|前編|後編|上編|中編|下編)|'
    rf'(?:Chapter|Episode|EP)[\s　.．_-]*[\d０-９]+)',
    re.IGNORECASE
)
# 「フロローグ」「ブロローグ」都不是正确拼写，但分别可能由半浊点丢失、浊点
# 混淆而来。先放进识别正则保证标题不被后续步骤当成正文删除或合并；文字层面的
# 纠正交给 normalize_punctuation 的 OCR_TYPOS。
AFTERWORD_RE = re.compile(r'^(後記|あとがき|おわりに|跋|编者按)')
SECTION_RE = re.compile(r'^[◆※☆★●○＊◇■□▼▽△▲]{1,5}$')

# 已知的"前书/后书"样板文字签名——常见于网络轻小说 PDF/EPUB 转换服务自动
# 加在书首/书尾的版权声明、转载须知、站点介绍（"小説家になろう"/"タテ書き
# 小説ネット"这类站点的固定文案）。不是标题也不是正文，命中即视为需要整段
# 剥离的前书/后书内容。列表式常量，方便以后遇到别的站点样板再往里加签名，
# 而不是搞一套容易误伤真实前言/后记的模糊启发式。
KNOWN_BOILERPLATE_RE = re.compile(
    r'小説家になろう|タテ書き小説ネット|ＰＤＦ小説ネット|発足にあたって|'
    r'ルビ対応|無断で|転載、改変、再配布、販売|引用の範囲'
)

# 对白标记
DIALOGUE_START = ('「', '『', '（')
DIALOGUE_END   = ('」', '』', '）')

# 标点替换规则
PUNCT_RULES = [
    # 混合半角/全角句点、中点和省略号会在竖排阅读器中使用不同字形基线，
    # 形成一高一低的 '..…'。只要连续标点中含句点或省略号且长度>=2，
    # 统一为日文规范的双省略号。单个中点仍保留。
    (re.compile(r'(?=[.．・…]{2,})(?=[.．・…]*[.．…])[.．・…]{2,}'), '……'),
    (re.compile(r'\.{2,}'),    '……'),
    (re.compile(r'…{2,}'),     '……'),
    (re.compile(r'-{2,}'),     '——'),
    (re.compile(r'ー{2,}'),    '——'),
    (re.compile(r'\(([^)]{1,20})\)'), r'（\1）'),
    (re.compile(r'　+$'),      ''),
    (re.compile(r' +$'),       ''),
]

CHAPTER_TITLE_SPACE_RE = re.compile(
    rf'^((?:プロローグ|序章|終章|エピローグ|'
    rf'第[\s　]*{JP_CHAPTER_NUMBER}[\s　]*{CHAPTER_UNIT}|'
    rf'{JP_CHAPTER_NUMBER}[\s　]*[話章節回])(?=\S))'
)

# 合并断句：句子结束符
SENTENCE_END_RE = re.compile(r'[。．！？!?」』）～…—\n]$')

# 日语接续词/助词（句尾出现时不应断句）
CONJUNCTION_ENDINGS = re.compile(r'(て|で|し|から|けど|けれど|ので|のに|ながら|ため|たり|ば|と|が|も|は|を|に|へ|の)$')

# Ruby 正则：｜漢字《よみ》
RUBY_RE = re.compile(r'[｜\|]([^《]+)《([^》]+)》')

# 纯数字（±空白），1~6 位：页码，或者本身就是章节/篇章序号（比如轻小说
# 常见的"００１""００２"这种裸数字当小标题用）。两种含义没法从文字
# 本身分辨，所以统一按"结构性标记，不参与断句合并、按来源决定要不要
# 当页码删掉"处理，而不是当成普通正文文字。
BARE_NUMBER_RE = re.compile(r'^[\d\s]{1,6}$')

DEDUP_EXACT = True
SEMANTIC_DUP_WINDOW = 12
SEMANTIC_DUP_SIMILARITY = 0.95
MAX_MERGE_LINES = 20
OVERLAP_MIN_CHARS = 5
OVERLAP_SCAN_WINDOW = 12
# 跨页页眉模糊聚类阈值。页眉可能有一两个 OCR 错字，仍应被识别为同一文本。
HEADER_SIMILARITY = 0.6


# ── 步骤 1：阅读顺序恢复 ────────────────────────────────────────────────────

def reading_order_step(doc: UnifiedDocument) -> UnifiedDocument:
    """GapTree 列聚类阅读顺序恢复"""
    from engine.reading_order import restore_reading_order
    return restore_reading_order(doc)


# ── 步骤 2：清理残留页眉/页脚（位置感知）────────────────────────────────────

def _cluster_similar_texts(texts: list[str], threshold: float) -> dict[str, int]:
    """
    对文本做简单的并查集模糊聚类：两条文本相似度 ≥ threshold 就合并到同一簇。
    返回 {文本: 簇id}。设了个规模上限，候选文本太多（大部头书）就跳过聚类，
    避免 O(n²) 相似度比较拖垮性能——那种规模下退化为"精确匹配"没有聚类，
    仍然正确，只是不再能合并 OCR 噪声导致的近似重复。
    """
    unique = list(dict.fromkeys(texts))  # 保序去重
    n = len(unique)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    MAX_CANDIDATES = 400
    if n <= MAX_CANDIDATES:
        for x in range(n):
            for y in range(x + 1, n):
                if _text_similarity(unique[x], unique[y]) >= threshold:
                    union(x, y)

    return {unique[i]: find(i) for i in range(n)}


def _detect_toc_like_pages(doc: UnifiedDocument) -> set[int]:
    """
    找出"同一页里有 ≥2 个明显不同的章节标题候选"的页——基本可以断定是
    印刷版目录/索引页，而不是真正的章节开头（真正的章节标题不会跟别的
    章节标题挤在同一页）。

    这个结果有两个用途：
        1. detect_chapters 用它来避免把目录页列出的每一条都提升成章节
        2. clean_metadata_blocks 用它把这些页的文字排除在"跨页页眉"聚类
           候选池之外——目录页天然会印出和正文里真章节标题很相似的文字
           （标题本身、甚至副标题片段），拿去跟正文做相似度聚类，会把
           目录页的预告文字和正文里真正的章节标题错误地聚成一类，
           导致真正的章节标题被当成"页眉的重复出现"删掉。
    """
    # 用"章节标识符"本身（比如"第一章""第三章""プロローグ"）来判断是否
    # 明显不同，而不是拿整句文字算相似度——"第一章"和"第三章"整句的编辑
    # 距离相似度能到 0.67（只差一个数字字符），比"魔王"被 OCR 错读成"魔玉"
    # 这类噪声的相似度还高，直接拿相似度阈值判断会把"这明明是两个不同的
    # 章节"误判成"同一章节的噪声变体"。只要提取出的标识符字符串本身不同
    # （"一"≠"三"），就一定是不同章节，不需要模糊比较。
    page_ids: dict[int, set[str]] = {}
    for b in doc.blocks:
        if b.type not in (BlockType.PARAGRAPH, BlockType.CHAPTER):
            continue
        t = b.text.strip()
        if not t:
            continue
        normalized = re.sub(r'[\s　]+', '', t)
        m = CHAPTER_RE.match(normalized) or CHAPTER_RE.match(t)
        if m:
            page_ids.setdefault(b.page, set()).add(m.group(1))

    return {page for page, ids in page_ids.items() if len(ids) >= 2}


def clean_metadata_blocks(doc: UnifiedDocument) -> UnifiedDocument:
    """
    删除类型为 HEADER_FOOTER 的块，以及：
        - 纯数字行（页码）
        - 跨页页眉：同一段文字（或模糊相似的变体）出现在 ≥ 2 个不同页码上、
          长度 ≤ 25（很多排版会在每页顶部印当前章节名当"柱"，这种页眉哪怕
          文字本身长得像"第一章"也要按页眉处理——否则每翻几页就会被
          detect_chapters 误判成又一个新章节，EPUB 里章节标题反复出现）
        - 与章节标题完全重复的短行
        - 位置感知：bbox y < 0.15 或 y > 0.85 的短文本视为页眉/页脚候选

    页眉判定用模糊聚类而不是精确字符串匹配：同一个页眉在不同页被重新扫描，
    每次 OCR 出来的噪声都不一样（"魔王"读成"魔玉"、破折号长短不一），精确
    匹配会把每次出现都当成"只出现过一次"的不同文本，永远达不到跨页阈值，
    页眉就漏网了。聚类后按"这一簇覆盖了几个不同页码"判定，同一簇里但只在
    单页内重复（OCR 把同一标题读了两三遍）不算跨页页眉，交给 remove_duplicates
    的模糊去重处理。
    """
    doc = copy.deepcopy(doc)

    # PDF 文字层直读在提取阶段就已经用坐标位置（页面底部 20%）精确判定过
    # 页码并跳过了，这里"文字整行都是数字"这条粗糙的兜底规则对它来说既没
    # 必要、还会误伤——竖排小说里章节/篇章序号常常就是单独一行的纯数字
    # （比如"００１"这种全角数字），跟真正的页码长得一样但根本不是页码，
    # 位置也不在页面底部，只有这条不看位置、纯看"像不像数字"的规则会把
    # 它当页码删掉。所以只对不是文字层直读来源的文档启用这条规则。
    skip_bare_digit_rule = doc.metadata.source_engine == "pdf_text_layer"

    # 目录/索引页要整页排除在"跨页页眉"聚类之外（原因见 _detect_toc_like_pages
    # 的说明），否则目录页预告的标题文字会和正文里真正的章节标题被错误地
    # 聚成一类，真正的章节标题反而被当成"页眉的重复出现"删掉。
    toc_like_pages = _detect_toc_like_pages(doc)

    HEADER_MAX = 25
    candidate_texts: list[str] = []
    text_pages: dict[str, set[int]] = {}
    for b in doc.blocks:
        if b.page in toc_like_pages:
            continue
        # 注意：OCR 适配器在识别阶段就会用同一套章节正则，把"看起来像章节
        # 标题"的行提前标成 CHAPTER（不是等 Formatter 的 detect_chapters
        # 来判定）。真正的跨页页眉大多数就是章节名，所以这里必须把已经是
        # CHAPTER 类型的块也纳入候选池，否则聚类永远看不到它们，页眉判断
        # 会系统性漏掉所有已经被提前识别成"章节"的页眉。
        if b.type in (BlockType.PARAGRAPH, BlockType.HEADER_FOOTER, BlockType.CHAPTER):
            t = b.text.strip()
            if t and len(t) <= HEADER_MAX:
                candidate_texts.append(t)
                text_pages.setdefault(t, set()).add(b.page)

    cluster_of = _cluster_similar_texts(candidate_texts, HEADER_SIMILARITY)

    cluster_pages: dict[int, set[int]] = {}
    for t, cid in cluster_of.items():
        cluster_pages.setdefault(cid, set()).update(text_pages[t])

    running_header_clusters = {cid for cid, pages in cluster_pages.items() if len(pages) >= 2}

    # 标题属于书籍结构，不能因跨页重复而删除。即使排版上看起来像页眉，
    # 在没有版面语义的前提下也不能牺牲章节标题来换取页眉清理率。
    cluster_keep_idx: dict[int, int] = {}
    cluster_keep_is_chapter_match: dict[int, bool] = {}
    for i, b in enumerate(doc.blocks):
        if b.type not in (BlockType.PARAGRAPH, BlockType.CHAPTER, BlockType.HEADER_FOOTER):
            continue
        t = b.text.strip()
        cid = cluster_of.get(t)
        if cid is None or cid not in running_header_clusters:
            continue
        normalized = re.sub(r'[\s　]+', '', t)
        is_match = bool(CHAPTER_RE.match(normalized) or CHAPTER_RE.match(t))
        if cid not in cluster_keep_idx or (is_match and not cluster_keep_is_chapter_match[cid]):
            cluster_keep_idx[cid] = i
            cluster_keep_is_chapter_match[cid] = is_match

    removed = 0
    kept: list[Block] = []
    for i, b in enumerate(doc.blocks):
        if b.type == BlockType.HEADER_FOOTER:
            removed += 1
            continue
        if b.type == BlockType.CHAPTER:
            kept.append(b)
            continue
        if b.type == BlockType.PARAGRAPH:
            t = b.text.strip()
            normalized = re.sub(r'[\s　]+', '', t)
            if CHAPTER_RE.match(normalized) or CHAPTER_RE.match(t):
                kept.append(b)
                continue
            if not skip_bare_digit_rule and BARE_NUMBER_RE.match(t):
                removed += 1
                continue
            cid = cluster_of.get(t)
            if cid is not None and cid in running_header_clusters and cluster_keep_idx.get(cid) != i:
                removed += 1
                continue
            # 位置感知：bbox 靠近页面顶部或底部的短文本。
            #
            # 竖排日文正文列本来就从页面上缘附近开始，单看 ``bbox.y < 0.15``
            # 会把短正文列、对白和 Ruby 整列误删成页眉。竖排模式下只有
            # “浅而横”的边缘条带才按位置删除；真正的竖排正文列（高度明显
            # 大于宽度）必须保留。横排文档继续沿用原来的顶部/底部规则。
            if b.bbox and len(t) <= HEADER_MAX:
                y = float(b.bbox.y)
                h = max(0.0, float(b.bbox.h))
                w = max(0.0, float(b.bbox.w))
                writing_direction = str(getattr(doc.metadata, "writing_direction", "") or "")
                vertical = writing_direction.startswith("vertical")
                edge_hit = y < 0.15 or y > 0.85
                if vertical:
                    shallow_horizontal_strip = h <= 0.10 and (w >= 0.10 or w >= h * 1.35)
                    positional_header = edge_hit and shallow_horizontal_strip
                else:
                    positional_header = edge_hit
                if positional_header:
                    removed += 1
                    continue
        kept.append(b)

    doc.blocks = kept
    doc.add_log("clean_metadata", f"删除 {removed} 个页眉/页码块", removed)
    return doc


# ── 步骤 2.4：拆出内嵌章节标题 ──────────────────────────────────────────────

EMBEDDED_CHAPTER_RE = re.compile(
    rf'(序章|終章|プロローグ|フロローグ|ブロローグ|エピローグ|'
    rf'第[\s　]*{JP_CHAPTER_NUMBER}[\s　]*{CHAPTER_UNIT}(?!{CHAPTER_CONTINUATION})|'
    rf'{JP_CHAPTER_NUMBER}[\s　]*[話章節回](?=$|[\s　:：・—―「『【（(]|前編|後編|上編|中編|下編))'
)
# 只在真正的句子边界（句末标点/引号收尾/换行）之后才认定是新标题的开始，
# 避免把叙述里偶然提到"序章""プロローグ"这类词的普通句子（比如"それは
# まるでこの物語の序章に過ぎなかった"）误当成一个新标题来拆分。
SENTENCE_BOUNDARY_RE = re.compile(r'[。！？」』\n]$')


def split_embedded_chapter_titles(doc: UnifiedDocument) -> UnifiedDocument:
    """
    章节标题偶尔不在块开头，而是跟前一段内容粘连在同一个块里——PDF 文字层
    按列分块时，如果标题所在列和上一段末尾恰好落在同一分列区间，就会被
    不加区分地拼成一段。CHAPTER_RE 只在块开头匹配（re.match 语义），这种
    情况下永远匹配不到，标题永远无法被识别、还会跟前面内容一起被当成
    正文、结果既不出现在目录里，也没有从正文里独立出来。

    这里额外在块内部搜索一次，只在"紧跟在句子边界之后"才认定是真正的
    标题开始，把块从这里切开：前半段留在原处（原类型不变），后半段单独
    提升成新块，交给后面的 detect_chapters/strip_chapter_notes 按正常
    流程处理。只处理 PARAGRAPH 类型的块。

    只在 pdf_text_layer/epub_import 来源生效，理由同 strip_boilerplate_matter
    ——这是精确文本来源特有的按列/按标签合并伪影，不是 OCR 噪声的性质。
    """
    doc = copy.deepcopy(doc)

    if doc.metadata.source_engine not in ("pdf_text_layer", "epub_import"):
        doc.add_log("split_embedded_chapter_titles", "跳过（仅对 PDF 文字层直读/EPUB 导入来源生效）", 0)
        return doc

    result: list[Block] = []
    split_count = 0

    for b in doc.blocks:
        if b.type != BlockType.PARAGRAPH:
            result.append(b)
            continue

        t = b.text
        stripped = t.strip()
        if CHAPTER_RE.match(stripped):
            # 已经在块开头，交给 detect_chapters 处理，不需要在这里切
            result.append(b)
            continue

        m = None
        for cand in EMBEDDED_CHAPTER_RE.finditer(t):
            pos = cand.start()
            if pos == 0:
                continue
            preceding = t[:pos].rstrip()
            if not preceding or SENTENCE_BOUNDARY_RE.search(preceding):
                m = cand
                break

        if m is None:
            result.append(b)
            continue

        before = t[:m.start()].strip()
        after = t[m.start():].strip()
        if not after:
            result.append(b)
            continue

        if before:
            nb = copy.copy(b)
            nb.text = before
            nb.modified_by = "split_embedded_chapter_titles"
            result.append(nb)

        cb = copy.copy(b)
        cb.text = after
        cb.modified_by = "split_embedded_chapter_titles"
        result.append(cb)
        split_count += 1

    doc.blocks = result
    doc.add_log("split_embedded_chapter_titles", f"从正文中拆出 {split_count} 处内嵌章节标题", split_count)
    return doc


# ── 步骤 2.5：删除逐章（前書）/（後書き）编辑备注 ──────────────────────────────

# タテ書き小説ネット这类站点的 PDF，会在每一话正文附近单独排一小段给读者看的
# 作者短评/预告（"这是序章第1话，讲的是……""下一话是……的节选"这种），版式上
# 跟"数字＋（前書）"或"数字＋（後書き）"的小标记连在一起。这段版式在原页面里
# 跟正文是独立元素，但按列提取/阅读顺序恢复经常会把它跟正文粘连，或者被
# 误判成一个新章节标题（比如"プロローグ第１話です。"这种描述性文字，被
# CHAPTER_RE 的"プロローグ"前缀误判成了真正的章节标题，实际上是这段备注
# 的一部分）。
CHAPTER_NOTE_MARKER_RE = re.compile(r'[０-９0-9]{2,4}（(?:前書|後書き)）')


def strip_chapter_notes(doc: UnifiedDocument) -> UnifiedDocument:
    """
    删除逐章附带的"（前書）/（後書き）"编辑备注——一旦某个块的文字里出现
    "数字＋（前書）"或"数字＋（後書き）"标记（不管是不是在块开头），就从这个
    标记开始（含标记本身）持续删除后续块，直到遇到下一个"纯裸数字"块
    （比如"００２"本身、不带（前書）/（後書き）后缀）为止——只拿裸数字块当
    "备注结束、真正下一话开始"的信号，而不是随便一个匹配 CHAPTER_RE 的词，
    因为备注文字本身也可能用"プロローグ"这类词开头（"プロローグ第１話です。"
    整句其实是备注，不是真正的章节标题），拿 CHAPTER_RE 当结束信号会被这种
    情况骗过去、提前放行。

    标记之前的文字（如果标记不在块开头、前面还有真正正文）原样保留；
    IMAGE_REF 图片块任何时候都不受影响，直接放行。

    只在 pdf_text_layer/epub_import 来源生效，理由同 strip_boilerplate_matter
    （OCR 来源的噪声性质不同，不该套用同一套规则）。
    """
    doc = copy.deepcopy(doc)

    if doc.metadata.source_engine not in ("pdf_text_layer", "epub_import"):
        doc.add_log("strip_chapter_notes", "跳过（仅对 PDF 文字层直读/EPUB 导入来源生效）", 0)
        return doc

    result: list[Block] = []
    removed = 0
    skipping = False

    for b in doc.blocks:
        if b.type == BlockType.IMAGE_REF:
            result.append(b)
            continue

        t = b.text.strip()

        if skipping:
            if BARE_NUMBER_RE.match(t) and not CHAPTER_NOTE_MARKER_RE.search(t):
                skipping = False
                # 不 continue——落到下面按正常块处理（真正新一话的裸数字标题）
            else:
                removed += 1
                continue

        m = CHAPTER_NOTE_MARKER_RE.search(t)
        if not m:
            result.append(b)
            continue

        before = t[:m.start()].strip()
        if before:
            nb = copy.copy(b)
            nb.text = before
            nb.modified_by = "strip_chapter_notes"
            result.append(nb)
        removed += 1
        skipping = True

    doc.blocks = result
    doc.add_log("strip_chapter_notes", f"删除 {removed} 个逐章前书/后书备注块", removed)
    return doc

# ── 步骤 2.5：基于页码信息修复跨页断句 ─────────────────────────────────────
CROSS_PAGE_TEXT_TYPES = {BlockType.PARAGRAPH, BlockType.DIALOGUE, BlockType.RUBY}
STRONG_SENTENCE_ENDINGS = ("。", "！", "？", "!", "?", "‼", "⁉", "…", "‥", "」", "』", "）", ")", "】", "》")
CONTINUATION_PREFIXES = ("が", "を", "に", "へ", "と", "で", "も", "は", "の", "や", "から", "まで", "より", "って", "ので", "のに", "けれど", "ながら", "つつ", "たり", "て", "し")
POST_QUOTE_CONTINUATION_PREFIXES = (
    "から", "まで", "より", "って", "を", "が", "に", "へ", "と", "で", "は", "の",
)
QUOTE_PAIRS = {"「": "」", "『": "』", "（": "）", "(": ")", "【": "】", "《": "》"}
CROSS_PAGE_AUTO_MERGE_THRESHOLD = 6


def _append_modified_by(existing: str, step: str) -> str:
    parts = [p for p in (existing or "").split(",") if p]
    if step not in parts:
        parts.append(step)
    return ",".join(parts)


def _block_page_index(block: Block) -> int | None:
    value = getattr(block, "page_index", None)
    if value is not None:
        return value
    page = getattr(block, "page", 0)
    return int(page) if page else None


def _block_order_in_page(block: Block) -> int:
    value = getattr(block, "order_in_page", None)
    if value is not None:
        return int(value)
    return int(getattr(block, "reading_order", 0) or 0)


def _get_page_text_blocks(blocks: list[Block]) -> dict[int, list[Block]]:
    pages: dict[int, list[Block]] = {}
    for block in blocks:
        page_index = _block_page_index(block)
        if page_index is None or block.type not in CROSS_PAGE_TEXT_TYPES:
            continue
        if not (block.text or "").strip():
            continue
        pages.setdefault(page_index, []).append(block)
    for page_blocks in pages.values():
        page_blocks.sort(key=_block_order_in_page)
    return pages


def _has_strong_sentence_ending(text: str) -> bool:
    text = (text or "").rstrip()
    if not text:
        return True
    return text.endswith(STRONG_SENTENCE_ENDINGS)


def _has_unclosed_quote(text: str) -> bool:
    text = text or ""
    for opening, closing in QUOTE_PAIRS.items():
        if text.count(opening) > text.count(closing):
            return True
    return False


def _starts_with_japanese_character(text: str) -> bool:
    text = (text or "").lstrip(" 　")
    if not text:
        return False
    first = text[0]
    return (
        "\u3040" <= first <= "\u30ff"
        or "\u3400" <= first <= "\u9fff"
        # Japanese iteration/abbreviation marks are punctuation-codepoint
        # characters but can begin the continuation of a word split by a PDF
        # text layer, e.g. ``日`` + ``々だった``.
        or first in "々〃〆ヶヵー"
    )


def _is_post_quote_continuation(left: str, right: str) -> bool:
    """Return whether a quote/bracket is followed by a grammatical particle.

    ``』/」`` is not necessarily a sentence boundary: quoted nouns and dialogue
    are often followed by ``を/と/が/...``.  PDF text layers frequently split
    exactly at that boundary, so treating every closing quote as a hard stop
    leaves visibly broken paragraphs.
    """
    left = (left or "").rstrip(" \t\r\n　")
    right = (right or "").lstrip(" \t\r\n　")
    return bool(
        left
        and right
        and left.endswith(("」", "』", "）", ")", "】", "》"))
        and right.startswith(POST_QUOTE_CONTINUATION_PREFIXES)
    )




def _looks_like_list_item(text: str) -> bool:
    text = (text or "").lstrip(" 　")
    return bool(re.match(r"^([-*+]|\d+[.)．、]|[・●◼])", text))


def _looks_like_title(block: Block, text: str) -> bool:
    if block.type in {BlockType.CHAPTER, BlockType.SECTION, BlockType.TOC_ENTRY}:
        return True
    text = (text or "").strip()
    return text.startswith("#") or bool(CHAPTER_RE.match(re.sub(r"[\s　]+", "", text)))


def _join_cross_page_text(left: str, right: str) -> str:
    return (left or "").rstrip(" \t\r\n　") + (right or "").lstrip(" \t\r\n　")


def _cross_page_merge_score(previous: Block, next_block: Block) -> int:
    score = 0
    left = (previous.text or "").rstrip()
    right = (next_block.text or "").lstrip()
    previous_page = _block_page_index(previous)
    next_page = _block_page_index(next_block)
    if previous_page is not None:
        score += 2
    if previous_page is not None and next_page == previous_page + 1:
        score += 4
    else:
        return -100
    if previous.type not in CROSS_PAGE_TEXT_TYPES or next_block.type not in CROSS_PAGE_TEXT_TYPES:
        return -100
    post_quote_continuation = _is_post_quote_continuation(left, right)
    if not _has_strong_sentence_ending(left) or post_quote_continuation:
        score += 4
    else:
        score -= 10
    if _has_unclosed_quote(left):
        score += 4
    if right.startswith(CONTINUATION_PREFIXES):
        score += 3
    if _starts_with_japanese_character(right):
        score += 1
    if _looks_like_title(next_block, right):
        score -= 10
    if _looks_like_list_item(right):
        score -= 8
    if right.startswith(("「", "『")) and not _has_unclosed_quote(left):
        score -= 6
    return score


def _non_text_page_numbers(doc: UnifiedDocument) -> set[int]:
    """Pages that must form a hard boundary for text-flow merging."""
    non_text_types = {
        BlockType.COVER, BlockType.COLOR_ILLUS, BlockType.BLANK, BlockType.TOC_PAGE,
        BlockType.ILLUSTRATION, BlockType.COLOPHON, BlockType.TITLE_PAGE,
        BlockType.FRONTISPIECE, BlockType.INSERT, BlockType.ADVERTISEMENT,
        BlockType.INDEX_PAGE, BlockType.MAP_PAGE, BlockType.CHARACTER_SHEET,
    }
    return {int(page.page_no) for page in getattr(doc, "pages", []) if page.page_type in non_text_types}


def merge_cross_page_sentences(doc: UnifiedDocument) -> UnifiedDocument:
    """仅合并相邻文本页的上一页最后正文块和下一页第一正文块。"""
    doc = copy.deepcopy(doc)
    pages = _get_page_text_blocks(doc.blocks)
    non_text_pages = _non_text_page_numbers(doc)
    if not pages:
        doc.add_log("cross_page_merge", "文档没有可靠页码信息，跳过跨页断句恢复", 0)
        return doc

    removed_ids: set[int] = set()
    merged_count = 0
    details: list[str] = []
    for page_index in sorted(pages):
        if page_index + 1 not in pages:
            continue
        if page_index in non_text_pages or page_index + 1 in non_text_pages:
            continue
        previous = pages[page_index][-1]
        next_block = pages[page_index + 1][0]
        if id(previous) in removed_ids or id(next_block) in removed_ids:
            continue
        score = _cross_page_merge_score(previous, next_block)
        if score < CROSS_PAGE_AUTO_MERGE_THRESHOLD:
            continue
        original_left = previous.text
        original_right = next_block.text
        previous.text = _join_cross_page_text(original_left, original_right)
        previous.modified_by = _append_modified_by(previous.modified_by, "merge_cross_page_sentences")
        source_pages = list(dict.fromkeys((previous.metadata or {}).get("source_pages", []) + [page_index, page_index + 1]))
        previous.metadata = {
            **(previous.metadata or {}),
            "cross_page_merged": True,
            "source_pages": source_pages,
            "source_block_ids": [(previous.id or str(id(previous))), (next_block.id or str(id(next_block)))],
            "cross_page_merge": {
                "from_page": page_index,
                "to_page": page_index + 1,
                "left_text": original_left,
                "right_text": original_right,
                "score": score,
            },
        }
        removed_ids.add(id(next_block))
        merged_count += 1
        if len(details) < 5:
            details.append(f"Page {page_index} → Page {page_index + 1} 置信分:{score}")
    doc.blocks = [block for block in doc.blocks if id(block) not in removed_ids]
    suffix = "；" + "；".join(details) if details else ""
    doc.add_log("cross_page_merge", f"恢复 {merged_count} 处跨页断句{suffix}", merged_count)
    return doc



def _get_page_layout_text_blocks(blocks: list[Block]) -> dict[int, list[Block]]:
    """固定排版专用分页索引。

    与 ``_get_page_text_blocks`` 不同，这里保留已经被安全跨页合并清空的正文块。
    这些空块是重要的物理分页占位符：再次运行 Formatter 时，绝不能跳过它们，
    否则会把同页第二段错误地当成“下一页第一段”继续吞进上一页。
    """
    pages: dict[int, list[Block]] = {}
    for block in blocks:
        page_index = _block_page_index(block)
        if page_index is None or block.type not in CROSS_PAGE_TEXT_TYPES:
            continue
        is_consumed_placeholder = bool((block.metadata or {}).get("consumed_by_cross_page_merge"))
        if not (block.text or "").strip() and not is_consumed_placeholder:
            continue
        pages.setdefault(page_index, []).append(block)
    for page_blocks in pages.values():
        page_blocks.sort(key=_block_order_in_page)
    return pages


def _restore_consumed_boundary_if_needed(previous: Block, placeholder: Block) -> bool:
    """恢复被文本替换覆盖掉的跨页续句，但绝不越过占位块。

    文本替换可能重写上一页末块，却保留下一页已清空的物理块。此时再次 Formatter
    不能选取下一页第二段；若上一页末块已经不含原续句，则从占位块 ``ocr_raw``
    恢复一次。返回是否实际恢复了文字。
    """
    metadata = placeholder.metadata or {}
    consumed_by = metadata.get("consumed_by_cross_page_merge")
    if not consumed_by or consumed_by != previous.id:
        return False
    continuation = placeholder.ocr_raw or metadata.get("consumed_text", "")
    continuation = continuation or ""
    if not continuation.strip():
        return False
    left = previous.text or ""
    compact_left = re.sub(r"[\s　]+", "", left)
    compact_right = re.sub(r"[\s　]+", "", continuation)
    # 已经包含该续句时保持幂等，不能重复追加。
    if compact_right and compact_right in compact_left:
        return False
    # 上一页已经成为完整句时，宁可不恢复，也不能误粘下一段。
    if _has_strong_sentence_ending(left):
        return False
    previous.text = _join_cross_page_text(left, continuation)
    previous.modified_by = _append_modified_by(previous.modified_by, "merge_cross_page_sentences_layout_safe")
    previous.metadata = {
        **(previous.metadata or {}),
        "cross_page_merged": True,
        "layout_safe": True,
        "restored_from_consumed_placeholder": placeholder.id,
    }
    return True


def merge_cross_page_sentences_layout_safe(doc: UnifiedDocument) -> UnifiedDocument:
    """固定排版/文本替换后的安全跨页接续。

    仅把下一页首正文块的文字追加到上一页末正文块，并清空来源块文字；
    不删除 Block，不改变页码、坐标、顺序、图片锚点或其它结构信息。
    已清空的来源块始终作为分页边界占位符，后续执行不会越过它误吞下一段。
    """
    doc = copy.deepcopy(doc)
    pages = _get_page_layout_text_blocks(doc.blocks)
    non_text_pages = _non_text_page_numbers(doc)
    if not pages:
        doc.add_log("cross_page_merge", "文档没有可靠页码信息，跳过安全跨页断句恢复", 0)
        return doc

    merged_count = 0
    details: list[str] = []
    consumed_ids: set[str] = set()
    for page_index in sorted(pages):
        if page_index + 1 not in pages:
            continue
        if page_index in non_text_pages or page_index + 1 in non_text_pages:
            continue
        previous = pages[page_index][-1]
        next_block = pages[page_index + 1][0]
        if previous.id in consumed_ids or next_block.id in consumed_ids:
            continue

        # 下一页第一物理正文块若是之前留下的空占位符，就只能恢复该块原续句
        # 或保持不动；绝不能跳到下一页第二段。
        if not (next_block.text or "").strip():
            if _restore_consumed_boundary_if_needed(previous, next_block):
                merged_count += 1
                if len(details) < 5:
                    details.append(f"Page {page_index} → Page {page_index + 1} 从占位块恢复")
            continue

        score = _cross_page_merge_score(previous, next_block)
        if score < CROSS_PAGE_AUTO_MERGE_THRESHOLD:
            continue

        original_left = previous.text or ""
        original_right = next_block.text or ""
        previous.text = _join_cross_page_text(original_left, original_right)
        previous.modified_by = _append_modified_by(previous.modified_by, "merge_cross_page_sentences_layout_safe")
        previous.metadata = {
            **(previous.metadata or {}),
            "cross_page_merged": True,
            "layout_safe": True,
            "source_pages": list(dict.fromkeys((previous.metadata or {}).get("source_pages", []) + [page_index, page_index + 1])),
            "source_block_ids": [previous.id, next_block.id],
        }
        if not next_block.ocr_raw:
            next_block.ocr_raw = original_right
        next_block.text = ""
        next_block.modified_by = _append_modified_by(next_block.modified_by, "merge_cross_page_sentences_layout_safe")
        next_block.metadata = {
            **(next_block.metadata or {}),
            "consumed_by_cross_page_merge": previous.id,
            "consumed_text": original_right,
            "layout_safe": True,
        }
        consumed_ids.add(next_block.id)
        merged_count += 1
        if len(details) < 5:
            details.append(f"Page {page_index} → Page {page_index + 1} 置信分:{score}")

    suffix = "；" + "；".join(details) if details else ""
    doc.add_log("cross_page_merge", f"安全恢复 {merged_count} 处跨页断句{suffix}", merged_count)
    return doc


# ── 步骤 2.75：重叠块解析（必须早于断句合并）──────────────────────────────

def _compact_overlap_text(text: str) -> str:
    """仅移除排版空白，保留标点，以便把索引映射回原文。"""
    return re.sub(r"[\s　]+", "", text or "")


def _longest_suffix_prefix_overlap(left: str, right: str, minimum: int = OVERLAP_MIN_CHARS) -> int:
    """返回 left 后缀与 right 前缀的最长重叠字符数。"""
    a = _compact_overlap_text(left)
    b = _compact_overlap_text(right)
    limit = min(len(a), len(b))
    for size in range(limit, minimum - 1, -1):
        if a[-size:] == b[:size]:
            return size
    return 0


def _strip_compact_prefix(original: str, compact_chars: int) -> str:
    """从原文开头消费指定数量的非空白字符，返回未消费部分。"""
    if compact_chars <= 0:
        return original
    consumed = 0
    for index, ch in enumerate(original):
        if not ch.isspace() and ch != "　":
            consumed += 1
        if consumed >= compact_chars:
            return original[index + 1:]
    return ""


def _repair_internal_prefix_repeat(text: str) -> tuple[str, bool]:
    """
    修复单个块内部的“残句 + 更完整重扫句”重复。

    典型输入：
      最初は...そしてその源最初は...そしてその源がドンドン...
    后半段是前半段的更完整版本，应删除前面的残缺副本。
    只处理长度足够、重复起点明确且后一个版本更长的情况，避免误伤修辞重复。
    """
    compact = _compact_overlap_text(text)
    if len(compact) < 40:
        return text, False

    # 在块中寻找重复出现的长前缀。起点允许位于正文中部，因为前面可能还有
    # 完整的上一句。使用 16~48 字符锚点，越长越可靠。
    for anchor_len in range(min(48, len(compact) // 2), 15, -1):
        for first in range(0, len(compact) - anchor_len * 2 + 1):
            anchor = compact[first:first + anchor_len]
            second = compact.find(anchor, first + anchor_len)
            if second < 0:
                continue
            fragment = compact[first:second]
            tail = compact[second:]
            # 前一版本必须大体是后一版本的前缀，且后一版本确实更完整。
            compare_len = min(len(fragment), len(tail))
            if compare_len < 20:
                continue
            ratio = difflib.SequenceMatcher(None, fragment[:compare_len], tail[:compare_len]).ratio()
            if ratio < 0.92 or len(tail) <= len(fragment):
                continue

            # 将 compact 索引映射回原字符串索引。
            positions = [i for i, ch in enumerate(text) if not ch.isspace() and ch != "　"]
            if second >= len(positions) or first >= len(positions):
                continue
            repaired = text[:positions[first]] + text[positions[second]:]
            return repaired, True
    return text, False


def merge_overlapping_blocks(doc: UnifiedDocument) -> UnifiedDocument:
    """
    合并相邻 OCR 块的公共前后缀，并删除被完整包含的短副本。

    与 remove_duplicates 的区别：这里不是判断“两块是否相似后删一块”，而是
    精确消费 left 的后缀 / right 的前缀重叠，避免 merge_broken_sentences 直接
    相加后生成“最初は…最初は…”一类嵌套重复。
    """
    doc = copy.deepcopy(doc)
    text_types = {BlockType.PARAGRAPH, BlockType.DIALOGUE, BlockType.RUBY}
    result: list[Block] = []
    merged = 0
    contained = 0
    internal = 0

    for source in doc.blocks:
        block = copy.copy(source)
        repaired, changed = _repair_internal_prefix_repeat(block.text or "")
        if changed:
            block.ocr_raw = block.ocr_raw or block.text
            block.text = repaired
            block.modified_by = _append_modified_by(block.modified_by, "merge_overlapping_blocks")
            internal += 1

        if block.type not in text_types or not (block.text or "").strip():
            result.append(block)
            continue

        # 只检查最近若干文本块；结构块不会被跨越合并。
        previous_index = None
        for idx in range(len(result) - 1, max(-1, len(result) - OVERLAP_SCAN_WINDOW - 1), -1):
            if result[idx].type not in text_types:
                break
            previous_index = idx
            break
        if previous_index is None:
            result.append(block)
            continue

        previous = result[previous_index]
        left = _compact_overlap_text(previous.text)
        right = _compact_overlap_text(block.text)
        same_or_adjacent_page = abs(int(getattr(previous, "page", 0) or 0) - int(getattr(block, "page", 0) or 0)) <= 1
        if not same_or_adjacent_page:
            result.append(block)
            continue

        # 完整包含：保留质量更高/更完整的一块。
        if len(right) >= OVERLAP_MIN_CHARS and right in left:
            contained += 1
            continue
        if len(left) >= OVERLAP_MIN_CHARS and left in right:
            block.modified_by = _append_modified_by(block.modified_by, "merge_overlapping_blocks")
            result[previous_index] = block
            contained += 1
            continue

        overlap = _longest_suffix_prefix_overlap(previous.text, block.text)
        if overlap:
            suffix = _strip_compact_prefix(block.text, overlap)
            original_previous_text = previous.text
            previous.text = previous.text.rstrip(" \t\r\n　") + suffix.lstrip(" \t\r\n　")
            previous.ocr_raw = (previous.ocr_raw or original_previous_text) + (block.ocr_raw or block.text)
            previous.modified_by = _append_modified_by(previous.modified_by, "merge_overlapping_blocks")
            merged += 1
            continue

        result.append(block)

    doc.blocks = result
    doc.add_log(
        "merge_overlaps",
        f"重叠拼接 {merged} 处，包含副本 {contained} 处，块内重复 {internal} 处",
        merged + contained + internal,
    )
    return doc

# ── 步骤 3：合并断句（接续词感知）─────────────────────────────────────────────

# 高置信度的“词中间被换列”边界。这里不做任意“汉字+汉字”合并，
# 那会把大量本来独立的无句号段落误粘。仅收录 OCR/竖排小说中反复出现、
# 且跨边界后能组成明确词语的字对；后续可继续补充。
_SPLIT_COMPOUND_BOUNDARIES = {
    ("輪", "郭"), ("性", "別"), ("年", "齢"), ("全", "身"),
    ("王", "城"), ("軍", "勢"), ("光", "景"), ("感", "情"),
    # 日文常见假名词中断裂（ように / 当てはまりません）。
    ("よ", "う"), ("は", "ま"),
}


def _should_merge_with_next(current_text: str) -> bool:
    """判断当前文本块是否应与下一块合并（只看左块）。"""
    text_at_start = current_text.lstrip(' \u3000')
    # 引号还没配对闭合就必须继续合并，不管末尾长得多像句子结束——一句
    # 对白里包含好几个句子很正常（"「……。……。……」"），中间某句用
    # 句号收尾时如果直接判定"句子完整了"就不再合并，对白会在句号处被
    # 提前切断，等不到真正的收尾引号。只认块首的引号：正文中间孤立的
    # 「常是竖排 OCR 把破折号误读出来的字符，不能让它吞掉后续完整句子。
    if text_at_start.startswith('「') and current_text.count('「') > current_text.count('」'):
        return True
    if text_at_start.startswith('『') and current_text.count('『') > current_text.count('』'):
        return True
    if SENTENCE_END_RE.search(current_text):
        return False
    if CONJUNCTION_ENDINGS.search(current_text):
        return True
    if len(current_text.strip()) < 5:
        return True
    # 普通段落即使没有句号，也可能是作者刻意保留的独立强调段、独白或
    # OCR 已正确识别出的段落边界。不能仅凭“末尾无句号”就一路吞并后文。
    # 跨页断句由 merge_cross_page_sentences 专门处理；同页这里只合并有
    # 明确接续词或极短碎片的高置信度情况。
    return False


def _should_merge_pair(current_text: str, next_text: str, *, pdf_text_mode: bool = False) -> bool:
    """宽松判断 OCR 断列。

    只要前块没有明确句末标点、后块不是新对白/标题/列表，就优先视为
    同一句的续写。文本对比工作区负责处理少量复杂误合并，因此这里以
    恢复跨列、跨行和跨页连续阅读为优先。
    """
    if _should_merge_with_next(current_text):
        return True
    left = (current_text or "").rstrip(" \t\r\n　")
    right = (next_text or "").lstrip(" \t\r\n　")
    if not left or not right:
        return False
    if pdf_text_mode and _is_post_quote_continuation(left, right):
        return True
    if SENTENCE_END_RE.search(left):
        return False
    if right.startswith(("「", "『")) and not _has_unclosed_quote(left):
        return False
    # 这些词更常见于新句/新段开头；避免把独立强调句或无句号名词句
    # 一路吞进下一段。除此之外仍按宽松接续处理。
    new_sentence_prefixes = (
        "それが", "それは", "これは", "あれは", "俺は", "私は", "次の",
        "だが", "しかし", "ところが", "一方", "つまり", "要するに", "そもそも",
    )
    if right.startswith(new_sentence_prefixes):
        return False
    if _looks_like_list_item(right) or CHAPTER_RE.match(re.sub(r"[\s　]+", "", right)):
        return False
    if right.startswith(CONTINUATION_PREFIXES):
        return True
    if (left[-1], right[0]) in _SPLIT_COMPOUND_BOUNDARIES:
        return True
    # 日文 OCR 的新列通常直接从假名、汉字或片假名继续。
    return _starts_with_japanese_character(right)


_MERGEABLE_TYPES = {BlockType.PARAGRAPH, BlockType.DIALOGUE}


def merge_broken_sentences(doc: UnifiedDocument) -> UnifiedDocument:
    """
    竖排 OCR 宽松断句合并。当前块没有明确句末标点、下一块不像
    新标题/新对白时优先接回；复杂边界留给文本对比工作区人工校正。

    对白（DIALOGUE）也要参与合并，不能只处理 PARAGRAPH：一句对白因为
    列高/页面限制被切成两三段时，开头那一段光看自己"「"没有配对的
    "」"，会被分类成 DIALOGUE（按 startswith「/endswith」判断），后续
    段落也可能是 DIALOGUE 或 PARAGRAPH（取决于这一段本身有没有带上
    「/」）。之前这里只认 PARAGRAPH，对白被切开后就再也合不回去，
    还会被后面的 fix_dash_artifacts 步骤把孤立的「误判成破折号。
    """
    doc = copy.deepcopy(doc)

    # PDF 文字层直读的分页是从 PDF 本身精确拿到的，句子/对白跨页断开是
    # 常态（版面排到页底就换页，跟句子说没说完没关系）。OCR 文档也允许
    # "本页末尾没有句末符 → 下一页首行"的高置信度跨页续写，避免换页把
    # 一个完整句子硬切开；有句末符的 OCR 块仍不跨页合并。
    pdf_text_mode = is_pdf_text_layer_mode_enabled(doc)
    allow_cross_page = pdf_text_mode

    merged_count = 0
    i = 0
    result: list[Block] = []

    while i < len(doc.blocks):
        b = doc.blocks[i]

        if b.type not in _MERGEABLE_TYPES:
            result.append(b)
            i += 1
            continue

        # 裸数字块（页码，或者轻小说里常见的"００１""００２"这种当小标题
        # 用的篇章序号）不参与合并——它太短，会被下面 len<5 那条"太短肯定
        # 要接着下一段"的规则强行吞掉，直接粘到正文开头，看起来就像标题
        # 消失了。裸数字不是断句断出来的残片，不该套用断句合并的逻辑。
        if (
            CHAPTER_RE.match(b.text.strip())
            or BARE_NUMBER_RE.match(b.text.strip())
            or (b.metadata or {}).get("exclude_from_sentence_merge")
        ):
            result.append(b)
            i += 1
            continue

        merge_count = 0
        while (
            not (b.metadata or {}).get("dialogue_auto_merge_blocked")
            and i + 1 < len(doc.blocks)
            and merge_count < MAX_MERGE_LINES
            and doc.blocks[i + 1].type in _MERGEABLE_TYPES
            and (
                allow_cross_page
                or doc.blocks[i + 1].page == b.page
                or (
                    doc.blocks[i + 1].page == b.page + 1
                    and not SENTENCE_END_RE.search(b.text.rstrip())
                )
            )
            and (
                _should_merge_pair(
                    b.text, doc.blocks[i + 1].text, pdf_text_mode=pdf_text_mode
                )
                or (
                    doc.blocks[i + 1].page == b.page + 1
                    and not SENTENCE_END_RE.search(b.text.rstrip())
                )
            )
            and not CHAPTER_RE.match(doc.blocks[i + 1].text.strip())
            and not BARE_NUMBER_RE.match(doc.blocks[i + 1].text.strip())
            and not (doc.blocks[i + 1].metadata or {}).get("exclude_from_sentence_merge")
        ):
            next_b = doc.blocks[i + 1]
            b = copy.copy(b)
            b.text = b.text + next_b.text
            b.ocr_raw = b.ocr_raw + next_b.ocr_raw
            b.metadata = dict(b.metadata or {})
            merged_source_ids = [
                str(x) for x in (b.metadata.get("source_block_ids") or [b.id])
            ] + [
                str(x) for x in ((next_b.metadata or {}).get("source_block_ids") or [next_b.id])
            ]
            b.metadata["source_block_ids"] = list(dict.fromkeys(merged_source_ids))
            b.modified_by = "merge_broken_sentences"
            i += 1
            merge_count += 1
            merged_count += 1

        result.append(b)
        i += 1

    doc.blocks = result
    doc.add_log("merge_sentences", f"合并 {merged_count} 次断句", merged_count)
    return doc



# ── 步骤 4：删除重复块 ───────────────────────────────────────────────────────



DEDUP_LEADING_MARKS_RE = re.compile(r"^[\s　◆※☆★●○＊◇■□▼▽△▲◼・･\-—–ー]+")
DEDUP_PUNCT_RE = re.compile(
    r"[。．、，,.！？!?…‥・･:：;；"
    r"「」『』（）()\[\]【】《》〈〉"
    r"\"'“”‘’]+"
)


def _normalize_for_semantic_dedup(text: str) -> str:
    """近邻语义去重用规范化：忽略空白、全半角、项目符号和轻微标点差异。"""
    text = unicodedata.normalize("NFKC", text or "")
    text = DEDUP_LEADING_MARKS_RE.sub("", text)
    text = re.sub(r"[\s　]+", "", text)
    text = DEDUP_PUNCT_RE.sub("", text)
    return text.casefold()


def _is_protected_title(block: Block, normalized_text: str) -> bool:
    """标题是结构内容，不能由正文去重规则删除。"""
    return block.type in {BlockType.CHAPTER, BlockType.SECTION} or bool(
        CHAPTER_RE.match(normalized_text)
    )


def _dedup_quality(block: Block) -> tuple:
    text = (block.text or "").strip()
    normalized = _normalize_for_semantic_dedup(text)
    punctuation_score = sum(1 for ch in text if ch in "。！？!?」』）")
    malformed_penalty = text.count("..") + text.count(".・") + text.count("・..") + text.count("�")
    naturalness_score = text.count("など") - text.count("なぞ")
    confidence = float(getattr(block, "confidence", 0.0) or 0.0)
    return (-malformed_penalty, punctuation_score, naturalness_score, len(normalized), confidence)


def _is_near_duplicate(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    shorter, longer = sorted((a, b), key=len)
    if len(shorter) >= 12 and shorter in longer:
        return len(shorter) / len(longer) >= 0.30
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    length = max(len(a), len(b))
    if length >= 60:
        return ratio >= 0.94
    if length >= 30:
        return ratio >= 0.95
    if length >= 15:
        return ratio >= 0.97
    return False


def _remove_repeated_block_runs(doc: UnifiedDocument) -> tuple[UnifiedDocument, int]:
    """Remove an immediately repeated prose run, not merely one nearby block.

    OCR/Markdown reconstruction can duplicate a whole page-sized sequence.  Each
    member is too far from its twin for the ordinary near-neighbour resolver, so
    detect exact normalized runs of at least four text blocks and substantial
    total prose.  Titles and image boundaries stop a run, preventing chapter or
    asset structure from being swallowed.
    """
    out = copy.deepcopy(doc)
    blocks = out.blocks
    text_types = {BlockType.PARAGRAPH, BlockType.DIALOGUE, BlockType.RUBY}
    removed = 0
    changed = True
    while changed:
        changed = False
        normalized = [
            _normalize_for_semantic_dedup(b.text) if b.type in text_types else ""
            for b in blocks
        ]
        n = len(blocks)
        for second_start in range(4, n):
            max_gap = min(80, second_start)
            matched = None
            for gap in range(4, max_gap + 1):
                first_start = second_start - gap
                run = 0
                total_chars = 0
                long_members = 0
                while second_start + run < n and first_start + run < second_start:
                    left = blocks[first_start + run]
                    right = blocks[second_start + run]
                    if left.type not in text_types or right.type not in text_types:
                        break
                    a = normalized[first_start + run]
                    b = normalized[second_start + run]
                    if not a or a != b:
                        break
                    total_chars += len(a)
                    long_members += int(len(a) >= 30)
                    run += 1
                if run >= 4 and total_chars >= 160 and long_members >= 2:
                    matched = (second_start, run)
                    break
            if matched:
                start, run = matched
                del blocks[start:start + run]
                removed += run
                changed = True
                break
    out.blocks = blocks
    return out, removed


def remove_semantic_duplicates(doc: UnifiedDocument) -> UnifiedDocument:
    """仅在近邻窗口内删除语义重复块，优先保留更完整、更自然的版本。"""
    doc = copy.deepcopy(doc)
    text_types = {BlockType.PARAGRAPH, BlockType.RUBY}
    result: list[Block] = []
    removed = 0

    for block in doc.blocks:
        normalized = _normalize_for_semantic_dedup(block.text)
        if block.type not in text_types or not normalized or _is_protected_title(block, normalized):
            result.append(block)
            continue

        candidate_indexes: list[int] = []
        checked = 0
        for index in range(len(result) - 1, -1, -1):
            previous = result[index]
            if previous.type not in text_types:
                continue
            checked += 1
            if checked > SEMANTIC_DUP_WINDOW:
                break
            previous_normalized = _normalize_for_semantic_dedup(previous.text)
            if _is_protected_title(previous, previous_normalized):
                continue
            if (
                previous.type == BlockType.DIALOGUE
                and block.type == BlockType.DIALOGUE
                and getattr(previous, "page", 0) != getattr(block, "page", 0)
            ):
                continue
            if _is_near_duplicate(previous_normalized, normalized):
                candidate_indexes.append(index)

        if not candidate_indexes:
            result.append(block)
            continue

        candidates = [(index, result[index]) for index in candidate_indexes]
        candidates.append((-1, block))
        best_index, best_block = max(candidates, key=lambda item: _dedup_quality(item[1]))
        insert_at = min(candidate_indexes)
        for index in sorted(candidate_indexes, reverse=True):
            del result[index]
            removed += 1
        if best_index == -1:
            result.insert(insert_at, block)
        else:
            result.insert(insert_at, best_block)

    doc.blocks = result
    doc.add_log("remove_duplicates", f"删除 {removed} 个近邻重复块", removed)
    return doc

def remove_duplicates(doc: UnifiedDocument) -> UnifiedDocument:
    """删除整段重复运行，再执行近邻语义 Duplicate Resolver。"""
    run_cleaned, run_removed = _remove_repeated_block_runs(doc)
    out = remove_semantic_duplicates(run_cleaned)
    if run_removed:
        out.add_log("remove_duplicate_runs", f"删除 {run_removed} 个整段重复块", run_removed)
    return out


# ── 步骤 5：修复破折号被误读成引号/竖线 ────────────────────────────────────────

def _fix_ocr_dash_artifacts(text: str) -> str:
    """
    Apple Vision 竖排 OCR 有个常见的字符混淆：把破折号「ー」「——」
    认成「「」（引号）或「｜/|」（竖线）。这类误读有个共同特征，据此
    能比较可靠地识别、又不会误伤真正的对白引号或 ruby 标记：
        - 误读成的「｜/|」不会紧跟着 ruby 读音标记《…》，真正的 ruby
          标记「｜漢字《よみ》」里的｜后面一定紧跟着《
        - 误读成的「「」找不到配对的「」」，真正的对白「…」总是成对出现
    命中就把这个符号还原成破折号「——」。
    """
    if not text:
        return text

    if '|' in text or '｜' in text:
        ruby_marker_positions = {m.start() for m in RUBY_RE.finditer(text)}
        if any(ch in '｜|' for i, ch in enumerate(text) if i not in ruby_marker_positions):
            chars = []
            for i, ch in enumerate(text):
                if ch in '｜|' and i not in ruby_marker_positions:
                    chars.append('——')
                else:
                    chars.append(ch)
            text = ''.join(chars)

    if '「' in text:
        out: list[str] = []
        pending_open: list[int] = []
        for ch in text:
            if ch == '「':
                out.append(ch)
                pending_open.append(len(out) - 1)
            elif ch == '」':
                if pending_open:
                    pending_open.pop()
                out.append(ch)
            else:
                out.append(ch)
        for idx in pending_open:
            out[idx] = '——'
        text = ''.join(out)

    # 竖线和孤立引号可能紧挨着（同一个破折号的两笔误读被拆成两个符号），
    # 各自被还原成 —— 后会连成 ————，这里收拢成一个
    text = re.sub(r'(——|ー){2,}', '——', text)

    return text


def fix_ocr_dash_artifacts(doc: UnifiedDocument) -> UnifiedDocument:
    """对每个文本块应用 _fix_ocr_dash_artifacts。放在对白拆分之前跑，
    避免误读出来的「/｜先把后面的对白识别/ruby识别带偏。

    PDF 文字层直读的文档整个跳过这一步——这条规则专门修"OCR 把破折号
    误读成「/｜"，文字层直读是从 PDF 字体数据里逐字符精确取出来的，
    根本不存在"误读"这回事；如果一句对白因为跨列/跨页被切断，切开的
    那一段本身就会有落单的「或」（跟误读毫无关系，纯粹是断句），这条
    规则会把落单的「当成误读的破折号删掉，反而把好端端的引号改错了。
    """
    if is_pdf_text_layer_mode_enabled(doc):
        return copy.deepcopy(doc)

    doc = copy.deepcopy(doc)

    text_types = {BlockType.PARAGRAPH, BlockType.DIALOGUE, BlockType.CHAPTER, BlockType.SECTION}
    fixed_count = 0

    for b in doc.blocks:
        if b.type not in text_types:
            continue
        if (b.metadata or {}).get("dialogue_auto_merge_blocked"):
            # 未确定闭引号位置的对白必须保留原始「，不能把它当成 OCR 误读
            # 的破折号改成 ——；否则人工/AI 后续连原对白边界都无法恢复。
            continue
        original = b.text
        b.text = _fix_ocr_dash_artifacts(b.text)
        if b.text != original:
            b.ocr_raw = b.ocr_raw or original
            b.modified_by = "fix_ocr_dash_artifacts"
            fixed_count += 1

    doc.add_log("fix_ocr_dash_artifacts", f"修复 {fixed_count} 处误读的破折号", fixed_count)
    return doc


# ── 步骤 5.5：短列缺失闭引号与跨块对白续接修复 ────────────────────────────

# 竖排小说中，短对白通常独占一列。OCR 偶尔会漏掉列尾的闭引号「」中的「」」，
# 也会把同一句对白从列中间切成两个块。修复时必须先判断下一块是否仍是本句，
# 不能看到当前块很短就立即补」；否则 ``ここ / はわたしに…`` 会被错误变成
# ``ここ」 / はわたしに…``。
SHORT_DIALOGUE_MAX_CHARS = 45
MERGED_DIALOGUE_MAX_CHARS = 180
_DIALOGUE_INCOMPLETE_END_RE = re.compile(
    # 省略号/破折号经常正是对白的完整结尾（迟疑、打断、话音中止），
    # 不能仅凭它们就认定下一块仍属对白，否则会把后续叙述整段吞进去。
    r'(?:[、，,・]|(?:て|で|し|が|を|に|へ|と|から|ので|のに|けど|けれど|ながら|ため))$'
)
# 下一块以这些助词、接续形式或句尾辅助表达开头时，几乎不可能是一个全新的
# 普通叙述段；它更可能是上一列/上一块的后半句。长词优先，避免 ``ので`` 先被
# ``の`` 命中虽然结果相同但可读性较差。
_DIALOGUE_CONTINUATION_PREFIXES = tuple(sorted(set(CONTINUATION_PREFIXES + (
    "んだ", "んです", "のだ", "のです", "なのだ", "なのです",
    "だ", "です", "ます", "ない", "なかった", "な", "よ", "ぞ", "ぜ",
    "か", "ね", "わ", "さ",
)), key=len, reverse=True))

# 只凭左块末尾判断时使用更严格的接续集合。``の/は/も/が`` 等既可能是
# 格助词，也经常作为完整对白的句末表达，不能据此吞掉下一段叙述。
_DIALOGUE_STRONG_LEFT_CONTINUATION_RE = re.compile(
    r'(?:て|で|し|が|を|に|へ|と|から|ので|のに|けど|けれど|ながら|ため)$'
)
_DIALOGUE_SHORT_CONTINUATION_MAX = 24


def _strip_imported_block_marker(text: str) -> tuple[str, str]:
    """返回 (可保留前缀, 正文)。只忽略空白和常见 UI/Markdown 块标记。"""
    original = text or ""
    prefix_match = re.match(r"^(?P<prefix>[\s　]*(?:[◼■●・･]\s*)?)", original)
    prefix = prefix_match.group("prefix") if prefix_match else ""
    return prefix, original[len(prefix):]


def _looks_like_dialogue_continuation(
    left_body: str,
    right_text: str,
    *,
    reopening_existing_quote: bool = False,
    pdf_text_mode: bool = False,
) -> bool:
    """判断 right 是否高置信度地继续一个尚未结束的「对白。

    只使用语法边界信号，不凭“下一块也是日文”就合并，避免把真正的独白或
    叙述吞进对白。典型安全信号：``ここ / は…``、``その源 / が…``、
    ``任せて / 逃げる…``、``している / んだ``。
    """
    left = (left_body or "").rstrip(" \t\r\n　")
    _, right_content = _strip_imported_block_marker(right_text)
    right = right_content.lstrip(" \t\r\n　")
    if not left or not right:
        return False
    if right.startswith(("「", "『", "#")) or _looks_like_list_item(right):
        return False

    if reopening_existing_quote:
        # Normal image OCR keeps the long-standing conservative recovery for
        # an unmistakable noun/指示词 + particle split (``ここ」 / は…``).
        # The more aggressive cases below are isolated to selectable-PDF mode.
        strict_particles = ("から", "まで", "より", "って", "は", "が", "を", "に", "へ", "と", "で", "も", "の", "や")
        noun_like_left = left.endswith((
            "ここ", "そこ", "あそこ", "これ", "それ", "あれ", "こと", "もの", "ところ"
        ))
        if noun_like_left and right.startswith(strict_particles):
            return True
        if not pdf_text_mode:
            return False

        # Vertical PDF glyph extraction can attach a close quote to the wrong
        # physical column. A later real quote, split long mark/iteration mark,
        # or polite auxiliary are strong PDF-specific reopening signals.
        if right.endswith("」") and "「" not in right:
            return True
        if right.startswith(("、", "。", "！", "？", "!", "?", "ー", "々")):
            return True
        if (left[-1], right[0]) in _SPLIT_COMPOUND_BOUNDARIES:
            return True
        if left.endswith("これ") and right.startswith("っぽっち"):
            return True
        if left.endswith(("てお", "でお")) and right.startswith(("ります", "りません", "りました")):
            return True
        if "\u30a0" <= left[-1] <= "\u30ff" and right.startswith("ー"):
            return True
        noun_like_left = (
            left and ("\u3400" <= left[-1] <= "\u9fff" or "\u30a0" <= left[-1] <= "\u30ff")
        )
        return noun_like_left and right.startswith(strict_particles)

    # 未闭合对白后面若出现“不带开引号、但带闭引号”的块，几乎可以确定是
    # 同一对白的尾列，例如 ``「どうしたの？`` + ``しっかりして」``。
    if right.endswith("」") and "「" not in right:
        return True

    if (left[-1], right[0]) in _SPLIT_COMPOUND_BOUNDARIES:
        return True

    # 右块直接从助词/辅助表达开始，是最可靠的跨列信号。
    if right.startswith(_DIALOGUE_CONTINUATION_PREFIXES):
        return True

    # 只凭左块末尾接续时，右块必须是很短的残片。这样
    # ``はわたしに任せて / 逃げるのだ`` 能接回，而
    # ``思わないの / 実際に俺は…`` 不会把完整叙述吞进去。
    if (
        _DIALOGUE_STRONG_LEFT_CONTINUATION_RE.search(left)
        and (
            len(right) <= _DIALOGUE_SHORT_CONTINUATION_MAX
            or not _has_strong_sentence_ending(right)
        )
    ):
        return True
    return False


def _dialogue_candidate(text: str) -> tuple[str, str, str, bool] | None:
    """解析可修复的块。

    返回 ``(prefix, body_without_outer_quotes, trailing_ws, had_closing_quote)``。
    既接受真正漏」的块，也接受旧版错误提前补成 ``ここ」`` 的块，后者只有在
    下一块确认是续句时才会撤销闭引号。
    """
    original = text or ""
    prefix, content = _strip_imported_block_marker(original)
    right_ws_len = len(content) - len(content.rstrip())
    trailing_ws = content[len(content) - right_ws_len:] if right_ws_len else ""
    stripped = content[:-right_ws_len] if right_ws_len else content
    if not stripped.startswith("「"):
        return None
    if "『" in stripped or "』" in stripped:
        return None
    if stripped.count("「") != 1 or stripped.count("」") not in {0, 1}:
        return None
    had_closing = stripped.endswith("」") and stripped.count("」") == 1
    if stripped.count("」") == 1 and not had_closing:
        return None
    body = stripped[1:-1] if had_closing else stripped[1:]
    return prefix, body, trailing_ws, had_closing


def repair_short_dialogue_closing_quotes(doc: UnifiedDocument) -> UnifiedDocument:
    """修复短对白缺失闭引号，并先恢复高置信度的跨块对白续句。

    例一，独立短对白：
        ``「俺はもっと強くあらねばならない`` + ``それが導き出した結論だ。``
        → ``「俺はもっと強くあらねばならない」`` + 独白保持独立。

    例二，跨列对白：
        ``「待て！…ここ`` + ``はわたしに任せて逃げるのだ`` + ``男装の少女は…``
        → ``「待て！…ここはわたしに任せて逃げるのだ」`` + 叙述保持独立。

    约束：
        - 只处理相邻 PARAGRAPH / DIALOGUE 块；
        - 只有助词、接续词、词中断裂等高置信度信号才合并下一块；
        - 不把普通日文段落仅因“看起来像日文”就吞进对白；
        - 已由旧版错误提前补上的」也可在确认续句后撤销并重新放到正确位置。
    """
    doc = copy.deepcopy(doc)
    pdf_text_mode = is_pdf_text_layer_mode_enabled(doc)
    repaired = 0
    merged = 0
    result: list[Block] = []
    i = 0

    while i < len(doc.blocks):
        block = doc.blocks[i]
        if block.type not in {BlockType.PARAGRAPH, BlockType.DIALOGUE}:
            result.append(block)
            i += 1
            continue

        candidate = _dialogue_candidate(block.text or "")
        if candidate is None:
            result.append(block)
            i += 1
            continue

        prefix, body, trailing_ws, had_closing = candidate
        original_block_text = block.text or ""
        merged_any = False
        consumed_explicit_closing = False
        consumed_texts: list[str] = []
        source_ids = [block.id or str(id(block))]

        # 只有下一块确实延续当前语法时才消费。这样 standalone 短对白不会吞掉
        # 后面的独白；``ここ」`` 这种旧版误补也只有在此处才撤销。
        while i + 1 < len(doc.blocks):
            next_block = doc.blocks[i + 1]
            if next_block.type not in {BlockType.PARAGRAPH, BlockType.DIALOGUE}:
                break
            if _looks_like_title(next_block, next_block.text or ""):
                break
            if not _looks_like_dialogue_continuation(
                body,
                next_block.text or "",
                reopening_existing_quote=had_closing and not merged_any,
                pdf_text_mode=pdf_text_mode,
            ):
                break
            _, next_content = _strip_imported_block_marker(next_block.text or "")
            next_piece = next_content.strip(" \t\r\n　")
            if not next_piece:
                break
            # 续接尾块本身可能带着真正的闭引号。把它作为结构符号消费，
            # 不要拼进 body 后再额外补一个」，否则会产生 ``」」`` 或孤立」块。
            if next_piece.endswith("」") and "「" not in next_piece:
                next_piece = next_piece[:-1].rstrip(" \t\r\n　")
                consumed_explicit_closing = True
            body = body.rstrip(" \t\r\n　") + next_piece
            consumed_texts.append(next_block.text or "")
            source_ids.append(next_block.id or str(id(next_block)))
            i += 1
            merged += 1
            merged_any = True
            if consumed_explicit_closing:
                break
            # 已经形成语法完整的对白后，下一块若没有续句信号，循环自然停止。

        body_stripped = body.strip()
        max_chars = MERGED_DIALOGUE_MAX_CHARS if merged_any else SHORT_DIALOGUE_MAX_CHARS
        # 导入的 Markdown/OCR 里，一条完整对白可能远长于“短对白”阈值。
        # 只要后面还有一个已判定为非续行的新块，就允许在明确边界处补引号。
        if (
            i + 1 < len(doc.blocks)
            and (
                any(ch in body_stripped for ch in "。！？!?……、，,")
                or len(set(body_stripped)) >= 6
            )
        ):
            max_chars = max(max_chars, 260)
        next_starts_new_dialogue = False
        if i + 1 < len(doc.blocks):
            _, remaining_content = _strip_imported_block_marker(doc.blocks[i + 1].text or "")
            next_starts_new_dialogue = remaining_content.lstrip(" \t\r\n　").startswith(("「", "『"))
        should_close = (
            bool(body_stripped)
            and len(body_stripped) <= max_chars
            and "\n" not in body_stripped
            and "\r" not in body_stripped
            and (
                consumed_explicit_closing
                or not _DIALOGUE_INCOMPLETE_END_RE.search(body_stripped)
                # ``…戻るから / 「え？」`` 这类口语句末的 から 已经由下一条
                # 新对白明确划出边界；若前面确实合并过续行，应在这里闭合。
                or next_starts_new_dialogue
            )
        )

        # 原本已有正确闭引号、也没有发生续接时无需改动。
        if had_closing and not merged_any:
            result.append(block)
            i += 1
            continue

        if should_close:
            new_block = copy.copy(block)
            new_block.ocr_raw = new_block.ocr_raw or original_block_text
            new_block.text = prefix + "「" + body_stripped + "」" + trailing_ws
            new_block.type = BlockType.DIALOGUE
            new_block.modified_by = _append_modified_by(
                new_block.modified_by, "repair_short_dialogue_closing_quotes"
            )
            if merged_any:
                new_block.metadata = {
                    **(new_block.metadata or {}),
                    "dialogue_continuation_merged": True,
                    "source_block_ids": source_ids,
                    "consumed_texts": consumed_texts,
                }
            result.append(new_block)
            repaired += 1
        else:
            # 无法确定闭引号位置时不擅自补符号。若已合并了高置信度续句，则保留
            # 合并结果但仍让引号保持未闭合，供人工或 AI 检查；绝不继续吞叙述。
            if merged_any:
                new_block = copy.copy(block)
                new_block.ocr_raw = new_block.ocr_raw or original_block_text
                new_block.text = prefix + "「" + body_stripped + trailing_ws
                new_block.type = BlockType.DIALOGUE
                new_block.modified_by = _append_modified_by(
                    new_block.modified_by, "merge_dialogue_continuation"
                )
                new_block.metadata = {
                    **(new_block.metadata or {}),
                    "dialogue_continuation_merged": True,
                    "quote_needs_review": True,
                    "dialogue_auto_merge_blocked": True,
                    "source_block_ids": source_ids,
                    "consumed_texts": consumed_texts,
                }
                result.append(new_block)
            else:
                # 高置信度规则仍无法确定闭引号位置时，保留原文并明确标记
                # “待检查”。后续 merge_broken_sentences 不得再利用未配对引号
                # 自动吞并下一段，避免把不确定问题扩大成正文粘连/丢失。
                new_block = copy.copy(block)
                new_block.type = BlockType.DIALOGUE
                new_block.metadata = {
                    **(new_block.metadata or {}),
                    "quote_needs_review": True,
                    "dialogue_auto_merge_blocked": True,
                }
                result.append(new_block)

        i += 1

    doc.blocks = result
    doc.add_log(
        "repair_dialogue_quotes",
        f"补全 {repaired} 处对白闭引号，续接 {merged} 个跨块对白残片",
        repaired + merged,
    )
    return doc


# ── 步骤 6：对白独立换行（迭代拆分）─────────────────────────────────────────



def _is_dialogue_start(text: str, start: int) -> bool:
    if start == 0:
        return True
    prefix = text[:start].rstrip()
    if not prefix:
        return True
    return prefix[-1] in "。！？!?\n"


def restore_dialogue_breaks(doc: UnifiedDocument) -> UnifiedDocument:
    """
    拆分 PARAGRAPH/DIALOGUE 块中连续的「...」人物对白。

    只拆「」对白，不拆『』术语引用；普通段落里的句中术语引用会保留原样。
    """
    doc = copy.deepcopy(doc)
    dialogue_re = re.compile(r"「[^」]*」", re.DOTALL)
    split_count = 0
    result: list[Block] = []

    for block in doc.blocks:
        if block.type not in {BlockType.PARAGRAPH, BlockType.DIALOGUE}:
            result.append(block)
            continue

        text = block.text or ""
        # 一个块本身就是完整的「…」对白时，也必须标记为 DIALOGUE。
        # 旧实现只有在同一块还带叙述或第二条对白时才拆分；单独一条对白会
        # 继续保持 PARAGRAPH，随后被加上正文缩进，EPUB 中看起来就不像
        # 独立对白列。嵌套的『术语』仍保留在同一条对白内部。
        whole_dialogue = re.fullmatch(r"\s*「[^」]*」\s*", text, re.DOTALL)
        if whole_dialogue:
            dialogue = copy.deepcopy(block)
            dialogue.text = text.strip()
            dialogue.type = BlockType.DIALOGUE
            dialogue.modified_by = _append_modified_by(
                dialogue.modified_by, "restore_dialogue_breaks"
            )
            result.append(dialogue)
            if block.type != BlockType.DIALOGUE or dialogue.text != text:
                split_count += 1
            continue

        matches = list(dialogue_re.finditer(text))
        if not matches:
            result.append(block)
            continue

        pieces: list[Block] = []
        cursor = 0
        for match in matches:
            # 对白不仅会出现在块首或上一条对白之后，也常直接跟在完整叙述句后：
            #   後はこの街の市長に事情を伝え…簡単なお仕事だ。「え……ええ？」
            # 句号/问号/感叹号已经明确结束叙述，因此后面的完整「…」应独立成块。
            prev = text[match.start() - 1] if match.start() > 0 else ""
            prefix = text[:match.start()].strip()
            # 连续对白只允许前缀本身完全由一个或多个「…」组成。
            # 不能仅凭前一个字符是『/』就拆分，否则
            #   『怒り』「憎しみ」「蔑み」『恐怖』
            # 这类句中列举会被误当人物对白，整句被切碎。
            dialogue_chain = bool(prefix) and re.fullmatch(r'(?:「[^」]*」\s*)+', prefix, re.DOTALL) is not None
            valid = (
                block.type == BlockType.DIALOGUE
                or _is_dialogue_start(text, match.start())
                or prev in "。！？!?"
                or dialogue_chain
            )
            if not valid:
                continue

            before = text[cursor:match.start()].strip()
            if before:
                paragraph = copy.deepcopy(block)
                paragraph.id = uuid.uuid4().hex
                paragraph.metadata = {**(paragraph.metadata or {}), "formatter_source_block_ids": [str(block.id)]}
                paragraph.text = before
                paragraph.type = BlockType.PARAGRAPH
                paragraph.modified_by = "restore_dialogue_breaks"
                pieces.append(paragraph)

            dialogue = copy.deepcopy(block)
            dialogue.id = uuid.uuid4().hex
            dialogue.metadata = {**(dialogue.metadata or {}), "formatter_source_block_ids": [str(block.id)]}
            dialogue.text = match.group(0).strip()
            dialogue.type = BlockType.DIALOGUE
            dialogue.modified_by = "restore_dialogue_breaks"
            pieces.append(dialogue)
            cursor = match.end()
            split_count += 1

        tail = text[cursor:].strip()
        if tail:
            paragraph = copy.deepcopy(block)
            paragraph.id = uuid.uuid4().hex
            paragraph.metadata = {**(paragraph.metadata or {}), "formatter_source_block_ids": [str(block.id)]}
            paragraph.text = tail
            paragraph.type = BlockType.PARAGRAPH
            paragraph.modified_by = "restore_dialogue_breaks"
            pieces.append(paragraph)

        if len(pieces) >= 2:
            result.extend(pieces)
        else:
            result.append(block)

    doc.blocks = result
    doc.add_log("dialogue_restore", f"分离 {split_count} 条对白", split_count)
    return doc


# ── 步骤 7：缩进和分节符恢复 ─────────────────────────────────────────────────

def restore_indents_and_breaks(doc: UnifiedDocument) -> UnifiedDocument:
    """
    恢复段落缩进（全角空格）和分节符检测。
    - PARAGRAPH 开头添加全角空格缩进
    - DIALOGUE 不添加缩进
    - 检测分节符号行（◆※☆★●○＊等）→ SECTION 类型
    """
    doc = copy.deepcopy(doc)

    indent_count = 0
    section_count = 0

    for b in doc.blocks:
        if b.type == BlockType.PARAGRAPH:
            t = b.text.strip()
            # 检测分节符
            if SECTION_RE.match(t):
                b.type = BlockType.SECTION
                b.modified_by = "restore_indents_and_breaks"
                section_count += 1
                continue
            # 添加缩进
            if not t.startswith('　') and not t.startswith(' '):
                b.text = '　' + t
                b.modified_by = "restore_indents_and_breaks"
                indent_count += 1

    doc.add_log("restore_indents", f"缩进 {indent_count} 段，分节符 {section_count} 个",
                indent_count + section_count)
    return doc


# ── 步骤 8：振假名恢复 ───────────────────────────────────────────────────────

def recover_ruby(doc: UnifiedDocument) -> UnifiedDocument:
    """
    检测并恢复振假名标注。
    ｜漢字《よみ》 → 内部标记 漢字|よみ
    同时将含有 ruby 标注的 block 标记为 RUBY 类型。
    """
    doc = copy.deepcopy(doc)

    ruby_count = 0
    TEXT_TYPES = {BlockType.PARAGRAPH, BlockType.DIALOGUE}

    for b in doc.blocks:
        if b.type not in TEXT_TYPES:
            continue

        matches = list(RUBY_RE.finditer(b.text))
        if not matches:
            continue

        if not b.ocr_raw:
            b.ocr_raw = b.text

        new_text = b.text
        for m in reversed(matches):
            base = m.group(1)
            reading = m.group(2)
            new_text = new_text[:m.start()] + f"{base}|{reading}" + new_text[m.end():]
            ruby_count += 1

        b.text = new_text
        b.type = BlockType.RUBY
        b.modified_by = "recover_ruby"

    doc.add_log("recover_ruby", f"恢复 {ruby_count} 处振假名", ruby_count)
    return doc


# ── 步骤 9：章节识别（正规化匹配）──────────────────────────────────────────

def detect_chapters(doc: UnifiedDocument) -> UnifiedDocument:
    """
    重新扫描所有 PARAGRAPH 块，将匹配章节正则的升级为 CHAPTER/SECTION 块。
    正规化处理：匹配前去除多余空格。

    印刷版目录页本身就会把"第一章""第三章""第四章"……一整页排在一起——
    如果目录页没有在 Page Manager 里被手动标成"目录"跳过 OCR（详见
    apple_vision_adapter 的"只有正文标签的页才 OCR"），它的文字会被当成
    普通正文识别出来，而其中每一行列出的章节名单独看都能匹配章节正则，
    照单全收的话就会把目录列表里的每一条都错误地提升成一个"新章节"。
    真正的章节标题不会出现"同一页里有两个不同的章节名"这种情况——
    一个物理页最多对应一个章节开头。所以先扫一遍统计每页有几个不同
    （去重后）的候选标题，同一页出现 ≥2 个不同候选时，整页都当作疑似
    目录/索引页处理，不提升为章节（留作普通段落）。
    """
    doc = copy.deepcopy(doc)

    suspect_pages = _detect_toc_like_pages(doc)

    # 网络轻小说 PDF（"小説家になろう"/"タテ書き小説ネット"这类站点导出）
    # 常用裸数字当每话的序号标题（"００１""００２"……），不是"第一章"
    # 这种格式，CHAPTER_RE 匹配不到。这类裸数字块在 merge_broken_sentences
    # 里已经被当结构性标记保护起来、不会跟正文粘连了，这里只在 PDF 文字层
    # 直读来源上把它们也当章节候选——OCR 来源的裸数字块在更早的
    # clean_metadata 步骤就已经当页码删掉了，不会有残留，不需要处理。
    recognize_bare_numbers = doc.metadata.source_engine == "pdf_text_layer"

    candidates: list[tuple[int, int, str, str]] = []  # (block_idx, page, normalized, original)
    for i, b in enumerate(doc.blocks):
        if b.type not in (BlockType.PARAGRAPH, BlockType.CHAPTER):
            continue
        t = b.text.strip()
        if not t:
            continue
        normalized = re.sub(r'[\s　]+', '', t)
        if b.type == BlockType.CHAPTER:
            # 已经是 CHAPTER 类型的块（比如 EPUB/DOCX 导入时按 <h1> 或原有格式
            # 标记出来的）本身就是权威的章节标题，不管文字长不长得像
            # CHAPTER_RE 都要保留、重新纳入 toc——否则重新跑一遍 detect_chapters
            # 会把"００１"这种不匹配任何章节正则、但本来就是真章节标题的书
            # （网络小说站常见的裸数字分话）目录冲没大半，只剩下少数碰巧匹配
            # 正则的标题（比如"プロローグ"）。
            candidates.append((i, b.page, normalized, t))
        elif CHAPTER_RE.match(normalized) or CHAPTER_RE.match(t):
            candidates.append((i, b.page, normalized, t))
        elif recognize_bare_numbers and BARE_NUMBER_RE.match(t):
            candidates.append((i, b.page, normalized, t))

    doc.toc = []
    chapter_index = 0
    suppressed = 0

    for i, page, normalized, t in candidates:
        if page in suspect_pages:
            suppressed += 1
            # OCR 适配器在识别阶段会按同样的正则把"看起来像章节标题"的行
            # 直接标成 CHAPTER（见 apple_vision_adapter.lines_to_blocks）。
            # 这里判定为目录/索引页后，如果那一行已经带着这个先入为主的
            # CHAPTER 类型，必须显式降级回 PARAGRAPH，否则只是跳过"提升"
            # 这一步，它原有的 CHAPTER 类型不会被清掉，疑似目录页里的每一条
            # 依然会被当成真正的章节输出到 EPUB 里。
            if doc.blocks[i].type == BlockType.CHAPTER:
                doc.blocks[i].type = BlockType.PARAGRAPH
                doc.blocks[i].modified_by = "detect_chapters"
            continue

        b = doc.blocks[i]
        b.type = BlockType.CHAPTER
        b.modified_by = "detect_chapters"
        chapter_index += 1
        b.chapter_index = chapter_index

        doc.toc.append(TocEntry(
            title=t,
            chapter_index=chapter_index,
            block_index=i,
        ))

    msg = f"识别 {chapter_index} 个章节"
    if suppressed:
        msg += f"，疑似目录/索引页跳过 {suppressed} 处（{len(suspect_pages)} 页）"
    doc.add_log("detect_chapters", msg, chapter_index)
    return doc


# ── 步骤 9.5：剥离前书/后书样板文字 ────────────────────────────────────────────

# 半角/全角罗马字母、数字、URL 常见符号——竖排阅读顺序恢复失败产生的乱码
# 段落会把 URL 片段、日期戳跟汉字假名字符级搅在一起，这类符号占比会明显偏高。
# 全角形式（Ａ-Ｚ０-９）也要算，PDF 原文里日期/编号常用全角数字，乱序后
# 一样会跟汉字混在一起，只统计半角会严重低估。
_GARBLED_CHAR_RE = re.compile(r'[A-Za-z0-9Ａ-Ｚａ-ｚ０-９/:.\-]')


def _looks_garbled(text: str) -> bool:
    """
    粗略判断一段文字是不是"看起来像乱码"——本身不含 KNOWN_BOILERPLATE_RE 的
    关键词，但罗马字母/数字/URL符号的占比明显偏高，不像正常对白/叙述那样
    成句。正常日文正文里角色名音译、数字台词也会带一点半角字符，所以阈值
    不能设太低，只挑"明显不正常"的（乱码段实测占比 0.4+，正常正文接近 0）。
    """
    t = text.strip()
    if not t or '「' in t or '」' in t:
        return False
    noise = len(_GARBLED_CHAR_RE.findall(t))
    return len(t) >= 6 and noise / len(t) > 0.2


def strip_boilerplate_matter(doc: UnifiedDocument) -> UnifiedDocument:
    """
    删除"前书/后书"——即不属于任何真正章节标题、也不属于正文的网站样板文字
    （版权声明、转载须知、站点介绍等）。必须跑在 detect_chapters 之后，因为要
    依赖它产出的 doc.toc 来定位"第一个真正章节之前"和"最后内容"的边界。

    前书：doc.toc[0].block_index 之前的所有块。只有这个区间内命中
    KNOWN_BOILERPLATE_RE 才整体删除——真正的作者手写楔子/前言不会命中，不受影响。
    区间内的 IMAGE_REF 块（扫描的扉页/目录页/插图等原图页面）不算样板文字，
    原样保留——否则会连带把真正的扫描目录页图片一起删掉，EPUB 里就再也看不到
    原书的目录扫描页了，nav.xhtml 也该跟在它后面而不是把它一并抹掉。

    后书：从文档末尾往前找到第一个命中 KNOWN_BOILERPLATE_RE 的块，把它自己以及
    它之后的所有块都删掉；再往前多吸收几个"看起来乱码"的相邻块（同一次乱序
    恢复失败通常连续影响好几段），直到遇到一段不乱码的块为止，避免正常正文
    被误伤。同样保留这个区间里的 IMAGE_REF 块（比如版权页扫描图）。

    只在 pdf_text_layer（PDF 文字层直读）和 epub_import（EPUB 逆向导入）这两个
    来源上生效——这两条路径的文字都是从原始文本里精确解析出来的，"乱码"是
    竖排阅读顺序恢复失败的产物；OCR 来源（Apple Vision/PaddleOCR）的噪声是
    另一种性质（认错字，不是整段乱序夹带站点样板），不该套用同一套规则去删，
    所以 OCR 来源直接跳过，不做任何删除，保证不影响 OCR 流程。
    """
    doc = copy.deepcopy(doc)

    if doc.metadata.source_engine not in ("pdf_text_layer", "epub_import"):
        doc.add_log("strip_boilerplate_matter", "跳过（仅对 PDF 文字层直读/EPUB 导入来源生效）", 0)
        return doc

    removed = 0

    # 用第一个真正的 CHAPTER 类型块定位"前书"边界，而不是 doc.toc[0]——EPUB
    # 逆向导入时前书桶自己也会占一条 toc 记录（对应 nav.xhtml 里的"前书页"），
    # 这种情况下 toc[0] 就是前书本身，不能拿来当"第一章在哪"的判断依据。
    first_chapter_idx = next(
        (i for i, b in enumerate(doc.blocks) if b.type == BlockType.CHAPTER), None
    )
    if first_chapter_idx:
        front_span = doc.blocks[:first_chapter_idx]
        if any(KNOWN_BOILERPLATE_RE.search(b.text) for b in front_span):
            kept_front = [b for b in front_span if b.type == BlockType.IMAGE_REF]
            shift = len(front_span) - len(kept_front)
            removed += shift
            doc.blocks = kept_front + doc.blocks[first_chapter_idx:]
            new_first_idx = len(kept_front)
            # 指向被删范围内的 toc 条目（比如"前书页"这种伪目录项）要去掉，
            # 剩下的条目按实际发生的位移量（不是原来的 first_chapter_idx，
            # 因为保留下来的图片页占了几个位置）调整 block_index。
            doc.toc = [t for t in doc.toc if t.block_index >= first_chapter_idx]
            for entry in doc.toc:
                entry.block_index += new_first_idx - first_chapter_idx

    cut_from = None
    for i in range(len(doc.blocks) - 1, -1, -1):
        b = doc.blocks[i]
        if b.type == BlockType.IMAGE_REF:
            # 图片页（比如版权页扫描图）不参与乱码/签名判定，跳过继续往前找，
            # 不能因为文字是空的就当"正常正文"提前打断扫描。
            continue
        if KNOWN_BOILERPLATE_RE.search(b.text):
            cut_from = i
            break
        if not _looks_garbled(b.text):
            break

    if cut_from is not None:
        # 命中签名的那一块本身也算后书的一部分，一并删除；再往前吸收连续的
        # 相邻块——不管是也命中签名（同一段样板文字被切成好几段）还是看起来
        # 乱码（同一次乱序恢复失败通常连续影响好几段），直到遇到一段正常
        # 正文为止，不会跳着往前扫。
        while cut_from > 0 and (
            doc.blocks[cut_from - 1].type == BlockType.IMAGE_REF
            or _looks_garbled(doc.blocks[cut_from - 1].text)
            or KNOWN_BOILERPLATE_RE.search(doc.blocks[cut_from - 1].text)
        ):
            cut_from -= 1
        back_span = doc.blocks[cut_from:]
        kept_back = [b for b in back_span if b.type == BlockType.IMAGE_REF]
        removed += len(back_span) - len(kept_back)
        doc.blocks = doc.blocks[:cut_from] + kept_back

    doc.add_log("strip_boilerplate_matter", f"删除 {removed} 个前书/后书样板块", removed)
    return doc


# ── 步骤 9.8：清理孤立闭引号 ─────────────────────────────────────────────────

def remove_orphan_closing_quotes(doc: UnifiedDocument) -> UnifiedDocument:
    """删除被 OCR/AI 拆成独立文本块的闭引号。

    单独一行的 ``」``/``』`` 不承载正文。若前一文本块存在未闭合的对应
    开引号，就把闭引号接回前块；否则直接删除。EPUB Builder 还会做一次
    最终防御过滤，保证旧工程直接导出时也不会重新出现。
    """
    doc = copy.deepcopy(doc)
    pairs = {"」": "「", "』": "『"}
    text_types = {BlockType.PARAGRAPH, BlockType.DIALOGUE, BlockType.RUBY}
    result: list[Block] = []
    repaired = 0
    removed = 0

    for block in doc.blocks:
        token = (block.text or "").strip()
        if block.type in text_types and token in pairs:
            opener = pairs[token]
            previous = next((b for b in reversed(result) if b.type in text_types and (b.text or "").strip()), None)
            if previous is not None:
                previous_text = previous.text or ""
                if previous_text.count(opener) > previous_text.count(token):
                    previous.text = previous_text.rstrip() + token
                    previous.type = BlockType.DIALOGUE if opener == "「" else previous.type
                    previous.modified_by = "remove_orphan_closing_quotes"
                    repaired += 1
                    continue
            removed += 1
            continue
        result.append(block)

    doc.blocks = result
    doc.add_log(
        "remove_orphan_closing_quotes",
        f"接回 {repaired} 个孤立闭引号，删除 {removed} 个无归属闭引号",
        repaired + removed,
    )
    return doc


# ── 步骤 10：标点规范化 ───────────────────────────────────────────────────────

def normalize_punctuation(doc: UnifiedDocument) -> UnifiedDocument:
    """统一标点格式 + OCR 常见错字修正"""
    OCR_TYPOS = [
        (re.compile(r'現れたてから'), '現れてから'),
        (re.compile(r'消えたてから'), '消えてから'),
        (re.compile(r'出たてきた'),   '出てきた'),
        # フ/プ（半浊点丢失）是竖排 PDF 字符提取常见的错读，"フロローグ"
        # 不是真实存在的日语词，只可能是"プロローグ"半浊点丢失的产物。
        (re.compile(r'フロローグ'),   'プロローグ'),
    ]

    from engine.char_normalizer import normalize_ocr_codepoints

    doc = copy.deepcopy(doc)
    fixed_count = 0
    codepoint_counts: dict[str, int] = {}

    text_types = {BlockType.PARAGRAPH, BlockType.DIALOGUE,
                  BlockType.CHAPTER, BlockType.SECTION, BlockType.RUBY}

    for b in doc.blocks:
        if b.type not in text_types:
            continue

        original = b.text

        # 码位级规范化（NFD 假名合成/半角片假名/康熙部首/控制符/连字符变体）
        b.text, cp_counts = normalize_ocr_codepoints(b.text)
        for key, n in cp_counts.items():
            codepoint_counts[key] = codepoint_counts.get(key, 0) + n

        for pattern, repl in PUNCT_RULES:
            b.text = pattern.sub(repl, b.text)

        for pattern, repl in OCR_TYPOS:
            b.text = pattern.sub(repl, b.text)

        # 章节标识与标题之间保持一个空格。
        # 例如：プロローグ序章 -> プロローグ 序章
        #      第一章魔王 -> 第一章 魔王
        if b.type in {BlockType.CHAPTER, BlockType.SECTION}:
            b.text = CHAPTER_TITLE_SPACE_RE.sub(r'\1 ', b.text)

        if b.text != original:
            b.ocr_raw = b.ocr_raw or original
            b.modified_by = "normalize_punctuation"
            fixed_count += 1

    cp_note = ""
    if codepoint_counts:
        labels = {"nfd_kana": "NFD假名合成", "halfwidth_kana": "半角片假名",
                  "kangxi_radical": "康熙部首", "control_char": "控制符",
                  "dash_variant": "连字符变体"}
        cp_note = "；码位修正：" + "、".join(
            f"{labels.get(k, k)}×{v}" for k, v in sorted(codepoint_counts.items()))
    doc.add_log("normalize_punctuation", f"修正 {fixed_count} 处标点/错字{cp_note}", fixed_count)
    return doc


# ── Pipeline：组合所有步骤 ────────────────────────────────────────────────────

PRESERVE_OCR_LAYOUT_SKIP_STEPS = {
    # 这些步骤会重排、合并、删除、拆分或缩进块。开启固定原 OCR 排版时跳过，
    # 让导入或 OCR 产生的原始块、段落及分页结构保持一致。
    "reading_order",
    "clean_metadata",
    "split_embedded_titles",
    "strip_chapter_notes",
    "merge_overlaps",
    "merge_sentences",
    "remove_duplicates",
    "dialogue_restore",
    "restore_indents",
    "strip_boilerplate",
}

def is_preserve_ocr_layout_enabled(doc: UnifiedDocument) -> bool:
    """是否启用「固定原 OCR 排版」模式。"""
    return bool(getattr(doc.metadata, "preserve_ocr_layout", False))


def is_pdf_text_layer_mode_enabled(doc: UnifiedDocument) -> bool:
    """Whether the isolated selectable-PDF Formatter profile is enabled."""
    return bool(getattr(doc.metadata, "pdf_text_layer_mode", False))


def _safe_toc_index(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return -1


def _sync_toc_after_formatter_step(before: UnifiedDocument, after: UnifiedDocument) -> None:
    """Keep TOC indices/titles aligned after every formatter transformation.

    Most formatter steps deepcopy the previous TOC while inserting, merging or
    deleting blocks.  For those inherited entries, the stable pre-step block ID
    is the safest anchor.  ``detect_chapters`` is different: it deliberately
    rebuilds the TOC.  If that newly produced TOC already points to matching
    title blocks, its own indices must win over a stale/reordered legacy TOC from
    ``before``; otherwise the generic preservation pass could undo the rebuild.
    """
    title_types = {BlockType.CHAPTER, BlockType.SECTION}
    before_toc = list(getattr(before, "toc", []) or [])
    after_toc = list(getattr(after, "toc", []) or [])

    before_ids_by_position: list[str] = []
    before_ids_by_chapter: dict[int, list[str]] = {}
    before_ids_by_title: dict[str, list[str]] = {}
    for toc in before_toc:
        old_index = _safe_toc_index(getattr(toc, "block_index", -1))
        block_id = ""
        if 0 <= old_index < len(before.blocks):
            old_block = before.blocks[old_index]
            if old_block.type in title_types:
                block_id = str(old_block.id)
                before_ids_by_chapter.setdefault(
                    max(0, _safe_toc_index(getattr(toc, "chapter_index", 0))), []
                ).append(block_id)
                key = str(getattr(toc, "title", "") or "").strip()
                if key:
                    before_ids_by_title.setdefault(key, []).append(block_id)
        before_ids_by_position.append(block_id)

    def _toc_signature(entries) -> list[tuple[int, str, int]]:
        return [
            (
                max(0, _safe_toc_index(getattr(item, "chapter_index", 0))),
                str(getattr(item, "title", "") or "").strip(),
                _safe_toc_index(getattr(item, "block_index", -1)),
            )
            for item in entries
        ]

    def _current_target(item) -> int | None:
        current_index = _safe_toc_index(getattr(item, "block_index", -1))
        if not (0 <= current_index < len(after.blocks)):
            return None
        candidate = after.blocks[current_index]
        if candidate.type not in title_types:
            return None
        chapter_index = max(0, _safe_toc_index(getattr(item, "chapter_index", 0)))
        candidate_chapter = max(0, _safe_toc_index(getattr(candidate, "chapter_index", 0)))
        if chapter_index and candidate_chapter and candidate_chapter != chapter_index:
            return None
        return current_index

    # A deliberately rebuilt TOC differs from the inherited raw signature and
    # is considered authoritative only when every entry is already internally
    # consistent (correct title text and unique valid title-block index).
    current_targets = [_current_target(item) for item in after_toc]
    prefer_current_toc = bool(after_toc) and _toc_signature(before_toc) != _toc_signature(after_toc)
    if prefer_current_toc:
        prefer_current_toc = (
            all(index is not None for index in current_targets)
            and len(set(current_targets)) == len(current_targets)
            and all(
                str(after.blocks[index].text or "").strip()
                == str(getattr(item, "title", "") or "").strip()
                for item, index in zip(after_toc, current_targets)
                if index is not None
            )
        )

    index_by_id = {str(block.id): index for index, block in enumerate(after.blocks)}
    title_indices = [
        index for index, block in enumerate(after.blocks)
        if block.type in title_types and str(block.text or "").strip()
    ]
    used: set[int] = set()

    for position, toc in enumerate(after_toc):
        chapter_index = max(0, _safe_toc_index(getattr(toc, "chapter_index", 0)))
        source_id = before_ids_by_position[position] if position < len(before_ids_by_position) else ""
        current_index = current_targets[position]

        resolved = current_index if prefer_current_toc else None
        if resolved is None and source_id:
            candidate_index = index_by_id.get(source_id)
            if candidate_index is not None and candidate_index not in used:
                resolved = candidate_index
        if resolved is None and current_index is not None and current_index not in used:
            resolved = current_index

        if resolved is None and chapter_index:
            resolved = next((
                index_by_id[item]
                for item in before_ids_by_chapter.get(chapter_index, [])
                if item in index_by_id and index_by_id[item] not in used
            ), None)
        if resolved is None and chapter_index:
            resolved = next((
                index for index in title_indices
                if index not in used
                and max(0, _safe_toc_index(getattr(after.blocks[index], "chapter_index", 0))) == chapter_index
            ), None)

        if resolved is None:
            old_title = str(getattr(toc, "title", "") or "").strip()
            candidate_ids = before_ids_by_title.get(old_title, [])
            resolved = next((
                index_by_id[item]
                for item in candidate_ids
                if item in index_by_id and index_by_id[item] not in used
            ), None)

        if resolved is None:
            continue
        target = after.blocks[resolved]
        if target.type not in title_types:
            continue
        toc.block_index = resolved
        toc.title = str(target.text or "").strip()
        used.add(resolved)


PIPELINE_STEPS = [
    ("reading_order",       reading_order_step),
    ("ai_correction",       ai_correction_step),
    ("clean_metadata",      clean_metadata_blocks),
    ("split_embedded_titles", split_embedded_chapter_titles),
    ("column_sentence_reflow", reflow_columns_into_sentences),
    ("strip_chapter_notes", strip_chapter_notes),
    ("cross_page_merge",    merge_cross_page_sentences),
    ("merge_overlaps",      merge_overlapping_blocks),
    # 短对白闭引号必须先补，再做断句合并。否则一个块首未闭合的「会让
    # merge_broken_sentences 把后续整页正文都当成同一段对白吞进去。
    ("repair_dialogue_quotes", repair_short_dialogue_closing_quotes),
    ("merge_sentences",     merge_broken_sentences),
    ("remove_duplicates",   remove_duplicates),
    ("fix_dash_artifacts",  fix_ocr_dash_artifacts),
    ("dialogue_restore",    restore_dialogue_breaks),
    ("restore_indents",     restore_indents_and_breaks),
    ("recover_ruby",        recover_ruby),
    ("detect_chapters",     detect_chapters),
    ("remove_orphan_quotes", remove_orphan_closing_quotes),
    ("strip_boilerplate",   strip_boilerplate_matter),
    ("normalize_punct",     normalize_punctuation),
]


def run_pipeline(
    doc: UnifiedDocument,
    steps: list[str] | None = None,
    verbose: bool = True,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    repo_path: str | None = None,
) -> UnifiedDocument:
    """
    运行 Formatter Pipeline。

    每一步跑完都会 commit 进版本仓库（内容寻址存储，历史持久化在磁盘上，
    而不是堆在内存的一个 list 里）。仓库位置：
        - 如果传入的 doc 已经关联了仓库（比如是从之前的 commit 载入的），
          沿用同一个仓库，历史链条不会断。
        - 否则用 repo_path（未提供则自动分配一个临时目录）新建一个仓库，
          调用方不需要关心仓库路径也能获得完整的 Undo 历史。

    Args:
        doc:   输入的 UnifiedDocument
        steps: 要运行的步骤 id 列表（None = 全部按序运行）
        verbose: 打印每步日志
        progress_callback: 进度回调 (step_name, current_step, total_steps)
        repo_path: 版本仓库目录（None = 自动分配临时目录，或沿用 doc 已有仓库）

    Returns:
        处理后的新 UnifiedDocument（.repo/.commit_id 指向最新提交）
    """
    to_run = list(steps) if steps is not None else [s for s, _ in PIPELINE_STEPS]
    pdf_text_mode = is_pdf_text_layer_mode_enabled(doc)
    if pdf_text_mode and to_run:
        # Selectable-PDF text must first be put into reading order, then its
        # physical columns must be reconstructed *before* quote completion,
        # short-block cleanup, overlap merging, or deduplication.  The old
        # order could invent ``これ」/っぽっち`` and delete tails such as
        # ``た。`` / ``い。`` before they had a chance to rejoin the stem.
        to_run = [s for s in to_run if s not in {"pdf_text_prepare", "pdf_text_finalize"}]
        ordered: list[str] = []
        if "reading_order" in to_run:
            ordered.append("reading_order")
            to_run.remove("reading_order")
        ordered.append("pdf_text_prepare")
        ordered.extend(to_run)
        ordered.append("pdf_text_finalize")
        to_run = ordered

    # 依赖保护：断句合并会把块首未闭合的「视为跨块对白并持续吞并后文。
    # 因此只要请求 merge_sentences，就必须先运行短对白闭引号修复；不能依赖
    # GUI 是否恰好把该步骤列出来。旧 UI 的“全部运行”遗漏了这个步骤，最终会
    # 先把整页正文合进对白，再由 fix_dash_artifacts 把孤立「改成——。
    if "merge_sentences" in to_run:
        merge_index = to_run.index("merge_sentences")
        if "repair_dialogue_quotes" not in to_run:
            to_run.insert(merge_index, "repair_dialogue_quotes")
        elif to_run.index("repair_dialogue_quotes") > merge_index:
            to_run.remove("repair_dialogue_quotes")
            to_run.insert(merge_index, "repair_dialogue_quotes")

    step_map = {s: fn for s, fn in PIPELINE_STEPS}
    if pdf_text_mode:
        from engine.pdf_text_layer_formatter import (
            clean_pdf_text_metadata,
            finalize_pdf_text_layer,
            normalize_pdf_text_punctuation,
            prepare_pdf_text_layer,
            preserve_pdf_afterwords,
            preserve_pdf_boilerplate,
            preserve_pdf_orphan_quotes,
            remove_pdf_coordinate_duplicates,
            restore_pdf_indents,
            skip_pdf_cross_page_merge,
            skip_pdf_dialogue_auto_close,
            restore_pdf_dialogue_columns,
            skip_pdf_overlap_merge,
            skip_pdf_sentence_merge,
        )
        step_map.update({
            "pdf_text_prepare": prepare_pdf_text_layer,
            "pdf_text_finalize": finalize_pdf_text_layer,
            "clean_metadata": clean_pdf_text_metadata,
            "strip_chapter_notes": preserve_pdf_afterwords,
            "cross_page_merge": skip_pdf_cross_page_merge,
            "merge_overlaps": skip_pdf_overlap_merge,
            "repair_dialogue_quotes": skip_pdf_dialogue_auto_close,
            "merge_sentences": skip_pdf_sentence_merge,
            "remove_duplicates": remove_pdf_coordinate_duplicates,
            "dialogue_restore": restore_pdf_dialogue_columns,
            "restore_indents": restore_pdf_indents,
            "remove_orphan_quotes": preserve_pdf_orphan_quotes,
            "strip_boilerplate": preserve_pdf_boilerplate,
            "normalize_punct": normalize_pdf_text_punctuation,
        })
    total = len(to_run)

    effective_repo_path = repo_path
    if doc.repo is not None:
        effective_repo_path = str(doc.repo.path)
    elif effective_repo_path is None:
        effective_repo_path = new_temp_repo_path()

    current = doc
    for step_idx, step_id in enumerate(to_run):
        fn = step_map.get(step_id)
        if fn is None:
            print(f"  ⚠️  未知步骤: {step_id}")
            continue
        preserve_layout = is_preserve_ocr_layout_enabled(current)
        # 只有用户明确勾选“固定原 OCR 排版”时才启用保守处理；默认模式
        # 无论是否经过文本替换或 AI，都允许再次执行完整宽松 Formatter。
        if step_id == "cross_page_merge" and preserve_layout:
            fn = merge_cross_page_sentences_layout_safe
        preserve_layout_unsafe = (
            preserve_layout
            and step_id in PRESERVE_OCR_LAYOUT_SKIP_STEPS
        )
        if preserve_layout_unsafe:
            result = copy.deepcopy(current)
            message = "固定原 OCR 排版：跳过会改变段落结构的步骤"
            result.add_log(step_id, message, 0)
            result.repo = current.repo
            result.commit_id = current.commit_id
            last_log = result.processing_log[-1]
            commit_id = result.commit(effective_repo_path, step_id, last_log.get("message", ""))
            current = result
            if verbose:
                print(f"  ⏭  {step_id} ... [{commit_id[:8]}] {last_log.get('message', 'skipped')}")
            continue
        if verbose:
            print(f"  ▶  {step_id} ...", end=" ")

        if progress_callback:
            progress_callback(step_id, step_idx, total)

        result = fn(current)
        # Ruby is immutable side-channel evidence. Structural formatter steps
        # may split/merge/rebuild blocks, so carry it from the pre-step document
        # before committing the new version. Ambiguous mappings fail closed.
        try:
            from adapters.findtext_centernet_ruby import carry_ruby_overlay, has_ruby_overlay
            if has_ruby_overlay(current):
                carry_ruby_overlay(current, result)
        except Exception:
            # Optional Ruby preservation must never break the formatter path.
            pass
        _sync_toc_after_formatter_step(current, result)
        # 步骤函数内部 deepcopy 出的新对象不会带着上一版的 repo/commit_id，
        # 这里接续上，保证 commit 链条完整。
        result.repo = current.repo
        result.commit_id = current.commit_id

        last_log = result.processing_log[-1] if result.processing_log else {}
        commit_id = result.commit(effective_repo_path, step_id, last_log.get("message", ""))
        current = result

        if verbose:
            print(f"[{commit_id[:8]}] {last_log.get('message', 'done')}")

    if progress_callback:
        progress_callback("done", total, total)

    return current


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Novel Formatter Engine")
    parser.add_argument("input_json",  help="输入 UnifiedDocument JSON")
    parser.add_argument("output_json", help="输出 JSON 路径")
    parser.add_argument(
        "--preserve-ocr-layout", action="store_true",
        help="固定原 OCR 块/段落结构，跳过合并断句、对白拆分、缩进等重排步骤"
    )
    parser.add_argument(
        "--pdf-text-layer-mode", action="store_true",
        help="启用可选择PDF文字层专用规则；与普通图片OCR Formatter隔离"
    )
    parser.add_argument(
        "--steps", nargs="*",
        help="指定运行步骤（空=全部）：reading_order clean_metadata merge_sentences "
             "remove_duplicates fix_dash_artifacts dialogue_restore restore_indents "
             "recover_ruby detect_chapters normalize_punct"
    )
    args = parser.parse_args()

    with open(args.input_json, encoding="utf-8") as f:
        doc = UnifiedDocument.from_json(f.read())

    if args.pdf_text_layer_mode:
        doc.metadata.pdf_text_layer_mode = True
        doc.metadata.preserve_ocr_layout = False
    elif args.preserve_ocr_layout:
        doc.metadata.preserve_ocr_layout = True

    print(f"📥  读入: {len(doc.blocks)} 个块，{len(doc.pages)} 页")
    result = run_pipeline(doc, steps=args.steps)
    print(f"📤  输出: {len(result.blocks)} 个块，{len(result.toc)} 个章节")

    with open(args.output_json, "w", encoding="utf-8") as f:
        f.write(result.to_json())
    print(f"💾  已写入: {args.output_json}")
