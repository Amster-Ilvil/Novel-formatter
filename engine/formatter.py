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
    4.  remove_duplicates        — 删除重复段落和重复对白（含章节标题模糊去重）
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
from pathlib import Path
from typing import Optional, Callable
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.document import UnifiedDocument, Block, BlockType, TocEntry, new_temp_repo_path
from engine.ai_formatter import AIFormatterStep, ai_correction_step

AI_STEP_ENABLED = True

# ── 常量 ──────────────────────────────────────────────────────────────────────

# 章节/小节标题正则（正规化后匹配：先去除多余空格）
# "第[数字][章話節巻]"后面接否定预查：排除"第１０話ですが……""第一章は
# 約７５話で……"这类正文里提到章节号、但本身是完整句子（后面直接接て
# にをは等助词或です/だ）的假阳性——真正的章节标题后面要么就结束了，
# 要么接的是空格/顿号再接副标题，不会直接用助词/系动词把它接成一个句子。
CHAPTER_RE = re.compile(
    r'^(序章|終章|プロローグ|フロローグ|ブロローグ|エピローグ|後記|あとがき|'
    r'幕間[\s　]?.*|'
    r'第[一二三四五六七八九十百〇零\d]+[章話節巻](?!(は|が|を|に|で|と|も|の|です|だ|という))'
    r'|Chapter\s*\d+)',
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
    (re.compile(r'\.{3,}'),    '……'),
    (re.compile(r'…{2,}'),     '……'),
    (re.compile(r'-{2,}'),     '——'),
    (re.compile(r'ー{2,}'),    '——'),
    (re.compile(r'\(([^)]{1,20})\)'), r'（\1）'),
    (re.compile(r'　+$'),      ''),
    (re.compile(r' +$'),       ''),
]

CHAPTER_TITLE_SPACE_RE = re.compile(
    r'^((?:プロローグ|序章|終章|エピローグ|'
    r'第[一二三四五六七八九十百〇零\d]+[章話節巻])(?=\S))'
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
MAX_MERGE_LINES = 8
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
            # 位置感知：bbox 靠近页面顶部或底部的短文本
            if b.bbox and len(t) <= HEADER_MAX:
                y = b.bbox.y
                if y < 0.15 or y > 0.85:
                    removed += 1
                    continue
        kept.append(b)

    doc.blocks = kept
    doc.add_log("clean_metadata", f"删除 {removed} 个页眉/页码块", removed)
    return doc


# ── 步骤 2.4：拆出内嵌章节标题 ──────────────────────────────────────────────

EMBEDDED_CHAPTER_RE = re.compile(
    r'(序章|終章|プロローグ|フロローグ|ブロローグ|エピローグ|'
    r'第[一二三四五六七八九十百〇零\d]+[章話節巻](?!(は|が|を|に|で|と|も|の|です|だ|という)))'
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


# ── 步骤 3：合并断句（接续词感知）─────────────────────────────────────────────

def _should_merge_with_next(current_text: str) -> bool:
    """判断当前文本块是否应与下一块合并"""
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
    if len(current_text) < 5:
        return True
    return not SENTENCE_END_RE.search(current_text)


_MERGEABLE_TYPES = {BlockType.PARAGRAPH, BlockType.DIALOGUE}


def merge_broken_sentences(doc: UnifiedDocument) -> UnifiedDocument:
    """
    竖排 OCR 断句合并，增加日语接续词感知。
    当前块末尾是て/で/し/から/けど等接续词时优先合并。

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
    allow_cross_page = doc.metadata.source_engine == "pdf_text_layer"

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
        if CHAPTER_RE.match(b.text.strip()) or BARE_NUMBER_RE.match(b.text.strip()):
            result.append(b)
            i += 1
            continue

        merge_count = 0
        while (
            i + 1 < len(doc.blocks)
            and merge_count < MAX_MERGE_LINES
            and doc.blocks[i + 1].type in _MERGEABLE_TYPES
            and (
                allow_cross_page
                or doc.blocks[i + 1].page == b.page
                or (
                    doc.blocks[i + 1].page > b.page
                    and not SENTENCE_END_RE.search(b.text.rstrip())
                )
            )
            and _should_merge_with_next(b.text)
            and not CHAPTER_RE.match(doc.blocks[i + 1].text.strip())
            and not BARE_NUMBER_RE.match(doc.blocks[i + 1].text.strip())
        ):
            next_b = doc.blocks[i + 1]
            b = copy.copy(b)
            b.text = b.text + next_b.text
            b.ocr_raw = b.ocr_raw + next_b.ocr_raw
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

def _normalize_for_dedup(text: str) -> str:
    """去除空白后用于去重比较——竖排大字号章节标题常被 Vision 拆成多行
    观测结果，各自还可能带不同的前后空格/全角空格，精确字符串比较会漏判。"""
    return re.sub(r'[\s　]+', '', text)


def _text_similarity(a: str, b: str) -> float:
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio()


def _is_protected_title(block: Block, normalized_text: str) -> bool:
    """标题是结构内容，不能由正文去重规则删除。"""
    return block.type in {BlockType.CHAPTER, BlockType.SECTION} or bool(
        CHAPTER_RE.match(normalized_text)
    )


def remove_duplicates(doc: UnifiedDocument) -> UnifiedDocument:
    """
    去重：
        - 对白全文去重（同一句对白整本书只保留第一次）
        - 相邻块去重，按去除空白后的规范化文本比较（而不是精确字符串比较），
          避免「序章」「序章 」「 序章」这类因空格不同而被判定为"不同"从而
          都被保留
        - 章节/分节标题一律保留，即使相邻文本完全相同；标题重复可能是
          原书的结构，而非 OCR 噪声，不能由去重步骤擅自删除
    """
    doc = copy.deepcopy(doc)

    removed = 0
    result: list[Block] = []
    seen_dialogues: set[str] = set()

    for b in doc.blocks:
        if b.type == BlockType.DIALOGUE:
            if b.text in seen_dialogues:
                removed += 1
                continue
            seen_dialogues.add(b.text)

        normalized = _normalize_for_dedup(b.text)
        if normalized:
            is_title = _is_protected_title(b, normalized)
            previous_is_title = bool(result) and _is_protected_title(
                result[-1], _normalize_for_dedup(result[-1].text)
            )
            if (
                result
                and not is_title
                and not previous_is_title
                and _normalize_for_dedup(result[-1].text) == normalized
            ):
                removed += 1
                continue

        result.append(b)

    doc.blocks = result
    doc.add_log("remove_duplicates", f"删除 {removed} 个重复块", removed)
    return doc


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
    if doc.metadata.source_engine == "pdf_text_layer":
        return copy.deepcopy(doc)

    doc = copy.deepcopy(doc)

    text_types = {BlockType.PARAGRAPH, BlockType.DIALOGUE, BlockType.CHAPTER, BlockType.SECTION}
    fixed_count = 0

    for b in doc.blocks:
        if b.type not in text_types:
            continue
        original = b.text
        b.text = _fix_ocr_dash_artifacts(b.text)
        if b.text != original:
            b.ocr_raw = b.ocr_raw or original
            b.modified_by = "fix_ocr_dash_artifacts"
            fixed_count += 1

    doc.add_log("fix_ocr_dash_artifacts", f"修复 {fixed_count} 处误读的破折号", fixed_count)
    return doc


# ── 步骤 6：对白独立换行（迭代拆分）─────────────────────────────────────────

def _quote_close_open_types(text: str) -> dict[int, str]:
    """
    扫描文本，按"最近未闭合的起始符号"给每个引号收尾符（」/』）标注它
    真正配对的起始符号类型（'「' 或 '『'），而不是死板地要求「配」、
    『配』。

    原文/OCR 经常会把『术语引用』的收尾符 』错打或错读成」（两者字形
    接近），如果只看字面收尾符号判断"上一句是不是已经结束"，就会把这类
    因误读/误打产生的」误认成一段真正对白的结束，进而把它后面紧跟的
    下一个引号错误地当成新的独立对白拆出来，把一整句连贯的叙述（比如
    并列列举多个用引号标出的名词）硬生生切断。按实际配对关系（栈追踪）
    确定"这个收尾符号原本是从『开始的"之后，才能正确地把它排除在
    "对白结束"的判断之外。
    """
    stack: list[str] = []
    result: dict[int, str] = {}
    for i, ch in enumerate(text):
        if ch in ('「', '『'):
            stack.append(ch)
        elif ch in ('」', '』'):
            if stack:
                result[i] = stack.pop()
    return result


def restore_dialogue_breaks(doc: UnifiedDocument) -> UnifiedDocument:
    """
    迭代拆分 PARAGRAPH 块中内嵌的对白。
    支持多轮对白混合叙述的复杂场景。

    只拆「」（真正的人物对白），不拆『』——『』在日文轻小说里更常用来
    标记书名/招式名/术语/强调等叙述内嵌引用（例如"…その使い手たる『勇者』
    である。"是地の文，『勇者』只是被引用的称号，不是有人在说话）。
    之前『』也会触发拆分，导致一整句连贯的叙述被硬切成好几段。
    """
    doc = copy.deepcopy(doc)

    # 「…」只有位于块首、或紧跟一句已结束的叙述时才视为独立对白。句中
    # 的「何か」「理由」等是术语/强调，拆开会把完整叙述人为断成三块。
    dialogue_re = re.compile(r'「[^」]*」', re.DOTALL)

    split_count = 0
    result: list[Block] = []

    for b in doc.blocks:
        if b.type != BlockType.PARAGRAPH:
            result.append(b)
            continue

        text = b.text
        # 按真实配对类型标注每个收尾符号，供下面判断"这个」是不是真的
        # 结束了一段人物对白"，而不是被误读/误打的『术语引用』收尾。
        close_open_types = _quote_close_open_types(text)
        sub_blocks: list[Block] = []
        cursor = 0
        for m in dialogue_re.finditer(text):
            # 光有引号不足以说明是人物说话。若它前面不是句子边界，就保留
            # 在当前叙述中；下一处真正的对白仍可在后续循环中被识别。
            preceding = text[:m.start()].rstrip()
            if not preceding:
                is_dialogue = m.start() == 0
            else:
                last_ch = preceding[-1]
                if last_ch in '。！？\n':
                    is_dialogue = True
                elif last_ch in '」』':
                    # 只有真正配对自「的收尾符才算一段独立对白刚刚结束；
                    # 配对自『的收尾符（哪怕字面上被误读/误打成」）只是
                    # 术语引用结束，后面紧跟的引号仍属于同一句叙述，不
                    # 应该被当成新对白拆出来。
                    last_idx = len(preceding) - 1
                    is_dialogue = close_open_types.get(last_idx) == '「'
                else:
                    is_dialogue = False
            if not is_dialogue:
                continue

            before = text[cursor:m.start()]
            dialogue = m.group(0)
            if before.strip():
                nb = copy.copy(b)
                nb.text = before.strip()
                nb.type = BlockType.PARAGRAPH
                nb.modified_by = "restore_dialogue_breaks"
                sub_blocks.append(nb)

            db = copy.copy(b)
            db.text = dialogue.strip()
            db.type = BlockType.DIALOGUE
            db.modified_by = "restore_dialogue_breaks"
            sub_blocks.append(db)
            split_count += 1
            cursor = m.end()

        remainder = text[cursor:]
        if remainder.strip():
            nb = copy.copy(b)
            nb.text = remainder.strip()
            nb.type = BlockType.PARAGRAPH
            nb.modified_by = "restore_dialogue_breaks"
            sub_blocks.append(nb)

        if len(sub_blocks) > 1:
            result.extend(sub_blocks)
        else:
            result.append(b)

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

    doc = copy.deepcopy(doc)
    fixed_count = 0

    text_types = {BlockType.PARAGRAPH, BlockType.DIALOGUE,
                  BlockType.CHAPTER, BlockType.SECTION, BlockType.RUBY}

    for b in doc.blocks:
        if b.type not in text_types:
            continue

        original = b.text

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

    doc.add_log("normalize_punctuation", f"修正 {fixed_count} 处标点/错字", fixed_count)
    return doc


# ── Pipeline：组合所有步骤 ────────────────────────────────────────────────────

PIPELINE_STEPS = [
    ("reading_order",       reading_order_step),
    ("ai_correction",       ai_correction_step),
    ("clean_metadata",      clean_metadata_blocks),
    ("split_embedded_titles", split_embedded_chapter_titles),
    ("strip_chapter_notes", strip_chapter_notes),
    ("merge_sentences",     merge_broken_sentences),
    ("remove_duplicates",   remove_duplicates),
    ("fix_dash_artifacts",  fix_ocr_dash_artifacts),
    ("dialogue_restore",    restore_dialogue_breaks),
    ("restore_indents",     restore_indents_and_breaks),
    ("recover_ruby",        recover_ruby),
    ("detect_chapters",     detect_chapters),
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
    to_run = steps or [s for s, _ in PIPELINE_STEPS]
    step_map = {s: fn for s, fn in PIPELINE_STEPS}
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
        if verbose:
            print(f"  ▶  {step_id} ...", end=" ")

        if progress_callback:
            progress_callback(step_id, step_idx, total)

        result = fn(current)
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
        "--steps", nargs="*",
        help="指定运行步骤（空=全部）：reading_order clean_metadata merge_sentences "
             "remove_duplicates fix_dash_artifacts dialogue_restore restore_indents "
             "recover_ruby detect_chapters normalize_punct"
    )
    args = parser.parse_args()

    with open(args.input_json, encoding="utf-8") as f:
        doc = UnifiedDocument.from_json(f.read())

    print(f"📥  读入: {len(doc.blocks)} 个块，{len(doc.pages)} 页")
    result = run_pipeline(doc, steps=args.steps)
    print(f"📤  输出: {len(result.blocks)} 个块，{len(result.toc)} 个章节")

    with open(args.output_json, "w", encoding="utf-8") as f:
        f.write(result.to_json())
    print(f"💾  已写入: {args.output_json}")
