#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text Extractor —— 把各种格式的"高质量文本来源"统一转成 Paragraph 列表，
供 engine/alignment.py 跟 OCR 结果对齐。

docx / epub / json 已经有对应的 adapter 能转成 UnifiedDocument（docx_adapter /
epub_adapter / UnifiedDocument.from_json 本身），这里直接复用，不重新写一遍
解析逻辑——只是从 UnifiedDocument 的 text_blocks() 里再抽一层 Paragraph 出来。
txt / markdown / html 目前没有对应的 UnifiedDocument 导入器，这里给一个足够用
的最小实现（按空行分段，标题按 CHAPTER_RE / markdown # 号识别）。

新增一种来源格式：在 EXTRACTORS 里加一行映射即可，不需要改 alignment/
replacement 任何代码。
"""

from __future__ import annotations

import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.document import UnifiedDocument, BlockType
from models.paragraph import Paragraph
from engine.formatter import CHAPTER_RE


def _paragraphs_from_unified_doc(doc: UnifiedDocument, source_name: str) -> list[Paragraph]:
    paragraphs: list[Paragraph] = []
    current_chapter = ""
    idx = 0
    for b in doc.text_blocks():
        text = b.text.strip()
        if not text:
            continue
        is_title = b.type in (BlockType.CHAPTER, BlockType.SECTION)
        if b.type == BlockType.CHAPTER:
            current_chapter = text
        paragraphs.append(Paragraph(
            text=text, index=idx, chapter=current_chapter,
            source=source_name, is_title=is_title,
        ))
        idx += 1
    return paragraphs


def extract_from_docx(path: str) -> list[Paragraph]:
    from adapters.docx_adapter import import_docx
    doc = import_docx(path, verbose=False)
    return _paragraphs_from_unified_doc(doc, Path(path).name)


def extract_from_epub(path: str) -> list[Paragraph]:
    from adapters.epub_adapter import import_epub
    doc = import_epub(path, verbose=False)
    return _paragraphs_from_unified_doc(doc, Path(path).name)


def _is_paddleocr_vl_export(data) -> bool:
    """识别 PaddleOCR-VL 在线服务导出的原始 JSON（逐页 dict 组成的 list，每页带
    `markdown`/`prunedResult` 字段）—— 跟本项目 UnifiedDocument.to_json() 的输出
    （单个 dict，顶层是 metadata/pages/blocks）完全是两套 schema，不能直接当
    UnifiedDocument 解析（旧代码在这里会因为 list 没有 .get 直接崩溃）。"""
    if not isinstance(data, list) or not data:
        return False
    return all(isinstance(p, dict) and "markdown" in p for p in data)


def extract_from_paddleocr_vl_export(path: str, data: list[dict] | None = None) -> list[Paragraph]:
    """PaddleOCR-VL 在线服务导出的 JSON：每页一个 dict，`markdown.text` 已经是
    干净的 markdown（`#`标题 + 空行分段），直接拼接全书页面复用 markdown 解析逻辑，
    不用逐个 parsing_res_list block 去重建结构（那是给 OCR 主文档用的，这里只要
    干净正文）。"""
    if data is None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    page_texts = [p.get("markdown", {}).get("text", "") for p in data]
    combined = "\n\n".join(t for t in page_texts if t.strip())
    return _paragraphs_from_markdown_text(combined, Path(path).name)


def extract_from_json(path: str) -> list[Paragraph]:
    raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw)
    if _is_paddleocr_vl_export(data):
        return extract_from_paddleocr_vl_export(path, data=data)
    if isinstance(data, list):
        raise ValueError(
            f"{Path(path).name} 是 JSON 数组，但不是已知的 PaddleOCR-VL 导出格式"
            "（缺少每页的 markdown 字段），也不是本项目的 UnifiedDocument 格式"
            "（应为顶层 dict，含 metadata/pages/blocks）。"
        )
    doc = UnifiedDocument.from_dict(data)
    return _paragraphs_from_unified_doc(doc, Path(path).name)


# CHAPTER_RE 的否定预查是为日语写的（の/は/が…），纯文本来源常见中文小说，
# 续接字/词的可能性没法穷举（"的""了""内容""发生"……），逐个往排除表里加字符
# 是打地鼠。这里用两条互补的信号：
#   1. 紧跟在 CHAPTER_RE 匹配到的前缀后面，如果立刻是常见续接助词/字
#      （"の""的"……），肯定是整句话，不是标题。
#   2. 整行是不是以句末标点（。！？等）收尾——真正的章节标题即使带副标题
#      也顶多到"――"这种收尾（比如"プロローグ　魔王敗れる。そして――"，
#      标题内部允许出现句号，但整行不会以句号收尾），一旦整行以句末标点
#      结束，说明这是一句完整的话，只是恰好提到了章节号。
#      注意只看"行尾"，不能看"匹配前缀之后哪里出现过标点"——标题内部本身
#      可能带一个句号构成戏剧性停顿，那种情况仍然是标题，不该被拒绝。
_TITLE_CONTINUATION_RE = re.compile(r'^(の|は|が|を|に|で|と|も|です|だ|という|的|了|着|过|之|时|吗|呢)')
_SENTENCE_END_PUNCT = "。！？.!?"


def _looks_like_title(line: str) -> bool:
    if len(line) >= 40:
        return False
    m = CHAPTER_RE.match(line)
    if not m:
        return False
    if _TITLE_CONTINUATION_RE.match(line[m.end():]):
        return False
    stripped = line.rstrip("　 ")
    return not (stripped and stripped[-1] in _SENTENCE_END_PUNCT)


def extract_from_txt(path: str) -> list[Paragraph]:
    """纯文本：按空行分段；单独一行且匹配 CHAPTER_RE 的当标题。"""
    text = Path(path).read_text(encoding="utf-8")
    raw_paragraphs = re.split(r'\n\s*\n', text)
    paragraphs: list[Paragraph] = []
    current_chapter = ""
    idx = 0
    for raw in raw_paragraphs:
        t = raw.strip()
        if not t:
            continue
        # 段落内部可能还有单个换行（软换行），当作同一段落合并成一行——
        # 纯文本没有 OCR 那种"逐行是逐段"的保证，空行才是真正的段落边界。
        joined = re.sub(r'[\s　]*\n[\s　]*', '', t)
        is_title = _looks_like_title(joined)
        if is_title:
            current_chapter = joined
        paragraphs.append(Paragraph(
            text=joined, index=idx, chapter=current_chapter,
            source=Path(path).name, is_title=is_title,
        ))
        idx += 1
    return paragraphs


_MD_HEADING_RE = re.compile(r'^#{1,6}\s*(.+?)\s*#*$')
_MD_IMG_TAG_RE = re.compile(r'<[^>]+>')


def _paragraphs_from_markdown_text(text: str, source_name: str) -> list[Paragraph]:
    raw_paragraphs = re.split(r'\n\s*\n', text)
    paragraphs: list[Paragraph] = []
    current_chapter = ""
    idx = 0
    for raw in raw_paragraphs:
        t = raw.strip()
        if not t:
            continue
        m = _MD_HEADING_RE.match(t)
        if m:
            title_text = m.group(1).strip()
            current_chapter = title_text
            paragraphs.append(Paragraph(
                text=title_text, index=idx, chapter=current_chapter,
                source=source_name, is_title=True,
            ))
            idx += 1
            continue
        # PaddleOCR-VL 导出的 markdown 里，图片块是一整段 <div><img .../></div>，
        # 不是正文，直接跳过（图片结构本来就该来自 OCR 那一份，不该被替换进度对不上）。
        if _MD_IMG_TAG_RE.match(t) and not _MD_IMG_TAG_RE.sub('', t).strip():
            continue
        joined = re.sub(r'[\s　]*\n[\s　]*', '', t)
        is_title = _looks_like_title(joined)
        if is_title:
            current_chapter = joined
        paragraphs.append(Paragraph(
            text=joined, index=idx, chapter=current_chapter,
            source=source_name, is_title=is_title,
        ))
        idx += 1
    return paragraphs


def extract_from_markdown(path: str) -> list[Paragraph]:
    """Markdown：`#`标题行直接当章节标题；其余按空行分段，跟 txt 一致。"""
    text = Path(path).read_text(encoding="utf-8")
    return _paragraphs_from_markdown_text(text, Path(path).name)


class _HTMLParagraphParser(HTMLParser):
    """独立 HTML 文件的最小段落提取——不像 epub_adapter 里的解析器那样需要
    处理 zip 内图片/ruby，这里只关心标题(h1~h6)和段落(p)文字。"""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.paragraphs: list[tuple[str, bool]] = []  # (text, is_title)
        self._tag: str | None = None
        self._buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6", "p") and self._tag is None:
            self._tag = tag
            self._buf = []

    def handle_data(self, data):
        if self._tag is not None:
            self._buf.append(data)

    def handle_endtag(self, tag):
        if tag == self._tag:
            text = "".join(self._buf).strip()
            if text:
                self.paragraphs.append((text, tag.startswith("h")))
            self._tag = None


def extract_from_html(path: str) -> list[Paragraph]:
    html_text = Path(path).read_text(encoding="utf-8", errors="replace")
    parser = _HTMLParagraphParser()
    parser.feed(html_text)

    paragraphs: list[Paragraph] = []
    current_chapter = ""
    for idx, (text, is_title) in enumerate(parser.paragraphs):
        if is_title:
            current_chapter = text
        paragraphs.append(Paragraph(
            text=text, index=idx, chapter=current_chapter,
            source=Path(path).name, is_title=is_title,
        ))
    return paragraphs


EXTRACTORS: dict[str, Callable[[str], list[Paragraph]]] = {
    ".docx": extract_from_docx,
    ".epub": extract_from_epub,
    ".json": extract_from_json,
    ".txt": extract_from_txt,
    ".md": extract_from_markdown,
    ".markdown": extract_from_markdown,
    ".html": extract_from_html,
    ".htm": extract_from_html,
}


def extract_paragraphs(path: str) -> list[Paragraph]:
    ext = Path(path).suffix.lower()
    extractor = EXTRACTORS.get(ext)
    if extractor is None:
        available = ", ".join(sorted(EXTRACTORS.keys()))
        raise ValueError(f"不支持的来源格式: {ext}。目前支持: {available}")
    return extractor(path)
