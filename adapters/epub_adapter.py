#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EPUB 逆向导入适配器
把已有的 .epub（本项目自己生成过的，或任意第三方 EPUB3/EPUB2）转换为 UnifiedDocument，
方便回炉重跑 Formatter Pipeline，或者当作 Format Profile 的参考格式来源。

不依赖任何第三方库——用 zipfile + 标准库 html.parser / xml.etree 解析，
和本项目其它 adapter（docx_adapter 用 python-docx 除外）保持"能不加依赖就不加"的一致性。

用法：
    python epub_adapter.py input.epub output.json
"""

from __future__ import annotations

import posixpath
import re
import sys
import uuid
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.document import UnifiedDocument, Block, BlockType, Metadata, TocEntry

# 段落文本里判断是不是"对白"用的 class 名（本项目 epub_builder 自己写的约定，
# 第三方 EPUB 常见的写法也大同小异，直接按 class 里有没有这个词判断，不严格匹配）
DIALOGUE_CLASS_HINT = "dialogue"

# 章节/小节标题标签（h3~h6 也当小节处理，兼容第三方 EPUB 用更深层级标题的情况）
CHAPTER_TAGS = {"h1"}
SECTION_TAGS = {"h2", "h3", "h4", "h5", "h6"}
LEAF_TAGS = CHAPTER_TAGS | SECTION_TAGS | {"p"}


def _local(tag: str) -> str:
    """去掉 XML 命名空间前缀，只留局部标签名——不同工具写的 EPUB 命名空间
    声明五花八门（本项目自己早期版本写的 container.xml 命名空间甚至是错的，
    Sigil 重新保存后又会被纠正），按局部名匹配比死抠命名空间 URI 稳得多。"""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _find_by_local(root: ET.Element, name: str):
    return [el for el in root.iter() if _local(el.tag) == name]


class _ChapterHTMLParser(HTMLParser):
    """把单个章节 xhtml 解析成 Block 列表。"""

    def __init__(self, zf: zipfile.ZipFile, file_dir: str, img_dir: Path):
        super().__init__(convert_charrefs=True)
        self.zf = zf
        self.file_dir = file_dir
        self.img_dir = img_dir
        self.blocks: list[Block] = []

        self._tag_stack: list[str] = []
        self._buf: list[str] = []
        self._leaf_tag: Optional[str] = None
        self._leaf_class = ""
        self._has_ruby = False
        self._in_ruby = False
        self._in_rt = False
        self._ruby_base: list[str] = []
        self._ruby_reading: list[str] = []
        self._img_count = 0

    # ── 辅助 ────────────────────────────────────────────────────────────────

    def _flush_leaf(self):
        if self._leaf_tag is None:
            return
        text = "".join(self._buf).strip()
        tag, cls, has_ruby = self._leaf_tag, self._leaf_class, self._has_ruby
        self._leaf_tag = None
        self._buf = []
        self._leaf_class = ""
        self._has_ruby = False
        if not text:
            return

        if tag in CHAPTER_TAGS:
            btype = BlockType.CHAPTER
        elif tag in SECTION_TAGS:
            btype = BlockType.SECTION
        elif has_ruby:
            btype = BlockType.RUBY
        elif DIALOGUE_CLASS_HINT in cls:
            btype = BlockType.DIALOGUE
        else:
            btype = BlockType.PARAGRAPH

        self.blocks.append(Block(type=btype, text=text, confidence=1.0))

    def _emit_image(self, src: str):
        if not src:
            return
        img_path = posixpath.normpath(posixpath.join(self.file_dir, src))
        try:
            data = self.zf.read(img_path)
        except KeyError:
            return
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self._img_count += 1
        fname = f"{Path(img_path).stem}_{uuid.uuid4().hex[:6]}{Path(img_path).suffix}"
        out_path = self.img_dir / fname
        out_path.write_bytes(data)

        anchor = f"block_{len(self.blocks) - 1}" if self.blocks else "start"
        self.blocks.append(Block(
            type=BlockType.IMAGE_REF,
            image_path=str(out_path),
            image_anchor=anchor,
        ))

    # ── HTMLParser 回调 ───────────────────────────────────────────────────────

    def handle_starttag(self, tag, attrs):
        self._start(tag, attrs)

    def handle_startendtag(self, tag, attrs):
        self._start(tag, attrs)
        self._end(tag)

    def _start(self, tag, attrs):
        self._tag_stack.append(tag)
        attrd = dict(attrs)

        if tag == "img":
            self._flush_leaf()
            self._emit_image(attrd.get("src", ""))
            return

        if tag in LEAF_TAGS and self._leaf_tag is None:
            self._leaf_tag = tag
            self._leaf_class = attrd.get("class", "") or ""
            self._buf = []
            self._has_ruby = False
            return

        if tag == "ruby":
            self._in_ruby = True
            self._ruby_base = []
            return

        if tag == "rt":
            self._in_rt = True
            self._ruby_reading = []
            return

        if tag == "br" and self._leaf_tag is not None:
            self._buf.append("\n")

    def handle_endtag(self, tag):
        self._end(tag)

    def _end(self, tag):
        if self._tag_stack and self._tag_stack[-1] == tag:
            self._tag_stack.pop()

        if tag == "rt":
            self._in_rt = False
            return

        if tag == "ruby":
            self._in_ruby = False
            self._has_ruby = True
            base = "".join(self._ruby_base)
            reading = "".join(self._ruby_reading)
            self._buf.append(f"{base}|{reading}" if reading else base)
            return

        if tag == self._leaf_tag:
            self._flush_leaf()

    def handle_data(self, data):
        if self._in_rt:
            self._ruby_reading.append(data)
        elif self._in_ruby:
            self._ruby_base.append(data)
        elif self._leaf_tag is not None:
            self._buf.append(data)


class _NavParser(HTMLParser):
    """解析 nav.xhtml，提取 <a href="...">标题</a> 列表（保持文档顺序）。"""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.entries: list[tuple[str, str]] = []
        self._href: Optional[str] = None
        self._buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self._href = dict(attrs).get("href")
            self._buf = []

    def handle_data(self, data):
        if self._href is not None:
            self._buf.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._href is not None:
            title = "".join(self._buf).strip()
            if title:
                self.entries.append((self._href, title))
            self._href = None


def _locate_opf(zf: zipfile.ZipFile) -> str:
    container = zf.read("META-INF/container.xml").decode("utf-8", errors="replace")
    root = ET.fromstring(container)
    for el in _find_by_local(root, "rootfile"):
        full_path = el.get("full-path")
        if full_path:
            return full_path
    raise ValueError("META-INF/container.xml 里找不到 rootfile，不是合法的 EPUB")


def _parse_opf(opf_text: str) -> tuple[dict[str, tuple[str, str]], list[str], Optional[str], Metadata]:
    """返回 (manifest {id: (href, media_type)}, spine_ids, nav_item_id, metadata)"""
    root = ET.fromstring(opf_text)

    manifest: dict[str, tuple[str, str]] = {}
    nav_id = None
    for item in _find_by_local(root, "item"):
        item_id = item.get("id")
        href = item.get("href")
        media_type = item.get("media-type", "")
        if not item_id or not href:
            continue
        manifest[item_id] = (href, media_type)
        props = item.get("properties", "") or ""
        if "nav" in props.split():
            nav_id = item_id

    spine_ids = [el.get("idref") for el in _find_by_local(root, "itemref") if el.get("idref")]

    meta = Metadata(source_engine="epub_import")
    for el in _find_by_local(root, "title"):
        meta.title = (el.text or "").strip()
        break
    for el in _find_by_local(root, "creator"):
        meta.author = (el.text or "").strip()
        break
    for el in _find_by_local(root, "language"):
        meta.language = (el.text or "").strip() or meta.language
        break
    for el in _find_by_local(root, "publisher"):
        meta.publisher = (el.text or "").strip()
        break

    return manifest, spine_ids, nav_id, meta


def import_epub(epub_path: str, verbose: bool = True) -> UnifiedDocument:
    """
    导入 EPUB 文件，转换为 UnifiedDocument。

    章节标题(h1)/小节标题(h2~h6)/对话(class=dialogue的p)/普通段落(p)/振假名(ruby+rt)/
    插图(img) 都会还原成对应的 BlockType；doc.toc 直接用 nav.xhtml 里的权威目录重建，
    而不是重新猜测——EPUB 本身自带目录，没必要再跑一遍章节识别正则。
    """
    src = Path(epub_path)
    if not src.exists():
        raise FileNotFoundError(f"文件不存在: {epub_path}")

    if verbose:
        print(f"📖  导入 EPUB: {epub_path}")

    img_dir = src.parent / f"_epub_images_{src.stem}"

    doc = UnifiedDocument()

    with zipfile.ZipFile(src) as zf:
        opf_path = _locate_opf(zf)
        opf_dir = posixpath.dirname(opf_path)
        opf_text = zf.read(opf_path).decode("utf-8", errors="replace")
        manifest, spine_ids, nav_id, meta = _parse_opf(opf_text)
        doc.metadata = meta

        nav_full_href = None
        nav_entries: list[tuple[str, str]] = []
        if nav_id and nav_id in manifest:
            nav_href = manifest[nav_id][0]
            nav_full_href = posixpath.normpath(posixpath.join(opf_dir, nav_href))
            try:
                nav_text = zf.read(nav_full_href).decode("utf-8", errors="replace")
                nav_parser = _NavParser()
                nav_parser.feed(nav_text)
                nav_dir = posixpath.dirname(nav_full_href)
                nav_entries = [
                    (posixpath.normpath(posixpath.join(nav_dir, href.split("#")[0])), title)
                    for href, title in nav_parser.entries
                ]
            except KeyError:
                pass

        chapter_index = 0
        file_index = 0
        first_block_of_file: dict[str, int] = {}

        for item_id in spine_ids:
            item = manifest.get(item_id)
            if not item:
                continue
            href, media_type = item
            full_href = posixpath.normpath(posixpath.join(opf_dir, href))
            if full_href == nav_full_href:
                continue
            if "html" not in media_type and not href.endswith((".xhtml", ".html", ".htm")):
                continue
            try:
                raw = zf.read(full_href).decode("utf-8", errors="replace")
            except KeyError:
                continue

            file_dir = posixpath.dirname(full_href)
            parser = _ChapterHTMLParser(zf, file_dir, img_dir)
            parser.feed(raw)
            parser.close()
            file_index += 1

            base_idx = len(doc.blocks)
            first_chapter_idx = None
            for blk in parser.blocks:
                # 每个 spine 文件当一"页"处理——EPUB 章节本身没有物理页码，但
                # engine.formatter 里好几处判定（比如疑似目录页检测）是按
                # block.page 分组的，如果所有 block 都用默认页码 0，不同文件
                # 里恰好都匹配章节正则的标题会被错误地聚到"同一页"，触发"一页
                # 出现≥2个不同章节标识符→疑似目录页"的误判，导致真章节被抑制。
                blk.page = file_index
                if blk.type == BlockType.CHAPTER:
                    if first_chapter_idx is None:
                        first_chapter_idx = len(doc.blocks)
                    chapter_index += 1
                    blk.chapter_index = chapter_index
                doc.blocks.append(blk)

            if base_idx < len(doc.blocks):
                first_block_of_file[full_href] = first_chapter_idx if first_chapter_idx is not None else base_idx

        for href, title in nav_entries:
            block_idx = first_block_of_file.get(href)
            if block_idx is None:
                continue
            ci = doc.blocks[block_idx].chapter_index or 0
            doc.toc.append(TocEntry(title=title, chapter_index=ci, block_index=block_idx))

    n_images = sum(1 for b in doc.blocks if b.type == BlockType.IMAGE_REF)
    doc.add_log("epub_import", f"导入 {len(doc.blocks)} 个块，{len(doc.toc)} 个目录条目，{n_images} 张图片", len(doc.blocks))

    if verbose:
        print(f"  ✅  导入完成: {len(doc.blocks)} 个块，{len(doc.toc)} 个目录条目，{n_images} 张图片")

    return doc


# ── 参考 EPUB 排版特征提取（供 Format Profile 使用） ────────────────────────────

def analyze_style(epub_path: str) -> dict:
    """
    从一本参考 EPUB 里提取排版特征，供 Format Profile 使用：
        css                 —— 原样拼接的样式表内容（"参考其格式"最直接的落地：
                                直接复用参考书的排版 CSS，而不是重新猜规则）
        vertical            —— 从 CSS 的 writing-mode 判定是否竖排
        font_family         —— 从 CSS body 规则里提取
        line_height         —— 同上
        paragraph_indent    —— "fullwidth_space"（正文字符本身带全角空格缩进）
                                或 "css_text_indent"（缩进靠 CSS text-indent，字符本身不带）
        dialogue_quote_style—— 抽样对白段落首字符，取样本里最常见的引号风格
        sample_title / sample_author —— 参考书自身的元数据，仅用于展示
    """
    doc = import_epub(epub_path, verbose=False)

    css_parts = []
    with zipfile.ZipFile(epub_path) as zf:
        for name in zf.namelist():
            if name.lower().endswith(".css"):
                try:
                    css_parts.append(zf.read(name).decode("utf-8", errors="replace"))
                except KeyError:
                    continue
    css = "\n\n".join(css_parts)

    vertical = bool(re.search(r'writing-mode\s*:\s*vertical', css))

    m = re.search(r'font-family\s*:\s*([^;]+);', css)
    font_family = m.group(1).strip() if m else ""
    m = re.search(r'line-height\s*:\s*([\d.]+)', css)
    line_height = float(m.group(1)) if m else 1.8

    paragraphs = [b.text for b in doc.blocks if b.type == BlockType.PARAGRAPH][:50]
    indented = sum(1 for t in paragraphs if t[:1] in ("　", " "))
    paragraph_indent = (
        "fullwidth_space" if paragraphs and indented / len(paragraphs) > 0.5
        else "css_text_indent"
    )

    dialogues = [b.text for b in doc.blocks if b.type == BlockType.DIALOGUE][:50]
    bracket = sum(1 for t in dialogues if t[:1] in ("「", "『"))
    curly_quote = sum(1 for t in dialogues if t[:1] in ("“", '"'))
    dash = sum(1 for t in dialogues if t[:1] in ("—", "－", "-"))
    counts = {"「」": bracket, "“”": curly_quote, "—": dash}
    dialogue_quote_style = max(counts, key=counts.get) if any(counts.values()) else "「」"

    return {
        "css": css,
        "vertical": vertical,
        "font_family": font_family,
        "line_height": line_height,
        "paragraph_indent": paragraph_indent,
        "dialogue_quote_style": dialogue_quote_style,
        "sample_title": doc.metadata.title,
        "sample_author": doc.metadata.author,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EPUB → UnifiedDocument JSON")
    parser.add_argument("input_epub", help="输入 EPUB 文件")
    parser.add_argument("output_json", help="输出 JSON 路径")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    result = import_epub(args.input_epub, verbose=not args.quiet)

    with open(args.output_json, "w", encoding="utf-8") as f:
        f.write(result.to_json())

    print(f"💾  已写入: {args.output_json}")
