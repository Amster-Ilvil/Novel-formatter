#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Word Builder
从 UnifiedDocument 直接生成竖排 Word（.docx），供人工校对/进一步编辑用。
移植自用户提供的 PDF.py 的竖排设置（w:textDirection = tbRl）。

用法（命令行）：
    python word_builder.py input.json output.docx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.document import UnifiedDocument, Block, BlockType


def _set_vertical_layout(doc):
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    sect_pr = doc.sections[0]._sectPr
    text_dir = OxmlElement('w:textDirection')
    text_dir.set(qn('w:val'), 'tbRl')
    sect_pr.append(text_dir)


def _add_page_break(doc):
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    para = doc.add_paragraph()
    run = para.add_run()
    br = OxmlElement('w:br')
    br.set(qn('w:type'), 'page')
    run._r.append(br)


def _block_text_for_word(b: Block) -> str:
    if b.type == BlockType.RUBY:
        # Word 里不做真正的 ruby 注音排版，退化成"底字（读音）"的可读形式
        import re
        return re.sub(r'([^\s|]+)\|([^\s|]+)', r'\1（\2）', b.text)
    return b.text


def build_word(
    doc: UnifiedDocument,
    output_path: str,
    vertical: bool = True,
    page_breaks: bool = True,
    verbose: bool = True,
) -> None:
    """
    从 UnifiedDocument 生成 Word 文档。

    Args:
        doc: 输入文档
        output_path: 输出 .docx 路径
        vertical: 是否设置竖排版式（w:textDirection=tbRl）
        page_breaks: 是否在原书每个物理页之间插入分页符
        verbose: 打印进度
    """
    try:
        from docx import Document
        from docx.shared import Pt
    except ImportError:
        raise ImportError("请安装 python-docx: pip install python-docx")

    word = Document()
    if vertical:
        _set_vertical_layout(word)

    for p in word.paragraphs:
        p._element.getparent().remove(p._element)

    TEXT_TYPES = {BlockType.PARAGRAPH, BlockType.DIALOGUE, BlockType.CHAPTER,
                  BlockType.SECTION, BlockType.RUBY}

    prev_page: int | None = None
    written = 0

    for b in doc.blocks:
        if b.type not in TEXT_TYPES:
            continue

        if page_breaks and prev_page is not None and b.page != prev_page and written > 0:
            _add_page_break(word)
        prev_page = b.page

        text = _block_text_for_word(b)
        if not text.strip():
            continue

        para = word.add_paragraph()
        run = para.add_run(text)
        if b.type == BlockType.CHAPTER:
            run.bold = True
            run.font.size = Pt(16)
        written += 1

    word.save(output_path)

    if verbose:
        size_kb = Path(output_path).stat().st_size // 1024
        print(f"✅  Word 已生成: {output_path}  ({size_kb} KB, {written} 段)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UnifiedDocument → Word (.docx)")
    parser.add_argument("input_json", help="Formatter 输出的 JSON")
    parser.add_argument("output_docx", help="输出 .docx 路径")
    parser.add_argument("--horizontal", action="store_true", help="横排模式（默认竖排）")
    parser.add_argument("--no-page-breaks", action="store_true", help="不按原书页插入分页符")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    with open(args.input_json, encoding="utf-8") as f:
        doc = UnifiedDocument.from_json(f.read())

    build_word(
        doc,
        output_path=args.output_docx,
        vertical=not args.horizontal,
        page_breaks=not args.no_page_breaks,
        verbose=not args.quiet,
    )
