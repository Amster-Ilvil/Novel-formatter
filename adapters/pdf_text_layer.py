#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF 文字层适配器
针对"带文字层"（selectable text）的 PDF —— 这类 PDF 不需要 OCR，直接读取
PyMuPDF 暴露的字符级数据（坐标 + 字号），比图像 OCR 更准确、没有噪点，
还能靠字号直接识别振假名（ルビ通常用比正文小得多的字号排印）。

移植自用户提供的 PDF.py（竖排小说 PDF → Word 脚本），核心算法原样保留：
    - 按字号阈值跳过振假名（不像 OCR 路线那样试图保留成 ruby 标注，
      这里字号信息在"扁平化成文字"之后就没有了，唯一可靠的处理方式
      就是在提取阶段直接按字号过滤掉）
    - 按 X 坐标分列（COL_WIDTH 一列），列内按 Y 排序 —— 处理竖排日文
      的阅读顺序，比图像 OCR 路线的 GapTree 更精确，因为这里是从
      PDF 里拿到的精确字符坐标，不是靠 bounding box 估算
    - 页码识别：纯数字 + 位于页面底部 20% 区域
    - 段落末尾粘连的页码数字（没被识别成独立一列）额外剔除一次

依赖：
    pip install pymupdf python-docx

用法：
    python pdf_text_layer.py input.pdf output.json
"""

from __future__ import annotations

import re
import sys
from itertools import groupby
from pathlib import Path
from typing import Optional, Callable

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.document import (
    UnifiedDocument, Block, BlockType, PageInfo, BoundingBox, Metadata, TocEntry
)

COL_WIDTH = 20
FURIGANA_SIZE_THRESHOLD = 8.0

CHAPTER_RE = re.compile(
    # フロローグ：竖排 PDF 字符提取常把"プ"的半浊点丢掉错读成"フ"，
    # 「フロローグ」本身不是真实存在的日语词，只可能是"プロローグ"的错读，
    # 一并纳入识别正则，避免序章因此漏检、跟正文粘连在一起。
    r'^(序章|終章|プロローグ|フロローグ|エピローグ|後記|あとがき|'
    r'幕間[\s　]?.*|'
    r'第[一二三四五六七八九十百〇零\d]+[章話節巻](?!(は|が|を|に|で|と|も|の|です|だ|という))'
    r'|Chapter\s*\d+)',
    re.IGNORECASE
)
DIALOGUE_START = ('「', '『')
DIALOGUE_END = ('」', '』')

TRAILING_PAGE_NUM_RE = re.compile(r'\d+$')


def has_text_layer(pdf_path: str, min_chars: int = 20, sample_pages: int = 5) -> bool:
    """
    粗略判断 PDF 有没有可提取的文字层：抽样前几页，看看能不能拿到实质性文字。
    扫描版 PDF（整页是图片）这里几乎总是拿到空字符串或极少字符。
    """
    try:
        import fitz
    except ImportError:
        return False

    try:
        pdf = fitz.open(pdf_path)
    except Exception:
        return False

    total_chars = 0
    for page in pdf[:sample_pages]:
        total_chars += len(page.get_text("text").strip())
    pdf.close()
    return total_chars >= min_chars


def _is_page_number(chars: list[dict], page_height: float) -> bool:
    """一列字符是否是页码：纯数字（1~4位）且位于页面底部 20% 区域"""
    if not chars:
        return False
    text = "".join(ch["c"] for ch in chars)
    if not re.fullmatch(r'\d{1,4}', text.strip()):
        return False
    avg_y = sum(ch["y"] for ch in chars) / len(chars)
    return avg_y > page_height * 0.80


def _detect_block_type(text: str) -> BlockType:
    if CHAPTER_RE.match(text):
        return BlockType.CHAPTER
    if text.startswith(DIALOGUE_START) or text.endswith(DIALOGUE_END):
        return BlockType.DIALOGUE
    return BlockType.PARAGRAPH


def extract_pdf_text_layer(
    pdf_path: str,
    page_overrides: dict[int, str] | None = None,
    verbose: bool = True,
    furigana_threshold: float = FURIGANA_SIZE_THRESHOLD,
    col_width: float = COL_WIDTH,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> UnifiedDocument:
    """
    从带文字层的 PDF 直接提取正文，跳过振假名和页码，按竖排阅读顺序
    （右→左列、列内上→下）分列输出。每一列对应一个 Block。

    Args:
        pdf_path: PDF 文件路径
        page_overrides: {页码: BlockType字符串} —— 非"正文"的页（封面/插图/
                        目录扫描页等）直接跳过文字层提取，整页作为 IMAGE_REF
                        占位（PDF 页面本身不渲染成图片，只是不产生文字块；
                        如果需要页面缩略图，请走 Page Manager 的图片流程）
        verbose: 是否打印进度
        furigana_threshold: 字号小于此值的字符视为振假名，提取时跳过
        col_width: 分列宽度（PDF 坐标单位，通常约等于正文字号）
        progress_callback: (current, total, label) -> None

    Returns:
        UnifiedDocument
    """
    try:
        import fitz
    except ImportError:
        raise ImportError("请安装 PyMuPDF: pip install pymupdf")

    src = Path(pdf_path)
    if not src.exists():
        raise FileNotFoundError(f"文件不存在: {pdf_path}")

    overrides = {int(k): v for k, v in (page_overrides or {}).items()}

    pdf = fitz.open(pdf_path)
    doc = UnifiedDocument()
    doc.metadata = Metadata(source_engine="pdf_text_layer", language="ja")

    order_counter = 0
    chapter_index = 0
    text_page_count = 0
    skipped_page_count = 0
    furigana_chars_skipped = 0
    page_number_cols_skipped = 0

    total = len(pdf)
    if verbose:
        print(f"📂  PDF 文字层提取：共 {total} 页")

    for page_idx, page in enumerate(pdf):
        page_no = page_idx + 1

        override_type = overrides.get(page_no)
        if override_type and override_type != BlockType.PARAGRAPH.value:
            # 非正文页整页跳过提取（封面/插图/目录扫描页等）
            try:
                ptype = BlockType(override_type)
            except ValueError:
                ptype = BlockType.UNKNOWN
            doc.pages.append(PageInfo(page_no=page_no, page_type=ptype, confidence=1.0))
            skipped_page_count += 1
            if verbose:
                print(f"  [{page_no:3d}/{total}] 跳过提取（已标注为 {override_type}）")
            if progress_callback is not None:
                progress_callback(page_no, total, f"跳过（{override_type}）")
            continue

        page_height = page.rect.height
        data = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        chars: list[dict] = []
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    font_size = span.get("size", 12)
                    if font_size < furigana_threshold:
                        furigana_chars_skipped += len(span.get("chars", []))
                        continue
                    for char in span.get("chars", []):
                        c = char["c"]
                        if c.strip():
                            chars.append({
                                "c": c,
                                "x": char["origin"][0],
                                "y": char["origin"][1],
                                "size": font_size,
                            })

        chars.sort(key=lambda ch: (-round(ch["x"] / col_width), ch["y"]))

        page_blocks: list[Block] = []
        page_w = page.rect.width

        for _, group in groupby(chars, key=lambda ch: -round(ch["x"] / col_width)):
            group = list(group)

            if _is_page_number(group, page_height):
                page_number_cols_skipped += 1
                continue

            col_text = "".join(ch["c"] for ch in group).strip()
            if not col_text:
                continue

            # 段落末尾粘连的页码数字（没被单独分成一列，跟在正文最后）
            # ——对应 Word 通配符替换 ([0-9]{1,})(^13) → \2
            stripped = TRAILING_PAGE_NUM_RE.sub('', col_text)
            if stripped != col_text and stripped.strip():
                col_text = stripped

            xs = [ch["x"] for ch in group]
            ys = [ch["y"] for ch in group]
            avg_size = sum(ch["size"] for ch in group) / len(group)
            bbox = BoundingBox(
                x=max(0.0, min(xs)) / page_w if page_w else 0.0,
                y=max(0.0, min(ys)) / page_height if page_height else 0.0,
                w=(max(xs) - min(xs) + avg_size) / page_w if page_w else 0.0,
                h=(max(ys) - min(ys) + avg_size) / page_height if page_height else 0.0,
            )

            btype = _detect_block_type(col_text)
            b = Block(
                type=btype,
                text=col_text,
                ocr_raw=col_text,
                page=page_no,
                bbox=bbox,
                reading_order=order_counter + len(page_blocks),
                confidence=1.0,
            )
            if btype == BlockType.CHAPTER:
                chapter_index += 1
                b.chapter_index = chapter_index
                doc.toc.append(TocEntry(
                    title=col_text, chapter_index=chapter_index,
                    block_index=len(doc.blocks) + len(page_blocks),
                ))
            page_blocks.append(b)

        doc.blocks.extend(page_blocks)
        order_counter += len(page_blocks)

        ptype = BlockType.PARAGRAPH if page_blocks else BlockType.BLANK
        doc.pages.append(PageInfo(page_no=page_no, page_type=ptype, confidence=1.0))
        if page_blocks:
            text_page_count += 1

        if verbose:
            print(f"  [{page_no:3d}/{total}] → {len(page_blocks)} 列（已跳过振假名/页码）")
        if progress_callback is not None:
            progress_callback(page_no, total, f"{len(page_blocks)} 列")

    pdf.close()

    doc.add_log("pdf_text_layer", f"提取完成，跳过 {skipped_page_count} 个非正文页", text_page_count)
    doc.add_log("furigana_filter", f"按字号阈值跳过振假名字符 {furigana_chars_skipped} 个", furigana_chars_skipped)
    doc.add_log("page_number_filter", f"跳过页码列 {page_number_cols_skipped} 处", page_number_cols_skipped)
    doc.add_log("chapter_detect", f"识别章节 {chapter_index} 个", chapter_index)

    if verbose:
        print(f"\n✅  完成: {text_page_count} 正文页，{chapter_index} 章节，"
              f"{len(doc.blocks)} 个块（已跳过 {furigana_chars_skipped} 个振假名字符）")

    return doc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PDF 文字层 → 统一文档模型 JSON")
    parser.add_argument("input_pdf", help="PDF 文件路径")
    parser.add_argument("output_json", help="输出 JSON 路径")
    parser.add_argument("--overrides", "-o", default=None)
    parser.add_argument("--furigana-threshold", type=float, default=FURIGANA_SIZE_THRESHOLD)
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    overrides = {}
    if args.overrides:
        import json
        with open(args.overrides, encoding="utf-8") as f:
            overrides = json.load(f)

    doc = extract_pdf_text_layer(
        args.input_pdf, page_overrides=overrides, verbose=not args.quiet,
        furigana_threshold=args.furigana_threshold,
    )

    with open(args.output_json, "w", encoding="utf-8") as f:
        f.write(doc.to_json())
    print(f"\n💾  已写入: {args.output_json}")
