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
    - 按真实 X 基线自适应分列，列内按 Y 排序 —— 处理竖排日文
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
from pathlib import Path
from typing import Optional, Callable
import statistics

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.document import (
    UnifiedDocument, Block, BlockType, PageInfo, BoundingBox, Metadata, TocEntry
)

COL_WIDTH = 20
FURIGANA_SIZE_THRESHOLD = 8.0

# ``COL_WIDTH`` used to be used directly as ``round(axis / COL_WIDTH)``.  That
# is unsafe for real Japanese vertical PDFs: this book, for example, has body
# columns about 15.735 pt apart, so a 20 pt bucket periodically collapses two
# adjacent physical columns into one and then interleaves their glyphs by Y.
# Keep the public parameter for compatibility, but use it only as an upper
# bound for *same-track baseline drift*.  Actual tracks are clustered around
# their measured origins.
_MIN_TRACK_TOLERANCE = 0.75
_MAX_TRACK_TOLERANCE = 5.0
_FONT_TRACK_TOLERANCE_RATIO = 0.42
_COL_WIDTH_TOLERANCE_RATIO = 0.22

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


def _page_orientation(data: dict) -> str:
    """按 PyMuPDF rawdict 的 line 级 wmode/dir + 字符几何判定页面书写方向。

    返回 "vertical" 或 "horizontal"。判定极其保守：默认竖排（与旧行为一致），
    只有 wmode/dir 与字符几何两类证据都强烈指向横排时才切换——避免把
    "标了横排 dir 但实际按竖排逐字定位"的 PDF 误判而破坏原有正确输出。
    """
    vert_chars = 0
    horiz_chars = 0
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            n = sum(len(s.get("chars", [])) for s in line.get("spans", []))
            if n == 0:
                continue
            if int(line.get("wmode", 0)) == 1:
                vert_chars += n
                continue
            direction = line.get("dir", (1, 0))
            dir_vertical = abs(direction[1]) > abs(direction[0])
            if dir_vertical:
                vert_chars += n
                continue
            # wmode=0 且 dir 水平：用该行字符实际散布方向佐证
            xs, ys = [], []
            for span in line.get("spans", []):
                for ch in span.get("chars", []):
                    xs.append(ch["origin"][0])
                    ys.append(ch["origin"][1])
            if len(xs) >= 3 and (max(ys) - min(ys)) > (max(xs) - min(xs)):
                vert_chars += n  # 几何上是竖列，dir 标注不可信
            else:
                horiz_chars += n
    # 横排需要压倒性多数（2:1）才切换；否则保持旧竖排路径
    if horiz_chars > max(1, vert_chars) * 2:
        return "horizontal"
    return "vertical"


def _detect_block_type(text: str) -> BlockType:
    if CHAPTER_RE.match(text):
        return BlockType.CHAPTER
    if text.startswith(DIALOGUE_START) or text.endswith(DIALOGUE_END):
        return BlockType.DIALOGUE
    return BlockType.PARAGRAPH


def _track_tolerance(chars: list[dict], col_width: float) -> float:
    """Return a conservative baseline-drift tolerance for one physical track.

    Vertical Japanese punctuation is not always placed on exactly the same X
    origin as ordinary glyphs.  In the supplied PDF, for example, ``!`` can be
    shifted by about 2.95 pt while the neighbouring *column* is 15.735 pt away.
    A tolerance derived from the actual font size keeps such punctuation in its
    column without ever treating the full column pitch as a grouping bucket.
    """
    sizes = [float(ch.get("size", 0) or 0) for ch in chars if float(ch.get("size", 0) or 0) > 0]
    median_size = statistics.median(sizes) if sizes else 10.0
    width_hint = abs(float(col_width or COL_WIDTH))
    tolerance = min(
        median_size * _FONT_TRACK_TOLERANCE_RATIO,
        width_hint * _COL_WIDTH_TOLERANCE_RATIO if width_hint else _MAX_TRACK_TOLERANCE,
        _MAX_TRACK_TOLERANCE,
    )
    return max(_MIN_TRACK_TOLERANCE, tolerance)


def _cluster_text_tracks(
    chars: list[dict],
    orientation: str,
    col_width: float = COL_WIDTH,
) -> list[list[dict]]:
    """Cluster glyphs into real physical columns/rows without interleaving.

    The old implementation snapped every origin to a fixed 20 pt grid.  Grid
    phase is arbitrary, so even perfectly regular 15.7 pt columns can collide
    in the same rounded bucket.  This routine instead clusters neighbouring
    baselines by *distance*.  It deliberately tolerates only small within-track
    shifts (vertical ``!?`` / rotated punctuation), never a whole column pitch.

    Reading order remains unchanged:
      * vertical page: right -> left tracks, top -> bottom glyphs;
      * horizontal page: top -> bottom tracks, left -> right glyphs.
    """
    if not chars:
        return []

    vertical = orientation != "horizontal"
    axis = "x" if vertical else "y"
    inline = "y" if vertical else "x"
    reverse_tracks = vertical
    tolerance = _track_tolerance(chars, col_width)

    # First pass: collect nearby origins into baseline clusters.  Median is used
    # rather than a rounded grid so the result is independent of page offset /
    # crop box and robust against a few shifted punctuation glyphs.
    ordered = sorted(
        chars,
        key=lambda ch: (
            -float(ch[axis]) if reverse_tracks else float(ch[axis]),
            float(ch[inline]),
            int(ch.get("source_order", 0)),
        ),
    )
    clusters: list[dict] = []
    for ch in ordered:
        pos = float(ch[axis])
        best_index = -1
        best_distance = tolerance + 1.0
        # Pages normally have only a few dozen tracks.  Scan all existing
        # clusters rather than relying on insertion proximity; this keeps the
        # result correct even for unusual shifted/rotated punctuation.
        for idx in range(len(clusters)):
            distance = abs(pos - float(clusters[idx]["center"]))
            if distance <= tolerance and distance < best_distance:
                best_index = idx
                best_distance = distance
        if best_index < 0:
            clusters.append({"center": pos, "axis_values": [pos], "chars": [ch]})
        else:
            cluster = clusters[best_index]
            cluster["chars"].append(ch)
            cluster["axis_values"].append(pos)
            cluster["center"] = statistics.median(cluster["axis_values"])

    clusters.sort(key=lambda item: float(item["center"]), reverse=reverse_tracks)
    result: list[list[dict]] = []
    for cluster in clusters:
        track = list(cluster["chars"])
        track.sort(key=lambda ch: (float(ch[inline]), int(ch.get("source_order", 0))))
        result.append(track)
    return result


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
        col_width: 兼容参数；仅作为同一文字列基线漂移容差的上限提示，不再用于固定网格分桶
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
    doc.metadata = Metadata(
        source_engine="pdf_text_layer", language="ja", pdf_text_layer_mode=True
    )

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
        source_order = 0
        for block_index, block in enumerate(data.get("blocks", [])):
            for line_index, line in enumerate(block.get("lines", [])):
                for span_index, span in enumerate(line.get("spans", [])):
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
                                # Stable tiebreak only.  Reading order continues
                                # to come from geometry, not PDF object order.
                                "source_order": source_order,
                                "block_index": block_index,
                                "line_index": line_index,
                                "span_index": span_index,
                            })
                            source_order += 1

        # wmode/dir 硬化的方向判定：竖排页走右→左、列内上→下；横排页走
        # 上→下、行内左→右。关键变化：不再用固定 20pt 网格分桶，而是按
        # 实际字符基线自适应聚类，避免相邻物理列被错误交错。
        orientation = _page_orientation(data)
        tracks = _cluster_text_tracks(chars, orientation, col_width)

        page_blocks: list[Block] = []
        page_w = page.rect.width

        for group in tracks:

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
