#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF 文字层适配器。

职责刻意保持简单：只把 selectable PDF 的正文字符、几何和物理阅读顺序
可靠地提取成 UnifiedDocument，不做章节/目录识别，也不做语义改写。

只有两类必须依赖原始 PDF 字号/坐标才能安全判断的噪声在这里处理：
    - 小字号 Ruby / 振假名；
    - 几何确认的页脚 ASCII 页码。
另外保留 Phase35 已验证的“损坏 ToUnicode 字形恢复”：只有 CFF 字体家族签名
与 glyph name 同时命中时才恢复 U+FFFD，并且只对康熙部首/竖排表现字符做
目标化语义归一化，不对普通正文做全局 NFKC。

分页接续、物理列重组、前书/后书、站点生成页、缩进等都交给后续
``PDF 格式处理``；章节/TOC 则交给 AI。这样提取层只回答一件事：
“PDF 正文有没有完整、按真实坐标读出来？”

依赖：
    pip install pymupdf python-docx fonttools

用法：
    python pdf_text_layer.py input.pdf output.json
"""

from __future__ import annotations

import re
import sys
import gc
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Callable
import statistics
import unicodedata
import io

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.document import (
    UnifiedDocument, Block, BlockType, PageInfo, BoundingBox, Metadata
)



@contextmanager
def _suspend_automatic_gc():
    """Pause cyclic GC while building very large selectable-PDF documents.

    ``Block``/metadata objects are overwhelmingly acyclic and are released by
    reference counting.  On multi-thousand-page PDFs, CPython's periodic cyclic
    collector repeatedly walks the ever-growing document graph and can turn the
    final few hundred pages from seconds into minutes.  Pause only automatic
    cyclic collection for the page loop, then restore the caller's GC state.
    """
    was_enabled = gc.isenabled()
    if was_enabled:
        gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()

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
VOLUME_END_RE = re.compile(rf'^第[\s　]*{JP_CHAPTER_NUMBER}[\s　]*巻[\s　]*了$')
CHAPTER_RE = re.compile(
    rf'^(序章|終章|プロローグ|フロローグ|ブロローグ|エピローグ|後記|あとがき|'
    rf'幕間(?:[\s　:：・—―-].*)?|'
    rf'第[\s　]*{JP_CHAPTER_NUMBER}[\s　]*{CHAPTER_UNIT}(?!{CHAPTER_CONTINUATION})|'
    rf'{JP_CHAPTER_NUMBER}[\s　]*[話章節回](?=$|[\s　:：・—―「『【（(]|前編|後編|上編|中編|下編)|'
    rf'(?:Chapter|Episode|EP)[\s　.．_-]*[\d０-９]+)',
    re.IGNORECASE
)
DIALOGUE_START = ('「', '『', '﹁', '﹃')
DIALOGUE_END = ('」', '』', '﹂', '﹄')

ASCII_PAGE_NUM_RE = re.compile(r'^[0-9]{1,4}$')
_FOOTER_PAGE_NUMBER_Y_RATIO = 0.88
_FOOTER_PAGE_NUMBER_CENTER_RATIO = 0.16


# Some Japanese vertical-layout PDF generators embed presentation-form glyphs
# without a usable ToUnicode entry.  MuPDF then reports U+FFFD even though the
# CFF charset still preserves a stable glyph name.  The TT*o* subset family
# below is used by this class of PDFs and the names correspond to the visible
# vertical glyphs.  Recovery is enabled only after a font charset contains a
# strong signature from this family, so arbitrary CFF fonts with a coincidental
# gNNNN name are never rewritten.
_VERTICAL_SUBSET_GLYPH_NAME_MAP = {
    "g7893": "ー",
    "g7920": "ぁ",
    "g7921": "ぃ",
    "g7923": "ぇ",
    "g7925": "っ",
    "g7926": "ゃ",
    "g7927": "ゅ",
    "g7928": "ょ",
    "g7931": "ィ",
    "g7933": "ェ",
    "g7935": "ッ",
    "g7936": "ャ",
    "g7937": "ュ",
    "g7938": "ョ",
}
_VERTICAL_SUBSET_SIGNATURE = frozenset({"g7893", "g7925", "g7926", "g7935", "g7936"})


def _semantic_pdf_char(value: str) -> str:
    """Normalize only objectively presentational PDF Unicode forms.

    This deliberately does *not* run NFKC over arbitrary Japanese text.  It
    fixes only Kangxi radicals incorrectly exposed as text and Unicode vertical
    presentation forms such as ︑/︒/﹁/﹂.  Ordinary kanji, width choices and
    user text remain byte-for-byte untouched.
    """
    value = str(value or "")
    if not value:
        return value
    out: list[str] = []
    for ch in value:
        cp = ord(ch)
        decomp = unicodedata.decomposition(ch)
        if 0x2F00 <= cp <= 0x2FD5 or decomp.startswith("<vertical>"):
            normalized = unicodedata.normalize("NFKC", ch)
            out.append(normalized or ch)
        else:
            out.append(ch)
    return "".join(out)


def _build_cff_glyph_rescue_map(pdf) -> dict[tuple[str, int], tuple[str, str]]:
    """Return ``(trace_font, gid) -> (unicode, glyph_name)`` for safe CFF repairs.

    Recovery is validated at the *document family* level. Japanese publishing
    tools often subset the same source font into several TT...o.. CFF programs;
    a later subset may contain only one or two of the special vertical glyphs.
    Requiring every subset to carry the whole signature would leave many U+FFFD
    glyphs unresolved, so we first verify the union across all related subsets
    and then resolve known names in each member font.
    """
    try:
        from fontTools.cffLib import CFFFontSet
    except Exception:
        return {}

    seen_xrefs: set[int] = set()
    parsed_fonts: list[tuple[str, list[str]]] = []
    family_names: set[str] = set()
    for page in pdf:
        try:
            fonts = page.get_fonts(full=True)
        except Exception:
            continue
        for font_entry in fonts:
            try:
                xref = int(font_entry[0])
            except Exception:
                continue
            if xref <= 0 or xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            try:
                embedded_name, ext, _font_type, buffer = pdf.extract_font(xref)
                if str(ext or "").lower() != "cff" or not buffer:
                    continue
                cff = CFFFontSet()
                cff.decompile(io.BytesIO(buffer), None, False)
                if not cff.fontNames:
                    continue
                top = cff[cff.fontNames[0]]
                charset = list(top.charset or [])
                trace_font = str(embedded_name or cff.fontNames[0]).split("+")[-1]
                if not trace_font.startswith("TT") or "o" not in trace_font:
                    continue
                parsed_fonts.append((trace_font, charset))
                family_names.update(str(name) for name in charset)
            except Exception:
                continue

    if len(family_names & _VERTICAL_SUBSET_SIGNATURE) < 4:
        return {}

    mapping: dict[tuple[str, int], tuple[str, str]] = {}
    for trace_font, charset in parsed_fonts:
        for gid, glyph_name in enumerate(charset):
            repaired = _VERTICAL_SUBSET_GLYPH_NAME_MAP.get(str(glyph_name))
            if repaired:
                mapping[(trace_font, int(gid))] = (repaired, str(glyph_name))
    return mapping

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


def _is_page_number(
    chars: list[dict],
    page_height: float,
    page_width: float | None = None,
) -> bool:
    """Whether one residual track is an ASCII footer page number.

    Full-width digits are body text and must never be classified as page numbers.
    The footer band is intentionally narrower than the historical bottom-20%
    rule so colophon dates/times around 84% page height are preserved.
    """
    if not chars:
        return False
    text = "".join(str(ch.get("c", "")) for ch in chars).strip()
    if not ASCII_PAGE_NUM_RE.fullmatch(text):
        return False
    avg_y = sum(float(ch.get("y", 0.0)) for ch in chars) / len(chars)
    if avg_y <= float(page_height) * _FOOTER_PAGE_NUMBER_Y_RATIO:
        return False
    if page_width and page_width > 0:
        avg_x = sum(float(ch.get("x", 0.0)) for ch in chars) / len(chars)
        if abs(avg_x - float(page_width) / 2.0) > float(page_width) * _FOOTER_PAGE_NUMBER_CENTER_RATIO:
            return False
    return True




def _remove_footer_page_number_chars(
    chars: list[dict],
    page_width: float,
    page_height: float,
) -> tuple[list[dict], int]:
    """Remove isolated ASCII footer page numbers *before* track clustering.

    PDFNovels / タテ書き小説ネット places physical page numbers as a small
    horizontal ASCII digit run near the bottom centre of the page.  Filtering
    those glyphs before vertical-track clustering is important: otherwise one
    digit can land close enough to a body column to be merged into that column.

    Deliberately do **not** use ``\\d`` here.  Python's ``\\d`` also matches
    full-width Japanese digits (０-９), which are ordinary body text in these
    novels.  That was the root cause of conversions such as ``１７０cm -> ０cm``
    and ``５００枚 -> ００枚`` when a physical column ended after the leading
    digits.

    A candidate is removed only when all of these are true:
      * it is 1-4 ASCII digits;
      * glyph baselines are in the bottom 20% of the physical page;
      * the digits form a compact horizontal run on essentially one baseline;
      * the run is reasonably close to the page centre (the layout used by the
        supported PDFNovels family).

    Returns ``(kept_chars, removed_count)``.
    """
    if not chars or page_width <= 0 or page_height <= 0:
        return list(chars), 0

    footer = [
        ch for ch in chars
        if str(ch.get("c", "")) in "0123456789"
        and float(ch.get("y", 0.0)) > float(page_height) * _FOOTER_PAGE_NUMBER_Y_RATIO
    ]
    if not footer:
        return list(chars), 0

    # Cluster by near-identical baseline.  Footer digits in the target PDFs are
    # a normal horizontal run; body digits in vertical columns have distinct Y.
    footer.sort(key=lambda ch: (float(ch.get("y", 0.0)), float(ch.get("x", 0.0))))
    baseline_groups: list[list[dict]] = []
    for ch in footer:
        y = float(ch.get("y", 0.0))
        size = max(1.0, float(ch.get("size", 0.0) or 0.0))
        tolerance = max(0.75, min(3.0, size * 0.20))
        target = None
        for group in baseline_groups:
            gy = statistics.median(float(item.get("y", 0.0)) for item in group)
            if abs(y - gy) <= tolerance:
                target = group
                break
        if target is None:
            baseline_groups.append([ch])
        else:
            target.append(ch)

    remove_orders: set[int] = set()
    for group in baseline_groups:
        group = sorted(group, key=lambda ch: float(ch.get("x", 0.0)))
        run: list[dict] = []
        runs: list[list[dict]] = []
        for ch in group:
            if not run:
                run = [ch]
                continue
            prev = run[-1]
            prev_size = max(1.0, float(prev.get("size", 0.0) or 0.0))
            curr_size = max(1.0, float(ch.get("size", 0.0) or 0.0))
            max_gap = max(prev_size, curr_size) * 1.25
            if float(ch.get("x", 0.0)) - float(prev.get("x", 0.0)) <= max_gap:
                run.append(ch)
            else:
                runs.append(run)
                run = [ch]
        if run:
            runs.append(run)

        for digit_run in runs:
            text = "".join(str(ch.get("c", "")) for ch in digit_run)
            if not ASCII_PAGE_NUM_RE.fullmatch(text):
                continue
            xs = [float(ch.get("x", 0.0)) for ch in digit_run]
            sizes = [max(1.0, float(ch.get("size", 0.0) or 0.0)) for ch in digit_run]
            run_center = (min(xs) + max(xs) + statistics.median(sizes)) / 2.0
            if abs(run_center - page_width / 2.0) > page_width * _FOOTER_PAGE_NUMBER_CENTER_RATIO:
                continue
            for ch in digit_run:
                remove_orders.add(int(ch.get("source_order", -1)))

    if not remove_orders:
        return list(chars), 0
    kept = [ch for ch in chars if int(ch.get("source_order", -1)) not in remove_orders]
    return kept, len(chars) - len(kept)


def _page_orientation(data: dict) -> str:
    """按 PyMuPDF rawdict 的 line 级 wmode/dir + 字符几何判定页面书写方向。

    返回 ``"vertical"`` 或 ``"horizontal"``。这里故意把“单字符 line”视为
    *不确定*，而不是横排证据。很多日文竖排 PDF（尤其由排版软件逐字写入的
    PDF）会把每个竖排字形保存成 ``wmode=0 / dir=(1, 0)`` 的单字符 line。
    旧逻辑因此会把整页误判成横排，再按 Y 基线聚类，把十几条竖列逐字交错。

    只有 line 自身至少包含 3 个字符、且几何散布也支持横排时，才把它计入
    横排票数；否则保持竖排默认。这与 PyMuPDF 的坐标数据配合时，比盲信
    ``dir`` 标记更适合小说 PDF。
    """
    vert_chars = 0
    horiz_chars = 0
    ambiguous_chars = 0
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            chars = [
                ch
                for span in line.get("spans", [])
                for ch in span.get("chars", [])
                if str(ch.get("c", "")).strip()
            ]
            n = len(chars)
            if n == 0:
                continue
            if int(line.get("wmode", 0)) == 1:
                vert_chars += n
                continue
            direction = line.get("dir", (1, 0))
            if abs(direction[1]) > abs(direction[0]):
                vert_chars += n
                continue

            # ``wmode=0 / horizontal dir`` alone is not enough.  Require the
            # line's own geometry to confirm horizontal text.  Single-glyph
            # lines are common in vertical PDFs and carry no direction signal.
            if n < 3:
                ambiguous_chars += n
                continue
            xs = [float(ch["origin"][0]) for ch in chars]
            ys = [float(ch["origin"][1]) for ch in chars]
            x_span = max(xs) - min(xs)
            y_span = max(ys) - min(ys)
            if y_span > x_span * 1.25:
                vert_chars += n
            elif x_span > y_span * 1.25:
                horiz_chars += n
            else:
                ambiguous_chars += n

    # 横排必须有“可验证的多字符行”形成压倒性证据。大量单字 line 不会再
    # 把竖排小说误判为横排。若证据不足，沿用项目原本的竖排默认。
    reliable = vert_chars + horiz_chars
    if reliable > 0 and horiz_chars > max(1, vert_chars) * 2 and horiz_chars >= ambiguous_chars * 0.25:
        return "horizontal"
    return "vertical"


def _detect_block_type(text: str) -> BlockType:
    # '第一巻了' is a volume-end marker, not a chapter.
    text = str(text or '').strip()
    if VOLUME_END_RE.match(text):
        return BlockType.SECTION
    if CHAPTER_RE.match(text):
        return BlockType.CHAPTER

    # A quoted *term* at the beginning of a prose sentence is not dialogue.
    # Real-world vertical PDFs often put book/skill/place names such as
    # ``『豚の足亭』の一階`` or ``『射』を打ち放ち`` at a physical-column
    # boundary.  Treating every leading corner quote as speech breaks later
    # paragraph joining and can style ordinary narration as dialogue.
    for opener, closer in (("「", "」"), ("『", "』")):
        if text.startswith(opener):
            close_at = text.find(closer, 1)
            if close_at > 0 and text[close_at + 1 :].strip():
                return BlockType.PARAGRAPH

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

        def _track_inline_key(ch: dict):
            coord = float(ch[inline])
            if not vertical:
                return (coord, int(ch.get("source_order", 0)))

            # Vertical Japanese PDFs frequently encode punctuation such as ``!?``
            # as a tiny horizontal text run placed at the *same* Y anchor as the
            # preceding vertical glyph.  MuPDF may report the punctuation a few
            # ten-thousandths of a point earlier, which used to yield ``!?血``
            # instead of the visibly printed ``血!?``.  Bucket only near-identical
            # anchors (0.01 pt) and put native vertical glyphs before rotated runs.
            direction = ch.get("line_dir", (1.0, 0.0))
            native_vertical = int(ch.get("line_wmode", 0) or 0) == 1 or abs(float(direction[1])) > abs(float(direction[0]))
            orientation_priority = 0 if native_vertical else 1
            return (round(coord, 2), orientation_priority, coord, int(ch.get("source_order", 0)))

        track.sort(key=_track_inline_key)
        result.append(track)
    return result



def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge 1D intervals; used for whitespace-band detection on a PDF page."""
    if not intervals:
        return []
    ordered = sorted((float(a), float(b)) for a, b in intervals if float(b) >= float(a))
    merged: list[list[float]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(a, b) for a, b in merged]


def _detect_stacked_page_split(
    chars: list[dict],
    page_width: float,
    page_height: float,
) -> dict | None:
    """Detect a physical PDF page that contains two logical pages stacked vertically.

    The detector is intentionally conservative.  It looks for a *global* empty
    horizontal band close to the physical page midpoint, then requires substantial
    text on both sides and strong X-range overlap.  This is different from looking
    for a paragraph gap inside one column: the whitespace must be empty across the
    whole page's retained text layer.

    Returned coordinates stay in the original PDF coordinate system so later
    extraction can preserve provenance exactly.  ``split_y`` is the middle of the
    whitespace band and is therefore safe even when the upper/lower margins differ.
    """
    if not chars or page_height <= 0 or page_width <= 0:
        return None

    usable = [
        ch for ch in chars
        if str(ch.get("c", "")).strip()
        and "y0" in ch and "y1" in ch and "x0" in ch and "x1" in ch
    ]
    if len(usable) < 80:
        return None

    sizes = [float(ch.get("size", 0) or 0) for ch in usable if float(ch.get("size", 0) or 0) > 0]
    median_size = statistics.median(sizes) if sizes else 10.0
    min_gap = max(6.0, median_size * 0.72)

    # Ignore extreme top/bottom whitespace.  Only a center band can mean a 2-up
    # vertical stack; footer/page-number gaps are deliberately outside this window.
    candidates: list[tuple[float, float, float, float]] = []
    occupied = _merge_intervals([(ch["y0"], ch["y1"]) for ch in usable])
    for (_, prev_end), (next_start, _) in zip(occupied, occupied[1:]):
        gap = float(next_start) - float(prev_end)
        if gap < min_gap:
            continue
        split_y = (float(prev_end) + float(next_start)) / 2.0
        ratio = split_y / page_height
        if not (0.42 <= ratio <= 0.58):
            continue
        candidates.append((gap, split_y, float(prev_end), float(next_start)))
    if not candidates:
        return None

    # Prefer the widest center gap; midpoint closeness is the tie breaker.
    candidates.sort(key=lambda item: (item[0], -abs(item[1] / page_height - 0.5)), reverse=True)
    gap, split_y, gap_top, gap_bottom = candidates[0]

    top = [ch for ch in usable if float(ch.get("y", (ch["y0"] + ch["y1"]) / 2)) < split_y]
    bottom = [ch for ch in usable if float(ch.get("y", (ch["y0"] + ch["y1"]) / 2)) >= split_y]
    if not top or not bottom:
        return None

    min_side_chars = max(40, int(len(usable) * 0.18))
    if len(top) < min_side_chars or len(bottom) < min_side_chars:
        return None

    balance = min(len(top), len(bottom)) / max(len(top), len(bottom))
    if balance < 0.35:
        return None

    def _x_span(group: list[dict]) -> tuple[float, float]:
        return min(float(ch["x0"]) for ch in group), max(float(ch["x1"]) for ch in group)

    top_x0, top_x1 = _x_span(top)
    bot_x0, bot_x1 = _x_span(bottom)
    overlap = max(0.0, min(top_x1, bot_x1) - max(top_x0, bot_x0))
    overlap_ratio = overlap / max(1.0, min(top_x1 - top_x0, bot_x1 - bot_x0))
    if overlap_ratio < 0.55:
        return None

    # Both halves should occupy a meaningful vertical extent.  This blocks false
    # positives such as a chapter title above a single normal body page.
    top_y0 = min(float(ch["y0"]) for ch in top)
    top_y1 = max(float(ch["y1"]) for ch in top)
    bot_y0 = min(float(ch["y0"]) for ch in bottom)
    bot_y1 = max(float(ch["y1"]) for ch in bottom)
    top_extent = top_y1 - top_y0
    bot_extent = bot_y1 - bot_y0
    if min(top_extent, bot_extent) < page_height * 0.18:
        return None

    return {
        "split_y": split_y,
        "gap_top": gap_top,
        "gap_bottom": gap_bottom,
        "gap": gap,
        "balance": balance,
        "x_overlap_ratio": overlap_ratio,
        "top_chars": len(top),
        "bottom_chars": len(bottom),
        "median_font_size": median_size,
    }


def _logical_regions_for_page(
    chars: list[dict],
    page_width: float,
    page_height: float,
    split_mode: str,
) -> tuple[list[dict], dict | None]:
    """Return physical-source regions in logical reading order (top, then bottom)."""
    mode = str(split_mode or "auto").strip().lower()
    if mode not in {"auto", "off", "force"}:
        raise ValueError(f"未知 PDF 上下双页模式: {split_mode}")
    if mode == "off":
        return ([{"part": "full", "y0": 0.0, "y1": page_height}], None)

    detected = _detect_stacked_page_split(chars, page_width, page_height)
    if detected is None and mode == "force":
        # Forced mode still uses the geometric midpoint only as a fallback.  It
        # is never used by the default UI, but provides an explicit escape hatch
        # for unusual 2-up PDFs whose text layer has no clean center whitespace.
        detected = {
            "split_y": page_height / 2.0,
            "gap_top": page_height / 2.0,
            "gap_bottom": page_height / 2.0,
            "gap": 0.0,
            "balance": 0.0,
            "x_overlap_ratio": 0.0,
            "top_chars": sum(1 for ch in chars if float(ch.get("y", 0.0)) < page_height / 2.0),
            "bottom_chars": sum(1 for ch in chars if float(ch.get("y", 0.0)) >= page_height / 2.0),
            "median_font_size": 0.0,
            "forced": True,
        }
    if detected is None:
        return ([{"part": "full", "y0": 0.0, "y1": page_height}], None)

    split_y = float(detected["split_y"])
    return (
        [
            {"part": "top", "y0": 0.0, "y1": split_y},
            {"part": "bottom", "y0": split_y, "y1": page_height},
        ],
        detected,
    )


def _region_chars(chars: list[dict], y0: float, y1: float) -> list[dict]:
    """Project source chars into one logical page while preserving source coords."""
    result: list[dict] = []
    for ch in chars:
        cy = float(ch.get("y", 0.0))
        if not (float(y0) <= cy < float(y1)):
            continue
        projected = dict(ch)
        projected["source_x"] = float(ch.get("x", 0.0))
        projected["source_y"] = cy
        projected["x"] = float(ch.get("x", 0.0))
        projected["y"] = cy - float(y0)
        projected["x0"] = float(ch.get("x0", ch.get("x", 0.0)))
        projected["x1"] = float(ch.get("x1", ch.get("x", 0.0)))
        projected["y0"] = float(ch.get("y0", cy)) - float(y0)
        projected["y1"] = float(ch.get("y1", cy)) - float(y0)
        result.append(projected)
    return result


def extract_pdf_text_layer(
    pdf_path: str,
    page_overrides: dict[int, str] | None = None,
    verbose: bool = True,
    furigana_threshold: float = FURIGANA_SIZE_THRESHOLD,
    col_width: float = COL_WIDTH,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    split_stacked_pages: str = "auto",
    cancel_check: Optional[Callable[[], bool]] = None,
) -> UnifiedDocument:
    """
    从带文字层的 PDF 直接、忠实地提取正文字符和几何。
    竖排按右→左物理列、列内上→下；横排按上→下行、行内左→右。
    不在这一层识别章节/目录，也不做句子/标点修复。

    对“一个 PDF 物理页里上下叠放两个逻辑页”的 2-up 文件，默认先检测页面
    中央的全宽空白带，再把上下两块分别当成逻辑页提取。这样不会把相同 X
    基线的上下两页文字合并进同一竖列，也不依赖固定 ``height / 2`` 硬切。

    Args:
        pdf_path: PDF 文件路径
        page_overrides: {物理页码: BlockType字符串} —— 非"正文"的页（封面/插图/
                        目录扫描页等）直接跳过文字层提取。页码始终指原 PDF 的
                        物理页，不受上下双页拆分影响。
        verbose: 是否打印进度
        furigana_threshold: 字号小于此值的字符视为振假名，提取时跳过
        col_width: 兼容参数；仅作为同一文字列基线漂移容差的上限提示
        progress_callback: (current_physical_page, total_physical_pages, label) -> None
        split_stacked_pages: ``auto``（默认，保守检测）/ ``off`` / ``force``。
                            ``force`` 仅供特殊文件显式使用；自动模式不会硬切。
        cancel_check: 可选的协作式取消检查；返回 True 时在物理页边界停止。

    Returns:
        UnifiedDocument。``PageInfo.page_no`` 和 ``Block.page`` 使用逻辑页号；
        原 PDF 物理页与 top/bottom/full 的映射保存在 metadata 中。
    """
    try:
        import fitz
    except ImportError:
        raise ImportError("请安装 PyMuPDF: pip install pymupdf")

    src = Path(pdf_path)
    if not src.exists():
        raise FileNotFoundError(f"文件不存在: {pdf_path}")

    split_mode = str(split_stacked_pages or "auto").strip().lower()
    if split_mode not in {"auto", "off", "force"}:
        raise ValueError(f"未知 PDF 上下双页模式: {split_stacked_pages}")

    overrides = {int(k): v for k, v in (page_overrides or {}).items()}

    pdf = fitz.open(pdf_path)
    doc = UnifiedDocument()
    doc.metadata = Metadata(
        source_engine="pdf_text_layer", language="ja", pdf_text_layer_mode=True
    )

    order_counter = 0
    text_page_count = 0
    skipped_page_count = 0
    furigana_chars_skipped = 0
    page_number_cols_skipped = 0
    page_number_chars_skipped = 0
    stacked_page_count = 0
    page_map: dict[str, dict] = {}
    split_reports: dict[str, dict] = {}
    replacement_glyph_count = 0
    replacement_glyph_pages: dict[str, int] = {}
    auto_glyph_repair_count = 0
    auto_glyph_repair_pages: dict[str, int] = {}
    semantic_normalization_count = 0
    cff_glyph_rescue = _build_cff_glyph_rescue_map(pdf)

    total = len(pdf)
    if verbose:
        print(f"📂  PDF 文字层提取：共 {total} 个物理页")

    with _suspend_automatic_gc():
        for page_idx, page in enumerate(pdf):
            if cancel_check is not None and cancel_check():
                pdf.close()
                raise InterruptedError("PDF 文字层提取已停止")
            physical_page_no = page_idx + 1
            page_w = float(page.rect.width)
            page_height = float(page.rect.height)

            override_type = overrides.get(physical_page_no)
            if override_type and override_type != BlockType.PARAGRAPH.value:
                # 非正文页保持一个逻辑页；不对封面/插图/奥付等做自动 2-up 拆分。
                try:
                    ptype = BlockType(override_type)
                except ValueError:
                    ptype = BlockType.UNKNOWN
                logical_page_no = len(doc.pages) + 1
                doc.pages.append(PageInfo(
                    page_no=logical_page_no,
                    page_type=ptype,
                    width=int(round(page_w)),
                    height=int(round(page_height)),
                    confidence=1.0,
                ))
                page_map[str(logical_page_no)] = {
                    "physical_page": physical_page_no,
                    "part": "full",
                    "source_rect": [0.0, 0.0, page_w, page_height],
                    "skipped_type": override_type,
                }
                skipped_page_count += 1
                if verbose:
                    print(f"  [{physical_page_no:3d}/{total}] 跳过提取（已标注为 {override_type}）")
                if progress_callback is not None:
                    progress_callback(physical_page_no, total, f"跳过（{override_type}）")
                continue

            data = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

            # MuPDF rawdict exposes the broken Unicode value (U+FFFD) and geometry,
            # while get_texttrace() also exposes the embedded-font name + glyph id.
            # Keep that identity only for objectively damaged glyphs so repeated
            # occurrences can be reviewed once and expanded safely without storing
            # every character from the page.
            replacement_identity_by_bbox: dict[tuple[float, float, float, float], dict] = {}
            needs_texttrace = any(
                str(ch.get("c", "")) == "\ufffd"
                for raw_block in data.get("blocks", [])
                for raw_line in raw_block.get("lines", [])
                for raw_span in raw_line.get("spans", [])
                for ch in raw_span.get("chars", [])
            )
            if needs_texttrace:
                if cancel_check is not None and cancel_check():
                    pdf.close()
                    raise InterruptedError("PDF 文字层提取已停止")
                try:
                    for trace_span in page.get_texttrace():
                        trace_font = str(trace_span.get("font", "") or "")
                        for trace_char in trace_span.get("chars", ()):
                            if len(trace_char) < 4:
                                continue
                            codepoint, glyph_id, _origin, trace_bbox = trace_char[:4]
                            if int(codepoint) != 0xFFFD:
                                continue
                            gid = int(glyph_id)
                            key = tuple(round(float(v), 4) for v in trace_bbox)
                            repaired, glyph_name = cff_glyph_rescue.get((trace_font, gid), ("", ""))
                            replacement_identity_by_bbox[key] = {
                                "font": trace_font,
                                "glyph_id": gid,
                                "glyph_name": glyph_name,
                                "repaired": repaired,
                            }
                except Exception:
                    replacement_identity_by_bbox = {}

            chars: list[dict] = []
            source_order = 0
            for block_index, block in enumerate(data.get("blocks", [])):
                for line_index, line in enumerate(block.get("lines", [])):
                    for span_index, span in enumerate(line.get("spans", [])):
                        font_size = float(span.get("size", 12) or 12)
                        if font_size < furigana_threshold:
                            furigana_chars_skipped += len(span.get("chars", []))
                            continue
                        for char in span.get("chars", []):
                            raw_c = str(char.get("c", ""))
                            if not raw_c.strip():
                                continue
                            origin = char.get("origin", (0.0, 0.0))
                            bbox = char.get("bbox") or (origin[0], origin[1], origin[0], origin[1])
                            direction = line.get("dir", (1.0, 0.0))
                            c = _semantic_pdf_char(raw_c)
                            if c != raw_c:
                                semantic_normalization_count += 1
                            char_record = {
                                "c": c,
                                "pdf_raw_c": raw_c,
                                "x": float(origin[0]),
                                "y": float(origin[1]),
                                "x0": float(bbox[0]),
                                "y0": float(bbox[1]),
                                "x1": float(bbox[2]),
                                "y1": float(bbox[3]),
                                "size": font_size,
                                # Keep the source line orientation so mixed vertical
                                # body glyphs + rotated horizontal punctuation at the
                                # same visual anchor can be ordered deterministically.
                                "line_wmode": int(line.get("wmode", 0) or 0),
                                "line_dir": (float(direction[0]), float(direction[1])),
                                # Stable tiebreak only. Reading order comes from geometry.
                                "source_order": source_order,
                                "block_index": block_index,
                                "line_index": line_index,
                                "span_index": span_index,
                            }
                            if raw_c == "\ufffd":
                                identity = replacement_identity_by_bbox.get(
                                    tuple(round(float(v), 4) for v in bbox)
                                )
                                if identity is not None:
                                    char_record["pdf_font"] = str(identity.get("font") or "")
                                    char_record["pdf_glyph_id"] = int(identity.get("glyph_id", 0))
                                    glyph_name = str(identity.get("glyph_name") or "")
                                    if glyph_name:
                                        char_record["pdf_glyph_name"] = glyph_name
                                    repaired = str(identity.get("repaired") or "")
                                    if repaired:
                                        char_record["c"] = repaired
                                        char_record["pdf_text_auto_repaired"] = True
                                        auto_glyph_repair_count += 1
                                        page_key = str(physical_page_no)
                                        auto_glyph_repair_pages[page_key] = auto_glyph_repair_pages.get(page_key, 0) + 1
                            chars.append(char_record)
                            source_order += 1

            chars, removed_footer_digits = _remove_footer_page_number_chars(
                chars, page_w, page_height
            )
            page_number_chars_skipped += removed_footer_digits

            orientation = _page_orientation(data)
            regions, split_report = _logical_regions_for_page(
                chars, page_w, page_height, split_mode
            )
            if split_report is not None:
                stacked_page_count += 1
                split_reports[str(physical_page_no)] = dict(split_report)

            region_summaries: list[str] = []
            for region in regions:
                part = str(region["part"])
                region_y0 = float(region["y0"])
                region_y1 = float(region["y1"])
                region_height = max(1.0, region_y1 - region_y0)
                logical_page_no = len(doc.pages) + 1
                logical_chars = _region_chars(chars, region_y0, region_y1)
                tracks = _cluster_text_tracks(logical_chars, orientation, col_width)
                page_blocks: list[Block] = []

                for group in tracks:
                    if _is_page_number(group, region_height, page_w):
                        page_number_cols_skipped += 1
                        continue

                    col_text = "".join(ch["c"] for ch in group).strip()
                    if not col_text:
                        continue

                    xs = [float(ch["x"]) for ch in group]
                    ys = [float(ch["y"]) for ch in group]
                    avg_size = sum(float(ch["size"]) for ch in group) / len(group)
                    bbox = BoundingBox(
                        x=max(0.0, min(xs)) / page_w if page_w else 0.0,
                        y=max(0.0, min(ys)) / region_height if region_height else 0.0,
                        w=(max(xs) - min(xs) + avg_size) / page_w if page_w else 0.0,
                        h=(max(ys) - min(ys) + avg_size) / region_height if region_height else 0.0,
                    )

                    source_xs = [float(ch.get("source_x", ch["x"])) for ch in group]
                    source_ys = [float(ch.get("source_y", ch["y"] + region_y0)) for ch in group]
                    # 提取层不判断章节/对白等语义结构；统一保留为正文物理块。
                    # 后续 PDF 格式处理只负责版面，章节/TOC 最终交给 AI。
                    btype = BlockType.PARAGRAPH
                    block_metadata = {
                        "pdf_source_physical_page": physical_page_no,
                        "pdf_logical_part": part,
                        "pdf_source_region": [0.0, region_y0, page_w, region_y1],
                        "pdf_source_bbox": {
                            "x0": min(source_xs),
                            "y0": min(source_ys),
                            "x1": max(source_xs) + avg_size,
                            "y1": max(source_ys) + avg_size,
                        },
                    }

                    # Keep source coordinates only for objectively damaged glyphs.
                    # Storing every character bbox would bloat workspaces by tens of
                    # thousands of entries; U+FFFD locations are rare enough to keep
                    # and are exactly the places where GPT / OCR visual rescue is useful.
                    replacement_entries = []
                    for text_index, ch in enumerate(group):
                        if text_index >= len(col_text):
                            break
                        if str(ch.get("c", "")) != "\ufffd":
                            continue
                        source_y0 = float(ch.get("y0", ch.get("y", 0.0))) + region_y0
                        source_y1 = float(ch.get("y1", ch.get("y", 0.0))) + region_y0
                        entry = {
                            "text_index": text_index,
                            "source_bbox": [
                                float(ch.get("x0", ch.get("x", 0.0))),
                                source_y0,
                                float(ch.get("x1", ch.get("x", 0.0))),
                                source_y1,
                            ],
                        }
                        pdf_font = str(ch.get("pdf_font", "") or "")
                        glyph_id = ch.get("pdf_glyph_id")
                        if pdf_font and glyph_id is not None:
                            entry["pdf_font"] = pdf_font
                            entry["pdf_glyph_id"] = int(glyph_id)
                            entry["pdf_glyph_key"] = f"{pdf_font}:{int(glyph_id)}"
                        replacement_entries.append(entry)
                    if replacement_entries:
                        block_metadata["pdf_text_replacement_glyphs"] = replacement_entries
                        replacement_glyph_count += len(replacement_entries)
                        key = str(physical_page_no)
                        replacement_glyph_pages[key] = replacement_glyph_pages.get(key, 0) + len(replacement_entries)

                    b = Block(
                        type=btype,
                        text=col_text,
                        ocr_raw=col_text,
                        page=logical_page_no,
                        bbox=bbox,
                        reading_order=order_counter + len(page_blocks),
                        confidence=1.0,
                        text_direction="vertical-rl" if orientation == "vertical" else "horizontal-tb",
                        source_format="pdf_text_layer",
                        metadata=block_metadata,
                    )
                    page_blocks.append(b)

                doc.blocks.extend(page_blocks)
                order_counter += len(page_blocks)

                ptype = BlockType.PARAGRAPH if page_blocks else BlockType.BLANK
                doc.pages.append(PageInfo(
                    page_no=logical_page_no,
                    page_type=ptype,
                    width=int(round(page_w)),
                    height=int(round(region_height)),
                    confidence=1.0,
                ))
                page_map[str(logical_page_no)] = {
                    "physical_page": physical_page_no,
                    "part": part,
                    "orientation": orientation,
                    "source_rect": [0.0, region_y0, page_w, region_y1],
                }
                if split_report is not None:
                    page_map[str(logical_page_no)]["split_y"] = float(split_report["split_y"])
                if page_blocks:
                    text_page_count += 1
                region_summaries.append(f"{part}:{len(page_blocks)}")

            if verbose:
                split_note = " · 自动上下双页" if split_report is not None else ""
                print(
                    f"  [{physical_page_no:3d}/{total}] → "
                    f"{'+'.join(region_summaries)} 块 · {orientation}{split_note}"
                )
            if progress_callback is not None:
                if split_report is not None:
                    label = f"上下双页 → {region_summaries[0]} / {region_summaries[1]}"
                else:
                    label = f"{region_summaries[0] if region_summaries else 'full:0'}"
                progress_callback(physical_page_no, total, label)

    pdf.close()

    doc.metadata.pdf_text_split_mode = split_mode
    doc.metadata.pdf_text_physical_page_count = total
    doc.metadata.pdf_text_logical_page_count = len(doc.pages)
    doc.metadata.pdf_text_stacked_page_count = stacked_page_count
    doc.metadata.pdf_text_page_map = page_map
    doc.metadata.pdf_text_stacked_split_reports = split_reports
    doc.metadata.pdf_text_page_overrides = {str(k): str(v) for k, v in sorted(overrides.items())}
    doc.metadata.pdf_text_skipped_physical_pages = [
        int(page_no) for page_no, page_type in sorted(overrides.items())
        if str(page_type) != BlockType.PARAGRAPH.value
    ]
    doc.metadata.pdf_text_furigana_chars_skipped = furigana_chars_skipped
    doc.metadata.pdf_text_page_number_chars_skipped = page_number_chars_skipped
    doc.metadata.pdf_text_page_number_columns_skipped = page_number_cols_skipped
    extracted_char_count = sum(len(str(block.text or "")) for block in doc.blocks)
    doc.metadata.pdf_text_extracted_char_count = extracted_char_count
    doc.metadata.pdf_text_replacement_glyph_count = replacement_glyph_count
    doc.metadata.pdf_text_replacement_glyph_rate = (
        replacement_glyph_count / extracted_char_count if extracted_char_count else 0.0
    )
    doc.metadata.pdf_text_replacement_glyph_pages = dict(sorted(
        replacement_glyph_pages.items(), key=lambda item: int(item[0])
    ))
    doc.metadata.pdf_text_auto_glyph_repair_count = auto_glyph_repair_count
    doc.metadata.pdf_text_auto_glyph_repair_pages = dict(sorted(
        auto_glyph_repair_pages.items(), key=lambda item: int(item[0])
    ))
    doc.metadata.pdf_text_semantic_normalization_count = semantic_normalization_count

    doc.add_log(
        "pdf_text_layer",
        f"提取完成：{total} 个物理页 → {len(doc.pages)} 个逻辑页；跳过 {skipped_page_count} 个非正文页",
        text_page_count,
    )
    doc.add_log(
        "pdf_stacked_page_split",
        f"上下双页自动检测 {stacked_page_count} 个物理页（模式 {split_mode}）",
        stacked_page_count,
    )
    doc.add_log("furigana_filter", f"按字号阈值跳过振假名字符 {furigana_chars_skipped} 个", furigana_chars_skipped)
    doc.add_log(
        "page_number_filter",
        f"跳过页码字符 {page_number_chars_skipped} 个；残余独立页码列 {page_number_cols_skipped} 处",
        page_number_chars_skipped + page_number_cols_skipped,
    )
    doc.add_log("chapter_detect", "PDF文字层提取阶段不识别章节/目录；交由后续 AI", 0)
    if auto_glyph_repair_count:
        doc.add_log(
            "pdf_text_font_glyph_repair",
            f"从嵌入 CFF 竖排字形恢复 {auto_glyph_repair_count} 个缺失 Unicode 字符",
            auto_glyph_repair_count,
        )
    if semantic_normalization_count:
        doc.add_log(
            "pdf_text_semantic_unicode",
            f"语义还原 {semantic_normalization_count} 个康熙部首/竖排表现字符",
            semantic_normalization_count,
        )
    if replacement_glyph_count:
        doc.add_log(
            "pdf_text_damage",
            f"原 PDF 文字层含 {replacement_glyph_count} 个 U+FFFD 替换字形 "
            f"({doc.metadata.pdf_text_replacement_glyph_rate:.2%})；已保留原页坐标供视觉复核",
            replacement_glyph_count,
        )

    if verbose:
        print(
            f"\n✅  完成: {total} 物理页 → {len(doc.pages)} 逻辑页；"
            f"{text_page_count} 正文逻辑页，{len(doc.blocks)} 个物理文字块；章节/目录交由 AI；"
            f"自动拆分 {stacked_page_count} 个物理页"
        )

    return doc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PDF 文字层 → 统一文档模型 JSON")
    parser.add_argument("input_pdf", help="PDF 文件路径")
    parser.add_argument("output_json", help="输出 JSON 路径")
    parser.add_argument("--overrides", "-o", default=None)
    parser.add_argument("--furigana-threshold", type=float, default=FURIGANA_SIZE_THRESHOLD)
    parser.add_argument("--stacked-pages", choices=("auto", "off", "force"), default="auto",
                        help="上下双页文字层处理：auto=自动检测，off=关闭，force=强制按上下拆分")
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
        split_stacked_pages=args.stacked_pages,
    )

    with open(args.output_json, "w", encoding="utf-8") as f:
        f.write(doc.to_json())
    print(f"\n💾  已写入: {args.output_json}")
