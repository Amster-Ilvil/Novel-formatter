#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Loss-aware Formatter helpers for selectable PDF text layers.

The selectable-PDF path is deliberately isolated from image OCR.  A PDF text
layer usually contains correct glyphs but exposes *physical columns* rather
than logical paragraphs.  Therefore the safe order is:

1. preserve every source block and classify structural markers;
2. repair only deterministic quote reordering;
3. join physical columns/short tails without inventing punctuation;
4. let the ordinary Formatter work on the reconstructed logical blocks;
5. run PDF-safe cleanup/dedup/normalisation and verify character coverage.

No function in this module is used unless ``Metadata.pdf_text_layer_mode`` is
true, so Apple Vision/PaddleOCR behaviour stays unchanged.
"""
from __future__ import annotations

import copy
import re
import unicodedata
from collections import Counter
from typing import Iterable

from models.document import Block, BlockType, BoundingBox, UnifiedDocument

_TEXT_TYPES = {
    BlockType.PARAGRAPH,
    BlockType.DIALOGUE,
    BlockType.CHAPTER,
    BlockType.SECTION,
    BlockType.RUBY,
}
_JOINABLE_TYPES = {BlockType.PARAGRAPH, BlockType.DIALOGUE, BlockType.RUBY}

PDF_ASSET_MARKER_RE = re.compile(
    r"^[\s　]*[＜<]\s*[ｉiI]\s*[０-９0-9]{3,}\s*[｜|]\s*[０-９0-9]{2,}\s*[＞>][\s　]*$"
)
PDF_IMAGE_CAPTION_RE = re.compile(r"^[\s　]*【[^】]{1,80}】[\s　]*$")
PDF_AFTERWORD_MARKER_RE = re.compile(r"^[\s　]*[０-９0-9]{1,6}（(?:前書|後書き)）[\s　]*$")
PDF_BARE_CHAPTER_RE = re.compile(r"^[\s　]*[０-９0-9]{1,6}[\s　]*$")
PDF_SCENE_MARKER_RE = re.compile(r"^[\s　]*(?:(?:×{3}|◯|○|●|◇|◆|＊{1,3}|\*{1,3})|(?:＊{3,}[^＊\n]{1,80}＊{1,})|(?:\*{3,}[^*\n]{1,80}\*{1,}))[\s　]*$")
PDF_USAGE_HEADER_RE = re.compile(r"^[\s　]*[＜<]\s*使用方法\s*[＞>][\s　]*$")
PDF_QUESTION_WRAPPED_TERM_RE = re.compile(r"^\?[^?\n]{1,80}\?")
PDF_PLAIN_NOTE_HEADING_RE = re.compile(r"^[\s　]*(?:まえがき|前書き|前書|あとがき|後書き|後書)[\s　]*$")
PDF_NAMED_NOTE_RE = re.compile(
    # Some PDFNovels generations lose the final full-width ``）`` glyph in
    # the selectable text layer even though it is visible on the page.  Accept
    # that one source-loss shape as a note marker; the marker still has the
    # distinctive terminal ``（前書き`` / ``（後書き`` suffix.
    r"^[\s　]*(?P<title>.+?)（(?P<kind>前書(?:き)?|後書(?:き)?)）?[\s　]*$"
)
PDF_RETIRED_MARK_RE = re.compile(
    r"（(?:[０-９0-9]{1,4}年)?[０-９0-9]{1,2}月[０-９0-9]{1,2}日削除(?:予定)?）"
)
PDF_RETIRED_RANGE_MARK_RE = re.compile(
    r"^第[０-９0-9一二三四五六七八九十百]+章第[０-９0-9一二三四五六七八九十百]+話"
    r"[〜～~\-‐‑–—―]第[０-９0-9一二三四五六七八九十百]+話"
    r"（(?:[０-９0-9]{1,4}年)?[０-９0-9]{1,2}月[０-９0-9]{1,2}日削除予定）"
    r"(?:（前書(?:き)?）)?$"
)
PDF_REPEATED_SECTION_RE = re.compile(
    r"^第[０-９0-9一二三四五六七八九十百]+章[\s　]*"
    r"[\u3400-\u9fff\uf900-\ufaffァ-ヶA-Za-z0-9「『][^。！？!?\n]{0,80}$"
)
PDF_CHAPTER_PREFIXED_PROSE_RE = re.compile(
    r"^第[０-９0-9一二三四五六七八九十百]+[章話部巻]"
    r"(?:から|では|で|を|の|は|に|と|も|まで|より|へ|が)(?=\S)"
)

_VERTICAL_GLYPH_RE = re.compile(
    r"^[\u2e80-\u2fff\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff々〃〆ヶヵー、。！？!?…‥—―「」『』（）〈〉《》【】・：；\d０-９]+$"
)
_SHORT_TAIL_RE = re.compile(
    r"^[\u2e80-\u2fffぁ-んァ-ヶ一-龯\uf900-\ufaff々〃〆ヶヵー]{1,8}[。！？!?）」』】]?$"
)
_STRONG_END_RE = re.compile(r"[。．︒！？!?‼⁉…‥—―」』﹂﹄）)】》]$")
_JAPANESE_START_RE = re.compile(r"^[\u2e80-\u2fffぁ-んァ-ヶ一-龯\uf900-\ufaff々〃〆ヶヵー０-９]")

# High-confidence prefixes that cannot naturally start a new paragraph after
# an unfinished Japanese physical column.
_CONTINUATION_PREFIXES = (
    "が", "を", "に", "へ", "と", "で", "も", "は", "の", "や", "から", "まで", "より",
    "って", "ので", "のに", "けれど", "ながら", "つつ", "たり", "て", "し", "い", "う", "る",
    "た", "だ", "ます", "ました", "ません", "れば", "れば", "ー", "々", "、", "。", "！", "？",
    "!", "?", "」", "』",
)

_SIMULTANEOUS_OPEN = ("「「", "『『")
_ORPHAN_CLOSE = {"」", "』"}
_QUOTE_PAIRS = (("「", "」"), ("『", "』"), ("（", "）"), ("(", ")"), ("【", "】"), ("《", "》"))


def is_pdf_asset_marker(text: str) -> bool:
    return bool(PDF_ASSET_MARKER_RE.fullmatch(str(text or "")))


def _named_note_info(text: str) -> tuple[str, str] | None:
    """Return ``(base_title, kind)`` for Narou-style named pre/post notes."""
    match = PDF_NAMED_NOTE_RE.fullmatch(str(text or "").strip())
    if not match:
        return None
    title = match.group("title").strip(" \t\r\n　")
    kind = match.group("kind")
    if not title:
        return None
    return title, kind


def _is_chapter_prefixed_prose(text: str) -> bool:
    """True for ordinary prose grammatically led by ``第X章/話``.

    PDFNovels/Narou afterwords often start sentences like
    ``第２章から読まれた方は…``.  Upstream classification can tag such a
    line as CHAPTER from the prefix alone; a following Japanese case particle
    is strong evidence that this is prose, not a heading boundary.
    """
    return bool(PDF_CHAPTER_PREFIXED_PROSE_RE.match(str(text or "").strip()))


def _canonical_retired_title(text: str) -> str:
    """Normalize an explicitly retired chapter title for duplicate matching."""
    value = str(text or "").strip(" \t\r\n　")
    note = _named_note_info(value)
    if note:
        value = note[0]
    value = PDF_RETIRED_MARK_RE.sub("", value)
    return value.strip(" \t\r\n　")


def _append_modified_by(value: str, step: str) -> str:
    parts = [part for part in str(value or "").split(",") if part]
    if step not in parts:
        parts.append(step)
    return ",".join(parts)


def _compact_guard_text(text: str) -> str:
    """Characters used by the conservation guard; layout whitespace is ignored.

    The PDF preparation stage applies NFC normalization (including a number of
    CJK compatibility ideographs).  The guard must compare the same normalized
    character stream, otherwise a lossless ``落 -> 落`` normalization is reported
    as one removed plus one added glyph.
    """
    value = unicodedata.normalize("NFC", str(text or ""))
    return re.sub(r"[\s　]+", "", value)


def _counter(texts: Iterable[str]) -> Counter[str]:
    result: Counter[str] = Counter()
    for text in texts:
        result.update(_compact_guard_text(text))
    return result


def _normalise_block_text(text: str) -> str:
    """Remove invisible artifacts and collapse true one-glyph vertical stacks."""
    value = unicodedata.normalize("NFC", str(text or ""))
    value = value.replace("\ufeff", "").replace("\u200b", "").replace("\u2060", "")
    value = value.replace("\u00ad", "")
    value = value.replace("\r\n", "\n").replace("\r", "\n")

    raw_lines = value.split("\n")
    lines = [line.strip(" \t　") for line in raw_lines if line.strip(" \t　")]
    if len(lines) >= 4:
        compact = "".join(lines)
        short_ratio = sum(len(line) <= 2 for line in lines) / len(lines)
        if short_ratio >= 0.82 and _VERTICAL_GLYPH_RE.fullmatch(compact):
            return compact

    # A selectable-PDF block represents a physical column.  Internal line
    # breaks without an empty separator are wraps inside that same column, not
    # paragraph boundaries, so joining them is lossless.
    if lines and "\n\n" not in value:
        return "".join(lines)
    return "\n\n".join(part.strip(" \t\r\n　") for part in re.split(r"\n\s*\n", value) if part.strip())


def _is_text_block(block: Block) -> bool:
    return block.type in _JOINABLE_TYPES


def _text(block: Block) -> str:
    return str(block.text or "").strip(" \t\r\n　")


def _quote_balance(text: str, opener: str, closer: str) -> int:
    return str(text or "").count(opener) - str(text or "").count(closer)


def _has_unclosed_quote(text: str) -> bool:
    return any(_quote_balance(text, opener, closer) > 0 for opener, closer in _QUOTE_PAIRS)


def _is_structural_block(block: Block) -> bool:
    text = _text(block)
    if not text:
        return True
    if block.type in {BlockType.CHAPTER, BlockType.SECTION, BlockType.IMAGE_REF}:
        return True
    metadata = block.metadata or {}
    if (
        metadata.get("pdf_text_asset_marker")
        or metadata.get("pdf_text_image_caption")
        or metadata.get("pdf_text_named_note_marker")
        or metadata.get("pdf_text_plain_note_heading")
        or metadata.get("pdf_text_afterword_marker")
    ):
        return True
    return bool(
        PDF_ASSET_MARKER_RE.fullmatch(text)
        or PDF_IMAGE_CAPTION_RE.fullmatch(text)
        or PDF_AFTERWORD_MARKER_RE.fullmatch(text)
        or PDF_BARE_CHAPTER_RE.fullmatch(text)
        or PDF_SCENE_MARKER_RE.fullmatch(text)
        or PDF_USAGE_HEADER_RE.fullmatch(text)
        or PDF_PLAIN_NOTE_HEADING_RE.fullmatch(text)
    )


def _source_ids(block: Block) -> list[str]:
    values = list((block.metadata or {}).get("source_block_ids") or [])
    if not values and block.id:
        values = [block.id]
    return [str(value) for value in values if value]


def _source_texts(block: Block) -> list[str]:
    values = list((block.metadata or {}).get("pdf_source_texts") or [])
    if not values:
        values = [block.ocr_raw or block.text or ""]
    return [str(value) for value in values]


def _merge_bbox(left: BoundingBox | None, right: BoundingBox | None) -> BoundingBox | None:
    if left is None:
        return copy.copy(right) if right is not None else None
    if right is None:
        return copy.copy(left)
    x1 = min(left.x, right.x)
    y1 = min(left.y, right.y)
    x2 = max(left.x + left.w, right.x + right.w)
    y2 = max(left.y + left.h, right.y + right.h)
    return BoundingBox(x=x1, y=y1, w=max(0.0, x2 - x1), h=max(0.0, y2 - y1))


def _source_bbox_dict(block: Block, *, last: bool = False) -> dict:
    metadata = block.metadata or {}
    if last:
        value = metadata.get("pdf_last_source_bbox") or metadata.get("pdf_source_bbox") or {}
    else:
        value = metadata.get("pdf_first_source_bbox") or metadata.get("pdf_source_bbox") or {}
    return value if isinstance(value, dict) else {}


def _source_column_fullish(block: Block) -> bool | None:
    """Return whether the last source track reaches the normal PDF body bottom.

    Selectable vertical PDFs preserve one physical column per source track.  A
    wrapped paragraph normally fills the current track almost to the page-body
    bottom before continuing on the next track; a deliberate short paragraph
    does not.  This geometry signal is stronger than Japanese terminal
    punctuation because web novels frequently omit ``。`` on purpose.

    ``None`` means legacy/synthetic data does not carry enough geometry and the
    caller should keep the pre-geometry behaviour for compatibility.
    """
    metadata = block.metadata or {}
    region = metadata.get("pdf_last_source_region") or metadata.get("pdf_source_region")
    bbox = _source_bbox_dict(block, last=True)
    if not isinstance(region, (list, tuple)) or len(region) != 4 or not bbox:
        return None
    try:
        region_y0 = float(region[1])
        region_y1 = float(region[3])
        track_y1 = float(bbox["y1"])
        glyph = max(1.0, float(bbox["x1"]) - float(bbox["x0"]))
    except Exception:
        return None
    height = region_y1 - region_y0
    if height <= 0:
        return None
    body_bottom = metadata.get("pdf_last_source_body_bottom", metadata.get("pdf_source_body_bottom"))
    if body_bottom is not None:
        return abs(float(body_bottom) - track_y1) <= glyph * 1.5

    # PDFNovels body tracks in the regression corpus end around 89--92% of the
    # full page region.  85.5% accepts a genuine full continuation track while
    # rejecting short standalone paragraphs (~82% or less in the new dog-book
    # fixture).  A glyph-scaled bottom-gap cap keeps the test robust to modest
    # page-size/font-size changes.
    ratio_full = (track_y1 - region_y0) / height >= 0.855
    gap_full = (region_y1 - track_y1) <= max(84.0, glyph * 6.0)
    return bool(ratio_full and gap_full)


def _record_pdf_body_bottoms(blocks: list[Block]) -> None:
    """Infer a logical page's body bottom from repeated long vertical tracks.

    A stacked page can have asymmetric margins. Its region edge alone does
    not locate the text bottom, so require at least three agreeing tracks.
    """
    groups: dict[tuple, list[tuple[Block, float, float]]] = {}
    for block in blocks:
        if not _is_text_block(block):
            continue
        metadata = block.metadata or {}
        region = metadata.get("pdf_source_region")
        bbox = _source_bbox_dict(block)
        if not isinstance(region, (list, tuple)) or len(region) != 4 or not bbox:
            continue
        try:
            glyph = max(1.0, float(bbox["x1"]) - float(bbox["x0"]))
            y1 = float(bbox["y1"])
            if y1 - float(bbox["y0"]) < glyph * 10:
                continue
            key = (block.page, tuple(float(v) for v in region))
        except (KeyError, TypeError, ValueError):
            continue
        if _is_structural_block(block):
            continue
        groups.setdefault(key, []).append((block, y1, glyph))
    for items in groups.values():
        clusters: list[list[tuple[Block, float, float]]] = []
        for item in sorted(items, key=lambda entry: entry[1]):
            if clusters and abs(item[1] - clusters[-1][0][1]) <= min(item[2], clusters[-1][0][2]) * 0.6:
                clusters[-1].append(item)
            else:
                clusters.append([item])
        supported = [cluster for cluster in clusters if len(cluster) >= 3]
        if not supported:
            continue
        # Prefer the most supported bottom; lower coordinates break ties.
        cluster = max(supported, key=lambda group: (len(group), group[-1][1]))
        baseline = sum(item[1] for item in cluster) / len(cluster)
        for block, _y1, _glyph in items:
            block.metadata["pdf_source_body_bottom"] = baseline


def _chapter_candidate_physical_continuation(left: Block, right: Block) -> bool:
    """Allow a text line misclassified as CHAPTER to consume its next physical column.

    PDFNovels pages sometimes contain ordinary synopsis/afterword prose beginning
    with ``第2章...``.  The text classifier necessarily sees that prefix before
    page geometry is available and may mark the first full physical column as a
    chapter.  A real chapter heading is separated from body text by a visibly
    wider horizontal gap; a wrapped prose column instead continues at the
    immediately adjacent vertical track and restarts at the page-top baseline.

    Requiring all three signals -- long/full source column, adjacent track, and
    upward top-baseline restart -- keeps true chapter/body boundaries protected.
    """
    if left.type != BlockType.CHAPTER or right.type not in _JOINABLE_TYPES:
        return False
    if right.type == BlockType.CHAPTER or _is_structural_block(right):
        return False
    left_text = _text(left)
    right_text = _text(right)
    if not left_text or not right_text or _STRONG_END_RE.search(left_text):
        return False
    if not _JAPANESE_START_RE.match(right_text):
        return False

    lm = left.metadata or {}
    rm = right.metadata or {}
    lp = int(lm.get("pdf_source_physical_page") or left.page or 0)
    rp = int(rm.get("pdf_source_physical_page") or right.page or 0)
    if lp <= 0 or rp <= 0 or lp != rp:
        return False
    lb = _source_bbox_dict(left, last=True)
    rb = _source_bbox_dict(right, last=False)
    if not lb or not rb:
        return False
    try:
        lx0 = float(lb["x0"]); rx0 = float(rb["x0"])
        ly0 = float(lb["y0"]); ry0 = float(rb["y0"])
        ly1 = float(lb["y1"])
        lw = max(1.0, float(lb["x1"]) - lx0)
        rw = max(1.0, float(rb["x1"]) - rx0)
    except Exception:
        return False

    glyph = min(lw, rw)
    x_gap = lx0 - rx0
    # Adjacent vertical tracks in the supplied PDF are ~22.7 pt apart for a
    # 14 pt glyph; true heading->body transitions are ~68 pt apart.
    adjacent_track = glyph * 1.15 <= x_gap <= glyph * 2.35
    top_restart = (ly0 - ry0) >= max(5.0, glyph * 0.65)
    fullish_column = (ly1 - ly0) >= glyph * 20.0 or len(left_text) >= 24
    return adjacent_track and top_restart and fullish_column


def _same_page_indented_new_paragraph(left: Block, right: Block) -> bool:
    """Use vertical source geometry to stop a tail column swallowing a new paragraph.

    In PDFNovels vertical pages, physical continuation columns normally restart at
    the top baseline while a new paragraph is indented by roughly one full glyph.
    This signal is especially important when the preceding sentence itself ended
    in a short physical tail without punctuation (for example ``士団の戦い``).
    """
    lm = left.metadata or {}
    rm = right.metadata or {}
    # The top and bottom halves of one physical sheet are different logical
    # pages. Their absolute y offset is not a paragraph indentation.
    if int(lm.get("pdf_last_source_logical_page", left.page) or 0) != int(right.page or 0):
        return False
    lp = int(lm.get("pdf_last_source_physical_page") or lm.get("pdf_source_physical_page") or left.page or 0)
    rp = int(rm.get("pdf_source_physical_page") or right.page or 0)
    if lp <= 0 or rp <= 0 or lp != rp:
        return False
    lb = _source_bbox_dict(left, last=True)
    rb = _source_bbox_dict(right, last=False)
    if not lb or not rb:
        return False
    try:
        ly0 = float(lb["y0"]); ry0 = float(rb["y0"])
        lw = max(1.0, float(lb["x1"]) - float(lb["x0"]))
        rw = max(1.0, float(rb["x1"]) - float(rb["x0"]))
    except Exception:
        return False
    # One Japanese full-width glyph is ~14pt in the supplied PDF. Require a
    # clear downward shift, but scale the threshold to other font sizes.
    threshold = max(5.0, min(lw, rw) * 0.65)
    return (ry0 - ly0) >= threshold


def _merge_blocks(left: Block, right: Block, *, step: str, reopen_quote: bool = False) -> Block:
    merged = copy.copy(left)
    left_text = str(left.text or "").rstrip(" \t\r\n　")
    right_text = str(right.text or "").lstrip(" \t\r\n　")

    # When processing a legacy v1.3.7 result, a close quote may already have
    # been inserted at the physical column boundary.  Remove only that one
    # high-confidence premature closer; retain the real closer on the right.
    removed_guard_char = ""
    if reopen_quote and left_text.endswith(("」", "』")):
        removed_guard_char = left_text[-1]
        left_text = left_text[:-1]

    merged.text = left_text + right_text
    merged.ocr_raw = "".join(_source_texts(left) + _source_texts(right))
    merged.modified_by = _append_modified_by(merged.modified_by, step)
    merged.metadata = dict(merged.metadata or {})
    merged.metadata["source_block_ids"] = list(dict.fromkeys(_source_ids(left) + _source_ids(right)))
    merged.metadata["pdf_source_texts"] = _source_texts(left) + _source_texts(right)
    merged.metadata["pdf_physical_columns_merged"] = True
    first_bbox = _source_bbox_dict(left, last=False)
    last_bbox = _source_bbox_dict(right, last=True)
    if first_bbox:
        merged.metadata["pdf_first_source_bbox"] = dict(first_bbox)
    if last_bbox:
        merged.metadata["pdf_last_source_bbox"] = dict(last_bbox)
    right_metadata = right.metadata or {}
    merged.metadata["pdf_last_source_logical_page"] = right_metadata.get("pdf_last_source_logical_page", right.page)
    merged.metadata["pdf_last_source_physical_page"] = right_metadata.get(
        "pdf_last_source_physical_page", right_metadata.get("pdf_source_physical_page", right.page)
    )
    last_region = right_metadata.get("pdf_last_source_region") or right_metadata.get("pdf_source_region")
    if last_region:
        merged.metadata["pdf_last_source_region"] = list(last_region)
    last_bottom = right_metadata.get("pdf_last_source_body_bottom", right_metadata.get("pdf_source_body_bottom"))
    if last_bottom is not None:
        merged.metadata["pdf_last_source_body_bottom"] = last_bottom
    else:
        merged.metadata["pdf_last_source_body_bottom"] = None
    removed_counts = Counter((left.metadata or {}).get("pdf_guard_intentional_removed_chars") or {})
    removed_counts.update((right.metadata or {}).get("pdf_guard_intentional_removed_chars") or {})
    if removed_guard_char:
        removed_counts.update(removed_guard_char)
    if removed_counts:
        merged.metadata["pdf_guard_intentional_removed_chars"] = dict(removed_counts)
    merged.bbox = _merge_bbox(left.bbox, right.bbox)
    return merged


def _premature_quote_continuation(left: str, right: str) -> bool:
    if not left or not right or left[-1] not in "」』":
        return False
    closer = left[-1]
    opener = "「" if closer == "」" else "『"
    # Only an *outer* dialogue/quotation that starts the block may be reopened.
    # A narration such as ``父さんと同じ『潜入、捜索、暗殺』 / を行う``
    # contains a correctly closed inline quoted term; deleting that 』 loses
    # source text and changes meaning.
    if not left.lstrip(" \t　").startswith(opener):
        return False
    if right.startswith(("「", "『")):
        return False
    left_body = left[:-1]
    # The right column carries the actual close quote, begins with a punctuation
    # or suffix glyph, or completes a very characteristic split word.
    if right.endswith(closer) and opener not in right:
        return True
    if right.startswith(("、", "。", "！", "？", "!", "?", "ー", "々", "」", "』")):
        # ``?高等戦闘魔術師?`` / ``?黒蠅?`` are literal source-side
        # glossary/ruby-base markers in PDFNovels output, not punctuation that
        # proves the preceding quote was closed too early.
        if right.startswith("?") and PDF_QUESTION_WRAPPED_TERM_RE.match(right):
            return False
        return True
    if left_body.endswith("これ") and right.startswith("っぽっち"):
        return True
    if left_body.endswith(("てお", "でお")) and right.startswith(("ります", "りません", "りました")):
        return True
    if left_body and "ァ" <= left_body[-1] <= "ヶ" and right.startswith("ー"):
        return True
    # Do not reopen merely because the next paragraph begins with a particle
    # such as と/い/も.  Valid dialogue is very often followed by narration
    # beginning with those glyphs (``「…」と彼は言った``).  Reopening is only
    # safe when the right side carries the real closer or completes one of the
    # explicit physical split patterns above.
    return False


def _should_join_physical(left: Block, right: Block) -> tuple[bool, bool]:
    """Return ``(join, reopen_premature_quote)`` for two adjacent PDF columns."""
    # Text beginning with ``第X章`` can be ordinary synopsis/afterword prose.
    # The extractor classifies it before seeing neighbouring geometry, so allow
    # a chapter candidate to consume an immediately-adjacent physical tail when
    # the source coordinates prove that it is a wrapped column rather than a
    # real heading followed by body text.
    if _chapter_candidate_physical_continuation(left, right):
        return True, False
    if not _is_text_block(left) or not _is_text_block(right):
        return False, False
    if _is_structural_block(left) or _is_structural_block(right):
        return False, False

    left_text = _text(left)
    right_text = _text(right)
    if not left_text or not right_text:
        return False, False

    # A visibly indented next physical column is a new paragraph, not a
    # continuation.  Do this before generic Japanese-prefix joining, but never
    # override a genuinely unclosed quote: dialogue can span an indented column.
    if not _has_unclosed_quote(left_text) and _same_page_indented_new_paragraph(left, right):
        return False, False

    # An actually unclosed outer quote takes priority over any inline quote at
    # the column end.  Join without deleting the inline closer.
    if _has_unclosed_quote(left_text):
        if right_text.startswith(("「", "『", "﹁", "﹃")) and not right_text.startswith(("「「", "『『")):
            return False, False
        return True, False

    if _premature_quote_continuation(left_text, right_text):
        return True, True

    # Geometry is the primary boundary signal for ordinary prose.  If the last
    # source track ended well above the body bottom, it is a deliberate short
    # paragraph even when the author omitted terminal punctuation.  Keep quote-
    # proven continuations above this guard because dialogue may legitimately
    # wrap from a short-looking source fragment.
    fullish = _source_column_fullish(left)
    if fullish is False:
        return False, False

    # A correctly closed inline quoted term can still be followed by a particle
    # in the next physical column; join it while preserving the quote.
    if left_text.endswith(("」", "』")) and right_text.startswith((
        "から", "まで", "より", "って", "を", "が", "に", "へ", "と", "で", "は", "の", "も", "や"
    )):
        return True, False

    # Any positive quote balance means this logical dialogue/parenthetical has
    # not reached its actual closing glyph yet.  A brand-new opening quote is a
    # safety boundary; it is more likely the next speaker than a continuation.
    if _STRONG_END_RE.search(left_text):
        return False, False

    if right_text.startswith(("「", "『", "﹁", "﹃")):
        return False, False

    # Short suffix columns such as た。/い。/る。 must be consumed before any
    # cleanup or dedup step gets a chance to classify them as noise.
    if _SHORT_TAIL_RE.fullmatch(right_text):
        return True, False
    if right_text.startswith(_CONTINUATION_PREFIXES):
        return True, False

    # Selectable-PDF columns normally split a sentence at an arbitrary glyph.
    # If the left column has no terminal punctuation and the right begins with
    # Japanese text, joining is safer than inventing a paragraph boundary.
    return bool(_JAPANESE_START_RE.match(right_text)), False


def _repair_simultaneous_speech(blocks: list[Block]) -> tuple[list[Block], int]:
    """Repair ``「「...」 / 僕 / 」 / と神官...`` without inventing text."""
    result: list[Block] = []
    repaired = 0
    i = 0
    while i < len(blocks):
        if i + 3 < len(blocks):
            first, subject, orphan, continuation = blocks[i:i + 4]
            first_text = _text(first)
            subject_text = _text(subject)
            orphan_text = _text(orphan)
            continuation_text = _text(continuation)
            opens_twice = first_text.startswith(_SIMULTANEOUS_OPEN)
            close_char = "」" if first_text.startswith("「「") else "』"
            missing_one_close = opens_twice and first_text.count(first_text[0]) == first_text.count(close_char) + 1
            safe_subject = (
                _is_text_block(subject)
                and 1 <= len(subject_text) <= 12
                and not re.search(r"[。！？!?」』]$", subject_text)
            )
            safe_continuation = (
                _is_text_block(continuation)
                and continuation_text.startswith(("と", "が", "は", "も", "の", "を", "に"))
            )
            if (
                _is_text_block(first)
                and missing_one_close
                and safe_subject
                and orphan_text == close_char
                and safe_continuation
            ):
                fixed_first = copy.copy(first)
                fixed_first.text = str(first.text or "").rstrip() + close_char
                fixed_first.ocr_raw = "".join(_source_texts(first) + _source_texts(orphan))
                fixed_first.modified_by = _append_modified_by(first.modified_by, "pdf_text_prepare")
                fixed_first.metadata = dict(fixed_first.metadata or {})
                fixed_first.metadata["source_block_ids"] = _source_ids(first) + _source_ids(orphan)
                fixed_first.metadata["pdf_source_texts"] = _source_texts(first) + _source_texts(orphan)
                fixed_subject = _merge_blocks(subject, continuation, step="pdf_text_prepare")
                result.extend((fixed_first, fixed_subject))
                repaired += 1
                i += 4
                continue
        result.append(blocks[i])
        i += 1
    return result, repaired


def _join_pdf_physical_columns(blocks: list[Block]) -> tuple[list[Block], int]:
    result: list[Block] = []
    merged_count = 0
    i = 0
    while i < len(blocks):
        current = blocks[i]
        chapter_tail = (
            i + 1 < len(blocks)
            and _chapter_candidate_physical_continuation(current, blocks[i + 1])
        )
        if (not _is_text_block(current) or _is_structural_block(current)) and not chapter_tail:
            result.append(current)
            i += 1
            continue

        merge_guard = 0
        while i + 1 < len(blocks) and merge_guard < 64:
            nxt = blocks[i + 1]
            should_join, reopen = _should_join_physical(current, nxt)
            if not should_join:
                break
            current = _merge_blocks(current, nxt, step="pdf_text_prepare", reopen_quote=reopen)
            i += 1
            merge_guard += 1
            merged_count += 1
        result.append(current)
        i += 1
    return result, merged_count


def _same_or_adjacent_pdf_page(left: Block, right: Block, *, same_only: bool = False) -> bool:
    """Conservative locality guard for semantic PDF-column joins."""
    lp = int((left.metadata or {}).get("pdf_source_physical_page") or left.page or 0)
    rp = int((right.metadata or {}).get("pdf_source_physical_page") or right.page or 0)
    if lp <= 0 or rp <= 0:
        return left.page == right.page if same_only else abs(int(left.page or 0) - int(right.page or 0)) <= 1
    return lp == rp if same_only else rp in {lp, lp + 1}


def _merge_balanced_quote_fragments(blocks: list[Block]) -> tuple[list[Block], int]:
    """Join adjacent fragments only when an unmatched outer quote proves continuity."""
    out: list[Block] = []
    merged_count = 0
    i = 0
    while i < len(blocks):
        left = blocks[i]
        if i + 1 < len(blocks) and _is_text_block(left) and not _is_structural_block(left):
            right = blocks[i + 1]
            lt, rt = _text(left), _text(right)
            if (
                _is_text_block(right)
                and not _is_structural_block(right)
                and _same_or_adjacent_pdf_page(left, right)
                and _quote_balance(lt, "「", "」") > 0
                and _quote_balance(lt + rt, "「", "」") == 0
            ):
                merged = _merge_blocks(left, right, step="pdf_text_prepare")
                merged.type = BlockType.DIALOGUE if lt.lstrip().startswith("「") else BlockType.PARAGRAPH
                out.append(merged)
                merged_count += 1
                i += 2
                continue
        out.append(left)
        i += 1
    return out, merged_count


def _is_complete_inline_quote(text: str) -> bool:
    text = str(text or '').strip()
    if len(text) < 2:
        return False
    pairs = (("「", "」"), ("『", "』"))
    for opener, closer in pairs:
        if text.startswith(opener):
            close_at = text.find(closer, 1)
            return close_at == len(text) - 1
    return False


def _starts_with_quoted_term_and_tail(text: str) -> bool:
    text = str(text or '').strip()
    for opener, closer in (("「", "」"), ("『", "』")):
        if not text.startswith(opener):
            continue
        close_at = text.find(closer, 1)
        if close_at > 0 and text[close_at + 1 :].strip():
            return True
    return False


def _merge_explicit_semantic_continuations(blocks: list[Block]) -> tuple[list[Block], int]:
    """Join only punctuation / grammar-proven continuations left by PDF columns.

    This deliberately avoids generic ``no full stop => merge`` logic.  Long
    Japanese PDFs contain deliberate fragment paragraphs; only signatures that
    prove a physical-column split are joined here.
    """
    out: list[Block] = []
    merged_count = 0
    i = 0
    while i < len(blocks):
        left = blocks[i]
        if _is_text_block(left) and not _is_structural_block(left):
            lt = _text(left)

            # Narrative + inline quoted utterance + quotative tail, all on the
            # same physical page: ``...腕を組んで`` + ``「...」`` + ``と...``.
            if i + 2 < len(blocks):
                mid, right = blocks[i + 1], blocks[i + 2]
                mt, rt = _text(mid), _text(right)
                if (
                    left.type == BlockType.PARAGRAPH
                    and _is_text_block(mid)
                    and _is_text_block(right)
                    and not _is_structural_block(mid)
                    and not _is_structural_block(right)
                    and not _STRONG_END_RE.search(lt)
                    and _is_complete_inline_quote(mt)
                    and rt.startswith(("と", "って", "とも", "などと", "そう"))
                    and _same_or_adjacent_pdf_page(left, mid, same_only=True)
                    and _same_or_adjacent_pdf_page(mid, right, same_only=True)
                ):
                    merged = _merge_blocks(left, mid, step="pdf_text_prepare")
                    merged = _merge_blocks(merged, right, step="pdf_text_prepare")
                    merged.type = BlockType.PARAGRAPH
                    out.append(merged)
                    merged_count += 2
                    i += 3
                    continue

            if i + 1 < len(blocks):
                right = blocks[i + 1]
                rt = _text(right)
                if _is_text_block(right) and not _is_structural_block(right):
                    left_fullish = _source_column_fullish(left)
                    dash_cont = (
                        rt.startswith(("──", "――", "—", "―"))
                        and not _STRONG_END_RE.search(lt)
                        and left_fullish is not False
                        and _same_or_adjacent_pdf_page(left, right)
                    )
                    comma_quote = (
                        left.type == BlockType.PARAGRAPH
                        and lt.endswith("、")
                        and rt.startswith(("「", "『"))
                        and _same_or_adjacent_pdf_page(left, right, same_only=True)
                    )
                    quoted_term_cont = (
                        left.type == BlockType.PARAGRAPH
                        and not _STRONG_END_RE.search(lt)
                        and _starts_with_quoted_term_and_tail(rt)
                        and _same_or_adjacent_pdf_page(left, right)
                    )
                    if dash_cont or comma_quote or quoted_term_cont:
                        merged = _merge_blocks(left, right, step="pdf_text_prepare")
                        merged.type = BlockType.PARAGRAPH if (comma_quote or quoted_term_cont) else left.type
                        out.append(merged)
                        merged_count += 1
                        i += 2
                        continue

            # A comma-ended sentence followed by a parenthetical thought and a
            # very short completion is one prose sentence in this vertical-PDF
            # layout: ``だけど、`` + ``（帰りたい）`` + ``故郷に。``.
            if i + 2 < len(blocks):
                mid, right = blocks[i + 1], blocks[i + 2]
                mt, rt = _text(mid), _text(right)
                if (
                    left.type == BlockType.PARAGRAPH
                    and lt.endswith("、")
                    and mt.startswith("（") and mt.endswith("）")
                    and len(mt) <= 40
                    and right.type == BlockType.PARAGRAPH
                    and 0 < len(rt.strip()) <= 24
                    and _STRONG_END_RE.search(rt)
                    and _same_or_adjacent_pdf_page(left, mid, same_only=True)
                    and _same_or_adjacent_pdf_page(mid, right, same_only=True)
                ):
                    merged = _merge_blocks(left, mid, step="pdf_text_prepare")
                    merged = _merge_blocks(merged, right, step="pdf_text_prepare")
                    merged.type = BlockType.PARAGRAPH
                    out.append(merged)
                    merged_count += 2
                    i += 3
                    continue

        out.append(left)
        i += 1
    return out, merged_count


def _repair_orphan_quote_boundaries(blocks: list[Block]) -> tuple[list[Block], int]:
    """Attach orphan close quotes when ownership is provable; never delete text."""
    result: list[Block] = []
    repaired = 0
    for block in blocks:
        text = _text(block)
        if text and set(text) <= _ORPHAN_CLOSE and result:
            previous = result[-1]
            previous_text = _text(previous)
            needed = all(
                previous_text.count("「" if ch == "」" else "『") >= previous_text.count(ch) + text.count(ch)
                for ch in set(text)
            )
            if needed:
                fixed = _merge_blocks(previous, block, step="pdf_text_finalize")
                result[-1] = fixed
                repaired += 1
                continue
            block.metadata = dict(block.metadata or {})
            block.metadata.setdefault("pdf_text_review_flags", []).append("orphan_closing_quote")
        result.append(block)
    return result, repaired



def _balanced_dialogue_spans(text: str) -> list[tuple[int, int]]:
    """Return balanced outer ``「...」`` spans, including doubled/tripled speech.

    A regex such as ``「[^」]*」`` stops at the first closing glyph and therefore
    breaks simultaneous speech like ``「「「……！？」」」``.  Counting nesting
    depth preserves every original quote while still giving us exact split
    boundaries.
    """
    spans: list[tuple[int, int]] = []
    depth = 0
    start = -1
    for index, char in enumerate(str(text or "")):
        if char == "「":
            if depth == 0:
                start = index
            depth += 1
        elif char == "」" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append((start, index + 1))
                start = -1
    return spans


def _dialogue_span_is_logical(text: str, start: int, previous_end: int, block_type: BlockType) -> bool:
    """Whether a balanced quote span is a dialogue paragraph, not an inline term."""
    before_all = text[:start]
    between = text[previous_end:start]
    if not before_all.strip(" \t\r\n　"):
        return True
    if not between.strip(" \t\r\n　") and previous_end > 0:
        # Consecutive dialogue columns: ``「A」「B」``.
        return True
    prefix = before_all.rstrip(" \t\r\n　")
    if prefix.endswith(("。", "！", "？", "!", "?", "‼", "⁉", "\n")):
        return True
    # A block already classified as DIALOGUE is trustworthy only when the
    # candidate begins at its first non-whitespace character.  This avoids
    # turning an inline quoted term later in the block into a fake speech line.
    if block_type == BlockType.DIALOGUE and not text[:start].strip(" \t\r\n　"):
        return True
    return False


def _clone_pdf_piece(block: Block, text: str, block_type: BlockType, *, ordinal: int) -> Block:
    piece = copy.copy(block)
    piece.id = ""
    piece.text = text.strip(" \t\r\n　")
    piece.type = block_type
    piece.modified_by = _append_modified_by(piece.modified_by, "restore_pdf_dialogue_columns")
    piece.metadata = dict(piece.metadata or {})
    piece.metadata["pdf_dialogue_piece"] = block_type == BlockType.DIALOGUE
    piece.metadata["pdf_piece_ordinal"] = ordinal
    # Keep the same source map on every split piece.  Text comparison can still
    # jump to the original physical column, while the character guard validates
    # the concatenated output and guarantees no glyph was lost or invented.
    piece.metadata["source_block_ids"] = _source_ids(block)
    piece.metadata["pdf_source_texts"] = _source_texts(block)
    return piece


def restore_pdf_dialogue_columns(doc: UnifiedDocument, *, inplace: bool = False) -> UnifiedDocument:
    """Put each logical dialogue in its own block/line for PDF text-layer mode.

    Rules:
    - every balanced dialogue beginning at a paragraph boundary is emitted as a
      standalone ``BlockType.DIALOGUE``;
    - narration before or after it becomes a separate paragraph block;
    - inline quoted terms inside narration (``所谓「右手」的职位``) remain in
      the narration because they do not start at a logical paragraph boundary;
    - doubled/tripled simultaneous speech remains one complete dialogue block;
    - no character, punctuation, or quote glyph is added or removed.
    """
    out = doc if inplace else copy.deepcopy(doc)
    result: list[Block] = []
    split_dialogues = 0
    split_narrations = 0

    for block in out.blocks:
        if block.type not in {BlockType.PARAGRAPH, BlockType.DIALOGUE} or _is_structural_block(block):
            result.append(block)
            continue

        text = str(block.text or "")
        spans = _balanced_dialogue_spans(text)
        if not spans:
            result.append(block)
            continue

        accepted: list[tuple[int, int]] = []
        previous_end = 0
        for start, end in spans:
            if _dialogue_span_is_logical(text, start, previous_end, block.type):
                accepted.append((start, end))
                previous_end = end

        if not accepted:
            result.append(block)
            continue

        pieces: list[Block] = []
        cursor = 0
        ordinal = 0
        for start, end in accepted:
            before = text[cursor:start].strip(" \t\r\n　")
            if before:
                pieces.append(_clone_pdf_piece(block, before, BlockType.PARAGRAPH, ordinal=ordinal))
                ordinal += 1
                split_narrations += 1
            dialogue_text = text[start:end].strip(" \t\r\n　")
            if dialogue_text:
                pieces.append(_clone_pdf_piece(block, dialogue_text, BlockType.DIALOGUE, ordinal=ordinal))
                ordinal += 1
                split_dialogues += 1
            cursor = end

        tail = text[cursor:].strip(" \t\r\n　")
        if tail:
            pieces.append(_clone_pdf_piece(block, tail, BlockType.PARAGRAPH, ordinal=ordinal))
            split_narrations += 1

        if pieces:
            result.extend(pieces)
        else:
            result.append(block)

    out.blocks = result
    out.add_log(
        "dialogue_restore",
        f"PDF文字层对白独立成行：分离 {split_dialogues} 条对白、{split_narrations} 个相邻叙述段",
        split_dialogues + split_narrations,
    )
    return out


def _mark_suspicious_glue(block: Block) -> int:
    """Flag only suspicious *physical-column seam* glue; never scan prose globally.

    The old audit searched the fully reconstructed paragraph and therefore
    produced false positives for perfectly normal Japanese such as ``皆俺が``
    or ``鳴らし始めて``.  A PDF column/page glue warning is meaningful only
    when the suspicious character pattern actually crosses a source-fragment
    boundary.  Keep the rule conservative and attach the seam evidence for the
    later AI/manual review pass.
    """
    metadata = dict(block.metadata or {})
    source_texts = [str(value or "") for value in (metadata.get("pdf_source_texts") or [])]
    if len(source_texts) < 2:
        return 0

    # Only keep high-confidence incomplete-kana stems. Generic CJK-to-CJK
    # seams and Japanese continuative verb forms are valid: e.g. ``照らし`` +
    # ``出す`` and ``全部`` + ``僕が``. This is advisory review metadata, so
    # false negatives are preferable to flooding AI/manual review with normal
    # prose.
    patterns = (
        re.compile(r"(?:であ|なかっ|いなかっ|知らな|ことにな)[\u3400-\u9fff]"),
    )
    joined = "".join(source_texts)
    seams: list[int] = []
    offset = 0
    for value in source_texts[:-1]:
        offset += len(value)
        seams.append(offset)

    evidence: list[dict] = []
    for pattern in patterns:
        for match in pattern.finditer(joined):
            # The match must consume characters on both sides of a real source
            # seam.  Merely occurring somewhere inside a reconstructed block is
            # not evidence of PDF column/page glue.
            crossing = [seam for seam in seams if match.start() < seam < match.end()]
            for seam in crossing:
                evidence.append({
                    "seam": seam,
                    "match": match.group(0),
                    "context": joined[max(0, seam - 12): min(len(joined), seam + 12)],
                    "left_tail": joined[max(0, seam - 12):seam],
                    "right_head": joined[seam:min(len(joined), seam + 12)],
                })

    if not evidence:
        return 0

    block.metadata = metadata
    flags = list(block.metadata.get("pdf_text_review_flags") or [])
    if "possible_missing_text_glue" not in flags:
        flags.append("possible_missing_text_glue")
    block.metadata["pdf_text_review_flags"] = flags
    review_evidence = dict(block.metadata.get("pdf_text_review_evidence") or {})
    review_evidence["possible_missing_text_glue"] = evidence
    block.metadata["pdf_text_review_evidence"] = review_evidence
    return 1


def _set_source_guard(doc: UnifiedDocument) -> None:
    source_blocks = [
        block for block in doc.blocks
        if block.type in _TEXT_TYPES and not (block.metadata or {}).get("pdf_text_exclude_from_guard")
    ]
    def _guard_source_text(block: Block) -> str:
        # Explicit visual repair changes only objectively broken U+FFFD glyphs.
        # Keep ocr_raw untouched for audit, but treat the reviewed block.text as
        # the authoritative character baseline so the lossless guard does not
        # falsely report every repaired glyph as one missing + one extra char.
        if (block.metadata or {}).get("pdf_visual_repair_applied"):
            return block.text or ""
        return block.ocr_raw or block.text or ""

    counts = _counter(_guard_source_text(block) for block in source_blocks)
    doc.metadata.pdf_text_source_char_counts = dict(counts)
    doc.metadata.pdf_text_source_chars = sum(counts.values())
    doc.metadata.pdf_text_guard_report = {
        "source_chars": sum(counts.values()),
        "output_chars": sum(counts.values()),
        "missing_chars": 0,
        "extra_chars": 0,
        "passed": True,
    }


def prepare_pdf_text_layer(doc: UnifiedDocument, *, inplace: bool = False) -> UnifiedDocument:
    """Preserve source, classify structure, and reconstruct physical columns."""
    out = doc if inplace else copy.deepcopy(doc)
    changed = 0
    markers = 0
    afterwords = 0
    named_notes = 0
    free_titles = 0

    # Narou/PDFNovels often emits a named preface/afterword marker on one page
    # and repeats the same bare title at the start of the story page.  The bare
    # title does not necessarily match the generic ``第X話`` regex, so without
    # this relation it looks like ordinary prose and can be joined to the first
    # body column.  Learn the exact title names from the source itself instead
    # of guessing from language/style.
    named_note_bases: set[str] = set()
    for source_block in out.blocks:
        if source_block.type not in _TEXT_TYPES:
            continue
        info = _named_note_info(_text(source_block))
        if info:
            named_note_bases.add(info[0])

    for block in out.blocks:
        if block.type not in _TEXT_TYPES:
            continue
        original = block.text or ""
        block.metadata = dict(block.metadata or {})
        block.metadata.setdefault("source_block_ids", [block.id])
        block.metadata.setdefault("pdf_source_texts", [block.ocr_raw or original])
        normalised = _normalise_block_text(original)
        if normalised != original:
            block.ocr_raw = block.ocr_raw or original
            block.text = normalised
            block.modified_by = _append_modified_by(block.modified_by, "pdf_text_prepare")
            changed += 1
        text = _text(block)
        named_note = _named_note_info(text)
        if named_note:
            block.metadata["pdf_text_named_note_marker"] = named_note[1]
            block.metadata["pdf_text_named_note_base"] = named_note[0]
            block.metadata["exclude_from_sentence_merge"] = True
            named_notes += 1
        elif text in named_note_bases:
            # Strong source-derived evidence that this short free-form phrase
            # is a real episode title: the same phrase was used verbatim in a
            # neighbouring ``（前書き）/（後書き）`` marker.  Promote it before
            # physical-column joining so it cannot swallow the first sentence.
            block.type = BlockType.CHAPTER
            block.metadata["pdf_text_free_title"] = True
            block.metadata["exclude_from_sentence_merge"] = True
            free_titles += 1

        if PDF_PLAIN_NOTE_HEADING_RE.fullmatch(text):
            block.metadata["pdf_text_plain_note_heading"] = True
            block.metadata["exclude_from_sentence_merge"] = True

        if is_pdf_asset_marker(text):
            block.metadata["pdf_text_asset_marker"] = True
            block.metadata["exclude_from_sentence_merge"] = True
            markers += 1
        elif PDF_IMAGE_CAPTION_RE.fullmatch(text):
            block.metadata["pdf_text_image_caption"] = True
            block.metadata["exclude_from_sentence_merge"] = True
        elif PDF_AFTERWORD_MARKER_RE.fullmatch(text):
            block.metadata["pdf_text_afterword_marker"] = True
            block.metadata["exclude_from_sentence_merge"] = True
            afterwords += 1

    _record_pdf_body_bottoms(out.blocks)
    _set_source_guard(out)

    # This deterministic four-block reorder must happen before the generic
    # physical join, otherwise the subject glyph would be swallowed into the
    # simultaneous-speech quote.
    blocks, simultaneous = _repair_simultaneous_speech(out.blocks)
    blocks, joined = _join_pdf_physical_columns(blocks)
    blocks, quote_joined = _merge_balanced_quote_fragments(blocks)
    blocks, semantic_joined = _merge_explicit_semantic_continuations(blocks)
    blocks, orphan = _repair_orphan_quote_boundaries(blocks)
    # Deterministic legacy repair may remove a quote that an older Formatter
    # inserted at a physical column boundary.  Record it as an intentional
    # correction so the character guard still detects every *unexplained* loss.
    intentional_removed: Counter[str] = Counter()
    for block in blocks:
        intentional_removed.update((block.metadata or {}).get("pdf_guard_intentional_removed_chars") or {})
    if intentional_removed:
        expected = Counter(out.metadata.pdf_text_source_char_counts or {})
        expected.subtract(intentional_removed)
        expected = Counter({ch: count for ch, count in expected.items() if count > 0})
        out.metadata.pdf_text_source_char_counts = dict(expected)
        out.metadata.pdf_text_source_chars = sum(expected.values())
    out.blocks = blocks
    out.add_log(
        "pdf_text_prepare",
        (
            f"PDF文字层无损预处理：规范化 {changed} 个块，接回 {joined} 个物理列，"
            f"修复 {simultaneous + orphan + quote_joined} 处引号边界，接回 {semantic_joined} 处显式续段，"
            f"标记 {markers} 个资源编号、{afterwords} 段后记、{named_notes} 个具名前后书；"
            f"恢复 {free_titles} 个自由标题"
        ),
        changed + joined + simultaneous + orphan + quote_joined + semantic_joined + markers + afterwords + named_notes + free_titles,
    )
    return out


def clean_pdf_text_metadata(doc: UnifiedDocument) -> UnifiedDocument:
    """PDF-safe metadata cleanup: never delete short top/bottom continuation tails."""
    out = copy.deepcopy(doc)
    before = len(out.blocks)
    out.blocks = [block for block in out.blocks if block.type != BlockType.HEADER_FOOTER]
    removed = before - len(out.blocks)
    out.add_log("clean_metadata", f"PDF文字层安全清理：仅删除 {removed} 个已明确标记的页眉/页脚块", removed)
    return out


def _bbox_overlap_ratio(a: BoundingBox | None, b: BoundingBox | None) -> float:
    if a is None or b is None:
        return 0.0
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.w, b.x + b.w)
    y2 = min(a.y + a.h, b.y + b.h)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    smaller = min(max(a.w * a.h, 1e-9), max(b.w * b.h, 1e-9))
    return inter / smaller


def remove_pdf_coordinate_duplicates(doc: UnifiedDocument, *, inplace: bool = False) -> UnifiedDocument:
    """Remove only exact same-page text whose bounding boxes overlap strongly."""
    out = doc if inplace else copy.deepcopy(doc)
    result: list[Block] = []
    removed = 0
    for block in out.blocks:
        text = _compact_guard_text(block.text)
        duplicate = False
        if text and block.bbox is not None:
            for previous in reversed(result[-12:]):
                if previous.page != block.page or previous.bbox is None:
                    continue
                if _compact_guard_text(previous.text) != text:
                    continue
                if _bbox_overlap_ratio(previous.bbox, block.bbox) >= 0.72:
                    duplicate = True
                    break
        if duplicate:
            removed += 1
            continue
        result.append(block)
    out.blocks = result
    out.add_log("remove_duplicates", f"PDF文字层坐标去重：删除 {removed} 个重叠副本；保留正常重复修辞", removed)
    return out


def skip_pdf_overlap_merge(doc: UnifiedDocument) -> UnifiedDocument:
    out = copy.deepcopy(doc)
    out.add_log("merge_overlaps", "PDF文字层已在无损预处理阶段接续物理列；跳过可能吞字的模糊重叠合并", 0)
    return out


def _strip_named_pdf_chapter_notes(doc: UnifiedDocument, *, inplace: bool = False) -> tuple[UnifiedDocument, int]:
    """Remove explicit Narou/PDFNovels prefaces and afterwords conservatively.

    The format stage must not infer chapter semantics from prose.  Instead it
    uses only source-derived sentinels that the PDF itself gives us:

    * ``Title（前書き）`` -> remove until the exact bare ``Title`` appears;
    * ``Title（後書き）`` -> remove until the next explicit named preface or
      another bare title that is independently proven by a named note marker;
    * if no such sentinel is found nearby, fall back to same-physical-page
      removal so malformed text layers can never swallow an unknown episode.

    This handles multi-page author notes while retaining the black-magic safety
    rule that uncertain next-page text is preserved rather than guessed away.
    """
    out = doc if inplace else copy.deepcopy(doc)
    blocks = list(out.blocks)
    result: list[Block] = []
    removed = 0

    def _physical_page(block: Block) -> int:
        try:
            return int((block.metadata or {}).get("pdf_source_physical_page") or block.page or 0)
        except Exception:
            return int(block.page or 0)

    # Titles learned only from explicit source markers.  They are safe boundary
    # evidence without running chapter detection in the extraction/format stage.
    known_bases: set[str] = set()
    for source_block in blocks:
        info = _named_note_info(_text(source_block))
        if info:
            known_bases.add(info[0])

    # Notes longer than this are unusual.  Beyond the cap we deliberately fall
    # back to same-page deletion rather than risk deleting real book content.
    max_page_span = 12

    def _safe_boundary(start: int, base: str, kind: str) -> int | None:
        start_page = _physical_page(blocks[start])
        for j in range(start + 1, len(blocks)):
            candidate = blocks[j]
            page = _physical_page(candidate)
            if start_page and page and page - start_page > max_page_span:
                break
            text = _text(candidate)
            info = _named_note_info(text)
            if kind.startswith("前書"):
                if text == base:
                    return j
                # Another explicit note before the bare title means the source
                # is malformed/ambiguous; do not jump across it.
                if info:
                    break
            else:
                # Cross-page afterwords may continue for several generated PDF
                # pages.  Only the next *explicit named preface* proves where a
                # new episode starts.  Do not jump to a later bare title: a real
                # intervening episode may have no preface marker at all (as in
                # the dog-book regression around physical page 35).
                if info and info[1].startswith("前書"):
                    return j
        return None

    i = 0
    while i < len(blocks):
        block = blocks[i]
        text = _text(block)
        note = _named_note_info(text)
        plain_note = bool(PDF_PLAIN_NOTE_HEADING_RE.fullmatch(text))

        if note:
            base, kind = note
            boundary = _safe_boundary(i, base, kind)
            if boundary is not None:
                # Drop the marker and all note content before the proven
                # boundary.  Boundary itself is processed normally on next loop.
                removed += max(1, boundary - i)
                i = boundary
                continue

            # No proven cross-page boundary: safe fallback is the historical
            # same-page behaviour.  Remove marker + note fragments on that page
            # only, then preserve everything from the next physical page.
            start_page = _physical_page(block)
            removed += 1
            i += 1
            while i < len(blocks):
                page = _physical_page(blocks[i])
                if start_page and page and page > start_page:
                    break
                removed += 1
                i += 1
            continue

        if plain_note and text in {"あとがき", "後書き", "後書"}:
            # Plain terminal afterword heading: use the next explicit named
            # preface as the only cross-page proof; otherwise same-page only.
            start_page = _physical_page(block)
            boundary = None
            for j in range(i + 1, len(blocks)):
                page = _physical_page(blocks[j])
                if start_page and page and page - start_page > max_page_span:
                    break
                info = _named_note_info(_text(blocks[j]))
                if info and info[1].startswith("前書"):
                    boundary = j
                    break
            if boundary is not None:
                removed += max(1, boundary - i)
                i = boundary
                continue
            removed += 1
            i += 1
            while i < len(blocks):
                page = _physical_page(blocks[i])
                if start_page and page and page > start_page:
                    break
                removed += 1
                i += 1
            continue

        result.append(block)
        i += 1

    out.blocks = result
    return out, removed

def _remove_explicit_retired_duplicate_sections(doc: UnifiedDocument, *, inplace: bool = False) -> tuple[UnifiedDocument, int, Counter[str]]:
    """Drop only later duplicate chapters explicitly labelled ``削除予定``.

    Safety rule: a retired chapter starts a removable segment only when its
    canonical title has already appeared earlier in the same document.  The
    annotation alone is not enough: this preserves the first/only copy even if
    the source site labelled it for future deletion.  Once such a duplicate
    segment starts, its body is skipped through subsequent retired chapters;
    the run stops at the next non-retired named note or non-retired chapter.
    """
    out = doc if inplace else copy.deepcopy(doc)
    result: list[Block] = []
    removed_blocks = 0
    removed_counts: Counter[str] = Counter()
    seen_titles: set[str] = set()
    skipping = False

    def remember(text: str) -> None:
        if not text or len(text) > 160 or PDF_RETIRED_MARK_RE.search(text):
            return
        seen_titles.add(_canonical_retired_title(text))

    def drop(block: Block) -> None:
        nonlocal removed_blocks
        removed_blocks += 1
        if block.type in _TEXT_TYPES and not (block.metadata or {}).get("pdf_text_exclude_from_guard"):
            removed_counts.update(_compact_guard_text(block.text or ""))

    for block in out.blocks:
        text = _text(block)
        is_retired = bool(PDF_RETIRED_MARK_RE.search(text))
        hard_deleted = bool(is_retired and "削除予定" not in text)
        canonical = _canonical_retired_title(text) if text else ""
        named_note = _named_note_info(text)

        # Range headers such as ``第１章第７話〜第１５話（２月１０日削除予定）``
        # are editorial wrappers, not story text.  Removing the wrapper only is
        # safe even when the following chapter copy is the one we keep.
        if PDF_RETIRED_RANGE_MARK_RE.fullmatch(text):
            drop(block)
            continue

        if skipping:
            # A named preface without a retirement marker is a strong boundary
            # for the next live episode.  Preserve it and leave duplicate mode.
            if named_note and not is_retired:
                skipping = False
            elif PDF_PLAIN_NOTE_HEADING_RE.fullmatch(text):
                drop(block)
                continue
            elif block.type == BlockType.CHAPTER:
                if is_retired and canonical and (hard_deleted or canonical in seen_titles):
                    drop(block)
                    continue
                if not is_retired:
                    skipping = False

            if skipping:
                drop(block)
                continue

        if block.type == BlockType.CHAPTER and is_retired and canonical and (hard_deleted or canonical in seen_titles):
            skipping = True
            drop(block)
            continue

        result.append(block)
        remember(text)

    out.blocks = result
    return out, removed_blocks, removed_counts


def _dedupe_repeated_pdf_section_headers(doc: UnifiedDocument, *, inplace: bool = False) -> tuple[UnifiedDocument, int, Counter[str]]:
    """Keep one structural copy of a repeated ``第X章...`` page header."""
    out = doc if inplace else copy.deepcopy(doc)
    counts = Counter(
        _text(block) for block in out.blocks
        if PDF_REPEATED_SECTION_RE.fullmatch(_text(block))
    )
    repeated = {text for text, count in counts.items() if count >= 2}
    if not repeated:
        return out, 0, Counter()

    result: list[Block] = []
    seen: set[str] = set()
    removed = 0
    removed_counts: Counter[str] = Counter()
    for block in out.blocks:
        text = _text(block)
        if text not in repeated:
            result.append(block)
            continue
        if text not in seen:
            kept = copy.deepcopy(block)
            kept.type = BlockType.SECTION
            kept.metadata = dict(kept.metadata or {})
            kept.metadata["pdf_text_repeated_section_heading"] = True
            kept.metadata["exclude_from_sentence_merge"] = True
            result.append(kept)
            seen.add(text)
            continue
        removed += 1
        if block.type in _TEXT_TYPES and not (block.metadata or {}).get("pdf_text_exclude_from_guard"):
            removed_counts.update(_compact_guard_text(block.text or ""))

    out.blocks = result
    return out, removed, removed_counts


def preserve_pdf_afterwords(doc: UnifiedDocument, *, inplace: bool = False) -> UnifiedDocument:
    """Keep author prefaces/afterwords unless the explicit PDF option is disabled."""
    if bool(getattr(doc.metadata, "pdf_keep_afterwords", True)):
        out = doc if inplace else copy.deepcopy(doc)
        out.add_log("strip_chapter_notes", "PDF文字层：保留作者前书/后记（可在界面关闭）", 0)
        return out
    # Two PDFNovels generations exist in the wild.  Older exports use numeric
    # sentinels such as ``001（前書）``; newer exports use named markers such as
    # ``王女は復讐に身を焦がす（前書き）``.  Running the old numeric state
    # machine on a named-marker document is unsafe after PDF page numbers have
    # already been filtered, because its historical "bare number ends note"
    # sentinel may no longer exist.  Select exactly one protocol from evidence
    # present in the source instead of stacking both heuristics.
    has_named_notes = any(_named_note_info(_text(block)) for block in doc.blocks)
    if has_named_notes:
        out, named_removed = _strip_named_pdf_chapter_notes(doc, inplace=inplace)
    else:
        from engine.formatter import strip_chapter_notes
        out = strip_chapter_notes(doc)
        named_removed = 0
    # Intentional removal is not treated as accidental character loss.
    source_counts = Counter(getattr(out.metadata, "pdf_text_source_char_counts", {}) or {})
    current_counts = _counter(block.text for block in out.blocks if block.type in _TEXT_TYPES)
    removed_counts = source_counts - current_counts
    out.metadata.pdf_text_source_char_counts = dict(source_counts - removed_counts)
    out.metadata.pdf_text_source_chars = sum(out.metadata.pdf_text_source_char_counts.values())
    out.add_log("strip_named_chapter_notes", f"PDF文字层：删除 {named_removed} 个具名前书/后书块", named_removed)
    return out


def _strip_pdfnovels_generated_front_matter(doc: UnifiedDocument, *, inplace: bool = False) -> tuple[UnifiedDocument, int, Counter[str]]:
    """Remove only strongly identified PDFNovels/Narou generated front matter.

    This is deliberately source-specific rather than a fuzzy "drop everything
    before chapter one" heuristic.  We require the PDFNovels site signature plus
    multiple labelled metadata fields, copy title/author into document metadata,
    then remove only the generated prefix before ``序章``/``プロローグ`` or the
    first live chapter.
    """
    out = doc if inplace else copy.deepcopy(doc)
    texts = [_text(block) for block in out.blocks]
    signature = any("pdfnovels.net" in text or "タテ書き小説ネット" in text for text in texts[:80])
    labels = {"【小説タイトル】", "【Ｎコード】", "【作者名】", "【あらすじ】"}
    seen_labels = {text for text in texts[:80] if text in labels}
    if not signature or len(seen_labels) < 3:
        return out, 0, Counter()

    def value_after(label: str) -> str:
        try:
            index = texts.index(label, 0, min(80, len(texts)))
        except ValueError:
            return ""
        for value in texts[index + 1:min(index + 5, len(texts))]:
            value = value.strip(" \t\r\n　")
            if value and value not in labels:
                return value
        return ""

    title = value_after("【小説タイトル】")
    author = value_after("【作者名】")
    if title:
        out.metadata.title = title
    if author:
        out.metadata.author = author

    start = None
    for index, block in enumerate(out.blocks[:200]):
        text = _text(block)
        if text in {"序章", "プロローグ"}:
            start = index
            break
        if block.type == BlockType.CHAPTER and text and not PDF_RETIRED_MARK_RE.search(text):
            start = index
            break
    if start is None or start <= 0:
        return out, 0, Counter()

    removed_counts: Counter[str] = Counter()
    for block in out.blocks[:start]:
        if block.type in _TEXT_TYPES and not (block.metadata or {}).get("pdf_text_exclude_from_guard"):
            removed_counts.update(_compact_guard_text(block.text or ""))
    out.blocks = out.blocks[start:]
    return out, start, removed_counts


def _has_pdfnovels_signature(doc: UnifiedDocument) -> bool:
    texts = [_text(block) for block in doc.blocks[:120]]
    site = any("pdfnovels.net" in text or "タテ書き小説ネット" in text for text in texts)
    labels = {"【小説タイトル】", "【Ｎコード】", "【作者名】", "【あらすじ】"}
    return site and len({text for text in texts if text in labels}) >= 3


def _strip_pdfnovels_generated_back_matter(
    doc: UnifiedDocument,
    *,
    source_signature: bool,
    inplace: bool = False,
) -> tuple[UnifiedDocument, int, Counter[str]]:
    """Remove the PDFNovels generated terminal information page.

    The final PDFNovels page can mix horizontal URL/date text with vertical
    explanatory prose.  In extraction order those horizontal glyphs may be
    fragmented before the clear ``ＰＤＦ小説ネット発足にあたって`` heading,
    so once the heading is proven on the terminal physical page we remove the
    *whole page*, not only blocks after the heading.
    """
    out = doc if inplace else copy.deepcopy(doc)
    if not source_signature or not out.blocks:
        return out, 0, Counter()

    tail = out.blocks[-240:]
    terminal_page = max((int(block.page or 0) for block in out.blocks), default=0)
    generated_page = 0
    strong_markers = (
        "ＰＤＦ小説ネット発足にあたって",
        "PDF小説ネット発足にあたって",
        "この小説の詳細については以下のＵＲＬをご覧ください",
    )
    for block in tail:
        text = _text(block)
        if any(marker in text for marker in strong_markers):
            page = int(block.page or 0)
            if terminal_page and page >= max(1, terminal_page - 1):
                generated_page = page
                break
    if not generated_page:
        return out, 0, Counter()

    result: list[Block] = []
    removed = 0
    removed_counts: Counter[str] = Counter()
    for block in out.blocks:
        if int(block.page or 0) == generated_page:
            removed += 1
            if block.type in _TEXT_TYPES and not (block.metadata or {}).get("pdf_text_exclude_from_guard"):
                removed_counts.update(_compact_guard_text(block.text or ""))
            continue
        result.append(block)
    out.blocks = result
    return out, removed, removed_counts


def preserve_pdf_boilerplate(doc: UnifiedDocument, *, inplace: bool = False) -> UnifiedDocument:
    """Preserve source prose; remove only proven generated/retired matter.

    Site-generated front/back matter is independent from the user's decision to
    keep author-written prefaces/afterwords.  The old coupling meant enabling
    "保留作者前书/后记" also resurrected PDFNovels copyright/promo pages.
    """
    remove_generated = bool(getattr(doc.metadata, "pdf_remove_generated_matter", True))
    source_signature = _has_pdfnovels_signature(doc) if remove_generated else False
    if remove_generated:
        out, front_removed, removed_counts = _strip_pdfnovels_generated_front_matter(doc, inplace=inplace)
    else:
        out, front_removed, removed_counts = (doc if inplace else copy.deepcopy(doc)), 0, Counter()
    out, back_removed, back_counts = _strip_pdfnovels_generated_back_matter(
        out, source_signature=source_signature, inplace=True
    )
    removed_counts.update(back_counts)
    out, removed, retired_counts = _remove_explicit_retired_duplicate_sections(out, inplace=True)
    removed_counts.update(retired_counts)
    out, section_removed, section_removed_counts = _dedupe_repeated_pdf_section_headers(out, inplace=True)
    removed_counts.update(section_removed_counts)
    if removed_counts:
        expected = Counter(getattr(out.metadata, "pdf_text_source_char_counts", {}) or {})
        expected.subtract(removed_counts)
        expected = Counter({ch: count for ch, count in expected.items() if count > 0})
        out.metadata.pdf_text_source_char_counts = dict(expected)
        out.metadata.pdf_text_source_chars = sum(expected.values())
    if front_removed or removed or section_removed:
        out.add_log(
            "strip_boilerplate",
            (
                f"PDF文字层：删除 {front_removed} 个明确 PDFNovels 生成前置块、"
                f"{back_removed} 个明确 PDFNovels 生成末页块、"
                f"{removed} 个明确退役/重复块，合并 {section_removed} 个重复章标题页眉；"
                f"其余正文保持不变"
            ),
            front_removed + back_removed + removed + section_removed,
        )
    else:
        out.add_log("strip_boilerplate", "PDF文字层无损模式：未发现可证明的站点生成前置页或『削除予定』重复章节；跳过模糊样板删除", 0)
    return out


def normalize_pdf_text_punctuation(doc: UnifiedDocument, *, inplace: bool = False) -> UnifiedDocument:
    """Whitespace-only PDF normalisation; preserve ellipsis and original wording."""
    out = doc if inplace else copy.deepcopy(doc)
    changed = 0
    for block in out.blocks:
        if block.type not in _TEXT_TYPES:
            continue
        original = block.text or ""
        value = original.replace("\r\n", "\n").replace("\r", "\n").rstrip(" \t　")
        if value != original:
            block.ocr_raw = block.ocr_raw or original
            block.text = value
            block.modified_by = _append_modified_by(block.modified_by, "normalize_pdf_text_punctuation")
            changed += 1
    out.add_log("normalize_punctuation", f"PDF文字层忠实标点模式：仅清理 {changed} 处尾随空白，未压缩省略号或改写原文", changed)
    return out


def preserve_pdf_orphan_quotes(doc: UnifiedDocument) -> UnifiedDocument:
    out = copy.deepcopy(doc)
    blocks, repaired = _repair_orphan_quote_boundaries(out.blocks)
    unresolved = 0
    for block in blocks:
        token = _text(block)
        if token and set(token) <= _ORPHAN_CLOSE:
            block.metadata = dict(block.metadata or {})
            flags = list(block.metadata.get("pdf_text_review_flags") or [])
            if "orphan_closing_quote" not in flags:
                flags.append("orphan_closing_quote")
            block.metadata["pdf_text_review_flags"] = flags
            unresolved += 1
    out.blocks = blocks
    out.add_log("remove_orphan_closing_quotes", f"PDF文字层：接回 {repaired} 处孤立闭引号，保留并标记 {unresolved} 处不确定引号", repaired + unresolved)
    return out


def skip_pdf_cross_page_merge(doc: UnifiedDocument) -> UnifiedDocument:
    out = copy.deepcopy(doc)
    out.add_log("cross_page_merge", "PDF文字层已在无损预处理阶段跨页接续；跳过通用跨页推断", 0)
    return out


def skip_pdf_dialogue_auto_close(doc: UnifiedDocument) -> UnifiedDocument:
    out = copy.deepcopy(doc)
    flagged = 0
    for block in out.blocks:
        if block.type not in _JOINABLE_TYPES:
            continue
        text = _text(block)
        if any(_quote_balance(text, opener, closer) != 0 for opener, closer in (("「", "」"), ("『", "』"))):
            block.metadata = dict(block.metadata or {})
            flags = list(block.metadata.get("pdf_text_review_flags") or [])
            if "unbalanced_quote" not in flags:
                flags.append("unbalanced_quote")
            block.metadata["pdf_text_review_flags"] = flags
            flagged += 1
    out.add_log("repair_dialogue_quotes", f"PDF文字层不逐块猜补闭引号；标记 {flagged} 个不平衡块供对照复核", flagged)
    return out


def skip_pdf_sentence_merge(doc: UnifiedDocument) -> UnifiedDocument:
    out = copy.deepcopy(doc)
    out.add_log("merge_sentences", "PDF文字层物理列已先行接回；跳过普通OCR短块/接续词推断", 0)
    return out




def restore_pdf_indents(doc: UnifiedDocument, *, inplace: bool = False) -> UnifiedDocument:
    """Add visual paragraph indents without merging scene markers or neighbours."""
    out = doc if inplace else copy.deepcopy(doc)
    changed = 0
    for block in out.blocks:
        if block.type != BlockType.PARAGRAPH or _is_structural_block(block):
            continue
        original = block.text or ""
        stripped = original.lstrip(" \t　")
        if not stripped or stripped.startswith(("「", "『")):
            continue
        value = "　" + stripped
        if value != original:
            block.text = value
            block.modified_by = _append_modified_by(block.modified_by, "restore_pdf_indents")
            changed += 1
    out.add_log("restore_indents", f"PDF文字层安全缩进：调整 {changed} 个正文段，不合并相邻块", changed)
    return out


def _update_guard(out: UnifiedDocument) -> tuple[int, int, bool]:
    expected = Counter(getattr(out.metadata, "pdf_text_source_char_counts", {}) or {})
    actual = _counter(
        block.text for block in out.blocks
        if block.type in _TEXT_TYPES and not (block.metadata or {}).get("pdf_text_exclude_from_guard")
    )
    missing = expected - actual
    extra = actual - expected
    missing_count = sum(missing.values())
    extra_count = sum(extra.values())
    passed = missing_count == 0 and extra_count == 0
    out.metadata.pdf_text_output_chars = sum(actual.values())
    out.metadata.pdf_text_missing_chars = missing_count
    out.metadata.pdf_text_extra_chars = extra_count
    out.metadata.pdf_text_character_guard_passed = passed
    out.metadata.pdf_text_guard_report = {
        "source_chars": sum(expected.values()),
        "output_chars": sum(actual.values()),
        "missing_chars": missing_count,
        "extra_chars": extra_count,
        "passed": passed,
        "missing_preview": "".join(ch * min(count, 3) for ch, count in missing.most_common(20)),
        "extra_preview": "".join(ch * min(count, 3) for ch, count in extra.most_common(20)),
    }
    return missing_count, extra_count, passed


def _restore_pdf_chapter_structure(doc: UnifiedDocument) -> tuple[UnifiedDocument, int]:
    """Rebuild chapter types/TOC *after* retired-note cleanup.

    PDF cleanup can delete an explicitly retired duplicate chapter run after the
    ordinary ``detect_chapters`` step has already populated ``doc.toc``.  The
    surviving live copy may still be a PARAGRAPH because an earlier duplicate or
    same-page structural title made the first chapter scan ambiguous.  Re-run the
    conservative detector on the actual surviving blocks so EPUB/navigation never
    points at deleted titles.

    ``序章`` + ``プロローグ`` (and the corresponding epilogue pair) can legally
    share one physical page.  Treat the Japanese stage label as SECTION first so
    the episode title is not suppressed as a fake TOC page merely because two
    structure labels occur on that page.
    """
    out = copy.deepcopy(doc)
    adjusted = 0
    pairs = {("序章", "プロローグ"), ("終章", "エピローグ")}
    for index, block in enumerate(out.blocks[:-1]):
        left = _text(block)
        right_block = out.blocks[index + 1]
        right = _text(right_block)
        if (left, right) not in pairs or block.page != right_block.page:
            continue
        if block.type != BlockType.SECTION:
            block.type = BlockType.SECTION
            block.metadata = dict(block.metadata or {})
            block.metadata["pdf_text_stage_heading"] = True
            block.metadata["exclude_from_sentence_merge"] = True
            adjusted += 1

    # Runtime import avoids a module-import cycle; by finalize time formatter is
    # fully loaded.  detect_chapters clears stale TOC entries before rebuilding.
    from engine.formatter import detect_chapters
    out = detect_chapters(out)
    return out, adjusted


def _defer_pdf_chapter_detection_to_ai(doc: UnifiedDocument, *, inplace: bool = False) -> tuple[UnifiedDocument, int]:
    """Keep title text, but deliberately leave chapter/TOC decisions to AI.

    Selectable-PDF formatting owns lossless text cleanup and geometry only.
    Rule-based chapter guesses are useful internally while removing named notes,
    but they must not become publication navigation before the later AI chapter
    pass.  Preserve their provenance as metadata, demote them to ordinary text,
    clear chapter indexes/TOC, and let AI make the final structural decision.
    """
    out = doc if inplace else copy.deepcopy(doc)
    candidates = 0
    for block in out.blocks:
        if block.type == BlockType.CHAPTER:
            block.metadata = dict(block.metadata or {})
            block.metadata["pdf_text_pre_ai_structure_type"] = "chapter"
            block.metadata["pdf_text_chapter_candidate"] = True
            block.type = BlockType.PARAGRAPH
            candidates += 1
        if block.chapter_index:
            block.chapter_index = 0
    out.toc = []
    out.metadata.pdf_text_chapter_candidates_deferred = candidates
    out.metadata.pdf_chapter_detection_deferred_to_ai = True
    return out, candidates


def finalize_pdf_text_layer(doc: UnifiedDocument, *, inplace: bool = False) -> UnifiedDocument:
    """Finish PDF geometry/text cleanup, defer chapter/TOC detection, verify coverage."""
    out = doc if inplace else copy.deepcopy(doc)
    blocks, simultaneous = _repair_simultaneous_speech(out.blocks)
    blocks, orphan = _repair_orphan_quote_boundaries(blocks)
    # Lexical/semantic anomaly heuristics belong to the AI review stage.  The
    # previous seam regex still produced false positives on valid compounds
    # such as ``照らし / 出す`` in the 4693-page stress test.  The PDF format
    # stage therefore limits itself to geometry + character conservation.
    flagged = 0
    out.blocks = blocks
    out.metadata.pdf_text_semantic_review_deferred_to_ai = True
    out, chapter_candidates = _defer_pdf_chapter_detection_to_ai(out, inplace=True)
    missing, extra, passed = _update_guard(out)
    guard_text = "字符保全通过" if passed else f"疑似丢失 {missing} 字、额外 {extra} 字"
    out.add_log(
        "pdf_text_finalize",
        (f"PDF文字层收尾：修复 {simultaneous + orphan} 处引号边界，"
         f"物理列接缝仅做几何重组，疑难语义交由 AI；章节/目录交由 AI 识别，"
         f"保留 {chapter_candidates} 个规则候选的文字与来源证据；{guard_text}"),
        simultaneous + orphan + flagged + missing + extra,
    )
    return out
