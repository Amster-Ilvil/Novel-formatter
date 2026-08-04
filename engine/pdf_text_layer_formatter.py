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
PDF_SCENE_MARKER_RE = re.compile(r"^[\s　]*(?:◯|○|●|◇|◆|＊|\*{1,3})[\s　]*$")

_VERTICAL_GLYPH_RE = re.compile(
    r"^[\u3040-\u30ff\u3400-\u9fff々〃〆ヶヵー、。！？!?…‥—―「」『』（）〈〉《》【】・：；\d０-９]+$"
)
_SHORT_TAIL_RE = re.compile(
    r"^[ぁ-んァ-ヶ一-龯々〃〆ヶヵー]{1,8}[。！？!?）」』】]?$"
)
_STRONG_END_RE = re.compile(r"[。．！？!?‼⁉…‥—―」』）)】》]$")
_JAPANESE_START_RE = re.compile(r"^[ぁ-んァ-ヶ一-龯々〃〆ヶヵー]")

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


def _append_modified_by(value: str, step: str) -> str:
    parts = [part for part in str(value or "").split(",") if part]
    if step not in parts:
        parts.append(step)
    return ",".join(parts)


def _compact_guard_text(text: str) -> str:
    """Characters used by the conservation guard; layout whitespace is ignored."""
    return re.sub(r"[\s　]+", "", str(text or ""))


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
    if metadata.get("pdf_text_asset_marker") or metadata.get("pdf_text_image_caption"):
        return True
    return bool(
        PDF_ASSET_MARKER_RE.fullmatch(text)
        or PDF_IMAGE_CAPTION_RE.fullmatch(text)
        or PDF_AFTERWORD_MARKER_RE.fullmatch(text)
        or PDF_BARE_CHAPTER_RE.fullmatch(text)
        or PDF_SCENE_MARKER_RE.fullmatch(text)
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
    if not _is_text_block(left) or not _is_text_block(right):
        return False, False
    if _is_structural_block(left) or _is_structural_block(right):
        return False, False

    left_text = _text(left)
    right_text = _text(right)
    if not left_text or not right_text:
        return False, False

    # An actually unclosed outer quote takes priority over any inline quote at
    # the column end.  Join without deleting the inline closer.
    if _has_unclosed_quote(left_text):
        if right_text.startswith(("「", "『")) and not right_text.startswith(("「「", "『『")):
            return False, False
        return True, False

    if _premature_quote_continuation(left_text, right_text):
        return True, True

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

    if right_text.startswith(("「", "『")):
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
        if not _is_text_block(current) or _is_structural_block(current):
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


def restore_pdf_dialogue_columns(doc: UnifiedDocument) -> UnifiedDocument:
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
    out = copy.deepcopy(doc)
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
    """Flag likely lost-glyph glue for manual comparison; never guess missing text."""
    text = str(block.text or "")
    patterns = (
        r"(?:であ|らし|なかっ|いなかっ|知らな|ことにな)[\u3400-\u9fff]",
        r"[\u3400-\u9fff](?:僕|私|俺)(?=(?:は|が|を|に|も))",
    )
    if not any(re.search(pattern, text) for pattern in patterns):
        return 0
    block.metadata = dict(block.metadata or {})
    flags = list(block.metadata.get("pdf_text_review_flags") or [])
    if "possible_missing_text_glue" not in flags:
        flags.append("possible_missing_text_glue")
    block.metadata["pdf_text_review_flags"] = flags
    return 1


def _set_source_guard(doc: UnifiedDocument) -> None:
    source_blocks = [
        block for block in doc.blocks
        if block.type in _TEXT_TYPES and not (block.metadata or {}).get("pdf_text_exclude_from_guard")
    ]
    counts = _counter(block.ocr_raw or block.text or "" for block in source_blocks)
    doc.metadata.pdf_text_source_char_counts = dict(counts)
    doc.metadata.pdf_text_source_chars = sum(counts.values())
    doc.metadata.pdf_text_guard_report = {
        "source_chars": sum(counts.values()),
        "output_chars": sum(counts.values()),
        "missing_chars": 0,
        "extra_chars": 0,
        "passed": True,
    }


def prepare_pdf_text_layer(doc: UnifiedDocument) -> UnifiedDocument:
    """Preserve source, classify structure, and reconstruct physical columns."""
    out = copy.deepcopy(doc)
    changed = 0
    markers = 0
    afterwords = 0

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

    _set_source_guard(out)

    # This deterministic four-block reorder must happen before the generic
    # physical join, otherwise the subject glyph would be swallowed into the
    # simultaneous-speech quote.
    blocks, simultaneous = _repair_simultaneous_speech(out.blocks)
    blocks, joined = _join_pdf_physical_columns(blocks)
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
            f"修复 {simultaneous + orphan} 处引号边界，标记 {markers} 个资源编号、{afterwords} 段后记"
        ),
        changed + joined + simultaneous + orphan + markers + afterwords,
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


def remove_pdf_coordinate_duplicates(doc: UnifiedDocument) -> UnifiedDocument:
    """Remove only exact same-page text whose bounding boxes overlap strongly."""
    out = copy.deepcopy(doc)
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


def preserve_pdf_afterwords(doc: UnifiedDocument) -> UnifiedDocument:
    """Keep author prefaces/afterwords unless the explicit PDF option is disabled."""
    if bool(getattr(doc.metadata, "pdf_keep_afterwords", True)):
        out = copy.deepcopy(doc)
        out.add_log("strip_chapter_notes", "PDF文字层：保留作者前书/后记（可在界面关闭）", 0)
        return out
    from engine.formatter import strip_chapter_notes
    out = strip_chapter_notes(doc)
    # Intentional removal is not treated as accidental character loss.
    source_counts = Counter(getattr(out.metadata, "pdf_text_source_char_counts", {}) or {})
    current_counts = _counter(block.text for block in out.blocks if block.type in _TEXT_TYPES)
    removed_counts = source_counts - current_counts
    out.metadata.pdf_text_source_char_counts = dict(source_counts - removed_counts)
    out.metadata.pdf_text_source_chars = sum(out.metadata.pdf_text_source_char_counts.values())
    return out


def preserve_pdf_boilerplate(doc: UnifiedDocument) -> UnifiedDocument:
    """Do not silently delete selectable-PDF text; users can remove it manually."""
    out = copy.deepcopy(doc)
    out.add_log("strip_boilerplate", "PDF文字层无损模式：跳过网站样板/尾部模糊删除，避免误删正文或后记", 0)
    return out


def normalize_pdf_text_punctuation(doc: UnifiedDocument) -> UnifiedDocument:
    """Whitespace-only PDF normalisation; preserve ellipsis and original wording."""
    out = copy.deepcopy(doc)
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




def restore_pdf_indents(doc: UnifiedDocument) -> UnifiedDocument:
    """Add visual paragraph indents without merging scene markers or neighbours."""
    out = copy.deepcopy(doc)
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


def finalize_pdf_text_layer(doc: UnifiedDocument) -> UnifiedDocument:
    """Finish quote repairs, flag unresolved glue, and verify character coverage."""
    out = copy.deepcopy(doc)
    blocks, simultaneous = _repair_simultaneous_speech(out.blocks)
    blocks, orphan = _repair_orphan_quote_boundaries(blocks)
    flagged = 0
    for block in blocks:
        if _is_text_block(block):
            flagged += _mark_suspicious_glue(block)
    out.blocks = blocks
    missing, extra, passed = _update_guard(out)
    guard_text = "字符保全通过" if passed else f"疑似丢失 {missing} 字、额外 {extra} 字"
    out.add_log(
        "pdf_text_finalize",
        f"PDF文字层收尾：修复 {simultaneous + orphan} 处引号边界，标记 {flagged} 处疑似粘连；{guard_text}",
        simultaneous + orphan + flagged + missing + extra,
    )
    return out
