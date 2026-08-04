# -*- coding: utf-8 -*-
"""Layout reconstruction for authoritative strict replacement.

The replacement source remains the only text source. OCR blocks are used only as
layout/page/type references.  No OCR word is copied into the replacement text.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field

from engine.japanese_normalizer import MAP
from models.document import BlockType, UnifiedDocument
from models.paragraph import Paragraph

_TEXT_TYPES = {BlockType.PARAGRAPH, BlockType.DIALOGUE, BlockType.CHAPTER, BlockType.SECTION}
_OPEN_DIALOGUE = "「“"
_CLOSE_DIALOGUE = "」”"
# Only structural Japanese dialogue quotes are layout-neutral. Plain ASCII quotes
# can be semantic source content and must never be silently ignored.
_LAYOUT_QUOTES = "「」“”"
_DASHES = "—―─ー−ｰ"


@dataclass
class LayoutReferenceUnit:
    text: str
    type: BlockType
    page: int
    block_indices: list[int] = field(default_factory=list)


@dataclass
class ReflowSegment:
    text: str
    type: BlockType
    page: int
    source_index: int
    confidence: float = 1.0
    reference_blocks: list[int] = field(default_factory=list)
    quote_insertions: int = 0
    quote_moves: int = 0
    method: str = "literal"


@dataclass
class LayoutAnalysis:
    overlong_blocks: int = 0
    very_long_blocks: int = 0
    mixed_dialogue_blocks: int = 0
    unbalanced_dialogue_blocks: int = 0
    suspicious_blocks: list[int] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not (
            self.overlong_blocks
            or self.mixed_dialogue_blocks
            or self.unbalanced_dialogue_blocks
        )


@dataclass
class ReflowStats:
    source_blocks: int = 0
    output_blocks: int = 0
    reflowed_source_blocks: int = 0
    projected_segments: int = 0
    rule_segments: int = 0
    unresolved_blocks: int = 0
    quote_insertions: int = 0
    quote_moves: int = 0
    before: LayoutAnalysis = field(default_factory=LayoutAnalysis)
    after: LayoutAnalysis = field(default_factory=LayoutAnalysis)


def _dialogue_counts(text: str) -> tuple[int, int]:
    value = str(text or "")
    return (
        sum(value.count(ch) for ch in _OPEN_DIALOGUE),
        sum(value.count(ch) for ch in _CLOSE_DIALOGUE),
    )


def _dialogue_spans(text: str) -> list[tuple[int, int]]:
    """Return balanced Japanese dialogue-quote spans."""
    value = str(text or "")
    spans: list[tuple[int, int]] = []
    stack: list[tuple[str, int]] = []
    pairs = {"「": "」", "“": "”"}
    for index, char in enumerate(value):
        if char in pairs:
            stack.append((char, index))
            continue
        if char in _CLOSE_DIALOGUE and stack:
            opening, start = stack[-1]
            if pairs.get(opening) == char:
                stack.pop()
                if not stack:
                    spans.append((start, index + 1))
    return spans


def _looks_like_inline_quoted_term(value: str, start: int, end: int) -> bool:
    """Distinguish quoted nouns/terms from standalone spoken dialogue."""
    inner = value[start + 1:end - 1].strip()
    if not inner or len(inner) > 16 or any(ch in inner for ch in "。！？!?…"):
        return False
    prefix = value[:start].rstrip()
    suffix = value[end:].lstrip()
    if prefix and suffix:
        return True
    # Typical Japanese inline construction: 「魔王」と呼ぶ / 「力」の正体.
    if suffix.startswith(("と", "を", "が", "は", "の", "に", "で", "へ", "も", "って", "という", "など")):
        return True
    return False


def _is_mixed_dialogue(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    spans = _dialogue_spans(value)
    spoken_spans = [span for span in spans if not _looks_like_inline_quoted_term(value, *span)]
    if spans and not spoken_spans and _dialogue_counts(value)[0] == _dialogue_counts(value)[1]:
        return False
    if len(spoken_spans) >= 2:
        return True
    if len(spoken_spans) == 1:
        start, end = spoken_spans[0]
        prefix = value[:start].strip()
        suffix = value[end:].strip()
        # A complete quote which is the whole block is a normal dialogue block.
        if not prefix and not suffix:
            return False
        # Narration plus a spoken quote (or spoken quote plus narration) is mixed.
        if len(prefix) > 0 or len(suffix) > 0:
            return True
    # Unbalanced quote appearing well inside prose is suspicious even before the
    # separate unbalanced counter records it.
    first_open = min((value.find(ch) for ch in _OPEN_DIALOGUE if value.find(ch) >= 0), default=-1)
    last_close = max((value.rfind(ch) for ch in _CLOSE_DIALOGUE), default=-1)
    if first_open > 12 or (last_close >= 0 and len(value) - last_close - 1 > 12):
        return True
    return False


def analyze_layout_texts(texts: list[str]) -> LayoutAnalysis:
    result = LayoutAnalysis()
    for index, raw in enumerate(texts):
        text = str(raw or "").strip()
        if not text:
            continue
        suspicious = False
        if len(text) > 250:
            result.overlong_blocks += 1
            suspicious = True
        if len(text) > 400:
            result.very_long_blocks += 1
        if _is_mixed_dialogue(text):
            result.mixed_dialogue_blocks += 1
            suspicious = True
        opens, closes = _dialogue_counts(text)
        if opens != closes:
            result.unbalanced_dialogue_blocks += 1
            suspicious = True
        if suspicious:
            result.suspicious_blocks.append(index)
    return result


def _normalise_with_map(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    positions: list[int] = []
    for index, char in enumerate(str(text or "")):
        value = unicodedata.normalize("NFKC", char)
        for source, target in MAP.items():
            value = value.replace(source, target)
        for normalised in value:
            if normalised.isspace():
                continue
            chars.append(normalised)
            positions.append(index)
    return "".join(chars), positions


def layout_neutral_text(text: str) -> str:
    """Content stream used to prove reflow did not lose non-layout characters."""
    value = unicodedata.normalize("NFC", str(text or ""))
    return "".join(ch for ch in value if not ch.isspace() and ch not in _LAYOUT_QUOTES)


def _reference_units(doc: UnifiedDocument) -> list[LayoutReferenceUnit]:
    """Create stable OCR layout units and repair only obvious quote continuations."""
    units: list[LayoutReferenceUnit] = []
    for block_index, block in enumerate(doc.blocks):
        if block.type not in _TEXT_TYPES:
            continue
        text = str(block.text or "").strip()
        if not text:
            continue

        # OCR sometimes emits a closing quote as a tiny paragraph. Attach it to the
        # preceding unit so it cannot create a fake paragraph boundary.
        if re.fullmatch(r"[「」“”\"\s　]+", text):
            if units:
                units[-1].text += text
                units[-1].block_indices.append(block_index)
            continue

        # Cross-page dialogue continuation: next block has no opening quote but ends
        # with a closing quote. OCR often also leaves a premature close on the prior
        # block; remove that reference-only close before joining. Replacement text is
        # not changed here.
        if (
            block.type == BlockType.DIALOGUE
            and units
            and units[-1].type == BlockType.DIALOGUE
            and not text.startswith(tuple(_OPEN_DIALOGUE))
        ):
            previous = units[-1]
            if previous.text.endswith(tuple(_CLOSE_DIALOGUE)):
                previous.text = previous.text[:-1]
            previous.text += text
            previous.block_indices.append(block_index)
            continue

        units.append(LayoutReferenceUnit(
            text=text,
            type=block.type,
            page=int(getattr(block, "page", 0) or 0),
            block_indices=[block_index],
        ))
    return units


def _map_candidate_to_source(matches: list[difflib.Match], candidate_pos: int) -> int:
    for match in matches:
        if match.b <= candidate_pos <= match.b + match.size:
            return match.a + min(max(candidate_pos - match.b, 0), match.size)
    left = None
    right = None
    for match in matches:
        if match.b + match.size <= candidate_pos:
            left = match
        if match.b >= candidate_pos:
            right = match
            break
    if left is not None and right is not None:
        left_b = left.b + left.size
        left_a = left.a + left.size
        span = right.b - left_b
        if span > 0:
            return round(left_a + (candidate_pos - left_b) * (right.a - left_a) / span)
    if left is not None:
        return left.a + left.size + candidate_pos - (left.b + left.size)
    if right is not None:
        return right.a - (right.b - candidate_pos)
    return 0


def _snap_source_boundary(text: str, position: int) -> int:
    """Avoid cutting a Japanese em-dash run into two different paragraphs."""
    position = max(0, min(len(text), int(position)))
    left = position
    while left > 0 and text[left - 1] in _DASHES:
        left -= 1
    right = position
    while right < len(text) and text[right] in _DASHES:
        right += 1
    if right - left >= 2 and left <= position <= right:
        return right
    return position


def _combine_reference_units(group: list[LayoutReferenceUnit]) -> LayoutReferenceUnit:
    if not group:
        return LayoutReferenceUnit("", BlockType.PARAGRAPH, 0, [])
    block_type = group[0].type if all(unit.type == group[0].type for unit in group) else BlockType.PARAGRAPH
    return LayoutReferenceUnit(
        text="".join(unit.text for unit in group),
        type=block_type,
        page=group[0].page,
        block_indices=[index for unit in group for index in unit.block_indices],
    )


def _repair_dialogue_quotes(text: str, reference: LayoutReferenceUnit) -> tuple[str, int, int]:
    """Repair only structural dialogue quotes, never words or ordinary punctuation."""
    value = str(text or "").strip()
    if reference.type != BlockType.DIALOGUE or not value:
        return value, 0, 0

    ref = reference.text.strip()
    opening = next((ch for ch in _OPEN_DIALOGUE if ref.startswith(ch)), "「")
    closing = next((ch for ch in _CLOSE_DIALOGUE if ref.endswith(ch)), "」")
    insertions = 0
    moves = 0

    if not value.startswith(tuple(_OPEN_DIALOGUE)):
        value = opening + value
        insertions += 1

    if not value.endswith(tuple(_CLOSE_DIALOGUE)):
        internal = [i for i, ch in enumerate(value) if ch in _CLOSE_DIALOGUE]
        # A single close in the middle is the common cross-page corruption. Move it
        # instead of adding a second closing quote.
        if len(internal) == 1 and 0 < internal[0] < len(value) - 1:
            at = internal[0]
            value = value[:at] + value[at + 1:] + closing
            moves += 1
        else:
            value += closing
            insertions += 1
    return value, insertions, moves


def _safe_quote_split(text: str, source_index: int, page: int) -> list[ReflowSegment]:
    """Fallback: split only clear spoken 「...」 spans without guessing words."""
    value = str(text or "").strip()
    spans = [
        span for span in _dialogue_spans(value)
        if value[span[0]] == "「" and not _looks_like_inline_quoted_term(value, *span)
    ]
    if not spans:
        return []

    result: list[ReflowSegment] = []
    cursor = 0
    for start, end in spans:
        if start > cursor:
            prefix = value[cursor:start].strip()
            if prefix:
                result.append(ReflowSegment(prefix, BlockType.PARAGRAPH, page, source_index, 0.65, method="quote_rules"))
        dialogue = value[start:end].strip()
        if dialogue:
            result.append(ReflowSegment(dialogue, BlockType.DIALOGUE, page, source_index, 0.75, method="quote_rules"))
        cursor = end
    if cursor < len(value):
        suffix = value[cursor:].strip()
        if suffix:
            result.append(ReflowSegment(suffix, BlockType.PARAGRAPH, page, source_index, 0.65, method="quote_rules"))
    return result if len(result) > 1 else []


def _expand_anomalous_segments(segments: list[ReflowSegment]) -> list[ReflowSegment]:
    """Apply deterministic quote splitting to projected segments that remain mixed."""
    expanded: list[ReflowSegment] = []
    for segment in segments:
        if len(segment.text) <= 250 and not _is_mixed_dialogue(segment.text):
            expanded.append(segment)
            continue
        split = _safe_quote_split(segment.text, segment.source_index, segment.page)
        if not split:
            expanded.append(segment)
            continue
        for item in split:
            item.confidence = min(item.confidence, segment.confidence)
            item.reference_blocks = list(segment.reference_blocks)
            item.quote_insertions += segment.quote_insertions
            item.quote_moves += segment.quote_moves
            item.method = segment.method + "+quote_rules"
        expanded.extend(split)
    return expanded


def _clean_adjacent_quote_boundaries(segments: list[ReflowSegment]) -> list[ReflowSegment]:
    """Repair split quoted terms and duplicated structural boundary quotes."""
    cleaned: list[ReflowSegment] = []
    for segment in segments:
        text = str(segment.text or "").strip()
        if not text:
            continue
        if cleaned:
            previous = cleaned[-1]
            previous_text = previous.text.rstrip()

            # A short inline term may have been split by a page/block boundary:
            #   ...名は「魔王  +  」といった。
            # Join the blocks, but never do this for a sentence-like spoken quote.
            prev_opens, prev_closes = _dialogue_counts(previous_text)
            cur_opens, cur_closes = _dialogue_counts(text)
            if prev_opens == prev_closes + 1 and cur_closes == cur_opens + 1:
                open_at = max(previous_text.rfind(ch) for ch in _OPEN_DIALOGUE)
                close_positions = [text.find(ch) for ch in _CLOSE_DIALOGUE if text.find(ch) >= 0]
                close_at = min(close_positions) if close_positions else -1
                quoted = (previous_text[open_at + 1:] + (text[:close_at] if close_at >= 0 else "")).strip()
                if 0 < len(quoted) <= 24 and not any(ch in quoted for ch in "。！？!?…"):
                    previous.text = previous_text + text
                    previous.type = BlockType.PARAGRAPH
                    previous.confidence = min(previous.confidence, segment.confidence)
                    previous.reference_blocks.extend(
                        item for item in segment.reference_blocks if item not in previous.reference_blocks
                    )
                    previous.method += "+join_inline_quote"
                    continue

            # Example: narration ends with an OCR-carried 「 and the repaired next
            # dialogue begins with 「. Keep one structural opener only.
            if previous_text.endswith(tuple(_OPEN_DIALOGUE)) and text.startswith(tuple(_OPEN_DIALOGUE)):
                previous.text = previous_text[:-1].rstrip()
                previous.quote_moves += 1
            if previous.text.endswith(tuple(_CLOSE_DIALOGUE)) and text.startswith(tuple(_CLOSE_DIALOGUE)):
                text = text[1:].lstrip()
                segment.quote_moves += 1
            if not previous.text:
                cleaned.pop()
        segment.text = text
        if segment.text:
            cleaned.append(segment)
    return cleaned


def _project_one(
    text: str,
    source_index: int,
    page: int,
    units: list[LayoutReferenceUnit],
) -> tuple[list[ReflowSegment], float] | None:
    radius = 3 if len(text) > 400 else 2
    candidates = [unit for unit in units if unit.page > 0 and abs(unit.page - page) <= radius]
    if not candidates:
        return None

    source_norm, source_map = _normalise_with_map(text)
    candidate_norm = ""
    ranges: list[tuple[int, int, LayoutReferenceUnit]] = []
    for unit in candidates:
        unit_norm, _ = _normalise_with_map(unit.text)
        start = len(candidate_norm)
        candidate_norm += unit_norm
        ranges.append((start, len(candidate_norm), unit))
    if not source_norm or not candidate_norm:
        return None

    matcher = difflib.SequenceMatcher(None, source_norm, candidate_norm, autojunk=False)
    matches = [match for match in matcher.get_matching_blocks() if match.size >= 4]
    matched_chars = sum(match.size for match in matches)
    ratio = matched_chars / max(len(source_norm), 1)
    if not matches or ratio < 0.70:
        return None

    candidate_min = min(match.b for match in matches)
    candidate_max = max(match.b + match.size for match in matches)

    # Ignore OCR units that have no meaningful equal run in the authoritative text;
    # otherwise an OCR-only scream/header creates a false tiny paragraph.
    relevant: list[tuple[int, int, LayoutReferenceUnit]] = []
    for start, end, unit in ranges:
        if end <= candidate_min or start >= candidate_max:
            continue
        overlap = 0
        for match in matches:
            overlap += max(0, min(end, match.b + match.size) - max(start, match.b))
        unit_length = max(1, end - start)
        if overlap >= 4 or overlap / unit_length >= 0.28:
            relevant.append((start, end, unit))
    if len(relevant) < 2:
        return None

    mapped_boundaries: list[int] = []
    for _start, end, _unit in relevant[:-1]:
        normalised_pos = max(0, min(len(source_norm), _map_candidate_to_source(matches, end)))
        original_pos = source_map[normalised_pos] if normalised_pos < len(source_map) else len(text)
        mapped_boundaries.append(_snap_source_boundary(text, original_pos))

    output: list[ReflowSegment] = []
    last = 0
    group_start = 0
    for boundary_index, original_pos in enumerate(mapped_boundaries):
        if original_pos - last < 2 or len(text) - original_pos < 2:
            continue
        reference = _combine_reference_units([item[2] for item in relevant[group_start:boundary_index + 1]])
        segment_text, inserted, moved = _repair_dialogue_quotes(text[last:original_pos], reference)
        if segment_text:
            output.append(ReflowSegment(
                text=segment_text,
                type=reference.type,
                page=reference.page or page,
                source_index=source_index,
                confidence=min(1.0, ratio),
                reference_blocks=reference.block_indices,
                quote_insertions=inserted,
                quote_moves=moved,
                method="ocr_boundary_projection",
            ))
        last = original_pos
        group_start = boundary_index + 1

    reference = _combine_reference_units([item[2] for item in relevant[group_start:]])
    segment_text, inserted, moved = _repair_dialogue_quotes(text[last:], reference)
    if segment_text:
        output.append(ReflowSegment(
            text=segment_text,
            type=reference.type,
            page=reference.page or page,
            source_index=source_index,
            confidence=min(1.0, ratio),
            reference_blocks=reference.block_indices,
            quote_insertions=inserted,
            quote_moves=moved,
            method="ocr_boundary_projection",
        ))
    return (output, ratio) if len(output) > 1 else None


def _needs_reflow(text: str) -> bool:
    value = str(text or "").strip()
    if len(value) > 160:
        return True
    if _is_mixed_dialogue(value):
        return True
    opens, closes = _dialogue_counts(value)
    return opens != closes


def reflow_source_paragraphs(
    ocr_doc: UnifiedDocument,
    source_paragraphs: list[Paragraph],
    page_map: list[int],
) -> tuple[list[ReflowSegment], ReflowStats]:
    units = _reference_units(ocr_doc)
    stats = ReflowStats(source_blocks=len(source_paragraphs))
    stats.before = analyze_layout_texts([paragraph.text for paragraph in source_paragraphs])
    output: list[ReflowSegment] = []

    for index, paragraph in enumerate(source_paragraphs):
        text = str(paragraph.text or "").strip()
        page = page_map[index] if index < len(page_map) else 0
        if paragraph.is_title:
            output.append(ReflowSegment(text, BlockType.CHAPTER, page, index, method="title"))
            continue

        projected = _project_one(text, index, page, units) if _needs_reflow(text) else None
        if projected:
            segments, _ratio = projected
            segments = _expand_anomalous_segments(segments)
            output.extend(segments)
            stats.reflowed_source_blocks += 1
            stats.projected_segments += len(segments)
            stats.quote_insertions += sum(segment.quote_insertions for segment in segments)
            stats.quote_moves += sum(segment.quote_moves for segment in segments)
            continue

        rule_segments = _safe_quote_split(text, index, page) if _needs_reflow(text) else []
        if rule_segments:
            output.extend(rule_segments)
            stats.reflowed_source_blocks += 1
            stats.rule_segments += len(rule_segments)
            continue

        block_type = BlockType.DIALOGUE if (
            len(text) >= 2 and text.startswith(tuple(_OPEN_DIALOGUE)) and text.endswith(tuple(_CLOSE_DIALOGUE))
        ) else BlockType.PARAGRAPH
        output.append(ReflowSegment(text, block_type, page, index, method="literal"))
        if _needs_reflow(text):
            stats.unresolved_blocks += 1

    output = _clean_adjacent_quote_boundaries(output)
    stats.output_blocks = len(output)
    stats.quote_insertions = sum(segment.quote_insertions for segment in output)
    stats.quote_moves = sum(segment.quote_moves for segment in output)
    stats.after = analyze_layout_texts([segment.text for segment in output])
    return output, stats
