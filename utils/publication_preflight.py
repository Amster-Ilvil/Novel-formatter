# -*- coding: utf-8 -*-
"""Publication preflight and conservative deterministic repairs.

The preflight never calls AI.  It identifies structural/text defects that are
especially damaging to EPUB publication or translation.  The repair function
only performs operations whose source characters are demonstrably duplicated
or whose block typing/splitting is unambiguous; uncertain wording is left for
manual review.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
import re
from difflib import SequenceMatcher
from typing import Iterable

from engine.text_compare import (
    looks_like_chapter_title,
    normalise_for_alignment,
    split_dialogue_segments,
)
from models.document import Block, BlockType, TocEntry, UnifiedDocument


@dataclass(frozen=True)
class DuplicateRun:
    first_start: int
    second_start: int
    length: int


@dataclass(frozen=True)
class PublicationTextIssue:
    block_index: int
    block_id: str
    code: str
    severity: str
    message: str
    excerpt: str
    auto_fixable: bool = False
    related_block_index: int = -1

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PublicationRepairReport:
    changed_blocks: int = 0
    removed_blocks: int = 0
    overlap_repairs: int = 0
    dialogue_splits: int = 0
    type_repairs: int = 0
    duplicate_removals: int = 0
    quote_repairs: int = 0
    quote_joins: int = 0
    ocr_confusable_repairs: int = 0
    details: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PublicationPreflight:
    chapter_count: int = 0
    chapter_like_count: int = 0
    image_count: int = 0
    unplaced_image_count: int = 0
    duplicate_runs: tuple[DuplicateRun, ...] = field(default_factory=tuple)
    quote_imbalance_count: int = 0
    embedded_dialogue_count: int = 0
    text_issues: tuple[PublicationTextIssue, ...] = field(default_factory=tuple)

    @property
    def high_text_issue_count(self) -> int:
        return sum(1 for issue in self.text_issues if issue.severity == "high")

    @property
    def medium_text_issue_count(self) -> int:
        return sum(1 for issue in self.text_issues if issue.severity == "medium")

    @property
    def auto_fixable_count(self) -> int:
        return sum(1 for issue in self.text_issues if issue.auto_fixable)

    @property
    def translation_blocker_issues(self) -> tuple[PublicationTextIssue, ...]:
        """Issues likely to change sentence meaning or speaker attribution.

        These are stricter than ordinary publication warnings.  Typography,
        indentation and harmless punctuation differences are intentionally not
        blockers; broken sentence structure, quote ownership, duplicated text
        and large departures from OCR are.
        """
        return tuple(
            issue for issue in self.text_issues
            if issue.code in _TRANSLATION_BLOCKER_CODES
        )

    @property
    def translation_blocker_count(self) -> int:
        return len(self.translation_blocker_issues)

    @property
    def translation_ready(self) -> bool:
        return not (
            self.translation_blocker_count
            or self.unplaced_image_count
            or self.duplicate_runs
            or (self.chapter_like_count and not self.chapter_count)
        )

    @property
    def critical_messages(self) -> list[str]:
        messages: list[str] = []
        if self.chapter_like_count and not self.chapter_count:
            messages.append(f"检测到 {self.chapter_like_count} 个章节标题，但目录章节为 0")
        if self.unplaced_image_count:
            messages.append(f"有 {self.unplaced_image_count} 张插图尚未在文本对比中定位")
        if self.duplicate_runs:
            longest = max(run.length for run in self.duplicate_runs)
            messages.append(f"检测到 {len(self.duplicate_runs)} 组连续重复正文（最长 {longest} 段）")
        if self.quote_imbalance_count:
            messages.append(f"有 {self.quote_imbalance_count} 个段落的会话/引用引号不平衡")
        if self.high_text_issue_count:
            messages.append(f"正文中有 {self.high_text_issue_count} 处高风险残片、错拼或对白结构问题")
        if self.translation_blocker_count:
            messages.append(
                f"其中 {self.translation_blocker_count} 处可能改变句意或说话人，"
                "不建议直接交给翻译"
            )
        return messages

    @property
    def warning_messages(self) -> list[str]:
        messages: list[str] = []
        if self.embedded_dialogue_count:
            messages.append(f"有 {self.embedded_dialogue_count} 个普通段落混入完整对白")
        if self.medium_text_issue_count:
            messages.append(f"正文中有 {self.medium_text_issue_count} 处需要人工留意的问题")
        if self.auto_fixable_count:
            messages.append(f"其中 {self.auto_fixable_count} 处可用高置信度规则修复")
        return messages


_TEXT_SKIP = {
    BlockType.IMAGE_REF, BlockType.CHAPTER, BlockType.SECTION,
    BlockType.HEADER_FOOTER, BlockType.TOC_ENTRY,
}
_TERMINAL = "。！？!?」』）)]】》〉〕〗〙〛…‥"
_JA_CHAR = r"ぁ-んァ-ヶー一-龯々〆ヵヶ"
_KATAKANA_ONE_RE = re.compile(r"(?<=[ァ-ヶー])1(?=[ァ-ヶーぁ-ん一-龯・。、！？!?」』）】°\s]|$)")
_DEGREE_AS_PERIOD_RE = re.compile(r"(?<![0-9])°(?=[一-龯ぁ-んァ-ヶ「『])")
_KATAKANA_ONE_DEGREE_RE = re.compile(r"(?<=[ァ-ヶー])1°(?=[一-龯ぁ-んァ-ヶ「『])")
_MIXED_LATIN_OCR_RE = re.compile(r"(?<=[一-龯ぁ-んァ-ヶ])[Il|](?=[一-龯ぁ-んァ-ヶ])")
_UNFINISHED_CONNECTIVE_RE = re.compile(
    r"(?:けど|けれど|けれども|ので|のに|から|ながら|つつ|として|し|て|で)$"
)
_DUPLICATED_KANA_RE = re.compile(
    r"([ぁ-んァ-ヶ])\1(?:じゃ|だ|です|ない|の|。|、|」|』|$)"
)

# Codes in this set are not merely cosmetic.  Leaving one unresolved can
# change what a translator understands, who is speaking, or whether text was
# omitted/duplicated.
_TRANSLATION_BLOCKER_CODES = {
    "dialogue_quote_imbalance",
    "citation_quote_imbalance",
    "missing_dialogue_open",
    "missing_citation_open",
    "orphan_prefix",
    "stray_leading_closer",
    "nested_quote_damage",
    "large_departure_from_ocr",
    "adjacent_duplicate",
    "boundary_overlap",
    "short_suffix_fragment",
    "dialogue_type_mismatch",
    "unfinished_connective",
}


def _find_duplicate_runs(texts: list[str], *, window: int = 5, max_gap: int = 300) -> tuple[DuplicateRun, ...]:
    if len(texts) < window * 2:
        return ()
    keys = [normalise_for_alignment(text) for text in texts]
    seen: dict[tuple[str, ...], list[int]] = {}
    found: list[DuplicateRun] = []
    occupied: list[tuple[int, int]] = []

    for second in range(0, len(keys) - window + 1):
        token = tuple(keys[second:second + window])
        if not all(token):
            continue
        for first in reversed(seen.get(token, [])):
            if second - first > max_gap:
                break
            length = window
            while second + length < len(keys) and first + length < second and keys[first + length] == keys[second + length]:
                length += 1
            if length < window:
                continue
            if any(first >= a and second <= b for a, b in occupied):
                continue
            found.append(DuplicateRun(first, second, length))
            occupied.append((first, second + length))
            break
        seen.setdefault(token, []).append(second)
    return tuple(found)


def _boundary_overlap(previous: str, current: str, *, max_len: int = 24) -> str:
    """Return a highly reliable duplicated prefix/suffix at a block boundary."""
    left = (previous or "").rstrip()
    right = (current or "").lstrip()
    if not left or not right:
        return ""
    limit = min(max_len, len(left), len(right))
    for length in range(limit, 1, -1):
        candidate = right[:length]
        if left.endswith(candidate):
            # Two-character overlaps are only safe when a quote/punctuation is
            # part of the duplicate.  Longer overlaps are strong evidence.
            if length >= 4 or (length >= 2 and candidate[-1] in "」』）】"):
                return candidate
    return ""


def _excerpt(text: str, limit: int = 110) -> str:
    compact = (text or "").replace("\n", "↵").strip()
    return compact if len(compact) <= limit else compact[:limit] + "…"


def _compact_without_quotes(text: str) -> str:
    return re.sub(r"[\s　「」『』]", "", text or "")


def _raw_supports_missing_open(block: Block, opening: str, closing: str) -> bool:
    raw = str(block.ocr_raw or "").strip()
    current = str(block.text or "").strip()
    if not raw or not current or opening in current or current.count(closing) != 1:
        return False
    if raw.count(opening) != 1 or raw.count(closing) != 1:
        return False
    if not raw.startswith(opening) or not raw.endswith(closing) or not current.endswith(closing):
        return False
    return _compact_without_quotes(raw) == _compact_without_quotes(current)


def _repair_duplicate_citation_open(text: str) -> str | None:
    """Remove a demonstrably duplicated nested 『 opener.

    Japanese nested quotation should alternate 「...『...』...」.  Two 『
    openers with only one 』 closer in one block cannot be balanced as written.
    Removing the second opener preserves every lexical character and restores
    a single citation pair.
    """
    if text.count("『") != 2 or text.count("』") != 1:
        return None
    first = text.find("『")
    second = text.find("『", first + 1)
    close = text.find("』", second + 1)
    if first < 0 or second < 0 or close < 0 or not (first < second < close):
        return None
    repaired = text[:second] + text[second + 1:]
    return repaired if repaired.count("『") == repaired.count("』") == 1 else None


def _ocr_departure(block: Block) -> float | None:
    raw = re.sub(r"[\s　]+", "", str(block.ocr_raw or ""))
    current = re.sub(r"[\s　]+", "", str(block.text or ""))
    if min(len(raw), len(current)) < 30:
        return None
    # After safe paragraph/dialogue splitting, each child keeps the original
    # merged OCR text for audit.  A child fully contained in that source is not
    # a semantic rewrite and must not be treated as a large departure.
    if current in raw or raw in current:
        return 1.0
    ratio = len(current) / max(1, len(raw))
    if not 0.78 <= ratio <= 1.22:
        # Length alone is inconclusive because blocks can be deterministically
        # split or joined.  AI before/after review is the authoritative guard
        # for those operations.
        return None
    return SequenceMatcher(None, raw, current, autojunk=False).ratio()


def _issue(
    index: int, block: Block, code: str, severity: str, message: str,
    *, auto_fixable: bool = False, related: int = -1,
) -> PublicationTextIssue:
    return PublicationTextIssue(
        block_index=index,
        block_id=block.id,
        code=code,
        severity=severity,
        message=message,
        excerpt=_excerpt(block.text),
        auto_fixable=auto_fixable,
        related_block_index=related,
    )


def _inspect_text_issues(doc: UnifiedDocument) -> tuple[PublicationTextIssue, ...]:
    issues: list[PublicationTextIssue] = []
    previous_text: tuple[int, Block] | None = None
    for index, block in enumerate(doc.blocks):
        if block.type in _TEXT_SKIP or (block.metadata or {}).get("consumed"):
            continue
        text = str(block.text or "")
        stripped = text.strip()
        if not stripped:
            continue

        if stripped.count("「") != stripped.count("」"):
            issues.append(_issue(index, block, "dialogue_quote_imbalance", "high", "会话引号不平衡"))
        if stripped.count("『") != stripped.count("』"):
            issues.append(_issue(index, block, "citation_quote_imbalance", "high", "引用引号不平衡"))
        if "「" not in stripped and "」" in stripped:
            supported = _raw_supports_missing_open(block, "「", "」")
            issues.append(_issue(index, block, "missing_dialogue_open", "high", "存在闭会话引号但没有开引号", auto_fixable=supported))
        if "『" not in stripped and "』" in stripped:
            supported = _raw_supports_missing_open(block, "『", "』")
            issues.append(_issue(index, block, "missing_citation_open", "high", "存在闭引用引号但没有开引号", auto_fixable=supported))
        if re.search(rf"^[{_JA_CHAR}][。』」](?=.{{8,}})", stripped):
            issues.append(_issue(index, block, "orphan_prefix", "high", "段落以单字残片开头，可能改变句意"))
        if stripped.startswith(("」", "』", "）", "】", "。", "、")):
            issues.append(_issue(index, block, "stray_leading_closer", "high", "段落以闭引号或标点开头，疑似断句/归属错误"))
        if _KATAKANA_ONE_RE.search(stripped):
            issues.append(_issue(index, block, "katakana_long_mark_confusable", "medium", "片假名词中的数字 1 疑似长音符号 ー", auto_fixable=True))
        if _DEGREE_AS_PERIOD_RE.search(stripped) or _KATAKANA_ONE_DEGREE_RE.search(stripped):
            issues.append(_issue(index, block, "degree_as_japanese_period", "medium", "正文中的 ° 疑似误识别的句号", auto_fixable=True))
        if _MIXED_LATIN_OCR_RE.search(stripped):
            issues.append(_issue(index, block, "mixed_latin_ocr_confusable", "medium", "日文词中混入 I/l/|，疑似 OCR 字符"))
        if re.search(r"『[^』\n]{1,100}『[^』\n]{1,50}』", stripped):
            issues.append(_issue(
                index, block, "nested_quote_damage", "high", "引用引号疑似错误嵌套",
                auto_fixable=_repair_duplicate_citation_open(stripped) is not None,
            ))
        connective_core = stripped.rstrip("」』）】").rstrip()
        if (
            len(connective_core) >= 8
            and not connective_core.endswith(tuple(_TERMINAL))
            and _UNFINISHED_CONNECTIVE_RE.search(connective_core)
        ):
            dialogue_shaped = (
                block.type == BlockType.DIALOGUE
                or (stripped.startswith("「") and stripped.endswith("」"))
                or (stripped.startswith("『") and stripped.endswith("』"))
            )
            if dialogue_shaped:
                issues.append(_issue(index, block, "dialogue_trailing_connective", "medium", "对白以接续表达收尾；可能是自然口语，也可能被截断，建议抽查"))
            else:
                issues.append(_issue(index, block, "unfinished_connective", "high", "叙述句停在强接续词处，疑似漏文或错分行"))
        if _DUPLICATED_KANA_RE.search(stripped):
            issues.append(_issue(index, block, "duplicated_kana", "medium", "疑似重复假名；建议核对原图或 OCR 原始块"))
        departure = _ocr_departure(block)
        if departure is not None and departure < 0.68:
            issues.append(_issue(index, block, "large_departure_from_ocr", "high", f"处理后文本与 OCR 原始块差异过大（相似度 {departure:.0%}）"))
        if block.type != BlockType.DIALOGUE and re.search(r"「[^「」\n]+」", stripped):
            pieces = split_dialogue_segments(stripped)
            if len(pieces) > 1:
                issues.append(_issue(index, block, "embedded_dialogue", "medium", "普通正文中混入可独立拆分的完整对白", auto_fixable=True))
        if stripped.startswith("「") and stripped.endswith("」") and block.type != BlockType.DIALOGUE:
            issues.append(_issue(index, block, "dialogue_type", "low", "完整对白块类型不是 dialogue", auto_fixable=True))
        if block.type == BlockType.DIALOGUE and not (stripped.startswith("「") and stripped.endswith("」")):
            issues.append(_issue(index, block, "dialogue_type_mismatch", "medium", "dialogue 类型块没有完整会话引号"))

        if previous_text is not None:
            previous_index, previous_block = previous_text
            previous_stripped = str(previous_block.text or "").strip()
            if normalise_for_alignment(previous_stripped) and normalise_for_alignment(previous_stripped) == normalise_for_alignment(stripped):
                issues.append(_issue(index, block, "adjacent_duplicate", "high", "与上一正文块完全重复", auto_fixable=True, related=previous_index))
            overlap = _boundary_overlap(previous_stripped, stripped)
            if overlap:
                issues.append(_issue(index, block, "boundary_overlap", "medium", f"与上一块边界重复“{overlap}”", auto_fixable=True, related=previous_index))
            if previous_stripped and not previous_stripped.endswith(_TERMINAL) and len(stripped) <= 6 and stripped.endswith(_TERMINAL):
                issues.append(_issue(index, block, "short_suffix_fragment", "medium", "短尾部残片可能应接回上一句", auto_fixable=True, related=previous_index))
        previous_text = (index, block)
    return tuple(issues)


def _previous_text_block_without_barrier(rebuilt: list[Block]) -> Block | None:
    """Return the nearest textual block, but never cross a structural anchor."""
    for candidate in reversed(rebuilt):
        if candidate.type in {BlockType.IMAGE_REF, BlockType.CHAPTER, BlockType.SECTION, BlockType.TOC_ENTRY}:
            return None
        if (candidate.metadata or {}).get("consumed"):
            continue
        if str(candidate.text or "").strip():
            return candidate
    return None


def inspect_document_for_publication(doc: UnifiedDocument) -> PublicationPreflight:
    chapter_count = sum(
        1 for block in doc.blocks
        if block.type == BlockType.CHAPTER and str(block.text or "").strip()
    )
    chapter_like_count = sum(
        1 for block in doc.blocks
        if block.type != BlockType.CHAPTER and looks_like_chapter_title(str(block.text or ""))
    ) + chapter_count
    image_blocks = [block for block in doc.blocks if block.type == BlockType.IMAGE_REF]
    unplaced = sum(1 for block in image_blocks if (block.metadata or {}).get("placement_required"))

    texts: list[str] = []
    quote_imbalance = 0
    embedded_dialogue = 0
    for block in doc.blocks:
        if block.type in _TEXT_SKIP:
            continue
        text = str(block.text or "").strip()
        if not text or (block.metadata or {}).get("consumed"):
            continue
        texts.append(text)
        if text.count("「") != text.count("」") or text.count("『") != text.count("』"):
            quote_imbalance += 1
        if block.type != BlockType.DIALOGUE and "「" in text and "」" in text:
            embedded_dialogue += 1

    return PublicationPreflight(
        chapter_count=chapter_count,
        chapter_like_count=chapter_like_count,
        image_count=len(image_blocks),
        unplaced_image_count=unplaced,
        duplicate_runs=_find_duplicate_runs(texts),
        quote_imbalance_count=quote_imbalance,
        embedded_dialogue_count=embedded_dialogue,
        text_issues=_inspect_text_issues(doc),
    )


def _copy_block_with_text(block: Block, text: str, action: str) -> Block:
    result = copy.deepcopy(block)
    before = result.text
    result.ocr_raw = result.ocr_raw or before
    result.text = text
    result.modified_by = (result.modified_by + ",publication_high_confidence_repair").strip(",")
    result.metadata = dict(result.metadata or {})
    result.metadata.setdefault("publication_repair_audit", []).append({
        "action": action,
        "before": before,
        "after": text,
    })
    return result


def repair_high_confidence_publication_issues(doc: UnifiedDocument) -> tuple[UnifiedDocument, PublicationRepairReport]:
    """Apply only deterministic, character-preserving publication repairs."""
    result = copy.deepcopy(doc)
    rebuilt: list[Block] = []
    details: list[str] = []
    changed = removed = overlaps = dialogue_splits = type_repairs = duplicates = quote_repairs = quote_joins = ocr_confusables = 0

    for source_index, source_block in enumerate(result.blocks):
        block = copy.deepcopy(source_block)
        if block.type in _TEXT_SKIP or (block.metadata or {}).get("consumed") or not str(block.text or "").strip():
            rebuilt.append(block)
            continue

        previous = _previous_text_block_without_barrier(rebuilt)
        text = str(block.text or "")
        stripped = text.strip()

        confusable_fixed = _KATAKANA_ONE_RE.sub("ー", stripped)
        confusable_fixed = _DEGREE_AS_PERIOD_RE.sub("。", confusable_fixed)
        if confusable_fixed != stripped:
            block = _copy_block_with_text(block, confusable_fixed, "repair_ocr_confusable")
            text = block.text
            stripped = text.strip()
            ocr_confusables += 1
            changed += 1
            details.append(f"第 {source_index + 1} 块修复片假名长音/句号 OCR 混淆")

        if previous is not None and normalise_for_alignment(previous.text) == normalise_for_alignment(stripped):
            removed += 1
            duplicates += 1
            details.append(f"删除相邻重复块 {source_index + 1}")
            continue

        if previous is not None:
            joined_quote = False
            previous_stripped = str(previous.text or "").strip()
            for opening, closing in (("「", "」"), ("『", "』")):
                previous_balance = previous_stripped.count(opening) - previous_stripped.count(closing)
                current_balance = stripped.count(opening) - stripped.count(closing)
                if (
                    previous_balance == 1
                    and current_balance == -1
                    and not stripped.startswith(opening)
                    and (previous_stripped + stripped).count(opening)
                        == (previous_stripped + stripped).count(closing)
                ):
                    before_previous = previous.text
                    previous.text = previous_stripped + stripped
                    if opening == "「" and previous.text.startswith("「") and previous.text.endswith("」"):
                        previous.type = BlockType.DIALOGUE
                    previous.modified_by = (previous.modified_by + ",publication_high_confidence_repair").strip(",")
                    previous.metadata = dict(previous.metadata or {})
                    previous.metadata.setdefault("publication_repair_audit", []).append({
                        "action": "join_split_quote",
                        "before": before_previous,
                        "from_block": block.id,
                        "fragment": stripped,
                    })
                    removed += 1
                    changed += 1
                    quote_joins += 1
                    details.append(f"第 {source_index + 1} 块接回上一块未闭合{opening}{closing}内容")
                    joined_quote = True
                    break
            if joined_quote:
                continue

            overlap = _boundary_overlap(previous.text, stripped)
            if overlap and stripped.startswith(overlap):
                new_text = stripped[len(overlap):].lstrip()
                if new_text:
                    block = _copy_block_with_text(block, new_text, "trim_boundary_overlap")
                    text = new_text
                    stripped = new_text.strip()
                    changed += 1
                    overlaps += 1
                    details.append(f"第 {source_index + 1} 块删除重复边界“{overlap}”")
            # A very short suffix fragment after an unfinished previous block is
            # joined without inventing or deleting source characters.
            if (
                block.type == BlockType.PARAGRAPH
                and previous.type == BlockType.PARAGRAPH
                and not stripped.startswith(("「", "『", "」", "』"))
                and len(stripped) <= 6
                and stripped.endswith(_TERMINAL)
                and previous.text
                and not str(previous.text).rstrip().endswith(_TERMINAL)
            ):
                previous.text = str(previous.text).rstrip() + stripped
                previous.modified_by = (previous.modified_by + ",publication_high_confidence_repair").strip(",")
                previous.metadata = dict(previous.metadata or {})
                previous.metadata.setdefault("publication_repair_audit", []).append({
                    "action": "join_short_suffix_fragment", "from_block": block.id, "fragment": stripped,
                })
                removed += 1
                changed += 1
                details.append(f"第 {source_index + 1} 块短尾片接回上一句")
                continue

        if _raw_supports_missing_open(block, "「", "」"):
            block = _copy_block_with_text(block, "「" + stripped, "restore_dialogue_open_from_ocr_raw")
            block.type = BlockType.DIALOGUE
            text = block.text
            stripped = text.strip()
            quote_repairs += 1
            changed += 1
            details.append(f"第 {source_index + 1} 块依据 OCR 原文恢复会话开引号")
        elif _raw_supports_missing_open(block, "『", "』"):
            block = _copy_block_with_text(block, "『" + stripped, "restore_citation_open_from_ocr_raw")
            text = block.text
            stripped = text.strip()
            quote_repairs += 1
            changed += 1
            details.append(f"第 {source_index + 1} 块依据 OCR 原文恢复引用开引号")

        duplicate_citation_fixed = _repair_duplicate_citation_open(stripped)
        if duplicate_citation_fixed is not None:
            block = _copy_block_with_text(block, duplicate_citation_fixed, "remove_duplicate_nested_citation_open")
            text = block.text
            stripped = text.strip()
            quote_repairs += 1
            changed += 1
            details.append(f"第 {source_index + 1} 块删除重复的嵌套『开引号")

        pieces = split_dialogue_segments(text)
        if block.type != BlockType.DIALOGUE and len(pieces) > 1:
            for piece_index, piece in enumerate(pieces):
                if not piece:
                    continue
                child = copy.deepcopy(block)
                child.text = piece
                child.type = BlockType.DIALOGUE if piece.strip().startswith("「") and piece.strip().endswith("」") else BlockType.PARAGRAPH
                if piece_index:
                    child.id = Block(type=child.type).id
                    child.metadata = dict(child.metadata or {})
                    child.metadata["split_from_block"] = block.id
                child.modified_by = (child.modified_by + ",publication_high_confidence_repair").strip(",")
                rebuilt.append(child)
            dialogue_splits += len(pieces) - 1
            changed += 1
            details.append(f"第 {source_index + 1} 块拆分正文/对白")
            continue

        if stripped.startswith("「") and stripped.endswith("」") and block.type != BlockType.DIALOGUE:
            block.type = BlockType.DIALOGUE
            block.modified_by = (block.modified_by + ",publication_high_confidence_repair").strip(",")
            type_repairs += 1
            changed += 1
        rebuilt.append(block)

    result.blocks = rebuilt
    result.toc = []
    chapter_no = 0
    for index, block in enumerate(result.blocks):
        block.reading_order = index
        if block.type == BlockType.CHAPTER and str(block.text or "").strip():
            chapter_no += 1
            block.chapter_index = chapter_no
            result.toc.append(TocEntry(block.text.strip(), chapter_no, index))
    result.metadata.__dict__["publication_high_confidence_repair"] = {
        "changed_blocks": changed,
        "removed_blocks": removed,
        "overlap_repairs": overlaps,
        "dialogue_splits": dialogue_splits,
        "type_repairs": type_repairs,
        "duplicate_removals": duplicates,
        "quote_repairs": quote_repairs,
        "quote_joins": quote_joins,
        "ocr_confusable_repairs": ocr_confusables,
    }
    result.add_log(
        "publication_high_confidence_repair",
        f"发布前高置信度修复：修改 {changed}，删除 {removed}，边界重叠 {overlaps}，对白拆分 {dialogue_splits}",
        changed + removed,
    )
    return result, PublicationRepairReport(
        changed_blocks=changed,
        removed_blocks=removed,
        overlap_repairs=overlaps,
        dialogue_splits=dialogue_splits,
        type_repairs=type_repairs,
        duplicate_removals=duplicates,
        quote_repairs=quote_repairs,
        quote_joins=quote_joins,
        ocr_confusable_repairs=ocr_confusables,
        details=tuple(details),
    )
