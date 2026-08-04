#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lossless Unicode repair and compare-only normalization for Japanese OCR.

Two representations are intentionally kept separate:

``normalize_japanese_ocr_text``
    Produces the text that may be shown, selected and exported.  It only applies
    transformations that keep the same written character: canonical composition
    (for example ``か + combining dakuten`` -> ``が``), spacing dakuten attached
    to kana, half-width Japanese kana/punctuation, and Unicode vertical
    presentation forms.  It never removes a code point, never rewrites a CJK
    ideograph/radical, never strips IVS, and never changes dash or punctuation
    style aliases.

``japanese_ocr_comparison_key``
    Produces an ephemeral key used only to decide whether OCR candidates are
    visually/Unicode-equivalent.  The key may ignore invisible format controls,
    IVS and compatibility-codepoint distinctions, but it is never written back
    to OCR text or exported as authoritative text.

This split prevents a correct character from being silently replaced while
still eliminating false conflicts in the OCR comparison screen.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
import re
import unicodedata
from typing import Iterable

# Runs that Unicode defines as half-width Japanese kana or Japanese punctuation.
_HALFWIDTH_KANA_RUN = re.compile(r"[\uFF61-\uFF9F]+")
# Vertical/small punctuation presentation forms with a direct Unicode
# compatibility mapping back to the same punctuation character.
_CJK_PRESENTATION = re.compile(r"[\uFE10-\uFE1F\uFE30-\uFE4F]")
# Compatibility CJK blocks used only in the compare key.
_CJK_COMPATIBILITY = re.compile(r"[\uF900-\uFAFF\U0002F800-\U0002FA1F]")
_CJK_RADICAL = re.compile(r"[\u2E80-\u2EF3\u2F00-\u2FD5]")
# Characters ignored only in the compare key.  The real OCR text preserves them.
_INVISIBLE_FORMAT = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F"
    r"\u00AD\u034F\u061C\u180E\u200B-\u200F\u202A-\u202E"
    r"\u2060-\u206F\uFEFF]"
)
_SPACE_VARIANTS = re.compile(r"[\u00A0\u1680\u2000-\u200A\u202F\u205F]")
_VARIATION_SELECTOR = re.compile(r"[\uFE00-\uFE0F\U000E0100-\U000E01EF]")
_KANA_SPACING_DAKUTEN = re.compile(
    r"([\u3041-\u3096\u30A1-\u30FA])([\u309B\u309C])"
)

# Compare-only aliases.  These are never written back to the OCR text.
_COMPARE_PUNCTUATION_ALIASES = str.maketrans({
    "\uFF5E": "\u301C",  # FULLWIDTH TILDE -> WAVE DASH
    "\u00B7": "\u30FB",  # MIDDLE DOT -> KATAKANA MIDDLE DOT
    # Compare-only ASCII/fullwidth forms.  Authoritative OCR text remains
    # untouched; these mappings only prevent visually identical punctuation
    # from appearing as a false conflict in OCR Compare.
    "？": "?", "！": "!", "：": ":", "；": ";",
    "（": "(", "）": ")", "［": "[", "］": "]",
    "｛": "{", "｝": "}", "，": ",", "．": ".",
})

_CATEGORY_LABELS = {
    "nfc": "组合浊音/合成字符",
    "spacing_dakuten": "间隔浊点",
    "halfwidth_kana": "半角片假名/日文标点",
    "presentation_form": "竖排兼容标点",
    "compare_newline": "仅比较：换行码位",
    "compare_radical": "仅比较：康熙/CJK 部首",
    "compare_compat_ideograph": "仅比较：兼容汉字",
    "compare_variation_selector": "仅比较：异体字选择符",
    "compare_invisible": "仅比较：不可见/控制字符",
    "compare_space": "仅比较：空格码位",
    "compare_punctuation": "仅比较：标点码位",
    "compare_whitespace": "仅比较：OCR版面空白",
}


@dataclass(slots=True)
class OCRUnicodeNormalizationReport:
    """Aggregated applied and compare-only normalization counts."""

    counts: dict[str, int] = field(default_factory=dict)
    compare_only_counts: dict[str, int] = field(default_factory=dict)
    texts_changed: int = 0
    comparison_keys_changed: int = 0
    characters_before: int = 0
    characters_after: int = 0

    @property
    def total_changes(self) -> int:
        return sum(max(0, int(value or 0)) for value in self.counts.values())

    @property
    def total_compare_only_changes(self) -> int:
        return sum(
            max(0, int(value or 0)) for value in self.compare_only_counts.values()
        )

    @property
    def changed(self) -> bool:
        return bool(
            self.texts_changed
            or self.comparison_keys_changed
            or self.total_changes
            or self.total_compare_only_changes
        )

    @property
    def deleted_characters(self) -> int:
        """Always zero: applied normalization contains no deletion rule."""
        return 0

    def add_count(self, name: str, value: int = 1, *, compare_only: bool = False) -> None:
        amount = max(0, int(value or 0))
        if not amount:
            return
        target = self.compare_only_counts if compare_only else self.counts
        target[name] = target.get(name, 0) + amount

    def merge(self, other: "OCRUnicodeNormalizationReport") -> None:
        if not isinstance(other, OCRUnicodeNormalizationReport):
            return
        for name, value in other.counts.items():
            self.add_count(name, value)
        for name, value in other.compare_only_counts.items():
            self.add_count(name, value, compare_only=True)
        self.texts_changed += int(other.texts_changed or 0)
        self.comparison_keys_changed += int(other.comparison_keys_changed or 0)
        self.characters_before += int(other.characters_before or 0)
        self.characters_after += int(other.characters_after or 0)

    def summary_lines(self) -> list[str]:
        lines: list[str] = []
        for name, value in sorted(
            self.counts.items(), key=lambda item: (-int(item[1]), item[0])
        ):
            lines.append(f"{_CATEGORY_LABELS.get(name, name)}：{int(value)}")
        for name, value in sorted(
            self.compare_only_counts.items(),
            key=lambda item: (-int(item[1]), item[0]),
        ):
            lines.append(f"{_CATEGORY_LABELS.get(name, name)}：{int(value)}")
        return lines


def _difference_units(before: str, after: str) -> int:
    if before == after:
        return 0
    overlap = min(len(before), len(after))
    positional = sum(1 for index in range(overlap) if before[index] != after[index])
    return max(1, positional + abs(len(before) - len(after)))


def _normalize_regex_nfkc(text: str, pattern: re.Pattern[str]) -> tuple[str, int]:
    changed = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal changed
        source = match.group(0)
        target = unicodedata.normalize("NFKC", source)
        if target != source:
            changed += 1
        return target

    return pattern.sub(replace, text), changed


def _compose_spacing_dakuten(text: str) -> tuple[str, int]:
    changed = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal changed
        base, mark = match.groups()
        combining = "\u3099" if mark == "\u309B" else "\u309A"
        target = unicodedata.normalize("NFC", base + combining)
        if target != match.group(0):
            changed += 1
        return target

    return _KANA_SPACING_DAKUTEN.sub(replace, text), changed




def _compose_combining_sequences(text: str) -> tuple[str, int]:
    """Compose only base+combining clusters, preserving standalone CJK codepoints.

    Python's whole-string NFC also canonically folds a small number of CJK
    compatibility ideographs.  That is undesirable for authoritative OCR text,
    so composition is limited to clusters that actually contain combining marks.
    """
    if not text:
        return text, 0
    output: list[str] = []
    changed = 0
    index = 0
    while index < len(text):
        start = index
        index += 1
        while index < len(text) and unicodedata.combining(text[index]):
            index += 1
        cluster = text[start:index]
        if len(cluster) > 1:
            composed = unicodedata.normalize("NFC", cluster)
            if composed != cluster:
                changed += _difference_units(cluster, composed)
            output.append(composed)
        else:
            output.append(cluster)
    return "".join(output), changed

def normalize_japanese_ocr_text(text: str) -> tuple[str, OCRUnicodeNormalizationReport]:
    """Return an export-safe, non-deleting normalization of OCR text.

    No CJK base character is rewritten.  IVS, invisible characters, whitespace,
    line endings, punctuation aliases and dash variants remain in the returned
    text exactly as OCR produced them.  Only same-character Unicode composition
    and Japanese width/presentation forms are repaired.
    """
    source = str(text or "")
    report = OCRUnicodeNormalizationReport(
        characters_before=len(source), characters_after=len(source)
    )
    value = source

    value, count = _compose_spacing_dakuten(value)
    report.add_count("spacing_dakuten", count)

    before = value
    value = _HALFWIDTH_KANA_RUN.sub(
        lambda match: unicodedata.normalize("NFKC", match.group(0)), value
    )
    if value != before:
        report.add_count("halfwidth_kana", len(_HALFWIDTH_KANA_RUN.findall(before)))

    value, count = _normalize_regex_nfkc(value, _CJK_PRESENTATION)
    report.add_count("presentation_form", count)

    value, count = _compose_combining_sequences(value)
    report.add_count("nfc", count)

    report.characters_after = len(value)
    report.texts_changed = int(value != source)
    return value, report


def japanese_ocr_comparison_key(
    text: str,
) -> tuple[str, OCRUnicodeNormalizationReport]:
    """Build an aggressive but ephemeral equivalence key.

    This key is for comparison and voting only.  It must never be assigned to a
    document block, source editor, candidate text or EPUB export.
    """
    safe_text, report = normalize_japanese_ocr_text(text)
    source = safe_text
    value = safe_text

    newline_count = (
        value.count("\r\n")
        + value.count("\r")
        + value.count("\u2028")
        + value.count("\u2029")
    )
    if newline_count:
        value = value.replace("\r\n", "\n").replace("\r", "\n")
        value = value.replace("\u2028", "\n").replace("\u2029", "\n")
        report.add_count("compare_newline", newline_count, compare_only=True)

    value, count = _normalize_regex_nfkc(value, _CJK_RADICAL)
    report.add_count("compare_radical", count, compare_only=True)
    value, count = _normalize_regex_nfkc(value, _CJK_COMPATIBILITY)
    report.add_count("compare_compat_ideograph", count, compare_only=True)

    selector_count = len(_VARIATION_SELECTOR.findall(value))
    if selector_count:
        value = _VARIATION_SELECTOR.sub("", value)
        report.add_count(
            "compare_variation_selector", selector_count, compare_only=True
        )

    invisible_count = len(_INVISIBLE_FORMAT.findall(value))
    if invisible_count:
        value = _INVISIBLE_FORMAT.sub("", value)
        report.add_count("compare_invisible", invisible_count, compare_only=True)

    space_count = len(_SPACE_VARIANTS.findall(value))
    if space_count:
        value = _SPACE_VARIANTS.sub(" ", value)
        report.add_count("compare_space", space_count, compare_only=True)

    # Physical-column OCR engines disagree frequently about layout spaces and
    # line wraps.  They are ignored only in the ephemeral comparison key.
    before_whitespace = value
    value = re.sub(r"\s+", "", value)
    if value != before_whitespace:
        report.add_count(
            "compare_whitespace",
            _difference_units(before_whitespace, value),
            compare_only=True,
        )

    before = value
    value = value.translate(_COMPARE_PUNCTUATION_ALIASES)
    if value != before:
        report.add_count(
            "compare_punctuation", _difference_units(before, value), compare_only=True
        )

    value = unicodedata.normalize("NFC", value)
    report.comparison_keys_changed = int(value != source)
    return value, report


def normalize_text_collection(
    texts: Iterable[str],
) -> tuple[list[str], OCRUnicodeNormalizationReport]:
    output: list[str] = []
    total = OCRUnicodeNormalizationReport()
    for text in texts:
        normalized, report = normalize_japanese_ocr_text(str(text or ""))
        output.append(normalized)
        total.merge(report)
    return output, total


def normalize_text_collection_with_keys(
    texts: Iterable[str],
) -> tuple[list[str], list[str], OCRUnicodeNormalizationReport]:
    """Return safe text copies plus compare-only keys and one report."""
    safe_output: list[str] = []
    keys: list[str] = []
    total = OCRUnicodeNormalizationReport()
    for text in texts:
        source = str(text or "")
        safe, safe_report = normalize_japanese_ocr_text(source)
        key, key_report = japanese_ocr_comparison_key(safe)
        # Avoid counting the safe repairs twice because key generation starts by
        # calling the safe normalizer.
        key_report.counts.clear()
        key_report.texts_changed = 0
        key_report.characters_before = 0
        key_report.characters_after = 0
        safe_output.append(safe)
        keys.append(key)
        total.merge(safe_report)
        total.merge(key_report)
    return safe_output, keys, total


def comparison_keys_for_texts(texts: Iterable[str]) -> list[str]:
    return [japanese_ocr_comparison_key(str(text or ""))[0] for text in texts]


def normalize_document_copy(document):
    """Return a deep-copied document with only non-deleting repairs applied."""
    result = copy.deepcopy(document)
    total = OCRUnicodeNormalizationReport()
    for block in getattr(result, "blocks", []) or []:
        if not hasattr(block, "text"):
            continue
        normalized, report = normalize_japanese_ocr_text(
            str(getattr(block, "text", "") or "")
        )
        block.text = normalized
        total.merge(report)
    if total.changed and hasattr(result, "add_log"):
        details = "、".join(total.summary_lines()[:6])
        result.add_log(
            "ocr_unicode_standardize_lossless",
            f"OCR Unicode 无损标准化：{total.texts_changed} 个文本块；{details}",
            total.total_changes,
        )
    return result, total
