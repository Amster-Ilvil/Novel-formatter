"""Non-blocking post-import review warnings for OCR adjudication.

These are *risk indicators*, not automatic edits or ground-truth judgments.
The source evidence and accepted decisions remain unchanged.  Callers must
still inspect the scan before changing an accepted adjudication.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Sequence

_DUPLICATE = re.compile(r"([ぁ-んァ-ヶ一-龯]{3,8})\1")
_NUMBER = re.compile(r"(?<!\d)\d{1,4}(?!\d)")
_LATIN = re.compile(r"[A-Z]")
_JAPANESE = re.compile(r"[ぁ-んァ-ヶ一-龯]")
_KATAKANA = re.compile(r"[ァ-ヶー・]{3,}")
_PRINTED_NUMBER = re.compile(r"[0-9０-９]{1,4}")
_GLYPH_GROUPS = ("日目", "古吉", "白自", "王主玉", "方力", "口ロ", "一二三", "一ー―—",
                 "つっ", "ツッ", "やゃ", "ヤャ", "ゆゅ", "ユュ", "よょ", "ヨョ",
                 "あぁ", "アァ", "いぃ", "イィ", "うぅ", "ウゥ", "えぇ", "エェ", "おぉ", "オォ")
_PRINT_SIGNALS = re.compile(r"[!?―—─]|(?<![ァ-ヶ])ー(?![ァ-ヶ])")
_SMALL_KANA = frozenset("ぁぃぅぇぉっゃゅょァィゥェォッャュョ゛゜ﾞﾟ")
# Narrow collocation alerts are evidence-routing hints. They never replace text.
_UNLIKELY_COLLOCATION = re.compile(r"日を細め|(?:が|は)古だ[。！？]|無白覚|明けばない|それまででの|加うえ")


def independently_executed(candidate: dict) -> bool:
    return not candidate.get("seeded_reuse", False) and candidate.get("independently_executed", True) is not False


def independent_texts(item: dict) -> list[str]:
    return [
        str(candidate.get("text", "") or "").strip()
        for candidate in item.get("candidates") or ()
        if isinstance(candidate, dict) and independently_executed(candidate)
    ]


def consensus_scan_reasons(text: str) -> list[str]:
    if not text:
        return []
    reasons = []
    if _UNLIKELY_COLLOCATION.search(text):
        reasons.append("possible_shared_glyph_error")
    if _DUPLICATE.search(text):
        reasons.append("repeated_japanese_sequence")
    if "□" in text or "�" in text:
        reasons.append("unreadable_ocr_characters")
    if _JAPANESE.search(text) and _PRINT_SIGNALS.search(text):
        reasons.append("printing_or_punctuation_needs_scan")
    if _PRINTED_NUMBER.fullmatch(text):
        reasons.append("short_number_or_page_marker")
    return reasons


def candidate_contrasts(texts: Sequence[str]) -> list[dict]:
    """Give the reviewer small, exact change windows without suggesting a winner."""
    values = list(dict.fromkeys(value for value in texts if value))[:6]
    if len(values) < 2:
        return []
    base = values[0]
    contrasts = []
    for alternative in values[1:]:
        for op, a, b, c, d in SequenceMatcher(None, base, alternative, autojunk=False).get_opcodes():
            if op != "equal":
                contrasts.append({
                    "left": base[a:b], "right": alternative[c:d],
                    "left_window": base[max(0, a - 6):b + 6],
                    "right_window": alternative[max(0, c - 6):d + 6],
                })
                if len(contrasts) >= 16:
                    return contrasts
    return contrasts


def sensitive_contrast(left: str, right: str) -> bool:
    if left == right:
        return False
    if _KATAKANA.findall(left) != _KATAKANA.findall(right):
        return True
    for change in candidate_contrasts((left, right)):
        a, b = change["left"], change["right"]
        if any(ch.isdecimal() or ch in _SMALL_KANA or ch in "「」『』―—─!?！？" for ch in a + b):
            return True
        if a and b and any(
            set(a).intersection(group) and set(b).intersection(group)
            and len(set(a + b).intersection(group)) > 1
            for group in _GLYPH_GROUPS
        ):
            return True
    return False


def audit_import_quality_rows(rows: Sequence[dict[str, Any]], *, limit: int = 100) -> list[dict[str, Any]]:
    """Suggest original-image checks for accepted AI decisions, without altering them.

    Only inspect visible AI output and already-sealed OCR alternatives.  In
    particular, no external EPUB, dictionary or previously adjudicated result
    is an input to this method.
    """
    warnings: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        verdict = row.get('ai_verdict') or {}
        if not isinstance(verdict, dict) or verdict.get('delete_intentionally'):
            continue
        value = str(verdict.get('final_text') or '').strip()
        if not value:
            continue
        originals = [str(it.get('text') or '') for it in row.get('model_evidence', ()) if isinstance(it, dict)]
        reasons: list[str] = []
        if _DUPLICATE.search(value):
            # Repeated words can be deliberate in fiction, hence WARNING only.
            reasons.append('repeated_japanese_sequence')
        if value.count('『') != value.count('』') or value.count('「') != value.count('」'):
            reasons.append('unbalanced_japanese_brackets')
        original_nums = {n for original in originals for n in _NUMBER.findall(original)}
        output_nums = set(_NUMBER.findall(value))
        if original_nums and not output_nums and _LATIN.search(value):
            reasons.append('digits_replaced_by_latin_candidate')
        if originals:
            longest = max(map(len, originals))
            if longest >= 24 and len(value) < longest * 0.66:
                reasons.append('large_text_loss_vs_ocr_candidate')
        if reasons:
            warnings.append({
                'row_index': row.get('row_index'),
                'page': row.get('page'),
                'sentence_group_id': row.get('sentence_group_id'),
                'reasons': reasons,
                'sample': value[:100],
                'severity': 'needs_source_image_review',
            })
            if len(warnings) >= limit:
                break
    return warnings
