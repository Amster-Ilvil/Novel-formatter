"""Non-blocking post-import review warnings for OCR adjudication.

These are *risk indicators*, not automatic edits or ground-truth judgments.
The source evidence and accepted decisions remain unchanged.  Callers must
still inspect the scan before changing an accepted adjudication.
"""
from __future__ import annotations

import re
from typing import Any, Sequence

_DUPLICATE = re.compile(r"([ぁ-んァ-ヶ一-龯]{3,8})\1")
_NUMBER = re.compile(r"(?<!\d)\d{1,4}(?!\d)")
_LATIN = re.compile(r"[A-Z]")


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
