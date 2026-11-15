"""Conservative, display-only OCR review risk predicates."""
from __future__ import annotations
import re
import unicodedata

_NUMERIC = re.compile(r'\d+(?:[.,:/\-]\d+)*|[〇零一二三四五六七八九十百千万億亿]+(?=[年月日時分秒歳号])')
_REJECTION = ('拒识', '拒認', 'rejection', 'reject evidence', 'unrecognized', 'unrecognised', 'no text detected', '识别失败', '識別失敗')
_REJECT_GLYPHS = ('□', '�', '〓', '▯')


def has_numeric_date_disagreement(row) -> bool:
    texts = tuple(str(t or '') for t in (getattr(row, 'texts', ()) or ()))
    if len(texts) < 2:
        return False
    normalized = [unicodedata.normalize('NFKC', text) for text in texts]
    signatures = [tuple(_NUMERIC.findall(text)) for text in normalized]
    # Only flag non-empty numeric/date evidence which differs *between* OCR
    # sources.  A consistent year is not automatically an OCR error.
    return any(signatures) and len(set(signatures)) > 1


def has_rejection_evidence(row, state=None) -> bool:
    texts = [str(t or '') for t in (getattr(row, 'texts', ()) or ())]
    if any(any(glyph in text for glyph in _REJECT_GLYPHS) for text in texts):
        return True
    fields = list(getattr(row, 'warnings', ()) or ())
    fields += [str(getattr(row, 'reason', '') or '')]
    for candidate in getattr(state, 'candidates', ()) or ():
        fields.extend(str(v) for v in (getattr(candidate, 'audit_flags', ()) or ()))
        fields.extend((str(getattr(candidate, 'audit_level', '') or ''),
                       str(getattr(candidate, 'reason', '') or '')))
    haystack = ' '.join(str(v) for v in fields).lower()
    return any(marker in haystack for marker in _REJECTION)
