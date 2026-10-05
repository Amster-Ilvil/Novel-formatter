from __future__ import annotations

"""Small, conservative audit queue for OCR common-mode errors.

The normal multi-OCR conflict queue only sees rows where independent OCRs disagree.
This module looks at *locally agreed* rows and re-opens only a bounded set with
signals that are useful to a human/LLM final audit.  It never changes text by
itself and never treats corpus frequency as truth.
"""

from collections import Counter
from difflib import SequenceMatcher
import math
import re
import unicodedata
from typing import Sequence

_PLACEHOLDERS = ("□", "�")
_BRACKETS = (("「", "」"), ("『", "』"), ("（", "）"), ("【", "】"), ("〈", "〉"), ("《", "》"))
_REPEAT_RE = re.compile(r"(.{4,18})\1")
_KATAKANA_SHOUT_RE = re.compile(r"[ァ-ヺー][アイウエオ]{3,}")
_SHORT_MIXED_SCRIPT_RE = re.compile(r"^[一-龯々〆ヵヶ]+[ァ-ヺー]+$|^[ァ-ヺー]+[一-龯々〆ヵヶ]+$")


def _raw(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(text or "")))


def _body(text: str) -> str:
    value = _raw(text)
    return "".join(
        ch for ch in value
        if not unicodedata.category(ch).startswith(("P", "S", "Z", "C"))
    )


def _independent_texts(row: dict) -> list[str]:
    out: list[str] = []
    for item in row.get("model_evidence", []) or []:
        if not isinstance(item, dict) or not bool(item.get("independently_executed", True)):
            continue
        text = str(item.get("text", "") or "")
        if text and text not in out:
            out.append(text)
    return out


def _is_text_char(ch: str) -> bool:
    return len(ch) == 1 and not unicodedata.category(ch).startswith(("P", "S", "Z", "C"))


def _script(ch: str) -> str:
    cp = ord(ch)
    if 0x30A0 <= cp <= 0x30FF:
        return "katakana"
    if 0x3040 <= cp <= 0x309F:
        return "hiragana"
    if 0x3400 <= cp <= 0x9FFF or 0xF900 <= cp <= 0xFAFF:
        return "han"
    if ch.isascii() and ch.isalnum():
        return "latin"
    return "other"


def _learn_confusions(rows: Sequence[dict]) -> Counter[tuple[str, str]]:
    pairs: Counter[tuple[str, str]] = Counter()
    for row in rows:
        values: list[str] = []
        for text in _independent_texts(row):
            value = _body(text)
            if value and value not in values:
                values.append(value)
        if len(values) < 2:
            continue
        for ai in range(len(values)):
            for bi in range(ai + 1, len(values)):
                a, b = values[ai], values[bi]
                for tag, a1, a2, b1, b2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
                    if tag != "replace" or a2 - a1 != 1 or b2 - b1 != 1:
                        continue
                    x, y = a[a1], b[b1]
                    if x == y or not (_is_text_char(x) and _is_text_char(y)):
                        continue
                    sx, sy = _script(x), _script(y)
                    # Common-mode final audit is for likely *visual OCR* errors,
                    # not general Japanese language-model substitutions.  Pure
                    # hiragana grammar variants (か/が, も/で, etc.) create a
                    # very noisy queue and rarely affect translation when all
                    # OCRs agree.  Prioritise Han/Katakana/name-shape confusions.
                    allowed = (
                        sx in {"han", "katakana"}
                        and sy in {"han", "katakana"}
                    )
                    if not allowed:
                        continue
                    pairs[tuple(sorted((x, y)))] += 1
    return pairs


def _consensus_text(row: dict) -> str:
    values = [_raw(value) for value in _independent_texts(row) if _raw(value)]
    if not values:
        return ""
    # Common-mode audit is intentionally limited to rows where independent OCRs
    # already agree after the compare-only Unicode normalization.
    bodies = {_body(value) for value in values if _body(value)}
    if len(bodies) != 1:
        return ""
    return values[0]


def _structural_reasons(text: str) -> list[str]:
    reasons: list[str] = []
    if any(marker in text for marker in _PLACEHOLDERS):
        reasons.append("placeholder_in_consensus")
    for left, right in _BRACKETS:
        if text.count(left) != text.count(right):
            reasons.append(f"unbalanced_{left}{right}")
    compact = _raw(text)
    if len(compact) >= 12 and _REPEAT_RE.search(compact):
        reasons.append("repeated_phrase")
    # A leading open quote followed by a long narrative-looking continuation is
    # a common signature of neighbour/sentence adhesion.  Do not guess a fix;
    # merely route it for image-backed review.
    if compact.startswith(("「", "『")) and not compact.endswith(("」", "』")) and len(compact) >= 18:
        reasons.append("possible_dialogue_adhesion")

    # Generative/line OCR often normalises emphatic small vowels to full-size
    # vowels (e.g. コラァァァ -> コラアアア).  This is only a review trigger:
    # the sentence image remains the authority and no character is changed here.
    body = _body(compact)
    if _KATAKANA_SHOUT_RE.search(body):
        reasons.append("katakana_emphasis_small_kana_risk")

    # Very short mixed Han/Katakana tokens are disproportionately vulnerable to
    # visual homographs such as 二/ニ, 力/カ and 口/ロ.  Restrict this trigger to
    # exactly two body characters so ordinary mixed-script prose/proper-name+title
    # combinations are not flooded into the bounded final-audit queue.
    if len(body) == 2 and _SHORT_MIXED_SCRIPT_RE.fullmatch(body):
        reasons.append("short_han_katakana_homograph_risk")
    return list(dict.fromkeys(reasons))


def _sequence_omission_reason(consensus: Sequence[str], row_index: int) -> str | None:
    """Flag a likely missing first item in a compact 2→3 sequence.

    This does not infer the missing text.  It merely re-opens the row so the
    sentence image can prove whether a leading 一/1 item was cropped away.
    """
    if row_index < 0 or row_index + 1 >= len(consensus):
        return None
    current = _body(consensus[row_index])
    nxt = _body(consensus[row_index + 1])
    if not current or not nxt:
        return None
    m2 = re.fullmatch(r"(?:二|2)(.{1,6})", current)
    m3 = re.fullmatch(r"(?:三|3)(.{1,6})", nxt)
    if m2 and m3 and m2.group(1) == m3.group(1):
        return "possible_leading_sequence_omission"
    return None


def _ngram_counts(texts: Sequence[str]) -> dict[int, Counter[str]]:
    counts = {n: Counter() for n in (2, 3, 4, 5)}
    for value in texts:
        text = _body(value)
        for n, bucket in counts.items():
            for index in range(max(0, len(text) - n + 1)):
                bucket[text[index:index + n]] += 1
    return counts


def _lexical_candidates(
    rows: Sequence[dict],
    consensus: Sequence[str],
    *,
    limit: int,
) -> list[dict]:
    pair_counts = _learn_confusions(rows)
    alternatives: dict[str, set[str]] = {}
    for (left, right), count in pair_counts.items():
        if count < 2:
            continue
        alternatives.setdefault(left, set()).add(right)
        alternatives.setdefault(right, set()).add(left)

    corpus: list[str] = []
    for row in rows:
        # Use every distinct independently executed observation.  Using only
        # model-0 would bake one engine's systematic spelling error into the
        # book-frequency model (notably Han/Katakana proper-name confusions).
        seen: set[str] = set()
        for text in _independent_texts(row):
            key = _body(text)
            if not key or key in seen:
                continue
            seen.add(key)
            corpus.append(text)
    ngrams = _ngram_counts(corpus)

    ranked: list[dict] = []
    for row_index, raw_text in enumerate(consensus):
        text = _body(raw_text)
        if not text:
            continue
        best: dict | None = None
        for pos, char in enumerate(text):
            for alt in alternatives.get(char, ()):
                delta = 0.0
                evidence: list[dict] = []
                for n in (2, 3, 4, 5):
                    start_lo = max(0, pos - n + 1)
                    start_hi = min(pos, len(text) - n)
                    for start in range(start_lo, start_hi + 1):
                        original = text[start:start + n]
                        candidate = original[:pos - start] + alt + original[pos - start + 1:]
                        original_count = max(0, ngrams[n][original] - 1)
                        alt_count = ngrams[n][candidate]
                        if alt_count <= original_count:
                            continue
                        gain = math.log1p(alt_count) - math.log1p(original_count)
                        delta += gain * (n - 1)
                        if n >= 3 and alt_count >= 2:
                            evidence.append({
                                "from": original,
                                "to": candidate,
                                "from_count": original_count,
                                "to_count": alt_count,
                            })
                # Mixed Han/Katakana flips are especially useful for proper-name
                # consistency checks (e.g. OCR 二 vs ニ).  This is still only a
                # review signal, never an automatic replacement.
                script_bonus = 0.0
                if {_script(char), _script(alt)} == {"han", "katakana"}:
                    script_bonus = 4.0
                score = delta + script_bonus
                if score < 2.0 or not evidence:
                    continue
                item = {
                    "row_index": row_index,
                    "score": score,
                    "reason": "book_variant_outlier",
                    "char": char,
                    "alternate_char": alt,
                    "evidence": evidence[:4],
                }
                if best is None or float(item["score"]) > float(best["score"]):
                    best = item
        if best is not None:
            ranked.append(best)
    ranked.sort(key=lambda item: (-float(item["score"]), int(item["row_index"])))
    return ranked[:max(0, int(limit))]


def select_common_mode_risks(
    rows: Sequence[dict],
    *,
    lexical_limit: int = 64,
    max_total: int = 128,
) -> list[dict]:
    """Return a bounded advisory/review queue for locally agreed rows.

    Rows already editable/resolved are excluded.  The output is deterministic
    for a fixed OCR payload and contains only evidence/reasons; callers decide
    whether to reopen the row for AI review.
    """
    consensus = [_consensus_text(row) if not bool(row.get("editable")) else "" for row in rows]
    selected: dict[int, dict] = {}

    for row_index, text in enumerate(consensus):
        if not text:
            continue
        reasons = _structural_reasons(text)
        sequence_reason = _sequence_omission_reason(consensus, row_index)
        if sequence_reason:
            reasons.append(sequence_reason)
        if not reasons:
            continue
        selected[row_index] = {
            "row_index": row_index,
            "score": 1000.0 + 10.0 * len(reasons),
            "reasons": reasons,
            "kind": "structural",
        }

    lexical_pool = _lexical_candidates(rows, consensus, limit=max(int(lexical_limit), 512))
    lexical_selected = list(lexical_pool[:max(0, int(lexical_limit))])
    # Proper names are disproportionately sensitive to Han/Katakana lookalikes.
    # Keep a small supplement even when those rows fall below the global top-N.
    # This remains review-only; it never decides which script form is correct.
    already = {int(item["row_index"]) for item in lexical_selected}
    for item in lexical_pool:
        if int(item["row_index"]) in already:
            continue
        if float(item.get("score", 0.0)) < 4.5:
            continue
        if {_script(str(item.get("char", ""))), _script(str(item.get("alternate_char", "")))} != {"han", "katakana"}:
            continue
        lexical_selected.append(item)
        already.add(int(item["row_index"]))

    for item in lexical_selected:
        row_index = int(item["row_index"])
        current = selected.get(row_index)
        reason = {
            "kind": "book_variant_outlier",
            "char": item.get("char", ""),
            "alternate_char": item.get("alternate_char", ""),
            "evidence": item.get("evidence", []),
        }
        if current is None:
            selected[row_index] = {
                "row_index": row_index,
                "score": float(item.get("score", 0.0)),
                "reasons": [reason],
                "kind": "lexical",
            }
        else:
            current.setdefault("reasons", []).append(reason)
            current["score"] = max(float(current.get("score", 0.0)), float(item.get("score", 0.0)))

    ordered = sorted(
        selected.values(),
        key=lambda item: (-float(item.get("score", 0.0)), int(item.get("row_index", -1))),
    )
    return ordered[:max(0, int(max_total))]
