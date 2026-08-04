#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reference-assisted comparison for OCR and AI review.

A reference file can be an author draft, web-serialization text, proof copy, or
other mostly-correct version.  It is *not* assumed to be byte-identical to the
published edition.  The module therefore never replaces document text directly.
It finds high-confidence shared passages and uses them as evidence when judging
OCR/AI changes.

Design goals:

* standard-library only; no mandatory fuzzy-matching dependency;
* fast enough for 4,000-6,000 line novels through a rare n-gram shortlist;
* punctuation/whitespace tolerant, while preserving the original excerpt;
* conservative decisions when the draft and the published edition diverge;
* reusable both for whole-version scoring and per-AI-change review.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
import bisect
import re
import statistics
import unicodedata
from typing import Iterable, Sequence


_IMAGE_MARKER_RE = re.compile(r"^⟦插图｜.*⟧$")
_METADATA_RE = re.compile(
    r"^(?:总计\s*:|百度\s*:|有道\s*:|GPT\s*:|Sakura\s*:|冒険\s*/?$|"
    r"[^\s]{1,40}（由私人备份提供翻译）$|[〇零一二三四五六七八九十百千万0-9]+／[^\s]{1,40}$)"
)
_ONLY_DECORATION_RE = re.compile(r"^[#=_*\-—―\s]+$")

# These punctuation variants are visually/semantically equivalent for fuzzy
# matching.  Quotes and punctuation are then removed from the content key; the
# original text remains untouched for display and structure checks.
_PUNCT_TRANS = str.maketrans({
    "．": "。", ".": "。", "!": "！", "?": "？", "﹗": "！", "﹖": "？",
    "―": "—", "ｰ": "ー", "～": "〜", "…": "⋯", "“": "「", "”": "」",
    "‘": "『", "’": "』", "﹁": "「", "﹂": "」", "﹃": "『", "﹄": "』",
})
_CONTENT_DROP_RE = re.compile(
    r"[\s\u3000、。！？!?,.：:；;・…⋯‥—―ー~〜～「」『』“”‘’（）()\[\]【】《》〈〉〔〕〝〟]+"
)
_STRUCTURE_SPACE_RE = re.compile(r"[\s\u3000]+")


@dataclass(slots=True)
class ReferenceSegment:
    index: int
    line_number: int
    text: str
    content_key: str
    structure_key: str


@dataclass(slots=True)
class CorpusWindow:
    index: int
    start: int
    end: int
    text: str
    content_key: str
    structure_key: str


@dataclass(slots=True)
class ReferenceMatch:
    score: float = 0.0
    structure_score: float = 0.0
    start: int = -1
    end: int = -1
    line_start: int = -1
    line_end: int = -1
    text: str = ""
    exact: bool = False
    containment: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class ReferenceVersionScore:
    label: str
    reference_units: int
    evaluated_units: int
    exact_units: int
    contained_units: int
    near_98_units: int
    near_95_units: int
    matched_90_units: int
    matched_85_units: int
    ordered_90_units: int
    mean_score: float
    content_mean_score: float
    median_score: float
    overall_score: float

    @property
    def exact_rate(self) -> float:
        return self.exact_units / max(1, self.evaluated_units)

    @property
    def contained_rate(self) -> float:
        return self.contained_units / max(1, self.evaluated_units)

    @property
    def near_98_rate(self) -> float:
        return self.near_98_units / max(1, self.evaluated_units)

    @property
    def matched_90_rate(self) -> float:
        return self.matched_90_units / max(1, self.evaluated_units)

    @property
    def order_coverage(self) -> float:
        return self.ordered_90_units / max(1, self.evaluated_units)

    def to_dict(self) -> dict:
        data = asdict(self)
        data.update({
            "exact_rate": self.exact_rate,
            "contained_rate": self.contained_rate,
            "near_98_rate": self.near_98_rate,
            "matched_90_rate": self.matched_90_rate,
            "order_coverage": self.order_coverage,
        })
        return data


@dataclass(slots=True)
class ReferenceDecision:
    relation: str = "unmatched"  # supports_after | supports_before | neutral | unmatched
    confidence: float = 0.0
    before_score: float = 0.0
    after_score: float = 0.0
    reference_excerpt: str = ""
    line_start: int = -1
    line_end: int = -1

    def to_dict(self) -> dict:
        return asdict(self)


def normalise_reference_content(text: str) -> str:
    value = unicodedata.normalize("NFKC", text or "").translate(_PUNCT_TRANS)
    return _CONTENT_DROP_RE.sub("", value).casefold()


def normalise_reference_structure(text: str) -> str:
    value = unicodedata.normalize("NFKC", text or "").translate(_PUNCT_TRANS)
    return _STRUCTURE_SPACE_RE.sub("", value).casefold()


def _meaningful_reference_lines(text: str, *, keep_headings: bool = True) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    for line_number, raw in enumerate((text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"), 1):
        stripped = raw.strip().lstrip("\ufeff")
        if not stripped or _IMAGE_MARKER_RE.fullmatch(stripped):
            continue
        if _METADATA_RE.match(stripped) or _ONLY_DECORATION_RE.fullmatch(stripped):
            continue
        if stripped.startswith("#"):
            stripped = re.sub(r"^#+\s*", "", stripped).strip()
            if not keep_headings or not stripped:
                continue
        if len(normalise_reference_content(stripped)) < 2:
            continue
        result.append((line_number, stripped))
    return result


def reference_display_lines(text: str) -> list[tuple[int, str]]:
    """Return cleaned, human-readable lines for the compare workspace."""
    return _meaningful_reference_lines(text, keep_headings=True)


def _sample_ngrams(value: str, n: int = 5) -> set[str]:
    if not value:
        return set()
    if len(value) <= n:
        return {value}
    # Step 2 keeps the index compact while retaining enough overlap for OCR
    # substitutions.  Always include both ends because line breaks often move.
    grams = {value[i:i + n] for i in range(0, len(value) - n + 1, 2)}
    grams.add(value[:n])
    grams.add(value[-n:])
    return grams


def _similarity(query: str, candidate: str) -> tuple[float, bool, bool]:
    if not query or not candidate:
        return 0.0, False, False
    if query == candidate:
        return 1.0, True, True
    contained = query in candidate or candidate in query
    ratio = SequenceMatcher(None, query, candidate, autojunk=False).ratio()
    if contained:
        balance = min(len(query), len(candidate)) / max(len(query), len(candidate))
        # A line fully contained in a two-line window is strong evidence, but a
        # tiny generic phrase inside a long paragraph must not score as exact.
        ratio = max(ratio, 0.90 + 0.10 * balance)
    return min(1.0, ratio), False, contained


def _lis_length(values: Sequence[int]) -> int:
    tails: list[int] = []
    for value in values:
        pos = bisect.bisect_right(tails, value)
        if pos == len(tails):
            tails.append(value)
        else:
            tails[pos] = value
    return len(tails)


class ReferenceCorpus:
    """Searchable reference/draft corpus.

    ``max_window`` controls how many adjacent physical lines may form one search
    target.  Three lines handles most OCR/Formatter merge-split differences
    without letting unrelated paragraphs become one fuzzy candidate.
    """

    def __init__(self, segments: Sequence[ReferenceSegment], *, max_window: int = 3):
        self.segments = list(segments)
        self.max_window = max(1, min(4, int(max_window)))
        self.windows: list[CorpusWindow] = []
        self._gram_index: dict[str, list[int]] = defaultdict(list)
        self._build_windows()

    @classmethod
    def from_text(cls, text: str, *, max_window: int = 3) -> "ReferenceCorpus":
        segments: list[ReferenceSegment] = []
        for _idx, (line_number, value) in enumerate(_meaningful_reference_lines(text, keep_headings=True)):
            content = normalise_reference_content(value)
            if len(content) < 2:
                continue
            segments.append(ReferenceSegment(
                index=len(segments),
                line_number=line_number,
                text=value,
                content_key=content,
                structure_key=normalise_reference_structure(value),
            ))
        return cls(segments, max_window=max_window)

    @classmethod
    def from_lines(cls, lines: Iterable[str], *, max_window: int = 3) -> "ReferenceCorpus":
        text = "\n".join(str(item) for item in lines)
        return cls.from_text(text, max_window=max_window)

    def _build_windows(self):
        self.windows.clear()
        self._gram_index.clear()
        for start in range(len(self.segments)):
            text_parts: list[str] = []
            content_parts: list[str] = []
            structure_parts: list[str] = []
            for end in range(start, min(len(self.segments), start + self.max_window)):
                segment = self.segments[end]
                text_parts.append(segment.text)
                content_parts.append(segment.content_key)
                structure_parts.append(segment.structure_key)
                content_key = "".join(content_parts)
                if len(content_key) < 4:
                    continue
                window = CorpusWindow(
                    index=len(self.windows),
                    start=start,
                    end=end,
                    text="\n".join(text_parts),
                    content_key=content_key,
                    structure_key="".join(structure_parts),
                )
                self.windows.append(window)
                for gram in _sample_ngrams(content_key):
                    self._gram_index[gram].append(window.index)

    def __len__(self) -> int:
        return len(self.segments)

    def search(self, text: str, *, max_candidates: int = 96) -> ReferenceMatch:
        query_content = normalise_reference_content(text)
        query_structure = normalise_reference_structure(text)
        if len(query_content) < 4 or not self.windows:
            return ReferenceMatch()

        votes: Counter[int] = Counter()
        for gram in _sample_ngrams(query_content):
            postings = self._gram_index.get(gram, ())
            # Very common n-grams (e.g. ということ) are poor anchors and can
            # dominate both runtime and false matches.
            if len(postings) > 180:
                continue
            for index in postings:
                votes[index] += 1

        if votes:
            candidate_ids = [item[0] for item in votes.most_common(max_candidates)]
        else:
            # Short or heavily corrupted text may share no sampled 5-gram.  A
            # bounded length-aware fallback is still cheap for a single change.
            target_len = len(query_content)
            candidate_ids = [
                w.index for w in self.windows
                if 0.45 <= len(w.content_key) / max(1, target_len) <= 2.2
            ][:max_candidates]

        best = ReferenceMatch()
        for window_id in candidate_ids:
            window = self.windows[window_id]
            score, exact, contained = _similarity(query_content, window.content_key)
            if score + 1e-9 < best.score:
                continue
            structure_score = SequenceMatcher(
                None, query_structure, window.structure_key, autojunk=False
            ).ratio() if query_structure and window.structure_key else 0.0
            if score > best.score or structure_score > best.structure_score:
                first = self.segments[window.start]
                last = self.segments[window.end]
                best = ReferenceMatch(
                    score=score,
                    structure_score=structure_score,
                    start=window.start,
                    end=window.end,
                    line_start=first.line_number,
                    line_end=last.line_number,
                    text=window.text,
                    exact=exact,
                    containment=contained,
                )
        return best

    def score_candidate(self, lines: Iterable[str], *, label: str = "candidate") -> ReferenceVersionScore:
        candidate = ReferenceCorpus.from_lines(lines, max_window=self.max_window)
        scores: list[float] = []
        positions_90: list[int] = []
        exact = contained = near98 = near95 = match90 = match85 = 0
        content_scores: list[float] = []

        for segment in self.segments:
            # Very short utterances are duplicated throughout a novel and do not
            # provide reliable evidence about OCR quality.
            if len(segment.content_key) < 10:
                continue
            match = candidate.search(segment.text)
            score = match.score
            scores.append(score)
            content_scores.append(1.0 if match.containment else score)
            if match.exact:
                exact += 1
            if match.containment:
                contained += 1
            if score >= 0.98:
                near98 += 1
            if score >= 0.95:
                near95 += 1
            if score >= 0.90:
                match90 += 1
                positions_90.append(match.start)
            if score >= 0.85:
                match85 += 1

        evaluated = len(scores)
        mean_score = statistics.fmean(scores) if scores else 0.0
        content_mean_score = statistics.fmean(content_scores) if content_scores else 0.0
        median_score = statistics.median(scores) if scores else 0.0
        ordered = _lis_length(positions_90)
        near95_rate = near95 / max(1, evaluated)
        ordered_rate = ordered / max(1, evaluated)
        # This is a comparative reference score, not an accuracy percentage.
        # Content similarity dominates; ordered coverage prevents generic lines
        # matched elsewhere in the book from inflating the result.
        overall = 100.0 * (0.62 * mean_score + 0.23 * near95_rate + 0.15 * ordered_rate)
        return ReferenceVersionScore(
            label=label,
            reference_units=len(self.segments),
            evaluated_units=evaluated,
            exact_units=exact,
            contained_units=contained,
            near_98_units=near98,
            near_95_units=near95,
            matched_90_units=match90,
            matched_85_units=match85,
            ordered_90_units=ordered,
            mean_score=mean_score,
            content_mean_score=content_mean_score,
            median_score=median_score,
            overall_score=overall,
        )

    def judge_change(self, before: str, after: str) -> ReferenceDecision:
        before_match = self.search(before) if normalise_reference_content(before) else ReferenceMatch()
        after_match = self.search(after) if normalise_reference_content(after) else ReferenceMatch()
        before_score = before_match.score
        after_score = after_match.score
        best_match = after_match if after_score >= before_score else before_match
        margin = abs(after_score - before_score)

        relation = "unmatched"
        structure_margin = abs(after_match.structure_score - before_match.structure_score)
        # When lexical content is identical, quotation and punctuation structure
        # may be the only evidence (for example restoring a missing 「 opener).
        if (
            max(before_score, after_score) >= 0.95
            and margin <= 0.015
            and structure_margin >= 0.045
        ):
            relation = (
                "supports_after"
                if after_match.structure_score > before_match.structure_score
                else "supports_before"
            )
        elif max(before_score, after_score) >= 0.90:
            if after_score >= 0.94 and after_score - before_score >= 0.045:
                relation = "supports_after"
            elif before_score >= 0.94 and before_score - after_score >= 0.045:
                relation = "supports_before"
            else:
                relation = "neutral"
        elif max(before_score, after_score) >= 0.84 and margin >= 0.10:
            relation = "supports_after" if after_score > before_score else "supports_before"

        confidence = 0.0
        if relation in {"supports_after", "supports_before"}:
            decisive_margin = max(margin, structure_margin)
            confidence = min(1.0, max(before_score, after_score) * min(1.0, decisive_margin / 0.12))
        elif relation == "neutral":
            confidence = max(before_score, after_score) * 0.55

        return ReferenceDecision(
            relation=relation,
            confidence=confidence,
            before_score=before_score,
            after_score=after_score,
            reference_excerpt=best_match.text,
            line_start=best_match.line_start,
            line_end=best_match.line_end,
        )


def annotate_ai_review_changes(changes: Sequence[object], corpus: ReferenceCorpus | None) -> Sequence[object]:
    """Attach reference evidence to ``AIReviewChange``-like objects in place."""
    if corpus is None:
        return changes
    for change in changes:
        before = "\n".join(getattr(change, "before_lines", ()) or ())
        after = "\n".join(getattr(change, "after_lines", ()) or ())
        decision = corpus.judge_change(before, after)
        for name, value in (
            ("reference_relation", decision.relation),
            ("reference_confidence", decision.confidence),
            ("reference_before_score", decision.before_score),
            ("reference_after_score", decision.after_score),
            ("reference_excerpt", decision.reference_excerpt),
            ("reference_line_start", decision.line_start),
            ("reference_line_end", decision.line_end),
        ):
            try:
                setattr(change, name, value)
            except Exception:
                pass
    return changes




