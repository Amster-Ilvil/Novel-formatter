#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conservative whole-column OCR gate for per-glyph fallback recognition.

The ordinary OCR result is treated as evidence only. It is never used to
create glyph boxes or decide character count. A character is copied from the
column OCR only when all safety gates pass; every other slot keeps the existing
per-glyph result (glyph memory / PKStroke / ``□``).

Design notes absorbed from open OCR projects:
* PARSeq/NDLOCR-Lite: sequence recognizers are useful because they see context,
  but acceptance must remain token/slot based rather than trusting one average
  line score.
* PaddleOCR/MMOCR: detector/recognizer stages and their evidence are kept
  separate; this module consumes recognizer evidence but never changes layout.
* Heterogeneous consensus: an optional second OCR engine may scan the exact
  same original column once. Disagreement rejects only the affected slot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
import re
import sqlite3
import unicodedata
from typing import Iterable, Mapping, Sequence

from adapters.glyph_memory_db import default_db_path

_REPLACEMENT = set("□■◼◻�?？")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(slots=True)
class SelectiveCharEvidence:
    index: int
    ocr_char: str = ""
    fallback_char: str = "□"
    output_char: str = "□"
    accepted: bool = False
    score: float = 0.0
    reason: str = ""
    conflict: bool = False
    raw_character_confidence: float = 0.0
    effective_character_confidence: float = 0.0
    stability_support: int = 0
    stability_total: int = 0
    stability_ratio: float = 0.0
    variant_characters: list[str] = field(default_factory=list)
    cross_engine_support: bool = False
    character_calibration_samples: int = 0
    character_calibration_lower_bound: float | None = None
    primary_source_index: int | None = None
    alignment_status: str = ""
    secondary_characters: list[str] = field(default_factory=list)
    provisional: bool = False
    automatic_fallback: bool = True
    glyph_rescue_character: str = ""
    glyph_rescue_score: float = 0.0
    glyph_rescue_support: int = 0
    glyph_rescue_total: int = 0
    glyph_rescue_reason: str = ""
    glyph_rescue_view_characters: list[str] = field(default_factory=list)
    glyph_rescue_confirmed: bool = False


@dataclass(slots=True)
class SelectiveColumnReport:
    enabled: bool = False
    engine: str = ""
    candidate_text: str = ""
    raw_confidence: float = 0.0
    effective_confidence: float = 0.0
    threshold: float = 0.95
    exact_length: bool = False
    accepted_indices: list[int] = field(default_factory=list)
    rejected_indices: list[int] = field(default_factory=list)
    conflict_indices: list[int] = field(default_factory=list)
    unstable_indices: list[int] = field(default_factory=list)
    decisions: list[SelectiveCharEvidence] = field(default_factory=list)
    reason: str = ""
    calibration_samples: int = 0
    calibration_lower_bound: float | None = None
    require_stability: bool = False
    stability_evidence_count: int = 0
    ignored_variant_count: int = 0
    variant_summaries: list[dict] = field(default_factory=list)
    alignment_reference_text: str = ""
    alignment_reference_engine: str = ""
    length_recovered_by_alignment: bool = False
    alignment_ambiguous_indices: list[int] = field(default_factory=list)
    alignment_gap_indices: list[int] = field(default_factory=list)
    provisional_indices: list[int] = field(default_factory=list)
    automatic_fallback_indices: list[int] = field(default_factory=list)
    manual_only_indices: list[int] = field(default_factory=list)
    preserve_original_text: bool = False

    @property
    def accepted_count(self) -> int:
        return len(self.accepted_indices)


@dataclass(frozen=True, slots=True)
class EngineProfile:
    key: str
    confidence_penalty: float
    confidence_is_real: bool
    automatic_allowed: bool = True


_ENGINE_PROFILES: tuple[tuple[str, EngineProfile], ...] = (
    ("paddle_ocr", EngineProfile("paddle_ocr", 0.030, True)),
    ("paddle_structure", EngineProfile("paddle_structure", 0.040, True)),
    ("ndlocr", EngineProfile("ndlocr_lite", 0.040, True)),
    ("google_vision", EngineProfile("google_vision", 0.040, True)),
    ("yomitoku", EngineProfile("yomitoku", 0.025, True)),
    ("apple_vision", EngineProfile("apple_vision", 0.035, True, False)),
    # Manga-OCR currently returns a fixed 0.92 in this project, not a measured
    # probability. It may support another engine but is never sufficient alone.
    ("manga_ocr", EngineProfile("manga_ocr", 0.100, False)),
    # Hayai v2.1 also returns generated sequence text without calibrated token probabilities.
    # Treat its score as heuristic evidence only; cross-engine/stability gates remain authoritative.
    ("hayai_ocr", EngineProfile("hayai_ocr", 0.090, False)),
    ("manga_48px", EngineProfile("manga_48px", 0.120, False)),
    ("pdf_craft", EngineProfile("pdf_craft", 0.080, False)),
    ("paddle_vl", EngineProfile("paddle_vl", 0.100, False)),
)


def _engine_match_key(value: str) -> str:
    """Normalize both stable engine IDs and human-facing labels for matching.

    OCR evidence can arrive from saved metadata using either ``hayai_ocr`` or
    labels such as ``Hayai OCR v2.1``. Treat separators/version suffixes as
    presentation details so confidence policy cannot silently fall back to the
    generic profile merely because the GUI label was persisted.
    """
    raw = str(value or "").strip().lower()
    raw = re.sub(r"[\s\-./]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    return raw


def normalize_engine_name(value: str) -> str:
    raw = str(value or "").strip().lower()
    if raw.startswith("masked_column_ocr:"):
        raw = raw.split(":", 1)[1].strip()
    match_key = _engine_match_key(raw)
    compact_key = match_key.replace("_", "")
    for needle, profile in _ENGINE_PROFILES:
        if needle in match_key or needle.replace("_", "") in compact_key:
            return profile.key
    return match_key or "unknown"


def engine_profile(value: str) -> EngineProfile:
    key = normalize_engine_name(value)
    for _needle, profile in _ENGINE_PROFILES:
        if profile.key == key:
            return profile
    return EngineProfile(key, 0.060, False)


def clean_column_text(text: str) -> str:
    """Normalize transport noise without changing Japanese punctuation."""
    value = unicodedata.normalize("NFC", str(text or ""))
    value = _WHITESPACE_RE.sub("", value)
    return "".join(ch for ch in value if ch not in {"\ufeff", "\u200b", "\u2060"})


def _valid_ocr_char(ch: str) -> bool:
    return bool(ch and ch not in _REPLACEMENT and not ch.isspace())


def _clamp_confidence(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value or 0.0)))
    except Exception:
        return 0.0


def _wilson_lower_bound(correct: int, total: int, z: float = 1.959963984540054) -> float:
    if total <= 0:
        return 0.0
    p = correct / total
    denominator = 1.0 + (z * z) / total
    center = p + (z * z) / (2.0 * total)
    margin = z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * total)) / total)
    return max(0.0, min(1.0, (center - margin) / denominator))


class SelectiveOcrCalibrationDB:
    """Review-backed calibration and hard-negative evidence beside glyph memory."""

    def __init__(self, path=None):
        self.path = default_db_path() if path is None else path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=20)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS selective_ocr_calibration (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    engine TEXT NOT NULL,
                    raw_confidence REAL NOT NULL,
                    predicted_character TEXT NOT NULL,
                    confirmed_character TEXT NOT NULL,
                    correct INTEGER NOT NULL,
                    source TEXT NOT NULL DEFAULT 'manual_review',
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_selective_ocr_engine_score "
                "ON selective_ocr_calibration(engine, raw_confidence)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_selective_ocr_engine_character_score "
                "ON selective_ocr_calibration(engine, predicted_character, raw_confidence)"
            )

    def record(
        self,
        *,
        engine: str,
        raw_confidence: float,
        predicted_character: str,
        confirmed_character: str,
        source: str = "manual_review",
    ) -> bool:
        predicted = str(predicted_character or "")[:1]
        confirmed = str(confirmed_character or "")[:1]
        if not predicted or not confirmed:
            return False
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO selective_ocr_calibration(
                    engine, raw_confidence, predicted_character,
                    confirmed_character, correct, source, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalize_engine_name(engine),
                    _clamp_confidence(raw_confidence),
                    predicted,
                    confirmed,
                    int(predicted == confirmed),
                    str(source or "manual_review"),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        return True

    def estimate(self, *, engine: str, raw_confidence: float, minimum_samples: int = 20) -> tuple[int, float | None]:
        key = normalize_engine_name(engine)
        threshold = _clamp_confidence(raw_confidence)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS n, COALESCE(SUM(correct), 0) AS ok
                FROM selective_ocr_calibration
                WHERE engine = ? AND raw_confidence >= ?
                """,
                (key, threshold),
            ).fetchone()
        total = int(row["n"] or 0)
        correct = int(row["ok"] or 0)
        if total < int(minimum_samples):
            return total, None
        return total, _wilson_lower_bound(correct, total)

    def estimate_character(
        self,
        *,
        engine: str,
        predicted_character: str,
        raw_confidence: float,
        minimum_samples: int = 5,
    ) -> tuple[int, int, int, float | None]:
        key = normalize_engine_name(engine)
        predicted = str(predicted_character or "")[:1]
        threshold = _clamp_confidence(raw_confidence)
        if not predicted:
            return 0, 0, 0, None
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS n, COALESCE(SUM(correct), 0) AS ok
                FROM selective_ocr_calibration
                WHERE engine = ? AND predicted_character = ? AND raw_confidence >= ?
                """,
                (key, predicted, threshold),
            ).fetchone()
        total = int(row["n"] or 0)
        correct = int(row["ok"] or 0)
        errors = max(0, total - correct)
        lower = _wilson_lower_bound(correct, total) if total >= int(minimum_samples) else None
        return total, correct, errors, lower


def calibrated_confidence(
    *,
    engine: str,
    raw_confidence: float,
    calibration_db: SelectiveOcrCalibrationDB | None = None,
) -> tuple[float, int, float | None, EngineProfile]:
    profile = engine_profile(engine)
    raw = _clamp_confidence(raw_confidence)
    effective = max(0.0, raw - profile.confidence_penalty)
    samples = 0
    lower: float | None = None
    if calibration_db is not None:
        try:
            samples, lower = calibration_db.estimate(engine=profile.key, raw_confidence=raw)
        except Exception:
            samples, lower = 0, None
    if lower is not None:
        effective = min(effective, lower)
    return effective, samples, lower, profile


def _normalise_variants(
    variants: Sequence[Mapping[str, object]] | None,
    *,
    primary_text: str,
    primary_confidence: float,
    primary_engine: str,
    primary_character_confidences: Sequence[float] | None,
) -> list[dict]:
    out: list[dict] = []
    primary_chars = [
        _clamp_confidence(value) for value in (primary_character_confidences or [])
    ]
    out.append({
        "label": "selected",
        "text": primary_text,
        "confidence": _clamp_confidence(primary_confidence),
        "engine": normalize_engine_name(primary_engine),
        "character_confidences": primary_chars,
        "is_primary": True,
    })
    seen_labels: set[tuple[str, str, str]] = {
        ("selected", normalize_engine_name(primary_engine), primary_text)
    }
    for serial, raw in enumerate(variants or [], start=1):
        if not isinstance(raw, Mapping):
            continue
        # candidate_text already represents the selected OCR pass. Do not count
        # the same pass twice merely because it is also preserved in the audit.
        if bool(raw.get("selected")):
            continue
        text = clean_column_text(str(raw.get("text") or ""))
        if not text:
            continue
        label = str(raw.get("label") or raw.get("variant") or f"variant_{serial}")
        engine = normalize_engine_name(str(raw.get("engine") or primary_engine))
        key = (label, engine, text)
        if key in seen_labels:
            continue
        seen_labels.add(key)
        char_scores = [
            _clamp_confidence(value)
            for value in (raw.get("character_confidences") or [])
        ]
        out.append({
            "label": label,
            "text": text,
            "confidence": _clamp_confidence(raw.get("confidence")),
            "engine": engine,
            "character_confidences": char_scores,
            "is_primary": False,
        })
    return out


@dataclass(slots=True)
class _SlotAlignment:
    """Conservative mapping from one OCR string to fixed glyph slots."""

    source_indices: list[int | None]
    ambiguous_indices: set[int] = field(default_factory=set)
    gap_indices: set[int] = field(default_factory=set)
    edit_distance: int = 0
    usable: bool = True
    method: str = "direct"


def _levenshtein_slot_alignment(reference: str, query: str) -> _SlotAlignment:
    """Map ``query`` characters onto ``reference`` positions.

    The reference is already known to contain exactly one character per glyph
    box. For unequal lengths we inspect *all* optimal Levenshtein paths. A slot
    is mapped only when every optimal path assigns the same query character to
    it and no optimal path leaves that slot as a gap. This deliberately rejects
    the uncertain area around a missing/extra OCR character while preserving
    unaffected prefixes and suffixes.
    """
    n, m = len(reference), len(query)
    if n == m:
        return _SlotAlignment(list(range(n)), edit_distance=sum(a != b for a, b in zip(reference, query)))
    if not n or not m:
        return _SlotAlignment([None] * n, gap_indices=set(range(n)), edit_distance=max(n, m), usable=False, method="empty")
    max_delta = max(3, int(math.ceil(n * 0.15)))
    if abs(n - m) > max_delta:
        return _SlotAlignment([None] * n, gap_indices=set(range(n)), edit_distance=abs(n - m), usable=False, method="length_delta_too_large")

    # Forward minimum edit cost.
    forward = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        forward[i][0] = i
    for j in range(1, m + 1):
        forward[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = 0 if reference[i - 1] == query[j - 1] else 1
            forward[i][j] = min(
                forward[i - 1][j - 1] + sub,
                forward[i - 1][j] + 1,
                forward[i][j - 1] + 1,
            )

    # Backward minimum edit cost from each suffix state.
    backward = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        backward[i][m] = n - i
    for j in range(m - 1, -1, -1):
        backward[n][j] = m - j
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            sub = 0 if reference[i] == query[j] else 1
            backward[i][j] = min(
                backward[i + 1][j + 1] + sub,
                backward[i + 1][j] + 1,
                backward[i][j + 1] + 1,
            )

    total = forward[n][m]
    max_edits = max(3, int(math.ceil(max(n, m) * 0.18)))
    if total > max_edits:
        return _SlotAlignment([None] * n, gap_indices=set(range(n)), edit_distance=total, usable=False, method="edit_distance_too_large")

    candidates: list[set[int]] = [set() for _ in range(n)]
    gap_possible = [False] * n
    for i in range(n):
        for j in range(m + 1):
            # Reference character i is deleted (query has no corresponding char).
            if forward[i][j] + 1 + backward[i + 1][j] == total:
                gap_possible[i] = True
            if j < m:
                sub = 0 if reference[i] == query[j] else 1
                if forward[i][j] + sub + backward[i + 1][j + 1] == total:
                    candidates[i].add(j)

    mapping: list[int | None] = [None] * n
    ambiguous: set[int] = set()
    gaps: set[int] = set()
    for i, choices in enumerate(candidates):
        if gap_possible[i]:
            gaps.add(i)
        if len(choices) == 1 and not gap_possible[i]:
            mapping[i] = next(iter(choices))
        else:
            ambiguous.add(i)
    return _SlotAlignment(
        mapping,
        ambiguous_indices=ambiguous,
        gap_indices=gaps,
        edit_distance=total,
        usable=any(index is not None for index in mapping),
        method="optimal_levenshtein",
    )


def _align_variant_to_reference(item: Mapping[str, object], reference_text: str) -> dict:
    text = str(item.get("text") or "")
    alignment = _levenshtein_slot_alignment(reference_text, text)
    scores = list(item.get("character_confidences") or [])
    aggregate = _clamp_confidence(item.get("confidence"))
    slot_chars: list[str] = []
    slot_scores: list[float] = []
    for source_index in alignment.source_indices:
        if source_index is None or source_index >= len(text):
            slot_chars.append("")
            slot_scores.append(0.0)
            continue
        slot_chars.append(text[source_index])
        slot_scores.append(
            _clamp_confidence(scores[source_index])
            if source_index < len(scores) and _clamp_confidence(scores[source_index]) > 0
            else aggregate
        )
    aligned = dict(item)
    aligned.update({
        "slot_characters": slot_chars,
        "slot_confidences": slot_scores,
        "slot_source_indices": list(alignment.source_indices),
        "alignment_ambiguous_indices": set(alignment.ambiguous_indices),
        "alignment_gap_indices": set(alignment.gap_indices),
        "alignment_edit_distance": int(alignment.edit_distance),
        "alignment_usable": bool(alignment.usable),
        "alignment_method": alignment.method,
    })
    return aligned


def _choose_slot_reference(items: Sequence[Mapping[str, object]], slot_count: int) -> Mapping[str, object] | None:
    if not items or slot_count <= 0:
        return None
    primary = items[0]
    if len(str(primary.get("text") or "")) == slot_count:
        return primary
    primary_engine = str(primary.get("engine") or "")
    # Prefer an exact-length result from a genuinely different OCR. It provides
    # a safe slot scaffold but never supplies a missing primary character alone.
    for item in items[1:]:
        if len(str(item.get("text") or "")) == slot_count and str(item.get("engine") or "") != primary_engine:
            return item
    for item in items[1:]:
        if len(str(item.get("text") or "")) == slot_count:
            return item
    return None


def evaluate_selective_column(
    *,
    enabled: bool,
    candidate_text: str,
    raw_confidence: float,
    engine: str,
    fallback_characters: Sequence[str],
    fallback_reasons: Sequence[str] | None = None,
    fallback_sources: Sequence[Iterable[str]] | None = None,
    suspicious_indices: Iterable[int] = (),
    threshold: float = 0.95,
    forbid_apple_automatic: bool = True,
    reject_conflicts: bool = True,
    calibration_db: SelectiveOcrCalibrationDB | None = None,
    candidate_character_confidences: Sequence[float] | None = None,
    variants: Sequence[Mapping[str, object]] | None = None,
    require_stability: bool = False,
    preserve_provisional_ocr: bool = True,
    provisional_floor: float = 0.70,
) -> SelectiveColumnReport:
    threshold = max(0.50, min(0.999, float(threshold or 0.95)))
    provisional_floor = max(0.20, min(threshold, float(provisional_floor or 0.70)))
    fallback = [str(ch or "□")[:1] or "□" for ch in fallback_characters]
    reasons = list(fallback_reasons or [])
    sources = list(fallback_sources or [])
    suspicious = {int(index) for index in suspicious_indices}
    text = clean_column_text(candidate_text)
    effective, samples, lower, profile = calibrated_confidence(
        engine=engine,
        raw_confidence=raw_confidence,
        calibration_db=calibration_db,
    )
    normalised_variants = _normalise_variants(
        variants,
        primary_text=text,
        primary_confidence=raw_confidence,
        primary_engine=engine,
        primary_character_confidences=candidate_character_confidences,
    )
    reference_item = _choose_slot_reference(normalised_variants, len(fallback))
    reference_text = str((reference_item or {}).get("text") or "")
    aligned_variants = [
        _align_variant_to_reference(item, reference_text)
        for item in normalised_variants
    ] if reference_item is not None else []
    primary_aligned = aligned_variants[0] if aligned_variants else None
    usable_evidence = [item for item in aligned_variants if item.get("alignment_usable")]
    ignored_variants = len(normalised_variants) - len(usable_evidence)
    primary_exact = len(text) == len(fallback)
    primary_mapped = bool(primary_aligned and primary_aligned.get("alignment_usable"))
    primary_alignment_ambiguous = set((primary_aligned or {}).get("alignment_ambiguous_indices") or set())
    primary_alignment_gaps = set((primary_aligned or {}).get("alignment_gap_indices") or set())

    # Preserve all independent evidence rows for legacy documents. The current
    # UI only creates one main OCR plus one genuinely different second OCR, but
    # older documents may contain a same-engine stability variant.
    evidence_variants: list[dict] = list(aligned_variants)

    report = SelectiveColumnReport(
        enabled=bool(enabled),
        engine=profile.key,
        candidate_text=text,
        raw_confidence=_clamp_confidence(raw_confidence),
        effective_confidence=effective,
        threshold=threshold,
        exact_length=primary_exact,
        calibration_samples=samples,
        calibration_lower_bound=lower,
        require_stability=bool(require_stability),
        stability_evidence_count=len([item for item in evidence_variants if item.get("alignment_usable")]),
        ignored_variant_count=ignored_variants,
        alignment_reference_text=reference_text,
        alignment_reference_engine=str((reference_item or {}).get("engine") or ""),
        length_recovered_by_alignment=bool(not primary_exact and primary_mapped),
        alignment_ambiguous_indices=sorted(primary_alignment_ambiguous),
        alignment_gap_indices=sorted(primary_alignment_gaps),
        variant_summaries=[
            {
                "label": item.get("label", ""),
                "engine": item.get("engine", ""),
                "text": item.get("text", ""),
                "confidence": item.get("confidence", 0.0),
                "exact_length": len(str(item.get("text") or "")) == len(fallback),
                "alignment_method": item.get("alignment_method", "unavailable"),
                "alignment_edit_distance": int(item.get("alignment_edit_distance") or 0),
                "alignment_ambiguous_indices": sorted(item.get("alignment_ambiguous_indices") or []),
                "alignment_gap_indices": sorted(item.get("alignment_gap_indices") or []),
            }
            for item in aligned_variants
        ] if aligned_variants else [
            {
                "label": item.get("label", ""),
                "engine": item.get("engine", ""),
                "text": item.get("text", ""),
                "confidence": item.get("confidence", 0.0),
                "exact_length": len(str(item.get("text") or "")) == len(fallback),
                "alignment_method": "unavailable",
                "alignment_edit_distance": 0,
                "alignment_ambiguous_indices": [],
                "alignment_gap_indices": [],
            }
            for item in normalised_variants
        ],
    )
    if not enabled:
        report.reason = "disabled"
    elif not text:
        report.reason = "empty_column_ocr"
    elif reference_item is None:
        report.reason = f"length_mismatch:{len(text)}!={len(fallback)}"
    elif not primary_mapped:
        report.reason = "primary_sequence_alignment_failed"
    elif forbid_apple_automatic and profile.key == "apple_vision":
        report.reason = "apple_image_ocr_forbidden_in_automatic_stage"
    elif not profile.automatic_allowed:
        report.reason = "engine_profile_disallows_automatic_output"
    elif not profile.confidence_is_real:
        report.reason = "uncalibrated_engine_confidence"
    elif effective < threshold:
        report.reason = f"confidence_below_threshold:{effective:.4f}<{threshold:.4f}"
    else:
        report.reason = "eligible_aligned" if not primary_exact else "eligible"

    # With only one OCR result, a length mismatch cannot be located safely at
    # character level.  Re-running every box through PKStroke is both slow and
    # less reliable than the existing sequence OCR. Preserve the first OCR text
    # and send the column to manual review as a whole; a genuinely different
    # second OCR may still provide an exact-length alignment scaffold.
    if (
        enabled and profile.key != "unknown" and text
        and all(_valid_ocr_char(ch) for ch in text)
        and not primary_mapped and reference_item is None
        and report.reason.startswith(("length_mismatch", "primary_sequence_alignment_failed"))
    ):
        report.preserve_original_text = True

    globally_eligible = report.reason in {"eligible", "eligible_aligned"}
    primary_slot_chars = list((primary_aligned or {}).get("slot_characters") or [""] * len(fallback))
    primary_slot_scores = list((primary_aligned or {}).get("slot_confidences") or [0.0] * len(fallback))
    primary_source_indices = list((primary_aligned or {}).get("slot_source_indices") or [None] * len(fallback))
    stability_variants = evidence_variants[1:]
    cross_engine_items = [item for item in stability_variants if str(item.get("engine") or "") != profile.key]
    alignment_depends_on_secondary = bool(not primary_exact and reference_item is not None)

    for index, fallback_char in enumerate(fallback):
        ocr_char = primary_slot_chars[index] if index < len(primary_slot_chars) else ""
        fallback_reason = reasons[index] if index < len(reasons) else ""
        source_set = {
            str(value)
            for value in (sources[index] if index < len(sources) else [])
            if str(value)
        }
        raw_char_conf = (
            primary_slot_scores[index]
            if index < len(primary_slot_scores) and primary_slot_scores[index] > 0
            else report.raw_confidence
        )
        char_effective = max(0.0, raw_char_conf - profile.confidence_penalty)
        if report.calibration_lower_bound is not None:
            char_effective = min(char_effective, report.calibration_lower_bound)

        variant_chars: list[str] = []
        secondary_chars: list[str] = []
        support = 0
        cross_engine = False
        secondary_alignment_ambiguous = False
        secondary_alignment_gap = False
        for item in evidence_variants:
            slot_chars = list(item.get("slot_characters") or [])
            ch = slot_chars[index] if index < len(slot_chars) else ""
            variant_chars.append(ch or "∅")
            is_secondary = str(item.get("engine") or "") != profile.key
            if is_secondary:
                secondary_chars.append(ch or "∅")
                if index in set(item.get("alignment_ambiguous_indices") or set()):
                    secondary_alignment_ambiguous = True
                if index in set(item.get("alignment_gap_indices") or set()) or not ch:
                    secondary_alignment_gap = True
            if ch and ch == ocr_char and index not in set(item.get("alignment_ambiguous_indices") or set()):
                support += 1
                if is_secondary:
                    cross_engine = True
                slot_scores = list(item.get("slot_confidences") or [])
                variant_raw = slot_scores[index] if index < len(slot_scores) else float(item.get("confidence") or 0.0)
                variant_profile = engine_profile(str(item.get("engine") or ""))
                if variant_raw > 0 and variant_profile.confidence_is_real:
                    variant_effective = max(0.0, variant_raw - variant_profile.confidence_penalty)
                    char_effective = min(char_effective, variant_effective)
        stability_total = len(evidence_variants)
        stability_ratio = support / stability_total if stability_total else 0.0

        char_samples = char_correct = char_errors = 0
        char_lower: float | None = None
        if calibration_db is not None and ocr_char:
            try:
                char_samples, char_correct, char_errors, char_lower = calibration_db.estimate_character(
                    engine=profile.key,
                    predicted_character=ocr_char,
                    raw_confidence=raw_char_conf,
                )
            except Exception:
                char_samples = char_correct = char_errors = 0
                char_lower = None
        if char_lower is not None:
            char_effective = min(char_effective, char_lower)

        source_index = primary_source_indices[index] if index < len(primary_source_indices) else None
        evidence = SelectiveCharEvidence(
            index=index,
            ocr_char=ocr_char,
            fallback_char=fallback_char,
            output_char=fallback_char,
            score=char_effective,
            raw_character_confidence=raw_char_conf,
            effective_character_confidence=char_effective,
            stability_support=support,
            stability_total=stability_total,
            stability_ratio=stability_ratio,
            variant_characters=variant_chars,
            cross_engine_support=cross_engine,
            character_calibration_samples=char_samples,
            character_calibration_lower_bound=char_lower,
            primary_source_index=source_index,
            alignment_status=(
                "ambiguous" if index in primary_alignment_ambiguous
                else "gap" if index in primary_alignment_gaps or source_index is None
                else "aligned" if not primary_exact
                else "direct"
            ),
            secondary_characters=secondary_chars,
        )
        if report.preserve_original_text:
            evidence.reason = "column_length_mismatch_preserve_primary_for_manual"
            evidence.automatic_fallback = False
        elif not globally_eligible:
            # A valid, position-aligned OCR character below the strict trust
            # threshold is still more useful than synthesising a fake
            # handwriting trace. Keep it as provisional text, visibly mark it
            # for review, and do not invoke PKStroke automatically.
            provisional_reason = (
                report.reason in {
                    "engine_profile_disallows_automatic_output",
                    "uncalibrated_engine_confidence",
                }
                or report.reason.startswith("confidence_below_threshold")
            )
            can_preserve = bool(
                preserve_provisional_ocr
                and provisional_reason
                and profile.key != "unknown"
                and primary_mapped
                and index not in primary_alignment_ambiguous
                and index not in primary_alignment_gaps
                and source_index is not None
                and index not in suspicious
                and _valid_ocr_char(ocr_char)
                and raw_char_conf >= provisional_floor
            )
            if can_preserve:
                evidence.provisional = True
                evidence.automatic_fallback = False
                evidence.output_char = ocr_char
                evidence.reason = report.reason + ";provisional_preserved_no_pkstroke"
            else:
                evidence.reason = report.reason
        elif index in primary_alignment_ambiguous:
            evidence.reason = "primary_sequence_alignment_ambiguous"
        elif index in primary_alignment_gaps or source_index is None or not ocr_char:
            evidence.reason = "primary_sequence_alignment_gap"
        elif index in suspicious:
            evidence.reason = "suspicious_glyph_box"
        elif not _valid_ocr_char(ocr_char):
            evidence.reason = "invalid_ocr_character"
        elif char_effective < threshold:
            if preserve_provisional_ocr and raw_char_conf >= provisional_floor:
                evidence.provisional = True
                evidence.automatic_fallback = False
                evidence.output_char = ocr_char
                evidence.reason = (
                    f"character_confidence_below_threshold:{char_effective:.4f}<{threshold:.4f};"
                    "provisional_preserved_no_pkstroke"
                )
            else:
                evidence.reason = f"character_confidence_below_threshold:{char_effective:.4f}<{threshold:.4f}"
        elif char_errors >= 2 and char_correct < max(3, char_errors * 3):
            evidence.reason = "historical_high_confidence_character_errors"
        elif require_stability and not stability_variants:
            # A selected second OCR may fail or return no usable sequence.  The
            # old behaviour sent every otherwise valid main-OCR character to
            # three-view rescue and then PKStroke, turning one page into hundreds
            # of native requests. Preserve the aligned main OCR as provisional
            # text and flag the column for review instead; true gaps/invalid
            # characters still use automatic per-glyph fallback.
            if (
                preserve_provisional_ocr
                and _valid_ocr_char(ocr_char)
                and source_index is not None
                and index not in primary_alignment_ambiguous
                and index not in primary_alignment_gaps
                and index not in suspicious
                and raw_char_conf >= provisional_floor
            ):
                evidence.provisional = True
                evidence.automatic_fallback = False
                evidence.output_char = ocr_char
                evidence.reason = "stability_evidence_missing;provisional_preserved_no_pkstroke"
            else:
                evidence.reason = "stability_evidence_missing"
        elif require_stability and secondary_alignment_ambiguous:
            evidence.reason = "secondary_sequence_alignment_ambiguous"
            evidence.automatic_fallback = False
            report.unstable_indices.append(index)
        elif require_stability and secondary_alignment_gap:
            evidence.reason = "secondary_sequence_alignment_gap"
            evidence.automatic_fallback = False
            report.unstable_indices.append(index)
        elif require_stability and support != stability_total:
            evidence.reason = "stability_variant_disagrees"
            evidence.automatic_fallback = False
            report.unstable_indices.append(index)
        elif alignment_depends_on_secondary and not cross_engine:
            evidence.reason = "stability_variant_disagrees"
            evidence.automatic_fallback = False
            report.unstable_indices.append(index)
        elif fallback_reason in {
            "glyph_memory_exact_short_circuit_no_apple",
            "glyph_memory_strong_similar_short_circuit_no_apple",
        }:
            evidence.reason = "glyph_memory_kept"
        elif fallback_char == "□":
            evidence.accepted = True
            evidence.output_char = ocr_char
            if cross_engine:
                evidence.reason = (
                    "cross_engine_aligned_consensus_fills_unresolved"
                    if not primary_exact else "cross_engine_consensus_fills_unresolved"
                )
            else:
                evidence.reason = "high_confidence_column_ocr_fills_unresolved"
        elif fallback_char == ocr_char:
            evidence.accepted = True
            evidence.output_char = ocr_char
            evidence.score = min(0.999, max(char_effective, char_effective + 0.015))
            evidence.reason = "column_ocr_and_glyph_fallback_agree"
        else:
            evidence.conflict = True
            if reject_conflicts:
                evidence.output_char = "□"
                evidence.reason = "column_ocr_glyph_conflict_rejected"
            else:
                evidence.output_char = fallback_char
                evidence.reason = "column_ocr_glyph_conflict_keep_fallback"

        if evidence.accepted:
            evidence.automatic_fallback = False
            report.accepted_indices.append(index)
        else:
            report.rejected_indices.append(index)
        if evidence.provisional:
            report.provisional_indices.append(index)
        if evidence.automatic_fallback:
            report.automatic_fallback_indices.append(index)
        elif not evidence.accepted and not evidence.provisional:
            report.manual_only_indices.append(index)
        if evidence.conflict:
            report.conflict_indices.append(index)
        report.decisions.append(evidence)
    return report

def record_review_calibration(
    *,
    metadata: dict,
    old_text: str,
    new_text: str,
    glyph_confirmations: Sequence[dict] | None = None,
    calibration_db: SelectiveOcrCalibrationDB | None = None,
) -> int:
    """Learn whether previously accepted OCR characters were correct."""
    del old_text
    info = dict((metadata or {}).get("handwriting_selective_ocr") or {})
    if not info or info.get("calibration_recorded"):
        return 0
    candidate = clean_column_text(info.get("candidate_text", ""))
    decisions = {
        int(item.get("index")): dict(item)
        for item in (info.get("character_decisions") or [])
        if isinstance(item, Mapping) and str(item.get("index", "")).lstrip("-").isdigit()
    }
    if not candidate and not decisions:
        return 0
    accepted = {
        int(index)
        for index in info.get("accepted_indices", [])
        if str(index).lstrip("-").isdigit() and int(index) >= 0
    }
    if not accepted:
        return 0
    engine = str(info.get("engine") or "unknown")
    raw = float(info.get("raw_confidence") or 0.0)
    per_char_raw = [
        _clamp_confidence(value)
        for value in (info.get("candidate_character_confidences") or [])
    ]
    confirmed_map: dict[int, str] = {}
    for item in glyph_confirmations or []:
        try:
            index = int(item.get("index"))
            character = str(item.get("character") or item.get("text") or "")[:1]
        except Exception:
            continue
        if character:
            confirmed_map[index] = character
    final_chars = list(clean_column_text(new_text))
    if not confirmed_map and len(final_chars) == len(candidate):
        confirmed_map = {index: final_chars[index] for index in accepted if index < len(final_chars)}
    db = calibration_db or SelectiveOcrCalibrationDB()
    recorded = 0
    for index in sorted(accepted):
        if index not in confirmed_map:
            continue
        decision = decisions.get(index, {})
        predicted = str(decision.get("ocr_char") or "")[:1]
        if not predicted and index < len(candidate):
            predicted = candidate[index]
        if not predicted:
            continue
        decision_raw = _clamp_confidence(decision.get("raw_character_confidence"))
        raw_score = (
            decision_raw
            if decision_raw > 0
            else per_char_raw[index] if index < len(per_char_raw) and per_char_raw[index] > 0
            else raw
        )
        if db.record(
            engine=engine,
            raw_confidence=raw_score,
            predicted_character=predicted,
            confirmed_character=confirmed_map[index],
        ):
            recorded += 1
    return recorded
