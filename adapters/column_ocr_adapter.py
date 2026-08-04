#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixed-region vertical-column OCR for Japanese novel pages.

This adapter intentionally does *not* ask a layout model to discover text boxes.
The common OCR layer first applies the user's fixed normalized crop rectangle to
all pages.  We then detect complete physical vertical columns with deterministic connected-component geometry, isolate every column with a white mask at the original pixel size, and pass
those masked page images directly to one selected OCR engine.

The design goal is omission safety:

* fixed body region removes headers, page numbers, and illustrations before OCR;
* connected-component splitting works without optional OpenCV/SciPy packages and
  preserves every detected physical column as a masked recognition target;
* target glyph pixels are never stretched or resampled; other columns are white-masked;
* failed/empty column recognition is retried with a wider masked target;
* strict mode preserves every detected column; an unresolved OCR column becomes a visible manual-review placeholder instead of aborting the whole book.
"""
from __future__ import annotations

import math
import os
import json
import inspect
import re
import hashlib
import tempfile
import threading
from collections import OrderedDict
from difflib import SequenceMatcher
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageOps

from adapters.ocr_engine_common import is_spurious_ocr_item, run_ocr_engine
from adapters.ocr_recognition_bridge import (
    RECOGNITION_ENGINES,
    AppleVisionRecognitionSession,
    recognizer_iterator,
)
from adapters.image_preprocess import build_fallback_variant, build_fallback_variants, close_variants
from adapters.column_image_cleanup import cleanup_column_image, normalise_ruby_strength
from engine.column_sentence_reflow import (
    has_sentence_terminal,
    is_provisional_quote_terminal,
    join_column_parts,
    normalize_column_text,
    starts_post_quote_continuation,
)

# Compatibility hook for tests/plugins that monkeypatch the recognizer.
_recognizer_iterator = recognizer_iterator

PhaseCallback = Callable[[str, int, int, str], None]

SUPPORTED_RECOGNIZERS = {
    key: value for key, value in RECOGNITION_ENGINES.items() if key != "native"
}

COLUMN_DETECTOR_VERSION = "component-geometry-v11-short-column-full-coverage-lossless-body-pixels-v12-no-projection"
LEGACY_PROJECTION_DETECTOR_VERSION = "review-only-projection-v5-periodic-grid"

_SHARED_PREPARE_LOCK_GUARD = threading.Lock()
_SHARED_PREPARE_LOCKS: dict[str, threading.Lock] = {}

# SHA verification remains mandatory, but the same immutable run-local column
# files are validated by model 1/2/3 in succession.  Re-reading every PNG for
# every model wastes disk bandwidth.  This bounded memo stores only the digest
# of a file whose path/size/mtime tuple is unchanged; it is cleared at the start
# and end of each GUI OCR run and never becomes a cross-session content cache.
_TASK_SHA_MEMO_LOCK = threading.Lock()
_TASK_SHA_MEMO: OrderedDict[tuple[str, int, int], str] = OrderedDict()
_TASK_SHA_MEMO_LIMIT = 16384


def clear_task_file_sha_memo() -> None:
    with _TASK_SHA_MEMO_LOCK:
        _TASK_SHA_MEMO.clear()


def task_file_sha_memo_metrics() -> dict[str, int]:
    with _TASK_SHA_MEMO_LOCK:
        return {"entries": len(_TASK_SHA_MEMO), "limit": _TASK_SHA_MEMO_LIMIT}


def _file_sha256(path: str | Path) -> str:
    target = Path(path)
    stat = target.stat()
    key = (str(target.resolve()), int(stat.st_size), int(stat.st_mtime_ns))
    with _TASK_SHA_MEMO_LOCK:
        cached = _TASK_SHA_MEMO.get(key)
        if cached is not None:
            _TASK_SHA_MEMO.move_to_end(key)
            return cached
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    with _TASK_SHA_MEMO_LOCK:
        _TASK_SHA_MEMO[key] = value
        _TASK_SHA_MEMO.move_to_end(key)
        while len(_TASK_SHA_MEMO) > _TASK_SHA_MEMO_LIMIT:
            _TASK_SHA_MEMO.popitem(last=False)
    return value


def _shared_prepare_lock(cache_dir: Path, page_index: int) -> threading.Lock:
    key = str((cache_dir / f"p{page_index:05d}_columns.json").resolve())
    with _SHARED_PREPARE_LOCK_GUARD:
        return _SHARED_PREPARE_LOCKS.setdefault(key, threading.Lock())


def _normalise_column_input_profile(value: object) -> str:
    if value is None or not str(value).strip():
        # Direct adapters/plugins that predate profiles must keep the exact v8
        # call semantics: caller values are honoured and Apple Vision alone
        # forces smart crop when the option is omitted.
        return "v8_legacy"
    profile = str(value).strip().lower().replace("-", "_")
    if profile == "custom":
        return "custom"
    if profile in {"v8", "v8_exact", "v8_gui", "v8_compat", "v8_compatible"}:
        return "v8_exact"
    if profile in {"v8_legacy", "legacy", "legacy_v8", "direct_v8"}:
        return "v8_legacy"
    # Unknown persisted/plugin values fail closed to v8 legacy behavior rather
    # than silently enabling destructive custom preprocessing.
    return "v8_legacy"


def _column_input_contract_descriptor(
    *,
    recognition_engine: str,
    detector_mode: str,
    sensitivity: int,
    padding_percent: int,
    max_columns: int,
    fixed_region_rect: Sequence[float] | None,
    compact_transport: bool,
    auto_filter_ruby: bool,
    filter_fragments: bool,
    smart_crop: bool,
    ruby_strength: str,
    preserve_body_pixels: bool,
    input_profile: str,
) -> tuple[dict[str, object], str]:
    payload: dict[str, object] = {
        "cache_schema": "column-input-contract-v4-engine-independent",
        # The recognizer never participates in physical-column detection or
        # pixel generation.  Excluding it from the namespace lets models with
        # the same *effective* image contract reuse byte-identical run-local
        # crops, while compact/smart/ruby differences still receive isolated
        # directories through the remaining fields below.
        "detector_version": column_detector_version(detector_mode),
        "detector_mode": _normalise_column_detector_mode(detector_mode),
        "sensitivity": int(sensitivity),
        "padding_percent": int(padding_percent),
        "max_columns": int(max_columns),
        "fixed_region_rect": [float(v) for v in (fixed_region_rect or [])],
        "compact_transport": bool(compact_transport),
        "auto_filter_ruby": bool(auto_filter_ruby),
        "filter_fragments": bool(filter_fragments),
        "smart_crop": bool(smart_crop),
        "ruby_strength": normalise_ruby_strength(ruby_strength),
        "preserve_body_pixels": bool(preserve_body_pixels),
        "input_profile": _normalise_column_input_profile(input_profile),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return payload, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _shared_prepare_profile_dir(
    base_dir: Path,
    *,
    recognition_engine: str,
    detector_mode: str,
    sensitivity: int,
    padding_percent: int,
    max_columns: int,
    fixed_region_rect: Sequence[float] | None,
    compact_transport: bool,
    auto_filter_ruby: bool,
    filter_fragments: bool,
    smart_crop: bool,
    ruby_strength: str,
    preserve_body_pixels: bool,
    input_profile: str,
) -> tuple[Path, str]:
    """Return an immutable cache namespace for one effective OCR input contract.

    Namespacing is based on the complete *pixel-generation* contract rather
    than the recognizer name.  Two models therefore share one immutable crop
    directory only when every detector/crop/cleanup option is identical.
    Compact transport, smart crop, Ruby cleanup and all other pixel-affecting
    fields remain in the fingerprint, so incompatible model inputs can never
    overwrite one another during parallel first-round OCR.
    """
    payload, fingerprint = _column_input_contract_descriptor(
        recognition_engine=recognition_engine,
        detector_mode=detector_mode,
        sensitivity=sensitivity,
        padding_percent=padding_percent,
        max_columns=max_columns,
        fixed_region_rect=fixed_region_rect,
        compact_transport=compact_transport,
        auto_filter_ruby=auto_filter_ruby,
        filter_fragments=filter_fragments,
        smart_crop=smart_crop,
        ruby_strength=ruby_strength,
        preserve_body_pixels=preserve_body_pixels,
        input_profile=input_profile,
    )
    target = base_dir / f"input_{fingerprint}"
    target.mkdir(parents=True, exist_ok=True)
    contract_path = target / "input_contract.json"
    contract_lock = _shared_prepare_lock(target, 0)
    with contract_lock:
        expected_contract = {**payload, "fingerprint": fingerprint}
        if contract_path.exists():
            try:
                current = json.loads(contract_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise RuntimeError(f"OCR 输入合同缓存损坏: {contract_path}") from exc
            if current != expected_contract:
                raise RuntimeError(f"OCR 输入合同缓存内容与目录指纹不一致: {contract_path}")
        else:
            tmp = contract_path.with_name(
                f".{contract_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
            )
            try:
                tmp.write_text(
                    json.dumps(expected_contract, ensure_ascii=False, sort_keys=True),
                    encoding="utf-8",
                )
                os.replace(tmp, contract_path)
            finally:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
    return target, fingerprint


@dataclass(frozen=True)
class DetectedColumn:
    """One physical vertical text column in page pixel coordinates."""

    left: int
    top: int
    right: int
    bottom: int
    hard_left: int
    hard_right: int
    ink_score: float
    content_spans: tuple[tuple[int, int], ...] = ()
    estimated_chars: int = 0
    full_height_slot: bool = False
    supplemental_boxes: tuple[tuple[int, int, int, int], ...] = ()
    excluded_boxes: tuple[tuple[int, int, int, int], ...] = ()

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    def polygon(self) -> list[list[int]]:
        return [
            [self.left, self.top],
            [self.right, self.top],
            [self.right, self.bottom],
            [self.left, self.bottom],
        ]


@dataclass
class _Run:
    start: int
    end: int  # exclusive
    score: float

    @property
    def width(self) -> int:
        return max(0, self.end - self.start)


def _otsu_threshold(histogram: list[int]) -> int:
    """Return an Otsu grayscale threshold without OpenCV/numpy dependencies."""
    total = sum(histogram)
    if total <= 0:
        return 180
    weighted_total = sum(index * count for index, count in enumerate(histogram))
    background_weight = 0
    background_sum = 0.0
    best_variance = -1.0
    best_threshold = 180
    for threshold, count in enumerate(histogram):
        background_weight += count
        if background_weight <= 0:
            continue
        foreground_weight = total - background_weight
        if foreground_weight <= 0:
            break
        background_sum += threshold * count
        background_mean = background_sum / background_weight
        foreground_mean = (weighted_total - background_sum) / foreground_weight
        variance = background_weight * foreground_weight * (background_mean - foreground_mean) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold
    return int(best_threshold)


def _smooth(values: list[float], radius: int) -> list[float]:
    if radius <= 0 or len(values) < 3:
        return list(values)
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + value)
    out: list[float] = []
    for index in range(len(values)):
        left = max(0, index - radius)
        right = min(len(values), index + radius + 1)
        out.append((prefix[right] - prefix[left]) / max(1, right - left))
    return out


def _fill_short_false_gaps(active: list[bool], max_gap: int) -> list[bool]:
    out = list(active)
    index = 0
    while index < len(out):
        if out[index]:
            index += 1
            continue
        start = index
        while index < len(out) and not out[index]:
            index += 1
        end = index
        if start > 0 and end < len(out) and (end - start) <= max_gap:
            for pos in range(start, end):
                out[pos] = True
    return out


def _runs_from_active(active: list[bool], projection: list[float]) -> list[_Run]:
    runs: list[_Run] = []
    index = 0
    while index < len(active):
        if not active[index]:
            index += 1
            continue
        start = index
        while index < len(active) and active[index]:
            index += 1
        end = index
        runs.append(_Run(start, end, sum(projection[start:end])))
    return runs


def _percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 1.0
    ordered = sorted(values)
    pos = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def _dedupe_overlapping_runs(runs: list[_Run]) -> list[_Run]:
    """Remove duplicate projections without ever joining adjacent text columns.

    The former implementation joined every nearby narrow run to its neighbour.
    That made a ruby strip part of the body rectangle and, on pages with a narrow
    glyph column, could join two real physical columns into one wide box.  Only
    genuinely overlapping detections are duplicates; whitespace-separated runs
    must remain independent until ruby classification has finished.
    """
    ordered = sorted((run for run in runs if run.width > 0), key=lambda run: run.start)
    if len(ordered) < 2:
        return ordered
    result: list[_Run] = []
    for run in ordered:
        if not result or run.start >= result[-1].end:
            result.append(run)
            continue
        previous = result[-1]
        overlap = min(previous.end, run.end) - max(previous.start, run.start)
        smaller = max(1, min(previous.width, run.width))
        if overlap / smaller >= 0.55:
            # Core/full projections can describe the same physical band with
            # slightly different edges.  Keep the stronger evidence rather than
            # broadening it and accidentally reopening ruby pixels.
            previous_strength = previous.score * max(1.0, previous.width ** 0.5)
            run_strength = run.score * max(1.0, run.width ** 0.5)
            result[-1] = previous if previous_strength >= run_strength else run
        else:
            # Rare partial overlap caused by antialiasing: split at the midpoint
            # so both physical bands survive but never overlap one another.
            boundary = round((previous.end + run.start) / 2)
            boundary = max(previous.start + 1, min(run.end - 1, boundary))
            result[-1] = _Run(previous.start, boundary, previous.score)
            result.append(_Run(boundary, run.end, run.score))
    return result


def _run_metrics(mask: Image.Image, run: _Run) -> dict:
    top, bottom = _vertical_ink_bounds(mask, run.start, run.end)
    span = max(1, bottom - top)
    return {
        "run": run,
        "center": _run_center(run),
        "width": max(1, run.width),
        "score": max(0.0, float(run.score)),
        "top": top,
        "bottom": bottom,
        "span": span,
    }


def _split_overwide_runs(mask: Image.Image, runs: list[_Run]) -> list[_Run]:
    """Split only bands that are much wider than the page's normal body column.

    A low projection threshold can connect touching ruby to its base column, or
    connect two body columns through a dust/scan bridge.  Re-thresholding *inside*
    an overwide run at a higher local level reveals the independent dense bands
    without changing any source pixels.
    """
    if not runs:
        return []
    metrics = [_run_metrics(mask, run) for run in runs]
    long_widths = [
        item["width"] for item in metrics
        if item["span"] >= max(mask.height * 0.10, item["width"] * 2.2)
    ]
    widths = long_widths or [item["width"] for item in metrics]
    measured_typical = max(2.0, _percentile([int(value) for value in widths], 0.62))
    # A single connected projection has no peer from which to estimate normal
    # column width.  Cap the estimate by a conservative page-width prior so a
    # dust bridge joining two physical columns can still be split.  The later
    # long-span/visible-valley checks prevent a short bold title glyph from being
    # broken into strokes.
    page_prior = max(3.0, mask.width * 0.035)
    typical = min(measured_typical, page_prior) if len(runs) == 1 else measured_typical
    projection = _column_projection(mask, core_only=False)
    output: list[_Run] = []
    for run in runs:
        if run.width <= max(typical * 1.48, typical + 4):
            output.append(run)
            continue
        local = projection[run.start:run.end]
        peak = max(local or [0.0])
        if peak <= 0:
            output.append(run)
            continue
        gate = max(1.0, peak * 0.20)
        active = _fill_short_false_gaps(
            [value >= gate for value in local],
            max_gap=max(1, round(typical * 0.08)),
        )
        pieces = _runs_from_active(active, local)
        absolute = [
            _Run(run.start + piece.start, run.start + piece.end, piece.score)
            for piece in pieces
            if piece.width >= max(2, round(typical * 0.18))
        ]
        # A real split needs at least two long dense bands and a visible
        # whitespace valley.  This prevents a short bold title/kanji from being
        # split into its internal strokes while still separating body columns
        # joined by a one-pixel scan bridge.
        long_pieces = []
        for piece in absolute:
            piece_top, piece_bottom = _vertical_ink_bounds(mask, piece.start, piece.end)
            if piece_bottom - piece_top >= max(mask.height * 0.055, piece.width * 2.0):
                long_pieces.append(piece)
        visible_valleys = [
            right.start - left.end
            for left, right in zip(long_pieces, long_pieces[1:])
        ]
        if (
            len(long_pieces) >= 2
            and any(gap >= max(2, round(typical * 0.10)) for gap in visible_valleys)
        ):
            output.extend(long_pieces)
        else:
            output.append(run)
    return _dedupe_overlapping_runs(output)


def _normalised_projection_correlation(values: list[float], lag: int) -> float:
    """Return a stable zero-mean autocorrelation for one horizontal lag."""
    if lag <= 0 or lag >= len(values) - 2:
        return 0.0
    left = values[:-lag]
    right = values[lag:]
    if len(left) < 8:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = 0.0
    left_energy = 0.0
    right_energy = 0.0
    for a, b in zip(left, right):
        da = float(a) - left_mean
        db = float(b) - right_mean
        numerator += da * db
        left_energy += da * da
        right_energy += db * db
    denominator = math.sqrt(left_energy * right_energy)
    return numerator / denominator if denominator > 1e-9 else 0.0


def _estimate_periodic_column_pitch(mask: Image.Image) -> tuple[float, float]:
    """Estimate the fundamental body-column pitch from the whole page.

    Furigana can fill every x-gap between neighbouring body columns.  In that
    case connected-component/projection runs become sentence-sized wide bands,
    but the *body columns themselves* still repeat at a very regular horizontal
    interval.  Autocorrelation recovers that interval even when there is no
    visible whitespace valley to split on.

    The smallest strong local peak is preferred over its 2x/3x harmonics.  This
    is important on ordinary novels where alternating long/short columns often
    make the second harmonic numerically stronger than the true pitch.
    """
    projection = _smooth(_column_projection(mask, core_only=True), radius=1)
    width = len(projection)
    if width < 48 or max(projection or [0.0]) <= 0:
        return 0.0, 0.0

    min_pitch = max(7, round(width * 0.018))
    max_pitch = min(
        max(min_pitch + 2, round(width * 0.145)),
        max(min_pitch + 2, round(mask.height * 0.18)),
        260,
    )
    if max_pitch <= min_pitch:
        return 0.0, 0.0

    correlations = {
        lag: _normalised_projection_correlation(projection, lag)
        for lag in range(min_pitch, max_pitch + 1)
    }
    positive = [(score, lag) for lag, score in correlations.items() if score > 0]
    if not positive:
        return 0.0, 0.0
    best_score, _best_lag = max(positive)
    required = max(0.16, best_score * 0.56)
    local_peaks: list[tuple[int, float]] = []
    for lag in range(min_pitch + 1, max_pitch):
        score = correlations.get(lag, 0.0)
        if (
            score >= required
            and score >= correlations.get(lag - 1, score)
            and score >= correlations.get(lag + 1, score)
        ):
            local_peaks.append((lag, score))
    if not local_peaks:
        return 0.0, 0.0

    # Ignore an implausibly tiny stroke-period peak when a nearby 2x peak is
    # markedly stronger.  Normal vertical body columns generally occupy at
    # least ~2.5% of the fixed body width centre-to-centre.
    plausible_floor = max(min_pitch, round(width * 0.024))
    plausible = [item for item in local_peaks if item[0] >= plausible_floor]
    candidates = plausible or local_peaks
    pitch, score = min(candidates, key=lambda item: item[0])
    return float(pitch), float(score)


def _best_grid_phase(projection: list[float], pitch: float) -> float:
    """Choose the phase whose repeated body-width windows contain most ink."""
    step = max(2, int(round(pitch)))
    half = max(2, int(round(pitch * 0.29)))
    prefix = [0.0]
    for value in projection:
        prefix.append(prefix[-1] + max(0.0, float(value)))

    def window_mass(center: int) -> float:
        left = max(0, center - half)
        right = min(len(projection), center + half + 1)
        return prefix[right] - prefix[left]

    best_phase = 0
    best_score = -1.0
    for phase in range(step):
        masses = [window_mass(center) for center in range(phase, len(projection), step)]
        # The strongest slots establish body alignment; summing every weak edge
        # slot would bias the phase toward page shadows or running furniture.
        masses.sort(reverse=True)
        keep = max(4, round(len(masses) * 0.72))
        score = sum(masses[:keep])
        if score > best_score:
            best_score = score
            best_phase = phase
    return float(best_phase)


def _slot_body_run(
    mask: Image.Image,
    projection: list[float],
    *,
    center: float,
    pitch: float,
) -> _Run | None:
    """Extract only the dense body-glyph band from one periodic column slot."""
    slot_left = max(0, int(math.floor(center - pitch * 0.45)))
    slot_right = min(mask.width, int(math.ceil(center + pitch * 0.45)))
    if slot_right - slot_left < 3:
        return None
    local = projection[slot_left:slot_right]
    if not local or max(local) <= 0:
        return None

    # A body glyph normally occupies roughly half of the column pitch.  Search
    # for the maximum-mass body-sized window rather than thresholding the whole
    # slot; this prevents a nearby narrow ruby strip from broadening the box.
    target = max(3, min(len(local), int(round(pitch * 0.58))))
    prefix = [0.0]
    for value in local:
        prefix.append(prefix[-1] + max(0.0, float(value)))
    best_start = 0
    best_mass = -1.0
    for start in range(0, len(local) - target + 1):
        mass = prefix[start + target] - prefix[start]
        if mass > best_mass:
            best_mass = mass
            best_start = start
    if best_mass <= 0:
        return None

    left = slot_left + best_start
    right = left + target
    window = projection[left:right]
    peak = max(window or [0.0])
    gate = max(0.35, peak * 0.055)
    active = [value >= gate for value in window]
    active = _fill_short_false_gaps(active, max_gap=1)
    pieces = _runs_from_active(active, window)
    if pieces:
        # Keep the strongest body core but allow detached strokes inside the
        # target window by spanning all pieces near that core.
        strongest = max(pieces, key=lambda piece: piece.score)
        nearby = [
            piece for piece in pieces
            if piece.end >= strongest.start - max(1, round(pitch * 0.10))
            and piece.start <= strongest.end + max(1, round(pitch * 0.10))
        ]
        if nearby:
            left = left + min(piece.start for piece in nearby)
            right = left + (
                max(piece.end for piece in nearby) - min(piece.start for piece in nearby)
            )

    # Enforce a strict body-only maximum.  Ruby lives in the remaining side
    # space of the pitch slot and can no longer be exposed to any OCR engine.
    max_width = max(4, int(round(pitch * 0.68)))
    if right - left > max_width:
        centre = (left + right) / 2.0
        left = max(slot_left, int(round(centre - max_width / 2.0)))
        right = min(slot_right, left + max_width)
    top, bottom = _vertical_ink_bounds(mask, left, right)
    span = bottom - top
    mass = sum(projection[left:right])
    if span < max(5.0, pitch * 0.80, mask.height * 0.010):
        return None
    if mass < max(5.0, mask.height * 0.008):
        return None
    return _Run(left, right, mass)


def _split_connected_runs_by_periodic_grid(mask: Image.Image, runs: list[_Run]) -> list[_Run]:
    """Split ruby-bridged wide bands into one run per physical text column.

    This is a second, independent segmentation route used only when ordinary
    whitespace-valley splitting leaves bands wider than one physical column.
    It never merges existing runs and is therefore safe for strict-mode OCR.
    """
    ordered = _dedupe_overlapping_runs(runs)
    if not ordered:
        return []
    pitch, confidence = _estimate_periodic_column_pitch(mask)
    if pitch <= 0 or confidence < 0.16:
        return ordered
    if not any(run.width >= max(pitch * 1.42, pitch + 4) for run in ordered):
        return ordered

    projection = _smooth(_column_projection(mask, core_only=False), radius=1)
    phase = _best_grid_phase(projection, pitch)
    replacement: list[_Run] = []
    split_count = 0
    for run in ordered:
        if run.width < max(pitch * 1.42, pitch + 4):
            replacement.append(run)
            continue
        first_index = math.ceil((run.start - phase - pitch * 0.36) / pitch)
        last_index = math.floor((run.end - phase + pitch * 0.36) / pitch)
        candidates: list[_Run] = []
        for grid_index in range(first_index, last_index + 1):
            center = phase + grid_index * pitch
            candidate = _slot_body_run(mask, projection, center=center, pitch=pitch)
            if candidate is None:
                continue
            # The selected body core must materially overlap the original wide
            # connected band; this blocks unrelated margin/header slots.
            overlap = min(candidate.end, run.end) - max(candidate.start, run.start)
            if overlap >= max(2, round(candidate.width * 0.45)):
                candidates.append(candidate)
        candidates = _dedupe_overlapping_runs(candidates)
        expected = max(2, int(round(run.width / pitch)))
        if len(candidates) >= 2 and abs(len(candidates) - expected) <= 2:
            replacement.extend(candidates)
            split_count += 1
        else:
            replacement.append(run)

    result = _dedupe_overlapping_runs(replacement)
    # A periodic rescue is accepted only when it actually increases physical
    # column resolution and does not explode into tiny ruby/noise slots.
    if split_count <= 0 or len(result) <= len(ordered) or len(result) > max(80, len(ordered) * 5):
        return ordered
    return result


def _filter_ruby_side_runs(
    mask: Image.Image,
    runs: list[_Run],
) -> tuple[list[_Run], list[_Run]]:
    """Separate body columns from narrow furigana/side-note projections.

    Ruby is identified geometrically before any OCR image is produced.  The
    classifier intentionally requires a nearby stronger body column, so a true
    one/two-character physical column at the normal column pitch remains intact.
    """
    ordered = _dedupe_overlapping_runs(runs)
    if len(ordered) < 2:
        return ordered, []
    metrics = [_run_metrics(mask, run) for run in ordered]
    substantial = [
        item for item in metrics
        if item["span"] >= max(mask.height * 0.085, item["width"] * 2.0)
    ]
    reference = substantial or metrics
    typical_width = max(
        2.0,
        _percentile([int(item["width"]) for item in reference], 0.68),
    )
    anchors = [
        item for item in metrics
        if item["width"] >= typical_width * 0.56
        and item["span"] >= max(mask.height * 0.050, item["width"] * 1.8)
    ]
    if not anchors:
        return ordered, []
    pitch = _estimate_column_pitch([item["run"] for item in anchors])
    anchor_spans = [int(item["span"]) for item in anchors]
    typical_span = max(1.0, _percentile(anchor_spans, 0.55))

    kept: list[_Run] = []
    excluded: list[_Run] = []
    for item in metrics:
        run = item["run"]
        if item in anchors:
            kept.append(run)
            continue
        nearest = min(anchors, key=lambda anchor: abs(anchor["center"] - item["center"]))
        distance = abs(nearest["center"] - item["center"])
        near_body = distance <= max(typical_width * 1.05, pitch * 0.40 if pitch > 0 else 0.0)
        narrow = item["width"] <= typical_width * 0.64
        weaker = item["score"] <= max(12.0, nearest["score"] * 0.76)
        very_weak = item["score"] <= max(8.0, nearest["score"] * 0.22)
        shorter = item["span"] <= max(typical_span * 0.78, mask.height * 0.24)
        # A normal physical column sits roughly one pitch away.  It must never be
        # deleted merely because it contains only punctuation or a few glyphs.
        normal_pitch_slot = bool(pitch > 0 and distance >= pitch * 0.62)
        ruby_like = (
            near_body and narrow and weaker and (shorter or very_weak)
            and not normal_pitch_slot
        )
        if ruby_like:
            excluded.append(run)
        else:
            kept.append(run)
    # Safety: never allow a heuristic to remove most of the page.
    if len(kept) < max(1, round(len(ordered) * 0.45)):
        return ordered, []
    return _dedupe_overlapping_runs(kept), excluded


def _refine_body_run(mask: Image.Image, run: _Run, typical_width: float) -> _Run:
    """Keep the densest body-width core when ruby touches the main projection."""
    projection = _column_projection(mask, core_only=False)
    local = projection[run.start:run.end]
    if not local:
        return run
    top, bottom = _vertical_ink_bounds(mask, run.start, run.end)
    long_enough = (bottom - top) >= max(mask.height * 0.10, typical_width * 2.8)
    peak = max(local or [0.0])
    if long_enough and peak > 0:
        active = [value >= max(2.0, peak * 0.16) for value in local]
        active = _fill_short_false_gaps(active, max_gap=1)
        dense = _runs_from_active(active, local)
        peak_index = max(range(len(local)), key=lambda index: local[index])
        containing = [piece for piece in dense if piece.start <= peak_index < piece.end]
        if containing:
            core = max(containing, key=lambda piece: (piece.score, piece.width))
            if core.width >= max(3, round(typical_width * 0.42)):
                left = max(run.start, run.start + core.start)
                right = min(run.end, run.start + core.end)
                removed_mass = sum(local[: max(0, left - run.start)]) + sum(
                    local[max(0, right - run.start):]
                )
                if (
                    right - left >= max(3, round(typical_width * 0.42))
                    and removed_mass <= max(1.0, sum(local) * 0.28)
                    and (left > run.start or right < run.end)
                ):
                    return _Run(left, right, sum(projection[left:right]))
    if run.width <= max(typical_width * 1.18, typical_width + 3):
        return run
    target = max(3, min(run.width, round(typical_width * 1.04)))
    if len(local) <= target:
        return run
    prefix = [0.0]
    for value in local:
        prefix.append(prefix[-1] + value)
    best_start = 0
    best_score = -1.0
    for start in range(0, len(local) - target + 1):
        score = prefix[start + target] - prefix[start]
        if score > best_score:
            best_score = score
            best_start = start
    left = run.start + best_start
    right = left + target
    return _Run(left, right, max(0.0, best_score))


def _run_center(run: _Run) -> float:
    return (float(run.start) + float(run.end)) / 2.0


def _estimate_column_pitch(runs: list[_Run]) -> float:
    """Estimate normal centre-to-centre spacing while ignoring missing-column gaps."""
    ordered = sorted((run for run in runs if run.width > 0), key=lambda run: run.start)
    if len(ordered) < 2:
        return 0.0
    widths = [run.width for run in ordered]
    typical_width = max(2.0, _percentile(widths, 0.65))
    gaps = [
        _run_center(right) - _run_center(left)
        for left, right in zip(ordered, ordered[1:])
        if _run_center(right) - _run_center(left) >= typical_width * 1.05
    ]
    if not gaps:
        return 0.0
    # The lower-middle percentile rejects 2x/3x gaps created by omitted very
    # short columns, but remains stable on ordinary evenly spaced novel pages.
    pitch = _percentile([max(1, round(value * 1000)) for value in gaps], 0.42) / 1000.0
    return max(typical_width * 1.08, pitch)


def _recover_missing_pitch_runs(mask: Image.Image, runs: list[_Run]) -> list[_Run]:
    """Recover tiny short columns only inside an otherwise regular column grid.

    A physical Japanese column can contain just one or two punctuation/kana
    glyphs.  Its x projection may never reach the normal per-pixel threshold,
    even though the two neighbouring column centres leave an unmistakable 2x
    pitch gap.  Lowering the threshold for the entire page would turn dust and
    ruby into columns, so this pass searches only the predicted missing slots.
    """
    ordered = sorted(runs, key=lambda run: run.start)
    if len(ordered) < 4:
        return ordered
    pitch = _estimate_column_pitch(ordered)
    if pitch <= 0:
        return ordered

    widths = [run.width for run in ordered if run.width > 0]
    scores = [run.score for run in ordered if run.score > 0]
    typical_width = max(2.0, _percentile(widths, 0.65))
    typical_score = max(1.0, _percentile([max(1, round(value)) for value in scores], 0.55))
    projection = _column_projection(mask, core_only=False)
    recovered: list[_Run] = []

    for left_run, right_run in zip(ordered, ordered[1:]):
        left_center = _run_center(left_run)
        right_center = _run_center(right_run)
        centre_gap = right_center - left_center
        if centre_gap < pitch * 1.58 or centre_gap > pitch * 4.4:
            continue
        slots = max(2, int(round(centre_gap / pitch)))
        missing_count = slots - 1
        if missing_count <= 0 or missing_count > 3:
            continue
        actual_step = centre_gap / slots
        if not (pitch * 0.72 <= actual_step <= pitch * 1.28):
            continue

        for slot_index in range(1, slots):
            expected_center = left_center + actual_step * slot_index
            half_window = max(2.0, min(pitch * 0.34, typical_width * 0.78))
            search_left = max(left_run.end + 1, int(math.floor(expected_center - half_window)))
            search_right = min(right_run.start - 1, int(math.ceil(expected_center + half_window)))
            if search_right - search_left < 1:
                continue

            # One isolated antialiased speck can contribute about one projected
            # pixel.  Requiring a small vertical span and aggregate mass keeps
            # this local low threshold omission-safe.
            local_threshold = max(0.18, mask.height * 0.00022)
            active = [projection[x] >= local_threshold for x in range(search_left, search_right + 1)]
            active = _fill_short_false_gaps(active, max_gap=1)
            local_runs = _runs_from_active(active, projection[search_left:search_right + 1])
            candidates: list[_Run] = []
            for candidate in local_runs:
                absolute = _Run(
                    search_left + candidate.start,
                    search_left + candidate.end,
                    candidate.score,
                )
                centre_error = abs(_run_center(absolute) - expected_center)
                if centre_error > pitch * 0.28:
                    continue
                top, bottom = _vertical_ink_bounds(mask, absolute.start, absolute.end)
                vertical_span = bottom - top
                min_score = max(4.0, typical_score * 0.0030, mask.height * 0.0055)
                min_span = max(5.0, typical_width * 0.62, mask.height * 0.008)
                if absolute.score < min_score or vertical_span < min_span:
                    continue
                if absolute.width > max(typical_width * 1.9, pitch * 0.82):
                    continue
                candidates.append(absolute)
            if candidates:
                recovered.append(max(candidates, key=lambda run: (run.score, -abs(_run_center(run) - expected_center))))

    if not recovered:
        return ordered
    combined = sorted([*ordered, *recovered], key=lambda run: run.start)
    deduped: list[_Run] = []
    for run in combined:
        if deduped and run.start < deduped[-1].end:
            previous = deduped[-1]
            deduped[-1] = previous if previous.score >= run.score else run
        else:
            deduped.append(run)
    return deduped


def _recover_edge_pitch_runs(
    mask: Image.Image,
    runs: list[_Run],
    *,
    body_left: int = 0,
    body_right: int | None = None,
) -> list[_Run]:
    """Recover one/two-glyph physical columns at either edge of the body box.

    Interior omission recovery can infer a missing slot from two neighbouring
    columns.  At the left/right edge there is only one neighbour, so the old
    detector silently lost short endings such as ``う`` or ``だ。``.  The user's
    fixed blue rectangle supplies the missing boundary: search only the next
    regular-pitch slot inside that trusted body area and require real local ink.
    Blank page margin is never converted into a column.
    """
    ordered = sorted((run for run in runs if run.width > 0), key=lambda run: run.start)
    if len(ordered) < 4:
        return ordered
    pitch = _estimate_column_pitch(ordered)
    if pitch <= 0:
        return ordered

    body_left = max(0, min(mask.width - 1, int(body_left)))
    body_right = mask.width if body_right is None else int(body_right)
    body_right = max(body_left + 1, min(mask.width, body_right))
    widths = [run.width for run in ordered]
    scores = [run.score for run in ordered if run.score > 0]
    typical_width = max(2.0, _percentile(widths, 0.65))
    typical_score = max(1.0, _percentile([max(1, round(value)) for value in scores], 0.55))
    projection = _column_projection(mask, core_only=False)

    def find_candidate(expected_center: float, search_left: int, search_right: int) -> _Run | None:
        search_left = max(body_left, int(search_left))
        search_right = min(body_right - 1, int(search_right))
        if search_right - search_left < 1:
            return None
        local_threshold = max(0.18, mask.height * 0.00022)
        segment = projection[search_left:search_right + 1]
        active = [value >= local_threshold for value in segment]
        active = _fill_short_false_gaps(active, max_gap=1)
        candidates: list[_Run] = []
        for candidate in _runs_from_active(active, segment):
            absolute = _Run(
                search_left + candidate.start,
                search_left + candidate.end,
                candidate.score,
            )
            if abs(_run_center(absolute) - expected_center) > pitch * 0.34:
                continue
            top, bottom = _vertical_ink_bounds(mask, absolute.start, absolute.end)
            vertical_span = bottom - top
            # Stricter than the interior pass because page edges can contain dust
            # and binding/shadow remnants, while still accepting one normal glyph.
            min_score = max(5.0, typical_score * 0.0035, mask.height * 0.0060)
            min_span = max(6.0, typical_width * 0.58, mask.height * 0.0070)
            if absolute.score < min_score or vertical_span < min_span:
                continue
            if absolute.width > max(typical_width * 1.9, pitch * 0.82):
                continue
            candidates.append(absolute)
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda run: (run.score, -abs(_run_center(run) - expected_center)),
        )

    recovered: list[_Run] = []
    first = ordered[0]
    last = ordered[-1]
    edge_specs = (
        (-1, first, body_left, first.start - 1),
        (1, last, last.end + 1, body_right - 1),
    )
    for direction, anchor, edge_start, edge_end in edge_specs:
        anchor_center = _run_center(anchor)
        available = (anchor_center - body_left) if direction < 0 else (body_right - anchor_center)
        if available < pitch * 0.55:
            continue
        # Two slots cover rare pages ending in two tiny columns, but stop at the
        # first empty expected slot so a generous blank margin is not scanned.
        for slot in range(1, 3):
            expected_center = anchor_center + direction * pitch * slot
            if expected_center < body_left + 1 or expected_center > body_right - 1:
                break
            half_window = max(2.0, min(pitch * 0.36, typical_width * 0.82))
            left = math.floor(expected_center - half_window)
            right = math.ceil(expected_center + half_window)
            if direction < 0:
                right = min(right, edge_end)
            else:
                left = max(left, edge_start)
            candidate = find_candidate(expected_center, left, right)
            if candidate is None:
                break
            recovered.append(candidate)
            if direction < 0:
                edge_end = candidate.start - 1
            else:
                edge_start = candidate.end + 1

    if not recovered:
        return ordered
    combined = sorted([*ordered, *recovered], key=lambda run: run.start)
    deduped: list[_Run] = []
    for run in combined:
        if deduped and run.start < deduped[-1].end:
            previous = deduped[-1]
            deduped[-1] = previous if previous.score >= run.score else run
        else:
            deduped.append(run)
    return deduped


def _make_ink_mask(image: Image.Image, *, max_width: int = 1600, max_height: int = 2200):
    width, height = image.size
    scale = min(1.0, max_width / max(1, width), max_height / max(1, height))
    if scale < 1.0:
        sample = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.LANCZOS,
        )
    else:
        sample = image.copy()
    gray = ImageOps.autocontrast(sample.convert("L"), cutoff=1)
    threshold = max(85, min(238, _otsu_threshold(gray.histogram()) + 8))
    lookup = [255 if value <= threshold else 0 for value in range(256)]
    mask = gray.point(lookup, mode="L")
    sample.close()
    gray.close()
    return mask, scale, threshold


def _column_projection(mask: Image.Image, *, core_only: bool) -> list[float]:
    width, height = mask.size
    if core_only and height >= 40:
        top = round(height * 0.08)
        bottom = round(height * 0.92)
        source = mask.crop((0, top, width, max(top + 1, bottom)))
    else:
        source = mask
    try:
        collapsed = source.resize((width, 1), Image.Resampling.BOX)
        raw_values = (
            collapsed.get_flattened_data()
            if hasattr(collapsed, "get_flattened_data")
            else collapsed.getdata()
        )
        values = [float(value) * source.height / 255.0 for value in raw_values]
        collapsed.close()
        return values
    finally:
        if source is not mask:
            source.close()


def _detect_runs(mask: Image.Image, sensitivity: int, *, core_only: bool) -> list[_Run]:
    projection = _smooth(_column_projection(mask, core_only=core_only), radius=1)
    if not projection:
        return []
    sensitivity = max(1, min(100, int(sensitivity)))
    height = mask.height * (0.84 if core_only else 1.0)
    ratio = 0.0080 - (sensitivity / 100.0) * 0.0065
    peak_ratio = 0.050 - (sensitivity / 100.0) * 0.032
    threshold = max(2.0, height * ratio, max(projection) * peak_ratio)
    active = [value >= threshold for value in projection]
    active = _fill_short_false_gaps(active, max_gap=max(1, round(mask.width * 0.0018)))
    runs = _runs_from_active(active, projection)

    min_width = max(2, round(mask.width * 0.0012))
    min_score = max(12.0, height * (0.012 if core_only else 0.009))
    runs = [run for run in runs if run.width >= min_width and run.score >= min_score]
    # Never merge neighbouring runs here.  Ruby classification and regular-grid
    # recovery need the original independent x bands; early merging is both
    # irreversible and the direct cause of two-column/wide-column preview boxes.
    return _dedupe_overlapping_runs(runs)


def _vertical_ink_bounds(mask: Image.Image, left: int, right: int) -> tuple[int, int]:
    left = max(0, min(mask.width - 1, left))
    right = max(left + 1, min(mask.width, right))
    crop = mask.crop((left, 0, right, mask.height))
    try:
        collapsed = crop.resize((1, mask.height), Image.Resampling.BOX)
        raw_values = (
            collapsed.get_flattened_data()
            if hasattr(collapsed, "get_flattened_data")
            else collapsed.getdata()
        )
        rows = [float(value) * crop.width / 255.0 for value in raw_values]
        collapsed.close()
    finally:
        crop.close()
    active = [index for index, value in enumerate(rows) if value >= 0.75]
    if not active:
        return 0, mask.height
    return active[0], active[-1] + 1


def _vertical_content_spans(mask: Image.Image, left: int, right: int) -> tuple[list[tuple[int, int]], int]:
    """Return vertically separated ink blocks and a conservative glyph estimate.

    A full novel column can contain a very short paragraph/dialogue fragment
    separated by a large blank gap.  End-to-end recognizers sometimes return the
    long block and omit the short one.  The spans are only used for targeted
    recovery; ordinary continuous columns still need a single OCR call.
    """
    left = max(0, min(mask.width - 1, left))
    right = max(left + 1, min(mask.width, right))
    crop = mask.crop((left, 0, right, mask.height))
    try:
        collapsed = crop.resize((1, mask.height), Image.Resampling.BOX)
        raw_values = (
            collapsed.get_flattened_data()
            if hasattr(collapsed, "get_flattened_data")
            else collapsed.getdata()
        )
        rows = [float(value) * crop.width / 255.0 for value in raw_values]
        collapsed.close()
    finally:
        crop.close()

    threshold = max(0.65, (right - left) * 0.010)
    active = [value >= threshold for value in rows]
    active = _fill_short_false_gaps(active, max_gap=max(1, round(mask.height * 0.0015)))
    glyph_runs = _runs_from_active(active, rows)
    if not glyph_runs:
        return [], 0

    heights = [run.width for run in glyph_runs if run.width > 0]
    typical_height = max(3.0, _percentile(heights, 0.55))
    large_gap = max(10, round(mask.height * 0.014), round(typical_height * 1.25))
    spans: list[tuple[int, int]] = []
    start = glyph_runs[0].start
    end = glyph_runs[0].end
    for run in glyph_runs[1:]:
        if run.start - end > large_gap:
            spans.append((start, end))
            start, end = run.start, run.end
        else:
            end = run.end
    spans.append((start, end))

    # Row components slightly over-count kana with detached marks, therefore the
    # estimate is used only to detect severe under-recognition, never as text.
    estimated_chars = max(1, round(sum(max(1, run.width) for run in glyph_runs) / typical_height))
    return spans, estimated_chars


def _detached_punctuation_boxes(
    mask: Image.Image,
    run: _Run,
    *,
    hard_left: int,
    hard_right: int,
) -> list[tuple[int, int, int, int]]:
    """Return a few tiny side components that behave like hanging punctuation.

    The body rectangle itself stays ruby-free.  Legitimate punctuation can hang
    just outside that rectangle, so it is copied as an isolated component rather
    than widening the full column box.  A ruby annotation produces many repeated
    small components and is therefore rejected as a group.
    """
    body_width = max(1, run.width)
    # Ruby occupies the same vertical rows as its base glyphs.  Hanging
    # punctuation, by contrast, normally occupies an otherwise empty character
    # cell just outside the body band.  A body-row ink projection lets us reject
    # even a *single* small ruby component, which component-count-only logic
    # cannot distinguish safely.
    body_band = mask.crop((run.start, 0, run.end, mask.height))
    try:
        body_bytes = body_band.tobytes()
        body_stride = max(1, body_band.width)
        body_row_ink = [
            sum(1 for value in body_bytes[row * body_stride:(row + 1) * body_stride] if value)
            for row in range(body_band.height)
        ]
    finally:
        body_band.close()
    accepted: list[tuple[int, int, int, int]] = []
    for side_left, side_right in ((hard_left, run.start), (run.end, hard_right)):
        side_left = max(0, min(mask.width, int(side_left)))
        side_right = max(side_left, min(mask.width, int(side_right)))
        if side_right - side_left < 1:
            continue
        band = mask.crop((side_left, 0, side_right, mask.height))
        try:
            bw, bh = band.size
            data = bytearray(band.tobytes())
        finally:
            band.close()
        components: list[tuple[int, int, int, int, int]] = []
        for start in range(len(data)):
            if data[start] == 0:
                continue
            stack = [start]
            data[start] = 0
            area = 0
            min_x = max_x = start % bw
            min_y = max_y = start // bw
            while stack:
                index = stack.pop()
                area += 1
                x = index % bw
                y = index // bw
                min_x = min(min_x, x); max_x = max(max_x, x)
                min_y = min(min_y, y); max_y = max(max_y, y)
                for ny in range(max(0, y - 1), min(bh, y + 2)):
                    row = ny * bw
                    for nx in range(max(0, x - 1), min(bw, x + 2)):
                        neighbour = row + nx
                        if data[neighbour]:
                            data[neighbour] = 0
                            stack.append(neighbour)
            comp_w = max_x - min_x + 1
            comp_h = max_y - min_y + 1
            gap = (
                run.start - (side_left + max_x + 1)
                if side_right <= run.start
                else (side_left + min_x) - run.end
            )
            max_area = max(8, round(body_width * body_width * 0.42))
            body_overlap = sum(body_row_ink[min_y:max_y + 1])
            empty_body_cell = body_overlap <= max(2, round(area * 0.50))
            if (
                2 <= area <= max_area
                and comp_w <= max(4, round(body_width * 0.52))
                and comp_h <= max(5, round(body_width * 0.72))
                and gap <= max(4, round(body_width * 0.58))
                and empty_body_cell
            ):
                components.append((
                    side_left + min_x,
                    min_y,
                    side_left + max_x + 1,
                    max_y + 1,
                    area,
                ))
        # One or two hanging marks are plausible punctuation.  Repeated side
        # components are furigana/notes and must stay excluded at source.
        if 0 < len(components) <= 2:
            vertical_extent = max(item[3] for item in components) - min(item[1] for item in components)
            if vertical_extent <= max(body_width * 2.6, 18):
                accepted.extend((x0, y0, x1, y1) for x0, y0, x1, y1, _area in components)
    return accepted


def _remove_running_margin_artifacts(
    columns: Sequence[DetectedColumn],
    *,
    page_width: int,
    page_height: int,
) -> list[DetectedColumn]:
    """Drop isolated header/page-number boxes and trim them from body columns.

    A fixed GUI crop can still include a few pixels of a running header or page
    number, especially when the user intentionally leaves a generous top margin.
    Those fragments are geometrically different from normal vertical body text:
    they are isolated above the common body start, very short, or form a wide
    shallow horizontal band.  This pass runs *after* physical-column detection so
    it can use the page's own body-column consensus instead of a hard-coded crop.

    The pass is deliberately conservative.  It activates only when at least four
    substantial body columns establish a stable reference, and it never removes
    a continuous column that reaches into the body area.  When a header shares
    the same x slot as a body column, only the separated leading content span is
    discarded; the body pixels and physical slot are retained.
    """
    items = list(columns)
    if len(items) < 4 or page_width <= 0 or page_height <= 0:
        return items

    widths = [max(1, column.width) for column in items]
    heights = [max(1, column.height) for column in items]
    typical_width = max(2.0, _percentile(widths, 0.60))
    typical_height = max(4.0, _percentile(heights, 0.60))

    substantial_height = max(page_height * 0.24, typical_height * 0.52)
    max_body_width = max(typical_width * 2.15, page_width * 0.055)
    body_candidates = [
        column for column in items
        if column.height >= substantial_height and column.width <= max_body_width
    ]
    if len(body_candidates) < 4:
        return items

    def body_span(column: DetectedColumn) -> tuple[int, int]:
        spans = [span for span in column.content_spans if span[1] > span[0]]
        if not spans:
            return int(column.top), int(column.bottom)
        min_main_span = max(page_height * 0.035, max(1, column.width) * 2.0)
        substantial = [span for span in spans if (span[1] - span[0]) >= min_main_span]
        if substantial:
            # The earliest substantial span is normally the first real paragraph
            # block; a page number/header fragment is too short to qualify.
            return min(substantial, key=lambda span: span[0])
        return max(spans, key=lambda span: span[1] - span[0])

    anchors = [body_span(column) for column in body_candidates]
    body_top = _percentile([start for start, _ in anchors], 0.25)
    body_bottom = _percentile([end for _, end in anchors], 0.75)
    guard = max(5.0, typical_width * 0.55, page_height * 0.008)
    tiny_span_limit = max(typical_width * 3.25, page_height * 0.075)
    shallow_band_limit = max(typical_width * 4.5, page_height * 0.16)
    wide_band_width = max(typical_width * 2.20, page_width * 0.065)

    cleaned: list[DetectedColumn] = []
    for column in items:
        height = max(1, column.height)
        width = max(1, column.width)
        entirely_above = column.bottom < body_top - guard
        entirely_below = column.top > body_bottom + guard
        short_isolated = height <= tiny_span_limit
        wide_shallow = width >= wide_band_width and height <= shallow_band_limit

        # Running headers/page numbers above the body remain safe to remove once
        # four stable body columns establish a reliable top boundary.  The lower
        # margin is intentionally asymmetric: a legitimate final Japanese column
        # can contain only one punctuation mark or a short two/three-glyph phrase
        # after a large blank gap.  Geometry alone cannot distinguish that text
        # from a footer number, therefore narrow bottom fragments must be kept.
        # Only an unmistakably wide, shallow horizontal band is removed below.
        if entirely_above and (short_isolated or wide_shallow):
            continue
        if entirely_below and wide_shallow:
            continue
        if wide_shallow and column.bottom <= body_top + guard:
            continue

        spans = [span for span in column.content_spans if span[1] > span[0]]
        if spans:
            kept = list(spans)
            # Header/page-number glyphs can share the same x strip as a real body
            # column.  Remove only separated, short margin spans.
            while len(kept) > 1:
                start, end = kept[0]
                if end < body_top - guard and (end - start) <= tiny_span_limit:
                    kept.pop(0)
                else:
                    break
            # Do not trim trailing separated spans.  In vertical novels the
            # continuation at the bottom of a physical column may be only a
            # punctuation mark, small kana, or a short title/name after a large
            # intentional gap.  Deleting it is irreversible and more harmful
            # than retaining a possible footer inside the user's fixed region.
            if kept != spans:
                vertical_pad = max(3, round(page_height * 0.012))
                new_top = max(0, min(column.top, kept[0][0]) - vertical_pad)
                # Do not let the original header-derived top survive the trim.
                new_top = max(0, kept[0][0] - vertical_pad)
                new_bottom = min(page_height, kept[-1][1] + vertical_pad)
                column = DetectedColumn(
                    left=column.left,
                    top=new_top,
                    right=column.right,
                    bottom=max(new_top + 1, new_bottom),
                    hard_left=column.hard_left,
                    hard_right=column.hard_right,
                    ink_score=column.ink_score,
                    content_spans=tuple(kept),
                    estimated_chars=column.estimated_chars,
                    full_height_slot=column.full_height_slot,
                    supplemental_boxes=column.supplemental_boxes,
                )
        cleaned.append(column)

    # Never let a heuristic erase a large fraction of a page.  This fallback is
    # important for unusual title/opening layouts.
    if len(cleaned) < max(3, round(len(items) * 0.70)):
        return items
    return cleaned


def _detect_vertical_columns_projection(
    image_or_path: Image.Image | str | os.PathLike[str],
    *,
    sensitivity: int = 55,
    padding_percent: int = 10,
    max_columns: int = 80,
    fixed_region_rect: Sequence[float] | None = None,
    fixed_region_already_masked: bool = False,
) -> list[DetectedColumn]:
    """Detect Japanese vertical columns in right-to-left reading order.

    When ``fixed_region_rect`` is supplied, that user-confirmed blue rectangle is
    the authoritative body area.  Every physical column slot spans its complete
    top-to-bottom height and neighbouring slots meet at non-overlapping x
    midlines.  Tight ink spans are retained separately in ``content_spans`` for
    empty-column checks and targeted recovery, but never shrink the OCR mask.
    """
    owns_image = not isinstance(image_or_path, Image.Image)
    image = Image.open(image_or_path).convert("RGB") if owns_image else image_or_path.convert("RGB")
    if fixed_region_rect and not fixed_region_already_masked:
        fixed_masked = _mask_to_fixed_region(image, fixed_region_rect)
        image.close()
        image = fixed_masked
    body_bounds_px = _normalized_body_bounds(image, fixed_region_rect)
    try:
        mask, scale, _threshold = _make_ink_mask(image)
        try:
            runs = _split_connected_runs_by_periodic_grid(
                mask,
                _split_overwide_runs(
                    mask, _detect_runs(mask, sensitivity, core_only=True)
                ),
            )
            full_runs = _split_connected_runs_by_periodic_grid(
                mask,
                _split_overwide_runs(
                    mask, _detect_runs(mask, sensitivity, core_only=False)
                ),
            )
            runs, ruby_excluded = _filter_ruby_side_runs(mask, runs)
            full_runs, full_ruby_excluded = _filter_ruby_side_runs(mask, full_runs)
            ruby_excluded = _dedupe_overlapping_runs(
                [*ruby_excluded, *full_ruby_excluded]
            )
            # Chapter-opening pages may contain only short columns.  The central
            # band can then be empty, so retry using the complete fixed region.
            if len(runs) < 2:
                if len(full_runs) > len(runs):
                    runs = full_runs
            else:
                # Preserve short vertical title/subtitle columns that live only
                # near the top/bottom and therefore do not appear in the central
                # body projection.  Horizontal title glyphs are rejected by the
                # vertical-span test instead of being mistaken for many columns.
                typical = max(2.0, _percentile([run.width for run in runs], 0.65))
                supplemental: list[_Run] = []
                for candidate in full_runs:
                    overlaps = any(
                        min(candidate.end, base.end) - max(candidate.start, base.start)
                        > min(candidate.width, base.width) * 0.25
                        for base in runs
                    )
                    if overlaps or candidate.width > max(typical * 2.2, mask.width * 0.08):
                        continue
                    top, bottom = _vertical_ink_bounds(mask, candidate.start, candidate.end)
                    vertical_span = bottom - top
                    if vertical_span >= max(candidate.width * 2.0, mask.height * 0.045):
                        supplemental.append(candidate)
                if supplemental:
                    runs = _dedupe_overlapping_runs([*runs, *supplemental])
            # Recover one/two-character physical columns that are too light for
            # the ordinary x-projection threshold, but occupy a clear missing
            # slot in the otherwise regular vertical-column grid.
            runs = _recover_missing_pitch_runs(mask, runs)
            if body_bounds_px is not None:
                edge_body_left = max(0, min(mask.width - 1, math.floor(body_bounds_px[0] * scale)))
                edge_body_right = max(
                    edge_body_left + 1,
                    min(mask.width, math.ceil(body_bounds_px[2] * scale)),
                )
            else:
                edge_body_left, edge_body_right = 0, mask.width
            runs = _recover_edge_pitch_runs(
                mask, runs, body_left=edge_body_left, body_right=edge_body_right
            )
            # Recovery and supplemental-title passes can reintroduce one broad
            # connected band.  Reapply the independent periodic splitter before
            # the final ruby classification so every preview/OCR target remains
            # exactly one physical column.
            runs = _split_connected_runs_by_periodic_grid(mask, runs)
            # Recovery is grid-constrained, but run one final source-level ruby
            # classification so no narrow side strip can become a physical slot
            # through an edge/supplemental path.
            runs, late_ruby_excluded = _filter_ruby_side_runs(mask, runs)
            ruby_excluded = _dedupe_overlapping_runs(
                [*ruby_excluded, *late_ruby_excluded]
            )
            if not runs:
                return []
            if len(runs) > max_columns:
                raise RuntimeError(
                    f"检测到 {len(runs)} 个候选竖列，超过安全上限 {max_columns}。"
                    "请缩小固定正文框，或降低分列灵敏度。"
                )

            # Extremely wide bands usually mean the fixed region still includes a
            # horizontal illustration/header or the threshold is too permissive.
            widths = [run.width for run in runs]
            typical_width = max(2.0, _percentile(widths, 0.65))
            suspicious = [run for run in runs if run.width > max(mask.width * 0.24, typical_width * 4.0)]
            if suspicious and len(runs) <= 2:
                raise RuntimeError(
                    "未检测到稳定的竖排正文列；当前识别区域可能包含横排标题、插图或过多页边。"
                    "请重新框选纯正文区域，或调低分列灵敏度。"
                )

            runs = sorted(runs, key=lambda run: run.start)
            final_widths = [run.width for run in runs if run.width > 0]
            final_typical_width = max(
                2.0,
                _percentile(final_widths, 0.62) if final_widths else 2.0,
            )
            original_runs = list(runs)
            runs = [
                _refine_body_run(mask, run, final_typical_width)
                for run in original_runs
            ]
            trim_flags = [
                (refined.start > original.start, refined.end < original.end)
                for original, refined in zip(original_runs, runs)
            ]
            pitch = _estimate_column_pitch(runs)
            pad_ratio = max(0, min(30, int(padding_percent))) / 100.0
            inv_scale = 1.0 / max(scale, 1e-9)
            if body_bounds_px is not None:
                body_left_px, body_top_px, body_right_px, body_bottom_px = body_bounds_px
                body_left = max(0, min(mask.width - 1, math.floor(body_left_px * scale)))
                body_right = max(body_left + 1, min(mask.width, math.ceil(body_right_px * scale)))
            else:
                body_left_px, body_top_px, body_right_px, body_bottom_px = (
                    0, 0, image.width, image.height
                )
                body_left, body_right = 0, mask.width

            columns: list[DetectedColumn] = []
            for index, run in enumerate(runs):
                if index == 0:
                    if body_bounds_px is not None:
                        hard_left = body_left
                    else:
                        edge_context = max(run.width * 0.80, pitch * 0.52) if pitch > 0 else run.width * 1.25
                        hard_left = max(0, round(run.start - edge_context))
                else:
                    hard_left = round((runs[index - 1].end + run.start) / 2)
                if index == len(runs) - 1:
                    if body_bounds_px is not None:
                        hard_right = body_right
                    else:
                        edge_context = max(run.width * 0.80, pitch * 0.52) if pitch > 0 else run.width * 1.25
                        hard_right = min(mask.width, round(run.end + edge_context))
                else:
                    hard_right = round((run.end + runs[index + 1].start) / 2)
                hard_left = max(body_left, min(body_right - 1, hard_left))
                hard_right = max(hard_left + 1, min(body_right, hard_right))

                # Recognition rectangles are body-only.  ``hard_*`` still define
                # non-overlapping logical slots for NDLOCR routing and ordering,
                # while ``left/right`` are the actual pixels exposed to every OCR
                # engine and to the red preview box.
                pad = max(1, round(run.width * min(pad_ratio, 0.06)))
                trimmed_left, trimmed_right = trim_flags[index]
                left_pad = 0 if trimmed_left else pad
                right_pad = 0 if trimmed_right else pad
                for excluded in ruby_excluded:
                    if excluded.end <= run.start:
                        gap = run.start - excluded.end
                        if gap <= max(2, round(final_typical_width * 0.70)):
                            left_pad = 0
                    elif excluded.start >= run.end:
                        gap = excluded.start - run.end
                        if gap <= max(2, round(final_typical_width * 0.70)):
                            right_pad = 0
                crop_left = max(hard_left, run.start - left_pad)
                crop_right = min(hard_right, run.end + right_pad)
                supplemental_mask_boxes = _detached_punctuation_boxes(
                    mask,
                    run,
                    hard_left=hard_left,
                    hard_right=hard_right,
                )
                # Tight ink bounds remain useful as content metadata, but a
                # trusted GUI body rectangle must never be shrunk vertically.
                ink_top, ink_bottom = _vertical_ink_bounds(mask, crop_left, crop_right)
                if supplemental_mask_boxes:
                    ink_top = min(ink_top, min(box[1] for box in supplemental_mask_boxes))
                    ink_bottom = max(ink_bottom, max(box[3] for box in supplemental_mask_boxes))
                vertical_pad = max(3, round(mask.height * 0.012))
                ink_top = max(0, ink_top - vertical_pad)
                ink_bottom = min(mask.height, ink_bottom + vertical_pad)

                left_px = max(body_left_px, math.floor(crop_left * inv_scale))
                right_px = min(body_right_px, math.ceil(crop_right * inv_scale))
                hard_left_px = max(body_left_px, math.floor(hard_left * inv_scale))
                hard_right_px = min(body_right_px, math.ceil(hard_right * inv_scale))
                if body_bounds_px is not None:
                    top_px = body_top_px
                    bottom_px = body_bottom_px
                else:
                    top_px = max(0, math.floor(ink_top * inv_scale))
                    bottom_px = min(image.height, math.ceil(ink_bottom * inv_scale))
                if right_px - left_px < 4 or bottom_px - top_px < 4:
                    continue
                span_rows, estimated_chars = _vertical_content_spans(mask, crop_left, crop_right)
                span_rows.extend((box[1], box[3]) for box in supplemental_mask_boxes)
                span_rows = sorted(span_rows)
                content_spans = tuple(
                    (
                        max(body_top_px, math.floor(span_top * inv_scale)),
                        min(body_bottom_px, math.ceil(span_bottom * inv_scale)),
                    )
                    for span_top, span_bottom in span_rows
                    if span_bottom > span_top
                    and math.ceil(span_bottom * inv_scale) > body_top_px
                    and math.floor(span_top * inv_scale) < body_bottom_px
                )
                supplemental_boxes = tuple(
                    (
                        max(body_left_px, math.floor(box_left * inv_scale)),
                        max(body_top_px, math.floor(box_top * inv_scale)),
                        min(body_right_px, math.ceil(box_right * inv_scale)),
                        min(body_bottom_px, math.ceil(box_bottom * inv_scale)),
                    )
                    for box_left, box_top, box_right, box_bottom in supplemental_mask_boxes
                    if box_right > box_left and box_bottom > box_top
                )
                columns.append(DetectedColumn(
                    left=left_px,
                    top=top_px,
                    right=right_px,
                    bottom=bottom_px,
                    hard_left=hard_left_px,
                    hard_right=hard_right_px,
                    ink_score=run.score,
                    content_spans=content_spans,
                    estimated_chars=estimated_chars + len(supplemental_boxes),
                    full_height_slot=body_bounds_px is not None,
                    supplemental_boxes=supplemental_boxes,
                ))
            if body_bounds_px is None:
                columns = _remove_running_margin_artifacts(
                    columns,
                    page_width=image.width,
                    page_height=image.height,
                )
            return sorted(columns, key=lambda column: (-(column.left + column.right), column.top))
        finally:
            mask.close()
    finally:
        image.close()



@dataclass(frozen=True)
class _InkComponent:
    left: int
    top: int
    right: int
    bottom: int
    area: int

    @property
    def width(self) -> int:
        return max(1, self.right - self.left)

    @property
    def height(self) -> int:
        return max(1, self.bottom - self.top)

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2.0

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2.0


def _scanline_connected_ink_components(mask: Image.Image) -> list[_InkComponent]:
    """Label 8-connected ink components using only Pillow and Python.

    The ordinary column splitter must work in the main GUI environment even when
    OpenCV, SciPy, or NumPy are not installed.  A row-run union/find algorithm is
    substantially faster than a per-pixel Python flood fill and keeps memory
    proportional to the number of printed-ink runs rather than page pixels.
    """
    binary = mask.convert("L").point(lambda value: 255 if value else 0, mode="L")
    try:
        width, height = binary.size
        if width <= 0 or height <= 0:
            return []
        raw = binary.tobytes()
    finally:
        if binary is not mask:
            binary.close()

    parent: list[int] = []
    rank: list[int] = []
    runs: list[tuple[int, int, int, int]] = []  # y, x_start, x_end(exclusive), id

    def make_set() -> int:
        ident = len(parent)
        parent.append(ident)
        rank.append(0)
        return ident

    def find(value: int) -> int:
        root = value
        while parent[root] != root:
            root = parent[root]
        while parent[value] != value:
            next_value = parent[value]
            parent[value] = root
            value = next_value
        return root

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        if rank[root_left] < rank[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        if rank[root_left] == rank[root_right]:
            rank[root_left] += 1

    previous: list[tuple[int, int, int]] = []  # x_start, x_end, id
    ink = b"\xff"
    paper = b"\x00"
    for y in range(height):
        row = raw[y * width:(y + 1) * width]
        current: list[tuple[int, int, int]] = []
        cursor = 0
        while cursor < width:
            x_start = row.find(ink, cursor)
            if x_start < 0:
                break
            x_end = row.find(paper, x_start)
            if x_end < 0:
                x_end = width
            ident = make_set()
            current.append((x_start, x_end, ident))
            runs.append((y, x_start, x_end, ident))
            cursor = x_end + 1

        # Two sorted run lists can be joined in linear time.  End coordinates are
        # exclusive; equality therefore represents a one-pixel diagonal contact
        # and belongs to the same 8-connected printed glyph.
        previous_index = 0
        for x_start, x_end, ident in current:
            while (
                previous_index < len(previous)
                and previous[previous_index][1] < x_start
            ):
                previous_index += 1
            probe = previous_index
            while probe < len(previous) and previous[probe][0] <= x_end:
                prev_start, prev_end, prev_ident = previous[probe]
                if prev_end >= x_start and x_end >= prev_start:
                    union(ident, prev_ident)
                probe += 1
        previous = current

    if not runs:
        return []

    aggregated: dict[int, list[int]] = {}
    for y, x_start, x_end, ident in runs:
        root = find(ident)
        area = max(0, x_end - x_start)
        current = aggregated.get(root)
        if current is None:
            aggregated[root] = [x_start, y, x_end, y + 1, area]
        else:
            current[0] = min(current[0], x_start)
            current[1] = min(current[1], y)
            current[2] = max(current[2], x_end)
            current[3] = max(current[3], y + 1)
            current[4] += area

    return [
        _InkComponent(left, top, right, bottom, area)
        for left, top, right, bottom, area in aggregated.values()
        if right > left and bottom > top and area > 0
    ]


def _connected_ink_components(mask: Image.Image) -> list[_InkComponent]:
    """Return connected black-ink components without projection splitting.

    OpenCV/SciPy are optional accelerators only.  Earlier v6 code required NumPy
    plus one of those libraries in the GUI interpreter, although none is a base
    project dependency.  On a normal macOS install that made every selected OCR
    engine fail before the model was even started.  The built-in scanline fallback
    keeps ordinary component-based splitting available with Pillow alone.
    """
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None

    if np is not None:
        binary = (np.asarray(mask, dtype=np.uint8) > 0).astype("uint8")
        if binary.size <= 0 or int(binary.sum()) <= 0:
            return []
        try:
            import cv2  # type: ignore

            kernel = np.ones((2, 2), dtype=np.uint8)
            prepared = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
            count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
                prepared, connectivity=8
            )
            output: list[_InkComponent] = []
            for label in range(1, int(count)):
                left, top, width, height, area = [int(value) for value in stats[label]]
                if width > 0 and height > 0 and area > 0:
                    output.append(_InkComponent(left, top, left + width, top + height, area))
            return output
        except Exception:
            try:
                from scipy import ndimage  # type: ignore

                prepared = ndimage.binary_closing(
                    binary.astype(bool), structure=np.ones((2, 2), dtype=bool)
                )
                labels, _count = ndimage.label(
                    prepared, structure=np.ones((3, 3), dtype=int)
                )
                objects = ndimage.find_objects(labels)
                output: list[_InkComponent] = []
                for label, slices in enumerate(objects, start=1):
                    if not slices:
                        continue
                    ys, xs = slices
                    top, bottom = int(ys.start), int(ys.stop)
                    left, right = int(xs.start), int(xs.stop)
                    if right <= left or bottom <= top:
                        continue
                    area = int((labels[ys, xs] == label).sum())
                    if area > 0:
                        output.append(_InkComponent(left, top, right, bottom, area))
                return output
            except Exception:
                pass

    return _scanline_connected_ink_components(mask)


def _refresh_component_cluster_stats(cluster: dict) -> None:
    items = list(cluster.get("components") or [])
    if not items:
        cluster.update({
            "count": 0,
            "top": 0,
            "bottom": 0,
            "span": 0,
            "median_height": 0.0,
            "median_width": 0.0,
            "median_area": 0.0,
        })
        return
    cluster["count"] = len(items)
    cluster["top"] = min(item.top for item in items)
    cluster["bottom"] = max(item.bottom for item in items)
    cluster["span"] = int(cluster["bottom"] - cluster["top"])
    cluster["median_height"] = _percentile([item.height for item in items], 0.50)
    cluster["median_width"] = _percentile([item.width for item in items], 0.50)
    cluster["median_area"] = _percentile([item.area for item in items], 0.50)


def _component_grid_phase(clusters: Sequence[dict], pitch: float) -> tuple[float | None, int]:
    """Find the dominant physical-column phase without any pixel projection."""
    if pitch <= 0 or len(clusters) < 3:
        return None, 0
    tolerance = max(2.0, pitch * 0.19)
    best_phase: float | None = None
    best_score = -1.0
    best_count = 0
    for seed in clusters:
        phase = float(seed["center"])
        score = 0.0
        count = 0
        for cluster in clusters:
            center = float(cluster["center"])
            residual = abs(((center - phase + pitch / 2.0) % pitch) - pitch / 2.0)
            if residual <= tolerance:
                count += 1
                score += min(24.0, float(cluster.get("count", 1))) + min(
                    8.0, float(cluster.get("span", 0)) / max(1.0, pitch)
                )
        if (count, score) > (best_count, best_score):
            best_phase, best_count, best_score = phase, count, score
    return best_phase, best_count


def _cluster_component_columns(
    components: list[_InkComponent],
    *,
    page_width: int,
    page_height: int,
) -> tuple[list[dict], dict]:
    """Cluster printed-glyph components into physical vertical columns.

    This path is deliberately component-only: no x/y projection, periodic pixel
    sweep or per-character box generator is called.  A dominant body-glyph size
    establishes column centres; a regular centre grid is used only as geometry to
    recover very short physical columns and reject inter-column ruby bridges.
    """
    usable = [
        component for component in components
        if component.area >= 3
        and component.width <= max(12, round(page_width * 0.11))
        and component.height <= max(18, round(page_height * 0.18))
        and not (
            component.width >= max(10, round(page_width * 0.045))
            and component.height <= max(3, round(page_height * 0.004))
        )
    ]
    if not usable:
        return [], {"reason": "no_components"}

    size_pool = [
        component for component in usable
        if component.height >= 3
        and component.width >= 2
        and 0.12 <= component.width / max(1.0, component.height) <= 5.5
    ] or usable
    typical_height = max(4.0, _percentile([component.height for component in size_pool], 0.72))
    typical_width = max(3.0, _percentile([component.width for component in size_pool], 0.68))
    typical_area = max(5.0, _percentile([component.area for component in size_pool], 0.66))

    body_anchors: list[_InkComponent] = []
    for component in usable:
        size_ok = (
            component.height >= typical_height * 0.60
            and component.width >= max(1.0, typical_width * 0.20)
            and component.area >= max(3.0, typical_area * 0.18)
        )
        tall_ok = (
            component.height >= typical_height * 0.84
            and component.area >= max(3.0, typical_area * 0.11)
        )
        if size_ok or tall_ok:
            body_anchors.append(component)
    if not body_anchors:
        return [], {"reason": "no_body_anchors"}

    tolerance = max(2.0, typical_width * 0.70)
    clusters: list[dict] = []
    for component in sorted(body_anchors, key=lambda value: value.center_x):
        nearest = min(
            clusters,
            key=lambda cluster: abs(component.center_x - float(cluster["center"])),
            default=None,
        )
        nearest_distance = (
            abs(component.center_x - float(nearest["center"]))
            if nearest is not None else float("inf")
        )
        if nearest is None or nearest_distance > tolerance:
            clusters.append({
                "components": [component],
                "center": component.center_x,
                "weight": max(1.0, float(component.area)),
            })
        else:
            weight = max(1.0, float(component.area))
            old_weight = float(nearest["weight"])
            nearest["center"] = (
                float(nearest["center"]) * old_weight + component.center_x * weight
            ) / (old_weight + weight)
            nearest["weight"] = old_weight + weight
            nearest["components"].append(component)

    # Reunite radical fragments from one printed glyph, while staying far below
    # a normal physical-column pitch.
    clusters.sort(key=lambda value: float(value["center"]))
    merged: list[dict] = []
    merge_distance = max(2.0, typical_width * 0.86)
    for cluster in clusters:
        if merged and float(cluster["center"]) - float(merged[-1]["center"]) < merge_distance:
            previous = merged[-1]
            total = float(previous["weight"]) + float(cluster["weight"])
            previous["center"] = (
                float(previous["center"]) * float(previous["weight"])
                + float(cluster["center"]) * float(cluster["weight"])
            ) / max(1.0, total)
            previous["weight"] = total
            previous["components"].extend(cluster["components"])
        else:
            merged.append(cluster)
    clusters = merged
    for cluster in clusters:
        _refresh_component_cluster_stats(cluster)

    strong = [
        cluster for cluster in clusters
        if int(cluster["count"]) >= 2
        or int(cluster["span"]) >= max(typical_height * 1.45, page_height * 0.018)
    ]
    if not strong:
        strong = list(clusters)

    centers = sorted(float(cluster["center"]) for cluster in strong)
    gaps = [
        right - left for left, right in zip(centers, centers[1:])
        if right - left > typical_width * 1.02
    ]
    pitch = (
        _percentile([max(1, round(value * 100)) for value in gaps], 0.38) / 100.0
        if gaps else 0.0
    )
    if pitch < typical_width * 1.28:
        pitch = 0.0

    phase, aligned_count = _component_grid_phase(strong, pitch)
    grid_reliable = bool(
        phase is not None
        and pitch > 0
        and aligned_count >= max(3, round(len(strong) * 0.60))
    )

    if grid_reliable:
        alignment_tolerance = max(2.0, pitch * 0.22)
        aligned: list[dict] = []
        for cluster in strong:
            center = float(cluster["center"])
            slot = round((center - float(phase)) / pitch)
            expected = float(phase) + slot * pitch
            residual = abs(center - expected)
            if residual <= alignment_tolerance:
                cluster["grid_center"] = expected
                cluster["grid_slot"] = int(slot)
                aligned.append(cluster)
        # Only apply grid rejection when a clear majority remains.  This avoids
        # imposing a false regular grid on decorative or intentionally irregular
        # layouts, while removing half-pitch ruby/bridge clusters on body pages.
        if len(aligned) >= max(3, round(len(strong) * 0.55)):
            strong = aligned

    # Recover one/two-character physical columns by component evidence at missing
    # grid slots.  This is geometry, not pixel projection, and rejects lone dust.
    if grid_reliable and strong:
        occupied = {int(cluster.get("grid_slot", round((float(cluster["center"]) - float(phase)) / pitch))) for cluster in strong}
        min_slot, max_slot = min(occupied), max(occupied)
        candidate_slots = set(range(min_slot, max_slot + 1)) - occupied
        candidate_slots.update({min_slot - 1, min_slot - 2, max_slot + 1, max_slot + 2})
        recovered: list[dict] = []
        half_window = max(3.0, min(pitch * 0.34, typical_width * 0.90))
        for slot in sorted(candidate_slots):
            expected = float(phase) + slot * pitch
            if expected < 1 or expected > page_width - 1:
                continue
            nearby = [
                component for component in usable
                if abs(component.center_x - expected) <= half_window
            ]
            if not nearby:
                continue
            normal_component = any(
                component.height >= typical_height * 0.48
                and component.width >= max(2.0, typical_width * 0.22)
                and component.area >= max(8.0, typical_area * 0.10)
                for component in nearby
            )
            tiny_pair = (
                len(nearby) >= 2
                and sum(component.area for component in nearby) >= max(18.0, typical_area * 0.12)
                and max(component.center_y for component in nearby) - min(component.center_y for component in nearby)
                >= max(4.0, typical_height * 0.28)
            )
            if not (normal_component or tiny_pair):
                continue
            cluster = {
                "components": nearby,
                "center": expected,
                "grid_center": expected,
                "grid_slot": int(slot),
                "weight": float(sum(component.area for component in nearby)),
                "recovered_short": True,
            }
            _refresh_component_cluster_stats(cluster)
            recovered.append(cluster)
            occupied.add(slot)
        strong.extend(recovered)

    # Final local ruby guard for irregular pages that did not produce a reliable
    # grid.  A small weak cluster immediately beside a much stronger body column
    # cannot establish its own OCR slot.
    strong.sort(key=lambda value: float(value.get("grid_center", value["center"])))
    kept: list[dict] = []
    for cluster in strong:
        neighbours = [candidate for candidate in strong if candidate is not cluster]
        nearest = min(
            neighbours,
            key=lambda candidate: abs(
                float(candidate.get("grid_center", candidate["center"]))
                - float(cluster.get("grid_center", cluster["center"]))
            ),
            default=None,
        )
        ruby_like = False
        if nearest is not None and not cluster.get("recovered_short"):
            distance = abs(
                float(nearest.get("grid_center", nearest["center"]))
                - float(cluster.get("grid_center", cluster["center"]))
            )
            normal_slot = bool(pitch > 0 and distance >= pitch * 0.60)
            smaller = (
                float(cluster["median_height"]) < typical_height * 0.72
                and float(cluster["median_area"]) < typical_area * 0.46
            )
            weaker = float(cluster["weight"]) < float(nearest["weight"]) * 0.36
            near_body = distance <= max(typical_width * 1.25, pitch * 0.46 if pitch > 0 else 0.0)
            ruby_like = smaller and weaker and near_body and not normal_slot
        if not ruby_like:
            kept.append(cluster)
    if len(kept) < max(1, round(len(strong) * 0.50)):
        kept = strong

    kept.sort(key=lambda value: float(value.get("grid_center", value["center"])))
    return kept, {
        "typical_height": typical_height,
        "typical_width": typical_width,
        "typical_area": typical_area,
        "pitch": pitch,
        "grid_phase": phase,
        "grid_reliable": grid_reliable,
        "component_count": len(components),
        "anchor_count": len(body_anchors),
    }

def _component_content_spans(
    components: Sequence[_InkComponent],
    *,
    typical_height: float,
) -> tuple[tuple[int, int], ...]:
    intervals = sorted((component.top, component.bottom) for component in components)
    if not intervals:
        return ()
    gap_limit = max(4, round(typical_height * 0.95))
    merged: list[list[int]] = [[int(intervals[0][0]), int(intervals[0][1])]]
    for top, bottom in intervals[1:]:
        if int(top) - merged[-1][1] <= gap_limit:
            merged[-1][1] = max(merged[-1][1], int(bottom))
        else:
            merged.append([int(top), int(bottom)])
    return tuple((top, bottom) for top, bottom in merged if bottom > top)



def _component_row_count(
    components: Sequence[_InkComponent],
    *,
    typical_height: float,
) -> int:
    """Estimate printed character rows from component centres, without projection."""
    if not components:
        return 0
    centres = sorted(float(component.center_y) for component in components)
    tolerance = max(2.0, typical_height * 0.46)
    groups: list[list[float]] = []
    for centre in centres:
        if not groups or centre - (sum(groups[-1]) / len(groups[-1])) > tolerance:
            groups.append([centre])
        else:
            groups[-1].append(centre)
    return len(groups)


def _is_probable_detached_diacritic(
    component: _InkComponent,
    anchors: Sequence[_InkComponent],
    *,
    typical_width: float,
    typical_height: float,
    typical_area: float,
) -> bool:
    """Return whether a tiny component may be dakuten/handakuten or punctuation.

    This is deliberately recall-biased.  Ruby removal is reversible at text
    adjudication time; deleting a printed diacritic from every model input is
    not.  A small component is protected when it sits in the same character cell
    and close to a body component, including the common two-detached-dot case.
    """
    if component.area <= 0:
        return False
    if component.height > max(6.0, typical_height * 0.48):
        return False
    if component.width > max(6.0, typical_width * 0.52):
        return False
    if component.area > max(20.0, typical_area * 0.28):
        return False
    vertical_limit = max(3.0, typical_height * 0.48)
    horizontal_limit = max(4.0, typical_width * 0.46)
    for anchor in anchors:
        if anchor is component:
            continue
        vertical_distance = abs(component.center_y - anchor.center_y)
        if vertical_distance > vertical_limit:
            continue
        if component.right < anchor.left:
            horizontal_gap = anchor.left - component.right
        elif component.left > anchor.right:
            horizontal_gap = component.left - anchor.right
        else:
            horizontal_gap = 0
        if horizontal_gap <= horizontal_limit:
            return True
    return False


def _component_full_glyph_bounds(
    components: Sequence[_InkComponent],
    body_components: Sequence[_InkComponent],
    *,
    center: float,
    hard_left: int,
    hard_right: int,
    typical_width: float,
    typical_height: float,
    typical_area: float,
    pitch: float,
    mask: Image.Image | None = None,
    force_full_envelope: bool = False,
) -> tuple[int, int, tuple[tuple[int, int, int, int], ...]]:
    """Return an asymmetric full-glyph body band plus ruby boxes to blank.

    Ordinary OCR must not use the per-character projection path, but a physical
    column still has to contain every printed body stroke.  Earlier portable
    versions rebuilt a *symmetric* rectangle around the estimated grid centre.
    A radical-heavy glyph or an imperfect grid centre could therefore push the
    whole rectangle a few pixels to one side and clip the opposite edge.

    This implementation derives left and right edges independently from repeated
    connected-component evidence in the body rows.  The crop is then expanded by
    a tiny paper-colour clearance, clamped to the neighbour midpoints, and any
    sparse ruby components that remain inside that safe crop are explicitly
    blanked before OCR.  No x/y pixel projection is used here.
    """
    slot_width = max(1.0, float(hard_right - hard_left))
    normal_pitch = bool(
        pitch > 0
        and pitch >= typical_height * 0.72
        and pitch <= typical_height * 2.45
    )

    half_from_height = max(3.0, typical_height * 0.50)
    half_from_pitch = pitch * 0.31 if normal_pitch else half_from_height
    half_from_slot = slot_width * 0.38
    provisional_half = max(3.0, min(half_from_height, half_from_pitch, half_from_slot))

    raw_body_components = list(body_components)
    raw_body_row_count = max(
        1,
        _component_row_count(raw_body_components, typical_height=typical_height),
    )
    row_tolerance = max(2.0, typical_height * 0.46)
    side_band_tolerance = max(1.5, typical_width * 0.28)

    def repeated_band_count(component: _InkComponent) -> int:
        return sum(
            1
            for other in components
            if hard_left <= other.center_x < hard_right
            and abs(other.center_x - component.center_x) <= side_band_tolerance
            and other.area >= 2
        )

    # The initial x-cluster can still contain a small ruby strip when its centre
    # falls just inside the broad cluster tolerance.  Remove only sparse, small
    # side components here; a detached radical is repeated in most body rows and
    # therefore remains authoritative body evidence.
    initial_sparse_floor = max(3, round(raw_body_row_count * 0.60))
    filtered_body_components: list[_InkComponent] = []
    for component in raw_body_components:
        small_side = (
            abs(component.center_x - center) >= max(2.0, typical_width * 0.40)
            and component.height <= typical_height * 0.70
            and component.width <= max(5.0, typical_width * 0.75)
            and component.area <= max(12.0, typical_area * 0.52)
        )
        sparse = repeated_band_count(component) <= initial_sparse_floor
        protected_mark = _is_probable_detached_diacritic(
            component,
            raw_body_components,
            typical_width=typical_width,
            typical_height=typical_height,
            typical_area=typical_area,
        )
        if small_side and sparse and not protected_mark:
            continue
        filtered_body_components.append(component)
    body_components = filtered_body_components or raw_body_components
    body_ids = {id(component) for component in body_components}
    body_intervals = [(component.top, component.bottom) for component in body_components]
    body_row_count = max(
        1,
        _component_row_count(body_components, typical_height=typical_height),
    )

    # Candidate components stay inside one physical slot.  A generous radius is
    # safe because sparse side annotations are classified below and painted out;
    # the neighbour midpoint remains a hard upper bound.
    candidate_radius = min(
        slot_width * 0.47,
        max(provisional_half * 1.34, typical_height * 0.72),
    )
    nearby = [
        component
        for component in components
        if hard_left <= component.center_x < hard_right
        and abs(component.center_x - center) <= candidate_radius
        and component.area >= 2
        and component.height <= max(typical_height * 1.85, 10.0)
    ]

    def overlaps_body_row(component: _InkComponent) -> bool:
        return any(
            max(0, min(component.bottom, bottom) - max(component.top, top))
            >= max(1, round(component.height * 0.20))
            for top, bottom in body_intervals
        )

    # Start with the components that established the physical body column.  Add
    # detached radicals only when their x-band repeats through a substantial
    # portion of the body rows or their own size is body-like.  This is the key
    # distinction from ruby, which is both small and sparse.
    trusted_ids = set(body_ids)
    repeat_floor = max(3, round(body_row_count * 0.54))
    for component in nearby:
        if id(component) in trusted_ids or not overlaps_body_row(component):
            continue
        body_sized = (
            component.height >= typical_height * 0.60
            or component.area >= typical_area * 0.30
        )
        repeated_radical = repeated_band_count(component) >= repeat_floor
        within_body_cell = abs(component.center_x - center) <= max(
            provisional_half * 1.08,
            typical_height * 0.56,
        )
        if within_body_cell and (body_sized or repeated_radical):
            trusted_ids.add(id(component))

    trusted = [component for component in nearby if id(component) in trusted_ids]
    if not trusted:
        trusted = list(body_components)
    if not trusted:
        trusted = [
            component
            for component in nearby
            if abs(component.center_x - center) <= provisional_half
        ]

    # Group trusted fragments into printed character rows.  Row envelopes are
    # robust against an occasional bridge/noise component and provide an
    # asymmetric estimate when the grid centre is biased to one side.
    rows: list[list[_InkComponent]] = []
    for component in sorted(trusted, key=lambda item: item.center_y):
        if not rows:
            rows.append([component])
            continue
        row_center = sum(item.center_y for item in rows[-1]) / len(rows[-1])
        if component.center_y - row_center <= row_tolerance:
            rows[-1].append(component)
        else:
            rows.append([component])

    row_lefts: list[int] = []
    row_rights: list[int] = []
    row_centres: list[float] = []
    max_row_body_width = min(
        max(8.0, typical_height * 1.34, typical_width * 1.50),
        max(8.0, slot_width * 0.82),
    )
    for row in rows:
        anchors = [
            component
            for component in row
            if id(component) in body_ids
            or (
                abs(component.center_x - center) <= max(2.0, provisional_half * 0.72)
                and (
                    component.height >= typical_height * 0.34
                    or component.area >= typical_area * 0.07
                )
            )
        ]
        if not anchors:
            continue
        selected = list(row)
        row_left = min(component.left for component in selected)
        row_right = max(component.right for component in selected)
        # A single accidental bridge must not dictate the body crop.  Reduce an
        # implausibly wide row around the area-weighted anchor centre, but keep
        # normal long punctuation and square kanji cells intact.
        if row_right - row_left > max_row_body_width:
            weight = sum(max(1, component.area) for component in anchors)
            anchor_center = sum(
                component.center_x * max(1, component.area) for component in anchors
            ) / max(1, weight)
            half = max_row_body_width / 2.0
            row_left = max(hard_left, int(math.floor(anchor_center - half)))
            row_right = min(hard_right, int(math.ceil(anchor_center + half)))
        row_lefts.append(int(row_left))
        row_rights.append(int(row_right))
        row_centres.append((float(row_left) + float(row_right)) / 2.0)

    plausible_body = [
        component
        for component in trusted
        if component.width <= max(typical_height * 1.38, typical_width * 1.65)
        and (
            component.height >= typical_height * 0.30
            or component.area >= typical_area * 0.06
        )
    ] or trusted

    body_span = (
        max(component.bottom for component in trusted)
        - min(component.top for component in trusted)
        if trusted else 0
    )
    # A short dialogue fragment can contain one kana, a vertical ellipsis and a
    # closing quote.  The six ellipsis dots are six connected components, so a
    # raw ``len(row_lefts) >= 8`` test incorrectly treats the full-width kana and
    # quote as sparse side outliers.  Short columns are omission-sensitive: use
    # their complete trusted glyph envelope and rely on component-level ruby
    # blanking instead of percentile edge surgery.
    short_or_sparse_column = bool(
        force_full_envelope
        or body_span <= max(96.0, typical_height * 7.25)
        or len(plausible_body) <= 4
    )
    normal_printed_width = max(
        typical_height * 1.16,
        typical_width * 1.48,
    )
    short_full_width_components = [
        component
        for component in plausible_body
        if component.width <= normal_printed_width
        or (
            component.height <= typical_height * 0.45
            and component.width <= slot_width * 0.75
        )
    ] or plausible_body

    if row_lefts:
        if short_or_sparse_column:
            ink_left = min(
                min(row_lefts),
                min((component.left for component in short_full_width_components), default=min(row_lefts)),
            )
            ink_right = max(
                max(row_rights),
                max((component.right for component in short_full_width_components), default=max(row_rights)),
            )
        elif len(row_lefts) >= 8:
            # Sparse ruby/bridge tails normally occur in only a few character
            # rows.  The interquartile body envelope ignores those tails while
            # preserving a consistently asymmetric printed body column.
            ink_left = _percentile(row_lefts, 0.34)
            ink_right = _percentile(row_rights, 0.66)
        else:
            ink_left = min(row_lefts)
            ink_right = max(row_rights)
        body_center = _percentile(
            [int(round(value * 1000.0)) for value in row_centres],
            0.50,
        ) / 1000.0
    elif plausible_body:
        ink_left = min(component.left for component in plausible_body)
        ink_right = max(component.right for component in plausible_body)
        body_center = (
            sum(component.center_x * max(1, component.area) for component in plausible_body)
            / max(1, sum(max(1, component.area) for component in plausible_body))
        )
    else:
        ink_left = center - provisional_half
        ink_right = center + provisional_half
        body_center = center

    ink_left = float(ink_left)
    ink_right = float(ink_right)
    if ink_right <= ink_left:
        ink_left = body_center - 3.0
        ink_right = body_center + 3.0

    # Only a fragmented-stroke page needs a square-cell minimum.  When median
    # component width already resembles character height, the measured envelope
    # is authoritative and must stay close enough to leave hanging punctuation
    # outside the main body box.
    observed_width = max(1.0, ink_right - ink_left)
    fragmented = typical_width < typical_height * 0.45
    minimum_body_width = (
        min(slot_width * 0.62, typical_height * 0.86)
        if fragmented
        else observed_width
    )
    if observed_width < minimum_body_width:
        half = minimum_body_width / 2.0
        ink_left = body_center - half
        ink_right = body_center + half

    clearance = max(1, round(typical_height * 0.060))
    safe_left_limit = hard_left + 1 if hard_right - hard_left >= 5 else hard_left
    safe_right_limit = hard_right - 1 if hard_right - hard_left >= 5 else hard_right
    left = max(safe_left_limit, int(math.floor(ink_left)) - clearance)
    right = min(safe_right_limit, int(math.ceil(ink_right)) + clearance)

    max_crop_width = min(
        max(8, int(round(slot_width - 2))),
        max(10, int(round(typical_height * 1.42)), int(round(typical_width * 1.62))),
    )
    if right - left > max_crop_width:
        left = max(safe_left_limit, int(math.floor(body_center - max_crop_width / 2.0)))
        right = min(safe_right_limit, left + max_crop_width)
        if right - left < max_crop_width:
            left = max(safe_left_limit, right - max_crop_width)

    if right - left < 6:
        need = 6 - (right - left)
        left = max(safe_left_limit, left - math.ceil(need / 2))
        right = min(safe_right_limit, right + math.floor(need / 2))

    observed_half = max(3.0, body_center - float(left), float(right) - body_center)
    sparse_floor = max(3, round(body_row_count * 0.60))
    excluded: list[tuple[int, int, int, int]] = []

    def _tail_looks_like_annotation(
        component: _InkComponent,
        tail_left: int,
        tail_right: int,
    ) -> bool:
        if mask is None or tail_right <= tail_left:
            return False
        top = max(0, min(mask.height, component.top))
        bottom = max(top, min(mask.height, component.bottom))
        tail_left = max(0, min(mask.width, tail_left))
        tail_right = max(tail_left, min(mask.width, tail_right))
        if tail_right <= tail_left or bottom <= top:
            return False
        crop = mask.crop((tail_left, top, tail_right, bottom))
        try:
            raw = crop.tobytes()
            ink_count = sum(1 for value in raw if value)
            if ink_count <= 0:
                return False
            width = crop.width
            active_rows = 0
            for row in range(crop.height):
                segment = raw[row * width:(row + 1) * width]
                if any(segment):
                    active_rows += 1
        finally:
            crop.close()
        return (
            ink_count <= max(12, round(component.area * 0.40))
            and active_rows <= max(4, round(typical_height * 0.74))
        )

    # A ruby block can be connected to its base by a one-pixel scan bridge and
    # therefore share one connected-component box.  Sparse row-edge outliers are
    # cut off at the stable body envelope after a local topology/density check.
    stable_left = int(math.floor(ink_left))
    stable_right = int(math.ceil(ink_right))
    side_extension_floor = max(2, round(typical_width * 0.12))
    effective_lefts: list[int] = []
    effective_rights: list[int] = []
    def _tail_band_repetition(
        component: _InkComponent,
        tail_left: int,
        tail_right: int,
    ) -> int:
        if tail_right <= tail_left:
            return 0
        tail_center = (float(tail_left) + float(tail_right)) / 2.0
        x_tolerance = max(2.0, typical_width * 0.24)
        return sum(
            1
            for other in components
            if other is not component
            and hard_left <= other.center_x < hard_right
            and abs(other.center_x - tail_center) <= x_tolerance
            and other.height <= typical_height * 0.72
            and other.width <= max(6.0, typical_width * 0.78)
            and other.area <= max(14.0, typical_area * 0.56)
            and overlaps_body_row(other)
        )

    # Touching ruby often merges into the base component on *every* annotated
    # row.  In that case there are no separate side components for the function
    # above to count; instead several full-height body components share the same
    # small, low-density extension beyond the stable envelope.  Repetition of
    # that joined extension is strong annotation evidence, whereas a legitimate
    # kana/quote edge in a short column is already part of the full envelope and
    # therefore has no extension to count here.
    repeated_joined_right = sum(
        1
        for other in trusted
        if other.right - stable_right >= side_extension_floor
        and _tail_looks_like_annotation(other, stable_right, other.right)
    )
    repeated_joined_left = sum(
        1
        for other in trusted
        if stable_left - other.left >= side_extension_floor
        and _tail_looks_like_annotation(other, other.left, stable_left)
    )

    for component in trusted:
        core_left = int(component.left)
        core_right = int(component.right)
        # Never cut a normal-width connected printed glyph merely because one
        # side is sparse.  Kana such as ``も`` and vertical quote/bracket forms
        # have legitimate low-density side strokes.  A tail may be removed when
        # the whole component is implausibly wide *or* the same small side band
        # repeats beside other body rows, which is strong ruby evidence even if
        # a one-pixel bridge keeps this one occurrence inside the base component.
        wide_join_candidate = bool(
            component.width > normal_printed_width
            * (1.08 if short_or_sparse_column else 1.0)
        )
        right_repeat = max(
            _tail_band_repetition(component, stable_right, component.right),
            repeated_joined_right,
        )
        left_repeat = max(
            _tail_band_repetition(component, component.left, stable_left),
            repeated_joined_left,
        )
        right_tail = (
            (wide_join_candidate or right_repeat >= 2)
            and component.right - stable_right >= side_extension_floor
            and _tail_looks_like_annotation(component, stable_right, component.right)
        )
        left_tail = (
            (wide_join_candidate or left_repeat >= 2)
            and stable_left - component.left >= side_extension_floor
            and _tail_looks_like_annotation(component, component.left, stable_left)
        )
        if right_tail:
            core_right = min(core_right, stable_right)
            excluded.append((
                max(safe_left_limit, stable_right),
                max(0, component.top - 1),
                min(safe_right_limit, component.right + 1),
                component.bottom + 1,
            ))
        if left_tail:
            core_left = max(core_left, stable_left)
            excluded.append((
                max(safe_left_limit, component.left - 1),
                max(0, component.top - 1),
                min(safe_right_limit, stable_left),
                component.bottom + 1,
            ))
        if core_right > core_left:
            effective_lefts.append(core_left)
            effective_rights.append(core_right)

    # Final omission-safety gate: every trusted body core receives the requested
    # paper clearance.  This expansion is asymmetric and happens after outlier
    # ruby tails have been removed, so a one-sided grid-centre error can never
    # clip a genuine radical or punctuation stroke.
    if effective_lefts and effective_rights:
        required_left = min(effective_lefts) - clearance
        required_right = max(effective_rights) + clearance
        left = max(safe_left_limit, min(left, required_left))
        right = min(safe_right_limit, max(right, required_right))

    observed_half = max(3.0, body_center - float(left), float(right) - body_center)

    for component in components:
        if id(component) in trusted_ids:
            continue
        if component.right <= left or component.left >= right:
            continue
        if _is_probable_detached_diacritic(
            component,
            trusted or body_components,
            typical_width=typical_width,
            typical_height=typical_height,
            typical_area=typical_area,
        ):
            continue
        small = (
            component.height <= typical_height * 0.70
            and component.width <= max(5.0, typical_width * 0.75)
            and component.area <= max(12.0, typical_area * 0.52)
        )
        if not small or not overlaps_body_row(component):
            continue
        side_offset = abs(component.center_x - body_center)
        sparse_side_band = repeated_band_count(component) <= sparse_floor
        if side_offset >= observed_half * 0.54 and sparse_side_band:
            pad = 1
            excluded.append((
                max(left, component.left - pad),
                max(0, component.top - pad),
                min(right, component.right + pad),
                component.bottom + pad,
            ))

    deduped: list[tuple[int, int, int, int]] = []
    for box in sorted(excluded):
        if box not in deduped and box[2] > box[0] and box[3] > box[1]:
            deduped.append(box)
    return int(left), int(right), tuple(deduped)


def _component_punctuation_boxes(
    components: Sequence[_InkComponent],
    body_components: Sequence[_InkComponent],
    *,
    left: int,
    right: int,
    hard_left: int,
    hard_right: int,
    typical_width: float,
    typical_height: float,
    typical_area: float,
) -> tuple[tuple[int, int, int, int], ...]:
    """Recover isolated hanging punctuation by component geometry only.

    Ruby overlaps a base glyph's vertical cell and is rejected.  A legitimate
    hanging comma/full stop occupies an otherwise empty cell and is copied as a
    tiny supplemental box, so the strict body rectangle never needs widening.
    """
    body_intervals = [(item.top, item.bottom) for item in body_components]
    candidates: list[_InkComponent] = []
    for component in components:
        if not (hard_left <= component.center_x < hard_right):
            continue
        if left <= component.center_x < right:
            continue
        horizontal_gap = (
            left - component.right if component.center_x < left else component.left - right
        )
        if horizontal_gap > max(4.0, typical_width * 0.62):
            continue
        if component.area < 2 or component.area > max(10.0, typical_area * 0.42):
            continue
        if component.width > max(5.0, typical_width * 0.55):
            continue
        if component.height > max(6.0, typical_height * 0.72):
            continue
        vertical_overlap = sum(
            max(0, min(component.bottom, bottom) - max(component.top, top))
            for top, bottom in body_intervals
        )
        # Even one side glyph overlapping a body row is much more likely ruby
        # than punctuation.  Detached dakuten inside a body glyph is already part
        # of the body component cluster and never reaches this branch.
        if vertical_overlap > max(1, round(component.height * 0.18)):
            continue
        candidates.append(component)
    if not (0 < len(candidates) <= 2):
        return ()
    vertical_extent = max(item.bottom for item in candidates) - min(item.top for item in candidates)
    if vertical_extent > max(18.0, typical_height * 1.05):
        return ()
    return tuple((item.left, item.top, item.right, item.bottom) for item in candidates)

def _detect_vertical_columns_components(
    image_or_path: Image.Image | str | os.PathLike[str],
    *,
    sensitivity: int = 55,
    padding_percent: int = 10,
    max_columns: int = 80,
    fixed_region_rect: Sequence[float] | None = None,
    fixed_region_already_masked: bool = False,
) -> list[DetectedColumn]:
    """Detect physical columns from connected printed-glyph components.

    Normal OCR never calls the projection splitter or the per-character sweep.
    ``sensitivity`` and ``padding_percent`` remain accepted for settings/API
    compatibility, but no side padding is added because it can reopen ruby.
    """
    del sensitivity, padding_percent
    owns_image = not isinstance(image_or_path, Image.Image)
    image = Image.open(image_or_path).convert("RGB") if owns_image else image_or_path.convert("RGB")
    if fixed_region_rect and not fixed_region_already_masked:
        fixed_masked = _mask_to_fixed_region(image, fixed_region_rect)
        image.close()
        image = fixed_masked
    body_bounds_px = _normalized_body_bounds(image, fixed_region_rect)
    try:
        mask, scale, _threshold = _make_ink_mask(image)
        try:
            components = _connected_ink_components(mask)
            clusters, metrics = _cluster_component_columns(
                components,
                page_width=mask.width,
                page_height=mask.height,
            )
            if not clusters:
                return []
            if len(clusters) > max_columns:
                raise RuntimeError(
                    f"检测到 {len(clusters)} 个候选竖列，超过安全上限 {max_columns}。"
                    "请缩小固定正文框。"
                )

            typical_width = max(3.0, float(metrics.get("typical_width", 3.0) or 3.0))
            typical_height = max(4.0, float(metrics.get("typical_height", 4.0) or 4.0))
            typical_area = max(5.0, float(metrics.get("typical_area", 5.0) or 5.0))
            inv_scale = 1.0 / max(scale, 1e-9)
            if body_bounds_px is not None:
                body_left_px, body_top_px, body_right_px, body_bottom_px = body_bounds_px
                body_left = max(0, min(mask.width - 1, math.floor(body_left_px * scale)))
                body_right = max(body_left + 1, min(mask.width, math.ceil(body_right_px * scale)))
            else:
                body_left_px, body_top_px, body_right_px, body_bottom_px = (
                    0, 0, image.width, image.height
                )
                body_left, body_right = 0, mask.width

            centers = [float(cluster.get("grid_center", cluster["center"])) for cluster in clusters]
            columns: list[DetectedColumn] = []
            for index, cluster in enumerate(clusters):
                center = float(cluster.get("grid_center", cluster["center"]))
                if index == 0:
                    hard_left = body_left
                else:
                    hard_left = round((centers[index - 1] + center) / 2.0)
                if index == len(clusters) - 1:
                    hard_right = body_right
                else:
                    hard_right = round((center + centers[index + 1]) / 2.0)
                hard_left = max(body_left, min(body_right - 1, hard_left))
                hard_right = max(hard_left + 1, min(body_right, hard_right))

                cluster_components = list(cluster.get("components") or [])
                # Wide bridge components sit between two grid centres.  Only
                # components centred on this physical column establish its body
                # strip; recovered short columns still retain their tiny marks.
                centre_tolerance = max(2.0, typical_width * 0.70)
                body_components = [
                    component for component in cluster_components
                    if abs(component.center_x - center) <= centre_tolerance
                ] or cluster_components
                left, right, excluded = _component_full_glyph_bounds(
                    components,
                    body_components,
                    center=center,
                    hard_left=hard_left,
                    hard_right=hard_right,
                    typical_width=typical_width,
                    typical_height=typical_height,
                    typical_area=typical_area,
                    pitch=float(metrics.get("pitch", 0.0) or 0.0),
                    mask=mask,
                    force_full_envelope=bool(cluster.get("recovered_short")),
                )

                supplemental = _component_punctuation_boxes(
                    components,
                    body_components,
                    left=left,
                    right=right,
                    hard_left=hard_left,
                    hard_right=hard_right,
                    typical_width=typical_width,
                    typical_height=typical_height,
                    typical_area=typical_area,
                )
                supplemental_components = [
                    component for component in components
                    if any(
                        component.left == box[0]
                        and component.top == box[1]
                        and component.right == box[2]
                        and component.bottom == box[3]
                        for box in supplemental
                    )
                ]
                span_components = [*body_components, *supplemental_components]
                spans = _component_content_spans(
                    span_components,
                    typical_height=typical_height,
                )
                estimated_chars = max(
                    1,
                    _component_row_count(body_components, typical_height=typical_height)
                    + len(supplemental_components),
                )

                left_px = max(body_left_px, math.floor(left * inv_scale))
                right_px = min(body_right_px, math.ceil(right * inv_scale))
                hard_left_px = max(body_left_px, math.floor(hard_left * inv_scale))
                hard_right_px = min(body_right_px, math.ceil(hard_right * inv_scale))
                if body_bounds_px is not None:
                    top_px = body_top_px
                    bottom_px = body_bottom_px
                else:
                    span_top = min((top for top, _bottom in spans), default=0)
                    span_bottom = max((bottom for _top, bottom in spans), default=mask.height)
                    vertical_pad = max(3, round(mask.height * 0.012))
                    top_px = max(0, math.floor(max(0, span_top - vertical_pad) * inv_scale))
                    bottom_px = min(
                        image.height,
                        math.ceil(min(mask.height, span_bottom + vertical_pad) * inv_scale),
                    )
                if right_px - left_px < 3 or bottom_px - top_px < 3:
                    continue
                content_spans = tuple(
                    (
                        max(body_top_px, math.floor(top * inv_scale)),
                        min(body_bottom_px, math.ceil(bottom * inv_scale)),
                    )
                    for top, bottom in spans
                    if bottom > top
                )
                supplemental_boxes = tuple(
                    (
                        max(body_left_px, math.floor(box[0] * inv_scale)),
                        max(body_top_px, math.floor(box[1] * inv_scale)),
                        min(body_right_px, math.ceil(box[2] * inv_scale)),
                        min(body_bottom_px, math.ceil(box[3] * inv_scale)),
                    )
                    for box in supplemental
                    if box[2] > box[0] and box[3] > box[1]
                )
                columns.append(DetectedColumn(
                    left=left_px,
                    top=top_px,
                    right=right_px,
                    bottom=bottom_px,
                    hard_left=hard_left_px,
                    hard_right=hard_right_px,
                    ink_score=float(sum(component.area for component in body_components)),
                    content_spans=content_spans,
                    estimated_chars=int(estimated_chars),
                    full_height_slot=body_bounds_px is not None,
                    supplemental_boxes=supplemental_boxes,
                    excluded_boxes=tuple(
                        (
                            max(body_left_px, math.floor(box[0] * inv_scale)),
                            max(body_top_px, math.floor(box[1] * inv_scale)),
                            min(body_right_px, math.ceil(box[2] * inv_scale)),
                            min(body_bottom_px, math.ceil(box[3] * inv_scale)),
                        )
                        for box in excluded
                        if box[2] > box[0] and box[3] > box[1]
                    ),
                ))
            if body_bounds_px is None:
                columns = _remove_running_margin_artifacts(
                    columns,
                    page_width=image.width,
                    page_height=image.height,
                )
            return sorted(columns, key=lambda column: (-(column.left + column.right), column.top))
        finally:
            mask.close()
    finally:
        image.close()

def _normalise_column_detector_mode(value: str | None) -> str:
    token = str(value or "components").strip().lower().replace("-", "_")
    if token in {"legacy_projection", "projection", "review_projection"}:
        return "legacy_projection"
    return "components"


def column_detector_version(mode: str | None = None) -> str:
    return (
        LEGACY_PROJECTION_DETECTOR_VERSION
        if _normalise_column_detector_mode(mode) == "legacy_projection"
        else COLUMN_DETECTOR_VERSION
    )


def detect_vertical_columns(
    image_or_path: Image.Image | str | os.PathLike[str],
    *,
    sensitivity: int = 55,
    padding_percent: int = 10,
    max_columns: int = 80,
    fixed_region_rect: Sequence[float] | None = None,
    fixed_region_already_masked: bool = False,
    detector_mode: str = "components",
) -> list[DetectedColumn]:
    """Detect vertical columns using the ordinary no-projection component path.

    ``legacy_projection`` is accepted only for explicit diagnostics/backward
    compatibility.  GUI normal OCR, preview, multi-model OCR and retries always
    use ``components``.  Per-character review projections are generated later by
    ``handwriting_image_tools`` only after the user enables review.
    """
    if _normalise_column_detector_mode(detector_mode) == "legacy_projection":
        return _detect_vertical_columns_projection(
            image_or_path,
            sensitivity=sensitivity,
            padding_percent=padding_percent,
            max_columns=max_columns,
            fixed_region_rect=fixed_region_rect,
            fixed_region_already_masked=fixed_region_already_masked,
        )
    return _detect_vertical_columns_components(
        image_or_path,
        sensitivity=sensitivity,
        padding_percent=padding_percent,
        max_columns=max_columns,
        fixed_region_rect=fixed_region_rect,
        fixed_region_already_masked=fixed_region_already_masked,
    )

_OCR_PLACEHOLDER_CHARS = frozenset("□■◻◼�")


def _merge_ordered_ocr_text(parts: Sequence[str]) -> str:
    """Join ordered OCR blocks without duplicating overlapping fragments.

    Apple Vision/Paddle occasionally return both a whole text span and a
    partially overlapping sub-span.  Exact suffix/prefix overlap of at least
    two characters is safe to collapse; a one-character overlap is deliberately
    retained because consecutive Japanese characters can legitimately repeat.
    """
    merged = ""
    for raw in parts:
        part = str(raw or "").strip()
        if not part:
            continue
        if not merged:
            merged = part
            continue
        if len(part) >= 3 and part == merged[-len(part):]:
            continue
        overlap = 0
        for size in range(min(32, len(merged), len(part)), 1, -1):
            if merged[-size:] == part[:size]:
                overlap = size
                break
        merged += part[overlap:]
    return merged.strip()


def _ordered_authoritative_blocks(
    items: Sequence[dict], image_path: str
) -> list[dict]:
    """Sort blocks top-to-bottom inside one authoritative physical column.

    The common column layer has already fixed reading direction and removed
    neighbouring columns.  External engines are still free to return detected
    fragments in backend order, which is not stable across CPU/MPS/ONNX.  When
    usable geometry exists, sort by vertical position and preserve unboxed
    blocks in their original relative order at the end.
    """
    values = [dict(item) for item in items]
    if len(values) < 2 or not image_path:
        return values
    try:
        with Image.open(image_path) as opened:
            image_size = opened.size
    except Exception:
        return values
    boxed: list[tuple[float, float, int, dict]] = []
    unboxed: list[tuple[int, dict]] = []
    for index, item in enumerate(values):
        rect = _ocr_block_pixel_rect(item, image_size)
        if rect is None:
            unboxed.append((index, item))
            continue
        x1, y1, _x2, _y2 = rect
        boxed.append((y1, x1, index, item))
    if len(boxed) < 2:
        return values
    boxed.sort(key=lambda value: (value[0], value[1], value[2]))
    return [value[3] for value in boxed] + [value[1] for value in unboxed]


def _candidate_text(
    blocks: list[dict] | None,
    *,
    recognition_engine: str = "",
    image_path: str = "",
    authoritative_column: bool = False,
) -> tuple[str, float]:
    items = [dict(item) for item in (blocks or []) if str(item.get("text", "")).strip()]
    engine = str(recognition_engine or "").strip().lower()
    is_apple = engine in {"apple_vision", "macocr", "mac_ocr", "macos_ocr"}

    # Ruby/side text has already been removed from authoritative column images.
    # Re-running the Apple geometry heuristic can wrongly discard a legitimate
    # short dialogue/punctuation block, so keep that legacy filter only for
    # direct non-column callers.
    if is_apple and not authoritative_column:
        items = _filter_apple_vision_blocks(items, image_path=image_path)
    if authoritative_column:
        items = _ordered_authoritative_blocks(items, image_path)

    parts = [str(item.get("text", "")).strip() for item in items]
    text = _merge_ordered_ocr_text(parts)
    if is_apple:
        text = _strip_invalid_macocr_edge_noise(text)

    weighted_total = 0.0
    weight_sum = 0
    for item in items:
        part = "".join(str(item.get("text", "") or "").split())
        if not part:
            continue
        weight = max(1, len(part))
        try:
            value = float(item.get("confidence", 0.9) or 0.0)
        except (TypeError, ValueError, OverflowError):
            value = 0.0
        weighted_total += max(0.0, min(1.0, value)) * weight
        weight_sum += weight
    confidence = round(weighted_total / weight_sum, 12) if weight_sum else 0.0
    compact = "".join(text.split())
    if any(ch in _OCR_PLACEHOLDER_CHARS for ch in compact):
        # A partial/placeholder result may still be useful in manual review, but
        # must never masquerade as a confident automatic candidate.
        confidence = 0.0
    return text, confidence


def _ocr_block_pixel_rect(
    block: dict,
    image_size: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    """Normalize one OCR block box to upper-left pixel coordinates.

    NDLOCR-Lite currently returns pixel quadrilaterals, while some adapters and
    tests use normalized ``bbox`` tuples.  Page-first routing accepts both and
    rejects malformed/zero-area geometry instead of guessing a column.
    """
    width, height = image_size
    value = block.get("box")
    if value is None:
        value = block.get("bbox")
    if value is None:
        value = block.get("boundingBox")
    try:
        if isinstance(value, dict):
            if all(key in value for key in ("x", "y", "width", "height")):
                x1 = float(value["x"]); y1 = float(value["y"])
                x2 = x1 + float(value["width"]); y2 = y1 + float(value["height"])
            elif all(key in value for key in ("left", "top", "right", "bottom")):
                x1 = float(value["left"]); y1 = float(value["top"])
                x2 = float(value["right"]); y2 = float(value["bottom"])
            else:
                return None
        elif isinstance(value, (list, tuple)) and len(value) == 4 and all(
            isinstance(item, (int, float)) for item in value
        ):
            x1, y1, x2, y2 = [float(item) for item in value]
            # ``bbox`` from Apple-style adapters is (x, y, width, height).  Only
            # interpret it that way when the values are clearly normalized.
            if value is block.get("bbox") and max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
                x2 = x1 + x2
                y2 = y1 + y2
        elif isinstance(value, (list, tuple)) and value:
            points = [
                (float(point[0]), float(point[1]))
                for point in value
                if isinstance(point, (list, tuple)) and len(point) >= 2
            ]
            if len(points) < 2:
                return None
            x1 = min(point[0] for point in points); x2 = max(point[0] for point in points)
            y1 = min(point[1] for point in points); y2 = max(point[1] for point in points)
        else:
            return None
    except (TypeError, ValueError, OverflowError):
        return None

    values = (x1, y1, x2, y2)
    if all(-0.05 <= item <= 1.05 for item in values):
        x1 *= width; x2 *= width
        y1 *= height; y2 *= height
    x1, x2 = sorted((max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))))
    y1, y2 = sorted((max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))))
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return x1, y1, x2, y2


_MACOCR_JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff\u3400-\u9fff\uf900-\ufaff]")
_MACOCR_KANA_ONLY_RE = re.compile(r"^[\u3040-\u30ff\u31f0-\u31ffー〜～]+$")
_MACOCR_EDGE_PREFIX_RE = re.compile(
    r"^([\s　）)］\]」』】〕〉》〗〙〛「『【（(［\[|｜丨\-—–─ー・、。…‥:：;；]{2,12})(?=[\u3040-\u30ff\u31f0-\u31ff\u3400-\u9fff\uf900-\ufaff])"
)


def _strip_invalid_macocr_edge_noise(text: str) -> str:
    """Drop only impossible punctuation clusters hallucinated at column edges.

    Legitimate single opening quotes and Japanese dash/ellipsis leaders are
    preserved.  Mixed closing/opening garbage such as ``）「 ）正文`` is removed.
    """
    value = str(text or "").strip()
    match = _MACOCR_EDGE_PREFIX_RE.match(value)
    if not match:
        return value
    prefix = re.sub(r"[\s　]+", "", match.group(1))
    valid_leaders = {"「", "『", "（", "(", "〝", "——", "──", "…", "……", "‥", "ーー"}
    if prefix in valid_leaders:
        return value
    closers = "）)］]」』】〕〉》〗〙〛"
    openers = "「『【（(［[〝"
    first_close = min((prefix.find(ch) for ch in closers if ch in prefix), default=999)
    first_open = min((prefix.find(ch) for ch in openers if ch in prefix), default=999)
    close_count = sum(prefix.count(ch) for ch in closers)
    mixed_invalid = first_close < first_open or close_count >= 2
    symbol_only_noise = len(prefix) >= 3 and any(ch in prefix for ch in "|｜丨")
    if mixed_invalid or symbol_only_noise:
        return value[match.end(1):].lstrip()
    return value


def _filter_apple_vision_blocks(blocks: list[dict], *, image_path: str = "") -> list[dict]:
    """Remove geometric Ruby/side fragments before concatenating Vision blocks."""
    items = [dict(item) for item in blocks if str(item.get("text", "")).strip()]
    if len(items) < 2 or not image_path:
        return items
    try:
        with Image.open(image_path) as image:
            image_size = image.size
    except Exception:
        return items
    geometry = []
    for index, item in enumerate(items):
        rect = _ocr_block_pixel_rect(item, image_size)
        if rect is None:
            continue
        x1, y1, x2, y2 = rect
        text = str(item.get("text", "")).strip()
        jp = len(_MACOCR_JAPANESE_RE.findall(text))
        confidence = max(0.05, float(item.get("confidence", 0.9) or 0.9))
        area = max(1.0, (x2 - x1) * (y2 - y1))
        score = max(1, jp, len(text) // 2) * (area ** 0.5) * confidence
        geometry.append((index, item, rect, text, jp, area, score))
    if len(geometry) < 2:
        return items
    body = max(geometry, key=lambda value: value[-1])
    _body_index, _body_item, (bx1, by1, bx2, by2), _body_text, _body_jp, body_area, _ = body
    body_w = max(1.0, bx2 - bx1)
    body_h = max(1.0, by2 - by1)
    body_cx = (bx1 + bx2) / 2.0
    keep_indices: set[int] = set()
    for index, item, (x1, y1, x2, y2), text, jp, area, _score in geometry:
        if index == body[0]:
            keep_indices.add(index)
            continue
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        cx = (x1 + x2) / 2.0
        overlap = max(0.0, min(x2, bx2) - max(x1, bx1)) / max(1.0, min(width, body_w))
        side = abs(cx - body_cx) > body_w * 0.42 and overlap < 0.58
        kana_only = bool(_MACOCR_KANA_ONLY_RE.fullmatch(text))
        punctuation_only = jp == 0 and not any(ch.isalnum() for ch in text)
        ruby_like = side and kana_only and len(text) <= 20 and (
            width <= body_w * 0.78 or height <= body_h * 0.62 or area <= body_area * 0.42
        )
        side_fragment = side and area <= body_area * 0.24 and len(text) <= 10
        edge_punctuation = punctuation_only and len(text) <= 8 and area <= body_area * 0.30
        if ruby_like or side_fragment or edge_punctuation:
            continue
        keep_indices.add(index)
    # Unboxed observations are retained unless the boxed geometry already found
    # a clear body block and the observation is punctuation-only debris.
    for index, item in enumerate(items):
        if index in keep_indices:
            continue
        if any(entry[0] == index for entry in geometry):
            continue
        text = str(item.get("text", "")).strip()
        if text and (_MACOCR_JAPANESE_RE.search(text) or any(ch.isalnum() for ch in text)):
            keep_indices.add(index)
    filtered = [item for index, item in enumerate(items) if index in keep_indices]
    return filtered or [body[1]]


def _route_page_blocks_to_columns(
    blocks: list[dict] | None,
    columns: Sequence[DetectedColumn],
    image_size: tuple[int, int],
) -> tuple[dict[int, tuple[str, float]], set[int], dict[str, int]]:
    """Route one full-page NDLOCR result into authoritative physical slots.

    The blue fixed region and component-derived physical slots remain the
    source of truth.  A block is accepted only when its geometry belongs clearly
    to one slot.  Wide/ambiguous/unboxed blocks are never spread across columns;
    the affected slots fall back to the existing isolated-column OCR path.
    """
    grouped: dict[int, list[tuple[float, float, str, float]]] = {}
    fallback: set[int] = set()
    stats = {"accepted_blocks": 0, "ambiguous_blocks": 0, "unboxed_blocks": 0}
    if not columns:
        return {}, fallback, stats
    widths = sorted(max(1, column.hard_right - column.hard_left) for column in columns)
    median_width = float(widths[len(widths) // 2])

    for block in blocks or []:
        text = str(block.get("text") or "").strip()
        if not text:
            continue
        rect = _ocr_block_pixel_rect(block, image_size)
        if rect is None:
            stats["unboxed_blocks"] += 1
            continue
        x1, y1, x2, _y2 = rect
        block_width = max(1.0, x2 - x1)
        overlaps: list[tuple[float, int]] = []
        for index, column in enumerate(columns):
            overlap = max(0.0, min(x2, float(column.hard_right)) - max(x1, float(column.hard_left)))
            if overlap > 0:
                overlaps.append((overlap, index))
        if not overlaps:
            continue
        overlaps.sort(reverse=True)
        best_overlap, best_index = overlaps[0]
        second_overlap = overlaps[1][0] if len(overlaps) > 1 else 0.0
        center_x = (x1 + x2) / 2.0
        chosen = columns[best_index]
        center_inside = float(chosen.hard_left) <= center_x <= float(chosen.hard_right)
        wide = block_width > median_width * 1.20
        split_across_slots = second_overlap >= max(2.0, best_overlap * 0.42)
        weak_membership = best_overlap / block_width < 0.50 and not center_inside
        if (wide and split_across_slots) or weak_membership:
            stats["ambiguous_blocks"] += 1
            for overlap, index in overlaps:
                if overlap >= max(2.0, best_overlap * 0.30):
                    fallback.add(index)
            continue
        confidence = float(block.get("confidence", 0.9) or 0.9)
        grouped.setdefault(best_index, []).append((y1, x1, text, confidence))
        stats["accepted_blocks"] += 1

    routed: dict[int, tuple[str, float]] = {}
    for index, items in grouped.items():
        if index in fallback:
            continue
        items.sort(key=lambda item: (item[0], item[1]))
        text = "".join(item[2] for item in items).strip()
        if not text:
            continue
        weight_total = sum(max(1, len(item[2])) for item in items)
        confidence = sum(
            item[3] * max(1, len(item[2])) for item in items
        ) / max(1, weight_total)
        routed[index] = (text, confidence)
    return routed, fallback, stats


_NDLOCR_PAGE_MODES = {"column", "page", "hybrid"}


def _normalise_ndlocr_page_mode(
    value: object,
    *,
    legacy_page_batch: object = True,
) -> str:
    """Return a stable NDLOCR page/column strategy name.

    ``column_ndlocr_page_batch`` existed briefly as a boolean safety switch.
    Keep accepting it so saved settings, direct callers and older plugins do not
    break, while the GUI can expose the clearer three-mode strategy.
    """
    mode = str(value or "").strip().lower()
    aliases = {
        "auto": "hybrid",
        "smart": "hybrid",
        "mixed": "hybrid",
        "page_first": "hybrid",
        "page-only": "page",
        "page_only": "page",
        "full_page": "page",
        "per_column": "column",
        "column_only": "column",
        "legacy": "column",
    }
    mode = aliases.get(mode, mode)
    if mode in _NDLOCR_PAGE_MODES:
        return mode
    return "hybrid" if bool(legacy_page_batch) else "column"


def _valid_column_text(text: str) -> bool:
    value = str(text or "").strip()
    if not value or len(value) > 320:
        return False
    return not is_spurious_ocr_item(value)


def _paper_background(image: Image.Image) -> tuple[int, int, int]:
    """Estimate paper colour from near-white page-edge samples."""
    rgb = image.convert("RGB")
    try:
        width, height = rgb.size
        samples: list[tuple[int, int, int]] = []
        step_x = max(1, width // 40)
        step_y = max(1, height // 40)
        border = max(2, round(min(width, height) * 0.025))
        for y in range(0, height, step_y):
            for x in range(0, width, step_x):
                if x < border or x >= width - border or y < border or y >= height - border:
                    pixel = rgb.getpixel((x, y))
                    if sum(pixel) >= 570:
                        samples.append(pixel)
        if not samples:
            return (255, 255, 255)
        samples.sort(key=sum)
        keep = samples[len(samples) // 2:]
        return tuple(round(sum(pixel[i] for pixel in keep) / len(keep)) for i in range(3))
    finally:
        if rgb is not image:
            rgb.close()


def _normalized_body_bounds(
    image: Image.Image,
    fixed_region_rect: Sequence[float] | None,
) -> tuple[int, int, int, int] | None:
    """Convert a normalized GUI crop rectangle to clamped pixel bounds."""
    if not fixed_region_rect or len(fixed_region_rect) < 4:
        return None
    try:
        x0, y0, x1, y1 = [float(value) for value in fixed_region_rect[:4]]
    except (TypeError, ValueError):
        return None
    width, height = image.size
    x0 = max(0.0, min(1.0, x0))
    y0 = max(0.0, min(1.0, y0))
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    left = max(0, min(width - 1, int(round(width * min(x0, x1)))))
    top = max(0, min(height - 1, int(round(height * min(y0, y1)))))
    right = max(left + 1, min(width, int(round(width * max(x0, x1)))))
    bottom = max(top + 1, min(height, int(round(height * max(y0, y1)))))
    return left, top, right, bottom


def _mask_to_fixed_region(
    image: Image.Image,
    fixed_region_rect: Sequence[float] | None,
) -> Image.Image:
    """Return an RGB page with all pixels outside the fixed body made paper-white.

    The common OCR layer already performs this masking.  Reapplying it here is
    intentional defence in depth: adaptive retries and sentence-context OCR must
    never be able to reopen an unmasked original page and reintroduce running
    headers, page numbers or footer text.
    """
    rgb = image.convert("RGB")
    bounds = _normalized_body_bounds(rgb, fixed_region_rect)
    if bounds is None:
        return rgb
    canvas = Image.new("RGB", rgb.size, _paper_background(rgb))
    region = rgb.crop(bounds)
    try:
        canvas.paste(region, (bounds[0], bounds[1]))
    finally:
        region.close()
        rgb.close()
    return canvas


def _open_page_image(
    page_path: str,
    fixed_region_rect: Sequence[float] | None = None,
) -> Image.Image:
    with Image.open(page_path) as source:
        image = source.convert("RGB")
    if fixed_region_rect:
        masked = _mask_to_fixed_region(image, fixed_region_rect)
        image.close()
        return masked
    return image


def _column_source_boxes(column: DetectedColumn) -> list[tuple[int, int, int, int]]:
    boxes = [(int(column.left), int(column.top), int(column.right), int(column.bottom))]
    boxes.extend(
        tuple(int(value) for value in box[:4])
        for box in (getattr(column, "supplemental_boxes", ()) or ())
    )
    return [box for box in boxes if box[2] > box[0] and box[3] > box[1]]


def _column_source_union(column: DetectedColumn) -> tuple[int, int, int, int]:
    boxes = _column_source_boxes(column)
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _paste_source_boxes(
    destination: Image.Image,
    source: Image.Image,
    boxes: Sequence[tuple[int, int, int, int]],
    *,
    offset_x: int = 0,
    offset_y: int = 0,
) -> None:
    for raw_left, raw_top, raw_right, raw_bottom in boxes:
        left = max(0, min(source.width - 1, int(raw_left)))
        right = max(left + 1, min(source.width, int(raw_right)))
        top = max(0, min(source.height - 1, int(raw_top)))
        bottom = max(top + 1, min(source.height, int(raw_bottom)))
        region = source.crop((left, top, right, bottom)).convert("RGB")
        try:
            destination.paste(region, (left - offset_x, top - offset_y))
        finally:
            region.close()


def _column_exclusion_boxes(column: DetectedColumn) -> list[tuple[int, int, int, int]]:
    return [
        tuple(int(value) for value in box[:4])
        for box in (getattr(column, "excluded_boxes", ()) or ())
        if len(box) >= 4 and int(box[2]) > int(box[0]) and int(box[3]) > int(box[1])
    ]


def _blank_destination_boxes(
    destination: Image.Image,
    boxes: Sequence[tuple[int, int, int, int]],
    *,
    offset_x: int = 0,
    offset_y: int = 0,
    background: tuple[int, int, int] = (255, 255, 255),
) -> None:
    for raw_left, raw_top, raw_right, raw_bottom in boxes:
        left = max(0, min(destination.width, int(raw_left) - offset_x))
        right = max(left, min(destination.width, int(raw_right) - offset_x))
        top = max(0, min(destination.height, int(raw_top) - offset_y))
        bottom = max(top, min(destination.height, int(raw_bottom) - offset_y))
        if right > left and bottom > top:
            destination.paste(background, (left, top, right, bottom))


def _paste_column_source(
    destination: Image.Image,
    source: Image.Image,
    column: DetectedColumn,
    *,
    offset_x: int = 0,
    offset_y: int = 0,
    background: tuple[int, int, int] | None = None,
    erase_exclusions: bool = True,
) -> None:
    """Paste one detected column into ``destination``.

    ``erase_exclusions`` keeps the old review/debug Ruby-removal behaviour.
    Production OCR passes ``False`` (or uses the raw hard-slot path) so no
    component inside a physical column can be painted over before recognition.
    """
    _paste_source_boxes(
        destination,
        source,
        _column_source_boxes(column),
        offset_x=offset_x,
        offset_y=offset_y,
    )
    if erase_exclusions:
        _blank_destination_boxes(
            destination,
            _column_exclusion_boxes(column),
            offset_x=offset_x,
            offset_y=offset_y,
            background=background or _paper_background(source),
        )

def _masked_column_image(
    image: Image.Image,
    column: DetectedColumn,
    *,
    span: tuple[int, int] | None = None,
    retry: bool = False,
    background: tuple[int, int, int] | None = None,
    preserve_body_pixels: bool = True,
) -> Image.Image:
    """Keep one physical column on a full-size paper-colour canvas.

    The production default copies the complete logical hard slot directly from
    the fixed-region source page.  It never applies ``excluded_boxes`` or a
    density-derived body-core cut, so detached dakuten/handakuten and thin
    printed strokes remain available to every OCR model.  The old filtered body
    band is retained only for explicit review/debug calls.
    """
    paper = background or _paper_background(image)
    canvas = Image.new("RGB", image.size, paper)
    if preserve_body_pixels:
        left, right = int(column.hard_left), int(column.hard_right)
    else:
        left, right = int(column.left), int(column.right)
    if span is not None and not retry:
        span_top, span_bottom = span
        context = max(10, round(max(1, column.width) * 1.15))
        top = max(int(column.top), span_top - context) if column.full_height_slot else max(0, span_top - context)
        bottom = min(int(column.bottom), span_bottom + context) if column.full_height_slot else min(image.height, span_bottom + context)
    elif column.full_height_slot:
        top = int(column.top)
        bottom = int(column.bottom)
    else:
        context_scale = 2.4 if retry else 1.35
        context = max(14, round(max(1, column.width) * context_scale))
        top = max(0, int(column.top) - context)
        bottom = min(image.height, int(column.bottom) + context)
    left = max(0, min(image.width - 1, left))
    right = max(left + 1, min(image.width, right))
    top = max(0, min(image.height - 1, int(top)))
    bottom = max(top + 1, min(image.height, int(bottom)))
    if preserve_body_pixels:
        region = image.crop((left, top, right, bottom)).convert("RGB")
        try:
            canvas.paste(region, (left, top))
        finally:
            region.close()
    else:
        _paste_column_source(
            canvas,
            image,
            column,
            background=paper,
            erase_exclusions=True,
        )
    return canvas

class _SentencePageCache:
    """Small LRU of fixed-region page images for ordered sentence rendering."""

    def __init__(self, fixed_region_rect: Sequence[float] | None, max_pages: int = 4):
        self.fixed_region_rect = fixed_region_rect
        self.max_pages = max(1, int(max_pages or 1))
        self._images: OrderedDict[str, Image.Image] = OrderedDict()
        self._backgrounds: dict[str, tuple[int, int, int]] = {}

    def get(self, page_path: str) -> Image.Image:
        image = self._images.pop(page_path, None)
        if image is None:
            image = _open_page_image(page_path, self.fixed_region_rect)
        self._images[page_path] = image
        while len(self._images) > self.max_pages:
            old_path, old_image = self._images.popitem(last=False)
            old_image.close()
            self._backgrounds.pop(old_path, None)
        return image

    def background(self, page_path: str, image: Image.Image) -> tuple[int, int, int]:
        value = self._backgrounds.get(page_path)
        if value is None:
            value = _paper_background(image)
            self._backgrounds[page_path] = value
        return value

    def close(self) -> None:
        for image in self._images.values():
            image.close()
        self._images.clear()
        self._backgrounds.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def _sentence_group_image(
    page_columns: dict[str, list[DetectedColumn]],
    targets: Sequence[tuple[str, int]],
    *,
    fixed_region_rect: Sequence[float] | None = None,
    shared_page_cache: _SentencePageCache | None = None,
    preserve_body_pixels: bool = True,
) -> Image.Image:
    """Assemble original-pixel columns into one context image."""
    if not targets:
        raise ValueError("句组不能为空")

    strips: list[Image.Image] = []
    widths: list[int] = []
    max_height = 0
    background = (255, 255, 255)
    local_page_cache: dict[str, Image.Image] = {}
    try:
        for position, (page_path, column_index) in enumerate(targets):
            if shared_page_cache is not None:
                image = shared_page_cache.get(page_path)
            else:
                image = local_page_cache.get(page_path)
                if image is None:
                    image = _open_page_image(page_path, fixed_region_rect)
                    local_page_cache[page_path] = image
            if position == 0:
                background = (
                    shared_page_cache.background(page_path, image)
                    if shared_page_cache is not None
                    else _paper_background(image)
                )
            column = page_columns[page_path][column_index]
            if preserve_body_pixels:
                left = max(0, min(image.width - 1, int(column.hard_left)))
                right = max(left + 1, min(image.width, int(column.hard_right)))
                if column.full_height_slot:
                    top = max(0, int(column.top))
                    bottom = min(image.height, int(column.bottom))
                else:
                    vertical_context = max(14, round(max(1, column.width) * 1.35))
                    top = max(0, int(column.top) - vertical_context)
                    bottom = min(image.height, int(column.bottom) + vertical_context)
                strip = image.crop((left, top, right, bottom)).convert("RGB")
            else:
                union_left, union_top, union_right, union_bottom = _column_source_union(column)
                left = max(0, min(image.width - 1, union_left))
                right = max(left + 1, min(image.width, union_right))
                if column.full_height_slot:
                    top = max(0, min(int(column.top), union_top))
                    bottom = min(image.height, max(int(column.bottom), union_bottom))
                else:
                    vertical_context = max(14, round(max(1, column.width) * 1.35))
                    top = max(0, min(int(column.top), union_top) - vertical_context)
                    bottom = min(image.height, max(int(column.bottom), union_bottom) + vertical_context)
                strip = Image.new("RGB", (right - left, bottom - top), background)
                _paste_column_source(
                    strip, image, column, offset_x=left, offset_y=top,
                    background=background, erase_exclusions=True,
                )
            strips.append(strip)
            widths.append(strip.width)
            max_height = max(max_height, strip.height)

        typical_width = sorted(widths)[len(widths) // 2]
        gap = max(6, round(typical_width * 0.42))
        margin_x = max(10, round(typical_width * 0.55))
        margin_y = max(8, round(typical_width * 0.35))
        canvas_width = sum(widths) + gap * max(0, len(strips) - 1) + margin_x * 2
        canvas_height = max_height + margin_y * 2
        canvas = Image.new("RGB", (canvas_width, canvas_height), background)
        cursor = canvas_width - margin_x
        for strip in strips:
            cursor -= strip.width
            canvas.paste(strip, (cursor, margin_y))
            cursor -= gap
        return canvas
    finally:
        for strip in strips:
            strip.close()
        for image in local_page_cache.values():
            image.close()

def _sentence_group_page_runs(
    targets: Sequence[tuple[str, int]],
) -> list[tuple[str, list[int]]]:
    """Split one logical sentence group into contiguous per-page column runs."""
    runs: list[tuple[str, list[int]]] = []
    for page_path, column_index in targets:
        if runs and runs[-1][0] == page_path:
            runs[-1][1].append(column_index)
        else:
            runs.append((page_path, [column_index]))
    return runs


def _sentence_group_merged_box_image(
    page_columns: dict[str, list[DetectedColumn]],
    targets: Sequence[tuple[str, int]],
    *,
    fixed_region_rect: Sequence[float] | None = None,
    shared_page_cache: _SentencePageCache | None = None,
    preserve_body_pixels: bool = True,
) -> Image.Image:
    """Build a context image while keeping same-page column spacing."""
    if not targets:
        raise ValueError("句组不能为空")

    regions: list[Image.Image] = []
    widths: list[int] = []
    typical_column_widths: list[int] = []
    local_page_cache: dict[str, Image.Image] = {}
    background = (255, 255, 255)
    max_height = 0
    try:
        runs = _sentence_group_page_runs(targets)
        for run_index, (page_path, indices) in enumerate(runs):
            if shared_page_cache is not None:
                image = shared_page_cache.get(page_path)
            else:
                image = local_page_cache.get(page_path)
                if image is None:
                    image = _open_page_image(page_path, fixed_region_rect)
                    local_page_cache[page_path] = image
            if run_index == 0:
                background = (
                    shared_page_cache.background(page_path, image)
                    if shared_page_cache is not None
                    else _paper_background(image)
                )
            columns = [page_columns[page_path][index] for index in indices]
            if preserve_body_pixels:
                run_widths = [max(1, column.hard_right - column.hard_left) for column in columns]
                typical_column_widths.extend(run_widths)
                typical_width = sorted(run_widths)[len(run_widths) // 2]
                exact_slots = bool(columns and all(column.full_height_slot for column in columns))
                extra_x = 0 if exact_slots else max(3, round(typical_width * 0.16))
                extra_y = 0 if exact_slots else max(8, round(typical_width * 0.55))
                left = max(0, min(column.hard_left for column in columns) - extra_x)
                right = min(image.width, max(column.hard_right for column in columns) + extra_x)
                top = max(0, min(column.top for column in columns) - extra_y)
                bottom = min(image.height, max(column.bottom for column in columns) + extra_y)
            else:
                run_widths = [max(1, column.right - column.left) for column in columns]
                typical_column_widths.extend(run_widths)
                typical_width = sorted(run_widths)[len(run_widths) // 2]
                exact_slots = bool(columns and all(column.full_height_slot for column in columns))
                extra_x = 0 if exact_slots else max(3, round(typical_width * 0.16))
                extra_y = 0 if exact_slots else max(8, round(typical_width * 0.55))
                unions = [_column_source_union(column) for column in columns]
                left = max(0, min(box[0] for box in unions) - extra_x)
                right = min(image.width, max(box[2] for box in unions) + extra_x)
                top = max(0, min(box[1] for box in unions) - extra_y)
                bottom = min(image.height, max(box[3] for box in unions) + extra_y)
            body_bounds = _normalized_body_bounds(image, fixed_region_rect)
            if body_bounds is not None:
                body_left, body_top, body_right, body_bottom = body_bounds
                left = max(left, body_left); right = min(right, body_right)
                top = max(top, body_top); bottom = min(bottom, body_bottom)
            if right <= left or bottom <= top:
                raise ValueError("句组合并框无有效像素区域")
            if preserve_body_pixels:
                region = image.crop((left, top, right, bottom)).convert("RGB")
            else:
                region = Image.new("RGB", (right - left, bottom - top), background)
                for column in columns:
                    _paste_column_source(
                        region, image, column, offset_x=left, offset_y=top,
                        background=background, erase_exclusions=True,
                    )
            regions.append(region)
            widths.append(region.width)
            max_height = max(max_height, region.height)

        typical_width = (
            sorted(typical_column_widths)[len(typical_column_widths) // 2]
            if typical_column_widths else 24
        )
        run_gap = max(8, round(typical_width * 0.72))
        margin_x = max(10, round(typical_width * 0.60))
        margin_y = max(8, round(typical_width * 0.42))
        canvas_width = sum(widths) + run_gap * max(0, len(regions) - 1) + margin_x * 2
        canvas_height = max_height + margin_y * 2
        canvas = Image.new("RGB", (canvas_width, canvas_height), background)
        cursor = canvas_width - margin_x
        for region in regions:
            cursor -= region.width
            canvas.paste(region, (cursor, margin_y))
            cursor -= run_gap
        return canvas
    finally:
        for region in regions:
            region.close()
        for image in local_page_cache.values():
            image.close()

def _sentence_group_cache_key(
    targets: Sequence[tuple[str, int]],
    page_number_by_path: dict[str, int],
    *,
    global_merged: bool,
    fixed_region_rect: Sequence[float] | None,
) -> str:
    payload = {
        "targets": [
            [int(page_number_by_path.get(page_path, 0)), int(column_index)]
            for page_path, column_index in targets
        ],
        "global_merged": bool(global_merged),
        "fixed_region_rect": [round(float(value), 7) for value in (fixed_region_rect or [])],
    }
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


_SENTENCE_GROUP_TITLE_RE = re.compile(
    r"^(?:序章|終章|プロローグ|エピローグ|後記|あとがき|幕間|"
    r"第[一二三四五六七八九十百千〇零\d０-９]+[章話節巻回幕篇編]|"
    r"(?:Chapter|Episode|EP)\s*[.．_-]*[\d０-９]+)",
    re.I,
)


def _looks_sentence_group_title(text: str) -> bool:
    value = normalize_column_text(text)
    return bool(value and len(value) <= 100 and _SENTENCE_GROUP_TITLE_RE.match(value))


def _sentence_reocr_groups(
    ordered_targets: Sequence[tuple[str, int]],
    recognized: dict[tuple[str, int], tuple[str, float, str | None]],
    *,
    max_columns: int,
) -> list[list[tuple[str, int]]]:
    """Return complete 2+ column sentence groups using first-pass OCR tails."""
    groups: list[list[tuple[str, int]]] = []
    pending: list[tuple[str, int]] = []
    max_columns = max(2, int(max_columns or 2))

    def text_for(target: tuple[str, int]) -> str:
        result = recognized.get(target)
        return normalize_column_text(result[0] if result else "□")

    def flush_complete() -> None:
        nonlocal pending
        if len(pending) >= 2:
            groups.append(list(pending))
        pending = []

    for target in ordered_targets:
        text = text_for(target)
        if _looks_sentence_group_title(text):
            pending = []
            continue

        if pending:
            pending_text = join_column_parts(text_for(item) for item in pending)
            if is_provisional_quote_terminal(pending_text) and not starts_post_quote_continuation(text):
                flush_complete()

        pending.append(target)
        if is_provisional_quote_terminal(text):
            # One-column lookahead is required for ``「行く」と彼は言った。``.
            continue
        if has_sentence_terminal(text):
            flush_complete()
            continue
        if len(pending) >= max_columns:
            # No visible sentence end: keep the safe first-pass result rather
            # than asking OCR to interpret a very large, ambiguous canvas.
            pending = []

    # An unfinished document tail is deliberately not re-recognized.  The user
    # asked for context OCR only after a visible sentence terminal is reached.
    return groups


_LATIN_WORD_RE = re.compile(r"[A-Za-z]{2,}")
_KANA_ONLY_RE = re.compile(r"^[\u3040-\u30ffー〜～]+$")
_EDGE_DEBRIS_TRIM = " \t\r\n　-—–_|/\\:;,.，。!?！？()[]{}（）【】『』「」<>《》〈〉=＋+*~〜～"


def _japanese_positions(value: str) -> list[int]:
    return [
        index for index, char in enumerate(value)
        if (
            "\u3040" <= char <= "\u30ff"
            or "\u31f0" <= char <= "\u31ff"
            or "\u3400" <= char <= "\u9fff"
            or "\uf900" <= char <= "\ufaff"
        )
    ]


def _suspicious_edge_debris_affixes(text: str) -> tuple[str, str]:
    """Return non-body prefix/suffix hallucinations around Japanese prose.

    NDLOCR sentence re-recognition can interpret blank margins, crop seams, page
    numbers or tiny edge marks as Latin words, long digit runs or symbol strings.
    The candidate is compared with the per-column baseline before rejection, so
    genuine mixed-language or numeric content already seen in the source remains.
    """
    value = normalize_column_text(text)
    positions = _japanese_positions(value)
    if not positions:
        return "", ""

    def suspicious(part: str) -> str:
        compact = str(part or "").strip(_EDGE_DEBRIS_TRIM)
        if not compact or len(compact) > 120 or _japanese_positions(compact):
            return ""
        words = _LATIN_WORD_RE.findall(compact)
        letter_count = sum(len(word) for word in words)
        digit_count = sum(char.isdigit() for char in compact)
        ascii_payload = sum(
            char.isascii() and (char.isalnum() or char in ",._:;|/\\-+=*~")
            for char in compact
        )
        if len(words) >= 2 and letter_count >= 10:
            return compact
        if digit_count >= 2:
            return compact
        if len(compact) >= 5 and ascii_payload / max(1, len(compact)) >= 0.85:
            return compact
        return ""

    first, last = positions[0], positions[-1]
    return suspicious(value[:first]), suspicious(value[last + 1:])


def _affix_matches_baseline(candidate_affix: str, baseline_affix: str) -> bool:
    def compact(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "", str(value or "")).casefold()

    candidate_key = compact(candidate_affix)
    baseline_key = compact(baseline_affix)
    if not candidate_key or not baseline_key:
        return False
    if candidate_key == baseline_key:
        return True
    return SequenceMatcher(None, candidate_key, baseline_key, autojunk=False).ratio() >= 0.84


def _new_ndlocr_kana_annotation(baseline: str, candidate: str) -> str:
    """Find candidate-only short kana runs that resemble misplaced ruby text.

    This guard is deliberately limited to NDLOCR sentence re-OCR.  The first-pass
    per-column result is kept when the second pass inserts a compact kana reading
    while the surrounding sentence otherwise closely matches the baseline.
    """
    base = normalize_column_text(baseline)
    value = normalize_column_text(candidate)
    if not base or not value:
        return ""
    matcher = SequenceMatcher(None, base, value, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "insert":
            continue
        inserted = value[j1:j2].strip(_EDGE_DEBRIS_TRIM)
        if not (2 <= len(inserted) <= 12 and _KANA_ONLY_RE.fullmatch(inserted)):
            continue
        # Ruby text normally sits beside a kanji run and appears as a compact
        # insertion.  Require at least one CJK neighbour to avoid classifying an
        # ordinary ASCII-space correction as annotation debris.
        left = value[j1 - 1] if j1 > 0 else ""
        right = value[j2] if j2 < len(value) else ""
        cjk_neighbour = any("\u3400" <= char <= "\u9fff" for char in (left, right))
        if not cjk_neighbour and j1 == 0 and right:
            cjk_neighbour = "\u3400" <= right <= "\u9fff"
        if not cjk_neighbour:
            continue
        without = value[:j1] + value[j2:]
        before = matcher.ratio()
        after = SequenceMatcher(None, base, without, autojunk=False).ratio()
        if after >= 0.72 and after >= before + 0.035:
            return inserted
    return ""


def _validate_sentence_reocr_candidate(
    primary_parts: Sequence[str],
    candidate: str,
    confidence: float,
    *,
    recognition_engine: str = "",
) -> tuple[bool, str, str, str]:
    """Conservative quality gate for replacing joined per-column OCR text."""
    baseline = join_column_parts(primary_parts)
    value = normalize_column_text(candidate)
    if not value:
        return False, "句组 OCR 返回空文本", baseline, value
    if len(value) > 4096 or is_spurious_ocr_item(value):
        return False, "句组 OCR 返回异常文本", baseline, value
    if not has_sentence_terminal(value):
        return False, "句组 OCR 丢失了可见句末标点", baseline, value
    if float(confidence or 0.0) < 0.18:
        return False, "句组 OCR 置信度过低", baseline, value
    if _japanese_character_ratio(value) < 0.42:
        return False, "句组 OCR 日文字符比例异常", baseline, value

    candidate_prefix, candidate_suffix = _suspicious_edge_debris_affixes(value)
    baseline_prefix, baseline_suffix = _suspicious_edge_debris_affixes(baseline)
    if candidate_prefix and not _affix_matches_baseline(candidate_prefix, baseline_prefix):
        return False, "句组 OCR 新增疑似页码/数字/英文页边伪文字前缀", baseline, value
    if candidate_suffix and not _affix_matches_baseline(candidate_suffix, baseline_suffix):
        return False, "句组 OCR 新增疑似页码/数字/英文页边伪文字后缀", baseline, value

    if str(recognition_engine or "").strip().lower() == "ndlocr_lite":
        kana_annotation = _new_ndlocr_kana_annotation(baseline, value)
        if kana_annotation:
            return False, "NDLOCR 句组 OCR 新增疑似错位注音文字", baseline, value

    baseline_clean = baseline.replace("□", "")
    if baseline_clean:
        base_len = _compact_text_length(baseline_clean)
        value_len = _compact_text_length(value)
        ratio = value_len / max(1, base_len)
        if base_len >= 6 and not (0.58 <= ratio <= 1.48):
            return False, "句组 OCR 长度与逐列底稿差异过大", baseline, value
        similarity = SequenceMatcher(None, baseline_clean, value, autojunk=False).ratio()
        floor = 0.28 if "□" in baseline else 0.44
        if base_len >= 8 and similarity < floor:
            return False, "句组 OCR 与逐列底稿差异过大", baseline, value
    if _quote_balance_penalty(value) > _quote_balance_penalty(baseline) + 1:
        return False, "句组 OCR 引号失衡加重", baseline, value
    if value.count("□") > baseline.count("□"):
        return False, "句组 OCR 增加了未识别占位符", baseline, value
    return True, "通过句末、长度、相似度与符号安全校验", baseline, value


def _crop_column(
    image: Image.Image, column: DetectedColumn, *, retry: bool,
    background: tuple[int, int, int] | None = None,
    preserve_body_pixels: bool = True,
) -> Image.Image:
    """Backward-compatible helper returning a lossless masked page."""
    return _masked_column_image(
        image, column, retry=retry, background=background,
        preserve_body_pixels=preserve_body_pixels,
    )

def _crop_content_span(
    image: Image.Image, column: DetectedColumn, span: tuple[int, int], *,
    background: tuple[int, int, int] | None = None,
    preserve_body_pixels: bool = True,
) -> Image.Image:
    """Return a full-size masked page that exposes only one short block."""
    return _masked_column_image(
        image, column, span=span, retry=False, background=background,
        preserve_body_pixels=preserve_body_pixels,
    )

def _trim_region_to_primary_ink(region: Image.Image) -> Image.Image:
    """Conservatively trim side debris from a compact column crop.

    Column slots can be slightly wider than the true printed text band.  For
    Manga OCR this may expose punctuation or thin strokes from a neighbour
    column, which the decoder then treats as extra characters.  Keep only the
    dominant vertical ink run and a tiny safety pad; never rescale glyphs.
    """
    if region.width < 28 or region.height < 48:
        return region.copy()
    gray = region.convert("L")
    try:
        mask = gray.point(lambda p: 255 if p < 208 else 0, mode="L")
    finally:
        gray.close()
    try:
        if mask.getbbox() is None:
            return region.copy()
        collapsed = mask.resize((region.width, 1), Image.Resampling.BOX)
        try:
            flattened = getattr(collapsed, "get_flattened_data", None)
            projection = list(flattened() if callable(flattened) else collapsed.getdata())
        finally:
            collapsed.close()
        peak = max(projection or [0])
        threshold = max(6, int(peak * 0.18))
        runs = []
        start = None
        score = 0
        for idx, value in enumerate(projection + [0]):
            if value >= threshold:
                if start is None:
                    start = idx
                    score = 0
                score += int(value)
            elif start is not None:
                runs.append((start, idx, score))
                start = None
        min_run_width = max(3, round(region.width * 0.08))
        runs = [run for run in runs if (run[1] - run[0]) >= min_run_width]
        if not runs:
            bbox = mask.getbbox()
            if bbox is None:
                return region.copy()
            left, top, right, bottom = bbox
        else:
            left_run, right_run, _ = max(runs, key=lambda item: (item[2], item[1] - item[0]))
            x_pad = max(2, round(region.width * 0.04))
            left = max(0, left_run - x_pad)
            right = min(region.width, right_run + x_pad)
            focused = mask.crop((left, 0, right, region.height))
            try:
                bbox = focused.getbbox()
            finally:
                focused.close()
            if bbox is None:
                top, bottom = 0, region.height
            else:
                _, top, _, bottom = bbox
                y_pad = max(3, round(region.width * 0.10))
                top = max(0, top - y_pad)
                bottom = min(region.height, bottom + y_pad)
        if right - left < max(18, round(region.width * 0.45)):
            return region.copy()
        if bottom - top < max(30, round(region.height * 0.45)):
            return region.copy()
        cropped = region.crop((left, top, right, bottom)).convert("RGB")
        return cropped
    finally:
        mask.close()


def _isolated_column_image(
    image: Image.Image,
    column: DetectedColumn,
    *,
    retry: bool = True,
    background: tuple[int, int, int] | None = None,
    preserve_body_pixels: bool = True,
) -> Image.Image:
    """Return a compact original-pixel column with clean margins.

    In lossless mode the complete hard slot is copied directly and no
    dominant-band trimming is allowed.  This is intentionally slightly wider
    than the detector's body box: preserving a possible dakuten or radical is
    more important than removing a small amount of Ruby before model voting.
    """
    width, height = image.size
    if preserve_body_pixels:
        left = max(0, int(column.hard_left))
        right = min(width, int(column.hard_right))
        if column.full_height_slot:
            top = max(0, int(column.top))
            bottom = min(height, int(column.bottom))
        else:
            extra_y = max(12, round(max(1, column.width) * 1.35))
            top = max(0, int(column.top) - extra_y)
            bottom = min(height, int(column.bottom) + extra_y)
    else:
        union_left, union_top, union_right, union_bottom = _column_source_union(column)
        left = max(0, union_left)
        right = min(width, union_right)
        if column.full_height_slot:
            top = max(0, min(column.top, union_top))
            bottom = min(height, max(column.bottom, union_bottom))
        else:
            extra_y = max(12, round(max(1, column.width) * 1.35))
            top = max(0, min(column.top, union_top) - extra_y)
            bottom = min(height, max(column.bottom, union_bottom) + extra_y)
    if right <= left or bottom <= top:
        return image.copy()
    paper = background or _paper_background(image)
    if preserve_body_pixels:
        region = image.crop((left, top, right, bottom)).convert("RGB")
    else:
        region = Image.new("RGB", (right - left, bottom - top), paper)
        _paste_column_source(
            region, image, column,
            offset_x=left, offset_y=top, background=paper,
            erase_exclusions=True,
        )
        trimmed = _trim_region_to_primary_ink(region)
        region.close()
        region = trimmed
    margin_x = max(8, round(region.width * 0.16))
    margin_y = max(10, round(max(1, column.width) * 0.55))
    canvas = Image.new(
        "RGB",
        (region.width + margin_x * 2, region.height + margin_y * 2),
        paper,
    )
    try:
        canvas.paste(region, (margin_x, margin_y))
    finally:
        region.close()
    return canvas

def _compact_text_length(text: str) -> int:
    return len("".join(str(text or "").split()))


def _quote_score(text: str) -> int:
    value = str(text or "")
    return sum(value.count(ch) for ch in "「」『』")


def _quote_balance_penalty(text: str) -> int:
    value = str(text or "")
    return abs(value.count("「") - value.count("」")) + abs(value.count("『") - value.count("』"))


def _compact_transport_needs_fullsize_check(
    text: str,
    confidence: float,
    column: DetectedColumn,
    *,
    recognition_engine: str = "",
) -> bool:
    """Return whether compact transport has strong evidence of truncation.

    A full-size compatibility pass is expensive and counts as the column's one
    adaptive rescue.  Quote imbalance or a merely modest confidence score are
    sentence-level concerns and no longer trigger another OCR call by themselves.
    """
    value = normalize_column_text(text)
    if not _valid_column_text(value) or "□" in value or "�" in value:
        return True
    actual = _compact_text_length(value)
    expected = max(0, int(column.estimated_chars or 0))
    numeric_confidence = float(confidence or 0.0)
    if 0.0 < numeric_confidence < 0.25:
        return True
    if expected >= 5:
        ratio = actual / max(1, expected)
        if ratio < 0.42 or ratio > 2.10:
            return True
    return False


def _needs_additional_retry(text: str, column: DetectedColumn) -> bool:
    """Return whether another expensive OCR pass can still add information.

    Geometry-only short-block evidence is intentionally excluded here.  Once a
    separated block has already been recognized and merged successfully, the old
    implementation kept treating the same geometry as suspicious and needlessly
    ran the wide pass plus three preprocessing passes.  With three OCR models that
    could turn one physical column into 15-18 recognitions.
    """
    if not _valid_column_text(text):
        return True
    length = _compact_text_length(text)
    if column.estimated_chars >= 8 and length < max(2, round(column.estimated_chars * 0.56)):
        return True
    opens = text.count("「") + text.count("『")
    closes = text.count("」") + text.count("』")
    return (opens + closes) > 0 and opens != closes


def _needs_short_block_recovery(text: str, column: DetectedColumn) -> bool:
    """Initial recovery predicate, including physical separated-block evidence."""
    if _needs_additional_retry(text, column):
        return True
    if len(column.content_spans) >= 2:
        total = sum(max(1, bottom - top) for top, bottom in column.content_spans)
        shortest = min(max(1, bottom - top) for top, bottom in column.content_spans)
        # A small separated block is exactly the pattern most often omitted by a
        # whole-column transformer pass.  This triggers one split-block pass, not
        # every later fallback pass after the block has already been recovered.
        if shortest <= total * 0.30:
            return True
    return False


def _merge_text_parts(parts: list[str]) -> str:
    result = ""
    for raw in parts:
        part = str(raw or "").strip()
        if not part:
            continue
        if not result:
            result = part
            continue
        best = 0
        max_overlap = min(24, len(result), len(part))
        for size in range(max_overlap, 0, -1):
            if result[-size:] == part[:size]:
                best = size
                break
        result += part[best:]
    return result.strip()


def _prefer_recovered_text(primary: str, recovered: str, column: DetectedColumn) -> bool:
    if not _valid_column_text(recovered):
        return False
    if not _valid_column_text(primary):
        return True
    p_len = _compact_text_length(primary)
    r_len = _compact_text_length(recovered)
    ceiling = max(80, p_len * 2 + 20, column.estimated_chars * 2 + 20)
    if r_len > ceiling:
        return False
    if r_len >= p_len + 2:
        return True
    if r_len >= max(1, p_len - 1):
        if _quote_score(recovered) > _quote_score(primary):
            return True
        if (
            _quote_score(recovered) >= _quote_score(primary)
            and _quote_balance_penalty(recovered) < _quote_balance_penalty(primary)
        ):
            return True
    return False


@dataclass(frozen=True)
class _ColumnRescueDecision:
    method: str
    reason: str


def _normalise_column_rescue_policy(value: object) -> str:
    key = str(value or "adaptive").strip().lower()
    aliases = {
        "smart": "adaptive",
        "single": "adaptive",
        "one_pass": "adaptive",
        "none": "off",
        "disabled": "off",
        "full": "legacy",
        "complete": "legacy",
    }
    key = aliases.get(key, key)
    return key if key in {"adaptive", "off", "legacy"} else "adaptive"


def _choose_adaptive_column_rescue(
    text: str,
    confidence: float,
    column: DetectedColumn,
    *,
    compact_primary: bool,
    recognition_engine: str = "",
) -> _ColumnRescueDecision | None:
    """Choose at most one evidence-based rescue family for a physical column."""
    value = normalize_column_text(text)
    valid = _valid_column_text(value) and "□" not in value and "�" not in value
    actual = _compact_text_length(value) if valid else 0
    expected = max(0, int(column.estimated_chars or 0))
    ratio = actual / max(1, expected) if expected else 1.0
    severe_short = expected >= 5 and ratio < 0.45
    moderate_short = expected >= 6 and ratio < 0.68
    very_low_conf = 0.0 < float(confidence or 0.0) < 0.25

    engine = str(recognition_engine or "").strip().lower()
    # Manga OCR's worker always removes page-sized whitespace and rotates a
    # vertical text strip before inference.  Retrying the full-size masked page
    # therefore produces the same prepared model input, wastes one inference,
    # and can reintroduce the very page-mask hallucination this adapter avoids.
    if compact_primary and engine not in {"manga_ocr", "manga_48px", "yomitoku"} and (
        not valid or severe_short or very_low_conf
    ):
        return _ColumnRescueDecision(
            "primary_fullsize_fallback",
            "紧凑列图存在空白、占位符、极低置信或严重缺字，执行一次原尺寸兼容复核",
        )
    if len(column.content_spans) >= 2 and (not valid or moderate_short):
        return _ColumnRescueDecision(
            "short_blocks",
            "检测到分离文字段且主结果为空或明显不足，执行一次分段恢复",
        )
    if not valid:
        return _ColumnRescueDecision(
            "balanced_crop_2x",
            "主结果为空或含占位符，执行一次局部平衡放大",
        )
    if severe_short or (very_low_conf and moderate_short):
        if not column.full_height_slot and actual > 0:
            return _ColumnRescueDecision(
                "wide",
                "非固定区域列结果明显过短，执行一次扩边复核",
            )
        return _ColumnRescueDecision(
            "balanced_crop_2x",
            "固定物理列结果明显过短，执行一次局部平衡放大",
        )
    return None


@dataclass(frozen=True)
class _TextCandidate:
    text: str
    confidence: float
    method: str
    score: float


def _japanese_character_ratio(text: str) -> float:
    compact = [ch for ch in str(text or "") if not ch.isspace()]
    if not compact:
        return 0.0
    japanese = 0
    for ch in compact:
        code = ord(ch)
        if (
            0x3040 <= code <= 0x30FF
            or 0x31F0 <= code <= 0x31FF
            or 0x3400 <= code <= 0x4DBF
            or 0x4E00 <= code <= 0x9FFF
            or 0xF900 <= code <= 0xFAFF
            or ch in "、。！？…―ー・「」『』（）［］【】〈〉《》〜～＝"
        ):
            japanese += 1
    return japanese / max(1, len(compact))


_SMART_REOCR_LATIN_OR_NUMBER_RE = re.compile(r"(?:[A-Za-z]{2,}|[0-9０-９]{2,})")
_SMART_REOCR_REPEAT_RE = re.compile(r"(.)\1{4,}")


def _sentence_group_smart_reocr_decision(
    targets: Sequence[tuple[str, int]],
    recognized: dict[tuple[str, int], tuple[str, float, str | None]],
    selection_meta: dict[tuple[str, int], dict],
    page_columns: dict[str, list[DetectedColumn]],
    *,
    recognition_engine: str = "",
) -> tuple[bool, str]:
    """Return whether a complete sentence still merits an expensive second OCR.

    This is deliberately a *risk-evidence* gate.  Sentence length and page
    boundaries alone are not defects: a stable six-column or cross-page sentence
    is skipped just like a short same-page sentence.  Re-OCR is reserved for
    missing text, one-pass column rescue, conflicts, severe length mismatch,
    sentence-terminal problems or symbol anomalies.
    """
    items = list(targets)
    if len(items) < 2:
        return False, "单列句不需要整句二次识别"
    parts: list[str] = []
    confidences: list[float] = []
    expected_total = 0
    actual_total = 0
    for target in items:
        result = recognized.get(target)
        if not result or result[2]:
            return True, "存在未返回或失败的逐列结果"
        text = normalize_column_text(result[0])
        if not text or "□" in text or "�" in text:
            return True, "存在空列或未识别占位符"
        if is_spurious_ocr_item(text):
            return True, "逐列底稿含异常文本"
        confidence = float(result[1] or 0.0)
        if confidence <= 0.0:
            return True, "缺少可用于安全跳过的置信度"
        meta = selection_meta.get(target) or {}
        if bool(meta.get("conflict")):
            return True, "逐列候选存在冲突"
        if bool(meta.get("rescue_used")):
            return True, "句内至少一列经过自适应救援"
        selected_method = str(meta.get("selected_method") or "primary")
        if selected_method not in {"primary", "page_primary"}:
            return True, "逐列结果来自恢复或增强候选"
        try:
            column = page_columns[target[0]][target[1]]
        except Exception:
            return True, "缺少稳定列几何"
        expected = max(0, int(column.estimated_chars or 0))
        actual = _compact_text_length(text)
        if expected >= 4:
            ratio = actual / max(1, expected)
            if not (0.52 <= ratio <= 1.62):
                return True, "单列字数与黑像素估计差异较大"
        expected_total += expected
        actual_total += actual
        parts.append(text)
        confidences.append(confidence)

    baseline = join_column_parts(parts)
    if not baseline or not has_sentence_terminal(baseline):
        return True, "逐列底稿缺少明确句末"
    if is_provisional_quote_terminal(baseline):
        return True, "裸闭引号句保留上下文确认"
    if _quote_balance_penalty(baseline) > 0:
        return True, "逐列底稿引号不平衡"
    if _SMART_REOCR_LATIN_OR_NUMBER_RE.search(baseline):
        return True, "含连续英文或数字，保留上下文核验"
    if _SMART_REOCR_REPEAT_RE.search(baseline):
        return True, "存在异常连续重复字符"
    if _japanese_character_ratio(baseline) < 0.72:
        return True, "日文字符比例不足以安全跳过"
    if expected_total >= 8:
        total_ratio = actual_total / max(1, expected_total)
        if not (0.64 <= total_ratio <= 1.46):
            return True, "整句字数与黑像素估计差异较大"

    engine = str(recognition_engine or "").strip().lower()
    min_floor = 0.84 if engine == "ndlocr_lite" else 0.90
    average_floor = 0.88 if engine == "ndlocr_lite" else 0.93
    if min(confidences) < min_floor or sum(confidences) / len(confidences) < average_floor:
        return True, "逐列置信度不足以安全跳过"
    return False, "逐列底稿稳定完整，智能跳过重复整句 OCR"


def _repetition_penalty(text: str) -> float:
    value = "".join(str(text or "").split())
    penalty = 0.0
    for size in (2, 3, 4):
        for index in range(0, max(0, len(value) - size * 2 + 1)):
            chunk = value[index:index + size]
            if chunk and value[index + size:index + size * 2] == chunk:
                penalty += 1.5
    return min(9.0, penalty)


def _score_text_candidate(
    text: str,
    confidence: float,
    column: DetectedColumn,
    method: str,
) -> float:
    if not _valid_column_text(text):
        return -10_000.0
    value = str(text or "").strip()
    length = _compact_text_length(value)
    expected = max(0, int(column.estimated_chars or 0))
    score = max(0.0, min(1.0, float(confidence or 0.0))) * 22.0
    score += _japanese_character_ratio(value) * 13.0
    if expected >= 3:
        distance = abs(length - expected) / max(3, expected)
        score += max(-10.0, 9.0 - distance * 12.0)
        if length < max(2, round(expected * 0.48)):
            score -= 7.0
        if length > expected * 2.15 + 10:
            score -= 12.0
    else:
        score += min(5.0, length * 0.45)
    score -= _quote_balance_penalty(value) * 2.2
    score -= sum(value.count(ch) for ch in "□�|¦") * 5.0
    score -= _repetition_penalty(value)
    # Conservative method bias: the untouched original remains preferred unless
    # a fallback is materially stronger.  Binary OCR must earn its replacement.
    score += {
        "primary": 4.0,
        "page_primary": 4.0,
        "primary_fullsize_fallback": 3.8,
        "short_blocks": 2.5,
        "wide": 2.0,
        "balanced_full": 1.2,
        "balanced_crop_2x": 0.6,
        "adaptive_binary_crop_2x": -0.5,
    }.get(method, 0.0)
    return score


def _make_text_candidate(
    text: str,
    confidence: float,
    column: DetectedColumn,
    method: str,
) -> _TextCandidate | None:
    if not _valid_column_text(text):
        return None
    value = str(text or "").strip()
    numeric_confidence = float(confidence or 0.0)
    if any(ch in _OCR_PLACEHOLDER_CHARS for ch in value):
        numeric_confidence = 0.0
    return _TextCandidate(
        text=value,
        confidence=numeric_confidence,
        method=method,
        score=_score_text_candidate(value, numeric_confidence, column, method),
    )


def _select_text_candidate(
    candidates: list[_TextCandidate],
    column: DetectedColumn,
) -> tuple[_TextCandidate | None, bool, list[_TextCandidate]]:
    if not candidates:
        return None, False, []
    by_text: dict[str, _TextCandidate] = {}
    counts: dict[str, int] = {}
    for item in candidates:
        counts[item.text] = counts.get(item.text, 0) + 1
        previous = by_text.get(item.text)
        if previous is None or item.score > previous.score:
            by_text[item.text] = item
    ranked: list[_TextCandidate] = []
    for text, item in by_text.items():
        consensus_bonus = min(6.0, max(0, counts.get(text, 1) - 1) * 2.0)
        ranked.append(_TextCandidate(item.text, item.confidence, item.method, item.score + consensus_bonus))
    ranked.sort(key=lambda item: (item.score, item.confidence, len(item.text)), reverse=True)
    best = ranked[0]
    primary = next((item for item in ranked if item.method == "primary"), None)
    conflict = bool(
        len(ranked) >= 2
        and ranked[0].text != ranked[1].text
        and (
            abs(ranked[0].score - ranked[1].score) <= 4.0
            or min(ranked[0].confidence, ranked[1].confidence) >= 0.90
        )
    )

    if primary is not None and best.text != primary.text:
        similarity = SequenceMatcher(None, primary.text, best.text, autojunk=False).ratio()
        p_len = _compact_text_length(primary.text)
        b_len = _compact_text_length(best.text)
        # Do not replace a plausible original with a radically different OCR
        # hallucination merely because the fallback emitted more characters.
        radical = similarity < 0.32 and p_len >= max(3, round((column.estimated_chars or p_len) * 0.52))
        excessive = b_len > max(p_len * 1.75 + 8, (column.estimated_chars or 0) * 2.0 + 12)
        if best.score < primary.score + 3.5 or radical or excessive:
            conflict = conflict or best.text != primary.text
            best = primary
    return best, conflict, ranked


def _recognize_batch(
    engine: str,
    crop_paths: list[str],
    manifest_path: str,
    *,
    shortcut_name: str,
    cancel_check,
    verbose: bool,
    engine_options: dict | None = None,
    progress_callback=None,
) -> dict[str, tuple[str, float, str | None]]:
    results: dict[str, tuple[str, float, str | None]] = {}
    iterator = _recognizer_iterator(
        engine,
        crop_paths,
        manifest_path,
        shortcut_name=shortcut_name,
        cancel_check=cancel_check,
        verbose=verbose,
        engine_options=engine_options,
    )
    total = max(1, len(crop_paths))
    for current, (crop_path, blocks, error) in enumerate(iterator, start=1):
        key = str(crop_path)
        if error:
            results[key] = ("", 0.0, str(error))
        else:
            text, confidence = _candidate_text(
                blocks,
                recognition_engine=engine,
                image_path=key,
                authoritative_column=True,
            )
            results[key] = (text, confidence, None)
        if callable(progress_callback):
            progress_callback(current, total, key)
    return results


class _RecognizerSession:
    """Share Manga OCR's model across all adaptive passes."""

    def __init__(self, engine: str, *, shortcut_name: str, cancel_check, verbose: bool, temp: Path,
                 engine_options: dict | None = None, load_progress_callback=None):
        self.engine = engine
        self.shortcut_name = shortcut_name
        self.cancel_check = cancel_check
        self.verbose = verbose
        self.temp = temp
        self.engine_options = dict(engine_options or {})
        self.load_progress_callback = load_progress_callback
        self._manga = None
        self._manga48 = None
        self._yomitoku = None
        self._persistent = None
        self._jsonl_persistent = None
        self._serial = 0

    def __enter__(self):
        if self.engine == "manga_ocr":
            from adapters.manga_ocr_adapter import MangaOcrSession
            self._manga = MangaOcrSession(cancel_check=self.cancel_check, verbose=self.verbose)
            self._manga.__enter__()
        elif self.engine == "manga_48px":
            from adapters.manga_48px_adapter import Manga48pxSession
            self._manga48 = Manga48pxSession(
                cancel_check=self.cancel_check,
                verbose=self.verbose,
                load_progress_callback=self.load_progress_callback,
            )
            self._manga48.__enter__()
        elif self.engine == "yomitoku":
            from adapters.yomitoku_adapter import YomiTokuSession
            self._yomitoku = YomiTokuSession(
                mode=str(self.engine_options.get("mode") or "fast"),
                device=str(self.engine_options.get("device") or "auto"),
                detector_onnx=bool(self.engine_options.get("detector_onnx", True)),
                large_review=bool(self.engine_options.get("large_review", True)),
                review_threshold=float(self.engine_options.get("review_threshold", 0.82) or 0.82),
                cancel_check=self.cancel_check,
                verbose=self.verbose,
            )
            self._yomitoku.__enter__()
        # Preserve the monkeypatch hook used by tests/plugins.  Persistent native
        # sessions are enabled only when the stock bridge is active.
        elif _recognizer_iterator is recognizer_iterator and self.engine == "apple_vision":
            self._persistent = AppleVisionRecognitionSession(
                shortcut_name=self.shortcut_name,
                engine_options=self.engine_options,
                cancel_check=self.cancel_check,
            )
            self._persistent.__enter__()
        elif _recognizer_iterator is recognizer_iterator and self.engine == "ndlocr_lite":
            from adapters.ndlocr_lite_adapter import NDLOcrLiteSession
            self._persistent = NDLOcrLiteSession(
                cancel_check=self.cancel_check, verbose=self.verbose
            )
            self._persistent.__enter__()
        elif _recognizer_iterator is recognizer_iterator and self.engine == "paddle_ocr":
            # Paddle model startup is expensive. Keep exactly one JSONL worker
            # alive for primary recognition, adaptive rescue and sentence OCR
            # within this model run instead of reloading PP-OCR for every pass.
            from adapters.persistent_recognition_session import PersistentRecognitionSession
            self._jsonl_persistent = PersistentRecognitionSession(
                engine="paddle_ocr", engine_options=self.engine_options
            )
            self._jsonl_persistent.__enter__()
        return self

    def recognize_blocks(
        self,
        paths: list[str],
        *,
        progress_callback=None,
    ) -> dict[str, tuple[list[dict] | None, str | None]] | None:
        """Return raw blocks when the selected persistent engine supports them.

        Full-page NDLOCR routing needs geometry.  Other engines and plugin
        sessions keep using the stable text-only ``recognize`` contract.
        """
        if not paths:
            return {}
        if self.engine != "ndlocr_lite" or self._persistent is None:
            return None
        results: dict[str, tuple[list[dict] | None, str | None]] = {}
        total = max(1, len(paths))
        for current, (image_path, blocks, error) in enumerate(
            self._persistent.iter_recognize(paths), start=1
        ):
            key = str(image_path)
            results[key] = (None, str(error)) if error else (list(blocks or []), None)
            if callable(progress_callback):
                progress_callback(current, total, key)
        return results

    def recognize(
        self,
        paths: list[str],
        label: str,
        *,
        progress_callback=None,
    ) -> dict[str, tuple[str, float, str | None]]:
        if not paths:
            return {}
        if self._manga is not None:
            return self._manga.recognize(paths, progress_callback=progress_callback)
        if self._manga48 is not None:
            return self._manga48.recognize(paths, progress_callback=progress_callback)
        if self._yomitoku is not None:
            return self._yomitoku.recognize(paths, progress_callback=progress_callback)
        if self._persistent is not None:
            results: dict[str, tuple[str, float, str | None]] = {}
            total = max(1, len(paths))
            for current, (crop_path, blocks, error) in enumerate(
                self._persistent.iter_recognize(paths), start=1
            ):
                key = str(crop_path)
                if error:
                    results[key] = ("", 0.0, str(error))
                else:
                    text, confidence = _candidate_text(
                        blocks,
                        recognition_engine=self.engine,
                        image_path=key,
                        authoritative_column=True,
                    )
                    results[key] = (text, confidence, None)
                if callable(progress_callback):
                    progress_callback(current, total, key)
            return results
        if self._jsonl_persistent is not None:
            results: dict[str, tuple[str, float, str | None]] = {}
            total = max(1, len(paths))
            for current, (crop_path, blocks, error) in enumerate(
                self._jsonl_persistent.recognize(paths), start=1
            ):
                key = str(crop_path)
                if error:
                    results[key] = ("", 0.0, str(error))
                else:
                    text, confidence = _candidate_text(
                        blocks,
                        recognition_engine=self.engine,
                        image_path=key,
                        authoritative_column=True,
                    )
                    results[key] = (text, confidence, None)
                if callable(progress_callback):
                    progress_callback(current, total, key)
            return results
        self._serial += 1
        return _recognize_batch(
            self.engine,
            paths,
            str(self.temp / f"manifest_{self._serial:03d}_{label}.json"),
            shortcut_name=self.shortcut_name,
            cancel_check=self.cancel_check,
            verbose=self.verbose,
            engine_options=self.engine_options,
            progress_callback=progress_callback,
        )

    def __exit__(self, exc_type, exc, tb):
        if self._manga is not None:
            try:
                return self._manga.__exit__(exc_type, exc, tb)
            finally:
                self._manga = None
        if self._manga48 is not None:
            try:
                return self._manga48.__exit__(exc_type, exc, tb)
            finally:
                self._manga48 = None
        if self._yomitoku is not None:
            try:
                return self._yomitoku.__exit__(exc_type, exc, tb)
            finally:
                self._yomitoku = None
        if self._persistent is not None:
            try:
                return self._persistent.__exit__(exc_type, exc, tb)
            finally:
                self._persistent = None
        if self._jsonl_persistent is not None:
            try:
                return self._jsonl_persistent.__exit__(exc_type, exc, tb)
            finally:
                self._jsonl_persistent = None
        return False


def _session_recognize(
    session,
    paths: list[str],
    label: str,
    *,
    progress_callback=None,
) -> dict[str, tuple[str, float, str | None]]:
    """Call a session with live progress while preserving old plugin APIs."""

    method = session.recognize
    supports_callback = False
    try:
        parameters = inspect.signature(method).parameters.values()
        supports_callback = any(
            parameter.name == "progress_callback"
            or parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
    except (TypeError, ValueError):
        supports_callback = False

    if supports_callback:
        return method(paths, label, progress_callback=progress_callback)

    result = method(paths, label)
    if callable(progress_callback):
        total = max(1, len(paths))
        for current, path in enumerate(paths, start=1):
            progress_callback(current, total, str(path))
    return result


def _apply_column_cleanup(
    image: Image.Image,
    *,
    auto_filter_ruby: bool,
    filter_fragments: bool,
    smart_crop: bool,
    ruby_strength: str,
    background: tuple[int, int, int] | None = None,
    preserve_body_pixels: bool = True,
) -> Image.Image:
    if not (auto_filter_ruby or filter_fragments or smart_crop):
        return image
    result = cleanup_column_image(
        image,
        auto_filter_ruby=bool(auto_filter_ruby),
        filter_fragments=bool(filter_fragments),
        smart_crop=bool(smart_crop),
        ruby_strength=normalise_ruby_strength(ruby_strength),
        background=background,
        preserve_body_pixels=bool(preserve_body_pixels),
    )
    image.close()
    return result.image


def _prepare_page_crops(
    page_index: int,
    page_path: str,
    temp: Path,
    *,
    sensitivity: int,
    padding_percent: int,
    max_columns: int,
    fixed_region_rect: Sequence[float] | None = None,
    compact_transport: bool = False,
    auto_filter_ruby: bool = False,
    filter_fragments: bool = False,
    smart_crop: bool = False,
    ruby_strength: str = "standard",
    detector_mode: str = "components",
    preserve_body_pixels: bool = True,
) -> tuple[str, list[DetectedColumn], list[tuple[str, int]], str | None]:
    try:
        image = _open_page_image(page_path, fixed_region_rect)
        try:
            columns = detect_vertical_columns(
                image,
                sensitivity=sensitivity,
                padding_percent=padding_percent,
                max_columns=max_columns,
                fixed_region_rect=fixed_region_rect,
                fixed_region_already_masked=bool(fixed_region_rect),
                detector_mode=detector_mode,
            )
            if not columns:
                raise RuntimeError("固定正文区域内没有检测到可识别的竖列")
            generated: list[tuple[str, int]] = []
            background = _paper_background(image)
            for column_index, column in enumerate(columns, start=1):
                # The compact transport removes only guaranteed blank canvas.
                # Glyph pixels remain at their original resolution and are never
                # resampled.  If recognition returns empty, the caller retries
                # this column with the original full-size masked page.
                crop = (
                    _isolated_column_image(
                        image, column, retry=False, background=background,
                        preserve_body_pixels=preserve_body_pixels,
                    )
                    if (compact_transport or smart_crop)
                    else _crop_column(
                        image, column, retry=False, background=background,
                        preserve_body_pixels=preserve_body_pixels,
                    )
                )
                crop = _apply_column_cleanup(
                    crop,
                    auto_filter_ruby=auto_filter_ruby,
                    filter_fragments=filter_fragments,
                    smart_crop=smart_crop,
                    ruby_strength=ruby_strength,
                    background=background,
                    preserve_body_pixels=preserve_body_pixels,
                )
                crop_path = temp / f"p{page_index:05d}_c{column_index:03d}.png"
                crop.save(crop_path, format="PNG", compress_level=1)
                crop.close()
                generated.append((str(crop_path), column_index - 1))
            return page_path, columns, generated, None
        finally:
            image.close()
    except Exception as exc:
        return page_path, [], [], str(exc)


def _column_to_json(column: DetectedColumn) -> dict:
    return {
        "left": column.left, "top": column.top, "right": column.right,
        "bottom": column.bottom, "hard_left": column.hard_left,
        "hard_right": column.hard_right, "ink_score": column.ink_score,
        "content_spans": [list(span) for span in column.content_spans],
        "estimated_chars": column.estimated_chars,
        "full_height_slot": bool(column.full_height_slot),
        "supplemental_boxes": [list(box) for box in column.supplemental_boxes],
        "excluded_boxes": [list(box) for box in column.excluded_boxes],
    }


def _column_from_json(data: dict) -> DetectedColumn:
    return DetectedColumn(
        left=int(data["left"]), top=int(data["top"]),
        right=int(data["right"]), bottom=int(data["bottom"]),
        hard_left=int(data["hard_left"]), hard_right=int(data["hard_right"]),
        ink_score=float(data.get("ink_score", 0.0) or 0.0),
        content_spans=tuple(
            (int(span[0]), int(span[1])) for span in data.get("content_spans", [])
            if isinstance(span, (list, tuple)) and len(span) >= 2
        ),
        estimated_chars=int(data.get("estimated_chars", 0) or 0),
        full_height_slot=bool(data.get("full_height_slot", False)),
        supplemental_boxes=tuple(
            (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            for box in data.get("supplemental_boxes", [])
            if isinstance(box, (list, tuple)) and len(box) >= 4
        ),
        excluded_boxes=tuple(
            (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            for box in data.get("excluded_boxes", [])
            if isinstance(box, (list, tuple)) and len(box) >= 4
        ),
    )


def _prepare_page_crops_cached_unlocked(
    page_index: int,
    page_path: str,
    cache_dir: Path,
    *,
    sensitivity: int,
    padding_percent: int,
    max_columns: int,
    fixed_region_rect: Sequence[float] | None = None,
    compact_transport: bool = False,
    auto_filter_ruby: bool = False,
    filter_fragments: bool = False,
    smart_crop: bool = False,
    ruby_strength: str = "standard",
    detector_mode: str = "components",
    preserve_body_pixels: bool = True,
) -> tuple[str, list[DetectedColumn], list[tuple[str, int]], str | None]:
    """Prepare deterministic column masks once and reuse them across OCR models.

    ``cache_dir`` belongs to one GUI OCR run, so there is no stale cross-book
    cache risk.  A compact JSON sidecar protects against partially written files.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    sidecar = cache_dir / f"p{page_index:05d}_columns.json"
    if sidecar.exists():
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            expected = {
                "detector_version": column_detector_version(detector_mode),
                "detector_mode": _normalise_column_detector_mode(detector_mode),
                "page_path": str(page_path),
                "sensitivity": int(sensitivity),
                "padding_percent": int(padding_percent),
                "max_columns": int(max_columns),
                "fixed_region_rect": list(fixed_region_rect or []),
                "compact_transport": bool(compact_transport),
                "auto_filter_ruby": bool(auto_filter_ruby),
                "filter_fragments": bool(filter_fragments),
                "smart_crop": bool(smart_crop),
                "ruby_strength": normalise_ruby_strength(ruby_strength),
                "preserve_body_pixels": bool(preserve_body_pixels),
            }
            if all(payload.get(key) == value for key, value in expected.items()):
                columns = [_column_from_json(item) for item in payload.get("columns", [])]
                generated = []
                expected_hashes = list(payload.get("crop_sha256", []) or [])
                crop_files = list(payload.get("crop_files", []) or [])
                if len(expected_hashes) != len(crop_files):
                    raise ValueError("列图缓存缺少完整 SHA-256 审计")
                for zero_index, relative in enumerate(crop_files):
                    crop_path = cache_dir / str(relative)
                    if not crop_path.exists() or crop_path.stat().st_size <= 0:
                        raise FileNotFoundError(str(crop_path))
                    if _file_sha256(crop_path) != str(expected_hashes[zero_index]):
                        raise ValueError(f"列图缓存哈希不一致: {crop_path.name}")
                    generated.append((str(crop_path), zero_index))
                if columns and len(columns) == len(generated):
                    return page_path, columns, generated, None
        except Exception:
            # Incomplete cache: rebuild this page only.
            pass

    result = _prepare_page_crops(
        page_index, page_path, cache_dir,
        sensitivity=sensitivity, padding_percent=padding_percent,
        max_columns=max_columns, fixed_region_rect=fixed_region_rect,
        compact_transport=compact_transport,
        auto_filter_ruby=auto_filter_ruby,
        filter_fragments=filter_fragments,
        smart_crop=smart_crop,
        ruby_strength=ruby_strength,
        detector_mode=detector_mode,
        preserve_body_pixels=preserve_body_pixels,
    )
    returned_path, columns, generated, error = result
    if error or not columns:
        return result
    payload = {
        "detector_version": column_detector_version(detector_mode),
        "detector_mode": _normalise_column_detector_mode(detector_mode),
        "page_path": str(page_path),
        "sensitivity": int(sensitivity),
        "padding_percent": int(padding_percent),
        "max_columns": int(max_columns),
        "fixed_region_rect": list(fixed_region_rect or []),
        "compact_transport": bool(compact_transport),
        "auto_filter_ruby": bool(auto_filter_ruby),
        "filter_fragments": bool(filter_fragments),
        "smart_crop": bool(smart_crop),
        "ruby_strength": normalise_ruby_strength(ruby_strength),
        "preserve_body_pixels": bool(preserve_body_pixels),
        "columns": [_column_to_json(column) for column in columns],
        "crop_files": [Path(path).name for path, _ in generated],
        "crop_sha256": [_file_sha256(path) for path, _ in generated],
    }
    tmp = sidecar.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(sidecar)
    return returned_path, columns, generated, None


def _prepare_page_crops_cached(
    page_index: int,
    page_path: str,
    cache_dir: Path,
    *,
    sensitivity: int,
    padding_percent: int,
    max_columns: int,
    fixed_region_rect: Sequence[float] | None = None,
    compact_transport: bool = False,
    auto_filter_ruby: bool = False,
    filter_fragments: bool = False,
    smart_crop: bool = False,
    ruby_strength: str = "standard",
    detector_mode: str = "components",
    preserve_body_pixels: bool = True,
) -> tuple[str, list[DetectedColumn], list[tuple[str, int]], str | None]:
    """Thread-safe wrapper for two OCR engines sharing one crop cache."""
    lock = _shared_prepare_lock(cache_dir, page_index)
    with lock:
        return _prepare_page_crops_cached_unlocked(
            page_index,
            page_path,
            cache_dir,
            sensitivity=sensitivity,
            padding_percent=padding_percent,
            max_columns=max_columns,
            fixed_region_rect=fixed_region_rect,
            compact_transport=compact_transport,
            auto_filter_ruby=auto_filter_ruby,
            filter_fragments=filter_fragments,
            smart_crop=smart_crop,
            ruby_strength=ruby_strength,
            detector_mode=detector_mode,
            preserve_body_pixels=preserve_body_pixels,
        )


def _prepare_ndlocr_page_inputs(
    page_paths: Sequence[str],
    page_columns: dict[str, list[DetectedColumn]],
    temp: Path,
    *,
    fixed_region_rect: Sequence[float] | None,
    preserve_body_pixels: bool = True,
) -> tuple[list[str], dict[str, str], dict[str, tuple[int, int]]]:
    """Create one fixed-region page image per page for NDLOCR.

    The production path restores the v2 contract: NDLOCR receives the original
    fixed-region pixels, not a page reconstructed from component body bands.
    A legacy filtered page remains available only when explicitly requested.
    """
    temp.mkdir(parents=True, exist_ok=True)
    inputs: list[str] = []
    input_to_page: dict[str, str] = {}
    image_sizes: dict[str, tuple[int, int]] = {}
    for page_number, page_path in enumerate(page_paths, start=1):
        if page_path not in page_columns:
            continue
        image = _open_page_image(page_path, fixed_region_rect)
        try:
            target = temp / f"ndl_page_{page_number:05d}.png"
            if preserve_body_pixels:
                image.save(target, format="PNG", compress_level=1)
            else:
                background = _paper_background(image)
                filtered = Image.new("RGB", image.size, background)
                try:
                    for column in page_columns.get(page_path, []):
                        _paste_column_source(
                            filtered, image, column, background=background,
                            erase_exclusions=True,
                        )
                    filtered.save(target, format="PNG", compress_level=1)
                finally:
                    filtered.close()
            key = str(target)
            inputs.append(key)
            input_to_page[key] = page_path
            image_sizes[page_path] = image.size
        finally:
            image.close()
    return inputs, input_to_page, image_sizes

def _iter_column_pages(
    page_paths: list[str],
    *,
    recognition_engine: str,
    sensitivity: int,
    padding_percent: int,
    strict: bool,
    shortcut_name: str,
    max_columns: int,
    cancel_check,
    verbose: bool,
    phase_callback: PhaseCallback | None,
    engine_options: dict | None = None,
) -> Iterator[tuple[str, list[dict] | None, str | None]]:
    with tempfile.TemporaryDirectory(prefix="novel_formatter_column_ocr_") as temp_dir:
        temp = Path(temp_dir)
        runtime_options = dict(engine_options or {})
        column_preview_callback = runtime_options.pop("column_preview_callback", None)
        fixed_region_rect_raw = runtime_options.pop("column_fixed_region_rect", None)
        fixed_region_rect = None
        if isinstance(fixed_region_rect_raw, (list, tuple)) and len(fixed_region_rect_raw) >= 4:
            try:
                fixed_region_rect = tuple(float(value) for value in fixed_region_rect_raw[:4])
            except (TypeError, ValueError):
                fixed_region_rect = None
        input_profile = _normalise_column_input_profile(
            runtime_options.pop("column_input_profile", None)
        )
        shared_prepare_value = str(runtime_options.get("column_shared_prepare_dir", "") or "")
        shared_prepare_base = Path(shared_prepare_value) if shared_prepare_value else None
        shared_prepare: Path | None = None
        input_profile_fingerprint = ""
        shared_variant_value = str(runtime_options.get("column_shared_variant_dir", "") or "")
        shared_variants = Path(shared_variant_value) if shared_variant_value else None
        if shared_variants is not None:
            shared_variants.mkdir(parents=True, exist_ok=True)
        shared_sentence_value = str(runtime_options.get("column_shared_sentence_dir", "") or "")
        shared_sentences = Path(shared_sentence_value) if shared_sentence_value else None
        if shared_sentences is not None:
            shared_sentences.mkdir(parents=True, exist_ok=True)
        compact_primary_transport = bool(
            runtime_options.pop("column_compact_primary_transport", False)
        )
        detector_mode = _normalise_column_detector_mode(
            runtime_options.pop("column_detector_mode", "components")
        )
        # Normal OCR is never allowed to opt into the old projection column
        # splitter through persisted settings or plugin kwargs.  Per-character
        # projection remains isolated in the review module.
        if detector_mode != "components":
            detector_mode = "components"
        # Production OCR defaults to the v2-style lossless body-pixel contract.
        # Component geometry may determine column routing, but no component or
        # density rule is allowed to paint over pixels inside the physical slot.
        preserve_body_pixels = bool(
            runtime_options.pop("column_preserve_body_pixels", True)
        )
        auto_filter_ruby = bool(
            runtime_options.pop("column_auto_filter_ruby", True)
        )
        filter_fragments = bool(
            runtime_options.pop("column_filter_fragments", False)
        )
        requested_smart_crop = bool(
            runtime_options.pop("column_smart_crop", False)
        )
        ruby_strength = normalise_ruby_strength(
            runtime_options.pop("column_ruby_strength", "standard")
        )
        # Three explicit contracts coexist without ambiguity:
        # * v8_exact: the original v8 GUI production snapshot;
        # * v8_legacy: the original v8 direct-call/plugin semantics;
        # * custom: the user-selected interactive cleanup values.
        engine_key = str(recognition_engine or "").strip().lower()
        if input_profile == "v8_exact":
            auto_filter_ruby = True
            filter_fragments = False
            requested_smart_crop = True
            ruby_strength = "standard"
            preserve_body_pixels = True
            smart_crop = True
        elif input_profile == "v8_legacy":
            smart_crop = requested_smart_crop or engine_key in {
                "apple_vision", "macocr", "mac_ocr", "macos_ocr",
            }
        else:
            smart_crop = requested_smart_crop
        if str(recognition_engine or "").strip().lower() == "manga_ocr":
            # Manga OCR's ViT processor resizes the whole input.  A page-sized
            # mask containing one 40-60 px column makes the glyphs effectively
            # disappear and the decoder hallucinates fluent unrelated Japanese.
            # It must always receive a compact physical text region.
            compact_primary_transport = True
        elif str(recognition_engine or "").strip().lower() == "manga_48px":
            # The 48px AR network is also a line recognizer, not a page-layout
            # model. It receives compact physical columns and rotates them in
            # its worker before resizing to the trained 48-pixel height.
            compact_primary_transport = True
        elif str(recognition_engine or "").strip().lower() == "yomitoku":
            # YomiToku receives the authoritative compact physical column and
            # bypasses its page-level DBNet detector.  This keeps detached
            # radicals and punctuation in the same recognition polygon.
            compact_primary_transport = True
        _input_contract_payload, input_profile_fingerprint = (
            _column_input_contract_descriptor(
                recognition_engine=recognition_engine,
                detector_mode=detector_mode,
                sensitivity=sensitivity,
                padding_percent=padding_percent,
                max_columns=max_columns,
                fixed_region_rect=fixed_region_rect,
                compact_transport=compact_primary_transport,
                auto_filter_ruby=auto_filter_ruby,
                filter_fragments=filter_fragments,
                smart_crop=smart_crop,
                ruby_strength=ruby_strength,
                preserve_body_pixels=preserve_body_pixels,
                input_profile=input_profile,
            )
        )
        if shared_prepare_base is not None:
            shared_prepare, namespaced_fingerprint = _shared_prepare_profile_dir(
                shared_prepare_base,
                recognition_engine=recognition_engine,
                detector_mode=detector_mode,
                sensitivity=sensitivity,
                padding_percent=padding_percent,
                max_columns=max_columns,
                fixed_region_rect=fixed_region_rect,
                compact_transport=compact_primary_transport,
                auto_filter_ruby=auto_filter_ruby,
                filter_fragments=filter_fragments,
                smart_crop=smart_crop,
                ruby_strength=ruby_strength,
                preserve_body_pixels=preserve_body_pixels,
                input_profile=input_profile,
            )
            if namespaced_fingerprint != input_profile_fingerprint:
                raise RuntimeError("OCR 输入合同缓存指纹不一致")
        legacy_ndlocr_page_batch = runtime_options.pop(
            "column_ndlocr_page_batch", True
        )
        ndlocr_page_mode = _normalise_ndlocr_page_mode(
            runtime_options.pop("column_ndlocr_page_mode", None),
            legacy_page_batch=legacy_ndlocr_page_batch,
        )
        sentence_reocr_strategy = str(
            runtime_options.pop("column_sentence_context_strategy", "full") or "full"
        ).strip().lower()
        if sentence_reocr_strategy not in {"smart", "full"}:
            sentence_reocr_strategy = "full"

        raw_target_column_ids = runtime_options.pop("column_target_ids", None)
        target_column_ids: set[str] | None
        if raw_target_column_ids is None:
            target_column_ids = None
        elif isinstance(raw_target_column_ids, str):
            target_column_ids = {raw_target_column_ids} if raw_target_column_ids else set()
        else:
            target_column_ids = {
                str(value) for value in raw_target_column_ids if str(value)
            }
        raw_seed_results = runtime_options.pop("column_seed_results", {}) or {}
        seed_results = raw_seed_results if isinstance(raw_seed_results, dict) else {}
        seed_mark_selective = bool(
            runtime_options.pop("column_seed_mark_selective", False)
        )
        raw_sentence_target_ids = runtime_options.pop("column_sentence_target_ids", None)
        sentence_target_ids: set[str] | None
        if raw_sentence_target_ids is None:
            sentence_target_ids = None
        elif isinstance(raw_sentence_target_ids, str):
            sentence_target_ids = {raw_sentence_target_ids} if raw_sentence_target_ids else set()
        else:
            sentence_target_ids = {
                str(value) for value in raw_sentence_target_ids if str(value)
            }

        column_rescue_policy = _normalise_column_rescue_policy(
            runtime_options.pop("column_rescue_policy", "adaptive")
        )
        page_columns: dict[str, list[DetectedColumn]] = {}
        crop_to_target: dict[str, tuple[str, int]] = {}
        crop_input_sha256: dict[tuple[str, int], str] = {}
        ndlocr_page_input_sha256: dict[str, str] = {}
        crop_paths: list[str] = []
        detection_errors: dict[str, str] = {}

        jobs = [(index, path) for index, path in enumerate(page_paths, start=1)]
        prepare_cap = 6 if compact_primary_transport else 4
        worker_count = min(prepare_cap, max(1, os.cpu_count() or 2), len(jobs))
        def prepare_job(job):
            return (
                _prepare_page_crops_cached(
                    job[0], job[1], shared_prepare,
                    sensitivity=sensitivity,
                    padding_percent=padding_percent,
                    max_columns=max_columns,
                    fixed_region_rect=fixed_region_rect,
                    compact_transport=compact_primary_transport,
                    auto_filter_ruby=auto_filter_ruby,
                    filter_fragments=filter_fragments,
                    smart_crop=smart_crop,
                    ruby_strength=ruby_strength,
                    detector_mode=detector_mode,
                    preserve_body_pixels=preserve_body_pixels,
                )
                if shared_prepare is not None
                else _prepare_page_crops(
                    job[0], job[1], temp,
                    sensitivity=sensitivity,
                    padding_percent=padding_percent,
                    max_columns=max_columns,
                    fixed_region_rect=fixed_region_rect,
                    compact_transport=compact_primary_transport,
                    auto_filter_ruby=auto_filter_ruby,
                    filter_fragments=filter_fragments,
                    smart_crop=smart_crop,
                    ruby_strength=ruby_strength,
                    detector_mode=detector_mode,
                    preserve_body_pixels=preserve_body_pixels,
                )
            )

        pool = ThreadPoolExecutor(max_workers=worker_count)
        split_cancelled = False
        # Bound queued page preparation to two waves.  A 300-page book no longer
        # creates 300 Future objects and begins decoding far ahead of the model;
        # output order and prepared pixels remain exactly unchanged.
        from core.ocr_runtime_optimizer import OcrCancelled, iter_bounded_ordered
        try:
            prepared_pages = iter_bounded_ordered(
                pool,
                prepare_job,
                jobs,
                max_pending=max(1, worker_count * 2),
                cancel_check=cancel_check,
            )
            for page_index, prepared in enumerate(prepared_pages, start=1):
                page_path, columns, generated, error = prepared
                if cancel_check is not None and cancel_check():
                    split_cancelled = True
                    break
                if error:
                    detection_errors[page_path] = error
                else:
                    page_columns[page_path] = columns
                    for crop_path, zero_index in generated:
                        crop_paths.append(crop_path)
                        target = (page_path, zero_index)
                        crop_to_target[crop_path] = target
                        crop_input_sha256[target] = _file_sha256(crop_path)
                if phase_callback is not None:
                    label = os.path.basename(page_path)
                    if columns:
                        label = f"{label} · 预计 {len(columns)} 列"
                    phase_callback("split", page_index, len(page_paths), label)
                if callable(column_preview_callback):
                    # Even a page with zero detected columns belongs in the
                    # retained preview queue so the user can inspect the failed
                    # page manually instead of it disappearing from navigation.
                    column_preview_callback(page_path, columns or [], "split")
        except OcrCancelled:
            split_cancelled = True
        finally:
            pool.shutdown(wait=not split_cancelled, cancel_futures=split_cancelled)

        if split_cancelled:
            return

        if not crop_paths:
            details = "\n".join(
                f"• {os.path.basename(path)}：{message}"
                for path, message in list(detection_errors.items())[:10]
            )
            suffix = ("\n" + details) if details else ""
            raise RuntimeError(
                "普通分列没有生成任何正文列。请检查正文框、页面类型和扫描清晰度。"
                + suffix
            )
        # A single short/chapter-opening page may legitimately fail the geometry
        # detector while the rest of the book is usable.  Do not abort every OCR
        # model and discard completed pages.  Failed pages are emitted below as a
        # visible full-region manual-review placeholder, so the failure can never
        # become a silent omission.

        page_number_by_path = {path: index for index, path in enumerate(page_paths, start=1)}
        recognized: dict[tuple[str, int], tuple[str, float, str | None]] = {}
        candidate_map: dict[tuple[str, int], list[_TextCandidate]] = {}
        attempt_map: dict[tuple[str, int], list[str]] = {}
        selection_meta: dict[tuple[str, int], dict] = {}
        sentence_reocr_meta: dict[tuple[str, int], dict] = {}
        rescue_decisions: dict[tuple[str, int], _ColumnRescueDecision] = {}
        rescue_attempted: set[tuple[str, int]] = set()

        preprocess_options = dict(runtime_options)
        preprocess_enabled = bool(preprocess_options.get("column_preprocess_enabled", True))
        binary_enabled = bool(preprocess_options.get("column_preprocess_binary", True))
        compare_mode = str(preprocess_options.get("column_compare_mode", "full") or "full").strip().lower()
        fast_secondary = compare_mode == "fast_secondary"
        single_pass = compare_mode == "single_pass"
        sentence_reocr_enabled = bool(
            runtime_options.pop("column_sentence_context_reocr", False)
        )
        sentence_reocr_max = max(
            2,
            int(runtime_options.pop("column_sentence_context_max_columns", 10) or 10),
        )
        sentence_global_merged_box_requested = bool(
            runtime_options.pop("column_sentence_global_merged_box_reocr", False)
        )
        sentence_global_merged_box_reocr = bool(
            sentence_reocr_enabled and sentence_global_merged_box_requested
        )

        # Recognition itself used to be one opaque full-book batch.  After the
        # split phase the UI therefore appeared frozen until every column had
        # returned.  Report a normalized 0..1000 recognition timeline without
        # changing batching, model inputs, retry decisions or selected text.
        if single_pass:
            primary_end = 0.98
            recovery_end = primary_end
            wide_end = primary_end
            preprocess_end = primary_end
        elif fast_secondary:
            primary_end = 0.78
            recovery_end = primary_end
            wide_end = primary_end
            preprocess_end = 0.98
        else:
            primary_end = 0.58
            recovery_end = 0.70
            wide_end = 0.80
            preprocess_end = 0.98
        primary_scan_end = (
            max(0.08, primary_end - 0.08)
            if compact_primary_transport else primary_end
        )

        def emit_recognition_work(fraction: float, detail: str) -> None:
            if phase_callback is None:
                return
            value = max(0, min(1000, int(round(float(fraction) * 1000.0))))
            phase_callback("recognition_work", value, 1000, str(detail or "逐列识别"))

        def target_description(target: tuple[str, int] | None) -> str:
            if not target:
                return ""
            page_path, column_index = target
            page_number = page_number_by_path.get(page_path, 0)
            if page_number > 0:
                return f"第 {page_number} 页 · 第 {column_index + 1} 列"
            return f"第 {column_index + 1} 列"

        def make_batch_progress(
            start: float,
            end: float,
            stage_name: str,
            target_lookup,
            *,
            offset: int = 0,
            aggregate_total: int | None = None,
        ):
            def callback(current: int, total: int, path: str) -> None:
                total_i = max(1, int(aggregate_total or total or 1))
                current_i = max(0, min(total_i, int(offset) + int(current or 0)))
                ratio = current_i / total_i
                target = target_lookup(str(path)) if callable(target_lookup) else None
                location = target_description(target)
                detail = f"{stage_name} {current_i} / {total_i}"
                if location:
                    detail += f" · {location}"
                emit_recognition_work(start + (end - start) * ratio, detail)

            return callback

        def mark_attempt(target: tuple[str, int], method: str) -> None:
            methods = attempt_map.setdefault(target, [])
            if method not in methods:
                methods.append(method)

        def add_candidate(
            target: tuple[str, int],
            text: str,
            confidence: float,
            method: str,
        ) -> _TextCandidate | None:
            mark_attempt(target, method)
            column = page_columns[target[0]][target[1]]
            item = _make_text_candidate(text, confidence, column, method)
            if item is not None:
                candidate_map.setdefault(target, []).append(item)
            return item

        def immutable_column_id(target: tuple[str, int]) -> str:
            return f"p{page_number_by_path[target[0]]:05d}:c{target[1] + 1:03d}"

        consensus_seed_targets: set[tuple[str, int]] = set()
        if target_column_ids is not None:
            for page_path, columns in page_columns.items():
                for column_index in range(len(columns)):
                    target = (page_path, column_index)
                    column_id = immutable_column_id(target)
                    if column_id in target_column_ids:
                        continue
                    raw_seed = seed_results.get(column_id)
                    if not isinstance(raw_seed, dict):
                        continue
                    text = str(raw_seed.get("text", "") or "").strip()
                    if not text:
                        continue
                    try:
                        confidence = float(raw_seed.get("confidence", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        confidence = 0.0
                    item = add_candidate(target, text, confidence, "consensus_seed")
                    if item is not None:
                        recognized[target] = (item.text, item.confidence, None)
                        consensus_seed_targets.add(target)

        emit_recognition_work(0.0, "分列完成 · 正在加载识字模型")

        def model_load_progress(stage: str, current: int, total: int, detail: str) -> None:
            # Reserve the first 2% of recognition progress for environment/model
            # preparation.  The actual column pass starts at 2%, so the bar never
            # jumps backwards after a large 48px download completes.
            if total > 0:
                ratio = max(0.0, min(1.0, float(current) / float(total)))
            else:
                ratio = 0.0
            stage_floor = {
                "environment": 0.00,
                "connect": 0.002,
                "download": 0.003,
                "verify": 0.017,
                "source": 0.018,
                "ready": 0.019,
                "model": 0.019,
                "retry": 0.003,
            }.get(str(stage), 0.0)
            stage_span = {
                "environment": 0.002,
                "connect": 0.001,
                "download": 0.014,
                "verify": 0.001,
                "source": 0.001,
                "ready": 0.001,
                "model": 0.001,
                "retry": 0.001,
            }.get(str(stage), 0.001)
            emit_recognition_work(
                min(0.02, stage_floor + stage_span * ratio),
                str(detail or "正在准备识字模型"),
            )

        with _RecognizerSession(
            recognition_engine,
            shortcut_name=shortcut_name,
            cancel_check=cancel_check,
            verbose=verbose,
            temp=temp,
            engine_options=runtime_options,
            load_progress_callback=model_load_progress,
        ) as session:
            page_batch_requested = bool(
                target_column_ids is None
                and ndlocr_page_mode in {"page", "hybrid"}
                and recognition_engine == "ndlocr_lite"
                and fixed_region_rect is not None
            )
            page_batch_used = False
            page_batch_successful_pages: set[str] = set()
            page_batch_failed_pages: set[str] = set()
            page_stable_targets: set[tuple[str, int]] = set()
            page_only_locked_targets: set[tuple[str, int]] = set()
            page_batch_end = min(primary_scan_end, max(0.18, primary_scan_end * 0.72))
            raw_method = getattr(session, "recognize_blocks", None)
            if page_batch_requested and callable(raw_method):
                page_inputs, page_input_to_page, page_image_sizes = _prepare_ndlocr_page_inputs(
                    page_paths, page_columns, temp,
                    fixed_region_rect=fixed_region_rect,
                    preserve_body_pixels=preserve_body_pixels,
                )
                ndlocr_page_input_sha256.update({
                    page_input_to_page[path]: _file_sha256(path)
                    for path in page_inputs
                })

                def page_progress(current: int, total: int, path: str) -> None:
                    page_path = page_input_to_page.get(str(path), "")
                    page_number = page_number_by_path.get(page_path, 0)
                    detail = f"NDLOCR整页识别 {current} / {max(1, total)}"
                    if page_number:
                        detail += f" · 第 {page_number} 页"
                    emit_recognition_work(
                        0.02 + (page_batch_end - 0.02) * (max(0, current) / max(1, total)),
                        detail,
                    )

                emit_recognition_work(
                    0.02,
                    (
                        "NDLOCR高速整页 · 每页只执行一次主识别"
                        if ndlocr_page_mode == "page"
                        else "NDLOCR智能混合 · 整页识别后只补疑难列"
                    ),
                )
                raw_results = raw_method(page_inputs, progress_callback=page_progress)
                if raw_results is not None:
                    page_batch_used = True
                    for input_path in page_inputs:
                        page_path = page_input_to_page[input_path]
                        blocks, error = raw_results.get(
                            input_path, (None, "NDLOCR整页识别未返回该页")
                        )
                        if error:
                            page_batch_failed_pages.add(page_path)
                            continue
                        page_batch_successful_pages.add(page_path)
                        page_targets = {
                            (page_path, column_index)
                            for column_index in range(len(page_columns[page_path]))
                        }
                        for target in page_targets:
                            mark_attempt(target, "page_primary")
                        if ndlocr_page_mode == "page":
                            # High-speed mode deliberately performs no per-column
                            # quality rescue for a page that returned normally.
                            # Strict mode still preserves a visible placeholder
                            # for any physical slot that NDLOCR did not return.
                            page_only_locked_targets.update(page_targets)
                        routed, forced_fallback, _route_stats = _route_page_blocks_to_columns(
                            blocks,
                            page_columns[page_path],
                            page_image_sizes[page_path],
                        )
                        for column_index, (text, confidence) in routed.items():
                            if column_index in forced_fallback:
                                continue
                            target = (page_path, column_index)
                            item = add_candidate(target, text, confidence, "page_primary")
                            if item is None:
                                continue
                            needs_isolated_check = bool(
                                ndlocr_page_mode == "hybrid"
                                and _compact_transport_needs_fullsize_check(
                                    item.text,
                                    item.confidence,
                                    page_columns[page_path][column_index],
                                    recognition_engine="ndlocr_lite",
                                )
                            )
                            if needs_isolated_check:
                                # Keep the page candidate in candidate_map so the
                                # conservative selector can still use it if the
                                # isolated pass is worse, but do not mark the
                                # column complete yet.
                                continue
                            recognized[target] = (item.text, item.confidence, None)
                            page_stable_targets.add(target)
            if page_batch_used and ndlocr_page_mode == "page":
                # A genuine page-level engine error falls back to the old safe
                # path.  Missing/ambiguous columns on a successfully returned
                # page remain placeholders by design in high-speed mode.
                primary_paths = [
                    crop_path
                    for crop_path, target in crop_to_target.items()
                    if target[0] in page_batch_failed_pages
                ]
            else:
                primary_paths = [
                    crop_path
                    for crop_path, target in crop_to_target.items()
                    if target not in recognized
                ]
            primary_start = page_batch_end if page_batch_used else 0.02
            if primary_paths:
                emit_recognition_work(
                    primary_start,
                    (
                        (
                            f"智能混合逐列补识 · 剩余 {len(primary_paths)} 列"
                            if ndlocr_page_mode == "hybrid"
                            else f"整页调用失败，安全回退逐列 · 剩余 {len(primary_paths)} 列"
                        )
                        if page_batch_used
                        else "识字模型已就绪 · 开始逐列识别"
                    ),
                )
                primary = _session_recognize(
                    session,
                    primary_paths,
                    "primary",
                    progress_callback=make_batch_progress(
                        primary_start,
                        primary_scan_end,
                        (
                            "疑难列补识"
                            if page_batch_used and ndlocr_page_mode == "hybrid"
                            else "逐列识别"
                        ),
                        lambda path: crop_to_target.get(path),
                    ),
                )
                for crop_path in primary_paths:
                    target = crop_to_target[crop_path]
                    mark_attempt(target, "primary")
                    text, confidence, error = primary.get(
                        crop_path, ("", 0.0, "识字进程未返回该列")
                    )
                    if not error:
                        item = add_candidate(target, text, confidence, "primary")
                        if item is not None:
                            recognized[target] = (item.text, item.confidence, None)
            elif page_batch_used:
                if ndlocr_page_mode == "page":
                    unresolved = sum(
                        1
                        for page_path in page_batch_successful_pages
                        for column_index in range(len(page_columns[page_path]))
                        if (page_path, column_index) not in recognized
                    )
                    emit_recognition_work(
                        primary_scan_end,
                        f"NDLOCR高速整页完成 · {unresolved} 列保留待人工复核",
                    )
                else:
                    emit_recognition_work(primary_scan_end, "NDLOCR智能混合已覆盖全部物理列")

            # Adaptive rescue budget: after the primary page/column reading, each
            # physical column may enter at most one evidence-based rescue family.
            # The historical multi-route ladder remains available through the
            # explicit ``legacy`` policy, but is no longer the default.
            if column_rescue_policy == "adaptive" and (
                not single_pass or compact_primary_transport
            ):
                for page_path, columns in page_columns.items():
                    for column_index, column in enumerate(columns):
                        target = (page_path, column_index)
                        if target in page_stable_targets or target in page_only_locked_targets:
                            continue
                        attempts = attempt_map.get(target, [])
                        if "page_primary" in attempts and "primary" in attempts:
                            rescue_decisions[target] = _ColumnRescueDecision(
                                "primary",
                                "整页结果未能稳定归列，已使用一次隔离单列补识",
                            )
                            rescue_attempted.add(target)
                            continue
                        result = recognized.get(target)
                        decision = _choose_adaptive_column_rescue(
                            result[0] if result else "",
                            result[1] if result else 0.0,
                            column,
                            compact_primary=bool(
                                compact_primary_transport and "primary" in attempts
                            ),
                            recognition_engine=recognition_engine,
                        )
                        if decision is not None:
                            rescue_decisions[target] = decision

                planned = {
                    target: decision
                    for target, decision in rescue_decisions.items()
                    if target not in rescue_attempted
                    and (
                        not single_pass
                        or decision.method == "primary_fullsize_fallback"
                    )
                }
                if planned:
                    emit_recognition_work(
                        primary_end,
                        f"自适应列级救援 · {len(planned)} 列，每列最多一种路径",
                    )

                # Ordinary one-image rescue methods.
                rescue_paths_by_method: dict[str, list[str]] = {}
                rescue_path_to_target: dict[str, tuple[str, int]] = {}
                ordinary_by_page: dict[str, list[tuple[tuple[str, int], _ColumnRescueDecision]]] = {}
                for target, decision in planned.items():
                    if decision.method == "short_blocks":
                        continue
                    ordinary_by_page.setdefault(target[0], []).append((target, decision))
                serial = 0
                for page_path, specs in ordinary_by_page.items():
                    image = _open_page_image(page_path, fixed_region_rect)
                    try:
                        background = _paper_background(image)
                        for target, decision in specs:
                            serial += 1
                            _page_path, column_index = target
                            column = page_columns[page_path][column_index]
                            method = decision.method
                            mark_attempt(target, method)
                            if method == "primary_fullsize_fallback":
                                rescue_image = _crop_column(
                                    image, column, retry=False, background=background
                                )
                            elif method == "wide":
                                rescue_image = _crop_column(
                                    image, column, retry=True, background=background
                                )
                            elif method in {
                                "balanced_full",
                                "balanced_crop_2x",
                                "adaptive_binary_crop_2x",
                            }:
                                masked = _crop_column(
                                    image, column, retry=True, background=background
                                )
                                isolated = _isolated_column_image(
                                    image, column, retry=True, background=background
                                )
                                try:
                                    variant = build_fallback_variant(
                                        method,
                                        masked_page=masked,
                                        isolated_column=isolated,
                                    )
                                    rescue_image = variant.image
                                finally:
                                    masked.close()
                                    isolated.close()
                            else:
                                continue
                            rescue_image = _apply_column_cleanup(
                                rescue_image,
                                auto_filter_ruby=auto_filter_ruby,
                                filter_fragments=filter_fragments,
                                smart_crop=smart_crop,
                                ruby_strength=ruby_strength,
                                background=background,
                            )
                            path = temp / (
                                (
                                    f"primary_full_adaptive_{serial:06d}.png"
                                    if method == "primary_fullsize_fallback"
                                    else f"adaptive_rescue_{serial:06d}_{method}.png"
                                )
                            )
                            try:
                                rescue_image.save(path, format="PNG", compress_level=1)
                            finally:
                                rescue_image.close()
                            key = str(path)
                            rescue_paths_by_method.setdefault(method, []).append(key)
                            rescue_path_to_target[key] = target
                    finally:
                        image.close()

                method_labels = {
                    "primary_fullsize_fallback": "原尺寸兼容复核",
                    "wide": "扩边复核",
                    "balanced_full": "纸张平衡复核",
                    "balanced_crop_2x": "局部平衡放大",
                    "adaptive_binary_crop_2x": "局部二值化复核",
                }
                ordinary_total = sum(len(paths) for paths in rescue_paths_by_method.values())
                ordinary_done = 0
                for method, paths in rescue_paths_by_method.items():
                    if cancel_check is not None and cancel_check():
                        break
                    results = _session_recognize(
                        session,
                        paths,
                        method,
                        progress_callback=make_batch_progress(
                            primary_end,
                            preprocess_end,
                            method_labels.get(method, "自适应救援"),
                            lambda path: rescue_path_to_target.get(path),
                            offset=ordinary_done,
                            aggregate_total=max(1, ordinary_total),
                        ),
                    )
                    for path in paths:
                        target = rescue_path_to_target[path]
                        rescue_attempted.add(target)
                        text, confidence, error = results.get(
                            path, ("", 0.0, "自适应救援未返回")
                        )
                        if error:
                            continue
                        item = add_candidate(target, text, confidence, method)
                        if item is None:
                            continue
                        previous = recognized.get(target)
                        column = page_columns[target[0]][target[1]]
                        if previous is None or _prefer_recovered_text(
                            previous[0], item.text, column
                        ):
                            recognized[target] = (item.text, item.confidence, None)
                    ordinary_done += len(paths)

                # Separated short blocks are one rescue family even though each
                # visible span is isolated before the parts are merged.
                short_paths: list[str] = []
                short_path_to_target: dict[str, tuple[tuple[str, int], int]] = {}
                short_by_page: dict[str, list[tuple[str, int]]] = {}
                for target, decision in planned.items():
                    if decision.method == "short_blocks":
                        short_by_page.setdefault(target[0], []).append(target)
                for page_path, targets in short_by_page.items():
                    image = _open_page_image(page_path, fixed_region_rect)
                    try:
                        background = _paper_background(image)
                        for target in targets:
                            mark_attempt(target, "short_blocks")
                            _page_path, column_index = target
                            column = page_columns[page_path][column_index]
                            for span_index, span in enumerate(column.content_spans):
                                crop = _crop_content_span(
                                    image, column, span, background=background
                                )
                                crop = _apply_column_cleanup(
                                    crop,
                                    auto_filter_ruby=auto_filter_ruby,
                                    filter_fragments=filter_fragments,
                                    smart_crop=smart_crop,
                                    ruby_strength=ruby_strength,
                                    background=background,
                                )
                                path = temp / (
                                    f"adaptive_short_p{page_number_by_path[page_path]:05d}_"
                                    f"c{column_index + 1:03d}_s{span_index + 1:02d}.png"
                                )
                                try:
                                    crop.save(path, format="PNG", compress_level=1)
                                finally:
                                    crop.close()
                                key = str(path)
                                short_paths.append(key)
                                short_path_to_target[key] = (target, span_index)
                    finally:
                        image.close()
                if short_paths and not (cancel_check is not None and cancel_check()):
                    results = _session_recognize(
                        session,
                        short_paths,
                        "short_blocks",
                        progress_callback=make_batch_progress(
                            primary_end,
                            preprocess_end,
                            "分离短段恢复",
                            lambda path: short_path_to_target.get(path, (None, 0))[0],
                        ),
                    )
                    grouped: dict[tuple[str, int], list[tuple[int, str, float]]] = {}
                    for path, (target, span_index) in short_path_to_target.items():
                        rescue_attempted.add(target)
                        text, confidence, error = results.get(
                            path, ("", 0.0, "分离短段恢复未返回")
                        )
                        if error or not _valid_column_text(text):
                            continue
                        grouped.setdefault(target, []).append(
                            (span_index, text, confidence)
                        )
                    for target, parts in grouped.items():
                        parts.sort(key=lambda item: item[0])
                        merged = _merge_text_parts([item[1] for item in parts])
                        confidence = max([item[2] for item in parts] or [0.0])
                        item = add_candidate(target, merged, confidence, "short_blocks")
                        if item is None:
                            continue
                        previous = recognized.get(target)
                        column = page_columns[target[0]][target[1]]
                        if previous is None or _prefer_recovered_text(
                            previous[0], item.text, column
                        ):
                            recognized[target] = (item.text, item.confidence, None)
                if planned:
                    emit_recognition_work(
                        preprocess_end,
                        f"自适应列级救援完成 · 已执行 {len(rescue_attempted)} 列",
                    )

            # Compact transport is lossless but intentionally changes only the
            # amount of blank canvas around the glyphs.  To preserve compatibility,
            # any empty/invalid compact result is retried once with the original
            # full-size masked page before the normal recovery ladder begins.
            compact_fallback_targets = []
            if column_rescue_policy == "legacy" and compact_primary_transport:
                for target in crop_to_target.values():
                    # Page-first NDLOCR did not use a compact column image, so
                    # the compact/full-size compatibility check must not create
                    # an unrelated second pass for stable page-routed results or
                    # page-only placeholders.
                    if "primary" not in attempt_map.get(target, []):
                        continue
                    result = recognized.get(target)
                    if result is None or _compact_transport_needs_fullsize_check(
                        result[0], result[1], page_columns[target[0]][target[1]],
                        recognition_engine=recognition_engine,
                    ):
                        compact_fallback_targets.append(target)
            compact_fallback_paths: list[str] = []
            compact_fallback_to_target: dict[str, tuple[str, int]] = {}
            fallback_by_page: dict[str, list[tuple[str, int]]] = {}
            for target in compact_fallback_targets:
                fallback_by_page.setdefault(target[0], []).append(target)
            for page_path, targets in fallback_by_page.items():
                image = _open_page_image(page_path, fixed_region_rect)
                try:
                    background = _paper_background(image)
                    for target in targets:
                        _page_path, column_index = target
                        column = page_columns[page_path][column_index]
                        masked = _crop_column(
                            image, column, retry=False, background=background
                        )
                        masked = _apply_column_cleanup(
                            masked,
                            auto_filter_ruby=auto_filter_ruby,
                            filter_fragments=filter_fragments,
                            smart_crop=smart_crop,
                            ruby_strength=ruby_strength,
                            background=background,
                        )
                        path = temp / (
                            f"primary_full_p{page_number_by_path[page_path]:05d}_"
                            f"c{column_index + 1:03d}.png"
                        )
                        try:
                            masked.save(path, format="PNG", compress_level=1)
                        finally:
                            masked.close()
                        key = str(path)
                        compact_fallback_paths.append(key)
                        compact_fallback_to_target[key] = target
                finally:
                    image.close()
            if compact_fallback_paths and not (cancel_check is not None and cancel_check()):
                fallback_results = _session_recognize(
                    session,
                    compact_fallback_paths,
                    "primary_fullsize_fallback",
                    progress_callback=make_batch_progress(
                        primary_scan_end,
                        primary_end,
                        "全尺寸兼容回退",
                        lambda path: compact_fallback_to_target.get(path),
                    ),
                )
                for path, target in compact_fallback_to_target.items():
                    text, confidence, error = fallback_results.get(
                        path, ("", 0.0, "全尺寸兼容回退未返回")
                    )
                    if not error:
                        item = add_candidate(
                            target, text, confidence, "primary_fullsize_fallback"
                        )
                        if item is not None:
                            recognized[target] = (item.text, item.confidence, None)
            if column_rescue_policy == "legacy":
                emit_recognition_work(
                    primary_end,
                    (
                        f"紧凑列图完成 · {len(compact_fallback_paths)} 列执行全尺寸兼容回退"
                        if compact_primary_transport
                        else "逐列主识别完成 · 检查需要恢复的列"
                    ),
                )

            # Targeted short-block recovery: only suspicious multi-block or
            # severely under-recognized columns are split vertically.
            recovery_paths: list[str] = []
            recovery_to_target: dict[str, tuple[tuple[str, int], int]] = {}
            recovery_targets = [] if (
                column_rescue_policy != "legacy" or fast_secondary or single_pass
            ) else [
                (page_path, column_index)
                for page_path, columns in page_columns.items()
                for column_index, column in enumerate(columns)
                if (page_path, column_index) not in page_stable_targets
                and (page_path, column_index) not in page_only_locked_targets
                and len(column.content_spans) >= 2
                and (
                    (page_path, column_index) not in recognized
                    or _needs_short_block_recovery(
                        recognized[(page_path, column_index)][0], column
                    )
                )
            ]
            recovery_by_page: dict[str, list[tuple[str, int]]] = {}
            for target in recovery_targets:
                recovery_by_page.setdefault(target[0], []).append(target)
            for page_path, targets in recovery_by_page.items():
                image = _open_page_image(page_path, fixed_region_rect)
                try:
                    background = _paper_background(image)
                    for target in targets:
                        mark_attempt(target, "short_blocks")
                        _page_path, column_index = target
                        column = page_columns[page_path][column_index]
                        for span_index, span in enumerate(column.content_spans):
                            crop = _crop_content_span(
                                image, column, span, background=background
                            )
                            crop = _apply_column_cleanup(
                                crop,
                                auto_filter_ruby=auto_filter_ruby,
                                filter_fragments=filter_fragments,
                                smart_crop=smart_crop,
                                ruby_strength=ruby_strength,
                                background=background,
                            )
                            path = temp / (
                                f"recover_p{page_number_by_path[page_path]:05d}_"
                                f"c{column_index+1:03d}_s{span_index+1:02d}.png"
                            )
                            crop.save(path, format="PNG", compress_level=1)
                            crop.close()
                            key = str(path)
                            recovery_paths.append(key)
                            recovery_to_target[key] = (target, span_index)
                finally:
                    image.close()

            if recovery_paths and not (cancel_check is not None and cancel_check()):
                recovery_results = _session_recognize(
                    session,
                    recovery_paths,
                    "short_blocks",
                    progress_callback=make_batch_progress(
                        primary_end,
                        recovery_end,
                        "短块恢复",
                        lambda path: (
                            recovery_to_target.get(path, (None, 0))[0]
                            if path in recovery_to_target else None
                        ),
                    ),
                )
                grouped: dict[tuple[str, int], list[tuple[int, str, float]]] = {}
                for path, (target, span_index) in recovery_to_target.items():
                    text, confidence, error = recovery_results.get(path, ("", 0.0, "未返回短块"))
                    if error or not _valid_column_text(text):
                        continue
                    grouped.setdefault(target, []).append((span_index, text, confidence))
                for target, parts in grouped.items():
                    parts.sort(key=lambda item: item[0])
                    recovered_text = _merge_text_parts([item[1] for item in parts])
                    recovered_conf = max([item[2] for item in parts] or [0.0])
                    item = add_candidate(target, recovered_text, recovered_conf, "short_blocks")
                    if item is None:
                        continue
                    previous = recognized.get(target)
                    primary_text = previous[0] if previous else ""
                    column = page_columns[target[0]][target[1]]
                    if _prefer_recovered_text(primary_text, item.text, column):
                        recognized[target] = (item.text, item.confidence, None)
            if column_rescue_policy == "legacy":
                emit_recognition_work(recovery_end, "短块恢复检查完成 · 准备扩边重试")

            unresolved_or_suspicious: list[tuple[str, int]] = []
            if column_rescue_policy == "legacy":
                for page_path, columns in page_columns.items():
                    for column_index, column in enumerate(columns):
                        target = (page_path, column_index)
                        if target in page_stable_targets or target in page_only_locked_targets:
                            continue
                        result = recognized.get(target)
                        if single_pass:
                            continue
                        if result is None or (
                            not fast_secondary and _needs_additional_retry(result[0], column)
                        ):
                            unresolved_or_suspicious.append(target)

            # Wider full-page masked retry.
            retry_paths: list[str] = []
            retry_to_target: dict[str, tuple[str, int]] = {}
            retry_by_page: dict[str, list[tuple[str, int]]] = {}
            if not fast_secondary:
                for target in unresolved_or_suspicious:
                    retry_by_page.setdefault(target[0], []).append(target)
            retry_serial = 0
            for page_path, targets in retry_by_page.items():
                image = _open_page_image(page_path, fixed_region_rect)
                try:
                    background = _paper_background(image)
                    for target in targets:
                        retry_serial += 1
                        mark_attempt(target, "wide")
                        _page_path, column_index = target
                        crop = _crop_column(
                            image, page_columns[page_path][column_index],
                            retry=True, background=background,
                        )
                        crop = _apply_column_cleanup(
                            crop,
                            auto_filter_ruby=auto_filter_ruby,
                            filter_fragments=filter_fragments,
                            smart_crop=smart_crop,
                            ruby_strength=ruby_strength,
                            background=background,
                        )
                        retry_path = temp / f"retry_{retry_serial:06d}.png"
                        crop.save(retry_path, format="PNG", compress_level=1)
                        crop.close()
                        key = str(retry_path)
                        retry_paths.append(key)
                        retry_to_target[key] = target
                finally:
                    image.close()

            if retry_paths and not (cancel_check is not None and cancel_check()):
                retry_results = _session_recognize(
                    session,
                    retry_paths,
                    "wide",
                    progress_callback=make_batch_progress(
                        recovery_end,
                        wide_end,
                        "扩边重试",
                        lambda path: retry_to_target.get(path),
                    ),
                )
                for retry_path, target in retry_to_target.items():
                    text, confidence, error = retry_results.get(
                        retry_path, ("", 0.0, "识字进程未返回重试列")
                    )
                    if error:
                        continue
                    item = add_candidate(target, text, confidence, "wide")
                    if item is None:
                        continue
                    previous = recognized.get(target)
                    column = page_columns[target[0]][target[1]]
                    if previous is None or _prefer_recovered_text(previous[0], item.text, column):
                        recognized[target] = (item.text, item.confidence, None)
            if column_rescue_policy == "legacy":
                emit_recognition_work(wide_end, "扩边重试检查完成 · 准备增强候选")

            # Multi-route image preprocessing is deliberately targeted.  Good
            # original OCR columns are never reprocessed, while empty/short/quote-
            # broken columns receive three independent fallback images.
            preprocess_targets: list[tuple[str, int]] = []
            if column_rescue_policy == "legacy" and preprocess_enabled and not single_pass:
                for page_path, columns in page_columns.items():
                    for column_index, column in enumerate(columns):
                        target = (page_path, column_index)
                        if target in page_stable_targets or target in page_only_locked_targets:
                            continue
                        result = recognized.get(target)
                        if result is None or _needs_additional_retry(result[0], column):
                            preprocess_targets.append(target)

            variant_paths: dict[str, list[str]] = {}
            variant_to_target: dict[str, tuple[tuple[str, int], str]] = {}
            variant_specs: list[tuple[tuple[str, int], list[str], dict[str, Path], list[str]]] = []
            cache_base = shared_variants if shared_variants is not None else temp
            for target in preprocess_targets:
                page_path, column_index = target
                page_number = page_number_by_path[page_path]
                # Secondary comparison models normally need only an independent
                # primary reading.  If that reading is empty, try one compact 2x
                # fallback instead of repeating the full five-pass rescue ladder.
                expected_methods = (
                    ["balanced_crop_2x"]
                    if fast_secondary
                    else ["balanced_full", "balanced_crop_2x"]
                )
                if binary_enabled and not fast_secondary:
                    expected_methods.append("adaptive_binary_crop_2x")
                cached_paths = {
                    method: cache_base / f"p{page_number:05d}_c{column_index+1:03d}_{method}.png"
                    for method in expected_methods
                }
                missing = [
                    method for method, path in cached_paths.items()
                    if not path.exists() or path.stat().st_size <= 0
                ]
                variant_specs.append((target, expected_methods, cached_paths, missing))

            missing_by_page: dict[str, list[tuple[tuple[str, int], dict[str, Path], list[str]]]] = {}
            for target, _methods, cached_paths, missing in variant_specs:
                if missing:
                    missing_by_page.setdefault(target[0], []).append((target, cached_paths, missing))
            for page_path, specs in missing_by_page.items():
                image = _open_page_image(page_path, fixed_region_rect)
                try:
                    background = _paper_background(image)
                    for target, cached_paths, missing in specs:
                        _page_path, column_index = target
                        column = page_columns[page_path][column_index]
                        masked = _crop_column(
                            image, column, retry=True, background=background
                        )
                        isolated = _isolated_column_image(
                            image, column, retry=True, background=background
                        )
                        masked = _apply_column_cleanup(
                            masked,
                            auto_filter_ruby=auto_filter_ruby,
                            filter_fragments=filter_fragments,
                            smart_crop=smart_crop,
                            ruby_strength=ruby_strength,
                            background=background,
                        )
                        isolated = _apply_column_cleanup(
                            isolated,
                            auto_filter_ruby=auto_filter_ruby,
                            filter_fragments=filter_fragments,
                            smart_crop=smart_crop,
                            ruby_strength=ruby_strength,
                            background=background,
                        )
                        variants = []
                        try:
                            variants = build_fallback_variants(
                                masked_page=masked, isolated_column=isolated,
                                include_binary=binary_enabled,
                            )
                            for variant in variants:
                                if variant.name in missing:
                                    variant.image.save(
                                        cached_paths[variant.name], format="PNG", compress_level=1
                                    )
                        finally:
                            close_variants(variants)
                            masked.close()
                            isolated.close()
                finally:
                    image.close()

            for target, expected_methods, cached_paths, _missing in variant_specs:
                for method in expected_methods:
                    mark_attempt(target, method)
                    key = str(cached_paths[method])
                    variant_paths.setdefault(method, []).append(key)
                    variant_to_target[key] = (target, method)

            total_variant_paths = sum(len(paths) for paths in variant_paths.values())
            completed_variant_paths = 0
            for method, paths in variant_paths.items():
                if cancel_check is not None and cancel_check():
                    break
                method_labels = {
                    "balanced_full": "平衡增强",
                    "balanced_crop_2x": "局部放大增强",
                    "adaptive_binary_crop_2x": "二值化增强",
                }
                results = _session_recognize(
                    session,
                    paths,
                    method,
                    progress_callback=make_batch_progress(
                        wide_end,
                        preprocess_end,
                        method_labels.get(method, "增强候选识别"),
                        lambda path: (
                            variant_to_target.get(path, (None, ""))[0]
                            if path in variant_to_target else None
                        ),
                        offset=completed_variant_paths,
                        aggregate_total=max(1, total_variant_paths),
                    ),
                )
                for path in paths:
                    target, _ = variant_to_target[path]
                    text, confidence, error = results.get(path, ("", 0.0, "增强识别未返回"))
                    if not error:
                        add_candidate(target, text, confidence, method)
                completed_variant_paths += len(paths)
            if column_rescue_policy == "legacy":
                emit_recognition_work(preprocess_end, "增强候选检查完成 · 正在选择最终列文本")

            # Select conservatively from all valid candidates.  Exact agreement
            # gains consensus points; a radically different enhanced result does
            # not replace a plausible original and is instead marked as conflict.
            recognized.clear()
            for page_path, columns in page_columns.items():
                for column_index, column in enumerate(columns):
                    target = (page_path, column_index)
                    selected, conflict, ranked = _select_text_candidate(
                        candidate_map.get(target, []), column
                    )
                    if selected is None:
                        continue
                    recognized[target] = (selected.text, selected.confidence, None)
                    rescue_decision = rescue_decisions.get(target)
                    legacy_rescue_methods = [
                        method for method in attempt_map.get(target, [])
                        if method not in {"primary", "page_primary", "consensus_seed"}
                    ]
                    rescue_used = bool(
                        target in rescue_attempted or legacy_rescue_methods
                    )
                    rescue_method = (
                        rescue_decision.method
                        if rescue_decision and rescue_used
                        else (legacy_rescue_methods[-1] if legacy_rescue_methods else "")
                    )
                    rescue_reason = (
                        rescue_decision.reason
                        if rescue_decision and rescue_used
                        else (
                            "兼容旧版完整多轮恢复链"
                            if legacy_rescue_methods else ""
                        )
                    )
                    selection_meta[target] = {
                        "selected_method": selected.method,
                        "conflict": bool(conflict),
                        "consensus_seeded": bool(
                            seed_mark_selective and selected.method == "consensus_seed"
                        ),
                        "rescue_policy": column_rescue_policy,
                        "rescue_budget": (
                            1 if column_rescue_policy == "adaptive"
                            else (-1 if column_rescue_policy == "legacy" else 0)
                        ),
                        "rescue_used": rescue_used,
                        "rescue_method": rescue_method,
                        "rescue_reason": rescue_reason,
                        "candidates": [
                            {
                                "method": item.method,
                                "text": item.text,
                                "confidence": round(item.confidence, 4),
                                "score": round(item.score, 3),
                            }
                            for item in ranked[:4]
                        ],
                    }
            emit_recognition_work(1.0, "逐列识别完成 · 正在构建整句上下文")

            if sentence_reocr_enabled and not (cancel_check is not None and cancel_check()):
                ordered_targets = [
                    (page_path, column_index)
                    for page_path in page_paths
                    for column_index in range(len(page_columns.get(page_path, [])))
                ]
                groups = _sentence_reocr_groups(
                    ordered_targets,
                    recognized,
                    max_columns=sentence_reocr_max,
                )
                selected_groups: list[tuple[int, list[tuple[str, int]]]] = []
                skipped_group_count = 0
                for group_index, targets in enumerate(groups, start=1):
                    column_ids = [immutable_column_id(target) for target in targets]
                    should_reocr = True
                    skip_reason = "完整模式：所有完整多列句组均重识别"
                    if sentence_target_ids is not None and not any(
                        column_id in sentence_target_ids for column_id in column_ids
                    ):
                        should_reocr = False
                        skip_reason = "多模型分列结果已一致，跳过重复整句 OCR"
                    elif sentence_reocr_strategy == "smart":
                        should_reocr, skip_reason = _sentence_group_smart_reocr_decision(
                            targets,
                            recognized,
                            selection_meta,
                            page_columns,
                            recognition_engine=recognition_engine,
                        )
                    if should_reocr:
                        selected_groups.append((group_index, list(targets)))
                        continue
                    skipped_group_count += 1
                    baseline = join_column_parts(
                        recognized.get(target, ("□", 0.0, None))[0]
                        for target in targets
                    )
                    owner = targets[0]
                    page_runs = [
                        {
                            "page": page_number_by_path[page_path],
                            "column_indices": [index + 1 for index in indices],
                        }
                        for page_path, indices in _sentence_group_page_runs(targets)
                    ]
                    common_meta = {
                        "sentence_context_reocr_group": group_index,
                        "sentence_context_reocr_column_ids": column_ids,
                        "sentence_context_reocr_column_count": len(targets),
                        "sentence_context_reocr_layout": "smart_baseline",
                        "sentence_context_reocr_page_runs": page_runs,
                        "sentence_context_reocr_candidate": "",
                        "sentence_context_reocr_baseline": baseline,
                        "sentence_context_reocr_confidence": 0.0,
                        "sentence_context_reocr_accepted": False,
                        "sentence_context_reocr_skipped": True,
                        "sentence_context_reocr_strategy": sentence_reocr_strategy,
                        "sentence_context_reocr_reason": skip_reason,
                        "sentence_context_reocr_owner_column_id": column_ids[0],
                    }
                    for position, target in enumerate(targets):
                        sentence_reocr_meta[target] = {
                            **common_meta,
                            "sentence_context_reocr_position": position,
                            "sentence_context_reocr_owner": target == owner,
                        }

                if phase_callback is not None:
                    phase_callback(
                        "sentence_reocr",
                        0,
                        max(1, len(selected_groups)),
                        (
                            f"整句智能筛选：完整句组 {len(groups)}，"
                            f"需重识别 {len(selected_groups)}，"
                            f"直接采用逐列底稿 {skipped_group_count}"
                            if sentence_reocr_strategy == "smart"
                            else f"完整模式：准备重识别 {len(selected_groups)} 个句组"
                        ),
                    )

                group_paths: list[str] = []
                path_to_group: dict[
                    str, tuple[int, list[tuple[str, int]], str]
                ] = {}
                sentence_cache_dir = shared_sentences if shared_sentences is not None else temp
                with _SentencePageCache(fixed_region_rect, max_pages=4) as sentence_page_cache:
                    for group_index, targets in selected_groups:
                        cache_key = _sentence_group_cache_key(
                            targets, page_number_by_path,
                            global_merged=sentence_global_merged_box_reocr,
                            fixed_region_rect=fixed_region_rect,
                        )
                        if sentence_global_merged_box_reocr:
                            cached_global = sentence_cache_dir / (
                                f"sentence_{cache_key}_global_merged_boxes.png"
                            )
                            cached_fallback = sentence_cache_dir / (
                                f"sentence_{cache_key}_column_strips_fallback.png"
                            )
                            if cached_global.exists() and cached_global.stat().st_size > 0:
                                path = cached_global
                                layout_name = "global_merged_boxes"
                            elif cached_fallback.exists() and cached_fallback.stat().st_size > 0:
                                path = cached_fallback
                                layout_name = "column_strips_fallback"
                            else:
                                try:
                                    canvas = _sentence_group_merged_box_image(
                                        page_columns, targets,
                                        fixed_region_rect=fixed_region_rect,
                                        shared_page_cache=sentence_page_cache,
                                    )
                                    path = cached_global
                                    layout_name = "global_merged_boxes"
                                except Exception:
                                    # Geometry anomalies must not abort a whole-book OCR.
                                    # Fall back to the proven lossless strip layout for
                                    # only this sentence group.
                                    canvas = _sentence_group_image(
                                        page_columns, targets,
                                        fixed_region_rect=fixed_region_rect,
                                        shared_page_cache=sentence_page_cache,
                                    )
                                    path = cached_fallback
                                    layout_name = "column_strips_fallback"
                                try:
                                    temporary = path.with_suffix(path.suffix + ".tmp")
                                    canvas.save(temporary, format="PNG", compress_level=1)
                                    temporary.replace(path)
                                finally:
                                    canvas.close()
                        else:
                            path = sentence_cache_dir / (
                                f"sentence_{cache_key}_column_strips.png"
                            )
                            layout_name = "column_strips"
                            if not path.exists() or path.stat().st_size <= 0:
                                canvas = _sentence_group_image(
                                    page_columns, targets,
                                    fixed_region_rect=fixed_region_rect,
                                    shared_page_cache=sentence_page_cache,
                                )
                                try:
                                    temporary = path.with_suffix(path.suffix + ".tmp")
                                    canvas.save(temporary, format="PNG", compress_level=1)
                                    temporary.replace(path)
                                finally:
                                    canvas.close()
                        key = str(path)
                        group_paths.append(key)
                        path_to_group[key] = (group_index, list(targets), layout_name)
                        for target in targets:
                            mark_attempt(target, "sentence_context")

                if not group_paths and phase_callback is not None:
                    phase_callback(
                        "sentence_reocr",
                        1,
                        1,
                        (
                            "所有完整句组的逐列底稿均通过智能安全检查，无需重复 OCR"
                            if sentence_reocr_strategy == "smart" and groups
                            else "没有需要整句重识别的完整多列句组"
                        ),
                    )

                if group_paths:
                    total_groups = len(group_paths)
                    group_results = _session_recognize(
                        session,
                        group_paths,
                        "sentence_context",
                        progress_callback=(
                            (
                                lambda current, total, path: phase_callback(
                                    "sentence_reocr",
                                    current,
                                    max(total, 1),
                                    f"正在识别句组 {current} / {max(total, 1)}",
                                )
                            )
                            if phase_callback is not None else None
                        ),
                    )
                    for current, path in enumerate(group_paths, start=1):
                        group_index, targets, group_layout = path_to_group[path]
                        text, confidence, error = group_results.get(
                            path, ("", 0.0, "句组 OCR 未返回")
                        )
                        primary_parts = [
                            recognized.get(target, ("□", 0.0, None))[0]
                            for target in targets
                        ]
                        if error:
                            accepted = False
                            reason = str(error)
                            baseline = join_column_parts(primary_parts)
                            candidate = ""
                        else:
                            accepted, reason, baseline, candidate = (
                                _validate_sentence_reocr_candidate(
                                    primary_parts, text, confidence,
                                    recognition_engine=recognition_engine,
                                )
                            )
                        column_ids = [
                            f"p{page_number_by_path[target[0]]:05d}:c{target[1] + 1:03d}"
                            for target in targets
                        ]
                        owner = targets[0]
                        page_runs = [
                            {
                                "page": page_number_by_path[page_path],
                                "column_indices": [index + 1 for index in indices],
                            }
                            for page_path, indices in _sentence_group_page_runs(targets)
                        ]
                        common_meta = {
                            "sentence_context_reocr_group": group_index,
                            "sentence_context_reocr_column_ids": column_ids,
                            "sentence_context_reocr_column_count": len(targets),
                            "sentence_context_reocr_layout": group_layout,
                            "sentence_context_reocr_page_runs": page_runs,
                            "sentence_context_reocr_candidate": candidate,
                            "sentence_context_reocr_baseline": baseline,
                            "sentence_context_reocr_confidence": round(float(confidence or 0.0), 4),
                            "sentence_context_reocr_accepted": bool(accepted),
                            "sentence_context_reocr_skipped": False,
                            "sentence_context_reocr_strategy": sentence_reocr_strategy,
                            "sentence_context_reocr_reason": reason,
                            "sentence_context_reocr_owner_column_id": column_ids[0],
                            # Transient exact image used by sentence-context OCR.
                            # The GUI keeps the parent OCR temp root alive until
                            # OCR is cleared or the app exits; proofreading can
                            # therefore show the exact single/multi-column input.
                            "ocr_review_sentence_image_path": path,
                        }
                        for position, target in enumerate(targets):
                            sentence_reocr_meta[target] = {
                                **common_meta,
                                "sentence_context_reocr_position": position,
                                "sentence_context_reocr_owner": target == owner,
                            }
                        if phase_callback is not None:
                            phase_callback(
                                "sentence_reocr",
                                current,
                                total_groups,
                                f"句组 {group_index} · {len(targets)} 列 · "
                                f"{('合并框' if group_layout == 'global_merged_boxes' else ('合并框失败后条带回退' if group_layout == 'column_strips_fallback' else '逐列拼接'))} · "
                                f"{'采用' if accepted else '回退逐列'}",
                            )

        unresolved = [
            (page_path, column_index)
            for page_path, columns in page_columns.items()
            for column_index in range(len(columns))
            if (page_path, column_index) not in recognized
        ]
        cancelled = bool(cancel_check is not None and cancel_check())
        unresolved_set = set(unresolved)
        output_page_paths = list(page_paths)
        if cancelled:
            completed_page_numbers = [
                page_number_by_path.get(page_path, 0)
                for page_path, _column_index in recognized
            ]
            last_completed_page = max(completed_page_numbers or [0])
            output_page_paths = page_paths[:last_completed_page]

        # A physical column that survives projection detection but returns no OCR
        # text after primary, short-block and wide retries must never disappear.
        # Older strict mode aborted the whole book here.  That protected against
        # silent loss, but made the manual-review workflow unusable.  Preserve an
        # explicit placeholder block instead: the fixed bbox reaches the document,
        # risk analysis marks it at 100/100, and the native correction dialog opens
        # the exact column image for full-column Japanese IME/handwriting input.
        # ``□`` is intentional: if the user skips review it remains visible in the
        # exported text, so an unresolved column can never become silent omission.

        for page_index, page_path in enumerate(output_page_paths, start=1):
            if page_path in detection_errors:
                error_message = str(detection_errors[page_path] or "普通分列失败")
                try:
                    with Image.open(page_path) as opened:
                        width, height = opened.size
                        bounds = _normalized_body_bounds(opened, fixed_region_rect)
                except Exception:
                    width, height = (1, 1)
                    bounds = None
                if bounds is None:
                    left, top, right, bottom = (0, 0, max(1, width), max(1, height))
                else:
                    left, top, right, bottom = bounds
                column_id = f"p{page_index:05d}:c001"
                placeholder = {
                    "text": "□",
                    "confidence": 0.0,
                    "box": [[left, top], [right, top], [right, bottom], [left, bottom]],
                    "direction": "vertical",
                    "layout_group": "fixed_region_column",
                    "layout_order": 0,
                    "recognizer": recognition_engine,
                    "column_id": column_id,
                    "column_index": 1,
                    "column_expected_count": 1,
                    "black_ink_estimated_chars": 0,
                    "black_ink_content_spans": [],
                    "preserve_ocr_item": True,
                    "preserve_empty_ocr_column": True,
                    "column_ocr_empty": True,
                    "column_requires_handwriting": True,
                    "column_manual_placeholder": True,
                    "column_detection_failed": True,
                    "column_count_unverified": True,
                    "column_detector_mode": detector_mode,
                    "column_detector_version": column_detector_version(detector_mode),
                    "column_ocr_failure_reason": "column_detection_failed",
                    "column_detection_error": error_message,
                    "column_ocr_attempts": ["component_geometry_detection"],
                    "column_ocr_selected_variant": "manual_pending",
                    "column_ocr_compare_mode": compare_mode,
                    "column_auto_filter_ruby": bool(auto_filter_ruby),
                    "column_filter_fragments": bool(filter_fragments),
                    "column_smart_crop": bool(smart_crop),
                    "column_ruby_strength": ruby_strength,
                    "column_preserve_body_pixels": bool(preserve_body_pixels),
                    "column_ocr_input_contract": (
                        "lossless_hard_slot_v1" if preserve_body_pixels else "legacy_filtered_body_v1"
                    ),
                    "column_ocr_input_profile": input_profile,
                    "column_ocr_input_profile_sha256": input_profile_fingerprint,
                    "column_ocr_rescue_policy": column_rescue_policy,
                }
                if callable(column_preview_callback):
                    column_preview_callback(page_path, [], "recognition")
                if phase_callback is not None:
                    phase_callback(
                        "document_build",
                        page_index,
                        len(page_paths),
                        f"{os.path.basename(page_path)} · 分列失败，已保留人工复核占位",
                    )
                yield page_path, [placeholder], None
                continue
            columns = page_columns.get(page_path, [])
            expected = len(columns)
            blocks: list[dict] = []
            text_recognized_count = 0
            manual_pending_count = 0
            rescue_count = 0
            for column_index, column in enumerate(columns):
                target_key = (page_path, column_index)
                result = recognized.get(target_key)
                is_unresolved = target_key in unresolved_set or result is None
                if is_unresolved:
                    text = "□"
                    confidence = 0.0
                    manual_pending_count += 1
                else:
                    text, confidence, _error = result
                    text_recognized_count += 1
                column_id = f"p{page_index:05d}:c{column_index + 1:03d}"
                selected_meta = selection_meta.get(target_key, {})
                selected_method = str(selected_meta.get("selected_method", "") or "")
                if bool(selected_meta.get("rescue_used", False)):
                    rescue_count += 1
                column_attempts = list(attempt_map.get(target_key, ["primary"]))
                block = {
                    "text": text,
                    "confidence": confidence if not is_unresolved else 0.0,
                    "box": column.polygon(),
                    "direction": "vertical",
                    "layout_group": "fixed_region_column",
                    "layout_order": column_index,
                    "recognizer": recognition_engine,
                    "column_id": column_id,
                    "column_index": column_index + 1,
                    "column_expected_count": expected,
                    "black_ink_estimated_chars": int(column.estimated_chars or 0),
                    "black_ink_content_spans": [list(span) for span in column.content_spans],
                    "preserve_ocr_item": True,
                    "column_ocr_attempts": column_attempts,
                    "column_ocr_selected_variant": selected_method or ("manual_pending" if is_unresolved else "primary"),
                    "column_ocr_preprocess_used": selected_method in {
                        "balanced_full",
                        "balanced_crop_2x",
                        "adaptive_binary_crop_2x",
                    },
                    "column_ocr_candidate_conflict": bool(selected_meta.get("conflict", False)),
                    "column_consensus_seeded": bool(selected_meta.get("consensus_seeded", False)),
                    "column_ocr_candidates": list(selected_meta.get("candidates", [])),
                    "column_ocr_compare_mode": compare_mode,
                    "column_auto_filter_ruby": bool(auto_filter_ruby),
                    "column_filter_fragments": bool(filter_fragments),
                    "column_smart_crop": bool(smart_crop),
                    "column_ruby_strength": ruby_strength,
                    "column_preserve_body_pixels": bool(preserve_body_pixels),
                    "column_ocr_input_contract": (
                        "lossless_hard_slot_v1" if preserve_body_pixels else "legacy_filtered_body_v1"
                    ),
                    "column_ocr_input_profile": input_profile,
                    "column_ocr_input_profile_sha256": input_profile_fingerprint,
                    "column_ocr_input_sha256": crop_input_sha256.get(target_key, ""),
                    "column_ndlocr_page_input_sha256": (
                        ndlocr_page_input_sha256.get(page_path, "")
                        if recognition_engine == "ndlocr_lite" else ""
                    ),
                    "column_ocr_rescue_policy": column_rescue_policy,
                    "column_ocr_rescue_budget": int(selected_meta.get("rescue_budget", 0) or 0),
                    "column_ocr_rescue_used": bool(selected_meta.get("rescue_used", False)),
                    "column_ocr_rescue_method": str(selected_meta.get("rescue_method", "") or ""),
                    "column_ocr_rescue_reason": str(selected_meta.get("rescue_reason", "") or ""),
                    "column_ndlocr_page_mode": (
                        ndlocr_page_mode if recognition_engine == "ndlocr_lite" else ""
                    ),
                    "column_ocr_transport": (
                        (
                            "ndlocr_full_page_routed_with_isolated_fallback"
                            if "primary" in column_attempts
                            else "ndlocr_full_page_routed"
                        )
                        if "page_primary" in column_attempts
                        else (
                            (
                                "lossless_compact_with_fullsize_safety_fallback"
                                if "primary_fullsize_fallback" in column_attempts
                                else "lossless_compact"
                            )
                            if (compact_primary_transport or smart_crop)
                            else "fullsize_masked_page"
                        )
                    ),
                }
                block.update(sentence_reocr_meta.get(target_key, {}))
                if is_unresolved:
                    block.update({
                        "column_ocr_empty": True,
                        "column_requires_handwriting": True,
                        "preserve_empty_ocr_column": True,
                        "column_manual_placeholder": True,
                        "column_ocr_failure_reason": (
                            "ndlocr_page_only_unreturned_column"
                            if recognition_engine == "ndlocr_lite"
                            and ndlocr_page_mode == "page"
                            and "page_primary" in column_attempts
                            and "primary" not in column_attempts
                            else (
                                "single_primary_pass_returned_empty"
                                if single_pass
                                else (
                                    "adaptive_primary_and_single_rescue_returned_empty"
                                    if column_rescue_policy == "adaptive"
                                    else (
                                        "primary_pass_returned_empty_rescue_disabled"
                                        if column_rescue_policy == "off"
                                        else "all_original_and_preprocess_passes_returned_empty"
                                    )
                                )
                            )
                        ),
                    })
                blocks.append(block)
            preserved_count = len(blocks)
            if strict and not cancelled and preserved_count != expected:
                yield page_path, None, f"列数校验失败：预计 {expected}，已保全 {preserved_count}"
                continue
            if callable(column_preview_callback):
                column_preview_callback(page_path, columns, "recognition")
            if phase_callback is not None:
                phase_callback(
                    "document_build",
                    page_index,
                    len(page_paths),
                    f"{os.path.basename(page_path)} · 预计 {expected} / OCR有字 {text_recognized_count} / "
                    f"列级救援 {rescue_count} / 待人工 {manual_pending_count}",
                )
            if not blocks:
                yield page_path, None, "该页没有可保全的物理文字列"
            else:
                yield page_path, blocks, None

    if recognition_engine != "apple_vision":
        from adapters.ocr_runtime_catalog import mark_runtime_ready
        mark_runtime_ready(recognition_engine)

def run(
    *,
    recognition_engine: str = "manga_ocr",
    column_sensitivity: int = 55,
    column_padding_percent: int = 10,
    strict_column_validation: bool = True,
    column_sentence_context_reocr: bool = False,
    column_sentence_context_strategy: str = "full",
    column_sentence_global_merged_box_reocr: bool = False,
    column_compact_primary_transport: bool = False,
    column_rescue_policy: str = "adaptive",
    column_ndlocr_page_batch: bool = True,
    column_ndlocr_page_mode: str = "hybrid",
    column_sentence_context_max_columns: int = 10,
    max_columns: int = 80,
    shortcut_name: str = "ExtractText",
    verbose: bool = True,
    engine_options: dict | None = None,
    **kwargs,
):
    """Run masked-column isolation through one directly selected OCR engine."""
    if recognition_engine not in SUPPORTED_RECOGNIZERS:
        raise ValueError(f"不支持的精准分列识字引擎: {recognition_engine}")
    phase_callback = kwargs.pop("phase_callback", None)
    runtime_engine_options = dict(engine_options or {})
    runtime_engine_options.setdefault(
        "column_sentence_context_reocr", bool(column_sentence_context_reocr)
    )
    runtime_engine_options.setdefault(
        "column_sentence_context_strategy",
        str(column_sentence_context_strategy or "full"),
    )
    if str(recognition_engine or "").strip().lower() == "manga_ocr":
        runtime_engine_options["column_compact_primary_transport"] = True
    elif str(recognition_engine or "").strip().lower() == "manga_48px":
        runtime_engine_options["column_compact_primary_transport"] = True
    elif str(recognition_engine or "").strip().lower() == "yomitoku":
        runtime_engine_options["column_compact_primary_transport"] = True
    else:
        runtime_engine_options.setdefault(
            "column_compact_primary_transport",
            bool(column_compact_primary_transport),
        )
    runtime_engine_options.setdefault(
        "column_rescue_policy",
        _normalise_column_rescue_policy(column_rescue_policy),
    )
    legacy_mode_was_explicit = "column_ndlocr_page_batch" in runtime_engine_options
    legacy_mode_value = runtime_engine_options.get(
        "column_ndlocr_page_batch", column_ndlocr_page_batch
    )
    runtime_engine_options.setdefault(
        "column_ndlocr_page_batch",
        bool(column_ndlocr_page_batch),
    )
    if "column_ndlocr_page_mode" not in runtime_engine_options:
        runtime_engine_options["column_ndlocr_page_mode"] = _normalise_ndlocr_page_mode(
            None if legacy_mode_was_explicit else column_ndlocr_page_mode,
            legacy_page_batch=legacy_mode_value,
        )
    runtime_engine_options.setdefault(
        "column_sentence_global_merged_box_reocr",
        bool(column_sentence_global_merged_box_reocr),
    )
    runtime_engine_options.setdefault(
        "column_sentence_context_max_columns",
        max(2, int(column_sentence_context_max_columns or 10)),
    )

    def worker_fn(ocr_paths, cancel_check):
        yield from _iter_column_pages(
            list(ocr_paths),
            recognition_engine=recognition_engine,
            sensitivity=column_sensitivity,
            padding_percent=column_padding_percent,
            strict=bool(strict_column_validation),
            shortcut_name=shortcut_name,
            max_columns=max_columns,
            cancel_check=cancel_check,
            verbose=verbose,
            phase_callback=phase_callback,
            engine_options=runtime_engine_options,
        )

    return run_ocr_engine(
        worker_fn,
        source_engine=f"masked_column_ocr:{recognition_engine}",
        verbose=verbose,
        force_text_pages=True,
        strict_column_audit=bool(strict_column_validation),
        **kwargs,
    )
