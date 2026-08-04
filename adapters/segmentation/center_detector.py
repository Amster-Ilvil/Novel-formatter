#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight character-centre fallback for vertical printed text.

This is not a neural detector.  It borrows the useful CenterNet abstraction
(character centre + size) but derives candidates from connected components and
row projections, so it adds no model dependency.  It is used only when the
projection-valley partition is missing or clearly under-segmented.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Sequence


@dataclass(frozen=True, slots=True)
class CenterIntervalResult:
    intervals: tuple[tuple[int, int], ...]
    centers: tuple[float, ...]
    confidence: float
    estimated_height: float
    source_count: int


def _smooth(values: Sequence[int], radius: int) -> list[float]:
    if radius <= 0:
        return [float(value) for value in values]
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + float(value))
    out: list[float] = []
    for index in range(len(values)):
        left = max(0, index - radius)
        right = min(len(values), index + radius + 1)
        out.append((prefix[right] - prefix[left]) / max(1, right - left))
    return out


def _component_height(item: dict) -> float:
    return max(1.0, float(item.get("y1", 0)) - float(item.get("y0", 0)))


def _component_width(item: dict) -> float:
    return max(1.0, float(item.get("x1", 0)) - float(item.get("x0", 0)))


def detect_center_intervals(
    components: Sequence[dict],
    row_ink: Sequence[int],
    *,
    first_ink: int,
    last_ink: int,
    target_height: float,
    body_width: float,
) -> CenterIntervalResult:
    """Infer locally drifting intervals from component/ink centres.

    Detached tiny marks are attached to the nearest substantial centre rather
    than becoming centres themselves.  Boundaries are then moved to the least-
    ink row around each midpoint.
    """
    target = max(6.0, float(target_height))
    substantial = [
        item for item in components
        if _component_height(item) >= target * 0.24
        or _component_width(item) >= max(2.0, float(body_width) * 0.30)
        or int(item.get("area", 0)) >= max(4, int(round(target * body_width * 0.014)))
    ]
    if not substantial:
        return CenterIntervalResult(tuple(), tuple(), 0.0, target, 0)

    ordered = sorted(substantial, key=lambda item: float(item.get("cy", 0.0)))
    clusters: list[list[dict]] = []
    for item in ordered:
        cy = float(item.get("cy", 0.0))
        if not clusters:
            clusters.append([item])
            continue
        previous_centres = [float(part.get("cy", 0.0)) for part in clusters[-1]]
        previous_centre = sum(previous_centres) / len(previous_centres)
        previous_y1 = max(float(part.get("y1", 0.0)) for part in clusters[-1])
        gap = float(item.get("y0", 0.0)) - previous_y1
        # Components inside one printed glyph may be detached.  Merge only when
        # both centre distance and geometric gap are small relative to H.
        if abs(cy - previous_centre) <= target * 0.42 or gap <= target * 0.12:
            clusters[-1].append(item)
        else:
            clusters.append([item])

    raw_centers = [
        sum(float(item.get("cy", 0.0)) * max(1, int(item.get("area", 0))) for item in cluster)
        / max(1, sum(max(1, int(item.get("area", 0))) for item in cluster))
        for cluster in clusters
    ]
    # Split suspiciously distant centres by inserting local projection peaks.
    smooth = _smooth(row_ink, max(1, int(round(target * 0.025))))
    centers: list[float] = []
    for index, centre in enumerate(raw_centers):
        if centers:
            gap = centre - centers[-1]
            missing = max(0, int(round(gap / target)) - 1)
            if missing > 0 and gap >= target * 1.62:
                for step in range(1, missing + 1):
                    ideal = centers[-1] + gap * step / (missing + 1)
                    radius = max(2, int(round(target * 0.25)))
                    lo = max(first_ink, int(round(ideal)) - radius)
                    hi = min(last_ink - 1, int(round(ideal)) + radius)
                    if hi >= lo:
                        peak = max(range(lo, hi + 1), key=lambda y: (smooth[y], -abs(y - ideal)))
                        centers.append(float(peak))
        centers.append(float(centre))

    # Remove duplicate centres that are too close; the heavier local ink wins.
    deduped: list[float] = []
    for centre in sorted(centers):
        if deduped and centre - deduped[-1] < target * 0.43:
            old = deduped[-1]
            old_ink = smooth[max(0, min(len(smooth) - 1, int(round(old))))]
            new_ink = smooth[max(0, min(len(smooth) - 1, int(round(centre))))]
            if new_ink > old_ink:
                deduped[-1] = centre
        else:
            deduped.append(centre)
    centers = deduped
    if not centers:
        return CenterIntervalResult(tuple(), tuple(), 0.0, target, len(substantial))

    boundaries = [max(0, int(first_ink))]
    for left, right in zip(centers, centers[1:]):
        midpoint = (left + right) / 2.0
        radius = max(2, int(round(min(target * 0.28, (right - left) * 0.32))))
        lo = max(boundaries[-1] + 2, int(round(midpoint)) - radius)
        hi = min(int(last_ink) - 2, int(round(midpoint)) + radius)
        if hi < lo:
            cut = max(boundaries[-1] + 2, min(int(last_ink) - 2, int(round(midpoint))))
        else:
            cut = min(
                range(lo, hi + 1),
                key=lambda y: (
                    smooth[y] + abs(y - midpoint) * max(0.05, max(smooth[lo:hi + 1], default=1.0) * 0.025),
                    abs(y - midpoint),
                ),
            )
        boundaries.append(int(cut))
    boundaries.append(min(len(row_ink), max(boundaries[-1] + 2, int(last_ink))))
    intervals = tuple((a, b) for a, b in zip(boundaries, boundaries[1:]) if b - a >= 3)

    distances = [right - left for left, right in zip(centers, centers[1:])]
    if distances:
        med = float(median(distances))
        deviation = sum(abs(value - med) for value in distances) / max(1.0, len(distances) * target)
        regularity = max(0.0, 1.0 - deviation)
    else:
        regularity = 0.55
    occupied = sum(1 for a, b in intervals if sum(row_ink[a:b]) > 0)
    coverage = min(1.0, occupied / max(1, len(centers)))
    confidence = max(0.0, min(1.0, regularity * 0.62 + coverage * 0.38))
    return CenterIntervalResult(
        intervals=intervals,
        centers=tuple(centers),
        confidence=round(confidence, 4),
        estimated_height=target,
        source_count=len(substantial),
    )
