#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Geometry-only furigana/ruby filtering for a pre-detected vertical column.

The filter never attempts OCR.  It distinguishes the dominant body-text x band
from repeated small side components.  Small punctuation inside the body band is
preserved; only side clusters that are both size-consistent and vertically
repeated are classified as ruby.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Sequence


@dataclass(frozen=True, slots=True)
class RubyFilterResult:
    body_x0: int
    body_x1: int
    kept_components: tuple[dict, ...]
    ruby_components: tuple[dict, ...]
    ruby_boxes: tuple[tuple[int, int, int, int], ...]
    body_component_count: int
    ruby_component_count: int
    confidence: float


def _width(component: dict) -> int:
    return max(1, int(component.get("x1", 0)) - int(component.get("x0", 0)))


def _height(component: dict) -> int:
    return max(1, int(component.get("y1", 0)) - int(component.get("y0", 0)))


def _area(component: dict) -> int:
    return max(0, int(component.get("area", 0)))


def _vertical_clusters(items: list[dict], target_height: float) -> list[list[dict]]:
    if not items:
        return []
    ordered = sorted(items, key=lambda item: (float(item.get("cy", 0.0)), float(item.get("cx", 0.0))))
    clusters: list[list[dict]] = [[ordered[0]]]
    max_gap = max(3.0, float(target_height) * 1.30)
    for item in ordered[1:]:
        previous = clusters[-1][-1]
        gap = float(item.get("y0", 0)) - float(previous.get("y1", 0))
        if gap <= max_gap:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    return clusters


def classify_vertical_ruby(
    components: Sequence[dict],
    *,
    body_x0: int,
    body_x1: int,
    image_width: int,
    target_height: float,
) -> RubyFilterResult:
    """Classify repeated side annotations without removing body punctuation.

    Parameters are pixel coordinates in one already isolated physical column.
    ``body_x0/body_x1`` is the first-pass body band.  The result may tighten that
    band slightly and returns component lists suitable for the projection stage.
    """
    items = [dict(component) for component in components]
    if not items or body_x1 <= body_x0:
        return RubyFilterResult(
            max(0, int(body_x0)), min(int(image_width), int(body_x1)),
            tuple(items), tuple(), tuple(), len(items), 0, 0.0,
        )

    band_width = max(1.0, float(body_x1 - body_x0))
    inner_margin = max(1.0, band_width * 0.12)
    inner_x0 = float(body_x0) + inner_margin
    inner_x1 = float(body_x1) - inner_margin

    overlapping = [
        item for item in items
        if float(item.get("x1", 0)) > float(body_x0)
        and float(item.get("x0", 0)) < float(body_x1)
    ]
    substantial = [
        item for item in overlapping
        if _width(item) >= band_width * 0.34
        or _height(item) >= max(4.0, float(target_height) * 0.46)
        or _area(item) >= band_width * max(6.0, float(target_height)) * 0.018
    ]
    body_seed = substantial or overlapping or items
    median_width = float(median([_width(item) for item in body_seed])) if body_seed else band_width
    median_height = float(median([_height(item) for item in body_seed])) if body_seed else float(target_height)
    median_area = float(median([max(1, _area(item)) for item in body_seed])) if body_seed else 1.0

    small_limit_w = max(2.0, min(band_width * 0.58, median_width * 0.68))
    small_limit_h = max(3.0, min(float(target_height) * 0.62, median_height * 0.76))
    small_limit_area = max(4.0, median_area * 0.42)
    side_margin = max(1.0, band_width * 0.06)

    side_candidates: dict[str, list[dict]] = {"left": [], "right": []}
    for item in items:
        cx = float(item.get("cx", (float(item.get("x0", 0)) + float(item.get("x1", 0))) / 2.0))
        is_small = (
            _width(item) <= small_limit_w
            and (_height(item) <= small_limit_h or _area(item) <= small_limit_area)
        )
        if not is_small:
            continue
        # A tiny mark whose centre is inside the dominant body band may be a
        # legitimate comma, period, dakuten or detached stroke. Never classify
        # such a mark as ruby solely from its size.
        if inner_x0 <= cx <= inner_x1:
            continue
        if cx < float(body_x0) + side_margin:
            side_candidates["left"].append(item)
        elif cx > float(body_x1) - side_margin:
            side_candidates["right"].append(item)

    ruby_ids: set[int] = set()
    ruby_items: list[dict] = []
    cluster_scores: list[float] = []
    for side, candidates in side_candidates.items():
        for cluster in _vertical_clusters(candidates, target_height):
            if len(cluster) < 2:
                continue
            centres = [float(item.get("cy", 0.0)) for item in cluster]
            span = max(1.0, max(centres) - min(centres))
            density = min(1.0, len(cluster) * max(1.0, float(target_height) * 0.55) / span)
            widths = [_width(item) for item in cluster]
            consistency = 1.0 - min(1.0, (max(widths) - min(widths)) / max(1.0, median_width))
            score = 0.55 * density + 0.45 * consistency
            if score < 0.40:
                continue
            for item in cluster:
                ruby_ids.add(id(item))
                ruby_items.append(item)
            cluster_scores.append(score)

    # The copied dict objects have unique identities inside ``items``.
    kept = [item for item in items if id(item) not in ruby_ids]
    if not kept:
        kept = items
        ruby_items = []

    body_candidates = [
        item for item in kept
        if float(item.get("x1", 0)) > float(body_x0)
        and float(item.get("x0", 0)) < float(body_x1)
        and (
            _width(item) >= max(2.0, median_width * 0.48)
            or _height(item) >= max(3.0, float(target_height) * 0.38)
            or _area(item) >= max(4.0, median_area * 0.28)
        )
    ]
    if body_candidates:
        refined_x0 = min(int(item.get("x0", body_x0)) for item in body_candidates)
        refined_x1 = max(int(item.get("x1", body_x1)) for item in body_candidates)
        padding = max(1, int(round((refined_x1 - refined_x0) * 0.08)))
        refined_x0 = max(0, min(int(body_x0), refined_x0 - padding))
        refined_x1 = min(int(image_width), max(int(body_x1), refined_x1 + padding))
    else:
        refined_x0, refined_x1 = int(body_x0), int(body_x1)

    boxes = tuple(
        (int(item.get("x0", 0)), int(item.get("y0", 0)), int(item.get("x1", 0)), int(item.get("y1", 0)))
        for item in ruby_items
    )
    confidence = sum(cluster_scores) / len(cluster_scores) if cluster_scores else 0.0
    return RubyFilterResult(
        body_x0=refined_x0,
        body_x1=refined_x1,
        kept_components=tuple(kept),
        ruby_components=tuple(ruby_items),
        ruby_boxes=boxes,
        body_component_count=len(kept),
        ruby_component_count=len(ruby_items),
        confidence=round(float(confidence), 4),
    )


def detect_vertical_ruby_candidates(
    components: Sequence[dict],
    *,
    body_x0: int,
    body_x1: int,
    image_width: int,
    target_width: float,
    target_height: float,
) -> RubyFilterResult:
    """High-recall geometry telemetry for scheduling specialist Ruby OCR.

    Unlike :func:`classify_vertical_ruby`, this helper is *not* allowed to
    remove pixels from the normal OCR input.  It may therefore be deliberately
    more permissive: repeated half-size glyphs just outside the body band are
    recorded as candidate ROIs and later verified by findtextCenterNet.
    False positives cost a small ROI inference; false negatives would lose Ruby.
    """
    items = [dict(component) for component in components]
    if not items or body_x1 <= body_x0:
        return RubyFilterResult(
            max(0, int(body_x0)), min(int(image_width), int(body_x1)),
            tuple(items), tuple(), tuple(), len(items), 0, 0.0,
        )

    body_width = max(1.0, float(body_x1 - body_x0))
    glyph_w = max(body_width, float(target_width or body_width))
    glyph_h = max(4.0, float(target_height or glyph_w))
    side_reach = max(6.0, glyph_w * 0.95)
    max_w = max(4.0, glyph_w * 0.72)
    max_h = max(5.0, glyph_h * 0.78)
    max_area = max(10.0, glyph_w * glyph_h * 0.46)

    side_candidates: dict[str, list[dict]] = {"left": [], "right": []}
    for item in items:
        cx = float(item.get("cx", 0.0))
        w = _width(item); h = _height(item); area = max(1, _area(item))
        if w > max_w or h > max_h or area > max_area:
            continue
        # Require the component centre to be outside the body core.  A small
        # overlap is tolerated because anti-aliasing and connected radicals can
        # cross the estimated body edge by a few pixels.
        if float(body_x1) - 1.0 < cx <= float(body_x1) + side_reach:
            side_candidates["right"].append(item)
        elif float(body_x0) - side_reach <= cx < float(body_x0) + 1.0:
            side_candidates["left"].append(item)

    ruby_items: list[dict] = []
    scores: list[float] = []
    seen_boxes: set[tuple[int, int, int, int]] = set()
    for candidates in side_candidates.values():
        if not candidates:
            continue
        # Split candidates into rough x lanes first; punctuation at unrelated x
        # positions should not combine into a fake vertical reading.
        ordered_x = sorted(candidates, key=lambda item: float(item.get("cx", 0.0)))
        lanes: list[list[dict]] = []
        lane_gap = max(4.0, glyph_w * 0.42)
        for item in ordered_x:
            cx = float(item.get("cx", 0.0))
            if not lanes:
                lanes.append([item]); continue
            lane_cx = median([float(value.get("cx", 0.0)) for value in lanes[-1]])
            if abs(cx - lane_cx) <= lane_gap:
                lanes[-1].append(item)
            else:
                lanes.append([item])
        for lane in lanes:
            for cluster in _vertical_clusters(lane, glyph_h):
                if len(cluster) < 2:
                    continue
                centres = [float(item.get("cy", 0.0)) for item in cluster]
                span = max(1.0, max(centres) - min(centres))
                density = min(1.0, len(cluster) * glyph_h * 0.46 / span)
                widths = [_width(item) for item in cluster]
                consistency = 1.0 - min(1.0, (max(widths)-min(widths))/max(1.0,max_w))
                score = 0.62 * density + 0.38 * consistency
                if score < 0.34:
                    continue
                for item in cluster:
                    box=(int(item.get("x0",0)),int(item.get("y0",0)),int(item.get("x1",0)),int(item.get("y1",0)))
                    if box in seen_boxes:
                        continue
                    seen_boxes.add(box); ruby_items.append(item)
                scores.append(score)

    boxes = tuple(
        (int(item.get("x0",0)), int(item.get("y0",0)), int(item.get("x1",0)), int(item.get("y1",0)))
        for item in ruby_items
    )
    confidence = sum(scores)/len(scores) if scores else 0.0
    return RubyFilterResult(
        body_x0=int(body_x0), body_x1=int(body_x1),
        kept_components=tuple(items), ruby_components=tuple(ruby_items),
        ruby_boxes=boxes, body_component_count=len(items),
        ruby_component_count=len(ruby_items), confidence=round(float(confidence),4),
    )
