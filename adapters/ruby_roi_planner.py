#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plan small findtextCenterNet Ruby OCR regions from ordinary OCR geometry.

The ordinary vertical-column detector already knows which tiny side components
look suspicious enough to exclude from main OCR.  This module reuses that
run-local telemetry and never performs recognition itself.

Pipeline boundary:
    normal OCR column detection -> Ruby candidate boxes (geometry only)
        -> merge candidate columns/vertical clusters into context ROIs
        -> findtextCenterNet OCRs only those original-page ROIs.

No ROI pixel is ever fed back into normal OCR comparison/fusion.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image


@dataclass(frozen=True, slots=True)
class RubyCandidateColumn:
    index: int
    left: int
    top: int
    right: int
    bottom: int
    hard_left: int
    hard_right: int
    boxes: tuple[tuple[int, int, int, int], ...]
    confidence: float = 0.0

    @property
    def width(self) -> int:
        return max(1, self.right - self.left)


@dataclass(frozen=True, slots=True)
class RubyROI:
    page_no: int
    page_path: str
    x1: int
    y1: int
    x2: int
    y2: int
    column_indices: tuple[int, ...]
    candidate_boxes: tuple[tuple[int, int, int, int], ...]
    confidence: float

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_dict(self) -> dict:
        return {
            "page_no": self.page_no,
            "page_path": self.page_path,
            "box": [self.x1, self.y1, self.x2, self.y2],
            "column_indices": list(self.column_indices),
            "candidate_boxes": [list(box) for box in self.candidate_boxes],
            "confidence": round(float(self.confidence), 4),
        }


def _rect_union_area(rectangles: Sequence[tuple[int, int, int, int]]) -> int:
    rects = [r for r in rectangles if r[2] > r[0] and r[3] > r[1]]
    if not rects:
        return 0
    xs = sorted({value for r in rects for value in (r[0], r[2])})
    area = 0
    for left, right in zip(xs, xs[1:]):
        if right <= left:
            continue
        intervals = sorted(
            (r[1], r[3]) for r in rects if r[0] < right and r[2] > left
        )
        if not intervals:
            continue
        covered = 0
        start, end = intervals[0]
        for y1, y2 in intervals[1:]:
            if y1 <= end:
                end = max(end, y2)
            else:
                covered += max(0, end - start)
                start, end = y1, y2
        covered += max(0, end - start)
        area += (right - left) * covered
    return int(area)


def estimate_findtext_tiles(
    width: int, height: int, *, input_size: int = 768, step_ratio: float = 0.60,
) -> int:
    """Estimate upstream findtextCenterNet detector window count exactly.

    Upstream pads each dimension so 768px windows can advance by 60% of the
    input size. A small ROI therefore costs one detector window, while a full
    1200x1600 novel page costs six. The ROI planner optimises this number, not
    just raw pixel area.
    """
    size = max(1, int(input_size))
    step = max(1, int(size * float(step_ratio)))

    def axis(value: int) -> int:
        value = max(1, int(value))
        if value <= size:
            return 1
        return ((value - size + step - 1) // step) + 1

    return axis(width) * axis(height)


@dataclass(frozen=True, slots=True)
class RubyROIPlan:
    page_no: int
    page_path: str
    page_width: int
    page_height: int
    rois: tuple[RubyROI, ...]
    candidate_columns: int
    candidate_boxes: int
    source_sidecar: str = ""

    @property
    def roi_area(self) -> int:
        return _rect_union_area([(roi.x1, roi.y1, roi.x2, roi.y2) for roi in self.rois])

    @property
    def coverage_ratio(self) -> float:
        total = max(1, self.page_width * self.page_height)
        return min(1.0, self.roi_area / total)

    @property
    def estimated_detector_tiles(self) -> int:
        return sum(estimate_findtext_tiles(roi.width, roi.height) for roi in self.rois)

    @property
    def full_page_detector_tiles(self) -> int:
        return estimate_findtext_tiles(self.page_width, self.page_height)

    @property
    def estimated_tile_ratio(self) -> float:
        return self.estimated_detector_tiles / max(1, self.full_page_detector_tiles)

    def to_dict(self) -> dict:
        return {
            "page_no": self.page_no,
            "page_path": self.page_path,
            "page_size": [self.page_width, self.page_height],
            "candidate_columns": self.candidate_columns,
            "candidate_boxes": self.candidate_boxes,
            "roi_count": len(self.rois),
            "coverage_ratio": round(self.coverage_ratio, 6),
            "estimated_detector_tiles": self.estimated_detector_tiles,
            "full_page_detector_tiles": self.full_page_detector_tiles,
            "estimated_tile_ratio": round(self.estimated_tile_ratio, 6),
            "source_sidecar": self.source_sidecar,
            "rois": [roi.to_dict() for roi in self.rois],
        }


def _valid_box(raw: Sequence[object]) -> tuple[int, int, int, int] | None:
    if len(raw) < 4:
        return None
    try:
        x1, y1, x2, y2 = (int(round(float(raw[i]))) for i in range(4))
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _vertical_right_side_boxes(
    raw: dict, boxes: Sequence[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Keep only geometry that can actually sit on the Ruby side of a vertical column.

    The high-recall component probe intentionally records a few punctuation /
    detached-stroke false positives inside the body band.  Those are useful for
    normal Ruby filtering, but scheduling findtextCenterNet for them would turn
    most pages back into near-full-page work.  Japanese vertical Ruby is printed
    on the *right* of its base column, so the ROI scheduler requires the candidate
    centre to cross the ordinary OCR body envelope.

    This is deliberately a planner-only gate: raw telemetry stays untouched,
    normal OCR is unchanged, and the user can still select ``full_page`` for
    unusual typography.  A 1px tolerance handles anti-aliasing / rounded crops.
    """
    try:
        body_right = float(raw.get("right", 0) or 0)
        body_left = float(raw.get("left", 0) or 0)
    except (TypeError, ValueError):
        return list(boxes)
    if body_right <= body_left:
        return list(boxes)
    tolerance = max(1.0, (body_right - body_left) * 0.025)
    preferred = [
        box for box in boxes
        if ((float(box[0]) + float(box[2])) * 0.5) >= body_right - tolerance
    ]
    return preferred


def _candidate_columns(payload: dict) -> list[RubyCandidateColumn]:
    out: list[RubyCandidateColumn] = []
    for index, raw in enumerate(payload.get("columns", []) or []):
        if not isinstance(raw, dict):
            continue
        boxes: list[tuple[int, int, int, int]] = []
        for item in raw.get("ruby_candidate_boxes", []) or []:
            if isinstance(item, (list, tuple)):
                box = _valid_box(item)
                if box is not None and box not in boxes:
                    boxes.append(box)
        # Backward-compatible fallback only for *old* sidecars that did not
        # have an explicit Ruby-candidate field.  Modern columns deliberately
        # store an empty list when no candidate exists; treating all excluded
        # debris as Ruby would recreate full-page work through false positives.
        if not boxes and "ruby_candidate_boxes" not in raw:
            for item in raw.get("excluded_boxes", []) or []:
                if isinstance(item, (list, tuple)):
                    box = _valid_box(item)
                    if box is not None and box not in boxes:
                        boxes.append(box)
        if not boxes:
            continue

        # ``ruby_candidate_boxes`` is produced by the vertical-column detector.
        # Reduce scheduler false positives without deleting the raw evidence.
        # All OCR engines still receive the same Ruby-free body crop; only the
        # independent findtext ROI list is narrowed here.
        side_boxes = _vertical_right_side_boxes(raw, boxes)
        if not side_boxes:
            continue
        try:
            out.append(RubyCandidateColumn(
                index=index,
                left=int(raw.get("left", 0)), top=int(raw.get("top", 0)),
                right=int(raw.get("right", 0)), bottom=int(raw.get("bottom", 0)),
                hard_left=int(raw.get("hard_left", raw.get("left", 0))),
                hard_right=int(raw.get("hard_right", raw.get("right", 0))),
                boxes=tuple(side_boxes),
                confidence=float(raw.get("ruby_candidate_confidence", 0.65) or 0.65),
            ))
        except (TypeError, ValueError):
            continue
    return out


def _sidecar_score(payload: dict) -> tuple[int, int, float]:
    columns = _candidate_columns(payload)
    return (
        len(columns),
        sum(len(column.boxes) for column in columns),
        max((column.confidence for column in columns), default=0.0),
    )


def load_column_sidecars(root: str | Path | None) -> dict[str, tuple[dict, Path]]:
    """Return best run-local column sidecar per original page path."""
    if not root:
        return {}
    base = Path(root)
    if not base.exists():
        return {}
    chosen: dict[str, tuple[dict, Path]] = {}
    for sidecar in base.rglob("p*_columns.json"):
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            continue
        page_path = str(payload.get("page_path") or "")
        if not page_path:
            continue
        previous = chosen.get(page_path)
        if previous is None or _sidecar_score(payload) > _sidecar_score(previous[0]):
            chosen[page_path] = (payload, sidecar)
    return chosen


def _group_vertical_boxes(
    boxes: Iterable[tuple[int, int, int, int]], *, body_width: int,
) -> list[list[tuple[int, int, int, int]]]:
    ordered = sorted(set(boxes), key=lambda box: (box[1], box[0], box[3], box[2]))
    if not ordered:
        return []
    heights = [max(1, box[3] - box[1]) for box in ordered]
    median_height = statistics.median(heights) if heights else max(1, body_width // 2)
    gap_limit = max(18, int(round(body_width * 1.35)), int(round(median_height * 2.4)))
    groups: list[list[tuple[int, int, int, int]]] = [[ordered[0]]]
    for box in ordered[1:]:
        previous_bottom = max(item[3] for item in groups[-1])
        if box[1] - previous_bottom <= gap_limit:
            groups[-1].append(box)
        else:
            groups.append([box])
    return groups


def _fit_axis_to_single_tile(
    start: int, end: int, *, required_start: int, required_end: int,
    page_limit: int, tile_size: int = 768, min_context: int = 72,
    max_trim: int = 96,
) -> tuple[int, int]:
    """Conservatively trim a near-tile interval to one detector window.

    findtextCenterNet pads/crops at 768px.  A 769--864px context therefore
    costs two detector windows even when all Ruby evidence occupies a much
    smaller span.  Only trim when the original interval exceeds one tile by a
    small amount *and* every candidate keeps at least ``min_context`` pixels of
    surrounding page context.  This never changes candidate geometry itself.
    """
    start, end = int(start), int(end)
    required_start, required_end = int(required_start), int(required_end)
    page_limit = max(1, int(page_limit))
    tile_size = max(1, int(tile_size))
    if end <= start or end - start <= tile_size:
        return max(0, start), min(page_limit, end)
    if (end - start) - tile_size > max(0, int(max_trim)):
        return max(0, start), min(page_limit, end)
    if required_end <= required_start:
        return max(0, start), min(page_limit, end)
    context = max(0, int(min_context))
    needed_start = max(0, required_start - context)
    needed_end = min(page_limit, required_end + context)
    if needed_end - needed_start > tile_size:
        return max(0, start), min(page_limit, end)

    # Choose a tile-sized interval containing the required-context envelope and
    # staying as close as possible to the original ROI centre.
    preferred = int(round(((start + end) / 2.0) - tile_size / 2.0))
    lower = max(0, needed_end - tile_size)
    upper = min(needed_start, max(0, page_limit - tile_size))
    if lower > upper:
        return max(0, start), min(page_limit, end)
    new_start = min(max(preferred, lower), upper)
    new_end = min(page_limit, new_start + tile_size)
    new_start = max(0, new_end - tile_size)
    if new_start > needed_start or new_end < needed_end:
        return max(0, start), min(page_limit, end)
    return int(new_start), int(new_end)


def _tighten_near_tile_roi(
    roi: RubyROI, *, page_width: int, page_height: int, tile_size: int = 768,
) -> RubyROI:
    """Reduce accidental 2-window ROIs caused by a tiny amount of padding.

    The candidate envelope is the hard constraint.  We only trim up to 96 px
    and retain >=72 px context around the outermost Ruby hint, so the base text
    and line structure remain visible to findtextCenterNet.
    """
    boxes = [box for box in roi.candidate_boxes if box[2] > box[0] and box[3] > box[1]]
    if not boxes:
        return roi
    req_x1 = min(box[0] for box in boxes)
    req_y1 = min(box[1] for box in boxes)
    req_x2 = max(box[2] for box in boxes)
    req_y2 = max(box[3] for box in boxes)
    x1, x2 = _fit_axis_to_single_tile(
        roi.x1, roi.x2, required_start=req_x1, required_end=req_x2,
        page_limit=page_width, tile_size=tile_size, min_context=72, max_trim=96,
    )
    y1, y2 = _fit_axis_to_single_tile(
        roi.y1, roi.y2, required_start=req_y1, required_end=req_y2,
        page_limit=page_height, tile_size=tile_size, min_context=72, max_trim=96,
    )
    if (x1, y1, x2, y2) == (roi.x1, roi.y1, roi.x2, roi.y2):
        return roi
    return RubyROI(
        page_no=roi.page_no, page_path=roi.page_path,
        x1=x1, y1=y1, x2=x2, y2=y2,
        column_indices=roi.column_indices, candidate_boxes=roi.candidate_boxes,
        confidence=roi.confidence,
    )


def _merge_rois(
    rois: list[RubyROI], *, max_columns: int = 10, max_width: int = 720,
) -> list[RubyROI]:
    """Tile-aware packing of nearby Ruby contexts.

    Several tiny Ruby candidates should not become several 768x768 model calls.
    We greedily merge neighbouring original-page rectangles whenever the merged
    rectangle costs fewer (or no more) findtext detector windows. This keeps
    natural page pixels/column ordering while turning single-column hints into
    a small number of larger context frames.
    """
    if not rois:
        return []
    pending = sorted(rois, key=lambda roi: (roi.page_no, roi.x1, roi.y1, roi.x2, roi.y2))

    while True:
        best: tuple[tuple[float, ...], int, int, RubyROI] | None = None
        for i, left in enumerate(pending):
            for j in range(i + 1, len(pending)):
                right = pending[j]
                if right.page_path != left.page_path:
                    continue
                columns = tuple(sorted(set(left.column_indices) | set(right.column_indices)))
                if len(columns) > max(1, int(max_columns)):
                    continue
                x1, y1 = min(left.x1, right.x1), min(left.y1, right.y1)
                x2, y2 = max(left.x2, right.x2), max(left.y2, right.y2)
                width, height = x2 - x1, y2 - y1
                if width > max(128, int(max_width)):
                    continue

                x_gap = max(0, max(left.x1, right.x1) - min(left.x2, right.x2))
                if x_gap > 96:
                    continue

                separate_tiles = (
                    estimate_findtext_tiles(left.width, left.height)
                    + estimate_findtext_tiles(right.width, right.height)
                )
                merged_tiles = estimate_findtext_tiles(width, height)
                if merged_tiles > separate_tiles:
                    continue

                union_area = _rect_union_area([
                    (left.x1, left.y1, left.x2, left.y2),
                    (right.x1, right.y1, right.x2, right.y2),
                ])
                inflation = (width * height) / max(1, union_area)
                savings = separate_tiles - merged_tiles
                overlap_y = max(0, min(left.y2, right.y2) - max(left.y1, right.y1))
                overlap_x = max(0, min(left.x2, right.x2) - max(left.x1, right.x1))
                if savings <= 0 and (overlap_x <= 0 or overlap_y <= 0 or inflation > 1.65):
                    continue
                if savings > 0 and inflation > 3.25:
                    continue

                boxes = tuple(dict.fromkeys((*left.candidate_boxes, *right.candidate_boxes)))
                merged = RubyROI(
                    page_no=left.page_no, page_path=left.page_path,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    column_indices=columns, candidate_boxes=boxes,
                    confidence=max(left.confidence, right.confidence),
                )
                rank = (
                    float(savings),
                    -float(inflation),
                    float(len(boxes)),
                    -float(width * height),
                )
                if best is None or rank > best[0]:
                    best = (rank, i, j, merged)
        if best is None:
            break
        _rank, i, j, merged = best
        pending = [roi for k, roi in enumerate(pending) if k not in {i, j}] + [merged]
        pending.sort(key=lambda roi: (roi.page_no, roi.x1, roi.y1, roi.x2, roi.y2))

    return sorted(pending, key=lambda roi: (roi.page_no, roi.y1, -roi.x2))


def plan_ruby_rois_from_payload(
    *,
    page_no: int,
    page_path: str,
    payload: dict,
    page_width: int,
    page_height: int,
    neighbor_columns: int = 1,
    max_columns_per_roi: int = 10,
) -> RubyROIPlan:
    raw_columns = payload.get("columns", []) or []
    candidate_columns = _candidate_columns(payload)
    rois: list[RubyROI] = []
    for candidate in candidate_columns:
        groups = _group_vertical_boxes(candidate.boxes, body_width=candidate.width)
        for group in groups:
            start = max(0, candidate.index - max(0, int(neighbor_columns)))
            end = min(len(raw_columns), candidate.index + max(0, int(neighbor_columns)) + 1)
            if end - start > max_columns_per_roi:
                extra = (end - start) - max_columns_per_roi
                start += extra // 2
                end -= extra - extra // 2
            selected = raw_columns[start:end]
            if not selected:
                continue
            selected_lefts = [int(item.get("left", 0)) for item in selected]
            selected_rights = [int(item.get("right", page_width)) for item in selected]
            gy1 = min(box[1] for box in group)
            gy2 = max(box[3] for box in group)
            gx1 = min(box[0] for box in group)
            gx2 = max(box[2] for box in group)
            x1 = min([gx1, *selected_lefts])
            x2 = max([gx2, *selected_rights])
            context_y = max(72, int(round(candidate.width * 3.0)), int((gy2 - gy1) * 1.5))
            y1 = max(0, gy1 - context_y)
            y2 = min(page_height, gy2 + context_y)
            x_context = max(10, int(round(candidate.width * 0.45)))
            x1 = max(0, x1 - x_context)
            x2 = min(page_width, x2 + x_context)
            if x2 - x1 < 24 or y2 - y1 < 40:
                continue
            rois.append(RubyROI(
                page_no=page_no, page_path=page_path,
                x1=x1, y1=y1, x2=x2, y2=y2,
                column_indices=tuple(range(start, end)),
                candidate_boxes=tuple(group),
                confidence=float(candidate.confidence),
            ))
    merged = _merge_rois(
        rois,
        max_columns=max(1, int(max_columns_per_roi)),
        max_width=min(720, max(320, int(page_width * 0.72))),
    )
    # A few pixels of padding above 768 can double the upstream detector cost.
    # Tighten only near-boundary ROIs where every candidate still retains ample
    # context.  This is a scheduler optimisation; raw geometry is untouched.
    merged = [
        _tighten_near_tile_roi(roi, page_width=page_width, page_height=page_height)
        for roi in merged
    ]
    return RubyROIPlan(
        page_no=page_no,
        page_path=page_path,
        page_width=page_width,
        page_height=page_height,
        rois=tuple(merged),
        candidate_columns=len(candidate_columns),
        candidate_boxes=sum(len(column.boxes) for column in candidate_columns),
    )


def build_ruby_roi_plans(
    page_images: Sequence[tuple[int, str]],
    *,
    candidate_root: str | Path | None = None,
    candidate_payloads: dict | None = None,
    neighbor_columns: int = 1,
    max_columns_per_roi: int = 10,
) -> list[RubyROIPlan]:
    """Build ROI plans from in-memory OCR geometry, falling back to sidecars.

    ``candidate_payloads`` is the preferred production path because it is keyed
    by authoritative document page number and survives temporary OCR crop paths.
    Sidecars remain useful for diagnostics, older runs and standalone tests.
    """
    sidecars = load_column_sidecars(candidate_root)
    supplied = candidate_payloads if isinstance(candidate_payloads, dict) else {}
    plans: list[RubyROIPlan] = []

    def supplied_payload(page_no: int, page_path: str) -> dict | None:
        for key in (str(page_no), page_no, str(page_path)):
            value = supplied.get(key)
            if isinstance(value, dict):
                return value
        return None

    for page_no, page_path in page_images:
        payload = supplied_payload(page_no, page_path)
        sidecar_text = "document_metadata" if payload is not None else ""
        if payload is None:
            entry = sidecars.get(str(page_path))
            if entry is None:
                try:
                    resolved = str(Path(page_path).resolve())
                except OSError:
                    resolved = ""
                if resolved:
                    for raw_path, candidate in sidecars.items():
                        try:
                            if str(Path(raw_path).resolve()) == resolved:
                                entry = candidate
                                break
                        except OSError:
                            continue
            if entry is not None:
                payload, sidecar = entry
                sidecar_text = str(sidecar)
        # Missing geometry is represented by an empty plan rather than silently
        # scheduling a full-page OCR.  The caller may explicitly choose full_page
        # mode when that conservative slow fallback is wanted.
        if payload is None:
            payload = {"columns": []}
            sidecar_text = ""
        try:
            with Image.open(page_path) as image:
                width, height = image.size
        except Exception:
            continue
        plan = plan_ruby_rois_from_payload(
            page_no=page_no,
            page_path=page_path,
            payload=payload,
            page_width=width,
            page_height=height,
            neighbor_columns=neighbor_columns,
            max_columns_per_roi=max_columns_per_roi,
        )
        plans.append(RubyROIPlan(
            page_no=plan.page_no,
            page_path=plan.page_path,
            page_width=plan.page_width,
            page_height=plan.page_height,
            rois=plan.rois,
            candidate_columns=plan.candidate_columns,
            candidate_boxes=plan.candidate_boxes,
            source_sidecar=sidecar_text,
        ))
    return plans


def save_roi_plan_report(plans: Sequence[RubyROIPlan], target: str | Path) -> Path:
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "findtext-ruby-roi-plan-v2-tile-aware",
        "pages": [plan.to_dict() for plan in plans],
        "summary": {
            "pages_with_candidates": sum(1 for plan in plans if plan.rois),
            "roi_count": sum(len(plan.rois) for plan in plans),
            "candidate_columns": sum(plan.candidate_columns for plan in plans),
            "candidate_boxes": sum(plan.candidate_boxes for plan in plans),
            "roi_area": sum(plan.roi_area for plan in plans),
            "page_area": sum(plan.page_width * plan.page_height for plan in plans),
            "estimated_detector_tiles": sum(plan.estimated_detector_tiles for plan in plans),
            "full_page_detector_tiles": sum(plan.full_page_detector_tiles for plan in plans),
        },
    }
    page_area = max(1, int(payload["summary"]["page_area"]))
    payload["summary"]["coverage_ratio"] = round(
        int(payload["summary"]["roi_area"]) / page_area, 6
    )
    payload["summary"]["estimated_tile_ratio"] = round(
        int(payload["summary"]["estimated_detector_tiles"])
        / max(1, int(payload["summary"]["full_page_detector_tiles"])),
        6,
    )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
