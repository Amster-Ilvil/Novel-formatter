from __future__ import annotations

"""Model-agnostic horizontal reflow for vertical-column retry OCR.

The input is already an isolated, Ruby-free physical column.  One fixed grid is
estimated for the whole column; empty cells (indent / paragraph spacing) are
ignored, while punctuation and small kana remain inside their original cells.
No OCR text is used to place glyphs.  The strict-majority text contributes only
an expected *count* used as a fail-closed geometry check.
"""

from dataclasses import dataclass
from math import ceil, floor, sqrt
from typing import Sequence

from PIL import Image


@dataclass(frozen=True, slots=True)
class GridCell:
    left: int
    top: int
    right: int
    bottom: int
    ink_pixels: int


@dataclass(frozen=True, slots=True)
class ColumnGrid:
    cells: tuple[GridCell, ...]
    pitch: int
    phase: int
    boundary_ink_ratio: float
    peak_boundary_ink_ratio: float
    autocorrelation: float
    safe: bool
    reason: str = ""


@dataclass(slots=True)
class HorizontalReflowResult:
    image: Image.Image | None
    grids: tuple[ColumnGrid, ...]
    glyph_count: int
    expected_glyph_count: int
    safe: bool
    reason: str = ""
    window_start: int = 0
    window_end: int = 0


def _ink_geometry(image: Image.Image, threshold: int = 220):
    # Keep the production path Pillow-only, but do the projection in Pillow's C
    # loops rather than iterating every page pixel in Python.
    gray = image.convert("L")
    try:
        mask = gray.point(lambda value: 255 if value < threshold else 0, mode="L")
    finally:
        gray.close()
    try:
        width, height = mask.size
        bbox = mask.getbbox()
        if bbox is None:
            return [0] * height, [0] * width, 0, None
        row_avg = mask.resize((1, height), Image.Resampling.BOX)
        col_avg = mask.resize((width, 1), Image.Resampling.BOX)
        try:
            row_values = (
                row_avg.get_flattened_data()
                if hasattr(row_avg, "get_flattened_data") else row_avg.getdata()
            )
            col_values = (
                col_avg.get_flattened_data()
                if hasattr(col_avg, "get_flattened_data") else col_avg.getdata()
            )
            rows = [round(int(value) * width / 255.0) for value in row_values]
            cols = [round(int(value) * height / 255.0) for value in col_values]
        finally:
            row_avg.close()
            col_avg.close()
        total = int(sum(rows))
        return rows, cols, total, tuple(int(value) for value in bbox)
    finally:
        mask.close()


def _autocorrelation(rows: Sequence[int], lag: int, start: int, end: int) -> float:
    if lag <= 0 or end - start <= lag + 4:
        return -1.0
    a = [float(v) for v in rows[start:end - lag]]
    b = [float(v) for v in rows[start + lag:end]]
    if len(a) < 5:
        return -1.0
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    numerator = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sqrt(sum((x - ma) ** 2 for x in a))
    db = sqrt(sum((y - mb) ** 2 for y in b))
    return numerator / (da * db) if da > 1e-9 and db > 1e-9 else -1.0


def _candidate_pitches(rows: Sequence[int], nominal_pitch: int, start: int, end: int) -> list[tuple[int, float]]:
    nominal = max(10, int(nominal_pitch))
    low = max(10, round(nominal * 0.82))
    high = min(128, max(low + 2, round(nominal * 1.45)))
    scored: list[tuple[float, int, float]] = []
    for pitch in range(low, high + 1):
        corr = _autocorrelation(rows, pitch, start, end)
        # Japanese body glyphs are roughly square, but the actual vertical cell
        # is usually a little taller than the detected body width.  This is a
        # weak tie-breaker only; the projection periodicity remains dominant.
        preference = abs((pitch / nominal) - 1.15)
        scored.append((corr - 0.055 * preference, pitch, corr))
    scored.sort(reverse=True)
    output = [(pitch, corr) for _score, pitch, corr in scored[:6]]
    fallback = max(low, min(high, round(nominal * 1.15)))
    if fallback not in {pitch for pitch, _ in output}:
        output.append((fallback, _autocorrelation(rows, fallback, start, end)))
    return output


def _prefix(values: Sequence[int]) -> list[int]:
    result = [0]
    total = 0
    for value in values:
        total += int(value)
        result.append(total)
    return result


def _range_sum(prefix: Sequence[int], start: int, end: int) -> int:
    start = max(0, min(len(prefix) - 1, int(start)))
    end = max(start, min(len(prefix) - 1, int(end)))
    return int(prefix[end] - prefix[start])


def split_single_column_grid(
    image: Image.Image,
    *,
    nominal_pitch: int,
    threshold: int = 220,
    max_boundary_ink_ratio: float = 0.11,
    max_peak_boundary_ink_ratio: float = 0.18,
) -> ColumnGrid:
    """Find one fixed grid phase for an already isolated physical column."""
    rows, cols, total_ink, bbox = _ink_geometry(image, threshold=threshold)
    if bbox is None or total_ink < 20:
        return ColumnGrid((), 0, 0, 1.0, 1.0, -1.0, False, "no_usable_ink")
    ink_left, ink_top, ink_right, ink_bottom = bbox
    ink_width = max(1, ink_right - ink_left)
    row_prefix = _prefix(rows)
    width = image.width
    height = image.height
    best = None
    for pitch, corr in _candidate_pitches(rows, int(nominal_pitch), ink_top, ink_bottom):
        # A punctuation cell can contain very little ink.  The threshold is
        # deliberately tiny because Ruby/adjacent columns have already been
        # removed before this stage.
        occupied_min = max(4, round(ink_width * pitch * 0.0030))
        for phase in range(pitch):
            first_k = floor((0 - phase) / pitch) - 1
            last_k = ceil((height - phase) / pitch) + 1
            cells: list[GridCell] = []
            cut_ink = 0
            boundary_inks: list[tuple[int, int]] = []
            for k in range(first_k, last_k + 1):
                boundary = phase + k * pitch
                boundary_ink = _range_sum(row_prefix, boundary - 1, boundary + 2)
                cut_ink += boundary_ink
                boundary_inks.append((boundary, boundary_ink))
            for k in range(first_k, last_k):
                top = max(0, phase + k * pitch)
                bottom = min(height, phase + (k + 1) * pitch)
                if bottom <= top:
                    continue
                ink = _range_sum(row_prefix, top, bottom)
                if ink < occupied_min:
                    continue
                cells.append(GridCell(
                    max(0, ink_left - 3), top,
                    min(width, ink_right + 3), bottom,
                    int(ink),
                ))
            if not cells:
                continue
            boundary_ratio = cut_ink / max(1, total_ink)
            # Penalize grids that create implausibly tiny occupied fragments.
            ink_values = sorted(cell.ink_pixels for cell in cells)
            median = ink_values[len(ink_values) // 2]
            tiny = sum(1 for value in ink_values if value < max(occupied_min, median * 0.055))
            occupied_top = min(cell.top for cell in cells)
            occupied_bottom = max(cell.bottom for cell in cells)
            peak_boundary_ink_ratio = max(
                (ink / max(1, median) for boundary, ink in boundary_inks
                 if occupied_top < boundary < occupied_bottom),
                default=0.0,
            )
            # Total cut ink can hide one bad boundary in a long column.  Add a
            # small peak penalty so the selected phase also avoids a single
            # destructive cut without letting one sparse punctuation mark
            # dominate the periodicity score.
            score = (
                boundary_ratio
                + 0.018 * (tiny / max(1, len(cells)))
                + 0.080 * peak_boundary_ink_ratio
                - 0.025 * max(-0.25, min(1.0, corr))
            )
            candidate = (
                score, boundary_ratio, peak_boundary_ink_ratio, -corr,
                pitch, phase, tuple(cells), corr,
            )
            if best is None or candidate < best:
                best = candidate
    if best is None:
        return ColumnGrid((), 0, 0, 1.0, 1.0, -1.0, False, "grid_not_found")
    (
        _score, boundary_ratio, peak_boundary_ink_ratio, _neg_corr,
        pitch, phase, cells, corr,
    ) = best
    total_safe = boundary_ratio <= float(max_boundary_ink_ratio)
    peak_safe = peak_boundary_ink_ratio <= float(max_peak_boundary_ink_ratio)
    safe = bool(total_safe and peak_safe)
    reason = ""
    if not total_safe:
        reason = "boundary_cuts_ink"
    elif not peak_safe:
        reason = "peak_boundary_cuts_ink"
    return ColumnGrid(
        cells=cells,
        pitch=int(pitch),
        phase=int(phase),
        boundary_ink_ratio=float(boundary_ratio),
        peak_boundary_ink_ratio=float(peak_boundary_ink_ratio),
        autocorrelation=float(corr),
        safe=safe,
        reason=reason,
    )


def build_horizontal_reflow(
    isolated_columns: Sequence[Image.Image],
    *,
    nominal_pitches: Sequence[int],
    expected_glyph_count: int,
    scale: float = 1.35,
    gap_ratio: float = 0.14,
    side_margin_ratio: float = 0.55,
) -> HorizontalReflowResult:
    """Reflow occupied fixed-grid cells from one or more physical columns."""
    if not isolated_columns or len(isolated_columns) != len(nominal_pitches):
        return HorizontalReflowResult(None, (), 0, int(expected_glyph_count), False, "invalid_columns")
    expected = int(expected_glyph_count)
    if expected <= 0 or expected > 256:
        return HorizontalReflowResult(None, (), 0, expected, False, "invalid_expected_count")

    grids: list[ColumnGrid] = []
    ordered_crops: list[Image.Image] = []
    try:
        for image, nominal in zip(isolated_columns, nominal_pitches):
            grid = split_single_column_grid(image, nominal_pitch=max(10, int(nominal)))
            grids.append(grid)
            if not grid.safe:
                return HorizontalReflowResult(None, tuple(grids), sum(len(g.cells) for g in grids), expected, False, grid.reason)
            for cell in grid.cells:
                ordered_crops.append(image.crop((cell.left, cell.top, cell.right, cell.bottom)).convert("RGB"))

        glyph_count = len(ordered_crops)
        if glyph_count != expected:
            return HorizontalReflowResult(
                None, tuple(grids), glyph_count, expected, False,
                f"glyph_count_mismatch:{glyph_count}!={expected}",
            )
        if not ordered_crops:
            return HorizontalReflowResult(None, tuple(grids), 0, expected, False, "no_cells")

        scaled: list[Image.Image] = []
        for crop in ordered_crops:
            new_size = (
                max(1, round(crop.width * float(scale))),
                max(1, round(crop.height * float(scale))),
            )
            scaled.append(crop.resize(new_size, Image.Resampling.LANCZOS))
        target_cell = max(max(img.width for img in scaled), max(img.height for img in scaled))
        gap = max(4, round(target_cell * float(gap_ratio)))
        margin_x = max(12, round(target_cell * float(side_margin_ratio)))
        margin_y = max(8, round(target_cell * 0.30))
        canvas_width = margin_x * 2 + glyph_count * target_cell + max(0, glyph_count - 1) * gap
        canvas_height = target_cell + margin_y * 2
        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
        for index, glyph in enumerate(scaled):
            x = margin_x + index * (target_cell + gap) + (target_cell - glyph.width) // 2
            y = margin_y + (target_cell - glyph.height) // 2
            canvas.paste(glyph, (x, y))
        for image in scaled:
            image.close()
        return HorizontalReflowResult(canvas, tuple(grids), glyph_count, expected, True, "")
    finally:
        for crop in ordered_crops:
            crop.close()



def build_horizontal_sentence_reflow(
    isolated_columns: Sequence[Image.Image],
    *,
    nominal_pitches: Sequence[int],
    max_glyphs: int = 255,
    scale: float = 1.35,
    gap_ratio: float = 0.14,
    side_margin_ratio: float = 0.55,
) -> HorizontalReflowResult:
    """Build a geometry-only horizontal sentence strip from vertical columns.

    Unlike :func:`build_horizontal_reflow`, this production sentence transport
    does not require an OCR-derived expected character count.  Each already
    isolated Ruby-free physical column is split by one fixed geometric grid; the
    occupied cells are then emitted in vertical-reading order and concatenated
    left-to-right for a line recognizer such as the 48px AR model.

    The function fails closed when any column grid cuts material ink or when the
    sentence is too large for the recognizer sequence budget.  No recognized text
    is used to place, remove, or reorder glyphs.
    """
    if not isolated_columns or len(isolated_columns) != len(nominal_pitches):
        return HorizontalReflowResult(None, (), 0, 0, False, "invalid_columns")

    grids: list[ColumnGrid] = []
    ordered_crops: list[Image.Image] = []
    try:
        for image, nominal in zip(isolated_columns, nominal_pitches):
            grid = split_single_column_grid(image, nominal_pitch=max(10, int(nominal)))
            grids.append(grid)
            if not grid.safe:
                return HorizontalReflowResult(
                    None, tuple(grids),
                    sum(len(item.cells) for item in grids), 0, False, grid.reason,
                )
            for cell in grid.cells:
                ordered_crops.append(
                    image.crop((cell.left, cell.top, cell.right, cell.bottom)).convert("RGB")
                )

        glyph_count = len(ordered_crops)
        if glyph_count <= 0:
            return HorizontalReflowResult(None, tuple(grids), 0, 0, False, "no_cells")
        if glyph_count > max(1, int(max_glyphs)):
            return HorizontalReflowResult(
                None, tuple(grids), glyph_count, glyph_count, False,
                f"sequence_too_long:{glyph_count}>{int(max_glyphs)}",
            )

        scaled: list[Image.Image] = []
        try:
            for crop in ordered_crops:
                new_size = (
                    max(1, round(crop.width * float(scale))),
                    max(1, round(crop.height * float(scale))),
                )
                scaled.append(crop.resize(new_size, Image.Resampling.LANCZOS))
            target_cell = max(
                max(img.width for img in scaled),
                max(img.height for img in scaled),
            )
            gap = max(4, round(target_cell * float(gap_ratio)))
            margin_x = max(12, round(target_cell * float(side_margin_ratio)))
            margin_y = max(8, round(target_cell * 0.30))
            canvas_width = (
                margin_x * 2
                + glyph_count * target_cell
                + max(0, glyph_count - 1) * gap
            )
            canvas_height = target_cell + margin_y * 2
            canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
            for index, glyph in enumerate(scaled):
                x = margin_x + index * (target_cell + gap) + (target_cell - glyph.width) // 2
                y = margin_y + (target_cell - glyph.height) // 2
                canvas.paste(glyph, (x, y))
            return HorizontalReflowResult(
                canvas, tuple(grids), glyph_count, glyph_count, True, ""
            )
        finally:
            for image in scaled:
                image.close()
    finally:
        for crop in ordered_crops:
            crop.close()

def build_horizontal_focus_reflow(
    isolated_columns: Sequence[Image.Image],
    *,
    nominal_pitches: Sequence[int],
    expected_glyph_count: int,
    focus_start: int,
    focus_end: int,
    context_cells: int = 3,
    scale: float = 1.50,
    gap_ratio: float = 0.22,
    side_margin_ratio: float = 0.55,
) -> HorizontalReflowResult:
    """Reflow one verified disagreement window from fixed physical cells.

    This is intentionally stricter than a generic crop.  The complete set of
    isolated physical columns is first segmented with one fixed grid per column,
    and the *total* occupied-cell count must equal the pre-existing strict-majority
    comparison key.  Only then may a small deterministic window around the
    disagreement be emitted.  The OCR text is never used to guess geometry.
    """
    if not isolated_columns or len(isolated_columns) != len(nominal_pitches):
        return HorizontalReflowResult(None, (), 0, int(expected_glyph_count), False, "invalid_columns")
    expected = int(expected_glyph_count)
    start = int(focus_start)
    end = int(focus_end)
    context = max(1, min(6, int(context_cells)))
    if expected <= 0 or expected > 256 or not (0 <= start <= end <= expected):
        return HorizontalReflowResult(None, (), 0, expected, False, "invalid_focus_window")
    # An insertion in the dissenting OCR can have a zero-width span in the
    # majority key.  Anchor that case to the next physical cell so the window
    # still contains real source pixels on both sides of the disagreement.
    anchor_end = max(end, min(expected, start + 1))
    window_start = max(0, start - context)
    window_end = min(expected, anchor_end + context)
    if window_end - window_start < 3:
        return HorizontalReflowResult(None, (), 0, expected, False, "focus_window_too_short")

    grids: list[ColumnGrid] = []
    all_cells: list[tuple[Image.Image, GridCell]] = []
    crops: list[Image.Image] = []
    scaled: list[Image.Image] = []
    try:
        for image, nominal in zip(isolated_columns, nominal_pitches):
            grid = split_single_column_grid(image, nominal_pitch=max(10, int(nominal)))
            grids.append(grid)
            if not grid.safe:
                return HorizontalReflowResult(
                    None, tuple(grids), sum(len(g.cells) for g in grids), expected, False, grid.reason
                )
            all_cells.extend((image, cell) for cell in grid.cells)
        glyph_count = len(all_cells)
        if glyph_count != expected:
            return HorizontalReflowResult(
                None, tuple(grids), glyph_count, expected, False,
                f"glyph_count_mismatch:{glyph_count}!={expected}",
            )
        selected = all_cells[window_start:window_end]
        if not selected:
            return HorizontalReflowResult(None, tuple(grids), glyph_count, expected, False, "no_focus_cells")
        for image, cell in selected:
            crops.append(image.crop((cell.left, cell.top, cell.right, cell.bottom)).convert("RGB"))
        for crop in crops:
            scaled.append(crop.resize(
                (max(1, round(crop.width * float(scale))), max(1, round(crop.height * float(scale)))),
                Image.Resampling.LANCZOS,
            ))
        target_cell = max(max(img.width for img in scaled), max(img.height for img in scaled))
        gap = max(4, round(target_cell * float(gap_ratio)))
        margin_x = max(12, round(target_cell * float(side_margin_ratio)))
        margin_y = max(8, round(target_cell * 0.30))
        canvas_width = margin_x * 2 + len(scaled) * target_cell + max(0, len(scaled) - 1) * gap
        canvas_height = target_cell + margin_y * 2
        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
        for index, glyph in enumerate(scaled):
            x = margin_x + index * (target_cell + gap) + (target_cell - glyph.width) // 2
            y = margin_y + (target_cell - glyph.height) // 2
            canvas.paste(glyph, (x, y))
        return HorizontalReflowResult(
            canvas, tuple(grids), glyph_count, expected, True, "", window_start, window_end
        )
    finally:
        for image in scaled:
            image.close()
        for crop in crops:
            crop.close()
