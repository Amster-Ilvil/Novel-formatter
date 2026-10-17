# -*- coding: utf-8 -*-
"""Optional projection-based vertical glyph GRID CANDIDATES (never proof).

Designed for already-cropped full-height physical Japanese columns. The
result is a proposal for human/neighbor OCR alignment, NOT character-level
OCR ground truth. Ruby, dropped glyphs and abnormal punctuation trigger flags.
No OCR model is downloaded or used here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def suggest_vertical_cells(
    image_path: str | Path,
    *,
    expected_character_count: int | None = None,
    ink_threshold: int = 205,
    pitch_min_factor: float = 0.72,
    pitch_max_factor: float = 1.30,
) -> dict[str, Any]:
    from PIL import Image

    with Image.open(image_path) as source:
        gray = source.convert("L")
        gray.load()
    width, height = gray.size
    if width < 18 or height < width * 3:
        return {"ok": False, "reason": "not_a_narrow_physical_column", "boxes": []}
    # The center stripe suppresses marginal ruby characters and column borders.
    # It must not be treated as proof that ruby did not enter the main line.
    x0 = max(0, round(width * 0.20))
    x1 = min(width, max(x0 + 4, round(width * 0.80)))
    sample = gray.crop((x0, 0, x1, height))
    pixels = sample.tobytes()
    sw = x1 - x0
    proj = [
        sum(pixels[y * sw + x] < ink_threshold for x in range(sw))
        for y in range(height)
    ]
    total_ink = sum(proj)
    if total_ink < height:
        return {"ok": False, "reason": "too_little_main_column_ink", "boxes": []}
    lower = max(8, round(width * pitch_min_factor))
    upper = min(height // 3, round(width * pitch_max_factor))
    if lower > upper:
        return {"ok": False, "reason": "pitch_search_range_invalid", "boxes": []}
    pitches: list[tuple[float, int]] = []
    for pitch in range(lower, upper + 1):
        # Ink autocorrelation across repeated printed cells; large foreground
        # runs alone are insufficient evidence of a correct pitch.
        corr = sum(min(proj[y], proj[y + pitch]) for y in range(height - pitch))
        availability = sum(proj[: height - pitch]) or 1
        periodicity = corr / availability
        width_penalty = 0.035 * abs(pitch - width) / width
        pitches.append((periodicity - width_penalty, pitch))
    pitches.sort(reverse=True)
    _best_score, pitch = pitches[0]
    if _best_score < 0.12:
        return {
            "ok": False, "reason": "no_reliable_ink_periodicity",
            "boxes": [], "pitch_candidate": pitch,
        }

    # A good cut falls into the low-ink gap between characters. Evaluate both
    # crossing ink AND ink concentration in the predicted glyph center.
    phase_scores: list[tuple[float, int, float, float]] = []
    for phase in range(pitch):
        boundaries = range(phase, height, pitch)
        boundary_ink = sum(
            proj[y + delta]
            for y in boundaries for delta in (-1, 0, 1)
            if 0 <= y + delta < height
        )
        centers = range(phase + pitch // 2, height, pitch)
        center_ink = sum(
            proj[y + delta]
            for y in centers
            for delta in (-max(1, pitch // 8), 0, max(1, pitch // 8))
            if 0 <= y + delta < height
        )
        # Both terms use the same count of 3 sampled horizontal profiles.
        score = center_ink - 1.3 * boundary_ink
        phase_scores.append((score, phase, float(boundary_ink), float(center_ink)))
    phase_scores.sort(reverse=True)
    best, phase, border, center = phase_scores[0]
    if center <= 0 or best <= 0:
        return {"ok": False, "reason": "no_safe_grid_phase", "boxes": [], "pitch_candidate": pitch}
    boxes: list[tuple[int, int, int, int]] = []
    y = phase
    # Reject a nonempty leading cut rather than silently shifting index 0.
    if phase > 0 and sum(proj[:phase]) > max(3, total_ink * 0.015):
        return {
            "ok": False, "reason": "nonempty_leading_partial_cell",
            "pitch_candidate": pitch, "phase_candidate": phase, "boxes": [],
        }
    while y + pitch <= height:
        box_ink = sum(proj[y: y + pitch])
        boxes.append((x0, y, x1, y + pitch))
        y += pitch
    trailing_ink = sum(proj[y:])
    warnings: list[str] = []
    if trailing_ink > max(3, total_ink * 0.015):
        warnings.append("trailing_partial_glyph_or_footer")
    if expected_character_count is not None and abs(len(boxes) - int(expected_character_count)) > 1:
        warnings.append("ocr_length_disagrees_with_physical_grid")
    empty_cells = sum(sum(proj[b[1]:b[3]]) < max(2, pitch * 0.2) for b in boxes)
    if empty_cells:
        warnings.append("suspicious_empty_cells")
    if border / max(1, center) > 0.30:
        warnings.append("boundary_crosses_significant_ink")
    return {
        "ok": bool(boxes),
        "position_verified": False,
        "reason": "geometry_candidates_only",
        "boxes": boxes,
        "pitch": pitch,
        "phase": phase,
        "main_column_x": [x0, x1],
        "mean_border_to_center_ink": round(border / max(1, center), 6),
        "warnings": warnings,
        "requires_independent_ocr_anchors": True,
        "requires_visual_spot_check": True,
    }
