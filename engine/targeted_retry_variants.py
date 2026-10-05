"""Bounded, Ruby-free alternate inputs for local OCR conflict confirmation.

These images are evidence for a dissenting recognizer, never new model votes.
Full-row readings can confirm an exact strict-majority candidate. Proportional
masked crops stay diagnostic-only. For a small local disagreement, a whole-row
horizontal reflow may confirm only the original dispute when fixed physical cells
match the majority-key length and local text anchors map the target unambiguously.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

from PIL import Image, ImageChops

from adapters.column_ocr_adapter import _candidate_text, _isolated_column_image, _masked_column_image
from adapters.ocr_recognition_bridge import recognizer_iterator
from engine.targeted_retry_adjudicator import TargetedRetryEvidence, strict_majority_candidate
from engine.horizontal_reflow_retry import build_horizontal_reflow
from engine.ocr_unicode_standardizer import japanese_ocr_comparison_key


_COLUMN_ID = re.compile(r"^p(\d+):c(\d+)$")
_PUNCTUATION = frozenset("「」『』（）()？！?!、。…‥：；")
MAX_PIXELS_PER_INPUT = 8_000_000


@dataclass(frozen=True, slots=True)
class RetryInput:
    row_index: int
    kind: str
    path: str
    width: int
    height: int
    column_ids: tuple[str, ...]
    authoritative: bool
    full_row_reading: bool = True
    span_start: int = -1
    span_end: int = -1
    expected_comparison_key: str = ""
    grid_verified: bool = False
    glyph_count_matches_majority: bool = False
    grid_boundary_ink_ratio: float = 1.0
    grid_pitches: tuple[int, ...] = ()
    grid_peak_boundary_ink_ratio: float = 1.0
    horizontal_context_cells: int = 0
    physical_column_boxes: tuple[tuple[int, int, int, int], ...] = ()


def _pad(image: Image.Image, x: int, y: int) -> Image.Image:
    canvas = Image.new("RGB", (image.width + 2 * x, image.height + 2 * y), "white")
    canvas.paste(image, (x, y))
    return canvas


def _column_refs(cache, column_ids: tuple[str, ...]):
    refs = []
    for column_id in column_ids:
        match = _COLUMN_ID.fullmatch(column_id)
        if match is None:
            return []
        page = cache.pages.get(int(match.group(1)))
        index = int(match.group(2)) - 1
        if page is None or not 0 <= index < len(page.columns):
            return []
        refs.append((page, page.columns[index]))
    return refs if refs and len({page.page_number for page, _ in refs}) == 1 else []


def _sentence_bounds(source: Image.Image, columns) -> tuple[int, int, int, int]:
    return (
        max(0, min(column.left for column in columns) - 12),
        max(0, min(column.top for column in columns) - 24),
        min(source.width, max(column.right for column in columns) + 12),
        min(source.height, max(column.bottom for column in columns) + 24),
    )


def _sentence_image(source: Image.Image, columns) -> Image.Image:
    # Compose approved body pixels in original page coordinates.  This keeps
    # spacing and punctuation while removing adjacent columns and side Ruby.
    canvas = Image.new("RGB", source.size, "white")
    for column in columns:
        masked = _masked_column_image(source, column, preserve_body_pixels=False)
        try:
            merged = ImageChops.darker(canvas, masked)
        finally:
            masked.close()
            canvas.close()
        canvas = merged
    result = canvas.crop(_sentence_bounds(source, columns))
    canvas.close()
    return result


def _comparison_key(text: str) -> str:
    return str(japanese_ocr_comparison_key(str(text or ""))[0] or "")


def _focus_image(image: Image.Image, original: str, majority: str) -> Image.Image | None:
    """Build a diagnostic-only focus mask when text→geometry mapping is safe enough.

    A vertical OCR string is not a geometric transcript: insertions, deletions,
    punctuation and Ruby removal can shift character indices.  Therefore this
    helper deliberately refuses indels/length mismatches.  For a single short
    equal-length replacement it maps the edit into the *ink bounding box* rather
    than the full padded canvas.  The result remains diagnostic and can never
    auto-confirm a row.
    """
    original_key = _comparison_key(original)
    majority_key = _comparison_key(majority)
    if (
        len(original_key) < 4
        or len(majority_key) < 4
        or len(original_key) != len(majority_key)
        or image.height < 120
    ):
        return None
    changes = [
        op for op in SequenceMatcher(None, majority_key, original_key, autojunk=False).get_opcodes()
        if op[0] != "equal"
    ]
    if len(changes) != 1:
        return None
    tag, majority_start, majority_end, original_start, original_end = changes[0]
    if (
        tag != "replace"
        or majority_end - majority_start != original_end - original_start
        or not (1 <= majority_end - majority_start <= 2)
    ):
        return None

    gray = image.convert("L")
    try:
        # Dark print -> white mask, paper -> black.  Use a forgiving threshold
        # so slightly off-white scans do not expand the geometry to the canvas.
        ink_mask = gray.point(lambda value: 255 if value < 220 else 0, mode="1")
        try:
            bbox = ink_mask.getbbox()
        finally:
            ink_mask.close()
    finally:
        gray.close()
    if bbox is None:
        return None
    _left, ink_top, _right, ink_bottom = bbox
    ink_height = max(1, ink_bottom - ink_top)
    char_pitch = ink_height / max(1, len(original_key))
    center_index = (original_start + original_end) / 2.0
    center = round(ink_top + center_index * char_pitch)
    radius = max(48, round(char_pitch * 3.5))
    top = max(0, center - radius)
    bottom = min(image.height, center + radius)
    if bottom <= top or bottom - top >= image.height * 0.85:
        return None
    masked = Image.new("RGB", image.size, "white")
    region = image.crop((0, top, image.width, bottom))
    try:
        masked.paste(region, (0, top))
    finally:
        region.close()
    return masked


def build_retry_inputs(
    cache,
    row,
    row_index: int,
    model_index: int,
    directory: Path,
    *,
    cancel_check: Callable[[], bool] | None = None,
    include_standard: bool = True,
    include_horizontal: bool = True,
    include_diagnostic: bool = True,
) -> list[RetryInput]:
    """Create the requested retry stages for one dissenting model/row.

    ``include_*`` gates exist so the GUI can short-circuit expensive stages:
    canonical/expanded inputs run first, verified horizontal reflow is created
    only for rows that remain unresolved, and diagnostic-only masks need not be
    OCR'd for rows already confirmed by an authoritative retry.
    """
    column_ids = tuple(str(v) for v in (getattr(row, "column_ids", ()) or ()))
    refs = _column_refs(cache, column_ids)
    if not refs or (cancel_check is not None and cancel_check()):
        return []
    page = refs[0][0]
    columns = [column for _page, column in refs]
    images: list[tuple] = []
    column_boxes = ()
    with Image.open(page.page_path) as source:
        source.load()
        if len(columns) == 1:
            base = _isolated_column_image(source, columns[0], preserve_body_pixels=False)
            if include_standard:
                images.append(("single_column", base.copy(), True, True, -1, -1, "", False, False, 1.0, (), 1.0, 0))
            pad_x, pad_y = 24, 72
        else:
            base = _sentence_image(source, columns)
            left, top, right, bottom = _sentence_bounds(source, columns)
            # Keep the approved physical partition, rather than letting a
            # line recognizer estimate the entire sentence as one wide glyph.
            column_boxes = tuple((
                max(left, column.hard_left) - left,
                max(top, column.top - 24) - top,
                min(right, column.hard_right) - left,
                min(bottom, column.bottom + 24) - top,
            ) for column in columns)
            if include_standard:
                images.append(("sentence_context", base.copy(), True, True, -1, -1, "", False, False, 1.0, (), 1.0, 0))
            pad_x, pad_y = 55, 150
        if include_standard:
            images.append(("expanded_paper", _pad(base, pad_x, pad_y), True, True, -1, -1, "", False, False, 1.0, (), 1.0, 0))
            if base.width * base.height <= 650_000 and base.height < 1100:
                images.append(("sentence_2x", base.resize((base.width * 2, base.height * 2), Image.Resampling.LANCZOS), True, True, -1, -1, "", False, False, 1.0, (), 1.0, 0))
        texts = list(getattr(row, "texts", ()) or ())
        original = str(texts[model_index] or "") if model_index < len(texts) else ""
        strict_majority = strict_majority_candidate(row)
        majority = str(strict_majority[1] or "") if strict_majority is not None else ""
        # Model-agnostic horizontal reflow uses only already-isolated Ruby-free
        # physical columns.  For a small local disagreement we reflow the *whole*
        # row so every OCR engine keeps maximum lexical context, but adjudication
        # later inspects only the original disagreement position.  Unrelated
        # punctuation/characters may therefore be wrong in horizontal OCR without
        # invalidating an otherwise well-anchored target-glyph confirmation.
        if majority and include_horizontal:
            column_images = []
            try:
                for column in columns:
                    column_images.append(_isolated_column_image(
                        source, column, preserve_body_pixels=False
                    ))
                nominal_pitches = [max(10, int(column.width)) for column in columns]
                original_key = _comparison_key(original)
                majority_key = _comparison_key(majority)
                changes = [
                    op for op in SequenceMatcher(None, majority_key, original_key, autojunk=False).get_opcodes()
                    if op[0] != "equal"
                ]
                local_dispute = False
                dispute_start = dispute_end = -1
                if len(changes) == 1 and len(majority_key) <= 256:
                    _tag, i1, i2, j1, j2 = changes[0]
                    majority_changed = i2 - i1
                    original_changed = j2 - j1
                    if max(majority_changed, original_changed) <= 2:
                        local_dispute = True
                        dispute_start, dispute_end = int(i1), int(i2)

                reflow = build_horizontal_reflow(
                    column_images,
                    nominal_pitches=nominal_pitches,
                    expected_glyph_count=len(majority_key),
                )
                if reflow.safe and reflow.image is not None:
                    max_boundary = max(
                        (float(grid.boundary_ink_ratio) for grid in reflow.grids),
                        default=1.0,
                    )
                    max_peak_boundary = max(
                        (float(grid.peak_boundary_ink_ratio) for grid in reflow.grids),
                        default=1.0,
                    )
                    pitches = tuple(int(grid.pitch) for grid in reflow.grids)
                    images.append((
                        ("horizontal_reflow_dispute" if local_dispute else "horizontal_reflow"),
                        reflow.image, True, True, dispute_start, dispute_end, majority_key,
                        True, reflow.glyph_count == len(majority_key), max_boundary, pitches,
                        max_peak_boundary, 0,
                    ))
            finally:
                for column_image in column_images:
                    column_image.close()
        if include_diagnostic and majority and (any(ch in original + majority for ch in _PUNCTUATION) or original != majority):
            focus = _focus_image(base, original, majority)
            if focus is not None:
                images.append(("masked_phrase_diagnostic", focus, False, False, -1, -1, "", False, False, 1.0, (), 1.0, 0))
        base.close()

    directory.mkdir(parents=True, exist_ok=True)
    output: list[RetryInput] = []
    for (
        kind, image, authoritative, full_row_reading, span_start, span_end, expected_key,
        grid_verified, glyph_count_matches_majority, grid_boundary_ink_ratio, grid_pitches,
        grid_peak_boundary_ink_ratio, horizontal_context_cells,
    ) in images:
        try:
            if image.width * image.height > MAX_PIXELS_PER_INPUT:
                continue
            path = directory / f"r{row_index:05d}_m{model_index}_{kind}.png"
            image.save(path)
            boxes = column_boxes if not kind.startswith("horizontal_reflow") else ()
            if kind == "expanded_paper":
                boxes = tuple((l + pad_x, t + pad_y, r + pad_x, b + pad_y)
                              for l, t, r, b in boxes)
            elif kind == "sentence_2x":
                boxes = tuple(tuple(value * 2 for value in box) for box in boxes)
            output.append(RetryInput(
                row_index, kind, str(path), image.width, image.height, column_ids, authoritative,
                full_row_reading, span_start, span_end, expected_key,
                grid_verified, glyph_count_matches_majority, grid_boundary_ink_ratio, grid_pitches,
                grid_peak_boundary_ink_ratio, horizontal_context_cells, boxes,
            ))
        finally:
            image.close()
    return output



def partition_standard_retry_inputs(inputs: list[RetryInput]) -> tuple[list[RetryInput], list[RetryInput], list[RetryInput]]:
    """Return production retry tiers in cheapest-first order.

    Canonical/sentence-context rereads are cheapest and often sufficient.
    Expanded-paper and 2x raster variants are therefore held back until a row
    survives the previous tier.  Keeping this partition in the engine layer
    makes GUI orchestration deterministic and prevents a future refactor from
    accidentally batching canonical + expanded inputs together again.
    """
    canonical: list[RetryInput] = []
    expanded: list[RetryInput] = []
    scaled: list[RetryInput] = []
    for item in inputs:
        if item.kind in {"single_column", "sentence_context"}:
            canonical.append(item)
        elif item.kind == "expanded_paper":
            expanded.append(item)
        elif item.kind == "sentence_2x":
            scaled.append(item)
    return canonical, expanded, scaled

def recognize_retry_inputs(
    engine: str,
    model_index: int,
    model_label: str,
    rows,
    inputs: list[RetryInput],
    directory: Path,
    *,
    engine_options: dict | None = None,
    cancel_check: Callable[[], bool] | None = None,
    recognition_session=None,
) -> tuple[list[TargetedRetryEvidence], list[dict]]:
    """Reuse one model session where possible across all alternate views.

    Proportional/masked focus images remain diagnostic-only. Production local
    disputes use a whole-row fixed-grid ``horizontal_reflow_dispute`` image; the
    adjudicator later checks only its original disagreement position. Legacy
    ``horizontal_reflow_focus`` evidence remains readable for old projects.
    """
    by_path = {item.path: item for item in inputs}
    evidence: list[TargetedRetryEvidence] = []
    diagnostics: list[dict] = []
    # Heavy local OCR engines share one persistent retry session across standard
    # and horizontal inputs.  Apple Vision is the only exception because its
    # request configuration carries an explicit vertical flag.
    engine_key = str(engine or "").strip().lower()
    apple_engine = engine_key in {"apple_vision", "macocr", "mac_ocr", "macos_ocr"}
    if apple_engine:
        groups = [
            ("standard", [item for item in inputs if not item.kind.startswith("horizontal_reflow")]),
            ("horizontal", [item for item in inputs if item.kind.startswith("horizontal_reflow")]),
        ]
    else:
        # Keep heavy local recognizers in one persistent session.  Hayai, Manga
        # OCR and 48px already consume the prepared crop itself and can see the
        # wide/short horizontal geometry without a second model startup.
        groups = [("mixed", list(inputs))]
    for layout, group in groups:
        if not group:
            continue
        options = dict(engine_options or {})
        if layout == "horizontal":
            options["input_layout"] = "horizontal_reflow"
            options["vertical"] = False
            options["orientation"] = "auto"
            options["vertical_preprocess"] = "none"
        input_metadata = {
            item.path: {
                "layout": ("horizontal_reflow" if item.kind.startswith("horizontal_reflow") else "standard"),
                "expected_chars": max(
                    1,
                    len(str(item.expected_comparison_key or ""))
                    if item.kind.startswith("horizontal_reflow")
                    else 1,
                ),
                "retry_kind": item.kind,
                "physical_column_boxes": item.physical_column_boxes,
            }
            for item in group
        }
        if recognition_session is None:
            iterator = recognizer_iterator(
                engine, [item.path for item in group], str(directory / f"manifest_{layout}.json"),
                engine_options=options, input_metadata=input_metadata,
                cancel_check=cancel_check, verbose=False,
            )
        else:
            iterator = recognition_session.iter_recognize(
                [item.path for item in group],
                engine_options=options,
                input_metadata=input_metadata,
            )
        for path, blocks, error in iterator:
            if cancel_check is not None and cancel_check():
                raise InterruptedError("OCR 已停止")
            item = by_path.get(str(path))
            if item is None:
                continue
            horizontal = item.kind.startswith("horizontal_reflow")
            text, confidence = _candidate_text(
                blocks, recognition_engine=engine, image_path=item.path,
                # A horizontal reflow is already a single authoritative text
                # line even when it was assembled from several physical columns.
                # This also bypasses Apple side-text filtering intended for raw
                # page regions, which is inappropriate after Ruby-free reflow.
                authoritative_column=bool(item.authoritative and (len(item.column_ids) == 1 or horizontal)),
            ) if not error else ("", 0.0)
            diagnostics.append({
                "row_index": item.row_index, "model": model_label, "kind": item.kind,
                "text": text, "error": error, "size": [item.width, item.height],
                "authoritative": item.authoritative,
                "full_row_reading": item.full_row_reading,
                "input_layout": ("horizontal" if horizontal else "standard"),
                "span": ([item.span_start, item.span_end] if item.span_start >= 0 else None),
                "horizontal_context_cells": int(item.horizontal_context_cells),
            })
            if not text.strip() or not item.authoritative:
                continue
            original = str(rows[item.row_index].texts[model_index] or "")
            evidence.append(TargetedRetryEvidence(
                row_index=item.row_index, model_index=model_index,
                model_label=model_label, original_text=original, retry_text=text,
                retry_kind=item.kind, confidence=confidence,
                input_sha256=hashlib.sha256(Path(item.path).read_bytes()).hexdigest(),
                input_path=item.path,
                details={
                    "size": [item.width, item.height], "column_ids": list(item.column_ids),
                    "ruby_free": True, "full_row_reading": bool(item.full_row_reading),
                    "input_layout": ("horizontal" if horizontal else "standard"),
                    "verification_scope": (
                        "dispute_only" if item.kind == "horizontal_reflow_dispute"
                        else ("verified_window" if item.kind == "horizontal_reflow_focus" else "full_row")
                    ),
                    "verified_span_reading": bool(
                        item.authoritative and not item.full_row_reading
                        and item.span_start >= 0 and item.span_end > item.span_start
                        and item.expected_comparison_key
                    ),
                    "span_start": int(item.span_start), "span_end": int(item.span_end),
                    "expected_comparison_key": str(item.expected_comparison_key or ""),
                    "grid_verified": bool(item.grid_verified),
                    "glyph_count_matches_majority": bool(item.glyph_count_matches_majority),
                    "grid_boundary_ink_ratio": float(item.grid_boundary_ink_ratio),
                    "grid_peak_boundary_ink_ratio": float(item.grid_peak_boundary_ink_ratio),
                    "grid_pitches": list(item.grid_pitches),
                    "horizontal_context_cells": int(item.horizontal_context_cells),
                    "retry_counts_as_independent_vote": False,
                },
            ))
    return evidence, diagnostics
