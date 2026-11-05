"""Equivalence guards for the OCR performance work.

The segmentation reference in ``tests/fixtures/segmentation_reference.json`` was
captured from the baseline commit b962ff0 (before the single-pass connected
components rewrite) by running this file as a script from that checkout::

    cd /tmp/nf-base && /path/to/.venv/bin/python /path/to/tests/test_performance_equivalence.py \
        --write /path/to/tests/fixtures/segmentation_reference.json

The column-image reference in ``tests/fixtures/column_image_reference.json`` was
captured from commit 858427f the same way with ``--write-columns``.  It pins the
exact pixels (mode, size and raw bytes) of every per-column OCR input built from
the generated page, so any future change to column masking, paper-background
estimation or ink framing must stay byte-identical.

The page is generated deterministically in code, so no image is stored.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
import sys
from pathlib import Path

REFERENCE = Path(__file__).resolve().parent / "fixtures" / "segmentation_reference.json"
COLUMN_REFERENCE = Path(__file__).resolve().parent / "fixtures" / "column_image_reference.json"
COLUMN_ENGINE = "apple_vision"
PAGE_SIZE = (1240, 1754)
SEED = 4242


def make_page():
    """A vertical-text-like page: glyph clusters in columns, ruby marks, a frame."""
    from PIL import Image, ImageDraw

    rng = random.Random(SEED)
    width, height = PAGE_SIZE
    image = Image.new("L", PAGE_SIZE, 250)
    draw = ImageDraw.Draw(image)
    glyph, pitch = 34, 58
    x = width - 120
    while x > 110:
        y = 140
        limit = rng.randint(height // 2, height - 140)
        while y < limit:
            if rng.random() < 0.08:
                y += glyph
                continue
            for _ in range(rng.randint(2, 5)):
                x0 = x + rng.randint(0, glyph - 8)
                y0 = y + rng.randint(0, glyph - 8)
                if rng.random() < 0.5:
                    draw.rectangle((x0, y0, x0 + rng.randint(3, 9), y0 + rng.randint(10, glyph - 4)), fill=rng.randint(0, 60))
                else:
                    draw.line((x0, y0, x0 + rng.randint(-8, 8), y0 + rng.randint(4, 20)), fill=rng.randint(0, 60), width=3)
            if rng.random() < 0.12:
                draw.rectangle((x + glyph + 4, y + 4, x + glyph + 12, y + 12), fill=40)
            y += glyph + rng.randint(2, 6)
        x -= pitch
    draw.rectangle((60, 60, width - 60, 64), fill=30)
    draw.text((width // 2 - 10, height - 90), "12", fill=20)
    return image


def ink_mask(image):
    return image.point(lambda value: 255 if value < 128 else 0)


def _column_signature(column):
    if dataclasses.is_dataclass(column):
        return dataclasses.asdict(column)
    return repr(column)


def segmentation_signature() -> dict:
    """Components and detected columns of the generated page for the cwd checkout."""
    from adapters.column_ocr_adapter import _scanline_connected_ink_components, detect_vertical_columns

    page = make_page()
    components = _scanline_connected_ink_components(ink_mask(page))
    columns = detect_vertical_columns(page.convert("RGB"))
    payload = {
        "page_size": list(PAGE_SIZE),
        "seed": SEED,
        "components": [[c.left, c.top, c.right, c.bottom, c.area] for c in components],
        "columns": [_column_signature(column) for column in columns],
    }
    return json.loads(json.dumps(payload, sort_keys=True, default=str))


def _image_digest(image) -> dict:
    """SHA-256 over mode, size and raw pixel bytes of one OCR input image."""
    digest = hashlib.sha256()
    digest.update(image.mode.encode("ascii"))
    digest.update(f"{image.width}x{image.height}".encode("ascii"))
    digest.update(image.tobytes())
    return {"mode": image.mode, "size": [image.width, image.height], "sha256": digest.hexdigest()}


def column_image_signature() -> dict:
    """Per-column OCR input images of the generated page for the cwd checkout.

    Mirrors the production preparation path: the paper background is estimated
    once per page, each column goes through the canonical visibility viewport
    and the final ink framing, and every content span is rendered through the
    short-block recovery transport.  The raw masked layer is pinned as well.
    """
    import adapters.column_ocr_adapter as column_ocr

    page = make_page().convert("RGB")
    columns = column_ocr.detect_vertical_columns(page)
    background = column_ocr._paper_background(page)
    profile = column_ocr.transport_profile_id(COLUMN_ENGINE)
    images: list[dict] = []

    def record(kind: str, column_index: int, preserve: bool, image, span_index: int | None = None) -> None:
        try:
            entry = {"kind": kind, "column": column_index, "preserve_body_pixels": preserve}
            if span_index is not None:
                entry["span"] = span_index
            entry.update(_image_digest(image))
            images.append(entry)
        finally:
            image.close()

    for preserve in (False, True):
        for column_index, column in enumerate(columns):
            record("masked_layer", column_index, preserve, column_ocr._masked_column_image(
                page, column, retry=False, background=background, preserve_body_pixels=preserve,
            ))
            viewport = column_ocr._column_visibility_viewport(
                page, column, mode="compact", background=background,
                preserve_body_pixels=preserve, transport_profile=profile,
            )
            try:
                framed = column_ocr._tighten_ink_framing(viewport, column.width, profile)
            finally:
                viewport.close()
            record("transport", column_index, preserve, framed)
            for span_index, span in enumerate(column.content_spans or ()):
                record("short_block_span", column_index, preserve, column_ocr._crop_content_span(
                    page, column, span, background=background, preserve_body_pixels=preserve,
                ), span_index)
    page.close()
    payload = {
        "page_size": list(PAGE_SIZE),
        "seed": SEED,
        "engine": COLUMN_ENGINE,
        "transport_profile": profile,
        "background": list(background),
        "columns": len(columns),
        "images": images,
    }
    return json.loads(json.dumps(payload, sort_keys=True))


def test_segmentation_matches_baseline_reference():
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    current = segmentation_signature()
    assert reference["components"], "reference must contain ink components"
    assert reference["columns"], "reference must contain detected columns"
    assert current["components"] == reference["components"]
    assert current["columns"] == reference["columns"]


def test_column_images_match_reference():
    reference = json.loads(COLUMN_REFERENCE.read_text(encoding="utf-8"))
    current = column_image_signature()
    kinds = {entry["kind"] for entry in reference["images"]}
    assert reference["columns"] > 1, "reference must contain several columns"
    assert {"masked_layer", "transport", "short_block_span"} <= kinds
    assert current["background"] == reference["background"]
    assert current["columns"] == reference["columns"]
    assert len(current["images"]) == len(reference["images"])
    for got, want in zip(current["images"], reference["images"]):
        assert got == want


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    try:
        _, flag, out = sys.argv
    except ValueError:
        flag, out = "", ""
    if flag in {"--write", "--write-columns"} and out:
        target = Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = segmentation_signature() if flag == "--write" else column_image_signature()
        target.write_text(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        if flag == "--write":
            print(f"{target}: {len(data['components'])} components, {len(data['columns'])} columns")
        else:
            print(f"{target}: {data['columns']} columns, {len(data['images'])} images")
    else:
        raise SystemExit("usage: test_performance_equivalence.py --write OUT.json | --write-columns OUT.json")
