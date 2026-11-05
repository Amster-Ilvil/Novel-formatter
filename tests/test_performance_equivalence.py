"""Equivalence guards for the OCR performance work.

The segmentation reference in ``tests/fixtures/segmentation_reference.json`` was
captured from the baseline commit b962ff0 (before the single-pass connected
components rewrite) by running this file as a script from that checkout::

    cd /tmp/nf-base && /path/to/.venv/bin/python /path/to/tests/test_performance_equivalence.py \
        --write /path/to/tests/fixtures/segmentation_reference.json

The page is generated deterministically in code, so no image is stored.
"""
from __future__ import annotations

import dataclasses
import json
import os
import random
import sys
from pathlib import Path

REFERENCE = Path(__file__).resolve().parent / "fixtures" / "segmentation_reference.json"
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


def test_segmentation_matches_baseline_reference():
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    current = segmentation_signature()
    assert reference["components"], "reference must contain ink components"
    assert reference["columns"], "reference must contain detected columns"
    assert current["components"] == reference["components"]
    assert current["columns"] == reference["columns"]


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    try:
        _, flag, out = sys.argv
    except ValueError:
        flag, out = "", ""
    if flag == "--write" and out:
        target = Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = segmentation_signature()
        target.write_text(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"{target}: {len(data['components'])} components, {len(data['columns'])} columns")
    else:
        raise SystemExit("usage: test_performance_equivalence.py --write OUT.json")
