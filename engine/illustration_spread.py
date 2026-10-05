# -*- coding: utf-8 -*-
"""Conservative double-page illustration detection.

The detector is intentionally non-destructive: it never stitches, crops, or
rewrites images.  It only emits high-confidence pairing evidence that EPUB can
encode with ``page-spread-left/right``.  False negatives are preferable to
joining two independent illustrations.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from statistics import mean

from PIL import Image, ImageStat

from models.document import BlockType, UnifiedDocument


_ELIGIBLE_PAGE_TYPES = {
    BlockType.COLOR_ILLUS,
    BlockType.FRONTISPIECE,
    BlockType.ILLUSTRATION,
    BlockType.INSERT,
}


@dataclass(frozen=True, slots=True)
class IllustrationSpreadPair:
    first_block_index: int
    second_block_index: int
    first_page: int
    second_page: int
    confidence: float
    seam_similarity: float
    seam_correlation: float
    seam_ink_ratio: float
    method: str = "inner-edge-continuity-v1"

    def to_dict(self) -> dict:
        return {
            "first_block_index": self.first_block_index,
            "second_block_index": self.second_block_index,
            "first_page": self.first_page,
            "second_page": self.second_page,
            "confidence": round(self.confidence, 4),
            "seam_similarity": round(self.seam_similarity, 4),
            "seam_correlation": round(self.seam_correlation, 4),
            "seam_ink_ratio": round(self.seam_ink_ratio, 4),
            "method": self.method,
        }


def _page_type_map(doc: UnifiedDocument) -> dict[int, BlockType]:
    return {int(page.page_no): page.page_type for page in (doc.pages or [])}


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 4:
        return 0.0
    ml, mr = mean(left), mean(right)
    dl = [value - ml for value in left]
    dr = [value - mr for value in right]
    denom = sqrt(sum(value * value for value in dl) * sum(value * value for value in dr))
    if denom <= 1e-9:
        return 0.0
    return max(-1.0, min(1.0, sum(a * b for a, b in zip(dl, dr)) / denom))


def _edge_profile(path: str, *, side: str, target_height: int = 256) -> tuple[list[tuple[float, float, float]], float, float] | None:
    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
    except Exception:
        return None
    try:
        if image.width < 80 or image.height < 120:
            return None
        scale = target_height / max(1.0, float(image.height))
        target_width = max(40, int(round(image.width * scale)))
        image = image.resize((target_width, target_height), getattr(Image, "Resampling", Image).BILINEAR)
        # Use a narrow inner-edge band but skip the outermost pixel, which can
        # be scanner/crop noise.  The row-wise RGB profile preserves seam art.
        strip_w = max(3, min(10, int(round(image.width * 0.025))))
        if side == "left":
            crop = image.crop((1, 0, 1 + strip_w, image.height))
        else:
            crop = image.crop((image.width - 1 - strip_w, 0, image.width - 1, image.height))
        try:
            px = crop.load()
            profile: list[tuple[float, float, float]] = []
            ink_rows = 0
            for y in range(crop.height):
                values = [px[x, y] for x in range(crop.width)]
                rgb = tuple(sum(value[channel] for value in values) / len(values) for channel in range(3))
                profile.append(rgb)
                # Reject blank paper seams. Saturated or genuinely dark pixels
                # count as artwork/linework evidence.
                r, g, b = rgb
                if min(r, g, b) < 225 or (max(r, g, b) - min(r, g, b)) > 20:
                    ink_rows += 1
            luminance = [0.299 * r + 0.587 * g + 0.114 * b for r, g, b in profile]
            variance = ImageStat.Stat(crop.convert("L")).var[0]
            return profile, ink_rows / max(1, crop.height), float(variance)
        finally:
            crop.close()
    finally:
        image.close()


def _seam_score(first_path: str, second_path: str) -> tuple[float, float, float, float] | None:
    # Japanese RTL reading order: first page is the right-hand page, so its
    # *left* edge meets the second (left-hand) page's *right* edge.
    first = _edge_profile(first_path, side="left")
    second = _edge_profile(second_path, side="right")
    if first is None or second is None:
        return None
    left_profile, left_ink, left_var = first
    right_profile, right_ink, right_var = second
    if len(left_profile) != len(right_profile):
        return None

    mad = mean(
        (abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])) / (3.0 * 255.0)
        for a, b in zip(left_profile, right_profile)
    )
    similarity = max(0.0, 1.0 - mad)
    left_luma = [0.299 * r + 0.587 * g + 0.114 * b for r, g, b in left_profile]
    right_luma = [0.299 * r + 0.587 * g + 0.114 * b for r, g, b in right_profile]
    correlation = max(0.0, _pearson(left_luma, right_luma))
    ink = min(left_ink, right_ink)
    # A white/flat gutter can look perfectly similar but is not evidence that
    # two pages are one artwork.
    texture = min(1.0, min(left_var, right_var) / 120.0)
    confidence = 0.58 * similarity + 0.22 * correlation + 0.12 * min(1.0, ink * 2.2) + 0.08 * texture
    return confidence, similarity, correlation, ink


def _compatible_dimensions(first_path: str, second_path: str) -> bool:
    try:
        with Image.open(first_path) as a, Image.open(second_path) as b:
            if min(a.width, a.height, b.width, b.height) <= 0:
                return False
            ratio_a = a.width / a.height
            ratio_b = b.width / b.height
            if abs(ratio_a - ratio_b) > 0.08:
                return False
            width_ratio = min(a.width, b.width) / max(a.width, b.width)
            height_ratio = min(a.height, b.height) / max(a.height, b.height)
            return width_ratio >= 0.88 and height_ratio >= 0.88
    except Exception:
        return False


def detect_likely_illustration_spreads(
    doc: UnifiedDocument,
    *,
    min_confidence: float = 0.88,
    min_similarity: float = 0.88,
    min_ink_ratio: float = 0.10,
) -> list[IllustrationSpreadPair]:
    """Return only high-confidence adjacent illustration pairs.

    Detection requires consecutive physical pages, no substantive block between
    the image references, compatible dimensions, illustration page types, and
    strong inner-edge continuity.  It is safe to ignore the result entirely.
    """
    page_types = _page_type_map(doc)
    image_entries = [
        (index, block) for index, block in enumerate(doc.blocks or [])
        if block.type == BlockType.IMAGE_REF and block.image_path
    ]
    results: list[IllustrationSpreadPair] = []
    used: set[int] = set()

    for entry_index in range(len(image_entries) - 1):
        first_index, first = image_entries[entry_index]
        second_index, second = image_entries[entry_index + 1]
        if first_index in used or second_index in used:
            continue
        p1, p2 = int(first.page or 0), int(second.page or 0)
        if p1 <= 0 or p2 != p1 + 1:
            continue
        # There must be no real text or another asset between the two image
        # references. Structural empty/consumed placeholders do not count.
        between = doc.blocks[first_index + 1:second_index]
        if any(
            block.type != BlockType.IMAGE_REF
            and not (block.metadata or {}).get("consumed")
            and str(block.text or "").strip()
            for block in between
        ):
            continue

        type1 = page_types.get(p1)
        type2 = page_types.get(p2)
        meta1 = str((first.metadata or {}).get("page_type") or "")
        meta2 = str((second.metadata or {}).get("page_type") or "")
        eligible1 = type1 in _ELIGIBLE_PAGE_TYPES or meta1 in {value.value for value in _ELIGIBLE_PAGE_TYPES}
        eligible2 = type2 in _ELIGIBLE_PAGE_TYPES or meta2 in {value.value for value in _ELIGIBLE_PAGE_TYPES}
        if not (eligible1 and eligible2):
            continue

        if not (Path(first.image_path).exists() and Path(second.image_path).exists()):
            continue
        if not _compatible_dimensions(first.image_path, second.image_path):
            continue
        score = _seam_score(first.image_path, second.image_path)
        if score is None:
            continue
        confidence, similarity, correlation, ink = score
        if confidence < min_confidence or similarity < min_similarity or ink < min_ink_ratio:
            continue
        results.append(IllustrationSpreadPair(
            first_block_index=first_index,
            second_block_index=second_index,
            first_page=p1,
            second_page=p2,
            confidence=confidence,
            seam_similarity=similarity,
            seam_correlation=correlation,
            seam_ink_ratio=ink,
        ))
        used.update({first_index, second_index})
    return results
