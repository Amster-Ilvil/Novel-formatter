# -*- coding: utf-8 -*-
"""V5.5: evidence-bound, conservative glyph clustering and OCR slot reuse.

Offline. Does not load OCR weights, rewrite model output, or approve decisions.
Callers supply the true physical-column glyph crop coordinates (or use the
optional geometric alignment helper) and the immutable V5 OCR candidates.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "novel_formatter.glyph_batch.v5_5"
# Visually similar or occasionally identical print glyphs are NOT evidence of
# a Unicode code point. Require a separately approved editorial convention.
CODEPOINT_AMBIGUITIES = (
    frozenset(("ニ", "二")),
    frozenset(("一", "ー")),
    frozenset(("力", "カ")),
    frozenset(("口", "ロ")),
    frozenset(("工", "エ")),
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def candidate_digest(texts: Sequence[str]) -> str:
    """Bound a slot decision to the exact ordered raw OCR candidates."""
    payload = json.dumps(list(texts), ensure_ascii=False, separators=(",", ":"))
    return sha256_bytes(payload.encode("utf-8"))


def _box(value: Sequence[int], width: int, height: int) -> tuple[int, int, int, int]:
    if len(value) != 4:
        raise ValueError("bounding box must be x0,y0,x1,y1 (absolute pixels)")
    x0, y0, x1, y1 = (int(v) for v in value)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("glyph bounding box lies outside the ORIGINAL image")
    if x1 - x0 < 4 or y1 - y0 < 4:
        raise ValueError("glyph bounding box too small")
    return x0, y0, x1, y1


@dataclass(frozen=True)
class GlyphProof:
    task_id: str
    page_id: str
    column_id: str
    source_sha256: str
    box: tuple[int, int, int, int]
    pixels_sha256: str
    size: tuple[int, int]
    ink_ratio: float
    center_ink_ratio: float
    position_verified: bool = False

    @property
    def physical_key(self) -> tuple[str, tuple[int, int, int, int]]:
        # Labels alone never make reused image bytes independent physical evidence.
        return (self.source_sha256, self.box)

    @property
    def glyph_key(self) -> str:
        return f"{self.size[0]}x{self.size[1]}:{self.pixels_sha256}"


def make_glyph_proof(
    image_path: str | Path,
    *,
    task_id: str,
    page_id: str,
    column_id: str,
    box: Sequence[int],
    ink_threshold: int = 205,
    position_verified: bool = False,
) -> GlyphProof:
    """Hash the actual full source image AND the exact unresized crop pixels.

    No cached screenshot, interpolation, blank-crop or arbitrary rescale may
    masquerade as independently positioned original-glyph evidence.
    """
    from PIL import Image

    path = Path(image_path)
    source_sha = sha256_bytes(path.read_bytes())
    with Image.open(path) as source:
        source.load()
        bounds = _box(box, source.width, source.height)
        crop = source.crop(bounds).convert("L")
    width, height = crop.size
    data = crop.tobytes()
    ink = sum(pixel < ink_threshold for pixel in data)
    ratio = ink / max(1, len(data))
    if ink < 10 or not (0.012 <= ratio <= 0.75):
        raise ValueError("blank, near-blank, or saturated glyph crop")
    # A plausible glyph has ink in its central area; a border/grid line does
    # not qualify for batch propagation merely because it repeats exactly.
    cx0, cx1 = width // 4, max(width // 4 + 1, (width * 3) // 4)
    cy0, cy1 = height // 4, max(height // 4 + 1, (height * 3) // 4)
    center_count = sum(
        data[y * width + x] < ink_threshold
        for y in range(cy0, cy1)
        for x in range(cx0, cx1)
    )
    center_ratio = center_count / max(1, (cx1 - cx0) * (cy1 - cy0))
    if center_ratio < 0.008:
        raise ValueError("glyph crop has no central ink")
    return GlyphProof(
        task_id=str(task_id), page_id=str(page_id),
        column_id=str(column_id), source_sha256=source_sha,
        box=bounds, pixels_sha256=sha256_bytes(data),
        size=(width, height), ink_ratio=round(ratio, 6),
        center_ink_ratio=round(center_ratio, 6),
        position_verified=bool(position_verified),
    )


def exact_glyph_groups(
    proofs: Sequence[GlyphProof], *, minimum_distinct_positions: int = 2
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate true physical repeats from repeated references to one crop.

    Exact means equal original grayscale bytes AND equal crop dimensions.
    A hash collision is not a substitute for independently bound source
    provenance. Blank or invalid crops are rejected at proof construction.
    """
    bins: dict[str, dict[tuple, GlyphProof]] = defaultdict(dict)
    duplicates: list[dict[str, Any]] = []
    for proof in proofs:
        key = proof.glyph_key
        physical = proof.physical_key
        if physical in bins[key]:
            duplicates.append({
                "task_id": proof.task_id, "glyph_key": key,
                "reason": "same_physical_crop_referenced_more_than_once",
            })
            continue
        bins[key][physical] = proof
    groups: list[dict[str, Any]] = []
    for key, positions in sorted(bins.items()):
        if len(positions) < minimum_distinct_positions:
            continue
        members = sorted(positions.values(), key=lambda p: (p.page_id, p.column_id, p.box, p.task_id))
        groups.append({
            "schema": SCHEMA,
            "glyph_key": key,
            "member_count": len(members),
            "members": [asdict(p) for p in members],
            "possible_review_savings": len(members) - 1,
            "approved": False,
        })
    return groups, duplicates


def _is_ambiguous_codepoint(a: str, b: str) -> bool:
    return any(a in group and b in group for group in CODEPOINT_AMBIGUITIES)


def make_substitution_slots(task_id: str, texts: Sequence[str]) -> list[dict[str, Any]]:
    """Extract ALL one-to-one substitutions, even within composite conflicts.

    Insertions, deletions, and length-changing replacements stay unresolved.
    No fuzzy matching or substring offsets are allowed for write-back.
    """
    if len(texts) < 2:
        return []
    a, b = str(texts[0]), str(texts[1])
    fingerprint = candidate_digest(texts)
    matcher = SequenceMatcher(None, a, b, autojunk=False)
    slots: list[dict[str, Any]] = []
    for tag, a0, a1, b0, b1 in matcher.get_opcodes():
        if tag != "replace" or a1 - a0 != b1 - b0:
            continue
        for off in range(a1 - a0):
            i, j = a0 + off, b0 + off
            slots.append({
                "task_id": task_id,
                "a_index": i,
                "b_index": j,
                "a": a[i],
                "b": b[j],
                "candidate_sha256": fingerprint,
                "status": "unreviewed",
            })
    return slots


def map_column_characters(
    text: str,
    *,
    cell_count: int,
    verified_anchors: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    """Conservative monotonically anchored char-index -> physical-cell mapping.

    Anchors refer to VERIFIED char positions and physical cells, not OCR guesses.
    When counts agree, an exact one-to-one map is returned. For unequal counts,
    every interval between anchors must have equal cardinality: never silently
    slide characters to the next printed glyph after a missing/ruby character.
    """
    anchors = {int(k): int(v) for k, v in (verified_anchors or {}).items()}
    if cell_count <= 0 or not text:
        return {"ok": False, "reason": "empty_cells_or_text", "mapping": {}}
    if any(i < 0 or i >= len(text) or c < 0 or c >= cell_count for i, c in anchors.items()):
        return {"ok": False, "reason": "anchor_out_of_range", "mapping": {}}
    ordered = sorted(anchors.items())
    if any(ordered[n][1] >= ordered[n + 1][1] for n in range(len(ordered) - 1)):
        return {"ok": False, "reason": "non_monotonic_anchors", "mapping": {}}
    if not anchors and len(text) == cell_count:
        return {"ok": True, "reason": "length_equal_unverified", "mapping": {i: i for i in range(cell_count)}, "requires_visual_check": True}
    if not anchors:
        return {"ok": False, "reason": "unequal_length_no_anchors", "mapping": {}}
    boundaries = [(-1, -1), *ordered, (len(text), cell_count)]
    mapping = dict(ordered)
    ambiguous: list[tuple[int, int]] = []
    for (ai, ac), (bi, bc) in zip(boundaries, boundaries[1:]):
        left_chars = bi - ai - 1
        left_cells = bc - ac - 1
        if left_chars != left_cells:
            ambiguous.append((ai + 1, bi))
            continue
        for delta in range(1, left_chars + 1):
            mapping[ai + delta] = ac + delta
    return {
        "ok": bool(mapping),
        "reason": "anchored_partial" if ambiguous else "anchored_complete",
        "mapping": mapping,
        "ambiguous_intervals": ambiguous,
        "requires_visual_check": bool(ambiguous),
    }


def _review_choice(
    decision: Mapping[str, Any],
    *,
    slot: Mapping[str, Any],
    proof: GlyphProof,
) -> tuple[str | None, str]:
    """One approved group decision never overrides conflicting local context."""
    if decision.get("approved") is not True:
        return None, "not_approved"
    if not proof.position_verified:
        return None, "physical_position_not_verified"
    if decision.get("glyph_key") != proof.glyph_key:
        return None, "glyph_proof_changed"
    if slot.get("task_id") != proof.task_id:
        return None, "task_proof_mismatch"
    if slot.get("candidate_sha256") != decision.get("candidate_sha256_by_task", {}).get(proof.task_id):
        return None, "ocr_candidates_changed"
    binding = decision.get("proof_by_task", {}).get(proof.task_id)
    if not isinstance(binding, dict):
        return None, "no_physical_provenance_binding"
    if (binding.get("source_sha256") != proof.source_sha256
        or binding.get("page_id") != proof.page_id
        or binding.get("column_id") != proof.column_id
        or tuple(binding.get("box", ())) != proof.box):
        return None, "physical_provenance_changed"
    if proof.task_id in decision.get("exceptions", ()):
        return None, "exception"
    choice = decision.get("character")
    if not isinstance(choice, str) or len(choice) != 1 or choice not in (slot.get("a"), slot.get("b")):
        return None, "not_an_existing_single_character_candidate"
    if _is_ambiguous_codepoint(str(slot["a"]), str(slot["b"])):
        if decision.get("codepoint_policy") != "explicit_book_editorial":
            return None, "same_shape_different_unicode_requires_editorial_policy"
    if len(str(decision.get("review_note", "")).strip()) < 12:
        return None, "review_note_too_short"
    if proof.physical_key in [
        (str(x["source_sha256"]), str(x["page_id"]), str(x["column_id"]), tuple(x["box"]))
        for x in decision.get("excluded_physical_positions", ())
    ]:
        return None, "physical_position_excluded"
    # Review scope can be restricted to explicitly checked task IDs.
    allowed = decision.get("allowed_task_ids")
    if allowed is None and decision.get("all_contexts_reviewed") is not True:
        return None, "group_context_review_missing"
    if allowed is not None and proof.task_id not in allowed:
        return None, "context_not_reviewed"
    return choice, "approved_exact_glyph"


def propose_slot(
    slot: Mapping[str, Any], proof: GlyphProof, decision: Mapping[str, Any]
) -> dict[str, Any]:
    choice, reason = _review_choice(decision, slot=slot, proof=proof)
    return {
        "task_id": slot["task_id"],
        "a_index": slot["a_index"],
        "b_index": slot["b_index"],
        "candidate_sha256": slot["candidate_sha256"],
        "character": choice,
        "status": "suggested" if choice is not None else "needs_review",
        "reason": reason,
        "glyph_key": proof.glyph_key,
    }


def compose_complete_sentence(
    task_id: str, texts: Sequence[str], suggestions: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Compose a full answer only if EVERY differing edit is approved.

    Ambiguous inserts/deletes, character substitutions without a suggestion,
    and stale ordered OCR source hashes MUST produce no final sentence.
    """
    if len(texts) < 2:
        return {"task_id": task_id, "complete": False, "reason": "need_two_raw_candidates"}
    a, b = str(texts[0]), str(texts[1])
    if a == b:
        return {"task_id": task_id, "complete": False, "reason": "no_editable_conflict_in_first_two_models"}
    if any(str(t) not in (a, b) for t in texts[2:]):
        return {"task_id": task_id, "complete": False, "reason": "independent_third_candidate_needs_its_own_alignment"}
    fingerprint = candidate_digest(texts)
    by_position: dict[tuple[int, int], Mapping[str, Any]] = {}
    for item in suggestions:
        if item.get("task_id") != task_id:
            continue
        try:
            key = (int(item["a_index"]), int(item["b_index"]))
        except (KeyError, ValueError, TypeError):
            continue
        if key in by_position:
            return {"task_id": task_id, "complete": False, "reason": "duplicate_slot_decision"}
        by_position[key] = item
    result = list(a)
    missing: list[dict[str, Any]] = []
    for tag, a0, a1, b0, b1 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if tag != "replace" or a1 - a0 != b1 - b0:
            missing.append({"span_a": [a0, a1], "span_b": [b0, b1], "reason": "insert_delete_or_nonparallel_replace"})
            continue
        for delta in range(a1 - a0):
            i, j = a0 + delta, b0 + delta
            item = by_position.get((i, j))
            char = item.get("character") if item else None
            if not item or item.get("status") != "suggested" or item.get("candidate_sha256") != fingerprint or char not in (a[i], b[j]):
                missing.append({"span_a": [i, i + 1], "span_b": [j, j + 1], "reason": "unapproved_or_stale_slot"})
                continue
            result[i] = char
    if missing:
        return {"task_id": task_id, "complete": False, "missing": missing, "resolved_slot_count": sum(x.get("status") == "suggested" for x in by_position.values())}
    return {
        "task_id": task_id,
        "complete": True,
        "selected_text": "".join(result),
        "source_candidate_sha256": fingerprint,
        "status": "draft_only_not_written_back",
    }


def summarize_batch(proofs: Sequence[GlyphProof], groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    unique = {p.physical_key for p in proofs}
    grouped = {(
        str(p["source_sha256"]), tuple(p["box"])
    ) for group in groups for p in group["members"]}
    return {
        "schema": SCHEMA,
        "unique_physical_positions": len(unique),
        "repeated_physical_positions": len(grouped),
        "exact_repeat_group_count": len(groups),
        "potential_repeated_visual_checks_avoided": sum(max(0, int(g["member_count"]) - 1) for g in groups),
        "approved_decision_count": 0,
        "note": "Potential savings are not adjudication completions. No automatic OCR writes.",
    }
