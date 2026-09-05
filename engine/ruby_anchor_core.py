#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dependency-free Ruby anchor mapping across authoritative text edits.

Ruby readings are immutable side-channel evidence.  Exact base/context matching
always wins.  When the authoritative OCR/AI text corrects the Ruby *base* itself,
we map the old span into the new text (the same position-map idea used by rich
text editors) and only migrate the reading when the change is independently
supported by findtextCenterNet or an explicitly trusted correction candidate.

The strict path is intentionally asymmetric:

* With locked findtext evidence, a small substitution/insertion/deletion may
  update the base, including a single-character base, but the mapped target must
  equal the base that findtext actually saw for the immutable reading.
* Without locked findtext evidence, only a narrow same-length multi-character
  substitution is eligible and at least one original context side must survive.
* Broad rewrites, ambiguous positions, boundary-crossing length changes and any
  attempt to change the reading fail closed.

This mirrors ProseMirror-style position maps rather than fuzzy global search:
derive a deterministic old->new span first, then validate the semantic evidence.
"""
from __future__ import annotations

import copy
import hashlib
from difflib import SequenceMatcher
from typing import Iterable

ANCHOR_POLICY = "edit-span-findtext-verified-v4"
CONTEXT_CHARS = 18


def _sha256_text(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def occurrences(text: str, needle: str) -> list[int]:
    if not needle:
        return []
    out: list[int] = []
    pos = 0
    while True:
        found = text.find(needle, pos)
        if found < 0:
            return out
        out.append(found)
        pos = found + max(1, len(needle))


def _overlaps_used(pos: int, length: int, used: Iterable[tuple[int, int]]) -> bool:
    end = pos + length
    return any(pos < used_end and end > used_start for used_start, used_end in used)


def resolve_exact(text: str, annotation: dict, used: Iterable[tuple[int, int]] = ()) -> int | None:
    """Resolve an unchanged Ruby base using the conservative legacy policy."""
    base = str(annotation.get("base", "") or "")
    if not base:
        return None
    candidates = [
        pos for pos in occurrences(text, base)
        if not _overlaps_used(pos, len(base), used)
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None

    left = str(annotation.get("left_context", "") or "")
    right = str(annotation.get("right_context", "") or "")
    ranked: list[tuple[int, int]] = []
    for pos in candidates:
        exact = 0
        if left and text[max(0, pos - len(left)):pos] == left:
            exact += 1
        if right and text[pos + len(base):pos + len(base) + len(right)] == right:
            exact += 1
        ranked.append((exact, pos))
    ranked.sort(reverse=True)
    if ranked[0][0] <= 0:
        return None
    if len(ranked) > 1 and ranked[1][0] == ranked[0][0]:
        return None
    return ranked[0][1]


def _source_span(source_text: str, annotation: dict) -> tuple[int, int] | None:
    base = str(annotation.get("base", "") or "")
    if not base or not source_text:
        return None
    try:
        start = int(annotation.get("source_offset"))
    except (TypeError, ValueError, OverflowError):
        start = -1
    if start >= 0 and source_text[start:start + len(base)] == base:
        return start, start + len(base)
    pos = resolve_exact(source_text, annotation, ())
    if pos is None:
        return None
    return pos, pos + len(base)


def _map_boundary(
    opcodes: list[tuple[str, int, int, int, int]], pos: int, *, assoc: int,
) -> int | None:
    """Map an old-text boundary into the new text.

    ``assoc`` follows editor position-map semantics: -1 associates a boundary
    with content on its left, +1 with content on its right.  This matters for an
    insertion exactly at a boundary.  Positions inside equal-length replacement
    chunks map one-to-one; positions inside unequal replacements are ambiguous.
    """
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "insert":
            if pos == i1:
                return j1 if assoc < 0 else j2
            continue
        if pos < i1 or pos > i2:
            continue
        if tag == "equal":
            return j1 + (pos - i1)
        if tag == "replace":
            if pos == i1:
                return j1
            if pos == i2:
                return j2
            if (i2 - i1) == (j2 - j1):
                return j1 + (pos - i1)
            return None
        if tag == "delete" and pos in (i1, i2):
            return j1
    if opcodes and pos == opcodes[-1][2]:
        return opcodes[-1][4]
    return None


def _exact_context_sides(current: str, pos: int, candidate: str, annotation: dict) -> int:
    left = str(annotation.get("left_context", "") or "")
    right = str(annotation.get("right_context", "") or "")
    exact = 0
    if left and current[max(0, pos - len(left)):pos] == left:
        exact += 1
    if right and current[pos + len(candidate):pos + len(candidate) + len(right)] == right:
        exact += 1
    return exact


def _levenshtein(a: str, b: str, *, limit: int = 3) -> int:
    """Small bounded edit distance; values above ``limit`` collapse to limit+1."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        row_min = i
        for j, cb in enumerate(b, start=1):
            value = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ca != cb),
            )
            current.append(value)
            row_min = min(row_min, value)
        if row_min > limit:
            return limit + 1
        previous = current
    return previous[-1]


def _safe_legacy_base_correction(old_base: str, new_base: str) -> tuple[bool, list[int]]:
    """Legacy fallback: narrow same-length substitution with lexical continuity."""
    if not old_base or not new_base or old_base == new_base:
        return False, []
    if len(old_base) != len(new_base) or len(old_base) < 2:
        return False, []
    if any(ch.isspace() for ch in new_base):
        return False, []
    changed = [i for i, (old, new) in enumerate(zip(old_base, new_base)) if old != new]
    max_changes = 1 if len(old_base) <= 6 else min(2, max(1, len(old_base) // 4))
    if not changed or len(changed) > max_changes:
        return False, changed
    if len(old_base) - len(changed) < 1:
        return False, changed
    return True, changed


def _safe_line_verified_correction(old_base: str, new_base: str) -> tuple[bool, int]:
    """Bounded edit accepted only inside an already matched findtext line/ROI.

    This is used before the authoritative annotation has stored
    ``findtext_detected_base``.  The caller has already established a strong
    text/geometry match, so one insertion/deletion/substitution in a
    multi-character base may be wrapped without changing prose.
    """
    if not old_base or not new_base or old_base == new_base:
        return False, 0
    if min(len(old_base), len(new_base)) < 2:
        return False, 0
    if any(ch.isspace() or ch in "\r\n" for ch in new_base):
        return False, 0
    max_len = max(len(old_base), len(new_base))
    max_edits = 1 if max_len <= 6 else min(2, max(1, max_len // 4))
    distance = _levenshtein(old_base, new_base, limit=max_edits)
    return (0 < distance <= max_edits), distance


def _findtext_verified_correction(old_base: str, new_base: str, annotation: dict) -> tuple[bool, int]:
    """Accept a small base edit only when it exactly matches locked findtext evidence."""
    detected = str(annotation.get("findtext_detected_base", "") or "")
    detected_reading = str(annotation.get("findtext_detected_reading", "") or "")
    reading = str(annotation.get("reading", "") or "")
    if not detected or new_base != detected:
        return False, 0
    if detected_reading and detected_reading != reading:
        return False, 0
    if not old_base or not new_base or old_base == new_base:
        return False, 0
    if any(ch.isspace() or ch in "\r\n" for ch in new_base):
        return False, 0
    # Short Ruby bases may have one OCR insertion/deletion/substitution.  Longer
    # compounds get at most two edits, still capped at 25% of the larger base.
    max_len = max(len(old_base), len(new_base))
    max_edits = 1 if max_len <= 6 else min(2, max(1, max_len // 4))
    distance = _levenshtein(old_base, new_base, limit=max_edits)
    return (0 < distance <= max_edits), distance


def _changes_touching_span(
    opcodes: list[tuple[str, int, int, int, int]], start: int, end: int,
) -> tuple[bool, bool, list[dict]]:
    """Describe base-touching edits and flag unsafe boundary-crossing length changes."""
    touched = False
    unsafe = False
    edits: list[dict] = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        if tag == "insert":
            # Insertions strictly inside the base are attributable to it.  An
            # insertion exactly at start/end could belong to neighbouring prose.
            if start < i1 < end:
                touched = True
                edits.append({"tag": tag, "old": [i1, i2], "new": [j1, j2]})
            continue
        overlaps = not (i2 <= start or i1 >= end)
        if not overlaps:
            continue
        touched = True
        edits.append({"tag": tag, "old": [i1, i2], "new": [j1, j2]})
        crosses = i1 < start or i2 > end
        if crosses and (i2 - i1) != (j2 - j1):
            unsafe = True
    return touched, unsafe, edits


def _mapped_candidate_has_unique_context(
    source_text: str, current: str, start: int, end: int, new_start: int, new_end: int, candidate: str,
) -> bool:
    """Require a unique local anchor when the migrated base occurs repeatedly.

    A final-text diff is weaker evidence than an editor's recorded transaction
    map.  If the findtext-verified target spelling appears more than once, keep
    the migration only when unchanged text immediately around the mapped span
    forms a unique local anchor in the new paragraph.
    """
    if len(occurrences(current, candidate)) <= 1:
        return True

    max_context = 12
    left_source = source_text[max(0, start - max_context):start]
    right_source = source_text[end:end + max_context]
    left_current = current[max(0, new_start - max_context):new_start]
    right_current = current[new_end:new_end + max_context]

    left_shared = 0
    for old, new in zip(reversed(left_source), reversed(left_current)):
        if old != new:
            break
        left_shared += 1
    right_shared = 0
    for old, new in zip(right_source, right_current):
        if old != new:
            break
        right_shared += 1

    # No surviving neighbour means a repeated target is not safely identifiable.
    if left_shared <= 0 and right_shared <= 0:
        return False
    left_anchor = left_current[len(left_current) - min(left_shared, 8):] if left_shared else ""
    right_anchor = right_current[:min(right_shared, 8)] if right_shared else ""
    anchor = left_anchor + candidate + right_anchor
    return bool(anchor) and current.count(anchor) == 1



def reanchor_unchanged_base_by_edit_span(
    source_text: str,
    current: str,
    annotation: dict,
    used: Iterable[tuple[int, int]] = (),
) -> dict | None:
    """Track one *unchanged* Ruby base by its old source span across prose edits.

    This is deliberately stricter than :func:`resolve_exact`.  A repeated base
    is not allowed to become "safe" merely because another occurrence was
    deleted and the current text now contains the spelling only once.  The old
    source span must live inside an equal diff block, and the equal neighbour
    text around that span must identify the same occurrence uniquely in both
    the old and the new paragraph.

    The function never changes ``base`` or ``reading``.  It only refreshes the
    positional/context anchor after a proven identity-preserving edit map.
    """
    source_text = str(source_text or "")
    current = str(current or "")
    base = str(annotation.get("base", "") or "")
    reading = str(annotation.get("reading", "") or "")
    if not source_text or not current or not base or not reading:
        return None
    span = _source_span(source_text, annotation)
    if span is None:
        return None
    start, end = span
    if source_text[start:end] != base:
        return None

    # If the text is byte-for-byte unchanged, the locked source offset itself is
    # the strongest possible identity.  This also makes repeated homographs with
    # different readings deterministic without any global occurrence guessing.
    if current == source_text:
        if _overlaps_used(start, len(base), used):
            return None
        anchored = copy.deepcopy(annotation)
        anchored.update({
            "source_offset": start,
            "source_occurrence": current[:start].count(base),
            "left_context": current[max(0, start - CONTEXT_CHARS):start],
            "right_context": current[end:end + CONTEXT_CHARS],
            "anchor_version": max(4, int(anchored.get("anchor_version", 1) or 1)),
        })
        return {
            "position": start,
            "annotation": anchored,
            "mode": "exact_source_span",
            "changed_positions": [],
        }

    opcodes = SequenceMatcher(None, source_text, current, autojunk=False).get_opcodes()
    equal = None
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal" and i1 <= start and end <= i2:
            equal = (i1, i2, j1, j2)
            break
    if equal is None:
        return None
    i1, i2, j1, j2 = equal
    new_start = j1 + (start - i1)
    new_end = new_start + len(base)
    if current[new_start:new_end] != base:
        return None
    if _overlaps_used(new_start, len(base), used):
        return None

    source_count = len(occurrences(source_text, base))
    current_count = len(occurrences(current, base))
    repeated_identity = source_count > 1 or current_count > 1
    if repeated_identity:
        # Use only characters that SequenceMatcher proved equal *contiguously*
        # with this source span.  A base-only equal block is not enough to tell
        # two homographs apart.  The largest local anchor available is required
        # to be unique on both sides of the edit.
        left_len = min(12, max(0, start - i1))
        right_len = min(12, max(0, i2 - end))
        if left_len + right_len <= 0:
            return None
        old_anchor_start = start - left_len
        old_anchor_end = end + right_len
        new_anchor_start = new_start - left_len
        new_anchor_end = new_end + right_len
        old_anchor = source_text[old_anchor_start:old_anchor_end]
        new_anchor = current[new_anchor_start:new_anchor_end]
        if not old_anchor or old_anchor != new_anchor:
            return None
        if source_text.count(old_anchor) != 1 or current.count(new_anchor) != 1:
            return None

    anchored = copy.deepcopy(annotation)
    history = list(anchored.get("position_map_history") or [])
    history.append({
        "policy": ANCHOR_POLICY,
        "source_span": [start, end],
        "target_span": [new_start, new_end],
        "source_occurrences": source_count,
        "target_occurrences": current_count,
        "source_text_sha256": _sha256_text(source_text),
        "target_text_sha256": _sha256_text(current),
    })
    anchored.update({
        "source_offset": new_start,
        "source_occurrence": current[:new_start].count(base),
        "left_context": current[max(0, new_start - CONTEXT_CHARS):new_start],
        "right_context": current[new_end:new_end + CONTEXT_CHARS],
        "position_reanchored": True,
        "position_reanchor_policy": ANCHOR_POLICY,
        "position_map_history": history,
        "anchor_version": max(4, int(anchored.get("anchor_version", 1) or 1)),
    })
    return {
        "position": new_start,
        "annotation": anchored,
        "mode": "exact_edit_span",
        "changed_positions": [],
    }

def migrate_base_by_edit_span(
    source_text: str,
    current: str,
    annotation: dict,
    used: Iterable[tuple[int, int]] = (),
    *,
    require_base_evidence: bool = True,
) -> dict | None:
    """Map one Ruby anchor through a narrowly-scoped authoritative base edit."""
    source_text = str(source_text or "")
    current = str(current or "")
    base = str(annotation.get("base", "") or "")
    reading = str(annotation.get("reading", "") or "")
    if not source_text or not current or not base or not reading:
        return None
    span = _source_span(source_text, annotation)
    if span is None:
        return None
    start, end = span
    if source_text[start:end] != base:
        return None

    opcodes = SequenceMatcher(None, source_text, current, autojunk=False).get_opcodes()
    touched_change, unsafe_boundary_change, edit_ops = _changes_touching_span(opcodes, start, end)
    if not touched_change or unsafe_boundary_change:
        return None

    new_start = _map_boundary(opcodes, start, assoc=1)
    new_end = _map_boundary(opcodes, end, assoc=-1)
    if new_start is None or new_end is None or new_end <= new_start:
        return None
    candidate = current[new_start:new_end]
    if _overlaps_used(new_start, len(candidate), used):
        return None

    strong_findtext_evidence = False
    edit_distance = 0
    changed_positions: list[int] = []
    if require_base_evidence:
        detected = str(annotation.get("findtext_detected_base", "") or "")
        if detected:
            strong_findtext_evidence, edit_distance = _findtext_verified_correction(base, candidate, annotation)
            # A present findtext base is authoritative evidence.  If its base or
            # immutable reading disagrees with the mapped candidate, do not fall
            # through to weaker legacy candidates.
            if not strong_findtext_evidence:
                return None
        else:
            trusted = {
                str(value) for value in (annotation.get("base_correction_candidates") or [])
                if str(value)
            }
            if candidate not in trusted:
                return None
            safe, changed_positions = _safe_legacy_base_correction(base, candidate)
            if not safe:
                return None
            edit_distance = len(changed_positions)
    else:
        # Initial findtext injection already has a strong line/geometry match but
        # has not yet stored its detected base into the authoritative annotation.
        # It may therefore bridge one missing/extra OCR glyph in a multi-character
        # base while still wrapping (never replacing) authoritative prose.
        safe, edit_distance = _safe_line_verified_correction(base, candidate)
        if not safe:
            return None
        if len(base) == len(candidate):
            changed_positions = [i for i, (old, new) in enumerate(zip(base, candidate)) if old != new]

    if not changed_positions and len(base) == len(candidate):
        changed_positions = [i for i, (old, new) in enumerate(zip(base, candidate)) if old != new]

    if strong_findtext_evidence and not _mapped_candidate_has_unique_context(
        source_text, current, start, end, new_start, new_end, candidate,
    ):
        return None

    # Legacy/trusted-candidate migration still needs textual continuity.  A
    # findtext-verified target can survive nearby AI prose edits because the
    # deterministic span map + immutable reading provides independent evidence.
    if not strong_findtext_evidence and _exact_context_sides(current, new_start, candidate, annotation) <= 0:
        return None

    migrated = copy.deepcopy(annotation)
    original = str(migrated.get("base_original", "") or base)
    history = list(migrated.get("base_edit_history") or [])
    edit_kind = "substitution" if len(base) == len(candidate) else ("insertion" if len(candidate) > len(base) else "deletion")
    history.append({
        "policy": ANCHOR_POLICY,
        "old_base": base,
        "new_base": candidate,
        "edit_kind": edit_kind,
        "edit_distance": int(edit_distance),
        "changed_positions": changed_positions,
        "source_span": [start, end],
        "target_span": [new_start, new_end],
        "edit_ops": edit_ops,
        "findtext_verified": bool(strong_findtext_evidence),
        "source_text_sha256": _sha256_text(source_text),
        "target_text_sha256": _sha256_text(current),
    })
    migrated.update({
        "base": candidate,
        "base_original": original,
        "base_previous": base,
        "base_migrated": True,
        "base_migration_policy": ANCHOR_POLICY,
        "base_migration_kind": edit_kind,
        "base_migration_distance": int(edit_distance),
        "base_edit_history": history,
        "source_offset": new_start,
        "source_occurrence": current[:new_start].count(candidate),
        "left_context": current[max(0, new_start - CONTEXT_CHARS):new_start],
        "right_context": current[new_end:new_end + CONTEXT_CHARS],
        "anchor_version": max(4, int(migrated.get("anchor_version", 1) or 1)),
    })
    return {
        "position": new_start,
        "annotation": migrated,
        "mode": "base_edit_span",
        "changed_positions": changed_positions,
        "edit_kind": edit_kind,
        "edit_distance": int(edit_distance),
    }


def resolve_annotation_for_text(
    current: str,
    annotation: dict,
    used: Iterable[tuple[int, int]] = (),
    *,
    source_text: str = "",
    require_base_evidence: bool = True,
) -> dict | None:
    """Resolve one locked Ruby anchor against the current authoritative prose.

    Exact matching normally wins.  One exception is important for OCR repair:
    when findtextCenterNet explicitly recorded a *different* base for the same
    immutable reading, try the source->current edit-span mapping first.  This
    prevents an unrelated surviving copy of the old OCR typo elsewhere in the
    paragraph from stealing the reading after the original span was corrected.
    """
    base = str(annotation.get("base", "") or "")
    detected = str(annotation.get("findtext_detected_base", "") or "")
    prefer_verified_migration = bool(
        source_text and require_base_evidence and detected and detected != base
    )
    if prefer_verified_migration:
        migrated = migrate_base_by_edit_span(
            source_text, current, annotation, used,
            require_base_evidence=True,
        )
        if migrated is not None:
            return migrated

    # Repeated homographs need source-span identity, not "there is only one left
    # now" reasoning.  The same rule also protects a formerly unique Ruby base
    # when AI duplicates the surrounding phrase.
    source_repeats = bool(source_text and len(occurrences(source_text, base)) > 1)
    current_repeats = bool(len(occurrences(current, base)) > 1)
    source_identity_required = bool(source_text and (source_repeats or current_repeats))
    if source_identity_required or (source_text and current == source_text):
        anchored = reanchor_unchanged_base_by_edit_span(
            source_text, current, annotation, used,
        )
        if anchored is not None:
            return anchored
        if source_identity_required:
            return None

    pos = resolve_exact(current, annotation, used)
    if pos is not None:
        return {
            "position": pos,
            "annotation": copy.deepcopy(annotation),
            "mode": "exact",
            "changed_positions": [],
        }
    if prefer_verified_migration:
        return None
    return migrate_base_by_edit_span(
        source_text, current, annotation, used,
        require_base_evidence=require_base_evidence,
    )


__all__ = [
    "ANCHOR_POLICY",
    "occurrences",
    "resolve_exact",
    "reanchor_unchanged_base_by_edit_span",
    "migrate_base_by_edit_span",
    "resolve_annotation_for_text",
]
