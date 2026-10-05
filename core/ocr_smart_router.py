from __future__ import annotations

"""Pure routing helpers for role-based multi OCR.

The router never changes OCR text or consensus semantics.  It only decides
whether an expensive sentence-role pass is necessary and, when possible,
restricts that pass to the physical columns belonging to still-conflicting
sentences.
"""

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class SentenceRoutePlan:
    mode: str  # skip | full | selective
    reason: str
    target_row_count: int = 0
    target_column_ids: tuple[str, ...] = ()
    # Exact physical-column membership for every routed sentence.  Passing
    # only the union of column IDs forces the recognizer to guess sentence
    # boundaries again from seed OCR tails, which can silently skip genuine
    # conflicts.  Keep the bootstrap comparison's sentence geometry intact.
    target_groups: tuple[tuple[str, ...], ...] = ()

    @property
    def should_run(self) -> bool:
        return self.mode in {"full", "selective"}


def _column_ids(rows: Iterable[object]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for row in rows:
        for value in tuple(getattr(row, "column_ids", ()) or ()):
            column_id = str(value or "")
            if column_id and column_id not in seen:
                seen.add(column_id)
                ordered.append(column_id)
    return tuple(ordered)




def _column_groups(rows: Iterable[object]) -> tuple[tuple[str, ...], ...]:
    seen: set[tuple[str, ...]] = set()
    ordered: list[tuple[str, ...]] = []
    for row in rows:
        group = tuple(
            str(value or "") for value in tuple(getattr(row, "column_ids", ()) or ())
            if str(value or "")
        )
        if group and group not in seen:
            seen.add(group)
            ordered.append(group)
    return tuple(ordered)

def constrain_sentence_groups_for_engine(
    engine_id: str,
    groups: Iterable[Sequence[str]],
) -> tuple[tuple[tuple[str, ...], ...], int]:
    """Normalize routed sentence groups; no recognizer-specific pruning remains."""
    del engine_id
    normalized: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for raw in groups or ():
        group = tuple(str(value or "") for value in tuple(raw or ()) if str(value or ""))
        if group and group not in seen:
            seen.add(group)
            normalized.append(group)
    return tuple(normalized), 0

def plan_sentence_role(
    comparison,
    *,
    bootstrap_document_count: int,
    smart_enabled: bool = True,
) -> SentenceRoutePlan:
    """Plan the sentence main-role pass after page/column bootstrap.

    Safety rules:
    * no smart routing -> preserve the historical full sentence-role pass;
    * fewer than two independent bootstrap documents -> full sentence pass,
      because one OCR source cannot establish agreement/disagreement;
    * two or more independent bootstrap documents with no conflict -> skip;
    * otherwise OCR only sentences that still disagree.
    """
    if not smart_enabled:
        return SentenceRoutePlan("full", "smart_router_disabled")
    if int(bootstrap_document_count or 0) < 2:
        return SentenceRoutePlan("full", "insufficient_independent_bootstrap_evidence")
    rows: Sequence[object] = tuple(getattr(comparison, "rows", ()) or ())
    conflicts = tuple(row for row in rows if bool(getattr(row, "is_conflict", False)))
    if not conflicts:
        return SentenceRoutePlan("skip", "bootstrap_models_agree")
    columns = _column_ids(conflicts)
    groups = _column_groups(conflicts)
    if not columns or len(groups) != len(conflicts):
        # If row identity cannot be mapped back to physical columns, falling
        # back to a full sentence pass is safer than silently skipping evidence.
        return SentenceRoutePlan(
            "full",
            "conflicts_without_physical_column_ids",
            target_row_count=len(conflicts),
        )
    return SentenceRoutePlan(
        "selective",
        "bootstrap_conflicts_only",
        target_row_count=len(conflicts),
        target_column_ids=columns,
        target_groups=groups,
    )
