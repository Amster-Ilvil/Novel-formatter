from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class AdjudicationRowDelta:
    """One stable-row before/after delta in the adjudication history.

    Payloads are immutable JSON strings owned by the UI adapter.  Keeping the
    core history unaware of FusionDecisionState makes it reusable and prevents
    Qt/engine objects from leaking into the pure core layer.
    """

    row_key: str
    row_index: int
    before_payload: str
    after_payload: str
    before_token: str
    after_token: str


@dataclass(frozen=True, slots=True)
class AdjudicationHistoryEntry:
    operation: str
    label: str
    deltas: tuple[AdjudicationRowDelta, ...]

    @property
    def count(self) -> int:
        return len(self.deltas)


class AdjudicationDeltaHistory:
    """Bounded session history with optimistic per-row conflict guards.

    Undo is permitted only while every row touched by the newest entry still
    matches that entry's *after* token.  Redo similarly requires the *before*
    token.  Unrelated rows may change without invalidating the entry.
    """

    def __init__(self, max_entries: int = 100):
        self.max_entries = max(1, int(max_entries or 1))
        self._entries: list[AdjudicationHistoryEntry] = []
        self._cursor = 0

    def clear(self) -> None:
        self._entries.clear()
        self._cursor = 0

    @property
    def can_undo(self) -> bool:
        return self._cursor > 0

    @property
    def can_redo(self) -> bool:
        return self._cursor < len(self._entries)

    @property
    def undo_label(self) -> str:
        return self._entries[self._cursor - 1].label if self.can_undo else ""

    @property
    def redo_label(self) -> str:
        return self._entries[self._cursor].label if self.can_redo else ""

    @property
    def next_undo_entry(self) -> AdjudicationHistoryEntry | None:
        return self._entries[self._cursor - 1] if self.can_undo else None

    @property
    def next_redo_entry(self) -> AdjudicationHistoryEntry | None:
        return self._entries[self._cursor] if self.can_redo else None

    def record(self, entry: AdjudicationHistoryEntry) -> None:
        if not entry.deltas:
            return
        if self._cursor < len(self._entries):
            del self._entries[self._cursor :]
        self._entries.append(entry)
        if len(self._entries) > self.max_entries:
            overflow = len(self._entries) - self.max_entries
            del self._entries[:overflow]
        self._cursor = len(self._entries)

    @staticmethod
    def _tokens_match(
        entry: AdjudicationHistoryEntry,
        current_tokens: Mapping[str, str],
        *,
        undo: bool,
    ) -> bool:
        for delta in entry.deltas:
            expected = delta.after_token if undo else delta.before_token
            if str(current_tokens.get(delta.row_key, "") or "") != str(expected or ""):
                return False
        return True

    def peek_undo(self, current_tokens: Mapping[str, str]) -> AdjudicationHistoryEntry | None:
        if not self.can_undo:
            return None
        entry = self._entries[self._cursor - 1]
        return entry if self._tokens_match(entry, current_tokens, undo=True) else None

    def peek_redo(self, current_tokens: Mapping[str, str]) -> AdjudicationHistoryEntry | None:
        if not self.can_redo:
            return None
        entry = self._entries[self._cursor]
        return entry if self._tokens_match(entry, current_tokens, undo=False) else None

    def commit_undo(self, entry: AdjudicationHistoryEntry) -> bool:
        if not self.can_undo or self._entries[self._cursor - 1] is not entry:
            return False
        self._cursor -= 1
        return True

    def commit_redo(self, entry: AdjudicationHistoryEntry) -> bool:
        if not self.can_redo or self._entries[self._cursor] is not entry:
            return False
        self._cursor += 1
        return True
