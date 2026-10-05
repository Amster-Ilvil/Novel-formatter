from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Iterable, Mapping


@dataclass(frozen=True, slots=True)
class BatchChange:
    """One immutable proposed change shown before a batch write.

    ``key`` is a stable domain identity (block id, sentence-group id, etc.).
    The core object is deliberately Qt-free so planning, auditing and tests do
    not depend on the desktop runtime.
    """

    key: str
    before: str
    after: str
    category: str = "change"
    detail: str = ""
    metadata: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @property
    def changed(self) -> bool:
        return self.before != self.after

    @classmethod
    def create(
        cls,
        key: str,
        before: str,
        after: str,
        *,
        category: str = "change",
        detail: str = "",
        metadata: Mapping[str, object] | None = None,
    ) -> "BatchChange":
        items = tuple(
            sorted((str(k), str(v)) for k, v in dict(metadata or {}).items())
        )
        return cls(
            key=str(key or ""),
            before=str(before or ""),
            after=str(after or ""),
            category=str(category or "change"),
            detail=str(detail or ""),
            metadata=items,
        )


def content_state_token(value: object, *, lineage: str = "") -> str:
    """Return a stable content fingerprint suitable for guarded restore checks.

    The token deliberately binds content separately from any repository/version
    pointer.  Callers may provide a lineage string (commit id, workspace id,
    etc.) as extra context, but content changes are always detected even when
    the lineage pointer itself does not move.
    """
    from core.artifact_pipeline import stable_json_hash

    return f"{str(lineage or '')}:{stable_json_hash(value)}"


@dataclass(frozen=True, slots=True)
class BatchRestoreGuard:
    """Small immutable guard used by one-step Preview → Apply → Restore flows."""

    operation: str
    after_token: str
    change_set_fingerprint: str

    def can_restore(self, current_token: str) -> bool:
        return bool(self.after_token) and str(current_token or "") == self.after_token


@dataclass(frozen=True, slots=True)
class BatchChangeSet:
    """Immutable preview contract for a potentially large batch operation."""

    title: str
    changes: tuple[BatchChange, ...]
    source_token: str = ""
    operation: str = "batch"

    @classmethod
    def from_changes(
        cls,
        title: str,
        changes: Iterable[BatchChange],
        *,
        source_token: str = "",
        operation: str = "batch",
    ) -> "BatchChangeSet":
        return cls(
            title=str(title or "批量修改预览"),
            changes=tuple(change for change in changes if change.changed),
            source_token=str(source_token or ""),
            operation=str(operation or "batch"),
        )

    @property
    def count(self) -> int:
        return len(self.changes)

    @property
    def fingerprint(self) -> str:
        payload = {
            "operation": self.operation,
            "source_token": self.source_token,
            "changes": [
                {
                    "key": item.key,
                    "before": item.before,
                    "after": item.after,
                    "category": item.category,
                    "detail": item.detail,
                    "metadata": list(item.metadata),
                }
                for item in self.changes
            ],
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return sha256(raw.encode("utf-8")).hexdigest()

    def category_counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for item in self.changes:
            result[item.category] = result.get(item.category, 0) + 1
        return result

    def compact_summary(self) -> str:
        if not self.changes:
            return "没有实际变更"
        categories = self.category_counts()
        details = " · ".join(f"{key} {value}" for key, value in sorted(categories.items()))
        return f"共 {self.count} 项变更" + (f" · {details}" if details else "")
