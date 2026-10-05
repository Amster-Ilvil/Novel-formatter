from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

MAIN_ROLE_ORDER = ("column", "page", "sentence")
REVIEW_ROLE_ORDER = ("review1", "review2", "review3")
ROLE_ORDER = MAIN_ROLE_ORDER + REVIEW_ROLE_ORDER
MULTI_OCR_ROLE_SCHEMA = 2
MULTI_OCR_SLOT_SCHEMA = 3
# Explicit-role schema 1 remains readable for safe upgrade/recovery.  Schema 0
# (old unlabeled multi-model sessions) is still rejected because model order alone
# is not enough to prove evidence granularity.
SUPPORTED_MULTI_OCR_ROLE_SCHEMAS = frozenset({1, MULTI_OCR_ROLE_SCHEMA, MULTI_OCR_SLOT_SCHEMA})
LEGACY_SENTENCE_REFLOW_TRANSPORT_VERSION = "48px-horizontal-sentence-reflow-v1"
ROLE_LABELS = {
    "column": "逐列主模型",
    "page": "整页主模型",
    # Compatibility key: historical workspaces store the third main slot as
    # ``sentence``. Phase27 redefines that slot as a second full-column pass.
    "sentence": "全列主模型",
    "review1": "分歧复核 1",
    "review2": "分歧复核 2",
    "review3": "分歧复核 3",
}


@dataclass(frozen=True, slots=True)
class MultiOcrRolePlan:
    column: str = ""
    page: str = ""
    sentence: str = ""
    review1: str = ""
    review2: str = ""
    review3: str = ""

    @classmethod
    def from_mapping(cls, values: Mapping[str, object] | None) -> "MultiOcrRolePlan":
        source = values or {}
        return cls(**{
            role: str(source.get(role, "") or "").strip()
            for role in ROLE_ORDER
        })

    def as_dict(self) -> dict[str, str]:
        return {role: str(getattr(self, role) or "") for role in ROLE_ORDER}

    @property
    def main_roles(self) -> tuple[tuple[str, str], ...]:
        return tuple((role, getattr(self, role)) for role in MAIN_ROLE_ORDER if getattr(self, role))

    @property
    def review_roles(self) -> tuple[tuple[str, str], ...]:
        return tuple((role, getattr(self, role)) for role in REVIEW_ROLE_ORDER if getattr(self, role))

    @property
    def selected_roles(self) -> tuple[tuple[str, str], ...]:
        return self.main_roles + self.review_roles

    @property
    def selected_engines(self) -> tuple[str, ...]:
        return tuple(engine for _role, engine in self.selected_roles)

    @property
    def can_run(self) -> bool:
        return bool(self.main_roles)

    @property
    def preferred_execution_roles(self) -> tuple[tuple[str, str], ...]:
        """Return deterministic runtime order without changing role semantics.

        Column and page roles are both independent main evidence.  When their
        resource classes can overlap the scheduler starts both concurrently, so
        their tuple order has no throughput cost.  When they *cannot* overlap,
        prefer the column role first: it establishes canonical physical/sentence
        geometry early and prevents a slow full-page recognizer from occupying
        the UI for minutes before any other selected model gets a turn.

        The legacy ``sentence`` slot is now the full-column main role.  It still
        runs after the first column/page bootstrap for deterministic resource use,
        but it reads every shared physical column and never consumes sentence-reflow
        images.
        """
        roles = list(self.main_roles)
        priority = {"column": 0, "page": 1, "sentence": 2}
        roles.sort(key=lambda item: priority.get(item[0], 9))
        return tuple(roles)

    def duplicate_engines(self) -> tuple[str, ...]:
        seen: set[str] = set()
        duplicates: list[str] = []
        for engine in self.selected_engines:
            if engine in seen and engine not in duplicates:
                duplicates.append(engine)
            seen.add(engine)
        return tuple(duplicates)

    def validate(self, *, compatible_engines: Iterable[str] | None = None) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.can_run:
            issues.append("至少选择一个主识别模型（逐列 / 整页 / 全列）。")
        duplicates = self.duplicate_engines()
        if duplicates:
            issues.append(
                "同一个 OCR 模型只能占用一个角色；需要同模型分歧确认时使用定向重试，不重复占槽："
                + "、".join(duplicates)
            )
        if compatible_engines is not None:
            allowed = {str(value) for value in compatible_engines}
            invalid = [engine for engine in self.selected_engines if engine not in allowed]
            if invalid:
                issues.append("当前 OCR 模式不支持：" + "、".join(dict.fromkeys(invalid)))
        return tuple(issues)


def runtime_input_role(role: str, engine_id: str = "") -> str:
    """Map a UI role to the actual pixel transport used at runtime.

    Disagreement-review roles re-read only conflicting physical columns.
    The historical ``sentence`` slot is retained as a persistence/API key only;
    its Phase27 runtime transport is now ``column`` so it performs an independent
    full pass over every shared physical column.
    """
    normalized = str(role or "").strip().lower()
    engine = str(engine_id or "").strip().lower()
    if normalized in REVIEW_ROLE_ORDER:
        return "column"
    if normalized == "sentence":
        return "column"
    if normalized in MAIN_ROLE_ORDER:
        return normalized
    return "auto"


def resources_can_overlap(left: str, right: str) -> bool:
    """Return whether two OCR runtime classes are safe/useful to overlap.

    Role scheduling uses this only for the full-page + full-column bootstrap
    wave.  The rule is intentionally conservative: CPU-heavy recognizers do
    not overlap each other, and Apple Vision is not overlapped with another
    Apple-GPU/MPS model.  The important fast path is CPU/ONNX + MPS (for
    example NDLOCR-Lite + Hayai OCR), plus network-bound remote work.
    """
    a = str(left or "").strip().lower()
    b = str(right or "").strip().lower()
    if not a or not b or a == b:
        return False
    if "remote_model" in {a, b}:
        return True
    pair = frozenset((a, b))
    safe_pairs = {
        frozenset(("onnx_model", "mps_model")),
        frozenset(("cpu_model", "mps_model")),
        frozenset(("onnx_model", "cuda_model")),
        frozenset(("cpu_model", "cuda_model")),
        frozenset(("onnx_model", "native_vision")),
        frozenset(("cpu_model", "native_vision")),
    }
    return pair in safe_pairs


def role_display(role: str, engine_label: str = "") -> str:
    base = ROLE_LABELS.get(str(role or ""), str(role or ""))
    return f"{base} · {engine_label}" if engine_label else base
