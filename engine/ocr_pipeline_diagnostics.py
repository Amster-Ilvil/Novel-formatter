#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dependency-free integrity checks for OCR alignment and adjudication state.

The OCR engines, comparison workspace, image review and AI adjudicator remain
independent subsystems.  This module validates only their shared contracts so a
bad hand-off is reported at the boundary where it occurs, before text is lost or
written to a different physical column.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class OcrPipelineIssue:
    stage: str
    code: str
    message: str
    severity: str = "error"
    row_index: int | None = None
    column_ids: tuple[str, ...] = ()


@dataclass(slots=True)
class OcrPipelineAudit:
    stage: str
    issues: list[OcrPipelineIssue] = field(default_factory=list)
    row_count: int = 0
    model_count: int = 0

    @property
    def errors(self) -> tuple[OcrPipelineIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[OcrPipelineIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity != "error")

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(
        self,
        code: str,
        message: str,
        *,
        severity: str = "error",
        row_index: int | None = None,
        column_ids: Sequence[str] = (),
    ) -> None:
        self.issues.append(OcrPipelineIssue(
            stage=self.stage,
            code=str(code),
            message=str(message),
            severity=str(severity),
            row_index=row_index,
            column_ids=tuple(str(value) for value in column_ids if str(value)),
        ))

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "ok": self.ok,
            "row_count": self.row_count,
            "model_count": self.model_count,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [asdict(issue) for issue in self.issues],
        }

    def raise_for_errors(self, prefix: str = "OCR 管线合同校验失败") -> None:
        if not self.errors:
            return
        details = "；".join(
            f"{issue.code}: {issue.message}" for issue in self.errors[:8]
        )
        remainder = len(self.errors) - 8
        if remainder > 0:
            details += f"；另有 {remainder} 项"
        raise ValueError(f"{prefix}：{details}")


def _column_key(value: object) -> tuple[str, ...]:
    return tuple(str(item) for item in (value or ()) if str(item))


def audit_comparison(comparison, *, expected_models: int | None = None) -> OcrPipelineAudit:
    rows = list(getattr(comparison, "rows", ()) or ())
    labels = list(getattr(comparison, "labels", ()) or ())
    model_count = int(expected_models if expected_models is not None else len(labels))
    audit = OcrPipelineAudit("comparison", row_count=len(rows), model_count=model_count)
    if model_count < 1 or model_count > 6:
        audit.add("model_count", f"OCR 对比必须有 1～6 个模型，当前为 {model_count}")

    column_owner: dict[str, int] = {}
    seen_row_indices: set[int] = set()
    for position, row in enumerate(rows):
        try:
            row_index = int(getattr(row, "index", position))
        except (TypeError, ValueError, OverflowError):
            row_index = -1
        if row_index in seen_row_indices:
            audit.add("duplicate_row_index", f"行索引 {row_index} 重复", row_index=position)
        seen_row_indices.add(row_index)
        if row_index != position:
            audit.add(
                "row_index_drift",
                f"第 {position + 1} 行记录的索引为 {row_index}",
                row_index=position,
            )

        texts = list(getattr(row, "texts", ()) or ())
        if len(texts) != model_count:
            audit.add(
                "candidate_cardinality",
                f"第 {position + 1} 行有 {len(texts)} 个候选，但模型数为 {model_count}",
                row_index=position,
            )
        confidences = tuple(getattr(row, "model_confidences", ()) or ())
        if confidences and len(confidences) != model_count:
            audit.add(
                "confidence_cardinality",
                f"第 {position + 1} 行置信度数量 {len(confidences)} 与模型数不一致",
                row_index=position,
            )

        columns = _column_key(getattr(row, "column_ids", ()))
        if len(columns) != len(set(columns)):
            audit.add(
                "duplicate_column_in_row",
                f"第 {position + 1} 行内部包含重复物理列 ID",
                row_index=position,
                column_ids=columns,
            )
        for column_id in columns:
            previous = column_owner.get(column_id)
            if previous is not None and previous != position:
                audit.add(
                    "column_owned_by_multiple_rows",
                    f"物理列 {column_id} 同时属于第 {previous + 1}、{position + 1} 行",
                    row_index=position,
                    column_ids=columns,
                )
            else:
                column_owner[column_id] = position

        for seeded in tuple(getattr(row, "consensus_seeded_models", ()) or ()):
            try:
                seeded_index = int(seeded)
            except (TypeError, ValueError, OverflowError):
                seeded_index = -1
            if not 0 <= seeded_index < model_count:
                audit.add(
                    "seeded_model_index",
                    f"第 {position + 1} 行的共识复用模型索引 {seeded!r} 越界",
                    row_index=position,
                    column_ids=columns,
                )

        if texts and not any(str(text or "").strip(" \r\n□�") for text in texts):
            audit.add(
                "all_candidates_empty",
                f"第 {position + 1} 行所有 OCR 候选均为空或仅含占位符",
                severity="warning",
                row_index=position,
                column_ids=columns,
            )
    return audit


def audit_fusion_states(comparison, states: Sequence[object]) -> OcrPipelineAudit:
    rows = list(getattr(comparison, "rows", ()) or ())
    values = list(states or ())
    audit = OcrPipelineAudit(
        "fusion",
        row_count=len(rows),
        model_count=len(list(getattr(comparison, "labels", ()) or ())),
    )
    if len(values) != len(rows):
        audit.add(
            "state_cardinality",
            f"融合状态 {len(values)} 行与对齐结果 {len(rows)} 行不一致",
        )
        return audit
    for position, (row, state) in enumerate(zip(rows, values)):
        columns = _column_key(getattr(row, "column_ids", ()))
        try:
            state_row = int(getattr(state, "row_index", -1))
        except (TypeError, ValueError, OverflowError):
            state_row = -1
        if state_row != position:
            audit.add(
                "state_row_drift",
                f"第 {position + 1} 行绑定到融合状态索引 {state_row}",
                row_index=position,
                column_ids=columns,
            )
        candidates = list(getattr(state, "candidates", ()) or ())
        selected = getattr(state, "selected_index", None)
        if selected is None:
            continue
        try:
            selected_index = int(selected)
        except (TypeError, ValueError, OverflowError):
            selected_index = -1
        if not 0 <= selected_index < len(candidates):
            audit.add(
                "selected_candidate_index",
                f"第 {position + 1} 行选择的候选索引 {selected!r} 已失效",
                row_index=position,
                column_ids=columns,
            )
            continue
        candidate = candidates[selected_index]
        if str(getattr(candidate, "audit_level", "") or "") == "historical_ocr_evidence":
            audit.add(
                "historical_candidate_selected",
                f"第 {position + 1} 行选择了只读历史证据",
                row_index=position,
                column_ids=columns,
            )
        text = str(getattr(candidate, "text", "") or "").strip()
        deleted = bool(getattr(candidate, "delete_intentionally", False))
        if not text and not deleted:
            audit.add(
                "implicit_empty_selection",
                f"第 {position + 1} 行选择了空文本，但没有有意删除标记",
                row_index=position,
                column_ids=columns,
            )
        if text and deleted:
            audit.add(
                "contradictory_delete_selection",
                f"第 {position + 1} 行同时包含正文和有意删除标记",
                row_index=position,
                column_ids=columns,
            )
    return audit


def audit_canonical_decisions(comparison, decisions: Iterable[dict]) -> OcrPipelineAudit:
    rows = list(getattr(comparison, "rows", ()) or ())
    audit = OcrPipelineAudit(
        "adjudication",
        row_count=len(rows),
        model_count=len(list(getattr(comparison, "labels", ()) or ())),
    )
    known_groups = {
        str(getattr(row, "sentence_group_id", "") or "")
        for row in rows
        if str(getattr(row, "sentence_group_id", "") or "")
    }
    rows_by_columns: dict[tuple[str, ...], list[int]] = {}
    for index, row in enumerate(rows):
        columns = _column_key(getattr(row, "column_ids", ()))
        if columns:
            rows_by_columns.setdefault(columns, []).append(index)

    seen: set[tuple[str, ...]] = set()
    for position, decision in enumerate(decisions or ()):
        if not isinstance(decision, dict):
            audit.add("decision_type", f"第 {position + 1} 个裁决不是对象")
            continue
        columns = _column_key(decision.get("column_ids"))
        group_id = str(decision.get("sentence_group_id", "") or "")
        if group_id:
            identity = ("sentence_group_id", group_id)
            if group_id not in known_groups:
                audit.add(
                    "unknown_decision_sentence_group",
                    f"裁决引用了当前对齐中不存在的句组 {group_id}",
                    column_ids=columns,
                )
        else:
            if not columns:
                audit.add("decision_without_identity", f"第 {position + 1} 个裁决缺少句组 ID 和稳定物理列 ID")
                continue
            identity = ("column_ids",) + columns
            owners = rows_by_columns.get(columns, [])
            if not owners:
                audit.add(
                    "unknown_decision_columns",
                    f"裁决引用了当前对齐中不存在的物理列组 {','.join(columns)}",
                    column_ids=columns,
                )
            elif len(owners) > 1:
                audit.add(
                    "ambiguous_legacy_decision_columns",
                    f"物理列组 {','.join(columns)} 对应 {len(owners)} 个句组；旧裁决缺少 sentence_group_id，不能安全套用",
                    column_ids=columns,
                )
        if identity in seen:
            audit.add(
                "duplicate_decision",
                f"同一裁决身份 {identity[-1]} 收到多个裁决",
                column_ids=columns,
            )
        seen.add(identity)
        status = str(decision.get("status", "") or "")
        if status not in {"accepted", "unresolved"}:
            audit.add(
                "decision_status",
                f"裁决 {identity[-1]} 的状态 {status!r} 无效",
                column_ids=columns,
            )
        final_text = str(decision.get("final_text", "") or "")
        deleted = bool(decision.get("delete_intentionally", False))
        if status == "accepted" and not final_text and not deleted:
            audit.add(
                "accepted_without_output",
                f"裁决 {identity[-1]} 已接受但没有正文或删除标记",
                column_ids=columns,
            )
        if final_text and deleted:
            audit.add(
                "contradictory_decision",
                f"裁决 {identity[-1]} 同时包含正文和删除标记",
                column_ids=columns,
            )
    return audit
