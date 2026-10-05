# -*- coding: utf-8 -*-
"""Deterministic OCR quality metrics for Novel Formatter golden sets."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Iterable, Mapping


def _distance_with_ops(reference: str, hypothesis: str) -> tuple[int, int, int, int]:
    """Levenshtein distance plus substitutions/deletions/insertions."""
    a = list(reference or "")
    b = list(hypothesis or "")
    rows = len(a) + 1
    cols = len(b) + 1
    dp = [[0] * cols for _ in range(rows)]
    op = [[""] * cols for _ in range(rows)]
    for i in range(1, rows):
        dp[i][0] = i
        op[i][0] = "D"
    for j in range(1, cols):
        dp[0][j] = j
        op[0][j] = "I"
    for i in range(1, rows):
        for j in range(1, cols):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                op[i][j] = "M"
                continue
            candidates = (
                (dp[i - 1][j - 1] + 1, "S"),
                (dp[i - 1][j] + 1, "D"),
                (dp[i][j - 1] + 1, "I"),
            )
            dp[i][j], op[i][j] = min(candidates, key=lambda item: (item[0], "SDI".index(item[1])))
    i, j = len(a), len(b)
    subs = dels = ins = 0
    while i or j:
        code = op[i][j]
        if code in {"M", "S"}:
            if code == "S":
                subs += 1
            i -= 1
            j -= 1
        elif code == "D":
            dels += 1
            i -= 1
        elif code == "I":
            ins += 1
            j -= 1
        else:
            break
    return dp[-1][-1], subs, dels, ins


@dataclass(frozen=True)
class OCRCaseScore:
    case_id: str
    reference_chars: int
    hypothesis_chars: int
    distance: int
    substitutions: int
    deletions: int
    insertions: int
    exact: bool
    cer: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class OCRBenchmarkReport:
    cases: int
    exact_cases: int
    reference_chars: int
    hypothesis_chars: int
    distance: int
    substitutions: int
    deletions: int
    insertions: int
    cer: float
    exact_rate: float
    case_scores: tuple[OCRCaseScore, ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["case_scores"] = [item.to_dict() for item in self.case_scores]
        return payload


def score_case(case_id: str, reference: str, hypothesis: str) -> OCRCaseScore:
    distance, subs, dels, ins = _distance_with_ops(reference, hypothesis)
    denominator = max(1, len(reference or ""))
    return OCRCaseScore(
        case_id=str(case_id),
        reference_chars=len(reference or ""),
        hypothesis_chars=len(hypothesis or ""),
        distance=distance,
        substitutions=subs,
        deletions=dels,
        insertions=ins,
        exact=(reference or "") == (hypothesis or ""),
        cer=distance / denominator,
    )


def evaluate_cases(cases: Iterable[Mapping[str, object]]) -> OCRBenchmarkReport:
    scores: list[OCRCaseScore] = []
    for index, row in enumerate(cases, start=1):
        case_id = str(row.get("id") or row.get("case_id") or index)
        reference = str(row.get("reference") if row.get("reference") is not None else row.get("truth") or "")
        hypothesis = str(row.get("hypothesis") if row.get("hypothesis") is not None else row.get("prediction") or "")
        scores.append(score_case(case_id, reference, hypothesis))
    ref_chars = sum(item.reference_chars for item in scores)
    hyp_chars = sum(item.hypothesis_chars for item in scores)
    distance = sum(item.distance for item in scores)
    exact = sum(1 for item in scores if item.exact)
    return OCRBenchmarkReport(
        cases=len(scores),
        exact_cases=exact,
        reference_chars=ref_chars,
        hypothesis_chars=hyp_chars,
        distance=distance,
        substitutions=sum(item.substitutions for item in scores),
        deletions=sum(item.deletions for item in scores),
        insertions=sum(item.insertions for item in scores),
        cer=distance / max(1, ref_chars),
        exact_rate=exact / max(1, len(scores)),
        case_scores=tuple(scores),
    )


def load_benchmark_manifest(path: str | Path) -> list[dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        rows = raw.get("cases") or []
    else:
        rows = raw
    if not isinstance(rows, list):
        raise ValueError("benchmark manifest 必须是 cases 数组或数组本身")
    return [dict(row) for row in rows if isinstance(row, dict)]
