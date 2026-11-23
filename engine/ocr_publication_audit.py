from __future__ import annotations

"""Evidence-aware, read-only final audit of GPT web OCR decisions.

This is an *anomaly queue*, not an OCR oracle. Never edits candidates or
asserts that a visual decision is correct from a model's self-reported score.
The module deliberately needs only the compact task and answer streams, so the
same code can be shipped in the small GPT exchange package.
"""

from collections import Counter
from difflib import SequenceMatcher
import re
from typing import Any, Mapping, Sequence

SCHEMA = "novel_formatter.ocr_publication_audit.v1"
NUMERIC = set("0123456789０１２３４５６７８９一二三四五六七八九十百千万億零〇壱弐参壹貳參")
VISUAL_HOMOGRAPHS = (
    set("日目曰"), set("二ニ三一ー"), set("力カ"), set("口ロ"),
    set("問間"), set("土工王上"), set("自白"), set("へヘ"),
)
NEGATION = re.compile(r"(?:ない|なかった|ません|ず|ぬ|無い|無し|不|未|非|不可|不能)")
SEQUENCE = re.compile(r"^(?:二|2)(.{1,8})[。．.]?$")


def _text_candidates(task: Mapping[str, Any]) -> list[str]:
    return [str(c.get("t") or "") for c in (task.get("c") or [])
            if isinstance(c, Mapping) and not bool(c.get("f", c.get("fail", False)))
            and str(c.get("t") or "").strip()]


def _choice_text(task: Mapping[str, Any], answer: Mapping[str, Any]) -> str:
    if bool(answer.get("unresolved")) or answer.get("delete_intentionally") is True:
        return ""
    if type(answer.get("pick")) is int:
        c = task.get("c") or []
        i = answer["pick"]
        if 0 <= i < len(c) and isinstance(c[i], Mapping):
            return str(c[i].get("t") or "")
        return ""
    return str(answer.get("text") or "")


def _diff_risks(texts: Sequence[str]) -> set[str]:
    reasons: set[str] = set()
    for i, a in enumerate(texts):
        for b in texts[i + 1:]:
            if a == b:
                continue
            nontrivial = False
            edits = [opcode for opcode in SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
                     if opcode[0] != "equal"]
            if len(edits) >= 2:
                # One whole-sentence pick may fix the first disagreement but
                # retain an independent typo from the chosen OCR. Require a
                # segment-by-segment check and allow a complete mixed text.
                reasons.add("multi_site_candidate_disagreement")
            for op, a0, a1, b0, b1 in edits:
                left, right = a[a0:a1], b[b0:b1]
                chars = set(left + right)
                if any(ch in NUMERIC for ch in chars):
                    reasons.add("numeric_disagreement")
                if any(chars & s for s in VISUAL_HOMOGRAPHS):
                    reasons.add("visually_confusable_characters")
                if NEGATION.search(left) or NEGATION.search(right):
                    reasons.add("negation_disagreement")
                if abs(len(left) - len(right)) >= 2 and (left.strip() or right.strip()):
                    reasons.add("possible_text_omission_or_adhesion")
                if (left.strip() or right.strip()) and not all(
                    not ch.isalnum() for ch in chars
                ):
                    nontrivial = True
            if nontrivial:
                reasons.add("semantic_candidate_disagreement")
    return reasons


def _sequence_risk(task: Mapping[str, Any], selected: str) -> bool:
    """A 2... then 3... pattern can indicate an undetected leading 1.

    A task's `after` may be the next 3-line, but *not* an authority to
    fabricate the missing 1-line. The only safe outcome is an image review.
    """
    curr = re.sub(r"\s+", "", selected)
    after = re.sub(r"\s+", "", str(task.get("a") or task.get("after") or ""))
    m = SEQUENCE.fullmatch(curr)
    if not m or not after:
        return False
    suffix = m.group(1).rstrip("。．.")
    return after.startswith(("三" + suffix, "3" + suffix))


def audit_answers(tasks: Sequence[Mapping[str, Any]], answers: Sequence[Mapping[str, Any]],
                  *, total_rows: int | None = None) -> dict:
    """Produce deterministic risk records without modifying any inputs.

    `visual_reviewed` / free-text reasons are intentionally NOT treated as
    proof; no self-declared confidence can certify scan fidelity.
    """
    by_id = {str(t.get("id") or ""): t for t in tasks}
    latest: dict[str, Mapping[str, Any]] = {}
    duplicates: list[str] = []
    for a in answers:
        tid = str(a.get("id") or "")
        if tid in latest:
            duplicates.append(tid)
        latest[tid] = a
    items: list[dict] = []
    counters: Counter[str] = Counter()
    confidence_scores: Counter[str] = Counter()
    for task in tasks:
        tid = str(task.get("id") or "")
        ans = latest.get(tid)
        signals: set[str] = set()
        severe: set[str] = set()
        if ans is None:
            severe.add("no_answer")
            selected = ""
        else:
            if bool(ans.get("unresolved")):
                severe.add("unresolved")
            if ans.get("delete_intentionally") is True:
                # An image-confirmed page number could be a valid deletion; an
                # automated audit cannot prove the image justified deletion.
                severe.add("destructive_deletion_requires_scan_verification")
            selected = _choice_text(task, ans)
            if "confidence" in ans:
                try:
                    confidence_scores[f"{float(ans['confidence']):.4f}"] += 1
                except (ValueError, TypeError):
                    signals.add("non_numeric_confidence")
        candidates = _text_candidates(task)
        signals.update(_diff_risks(candidates))
        if task.get("r") is not None or task.get("risk") is not None:
            signals.add("common_mode_review")
            risk = task.get("r") or task.get("risk") or {}
            if isinstance(risk, Mapping):
                if any("omission" in str(x) for x in (risk.get("reasons") or [])) or \
                   "omission" in str(risk.get("reason") or ""):
                    severe.add("possible_leading_sequence_omission")
        if _sequence_risk(task, selected) or bool(task.get("requires_full_page_review")):
            severe.add("possible_leading_sequence_omission")
        image_path = str(task.get("i") or task.get("img") or "")
        image_required = bool(signals & {
            "numeric_disagreement", "visually_confusable_characters", "negation_disagreement",
            "possible_text_omission_or_adhesion", "common_mode_review"
        }) or bool(severe)
        if image_required and not image_path:
            severe.add("no_row_image_for_high_risk")
        if ans is not None and "text" in ans and selected and candidates:
            if selected not in candidates:
                signals.add("candidate_external_full_text")
            if len(selected) > 1.6 * max(len(x) for x in candidates):
                signals.add("large_freely_rewritten_text")
        # A generic "image+context" claim cannot certify image inspection;
        # track this book-wide without swamping each individual risk row.
        risk_level = (
            "critical" if severe else
            "high" if (image_required or bool(signals & {
                "candidate_external_full_text", "large_freely_rewritten_text",
            "multi_site_candidate_disagreement"
            })) else
            "advisory" if signals else "low"
        )
        for code in signals | severe:
            counters[code] += 1
        if risk_level in {"critical", "high"}:
            items.append({
                "id": tid, "row": task.get("row"), "page": task.get("p", task.get("page")),
                "risk_level": risk_level, "reasons": sorted(signals | severe),
                "image": image_path,
                "column_ids": list(task.get("cols") or []),
                "status": str(task.get("s") or task.get("status") or ""),
                "selected_text": selected,
                "candidate_texts": candidates[:6],
                "choice_kind": (
                    "missing" if ans is None else "unresolved" if ans.get("unresolved") else
                    "delete" if ans.get("delete_intentionally") is True else
                    "pick" if "pick" in ans else "text" if "text" in ans else "unknown"
                ),
            })
    score_count = sum(confidence_scores.values())
    most_common_score, frequency = confidence_scores.most_common(1)[0] if confidence_scores else ("", 0)
    score_concentration = round(frequency / score_count, 4) if score_count else 0.0
    if score_count >= 20 and score_concentration >= 0.90:
        counters["uniform_confidence_not_calibrated"] = frequency
    critical = sum(x["risk_level"] == "critical" for x in items)
    high = sum(x["risk_level"] == "high" for x in items)
    flagged_ids = {x["id"] for x in items}
    advisory = sum(
        1 for task in tasks
        if str(task.get("id") or "") not in flagged_ids
        and _diff_risks(_text_candidates(task))
    )
    generic_reasons = sum(
        str(x.get("reason") or "") in {"image+context", "candidate_and_context"}
        for x in answers
    )
    if generic_reasons:
        counters["generic_reason_not_visual_proof"] = generic_reasons
    if score_count >= 20 and len(confidence_scores) <= 2:
        counters["confidence_scores_only_one_or_two_values"] = score_count
    total = len(tasks)
    return {
        "schema": SCHEMA,
        "contract": "read_only; highlights review priorities, NOT correct text and NOT scan verification",
        "task_count": total,
        "answer_count": len(answers),
        "distinct_answered": sum(t.get("id") in latest for t in tasks),
        "total_rows": total_rows,
        "unsubmitted_rows": max(0, total_rows - total) if total_rows is not None else None,
        "all_text_proofed": False,
        "risk_item_count": len(items),
        "critical_count": critical,
        "high_risk_count": high,
        "advisory_count": advisory,
        "review_count": high,
        "duplicate_answer_ids": sorted(set(duplicates)),
        "confidence_mode": most_common_score,
        "confidence_mode_ratio": score_concentration,
        "risk_reason_counts": dict(sorted(counters.items())),
        "items": items,
    }
