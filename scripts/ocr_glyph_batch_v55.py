#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only V5.5 proof audit and review proposal tool.

Example:
  python scripts/ocr_glyph_batch_v55.py \
    --conflicts 03_conflict_items.jsonl \
    --glyph-jobs physical_glyph_jobs.jsonl \
    --out V55_REVIEW
  python scripts/ocr_glyph_batch_v55.py \
    --conflicts 03_conflict_items.jsonl \
    --glyph-jobs physical_glyph_jobs.jsonl \
    --decisions V55_REVIEW/GROUP_DECISIONS_TEMPLATE.json \
    --out V55_REVIEWED

The GROUP_DECISIONS_TEMPLATE defaults to approved=false. Never imports
drafts to V5 or writes back to raw model evidence.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.ocr_glyph_batch_v55 import (
    candidate_digest,
    compose_complete_sentence,
    exact_glyph_groups,
    make_glyph_proof,
    make_substitution_slots,
    propose_slot,
    summarize_batch,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number}: expected JSON object")
        values.append(value)
    return values


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def extract_candidates(record: dict[str, Any]) -> list[str]:
    """Handle current V5 records and compact historical V5 tasks."""
    candidates = record.get("candidates", record.get("c", ()))
    values = []
    for row in candidates:
        if isinstance(row, str):
            values.append(row)
        elif isinstance(row, dict):
            values.append(str(row.get("raw_text", row.get("text", row.get("t", ""))) or ""))
    return values


def record_id(record: dict[str, Any]) -> str:
    return str(record.get("item_id", record.get("id", ""))).strip()


def _proof_binding(p) -> dict[str, Any]:
    return {
        "source_sha256": p.source_sha256,
        "page_id": p.page_id,
        "column_id": p.column_id,
        "box": list(p.box),
    }


def make_template(
    groups: list[dict[str, Any]],
    proofs_by_position: dict[tuple, tuple],
    slots_by_job: dict[tuple, dict[str, Any]],
) -> list[dict[str, Any]]:
    templates = []
    for group in groups:
        member_data = []
        unique_tasks = set()
        for member in group["members"]:
            # Reconstruct only keys that were just hashed from real pixel crops.
            physical = (member["source_sha256"], tuple(member["box"]))
            pair = proofs_by_position.get(physical)
            if pair is None:
                continue
            proof, job = pair
            indices = (int(job["a_index"]), int(job["b_index"]))
            slot = slots_by_job.get((proof.task_id, *indices))
            if slot is None:
                continue
            member_data.append((proof, slot, job))
            unique_tasks.add(proof.task_id)
        # Two glyphs in the same task could demand different per-slot binding;
        # this initial adapter excludes that unsafe case rather than guessing.
        if len(member_data) != len(unique_tasks):
            templates.append({
                "glyph_key": group["glyph_key"], "approved": False,
                "blocked": "multiple_positions_for_same_task_use_manual_review",
                "member_count": len(member_data),
            })
            continue
        templates.append({
            "glyph_key": group["glyph_key"],
            "approved": False,
            "character": "",
            "review_note": "",
            "codepoint_policy": "visual_only",
            "allowed_task_ids": sorted(unique_tasks),
            "exceptions": [],
            "candidate_sha256_by_task": {
                proof.task_id: slot["candidate_sha256"]
                for proof, slot, _job in member_data
            },
            "proof_by_task": {
                proof.task_id: _proof_binding(proof)
                for proof, _slot, _job in member_data
            },
            "member_choices": [
                {"task_id": proof.task_id,
                 "a": slot["a"], "b": slot["b"],
                 "a_index": slot["a_index"], "b_index": slot["b_index"],
                 "position_verified": proof.position_verified}
                for proof, slot, _job in member_data
            ],
        })
    return templates


def build(
    *,
    conflicts_path: Path,
    jobs_path: Path,
    out_dir: Path,
    decisions_path: Path | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    conflicts = read_jsonl(conflicts_path)
    tasks: dict[str, dict[str, Any]] = {}
    for record in conflicts:
        task_id = record_id(record)
        if not task_id:
            raise ValueError("V5 conflict record missing stable item_id/id")
        if task_id in tasks:
            raise ValueError("duplicate V5 task id: " + task_id)
        texts = extract_candidates(record)
        if len(texts) < 2:
            continue
        if record.get("model_action_required") is False:
            continue  # frozen item
        tasks[task_id] = {"texts": texts, "slots": make_substitution_slots(task_id, texts)}
    jobs = read_jsonl(jobs_path)
    proofs = []
    accepted_jobs = []
    rejected_jobs = []
    slots_by_job = {}
    proofs_by_position = {}
    for row_no, job in enumerate(jobs, 1):
        task_id = str(job.get("task_id", ""))
        task = tasks.get(task_id)
        if task is None:
            rejected_jobs.append({"row": row_no, "task_id": task_id, "reason": "not_editable_or_missing_v5_task"})
            continue
        try:
            ai, bi = int(job["a_index"]), int(job["b_index"])
        except (ValueError, KeyError, TypeError):
            rejected_jobs.append({"row": row_no, "task_id": task_id, "reason": "invalid_ocr_slot_offsets"})
            continue
        slot = next(
            (x for x in task["slots"] if x["a_index"] == ai and x["b_index"] == bi),
            None,
        )
        if slot is None:
            rejected_jobs.append({"row": row_no, "task_id": task_id, "reason": "slot_not_in_immutable_ocr_candidates"})
            continue
        try:
            proof = make_glyph_proof(
                job["image_path"], task_id=task_id,
                page_id=str(job["page_id"]), column_id=str(job["column_id"]),
                box=job["box"], position_verified=job.get("position_verified") is True,
            )
        except (OSError, ValueError, KeyError) as exc:
            rejected_jobs.append({
                "row": row_no, "task_id": task_id,
                "reason": "invalid_original_scan_crop:" + str(exc)[:180],
            })
            continue
        proofs.append(proof)
        accepted_jobs.append({"task_id": task_id, "slot": slot, "proof": asdict(proof)})
        slots_by_job[(task_id, ai, bi)] = slot
        proofs_by_position[proof.physical_key] = (proof, job)
    groups, duplicates = exact_glyph_groups(proofs)
    templates = make_template(groups, proofs_by_position, slots_by_job)
    decisions = []
    if decisions_path is not None:
        value = json.loads(decisions_path.read_text(encoding="utf-8-sig"))
        if not isinstance(value, list):
            raise ValueError("review decisions must be JSON array")
        decisions = value
    decisions_by_key = {}
    for row in decisions:
        key = row.get("glyph_key")
        if not key or key in decisions_by_key:
            raise ValueError("missing/duplicate reviewed glyph_key")
        decisions_by_key[key] = row

    proposals = []
    for group in groups:
        reviewed = decisions_by_key.get(group["glyph_key"])
        if reviewed is None:
            continue
        for member in group["members"]:
            physical = (member["source_sha256"], tuple(member["box"]))
            proof_and_job = proofs_by_position.get(physical)
            if proof_and_job is None:
                continue
            proof, job = proof_and_job
            slot = slots_by_job.get((proof.task_id, int(job["a_index"]), int(job["b_index"])))
            if slot is None:
                continue
            proposals.append(propose_slot(slot, proof, reviewed))
    by_task = defaultdict(list)
    for proposal in proposals:
        by_task[proposal["task_id"]].append(proposal)
    answers = []
    partial = []
    for task_id, task in tasks.items():
        if task_id not in by_task:
            continue
        result = compose_complete_sentence(task_id, task["texts"], by_task[task_id])
        if result.get("complete"):
            answers.append({
                "item_id": task_id,
                "selected_text": result["selected_text"],
                "source": "glyph_batch_reviewed_draft",
                "candidate_sha256": result["source_candidate_sha256"],
                "NOT_APPROVED_FOR_IMPORT": True,
            })
        else:
            partial.append(result)

    write_jsonl(out_dir / "GLYPH_POSITION_AUDIT.jsonl", accepted_jobs)
    write_jsonl(out_dir / "GLYPH_REJECTED_JOBS.jsonl", rejected_jobs)
    write_jsonl(out_dir / "SLOT_SUGGESTIONS_REVIEW_REQUIRED.jsonl", proposals)
    write_jsonl(out_dir / "V5_ANSWERS_DRAFT_NOT_IMPORTABLE.jsonl", answers)
    write_jsonl(out_dir / "PARTIAL_TASKS.jsonl", partial)
    (out_dir / "GROUPS_UNAPPROVED.json").write_text(
        json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "GROUP_DECISIONS_TEMPLATE.json").write_text(
        json.dumps(templates, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = summarize_batch(proofs, groups)
    summary.update({
        "input_task_count": len(tasks),
        "input_job_count": len(jobs),
        "accepted_glyph_job_count": len(accepted_jobs),
        "rejected_glyph_job_count": len(rejected_jobs),
        "duplicate_physical_crop_count": len(duplicates),
        "full_answer_draft_count": len(answers),
        "partial_answer_task_count": len(partial),
        "all_outputs_read_only": True,
        "model_or_network_calls": 0,
        "note": "Draft text is NOT a V5 AI_IMPORT file; manually review before import.",
    })
    (out_dir / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conflicts", type=Path, required=True)
    parser.add_argument("--glyph-jobs", type=Path, required=True)
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(
        conflicts_path=args.conflicts,
        jobs_path=args.glyph_jobs,
        out_dir=args.out,
        decisions_path=args.decisions,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
