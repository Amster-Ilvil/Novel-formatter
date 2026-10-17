#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert ORIGINAL V5 task ZIP + V5.3 1036-slot audit into V5.5 inputs.

This converter does NOT claim V5.3 has physical bounding boxes. In particular,
column IDs, pixel hashes, and anchor scores are NOT proof of where the glyph
sits on the original page. Fill image_path, box, and position_verified only
after checking the actual original column/page and adjacent OCR anchors.

Run:
  python scripts/ocr_glyph_v53_to_v55.py --package ORIGINAL_V5.zip \
      --audit 1036_slot_audit.csv --out V55_INPUT
  # Manually complete V55_INPUT/GLYPH_JOBS_NEED_COORDINATES.jsonl
  python scripts/ocr_glyph_batch_v55.py \
      --conflicts V55_INPUT/V5_EDITABLE_CONFLICTS.jsonl \
      --glyph-jobs V55_INPUT/GLYPH_JOBS_NEED_COORDINATES.jsonl \
      --out V55_REVIEW

The output files never rewrite original model texts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.ocr_glyph_batch_v55 import make_substitution_slots
from scripts.ocr_glyph_batch_v55 import extract_candidates, read_jsonl, write_jsonl


def _read_package(path: Path) -> tuple[str, list[dict]]:
    supported = ("04_ai_tasks_compact.jsonl", "03_conflict_items.jsonl", "04_all_items_index.jsonl")
    if path.is_file() and path.suffix.lower() == ".jsonl":
        return path.name, read_jsonl(path)
    if path.is_dir():
        for name in supported:
            matches = list(path.rglob(name))
            if len(matches) > 1:
                raise ValueError(f"Multiple {name} files: specify exact JSONL")
            if matches:
                return name, read_jsonl(matches[0])
    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as source:
            names = source.namelist()
            for basename in supported:
                members = [p for p in names if Path(p).name == basename]
                if len(members) > 1:
                    raise ValueError(f"Duplicate {basename} entries in archive")
                if members:
                    info = source.getinfo(members[0])
                    if info.file_size > 128 * 1024 * 1024:
                        raise ValueError("Task file exceeds safe size cap")
                    text = source.read(info).decode("utf-8-sig")
                    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
                    if any(not isinstance(row, dict) for row in rows):
                        raise ValueError("Task JSONL includes non-object entries")
                    return basename, rows
    raise FileNotFoundError("Cannot locate supported V5 compact or native task JSONL")


def _task_id(row: dict) -> str:
    return str(
        row.get("id") or row.get("item_id") or row.get("task_id") or row.get("row_id") or ""
    ).strip()


def convert(package_path: Path, audit_path: Path, out: Path) -> dict:
    name, source = _read_package(package_path)
    tasks: dict[str, dict] = {}
    conflicts: list[dict] = []
    for row in source:
        task = _task_id(row)
        if not task:
            raise ValueError("Original V5 item lacks a stable task ID")
        if task in tasks:
            raise ValueError(f"Duplicate original task ID: {task}")
        texts = extract_candidates(row)
        tasks[task] = {"raw_candidates": texts, "original": row}
        if len(texts) >= 2 and len(set(texts)) >= 2:
            if row.get("model_action_required") is not False:
                conflicts.append({
                    "item_id": task,
                    "model_action_required": True,
                    "candidates": [{"raw_text": text} for text in texts],
                })

    all_slots = {
        (record["item_id"], slot["a_index"]): slot
        for record in conflicts
        for slot in make_substitution_slots(
            record["item_id"], extract_candidates(record)
        )
    }
    # (item_id, a_index) is unique for the SequenceMatcher substitutions.
    seen: set[tuple[str, int]] = set()
    jobs: list[dict] = []
    rejected: list[dict] = []
    statuses: Counter[str] = Counter()
    with audit_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            task = str(row.get("task_id") or "").strip()
            try:
                index = int(row["offset_a"])
            except (ValueError, KeyError, TypeError):
                rejected.append({"task_id": task, "reason": "invalid_v53_a_index"})
                continue
            key = (task, index)
            if key in seen:
                raise ValueError(f"Duplicate V5.3 slot row: {key}")
            seen.add(key)
            slot = all_slots.get(key)
            if slot is None:
                rejected.append({"task_id": task, "a_index": index, "reason": "not_a_matching_v5_substitution"})
                continue
            pair = str(row.get("pair") or "")
            expected_pair = f'{slot["a"]}¦{slot["b"]}'
            if pair and pair != expected_pair:
                rejected.append({
                    "task_id": task, "a_index": index,
                    "reason": "v53_pair_differs_from_original_v5_candidates",
                    "v53_pair": pair, "original_v5_pair": expected_pair,
                })
                continue
            state = str(row.get("status") or "unknown")
            statuses[state] += 1
            jobs.append({
                "task_id": task,
                "a_index": slot["a_index"],
                "b_index": slot["b_index"],
                "candidate_sha256": slot["candidate_sha256"],
                "a": slot["a"], "b": slot["b"],
                "page_id": str(row.get("source_page") or ""),
                "column_id": str(row.get("col_id") or ""),
                # Crucial: V5.3 audit did NOT contain original coordinates.
                "image_path": "",
                "box": [],
                "position_verified": False,
                "v53_status": state,
                "v53_group_id": str(row.get("group_id") or ""),
                "v53_pixel_sha256": str(row.get("pixel_sha256") or ""),
                "v53_anchor_score": str(row.get("anchor_score") or ""),
                "v53_stage": str(row.get("stage") or ""),
                "review_instruction": "Supply original image and actual x0,y0,x1,y1; confirm target glyph and OCR-neighbor anchors.",
            })
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "V5_EDITABLE_CONFLICTS.jsonl", conflicts)
    write_jsonl(out / "GLYPH_JOBS_NEED_COORDINATES.jsonl", jobs)
    write_jsonl(out / "REJECTED_OR_CHANGED_SLOTS.jsonl", rejected)
    summary = {
        "source_v5_entry": name,
        "source_v5_task_count": len(tasks),
        "editable_conflict_task_count": len(conflicts),
        "matched_v53_slot_count": len(jobs),
        "unmatched_or_changed_v53_slots": len(rejected),
        "status_counts": dict(statuses),
        "verified_bbox_count": 0,
        "completed_adjudication_count": 0,
        "source_candidate_texts_not_modified": True,
        "requires_original_images_and_explicit_pixel_coordinates": True,
    }
    (out / "SUMMARY_NEEDS_ORIGINAL_BBOX.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        convert(args.package, args.audit, args.out), ensure_ascii=False, indent=2
    ))


if __name__ == "__main__":
    main()
