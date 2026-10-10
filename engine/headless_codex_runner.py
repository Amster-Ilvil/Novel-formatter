#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-command Codex adjudication runner for an existing round-trip package."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from engine.headless_adjudication import (
    DECISIONS_SCHEMA,
    HeadlessAdjudicationError,
    build_adjudication_task,
    load_multi_package,
    run_from_decisions,
    task_prompt,
    write_task,
)

CODEX_RUN_REPORT_SCHEMA = "novel_formatter.headless_codex_run.v1"


class CodexRunnerError(HeadlessAdjudicationError):
    pass


def decisions_json_schema(task: dict) -> dict:
    binding = dict((task.get("output_contract") or {}).get("source") or {})
    source_properties = {}
    for key, value in binding.items():
        source_properties[key] = {"const": value}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["schema", "source", "decisions"],
        "properties": {
            "schema": {"const": DECISIONS_SCHEMA},
            "source": {
                "type": "object",
                "additionalProperties": False,
                "required": list(source_properties),
                "properties": source_properties,
            },
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "row_id", "corrected_text", "confidence", "reason",
                        "needs_manual_review", "delete_intentionally",
                    ],
                    "properties": {
                        "row_id": {"type": "string", "minLength": 1},
                        "corrected_text": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "reason": {"type": "string"},
                        "needs_manual_review": {"type": "boolean"},
                        "delete_intentionally": {"type": "boolean"},
                    },
                },
            },
        },
    }


def _safe_row_stem(row: dict, position: int) -> str:
    row_id = str(row.get("row_id", "") or "")
    token = "".join(ch if ch.isalnum() else "_" for ch in row_id).strip("_")
    if not token:
        token = f"row_{position:06d}"
    return f"{position:06d}_{token[:72]}"


def _resolve_source_image(path_value: str, package_path: str | Path) -> Path | None:
    raw = str(path_value or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (Path(package_path).resolve().parent / path).resolve()
    try:
        if path.is_file():
            return path
    except OSError:
        return None
    return None


def _normalized_xywh(value) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x, y, w, h = (float(value[i]) for i in range(4))
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(number == number for number in (x, y, w, h)) or w <= 0 or h <= 0:
        return None
    return x, y, w, h


def _stage_task_visual_evidence(
    task: dict,
    package_path: str | Path,
    output_dir: str | Path,
) -> dict:
    """Copy/crop row evidence into the Codex workspace.

    ``codex exec --sandbox workspace-write`` is intentionally launched in a
    dedicated output folder.  A source book can live elsewhere, so raw absolute
    image paths are not a reliable visual contract for the agent.  This helper
    stages only the selected rows' evidence into ``evidence/`` and rewrites the
    task's ``image_path`` to a workspace-local path.  Original source paths are
    retained for auditing, never modified.
    """
    rows = task.get("rows") or []
    if not rows:
        return {"staged": 0, "missing": 0, "full_page": 0, "cropped": 0}
    evidence_dir = Path(output_dir) / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    staged = missing = full_page = cropped = 0
    # Pillow is already an application dependency, but keep the import local so
    # text-only prepare/validate commands do not pay image-module startup cost.
    try:
        from PIL import Image
    except Exception:
        Image = None

    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        original = str(row.get("image_path", "") or "")
        row["source_image_path"] = original
        source = _resolve_source_image(original, package_path)
        if source is None or Image is None:
            row["visual_evidence_status"] = "missing_source_image"
            missing += 1
            continue
        target = evidence_dir / f"{_safe_row_stem(row, position)}.png"
        bbox = _normalized_xywh(row.get("source_bbox"))
        try:
            with Image.open(source) as im:
                im = im.convert("RGB")
                width, height = im.size
                crop_box = None
                if bbox is not None:
                    x, y, w, h = bbox
                    # Round-trip source_bbox is normalized x/y/w/h.  Tolerate
                    # pixel-space xywh as well for old/custom packages.
                    normalized = max(abs(x), abs(y), abs(w), abs(h)) <= 1.5
                    if normalized:
                        left = x * width
                        top = y * height
                        right = (x + w) * width
                        bottom = (y + h) * height
                    else:
                        left, top, right, bottom = x, y, x + w, y + h
                    # Keep context around the sentence/columns.  Vertical OCR
                    # benefits from more side context than top/bottom context.
                    pad_x = max(12.0, (right - left) * 1.35, width * 0.012)
                    pad_y = max(12.0, (bottom - top) * 0.035, height * 0.008)
                    left = max(0, int(round(left - pad_x)))
                    top = max(0, int(round(top - pad_y)))
                    right = min(width, int(round(right + pad_x)))
                    bottom = min(height, int(round(bottom + pad_y)))
                    if right > left + 4 and bottom > top + 4:
                        crop_box = (left, top, right, bottom)
                if crop_box is not None:
                    evidence = im.crop(crop_box)
                    # Avoid huge page-height crops while keeping glyph detail.
                    if evidence.height > 2200:
                        new_h = 2200
                        new_w = max(1, round(evidence.width * new_h / evidence.height))
                        evidence = evidence.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    evidence.save(target, format="PNG", optimize=True)
                    row["visual_evidence_status"] = "cropped"
                    row["visual_evidence_crop_pixels"] = list(crop_box)
                    cropped += 1
                else:
                    preview = im
                    if max(width, height) > 2200:
                        ratio = 2200 / max(width, height)
                        preview = im.resize(
                            (max(1, round(width * ratio)), max(1, round(height * ratio))),
                            Image.Resampling.LANCZOS,
                        )
                    preview.save(target, format="PNG", optimize=True)
                    row["visual_evidence_status"] = "full_page"
                    full_page += 1
        except Exception as exc:
            row["visual_evidence_status"] = "image_error"
            row["visual_evidence_error"] = str(exc)[:300]
            missing += 1
            continue
        # Use a path relative to Codex cwd.  view_image can open this directly.
        row["image_path"] = str(Path("evidence") / target.name)
        staged += 1
    return {"staged": staged, "missing": missing, "full_page": full_page, "cropped": cropped}


def _codex_executable(value: str = "codex") -> str:
    explicit = str(value or "codex")
    found = shutil.which(explicit)
    if found:
        return found
    path = Path(explicit)
    if path.is_file():
        return str(path.resolve())
    raise CodexRunnerError(
        "未找到 Codex CLI。请先安装/登录 Codex，或使用 --codex-bin 指定可执行文件。"
    )


def run_codex_adjudication(
    package_path: str | Path,
    output_dir: str | Path,
    *,
    codex_bin: str = "codex",
    model: str = "",
    reasoning_effort: str = "high",
    scope: str = "conflicts",
    context_rows: int = 2,
    include_existing_edits: bool = False,
    require_complete: bool = False,
    writing_mode: str = "auto",
    timeout_seconds: int = 7200,
    prepare_only: bool = False,
    skeleton_template: str | Path | None = None,
) -> dict:
    package = load_multi_package(package_path)
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    task = build_adjudication_task(
        package,
        scope=scope,
        context_rows=context_rows,
        include_existing_edits=include_existing_edits,
    )
    evidence_report = _stage_task_visual_evidence(task, package_path, folder)
    task_path = write_task(task, folder / "codex_task.json")
    prompt_path = folder / "codex_prompt.md"
    prompt_text = task_prompt(task, decisions_path="codex_decisions.json") + (
        "\n## 本次运行方式\n"
        "任务文件位于 `codex_task.json`。先完整读取该文件。"
        "row.image_path 若存在，已经被 Novel Formatter 放入当前工作目录的 evidence/；"
        "文字候选不足时应使用 view_image 查看对应裁片；"
        "不得修改任何输入文件。最终回复必须严格符合提供的 JSON Schema。\n"
        "本次 codex exec 已使用 --output-last-message；不要调用 shell 自己写 decisions 文件，"
        "只需把最终 decisions JSON 作为最后回复。\n"
    )
    prompt_path.write_text(prompt_text, encoding="utf-8")
    schema_path = folder / "codex_decisions.schema.json"
    schema_path.write_text(
        json.dumps(decisions_json_schema(task), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    decisions_path = folder / "codex_decisions.json"
    trace_path = folder / "codex_trace.jsonl"

    selected_rows = len(task.get("rows") or [])
    if selected_rows == 0:
        # No agent call is needed; create a structurally valid empty decisions
        # file and let the ordinary validator/exporter finish the book.
        decisions = {
            "schema": DECISIONS_SCHEMA,
            "source": dict((task.get("output_contract") or {}).get("source") or {}),
            "decisions": [],
        }
        decisions_path.write_text(json.dumps(decisions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif prepare_only:
        report = {
            "schema": CODEX_RUN_REPORT_SCHEMA,
            "status": "awaiting_decisions",
            "reason": "prepare_only",
            "task": str(task_path),
            "prompt": str(prompt_path),
            "output_schema": str(schema_path),
            "decisions": str(decisions_path),
            "selected_rows": selected_rows,
            "visual_evidence": evidence_report,
        }
        (folder / "codex_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return report
    else:
        exe = _codex_executable(codex_bin)
        cmd = [
            exe, "exec",
            "--json",
            "--sandbox", "workspace-write",
            "--ask-for-approval", "never",
            "--output-schema", str(schema_path.name),
            "--output-last-message", str(decisions_path.name),
        ]
        if model:
            cmd.extend(["--model", str(model)])
        if reasoning_effort:
            cmd.extend(["--config", f"model_reasoning_effort={reasoning_effort}"])
        cmd.append("-")
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(folder),
                input=prompt_text,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=max(60, int(timeout_seconds)),
                env=os.environ.copy(),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise CodexRunnerError(f"Codex 裁决超时（{timeout_seconds}s）。任务文件已保留在 {folder}。") from exc
        trace_path.write_text(proc.stdout or "", encoding="utf-8")
        (folder / "codex_stderr.log").write_text(proc.stderr or "", encoding="utf-8")
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-3000:]
            raise CodexRunnerError(
                f"Codex CLI 退出 code={proc.returncode}。任务未写回 OCR。\n{tail}"
            )
        if not decisions_path.is_file() or decisions_path.stat().st_size <= 2:
            raise CodexRunnerError("Codex 运行结束但没有生成 codex_decisions.json。")

    vertical = None if writing_mode == "auto" else writing_mode == "vertical"
    pipeline_report = run_from_decisions(
        package_path,
        decisions_path,
        folder,
        scope=scope,
        include_existing_edits=include_existing_edits,
        require_complete=require_complete,
        vertical=vertical,
        skeleton_template=skeleton_template,
    )
    report = {
        "schema": CODEX_RUN_REPORT_SCHEMA,
        "status": "completed",
        "task": str(task_path),
        "prompt": str(prompt_path),
        "output_schema": str(schema_path),
        "decisions": str(decisions_path),
        "trace": str(trace_path) if trace_path.exists() else "",
        "selected_rows": selected_rows,
        "visual_evidence": evidence_report,
        "pipeline": pipeline_report,
    }
    (folder / "codex_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
