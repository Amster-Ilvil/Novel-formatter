#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Novel Formatter headless adjudication CLI.

The CLI deliberately starts *after OCR is finished*.  It never imports or starts
PySide6, Page Manager, OCR engines, body-ROI selection, or model runtimes.

Primary workflow::

    python -m novel_formatter.cli adjudicate /path/to/project --output-dir out

``adjudicate`` accepts an existing Novel Formatter project folder, a sealed
round-trip JSON, or a fusion/skeleton ZIP.  Previous human/AI decisions are
reset from raw OCR candidates before a new Codex run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from engine.headless_adjudication import (
    HeadlessAdjudicationError,
    apply_decisions,
    build_adjudication_task,
    export_skeleton_epub,
    load_decisions,
    load_multi_package,
    run_from_decisions,
    save_document,
    save_unresolved,
    task_prompt,
    validate_decisions,
    write_task,
)
from engine.headless_codex_runner import CodexRunnerError, run_codex_adjudication
from engine.headless_workspace_source import resolve_fresh_ocr_source
from engine.ocr_roundtrip_package import save_package


def _json_print(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _load_task(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise HeadlessAdjudicationError("task JSON 必须是对象。")
    return value


def _allowed_from_task(task: dict | None):
    if not task:
        return None
    return [str(row.get("row_id", "") or "") for row in task.get("rows") or []]


def cmd_prepare(args) -> int:
    package = load_multi_package(args.package)
    task = build_adjudication_task(
        package,
        scope=args.scope,
        context_rows=args.context_rows,
        include_existing_edits=args.include_existing_edits,
    )
    write_task(task, args.output)
    if args.prompt:
        Path(args.prompt).write_text(
            task_prompt(task, decisions_path=args.decisions_name), encoding="utf-8"
        )
    _json_print({
        "status": "ok",
        "task": str(Path(args.output)),
        "prompt": str(Path(args.prompt)) if args.prompt else "",
        "selected_rows": len(task.get("rows") or []),
        "skipped_existing_edits": int((task.get("policy") or {}).get("skipped_existing_edits", 0) or 0),
        "candidate_sha256": str((task.get("source") or {}).get("candidate_sha256", "") or ""),
    })
    return 0


def cmd_validate(args) -> int:
    package = load_multi_package(args.package)
    decisions = load_decisions(args.decisions)
    task = _load_task(args.task) if args.task else None
    result = validate_decisions(
        package,
        decisions,
        allowed_row_ids=_allowed_from_task(task),
        require_complete=args.require_complete,
    )
    _json_print({
        "status": "ok",
        "decision_count": len(result["normalized"]),
        "unresolved_count": len(result["unresolved"]),
        "candidate_sha256": result["candidate_sha256"],
    })
    return 0


def cmd_apply(args) -> int:
    package = load_multi_package(args.package)
    decisions = load_decisions(args.decisions)
    task = _load_task(args.task) if args.task else None
    result = apply_decisions(
        package,
        decisions,
        allowed_row_ids=_allowed_from_task(task),
        require_complete=args.require_complete,
    )
    save_package(result.package, args.output_package)
    if args.output_document:
        save_document(result.document, args.output_document)
    if args.unresolved:
        save_unresolved(result.package, result.unresolved, args.unresolved)
    _json_print({
        "status": "ok",
        "applied": result.applied,
        "unresolved": len(result.unresolved),
        "raw_candidates_unchanged": result.candidate_sha256_before == result.candidate_sha256_after,
        "candidate_sha256": result.candidate_sha256_after,
        "output_package": str(Path(args.output_package)),
        "output_document": str(Path(args.output_document)) if args.output_document else "",
    })
    return 0


def cmd_export_skeleton(args) -> int:
    package = load_multi_package(args.package)
    vertical = None if args.writing_mode == "auto" else args.writing_mode == "vertical"
    report = export_skeleton_epub(package, args.output, vertical=vertical)
    _json_print({"status": "ok", **report})
    return 0


def cmd_run(args) -> int:
    vertical = None if args.writing_mode == "auto" else args.writing_mode == "vertical"
    report = run_from_decisions(
        args.package,
        args.decisions,
        args.output_dir,
        scope=args.scope,
        include_existing_edits=args.include_existing_edits,
        require_complete=args.require_complete,
        vertical=vertical,
    )
    _json_print({"status": "ok", **report})
    return 0


def cmd_adjudicate(args) -> int:
    """Fresh re-adjudication from an already-OCR'd workspace/package only."""
    folder = Path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    resolved = resolve_fresh_ocr_source(args.source, folder / "source")
    fresh_package = save_package(resolved.package, folder / "fresh_roundtrip.json")
    source_report = {
        "kind": resolved.kind,
        "source_path": resolved.source_path,
        "previous_authority_count_ignored": resolved.previous_authority_count,
        "reset_rows": resolved.reset_rows,
        "fresh_roundtrip": str(fresh_package),
        "skeleton_template": resolved.skeleton_template,
    }

    if args.decisions:
        vertical = None if args.writing_mode == "auto" else args.writing_mode == "vertical"
        report = run_from_decisions(
            fresh_package,
            args.decisions,
            folder,
            scope=args.scope,
            include_existing_edits=False,
            require_complete=args.require_complete,
            vertical=vertical,
            skeleton_template=resolved.skeleton_template or None,
        )
        payload = {"status": "completed", "source": source_report, "adjudication": report}
    else:
        report = run_codex_adjudication(
            fresh_package,
            folder,
            codex_bin=args.codex_bin,
            model=args.codex_model,
            reasoning_effort=args.reasoning_effort,
            scope=args.scope,
            context_rows=args.context_rows,
            include_existing_edits=False,
            require_complete=args.require_complete,
            writing_mode=args.writing_mode,
            timeout_seconds=args.codex_timeout,
            prepare_only=args.prepare_only,
            skeleton_template=resolved.skeleton_template or None,
        )
        payload = {"status": report.get("status", "unknown"), "source": source_report, "adjudication": report}
    (folder / "workspace_adjudication_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _json_print(payload)
    return 0 if payload["status"] == "completed" or args.prepare_only else 4


def _add_codex_options(parser):
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--codex-model", default="")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high", "xhigh"), default="high")
    parser.add_argument("--codex-timeout", type=int, default=7200)
    parser.add_argument("--scope", choices=("conflicts", "all"), default="conflicts")
    parser.add_argument("--context-rows", type=int, default=2)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--writing-mode", choices=("auto", "vertical", "horizontal"), default="auto")
    parser.add_argument("--prepare-only", action="store_true", help="只生成 fresh task，不调用 Codex")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m novel_formatter.cli",
        description="Novel Formatter 已完成 OCR 工作区 → fresh Codex 裁决 → skeleton EPUB（不运行 OCR/GUI）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("adjudicate", help="从已有 OCR 工作区/包重新裁决；忽略旧裁决并导出 skeleton EPUB")
    p.add_argument("source", help="Novel Formatter 项目目录、round-trip JSON 或融合/骨架 ZIP")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--decisions", help="已有 fresh decisions JSON；提供时不调用 Codex（测试/离线接力）")
    _add_codex_options(p)
    p.set_defaults(func=cmd_adjudicate)

    # Lower-level commands remain available for debugging/automation, but they
    # operate on an already prepared round-trip package only.
    p = sub.add_parser("prepare-codex", help="从 sealed 多模型包生成只读 Codex 裁决任务")
    p.add_argument("package")
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--prompt")
    p.add_argument("--decisions-name", default="codex_decisions.json")
    p.add_argument("--scope", choices=("conflicts", "all"), default="conflicts")
    p.add_argument("--context-rows", type=int, default=2)
    p.add_argument("--include-existing-edits", action="store_true")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("validate-decisions", help="只校验外部/Codex decisions，不写回")
    p.add_argument("package")
    p.add_argument("decisions")
    p.add_argument("--task")
    p.add_argument("--require-complete", action="store_true")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("apply-decisions", help="严格写回 resolved decisions；原始 OCR candidates 保持只读")
    p.add_argument("package")
    p.add_argument("decisions")
    p.add_argument("--task")
    p.add_argument("--output-package", required=True)
    p.add_argument("--output-document")
    p.add_argument("--unresolved")
    p.add_argument("--require-complete", action="store_true")
    p.set_defaults(func=cmd_apply)

    p = sub.add_parser("export-skeleton", help="从已裁决 round-trip 包导出 stable-ID skeleton EPUB")
    p.add_argument("package")
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--writing-mode", choices=("auto", "vertical", "horizontal"), default="auto")
    p.set_defaults(func=cmd_export_skeleton)

    p = sub.add_parser("run", help="已有 package + decisions → 校验/写回 → skeleton EPUB")
    p.add_argument("package")
    p.add_argument("decisions")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--scope", choices=("conflicts", "all"), default="conflicts")
    p.add_argument("--include-existing-edits", action="store_true")
    p.add_argument("--require-complete", action="store_true")
    p.add_argument("--writing-mode", choices=("auto", "vertical", "horizontal"), default="auto")
    p.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (HeadlessAdjudicationError, CodexRunnerError, FileNotFoundError, json.JSONDecodeError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
