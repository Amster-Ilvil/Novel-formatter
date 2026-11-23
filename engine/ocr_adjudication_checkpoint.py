"""Store a shared compressed OCR baseline and small, standalone run deltas."""
from __future__ import annotations

import gzip
import io
import json
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from engine.ocr_roundtrip_package import (
    RoundtripPackageError, _seal_package, _validate_immutable_manifest,
)
from utils.atomic_io import atomic_open, atomic_write_json

CHECKPOINT_SCHEMA = "novel_formatter.ocr_adjudication_checkpoint.v1"
_BASE_NAME = re.compile(r"ocr-base-([a-f0-9]{64})\.json\.gz")


def save_checkpoint(baseline: dict, reviewed: dict, report: dict, states, folder: Path, title: str):
    # Package IDs and export times are transport metadata. A stable identity
    # lets successive runs with the same immutable OCR share one baseline.
    base = {**baseline, "package_id": "ocr_adjudication_baseline", "created_at_unix": 0}
    _seal_package(base)
    digest = base["immutable_manifest_sha256"]
    base_path = folder / f"ocr-base-{digest}.json.gz"
    if not base_path.is_file():
        with atomic_open(base_path, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=4, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8") as stream:
                    json.dump(base, stream, ensure_ascii=False, separators=(",", ":"))

    from engine.ocr_compare_view_model import is_explicit_fusion_selection_origin

    before_items = base["editable_items"]
    after_items = reviewed["editable_items"]
    if len(before_items) != len(after_items) or len(before_items) != len(states):
        raise RoundtripPackageError("裁决保存行数与底稿不一致。")
    decisions = []
    uncertain = set(report.get("low_uncertain") or ())
    for before, after, state in zip(before_items, after_items, states):
        row_id = before["row_id"]
        if after["row_id"] != row_id:
            raise RoundtripPackageError("裁决保存行 ID 与底稿不一致。")
        row = {"row_id": row_id}
        if before["edited_text"] != after["edited_text"] or before["delete_intentionally"] != after["delete_intentionally"]:
            row.update(edited_text=after["edited_text"], delete_intentionally=after["delete_intentionally"])
        if row_id in uncertain or state.unresolved:
            row["unresolved"] = True
        elif is_explicit_fusion_selection_origin(state.selection_origin):
            row["selection_origin"] = state.selection_origin
        if len(row) > 1:
            decisions.append(row)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]
    report_path = folder / f"{title}_AI裁决报告_{stamp}.json"
    checkpoint_path = folder / f"{title}_AI裁决变更_{stamp}.json"
    atomic_write_json(report_path, report, indent=None)
    atomic_write_json(checkpoint_path, {
        "schema": CHECKPOINT_SCHEMA, "baseline": base_path.name, "decisions": decisions,
    }, indent=None)
    return checkpoint_path, report_path


def restore_checkpoint(checkpoint: dict, folder: Path) -> dict:
    name = checkpoint.get("baseline")
    match = _BASE_NAME.fullmatch(name) if isinstance(name, str) else None
    if match is None:
        raise RoundtripPackageError("裁决变更包的底稿文件名无效。")
    base_path = folder / name
    if base_path.is_symlink() or not base_path.is_file():
        raise RoundtripPackageError(f"找不到同目录下的裁决底稿：{name}")
    with gzip.open(base_path, "rt", encoding="utf-8") as stream:
        package = json.load(stream)
    _validate_immutable_manifest(package)
    if package.get("immutable_manifest_sha256") != match[1]:
        raise RoundtripPackageError("裁决变更包底稿校验失败。")
    entries = checkpoint.get("decisions")
    if not isinstance(entries, list):
        raise RoundtripPackageError("裁决变更包缺少变更记录。")
    by_id = {item["row_id"]: item for item in package["editable_items"]}
    authority = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - {
            "row_id", "edited_text", "delete_intentionally", "unresolved", "selection_origin",
        }:
            raise RoundtripPackageError("裁决变更包记录格式无效。")
        row_id = entry.get("row_id")
        if not isinstance(row_id, str) or row_id not in by_id or row_id in authority:
            raise RoundtripPackageError("裁决变更包包含未知或重复行 ID。")
        if ("edited_text" in entry) != ("delete_intentionally" in entry):
            raise RoundtripPackageError("裁决变更包缺少编辑或删除字段。")
        if "edited_text" in entry:
            if not isinstance(entry["edited_text"], str) or type(entry["delete_intentionally"]) is not bool:
                raise RoundtripPackageError("裁决变更包文字或删除标记无效。")
            by_id[row_id].update(edited_text=entry["edited_text"], delete_intentionally=entry["delete_intentionally"])
        origin = entry.get("selection_origin", "")
        unresolved = entry.get("unresolved", False)
        from engine.ocr_compare_view_model import is_explicit_fusion_selection_origin
        if type(unresolved) is not bool or not isinstance(origin, str) or (origin and (
            unresolved or not is_explicit_fusion_selection_origin(origin)
        )):
            raise RoundtripPackageError("裁决变更包的确认状态无效。")
        authority[row_id] = {"selection_origin": origin, "unresolved": unresolved}
    package["adjudication_checkpoint_authority"] = authority
    return _seal_package(package)
