#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export/import packages for pixel → stroke → Japanese character learning.

A learning package is deliberately editable outside the program. It contains:

* the exact segmented pixel image;
* a normalized 64×64 database image;
* the exact PKStroke point payload and a rendered trace preview when available;
* Apple Vision Top-N candidates, PKStrokeRecognizer output and fused candidates;
* an UTF-8-BOM CSV where the user fills ``confirmed_character`` and sets
  ``approved`` to 1.

Importing the edited package adds only approved rows to the local glyph-memory
SQLite database. Future exact matches are returned before Apple Vision or
PKStrokeRecognizer is invoked.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Any, Iterable, Sequence
import zipfile

from PIL import Image, ImageDraw, ImageOps

from adapters.unicode_safety import clean_json_value, clean_text, dumps as safe_json_dumps
from utils.safe_archive import safe_extract_zip

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEBUG_ROOT = _PROJECT_ROOT / "debug" / "glyph_learning"
_LATEST_FILE = _DEBUG_ROOT / "latest-session.json"


def _logical_glyph(value: Any) -> str:
    text = clean_text(value).strip()
    if not text:
        return ""
    result = text[0]
    import unicodedata
    for ch in text[1:]:
        if unicodedata.combining(ch) or "\ufe00" <= ch <= "\ufe0f":
            result += ch
        else:
            break
    return result


def _safe_filename(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in clean_text(value))
    return text[:80] or "glyph"


def _render_strokes(strokes: Sequence[Sequence[tuple[float, float]]], width: float, height: float) -> Image.Image:
    canvas = Image.new("RGB", (max(1, int(round(width))), max(1, int(round(height)))), "white")
    draw = ImageDraw.Draw(canvas)
    for stroke in strokes:
        if len(stroke) >= 2:
            draw.line([(float(x), float(y)) for x, y in stroke], fill="black", width=4, joint="curve")
    gray = ImageOps.grayscale(canvas)
    bbox = ImageOps.invert(gray).getbbox()
    if bbox:
        x0, y0, x1, y1 = bbox
        pad = 18
        canvas = canvas.crop((max(0, x0-pad), max(0, y0-pad), min(canvas.width, x1+pad), min(canvas.height, y1+pad)))
    return canvas


def _stroke_payload(strokes: Sequence[Sequence[tuple[float, float]]], width: float, height: float) -> dict[str, Any]:
    serialized = []
    for stroke_index, stroke in enumerate(strokes):
        points = []
        total = 0.0
        previous = None
        for x, y in stroke:
            current = (float(x), float(y))
            if previous is not None:
                total += ((current[0]-previous[0])**2 + (current[1]-previous[1])**2) ** 0.5
            previous = current
            points.append({
                "x": round(current[0], 4), "y": round(current[1], 4),
                "time": round(total/180.0, 4), "width": 3.2,
                "opacity": 1.0, "force": 0.55,
            })
        if len(points) >= 2:
            serialized.append({"id": f"stroke-{stroke_index+1}", "glyphIndex": 0, "points": points})
    return {
        "preferredLanguages": ["ja-JP"], "canvasWidth": float(width), "canvasHeight": float(height),
        "glyphCount": 1, "singleGlyphOnly": True, "playbackMode": "single_glyph_manual_equivalent",
        "pointInterval": 0.010, "strokeGap": 0.065, "strokes": serialized,
    }


def _csv_truthy(value: Any) -> bool:
    return clean_text(value).strip().lower() in {"1", "true", "yes", "y", "是", "已确认", "ok"}


@dataclass
class GlyphLearningSession:
    root: Path
    scope_key: str
    session_id: str
    records: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(cls, *, scope_key: str, label: str = "") -> "GlyphLearningSession":
        _DEBUG_ROOT.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        suffix = _safe_filename(label)[:28]
        session_id = f"session-{stamp}" + (f"-{suffix}" if suffix else "")
        root = _DEBUG_ROOT / session_id
        (root / "pixels").mkdir(parents=True, exist_ok=True)
        (root / "normalized").mkdir(parents=True, exist_ok=True)
        (root / "traces").mkdir(parents=True, exist_ok=True)
        (root / "payloads").mkdir(parents=True, exist_ok=True)
        return cls(root=root, scope_key=clean_text(scope_key) or "__global__", session_id=session_id)

    def record(
        self,
        *,
        image: Image.Image,
        page: int,
        column: int,
        glyph_index: int,
        segment: Any | None,
        decision: Any,
        apple_detail: dict[str, Any] | None = None,
        vision_groups: Any | None = None,
        is_symbol: bool = False,
    ) -> dict[str, Any]:
        from adapters.glyph_memory_db import normalize_glyph

        record_id = f"p{int(page):04d}-c{int(column):03d}-g{int(glyph_index):04d}"
        pixel_rel = Path("pixels") / f"{record_id}.png"
        norm_rel = Path("normalized") / f"{record_id}.png"
        image.convert("RGB").save(self.root / pixel_rel, format="PNG", compress_level=2)
        normalize_glyph(image).save(self.root / norm_rel, format="PNG", compress_level=3)

        apple_detail = dict(apple_detail or {})
        strokes = apple_detail.get("strokes") or []
        width = float(apple_detail.get("canvas_width") or 560.0)
        height = float(apple_detail.get("canvas_height") or 640.0)
        trace_rel = ""
        payload_rel = ""
        if strokes:
            trace_rel_path = Path("traces") / f"{record_id}.png"
            payload_rel_path = Path("payloads") / f"{record_id}.json"
            _render_strokes(strokes, width, height).save(self.root / trace_rel_path, format="PNG", compress_level=2)
            (self.root / payload_rel_path).write_text(
                safe_json_dumps(_stroke_payload(strokes, width, height), indent=2), encoding="utf-8"
            )
            trace_rel = trace_rel_path.as_posix()
            payload_rel = payload_rel_path.as_posix()

        candidates = []
        for candidate in getattr(decision, "candidates", []) or []:
            candidates.append({
                "text": clean_text(getattr(candidate, "text", "")),
                "score": round(float(getattr(candidate, "score", 0.0) or 0.0), 6),
                "source": clean_text(getattr(candidate, "source", "")),
                "reason": clean_text(getattr(candidate, "reason", "")),
            })
        def vision_items(group_name: str) -> list[dict[str, Any]]:
            if not isinstance(vision_groups, dict):
                return []
            output = []
            for item in vision_groups.get(group_name, []) or []:
                output.append({
                    "text": clean_text(getattr(item, "text", "")),
                    "confidence": round(float(getattr(item, "confidence", 0.0) or 0.0), 6),
                    "rank": int(getattr(item, "rank", 0) or 0),
                    "language_correction": bool(getattr(item, "language_correction", group_name == "corrected")),
                })
            return output

        corrected = vision_items("corrected")
        raw = vision_items("raw")
        if not corrected:
            corrected = [item for item in candidates if "apple_vision_corrected" in item["source"]]
        if not raw:
            raw = [item for item in candidates if "apple_vision_raw" in item["source"]]
        memory = [item for item in candidates if "glyph_memory" in item["source"]]
        chosen = _logical_glyph(getattr(decision, "chosen_char", ""))
        apple_result = _logical_glyph(apple_detail.get("result", ""))
        record = {
            "record_id": record_id,
            "scope_key": self.scope_key,
            "page": int(page), "column": int(column), "glyph_index": int(glyph_index),
            "segment_y0": int(getattr(segment, "y0", 0) or 0),
            "segment_y1": int(getattr(segment, "y1", 0) or 0),
            "segment_x0": int(getattr(segment, "x0", 0) or 0),
            "segment_x1": int(getattr(segment, "x1", 0) or 0),
            "ink_pixels": int(getattr(segment, "ink_pixels", 0) or 0),
            "pixel_file": pixel_rel.as_posix(), "normalized_file": norm_rel.as_posix(),
            "trace_file": trace_rel, "trace_payload_file": payload_rel,
            "apple_result": apple_result,
            "apple_error": clean_text(apple_detail.get("error", "")),
            "vision_corrected": corrected, "vision_raw": raw,
            "memory_candidates": memory, "fused_candidates": candidates,
            "output_character": chosen,
            "confirmed_character": "",
            "approved": "0",
            "is_symbol": bool(is_symbol),
            "reason": clean_text(getattr(decision, "reason", "")),
            "score": round(float(getattr(decision, "score", 0.0) or 0.0), 6),
        }
        self.records.append(clean_json_value(record))
        return record

    def finalize(self) -> dict[str, str]:
        manifest = {
            "format": "NovelFormatterGlyphLearningPackage",
            "version": 1,
            "session_id": self.session_id,
            "scope_key": self.scope_key,
            "created_at": datetime.now().isoformat(),
            "record_count": len(self.records),
            "records": self.records,
        }
        (self.root / "manifest.json").write_text(safe_json_dumps(manifest, indent=2), encoding="utf-8")
        fields = [
            "record_id", "page", "column", "glyph_index", "pixel_file", "normalized_file",
            "trace_file", "trace_payload_file", "apple_result", "output_character",
            "confirmed_character", "approved", "is_symbol", "reason", "score",
            "vision_corrected", "vision_raw", "memory_candidates", "fused_candidates",
            "scope_key", "segment_y0", "segment_y1", "segment_x0", "segment_x1", "ink_pixels",
        ]
        with (self.root / "glyphs.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record in self.records:
                row = dict(record)
                for name in ("vision_corrected", "vision_raw", "memory_candidates", "fused_candidates"):
                    row[name] = safe_json_dumps(row.get(name, []), ensure_ascii=False)
                writer.writerow({name: clean_text(row.get(name, "")) if isinstance(row.get(name), str) else row.get(name, "") for name in fields})
        readme = (
            "Novel Formatter 字形学习包\n\n"
            "1. 打开 glyphs.csv。\n"
            "2. 查看 pixel_file（原始单字像素）、trace_file（实际提交给 Apple 的轨迹）、"
            "apple_result 和 Vision 候选。\n"
            "3. 在 confirmed_character 填入正确字符，并把 approved 改为 1。\n"
            "4. 回到程序点击“导入已修改字形包”。\n"
            "未批准的行不会写入数据库。数据库命中后会直接输出，不再调用 Apple。\n"
        )
        (self.root / "README.txt").write_text(readme, encoding="utf-8")
        zip_path = self.root.with_suffix(".zip")
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(self.root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(self.root.parent))
        _LATEST_FILE.write_text(safe_json_dumps({"root": str(self.root), "zip": str(zip_path)}, indent=2), encoding="utf-8")
        # Keep only the newest 12 unpacked sessions/zips.
        sessions = sorted(_DEBUG_ROOT.glob("session-*"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in sessions[24:]:
            try:
                shutil.rmtree(old) if old.is_dir() else old.unlink()
            except Exception:
                pass
        return {"root": str(self.root), "zip": str(zip_path), "csv": str(self.root / "glyphs.csv")}


def latest_session_paths() -> dict[str, str]:
    try:
        payload = json.loads(_LATEST_FILE.read_text(encoding="utf-8"))
        return {key: str(value) for key, value in payload.items()}
    except Exception:
        return {}


def export_latest_session(destination: str | os.PathLike[str]) -> Path:
    latest = latest_session_paths()
    source = Path(latest.get("zip", ""))
    if not source.exists():
        raise FileNotFoundError("尚未生成字形学习包。请先运行一次日语字形候选融合 OCR。")
    target = Path(destination)
    if target.suffix.lower() != ".zip":
        target = target.with_suffix(".zip")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _locate_package_root(path: Path) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    if path.is_dir():
        if (path / "glyphs.csv").exists():
            return path, None
        matches = list(path.glob("*/glyphs.csv"))
        if matches:
            return matches[0].parent, None
    if path.suffix.lower() == ".zip":
        temp = tempfile.TemporaryDirectory(prefix="novel_formatter_glyph_import_")
        safe_extract_zip(path, temp.name)
        base = Path(temp.name)
        matches = list(base.rglob("glyphs.csv"))
        if not matches:
            temp.cleanup()
            raise ValueError("ZIP 中没有 glyphs.csv")
        return matches[0].parent, temp
    if path.name.lower() == "glyphs.csv":
        return path.parent, None
    raise ValueError("请选择字形学习包 ZIP、解压目录或 glyphs.csv")


def import_learning_package(
    path: str | os.PathLike[str], *, scope_override: str | None = None,
    also_global: bool = False, db=None,
) -> dict[str, Any]:
    from adapters.glyph_memory_db import GlyphMemoryDB
    database = db or GlyphMemoryDB()
    root, temp = _locate_package_root(Path(path))
    imported = skipped = errors = 0
    try:
        with (root / "glyphs.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                confirmed = _logical_glyph(row.get("confirmed_character", ""))
                if not confirmed or not _csv_truthy(row.get("approved", "")):
                    skipped += 1
                    continue
                pixel = root / clean_text(row.get("pixel_file", ""))
                if not pixel.exists():
                    errors += 1
                    continue
                scope = clean_text(scope_override or row.get("scope_key") or "__global__")
                trace_json = ""
                trace_path = root / clean_text(row.get("trace_payload_file", ""))
                if trace_path.exists():
                    trace_json = trace_path.read_text(encoding="utf-8", errors="ignore")
                try:
                    with Image.open(pixel) as image:
                        database.add_sample(
                            image.convert("RGB"), confirmed, scope_key=scope,
                            source="edited_learning_package", also_global=also_global,
                            apple_result=clean_text(row.get("apple_result", "")),
                            vision_candidates_json=clean_text(row.get("vision_corrected", "")) + clean_text(row.get("vision_raw", "")),
                            trace_json=trace_json,
                            is_symbol=_csv_truthy(row.get("is_symbol", "")),
                        )
                    imported += 1
                except Exception:
                    errors += 1
    finally:
        if temp is not None:
            temp.cleanup()
    return {"imported": imported, "skipped": skipped, "errors": errors, "database": str(database.path)}


def export_database_bundle(destination: str | os.PathLike[str], *, db=None) -> Path:
    from adapters.glyph_memory_db import GlyphMemoryDB
    database = db or GlyphMemoryDB()
    target = Path(destination)
    if target.suffix.lower() != ".zip":
        target = target.with_suffix(".zip")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="novel_formatter_glyph_db_export_") as temp_dir:
        root = Path(temp_dir) / "glyph-memory-database"
        root.mkdir(parents=True)
        backup = root / "glyph_memory.sqlite3"
        database.backup_to(backup)
        rows = database.export_rows()
        (root / "samples.json").write_text(safe_json_dumps(rows, indent=2), encoding="utf-8")
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for item in root.rglob("*"):
                if item.is_file():
                    archive.write(item, item.relative_to(root.parent))
    return target


def import_database_bundle(path: str | os.PathLike[str], *, db=None) -> dict[str, Any]:
    from adapters.glyph_memory_db import GlyphMemoryDB
    database = db or GlyphMemoryDB()
    source = Path(path)
    temp = None
    try:
        if source.suffix.lower() == ".zip":
            temp = tempfile.TemporaryDirectory(prefix="novel_formatter_glyph_db_import_")
            safe_extract_zip(source, temp.name)
            matches = list(Path(temp.name).rglob("glyph_memory.sqlite3"))
            if not matches:
                raise ValueError("ZIP 中没有 glyph_memory.sqlite3")
            source = matches[0]
        if source.suffix.lower() not in {".sqlite3", ".db"}:
            raise ValueError("请选择数据库 ZIP 或 glyph_memory.sqlite3")
        merged = database.merge_from(source)
        return {"imported": merged, "database": str(database.path)}
    finally:
        if temp is not None:
            temp.cleanup()
