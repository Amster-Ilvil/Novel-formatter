#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small persistent cache for optional findtextCenterNet Ruby ROI results.

The cache is deliberately outside normal OCR/fusion state.  Keys are derived
from the original source image bytes, ROI coordinates and runtime fingerprint;
values are upstream JSON payloads only.  Corrupt entries are ignored.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

_CACHE_LOCK = threading.RLock()
_SHA_LOCK = threading.RLock()
_SHA_MEMO: "OrderedDict[tuple[str, int, int], str]" = OrderedDict()
_SHA_MEMO_LIMIT = 64
SCHEMA = "findtext-ruby-cache-v1"


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    target = Path(path)
    stat = target.stat()
    key = (str(target.resolve()), int(stat.st_size), int(stat.st_mtime_ns))
    with _SHA_LOCK:
        cached = _SHA_MEMO.get(key)
        if cached is not None:
            _SHA_MEMO.move_to_end(key)
            return cached
    h = hashlib.sha256()
    with target.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    value = h.hexdigest()
    with _SHA_LOCK:
        _SHA_MEMO[key] = value
        _SHA_MEMO.move_to_end(key)
        while len(_SHA_MEMO) > _SHA_MEMO_LIMIT:
            _SHA_MEMO.popitem(last=False)
    return value


def runtime_fingerprint(source_dir: str | Path, *, upstream_commit: str) -> str:
    """Cheap backend-aware runtime identity without hashing huge model files."""
    source = Path(source_dir)
    parts = [str(upstream_commit)]
    # Mirror upstream run_ocr.py backend priority.  Include only the active
    # backend's filesystem identity so a completed CoreML/ONNX install does not
    # depend on absent Torch checkpoints.
    coreml = ("TextDetector.mlpackage", "TransformerEncoder.mlpackage", "TransformerDecoder.mlpackage")
    onnx = ("TextDetector.quant.onnx", "TransformerEncoder.onnx", "TransformerDecoder.onnx")
    torch = ("model.pt", "model3.pt")
    if all((source / name).is_dir() for name in coreml):
        parts.append("backend:coreml")
        names = coreml
    elif ((source / "TextDetector.quant.onnx").is_file() or (source / "TextDetector.onnx").is_file()) \
            and (source / "TransformerEncoder.onnx").is_file() and (source / "TransformerDecoder.onnx").is_file():
        parts.append("backend:onnx")
        names = onnx
    else:
        parts.append("backend:torch")
        names = torch
    for name in names:
        path = source / name
        try:
            stat = path.stat()
            parts.append(f"{name}:{stat.st_size}:{stat.st_mtime_ns}")
        except OSError:
            parts.append(f"{name}:missing")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]


def make_cache_key(
    *, page_digest: str, roi_box: tuple[int, int, int, int], runtime_id: str,
) -> str:
    raw = json.dumps(
        {"schema": SCHEMA, "page": page_digest, "roi": list(roi_box), "runtime": runtime_id},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class RubyResultCache:
    def __init__(self, root: str | Path, *, max_entries: int = 512, max_bytes: int = 768 * 1024 * 1024):
        self.root = Path(root)
        self.max_entries = max(16, int(max_entries))
        self.max_bytes = max(32 * 1024 * 1024, int(max_bytes))

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        path = self._path(key)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("cache_schema") != SCHEMA:
                return None
            data = payload.get("payload")
            if not isinstance(data, dict):
                return None
            try:
                os.utime(path, None)
            except OSError:
                pass
            return data
        except Exception:
            return None

    def put(self, key: str, payload: dict) -> None:
        if not isinstance(payload, dict):
            return
        path = self._path(key)
        tmp = path.with_suffix(".tmp")
        record = {
            "cache_schema": SCHEMA,
            "created": int(time.time()),
            "payload": payload,
        }
        with _CACHE_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(record, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            os.replace(tmp, path)

    def prune(self) -> None:
        with _CACHE_LOCK:
            try:
                files = [p for p in self.root.rglob("*.json") if p.is_file()]
            except OSError:
                return
            info = []
            total = 0
            for path in files:
                try:
                    stat = path.stat()
                except OSError:
                    continue
                total += stat.st_size
                info.append((stat.st_mtime_ns, stat.st_size, path))
            if len(info) <= self.max_entries and total <= self.max_bytes:
                return
            info.sort(key=lambda item: item[0])
            while info and (len(info) > self.max_entries or total > self.max_bytes):
                _mtime, size, path = info.pop(0)
                try:
                    path.unlink()
                    total -= size
                except OSError:
                    pass

    def clear(self) -> None:
        with _CACHE_LOCK:
            if not self.root.exists():
                return
            for path in sorted(self.root.rglob("*"), reverse=True):
                try:
                    if path.is_file() or path.is_symlink():
                        path.unlink()
                    elif path.is_dir():
                        path.rmdir()
                except OSError:
                    pass
