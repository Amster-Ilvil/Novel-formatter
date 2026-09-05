#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistent, upstream-native findtextCenterNet worker.

Novel Formatter deliberately does *not* reimplement findtextCenterNet here.
The worker changes cwd/sys.path to the pinned upstream source tree, imports the
upstream ``run_ocr.py`` exactly as shipped, and reuses its already-created
``processer`` for all Smart-ROI images in this process.

Protocol (JSONL over stdin/stdout):
  request: {"request_id": 1, "images": ["/tmp/a.png", ...], "resize": 1.0}
  result:  {"request_id": 1, "items": [{"path":..., "ok":true,"payload":...}]}

All noisy upstream prints are redirected to stderr so stdout remains a clean
machine protocol.  The upstream sidecar ``<image>.json`` is read verbatim and
returned; its schema and Ruby semantics therefore remain upstream-owned.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from pathlib import Path


def _load_upstream(source_root: Path):
    source_root = source_root.resolve()
    os.chdir(source_root)
    sys.path.insert(0, str(source_root))
    # run_ocr.py selects CoreML -> ONNX -> Torch exactly as upstream defines and
    # instantiates OCR_Processer at module import.  Keep its banner/noise off the
    # JSONL protocol channel.
    with contextlib.redirect_stdout(sys.stderr):
        import run_ocr as upstream  # type: ignore
    models = list(getattr(upstream, "models", []) or [])
    if not models or not hasattr(upstream, "processer"):
        raise RuntimeError("上游 run_ocr.py 未能选择可用模型后端")
    return upstream, str(models[0])


def _process_one(upstream, image_path: str, resize: float) -> dict:
    path = Path(image_path).resolve()
    sidecar = Path(str(path) + ".json")
    sidecar.unlink(missing_ok=True)
    with contextlib.redirect_stdout(sys.stderr):
        upstream.processer.call_OCR(str(path), resize)
    if not sidecar.is_file():
        raise RuntimeError(f"上游未生成 JSON：{sidecar.name}")
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    finally:
        # Parent owns persistent caching.  Do not leave transient ROI sidecars
        # behind in Novel Formatter's temporary directory.
        sidecar.unlink(missing_ok=True)
    if not isinstance(payload, dict):
        raise RuntimeError("上游 JSON 顶层不是对象")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    args = parser.parse_args()
    source_root = Path(args.source_root)
    try:
        upstream, backend = _load_upstream(source_root)
    except Exception as exc:
        print(json.dumps({"type": "ready", "ready": False, "error": str(exc)}, ensure_ascii=False), flush=True)
        traceback.print_exc(file=sys.stderr)
        return 2

    print(json.dumps({
        "type": "ready",
        "ready": True,
        "backend": backend,
        "upstream": "lithium0003/findtextCenterNet",
    }, ensure_ascii=False), flush=True)

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            request = json.loads(raw)
        except Exception as exc:
            print(json.dumps({"type": "protocol_error", "error": str(exc)}, ensure_ascii=False), flush=True)
            continue
        if request.get("command") == "close":
            print(json.dumps({"type": "closed"}, ensure_ascii=False), flush=True)
            return 0
        request_id = int(request.get("request_id", 0) or 0)
        images = [str(item) for item in (request.get("images") or [])]
        try:
            resize = float(request.get("resize", 1.0) or 1.0)
        except Exception:
            resize = 1.0
        items = []
        for image_path in images:
            try:
                payload = _process_one(upstream, image_path, resize)
                items.append({"path": image_path, "ok": True, "payload": payload})
            except Exception as exc:
                items.append({"path": image_path, "ok": False, "error": str(exc)})
                traceback.print_exc(file=sys.stderr)
        print(json.dumps({
            "request_id": request_id,
            "items": items,
            "request_done": True,
        }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
