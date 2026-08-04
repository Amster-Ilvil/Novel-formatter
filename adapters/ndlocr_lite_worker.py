#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NDLOCR-Lite JSONL worker.

Loads the official checkout's detector and three PARSeq recognizers once, then
streams one normalized block list per input page.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def _pick_model(model_dir: Path, kind: str) -> Path:
    models = sorted(model_dir.glob("*.onnx"))
    if kind == "det":
        candidates = [p for p in models if "deim" in p.name.lower()]
    elif kind == "30":
        candidates = [p for p in models if "parseq" in p.name.lower() and ("-30-" in p.name or "30" in p.stem.split("-")[-3:])]
    elif kind == "50":
        candidates = [p for p in models if "parseq" in p.name.lower() and ("-50-" in p.name or "50" in p.stem.split("-")[-3:])]
    else:
        candidates = [
            p for p in models
            if "parseq" in p.name.lower() and "-30-" not in p.name and "-50-" not in p.name
        ]
    if not candidates:
        raise FileNotFoundError(f"NDLOCR-Lite 缺少 {kind} 模型: {model_dir}")
    return max(candidates, key=lambda p: p.stat().st_size)


def _quad_from_bbox(value):
    if not value:
        return None
    if isinstance(value, dict):
        if all(key in value for key in ("x", "y", "width", "height")):
            x1 = float(value["x"])
            y1 = float(value["y"])
            x2 = x1 + float(value["width"])
            y2 = y1 + float(value["height"])
            return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        if all(key in value for key in ("left", "top", "right", "bottom")):
            x1 = float(value["left"])
            y1 = float(value["top"])
            x2 = float(value["right"])
            y2 = float(value["bottom"])
            return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        for key in ("points", "vertices", "polygon"):
            if value.get(key):
                value = value[key]
                break
        else:
            return None
    if len(value) == 4 and isinstance(value[0], (int, float)):
        x1, y1, x2, y2 = value
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    points = []
    for point in value:
        if isinstance(point, dict) and "x" in point and "y" in point:
            points.append([float(point["x"]), float(point["y"])])
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            points.append([float(point[0]), float(point[1])])
    return points or None


def _load_runtime(root: Path):
    """Load the official NDLOCR detector/recognizers exactly once."""
    src = root / "src"
    sys.path.insert(0, str(src))
    try:
        import numpy as np
        from PIL import Image
        import ocr as ndlocr
    except Exception as exc:
        raise RuntimeError(f"NDLOCR-Lite 导入失败: {exc}") from exc

    try:
        model_dir = src / "model"
        config_dir = src / "config"
        ns = SimpleNamespace(
            det_weights=str(_pick_model(model_dir, "det")),
            det_classes=str(config_dir / "ndl.yaml"),
            det_score_threshold=0.2,
            det_conf_threshold=0.25,
            det_iou_threshold=0.2,
            rec_weights30=str(_pick_model(model_dir, "30")),
            rec_weights50=str(_pick_model(model_dir, "50")),
            rec_weights=str(_pick_model(model_dir, "100")),
            rec_classes=str(config_dir / "NDLmoji.yaml"),
            device="cpu",
            enable_tcy=False,
            simple_mode=False,
        )
        detector = ndlocr.get_detector(ns)
        recognizer100 = ndlocr.get_recognizer(args=ns)
        recognizer30 = ndlocr.get_recognizer(args=ns, weights_path=ns.rec_weights30)
        recognizer50 = ndlocr.get_recognizer(args=ns, weights_path=ns.rec_weights50)
    except Exception as exc:
        raise RuntimeError(f"NDLOCR-Lite 模型初始化失败: {exc}") from exc

    run_page = getattr(ndlocr, "_run_ocr_on_image_array", None)
    if run_page is None:
        raise RuntimeError(
            "当前 NDLOCR-Lite 版本缺少页面推理 API，请删除 .ocr-runtimes/ndlocr-lite 后重试"
        )
    return np, Image, run_page, detector, recognizer30, recognizer50, recognizer100


def _recognize_one(path: str, *, root: Path, runtime) -> dict:
    np, Image, run_page, detector, recognizer30, recognizer50, recognizer100 = runtime
    try:
        with Image.open(path) as opened:
            img = np.array(opened.convert("RGB"))
        result = run_page(
            detector=detector,
            recognizer30=recognizer30,
            recognizer50=recognizer50,
            recognizer100=recognizer100,
            inputname=Path(path).name,
            img=img,
            outputpath=str(root / ".novel_formatter_output"),
            save_viz=False,
        )
        rows = result.get("json_lines") or result.get("contents") or []
        blocks = []
        for row in rows:
            text = str(row.get("text") or row.get("STRING") or "").strip()
            if not text:
                continue
            blocks.append({
                "text": text,
                "confidence": float(row.get("confidence", row.get("conf", 0.9)) or 0.9),
                "box": _quad_from_bbox(row.get("boundingBox") or row.get("box")),
                "direction": "vertical" if str(row.get("isVertical", "")).lower() == "true" else "horizontal",
            })
        return {"ok": True, "path": path, "blocks": blocks}
    except Exception as exc:
        return {"ok": False, "path": path, "error": str(exc)}


def _serve(root: Path, runtime) -> None:
    """JSONL request server used by Novel Formatter's persistent OCR session.

    Compatibility note: newer clients call the terminal marker ``"batch_done"``;
    this worker retains Novel-formatter-1's ``type=request_done`` contract and
    the merged clients accept both without changing request boundaries.

    Request:  {"request_id": "...", "images": ["..."]}
    Response: one ordinary result per image followed by a request_done marker.
    """
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            request = json.loads(raw)
        except Exception as exc:
            print(json.dumps({"ok": False, "path": "", "error": f"请求解析失败: {exc}"}, ensure_ascii=False), flush=True)
            continue
        request_id = str(request.get("request_id") or "")
        if request.get("command") == "close":
            print(json.dumps({"type": "request_done", "request_id": request_id, "count": 0}, ensure_ascii=False), flush=True)
            return
        images = [str(item) for item in (request.get("images") or []) if str(item)]
        for path in images:
            payload = _recognize_one(path, root=root, runtime=runtime)
            payload["request_id"] = request_id
            print(json.dumps(payload, ensure_ascii=False), flush=True)
        print(json.dumps({
            "type": "request_done",
            "request_id": request_id,
            "count": len(images),
        }, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--server", action="store_true")
    parser.add_argument("images", nargs="*")
    args = parser.parse_args()

    root = Path(args.source_root).resolve()
    try:
        runtime = _load_runtime(root)
    except Exception as exc:
        print(json.dumps({"ok": False, "path": "", "error": str(exc)}, ensure_ascii=False), flush=True)
        raise SystemExit(1)

    if args.server:
        print(json.dumps({"type": "ready", "ok": True}, ensure_ascii=False), flush=True)
        _serve(root, runtime)
        return
    if not args.images:
        print(json.dumps({"ok": False, "path": "", "error": "没有输入图片"}, ensure_ascii=False), flush=True)
        raise SystemExit(2)
    for path in args.images:
        print(json.dumps(_recognize_one(path, root=root, runtime=runtime), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
