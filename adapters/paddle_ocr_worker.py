#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PaddleOCR worker —— 独立运行于 .venv-paddle（Python 3.13）里的子进程脚本。

存在的原因：
    PaddlePaddle 目前没有 Python 3.14 的预编译包，而主项目（gui_pyside6.py 等）
    跑在系统默认的 3.14 上。这里用一个独立 venv（3.13）装 paddleocr，本脚本只用
    标准库 + paddleocr，不 import 项目里任何模块，这样才能被 3.14 侧的
    paddle_ocr_adapter.py 通过 subprocess 调用而不互相污染环境。

支持三种 pipeline（--pipeline 参数）：
    ocr        PaddleOCR —— 纯文字检测+识别，逐行输出，最快最省资源（默认）
    structure  PPStructureV3 —— 在 ocr 基础上加文档方向矫正/版面分析，实测
               它的 overall_ocr_res 字段和 PaddleOCR.predict() 输出结构完全一样，
               所以直接复用同一套解析逻辑，只是识别前多了一步版面预处理。
               需要额外依赖：pip install "paddlex[ocr]"
    vl         PaddleOCRVL —— 基于视觉语言模型的文档解析，直接输出按版面分好的
               大块内容（parsing_res_list，每块有 label/content/polygon_points），
               不是逐行输出。识别质量通常更好，但模型本体几个 GB，首次用会下载
               比较久。跳过表格/图片/公式类型的块，避免把非正文内容当文字块混进去。

协议：
    参数为若干图片路径。每处理完一张图，往 stdout 打印一行 JSON 并 flush，
    调用方（paddle_ocr_adapter.py）按行读取即可实现逐页进度回调。
    坐标是像素值（原图坐标系），不做归一化——归一化交给 3.14 侧（有 PIL）算。

    成功一行：
        {"ok": true, "path": "...", "blocks": [
            {"text": "...", "confidence": 0.98, "box": [[x,y],[x,y],[x,y],[x,y]]}, ...
        ]}
    失败一行：
        {"ok": false, "path": "...", "error": "..."}
"""

from __future__ import annotations

import argparse
import json
import os
import sys

try:
    # Direct worker execution adds adapters/ to sys.path.
    from paddle_ocr_models import (
        PADDLE_OCR_VERSION,
        PADDLE_DETECTION_MODEL,
        PADDLE_RECOGNITION_MODEL,
    )
except ImportError:  # Imported as adapters.paddle_ocr_worker in tests/tools.
    from adapters.paddle_ocr_models import (
        PADDLE_OCR_VERSION,
        PADDLE_DETECTION_MODEL,
        PADDLE_RECOGNITION_MODEL,
    )

# PaddleOCRVL parsing_res_list 里这些 label 不是正文文字块，直接跳过
_VL_SKIP_LABELS = {"table", "image", "figure", "chart", "formula", "seal"}


def _blocks_from_ocr_result(res) -> list[dict]:
    """PaddleOCR.predict() 单页结果 -> 统一 block 列表（ocr / structure 共用）"""
    blocks = []
    for text, score, poly in zip(res["rec_texts"], res["rec_scores"], res["rec_polys"]):
        if not text or not text.strip():
            continue
        blocks.append({
            "text": text,
            "confidence": float(score),
            "box": [[float(p[0]), float(p[1])] for p in poly],
        })
    return blocks


def _blocks_from_vl_result(res) -> list[dict]:
    """PaddleOCRVL.predict() 单页结果 -> 统一 block 列表"""
    blocks = []
    for item in res["parsing_res_list"]:
        if item.label in _VL_SKIP_LABELS:
            continue
        text = (item.content or "").strip()
        if not text:
            continue
        box = [[float(p[0]), float(p[1])] for p in item.polygon_points]
        blocks.append({
            "text": text,
            "confidence": 0.95,   # VL 结果不带逐块置信度，给个固定的高置信度
            "box": box,
            "label": item.label,
        })
    return blocks


def _medium_ocr_kwargs(lang: str) -> dict:
    """Common PP-OCRv6 medium model selection shared by OCR and Structure."""
    return dict(
        lang=lang,
        ocr_version=PADDLE_OCR_VERSION,
        text_detection_model_name=PADDLE_DETECTION_MODEL,
        text_recognition_model_name=PADDLE_RECOGNITION_MODEL,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
    )


def build_engine(
    pipeline: str,
    lang: str,
    *,
    vl_backend: str = "paddle",
    vl_server_url: str = "",
    vl_api_model_name: str = "PaddlePaddle/PaddleOCR-VL-1.6",
):
    if pipeline == "structure":
        from paddleocr import PPStructureV3
        # PP-Structure 保留版面分析，但文字检测与识别统一锁定到与普通 OCR
        # 相同的 PP-OCRv6 medium 模型，避免两个 Paddle 入口得到不同文字结果。
        common_kwargs = dict(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_table_recognition=False,
            use_formula_recognition=False,
            use_seal_recognition=False,
            use_chart_recognition=False,
            layout_detection_model_name="PP-DocLayout-S",
            text_detection_model_name=PADDLE_DETECTION_MODEL,
            text_recognition_model_name=PADDLE_RECOGNITION_MODEL,
        )
        try:
            engine = PPStructureV3(**common_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"无法加载 {PADDLE_DETECTION_MODEL} + {PADDLE_RECOGNITION_MODEL}。"
                "请运行 repair_runtime.command 或删除 .venv-paddle 后重试；"
                "程序不会再静默退回 PP-OCRv5。"
            ) from exc
        return engine, "structure"
    elif pipeline == "vl":
        from paddleocr import PaddleOCRVL
        # 显式锁定 v1.6，不依赖库的默认值。Apple Silicon 的 MLX 路径
        # 严格使用 PaddleOCR 官方提供的 mlx-vlm-server 接口；布局/预处理
        # 仍由 PaddleOCR 客户端负责，不改变后续 parsing_res_list 协议。
        backend = str(vl_backend or "paddle").strip().lower()
        if backend == "mlx":
            if not str(vl_server_url or "").strip():
                raise RuntimeError("已选择 MLX-VLM，但未提供本地服务地址")
            engine = PaddleOCRVL(
                pipeline_version="v1.6",
                vl_rec_backend="mlx-vlm-server",
                vl_rec_server_url=str(vl_server_url),
                vl_rec_api_model_name=(
                    str(vl_api_model_name or "PaddlePaddle/PaddleOCR-VL-1.6")
                ),
                device="cpu",
            )
            setattr(engine, "_novel_formatter_vl_backend", "mlx")
            return engine, "vl"
        engine = PaddleOCRVL(pipeline_version="v1.6")
        setattr(engine, "_novel_formatter_vl_backend", "paddle")
        return engine, "vl"
    else:
        from paddleocr import PaddleOCR
        try:
            engine = PaddleOCR(**_medium_ocr_kwargs(lang))
            setattr(engine, "_novel_formatter_model_profile", "ppocr_v6_medium")
        except Exception as exc:
            try:
                engine = PaddleOCR(
                    lang=lang,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=True,
                )
                setattr(engine, "_novel_formatter_model_profile", "paddle_default")
                print(
                    "PaddleOCR warning: "
                    f"{PADDLE_DETECTION_MODEL}+{PADDLE_RECOGNITION_MODEL} unavailable; "
                    "fallback to PaddleOCR default Japanese models.",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"无法加载 {PADDLE_DETECTION_MODEL} + {PADDLE_RECOGNITION_MODEL}，"
                    "且 PaddleOCR 默认日文模型也不可用。请检查网络、删除 .venv-paddle 后重试，"
                    "或改用 NDLOCR-Lite / Manga OCR。"
                ) from fallback_exc
        return engine, "ocr"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--probe", action="store_true", help="只初始化模型并报告状态")
    parser.add_argument("images", nargs="*", help="图片路径列表")
    parser.add_argument("--lang", default="japan")
    parser.add_argument("--pipeline", default="ocr", choices=["ocr", "structure", "vl"])
    parser.add_argument("--vl-backend", default="paddle", choices=["paddle", "mlx"])
    parser.add_argument("--vl-server-url", default="")
    parser.add_argument(
        "--vl-api-model-name", default="PaddlePaddle/PaddleOCR-VL-1.6"
    )
    args = parser.parse_args()

    try:
        engine, pipeline = build_engine(
            args.pipeline,
            args.lang,
            vl_backend=args.vl_backend,
            vl_server_url=args.vl_server_url,
            vl_api_model_name=args.vl_api_model_name,
        )
    except Exception as mlx_start_exc:
        if args.pipeline != "vl" or args.vl_backend != "mlx":
            raise
        # Compatibility safety net for a mismatched/older PaddleOCR install:
        # preserve OCR functionality rather than crashing the complete job.
        print(
            "PaddleOCR-VL MLX warning: 后端初始化失败，自动回退 Paddle："
            + str(mlx_start_exc),
            file=sys.stderr,
            flush=True,
        )
        engine, pipeline = build_engine("vl", args.lang, vl_backend="paddle")
        setattr(engine, "_novel_formatter_vl_backend", "paddle-fallback")

    if args.probe:
        print(json.dumps({
            "ok": True,
            "probe": True,
            "pipeline": pipeline,
            "model_source": os.environ.get("PADDLE_PDX_MODEL_SOURCE", "huggingface"),
            "model_profile": getattr(engine, "_novel_formatter_model_profile", pipeline),
            "vl_backend": getattr(engine, "_novel_formatter_vl_backend", ""),
        }, ensure_ascii=False), flush=True)
        return

    fallback_engine = None

    def process_path(path: str, request_id=None) -> dict:
        nonlocal engine, fallback_engine
        active_backend = getattr(engine, "_novel_formatter_vl_backend", "")
        try:
            try:
                results = engine.predict(path)
            except Exception as mlx_exc:
                # Runtime safety net: if the official MLX service fails after
                # startup (server/model/network edge case), retry the same page
                # with native Paddle and keep using it for the rest of this worker.
                if pipeline != "vl" or active_backend != "mlx":
                    raise
                print(
                    "PaddleOCR-VL MLX warning: 推理失败，当前页及后续页面自动回退 Paddle："
                    + str(mlx_exc),
                    file=sys.stderr,
                    flush=True,
                )
                if fallback_engine is None:
                    fallback_engine, _ = build_engine(
                        "vl", args.lang, vl_backend="paddle"
                    )
                engine = fallback_engine
                active_backend = "paddle-fallback"
                results = engine.predict(path)
            blocks = []
            for res in results:
                if pipeline == "vl":
                    blocks.extend(_blocks_from_vl_result(res))
                elif pipeline == "structure":
                    data = getattr(res, "json", None) or res
                    if isinstance(data, dict) and "res" in data:
                        data = data["res"]
                    if isinstance(data, dict) and data.get("overall_ocr_res"):
                        blocks.extend(_blocks_from_ocr_result(data["overall_ocr_res"]))
                    else:
                        blocks.extend(_blocks_from_ocr_result(res))
                else:
                    blocks.extend(_blocks_from_ocr_result(res))
            payload = {
                "ok": True,
                "path": path,
                "blocks": blocks,
                "backend": active_backend,
            }
        except Exception as exc:
            payload = {"ok": False, "path": path, "error": str(exc)}
        if request_id is not None:
            payload["request_id"] = request_id
        return payload

    if args.server:
        for line in sys.stdin:
            try:
                request = json.loads(line)
            except Exception:
                continue
            if request.get("command") == "close":
                break
            request_id = request.get("request_id")
            for path in list(request.get("images") or []):
                print(json.dumps(process_path(str(path), request_id), ensure_ascii=False), flush=True)
            print(json.dumps({"batch_done": True, "request_id": request_id}, ensure_ascii=False), flush=True)
    else:
        if not args.images:
            parser.error("至少提供一个图片路径，或使用 --server")
        for path in args.images:
            print(json.dumps(process_path(path), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
