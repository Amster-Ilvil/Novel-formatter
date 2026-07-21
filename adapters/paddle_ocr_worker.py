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
import sys

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


def build_engine(pipeline: str, lang: str):
    if pipeline == "structure":
        from paddleocr import PPStructureV3
        # 默认配置会同时加载 7 个模型（含文档方向矫正、UVDoc 展平、layout
        # "plus-L" 大模型、OCRv5 "server" 级文字检测/识别），在内存有限的机器
        # 上很容易被系统直接杀掉。这里关掉不需要的方向矫正/展平，版面检测和
        # 文字检测/识别都换成轻量级（S / mobile）模型，减少同时常驻内存的模型数量。
        rec_model = f"{lang}_PP-OCRv3_mobile_rec" if lang == "japan" else "PP-OCRv5_mobile_rec"
        engine = PPStructureV3(
            use_doc_orientation_classify=False, use_doc_unwarping=False,
            use_table_recognition=False, use_formula_recognition=False,
            use_seal_recognition=False, use_chart_recognition=False,
            layout_detection_model_name="PP-DocLayout-S",
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name=rec_model,
        )
        return engine, "structure"
    elif pipeline == "vl":
        from paddleocr import PaddleOCRVL
        # 显式锁定 v1.6，不依赖库的默认值（避免以后 paddleocr 升级默认版本时
        # 悄悄换成别的模型）。
        return PaddleOCRVL(pipeline_version="v1.6"), "vl"
    else:
        from paddleocr import PaddleOCR
        engine = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
        )
        return engine, "ocr"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", help="图片路径列表")
    parser.add_argument("--lang", default="japan")
    parser.add_argument("--pipeline", default="ocr", choices=["ocr", "structure", "vl"])
    args = parser.parse_args()

    engine, pipeline = build_engine(args.pipeline, args.lang)

    for path in args.images:
        try:
            results = engine.predict(path)
            blocks = []
            for res in results:
                if pipeline == "vl":
                    blocks.extend(_blocks_from_vl_result(res))
                elif pipeline == "structure":
                    blocks.extend(_blocks_from_ocr_result(res["overall_ocr_res"]))
                else:
                    blocks.extend(_blocks_from_ocr_result(res))
            print(json.dumps({"ok": True, "path": path, "blocks": blocks}, ensure_ascii=False))
        except Exception as e:
            print(json.dumps({"ok": False, "path": path, "error": str(e)}, ensure_ascii=False))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
