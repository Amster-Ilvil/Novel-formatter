# -*- coding: utf-8 -*-
"""Selectable OCR engine catalog shared by OCR and settings UI."""

import sys

OCR_ADAPTERS = [
    ("apple_vision", "Apple OCR", "macOS", "#4A3FA3",
     "Apple Vision 原生 OCR；Intel / Apple Silicon 本机按架构构建 Helper", True),
    ("windows_snipping_ocr", "Windows Snipping OCR", "Windows 10/11", "#2563EB",
     "复用系统截图工具 Snipping Tool 自带 OneOCR；本地离线，不下载或打包 Microsoft DLL/模型", True),
    ("paddle_ocr",   "PaddleOCR",        "跨平台", "#C0542F",
     "百度 PaddleOCR，坐标为像素值数组（首次使用会自动创建独立环境并下载模型）", True),
    ("paddle_aistudio", "PaddleOCR · AI Studio API", "云端", "#2B6CB0",
     "百度 AI Studio PaddleOCR 云端服务；支持模型页同步 API 与官方异步 v2 jobs，整页调用不走分栏扇出", True),
    ("ndlocr_lite",  "NDLOCR-Lite",      "图书OCR", "#0F766E",
     "国立国会图书馆的轻量图书/杂志 OCR；首次使用自动下载官方源码、ONNX 模型和独立依赖环境", True),
    ("manga_48px",   "48px AR OCR",       "漫画行识别", "#7C3AED",
     "Manga Image Translator 自回归漫画 OCR；文字列校正后缩放到 48px，Beam Search 逐字生成；首次使用下载约 195 MB 权重", True),
    ("hayai_ocr",    "Hayai OCR",    "高速 CJK", "#0F766E",
     "Hayai OCR（约 150M）；NaFlex 多行/竖排识别，支持批处理、MPS/CUDA/CPU 与 INT4/INT8；页面强制先物理分列", True),
]

def adapter_available_on_current_platform(adapter_id: str) -> bool:
    key = str(adapter_id or "").strip().lower()
    if key == "windows_snipping_ocr":
        return sys.platform == "win32"
    if key == "apple_vision":
        return sys.platform == "darwin"
    return True


__all__ = ["OCR_ADAPTERS", "adapter_available_on_current_platform"]
