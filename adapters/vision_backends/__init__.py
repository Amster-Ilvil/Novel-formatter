#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision Backend 注册表 + 工厂。

新增一种 OCR 识别方式：写一个 VisionBackend 子类，在这里注册一行，
apple_vision_adapter.py 和 GUI 都不需要改动。

两类 backend 都注册在同一个 _REGISTRY 里：
    - 真正实现、验证过可用的：shortcut / ocrmac-vision / ocrmac-livetext。
    - 占位 stub（stub_backends.py）：simpleocr / mac-ocr / seeV /
      ocrmypdf-appleocr——recognize() 会抛 NotImplementedError，
      is_available() 恒为 False（所以 auto() 永远不会选中它们），只是把
      "这个工具将来要不要接入"预先占好架构位置，真正实现时不需要改
      BackendFactory/apple_vision_adapter.py/GUI 任何一行，只需要把对应
      stub 类的 recognize()/capabilities()/is_available() 填成真的。
"""

from __future__ import annotations

from typing import Callable

from .base import VisionBackend, OCRResult, OCRBlock, OCRConfig, BackendCapabilities
from .shortcut_backend import ShortcutBackend
from .ocrmac_backend import OCRMacBackend
from .stub_backends import (
    SimpleOCRBackend, MacOCRBackend, SeeVBackend, OCRmyPDFAppleOCRBackend,
)

_REGISTRY: dict[str, Callable[[], VisionBackend]] = {
    "shortcut": ShortcutBackend,
    "ocrmac-vision": lambda: OCRMacBackend(framework="vision"),
    "ocrmac-livetext": lambda: OCRMacBackend(framework="livetext"),
    "simpleocr": SimpleOCRBackend,
    "mac-ocr": MacOCRBackend,
    "seeV": SeeVBackend,
    "ocrmypdf-appleocr": OCRmyPDFAppleOCRBackend,
}

# Auto 模式的优先级按"竖排还是横排"分两条不同的表——这不是随便猜的，是
# 用真实竖排轻小说页面实测出来的结果（2026-07-19 调试记录）：
#   VNRecognizeTextRequest 的 accurate 模式（ocrmac-vision）在这张竖排测试页
#   上只识别出标题（2 个 block），正文一整页完全没找到——不是置信度低，是
#   Vision 自己的文字区域检测阶段就没把正文切出观测结果，属于这个 API 对
#   竖排东亚文字排版的已知短板，不是本项目代码能修的。
#   ocrmac-livetext（VisionKit ImageAnalyzer）在同一张页面上能完整识别正文
#   （字符覆盖度跟"快捷指令"参考结果几乎一致，410 vs 412 字符，差异只是
#   两三个字符级的识别分歧，不是内容缺失）。
# 所以竖排场景（本项目的主要目标场景：日文/中文竖排轻小说）优先选
# livetext；横排场景 ocrmac-vision 已验证工作正常（有真实 confidence，
# livetext 没有），优先选它。shortcut 两种场景都排最后当保底。
_AUTO_PRIORITY_VERTICAL = ["ocrmac-livetext", "ocrmac-vision", "shortcut"]
_AUTO_PRIORITY_HORIZONTAL = ["ocrmac-vision", "shortcut", "ocrmac-livetext"]


class BackendFactory:

    @staticmethod
    def create(name: str, vertical: bool = True) -> VisionBackend:
        if name == "auto":
            return BackendFactory.auto(vertical=vertical)
        factory = _REGISTRY.get(name)
        if factory is None:
            available = ", ".join(_REGISTRY.keys())
            raise ValueError(f"未知 OCR Backend: {name!r}。可用: {available}")
        return factory()

    @staticmethod
    def auto(vertical: bool = True) -> VisionBackend:
        """按竖排/横排对应的优先级顺序挑第一个当前环境下可用的 backend。"""
        priority = _AUTO_PRIORITY_VERTICAL if vertical else _AUTO_PRIORITY_HORIZONTAL
        last_reason = "没有已注册的 backend"
        for name in priority:
            backend = _REGISTRY[name]()
            available, reason = backend.is_available()
            if available:
                return backend
            last_reason = reason
        raise RuntimeError(f"没有可用的 OCR backend（最后一个候选的原因：{last_reason}）")

    @staticmethod
    def available_backends() -> list[str]:
        return list(_REGISTRY.keys())
