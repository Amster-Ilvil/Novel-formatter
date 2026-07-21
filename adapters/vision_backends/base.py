#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision Backend 抽象层。

Apple Vision OCR 有好几种拿到识别结果的方式（"快捷指令" App、纯 Python 直接
调用 Vision/VisionKit 框架……以后还可能有更多），它们都只是"怎么把一张图片
变成文字"这一件事的不同实现。apple_vision_adapter.py（页面分类 + 组装
UnifiedDocument）不应该关心具体是哪种实现——新增一种识别方式只需要新增一个
VisionBackend 子类并注册进 BackendFactory，不需要改动 Formatter/EPUB 任何代码。

对应 docs/SDS.md 第五章预留的 OCRProvider 接口（detect/recognize/
supports_vertical/supports_bbox），这里是它的第一次实际落地。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OCRBlock:
    """一条识别结果（对应"快捷指令"路线下的一行文字，或 Vision 框架的一个
    text observation）。bbox/confidence 在拿不到的 backend 上留空/默认值，
    不伪造精度。"""
    text: str
    confidence: float = 1.0
    bbox: Optional[tuple[float, float, float, float]] = None  # (x, y, w, h)，归一化坐标
    language: str = ""


@dataclass
class OCRResult:
    """单张图片的识别结果。full_text 是按阅读顺序拼好、可以直接喂给现有
    lines_to_blocks() 的多行文字（按行 split）；blocks 额外保留每条识别结果
    的坐标/置信度，供以后想用坐标信息的功能（版面分析、低置信度复核……）取用，
    现在没人用也不会丢失这份数据。

    page/rotation 默认 None，不是 0——None 明确表示"这个 backend 没有填这项
    信息"，跟"已经确认是第 0 页/旋转角度 0"是两回事，不能用假的默认值把
    "不知道"伪装成"知道"。page 通常由调用方（apple_vision_adapter.run() 的
    循环）在拿到结果后回填，backend 本身只认识一张图片，不知道自己是第几页。
    rotation 目前没有任何 backend 做旋转检测，一直是 None，等真正实现了
    再由对应 backend 填。"""
    full_text: str
    blocks: list[OCRBlock] = field(default_factory=list)
    language: str = ""
    rotation: Optional[float] = None
    page: Optional[int] = None


@dataclass
class OCRConfig:
    """OCR 识别参数——GUI/CLI 只需要构造一个 OCRConfig 传给 backend，不需要
    知道具体 backend 支持哪些参数（不支持的参数 backend 自己忽略）。"""
    recognition_level: str = "accurate"   # 目前只有 "accurate" 对日语/中文在这台机器上真正可用
    languages: list[str] = field(default_factory=lambda: ["ja-JP", "zh-Hans", "zh-Hant", "en-US"])
    vertical: bool = True                 # 竖排（右→左列/列内上→下）还是横排，决定阅读顺序重建方式
    shortcut_name: str = "ExtractText"    # backend="shortcut" 时使用
    timeout: float = 90.0


@dataclass
class BackendCapabilities:
    """某个 backend 实际支持什么——GUI/Pipeline 据此自动启用或隐藏对应功能，
    而不是在业务代码里为每个具体 backend 写 if/elif 特殊判断。新增 backend
    时按实际能力如实填写，没有的能力就是 False，不要为了好看谎报；没有实际
    实现过、只是从文档/调研里看到的"应该支持"，也要在 backend 自己的
    capabilities 里用注释注明是"文档声称、未实测"，不能当成确认过的能力。

    每个字段的含义：
        accurate/fast            是否能选择 Vision 的高精度/高速度识别档位
        bbox/confidence          逐条识别结果是否带坐标 / 是否有真实置信度
                                  （不是固定 1.0）
        language                 是否能显式指定/限定识别语言（不是"能不能
                                  识别日语"，是"能不能告诉它只识别日语"）
        language_correction      是否能开启语言模型对识别结果做拼写校正
        vertical_text            是否对竖排东亚文字有专门优化（不是"能识别
                                  竖排"，是"对竖排的阅读顺序/分行有专门处理"）
        searchable_pdf           能不能直接产出带可搜索文字层的 PDF
        pdf                      能不能直接处理 PDF 输入（不需要先转图片）
        batch                    是否原生支持批量/多图一次调用
    """
    accurate: bool = False
    fast: bool = False
    bbox: bool = False
    confidence: bool = False
    language: bool = False
    language_correction: bool = False
    vertical_text: bool = False
    searchable_pdf: bool = False
    pdf: bool = False
    batch: bool = False


class VisionBackend(ABC):
    """所有 OCR 识别方式的统一接口。"""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def capabilities(self) -> BackendCapabilities:
        """默认最保守（什么都没有）。有能力的 backend 自己覆写。"""
        return BackendCapabilities()

    @abstractmethod
    def recognize(self, image_path: str, config: OCRConfig) -> OCRResult:
        """识别单张图片，返回 OCRResult。失败时返回 full_text="" 的空结果，
        不抛异常中断整批处理（跟原来 extract_text_via_shortcut() 的容错行为一致）。"""
        ...

    def is_available(self) -> tuple[bool, str]:
        """检测这个 backend 当前环境下能不能用。返回 (可用, 不可用时的说明)。
        默认假设可用，需要额外依赖/外部程序的 backend 应该覆写这个方法。"""
        return True, ""
