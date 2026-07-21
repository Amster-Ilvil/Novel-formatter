#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
未接入的 OCR 工具占位 backend——注册进架构，但 recognize() 还没写，调用会
抛 NotImplementedError。目的是让"以后要不要支持这个工具"变成一个局部决定
（填 recognize()、按实测结果修正 capabilities、把 is_available() 换成真的
检测逻辑），而不需要改 BackendFactory/apple_vision_adapter.py/GUI 任何一行。

capabilities 里标 True 的字段，来源是这次对话过程里对各个项目 README/
issue/HN 帖的调研（细节见各个类自己的注释），**没有一项是靠实际跑通验证过
的**——seeV/mac-ocr(Swift) 在这台机器上连编译都失败（只有 Xcode Command Line
Tools，没有完整 Xcode），simpleocr 这个项目本身根本没找到。真正接入时（写
recognize()）必须重新核实一遍这些字段，不能直接当已确认的事实使用。
"""

from __future__ import annotations

from .base import VisionBackend, OCRResult, OCRConfig, BackendCapabilities


class SimpleOCRBackend(VisionBackend):
    """占位——"tobilg/simpleocr" 这个项目在 GitHub/PyPI/Homebrew 上都没找到，
    很可能是之前某份 AI 生成的工具对比表里的虚构项目。capabilities 全部留空：
    没有真实项目可参照，不编造一个不存在的工具"应该"有什么能力。
    如果哪天找到了真实存在的同名/同类项目，第一步是重新核实这个类的每一个
    字段，而不是直接把下面两个 backend 的模板抄过来。"""

    @property
    def name(self) -> str:
        return "simpleocr"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities()

    def is_available(self) -> tuple[bool, str]:
        return False, "尚未接入（找不到 simpleocr 这个项目，未核实是否真实存在）"

    def recognize(self, image_path: str, config: OCRConfig) -> OCRResult:
        raise NotImplementedError("simpleocr backend 尚未实现——先确认这个项目是否真实存在")


class MacOCRBackend(VisionBackend):
    """占位——对应 privatenumber/mac-ocr（npm 包，预编译二进制，用 Vision
    框架）。README 提到支持批量处理、生成可搜索 PDF，据此标 pdf/batch/
    searchable_pdf=True；bbox/confidence/language/accurate/fast 这次调研
    没查到具体细节，留 False。没有在这台机器上装过/跑过，上面几项也是文档
    转述，不是实测。"""

    @property
    def name(self) -> str:
        return "mac-ocr"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            pdf=True,             # 文档声称：可直接处理 PDF 输入（未实测）
            batch=True,           # 文档声称：可批量处理多张图/PDF（未实测）
            searchable_pdf=True,  # 文档声称：可生成可搜索 PDF（未实测）
        )

    def is_available(self) -> tuple[bool, str]:
        return False, "尚未接入（未对接 mac-ocr 的 CLI 输出格式）"

    def recognize(self, image_path: str, config: OCRConfig) -> OCRResult:
        raise NotImplementedError("mac-ocr backend 尚未实现——需要先对接它的 CLI 输出格式")


class SeeVBackend(VisionBackend):
    """占位——对应 Nexuist/seeV（Swift CLI，直接包 Vision 框架）。这次调研
    没有找到 JSON/bbox/confidence/自定义词汇的确认信息，capabilities 全部
    留空，不要因为它"是个 Vision 框架的封装"就假设它有这些能力。实测在这台
    机器上 `swift build -c release` 编译失败（只有 Command Line Tools，
    没有完整 Xcode.app）——即使写好了 recognize()，这台机器目前也跑不起来。"""

    @property
    def name(self) -> str:
        return "seeV"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities()

    def is_available(self) -> tuple[bool, str]:
        return False, "尚未接入（且这台机器上 swift build 编译 seeV 失败，缺完整 Xcode）"

    def recognize(self, image_path: str, config: OCRConfig) -> OCRResult:
        raise NotImplementedError("seeV backend 尚未实现——且这台机器目前编译不了 seeV")


class OCRmyPDFAppleOCRBackend(VisionBackend):
    """占位——对应 mkyt/OCRmyPDF-AppleOCR（OCRmyPDF 的插件，PyPI 包
    ocrmypdf-appleocr）。跟这里其它 backend 不同，它的定位是"给整份 PDF 加
    文字层"，不是"识别单张图返回文字"，接入时 recognize() 这个单图接口可能
    根本不适合它，更合理的做法是在 apple_vision_adapter 之外单独走一条
    "PDF → OCRmyPDF-AppleOCR → 带文字层的 PDF → pdf_text_layer.py 读取"的
    流程，而不是勉强套进这个单图 VisionBackend 接口——先占位注册，真正实现
    时要重新评估要不要走 VisionBackend 这条路。
    recognition_level 有 fast/accurate/livetext 三档、livetext 是 macOS 13+
    默认值，这几点在 GitHub issue/HN 帖子里确认过；vertical_text=True 也是
    经独立信源确认（livetext 模式对竖排东亚文字支持明显更好），标为 True。
    searchable_pdf=True 是它的核心功能，也标 True。其余未核实，留 False。"""

    @property
    def name(self) -> str:
        return "ocrmypdf-appleocr"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            accurate=True,        # 确认：recognition_level 支持 accurate
            fast=True,            # 确认：recognition_level 支持 fast（CJK 是否可用未知）
            vertical_text=True,   # 确认：livetext 模式对竖排东亚文字支持明显更好
            searchable_pdf=True,  # 确认：这是它的核心功能——给 PDF 加文字层
        )

    def is_available(self) -> tuple[bool, str]:
        return False, "尚未接入（且它是 PDF 级插件，接入前需评估是否适合单图 recognize() 接口）"

    def recognize(self, image_path: str, config: OCRConfig) -> OCRResult:
        raise NotImplementedError(
            "ocrmypdf-appleocr backend 尚未实现——它本质是 PDF 加文字层插件，"
            "接入前应先评估是否该走 pdf_text_layer.py 那条路径，而不是这个单图接口"
        )
