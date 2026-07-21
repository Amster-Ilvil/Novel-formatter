#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ocrmac Backend —— 纯 Python 直接调用 Apple Vision/VisionKit 框架
（pip install ocrmac），不需要"快捷指令" App 这层中间依赖。

两种 framework：
    "vision"    VNRecognizeTextRequest 的 accurate 模式。本项目的目标语言
                （日语/中文）在这台机器上只有 accurate 模式可用——fast 模式
                实测 language_preference 只接受西欧语言，日语/中文会被
                Vision 当成拉丁字符乱识别，所以这里不对外暴露 fast 模式。
    "livetext"  VisionKit 的 ImageAnalyzer（macOS 13+ 默认引擎），对竖排
                东亚文字的识别明显更好，但返回的是逐字符 observation、
                没有真实置信度（固定 1.0）、速度也更慢，建议先用 vision
                模式，效果不理想再切换。

依赖: pip install ocrmac
"""

from __future__ import annotations

import os

from .base import VisionBackend, OCRResult, OCRBlock, OCRConfig, BackendCapabilities


def _order_annotations(
    annotations: list[tuple[str, float, tuple[float, float, float, float]]],
    vertical: bool = True,
) -> list[str]:
    """
    把 Vision/VisionKit 返回的 (text, confidence, (x, y, w, h)) 列表按阅读顺序
    排好，拼成"每行一条"的文字列表返回。

    坐标系陷阱：Vision 框架的归一化坐标原点在图片左下角、y 轴向上（不是
    图形学常见的左上角原点、y 向下）——ocrmac 自己的 convert_coordinates_pil
    用 `1 - y` 做换算就是因为这个。所以"从上到下"必须按 y 降序排，不是升序。

    竖排日文（vertical=True）：先把 annotation 按中心 x 坐标聚类成"列"
    （不要求精确对齐，允许 ±1.2 倍典型宽度的误差——accurate 模式返回的是
    已经成句的文字块，不是逐字符，列内轻微的横向抖动很正常），列间按
    x 降序排列（右→左），列内按 y 降序排列（上→下），列内文字直接拼接
    （日文本身不用空格断词）。livetext 模式返回的是逐字符 annotation，
    同一套聚类逻辑一样适用，只是"列"的宽度会随字符本身的宽度自动收窄。

    横排（vertical=False）：按 y 降序（行从上到下）、同一行内按 x 升序
    （从左到右）直接排序，不需要聚类。
    """
    items = [(t, c, bbox) for t, c, bbox in annotations if t and t.strip()]
    if not items:
        return []

    if not vertical:
        items.sort(key=lambda it: (-it[2][1], it[2][0]))
        return [it[0] for it in items]

    widths = sorted(it[2][2] for it in items)
    median_w = widths[len(widths) // 2] or 0.02
    col_gap = median_w * 1.2

    items.sort(key=lambda it: -(it[2][0] + it[2][2] / 2))
    columns: list[list[tuple]] = []
    for it in items:
        cx = it[2][0] + it[2][2] / 2
        placed = False
        for col in columns:
            col_cx = col[0][2][0] + col[0][2][2] / 2
            if abs(col_cx - cx) < col_gap:
                col.append(it)
                placed = True
                break
        if not placed:
            columns.append([it])

    lines: list[str] = []
    for col in columns:
        col.sort(key=lambda it: -it[2][1])
        lines.append("".join(it[0] for it in col))
    return lines


class OCRMacBackend(VisionBackend):

    def __init__(self, framework: str = "vision"):
        self.framework = framework

    @property
    def name(self) -> str:
        return f"ocrmac-{self.framework}"

    @property
    def capabilities(self) -> BackendCapabilities:
        if self.framework == "livetext":
            return BackendCapabilities(
                accurate=False,   # livetext 没有 accurate/fast 档位可选，是单一模式
                fast=False,
                bbox=True,
                # confidence 固定返回 1.0（ocrmac 自己文档写的），不是真实
                # 置信度，如实标 False，不然低置信度复核之类的功能会被这种
                # 假 1.0 骗过去，永远触发不了。
                confidence=False,
                language=False,   # OCR(..., framework="livetext") 不接受 language_preference 参数
                language_correction=False,
                # VisionKit 的 ImageAnalyzer 是 livetext 模式存在的理由——
                # 对竖排东亚文字的识别/分行明显优于旧版 VNRecognizeTextRequest，
                # 这点在 OCRmyPDF-AppleOCR 的文档/issue 里有独立验证。
                vertical_text=True,
                searchable_pdf=False,
                pdf=False,
                batch=False,
            )
        return BackendCapabilities(
            accurate=True,      # 显式请求 recognition_level="accurate"
            fast=False,         # 本项目目标语言在这台机器上 fast 模式不可用，backend 不对外暴露
            bbox=True,
            confidence=True,    # accurate 模式返回真实置信度（0~1 连续值，非固定 1.0）
            language=True,      # 传 language_preference 列表，能显式限定识别语言
            language_correction=False,  # ocrmac 没有暴露 Vision 的 usesLanguageCorrection 参数
            vertical_text=False,  # VNRecognizeTextRequest 对竖排的阅读顺序没有专门优化，
                                   # 这正是本 backend 要自己做 _order_annotations 列聚类的原因
            searchable_pdf=False,
            pdf=False,
            batch=False,
        )

    def is_available(self) -> tuple[bool, str]:
        try:
            import ocrmac  # noqa: F401
            return True, ""
        except ImportError:
            return False, "未安装 ocrmac，请运行: pip install ocrmac"

    def recognize(self, image_path: str, config: OCRConfig) -> OCRResult:
        # 之前这里 import 失败会直接 sys.exit(1)——这是个致命 bug：
        # recognize() 是在 GUI/批处理的单张图片循环里被调用的，sys.exit(1)
        # 会连整个应用进程一起杀掉，而不只是让这一张图失败。改成跟
        # is_available() 一致的方式：抛异常，交给调用方（GUI worker 线程/
        # 批处理循环）按它们已有的 try/except 处理成"这一步失败"，不影响
        # 应用本身或其它已排队的图片。
        try:
            from ocrmac import ocrmac
        except ImportError as e:
            raise RuntimeError("未安装 ocrmac，请运行: pip install ocrmac") from e

        try:
            if self.framework == "livetext":
                annotations = ocrmac.OCR(image_path, framework="livetext").recognize()
            else:
                annotations = ocrmac.OCR(
                    image_path, recognition_level="accurate",
                    language_preference=config.languages,
                ).recognize()
        except Exception:
            # 之前这里只打印 str(e) 就吞掉异常返回空结果——看起来"识别失败"
            # 但真正的报错原因（比如语言码不合法、框架加载失败等）连堆栈都
            # 没留下，没法排查。这里打印完整 traceback 到控制台/日志，方便
            # 定位真实原因；但仍然返回空结果而不是往上抛——单张图片识别
            # 失败不应该中断整批 OCR，跟其它步骤"个别页出错不影响整批"的
            # 处理方式保持一致。
            import traceback
            print(f"  ⚠️  ocrmac 识别失败: {os.path.basename(image_path)}")
            traceback.print_exc()
            return OCRResult(full_text="")

        # 防御性检查：ocrmac 在某些异常情况下可能返回 None 或非预期结构
        # （而不是抛异常），不校验的话下面的解包/属性访问会直接崩溃，
        # 且崩溃点离真正原因（ocrmac 内部返回了什么）很远，不好排查。
        if not annotations:
            return OCRResult(full_text="")

        ordered_lines = _order_annotations(annotations, vertical=config.vertical)
        blocks = [
            OCRBlock(text=t, confidence=c, bbox=bbox)
            for t, c, bbox in annotations if t and t.strip()
        ]
        return OCRResult(full_text="\n".join(ordered_lines), blocks=blocks)
