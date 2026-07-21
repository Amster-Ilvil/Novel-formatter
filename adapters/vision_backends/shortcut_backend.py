#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shortcut Backend —— 调用 macOS"快捷指令" App 间接使用 Vision 框架。
项目最早的实现方式，保留作为不想装任何 Python 依赖时的兼容方案。
"""

from __future__ import annotations

import os
import subprocess
import sys

from .base import VisionBackend, OCRResult, OCRBlock, OCRConfig, BackendCapabilities


class ShortcutBackend(VisionBackend):

    @property
    def name(self) -> str:
        return "shortcut"

    @property
    def capabilities(self) -> BackendCapabilities:
        # "从图像中提取文字"这个快捷指令动作本身没有任何可配置参数——不能
        # 指定 accurate/fast、不能指定语言、不能开语言校正，只返回拼接好的
        # 纯文本，没有坐标/置信度/PDF/批量能力。即使 Vision 框架内部可能默认
        # 用了高精度模式，这个 backend 也没有办法验证或者控制它，所以不能
        # 报 accurate=True——capability 说的是"这个 backend 能不能保证/
        # 选择"，不是"底层框架有没有可能这么做"。全部字段如实留 False。
        return BackendCapabilities()

    def is_available(self) -> tuple[bool, str]:
        try:
            subprocess.run(["which", "shortcuts"], capture_output=True, timeout=5)
            return True, ""
        except Exception:
            return False, "找不到 shortcuts 命令，请确认 macOS ≥ Monterey (12)"

    def recognize(self, image_path: str, config: OCRConfig) -> OCRResult:
        try:
            result = subprocess.run(
                ["shortcuts", "run", config.shortcut_name, "-i", image_path],
                capture_output=True, text=True, timeout=config.timeout,
            )
        except subprocess.TimeoutExpired:
            print(f"  ⚠️  超时: {os.path.basename(image_path)}")
            return OCRResult(full_text="")
        except FileNotFoundError:
            print("  ❌  找不到 shortcuts 命令，请确认 macOS ≥ Monterey (12)")
            sys.exit(1)

        if result.returncode != 0:
            print(f"  ⚠️  识别失败: {os.path.basename(image_path)}")
            if result.stderr:
                print(f"      {result.stderr.strip()}")
            return OCRResult(full_text="")

        text = result.stdout.strip()
        # 快捷指令只返回拼接好的纯文本，没有逐条坐标/置信度——按行拆成
        # OCRBlock，bbox/confidence 用默认值，不伪造这个 backend 没有的精度。
        blocks = [OCRBlock(text=line) for line in text.splitlines() if line.strip()]
        return OCRResult(full_text=text, blocks=blocks)
