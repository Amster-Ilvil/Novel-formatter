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
import time

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
            probe = subprocess.run(
                ["which", "shortcuts"], capture_output=True, text=True, timeout=5,
            )
            if probe.returncode != 0 or not str(probe.stdout or "").strip():
                return False, "找不到 shortcuts 命令，请确认 macOS ≥ Monterey (12)"
            return True, ""
        except Exception as exc:
            return False, f"无法检查 shortcuts 命令：{exc}"

    def recognize(self, image_path: str, config: OCRConfig) -> OCRResult:
        shortcut_name = str(config.shortcut_name or "").strip()
        if not shortcut_name:
            raise RuntimeError("Apple OCR 快捷指令名称为空，请填写 ExtractText 或实际快捷指令名称")
        cancel_check = getattr(self, "cancel_check", None)
        try:
            # 保持直接使用 backend 的旧调用兼容性；通过 OCR session 运行时会
            # 注入 cancel_check，此时改用可轮询的 Popen，停止按钮无需等待整段超时。
            if not callable(cancel_check):
                try:
                    result = subprocess.run(
                        ["shortcuts", "run", shortcut_name, "-i", image_path],
                        capture_output=True, text=True, timeout=max(1.0, float(config.timeout)),
                    )
                except subprocess.TimeoutExpired:
                    print(f"  ⚠️  超时: {os.path.basename(image_path)}")
                    return OCRResult(full_text="")
                stdout = result.stdout
                stderr = result.stderr
                returncode = int(result.returncode or 0)
            else:
                from adapters.subprocess_watchdog import isolated_process_kwargs, terminate_process
                process = subprocess.Popen(
                    ["shortcuts", "run", shortcut_name, "-i", image_path],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    **isolated_process_kwargs(),
                )
                deadline = time.monotonic() + max(1.0, float(config.timeout))
                while True:
                    if cancel_check():
                        terminate_process(process)
                        raise InterruptedError("Apple OCR 快捷指令已停止")
                    try:
                        stdout, stderr = process.communicate(timeout=0.25)
                        break
                    except subprocess.TimeoutExpired:
                        if time.monotonic() >= deadline:
                            terminate_process(process)
                            print(f"  ⚠️  超时: {os.path.basename(image_path)}")
                            return OCRResult(full_text="")
                returncode = int(process.returncode or 0)
        except FileNotFoundError:
            print("  ❌  找不到 shortcuts 命令，请确认 macOS ≥ Monterey (12)")
            sys.exit(1)

        if returncode != 0:
            detail = str(stderr or stdout or "未知错误").strip()
            raise RuntimeError(
                f"Apple OCR 快捷指令 {shortcut_name!r} 调用失败 "
                f"({os.path.basename(image_path)}): {detail}"
            )

        text = str(stdout or "").strip()
        # 快捷指令只返回拼接好的纯文本，没有逐条坐标/置信度——按行拆成
        # OCRBlock，bbox/confidence 用默认值，不伪造这个 backend 没有的精度。
        blocks = [OCRBlock(text=line) for line in text.splitlines() if line.strip()]
        return OCRResult(full_text=text, blocks=blocks)
