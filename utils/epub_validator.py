#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EPUB 构建后校验（W3C epubcheck，可选）。

epubcheck 是 EPUB 校验的事实标准，但需要 Java 运行时。策略：
- 检测不到可用的 Java（macOS 自带的 /usr/bin/java 常是无 JRE 的占位符，
  必须真正跑 `java -version` 验证）→ 返回 skipped，不打扰用户；
- pip 包 `epubcheck` 未安装 → 同样 skipped 并给出提示；
- 可用 → 运行校验，返回错误/警告列表。
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field


@dataclass
class EpubValidationReport:
    skipped: bool = False
    skip_reason: str = ""
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.skipped:
            return f"epubcheck 校验已跳过（{self.skip_reason}）"
        if self.valid and not self.warnings:
            return "epubcheck 校验通过 ✓"
        if self.valid:
            return f"epubcheck 校验通过，{len(self.warnings)} 条警告"
        return f"epubcheck 发现 {len(self.errors)} 个错误、{len(self.warnings)} 条警告"


def java_available(timeout: float = 10.0) -> bool:
    """真正执行 java -version 验证（占位符 java 会失败退出）。"""
    try:
        result = subprocess.run(
            ["java", "-version"], capture_output=True, timeout=timeout
        )
        return result.returncode == 0
    except Exception:
        return False


def validate_epub(epub_path: str, timeout: float = 180.0) -> EpubValidationReport:
    if not java_available():
        return EpubValidationReport(
            skipped=True,
            skip_reason="未检测到 Java 运行时；安装后即可自动启用（如 brew install temurin）",
        )
    try:
        from epubcheck import EpubCheck
    except ImportError:
        return EpubValidationReport(
            skipped=True,
            skip_reason="未安装 epubcheck 包（pip install epubcheck）",
        )

    try:
        result = EpubCheck(epub_path)
    except Exception as exc:
        return EpubValidationReport(skipped=True, skip_reason=f"epubcheck 运行失败: {exc}")

    report = EpubValidationReport(valid=bool(result.valid))
    for msg in getattr(result, "messages", []) or []:
        level = str(getattr(msg, "level", "")).upper()
        location = getattr(msg, "location", "") or ""
        line = getattr(msg, "line", None)
        loc = f"{location}:{line}" if line else str(location)
        entry = f"[{getattr(msg, 'id', '')}] {getattr(msg, 'message', msg)} ({loc})"
        if level in ("ERROR", "FATAL"):
            report.errors.append(entry)
        elif level == "WARNING":
            report.warnings.append(entry)
    return report
