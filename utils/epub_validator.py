#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional official EPUBCheck validation bridge.

Novel Formatter's built-in quality gate always runs.  This module adds the W3C/
DAISY EPUBCheck pass when a Java runtime and validator are available, without
network downloads or silent tool updates.

Resolution order:
1. an explicit ``NOVEL_FORMATTER_EPUBCHECK_JAR``/``EPUBCHECK_JAR``;
2. a bundled ``tools/epubcheck/epubcheck.jar`` if a distributor ships one;
3. the installed Python ``epubcheck`` wrapper (which bundles upstream EPUBCheck).

The project publishes against stable EPUB 3.3. EPUBCheck 5.4+ may additionally
report EPUB 3.4 Candidate Recommendation observations; these are surfaced, not
used to silently rewrite publication content.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from dataclasses import dataclass, field


@dataclass
class EpubValidationReport:
    skipped: bool = False
    skip_reason: str = ""
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    infos: list[str] = field(default_factory=list)
    engine: str = ""
    version: str = ""
    spec_target: str = ""

    def summary(self) -> str:
        suffix = ""
        if self.version:
            suffix = f" · EPUBCheck {self.version}"
        if self.skipped:
            return f"EPUBCheck 校验已跳过（{self.skip_reason}）"
        if self.valid and not self.warnings:
            return f"EPUBCheck 校验通过 ✓{suffix}"
        if self.valid:
            return f"EPUBCheck 校验通过，{len(self.warnings)} 条警告{suffix}"
        return f"EPUBCheck 发现 {len(self.errors)} 个错误、{len(self.warnings)} 条警告{suffix}"


def java_available(timeout: float = 10.0) -> bool:
    """Actually execute ``java -version`` (macOS may expose a placeholder)."""
    try:
        result = subprocess.run(["java", "-version"], capture_output=True, timeout=timeout)
        return result.returncode == 0
    except Exception:
        return False


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", str(version or ""))[:3])


def _spec_target(version: str) -> str:
    if _version_tuple(version) >= (5, 4):
        return "EPUB 3.4 Candidate Recommendation rules (publication target remains EPUB 3.3)"
    return "EPUB 3.3"


def _jar_candidates() -> list[Path]:
    paths: list[Path] = []
    for key in ("NOVEL_FORMATTER_EPUBCHECK_JAR", "EPUBCHECK_JAR"):
        value = str(os.environ.get(key, "") or "").strip()
        if value:
            paths.append(Path(value).expanduser())
    project_root = Path(__file__).resolve().parent.parent
    paths.append(project_root / "tools" / "epubcheck" / "epubcheck.jar")
    seen: set[str] = set()
    out: list[Path] = []
    for candidate in paths:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            out.append(candidate)
    return out


def _jar_version(jar: Path, timeout: float = 20.0) -> str:
    try:
        result = subprocess.run(
            ["java", "-jar", str(jar), "--version"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        text = (result.stdout or "") + "\n" + (result.stderr or "")
        match = re.search(r"(?:EPUBCheck\s*)?v?(\d+\.\d+(?:\.\d+)?)", text, re.I)
        return match.group(1) if match else ""
    except Exception:
        return ""


def _message_location(message: dict) -> str:
    locations = message.get("locations") or []
    if isinstance(locations, dict):
        locations = [locations]
    if not isinstance(locations, list) or not locations:
        return ""
    loc = locations[0] if isinstance(locations[0], dict) else {}
    path = str(loc.get("path") or loc.get("url") or "")
    line = loc.get("line")
    column = loc.get("column")
    bits = [path] if path else []
    if isinstance(line, int) and line >= 0:
        bits.append(str(line))
        if isinstance(column, int) and column >= 0:
            bits[-1] += f":{column}"
    return ":".join(bits)


def _run_jar(epub_path: str, jar: Path, timeout: float) -> EpubValidationReport:
    version = _jar_version(jar)
    with tempfile.TemporaryDirectory(prefix="novel-formatter-epubcheck-") as td:
        report_path = Path(td) / "report.json"
        try:
            proc = subprocess.run(
                ["java", "-jar", str(jar), str(epub_path), "--json", str(report_path), "--warn"],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except Exception as exc:
            return EpubValidationReport(skipped=True, skip_reason=f"EPUBCheck JAR 运行失败: {exc}")
        if not report_path.is_file():
            detail = (proc.stderr or proc.stdout or "").strip()
            return EpubValidationReport(
                skipped=True,
                skip_reason=f"EPUBCheck 未生成 JSON 报告{(': ' + detail[:300]) if detail else ''}",
            )
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return EpubValidationReport(skipped=True, skip_reason=f"EPUBCheck JSON 无法解析: {exc}")

    result = EpubValidationReport(
        valid=True,
        engine="official-jar",
        version=version,
        spec_target=_spec_target(version),
    )
    messages = data.get("messages") if isinstance(data, dict) else []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        level = str(msg.get("severity") or msg.get("level") or "").upper()
        ident = str(msg.get("ID") or msg.get("id") or "")
        message = str(msg.get("message") or "")
        location = _message_location(msg)
        entry = f"[{ident}] {message}" if ident else message
        if location:
            entry += f" ({location})"
        if level in {"ERROR", "FATAL"}:
            result.errors.append(entry)
        elif level == "WARNING":
            result.warnings.append(entry)
        elif entry:
            result.infos.append(entry)
    result.valid = not result.errors and proc.returncode == 0
    # EPUBCheck can occasionally fail before producing messages. Preserve the
    # process exit state as an error so an empty report never becomes a pass.
    if proc.returncode != 0 and not result.errors:
        detail = (proc.stderr or proc.stdout or "").strip()
        result.errors.append(detail[:1000] or f"EPUBCheck 退出码 {proc.returncode}")
        result.valid = False
    return result


def _wrapper_version() -> str:
    try:
        return importlib.metadata.version("epubcheck")
    except Exception:
        return ""


def _run_python_wrapper(epub_path: str) -> EpubValidationReport:
    try:
        from epubcheck import EpubCheck
    except ImportError:
        return EpubValidationReport(
            skipped=True,
            skip_reason="未找到可用的 EPUBCheck JAR / Python wrapper；可安装官方 EPUBCheck 5.4.x（项目仍以稳定 EPUB 3.3 为发布目标）",
        )

    version = _wrapper_version()
    try:
        result = EpubCheck(epub_path)
    except Exception as exc:
        return EpubValidationReport(skipped=True, skip_reason=f"epubcheck 运行失败: {exc}")

    report = EpubValidationReport(
        valid=bool(result.valid),
        engine="python-wrapper",
        version=version,
        spec_target=_spec_target(version),
    )
    for msg in getattr(result, "messages", []) or []:
        level = str(getattr(msg, "level", "")).upper()
        location = getattr(msg, "location", "") or ""
        line = getattr(msg, "line", None)
        loc = f"{location}:{line}" if line else str(location)
        entry = f"[{getattr(msg, 'id', '')}] {getattr(msg, 'message', msg)}"
        if loc:
            entry += f" ({loc})"
        if level in ("ERROR", "FATAL"):
            report.errors.append(entry)
        elif level == "WARNING":
            report.warnings.append(entry)
        else:
            report.infos.append(entry)
    if report.errors:
        report.valid = False
    return report


def validate_epub(epub_path: str, timeout: float = 180.0) -> EpubValidationReport:
    if not java_available():
        return EpubValidationReport(
            skipped=True,
            skip_reason="未检测到 Java 运行时；安装 Java 后即可启用官方 EPUBCheck",
        )

    jars = _jar_candidates()
    if jars:
        return _run_jar(epub_path, jars[0], timeout)
    return _run_python_wrapper(epub_path)
