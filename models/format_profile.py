#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Format Profile —— 用户自定义的排版格式。

对应用户诉求"提交参考 EPUB 学习格式、新建自己的格式、支持删除/导入导出"：
    - FormatProfile：单个格式的数据结构（可以来自参考 EPUB，也可以手写）
    - FormatProfileStore：纯文件持久化，一个 profile 一个 JSON 文件，
      和项目里 Repository 用目录+文件存版本历史的思路一致，不引入数据库。
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

PROFILES_DIR = Path.home() / ".novel_formatter" / "format_profiles"


@dataclass
class FormatProfile:
    name: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    vertical: bool = True
    css: str = ""
    dialogue_quote_style: str = "「」"
    paragraph_indent: str = "fullwidth_space"
    line_height: float = 1.8
    font_family: str = ""
    source: str = "manual"          # "manual" | "reference_epub"
    reference_name: str = ""        # 参考 EPUB 的文件名（source=reference_epub 时）
    notes: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "vertical": self.vertical,
            "css": self.css,
            "dialogue_quote_style": self.dialogue_quote_style,
            "paragraph_indent": self.paragraph_indent,
            "line_height": self.line_height,
            "font_family": self.font_family,
            "source": self.source,
            "reference_name": self.reference_name,
            "notes": self.notes,
            "created_at": self.created_at,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> FormatProfile:
        return cls(
            id=d.get("id", uuid.uuid4().hex[:12]),
            name=d.get("name", "未命名格式"),
            vertical=d.get("vertical", True),
            css=d.get("css", ""),
            dialogue_quote_style=d.get("dialogue_quote_style", "「」"),
            paragraph_indent=d.get("paragraph_indent", "fullwidth_space"),
            line_height=d.get("line_height", 1.8),
            font_family=d.get("font_family", ""),
            source=d.get("source", "manual"),
            reference_name=d.get("reference_name", ""),
            notes=d.get("notes", ""),
            created_at=d.get("created_at", time.time()),
        )

    @classmethod
    def from_json(cls, s: str) -> FormatProfile:
        return cls.from_dict(json.loads(s))

    @classmethod
    def from_reference_epub(cls, epub_path: str, name: str) -> FormatProfile:
        """从参考 EPUB 学习排版特征（CSS / XHTML / 结构分析）。"""
        import re
        import zipfile

        css_content = ""
        vertical = True
        dialogue_quote_style = "「」"
        paragraph_indent = "fullwidth_space"
        font_family = ""
        line_height = 1.8
        sample_title = ""

        def decode(raw):
            for enc in ("utf-8", "shift_jis", "euc-jp", "cp932"):
                try:
                    return raw.decode(enc).lstrip("\ufeff")
                except UnicodeDecodeError:
                    continue
            return raw.decode("utf-8", errors="ignore")

        try:
            with zipfile.ZipFile(epub_path, "r") as zf:
                css_chunks = []
                xhtml_files = [
                    n for n in zf.namelist()
                    if n.lower().endswith((".xhtml", ".html", ".htm"))
                ]

                # A. 独立 CSS
                for filename in zf.namelist():
                    if filename.lower().endswith(".css"):
                        try:
                            css = decode(zf.read(filename))
                            if len(css.strip()) > 10:
                                css_chunks.append(
                                    f"/* ===== {filename} ===== */\n{css}"
                                )
                        except Exception:
                            continue

                # B. <style> 标签
                inline_chunks = []
                # 不限制20页，避免轻小说前几页没有样式定义的问题
                for filename in xhtml_files:
                    try:
                        html = decode(zf.read(filename))
                        html = re.sub(
                            r"<!\[CDATA\[(.*?)\]\]>",
                            r"\1",
                            html,
                            flags=re.S
                        )

                        for match in re.findall(
                            r"<style[^>]*>(.*?)</style>",
                            html,
                            flags=re.I | re.S
                        ):
                            css = re.sub(r"<!--|-->", "", match).strip()
                            if css:
                                inline_chunks.append(
                                    f"/* ===== {filename} style ===== */\n{css}"
                                )

                        # C. style="" 行内样式
                        for style in re.findall(
                            r'\sstyle\s*=\s*["\'](.*?)["\']',
                            html,
                            flags=re.I | re.S
                        ):
                            if any(k in style for k in (
                                "text-indent",
                                "line-height",
                                "writing-mode",
                                "font",
                                "margin"
                            )):
                                inline_chunks.append(
                                    f"/* ===== {filename} inline ===== */\n"
                                    f"* {{{style}}}"
                                )

                    except Exception:
                        continue

                if not css_chunks:
                    css_chunks.extend(inline_chunks)
                elif inline_chunks:
                    css_chunks.extend(inline_chunks)

                # D. 结构推断 CSS
                structure_hints = []
                sample_html = ""
                for filename in xhtml_files[:30]:
                    try:
                        sample_html += decode(zf.read(filename))
                    except Exception:
                        pass

                if re.search(
                    r"writing-mode\s*:\s*(vertical|vertical-rl|tb-rl)",
                    sample_html,
                    re.I
                ):
                    vertical = True

                if "text-indent" in sample_html:
                    paragraph_indent = "css_text_indent"

                if "<ruby" in sample_html.lower():
                    structure_hints.append(
                        "/* EPUB contains ruby annotations */\nruby { ruby-position: over; }"
                    )

                if "tcy" in sample_html.lower():
                    structure_hints.append(
                        "/* EPUB contains tcy (vertical numbers) */"
                    )

                if structure_hints:
                    css_chunks.extend(structure_hints)

                css_content = "\n\n".join(css_chunks)

                # 从最终结果分析特征
                if css_content:
                    if re.search(
                        r"writing-mode\s*:\s*(vertical|vertical-rl|tb-rl)",
                        css_content,
                        re.I
                    ):
                        vertical = True

                    m = re.search(
                        r"font-family\s*:\s*([^;]+)",
                        css_content,
                        re.I
                    )
                    if m:
                        font_family = m.group(1).strip().strip('"')

                    m = re.search(
                        r"line-height\s*:\s*([0-9.]+)",
                        css_content,
                        re.I
                    )
                    if m:
                        try:
                            line_height = float(m.group(1))
                        except ValueError:
                            pass

                    if "text-indent" in css_content:
                        paragraph_indent = "css_text_indent"

                # OPF 标题
                for filename in zf.namelist():
                    if filename.lower().endswith(".opf"):
                        try:
                            opf = decode(zf.read(filename))
                            title = re.search(
                                r"<dc:title[^>]*>(.*?)</dc:title>",
                                opf,
                                re.I | re.S
                            )
                            if title:
                                sample_title = title.group(1).strip()
                            break
                        except Exception:
                            continue

                # 对话符号统计
                stats = {
                    "「」": len(re.findall(r"「[^」]*」", sample_html)),
                    "『』": len(re.findall(r"『[^』]*』", sample_html)),
                }
                if stats["『』"] > stats["「」"]:
                    dialogue_quote_style = "『』"

        except Exception:
            pass

        ref_name = Path(epub_path).name
        notes = (
            f"来源: {ref_name}"
            + (f"《{sample_title}》" if sample_title else "")
            + f" ・ {'竖排' if vertical else '横排'}"
            + (
                f" ・ CSS/结构已学习（{len(css_content)} 字符）"
                if css_content.strip()
                else " ・ ⚠️ 未发现可提取样式"
            )
        )

        return cls(
            name=name,
            vertical=vertical,
            css=css_content,
            dialogue_quote_style=dialogue_quote_style,
            paragraph_indent=paragraph_indent,
            line_height=line_height,
            font_family=font_family,
            source="reference_epub",
            reference_name=ref_name,
            notes=notes,
        )



class FormatProfileStore:
    """纯文件存储：~/.novel_formatter/format_profiles/<id>.json，一个 profile 一个文件。"""

    def __init__(self, base_dir: Path | str = PROFILES_DIR):
        self.base_dir = Path(base_dir)

    def _path(self, profile_id: str) -> Path:
        return self.base_dir / f"{profile_id}.json"

    def list(self) -> list[FormatProfile]:
        if not self.base_dir.exists():
            return []
        profiles = []
        for f in sorted(self.base_dir.glob("*.json")):
            try:
                profiles.append(FormatProfile.from_json(f.read_text(encoding="utf-8")))
            except Exception:
                continue
        return sorted(profiles, key=lambda p: p.created_at)

    def get(self, profile_id: str) -> FormatProfile | None:
        path = self._path(profile_id)
        if not path.exists():
            return None
        return FormatProfile.from_json(path.read_text(encoding="utf-8"))

    def get_by_name(self, name: str) -> FormatProfile | None:
        return next((p for p in self.list() if p.name == name), None)

    def save(self, profile: FormatProfile) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._path(profile.id).write_text(profile.to_json(), encoding="utf-8")

    def delete(self, profile_id: str) -> None:
        path = self._path(profile_id)
        if path.exists():
            path.unlink()

    def export_to(self, profile_id: str, dest_path: str) -> None:
        profile = self.get(profile_id)
        if profile is None:
            raise ValueError(f"格式不存在: {profile_id}")
        Path(dest_path).write_text(profile.to_json(), encoding="utf-8")

    def import_from(self, src_path: str) -> FormatProfile:
        profile = FormatProfile.from_json(Path(src_path).read_text(encoding="utf-8"))
        # 导入的 profile 换一个新 id，避免和本地已有的同 id 文件互相覆盖
        # （比如用户在两台机器上各自导出过同一个格式再导回来）。
        profile.id = uuid.uuid4().hex[:12]
        self.save(profile)
        return profile
