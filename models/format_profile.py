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

    @staticmethod
    def infer_css_from_xhtml_structure(html: str) -> tuple[str, list[str]]:
        """从 XHTML 的 class 命名约定与结构约定推断 CSS。

        很多正版 EPUB（尤其被去 DRM 工具清空过 CSS 的书）的排版信息不在
        样式表里，而是编码在 XHTML 的 class 名与标记结构中——电书协/出版社
        的命名本身就写明了样式：``start-5em`` = 块前空 5em、``font-1em20`` =
        1.20 倍字号、``fit`` = 插图自适应、``tcy`` = 縦中横。本函数按这些
        约定生成等效 CSS，并返回 (css, 结构说明列表)。
        """
        import re

        found: dict[str, int] = {}
        for m in re.finditer(r'class="([^"]+)"', html):
            for c in m.group(1).split():
                found[c] = found.get(c, 0) + 1

        rules: list[str] = []
        notes: list[str] = []

        def add(cls_name: str, css_body: str):
            rules.append(f".{cls_name} {{ {css_body} }}  /* ×{found[cls_name]} */")

        for c in sorted(found):
            if c in ("koboSpan", "main", "vrtl", "hltr", "p-text", "p-image"):
                continue  # 结构类，单独处理
            m = re.fullmatch(r"(start|end)-(\d+(?:em)?)", c)
            if m:
                side = "start" if m.group(1) == "start" else "end"
                val = m.group(2) if m.group(2).endswith("em") else m.group(2) + "em"
                add(c, f"margin-block-{side}: {val};")
                continue
            m = re.fullmatch(r"indent-(\d+)em", c)
            if m:
                add(c, f"text-indent: {m.group(1)}em;")
                continue
            m = re.fullmatch(r"font-(\d)em(\d\d)", c)
            if m:
                add(c, f"font-size: {m.group(1)}.{m.group(2)}em;")
                continue
            m = re.fullmatch(r"font-(\d+)per", c)
            if m:
                add(c, f"font-size: {int(m.group(1))}%;")
                continue
            if c == "fit":
                add(c, "width: auto; height: auto; max-width: 100%; max-height: 100%;")
                continue
            if c == "bold":
                add(c, "font-weight: bold;")
                continue
            if c in ("center", "align-center"):
                add(c, "text-align: center;")
                continue
            if c == "tcy":
                add(c, "-webkit-text-combine: horizontal; -epub-text-combine: horizontal; text-combine-upright: all;")
                continue

        # 结构类
        if "vrtl" in found or "hltr" in found:
            rules.insert(0, (
                "html.vrtl { -epub-writing-mode: vertical-rl; -webkit-writing-mode: vertical-rl; writing-mode: vertical-rl; }\n"
                "html.hltr { -epub-writing-mode: horizontal-tb; -webkit-writing-mode: horizontal-tb; writing-mode: horizontal-tb; }"
            ))
            notes.append(
                f"按页混排：竖排页 ×{found.get('vrtl', 0)}、横排页 ×{found.get('hltr', 0)}（html class 控制）"
            )
        if "main" in found:
            rules.append("div.main { margin: 3% 2%; }  /* 正文容器 */")
        if "p-image" in found:
            rules.append("body.p-image { margin: 0; padding: 0; }  /* 整页插图 */")
            notes.append(f"插图页模式：body.p-image + img.fit ×{found.get('fit', 0)}")

        fullwidth_indents = len(re.findall(r"<p[^>]*>　", html))
        if fullwidth_indents > 10:
            rules.append("p { margin: 0; padding: 0; text-indent: 0; }  /* 缩进用全角空格，正文不加 CSS 缩进 */")
            notes.append(f"段落缩进：正文用全角空格手动缩进（×{fullwidth_indents}），CSS 不应再加 text-indent")
        if "<ruby" in html.lower():
            rules.append("ruby rt { font-size: 0.5em; }")

        css = ""
        if rules:
            css = "/* ── 从 XHTML class 命名约定推断的样式 ── */\n" + "\n".join(rules)
        return css, notes

    @classmethod
    def from_reference_epub(cls, epub_path: str, name: str) -> FormatProfile:
        """从参考 EPUB 学习排版特征（CSS / XHTML / 结构分析）。"""
        import re
        import zipfile

        css_content = ""
        structure_notes: list[str] = []
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
                # 不限制20页，避免轻小说前几页没有样式定义的问题。
                # 同一段 <style> 常在每个 xhtml 里原样重复（如 Kobo 的
                # .koboSpan 样板），按规范化内容去重，只保留第一次出现。
                seen_inline: set[str] = set()
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
                                key = re.sub(r"\s+", " ", css)
                                if key in seen_inline:
                                    continue
                                seen_inline.add(key)
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
                                key = "inline:" + re.sub(r"\s+", " ", style)
                                if key in seen_inline:
                                    continue
                                seen_inline.add(key)
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
                for filename in xhtml_files[:300]:
                    try:
                        sample_html += decode(zf.read(filename))
                    except Exception:
                        pass

                # D2. XHTML class 命名约定 → CSS（排版编码在标记里的书，
                # 如被去 DRM 工具清空 CSS 的电书协 EPUB）
                inferred_css, structure_notes = cls.infer_css_from_xhtml_structure(sample_html)
                if inferred_css:
                    structure_hints.append(inferred_css)

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
        if structure_notes:
            notes += "\n结构约定：" + "；".join(structure_notes)

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
