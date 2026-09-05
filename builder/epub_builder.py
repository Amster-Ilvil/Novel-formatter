#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EPUB Builder
从 UnifiedDocument 生成 EPUB3 电子书。

支持：
    - EPUB3 + writing-mode: vertical-rl（竖排）
    - 封面图片（cover-image metadata）
    - 章节自动分文件
    - 插图按锚点插入正文（而非统一放到最后）
    - TOC 自动生成（nav.xhtml 进入 manifest，默认不进入 spine）
    - 元数据写入 content.opf（标题/作者/语言/ISBN）

依赖：
    pip install ebooklib

用法（命令行）：
    python epub_builder.py input.json output.epub
    python epub_builder.py input.json output.epub --template denki   # 使用電撃文庫模板
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import shutil
import sys
import tempfile
import uuid
import zipfile
from functools import wraps
from pathlib import Path
from textwrap import dedent

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.document import UnifiedDocument, Block, BlockType
from utils.atomic_io import atomic_write_text

def _workspace_for_output(output_path: str) -> Path:
    out = Path(output_path).expanduser()
    return out.parent / f"_epub_tmp_{out.stem}"


def _cleanup_workspace(func):
    """Remove stale and failed-build workspaces without masking build errors."""
    @wraps(func)
    def wrapper(doc, output_path: str, *args, **kwargs):
        workspace = _workspace_for_output(output_path)
        shutil.rmtree(workspace, ignore_errors=True)
        try:
            return func(doc, output_path, *args, **kwargs)
        finally:
            shutil.rmtree(workspace, ignore_errors=True)
    return wrapper


# ── CSS 模板 ─────────────────────────────────────────────────────────────────

CSS_TEMPLATES: dict[str, str] = {

    "denki": dedent("""\
        /* Novel Formatter Studio — 電撃文庫 竖排样式 */
        @charset "UTF-8";

        html {
            -epub-writing-mode: vertical-rl;
            writing-mode: vertical-rl;
            -webkit-writing-mode: vertical-rl;
        }
        body {
            margin: 5%;
            font-family: "ヒラギノ明朝 Pro", "Hiragino Mincho Pro",
                         "游明朝", "YuMincho", "MS 明朝", serif;
            line-height: 1.8;
        }
        h1 {
            font-weight: bold;
            margin-bottom: 2em;
            letter-spacing: 0.15em;
            text-align: center;
        }
        p {
            text-indent: 1em;
            margin: 0 0 0.4em 0;
            text-align: justify;
            -epub-text-align-last: justify;
        }
        p.dialogue {
            text-indent: 0;
        }
        p.section-break {
            text-indent: 0;
            text-align: center;
            margin: 1em 0;
        }
        /* 数字/英文竖排 */
        .tcy {
            -epub-text-combine: horizontal;
            -webkit-text-combine: horizontal;
            text-combine-upright: all;
        }
        /* Ruby 振假名 */
        ruby rt {
            -epub-ruby-position: right;
            ruby-position: over;
        }
        /* 插图页 */
        .illus-page {
            text-align: center;
            margin: 0;
            padding: 0;
        }
        .illus-page img {
            max-width: 100%;
            max-height: 95vh;
            object-fit: contain;
        }
        /* 封面 */
        .cover-page {
            margin: 0;
            padding: 0;
            text-align: center;
        }
        .cover-page img {
            width: 100%;
            height: 100vh;
            object-fit: contain;
        }
    """),

    "mf": dedent("""\
        /* Novel Formatter Studio — MF文庫J 竖排样式 */
        @charset "UTF-8";

        html {
            -epub-writing-mode: vertical-rl;
            writing-mode: vertical-rl;
        }
        body {
            margin: 4% 5%;
            font-family: "游明朝", "YuMincho", "ヒラギノ明朝 Pro", serif;
            line-height: 1.9;
        }
        h1 { font-weight: bold; letter-spacing: 0.1em; }
        p   { text-indent: 1em; margin: 0; }
        p.dialogue { text-indent: 0; }
        .tcy { text-combine-upright: all; }
    """),

    "web": dedent("""\
        /* Novel Formatter Studio — 横排 Web 小说样式 */
        @charset "UTF-8";

        body {
            margin: 2em auto;
            max-width: 680px;
            font-family: "ヒラギノ角ゴ Pro", "Hiragino Kaku Gothic Pro",
                         "Noto Sans CJK JP", sans-serif;
            line-height: 1.9;
        }
        h1 { margin: 1.5em 0 1em; }
        p   { text-indent: 1em; margin: 0 0 0.5em; }
        p.dialogue { text-indent: 0; }
    """),
}

DEFAULT_TEMPLATE = "denki"

# 第一个真正章节之前的内容（封面/扉页/目录扫描页/网站样板文字等）统一装进这个
# 标题固定的桶里。它不是一个真正的章节标题，不应该出现在读者可见的目录里。
FRONT_MATTER_TITLE = "前书页"

_EPUB_IMAGE_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
}


def _same_file(left: str | Path, right: str | Path) -> bool:
    try:
        return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()
    except Exception:
        return str(left) == str(right)


def _sniff_epub_image(src: Path) -> tuple[str, str] | None:
    """Return the real core image extension/MIME from file bytes.

    OCR/image tools frequently save PNG bytes under a .jpg/.jpeg filename.
    EPUB readers trust the manifest MIME and may reject that mismatch, so the
    builder must not rely on the source suffix alone.
    """
    try:
        head = src.read_bytes()[:32]
    except Exception:
        return None
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png", "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return ".jpg", "image/jpeg"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return ".gif", "image/gif"
    if head.lstrip().startswith(b"<svg") or b"<svg" in head.lower():
        return ".svg", "image/svg+xml"
    return None


def _optimize_png_lossless(path: Path) -> int:
    """无损重压 PNG（Pillow optimize，保留 ICC 配置）。

    像素数据不变；仅当输出比原文件小才替换。返回节省的字节数。
    借鉴 Calibre ebook-polish 的"最小改动"哲学：失败/变大都保留原文件。
    """
    try:
        import io
        from PIL import Image

        original = path.read_bytes()
        with Image.open(io.BytesIO(original)) as image:
            image.load()
            buf = io.BytesIO()
            save_kwargs: dict = {"format": "PNG", "optimize": True}
            icc = image.info.get("icc_profile")
            if icc:
                save_kwargs["icc_profile"] = icc
            image.save(buf, **save_kwargs)
        data = buf.getvalue()
        if len(data) < len(original):
            path.write_bytes(data)
            return len(original) - len(data)
    except Exception:
        pass
    return 0


def _copy_epub_image(
    src: Path,
    image_dir: Path,
    base_name: str,
    *,
    preserve_bytes: bool = False,
) -> tuple[str, str]:
    """Copy/convert an image to an EPUB core media type.

    Safe formats are selected from their real bytes, not merely their suffix.
    Everything else is converted to PNG with Pillow.
    """
    detected = _sniff_epub_image(src)
    if detected is not None:
        ext, media_type = detected
        filename = f"{base_name}{ext}"
        shutil.copy2(src, image_dir / filename)
        if ext == ".png" and not preserve_bytes:
            _optimize_png_lossless(image_dir / filename)
        return filename, media_type

    # Keep compatibility with already-core files whose tiny/test payload cannot
    # be sniffed.  Real PNG/JPEG mismatches were handled above by magic bytes.
    ext = src.suffix.lower()
    if ext in _EPUB_IMAGE_MIME:
        filename = f"{base_name}{ext}"
        shutil.copy2(src, image_dir / filename)
        return filename, _EPUB_IMAGE_MIME[ext]

    try:
        # Optional HEIC/HEIF support.  Importing registers an opener for Pillow.
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except Exception:
            pass
        from PIL import Image

        with Image.open(src) as image:
            image.load()
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGBA" if "transparency" in image.info else "RGB")
            filename = f"{base_name}.png"
            image.save(image_dir / filename, format="PNG", optimize=True)
        return filename, "image/png"
    except Exception as exc:
        raise ValueError(
            f"图片无法写入 EPUB：{src}\n"
            "EPUB 仅直接支持 JPG/PNG/GIF/SVG；HEIC/TIFF/BMP 需要 Pillow"
            "（HEIC 还需要 pillow-heif）进行转换。"
        ) from exc


def _css_is_usable(css: str) -> bool:
    css = str(css or "").strip()
    if not css or "{" not in css or "}" not in css:
        return False
    if css.count("{") != css.count("}"):
        return False
    # Reject truncated declarations such as the literal "-webkit-" that can
    # otherwise survive into the final stylesheet and invalidate a rule block.
    if re.search(r"(?m)^\s*-(?:webkit|epub)-?\s*$", css):
        return False
    for line in css.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("/*", "*", "@", "}", "{")):
            continue
        if stripped.endswith("-") and ":" not in stripped:
            return False
    return True


def document_ai_css(doc: UnifiedDocument) -> str:
    """Return only a structurally usable locked AI stylesheet."""
    meta = getattr(doc, "metadata", None)
    if not meta:
        return ""
    if getattr(meta, "ai_processing_mode", "") != "typeset":
        return ""
    if not bool(getattr(meta, "ai_layout_locked", False)):
        return ""
    css = str(getattr(meta, "ai_epub_css", "") or "").strip()
    return css if _css_is_usable(css) else ""


def resolve_epub_css(
    doc: UnifiedDocument,
    css_template: str = DEFAULT_TEMPLATE,
    custom_css: str | None = None,
) -> tuple[str, str]:
    """Resolve CSS without leaking AI styles into OCR/replacement/correction exports.

    Returns ``(css, source)`` where source is ``ai`` / ``custom`` / ``template``.
    """
    ai_css = document_ai_css(doc)
    if ai_css:
        return ai_css, "ai"
    if custom_css:
        return custom_css, "custom"
    return CSS_TEMPLATES.get(css_template, CSS_TEMPLATES[DEFAULT_TEMPLATE]), "template"


# ── XHTML 生成辅助 ────────────────────────────────────────────────────────────

def _xhtml_wrap(title: str, body_content: str, css_filename: str = "style.css") -> str:
    """
    注意：body_content 是多行、且自带缩进的变量内容（章节正文/nav 列表等），
    不能把它嵌进 f-string 以后再整体 dedent() —— dedent() 是按"所有行公共前导
    空白"来算的，body_content 内部的缩进（通常比外层模板浅）会把这个公共前
    缀拉低甚至拉到 0，导致 <?xml ...?> 声明前面残留缩进空格。XML 声明前不允
    许出现任何字符（哪怕是空格），这样生成的文件在严格的 XML 解析器（大多数
    正版 EPUB 阅读器/校验器都是）眼里就是畸形文档。
    所以要把"纯静态、缩进一致"的头尾模板单独 dedent，body_content 原样拼进去。
    """
    header = dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE html>
        <html xmlns="http://www.w3.org/1999/xhtml"
              xmlns:epub="http://www.idpf.org/2007/ops"
              xml:lang="ja">
        <head>
          <meta charset="UTF-8"/>
          <title>{_esc(title)}</title>
          <link rel="stylesheet" type="text/css" href="../styles/{css_filename}"/>
        </head>
        <body>
        """)
    footer = dedent("""\
        </body>
        </html>
    """)
    return header + body_content + "\n" + footer


def _esc(s: str) -> str:
    """XML/HTML 转义"""
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;"))


def _ruby_to_xhtml(text: str) -> str:
    """Convert supported ruby notations to XHTML without swallowing following prose.

    Preferred input is Aozora-style ``｜漢字《よみ》`` because the reading has an
    explicit closing delimiter.  Legacy ``漢字|よみ`` remains supported for old
    project JSON; its fallback parser restricts readings to kana and protects a
    likely following particle instead of greedily consuming the rest of a sentence.
    """
    import re

    value = str(text or "")
    aozora = re.compile(r"[｜|]([^《]+)《([^》]+)》")
    if aozora.search(value):
        parts: list[str] = []
        last = 0
        for m in aozora.finditer(value):
            if m.start() > last:
                parts.append(_esc(value[last:m.start()]))
            parts.append(f'<ruby>{_esc(m.group(1))}<rt>{_esc(m.group(2))}</rt></ruby>')
            last = m.end()
        if last < len(value):
            parts.append(_esc(value[last:]))
        return ''.join(parts)

    marker = re.compile(r"([^\s|]{1,24})\|([ぁ-ゖァ-ヺー]{1,32})")
    particles = set("をがにへとはもので")
    parts: list[str] = []
    last = 0
    for m in marker.finditer(value):
        reading = m.group(2)
        effective_end = m.end()
        following = value[effective_end:effective_end + 1]
        if (
            len(reading) >= 3
            and reading[-1] in particles
            and following
            and re.match(r"[一-龯々〆ヵヶァ-ヶーA-Za-z0-9０-９、。！？!?]", following)
        ):
            reading = reading[:-1]
            effective_end -= 1
        if not reading:
            continue
        if m.start() > last:
            parts.append(_esc(value[last:m.start()]))
        parts.append(f'<ruby>{_esc(m.group(1))}<rt>{_esc(reading)}</rt></ruby>')
        last = effective_end
    if last < len(value):
        parts.append(_esc(value[last:]))
    return ''.join(parts) if parts else _esc(value)


def _ruby_source_matches_current_text(marked: str, current_text: str) -> bool:
    """Reject stale Ruby markup that would resurrect superseded prose at export.

    ``ruby_aozora`` is a side-channel rendering of ``Block.text``. If any rebuild
    path ever forgets to refresh it, publication must prefer current authoritative
    prose without Ruby rather than silently exporting an older text snapshot.
    """
    if not re.search(r"[｜|][^《]+《[^》]+》", marked or ""):
        return False
    plain = re.sub(r"[｜|]([^《\n]+)《([^》\n]+)》", r"\1", str(marked or ""))
    return _sanitize_export_text(plain) == _sanitize_export_text(current_text or "")


def _ruby_export_source(block: Block, *, allow_ruby: bool = True) -> str:
    """Prefer Ruby markup only when explicitly allowed and prose is current."""
    metadata = block.metadata if isinstance(getattr(block, "metadata", None), dict) else {}
    if not allow_ruby:
        return _sanitize_export_text(str(block.text or ""))
    current = str(block.text or "")
    # Historical/imported BlockType.RUBY stores the authoritative structured
    # source in ``ocr_raw`` while ``text`` may use the legacy ``base|reading``
    # representation.  This is not the optional findtext side-channel and must
    # keep the pre-feature EPUB round-trip semantics.
    if block.type == BlockType.RUBY:
        structural = str(getattr(block, "ocr_raw", "") or "")
        if re.search(r"[｜|][^《]+《[^》]+》", structural):
            return _sanitize_export_text(structural)
    preserved = str(metadata.get("ruby_aozora") or "")
    if _ruby_source_matches_current_text(preserved, current):
        return _sanitize_export_text(preserved)
    source = str(getattr(block, "ocr_raw", "") or "")
    if _ruby_source_matches_current_text(source, current):
        return _sanitize_export_text(source)
    return _sanitize_export_text(current)


_ORPHAN_CLOSING_QUOTES = {"」", "』"}
_MIXED_ELLIPSIS_RE = re.compile(r"(?=[.．・…]{2,})(?=[.．・…]*[.．…])[.．・…]{2,}")

def _sanitize_export_text(text: str) -> str:
    """EPUB 最终导出防线：统一省略号并清掉首尾无意义空白。"""
    text = (text or "").strip(" \t\r\n")
    text = _MIXED_ELLIPSIS_RE.sub("……", text)
    return text


def _repair_plain_to_xhtml(text: str) -> str:
    """Preserve explicit OCR line breaks in AI-repair XHTML as <br/>."""
    return "<br/>".join(_esc(part) for part in str(text or "").split("\n"))


def _repair_ruby_to_xhtml(text: str) -> str:
    """Ruby conversion with the same explicit newline preservation."""
    return "<br/>".join(_ruby_to_xhtml(part) for part in str(text or "").split("\n"))

def _repair_attrs(b: Block) -> str:
    """Return opt-in stable AI-repair attributes without affecting normal EPUBs."""
    metadata = b.metadata if isinstance(getattr(b, "metadata", None), dict) else {}
    item_id = str(metadata.get("ai_repair_item_id", "") or "").strip()
    html_id = str(metadata.get("ai_repair_html_id", "") or "").strip()
    if not item_id or not html_id:
        return ""
    attrs = [
        f'id="{_esc(html_id)}"',
        f'data-item-id="{_esc(item_id)}"',
        f'data-page="{int(getattr(b, "page", 0) or 0)}"',
    ]
    content_format = str(metadata.get("ai_repair_content_format", "") or "").strip()
    if content_format:
        attrs.append(f'data-content-format="{_esc(content_format)}"')
    baseline_sha256 = str(metadata.get("ai_repair_baseline_sha256", "") or "").strip()
    if baseline_sha256:
        attrs.append(f'data-baseline-sha256="{_esc(baseline_sha256)}"')
    export_revision = metadata.get("ai_repair_export_revision")
    if export_revision not in (None, ""):
        try:
            attrs.append(f'data-export-revision="{int(export_revision)}"')
        except (TypeError, ValueError, OverflowError):
            pass
    column_ids = metadata.get("ai_repair_column_ids") or metadata.get("source_column_ids") or []
    if isinstance(column_ids, str):
        column_ids = [column_ids]
    if isinstance(column_ids, (list, tuple, set)):
        value = ",".join(str(item) for item in column_ids if str(item))
        if value:
            attrs.append(f'data-column-ids="{_esc(value)}"')
    bbox = metadata.get("ai_repair_bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            value = ",".join(f"{float(v):.8g}" for v in bbox[:4])
        except (TypeError, ValueError, OverflowError):
            value = ""
        if value:
            attrs.append(f'data-bbox="{_esc(value)}"')
    if bool(metadata.get("ai_repair_delete_intentionally", False)):
        attrs.append('data-delete-intentionally="true"')
    return " " + " ".join(attrs)


def _block_to_xhtml(b: Block, *, allow_ruby: bool = True) -> str:
    """把单个 Block 转成 XHTML；Ruby OFF 时绝不读取 Ruby side-channel。"""
    raw = _sanitize_export_text(b.text or "")
    attrs = _repair_attrs(b)
    ruby_raw = _ruby_export_source(b, allow_ruby=allow_ruby)
    has_ruby = bool(re.search(r"[｜|][^《]+《[^》]+》", ruby_raw))

    def render(value: str) -> str:
        if has_ruby:
            return _repair_ruby_to_xhtml(value) if attrs else _ruby_to_xhtml(value)
        return _repair_plain_to_xhtml(value) if attrs else _esc(value)

    if b.type == BlockType.RUBY:
        content = render(ruby_raw)
        return f'    <p class="normal"{attrs}>{content}</p>\n'

    if b.type == BlockType.CHAPTER:
        # XHTML 已经使用 h1，不再保留 Markdown 标题前缀。
        value = ruby_raw if has_ruby else raw
        value = re.sub(r"^\s*#+\s*", "", value).strip()
        return f"    <h1{attrs}>{render(value)}</h1>\n"

    if b.type == BlockType.SECTION:
        value = ruby_raw if has_ruby else raw
        value = re.sub(r"^\s*#+\s*", "", value).strip()
        return f"    <h2{attrs}>{render(value)}</h2>\n"

    if b.type == BlockType.DIALOGUE:
        value = (ruby_raw if has_ruby else raw).lstrip("　")
        return f'    <p class="dialogue"{attrs}>{render(value)}</p>\n'

    # 普通段落：统计开头的全角空格数量。
    value = ruby_raw if has_ruby else raw
    indent_count = 0
    while value.startswith("　"):
        value = value[1:]
        indent_count += 1
    cls = "normal indent" if indent_count >= 1 else "normal"
    return f'    <p class="{cls}"{attrs}>{render(value)}</p>\n'


# ── 核心构建函数 ──────────────────────────────────────────────────────────────

@_cleanup_workspace
def build_epub(
    doc: UnifiedDocument,
    output_path: str,
    css_template: str = DEFAULT_TEMPLATE,
    vertical: bool = False,
    verbose: bool = True,
    custom_css: str | None = None,
    preserve_image_bytes: bool = False,
    include_nav_in_spine: bool = False,
) -> None:
    """
    从 UnifiedDocument 生成 EPUB3 文件。

    策略：
        - 每个 CHAPTER 块开始一个新的 chapter_XX.xhtml
        - IMAGE_REF 块根据锚点插在对应正文段落后（作为独立 xhtml 页）
        - cover / color_illus 页来自 PageInfo，单独生成 xhtml

    custom_css: 传入时优先于 css_template——用于 Format Profile（从参考 EPUB 学习
    /手写的自定义排版），此时 css_template 只在 custom_css 为空时才生效。
    include_nav_in_spine: 兼容极少数旧阅读器；默认 False，避免目录被当作正文翻到。
    """
    out = Path(output_path)
    tmp = out.parent / f"_epub_tmp_{out.stem}"
    tmp.mkdir(parents=True, exist_ok=True)

    # 目录结构
    oebps = tmp / "EPUB"
    (oebps / "content").mkdir(parents=True)
    (oebps / "styles").mkdir(parents=True)
    (oebps / "images").mkdir(parents=True)
    (tmp / "META-INF").mkdir(parents=True)

    meta = doc.metadata
    ruby_export_enabled = bool(getattr(meta, "ruby_preservation_enabled", False))
    book_id = str(uuid.uuid4())
    title  = meta.title  or "Untitled"
    author = meta.author or "Unknown"
    lang   = meta.language or "ja"

    css_content, css_source = resolve_epub_css(doc, css_template, custom_css)

    # AI 纠错排版版本同时保存一份可编辑的外部 CSS。EPUB 内仍写入标准的
    # EPUB/styles/style.css；OCR、普通替换和“仅 AI 纠错”不会生成该伴随文件。
    companion: Path | None = None
    companion_css: str | None = None
    if css_source == "ai":
        css_name = str(getattr(meta, "ai_epub_css_name", "") or "ai_typeset.css")
        css_suffix = Path(css_name).suffix or ".css"
        companion = out.with_name(f"{out.stem}.ai-typeset{css_suffix}")
        companion_css = css_content

    # 正文统一首行缩进；对白和分节符不缩进。
    # 不再用 p.normal { text-indent: 0 !important; } 覆盖模板默认值。
    css_content += """
p.normal {
    text-indent: 1em !important;
}
p.dialogue,
p.section-break {
    text-indent: 0 !important;
}
/* 竖排标点统一使用正文方向，避免混合句点/省略号产生基线错位。 */
html, body, p {
    text-orientation: mixed;
    font-variant-east-asian: normal;
}
"""

    if not vertical:
        # 移除完整的 writing-mode 声明。旧正则只识别 -epub- 前缀，
        # 在 -webkit-writing-mode 上从中间开始匹配，留下无效的 "-webkit-"。
        css_content = re.sub(
            r'(?im)^\s*(?:-epub-|-webkit-)?writing-mode\s*:\s*[^;}]*(?:;)?\s*$',
            '',
            css_content,
        )

    (oebps / "styles" / "style.css").write_text(css_content, encoding="utf-8")

    # ── 处理封面图片 ─────────────────────────────────────────────────────────
    cover_page = next((p for p in doc.pages if p.page_type == BlockType.COVER), None)
    cover_img_id = None
    cover_img_href = None
    cover_img_mime = None

    if cover_page and cover_page.image_path and Path(cover_page.image_path).exists():
        src = Path(cover_page.image_path)
        filename, cover_img_mime = _copy_epub_image(
            src, oebps / "images", "cover", preserve_bytes=preserve_image_bytes
        )
        cover_img_id   = "cover-img"
        cover_img_href = f"images/{filename}"
        if verbose:
            print(f"  🖼️  封面: {src.name}")

    # ── 处理所有图片 ─────────────────────────────────────────────────────────
    # block_index → (img_id, href_in_epub)
    image_manifest: dict[int, tuple[str, str]] = {}

    illus_idx = 0
    for block_index, b in enumerate(doc.blocks):
        # 强制跳过没有图片路径或不是图片引用的 block。
        if not getattr(b, "image_path", None):
            continue
        if str(b.type) not in ("image_ref", "IMAGE_REF", "BlockType.IMAGE_REF"):
            continue

        src = Path(b.image_path)
        if not src.exists():
            if verbose:
                print(f"  ⚠️ 插图文件不存在，已跳过: {src}")
            continue
        # 只跳过“实际被选作 EPUB 封面”的那一张。页面管理有时会把
        # 卷首彩页、扉页等连续多页都标为 cover；旧逻辑因此把第 2～N 张
        # 也全部丢掉，导致插图数量和位置错乱。其余 IMAGE_REF 必须按
        # UnifiedDocument.blocks 原始顺序进入 spine。
        if cover_page and _same_file(b.image_path, cover_page.image_path):
            continue

        illus_idx += 1
        fname, _mime = _copy_epub_image(
            src, oebps / "images", f"illus_{illus_idx:03d}",
            preserve_bytes=preserve_image_bytes,
        )
        image_manifest[block_index] = (f"img-{illus_idx:03d}", f"images/{fname}")

    # 按对象身份缓存索引，避免章节循环中反复执行 list.index 的 O(n²) 扫描。
    block_index_by_identity = {id(block): index for index, block in enumerate(doc.blocks)}

    # ── 将 blocks 分组为章节 ─────────────────────────────────────────────────
    # chapter 0 = 书前（序言之前的内容）
    chapters: list[tuple[str, list[Block]]] = []
    current_title = FRONT_MATTER_TITLE
    current_blocks: list[Block] = []

    for b in doc.blocks:
        if b.type == BlockType.IMAGE_REF:
            current_blocks.append(b)
            continue
        if b.type == BlockType.CHAPTER:
            if current_blocks:
                chapters.append((current_title, current_blocks))
            current_title  = b.text
            current_blocks = [b]
        else:
            current_blocks.append(b)

    if current_blocks:
        chapters.append((current_title, current_blocks))

    # AI/OCR 偶尔会在文档末尾留下一个只有章节标题、没有正文/插图的伪章节。
    # 这种空壳章节不应进入 nav、manifest 或 spine。只从末尾连续清理，
    # 不影响正文中有意存在的标题页。
    def _has_substantive_chapter_content(blocks: list[Block]) -> bool:
        for block in blocks:
            if (block.metadata or {}).get("consumed"):
                continue
            if bool((block.metadata or {}).get("ai_repair_keep_empty", False)):
                return True
            if block.type == BlockType.IMAGE_REF:
                return True
            if block.type == BlockType.CHAPTER:
                continue
            text = _sanitize_export_text(block.text or "")
            if text and text not in _ORPHAN_CLOSING_QUOTES:
                return True
        return False

    while chapters and chapters[-1][0] != FRONT_MATTER_TITLE and not _has_substantive_chapter_content(chapters[-1][1]):
        removed_title, _ = chapters.pop()
        if verbose:
            print(f"  ⚠️ 跳过末尾空章节: {removed_title}")

    # ── 生成每章 XHTML ───────────────────────────────────────────────────────
    # (id, href, title, in_toc) —— in_toc=False 的条目（插图页、章节续篇）
    # 仍然写入 spine/manifest 保证翻页顺序正确，但不出现在读者可见的目录里，
    # 避免"目次"被大量"挿絵"/"（続き）"条目淹没。
    content_files: list[tuple[str, str, str, bool]] = []

    # 封面页
    if cover_img_href:
        cover_body = f'  <div class="cover-page">\n    <img src="../{cover_img_href}" alt="{_esc(title)}"/>\n  </div>'
        xhtml = _xhtml_wrap(title, cover_body)
        fname = "content/cover.xhtml"
        (oebps / fname).write_text(xhtml, encoding="utf-8")
        content_files.append(("cover-page", fname, "表紙", False))

    # 正文章节
    # 关键：插图必须在 spine 中出现在它前面的正文之后、后面的正文之前，
    # 与 blocks 里的原始顺序完全一致 —— 否则阅读器翻页顺序会乱掉
    # （旧实现是"整章文字合并成一个文件 + 插图页单独提前写入"，
    # 插图在 spine 里永远排在本章文字前面，与实际阅读顺序不符）。
    # 做法：把每一章按插图位置切成若干文字片段，插图出现处就把
    # 之前攒的文字片段先落盘、加入 spine，再落盘插图页，如此交替。
    #
    # nav.xhtml（目录）要插在"前书页"（封面/扉页/目录扫描页/空白页等
    # 正文之前的内容）之后、第一个真正的章节之前——记录下这个插入位置。
    nav_insert_index = len(content_files)

    for ch_idx, (ch_title, blocks) in enumerate(chapters):
        ch_id = f"ch{ch_idx + 1:03d}"
        lines: list[str] = []
        fragment_idx = 0
        chapter_has_content = False

        def _flush_text_fragment(is_first: bool):
            nonlocal lines, fragment_idx
            if not lines:
                return
            fragment_idx += 1
            frag_id = ch_id if fragment_idx == 1 else f"{ch_id}_p{fragment_idx}"
            body = f'  <section epub:type="chapter">\n' + "".join(lines) + "  </section>"
            frag_title = ch_title if is_first else f"{ch_title}（続き）"
            xhtml = _xhtml_wrap(frag_title, body)
            fname = f"content/{frag_id}.xhtml"
            (oebps / fname).write_text(xhtml, encoding="utf-8")
            # 前书桶不是真正的章节标题，哪怕是它的第一个片段也不该进目录——
            # 之前这里直接拿 is_first 当 in_toc，导致"前书页"被当成正常章节列进 nav.xhtml。
            in_toc = is_first and ch_title != FRONT_MATTER_TITLE
            content_files.append((frag_id, fname, frag_title, in_toc))
            lines = []

        for b in blocks:
            if (b.metadata or {}).get("consumed"):
                continue
            if b.type == BlockType.IMAGE_REF:
                # 插图前面攒的文字必须先落盘，保持它在 spine 中位于插图之前
                _flush_text_fragment(is_first=(fragment_idx == 0))

                # 使用 block 身份作为图片索引，避免 title_page/toc_page 等
                # 无正文锚点的 IMAGE_REF 互相覆盖
                block_index = block_index_by_identity.get(id(b), -1)
                img_info = image_manifest.get(block_index)
                if img_info:
                    img_id, img_href = img_info
                    illus_body = (
                        f'  <div class="illus-page">\n'
                        f'    <img src="../{img_href}" alt="挿絵"/>\n'
                        f'  </div>'
                    )
                    illus_xhtml = _xhtml_wrap("挿絵", illus_body)
                    illus_fname = f"content/{ch_id}_illus_{img_id}.xhtml"
                    (oebps / illus_fname).write_text(illus_xhtml, encoding="utf-8")
                    content_files.append((f"{ch_id}-{img_id}", illus_fname, "挿絵", False))
                    chapter_has_content = True
            else:
                clean_text = _sanitize_export_text(b.text or "")
                keep_empty = bool((b.metadata or {}).get("ai_repair_keep_empty", False))
                if (not clean_text and not keep_empty) or clean_text in _ORPHAN_CLOSING_QUOTES:
                    continue
                export_block = copy.copy(b)
                export_block.text = clean_text
                xhtml_piece = _block_to_xhtml(
                    export_block,
                    # Existing/imported structural Ruby is part of the document
                    # and keeps the historical round-trip behavior.  The OCR
                    # Ruby switch gates only findtext side-channel metadata on
                    # ordinary prose blocks.
                    allow_ruby=(ruby_export_enabled or export_block.type == BlockType.RUBY),
                )
                lines.append(xhtml_piece)
                chapter_has_content = True

        _flush_text_fragment(is_first=(fragment_idx == 0))

        if ch_idx == 0 and ch_title == FRONT_MATTER_TITLE:
            nav_insert_index = len(content_files)

        if verbose and chapter_has_content:
            print(f"  📄 章节: {ch_title[:30]}  ({fragment_idx} 个文字片段)")

    # ── nav.xhtml（TOC）────────────────────────────────────────────────────
    # 只列出真正的章节起始页，插图页/续篇片段不占目录条目（但仍在 spine 里，翻页能翻到）
    #
    # content_files 里的 href（如 "content/ch001.xhtml"）是相对 content.opf
    # （在 EPUB/ 根目录）算的，manifest/spine 用这个是对的。但 nav.xhtml 自己
    # 就放在 EPUB/content/ 目录下，链接要相对 nav.xhtml 自身算，所以这里必须
    # 把 "content/" 前缀去掉，否则点目录会跳到不存在的 content/content/xxx.xhtml，
    # 目录形同虚设。
    def _nav_href(href: str) -> str:
        return href[len("content/"):] if href.startswith("content/") else href

    toc_items = "\n".join(
        f'      <li><a href="{_nav_href(href)}">{_esc(ctitle)}</a></li>'
        for fid, href, ctitle, in_toc in content_files if in_toc
    )
    nav_body = dedent(f"""\
      <nav epub:type="toc">
        <h1>目次</h1>
        <ol>
    {toc_items}
        </ol>
      </nav>
    """)
    nav_xhtml = _xhtml_wrap("目次", nav_body)
    (oebps / "content" / "nav.xhtml").write_text(nav_xhtml, encoding="utf-8")

    # ── content.opf ─────────────────────────────────────────────────────────
    # manifest items
    manifest_lines = [
        '    <item id="nav" href="content/nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
        '    <item id="css" href="styles/style.css" media-type="text/css"/>',
    ]
    if cover_img_href:
        manifest_lines.append(
            f'    <item id="{cover_img_id}" href="{cover_img_href}" media-type="{cover_img_mime}" properties="cover-image"/>'
        )

    for img_id, img_href in image_manifest.values():
        ext  = Path(img_href).suffix.lower()
        mime = _EPUB_IMAGE_MIME.get(ext, "image/png")
        manifest_lines.append(f'    <item id="{img_id}" href="{img_href}" media-type="{mime}"/>')

    for fid, href, _, _ in content_files:
        # Raster illustration XHTML is ordinary XHTML; properties="svg" is
        # only legal when the content document actually contains embedded SVG.
        manifest_lines.append(
            f'    <item id="{fid}" href="{href}" media-type="application/xhtml+xml"/>'
        )

    # spine items（页面顺序，包含插图页/续篇片段，保证翻页顺序正确）。
    # EPUB 3 的 nav 文档只需进入 manifest；默认不进入 spine，避免部分阅读器
    # 把“目次”当作正文页面翻到。仅在显式兼容旧阅读器时插入。
    spine_fids = [fid for fid, _, _, _ in content_files]
    if include_nav_in_spine:
        spine_fids.insert(nav_insert_index, "nav")
    spine_lines = [f'    <itemref idref="{fid}"/>' for fid in spine_fids]

    # cover meta
    cover_meta = f'\n    <meta name="cover" content="{cover_img_id}"/>' if cover_img_id else ""

    # 竖排日文书从右往左翻页；横排则是常规从左往右翻页
    page_direction = "rtl" if vertical else "ltr"

    # 和 _xhtml_wrap 同样的坑：manifest_lines/spine_lines/cover_meta 都是多行、
    # 自带缩进的变量内容，不能整体塞进一个大的 dedent(f"""...""") 里——那样
    # 公共前导空白会被这些内部缩进拉偏，导致 <?xml ...?> 声明前残留空格，
    # 变成不合法的 XML。所以把静态骨架拆成几段各自 dedent，变量内容原样拼接。
    opf_head = dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <package xmlns="http://www.idpf.org/2007/opf"
                 version="3.0"
                 xml:lang="{lang}"
                 unique-identifier="uid">

          <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
            <dc:title>{_esc(title)}</dc:title>
            <dc:creator>{_esc(author)}</dc:creator>
            <dc:language>{lang}</dc:language>
            <dc:publisher>{_esc(meta.publisher or '')}</dc:publisher>
            <dc:identifier id="uid">urn:uuid:{book_id}</dc:identifier>""")
    opf_mid = dedent("""

          </metadata>

          <manifest>
        """)
    opf_mid2 = dedent(f"""

          </manifest>

          <spine page-progression-direction="{page_direction}">
        """)
    opf_tail = dedent("""

          </spine>

        </package>
    """)

    opf = (
        opf_head + cover_meta + opf_mid
        + chr(10).join(manifest_lines) + opf_mid2
        + chr(10).join(spine_lines) + opf_tail
    )
    (oebps / "content.opf").write_text(opf, encoding="utf-8")

    # ── META-INF/container.xml ───────────────────────────────────────────────
    (tmp / "META-INF" / "container.xml").write_text(dedent("""\
        <?xml version="1.0" encoding="UTF-8"?>
        <container version="1.0"
                   xmlns="urn:oasis:schemas:container">
          <rootfiles>
            <rootfile full-path="EPUB/content.opf"
                      media-type="application/oebps-package+xml"/>
          </rootfiles>
        </container>
    """), encoding="utf-8")

    # ── 打包 EPUB（zip）─────────────────────────────────────────────────────
    # Never delete/overwrite the user's previous book until a complete new ZIP
    # has been written and reopened successfully.  A failed build therefore
    # leaves the last known-good EPUB untouched.
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, staged_name = tempfile.mkstemp(
        prefix=f".{out.stem}.", suffix=".epub.tmp", dir=out.parent
    )
    os.close(fd)
    staged_out = Path(staged_name)
    try:
        with zipfile.ZipFile(staged_out, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            # mimetype 必须是第一个文件，且不压缩
            zf.writestr(
                zipfile.ZipInfo("mimetype"),
                "application/epub+zip",
                compress_type=zipfile.ZIP_STORED,
            )
            for fpath in tmp.rglob("*"):
                if fpath.is_file():
                    arcname = str(fpath.relative_to(tmp))
                    zf.write(fpath, arcname)

        # Reopen before commit to catch central-directory/truncation failures.
        with zipfile.ZipFile(staged_out, "r") as check_zip:
            names = check_zip.namelist()
            if not names or names[0] != "mimetype":
                raise RuntimeError("EPUB 打包失败：mimetype 不是 ZIP 第一项")
            if check_zip.read("mimetype") != b"application/epub+zip":
                raise RuntimeError("EPUB 打包失败：mimetype 内容无效")
            bad_member = check_zip.testzip()
            if bad_member:
                raise RuntimeError(f"EPUB 打包失败：ZIP 成员损坏 {bad_member}")
        os.replace(staged_out, out)
        # The optional companion CSS belongs to the committed EPUB version.
        if companion is not None and companion_css is not None:
            atomic_write_text(companion, companion_css)
            if verbose:
                print(f"  🎨 AI 排版 CSS: {companion.name}")
    finally:
        staged_out.unlink(missing_ok=True)

    # ── 清理临时目录 ─────────────────────────────────────────────────────────
    shutil.rmtree(tmp)

    if verbose:
        size_kb = out.stat().st_size // 1024
        print(f"\n✅  EPUB 已生成: {out}  ({size_kb} KB)")
        print(f"    章节: {len(chapters)}，图片: {illus_idx}，内容文件: {len(content_files)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UnifiedDocument → EPUB3")
    parser.add_argument("input_json",   help="Formatter 输出的 JSON")
    parser.add_argument("output_epub",  help="输出 EPUB 路径")
    parser.add_argument(
        "--template", "-t",
        choices=list(CSS_TEMPLATES.keys()),
        default=DEFAULT_TEMPLATE,
        help=f"CSS 模板（默认: {DEFAULT_TEMPLATE}）",
    )
    parser.add_argument("--vertical", action="store_true", help="竖排模式（默认横排）")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    with open(args.input_json, encoding="utf-8") as f:
        doc = UnifiedDocument.from_json(f.read())

    build_epub(
        doc,
        output_path=args.output_epub,
        css_template=args.template,
        vertical=args.vertical,
        verbose=not args.quiet,
    )
