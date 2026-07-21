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
    - TOC 自动生成（nav.xhtml + content.opf spine）
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
import uuid
import zipfile
from pathlib import Path
from textwrap import dedent

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.document import UnifiedDocument, Block, BlockType

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
            font-size: 100%;
            line-height: 1.8;
            color: #1a1a1a;
            background-color: #fafaf8;
        }
        h1 {
            font-size: 1.3em;
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
            font-size: 0.5em;
            -epub-ruby-position: right;
            ruby-position: under;
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
            font-size: 0.95em;
            line-height: 1.9;
            color: #222;
        }
        h1 { font-size: 1.2em; font-weight: bold; letter-spacing: 0.1em; }
        p   { text-indent: 1em; margin: 0; }
        p.dialogue { text-indent: 0; }
        .tcy { text-combine-upright: all; }
        ruby rt { font-size: 0.5em; }
    """),

    "web": dedent("""\
        /* Novel Formatter Studio — 横排 Web 小说样式 */
        @charset "UTF-8";

        body {
            margin: 2em auto;
            max-width: 680px;
            font-family: "ヒラギノ角ゴ Pro", "Hiragino Kaku Gothic Pro",
                         "Noto Sans CJK JP", sans-serif;
            font-size: 1em;
            line-height: 1.9;
            color: #222;
        }
        h1 { font-size: 1.4em; margin: 1.5em 0 1em; }
        p   { text-indent: 1em; margin: 0 0 0.5em; }
        p.dialogue { text-indent: 0; }
    """),
}

DEFAULT_TEMPLATE = "denki"

# 第一个真正章节之前的内容（封面/扉页/目录扫描页/网站样板文字等）统一装进这个
# 标题固定的桶里。它不是一个真正的章节标题，不应该出现在读者可见的目录里。
FRONT_MATTER_TITLE = "前书页"


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
    """将 漢字|よみ 格式转换为 <ruby>漢字<rt>よみ</rt></ruby>"""
    import re
    parts = []
    last = 0
    for m in re.finditer(r'([^\s|]+)\|([^\s|]+)', text):
        if m.start() > last:
            parts.append(_esc(text[last:m.start()]))
        base = _esc(m.group(1))
        reading = _esc(m.group(2))
        parts.append(f'<ruby>{base}<rt>{reading}</rt></ruby>')
        last = m.end()
    if last < len(text):
        parts.append(_esc(text[last:]))
    return ''.join(parts)


def _block_to_xhtml(b: Block) -> str:
    """
    把单个 Block 转成 XHTML 片段。
    
    检测段落开头是否有全角空格（\u3000），并将其转换为 CSS 缩进类。
    """
    raw = b.text or ""

    if b.type == BlockType.RUBY:
        content = _ruby_to_xhtml(raw)
        return f'    <p class="normal">{content}</p>\n'

    if b.type == BlockType.CHAPTER:
        return f"    <h1>{_esc(raw)}</h1>\n"

    if b.type == BlockType.SECTION:
        return f"    <h2>{_esc(raw)}</h2>\n"

    if b.type == BlockType.DIALOGUE:
        # 对白去掉前导全角空格，但保留其他内容
        raw = raw.lstrip("　")
        return f'    <p class="dialogue">{_esc(raw)}</p>\n'

    # 普通段落：统计开头的全角空格数量
    indent_count = 0
    while raw.startswith("　"):
        raw = raw[1:]
        indent_count += 1

    # 根据缩进数量决定 CSS 类
    if indent_count >= 1:
        cls = "normal indent"
    else:
        cls = "normal"

    return f'    <p class="{cls}">{_esc(raw)}</p>\n'


# ── 核心构建函数 ──────────────────────────────────────────────────────────────

def build_epub(
    doc: UnifiedDocument,
    output_path: str,
    css_template: str = DEFAULT_TEMPLATE,
    vertical: bool = False,
    verbose: bool = True,
    custom_css: str | None = None,
) -> None:
    """
    从 UnifiedDocument 生成 EPUB3 文件。

    策略：
        - 每个 CHAPTER 块开始一个新的 chapter_XX.xhtml
        - IMAGE_REF 块根据锚点插在对应正文段落后（作为独立 xhtml 页）
        - cover / color_illus 页来自 PageInfo，单独生成 xhtml

    custom_css: 传入时优先于 css_template——用于 Format Profile（从参考 EPUB 学习
    /手写的自定义排版），此时 css_template 只在 custom_css 为空时才生效。
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
    book_id = str(uuid.uuid4())
    title  = meta.title  or "Untitled"
    author = meta.author or "Unknown"
    lang   = meta.language or "ja"

    css_content = custom_css if custom_css else CSS_TEMPLATES.get(css_template, CSS_TEMPLATES[DEFAULT_TEMPLATE])

    # 追加缩进样式（用于处理 Formatter 添加的全角空格）
    # 使用 !important 确保覆盖模板中的 p 样式
    css_content += """
p.normal {
    text-indent: 0 !important;
}
p.normal.indent {
    text-indent: 1em !important;
}
p.dialogue {
    text-indent: 0 !important;
}
"""

    if not vertical:
        # 移除 writing-mode 相关声明
        css_content = re.sub(r'[^{]*writing-mode[^;]*;', '', css_content)

    (oebps / "styles" / "style.css").write_text(css_content, encoding="utf-8")

    # ── 处理封面图片 ─────────────────────────────────────────────────────────
    cover_page = next((p for p in doc.pages if p.page_type == BlockType.COVER), None)
    cover_img_id = None
    cover_img_href = None

    if cover_page and cover_page.image_path and Path(cover_page.image_path).exists():
        src = Path(cover_page.image_path)
        ext = src.suffix.lower()
        dest = oebps / "images" / f"cover{ext}"
        shutil.copy2(src, dest)
        cover_img_id   = "cover-img"
        cover_img_href = f"images/cover{ext}"
        if verbose:
            print(f"  🖼️  封面: {src.name}")

    # ── 处理所有图片 ─────────────────────────────────────────────────────────
    # block_index → (img_id, href_in_epub)
    image_manifest: dict[int, tuple[str, str]] = {}

    illus_idx = 0
    for block_index, b in enumerate(doc.blocks):
        # ----- EPUB IMAGE DEBUG / COMPATIBLE IMAGE DETECTION -----
        if verbose:
            print(f"[EPUB DEBUG] block #{block_index}")
            print(f"  type = {repr(getattr(b, 'type', None))}")
            print(f"  image_path = {repr(getattr(b, 'image_path', None))}")

        # 强制跳过没有图片路径的 block
        if getattr(b, "image_path", None) is None:
            if verbose:
                print("  skip: no image_path")
            continue

        # 支持字符串类型和枚举类型
        if str(b.type) not in ("image_ref", "IMAGE_REF", "BlockType.IMAGE_REF"):
            if verbose:
                print(f"  skip: unsupported type {str(b.type)}")
            continue

        src = Path(b.image_path)
        if verbose:
            print(f"[EPUB IMAGE] {src} exists={src.exists()}")
        if not src.exists():
            if verbose:
                print(f"[EPUB IMAGE MISS] {src}")
            continue
        # 跳过封面（已单独处理）
        page_info = next((p for p in doc.pages if p.image_path == b.image_path), None)
        if page_info and page_info.page_type == BlockType.COVER:
            continue

        illus_idx += 1
        ext  = src.suffix.lower()
        fname = f"illus_{illus_idx:03d}{ext}"
        shutil.copy2(src, oebps / "images" / fname)
        image_manifest[block_index] = (f"img-{illus_idx:03d}", f"images/{fname}")

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
        content_files.append(("cover-page", fname, "表紙", True))

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
            if b.type == BlockType.IMAGE_REF:
                # 插图前面攒的文字必须先落盘，保持它在 spine 中位于插图之前
                _flush_text_fragment(is_first=(fragment_idx == 0))

                # 使用 block 身份作为图片索引，避免 title_page/toc_page 等
                # 无正文锚点的 IMAGE_REF 互相覆盖
                try:
                    block_index = doc.blocks.index(b)
                except ValueError:
                    block_index = -1
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
                xhtml_piece = _block_to_xhtml(b)
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
        ext = Path(cover_img_href).suffix.lower()
        mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
        manifest_lines.append(
            f'    <item id="{cover_img_id}" href="{cover_img_href}" media-type="{mime}" properties="cover-image"/>'
        )

    for img_id, img_href in image_manifest.values():
        ext  = Path(img_href).suffix.lower()
        mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
        manifest_lines.append(f'    <item id="{img_id}" href="{img_href}" media-type="{mime}"/>')

    for fid, href, _, _ in content_files:
        props = ' properties="svg"' if "illus" in fid else ""
        manifest_lines.append(
            f'    <item id="{fid}" href="{href}" media-type="application/xhtml+xml"{props}/>'
        )

    # spine items（页面顺序，包含插图页/续篇片段，保证翻页顺序正确）
    # nav 插在"前书页"（封面/扉页/目录扫描页等）之后、第一章正文之前，
    # 而不是无条件排在最前面（那样会跑到封面前面去）。
    spine_fids = [fid for fid, _, _, _ in content_files]
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
    if out.exists():
        out.unlink()

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
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