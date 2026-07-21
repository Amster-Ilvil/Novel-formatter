# -*- coding: utf-8 -*-
"""PaddleOCR-VL importer with JSON/Markdown compatibility."""

from pathlib import Path
import json
import re
from html.parser import HTMLParser

from models.document import UnifiedDocument, Block, BlockType, PageInfo


def _safe_type(value):
    try:
        return BlockType(value)
    except Exception:
        return BlockType.PARAGRAPH


def _get_override(page, page_overrides):
    if not page_overrides:
        return BlockType.PARAGRAPH
    value = page_overrides.get(page, page_overrides.get(str(page)))
    return _safe_type(value) if value else BlockType.PARAGRAPH


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _walk_nodes(value):
    if isinstance(value, list):
        for item in value:
            yield from _walk_nodes(item)
    elif isinstance(value, dict):
        for key in ("blocks", "layout", "parsing_res"):
            if key in value:
                yield from _walk_nodes(value[key])
        if "text" in value or "content" in value:
            yield value


class _HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []

    def handle_data(self, data):
        self.text.append(data)

    def get_text(self):
        return "".join(self.text).strip()


def _strip_html(html):
    if not html:
        return ""

    # PaddleOCR-VL may return text/markdown/content as dict objects.
    if isinstance(html, dict):
        html = html.get("text") or html.get("content") or html.get("markdown") or str(html)
    elif not isinstance(html, str):
        html = str(html)

    html = html.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    parser = _HTMLTextExtractor()
    parser.feed(html)
    return parser.get_text()


def import_paddle_json(
    json_path,
    image_folder,
    page_overrides=None,
    offset=0,
    page_images=None,
    strip_special_text=True,
    add_images_for_paragraph=False
):
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))

    if isinstance(data, dict):
        pages = data.get("pages")
        if pages is None:
            pages = [data]
    elif isinstance(data, list):
        pages = data
    else:
        pages = []

    if not pages:
        pages = [data]

    doc = UnifiedDocument()

    # ---- 归一化页序号 ----
    # PaddleOCR-VL 不同导出版本的 page_index 有的从 0 开始、有的从 1 开始；
    # 之前直接拿 "page_index + offset" 当 1-based 页码、再用 "page - 1"
    # 反推 page_images 列表下标，两步各自的起点假设互相矛盾——当本地
    # 扫描图按最常见的 0 起始命名（000000.jpg 对应物理第 1 页，见项目
    # 约定）时，offset 常常是 0，"page - 1" 就会算出 -1，导致第一页
    # （往往正是封面）的本地图片路径直接丢失、后续每一页也都错位一张。
    #
    # 这里先不管 JSON 里 page_index 本身从几开始，统一减去这批数据里的
    # 最小值，得到一个必然从 0 开始的 norm_idx；offset 则只表示"这批
    # JSON 对应 page_images 列表里的第几个位置开始"（0 = 从头开始，即
    # 完整导入整本书；只导入书的一部分、要跟已加载的完整图片列表对齐
    # 时才需要传非 0 值）。1-based 的 page（用于匹配 page_overrides 的
    # 编号习惯）和 0-based 的 img_idx（用于索引 page_images 列表）各自
    # 独立算出，不再互相借用。
    raw_indices = [int(_as_dict(rp).get("page_index", i)) for i, rp in enumerate(pages)]
    min_raw_index = min(raw_indices) if raw_indices else 0

    for i, raw_page in enumerate(pages):
        p = _as_dict(raw_page)
        norm_idx = raw_indices[i] - min_raw_index   # 0-based，本 JSON 内部序号
        page = norm_idx + offset + 1                # 1-based，对齐 page_overrides
        img_idx = norm_idx + offset                 # 0-based，对齐 page_images 列表
        page_type = _get_override(page, page_overrides)
        is_explicit_override = bool(
            page_overrides and (page_overrides.get(page) or page_overrides.get(str(page)))
        )

        # ---- 先算出本页对应的本地图片路径：既要用来生成 IMAGE_REF block，
        #      也要记进 doc.pages，供 epub_builder.py 识别封面/扉页/目录页等
        #      特殊页面（它按 doc.pages 里 page_type==COVER 找封面图，而不是
        #      按 block 类型——block 的类型统一是 IMAGE_REF，本身不携带
        #      "这是封面还是普通插图"这层语义）。----
        local_image = None
        if page_images:
            if 0 <= img_idx < len(page_images):
                candidate = page_images[img_idx]
                if candidate and Path(candidate).exists():
                    local_image = str(Path(candidate).resolve())

        raw_text = p.get("text") or p.get("markdown") or p.get("content") or ""

        if isinstance(raw_text, dict):
            raw_text = (
                raw_text.get("text")
                or raw_text.get("content")
                or raw_text.get("markdown")
                or str(raw_text)
            )
        elif not isinstance(raw_text, str):
            raw_text = str(raw_text)

        plain_text = _strip_html(raw_text)

        special_types = {
            getattr(BlockType, "COVER", None),
            getattr(BlockType, "TITLE_PAGE", None),
            getattr(BlockType, "TOC_PAGE", None),
            getattr(BlockType, "COLOPHON", None),
            getattr(BlockType, "BLANK", None),
        }
        is_special = page_type in special_types and strip_special_text
        is_blank = page_type == BlockType.BLANK

        # ---- 记录页面元信息（doc.pages），与 apple_vision_adapter.py /
        #      paddle_ocr_adapter.py 等其它输入路径保持一致的结构，下游
        #      （EPUB Builder 的封面检测、页面管理回看）才能正常工作。----
        doc.pages.append(PageInfo(
            page_no=page,
            page_type=page_type,
            image_path=local_image or "",
            confidence=1.0 if is_explicit_override else 0.9,
        ))

        # ---- 空白页：不产生任何 block（文字或图片），与本项目其它 OCR
        #      输入路径的约定一致，直接跳到下一页。----
        if is_blank:
            continue

        parsing_blocks = _as_dict(p.get("prunedResult")).get("parsing_res_list") or p.get("parsing_res_list")
        if isinstance(parsing_blocks, list) and not is_special:
            for block_index, raw_block in enumerate(parsing_blocks):
                if not isinstance(raw_block, dict):
                    continue
                raw_text = raw_block.get("block_content") or raw_block.get("text") or raw_block.get("content") or ""
                for line in _strip_html(raw_text).splitlines():
                    line = line.strip()
                    if line:
                        doc.blocks.append(Block(
                            type=page_type,
                            text=line,
                            page=page,
                            page_index=norm_idx,
                            page_number=page,
                            order_in_page=raw_block.get("block_order") or block_index,
                            source_format="json",
                            metadata={"bbox": raw_block.get("block_bbox")},
                        ))
        # ---- 按行拆分，每行一个 Block，并保留页码元数据 ----
        elif plain_text and not is_special:
            lines = plain_text.splitlines()
            for block_index, line in enumerate(lines):
                line = line.strip()
                if line:
                    doc.blocks.append(Block(type=page_type, text=line, page=page, page_index=norm_idx, page_number=page, order_in_page=block_index, source_format="json"))

        # ---- 处理图片 ----
        if local_image:
            if is_special or (add_images_for_paragraph and page_type == BlockType.PARAGRAPH):
                doc.blocks.append(Block(type=BlockType.IMAGE_REF, page=page, image_path=local_image))
        else:
            images = p.get("images") or {}
            if isinstance(images, list):
                images = {x: x for x in images if isinstance(x, str)}
            elif not isinstance(images, dict):
                images = {}
            if is_special or (add_images_for_paragraph and page_type == BlockType.PARAGRAPH):
                for rel_path in images.keys():
                    candidate = Path(image_folder) / rel_path
                    if candidate.exists():
                        doc.blocks.append(Block(type=BlockType.IMAGE_REF, page=page, image_path=str(candidate.resolve())))

        # ---- 兼容旧格式：嵌套块中的文本也按行拆分 ----
        if not plain_text and not is_special:
            for node in _walk_nodes(p):
                text = node.get("text") or node.get("content")
                if text:
                    # 同样按行拆分，但嵌套块通常不会有多行，保留原逻辑以兼容
                    for line in str(text).splitlines():
                        line = line.strip()
                        if line:
                            doc.blocks.append(Block(type=page_type, text=line, page=page, page_index=norm_idx, page_number=page, source_format="json"))

    return doc


def import_paddle_md(md_path, image_folder, page_overrides=None, offset=0):
    md_path = Path(md_path)
    sibling_json = md_path.with_suffix(".json")
    if sibling_json.exists():
        return import_paddle_json(sibling_json, image_folder, page_overrides=page_overrides, offset=offset)

    doc = UnifiedDocument()
    page = offset

    for line in md_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue

        if "<img" in s or re.match(r"^!\[.*\]\(.*\)$", s):
            matches = re.findall(r'(?:\(|src=["\'])([^"\')>]+)', s)
            if matches:
                doc.blocks.append(
                    Block(
                        type=BlockType.IMAGE_REF,
                        page=page,
                        image_path=str(Path(image_folder) / matches[0]),
                    )
                )
            continue

        if s.startswith("#"):
            doc.blocks.append(
                Block(
                    type=BlockType.CHAPTER,
                    text=s.lstrip("#").strip(),
                    page=page,
                )
            )
        else:
            doc.blocks.append(
                Block(
                    type=_get_override(page, page_overrides),
                    text=s,
                    page=page,
                )
            )

    return doc