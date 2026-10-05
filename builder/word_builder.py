#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a vertical DOCX with column provenance verification."""
from __future__ import annotations

import argparse
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.document import UnifiedDocument, Block, BlockType


def _set_vertical_layout(doc):
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    sect_pr = doc.sections[0]._sectPr
    text_dir = OxmlElement('w:textDirection')
    text_dir.set(qn('w:val'), 'tbRl')
    sect_pr.append(text_dir)


def _add_page_break(doc):
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    para = doc.add_paragraph()
    run = para.add_run()
    br = OxmlElement('w:br')
    br.set(qn('w:type'), 'page')
    run._r.append(br)


def _ruby_source_matches_current_text(marked: str, current_text: str) -> bool:
    if not re.search(r"[｜|][^《]+《[^》]+》", marked or ""):
        return False
    plain = re.sub(r"[｜|]([^《\n]+)《([^》\n]+)》", r"\1", str(marked or ""))
    return plain == str(current_text or "")


def _block_text_for_word(b: Block, *, allow_ruby: bool = True) -> str:
    metadata = b.metadata if isinstance(getattr(b, "metadata", None), dict) else {}
    current = str(b.text or "")
    if not allow_ruby:
        return current
    source = str(metadata.get("ruby_aozora") or "")
    if _ruby_source_matches_current_text(source, current):
        return re.sub(r"[｜|]([^《]+)《([^》]+)》", r"\1（\2）", source)
    if b.type != BlockType.RUBY:
        return current

    source = str(b.ocr_raw or "")
    if _ruby_source_matches_current_text(source, current):
        return re.sub(r"[｜|]([^《]+)《([^》]+)》", r"\1（\2）", source)

    # Legacy ``base|reading`` JSON has no explicit end delimiter.  Restrict the
    # reading to kana and, when it has greedily captured a following particle
    # before a kanji/punctuation boundary, put that particle back into prose.
    value = str(b.text or "")
    marker = re.compile(r"([^\s|]{1,24})\|([ぁ-ゖァ-ヺー]{1,32})")
    particles = set("をがにへとはもので")
    output: list[str] = []
    last = 0
    for match in marker.finditer(value):
        reading = match.group(2)
        effective_end = match.end()
        following = value[effective_end:effective_end + 1]
        if (len(reading) >= 3 and reading[-1] in particles and following
                and re.match(r"[一-龯々〆ヵヶァ-ヶーA-Za-z0-9０-９、。！？!?]", following)):
            reading = reading[:-1]
            effective_end -= 1
        output.append(value[last:match.start()])
        output.append(f"{match.group(1)}（{reading}）")
        last = effective_end
    output.append(value[last:])
    return "".join(output)




def _ruby_source_for_word(block: Block, *, allow_ruby: bool) -> str:
    """Return current Aozora-style Ruby source, or an empty string if unavailable.

    The source is accepted only when stripping Ruby produces exactly the current
    authoritative ``Block.text``.  This keeps stale side-channel metadata from
    resurrecting text that the user edited after OCR.
    """
    if not allow_ruby:
        return ""
    metadata = block.metadata if isinstance(getattr(block, "metadata", None), dict) else {}
    current = str(block.text or "")
    source = str(metadata.get("ruby_aozora") or "")
    if _ruby_source_matches_current_text(source, current):
        return source
    if block.type == BlockType.RUBY:
        source = str(block.ocr_raw or "")
        if _ruby_source_matches_current_text(source, current):
            return source
    return ""


def _iter_word_ruby_segments(block: Block, *, allow_ruby: bool):
    """Yield ``(plain, reading_or_none)`` segments for native Word Ruby output."""
    source = _ruby_source_for_word(block, allow_ruby=allow_ruby)
    if source:
        marker = re.compile(r"[｜|]([^《]+)《([^》]+)》")
        last = 0
        for match in marker.finditer(source):
            if match.start() > last:
                yield source[last:match.start()], None
            yield match.group(1), match.group(2)
            last = match.end()
        if last < len(source):
            yield source[last:], None
        return

    # Historical BlockType.RUBY JSON may contain only ``base|reading``.  Keep
    # the legacy parser's conservative particle protection, but emit real
    # w:ruby instead of flattening the reading into parentheses.
    value = str(block.text or "")
    if allow_ruby and block.type == BlockType.RUBY:
        marker = re.compile(r"([^\s|]{1,24})\|([ぁ-ゖァ-ヺー]{1,32})")
        particles = set("をがにへとはもので")
        last = 0
        matched = False
        for match in marker.finditer(value):
            reading = match.group(2)
            effective_end = match.end()
            following = value[effective_end:effective_end + 1]
            if (len(reading) >= 3 and reading[-1] in particles and following
                    and re.match(r"[一-龯々〆ヵヶァ-ヶーA-Za-z0-9０-９、。！？!?]", following)):
                reading = reading[:-1]
                effective_end -= 1
            if not reading:
                continue
            matched = True
            if match.start() > last:
                yield value[last:match.start()], None
            yield match.group(1), reading
            last = effective_end
        if matched:
            if last < len(value):
                yield value[last:], None
            return

    yield value, None


def _append_plain_run(paragraph, text: str, *, bold: bool = False, font_size=None) -> None:
    if not text:
        return
    text = unicodedata.normalize("NFC", str(text or ""))
    run = paragraph.add_run(text)
    if bold:
        run.bold = True
    if font_size is not None:
        run.font.size = font_size


def _append_native_ruby(paragraph, base: str, reading: str, *, bold: bool = False, font_size=None) -> None:
    """Append standards-compliant WordprocessingML Ruby to ``paragraph``.

    Building the XML tree through ``OxmlElement`` means XML entities and
    punctuation are escaped by lxml instead of being interpolated into raw XML.
    """
    base = unicodedata.normalize("NFC", str(base or ""))
    reading = unicodedata.normalize("NFC", str(reading or ""))
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    ruby = OxmlElement("w:ruby")
    ruby_pr = OxmlElement("w:rubyPr")

    align = OxmlElement("w:rubyAlign")
    align.set(qn("w:val"), "center")
    ruby_pr.append(align)

    # Half-size reading is a broadly compatible Word default.  hps values are
    # in half-points; base size is only emitted when a concrete font size exists.
    if font_size is not None:
        base_half_points = max(2, int(round(float(font_size.pt) * 2)))
        rt_half_points = max(2, int(round(base_half_points * 0.5)))
        hps = OxmlElement("w:hps")
        hps.set(qn("w:val"), str(rt_half_points))
        ruby_pr.append(hps)
        hps_base = OxmlElement("w:hpsBaseText")
        hps_base.set(qn("w:val"), str(base_half_points))
        ruby_pr.append(hps_base)

    lid = OxmlElement("w:lid")
    lid.set(qn("w:val"), "ja-JP")
    ruby_pr.append(lid)
    ruby.append(ruby_pr)

    def _make_r(text: str, *, is_base: bool):
        run = OxmlElement("w:r")
        if bold or font_size is not None:
            rpr = OxmlElement("w:rPr")
            if bold:
                b = OxmlElement("w:b")
                rpr.append(b)
            if font_size is not None:
                sz = OxmlElement("w:sz")
                value = max(2, int(round(float(font_size.pt) * 2)))
                if not is_base:
                    value = max(2, int(round(value * 0.5)))
                sz.set(qn("w:val"), str(value))
                rpr.append(sz)
            run.append(rpr)
        node = OxmlElement("w:t")
        if text[:1].isspace() or text[-1:].isspace():
            node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        node.text = text
        run.append(node)
        return run

    rt = OxmlElement("w:rt")
    rt.append(_make_r(reading, is_base=False))
    ruby.append(rt)

    ruby_base = OxmlElement("w:rubyBase")
    ruby_base.append(_make_r(base, is_base=True))
    ruby.append(ruby_base)
    paragraph._p.append(ruby)

    # Compatibility fallback for consumers such as python-docx that ignore
    # ``w:ruby`` entirely. Word hides this run, while simple text extractors see
    # the historical ``base（reading）`` form instead of silently losing the base.
    fallback_run = paragraph.add_run(f"{base}（{reading}）")
    fallback_rpr = fallback_run._r.get_or_add_rPr()
    vanish = OxmlElement("w:vanish")
    fallback_rpr.append(vanish)


def _write_block_to_paragraph(paragraph, block: Block, *, allow_ruby: bool, bold: bool = False, font_size=None) -> str:
    """Write one block and return the plain authoritative text for verification."""
    pieces = list(_iter_word_ruby_segments(block, allow_ruby=allow_ruby))
    for text, reading in pieces:
        if reading is None:
            _append_plain_run(paragraph, text, bold=bold, font_size=font_size)
        else:
            _append_native_ruby(
                paragraph, text, reading, bold=bold, font_size=font_size
            )
    # Ruby readings are annotations; verification compares the visible base text.
    return "".join(text for text, _reading in pieces)


def _paragraph_plain_text(paragraph) -> str:
    """Extract DOCX visible base text while excluding w:rt annotation text."""
    from docx.oxml.ns import qn

    result: list[str] = []
    for node in paragraph._p.iter(qn("w:t")):
        parent = node.getparent()
        in_reading = False
        hidden = False
        while parent is not None and parent is not paragraph._p:
            if parent.tag == qn("w:rt"):
                in_reading = True
                break
            if parent.tag == qn("w:r"):
                rpr = parent.find(qn("w:rPr"))
                if rpr is not None and rpr.find(qn("w:vanish")) is not None:
                    hidden = True
                    break
            parent = parent.getparent()
        if not in_reading and not hidden:
            result.append(node.text or "")
    return "".join(result)


def _source_column_ids(block: Block) -> list[str]:
    metadata = block.metadata or {}
    values = metadata.get("source_column_ids") or []
    if isinstance(values, str):
        values = [values]
    result = [str(value) for value in values if str(value)]
    column_id = str(metadata.get("column_id", ""))
    if column_id and column_id not in result:
        result.append(column_id)
    return result


def _expected_column_manifest(doc: UnifiedDocument) -> tuple[list[str], dict[str, int]]:
    audit = getattr(doc.metadata, "column_ocr_audit", {}) or {}
    pages = audit.get("pages", {}) if isinstance(audit, dict) else {}
    expected: list[str] = []
    page_by_id: dict[str, int] = {}
    for page_key, report in sorted(pages.items(), key=lambda item: int(item[0])):
        page_no = int(page_key)
        for column_id in report.get("column_ids", []) or []:
            column_id = str(column_id)
            if not column_id:
                continue
            expected.append(column_id)
            page_by_id[column_id] = page_no
    return expected, page_by_id


def _verify_docx_text(temp_path: Path, expected_texts: list[str]) -> None:
    from docx import Document
    reopened = Document(str(temp_path))
    actual = [
        text for paragraph in reopened.paragraphs
        if (text := _paragraph_plain_text(paragraph))
    ]
    if actual != expected_texts:
        first = next(
            (index for index, (left, right) in enumerate(zip(expected_texts, actual), start=1) if left != right),
            min(len(expected_texts), len(actual)) + 1,
        )
        raise RuntimeError(
            "DOCX 保存后复读校验失败：写入前后的段落序列不一致，"
            f"首个差异位于第 {first} 段（预计 {len(expected_texts)}，复读 {len(actual)}）。"
        )


def build_word(
    doc: UnifiedDocument,
    output_path: str,
    vertical: bool = True,
    page_breaks: bool = True,
    verbose: bool = True,
) -> dict:
    """Generate DOCX atomically and verify every fixed-region source column."""
    try:
        from docx import Document
        from docx.shared import Pt
    except ImportError:
        raise ImportError("请安装 python-docx: pip install python-docx")

    word = Document()
    if vertical:
        _set_vertical_layout(word)
    for paragraph in list(word.paragraphs):
        paragraph._element.getparent().remove(paragraph._element)

    text_types = {
        BlockType.PARAGRAPH, BlockType.DIALOGUE, BlockType.CHAPTER,
        BlockType.SECTION, BlockType.RUBY,
    }
    ruby_export_enabled = bool(
        getattr(getattr(doc, "metadata", None), "ruby_preservation_enabled", False)
    )
    expected_ids, page_by_id = _expected_column_manifest(doc)
    represented_ids: list[str] = []
    written_texts: list[str] = []
    prev_page: int | None = None

    for block in doc.blocks:
        if (block.metadata or {}).get("consumed"):
            continue
        if block.type not in text_types:
            continue
        allow_ruby = (ruby_export_enabled or block.type == BlockType.RUBY)
        # Decide emptiness from authoritative prose, not the annotation channel.
        current_text = str(block.text or "")
        if not current_text.strip():
            continue
        if page_breaks and prev_page is not None and block.page != prev_page and written_texts:
            _add_page_break(word)
        prev_page = block.page
        paragraph = word.add_paragraph()
        is_chapter = block.type == BlockType.CHAPTER
        written = _write_block_to_paragraph(
            paragraph,
            block,
            allow_ruby=allow_ruby,
            bold=is_chapter,
            font_size=(Pt(16) if is_chapter else None),
        )
        if not written.strip():
            # Defensive cleanup if malformed legacy Ruby yielded no base text.
            paragraph._element.getparent().remove(paragraph._element)
            continue
        written_texts.append(unicodedata.normalize("NFC", written))
        represented_ids.extend(_source_column_ids(block))

    if expected_ids:
        expected_set = set(expected_ids)
        represented_set = set(represented_ids)
        missing = [column_id for column_id in expected_ids if column_id not in represented_set]
        unexpected = [column_id for column_id in represented_set if column_id not in expected_set]
        audit = getattr(doc.metadata, "column_ocr_audit", {}) or {}
        model_ok = bool(audit.get("model_integrity_passed"))
        if not model_ok or missing or unexpected:
            details = []
            if not model_ok:
                details.append("OCR 文档模型阶段的列对账尚未通过")
            if missing:
                details.append("缺少：" + ", ".join(missing[:12]))
            if unexpected:
                details.append("额外：" + ", ".join(unexpected[:12]))
            raise RuntimeError(
                "DOCX 导出前列 ID 对账失败，已停止保存以避免末列丢失。\n"
                + "\n".join(f"• {item}" for item in details)
            )

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_name(f".{target.stem}.writing-{os.getpid()}.docx")
    try:
        word.save(str(temp_path))
        _verify_docx_text(temp_path, written_texts)
        temp_path.replace(target)
    finally:
        temp_path.unlink(missing_ok=True)

    # A Formatter may split one physical source column into multiple output
    # paragraphs.  DOCX coverage therefore counts unique source-column IDs,
    # while exact missing/extra ID checks still prevent a duplicated middle
    # column from concealing an omitted final column.
    represented_unique_ids = list(dict.fromkeys(represented_ids))
    per_page_docx: dict[int, int] = Counter(
        page_by_id[column_id] for column_id in represented_unique_ids if column_id in page_by_id
    )
    if expected_ids:
        audit = doc.metadata.column_ocr_audit
        for page_key, report in audit.get("pages", {}).items():
            page_no = int(page_key)
            report["docx_written"] = int(per_page_docx.get(page_no, 0))
            report["docx_passed"] = (
                int(report.get("expected", 0))
                == int(report.get("recognized", 0))
                == int(report.get("model_written", 0))
                == int(report.get("docx_written", 0))
            )
        totals = audit.setdefault("totals", {})
        totals["docx_written"] = len(represented_unique_ids)
        last_page = int(audit.get("last_ocr_page", 0) or 0)
        last_report = audit.get("pages", {}).get(str(last_page), {})
        audit["last_page_all_columns_exported"] = bool(last_report.get("docx_passed"))
        audit["docx_integrity_passed"] = (
            len(represented_unique_ids) == len(expected_ids)
            and all(bool(report.get("docx_passed")) for report in audit.get("pages", {}).values())
            and bool(audit["last_page_all_columns_exported"])
        )
        doc.metadata.column_ocr_integrity_passed = bool(audit["docx_integrity_passed"])

    report = {
        "path": str(target),
        "paragraphs_written": len(written_texts),
        "expected_columns": len(expected_ids),
        "docx_written_columns": len(represented_unique_ids),
        "source_column_references": len(represented_ids),
        "last_page_all_columns_exported": bool(
            (getattr(doc.metadata, "column_ocr_audit", {}) or {}).get("last_page_all_columns_exported", True)
        ),
    }
    if verbose:
        size_kb = target.stat().st_size // 1024
        audit_text = ""
        if expected_ids:
            audit_text = f", 列对账 {len(expected_ids)}/{len(represented_unique_ids)}"
        print(f"✅  Word 已生成: {target}  ({size_kb} KB, {len(written_texts)} 段{audit_text})")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UnifiedDocument → Word (.docx)")
    parser.add_argument("input_json", help="Formatter 输出的 JSON")
    parser.add_argument("output_docx", help="输出 .docx 路径")
    parser.add_argument("--horizontal", action="store_true", help="横排模式（默认竖排）")
    parser.add_argument("--no-page-breaks", action="store_true", help="不按原书页插入分页符")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()
    with open(args.input_json, encoding="utf-8") as fh:
        document = UnifiedDocument.from_json(fh.read())
    build_word(
        document,
        output_path=args.output_docx,
        vertical=not args.horizontal,
        page_breaks=not args.no_page_breaks,
        verbose=not args.quiet,
    )
