"""Optional LTR (horizontal) conversion before EPUB export.

Covers ``builder/text_direction.py`` and its integration in
``builder.epub_builder.build_epub``.  No ``ui`` package (PySide6) is imported.
"""
from __future__ import annotations

import re
import unicodedata
import zipfile
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from builder.epub_builder import CSS_TEMPLATES, _xhtml_wrap, build_epub
from builder.text_direction import (
    VERTICAL_TO_HORIZONTAL_PUNCT,
    TextDirectionMode,
    coerce_text_direction,
    convert_css_to_ltr,
    convert_xhtml_to_ltr,
    has_vertical_writing_mode,
    ltr_stylesheet,
    normalize_punctuation_for_horizontal,
)
from engine.illustration_spread import detect_likely_illustration_spreads
from models.document import Block, BlockType, PageInfo, UnifiedDocument

_ROOT = Path(__file__).resolve().parent.parent

# Pairs exactly as listed in the task specification.
_SPEC_PAIRS = [
    ("︒", "。"), ("︑", "、"), ("︐", "，"), ("︓", "："), ("︔", "；"),
    ("︕", "！"), ("︖", "？"), ("︵", "（"), ("︶", "）"), ("︷", "｛"),
    ("︸", "｝"), ("︹", "〔"), ("︺", "〕"), ("︻", "【"), ("︼", "】"),
    ("︽", "《"), ("︾", "》"), ("︿", "〈"), ("﹀", "〉"), ("﹁", "「"),
    ("﹂", "」"), ("﹃", "『"), ("﹄", "』"), ("︙", "…"), ("︰", "‥"),
    ("︱", "—"), ("︲", "–"), ("﹇", "［"), ("﹈", "］"),
]


# ── punctuation table ────────────────────────────────────────────────────────

def test_punctuation_table_matches_unicode_names() -> None:
    prefix = "PRESENTATION FORM FOR VERTICAL "
    for vertical, horizontal in VERTICAL_TO_HORIZONTAL_PUNCT.items():
        code = ord(vertical)
        assert 0xFE10 <= code <= 0xFE19 or 0xFE30 <= code <= 0xFE4F, hex(code)
        vertical_name = unicodedata.name(vertical)
        assert vertical_name.startswith(prefix), vertical_name
        # U+FE18 carries the official Unicode misspelling "BRAKCET".
        derived = vertical_name.removeprefix(prefix).replace("BRAKCET", "BRACKET")
        assert unicodedata.name(horizontal).removeprefix("FULLWIDTH ") == derived, (hex(code), derived)


def test_punctuation_table_has_no_collisions_and_covers_spec() -> None:
    keys = list(VERTICAL_TO_HORIZONTAL_PUNCT)
    values = list(VERTICAL_TO_HORIZONTAL_PUNCT.values())
    assert len(set(values)) == len(values)
    assert not set(keys) & set(values)
    assert all(len(key) == 1 and len(value) == 1 for key, value in VERTICAL_TO_HORIZONTAL_PUNCT.items())
    for vertical, horizontal in _SPEC_PAIRS:
        assert VERTICAL_TO_HORIZONTAL_PUNCT.get(vertical) == horizontal, (hex(ord(vertical)), horizontal)


def test_normalize_all_vertical_forms() -> None:
    source = "".join(VERTICAL_TO_HORIZONTAL_PUNCT)
    expected = "".join(VERTICAL_TO_HORIZONTAL_PUNCT.values())
    assert normalize_punctuation_for_horizontal(source) == expected
    assert normalize_punctuation_for_horizontal(expected) == expected


def test_normalize_keeps_ordinary_japanese() -> None:
    text = "「こんにちは」と彼は言った。『はい』、そうです……――【完】（笑）！？"
    assert normalize_punctuation_for_horizontal(text) == text


def test_coerce_text_direction() -> None:
    assert coerce_text_direction(None) is TextDirectionMode.KEEP
    assert coerce_text_direction("") is TextDirectionMode.KEEP
    assert coerce_text_direction("LTR") is TextDirectionMode.LTR
    assert coerce_text_direction(TextDirectionMode.KEEP) is TextDirectionMode.KEEP
    with pytest.raises(ValueError):
        coerce_text_direction("rtl-please")


# ── XHTML conversion ─────────────────────────────────────────────────────────

_VERTICAL_XHTML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="ja" lang="ja" dir="rtl">
<head><title>︽題︾</title><style>p { writing-mode: vertical-rl; color: red; }</style><script>var s = "︒";</script></head>
<body dir="rtl" style="writing-mode: vertical-rl; margin: 0">
<p style="-webkit-writing-mode: vertical-rl">彼は<span class="tcy">12</span>歳<span style="text-combine-upright: all">!?</span>︒</p>
<p><ruby>漢字<rt>かんじ</rt></ruby>﹁はい﹂</p>
<p title="︒属性">⟦插图｜︒marker⟧本文︑</p>
<!-- ︒ comment -->
<span id="keep" class="tcy">34</span>
</body>
</html>
"""


def test_convert_xhtml_to_ltr_rewrites_direction_and_markup() -> None:
    out = convert_xhtml_to_ltr(_VERTICAL_XHTML)
    assert 'lang="ja" dir="ltr">' in out
    assert '<body dir="ltr" style="margin: 0">' in out
    assert "rtl" not in out
    assert "vertical-rl" not in out
    assert "<style>p { color: red; }</style>" in out
    assert "<p>彼は12歳!?。</p>" in out
    assert "<ruby>漢字<rt>かんじ</rt></ruby>「はい」" in out
    assert "<title>《題》</title>" in out


def test_convert_xhtml_to_ltr_leaves_attributes_markers_scripts_comments() -> None:
    out = convert_xhtml_to_ltr(_VERTICAL_XHTML)
    assert '<p title="︒属性">⟦插图｜︒marker⟧本文、</p>' in out
    assert '<script>var s = "︒";</script>' in out
    assert "<!-- ︒ comment -->" in out
    # A tcy span with other semantics (id) is kept.
    assert '<span id="keep" class="tcy">34</span>' in out


def test_convert_xhtml_to_ltr_is_idempotent() -> None:
    once = convert_xhtml_to_ltr(_VERTICAL_XHTML)
    assert convert_xhtml_to_ltr(once) == once


def test_builder_xhtml_already_ltr_is_unchanged_byte_for_byte() -> None:
    page = _xhtml_wrap("第一章", '  <section epub:type="chapter">\n    <p class="normal">本文です。</p>\n  </section>', lang="ja")
    converted = convert_xhtml_to_ltr(page)
    expected = page.replace('lang="ja">', 'lang="ja" dir="ltr">', 1).replace("<body>", '<body dir="ltr">', 1)
    assert converted == expected
    assert convert_xhtml_to_ltr(converted) == converted


# ── CSS conversion ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("template", ["denki", "mf"])
def test_convert_css_to_ltr_replaces_vertical_rules(template: str) -> None:
    css = CSS_TEMPLATES[template]
    assert has_vertical_writing_mode(css)
    out = convert_css_to_ltr(css)
    assert not has_vertical_writing_mode(out)
    assert "vertical-rl" not in out
    assert "text-combine-upright" not in out
    assert "writing-mode: horizontal-tb" in out
    assert out.count("!important") == css.count("!important")
    assert out.endswith(ltr_stylesheet())
    assert convert_css_to_ltr(out) == out


def test_convert_css_to_ltr_keeps_justify() -> None:
    out = convert_css_to_ltr(CSS_TEMPLATES["denki"])
    assert "text-align: justify;" in out


# ── EPUB build integration ───────────────────────────────────────────────────

def _text_doc(*, with_vertical_punct: bool = True) -> UnifiedDocument:
    doc = UnifiedDocument()
    doc.metadata.title = "テスト"
    doc.metadata.author = "著者"
    blocks = [
        Block(BlockType.CHAPTER, "第一章"),
        Block(BlockType.PARAGRAPH, "　彼は12歳だった！？"),
        Block(BlockType.DIALOGUE, "「そうか」"),
    ]
    if with_vertical_punct:
        blocks.append(Block(BlockType.PARAGRAPH, "︻完︼"))
    doc.blocks = blocks
    return doc


def _normalize(text: str) -> str:
    text = re.sub(r"urn:uuid:[0-9a-f-]+", "urn:uuid:X", text)
    return re.sub(r'<meta property="dcterms:modified">[^<]*</meta>', "", text)


def _build(tmp_path: Path, name: str, doc: UnifiedDocument, **kwargs) -> dict[str, str]:
    path = tmp_path / f"{name}.epub"
    build_epub(doc, str(path), verbose=False, **kwargs)
    with zipfile.ZipFile(path) as archive:
        return {
            member: _normalize(archive.read(member).decode("utf-8"))
            for member in archive.namelist()
            if member.endswith((".opf", ".css", ".xhtml", ".ncx"))
        }


def _spine_direction(files: dict[str, str]) -> str:
    match = re.search(r'page-progression-direction="(\w+)"', files["EPUB/content.opf"])
    assert match
    return match.group(1)


def test_vertical_ltr_build_is_horizontal(tmp_path: Path) -> None:
    files = _build(tmp_path, "ltr", _text_doc(), vertical=True, text_direction=TextDirectionMode.LTR)
    css = files["EPUB/styles/style.css"]
    assert _spine_direction(files) == "ltr"
    assert not any("vertical-rl" in text for text in files.values())
    assert "writing-mode: horizontal-tb" in css
    for member, text in files.items():
        if member.endswith(".xhtml"):
            assert '<body dir="ltr">' in text, member
            assert 'lang="ja" dir="ltr">' in text, member
    chapter = files["EPUB/content/ch001.xhtml"]
    assert 'class="tcy"' not in chapter
    assert "【完】" in chapter and "︻" not in chapter


def test_vertical_ltr_build_passes_quality_gate(tmp_path: Path) -> None:
    from utils.epub_quality_gate import validate_epub_quality

    path = tmp_path / "gate.epub"
    build_epub(_text_doc(), str(path), vertical=True, verbose=False, text_direction="ltr")
    report = validate_epub_quality(path, expect_vertical=False)
    assert report.valid, report.errors


def test_vertical_keep_matches_default(tmp_path: Path) -> None:
    default = _build(tmp_path, "default", _text_doc(), vertical=True)
    keep = _build(tmp_path, "keep", _text_doc(), vertical=True, text_direction="keep")
    assert keep == default
    assert _spine_direction(keep) == "rtl"
    assert "writing-mode: vertical-rl" in keep["EPUB/styles/style.css"]
    assert 'class="tcy"' in keep["EPUB/content/ch001.xhtml"]
    assert "dir=" not in keep["EPUB/content/ch001.xhtml"]


def test_horizontal_ltr_only_adds_dir(tmp_path: Path) -> None:
    keep = _build(tmp_path, "h_keep", _text_doc(with_vertical_punct=False), vertical=False)
    ltr = _build(tmp_path, "h_ltr", _text_doc(with_vertical_punct=False), vertical=False, text_direction="ltr")
    assert set(ltr) == set(keep)
    assert ltr["EPUB/content.opf"] == keep["EPUB/content.opf"]
    assert ltr["EPUB/styles/style.css"] == keep["EPUB/styles/style.css"]
    assert _spine_direction(ltr) == "ltr"
    for member, text in keep.items():
        if member.endswith(".xhtml"):
            assert ltr[member].replace(' dir="ltr"', "") == text, member
            assert '<body dir="ltr">' in ltr[member]


# ── double-page illustration order ───────────────────────────────────────────

def _spread_halves(tmp_path: Path) -> tuple[Path, Path]:
    """Split one artwork into (right half, left half) = RTL reading order."""
    art = Image.new("RGB", (400, 300))
    draw = ImageDraw.Draw(art)
    for y in range(300):
        draw.line([(0, y), (399, y)], fill=((y * 7) % 200, (y * 13) % 200, (y * 29) % 200))
    draw.rectangle([(20, 20), (100, 60)], fill=(0, 0, 255))
    draw.rectangle([(300, 20), (380, 60)], fill=(255, 0, 0))
    right_half = tmp_path / "page1_right_half.png"
    left_half = tmp_path / "page2_left_half.png"
    art.crop((200, 0, 400, 300)).save(right_half)
    art.crop((0, 0, 200, 300)).save(left_half)
    return right_half, left_half


def _spread_doc(tmp_path: Path) -> UnifiedDocument:
    right_half, left_half = _spread_halves(tmp_path)
    doc = UnifiedDocument()
    doc.metadata.title = "見開き"
    doc.pages = [
        PageInfo(page_no=1, page_type=BlockType.ILLUSTRATION, image_path=str(right_half)),
        PageInfo(page_no=2, page_type=BlockType.ILLUSTRATION, image_path=str(left_half)),
    ]
    doc.blocks = [
        Block(BlockType.CHAPTER, "第一章"),
        Block(BlockType.PARAGRAPH, "前の本文。"),
        Block(BlockType.IMAGE_REF, page=1, image_path=str(right_half)),
        Block(BlockType.IMAGE_REF, page=2, image_path=str(left_half)),
        Block(BlockType.PARAGRAPH, "後の本文。"),
    ]
    return doc


def _spine(files: dict[str, str]) -> list[tuple[str, str]]:
    return re.findall(r'<itemref idref="([^"]+)"(?: properties="([^"]+)")?/>', files["EPUB/content.opf"])


def test_spread_fixture_is_detected(tmp_path: Path) -> None:
    pairs = detect_likely_illustration_spreads(_spread_doc(tmp_path))
    (pair,) = pairs
    assert (pair.first_page, pair.second_page) == (1, 2)


def test_vertical_keep_spread_order_unchanged(tmp_path: Path) -> None:
    files = _build(tmp_path, "spread_keep", _spread_doc(tmp_path), vertical=True)
    assert _spine(files) == [
        ("ch001", ""),
        ("ch001-img-001", "page-spread-right"),
        ("ch001-img-002", "page-spread-left"),
        ("ch001_p2", ""),
    ]


def test_vertical_ltr_spread_halves_follow_horizontal_reading(tmp_path: Path) -> None:
    files = _build(tmp_path, "spread_ltr", _spread_doc(tmp_path), vertical=True, text_direction="ltr")
    assert _spine(files) == [
        ("ch001", ""),
        ("ch001-img-002", "page-spread-left"),
        ("ch001-img-001", "page-spread-right"),
        ("ch001_p2", ""),
    ]
    # The left-hand page shows the left half of the artwork.
    left_page = files["EPUB/content/ch001_illus_img-002.xhtml"]
    assert "images/illus_002.png" in left_page
    assert _spine_direction(files) == "ltr"


# ── UI localization (file-level, no PySide6) ─────────────────────────────────

def test_ui_strings_are_localized() -> None:
    tab = (_ROOT / "ui" / "epub" / "tab.py").read_text(encoding="utf-8")
    localization = (_ROOT / "ui" / "localization.py").read_text(encoding="utf-8")
    label = "转换为从左到右横排版式"
    tooltip = "导出前将竖排转为从左到右横排：横排 CSS、ltr 翻页、竖排标点改为横排字形"
    for key in (label, tooltip):
        assert f'"{key}"' in tab
        assert localization.count(f'"{key}"') == 2
    assert '"Convert to left-to-right (horizontal) layout"' in localization
