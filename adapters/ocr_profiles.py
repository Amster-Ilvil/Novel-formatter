#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent OCR profiles for Japanese vertical and Simplified Chinese horizontal text.

The profile boundary is deliberately small and side-effect free.  UI code chooses
one profile and snapshots it at run start; adapters consume the immutable profile
key.  Japanese column detection, ruby cleanup, handwriting review and sentence
reflow remain available only to the Japanese profile.  The Chinese profile uses
whole-page horizontal reading order and never enters the vertical-column pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

JA_VERTICAL = "ja_vertical"
ZH_HANS_HORIZONTAL = "zh_hans_horizontal"
DEFAULT_OCR_MODE = JA_VERTICAL
PROFILE_VERSION = 1


@dataclass(frozen=True)
class OCRProfile:
    key: str
    label: str
    language: str
    writing_direction: str
    vertical: bool
    apple_languages: tuple[str, ...]
    paddle_lang: str
    google_language_hints: tuple[str, ...]
    compatible_engines: frozenset[str]
    allow_column_pipeline: bool
    allow_japanese_handwriting: bool
    preserve_layout_by_default: bool


_PROFILES = {
    JA_VERTICAL: OCRProfile(
        key=JA_VERTICAL,
        label="日文竖排",
        language="ja",
        writing_direction="vertical-rl",
        vertical=True,
        apple_languages=("ja-JP",),
        paddle_lang="japan",
        google_language_hints=("ja",),
        compatible_engines=frozenset({
            "apple_vision", "pdf_craft", "paddle_ocr", "ndlocr_lite",
            "yomitoku", "manga_48px", "manga_ocr", "google_vision",
        }),
        allow_column_pipeline=True,
        allow_japanese_handwriting=True,
        preserve_layout_by_default=False,
    ),
    ZH_HANS_HORIZONTAL: OCRProfile(
        key=ZH_HANS_HORIZONTAL,
        label="简体中文横排",
        language="zh-CN",
        writing_direction="horizontal-tb-lr",
        vertical=False,
        # Apple Vision accepts BCP-47 identifiers.  Put the most specific
        # Simplified-Chinese locale first and retain an English fallback for
        # mixed ISBN/Latin headings without enabling automatic language drift.
        apple_languages=("zh-Hans", "zh-CN", "en-US"),
        paddle_lang="ch",
        google_language_hints=("zh-CN", "zh"),
        compatible_engines=frozenset({"apple_vision", "paddle_ocr", "google_vision"}),
        allow_column_pipeline=False,
        allow_japanese_handwriting=False,
        preserve_layout_by_default=True,
    ),
}

_CHINESE_CHAPTER_RE = re.compile(
    r"^(?:"
    r"第\s*[〇零一二两三四五六七八九十百千万亿0-9０-９]+\s*[章节卷部篇回集幕]"
    r"(?:\s*[:：、.．·\-—]\s*.*)?|"
    r"序章|楔子|引子|前言|序言|后记|跋|尾声|终章|终篇|番外(?:篇)?|"
    r"(?:上|中|下)篇|(?:Chapter|CHAPTER)\s*[0-9０-９IVXLCDM]+"
    r")$",
    re.IGNORECASE,
)
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_SPACE_BETWEEN_CJK_RE = re.compile(
    r"(?<=[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff])\s+"
    r"(?=[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff])"
)
_SPACE_BEFORE_CJK_PUNCT_RE = re.compile(r"\s+([，。！？；：、）》】」』…])")
_SPACE_AFTER_OPEN_CJK_PUNCT_RE = re.compile(r"([（《【「『])\s+")


def normalize_ocr_mode(value: str | None) -> str:
    key = str(value or "").strip().lower()
    aliases = {
        "ja": JA_VERTICAL,
        "jp": JA_VERTICAL,
        "japanese": JA_VERTICAL,
        "vertical": JA_VERTICAL,
        "zh": ZH_HANS_HORIZONTAL,
        "zh-cn": ZH_HANS_HORIZONTAL,
        "zh_hans": ZH_HANS_HORIZONTAL,
        "chinese": ZH_HANS_HORIZONTAL,
        "horizontal": ZH_HANS_HORIZONTAL,
    }
    key = aliases.get(key, key)
    return key if key in _PROFILES else DEFAULT_OCR_MODE


def get_ocr_profile(value: str | None = None) -> OCRProfile:
    return _PROFILES[normalize_ocr_mode(value)]


def is_engine_compatible(engine_id: str, mode: str | None) -> bool:
    return str(engine_id or "") in get_ocr_profile(mode).compatible_engines


def is_chinese_horizontal(mode: str | None) -> bool:
    return normalize_ocr_mode(mode) == ZH_HANS_HORIZONTAL


def is_chapter_title(text: str, mode: str | None) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    if is_chinese_horizontal(mode):
        return bool(_CHINESE_CHAPTER_RE.match(value))
    # Imported lazily to avoid a module cycle during adapter startup.
    from adapters.apple_vision_adapter import CHAPTER_RE
    return bool(CHAPTER_RE.match(value))


def normalize_chinese_text(text: str) -> str:
    """Conservative OCR cleanup for horizontal Simplified Chinese.

    It removes only layout spaces that were inserted *between Han characters*
    or next to full-width Chinese punctuation.  Latin words, numbers, URLs and
    mixed-language spacing are preserved.
    """
    value = unicodedata.normalize("NFC", str(text or ""))
    value = value.replace("\u00a0", " ").replace("\u3000", " ")
    value = _SPACE_BETWEEN_CJK_RE.sub("", value)
    value = _SPACE_BEFORE_CJK_PUNCT_RE.sub(r"\1", value)
    value = _SPACE_AFTER_OPEN_CJK_PUNCT_RE.sub(r"\1", value)
    return value.strip()


def normalize_lines(lines: list[str], mode: str | None) -> list[str]:
    if not is_chinese_horizontal(mode):
        return [str(line).strip() for line in lines if str(line).strip()]
    return [
        cleaned for line in lines
        if (cleaned := normalize_chinese_text(str(line)))
    ]


def classify_page_lines(lines: list[str], mode: str | None):
    """Profile-aware page classification without importing a concrete OCR engine."""
    from models.document import BlockType

    cleaned = normalize_lines(lines, mode)
    joined = "".join(cleaned)
    char_count = len(joined)
    punct = set("。、「」『』,.!?！？…・：；，、（）《》【】‘’“”")
    punct_count = sum(1 for ch in joined if ch in punct)
    if any(is_chapter_title(line, mode) for line in cleaned):
        return BlockType.PARAGRAPH
    if char_count < 5:
        return BlockType.BLANK
    if char_count < 30:
        return BlockType.ILLUSTRATION
    if char_count < 90 and punct_count < 3:
        return BlockType.ILLUSTRATION
    return BlockType.PARAGRAPH


def auto_classify_pages(image_paths: list[str], lines_per_page: list[list[str]], mode: str | None):
    from models.document import BlockType

    result = []
    for index, _path in enumerate(image_paths):
        lines = lines_per_page[index] if index < len(lines_per_page) else []
        page_type = classify_page_lines(lines, mode)
        if index == 0 and page_type in {BlockType.BLANK, BlockType.ILLUSTRATION}:
            page_type = BlockType.COVER
        result.append(page_type)
    return result


def apply_profile_metadata(doc, mode: str | None) -> None:
    profile = get_ocr_profile(mode)
    doc.metadata.language = profile.language
    doc.metadata.ocr_mode = profile.key
    doc.metadata.writing_direction = profile.writing_direction
    doc.metadata.ocr_profile_version = PROFILE_VERSION
    doc.metadata.formatter_profile = (
        "zh_hans_horizontal" if profile.key == ZH_HANS_HORIZONTAL else "ja_light_novel"
    )
    if profile.preserve_layout_by_default:
        doc.metadata.preserve_ocr_layout = True


__all__ = [
    "JA_VERTICAL", "ZH_HANS_HORIZONTAL", "DEFAULT_OCR_MODE", "OCRProfile",
    "get_ocr_profile", "normalize_ocr_mode", "is_engine_compatible",
    "is_chinese_horizontal", "is_chapter_title", "normalize_chinese_text",
    "normalize_lines", "classify_page_lines", "auto_classify_pages",
    "apply_profile_metadata",
]
