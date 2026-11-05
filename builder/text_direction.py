# -*- coding: utf-8 -*-
"""Optional conversion of vertical Japanese EPUB output to horizontal LTR.

The module is pure (no Qt, no ``adapters``) and works on strings only.  The
EPUB builder emits XHTML/CSS from string templates, so a small regex tokenizer
is used instead of a DOM parser: untouched bytes stay byte-identical, which
keeps the conversion idempotent and adds no dependency.
"""
from __future__ import annotations

import re
from enum import Enum

__all__ = [
    "TextDirectionMode",
    "VERTICAL_TO_HORIZONTAL_PUNCT",
    "coerce_text_direction",
    "normalize_punctuation_for_horizontal",
    "convert_xhtml_to_ltr",
    "convert_css_to_ltr",
    "has_vertical_writing_mode",
    "ltr_stylesheet",
]


class TextDirectionMode(str, Enum):
    """Export-time text direction handling.

    ``KEEP`` reproduces the historical builder output exactly; ``LTR`` forces a
    horizontal left-to-right publication.
    """

    KEEP = "keep"
    LTR = "ltr"


def coerce_text_direction(value: TextDirectionMode | str | None) -> TextDirectionMode:
    """Accept an enum member, its value/name (case-insensitive) or ``None``."""
    if isinstance(value, TextDirectionMode):
        return value
    key = str(value or "").strip().lower()
    if not key:
        return TextDirectionMode.KEEP
    try:
        return TextDirectionMode(key)
    except ValueError:
        raise ValueError(f"Unknown text direction mode: {value!r}") from None


# Vertical presentation forms (U+FE10–FE19, U+FE30–FE4F) → ordinary forms.
VERTICAL_TO_HORIZONTAL_PUNCT: dict[str, str] = {
    "\ufe10": "\uff0c",  # ︐ → ，
    "\ufe11": "\u3001",  # ︑ → 、
    "\ufe12": "\u3002",  # ︒ → 。
    "\ufe13": "\uff1a",  # ︓ → ：
    "\ufe14": "\uff1b",  # ︔ → ；
    "\ufe15": "\uff01",  # ︕ → ！
    "\ufe16": "\uff1f",  # ︖ → ？
    "\ufe17": "\u3016",  # ︗ → 〖
    "\ufe18": "\u3017",  # ︘ → 〗
    "\ufe19": "\u2026",  # ︙ → …
    "\ufe30": "\u2025",  # ︰ → ‥
    "\ufe31": "\u2014",  # ︱ → —
    "\ufe32": "\u2013",  # ︲ → –
    "\ufe35": "\uff08",  # ︵ → （
    "\ufe36": "\uff09",  # ︶ → ）
    "\ufe37": "\uff5b",  # ︷ → ｛
    "\ufe38": "\uff5d",  # ︸ → ｝
    "\ufe39": "\u3014",  # ︹ → 〔
    "\ufe3a": "\u3015",  # ︺ → 〕
    "\ufe3b": "\u3010",  # ︻ → 【
    "\ufe3c": "\u3011",  # ︼ → 】
    "\ufe3d": "\u300a",  # ︽ → 《
    "\ufe3e": "\u300b",  # ︾ → 》
    "\ufe3f": "\u3008",  # ︿ → 〈
    "\ufe40": "\u3009",  # ﹀ → 〉
    "\ufe41": "\u300c",  # ﹁ → 「
    "\ufe42": "\u300d",  # ﹂ → 」
    "\ufe43": "\u300e",  # ﹃ → 『
    "\ufe44": "\u300f",  # ﹄ → 』
    "\ufe47": "\uff3b",  # ﹇ → ［
    "\ufe48": "\uff3d",  # ﹈ → ］
}

_PUNCT_TRANSLATION = str.maketrans(VERTICAL_TO_HORIZONTAL_PUNCT)
_VERTICAL_CHAR_RE = re.compile("[" + "".join(VERTICAL_TO_HORIZONTAL_PUNCT) + "]")
# Internal asset markers such as ``⟦插图｜…⟧`` must survive byte-for-byte.
_MARKER_RE = re.compile(r"⟦[^⟦⟧]*⟧")


def normalize_punctuation_for_horizontal(text: str) -> str:
    """Replace vertical presentation forms only; ``「」。、`` are left alone."""
    return str(text or "").translate(_PUNCT_TRANSLATION)


def _normalize_text_node(text: str) -> str:
    if not _VERTICAL_CHAR_RE.search(text):
        return text
    parts: list[str] = []
    last = 0
    for match in _MARKER_RE.finditer(text):
        parts.append(normalize_punctuation_for_horizontal(text[last:match.start()]))
        parts.append(match.group(0))
        last = match.end()
    parts.append(normalize_punctuation_for_horizontal(text[last:]))
    return "".join(parts)


# ── CSS declarations ─────────────────────────────────────────────────────────

_WRITING_MODE_PROPS = {"writing-mode", "-epub-writing-mode", "-webkit-writing-mode"}
_COMBINE_PROPS = {
    "text-combine-upright",
    "-webkit-text-combine-upright",
    "-webkit-text-combine",
    "-epub-text-combine",
}
_ORIENTATION_PROPS = {"text-orientation", "-webkit-text-orientation", "-epub-text-orientation"}
_VERTICAL_VALUE_RE = re.compile(r"^\s*(?:vertical-|tb-|tb\b)", re.I)

_CSS_DECL_RE = re.compile(
    r"(?<![\w-])(?P<prop>(?:-epub-|-webkit-)?"
    r"(?:writing-mode|text-orientation|text-combine(?:-upright)?))"
    r"\s*:\s*(?P<val>[^;{}]*)(?:;[ \t]*|(?=\}))",
    re.I,
)

_LTR_MARKER = "/* novel-formatter: horizontal-tb ltr */"


def _vertical_declaration_kind(prop: str, value: str) -> str | None:
    prop = prop.strip().lower()
    if prop in _WRITING_MODE_PROPS:
        return "writing-mode" if _VERTICAL_VALUE_RE.match(value) else None
    if prop in _COMBINE_PROPS:
        return "combine"
    if prop in _ORIENTATION_PROPS:
        return "orientation"
    return None


def _strip_vertical_css(css: str) -> str:
    """Drop vertical-only declarations; lines emptied by the removal vanish."""

    def _sub(match: re.Match[str]) -> str:
        return "" if _vertical_declaration_kind(match.group("prop"), match.group("val")) else match.group(0)

    lines: list[str] = []
    for line in str(css or "").splitlines(keepends=True):
        new_line = _CSS_DECL_RE.sub(_sub, line)
        if new_line != line and not new_line.strip():
            continue
        lines.append(new_line)
    return "".join(lines)


def has_vertical_writing_mode(css: str) -> bool:
    """Return True when the stylesheet declares any vertical writing mode."""
    return any(
        _vertical_declaration_kind(m.group("prop"), m.group("val")) == "writing-mode"
        for m in _CSS_DECL_RE.finditer(str(css or ""))
    )


def ltr_stylesheet() -> str:
    """Horizontal LTR root rules appended in place of the vertical ones.

    ``text-align: start`` sits on ``html, body`` only, so more specific template
    rules such as ``p { text-align: justify; }`` keep their justification.
    """
    return (
        f"{_LTR_MARKER}\n"
        "html, body {\n"
        "    writing-mode: horizontal-tb;\n"
        "    -epub-writing-mode: horizontal-tb;\n"
        "    -webkit-writing-mode: horizontal-tb;\n"
        "    direction: ltr;\n"
        "    text-align: start;\n"
        "}\n"
    )


def convert_css_to_ltr(css: str) -> str:
    """Replace vertical rules with :func:`ltr_stylesheet` (no ``!important``).

    Vertical ``writing-mode``, ``text-orientation`` and ``text-combine*``
    declarations are removed at generation time; the LTR block is appended once.
    """
    stripped = _strip_vertical_css(css)
    if _LTR_MARKER in stripped:
        return stripped
    base = stripped.rstrip("\n")
    return (base + "\n\n" if base else "") + ltr_stylesheet()


# ── XHTML ────────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(
    r"<!--.*?-->"
    r"|<!$$CDATA$$.*?$$$$>"
    r"|<\?.*?\?>"
    r"|<![^>]*>"
    r"|<(?P<raw>script|style)\b[^>]*(?<!/)>.*?</(?P=raw)\s*>"
    r"|</?[A-Za-z][^\s/>]*(?:\s+[^\s=/>]+(?:\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s\"'>]+))?)*\s*/?>",
    re.S | re.I,
)
_TAG_NAME_RE = re.compile(r"^<(/?)([A-Za-z][^\s/>]*)")
_ATTR_RE = re.compile(r"([^\s=/>]+)(?:\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s\"'>]+))?")
_STYLE_ATTR_RE = re.compile(r"(\s+)style(\s*=\s*)([\"'])(.*?)\3", re.I | re.S)
_DIR_ATTR_RE = re.compile(r"(\sdir\s*=\s*)([\"'])(.*?)\2", re.I | re.S)
_TCY_CLASSES = {"tcy", "tate-chu-yoko", "tatechuyoko", "text-combine"}


def _tag_attributes(token: str) -> dict[str, str]:
    head = _TAG_NAME_RE.match(token)
    body = token[head.end() if head else 0:].rstrip(">").rstrip("/")
    attrs: dict[str, str] = {}
    for name, raw in _ATTR_RE.findall(body):
        value = raw[1:-1] if raw[:1] in {'"', "'"} else raw
        attrs[name.lower()] = value
    return attrs


def _strip_inline_style(token: str) -> tuple[str, bool]:
    """Remove vertical inline declarations. Returns (token, removed_combine)."""
    match = _STYLE_ATTR_RE.search(token)
    if not match:
        return token, False
    kept: list[str] = []
    removed = False
    removed_combine = False
    for decl in match.group(4).split(";"):
        prop, sep, value = decl.partition(":")
        kind = _vertical_declaration_kind(prop, value) if sep else None
        if kind:
            removed = True
            removed_combine = removed_combine or kind == "combine"
        elif decl.strip():
            kept.append(decl.strip())
    if not removed:
        return token, False
    if kept:
        quote = match.group(3)
        replacement = f"{match.group(1)}style{match.group(2)}{quote}{'; '.join(kept)}{quote}"
    else:
        replacement = ""
    return token[:match.start()] + replacement + token[match.end():], removed_combine


def _set_dir_ltr(token: str) -> str:
    match = _DIR_ATTR_RE.search(token)
    if match:
        if match.group(3).strip().lower() == "ltr":
            return token
        return token[:match.start(3)] + "ltr" + token[match.end(3):]
    cut = len(token) - (2 if token.endswith("/>") else 1)
    head = token[:cut].rstrip()
    return head + ' dir="ltr"' + token[len(head):] if token[len(head):cut].strip() == "" else token


def _is_tcy_wrapper(original: str, converted: str, removed_combine: bool) -> bool:
    original_classes = set(_tag_attributes(original).get("class", "").split())
    if not (removed_combine or original_classes & _TCY_CLASSES):
        return False
    attrs = _tag_attributes(converted)
    if set(attrs) - {"class"}:
        return False
    return set(attrs.get("class", "").split()) <= _TCY_CLASSES


def _convert_tag(token: str, span_stack: list[bool]) -> str:
    head = _TAG_NAME_RE.match(token)
    if not head:
        return token
    closing = head.group(1) == "/"
    name = head.group(2).lower()
    if closing:
        if name == "span" and span_stack:
            return "" if span_stack.pop() else token
        return token
    converted, removed_combine = _strip_inline_style(token)
    if name in {"html", "body"}:
        converted = _set_dir_ltr(converted)
    if name == "span" and not token.endswith("/>"):
        unwrap = _is_tcy_wrapper(token, converted, removed_combine)
        span_stack.append(unwrap)
        if unwrap:
            return ""
    return converted


def _convert_raw_element(token: str, name: str) -> str:
    if name.lower() != "style":
        return token
    open_end = token.index(">") + 1
    close_start = token.lower().rfind("</")
    inner = token[open_end:close_start]
    converted = _strip_vertical_css(inner)
    return token if converted == inner else token[:open_end] + converted + token[close_start:]


def convert_xhtml_to_ltr(xhtml: str) -> str:
    """Convert one XHTML content document to horizontal LTR.

    * ``dir="ltr"`` on ``<html>`` and ``<body>`` (an existing ``rtl`` is replaced);
    * inline vertical ``writing-mode`` / ``text-orientation`` / ``text-combine*``
      declarations are removed; an emptied ``style`` attribute disappears;
    * tate-chu-yoko wrappers (``<span class="tcy">`` or a bare span whose only
      styling was ``text-combine-upright``) are unwrapped into plain text;
    * vertical punctuation forms are normalized in text nodes only, never in
      attributes, comments, ``<script>``/``<style>`` or ``⟦…⟧`` markers;
    * ``<ruby>``/``<rt>``/``<rp>`` markup is preserved as-is.

    The function is idempotent and returns already-LTR horizontal content
    unchanged.
    """
    source = str(xhtml or "")
    out: list[str] = []
    span_stack: list[bool] = []
    pos = 0
    for match in _TOKEN_RE.finditer(source):
        if match.start() > pos:
            out.append(_normalize_text_node(source[pos:match.start()]))
        token = match.group(0)
        raw = match.group("raw")
        if raw:
            out.append(_convert_raw_element(token, raw))
        elif token.startswith(("<!", "<?")):
            out.append(token)
        else:
            out.append(_convert_tag(token, span_stack))
        pos = match.end()
    if pos < len(source):
        out.append(_normalize_text_node(source[pos:]))
    return "".join(out)
