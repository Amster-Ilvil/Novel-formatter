# -*- coding: utf-8 -*-
"""Opt-in, publication-only vertical->horizontal EPUB conversion.

Adapted from the goals of upstream PR #12 for the 2026-11-13 local tree.
The default output never passes through this module. We only alter layout
markup and Unicode vertical presentation forms; original document blocks,
Ruby annotations, illustration paths and canonical adjudication remain intact.
"""
from __future__ import annotations

import re
from enum import Enum


class TextDirectionMode(str, Enum):
    KEEP = 'keep'
    LTR = 'ltr'


def coerce_text_direction(mode: TextDirectionMode | str | None) -> TextDirectionMode:
    if mode is None or str(mode).strip() == '':
        return TextDirectionMode.KEEP
    if isinstance(mode, TextDirectionMode):
        return mode
    return TextDirectionMode(str(mode).strip().lower())


VERTICAL_TO_HORIZONTAL_PUNCT = {
    '\ufe10': '，', '\ufe11': '、', '\ufe12': '。', '\ufe13': '：', '\ufe14': '；',
    '\ufe15': '！', '\ufe16': '？', '\ufe17': '〖', '\ufe18': '〗', '\ufe19': '…',
    '\ufe30': '‥', '\ufe31': '—', '\ufe32': '–', '\ufe35': '（', '\ufe36': '）',
    '\ufe37': '｛', '\ufe38': '｝', '\ufe39': '〔', '\ufe3a': '〕',
    '\ufe3b': '【', '\ufe3c': '】', '\ufe3d': '《', '\ufe3e': '》',
    '\ufe3f': '〈', '\ufe40': '〉', '\ufe41': '「', '\ufe42': '」',
    '\ufe43': '『', '\ufe44': '』', '\ufe47': '［', '\ufe48': '］',
}
_PUNCT_TABLE = str.maketrans(VERTICAL_TO_HORIZONTAL_PUNCT)
_MARKERS = re.compile(r'⟦[^⟦⟧]*⟧')


def normalize_punctuation_for_horizontal(text: str) -> str:
    text = str(text)
    result, prev = [], 0
    for m in _MARKERS.finditer(text):
        result.extend([text[prev:m.start()].translate(_PUNCT_TABLE), m.group(0)])
        prev = m.end()
    result.append(text[prev:].translate(_PUNCT_TABLE))
    return ''.join(result)


# Only delete vertical declarations; ordinary horizontal rules remain untouched.
_DECL = re.compile(
    r'(?<![\w-])(?P<prop>(?:-webkit-|-epub-)?(?:writing-mode|text-orientation|text-combine(?:-upright)?))\s*:\s*(?P<value>[^;{}]*)(?P<semi>;?)',
    re.I,
)
_LTR_MARK = '/* novel-formatter optional LTR */'


def _remove_vertical_css(css: str) -> str:
    def replacement(m):
        p, v = m.group('prop').lower(), m.group('value').strip().lower()
        if 'writing-mode' in p:
            return '' if v.startswith(('vertical-', 'tb-', 'tb')) else m.group(0)
        if 'text-orientation' in p or 'text-combine' in p:
            return ''
        return m.group(0)
    return _DECL.sub(replacement, css)


def convert_css_to_ltr(css: str) -> str:
    css = str(css)
    if _LTR_MARK in css:
        return css
    css = _remove_vertical_css(css)
    return css + '\n' + _LTR_MARK + '\nhtml,body{writing-mode:horizontal-tb;-webkit-writing-mode:horizontal-tb;-epub-writing-mode:horizontal-tb;direction:ltr;}\n'


# Regex only tokenizes top-level markup. It never attempts to parse HTML with
# nested regexes: CDATA, comments, script and style are protected as whole tokens.
_TOKEN = re.compile(
    r'<!--.*?-->|<!\[CDATA\[.*?\]\]>|<\?.*?\?>|<![^>]*>|'
    r'<(?P<raw>script|style)\b[^>]*>.*?</(?P=raw)\s*>|<[^>]+>',
    re.I | re.S,
)
_STYLE_ATTR = re.compile(r'(\sstyle\s*=\s*)(["\'])(.*?)\2', re.I | re.S)
_DIR_ATTR = re.compile(r'(\sdir\s*=\s*)(["\'])(.*?)\2', re.I | re.S)
_CLASS_ATTR = re.compile(r'\sclass\s*=\s*["\']([^"\']*)["\']', re.I)
_TCY = {'tcy', 'tate-chu-yoko', 'tatechuyoko', 'text-combine'}


def _set_dir_ltr(tag: str) -> str:
    m = _DIR_ATTR.search(tag)
    if m:
        return tag[:m.start(3)] + 'ltr' + tag[m.end(3):]
    pos = tag.rfind('/>')
    if pos < 0:
        pos = tag.rfind('>')
    return tag[:pos] + ' dir="ltr"' + tag[pos:] if pos >= 0 else tag


def _tag_convert(tag: str) -> str:
    m = _STYLE_ATTR.search(tag)
    if m:
        before = m.group(3)
        after = _remove_vertical_css(before)
        if after != before:
            if after.strip(' ;'):
                replacement = f'{m.group(1)}{m.group(2)}{after}{m.group(2)}'
            else:
                replacement = ''
            tag = tag[:m.start()] + replacement + tag[m.end():]
    if re.match(r'<(?:html|body)\b', tag, re.I):
        tag = _set_dir_ltr(tag)
    return tag


def convert_xhtml_to_ltr(xhtml: str) -> str:
    """Convert emitted XHTML without touching attributes, scripts, comments, ruby.

    Existing `tcy` spans are kept as harmless inline spans. Their CSS combine
    rule is removed, so the text reads horizontally; keeping element IDs and
    markup avoids breaking anchored Ruby or semantic references.
    """
    source = str(xhtml)
    out, pos = [], 0
    for m in _TOKEN.finditer(source):
        out.append(normalize_punctuation_for_horizontal(source[pos:m.start()]))
        token = m.group(0)
        raw = m.group('raw')
        if raw and raw.lower() == 'style':
            i, j = token.find('>') + 1, token.lower().rfind('</')
            token = token[:i] + _remove_vertical_css(token[i:j]) + token[j:]
        elif raw or token.startswith(('<!--', '<![CDATA[', '<!', '<?')):
            pass
        else:
            token = _tag_convert(token)
        out.append(token)
        pos = m.end()
    out.append(normalize_punctuation_for_horizontal(source[pos:]))
    return ''.join(out)
