# -*- coding: utf-8 -*-
"""Japanese OCR tolerant normalizer. Used only for matching scores."""
import unicodedata

MAP = {
    "ツ":"ッ",
    "ヽ":"ゝ",
    "―":"ー",
    "─":"ー",
    "−":"ー",
    "ｰ":"ー",
    "　":" ",
    "口":"ロ",
    "工":"エ",
}

def normalize_japanese(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    for a,b in MAP.items():
        text = text.replace(a,b)
    return text

def compare_key(text):
    return "".join(normalize_japanese(text).split())
