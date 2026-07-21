
import difflib
import re

def normalize_for_match(text: str) -> str:
    if not text:
        return ""
    text = text.replace("　", " ")
    text = re.sub(r"\s+", "", text)
    return text

def similarity(a: str, b: str) -> float:
    a = normalize_for_match(a)
    b = normalize_for_match(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()

def best_match(source, candidates):
    best = None
    score = 0.0
    for item in candidates:
        s = similarity(source, item)
        if s > score:
            score = s
            best = item
    return best, score
