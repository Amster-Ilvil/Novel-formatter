"""Anchor scoring utilities for v2.4 alignment."""

def anchor_score(text):
    if not text:
        return 0.0
    return min(1.0, len(text)/20.0)
