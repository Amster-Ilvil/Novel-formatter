"""Dangerous replacement detector for OCR correction."""

DEFAULT_BLOCKLIST=[("人間","入間"),("魔王","磨王")]

def is_suspicious(old,new):
    return (old,new) in DEFAULT_BLOCKLIST
