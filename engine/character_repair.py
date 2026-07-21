"""v2.3 character level OCR repair engine."""

import difflib

def repair_candidates(ocr_text, gt_text):
    sm=difflib.SequenceMatcher(None, ocr_text, gt_text)
    return [op for op in sm.get_opcodes() if op[0] != "equal"]

def apply_safe_repair(ocr_text, gt_text, confidence=0.0):
    if confidence < 0.85:
        return ocr_text
    return gt_text
