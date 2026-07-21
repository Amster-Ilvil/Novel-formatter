# -*- coding: utf-8 -*-
"""Safe Japanese OCR line reconstruction."""
import re

def can_join(left,right,gt_candidates=None):
    if not left or not right:return False
    if gt_candidates and left+right in gt_candidates:return True
    if re.search(r'[一-龯ぁ-んァ-ン]$',left) and re.match(r'^[一-龯ぁ-んァ-ン]',right):
        return False
    return False

def reconstruct(lines,gt_candidates=None):
    out=[]; i=0
    while i<len(lines):
        if i+1<len(lines) and can_join(lines[i],lines[i+1],gt_candidates):
            out.append(lines[i]+lines[i+1]); i+=2
        else:
            out.append(lines[i]); i+=1
    return out
