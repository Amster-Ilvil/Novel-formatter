# -*- coding: utf-8 -*-
import difflib

def normalize(t):
    return ''.join((t or '').split())

def is_duplicate(a,b,threshold=0.97):
    return difflib.SequenceMatcher(None,normalize(a),normalize(b)).ratio()>=threshold

def merge_duplicates(blocks):
    out=[]
    for b in blocks:
        if out and is_duplicate(out[-1],b):
            continue
        out.append(b)
    return out
