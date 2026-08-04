#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reusable segmentation helpers for printed Japanese vertical text.

These modules implement clean-room geometry inspired by common OCR pipelines:
projection valleys, connected-component centres, and ruby/furigana filtering.
They intentionally do not import external OCR repositories or add runtime model
requirements.
"""

from .center_detector import CenterIntervalResult, detect_center_intervals
from .ruby_filter import RubyFilterResult, classify_vertical_ruby

__all__ = [
    "CenterIntervalResult",
    "RubyFilterResult",
    "classify_vertical_ruby",
    "detect_center_intervals",
]
