# -*- coding: utf-8 -*-
"""v2.4.1 OCR consumed span tracker."""
from dataclasses import dataclass

@dataclass
class SpanUsage:
    start:int
    end:int
    confidence:float=0.0

class ConsumedSpanTracker:
    def __init__(self):
        self.spans=[]
        self.blocked_duplicate=0

    def overlap_ratio(self,a,b):
        left=max(a[0],b[0]); right=min(a[1],b[1])
        if right<=left:return 0.0
        return (right-left)/max(min(a[1]-a[0],b[1]-b[0]),1)

    def check(self,start,end,confidence=0.0):
        for s in self.spans:
            if self.overlap_ratio((start,end),(s.start,s.end))>0.8:
                self.blocked_duplicate+=1
                return False
        return True

    def consume(self,start,end,confidence=0.0):
        if self.check(start,end,confidence):
            self.spans.append(SpanUsage(start,end,confidence))
            return True
        return False
