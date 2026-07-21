"""Passim-style anchor alignment skeleton.
Keeps long document synchronization separate from block replacement.
"""


def find_anchors(source_blocks, target_blocks):
    anchors=[]
    for i,s in enumerate(source_blocks):
        for j,t in enumerate(target_blocks):
            if s and s in t:
                anchors.append((i,j))
                break
    return anchors
