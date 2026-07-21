"""Long document alignment helper for Novel-formatter v2.4.
Provides sliding-window candidate generation for OCR/GT alignment.
"""

def build_windows(blocks, sizes=(1,2,3,5)):
    result=[]
    for size in sizes:
        for i in range(max(0, len(blocks)-size+1)):
            result.append({"start":i,"end":i+size,"text":"".join(str(x) for x in blocks[i:i+size])})
    return result


def find_candidates(ocr_blocks, gt_blocks, window=3):
    candidates=[]
    for item in build_windows(gt_blocks, (1,2,3,5)[:window]):
        candidates.append(item)
    return candidates
