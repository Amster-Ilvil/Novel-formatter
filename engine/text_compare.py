#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Line-oriented document comparison helpers.

The GUI uses this module to build a stable side-by-side view without modifying
its source documents.  Alignment is deliberately deterministic:

1. exact/normalised anchors are located with ``SequenceMatcher``;
2. changed windows are aligned with a Needleman-Wunsch style dynamic program;
3. character-level opcodes are generated for rich diff rendering.

No OCR correction or semantic rewriting happens here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
from functools import lru_cache
from difflib import SequenceMatcher
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Sequence

from models.document import Block, BlockType, TocEntry, UnifiedDocument

TEXT_TYPES = {
    BlockType.PARAGRAPH,
    BlockType.DIALOGUE,
    BlockType.CHAPTER,
    BlockType.SECTION,
    BlockType.RUBY,
    BlockType.FOOTNOTE,
    BlockType.TOC_ENTRY,
    BlockType.IMAGE_REF,
}

TERMINAL_PUNCTUATION = tuple(
    "。．！？!?‼⁉…‥—―～〜」』）)]】》〉〕〗〙〛：:；;、，,"
)

# “句末缺句号”审校使用更严格的终止符集合。逗号、顿号、冒号和分号
# 通常意味着句子仍在继续，因此需要在文本对比页中提示人工复核。保留
# TERMINAL_PUNCTUATION 供旧逻辑兼容，避免改变其他调用方的语义。
SENTENCE_END_PUNCTUATION = tuple(
    "。．！？!?‼⁉…‥—―～〜」』）)]】》〉〕〗〙〛"
)

NON_PROSE_REVIEW_TYPES = {
    "chapter", "section", "toc_entry", "image_ref",
}

IMAGE_MARKER_RE = re.compile(r"^⟦插图｜(?P<id>[^｜⟧]+)｜(?P<name>.*?)⟧$")
ALIGNMENT_GAP_PREFIX = "⟦对齐占位｜"
CHAPTER_TITLE_RE = re.compile(
    r"^(?:#+\s*)?(?:プロローグ|ブロローグ|エピローグ|序章|終章|幕間|間章|"
    r"第[一二三四五六七八九十百千万〇零0-9]+[章話部巻]|"
    r"[一二三四五六七八九十百千万〇零0-9]+章)(?:\s|　|『|「|$).*"
)


def image_marker_for_block(block: Block) -> str:
    """Return a stable, movable plain-text marker for one illustration block.

    The marker carries the block id so cut/paste in the compare editor also
    moves the real IMAGE_REF instead of creating visible marker text in EPUB.
    """
    name = (block.image_path or "插图").replace("｜", "-").replace("⟧", "]")
    try:
        from pathlib import Path
        name = Path(name).name or "插图"
    except Exception:
        pass
    return f"⟦插图｜{block.id}｜{name}⟧"


def parse_image_marker(text: str) -> str | None:
    match = IMAGE_MARKER_RE.fullmatch((text or "").strip())
    return match.group("id") if match else None


def is_alignment_placeholder(text: str) -> bool:
    """Return True for compact-view rows that must never enter final text."""
    return (text or "").strip().startswith(ALIGNMENT_GAP_PREFIX)


def looks_like_chapter_title(text: str) -> bool:
    stripped = re.sub(r"^\s*#+\s*", "", (text or "").strip())
    return bool(stripped and len(stripped) <= 160 and CHAPTER_TITLE_RE.match(stripped))


@dataclass(slots=True)
class CompareLine:
    text: str
    block_ids: list[str] = field(default_factory=list)
    block_indices: list[int] = field(default_factory=list)
    block_type: str = "paragraph"
    page: int = 0


@dataclass(slots=True)
class AlignedLine:
    left: CompareLine | None
    right: CompareLine | None
    similarity: float = 0.0
    tag: str = "replace"  # equal | replace | delete | insert


@lru_cache(maxsize=65536)
def normalise_for_alignment(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"[\s　]+", "", text)
    # Alignment should not fail merely because punctuation was normalised.
    text = text.translate(str.maketrans({"．": "。", "!": "！", "?": "？"}))
    return text.casefold()


def import_compare_text(path: str | Path) -> tuple[UnifiedDocument, dict[str, int]]:
    """Import TXT/Markdown for side-by-side comparison without blank-row noise.

    Markdown commonly places one empty physical line between every paragraph.
    Treating those empty separators as real document blocks doubles the row count
    and can make SequenceMatcher align thousands of identical empty strings into
    one large blank region.  Only non-empty content rows become compare blocks.
    """
    src = Path(path)
    raw = src.read_text(encoding="utf-8-sig", errors="replace")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    doc = UnifiedDocument()
    doc.metadata.source_engine = f"external_compare:{src.name}"
    doc.metadata.__dict__["external_compare"] = True
    doc.metadata.__dict__["external_compare_source"] = src.name
    skipped_blank = 0
    skipped_decoration = 0
    for physical in raw.split("\n"):
        text = physical.lstrip("\ufeff").rstrip()
        stripped = text.strip()
        if not stripped:
            skipped_blank += 1
            continue
        # Ignore Markdown fences / horizontal rules, but preserve ordinary list
        # bullets and explicit movable image markers.
        if re.fullmatch(r"(?:```+|~~~+|[-_*]{3,})", stripped):
            skipped_decoration += 1
            continue
        if stripped.startswith("#"):
            text = re.sub(r"^\s*#+\s*", "", text).strip()
            btype = BlockType.CHAPTER
        elif looks_like_chapter_title(stripped):
            text = stripped
            btype = BlockType.CHAPTER
        elif stripped.startswith("「") and stripped.endswith("」"):
            text = stripped
            btype = BlockType.DIALOGUE
        else:
            text = stripped
            btype = BlockType.PARAGRAPH
        block = Block(type=btype, text=text, ocr_raw=text, source_format="external_compare")
        block.reading_order = len(doc.blocks)
        doc.blocks.append(block)
    stats = {
        "content_rows": len(doc.blocks),
        "skipped_blank_rows": skipped_blank,
        "skipped_decoration_rows": skipped_decoration,
    }
    doc.metadata.__dict__["external_compare_import_stats"] = dict(stats)
    return doc, stats


def _nonempty_compare_lines(lines: Sequence[CompareLine]) -> list[CompareLine]:
    """Drop physical empty rows before whole-document alignment.

    Manual alignment placeholders live only in the editors.  They are not source
    prose and must not participate in a fresh full-book alignment.
    """
    return [line for line in lines if str(line.text or "").strip() or parse_image_marker(line.text)]


def document_lines(doc: UnifiedDocument | None) -> list[CompareLine]:
    if doc is None:
        return []
    result: list[CompareLine] = []
    for index, block in enumerate(doc.blocks):
        if block.type not in TEXT_TYPES:
            continue
        if (block.metadata or {}).get("consumed"):
            continue
        if block.type == BlockType.IMAGE_REF:
            result.append(CompareLine(
                text=image_marker_for_block(block),
                block_ids=[block.id],
                block_indices=[index],
                block_type=BlockType.IMAGE_REF.value,
                page=int(getattr(block, "page", 0) or 0),
            ))
            continue
        text = (block.text or "").replace("\r\n", "\n").replace("\r", "\n")
        # A legacy block can itself contain hard line breaks.  Keep each physical
        # line visible and attach the original block id to the first row only;
        # subsequent rows become insertable children when applied back.
        parts = text.split("\n")
        emitted_index = 0
        for part in parts:
            if not str(part or "").strip():
                continue
            result.append(CompareLine(
                text=part,
                block_ids=[block.id] if emitted_index == 0 else [],
                block_indices=[index] if emitted_index == 0 else [],
                block_type=block.type.value,
                page=int(getattr(block, "page", 0) or 0),
            ))
            emitted_index += 1
    return result




@lru_cache(maxsize=65536)
def line_similarity(left: str, right: str) -> float:
    a = normalise_for_alignment(left)
    b = normalise_for_alignment(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _pair_tag(left: CompareLine | None, right: CompareLine | None, score: float) -> str:
    if left is None:
        return "insert"
    if right is None:
        return "delete"
    return "equal" if score >= 0.999999 else "replace"


def _zip_fallback(left: Sequence[CompareLine], right: Sequence[CompareLine]) -> list[AlignedLine]:
    rows: list[AlignedLine] = []
    count = max(len(left), len(right))
    for i in range(count):
        l = left[i] if i < len(left) else None
        r = right[i] if i < len(right) else None
        score = line_similarity(l.text, r.text) if l is not None and r is not None else 0.0
        rows.append(AlignedLine(l, r, score, _pair_tag(l, r, score)))
    return rows


def _align_changed_window(left: Sequence[CompareLine], right: Sequence[CompareLine]) -> list[AlignedLine]:
    """Global alignment for one changed window.

    The cost model favours pairing lines that share meaningful text, while a
    weak pair is more cheaply represented as a delete+insert gap.  Large windows
    use a linear fallback to avoid quadratic memory growth on whole books.
    """
    n, m = len(left), len(right)
    if not n or not m:
        return _zip_fallback(left, right)
    if n * m > 16_000:
        # 锚点分段（bertalign 两步法思路）：先找两侧唯一的高置信行当锚点，
        # 锚点之间的小窗口再各自跑 DP；找不到锚点才退回旧的线性降级。
        return _anchored_align(left, right)

    gap_cost = 0.62
    merge_penalty = 0.18   # 1-2/2-1 合并的额外代价：优于两个 gap，劣于干净的 1-1
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    move = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * gap_cost
        move[i][0] = "up"
    for j in range(1, m + 1):
        dp[0][j] = j * gap_cost
        move[0][j] = "left"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sim = line_similarity(left[i - 1].text, right[j - 1].text)
            # Very weak lines should normally become two gap rows instead of a
            # misleading visual pair.
            pair_cost = (1.0 - sim) if sim >= 0.22 else 1.32
            candidates = [
                (dp[i - 1][j - 1] + pair_cost, "diag"),
                (dp[i - 1][j] + gap_cost, "up"),
                (dp[i][j - 1] + gap_cost, "left"),
            ]
            # OCR 断行场景：左侧两行合并后对应右侧一行（2-1），或反之（1-2）。
            # 合并相似度显著时，这两种走法能把"各半行都低于 0.22 → 两个假 gap"
            # 的窗口正确配对。
            # 合并只对真实断行有意义：要求两段各自非空、拼接后有实质长度，
            # 避免超短行（如单字符测试行/序号行）因子串相似而误并。
            if i >= 2 and len(left[i - 2].text) >= 2 and len(left[i - 1].text) >= 2 \
                    and len(left[i - 2].text) + len(left[i - 1].text) >= 8:
                sim21 = line_similarity(
                    left[i - 2].text + left[i - 1].text, right[j - 1].text)
                if sim21 >= 0.5:
                    candidates.append((dp[i - 2][j - 1] + (1.0 - sim21) + merge_penalty, "m21"))
            if j >= 2 and len(right[j - 2].text) >= 2 and len(right[j - 1].text) >= 2 \
                    and len(right[j - 2].text) + len(right[j - 1].text) >= 8:
                sim12 = line_similarity(
                    left[i - 1].text, right[j - 2].text + right[j - 1].text)
                if sim12 >= 0.5:
                    candidates.append((dp[i - 1][j - 2] + (1.0 - sim12) + merge_penalty, "m12"))
            dp[i][j], move[i][j] = min(candidates, key=lambda item: item[0])

    rows: list[AlignedLine] = []
    i, j = n, m
    while i or j:
        direction = move[i][j]
        if direction == "diag":
            l, r = left[i - 1], right[j - 1]
            score = line_similarity(l.text, r.text)
            rows.append(AlignedLine(l, r, score, _pair_tag(l, r, score)))
            i -= 1
            j -= 1
        elif direction == "m21":
            # 左两行 ↔ 右一行：首行配对，第二行保留为相邻 gap 行（行式输出不变）
            l1, l2, r = left[i - 2], left[i - 1], right[j - 1]
            score = line_similarity(l1.text + l2.text, r.text)
            rows.append(AlignedLine(l2, None, 0.0, "delete"))
            rows.append(AlignedLine(l1, r, score, _pair_tag(l1, r, score)))
            i -= 2
            j -= 1
        elif direction == "m12":
            l, r1, r2 = left[i - 1], right[j - 2], right[j - 1]
            score = line_similarity(l.text, r1.text + r2.text)
            rows.append(AlignedLine(None, r2, 0.0, "insert"))
            rows.append(AlignedLine(l, r1, score, _pair_tag(l, r1, score)))
            i -= 1
            j -= 2
        elif direction == "up":
            l = left[i - 1]
            rows.append(AlignedLine(l, None, 0.0, "delete"))
            i -= 1
        else:
            r = right[j - 1]
            rows.append(AlignedLine(None, r, 0.0, "insert"))
            j -= 1
    rows.reverse()
    return rows


def _anchored_align(left: Sequence[CompareLine], right: Sequence[CompareLine]) -> list[AlignedLine]:
    """大窗口的锚点分段对齐。

    锚点 = 规范化后两侧各只出现一次、且互相相等的行；按左右序号同时递增
    （单调）筛选后，把窗口切成锚点之间的小段分别对齐。整本书级别的
    replace 窗口由 O(n·m) 降为近似 Σ(段内代价)。无锚点时退回旧降级。
    """
    left_keys = [normalise_for_alignment(item.text) for item in left]
    right_keys = [normalise_for_alignment(item.text) for item in right]

    def _unique_index(keys):
        seen: dict[str, int] = {}
        dup: set[str] = set()
        for idx, key in enumerate(keys):
            if not key or len(key) < 4:
                continue
            if key in seen:
                dup.add(key)
            else:
                seen[key] = idx
        return {k: v for k, v in seen.items() if k not in dup}

    lu = _unique_index(left_keys)
    ru = _unique_index(right_keys)
    shared = [(lu[k], ru[k]) for k in lu.keys() & ru.keys()]
    shared.sort()
    # 保持右侧单调递增（贪心即可：锚点本就稀疏且可信）
    anchors: list[tuple[int, int]] = []
    last_r = -1
    for li, ri in shared:
        if ri > last_r:
            anchors.append((li, ri))
            last_r = ri
    if not anchors:
        return _zip_fallback(left, right)

    rows: list[AlignedLine] = []
    prev_l, prev_r = 0, 0
    for li, ri in anchors:
        seg_l, seg_r = left[prev_l:li], right[prev_r:ri]
        if seg_l or seg_r:
            if len(seg_l) * len(seg_r) > 16_000:
                rows.extend(_zip_fallback(seg_l, seg_r))
            else:
                rows.extend(_align_changed_window(seg_l, seg_r))
        rows.append(AlignedLine(left[li], right[ri], 1.0, "equal"))
        prev_l, prev_r = li + 1, ri + 1
    seg_l, seg_r = left[prev_l:], right[prev_r:]
    if seg_l or seg_r:
        if len(seg_l) * len(seg_r) > 16_000:
            rows.extend(_zip_fallback(seg_l, seg_r))
        else:
            rows.extend(_align_changed_window(seg_l, seg_r))
    return rows


def align_lines(left: Sequence[CompareLine], right: Sequence[CompareLine]) -> list[AlignedLine]:
    # Defensive cleanup for legacy external documents that already contain one
    # empty block between every Markdown paragraph.  Re-alignment should never
    # render those separators as thousands of blank compare rows.
    left = _nonempty_compare_lines(left)
    right = _nonempty_compare_lines(right)
    left_keys = [normalise_for_alignment(item.text) for item in left]
    right_keys = [normalise_for_alignment(item.text) for item in right]
    matcher = SequenceMatcher(None, left_keys, right_keys, autojunk=False)
    rows: list[AlignedLine] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for l, r in zip(left[i1:i2], right[j1:j2]):
                rows.append(AlignedLine(l, r, 1.0, "equal"))
        elif tag == "delete":
            rows.extend(AlignedLine(item, None, 0.0, "delete") for item in left[i1:i2])
        elif tag == "insert":
            rows.extend(AlignedLine(None, item, 0.0, "insert") for item in right[j1:j2])
        else:
            rows.extend(_align_changed_window(left[i1:i2], right[j1:j2]))
    return rows



def collapse_trusted_right_gap_runs(
    rows: Sequence[AlignedLine], *, min_run: int = 4
) -> tuple[list[AlignedLine], int, int]:
    """Collapse long left-only OCR runs in trusted-right comparison view.

    A reliable full manuscript often removes many OCR fragments and duplicate
    blocks. Normal row alignment therefore renders dozens of empty right rows,
    which looks like missing imported text even though the trusted document has
    no blank blocks. This is only a display transform: trusted right-hand text is
    never removed, and the synthetic marker is ignored by authoritative layout.

    Returns ``(rows, collapsed_source_rows, collapsed_runs)``.
    """
    min_run = max(2, int(min_run or 0))
    output: list[AlignedLine] = []
    collapsed_rows = 0
    collapsed_runs = 0
    i = 0
    noncollapsible_types = {
        BlockType.IMAGE_REF.value, BlockType.CHAPTER.value,
        BlockType.SECTION.value, BlockType.TOC_ENTRY.value,
    }

    while i < len(rows):
        row = rows[i]
        if not (
            row.left is not None and row.right is None
            and row.left.block_type not in noncollapsible_types
            and str(row.left.text or "").strip()
        ):
            output.append(row)
            i += 1
            continue

        j = i
        run: list[AlignedLine] = []
        while j < len(rows):
            candidate = rows[j]
            if not (
                candidate.left is not None and candidate.right is None
                and candidate.left.block_type not in noncollapsible_types
                and str(candidate.left.text or "").strip()
            ):
                break
            run.append(candidate)
            j += 1

        if len(run) < min_run:
            output.extend(run)
            i = j
            continue

        first = str(run[0].left.text or "").strip().replace("\n", " ")
        last = str(run[-1].left.text or "").strip().replace("\n", " ")
        preview = first[:28]
        if len(run) > 1 and last != first:
            preview += " … " + last[:28]
        page_values = [int(item.left.page or 0) for item in run if item.left is not None]
        page = next((value for value in page_values if value > 0), 0)
        left_summary = CompareLine(
            text=f"⟦左侧 OCR 独有/重复 {len(run)} 段（已折叠）｜{preview}⟧",
            block_ids=[], block_indices=[], block_type="alignment_gap", page=page,
        )
        right_summary = CompareLine(
            text=f"{ALIGNMENT_GAP_PREFIX}左侧独有 {len(run)} 段｜不会写入可信正文⟧",
            block_ids=[], block_indices=[], block_type="alignment_gap", page=page,
        )
        output.append(AlignedLine(left_summary, right_summary, 1.0, "collapsed_delete"))
        collapsed_rows += len(run)
        collapsed_runs += 1
        i = j

    return output, collapsed_rows, collapsed_runs

def inherit_right_structure_from_alignment(rows: Sequence[AlignedLine]) -> list[AlignedLine]:
    """Project left/OCR structure onto a flat imported right-hand text.

    Plain TXT/Markdown imports have no block ids, page numbers or image anchors.
    When they are aligned against an OCR/Formatter document, this helper:

    * mirrors IMAGE_REF marker rows onto the right side;
    * reuses the aligned left block id/page for matching right text rows;
    * preserves chapter/section/dialogue structure where the left row is known;
    * leaves genuinely inserted right-only lines unreferenced so they become new
      blocks when applied.

    The visible right text remains unchanged except for inserted image marker rows.
    """
    projected: list[AlignedLine] = []
    structural_types = {
        BlockType.CHAPTER.value, BlockType.SECTION.value,
        BlockType.TOC_ENTRY.value, BlockType.DIALOGUE.value,
    }
    for row in rows:
        left = row.left
        right = row.right
        if left is not None and left.block_type == BlockType.IMAGE_REF.value:
            mirrored = CompareLine(
                text=left.text,
                block_ids=list(left.block_ids),
                block_indices=list(left.block_indices),
                block_type=BlockType.IMAGE_REF.value,
                page=int(left.page or 0),
            )
            projected.append(AlignedLine(left, mirrored, 1.0, "equal"))
            # A weak aligner must never sacrifice a real imported text line by
            # pairing it with an image marker.  Preserve such a line as a fresh
            # insertion immediately after the mirrored marker.
            if right is not None and parse_image_marker(right.text) is None and right.text.strip():
                projected.append(AlignedLine(None, CompareLine(
                    text=right.text, block_ids=[], block_indices=[],
                    block_type=right.block_type, page=int(right.page or 0),
                ), 0.0, "insert"))
            continue
        if right is not None:
            if left is not None:
                inherited_type = left.block_type if left.block_type in structural_types else right.block_type
                inherited = CompareLine(
                    text=right.text,
                    block_ids=list(left.block_ids),
                    block_indices=list(left.block_indices),
                    block_type=inherited_type or right.block_type,
                    page=int(left.page or right.page or 0),
                )
                projected.append(AlignedLine(left, inherited, row.similarity, row.tag))
            else:
                inserted = CompareLine(
                    text=right.text,
                    block_ids=[],
                    block_indices=[],
                    block_type=right.block_type,
                    page=int(right.page or 0),
                )
                projected.append(AlignedLine(left, inserted, row.similarity, row.tag))
            continue
        projected.append(row)
    return projected

@lru_cache(maxsize=32768)
def character_opcodes(left: str, right: str):
    return tuple(SequenceMatcher(None, left or "", right or "", autojunk=False).get_opcodes())


def document_revision_token(doc: UnifiedDocument | None) -> tuple:
    """Cheap identity token used by the GUI to avoid redundant whole-book work.

    It intentionally ignores Page Manager overlays: page assets are synchronized
    independently at EPUB handoff, so changing a cover classification must not
    trigger another text alignment/render pass.
    """
    if doc is None:
        return (None, 0, 0, "", "")
    blocks = doc.blocks or []
    first = blocks[0] if blocks else None
    last = blocks[-1] if blocks else None
    return (
        id(doc),
        len(blocks),
        len(getattr(doc, "processing_log", []) or []),
        str(getattr(first, "id", "") or ""),
        str(getattr(last, "id", "") or ""),
    )


def lacks_terminal_punctuation(text: str) -> bool:
    stripped = (text or "").rstrip()
    if not stripped or parse_image_marker(stripped):
        return False
    return not stripped.endswith(TERMINAL_PUNCTUATION)


def needs_sentence_end_review(text: str, block_type: str = "paragraph") -> bool:
    """Return True when a prose row probably ends before its sentence does.

    The review marker is deliberately stricter than ``lacks_terminal_punctuation``:
    commas and colons are not accepted as sentence endings.  Structural rows such
    as chapter titles, TOC entries, image markers and compact alignment gaps are
    excluded so the navigation list remains useful on long novels.
    """
    stripped = (text or "").strip()
    kind = str(getattr(block_type, "value", block_type) or "paragraph").lower()
    if not stripped or kind in NON_PROSE_REVIEW_TYPES:
        return False
    if parse_image_marker(stripped) or is_alignment_placeholder(stripped):
        return False
    # Some PDF/EPUB text layers expose illustration resource ids such as
    # ``＜ｉ２０５５１６｜１０４１７＞`` as a standalone pseudo-line.
    if re.fullmatch(r"[＜<][^＞>]{1,160}[＞>]", stripped):
        return False
    if looks_like_chapter_title(stripped):
        return False
    # OCR chapter/page counters are commonly full-width digits with no punctuation.
    if re.fullmatch(r"[\s　0-9０-９〇零一二三四五六七八九十百千万]+", stripped):
        return False
    return not stripped.endswith(SENTENCE_END_PUNCTUATION)


def sentence_end_review_rows(
    lines: Sequence[str], block_types: Sequence[str] | None = None,
) -> list[int]:
    """Return stable row numbers that need sentence-end review."""
    kinds = list(block_types or [])
    result: list[int] = []
    for index, text in enumerate(lines):
        kind = kinds[index] if index < len(kinds) else "paragraph"
        if needs_sentence_end_review(text, kind):
            result.append(index)
    return result


def sentence_end_navigation_target(
    rows: Sequence[int], current_row: int, direction: int, *, focused: bool = False,
) -> tuple[int, int] | None:
    """Resolve previous/next review row with deterministic wrap-around.

    On the first jump the issue under the current cursor is included. Once a
    review row is focused, subsequent jumps move strictly away from that row.
    The return value is ``(row, index_in_sorted_rows)``.
    """
    ordered = sorted({int(row) for row in rows if int(row) >= 0})
    if not ordered:
        return None
    current = max(0, int(current_row))
    if direction < 0:
        candidates = [row for row in ordered if row < current or (not focused and row == current)]
        target = candidates[-1] if candidates else ordered[-1]
    else:
        candidates = [row for row in ordered if row > current or (not focused and row == current)]
        target = candidates[0] if candidates else ordered[0]
    return target, ordered.index(target)


def split_dialogue_segments(text: str) -> list[str]:
    """Place complete Japanese corner-bracket dialogue on independent lines.

    Only balanced ``「...」`` spans are split.  Unbalanced source remains intact
    for manual review; quoted terms in ``『...』`` are not treated as dialogue.
    """
    if parse_image_marker(text):
        return [text]
    output: list[str] = []
    for physical_line in (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if not physical_line:
            output.append("")
            continue
        cursor = 0
        pieces: list[str] = []
        while cursor < len(physical_line):
            opening = physical_line.find("「", cursor)
            if opening < 0:
                tail = physical_line[cursor:]
                if tail:
                    pieces.append(tail)
                break
            closing = physical_line.find("」", opening + 1)
            if closing < 0:
                # Keep the unresolved remainder together; never invent a quote.
                pieces.append(physical_line[cursor:])
                break
            prefix = physical_line[cursor:opening]
            if prefix:
                pieces.append(prefix)
            pieces.append(physical_line[opening:closing + 1])
            cursor = closing + 1
        cleaned = [piece.strip("\n") for piece in pieces if piece != ""]
        output.extend(cleaned or [""])
    return output


def _coerce_record_type(value: str, text: str, fallback: BlockType) -> BlockType:
    stripped = (text or "").strip()
    if parse_image_marker(stripped):
        return BlockType.IMAGE_REF
    if stripped.startswith("「") and stripped.endswith("」"):
        return BlockType.DIALOGUE
    try:
        candidate = BlockType(str(value or ""))
    except Exception:
        candidate = fallback
    if candidate in {BlockType.CHAPTER, BlockType.SECTION, BlockType.TOC_ENTRY}:
        return candidate
    # A compare row can temporarily lose its UI-side type reference after manual
    # line operations.  Never demote an original chapter merely because that
    # transient reference says paragraph.  Obvious chapter titles are promoted
    # as a second safety net so TOC generation survives whole-text paste/edit.
    if fallback in {BlockType.CHAPTER, BlockType.SECTION, BlockType.TOC_ENTRY}:
        return fallback
    if looks_like_chapter_title(stripped):
        return BlockType.CHAPTER
    return BlockType.PARAGRAPH


def apply_compare_records(source: UnifiedDocument, records: Sequence[CompareLine]) -> tuple[UnifiedDocument, int]:
    """Rebuild the right-hand text sequence as the authoritative document.

    The previous partial-patch implementation left every source block that had
    lost its row reference in place.  A multi-line paste or realignment could
    therefore insert a new copy of a scene while the old scene survived at the
    end of the book.  This implementation writes exactly the non-empty right
    rows in their displayed order, reuses referenced blocks once, and carries
    movable IMAGE_REF markers as real image blocks.
    """
    result = copy.deepcopy(source)
    block_by_id = {block.id: block for block in result.blocks}
    source_text_ids = {
        block.id for block in result.blocks
        if block.type in TEXT_TYPES and block.type != BlockType.IMAGE_REF
        and not (block.metadata or {}).get("consumed")
    }
    toc_titles = {normalise_for_alignment(entry.title) for entry in (result.toc or []) if entry.title}
    used_ids: set[str] = set()
    emitted_text_ids: set[str] = set()
    emitted_image_ids: set[str] = set()
    emitted_texts_by_id: dict[str, str] = {}
    rebuilt: list[Block] = []
    previous_page = 0
    changed = 0
    duplicate_ref_rows = 0
    last_text_record_index = max(
        (index for index, item in enumerate(records) if item.text.strip() and not parse_image_marker(item.text)),
        default=-1,
    )

    for record_index, record in enumerate(records):
        raw_text = record.text or ""
        marker_id = parse_image_marker(raw_text)
        if marker_id:
            original = block_by_id.get(marker_id)
            if original is None or original.type != BlockType.IMAGE_REF:
                # A damaged/foreign marker must never leak into EPUB正文.
                changed += 1
                continue
            if marker_id in emitted_image_ids:
                duplicate_ref_rows += 1
                changed += 1
                continue
            image_block = copy.deepcopy(original)
            if record.page:
                image_block.page = int(record.page)
            image_block.metadata = dict(image_block.metadata or {})
            previous = rebuilt[-1] if rebuilt else None
            image_block.image_anchor = previous.id if previous is not None else "start"
            # A previously unplaced image becomes authoritative once its marker
            # is moved before later正文.  Leaving it at the very end keeps the
            # warning so the user knows placement is still unresolved.
            if record_index < last_text_record_index:
                image_block.metadata.pop("placement_required", None)
            image_block.metadata.setdefault("manual_compare_audit", []).append({
                "action": "image_marker_position", "marker": raw_text.strip(),
                "anchor": image_block.image_anchor,
            })
            rebuilt.append(image_block)
            emitted_image_ids.add(marker_id)
            used_ids.add(marker_id)
            previous_page = int(image_block.page or previous_page or 0)
            continue

        text = raw_text.strip("\r\n")
        if not text.strip():
            continue

        ids = [bid for bid in record.block_ids if bid in block_by_id and block_by_id[bid].type != BlockType.IMAGE_REF]
        primary_id = next((bid for bid in ids if bid not in used_ids), None)
        if primary_id is not None:
            original = block_by_id[primary_id]
            block = copy.deepcopy(original)
            consumed_ids = [bid for bid in ids if bid != primary_id]
            used_ids.update(ids)
            emitted_text_ids.add(primary_id)
            if consumed_ids:
                changed += len(consumed_ids)
            if block.text != text:
                before = block.text
                block.ocr_raw = block.ocr_raw or before
                block.text = text
                block.modified_by = (block.modified_by + ",manual_aligned_compare_edit").strip(",")
                block.metadata.setdefault("manual_compare_audit", []).append({"before": before, "after": text})
                changed += 1
            block.type = _coerce_record_type(record.block_type, text, original.type)
            if normalise_for_alignment(text) in toc_titles or looks_like_chapter_title(text):
                block.type = BlockType.CHAPTER
            if record.page:
                block.page = int(record.page)
            previous_page = int(block.page or previous_page or 0)
            emitted_texts_by_id[primary_id] = text
            rebuilt.append(block)
            continue

        # The same source id appearing a second time is normally an alignment
        # metadata duplication, not an intentional duplicated scene.  Skip an
        # identical repeat; genuinely new user text has no reusable id and is
        # emitted below as a new block.
        repeated_id = next((bid for bid in ids if bid in emitted_texts_by_id), None)
        if repeated_id is not None and emitted_texts_by_id[repeated_id] == text:
            duplicate_ref_rows += 1
            changed += 1
            continue

        fallback = BlockType.DIALOGUE if text.strip().startswith("「") and text.strip().endswith("」") else BlockType.PARAGRAPH
        block_type = _coerce_record_type(record.block_type, text, fallback)
        if normalise_for_alignment(text) in toc_titles or looks_like_chapter_title(text):
            block_type = BlockType.CHAPTER
        new_block = Block(
            type=block_type,
            text=text,
            page=(int(record.page or 0) or previous_page),
        )
        new_block.modified_by = "manual_aligned_compare_edit"
        new_block.metadata["inserted_from_text_compare"] = True
        rebuilt.append(new_block)
        previous_page = int(new_block.page or previous_page or 0)
        changed += 1

    # Every image marker should be visible in the editor.  For backward
    # compatibility with an older compare view that did not show markers, keep
    # untouched source images only when no image marker appeared at all.
    if not emitted_image_ids:
        # Older compare views did not expose image markers.  Preserve those
        # images at their original boundary instead of appending every image at
        # the end of the book.
        source_sequence = list(result.blocks)
        for source_index, source_block in enumerate(source_sequence):
            if source_block.type != BlockType.IMAGE_REF:
                continue
            image_block = copy.deepcopy(source_block)
            next_text_id = next((
                candidate.id for candidate in source_sequence[source_index + 1:]
                if candidate.type in TEXT_TYPES and candidate.type != BlockType.IMAGE_REF
                and candidate.id in emitted_text_ids
            ), None)
            if next_text_id is not None:
                insert_at = next((idx for idx, candidate in enumerate(rebuilt) if candidate.id == next_text_id), len(rebuilt))
            else:
                previous_text_id = next((
                    candidate.id for candidate in reversed(source_sequence[:source_index])
                    if candidate.type in TEXT_TYPES and candidate.type != BlockType.IMAGE_REF
                    and candidate.id in emitted_text_ids
                ), None)
                if previous_text_id is None:
                    insert_at = 0
                else:
                    previous_index = next((idx for idx, candidate in enumerate(rebuilt) if candidate.id == previous_text_id), len(rebuilt) - 1)
                    insert_at = previous_index + 1
                    while insert_at < len(rebuilt) and rebuilt[insert_at].type == BlockType.IMAGE_REF:
                        insert_at += 1
            rebuilt.insert(insert_at, image_block)

    result.blocks = rebuilt
    for index, block in enumerate(result.blocks):
        block.reading_order = index

    result.toc = []
    chapter_no = 0
    for index, block in enumerate(result.blocks):
        if block.type != BlockType.CHAPTER or not (block.text or "").strip():
            continue
        chapter_no += 1
        block.chapter_index = chapter_no
        result.toc.append(TocEntry(block.text.strip(), chapter_no, index))

    unrepresented = len(source_text_ids - emitted_text_ids)
    result.metadata.__dict__["manual_compare_unrepresented_source_blocks"] = unrepresented
    result.metadata.__dict__["manual_compare_duplicate_reference_rows"] = duplicate_ref_rows
    result.add_log(
        "manual_aligned_compare_edit",
        f"文本对比工作区重建正文 {len(result.blocks)} 块；修改 {changed} 处；跳过重复引用 {duplicate_ref_rows} 行",
        changed,
    )
    return result, changed

