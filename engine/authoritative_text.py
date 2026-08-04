# -*- coding: utf-8 -*-
"""Quality gates for producing an authoritative novel text.

The authoritative text is the only body text allowed to flow into strict OCR
replacement. Page assets may be inherited later, but OCR prose must never be
reintroduced after this gate.
"""
from __future__ import annotations

import copy
import difflib
import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, asdict

from models.document import Block, BlockType, TocEntry, UnifiedDocument

_TEXT_TYPES = {
    BlockType.PARAGRAPH, BlockType.DIALOGUE, BlockType.CHAPTER,
    BlockType.SECTION, BlockType.RUBY, BlockType.FOOTNOTE,
}
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\((?:data:image/[^)]*|[^)]*\.(?:png|jpe?g|gif|webp|bmp|tiff?)(?:\?[^)]*)?)\)", re.I | re.S)
_BASE64_RE = re.compile(r"data:image/[a-z0-9.+-]+;base64,", re.I)
_PROTOCOL_ONLY_RE = re.compile(r"^(?:[pcdsrftu]|b\d{1,6})$", re.I)
_OBVIOUS_OCR_FIXES = (
    ("であろうう", "であろう"),
    ("してている", "している"),
    ("こここ", "ここ"),
    ("いつつの間", "いつの間"),
    ("かかかる", "かかる"),
    ("まとともに", "まともに"),
    ("こういうう", "こういう"),
    ("そんないいい", "そんないい"),
    ("いたただ", "いただ"),
    ("かすかかな", "かすかな"),
    ("しつかり", "しっかり"),
    ("気を失つて", "気を失って"),
)

# 逐条 AI 结果落盘后再应用的高置信度 OCR 修复。放在 AI 后而不是 AI 前，
# 可以继续复用 v8/v9 的逐条断点签名；这里只收录无需上下文推理、不会改写
# 情节或句意的固定误识别。
_POST_INDEXED_OCR_FIXES = (
    ("賛沢", "贅沢"),
    ("冒渉する", "冒涜する"),
    ("閤を払い", "闇を払い"),
    ("モンスターグごとき", "モンスターごとき"),
    ("モンスターグとき", "モンスターごとき"),
    ("みつともない", "みっともない"),
    ("あっとという間", "あっという間"),
    ("やんなざい", "やんなさい"),
    ("こちらののど笛", "こちらの喉笛"),
    ("なぜかかりータ", "なぜかリータ"),
    ("形をとったものなのだ。さげすもちろん", "形をとったものなのだ。もちろん"),
    ("漂ってくる。ほんりゅうあまた", "漂ってくる。"),
)


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"[\s　]+", "", text)


def _body_blocks(doc: UnifiedDocument):
    return [b for b in doc.blocks if b.type in _TEXT_TYPES and str(b.text or "").strip()]


def _body_text(doc: UnifiedDocument) -> str:
    return "".join(_norm(b.text) for b in _body_blocks(doc))


def _quote_imbalance(text: str) -> int:
    return abs(text.count("「") - text.count("」")) + abs(text.count("『") - text.count("』"))


def _duplicate_runs(doc: UnifiedDocument, min_blocks: int = 4, min_chars: int = 160) -> list[tuple[int, int, int]]:
    """Return repeated adjacent block runs as (start, repeat_start, length).

    Uses fixed-size n-gram fingerprints to find candidates, then extends matches.
    This remains fast for full novels with thousands of blocks.
    """
    blocks = _body_blocks(doc)
    keys = [_norm(b.text) for b in blocks]
    n = len(keys)
    if n < min_blocks * 2:
        return []
    grams: dict[tuple[str, ...], list[int]] = {}
    for i in range(0, n - min_blocks + 1):
        gram = tuple(keys[i:i + min_blocks])
        grams.setdefault(gram, []).append(i)
    found: list[tuple[int, int, int]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for positions in grams.values():
        if len(positions) < 2:
            continue
        for left_index in range(len(positions) - 1):
            a = positions[left_index]
            for b in positions[left_index + 1:]:
                if b - a < min_blocks or (a, b) in seen_pairs:
                    continue
                seen_pairs.add((a, b))
                length = min_blocks
                while b + length < n and a + length < b and keys[a + length] == keys[b + length]:
                    length += 1
                if sum(len(x) for x in keys[a:a + length]) >= min_chars:
                    found.append((a, b, length))
                # For adjacent overlap-batch duplication, the nearest repeat is sufficient.
                break
    compact: list[tuple[int, int, int]] = []
    covered_until = -1
    for item in sorted(found, key=lambda x: (x[0], x[1], -x[2])):
        if item[0] < covered_until:
            continue
        compact.append(item)
        covered_until = item[1] + item[2]
    return compact


def _suspicious_fragments(doc: UnifiedDocument) -> list[str]:
    warnings: list[str] = []
    for block in _body_blocks(doc):
        text = str(block.text or "")
        if _BASE64_RE.search(text) or _MD_IMAGE_RE.search(text):
            warnings.append("markdown_image_or_base64")
        if _PROTOCOL_ONLY_RE.fullmatch(text.strip()):
            warnings.append("protocol_alias_in_body")
        if len(_norm(text)) >= 4 and re.search(r"(?:であろうう|してている|こここ|いつつの間|かかかる|まとともに|こういうう|そんないいい|いたただ|かすかかな|しつかり|気を失つて)", text):
            warnings.append("obvious_repeated_kana")
        if block.type == BlockType.PARAGRAPH and text.lstrip().startswith("」"):
            warnings.append("orphan_closing_quote")
        if len(_norm(text)) > 1200:
            warnings.append("overlong_block")
    return warnings


def _source_ids(block: Block, source_by_id: dict[str, Block]) -> tuple[str, ...]:
    ids = tuple(str(x) for x in ((block.metadata or {}).get("source_block_ids") or []) if str(x) in source_by_id)
    if ids:
        return ids
    return (block.id,) if block.id in source_by_id else ()


def _coverage(left: str, right: str) -> float:
    if not left:
        return 1.0
    matcher = difflib.SequenceMatcher(None, left, right, autojunk=False)
    return sum(m.size for m in matcher.get_matching_blocks() if m.size) / max(len(left), 1)


def _group_is_safe(source_parts: list[str], output_parts: list[str], *, phase: str) -> tuple[bool, str]:
    if not source_parts or not output_parts or any(not str(x or "").strip() for x in output_parts):
        return False, "empty"
    if any(_PROTOCOL_ONLY_RE.fullmatch(str(x or "").strip()) for x in output_parts):
        return False, "protocol_leak"
    source = "".join(_norm(x) for x in source_parts)
    output = "".join(_norm(x) for x in output_parts)
    if not source or not output:
        return False, "empty"
    if phase == "correction" and (len(source_parts) != 1 or len(output_parts) != 1):
        return False, "correction_structure"

    ratio = len(output) / max(len(source), 1)
    n = len(source)
    if n >= 160:
        bounds, minimum = (0.76, 1.35), 0.58
    elif n >= 40:
        bounds, minimum = (0.68, 1.45), 0.52
    elif n >= 12:
        bounds, minimum = (0.45, 1.90), 0.38
    elif n >= 4:
        bounds, minimum = (0.28, 2.50), 0.22
    else:
        bounds, minimum = (0.20, 3.00), 0.0
    if not (bounds[0] <= ratio <= bounds[1]):
        return False, "length"
    if minimum and _coverage(source, output) < minimum:
        return False, "low_coverage"

    for part in source_parts:
        compact = _norm(part)
        if len(compact) >= 80:
            member_min = 0.42
        elif len(compact) >= 40:
            member_min = 0.38
        elif len(compact) >= 12:
            member_min = 0.30
        elif len(compact) >= 4:
            member_min = 0.18
        else:
            member_min = 0.0
        if member_min and _coverage(compact, output) < member_min:
            return False, "missing_member"

    # A split operation may return several blocks, but exact duplicate outputs are
    # never a valid way to represent a non-repeated source range.
    compact_outputs = [_norm(x) for x in output_parts if _norm(x)]
    if len(compact_outputs) != len(set(compact_outputs)) and len(_norm("".join(source_parts))) >= 12:
        doubled = "".join(compact_outputs)
        if _coverage(source, doubled) < 0.86:
            return False, "duplicate_output"
    return True, ""


def apply_conservative_ocr_fixes(doc: UnifiedDocument) -> int:
    """Apply only unambiguous repeated-kana/old-size OCR corrections."""
    changed = 0
    for block in doc.blocks:
        if block.type not in _TEXT_TYPES or not str(block.text or ""):
            continue
        original = block.text
        text = original
        for wrong, right in _OBVIOUS_OCR_FIXES:
            text = text.replace(wrong, right)
        if text != original:
            block.ocr_raw = block.ocr_raw or original
            block.text = text
            block.modified_by = "authoritative_obvious_ocr_fix"
            changed += 1
    if changed:
        doc.add_log("authoritative_obvious_ocr_fix", f"确定性修复 {changed} 个明显 OCR 重复字", changed)
    return changed


def apply_post_indexed_ocr_fixes(doc: UnifiedDocument) -> int:
    """Apply fixed, context-free OCR repairs after indexed AI proofreading.

    These replacements intentionally run after checkpoint restoration so improving the
    local repair table never invalidates already-paid AI row caches. Every changed block
    keeps ``ocr_raw`` and source lineage for review.
    """
    changed = 0
    for block in doc.blocks:
        if block.type not in _TEXT_TYPES or not str(block.text or ""):
            continue
        original = str(block.text or "")
        text = original
        for wrong, right in _POST_INDEXED_OCR_FIXES:
            text = text.replace(wrong, right)
        if text == original:
            continue
        block.ocr_raw = block.ocr_raw or original
        block.text = text
        previous = str(block.modified_by or "")
        block.modified_by = ",".join(x for x in (previous, "post_indexed_ocr_fix") if x)
        changed += 1
    if changed:
        doc.add_log("post_indexed_ocr_fix", f"追加修复 {changed} 个高置信度 OCR 错字块", changed)
    return changed


def promote_authoritative_chapters(doc: UnifiedDocument) -> int:
    """Rebuild chapter types/TOC for page-less external Markdown documents."""
    from engine.formatter import CHAPTER_RE

    chapter_index = 0
    changed = 0
    doc.toc = []
    current = 0
    for index, block in enumerate(doc.blocks):
        if block.type in _TEXT_TYPES:
            text = str(block.text or "").strip()
            normalized = re.sub(r"[\s　]+", "", text)
            is_heading = block.type == BlockType.CHAPTER or (
                block.type == BlockType.PARAGRAPH and len(normalized) <= 120 and bool(CHAPTER_RE.match(normalized) or CHAPTER_RE.match(text))
            )
            if is_heading:
                if block.type != BlockType.CHAPTER:
                    block.type = BlockType.CHAPTER
                    block.modified_by = "authoritative_chapter_detection"
                    changed += 1
                chapter_index += 1
                current = chapter_index
                block.chapter_index = current
                doc.toc.append(TocEntry(title=text, chapter_index=current, block_index=index))
                continue
        if current and block.chapter_index != current:
            block.chapter_index = current
    if chapter_index:
        doc.add_log("authoritative_chapter_detection", f"重建 {chapter_index} 个章节及目录", changed)
    return changed


def reconcile_authoritative_result(source: UnifiedDocument, result: UnifiedDocument, *, phase: str) -> UnifiedDocument:
    """Rebuild AI output from verified source-lineage groups.

    Every source block is emitted exactly once. Unsafe short replacements, compact
    protocol aliases, overlapping source ranges and omitted blocks are restored from
    the source document rather than being allowed to corrupt the authoritative text.
    """
    output = copy.deepcopy(result)
    source_by_id = {b.id: b for b in source.blocks if b.type in _TEXT_TYPES and str(b.text or "").strip()}
    source_rank = {b.id: i for i, b in enumerate(source.blocks)}

    groups: list[tuple[tuple[str, ...], list[Block]]] = []
    for block in output.blocks:
        if block.type not in _TEXT_TYPES or not str(block.text or "").strip():
            continue
        ids = _source_ids(block, source_by_id)
        if not ids:
            continue
        if groups and groups[-1][0] == ids:
            groups[-1][1].append(block)
        else:
            groups.append((ids, [block]))

    entries: list[tuple[int, list[Block]]] = []
    occupied: set[str] = set()
    recovered = 0
    protocol_recovered = 0
    overlap_recovered = 0
    adjacency_recovered = 0
    for ids, blocks in groups:
        ranks = [source_rank[x] for x in ids if x in source_rank]
        valid_contiguous = bool(ranks) and sorted(ranks) == list(range(min(ranks), max(ranks) + 1))
        conflict = bool(occupied.intersection(ids))
        source_parts = [source_by_id[x].text for x in ids if x in source_by_id]
        safe, reason = _group_is_safe(source_parts, [b.text for b in blocks], phase=phase)
        if not valid_contiguous:
            safe, reason = False, "noncontiguous"
        if conflict:
            safe, reason = False, "overlap"

        if safe:
            kept = [copy.deepcopy(b) for b in blocks]
            entries.append((min(ranks), kept))
            occupied.update(ids)
            continue

        if conflict:
            # A later overlapping operation is discarded. The already accepted range
            # remains authoritative; uncovered ids are restored in the final pass.
            overlap_recovered += 1
            continue
        fallback = []
        for sid in ids:
            if sid not in source_by_id or sid in occupied:
                continue
            b = copy.deepcopy(source_by_id[sid])
            b.metadata = {**(b.metadata or {}), "source_block_ids": [sid], "ai_recovery_reason": reason}
            b.modified_by = "authoritative_source_recovery"
            fallback.append(b)
            occupied.add(sid)
        if fallback:
            entries.append((min(source_rank[b.id] for b in fallback), fallback))
            recovered += len(fallback)
            if reason == "protocol_leak":
                protocol_recovered += len(fallback)

    for sid, block in source_by_id.items():
        if sid in occupied:
            continue
        b = copy.deepcopy(block)
        b.metadata = {**(b.metadata or {}), "source_block_ids": [sid], "ai_recovery_reason": "omitted"}
        b.modified_by = "authoritative_source_recovery"
        entries.append((source_rank[sid], [b]))
        occupied.add(sid)
        recovered += 1

    entries.sort(key=lambda x: x[0])
    body: list[Block] = []
    for _, blocks in entries:
        body.extend(blocks)

    # In the correction phase, block positions are locked.  If AI makes an output
    # block absorb or duplicate its neighbour, restore both aligned source blocks.
    # This is deliberately conservative: losing two local corrections is preferable
    # to aborting the full paid run or accepting duplicated prose.
    if phase == "correction" and len(body) == len(source_by_id):
        source_order = sorted(source_by_id.values(), key=lambda b: source_rank[b.id])
        for index in range(len(body) - 1):
            after_issue = _adjacent_pair_issue(body[index], body[index + 1])
            before_issue = _adjacent_pair_issue(source_order[index], source_order[index + 1])
            if not after_issue or before_issue:
                continue
            for offset in (0, 1):
                src = source_order[index + offset]
                restored = copy.deepcopy(src)
                restored.metadata = {
                    **(restored.metadata or {}),
                    "source_block_ids": [src.id],
                    "ai_recovery_reason": "introduced_adjacent_duplicate",
                }
                restored.modified_by = "authoritative_source_recovery"
                body[index + offset] = restored
            recovered += 2
            adjacency_recovered += 1

    # Exact adjacent duplicates caused by two different source ranges are resolved by
    # reverting the later range to its own source. Intentional source repetition stays.
    for index in range(1, len(body)):
        left, right = body[index - 1], body[index]
        if len(_norm(right.text)) < 12 or _norm(left.text) != _norm(right.text):
            continue
        left_ids = _source_ids(left, source_by_id)
        right_ids = _source_ids(right, source_by_id)
        left_src = "".join(source_by_id[x].text for x in left_ids if x in source_by_id)
        right_src = "".join(source_by_id[x].text for x in right_ids if x in source_by_id)
        if _norm(left_src) == _norm(right_src):
            continue
        restored = []
        for sid in right_ids:
            if sid in source_by_id:
                b = copy.deepcopy(source_by_id[sid])
                b.metadata = {**(b.metadata or {}), "source_block_ids": [sid], "ai_recovery_reason": "adjacent_duplicate"}
                b.modified_by = "authoritative_source_recovery"
                restored.append(b)
        if restored:
            body[index:index + 1] = restored
            recovered += len(restored)

    # Reinsert non-text assets using their original source rank.
    assets = [copy.deepcopy(b) for b in source.blocks if b.type not in _TEXT_TYPES]
    ordered = list(body)
    for asset in assets:
        rank = source_rank.get(asset.id, len(source.blocks))
        insert_at = len(ordered)
        for i, block in enumerate(ordered):
            ids = _source_ids(block, source_by_id)
            block_rank = min((source_rank[x] for x in ids if x in source_rank), default=len(source.blocks))
            if block_rank > rank:
                insert_at = i
                break
        ordered.insert(insert_at, asset)
    output.blocks = [b for b in ordered if b.type not in _TEXT_TYPES or str(b.text or "").strip()]
    # The correction phase is a strict one-source-block -> one-output-block
    # transaction.  Never run sentence merging here: even a character-preserving
    # merge changes the block count and makes the correction gate report false
    # data loss/duplication.  High-confidence continuation merging belongs only to
    # the later layout phase.
    layout_adjacency_recovered = 0
    if phase == "layout":
        from engine.formatter import merge_broken_sentences
        output = merge_broken_sentences(output)
    apply_conservative_ocr_fixes(output)
    promote_authoritative_chapters(output)
    if phase == "layout":
        output, layout_adjacency_recovered = _recover_introduced_layout_adjacency(source, output)
        # Restoring source blocks may restore paragraph types; rebuild chapter/TOC
        # metadata once more without changing body text.
        promote_authoritative_chapters(output)
    output.add_log(
        "authoritative_lineage_reconcile",
        f"逐来源块重建：恢复 {recovered}，协议占位恢复 {protocol_recovered}，重叠操作拒绝 {overlap_recovered}，相邻污染恢复 {adjacency_recovered}，排版邻接回退 {layout_adjacency_recovered}",
        recovered + overlap_recovered + layout_adjacency_recovered,
    )
    return output


def _adjacent_pair_issue(left: Block, right: Block) -> str | None:
    a, b = _norm(left.text), _norm(right.text)
    if not a or not b:
        return None
    if len(b) >= 8 and a == b:
        return "adjacent_exact_duplicate"
    if 8 <= len(b) < len(a) and (a.endswith(b) or b in a[-max(80, len(b) + 20):]):
        return "adjacent_covered_tail"
    return None


def _adjacent_duplicate_issues(doc: UnifiedDocument) -> list[str]:
    blocks = _body_blocks(doc)
    return [issue for left, right in zip(blocks, blocks[1:]) if (issue := _adjacent_pair_issue(left, right))]


def _recover_introduced_layout_adjacency(
    source: UnifiedDocument, output: UnifiedDocument
) -> tuple[UnifiedDocument, int]:
    """Locally roll back layout operations that introduce adjacent duplication.

    The typesetting model is allowed to merge/split contiguous source ranges.  A
    malformed patch can nevertheless make the left output absorb the right range
    while the right output is also kept.  Rejecting the entire paid full-book run is
    wasteful; restore only the source ranges participating in a newly introduced
    exact/tail duplicate and preserve all other accepted layout work.
    """
    current = copy.deepcopy(output)
    source_by_id = {
        b.id: b for b in source.blocks
        if b.type in _TEXT_TYPES and str(b.text or "").strip()
    }
    source_rank = {b.id: i for i, b in enumerate(source.blocks)}
    recovered_boundaries = 0

    for _ in range(8):
        body = [b for b in current.blocks if b.type in _TEXT_TYPES and str(b.text or "").strip()]
        restore_ids: set[str] = set()
        for left, right in zip(body, body[1:]):
            issue = _adjacent_pair_issue(left, right)
            if not issue:
                continue
            left_ids = _source_ids(left, source_by_id)
            right_ids = _source_ids(right, source_by_id)
            if not left_ids or not right_ids:
                continue
            left_ranked = [sid for sid in left_ids if sid in source_rank]
            right_ranked = [sid for sid in right_ids if sid in source_rank]
            if not left_ranked or not right_ranked:
                continue
            src_left_id = max(left_ranked, key=lambda sid: source_rank[sid])
            src_right_id = min(right_ranked, key=lambda sid: source_rank[sid])
            # Preserve intentional repetition already present at the corresponding
            # source boundary.  Only AI/layout-introduced adjacency is rolled back.
            if _adjacent_pair_issue(source_by_id[src_left_id], source_by_id[src_right_id]):
                continue
            restore_ids.update(left_ids)
            restore_ids.update(right_ids)
            recovered_boundaries += 1

        if not restore_ids:
            break

        # If a merged/split output block touches one bad source id, restore its whole
        # lineage range.  Partial retention would duplicate the untouched members.
        changed = True
        while changed:
            changed = False
            for block in current.blocks:
                if block.type not in _TEXT_TYPES:
                    continue
                ids = set(_source_ids(block, source_by_id))
                if ids and ids.intersection(restore_ids) and not ids.issubset(restore_ids):
                    restore_ids.update(ids)
                    changed = True

        entries: list[tuple[int, int, int, Block]] = []
        for original_index, block in enumerate(current.blocks):
            if block.type in _TEXT_TYPES:
                ids = _source_ids(block, source_by_id)
                if ids and set(ids).intersection(restore_ids):
                    continue
                rank = min((source_rank[sid] for sid in ids if sid in source_rank), default=len(source.blocks) + original_index)
            else:
                rank = source_rank.get(block.id, len(source.blocks) + original_index)
            entries.append((rank, 1, original_index, copy.deepcopy(block)))

        for sid in sorted(restore_ids, key=lambda item: source_rank.get(item, 10**9)):
            if sid not in source_by_id:
                continue
            block = copy.deepcopy(source_by_id[sid])
            block.metadata = {
                **(block.metadata or {}),
                "source_block_ids": [sid],
                "ai_recovery_reason": "introduced_layout_adjacency",
            }
            block.modified_by = "authoritative_source_recovery"
            entries.append((source_rank[sid], 0, source_rank[sid], block))

        entries.sort(key=lambda item: (item[0], item[1], item[2]))
        current.blocks = [item[3] for item in entries]

    if recovered_boundaries:
        current.add_log(
            "authoritative_layout_adjacency_recovery",
            f"局部回退 {recovered_boundaries} 处 AI 新增相邻重复/覆盖",
            recovered_boundaries,
        )
    return current, recovered_boundaries


def _introduced_correction_adjacent_issues(source: UnifiedDocument, result: UnifiedDocument) -> list[str]:
    """Return only adjacency issues introduced by correction.

    Correction is position locked after lineage reconciliation.  Comparing the
    output's raw issue count with the source count is too coarse: fixing text can
    shift a tail-match heuristic even when no duplicate was created.  Compare each
    aligned pair instead and flag it only when the corresponding source pair was
    clean.
    """
    source_blocks = _body_blocks(source)
    result_blocks = _body_blocks(result)
    if len(source_blocks) != len(result_blocks):
        return ["unaligned_block_count"]
    introduced: list[str] = []
    for index in range(len(result_blocks) - 1):
        after = _adjacent_pair_issue(result_blocks[index], result_blocks[index + 1])
        if not after:
            continue
        before = _adjacent_pair_issue(source_blocks[index], source_blocks[index + 1])
        if not before:
            introduced.append(after)
    return introduced



@dataclass
class AuthorityReport:
    passed: bool
    publish_ready: bool
    source_chars: int
    output_chars: int
    similarity: float
    source_blocks: int
    output_blocks: int
    quote_imbalance_before: int
    quote_imbalance_after: int
    duplicate_runs_before: int
    duplicate_runs_after: int
    suspicious_after: int
    reasons: list[str]
    warnings: list[str]
    source_hash: str
    output_hash: str

    def to_dict(self) -> dict:
        return asdict(self)


def validate_authoritative_transition(source: UnifiedDocument, result: UnifiedDocument, *, phase: str) -> AuthorityReport:
    source_text = _body_text(source)
    output_text = _body_text(result)
    if source_text == output_text:
        similarity = 1.0
    else:
        try:
            from rapidfuzz.fuzz import ratio as _rapid_ratio
            similarity = _rapid_ratio(source_text, output_text, score_cutoff=0) / 100.0
        except Exception:
            # SequenceMatcher is retained as a dependency-free fallback for smaller files.
            if max(len(source_text), len(output_text)) <= 50_000:
                similarity = difflib.SequenceMatcher(None, source_text, output_text, autojunk=False).ratio()
            else:
                sample = 20_000
                similarity = difflib.SequenceMatcher(
                    None, source_text[:sample] + source_text[-sample:],
                    output_text[:sample] + output_text[-sample:], autojunk=False,
                ).ratio()
    before_dup = len(_duplicate_runs(source))
    after_dup = len(_duplicate_runs(result))
    before_quote = _quote_imbalance(source_text)
    after_quote = _quote_imbalance(output_text)
    suspicious_items = _suspicious_fragments(result)
    adjacent_before = len(_adjacent_duplicate_issues(source))
    adjacent_after = len(_adjacent_duplicate_issues(result))
    introduced_correction_adjacent = (
        _introduced_correction_adjacent_issues(source, result) if phase == "correction" else []
    )
    empty_after = sum(1 for b in result.blocks if b.type in _TEXT_TYPES and not str(b.text or "").strip())
    protocol_after = suspicious_items.count("protocol_alias_in_body")
    suspicious = len(suspicious_items) + adjacent_after + empty_after
    reasons: list[str] = []
    warnings: list[str] = []

    if not output_text:
        reasons.append("正文为空")
    if source_text:
        ratio = len(output_text) / max(1, len(source_text))
        # Correction must be almost lossless; layout may move punctuation and merge blocks.
        min_ratio = 0.985 if phase == "correction" else 0.975
        max_ratio = 1.015 if phase == "correction" else 1.025
        if not (min_ratio <= ratio <= max_ratio):
            reasons.append(f"字符数量异常：{ratio:.3%}")
        min_similarity = 0.965 if phase == "correction" else 0.94
        if similarity < min_similarity:
            reasons.append(f"正文相似度过低：{similarity:.3%}")
    if after_dup > before_dup:
        reasons.append(f"新增连续重复：{before_dup}→{after_dup}")
    elif phase == "layout" and after_dup:
        # Existing residual duplicates are a review item, not a reason to throw away
        # every successfully completed and billed AI batch.
        warnings.append(f"仍有 {after_dup} 组连续重复")
    if after_quote > before_quote + 2:
        reasons.append(f"引号失衡恶化：{before_quote}→{after_quote}")
    elif phase == "layout" and after_quote > 2:
        # The model may correctly preserve an already-unbalanced OCR source. Keep the
        # draft and surface the exact count for targeted repair.
        warnings.append(f"全书引号仍不平衡：差值 {after_quote}")
    if phase == "correction" and len(_body_blocks(result)) != len(_body_blocks(source)):
        reasons.append("纯纠错阶段改变了正文块数量")
    if protocol_after:
        reasons.append(f"正文混入 {protocol_after} 个 AI 协议占位符")
    if empty_after:
        reasons.append(f"正文仍有 {empty_after} 个空块")
    if phase == "correction":
        if introduced_correction_adjacent:
            reasons.append(f"纯纠错阶段新增相邻重复/覆盖：{len(introduced_correction_adjacent)}")
    elif adjacent_after > adjacent_before:
        # Reconciliation already performs a source-lineage local rollback.  Any
        # residual ambiguous case is saved as a non-publish-ready draft rather than
        # discarding every completed AI batch and forcing the user to spend tokens
        # again.  Protocol leaks, empty blocks, gross loss and low similarity remain
        # hard failures above.
        warnings.append(f"排版后仍新增 {adjacent_after - adjacent_before} 处相邻重复/覆盖，已标记待复核")
    elif adjacent_after:
        warnings.append(f"仍有 {adjacent_after} 处相邻重复或覆盖")
    if suspicious:
        warnings.append(f"仍有 {suspicious} 项可疑文本")
    return AuthorityReport(
        passed=not reasons,
        publish_ready=(not reasons and not warnings),
        source_chars=len(source_text), output_chars=len(output_text), similarity=similarity,
        source_blocks=len(_body_blocks(source)), output_blocks=len(_body_blocks(result)),
        quote_imbalance_before=before_quote, quote_imbalance_after=after_quote,
        duplicate_runs_before=before_dup, duplicate_runs_after=after_dup,
        suspicious_after=suspicious, reasons=reasons, warnings=warnings,
        source_hash=hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        output_hash=hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
    )


def sanitize_authoritative_source(doc: UnifiedDocument) -> UnifiedDocument:
    """Remove non-prose payloads before AI. Never alter valid prose."""
    result = copy.deepcopy(doc)
    cleaned = []
    removed = 0
    for block in result.blocks:
        text = str(block.text or "")
        if block.type in _TEXT_TYPES and (_BASE64_RE.search(text) or _MD_IMAGE_RE.fullmatch(text.strip())):
            removed += 1
            continue
        cleaned.append(block)
    result.blocks = cleaned
    result.add_log("authoritative_sanitize", "过滤 Markdown 图片/Base64，建立权威正文输入", removed)
    return result


def mark_authoritative(doc: UnifiedDocument, report: AuthorityReport) -> UnifiedDocument:
    result = copy.deepcopy(doc)
    result.metadata.ai_processing_mode = "authoritative_text"
    result.metadata.ai_layout_locked = True
    # Metadata is intentionally extensible and serializes all fields in __dict__.
    result.metadata.authoritative_text = True
    result.metadata.authoritative_publish_ready = bool(report.publish_ready)
    result.metadata.authoritative_draft = not bool(report.publish_ready)
    result.metadata.authoritative_report = report.to_dict()
    if report.publish_ready:
        message = "权威正文已通过完整性与发布校验"
    else:
        message = "权威正文完整性通过，已保存为待复核草稿：" + "；".join(report.warnings)
    result.add_log("authoritative_text", message, len(report.warnings))
    return result
