#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Risk scoring for OCR + manual Japanese correction.

The detector is deliberately conservative: it never changes OCR text.  It only
annotates fixed-region physical columns so the review UI can send the user to
likely errors first.  The original OCR remains the authoritative baseline until
an explicit manual edit is applied.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import re
import unicodedata

from models.document import Block, UnifiedDocument

_JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff々〆ヶ]")
_UNKNOWN = set("□■◻◼�")
_NOISE = set("|¦\\`^_")
_IGNORED_FOR_COUNT = set(" \t\r\n　")
_PAIRS = (("「", "」"), ("『", "』"), ("（", "）"), ("(", ")"), ("【", "】"), ("［", "］"), ("〈", "〉"), ("《", "》"))


@dataclass(slots=True)
class ReviewRisk:
    score: int = 0
    reasons: list[str] = field(default_factory=list)
    indices: set[int] = field(default_factory=set)

    def add(self, points: int, reason: str, indices=()):
        self.score = min(100, self.score + max(0, int(points)))
        if reason and reason not in self.reasons:
            self.reasons.append(reason)
        for index in indices:
            try:
                self.indices.add(max(0, int(index)))
            except (TypeError, ValueError):
                pass


def _reviewable(block: Block) -> bool:
    meta = block.metadata or {}
    return meta.get("layout_group") == "fixed_region_column" and bool(meta.get("column_id"))


def _logical_chars(text: str) -> list[str]:
    return [ch for ch in str(text or "") if ch not in _IGNORED_FOR_COUNT]


def _ink_comparable_chars(text: str) -> list[str]:
    """Characters comparable to the geometry-only black-ink glyph estimate.

    Japanese punctuation/ellipses are frequently represented by small or
    disconnected components, so counting them as full glyphs systematically
    makes OCR look longer than the image estimate.  Keep letters, numbers,
    symbols and marks, but ignore punctuation/separators/control characters.
    """
    output: list[str] = []
    for ch in str(text or ""):
        if ch in _IGNORED_FOR_COUNT:
            continue
        category = unicodedata.category(ch)
        if category.startswith(("P", "Z", "C")):
            continue
        output.append(ch)
    return output


def _adjacent_repeat(text: str) -> tuple[str, list[int]] | None:
    """Find an adjacent repeated chunk such as 嫉妬嫉妬 or 込み込み.

    Single-character doubles are intentionally ignored because they are common
    in Japanese.  The review UI should flag suspicious duplication, not rewrite
    language automatically.
    """
    chars = list(str(text or ""))
    for width in range(min(8, len(chars) // 2), 2, -1):
        for start in range(0, len(chars) - width * 2 + 1):
            left = chars[start:start + width]
            right = chars[start + width:start + width * 2]
            if (
                left == right
                and any(_JAPANESE_RE.search(ch) for ch in left)
                and any(unicodedata.category(ch).startswith(("L", "N")) for ch in left)
            ):
                return "".join(left), list(range(start, start + width * 2))
    return None




def _candidate_diff_indices(reference: str, candidate: str) -> set[int]:
    """Return reference-side character positions that differ from a candidate.

    Insertions are attached to the nearest visible character so the review UI
    can still jump to a concrete frame.  This is intentionally symmetric in
    spirit: it never decides which OCR route is correct.
    """
    left = str(reference or "")
    right = str(candidate or "")
    changed: set[int] = set()
    if left == right:
        return changed
    matcher = SequenceMatcher(a=left, b=right, autojunk=False)
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if i1 < i2:
            changed.update(range(i1, i2))
        elif left:
            changed.add(min(len(left) - 1, max(0, i1)))
        else:
            changed.add(0)
    return changed


def _candidate_conflicts(text: str, candidates) -> tuple[set[int], list[str]]:
    """Find credible character-level disagreements among OCR routes.

    A high-confidence, close-scoring alternative is enough to request review,
    even when the column-level selector did not mark the whole column as a
    conflict.  This catches visually similar substitutions such as 負/員 while
    remaining neutral about which route is right.
    """
    if not isinstance(candidates, list) or not candidates:
        return set(), []
    parsed: list[tuple[str, str, float, float]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        value = str(item.get("text", "") or "")
        if not value or value == text:
            continue
        try:
            confidence = float(item.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        try:
            score = float(item.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        parsed.append((str(item.get("method", "候选") or "候选"), value, confidence, score))
    if not parsed:
        return set(), []
    all_scores = [score for *_rest, score in parsed]
    for item in candidates:
        if isinstance(item, dict):
            try:
                all_scores.append(float(item.get("score", 0.0) or 0.0))
            except (TypeError, ValueError):
                pass
    best_score = max(all_scores or [0.0])
    tolerance = max(1.25, abs(best_score) * 0.08)
    indices: set[int] = set()
    previews: list[str] = []
    for method, value, confidence, score in parsed:
        ratio = SequenceMatcher(a=text, b=value, autojunk=False).ratio()
        credible = confidence >= 0.72 and (score >= best_score - tolerance or confidence >= 0.90)
        if not credible or ratio < 0.45:
            continue
        changed = _candidate_diff_indices(text, value)
        if not changed:
            continue
        indices.update(changed)
        if len(previews) < 4:
            previews.append(f"{method}={value[:18]}")
    return indices, previews


def analyze_block(block: Block) -> ReviewRisk:
    meta = block.metadata or {}
    if bool(meta.get("ocr_review_risk_resolved")):
        return ReviewRisk()
    text = str(block.text or "")
    chars = list(text)
    risk = ReviewRisk()

    if not text.strip():
        risk.add(100, "OCR 返回空列")
        return risk

    unknown_indices = [i for i, ch in enumerate(chars) if ch in _UNKNOWN]
    if unknown_indices:
        risk.add(85, f"包含 {len(unknown_indices)} 个未决/替代字符", unknown_indices)

    noise_indices = [i for i, ch in enumerate(chars) if ch in _NOISE or unicodedata.category(ch) == "Cc"]
    if noise_indices:
        risk.add(55, "包含 OCR 常见杂符号", noise_indices)

    # ASCII letters inside Japanese prose are often ruby/garbage; digits are
    # only a light warning because chapter numbers and quantities are valid.
    ascii_letter_indices = [i for i, ch in enumerate(chars) if ch.isascii() and ch.isalpha()]
    if ascii_letter_indices and _JAPANESE_RE.search(text):
        risk.add(28, "日文列中混入拉丁字母", ascii_letter_indices)

    odd_digit_indices = [i for i, ch in enumerate(chars) if ch.isascii() and ch.isdigit()]
    if len(odd_digit_indices) >= 2 and _JAPANESE_RE.search(text):
        risk.add(12, "日文列中数字较多，请确认是否为 OCR 污染", odd_digit_indices)

    # Several local OCR backends (notably Apple Live Text, Hayai and some
    # manga recognizers) use 0.0 to mean "backend does not expose a calibrated
    # block confidence", not "the recognizer is certain this result is bad".
    # Treat zero/negative confidence as unavailable unless the adapter
    # explicitly says the value is meaningful.  This prevents every physical
    # column from being painted as high-risk simply because the backend has no
    # confidence API.  Positive confidences keep the historical thresholds.
    confidence = float(block.confidence or 0.0)
    confidence_available_flag = meta.get("ocr_confidence_available")
    confidence_available = (
        bool(confidence_available_flag)
        if confidence_available_flag is not None
        else confidence > 0.0
    )
    if confidence_available:
        if confidence < 0.60:
            risk.add(55, f"OCR 置信度很低（{confidence:.2f}）")
        elif confidence < 0.82:
            risk.add(34, f"OCR 置信度偏低（{confidence:.2f}）")
        elif confidence < 0.93:
            risk.add(18, f"OCR 置信度一般（{confidence:.2f}），相近字仍需抽查")

    # A raw physical vertical column is not a linguistic sentence boundary.
    # Quotes/brackets routinely open in one column and close in the next, so
    # balance checks belong to the reflowed sentence stage.
    raw_physical_column = bool(
        meta.get("layout_group") == "fixed_region_column"
        and not meta.get("column_sentence_reflow")
        and not meta.get("atomic_ocr_sentence")
    )
    if not raw_physical_column:
        for opening, closing in _PAIRS:
            if text.count(opening) != text.count(closing):
                indices = [i for i, ch in enumerate(chars) if ch in {opening, closing}]
                risk.add(32, f"括号/引号不平衡：{opening}{closing}", indices)

    repeated = _adjacent_repeat(text)
    if repeated:
        chunk, indices = repeated
        risk.add(28, f"存在相邻重复片段“{chunk}”", indices)

    # After column-sentence reflow the block text may contain several physical
    # columns, while ``black_ink_estimated_chars`` still belongs to the owner
    # (first) column.  Comparing the whole sentence against that one-column
    # estimate creates systematic false positives.  Physical-column risk is
    # already evaluated before reflow in the single-OCR workflow, so sentence
    # blocks keep only sentence-level checks here.
    reflowed_multi_column = bool(
        meta.get("column_sentence_reflow")
        and int(meta.get("column_count", 1) or 1) > 1
    )
    estimated = 0 if reflowed_multi_column else int(meta.get("black_ink_estimated_chars", 0) or 0)
    actual = len(_ink_comparable_chars(text)) if estimated > 0 else len(_logical_chars(text))
    if estimated > 0:
        delta = actual - estimated
        # Geometry is deliberately coarse on short vertical columns.  Requiring
        # at least a three-glyph delta prevents normal punctuation/connected
        # components from turning short, otherwise clean dialogue columns into
        # mandatory review items.  Large deviations remain high-risk.
        tolerance = max(3, round(estimated * 0.25))
        if abs(delta) > tolerance:
            direction = "多" if delta > 0 else "少"
            risk.add(42, f"OCR 主体字符比黑像素估计{direction} {abs(delta)}（OCR {actual} / 估计 {estimated}）")
        elif abs(delta) >= max(3, round(estimated * 0.15)):
            risk.add(18, f"OCR 主体字符与黑像素估计存在偏差（OCR {actual} / 估计 {estimated}）")

    segmentation = meta.get("handwriting_input_glyph_segmentation") or {}
    if isinstance(segmentation, dict):
        physical = int(segmentation.get("review_physical_slot_count", 0) or 0)
        original = int(segmentation.get("review_original_text_count", actual) or actual)
        if physical and physical != original:
            delta = physical - original
            direction = "漏识" if delta > 0 else "疑似多识"
            points = 72 if delta > 0 else 45
            risk.add(points, f"逐字推子检测到 {physical} 个物理字框，OCR 为 {original} 字（{direction} {abs(delta)}）")

    spans = meta.get("black_ink_content_spans") or []
    if isinstance(spans, (list, tuple)) and len(spans) >= 2 and estimated and actual < max(2, round(estimated * 0.68)):
        risk.add(48, "多个独立文字块但 OCR 文本过短，可能漏掉短句")

    candidates = meta.get("column_ocr_candidates") or []
    conflict_indices, conflict_previews = _candidate_conflicts(text, candidates)
    if conflict_indices:
        preview = "；".join(conflict_previews)
        risk.add(48, "多路 OCR 候选出现可信的逐字差异（不预判哪一路正确）"
                 + (f"：{preview}" if preview else ""), sorted(conflict_indices))
    elif bool(meta.get("column_ocr_candidate_conflict")):
        preview_parts = []
        if isinstance(candidates, list):
            for item in candidates[:3]:
                if not isinstance(item, dict):
                    continue
                method = str(item.get("method", "候选"))
                candidate_text = str(item.get("text", ""))
                if candidate_text:
                    preview_parts.append(f"{method}={candidate_text[:16]}")
        preview = "；".join(preview_parts)
        risk.add(35, "多路 OCR 候选结果接近但不一致" + (f"：{preview}" if preview else ""))

    disagreements = meta.get("handwriting_review_disagreements") or []
    if isinstance(disagreements, list):
        strong = []
        for item in disagreements:
            try:
                score = float(item.get("score", 0.0) or 0.0)
                index = int(item.get("index", -1))
            except (TypeError, ValueError, AttributeError):
                continue
            if score >= 0.72:
                strong.append((index, str(item.get("ocr", "")), str(item.get("candidate", "")), score))
        if strong:
            preview = "、".join(f"{ocr or '∅'}→{cand}" for _, ocr, cand, _ in strong[:4])
            points = 45 if any(score >= 0.88 for *_, score in strong) else 32
            risk.add(points, f"手写轨迹候选与 OCR 冲突：{preview}", [i for i, *_ in strong if i >= 0])

    if text.startswith(("|", "」", "』")) or text.endswith(("|", "「", "『")):
        risk.add(22, "列首/列尾符号形态异常", [0, max(0, len(chars) - 1)])

    return risk


def annotate_ocr_review_risks(doc: UnifiedDocument) -> dict:
    """Annotate physical OCR columns and return an aggregate review report."""
    reviewed = 0
    suspicious = 0
    high = 0
    confidence_unavailable = 0
    reason_counter: Counter[str] = Counter()
    page_counts: Counter[int] = Counter()

    for block in doc.blocks:
        if not _reviewable(block):
            continue
        reviewed += 1
        meta = dict(block.metadata or {})
        try:
            raw_confidence = float(block.confidence or 0.0)
        except (TypeError, ValueError):
            raw_confidence = 0.0
        explicit_confidence = meta.get("ocr_confidence_available")
        if explicit_confidence is False or (explicit_confidence is None and raw_confidence <= 0.0):
            confidence_unavailable += 1
        risk = analyze_block(block)
        meta["ocr_review_risk_score"] = int(risk.score)
        meta["ocr_review_reasons"] = list(risk.reasons)
        meta["ocr_review_indices"] = sorted(risk.indices)
        meta["ocr_review_required"] = bool(risk.score >= 25)
        block.metadata = meta
        if risk.score >= 25:
            suspicious += 1
            page_counts[int(block.page or block.page_number or block.page_index or 0)] += 1
            for reason in risk.reasons:
                reason_counter[reason.split("：", 1)[0]] += 1
        if risk.score >= 60:
            high += 1

    report = {
        "version": 1,
        "columns": reviewed,
        "suspicious_columns": suspicious,
        "high_risk_columns": high,
        "confidence_unavailable_columns": confidence_unavailable,
        "pages_with_issues": dict(sorted(page_counts.items())),
        "top_reasons": reason_counter.most_common(12),
    }
    # Metadata is a dataclass but intentionally allows adding a stable field.
    try:
        doc.metadata.ocr_review_report = report
    except Exception:
        pass
    doc.add_log(
        "ocr_manual_review_risk",
        f"OCR 疑点筛查：{reviewed} 列中 {suspicious} 列需复核，{high} 列高风险；未自动修改正文",
        suspicious,
    )
    return report
