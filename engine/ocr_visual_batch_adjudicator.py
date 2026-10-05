# -*- coding: utf-8 -*-
"""Publication-safe visual verification for multi-model OCR conflicts.

The local OCR results remain immutable evidence.  Vision is deliberately split
into two passes: the first pass transcribes pixels without seeing candidates;
the second compares that independent transcription with A/B/C/D.  A local
validator then blocks unsafe punctuation, structure, coverage, majority
overrides and visually sensitive glyph changes from automatic write-back.

The model is therefore a first visual verifier, not an unquestioned final
authority.  Any unsupported or structurally unsafe answer remains in the human
review queue with explicit audit issues.
"""
from __future__ import annotations

import copy
import json
import math
import re
import tempfile
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Sequence

from PIL import Image, ImageDraw, ImageFont

from engine.ai_document_processor import parse_json_reply
from engine.ocr_image_text_review import build_review_entries, render_review_image

SCHEMA = "novel_formatter.ocr_visual_batch_adjudication.v3"


@dataclass(slots=True)
class VisualBatchOptions:
    batch_items: int = 16
    risk_only: bool = True
    allow_novel_text: bool = True
    reasoning_effort: str = "low"
    max_sheet_width: int = 2048
    cell_width: int = 300
    cell_height: int = 420
    routing_mode: str = "smart"  # smart | exhaustive | lean
    concurrency: int = 0  # 0 = auto
    max_concurrency: int = 8
    retry_incomplete: bool = True
    transcription_first: bool = True
    targeted_verify: bool = True
    # When provided, visual adjudication may only consider these stable row IDs.
    # This is used after local Paddle adjudication so locally-resolved rows never
    # consume visual-model requests. Empty means keep the normal routing scope.
    target_row_ids: tuple[str, ...] = ()

    def normalised(self) -> "VisualBatchOptions":
        routing = str(self.routing_mode or "smart").strip().lower()
        if routing not in {"smart", "exhaustive", "lean"}:
            routing = "smart"
        return VisualBatchOptions(
            batch_items=max(4, min(32, int(self.batch_items or 16))),
            risk_only=bool(self.risk_only),
            allow_novel_text=bool(self.allow_novel_text),
            reasoning_effort=(
                str(self.reasoning_effort or "low").lower()
                if str(self.reasoning_effort or "low").lower() in {"low", "high", "max"}
                else "low"
            ),
            max_sheet_width=max(1200, min(4096, int(self.max_sheet_width or 2048))),
            cell_width=max(220, min(520, int(self.cell_width or 300))),
            cell_height=max(280, min(760, int(self.cell_height or 420))),
            routing_mode=routing,
            concurrency=max(0, min(16, int(self.concurrency or 0))),
            max_concurrency=max(1, min(16, int(self.max_concurrency or 8))),
            retry_incomplete=bool(self.retry_incomplete),
            transcription_first=bool(self.transcription_first),
            targeted_verify=bool(self.targeted_verify),
            target_row_ids=tuple(str(value) for value in (self.target_row_ids or ()) if str(value)),
        )


def _candidate_texts(item: dict) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for row in item.get("candidates") or []:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text", "") or "")
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    fused = str(item.get("character_fused_text", "") or "")
    if fused and fused not in seen:
        out.append(fused)
    return out[:4]


def _model_texts(item: dict) -> list[str]:
    return [
        str(row.get("text", "") or "")
        for row in (item.get("candidates") or [])
        if isinstance(row, dict) and str(row.get("text", "") or "")
    ]


def _row_confidence(item: dict) -> float:
    try:
        return float(item.get("confidence", 0.0) or 0.0)
    except Exception:
        return 0.0


def _warnings(item: dict) -> list[str]:
    return [
        str(value)
        for value in [
            *(item.get("warnings") or []),
            *(item.get("character_fusion_warnings") or []),
            *(item.get("alignment_notes") or []),
        ]
        if str(value or "").strip()
    ]


_PUNCT_ONLY_RE = re.compile(r"[\s\u3000、。・，．！？!?…‥—―ー－‐‑–—~〜～『』「」（）()［］\[\]【】〈〉《》：:；;,.·]+")
_HIGH_RISK_RE = re.compile(
    r"(?:\d|[A-Za-z]|ない|なかった|なく|ません|ぬ|ず|無|未|非|否|不能|不可|"
    r"[ァ-ヺー]{4,})"
)


def _punctuation_equivalent(values: Sequence[str]) -> bool:
    compact = {_PUNCT_ONLY_RE.sub("", str(value or "")) for value in values}
    return len(compact) <= 1


def _contains_high_risk_text(values: Sequence[str]) -> bool:
    return any(_HIGH_RISK_RE.search(str(value or "")) for value in values)


def _text_has_deterministic_risk(value: str) -> bool:
    text = str(value or "")
    return bool(
        (text.count("「") != text.count("」"))
        or (text.count("『") != text.count("』"))
        or (_JAPANESE_CHAR_RE.search(text) and ("?" in text or "!" in text))
        or _INTERNAL_ASCII_SPACE_RE.search(text)
        or _ISOLATED_LEADING_NUMBER_RE.search(text)
        or any(pattern.search(text) for pattern in _KNOWN_OCR_LANGUAGE_ANOMALIES)
    )


def _sensitive_punctuation_variation(values: Sequence[str]) -> bool:
    texts = [str(value or "") for value in values]
    if any("?" in value or "!" in value for value in texts):
        return True
    signatures = {
        (value.count("「"), value.count("」"), value.count("『"), value.count("』"))
        for value in texts
    }
    return len(signatures) > 1 or any(_INTERNAL_ASCII_SPACE_RE.search(value) for value in texts)


_JAPANESE_CHAR_RE = re.compile(r"[ぁ-んァ-ヶ一-龯々〆ヵヶ]")
_INTERNAL_ASCII_SPACE_RE = re.compile(
    r"(?<=[ぁ-んァ-ヶ一-龯々〆ヵヶ「『（【。！？」』]) +(?=[ぁ-んァ-ヶ一-龯々〆ヵヶ「『）】])"
)
_ISOLATED_LEADING_NUMBER_RE = re.compile(r"^\s*\d{1,3}(?=[「『ぁ-んァ-ヶ一-龯々〆ヵヶ])")
_KNOWN_OCR_LANGUAGE_ANOMALIES = (
    re.compile(r"明けばない"),
    re.compile(r"それまででの"),
    re.compile(r"持つて"),
    re.compile(r"加うえ"),
)
_SENSITIVE_GLYPH_GROUPS = (
    frozenset(("っ", "つ")), frozenset(("ゃ", "や")),
    frozenset(("ゅ", "ゆ")), frozenset(("ょ", "よ")),
    frozenset(("ぁ", "あ")), frozenset(("ぃ", "い")),
    frozenset(("ぅ", "う")), frozenset(("ぇ", "え")),
    frozenset(("ぉ", "お")), frozenset(("は", "ば")),
    frozenset(("は", "ぱ")), frozenset(("ば", "ぱ")),
    frozenset(("ー", "一")), frozenset(("ー", "」")),
    frozenset(("？", "?")), frozenset(("！", "!")),
)
_BALANCED_PAIRS = (
    ("「", "」", "dialogue_quote_unbalanced"),
    ("『", "』", "citation_quote_unbalanced"),
    ("（", "）", "fullwidth_parenthesis_unbalanced"),
    ("【", "】", "square_bracket_unbalanced"),
    ("［", "］", "fullwidth_bracket_unbalanced"),
    ("〈", "〉", "angle_bracket_unbalanced"),
    ("《", "》", "double_angle_bracket_unbalanced"),
)


def _compact_visual_text(value: str) -> str:
    return re.sub(r"[\s\u3000]+", "", str(value or ""))


def _sensitive_glyph_change(left: str, right: str) -> bool:
    a = _compact_visual_text(left)
    b = _compact_visual_text(right)
    if a == b:
        return False
    for tag, a1, a2, b1, b2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        old = a[a1:a2]
        new = b[b1:b2]
        if any(frozenset((x, y)) in _SENSITIVE_GLYPH_GROUPS for x in old for y in new):
            return True
        if any(ch in "っゃゅょぁぃぅぇぉッャュョァィゥェォ" for ch in old + new):
            return True
        if any(ch in "゛゜ﾞﾟ" for ch in old + new):
            return True
    return False


def _evidence_coverage_issues(entry) -> list[str]:
    """Verify that every declared physical column has source geometry."""
    issues: list[str] = []
    column_ids = tuple(str(value) for value in (getattr(entry, "column_ids", ()) or ()) if str(value))
    regions = [value for value in (getattr(entry, "regions", ()) or ()) if isinstance(value, dict)]
    expected = max(1, int(getattr(entry, "column_count", 1) or 1), len(column_ids))
    if not regions:
        return ["missing_source_regions"]
    if len(regions) < expected:
        issues.append(f"incomplete_column_coverage:{len(regions)}/{expected}")
    if column_ids:
        region_ids = {
            str(region.get("column_id", "") or "")
            for region in regions
            if str(region.get("column_id", "") or "")
        }
        if region_ids:
            missing = [value for value in column_ids if value not in region_ids]
            if missing:
                issues.append(f"missing_column_ids:{len(missing)}")
    expected_pages: set[int] = set()
    for value in (getattr(entry, "pages", ()) or ()):
        try:
            page_no = int(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if page_no > 0:
            expected_pages.add(page_no)
    region_pages: set[int] = set()
    for region in regions:
        try:
            page_no = int(region.get("page", 0) or 0)
        except (TypeError, ValueError, OverflowError):
            continue
        if page_no > 0:
            region_pages.add(page_no)
    if expected_pages and not expected_pages.issubset(region_pages):
        issues.append("incomplete_page_coverage")
    return list(dict.fromkeys(issues))


def _rendered_coverage_issues(path: Path, expected_columns: int) -> list[str]:
    try:
        with Image.open(path) as image:
            width, height = image.size
    except Exception:
        return ["unreadable_visual_evidence"]
    if width <= 0 or height <= 0:
        return ["empty_visual_evidence"]
    if expected_columns > 1 and width < max(48, expected_columns * 24):
        return [f"implausibly_narrow_evidence:{width}px/{expected_columns}cols"]
    return []


def _deterministic_audit(
    item: dict,
    before: str,
    proposed: str,
    transcription: str,
    choice: str,
    coverage_issues: Sequence[str] = (),
) -> tuple[list[str], list[str]]:
    """Return ``(blocking_issues, targeted_verification_reasons)``."""
    blocking = [str(value) for value in coverage_issues if str(value)]
    verify: list[str] = []
    proposed = str(proposed or "")
    transcription = str(transcription or "")
    compact_proposed = _compact_visual_text(proposed)
    compact_transcription = _compact_visual_text(transcription)

    if not compact_transcription:
        blocking.append("independent_transcription_missing_or_uncertain")
    elif compact_proposed != compact_transcription:
        candidates = _candidate_texts(item)
        matching_other = any(
            _compact_visual_text(candidate) == compact_transcription
            and _compact_visual_text(candidate) != compact_proposed
            for candidate in candidates
        )
        if matching_other:
            blocking.append("independent_transcription_supports_other_candidate")
        else:
            ratio = SequenceMatcher(
                None, compact_proposed, compact_transcription, autojunk=False
            ).ratio()
            threshold = 0.985 if max(len(compact_proposed), len(compact_transcription)) >= 20 else 0.96
            if ratio < threshold:
                blocking.append(f"independent_transcription_mismatch:{ratio:.3f}")
            else:
                verify.append("minor_transcription_mismatch")

    for opener, closer, code in _BALANCED_PAIRS:
        if proposed.count(opener) != proposed.count(closer):
            blocking.append(code)
        if before.count(opener) == before.count(closer) and (
            before.count(opener), before.count(closer)
        ) != (proposed.count(opener), proposed.count(closer)):
            blocking.append(f"{code}:structure_changed")
    if _JAPANESE_CHAR_RE.search(proposed) and ("?" in proposed or "!" in proposed):
        blocking.append("ascii_question_or_exclamation_in_japanese")
    if _INTERNAL_ASCII_SPACE_RE.search(proposed):
        blocking.append("ascii_space_inside_japanese")
    if _ISOLATED_LEADING_NUMBER_RE.search(proposed):
        blocking.append("isolated_leading_number_or_page_noise")
    if any(pattern.search(proposed) for pattern in _KNOWN_OCR_LANGUAGE_ANOMALIES):
        blocking.append("known_japanese_ocr_anomaly")

    majority_text, majority_count, model_count, _majority_conf = _majority_profile(item)
    if model_count >= 3 and majority_count >= 2 and proposed != majority_text:
        verify.append("overrides_2_of_3_ocr_majority")
    comparison_base = majority_text or before
    if _sensitive_glyph_change(comparison_base, proposed):
        verify.append("sensitive_small_kana_dakuten_or_symbol_change")
    if choice == "X":
        verify.append("novel_text_x_requires_secondary_verification")
    if proposed != before and len(compact_proposed) >= 12:
        ratio = SequenceMatcher(
            None, _compact_visual_text(before), compact_proposed, autojunk=False
        ).ratio()
        if ratio < 0.72:
            blocking.append(f"large_unverified_rewrite:{ratio:.3f}")
    return list(dict.fromkeys(blocking)), list(dict.fromkeys(verify))


def _majority_profile(item: dict) -> tuple[str, int, int, float]:
    texts = _model_texts(item)
    if not texts:
        return "", 0, 0, 0.0
    counts = Counter(texts)
    majority_text, majority_count = counts.most_common(1)[0]
    confidences: list[float] = []
    for row in item.get("candidates") or []:
        if not isinstance(row, dict) or str(row.get("text", "") or "") != majority_text:
            continue
        try:
            value = float(row.get("confidence", 0.0) or 0.0)
        except Exception:
            value = 0.0
        if value > 0:
            confidences.append(value)
    majority_conf = min(confidences) if confidences else 0.0
    return majority_text, majority_count, len(texts), majority_conf


def _route_item(item: dict, routing_mode: str) -> tuple[bool, str, int]:
    """Return ``(send, reason, priority)``.

    Priority 3 is strongest evidence need; 1 is ordinary disagreement.  The
    default smart policy is deliberately conservative: it only skips a 2:1
    majority when the disagreement is punctuation/spacing-only and the local
    confidence is strong.  Lean mode additionally skips clean, high-confidence
    2:1 majorities unless the text contains high-risk signals.
    """
    candidates = [value for value in _candidate_texts(item) if value]
    unique = len(set(candidates))
    warnings = _warnings(item)
    local_reocr = bool(item.get("local_reocr_recommended", False))
    confidence = _row_confidence(item)
    alignment_repaired = bool(item.get("alignment_repaired", False))
    model_texts = _model_texts(item)

    if local_reocr:
        return True, "local_reocr_recommended", 3
    if warnings or alignment_repaired:
        return True, "local_warning_or_alignment", 3
    if unique >= 3:
        return True, "three_way_or_more_disagreement", 3
    if unique == 0:
        return False, "no_candidate_text", 0
    if unique == 1:
        if candidates and _text_has_deterministic_risk(candidates[0]):
            return True, "common_mode_text_anomaly", 3
        if confidence < 0.90:
            return True, "common_mode_low_confidence", 3
        return False, "high_confidence_consensus", 0

    # Two distinct candidate texts.
    high_risk = _contains_high_risk_text(candidates)
    majority_text, majority_count, model_count, majority_conf = _majority_profile(item)
    has_majority = model_count >= 3 and majority_count >= 2

    if routing_mode == "exhaustive":
        return True, "all_disagreements", 2 if has_majority else 3

    if not has_majority:
        return True, "no_local_majority", 3
    if high_risk:
        return True, "high_risk_majority_disagreement", 3
    if confidence < 0.94 or (majority_conf and majority_conf < 0.90):
        return True, "weak_majority_confidence", 2

    if _punctuation_equivalent(candidates) and confidence >= 0.96:
        if _sensitive_punctuation_variation(candidates):
            return True, "sensitive_punctuation_disagreement", 3
        return False, "safe_punctuation_majority", 0

    if routing_mode == "lean" and confidence >= 0.97 and (not majority_conf or majority_conf >= 0.95):
        return False, "lean_high_confidence_majority", 0

    # Smart mode keeps substantive 2:1 disagreements because historical OCR
    # evaluation showed that majority voting is not reliable enough to freeze
    # publication text on its own.
    return True, "substantive_majority_disagreement", 1


def _is_risky(item: dict) -> bool:
    return _route_item(item, "smart")[0]


def analyse_routing(package: dict, *, routing_mode: str = "smart", risk_only: bool = True) -> dict:
    """Cheap preflight statistics for UI/reporting; does not touch source images."""
    mode = str(routing_mode or "smart").strip().lower()
    if mode not in {"smart", "exhaustive", "lean"}:
        mode = "smart"
    reasons: Counter[str] = Counter()
    priorities: Counter[int] = Counter()
    targets = 0
    skipped = 0
    for item in list(package.get("editable_items") or []):
        if not risk_only:
            send = bool(_candidate_texts(item))
            reason = "risk_filter_disabled" if send else "no_candidate_text"
            priority = 1 if send else 0
        else:
            send, reason, priority = _route_item(item, mode)
        reasons[reason] += 1
        if send:
            targets += 1
            priorities[priority] += 1
        else:
            skipped += 1
    return {
        "routing_mode": mode,
        "target_items": targets,
        "skipped_low_risk": skipped,
        "reasons": dict(reasons),
        "priority_counts": {str(k): v for k, v in sorted(priorities.items())},
    }


def _fit_crop(image: Image.Image, width: int, height: int) -> Image.Image:
    im = image.convert("RGB")
    ratio = min(width / max(1, im.width), height / max(1, im.height), 1.0)
    if ratio < 1.0:
        im = im.resize((max(1, round(im.width * ratio)), max(1, round(im.height * ratio))), Image.Resampling.LANCZOS)
    return im


def _build_contact_sheet(entries: Sequence[tuple[int, Path]], output: Path, options: VisualBatchOptions) -> Path:
    if not entries:
        raise ValueError("没有可生成证据板的图片。")
    cell_w, cell_h = options.cell_width, options.cell_height
    cols = max(2, min(8, options.max_sheet_width // cell_w))
    rows = math.ceil(len(entries) / cols)
    pad = 10
    header_h = 28
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for pos, (alias, path) in enumerate(entries):
        col = pos % cols
        row = pos // cols
        left, top = col * cell_w, row * cell_h
        draw.rectangle((left, top, left + cell_w - 1, top + cell_h - 1), outline=(190, 190, 190), width=1)
        draw.text((left + pad, top + 7), f"#{alias}", fill=(0, 0, 0), font=font)
        with Image.open(path) as opened:
            crop = _fit_crop(opened, cell_w - 2 * pad, cell_h - header_h - 2 * pad)
        x = left + (cell_w - crop.width) // 2
        y = top + header_h + max(0, (cell_h - header_h - crop.height) // 2)
        canvas.paste(crop, (x, y))
        crop.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="WEBP", quality=86, method=4)
    canvas.close()
    return output


def _compact_candidate_line(
    alias: int,
    item: dict,
    transcription: str = "",
) -> tuple[str, dict[str, str]]:
    labels = "ABCD"
    values = _candidate_texts(item)
    mapping = {labels[i]: text for i, text in enumerate(values[:4])}
    chunks = [f"{alias}", f"I={transcription or 'UNCERTAIN'}"]
    for key, value in mapping.items():
        chunks.append(f"{key}={value}")
    return "|".join(chunks), mapping


TRANSCRIPTION_PROMPT = """You are a literal Japanese publication-image transcriber. The contact sheet contains numbered TARGET crops. OCR candidates are intentionally hidden.\nFor every number, independently transcribe every visible TARGET character in Japanese reading order. Preserve exact small kana, dakuten/handakuten, full-width punctuation, quotes, long-vowel marks, digits and spaces. Never repair grammar, infer clipped text, translate, normalize or continue beyond visible pixels.\nIf any part of the target is cropped out, ambiguous, too small, or not fully represented, return Q for that number.\nOutput compact JSON only: {\"t\":[[id,\"exact visible text\"],[id,\"Q\"]]}. No explanations."""


COMPARISON_PROMPT = """You are the second-stage verifier for Japanese publication OCR. The attached sheet is the same source evidence.\nI is an independent image transcription created before candidates were shown. A/B/C/D are local OCR candidates. Treat image pixels as highest authority and candidates only as hints. Check the entire target, including text where all candidates agree.\nReturn A/B/C/D only when that complete candidate is supported by both the image and I. Use X only when all candidates are wrong and the complete exact image text is readable. Use Q when I is UNCERTAIN, evidence is incomplete, punctuation/small kana/dakuten is ambiguous, or the crop does not cover all proposed characters.\nDo not translate, polish, normalize, repair by grammar alone, or accept text outside the visible crop.\nOutput compact JSON only: {\"r\":[[id,\"A\"],[id,\"X\",\"exact text\"],[id,\"Q\"]]}. No explanations.\nITEMS\n{{ITEMS}}"""


TARGETED_VERIFY_PROMPT = """Independently re-check this single high-risk Japanese OCR decision. Inspect the pixels character by character and actively try to disprove the proposed text. Pay special attention to small kana, dakuten/handakuten, は/ば/ぱ, っ/つ, ー/一/」, full-width punctuation, quote balance, leading page numbers, and whether every proposed character is visible. Grammar may flag suspicion but must not invent text.\nReturn {\"v\":\"A\"} only if the proposed text is completely supported. Return {\"v\":\"R\"} if another supplied text is clearly supported, or {\"v\":\"Q\"} when evidence is incomplete or ambiguous. No explanation.\n{{ITEM}}"""


def _parse_transcription_reply(raw: str, valid_aliases: set[int]) -> dict[int, str]:
    data = parse_json_reply(raw)
    rows = data.get("t", data.get("transcriptions", [])) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError("AI 独立抄写返回缺少 t 数组。")
    result: dict[int, str] = {}
    for row in rows:
        if isinstance(row, dict):
            alias = row.get("i", row.get("id"))
            text = row.get("t", row.get("text", "Q"))
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            alias, text = row[0], row[1]
        else:
            continue
        try:
            alias_i = int(alias)
        except Exception:
            continue
        if alias_i not in valid_aliases or alias_i in result:
            continue
        value = str(text or "").strip()
        if value.upper() in {"Q", "UNCERTAIN", "INSUFFICIENT_EVIDENCE"}:
            value = ""
        result[alias_i] = value
    return result


def _parse_reply(raw: str, valid_aliases: set[int]) -> dict[int, tuple[str, str]]:
    data = parse_json_reply(raw)
    if isinstance(data, dict):
        rows = data.get("r", data.get("results", []))
    else:
        rows = data
    if not isinstance(rows, list):
        raise ValueError("AI 裁决返回缺少 r 数组。")
    result: dict[int, tuple[str, str]] = {}
    for row in rows:
        if isinstance(row, dict):
            alias = row.get("i", row.get("id"))
            choice = row.get("c", row.get("choice", "Q"))
            text = row.get("t", row.get("text", ""))
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            alias, choice = row[0], row[1]
            text = row[2] if len(row) >= 3 else ""
        else:
            continue
        try:
            alias_i = int(alias)
        except Exception:
            continue
        if alias_i not in valid_aliases or alias_i in result:
            continue
        choice_s = str(choice or "Q").strip().upper()
        if choice_s not in {"A", "B", "C", "D", "X", "Q"}:
            choice_s = "Q"
        result[alias_i] = (choice_s, str(text or ""))
    return result


def _parse_targeted_reply(raw: str) -> str:
    data = parse_json_reply(raw)
    if isinstance(data, dict):
        value = data.get("v", data.get("verdict", "Q"))
    else:
        value = data
    verdict = str(value or "Q").strip().upper()
    return verdict if verdict in {"A", "R", "Q"} else "Q"


def _novel_text_safe(text: str, candidates: Sequence[str]) -> bool:
    value = str(text or "").strip()
    if not value or len(value) > 512:
        return False
    best = 0.0
    for candidate in candidates:
        candidate = str(candidate or "")
        if not candidate:
            continue
        best = max(best, SequenceMatcher(None, candidate, value, autojunk=False).ratio())
    return best >= (0.55 if len(value) < 20 else 0.68)


def _auto_concurrency(client, opts: VisualBatchOptions) -> tuple[int, int, bool]:
    if opts.concurrency > 0:
        return opts.concurrency, opts.concurrency, False
    provider = str(getattr(client, "provider", "") or "").lower()
    # Local/custom endpoints are not assumed to be able to handle parallel
    # multimodal uploads.  Online managed providers start at four, following the
    # same conservative cold-start principle as mature translation schedulers.
    start = 1 if provider in {"ollama", "custom"} else 4
    maximum = 1 if provider in {"ollama", "custom"} else opts.max_concurrency
    return min(start, maximum), maximum, True


def _adjudicate_visual_batches_v2_legacy(
    client,
    package: dict,
    fused_document,
    *,
    options: VisualBatchOptions | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[dict, dict]:
    opts = (options or VisualBatchOptions()).normalised()
    result = copy.deepcopy(package)
    items = list(result.get("editable_items") or [])
    entries = build_review_entries(fused_document)
    entry_by_row = {
        int(entry.source_row_index): entry
        for entry in entries
        if int(getattr(entry, "source_row_index", -1)) >= 0
    }

    route_records: list[tuple[int, int, str]] = []
    skipped_reasons: Counter[str] = Counter()
    target_id_set = set(opts.target_row_ids)
    for idx, item in enumerate(items):
        item_row_id = str(item.get("row_id", "") or "")
        if target_id_set and item_row_id not in target_id_set:
            skipped_reasons["pre_adjudicated_local"] += 1
            continue
        if not opts.risk_only:
            send = bool(_candidate_texts(item))
            reason, priority = ("risk_filter_disabled", 1) if send else ("no_candidate_text", 0)
        else:
            send, reason, priority = _route_item(item, opts.routing_mode)
        if send:
            route_records.append((idx, priority, reason))
        else:
            skipped_reasons[reason] += 1
    # Preserve stable row/page order so neighbouring physical evidence naturally
    # lands on the same sheet. Risk priority is reported but does not reshuffle
    # the immutable comparison order.
    target_indices = [idx for idx, _priority, _reason in route_records]
    route_reason_by_index = {idx: reason for idx, _priority, reason in route_records}

    report_items: list[dict] = []
    unresolved: list[str] = []
    changed = 0
    exact_choices = 0
    novel_choices = 0
    requests = 0
    incomplete_retries = 0
    transport_split_retries = 0
    counter_lock = threading.Lock()
    concurrency_start, concurrency_max, auto_concurrency = _auto_concurrency(client, opts)
    current_concurrency = concurrency_start
    peak_concurrency = current_concurrency

    with tempfile.TemporaryDirectory(prefix="nf_ocr_ai_evidence_") as temp_raw:
        temp = Path(temp_raw)
        prepared: list[tuple[int, Path]] = []
        for idx in target_indices:
            if cancel_check and cancel_check():
                raise RuntimeError("AI 裁决已取消。")
            entry = entry_by_row.get(idx)
            if entry is None:
                unresolved.append(str(items[idx].get("row_id", "")))
                continue
            path = Path(render_review_image(entry, temp / f"crop-{idx:06d}.png") or "")
            if not path.is_file():
                unresolved.append(str(items[idx].get("row_id", "")))
                continue
            prepared.append((idx, path))

        batches = [prepared[i:i + opts.batch_items] for i in range(0, len(prepared), opts.batch_items)]
        if progress_callback:
            progress_callback({
                "stage": "准备视觉证据", "current": 0, "total": max(1, len(batches)),
                "target_rows": len(target_indices), "visual_rows": len(prepared),
                "skipped_rows": len(items) - len(target_indices), "routing_mode": opts.routing_mode,
                "concurrency": current_concurrency,
            })

        def make_request(batch: Sequence[tuple[int, Path]], request_tag: str, depth: int = 0):
            nonlocal requests, incomplete_retries, transport_split_retries
            if cancel_check and cancel_check():
                raise RuntimeError("AI 裁决已取消。")
            alias_to_index: dict[int, int] = {}
            lines: list[str] = []
            sheet_entries: list[tuple[int, Path]] = []
            for local_alias, (idx, crop_path) in enumerate(batch, 1):
                alias_to_index[local_alias] = idx
                line, _mapping = _compact_candidate_line(local_alias, items[idx])
                lines.append(line)
                sheet_entries.append((local_alias, crop_path))
            sheet = _build_contact_sheet(sheet_entries, temp / f"sheet-{request_tag}.webp", opts)
            prompt = PROMPT.replace("{{ITEMS}}", "\n".join(lines))
            try:
                raw = client.call_json(prompt, [sheet], temperature=0.0, reasoning_effort=opts.reasoning_effort)
                with counter_lock:
                    requests += 1
            except Exception as exc:
                transient = bool(getattr(client, "_looks_transient_error", lambda _exc: False)(exc))
                if transient and len(batch) > 16 and depth < 2:
                    with counter_lock:
                        transport_split_retries += 1
                    mid = max(1, len(batch) // 2)
                    left = make_request(batch[:mid], request_tag + "a", depth + 1)
                    right = make_request(batch[mid:], request_tag + "b", depth + 1)
                    return {**left, **right}
                raise
            parsed = _parse_reply(raw, set(alias_to_index))
            converted = {alias_to_index[alias]: value for alias, value in parsed.items() if alias in alias_to_index}
            missing = [(idx, path) for idx, path in batch if idx not in converted]
            if opts.retry_incomplete and missing and len(missing) >= max(2, math.ceil(len(batch) * 0.15)) and depth < 1:
                with counter_lock:
                    incomplete_retries += 1
                retry_values = make_request(missing, request_tag + "m", depth + 1)
                converted.update(retry_values)
            return converted

        completed_batches = 0
        batch_cursor = 0
        all_decisions: dict[int, tuple[str, str]] = {}
        while batch_cursor < len(batches):
            if cancel_check and cancel_check():
                raise RuntimeError("AI 裁决已取消。")
            wave = batches[batch_cursor:batch_cursor + current_concurrency]
            wave_base = batch_cursor
            usage_before = client.usage_snapshot() if hasattr(client, "usage_snapshot") else {}
            with ThreadPoolExecutor(max_workers=max(1, current_concurrency), thread_name_prefix="nf-ai-adjudicate") as pool:
                future_map = {
                    pool.submit(make_request, batch, f"{wave_base + offset + 1:04d}"): (offset, batch)
                    for offset, batch in enumerate(wave)
                }
                for future in as_completed(future_map):
                    decisions = future.result()
                    all_decisions.update(decisions)
                    completed_batches += 1
                    if progress_callback:
                        progress_callback({
                            "stage": "AI 批量视觉裁决", "current": completed_batches, "total": len(batches),
                            "batch_rows": len(future_map[future][1]), "sheet_count": 1,
                            "target_rows": len(target_indices), "routing_mode": opts.routing_mode,
                            "concurrency": current_concurrency, "peak_concurrency": peak_concurrency,
                        })
            batch_cursor += len(wave)
            usage_after = client.usage_snapshot() if hasattr(client, "usage_snapshot") else {}
            retry_delta = int(usage_after.get("transport_retries", 0) or 0) - int(usage_before.get("transport_retries", 0) or 0)
            if auto_concurrency:
                if retry_delta > 0:
                    current_concurrency = max(1, current_concurrency // 2)
                else:
                    current_concurrency = min(concurrency_max, current_concurrency + 1)
                peak_concurrency = max(peak_concurrency, current_concurrency)

        for idx, _crop in prepared:
            item = items[idx]
            row_id = str(item.get("row_id", "") or "")
            before = str(item.get("edited_text", item.get("original_fused_text", "")) or "")
            candidates = _candidate_texts(item)
            mapping = {"ABCD"[i]: text for i, text in enumerate(candidates[:4])}
            choice, xtext = all_decisions.get(idx, ("Q", ""))
            after = before
            reason = ""
            confidence = "low_uncertain"
            if choice in mapping:
                after = mapping[choice]
                reason = f"视觉证据选择候选 {choice}"
                confidence = "high_consensus"
                exact_choices += 1
            elif choice == "X" and opts.allow_novel_text and _novel_text_safe(xtext, candidates):
                after = xtext
                reason = "所有现有候选均不符；视觉证据给出窄幅修正"
                confidence = "medium_context"
                novel_choices += 1
            else:
                unresolved.append(row_id)
                reason = "视觉证据不足、AI 未完整返回，或 X 修正未通过本地防改写校验"
            if confidence != "low_uncertain" and after != before:
                item["edited_text"] = after
                changed += 1
            report_items.append({
                "item_id": row_id,
                "page": int(item.get("page", 0) or 0),
                "before": before,
                "after": after if confidence != "low_uncertain" else before,
                "proposed_text": after if confidence != "low_uncertain" else "",
                "evidence_source": ["VISUAL_BATCH", choice],
                "route_reason": route_reason_by_index.get(idx, ""),
                "change_type": "other" if after != before else "unchanged",
                "confidence": confidence,
                "needs_human_review": confidence == "low_uncertain",
                "reason": reason,
                "audit_issues": [],
            })

    unique_unresolved = sorted(set(value for value in unresolved if value))
    usage = client.usage_snapshot() if hasattr(client, "usage_snapshot") else {}
    stats = {
        "target_items": len(target_indices),
        "visual_items": len(prepared),
        "skipped_low_risk": len(items) - len(target_indices),
        "pre_adjudicated_local": int(skipped_reasons.get("pre_adjudicated_local", 0) or 0),
        "target_scope_rows": len(target_id_set) if target_id_set else len(items),
        "missing_visual_evidence": max(0, len(target_indices) - len(prepared)),
        "applied_changes": changed,
        "accepted_existing_candidates": exact_choices,
        "accepted_novel_text": novel_choices,
        "uncertain_items": len(unique_unresolved),
        "request_count": requests,
        "incomplete_retry_batches": incomplete_retries,
        "transport_split_retries": transport_split_retries,
        "concurrency_start": concurrency_start,
        "concurrency_peak": peak_concurrency,
        "concurrency_end": current_concurrency,
        "auto_concurrency": auto_concurrency,
    }
    report = {
        "schema": SCHEMA,
        "mode": "visual_evidence_batch",
        "routing_mode": opts.routing_mode,
        "stats": stats,
        # Keep v1 flat keys for compatibility with existing consumers.
        "target_rows": len(target_indices),
        "visual_rows": len(prepared),
        "request_count": requests,
        "batch_items": opts.batch_items,
        "accepted_existing_candidates": exact_choices,
        "accepted_novel_text": novel_choices,
        "changed_rows": changed,
        "skipped_reasons": dict(skipped_reasons),
        "low_uncertain": unique_unresolved,
        "items": report_items,
        "usage": usage,
    }
    return result, report


def adjudicate_visual_batches(
    client,
    package: dict,
    fused_document,
    *,
    options: VisualBatchOptions | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[dict, dict]:
    """Run transcription-first visual verification with guarded write-back."""
    opts = (options or VisualBatchOptions()).normalised()
    result = copy.deepcopy(package)
    items = list(result.get("editable_items") or [])
    entries = build_review_entries(fused_document)
    entry_by_row = {
        int(entry.source_row_index): entry
        for entry in entries
        if int(getattr(entry, "source_row_index", -1)) >= 0
    }

    route_records: list[tuple[int, int, str]] = []
    skipped_reasons: Counter[str] = Counter()
    target_id_set = set(opts.target_row_ids)
    for idx, item in enumerate(items):
        item_row_id = str(item.get("row_id", "") or "")
        if target_id_set and item_row_id not in target_id_set:
            skipped_reasons["pre_adjudicated_local"] += 1
            continue
        if not opts.risk_only:
            send = bool(_candidate_texts(item))
            reason, priority = ("risk_filter_disabled", 1) if send else ("no_candidate_text", 0)
        else:
            send, reason, priority = _route_item(item, opts.routing_mode)
        if send:
            route_records.append((idx, priority, reason))
        else:
            skipped_reasons[reason] += 1
    target_indices = [idx for idx, _priority, _reason in route_records]
    route_reason_by_index = {idx: reason for idx, _priority, reason in route_records}

    report_items: list[dict] = []
    unresolved: list[str] = []
    changed = 0
    exact_choices = 0
    novel_choices = 0
    blocked_by_validator = 0
    requests = 0
    transcription_requests = 0
    comparison_requests = 0
    targeted_verify_requests = 0
    incomplete_retries = 0
    transport_split_retries = 0
    counter_lock = threading.Lock()
    concurrency_start, concurrency_max, auto_concurrency = _auto_concurrency(client, opts)
    current_concurrency = concurrency_start
    peak_concurrency = current_concurrency

    def invoke(prompt: str, image: Path, stage: str, effort: str | None = None) -> str:
        nonlocal requests, transcription_requests, comparison_requests, targeted_verify_requests
        raw = client.call_json(
            prompt,
            [image],
            temperature=0.0,
            reasoning_effort=effort or opts.reasoning_effort,
        )
        with counter_lock:
            requests += 1
            if stage == "transcription":
                transcription_requests += 1
            elif stage == "comparison":
                comparison_requests += 1
            else:
                targeted_verify_requests += 1
        return raw

    with tempfile.TemporaryDirectory(prefix="nf_ocr_ai_evidence_") as temp_raw:
        temp = Path(temp_raw)
        prepared: list[tuple[int, Path]] = []
        coverage_by_index: dict[int, list[str]] = {}
        for idx in target_indices:
            if cancel_check and cancel_check():
                raise RuntimeError("AI 裁决已取消。")
            entry = entry_by_row.get(idx)
            if entry is None:
                coverage_by_index[idx] = ["missing_review_entry"]
                continue
            coverage = _evidence_coverage_issues(entry)
            if coverage:
                coverage_by_index[idx] = coverage
                continue
            # Preferred sentence images may be stale or contain only one of many
            # declared columns. Force reconstruction from immutable regions.
            render_entry = copy.copy(entry)
            render_entry.preferred_image_path = ""
            path = Path(render_review_image(render_entry, temp / f"crop-{idx:06d}.png") or "")
            if not path.is_file():
                coverage_by_index[idx] = ["missing_rendered_visual_evidence"]
                continue
            expected_columns = max(1, int(getattr(entry, "column_count", 1) or 1), len(entry.column_ids or ()))
            coverage = _rendered_coverage_issues(path, expected_columns)
            if coverage:
                coverage_by_index[idx] = coverage
                continue
            prepared.append((idx, path))

        batches = [prepared[i:i + opts.batch_items] for i in range(0, len(prepared), opts.batch_items)]
        if progress_callback:
            progress_callback({
                "stage": "准备完整视觉证据", "current": 0, "total": max(1, len(batches)),
                "target_rows": len(target_indices), "visual_rows": len(prepared),
                "coverage_failures": len(coverage_by_index), "routing_mode": opts.routing_mode,
                "concurrency": current_concurrency,
            })

        def make_request(batch: Sequence[tuple[int, Path]], request_tag: str, depth: int = 0):
            nonlocal incomplete_retries, transport_split_retries
            if cancel_check and cancel_check():
                raise RuntimeError("AI 裁决已取消。")
            alias_to_index = {alias: idx for alias, (idx, _path) in enumerate(batch, 1)}
            sheet_entries = [(alias, path) for alias, (_idx, path) in enumerate(batch, 1)]
            sheet = _build_contact_sheet(sheet_entries, temp / f"sheet-{request_tag}.webp", opts)
            try:
                if opts.transcription_first:
                    raw_transcription = invoke(TRANSCRIPTION_PROMPT, sheet, "transcription")
                    transcriptions = _parse_transcription_reply(raw_transcription, set(alias_to_index))
                else:
                    transcriptions = {alias: "" for alias in alias_to_index}

                lines: list[str] = []
                for alias, idx in alias_to_index.items():
                    line, _mapping = _compact_candidate_line(alias, items[idx], transcriptions.get(alias, ""))
                    lines.append(line)
                raw_decision = invoke(
                    COMPARISON_PROMPT.replace("{{ITEMS}}", "\n".join(lines)),
                    sheet,
                    "comparison",
                )
                decisions = _parse_reply(raw_decision, set(alias_to_index))
            except Exception as exc:
                transient = bool(getattr(client, "_looks_transient_error", lambda _exc: False)(exc))
                if transient and len(batch) > 4 and depth < 2:
                    with counter_lock:
                        transport_split_retries += 1
                    mid = max(1, len(batch) // 2)
                    left = make_request(batch[:mid], request_tag + "a", depth + 1)
                    right = make_request(batch[mid:], request_tag + "b", depth + 1)
                    return {**left, **right}
                raise

            converted: dict[int, tuple[str, str, str]] = {}
            for alias, idx in alias_to_index.items():
                transcription = transcriptions.get(alias, "")
                if opts.transcription_first and not transcription:
                    converted[idx] = ("Q", "", "")
                    continue
                choice, text = decisions.get(alias, ("Q", ""))
                converted[idx] = (choice, text, transcription)

            missing = [row for alias, row in enumerate(batch, 1) if alias not in transcriptions or alias not in decisions]
            if opts.retry_incomplete and missing and depth < 1:
                with counter_lock:
                    incomplete_retries += 1
                converted.update(make_request(missing, request_tag + "m", depth + 1))
            return converted

        completed_batches = 0
        batch_cursor = 0
        all_decisions: dict[int, tuple[str, str, str]] = {}
        while batch_cursor < len(batches):
            if cancel_check and cancel_check():
                raise RuntimeError("AI 裁决已取消。")
            wave = batches[batch_cursor:batch_cursor + current_concurrency]
            wave_base = batch_cursor
            usage_before = client.usage_snapshot() if hasattr(client, "usage_snapshot") else {}
            with ThreadPoolExecutor(max_workers=max(1, current_concurrency), thread_name_prefix="nf-ai-adjudicate") as pool:
                future_map = {
                    pool.submit(make_request, batch, f"{wave_base + offset + 1:04d}"): batch
                    for offset, batch in enumerate(wave)
                }
                for future in as_completed(future_map):
                    all_decisions.update(future.result())
                    completed_batches += 1
                    if progress_callback:
                        progress_callback({
                            "stage": "独立抄写与候选比较", "current": completed_batches,
                            "total": len(batches), "batch_rows": len(future_map[future]),
                            "target_rows": len(target_indices), "routing_mode": opts.routing_mode,
                            "concurrency": current_concurrency, "peak_concurrency": peak_concurrency,
                        })
            batch_cursor += len(wave)
            usage_after = client.usage_snapshot() if hasattr(client, "usage_snapshot") else {}
            retry_delta = int(usage_after.get("transport_retries", 0) or 0) - int(usage_before.get("transport_retries", 0) or 0)
            if auto_concurrency:
                current_concurrency = (
                    max(1, current_concurrency // 2)
                    if retry_delta > 0
                    else min(concurrency_max, current_concurrency + 1)
                )
                peak_concurrency = max(peak_concurrency, current_concurrency)

        path_by_index = dict(prepared)
        for idx in target_indices:
            item = items[idx]
            row_id = str(item.get("row_id", "") or "")
            before = str(item.get("edited_text", item.get("original_fused_text", "")) or "")
            candidates = _candidate_texts(item)
            mapping = {"ABCD"[i]: text for i, text in enumerate(candidates[:4])}
            choice, xtext, transcription = all_decisions.get(idx, ("Q", "", ""))
            proposed = before
            preliminary_valid = False
            if choice in mapping:
                proposed = mapping[choice]
                preliminary_valid = True
            elif choice == "X" and opts.allow_novel_text and _novel_text_safe(xtext, candidates):
                proposed = xtext
                preliminary_valid = True

            blocking, verify_reasons = _deterministic_audit(
                item, before, proposed, transcription, choice, coverage_by_index.get(idx, ()),
            )
            if not preliminary_valid:
                blocking.append("visual_decision_uncertain_or_invalid")

            secondary_verdict = ""
            if preliminary_valid and proposed != before and verify_reasons and not blocking:
                if opts.targeted_verify and idx in path_by_index:
                    majority_text, _count, _total, _confidence = _majority_profile(item)
                    detail = json.dumps({
                        "proposed": proposed,
                        "independent_transcription": transcription,
                        "local_majority": majority_text,
                        "candidates": candidates,
                        "risk_reasons": verify_reasons,
                    }, ensure_ascii=False, separators=(",", ":"))
                    try:
                        raw_verify = invoke(
                            TARGETED_VERIFY_PROMPT.replace("{{ITEM}}", detail),
                            path_by_index[idx],
                            "targeted",
                            effort="high",
                        )
                        secondary_verdict = _parse_targeted_reply(raw_verify)
                    except Exception:
                        secondary_verdict = "Q"
                    if secondary_verdict != "A":
                        blocking.append("secondary_visual_verification_not_accepted")
                else:
                    blocking.append("secondary_visual_verification_required")

            blocking = list(dict.fromkeys(blocking))
            needs_review = bool(blocking)
            if needs_review:
                unresolved.append(row_id)
                blocked_by_validator += 1
                confidence = "low_uncertain"
                reason = "程序安全闸门阻止自动落账：" + "、".join(blocking)
                after = before
            else:
                after = proposed
                if choice == "X":
                    confidence = "validated_novel_text"
                    novel_choices += 1
                elif proposed == before:
                    confidence = "validated_unchanged"
                    exact_choices += 1
                else:
                    confidence = "validated_existing_candidate"
                    exact_choices += 1
                reason = "独立抄写、候选比较与程序校验一致"
                if secondary_verdict == "A":
                    reason += "；高风险项已通过第二次视觉反证"
                if after != before:
                    item["edited_text"] = after
                    changed += 1

            report_items.append({
                "item_id": row_id,
                "page": int(item.get("page", 0) or 0),
                "before": before,
                "after": after,
                "proposed_text": proposed if preliminary_valid else "",
                "image_transcription": transcription,
                "evidence_source": ["VISUAL_TRANSCRIPTION", "VISUAL_COMPARE", choice],
                "route_reason": route_reason_by_index.get(idx, ""),
                "change_type": "other" if proposed != before else "unchanged",
                "confidence": confidence,
                "needs_human_review": needs_review,
                "reason": reason,
                "audit_issues": blocking,
                "targeted_verification_reasons": verify_reasons,
                "secondary_verdict": secondary_verdict,
            })

    unique_unresolved = sorted(set(value for value in unresolved if value))
    usage = client.usage_snapshot() if hasattr(client, "usage_snapshot") else {}
    stats = {
        "target_items": len(target_indices),
        "visual_items": len(prepared),
        "skipped_low_risk": len(items) - len(target_indices),
        "pre_adjudicated_local": int(skipped_reasons.get("pre_adjudicated_local", 0) or 0),
        "target_scope_rows": len(target_id_set) if target_id_set else len(items),
        "missing_visual_evidence": len(coverage_by_index),
        "evidence_coverage_failures": len(coverage_by_index),
        "applied_changes": changed,
        "accepted_existing_candidates": exact_choices,
        "accepted_novel_text": novel_choices,
        "blocked_by_validator": blocked_by_validator,
        "uncertain_items": len(unique_unresolved),
        "request_count": requests,
        "transcription_requests": transcription_requests,
        "comparison_requests": comparison_requests,
        "targeted_verify_requests": targeted_verify_requests,
        "incomplete_retry_batches": incomplete_retries,
        "transport_split_retries": transport_split_retries,
        "concurrency_start": concurrency_start,
        "concurrency_peak": peak_concurrency,
        "concurrency_end": current_concurrency,
        "auto_concurrency": auto_concurrency,
    }
    report = {
        "schema": SCHEMA,
        "mode": "transcription_first_visual_verification",
        "routing_mode": opts.routing_mode,
        "stats": stats,
        "target_rows": len(target_indices),
        "visual_rows": len(prepared),
        "request_count": requests,
        "batch_items": opts.batch_items,
        "accepted_existing_candidates": exact_choices,
        "accepted_novel_text": novel_choices,
        "changed_rows": changed,
        "skipped_reasons": dict(skipped_reasons),
        "low_uncertain": unique_unresolved,
        "items": report_items,
        "usage": usage,
    }
    return result, report
