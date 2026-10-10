from __future__ import annotations

import copy
import re
import unicodedata
from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

from models.document import Block, BlockType, UnifiedDocument


# CJK Radicals Supplement characters do not generally have NFKC mappings.
# Keep this intentionally small and explicit: publication mode must never guess
# a radical-to-ideograph substitution that is not visually/source verified.
VERIFIED_CJK_RADICAL_MAP = {
    "⻘": "青",  # U+2ED8 CJK RADICAL BLUE -> U+9752 CJK UNIFIED IDEOGRAPH-9752
}


@dataclass(frozen=True)
class PublicationCorrection:
    before: str
    after: str
    reason: str
    source: str = "editorial_proof"


@dataclass(frozen=True)
class AppliedCorrection:
    block_id: str
    page: int
    before: str
    after: str
    reason: str
    source: str


def _is_text_block(block: Block) -> bool:
    return block.type != BlockType.IMAGE_REF


def normalize_verified_publication_unicode(text: str) -> str:
    """Normalize only source-verified compatibility characters.

    Unlike NFKC, this function deliberately preserves ordinary Japanese glyph
    identity and punctuation.  It only converts code points with an explicit,
    verified publication mapping.
    """
    value = str(text or "")
    return "".join(VERIFIED_CJK_RADICAL_MAP.get(ch, ch) for ch in value)


def normalize_vertical_parentheses(text: str) -> str:
    """Use Japanese full-width parentheses in Japanese vertical prose.

    This is a typography-only conversion.  E-mail/URL-like strings are left
    alone, and no lexical characters are changed.
    """
    value = str(text or "")
    if "@" in value or "://" in value:
        return value
    if not re.search(r"[\u3040-\u30ff\u3400-\u9fff]", value):
        return value
    return value.replace("(", "（").replace(")", "）")


def apply_exact_publication_corrections(
    doc: UnifiedDocument,
    corrections: Sequence[PublicationCorrection],
    *,
    copy_document: bool = True,
) -> tuple[UnifiedDocument, list[AppliedCorrection]]:
    """Apply a small audited list of exact hard-error corrections.

    Each ``before`` string must occur exactly once across text blocks.  This
    prevents a publication pass from becoming a global search/replace or style
    rewrite.  The audit ledger is returned and also stored in metadata when the
    model supports dynamic attributes.
    """
    out = doc.snapshot_clone() if copy_document else doc
    applied: list[AppliedCorrection] = []

    for corr in corrections:
        matches: list[Block] = []
        for block in out.blocks:
            if _is_text_block(block) and corr.before in (block.text or ""):
                matches.append(block)
        if len(matches) != 1:
            raise ValueError(
                f"publication correction must match exactly once: {corr.before!r}; matches={len(matches)}"
            )
        block = matches[0]
        block.text = (block.text or "").replace(corr.before, corr.after, 1)
        block.metadata = dict(block.metadata or {})
        block.metadata.setdefault("publication_proof_corrections", []).append(asdict(corr))
        applied.append(
            AppliedCorrection(
                block_id=block.id,
                page=int(block.page or 0),
                before=corr.before,
                after=corr.after,
                reason=corr.reason,
                source=corr.source,
            )
        )

    try:
        out.metadata.publication_proof_corrections = [asdict(x) for x in applied]
    except Exception:
        pass
    return out, applied


def merge_split_dialogues_across_images(
    doc: UnifiedDocument,
    *,
    copy_document: bool = True,
    max_intervening_images: int = 1,
) -> tuple[UnifiedDocument, list[dict]]:
    """Merge mechanically unambiguous dialogue continuations across an image.

    Printed books can place a full-page illustration in the middle of a sentence.
    For a reflowable EPUB, leaving the two quote fragments as separate paragraphs
    is semantically wrong.  When the first fragment has an unmatched Japanese
    opening quote and the first following text block closes that same quote, the
    text is merged.  Intervening image blocks are moved immediately after the
    completed dialogue so the illustration remains at the nearest paragraph
    boundary without being dropped or re-encoded.
    """
    out = doc.snapshot_clone() if copy_document else doc
    src = list(out.blocks)
    dst: list[Block] = []
    report: list[dict] = []
    i = 0

    while i < len(src):
        block = src[i]
        text = block.text or ""
        if _is_text_block(block) and text.count("「") > text.count("」"):
            j = i + 1
            images: list[Block] = []
            while j < len(src) and src[j].type == BlockType.IMAGE_REF and len(images) < max_intervening_images:
                images.append(src[j])
                j += 1
            if j < len(src) and images:
                nxt = src[j]
                nxt_text = nxt.text or ""
                combined = text + nxt_text
                if (
                    _is_text_block(nxt)
                    and nxt_text.count("」") > nxt_text.count("「")
                    and combined.count("「") == combined.count("」")
                ):
                    merged = copy.deepcopy(block)
                    merged.text = combined
                    merged.type = BlockType.DIALOGUE
                    merged.metadata = dict(merged.metadata or {})
                    merged.metadata["publication_merge_reason"] = "unambiguous_split_dialogue_across_image"
                    merged.metadata["publication_merged_block_ids"] = [block.id, nxt.id]
                    merged.metadata["publication_intervening_image_ids"] = [b.id for b in images]
                    dst.append(merged)
                    dst.extend(images)
                    report.append(
                        {
                            "block_ids": [block.id, nxt.id],
                            "pages": [int(block.page or 0), int(nxt.page or 0)],
                            "image_ids": [b.id for b in images],
                            "image_pages": [int(b.page or 0) for b in images],
                            "text": combined,
                        }
                    )
                    i = j + 1
                    continue
        dst.append(block)
        i += 1

    out.blocks = dst
    try:
        out.metadata.publication_cross_image_dialogue_merges = report
    except Exception:
        pass
    return out, report


def publication_lint(doc: UnifiedDocument) -> dict:
    """Return conservative structural warnings for a Japanese publication pass."""
    replacement = 0
    cjk_radicals: list[dict] = []
    controls: list[dict] = []
    unbalanced_dialogues: list[dict] = []
    unterminated_prose: list[dict] = []

    for block in doc.blocks:
        if not _is_text_block(block):
            continue
        text = block.text or ""
        replacement += text.count("\ufffd")
        for ch in text:
            cp = ord(ch)
            if 0x2E80 <= cp <= 0x2EFF:
                cjk_radicals.append({"block_id": block.id, "page": int(block.page or 0), "char": ch, "codepoint": f"U+{cp:04X}"})
            if unicodedata.category(ch) == "Cc" and ch not in "\n\r\t":
                controls.append({"block_id": block.id, "page": int(block.page or 0), "codepoint": f"U+{cp:04X}"})
        if text.count("「") != text.count("」"):
            unbalanced_dialogues.append({"block_id": block.id, "page": int(block.page or 0), "text": text})
        stripped = text.strip()
        if (
            block.type == BlockType.PARAGRAPH
            and stripped
            and not bool((block.metadata or {}).get("publication_non_sentence_field", False))
            and stripped not in {"◆", "◇", "＊", "＊＊＊", "×××", "【完】", "︻完︼"}
            # Deliberate suspense / continuation punctuation is valid prose and
            # should not be promoted to a hard-error lint item.
            and not re.search(r"(?:──|――|—|―)[、,]$", stripped)
            and not re.search(r"[。！？!?”」』）)]$", stripped)
        ):
            unterminated_prose.append({"block_id": block.id, "page": int(block.page or 0), "text": stripped})

    return {
        "replacement_glyphs": replacement,
        "cjk_radicals": cjk_radicals,
        "control_characters": controls,
        "unbalanced_dialogues": unbalanced_dialogues,
        "unterminated_prose": unterminated_prose,
        "passed": not (replacement or cjk_radicals or controls or unbalanced_dialogues or unterminated_prose),
    }
