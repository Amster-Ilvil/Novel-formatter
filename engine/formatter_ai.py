# -*- coding: utf-8 -*-
"""Formatter workspace specific AI prompts.

These prompts are deliberately independent from the Text Replacement workspace.
The Formatter first relies on deterministic local rules for short/simple fragments;
AI is used only for long or structurally ambiguous passages.
"""
from __future__ import annotations

FORMATTER_RULE_CHECKLIST = """
1. Keep chapter/section order, facts, names, viewpoint, tone and all source content.
2. Join OCR line/column breaks only when the Japanese sentence clearly continues.
3. Keep every dialogue turn in its own block and separate dialogue from narration.
4. Repair split dialogue across blocks only when the next block is an unmistakable continuation; never swallow narration into quotes.
5. Do not add a closing quote merely because a short line begins with 「 or 『; place it only at the real end of the utterance.
6. Split a long block when it contains multiple dialogue turns or dialogue plus narration.
7. Merge long prose fragments that are one sentence across page/column boundaries.
8. Remove demonstrable accidental OCR/overlap/batch duplicates, including a whole consecutive passage repeated twice; preserve intentional rhetorical repetition and repeated dialogue.
9. Preserve ruby notation, chapter titles, section breaks, image anchors and block order.
10. Do not invent, summarize, translate, embellish or delete text.
""".strip()

FORMATTER_AI_LAYOUT_PROMPT = f"""Typeset Japanese light-novel OCR using the Formatter workspace rules below.
Local deterministic Formatter rules already handle short and simple fragments. Concentrate on long blocks, multi-sentence blocks, multiple dialogue turns inside one block, dialogue mixed with narration, and difficult cross-block/page continuations. Leave clean short paragraphs and clean standalone dialogue unchanged.
Do layout/structure work only. Preserve the original wording and characters except for moving existing punctuation needed to restore block boundaries.

FORMATTER RULES:
{FORMATTER_RULE_CHECKLIST}

Input is one ordered chapter part: {{"b":[[id,type,text],...],"g":0|1}}. Types: p=paragraph,d=dialogue,c=chapter,s=section,r=ruby,f=footnote,t=toc,u=other.
Return ONLY changed contiguous ranges as compact JSON: {{"o":[[[source_ids...],[[type,text],...]],...]}}. Each operation replaces exactly those contiguous input ids; operations must not overlap. Omit unchanged ranges. Ignore g and never return CSS.
If nothing changes return {{"o":[]}}. Return no explanations or Markdown. Every replacement text must be non-empty.
INPUT:\n{{{{INPUT}}}}"""

FORMATTER_AI_CORRECTION_PROMPT = f"""Proofread Japanese light-novel OCR for the Formatter workspace.
Correct only clear OCR character mistakes, unmistakably missing characters, broken punctuation and obvious grammar caused by OCR. Preserve wording, style, facts, block order, block count and block type. Do not merge, split, reformat, summarize or delete.
Use the checklist as context, but this mode performs correction only:
{FORMATTER_RULE_CHECKLIST}

Input: {{"b":[[id,type,text],...]}}.
Return ONLY changed blocks as compact JSON: {{"c":[[id,corrected_text],...]}}. Omit unchanged blocks. If nothing changes return {{"c":[]}}.
No explanations, unchanged text or Markdown.
INPUT:\n{{{{INPUT}}}}"""

FORMATTER_AI_CORRECTION_LAYOUT_PROMPT = f"""Correct and typeset OCR into a readable Japanese light-novel text.
This is the second, structural pass after a separate character-correction pass. Repair only boundaries, paragraphing, dialogue separation, and cross-line/cross-page joins. Never repeat context, never append a previous batch, and never duplicate any source range. Leave clean blocks unchanged.

FORMATTER RULES:
{FORMATTER_RULE_CHECKLIST}

Input is one ordered chapter part: {{"b":[[id,type,text],...],"g":0|1}}. Types: p=paragraph,d=dialogue,c=chapter,s=section,r=ruby,f=footnote,t=toc,u=other.
Repair structure and any remaining unmistakable OCR error. Return ONLY changed contiguous ranges as compact JSON: {{"o":[[[source_ids...],[[type,text],...]],...]}}. Each operation replaces exactly those contiguous input ids; operations must not overlap. Omit unchanged ranges. Ignore g and never return CSS.
Never invent, summarize, translate, rename or delete content. If nothing changes return {{"o":[]}}. Return no explanations or Markdown. Every replacement text must be non-empty.
INPUT:\n{{{{INPUT}}}}"""


def formatter_ai_prompt(mode: str) -> tuple[str, str]:
    """Return ``(processor_mode, prompt)`` for a Formatter AI action."""
    normalized = str(mode or "layout").strip().lower()
    if normalized == "correction":
        return "correction", FORMATTER_AI_CORRECTION_PROMPT
    if normalized in {"correction_layout", "correct_layout", "both"}:
        return "typeset", FORMATTER_AI_CORRECTION_LAYOUT_PROMPT
    return "typeset", FORMATTER_AI_LAYOUT_PROMPT
