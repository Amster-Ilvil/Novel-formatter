from __future__ import annotations

import copy
from typing import Sequence


def stage_auto_select(comparison, row_texts: Sequence[Sequence[str]], labels: Sequence[str]):
    """Return an auto-selected comparison/fusion snapshot without mutating inputs.

    ``row_texts`` is captured by the UI from its authoritative full-text cache
    before entering this pure engine function.  The algorithm intentionally uses
    the exact same production helpers as the historical ``_auto_select_all``
    action so Preview and Apply share one staged object instead of recomputing.
    """
    from engine.multi_ocr_compare import choose_best_text, refresh_row_character_fusion
    from engine.ocr_compare_view_model import build_fusion_states

    staged_comparison = copy.deepcopy(comparison)
    for row_index, row in enumerate(staged_comparison.rows):
        texts = [str(value or "") for value in (row_texts[row_index] if row_index < len(row_texts) else row.texts)]
        choice, confidence, reason, warnings = choose_best_text(texts)
        row.texts[:len(texts)] = texts
        row.chosen_index = choice
        row.confidence = confidence
        row.reason = reason
        row.warnings = warnings
        refresh_row_character_fusion(row, labels)

    staged_states = build_fusion_states(staged_comparison.rows, auto_choose=True)
    return staged_comparison, staged_states
