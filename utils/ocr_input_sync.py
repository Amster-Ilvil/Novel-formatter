from __future__ import annotations

from pathlib import Path
from typing import Iterable


def resolve_ocr_run_inputs(
    pending_inputs: Iterable[str],
    *,
    input_origin: str = "direct",
    page_manager_images: Iterable[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Resolve the immutable input list handed to an OCR run.

    When the input originated in Page Manager, its current explicit page list is
    authoritative.  This prevents a deleted page from being reintroduced by
    re-expanding the original folder or PDF.  Direct OCR-tab selections remain
    independent from any unrelated Page Manager session.
    """
    candidates = list(pending_inputs or [])
    if str(input_origin or "direct") == "page_manager" and page_manager_images is not None:
        candidates = list(page_manager_images or [])

    valid: list[str] = []
    missing: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        value = str(raw or "").strip()
        if not value:
            continue
        try:
            normalized = str(Path(value).expanduser().resolve())
        except Exception:
            normalized = value
        if normalized in seen:
            continue
        seen.add(normalized)
        try:
            exists = Path(normalized).exists()
        except Exception:
            exists = False
        if exists:
            valid.append(normalized)
        else:
            missing.append(normalized)
    return valid, missing
