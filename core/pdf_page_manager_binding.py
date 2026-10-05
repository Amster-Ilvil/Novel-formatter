#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only Page Manager -> PDF text-layer source/type binding.

The Page Manager is the authority for *explicitly confirmed* page types.  This
module intentionally contains no Qt code so the lineage/matching policy is easy
to regression-test in cloud/headless environments.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Mapping


def _safe_resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    try:
        return path.resolve()
    except Exception:
        return path


def _same_single_pdf_source(candidate: Path, requested: Path) -> bool:
    """Match the one managed PDF to a user-selected source conservatively.

    Project import copies the original into ``source/original`` while preserving
    basename and bytes.  Exact path/samefile is preferred; basename+size is only
    accepted because the caller already proved there is exactly one PDF source.
    """
    try:
        if candidate == requested.resolve():
            return True
    except Exception:
        pass
    try:
        if candidate.samefile(requested):
            return True
    except Exception:
        pass
    try:
        return candidate.name == requested.name and candidate.stat().st_size == requested.stat().st_size
    except Exception:
        return False


def resolve_pdf_page_manager_context(
    *,
    pdf_path: str | Path | None,
    raw_inputs: Iterable[str | Path],
    managed_page_count: int,
    page_overrides: Mapping[int | str, str] | None,
    auto_suggested: Iterable[int] | None,
    original_pdf_sources: Iterable[str | Path] | None = None,
    pdf_physical_page_map: Mapping[int | str, int | str] | None = None,
    page_count_func: Callable[[str], int] | None = None,
) -> dict:
    """Return the safe current Page Manager PDF source and physical-page overrides.

    Binding is deliberately disabled if the Page Manager no longer has a simple
    1:1 physical-page lineage (deleted pages, scan splitting, multiple PDFs, or
    image-only inputs).  False negatives are preferable to silently applying a
    cover/illustration label to the wrong PDF physical page.
    """
    pdf_sources: list[Path] = []
    durable_sources = list(original_pdf_sources or [])
    source_values = durable_sources if durable_sources else list(raw_inputs or [])
    for raw in source_values:
        path = _safe_resolve(raw)
        if path.suffix.lower() == ".pdf" and path.exists():
            pdf_sources.append(path)

    if len(pdf_sources) != 1:
        reason = (
            "页面管理当前不是原始 PDF 来源（可能已删除/预处理为图片）"
            if not pdf_sources
            else "页面管理当前包含多个 PDF；为避免页码错绑，不自动套用标记"
        )
        return {"matched": False, "reason": reason, "page_overrides": {}}

    source_pdf = pdf_sources[0]
    if pdf_path:
        requested = _safe_resolve(pdf_path)
        if not _same_single_pdf_source(source_pdf, requested):
            return {
                "matched": False,
                "source_pdf": str(source_pdf),
                "reason": "所选 PDF 与页面管理当前 PDF 不是同一来源",
                "page_overrides": {},
            }

    if page_count_func is None:
        from adapters.pdf_input import pdf_page_count
        page_count_func = pdf_page_count
    try:
        physical_pages = int(page_count_func(str(source_pdf)))
    except Exception as exc:
        return {
            "matched": False,
            "source_pdf": str(source_pdf),
            "reason": f"无法读取页面管理 PDF 页数：{exc}",
            "page_overrides": {},
        }

    managed_page_count = int(managed_page_count or 0)
    auto = {int(v) for v in (auto_suggested or [])}
    logical_to_physical = {
        int(k): int(v) for k, v in dict(pdf_physical_page_map or {}).items()
        if int(k) > 0 and 1 <= int(v) <= physical_pages
    }

    if logical_to_physical:
        # Persisted provenance is authoritative after preprocessing/deletion.
        # Missing physical pages are explicit deletions and therefore must be
        # skipped by PDF text extraction rather than silently reappearing.
        if any(page_no > managed_page_count for page_no in logical_to_physical):
            return {
                "matched": False, "source_pdf": str(source_pdf), "page_count": physical_pages,
                "reason": "页面管理 PDF 物理页映射与当前逻辑页数量不一致", "page_overrides": {},
            }
        confirmed: dict[int, str] = {}
        conflicts: set[int] = set()
        for logical_page, page_type in dict(page_overrides or {}).items():
            logical_page = int(logical_page)
            if logical_page in auto or logical_page not in logical_to_physical:
                continue
            physical_page = logical_to_physical[logical_page]
            value = str(page_type)
            previous = confirmed.get(physical_page)
            if previous is not None and previous != value:
                conflicts.add(physical_page)
            else:
                confirmed[physical_page] = value
        if conflicts:
            return {
                "matched": False, "source_pdf": str(source_pdf), "page_count": physical_pages,
                "reason": "同一 PDF 物理页被拆成多个逻辑页且页类型互相冲突：" + "、".join(map(str, sorted(conflicts))),
                "page_overrides": {},
            }
        present_physical = set(logical_to_physical.values())
        deleted_physical = [page_no for page_no in range(1, physical_pages + 1) if page_no not in present_physical]
        for page_no in deleted_physical:
            confirmed.setdefault(page_no, "deleted")
    else:
        if managed_page_count != physical_pages:
            return {
                "matched": False,
                "source_pdf": str(source_pdf),
                "page_count": physical_pages,
                "reason": (
                    f"页面管理现有 {managed_page_count} 页，但原 PDF 有 {physical_pages} 页；"
                    "没有持久物理页映射，为避免错绑已停用自动同步"
                ),
                "page_overrides": {},
            }
        confirmed = {
            int(page_no): str(page_type)
            for page_no, page_type in dict(page_overrides or {}).items()
            if int(page_no) not in auto and 1 <= int(page_no) <= physical_pages
        }
    return {
        "matched": True,
        "source_pdf": str(source_pdf),
        "page_count": physical_pages,
        "confirmed_count": len(confirmed),
        "skip_count": sum(1 for value in confirmed.values() if value != "paragraph"),
        "page_overrides": confirmed,
        "pdf_physical_page_map": logical_to_physical,
        "deleted_physical_pages": sorted(
            page_no for page_no, page_type in confirmed.items() if str(page_type) == "deleted"
        ),
        "reason": "",
    }
