# -*- coding: utf-8 -*-
"""Declarative contracts for the independent-but-linked desktop workspaces.

The GUI historically wired every panel directly to every other panel.  These
small immutable specifications make the allowed data flow explicit without
changing any OCR, Formatter or EPUB algorithm.  They are intentionally free of
Qt imports so command-line tools and contract tests can use them as well.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class WorkspaceSpec:
    key: str
    section: str
    consumes: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    heavy_when_hidden: bool = False


WORKSPACE_SPECS: tuple[WorkspaceSpec, ...] = (
    WorkspaceSpec("pages", "page", produces=("page_context",), heavy_when_hidden=True),
    WorkspaceSpec("ocr", "ocr", consumes=("page_context",), produces=("ocr_document", "multi_ocr_session"), heavy_when_hidden=True),
    WorkspaceSpec("pdf_text", "ocr", consumes=("page_context",), produces=("ocr_document",)),
    WorkspaceSpec("formatter", "format", consumes=("ocr_document", "reviewed_document"), produces=("formatted_document",)),
    WorkspaceSpec("text_compare", "proof", consumes=("document_versions", "page_context"), produces=("reviewed_document",), heavy_when_hidden=True),
    WorkspaceSpec("replacement", "proof", consumes=("ocr_document", "formatted_document", "reviewed_document"), produces=("reviewed_document",)),
    WorkspaceSpec("ocr_compare", "proof", consumes=("multi_ocr_session",), produces=("reviewed_document", "stable_row"), heavy_when_hidden=True),
    WorkspaceSpec("image_review", "proof", consumes=("ocr_document", "stable_row"), produces=("reviewed_document", "stable_row"), heavy_when_hidden=True),
    WorkspaceSpec("epub", "epub", consumes=("reviewed_document", "formatted_document", "page_context"), produces=("epub",), heavy_when_hidden=True),
    WorkspaceSpec("system", "system", produces=("preferences",)),
)


def workspace_spec_map(specs: Iterable[WorkspaceSpec] = WORKSPACE_SPECS) -> Mapping[str, WorkspaceSpec]:
    values = tuple(specs)
    mapping = {spec.key: spec for spec in values}
    if len(mapping) != len(values):
        raise ValueError("workspace keys must be unique")
    return mapping


def validate_workspace_contracts(specs: Iterable[WorkspaceSpec] = WORKSPACE_SPECS) -> tuple[str, ...]:
    """Return contract errors instead of raising during normal application use."""
    values = tuple(specs)
    errors: list[str] = []
    keys = [spec.key for spec in values]
    if len(keys) != len(set(keys)):
        errors.append("duplicate workspace key")
    producers: dict[str, set[str]] = {}
    for spec in values:
        if not spec.key or not spec.section:
            errors.append("workspace key/section cannot be empty")
        for topic in spec.produces:
            producers.setdefault(topic, set()).add(spec.key)
    external_topics = {"document_versions"}
    for spec in values:
        for topic in spec.consumes:
            if topic not in producers and topic not in external_topics:
                errors.append(f"{spec.key} consumes unproduced topic {topic}")
    return tuple(errors)
