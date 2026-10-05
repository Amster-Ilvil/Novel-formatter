# -*- coding: utf-8 -*-
"""Stable, UI-agnostic command descriptors for the desktop command palette.

The palette is intentionally only an index over existing actions.  It does not
own workflow behavior, project state or task scheduling.  Keeping ranking here
Qt-free makes the command surface cheap to test and safe to reuse from future
CLI/help surfaces.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


_WS_RE = re.compile(r"\s+")


def normalize_command_query(value: object) -> str:
    text = str(value or "").strip().casefold()
    return _WS_RE.sub(" ", text)


@dataclass(frozen=True, slots=True)
class CommandSpec:
    command_id: str
    title: str
    category: str = ""
    keywords: tuple[str, ...] = ()
    shortcut: str = ""
    priority: int = 100

    @property
    def search_text(self) -> str:
        return normalize_command_query(
            " ".join((self.title, self.category, *self.keywords, self.command_id))
        )


def _score(spec: CommandSpec, query: str) -> int | None:
    query = normalize_command_query(query)
    if not query:
        return 0
    haystack = spec.search_text
    tokens = tuple(token for token in query.split(" ") if token)
    if any(token not in haystack for token in tokens):
        return None
    title = normalize_command_query(spec.title)
    category = normalize_command_query(spec.category)
    score = 0
    if title == query:
        score += 500
    elif title.startswith(query):
        score += 300
    elif query in title:
        score += 220
    for token in tokens:
        if title.startswith(token):
            score += 80
        elif token in title:
            score += 55
        elif token in category:
            score += 25
        else:
            score += 10
    return score


def rank_command_specs(specs: Iterable[CommandSpec], query: object = "") -> tuple[CommandSpec, ...]:
    """Return deterministic search results without mutating the command catalog."""
    normalized = normalize_command_query(query)
    ranked: list[tuple[int, int, str, str, CommandSpec]] = []
    for spec in specs:
        score = _score(spec, normalized)
        if score is None:
            continue
        ranked.append((-score, int(spec.priority), spec.category, spec.title, spec))
    ranked.sort(key=lambda row: row[:4])
    return tuple(row[4] for row in ranked)


__all__ = ["CommandSpec", "normalize_command_query", "rank_command_specs"]
