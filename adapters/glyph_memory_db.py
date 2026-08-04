#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistent glyph-memory database for Japanese OCR correction.

The database never rewrites Apple's Vision model.  It stores normalized glyph
bitmaps that the user has explicitly confirmed and uses exact/near-image
matching to re-rank future OCR candidates.  Current-book samples are preferred;
global samples are only a fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
from typing import Iterable, Sequence

from PIL import Image, ImageOps

_DB_NAME = "glyph_memory.sqlite3"
_GLOBAL_SCOPE = "__global__"
_NORM_SIZE = 64


def _app_support_root() -> Path:
    if os.name == "posix" and Path.home().joinpath("Library").exists():
        return Path.home() / "Library" / "Application Support" / "NovelFormatter"
    return Path.home() / ".novel_formatter"


def default_db_path() -> Path:
    root = _app_support_root()
    root.mkdir(parents=True, exist_ok=True)
    return root / _DB_NAME


def review_cache_root() -> Path:
    if os.name == "posix" and Path.home().joinpath("Library").exists():
        root = Path.home() / "Library" / "Caches" / "NovelFormatter" / "glyph-review"
    else:
        root = Path.home() / ".cache" / "novel_formatter" / "glyph-review"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _page_content_fingerprint(path_value: str) -> str:
    """Cheap path-independent fingerprint for one source page.

    Book scope used to include mutable OCR metadata (especially title) and the
    parent directory name.  A second OCR pass could therefore create a new
    scope even for the same images.  Fingerprinting the page bytes keeps the
    scope stable after title recognition, file moves and temporary work-folder
    changes while avoiding a full-file hash for very large scans.
    """
    path = Path(str(path_value or ""))
    if not path.exists() or not path.is_file():
        return ""
    try:
        stat = path.stat()
        sample_size = 131072
        digest = hashlib.sha256()
        digest.update(str(int(stat.st_size)).encode("ascii"))
        with path.open("rb") as handle:
            digest.update(handle.read(sample_size))
            if stat.st_size > sample_size:
                handle.seek(max(0, stat.st_size - sample_size))
                digest.update(handle.read(sample_size))
        return digest.hexdigest()[:24]
    except Exception:
        return ""


def document_scope_key(doc) -> str:
    """Return a stable current-book scope and persist it on the document.

    The old key mixed title/author and path names.  Title is often populated
    only *after* the first OCR pass, so learned rows became invisible on the
    next pass.  Prefer a persisted key; otherwise derive it from source-page
    content.  Mutable bibliographic metadata is only a final fallback.
    """
    metadata = getattr(doc, "metadata", None)
    persisted = str(getattr(metadata, "glyph_memory_scope_key", "") or "").strip()
    if persisted.startswith("book:"):
        return persisted

    page_parts: list[str] = []
    pages = getattr(doc, "pages", []) or []
    for page in list(pages)[:3]:
        fingerprint = _page_content_fingerprint(str(getattr(page, "image_path", "") or ""))
        if fingerprint:
            page_parts.append(f"page-sha:{fingerprint}")

    parts = page_parts
    if not parts:
        for name in ("isbn", "author", "series", "volume", "title"):
            value = str(getattr(metadata, name, "") or "").strip()
            if value:
                parts.append(f"{name}:{value}")
    if not parts:
        parts.append("untitled")
    digest = hashlib.sha256("\n".join(parts).encode("utf-8", errors="ignore")).hexdigest()[:24]
    scope = f"book:{digest}"
    if metadata is not None:
        try:
            setattr(metadata, "glyph_memory_scope_key", scope)
        except Exception:
            pass
    return scope


def _pixels(image: Image.Image) -> list[int]:
    getter = getattr(image, "get_flattened_data", None)
    if callable(getter):
        return list(getter())
    return list(image.getdata())


def _otsu(values: Sequence[int]) -> int:
    hist = [0] * 256
    for value in values:
        hist[int(value)] += 1
    total = len(values)
    if total <= 0:
        return 180
    sum_all = sum(i * hist[i] for i in range(256))
    weight_b = 0
    sum_b = 0
    best = 127
    best_var = -1.0
    for threshold in range(256):
        weight_b += hist[threshold]
        if not weight_b:
            continue
        weight_f = total - weight_b
        if not weight_f:
            break
        sum_b += threshold * hist[threshold]
        mean_b = sum_b / weight_b
        mean_f = (sum_all - sum_b) / weight_f
        variance = weight_b * weight_f * (mean_b - mean_f) ** 2
        if variance > best_var:
            best_var = variance
            best = threshold
    return max(65, min(235, best + 8))


def normalize_glyph(image: Image.Image, *, size: int = _NORM_SIZE) -> Image.Image:
    """Normalize a printed glyph without changing its aspect ratio."""
    gray = ImageOps.grayscale(image)
    data = _pixels(gray)
    threshold = _otsu(data)
    binary = gray.point(lambda value: 0 if value < threshold else 255, mode="L")
    inverted = ImageOps.invert(binary)
    bbox = inverted.getbbox()
    if bbox:
        binary = binary.crop(bbox)
    else:
        binary = Image.new("L", (1, 1), 255)
    margin = max(3, int(round(size * 0.11)))
    target = max(1, size - margin * 2)
    scale = min(target / max(1, binary.width), target / max(1, binary.height))
    resized = binary.resize(
        (max(1, int(round(binary.width * scale))), max(1, int(round(binary.height * scale)))),
        Image.Resampling.NEAREST,
    )
    canvas = Image.new("L", (size, size), 255)
    x = (size - resized.width) // 2
    y = (size - resized.height) // 2
    canvas.paste(resized, (x, y))
    return canvas.point(lambda value: 0 if value < 128 else 255, mode="L")


def _bitmap_bytes(image: Image.Image) -> bytes:
    norm = normalize_glyph(image)
    return bytes(1 if value < 128 else 0 for value in _pixels(norm))


def _exact_hash(bitmap: bytes) -> str:
    return hashlib.sha256(bitmap).hexdigest()


def _dhash(image: Image.Image) -> str:
    gray = ImageOps.grayscale(image).resize((9, 8), Image.Resampling.BILINEAR)
    values = _pixels(gray)
    bits = 0
    index = 0
    for y in range(8):
        for x in range(8):
            if values[y * 9 + x] > values[y * 9 + x + 1]:
                bits |= 1 << index
            index += 1
    return f"{bits:016x}"


def _hamming_hex(left: str, right: str) -> int:
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except Exception:
        return 64


_DIACRITIC_SENSITIVE_KANA = set(
    "うゔかがきぎくぐけげこござじずぜぞさしすせそ"
    "ただちぢつづてでとどはばぱひびぴふぶぷへべぺほぼぽゝゞ"
    "ウヴカガキギクグケゲコゴサザシジスズセゼソゾ"
    "タダチヂツヅテデトドハバパヒビピフブプヘベペホボポヽヾ"
)


def _detached_diacritic_mark_count(bitmap: bytes, *, size: int = _NORM_SIZE) -> int:
    """Count small detached upper-right marks in a normalized kana bitmap.

    This is intentionally used only to veto *automatic* fuzzy-memory output.
    A dakuten/handakuten contributes very little to Dice/pixel similarity, so
    without this topology check a confirmed ``た`` or ``て`` can incorrectly
    replace printed ``だ`` or ``で``.  Candidates remain visible for review.
    """
    if not bitmap or len(bitmap) != size * size:
        return 0
    total_ink = int(sum(bitmap))
    if total_ink <= 0:
        return 0

    seen = bytearray(size * size)
    components: list[tuple[int, int, int, int, int]] = []
    for start, value in enumerate(bitmap):
        if not value or seen[start]:
            continue
        seen[start] = 1
        stack = [start]
        area = 0
        min_x = min_y = size
        max_x = max_y = 0
        while stack:
            index = stack.pop()
            y, x = divmod(index, size)
            area += 1
            min_x = min(min_x, x); max_x = max(max_x, x)
            min_y = min(min_y, y); max_y = max(max_y, y)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < size and 0 <= ny < size:
                        neighbour = ny * size + nx
                        if bitmap[neighbour] and not seen[neighbour]:
                            seen[neighbour] = 1
                            stack.append(neighbour)
        components.append((area, min_x, min_y, max_x, max_y))

    min_area = max(10, int(round(total_ink * 0.025)))
    max_area = max(min_area, int(round(total_ink * 0.145)))
    count = 0
    for area, min_x, min_y, max_x, max_y in components:
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        if not (min_area <= area <= max_area):
            continue
        if center_x < size * 0.59 or center_y > size * 0.41:
            continue
        if width > size * 0.30 or height > size * 0.27:
            continue
        count += 1
    return min(2, count)


def _pixel_difference(left: bytes, right: bytes) -> float:
    if not left or len(left) != len(right):
        return 1.0
    return sum(a != b for a, b in zip(left, right)) / len(left)


def _shifted_ink_similarity(
    left: bytes,
    right: bytes,
    *,
    size: int = _NORM_SIZE,
    max_shift: int = 2,
) -> float:
    """Best Dice overlap after a tiny translation of a normalized glyph."""
    if not left or len(left) != len(right) or len(left) != size * size:
        return 0.0
    denominator = sum(left) + sum(right)
    if denominator <= 0:
        return 0.0
    best = 0.0
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            intersection = 0
            for y in range(size):
                ry = y - dy
                if ry < 0 or ry >= size:
                    continue
                left_row = y * size
                right_row = ry * size
                for x in range(size):
                    rx = x - dx
                    if 0 <= rx < size and left[left_row + x] and right[right_row + rx]:
                        intersection += 1
            best = max(best, (2.0 * intersection) / denominator)
    return max(0.0, min(1.0, best))


@dataclass(slots=True)
class GlyphMemoryMatch:
    character: str
    score: float
    scope_key: str
    exact: bool
    pixel_difference: float
    hash_distance: int
    confirmations: int
    source: str = "glyph_memory"
    shape_similarity: float = 0.0
    topology_mismatch: bool = False


class GlyphMemoryDB:
    def __init__(self, path: str | os.PathLike[str] | None = None):
        self.path = Path(path) if path else default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=20)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS glyph_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope_key TEXT NOT NULL,
                    character TEXT NOT NULL,
                    exact_hash TEXT NOT NULL,
                    dhash TEXT NOT NULL,
                    bitmap BLOB NOT NULL,
                    ink_pixels INTEGER NOT NULL,
                    source TEXT NOT NULL DEFAULT 'manual_review',
                    confirmations INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(scope_key, character, exact_hash)
                );
                CREATE INDEX IF NOT EXISTS idx_glyph_scope_hash
                    ON glyph_samples(scope_key, exact_hash);
                CREATE INDEX IF NOT EXISTS idx_glyph_scope_dhash
                    ON glyph_samples(scope_key, dhash);
                CREATE INDEX IF NOT EXISTS idx_glyph_ink_pixels
                    ON glyph_samples(ink_pixels);
                CREATE TABLE IF NOT EXISTS correction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope_key TEXT NOT NULL,
                    previous_text TEXT,
                    confirmed_text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(glyph_samples)")}
            migrations = {
                "apple_result": "TEXT NOT NULL DEFAULT ''",
                "vision_candidates_json": "TEXT NOT NULL DEFAULT ''",
                "trace_json": "TEXT NOT NULL DEFAULT ''",
                "is_symbol": "INTEGER NOT NULL DEFAULT 0",
            }
            for name, definition in migrations.items():
                if name not in columns:
                    conn.execute(f"ALTER TABLE glyph_samples ADD COLUMN {name} {definition}")

    @staticmethod
    def global_scope() -> str:
        return _GLOBAL_SCOPE

    def add_sample(
        self,
        image: Image.Image,
        character: str,
        *,
        scope_key: str,
        source: str = "manual_review",
        also_global: bool = False,
        apple_result: str = "",
        vision_candidates_json: str = "",
        trace_json: str = "",
        is_symbol: bool = False,
    ) -> None:
        character = str(character or "").strip()
        # A database row must always represent exactly one segmented glyph.
        # Refuse multi-character labels so an accidental "one box, two chars"
        # correction can never poison exact-hash lookup.
        if not character or character == "□" or len(character) != 1:
            return
        bitmap = _bitmap_bytes(image)
        exact = _exact_hash(bitmap)
        dhash = _dhash(normalize_glyph(image))
        ink = sum(bitmap)
        now = datetime.now(timezone.utc).isoformat()
        scopes = [str(scope_key or _GLOBAL_SCOPE)]
        if also_global and _GLOBAL_SCOPE not in scopes:
            scopes.append(_GLOBAL_SCOPE)
        with self._connect() as conn:
            for scope in scopes:
                # An exact bitmap is authoritative after explicit confirmation.
                # Remove stale/conflicting labels for the same bitmap so future
                # exact lookup has a single deterministic answer.
                conn.execute(
                    "DELETE FROM glyph_samples WHERE scope_key=? AND exact_hash=? AND character<>?",
                    (scope, exact, character),
                )
                conn.execute(
                    """
                    INSERT INTO glyph_samples
                        (scope_key, character, exact_hash, dhash, bitmap, ink_pixels,
                         source, confirmations, created_at, updated_at, apple_result,
                         vision_candidates_json, trace_json, is_symbol)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(scope_key, character, exact_hash) DO UPDATE SET
                        confirmations = confirmations + 1,
                        source = excluded.source,
                        apple_result = CASE WHEN excluded.apple_result <> '' THEN excluded.apple_result ELSE glyph_samples.apple_result END,
                        vision_candidates_json = CASE WHEN excluded.vision_candidates_json <> '' THEN excluded.vision_candidates_json ELSE glyph_samples.vision_candidates_json END,
                        trace_json = CASE WHEN excluded.trace_json <> '' THEN excluded.trace_json ELSE glyph_samples.trace_json END,
                        is_symbol = MAX(glyph_samples.is_symbol, excluded.is_symbol),
                        updated_at = excluded.updated_at
                    """,
                    (scope, character, exact, dhash, bitmap, ink, source, now, now,
                     str(apple_result or ""), str(vision_candidates_json or ""),
                     str(trace_json or ""), 1 if is_symbol else 0),
                )

    def lookup(
        self,
        image: Image.Image,
        *,
        scope_key: str,
        include_global: bool = True,
        limit: int = 8,
        max_hash_distance: int = 13,
        max_pixel_difference: float = 0.20,
    ) -> list[GlyphMemoryMatch]:
        bitmap = _bitmap_bytes(image)
        exact = _exact_hash(bitmap)
        dhash = _dhash(normalize_glyph(image))
        query_mark_count = _detached_diacritic_mark_count(bitmap)
        scopes = [str(scope_key or _GLOBAL_SCOPE)]
        if include_global and _GLOBAL_SCOPE not in scopes:
            scopes.append(_GLOBAL_SCOPE)
        placeholders = ",".join("?" for _ in scopes)
        with self._connect() as conn:
            exact_rows = conn.execute(
                f"SELECT * FROM glyph_samples WHERE scope_key IN ({placeholders}) AND exact_hash=?",
                (*scopes, exact),
            ).fetchall()
            exact_scope_fallback = False
            if not exact_rows:
                # A user-confirmed exact bitmap remains authoritative even when
                # the same book is reopened from another path or its generated
                # document scope changes.  Search every scope only for an exact
                # normalized bitmap; this does not broaden fuzzy matching.
                exact_rows = conn.execute(
                    "SELECT * FROM glyph_samples WHERE exact_hash=?",
                    (exact,),
                ).fetchall()
                # Do not short-circuit if different books explicitly assigned
                # different characters to the same normalized bitmap.
                if len({str(row["character"]) for row in exact_rows}) > 1:
                    exact_rows = []
                exact_scope_fallback = bool(exact_rows)
            if exact_rows:
                best: dict[str, GlyphMemoryMatch] = {}
                for row in exact_rows:
                    scope_bonus = 0.006 if row["scope_key"] == scopes[0] else 0.0
                    base_score = 0.993 if exact_scope_fallback else 0.994
                    match = GlyphMemoryMatch(
                        character=row["character"], score=min(1.0, base_score + scope_bonus),
                        scope_key=row["scope_key"], exact=True, pixel_difference=0.0,
                        hash_distance=0, confirmations=int(row["confirmations"] or 1),
                        shape_similarity=1.0,
                    )
                    previous = best.get(match.character)
                    if previous is None or (match.score, match.confirmations) > (previous.score, previous.confirmations):
                        best[match.character] = match
                return sorted(best.values(), key=lambda item: (-item.score, -item.confirmations))[:limit]

            rows = conn.execute(
                f"SELECT * FROM glyph_samples WHERE scope_key IN ({placeholders})",
                tuple(scopes),
            ).fetchall()
        ranked: dict[str, GlyphMemoryMatch] = {}
        for row in rows:
            distance = _hamming_hex(dhash, row["dhash"])
            if distance > max_hash_distance:
                continue
            stored = bytes(row["bitmap"])
            difference = _pixel_difference(bitmap, stored)
            shape_similarity = _shifted_ink_similarity(bitmap, stored)
            if difference > max_pixel_difference and shape_similarity < 0.72:
                continue
            scope_bonus = 0.025 if row["scope_key"] == scopes[0] else 0.0
            confirmation_bonus = min(0.025, max(0, int(row["confirmations"] or 1) - 1) * 0.004)
            hash_similarity = max(0.0, 1.0 - distance / 64.0)
            background_similarity = max(0.0, 1.0 - min(1.0, difference))
            topology_mismatch = False
            if str(row["character"] or "") in _DIACRITIC_SENSITIVE_KANA:
                topology_mismatch = (
                    query_mark_count != _detached_diacritic_mark_count(stored)
                )
            score = (
                0.70 * shape_similarity
                + 0.20 * hash_similarity
                + 0.10 * background_similarity
                + scope_bonus
                + confirmation_bonus
                - (0.18 if topology_mismatch else 0.0)
            )
            match = GlyphMemoryMatch(
                character=row["character"], score=max(0.0, min(0.989, score)),
                scope_key=row["scope_key"], exact=False,
                pixel_difference=difference, hash_distance=distance,
                confirmations=int(row["confirmations"] or 1),
                shape_similarity=shape_similarity,
                topology_mismatch=topology_mismatch,
            )
            previous = ranked.get(match.character)
            if previous is None or match.score > previous.score:
                ranked[match.character] = match
        # Scope keys legitimately change in older databases because the legacy
        # scope included title/path data.  Always merge a *strict* near-shape
        # search across every scope.  Previously this ran only when current-book
        # and global lookup returned nothing; one weak global look-alike could
        # therefore hide the real 0.90+ match stored under the old book scope.
        ink_total = sum(bitmap)
        ink_tolerance = max(10, int(round(max(1, ink_total) * 0.20)))
        with self._connect() as conn:
            fallback_rows = conn.execute(
                "SELECT * FROM glyph_samples WHERE ink_pixels BETWEEN ? AND ?",
                (max(0, ink_total - ink_tolerance), ink_total + ink_tolerance),
            ).fetchall()
        for row in fallback_rows:
            distance = _hamming_hex(dhash, row["dhash"])
            if distance > min(10, max_hash_distance):
                continue
            stored = bytes(row["bitmap"])
            difference = _pixel_difference(bitmap, stored)
            shape_similarity = _shifted_ink_similarity(bitmap, stored)
            # Cross-scope matching is deliberately stricter than ordinary
            # current-book matching.  It is strong enough for the same printed
            # font with a slightly shifted crop, but rejects broad shape guesses.
            if difference > min(0.18, max_pixel_difference) and shape_similarity < 0.82:
                continue
            confirmation_bonus = min(0.025, max(0, int(row["confirmations"] or 1) - 1) * 0.004)
            hash_similarity = max(0.0, 1.0 - distance / 64.0)
            background_similarity = max(0.0, 1.0 - min(1.0, difference))
            legacy_scope_bonus = 0.025 if row["scope_key"] not in scopes else 0.0
            topology_mismatch = False
            if str(row["character"] or "") in _DIACRITIC_SENSITIVE_KANA:
                topology_mismatch = (
                    query_mark_count != _detached_diacritic_mark_count(stored)
                )
            score = (
                0.73 * shape_similarity
                + 0.18 * hash_similarity
                + 0.09 * background_similarity
                + confirmation_bonus
                + legacy_scope_bonus
                - (0.18 if topology_mismatch else 0.0)
            )
            match = GlyphMemoryMatch(
                character=row["character"], score=max(0.0, min(0.975, score)),
                scope_key=row["scope_key"], exact=False,
                pixel_difference=difference, hash_distance=distance,
                confirmations=int(row["confirmations"] or 1),
                source="glyph_memory_cross_scope",
                shape_similarity=shape_similarity,
                topology_mismatch=topology_mismatch,
            )
            previous = ranked.get(match.character)
            if previous is None or (match.score, match.confirmations) > (previous.score, previous.confirmations):
                ranked[match.character] = match
        return sorted(ranked.values(), key=lambda item: (-item.score, item.pixel_difference, item.hash_distance))[:limit]

    def record_correction(self, *, scope_key: str, previous_text: str, confirmed_text: str, source: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO correction_history(scope_key, previous_text, confirmed_text, source, created_at) VALUES (?, ?, ?, ?, ?)",
                (scope_key, previous_text, confirmed_text, source, now),
            )

    def stats(self, *, scope_key: str | None = None) -> dict:
        with self._connect() as conn:
            if scope_key:
                row = conn.execute(
                    "SELECT COUNT(*) AS samples, COUNT(DISTINCT character) AS characters, COALESCE(SUM(confirmations),0) AS confirmations FROM glyph_samples WHERE scope_key=?",
                    (scope_key,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) AS samples, COUNT(DISTINCT character) AS characters, COALESCE(SUM(confirmations),0) AS confirmations FROM glyph_samples"
                ).fetchone()
            corrections = conn.execute("SELECT COUNT(*) FROM correction_history").fetchone()[0]
        return {
            "samples": int(row["samples"] or 0),
            "characters": int(row["characters"] or 0),
            "confirmations": int(row["confirmations"] or 0),
            "corrections": int(corrections or 0),
            "path": str(self.path),
        }

    def backup_to(self, destination: str | os.PathLike[str]) -> Path:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as source, sqlite3.connect(str(target)) as dest:
            source.backup(dest)
        return target

    def export_rows(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, scope_key, character, exact_hash, dhash, ink_pixels, source, confirmations, "
                "created_at, updated_at, apple_result, vision_candidates_json, trace_json, is_symbol "
                "FROM glyph_samples ORDER BY scope_key, character, id"
            ).fetchall()
        return [dict(row) for row in rows]

    def merge_from(self, source_path: str | os.PathLike[str]) -> int:
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        merged = 0
        source = sqlite3.connect(str(source_path))
        source.row_factory = sqlite3.Row
        try:
            columns = {row[1] for row in source.execute("PRAGMA table_info(glyph_samples)")}
            if not columns:
                raise ValueError("来源数据库没有 glyph_samples 表")
            rows = source.execute("SELECT * FROM glyph_samples").fetchall()
            with self._connect() as dest:
                for row in rows:
                    data = dict(row)
                    now = datetime.now(timezone.utc).isoformat()
                    dest.execute(
                        """
                        INSERT INTO glyph_samples
                            (scope_key, character, exact_hash, dhash, bitmap, ink_pixels, source,
                             confirmations, created_at, updated_at, apple_result,
                             vision_candidates_json, trace_json, is_symbol)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(scope_key, character, exact_hash) DO UPDATE SET
                            confirmations = MAX(glyph_samples.confirmations, excluded.confirmations),
                            source = excluded.source, updated_at = excluded.updated_at,
                            apple_result = CASE WHEN excluded.apple_result <> '' THEN excluded.apple_result ELSE glyph_samples.apple_result END,
                            vision_candidates_json = CASE WHEN excluded.vision_candidates_json <> '' THEN excluded.vision_candidates_json ELSE glyph_samples.vision_candidates_json END,
                            trace_json = CASE WHEN excluded.trace_json <> '' THEN excluded.trace_json ELSE glyph_samples.trace_json END,
                            is_symbol = MAX(glyph_samples.is_symbol, excluded.is_symbol)
                        """,
                        (
                            str(data.get("scope_key") or _GLOBAL_SCOPE), str(data.get("character") or ""),
                            str(data.get("exact_hash") or ""), str(data.get("dhash") or ""),
                            bytes(data.get("bitmap") or b""), int(data.get("ink_pixels") or 0),
                            str(data.get("source") or "database_import"), int(data.get("confirmations") or 1),
                            str(data.get("created_at") or now), now, str(data.get("apple_result") or ""),
                            str(data.get("vision_candidates_json") or ""), str(data.get("trace_json") or ""),
                            int(data.get("is_symbol") or 0),
                        ),
                    )
                    merged += 1
        finally:
            source.close()
        return merged

    def clear_scope(self, scope_key: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM glyph_samples WHERE scope_key=?", (scope_key,))
            return int(cursor.rowcount or 0)


def stage_review_glyphs(block_id: str, glyph_images: Sequence[Image.Image]) -> list[str]:
    target = review_cache_root() / str(block_id)
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for index, image in enumerate(glyph_images):
        path = target / f"glyph_{index:04d}.png"
        (image if image.mode == "RGB" else image.convert("RGB")).save(path, format="PNG", compress_level=0)
        paths.append(str(path))
    return paths


def learn_reviewed_column(
    *,
    glyph_paths: Sequence[str],
    confirmed_text: str,
    previous_text: str,
    scope_key: str,
    also_global: bool = False,
    skip_indices: Iterable[int] | None = None,
    db: GlyphMemoryDB | None = None,
) -> int:
    """Learn only when character count still aligns with segmented glyph count."""
    chars = list(str(confirmed_text or ""))
    if len(chars) != len(glyph_paths):
        return 0
    skipped = {int(index) for index in (skip_indices or [])}
    database = db or GlyphMemoryDB()
    learned = 0
    for index, (path, character) in enumerate(zip(glyph_paths, chars)):
        if index in skipped:
            continue
        if character == "□" or not str(character).strip():
            continue
        try:
            with Image.open(path) as image:
                database.add_sample(
                    image.convert("RGB"), character, scope_key=scope_key,
                    source="manual_review", also_global=also_global,
                )
            learned += 1
        except Exception:
            continue
    if learned:
        database.record_correction(
            scope_key=scope_key,
            previous_text=str(previous_text or ""),
            confirmed_text=str(confirmed_text or ""),
            source="handwriting_trace_review",
        )
    return learned


def learn_reviewed_glyphs(
    *,
    glyph_paths: Sequence[str],
    confirmations: Iterable[dict],
    previous_text: str,
    confirmed_text: str,
    scope_key: str,
    also_global: bool = False,
    skip_indices: Iterable[int] | None = None,
    db: GlyphMemoryDB | None = None,
) -> dict:
    """Learn explicitly confirmed box→character pairs.

    Unlike :func:`learn_reviewed_column`, this path does not require the edited
    column text length to match the segmented box count.  It is therefore safe
    for per-box review and IME input.  Suspicious oversized boxes can be passed
    through ``skip_indices`` and are never stored.
    """
    database = db or GlyphMemoryDB()
    skipped = {int(index) for index in (skip_indices or [])}
    learned = 0
    ignored = 0
    invalid = 0
    seen: set[int] = set()

    for item in confirmations or []:
        try:
            index = int(item.get("index"))
        except Exception:
            invalid += 1
            continue
        if index in seen:
            continue
        seen.add(index)
        if index < 0 or index >= len(glyph_paths):
            invalid += 1
            continue
        if index in skipped:
            ignored += 1
            continue
        character = str(item.get("character") or "").strip()
        if not character or character == "□" or len(character) != 1:
            invalid += 1
            continue
        try:
            with Image.open(glyph_paths[index]) as image:
                database.add_sample(
                    image.convert("RGB"), character, scope_key=scope_key,
                    source="manual_glyph_review", also_global=also_global,
                )
            learned += 1
        except Exception:
            invalid += 1

    if learned:
        database.record_correction(
            scope_key=scope_key,
            previous_text=str(previous_text or ""),
            confirmed_text=str(confirmed_text or ""),
            source="handwriting_trace_glyph_review",
        )
    return {"learned": learned, "skipped": ignored, "invalid": invalid}


def cleanup_staged_glyphs(glyph_paths: Iterable[str]) -> None:
    parents = set()
    for item in glyph_paths:
        try:
            parents.add(Path(item).resolve().parent)
        except Exception:
            pass
    for parent in parents:
        try:
            if review_cache_root() in parent.parents or parent == review_cache_root():
                shutil.rmtree(parent, ignore_errors=True)
        except Exception:
            pass
