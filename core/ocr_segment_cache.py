from __future__ import annotations

"""Small content-addressed cache for individual OCR crop/sentence results."""

import hashlib
import json
import math
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Mapping

CACHE_SCHEMA = "novel-formatter-ocr-segment-cache-v1"
SQLITE_SCHEMA_VERSION = 1
_LOCKS_GUARD = threading.Lock()
_LOCKS: dict[str, threading.Lock] = {}


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_fingerprint(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_key(*, image_sha256: str, engine: str, operation: str, runtime_id: str) -> str:
    return stable_fingerprint({
        "schema": CACHE_SCHEMA,
        "image_sha256": str(image_sha256 or ""),
        "engine": str(engine or ""),
        "operation": str(operation or ""),
        "runtime_id": str(runtime_id or ""),
    })


def _lock_for(path: Path) -> threading.Lock:
    key = str(path)
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.Lock())


class OcrSegmentCache:
    def __init__(self, root: str | os.PathLike[str], *, engine: str, runtime_id: str):
        self.root = Path(root)
        self.engine = str(engine or "")
        self.runtime_id = str(runtime_id or "")
        self.enabled = bool(self.engine and self.runtime_id and str(root or ""))
        self._db_path = self.root / "segment_cache.sqlite3"
        self._db_lock = threading.RLock()
        self._db: sqlite3.Connection | None = None
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)
            self._open_db()
        # ``mtime+size`` alone is insufficient: copy/restore tools can preserve
        # timestamps while replacing bytes.  Include ctime/inode so same-path
        # replacement inside one OCR run cannot accidentally reuse an old hash.
        self._sha_by_path: dict[str, tuple[int, int, int, int, str]] = {}

    def _remove_db_files(self) -> None:
        for candidate in (self._db_path, Path(str(self._db_path) + "-wal"), Path(str(self._db_path) + "-shm")):
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _is_locked_error(exc: BaseException) -> bool:
        message = str(exc).lower()
        return "locked" in message or "busy" in message

    def _open_db(self) -> None:
        if not self.enabled or self._db is not None:
            return

        existed = self._db_path.exists() and self._db_path.stat().st_size > 0
        try:
            conn = sqlite3.connect(str(self._db_path), timeout=15.0, check_same_thread=False)
            if existed:
                meta_exists = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'"
                ).fetchone()
                if not meta_exists:
                    conn.close()
                    self._remove_db_files()
                    existed = False
                    conn = sqlite3.connect(str(self._db_path), timeout=15.0, check_same_thread=False)
                else:
                    row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
                    if row is None or str(row[0]) != str(SQLITE_SCHEMA_VERSION):
                        conn.close()
                        self._remove_db_files()
                        existed = False
                        conn = sqlite3.connect(str(self._db_path), timeout=15.0, check_same_thread=False)
        except sqlite3.DatabaseError as exc:
            try:
                conn.close()  # type: ignore[name-defined]
            except Exception:
                pass
            if self._is_locked_error(exc):
                raise
            # Segment OCR cache is explicitly rebuildable, never authoritative.
            # Corrupt/unknown cache databases are replaced instead of being
            # relabelled as the current schema.
            self._remove_db_files()
            conn = sqlite3.connect(str(self._db_path), timeout=15.0, check_same_thread=False)
            existed = False

        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS entries (
                key TEXT PRIMARY KEY,
                engine TEXT NOT NULL,
                operation TEXT NOT NULL,
                runtime_id TEXT NOT NULL,
                text TEXT NOT NULL,
                confidence REAL NOT NULL,
                error TEXT NOT NULL DEFAULT '',
                updated_at REAL NOT NULL
            )"""
        )
        if not existed:
            conn.execute(
                "INSERT INTO meta(key,value) VALUES('schema_version',?)",
                (str(SQLITE_SCHEMA_VERSION),),
            )
        conn.execute(f"PRAGMA user_version={int(SQLITE_SCHEMA_VERSION)}")
        conn.commit()
        self._db = conn

    def close(self) -> None:
        with self._db_lock:
            if self._db is None:
                return
            try:
                self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                pass
            try:
                self._db.close()
            finally:
                self._db = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _db_load_key(self, key: str, operation: str) -> tuple[str, float, str | None] | None:
        if not self.enabled:
            return None
        self._open_db()
        with self._db_lock:
            assert self._db is not None
            row = self._db.execute(
                "SELECT engine, operation, runtime_id, text, confidence, error "
                "FROM entries WHERE key=?",
                (str(key),),
            ).fetchone()
        if row is None:
            return None
        engine, stored_operation, runtime_id, text, confidence, error = row
        if str(engine) != self.engine or str(stored_operation) != str(operation or ""):
            return None
        if str(runtime_id) != self.runtime_id or str(error or ""):
            return None
        try:
            confidence_value = float(confidence or 0.0)
        except (TypeError, ValueError, OverflowError):
            confidence_value = 0.0
        if not math.isfinite(confidence_value):
            confidence_value = 0.0
        return str(text or ""), confidence_value, None

    def _db_save_key(
        self, key: str, operation: str, result: tuple[str, float, str | None]
    ) -> None:
        text, confidence, error = result
        if error or not str(text or "").strip():
            return
        try:
            confidence_value = float(confidence or 0.0)
        except (TypeError, ValueError, OverflowError):
            confidence_value = 0.0
        if not math.isfinite(confidence_value):
            confidence_value = 0.0
        self._open_db()
        with self._db_lock:
            assert self._db is not None
            self._db.execute(
                """INSERT INTO entries
                   (key,engine,operation,runtime_id,text,confidence,error,updated_at)
                   VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(key) DO UPDATE SET
                     engine=excluded.engine, operation=excluded.operation,
                     runtime_id=excluded.runtime_id, text=excluded.text,
                     confidence=excluded.confidence, error=excluded.error,
                     updated_at=excluded.updated_at""",
                (str(key), self.engine, str(operation or ""), self.runtime_id,
                 str(text or ""), confidence_value, "", time.time()),
            )
            self._db.commit()

    def _digest(self, path: str) -> str:
        stat = os.stat(path)
        sig = (
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(getattr(stat, "st_ctime_ns", 0) or 0),
            int(getattr(stat, "st_ino", 0) or 0),
        )
        cached = self._sha_by_path.get(path)
        if cached is not None and cached[:4] == sig:
            return cached[4]
        digest = file_sha256(path)
        self._sha_by_path[path] = (*sig, digest)
        return digest

    def _entry(
        self,
        path: str,
        operation: str,
        *,
        image_sha256: str | None = None,
    ) -> tuple[str, Path]:
        digest = str(image_sha256 or "") or self._digest(path)
        key = make_key(
            image_sha256=digest,
            engine=self.engine,
            operation=operation,
            runtime_id=self.runtime_id,
        )
        return key, self.root / key[:2] / f"{key}.json"

    def load(
        self,
        path: str,
        operation: str,
        *,
        image_sha256: str | None = None,
    ) -> tuple[str, float, str | None] | None:
        if not self.enabled or not os.path.isfile(path):
            return None
        key, _entry = self._entry(path, operation, image_sha256=image_sha256)
        return self._db_load_key(key, operation)

    def save(
        self,
        path: str,
        operation: str,
        result: tuple[str, float, str | None],
        *,
        image_sha256: str | None = None,
    ) -> None:
        if not self.enabled or not os.path.isfile(path):
            return
        text, confidence, error = result
        # A blank recognition from a physical text crop is not useful durable
        # evidence.  Treat it like a soft miss so a later run can recover from a
        # transient helper/model hiccup instead of replaying emptiness forever.
        if error or not str(text or "").strip():
            return
        key, _entry = self._entry(path, operation, image_sha256=image_sha256)
        try:
            confidence_value = float(confidence or 0.0)
        except (TypeError, ValueError, OverflowError):
            confidence_value = 0.0
        if not math.isfinite(confidence_value):
            confidence_value = 0.0
        self._db_save_key(key, operation, (str(text or ""), confidence_value, None))

    def load_many(
        self,
        paths: list[str],
        operation: str,
        *,
        image_sha256_by_path: Mapping[str, str] | None = None,
    ) -> tuple[dict[str, tuple[str, float, str | None]], list[str]]:
        hits: dict[str, tuple[str, float, str | None]] = {}
        if not self.enabled:
            return hits, list(paths)
        digest_map = image_sha256_by_path or {}
        key_by_path: dict[str, str] = {}
        for raw in paths:
            path = str(raw)
            if not os.path.isfile(path):
                continue
            key, _entry = self._entry(
                path, operation, image_sha256=str(digest_map.get(path, "") or "") or None
            )
            key_by_path[path] = key

        rows_by_key: dict[str, tuple[str, float, str | None]] = {}
        keys = list(key_by_path.values())
        self._open_db()
        with self._db_lock:
            assert self._db is not None
            for offset in range(0, len(keys), 800):
                batch = keys[offset:offset + 800]
                if not batch:
                    continue
                placeholders = ",".join("?" for _ in batch)
                for row in self._db.execute(
                    f"SELECT key,engine,operation,runtime_id,text,confidence,error "
                    f"FROM entries WHERE key IN ({placeholders})", batch
                ).fetchall():
                    key, engine, stored_operation, runtime_id, text_value, confidence, error = row
                    if str(engine) != self.engine or str(stored_operation) != str(operation or ""):
                        continue
                    if str(runtime_id) != self.runtime_id or str(error or ""):
                        continue
                    try:
                        conf = float(confidence or 0.0)
                    except (TypeError, ValueError, OverflowError):
                        conf = 0.0
                    if not math.isfinite(conf):
                        conf = 0.0
                    rows_by_key[str(key)] = (str(text_value or ""), conf, None)

        misses: list[str] = []
        for raw in paths:
            path = str(raw)
            key = key_by_path.get(path)
            if key and key in rows_by_key:
                hits[path] = rows_by_key[key]
                continue
            misses.append(path)
        return hits, misses
