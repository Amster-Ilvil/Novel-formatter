"""Local, per-model Gemini limits shared by processes on this machine."""
from __future__ import annotations

import os
import sqlite3
import time
from contextlib import closing
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

OCR_MODEL = "gemini-3.5-flash-lite"
ADJUDICATION_MODEL = OCR_MODEL
KNOWN_CHOICES = (OCR_MODEL, "gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash")
DEFAULT_RPM = 5
DEFAULT_TPM = 60_000
DEFAULT_RPD = 100


class LocalQuotaExceeded(RuntimeError):
    """A request was blocked by the local quota guard before sending."""


@lru_cache(maxsize=1)
def _pacific_time() -> ZoneInfo:
    return ZoneInfo("America/Los_Angeles")


class SharedModelQuota:
    """Reserve each request in a single SQLite transaction before sending."""

    def __init__(self, model, *, rpm=DEFAULT_RPM, tpm=DEFAULT_TPM, rpd=DEFAULT_RPD, database=None):
        from ai.config import _config_dir

        self.model = str(model)
        self.rpm = max(1, int(rpm))
        self.tpm = max(1, int(tpm))
        self.rpd = max(1, int(rpd))
        self.database = Path(
            database or os.environ.get("NOVEL_FORMATTER_GEMINI_QUOTA_DB")
            or _config_dir() / "gemini_quotas.sqlite3"
        )

    def _reserve(self, estimated_tokens, *, now=None) -> float:
        now = time.time() if now is None else float(now)
        day = datetime.fromtimestamp(now, _pacific_time()).date().isoformat()
        requested = max(1, int(estimated_tokens))
        if requested > self.tpm:
            raise LocalQuotaExceeded(
                f"{self.model} 单次请求估算约 {requested:,} 输入 tokens，超过本地 TPM 上限 {self.tpm:,}。"
            )
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.database, timeout=5)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "CREATE TABLE IF NOT EXISTS request_log "
                "(model TEXT NOT NULL, ts REAL NOT NULL, day TEXT NOT NULL, tokens INTEGER NOT NULL)"
            )
            db.execute("CREATE INDEX IF NOT EXISTS quota_model_time ON request_log(model, ts)")
            db.execute("DELETE FROM request_log WHERE ts < ?", (now - 172_800,))
            daily_requests = db.execute(
                "SELECT COUNT(*) FROM request_log WHERE model = ? AND day = ?", (self.model, day)
            ).fetchone()[0]
            if daily_requests >= self.rpd:
                raise LocalQuotaExceeded(
                    f"{self.model} 已达本地每日上限 {self.rpd} 次，额度将在美国太平洋时间午夜重置。"
                )
            window = db.execute(
                "SELECT ts, tokens FROM request_log WHERE model = ? AND ts > ? ORDER BY ts",
                (self.model, now - 60),
            ).fetchall()
            remaining_tokens = sum(tokens for _, tokens in window)
            remaining_requests = len(window)
            for ts, tokens in window:
                if remaining_requests < self.rpm and remaining_tokens + requested <= self.tpm:
                    break
                remaining_requests -= 1
                remaining_tokens -= tokens
                if remaining_requests < self.rpm and remaining_tokens + requested <= self.tpm:
                    return max(0.05, ts + 60 - now)
            db.execute("INSERT INTO request_log VALUES (?, ?, ?, ?)", (self.model, now, day, requested))
        return 0.0

    def acquire(self, estimated_tokens, *, cancel_check=None, wait_callback=None) -> None:
        while not (cancel_check and cancel_check()):
            try:
                wait = self._reserve(estimated_tokens)
            except (OSError, sqlite3.Error, ZoneInfoNotFoundError):
                raise LocalQuotaExceeded("Gemini 本地额度记录不可用，请检查应用设置目录和时区数据。") from None
            if wait <= 0:
                return
            if wait_callback:
                wait_callback(wait)
            deadline = time.monotonic() + wait
            while (remaining := deadline - time.monotonic()) > 0:
                if cancel_check and cancel_check():
                    break
                time.sleep(min(0.25, remaining))
        raise RuntimeError("Gemini 请求已取消")
