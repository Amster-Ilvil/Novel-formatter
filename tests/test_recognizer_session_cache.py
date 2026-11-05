import random
import threading
import time
from collections import deque

import pytest
from PIL import Image, ImageDraw

from adapters.column_ocr_adapter import _scanline_connected_ink_components
from adapters.recognizer_session_cache import RecognizerSessionCache, SessionKey, options_fingerprint


class FakeSession:
    def __init__(self, name, closed):
        self.name = name
        self.closed = closed
        self.cancel_check = None
        self.dead = False

    def __exit__(self, *exc):
        self.closed.append(self.name)


def make_loader(name, closed, calls, delay=0.0):
    def load():
        calls.append(name)
        if delay:
            time.sleep(delay)
        return FakeSession(name, closed)
    return load


KEY_A = SessionKey("engine", "model", "", "a")
KEY_B = SessionKey("engine", "model", "", "b")
KEY_C = SessionKey("other", "model", "", "c")


def test_reuse_lru_eviction_and_invalidate():
    closed, calls = [], []
    cache = RecognizerSessionCache(2)
    first = cache.get(KEY_A, make_loader("a", closed, calls))
    assert cache.get(KEY_A, make_loader("a2", closed, calls)) is first
    cache.get(KEY_B, make_loader("b", closed, calls))
    cache.get(KEY_C, make_loader("c", closed, calls))
    assert calls == ["a", "b", "c"]
    assert closed == ["a"]
    assert KEY_A not in cache.keys()
    assert cache.invalidate("other") == 1
    assert [key.engine for key in cache.keys()] == ["engine"]


def test_concurrent_get_loads_once():
    closed, calls, got = [], [], []
    cache = RecognizerSessionCache(2)
    threads = [
        threading.Thread(target=lambda: got.append(cache.get(KEY_A, make_loader("t", closed, calls, 0.05))))
        for _ in range(10)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(calls) == 1
    assert len({id(session) for session in got}) == 1


def test_lease_is_reentrant_exclusive_and_rebinds():
    closed, calls, order = [], [], []
    cache = RecognizerSessionCache(2)
    loader = make_loader("t", closed, calls)
    with cache.lease(KEY_A, loader, rebind=lambda s: setattr(s, "cancel_check", "job2")) as outer:
        with cache.lease(KEY_A, loader) as inner:
            assert outer is inner
            assert outer.cancel_check == "job2"

    def holder():
        with cache.lease(KEY_A, loader):
            order.append("holder-start")
            time.sleep(0.2)
            order.append("holder-end")

    def waiter():
        time.sleep(0.05)
        with cache.lease(KEY_A, loader):
            order.append("waiter")

    threads = [threading.Thread(target=holder), threading.Thread(target=waiter)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert order == ["holder-start", "holder-end", "waiter"]


def test_failure_discards_and_dead_session_reloads():
    closed, calls = [], []
    cache = RecognizerSessionCache(2, is_alive=lambda session: not session.dead)
    with pytest.raises(ValueError):
        with cache.lease(KEY_A, make_loader("broken", closed, calls)):
            raise ValueError("boom")
    assert KEY_A not in cache.keys()
    assert "broken" in closed

    stale = cache.get(KEY_A, make_loader("stale", closed, calls))
    stale.dead = True
    with cache.lease(KEY_A, make_loader("fresh", closed, calls)) as fresh:
        assert fresh is not stale
        assert fresh.name == "fresh"


def test_zero_capacity_closes_after_use():
    closed, calls = [], []
    cache = RecognizerSessionCache(0)
    with cache.lease(KEY_A, make_loader("zero", closed, calls)):
        pass
    assert closed == ["zero"]
    assert len(cache) == 0


def test_fingerprint_ignores_column_options_and_callables():
    assert options_fingerprint({"device": "cpu", "column_x": 1, "cb": print}) == options_fingerprint({"device": "cpu"})
    assert options_fingerprint({"device": "cpu"}) != options_fingerprint({"device": "mps"})


def brute_force_components(mask):
    width, height = mask.size
    pixels = mask.convert("L").load() if width and height else None
    seen = set()
    result = []
    for y in range(height):
        for x in range(width):
            if not pixels[x, y] or (x, y) in seen:
                continue
            seen.add((x, y))
            queue = deque([(x, y)])
            left, top, right, bottom, area = x, y, x + 1, y + 1, 0
            while queue:
                px, py = queue.popleft()
                area += 1
                left, top = min(left, px), min(top, py)
                right, bottom = max(right, px + 1), max(bottom, py + 1)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        nx, ny = px + dx, py + dy
                        if 0 <= nx < width and 0 <= ny < height and pixels[nx, ny] and (nx, ny) not in seen:
                            seen.add((nx, ny))
                            queue.append((nx, ny))
            result.append((left, top, right, bottom, area))
    return result


def as_tuples(components):
    return [(c.left, c.top, c.right, c.bottom, c.area) for c in components]


def test_scanline_components_match_brute_force_flood_fill():
    rng = random.Random(1234)
    masks = [Image.new("L", (0, 4)), Image.new("L", (4, 0)), Image.new("L", (6, 6)),
             Image.new("L", (6, 6), 255), Image.new("1", (5, 7), 1)]
    for _ in range(250):
        width, height = rng.randint(1, 40), rng.randint(1, 40)
        density = rng.random()
        mask = Image.new("L", (width, height))
        mask.putdata([255 if rng.random() < density else 0 for _ in range(width * height)])
        masks.append(mask)
    spiral = Image.new("L", (80, 80))
    draw = ImageDraw.Draw(spiral)
    for k in range(0, 40, 4):
        draw.rectangle((k, k, 79 - k, 79 - k), outline=255)
    for x in range(0, 80, 3):
        draw.line((x, 0, 79, 79 - x), fill=255)
    masks.append(spiral)
    for mask in masks:
        assert as_tuples(_scanline_connected_ink_components(mask)) == brute_force_components(mask)


def _lease_raising(cache, loader, exc):
    with pytest.raises(type(exc)):
        with cache.lease(KEY_A, loader) as session:
            raise exc
    return session


def test_cancelled_lease_keeps_session_and_real_error_reloads():
    from core.ocr_runtime_optimizer import OcrCancelled

    closed, calls = [], []
    cache = RecognizerSessionCache(2, is_alive=lambda session: not session.dead)
    loader = make_loader("s", closed, calls)
    first = _lease_raising(cache, loader, OcrCancelled("stop"))
    with cache.lease(KEY_A, loader) as second:
        assert second is first
    assert calls == ["s"]
    assert closed == []
    assert cache.metrics()["cancel_reuses"] == 1

    _lease_raising(cache, loader, RuntimeError("worker crashed"))
    assert closed == ["s"]
    with cache.lease(KEY_A, loader) as third:
        assert third is not first
    assert calls == ["s", "s"]


def test_process_cancelled_and_wrapped_stop_are_cancellations():
    from adapters.recognizer_session_cache import is_cancellation
    from adapters.subprocess_watchdog import ProcessCancelled

    closed, calls = [], []
    cache = RecognizerSessionCache(2, is_alive=lambda session: not session.dead)
    loader = make_loader("p", closed, calls)
    first = _lease_raising(cache, loader, ProcessCancelled("stop"))
    try:
        raise RuntimeError("helper failed") from InterruptedError("stop")
    except RuntimeError as wrapped:
        assert is_cancellation(wrapped)
        _lease_raising(cache, loader, wrapped)
    with cache.lease(KEY_A, loader) as again:
        assert again is first
    assert calls == ["p"]
    assert not is_cancellation(RuntimeError("plain"))
    assert not is_cancellation(None)


def test_cancel_with_dead_worker_or_opt_out_reloads():
    closed, calls = [], []
    cache = RecognizerSessionCache(2, is_alive=lambda session: not session.dead)
    loader = make_loader("d", closed, calls)
    with pytest.raises(InterruptedError):
        with cache.lease(KEY_A, loader) as session:
            session.dead = True
            raise InterruptedError("stop killed the worker")
    assert closed == ["d"]
    with cache.lease(KEY_A, loader) as fresh:
        fresh.discard_on_cancel = True
        opted_out = fresh
    assert calls == ["d", "d"]
    _lease_raising(cache, loader, InterruptedError("stop"))
    with cache.lease(KEY_A, loader) as reloaded:
        assert reloaded is not opted_out
    assert calls == ["d", "d", "d"]
