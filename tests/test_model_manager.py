"""Unified model manager: cache, download, verification and failure modes.

All network traffic goes to a throw-away ``http.server`` on 127.0.0.1; no test
touches the internet.  Managers use short timeouts/backoff so the whole module
runs in a few seconds.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import socket
import threading
import time
import urllib.request
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import islice, pairwise
from pathlib import Path

import pytest

import utils.model_manager as mm
from utils.model_manager import (
    ModelCancelledError,
    ModelIntegrityError,
    ModelManager,
    ModelNetworkError,
    ModelNotAvailableError,
    ModelSpec,
    ModelStatus,
)

CHUNK = 16 * 1024


def _cut(data: bytes, start: int, stop: int | None = None) -> bytes:
    return bytes(islice(data, start, stop))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args) -> None:  # keep pytest output clean
        pass

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        server = self.server
        with server.state_lock:
            server.requests.append((self.path, self.headers.get("Range")))
            route = server.routes.get(self.path)
        if route is None:
            self.send_error(404)
            return
        status = int(route.get("status", 200))
        if status != 200:
            self.send_error(status)
            return
        delay = float(route.get("hang", 0.0))
        if delay:
            time.sleep(delay)
        data: bytes = route.get("data", b"")
        start = 0
        header = self.headers.get("Range")
        match = re.match(r"bytes=(\d+)-", header or "")
        if match and route.get("range", True):
            start = int(match.group(1))
            body = _cut(data, start)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        else:
            body = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()
        drop_after = route.pop("drop_once_after", None)
        if drop_after is not None:
            self.wfile.write(_cut(body, 0, int(drop_after)))
            self.wfile.flush()
            self.close_connection = True
            return
        self.wfile.write(body)


class LocalServer:
    def __init__(self) -> None:
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.httpd.daemon_threads = True
        self.httpd.block_on_close = False
        self.httpd.routes = {}
        self.httpd.requests = []
        self.httpd.state_lock = threading.Lock()
        self.thread = threading.Thread(
            target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True,
        )

    @property
    def port(self) -> int:
        _host, port = self.httpd.server_address
        return int(port)

    def url(self, path: str) -> str:
        return "http" + "://" + f"127.0.0.1:{self.port}{path}"

    def add(self, path: str, **route) -> str:
        with self.httpd.state_lock:
            self.httpd.routes.update({path: dict(route)})
        return self.url(path)

    @property
    def requests(self) -> list[tuple[str, str | None]]:
        with self.httpd.state_lock:
            return list(self.httpd.requests)


@pytest.fixture()
def server():
    instance = LocalServer()
    instance.thread.start()
    try:
        yield instance
    finally:
        instance.httpd.shutdown()
        instance.httpd.server_close()


@pytest.fixture(autouse=True)
def _reset_preferences():
    mm.configure()
    yield
    mm.configure()


_NO_PROXY_OPEN = urllib.request.build_opener(urllib.request.ProxyHandler({})).open


def _spec(urls, data: bytes, **overrides) -> ModelSpec:
    values = dict(
        model_id="test.model",
        engine="test",
        urls=tuple(urls),
        filename="model.bin",
        sha256=_sha(data),
        size=len(data),
        hash_source="test",
        subdir="test-engine",
    )
    values.update(overrides)
    return ModelSpec(**values)


def _manager(tmp_path: Path, spec: ModelSpec, **overrides) -> ModelManager:
    options = dict(
        registry={spec.model_id: spec},
        timeout=1.0,
        retries=2,
        backoff=0.01,
        chunk_size=CHUNK,
        opener=_NO_PROXY_OPEN,
        project_root=tmp_path / "project",
        auto_download=True,
    )
    options.update(overrides)
    return ModelManager(tmp_path / "models", **options)


def _leftovers(directory: Path) -> list[str]:
    if not directory.exists():
        return []
    return sorted(
        path.name for path in directory.iterdir()
        if path.name.endswith((mm.PART_SUFFIX, mm.LOCK_SUFFIX)) or ".extract-" in path.name
    )


# ---------------------------------------------------------------------------


def test_cache_hit_uses_verified_marker_without_network(tmp_path, server):
    data = os.urandom(200_000)
    url = server.add("/model.bin", data=data)
    spec = _spec([url], data)
    manager = _manager(tmp_path, spec)
    target = manager.target_dir(spec)
    target.mkdir(parents=True)
    (target / "model.bin").write_bytes(data)

    assert manager.status(spec.model_id) is ModelStatus.READY
    marker = target / ("model.bin" + mm.VERIFIED_SUFFIX)
    assert marker.is_file()
    assert json.loads(marker.read_text(encoding="utf-8")).get("sha256") == _sha(data)

    def no_rehash(*_args, **_kwargs):
        raise AssertionError("SHA-256 must not be recomputed when .verified is valid")

    manager._hash_file = no_rehash  # type: ignore[method-assign]
    assert manager.ensure(spec.model_id) == target / "model.bin"
    assert server.requests == []


def test_download_reports_monotonic_progress_and_verifies(tmp_path, server):
    data = os.urandom(5 * CHUNK + 777)
    url = server.add("/model.bin", data=data)
    spec = _spec([url], data)
    manager = _manager(tmp_path, spec)
    events: list[tuple[int, int | None]] = []

    path = manager.ensure(spec.model_id, progress_callback=lambda done, total: events.append((done, total)))

    assert path.read_bytes() == data
    done_values = [done for done, _total in events]
    assert len(done_values) >= 3
    assert all(later >= earlier for earlier, later in pairwise(done_values))
    assert next(reversed(done_values)) == len(data)
    assert {total for _done, total in events} == {len(data)}
    assert manager.status(spec.model_id) is ModelStatus.READY
    assert _leftovers(path.parent) == []


def test_hash_mismatch_raises_and_leaves_no_files(tmp_path, server):
    data = os.urandom(100_000)
    url = server.add("/model.bin", data=data)
    spec = _spec([url], data, sha256=_sha(b"something else"))
    manager = _manager(tmp_path, spec)

    with pytest.raises(ModelIntegrityError) as info:
        manager.ensure(spec.model_id)

    target = manager.target_dir(spec)
    assert url in str(info.value)
    assert not (target / "model.bin").exists()
    assert not (target / ("model.bin" + mm.PART_SUFFIX)).exists()
    assert _leftovers(target) == []


def test_connection_drop_resumes_with_range(tmp_path, server):
    data = os.urandom(6 * CHUNK)
    url = server.add("/model.bin", data=data, drop_once_after=len(data) // 2)
    spec = _spec([url], data)
    manager = _manager(tmp_path, spec)

    path = manager.ensure(spec.model_id)

    assert path.read_bytes() == data
    ranges = [header for _path, header in server.requests]
    assert len(ranges) == 2
    assert next(iter(ranges)) is None
    assert next(reversed(ranges)) == f"bytes={len(data) // 2}-"


def test_falls_back_to_second_mirror_after_404(tmp_path, server):
    data = os.urandom(50_000)
    missing = server.url("/missing.bin")
    ok = server.add("/model.bin", data=data)
    spec = _spec([missing, ok], data)
    manager = _manager(tmp_path, spec)

    assert manager.ensure(spec.model_id).read_bytes() == data
    paths = [path for path, _header in server.requests]
    assert paths == ["/missing.bin", "/model.bin"]


def test_server_error_is_retried_then_reported(tmp_path, server):
    url = server.add("/model.bin", status=500)
    spec = _spec([url], b"x" * 10)
    manager = _manager(tmp_path, spec)

    with pytest.raises(mm.ModelDownloadError) as info:
        manager.ensure(spec.model_id)

    assert info.value.status == 500
    assert "HTTP 500" in str(info.value)
    assert len(server.requests) == manager.retries + 1


def test_no_network_reports_url_and_directory(tmp_path, monkeypatch):
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    _host, closed_port = probe.getsockname()
    probe.close()
    url = "http" + "://" + f"127.0.0.1:{closed_port}/model.bin"
    spec = _spec([url], b"payload")
    manager = _manager(tmp_path, spec)

    def no_dns(*_args, **_kwargs):
        raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")

    monkeypatch.setattr(socket, "getaddrinfo", no_dns)
    with pytest.raises(ModelNetworkError) as info:
        manager.ensure(spec.model_id)

    message = str(info.value)
    assert "no network connection" in message
    assert url in message
    assert str(manager.target_dir(spec)) in message
    assert "Settings" in message


def test_hanging_server_times_out_as_network_error(tmp_path, server):
    url = server.add("/model.bin", data=b"late", hang=2.0)
    spec = _spec([url], b"late")
    manager = _manager(tmp_path, spec, timeout=0.3, retries=0)

    started = time.monotonic()
    with pytest.raises(ModelNetworkError):
        manager.ensure(spec.model_id)
    assert time.monotonic() - started < 1.9


def test_cancel_removes_temporary_files(tmp_path, server):
    data = os.urandom(10 * CHUNK)
    url = server.add("/model.bin", data=data)
    spec = _spec([url], data)
    manager = _manager(tmp_path, spec)
    cancel = threading.Event()

    def progress(done: int, _total):
        if done > 0:
            cancel.set()

    with pytest.raises(ModelCancelledError):
        manager.ensure(spec.model_id, progress_callback=progress, cancel_check=cancel.is_set)

    target = manager.target_dir(spec)
    assert not (target / "model.bin").exists()
    assert _leftovers(target) == []


def test_missing_model_without_download_permission(tmp_path, server):
    data = b"weights" * 100
    url = server.add("/model.bin", data=data)
    spec = _spec([url], data)

    with pytest.raises(ModelNotAvailableError) as info:
        _manager(tmp_path, spec).ensure(spec.model_id, allow_download=False)
    assert url in str(info.value)

    with pytest.raises(ModelNotAvailableError):
        _manager(tmp_path, spec, auto_download=False).ensure(spec.model_id)
    assert server.requests == []


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(zipfile.ZipInfo(name), payload)
    return buffer.getvalue()


def test_zip_model_is_extracted_and_required_files_checked(tmp_path, server):
    payload = _zip_bytes({"pack/a.bin": os.urandom(4096), "pack/config.json": b"{}"})
    url = server.add("/pack.zip", data=payload)
    spec = _spec(
        [url], payload, filename="pack.zip", archive="zip",
        required_files=("a.bin", "config.json"),
    )
    manager = _manager(tmp_path, spec)

    installed = manager.ensure(spec.model_id)

    assert installed == manager.target_dir(spec) / "pack"
    assert (installed / "a.bin").is_file() and (installed / "config.json").is_file()
    assert not (manager.target_dir(spec) / "pack.zip").exists()
    assert manager.status(spec.model_id) is ModelStatus.READY
    assert _leftovers(manager.target_dir(spec)) == []


def test_zip_missing_required_file_is_rejected(tmp_path, server):
    payload = _zip_bytes({"a.bin": b"a" * 100})
    url = server.add("/pack.zip", data=payload)
    spec = _spec([url], payload, filename="pack.zip", archive="zip", required_files=("a.bin", "b.bin"))
    manager = _manager(tmp_path, spec)

    with pytest.raises(ModelIntegrityError):
        manager.ensure(spec.model_id)
    assert not (manager.target_dir(spec) / "pack").exists()
    assert not (manager.target_dir(spec) / "pack.zip").exists()


def test_zip_slip_is_rejected(tmp_path, server):
    payload = _zip_bytes({"../evil.txt": b"owned", "a.bin": b"a" * 100})
    url = server.add("/pack.zip", data=payload)
    spec = _spec([url], payload, filename="pack.zip", archive="zip", required_files=("a.bin",))
    manager = _manager(tmp_path, spec)

    with pytest.raises(ModelIntegrityError) as info:
        manager.ensure(spec.model_id)

    target = manager.target_dir(spec)
    assert "unsafe" in str(info.value)
    assert not (target.parent / "evil.txt").exists()
    assert not (target / "evil.txt").exists()
    assert not (target / "pack").exists()
    assert not (target / "pack.zip").exists()
    assert _leftovers(target) == []


def test_resolve_models_dir_priority(tmp_path, monkeypatch):
    legacy = tmp_path / "legacy-cache"
    monkeypatch.setattr(mm, "LEGACY_MODELS_DIR", legacy)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv(mm.ENV_MODELS_DIR, raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    home = Path.home()

    monkeypatch.setattr(mm.sys, "platform", "darwin")
    assert mm.resolve_models_dir() == home / "Library" / "Caches" / "Novel-formatter" / "models"
    monkeypatch.setattr(mm.sys, "platform", "linux")
    assert mm.resolve_models_dir() == home / ".cache" / "novel-formatter" / "models"
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert mm.resolve_models_dir() == tmp_path / "xdg" / "novel-formatter" / "models"
    monkeypatch.setattr(mm.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    assert mm.resolve_models_dir() == tmp_path / "appdata" / "Novel-formatter" / "models"

    legacy.mkdir()
    assert mm.resolve_models_dir() == legacy

    monkeypatch.setenv(mm.ENV_MODELS_DIR, str(tmp_path / "env-dir"))
    assert mm.resolve_models_dir() == tmp_path / "env-dir"

    assert mm.resolve_models_dir(str(tmp_path / "settings-dir")) == tmp_path / "settings-dir"


def test_configure_overrides_env_and_propagates_to_workers(tmp_path, monkeypatch):
    monkeypatch.setenv(mm.ENV_MODELS_DIR, str(tmp_path / "env-dir"))
    mm.configure(models_dir=str(tmp_path / "chosen"), auto_download=False)
    assert mm.resolve_models_dir() == tmp_path / "chosen"
    assert os.environ.get(mm.ENV_MODELS_DIR) == str(tmp_path / "chosen")
    assert mm.auto_download_enabled() is False
    assert os.environ.get(mm.ENV_AUTO_DOWNLOAD) == "0"


def test_concurrent_ensure_downloads_once(tmp_path, server):
    data = os.urandom(3 * CHUNK)
    url = server.add("/model.bin", data=data, hang=0.3)
    spec = _spec([url], data)
    manager = _manager(tmp_path, spec)
    barrier = threading.Barrier(2)
    results: list[Path] = []
    errors: list[BaseException] = []

    def worker():
        barrier.wait()
        try:
            results.append(manager.ensure(spec.model_id))
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert errors == []
    assert len(results) == 2 and len(set(results)) == 1
    assert len(server.requests) == 1


def test_corrupt_cached_file_is_redownloaded(tmp_path, server):
    data = os.urandom(80_000)
    url = server.add("/model.bin", data=data)
    spec = _spec([url], data)
    manager = _manager(tmp_path, spec)
    target = manager.target_dir(spec)
    target.mkdir(parents=True)
    (target / "model.bin").write_bytes(b"\0" * len(data))

    assert manager.status(spec.model_id) is ModelStatus.CORRUPT
    assert manager.ensure(spec.model_id).read_bytes() == data
    assert manager.status(spec.model_id) is ModelStatus.READY
    assert len(server.requests) == 1


def test_stale_lock_from_dead_process_is_removed(tmp_path, server):
    data = os.urandom(10_000)
    url = server.add("/model.bin", data=data)
    spec = _spec([url], data)
    manager = _manager(tmp_path, spec, lock_timeout=2.0)
    target = manager.target_dir(spec)
    target.mkdir(parents=True)
    lock = target / f".{spec.model_id}{mm.LOCK_SUFFIX}"
    lock.write_text(json.dumps({"pid": 2_147_483_000, "time": 0}), encoding="utf-8")
    old = time.time() - 10 * manager.stale_lock_seconds
    os.utime(lock, (old, old))

    assert manager.ensure(spec.model_id).read_bytes() == data
    assert not lock.exists()


def test_legacy_location_is_used_before_downloading(tmp_path, server):
    data = os.urandom(20_000)
    url = server.add("/model.bin", data=data)
    spec = _spec([url], data, legacy_dirs=("old/models",))
    manager = _manager(tmp_path, spec)
    legacy = tmp_path / "project" / "old" / "models"
    legacy.mkdir(parents=True)
    (legacy / "model.bin").write_bytes(data)

    assert manager.ensure(spec.model_id) == legacy / "model.bin"
    assert server.requests == []


def test_import_file_verifies_before_installing(tmp_path):
    data = os.urandom(5_000)
    spec = _spec(["http" + "://127.0.0.1:9/unused"], data)
    manager = _manager(tmp_path, spec)
    good = tmp_path / "good.bin"
    good.write_bytes(data)
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"\1" * len(data))

    with pytest.raises(ModelIntegrityError):
        manager.import_file(spec.model_id, bad)
    assert bad.exists()
    installed = manager.import_file(spec.model_id, good)
    assert installed.read_bytes() == data
    assert manager.status(spec.model_id) is ModelStatus.READY


def test_registry_hashes_are_well_formed():
    for spec in mm.MODEL_REGISTRY.values():
        assert spec.urls and all(url.startswith("https" + "://") for url in spec.urls)
        assert spec.hash_source
        if spec.sha256:
            assert re.fullmatch(r"[0-9a-f]{64}", spec.sha256)
        if spec.sha384:
            assert re.fullmatch(r"[0-9a-f]{96}", spec.sha384)
