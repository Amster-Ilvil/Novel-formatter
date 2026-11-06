"""Unified model manager: cache, download, verification and failure modes.

All network traffic goes to a throw-away ``http.server`` on 127.0.0.1; no test
touches the internet.  Managers use short timeouts/backoff so the whole module
runs in a few seconds.
"""
from __future__ import annotations

import errno
import hashlib
import http.client
import io
import json
import multiprocessing
import os
import re
import socket
import threading
import time
import urllib.request
import urllib.response
import zipfile
from email.message import Message
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
        if route.get("redirect"):
            self.send_response(int(route.get("status", 302)))
            self.send_header("Location", route.get("redirect"))
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        status = int(route.get("status", 200))
        if status != 200:
            self.send_error(status)
            return
        entered = route.get("entered")
        release = route.get("release")
        if entered is not None:
            entered.set()
        if release is not None and not release.wait(10):
            self.send_error(503)
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
            total = "*" if route.get("unknown_size") else str(len(data))
            content_range = route.get("content_range", f"bytes {start}-{len(data) - 1}/{total}")
            if content_range is not None:
                self.send_header("Content-Range", content_range)
        else:
            body = data
            self.send_response(200)
        if route.get("unknown_size"):
            self.send_header("Connection", "close")
            self.close_connection = True
        else:
            self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()
        drop_after = route.pop("drop_once_after", None)
        if drop_after is not None:
            self.wfile.write(_cut(body, 0, int(drop_after)))
            self.wfile.flush()
            self.close_connection = True
            return
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass


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


_NO_PROXY_OPEN = urllib.request.build_opener(
    urllib.request.ProxyHandler({}), mm._SafeRedirectHandler(),
).open


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
    # OS lock files persist intentionally; partial downloads and staging do not.
    return sorted(
        path.name for path in directory.iterdir()
        if path.name.endswith(mm.PART_SUFFIX) or ".extract-" in path.name
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


def test_hanging_server_reports_model_and_applied_timeout(tmp_path, server):
    url = server.add("/model.bin", data=b"late", hang=2.0)
    spec = _spec([url], b"late")
    manager = _manager(tmp_path, spec, timeout=0.3, retries=0)

    started = time.monotonic()
    with pytest.raises(ModelNetworkError) as info:
        manager.ensure(spec.model_id)
    assert time.monotonic() - started < 1.9
    message = str(info.value)
    assert spec.model_id in message
    assert "timed out" in message
    assert "timeout 0.3 seconds" in message
    assert "no network connection" not in message
    assert "manually" in message
    assert url in message
    assert str(manager.target_dir(spec)) in message


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


def test_stale_lock_from_dead_process_is_reused(tmp_path, server):
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
    original_inode = lock.stat().st_ino

    assert manager.ensure(spec.model_id).read_bytes() == data
    assert lock.is_file()
    assert lock.stat().st_ino == original_inode
    with manager._process_lock(spec, target, None):
        assert lock.is_file()


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


@pytest.mark.parametrize("intermediate", (False, True))
@pytest.mark.parametrize("target_kind", ("http", "ftp", "credentials"))
def test_default_transport_rejects_unsafe_https_redirect_before_target_request(
    tmp_path, server, monkeypatch, target_kind, intermediate,
):
    data = b"weights"
    destination = server.add("/target.bin", data=data)
    if target_kind == "ftp":
        destination = "ftp" + "://127.0.0.1/target.bin"
    elif target_kind == "credentials":
        destination = "https" + "://user:password@127.0.0.1/target.bin"
    origin = "https" + "://models.invalid/model.bin"
    spec = _spec((origin,), data)
    middle = "https" + "://mirror.invalid/model.bin"
    requested = []

    def https_redirect(_handler, request):
        requested.append(request.full_url)
        headers = Message()
        target = middle if intermediate and request.full_url == origin else destination
        headers.add_header("Location", target)
        headers.add_header("Content-Length", "0")
        response = urllib.response.addinfourl(io.BytesIO(), headers, request.full_url, 302)
        response.msg = "Found"
        return response

    monkeypatch.setattr(urllib.request.HTTPSHandler, "https_open", https_redirect)
    manager = _manager(tmp_path, spec, opener=None, retries=0)
    with pytest.raises(mm.ModelDownloadError, match="unsafe redirect"):
        manager.ensure(spec.model_id)
    assert requested == ([origin, middle] if intermediate else [origin])
    assert server.requests == []
    assert not manager.installed_path(spec, manager.target_dir(spec)).exists()


def test_default_transport_allows_https_redirect_chain(tmp_path, monkeypatch):
    data = b"verified HTTPS weights"
    origin = "https" + "://models.invalid/model.bin"
    middle = "https" + "://mirror.invalid/model.bin"
    destination = "https" + "://cdn.invalid/model.bin"
    redirects = {origin: middle, middle: destination}
    requested = []

    def https_response(_handler, request):
        requested.append(request.full_url)
        headers = Message()
        if request.full_url in redirects:
            headers.add_header("Location", redirects[request.full_url])
            headers.add_header("Content-Length", "0")
            response = urllib.response.addinfourl(io.BytesIO(), headers, request.full_url, 302)
            response.msg = "Found"
        else:
            assert request.full_url == destination
            headers.add_header("Content-Length", str(len(data)))
            response = urllib.response.addinfourl(io.BytesIO(data), headers, request.full_url, 200)
            response.msg = "OK"
        return response

    monkeypatch.setattr(urllib.request.HTTPSHandler, "https_open", https_response)
    spec = _spec((origin,), data)
    manager = _manager(tmp_path, spec, opener=None, retries=0)
    assert manager.ensure(spec.model_id).read_bytes() == data
    assert requested == [origin, middle, destination]
    assert manager.verify(spec.model_id, force=True) is ModelStatus.READY


@pytest.mark.parametrize(
    "url",
    (
        "http" + "://models.invalid/model.bin",
        "http" + "://127.0.0.1.models.invalid/model.bin",
        "ftp" + "://127.0.0.1/model.bin",
        "file" + ":///tmp/model.bin",
    ),
)
def test_unsafe_initial_url_never_reaches_transport(tmp_path, url):
    calls = []

    def forbidden_open(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("Unsafe URL reached transport")

    spec = _spec((url,), b"weights")
    manager = _manager(tmp_path, spec, opener=forbidden_open)
    with pytest.raises(mm.ModelDownloadError, match="unsafe URL"):
        manager.ensure(spec.model_id)
    assert calls == []


def test_loopback_redirect_is_allowed(tmp_path, server):
    data = b"weights"
    target = server.add("/target.bin", data=data)
    origin = server.add("/redirect.bin", redirect=target)
    spec = _spec((origin,), data)
    assert _manager(tmp_path, spec).ensure(spec.model_id).read_bytes() == data
    assert [path for path, _range in server.requests] == ["/redirect.bin", "/target.bin"]


@pytest.mark.parametrize("known_size", (False, True))
@pytest.mark.parametrize("resume", (False, True))
@pytest.mark.parametrize("unknown_response_size", (False, True))
def test_download_size_limit_removes_partial_file(
    tmp_path, server, known_size, resume, unknown_response_size,
):
    data = os.urandom(3 * CHUNK)
    limit = 2 * CHUNK
    url = server.add("/oversized.bin", data=data, unknown_size=unknown_response_size)
    spec = _spec((url,), data, size=limit if known_size else None)
    manager = _manager(tmp_path, spec, max_download_bytes=limit, retries=0)
    directory = manager.target_dir(spec)
    directory.mkdir(parents=True)
    part = directory / (spec.filename + mm.PART_SUFFIX)
    if resume:
        part.write_bytes(_cut(data, 0, CHUNK))
    progress = []
    with pytest.raises(ModelIntegrityError, match=f"limit of {limit} bytes") as info:
        manager.ensure(
            spec.model_id,
            progress_callback=lambda done, total: progress.append(done),
        )
    assert spec.model_id in str(info.value)
    assert not part.exists()
    assert not manager.installed_path(spec, directory).exists()
    assert all(done <= limit for done in progress)
    assert len(server.requests) == 1
    ((_path, range_header),) = server.requests
    assert range_header == (f"bytes={CHUNK}-" if resume else None)


@pytest.mark.parametrize("known_size", (False, True))
@pytest.mark.parametrize("oversized", (False, True))
def test_resume_when_server_ignores_range(tmp_path, server, known_size, oversized):
    data = b"a" * CHUNK + b"b" * CHUNK
    payload = data + b"extra" if oversized else data
    url = server.add("/model.bin", data=payload, range=False, unknown_size=True)
    spec = _spec((url,), data, size=len(data) if known_size else None)
    manager = _manager(tmp_path, spec, max_download_bytes=len(data), retries=0)
    directory = manager.target_dir(spec)
    directory.mkdir(parents=True)
    part = directory / (spec.filename + mm.PART_SUFFIX)
    part.write_bytes(_cut(data, 0, CHUNK))

    if oversized:
        with pytest.raises(ModelIntegrityError, match="exceeds the limit"):
            manager.ensure(spec.model_id)
        assert not manager.installed_path(spec, directory).exists()
    else:
        assert manager.ensure(spec.model_id).read_bytes() == data
        assert manager.verify(spec.model_id, force=True) is ModelStatus.READY
    assert not part.exists()
    assert server.requests == [("/model.bin", f"bytes={CHUNK}-")]


@pytest.mark.parametrize(
    "content_range",
    (
        None,
        "not a byte range",
        f"bytes 0-{3 * CHUNK - 1}/{3 * CHUNK}",
        f"bytes {CHUNK}-{CHUNK - 1}/{3 * CHUNK}",
        f"bytes {CHUNK}-{3 * CHUNK}/{3 * CHUNK}",
        f"bytes {CHUNK}-{2 * CHUNK - 1}/{3 * CHUNK}",
    ),
)
def test_resume_rejects_invalid_content_range(tmp_path, server, content_range):
    data = b"r" * (3 * CHUNK)
    url = server.add("/model.bin", data=data, content_range=content_range)
    spec = _spec((url,), data)
    manager = _manager(tmp_path, spec, retries=0)
    directory = manager.target_dir(spec)
    directory.mkdir(parents=True)
    part = directory / (spec.filename + mm.PART_SUFFIX)
    part.write_bytes(_cut(data, 0, CHUNK))

    with pytest.raises(ModelIntegrityError, match="Content-Range"):
        manager.ensure(spec.model_id)
    assert not part.exists()
    assert not manager.installed_path(spec, directory).exists()
    assert server.requests == [("/model.bin", f"bytes={CHUNK}-")]


@pytest.mark.parametrize("known_size", (False, True))
def test_existing_oversized_partial_is_deleted_without_request(tmp_path, server, known_size):
    data = b"abcdefgh"
    url = server.add("/model.bin", data=data)
    spec = _spec((url,), data, size=len(data) if known_size else None)
    manager = _manager(tmp_path, spec, max_download_bytes=len(data))
    directory = manager.target_dir(spec)
    directory.mkdir(parents=True)
    part = directory / (spec.filename + mm.PART_SUFFIX)
    part.write_bytes(data + b"extra")
    with pytest.raises(ModelIntegrityError, match="exceeds the limit"):
        manager.ensure(spec.model_id)
    assert not part.exists()
    assert server.requests == []


def test_known_spec_size_takes_precedence_over_fallback_limit(tmp_path, server):
    data = b"weights"
    url = server.add("/model.bin", data=data)
    spec = _spec((url,), data)
    manager = _manager(tmp_path, spec, max_download_bytes=1)
    assert manager.ensure(spec.model_id).read_bytes() == data


@pytest.mark.parametrize("known_size", (False, True))
@pytest.mark.parametrize("resume", (False, True))
def test_size_limit_does_not_replace_existing_verified_model(tmp_path, server, known_size, resume):
    data = b"v" * (2 * CHUNK)
    url = server.add("/model.bin", data=data + b"extra", unknown_size=True)
    spec = _spec((url,), data, size=len(data) if known_size else None)
    manager = _manager(tmp_path, spec, max_download_bytes=len(data), retries=0)
    directory = manager.target_dir(spec)
    directory.mkdir(parents=True)
    installed = manager.installed_path(spec, directory)
    installed.write_bytes(data)
    assert manager.verify(spec.model_id, force=True) is ModelStatus.READY
    marker = manager._marker_path(installed)
    original_marker = marker.read_bytes()
    part = directory / (spec.filename + mm.PART_SUFFIX)
    if resume:
        part.write_bytes(_cut(data, 0, CHUNK))

    # Exercise the installation transaction directly: ensure() would return
    # this valid cached file without attempting a download.
    with manager._process_lock(spec, directory, None) as heartbeat:
        with pytest.raises(ModelIntegrityError, match="exceeds the limit"):
            manager._download_and_install(spec, directory, None, None, heartbeat)
    assert installed.read_bytes() == data
    assert marker.read_bytes() == original_marker
    assert not part.exists()
    assert manager.verify(spec.model_id, force=True) is ModelStatus.READY
    assert server.requests == [("/model.bin", f"bytes={CHUNK}-" if resume else None)]


def test_default_download_limit_is_four_gib(tmp_path):
    spec = _spec((), b"weights")
    assert _manager(tmp_path, spec).max_download_bytes == 4 * 1024 * 1024 * 1024


def test_incomplete_read_partial_obeys_size_limit(tmp_path):
    class PartialResponse(io.BytesIO):
        status = 200
        headers = {}

        def read(self, size=-1):
            raise http.client.IncompleteRead(b"x" * 17, 30)

    spec = _spec(("https" + "://models.invalid/model.bin",), b"x" * 8, size=None)
    manager = _manager(
        tmp_path, spec, max_download_bytes=8,
        opener=lambda request, timeout: PartialResponse(),
    )
    with pytest.raises(ModelIntegrityError, match="limit of 8 bytes"):
        manager.ensure(spec.model_id)
    assert _leftovers(manager.target_dir(spec)) == []
    assert not manager.installed_path(spec, manager.target_dir(spec)).exists()


def test_response_read_timeout_is_distinct_from_offline(tmp_path):
    class TimeoutResponse(io.BytesIO):
        status = 200
        headers = {}

        def read(self, size=-1):
            raise TimeoutError("read timed out")

    spec = _spec(("https" + "://models.invalid/model.bin",), b"weights")
    manager = _manager(
        tmp_path, spec, timeout=0.25, retries=0,
        opener=lambda request, timeout: TimeoutResponse(),
    )
    with pytest.raises(ModelNetworkError) as info:
        manager.ensure(spec.model_id)
    message = str(info.value)
    assert "timeout 0.25 seconds" in message
    assert spec.model_id in message
    assert "manually" in message
    assert "no network connection" not in message


def test_force_verify_bypasses_matching_marker(tmp_path, server):
    data = b"original weights"
    url = server.add("/model.bin", data=data)
    spec = _spec((url,), data)
    manager = _manager(tmp_path, spec)
    directory = manager.target_dir(spec)
    directory.mkdir(parents=True)
    path = directory / spec.filename
    path.write_bytes(data)
    assert manager.verify(spec.model_id, force=True) is ModelStatus.READY
    original_stat = path.stat()
    path.write_bytes(b"x" * len(data))
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert manager.verify(spec.model_id, force=False) is ModelStatus.READY
    assert manager.verify(spec.model_id, force=True) is ModelStatus.CORRUPT
    assert server.requests == []


def test_force_verify_reports_missing_file_without_downloading(tmp_path, server):
    spec = _spec((server.url("/unused.bin"),), b"weights")
    assert _manager(tmp_path, spec).verify(spec.model_id, force=True) is ModelStatus.MISSING
    assert server.requests == []


def test_force_verify_does_not_claim_full_verification_of_extracted_files(tmp_path):
    spec = _spec((), b"archive", filename="pack.zip", archive="zip")
    manager = _manager(tmp_path, spec)
    with pytest.raises(mm.ModelError, match="Full verification of extracted model"):
        manager.verify(spec.model_id, force=True)


def test_dictionary_size_matches_review_without_changing_digest_or_urls():
    spec = mm.MODEL_REGISTRY.get("manga_48px.dictionary")
    assert spec.size == 186_651
    assert spec.sha256 == "f5722368146aa0fbcc9f4726866e4efc3203318ebb66c811d8cbbe915576538a"
    assert spec.urls == (
        mm._https("huggingface.co", f"{mm._HF_48PX}/alphabet-all-v7.txt?download=true"),
        mm._https("github.com", f"{mm._GH_48PX}/alphabet-all-v7.txt"),
    )


def _download_in_process(root, spec, results, waiting=None):
    try:
        if waiting is not None:
            if os.name == "nt":
                import msvcrt as lock_module
                lock_name = "locking"
            else:
                import fcntl as lock_module
                lock_name = "flock"
            acquire = getattr(lock_module, lock_name)

            def observed_acquire(*args):
                try:
                    return acquire(*args)
                except OSError as exc:
                    if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        waiting.set()
                    raise

            setattr(lock_module, lock_name, observed_acquire)
        manager = _manager(root, spec, timeout=10.0, lock_timeout=10.0, stale_lock_seconds=0.01)
        path = manager.ensure(spec.model_id)
        results.put(("ok", str(path), _sha(path.read_bytes())))
    except BaseException as exc:
        results.put(("error", type(exc).__name__, str(exc)))


def _hold_lock_in_process(root, spec, ready):
    manager = _manager(root, spec)
    directory = manager.target_dir(spec)
    directory.mkdir(parents=True, exist_ok=True)
    with manager._process_lock(spec, directory, None):
        ready.set()
        threading.Event().wait()


def test_two_processes_with_stale_lock_download_once(tmp_path, server):
    data = os.urandom(4 * CHUNK)
    entered = threading.Event()
    release = threading.Event()
    url = server.add("/model.bin", data=data, entered=entered, release=release)
    spec = _spec((url,), data)
    manager = _manager(tmp_path, spec)
    directory = manager.target_dir(spec)
    directory.mkdir(parents=True)
    lock = directory / f".{spec.model_id}{mm.LOCK_SUFFIX}"
    lock.write_text(json.dumps({"pid": 2_147_483_000, "time": 0}), encoding="utf-8")
    old = time.time() - 86_400
    os.utime(lock, (old, old))
    original_inode = lock.stat().st_ino

    context = multiprocessing.get_context("spawn")
    waiting = context.Event()
    results = context.Queue()
    first = context.Process(target=_download_in_process, args=(tmp_path, spec, results))
    second = context.Process(target=_download_in_process, args=(tmp_path, spec, results, waiting))
    processes = [first, second]
    try:
        first.start()
        assert entered.wait(10), "First downloader did not reach the server"
        second.start()
        assert waiting.wait(10), "Second downloader did not contend on the OS lock"
        assert server.requests == [("/model.bin", None)]
        assert lock.stat().st_ino == original_inode
        release.set()
        for process in processes:
            process.join(15)
        assert all(not process.is_alive() for process in processes)
        assert all(process.exitcode == 0 for process in processes)
        received = [results.get(timeout=3) for _ in processes]
        assert all(status == "ok" for status, _path, _digest in received), received
        assert {path for _status, path, _digest in received} == {
            str(manager.installed_path(spec, directory)),
        }
        assert {digest for _status, _path, digest in received} == {_sha(data)}
        assert len(server.requests) == 1
        assert lock.stat().st_ino == original_inode
        assert manager.verify(spec.model_id, force=True) is ModelStatus.READY
        assert _leftovers(directory) == []
    finally:
        release.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
            if process.pid is not None:
                process.join(5)
        results.close()
        results.join_thread()


def test_live_lock_is_not_stolen_and_process_exit_releases_it(tmp_path, server):
    data = b"weights"
    url = server.add("/model.bin", data=data)
    spec = _spec((url,), data)
    manager = _manager(tmp_path, spec, lock_timeout=0.2, stale_lock_seconds=0.01)
    directory = manager.target_dir(spec)
    directory.mkdir(parents=True)
    lock = directory / f".{spec.model_id}{mm.LOCK_SUFFIX}"
    lock.write_text("old lock contents", encoding="utf-8")
    original_inode = lock.stat().st_ino
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    process = context.Process(target=_hold_lock_in_process, args=(tmp_path, spec, ready))
    process.start()
    try:
        assert ready.wait(10), "Child did not acquire the OS lock"
        old = time.time() - 86_400
        os.utime(lock, (old, old))
        with pytest.raises(mm.ModelError, match="Timed out waiting for another download"):
            manager.ensure(spec.model_id)
        cancel_checks = []

        def cancel_after_contention():
            cancel_checks.append(True)
            return len(cancel_checks) == 2

        with pytest.raises(ModelCancelledError) as cancelled:
            manager.ensure(spec.model_id, cancel_check=cancel_after_contention)
        assert cancelled.value.model_id == spec.model_id
        assert len(cancel_checks) == 2
        assert process.is_alive()
        assert server.requests == []
        assert lock.stat().st_ino == original_inode
    finally:
        if process.is_alive():
            process.terminate()
        process.join(5)
    assert not process.is_alive()
    assert manager.ensure(spec.model_id).read_bytes() == data
    assert len(server.requests) == 1
    assert lock.stat().st_ino == original_inode


def test_registry_hashes_are_well_formed():
    for spec in mm.MODEL_REGISTRY.values():
        assert spec.urls and all(url.startswith("https" + "://") for url in spec.urls)
        assert spec.hash_source
        if spec.sha256:
            assert re.fullmatch(r"[0-9a-f]{64}", spec.sha256)
        if spec.sha384:
            assert re.fullmatch(r"[0-9a-f]{96}", spec.sha384)


def test_48px_does_not_retry_unsafe_redirect_through_curl(tmp_path, monkeypatch):
    from adapters import manga_48px_runtime as runtime

    data = b"weights"
    url = "https" + "://models.invalid/model.bin"
    spec = _spec((url,), data)
    fallback_calls = []

    def unsafe_redirect(request, timeout):
        raise mm._UnsafeDownloadURL("unsafe HTTPS redirect")

    manager = _manager(tmp_path, spec, opener=unsafe_redirect, retries=0)
    monkeypatch.setattr(mm, "default_manager", lambda: manager)
    monkeypatch.setattr(runtime, "_seed_from_local", lambda *args, **kwargs: False)
    monkeypatch.setattr(runtime.shutil, "which", lambda name: "/usr/bin/curl")

    def forbidden_fallback(*args, **kwargs):
        fallback_calls.append((args, kwargs))
        raise AssertionError("Security rejection reached curl fallback")

    monkeypatch.setattr(mm.subprocess, "Popen", forbidden_fallback)
    with pytest.raises(mm.ModelDownloadError, match="unsafe redirect"):
        runtime._download(
            (url,), tmp_path / "model.bin", sha256=_sha(data), expected_size=len(data),
        )
    assert fallback_calls == []


def test_48px_curl_transport_restricts_initial_and_redirect_protocols(tmp_path, monkeypatch):
    commands = []

    def capture_command(command, **kwargs):
        commands.append((command, kwargs))
        raise OSError("command captured without execution")

    monkeypatch.setattr(mm.shutil, "which", lambda name: "/usr/bin/curl")
    monkeypatch.setattr(mm.subprocess, "Popen", capture_command)
    with pytest.raises(ModelNetworkError, match="command captured"):
        with mm._CurlResponse(
            "https" + "://models.invalid/model.bin", timeout=1,
            check_cancel=lambda: None, context={},
        ):
            pytest.fail("capture must not execute curl")
    ((command, kwargs),) = commands
    options = tuple(pairwise(command))
    assert next(islice(command, 1, 2)) == "-q"
    assert ("--proto", "=https") in options
    assert ("--proto-redir", "=https") in options
    assert ("--output", "-") in options
    assert "--continue-at" not in command
    assert "--retry" not in command
    assert "--insecure" not in command and "-k" not in command
    assert kwargs["shell"] is False
    assert kwargs["stdout"] == mm.subprocess.PIPE
    assert kwargs["stderr"] == mm.subprocess.PIPE


_REAL_POPEN = mm.subprocess.Popen


class _MemoryResponse(io.BytesIO):
    status = 200
    headers = {}


def _offline_transport(request, timeout):
    raise urllib.error.URLError("forced Python HTTPS failure")


def _controlled_curl(monkeypatch, script):
    processes, commands = [], []

    def launch(command, **kwargs):
        commands.append((command, kwargs))
        process = _REAL_POPEN([mm.sys.executable, "-u", "-c", script], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(mm.shutil, "which", lambda name: mm.sys.executable)
    monkeypatch.setattr(mm.subprocess, "Popen", launch)
    return processes, commands


def _assert_curl_reaped(processes):
    assert processes
    assert all(process.poll() is not None for process in processes)
    assert all(process.stdout.closed and process.stderr.closed for process in processes)
    assert not any(
        thread.is_alive() and thread.name.startswith("model-curl-")
        for thread in threading.enumerate()
    )


@pytest.mark.parametrize("primary_error", ("network", "http"))
def test_curl_fallback_streams_verifies_and_restarts_partial(tmp_path, monkeypatch, primary_error):
    from adapters import manga_48px_runtime as runtime

    data = b"verified fallback weights" * 8000
    url = "https" + "://models.invalid/model.bin"
    spec = _spec((url,), data)
    requests, progress = [], []

    def failed(request, timeout):
        requests.append(request)
        if primary_error == "http":
            raise urllib.error.HTTPError(url, 503, "unavailable", {}, None)
        return _offline_transport(request, timeout)

    manager = _manager(tmp_path, spec, opener=failed, retries=0)
    monkeypatch.setattr(mm, "default_manager", lambda: manager)
    monkeypatch.setattr(runtime, "_seed_from_local", lambda *args, **kwargs: False)
    processes, commands = _controlled_curl(
        monkeypatch, "import sys; sys.stdout.buffer.write(b'verified fallback weights' * 8000)",
    )
    destination = tmp_path / spec.filename
    part = destination.with_name(destination.name + mm.PART_SUFFIX)
    part.write_bytes(b"old partial")
    result = runtime._download(
        spec.urls, destination, sha256=spec.sha256, expected_size=spec.size,
        progress_callback=lambda stage, done, total, detail: progress.append((stage, done, total)),
    )
    assert result == destination
    assert _sha(result.read_bytes()) == spec.sha256
    assert manager.verify(spec.model_id, target_dir=tmp_path) is ModelStatus.READY
    assert next(iter(requests)).get_header("Range") == "bytes=11-"
    assert len(processes) == 1 and not part.exists()
    assert all(str(part) not in command for command, _kwargs in commands)
    assert ("download", len(data), len(data)) in progress
    assert all(left_done <= right_done for (_, left_done, _), (_, right_done, _) in pairwise(progress))
    _assert_curl_reaped(processes)


@pytest.mark.parametrize("known_size", (False, True))
@pytest.mark.parametrize("resume", (False, True))
def test_curl_fallback_bounds_actual_writes_preserves_verified_model(
    tmp_path, monkeypatch, known_size, resume,
):
    data = b"v" * (2 * CHUNK)
    spec = _spec(("https" + "://models.invalid/model.bin",), data, size=len(data) if known_size else None)
    manager = _manager(
        tmp_path, spec, opener=_offline_transport, retries=0, max_download_bytes=len(data),
    )
    directory = manager.target_dir(spec)
    directory.mkdir(parents=True)
    installed = directory / spec.filename
    installed.write_bytes(data)
    assert manager.verify(spec.model_id) is ModelStatus.READY
    marker = manager._marker_path(installed)
    marker_bytes = marker.read_bytes()
    part = directory / (spec.filename + mm.PART_SUFFIX)
    if resume:
        part.write_bytes(b"partial")
    writes = []
    original_open = Path.open

    class AuditedWriter:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def write(self, chunk):
            end = self.stream.tell() + len(chunk)
            assert end <= len(data), "oversized bytes reached the actual writer"
            writes.append(end)
            return self.stream.write(chunk)

    def audited_open(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        return AuditedWriter(stream) if path == part and mode in ("wb", "ab") else stream

    monkeypatch.setattr(Path, "open", audited_open)
    processes, _commands = _controlled_curl(
        monkeypatch,
        "import os, threading\n"
        "while True:\n"
        "    os.write(1, b'x' * 65536)\n",
    )
    with manager._process_lock(spec, directory, None) as heartbeat:
        with pytest.raises(ModelIntegrityError, match="exceeds the limit"):
            manager._download_and_install(
                spec, directory, None, None, heartbeat, curl_fallback=True,
            )
    assert writes and max(writes) <= len(data)
    assert not part.exists()
    assert installed.read_bytes() == data
    assert marker.read_bytes() == marker_bytes
    assert manager.verify(spec.model_id) is ModelStatus.READY
    assert len(processes) == 1
    _assert_curl_reaped(processes)


@pytest.mark.parametrize("mode", ("cancel", "timeout", "cancel_ignores_terminate"))
def test_curl_fallback_interrupt_reaps_process_and_releases_lock(tmp_path, monkeypatch, mode):
    if mode == "cancel_ignores_terminate" and os.name == "nt":
        pytest.skip("SIGTERM handler is POSIX-specific")
    data = b"complete"
    spec = _spec(("https" + "://models.invalid/model.bin",), data)
    manager = _manager(tmp_path, spec, opener=_offline_transport, retries=0, timeout=0.5)
    prefix = "import os, signal, threading\n"
    if mode == "cancel_ignores_terminate":
        prefix += "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    processes, _commands = _controlled_curl(
        monkeypatch, prefix + "os.write(1, b'x')\nthreading.Event().wait(60)\n",
    )
    cancel = threading.Event()
    progress = []

    def on_progress(done, total):
        progress.append(done)
        if done and mode.startswith("cancel"):
            cancel.set()

    error = ModelCancelledError if mode.startswith("cancel") else ModelNetworkError
    started = time.monotonic()
    with pytest.raises(error) as caught:
        manager.ensure_spec(
            spec, curl_fallback=True, progress_callback=on_progress, cancel_check=cancel.is_set,
        )
    assert time.monotonic() - started < 8
    assert 1 in progress
    if mode == "timeout":
        assert "timed out" in str(caught.value)
        assert isinstance(caught.value.__cause__, ModelNetworkError)
    directory = manager.target_dir(spec)
    assert _leftovers(directory) == []
    assert not (directory / spec.filename).exists()
    _assert_curl_reaped(processes)
    # A fresh ensure needs both the thread lock and the OS lock.
    manager._open = lambda request, timeout: _MemoryResponse(data)
    assert manager.ensure_spec(spec, curl_fallback=True).read_bytes() == data
    assert len(processes) == 1


@pytest.mark.parametrize("failure", ("policy", "hash", "limit", "cancel", "range"))
def test_primary_security_failures_never_start_curl_or_next_mirror(tmp_path, monkeypatch, failure):
    data = b"weights"
    urls = ("https" + "://models.invalid/a", "https" + "://models.invalid/b")
    spec = _spec(urls, data)
    calls, launches = [], []
    cancel = threading.Event()

    def primary(request, timeout):
        calls.append(request.full_url)
        if failure == "policy":
            raise mm._UnsafeDownloadURL("unsafe HTTPS redirect")
        if failure == "cancel":
            cancel.set()
        response = _MemoryResponse(b"x" * (len(data) + (failure == "limit")))
        if failure == "range":
            response.status = 206
            response.headers = {"Content-Range": "bytes 0-6/7"}
        return response

    def forbidden(*args, **kwargs):
        launches.append(True)
        raise AssertionError("terminal failure reached curl")

    monkeypatch.setattr(mm.subprocess, "Popen", forbidden)
    manager = _manager(tmp_path, spec, opener=primary, retries=2)
    if failure == "range":
        directory = manager.target_dir(spec)
        directory.mkdir(parents=True)
        (directory / (spec.filename + mm.PART_SUFFIX)).write_bytes(b"w")
    expected = (
        mm.ModelDownloadError if failure == "policy" else
        ModelCancelledError if failure == "cancel" else ModelIntegrityError
    )
    with pytest.raises(expected):
        manager.ensure_spec(spec, curl_fallback=True, cancel_check=cancel.is_set)
    assert calls == [next(iter(urls))]
    assert launches == []
    assert not (manager.target_dir(spec) / spec.filename).exists()


@pytest.mark.parametrize("failure", ("hash", "limit", "timeout", "policy"))
def test_curl_terminal_failures_never_try_another_origin(tmp_path, monkeypatch, failure):
    data = b"weights"
    urls = ("https" + "://models.invalid/a", "https" + "://models.invalid/b")
    spec = _spec(urls, data)
    scripts = {
        "hash": "import os; os.write(1, b'xxxxxxx')",
        "limit": "import os; os.write(1, b'xxxxxxxx')",
        "timeout": "import sys; sys.exit(28)",
        "policy": "import sys; sys.exit(1)",
    }
    processes, _commands = _controlled_curl(monkeypatch, scripts[failure])
    manager = _manager(tmp_path, spec, opener=_offline_transport, retries=0)
    expected = (
        ModelIntegrityError if failure in ("hash", "limit") else
        ModelNetworkError if failure == "timeout" else mm.ModelDownloadError
    )
    with pytest.raises(expected):
        manager.ensure_spec(spec, curl_fallback=True)
    assert len(processes) == 1
    assert _leftovers(manager.target_dir(spec)) == []
    _assert_curl_reaped(processes)


@pytest.mark.parametrize("failure", ("missing", "spawn", "exit"))
def test_curl_failure_preserves_primary_cause_and_bounds_diagnostics(tmp_path, monkeypatch, failure):
    spec = _spec(("https" + "://models.invalid/model.bin",), b"weights")
    manager = _manager(tmp_path, spec, opener=_offline_transport, retries=0)
    processes = []
    if failure == "missing":
        monkeypatch.setattr(mm.shutil, "which", lambda name: None)
    elif failure == "spawn":
        monkeypatch.setattr(mm.shutil, "which", lambda name: "/missing/curl")
    else:
        processes, _commands = _controlled_curl(
            monkeypatch,
            "import os, sys\n"
            "for _ in range(256):\n"
            "    os.write(2, b'D' * 4096)\n"
            "os.write(2, b'diagnostic-tail')\n"
            "sys.exit(22)\n",
        )
    with pytest.raises(ModelNetworkError) as caught:
        manager.ensure_spec(spec, curl_fallback=True)
    message = str(caught.value)
    assert "curl" in message
    assert "forced Python HTTPS failure" in message
    assert isinstance(caught.value.__cause__, ModelNetworkError)
    assert len(message) < 12_000
    if failure == "missing":
        assert "unavailable" in message
    elif failure == "spawn":
        assert "Cannot start" in message
    else:
        assert "status 22" in message and "diagnostic-tail" in message
        _assert_curl_reaped(processes)
    assert _leftovers(manager.target_dir(spec)) == []


def test_curl_fallback_keeps_existing_file_until_new_hash_verifies(tmp_path, monkeypatch):
    old, new = b"old verified model", b"replacement model"
    url = "https" + "://models.invalid/model.bin"
    old_spec = _spec((url,), old)
    spec = _spec((url,), new)
    manager = _manager(tmp_path, spec, opener=_offline_transport, retries=0)
    directory = manager.target_dir(spec)
    directory.mkdir(parents=True)
    installed = directory / spec.filename
    installed.write_bytes(old)
    manager._write_marker(old_spec, installed, {"sha256": _sha(old)})
    processes, _commands = _controlled_curl(monkeypatch, "import os; os.write(1, b'bad')")
    with pytest.raises(ModelIntegrityError):
        manager.ensure_spec(spec, curl_fallback=True)
    assert installed.read_bytes() == old
    _assert_curl_reaped(processes)


def test_runtime_local_import_and_disabled_download_do_not_start_curl(tmp_path, monkeypatch):
    from adapters import manga_48px_runtime as runtime

    data = b"local verified weights"
    url = "https" + "://models.invalid/model.bin"
    spec = _spec((url,), data)
    manager = _manager(tmp_path, spec, auto_download=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("offline/local path opened a transport")

    manager._open = forbidden
    monkeypatch.setattr(mm.subprocess, "Popen", forbidden)
    monkeypatch.setattr(mm, "default_manager", lambda: manager)
    monkeypatch.setattr(runtime, "PROJECT_ROOT", tmp_path / "empty-project")
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "empty-home")
    destination = tmp_path / "cache" / spec.filename
    with pytest.raises(ModelNotAvailableError):
        runtime._download(spec.urls, destination, sha256=spec.sha256, expected_size=spec.size)
    source = tmp_path / "manual.bin"
    source.write_bytes(data)
    monkeypatch.setenv("NOVEL_FORMATTER_MANGA_48PX_DICT_FILE", str(source))
    assert runtime._download(
        spec.urls, destination, sha256=spec.sha256, expected_size=spec.size,
    ).read_bytes() == data
    assert source.read_bytes() == data
    assert manager.ensure_spec(spec, target_dir=destination.parent, allow_download=False, curl_fallback=True) == destination


def _curl_download_in_process(root, spec, entered, release, waiting, counts, results):
    primary_slot, curl_slot = range(2)
    try:
        if waiting is not None:
            if os.name == "nt":
                import msvcrt as lock_module
                lock_name = "locking"
            else:
                import fcntl as lock_module
                lock_name = "flock"
            acquire = getattr(lock_module, lock_name)

            def observed_acquire(*args):
                try:
                    return acquire(*args)
                except OSError as exc:
                    if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        waiting.set()
                    raise

            setattr(lock_module, lock_name, observed_acquire)

        def primary(request, timeout):
            with counts.get_lock():
                counts[primary_slot] += 1
            return _offline_transport(request, timeout)

        processes = []

        def launch(command, **kwargs):
            with counts.get_lock():
                counts[curl_slot] += 1
            kwargs["stdin"] = mm.subprocess.PIPE
            process = _REAL_POPEN(
                [
                    mm.sys.executable, "-u", "-c",
                    "import sys; sys.stdin.buffer.read(1); sys.stdout.buffer.write(b'concurrent weights')",
                ],
                **kwargs,
            )
            processes.append(process)
            entered.set()
            if not release.wait(10):
                process.kill()
                process.wait()
                raise AssertionError("parent did not release controlled subprocess")
            process.stdin.write(b"x")
            process.stdin.close()
            return process

        mm.shutil.which = lambda name: mm.sys.executable
        mm.subprocess.Popen = launch
        manager = _manager(root, spec, opener=primary, retries=0, timeout=10, lock_timeout=10)
        path = manager.ensure_spec(spec, curl_fallback=True)
        if processes:
            _assert_curl_reaped(processes)
        results.put(("ok", str(path), _sha(path.read_bytes())))
    except BaseException as exc:
        results.put(("error", type(exc).__name__, str(exc)))


def test_curl_fallback_two_processes_download_once_under_same_lock(tmp_path):
    data = b"concurrent weights"
    spec = _spec(("https" + "://models.invalid/model.bin",), data)
    context = multiprocessing.get_context("spawn")
    entered, release, waiting = context.Event(), context.Event(), context.Event()
    counts = context.Array("i", (0, 0))
    results = context.Queue()
    first = context.Process(
        target=_curl_download_in_process,
        args=(tmp_path, spec, entered, release, None, counts, results),
    )
    second = context.Process(
        target=_curl_download_in_process,
        args=(tmp_path, spec, entered, release, waiting, counts, results),
    )
    processes = [first, second]
    manager = _manager(tmp_path, spec)
    directory = manager.target_dir(spec)
    lock = directory / f".{spec.model_id}{mm.LOCK_SUFFIX}"
    try:
        first.start()
        assert entered.wait(10), "first process did not start fallback"
        original_inode = lock.stat().st_ino
        second.start()
        assert waiting.wait(10), "second process did not contend on the actual OS lock"
        with counts.get_lock():
            assert tuple(counts) == (1, 1)
        release.set()
        for process in processes:
            process.join(15)
        assert all(not process.is_alive() and process.exitcode == 0 for process in processes)
        received = [results.get(timeout=3) for _ in processes]
        assert all(status == "ok" for status, _path, _digest in received), received
        assert {path for _status, path, _digest in received} == {str(directory / spec.filename)}
        assert {digest for _status, _path, digest in received} == {spec.sha256}
        assert tuple(counts) == (1, 1)
        assert lock.stat().st_ino == original_inode
        assert manager.verify(spec.model_id) is ModelStatus.READY
        assert _leftovers(directory) == []
    finally:
        release.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
            if process.pid is not None:
                process.join(5)
        results.close()
        results.join_thread()


@pytest.fixture(scope="module")
def curl_test_certificate(tmp_path_factory):
    import ssl

    curl = mm.shutil.which("curl")
    openssl = mm.shutil.which("openssl")
    if not curl or not openssl:
        pytest.skip("real curl TLS tests require curl and openssl executables")
    root = tmp_path_factory.mktemp("curl-local-tls")
    cert, key = root / "cert.pem", root / "key.pem"
    result = mm.subprocess.run(
        [
            openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-keyout", str(key), "-out", str(cert), "-days", "1",
            "-subj", "/CN=127.0.0.1",
            "-addext", "subjectAltName=IP:127.0.0.1",
        ],
        capture_output=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    return curl, cert, context


@pytest.fixture()
def local_tls_server(curl_test_certificate):
    _curl, _cert, context = curl_test_certificate
    instance = LocalServer()
    instance.httpd.socket = context.wrap_socket(instance.httpd.socket, server_side=True)
    instance.thread.start()
    try:
        yield instance
    finally:
        instance.httpd.shutdown()
        instance.httpd.server_close()


@pytest.mark.parametrize(
    "scenario", ("success", "https_redirect", "downgrade", "credentials", "untrusted", "oversized"),
)
def test_real_curl_local_tls_and_streaming_policy(
    tmp_path, monkeypatch, server, local_tls_server, curl_test_certificate, scenario,
):
    curl, cert, _context = curl_test_certificate
    data = b"real curl verified weights"
    body = data + b"extra" if scenario == "oversized" else data
    destination_url = server.add("/forbidden", data=data)
    route = (
        dict(redirect=destination_url) if scenario == "downgrade"
        else dict(data=body, unknown_size=scenario == "oversized")
    )
    if scenario in ("https_redirect", "credentials"):
        redirect = local_tls_server.add("/final", data=data).replace("http:", "https:", 1)
        if scenario == "credentials":
            redirect = redirect.replace("://", "://user:password@", 1)
        route = dict(redirect=redirect)
    url = local_tls_server.add("/weights", **route).replace("http:", "https:", 1)
    spec = _spec((url,), data)
    processes, commands = [], []
    home = tmp_path / "curl-home"
    home.mkdir()
    # If -q is lost, this config would disable verification and bypass our writer.
    unwanted_output = tmp_path / "curlrc-output"
    (home / ".curlrc").write_text(
        f'insecure\noutput = "{unwanted_output}"\n', encoding="utf-8",
    )

    def launch(command, **kwargs):
        assert next(iter(command)) == curl
        commands.append(list(command))
        command = list(command) + ["--noproxy", "*"]
        if scenario != "untrusted":
            command.extend(["--cacert", str(cert)])
        environment = dict(os.environ, HOME=str(home), CURL_HOME=str(home))
        for name in ("CURL_CA_BUNDLE", "SSL_CERT_FILE", "SSL_CERT_DIR"):
            environment.pop(name, None)
        kwargs["env"] = environment
        process = _REAL_POPEN(command, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(mm.subprocess, "Popen", launch)
    manager = _manager(tmp_path, spec, opener=_offline_transport, retries=0, timeout=5)
    if scenario in ("success", "https_redirect"):
        result = manager.ensure_spec(spec, curl_fallback=True)
        assert result.read_bytes() == data
        assert manager.verify(spec.model_id) is ModelStatus.READY
    else:
        expected = {
            "downgrade": mm.ModelDownloadError,
            "credentials": mm.ModelDownloadError,
            "untrusted": ModelNetworkError,
            "oversized": ModelIntegrityError,
        }[scenario]
        with pytest.raises(expected) as caught:
            manager.ensure_spec(spec, curl_fallback=True)
        if scenario == "downgrade":
            assert "status 1" in str(caught.value)
        elif scenario == "credentials":
            assert "status 67" in str(caught.value)
        elif scenario == "untrusted":
            assert "status 60" in str(caught.value)
        else:
            assert "exceeds the limit" in str(caught.value)
        assert not (manager.target_dir(spec) / spec.filename).exists()
    assert len(commands) == 1
    assert not unwanted_output.exists()
    assert server.requests == [], "HTTPS downgrade contacted the HTTP destination"
    expected_requests = [] if scenario == "untrusted" else [("/weights", None)]
    if scenario == "https_redirect":
        expected_requests.append(("/final", None))
    assert local_tls_server.requests == expected_requests
    assert _leftovers(manager.target_dir(spec)) == []
    _assert_curl_reaped(processes)


def test_curl_progress_does_not_regress_after_partial_primary_response(tmp_path, monkeypatch):
    data = b"complete weights"
    spec = _spec(("https" + "://models.invalid/model.bin",), data)

    class InterruptedResponse(_MemoryResponse):
        def read(self, size=-1):
            if self.tell():
                raise OSError("primary connection interrupted")
            return super().read(8)

    manager = _manager(
        tmp_path, spec, opener=lambda request, timeout: InterruptedResponse(data), retries=0,
    )
    processes, _commands = _controlled_curl(
        monkeypatch, "import os; os.write(1, b'complete weights')",
    )
    progress = []
    path = manager.ensure_spec(
        spec, curl_fallback=True, progress_callback=lambda done, total: progress.append(done),
    )
    assert path.read_bytes() == data
    assert 8 in progress
    assert all(before <= after for before, after in pairwise(progress))
    assert next(reversed(progress)) == len(data)
    _assert_curl_reaped(processes)


def test_curl_timeout_after_stdout_eof_still_reaps_process(tmp_path, monkeypatch):
    data = b"weights"
    spec = _spec(("https" + "://models.invalid/model.bin",), data)
    manager = _manager(tmp_path, spec, opener=_offline_transport, retries=0, timeout=0.5)
    processes, _commands = _controlled_curl(
        monkeypatch,
        "import os, threading\nos.write(1, b'weights')\nos.close(1)\nthreading.Event().wait(60)\n",
    )
    with pytest.raises(ModelNetworkError, match="timed out"):
        manager.ensure_spec(spec, curl_fallback=True)
    assert not (manager.target_dir(spec) / spec.filename).exists()
    assert _leftovers(manager.target_dir(spec)) == []
    _assert_curl_reaped(processes)


def test_curl_http_failure_restarts_next_origin_without_appending(tmp_path, monkeypatch):
    data = b"verified mirror"
    urls = ("https" + "://models.invalid/first", "https" + "://models.invalid/second")
    spec = _spec(urls, data)
    scripts = iter((
        "import os, sys; os.write(1, b'bad partial'); sys.exit(22)",
        "import os; os.write(1, b'verified mirror')",
    ))
    processes, commands = [], []

    def launch(command, **kwargs):
        commands.append(command)
        process = _REAL_POPEN([mm.sys.executable, "-u", "-c", next(scripts)], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(mm.shutil, "which", lambda name: mm.sys.executable)
    monkeypatch.setattr(mm.subprocess, "Popen", launch)
    manager = _manager(tmp_path, spec, opener=_offline_transport, retries=0)
    path = manager.ensure_spec(spec, curl_fallback=True)
    assert _sha(path.read_bytes()) == spec.sha256
    assert tuple(next(reversed(command)) for command in commands) == urls
    assert _leftovers(path.parent) == []
    _assert_curl_reaped(processes)


def test_runtime_curl_cancellation_preserves_public_exception(tmp_path, monkeypatch):
    from adapters import manga_48px_runtime as runtime

    data = b"weights"
    spec = _spec(("https" + "://models.invalid/model.bin",), data)
    manager = _manager(tmp_path, spec, opener=_offline_transport, retries=0)
    monkeypatch.setattr(mm, "default_manager", lambda: manager)
    monkeypatch.setattr(runtime, "_seed_from_local", lambda *args, **kwargs: False)
    processes, _commands = _controlled_curl(
        monkeypatch, "import os, threading; os.write(1, b'x'); threading.Event().wait(60)",
    )
    cancel = threading.Event()

    def progress(stage, done, total, detail):
        if stage == "download" and done:
            cancel.set()

    with pytest.raises(runtime.DownloadCancelled) as caught:
        runtime._download(
            spec.urls, tmp_path / spec.filename, sha256=spec.sha256, expected_size=spec.size,
            progress_callback=progress, cancel_check=cancel.is_set,
        )
    assert isinstance(caught.value.__cause__, ModelCancelledError)
    assert _leftovers(tmp_path) == []
    _assert_curl_reaped(processes)
