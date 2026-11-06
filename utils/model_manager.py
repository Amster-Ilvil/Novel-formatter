#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified ML model cache: locate -> download -> verify -> use.

Every downloadable OCR model goes through one code path:

1. look for an already installed copy (configured models folder, then the
   legacy project locations so existing installations keep working);
2. verify it (size + SHA-256/SHA-384); a ``.verified`` marker keyed on
   ``(size, mtime_ns, digest)`` avoids re-hashing large checkpoints on every
   start;
3. otherwise download into ``<name>.part`` in the same directory, streaming in
   1 MiB blocks while hashing, with timeouts, retries with backoff, mirror
   fallback and HTTP ``Range`` resume;
4. verify, then atomically ``os.replace`` into place; archives are extracted
   through :mod:`utils.safe_archive` into a temporary directory that is
   atomically renamed afterwards.

A corrupt or partial download is never installed under its final name.
An existing file is preserved until its replacement verifies. Concurrent
callers are serialised per model with a :class:`threading.Lock` plus an
OS-managed inter-process lock held on an open file descriptor.

This module is stdlib-only and must not import Qt.
"""
from __future__ import annotations

import enum
import errno
import hashlib
import http.client
import ipaddress
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Literal

from utils.safe_archive import UnsafeArchiveError, safe_extract_zip, safe_member_parts

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR_NAME = "Novel-formatter"
ENV_MODELS_DIR = "NOVEL_FORMATTER_MODELS_DIR"
ENV_AUTO_DOWNLOAD = "NOVEL_FORMATTER_MODEL_AUTO_DOWNLOAD"
# Directory used by earlier releases for downloaded weights.  When it exists it
# stays the default so already installed models are found without a download.
LEGACY_MODELS_DIR = PROJECT_ROOT / ".model-cache"

CHUNK_SIZE = 1024 * 1024
MAX_DOWNLOAD_BYTES = 4 * 1024 * 1024 * 1024
PART_SUFFIX = ".part"
LOCK_SUFFIX = ".lock"
VERIFIED_SUFFIX = ".verified"
USER_AGENT = "Novel-Formatter-ModelManager/1.0"
PIN_DATE = "2026-11-05"
_ARCHIVE_SUFFIX_RE = re.compile(r"(?i)(\.tar\.gz|\.tgz|\.tar\.xz|\.tar\.bz2|\.tar|\.zip)$")

ArchiveKind = Literal["none", "zip", "tar"]
ProgressCallback = Callable[[int, "int | None"], None]
CancelCheck = Callable[[], bool]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ModelError(RuntimeError):
    """Base class for all model-management failures (human-readable message)."""

    def __init__(
        self,
        message: str,
        *,
        model_id: str = "",
        url: str = "",
        directory: Path | str | None = None,
    ) -> None:
        super().__init__(message)
        self.model_id = model_id
        self.url = url
        self.directory = Path(directory) if directory else None


class ModelNetworkError(ModelError):
    """No network connection, DNS failure, refused connection or timeout."""


class ModelDownloadError(ModelError):
    """The server answered with an HTTP error status."""

    def __init__(self, message: str, *, status: int | None = None, **kwargs) -> None:
        super().__init__(message, **kwargs)
        self.status = status


class ModelIntegrityError(ModelError):
    """Size, hash, archive layout or archive safety check failed."""


class ModelCancelledError(ModelError):
    """The caller's ``cancel_check`` requested cancellation."""


class ModelNotAvailableError(ModelError):
    """The model is missing and downloading is not allowed."""


class _TransientError(Exception):
    """Internal: retryable network-level failure."""


class _TransientTimeout(_TransientError):
    """Internal: a request or response read exceeded the configured timeout."""


class _UnsafeDownloadURL(ValueError):
    """Internal: reject a URL before opening its transport."""


def _validate_download_url(url: str, *, redirect_from: str | None = None) -> None:
    parsed = urllib.parse.urlsplit(url)
    if not parsed.hostname or parsed.username is not None or parsed.password is not None:
        raise _UnsafeDownloadURL("download URL must have a host and no embedded credentials")
    if parsed.scheme == "https":
        return
    if redirect_from is not None and urllib.parse.urlsplit(redirect_from).scheme == "https":
        raise _UnsafeDownloadURL("unsafe HTTPS redirect: the target must also use HTTPS")
    if parsed.scheme == "http":
        # Numeric loopback addresses only: no arbitrary HTTP hosts or DNS aliases.
        try:
            if ipaddress.ip_address(parsed.hostname).is_loopback:
                return
        except ValueError:
            pass
    raise _UnsafeDownloadURL("download URL must use HTTPS (HTTP is limited to numeric loopback)")


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_download_url(newurl, redirect_from=req.full_url)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _HTTPStatusError(Exception):
    """Internal: HTTP error status from the server."""

    def __init__(self, status: int, reason: str = "") -> None:
        super().__init__(f"HTTP {status} {reason}".strip())
        self.status = int(status)

    @property
    def retryable(self) -> bool:
        return self.status in {408, 425, 429} or self.status >= 500


class _CurlResponse:
    """Bounded, cancellable stdout stream; never gives curl a filesystem path."""

    status = 200

    def __init__(self, url, *, timeout, check_cancel, context):
        self.url = url
        self.timeout = timeout
        self.check_cancel = check_cancel
        self.context = context
        self.headers = {}
        self._queue = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._pending = b""
        self._stderr = bytearray()
        self._threads = []
        self._proc = None
        self._started = self._activity = time.monotonic()

    def __enter__(self):
        self.check_cancel()
        curl = shutil.which("curl")
        if not curl:
            raise ModelNetworkError(
                "System curl is unavailable; the Python HTTPS transport also failed.",
                **self.context,
            )
        # -q must be first: a user curlrc must not enable insecure TLS, redirects
        # to other protocols, extra transfers, or writing directly to a file.
        command = [
            curl, "-q",
            "--proto", "=https", "--proto-redir", "=https",
            "--disallow-username-in-url", "--globoff",
            "--location", "--max-redirs", "10", "--fail",
            "--silent", "--show-error",
            "--connect-timeout", str(self.timeout), "--max-time", "1800",
            "--output", "-", "--url", self.url,
        ]
        try:
            self._proc = subprocess.Popen(
                command, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                bufsize=0, shell=False,
            )
        except OSError as exc:
            raise ModelNetworkError(f"Cannot start system curl: {exc}", **self.context) from exc
        self._started = self._activity = time.monotonic()
        try:
            for target, name in (
                (self._read_stdout, "model-curl-stdout"),
                (self._read_stderr, "model-curl-stderr"),
            ):
                thread = threading.Thread(target=target, name=name, daemon=True)
                thread.start()
                self._threads.append(thread)
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def _put(self, item):
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.05)
                return
            except queue.Full:
                pass

    def _read_stdout(self):
        try:
            while not self._stop.is_set():
                chunk = self._proc.stdout.read(64 * 1024)
                if not chunk:
                    break
                self._put(chunk)
        except Exception as exc:
            self._put(exc)
        finally:
            self._put(None)

    def _read_stderr(self):
        try:
            while not self._stop.is_set():
                chunk = self._proc.stderr.read(4096)
                if not chunk:
                    break
                self._stderr.extend(chunk)
                del self._stderr[:-8192]
        except OSError as exc:
            self._put(exc)

    def _check(self):
        self.check_cancel()
        now = time.monotonic()
        if now - self._activity >= self.timeout or now - self._started >= 1800:
            raise ModelNetworkError(
                f"System curl timed out (inactivity timeout {self.timeout:g} seconds; "
                "total timeout 1800 seconds).",
                **self.context,
            )

    def read(self, size):
        self._check()
        while not self._pending:
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                self._check()
                continue
            if isinstance(item, Exception):
                raise ModelNetworkError(f"Cannot read system curl output: {item}", **self.context) from item
            if item is None:
                while self._proc.poll() is None:
                    self._check()
                    self._stop.wait(0.05)
                for thread in self._threads:
                    while thread.is_alive():
                        thread.join(0.05)
                        self._check()
                code = self._proc.returncode
                if code:
                    detail = self._stderr.decode("utf-8", errors="replace").strip()
                    message = f"System curl exited with status {code}: {detail}"
                    if code == 28:
                        raise ModelNetworkError(f"System curl timed out. {message}", **self.context)
                    if code in (1, 3, 47, 67):
                        raise ModelDownloadError(message, **self.context)
                    raise _TransientError(message)
                return b""
            self._pending = item
            self._activity = time.monotonic()
        chunk = self._pending[:size]
        self._pending = self._pending[size:]
        return chunk

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        if self._proc is not None:
            try:
                if self._proc.poll() is None:
                    self._proc.terminate()
                try:
                    self._proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait()
            finally:
                for thread in self._threads:
                    thread.join()
                self._proc.stdout.close()
                self._proc.stderr.close()


# ---------------------------------------------------------------------------
# Specs and registry
# ---------------------------------------------------------------------------


class ModelStatus(enum.Enum):
    READY = "ready"
    MISSING = "missing"
    CORRUPT = "corrupt"


@dataclass(frozen=True)
class ModelSpec:
    """Immutable description of one downloadable model artifact.

    ``filename`` is the downloaded file.  For archives the installed artifact is
    the directory ``install_name`` next to it and ``required_files`` (relative
    to that directory) must exist after extraction.  ``subdir`` groups files of
    one engine below the models folder; ``legacy_dirs`` (relative to the
    project root) are searched for existing installations.
    """

    model_id: str
    engine: str
    urls: tuple[str, ...]
    filename: str
    sha256: str | None
    size: int | None
    archive: ArchiveKind = "none"
    required_files: tuple[str, ...] = ()
    hash_source: str = ""
    subdir: str = ""
    sha384: str | None = None
    legacy_dirs: tuple[str, ...] = ()
    min_size: int = 1
    label: str = ""
    curl_fallback: bool = False

    @property
    def directory_name(self) -> str:
        return self.subdir or self.model_id

    @property
    def install_name(self) -> str:
        if self.archive == "none":
            return self.filename
        stripped = _ARCHIVE_SUFFIX_RE.sub("", self.filename)
        if stripped == self.filename:
            return self.filename + ".d"
        return stripped or self.model_id

    @property
    def display_name(self) -> str:
        return self.label or self.model_id

    @property
    def primary_url(self) -> str:
        return next(iter(self.urls), "(no known download URL)")


def _https(host: str, path: str) -> str:
    """Assemble an HTTPS URL from host and path (keeps mirror lists compact)."""
    return "https" + "://" + host + path


def _short(digest: str) -> str:
    return f"{digest:.16}..."


_HF_48PX_REVISION = "3e29cd63a0ce7d1b4013b0a6e56da4cddaf4fe5b"
_HF_48PX = f"/zyddnys/manga-image-translator/resolve/{_HF_48PX_REVISION}"
_GH_48PX = "/zyddnys/manga-image-translator/releases/download/beta-0.3"
_OV_NAME = "handwritten-japanese-recognition-0001"
_OV_ROOT = f"/repositories/open_model_zoo/2023.0/models_bin/1/{_OV_NAME}/FP16-INT8"
_OMZ_CHARLIST = (
    "/openvinotoolkit/open_model_zoo/7cc29a91472b4cb1289a11e655ba3e188e1d4a31"
    "/data/dataset_classes/kondate_nakayosi.txt"
)

MODEL_REGISTRY: dict[str, ModelSpec] = {
    spec.model_id: spec
    for spec in (
        ModelSpec(
            model_id="manga_48px.checkpoint",
            engine="manga_48px",
            urls=(
                _https("huggingface.co", f"{_HF_48PX}/ocr_ar_48px.ckpt?download=true"),
                _https("github.com", f"{_GH_48PX}/ocr_ar_48px.ckpt"),
            ),
            filename="ocr_ar_48px.ckpt",
            sha256="29daa46d080818bb4ab239a518a88338cbccff8f901bef8c9db191a7cb97671d",
            size=204_290_192,
            hash_source="existing-code",
            subdir="manga-48px-ar",
            legacy_dirs=(".model-cache/manga-48px-ar",),
            min_size=100_000_000,
            label="48px AR checkpoint",
            curl_fallback=True,
        ),
        ModelSpec(
            model_id="manga_48px.dictionary",
            engine="manga_48px",
            urls=(
                _https("huggingface.co", f"{_HF_48PX}/alphabet-all-v7.txt?download=true"),
                _https("github.com", f"{_GH_48PX}/alphabet-all-v7.txt"),
            ),
            filename="alphabet-all-v7.txt",
            sha256="f5722368146aa0fbcc9f4726866e4efc3203318ebb66c811d8cbbe915576538a",
            size=186_651,
            hash_source="existing-code",
            subdir="manga-48px-ar",
            legacy_dirs=(".model-cache/manga-48px-ar",),
            min_size=1_000,
            label="48px AR dictionary",
            curl_fallback=True,
        ),
        ModelSpec(
            model_id="openvino_handwriting.xml",
            engine="openvino_handwriting",
            urls=(_https("storage.openvinotoolkit.org", f"{_OV_ROOT}/{_OV_NAME}.xml"),),
            filename=f"{_OV_NAME}.xml",
            sha256="12922031ae75163bc19c82e3ddc9dd8a5567ec50b2b62b72244dfc129ab820e4",
            size=97_226,
            hash_source=f"sha384: existing-code; sha256: pinned-from-official-download {PIN_DATE}",
            subdir="handwriting_openvino",
            sha384=(
                "b8cac8a7a57ec8e741d336517e7406333aef866cf5976e28ef44f8f5"
                "afb43b116fc657050592229d28a932c9f1ed77c2"
            ),
            legacy_dirs=("models/handwriting_openvino",),
            label="OpenVINO handwriting model (xml)",
        ),
        ModelSpec(
            model_id="openvino_handwriting.bin",
            engine="openvino_handwriting",
            urls=(_https("storage.openvinotoolkit.org", f"{_OV_ROOT}/{_OV_NAME}.bin"),),
            filename=f"{_OV_NAME}.bin",
            sha256="afba3ba73b2d8e5a7a8f13107c77800fd3a96634cf37e84f1e0256e1b4b443a8",
            size=17_035_726,
            hash_source=f"sha384: existing-code; sha256: pinned-from-official-download {PIN_DATE}",
            subdir="handwriting_openvino",
            sha384=(
                "2120d799e261f6abbf46215de79592c3c334d955a8220c6e7c74791ce"
                "7d6662292080d90df593865c145d40516df6c7b"
            ),
            legacy_dirs=("models/handwriting_openvino",),
            label="OpenVINO handwriting model (bin)",
        ),
        ModelSpec(
            model_id="openvino_handwriting.charlist",
            engine="openvino_handwriting",
            urls=(_https("raw.githubusercontent.com", _OMZ_CHARLIST),),
            filename="kondate_nakayosi.txt",
            sha256="9851b52c412e2d8ae8a6ab3f9eaa69a53c7b3c9b87c2b5dd75f2ed8d7312ddf0",
            size=17_711,
            hash_source=f"pinned-from-official-download {PIN_DATE}",
            subdir="handwriting_openvino",
            legacy_dirs=("models/handwriting_openvino",),
            label="OpenVINO handwriting character list",
        ),
    )
}


def get_spec(model_id: str, registry: dict[str, ModelSpec] | None = None) -> ModelSpec:
    table = MODEL_REGISTRY if registry is None else registry
    spec = table.get(model_id)
    if spec is None:
        raise ModelError(f"Unknown model id {model_id!r}.", model_id=model_id)
    return spec


def specs_for_engine(engine: str, registry: dict[str, ModelSpec] | None = None) -> tuple[ModelSpec, ...]:
    table = MODEL_REGISTRY if registry is None else registry
    return tuple(spec for spec in table.values() if spec.engine == engine)


# ---------------------------------------------------------------------------
# Preferences and directory resolution
# ---------------------------------------------------------------------------

_PREFERENCES_LOCK = threading.Lock()
_PREFERENCES: dict[str, object] = {"models_dir": None, "auto_download": None}
_ORIGINAL_ENV = {
    ENV_MODELS_DIR: os.environ.get(ENV_MODELS_DIR),
    ENV_AUTO_DOWNLOAD: os.environ.get(ENV_AUTO_DOWNLOAD),
}


def _restore_env(name: str) -> None:
    original = _ORIGINAL_ENV.get(name)
    if original is None:
        os.environ.pop(name, None)
    else:
        os.environ.update({name: original})


def configure(*, models_dir: str | os.PathLike[str] | None = None, auto_download: bool | None = None) -> None:
    """Apply the user's Settings for this process and its worker subprocesses.

    An empty ``models_dir`` means "use the default".  The values are mirrored
    into environment variables so OCR worker processes resolve the same folder.
    """
    with _PREFERENCES_LOCK:
        text = str(models_dir or "").strip()
        _PREFERENCES.update({"models_dir": text or None})
        if text:
            os.environ.update({ENV_MODELS_DIR: str(Path(text).expanduser())})
        else:
            _restore_env(ENV_MODELS_DIR)
        _PREFERENCES.update({"auto_download": None if auto_download is None else bool(auto_download)})
        if auto_download is None:
            _restore_env(ENV_AUTO_DOWNLOAD)
        else:
            os.environ.update({ENV_AUTO_DOWNLOAD: "1" if auto_download else "0"})


def configured_models_dir() -> str | None:
    """Explicit models folder from Settings (``None`` when using defaults)."""
    with _PREFERENCES_LOCK:
        value = _PREFERENCES.get("models_dir")
    return str(value) if value else None


def auto_download_enabled() -> bool:
    """Settings value, then ``NOVEL_FORMATTER_MODEL_AUTO_DOWNLOAD``; default on."""
    with _PREFERENCES_LOCK:
        value = _PREFERENCES.get("auto_download")
    if value is not None:
        return bool(value)
    raw = str(os.environ.get(ENV_AUTO_DOWNLOAD, "") or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


_EXPLICIT_REQUEST = threading.local()


@contextmanager
def user_requested_download() -> Iterator[None]:
    """Mark downloads in this thread as explicitly requested by the user.

    Used by "download model" buttons, manual updates and first deployment so
    they keep working when automatic downloads are disabled in Settings.
    ``allow_download=False`` still wins.
    """
    previous = bool(getattr(_EXPLICIT_REQUEST, "active", False))
    _EXPLICIT_REQUEST.active = True
    try:
        yield
    finally:
        _EXPLICIT_REQUEST.active = previous


def download_explicitly_requested() -> bool:
    return bool(getattr(_EXPLICIT_REQUEST, "active", False))


def explicit_models_dir_configured() -> bool:
    """True when Settings or the environment chose a models folder explicitly."""
    return bool(configured_models_dir() or str(os.environ.get(ENV_MODELS_DIR, "") or "").strip())


def default_platform_models_dir() -> Path:
    """Per-user cache location following each platform's conventions."""
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Caches" / APP_DIR_NAME / "models"
    if sys.platform.startswith("win"):
        local = str(os.environ.get("LOCALAPPDATA", "") or "").strip()
        base = Path(local) if local else home / "AppData" / "Local"
        return base / APP_DIR_NAME / "models"
    xdg = str(os.environ.get("XDG_CACHE_HOME", "") or "").strip()
    base = Path(xdg).expanduser() if xdg else home / ".cache"
    return base / APP_DIR_NAME.lower() / "models"


def resolve_models_dir(settings_path: str | None = None) -> Path:
    """Return the models folder.

    Priority: explicit ``settings_path`` (or the value applied via
    :func:`configure`) > ``NOVEL_FORMATTER_MODELS_DIR`` > existing legacy
    project cache > platform cache directory.
    """
    explicit = str(settings_path or "").strip() or (configured_models_dir() or "")
    if explicit:
        return Path(explicit).expanduser()
    env_value = str(os.environ.get(ENV_MODELS_DIR, "") or "").strip()
    if env_value:
        return Path(env_value).expanduser()
    if LEGACY_MODELS_DIR.is_dir():
        return LEGACY_MODELS_DIR
    return default_platform_models_dir()


def directory_ready(directory: Path, *, pattern: str, min_count: int, min_size: int) -> bool:
    """Presence/size check for externally managed model folders (no hashes)."""
    try:
        files = [path for path in Path(directory).glob(pattern) if path.is_file()]
        return len(files) >= min_count and all(path.stat().st_size >= min_size for path in files)
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _thread_lock_for(key: str) -> threading.Lock:
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.Lock())


def _safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value) or "model"


def _cancelled(cancel_check: CancelCheck | None) -> bool:
    if cancel_check is None:
        return False
    try:
        return bool(cancel_check())
    except Exception:
        return False


def _response_total(response, offset: int) -> int | None:
    content_range = str(response.headers.get("Content-Range") or "")
    match = re.search(r"/(\d+)\s*$", content_range)
    if match:
        return int(match.group(1))
    try:
        length = int(response.headers.get("Content-Length") or -1)
    except (TypeError, ValueError):
        length = -1
    if length < 0:
        return None
    status = int(getattr(response, "status", 200) or 200)
    return offset + length if status == 206 else length


class ModelManager:
    """Locate, download and verify models described by :class:`ModelSpec`."""

    def __init__(
        self,
        models_dir: str | os.PathLike[str] | None = None,
        *,
        registry: dict[str, ModelSpec] | None = None,
        auto_download: bool | None = None,
        timeout: float = 30.0,
        retries: int = 2,
        backoff: float = 1.0,
        lock_timeout: float = 3600.0,
        stale_lock_seconds: float = 900.0,
        chunk_size: int = CHUNK_SIZE,
        max_download_bytes: int = MAX_DOWNLOAD_BYTES,
        opener: Callable[..., object] | None = None,
        project_root: Path | None = None,
    ) -> None:
        self._models_dir = Path(models_dir).expanduser() if models_dir else None
        self.registry = MODEL_REGISTRY if registry is None else registry
        self._auto_download = auto_download
        self.timeout = float(timeout)
        self.retries = max(0, int(retries))
        self.backoff = max(0.0, float(backoff))
        self.lock_timeout = float(lock_timeout)
        # Retained for constructor compatibility; OS locks do not expire by age.
        self.stale_lock_seconds = float(stale_lock_seconds)
        self.chunk_size = max(1, int(chunk_size))
        self.max_download_bytes = int(max_download_bytes)
        if self.max_download_bytes <= 0:
            raise ValueError("max_download_bytes must be positive")
        # An explicitly injected opener is a trusted transport, used by tests.
        self._open = opener or urllib.request.build_opener(_SafeRedirectHandler()).open
        self.project_root = Path(project_root) if project_root else PROJECT_ROOT

    # -- configuration -----------------------------------------------------

    @property
    def models_dir(self) -> Path:
        return self._models_dir if self._models_dir is not None else resolve_models_dir()

    @property
    def auto_download(self) -> bool:
        if download_explicitly_requested():
            return True
        return auto_download_enabled() if self._auto_download is None else bool(self._auto_download)

    def spec(self, model_id: str) -> ModelSpec:
        return get_spec(model_id, self.registry)

    def target_dir(self, model: str | ModelSpec) -> Path:
        spec = self.spec(model) if isinstance(model, str) else model
        return self.models_dir / spec.directory_name

    @staticmethod
    def installed_path(spec: ModelSpec, directory: Path) -> Path:
        return Path(directory) / spec.install_name

    def _candidate_dirs(self, spec: ModelSpec, target_dir: Path | None) -> list[Path]:
        if target_dir is not None:
            return [Path(target_dir)]
        out: list[Path] = []
        seen: set[str] = set()
        for directory in [self.target_dir(spec)] + [self.project_root / item for item in spec.legacy_dirs]:
            key = os.path.normcase(os.path.abspath(directory))
            if key not in seen:
                seen.add(key)
                out.append(directory)
        return out

    # -- status --------------------------------------------------------------

    def status(
        self,
        model_id: str,
        *,
        target_dir: Path | None = None,
        deep: bool = True,
    ) -> ModelStatus:
        """READY if any candidate location holds a valid copy.

        ``deep=False`` skips hashing files that have no valid ``.verified``
        marker (size checks still apply); :meth:`ensure` always verifies.
        """
        return self.spec_status(self.spec(model_id), target_dir=target_dir, deep=deep)

    def spec_status(self, spec: ModelSpec, *, target_dir: Path | None = None, deep: bool = True) -> ModelStatus:
        result = ModelStatus.MISSING
        for directory in self._candidate_dirs(spec, target_dir):
            state = self._check(spec, self.installed_path(spec, directory), deep=deep)
            if state is ModelStatus.READY:
                return state
            if state is ModelStatus.CORRUPT:
                result = state
        return result

    def verify(
        self, model_id: str, *, force: bool = True, target_dir: Path | None = None,
    ) -> ModelStatus:
        """Check a model without downloading; ``force`` bypasses file hash markers.

        Full verification of extracted archives is unavailable: their source
        archives are removed after installation and no per-file hashes are stored.
        """
        spec = self.spec(model_id)
        if force and spec.archive != "none":
            raise ModelError(
                f"Full verification of extracted model {spec.model_id} is unavailable: "
                f"the original archive and per-file hashes are not retained.",
                model_id=spec.model_id,
            )
        result = ModelStatus.MISSING
        for directory in self._candidate_dirs(spec, target_dir):
            state = self._check(spec, self.installed_path(spec, directory), deep=True, force=force)
            if state is ModelStatus.READY:
                return state
            if state is ModelStatus.CORRUPT:
                result = state
        return result

    def locate(self, model_id: str, *, target_dir: Path | None = None) -> Path | None:
        """Return the verified installed path without any network access."""
        return self._find_ready(self.spec(model_id), target_dir)

    def _find_ready(self, spec: ModelSpec, target_dir: Path | None) -> Path | None:
        for directory in self._candidate_dirs(spec, target_dir):
            path = self.installed_path(spec, directory)
            if self._check(spec, path, deep=True) is ModelStatus.READY:
                return path
        return None

    @staticmethod
    def _marker_path(path: Path) -> Path:
        if path.is_dir():
            return path / VERIFIED_SUFFIX
        return path.with_name(path.name + VERIFIED_SUFFIX)

    def _check(
        self, spec: ModelSpec, path: Path, *, deep: bool, force: bool = False,
    ) -> ModelStatus:
        if spec.archive != "none":
            return self._check_extracted(spec, path)
        try:
            stat = path.stat()
        except FileNotFoundError:
            return ModelStatus.MISSING
        except OSError:
            return ModelStatus.CORRUPT
        if not path.is_file():
            return ModelStatus.CORRUPT
        if spec.size is not None and stat.st_size != spec.size:
            return ModelStatus.CORRUPT
        if stat.st_size < max(1, spec.min_size):
            return ModelStatus.CORRUPT
        if not (spec.sha256 or spec.sha384):
            return ModelStatus.READY
        # .verified optimizes a trusted local cache; it is not a security boundary.
        if not force and self._marker_valid(spec, path, stat):
            return ModelStatus.READY
        if not deep and not force:
            return ModelStatus.READY
        try:
            digests = self._hash_file(spec, path)
        except OSError:
            return ModelStatus.CORRUPT
        if self._digest_problem(spec, stat.st_size, digests):
            return ModelStatus.CORRUPT
        self._write_marker(spec, path, digests)
        return ModelStatus.READY

    @staticmethod
    def _check_extracted(spec: ModelSpec, path: Path) -> ModelStatus:
        if not path.exists():
            return ModelStatus.MISSING
        if not path.is_dir():
            return ModelStatus.CORRUPT
        if any(not (path / item).is_file() for item in spec.required_files):
            return ModelStatus.CORRUPT
        if not spec.sha256:
            return ModelStatus.READY
        try:
            marker = json.loads((path / VERIFIED_SUFFIX).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ModelStatus.CORRUPT
        if str(marker.get("archive_sha256", "")).lower() != spec.sha256.lower():
            return ModelStatus.CORRUPT
        return ModelStatus.READY

    def _marker_valid(self, spec: ModelSpec, path: Path, stat: os.stat_result) -> bool:
        try:
            marker = json.loads(self._marker_path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if int(marker.get("size", -1)) != stat.st_size or int(marker.get("mtime_ns", -1)) != stat.st_mtime_ns:
            return False
        if spec.sha256 and str(marker.get("sha256", "")).lower() != spec.sha256.lower():
            return False
        if spec.sha384 and str(marker.get("sha384", "")).lower() != spec.sha384.lower():
            return False
        return True

    def _write_marker(self, spec: ModelSpec, path: Path, digests: dict[str, str]) -> None:
        temp = None
        try:
            stat = path.stat()
            payload = {
                "model_id": spec.model_id,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": digests.get("sha256", ""),
                "sha384": digests.get("sha384", ""),
            }
            marker = self._marker_path(path)
            fd, name = tempfile.mkstemp(
                prefix=f"{marker.name}.", suffix=".tmp", dir=marker.parent,
            )
            temp = Path(name)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, sort_keys=True))
            os.replace(temp, marker)
        except OSError:
            # A read-only legacy folder simply means we re-hash next time.
            pass
        finally:
            if temp is not None:
                try:
                    temp.unlink(missing_ok=True)
                except OSError:
                    pass

    # -- hashing ---------------------------------------------------------------

    @staticmethod
    def _new_hashers(spec: ModelSpec) -> dict:
        hashers = {"sha256": hashlib.sha256()}
        if spec.sha384:
            hashers.update({"sha384": hashlib.sha384()})
        return hashers

    def _hash_into(self, path: Path, hashers: dict) -> None:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(self.chunk_size), b""):
                for hasher in hashers.values():
                    hasher.update(chunk)

    def _hash_file(self, spec: ModelSpec, path: Path) -> dict[str, str]:
        hashers = self._new_hashers(spec)
        self._hash_into(path, hashers)
        return {name: hasher.hexdigest() for name, hasher in hashers.items()}

    @staticmethod
    def _digest_problem(spec: ModelSpec, size: int, digests: dict[str, str]) -> str:
        if spec.size is not None and size != spec.size:
            return f"size {size} bytes, expected {spec.size} bytes"
        if size < max(1, spec.min_size):
            return f"size {size} bytes is below the minimum of {spec.min_size} bytes"
        actual256 = digests.get("sha256", "")
        if spec.sha256 and actual256.lower() != spec.sha256.lower():
            return f"SHA-256 {_short(actual256)} != expected {_short(spec.sha256)}"
        actual384 = digests.get("sha384", "")
        if spec.sha384 and actual384.lower() != spec.sha384.lower():
            return f"SHA-384 {_short(actual384)} != expected {_short(spec.sha384)}"
        return ""

    # -- public ensure -----------------------------------------------------------

    @staticmethod
    def _destination_lock_key(spec: ModelSpec, directory: Path) -> str:
        return f"{spec.model_id}|{os.path.normcase(os.path.realpath(directory))}"

    def ensure(
        self,
        model_id: str,
        *,
        progress_callback: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
        allow_download: bool = True,
        target_dir: Path | None = None,
    ) -> Path:
        """Return the path of a verified model, downloading it if needed."""
        return self.ensure_spec(
            self.spec(model_id),
            progress_callback=progress_callback,
            cancel_check=cancel_check,
            allow_download=allow_download,
            target_dir=target_dir,
        )

    def ensure_spec(
        self,
        spec: ModelSpec,
        *,
        progress_callback: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
        allow_download: bool = True,
        target_dir: Path | None = None,
        curl_fallback: bool | None = None,
    ) -> Path:
        """Like :meth:`ensure` for a spec that is not (or differently) registered.

        ``target_dir`` pins the exact directory and disables legacy lookup.
        ``curl_fallback=None`` follows the registered artifact policy. An explicit
        bool overrides it. Only network/HTTP failures permit a fresh curl attempt,
        within the same lock, bounded writer and verification cycle.
        """
        if curl_fallback is None:
            curl_fallback = spec.curl_fallback
        directory = Path(target_dir) if target_dir is not None else self.target_dir(spec)
        lock_key = self._destination_lock_key(spec, directory)
        with _thread_lock_for(lock_key):
            found = self._find_ready(spec, target_dir)
            if found is not None:
                return found
            if not allow_download or not self.auto_download:
                raise self._not_available(spec, directory)
            if not spec.urls:
                raise self._not_available(spec, directory, reason="no download URL is known")
            directory.mkdir(parents=True, exist_ok=True)
            with self._process_lock(spec, directory, cancel_check) as heartbeat:
                # Another process may have finished while we waited for the lock.
                found = self._find_ready(spec, target_dir)
                if found is not None:
                    return found
                # Preserve the installed copy until a replacement has verified.
                return self._download_and_install(
                    spec, directory, progress_callback, cancel_check, heartbeat,
                    curl_fallback=curl_fallback,
                )

    def import_file(
        self,
        model_id_or_spec: str | ModelSpec,
        source: Path,
        *,
        target_dir: Path | None = None,
        move: bool = False,
    ) -> Path:
        """Copy, verify and atomically install under the same locks as ensure.

        With ``move=True``, delete the unchanged source only after installation.
        A source initially aliasing the destination is retained. If source cleanup
        fails or its identity/content metadata changed, raise ModelError reporting
        partial success; the verified installed model and its marker remain intact.
        """
        spec = self.spec(model_id_or_spec) if isinstance(model_id_or_spec, str) else model_id_or_spec
        if spec.archive != "none":
            raise ModelError(
                f"Model {spec.model_id} is an archive; place the extracted folder in the models folder instead.",
                model_id=spec.model_id,
            )
        source = Path(source)
        directory = Path(target_dir) if target_dir is not None else self.target_dir(spec)
        with _thread_lock_for(self._destination_lock_key(spec, directory)):
            directory.mkdir(parents=True, exist_ok=True)
            with self._process_lock(spec, directory, None):
                return self._import_file_locked(spec, source, directory, move=move)

    @staticmethod
    def _source_fingerprint(source: Path) -> tuple:
        # lstat also detects replacement of a symlink pointing to the same inode.
        return tuple(
            (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            for stat in (source.lstat(), source.stat())
        )

    def _import_file_locked(
        self, spec: ModelSpec, source: Path, directory: Path, *, move: bool,
    ) -> Path:
        final = self.installed_path(spec, directory)
        try:
            fingerprint = self._source_fingerprint(source) if move else None
            aliases_final = source.samefile(final) if final.exists() else False
        except OSError as exc:
            raise ModelError(
                f"Cannot read {source} for model {spec.model_id}: {exc}",
                model_id=spec.model_id, directory=directory,
            ) from exc
        fd, name = tempfile.mkstemp(
            prefix=f".{_safe_token(spec.filename)}.", suffix=".importing", dir=directory,
        )
        temp = Path(name)
        try:
            os.close(fd)
            shutil.copy2(source, temp)
            size = temp.stat().st_size
            digests = self._hash_file(spec, temp)
            problem = self._digest_problem(spec, size, digests)
            if problem:
                raise ModelIntegrityError(
                    f"File {source} does not match model {spec.model_id} ({problem}).",
                    model_id=spec.model_id, directory=directory,
                )
            os.replace(temp, final)
        finally:
            temp.unlink(missing_ok=True)
        self._write_marker(spec, final, digests)
        if move and not aliases_final:
            try:
                if self._source_fingerprint(source) != fingerprint or source.samefile(final):
                    raise OSError("source changed during import; it was not removed")
                source.unlink()
            except OSError as exc:
                raise ModelError(
                    f"Model {spec.model_id} is installed and verified at {final}, but source "
                    f"{source} could not be removed; move is incomplete: {exc}",
                    model_id=spec.model_id, directory=directory,
                ) from exc
        return final

    # -- messages ------------------------------------------------------------------

    @staticmethod
    def _not_available(spec: ModelSpec, directory: Path, *, reason: str = "") -> ModelNotAvailableError:
        why = reason or "automatic model download is disabled"
        return ModelNotAvailableError(
            f"Model {spec.model_id} is not installed in {directory} and {why}. "
            f"Enable 'Download models automatically when needed' in Settings, or download it manually "
            f"from {spec.primary_url} and place it in {directory}.",
            model_id=spec.model_id, url=spec.primary_url, directory=directory,
        )

    @staticmethod
    def _cancelled_error(spec: ModelSpec, directory: Path, url: str = "") -> ModelCancelledError:
        return ModelCancelledError(
            f"Download of model {spec.model_id} was cancelled.",
            model_id=spec.model_id, url=url, directory=directory,
        )

    # -- locking ----------------------------------------------------------------------

    @contextmanager
    def _process_lock(
        self, spec: ModelSpec, directory: Path, cancel_check: CancelCheck | None,
    ) -> Iterator[Callable[[], None]]:
        lock = directory / f".{_safe_token(spec.model_id)}{LOCK_SUFFIX}"
        deadline = time.monotonic() + self.lock_timeout
        fd = os.open(str(lock), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            while True:
                if _cancelled(cancel_check):
                    raise self._cancelled_error(spec, directory)
                try:
                    if os.name == "nt":
                        import msvcrt
                        os.lseek(fd, 0, os.SEEK_SET)
                        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    if time.monotonic() >= deadline:
                        raise ModelError(
                            f"Timed out waiting for another download of model {spec.model_id} "
                            f"after {self.lock_timeout:g} seconds (lock file {lock}). "
                            f"Wait for the other instance to finish; do not delete the lock file.",
                            model_id=spec.model_id, directory=directory,
                        ) from exc
                    time.sleep(0.1)

            # Keep the callback contract; an OS-held lock needs no heartbeat.
            yield lambda: None
        finally:
            # Closing (including process exit) releases the OS lock. Never unlink:
            # replacing its inode would let another process acquire a different lock.
            os.close(fd)

    # -- download --------------------------------------------------------------------

    def _sleep_backoff(self, attempt: int, spec: ModelSpec, directory: Path, cancel_check: CancelCheck | None) -> None:
        deadline = time.monotonic() + self.backoff * (2 ** (attempt - 1))
        while time.monotonic() < deadline:
            if _cancelled(cancel_check):
                raise self._cancelled_error(spec, directory)
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    def _download_and_install(
        self,
        spec: ModelSpec,
        directory: Path,
        progress_callback: ProgressCallback | None,
        cancel_check: CancelCheck | None,
        heartbeat: Callable[[], None],
        *,
        curl_fallback: bool = False,
    ) -> Path:
        reported = 0

        def report(done: int, total: int | None) -> None:
            nonlocal reported
            reported = max(reported, int(done))
            if progress_callback is not None:
                progress_callback(reported, total)

        try:
            return self._download_attempts(spec, directory, report, cancel_check, heartbeat)
        except (ModelNetworkError, ModelDownloadError) as primary:
            if _cancelled(cancel_check):
                raise self._cancelled_error(spec, directory) from primary
            if not curl_fallback or (
                isinstance(primary, ModelDownloadError) and primary.status is None
            ):
                raise
            part = directory / (spec.filename + PART_SUFFIX)
            try:
                return self._download_attempts(
                    spec, directory, report, cancel_check, heartbeat,
                    use_curl=True,
                )
            except (ModelCancelledError, ModelIntegrityError):
                raise
            except ModelError as fallback:
                raise type(fallback)(
                    f"{fallback} Python HTTPS failure: {primary}",
                    model_id=spec.model_id, url=fallback.url or spec.primary_url,
                    directory=directory,
                ) from primary
            finally:
                # Curl attempts never resume, including after cancellation or timeout.
                part.unlink(missing_ok=True)

    def _download_attempts(
        self, spec, directory, progress_callback, cancel_check, heartbeat,
        *, use_curl=False,
    ) -> Path:
        part = directory / (spec.filename + PART_SUFFIX)
        downloaded_file = directory / spec.filename
        reported = 0

        def report(done: int, total: int | None) -> None:
            nonlocal reported
            reported = max(reported, int(done))
            if progress_callback is None:
                return
            try:
                progress_callback(reported, total)
            except Exception:
                pass

        attempts: deque[str] = deque(maxlen=8)
        last_http: _HTTPStatusError | None = None
        last_http_url = ""
        last_network: BaseException | None = None
        last_was_http = False
        for url in spec.urls:
            for attempt in range(1 if use_curl else self.retries + 1):
                if attempt:
                    self._sleep_backoff(attempt, spec, directory, cancel_check)
                try:
                    digests, size = self._fetch(
                        spec, url, part, directory, report, cancel_check, heartbeat,
                        use_curl=use_curl,
                    )
                except (ModelCancelledError, ModelIntegrityError):
                    part.unlink(missing_ok=True)
                    raise
                except _HTTPStatusError as exc:
                    attempts.append(f"{url}: HTTP {exc.status}")
                    last_http, last_http_url, last_was_http = exc, url, True
                    if exc.retryable:
                        continue
                    break
                except _TransientError as exc:
                    attempts.append(f"{url}: {exc}")
                    last_network, last_was_http = exc, False
                    continue
                problem = self._digest_problem(spec, size, digests)
                if problem:
                    part.unlink(missing_ok=True)
                    raise ModelIntegrityError(
                        f"Downloaded file for model {spec.model_id} from {url} failed verification "
                        f"({problem}); the file was deleted. Download it manually "
                        f"from {spec.primary_url} and place it in {directory}.",
                        model_id=spec.model_id, url=url, directory=directory,
                    )
                os.replace(part, downloaded_file)
                if spec.archive == "none":
                    self._write_marker(spec, downloaded_file, digests)
                    return downloaded_file
                return self._extract(spec, downloaded_file, directory, url)

        primary = spec.primary_url
        tried = "; ".join(attempts)
        if last_http is not None and (last_was_http or last_network is None):
            raise ModelDownloadError(
                f"Cannot download model {spec.model_id}: the server returned HTTP {last_http.status} "
                f"for {last_http_url}. Try again later, or download it manually from {primary} and place it "
                f"in {directory}, or set a local models folder in Settings. (Attempts: {tried})",
                status=last_http.status, model_id=spec.model_id, url=last_http_url, directory=directory,
            )
        reason = (
            f"request timed out (timeout {self.timeout:g} seconds)"
            if isinstance(last_network, _TransientTimeout)
            else "no network connection"
        )
        raise ModelNetworkError(
            f"Cannot download model {spec.model_id}: {reason}. Download it manually from "
            f"{primary} and place it in {directory}, or set a local models folder in Settings. "
            f"(Last error: {last_network or 'unknown error'})",
            model_id=spec.model_id, url=primary, directory=directory,
        )

    def _fetch(
        self,
        spec: ModelSpec,
        url: str,
        part: Path,
        directory: Path,
        report: Callable[[int, "int | None"], None],
        cancel_check: CancelCheck | None,
        heartbeat: Callable[[], None],
        *,
        use_curl: bool = False,
    ) -> tuple[dict[str, str], int]:
        if _cancelled(cancel_check):
            raise self._cancelled_error(spec, directory, url)
        try:
            _validate_download_url(url)
            if use_curl and urllib.parse.urlsplit(url).scheme != "https":
                raise _UnsafeDownloadURL("system curl requires HTTPS")
        except ValueError as exc:
            raise ModelDownloadError(
                f"Cannot download model {spec.model_id}: unsafe URL {url} ({exc}).",
                model_id=spec.model_id, url=url, directory=directory,
            ) from exc
        limit = spec.size if spec.size is not None else self.max_download_bytes

        def check_size(size: int) -> None:
            if size > limit:
                raise ModelIntegrityError(
                    f"Download of model {spec.model_id} exceeds the limit of {limit} bytes "
                    f"({size} bytes including any resumed data); the partial file was deleted. "
                    f"Download it manually from {spec.primary_url} and place it in {directory}.",
                    model_id=spec.model_id, url=url, directory=directory,
                )

        if use_curl:
            part.unlink(missing_ok=True)
        offset = part.stat().st_size if part.exists() else 0
        check_size(offset)
        if spec.size is not None and offset and offset == spec.size:
            hashers = self._new_hashers(spec)
            self._hash_into(part, hashers)
            report(offset, spec.size)
            return {name: h.hexdigest() for name, h in hashers.items()}, offset

        headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
        if offset:
            headers.update({"Range": f"bytes={offset}-"})
        request = urllib.request.Request(url, headers=headers)

        def check_cancel() -> None:
            if _cancelled(cancel_check):
                raise self._cancelled_error(spec, directory, url)

        try:
            if use_curl:
                response = _CurlResponse(
                    url, timeout=self.timeout, check_cancel=check_cancel,
                    context=dict(model_id=spec.model_id, url=url, directory=directory),
                )
            else:
                response = self._open(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and offset:
                part.unlink(missing_ok=True)
                raise _TransientError("server rejected the resume range; restarting") from exc
            raise _HTTPStatusError(exc.code, str(exc.reason or "")) from exc
        except _UnsafeDownloadURL as exc:
            raise ModelDownloadError(
                f"Cannot download model {spec.model_id}: unsafe redirect from {url} ({exc}).",
                model_id=spec.model_id, url=url, directory=directory,
            ) from exc
        except (urllib.error.URLError, OSError, http.client.HTTPException, ValueError) as exc:
            reason = getattr(exc, "reason", exc)
            error_type = _TransientTimeout if isinstance(reason, TimeoutError) else _TransientError
            raise error_type(f"{type(reason).__name__}: {reason}") from exc

        with response:
            status = int(getattr(response, "status", 200) or 200)
            if status >= 400:
                raise _HTTPStatusError(status)
            range_end: int | None = None
            content_range = str(response.headers.get("Content-Range") or "")

            def reject_range() -> None:
                raise ModelIntegrityError(
                    f"Download of model {spec.model_id} has an invalid Content-Range "
                    f"({content_range!r}, requested offset {offset}); the partial file was deleted. "
                    f"Download it manually from {spec.primary_url} and place it in {directory}.",
                    model_id=spec.model_id, url=url, directory=directory,
                )

            if status == 206:
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", content_range.strip())
                if match is None:
                    reject_range()
                start, range_end = int(match.group(1)), int(match.group(2))
                if start != offset or start > range_end:
                    reject_range()
                if match.group(3) != "*" and range_end >= int(match.group(3)):
                    reject_range()
                length = response.headers.get("Content-Length")
                if length is not None:
                    try:
                        response_length = int(length)
                    except (TypeError, ValueError):
                        reject_range()
                    if response_length != range_end - start + 1:
                        reject_range()
            elif offset:
                # Server ignored Range: restart instead of appending a second copy.
                offset = 0
            mode = "ab" if offset else "wb"
            total = _response_total(response, offset)
            if total is None:
                total = spec.size
            if total is not None:
                check_size(total)
            hashers = self._new_hashers(spec)
            if offset:
                self._hash_into(part, hashers)
            downloaded = offset
            report(downloaded, total)
            with part.open(mode) as output:
                def write_chunk(chunk: bytes) -> None:
                    nonlocal downloaded
                    check_size(downloaded + len(chunk))
                    if range_end is not None and downloaded + len(chunk) > range_end + 1:
                        reject_range()
                    output.write(chunk)
                    for hasher in hashers.values():
                        hasher.update(chunk)
                    downloaded += len(chunk)
                    report(downloaded, total)

                while True:
                    if _cancelled(cancel_check):
                        raise self._cancelled_error(spec, directory, url)
                    try:
                        chunk = response.read(min(self.chunk_size, limit - downloaded + 1))
                    except http.client.IncompleteRead as exc:
                        if exc.partial:
                            write_chunk(exc.partial)
                        raise _TransientError(f"connection closed after {downloaded} bytes") from exc
                    except (OSError, http.client.HTTPException, ValueError) as exc:
                        error_type = _TransientTimeout if isinstance(exc, TimeoutError) else _TransientError
                        raise error_type(f"{type(exc).__name__}: {exc}") from exc
                    if not chunk:
                        break
                    write_chunk(chunk)
                    heartbeat()
            expected = max(total or 0, range_end + 1 if range_end is not None else 0)
            if expected > downloaded and not use_curl:
                raise _TransientError(f"connection closed after {downloaded} of {expected} bytes")
        return {name: h.hexdigest() for name, h in hashers.items()}, downloaded

    # -- extraction -------------------------------------------------------------------

    def _extract(self, spec: ModelSpec, archive_path: Path, directory: Path, url: str) -> Path:
        install = self.installed_path(spec, directory)
        staging = Path(tempfile.mkdtemp(prefix=f".{_safe_token(spec.install_name)}.extract-", dir=str(directory)))
        backup: Path | None = None
        try:
            try:
                if spec.archive == "zip":
                    safe_extract_zip(archive_path, staging)
                else:
                    _safe_extract_tar(archive_path, staging)
            except (UnsafeArchiveError, OSError, tarfile.TarError, ValueError) as exc:
                raise ModelIntegrityError(
                    f"Archive for model {spec.model_id} from {url} is unsafe or damaged ({exc}); it was deleted. "
                    f"Download it manually from {spec.primary_url} and place the extracted folder at {install}.",
                    model_id=spec.model_id, url=url, directory=directory,
                ) from exc
            root = staging
            children = list(staging.iterdir())
            missing_at_top = any(not (staging / item).is_file() for item in spec.required_files)
            if spec.required_files and missing_at_top and len(children) == 1:
                (only,) = children
                if only.is_dir():
                    root = only
            missing = [item for item in spec.required_files if not (root / item).is_file()]
            if missing:
                raise ModelIntegrityError(
                    f"Archive for model {spec.model_id} from {url} is missing required files: "
                    f"{', '.join(missing)}; it was deleted.",
                    model_id=spec.model_id, url=url, directory=directory,
                )
            (root / VERIFIED_SUFFIX).write_text(
                json.dumps({"model_id": spec.model_id, "archive_sha256": spec.sha256 or ""}, sort_keys=True),
                encoding="utf-8",
            )
            if install.exists():
                backup = directory / f".{_safe_token(spec.install_name)}.previous-{uuid.uuid4().hex}"
                os.replace(install, backup)
            try:
                os.replace(root, install)
            except OSError:
                if backup is not None and not install.exists():
                    os.replace(backup, install)
                    backup = None
                raise
            return install
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            if backup is not None:
                shutil.rmtree(backup, ignore_errors=True)
            archive_path.unlink(missing_ok=True)


def _safe_extract_tar(source: Path, destination: Path) -> None:
    """Extract only regular files/directories with traversal-safe names."""
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    root_resolved = root.resolve(strict=False)
    with tarfile.open(source, "r:*") as archive:
        members = archive.getmembers()
        for member in members:
            safe_member_parts(member.name)
            if not (member.isfile() or member.isdir()):
                raise UnsafeArchiveError(f"TAR contains a link or special file: {member.name!r}")
        for member in members:
            target = root.joinpath(*safe_member_parts(member.name))
            resolved = target.resolve(strict=False)
            if resolved != root_resolved and root_resolved not in resolved.parents:
                raise UnsafeArchiveError(f"TAR member escapes the destination: {member.name!r}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = archive.extractfile(member)
            if stream is None:
                raise UnsafeArchiveError(f"TAR member cannot be read: {member.name!r}")
            with stream, target.open("wb") as output:
                shutil.copyfileobj(stream, output, length=CHUNK_SIZE)


_DEFAULT_MANAGER: ModelManager | None = None
_DEFAULT_MANAGER_LOCK = threading.Lock()


def default_manager() -> ModelManager:
    """Process-wide manager that follows :func:`configure` / env changes."""
    global _DEFAULT_MANAGER
    with _DEFAULT_MANAGER_LOCK:
        if _DEFAULT_MANAGER is None:
            _DEFAULT_MANAGER = ModelManager()
        return _DEFAULT_MANAGER


__all__ = [
    "ENV_AUTO_DOWNLOAD",
    "ENV_MODELS_DIR",
    "MODEL_REGISTRY",
    "ModelCancelledError",
    "ModelDownloadError",
    "ModelError",
    "ModelIntegrityError",
    "ModelManager",
    "ModelNetworkError",
    "ModelNotAvailableError",
    "ModelSpec",
    "ModelStatus",
    "auto_download_enabled",
    "configure",
    "configured_models_dir",
    "default_manager",
    "default_platform_models_dir",
    "directory_ready",
    "download_explicitly_requested",
    "explicit_models_dir_configured",
    "get_spec",
    "resolve_models_dir",
    "specs_for_engine",
    "user_requested_download",
]
