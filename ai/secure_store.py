# -*- coding: utf-8 -*-
"""OS-backed storage for AI credentials.

macOS uses the user's login Keychain. Windows uses DPAPI and stores only the
machine/user-bound ciphertext in the application config directory. Linux has no
universal dependency-free system keyring, so persistence is deliberately
unavailable there instead of silently falling back to plaintext JSON.
"""
from __future__ import annotations

import base64
import ctypes
import getpass
import os
import stat
import subprocess
import sys
from ctypes import wintypes
from pathlib import Path

_SERVICE = "NovelFormatter.AI"
_ACCOUNT = "api_key"


def _config_dir() -> Path:
    override = os.environ.get("NOVEL_FORMATTER_CONFIG_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "NovelFormatter"
    if os.name == "nt":
        return Path(os.environ.get("APPDATA", Path.home())) / "NovelFormatter"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "novel-formatter"


def persistence_available() -> bool:
    if sys.platform == "darwin":
        return Path("/usr/bin/security").exists()
    return os.name == "nt"


def _mac_load(service: str = _SERVICE, account: str = _ACCOUNT) -> str:
    result = subprocess.run(
        ["/usr/bin/security", "find-generic-password", "-a", account, "-s", service, "-w"],
        text=True,
        capture_output=True,
        timeout=8,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _mac_save(value: str, service: str = _SERVICE, account: str = _ACCOUNT, label: str = "Novel Formatter AI API Key") -> bool:
    # security(1) is the system-supported Keychain CLI. Avoid shell=True so the
    # credential cannot be reinterpreted as shell syntax.
    result = subprocess.run(
        [
            "/usr/bin/security", "add-generic-password", "-U",
            "-a", account, "-s", service, "-l", label,
            "-w", value,
        ],
        text=True,
        capture_output=True,
        timeout=8,
        check=False,
    )
    return result.returncode == 0


def _mac_delete(service: str = _SERVICE, account: str = _ACCOUNT) -> None:
    subprocess.run(
        ["/usr/bin/security", "delete-generic-password", "-a", account, "-s", service],
        text=True,
        capture_output=True,
        timeout=8,
        check=False,
    )


class _DATA_BLOB(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]


def _blob_from_bytes(value: bytes):
    buf = ctypes.create_string_buffer(value)
    blob = _DATA_BLOB(len(value), ctypes.cast(buf, ctypes.POINTER(ctypes.c_byte)))
    return blob, buf


def _dpapi_protect(value: bytes) -> bytes:
    in_blob, _in_buf = _blob_from_bytes(value)
    out_blob = _DATA_BLOB()
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    if not crypt32.CryptProtectData(
        ctypes.byref(in_blob), "NovelFormatter", None, None, None, 0,
        ctypes.byref(out_blob),
    ):
        raise ctypes.WinError()
    try:
        return ctypes.string_at(out_blob.pbData, out_blob.cbData)
    finally:
        kernel32.LocalFree(out_blob.pbData)


def _dpapi_unprotect(value: bytes) -> bytes:
    in_blob, _in_buf = _blob_from_bytes(value)
    out_blob = _DATA_BLOB()
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    if not crypt32.CryptUnprotectData(
        ctypes.byref(in_blob), None, None, None, None, 0,
        ctypes.byref(out_blob),
    ):
        raise ctypes.WinError()
    try:
        return ctypes.string_at(out_blob.pbData, out_blob.cbData)
    finally:
        kernel32.LocalFree(out_blob.pbData)


def _windows_path(filename: str = "ai_key.dpapi") -> Path:
    return _config_dir() / filename


def _windows_load(filename: str = "ai_key.dpapi") -> str:
    path = _windows_path(filename)
    if not path.exists():
        return ""
    encrypted = base64.b64decode(path.read_bytes(), validate=True)
    return _dpapi_unprotect(encrypted).decode("utf-8")


def _windows_save(value: str, filename: str = "ai_key.dpapi") -> bool:
    path = _windows_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    encrypted = _dpapi_protect(value.encode("utf-8"))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(base64.b64encode(encrypted))
    try:
        tmp.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    os.replace(tmp, path)
    return True


def _windows_delete(filename: str = "ai_key.dpapi") -> None:
    _windows_path(filename).unlink(missing_ok=True)



def _normalise_secret_name(value: str) -> str:
    """Return a filesystem/keychain-safe logical secret name."""
    cleaned = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(value or "").strip())
    cleaned = cleaned.strip("._")
    if not cleaned:
        raise ValueError("secret name is empty")
    return cleaned[:96]


def load_named_secret(name: str, *, account: str = "token") -> str:
    """Load an independent application credential without plaintext fallback.

    Named secrets share the same OS-backed policy as the legacy AI key but use
    separate Keychain services / DPAPI files, so an OCR token can never replace
    the LLM provider credential.
    """
    safe = _normalise_secret_name(name)
    service = f"NovelFormatter.{safe}"
    filename = f"{safe}.dpapi"
    try:
        if sys.platform == "darwin" and persistence_available():
            return _mac_load(service=service, account=str(account or "token"))
        if os.name == "nt":
            return _windows_load(filename)
    except Exception:
        return ""
    return ""


def save_named_secret(name: str, value: str, *, account: str = "token", label: str = "") -> bool:
    safe = _normalise_secret_name(name)
    value = str(value or "")
    if not value:
        delete_named_secret(safe, account=account)
        return True
    service = f"NovelFormatter.{safe}"
    filename = f"{safe}.dpapi"
    try:
        if sys.platform == "darwin" and persistence_available():
            return _mac_save(
                value,
                service=service,
                account=str(account or "token"),
                label=str(label or f"Novel Formatter {safe}"),
            )
        if os.name == "nt":
            return _windows_save(value, filename)
    except Exception:
        return False
    return False


def delete_named_secret(name: str, *, account: str = "token") -> None:
    safe = _normalise_secret_name(name)
    service = f"NovelFormatter.{safe}"
    filename = f"{safe}.dpapi"
    try:
        if sys.platform == "darwin" and persistence_available():
            _mac_delete(service=service, account=str(account or "token"))
        elif os.name == "nt":
            _windows_delete(filename)
    except Exception:
        pass


def load_secret() -> str:
    try:
        if sys.platform == "darwin" and persistence_available():
            return _mac_load()
        if os.name == "nt":
            return _windows_load()
    except Exception:
        return ""
    return ""


def save_secret(value: str) -> bool:
    value = str(value or "")
    if not value:
        delete_secret()
        return True
    try:
        if sys.platform == "darwin" and persistence_available():
            return _mac_save(value)
        if os.name == "nt":
            return _windows_save(value)
    except Exception:
        return False
    return False


def delete_secret() -> None:
    try:
        if sys.platform == "darwin" and persistence_available():
            _mac_delete()
        elif os.name == "nt":
            _windows_delete()
    except Exception:
        pass
