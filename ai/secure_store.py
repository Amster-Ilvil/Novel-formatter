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
        return Path("/System/Library/Frameworks/Security.framework/Security").exists()
    return os.name == "nt"


def _mac_security_framework():
    """Use the native Keychain C API, never put credentials in process argv.

    The Security.framework path is a fixed macOS system framework, not a
    downloaded library or a caller-supplied path. The framework handles
    Keychain ACLs and any user-interaction prompts.
    """
    security = ctypes.CDLL("/System/Library/Frameworks/Security.framework/Security")
    core = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    u32 = ctypes.c_uint32
    ptr = ctypes.c_void_p
    security.SecKeychainFindGenericPassword.argtypes = [
        ptr, u32, ctypes.c_char_p, u32, ctypes.c_char_p,
        ctypes.POINTER(u32), ctypes.POINTER(ptr), ctypes.POINTER(ptr),
    ]
    security.SecKeychainFindGenericPassword.restype = ctypes.c_int32
    security.SecKeychainAddGenericPassword.argtypes = [
        ptr, u32, ctypes.c_char_p, u32, ctypes.c_char_p,
        u32, ptr, ctypes.POINTER(ptr),
    ]
    security.SecKeychainAddGenericPassword.restype = ctypes.c_int32
    security.SecKeychainItemModifyAttributesAndData.argtypes = [ptr, ptr, u32, ptr]
    security.SecKeychainItemModifyAttributesAndData.restype = ctypes.c_int32
    security.SecKeychainItemDelete.argtypes = [ptr]
    security.SecKeychainItemDelete.restype = ctypes.c_int32
    security.SecKeychainItemFreeContent.argtypes = [ptr, ptr]
    security.SecKeychainItemFreeContent.restype = ctypes.c_int32
    core.CFRelease.argtypes = [ptr]
    core.CFRelease.restype = None
    return security, core


def _mac_find(service: str, account: str, *, fetch_password: bool = False):
    security, core = _mac_security_framework()
    svc = service.encode("utf-8")
    acct = account.encode("utf-8")
    pw_len = ctypes.c_uint32()
    pw_data = ctypes.c_void_p()
    item = ctypes.c_void_p()
    status = security.SecKeychainFindGenericPassword(
        None, len(svc), svc, len(acct), acct,
        ctypes.byref(pw_len) if fetch_password else None,
        ctypes.byref(pw_data) if fetch_password else None,
        ctypes.byref(item),
    )
    return security, core, int(status), item, pw_len, pw_data


def _mac_load(service: str = _SERVICE, account: str = _ACCOUNT) -> str:
    security, core, status, item, pw_len, pw_data = _mac_find(service, account, fetch_password=True)
    try:
        if status != 0 or not pw_data.value:
            return ""
        return ctypes.string_at(pw_data, pw_len.value).decode("utf-8")
    finally:
        if pw_data.value:
            security.SecKeychainItemFreeContent(None, pw_data)
        if item.value:
            core.CFRelease(item)


def _mac_save(value: str, service: str = _SERVICE, account: str = _ACCOUNT,
              label: str = "Novel Formatter AI API Key") -> bool:
    """Store a Keychain item without exposing the secret in ps or argv.

    Existing items are modified rather than deleted so their access controls
    survive ordinary updates. label is preserved in the public interface;
    the system item's service name is the default display label.
    """
    del label
    security, core = _mac_security_framework()
    svc, acct = service.encode("utf-8"), account.encode("utf-8")
    secret = ctypes.create_string_buffer(value.encode("utf-8"))
    status = security.SecKeychainAddGenericPassword(
        None, len(svc), svc, len(acct), acct,
        len(secret) - 1, ctypes.cast(secret, ctypes.c_void_p), None,
    )
    if status == 0:
        return True
    if status != -25299:  # errSecDuplicateItem
        return False
    security, core, found, item, _pw_len, _pw_data = _mac_find(service, account)
    if found != 0 or not item.value:
        return False
    try:
        return security.SecKeychainItemModifyAttributesAndData(
            item, None, len(secret) - 1, ctypes.cast(secret, ctypes.c_void_p)
        ) == 0
    finally:
        core.CFRelease(item)


def _mac_delete(service: str = _SERVICE, account: str = _ACCOUNT) -> None:
    security, core, status, item, _pw_len, _pw_data = _mac_find(service, account)
    if status == -25300:  # errSecItemNotFound: deleting an absent item is OK
        return
    if status != 0 or not item.value:
        raise OSError(f"Keychain item lookup failed: OSStatus {status}")
    try:
        result = security.SecKeychainItemDelete(item)
        if result != 0:
            raise OSError(f"Keychain item removal failed: OSStatus {result}")
    finally:
        core.CFRelease(item)


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
