# -*- coding: utf-8 -*-
"""Cross-platform manual download transports.

Windows prefers BITS so downloads use the operating system proxy and certificate
store. If BITS is unavailable, curl.exe and Python HTTPS remain explicit
fallbacks. macOS/Linux keep Python HTTPS first, then system curl. The public first-deployment
bootstrap may call these transports only to install missing compatible resources;
subsequent model version checks and replacements remain manual.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Callable, Iterable

ProgressCallback = Callable[[str, int, int, str], None]
CancelCheck = Callable[[], bool]
USER_AGENT = "Novel-Formatter-Manual-Downloader/2.0"


class DownloadError(RuntimeError):
    pass


class DownloadCancelled(DownloadError):
    pass


def _emit(callback: ProgressCallback | None, stage: str, current: int, total: int, detail: str) -> None:
    if callback is None:
        return
    try:
        callback(str(stage), max(0, int(current)), max(0, int(total)), str(detail))
    except Exception:
        pass


def _cancelled(cancel_check: CancelCheck | None) -> bool:
    try:
        return bool(cancel_check and cancel_check())
    except Exception:
        return False


def _hidden_process_kwargs() -> dict:
    if os.name != "nt":
        return {}
    return {"creationflags": int(getattr(subprocess, "CREATE_NO_WINDOW", 0))}


def _powershell_executable() -> str:
    for name in ("powershell.exe", "pwsh.exe", "powershell", "pwsh"):
        value = shutil.which(name)
        if value:
            return value
    return ""


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/octet-stream,*/*;q=0.8",
            "Cache-Control": "no-cache",
        },
    )


def _download_python(
    url: str,
    part: Path,
    *,
    label: str,
    timeout: float,
    progress_callback: ProgressCallback | None,
    cancel_check: CancelCheck | None,
) -> None:
    with urllib.request.urlopen(_request(url), timeout=timeout) as response, part.open("wb") as output:
        try:
            total = int(response.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            total = 0
        current = 0
        while True:
            if _cancelled(cancel_check):
                raise DownloadCancelled(f"用户取消下载 {label}")
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            current += len(chunk)
            _emit(progress_callback, "download", current, total, f"{label} · Python HTTPS")


def _download_curl(
    url: str,
    part: Path,
    *,
    label: str,
    timeout: float,
    progress_callback: ProgressCallback | None,
    cancel_check: CancelCheck | None,
) -> None:
    curl = shutil.which("curl.exe" if os.name == "nt" else "curl") or shutil.which("curl")
    if not curl:
        raise DownloadError("系统未找到 curl")
    command = [
        curl,
        "--location",
        "--fail",
        "--silent",
        "--show-error",
        "--retry",
        "3",
        "--retry-delay",
        "1",
        "--connect-timeout",
        "20",
        "--max-time",
        str(max(60, int(timeout))),
        "--output",
        str(part),
        url,
    ]
    stderr_file = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=stderr_file,
        text=True,
        **_hidden_process_kwargs(),
    )
    started = time.monotonic()
    try:
        while process.poll() is None:
            if _cancelled(cancel_check):
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise DownloadCancelled(f"用户取消下载 {label}")
            if time.monotonic() - started > timeout + 30:
                process.kill()
                raise DownloadError(f"下载 {label} 超时")
            current = part.stat().st_size if part.exists() else 0
            _emit(progress_callback, "download", current, 0, f"{label} · 系统 curl")
            time.sleep(0.25)
        if process.returncode != 0:
            stderr_file.flush()
            stderr_file.seek(0)
            detail = stderr_file.read().strip()
            raise DownloadError(detail[-1200:] or f"curl 退出码 {process.returncode}")
    finally:
        if process.poll() is None:
            process.kill()
        stderr_file.close()


_BITS_SCRIPT = r'''
param(
  [Parameter(Mandatory=$true)][string]$Url,
  [Parameter(Mandatory=$true)][string]$Destination,
  [Parameter(Mandatory=$true)][string]$CancelFile,
  [Parameter(Mandatory=$true)][string]$ProgressFile,
  [Parameter(Mandatory=$true)][int]$TimeoutSeconds,
  [Parameter(Mandatory=$true)][string]$JobName
)
$ErrorActionPreference = 'Stop'
Import-Module BitsTransfer -ErrorAction Stop
$job = $null
try {
  if (Test-Path -LiteralPath $Destination) { Remove-Item -LiteralPath $Destination -Force }
  $job = Start-BitsTransfer -Source $Url -Destination $Destination -DisplayName $JobName -Description 'Novel Formatter manual OCR model download' -Asynchronous
  $started = Get-Date
  while ($true) {
    if (Test-Path -LiteralPath $CancelFile) {
      Remove-BitsTransfer -BitsJob $job -Confirm:$false -ErrorAction SilentlyContinue
      exit 1223
    }
    $job = Get-BitsTransfer -Id $job.Id -ErrorAction Stop
    $progress = @{
      transferred = [int64]$job.BytesTransferred
      total = [int64]$job.BytesTotal
      state = [string]$job.JobState
    } | ConvertTo-Json -Compress
    [System.IO.File]::WriteAllText($ProgressFile, $progress, [System.Text.UTF8Encoding]::new($false))
    switch ([string]$job.JobState) {
      'Transferred' {
        Complete-BitsTransfer -BitsJob $job
        exit 0
      }
      'Error' {
        $detail = [string]$job.ErrorDescription
        Remove-BitsTransfer -BitsJob $job -Confirm:$false -ErrorAction SilentlyContinue
        throw "BITS error: $detail"
      }
      'TransientError' { Resume-BitsTransfer -BitsJob $job -Asynchronous -ErrorAction SilentlyContinue | Out-Null }
      'Suspended' { Resume-BitsTransfer -BitsJob $job -Asynchronous -ErrorAction SilentlyContinue | Out-Null }
      'Cancelled' { throw 'BITS transfer cancelled' }
    }
    if (((Get-Date) - $started).TotalSeconds -gt $TimeoutSeconds) {
      Remove-BitsTransfer -BitsJob $job -Confirm:$false -ErrorAction SilentlyContinue
      throw 'BITS transfer timeout'
    }
    Start-Sleep -Milliseconds 250
  }
}
finally {
  if ($job -ne $null) {
    $state = [string]$job.JobState
    if ($state -notin @('Transferred','Acknowledged','Cancelled')) {
      Remove-BitsTransfer -BitsJob $job -Confirm:$false -ErrorAction SilentlyContinue
    }
  }
}
'''


def _download_windows_bits(
    url: str,
    part: Path,
    *,
    label: str,
    timeout: float,
    progress_callback: ProgressCallback | None,
    cancel_check: CancelCheck | None,
) -> None:
    if os.name != "nt":
        raise DownloadError("BITS 仅适用于 Windows")
    powershell = _powershell_executable()
    if not powershell:
        raise DownloadError("未找到 Windows PowerShell")
    with tempfile.TemporaryDirectory(prefix="nf-bits-") as temp_dir:
        temp = Path(temp_dir)
        script = temp / "download.ps1"
        cancel_file = temp / "cancel"
        progress_file = temp / "progress.json"
        script.write_text(_BITS_SCRIPT, encoding="utf-8-sig")
        job_name = "NovelFormatter-" + uuid.uuid4().hex
        command = [
            powershell,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-Url",
            url,
            "-Destination",
            str(part),
            "-CancelFile",
            str(cancel_file),
            "-ProgressFile",
            str(progress_file),
            "-TimeoutSeconds",
            str(max(60, int(timeout))),
            "-JobName",
            job_name,
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            **_hidden_process_kwargs(),
        )
        output: list[str] = []
        try:
            while process.poll() is None:
                if _cancelled(cancel_check):
                    cancel_file.touch()
                try:
                    payload = json.loads(progress_file.read_text(encoding="utf-8-sig"))
                except Exception:
                    payload = {}
                current = int(payload.get("transferred") or (part.stat().st_size if part.exists() else 0))
                total = int(payload.get("total") or 0)
                state = str(payload.get("state") or "连接中")
                _emit(progress_callback, "download", current, total, f"{label} · Windows BITS · {state}")
                time.sleep(0.25)
            if process.stdout is not None:
                output.extend(line.rstrip() for line in process.stdout.readlines())
            code = process.wait(timeout=5)
            if code == 1223 or _cancelled(cancel_check):
                raise DownloadCancelled(f"用户取消下载 {label}")
            if code != 0:
                raise DownloadError("\n".join(output[-20:]) or f"BITS 退出码 {code}")
        finally:
            if process.poll() is None:
                cancel_file.touch()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()


def _transport_order() -> list[tuple[str, Callable]]:
    if os.name == "nt":
        return [
            ("Windows BITS", _download_windows_bits),
            ("Windows curl.exe", _download_curl),
            ("Python HTTPS", _download_python),
        ]
    transports = [("Python HTTPS", _download_python)]
    if shutil.which("curl"):
        transports.append(("系统 curl", _download_curl))
    return transports


def download_file(
    url: str,
    destination: Path,
    *,
    label: str,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
    timeout: float = 120.0,
) -> Path:
    """Download one URL after an explicit user action using platform transports."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    errors: list[str] = []
    for transport_name, transport in _transport_order():
        if _cancelled(cancel_check):
            raise DownloadCancelled(f"用户取消下载 {label}")
        part.unlink(missing_ok=True)
        try:
            _emit(progress_callback, "connect", 0, 0, f"{label} · 准备使用 {transport_name}")
            transport(
                url,
                part,
                label=label,
                timeout=timeout,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
            )
            if not part.is_file() or part.stat().st_size <= 0:
                raise DownloadError("下载结果为空")
            os.replace(part, destination)
            _emit(progress_callback, "download", destination.stat().st_size, destination.stat().st_size, f"{label} · {transport_name} 下载完成")
            return destination
        except DownloadCancelled:
            raise
        except Exception as exc:
            errors.append(f"{transport_name}: {exc}")
        finally:
            part.unlink(missing_ok=True)
    raise DownloadError(f"{label} 下载失败：" + "；".join(errors))


def download_first(
    urls: Iterable[str],
    destination: Path,
    *,
    label: str,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
    timeout: float = 120.0,
) -> Path:
    errors: list[str] = []
    for url in [str(item) for item in urls if str(item)]:
        try:
            return download_file(
                url,
                destination,
                label=label,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
                timeout=timeout,
            )
        except DownloadCancelled:
            raise
        except Exception as exc:
            errors.append(str(exc))
    raise DownloadError(f"{label} 的所有官方下载源均失败：" + "；".join(errors))


__all__ = [
    "DownloadCancelled",
    "DownloadError",
    "download_file",
    "download_first",
]
