"""Minimal, credential-safe Gemini REST JSON-image transport for OCR.

No SDK dependency and no implicit network calls. An API key is explicitly required
on invocation. Only Google's fixed HTTPS endpoint is allowed; image bytes and
credentials are never included in exceptions or diagnostic logs.
"""
from __future__ import annotations

import base64
import json
import os
import re
import socket
import ssl
import threading
import time
from http.client import HTTPSConnection, IncompleteRead
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPSHandler, Request, build_opener

from .http_transport import RejectRedirects, open_direct

from ai.config import validate_api_key
from ai.gemini_free_tier import DEFAULT_RPD, DEFAULT_RPM, DEFAULT_TPM, OCR_MODEL, SharedModelQuota
from ai.request_limiter import RequestLimiter, retry_delay_seconds

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/"
DEFAULT_MODEL = OCR_MODEL
_RETRYABLE_HTTP_STATUS = {408, 429, 500, 502, 503, 504}
OCR_SCHEMA = {
    "type": "object", "properties": {"items": {
        "type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "string"}, "text": {"type": "string"}},
            "required": ["id", "text"]}}}, "required": ["items"]
}


class GeminiTransportError(RuntimeError):
    """A Gemini request or response did not satisfy its contract."""


class GeminiRequestError(GeminiTransportError):
    """The request failed before Gemini returned an OCR response."""

    def __init__(self, message: str, *, retryable: bool = False, status_code: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


class GeminiResponseRejected(GeminiTransportError):
    """The provider explicitly stopped processing this content."""


class _GeminiHTTPSConnection(HTTPSConnection):
    def connect(self) -> None:
        # HTTPConnection.send() invokes connect() before it sends the POST.
        try:
            super().connect()
            return
        except (ssl.SSLEOFError, ssl.SSLZeroReturnError, ConnectionResetError, BrokenPipeError):
            pass
        except ssl.SSLError as exc:
            if exc.errno != ssl.SSL_ERROR_SYSCALL:
                raise
        if self.sock is not None:
            self.sock.close()
            self.sock = None
        super().connect()


class _GeminiHTTPSHandler(HTTPSHandler):
    def https_open(self, req):
        return self.do_open(_GeminiHTTPSConnection, req, context=self._context)


def _network_failure(exc: BaseException) -> tuple[str, bool]:
    # Never display the raw proxy/URL/SSL exception, which may include credentials.
    reason = exc.reason if isinstance(exc, URLError) else exc
    if isinstance(reason, ssl.SSLCertVerificationError):
        return "TLS 证书验证失败", False
    if isinstance(reason, ssl.SSLError):
        return "TLS 连接中断", True
    if isinstance(reason, socket.gaierror):
        return "DNS 解析失败", True
    if isinstance(reason, TimeoutError):
        return "连接或响应超时", True
    if isinstance(reason, (ConnectionError, IncompleteRead)):
        return "连接中断或响应未传输完整", True
    # urllib may wrap platform-specific socket/TLS details in a plain string.
    # Classify only known-safe categories; never surface that raw value.
    safe_hint = str(reason).lower()
    if any(value in safe_hint for value in ("certificate verify failed", "certificate has expired")):
        return "TLS 证书验证失败", False
    if any(value in safe_hint for value in ("name or service not known", "temporary failure in name resolution", "getaddrinfo failed")):
        return "DNS 解析失败", True
    if any(value in safe_hint for value in ("timed out", "timeout", "time-out")):
        return "连接或响应超时", True
    if any(value in safe_hint for value in ("ssl", "tls", "unexpected eof")):
        return "TLS 连接中断", True
    if any(value in safe_hint for value in ("connection reset", "connection aborted", "remote end closed", "incomplete read")):
        return "连接中断或响应未传输完整", True
    return "无法连接 Gemini 服务", True


def safe_model(model: str) -> str:
    value = str(model or DEFAULT_MODEL).strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{3,100}", value):
        raise ValueError("Gemini 模型 ID 不合法")
    return value


def response_text(payload: dict) -> str:
    candidates = payload.get("candidates") or []
    if not isinstance(candidates, list) or (candidates and not isinstance(candidates[0], dict)):
        raise GeminiTransportError("Gemini 响应结构非法")
    candidate = candidates[0] if candidates else {}
    finish_reason = candidate.get("finishReason")
    feedback = payload.get("promptFeedback")
    if (isinstance(feedback, dict) and feedback.get("blockReason")) or str(finish_reason) in {
        "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII",
        "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "MODEL_ARMOR",
    }:
        raise GeminiResponseRejected("Gemini 拒绝识别本批图像")
    content = candidate.get("content") or {}
    if not isinstance(content, dict):
        raise GeminiTransportError("Gemini 响应内容格式非法")
    pieces = content.get("parts") or []
    if not isinstance(pieces, list):
        raise GeminiTransportError("Gemini 响应内容格式非法")
    result = "".join(str(p.get("text") or "") for p in pieces if isinstance(p, dict))
    if not result:
        raise GeminiTransportError("Gemini 未返回文本（可能受安全策略拦截或输出被截断）")
    if finish_reason not in (None, "STOP", "FINISH_REASON_STOP"):
        raise GeminiTransportError("Gemini 响应未正常完成：" + str(finish_reason))
    return result


def validate_items(raw: str, expected: list[str], *, allow_partial: bool = False) -> dict[str, str]:
    try:
        data = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise GeminiTransportError("Gemini 响应不是完整 JSON") from exc
    rows = data.get("items") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise GeminiTransportError("Gemini JSON 缺少 items 数组")
    expected_set = set(expected)
    if len(expected) != len(expected_set):
        raise ValueError("本地行 ID 重复")
    result: dict[str, str] = {}
    seen: set[str] = set()
    for row in rows:
        key = row.get("id") if isinstance(row, dict) else None
        if not isinstance(key, str) or key not in expected_set:
            if allow_partial:
                continue
            raise GeminiTransportError("Gemini 行格式不正确")
        if key in seen:
            if allow_partial:
                result.pop(key, None)
                continue
            raise GeminiTransportError("Gemini 返回未知或重复行 ID")
        seen.add(key)
        if not isinstance(row.get("text"), str):
            if allow_partial:
                continue
            raise GeminiTransportError("Gemini 行格式不正确")
        result[key] = row["text"]
    if not allow_partial and set(result) != expected_set:
        raise GeminiTransportError("Gemini 返回行数/ID 与本地切列清单不一致")
    return result


class GeminiJsonTransport:
    def __init__(self, *, api_key: str = "", model: str = DEFAULT_MODEL,
                 request_timeout: float = 120, rpm: int = 0, tpm: int = 0,
                 max_retries: int = 2, request_fn=None, free_mode: bool = False,
                 rpd: int = 0, fallback_model: str = "", enable_fallback: bool = False,
                 quota_database=None):
        self.api_key = api_key  # never serialize
        self.model = safe_model(model)
        self.timeout = max(5., min(600., float(request_timeout)))
        self.limiter = RequestLimiter(rpm, tpm)
        self.max_retries = max(0, min(4, int(max_retries)))
        self.request_fn = request_fn
        self.local_quota = SharedModelQuota(
            self.model, rpm=rpm or DEFAULT_RPM, tpm=tpm or DEFAULT_TPM,
            rpd=rpd or DEFAULT_RPD, database=quota_database,
        ) if free_mode else None
        self._fallback = None
        if enable_fallback and str(fallback_model or "").strip() and safe_model(fallback_model) != self.model:
            self._fallback = GeminiJsonTransport(
                api_key=api_key, model=fallback_model, request_timeout=self.timeout,
                rpm=rpm, tpm=tpm, rpd=rpd, free_mode=free_mode, quota_database=quota_database,
                max_retries=0, request_fn=self.request_fn,
            )
        self._thread_state = threading.local()
        self._lock = threading.Lock()
        self.usage = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def request_provenance(self) -> dict:
        used_model = getattr(self._thread_state, "used_model", self.model)
        return {"model": used_model, "fallback_used": used_model != self.model}

    def _record_usage(self, *, prompt: int, completion: int, total: int) -> None:
        with self._lock:
            self.usage["requests"] += 1
            self.usage["prompt_tokens"] += prompt
            self.usage["completion_tokens"] += completion
            self.usage["total_tokens"] += total

    @staticmethod
    def make_body(prompt: str, images: list[Path], *, schema: dict | None,
                  max_output_tokens: int = 8192, json_mode: bool = True,
                  temperature: float = 0.0) -> dict:
        parts = [{"text": prompt}]
        for path in images:
            suffix = path.suffix.lower()
            mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(suffix, "image/png")
            parts.append({"inlineData": {"mimeType": mime,
                                         "data": base64.b64encode(path.read_bytes()).decode("ascii")}})
        generation = {"temperature": max(0.0, min(2.0, float(temperature))),
                      "maxOutputTokens": max(256, min(65536, int(max_output_tokens)))}
        if json_mode or schema is not None:
            generation["responseMimeType"] = "application/json"
        if schema is not None:
            generation["responseJsonSchema"] = schema
        return {"contents": [{"role": "user", "parts": parts}], "generationConfig": generation}

    def generate(self, prompt: str, images: list[str | Path], *, schema: dict | None = None,
                 max_output_tokens: int = 8192, cancel_check=None,
                 text_only: bool = False, json_mode: bool = True,
                 temperature: float = 0.0, progress_callback=None,
                 single_attempt: bool = False, usage_callback=None) -> str:
        self._thread_state.used_model = self.model
        key = validate_api_key(self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "")
        model = safe_model(self.model)
        paths = [Path(p) for p in images]
        if (not paths and not text_only) or not all(p.is_file() for p in paths):
            raise ValueError("Gemini OCR 必须提供存在的列切片或证据板")
        body = self.make_body(prompt, paths, schema=schema,
                              max_output_tokens=max_output_tokens, json_mode=json_mode,
                              temperature=temperature)
        url = ENDPOINT + model + ":generateContent"
        request = Request(url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
        request.add_unredirected_header("x-goog-api-key", key)
        max_attempts = 1 if single_attempt else self.max_retries + 1
        open_request = self.request_fn or (
            build_opener(RejectRedirects(), _GeminiHTTPSHandler()).open if single_attempt else open_direct
        )

        def report(event: dict) -> None:
            if progress_callback:
                try:
                    progress_callback(event)
                except Exception:
                    pass

        def record_usage(**tokens) -> None:
            self._record_usage(**tokens)
            if usage_callback:
                usage_callback(**tokens)

        for attempt in range(max_attempts):
            if cancel_check and cancel_check():
                raise GeminiTransportError("Gemini OCR 已取消")
            if self.local_quota is not None:
                # Reserve a local estimate without a second API call to count tokens.
                estimate = max(128, len(prompt.encode("utf-8")) // 2 + 8192 * len(paths))
                self.local_quota.acquire(
                    estimate, cancel_check=cancel_check,
                    wait_callback=lambda seconds: report({"event": "quota_wait", "delay_seconds": seconds}),
                )
            else:
                self.limiter.acquire(estimated_tokens=max(128, len(prompt) // 2), cancel_check=cancel_check)
            if cancel_check and cancel_check():
                raise GeminiTransportError("Gemini OCR 已取消")
            report({"event": "attempt", "attempt": attempt + 1, "max_attempts": max_attempts})
            try:
                with open_request(request, timeout=self.timeout) as stream:
                    try:
                        response = json.loads(stream.read().decode("utf-8"))
                    except ValueError:
                        raise GeminiTransportError("Gemini 响应不是完整 JSON") from None
                if not isinstance(response, dict):
                    raise GeminiTransportError("Gemini 响应结构非法")
                raw = response_text(response)
                usage = response.get("usageMetadata") or {}
                tokens = {
                    "prompt": int(usage.get("promptTokenCount") or 0),
                    "completion": int(usage.get("candidatesTokenCount") or 0),
                    "total": int(usage.get("totalTokenCount") or 0),
                }
                record_usage(**tokens)
                return raw
            except HTTPError as exc:
                # Keep HTTP error messages credential-free (some backends echo headers).
                exc.close()
                retryable = exc.code in _RETRYABLE_HTTP_STATUS
                if exc.code == 429 and not single_attempt and self._fallback is not None:
                    try:
                        raw = self._fallback.generate(
                            prompt, paths, schema=schema, max_output_tokens=max_output_tokens,
                            cancel_check=cancel_check, text_only=text_only, json_mode=json_mode,
                            temperature=temperature, single_attempt=True, usage_callback=record_usage,
                        )
                    except GeminiRequestError:
                        raise
                    except GeminiTransportError:
                        raise GeminiRequestError("Gemini 备用模型未返回可用结果。") from None
                    self._thread_state.used_model = self._fallback.model
                    return raw
                if not retryable or attempt + 1 == max_attempts or (exc.code == 429 and self.local_quota):
                    if progress_callback:
                        try:
                            progress_callback({
                                "event": "failed", "attempt": attempt + 1,
                                "max_attempts": max_attempts,
                                "reason": f"HTTP {exc.code}",
                            })
                        except Exception:
                            pass
                    raise GeminiRequestError(
                        f"Gemini API HTTP {exc.code}", retryable=retryable, status_code=exc.code,
                    ) from None
                delay = retry_delay_seconds(exc, attempt)
                retry_reason = f"HTTP {exc.code}"
            except (URLError, OSError, IncompleteRead) as exc:
                message, retryable = _network_failure(exc)
                if not retryable or attempt + 1 == max_attempts:
                    if progress_callback:
                        try:
                            progress_callback({
                                "event": "failed", "attempt": attempt + 1,
                                "max_attempts": max_attempts,
                                "reason": message,
                            })
                        except Exception:
                            pass
                    raise GeminiRequestError(
                        f"Gemini API 网络请求失败：{message}（已尝试 {attempt + 1} 次）",
                        retryable=retryable,
                    ) from None
                delay = retry_delay_seconds(exc, attempt)
                retry_reason = message
            if progress_callback:
                try:
                    progress_callback({
                        "event": "retry", "attempt": attempt + 1,
                        "next_attempt": attempt + 2,
                        "max_attempts": max_attempts,
                        "delay_seconds": round(delay, 1),
                        "reason": retry_reason,
                    })
                except Exception:
                    pass
            deadline = time.monotonic() + delay
            while (remaining := deadline - time.monotonic()) > 0:
                if cancel_check and cancel_check():
                    raise GeminiTransportError("Gemini OCR 已取消")
                time.sleep(min(remaining, 0.2) if cancel_check else remaining)
        raise GeminiTransportError("Gemini 请求重试次数已耗尽")
