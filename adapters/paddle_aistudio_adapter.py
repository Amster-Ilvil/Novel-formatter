#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Baidu AI Studio PaddleOCR cloud adapter.

Two API contracts are intentionally supported side-by-side:

* ``sync``: the model-specific synchronous URL copied from the PaddleOCR
  AI Studio task page.  Authentication uses ``Authorization: token ...`` and
  the request body contains a base64 encoded image.
* ``async_v2``: the public jobs endpoint.  Authentication uses
  ``Authorization: Bearer ...``; a job is submitted and polled until a JSONL
  result is ready.

The adapter always operates on the *full page*.  Novel Formatter deliberately
never fans this remote service out through the physical-column crop pipeline,
because doing so would multiply remote page usage and destroy cloud layout
context.  The normal multi-engine fusion stage can still combine this result
with locally columnized OCR engines.
"""
from __future__ import annotations

import base64
import ipaddress
import json
import os
import random
import time
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urljoin, urlparse

import httpx

from adapters.ocr_engine_common import run_ocr_engine
from ai.redaction import redact_secrets

DEFAULT_ASYNC_JOB_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
DEFAULT_REQUEST_TIMEOUT = 180.0
DEFAULT_POLL_INTERVAL = 5.0
DEFAULT_MAX_RETRIES = 3
SECRET_NAME = "PaddleOCR.AIStudio"
SECRET_ACCOUNT = "access_token"

SYNC_FAMILIES = ("ppocr", "ppstructure", "vl", "layout_vl")

# Current official PaddleOCR API SDK model registry (2026). Keep this local
# allow-list explicit instead of accepting arbitrary strings: it prevents a
# stale/typo model name from burning a remote request only to get code 10007.
ASYNC_MODELS = (
    "PP-OCRv6",
    "PP-OCRv5",
    "PP-OCRv5-latin",
    "PP-StructureV3",
    "PaddleOCR-VL-1.6",
    "PaddleOCR-VL-1.5",
    "PaddleOCR-VL",
)
_OCR_ASYNC_MODELS = frozenset({"PP-OCRv6", "PP-OCRv5", "PP-OCRv5-latin"})
_STRUCTURE_ASYNC_MODELS = frozenset({"PP-StructureV3"})
_VL_ASYNC_MODELS = frozenset({"PaddleOCR-VL", "PaddleOCR-VL-1.5", "PaddleOCR-VL-1.6"})
MAX_ASYNC_LOCAL_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_RESULT_JSONL_BYTES = 64 * 1024 * 1024
MAX_RESULT_REDIRECTS = 3
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
_RETRYABLE_ASYNC_CODES = frozenset({10010, 12002})
_LAYOUT_IGNORED_LABELS = frozenset({
    "number", "footnote", "header", "header_image", "footer",
    "footer_image", "aside_text",
})


class PaddleAIStudioError(RuntimeError):
    """Cloud API error whose message is safe to display in the UI."""


def _cancelled(cancel_check: Callable[[], bool] | None) -> bool:
    try:
        return bool(cancel_check and cancel_check())
    except Exception:
        return False


def _sleep_cancelable(seconds: float, cancel_check=None) -> None:
    deadline = time.monotonic() + max(0.0, float(seconds))
    while time.monotonic() < deadline:
        if _cancelled(cancel_check):
            raise PaddleAIStudioError("PaddleOCR AI Studio 任务已取消。")
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))


def _validate_url(url: str, *, label: str) -> str:
    value = str(url or "").strip()
    if not value:
        raise ValueError(f"{label}不能为空。")
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"{label}必须是 https:// URL。")
    if parsed.username or parsed.password:
        raise ValueError(f"{label}不能包含 URL 用户名/密码。")
    return value.rstrip("/")


def _validate_result_url(url: str, *, label: str = "异步结果 URL") -> str:
    """Validate a server-provided signed result URL without overfitting BOS hosts.

    The official API currently returns pre-signed object-storage URLs.  They do
    not need the AI Studio Authorization header.  We deliberately avoid a hard
    host allow-list because Baidu can move result objects between BOS/CDN
    domains, but reject obvious local/private literal targets and local host
    names so a malformed upstream response cannot turn Novel Formatter into a
    trivial localhost/metadata fetcher.
    """
    value = _validate_url(url, label=label)
    host = (urlparse(value).hostname or "").strip().lower().rstrip(".")
    if not host:
        raise ValueError(f"{label}缺少主机名。")
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
        raise ValueError(f"{label}不能指向本机或 .local 地址。")
    try:
        addr = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        addr = None
    if addr is not None and (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    ):
        raise ValueError(f"{label}不能指向私有/本机 IP。")
    return value


def _safe_remote_text(value: object, *, secrets=(), limit: int = 1200) -> str:
    text = redact_secrets(value, secrets=secrets).replace("\x00", "").strip()
    return text[-max(1, int(limit)):]


def _decode_jsonl_bytes(value: bytes) -> str:
    try:
        return bytes(value).decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise PaddleAIStudioError(
            "PaddleOCR AI Studio 异步结果不是有效的 UTF-8 JSONL。"
        ) from exc


def _validate_service_url(url: str, *, label: str) -> str:
    """Validate a credential-bearing AI Studio endpoint.

    Unlike signed result URLs, this endpoint receives the user's Access Token.
    Restrict it to the official ``*.aistudio-app.com`` serving domain so a
    persisted/mistyped URL cannot exfiltrate a Keychain/DPAPI credential.
    """
    value = _validate_url(url, label=label)
    host = (urlparse(value).hostname or "").lower().rstrip(".")
    if host != "aistudio-app.com" and not host.endswith(".aistudio-app.com"):
        raise ValueError(
            f"{label}必须使用百度 AI Studio 官方 *.aistudio-app.com 域名；"
            "为防止 Access Token 被发送到第三方服务器，已拒绝该 URL。"
        )
    return value


def resolve_token(explicit: str = "") -> str:
    value = str(explicit or "").strip()
    if value:
        return value
    # Prefer the environment variable used by the official PaddleOCR SDK.
    # Keep the older Novel Formatter variable as a compatibility fallback.
    for env_name in ("PADDLEOCR_ACCESS_TOKEN", "AISTUDIO_ACCESS_TOKEN"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    try:
        from ai.secure_store import load_named_secret
        return str(load_named_secret(SECRET_NAME, account=SECRET_ACCOUNT) or "").strip()
    except Exception:
        return ""


def _extract_api_code(body: str) -> int | None:
    try:
        payload = json.loads(str(body or ""))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    raw = payload.get("code", payload.get("errorCode"))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _http_error_message(status: int, body: str = "", *, secrets=()) -> str:
    # Never echo Authorization headers or request objects.  Even error bodies
    # are bounded because upstream proxies can include a large HTML document.
    excerpt = _safe_remote_text(body, secrets=secrets, limit=1200)
    api_code = _extract_api_code(body)
    prefix = f"PaddleOCR AI Studio HTTP {int(status)}"
    if api_code == 12001:
        return prefix + "：已达到当日页面配额。"
    if api_code == 12002:
        return prefix + "：请求频率过高，请稍后重试。"
    if status == 401:
        return prefix + "：Token 无效或已过期。"
    if status == 403:
        return prefix + "：Token 错误、URL 与 Token 不匹配，或当前服务拒绝访问。"
    if status == 413:
        return prefix + "：上传文件超过服务大小限制。"
    if status == 422:
        return prefix + "：请求参数不被当前模型服务接受。" + (f" {excerpt}" if excerpt else "")
    if status == 429:
        return prefix + "：已达到页面配额或请求频率过高。"
    if status == 500:
        return (
            prefix
            + "：AI Studio 服务端内部错误。请优先使用官方异步 v2 jobs + "
            "PaddleOCR-VL-1.6；若仍失败通常是远端服务临时故障。"
            + (f" {excerpt}" if excerpt else "")
        )
    return prefix + (f"：{excerpt}" if excerpt else "。")


def _should_retry_response(response: httpx.Response) -> bool:
    status = int(response.status_code)
    if status not in _RETRYABLE_STATUS:
        return False
    if status != 429:
        return True
    # Current sync quota docs use HTTP 429 for the daily page ceiling, which
    # will not recover after a short sleep.  Async v2 code 12002 specifically
    # means request frequency too high and is safe to back off/retry.
    if response.headers.get("Retry-After"):
        return True
    return _extract_api_code(response.text) == 12002


def _response_json(response: httpx.Response) -> dict:
    try:
        data = response.json()
    except Exception as exc:
        raise PaddleAIStudioError(
            f"PaddleOCR AI Studio 返回了非 JSON 响应（HTTP {response.status_code}）。"
        ) from exc
    if not isinstance(data, dict):
        raise PaddleAIStudioError("PaddleOCR AI Studio 返回格式异常：顶层不是 JSON object。")
    return data


def _request_with_retries(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    cancel_check=None,
    redaction_secrets=(),
    **kwargs,
) -> httpx.Response:
    """Issue one request with quota-safe retry semantics.

    GET/HEAD/OPTIONS are replay-safe and may be retried after transient
    transport or HTTP failures.  OCR submissions are POST requests and can
    consume quota or create a remote job.  After a read/write/protocol failure
    we cannot know whether the server already accepted that POST, so we stop
    instead of blindly submitting the same page again.  Connect/pool failures
    that occur before a request can be accepted remain safe to retry.
    """
    retries = max(0, int(max_retries))
    verb = str(method or "GET").upper()
    replay_safe = verb in {"GET", "HEAD", "OPTIONS"}
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        # Never reuse a response object from an earlier attempt when the
        # current attempt fails before an HTTP response exists.
        response: httpx.Response | None = None
        if _cancelled(cancel_check):
            raise PaddleAIStudioError("PaddleOCR AI Studio 任务已取消。")
        try:
            response = client.request(verb, url, **kwargs)
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
            last_exc = exc
            definitely_pre_accept = isinstance(
                exc, (httpx.ConnectTimeout, httpx.ConnectError, httpx.PoolTimeout)
            )
            if not replay_safe and not definitely_pre_accept:
                raise PaddleAIStudioError(
                    "PaddleOCR AI Studio 提交请求后连接中断，远端是否已受理无法确认；"
                    "为避免重复任务或重复消耗页面配额，已停止自动重试。"
                ) from exc
            if attempt >= retries:
                raise PaddleAIStudioError(f"无法连接 PaddleOCR AI Studio：{type(exc).__name__}") from exc
        else:
            if response.status_code < 400:
                return response
            should_retry = _should_retry_response(response)
            if not replay_safe:
                # A response-level retry of a POST is only automatic when the
                # service explicitly says this request was throttled (12002).
                # 5xx/408 may have happened after the OCR job was accepted.
                should_retry = (
                    int(response.status_code) == 429
                    and _extract_api_code(response.text) == 12002
                )
            if not should_retry or attempt >= retries:
                raise PaddleAIStudioError(
                    _http_error_message(
                        response.status_code, response.text, secrets=redaction_secrets
                    )
                )
            last_exc = PaddleAIStudioError(
                _http_error_message(
                    response.status_code, response.text, secrets=redaction_secrets
                )
            )

        # Full jitter keeps several parallel OCR engines from retrying in lockstep.
        cap = min(12.0, 0.7 * (2 ** attempt))
        retry_after = 0.0
        try:
            retry_after = float(response.headers.get("Retry-After", "0")) if response is not None else 0.0
        except (TypeError, ValueError):
            retry_after = 0.0
        delay = max(retry_after, random.uniform(0.0, cap))
        _sleep_cancelable(delay, cancel_check)
    raise PaddleAIStudioError(f"PaddleOCR AI Studio 请求失败：{last_exc or '未知网络错误'}")




def _async_api_error_message(data: dict, *, secrets=()) -> str:
    try:
        code = int(data.get("code") or 0)
    except (TypeError, ValueError):
        code = -1
    detail = data.get("data") if isinstance(data.get("data"), dict) else {}
    raw_message = _safe_remote_text(detail.get("errorMsg") or data.get("msg") or "未知错误", secrets=secrets, limit=600)
    mapping = {
        401: "Token 无效或已过期。",
        10001: "上传文件为空。",
        10002: "文件 URL 无法识别。",
        10003: "文件大小超过服务限制。",
        10004: "文件格式不受支持。",
        10005: "文件内容无法解析。",
        10006: "文件页数超过服务限制。",
        10007: "模型参数无效或模型不存在。",
        10008: "optionalPayload 请求参数不适用于当前模型。",
        10009: "同一 batchId 的任务数量已达上限。",
        10010: "任务提交队列已满。",
        11001: "jobId 不存在。",
        11002: "job 已过期。",
        11003: "远端 OCR / 文档解析任务失败。",
        12001: "已达到当日页面配额。",
        12002: "请求频率过高。",
    }
    message = mapping.get(code, raw_message or "未知错误。")
    # For parameter / job failures the server's errorMsg is useful and normally
    # contains no credential. Bound it so an upstream response cannot flood UI.
    if code in {10007, 10008, 11003} and raw_message and raw_message not in message:
        message = f"{message} {raw_message[-600:]}"
    return f"PaddleOCR AI Studio v2 错误 {code}: {message}"


def _request_async_json_with_retries(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    max_retries: int,
    cancel_check=None,
    redaction_secrets=(),
    **kwargs,
) -> dict:
    """HTTP retry plus API-level retry for v2's code-in-200 responses."""
    retries = max(0, int(max_retries))
    for api_attempt in range(retries + 1):
        response = _request_with_retries(
            client, method, url, max_retries=retries,
            cancel_check=cancel_check, redaction_secrets=redaction_secrets, **kwargs,
        )
        data = _response_json(response)
        try:
            code = int(data.get("code") or 0)
        except (TypeError, ValueError):
            code = -1
        if code == 0:
            return data
        if code not in _RETRYABLE_ASYNC_CODES or api_attempt >= retries:
            raise PaddleAIStudioError(_async_api_error_message(data, secrets=redaction_secrets))
        cap = min(12.0, 0.7 * (2 ** api_attempt))
        _sleep_cancelable(random.uniform(0.0, cap), cancel_check)
    raise PaddleAIStudioError("PaddleOCR AI Studio v2 请求失败。")


def _quad_from_box(value) -> list[list[float]] | None:
    if not isinstance(value, (list, tuple)):
        return None
    if len(value) >= 4 and all(isinstance(item, (int, float)) for item in value[:4]):
        x1, y1, x2, y2 = (float(item) for item in value[:4])
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    points: list[list[float]] = []
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                points.append([float(item[0]), float(item[1])])
            except (TypeError, ValueError):
                continue
    if len(points) >= 4:
        return points[:4]
    return None


def _find_ocr_payloads(value) -> Iterable[dict]:
    """Yield nested PaddleOCR dictionaries containing recognized text arrays."""
    if isinstance(value, dict):
        if isinstance(value.get("rec_texts"), list):
            yield value
        # ``overall_ocr_res`` is common in layout/VL pipelines; recurse rather
        # than depending on one PaddleOCR release's exact nesting.
        for child in value.values():
            yield from _find_ocr_payloads(child)
    elif isinstance(value, list):
        for child in value:
            yield from _find_ocr_payloads(child)


def _ocr_blocks(payload: dict) -> list[dict]:
    texts = payload.get("rec_texts") or []
    scores = payload.get("rec_scores") or []
    polys = payload.get("rec_polys") or []
    boxes = payload.get("rec_boxes") or []
    out: list[dict] = []
    for index, raw_text in enumerate(texts):
        text = str(raw_text or "").strip()
        if not text:
            continue
        try:
            confidence = float(scores[index]) if index < len(scores) else 0.9
        except (TypeError, ValueError):
            confidence = 0.9
        box = None
        if index < len(polys):
            box = _quad_from_box(polys[index])
        if box is None and index < len(boxes):
            box = _quad_from_box(boxes[index])
        out.append({
            "text": text,
            "confidence": min(1.0, max(0.0, confidence)),
            "box": box,
            "layout_order": index,
            "label": "text",
        })
    return out


def _layout_blocks(payload: dict) -> list[dict]:
    raw = payload.get("parsing_res_list")
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        label = str(item.get("block_label") or item.get("label") or "text").strip().lower()
        if label in _LAYOUT_IGNORED_LABELS:
            continue
        text = str(
            item.get("block_content")
            or item.get("content")
            or item.get("text")
            or ""
        ).strip()
        if not text:
            continue
        try:
            order = int(item.get("block_order", index))
        except (TypeError, ValueError):
            order = index
        box = _quad_from_box(item.get("block_bbox") or item.get("bbox") or item.get("box"))
        try:
            confidence = float(
                item.get("block_score", item.get("score", item.get("confidence", 0.9)))
            )
        except (TypeError, ValueError):
            confidence = 0.9
        out.append({
            "text": text,
            "confidence": min(1.0, max(0.0, confidence)),
            "box": box,
            "layout_order": order,
            "label": label or "text",
        })
    out.sort(key=lambda item: int(item.get("layout_order", 0)))
    return out


def parse_pruned_result(value: dict | None) -> list[dict]:
    """Normalize PP-OCR / Structure / VL pruned JSON to Novel Formatter blocks."""
    payload = value if isinstance(value, dict) else {}
    layout = _layout_blocks(payload)
    if layout:
        return layout
    blocks: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for candidate in _find_ocr_payloads(payload):
        for block in _ocr_blocks(candidate):
            # Recursive result objects can expose the same overall OCR payload
            # through two aliases.  Do not duplicate identical text/geometry.
            signature = (block["text"], json.dumps(block.get("box"), sort_keys=True))
            if signature in seen:
                continue
            seen.add(signature)
            blocks.append(block)
    if blocks:
        for index, block in enumerate(blocks):
            block["layout_order"] = index
        return blocks
    return []


def parse_sync_response(data: dict, *, secrets=()) -> list[dict]:
    try:
        error_code = int(data.get("errorCode") or 0)
    except (TypeError, ValueError):
        error_code = -1
    if error_code:
        message = _safe_remote_text(data.get("errorMsg") or "未知错误", secrets=secrets, limit=600)
        raise PaddleAIStudioError(
            f"PaddleOCR AI Studio API 错误 {error_code}: {message}"
        )
    result = data.get("result")
    if not isinstance(result, dict):
        raise PaddleAIStudioError("PaddleOCR AI Studio 响应缺少 result。")

    blocks: list[dict] = []
    for key in ("ocrResults", "layoutParsingResults"):
        items = result.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            candidate = item.get("prunedResult")
            if isinstance(candidate, dict):
                blocks.extend(parse_pruned_result(candidate))
    if blocks:
        for index, block in enumerate(blocks):
            block["layout_order"] = index
        return blocks

    # Some service revisions place pruned fields directly under ``result``.
    blocks = parse_pruned_result(result)
    if blocks:
        return blocks

    # Last-resort VL fallback.  Structured geometry is preferred whenever
    # present, but retaining markdown is safer than silently returning no text.
    for container in (result, *(item for key in ("layoutParsingResults",) for item in (result.get(key) or []) if isinstance(item, dict))):
        markdown = container.get("markdown") if isinstance(container, dict) else None
        if isinstance(markdown, dict):
            text = str(markdown.get("text") or markdown.get("markdownText") or "").strip()
            if text:
                return [{"text": text, "confidence": 0.8, "box": None, "label": "markdown"}]
        elif isinstance(markdown, str) and markdown.strip():
            return [{"text": markdown.strip(), "confidence": 0.8, "box": None, "label": "markdown"}]
        if isinstance(container, dict):
            direct_markdown = str(container.get("markdownText") or container.get("markdown_text") or "").strip()
            if direct_markdown:
                return [{"text": direct_markdown, "confidence": 0.8, "box": None, "label": "markdown"}]
    return []


def parse_async_jsonl(text: str) -> list[dict]:
    blocks: list[dict] = []
    for line_no, line in enumerate(str(text or "").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PaddleAIStudioError(f"异步结果 JSONL 第 {line_no} 行无法解析。") from exc
        if not isinstance(item, dict):
            continue
        candidate = item.get("result") if isinstance(item.get("result"), dict) else item
        parsed = parse_pruned_result(candidate)
        if not parsed and isinstance(candidate, dict):
            # Some jobs return a shape equivalent to the synchronous result.
            try:
                parsed = parse_sync_response({"errorCode": 0, "result": candidate})
            except PaddleAIStudioError:
                parsed = []
        blocks.extend(parsed)
    for index, block in enumerate(blocks):
        block["layout_order"] = index
    return blocks


def _normalise_sync_family(value: str) -> str:
    family = str(value or "ppocr").strip().lower()
    aliases = {
        "ocr": "ppocr",
        "pp-ocr": "ppocr",
        "ppocr": "ppocr",
        "structure": "ppstructure",
        "pp-structure": "ppstructure",
        "ppstructure": "ppstructure",
        "pp-structurev3": "ppstructure",
        "vl": "vl",
        "paddleocr-vl": "vl",
        # Legacy persisted value from Hardened r3. Keep it as a conservative
        # common-denominator family so old profiles remain valid and do not
        # suddenly send a parameter that a VL endpoint rejects.
        "layout": "layout_vl",
        "layout_vl": "layout_vl",
    }
    family = aliases.get(family, family)
    if family not in SYNC_FAMILIES:
        raise ValueError(
            "同步服务类型必须是 ppocr、ppstructure、vl 或旧版 layout_vl；"
            f"收到 {value!r}。"
        )
    return family


def _build_sync_payload(
    *,
    content: str,
    sync_family: str,
    use_doc_orientation_classify: bool,
    use_doc_unwarping: bool,
    use_textline_orientation: bool,
) -> dict:
    family = _normalise_sync_family(sync_family)
    payload = {
        "file": content,
        "fileType": 1,
        "useDocOrientationClassify": bool(use_doc_orientation_classify),
        "useDocUnwarping": bool(use_doc_unwarping),
        "visualize": False,
    }
    # Official 2026 SDK exposes text-line orientation for both OCR and
    # PP-StructureV3, but not for PaddleOCR-VL.  ``layout_vl`` is the legacy
    # common-denominator profile and intentionally omits it.
    if family in {"ppocr", "ppstructure"}:
        payload["useTextlineOrientation"] = bool(use_textline_orientation)
    return payload


def _build_async_optional_payload(
    *,
    model: str,
    use_doc_orientation_classify: bool,
    use_doc_unwarping: bool,
    use_textline_orientation: bool,
) -> dict:
    payload = {
        "useDocOrientationClassify": bool(use_doc_orientation_classify),
        "useDocUnwarping": bool(use_doc_unwarping),
    }
    # The official 2026 SDK exposes text-line orientation for OCR and
    # PP-StructureV3. PaddleOCR-VL has a different option surface and must not
    # receive this key (otherwise the service can return code 10008).
    if str(model) in (_OCR_ASYNC_MODELS | _STRUCTURE_ASYNC_MODELS):
        payload["useTextlineOrientation"] = bool(use_textline_orientation)
    elif str(model) in _VL_ASYNC_MODELS:
        # PaddleOCR-VL's official jobs example accepts this document-parsing
        # option. Keep it explicit and disabled, matching the no-chart OCR
        # workflow used by Novel Formatter.
        payload["useChartRecognition"] = False
    return payload


def _sync_image(
    path: str,
    *,
    token: str,
    api_url: str,
    timeout: float,
    max_retries: int,
    use_doc_orientation_classify: bool,
    use_doc_unwarping: bool,
    use_textline_orientation: bool,
    sync_family: str = "ppocr",
    cancel_check=None,
    client: httpx.Client | None = None,
) -> list[dict]:
    content = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    payload = _build_sync_payload(
        content=content,
        sync_family=sync_family,
        use_doc_orientation_classify=use_doc_orientation_classify,
        use_doc_unwarping=use_doc_unwarping,
        use_textline_orientation=use_textline_orientation,
    )
    own_client = client is None
    client = client or httpx.Client(timeout=httpx.Timeout(float(timeout)))
    try:
        response = _request_with_retries(
            client, "POST", _validate_service_url(api_url, label="同步 API URL"),
            headers={
                "Authorization": f"token {token}",
                "Content-Type": "application/json",
                "User-Agent": "NovelFormatterStudio/AIStudio-PaddleOCR",
            },
            json=payload,
            max_retries=max_retries,
            cancel_check=cancel_check,
            redaction_secrets=(token,),
        )
        return parse_sync_response(_response_json(response), secrets=(token,))
    finally:
        if own_client:
            client.close()


def _async_envelope(data: dict, *, secrets=()) -> dict:
    """Validate the documented v2 envelope and return its ``data`` object."""
    try:
        code = int(data.get("code") or 0)
    except (TypeError, ValueError):
        code = -1
    if code:
        raise PaddleAIStudioError(_async_api_error_message(data, secrets=secrets))
    payload = data.get("data")
    if isinstance(payload, dict):
        return payload
    # Keep a compatibility fallback for pre-release/example gateways that put
    # job fields at top-level, but the documented API uses data.*.
    return data


def _job_state(data: dict, *, secrets=()) -> tuple[str, dict]:
    result = _async_envelope(data, secrets=secrets)
    state = str(result.get("state") or result.get("status") or "").strip().lower()
    return state, result


def _download_result_jsonl(
    client: httpx.Client,
    url: str,
    *,
    max_retries: int,
    cancel_check=None,
) -> str:
    """Download signed JSONL with a hard streaming size cap.

    Authorization is intentionally absent. Redirects are followed manually so
    every destination is re-validated and a malicious/incorrect signed URL
    cannot bounce the client to localhost or a private literal address.
    """
    current = _validate_result_url(url, label="异步结果 URL")
    redirects = 0
    while True:
        retries = max(0, int(max_retries))
        redirected_to: str | None = None
        for attempt in range(retries + 1):
            if _cancelled(cancel_check):
                raise PaddleAIStudioError("PaddleOCR AI Studio 任务已取消。")
            try:
                with client.stream(
                    "GET",
                    current,
                    headers={"User-Agent": "NovelFormatterStudio/AIStudio-PaddleOCR"},
                    follow_redirects=False,
                ) as response:
                    status = int(response.status_code)
                    if status in {301, 302, 303, 307, 308}:
                        location = str(response.headers.get("Location") or "").strip()
                        if not location:
                            raise PaddleAIStudioError("异步结果下载重定向缺少 Location。")
                        redirected_to = _validate_result_url(
                            urljoin(current, location), label="异步结果重定向 URL"
                        )
                        break
                    if status >= 400:
                        if _should_retry_response(response) and attempt < retries:
                            cap = min(12.0, 0.7 * (2 ** attempt))
                            _sleep_cancelable(random.uniform(0.0, cap), cancel_check)
                            continue
                        raise PaddleAIStudioError(_http_error_message(status))

                    raw_length = response.headers.get("Content-Length")
                    if raw_length:
                        try:
                            announced = int(raw_length)
                        except (TypeError, ValueError):
                            announced = -1
                        if announced > MAX_RESULT_JSONL_BYTES:
                            raise PaddleAIStudioError(
                                "PaddleOCR AI Studio 异步结果 JSONL 超过 64 MB 安全上限；已停止下载。"
                            )

                    chunks = bytearray()
                    for chunk in response.iter_bytes():
                        if _cancelled(cancel_check):
                            raise PaddleAIStudioError("PaddleOCR AI Studio 任务已取消。")
                        if not chunk:
                            continue
                        if len(chunks) + len(chunk) > MAX_RESULT_JSONL_BYTES:
                            raise PaddleAIStudioError(
                                "PaddleOCR AI Studio 异步结果 JSONL 超过 64 MB 安全上限；已停止下载。"
                            )
                        chunks.extend(chunk)
                    return _decode_jsonl_bytes(bytes(chunks))
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                if attempt >= retries:
                    raise PaddleAIStudioError(
                        f"无法下载 PaddleOCR AI Studio 异步结果：{type(exc).__name__}"
                    ) from exc
                cap = min(12.0, 0.7 * (2 ** attempt))
                _sleep_cancelable(random.uniform(0.0, cap), cancel_check)
        if redirected_to is None:
            raise PaddleAIStudioError("PaddleOCR AI Studio 异步结果下载失败。")
        redirects += 1
        if redirects > MAX_RESULT_REDIRECTS:
            raise PaddleAIStudioError("PaddleOCR AI Studio 异步结果重定向次数过多。")
        current = redirected_to


def _async_image(
    path: str,
    *,
    token: str,
    job_url: str,
    model: str,
    timeout: float,
    max_retries: int,
    poll_interval: float,
    max_job_wait: float,
    use_doc_orientation_classify: bool,
    use_doc_unwarping: bool,
    use_textline_orientation: bool,
    cancel_check=None,
    client: httpx.Client | None = None,
) -> list[dict]:
    if model not in ASYNC_MODELS:
        raise ValueError(
            f"异步 v2 当前只允许官方 SDK 已登记的模型：{', '.join(ASYNC_MODELS)}；收到 {model!r}。"
        )
    base = _validate_service_url(job_url or DEFAULT_ASYNC_JOB_URL, label="异步 jobs URL")
    optional_payload = _build_async_optional_payload(
        model=model,
        use_doc_orientation_classify=use_doc_orientation_classify,
        use_doc_unwarping=use_doc_unwarping,
        use_textline_orientation=use_textline_orientation,
    )
    own_client = client is None
    client = client or httpx.Client(timeout=httpx.Timeout(float(timeout)))
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "NovelFormatterStudio/AIStudio-PaddleOCR",
    }
    try:
        upload_path = Path(path)
        try:
            upload_size = upload_path.stat().st_size
        except OSError as exc:
            raise PaddleAIStudioError(f"无法读取待上传文件：{upload_path.name}") from exc
        if upload_size > MAX_ASYNC_LOCAL_UPLOAD_BYTES:
            raise PaddleAIStudioError(
                "PaddleOCR AI Studio 异步 API 本地上传上限为 50 MB；"
                f"当前文件约 {upload_size / (1024 * 1024):.1f} MB。"
            )
        # Use immutable bytes rather than a live file handle. A multipart retry
        # must be replayable; reusing an already-consumed stream can otherwise
        # send an empty/truncated file on the second attempt after 503/429.
        upload_bytes = upload_path.read_bytes()
        submitted = _request_async_json_with_retries(
            client, "POST", base,
            headers=headers,
            data={"model": model, "optionalPayload": json.dumps(optional_payload)},
            files={"file": (upload_path.name, upload_bytes, "application/octet-stream")},
            max_retries=max_retries,
            cancel_check=cancel_check,
            redaction_secrets=(token,),
        )
        result = _async_envelope(submitted, secrets=(token,))
        job_id = str(result.get("jobId") or result.get("job_id") or "").strip()
        if not job_id:
            raise PaddleAIStudioError("异步 PaddleOCR 响应缺少 jobId。")

        deadline = time.monotonic() + max(30.0, float(max_job_wait))
        status_result: dict = {}
        while True:
            if _cancelled(cancel_check):
                raise PaddleAIStudioError("PaddleOCR AI Studio 任务已取消。")
            if time.monotonic() >= deadline:
                raise PaddleAIStudioError(f"PaddleOCR AI Studio 异步任务 {job_id} 等待超时。")
            status_data = _request_async_json_with_retries(
                client, "GET", f"{base}/{job_id}", headers=headers,
                max_retries=max_retries, cancel_check=cancel_check,
                redaction_secrets=(token,),
            )
            state, status_result = _job_state(status_data, secrets=(token,))
            if state in {"done", "success", "succeeded", "completed", "finished"}:
                break
            if state in {"failed", "failure", "error", "cancelled", "canceled"}:
                message = _safe_remote_text(
                    status_result.get("errorMsg") or status_result.get("message") or state,
                    secrets=(token,), limit=600,
                )
                raise PaddleAIStudioError(f"PaddleOCR AI Studio 异步任务失败：{message}")
            _sleep_cancelable(max(0.5, float(poll_interval)), cancel_check)

        result_url = status_result.get("resultUrl")
        if isinstance(result_url, dict):
            json_url = str(result_url.get("jsonUrl") or result_url.get("json_url") or "").strip()
        else:
            json_url = str(status_result.get("jsonUrl") or status_result.get("json_url") or "").strip()
        if not json_url:
            raise PaddleAIStudioError("异步任务完成，但响应缺少 resultUrl.jsonUrl。")
        result_text = _download_result_jsonl(
            client, json_url, max_retries=max_retries, cancel_check=cancel_check
        )
        return parse_async_jsonl(result_text)
    finally:
        if own_client:
            client.close()


def run(
    *,
    mode: str = "async_v2",
    token: str = "",
    api_url: str = "",
    sync_family: str = "ppocr",
    async_job_url: str = DEFAULT_ASYNC_JOB_URL,
    async_model: str = "PaddleOCR-VL-1.6",
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    max_job_wait: float = 900.0,
    use_doc_orientation_classify: bool = False,
    use_doc_unwarping: bool = False,
    use_textline_orientation: bool = False,
    verbose: bool = True,
    **kwargs,
):
    access_token = resolve_token(token)
    if not access_token:
        raise ValueError(
            "请填写 PaddleOCR AI Studio Access Token，或设置 PADDLEOCR_ACCESS_TOKEN（兼容 AISTUDIO_ACCESS_TOKEN）；"
            "macOS/Windows 也可在 Novel Formatter 中保存到系统凭据库。"
        )
    service_mode = str(mode or "async_v2").strip().lower()
    if service_mode not in {"sync", "async_v2"}:
        raise ValueError("PaddleOCR AI Studio 模式必须是 sync 或 async_v2。")
    if service_mode == "sync":
        sync_url = _validate_service_url(api_url, label="同步 API URL")
        sync_service_family = _normalise_sync_family(sync_family)
    else:
        sync_url = ""
        sync_service_family = "ppocr"
        _validate_service_url(async_job_url or DEFAULT_ASYNC_JOB_URL, label="异步 jobs URL")

    run_stats = {
        "attempted_pages": 0,
        "successful_pages": 0,
        "failed_pages": 0,
        "failure_samples": [],
    }

    def worker_fn(ocr_paths, cancel_check):
        # One HTTP client is reused for the entire book/run to keep HTTP/TLS
        # connection setup out of the per-page hot path.
        with httpx.Client(timeout=httpx.Timeout(float(request_timeout))) as client:
            for path in ocr_paths:
                if _cancelled(cancel_check):
                    break
                run_stats["attempted_pages"] += 1
                try:
                    if service_mode == "sync":
                        blocks = _sync_image(
                            path,
                            token=access_token,
                            api_url=sync_url,
                            timeout=request_timeout,
                            max_retries=max_retries,
                            use_doc_orientation_classify=use_doc_orientation_classify,
                            use_doc_unwarping=use_doc_unwarping,
                            use_textline_orientation=use_textline_orientation,
                            sync_family=sync_service_family,
                            cancel_check=cancel_check,
                            client=client,
                        )
                    else:
                        blocks = _async_image(
                            path,
                            token=access_token,
                            job_url=async_job_url or DEFAULT_ASYNC_JOB_URL,
                            model=str(async_model or "PaddleOCR-VL-1.6"),
                            timeout=request_timeout,
                            max_retries=max_retries,
                            poll_interval=poll_interval,
                            max_job_wait=max_job_wait,
                            use_doc_orientation_classify=use_doc_orientation_classify,
                            use_doc_unwarping=use_doc_unwarping,
                            use_textline_orientation=use_textline_orientation,
                            cancel_check=cancel_check,
                            client=client,
                        )
                    run_stats["successful_pages"] += 1
                    yield path, blocks, None
                except Exception as exc:
                    run_stats["failed_pages"] += 1
                    safe_error = redact_secrets(exc, secrets=(access_token,))
                    if len(run_stats["failure_samples"]) < 5:
                        run_stats["failure_samples"].append(str(safe_error)[:600])
                    yield path, None, safe_error

    doc = run_ocr_engine(
        worker_fn,
        source_engine=f"paddle_aistudio_{service_mode}",
        verbose=verbose,
        **kwargs,
    )
    stats_snapshot = {
        "mode": service_mode,
        "sync_family": sync_service_family if service_mode == "sync" else None,
        "async_model": str(async_model or "PaddleOCR-VL-1.6") if service_mode == "async_v2" else None,
        "attempted_pages": int(run_stats["attempted_pages"]),
        "successful_pages": int(run_stats["successful_pages"]),
        "failed_pages": int(run_stats["failed_pages"]),
        "failure_samples": list(run_stats["failure_samples"]),
    }
    doc.metadata.__dict__["paddle_aistudio_api"] = stats_snapshot
    if (
        stats_snapshot["attempted_pages"] > 0
        and stats_snapshot["successful_pages"] == 0
        and not _cancelled(kwargs.get("cancel_check"))
    ):
        detail = stats_snapshot["failure_samples"][0] if stats_snapshot["failure_samples"] else "所有页面均调用失败。"
        raise PaddleAIStudioError(
            "PaddleOCR AI Studio 本轮没有任何页面成功返回；"
            f"已停止把空结果送入多模型共识。首个错误：{detail}"
        )
    return doc


__all__ = [
    "ASYNC_MODELS",
    "SYNC_FAMILIES",
    "DEFAULT_ASYNC_JOB_URL",
    "PaddleAIStudioError",
    "parse_pruned_result",
    "parse_sync_response",
    "parse_async_jsonl",
    "resolve_token",
    "run",
]
