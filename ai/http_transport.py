"""HTTP opener for direct provider requests."""
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, build_opener

from .redaction import redact_secrets


class RejectRedirects(HTTPRedirectHandler):
    """Return HTTP 3xx as an error without contacting the Location URL."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def open_direct(request, *, timeout):
    return build_opener(RejectRedirects()).open(request, timeout=timeout)


def validate_service_url(url: str) -> str:
    """Accept explicit HTTP(S) endpoints while keeping credentials out of URLs."""
    try:
        parts = urlsplit(url)
        valid = parts.scheme in {"https", "http"} and parts.hostname and parts.port != 0
    except ValueError:
        valid = False
    if not valid or parts.username is not None or parts.password is not None or parts.fragment:
        raise ValueError("Base URL 必须是有效的 HTTP(S) 地址，且不能包含登录凭据或片段。")
    return url


class ProviderRequestError(RuntimeError):
    """An SDK failure that carries only safe diagnostic and retry information."""

    def __init__(self, message: str, *, status_code=None, retry_after=None):
        super().__init__(message)
        self.status_code = status_code
        self.headers = {"retry-after": retry_after} if retry_after is not None else {}


def safe_sdk_error(exc: Exception, api_keys: str) -> Exception:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        headers = getattr(getattr(exc, "response", None), "headers", None) or {}
        retry_after = None
        try:
            retry_after = float(headers.get("retry-after"))
        except (TypeError, ValueError):
            pass
        hint = "，请填写服务的最终 Base URL" if 300 <= status < 400 else ""
        return ProviderRequestError(f"AI 服务请求失败（HTTP {status}）{hint}", status_code=status,
                                    retry_after=retry_after)
    if type(exc).__module__.split(".")[0] in {"openai", "anthropic", "httpx", "httpcore"}:
        return ProviderRequestError("AI 服务连接失败（connection error）")
    message = str(exc)
    safe_message = redact_secrets(message, secrets=(key.strip() for key in api_keys.split(",")))
    return ProviderRequestError(safe_message) if safe_message != message else exc
