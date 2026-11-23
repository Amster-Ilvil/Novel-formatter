"""Local text and image input bridge for official Sign in with ChatGPT DevKit.

Novel Formatter never parses or stores OAuth tokens. The official @siwc/local
SDK owns OAuth, token rotation, and an OS-encrypted connection store in a local
Node helper. A fresh process per request avoids a network-listening bridge.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .base import AIProvider


class ChatGPTSubscriptionError(RuntimeError):
    """Sanitized error: never includes OAuth tokens, submitted prompts, or SDK output."""


def _node_path() -> str:
    candidates = [shutil.which('node'), '/opt/homebrew/bin/node', '/usr/local/bin/node']
    for name in candidates:
        if name and Path(name).is_file() and os.access(name, os.X_OK):
            return name
    raise ChatGPTSubscriptionError('未安装 Node.js 22.12+；请先执行 ChatGPT 订阅登录安装脚本。')


def _runtime_ready() -> None:
    if sys.platform != 'darwin':
        raise ChatGPTSubscriptionError('订阅授权只在用户自己的 macOS 本地运行，不支持在云端 Linux 代为登录。')
    base = Path.home() / 'Library' / 'Application Support' / 'NovelFormatter'
    if not (base / 'siwc-devkit' / 'packages' / 'local' / 'dist' / 'index.js').is_file():
        raise ChatGPTSubscriptionError('尚未安装官方 DevKit，请执行 siwc_bridge/install_macos.sh。')
    if not (base / 'siwc-keychain-crypto').is_file():
        raise ChatGPTSubscriptionError('macOS Keychain 凭据工具缺失，请重新执行安装脚本。')


def chatgpt_subscription_request(action: str, *, timeout: int = 180, **payload: Any) -> Any:
    """Execute one allowlisted action with no HTTP listener or secret IPC fields."""
    if action not in {'session', 'login', 'logout', 'profiles', 'models', 'respond'}:
        raise ValueError('Unsupported ChatGPT subscription action')
    _runtime_ready()
    command = {'action': action, **payload}
    data = json.dumps(command, ensure_ascii=False, separators=(',', ':')) + '\n'
    bridge = Path(__file__).resolve().parent.parent / 'siwc_bridge' / 'bridge.mjs'
    # Do not allow developer/debug environment variables to intercept OAuth or
    # dump user text. The official SDK also disables its verbose HTTP logging.
    env = os.environ.copy()
    for key in ('NODE_OPTIONS', 'OPENAI_LOG', 'OPENAI_CUSTOM_HEADERS', 'NF_SIWC_SDK_ROOT'):
        env.pop(key, None)
    try:
        result = subprocess.run(
            [_node_path(), str(bridge)], input=data, text=True, capture_output=True,
            timeout=max(10, min(int(timeout), 660)), env=env, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ChatGPTSubscriptionError('ChatGPT 请求超时；如正在登录，请在浏览器中完成授权后重试。') from exc
    except OSError as exc:
        raise ChatGPTSubscriptionError('无法启动本地 ChatGPT 订阅桥接进程。') from exc
    if result.returncode != 0:
        raise ChatGPTSubscriptionError('ChatGPT 本地桥接进程异常退出，请检查官方 DevKit 安装。')
    try:
        reply = json.loads(result.stdout.strip())
    except (ValueError, TypeError) as exc:
        raise ChatGPTSubscriptionError('ChatGPT 桥接返回格式无效。') from exc
    if not isinstance(reply, dict) or not reply.get('ok'):
        error = reply.get('error', {}) if isinstance(reply, dict) else {}
        code = str(error.get('code', 'siwc_error')) if isinstance(error, dict) else 'siwc_error'
        # Error codes are allowlisted on the Node side; don't echo token-bearing body.
        if not code.replace('_', '').replace('-', '').isalnum() or len(code) > 70:
            code = 'siwc_error'
        raise ChatGPTSubscriptionError(f'ChatGPT 订阅请求失败（{code}）。请检查登录、授权、模型及额度。')
    return reply.get('data')


class ChatGPTSubscriptionProvider(AIProvider):
    """Use officially supported Responses inference with OS-secured OAuth.

    Text is handled by this AIProvider; the image path is implemented by
    MultimodalClient using the exact same local bridge and account session.
    No OpenAI API key is synthesized, read, or forwarded.
    """

    @property
    def name(self) -> str:
        return 'chatgpt_subscription'

    def __init__(self, api_key: str = '', model: str = '', **kwargs: Any):
        super().__init__('', model, **kwargs)
        self.model = model
        if not model:
            raise ChatGPTSubscriptionError('请先登录 ChatGPT，并从实际账号模型列表选择模型。')

    def _call_llm(self, prompt: str, temperature: float = 0) -> str:
        # Temperature is unsupported by this restricted preview route.
        # Do not send misleading fake params or silently fallback to paid API.
        response = chatgpt_subscription_request(
            'respond', model=self.model, prompt=prompt,
            timeout=int(self.kwargs.get('request_timeout', 180)),
        )
        content = response.get('text') if isinstance(response, dict) else None
        if not isinstance(content, str):
            raise ChatGPTSubscriptionError('ChatGPT 订阅响应缺少文本内容。')
        # Plan usage is account-managed; SDK returns no token usage. Do not
        # fabricate token counts in the application's usage statistics.
        return content
