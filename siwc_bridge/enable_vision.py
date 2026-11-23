#!/usr/bin/env python3
"""Novel Formatter noncommercial DevKit adaptation: opt-in image inputs.

Applies narrowly to the reviewed official @siwc/local version pinned in install_macos.sh.
Original upstream remains a separate checkout, with its upstream license/notices.
Modified Work files retain copyright and are marked in this adapter script.
The SDK continues to own OAuth, encrypted credentials and Responses streaming.
"""
from __future__ import annotations

from pathlib import Path
import sys

TYPES_OLD = '''  content: string;
}

export interface StreamResponseOptions {'''
TYPES_NEW = '''  // Novel Formatter 2026-11-22 noncommercial adaptation: image input content.
  content: string | Array<
    | { type: "input_text"; text: string }
    | { type: "input_image"; image_url: string; detail?: "auto" | "low" | "high" }
  >;
}

export interface StreamResponseOptions {'''

RESP_OLD = '''  if (!Array.isArray(input) || input.some((message) =>
    !isObject(message) || !["user", "assistant", "developer"].includes(String(message.role)) || typeof message.content !== "string"
  )) {'''
RESP_NEW = '''  // Novel Formatter 2026-11-22 noncommercial adaptation: validated inline image data.
  // Keep image data inside SDK-owned Responses stream; never expose OAuth tokens.
  const validContent = (content: unknown): boolean => {
    if (typeof content === "string") return true;
    if (!Array.isArray(content) || content.length === 0 || content.length > 11) return false;
    return content.every((item: unknown) => {
      if (!isObject(item)) return false;
      if (item.type === "input_text") return typeof item.text === "string";
      if (item.type !== "input_image" || typeof item.image_url !== "string") return false;
      if (item.image_url.length > 20 * 1024 * 1024) return false;
      if (!/^data:image\\/(png|jpeg|webp);base64,[A-Za-z0-9+/]+={0,2}$/.test(item.image_url)) return false;
      return item.detail === undefined || ["auto", "low", "high"].includes(String(item.detail));
    });
  };
  if (!Array.isArray(input) || input.some((message) =>
    !isObject(message) || !["user", "assistant", "developer"].includes(String(message.role)) || !validContent(message.content)
  )) {'''


def patch(path: Path, before: str, after: str) -> str:
    original = path.read_text(encoding='utf-8')
    if after in original:
        return 'already-patched'
    if original.count(before) != 1:
        raise RuntimeError(f'Unsupported upstream DevKit version: unexpected code in {path}')
    path.write_text(original.replace(before, after), encoding='utf-8')
    return 'patched'


def run(root: Path) -> None:
    root = root.resolve()
    target = root / 'packages' / 'local' / 'src'
    if not (root / 'LICENSE').exists():
        raise RuntimeError('Missing upstream DevKit LICENSE')
    print('types.ts:', patch(target / 'types.ts', TYPES_OLD, TYPES_NEW))
    print('responses.ts:', patch(target / 'responses.ts', RESP_OLD, RESP_NEW))
    print('A noncommercial adaptation was applied; upstream license and notices remain in the checkout.')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: enable_vision.py <official-devkit-directory>')
    run(Path(sys.argv[1]))
