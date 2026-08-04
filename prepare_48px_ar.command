#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
CACHE_DIR="$ROOT/.model-cache/manga-48px-ar"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
elif [[ -x "$ROOT/venv/bin/python" ]]; then
  PYTHON="$ROOT/venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
else
  echo "未找到 Python 3。" >&2
  exit 1
fi

mkdir -p "$CACHE_DIR"
cd "$ROOT"
"$PYTHON" "$ROOT/adapters/manga_48px_runtime.py" --cache-dir "$CACHE_DIR"
printf '\n准备完成。按回车键关闭窗口。\n'
read -r _ || true
