#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SOURCE="$ROOT/tools/apple_vision_helper/AppleVisionOCRHelper.swift"
OUTPUT_DIR="$ROOT/tools/apple_vision_helper/bin"
OUTPUT="$OUTPUT_DIR/apple_vision_helper"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Apple Vision Helper 只能在 macOS 上编译。" >&2
  exit 1
fi
if ! command -v xcrun >/dev/null 2>&1; then
  echo "未找到 xcrun。请先安装 Xcode Command Line Tools。" >&2
  exit 1
fi
if [[ ! -f "$SOURCE" ]]; then
  echo "缺少源码：$SOURCE" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-13.0}"
MODULE_CACHE_DIR="$ROOT/tools/apple_vision_helper/.module-cache"
mkdir -p "$MODULE_CACHE_DIR"
export CLANG_MODULE_CACHE_PATH="$MODULE_CACHE_DIR"
xcrun swiftc "$SOURCE" \
  -parse-as-library \
  -O \
  -framework Foundation \
  -framework Vision \
  -framework VisionKit \
  -framework AppKit \
  -o "$OUTPUT"
chmod +x "$OUTPUT"
echo "Apple Vision Helper 已生成：$OUTPUT"
