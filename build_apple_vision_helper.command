#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SOURCE="$ROOT/tools/apple_vision_helper/AppleVisionOCRHelper.swift"
ARCH_RAW="$(uname -m)"
case "$ARCH_RAW" in
  arm64|aarch64) ARCH="arm64" ;;
  x86_64|amd64) ARCH="x86_64" ;;
  *) ARCH="$ARCH_RAW" ;;
esac
DEFAULT_OUTPUT_DIR="$HOME/Library/Caches/NovelFormatter/apple_vision_helper/$ARCH"
OUTPUT_DIR="${NOVEL_FORMATTER_APPLE_VISION_HELPER_DIR:-$DEFAULT_OUTPUT_DIR}"
OUTPUT="$OUTPUT_DIR/apple_vision_helper"
SDK_STAMP="$OUTPUT_DIR/apple_vision_helper.sdk-version"
ARCH_STAMP="$OUTPUT_DIR/apple_vision_helper.arch"
SOURCE_STAMP="$OUTPUT_DIR/apple_vision_helper.source-sha256"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Apple Vision Helper 只能在 macOS 上编译。" >&2
  exit 1
fi
if [[ ! -f "$SOURCE" ]]; then
  echo "缺少源码：$SOURCE" >&2
  exit 1
fi

# Prefer the active Xcode/Command Line Tools toolchain, but also accept a
# standalone swiftc already on PATH. This keeps fresh Intel and Apple Silicon
# installations from failing solely because xcrun itself is absent.
if command -v xcrun >/dev/null 2>&1; then
  SWIFTC="$(xcrun --find swiftc 2>/dev/null || true)"
  SDK_PATH="$(xcrun --sdk macosx --show-sdk-path 2>/dev/null || true)"
  SDK_VERSION="$(xcrun --sdk macosx --show-sdk-version 2>/dev/null || true)"
else
  SWIFTC="$(command -v swiftc || true)"
  SDK_PATH=""
  SDK_VERSION=""
fi
if [[ -z "$SWIFTC" || ! -x "$SWIFTC" ]]; then
  SWIFTC="$(command -v swiftc || true)"
fi
if [[ -z "$SWIFTC" ]]; then
  echo "未找到 Swift 编译器。请安装 Xcode Command Line Tools（xcode-select --install）或 Xcode。" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-13.0}"
export MACOSX_DEPLOYMENT_TARGET="$DEPLOYMENT_TARGET"
MODULE_CACHE_DIR="$OUTPUT_DIR/.module-cache"
mkdir -p "$MODULE_CACHE_DIR"
export CLANG_MODULE_CACHE_PATH="$MODULE_CACHE_DIR"
SWIFT_DEFINES=()
SDK_MAJOR="${SDK_VERSION%%.*}"
if [[ "$SDK_MAJOR" =~ ^[0-9]+$ ]] && (( SDK_MAJOR >= 26 )); then
  SWIFT_DEFINES+=("-D" "NOVEL_FORMATTER_VISION26")
fi
if [[ "$SDK_MAJOR" =~ ^[0-9]+$ ]] && (( SDK_MAJOR >= 27 )); then
  SWIFT_DEFINES+=("-D" "NOVEL_FORMATTER_VISION27")
fi
TMP_OUTPUT="$OUTPUT_DIR/.apple_vision_helper.$$.tmp"
cleanup() { rm -f "$TMP_OUTPUT"; }
trap cleanup EXIT

TARGET="${ARCH}-apple-macosx${DEPLOYMENT_TARGET}"
ARGS=(
  "$SOURCE"
  "${SWIFT_DEFINES[@]}"
  -module-cache-path "$MODULE_CACHE_DIR"
  -parse-as-library
  -O
  -target "$TARGET"
  -framework Foundation
  -framework Vision
  -framework VisionKit
  -framework AppKit
  -o "$TMP_OUTPUT"
)
if [[ -n "$SDK_PATH" ]]; then
  ARGS+=( -sdk "$SDK_PATH" )
fi
"$SWIFTC" "${ARGS[@]}"
chmod +x "$TMP_OUTPUT"
if command -v codesign >/dev/null 2>&1; then
  codesign --force --sign - --timestamp=none "$TMP_OUTPUT" >/dev/null 2>&1 || true
fi
mv -f "$TMP_OUTPUT" "$OUTPUT"
trap - EXIT
chmod +x "$OUTPUT"
if [[ -n "$SDK_VERSION" ]]; then
  printf "%s\n" "$SDK_VERSION" > "$SDK_STAMP"
fi
printf "%s\n" "$ARCH" > "$ARCH_STAMP"
if [[ -n "${NOVEL_FORMATTER_APPLE_VISION_HELPER_SOURCE_SHA256:-}" ]]; then
  printf "%s\n" "$NOVEL_FORMATTER_APPLE_VISION_HELPER_SOURCE_SHA256" > "$SOURCE_STAMP"
fi
echo "Apple Vision Helper 已生成：${OUTPUT}（macOS SDK ${SDK_VERSION:-unknown} · ${ARCH} · target ${TARGET}）"
