#!/bin/zsh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
APP="$ROOT/AppleStrokeRecognizer.app"
CONTENTS="$APP/Contents"
MACOS_DIR="$CONTENTS/MacOS"
OUT="$MACOS_DIR/apple-stroke-recognizer"
COMPAT_DIR="$ROOT/bin"
COMPAT_OUT="$COMPAT_DIR/apple-stroke-recognizer"
SRC="$ROOT/Sources/main.swift"

if ! command -v xcrun >/dev/null 2>&1; then
  echo "错误：未找到 xcrun。请安装 Xcode 27 或更高版本。" >&2
  exit 2
fi
SDK_VERSION="$(xcrun --sdk macosx --show-sdk-version 2>/dev/null || true)"
SDK_MAJOR="${SDK_VERSION%%.*}"
if [[ -z "$SDK_MAJOR" || "$SDK_MAJOR" -lt 27 ]]; then
  echo "错误：当前 macOS SDK 为 ${SDK_VERSION:-未知}，PKStrokeRecognizer 需要 Xcode 27 / macOS 27 SDK。" >&2
  exit 3
fi

rm -rf "$APP"
mkdir -p "$MACOS_DIR" "$COMPAT_DIR"
cat > "$CONTENTS/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleDevelopmentRegion</key><string>ja</string>
<key>CFBundleExecutable</key><string>apple-stroke-recognizer</string>
<key>CFBundleIdentifier</key><string>com.novelformatter.apple-stroke-recognizer</string>
<key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
<key>CFBundleName</key><string>AppleStrokeRecognizer</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleShortVersionString</key><string>1.8</string>
<key>CFBundleVersion</key><string>11</string>
<key>LSMinimumSystemVersion</key><string>27.0</string>
<key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST

xcrun --sdk macosx swiftc \
  -O -parse-as-library \
  -framework Foundation -framework AppKit -framework PencilKit -framework Vision \
  "$SRC" -o "$OUT"
chmod +x "$OUT"
if command -v codesign >/dev/null 2>&1; then
  codesign --force --deep --sign - "$APP" >/dev/null 2>&1 || true
fi
ln -sf "../AppleStrokeRecognizer.app/Contents/MacOS/apple-stroke-recognizer" "$COMPAT_OUT"

echo "已生成 Apple 识别桥接与手动测试 App：$APP"
"$OUT" --status
