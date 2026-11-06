#!/usr/bin/env bash
# Builds the native SwiftUI shell "Novel Formatter Studio (Swift).app".
# The Python core is embedded under Contents/Resources/app and launched at
# runtime through bootstrap.py — identical to packaging/launcher.zsh.
# Env: VERSION (required), LABEL (default: machine arch), EXPECTED_ARCH
# (default: machine arch).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VERSION="${VERSION:-$(tr -d '[:space:]' < VERSION)}"
LABEL="${LABEL:-$(uname -m)}"
EXPECTED_ARCH="${EXPECTED_ARCH:-$(uname -m)}"
PRODUCT_NAME="Novel Formatter Studio"
BASENAME="NovelFormatterSwift_${VERSION}_macOS_${LABEL}"
DIST="$ROOT/dist"
PACKAGE="$ROOT/package"
APP="$DIST/NovelFormatterSwift/${PRODUCT_NAME}.app"
CONTENTS="$APP/Contents"
APP_ZIP="$PACKAGE/${BASENAME}.app.zip"
DMG="$PACKAGE/${BASENAME}.dmg"
DMG_STAGE="$ROOT/.dmg-stage-swift"

test "$(uname -m)" = "$EXPECTED_ARCH"

python3 scripts/privacy_audit.py

echo "==> Compiling SwiftUI shell ($EXPECTED_ARCH)"
rm -rf "$DIST/NovelFormatterSwift" "$PACKAGE"
mkdir -p "$CONTENTS/MacOS" "$CONTENTS/Resources"

xcrun --sdk macosx swiftc \
  -O -whole-module-optimization \
  -target "$EXPECTED_ARCH-apple-macosx13.0" \
  -parse-as-library \
  -framework SwiftUI -framework AppKit -framework Foundation -framework CryptoKit \
  macos/NovelFormatterApp/Sources/*.swift \
  -o "$CONTENTS/MacOS/NovelFormatterSwift"

echo "==> Embedding Python core"
rsync -a \
  --exclude '.git/' \
  --exclude '.runtime/' \
  --exclude '.model-cache/' \
  --exclude '.ocr-runtimes/' \
  --exclude '.ocr-runtime-state/' \
  --exclude '.manual-model-updates/' \
  --exclude '.windows-runtime-state/' \
  --exclude 'debug/' \
  --exclude '.venv*/' \
  --exclude 'venv/' \
  --exclude '.env' \
  --exclude '.env.*' \
  --exclude '.pytest_cache/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  --exclude 'epub_workspace/' \
  --exclude 'dist/' \
  --exclude 'build/' \
  --exclude 'package/' \
  --exclude 'macos/' \
  --exclude 'legacy/' \
  --exclude '*.app/' \
  --exclude '*.epub' \
  --exclude '*.docx' \
  "$ROOT/" "$CONTENTS/Resources/app/"

echo "==> Installing app icon"
ICON_SOURCE="$ROOT/assets/AppIcon.icns"
if [[ -f "$ICON_SOURCE" ]]; then
  cp "$ICON_SOURCE" "$CONTENTS/Resources/AppIcon.icns"
else
  sips -s format icns "$ROOT/icon.ico" --out "$CONTENTS/Resources/AppIcon.icns" >/dev/null
fi

cat > "$CONTENTS/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleDevelopmentRegion</key>
	<string>en</string>
	<key>CFBundleDisplayName</key>
	<string>Novel Formatter Studio</string>
	<key>CFBundleExecutable</key>
	<string>NovelFormatterSwift</string>
	<key>CFBundleIconFile</key>
	<string>AppIcon</string>
	<key>CFBundleIdentifier</key>
	<string>com.novelformatter.studio.swift</string>
	<key>CFBundleInfoDictionaryVersion</key>
	<string>6.0</string>
	<key>CFBundleName</key>
	<string>Novel Formatter Studio</string>
	<key>CFBundlePackageType</key>
	<string>APPL</string>
	<key>CFBundleShortVersionString</key>
	<string>$VERSION</string>
	<key>CFBundleVersion</key>
	<string>$VERSION</string>
	<key>LSMinimumSystemVersion</key>
	<string>13.0</string>
	<key>NSHighResolutionCapable</key>
	<true/>
	<key>NSHumanReadableCopyright</key>
	<string>Novel Formatter Studio</string>
</dict>
</plist>
PLIST
printf 'APPL????' > "$CONTENTS/PkgInfo"

echo "==> Signing (ad-hoc) and verifying"
xattr -cr "$APP" || true
/usr/bin/codesign --force --deep --sign - --timestamp=none "$APP"
/usr/bin/codesign --verify --deep --strict --verbose=2 "$APP"
/usr/bin/plutil -lint "$CONTENTS/Info.plist"

echo "==> Packaging DMG + app.zip"
mkdir -p "$PACKAGE" "$DMG_STAGE"
/usr/bin/ditto -c -k --sequesterRsrc --keepParent "$APP" "$APP_ZIP"
/usr/bin/ditto "$APP" "$DMG_STAGE/${PRODUCT_NAME}.app"
ln -s /Applications "$DMG_STAGE/Applications"
/usr/bin/hdiutil create \
  -volname "$PRODUCT_NAME (Swift)" \
  -srcfolder "$DMG_STAGE" \
  -ov -format UDZO "$DMG"
/usr/bin/hdiutil verify "$DMG"
rm -rf "$DMG_STAGE"

printf 'Created: %s\nCreated: %s\n' "$DMG" "$APP_ZIP"
