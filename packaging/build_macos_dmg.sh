#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VERSION="${VERSION:-$(tr -d '[:space:]' < VERSION)}"
LABEL="${LABEL:-$(uname -m)}"
PRODUCT_NAME="Novel Formatter Studio"
BASENAME="NovelFormatter_${VERSION}_macOS_${LABEL}"
DIST="$ROOT/dist"
PACKAGE="$ROOT/package"
APP="$DIST/NovelFormatterStudio/${PRODUCT_NAME}.app"
APP_ZIP="$PACKAGE/${BASENAME}.app.zip"
DMG="$PACKAGE/${BASENAME}.dmg"
DMG_STAGE="$ROOT/.dmg-stage"

python3 scripts/privacy_audit.py
NF_SKIP_PROJECT_APP=1 zsh packaging/build_mac_app.command "$VERSION"

test -d "$APP"
rm -rf "$PACKAGE" "$DMG_STAGE"
mkdir -p "$PACKAGE" "$DMG_STAGE"

# The distributed app contains only the clean tracked source copied by the
# release builder. No model weights, virtual environments, logs or user files.
xattr -cr "$APP" || true
/usr/bin/codesign --force --deep --sign - --timestamp=none "$APP"
/usr/bin/codesign --verify --deep --strict --verbose=2 "$APP"
/usr/bin/plutil -lint "$APP/Contents/Info.plist"

/usr/bin/ditto -c -k --sequesterRsrc --keepParent "$APP" "$APP_ZIP"
/usr/bin/ditto "$APP" "$DMG_STAGE/${PRODUCT_NAME}.app"
ln -s /Applications "$DMG_STAGE/Applications"
/usr/bin/hdiutil create \
  -volname "$PRODUCT_NAME" \
  -srcfolder "$DMG_STAGE" \
  -ov -format UDZO "$DMG"
/usr/bin/hdiutil verify "$DMG"

rm -rf "$DMG_STAGE"
printf 'Created: %s\nCreated: %s\n' "$DMG" "$APP_ZIP"
