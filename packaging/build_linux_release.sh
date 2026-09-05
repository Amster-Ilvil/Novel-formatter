#!/usr/bin/env bash
# Build Linux x86_64 / ARM64 AppImage, DEB and portable tar.gz release assets.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${VERSION:-$(tr -d '[:space:]' < "$ROOT/VERSION")}"
DIST="$ROOT/dist"
WORK="$DIST/linux-work"

case "$(uname -m)" in
  x86_64|amd64)
    MACHINE="x86_64"
    LABEL="x86_64"
    DEB_ARCH="amd64"
    APPIMAGE_ARCH="x86_64"
    ;;
  aarch64|arm64)
    MACHINE="aarch64"
    LABEL="ARM64"
    DEB_ARCH="arm64"
    APPIMAGE_ARCH="aarch64"
    ;;
  *)
    echo "Unsupported build architecture: $(uname -m)" >&2
    exit 2
    ;;
esac

STAGE="$WORK/NovelFormatter-Linux-$LABEL"
APP_SOURCE="$STAGE/share/novel-formatter/app"
TAR_OUT="$DIST/NovelFormatter_${VERSION}_Linux_${LABEL}.tar.gz"
DEB_OUT="$DIST/NovelFormatter_${VERSION}_Linux_${LABEL}.deb"
APPIMAGE_OUT="$DIST/NovelFormatter_${VERSION}_Linux_${LABEL}.AppImage"

rm -rf "$WORK"
rm -f "$TAR_OUT" "$DEB_OUT" "$APPIMAGE_OUT"
mkdir -p "$APP_SOURCE" "$STAGE/bin" "$STAGE/libexec" \
  "$STAGE/share/applications" "$STAGE/share/icons/hicolor/256x256/apps" "$DIST"

# Privacy boundary: stage only files tracked by the current Git commit.
TMP_TAR="$WORK/source.tar"
mkdir -p "$WORK"
(
  cd "$ROOT"
  git archive --format=tar HEAD -o "$TMP_TAR"
)
tar -xf "$TMP_TAR" -C "$APP_SOURCE"
rm -f "$TMP_TAR"
rm -rf "$APP_SOURCE/.github" "$APP_SOURCE/tests" "$APP_SOURCE/packaging"
rm -f "$APP_SOURCE/.gitignore"

# A tracked-file manifest lets the launcher update release source without
# deleting writable OCR/model/runtime directories created by the user.
(
  cd "$APP_SOURCE"
  find . -type f -printf '%P\n' \
    ! -name '.release-manifest.txt' \
    | LC_ALL=C sort > .release-manifest.txt
)

install -m 0755 "$ROOT/packaging/launcher_linux.sh" "$STAGE/bin/novel-formatter"
UV_BIN="$(command -v uv || true)"
if [[ -z "$UV_BIN" || ! -x "$UV_BIN" ]]; then
  echo "uv is required to build the Linux release. Install it before running this script." >&2
  exit 2
fi
install -m 0755 "$UV_BIN" "$STAGE/libexec/uv"

# Convert the repository icon to a standard Linux PNG.
python3 - "$ROOT/icon.ico" "$STAGE/share/icons/hicolor/256x256/apps/novelformatter.png" <<'PY'
from pathlib import Path
import sys
from PIL import Image
src, dst = map(Path, sys.argv[1:3])
with Image.open(src) as im:
    if getattr(im, "n_frames", 1) > 1:
        best = 0
        best_area = 0
        for i in range(im.n_frames):
            im.seek(i)
            area = im.width * im.height
            if area > best_area:
                best, best_area = i, area
        im.seek(best)
    image = im.convert("RGBA")
    image.thumbnail((256, 256), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    canvas.alpha_composite(image, ((256-image.width)//2, (256-image.height)//2))
    dst.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst, "PNG")
PY

cat > "$STAGE/share/applications/novelformatter.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=Novel Formatter Studio
Comment=Japanese vertical novel OCR, multi-model review and EPUB production
Exec=novel-formatter
Icon=novelformatter
Terminal=false
Categories=Office;Graphics;
StartupNotify=true
EOF

cat > "$STAGE/RELEASE-README.txt" <<EOF
Novel Formatter Studio $VERSION - Linux $LABEL
================================================

Recommended: run the AppImage. The portable tar.gz is also user-local and does
not require administrator privileges. The DEB package installs a system launcher
and therefore normally uses your distribution package manager.

First launch prepares a managed Python 3.12 runtime and the main GUI dependencies
inside your user data directory. OCR-specific runtimes and model weights are NOT
bundled and remain on-demand after you start the corresponding OCR function and
confirm installation.

Apple Live Text / Apple Vision and MLX are macOS-specific and are not available
on Linux. Cross-platform OCR backends remain available when their runtime/model
requirements support this architecture.

Privacy: this package is generated from a clean Git checkout and contains no
.env file, API key, developer cache, log, model cache, OCR output or user books.
EOF

# Portable tar.gz.
tar -C "$WORK" -czf "$TAR_OUT" "$(basename "$STAGE")"

# Debian/Ubuntu package. Application data lives below /opt; a stable command and
# desktop integration are installed in standard locations.
DEB_ROOT="$WORK/deb-root"
mkdir -p "$DEB_ROOT/DEBIAN" "$DEB_ROOT/opt/NovelFormatter" \
  "$DEB_ROOT/usr/bin" "$DEB_ROOT/usr/share/applications" \
  "$DEB_ROOT/usr/share/icons/hicolor/256x256/apps"
cp -a "$STAGE/." "$DEB_ROOT/opt/NovelFormatter/"
ln -s /opt/NovelFormatter/bin/novel-formatter "$DEB_ROOT/usr/bin/novel-formatter"
cp "$STAGE/share/applications/novelformatter.desktop" "$DEB_ROOT/usr/share/applications/novelformatter.desktop"
cp "$STAGE/share/icons/hicolor/256x256/apps/novelformatter.png" \
  "$DEB_ROOT/usr/share/icons/hicolor/256x256/apps/novelformatter.png"
cat > "$DEB_ROOT/DEBIAN/control" <<EOF
Package: novel-formatter-studio
Version: $VERSION
Section: text
Priority: optional
Architecture: $DEB_ARCH
Maintainer: Novel Formatter contributors
Depends: bash, ca-certificates, libgl1, libegl1, libxkbcommon0, libdbus-1-3, fontconfig
Recommends: poppler-utils
Description: Novel Formatter Studio
 Japanese vertical novel OCR, multi-model review and EPUB production tool.
 Main GUI dependencies are prepared in the current user's data directory;
 OCR-specific models and runtimes remain on-demand.
EOF
dpkg-deb --root-owner-group --build "$DEB_ROOT" "$DEB_OUT" >/dev/null

# AppImage: keep the same /opt-style payload so launcher behavior is identical
# across AppImage, DEB and portable tar.gz.
APPDIR="$WORK/NovelFormatter.AppDir"
mkdir -p "$APPDIR/usr/opt/NovelFormatter"
cp -a "$STAGE/." "$APPDIR/usr/opt/NovelFormatter/"
cat > "$APPDIR/AppRun" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
APPDIR="${APPDIR:-$(cd "$(dirname "$0")" && pwd)}"
exec "$APPDIR/usr/opt/NovelFormatter/bin/novel-formatter" "$@"
EOF
chmod 0755 "$APPDIR/AppRun"
cp "$STAGE/share/applications/novelformatter.desktop" "$APPDIR/novelformatter.desktop"
cp "$STAGE/share/icons/hicolor/256x256/apps/novelformatter.png" "$APPDIR/novelformatter.png"
ln -s novelformatter.png "$APPDIR/.DirIcon"

APPIMAGETOOL="$WORK/appimagetool-${APPIMAGE_ARCH}.AppImage"
curl --fail --location --retry 3 --connect-timeout 20 \
  "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-${APPIMAGE_ARCH}.AppImage" \
  -o "$APPIMAGETOOL"
chmod 0755 "$APPIMAGETOOL"
ARCH="$APPIMAGE_ARCH" APPIMAGE_EXTRACT_AND_RUN=1 "$APPIMAGETOOL" "$APPDIR" "$APPIMAGE_OUT" >/dev/null
chmod 0755 "$APPIMAGE_OUT"

for asset in "$TAR_OUT" "$DEB_OUT" "$APPIMAGE_OUT"; do
  test -s "$asset"
  echo "Created $asset ($(stat -c '%s' "$asset") bytes)"
done
