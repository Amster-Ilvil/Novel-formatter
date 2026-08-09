#!/bin/zsh
# 构建 "Novel Formatter Studio.app" macOS 应用包。
# 用法：双击运行，或 ./packaging/build_mac_app.command [版本号]
set -eu

PROJECT_DIR=${0:A:h:h}
PACKAGING_DIR="$PROJECT_DIR/packaging"
VERSION="${1:-1.0}"
DATE_TAG="$(date +%Y%m%d)"
DIST_ROOT="$PROJECT_DIR/dist"
PKG_DIR="$DIST_ROOT/NovelFormatterStudio"
APP_DIR="$PKG_DIR/Novel Formatter Studio.app"
CONTENTS="$APP_DIR/Contents"
APP_SOURCE="$CONTENTS/Resources/app"
ZIP_NAME="NovelFormatterStudio_macOS_v${VERSION}_${DATE_TAG}.zip"

echo "==> Building Novel Formatter Studio.app v$VERSION"
rm -rf "$PKG_DIR"
mkdir -p "$CONTENTS/MacOS" "$CONTENTS/Resources" "$APP_SOURCE"

echo "==> Copying app source"
rsync -a \
  --exclude '.git/' \
  --exclude '.github/' \
  --exclude '.gitignore' \
  --exclude '.runtime/' \
  --exclude '.model-cache/' \
  --exclude '.ocr-runtimes/' \
  --exclude '.ocr-runtime-state/' \
  --exclude '.manual-model-updates/' \
  --exclude '.windows-runtime-state/' \
  --exclude 'debug/' \
  --exclude 'tests/' \
  --exclude '.venv*/' \
  --exclude 'venv/' \
  --exclude '.env' \
  --exclude '.env.*' \
  --exclude '.pytest_cache/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '*.log' \
  --exclude '*.db' \
  --exclude '*.sqlite*' \
  --exclude '.DS_Store' \
  --exclude 'epub_workspace/' \
  --exclude 'output/' \
  --exclude 'outputs/' \
  --exclude 'dist/' \
  --exclude 'build/' \
  --exclude 'packaging/' \
  --exclude 'legacy/' \
  --exclude '*.command' \
  --exclude '*.app/' \
  --exclude '*.epub' \
  --exclude '*.docx' \
  "$PROJECT_DIR/" "$APP_SOURCE/"

echo "==> Converting icon"
sips -s format icns "$PROJECT_DIR/icon.ico" --out "$CONTENTS/Resources/AppIcon.icns" >/dev/null

echo "==> Writing bundle metadata"
cat >"$CONTENTS/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleDevelopmentRegion</key>
	<string>zh_CN</string>
	<key>CFBundleDisplayName</key>
	<string>Novel Formatter Studio</string>
	<key>CFBundleExecutable</key>
	<string>launcher</string>
	<key>CFBundleIconFile</key>
	<string>AppIcon</string>
	<key>CFBundleIdentifier</key>
	<string>com.novelformatter.studio</string>
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
	<string>12.0</string>
	<key>NSHighResolutionCapable</key>
	<true/>
	<key>NSHumanReadableCopyright</key>
	<string>Novel Formatter Studio</string>
</dict>
</plist>
PLIST
printf 'APPL????' >"$CONTENTS/PkgInfo"

cp "$PACKAGING_DIR/launcher.zsh" "$CONTENTS/MacOS/launcher"
chmod +x "$CONTENTS/MacOS/launcher"

cat >"$PKG_DIR/开始使用.txt" <<'TXT'
Novel Formatter Studio（macOS）
================================

第一次使用
----------
1. 把“Novel Formatter Studio.app”拖到“应用程序”或桌面。
2. 双击打开。程序只准备独立 Python 和主程序界面依赖，不下载 OCR 模型或 OCR 专用依赖。
3. 第一次开始某个 OCR 时，程序先检查对应运行环境；缺失时弹窗说明，只有确认后才下载。
4. 下载优先尝试国内可用镜像，失败时回退官方源。
5. 已安装的 OCR 模型不会在启动时检查或替换；后续更新只能在设置页手动执行。
6. macOS 提示无法验证开发者时，右键 App → 打开。

隐私
----
- 发布包只从 GitHub 干净源码构建，不包含作者电脑路径、用户名、邮箱、令牌、运行日志、调试输入、OCR 结果或用户文档。
- OCR 模型权重、虚拟环境和模型缓存不进入发布包。
- 本机运行状态、缓存和日志只保存在本地应用支持目录，不属于 Git 仓库。
- API 密钥只在运行时填写，不要写入源码或提交 .env 文件。

排障
----
- 可运行“重置Mac运行环境.command”后重新打开。
- 重置只删除应用运行环境和下载缓存，不删除用户选择的小说文件。
TXT

cat >"$PKG_DIR/重置Mac运行环境.command" <<'RESET'
#!/bin/zsh
SUPPORT_DIR="$HOME/Library/Application Support/Novel Formatter Studio"
echo "即将删除 Novel Formatter Studio 的独立运行环境："
echo "  $SUPPORT_DIR"
echo "（不会删除任何小说文件或设置以外的数据）"
echo
read "?按回车确认删除并重置，或按 Ctrl+C 取消: "
rm -rf "$SUPPORT_DIR/native-runtime-v1" "$SUPPORT_DIR/downloads" "$SUPPORT_DIR/app-source" "$SUPPORT_DIR/logs"
echo "已重置。下次打开 App 会重新下载运行环境。"
read "?按回车关闭窗口: "
RESET
chmod +x "$PKG_DIR/重置Mac运行环境.command"

echo "==> Zipping"
cd "$DIST_ROOT"
rm -f "$ZIP_NAME"
/usr/bin/ditto -c -k --keepParent "NovelFormatterStudio" "$ZIP_NAME"

if [[ "${NF_SKIP_PROJECT_APP:-0}" != "1" ]]; then
  echo "==> Installing persistent app shell into project folder (fusion mode)"
  PROJECT_APP="$PROJECT_DIR/Novel Formatter Studio.app"
  if [ -d "$PROJECT_APP" ]; then
    echo "    Keeping existing app shell: $PROJECT_APP"
    mkdir -p "$PROJECT_APP/Contents/MacOS" "$PROJECT_APP/Contents/Resources"
    cp "$CONTENTS/Info.plist" "$PROJECT_APP/Contents/Info.plist"
    cp "$CONTENTS/PkgInfo" "$PROJECT_APP/Contents/PkgInfo"
    cp "$CONTENTS/MacOS/launcher" "$PROJECT_APP/Contents/MacOS/launcher"
    cp "$CONTENTS/Resources/AppIcon.icns" "$PROJECT_APP/Contents/Resources/AppIcon.icns"
    chmod +x "$PROJECT_APP/Contents/MacOS/launcher"
  else
    cp -R "$APP_DIR" "$PROJECT_APP"
  fi
else
  PROJECT_APP="(skipped for clean release build)"
fi

echo
echo "构建完成："
echo "  项目内 App: $PROJECT_APP"
echo "  独立分发 App: $APP_DIR"
echo "  Zip:  $DIST_ROOT/$ZIP_NAME"
