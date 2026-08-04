#!/bin/zsh
set -u
setopt PIPE_FAIL
setopt NULL_GLOB

BUNDLE_CONTENTS=${0:A:h:h}
APP_BUNDLE=${BUNDLE_CONTENTS:h}
# 优先"融合"模式：.app 放在项目文件夹里时，直接运行旁边的实时源码；
# 否则回退到 .app 内嵌的源码副本（独立分发模式）。
if [[ -f "${APP_BUNDLE:h}/gui_pyside6.py" ]]; then
  APP_SOURCE="${APP_BUNDLE:h}"
else
  APP_SOURCE="$BUNDLE_CONTENTS/Resources/app"
fi
SUPPORT_DIR="$HOME/Library/Application Support/Novel Formatter Studio"
RUNTIME_DIR="$SUPPORT_DIR/native-runtime-v1"
DOWNLOAD_DIR="$SUPPORT_DIR/downloads"
LOG_FILE="$SUPPORT_DIR/launcher.log"
REQ_FILE="$APP_SOURCE/requirements.txt"
UPDATE_DIR="${NF_UPDATE_DIR:-$HOME/Desktop/更新包}"
UPDATE_HASH_FILE="$SUPPORT_DIR/.last-update-hash"
PYTHON_RELEASE="20251010"
PYTHON_VERSION="3.12.12"

mkdir -p "$SUPPORT_DIR" "$DOWNLOAD_DIR"
exec >>"$LOG_FILE" 2>&1
print "\n==== $(date '+%Y-%m-%d %H:%M:%S') launch Novel Formatter Studio ===="
print "BOOT_STAGE=launcher_start"
print "app_bundle=$APP_BUNDLE"
print "app_source=$APP_SOURCE"
print "uname=$(/usr/bin/uname -a)"

show_error() {
  local message="$1"
  local safe_message="${message//\"/\\\"}"
  print "ERROR: $message"
  /usr/bin/osascript -e "display dialog \"$safe_message\" buttons {\"好\"} default button 1 with icon stop" \
    >/dev/null 2>&1 || true
}

show_notice() {
  local message="$1"
  local safe_message="${message//\"/\\\"}"
  /usr/bin/osascript -e "display notification \"$safe_message\" with title \"Novel Formatter Studio\"" \
    >/dev/null 2>&1 || true
}

notify_install() {
  show_notice "首次运行正在准备独立 Python 和 Qt，下载完成后会自动打开。"
}

if [[ "$(/usr/bin/uname -s)" != "Darwin" ]]; then
  show_error "此版本仅支持 macOS。"
  exit 2
fi

ARCH="$(/usr/bin/uname -m)"
case "$ARCH" in
  arm64) PY_ARCH="aarch64" ;;
  x86_64) PY_ARCH="x86_64" ;;
  *)
    show_error "不支持的 Mac 架构：$ARCH"
    exit 2
    ;;
esac

# ---------------------------------------------------------------------------
# 自动更新：检查桌面「更新包」文件夹里最新的 Novel Formatter 更新 zip。
# 只匹配本应用的包名，忽略其他应用的 zip；同一个包只应用一次（按 SHA256 记录）。
# ---------------------------------------------------------------------------
find_package_root() {
  local search_dir="$1" root=""
  if [[ -f "$search_dir/gui_pyside6.py" || -f "$search_dir/run.py" ]]; then
    print -r -- "$search_dir"
    return 0
  fi
  root="$(/usr/bin/find "$search_dir" -mindepth 1 -maxdepth 3 -type f \( -name gui_pyside6.py -o -name run.py \) -print -quit 2>/dev/null)"
  if [[ -n "$root" ]]; then
    /usr/bin/dirname "$root"
    return 0
  fi
  return 1
}

apply_update_package() {
  local zip_path="$1" zip_hash="$2"
  local tmp_dir root_dir
  print "BOOT_STAGE=apply_update package=$zip_path"
  tmp_dir="$(/usr/bin/mktemp -d /private/tmp/novel-formatter-update.XXXXXX)" || return 1
  {
    /usr/bin/unzip -q -o "$zip_path" -d "$tmp_dir" || {
      show_error "更新包解压失败，已跳过本次更新：\n$zip_path"
      return 1
    }
    root_dir="$(find_package_root "$tmp_dir" || true)"
    if [[ -z "$root_dir" ]]; then
      show_error "更新包里没有找到 Novel Formatter 源码（缺少 gui_pyside6.py），已跳过：\n$zip_path"
      return 1
    fi
    print "update_root=$root_dir"
    /usr/bin/rsync -a --delete \
      --exclude '.git/' \
      --exclude '.agents/' \
      --exclude '.codex/' \
      --exclude '.venv*/' \
      --exclude 'venv/' \
      --exclude 'env/' \
      --exclude '.env' \
      --exclude '.env.*' \
      --exclude '.pytest_cache/' \
      --exclude '__pycache__/' \
      --exclude '*.pyc' \
      --exclude '*.db' \
      --exclude '*.sqlite' \
      --exclude '*.sqlite3' \
      --exclude '*.epub' \
      --exclude '*.docx' \
      --exclude 'epub_workspace/' \
      --exclude 'output/' \
      --exclude 'outputs/' \
      --exclude 'dist/' \
      --exclude 'build/' \
      --exclude 'packaging/' \
      --exclude 'legacy/' \
      --exclude '.DS_Store' \
      --exclude '*.command' \
      --exclude '*.app/' \
      "$root_dir/" "$APP_SOURCE/" || {
      show_error "更新文件复制失败，请查看日志：$LOG_FILE"
      return 1
    }
    print -r -- "$zip_hash" >"$UPDATE_HASH_FILE"
    show_notice "已自动更新：${zip_path:t}"
    print "BOOT_STAGE=update_applied"
    return 0
  } always {
    /bin/rm -rf "$tmp_dir"
  }
}

check_for_update() {
  [[ -d "$UPDATE_DIR" ]] || return 0
  local candidates latest zip_hash last_hash
  candidates=(
    "$UPDATE_DIR"/Novel-formatter*.zip(N.om)
    "$UPDATE_DIR"/Novel-Formatter*.zip(N.om)
    "$UPDATE_DIR"/NovelFormatter*.zip(N.om)
    "$UPDATE_DIR"/Novel\ Formatter*.zip(N.om)
    "$UPDATE_DIR"/小说格式*.zip(N.om)
    "$UPDATE_DIR"/小说排版*.zip(N.om)
  )
  latest="${candidates[1]:-}"
  [[ -n "$latest" && -f "$latest" ]] || return 0
  zip_hash="$(/usr/bin/shasum -a 256 "$latest" | /usr/bin/awk '{print $1}')"
  last_hash=""
  [[ -f "$UPDATE_HASH_FILE" ]] && last_hash="$(<"$UPDATE_HASH_FILE")"
  if [[ "$zip_hash" == "$last_hash" ]]; then
    print "BOOT_STAGE=update_skip_same package=${latest:t}"
    return 0
  fi
  apply_update_package "$latest" "$zip_hash" || true
}

print "BOOT_STAGE=check_update dir=$UPDATE_DIR"
check_for_update

# ---------------------------------------------------------------------------
# 独立 Python 运行环境（下载一次，之后复用）
# ---------------------------------------------------------------------------
ARCHIVE="cpython-${PYTHON_VERSION}+${PYTHON_RELEASE}-${PY_ARCH}-apple-darwin-install_only_stripped.tar.gz"
PYTHON_GITHUB="https://github.com/astral-sh/python-build-standalone/releases/download/${PYTHON_RELEASE}/${ARCHIVE}"
PYTHON_MIRROR="https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone/${PYTHON_RELEASE}/${ARCHIVE}"
ARCHIVE_PATH="$DOWNLOAD_DIR/$ARCHIVE"
SUMS_PATH="$DOWNLOAD_DIR/SHA256SUMS-${PYTHON_RELEASE}"
SUMS_GITHUB="https://github.com/astral-sh/python-build-standalone/releases/download/${PYTHON_RELEASE}/SHA256SUMS"
SUMS_MIRROR="https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone/${PYTHON_RELEASE}/SHA256SUMS"
PY="$RUNTIME_DIR/python/bin/python3"

clean_injected_environment() {
  unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONUSERBASE
  unset QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH QML2_IMPORT_PATH QML_IMPORT_PATH
  unset DYLD_LIBRARY_PATH DYLD_FRAMEWORK_PATH DYLD_FALLBACK_LIBRARY_PATH DYLD_FALLBACK_FRAMEWORK_PATH
  export PYTHONNOUSERSITE=1
  export PYTHONUNBUFFERED=1
  export QT_API=pyside6
  export QT_ENABLE_HIGHDPI_SCALING=1
}

fetch_file() {
  local output="$1"
  shift
  local url
  for url in "$@"; do
    print "download=$url"
    if /usr/bin/curl --fail --location --retry 3 --connect-timeout 20 --speed-time 30 --speed-limit 1024 \
      --output "$output.part" "$url"; then
      /bin/mv -f "$output.part" "$output"
      return 0
    fi
    /bin/rm -f "$output.part"
  done
  return 1
}

install_python_runtime() {
  print "BOOT_STAGE=install_standalone_python"
  notify_install
  /bin/rm -rf "$RUNTIME_DIR.new"
  /bin/mkdir -p "$RUNTIME_DIR.new"
  if [[ ! -s "$ARCHIVE_PATH" ]]; then
    fetch_file "$ARCHIVE_PATH" "$PYTHON_GITHUB" "$PYTHON_MIRROR" || {
      show_error "无法下载独立 Python。请检查网络后重试。日志：$LOG_FILE"
      return 1
    }
  fi
  if [[ ! -s "$SUMS_PATH" ]]; then
    fetch_file "$SUMS_PATH" "$SUMS_GITHUB" "$SUMS_MIRROR" || {
      show_error "无法下载 Python 校验文件。请检查网络后重试。"
      return 1
    }
  fi
  local expected actual
  expected=$(/usr/bin/awk -v filename="$ARCHIVE" '$2 == filename {print $1; exit}' "$SUMS_PATH")
  actual=$(/usr/bin/shasum -a 256 "$ARCHIVE_PATH" | /usr/bin/awk '{print $1}')
  if [[ -z "$expected" || "$expected" != "$actual" ]]; then
    print "checksum_expected=$expected"
    print "checksum_actual=$actual"
    /bin/rm -f "$ARCHIVE_PATH" "$SUMS_PATH"
    show_error "独立 Python 校验失败，文件已删除。请重新运行。"
    return 1
  fi
  /usr/bin/tar -xzf "$ARCHIVE_PATH" -C "$RUNTIME_DIR.new" || {
    /bin/rm -f "$ARCHIVE_PATH"
    show_error "独立 Python 压缩包损坏，已删除，请重新运行。"
    return 1
  }
  if [[ ! -x "$RUNTIME_DIR.new/python/bin/python3" ]]; then
    show_error "独立 Python 解压后缺少可执行文件。"
    return 1
  fi
  /bin/rm -rf "$RUNTIME_DIR"
  /bin/mv "$RUNTIME_DIR.new" "$RUNTIME_DIR"
}

clean_injected_environment
if [[ ! -x "$PY" ]]; then
  install_python_runtime || exit 1
fi

print "BOOT_STAGE=verify_python"
PY_ARCH_ACTUAL=$("$PY" -c 'import platform; print(platform.machine())' 2>/dev/null || true)
PY_VERSION_ACTUAL=$("$PY" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null || true)
print "python=$PY"
print "python_version=$PY_VERSION_ACTUAL"
print "python_arch=$PY_ARCH_ACTUAL"
if [[ "$ARCH" == "arm64" && "$PY_ARCH_ACTUAL" != "arm64" ]]; then
  /bin/rm -rf "$RUNTIME_DIR"
  show_error "独立 Python 架构不匹配，已清除运行环境，请重新运行。"
  exit 1
fi

REQ_HASH=$(/usr/bin/shasum -a 256 "$REQ_FILE" | /usr/bin/awk '{print $1}')
STAMP="$RUNTIME_DIR/.requirements-${REQ_HASH}"
if [[ ! -f "$STAMP" ]]; then
  print "BOOT_STAGE=install_pinned_dependencies"
  notify_install
  "$PY" -m pip install --disable-pip-version-check --upgrade "pip<26" || exit 1
  if ! "$PY" -m pip install --disable-pip-version-check --prefer-binary -r "$REQ_FILE"; then
    show_error "macOS 依赖安装失败。请打开日志查看具体包名：$LOG_FILE"
    exit 1
  fi
  /bin/rm -f "$RUNTIME_DIR"/.requirements-*
  /usr/bin/touch "$STAMP"
fi

print "BOOT_STAGE=qt_import_probe"
"$PY" - <<'PY'
import platform
import PySide6
from PySide6 import QtCore, QtGui, QtWidgets
assert platform.system() == "Darwin"
assert PySide6.__version__.startswith("6.")
print("qt_version=", PySide6.__version__)
print("qt_library_paths=", QtCore.QCoreApplication.libraryPaths())
PY
PROBE_STATUS=$?
if [[ $PROBE_STATUS -ne 0 ]]; then
  /bin/mv "$RUNTIME_DIR" "$RUNTIME_DIR.failed-$(date +%s)" 2>/dev/null || true
  show_error "Qt 导入检查失败，已隔离损坏环境。请重新运行一次。"
  exit $PROBE_STATUS
fi

if [[ "${NF_PREPARE_ONLY:-0}" == "1" ]]; then
  print "BOOT_STAGE=prepare_only_done"
  exit 0
fi

print "BOOT_STAGE=run_main"
cd "$APP_SOURCE" || exit 1
if [[ "$ARCH" == "arm64" ]]; then
  /usr/bin/arch -arm64 "$PY" -X faulthandler gui_pyside6.py
else
  /usr/bin/arch -x86_64 "$PY" -X faulthandler gui_pyside6.py
fi
STATUS=$?
print "BOOT_STAGE=main_exit status=$STATUS"
if [[ $STATUS -eq 139 ]]; then
  /bin/mv "$RUNTIME_DIR" "$RUNTIME_DIR.sigsegv-$(date +%s)" 2>/dev/null || true
  show_error "检测到原生崩溃（139）。运行环境已自动隔离；再次运行将重新下载干净环境。日志：$LOG_FILE"
elif [[ $STATUS -ne 0 ]]; then
  show_error "程序异常退出（代码 $STATUS）。日志：$LOG_FILE"
fi
exit $STATUS
