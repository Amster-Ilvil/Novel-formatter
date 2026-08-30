#!/bin/zsh
# Privacy-safe macOS launcher. It prepares only the application runtime and GUI
# dependencies. OCR resources are downloaded only after an OCR-start confirmation.
set -u
setopt PIPE_FAIL
setopt NULL_GLOB

BUNDLE_CONTENTS=${0:A:h:h}
APP_BUNDLE=${BUNDLE_CONTENTS:h}
SUPPORT_DIR="$HOME/Library/Application Support/Novel Formatter Studio"
RUNTIME_DIR="$SUPPORT_DIR/native-runtime-v1"
DOWNLOAD_DIR="$SUPPORT_DIR/downloads"
LOG_DIR="$SUPPORT_DIR/logs"
LOG_FILE="$LOG_DIR/launcher.log"
PYTHON_RELEASE="20251010"
PYTHON_VERSION="3.12.12"

mkdir -p "$SUPPORT_DIR" "$DOWNLOAD_DIR" "$LOG_DIR"
exec >>"$LOG_FILE" 2>&1
print "\n==== $(date '+%Y-%m-%d %H:%M:%S') Novel Formatter launch ===="
print "BOOT_STAGE=launcher_start"

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

if [[ "$(/usr/bin/uname -s)" != "Darwin" ]]; then
  show_error "此启动器仅支持 macOS。"
  exit 2
fi

ARCH="$(/usr/bin/uname -m)"
case "$ARCH" in
  arm64) PY_ARCH="aarch64" ;;
  x86_64) PY_ARCH="x86_64" ;;
  *) show_error "当前 Mac 架构不受支持。"; exit 2 ;;
esac

# Fusion mode runs the repository next to the app shell. A standalone app copies
# its embedded source into Application Support so model/runtime directories stay
# writable and the signed/read-only bundle is never modified at runtime.
if [[ -f "${APP_BUNDLE:h}/gui_pyside6.py" ]]; then
  APP_SOURCE="${APP_BUNDLE:h}"
  print "BOOT_STAGE=source_fusion"
else
  EMBEDDED_SOURCE="$BUNDLE_CONTENTS/Resources/app"
  APP_SOURCE="$SUPPORT_DIR/app-source"
  if [[ ! -f "$EMBEDDED_SOURCE/gui_pyside6.py" ]]; then
    show_error "应用包内缺少程序文件，请重新下载发布包。"
    exit 2
  fi
  mkdir -p "$APP_SOURCE"

  # A manual in-app Git update writes this ignored state file. Preserve that
  # verified source across restarts instead of copying the older embedded bundle
  # over it. If a newly installed app bundle carries a *newer version*, the
  # embedded release becomes authoritative again and refreshes app-source.
  should_sync_embedded=1
  SOURCE_UPDATE_STATE="$APP_SOURCE/.runtime/source_update_state.json"
  EMBEDDED_VERSION_FILE="$EMBEDDED_SOURCE/VERSION"
  CURRENT_VERSION_FILE="$APP_SOURCE/VERSION"
  version_ge() {
    /usr/bin/awk -v a="$1" -v b="$2" 'BEGIN {
      gsub(/^v/, "", a); gsub(/^v/, "", b);
      split(a, A, "."); split(b, B, ".");
      for (i=1; i<=4; i++) {
        av=(A[i] == "" ? 0 : A[i]+0); bv=(B[i] == "" ? 0 : B[i]+0);
        if (av > bv) exit 0; if (av < bv) exit 1;
      }
      exit 0;
    }'
  }
  if [[ -f "$SOURCE_UPDATE_STATE" && -f "$CURRENT_VERSION_FILE" && -f "$EMBEDDED_VERSION_FILE" ]]; then
    current_version="$(/bin/cat "$CURRENT_VERSION_FILE" | /usr/bin/tr -d '[:space:]')"
    embedded_version="$(/bin/cat "$EMBEDDED_VERSION_FILE" | /usr/bin/tr -d '[:space:]')"
    if [[ -n "$current_version" && -n "$embedded_version" ]] && version_ge "$current_version" "$embedded_version"; then
      should_sync_embedded=0
      print "BOOT_STAGE=source_standalone_git_preserved"
    fi
  fi

  if [[ "$should_sync_embedded" == "1" ]]; then
    /usr/bin/rsync -a --delete \
      --exclude '.git/' \
      --exclude '.runtime/' \
      --exclude '.venv*/' \
      --exclude '.model-cache/' \
      --exclude '.ocr-runtimes/' \
      --exclude '.manual-model-updates/' \
      --exclude '__pycache__/' \
      --exclude '.pytest_cache/' \
      --exclude '*.pyc' \
      --exclude '*.log' \
      "$EMBEDDED_SOURCE/" "$APP_SOURCE/" || {
        show_error "无法准备可写的应用运行目录。"
        exit 1
      }
    print "BOOT_STAGE=source_standalone"
  fi
fi

REQ_FILE="$APP_SOURCE/requirements.txt"
BOOTSTRAP_FILE="$APP_SOURCE/bootstrap.py"
if [[ ! -f "$REQ_FILE" || ! -f "$BOOTSTRAP_FILE" ]]; then
  show_error "发布包缺少部署文件，请重新下载。"
  exit 2
fi

clean_injected_environment() {
  unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONUSERBASE
  unset QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH QML2_IMPORT_PATH QML_IMPORT_PATH
  unset DYLD_LIBRARY_PATH DYLD_FRAMEWORK_PATH DYLD_FALLBACK_LIBRARY_PATH DYLD_FALLBACK_FRAMEWORK_PATH
  export PYTHONNOUSERSITE=1
  export PYTHONUNBUFFERED=1
  export QT_API=pyside6
  export QT_ENABLE_HIGHDPI_SCALING=1
  export PIP_DISABLE_PIP_VERSION_CHECK=1
  export PIP_NO_INPUT=1
}

fetch_file() {
  local output="$1"
  shift
  local url
  for url in "$@"; do
    print "BOOT_STAGE=download_runtime"
    if /usr/bin/curl --fail --location --retry 3 --connect-timeout 20 --speed-time 30 --speed-limit 1024 \
      --output "$output.part" "$url"; then
      /bin/mv -f "$output.part" "$output"
      return 0
    fi
    /bin/rm -f "$output.part"
  done
  return 1
}

ARCHIVE="cpython-${PYTHON_VERSION}+${PYTHON_RELEASE}-${PY_ARCH}-apple-darwin-install_only_stripped.tar.gz"
ARCHIVE_PATH="$DOWNLOAD_DIR/$ARCHIVE"
SUMS_PATH="$DOWNLOAD_DIR/SHA256SUMS-${PYTHON_RELEASE}"
MIRROR_ROOT="https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone/${PYTHON_RELEASE}"
OFFICIAL_ROOT="https://github.com/astral-sh/python-build-standalone/releases/download/${PYTHON_RELEASE}"
PY="$RUNTIME_DIR/python/bin/python3"

install_python_runtime() {
  print "BOOT_STAGE=install_standalone_python"
  show_notice "首次运行只准备独立 Python 和主程序依赖；OCR 模型将在开始 OCR 并确认后下载。"
  /bin/rm -rf "$RUNTIME_DIR.new"
  /bin/mkdir -p "$RUNTIME_DIR.new"
  if [[ ! -s "$ARCHIVE_PATH" ]]; then
    fetch_file "$ARCHIVE_PATH" "$MIRROR_ROOT/$ARCHIVE" "$OFFICIAL_ROOT/$ARCHIVE" || {
      show_error "无法下载独立 Python，请检查网络后重试。"
      return 1
    }
  fi
  if [[ ! -s "$SUMS_PATH" ]]; then
    fetch_file "$SUMS_PATH" "$MIRROR_ROOT/SHA256SUMS" "$OFFICIAL_ROOT/SHA256SUMS" || {
      show_error "无法下载 Python 校验文件，请检查网络后重试。"
      return 1
    }
  fi
  local expected actual
  expected=$(/usr/bin/awk -v filename="$ARCHIVE" '$2 == filename || $2 == "*" filename {print $1; exit}' "$SUMS_PATH")
  actual=$(/usr/bin/shasum -a 256 "$ARCHIVE_PATH" | /usr/bin/awk '{print $1}')
  if [[ -z "$expected" || "$expected" != "$actual" ]]; then
    /bin/rm -f "$ARCHIVE_PATH" "$SUMS_PATH"
    show_error "独立 Python SHA-256 校验失败，文件已删除。"
    return 1
  fi
  /usr/bin/tar -xzf "$ARCHIVE_PATH" -C "$RUNTIME_DIR.new" || {
    /bin/rm -f "$ARCHIVE_PATH"
    show_error "独立 Python 压缩包损坏，已删除。"
    return 1
  }
  if [[ ! -x "$RUNTIME_DIR.new/python/bin/python3" ]]; then
    show_error "独立 Python 解压后不可用。"
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
if [[ "$ARCH" == "arm64" && "$PY_ARCH_ACTUAL" != "arm64" ]]; then
  /bin/rm -rf "$RUNTIME_DIR"
  show_error "独立 Python 架构不匹配，已清除，请重新运行。"
  exit 1
fi

ARGS=("$BOOTSTRAP_FILE" --install-main-deps)
if [[ "${NF_PREPARE_ONLY:-0}" != "1" ]]; then
  ARGS+=(--launch)
fi

print "BOOT_STAGE=prepare_application_dependencies"
cd "$APP_SOURCE" || exit 1
if [[ "$ARCH" == "arm64" ]]; then
  /usr/bin/arch -arm64 "$PY" -X faulthandler "${ARGS[@]}"
else
  /usr/bin/arch -x86_64 "$PY" -X faulthandler "${ARGS[@]}"
fi
STATUS=$?
print "BOOT_STAGE=exit status=$STATUS"
if [[ $STATUS -ne 0 ]]; then
  show_error "程序依赖准备或启动失败（代码 $STATUS）。可查看应用支持目录中的本地日志。"
fi
exit $STATUS
