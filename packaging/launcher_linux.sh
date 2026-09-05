#!/usr/bin/env bash
# Linux launcher for AppImage / DEB / portable tar.gz releases.
# It prepares only Python + main GUI dependencies in the user's home directory.
# OCR-specific runtimes and models remain on-demand and are never bundled here.
set -euo pipefail

SELF="$(readlink -f "${BASH_SOURCE[0]}")"
PACKAGE_ROOT="$(cd "$(dirname "$SELF")/.." && pwd)"
EMBEDDED_SOURCE="$PACKAGE_ROOT/share/novel-formatter/app"
UV_BIN="$PACKAGE_ROOT/libexec/uv"

XDG_DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
XDG_STATE_HOME="${XDG_STATE_HOME:-$HOME/.local/state}"
DATA_ROOT="$XDG_DATA_HOME/NovelFormatter"
CACHE_ROOT="$XDG_CACHE_HOME/NovelFormatter"
STATE_ROOT="$XDG_STATE_HOME/NovelFormatter"
APP_SOURCE="$DATA_ROOT/app-source"
RUNTIME_ROOT="$DATA_ROOT/linux-runtime-v1"
VENV_DIR="$RUNTIME_ROOT/venv"
PYTHON_INSTALL_DIR="$RUNTIME_ROOT/python"
LOG_DIR="$STATE_ROOT/logs"
LOG_FILE="$LOG_DIR/launcher.log"
PY="$VENV_DIR/bin/python"

mkdir -p "$DATA_ROOT" "$CACHE_ROOT" "$STATE_ROOT" "$LOG_DIR"
if [[ "${CI:-}" == "true" || "${NF_LAUNCHER_CONSOLE_LOG:-0}" == "1" ]]; then
  exec > >(tee -a "$LOG_FILE") 2>&1
else
  exec >>"$LOG_FILE" 2>&1
fi
printf '\n==== %s Novel Formatter Linux launch ====\n' "$(date '+%Y-%m-%d %H:%M:%S')"
printf 'BOOT_STAGE=launcher_start package_root=%s\n' "$PACKAGE_ROOT"

notify_error() {
  local message="$1"
  printf 'ERROR: %s\n' "$message" >&2
  if command -v zenity >/dev/null 2>&1; then
    zenity --error --title="Novel Formatter Studio" --text="$message" >/dev/null 2>&1 || true
  elif command -v kdialog >/dev/null 2>&1; then
    kdialog --error "$message" --title "Novel Formatter Studio" >/dev/null 2>&1 || true
  fi
}

if [[ "$(uname -s)" != "Linux" ]]; then
  notify_error "This package supports Linux only."
  exit 2
fi

case "$(uname -m)" in
  x86_64|amd64) EXPECTED_ARCH="x86_64" ;;
  aarch64|arm64) EXPECTED_ARCH="aarch64" ;;
  *) notify_error "Unsupported Linux architecture: $(uname -m)"; exit 2 ;;
esac

if [[ ! -x "$UV_BIN" ]]; then
  notify_error "The packaged uv runtime helper is missing or not executable. Please download the release again."
  exit 2
fi
if [[ ! -f "$EMBEDDED_SOURCE/bootstrap.py" || ! -f "$EMBEDDED_SOURCE/requirements.txt" ]]; then
  notify_error "The packaged application source is incomplete. Please download the release again."
  exit 2
fi

version_ge() {
  local a="${1#v}" b="${2#v}" top
  top="$(printf '%s\n%s\n' "$a" "$b" | sort -V | tail -n 1)"
  [[ "$top" == "$a" ]]
}

sync_embedded_source() {
  local new_manifest="$EMBEDDED_SOURCE/.release-manifest.txt"
  local old_manifest="$APP_SOURCE/.release-manifest.txt"
  local embedded_version current_version update_state
  embedded_version="$(tr -d '[:space:]' < "$EMBEDDED_SOURCE/VERSION" 2>/dev/null || true)"
  current_version="$(tr -d '[:space:]' < "$APP_SOURCE/VERSION" 2>/dev/null || true)"
  update_state="$APP_SOURCE/.runtime/source_update_state.json"

  if [[ -f "$update_state" && -n "$current_version" && -n "$embedded_version" ]] && version_ge "$current_version" "$embedded_version"; then
    printf 'BOOT_STAGE=source_git_update_preserved current=%s embedded=%s\n' "$current_version" "$embedded_version"
    return 0
  fi

  if [[ -f "$old_manifest" && -f "$new_manifest" ]] && cmp -s "$old_manifest" "$new_manifest"; then
    printf 'BOOT_STAGE=source_already_current version=%s\n' "$embedded_version"
    return 0
  fi

  mkdir -p "$APP_SOURCE"
  if [[ -f "$old_manifest" && -f "$new_manifest" ]]; then
    while IFS= read -r relative; do
      [[ -z "$relative" ]] && continue
      case "$relative" in
        .runtime/*|.venv*|.model-cache/*|.ocr-runtimes/*|.manual-model-updates/*) continue ;;
      esac
      if ! grep -Fqx -- "$relative" "$new_manifest"; then
        rm -f -- "$APP_SOURCE/$relative" 2>/dev/null || true
      fi
    done < "$old_manifest"
  fi

  # Overlay the new tracked release files while keeping writable OCR/model/runtime
  # directories that belong to the user installation.
  cp -a "$EMBEDDED_SOURCE/." "$APP_SOURCE/"
  printf 'BOOT_STAGE=source_synced version=%s\n' "$embedded_version"
}

clean_injected_environment() {
  unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONUSERBASE
  unset QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH QML2_IMPORT_PATH QML_IMPORT_PATH
  unset LD_PRELOAD
  export LD_LIBRARY_PATH="$PACKAGE_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export PYTHONNOUSERSITE=1
  export PYTHONUNBUFFERED=1
  export QT_API=pyside6
  export PIP_DISABLE_PIP_VERSION_CHECK=1
  export PIP_NO_INPUT=1
  export UV_CACHE_DIR="$CACHE_ROOT/uv"
  export UV_PYTHON_INSTALL_DIR="$PYTHON_INSTALL_DIR"
  export UV_MANAGED_PYTHON=1
  export UV_PYTHON_DOWNLOADS=automatic
}

prepare_runtime() {
  local req_file="$APP_SOURCE/requirements.txt"
  local req_hash marker_hash=""
  req_hash="$(sha256sum "$req_file" | awk '{print $1}')"
  [[ -f "$RUNTIME_ROOT/requirements.sha256" ]] && marker_hash="$(tr -d '[:space:]' < "$RUNTIME_ROOT/requirements.sha256")"

  if [[ ! -x "$PY" ]]; then
    printf 'BOOT_STAGE=create_managed_python_venv arch=%s\n' "$EXPECTED_ARCH"
    mkdir -p "$RUNTIME_ROOT"
    "$UV_BIN" venv --python 3.12 --managed-python "$VENV_DIR"
  fi

  if [[ "$marker_hash" != "$req_hash" ]]; then
    printf 'BOOT_STAGE=install_main_dependencies\n'
    "$UV_BIN" pip install --python "$PY" -r "$req_file"
    printf '%s\n' "$req_hash" > "$RUNTIME_ROOT/requirements.sha256"
  else
    printf 'BOOT_STAGE=main_dependencies_ready\n'
  fi
}

clean_injected_environment
sync_embedded_source
prepare_runtime

cd "$APP_SOURCE"
if [[ "${NF_PREPARE_ONLY:-0}" == "1" ]]; then
  printf 'BOOT_STAGE=smoke_import\n'
  QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}" "$PY" - <<'PY'
import platform
import PySide6
import fitz
import PIL
import docx
import httpx
import gui_pyside6
print("LINUX_PREPARE_OK", platform.machine(), PySide6.__version__)
PY
  exit 0
fi

printf 'BOOT_STAGE=launch_gui\n'
exec "$PY" "$APP_SOURCE/bootstrap.py" --launch
