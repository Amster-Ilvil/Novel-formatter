#!/bin/zsh
set -u
setopt NULL_GLOB

cd "$(dirname "$0")"

APP_NAME="Novel Formatter Studio"
PACKAGE_DIR="$HOME/Desktop/更新包"
ZIP_PATH="${1:-}"
TMP_DIR=""

pause_and_exit() {
  local exit_code="${1:-1}"
  echo
  echo "Press Enter to close this window."
  read
  exit "$exit_code"
}

cleanup() {
  if [ -n "${TMP_DIR:-}" ] && [ -d "$TMP_DIR" ]; then
    rm -rf "$TMP_DIR"
  fi
}
trap cleanup EXIT

normalize_path() {
  local raw="$1"
  raw="${raw/#\~/$HOME}"
  raw="${raw%\"}"
  raw="${raw#\"}"
  raw="${raw%\'}"
  raw="${raw#\'}"
  raw="${raw% }"
  print -r -- "$raw"
}

find_package_root() {
  local search_dir="$1"
  local root=""

  if [ -f "$search_dir/gui_pyside6.py" ] || [ -f "$search_dir/run.py" ]; then
    print -r -- "$search_dir"
    return 0
  fi

  root="$(find "$search_dir" -mindepth 1 -maxdepth 4 -type f \( -name gui_pyside6.py -o -name run.py \) -print -quit 2>/dev/null)"
  if [ -n "$root" ]; then
    dirname "$root"
    return 0
  fi

  return 1
}

choose_zip() {
  local candidates latest typed

  candidates=(
    "$PACKAGE_DIR"/Novel-formatter*.zip(N.om)
    "$PACKAGE_DIR"/Novel-formatter-core*.zip(N.om)
    "$PACKAGE_DIR"/NovelFormatter*.zip(N.om)
    "$PACKAGE_DIR"/NovelFormatter-core*.zip(N.om)
    "$PACKAGE_DIR"/Novel\ Formatter*.zip(N.om)
    "$PACKAGE_DIR"/小说格式*.zip(N.om)
    "$PACKAGE_DIR"/小说排版*.zip(N.om)
    "$HOME/Downloads"/Novel-formatter*.zip(N.om)
    "$HOME/Downloads"/Novel-formatter-core*.zip(N.om)
    "$HOME/Downloads"/NovelFormatter*.zip(N.om)
  )

  latest="${candidates[1]:-}"
  if [ -n "$latest" ]; then
    echo "Found latest package:"
    echo "$latest"
    echo
    echo "Press Enter to use it, or paste another .zip path:"
    read -r typed
    if [ -n "$typed" ]; then
      ZIP_PATH="$typed"
    else
      ZIP_PATH="$latest"
    fi
  else
    echo "No Novel Formatter package was found in:"
    echo "$PACKAGE_DIR"
    echo "$HOME/Downloads"
    echo
    echo "Paste the full path of the Novel Formatter .zip package:"
    read -r ZIP_PATH
  fi
}

find_python() {
  local candidate
  for candidate in python3.13 /usr/local/bin/python3 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 python3 python3.14 python3.12 python3.11 python3.10 python; do
    if [[ "$candidate" == */* ]]; then
      [ -x "$candidate" ] || continue
    else
      command -v "$candidate" >/dev/null 2>&1 || continue
      candidate="$(command -v "$candidate")"
    fi
    "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1 || continue
    print -r -- "$candidate"
    return 0
  done
  return 1
}

echo "$APP_NAME updater"
echo "Project: $(pwd)"
echo

if [ -z "$ZIP_PATH" ]; then
  choose_zip
fi

ZIP_PATH="$(normalize_path "$ZIP_PATH")"

if [ ! -f "$ZIP_PATH" ]; then
  echo
  echo "Package not found:"
  echo "$ZIP_PATH"
  pause_and_exit 1
fi

case "$ZIP_PATH" in
  *.zip|*.ZIP) ;;
  *)
    echo
    echo "This updater needs a .zip package:"
    echo "$ZIP_PATH"
    pause_and_exit 1
    ;;
esac

TMP_DIR="$(mktemp -d /private/tmp/novel-formatter-update.XXXXXX)"

echo "Unzipping package..."
unzip -q -o "$ZIP_PATH" -d "$TMP_DIR"
if [ $? -ne 0 ]; then
  echo
  echo "Failed to unzip package:"
  echo "$ZIP_PATH"
  pause_and_exit 1
fi

ROOT_DIR="$(find_package_root "$TMP_DIR" || true)"
if [ -z "$ROOT_DIR" ]; then
  echo
  echo "This does not look like a Novel Formatter source package:"
  echo "$ZIP_PATH"
  echo
  echo "Expected gui_pyside6.py or run.py inside the package."
  pause_and_exit 1
fi

echo "Package root:"
echo "$ROOT_DIR"
echo
echo "Updating files..."

rsync -a --delete \
  --exclude '.git/' \
  --exclude '.agents/' \
  --exclude '.codex/' \
  --exclude '.venv*/' \
  --exclude '.model-cache/' \
  --exclude '.ocr-runtimes/' \
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
  --exclude '*.app/' \
  --exclude 'Novel Formatter Studio.app/' \
  --exclude 'models/cache/' \
  --exclude 'models/checkpoints/' \
  --exclude 'model_cache/' \
  --exclude '.cache/' \
  --exclude 'run_novel_formatter.command' \
  --exclude 'update_novel_formatter.command' \
  "$ROOT_DIR/" \
  ./

if [ $? -ne 0 ]; then
  echo
  echo "File update failed."
  pause_and_exit 1
fi

chmod +x run_novel_formatter.command 2>/dev/null || true
chmod +x update_novel_formatter.command 2>/dev/null || true

PYTHON_BIN="$(find_python || true)"
if [ -n "$PYTHON_BIN" ]; then
  echo
  echo "Running Python syntax check..."
  "$PYTHON_BIN" -m compileall -q \
    adapters ai builder core engine models ui utils gui_pyside6.py run.py
  VERIFY_STATUS=$?
  if [ "$VERIFY_STATUS" -ne 0 ]; then
    echo
    echo "Syntax check failed. The files were updated, but the app may need manual repair."
    pause_and_exit "$VERIFY_STATUS"
  fi
else
  echo
  echo "Skipping syntax check because Python was not found."
fi

echo
echo "Update complete."
echo "Kept local virtual environments, model/cache folders, app bundle, output files, and local launch/update scripts."
echo
echo "You can now double-click run_novel_formatter.command."
pause_and_exit 0
