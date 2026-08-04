#!/bin/zsh
set -u

cd "$(dirname "$0")"

APP_NAME="Novel Formatter Studio"
PYTHON_BIN=""

# Use the same launcher as the macOS app so ZIP updates, the managed Python
# runtime, and the source tree all follow one path. The direct-Python fallback
# below keeps this script usable if the app shell has not been created yet.
APP_LAUNCHER="$PWD/Novel Formatter Studio.app/Contents/MacOS/launcher"
if [ -x "$APP_LAUNCHER" ]; then
  exec "$APP_LAUNCHER"
fi

echo "$APP_NAME launcher"
echo "Project: $(pwd)"
echo

python_supported() {
  "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1
}

find_python() {
  local candidate

  for candidate in python3.13 /usr/local/bin/python3 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 python3 python3.14 python3.12 python3.11 python3.10 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      candidate="$(command -v "$candidate")"
      if python_supported "$candidate"; then
        print -r -- "$candidate"
        return 0
      fi
    fi
  done

  if [ -x ".venv/bin/python" ] && python_supported ".venv/bin/python"; then
    print -r -- ".venv/bin/python"
    return 0
  fi

  return 1
}

pause_and_exit() {
  local exit_code="${1:-1}"
  echo
  echo "Press Enter to close this window."
  read
  exit "$exit_code"
}

PYTHON_BIN="$(find_python || true)"
if [ -z "$PYTHON_BIN" ]; then
  echo "Python 3.10 or newer was not found."
  echo "The app opens from Terminal with: python3 gui_pyside6.py"
  pause_and_exit 1
fi

echo "Using Python: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
echo "Python version: $("$PYTHON_BIN" -c 'import platform; print(platform.python_version())')"
echo

"$PYTHON_BIN" - <<'PYCHECK'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("PySide6") else 1)
PYCHECK
if [ $? -ne 0 ]; then
  echo "Missing dependency: PySide6"
  echo
  echo "Automatic dependency installation is disabled."
  echo "Install it manually, then double-click this file again:"
  echo "  $PYTHON_BIN -m pip install PySide6"
  pause_and_exit 1
fi

echo "Starting $APP_NAME..."
"$PYTHON_BIN" gui_pyside6.py
APP_EXIT_CODE=$?

if [ "$APP_EXIT_CODE" -ne 0 ]; then
  echo
  echo "$APP_NAME exited with error code $APP_EXIT_CODE."
  pause_and_exit "$APP_EXIT_CODE"
fi

exit 0
