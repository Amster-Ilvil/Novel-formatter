#!/bin/zsh
set -u

cd "$(dirname "$0")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

APP_NAME="Novel Formatter Studio"
PYTHON_BIN=""

echo "$APP_NAME launcher"
echo "Project: $(pwd)"
echo

python_supported() {
  "$1" -c 'import sys, platform; import PySide6, PIL, fitz, docx, httpx
raise SystemExit(0 if sys.version_info >= (3, 10) and (sys.argv[1] != "arm64" or platform.machine() == "arm64") else 1)' "$(uname -m)" >/dev/null 2>&1
}

find_python() {
  local candidate

  if [[ -n "${NOVEL_FORMATTER_PYTHON:-}" ]]; then
    candidate="$(command -v "$NOVEL_FORMATTER_PYTHON" 2>/dev/null || true)"
    if [[ -n "$candidate" ]] && python_supported "$candidate"; then
      print -r -- "$candidate"
      return 0
    fi
    return 1
  fi

  for candidate in .venv/bin/python "/Library/Frameworks/Python.framework/Versions/3.14/bin/python3" python3.13 /usr/local/bin/python3 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 python3 python3.14 python3.12 python3.11 python3.10 python; do
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
  if [[ -t 0 ]]; then read; fi
  exit "$exit_code"
}

PYTHON_BIN="$(find_python || true)"
if [ -z "$PYTHON_BIN" ]; then
  echo "No compatible native Python 3.10+ with PySide6, Pillow, PyMuPDF, python-docx and httpx was found."
  echo "Automatic runtime and dependency installation is disabled."
  echo "Select an existing environment with NOVEL_FORMATTER_PYTHON."
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
