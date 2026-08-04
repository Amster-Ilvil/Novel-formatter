#!/bin/zsh
set -eu
ROOT=${0:A:h}
APP="$ROOT/Novel Formatter Studio.app"
if [[ ! -d "$APP" ]]; then
  "$ROOT/packaging/build_mac_app.command"
fi
/usr/bin/open "$APP"
