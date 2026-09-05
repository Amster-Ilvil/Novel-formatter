#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the official ``mlx_vlm.server`` with a parent-process watchdog.

The MLX environment executes this tiny stdlib-only wrapper so a crashed/forced
Novel Formatter process cannot leave a Metal model server orphaned in memory.
All command-line arguments are forwarded unchanged to ``mlx_vlm.server``.
"""
from __future__ import annotations

import os
import runpy
import sys
import threading
import time


def _parent_watchdog(parent_pid: int) -> None:
    if parent_pid <= 1:
        return
    while True:
        time.sleep(1.0)
        # A re-parented process means Novel Formatter is gone even if the old
        # PID is quickly reused by an unrelated process.
        if os.getppid() != parent_pid:
            os._exit(0)
        try:
            # Signal 0 checks existence without modifying the target process.
            os.kill(parent_pid, 0)
        except (ProcessLookupError, PermissionError, OSError):
            os._exit(0)


def main() -> None:
    try:
        parent_pid = int(os.environ.get("NOVEL_FORMATTER_MLX_PARENT_PID", "0") or 0)
    except ValueError:
        parent_pid = 0
    if parent_pid > 1:
        threading.Thread(
            target=_parent_watchdog,
            args=(parent_pid,),
            name="novel-formatter-mlx-parent-watchdog",
            daemon=True,
        ).start()
    # Keep the upstream CLI contract exactly intact.
    sys.argv[0] = "mlx_vlm.server"
    runpy.run_module("mlx_vlm.server", run_name="__main__")


if __name__ == "__main__":
    main()
