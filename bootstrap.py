#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public application bootstrap.

The bootstrap may prepare Python packages required by the GUI, but it never
installs OCR-specific dependencies or downloads OCR models. Missing OCR
resources are handled only after the user starts OCR and confirms the prompt.
Installed models are never checked or updated here.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Novel Formatter 应用启动准备")
    parser.add_argument("--install-main-deps", action="store_true", help="安装主程序依赖")
    parser.add_argument("--prepare-models", action="store_true", help="兼容旧命令；已停用，不会下载 OCR 模型或依赖")
    parser.add_argument("--profile", default=os.environ.get("NOVEL_FORMATTER_BOOTSTRAP_PROFILE", "none"), choices=("auto", "lite", "standard", "full", "none"), help="兼容旧命令；OCR 预装档位已停用")
    parser.add_argument("--strict-models", action="store_true", help="兼容旧命令；OCR 预装已停用")
    parser.add_argument("--launch", action="store_true", help="准备完成后启动 GUI")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    from utils.deployment_bootstrap import (
        apply_application_download_environment,
        install_main_dependencies,
        select_application_download_sources,
    )

    sources = select_application_download_sources()
    apply_application_download_environment(sources)
    if args.install_main_deps:
        install_main_dependencies(sources=sources)
    if args.prepare_models:
        print(
            "OCR 预装已停用：启动阶段不会下载任何 OCR 模型或专用依赖。"
            "请在开始 OCR 后按界面提示确认安装。",
            flush=True,
        )
    if args.launch:
        env = dict(os.environ)
        process = subprocess.Popen([sys.executable, str(ROOT / "gui_pyside6.py")], cwd=ROOT, env=env)
        try:
            return int(process.wait())
        except KeyboardInterrupt:
            process.terminate()
            try:
                return int(process.wait(timeout=10))
            except subprocess.TimeoutExpired:
                process.kill()
                return int(process.wait(timeout=10))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
