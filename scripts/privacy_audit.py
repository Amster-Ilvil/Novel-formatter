#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fail a public build if tracked files look like local/private runtime data.

This intentionally audits only git-tracked files: GitHub Actions releases are built
from a clean checkout, so untracked files from a developer machine never enter the
release. The checks below are a second line of defence against accidentally
committing credentials, local absolute paths, logs, databases, model weights, or
user-generated documents.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/privacy_audit.py"
# This module intentionally contains generic /Users/... and /home/... regular
# expressions because it redacts those paths from runtime logs. The expressions
# themselves are not private data, so only the home-path check is skipped there;
# credential scanning remains active.
HOME_PATTERN_EXEMPT = {"utils/deployment_bootstrap.py"}

FORBIDDEN_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
}
FORBIDDEN_PARTS = {
    ".runtime",
    ".model-cache",
    ".ocr-runtimes",
    ".ocr-runtime-state",
    ".manual-model-updates",
    ".windows-runtime-state",
    ".venv",
    ".venv-app-windows",
    "venv",
    "epub_workspace",
    "debug",
    "output",
    "outputs",
    "dist",
    "build",
    "__pycache__",
}
FORBIDDEN_SUFFIXES = {
    ".log",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".epub",
    ".docx",
}
MODEL_SUFFIXES = {".pth", ".pt", ".onnx", ".ckpt", ".safetensors"}

# Construct a few prefixes in pieces so this audit file does not match itself.
SECRET_PATTERNS = [
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("OpenAI-style key", re.compile(r"(?<![A-Za-z0-9])" + "sk" + r"-[A-Za-z0-9_-]{20,}")),
    ("GitHub token", re.compile(r"(?<![A-Za-z0-9])" + "github_pat_" + r"[A-Za-z0-9_]{20,}")),
    ("GitHub token", re.compile(r"(?<![A-Za-z0-9])" + "gh" + r"[pousr]_[A-Za-z0-9]{20,}")),
    ("AWS access key", re.compile(r"(?<![A-Z0-9])" + "AKIA" + r"[0-9A-Z]{16}(?![A-Z0-9])")),
    ("Google API key", re.compile(r"(?<![A-Za-z0-9])" + "AIza" + r"[0-9A-Za-z_-]{30,}")),
    ("Hugging Face token", re.compile(r"(?<![A-Za-z0-9])" + "hf_" + r"[A-Za-z0-9]{20,}")),
]

# Allow documentation placeholders while rejecting actual-looking local home paths.
MAC_HOME = re.compile(r"/Users/(?!Shared(?:/|$)|runner(?:/|$)|<|USER(?:/|$)|username(?:/|$)|yourname(?:/|$)|path(?:/|$)|项目)[^/\s\"']+")
LINUX_HOME = re.compile(r"/home/(?!runner(?:/|$)|<|USER(?:/|$)|username(?:/|$)|yourname(?:/|$)|path(?:/|$))[^/\s\"']+")
WIN_HOME = re.compile(r"(?i)[A-Z]:\\Users\\(?!Public\\|runneradmin\\|<|USER\\|username\\|yourname\\|path\\)[^\\\r\n\"']+")


def tracked_files() -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit("privacy audit requires a git checkout")
    return [p.decode("utf-8", "surrogateescape") for p in proc.stdout.split(b"\0") if p]


def path_violations(paths: list[str]) -> list[str]:
    problems: list[str] = []
    for raw in paths:
        p = PurePosixPath(raw)
        lower_parts = {part.lower() for part in p.parts}
        if p.name.lower() in FORBIDDEN_NAMES:
            problems.append(f"forbidden tracked environment file: {raw}")
        if any(part.lower() in FORBIDDEN_PARTS for part in p.parts):
            problems.append(f"forbidden tracked runtime/generated path: {raw}")
        if p.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"forbidden tracked user/generated file: {raw}")
        if "models" in lower_parts and p.suffix.lower() in MODEL_SUFFIXES:
            problems.append(f"model weight must not be tracked: {raw}")
    return problems


def text_violations(paths: list[str]) -> list[str]:
    problems: list[str] = []
    for raw in paths:
        if raw == SELF:
            continue
        path = ROOT / raw
        try:
            if not path.is_file() or path.stat().st_size > 5 * 1024 * 1024:
                continue
            data = path.read_bytes()
        except OSError:
            continue
        if b"\x00" in data[:8192]:
            continue
        text = data.decode("utf-8", "ignore")
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(text):
                problems.append(f"possible {label} in tracked file: {raw}")
        if raw not in HOME_PATTERN_EXEMPT:
            if MAC_HOME.search(text):
                problems.append(f"possible private macOS home path in tracked file: {raw}")
            if LINUX_HOME.search(text):
                problems.append(f"possible private Linux home path in tracked file: {raw}")
            if WIN_HOME.search(text):
                problems.append(f"possible private Windows home path in tracked file: {raw}")
    return problems


def main() -> int:
    paths = tracked_files()
    problems = path_violations(paths) + text_violations(paths)
    if problems:
        print("PRIVACY AUDIT FAILED", file=sys.stderr)
        for item in sorted(set(problems)):
            print(f"- {item}", file=sys.stderr)
        return 1
    print(f"Privacy audit passed: {len(paths)} tracked files checked; no blocked local/private artifacts detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
