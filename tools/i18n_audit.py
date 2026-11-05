#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Static inventory of untranslated CJK UI strings.

Walks the Qt UI sources plus ``run.py`` with :mod:`ast` (stdlib only) and
collects every string literal / f-string that contains CJK characters.  Each
candidate is checked against the existing runtime localization layer
(``ui.localization.translate_text``, ``ui.dialog_localization.translate_dialog_text``
and the zh templates in ``ui.message_catalog``) for English.  Anything that
is neither translated nor covered by :data:`ALLOWLIST` / an automatically
skipped non-display context is reported as ``file:line: text``.

Usage::

    python tools/i18n_audit.py            # list untranslated strings, exit 1 if any
    python tools/i18n_audit.py --stats    # per-file counts only
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_TARGETS = ("ui", "run.py", "gui_pyside6.py")

_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f]")
_FORMAT_FIELD = re.compile(r"\{[^{}]*\}|%(?:[(][^)]*[)])?[-+ #0]*\d*(?:\.\d+)?[sdifr]")
_SAMPLE = "1"

# Files whose CJK literals *are* the translation catalog (keys) or source-
# language templates; auditing them would only report the keys themselves.
CATALOG_FILES = {
    "ui/localization.py": "translation catalog: zh keys of _EXACT_EN/_EXACT_JA/_PATTERNS",
    "ui/dialog_localization.py": "translation catalog: zh keys of dialog _EN/_JA/_ZH",
    "ui/message_catalog.py": "message catalog: zh source templates",
    "ui/localization_audit_catalog.py": "translation catalog: zh keys with EN/JA values (audit gap entries)",
}

# Calls whose string arguments never reach the screen.
_NON_DISPLAY_FUNCS = {
    "print": "terminal-only output",
    "setProperty": "Qt dynamic property name/value (identifier)",
    "property": "Qt dynamic property lookup (identifier)",
    "setObjectName": "Qt object name (identifier)",
    "findChild": "Qt object name lookup (identifier)",
    "QSettings": "settings key",
    "value": "settings key lookup",
    "setValue": "settings key",
}
_LOGGER_METHODS = {"debug", "info", "warning", "error", "exception", "critical", "log"}
_RE_FUNCS = {"compile", "match", "search", "fullmatch", "sub", "subn", "findall", "finditer", "split", "escape"}


@dataclass(frozen=True)
class AllowEntry:
    path_glob: str
    text_regex: str
    reason: str


# Strings that must stay untranslated.  Every entry carries the reason.
ALLOWLIST: tuple[AllowEntry, ...] = (
    AllowEntry("*", r"⟦插图｜", "illustration marker written into the document/EPUB body"),
    AllowEntry("*", r"^(简体中文|日本語|English)$", "language names are self-identifying in every locale"),
    AllowEntry("ui/epub/tab.py", r"^未命名图书$", "fallback EPUB file name; file names are not localized"),
    AllowEntry("ui/main_window/controller.py", r"^(恢复|断点|文字层|校对|原图|历史|完整性|空间|外部ocr)$",
               "command-palette search keywords matched against the user query (English synonyms already present)"),
    AllowEntry("ui/formatter/tab.py", r"^\x5b图片:", "image-block marker written to and parsed back from the editable body text"),
    AllowEntry("ui/formatter/tab.py", r"吾輩は猫である", "Japanese book-text sample used only for font-width measurement"),
    AllowEntry("*", r"^\S*[.](json|txt|zip|md|docx|epub|csv)$", "default export file names (no whitespace + extension); file names are not localized"),
    AllowEntry("ui/ocr/proofread_widgets.py", r"^国$", "reference glyph for QFontMetrics column-width measurement"),
    AllowEntry("ui/theme/tokens.py", r"QMainWindow [{]", "Qt stylesheet; CJK only appears in font-family names"),
    AllowEntry("ui/common/interaction_logic.py", r"^(保存|开始|应用|确定|确认|好的|開始|適用|確定|確認|実行)$",
               "_PRIMARY_PREFIXES: default-button detection matches raw button text (logic, not display)"),
)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    text: str

    def render(self) -> str:
        shown = self.text.replace("\n", "\\n")
        if len(shown) > 140:
            shown = shown[:137] + "..."
        return f"{self.path}:{self.line}: {shown}"


def _has_cjk(text: str) -> bool:
    return bool(_CJK.search(text))


def _translators():
    from ui.localization import LANG_EN, translate_text
    from ui.dialog_localization import translate_dialog_text
    from ui import message_catalog

    zh_templates = {
        templates.get("zh_CN", "")
        for templates in getattr(message_catalog, "_MESSAGES", {}).values()
    }

    def is_translated(sample: str, raw: str) -> bool:
        if raw in zh_templates:
            return True
        for candidate in {sample, raw}:
            if translate_text(candidate, LANG_EN) != candidate:
                return True
            if translate_dialog_text(candidate, LANG_EN) != candidate:
                return True
        return False

    return is_translated


def _call_name(node: ast.Call) -> tuple[str, str]:
    func = node.func
    if isinstance(func, ast.Name):
        return "", func.id
    if isinstance(func, ast.Attribute):
        owner = func.value
        owner_name = owner.id if isinstance(owner, ast.Name) else (owner.attr if isinstance(owner, ast.Attribute) else "")
        return owner_name, func.attr
    return "", ""


class _Collector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.items: list[tuple[int, str, str]] = []  # (line, raw, sample)
        self._skip: set[int] = set()

    # --- non-display contexts -------------------------------------------
    def _mark(self, node: ast.AST | None) -> None:
        if node is None:
            return
        for sub in ast.walk(node):
            self._skip.add(id(sub))

    def _mark_docstring(self, body: list[ast.stmt]) -> None:
        first = next(iter(body), None)
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            self._mark(first.value)

    def visit_Module(self, node: ast.Module) -> None:
        self._mark_docstring(node.body)
        self.generic_visit(node)

    def visit_FunctionDef(self, node) -> None:
        self._mark_docstring(node.body)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._mark_docstring(node.body)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        # Literals compared with data drive logic; translating them would
        # change behaviour, they are not display text.
        self._mark(node.left)
        for comp in node.comparators:
            self._mark(comp)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        self._mark(node.slice)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        owner, name = _call_name(node)
        if owner == "re" and name in _RE_FUNCS:
            self._mark(node)
        elif name in _NON_DISPLAY_FUNCS:
            self._mark(node)
        elif name in _LOGGER_METHODS and owner.lower().endswith(("log", "logger", "logging")):
            self._mark(node)
        self.generic_visit(node)

    # --- candidates ------------------------------------------------------
    def visit_Constant(self, node: ast.Constant) -> None:
        if id(node) in self._skip or not isinstance(node.value, str) or not _has_cjk(node.value):
            return
        raw = node.value
        self.items.append((node.lineno, raw, _FORMAT_FIELD.sub(_SAMPLE, raw)))

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        if id(node) in self._skip:
            return
        raw_parts: list[str] = []
        sample_parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                raw_parts.append(value.value)
                sample_parts.append(value.value)
            else:
                raw_parts.append("{…}")
                sample_parts.append(_SAMPLE)
        raw = "".join(raw_parts)
        if _has_cjk(raw):
            self.items.append((node.lineno, raw, "".join(sample_parts)))
        # Do not descend: constant parts were handled above.


def _iter_files(targets: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for target in targets:
        path = ROOT / target
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(p for p in sorted(path.rglob("*.py")) if "__pycache__" not in p.parts)
    return files


def _allowed(rel: str, text: str) -> bool:
    for entry in ALLOWLIST:
        if fnmatch.fnmatch(rel, entry.path_glob) and re.search(entry.text_regex, text):
            return True
    return False


def audit(targets: tuple[str, ...] = DEFAULT_TARGETS) -> list[Finding]:
    """Return untranslated CJK UI strings in *targets* (paths relative to repo root)."""
    is_translated = _translators()
    findings: list[Finding] = []
    for path in _iter_files(targets):
        rel = path.relative_to(ROOT).as_posix()
        if rel in CATALOG_FILES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        collector = _Collector()
        collector.visit(tree)
        for line, raw, sample in collector.items:
            if _allowed(rel, raw) or is_translated(sample, raw):
                continue
            findings.append(Finding(rel, line, raw))
    return findings


BACKEND_DIRS = ("adapters", "engine", "builder", "core", "utils")
_BACKEND_SINK = re.compile(r"(log|progress|status|callback|emit|notify)", re.I)


def _outer_literals(node: ast.AST) -> list[tuple[int, str, str]]:
    """Top-level string literals / f-strings inside *node* (no nested f-string parts)."""
    found: list[tuple[int, str, str]] = []

    def walk(sub: ast.AST) -> None:
        if isinstance(sub, ast.JoinedStr):
            raw = "".join(v.value if isinstance(v, ast.Constant) else "{…}" for v in sub.values)
            sample = "".join(v.value if isinstance(v, ast.Constant) else _SAMPLE for v in sub.values)
            if _has_cjk(raw):
                found.append((sub.lineno, raw, sample))
            return
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            if _has_cjk(sub.value):
                found.append((sub.lineno, sub.value, _FORMAT_FIELD.sub(_SAMPLE, sub.value)))
            return
        for child in ast.iter_child_nodes(sub):
            walk(child)

    walk(node)
    return found


def audit_backend(dirs: tuple[str, ...] = BACKEND_DIRS) -> list[Finding]:
    """Backend text that can reach the UI as-is: exception messages and
    log/progress/status callback arguments.  Checked against the same runtime
    translation layer used by message boxes and the OCR log."""
    is_translated = _translators()
    findings: list[Finding] = []
    for path in _iter_files(dirs):
        rel = path.relative_to(ROOT).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            candidates: list[tuple[int, str, str]] = []
            if isinstance(node, ast.Raise) and node.exc is not None:
                candidates = _outer_literals(node.exc)
            elif isinstance(node, ast.Call):
                _, name = _call_name(node)
                if name and name != "print" and _BACKEND_SINK.search(name):
                    for arg in list(node.args) + [kw.value for kw in node.keywords]:
                        candidates.extend(_outer_literals(arg))
            for line, raw, sample in candidates:
                if _allowed(rel, raw) or is_translated(sample, raw):
                    continue
                findings.append(Finding(rel, line, raw))
    unique: dict[tuple[str, int, str], Finding] = {}
    for item in findings:
        unique.setdefault((item.path, item.line, item.text), item)
    return list(unique.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List CJK UI strings that have no English translation.")
    parser.add_argument("targets", nargs="*", default=list(DEFAULT_TARGETS), help="files/dirs relative to the repo root")
    parser.add_argument("--stats", action="store_true", help="print per-file counts instead of every string")
    parser.add_argument("--backend", action="store_true",
                        help="audit backend exception/log/progress text (adapters, engine, builder, core, utils) instead of the UI")
    args = parser.parse_args(argv)
    if args.backend:
        targets = tuple(args.targets) if args.targets != list(DEFAULT_TARGETS) else BACKEND_DIRS
        findings = audit_backend(targets)
    else:
        findings = audit(tuple(args.targets))
    if args.stats:
        counts: dict[str, int] = {}
        for item in findings:
            counts[item.path] = counts.get(item.path, 0) + 1
        for rel in sorted(counts, key=lambda name: (-counts.get(name, 0), name)):
            print(f"{counts.get(rel, 0):6d}  {rel}")
    else:
        for item in findings:
            print(item.render())
    print(f"untranslated: {len(findings)}", file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
