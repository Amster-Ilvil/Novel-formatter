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
    """Require complete translations for UI and backend messages alike."""
    return _strict_translators()


# Corner-bracket quotes in an English message cite book/source tokens verbatim
# (e.g. 『削除予定』); they are content, not untranslated UI text.
_QUOTED_BOOK_TOKEN = re.compile("『[^』]*』|「[^」]*」")

# These exact UI descriptions demonstrate source syntax. Only the listed
# examples may remain CJK; the surrounding explanation must be translated.
_PRESERVED_BOOK_TOKENS: dict[str, tuple[str, ...]] = {
    "删除每话附带的（前書）/（後書き）编辑备注": ("（前書）", "（後書き）"),
    "恢复｜漢字《よみ》ruby标注": ("｜漢字《よみ》",),
    "识别章/话/回/Episode标题，完整生成TOC": ("章/话/回/Episode",),
    "检测与正文粘连的“序章 / 第X章 / 幕间 / 后记”等标题，并从正文块中单独拆出。": (
        "序章 / 第X章 / 幕间 / 后记",
    ),
    "识别每话开头或结尾的（前書き）/（後書き）及同类编辑备注。": (
        "（前書き）", "（後書き）",
    ),
    "识别 OCR 中残留的竖线、括号和假名组合，恢复｜漢字《よみ》格式。": (
        "｜漢字《よみ》",
    ),
    "识别序章、第X章、第X话、12话、第X回、Episode、幕间、终章、后记等标题，并逐条建立目录。": (
        "序章", "第X章", "第X话", "12话", "第X回", "幕间", "终章", "后记",
    ),
    "默认关闭。勾选后 PDF 文字层保留真实作者前书/后记；不勾选时允许删除“数字（前書/後書き）”段落。": (
        "前書/後書き",
    ),
}


def _strict_translators():
    """Fully-translated check for user-visible literals.

    A partial match (a generic template rewriting only a literal tail) changes
    the text but still leaves Chinese on screen, so the English rendering of
    the sample must contain no CJK at all.  Samples fill every slot with ASCII.
    """
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
            for translate in (translate_text, translate_dialog_text):
                rendered = translate(candidate, LANG_EN)
                chrome = _QUOTED_BOOK_TOKEN.sub("", rendered)
                for token in _PRESERVED_BOOK_TOKENS.get(raw, ()):
                    chrome = chrome.replace(token, "")
                if rendered != candidate and not _has_cjk(chrome):
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
    def visit_BinOp(self, node: ast.BinOp) -> None:
        if id(node) in self._skip:
            return
        if _is_concat(node):
            raw, sample, has_text = _flatten_text(node)
            if has_text:
                if _has_cjk(raw):
                    self.items.append((node.lineno, raw, sample))
                # Do not collect the template's literal parts twice. Other
                # operands can contain conditional UI text or nested calls;
                # visit those with the normal non-display context rules.
                pending = [node.right, node.left]
                while pending:
                    part = pending.pop()
                    if _is_concat(part):
                        pending.extend((part.right, part.left))
                    elif not isinstance(part, (ast.Constant, ast.JoinedStr)):
                        self.visit(part)
                return
        self.generic_visit(node)

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


# ── backend visibility classification (category A / B) ─────────────────────
# Category A ("visible") is text that reaches the screen.  The rules follow the
# UI wiring: exceptions are shown by show_error_dialog / QMessageBox (str(exc)
# or the last traceback line); worker log/progress/status/phase callbacks are
# connected to LocalizedLogView or progress labels; report/warning fields are
# rendered by review panels.  Category B ("internal") is everything else:
# regular expressions, comparison operands, keys, docstrings, print(), stdlib
# logging/warnings (no logging.Handler bridges into the UI), CLI help text and
# arguments of non-display helpers.

VISIBLE_BACKEND_DIRS = ("adapters", "engine")
VISIBLE = "A"
INTERNAL = "B"

_SINK_TOKENS = frozenset({"log", "progress", "status", "callback", "emit", "notify", "report", "phase"})
_QT_TEXT_CALLS = frozenset({
    "setText", "setWindowTitle", "setToolTip", "setPlaceholderText", "setStatusTip", "setTitle",
    "setLabelText", "setInformativeText", "setDetailedText", "setPlainText", "appendPlainText",
    "showMessage", "addItem", "addItems", "addTab", "setTabText", "setHeaderLabels", "setItemText",
    "QLabel", "QPushButton", "QCheckBox", "QRadioButton", "QGroupBox", "QAction", "QToolButton",
})
_MSGBOX_METHODS = frozenset({"information", "warning", "critical", "question"})
_LOGGER_OWNERS = frozenset({"logging", "logger", "log", "_logger", "_log", "LOGGER", "LOG"})
_PASSTHROUGH_CALLS = frozenset({"format", "format_map", "join", "str", "fromkeys", "list", "tuple", "sorted"})
_DEFAULT_VALUE_CALLS = frozenset({"get", "getattr", "pop", "setdefault"})
_COLLECT_METHODS = frozenset({"append", "add", "extend", "insert", "appendleft"})
_CLI_CALLS = frozenset({"add_argument", "ArgumentParser", "add_parser", "add_argument_group", "add_subparsers"})
_REPORT_KEYS = frozenset({
    "message", "msg", "reason", "reasons", "detail", "details", "summary", "warning", "warnings",
    "error", "errors", "note", "notes", "hint", "title", "label", "status", "status_text",
    "description", "explanation", "issue", "issues", "risk", "risks", "suggestion", "action",
    "caption", "tooltip", "headline", "stage", "phase",
})
_REPORT_CONTAINER = re.compile(
    r"(?:^|_)(?:warning|reason|issue|note|message|error|problem|risk|hint|summary|detail|notice|"
    r"blocker|suggestion|log)s?$",
    re.I,
)
_REPORT_FUNC = re.compile(
    r"(message|reason|summary|label|status|describe|description|hint|title|caption|explain|"
    r"warning|notice|detail|error)",
    re.I,
)
_EXC_NAME = re.compile(r"(Error|Exception|Failure|Interrupted|Cancelled|Canceled)$")
# Value objects whose text fields are rendered as reasons / statuses.
_REPORT_CTOR = re.compile(
    r"(Decision|Status|Probe|Component|Report|Issue|Warning|Quality|State|Notice|Summary|Step|Stage)$"
)
# engine/*_view_model.py feeds Qt views directly: its labels are display text.
_VIEW_MODEL_REASONS = frozenset({
    "assigned value", "return value", "collection element", "module-level data", "definition-level value",
})
_ASCII_LETTER = re.compile(r"[A-Za-z]")
_NAME_TOKEN = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


@dataclass(frozen=True)
class BackendLiteral:
    path: str
    line: int
    text: str
    sample: str
    category: str
    reason: str


def _name_tokens(name: str) -> list[str]:
    return [token.lower() for token in _NAME_TOKEN.findall(name)]


def _is_logger_call(owner: str, name: str) -> bool:
    if owner == "warnings":
        return True
    if name not in _LOGGER_METHODS:
        return False
    return owner in _LOGGER_OWNERS or owner.lower().endswith(("logger", "logging"))


def _sink_kind(call: ast.Call) -> str | None:
    owner, name = _call_name(call)
    if not name or name == "print" or owner == "re" or _is_logger_call(owner, name):
        return None
    if name in _QT_TEXT_CALLS or (name in _MSGBOX_METHODS and "MessageBox" in owner):
        return "Qt widget/dialog text"
    if any(token in _SINK_TOKENS or token.startswith("emit") for token in _name_tokens(name)):
        return "log/progress/status callback"
    return None


class _FileContext:
    """Parent links, enclosing scopes and per-scope names that reach a sink."""

    def __init__(self, tree: ast.AST) -> None:
        self.tree = tree
        self.parents: dict[int, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                self.parents[id(child)] = node
        self._visible: dict[int, set[str]] = {}
        self._forward: dict[str, tuple[list[str], set[str]]] | None = None

    def forwarded_params(self, name: str) -> tuple[list[str], set[str]] | None:
        """Parameters of a same-file function *name* that flow into raise/sink."""
        if self._forward is None:
            self._forward = {}
            for node in ast.walk(self.tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                params = [arg.arg for arg in node.args.posonlyargs + node.args.args]
                if params and params.pop(0) not in ("self", "cls"):
                    params = [arg.arg for arg in node.args.posonlyargs + node.args.args]
                names = params + [arg.arg for arg in node.args.kwonlyargs]
                visible = set(names) & self.visible_names(node)
                if visible:
                    self._forward.setdefault(node.name, (params, visible))
        return self._forward.get(name)

    def scope_of(self, node: ast.AST) -> ast.AST:
        cur = self.parents.get(id(node))
        while cur is not None:
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return cur
            cur = self.parents.get(id(cur))
        return self.tree

    def visible_names(self, scope: ast.AST) -> set[str]:
        cached = self._visible.get(id(scope))
        if cached is not None:
            return cached
        names: set[str] = set()
        for sub in ast.walk(scope):
            payload: list[ast.AST] = []
            if isinstance(sub, ast.Raise) and sub.exc is not None:
                payload.append(sub.exc)
            elif isinstance(sub, ast.Call) and _sink_kind(sub):
                payload.extend(sub.args)
                payload.extend(kw.value for kw in sub.keywords)
            for node in payload:
                for item in ast.walk(node):
                    if isinstance(item, ast.Name):
                        names.add(item.id)
                    elif isinstance(item, ast.Attribute):
                        names.add(item.attr)
        self._visible[id(scope)] = names
        return names

    def file_visible_names(self) -> set[str]:
        """Union of names reaching a raise/sink anywhere in the file."""
        cached = self._visible.get(-1)
        if cached is not None:
            return cached
        names: set[str] = set(self.visible_names(self.tree))
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names |= self.visible_names(node)
        self._visible[-1] = names
        return names


def _target_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            names.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            names.add(sub.attr)
    return names


def _classify_literal(node: ast.AST, ctx: _FileContext) -> tuple[str, str]:
    cur = node
    while True:
        par = ctx.parents.get(id(cur))
        if par is None:
            return INTERNAL, "module-level data"
        if isinstance(par, ast.Expr):
            return INTERNAL, "docstring / bare expression"
        if isinstance(par, ast.Compare):
            return INTERNAL, "comparison operand"
        if isinstance(par, ast.Subscript) and cur is par.slice:
            return INTERNAL, "subscript key"
        if isinstance(par, ast.Dict):
            if any(cur is key for key in par.keys):
                return INTERNAL, "dict key"
            key = next((k for k, v in zip(par.keys, par.values) if v is cur), None)
            if isinstance(key, ast.Constant) and isinstance(key.value, str) and key.value.lower() in _REPORT_KEYS:
                return VISIBLE, "report/status field"
        elif isinstance(par, ast.keyword):
            if par.arg and par.arg.lower() in _REPORT_KEYS:
                return VISIBLE, "report/status keyword"
        elif isinstance(par, ast.Attribute):
            if par.attr not in ("format", "format_map"):
                return INTERNAL, f"receiver of .{par.attr}()"
            call = ctx.parents.get(id(par))
            if not isinstance(call, ast.Call):
                return INTERNAL, "attribute access"
            cur = call
            continue
        elif isinstance(par, ast.Call):
            owner, name = _call_name(par)
            kind = _sink_kind(par)
            if kind:
                return VISIBLE, kind
            if isinstance(ctx.parents.get(id(par)), ast.Raise):
                return VISIBLE, "raise"
            if owner == "re" or name in _RE_FUNCS:
                return INTERNAL, "regular expression"
            if name == "print":
                return INTERNAL, "terminal output (print)"
            if _is_logger_call(owner, name):
                return INTERNAL, "stdlib logging/warnings (no UI handler)"
            if name in _CLI_CALLS:
                return INTERNAL, "CLI help text"
            if name in _NON_DISPLAY_FUNCS:
                return INTERNAL, _NON_DISPLAY_FUNCS.get(name, "non-display call")
            if name in _COLLECT_METHODS and isinstance(par.func, ast.Attribute):
                target = par.func.value
                target_name = target.id if isinstance(target, ast.Name) else getattr(target, "attr", "")
                visible = ctx.visible_names(ctx.scope_of(par))
                if target_name in visible or _REPORT_CONTAINER.search(target_name or ""):
                    return VISIBLE, "collected into a report/log list"
                return INTERNAL, "collection element"
            forwarded = ctx.forwarded_params(name) if name else None
            if forwarded is not None:
                params, visible = forwarded
                param = cur.arg if isinstance(cur, ast.keyword) else None
                if param is None:
                    index = next((n for n, arg in enumerate(par.args) if arg is cur), None)
                    if index is not None and index < len(params):
                        param = params[index]
                if param in visible:
                    return VISIBLE, "argument forwarded to raise/sink"
            if _EXC_NAME.search(name or ""):
                return VISIBLE, "exception message"
            if _REPORT_CTOR.search(name or ""):
                return VISIBLE, "report/decision object field"
            if name in _DEFAULT_VALUE_CALLS and cur is not next(iter(par.args), None):
                # ``mapping.get(key, "fallback")``: the key is internal, the
                # fallback is the value the caller shows -> keep walking up.
                cur = par
                continue
            if name not in _PASSTHROUGH_CALLS:
                return INTERNAL, f"argument of {name or 'call'}()"
        elif isinstance(par, ast.Raise):
            return VISIBLE, "raise"
        elif isinstance(par, ast.Return):
            scope = ctx.scope_of(par)
            if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if _REPORT_FUNC.search(scope.name):
                    return VISIBLE, "returned message"
                if scope.name in ctx.file_visible_names():
                    return VISIBLE, "returned into raise/sink"
            return INTERNAL, "return value"
        elif isinstance(par, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = par.targets if isinstance(par, ast.Assign) else [par.target]
            names: set[str] = set()
            for target in targets:
                names |= _target_names(target)
            if names & ctx.visible_names(ctx.scope_of(par)):
                return VISIBLE, "variable passed to raise/sink"
            # Name heuristics do not apply to UPPER_CASE constants (lookup tables,
            # matching hints); those count only when they actually reach a sink.
            if any((n.lower() in _REPORT_KEYS or _REPORT_CONTAINER.search(n)) and not n.isupper() for n in names):
                return VISIBLE, "report variable/attribute"
            return INTERNAL, "assigned value"
        elif isinstance(par, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda, ast.arguments)):
            return INTERNAL, "definition-level value"
        cur = par


def _is_concat(node: ast.AST | None) -> bool:
    return isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)


def _flatten_text(node: ast.AST) -> tuple[str, str, bool]:
    """(raw, sample, has_text) for a string expression; non-text operands become slots."""
    if _is_concat(node):
        left = _flatten_text(node.left)
        right = _flatten_text(node.right)
        return left.__getitem__(0) + right.__getitem__(0), left.__getitem__(1) + right.__getitem__(1), left.__getitem__(2) or right.__getitem__(2)
    if isinstance(node, ast.JoinedStr):
        raw = "".join(v.value if isinstance(v, ast.Constant) else "{…}" for v in node.values)
        sample = "".join(v.value if isinstance(v, ast.Constant) else _SAMPLE for v in node.values)
        return raw, sample, True
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value, _FORMAT_FIELD.sub(_SAMPLE, node.value), True
    return "{…}", _SAMPLE, False


def _iter_backend_literals(tree: ast.AST, ctx: _FileContext):
    """CJK string literals; a ``"a" + x + f"b{y}"`` chain is yielded once as a template."""
    for node in ast.walk(tree):
        parent = ctx.parents.get(id(node))
        if _is_concat(node):
            if _is_concat(parent):
                continue
            raw, sample, has_text = _flatten_text(node)
            if not has_text:
                continue
        elif isinstance(node, ast.JoinedStr):
            if isinstance(parent, ast.FormattedValue) or _is_concat(parent):
                continue
            raw, sample, _ = _flatten_text(node)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if isinstance(parent, ast.JoinedStr) or _is_concat(parent):
                continue
            raw, sample, _ = _flatten_text(node)
        else:
            continue
        if _has_cjk(raw):
            yield node, raw, sample


def classify_backend(dirs: tuple[str, ...] = VISIBLE_BACKEND_DIRS) -> list[BackendLiteral]:
    """Every CJK literal in *dirs* with its visibility category and reason."""
    items: list[BackendLiteral] = []
    for path in _iter_files(dirs):
        rel = path.relative_to(ROOT).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        except SyntaxError:
            continue
        ctx = _FileContext(tree)
        for node, raw, sample in _iter_backend_literals(tree, ctx):
            category, reason = _classify_literal(node, ctx)
            if category == INTERNAL and rel.endswith("_view_model.py") and reason in _VIEW_MODEL_REASONS:
                category, reason = VISIBLE, "view-model label"
            if category == VISIBLE and _allowed(rel, raw):
                category, reason = INTERNAL, "allowlisted marker"
            if category == VISIBLE and len(raw.strip()) <= 1:
                category, reason = INTERNAL, "single-character token"
            if category == VISIBLE and len(_ASCII_LETTER.findall(raw)) >= 20 and len(_CJK.findall(raw)) <= 3:
                category, reason = INTERNAL, "English text quoting a CJK token"
            items.append(BackendLiteral(rel, node.lineno, raw, sample, category, reason))
    items.sort(key=lambda item: (item.path, item.line, item.text))
    return items


def audit_backend_visible(dirs: tuple[str, ...] = VISIBLE_BACKEND_DIRS) -> list[Finding]:
    """Category-A backend literals without a complete English translation."""
    is_translated = _strict_translators()
    unique: dict[tuple[str, int, str], Finding] = {}
    for item in classify_backend(dirs):
        if item.category != VISIBLE or is_translated(item.sample, item.text):
            continue
        unique.setdefault((item.path, item.line, item.text), Finding(item.path, item.line, item.text))
    return list(unique.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List CJK UI strings that have no English translation.")
    parser.add_argument("targets", nargs="*", default=list(DEFAULT_TARGETS), help="files/dirs relative to the repo root")
    parser.add_argument("--stats", action="store_true", help="print per-file counts instead of every string")
    parser.add_argument("--backend", action="store_true",
                        help="audit backend exception/log/progress text (adapters, engine, builder, core, utils) instead of the UI")
    parser.add_argument("--backend-visible", action="store_true",
                        help="classify every CJK literal in adapters/ and engine/ as user-visible (A) or internal (B) "
                             "and list untranslated A literals; with --stats print the A/B breakdown by reason")
    args = parser.parse_args(argv)
    if args.backend_visible:
        targets = tuple(args.targets) if args.targets != list(DEFAULT_TARGETS) else VISIBLE_BACKEND_DIRS
        findings = audit_backend_visible(targets)
        if args.stats:
            reasons: dict[tuple[str, str], int] = {}
            for item in classify_backend(targets):
                key = (item.category, item.reason)
                reasons[key] = reasons.get(key, 0) + 1
            for category in (VISIBLE, INTERNAL):
                total = sum(n for (cat, _), n in reasons.items() if cat == category)
                print(f"{category} {total:6d}  total")
                ordered = sorted(((n, reason) for (cat, reason), n in reasons.items() if cat == category), reverse=True)
                for n, reason in ordered:
                    print(f"{category} {n:6d}  {reason}")
    elif args.backend:
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
