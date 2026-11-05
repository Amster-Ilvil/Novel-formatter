"""Regression gate for the UI/backend localization audit.

UI text and backend messages from builder/, core/ and utils/ must all be
covered by the runtime translation layer.  adapters/ and engine/ are tracked
debt: run `python tools/i18n_audit.py --backend --stats` to list them.
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT, ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import i18n_audit  # noqa: E402
from ui.localization_audit_catalog import AUDIT_CATALOG  # noqa: E402

_HAN = re.compile(r"[\u4e00-\u9fff]")
_KANA = re.compile(r"[\u3040-\u30ff]")
_CJK_RUN = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]+")
# Literal source-text markers the features detect (headings, editor notes, ruby
# markup).  They must stay verbatim in English descriptions.
_LITERAL_MARKERS = frozenset({
    "前書", "前書き", "後書き", "漢字", "よみ", "ルビ",
    "章", "话", "回", "第", "序章", "幕间", "终章", "后记",
})


def _report(findings) -> str:
    lines = [f"{f.path}:{f.line}: {f.text[:120]!r}" for f in findings[:25]]
    if len(findings) > 25:
        lines.append(f"... and {len(findings) - 25} more")
    return "\n".join(lines)


class UiAuditTest(unittest.TestCase):
    def test_ui_has_no_untranslated_cjk_literals(self) -> None:
        findings = i18n_audit.audit(tuple(i18n_audit.DEFAULT_TARGETS))
        self.assertEqual(findings, [], "untranslated UI strings:\n" + _report(findings))


class BackendAuditTest(unittest.TestCase):
    def test_priority_backend_messages_are_translated(self) -> None:
        findings = i18n_audit.audit_backend(("builder", "core", "utils"))
        self.assertEqual(findings, [], "untranslated backend messages:\n" + _report(findings))


class CatalogShapeTest(unittest.TestCase):
    def test_entries_are_well_formed(self) -> None:
        problems: list[str] = []
        for source, value in AUDIT_CATALOG.items():
            if not (isinstance(value, tuple) and len(value) == 2):
                problems.append(f"{source!r}: expected (en, ja) tuple")
                continue
            en, ja = value
            if not en.strip() or not ja.strip():
                problems.append(f"{source!r}: empty translation")
            want = source.count("{…}")
            if en.count("{…}") != want or ja.count("{…}") != want:
                problems.append(f"{source!r}: placeholder count mismatch")
            if source.count("\n") != en.count("\n") or source.count("\n") != ja.count("\n"):
                problems.append(f"{source!r}: line break count mismatch")
            leftover = [run for run in _CJK_RUN.findall(en) if run not in _LITERAL_MARKERS]
            if leftover:
                problems.append(f"{source!r}: CJK left in English text {leftover}")
        self.assertEqual(problems, [], "\n".join(problems[:25]))


if __name__ == "__main__":
    unittest.main()
