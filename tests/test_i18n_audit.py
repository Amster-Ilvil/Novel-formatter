"""Regression gate for the UI/backend localization audit.

UI text and backend messages from builder/, core/ and utils/ must all be
covered by the runtime translation layer.  In adapters/ and engine/ every CJK
literal is classified as user-visible (A) or internal (B); only A must be
translated.  Run `python tools/i18n_audit.py --backend-visible --stats` for
the A/B breakdown by reason.
"""
from __future__ import annotations

import ast
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT, ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import i18n_audit  # noqa: E402
from ui.localization_audit_catalog import AUDIT_CATALOG, BACKEND_EN  # noqa: E402

_HAN = re.compile(r"[\u4e00-\u9fff]")
_KANA = re.compile(r"[\u3040-\u30ff]")
_CJK_RUN = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]+")
# Literal source-text markers the features detect (headings, editor notes, ruby
# markup).  They must stay verbatim in English descriptions.
_LITERAL_MARKERS = frozenset({
    "前書", "前書き", "後書き", "漢字", "よみ", "ルビ",
    "章", "话", "回", "第", "序章", "幕间", "终章", "后记",
    "削除予定",
})


def _report(findings) -> str:
    lines = [f"{f.path}:{f.line}: {f.text[:120]!r}" for f in findings[:25]]
    if len(findings) > 25:
        lines.append(f"... and {len(findings) - 25} more")
    return "\n".join(lines)


class LiteralCollectionTest(unittest.TestCase):
    def collect(self, source: str) -> list[tuple[str, str]]:
        collector = i18n_audit._Collector()
        collector.visit(ast.parse(source))
        return [(raw, sample) for _line, raw, sample in collector.items]

    def test_mixed_cjk_and_latin_literal_is_collected(self) -> None:
        self.assertEqual(
            self.collect('label.setText("OCR processing status: 等待")'),
            [("OCR processing status: 等待", "OCR processing status: 等待")],
        )

    def test_fstring_collects_chrome_without_payload_keys(self) -> None:
        self.assertEqual(
            self.collect("""label.setText(f"OCR 正文：{book['正文']} · {path}")"""),
            [("OCR 正文：{…} · {…}", "OCR 正文：1 · 1")],
        )

    def test_concatenation_is_collected_as_one_template(self) -> None:
        self.assertEqual(
            self.collect("""label.setText("OCR 正文：" + book["正文"] + f" · 共 {count} 页")"""),
            [("OCR 正文：{…} · 共 {…} 页", "OCR 正文：1 · 共 1 页")],
        )

    def test_concatenation_keeps_conditional_literal_branches(self) -> None:
        items = self.collect(
            """label.setText("OCR 状态：" + ("等待" if pending else "完成"))"""
        )
        raw_texts = {raw for raw, _sample in items}
        self.assertIn("等待", raw_texts)
        self.assertIn("完成", raw_texts)

    def test_non_display_contexts_are_skipped(self) -> None:
        self.assertEqual(
            self.collect(
                """widget.setObjectName("正文标识")
settings.setValue("正文路径", "书籍/正文.txt")
print(f"内部调试：{path}")
if kind == "正文":
    pass
"""
            ),
            [],
        )


class TranslationCompletenessTest(unittest.TestCase):
    def test_partial_translation_does_not_pass_ui_audit(self) -> None:
        with patch("ui.dialog_localization.translate_dialog_text", return_value="OCR status: 等待"):
            with patch("ui.localization.translate_text", return_value="OCR status: 等待"):
                check = i18n_audit._translators()
                self.assertFalse(check("OCR 状态：等待", "OCR 状态：等待"))

    def test_complete_translation_passes_ui_audit(self) -> None:
        with patch("ui.dialog_localization.translate_dialog_text", side_effect=lambda text, language: text):
            with patch("ui.localization.translate_text", return_value="OCR status: waiting"):
                check = i18n_audit._translators()
                self.assertTrue(check("OCR 状态：等待", "OCR 状态：等待"))


    def test_known_book_syntax_is_not_untranslated_chrome(self) -> None:
        source = "恢复｜漢字《よみ》ruby标注"
        self.assertTrue(i18n_audit._translators()(source, source))

    def test_book_syntax_exemption_does_not_hide_untranslated_chrome(self) -> None:
        source = "恢复｜漢字《よみ》ruby标注"
        rendered = "Restore ｜漢字《よみ》 标注"
        with patch("ui.dialog_localization.translate_dialog_text", return_value=rendered):
            with patch("ui.localization.translate_text", return_value=rendered):
                self.assertFalse(i18n_audit._translators()(source, source))

    def test_book_syntax_exemption_is_limited_to_known_descriptions(self) -> None:
        source = "新的｜漢字《よみ》提示"
        rendered = "New ｜漢字《よみ》 hint"
        with patch("ui.dialog_localization.translate_dialog_text", return_value=rendered):
            with patch("ui.localization.translate_text", return_value=rendered):
                self.assertFalse(i18n_audit._translators()(source, source))


class UiAuditTest(unittest.TestCase):
    def test_ui_has_no_untranslated_cjk_literals(self) -> None:
        findings = i18n_audit.audit(tuple(i18n_audit.DEFAULT_TARGETS))
        self.assertEqual(findings, [], "untranslated UI strings:\n" + _report(findings))


class BackendAuditTest(unittest.TestCase):
    def test_priority_backend_messages_are_translated(self) -> None:
        findings = i18n_audit.audit_backend(("builder", "core", "utils"))
        self.assertEqual(findings, [], "untranslated backend messages:\n" + _report(findings))

    def test_visible_adapter_and_engine_messages_are_translated(self) -> None:
        findings = i18n_audit.audit_backend_visible(i18n_audit.VISIBLE_BACKEND_DIRS)
        self.assertEqual(findings, [], "untranslated category-A backend messages:\n" + _report(findings))


_CLASSIFIER_SAMPLE = '''
import logging
import re

logger = logging.getLogger(__name__)
TOKENS = {"前書き": 1}


def run(text, progress_callback=None, log_callback=None):
    """识别整页文本。"""
    if text == "后记":
        return None
    pattern = re.compile(r"第[一二三]+章")
    print("调试输出")
    logger.debug("内部调试")
    progress_callback("正在识别页面")
    log_callback(f"已完成 {text} 页")
    raise RuntimeError("模型加载失败")
'''


class BackendClassifierTest(unittest.TestCase):
    def test_rules_on_known_sample(self) -> None:
        tree = ast.parse(_CLASSIFIER_SAMPLE)
        ctx = i18n_audit._FileContext(tree)
        got = {
            raw: i18n_audit._classify_literal(node, ctx)
            for node, raw, _ in i18n_audit._iter_backend_literals(tree, ctx)
        }
        got = {raw: category for raw, (category, _reason) in got.items()}
        A, B = i18n_audit.VISIBLE, i18n_audit.INTERNAL
        expected = {
            "前書き": B,          # dict key
            "识别整页文本。": B,    # docstring
            "后记": B,            # comparison operand
            "第[一二三]+章": B,     # regular expression
            "调试输出": B,         # print
            "内部调试": B,         # stdlib logging
            "正在识别页面": A,      # progress callback
            "已完成 {…} 页": A,    # log callback f-string
            "模型加载失败": A,      # raise
        }
        self.assertEqual(got, expected)

    def test_every_literal_has_a_category(self) -> None:
        items = i18n_audit.classify_backend(i18n_audit.VISIBLE_BACKEND_DIRS)
        self.assertTrue(items)
        self.assertEqual({item.category for item in items}, {i18n_audit.VISIBLE, i18n_audit.INTERNAL})
        self.assertTrue(all(item.reason for item in items))


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

    def test_backend_entries_are_well_formed(self) -> None:
        problems: list[str] = []
        for source, english in BACKEND_EN.items():
            if not isinstance(english, str) or not english.strip():
                problems.append(f"{source!r}: empty translation")
                continue
            if english.count("{…}") != source.count("{…}"):
                problems.append(f"{source!r}: placeholder count mismatch")
            if source.count("\n") != english.count("\n"):
                problems.append(f"{source!r}: line break count mismatch")
            leftover = [run for run in _CJK_RUN.findall(english) if run not in _LITERAL_MARKERS]
            if leftover:
                problems.append(f"{source!r}: CJK left in English text {leftover}")
        self.assertEqual(problems, [], "\n".join(problems[:25]))


if __name__ == "__main__":
    unittest.main()
