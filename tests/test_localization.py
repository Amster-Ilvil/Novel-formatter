"""Qt-free checks for the runtime UI localization layer.

Covers the default-language rule, placeholder parity between source templates
and their translations, CJK leakage into English chrome, and coverage of the
strings introduced by the LTR export, model download and archive import work.
Nothing here imports PySide6; the widget-level smoke lives in test_ui_smoke.py.
"""
from __future__ import annotations

import re
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ui.dialog_localization as dialog_localization  # noqa: E402
import ui.localization as localization  # noqa: E402
import ui.message_catalog as message_catalog  # noqa: E402
from ui.localization import (  # noqa: E402
    DEFAULT_LANGUAGE, LANG_EN, LANG_JA, LANG_ZH, LANGUAGE_LABELS,
    normalize_language, template_placeholders, translate_text,
)
from ui.localization_audit_catalog import AUDIT_CATALOG  # noqa: E402

LANGUAGE_KEY = "reference_ui/language"
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_CJK_RUN = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_NAMED_FIELD = re.compile(r"\{[A-Za-z_]\w*\}")
# Self-identifying language names plus literal source-text markers that the
# formatter features detect; both must stay verbatim in English chrome.
_CJK_ALLOWLIST = frozenset(LANGUAGE_LABELS.values()) | {"繁體中文"} | frozenset({
    "前書", "前書き", "後書き", "漢字", "よみ", "ルビ",
    "章", "话", "回", "第", "序章", "幕间", "终章", "后记",
})


class _FakeSettings:
    """Minimal QSettings stand-in: ``value(key, default)`` over a dict."""

    def __init__(self, data: dict | None = None):
        self._data = dict(data or {})

    def value(self, key, default=None):
        return self._data.get(key, default)

    def setValue(self, key, value):
        self._data[key] = value


def _load_language(settings) -> str:
    # Mirrors SystemSettingsTab: a missing key falls back to DEFAULT_LANGUAGE.
    return normalize_language(str(settings.value(LANGUAGE_KEY, DEFAULT_LANGUAGE)))


def _bag(text: str) -> Counter:
    return Counter(template_placeholders(text))


class DefaultLanguageTest(unittest.TestCase):
    def test_default_is_english(self) -> None:
        self.assertEqual(DEFAULT_LANGUAGE, LANG_EN)

    def test_empty_store_gives_english(self) -> None:
        self.assertEqual(_load_language(_FakeSettings()), LANG_EN)

    def test_saved_choice_is_kept(self) -> None:
        for saved in (LANG_ZH, LANG_JA, LANG_EN):
            store = _FakeSettings({LANGUAGE_KEY: saved})
            self.assertEqual(_load_language(store), saved)
            self.assertEqual(store.value(LANGUAGE_KEY), saved, "loading must not overwrite the stored value")

    def test_unknown_values_fall_back_to_english(self) -> None:
        for value in (None, "", "fr_FR", "zh", "zh_TW", "ja", "garbage"):
            self.assertEqual(normalize_language(value), LANG_EN, repr(value))

    def test_settings_tab_uses_default_language(self) -> None:
        source = (ROOT / "ui" / "settings" / "tab.py").read_text(encoding="utf-8")
        self.assertIn(f'LANGUAGE_KEY = "{LANGUAGE_KEY}"', source)
        self.assertNotIn("LANGUAGE_KEY, LANG_ZH", source)
        self.assertNotIn("currentData() or LANG_ZH", source)
        self.assertNotIn("_language_combo, LANG_ZH", source)

    def test_switcher_offers_self_named_languages(self) -> None:
        source = (ROOT / "ui" / "settings" / "tab.py").read_text(encoding="utf-8")
        order = [source.find(f'addItem("{label}", {const})') for label, const in (
            ("English", "LANG_EN"), ("简体中文", "LANG_ZH"), ("日本語", "LANG_JA"),
        )]
        self.assertTrue(all(pos >= 0 for pos in order), order)
        self.assertEqual(order, sorted(order), "English must be offered first")
        self.assertEqual(LANGUAGE_LABELS.get(LANG_EN), "English")
        self.assertEqual(LANGUAGE_LABELS.get(LANG_JA), "日本語")
        self.assertEqual(LANGUAGE_LABELS.get(LANG_ZH), "简体中文")


class PlaceholderParityTest(unittest.TestCase):
    def _assert_parity(self, name: str, pairs) -> None:
        problems = [
            f"{key!r}: {dict(_bag(key))} != {dict(_bag(value))}"
            for key, value in pairs if _bag(key) != _bag(value)
        ]
        self.assertEqual(problems, [], f"{name} placeholder mismatch:\n" + "\n".join(problems[:25]))

    def test_exact_catalogs(self) -> None:
        self._assert_parity("_EXACT_EN", localization._EXACT_EN.items())
        self._assert_parity("_EXACT_JA", localization._EXACT_JA.items())

    def test_audit_templates(self) -> None:
        self._assert_parity("AUDIT_CATALOG en", ((key, en) for key, (en, _ja) in AUDIT_CATALOG.items()))
        self._assert_parity("AUDIT_CATALOG ja", ((key, ja) for key, (_en, ja) in AUDIT_CATALOG.items()))

    def test_dialog_catalogs(self) -> None:
        for name in ("_EN", "_JA", "_ZH"):
            self._assert_parity(f"dialog_localization.{name}", getattr(dialog_localization, name).items())

    def test_message_catalog(self) -> None:
        problems: list[str] = []
        for key, per_language in message_catalog._MESSAGES.items():
            reference = Counter(_NAMED_FIELD.findall(per_language.get(LANG_ZH, "")))
            for language, template in per_language.items():
                found = Counter(_NAMED_FIELD.findall(template))
                if found != reference:
                    problems.append(f"{key}/{language}: {dict(found)} != {dict(reference)}")
            values = {field.strip("{}"): "X" for field in reference}
            for language in (LANG_EN, LANG_JA, LANG_ZH):
                message_catalog.format_message(key, language, **values)
        self.assertEqual(problems, [], "\n".join(problems))

    def test_template_render_keeps_payload(self) -> None:
        self.assertEqual(translate_text("vol1.zip 等 3 个文件", LANG_EN), "vol1.zip and 3 file(s) in total")
        self.assertIn("vol1.zip", translate_text("vol1.zip 等 3 个文件", LANG_JA))


class EnglishChromeTest(unittest.TestCase):
    def test_english_values_have_no_cjk(self) -> None:
        leaks = []
        for key, value in localization._EXACT_EN.items():
            leftover = [run for run in _CJK_RUN.findall(value) if run not in _CJK_ALLOWLIST]
            if leftover:
                leaks.append(f"{key!r} -> {value!r}: {leftover}")
        self.assertEqual(leaks, [], "\n".join(leaks[:25]))


class FeatureStringCoverageTest(unittest.TestCase):
    # Literal UI strings taken from ui/epub/tab.py (LTR export),
    # ui/common/model_download.py (model download) and ui/pages/import_sources.py.
    LTR = (
        "转换为从左到右横排版式",
        "导出前将竖排转为从左到右横排：横排 CSS、ltr 翻页、竖排标点改为横排字形",
        "阅读方向",
        "横排 · 左开",
        "横排",
        "竖排",
    )
    MODEL_DOWNLOAD = (
        "下载 OCR 模型",
        "正在准备模型下载…",
        "模型已下载并通过校验。",
        "模型下载失败",
        "模型下载已取消。",
        "正在下载模型",
    )
    IMPORT_SUMMARY = ("vol1.cbz 等 4 个文件",)
    # English backend messages (builder/text_direction.py, utils/model_manager.py,
    # adapters/pdf_input.py archive errors, adapters/file_type.py).  The layer has
    # no EN→zh direction: zh returns the source unchanged, so these must pass
    # through byte-for-byte in every UI language.
    ENGLISH_BACKEND = (
        "Unknown text direction mode: 'x'",
        "Unknown model id 'paddle_v6'.",
        "Download of model paddle_v6 was cancelled.",
        "Model paddle_v6 is an archive; place the extracted folder in the models folder instead.",
        "Archive not found: /tmp/book.cbz",
        "No images found in archive: /tmp/book.cbz",
        "Archive extraction cancelled: /tmp/book.cbr",
    )

    def _assert_translated(self, strings) -> None:
        for language in (LANG_EN, LANG_JA):
            missing = [s for s in strings if translate_text(s, language) == s]
            self.assertEqual(missing, [], f"untranslated for {language}")
            if language == LANG_EN:
                leaked = [s for s in strings if _CJK.search(translate_text(s, language))]
                self.assertEqual(leaked, [], "CJK left in English output")

    def test_ltr_strings(self) -> None:
        self._assert_translated(self.LTR)

    def test_model_download_strings(self) -> None:
        self._assert_translated(self.MODEL_DOWNLOAD)

    def test_import_summary_template(self) -> None:
        self._assert_translated(self.IMPORT_SUMMARY)

    def test_ltr_strings_are_used_by_epub_tab(self) -> None:
        source = (ROOT / "ui" / "epub" / "tab.py").read_text(encoding="utf-8")
        for text in self.LTR:
            self.assertIn(f'"{text}"', source)

    def test_model_strings_are_used_by_download_ui(self) -> None:
        source = (ROOT / "ui" / "common" / "model_download.py").read_text(encoding="utf-8")
        for text in self.MODEL_DOWNLOAD:
            self.assertTrue(f'"{text}"' in source or f"'{text}'" in source, text)

    def test_english_backend_messages_pass_through(self) -> None:
        for language in (LANG_EN, LANG_JA, LANG_ZH):
            for text in self.ENGLISH_BACKEND:
                self.assertEqual(translate_text(text, language), text, f"{language}: {text!r}")

    def test_chinese_ui_is_identity(self) -> None:
        for text in self.LTR + self.MODEL_DOWNLOAD:
            self.assertEqual(translate_text(text, LANG_ZH), text)


if __name__ == "__main__":
    unittest.main()
