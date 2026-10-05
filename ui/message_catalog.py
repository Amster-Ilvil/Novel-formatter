# -*- coding: utf-8 -*-
"""Stable message-key catalog for runtime-composed UI text.

The catalog is intentionally UI-only.  Dynamic payload values (paths, model
names, counts, user profile names, hashes, exception details) are interpolated
verbatim after the localized template is selected, so localization can never
rewrite business identifiers or document content.
"""
from __future__ import annotations

from collections.abc import Mapping

from ui.localization import LANG_EN, LANG_JA, LANG_ZH, normalize_language


_MESSAGES: dict[str, Mapping[str, str]] = {
    "page.delete.confirm": {
        LANG_ZH: "确定要从当前导入中移除选中的 {count} 页吗？\n（只是从这次处理里去掉，不会删除原始图片文件）",
        LANG_EN: "Remove the selected {count} page(s) from this import?\n(They are removed only from this run; the original image files are not deleted.)",
        LANG_JA: "現在の読み込みから選択した {count} ページを外しますか？\n（今回の処理対象から外すだけで、元画像ファイルは削除しません）",
    },
    "formatter.import_docx.summary": {
        LANG_ZH: "从 DOCX 导入 {blocks} 个块，{chapters} 个章节\n可直接勾选步骤运行后处理。",
        LANG_EN: "Imported {blocks} blocks and {chapters} chapters from DOCX.\nYou can select steps and run post-processing directly.",
        LANG_JA: "DOCXから {blocks} ブロック、{chapters} 章を読み込みました。\nそのまま手順を選択して後処理を実行できます。",
    },
    "formatter.import_epub.summary": {
        LANG_ZH: "从 EPUB 导入 {blocks} 个块，{toc} 个目录条目\n勾选「前后书剥离」等步骤运行后处理，可清理残留的样板文字。",
        LANG_EN: "Imported {blocks} blocks and {toc} TOC entries from EPUB.\nSelect steps such as “Remove Front/Back Matter” and run post-processing to clean remaining boilerplate text.",
        LANG_JA: "EPUBから {blocks} ブロック、目次 {toc} 件を読み込みました。\n「前後付録の除去」などを選択して後処理を実行すると、残った定型文を整理できます。",
    },
    "common.saved_path": {
        LANG_ZH: "已保存：\n{path}",
        LANG_EN: "Saved:\n{path}",
        LANG_JA: "保存しました：\n{path}",
    },
    "format.delete.confirm": {
        LANG_ZH: "确定删除格式「{name}」？此操作不可撤销。",
        LANG_EN: "Delete format “{name}”? This cannot be undone.",
        LANG_JA: "形式「{name}」を削除しますか？この操作は取り消せません。",
    },
    "version.clear.confirm": {
        LANG_ZH: "确定清空「{name}」？其他两个版本不会受影响。",
        LANG_EN: "Clear “{name}”? The other two versions will not be affected.",
        LANG_JA: "「{name}」をクリアしますか？他の2バージョンには影響しません。",
    },
    "handwriting.empty_columns.confirm": {
        LANG_ZH: "还有 {count} 列 OCR 三次均为空，尚未人工输入。\n\n继续后会保留醒目的 □ 标记，不会静默删除这些列；建议返回逐列补全。\n\n仍要应用当前结果吗？",
        LANG_EN: "{count} column(s) are still empty after three OCR attempts and have not been entered manually.\n\nContinuing will keep conspicuous □ markers and will not silently delete these columns; returning to complete them column by column is recommended.\n\nApply the current result anyway?",
        LANG_JA: "{count} 列が3回OCRしても空のままで、まだ手動入力されていません。\n\n続行すると目立つ □ マークを保持し、これらの列を黙って削除しません。戻って列ごとに補完することを推奨します。\n\nそれでも現在の結果を適用しますか？",
    },
    "clear.workspace.confirm": {
        LANG_ZH: "确定清空当前{name}工作区？\n\n磁盘上的原始文件不会被删除。",
        LANG_EN: "Clear the current {name} workspace?\n\nOriginal files on disk will not be deleted.",
        LANG_JA: "現在の{name}ワークスペースをクリアしますか？\n\nディスク上の元ファイルは削除されません。",
    },
    "clear.workspace.failed": {
        LANG_ZH: "{name}工作区未能完全清空，请查看终端日志。",
        LANG_EN: "The {name} workspace could not be fully cleared. Check the terminal log.",
        LANG_JA: "{name}ワークスペースを完全にクリアできませんでした。ターミナルログを確認してください。",
    },
    "model_update.confirm.body": {
        LANG_ZH: "模型：{label}\n本地版本：{local}\n目标版本：{target}\n\n本操作只在本次确认后执行。更新会先下载和校验，再替换模型；NDLOCR-Lite 与 48px AR 会保留一个本地回退备份。\n\n继续吗？",
        LANG_EN: "Model: {label}\nLocal version: {local}\nTarget version: {target}\n\nThis action runs only after this confirmation. The update is downloaded and verified before replacement; NDLOCR-Lite and 48px AR keep one local rollback backup.\n\nContinue?",
        LANG_JA: "モデル：{label}\nローカル版：{local}\n対象版：{target}\n\nこの操作は今回の確認後にのみ実行します。更新はダウンロードと検証を行ってからモデルを置換し、NDLOCR-Lite と 48px AR はローカルのロールバック用バックアップを1つ保持します。\n\n続行しますか？",
    },
    "app_update.confirm.body": {
        LANG_ZH: "将从 {repository} / {branch} 更新程序代码。\n\n本地：v{local}\n远端：v{remote} · {remote_short}\n\n不会更新 OCR 模型，也不会删除虚拟环境、模型缓存、书籍、输出和日志。更新完成后需要重启程序。\n\n继续吗？",
        LANG_EN: "Application source will be updated from {repository} / {branch}.\n\nLocal: v{local}\nRemote: v{remote} · {remote_short}\n\nOCR models will not be updated, and virtual environments, model caches, books, outputs and logs will not be deleted. Restart the application after the update completes.\n\nContinue?",
        LANG_JA: "{repository} / {branch} からアプリのソースを更新します。\n\nローカル：v{local}\nリモート：v{remote} · {remote_short}\n\nOCRモデルは更新せず、仮想環境・モデルキャッシュ・書籍・出力・ログも削除しません。更新完了後はアプリの再起動が必要です。\n\n続行しますか？",
    },
    "app_update.complete": {
        LANG_ZH: "程序源码已安全更新到 v{version}。\n\n请关闭并重新启动 Novel Formatter。",
        LANG_EN: "Application source was safely updated to v{version}.\n\nClose and restart Novel Formatter.",
        LANG_JA: "アプリのソースを安全に v{version} へ更新しました。\n\nNovel Formatter を終了して再起動してください。",
    },
    "app_update.complete_dependencies": {
        LANG_ZH: "程序源码已安全更新到 v{version}。\n\n检测到 requirements 依赖清单变化。为避免更新过程擅自修改 Python/OCR 环境，本功能不会自动 pip install；请按项目安装说明更新主依赖后再启动。\n\n请关闭并重新启动 Novel Formatter。",
        LANG_EN: "Application source was safely updated to v{version}.\n\nThe requirements dependency list changed. To avoid modifying the Python/OCR environment during source update, this feature does not run pip install automatically; update the main dependencies according to the project installation instructions before restarting.\n\nClose and restart Novel Formatter.",
        LANG_JA: "アプリのソースを安全に v{version} へ更新しました。\n\nrequirements の依存関係一覧に変更があります。ソース更新中に Python/OCR 環境を勝手に変更しないため、この機能は pip install を自動実行しません。プロジェクトのインストール手順に従って主要依存関係を更新してから再起動してください。\n\nNovel Formatter を終了して再起動してください。",
    },
}


def format_message(key: str, language: str = LANG_ZH, /, **values: object) -> str:
    """Format a localized message by stable key.

    All values are interpolated *after* language selection.  Missing keys or
    missing placeholders fail loudly in development instead of silently
    returning a partially localized message.
    """
    message_key = str(key or "").strip()
    if message_key not in _MESSAGES:
        raise KeyError(f"Unknown UI message key: {message_key}")
    language = normalize_language(language)
    templates = _MESSAGES[message_key]
    template = templates.get(language) or templates[LANG_ZH]
    return template.format_map(dict(values))


def message_keys() -> tuple[str, ...]:
    return tuple(sorted(_MESSAGES))


__all__ = ["format_message", "message_keys"]
