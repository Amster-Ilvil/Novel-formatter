# -*- coding: utf-8 -*-
"""AI OCR correction pipeline step."""

import os
from pathlib import Path

from models.document import UnifiedDocument


def ai_correction_step(doc: UnifiedDocument):
    """可选 AI 校正步骤。未配置时安全跳过。"""
    if os.environ.get("NOVEL_FORMATTER_AI_ENABLED", "0") != "1":
        return doc

    try:
        from ai.openai_provider import OpenAIProvider
        from ai.deepseek_provider import DeepSeekProvider
        from ai.gemini_provider import GeminiProvider

        provider_name = os.environ.get("NOVEL_FORMATTER_AI_PROVIDER", "openai")
        key = os.environ.get("NOVEL_FORMATTER_AI_KEY", "")
        model = os.environ.get("NOVEL_FORMATTER_AI_MODEL", "")

        providers = {
            "openai": OpenAIProvider,
            "deepseek": DeepSeekProvider,
            "gemini": GeminiProvider,
        }
        cls = providers.get(provider_name, OpenAIProvider)
        if not key:
            return doc

        provider = cls(key, model=model)
        suggestions = provider.correct_ocr(doc)

        for item in suggestions:
            idx = item.block_index
            if idx >= 0 and idx < len(doc.blocks) and item.suggested.strip():
                doc.blocks[idx].text = item.suggested
                doc.blocks[idx].modified_by = "ai_correction"

        doc.add_log("ai_correction", f"AI applied {len(suggestions)} suggestions", len(suggestions))
    except Exception as e:
        doc.add_log("ai_correction", f"AI skipped: {e}", 0)

    return doc


class AIFormatterStep:
    """GUI 可调用 AI OCR 校正封装。"""
    def apply(self, document: UnifiedDocument) -> UnifiedDocument:
        result = ai_correction_step(document)
        for block in getattr(result, "blocks", []):
            if getattr(block, "modified_by", "") == "ai_correction":
                block.ai_modified = True
        return result
