# -*- coding: utf-8 -*-
"""AI OCR correction pipeline step."""

import os
import copy
from pathlib import Path

from models.document import UnifiedDocument


def ai_correction_step(doc: UnifiedDocument):
    """可选 AI 校正步骤。始终返回独立文档；未配置时安全跳过并记录原因。"""
    result = copy.deepcopy(doc)
    if os.environ.get("NOVEL_FORMATTER_AI_ENABLED", "0") != "1":
        result.add_log("ai_correction", "AI 未启用，跳过校正", 0)
        return result

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
            result.add_log("ai_correction", "AI 已启用但未配置 API Key，跳过校正", 0)
            return result

        provider = cls(key, model=model)
        suggestions = provider.correct_ocr(result)

        applied = 0
        for item in suggestions:
            idx = item.block_index
            if idx >= 0 and idx < len(result.blocks) and item.suggested.strip():
                result.blocks[idx].text = item.suggested
                result.blocks[idx].modified_by = "ai_correction"
                applied += 1

        result.add_log("ai_correction", f"AI applied {applied} suggestions", applied)
    except Exception as e:
        result.add_log("ai_correction", f"AI skipped: {e}", 0)

    return result



class AIFormatterStep:
    """GUI 可调用 AI OCR 校正封装。"""
    def apply(self, document: UnifiedDocument) -> UnifiedDocument:
        result = ai_correction_step(document)
        for block in getattr(result, "blocks", []):
            if getattr(block, "modified_by", "") == "ai_correction":
                block.ai_modified = True
        return result
