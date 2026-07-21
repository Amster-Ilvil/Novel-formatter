#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Provider 抽象基类
所有 AI 校正器（OpenAI / DeepSeek / Gemini）继承此类。

子类只需要实现 name 和 _call_llm（怎么把一个 prompt 发给具体的模型 API、
拿到回复文本——各家 SDK 不一样，这是唯一真正因 provider 而异的部分）。
批量切分 / prompt 拼装 / 响应解析这些和"发给哪家模型"无关的逻辑，
都由本基类里的 correct_ocr / reformat / _run_batches 统一实现，
避免每个 provider 子类都各写一遍几乎一样的循环体。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.document import UnifiedDocument, BlockType


@dataclass
class Suggestion:
    """AI 对某个 block 的修正/重排版建议"""
    block_index: int
    original: str
    suggested: str
    confidence: float = 0.0
    explanation: str = ""


TEXT_TYPES = {BlockType.PARAGRAPH, BlockType.DIALOGUE}


class AIProvider(ABC):
    """AI 校正器抽象基类"""

    def __init__(self, api_key: str, model: str = "", **kwargs):
        self.api_key = api_key
        self.model = model
        self.kwargs = kwargs

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def _call_llm(self, prompt: str, temperature: float) -> str:
        """把一个 prompt 发给具体模型 API，返回回复的纯文本。"""
        ...

    # ── 批次骨架（子类共用，不需要各自实现）───────────────────────────────────

    def _collect_text_items(self, doc: UnifiedDocument) -> list[tuple[int, str]]:
        return [
            (i, b.text) for i, b in enumerate(doc.blocks)
            if b.type in TEXT_TYPES and b.text.strip()
        ]

    def _run_batches(
        self,
        doc: UnifiedDocument,
        temperature: float,
        build_prompt_fn: Callable[[list[tuple[int, str]]], str],
        batch_size: int = 20,
        progress_callback: Optional[callable] = None,
    ) -> list[Suggestion]:
        """
        公共批次骨架：切分文本块 → build_prompt_fn 拼 prompt → _call_llm 拿回复
        → 解析成建议列表。correct_ocr 和 reformat 都基于这个骨架，只是各自的
        prompt 拼装方式（build_prompt_fn）不一样。
        """
        text_items = self._collect_text_items(doc)
        all_suggestions: list[Suggestion] = []

        for batch_idx in range(0, len(text_items), batch_size):
            batch = text_items[batch_idx:batch_idx + batch_size]
            prompt = build_prompt_fn(batch)
            reply = self._call_llm(prompt, temperature)
            items = self._parse_response(reply)

            for item in items:
                idx = item.get("index", -1)
                if idx < 0 or idx >= len(doc.blocks):
                    continue
                all_suggestions.append(Suggestion(
                    block_index=idx,
                    original=item.get("original", ""),
                    suggested=item.get("suggested", ""),
                    confidence=float(item.get("confidence", 0.5)),
                    explanation=item.get("explanation", ""),
                ))

            if progress_callback:
                done = min(batch_idx + batch_size, len(text_items))
                progress_callback(done / len(text_items))

        return all_suggestions

    # ── 对外能力 1：OCR 校正 ──────────────────────────────────────────────────

    def correct_ocr(
        self,
        doc: UnifiedDocument,
        prompt_template: str = "",
        temperature: float = 0.3,
        batch_size: int = 20,
        progress_callback: Optional[callable] = None,
    ) -> list[Suggestion]:
        """对文档中的文本 blocks 进行 OCR 错误校正，返回修正建议列表。"""
        return self._run_batches(
            doc, temperature,
            lambda batch: self._build_prompt(batch, prompt_template),
            batch_size, progress_callback,
        )

    # ── 对外能力 2：按 Format Profile 重新排版 ────────────────────────────────

    def reformat(
        self,
        doc: UnifiedDocument,
        profile,
        prompt_template: str = "",
        temperature: float = 0.3,
        batch_size: int = 20,
        progress_callback: Optional[callable] = None,
    ) -> list[Suggestion]:
        """
        按 Format Profile 描述的排版风格重新排版正文——只调整空白/分段/对话
        换行等排版，不改动措辞（呼应 SDS"AI 不能直接修改语义"的约束）。
        profile: models.format_profile.FormatProfile
        """
        return self._run_batches(
            doc, temperature,
            lambda batch: self._build_reformat_prompt(batch, profile, prompt_template),
            batch_size, progress_callback,
        )

    # ── prompt 拼装 ───────────────────────────────────────────────────────────

    def _build_prompt(self, texts: list[tuple[int, str]], template: str) -> str:
        """构建发送给 LLM 的 OCR 校正 prompt"""
        lines = []
        for idx, text in texts:
            lines.append(f"[{idx}] {text}")
        text_block = "\n".join(lines)
        if template:
            return template.replace("{{TEXT}}", text_block)
        return (
            "以下是日本语ライトノベルのOCR結果です。\n"
            "誤字脱字、文字化け、不自然な改行を修正してください。\n"
            "修正がある行のみ、JSON配列で返してください。\n"
            "各要素: {\"index\": 行番号, \"original\": \"元テキスト\", "
            "\"suggested\": \"修正後テキスト\", \"confidence\": 0.0-1.0, "
            "\"explanation\": \"修正理由\"}\n"
            "修正不要の場合は空配列 [] を返してください。\n\n"
            f"{text_block}"
        )

    def _build_reformat_prompt(self, texts: list[tuple[int, str]], profile, template: str) -> str:
        """构建发送给 LLM 的重新排版 prompt"""
        lines = []
        for idx, text in texts:
            lines.append(f"[{idx}] {text}")
        text_block = "\n".join(lines)
        if template:
            return (template
                    .replace("{{TEXT}}", text_block)
                    .replace("{{STYLE_NOTES}}", getattr(profile, "notes", "") or ""))
        style_notes = getattr(profile, "notes", "") or "无特殊要求，保持常见轻小说排版习惯即可。"
        return (
            "下面是需要重新排版的正文段落，请严格按照给定的目标排版风格调整——"
            "只调整空白、分段、对话换行、缩进这类排版细节，绝对不能改变原文的"
            "措辞、增删内容或改变语义。\n\n"
            f"目标排版风格: {style_notes}\n\n"
            "只对需要调整格式的行返回结果，用 JSON 数组表示。\n"
            "每个元素: {\"index\": 行号, \"original\": \"原文\", "
            "\"suggested\": \"重排版后的文本\", \"confidence\": 0.0-1.0, "
            "\"explanation\": \"调整说明\"}\n"
            "不需要调整就返回空数组 []。\n\n"
            f"{text_block}"
        )

    def _parse_response(self, response_text: str) -> list[dict]:
        """鲁棒解析 AI 返回结果：JSON / Markdown JSON / 普通文本。"""
        import json, re

        if not response_text:
            return []

        text = str(response_text).strip()

        # 去除 markdown code fence
        md_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if md_match:
            text = md_match.group(1).strip()

        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                for key in ("suggestions", "items", "results"):
                    if isinstance(result.get(key), list):
                        return result[key]
                if "text" in result:
                    return [{"index": -1, "suggested": str(result["text"])}]
        except Exception:
            pass

        arr = re.search(r"\[\s*\{.*?\}\s*\]", text, re.DOTALL)
        if arr:
            try:
                return json.loads(arr.group(0))
            except Exception:
                pass

        # 最后保留模型输出，避免成功调用被静默丢弃
        return [{"index": -1, "suggested": text}]
