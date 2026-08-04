#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Provider 插件系统。

按子模块导入使用（包顶层不做再导出）：
    from ai.provider_factory import create_provider   # 工厂：openai / deepseek /
                                                      # gemini / anthropic /
                                                      # openrouter / ollama / custom
    from ai.config import load_ai_settings
    from ai.base import AIProvider, Suggestion
"""
