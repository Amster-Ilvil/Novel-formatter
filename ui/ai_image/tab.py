from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QLabel, QPushButton, QCheckBox,
    QPlainTextEdit, QTextEdit, QProgressBar, QSplitter, QMessageBox,
)
from PySide6.QtCore import Qt, Signal, QTimer, QSettings

from models.document import UnifiedDocument, BlockType
from ui.dialogs import show_error_dialog
from ui.common.editor_controls import NoWheelComboBox
from ui.common.signals import WorkerSignals
from ui.common.toast import notify
from ui.common.styling import MUTED, LIGHT_PREVIEW_STYLE, accent_button, wrap_in_card
from ui.settings.ai_dialog import AISettingsDialog, ensure_ai_settings

class AIImageProcessingTab(QWidget):
    """Page-Manager-driven multimodal extraction/translation workspace.

    The model receives only pages admitted by Page Manager.  It returns already
    structured book text, so this path intentionally does not auto-run the
    traditional OCR comparison or Formatter pipeline.  EPUB handoff remains
    deterministic and local.
    """

    documents_ready = Signal(object, object)
    epub_requested = Signal(object, str)
    page_manager_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._page_images: list[str] = []
        self._page_types: dict[int, str] = {}
        self._source_doc: UnifiedDocument | None = None
        self._translated_doc: UnifiedDocument | None = None
        self._cancel_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._run_generation = 0
        self._busy = False
        self._signals = None
        self._context_signature = None
        self._run_started_at = 0.0
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._update_elapsed)
        # Large books can emit log/progress events from multiple concurrent AI
        # batches in short bursts.  Reuse the OCR log coalescer so GUI document
        # updates are bounded while every audit line remains preserved.
        from core.ocr_runtime_optimizer import CoalescedLineBuffer
        self._ai_log_buffer = CoalescedLineBuffer(max_lines=200000)
        self._ai_log_flush_timer = QTimer(self)
        self._ai_log_flush_timer.setInterval(80)
        self._ai_log_flush_timer.timeout.connect(self._flush_ai_log_buffer)
        self._last_progress_ui = 0.0
        self._settings_signature = None
        self._text_probe_state = "未测试"
        self._vision_probe_state = "未测试"
        self._text_probe_detail = ""
        self._vision_probe_detail = ""
        self._ui_settings = QSettings("NovelFormatter", "NovelFormatter")
        self._build()
        self._refresh_settings_summary()
        self._refresh_preset_summary()

    def _build(self):
        root = wrap_in_card(self)
        outer = QWidget()
        layout = QVBoxLayout(outer)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(10)

        top = QHBoxLayout()
        title = QLabel("AI 图文处理")
        title.setStyleSheet("font-size:15px;font-weight:700;")
        top.addWidget(title)
        subtitle = QLabel("独立模式 · 读取页面管理 · AI直接完成识别/结构/可选翻译 · 本地生成EPUB")
        subtitle.setStyleSheet(f"color:{MUTED};font-size:11px;")
        top.addWidget(subtitle)
        top.addStretch(1)
        page_btn = QPushButton("打开页面管理")
        page_btn.clicked.connect(self.page_manager_requested.emit)
        top.addWidget(page_btn)
        settings_btn = QPushButton("⚙ AI设置")
        settings_btn.clicked.connect(self._open_ai_settings)
        top.addWidget(settings_btn)
        layout.addLayout(top)

        source = QFrame(); source.setObjectName("settingsCard")
        source_wrap = QVBoxLayout(source); source_wrap.setContentsMargins(14, 10, 14, 10); source_wrap.setSpacing(6)
        source_top = QHBoxLayout()
        self._source_label = QLabel("页面管理：尚未导入页面")
        self._source_label.setWordWrap(True)
        source_top.addWidget(self._source_label, 1)
        self._test_connection_btn = QPushButton("测试连接")
        self._test_connection_btn.setToolTip("只验证当前 Provider / API Key / Base URL / 文本路由是否可达；不验证图片输入。")
        self._test_connection_btn.clicked.connect(self._test_connection)
        source_top.addWidget(self._test_connection_btn)
        self._test_btn = QPushButton("测试图片能力")
        self._test_btn.setToolTip("使用当前页面中的一张缩小图片发起一次真实多模态请求；这是正式跑书前最重要的测试。")
        self._test_btn.clicked.connect(self._test_model)
        source_top.addWidget(self._test_btn)
        source_wrap.addLayout(source_top)
        self._settings_label = QLabel("")
        self._settings_label.setWordWrap(True)
        self._settings_label.setStyleSheet(f"color:{MUTED};font-size:11px;")
        source_wrap.addWidget(self._settings_label)
        self._provider_status_label = QLabel("")
        self._provider_status_label.setWordWrap(True)
        self._provider_status_label.setStyleSheet(f"color:{MUTED};font-size:11px;")
        source_wrap.addWidget(self._provider_status_label)
        self._recommended_model_label = QLabel("")
        self._recommended_model_label.setWordWrap(True)
        self._recommended_model_label.setStyleSheet(f"color:{MUTED};font-size:11px;")
        source_wrap.addWidget(self._recommended_model_label)
        self._capability_label = QLabel("")
        self._capability_label.setWordWrap(True)
        self._capability_label.setStyleSheet(f"color:{MUTED};font-size:11px;")
        source_wrap.addWidget(self._capability_label)
        self._practice_label = QLabel("")
        self._practice_label.setWordWrap(True)
        self._practice_label.setStyleSheet(f"color:{MUTED};font-size:11px;")
        source_wrap.addWidget(self._practice_label)
        self._last_test_label = QLabel("")
        self._last_test_label.setWordWrap(True)
        self._last_test_label.setStyleSheet(f"color:{MUTED};font-size:11px;")
        source_wrap.addWidget(self._last_test_label)
        self._test_hint_label = QLabel("建议顺序：先点“测试连接”，再点“测试图片能力”，最后再正式处理整本书。")
        self._test_hint_label.setWordWrap(True)
        self._test_hint_label.setStyleSheet(f"color:{MUTED};font-size:11px;")
        source_wrap.addWidget(self._test_hint_label)
        layout.addWidget(source)

        options = QFrame(); options.setObjectName("settingsCard")
        grid = QGridLayout(options); grid.setContentsMargins(14, 12, 14, 12); grid.setHorizontalSpacing(14); grid.setVerticalSpacing(9)
        grid.addWidget(QLabel("处理任务"), 0, 0)
        self._mode = NoWheelComboBox()
        self._mode.addItem("仅提取原文", "extract")
        self._mode.addItem("提取原文 + 翻译", "extract_translate")
        self._mode.currentIndexChanged.connect(self._mode_changed)
        grid.addWidget(self._mode, 0, 1)

        grid.addWidget(QLabel("目标语言"), 0, 2)
        self._target = NoWheelComboBox()
        self._target.addItem("简体中文", "zh-Hans")
        self._target.addItem("繁體中文", "zh-Hant")
        self._target.addItem("English", "en")
        self._target.setEnabled(False)
        grid.addWidget(self._target, 0, 3)

        grid.addWidget(QLabel("质量 / Token"), 1, 0)
        self._quality = NoWheelComboBox()
        self._quality.addItem("均衡（推荐 · 1920px + 疑难局部复核）", "balanced")
        self._quality.addItem("快速（1600px · 不自动二审）", "fast")
        self._quality.addItem("出版（2560px + 高分辨率局部复核）", "publication")
        grid.addWidget(self._quality, 1, 1, 1, 3)

        self._trim_cb = QCheckBox("完整原页直送 API（推荐 · 防最右列/页顶漏字）")
        self._trim_cb.setChecked(True)
        self._trim_cb.setToolTip("开启后首轮每次只发送一个完整物理页面；若原图尺寸已合适则直接发送原始 PNG/JPEG/WebP，不做裁白边或二次压缩。")
        grid.addWidget(self._trim_cb, 2, 0, 1, 2)
        self._review_cb = QCheckBox("只对 AI 主动标记的不确定区域进行二次放大复核")
        self._review_cb.setChecked(True)
        grid.addWidget(self._review_cb, 2, 2, 1, 2)

        self._fastest_btn = QPushButton("一键最快")
        self._fastest_btn.setToolTip("切换为仅提取原文、1600px、完整原页直送 API 且不自动二审；GLM-5.3 Flash 默认 low 思考")
        self._fastest_btn.clicked.connect(self._apply_fastest_preset)
        grid.addWidget(self._fastest_btn, 3, 0)

        self._translate_fast_btn = QPushButton("一键提取并翻译")
        self._translate_fast_btn.setToolTip(
            "切换为提取原文 + 翻译、简体中文、均衡结构复核、完整原页直送 API；"
            "优先保留原书自然段和独立对白，GLM-5.3 Flash 默认 low 思考"
        )
        self._translate_fast_btn.clicked.connect(self._apply_extract_translate_preset)
        grid.addWidget(self._translate_fast_btn, 3, 1)

        self._preset_summary = QLabel("")
        self._preset_summary.setWordWrap(True)
        self._preset_summary.setStyleSheet(f"color:{MUTED};font-size:11px;")
        grid.addWidget(self._preset_summary, 3, 2, 1, 2)

        note = QLabel(
            "页面顺序、封面、插图和非正文页由页面管理决定；AI不重新分类。"
            "推荐开启“完整原页直送 API”：首轮一页一请求，让多模态模型自己处理纵排阅读顺序、段落、对白和翻译，程序只做校验，不先裁正文。"
            "均衡/出版模式仅对空页、异常短页、段落压平、页顶/最右列风险和疑难局部追加定向核验；跨页接续和补译优先走纯文本。"
        )
        note.setWordWrap(True); note.setStyleSheet(f"color:{MUTED};font-size:11px;")
        grid.addWidget(note, 4, 0, 1, 4)
        layout.addWidget(options)

        action = QHBoxLayout()
        self._start_btn = accent_button("开始 AI 图文处理")
        self._start_btn.clicked.connect(self._start)
        action.addWidget(self._start_btn)
        self._stop_btn = QPushButton("停止")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop)
        action.addWidget(self._stop_btn)
        action.addStretch(1)
        self._source_epub_btn = QPushButton("原文 → EPUB")
        self._source_epub_btn.setEnabled(False)
        self._source_epub_btn.clicked.connect(lambda: self._emit_epub(self._source_doc, "ai_image_source"))
        action.addWidget(self._source_epub_btn)
        self._translation_epub_btn = accent_button("译文 → EPUB", color="#7C3AED")
        self._translation_epub_btn.setEnabled(False)
        self._translation_epub_btn.clicked.connect(lambda: self._emit_epub(self._translated_doc, "ai_image_translation"))
        action.addWidget(self._translation_epub_btn)
        layout.addLayout(action)

        progress_row = QHBoxLayout()
        self._progress = QProgressBar(); self._progress.setRange(0, 100); self._progress.setValue(0)
        progress_row.addWidget(self._progress, 1)
        self._elapsed_label = QLabel("")
        self._elapsed_label.setStyleSheet(f"color:{MUTED};font-size:11px;")
        progress_row.addWidget(self._elapsed_label)
        layout.addLayout(progress_row)

        split = QSplitter(Qt.Horizontal)
        left = QWidget(); ll = QVBoxLayout(left); ll.setContentsMargins(0,0,0,0); ll.setSpacing(6)
        status_title = QLabel("处理状态"); status_title.setStyleSheet("font-weight:650;")
        ll.addWidget(status_title)
        self._status = QLabel("等待页面管理输入")
        self._status.setWordWrap(True); self._status.setStyleSheet(f"color:{MUTED};")
        ll.addWidget(self._status)
        self._log = QPlainTextEdit(); self._log.setReadOnly(True); self._log.setStyleSheet(LIGHT_PREVIEW_STYLE)
        ll.addWidget(self._log, 1)
        split.addWidget(left)

        right = QWidget(); rl = QVBoxLayout(right); rl.setContentsMargins(0,0,0,0); rl.setSpacing(6)
        preview_title = QLabel("AI 成稿预览"); preview_title.setStyleSheet("font-weight:650;")
        rl.addWidget(preview_title)
        self._preview = QPlainTextEdit(); self._preview.setReadOnly(True); self._preview.setStyleSheet(LIGHT_PREVIEW_STYLE)
        self._preview.setPlaceholderText("完成后显示前若干段原文/译文；完整内容会作为 UnifiedDocument 交给 EPUB Builder。")
        rl.addWidget(self._preview, 1)
        split.addWidget(right)
        split.setStretchFactor(0, 1); split.setStretchFactor(1, 1)
        layout.addWidget(split, 1)
        self._mode.currentIndexChanged.connect(self._refresh_preset_summary)
        self._target.currentIndexChanged.connect(self._refresh_preset_summary)
        self._quality.currentIndexChanged.connect(self._refresh_preset_summary)
        self._trim_cb.toggled.connect(self._refresh_preset_summary)
        self._review_cb.toggled.connect(self._refresh_preset_summary)
        root.addWidget(outer, 1)

    def _mode_changed(self):
        self._target.setEnabled(self._mode.currentData() == "extract_translate")
        self._refresh_preset_summary()

    @staticmethod
    def _probe_state_badge(state: str) -> str:
        mapping = {"未测试": "○ 未测试", "通过": "✓ 已通过", "失败": "✗ 失败"}
        return mapping.get(str(state or ""), str(state or "未测试"))

    @staticmethod
    def _provider_route_hint(provider: str, base_url: str) -> str:
        provider = str(provider or "").strip().lower()
        base_url = str(base_url or "").strip()
        if provider == "zhipu":
            return f"当前 Provider 路由：智谱 BigModel（国内） · {base_url or 'https://open.bigmodel.cn/api/paas/v4'}"
        if provider == "zai":
            return f"当前 Provider 路由：Z.AI（国际） · {base_url or 'https://api.z.ai/api/paas/v4/'}"
        if provider == "deepseek":
            return f"当前 Provider 路由：DeepSeek · {base_url or 'https://api.deepseek.com'}"
        if provider == "openai":
            return f"当前 Provider 路由：OpenAI · {base_url or 'https://api.openai.com/v1'}"
        if provider == "openrouter":
            return f"当前 Provider 路由：OpenRouter · {base_url or 'https://openrouter.ai/api/v1'}"
        if provider == "ollama":
            return f"当前 Provider 路由：Ollama（本地） · {base_url or 'http://127.0.0.1:11434/v1'}"
        if provider == "anthropic":
            return f"当前 Provider 路由：Anthropic · {base_url or '官方默认路由'}"
        if provider == "gemini":
            return f"当前 Provider 路由：Gemini · {base_url or '官方默认路由'}"
        if provider == "custom":
            return f"当前 Provider 路由：自定义接口 · {base_url or '未填写 Base URL'}"
        return f"当前 Provider 路由：{provider or '未配置'} · {base_url or '默认路由'}"

    @staticmethod
    def _provider_recommended_models(provider: str) -> str:
        provider = str(provider or "").strip().lower()
        mapping = {
            "zhipu": "首选 glm-5.3-flash；若要更稳的视觉理解，可改用账号内可用的视觉模型后再测图片能力。",
            "zai": "首选 glm-5.3-flash；国际站也建议先读模型列表，再用“测试图片能力”确认所选模型真的吃图。",
            "deepseek": "优先选择账号内明确支持图片输入的模型；不要只看 provider 名称，务必做图片能力测试。",
            "openai": "优先选择 GPT-4.1 / GPT-4o 一类明确支持图片的模型；文本模型只会通过连接测试，不代表能吃图。",
            "anthropic": "优先选择 Claude 3.5 / 3.7 等支持视觉输入的模型，并控制单页像素避免不必要的图像成本。",
            "gemini": "优先选择 Gemini 1.5 / 2.x 系列支持图片输入的模型；跑书前先测图片能力。",
            "openrouter": "优先选择路由中明确标注 Vision / Multimodal 的模型；不同上游的图片能力差异很大。",
            "ollama": "优先选择本地已安装的视觉模型；纯文本模型会连接成功，但无法完成 AI 图文处理。",
            "custom": "请选择接口方明确支持 image_url / 多模态聊天的模型，并确认 Base URL 真的是 OpenAI 兼容聊天端点。",
        }
        return mapping.get(provider, "请选择一个明确支持图片输入的模型，并先通过图片能力测试。")

    @staticmethod
    def _provider_image_capability_hint(provider: str) -> str:
        provider = str(provider or "").strip().lower()
        if provider in {"zhipu", "zai", "openai", "openrouter", "ollama", "custom", "deepseek"}:
            return "图片能力说明：该 Provider 走 OpenAI 兼容多模态链路；是否真的支持图片，取决于你选中的模型和网关是否接受 image_url。"
        if provider == "anthropic":
            return "图片能力说明：该 Provider 走 Claude Messages 多模态链路；通常兼容图片，但仍要以“测试图片能力”的真实请求为准。"
        if provider == "gemini":
            return "图片能力说明：该 Provider 走 Gemini generate_content 多模态链路；是否稳定吃图，仍需用真实页面测试。"
        return "图片能力说明：AI 图文处理不会凭模型名称盲猜视觉能力；只有真实图片请求通过，才算当前配置可用。"

    @staticmethod
    def _provider_best_practice_hint(provider: str, model: str) -> str:
        provider = str(provider or "").strip().lower()
        model = str(model or "").strip() or "未选模型"
        if provider in {"zhipu", "zai"}:
            return f"最佳实践：GLM 当前选择 {model}。GLM-5.3 Flash 为常驻思考模型：默认使用 reasoning_effort=low 提速；勾选深度思考后使用 high，不再发送 thinking=disabled。"
        if provider == "deepseek":
            return f"最佳实践：DeepSeek 当前选择 {model}。优先使用非思考模式跑整本书，必要时再对疑难页单独复核；图片能力必须先测。"
        if provider == "openrouter":
            return f"最佳实践：OpenRouter 当前选择 {model}。注意不同上游模型的上下文、图片限制和价格差异，跑整本书前一定先做双测试。"
        return f"最佳实践：当前选择 {model}。先通过“测试连接”与“测试图片能力”，再用快速模式跑一小段样本书确认结构与译文输出。"

    @staticmethod
    def _test_cache_key(provider: str, base_url: str, model: str) -> str:
        raw = json.dumps([str(provider or ""), str(base_url or ""), str(model or "")], ensure_ascii=False)
        return "ai_image_tests/" + hashlib.sha1(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _format_cached_test_entry(kind_label: str, state: str, when: str, detail: str) -> str:
        state_text = AIImageProcessingTab._probe_state_badge(state)
        when_text = when or "无记录"
        detail = str(detail or "").strip()
        if detail:
            detail = detail[:72]
            return f"{kind_label} {state_text} · {when_text} · {detail}"
        return f"{kind_label} {state_text} · {when_text}"

    def _load_cached_test_states(self, provider: str, base_url: str, model: str):
        key = self._test_cache_key(provider, base_url, model)
        text_state = str(self._ui_settings.value(f"{key}/text_state", "未测试") or "未测试")
        vision_state = str(self._ui_settings.value(f"{key}/vision_state", "未测试") or "未测试")
        text_when = str(self._ui_settings.value(f"{key}/text_when", "") or "")
        vision_when = str(self._ui_settings.value(f"{key}/vision_when", "") or "")
        text_detail = str(self._ui_settings.value(f"{key}/text_detail", "") or "")
        vision_detail = str(self._ui_settings.value(f"{key}/vision_detail", "") or "")
        return {
            "text": {"state": text_state, "when": text_when, "detail": text_detail},
            "vision": {"state": vision_state, "when": vision_when, "detail": vision_detail},
        }

    def _remember_test_result(self, kind: str, state: str, detail: str):
        try:
            from ai.config import load_ai_settings
            s = load_ai_settings()
            key = self._test_cache_key(s.provider, s.base_url, s.model)
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            self._ui_settings.setValue(f"{key}/{kind}_state", state)
            self._ui_settings.setValue(f"{key}/{kind}_when", now)
            self._ui_settings.setValue(f"{key}/{kind}_detail", str(detail or "")[:240])
            self._ui_settings.sync()
        except Exception:
            pass

    def _refresh_preset_summary(self):
        mode_text = str(self._mode.currentText() or "")
        quality_text = str(self._quality.currentText() or "")
        if self._mode.currentData() == "extract_translate":
            target_text = str(self._target.currentText() or "")
        else:
            target_text = "不翻译"
        trim_text = "完整原页直送API" if self._trim_cb.isChecked() else "本地裁白边模式"
        review_text = "自动二审开启" if self._review_cb.isChecked() else "不自动二审"
        self._preset_summary.setText(
            f"当前预设：{mode_text} · {target_text} · {quality_text} · {trim_text} · {review_text}"
        )

    def _apply_fastest_preset(self):
        self._mode.setCurrentIndex(max(0, self._mode.findData("extract")))
        self._quality.setCurrentIndex(max(0, self._quality.findData("fast")))
        self._trim_cb.setChecked(True)
        self._review_cb.setChecked(False)
        self._status.setText("已应用最快设置：仅提取原文 · 1600px · 完整原页直送API · 不自动二审 · GLM-5.3 Flash=low")

    def _apply_extract_translate_preset(self):
        self._mode.setCurrentIndex(max(0, self._mode.findData("extract_translate")))
        self._target.setCurrentIndex(max(0, self._target.findData("zh-Hans")))
        self._quality.setCurrentIndex(max(0, self._quality.findData("balanced")))
        self._trim_cb.setChecked(True)
        self._review_cb.setChecked(True)
        self._status.setText(
            "已应用提取并翻译设置：简体中文 · 均衡1920px · 完整原页直送API · 结构/疑难复核 · GLM-5.3 Flash=low"
        )

    def _update_elapsed(self):
        if not self._busy or not self._run_started_at:
            self._elapsed_label.clear()
            return
        elapsed = max(0, int(time.monotonic() - self._run_started_at))
        self._elapsed_label.setText(f"服务端处理中 · {elapsed} 秒")

    def _open_ai_settings(self):
        AISettingsDialog(self).exec()
        self._refresh_settings_summary()

    def _refresh_settings_summary(self):
        try:
            from ai.config import load_ai_settings
            s = load_ai_settings()
            signature = (str(s.provider or ""), str(s.model or ""), str(s.base_url or ""), bool(s.configured))
            if signature != self._settings_signature:
                self._settings_signature = signature
                self._text_probe_state = "未测试"
                self._vision_probe_state = "未测试"
            configured = "已配置" if s.configured else "未配置"
            self._settings_label.setText(
                f"当前 Provider：{s.provider or '未配置'} · 模型：{s.model or '未选模型'} · 密钥状态：{configured}"
            )
            self._provider_status_label.setText(
                f"{self._provider_route_hint(s.provider, s.base_url)}\n"
                f"测试状态：文本连接 {self._probe_state_badge(self._text_probe_state)} · 图片能力 {self._probe_state_badge(self._vision_probe_state)}"
            )
        except Exception:
            self._settings_label.setText("AI设置不可用")
            self._provider_status_label.setText("当前 Provider 状态不可用")

    def set_page_context(self, page_images, page_types=None):
        images = [str(Path(p)) for p in (page_images or [])]
        types = {int(k): str(getattr(v, "value", v)) for k, v in (page_types or {}).items()}
        identities = []
        for item in images:
            try:
                st = Path(item).stat()
                identities.append((item, int(st.st_size), int(st.st_mtime_ns)))
            except Exception:
                identities.append((item, None, None))
        signature = (tuple(identities), tuple(sorted(types.items())))
        changed = signature != self._context_signature
        if changed and self._busy:
            # Page Manager is authoritative.  Continuing an in-flight request
            # after page order/type changes could attach text to the wrong page.
            self._cancel_event.set()
            self._run_generation += 1
            self._set_busy(False)
        self._context_signature = signature
        self._page_images = images
        self._page_types = types
        if changed:
            self._source_doc = None; self._translated_doc = None
            self._source_epub_btn.setEnabled(False); self._translation_epub_btn.setEnabled(False)
            self._progress.setValue(0)
            self._preview.clear()
        total = len(self._page_images)
        asset_count = sum(1 for n in range(1, total + 1) if self._page_types.get(n, "paragraph") not in {"paragraph", "unknown"})
        if total:
            self._source_label.setText(f"页面管理：{total} 页 · 已确认非正文/资源页 {asset_count} 页 · AI只处理可OCR页面")
            self._status.setText("已读取页面管理；可以开始 AI 图文处理")
        else:
            self._source_label.setText("页面管理：尚未导入页面")
            self._status.setText("请先在页面管理导入 PDF、图片或图片文件夹")
        if changed:
            self._preview.clear()

    def reset_for_new_book(self):
        if self._busy:
            self._cancel_event.set()
        self.set_page_context([], {})
        self._log.clear()

    def _options(self):
        from engine.ai_image_pipeline import AIImageOptions
        quality = str(self._quality.currentData() or "balanced")
        auto_review = self._review_cb.isChecked() and quality != "fast"
        return AIImageOptions(
            mode=str(self._mode.currentData() or "extract"),
            target_language=str(self._target.currentData() or "zh-Hans"),
            quality=quality,
            batch_pages=2,
            direct_full_page_api=self._trim_cb.isChecked(),
            direct_original_when_safe=True,
            trim_white_margins=not self._trim_cb.isChecked(),
            edge_integrity_guard=(quality != "fast"),
            max_edge_integrity_audits=48 if quality == "publication" else 24,
            auto_review=auto_review,
            use_cache=True,
            max_visual_reviews=160 if quality == "publication" else 96,
            adaptive_batch_split=True,
            translation_repair=(self._mode.currentData() == "extract_translate" and quality != "fast"),
            max_translation_repairs=240 if quality == "publication" else 160,
            verify_source_fingerprint=True,
            audit_suspicious_short_pages=(quality != "fast"),
            continuity_text_check=(quality != "fast"),
            max_short_page_audits=40 if quality == "publication" else 24,
            max_continuity_checks=160 if quality == "publication" else 96,
            audit_flat_layout_pages=(quality != "fast"),
            max_structure_audits=64 if quality == "publication" else 32,
        )

    def _set_busy(self, busy: bool):
        self._busy = bool(busy)
        self._start_btn.setEnabled(not busy)
        self._stop_btn.setEnabled(busy)
        self._test_connection_btn.setEnabled(not busy)
        self._test_btn.setEnabled(not busy)
        self._fastest_btn.setEnabled(not busy)
        self._translate_fast_btn.setEnabled(not busy)
        if busy:
            self._run_started_at = time.monotonic()
            self._elapsed_timer.start()
            self._update_elapsed()
        else:
            self._elapsed_timer.stop()
            self._run_started_at = 0.0
            self._elapsed_label.clear()
            if hasattr(self, "_ai_log_buffer"):
                self._flush_ai_log_buffer(force_all=True)

    def _start(self):
        if self._busy:
            return
        if not self._page_images:
            notify(self, "请先在“页面管理”导入 PDF、单张图片、多个图片或图片文件夹。", "warning")
            return
        settings = ensure_ai_settings(self, "请先配置一个真正支持图片输入的 AI 模型。")
        if settings is None:
            return
        self._refresh_settings_summary()
        self._cancel_event = threading.Event()
        self._run_generation += 1
        generation = self._run_generation
        self._set_busy(True)
        self._source_doc = None; self._translated_doc = None
        self._source_epub_btn.setEnabled(False); self._translation_epub_btn.setEnabled(False)
        self._progress.setValue(0); self._clear_ai_log(); self._preview.clear()
        self._last_progress_ui = 0.0
        self._status.setText("准备页面并建立省 token 的视觉请求…")
        signals = WorkerSignals(); self._signals = signals
        signals.log.connect(self._append_log)
        signals.progress.connect(self._on_progress)
        signals.finished.connect(lambda payload, g=generation: self._finished(payload, g))
        signals.error.connect(lambda message, g=generation: self._failed(message, g))
        page_images = list(self._page_images); page_types = dict(self._page_types); options = self._options()

        def worker():
            try:
                from engine.ai_image_pipeline import AIImageBookProcessor
                processor = AIImageBookProcessor(settings, options)
                result = processor.run(
                    page_images,
                    page_types,
                    cancel_event=self._cancel_event,
                    progress_callback=lambda cur, total: signals.progress.emit(cur, total),
                    log_callback=signals.log.emit,
                )
                signals.finished.emit(result)
            except Exception as exc:
                import traceback
                signals.error.emit(f"{exc}\n\n{traceback.format_exc()}")

        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()

    def _stop(self):
        if self._busy:
            self._cancel_event.set()
            self._status.setText("正在停止；已完成请求和本地缓存会保留…")

    def _append_log(self, text):
        self._ai_log_buffer.push(text)
        if not self._ai_log_flush_timer.isActive():
            self._ai_log_flush_timer.start()

    def _flush_ai_log_buffer(self, *, force_all: bool = False) -> None:
        packets: list[str] = []
        while True:
            lines = self._ai_log_buffer.drain(max_lines=1200 if force_all else 320)
            if not lines:
                break
            packets.append("\n".join(lines))
            if not force_all:
                break
        if packets:
            self._log.appendPlainText("\n".join(packets))
            bar = self._log.verticalScrollBar()
            bar.setValue(bar.maximum())
        if force_all and not self._ai_log_buffer.pending():
            self._ai_log_flush_timer.stop()

    def _clear_ai_log(self) -> None:
        self._ai_log_buffer.clear()
        self._ai_log_flush_timer.stop()
        self._log.clear()

    def _on_progress(self, current: int, total: int):
        total = max(1, int(total or 1)); current = max(0, int(current or 0))
        now = time.monotonic()
        final = current >= total
        if not final and current > 1 and now - self._last_progress_ui < 0.08:
            return
        self._last_progress_ui = now
        self._progress.setValue(min(100, round(current * 100 / total)))
        self._status.setText(f"AI 图文处理：{current} / {total} 个正文页")

    def _finished(self, result, generation: int):
        if generation != self._run_generation:
            return
        self._set_busy(False); self._progress.setValue(100)
        self._source_doc = result.source_document
        self._translated_doc = result.translated_document
        self._source_epub_btn.setEnabled(self._source_doc is not None)
        stats = dict(result.stats or {})
        missing_translation = int(stats.get("translation_missing_blocks", 0) or 0)
        deferred_review = int(stats.get("review_deferred", 0) or 0)
        low_confidence = int(stats.get("low_confidence_blocks", 0) or 0)
        rescue_requested = int(stats.get("page_rescue_requested", 0) or 0)
        rescue_accepted = int(stats.get("page_rescue_accepted", 0) or 0)
        batch_splits = int(stats.get("batch_splits", 0) or 0)
        visual_concurrency_initial = int(stats.get("visual_concurrency_initial", 1) or 1)
        visual_concurrency_peak = int(stats.get("visual_concurrency_peak", visual_concurrency_initial) or visual_concurrency_initial)
        visual_concurrency_final = int(stats.get("visual_concurrency_final", visual_concurrency_peak) or visual_concurrency_peak)
        visual_concurrency_auto = bool(stats.get("visual_concurrency_auto", False))
        visual_parallel_waves = int(stats.get("visual_parallel_waves", 0) or 0)
        prepared_image_format = str(stats.get("prepared_image_format", "") or "").upper()
        boundary_checks = int(stats.get("boundary_checks", 0) or 0)
        boundary_deduped = int(stats.get("boundary_deduped", 0) or 0)
        short_audits = int(stats.get("short_page_audit_requested", 0) or 0)
        short_recovered = int(stats.get("short_page_audit_accepted", 0) or 0)
        structure_audits = int(stats.get("structure_audit_requested", 0) or 0)
        structure_recovered = int(stats.get("structure_audit_accepted", 0) or 0)
        structure_rejected = int(stats.get("structure_audit_rejected", 0) or 0)
        structure_unresolved_pages = list(stats.get("structure_unresolved_pages") or [])
        local_dialogue_splits = int(stats.get("local_dialogue_splits", 0) or 0)
        edge_audits = int(stats.get("edge_integrity_requested", 0) or 0)
        edge_recovered = int(stats.get("edge_integrity_accepted", 0) or 0)
        edge_rejected = int(stats.get("edge_integrity_rejected", 0) or 0)
        edge_unresolved_pages = list(stats.get("edge_unresolved_pages") or [])
        direct_full_page_api = bool(stats.get("direct_full_page_api", False))
        direct_original_pages = int(stats.get("direct_original_pages", 0) or 0)
        continuity_checks = int(stats.get("continuity_checks", 0) or 0)
        continuity_merged = int(stats.get("continuity_merged", 0) or 0)
        unresolved_pages = list(stats.get("unresolved_text_pages") or [])
        source_export_ready = bool(stats.get("source_export_ready", not unresolved_pages))
        translation_export_ready = bool(stats.get("translation_export_ready", self._translated_doc is not None and missing_translation == 0))
        translation_repair_requested = int(stats.get("translation_repair_requested", 0) or 0)
        translation_repair_accepted = int(stats.get("translation_repair_accepted", 0) or 0)
        self._source_epub_btn.setEnabled(self._source_doc is not None and source_export_ready)
        self._translation_epub_btn.setEnabled(self._translated_doc is not None and translation_export_ready)
        usage = dict(stats.get("usage") or {})
        visual_usage = dict(stats.get("usage_visual") or {})
        continuity_usage = dict(stats.get("usage_continuity") or {})
        translation_usage = dict(stats.get("usage_translation_repair") or {})
        translation_note = (
            f"；译文缺失 {missing_translation} 块（已保留原文，补齐前禁用译文 EPUB）"
            if missing_translation else ""
        )
        review_note = ""
        if deferred_review or low_confidence:
            review_note = f"；仍有疑难：待复核 {deferred_review} 区域 / 低置信 {low_confidence} 块"
        rescue_note = (
            f"；高清遗漏保护 {rescue_accepted}/{rescue_requested} 页恢复正文"
            if rescue_requested else ""
        )
        split_note = f"；批次自适应拆分 {batch_splits} 次" if batch_splits else ""
        concurrency_note = (
            f"；视觉并发 {visual_concurrency_initial}→峰值{visual_concurrency_peak}→结束{visual_concurrency_final}"
            + ("（自动）" if visual_concurrency_auto else "（固定）")
            + (f"，{visual_parallel_waves} 波" if visual_parallel_waves else "")
            + (f"，{prepared_image_format}" if prepared_image_format else "")
        )
        boundary_note = (
            f"；跨页重复核验 {boundary_deduped}/{boundary_checks} 处确认去重"
            if boundary_checks else ""
        )
        short_note = (
            f"；异常短页高清审计 {short_recovered}/{short_audits} 页补回更多正文"
            if short_audits else ""
        )
        structure_note = (
            f"；段落/对白结构高清复核 {structure_recovered}/{structure_audits} 页采用"
            + (f"，{structure_rejected} 页因防改写校验保留首轮" if structure_rejected else "")
            + (f"，仍有 {len(structure_unresolved_pages)} 页疑似压平" if structure_unresolved_pages else "")
            if structure_audits else ""
        )
        local_structure_note = (
            f"；本地零Token对白拆段 {local_dialogue_splits} 块"
            if local_dialogue_splits else ""
        )
        edge_note = (
            f"；最右列/页顶完整性复核 {edge_recovered}/{edge_audits} 页采用"
            + (f"，{edge_rejected} 页保留首轮待确认" if edge_rejected else "")
            if edge_audits else ""
        )
        direct_note = (
            f"；完整原页直送API（{direct_original_pages} 页直接使用源图）"
            if direct_full_page_api else "；本地裁白边兼容模式"
        )
        continuity_note = (
            f"；纯文本跨页接续 {continuity_merged}/{continuity_checks} 处确认合并"
            if continuity_checks else ""
        )
        preflight_note = ""
        if unresolved_pages:
            preview = ",".join(str(x) for x in unresolved_pages[:8])
            suffix = "…" if len(unresolved_pages) > 8 else ""
            preflight_note = f"；成书预检阻止直接导出：正文页 {preview}{suffix} 仍无可确认正文，请回页面管理核对页型或重试"
        elif not source_export_ready:
            if edge_unresolved_pages:
                preview = ",".join(str(x) for x in edge_unresolved_pages[:8])
                suffix = "…" if len(edge_unresolved_pages) > 8 else ""
                preflight_note = f"；出版级成书预检未通过：第 {preview}{suffix} 页的最右列/页顶完整性仍未确认，暂不开放直接 EPUB"
            elif structure_unresolved_pages:
                preview = ",".join(str(x) for x in structure_unresolved_pages[:8])
                suffix = "…" if len(structure_unresolved_pages) > 8 else ""
                preflight_note = f"；出版级成书预检未通过：第 {preview}{suffix} 页仍疑似段落/对白压平，暂不开放直接 EPUB"
            else:
                preflight_note = "；出版级成书预检未通过：仍有低置信/待视觉复核内容，暂不开放直接 EPUB"
        translation_repair_note = (
            f"；纯文本补译 {translation_repair_accepted}/{translation_repair_requested} 块"
            if translation_repair_requested else ""
        )
        self._status.setText(
            f"完成：处理 {stats.get('processed_pages', 0)} 页，跳过资源页 {stats.get('skipped_asset_pages', 0)} 页，"
            f"正文块 {stats.get('text_blocks', 0)}；局部复核 {stats.get('reviewed_regions', 0)}；缓存命中 {stats.get('cache_hits', 0)}"
            f"{direct_note}{rescue_note}{split_note}{concurrency_note}{short_note}{structure_note}{local_structure_note}{edge_note}{boundary_note}{continuity_note}{translation_repair_note}{review_note}{translation_note}{preflight_note}。"
            f"API usage：输入 {usage.get('prompt_tokens', 0):,} / 输出 {usage.get('completion_tokens', 0):,} tokens"
            f" / 网络重试 {usage.get('transport_retries', 0):,} 次。"
        )
        visual_total = int(visual_usage.get("total_tokens", 0) or 0)
        text_total = int(continuity_usage.get("total_tokens", 0) or 0) + int(translation_usage.get("total_tokens", 0) or 0)
        if visual_total or text_total:
            self._append_log(
                f"Token 分层：视觉处理 {visual_total:,}；纯文本跨页/补译 {text_total:,}；"
                "纯文本阶段不会再次上传页面图片。"
            )
        self._append_log(self._status.text())
        lines = []
        source_blocks = self._source_doc.text_blocks()[:80] if self._source_doc else []
        translation_by_source = {}
        if self._translated_doc:
            for b in self._translated_doc.text_blocks():
                sid = str((b.metadata or {}).get("source_block_id") or "")
                if sid:
                    translation_by_source[sid] = b.text
        for block in source_blocks:
            prefix = {BlockType.CHAPTER:"【章】", BlockType.SECTION:"【节】", BlockType.DIALOGUE:"【对白】"}.get(block.type, "")
            lines.append(f"{prefix}{block.text}")
            translated = translation_by_source.get(block.id)
            if translated:
                lines.append(f"  → {translated}")
        if self._source_doc and len(self._source_doc.text_blocks()) > len(source_blocks):
            lines.append("\n……预览仅显示前 80 块……")
        self._preview.setPlainText("\n".join(lines))
        self.documents_ready.emit(self._source_doc, self._translated_doc)

    def _failed(self, details: str, generation: int):
        if generation != self._run_generation:
            return
        self._set_busy(False)
        stopped = "已停止" in details
        self._status.setText("AI 图文处理已停止" if stopped else "AI 图文处理失败")
        if not stopped:
            show_error_dialog(self, "AI 图文处理失败", details)

    def _emit_epub(self, doc, kind: str):
        if doc is None:
            return
        try:
            from engine.ai_image_pipeline import stale_ai_image_pages
            stale_pages = stale_ai_image_pages(doc, self._page_images)
        except Exception as exc:
            QMessageBox.warning(self, "无法核对页面", f"EPUB 前页面指纹核对失败：{exc}")
            return
        if stale_pages:
            preview = ", ".join(str(x) for x in stale_pages[:12])
            suffix = "…" if len(stale_pages) > 12 else ""
            QMessageBox.warning(
                self,
                "页面已变化",
                f"第 {preview}{suffix} 页的原始图片在 AI 成稿后发生变化。\n\n"
                "为避免把旧文字绑定到新图片，已阻止直接生成 EPUB。请重新运行 AI 图文处理。",
            )
            return
        self.epub_requested.emit(doc, kind)

    def _test_connection(self):
        if self._busy:
            return
        settings = ensure_ai_settings(self, "请先配置一个可用的 AI Provider。")
        if settings is None:
            return
        self._refresh_settings_summary()
        self._set_busy(True); self._status.setText("正在测试 Provider 文本连接…")
        signals = WorkerSignals(); self._signals = signals
        signals.finished.connect(self._test_connection_finished); signals.error.connect(self._test_connection_failed)

        def worker():
            try:
                from ai.provider_factory import test_provider
                reply = test_provider(settings)
                signals.finished.emit({"reply": reply})
            except Exception as exc:
                import traceback
                signals.error.emit(f"{exc}\n\n{traceback.format_exc()}")

        self._worker_thread = threading.Thread(target=worker, daemon=True); self._worker_thread.start()

    def _test_connection_finished(self, payload):
        self._set_busy(False)
        self._text_probe_state = "通过"
        reply = str(payload.get("reply") or "")
        self._remember_test_result("text", "通过", reply)
        self._refresh_settings_summary()
        self._status.setText(f"连接测试通过 · 返回：{reply[:120]}")

    def _test_connection_failed(self, details):
        self._set_busy(False)
        self._text_probe_state = "失败"
        self._remember_test_result("text", "失败", str(details or ""))
        self._refresh_settings_summary()
        self._status.setText("连接测试失败")
        show_error_dialog(self, "连接测试失败", details)

    def _test_model(self):
        if self._busy:
            return
        if not self._page_images:
            notify(self, "请先在页面管理导入至少一页。", "warning")
            return
        settings = ensure_ai_settings(self, "请先配置一个支持图片输入的 AI 模型。")
        if settings is None:
            return
        # Prefer a page currently treated as正文/未分类; fall back to first page.
        test_path = self._page_images[0]
        for page_no, path in enumerate(self._page_images, start=1):
            if self._page_types.get(page_no, "paragraph") in {"paragraph", "unknown"}:
                test_path = path; break
        if QMessageBox.question(
            self, "测试图片能力",
            "将发送一张缩小到最长边 768px 的当前书页到所选 API。此操作会产生一次很小的真实 API 用量。继续？",
        ) != QMessageBox.Yes:
            return
        self._set_busy(True); self._status.setText("正在测试模型图片输入能力…")
        signals = WorkerSignals(); self._signals = signals
        signals.finished.connect(self._test_finished); signals.error.connect(self._test_failed)
        def worker():
            try:
                from engine.ai_image_pipeline import test_multimodal_model
                text, usage = test_multimodal_model(settings, test_path)
                signals.finished.emit({"text": text, "usage": usage})
            except Exception as exc:
                import traceback
                signals.error.emit(f"{exc}\n\n{traceback.format_exc()}")
        self._worker_thread = threading.Thread(target=worker, daemon=True); self._worker_thread.start()

    def _test_finished(self, payload):
        self._set_busy(False)
        self._vision_probe_state = "通过"
        usage = dict(payload.get("usage") or {})
        text = str(payload.get("text") or "")
        self._remember_test_result("vision", "通过", text)
        self._refresh_settings_summary()
        self._status.setText(f"图片能力测试通过 · 识别片段：{text[:80]} · tokens {usage.get('total_tokens', 0):,}")

    def _test_failed(self, details):
        self._set_busy(False)
        self._vision_probe_state = "失败"
        self._remember_test_result("vision", "失败", str(details or ""))
        self._refresh_settings_summary()
        self._status.setText("图片能力测试失败")
        show_error_dialog(self, "图片能力测试失败", details)

    def shutdown_cleanup(self):
        self._cancel_event.set()

