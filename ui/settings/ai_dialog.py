from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton, QComboBox,
    QLineEdit, QCheckBox, QDoubleSpinBox, QSpinBox, QFileDialog, QMessageBox,
    QScrollArea, QFrame,
)
from PySide6.QtCore import QSettings, QTimer

from ui.dialogs import show_error_dialog
from ui.common.signals import WorkerSignals
from ui.common.styling import ACC, MUTED, CARD, BORDER, accent_button

class AISettingsDialog(QDialog):
    """桌面端 AI 配置；本地设置优先，环境变量作为回退。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        from ai.config import load_ai_settings
        self.setWindowTitle("AI 服务设置")
        self.setMinimumWidth(560)
        # Comfortable default on normal displays; the global dialog polisher
        # still clamps this to the available screen on small displays.
        self.resize(760, 700)
        self._settings = load_ai_settings()
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 14)
        root.setSpacing(12)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll_host = QWidget()
        content = QVBoxLayout(scroll_host)
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(12)
        scroll.setWidget(scroll_host)
        root.addWidget(scroll, 1)

        note_card = QFrame()
        note_card.setObjectName("aiSettingsNoteCard")
        note_card.setStyleSheet(f"QFrame#aiSettingsNoteCard{{background:{CARD};border:1px solid {BORDER};border-radius:12px;}}")
        note_layout = QVBoxLayout(note_card)
        note_layout.setContentsMargins(14, 12, 14, 12)
        note = QLabel("模型等设置保存在本机。API Key 勾选保存时使用 macOS Keychain / Windows DPAPI；未勾选则只在本次运行内存中保留。")
        note.setWordWrap(True); note.setStyleSheet(f"color:{MUTED};border:none;background:transparent;"); note_layout.addWidget(note)
        content.addWidget(note_card)

        form = QFormLayout()
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(9)
        self.provider = QComboBox()
        for key, label in (("openai","OpenAI"),("anthropic","Claude / Anthropic"),("gemini","Google Gemini"),("deepseek","DeepSeek"),("zhipu","智谱 BigModel / GLM（国内）"),("zai","Z.AI / GLM（国际）"),("openrouter","OpenRouter"),("ollama","Ollama（本地）"),("custom","自定义 OpenAI 兼容接口")):
            self.provider.addItem(label,key)
        idx=self.provider.findData(self._settings.provider); self.provider.setCurrentIndex(max(0,idx)); self.provider.currentIndexChanged.connect(self._provider_changed)
        self.key = QLineEdit(self._settings.api_key); self.key.setEchoMode(QLineEdit.Password)
        key_row=QWidget(); kl=QHBoxLayout(key_row); kl.setContentsMargins(0,0,0,0); kl.addWidget(self.key,1)
        show=QCheckBox("显示"); show.toggled.connect(lambda on:self.key.setEchoMode(QLineEdit.Normal if on else QLineEdit.Password)); kl.addWidget(show)
        self.persist_key = QCheckBox("保存密钥到本机")
        try:
            from ai.config import has_persisted_api_key
            self.persist_key.setChecked(has_persisted_api_key())
        except Exception:
            self.persist_key.setChecked(False)
        self.model = QComboBox()
        self.model.setEditable(True)
        # Keep a concrete Python wrapper for the editable field.  On some
        # PySide6/macOS builds, querying QComboBox's private line edit after the
        # application event filter is installed can return a base QWidget.
        self.model_editor = QLineEdit(self.model)
        self.model.setLineEdit(self.model_editor)
        self.model.setProperty("nfNoTranslateItems", True)
        self.model.setInsertPolicy(QComboBox.NoInsert)
        self.model.setMaxVisibleItems(24)
        self.model.addItem(self._settings.model)
        self.model.setCurrentText(self._settings.model)
        self.model.setToolTip("输入 API Key 后可自动读取账号可用模型；下拉框仍允许手动填写模型名称。")
        self.model_editor.setPlaceholderText("输入密钥后加载模型，或手动填写")
        self.refresh_models_btn = QPushButton("刷新模型")
        self.refresh_models_btn.setToolTip("使用当前 Provider、API Key 和 Base URL 获取可用模型列表")
        self.refresh_models_btn.clicked.connect(lambda: self._load_models(show_errors=True))
        model_row = QWidget()
        model_layout = QHBoxLayout(model_row)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(8)
        model_layout.addWidget(self.model, 1)
        model_layout.addWidget(self.refresh_models_btn)
        self.base_url=QLineEdit(self._settings.base_url)
        self.glossary_path = QLineEdit(str(getattr(self._settings, "glossary_path", "") or ""))
        self.glossary_path.setPlaceholderText("可选：全书术语白名单（TXT / JSON）")
        glossary_row = QWidget(); glossary_layout = QHBoxLayout(glossary_row)
        glossary_layout.setContentsMargins(0, 0, 0, 0); glossary_layout.setSpacing(8)
        glossary_layout.addWidget(self.glossary_path, 1)
        glossary_btn = QPushButton("选择…"); glossary_btn.clicked.connect(self._choose_glossary)
        glossary_layout.addWidget(glossary_btn)
        self.deepseek_thinking=QCheckBox("启用思考模式（更慢、消耗额外推理 Tokens）")
        self.deepseek_thinking.setChecked(bool(getattr(self._settings,"deepseek_thinking",False)))
        self.deepseek_thinking.toggled.connect(self._sync_deepseek_controls)
        self.glm_thinking=QCheckBox("GLM 深度思考（GLM-5.3 Flash：关闭=low，开启=high；无法完全关闭）")
        self.glm_thinking.setChecked(bool(getattr(self._settings,"glm_thinking",False)))
        self.deepseek_effort=QComboBox(); self.deepseek_effort.addItem("High","high"); self.deepseek_effort.addItem("Max","max")
        _effort=self.deepseek_effort.findData(str(getattr(self._settings,"deepseek_reasoning_effort","high"))); self.deepseek_effort.setCurrentIndex(max(0,_effort))
        self.deepseek_effort.setToolTip("仅在 DeepSeek 思考模式开启时生效。纠错排版通常不需要思考模式。")
        self.ocr_repair_mode = QComboBox()
        self.ocr_repair_mode.addItem("可读性优先（推荐）", "readability")
        self.ocr_repair_mode.addItem("严格还原（不推测缺字）", "strict")
        _repair_mode = self.ocr_repair_mode.findData(str(getattr(self._settings, "ocr_repair_mode", "readability")))
        self.ocr_repair_mode.setCurrentIndex(max(0, _repair_mode))
        self.ocr_repair_mode.setToolTip("可读性优先：允许结合上下文补助词、假名和短缺损片段；严格还原：无法唯一确定时保留原文。")
        self.temperature=QDoubleSpinBox(); self.temperature.setRange(0.0,2.0); self.temperature.setSingleStep(0.1); self.temperature.setDecimals(2); self.temperature.setValue(self._settings.temperature)
        self.max_tokens=QSpinBox(); self.max_tokens.setRange(256,200000); self.max_tokens.setSingleStep(1024); self.max_tokens.setValue(self._settings.max_tokens)
        self.concurrency=QSpinBox(); self.concurrency.setRange(0,64); self.concurrency.setValue(self._settings.concurrency); self.concurrency.setSpecialValueText("自动"); self.concurrency.setToolTip("0=根据 RPM、Key 数和接口类型自动计算；在线接口通常为 8 路起，本地 Ollama 建议 1。")
        self.rpm_limit=QSpinBox(); self.rpm_limit.setRange(0,100000); self.rpm_limit.setValue(self._settings.rpm_limit); self.rpm_limit.setSpecialValueText("不限")
        self.tpm_limit=QSpinBox(); self.tpm_limit.setRange(0,100000000); self.tpm_limit.setSingleStep(10000); self.tpm_limit.setValue(self._settings.tpm_limit); self.tpm_limit.setSpecialValueText("不限")
        self.batch_chars=QSpinBox(); self.batch_chars.setRange(0,200000); self.batch_chars.setSingleStep(2000); self.batch_chars.setValue(self._settings.batch_chars); self.batch_chars.setSpecialValueText("不限"); self.batch_chars.setToolTip("可选的单批正文字符硬上限。通常保持不限，由输入 Token 预算自动切批。")
        self.batch_tokens=QSpinBox(); self.batch_tokens.setRange(0,200000); self.batch_tokens.setSingleStep(1000); self.batch_tokens.setValue(self._settings.batch_tokens); self.batch_tokens.setSpecialValueText("自动"); self.batch_tokens.setToolTip("按真实序列化请求估算输入 Token 后切批。DeepSeek 非思考自动模式：仅纠错约 48000，纠错排版约 32000；其他接口约 24000/16000。")
        self.request_timeout=QSpinBox(); self.request_timeout.setRange(15,1800); self.request_timeout.setSingleStep(15); self.request_timeout.setSuffix(" 秒"); self.request_timeout.setValue(self._settings.request_timeout)
        self.json_mode=QCheckBox("使用接口原生 JSON 模式（不支持时自动回退）"); self.json_mode.setChecked(self._settings.json_mode)
        form.addRow("AI Provider",self.provider); form.addRow("API Key",key_row); form.addRow("密钥存储",self.persist_key); form.addRow("Model",model_row); form.addRow("Base URL",self.base_url); form.addRow("术语白名单",glossary_row); form.addRow("GLM 思考",self.glm_thinking); form.addRow("DeepSeek 思考",self.deepseek_thinking); form.addRow("思考强度",self.deepseek_effort); form.addRow("OCR修复目标",self.ocr_repair_mode); form.addRow("Temperature",self.temperature); form.addRow("Max Tokens",self.max_tokens); form.addRow("并发请求",self.concurrency); form.addRow("单批输入 Tokens",self.batch_tokens); form.addRow("正文字符上限",self.batch_chars); form.addRow("请求超时",self.request_timeout); form.addRow("JSON 输出",self.json_mode); form.addRow("RPM 限制",self.rpm_limit); form.addRow("TPM 限制",self.tpm_limit)
        form_card = QFrame()
        form_card.setObjectName("aiSettingsFormCard")
        form_card.setStyleSheet(f"QFrame#aiSettingsFormCard{{background:{CARD};border:1px solid {BORDER};border-radius:12px;}}")
        form_layout = QVBoxLayout(form_card)
        form_layout.setContentsMargins(16, 14, 16, 14)
        form_layout.addLayout(form)
        content.addWidget(form_card)
        self._text_test_state = "未测试"
        self._vision_test_state = "未测试"
        self._text_test_detail = ""
        self._vision_test_detail = ""
        self._settings_store = QSettings("NovelFormatter", "NovelFormatter")
        self.provider_state=QLabel("")
        self.provider_state.setWordWrap(True)
        self.provider_state.setStyleSheet(f"color:{MUTED};")
        content.addWidget(self.provider_state)
        self.status=QLabel(""); self.status.setWordWrap(True); self.status.setStyleSheet(f"color:{MUTED};"); content.addWidget(self.status)
        self.guide_label=QLabel("建议顺序：先点“测试连接”，确认文本路由可达；再点“测试图片能力”，确认 AI 图文处理会用到的真实图片链路正常。")
        self.guide_label.setWordWrap(True)
        self.guide_label.setStyleSheet(f"color:{MUTED};font-size:11px;")
        content.addWidget(self.guide_label)
        content.addStretch(1)
        self.key.editingFinished.connect(lambda: self._load_models(show_errors=False))
        self.base_url.editingFinished.connect(lambda: self._load_models(show_errors=False))
        self.model.currentTextChanged.connect(lambda _=None: self._reset_test_states())
        buttons=QHBoxLayout(); self.test_btn=QPushButton("测试连接"); self.test_btn.setToolTip("只验证当前 Provider / API Key / Base URL / 文本路由是否可达；不验证图片输入。") ; self.test_btn.clicked.connect(self._test); buttons.addWidget(self.test_btn)
        self.test_vision_btn=QPushButton("测试图片能力"); self.test_vision_btn.setToolTip("生成一张本地 NF42 测试图，以与 AI 图文处理相同的 Base64 Data URL 方式验证当前模型是否真的读取图片")
        self.test_vision_btn.clicked.connect(self._test_vision); buttons.addWidget(self.test_vision_btn); buttons.addStretch()
        cancel=QPushButton("取消"); cancel.clicked.connect(self.reject); buttons.addWidget(cancel)
        save=accent_button("保存设置",color=ACC); save.clicked.connect(self._save); buttons.addWidget(save); root.addLayout(buttons)
        self._last_provider=str(self.provider.currentData()); self._provider_changed(initial=True); self._reset_test_states()

    def _choose_glossary(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择全书术语白名单", self.glossary_path.text(),
            "术语表 (*.txt *.tsv *.csv *.json);;所有文件 (*)",
        )
        if path:
            self.glossary_path.setText(path)

    def _current(self):
        from ai.config import AISettings, normalise_api_key
        return AISettings(provider=str(self.provider.currentData()),api_key=normalise_api_key(self.key.text()),model=self.model.currentText().strip(),base_url=self.base_url.text().strip(),temperature=self.temperature.value(),max_tokens=self.max_tokens.value(),concurrency=self.concurrency.value(),rpm_limit=self.rpm_limit.value(),tpm_limit=self.tpm_limit.value(),batch_chars=self.batch_chars.value(),batch_tokens=self.batch_tokens.value(),request_timeout=self.request_timeout.value(),json_mode=self.json_mode.isChecked(),glm_thinking=self.glm_thinking.isChecked(),deepseek_thinking=self.deepseek_thinking.isChecked(),deepseek_reasoning_effort=str(self.deepseek_effort.currentData() or "high"),deepseek_user_id="novel_formatter",ocr_repair_mode=str(self.ocr_repair_mode.currentData() or "readability"),glossary_path=self.glossary_path.text().strip())

    def _replace_model_items(self, models, preferred=""):
        preferred = str(preferred or self.model.currentText() or "").strip()
        values = []
        seen = set()
        for value in models or []:
            item = str(value or "").strip()
            if item and item not in seen:
                seen.add(item)
                values.append(item)
        if preferred and preferred not in seen:
            values.insert(0, preferred)

        self.model.blockSignals(True)
        self.model.clear()
        self.model.addItems(values)
        if preferred:
            self.model.setCurrentText(preferred)
        elif values:
            self.model.setCurrentIndex(0)
        self.model.blockSignals(False)

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

    def _refresh_provider_state(self):
        st = self._current()
        configured = "已配置" if st.configured else "未配置"
        self.provider_state.setText(
            f"当前 Provider：{st.provider or '未配置'} · 模型：{st.model or '未选模型'} · 密钥状态：{configured}\n"
            f"{self._provider_route_hint(st.provider, st.base_url)}\n"
            f"测试状态：文本连接 {self._probe_state_badge(self._text_test_state)} · 图片能力 {self._probe_state_badge(self._vision_test_state)}"
        )

    def _reset_test_states(self):
        self._text_test_state = "未测试"
        self._vision_test_state = "未测试"
        self._refresh_provider_state()

    def _cancel_test_request(self):
        self._test_request_id = int(getattr(self, "_test_request_id", 0)) + 1
        if hasattr(self, "test_btn"):
            self.test_btn.setEnabled(True)
        if hasattr(self, "test_vision_btn"):
            self.test_vision_btn.setEnabled(True)

    def _cancel_model_request(self):
        """Invalidate an in-flight model-list request and restore the controls.

        Provider changes can happen while the previous provider is still being
        queried.  The late result is discarded by request id, while this method
        makes sure the refresh button cannot remain stuck in ``读取中…``.
        """
        self._model_request_id = int(getattr(self, "_model_request_id", 0)) + 1
        self.refresh_models_btn.setEnabled(True)
        self.refresh_models_btn.setText("刷新模型")

    def _load_models(self, show_errors=False):
        # 每次请求都有序号；切换服务商或连续刷新时，旧请求返回后不会覆盖新列表。
        self._model_request_id = int(getattr(self, "_model_request_id", 0)) + 1
        request_id = self._model_request_id
        provider = str(self.provider.currentData() or "")
        key = self.key.text().strip()
        base_url = self.base_url.text().strip()
        if provider != "ollama" and key:
            from ai.config import api_key_validation_error, normalise_api_key
            key = normalise_api_key(key)
            key_error = api_key_validation_error(key)
            if key_error:
                self.status.setText(f"API Key 无效：{key_error}")
                if show_errors:
                    QMessageBox.warning(self, "API Key 无效", key_error)
                return
        if provider != "ollama" and not key:
            if show_errors:
                QMessageBox.warning(self, "缺少 API Key", "请先输入 API Key，再刷新模型列表。")
            return
        if provider == "custom" and not base_url:
            if show_errors:
                QMessageBox.warning(self, "缺少 Base URL", "自定义接口需要先填写 Base URL。")
            return

        self.refresh_models_btn.setEnabled(False)
        self.refresh_models_btn.setText("读取中…")
        self.status.setText("正在读取可用模型列表……")
        timeout = min(max(int(self.request_timeout.value()), 15), 60)

        def worker():
            try:
                from ai.model_catalog import fetch_available_models
                models = fetch_available_models(provider, key, base_url, timeout=timeout)
                signals.finished.emit({
                    "request_id": request_id,
                    "provider": provider,
                    "models": models,
                })
            except Exception as exc:
                signals.error.emit(str(exc))

        signals = WorkerSignals()
        self._model_signals = signals
        signals.finished.connect(self._on_models_loaded)
        signals.error.connect(
            lambda message, rid=request_id: self._on_models_error(message, show_errors, rid)
        )
        threading.Thread(target=worker, daemon=True).start()

    def _on_models_loaded(self, payload):
        request_id = int(payload.get("request_id", -1)) if isinstance(payload, dict) else -1
        provider = str(payload.get("provider", "")) if isinstance(payload, dict) else ""
        if request_id != int(getattr(self, "_model_request_id", 0)):
            return
        if provider != str(self.provider.currentData() or ""):
            return
        models = list(payload.get("models") or [])
        preferred = self.model.currentText().strip()
        self._replace_model_items(models, preferred)
        self.refresh_models_btn.setEnabled(True)
        self.refresh_models_btn.setText("刷新模型")
        self.status.setText(f"✓ 已读取 {len(models)} 个可用模型，可从下拉框选择；也可以手动输入。")

    def _on_models_error(self, message, show_errors=False, request_id=None):
        if request_id is not None and request_id != int(getattr(self, "_model_request_id", 0)):
            return
        self.refresh_models_btn.setEnabled(True)
        self.refresh_models_btn.setText("刷新模型")
        self.status.setText(f"模型列表读取失败：{message}。仍可在下拉框中手动填写模型名称。")
        if show_errors:
            QMessageBox.warning(self, "读取模型失败", f"{message}\n\n仍可手动填写模型名称。")


    def _sync_deepseek_controls(self, on: bool) -> None:
        is_deepseek = str(self.provider.currentData() or "") == "deepseek"
        self.deepseek_effort.setEnabled(is_deepseek and bool(on))
        self.temperature.setEnabled(not (is_deepseek and bool(on)))

    def _provider_changed(self, _index=None, initial=False):
        from ai.config import provider_defaults
        if not initial:
            self._cancel_model_request()
            self._cancel_test_request()
            self._reset_test_states()
        provider=str(self.provider.currentData()); model,url=provider_defaults(provider)
        old=getattr(self,'_last_provider',provider); old_model,old_url=provider_defaults(old)
        if not initial:
            current_model = self.model.currentText().strip()
            if not current_model or current_model == old_model:
                self._replace_model_items([model] if model else [], model)
            if not self.base_url.text().strip() or self.base_url.text().strip()==old_url:self.base_url.setText(url)
        self._last_provider=provider
        if provider == "ollama":
            self.key.setPlaceholderText("本地 Ollama 无需 API Key")
        elif provider == "custom":
            self.key.setPlaceholderText("API Key 可选（无鉴权接口可留空）")
        else:
            self.key.setPlaceholderText("请输入 API Key")
        is_deepseek=provider=="deepseek"
        is_glm=provider in {"zhipu", "zai"}
        self.glm_thinking.setEnabled(is_glm)
        self.deepseek_thinking.setEnabled(is_deepseek)
        self._sync_deepseek_controls(self.deepseek_thinking.isChecked())
        if is_deepseek:
            if not initial and self.max_tokens.value()==24000:self.max_tokens.setValue(48000)
            self.status.setText("DeepSeek 默认使用 V4-Flash 非思考模式；系统会固定静态提示前缀以提高缓存命中，并自动采用高并发批处理。")
        elif provider == "zhipu":
            self.status.setText("智谱国内开放平台：Base URL 应为 https://open.bigmodel.cn/api/paas/v4。AI 图文处理请再点“测试图片能力”，不要只测纯文本连接。")
        elif provider == "zai":
            self.status.setText("Z.AI 国际平台：默认 Base URL 为 https://api.z.ai/api/paas/v4/。国内 BigModel 密钥请改选“智谱 BigModel / GLM（国内）”。")
        elif initial or "DeepSeek" in self.status.text() or "智谱" in self.status.text() or "Z.AI" in self.status.text():
            self.status.setText("")
        self._refresh_provider_state()
        if not initial and provider=="ollama": self.concurrency.setValue(1)
        if not initial and (provider == "ollama" or self.key.text().strip()):
            QTimer.singleShot(0, lambda: self._load_models(show_errors=False))

    def _validate(self, require_key=True):
        st=self._current()
        if not st.model:return "请填写模型名称"
        if require_key and st.requires_key and not st.api_key:return "请填写 API Key"
        if st.requires_key and st.api_key:
            from ai.config import api_key_validation_error
            key_error = api_key_validation_error(st.api_key)
            if key_error:return key_error
        if st.provider=="custom" and not st.base_url:return "自定义接口必须填写 Base URL"
        if st.glossary_path and not Path(st.glossary_path).expanduser().is_file():return "术语白名单文件不存在"
        return ""

    def _test(self):
        error=self._validate(require_key=True)
        if error: QMessageBox.warning(self,"配置不完整",error); return
        self._test_request_id = int(getattr(self, "_test_request_id", 0)) + 1
        request_id = self._test_request_id
        provider = str(self.provider.currentData() or "")
        self.test_btn.setEnabled(False); self.status.setText("正在连接……"); st=self._current()
        def worker():
            try:
                from ai.provider_factory import test_provider
                signals.finished.emit({"reply": test_provider(st), "request_id": request_id, "provider": provider})
            except Exception:
                import traceback; signals.error.emit(traceback.format_exc())
        signals=self._test_signals=WorkerSignals()
        signals.finished.connect(self._on_test_finished)
        signals.error.connect(lambda err, rid=request_id: self._on_test_failed(err, rid))
        threading.Thread(target=worker,daemon=True).start()

    def _test_vision(self):
        error=self._validate(require_key=True)
        if error: QMessageBox.warning(self,"配置不完整",error); return
        self._test_request_id = int(getattr(self, "_test_request_id", 0)) + 1
        request_id = self._test_request_id
        provider = str(self.provider.currentData() or "")
        self.test_btn.setEnabled(False); self.test_vision_btn.setEnabled(False)
        self.status.setText("正在用本地测试图验证图片输入……"); st=self._current()
        def worker():
            try:
                from ai.provider_factory import probe_multimodal_provider
                signals.finished.emit({"reply": probe_multimodal_provider(st), "request_id": request_id, "provider": provider, "vision": True})
            except Exception:
                import traceback; signals.error.emit(traceback.format_exc())
        signals=self._test_signals=WorkerSignals()
        signals.finished.connect(self._on_test_finished)
        signals.error.connect(lambda err, rid=request_id: self._on_test_failed(err, rid))
        threading.Thread(target=worker,daemon=True).start()

    def _on_test_finished(self, payload):
        request_id = int(payload.get("request_id", -1))
        provider = str(payload.get("provider", ""))
        if request_id != int(getattr(self, "_test_request_id", 0)):
            return
        if provider != str(self.provider.currentData() or ""):
            return
        self.test_btn.setEnabled(True)
        self.test_vision_btn.setEnabled(True)
        if payload.get("vision"):
            self._vision_test_state = "通过"
            self._remember_test_result("vision", "通过", str(payload.get('reply', '') or ""))
            self.status.setText(f"✓ 图片输入成功：{payload.get('reply', '')}")
        else:
            self._text_test_state = "通过"
            self._remember_test_result("text", "通过", str(payload.get('reply', '') or ""))
            self.status.setText(f"✓ 连接成功：{payload.get('reply', '')}；AI 图文模式还建议点“测试图片能力”。")
        self._reset_test_states()

    def _on_test_failed(self, details: str, request_id: int):
        if request_id != int(getattr(self, "_test_request_id", 0)):
            return
        self.test_btn.setEnabled(True)
        self.test_vision_btn.setEnabled(True)
        lowered = str(details or "").lower()
        if "图片" in details or "image" in lowered or "nf42" in lowered:
            self._vision_test_state = "失败"
            self._remember_test_result("vision", "失败", details)
        else:
            self._text_test_state = "失败"
            self._remember_test_result("text", "失败", details)
        self._reset_test_states()
        self.status.setText("连接/图片能力测试失败")
        show_error_dialog(self,"AI 连接失败",details)

    def _save(self):
        error=self._validate(require_key=False)
        if error: QMessageBox.warning(self,"配置不完整",error); return
        from ai.config import api_key_storage_backend, save_ai_settings
        st=self._current()
        path=save_ai_settings(st, persist_api_key=self.persist_key.isChecked())
        if st.api_key and not self.persist_key.isChecked():
            self.status.setText("设置已保存；API Key 仅在本次程序运行期间保存在内存中")
        elif not st.api_key:
            self.status.setText("设置已保存；本地旧 API Key 已彻底移除")
        elif api_key_storage_backend() == "session-only":
            self.status.setText("设置已保存；当前平台无系统密钥库，API Key 已安全降级为仅本次运行")
        else:
            self.status.setText(f"已保存：{path}；API Key → {api_key_storage_backend()}")
        self.accept()


def ensure_ai_settings(parent, message: str):
    """确认 AI 连接已配置；未配置则提示并打开设置对话框。

    返回可用的 settings；用户取消或仍未配置时返回 None。
    Formatter 与文字校对共用这一入口，
    保证「配置后二次校验」在所有路径上一致。
    """
    from ai.config import load_ai_settings
    settings = load_ai_settings()
    if settings.configured:
        return settings
    QMessageBox.information(parent, "AI尚未配置", message)
    if AISettingsDialog(parent).exec() != QDialog.Accepted:
        return None
    settings = load_ai_settings()
    return settings if settings.configured else None

