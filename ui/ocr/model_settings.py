from __future__ import annotations

from functools import partial

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, QLabel,
    QCheckBox, QLineEdit, QPushButton, QRadioButton, QScrollArea, QFrame,
    QSizePolicy, QSpinBox, QTabWidget, QToolButton,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor, QAction

from ui.common.editor_controls import NoWheelComboBox, NoWheelDoubleSpinBox, NoWheelSpinBox
from ui.common.styling import BG, BORDER, CARD, CLICKABLE_BG, MUTED, ACC, ACC_TEXT, INK, TONAL, accent_button, make_badge, make_separator
from ui.responsive import configure_combo, preserve_button_text
from ui.widgets import ColorSwatch
from ui.design.components import DesignSwitch
from ui.design.metrics import OCR_LEFT_WIDTH, OCR_LEFT_MIN, OCR_LEFT_MAX, OCR_ENGINE_CARD_HEIGHT


def _build_reference_engine_panel(self, ocr_adapters):
    """Build the compact engine view from the supplied Phase-20 reference UI.

    The real controls stay alive in a hidden advanced container.  The visible
    switches below are synchronized proxies, so this changes presentation only
    and does not create a second OCR configuration path.
    """
    legacy = QWidget()
    legacy.setObjectName("ocrAdvancedEngineSettings")
    legacy_layout = QVBoxLayout(legacy)
    legacy_layout.setContentsMargins(0, 0, 0, 0)
    legacy_layout.setSpacing(5)
    def _adopt_layout_widgets(item):
        widget = item.widget()
        if widget is not None:
            widget.setParent(legacy)
            return
        layout = item.layout()
        if layout is not None:
            for index in range(layout.count()):
                _adopt_layout_widgets(layout.itemAt(index))

    while self._engine_settings_layout.count():
        item = self._engine_settings_layout.takeAt(0)
        _adopt_layout_widgets(item)
        legacy_layout.addItem(item)

    compact = QFrame()
    compact.setObjectName("ocrReferenceEnginePanel")
    compact.setStyleSheet(
        f"QFrame#ocrReferenceEnginePanel{{background:{CARD};border:none;border-radius:12px;}}"
    )
    box = QVBoxLayout(compact)
    box.setContentsMargins(4, 4, 4, 8)
    box.setSpacing(0)

    proxy_rows = [
        ("多模型 OCR 融合", self._multi_ocr_check),
        ("智能路由", self._multi_smart_router_check),
        ("物理分列", self._column_split_check),
        ("整句重识别", self._column_sentence_context_reocr_check),
    ]
    self._reference_engine_toggle_proxies = []
    for label_text, source in proxy_rows:
        row = QWidget()
        row.setObjectName("ocrReferenceToggleRow")
        row.setFixedHeight(36)
        row.setStyleSheet(
            f"QWidget#ocrReferenceToggleRow{{background:{CARD};border:none;border-bottom:1px solid {BORDER};}}"
            "QLabel{background:transparent;border:none;}"
        )
        rl = QHBoxLayout(row)
        rl.setContentsMargins(10, 8, 8, 8)
        rl.setSpacing(8)
        label = QLabel(label_text)
        label.setStyleSheet(f"color:{INK};font-size:11px;font-weight:600;border:none;")
        rl.addWidget(label, 1)
        proxy = DesignSwitch()
        proxy.setChecked(source.isChecked())
        proxy.setCursor(QCursor(Qt.PointingHandCursor))
        proxy.setToolTip(source.toolTip())
        def _from_proxy(checked, source=source, proxy=proxy):
            if source.isEnabled():
                source.setChecked(bool(checked))
            else:
                proxy.setChecked(source.isChecked())
        proxy.toggled.connect(_from_proxy)
        def _sync_proxy(checked, proxy=proxy):
            blocked = proxy.blockSignals(True)
            proxy.setChecked(bool(checked))
            proxy.blockSignals(blocked)
            proxy.set_progress(1.0 if checked else 0.0)
        source.toggled.connect(_sync_proxy)
        rl.addWidget(proxy, 0, Qt.AlignRight | Qt.AlignVCenter)
        box.addWidget(row)
        self._reference_engine_toggle_proxies.append((proxy, source))

    # Engine identity already lives in the top combobox, so the old Hayai/NDL
    # cards are kept only as hidden compatibility proxies for plugins/tests.
    # They are never inserted into the visible layout.
    def _hidden_engine_card(engine_id: str, title: str) -> QPushButton:
        card = QPushButton(title, compact)
        card.setObjectName("ocrReferenceEngineCard")
        card.setCheckable(True)
        idx = self._adapter_combo.findData(engine_id)
        card.setEnabled(idx >= 0)
        if idx >= 0:
            card.clicked.connect(lambda _checked=False, idx=idx: self._adapter_combo.setCurrentIndex(idx))
        card.hide()
        return card

    self._reference_engine_cards = {
        "hayai_ocr": _hidden_engine_card("hayai_ocr", "Hayai OCR"),
        "ndlocr_lite": _hidden_engine_card("ndlocr_lite", "NDLOCR-Lite"),
    }

    def _sync_hidden_engine_cards(*_args):
        active = str(self._adapter_combo.currentData() or "")
        for engine_id, card in self._reference_engine_cards.items():
            blocked = card.blockSignals(True)
            card.setChecked(engine_id == active)
            card.blockSignals(blocked)
    self._adapter_combo.currentIndexChanged.connect(_sync_hidden_engine_cards)
    _sync_hidden_engine_cards()


    # Advanced engine controls are now part of the normal page.  The old
    # “更多设置” disclosure created a second visual state and hid commonly used
    # controls, so the UI keeps one always-expanded configuration surface.
    legacy.setVisible(True)
    self._ocr_settings_tabs.setMinimumHeight(620)
    self._ocr_settings_tabs.setMaximumHeight(16777215)

    # Retain hidden compatibility attributes for older plugins/workspaces that
    # introspect these names; they no longer control visibility.
    advanced = QPushButton()
    advanced.setObjectName("ocrAdvancedEngineToggleCompat")
    advanced.setCheckable(True)
    advanced.setChecked(True)
    # Compatibility-only control: intentionally not inserted into any layout.
    # The removed “更多设置” entry therefore stays absent without hiding a live feature entry.
    advanced_action = QAction(compact)
    advanced_action.setCheckable(True)
    advanced_action.setChecked(True)
    advanced_action.setVisible(False)
    self._reference_engine_advanced_action = advanced_action

    self._engine_settings_layout.addWidget(compact)
    self._engine_settings_layout.addWidget(legacy)
    self._engine_settings_layout.addStretch(1)
    self._reference_engine_compact = compact
    self._reference_engine_advanced = legacy
    self._reference_engine_advanced_toggle = advanced



def build_ocr_model_settings(tab, top_row, ocr_adapters):
    # Legacy source-contract marker: ll.addWidget(mode_picker)
    # Legacy source-contract marker: ll.addWidget(engine_picker)
    # Legacy source-contract marker: ll.addWidget(self._ocr_settings_tabs)
    self = tab
    PAGE_STYLE = (f"QWidget#ocrSettingPage{{background:{CARD};border:1px solid {BORDER};"
                  "border-radius:14px;}")
    left_container = QWidget()
    left_container.setMinimumWidth(OCR_LEFT_MIN)
    left_container.setMaximumWidth(OCR_LEFT_MAX)
    left_container.setFixedWidth(OCR_LEFT_WIDTH)
    # 控制区保持稳定宽度，把更多横向空间留给页面与逐列预览。
    left_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
    left_container.setStyleSheet(f"background: {CARD}; border-right: 1px solid {BORDER};")
    left_outer = QVBoxLayout(left_container)
    left_outer.setContentsMargins(0, 0, 0, 0)
    left_outer.setSpacing(0)

    left = QWidget()
    left.setStyleSheet(f"background: {BG};")
    ll = QVBoxLayout(left)
    ll.setContentsMargins(15, 54, 13, 14)
    ll.setSpacing(13)
    left.setMinimumWidth(OCR_LEFT_MIN - 18)
    left.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)

    # OCR profile selector: Japanese vertical remains the untouched default;
    # Simplified Chinese horizontal owns an independent control snapshot and
    # never enters the Japanese column/ruby/handwriting pipeline.
    top_card = QFrame()
    top_card.setObjectName("ocrSettingCard")
    top_card.setStyleSheet(f"QFrame#ocrSettingCard{{background:{CARD};border:1px solid {BORDER};border-radius:14px;}}")
    top_card_layout = QVBoxLayout(top_card)
    top_card_layout.setContentsMargins(4, 12, 4, 6)
    top_card_layout.setSpacing(0)
    mode_picker = QWidget()
    mode_picker_layout = QVBoxLayout(mode_picker)
    mode_picker_layout.setContentsMargins(10, 0, 10, 8)
    mode_picker_layout.setSpacing(4)
    mode_label = QLabel("OCR 模式")
    mode_label.setStyleSheet(f"color: {MUTED}; font-size: 11px; font-weight: 600;")
    mode_picker_layout.addWidget(mode_label)
    self._ocr_mode_combo = NoWheelComboBox()
    self._ocr_mode_combo.addItem("日文竖排", "ja_vertical")
    self._ocr_mode_combo.addItem("简体中文横排", "zh_hans_horizontal")
    configure_combo(self._ocr_mode_combo)
    saved_mode = str(self._ocr_mode_settings.value("ocr/active_mode", "ja_vertical"))
    saved_index = self._ocr_mode_combo.findData(saved_mode)
    self._ocr_mode_combo.setCurrentIndex(max(0, saved_index))
    self._ocr_mode = str(self._ocr_mode_combo.currentData() or "ja_vertical")
    mode_picker_layout.addWidget(self._ocr_mode_combo)
    self._ocr_mode_summary = QLabel("")
    self._ocr_mode_summary.setWordWrap(True)
    self._ocr_mode_summary.setStyleSheet(
        f"color: {MUTED}; font-size: 10px; padding: 2px 2px 0;"
    )
    mode_picker_layout.addWidget(self._ocr_mode_summary)
    self._ocr_mode_summary.setVisible(False)
    top_card_layout.addWidget(mode_picker)

    # 旧版把所有引擎卡片和全部高级设置纵向堆叠，按钮很多且需要长距离滚动。
    # 新版保留同一批成员控件与调用链，只把入口压缩为一个引擎选择器，并按
    # “引擎 / 分列组句 / 逐字审校”分区，避免功能之间在视觉上互相混淆。
    engine_picker = QWidget()
    engine_picker_layout = QVBoxLayout(engine_picker)
    engine_picker_layout.setContentsMargins(10, 0, 10, 8)
    engine_picker_layout.setSpacing(4)
    engine_label = QLabel("识别引擎")
    engine_label.setStyleSheet(f"color: {MUTED}; font-size: 11px; font-weight: 600;")
    engine_picker_layout.addWidget(engine_label)
    self._adapter_combo = NoWheelComboBox()
    for aid, name, badge_text, _color, _desc, enabled in ocr_adapters:
        if enabled:
            # Keep the engine picker deliberately clean. Runtime/platform
            # notes still live in the descriptive text below, but the
            # selectable model name itself no longer carries Chinese badges
            # such as “跨平台 / 云端 / 图书OCR / 高速 CJK / 日文专用”.
            self._adapter_combo.addItem(name, aid)
    configure_combo(self._adapter_combo)
    engine_picker_layout.addWidget(self._adapter_combo)
    self._active_adapter_summary = QLabel("")
    self._active_adapter_summary.setWordWrap(True)
    self._active_adapter_summary.setStyleSheet(
        f"color: {MUTED}; font-size: 10px; padding: 2px 2px 0;"
    )
    engine_picker_layout.addWidget(self._active_adapter_summary)
    self._active_adapter_summary.setVisible(False)
    top_card_layout.addWidget(engine_picker)
    ll.addWidget(top_card)

    self._ocr_settings_tabs = QTabWidget()
    self._ocr_settings_tabs.setDocumentMode(True)
    self._ocr_settings_tabs.setTabPosition(QTabWidget.North)
    self._ocr_settings_tabs.tabBar().setDrawBase(False)
    self._ocr_settings_tabs.setUsesScrollButtons(False)
    self._ocr_settings_tabs.setElideMode(Qt.ElideNone)
    self._ocr_settings_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
    self._ocr_settings_tabs.setStyleSheet(
        f"QTabWidget{{background:{CARD};border:none;}}"
        f"QTabBar{{background:{CARD};border:none;}}"
        "QTabBar::tab { min-width: 82px; padding: 7px 10px; margin: 0; border-radius: 0; background: transparent; border: none; border-bottom: 1px solid #E2E5E9; }"
        f"QTabBar::tab:selected {{ background:{CARD}; color:{INK}; font-weight:700; border-bottom:2px solid #2563EB; }}"
        f"QTabBar::tab:hover {{ background:{CLICKABLE_BG}; }}"
        f"QTabWidget::pane {{ background:{CARD}; border:none; top:0px; }}"
    )
    # All three settings tabs are scrollable through the outer left pane and
    # show their complete configuration by default.
    self._ocr_settings_tabs.setMinimumHeight(620)
    self._ocr_settings_tabs.setMaximumHeight(16777215)
    engine_settings_page = QWidget()
    engine_settings_page.setMinimumWidth(350)
    engine_settings_page.setObjectName("ocrSettingPage")
    engine_settings_page.setStyleSheet(PAGE_STYLE)
    engine_settings_page.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
    self._engine_settings_layout = QVBoxLayout(engine_settings_page)
    self._engine_settings_layout.setContentsMargins(12, 12, 12, 12)
    self._engine_settings_layout.setSpacing(5)
    layout_settings_page = QWidget()
    layout_settings_page.setMinimumWidth(350)
    layout_settings_page.setObjectName("ocrSettingPage")
    layout_settings_page.setStyleSheet(PAGE_STYLE)
    layout_settings_page.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
    self._layout_settings_layout = QVBoxLayout(layout_settings_page)
    self._layout_settings_layout.setContentsMargins(12, 12, 12, 12)
    self._layout_settings_layout.setSpacing(5)
    review_settings_page = QWidget()
    review_settings_page.setMinimumWidth(350)
    review_settings_page.setObjectName("ocrSettingPage")
    review_settings_page.setStyleSheet(PAGE_STYLE)
    review_settings_page.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
    self._review_settings_layout = QVBoxLayout(review_settings_page)
    self._review_settings_layout.setContentsMargins(12, 12, 12, 12)
    self._review_settings_layout.setSpacing(5)
    self._ocr_settings_tabs.addTab(engine_settings_page, "引擎")
    self._ocr_settings_tabs.addTab(layout_settings_page, "分列与组句")
    self._ocr_settings_tabs.addTab(review_settings_page, "逐字审校")
    ll.addWidget(self._ocr_settings_tabs)

    self._adapter_cards: dict[str, QWidget] = {}
    for aid, name, badge_text, color, desc, enabled in ocr_adapters:
        card = QWidget()
        card.setCursor(QCursor(Qt.PointingHandCursor))
        card.setStyleSheet(
            f"background: {CLICKABLE_BG}; border: 1px solid #E2E5E9; border-radius: 12px; margin: 4px 10px;")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(10, 8, 10, 8)
        top_line = QHBoxLayout()
        dot = ColorSwatch(color)
        top_line.addWidget(dot)
        nlbl = QLabel(name)
        nlbl.setStyleSheet(f"font-weight: bold; font-size: 12px;")
        top_line.addWidget(nlbl)
        top_line.addStretch()
        top_line.addWidget(make_badge(badge_text, color if enabled else "#8C8B84"))
        cl.addLayout(top_line)
        dl = QLabel(desc)
        dl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        dl.setWordWrap(True)
        cl.addWidget(dl)
        if not enabled:
            na = QLabel("即将支持")
            na.setStyleSheet(f"color: #5F7189; font-size: 10px; font-style: italic;")
            na.setAlignment(Qt.AlignRight)
            cl.addWidget(na)
        card.mousePressEvent = partial(self._select_adapter, aid, enabled)
        # 保留卡片对象以兼容原高亮/插件调用，但不再把全部卡片同时显示。
        card.setVisible(False)
        self._adapter_cards[aid] = card

    self._adapter_combo.currentIndexChanged.connect(self._select_adapter_from_combo)

    self._multi_ocr_widget = QWidget()
    multi_layout = QVBoxLayout(self._multi_ocr_widget)
    multi_layout.setContentsMargins(10, 8, 10, 8)
    multi_layout.setSpacing(5)
    self._multi_ocr_check = QCheckBox("多模型 OCR（角色分工 · 4 模型）")
    self._multi_ocr_check.setToolTip(
        "开启后，上方单模型选择会锁定且不参与本次运行。四个槽位分别定义证据粒度："
        "逐列主模型 / 整页主模型 / 全列主模型 / 分歧复核模型。同一个 OCR 模型只能占用一个角色；"
        "同模型分歧确认由内部定向重试完成，不重复占用槽位。每个角色仍拥有独立的输入粒度、缓存身份与证据来源。"
        "共享 page/column/sentence 几何，"
        "最终送入识别器的像素 transport 由角色与模型专用 Profile 共同决定。"
    )
    multi_layout.addWidget(self._multi_ocr_check)

    self._multi_ocr_speed_note = QLabel(
        "✓ 角色实验：Hayai 逐列 · NDL 整页 · 48px 可全列 · Hayai/48px 可仅复核分歧"
    )
    self._multi_ocr_speed_note.setStyleSheet(
        "color:#5B6B80;font-size:10px;background:#F7F8FA;border:1px solid #E2E5E9;"
        "border-radius:6px;padding:5px 7px;"
    )
    self._multi_ocr_speed_note.setToolTip(
        "同一页面与物理列身份由共享几何保证；不同 OCR 可使用整页或模型优化后的单列输入图像。"
        "第三主模型槽完整读取全部物理列；分歧复核槽只读取残余冲突列，可直接比较全列识别与按需复核的时间和质量。"
    )
    multi_layout.addWidget(self._multi_ocr_speed_note)

    self._multi_smart_router_check = QCheckBox("旧整句智能路由（兼容状态）")
    self._multi_smart_router_check.setChecked(False)
    self._multi_smart_router_check.setToolTip(
        "Phase27 起第三主模型槽已改为全列主模型，固定完整读取所有物理列；"
        "此旧开关仅保留工作区兼容，不再影响执行。"
    )
    self._multi_smart_router_check.setVisible(False)
    multi_layout.addWidget(self._multi_smart_router_check)

    self._multi_local_retry_check = QCheckBox("主模型多数确认重试（可选）")
    self._multi_local_retry_check.setChecked(False)
    self._multi_local_retry_check.setToolTip(
        "仅对已有独立模型严格多数的疑难句做低成本确认重读；同模型重试不增加票数。"
        "默认关闭，避免再次出现整本级二次 OCR。"
    )
    self._multi_local_retry_check.setStyleSheet("font-size:10px;spacing:4px;font-weight:600;")
    self._multi_local_retry_check.setVisible(False)
    multi_layout.addWidget(self._multi_local_retry_check)

    from core.multi_ocr_roles import ROLE_LABELS, ROLE_ORDER
    self._multi_role_combos = {}
    self._multi_model_combos = []  # compatibility alias for older plugins/tests
    for role_index, role in enumerate(ROLE_ORDER):
        row = QHBoxLayout()
        ui_role_label = {
            "column": "逐列主模型",
            "page": "整页主模型",
            "sentence": "全列主模型",
            "review1": "分歧复核模型",
        }.get(role, ROLE_LABELS[role])
        label = QLabel(ui_role_label)
        label.setMinimumWidth(82)
        label.setStyleSheet(f"color: {MUTED}; font-size: 11px; font-weight: 600;")
        row.addWidget(label)
        combo = NoWheelComboBox()
        combo.addItem("未选择", "")
        for engine_id, engine_name, _badge, _color, _desc, enabled in ocr_adapters:
            if enabled:
                combo.addItem(engine_name, engine_id)
        # Manga OCR is deliberately review-only: it must never become a normal
        # single/page/full-column primary model. Its 224x224 recognizer is fed
        # short blocks derived only from disagreement physical columns.
        if role.startswith("review"):
            combo.addItem("Manga OCR", "manga_ocr")
        configure_combo(combo)
        combo.setEnabled(False)
        combo.setProperty("multi_ocr_role", role)
        combo.currentIndexChanged.connect(self._on_multi_ocr_role_changed)
        row.addWidget(combo, 1)
        multi_layout.addLayout(row)
        # New runs expose exactly one disagreement-review model.  review2/3
        # stay in the schema solely so older workspaces remain recoverable.
        if role in {"review2", "review3"}:
            label.setVisible(False)
            combo.setVisible(False)
        self._multi_role_combos[role] = combo
        self._multi_model_combos.append(combo)

    # Backwards attribute aliases.  Their meaning is now role based rather
    # than positional model voting: model1..3 = first-column/page/full-column,
    # model4..6 = disagreement review 1..3.  The third slot keeps the historical
    # ``sentence`` persistence key only for workspace compatibility.
    self._multi_model1_combo = self._multi_role_combos["column"]
    self._multi_model2_combo = self._multi_role_combos["page"]
    self._multi_model3_combo = self._multi_role_combos["sentence"]
    self._multi_model4_combo = self._multi_role_combos["review1"]
    self._multi_model5_combo = self._multi_role_combos["review2"]
    self._multi_model6_combo = self._multi_role_combos["review3"]

    # Legacy compatibility controls remain as hidden state only.  The new
    # role scheduler always performs disagreement-review early exit, so the
    # old column-level quick-consensus / first-two parallel controls no
    # longer define execution semantics.
    self._multi_early_consensus_check = QCheckBox()
    self._multi_early_consensus_check.setChecked(False)
    self._multi_early_consensus_check.setVisible(False)
    self._multi_parallel_first_check = QCheckBox()
    self._multi_parallel_first_check.setChecked(True)
    self._multi_parallel_first_check.setVisible(False)
    self._multi_sentence_speed_combo = NoWheelComboBox()
    self._multi_sentence_speed_combo.addItem("角色调度", "role_scheduler")
    self._multi_sentence_speed_combo.setVisible(False)

    multi_hint = QLabel(
        "共享几何只定义同一页、同一列、同一句和原图坐标；最终 OCR 图片不强制相同。"
        "角色决定识别粒度，Engine Profile 决定该模型的 padding/resize/整页或短块 transport。"
    )
    multi_hint.setWordWrap(True)
    multi_hint.setStyleSheet(f"color: #5F7189; font-size: 10px;")
    multi_layout.addWidget(multi_hint)

    self._multi_ocr_check.toggled.connect(self._on_multi_ocr_toggled)
    self._update_multi_ocr_option_state()
    self._engine_settings_layout.addWidget(self._multi_ocr_widget)

    self._engine_settings_layout.addWidget(make_separator())

    # Apple OCR uses explicit backends.  Live Text is the default; the
    # original Shortcuts route remains available as the last option.
    self._vision_backend_widget = QWidget()
    vbw = QVBoxLayout(self._vision_backend_widget)
    vbw.setContentsMargins(10, 0, 10, 0)
    vbw.setSpacing(2)
    vb_lbl = QLabel("识别方式")
    vb_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
    vbw.addWidget(vb_lbl)
    self._vision_backend_combo = NoWheelComboBox()
    self._vision_backend_combo.addItem("Apple Live Text · VisionKit ImageAnalyzer", "live_text")
    self._vision_backend_combo.addItem("Apple Vision · RecognizeTextRequest 坐标/候选", "native_helper")
    self._vision_backend_combo.addItem("macOS 快捷指令 · 稳定兼容通道", "shortcut")
    self._vision_backend_combo.setCurrentIndex(0)
    self._vision_backend_combo.currentIndexChanged.connect(self._on_vision_backend_changed)
    vbw.addWidget(self._vision_backend_combo)
    self._vision_vertical_check = QCheckBox("按日文竖排顺序组合观察结果（右→左、上→下）")
    self._vision_vertical_check.setChecked(True)
    vbw.addWidget(self._vision_vertical_check)
    vb_hint = QLabel("")
    vb_hint.setStyleSheet(f"color: #5F7189; font-size: 10px;")
    vb_hint.setWordWrap(True)
    vbw.addWidget(vb_hint)
    vb_hint.setVisible(False)
    self._vision_backend_hint = vb_hint
    self._engine_settings_layout.addWidget(self._vision_backend_widget)

    self._vision_helper_widget = QWidget()
    vh_form = QFormLayout(self._vision_helper_widget)
    vh_form.setContentsMargins(10, 4, 10, 0)
    vh_form.setSpacing(4)
    self._vision_language_correction_check = QCheckBox("启用日语语言校正")
    self._vision_language_correction_check.setChecked(True)
    vh_form.addRow(self._vision_language_correction_check)
    self._vision_candidate_spin = NoWheelSpinBox()
    self._vision_candidate_spin.setRange(1, 10)
    self._vision_candidate_spin.setValue(3)
    vh_form.addRow("候选数量", self._vision_candidate_spin)
    self._vision_min_height_spin = NoWheelDoubleSpinBox()
    self._vision_min_height_spin.setRange(0.0, 0.1)
    self._vision_min_height_spin.setDecimals(4)
    self._vision_min_height_spin.setSingleStep(0.001)
    self._vision_min_height_spin.setValue(0.005)
    vh_form.addRow("最小文字高度比例", self._vision_min_height_spin)
    self._vision_vertical_compat_check = QCheckBox("竖排列紧裁后左旋 90°识别（单次 OCR，推荐）")
    self._vision_vertical_compat_check.setChecked(True)
    self._vision_vertical_compat_check.setToolTip(
        "仅在输入图实际只保留一个狭长竖列时启用：先紧裁文字区域，再左旋成横排送入 Vision；"
        "不会增加 OCR 次数，返回坐标会映射回原图。"
    )
    vh_form.addRow(self._vision_vertical_compat_check)
    self._engine_settings_layout.addWidget(self._vision_helper_widget)

    self._vision_live_text_widget = QWidget()
    vlt = QVBoxLayout(self._vision_live_text_widget)
    vlt.setContentsMargins(10, 4, 10, 0)
    vlt.setSpacing(2)
    live_text_note = QLabel(
        "Live Text 直接分析原尺寸分列掩膜图并读取系统 transcript；不旋转、不紧裁、"
        "不重复 OCR。该通道不提供逐字置信度、候选或文字框。"
    )
    live_text_note.setWordWrap(True)
    live_text_note.setStyleSheet(f"color: #5F7189; font-size: 10px;")
    vlt.addWidget(live_text_note)
    self._engine_settings_layout.addWidget(self._vision_live_text_widget)
    self._vision_live_text_widget.setVisible(False)

    # 原快捷指令名称保留为显式通道。
    self._shortcut_widget = QWidget()
    sc_col = QVBoxLayout(self._shortcut_widget)
    sc_col.setContentsMargins(10, 4, 10, 0)
    sc_col.setSpacing(2)
    sc_lbl = QLabel("快捷指令名称")
    sc_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
    sc_col.addWidget(sc_lbl)
    self._shortcut_edit = QLineEdit("ExtractText")
    sc_col.addWidget(self._shortcut_edit)
    self._engine_settings_layout.addWidget(self._shortcut_widget)
    self._on_vision_backend_changed()

    # PaddleOCR AI Studio cloud API. Credentials never enter project/mode JSON.
    # The adapter intentionally consumes whole pages rather than physical
    # column crops, otherwise one remote page could fan out into 10-20 API
    # calls and lose the service's own layout context.
    self._paddle_aistudio_widget = QWidget()
    self._paddle_aistudio_widget.setVisible(False)
    pav = QVBoxLayout(self._paddle_aistudio_widget)
    pav.setContentsMargins(10, 8, 10, 0)
    pav.setSpacing(4)

    pa_mode_row = QHBoxLayout()
    pa_mode_lbl = QLabel("AI Studio 接口模式")
    pa_mode_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
    pa_mode_row.addWidget(pa_mode_lbl)
    self._paddle_aistudio_mode_combo = NoWheelComboBox()
    self._paddle_aistudio_mode_combo.addItem("官方异步 v2 jobs（推荐）", "async_v2")
    self._paddle_aistudio_mode_combo.addItem("同步 API（兼容旧模型页 URL）", "sync")
    pa_mode_row.addWidget(self._paddle_aistudio_mode_combo, 1)
    pav.addLayout(pa_mode_row)

    pa_url_lbl = QLabel("同步 API URL")
    pa_url_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
    pav.addWidget(pa_url_lbl)
    self._paddle_aistudio_url_edit = QLineEdit()
    self._paddle_aistudio_url_edit.setPlaceholderText(
        "从 aistudio.baidu.com/paddleocr/task → API 调用示例复制完整 https:// URL"
    )
    pav.addWidget(self._paddle_aistudio_url_edit)

    pa_sync_family_row = QHBoxLayout()
    pa_sync_family_lbl = QLabel("同步服务类型")
    pa_sync_family_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
    pa_sync_family_row.addWidget(pa_sync_family_lbl)
    self._paddle_aistudio_sync_family_combo = NoWheelComboBox()
    self._paddle_aistudio_sync_family_combo.addItem("PP-OCR（v5/v6）", "ppocr")
    self._paddle_aistudio_sync_family_combo.addItem("PP-StructureV3", "ppstructure")
    self._paddle_aistudio_sync_family_combo.addItem("PaddleOCR-VL", "vl")
    pa_sync_family_row.addWidget(self._paddle_aistudio_sync_family_combo, 1)
    pav.addLayout(pa_sync_family_row)

    pa_token_lbl = QLabel("AI Studio Access Token")
    pa_token_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
    pav.addWidget(pa_token_lbl)
    pa_token_row = QHBoxLayout()
    self._paddle_aistudio_token_edit = QLineEdit()
    self._paddle_aistudio_token_edit.setEchoMode(QLineEdit.EchoMode.Password)
    self._paddle_aistudio_token_edit.setPlaceholderText(
        "也可使用 PADDLEOCR_ACCESS_TOKEN；兼容 AISTUDIO_ACCESS_TOKEN"
    )
    pa_token_row.addWidget(self._paddle_aistudio_token_edit, 1)
    self._paddle_aistudio_save_token_btn = QPushButton("保存到系统")
    self._paddle_aistudio_save_token_btn.setToolTip("macOS Keychain / Windows DPAPI；Linux 不做明文持久化")
    pa_token_row.addWidget(self._paddle_aistudio_save_token_btn)
    self._paddle_aistudio_clear_token_btn = QPushButton("清除")
    pa_token_row.addWidget(self._paddle_aistudio_clear_token_btn)
    pav.addLayout(pa_token_row)

    pa_model_row = QHBoxLayout()
    pa_model_lbl = QLabel("异步 v2 模型")
    pa_model_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
    pa_model_row.addWidget(pa_model_lbl)
    self._paddle_aistudio_async_model_combo = NoWheelComboBox()
    self._paddle_aistudio_async_model_combo.addItem("PaddleOCR-VL-1.6（官方文档解析，推荐）", "PaddleOCR-VL-1.6")
    self._paddle_aistudio_async_model_combo.addItem("PP-OCRv6（官方 OCR）", "PP-OCRv6")
    self._paddle_aistudio_async_model_combo.addItem("PP-OCRv5", "PP-OCRv5")
    self._paddle_aistudio_async_model_combo.addItem("PP-OCRv5-latin（拉丁文字）", "PP-OCRv5-latin")
    self._paddle_aistudio_async_model_combo.addItem("PP-StructureV3", "PP-StructureV3")
    self._paddle_aistudio_async_model_combo.addItem("PaddleOCR-VL-1.5", "PaddleOCR-VL-1.5")
    self._paddle_aistudio_async_model_combo.addItem("PaddleOCR-VL", "PaddleOCR-VL")
    pa_model_row.addWidget(self._paddle_aistudio_async_model_combo, 1)
    pav.addLayout(pa_model_row)

    pa_flags_row = QHBoxLayout()
    self._paddle_aistudio_orientation_check = QCheckBox("文档方向校正")
    self._paddle_aistudio_unwarp_check = QCheckBox("文档去畸变")
    self._paddle_aistudio_textline_check = QCheckBox("文字行方向")
    pa_flags_row.addWidget(self._paddle_aistudio_orientation_check)
    pa_flags_row.addWidget(self._paddle_aistudio_unwarp_check)
    pa_flags_row.addWidget(self._paddle_aistudio_textline_check)
    pa_flags_row.addStretch(1)
    pav.addLayout(pa_flags_row)

    pa_limits_row = QHBoxLayout()
    pa_timeout_lbl = QLabel("单请求超时")
    pa_timeout_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
    pa_limits_row.addWidget(pa_timeout_lbl)
    self._paddle_aistudio_timeout_spin = QSpinBox()
    self._paddle_aistudio_timeout_spin.setRange(30, 900)
    self._paddle_aistudio_timeout_spin.setValue(180)
    self._paddle_aistudio_timeout_spin.setSuffix(" 秒")
    pa_limits_row.addWidget(self._paddle_aistudio_timeout_spin)
    pa_retry_lbl = QLabel("重试")
    pa_retry_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
    pa_limits_row.addWidget(pa_retry_lbl)
    self._paddle_aistudio_retry_spin = QSpinBox()
    self._paddle_aistudio_retry_spin.setRange(0, 6)
    self._paddle_aistudio_retry_spin.setValue(3)
    self._paddle_aistudio_retry_spin.setSuffix(" 次")
    pa_limits_row.addWidget(self._paddle_aistudio_retry_spin)
    pa_limits_row.addStretch(1)
    pav.addLayout(pa_limits_row)

    pa_hint = QLabel(
        "默认使用官方异步 v2 jobs：本地文件 multipart 上传、Bearer Token、jobId 轮询和 JSONL 结果下载；默认模型为 PaddleOCR-VL-1.6。"
        "同步模式的 URL 与 AI Studio 任务页模型绑定；请匹配选择 PP-OCR、PP-StructureV3 或 PaddleOCR-VL。"
        "异步 v2 按官方模型能力自动裁剪参数；PP-StructureV3 与 PP-OCR 支持文字行方向，VL 系列自动省略。"
        "云端引擎固定整页调用，不会把分栏 ROI 单独上传。"
    )
    pa_hint.setStyleSheet("color: #5F7189; font-size: 10px;")
    pa_hint.setWordWrap(True)
    pav.addWidget(pa_hint)
    self._engine_settings_layout.addWidget(self._paddle_aistudio_widget)
    self._paddle_aistudio_mode_combo.currentIndexChanged.connect(self._update_paddle_aistudio_option_state)
    self._paddle_aistudio_sync_family_combo.currentIndexChanged.connect(self._update_paddle_aistudio_option_state)
    self._paddle_aistudio_async_model_combo.currentIndexChanged.connect(self._update_paddle_aistudio_option_state)
    self._paddle_aistudio_save_token_btn.clicked.connect(self._save_paddle_aistudio_token)
    self._paddle_aistudio_clear_token_btn.clicked.connect(self._clear_paddle_aistudio_token)
    self._load_paddle_aistudio_token()
    self._update_paddle_aistudio_option_state()

    # Hayai OCR：crop recognizer；页面布局仍由 Formatter 管理。
    self._hayai_widget = QWidget()
    self._hayai_widget.setVisible(False)
    hyv = QVBoxLayout(self._hayai_widget)
    hyv.setContentsMargins(10, 8, 10, 0)
    hyv.setSpacing(4)
    hy_backend_row = QHBoxLayout()
    hy_backend_lbl = QLabel("Hayai 后端")
    hy_backend_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
    hy_backend_row.addWidget(hy_backend_lbl)
    self._hayai_backend_combo = NoWheelComboBox()
    self._hayai_backend_combo.addItem("PyTorch（推荐：CUDA / MPS / CPU）", "torch")
    self._hayai_backend_combo.addItem("LiteRT（CPU / 边缘端，实验）", "litert")
    self._hayai_backend_combo.currentIndexChanged.connect(self._update_hayai_option_state)
    hy_backend_row.addWidget(self._hayai_backend_combo, 1)
    hyv.addLayout(hy_backend_row)

    hy_device_row = QHBoxLayout()
    hy_device_lbl = QLabel("运行设备")
    hy_device_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
    hy_device_row.addWidget(hy_device_lbl)
    self._hayai_device_combo = NoWheelComboBox()
    self._hayai_device_combo.addItem("自动（CUDA → MPS → CPU）", "auto")
    self._hayai_device_combo.addItem("CUDA", "cuda")
    self._hayai_device_combo.addItem("Apple MPS", "mps")
    self._hayai_device_combo.addItem("CPU", "cpu")
    hy_device_row.addWidget(self._hayai_device_combo, 1)
    hyv.addLayout(hy_device_row)

    hy_quant_row = QHBoxLayout()
    hy_quant_lbl = QLabel("Torch 量化")
    hy_quant_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
    hy_quant_row.addWidget(hy_quant_lbl)
    self._hayai_quant_combo = NoWheelComboBox()
    self._hayai_quant_combo.addItem("不量化（默认 / 最高兼容）", "none")
    self._hayai_quant_combo.addItem("INT8 weight-only（约 2× 降显存）", "int8")
    self._hayai_quant_combo.addItem("INT4 weight-only（约 4× 降显存）", "int4")
    hy_quant_row.addWidget(self._hayai_quant_combo, 1)
    hyv.addLayout(hy_quant_row)

    hy_litert_row = QHBoxLayout()
    hy_litert_lbl = QLabel("LiteRT 量化")
    hy_litert_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
    hy_litert_row.addWidget(hy_litert_lbl)
    self._hayai_litert_quant_combo = NoWheelComboBox()
    self._hayai_litert_quant_combo.addItem("WI4（默认）", "wi4")
    self._hayai_litert_quant_combo.addItem("WI8 AFP32", "wi8_afp32")
    self._hayai_litert_quant_combo.addItem("Dynamic WI4", "dynamic_wi4")
    self._hayai_litert_quant_combo.addItem("Dynamic WI8", "dynamic_wi8")
    self._hayai_litert_quant_combo.addItem("Float", "none")
    hy_litert_row.addWidget(self._hayai_litert_quant_combo, 1)
    hyv.addLayout(hy_litert_row)

    hy_hint = QLabel(
        "Hayai 只负责已经分离的文字 crop，不做整页文字检测。Novel Formatter 会复用固定正文框、"
        "右→左物理列、Ruby 清理、共享 crop 与多模型融合；每段默认约 24 字，减少不必要的切段调用。"
    )
    hy_hint.setWordWrap(True)
    hy_hint.setStyleSheet(f"color: #5F7189; font-size: 10px;")
    hyv.addWidget(hy_hint)
    self._engine_settings_layout.addWidget(self._hayai_widget)
    self._update_hayai_option_state()

    # PaddleOCR 模型选择（只在选中 paddle_ocr 适配器时显示）
    self._paddle_model_widget = QWidget()
    self._paddle_model_widget.setVisible(False)
    pmv = QVBoxLayout(self._paddle_model_widget)
    pmv.setContentsMargins(10, 8, 10, 0)
    pmv.setSpacing(2)
    pm_lbl = QLabel("Paddle 模型")
    pm_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
    pmv.addWidget(pm_lbl)
    self._paddle_model_combo = NoWheelComboBox()
    self._paddle_model_combo.addItem("PaddleOCR（日文印刷体；优先 v6 medium）", "ocr")
    self._paddle_model_combo.addItem("PP-StructureV3（复杂版面与区域分析）", "structure")
    self._paddle_model_combo.addItem("PaddleOCR-VL-1.6（Apple Silicon 可用官方 MLX 加速）", "vl")
    pmv.addWidget(self._paddle_model_combo)
    vl_backend_row = QHBoxLayout()
    vl_backend_lbl = QLabel("VL 后端")
    vl_backend_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
    vl_backend_row.addWidget(vl_backend_lbl)
    self._paddle_vl_backend_combo = NoWheelComboBox()
    self._paddle_vl_backend_combo.addItem(
        "自动（Apple Silicon 优先官方 MLX；失败回退 Paddle）", "auto"
    )
    self._paddle_vl_backend_combo.addItem("MLX-VLM（Apple Silicon 官方后端）", "mlx")
    self._paddle_vl_backend_combo.addItem("Paddle（兼容后端）", "paddle")
    self._paddle_vl_backend_combo.setEnabled(False)
    self._paddle_vl_backend_combo.setToolTip(
        "仅对 PaddleOCR-VL-1.6 生效。自动模式在 Apple Silicon 上通过 PaddleOCR "
        "官方 mlx-vlm-server 接口使用 Apple GPU；MLX 安装/启动或实际推理失败时自动回退原生 Paddle。"
    )
    vl_backend_row.addWidget(self._paddle_vl_backend_combo, 1)
    pmv.addLayout(vl_backend_row)
    self._paddle_model_combo.currentIndexChanged.connect(self._update_paddle_vl_backend_state)
    source_row = QHBoxLayout()
    source_lbl = QLabel("模型下载源")
    source_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
    source_row.addWidget(source_lbl)
    self._paddle_source_combo = NoWheelComboBox()
    self._paddle_source_combo.addItem("自动重试（HF → ModelScope → BOS → AIStudio）", "auto")
    self._paddle_source_combo.addItem("ModelScope", "modelscope")
    self._paddle_source_combo.addItem("百度 BOS", "bos")
    self._paddle_source_combo.addItem("Hugging Face", "huggingface")
    self._paddle_source_combo.addItem("AIStudio", "aistudio")
    self._paddle_source_combo.setToolTip(
        "PaddleOCR 3.x 默认从 Hugging Face 下载。若当前网络无法访问，"
        "可改用 ModelScope 或百度 BOS；自动模式只在模型初始化阶段切换来源，不会重复 OCR 页面。"
        "PaddleOCR-VL 的 MLX-VLM 模型由官方 MLX 路径使用 Hugging Face 模型 ID 加载，此选项主要控制 Paddle 客户端/版面模型来源。"
    )
    source_row.addWidget(self._paddle_source_combo, 1)
    pmv.addLayout(source_row)

    paddle_action_row = QHBoxLayout()
    self._paddle_prepare_btn = QPushButton("安装 / 修复模型")
    self._paddle_prepare_btn.setToolTip("先创建独立环境并下载/初始化当前 Paddle 模型，不开始整本 OCR。")
    self._paddle_prepare_btn.clicked.connect(self._prepare_paddle_runtime)
    paddle_action_row.addWidget(self._paddle_prepare_btn)
    self._paddle_check_btn = QPushButton("检查状态")
    self._paddle_check_btn.clicked.connect(self._check_paddle_runtime)
    paddle_action_row.addWidget(self._paddle_check_btn)
    pmv.addLayout(paddle_action_row)

    self._paddle_runtime_status = QLabel("尚未检查 PaddleOCR 环境")
    self._paddle_runtime_status.setWordWrap(True)
    self._paddle_runtime_status.setStyleSheet(f"color: {MUTED}; font-size: 10px;")
    pmv.addWidget(self._paddle_runtime_status)
    pm_hint = QLabel("PP-OCRv6 / PP-Structure 保持原有 Paddle 路径。PaddleOCR-VL-1.6 在 Apple Silicon 上默认通过官方 MLX-VLM 后端加速，并使用独立环境避免依赖冲突；MLX 不可用或页级推理失败会自动回退 Paddle。")
    pm_hint.setStyleSheet(f"color: #5F7189; font-size: 10px;")
    pm_hint.setWordWrap(True)
    pmv.addWidget(pm_hint)
    self._engine_settings_layout.addWidget(self._paddle_model_widget)

    # Simplified-Chinese horizontal settings live in their own widget.
    # The widget is hidden in Japanese mode and contains no vertical-column
    # controls, so both execution paths remain structurally independent.
    self._chinese_horizontal_widget = QFrame()
    self._chinese_horizontal_widget.setObjectName("chineseHorizontalOcrCard")
    self._chinese_horizontal_widget.setStyleSheet(
        "QFrame#chineseHorizontalOcrCard { background: #F7F8FA; "
        "border: 1px solid #E2E5E9; border-radius: 10px; margin: 4px 10px; }"
    )
    chs_layout = QVBoxLayout(self._chinese_horizontal_widget)
    chs_layout.setContentsMargins(12, 10, 12, 10)
    chs_layout.setSpacing(6)
    chs_title = QLabel("简体中文横排版面")
    chs_title.setStyleSheet("font-size: 12px; font-weight: 700; color: #1C47B8;")
    chs_layout.addWidget(chs_title)
    chs_note = QLabel(
        "整页按从上到下、从左到右读取；不执行日文分列、Ruby 过滤、竖列旋转、"
        "逐列成句或日语手写识别。模式切换不会修改日文竖排设置。"
    )
    chs_note.setWordWrap(True)
    chs_note.setStyleSheet(f"color: {MUTED}; font-size: 10px;")
    chs_layout.addWidget(chs_note)
    self._chinese_merge_line_fragments_check = QCheckBox("合并同一横行内被拆开的文字框")
    self._chinese_merge_line_fragments_check.setChecked(True)
    self._chinese_merge_line_fragments_check.setToolTip(
        "仅合并纵向重叠且基线接近的相邻框，按左到右排序；不会跨行或跨段合并。"
    )
    chs_layout.addWidget(self._chinese_merge_line_fragments_check)
    self._chinese_header_filter_check = QCheckBox("自动过滤跨页重复页眉 / 页脚")
    self._chinese_header_filter_check.setChecked(True)
    self._chinese_header_filter_check.setToolTip(
        "按多页重复频率过滤短页眉页脚。与日文逐列成句的保全策略相互独立。"
    )
    chs_layout.addWidget(self._chinese_header_filter_check)
    self._layout_settings_layout.addWidget(self._chinese_horizontal_widget)

    # 单模型分列开关。多模型使用独立角色路由与共享几何，不能改写此状态。
    self._column_ocr_widget = QWidget()
    cov = QVBoxLayout(self._column_ocr_widget)
    cov.setContentsMargins(10, 10, 10, 0)
    cov.setSpacing(4)
    self._column_split_check = QCheckBox("启用日文物理分列")
    self._column_split_check.setChecked(False)
    self._column_split_check.setToolTip(
        "单模型专用：关闭时把原页直接交给所选 OCR 引擎；开启时按日文物理竖列识别。"
        "极窄的已裁单列图会自动按单列处理，不会再被拆成多个假列。"
    )
    self._column_split_check.toggled.connect(self._on_column_split_toggled)
    cov.addWidget(self._column_split_check)
    self._column_settings_widget = QWidget()
    settings = QVBoxLayout(self._column_settings_widget)
    settings.setContentsMargins(0, 2, 0, 0)
    settings.setSpacing(3)
    isolation_row = QHBoxLayout()
    isolation_lbl = QLabel("分列输入")
    isolation_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
    isolation_row.addWidget(isolation_lbl)
    self._column_isolation_mode_combo = NoWheelComboBox()
    self._column_isolation_mode_combo.addItem(
        "分列掩膜（推荐）：按模型裁白底视窗", "mask"
    )
    self._column_isolation_mode_combo.addItem(
        "分列显示：保留原页位置预览，OCR仍用单列输入", "display"
    )
    self._column_isolation_mode_combo.setCurrentIndex(0)
    self._column_isolation_mode_combo.setToolTip(
        "两种模式使用完全相同的正文/Ruby几何：\n"
        "• 分列掩膜：先生成原页白色隔离层，再裁成各 OCR 适合的宽上下文/紧凑视窗；效率更高。\n"
        "• 分列显示：保持原页尺寸，其余区域全部纸白，只开放当前正文列；不缩放、不混入 Ruby/相邻列。\n"
        "选择“分列显示”时，NDLOCR 也按逐列运行，以保证每次只开放一列。"
    )
    isolation_row.addWidget(self._column_isolation_mode_combo, 1)
    settings.addLayout(isolation_row)
    sens_row = QHBoxLayout()
    sens_lbl = QLabel("分列灵敏度")
    sens_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
    sens_row.addWidget(sens_lbl)
    self._column_sensitivity_spin = NoWheelSpinBox()
    self._column_sensitivity_spin.setRange(1, 100)
    self._column_sensitivity_spin.setValue(55)
    sens_row.addWidget(self._column_sensitivity_spin, 1)
    settings.addLayout(sens_row)
    pad_row = QHBoxLayout()
    pad_lbl = QLabel("列保护边距")
    pad_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
    pad_row.addWidget(pad_lbl)
    self._column_padding_spin = NoWheelSpinBox()
    self._column_padding_spin.setRange(0, 30)
    self._column_padding_spin.setValue(10)
    self._column_padding_spin.setSuffix(" %")
    pad_row.addWidget(self._column_padding_spin, 1)
    settings.addLayout(pad_row)

    # v8-compatible OCR input is the hidden default.  The normal UI does
    # not expose internal profile names; only explicit advanced changes
    # create a custom preprocessing contract.
    self._preserve_ruby_check = QCheckBox("保留原文 Ruby（findtextCenterNet）")
    self._preserve_ruby_check.setChecked(False)
    self._preserve_ruby_check.setToolTip(
        "默认关闭，因此不会下载模型、不会增加 OCR 时间。开启后普通 OCR 仍只识别分列正文，"
        "并强制从其临时列图中排除 Ruby；只有 findtextCenterNet 单独读取原始页面提取 "
        "rubybase ↔ 振假名关系。两条证据通道完全独立，findtextCenterNet 不参与正文多数票或字符融合。"
    )
    self._preserve_ruby_check.toggled.connect(self._on_preserve_ruby_toggled)
    cov.addWidget(self._preserve_ruby_check)

    ruby_scan_row = QHBoxLayout()
    ruby_scan_lbl = QLabel("Ruby 扫描范围")
    ruby_scan_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
    ruby_scan_row.addWidget(ruby_scan_lbl)
    self._ruby_scan_mode_combo = NoWheelComboBox()
    self._ruby_scan_mode_combo.addItem("智能 ROI（推荐）", "smart_roi")
    self._ruby_scan_mode_combo.addItem("全页精确扫描（慢）", "full_page")
    self._ruby_scan_mode_combo.setCurrentIndex(0)
    self._ruby_scan_mode_combo.setEnabled(False)
    self._ruby_scan_mode_combo.setToolTip(
        "智能 ROI 会复用普通 OCR 分列时顺手记录的疑似 Ruby 几何，只从未清理原图裁取相邻几列上下文交给 "
        "findtextCenterNet；没有候选的页面不会再次 OCR。全页模式仅用于漏检诊断。"
    )
    ruby_scan_row.addWidget(self._ruby_scan_mode_combo, 1)
    cov.addLayout(ruby_scan_row)

    # “分列与组句”不再使用高级设置折叠层。保留一个隐藏的兼容开关，
    # 但页面始终直接展示 Ruby / 碎片 / 裁剪等参数。
    self._column_preprocess_toggle = QToolButton()
    self._column_preprocess_toggle.setObjectName("columnPreprocessToggleCompat")
    self._column_preprocess_toggle.setCheckable(True)
    self._column_preprocess_toggle.setChecked(True)
    self._column_preprocess_toggle.hide()

    self._column_preprocess_body = QWidget()
    preprocess = QVBoxLayout(self._column_preprocess_body)
    preprocess.setContentsMargins(10, 2, 0, 2)
    preprocess.setSpacing(3)
    self._column_preprocess_body.setVisible(True)

    self._column_ruby_filter_check = QCheckBox("自动过滤日文 Ruby（振假名）")
    self._column_ruby_filter_check.setChecked(True)
    self._column_ruby_filter_check.setToolTip(
        "关闭时严格保留物理列内全部原始像素。开启后只在 OCR 临时列图中清理疑似侧边 Ruby；"
        "不会修改页面管理中的原图，但可能误删独立浊点、小假名或细笔画，建议先预览抽查。"
    )
    self._column_ruby_filter_check.toggled.connect(self._on_column_cleanup_control_changed)
    preprocess.addWidget(self._column_ruby_filter_check)

    self._column_fragment_filter_check = QCheckBox("删除残损小文字 / 邻列碎片")
    self._column_fragment_filter_check.setChecked(False)
    self._column_fragment_filter_check.setToolTip(
        "仅处理 OCR 临时列图中的小型孤立组件和邻列残影；关闭时使用无损正文像素合同。"
        "这是可选的破坏性清理，可能误删浊点、半浊点、标点或断离偏旁，建议只在残影明显时开启。"
    )
    self._column_fragment_filter_check.toggled.connect(self._on_column_cleanup_control_changed)
    preprocess.addWidget(self._column_fragment_filter_check)

    self._column_smart_crop_check = QCheckBox("分列 OCR 智能裁剪（不缩放）")
    self._column_smart_crop_check.setChecked(True)
    self._column_smart_crop_check.setToolTip(
        "只从正文可见层裁掉确定为空白的外侧画布并保留安全边距，文字像素不缩放、不锐化。"
        "Apple/NDL/Paddle/Google Vision 保留较宽纸白上下文；"
        "Hayai/48px 使用紧凑白底视窗。无论哪一种，相邻列与 Ruby 都不会重新进入正文 OCR。"
    )
    self._column_smart_crop_check.toggled.connect(self._on_column_cleanup_control_changed)
    preprocess.addWidget(self._column_smart_crop_check)

    ruby_strength_row = QHBoxLayout()
    ruby_strength_lbl = QLabel("Ruby 过滤强度")
    ruby_strength_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
    ruby_strength_row.addWidget(ruby_strength_lbl)
    self._column_ruby_strength_combo = NoWheelComboBox()
    self._column_ruby_strength_combo.addItem("弱：只去除很明确的细小注音", "weak")
    self._column_ruby_strength_combo.addItem("标准：轻小说推荐", "standard")
    self._column_ruby_strength_combo.addItem("强：同时清理更多邻列残影", "strong")
    self._column_ruby_strength_combo.setCurrentIndex(1)
    self._column_ruby_strength_combo.setToolTip(
        "只在启用 Ruby 过滤或残损碎片删除时生效。强度越高，侧边清理越积极；"
        "不会改变原始扫描页，只影响本次 OCR 的临时输入图。"
    )
    ruby_strength_row.addWidget(self._column_ruby_strength_combo, 1)
    preprocess.addLayout(ruby_strength_row)
    self._column_ruby_strength_combo.currentIndexChanged.connect(
        self._on_column_cleanup_control_changed
    )

    preprocess_reset_row = QHBoxLayout()
    preprocess_reset_hint = QLabel("保持默认可获得与稳定基线一致的 OCR 输入。")
    preprocess_reset_hint.setWordWrap(True)
    preprocess_reset_hint.setStyleSheet(f"color: {MUTED}; font-size: 10px;")
    preprocess_reset_row.addWidget(preprocess_reset_hint, 1)
    self._column_preprocess_reset_btn = QPushButton("恢复 OCR 默认设置")
    self._column_preprocess_reset_btn.setProperty("role", "secondary")
    self._column_preprocess_reset_btn.setToolTip(
        "恢复 Ruby 过滤开启、残损碎片删除关闭、智能裁剪开启和标准强度。"
    )
    self._column_preprocess_reset_btn.clicked.connect(
        self._reset_column_preprocess_defaults
    )
    preprocess_reset_row.addWidget(self._column_preprocess_reset_btn)
    preprocess.addLayout(preprocess_reset_row)
    settings.addWidget(self._column_preprocess_body)
    self._column_preprocess_customized = False
    self._apply_column_preprocess_defaults()

    self._column_compact_transport_check = QCheckBox(
        "无损紧凑列图加速（不缩放；严重空白/缺字才复核全尺寸）"
    )
    self._column_compact_transport_check.setChecked(True)
    self._column_compact_transport_check.setToolTip(
        "所有 OCR 先共用同一张正文可见层：相邻列与 Ruby 均为不透明纸白色。"
        "送入识别器时只裁掉部分纯白画布；Apple/NDL/Paddle 等布局 OCR 保留较宽上下文，"
        "Hayai/48px 使用较紧凑视窗。文字像素保持原尺寸且不重采样。"
        "只有空结果、占位符或与黑像素估计相比严重缺字时，"
        "才把全尺寸掩膜作为该列唯一一次救援；低置信或引号不平衡本身不会重复调用 OCR。"
    )
    settings.addWidget(self._column_compact_transport_check)
    rescue_row = QHBoxLayout()
    rescue_lbl = QLabel("列级救援")
    rescue_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
    rescue_row.addWidget(rescue_lbl)
    self._column_rescue_policy_combo = NoWheelComboBox()
    self._column_rescue_policy_combo.addItem(
        "自适应单次（推荐）：每列最多一种救援", "adaptive"
    )
    self._column_rescue_policy_combo.addItem(
        "关闭救援：只保留主识别与人工复核", "off"
    )
    self._column_rescue_policy_combo.addItem(
        "完整多轮：兼容旧版恢复链", "legacy"
    )
    self._column_rescue_policy_combo.setCurrentIndex(0)
    self._column_rescue_policy_combo.setToolTip(
        "自适应模式根据空列、严重缺字和分离短段，为每个物理列只选择一种恢复路径；"
        "不会再连续执行全尺寸、短块、扩边和三种增强。关闭救援速度最快，疑难列保留为原结果或 □。"
        "完整多轮保留旧版全部恢复流程，用于个别特殊书页兼容。多模型对比仍固定每模型每列一次。"
    )
    rescue_row.addWidget(self._column_rescue_policy_combo, 1)
    settings.addLayout(rescue_row)
    ndl_mode_row = QHBoxLayout()
    ndl_mode_lbl = QLabel("NDLOCR 分列策略")
    ndl_mode_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
    ndl_mode_row.addWidget(ndl_mode_lbl)
    self._ndlocr_page_mode_combo = NoWheelComboBox()
    self._ndlocr_page_mode_combo.addItem(
        "智能混合（推荐）：整页一次，疑难列再补识", "hybrid"
    )
    self._ndlocr_page_mode_combo.addItem(
        "高速整页：每页一次，漏列保留待人工", "page"
    )
    self._ndlocr_page_mode_combo.addItem(
        "强制逐列：共享统一分列后逐列识别", "column"
    )
    self._ndlocr_page_mode_combo.setCurrentIndex(0)
    self._ndlocr_page_mode_combo.setToolTip(
        "仅影响 NDLOCR-Lite。智能混合先在同一份统一物理分列几何上构造 Ruby-free 整页输入并识别一次，"
        "再把结果严格映射回共享列槽；只有空列、跨列歧义、低置信或明显异常列才逐列补识。"
        "高速整页每页只识别一次，漏列保留为 □ 进入复核。强制逐列仍复用相同的页面级分列，不重复检测。"
    )
    ndl_mode_row.addWidget(self._ndlocr_page_mode_combo, 1)
    settings.addLayout(ndl_mode_row)
    self._column_strict_check = QCheckBox("严格防漏：预计列、识别列、文档列和 DOCX 列必须一致")
    self._column_strict_check.setChecked(True)
    settings.addWidget(self._column_strict_check)
    row = QHBoxLayout()
    self._column_preview_btn = QPushButton("👁 预览分列与掩膜")
    self._column_preview_btn.clicked.connect(self._preview_column_split)
    row.addWidget(self._column_preview_btn)
    self._column_detect_btn = QPushButton("🔍 重新检测本地 OCR")
    self._column_detect_btn.clicked.connect(lambda: self._refresh_ocr_runtime_status(show_dialog=True, deep=True))
    row.addWidget(self._column_detect_btn)
    self._column_detect_status = QLabel("")
    self._column_detect_status.setStyleSheet(f"color: {MUTED}; font-size: 10px;")
    row.addWidget(self._column_detect_status, 1)
    settings.addLayout(row)
    hint = QLabel(
        "两种方式都先建立同一份 Ruby-free 正文可见层。界面可用原页位置检查“分列显示”，"
        "但真正送入 OCR 前都会按当前单列墨迹收紧，并在四边保留安全白边；不会再把整页白底交给模型。"
        "正文像素不缩放，Ruby 与相邻列始终保持纸白隔离。"
    )
    hint.setWordWrap(True)
    hint.setStyleSheet(f"color: #5F7189; font-size: 10px;")
    settings.addWidget(hint)

    self._handwriting_card_widget = QFrame()
    self._handwriting_card_widget.setObjectName("handwritingOcrCard")
    self._handwriting_card_widget.setStyleSheet(
        f"QFrame#handwritingOcrCard {{ background: #F7F8FA; border: 2px solid #9CC7FF; "
        "border-radius: 12px; margin: 8px 10px 2px 10px; }"
    )
    hw_layout = QVBoxLayout(self._handwriting_card_widget)
    hw_layout.setContentsMargins(12, 11, 12, 12)
    hw_layout.setSpacing(7)

    hw_head = QHBoxLayout()
    hw_title = QLabel("OCR + 日语手写人工纠错")
    hw_title.setStyleSheet(
        "font-size: 13px; font-weight: bold; color: #1C47B8; "
        "border: none; background: transparent; padding: 0;"
    )
    hw_head.addWidget(hw_title)
    # Compatibility-only state holder: the review page itself is permanently
    # expanded and no longer exposes a disclosure control.
    self._handwriting_card_toggle = QToolButton()
    self._handwriting_card_toggle.setObjectName("handwritingCardToggleCompat")
    self._handwriting_card_toggle.setCheckable(True)
    self._handwriting_card_toggle.setChecked(True)
    self._handwriting_card_toggle.hide()
    hw_head.addStretch(1)
    hw_badge = make_badge("安全复核", "#2F6BFF")
    hw_head.addWidget(hw_badge)
    hw_layout.addLayout(hw_head)

    self._handwriting_card_body = QWidget()
    self._handwriting_card_body.setVisible(True)
    hw_body_layout = QVBoxLayout(self._handwriting_card_body)
    hw_body_layout.setContentsMargins(0, 0, 0, 0)
    hw_body_layout.setSpacing(7)

    hw_desc = QLabel("先使用当前选定的普通 OCR 生成完整底稿，再根据置信度、黑像素字数、异常符号和描摹候选冲突定位疑点；人工确认前绝不自动覆盖正文。")
    hw_desc.setWordWrap(True)
    hw_desc.setStyleSheet(f"color: {MUTED}; font-size: 10px; border: none;")
    hw_body_layout.addWidget(hw_desc)

    self._handwriting_trace_check = QCheckBox("启用逐字审校（预览显示蓝色逐字框）")
    self._handwriting_trace_check.setChecked(False)
    self._handwriting_trace_check.setToolTip(
        "未勾选时，普通 OCR 不执行逐字投影，也不生成逐字框；只使用普通版面/物理列检测。"
        "勾选后才为逐字审校生成投影框，但不会自动开始 OCR。需要执行 OCR 后人工复核时，"
        "请单独点击“开始 OCR + 人工纠错”。"
    )
    self._handwriting_trace_check.toggled.connect(self._on_handwriting_trace_toggled)
    hw_body_layout.addWidget(self._handwriting_trace_check)

    mode_grid = QGridLayout()
    mode_grid.setContentsMargins(0, 0, 0, 0)
    mode_grid.setHorizontalSpacing(8)
    mode_grid.setVerticalSpacing(4)
    mode_lbl = QLabel("模式")
    mode_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
    mode_grid.addWidget(mode_lbl, 0, 0, 1, 2)
    self._handwriting_mode_auto = QRadioButton("只自动筛查")
    self._handwriting_mode_hybrid = QRadioButton("自动筛查 + 人工复核")
    self._handwriting_mode_manual = QRadioButton("人工复核全部列")
    self._handwriting_mode_hybrid.setChecked(True)
    mode_grid.addWidget(self._handwriting_mode_auto, 1, 0)
    mode_grid.addWidget(self._handwriting_mode_hybrid, 1, 1)
    mode_grid.addWidget(self._handwriting_mode_manual, 2, 0, 1, 2)
    hw_body_layout.addLayout(mode_grid)

    strategy_row = QHBoxLayout()
    strategy_lbl = QLabel("自动策略")
    strategy_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
    strategy_row.addWidget(strategy_lbl)
    self._handwriting_strategy_combo = NoWheelComboBox()
    self._handwriting_strategy_combo.addItem("保守（只标极高风险冲突）", "conservative")
    self._handwriting_strategy_combo.addItem("标准（推荐疑点范围）", "balanced")
    self._handwriting_strategy_combo.addItem("广泛（标记更多候选冲突）", "aggressive")
    self._handwriting_strategy_combo.setCurrentIndex(1)
    strategy_row.addWidget(self._handwriting_strategy_combo, 1)
    hw_body_layout.addLayout(strategy_row)

    backend_row = QHBoxLayout()
    backend_lbl = QLabel("描摹候选器")
    backend_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
    backend_row.addWidget(backend_lbl)
    self._handwriting_backend_combo = NoWheelComboBox()
    self._handwriting_backend_combo.addItem("自动候选（Apple PKStroke 优先；JLect 备用）", "auto")
    self._handwriting_backend_combo.addItem("Apple PKStrokeRecognizer（稳定分笔，macOS 27）", "apple")
    self._handwriting_backend_combo.addItem("OpenVINO 日语手写模型（兼容备用）", "openvino")
    self._handwriting_backend_combo.addItem("本地 JLect 笔画候选（最低备用）", "jlect")
    backend_row.addWidget(self._handwriting_backend_combo, 1)
    hw_body_layout.addLayout(backend_row)

    model_row = QHBoxLayout()
    self._handwriting_model_status = QLabel("")
    self._handwriting_model_status.setWordWrap(True)
    self._handwriting_model_status.setStyleSheet(f"color: {MUTED}; font-size: 10px;")
    model_row.addWidget(self._handwriting_model_status, 1)
    self._handwriting_model_btn = QPushButton("下载手写模型")
    self._handwriting_model_btn.clicked.connect(self._install_handwriting_model)
    model_row.addWidget(self._handwriting_model_btn)
    hw_body_layout.addLayout(model_row)

    apple_test_hint = QLabel("先手动画一个日语字，验证苹果识别器本身")
    apple_test_hint.setWordWrap(True)
    apple_test_hint.setStyleSheet(f"color: {MUTED}; font-size: 10px;")
    hw_body_layout.addWidget(apple_test_hint)
    apple_test_row = QHBoxLayout()
    apple_test_row.setContentsMargins(0, 0, 0, 0)
    apple_test_row.setSpacing(6)
    self._handwriting_apple_test_btn = QPushButton("打开 Apple 手写测试面板")
    self._handwriting_apple_test_btn.setToolTip(
        "打开原生 macOS 测试窗口：左侧用鼠标/触控板分笔书写，右侧显示由 PKDrawing 直接渲染的苹果输入预览，并显示 recognizedText() 首结果。"
    )
    self._handwriting_apple_test_btn.clicked.connect(self._open_apple_handwriting_test_panel)
    apple_test_row.addWidget(self._handwriting_apple_test_btn)
    self._handwriting_apple_auto_preview_btn = QPushButton("预览最新自动轨迹")
    self._handwriting_apple_auto_preview_btn.setToolTip(
        "将程序最近生成的自动点序列载入同一个原生测试面板，直接查看 PKDrawing 并调用 PKStrokeRecognizer。"
    )
    self._handwriting_apple_auto_preview_btn.clicked.connect(self._open_latest_apple_auto_trace)
    apple_test_row.addWidget(self._handwriting_apple_auto_preview_btn)
    hw_body_layout.addLayout(apple_test_row)
    self._handwriting_backend_combo.currentIndexChanged.connect(self._refresh_handwriting_model_status)

    self._handwriting_char_mask_check = QCheckBox("当前列内再次掩膜：过滤振假名 / 邻列残影")
    self._handwriting_char_mask_check.setChecked(True)
    self._handwriting_char_mask_check.setToolTip("不裁切、不缩放，保持原像素尺寸；只把当前列正文主字带以外的深色文字替换为估算纸张底色。")
    hw_body_layout.addWidget(self._handwriting_char_mask_check)
    self._handwriting_symbol_insert_check = QCheckBox("把句读点 / 引号 / 长音符差异纳入疑点")
    self._handwriting_symbol_insert_check.setChecked(True)
    self._handwriting_symbol_insert_check.setToolTip("仅用于标记符号疑点，不会自动插入或删除正文字符。")
    hw_body_layout.addWidget(self._handwriting_symbol_insert_check)

    self._handwriting_run_btn = accent_button("▶  开始 OCR + 人工纠错", color="#2F6BFF")
    preserve_button_text(self._handwriting_run_btn)
    self._handwriting_run_btn.setToolTip("自动启用分列掩膜；先运行当前 OCR，再进行疑点筛查和可选人工纠错。")
    self._handwriting_run_btn.clicked.connect(self._run_handwriting_ocr)
    hw_body_layout.addWidget(self._handwriting_run_btn)

    handwriting_hint = QLabel(
        "流程：固定正文区域 → 分列掩膜 → 当前 OCR 生成底稿 → 字数/置信度/符号/引号/重复片段筛查 → 描摹候选冲突 → 疑点优先人工修改 → 列尾组句。\n"
        "OCR 原文保留在 ocr_raw；候选永不自动覆盖。Apple 测试面板只用于验证系统分笔识别，人工纠错窗口可直接使用 macOS 日语输入法。"
    )
    handwriting_hint.setWordWrap(True)
    handwriting_hint.setStyleSheet(f"color: #5F7189; font-size: 10px;")
    hw_body_layout.addWidget(handwriting_hint)
    hw_layout.addWidget(self._handwriting_card_body)
    self._on_handwriting_trace_toggled(False)
    self._refresh_handwriting_model_status()
    cov.addWidget(self._column_settings_widget)
    split_enabled = self._column_split_check.isChecked()
    # The settings themselves stay visible; only actions that execute the
    # split pipeline are disabled until the feature is enabled.
    self._column_settings_widget.setVisible(True)
    self._column_preview_btn.setEnabled(split_enabled)
    # Keep the historical add call for compatibility with source-level
    # integration tests; adding it to the review layout immediately moves
    # the same widget into the dedicated “逐字审校” page.
    ll.addWidget(self._handwriting_card_widget)
    self._review_settings_layout.addWidget(self._handwriting_card_widget)
    self._layout_settings_layout.addWidget(self._column_ocr_widget)

    # OCR 结果进入 Formatter 之前执行的列级重排。
    self._ocr_reflow_widget = QWidget()
    orv = QVBoxLayout(self._ocr_reflow_widget)
    orv.setContentsMargins(10, 10, 10, 2)
    orv.setSpacing(3)
    self._column_sentence_reflow_check = QCheckBox("逐列成句：先按坐标归并物理列，再按列尾成句")
    self._column_sentence_reflow_check.setChecked(True)
    self._column_sentence_reflow_check.setToolTip("在 OCR 文档交给 Formatter 前执行；先保留全部 OCR 文字列（不自动清理页眉），再将同一横向位置的碎片合成真实竖列；只检查该列最后一个有效字符，同列中间标点不拆行，页末残句继续下一页")
    self._column_sentence_reflow_check.toggled.connect(
        self._update_sentence_context_reocr_state
    )
    orv.addWidget(self._column_sentence_reflow_check)
    self._column_sentence_context_reocr_check = QCheckBox(
        "整句上下文重识别：列尾无句末时接续后列，完成整句后重新 OCR"
    )
    self._column_sentence_context_reocr_check.setChecked(False)
    self._column_sentence_context_reocr_check.setToolTip(
        "独立开关。先逐列 OCR 判断句子边界；连续两列以上直到出现句末标点后，"
        "把这些列的原始像素无缩放地从右到左排成句组图，再调用当前 OCR 一次。"
        "句组结果必须通过句末、长度、相似度和符号安全校验，否则自动保留原逐列结果；支持跨页接续。"
    )
    self._column_sentence_context_reocr_check.toggled.connect(
        self._update_sentence_context_reocr_state
    )
    orv.addWidget(self._column_sentence_context_reocr_check)
    strategy_row = QHBoxLayout()
    strategy_label = QLabel("整句重识别策略")
    strategy_label.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
    strategy_row.addWidget(strategy_label)
    self._column_sentence_strategy_combo = NoWheelComboBox()
    self._column_sentence_strategy_combo.addItem(
        "自适应：仅有风险证据的句子（推荐）", "smart"
    )
    self._column_sentence_strategy_combo.addItem(
        "完整：全部多列句（原模式）", "full"
    )
    self._column_sentence_strategy_combo.setCurrentIndex(0)
    self._column_sentence_strategy_combo.setToolTip(
        "自适应模式不再把句子较长或跨页本身视为问题；只有空列、列级救援、候选冲突、"
        "严重字数差异、句末缺失、引号或异常符号等风险证据才执行整句 OCR。"
        "完整模式与旧版本一致，对所有完整多列句组重识别。"
    )
    strategy_row.addWidget(self._column_sentence_strategy_combo, 1)
    orv.addLayout(strategy_row)
    self._column_sentence_global_merged_box_check = QCheckBox(
        "疑难句重识别优先使用真实合并框"
    )
    self._column_sentence_global_merged_box_check.setChecked(False)
    self._column_sentence_global_merged_box_check.setToolTip(
        "只改变已经被句级风险门选中的句组图构造方式，不会额外扩大重识别范围。"
        "同页连续列按真实位置合并成原像素矩形框；跨页按阅读顺序无缩放拼接。"
        "合并框构造失败时安全回退列条带，候选仍须通过长度、相似度、句末和符号校验。"
    )
    orv.addWidget(self._column_sentence_global_merged_box_check)
    cap_row = QHBoxLayout()
    cap_lbl = QLabel("无句末安全上限")
    cap_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
    cap_row.addWidget(cap_lbl)
    self._column_sentence_max_spin = NoWheelSpinBox()
    self._column_sentence_max_spin.setRange(3, 256)
    self._column_sentence_max_spin.setValue(10)
    self._column_sentence_max_spin.setSuffix(" 列")
    self._column_sentence_max_spin.setToolTip("只能用上下按钮、点击输入或键盘调整；鼠标滚轮/触控板不会改变数值")
    cap_row.addWidget(self._column_sentence_max_spin, 1)
    orv.addLayout(cap_row)
    reflow_hint = QLabel("逐列成句默认开启：先按坐标归并真实竖列，再只检查列尾决定是否接续下一列。整句上下文重识别默认关闭；开启后，自适应策略只处理具有空列、列级救援、冲突、严重缺字、句末或符号异常的句子，句子较长或跨页本身不会触发二次 OCR。完整策略保留旧版全部多列句重识别。真实合并框只负责疑难句的图像布局，构造失败自动回退条带。")
    reflow_hint.setStyleSheet(f"color: #5F7189; font-size: 10px;")
    reflow_hint.setWordWrap(True)
    orv.addWidget(reflow_hint)
    self._update_sentence_context_reocr_state()
    # 逐字审校默认关闭。只有用户点击“开始 OCR + 人工纠错”时，
    # 本次运行才启用字符扫描与人工复核；普通“开始 OCR”始终独立。
    self._handwriting_trace_check.setChecked(False)
    self._handwriting_card_toggle.setChecked(True)
    self._on_handwriting_trace_toggled(False)
    self._ocr_settings_tabs.setCurrentIndex(0)
    self._layout_settings_layout.addWidget(self._ocr_reflow_widget)
    self._layout_settings_layout.addStretch(1)
    self._review_settings_layout.addStretch(1)

    _build_reference_engine_panel(self, ocr_adapters)
    ll.addStretch()

    # 适配器与参数放进滚动区：窗口高度不足时仍可完整查看全部内容。
    left_scroll = QScrollArea()
    left_scroll.setWidgetResizable(True)
    left_scroll.setFrameShape(QFrame.NoFrame)
    left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    left_scroll.setMinimumWidth(OCR_LEFT_WIDTH)
    left_scroll.setWidget(left)
    left_outer.addWidget(left_scroll, 1)

    # OCR 执行控制已移到下方日志标题栏右侧，左侧只保留参数与输入区。

    top_row.addWidget(left_container, 1)


