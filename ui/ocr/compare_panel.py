from __future__ import annotations

from functools import partial

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QPlainTextEdit, QLabel, QPushButton,
    QCheckBox, QComboBox, QProgressBar, QFrame, QSizePolicy, QScrollArea, QToolButton,
    QListView, QButtonGroup, QAbstractItemView, QMenu, QLineEdit,
)

from ui.common.editor_controls import MouseWheelPlainTextEdit
from ui.common.window_state import bind_splitter
from ui.common.styling import (
    BG,
    CARD, BORDER, INK, MUTED, SUBTLE, SUCCESS, LIGHT_PREVIEW_STYLE, EDITOR_SCROLLBAR_STYLE,
    accent_button, make_separator,
)
from ui.responsive import configure_combo
from ui.ocr.compare_widgets import DecisionQueueListModel, DecisionQueueDelegate
from ui.ocr.sentence_strip import SentenceStrip


def build_ocr_compare_panel(self):
    root = QVBoxLayout(self)
    root.setContentsMargins(0, 0, 0, 0)
    root.setSpacing(0)

    header = QWidget()
    header.setStyleSheet(f"background: {BG};")
    hl = QVBoxLayout(header)
    # Keep the command surface compact: the OCR source/fusion work area is
    # the primary workspace, while every existing action remains directly
    # available in the same header.
    hl.setContentsMargins(8, 3, 8, 3)
    hl.setSpacing(2)

    title_row = QHBoxLayout()
    title_row.setContentsMargins(0, 0, 0, 0)
    title_row.setSpacing(6)
    self._workspace_title = QLabel("OCR 对比 · 结果收件箱")
    self._workspace_title.setStyleSheet("font-size: 13px; font-weight: 700;")
    title_row.addWidget(self._workspace_title)
    self._adjudication_undo_btn = QPushButton("↶ 裁决")
    self._adjudication_undo_btn.setToolTip("撤销最近一次人工候选选择/重新打开；不会覆盖之后的 AI、外部导入或人工编辑")
    self._adjudication_undo_btn.clicked.connect(self._history_service.undo)
    self._adjudication_undo_btn.setEnabled(False)
    title_row.addWidget(self._adjudication_undo_btn)
    self._adjudication_redo_btn = QPushButton("↷ 裁决")
    self._adjudication_redo_btn.setToolTip("重做刚刚撤销的裁决；相关句发生后续修改时会拒绝执行")
    self._adjudication_redo_btn.clicked.connect(self._history_service.redo)
    self._adjudication_redo_btn.setEnabled(False)
    title_row.addWidget(self._adjudication_redo_btn)
    self._auto_btn = accent_button("⚙ 自动选优", color="#0E7490")
    self._auto_btn.clicked.connect(self._auto_select_all)
    self._auto_btn.setEnabled(False)
    title_row.addWidget(self._auto_btn)
    self._auto_restore_batch_btn = QPushButton("⟲ 撤销自动选优")
    self._auto_restore_batch_btn.setToolTip("仅恢复最近一次通过预览确认的自动选优；之后若有新的人工/AI裁决，恢复时会拒绝覆盖。")
    self._auto_restore_batch_btn.clicked.connect(self._restore_last_auto_select)
    self._auto_restore_batch_btn.setEnabled(False)
    title_row.addWidget(self._auto_restore_batch_btn)
    self._realign_btn = QPushButton("↔ 重新对齐")
    self._realign_btn.setToolTip("允许各栏新增或删除换行，并按当前文字重新逐句对齐、刷新红绿差异和融合候选。")
    self._realign_btn.clicked.connect(self._realign_and_auto)
    self._realign_btn.setEnabled(False)
    title_row.addWidget(self._realign_btn)
    self._restore_btn = QPushButton("↶ 恢复初始")
    self._restore_btn.setToolTip("恢复当前全部初始 OCR、初始对齐、红绿差异和融合候选；单 OCR 模式下恢复标准化前结果。")
    self._restore_btn.clicked.connect(self._restore_initial)
    self._restore_btn.setEnabled(False)
    title_row.addWidget(self._restore_btn)
    self._unicode_normalize_btn = accent_button("无损标准化", color="#0F766E")
    self._unicode_normalize_btn.setToolTip(
        "正文只修复不会改变字义的组合浊音、半角假名和竖排兼容标点；"
        "不删除任何 OCR 内容，不替换汉字/异体字，不改破折号和标点风格。"
        "兼容汉字、IVS、不可见字符等只生成临时比较键，用于消除假冲突，"
        "绝不写回正文或 EPUB；原始 OCR 仍可用“恢复初始”取回。"
    )
    self._unicode_normalize_btn.clicked.connect(self._standardize_unicode_variants)
    self._unicode_normalize_btn.setEnabled(False)
    title_row.addWidget(self._unicode_normalize_btn)
    for button in (self._adjudication_undo_btn, self._adjudication_redo_btn, self._auto_btn, self._auto_restore_batch_btn, self._realign_btn, self._restore_btn, self._unicode_normalize_btn):
        button.setMinimumHeight(30)
        button.setMaximumHeight(32)
        button.setStyleSheet(
            button.styleSheet() + "QPushButton{font-size:9px;padding:2px 6px;}"
        )
    self._summary = QLabel(
        "单模型 OCR 完成后请手动点击“从 OCR 识别载入”；多模型 OCR 完成后会自动载入。"
    )
    self._summary.setWordWrap(False)
    self._summary.setMaximumHeight(16)
    self._summary.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    self._summary.setStyleSheet(f"color: {MUTED}; font-size: 9px;")
    self._summary.setToolTip(self._summary.text())
    title_row.insertWidget(1, self._summary, 1)
    self._review_mode_group = QButtonGroup(self)
    self._review_mode_group.setExclusive(True)
    self._decision_mode_btn = QPushButton("逐句裁决")
    self._full_mode_btn = QPushButton("全文对比")
    for mode_key, button in (("decision", self._decision_mode_btn), ("full", self._full_mode_btn)):
        button.setCheckable(True)
        button.setMinimumHeight(30)
        button.setMaximumHeight(32)
        button.setStyleSheet(
            "QPushButton{font-size:9px;padding:2px 8px;border:1px solid #CDD3DA;border-radius:6px;}"
            "QPushButton:checked{background:#E4EEFF;color:#2559E0;font-weight:700;border-color:#2F6BFF;}"
        )
        button.clicked.connect(partial(self._set_review_mode, mode_key))
        self._review_mode_group.addButton(button)
    self._decision_mode_btn.setChecked(True)
    self._decision_mode_btn.setEnabled(False)
    self._full_mode_btn.setEnabled(False)
    # Keep the legacy mode buttons as hidden state mirrors so restored
    # sessions and older automation contracts remain compatible.  The
    # visible control is a single, unambiguous switch.
    self._decision_mode_btn.setParent(header)
    self._full_mode_btn.setParent(header)
    self._decision_mode_btn.setVisible(False)
    self._full_mode_btn.setVisible(False)
    # Use a checkable push button rather than a compact QCheckBox.  The
    # whole painted rectangle is now the click target, which avoids the
    # tiny indicator-only hit area and makes the switch reliable in the
    # compressed title row on macOS.
    self._full_text_compare_check = QPushButton("显示全文对比")
    self._full_text_compare_check.setCheckable(True)
    self._full_text_compare_check.setChecked(False)
    self._full_text_compare_check.setEnabled(False)
    self._full_text_compare_check.setCursor(Qt.PointingHandCursor)
    self._full_text_compare_check.setFocusPolicy(Qt.StrongFocus)
    self._full_text_compare_check.setToolTip(
        "打开：多份 OCR 结果显示完整全文并保持同步滚动；"
        "关闭：只显示当前稳定句，继续逐句裁决。只改变显示，不重新 OCR 或对齐。"
    )
    self._full_text_compare_check.setMinimumWidth(96)
    self._full_text_compare_check.setMinimumHeight(30)
    self._full_text_compare_check.setMaximumHeight(32)
    self._full_text_compare_check.setStyleSheet(
        "QPushButton{font-size:9px;padding:2px 8px;border:1px solid #CDD3DA;"
        "border-radius:6px;background:#FAFAFB;}"
        "QPushButton:hover{border-color:#2F6BFF;background:#F7F8FA;}"
        "QPushButton:checked{background:#E4EEFF;color:#2559E0;font-weight:700;"
        "border-color:#2F6BFF;}"
        "QPushButton:disabled{color:#5F7189;background:#E6EEF8;border-color:#D0D5DD;}"
    )
    self._full_text_compare_check.toggled.connect(self._toggle_full_text_compare)
    title_row.insertWidget(2, self._full_text_compare_check)
    self._ai_adjudicate_btn = accent_button("✦ AI裁决", color="#6D28D9")
    self._ai_adjudicate_btn.setToolTip(
        "把真正 OCR 分歧的原图裁切批量拼成证据板，一次提交几十到上百条给多模态模型；"
        "AI 默认只选择 A/B/C/D，不重做整页 OCR，不改写原始模型结果。"
    )
    self._ai_adjudicate_btn.setEnabled(False)
    self._ai_adjudicate_btn.setMinimumHeight(30)
    self._ai_adjudicate_btn.setMaximumHeight(32)
    self._ai_adjudicate_btn.setStyleSheet(self._ai_adjudicate_btn.styleSheet() + "QPushButton{font-size:9px;padding:2px 8px;}")
    self._ai_adjudicate_btn.clicked.connect(self._run_ai_adjudication)
    title_row.insertWidget(5, self._ai_adjudicate_btn)
    self._ai_adjudication_cancel_btn = QPushButton("停止AI裁决")
    self._ai_adjudication_cancel_btn.setVisible(False)
    self._ai_adjudication_cancel_btn.setMinimumHeight(30)
    self._ai_adjudication_cancel_btn.setMaximumHeight(32)
    self._ai_adjudication_cancel_btn.clicked.connect(self._cancel_ai_adjudication)
    title_row.insertWidget(6, self._ai_adjudication_cancel_btn)
    self._more_actions_btn = QToolButton()
    self._more_actions_btn.setText("更多操作 ▾")
    self._more_actions_btn.setCheckable(True)
    self._more_actions_btn.setToolTip("展开低频 AI 包、OCR 裁决、恢复会话和骨架 EPUB 操作。")
    self._more_actions_btn.setMinimumHeight(30)
    self._more_actions_btn.setMaximumHeight(32)
    self._more_actions_btn.setStyleSheet("font-size:9px;padding:2px 7px;")
    title_row.addWidget(self._more_actions_btn)
    hl.addLayout(title_row)

    source_inbox_row = QHBoxLayout()
    source_inbox_row.setContentsMargins(0, 0, 0, 0)
    source_inbox_row.setSpacing(5)
    self._single_source_state = QLabel("单 OCR：暂无可载入结果")
    self._single_source_state.setStyleSheet("color:#2F6BFF;font-size:10px;font-weight:600;")
    source_inbox_row.addWidget(self._single_source_state)
    self._load_single_from_ocr_btn = accent_button("＋ 从 OCR 识别载入", color="#2F6BFF")
    self._load_single_from_ocr_btn.setToolTip(
        "OCR 识别页完成单模型 OCR 后，只登记为可用来源；点击这里才载入本页。"
        "多模型 OCR 仍会自动载入，不会与单结果混合。"
    )
    self._load_single_from_ocr_btn.setEnabled(False)
    self._load_single_from_ocr_btn.clicked.connect(self._load_available_single_result)
    source_inbox_row.addWidget(self._load_single_from_ocr_btn)
    self._return_multi_btn = QPushButton("⇄ 返回多模型对比")
    self._return_multi_btn.setToolTip("恢复切换前的多模型文本、对齐、人工候选选择和未完成状态。")
    self._return_multi_btn.setVisible(False)
    self._return_multi_btn.clicked.connect(self._restore_suspended_multi)
    source_inbox_row.addWidget(self._return_multi_btn)
    self._single_export_btn = accent_button("⇩ 导出单OCR AI包", color="#7C3AED")
    self._single_export_btn.setToolTip(
        "导出当前载入的单 OCR 原格式校对包；完整保留封面、插图、目录、坐标、列 ID 和块顺序。"
    )
    self._single_export_btn.setEnabled(False)
    self._single_export_btn.clicked.connect(self._export_single_roundtrip_package)
    source_inbox_row.addWidget(self._single_export_btn)
    self._single_import_btn = accent_button("⇧ 导入单OCR AI包", color="#0F766E")
    self._single_import_btn.setToolTip(
        "按稳定 block ID 导入单 OCR 校对结果。导入后更新同一来源并直接应用，不新增重复副本。"
    )
    self._single_import_btn.clicked.connect(self._import_single_roundtrip_package)
    source_inbox_row.addWidget(self._single_import_btn)
    source_inbox_row.addSpacing(3)

    choose_row = QHBoxLayout()
    choose_row.setContentsMargins(0, 0, 0, 0)
    choose_row.setSpacing(5)
    self._choose_label = QLabel("当前句采用：")
    self._choose_label.setStyleSheet("font-size:10px;")
    choose_row.addWidget(self._choose_label)
    self._choose_buttons: list[QPushButton] = []
    for index in range(6):
        button = QPushButton(f"模型{index + 1}")
        button.setEnabled(False)
        button.setVisible(False)
        button.setMinimumHeight(30)
        button.setMaximumHeight(32)
        button.setStyleSheet("font-size:9px;padding:1px 6px;")
        button.clicked.connect(partial(self._choose_current_from_model, index))
        choose_row.addWidget(button)
        self._choose_buttons.append(button)
    self._row_state = QLabel("当前句：—")
    self._row_state.setStyleSheet("color:#0E7490; font-size:10px; font-weight:600;")
    choose_row.addWidget(self._row_state)
    choose_row.addStretch(1)

    self._export_texts_btn = accent_button("⇩ 分别导出文本…", color="#A16207")
    self._export_texts_btn.setToolTip("选择文件夹，将当前 2～6 份 OCR 文本分别导出为 UTF-8 TXT；全部裁决后同时导出融合稿。")
    self._export_texts_btn.setEnabled(False)
    self._export_texts_btn.clicked.connect(self._export_separate_texts)
    choose_row.addWidget(self._export_texts_btn)
    self._apply_btn = accent_button("✓ 应用融合稿", color=SUCCESS)
    self._apply_btn.setEnabled(False)
    self._apply_btn.clicked.connect(self._apply_result)
    choose_row.addWidget(self._apply_btn)
    for button in (self._export_texts_btn, self._apply_btn):
        button.setMinimumHeight(30)
        button.setMaximumHeight(32)
        button.setStyleSheet(
            button.styleSheet() + "QPushButton{font-size:9px;padding:2px 6px;}"
        )
    # Source intake and per-sentence choices share one dense row.  This
    # removes a full toolbar line without hiding or nesting any action.
    source_inbox_row.addLayout(choose_row, 1)
    hl.addLayout(source_inbox_row)

    multi_package_row = QHBoxLayout()
    multi_package_row.setContentsMargins(0, 0, 0, 0)
    multi_package_row.setSpacing(5)
    multi_package_label = QLabel("多模型包：")
    multi_package_label.setStyleSheet("font-size:10px;color:#5B6B80;")
    multi_package_row.addWidget(multi_package_label)
    self._export_ai_package_btn = accent_button("⇩ 单独导出融合JSON", color="#7C3AED")
    self._export_ai_package_btn.setToolTip(
        "仅单独导出当前融合稿与全部 OCR 候选 JSON。通常直接使用右侧“导出AI修复包”，"
        "它会把同一份完整融合 JSON 一并装入 AI 修复包，同时生成最终出版图片、干净资源框架和硬审计说明。"
    )
    self._export_ai_package_btn.setEnabled(False)
    self._export_ai_package_btn.clicked.connect(self._export_multi_roundtrip_package)
    multi_package_row.addWidget(self._export_ai_package_btn)
    source_correction_row = QHBoxLayout()
    source_correction_row.setContentsMargins(0, 0, 0, 0)
    source_correction_row.setSpacing(5)
    source_correction_label = QLabel("OCR 裁决：")
    source_correction_label.setStyleSheet("font-size:10px;color:#5B6B80;")
    source_correction_row.addWidget(source_correction_label)
    self._export_source_correction_btn = accent_button("⇩ 导出 AI OCR 裁决包", color="#9333EA")
    self._export_source_correction_btn.setToolTip(
        "导出 AI OCR 裁决包。当前全部原始 OCR 永久只读；此前已接受的冲突裁决会作为只读参考重新开放复审，"
        "AI 可保留此前当前格式结果，也可提交更好的 final_text；精确一致/共同候选仍按当前策略锁定。"
    )
    self._export_source_correction_btn.setEnabled(False)
    self._export_source_correction_btn.clicked.connect(self._export_model_source_correction_package)
    source_correction_row.addWidget(self._export_source_correction_btn)
    self._import_source_correction_btn = accent_button("⇧ 导入 AI OCR 裁决", color="#0D9488")
    self._import_source_correction_btn.setToolTip(
        "仅导入当前格式 AI OCR 裁决 JSON/ZIP。后导入的已接受结果可改进同一稳定句；"
        "缺失/未决行不会抹掉前一包的好结果，且绝不改写原始 OCR。"
    )
    self._import_source_correction_btn.setEnabled(False)
    self._import_source_correction_btn.clicked.connect(self._import_model_source_correction_result)
    source_correction_row.addWidget(self._import_source_correction_btn)
    self._restore_source_session_btn = QPushButton("↻ 恢复纠错会话")
    self._restore_source_session_btn.setToolTip(
        "程序崩溃或重启后，只从当前格式裁决 ZIP 恢复当前 2～6 份 OCR 文档、物理列、密封对齐和人工选择。"
        "旧格式裁决包在稳定版前不兼容；恢复不会重新 OCR。"
    )
    self._restore_source_session_btn.clicked.connect(self._restore_model_source_correction_session)
    source_correction_row.addWidget(self._restore_source_session_btn)
    self._export_fusion_skeleton_btn = accent_button("⇩ 融合结果＋骨架EPUB", color="#2559E0")
    self._export_fusion_skeleton_btn.setToolTip(
        "导出当前全部模型候选、人工融合结果、逐源纠错审计和干净稳定 ID 骨架 EPUB。"
    )
    self._export_fusion_skeleton_btn.setEnabled(False)
    self._export_fusion_skeleton_btn.clicked.connect(self._export_fusion_and_skeleton)
    source_correction_row.addWidget(self._export_fusion_skeleton_btn)
    self._source_correction_state = QLabel("尚未导入 OCR 裁决")
    self._source_correction_state.setStyleSheet("font-size:9px;color:#5B6B80;")
    source_correction_row.addWidget(self._source_correction_state)
    source_correction_row.addStretch(1)
    self._export_ai_repair_epub_btn = accent_button("⇩ 导出AI修复包", color="#4338CA")
    self._export_ai_repair_epub_btn.setToolTip(
        "导出当前 V5 多模型分歧裁决包：只开放当前策略判定需要外部裁决的稳定句，"
        "完整保留角色、独立执行/seeded reuse、物理列与视觉证据；本地已完成裁决保持冻结。"
    )
    self._export_ai_repair_epub_btn.setEnabled(False)
    self._export_ai_repair_epub_btn.clicked.connect(self._export_ai_repair_epub)
    multi_package_row.addWidget(self._export_ai_repair_epub_btn)
    self._import_ai_repair_result_btn = accent_button("⇧ 导入AI修复结果", color="#047857")
    self._import_ai_repair_result_btn.setToolTip(
        "只导入当前 V5 decisions.json 或包含该文件的 ZIP。package_id、structure_sha256 和可编辑 ID 必须与当前会话完全一致。"
    )
    self._import_ai_repair_result_btn.setEnabled(False)
    self._import_ai_repair_result_btn.setVisible(False)
    self._import_ai_repair_result_btn.clicked.connect(self._import_ai_repair_result)
    self._import_ai_package_btn = accent_button("⇧ 导入AI融合包", color="#0F766E")
    self._import_ai_package_btn.setToolTip(
        "严格按 row ID 导入外部大模型融合结果，完整替换当前融合正文并直接应用；"
        "缺行、重复 ID 或结构改变会拒绝导入。"
    )
    self._import_ai_package_btn.setEnabled(False)
    self._import_ai_package_btn.clicked.connect(self._import_multi_roundtrip_package)
    multi_package_row.addWidget(self._import_ai_package_btn)
    multi_package_row.addStretch(1)
    for button in (
        self._load_single_from_ocr_btn, self._return_multi_btn, self._single_export_btn, self._single_import_btn,
        self._export_ai_package_btn, self._export_source_correction_btn,
        self._import_source_correction_btn, self._restore_source_session_btn, self._export_fusion_skeleton_btn,
        self._export_ai_repair_epub_btn, self._import_ai_repair_result_btn,
        self._ai_adjudicate_btn, self._ai_adjudication_cancel_btn,
        self._import_ai_package_btn,
    ):
        button.setMinimumHeight(30)
        button.setMaximumHeight(32)
        button.setStyleSheet(button.styleSheet() + "QPushButton{font-size:9px;padding:2px 6px;}")
    self._advanced_actions_panel = QWidget()
    advanced_actions_layout = QVBoxLayout(self._advanced_actions_panel)
    advanced_actions_layout.setContentsMargins(0, 1, 0, 0)
    advanced_actions_layout.setSpacing(2)
    advanced_actions_layout.addLayout(multi_package_row)
    advanced_actions_layout.addLayout(source_correction_row)
    self._source_correction_progress = QProgressBar()
    self._source_correction_progress.setRange(0, 100)
    self._source_correction_progress.setValue(0)
    self._source_correction_progress.setTextVisible(True)
    self._source_correction_progress.setMaximumHeight(16)
    self._source_correction_progress.setVisible(False)
    advanced_actions_layout.addWidget(self._source_correction_progress)
    self._ai_adjudication_progress = QProgressBar()
    self._ai_adjudication_progress.setRange(0, 100)
    self._ai_adjudication_progress.setValue(0)
    self._ai_adjudication_progress.setTextVisible(True)
    self._ai_adjudication_progress.setMaximumHeight(16)
    self._ai_adjudication_progress.setVisible(False)
    advanced_actions_layout.addWidget(self._ai_adjudication_progress)
    self._advanced_actions_panel.setVisible(False)
    self._more_actions_btn.toggled.connect(self._set_advanced_actions_visible)
    hl.addWidget(self._advanced_actions_panel)

    header.setVisible(False)
    self._legacy_compare_header = header
    root.addWidget(header)
    root.addWidget(make_separator())

    main_splitter = QSplitter(Qt.Vertical)
    self._main_splitter = main_splitter
    main_splitter.setChildrenCollapsible(False)

    self._source_area = QWidget()
    source_layout = QVBoxLayout(self._source_area)
    source_layout.setContentsMargins(10, 4, 10, 4)
    source_layout.setSpacing(0)
    # Stack the model books vertically.  Every model now receives the full
    # workspace width, so long Japanese sentences wrap instead of being
    # clipped inside three narrow side-by-side columns.  A nested splitter
    # keeps all model areas independently resizable without touching OCR or
    # alignment state.
    self._source_splitter = QSplitter(Qt.Vertical)
    self._source_splitter.setChildrenCollapsible(False)
    self._source_splitter.setHandleWidth(4)
    self._source_splitter.setStyleSheet(
        "QSplitter::handle{background:#E3ECF7;border-radius:2px;margin:1px 90px;}"
    )
    for index in range(6):
        panel = QWidget()
        panel.setMinimumHeight(72)
        panel.setStyleSheet(f"background:{CARD}; border:1px solid {BORDER}; border-radius:12px;")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(7, 4, 7, 4)
        panel_layout.setSpacing(2)
        label = QLabel(f"模型{index + 1} · 未载入")
        label.setStyleSheet("font-size:10px; font-weight:700;")
        label.setWordWrap(False)
        label.setMaximumHeight(17)
        panel_layout.addWidget(label)
        editor = MouseWheelPlainTextEdit()
        editor.setPlaceholderText("该模型的逐句 OCR 结果")
        editor.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        editor.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        editor.setStyleSheet(
            "QPlainTextEdit {" + LIGHT_PREVIEW_STYLE + "}" + EDITOR_SCROLLBAR_STYLE
        )
        editor.setTabChangesFocus(True)
        editor.cursorPositionChanged.connect(partial(self._source_cursor_changed, index))
        editor.verticalScrollBar().valueChanged.connect(partial(self._source_scroll_changed, index))
        editor.textChanged.connect(self._source_text_changed)
        panel_layout.addWidget(editor, 1)
        self._source_splitter.addWidget(panel)
        panel.setVisible(False)
        self._source_panels.append(panel)
        self._source_labels.append(label)
        self._source_editors.append(editor)
    self._source_splitter.setSizes([120] * 6)
    source_layout.addWidget(self._source_splitter, 1)
    self._source_area.setVisible(False)
    main_splitter.addWidget(self._source_area)

    self._result_panel = QFrame()
    self._result_panel.setObjectName("ocrCompareResultPanel")
    self._result_panel.setStyleSheet(
        f"QFrame#ocrCompareResultPanel{{background:transparent;border:none;}}"
    )
    result_layout = QVBoxLayout(self._result_panel)
    result_layout.setContentsMargins(28, 10, 28, 12)
    result_layout.setSpacing(8)
    result_header = QHBoxLayout()
    result_header.setContentsMargins(0, 0, 0, 0)
    result_header.setSpacing(5)
    self._result_title = QLabel("融合结果（真正一致与两模型共同候选均自动保留；真正分歧需裁决）")
    self._result_title.setStyleSheet("font-size:11px; font-weight:700;")
    result_header.addWidget(self._result_title)
    self._review_only_check = QCheckBox("只显示需判断")
    self._review_only_check.setChecked(True)
    self._review_only_check.setToolTip(
        "默认隐藏真正一致和已经选择的融合行；真正分歧会显示全部不同候选，"
        "快速共识产生的稳定共同候选自动保留，不再制造二次确认。"
        "勾选后自动前往下一组；取消勾选可查看全部融合结果。"
    )
    self._review_only_check.setStyleSheet("font-size:10px;spacing:4px;font-weight:600;")
    self._review_only_check.toggled.connect(self._set_review_only)
    result_header.addWidget(self._review_only_check)
    self._single_card_check = QCheckBox("只显示一个对比框")
    self._single_card_check.setChecked(True)
    self._single_card_check.setToolTip(
        "勾选后，上方每个模型只显示当前句，下方只创建当前一个对比框。"
        "选择候选后自动进入下一句；完整 OCR 文本仍保存在内部，不影响重新对齐、导出或应用。"
    )
    self._single_card_check.setStyleSheet("font-size:10px;spacing:4px;font-weight:600;")
    self._single_card_check.toggled.connect(self._set_single_card_mode)
    self._single_card_check.setVisible(False)
    result_header.addWidget(self._single_card_check)
    self._auto_advance_check = QCheckBox("选择后自动下一条")
    self._auto_advance_check.setChecked(True)
    self._auto_advance_check.setToolTip("关闭后选择会保留在当前句，便于再次核对；默认开启。")
    self._auto_advance_check.setStyleSheet("font-size:10px;spacing:4px;font-weight:600;")
    result_header.addWidget(self._auto_advance_check)
    self._detail = QLabel("当前句：—")
    self._detail.setStyleSheet(f"color:{MUTED}; font-size:9px;")
    self._detail.setMaximumWidth(520)
    self._detail.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    result_header.addWidget(self._detail)
    self._virtual_hint = QLabel("")
    self._virtual_hint.setStyleSheet("color:#5B6B80;font-size:9px;")
    self._virtual_hint.setToolTip("大文档只创建当前附近的候选控件；全部句子、选择和修改仍保存在内存状态中。")
    result_header.addWidget(self._virtual_hint)
    result_header.addStretch(1)
    self._prev_group_btn = QPushButton("← 上一分歧")
    self._prev_group_btn.setToolTip("定位上一条仍需判断的句子。")
    self._prev_group_btn.setMinimumHeight(23)
    self._prev_group_btn.setMaximumHeight(25)
    self._prev_group_btn.setFixedWidth(92)
    self._prev_group_btn.setStyleSheet("QPushButton{font-size:9px;padding:2px 6px;}")
    self._prev_group_btn.clicked.connect(self._jump_previous_group)
    result_header.addWidget(self._prev_group_btn)
    self._resolved_history_check = QCheckBox("显示已裁决")
    self._resolved_history_check.setChecked(False)
    self._resolved_history_check.setToolTip(
        "打开后只读浏览明确裁决过的句子；可按本地、人工、本地 AI、云端 AI 分别筛选。"
        "自动一致/两模型共同候选不计入裁决历史。"
    )
    self._resolved_history_check.setStyleSheet("font-size:10px;spacing:4px;font-weight:600;")
    self._resolved_history_check.toggled.connect(self._toggle_resolved_history)
    result_header.addWidget(self._resolved_history_check)
    self._resolved_history_filter = QComboBox()
    configure_combo(self._resolved_history_filter)
    self._resolved_history_filter.setMinimumContentsLength(0)
    self._resolved_history_filter.setFixedWidth(92)
    self._resolved_history_filter.setMinimumHeight(25)
    self._resolved_history_filter.setMaximumHeight(25)
    self._resolved_history_filter.setStyleSheet(
        "QComboBox{font-size:9px;padding:1px 4px;}"
        "QComboBox::drop-down{width:16px;}"
    )
    self._resolved_history_filter.setFixedHeight(25)
    self._resolved_history_filter.addItem("全部已裁决", "all")
    self._resolved_history_filter.addItem("本地裁决", "local")
    self._resolved_history_filter.addItem("人工裁决", "human")
    self._resolved_history_filter.addItem("本地 AI", "local_ai")
    self._resolved_history_filter.addItem("云端 AI", "cloud_ai")
    self._resolved_history_filter.setEnabled(False)
    self._resolved_history_filter.setVisible(False)
    self._resolved_history_filter.currentIndexChanged.connect(self._resolved_history_filter_changed)
    result_header.addWidget(self._resolved_history_filter)
    self._next_group_btn = QPushButton("下一分歧 →")
    self._next_group_btn.setToolTip("定位下一条仍需判断的句子。")
    self._next_group_btn.setMinimumHeight(23)
    self._next_group_btn.setMaximumHeight(25)
    self._next_group_btn.setFixedWidth(92)
    self._next_group_btn.setStyleSheet("QPushButton{font-size:9px;padding:2px 6px;}")
    self._next_group_btn.clicked.connect(self._jump_next_group)
    result_header.addWidget(self._next_group_btn)
    self._next_sentence_btn = QPushButton("下一句 →")
    self._next_sentence_btn.setToolTip(
        "前往 OCR 对比的下一句；图文对照会记住并同步到同一稳定行。"
    )
    self._next_sentence_btn.setEnabled(False)
    self._next_sentence_btn.setMinimumHeight(23)
    self._next_sentence_btn.setMaximumHeight(25)
    self._next_sentence_btn.setFixedWidth(92)
    self._next_sentence_btn.setStyleSheet(
        "QPushButton{font-size:10px;padding:2px 7px;}"
    )
    self._next_sentence_btn.clicked.connect(self._jump_next_sentence)
    result_header.addWidget(self._next_sentence_btn)
    # Compact Phase 20 result card: navigation/filter controls remain wired
    # and are available from the command menu / queue, but do not crowd the
    # candidate header in the default view.
    self._result_title.setText("候选结果")
    self._result_title.setVisible(False)
    for widget in (
        self._review_only_check, self._auto_advance_check, self._detail,
        self._virtual_hint, self._prev_group_btn, self._resolved_history_check,
        self._resolved_history_filter, self._next_group_btn, self._next_sentence_btn,
    ):
        widget.setVisible(False)
    result_layout.addLayout(result_header)

    self._fusion_scroll = QScrollArea()
    self._fusion_scroll.setWidgetResizable(True)
    self._fusion_scroll.setStyleSheet(
        "QScrollArea{border:1px solid #D7E3F4;border-radius:14px;background:#FFFFFF;}" + EDITOR_SCROLLBAR_STYLE
    )
    self._fusion_content = QWidget()
    self._fusion_content.setStyleSheet("background:#FFFFFF;")
    self._fusion_layout = QVBoxLayout(self._fusion_content)
    self._fusion_layout.setContentsMargins(8, 8, 8, 8)
    self._fusion_layout.setSpacing(7)
    self._fusion_layout.addStretch(1)
    self._fusion_scroll.setWidget(self._fusion_content)

    self._fusion_body_splitter = QSplitter(Qt.Horizontal)
    self._fusion_body_splitter.setChildrenCollapsible(False)
    self._fusion_body_splitter.setHandleWidth(4)
    self._decision_queue_panel = QFrame()
    self._decision_queue_panel.setObjectName("decisionQueuePanel")
    self._decision_queue_panel.setMinimumWidth(315)
    self._decision_queue_panel.setMaximumWidth(350)
    self._decision_queue_panel.setStyleSheet(
        "QFrame#decisionQueuePanel{background:#FFFFFF;border:1px solid #D7E3F4;border-radius:14px;}"
    )
    decision_queue_layout = QVBoxLayout(self._decision_queue_panel)
    decision_queue_layout.setContentsMargins(12, 12, 12, 12)
    decision_queue_layout.setSpacing(9)
    decision_queue_header = QHBoxLayout()
    self._decision_queue_title = QLabel("分歧队列")
    self._decision_queue_title.setStyleSheet("font-size:12px;font-weight:750;color:#14202E;")
    decision_queue_header.addWidget(self._decision_queue_title)
    self._decision_queue_count = QLabel("0")
    self._decision_queue_count.setVisible(False)
    self._active_review_check = QCheckBox("高风险优先")
    self._active_review_check.setChecked(True)
    self._active_review_check.setToolTip(
        "只改变人工复核顺序，不改变正文、OCR 候选或自动裁决。"
        "优先显示三方分歧、空/占位符、长度差异、数字/等级内容和高共识熵句。"
    )
    self._active_review_check.setStyleSheet("font-size:9px;spacing:3px;font-weight:600;")
    self._active_review_check.toggled.connect(self._toggle_active_review_queue)
    # Keep the high-risk filter as a backend control.  The reference view keeps
    # the queue header visually quiet; the same toggle is exposed in the queue
    # context menu below.
    self._active_review_check.setVisible(False)
    decision_queue_header.addStretch(1)
    decision_queue_layout.addLayout(decision_queue_header)

    # Review-console controls inspired by mature diff/annotation tools: keep
    # scope and search immediately above the virtual queue instead of hiding
    # them in an overflow menu.  These controls only filter/navigate existing
    # decision state; they never mutate OCR text or adjudication authority.
    queue_controls = QHBoxLayout()
    queue_controls.setContentsMargins(0, 0, 0, 0)
    queue_controls.setSpacing(6)
    self._decision_queue_scope = QComboBox()
    configure_combo(self._decision_queue_scope)
    self._decision_queue_scope.addItem("待判断", "pending")
    self._decision_queue_scope.addItem("高风险优先", "risk")
    self._decision_queue_scope.addItem("三方分歧", "threeway")
    self._decision_queue_scope.addItem("空/占位符", "broken")
    self._decision_queue_scope.addItem("已裁决", "history")
    self._decision_queue_scope.setFixedWidth(108)
    self._decision_queue_scope.setMinimumHeight(28)
    self._decision_queue_scope.setToolTip("切换待判断、高风险优先或已裁决历史；只改变浏览范围。")
    self._decision_queue_scope.currentIndexChanged.connect(self._decision_queue_scope_changed)
    queue_controls.addWidget(self._decision_queue_scope)
    self._decision_queue_search = QLineEdit()
    self._decision_queue_search.setClearButtonEnabled(True)
    self._decision_queue_search.setPlaceholderText("筛选页码 / 候选文字")
    self._decision_queue_search.setMinimumHeight(28)
    self._decision_queue_search.setStyleSheet(
        "QLineEdit{border:1px solid #D7E3F4;border-radius:7px;padding:3px 8px;background:#FAFCFF;}"
        "QLineEdit:focus{border-color:#4F7CFF;background:#FFFFFF;}"
    )
    self._decision_queue_search_timer = QTimer(self)
    self._decision_queue_search_timer.setSingleShot(True)
    self._decision_queue_search_timer.setInterval(120)
    self._decision_queue_search_timer.timeout.connect(self._decision_queue_search_changed)
    self._decision_queue_search.textChanged.connect(lambda _text: self._decision_queue_search_timer.start())
    queue_controls.addWidget(self._decision_queue_search, 1)
    # Reuse the existing history subgroup selector in the visible queue.
    queue_controls.addWidget(self._resolved_history_filter)
    decision_queue_layout.addLayout(queue_controls)

    self._decision_queue_stats = QLabel("待判 0 · 三方 0 · 空/占位 0 · 已裁决 0")
    self._decision_queue_stats.setStyleSheet("color:#667085;font-size:8.5px;font-weight:600;")
    self._decision_queue_stats.setWordWrap(True)
    decision_queue_layout.addWidget(self._decision_queue_stats)

    self._compare_sentence_strip = SentenceStrip(self._decision_queue_panel)
    self._compare_sentence_strip.setToolTip("整本裁决总览：橙=待判断，蓝=人工/AI裁决，绿=稳定；点击可跳转。")
    self._compare_sentence_strip.jump.connect(self._select_row)
    decision_queue_layout.addWidget(self._compare_sentence_strip)
    self._decision_queue_shortcut_hint = QLabel("F7 下一分歧 · Shift+F7 上一 · Alt+1…9 作底稿 · Ctrl+Alt+1…9 直接采用 · Alt+I 图文证据")
    self._decision_queue_shortcut_hint.setStyleSheet("color:#7B8797;font-size:8.5px;")
    self._decision_queue_shortcut_hint.setWordWrap(True)
    decision_queue_layout.addWidget(self._decision_queue_shortcut_hint)

    self._decision_queue = QListView()
    self._decision_queue.setUniformItemSizes(True)
    self._decision_queue.setSpacing(5)
    self._decision_queue.setSelectionMode(QAbstractItemView.SingleSelection)
    self._decision_queue.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    self._decision_queue.setStyleSheet(
        "QListView{background:#FFFFFF;border:none;font-size:10px;outline:0;}"
    )
    self._decision_queue_delegate = DecisionQueueDelegate(self._decision_queue)
    self._decision_queue.setItemDelegate(self._decision_queue_delegate)
    self._decision_queue_model = DecisionQueueListModel(
        self._decision_queue_display_text, self._decision_queue_tooltip, self
    )
    self._decision_queue.setModel(self._decision_queue_model)
    self._decision_queue.clicked.connect(self._decision_queue_item_clicked)
    self._decision_queue.activated.connect(self._decision_queue_item_clicked)
    decision_queue_layout.addWidget(self._decision_queue, 1)
    self._fusion_body_splitter.addWidget(self._decision_queue_panel)
    self._fusion_body_splitter.addWidget(self._fusion_scroll)
    self._fusion_body_splitter.setStretchFactor(0, 0)
    self._fusion_body_splitter.setStretchFactor(1, 1)
    self._fusion_body_splitter.setSizes([330, 790])
    result_layout.addWidget(self._fusion_body_splitter, 1)

    # Reference-style adjudication card.  It is a UI facade over the existing
    # adjudication/manual-edit/navigation controls, not a second decision path.
    self._proof_suggestion_card = QFrame()
    self._proof_suggestion_card.setObjectName("proofSuggestionCard")
    self._proof_suggestion_card.setStyleSheet(
        f"QFrame#proofSuggestionCard{{background:{CARD};border:1px solid {BORDER};border-radius:14px;}}"
    )
    suggestion_layout = QVBoxLayout(self._proof_suggestion_card)
    suggestion_layout.setContentsMargins(16, 14, 16, 14)
    suggestion_layout.setSpacing(9)
    suggestion_header = QHBoxLayout()
    suggestion_header.setContentsMargins(0, 0, 0, 0)
    suggestion_header.setSpacing(8)
    suggestion_title = QLabel("AI 裁决")
    suggestion_title.setStyleSheet("font-size:12px;font-weight:750;")
    suggestion_header.addWidget(suggestion_title)
    self._ai_provider_badge = QLabel("")
    self._ai_provider_badge.setStyleSheet(
        f"color:{MUTED};font-size:10px;background:#F5F8FC;border:1px solid {BORDER};"
        "border-radius:8px;padding:3px 8px;"
    )
    suggestion_header.addWidget(self._ai_provider_badge)
    suggestion_header.addStretch(1)
    ai_settings_btn = QPushButton("AI 服务…")
    ai_settings_btn.setFlat(True)
    ai_settings_btn.setStyleSheet(f"color:#2F6BFF;border:none;background:transparent;font-weight:650;")
    suggestion_header.addWidget(ai_settings_btn)
    suggestion_layout.addLayout(suggestion_header)

    def refresh_ai_provider_badge():
        try:
            from ai.config import load_ai_settings
            st = load_ai_settings()
            names = {
                "openai": "OpenAI", "anthropic": "Claude", "gemini": "Gemini",
                "deepseek": "DeepSeek", "zhipu": "GLM 国内", "zai": "GLM 国际",
                "openrouter": "OpenRouter", "ollama": "Ollama", "custom": "自定义",
            }
            provider = names.get(str(st.provider or "").lower(), str(st.provider or "未配置"))
            model = str(st.model or "未选模型")
            self._ai_provider_badge.setText(f"{provider} · {model}")
        except Exception:
            self._ai_provider_badge.setText("AI 服务未配置")

    def open_ai_settings():
        from ui.settings.ai_dialog import AISettingsDialog
        AISettingsDialog(self).exec()
        refresh_ai_provider_badge()

    ai_settings_btn.clicked.connect(open_ai_settings)
    self._refresh_ai_provider_badge = refresh_ai_provider_badge
    refresh_ai_provider_badge()
    manual_label = QLabel("手动编辑当前裁决文本")
    manual_label.setStyleSheet("font-size:11px;font-weight:700;")
    suggestion_layout.addWidget(manual_label)
    self._manual_decision_editor = MouseWheelPlainTextEdit()
    self._manual_decision_editor.setPlaceholderText("可直接输入最终文本；不会覆盖任何 OCR 模型原文。")
    self._manual_decision_editor.setMinimumHeight(68)
    self._manual_decision_editor.setMaximumHeight(168)
    self._manual_decision_editor.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    self._manual_decision_editor.setStyleSheet(
        f"QPlainTextEdit{{background:{CARD};border:1px solid {BORDER};border-radius:9px;"
        f"padding:8px;color:{INK};}}" + EDITOR_SCROLLBAR_STYLE
    )
    suggestion_layout.addWidget(self._manual_decision_editor)
    manual_actions = QHBoxLayout()
    manual_actions.setContentsMargins(0, 0, 0, 0)
    manual_actions.setSpacing(7)
    self._manual_decision_state = QLabel("草稿区 · 选择“作为底稿”后修改，未确认前不会进入裁决结果")
    self._manual_decision_state.setStyleSheet("color:#7B8797;font-size:8.5px;")
    self._manual_decision_state.setWordWrap(True)
    manual_actions.addWidget(self._manual_decision_state, 1)
    self._manual_decision_discard_btn = QPushButton("放弃草稿")
    self._manual_decision_discard_btn.setFixedHeight(28)
    self._manual_decision_discard_btn.clicked.connect(self._discard_manual_decision_draft)
    manual_actions.addWidget(self._manual_decision_discard_btn)
    self._manual_decision_confirm_btn = accent_button("确认裁决")
    self._manual_decision_confirm_btn.setFixedHeight(28)
    self._manual_decision_confirm_btn.setToolTip("确认当前手动文本；停留在本句")
    self._manual_decision_confirm_btn.clicked.connect(self._commit_manual_decision_editor)
    manual_actions.addWidget(self._manual_decision_confirm_btn)
    self._manual_decision_confirm_next_btn = QPushButton("确认并下一分歧")
    self._manual_decision_confirm_next_btn.setFixedHeight(28)
    self._manual_decision_confirm_next_btn.setToolTip("Ctrl+Shift+Enter：确认后进入下一分歧")
    self._manual_decision_confirm_next_btn.clicked.connect(self._commit_manual_decision_and_next)
    manual_actions.addWidget(self._manual_decision_confirm_next_btn)
    suggestion_layout.addLayout(manual_actions)

    self._manual_decision_syncing = False
    self._manual_decision_drafts = {}
    self._manual_decision_draft_sources = {}
    # Kept as a compatibility object for older restored UI state, but manual
    # decisions are no longer auto-committed after a typing delay.
    self._manual_decision_commit_timer = QTimer(self)
    self._manual_decision_commit_timer.setSingleShot(True)
    self._manual_decision_commit_timer.setInterval(350)
    self._manual_decision_fit_timer = QTimer(self)
    self._manual_decision_fit_timer.setSingleShot(True)
    self._manual_decision_fit_timer.timeout.connect(self._fit_manual_decision_editor_height)
    self._manual_decision_editor.textChanged.connect(self._manual_decision_text_changed)
    self._manual_decision_editor.textChanged.connect(self._schedule_manual_decision_editor_fit)

    # Queue stays full-height on the left; the right column mirrors the master:
    # candidates above, adjudication suggestion below.  Reparenting the
    # existing fusion scroll preserves all virtual-row state and signals.
    proof_right = QWidget()
    proof_right.setStyleSheet("background:transparent;border:none;")
    proof_right_layout = QHBoxLayout(proof_right)
    proof_right_layout.setContentsMargins(0, 0, 0, 0)
    proof_right_layout.setSpacing(0)
    self._proof_evidence_splitter = QSplitter(Qt.Horizontal, proof_right)
    self._proof_evidence_splitter.setChildrenCollapsible(False)
    self._proof_evidence_splitter.setHandleWidth(5)
    self._proof_evidence_splitter.setStyleSheet(
        "QSplitter::handle{background:#E3ECF7;border-radius:2px;margin:28px 1px;}"
    )
    self._proof_reference_host = QFrame(proof_right)
    self._proof_reference_host.setObjectName("ocrReferenceEvidence")
    self._proof_reference_host.setMinimumWidth(300)
    self._proof_reference_host.setStyleSheet(
        "QFrame#ocrReferenceEvidence{background:#FFFFFF;border:1px solid #D7E3F4;border-radius:12px;}"
    )
    self._proof_reference_layout = QVBoxLayout(self._proof_reference_host)
    self._proof_reference_layout.setContentsMargins(10, 9, 10, 10)
    self._proof_reference_layout.setSpacing(7)
    reference_header = QHBoxLayout()
    reference_title = QLabel("原图证据")
    reference_title.setStyleSheet("font-size:11px;font-weight:750;color:#14202E;border:none;")
    reference_header.addWidget(reference_title)
    reference_header.addStretch(1)
    open_review = QPushButton("图文对照 ↗")
    open_review.setFixedHeight(25)
    open_review.setToolTip("在图文对照中打开同一稳定句（Alt+I）")
    open_review.clicked.connect(self._request_image_review)
    # Compatibility/backend identity for the Phase 20 compact command bar.
    # The reference-card button remains the authoritative action; compact
    # proxies mirror its enabled state instead of inventing a second state.
    self._image_review_mode_btn = open_review
    reference_header.addWidget(open_review)
    self._proof_reference_layout.addLayout(reference_header)
    reference_hint = QLabel("只读像素证据 · 不随候选修改")
    reference_hint.setStyleSheet("color:#7B8797;font-size:8.5px;border:none;")
    self._proof_reference_layout.addWidget(reference_hint)
    self._proof_evidence_splitter.addWidget(self._proof_reference_host)
    self._mounted_reference_row = None
    output_column = QWidget(proof_right)
    output_layout = QVBoxLayout(output_column)
    output_layout.setContentsMargins(0, 0, 0, 0)
    output_layout.setSpacing(8)
    self._fusion_body_splitter.replaceWidget(1, proof_right)
    candidate_nav = QHBoxLayout()
    candidate_nav.setContentsMargins(4, 0, 4, 0)
    candidate_nav.setSpacing(6)
    candidate_meta = QLabel("候选对比 · 红底=替换 · 橙底=增删/缺失")
    candidate_meta.setStyleSheet("color:#667085;font-size:9px;font-weight:600;")
    candidate_nav.addWidget(candidate_meta)
    candidate_nav.addStretch(1)
    prev_diff = QPushButton("← 上一分歧")
    prev_diff.setFixedHeight(25)
    prev_diff.clicked.connect(self._jump_previous_group)
    candidate_nav.addWidget(prev_diff)
    next_diff = QPushButton("下一分歧 →")
    next_diff.setFixedHeight(25)
    next_diff.clicked.connect(self._jump_next_group)
    candidate_nav.addWidget(next_diff)
    output_layout.addLayout(candidate_nav)
    output_layout.addWidget(self._fusion_scroll, 1)
    self._fusion_scroll.setMinimumHeight(300)
    self._fusion_scroll.setMaximumHeight(16777215)
    output_layout.addWidget(self._proof_suggestion_card)
    self._proof_evidence_splitter.addWidget(output_column)
    self._proof_evidence_splitter.setStretchFactor(0, 2)
    self._proof_evidence_splitter.setStretchFactor(1, 3)
    self._proof_evidence_splitter.setSizes([380, 720])
    self._proof_evidence_splitter.splitterMoved.connect(
        lambda _pos, _index: self._schedule_manual_decision_editor_fit()
    )
    proof_right_layout.addWidget(self._proof_evidence_splitter, 1)
    bind_splitter(self._proof_evidence_splitter, "compare_reference_evidence_v2")
    self._proof_right_column = proof_right

    def sync_proof_mode_geometry(checked: bool):
        full = bool(checked)
        self._proof_suggestion_card.setVisible(True)
        self._fusion_scroll.setMaximumHeight(16777215)

    self._full_text_compare_check.toggled.connect(sync_proof_mode_geometry)

    self._result_hint = QLabel(
        "逐句裁决：左侧队列定位未决句，上方已载入模型只显示当前句；绿色=一致，红色=差异。"
        "候选卡始终完整展开，红底=替换、橙底=增删；融合候选可直接修改，“显示全文对比”只切换显示。"
    )
    self._result_hint.setWordWrap(False)
    self._result_hint.setMaximumHeight(16)
    self._result_hint.setStyleSheet("color:#5B6B80; font-size:9px;")
    self._result_hint.setVisible(False)
    result_layout.addWidget(self._result_hint)
    # Phase 20 compact command bar.  The historical dense header remains
    # instantiated as an action backend (and for restored-session contracts),
    # but is hidden from the default workspace.  Every former action is exposed
    # through the compact buttons or the More menu, so functionality is not
    # removed—only the visual hierarchy changes.
    compact_header = QFrame()
    compact_header.setObjectName("ocrCompareCompactHeader")
    compact_header.setStyleSheet(
        f"QFrame#ocrCompareCompactHeader{{background:{CARD};border:1px solid {BORDER};border-radius:12px;}}"
    )
    compact_layout = QVBoxLayout(compact_header)
    compact_layout.setContentsMargins(12, 8, 12, 8)
    compact_layout.setSpacing(4)
    compact_row = QHBoxLayout()
    compact_row.setContentsMargins(0, 0, 0, 0)
    compact_row.setSpacing(8)
    compact_layout.addLayout(compact_row)
    compact_title = QLabel("全文总览")
    compact_title.setStyleSheet("font-size:12px;font-weight:750;")
    compact_row.addWidget(compact_title)
    self._compact_title_label = compact_title
    # Opt-in reading layout: original source editors and disagreement widgets
    # remain alive, with their state unchanged, while the book text fills the UI.
    self._full_only_check = QPushButton("多模型全文对照")
    self._full_only_check.setObjectName("ocrCompareFullOnlyToggle")
    self._full_only_check.setCheckable(True)
    self._full_only_check.setToolTip(
        "开启后四栏同步显示所有 OCR 模型的完整对齐正文和融合结果，"
        "红色标记不同字符，淡红底标记分歧行。关闭后恢复逐句候选面板。"
    )
    self._full_only_check.setStyleSheet(
        "QPushButton{font-size:10px;padding:5px 12px;border:1px solid #D7E3F4;"
        "border-radius:7px;background:#FFFFFF;}"
        "QPushButton:checked{background:#E4EEFF;color:#2559E0;"
        "border-color:#2F6BFF;font-weight:700;}"
    )
    self._full_only_check.toggled.connect(self._set_full_only_view)
    compact_row.addWidget(self._full_only_check)
    self._compact_compare_state = QLabel("等待 OCR 结果")
    self._compact_compare_state.setStyleSheet(f"color:{MUTED};font-size:10px;")
    compact_row.addWidget(self._compact_compare_state, 1)

    compact_full = QPushButton("显示全文")
    compact_full.setCheckable(False)
    compact_full.setEnabled(False)
    compact_full.setToolTip("当前为全文总览；点击可从任何兼容恢复状态回到全文显示")
    compact_full.clicked.connect(self._show_multimodel_full)
    compact_row.addWidget(compact_full)
    self._compact_full_compare_btn = compact_full

    compact_image_review = QPushButton("图文对照")
    compact_image_review.setToolTip("在当前工作台切换到逐句图文对照，保留稳定句和草稿")
    compact_image_review.clicked.connect(self._request_image_review)
    compact_row.addWidget(compact_image_review)
    self._compact_image_review_btn = compact_image_review

    compact_ai = accent_button("AI 裁决", color="#6D28D9")
    compact_ai.clicked.connect(self._ai_adjudicate_btn.click)
    compact_row.addWidget(compact_ai)
    self._compact_ai_btn = compact_ai

    # These three package actions are common parts of the OCR comparison loop;
    # surface their existing backend buttons directly in the compact toolbar
    # so enablement and progress state stay authoritative.
    self._export_source_correction_btn.setText("导出 AI 裁决包")
    compact_row.addWidget(self._export_source_correction_btn)
    self._import_source_correction_btn.setText("导入 AI 裁决包")
    compact_row.addWidget(self._import_source_correction_btn)
    self._export_fusion_skeleton_btn.setText("融合结果＋骨架 EPUB")
    compact_row.addWidget(self._export_fusion_skeleton_btn)

    compact_apply_all = accent_button("应用整本")
    compact_apply_all.setToolTip("将当前全文裁决结果应用到后续工作流")
    compact_apply_all.clicked.connect(self._apply_btn.click)
    compact_row.addWidget(compact_apply_all)
    self._compact_apply_all_btn = compact_apply_all

    # Unified visible progress for long-running compare/exchange tasks.  The
    # historical progress bars live in the hidden compatibility panel, so they
    # cannot provide feedback in the Phase 20/37 compact workspace.  Keep those
    # backend bars for compatibility, but mirror every active AI/export/import
    # task here where the user can actually see it.
    self._compact_task_progress = QProgressBar()
    self._compact_task_progress.setRange(0, 100)
    self._compact_task_progress.setValue(0)
    self._compact_task_progress.setTextVisible(True)
    self._compact_task_progress.setMaximumHeight(16)
    self._compact_task_progress.setFormat("等待 OCR 结果")
    self._compact_task_progress.setVisible(False)

    more = QToolButton()
    more.setText("更多操作 ▾")
    more.setPopupMode(QToolButton.InstantPopup)
    more_menu = QMenu(more)
    action_specs = [
        ("从 OCR 识别载入", self._load_single_from_ocr_btn),
        ("返回多模型对比", self._return_multi_btn),
        ("自动选优", self._auto_btn),
        ("撤销裁决", self._adjudication_undo_btn),
        ("重做裁决", self._adjudication_redo_btn),
        ("撤销自动选优", self._auto_restore_batch_btn),
        ("重新对齐", self._realign_btn),
        ("恢复初始", self._restore_btn),
        ("无损标准化", self._unicode_normalize_btn),
        ("上一分歧", self._prev_group_btn),
        ("下一分歧", self._next_group_btn),
        ("下一句", self._next_sentence_btn),
        ("导出单 OCR 校对包", self._single_export_btn),
        ("导入单 OCR 校对结果", self._single_import_btn),
        ("导出 OCR 文本", self._export_texts_btn),
        ("应用融合稿", self._apply_btn),
        ("导出融合 JSON", self._export_ai_package_btn),
        ("导出 AI OCR 裁决包", self._export_source_correction_btn),
        ("导入 AI OCR 裁决", self._import_source_correction_btn),
        ("恢复纠错会话", self._restore_source_session_btn),
        ("融合结果＋骨架 EPUB", self._export_fusion_skeleton_btn),
        ("导出 AI 修复包", self._export_ai_repair_epub_btn),
        ("导入 AI 融合包", self._import_ai_package_btn),
    ]
    self._compact_compare_actions = []
    for text, button in action_specs:
        action = more_menu.addAction(text)
        action.triggered.connect(button.click)
        self._compact_compare_actions.append((action, button))
    # High-frequency Phase 20 proxy controls are also available here so the
    # visible OCR-result command bar can disappear in the pixel-perfect view
    # without hiding functionality.
    more_menu.addSeparator()
    show_full_action = more_menu.addAction("显示全文")
    show_full_action.setCheckable(False)
    show_full_action.triggered.connect(lambda _checked=False: self._show_multimodel_full())
    ai_action = more_menu.addAction("AI 裁决")
    ai_action.triggered.connect(compact_ai.click)
    ai_settings_action = more_menu.addAction("AI 服务设置…")
    ai_settings_action.triggered.connect(open_ai_settings)
    more_menu.addSeparator()
    high_risk_action = more_menu.addAction("高风险优先")
    high_risk_action.setCheckable(True)
    high_risk_action.setChecked(self._active_review_check.isChecked())
    high_risk_action.triggered.connect(lambda checked: self._active_review_check.setChecked(bool(checked)))
    show_history = more_menu.addAction("显示已裁决")
    show_history.setCheckable(True)
    show_history.triggered.connect(lambda checked: self._resolved_history_check.setChecked(bool(checked)))
    self._compact_history_action = show_history

    def sync_compact_actions():
        for action, button in self._compact_compare_actions:
            action.setEnabled(button.isEnabled())
            # Backend buttons live inside a deliberately hidden compatibility
            # panel. QWidget.isVisible() therefore becomes False merely because
            # an ancestor is hidden, which used to make the proxy menu hide the
            # very actions it is supposed to expose. Respect only an explicit
            # hide on the backend control; a hidden parent must not remove the
            # action from the visible compact menu.
            action.setVisible((not button.isHidden()) or button in (
                self._single_export_btn, self._single_import_btn, self._import_ai_package_btn,
            ))
        high_risk_action.setChecked(self._active_review_check.isChecked())
        self._compact_history_action.setChecked(self._resolved_history_check.isChecked())
        self._sync_compact_compare_controls()
        show_full_action.setChecked(False)
        show_full_action.setText("显示全文")
        show_full_action.setVisible(self._review_mode == "decision")
        show_full_action.setEnabled(self._compact_full_compare_btn.isEnabled())
        ai_action.setEnabled(self._compact_ai_btn.isEnabled())

    more_menu.aboutToShow.connect(sync_compact_actions)
    more.setMenu(more_menu)
    compact_row.addWidget(more)
    self._compact_compare_more_btn = more
    compact_layout.addWidget(self._compact_task_progress)
    # The supplied proofing master goes straight from the sub-tab row into the
    # queue/candidate workspace.  Keep all proxy buttons alive as hidden
    # backends, and surface only the unobtrusive overflow menu beside
    # “候选结果”.
    compact_header.setVisible(True)   # 统一工具栏，按钮随全文/逐句模式互斥显示
    self._compact_compare_header = compact_header
    # Functionality is preserved without a persistent visual button: right-click
    # anywhere in the queue opens the exact same overflow menu.
    more.setVisible(True)
    self._decision_queue.setContextMenuPolicy(Qt.CustomContextMenu)
    self._decision_queue.customContextMenuRequested.connect(
        lambda pos: more_menu.exec(self._decision_queue.viewport().mapToGlobal(pos))
    )
    self._decision_queue.setToolTip("右键可打开更多操作")
    root.addWidget(compact_header)
    # Keyboard shortcuts remain active but the persistent instruction strip is
    # removed to return that vertical space to the decision workspace.
    self._keyboard_hint = None
    self.install_keyboard_flow()

    main_splitter.addWidget(self._result_panel)
    main_splitter.setStretchFactor(0, 5)
    main_splitter.setStretchFactor(1, 4)
    main_splitter.setSizes([460, 360])
    root.addWidget(main_splitter, 1)
    # A separate read-only text surface avoids creating thousands of decision
    # cards just to read a complete book. Nothing here is export authority.
    self._full_overview_panel = QFrame()
    self._full_overview_panel.setObjectName("ocrCompareFullOnlyPanel")
    self._full_overview_panel.setStyleSheet(
        "QFrame#ocrCompareFullOnlyPanel{background:#FFFFFF;"
        "border:1px solid #D7E3F4;border-radius:12px;}"
    )
    only_layout = QVBoxLayout(self._full_overview_panel)
    only_layout.setContentsMargins(18, 12, 18, 12)
    only_layout.setSpacing(8)
    self._full_overview_note = QLabel("多模型全文对照 · 逐行对应 · 红色=字符差异，浅红=模型分歧")
    self._full_overview_note.setStyleSheet("color:#44546B;font-size:11px;font-weight:600;")
    only_layout.addWidget(self._full_overview_note)
    # One shared scroll surface keeps 4,058+ sentence rows aligned without
    # independent editor offsets or the old one-sentence-only source widgets.
    from ui.ocr.full_book_comparison import FullBookComparisonTable
    self._full_comparison_table = FullBookComparisonTable(self._full_overview_panel)
    self._full_comparison_table.doubleClicked.connect(self._open_full_comparison_row)
    only_layout.addWidget(self._full_comparison_table, 1)
    self._full_overview_panel.setVisible(False)
    root.addWidget(self._full_overview_panel, 1)
    bind_splitter(main_splitter, "compare_main")
    QTimer.singleShot(0, lambda: self._set_review_mode("full", force=True))
